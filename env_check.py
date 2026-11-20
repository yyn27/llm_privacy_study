#!/usr/bin/env python3
"""M0: SmartVDI hardware/environment audit, model sizing, and BF16 LoRA smoke test.

Use synthetic, non-medical QA only. This script does NOT generate research findings.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback

import torch
import yaml
from huggingface_hub import model_info
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from configs.lora_config import TARGET_MODULES, lora_config

TEST_PAIRS = [
    ("Complete the sentence: The color of the clear daytime sky is", " blue."),
    ("Complete the sentence: One plus one equals", " two."),
    ("Complete the sentence: A week has", " seven days."),
    ("Complete the sentence: The opposite of cold is", " hot."),
    ("Complete the sentence: The first month of the year is", " January."),
    ("Complete the sentence: A triangle has", " three sides."),
    ("Complete the sentence: Water freezes into", " ice."),
    ("Complete the sentence: The opposite of left is", " right."),
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # effective only for child processes
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_text(args: list[str], timeout: int = 90) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return f"$ {' '.join(args)}\nexit={proc.returncode}\n{proc.stdout}{proc.stderr}\n"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"$ {' '.join(args)}\nERROR: {exc}\n"


def environment_report(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                      if d.metadata.get("Name"))
    cuda = torch.cuda.is_available()
    props = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            props.append({"index": i, "name": p.name, "total_memory_gib":
                          round(p.total_memory / 2**30, 3),
                          "bf16_supported": torch.cuda.is_bf16_supported()})
    usage = shutil.disk_usage(out)
    info = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": cuda, "devices": props,
        "disk_total_gib": round(usage.total / 2**30, 2),
        "disk_free_gib": round(usage.free / 2**30, 2),
        "cwd": str(Path.cwd()), "output_dir": str(out.resolve()),
    }
    lines = ["# M0 environment report (no tokens or secrets are saved)",
             json.dumps(info, ensure_ascii=False, indent=2),
             run_text(["nvidia-smi", "-L"]),
             run_text(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                       "--format=csv,noheader"]),
             "# pip freeze (installed packages)", *packages]
    (out / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "env.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    return info


def unique_parameter_counts(model) -> tuple[int, int]:
    """Count unique tensor objects; exclude input/output embedding tensors by identity."""
    excluded = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            excluded.update(id(p) for p in module.parameters())
    seen = set()
    total = non_embedding = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        total += n
        if id(p) not in excluded:
            non_embedding += n
    return total, non_embedding


def resolve_revision(model_id: str, local_files_only: bool) -> str:
    if local_files_only:
        return "offline; see local model config _commit_hash"
    try:
        return model_info(model_id).sha or "unknown"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def build_batch(tokenizer, step: int, max_length: int, device):
    prompt, answer = TEST_PAIRS[step % len(TEST_PAIRS)]
    # Shared plain text QA template (not model-specific chat templates).
    prefix = f"Question: {prompt}\nAnswer:"
    target = answer + (tokenizer.eos_token or "")
    left = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    right = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not right:
        raise RuntimeError("Answer tokenization produced no supervised tokens")
    if len(left) + len(right) > max_length:
        raise ValueError("max_length is too short for the M0 sample")
    ids = left + right
    labels = [-100] * len(left) + right
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    return {"input_ids": x, "attention_mask": mask, "labels": y}


def run_one(key: str, entry: dict, args) -> dict:
    model_id = entry["id"]
    row = {"model": key, "model_id": model_id, "status": "started",
           "base_total_parameters": "", "base_non_embedding_parameters": "",
           "trainable_parameters": "", "lora_trainable_pct_total": "",
           "lora_trainable_pct_non_embedding": "", "matched_target_modules": "",
           "model_revision": "", "smoke_steps": 0, "last_loss": "",
           "elapsed_seconds": "", "peak_gpu_memory_gib": "", "error": ""}
    started = time.perf_counter()
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id,
            trust_remote_code=False, local_files_only=args.local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=False,
            local_files_only=args.local_files_only,
            low_cpu_mem_usage=True, attn_implementation="eager")
        base_total, base_non_emb = unique_parameter_counts(model)
        row["base_total_parameters"] = base_total
        row["base_non_embedding_parameters"] = base_non_emb
        present = sorted({s for n, _ in model.named_modules()
                          for s in TARGET_MODULES if n.endswith("." + s)})
        missing = sorted(set(TARGET_MODULES) - set(present))
        if missing:
            raise RuntimeError(f"LoRA targets not found: {missing}")
        row["matched_target_modules"] = ",".join(present)
        row["model_revision"] = (getattr(model.config, "_commit_hash", None)
                                 or resolve_revision(model_id, args.local_files_only))
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row["trainable_parameters"] = trainable
        row["lora_trainable_pct_total"] = round(100 * trainable / base_total, 6)
        row["lora_trainable_pct_non_embedding"] = round(
            100 * trainable / base_non_emb, 6) if base_non_emb else ""
        if args.smoke_steps:
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 GPU training requested but CUDA/BF16 unavailable")
            model.to("cuda:0")
            model.train()
            torch.cuda.reset_peak_memory_stats(0)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=args.lr)
            loss_history = []
            for step in range(args.smoke_steps):
                batch = build_batch(tokenizer, step, args.max_length,
                                    torch.device("cuda:0"))
                optimizer.zero_grad(set_to_none=True)
                output = model(**batch)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at step {step + 1}")
                loss.backward()
                optimizer.step()
                value = float(loss.detach().cpu())
                loss_history.append(value)
                row["smoke_steps"] = step + 1
                print(f"[{key}] smoke step {step + 1}/{args.smoke_steps} loss={value:.4f}",
                      flush=True)
            row["last_loss"] = loss_history[-1]
            row["peak_gpu_memory_gib"] = round(
                torch.cuda.max_memory_allocated(0) / 2**30, 3)
            row["loss_history"] = loss_history
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        row["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="gemma,llama,qwen",
                        help="Comma-separated names from configs/models.yaml")
    parser.add_argument("--smoke-steps", type=int, default=10,
                        help="Number of BF16 optimizer steps per model (0 = inspect only)")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/m0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_steps < 0 or args.max_length < 32:
        parser.error("--smoke-steps must be >= 0 and --max-length >= 32")
    seed_everything(args.seed)
    out = (ROOT / args.output_dir).resolve()
    env = environment_report(out)
    if args.smoke_steps and (not env["cuda_available"] or
                              not torch.cuda.is_bf16_supported()):
        print("ERROR: requested BF16 smoke test requires CUDA BF16; "
              "use --smoke-steps 0 for inventory only.", file=sys.stderr)
        return 2
    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text(
        encoding="utf-8"))["models"]
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = set(keys) - set(registry)
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    rows = []
    for key in keys:
        seed_everything(args.seed)
        print(f"=== {key}: {registry[key]['id']} ===", flush=True)
        row = run_one(key, registry[key], args)
        rows.append(row)
        # Persist partial results immediately in case of VDI session interruption.
        (out / "models_table.json").write_text(json.dumps(
            rows, indent=2, ensure_ascii=False), encoding="utf-8")
        headers = ["model", "model_id", "status", "base_total_parameters",
                   "base_non_embedding_parameters", "trainable_parameters",
                   "lora_trainable_pct_total", "lora_trainable_pct_non_embedding",
                   "matched_target_modules", "model_revision", "smoke_steps",
                   "last_loss", "elapsed_seconds", "peak_gpu_memory_gib", "error"]
        with (out / "models_table.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{key}] status={row['status']} {row['error']}", flush=True)
    return 1 if any(r["status"] != "ok" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
