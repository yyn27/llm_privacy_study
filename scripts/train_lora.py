#!/usr/bin/env python3
"""M3 sequential one-model LoRA training with resumable checkpoints and online Canary audit."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.lora_config import lora_config
from scripts.utils import (seed_all, read_jsonl, append_jsonl, encode_qa,
                           tensor_batch, batch_order, save_training_state,
                           restore_training_state, sha256, git_hash)
from scripts.audit import audit_canaries, validation_loss


def train(args):
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tc, ac = config["train"], config["audit"]
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.max_steps is not None:
        tc["max_steps"] = args.max_steps
    if args.audit_interval is not None:
        tc["audit_interval"] = args.audit_interval
    if args.reference_limit is not None:
        ac["reference_limit"] = args.reference_limit
    if args.val_limit is not None:
        ac["val_limit"] = args.val_limit
    model_id = yaml.safe_load((ROOT/"configs/models.yaml").read_text(
        encoding="utf-8"))["models"][args.model]["id"]
    seed = tc["seed"]
    seed_all(seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("M3 requires CUDA with BF16")
    if tc["per_device_batch_size"] != 1 or tc["optimizer"] != "adamw_torch":
        raise ValueError("This implementation supports microbatch=1, adamw_torch only")
    if tc["scheduler"] != "constant_with_warmup":
        raise ValueError("Nonconstant schedule invalidates stop-policy trajectory replay")
    if tc["bf16"] is not True:
        raise ValueError("M3 protocol requires BF16")
    train_path = ROOT / "canaries" / args.model / "train_with_canaries.jsonl"
    meta_path = ROOT / "canaries" / args.model / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["m1_train_sha256"] != sha256(ROOT/"data/train.jsonl"):
        raise ValueError("M1 training data mutated after M2")
    if meta["m1_manifest_sha256"] != sha256(ROOT/"data/manifest.json"):
        raise ValueError("M1 manifest mutated after M2")
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(ROOT/"data/val.jsonl")
    injected = read_jsonl(ROOT/"canaries"/args.model/"injected.jsonl")
    nulls = read_jsonl(ROOT/"canaries"/args.model/"nullgate.jsonl")
    references = read_jsonl(ROOT/"canaries"/args.model/"reference.jsonl")
    if len(train_rows) != meta["final_train_samples"]:
        raise ValueError("Training dataset size mismatch")
    val_rows = val_rows[:ac["val_limit"]] if ac["val_limit"] else val_rows
    ref_limit = ac["reference_limit"]
    if not 2 <= ref_limit <= meta["reference_per_type"]:
        raise ValueError("Reference limit exceeds available candidates")
    out = ROOT / "runs" / args.model / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out/"config.json"
    cfg = {
        "model": args.model, "model_id": model_id, "seed": seed,
        "role": "calibration" if seed == 42 else "evaluation",
        "train": tc, "audit": ac, "lora": lora_config.to_dict(),
        "git_commit": git_hash(ROOT), "train_data_sha256": sha256(train_path),
        "m2_meta_sha256": sha256(meta_path),
        "freeze_sha256": sha256(ROOT/"configs/freeze.md"),
        "status": "SMOKE_TEST" if ref_limit < 1000 or tc["max_steps"] <= 100
                  else "UNFROZEN_FORMAL_CANDIDATE",
    }
    if cfg_path.exists():
        previous = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("model", "model_id", "seed", "train", "audit",
                    "train_data_sha256", "m2_meta_sha256", "freeze_sha256"):
            if previous[key] != cfg[key]:
                raise ValueError(f"Existing run config mismatch on {key}; "
                                 "do not reuse a run directory with changed configuration")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2, default=str),
                            encoding="utf-8")
    print(f"Loading {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=args.local_files_only,
                                         trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, local_files_only=args.local_files_only,
        trust_remote_code=False, low_cpu_mem_usage=True,
        attn_implementation="eager")
    device = torch.device("cuda:0")
    model.to(device)
    model.config.use_cache = False
    if args.resume:
        checkpoints = sorted((out/"adapters").glob("step_*"))
        if not checkpoints:
            raise ValueError("No checkpoints available for --resume")
        ckpt = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))
        model = PeftModel.from_pretrained(model, ckpt/"adapter", is_trainable=True)
    else:
        if (out/"adapters").exists() or (out/"metrics.jsonl").exists():
            raise FileExistsError("Existing checkpoints/logs: specify --resume; "
                                  "never overwrite a run implicitly")
        model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tc["learning_rate"])
    step, microstep, last_loss = 0, 0, None
    if args.resume:
        saved = restore_training_state(ckpt, optimizer)
        step, microstep, last_loss = (saved["global_step"],
                                      saved["microstep"], saved["train_loss"])
        for group in optimizer.state.values():
            for k, v in group.items():
                if torch.is_tensor(v):
                    group[k] = v.to(device)
        print(f"Resumed step={step}, microstep={microstep}", flush=True)
    encoded = [encode_qa(tok, x["prompt"], x["answer"], tc["max_length"])
               for x in train_rows]
    accum = tc["gradient_accumulation_steps"]
    if accum < 1 or tc["audit_interval"] < 1:
        raise ValueError("Invalid gradient accumulation / audit interval")
    epoch_size = len(encoded)
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while step < tc["max_steps"]:
        model.train()
        current_epoch, offset = divmod(microstep, epoch_size)
        chosen = batch_order(epoch_size, seed, current_epoch)[offset]
        batch = tensor_batch([encoded[chosen]], tok.pad_token_id, device)
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at microstep={microstep}")
        (loss / accum).backward()
        last_loss = float(loss.detach().cpu())
        microstep += 1
        if microstep % accum != 0:
            continue
        step += 1
        warmup = tc["warmup_steps"]
        lr = tc["learning_rate"] * (min(step / warmup, 1.0) if warmup else 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % tc["audit_interval"] == 0 or step == tc["max_steps"]:
            folder = out/"adapters"/f"step_{step:06d}"
            save_training_state(folder, model, optimizer, step, microstep, last_loss, seed)
            metrics_file = out/"metrics.jsonl"
            recorded = {x["step"] for x in read_jsonl(metrics_file)} if metrics_file.exists() else set()
            if step not in recorded:
                print(f"Audit checkpoint {step} ({args.model})", flush=True)
                val = validation_loss(model, tok, val_rows, tc["max_length"], device)
                audit = audit_canaries(
                    model, tok, injected, nulls, references, tc["max_length"], device,
                    ref_limit=ref_limit, batch_size=ac["scoring_batch_size"],
                    max_new_tokens=ac["max_new_tokens"])
                record = {"model": args.model, "seed": seed,
                          "step": step, "epoch": microstep/epoch_size,
                          "train_loss": last_loss, "lr": lr,
                          "val_loss": val["mixed_loss"],
                          "val_utility_loss": val["utility_loss"],
                          "val_probe_loss": val["probe_loss"],
                          **audit, "time_s": {"train_elapsed": time.perf_counter()-start_time},
                          "checkpoint": str(folder.relative_to(ROOT))}
                append_jsonl(metrics_file, record)
                print(f"Step {step}: train={last_loss:.4f}, "
                      f"val={val['mixed_loss']:.4f}, "
                      f"audit_s={audit['audit_seconds']:.1f}", flush=True)
        elif step % 10 == 0:
            print(f"step={step} loss={last_loss:.4f}", flush=True)
    print(f"M3 finished: model={args.model} seed={seed} step={step} run={out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["gemma", "llama", "qwen"], required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--config", default="configs/train_args.yaml")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--audit-interval", type=int)
    p.add_argument("--reference-limit", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
