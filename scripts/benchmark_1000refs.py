#!/usr/bin/env python3
"""Benchmark 1000-reference Canary auditing on saved M3 adapters; NO training.

Old smoke100 adapters are trained on M2-v1. With --canary-version v2,
results are TIMING ONLY, never valid exposure outcomes.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from utils import read_jsonl, sha256
from audit import audit_canaries


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = torch.device("cuda:0")
    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text(
        encoding="utf-8"))["models"]
    model_id = registry[args.model]["id"]
    ckpt = ROOT / "runs" / args.model / str(args.seed) / args.run_tag / (
        f"adapters/step_{args.step:06d}/adapter")
    if not ckpt.is_dir():
        raise FileNotFoundError(ckpt)
    data = ROOT / ("canaries_v2" if args.canary_version == "v2" else "canaries") / args.model
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    expected_version = "m2_canaries_v2" if args.canary_version == "v2" else "m2_canaries_v1"
    if meta["m2_version"] != expected_version:
        raise ValueError("Canary dataset version mismatch")
    injected = read_jsonl(data / "injected.jsonl")
    nulls = read_jsonl(data / "nullgate.jsonl")
    refs = read_jsonl(data / "reference.jsonl")
    if len(refs) != 3000:
        raise ValueError("Require exactly 3000 reference candidates")
    if args.targets_per_category:
        def subset(items):
            selected = []
            for kind in ("id", "phone", "name_dx"):
                matches = [x for x in items if x["category"] == kind]
                if len(matches) < args.targets_per_category:
                    raise ValueError("Insufficient target canaries")
                selected.extend(matches[:args.targets_per_category])
            return selected
        injected, nulls = subset(injected), subset(nulls)
    else:
        if len(injected) != 27 or len(nulls) != 27:
            raise ValueError("Expected 27 injected and 27 null targets")
    print(f"Loading {model_id} and adapter {ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id,
                                        local_files_only=args.local_files_only,
                                        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    import hashlib
    fingerprint = hashlib.sha256(tok.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    if meta.get("tokenizer_sha256") and fingerprint != meta["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from M2-v2 generation; do not benchmark")
    saved_run = json.loads((ckpt.parent.parent.parent / "config.json").read_text(encoding="utf-8"))
    if saved_run.get("model_id") != model_id or saved_run.get("seed") != args.seed:
        raise ValueError("Adapter run metadata does not match requested model/seed")
    if saved_run.get("tokenizer_sha256") != fingerprint:
        raise ValueError("Tokenizer changed since adapter training")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        attn_implementation="eager", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, ckpt, is_trainable=False).to(device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = audit_canaries(
        model, tok, injected, nulls, refs, 512, device,
        ref_limit=1000, batch_size=args.batch_size,
        max_new_tokens=48)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scores = result["canary_scores"]
    count = len(scores)
    if count != len(injected) + len(nulls):
        raise AssertionError("Incomplete score set")
    if any(x["reference_count"] != 1000 for x in scores):
        raise AssertionError("Reference budget mismatch")
    peak = torch.cuda.max_memory_allocated()/1024**3
    reserved = torch.cuda.max_memory_reserved()/1024**3
    summary = {
        "purpose": "COMPUTE_BENCHMARK_ONLY_NOT_SCIENTIFIC_RESULT",
        "model": args.model, "model_id": model_id,
        "canary_version": args.canary_version,
        "adapter_origin": "M2-v1 smoke100 step 100",
        "canary_dataset_sha256": sha256(data/"meta.json"),
        "adapter": str(ckpt.relative_to(ROOT)),
        "reference_per_target": 1000,
        "targets_scored": count,
        "target_answers_scored": count * 1001,
        "scoring_batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "seconds_per_target": elapsed/count,
        "peak_gpu_allocated_gib": round(peak, 3),
        "peak_gpu_reserved_gib": round(reserved, 3),
        "estimated_54_target_seconds": elapsed/count*54,
        "note": ("Old M2-v1 adapter evaluated against M2-v2 strings; "
                 "do not interpret scores as exposure or leakage.")
                if args.canary_version == "v2" else
                "M2-v1 adapter and M2-v1 data: timing only, not formal evidence.",
    }
    output = ROOT/"outputs"/"benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.model}_{args.canary_version}_{count}targets_1000refs"
    path = output/(suffix+".json")
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Benchmark already exists: {path}; use --overwrite")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["qwen", "gemma", "llama"], required=True)
    p.add_argument("--canary-version", choices=["v1", "v2"], default="v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", default="smoke100")
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--targets-per-category", type=int, choices=range(1, 10),
                   default=None, metavar="N",
                   help="Score N injected and N null per category; omit for all 54")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
