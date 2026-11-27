#!/usr/bin/env python3
"""Inspect M3 audit completeness, step-zero baseline and canary contact counts."""
import argparse
import json
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(folder, interval, total, require_zero):
    config = json.loads((folder/"config.json").read_text(encoding="utf-8"))
    rows = read(folder/"metrics.jsonl")
    expected = ([0] if require_zero else []) + list(range(interval, total+1, interval))
    if total % interval:
        expected.append(total)
    observed = [r["step"] for r in rows]
    if observed != expected:
        raise ValueError(f"{folder}: expected steps {expected}, got {observed}")
    seen_previous = {}
    for r in rows:
        scores = r["canary_scores"]
        if len(scores) != 54:
            raise ValueError(f"Step {r['step']}: expected 54 canaries, got {len(scores)}")
        injected = [x for x in scores if x["membership"] == "injected"]
        nulls = [x for x in scores if x["membership"] == "nullgate"]
        if len(injected) != 27 or len(nulls) != 27:
            raise ValueError("Injected/null canary counts not 27/27")
        if require_zero:
            for score in scores:
                count = score.get("actual_seen_so_far")
                if not isinstance(count, int) or count < 0:
                    raise ValueError("Missing actual_seen_so_far")
                if r["step"] == 0 and count != 0:
                    raise ValueError("Step 0 must have no canary exposure")
                if score["membership"] == "nullgate" and count != 0:
                    raise ValueError("Null-gate unexpectedly present in training")
                key = score["canary_id"]
                if count < seen_previous.get(key, 0):
                    raise ValueError("Canary contact count decreased")
                seen_previous[key] = count
        if r["step"] > 0:
            checkpoint = folder/"adapters"/f"step_{r['step']:06d}"
            if not (checkpoint/"adapter").is_dir() or not (checkpoint/"trainer_state.pt").is_file():
                raise FileNotFoundError(f"Checkpoint incomplete: {checkpoint}")
    print(f"PASS: {folder}  steps={observed}, records={len(rows)}, model={config['model']}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-tag",default="smoke100")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--models",default="qwen,gemma,llama")
    p.add_argument("--audit-interval",type=int,default=20)
    p.add_argument("--max-steps",type=int,default=100)
    p.add_argument("--require-step-zero",action="store_true")
    args=p.parse_args()
    for key in args.models.split(","):
        path=Path("runs")/key/str(args.seed)/args.run_tag
        check(path,args.audit_interval,args.max_steps,args.require_step_zero)


if __name__=="__main__":
    main()
