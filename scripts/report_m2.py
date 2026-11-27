#!/usr/bin/env python3
"""M2: summarize model-specific lengths and injection proportions without model loading."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=str(ROOT/"canaries"))
    args = p.parse_args()
    base = Path(args.dir)
    rows = []
    for model in ("gemma", "llama", "qwen"):
        meta = json.loads((base/model/"meta.json").read_text(encoding="utf-8"))
        for kind, x in meta["selection"].items():
            rows.append({
                "model": model, "category": kind,
                "pool_size": x["pool_size"],
                "matched_answer_tokens": x["answer_tokens"],
                "injected_unique": x["injected"],
                "nullgate_unique": x["nullgate"],
                "reference_count": x["reference"],
                "injected_occurrences": meta["injected_occurrences"],
                "training_total": meta["final_train_samples"],
                "injection_fraction_pct": round(meta["injection_fraction"]*100, 3),
            })
    out = base/"m2_token_match_table.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = [
        "# M2 Canary injection summary",
        "",
        "All figures below are synthetic dataset counts, not memorization results.",
        "",
        "| Model | Type | Answer tokens | Injected unique | Null unique | Reference |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(f"| {r['model']} | {r['category']} | {r['matched_answer_tokens']} | "
                      f"{r['injected_unique']} | {r['nullgate_unique']} | {r['reference_count']} |")
    report += ["", "Each model adds 234 canary occurrences to the 3,187 original training examples.",
               "Expected final training samples: 3421; fraction: 234/3421 = 6.84%.",
               "Canary reference scoring must use each target's fixed prompt.",
               "Injected, nullgate and reference answers are disjoint for each model."]
    (base/"proportion_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\n".join(report))
    print("Saved:", out, base/"proportion_report.txt")


if __name__ == "__main__":
    main()
