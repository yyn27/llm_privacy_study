#!/usr/bin/env python3
"""M1 fail-closed validation: split isolation, schemas, source and token budget."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def validate(data):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["synthea"] is None:
        raise ValueError("Missing pinned Synthea generation provenance")
    selected = manifest["patient_ids"]
    splits = ("train", "val", "test")
    for i, left in enumerate(splits):
        ids = selected[left]
        if len(ids) != manifest["selected_patient_counts"][left] or len(set(ids)) != len(ids):
            raise ValueError(f"{left} patient count/uniqueness mismatch")
        for right in splits[i+1:]:
            if set(ids) & set(selected[right]):
                raise ValueError(f"Patient overlap: {left} & {right}")
    sample_ids = set()
    for split in splits:
        items = list(read(data / f"{split}.jsonl"))
        expected = manifest["sample_counts"][split]
        if len(items) != expected["total"]:
            raise ValueError(f"{split}: sample count mismatch")
        counts = Counter()
        for item in items:
            if item["patient_id"] not in selected[split]:
                raise ValueError(f"{split}: patient ID contamination")
            if item["split"] != split:
                raise ValueError("Sample split mismatch")
            if item["sample_id"] in sample_ids:
                raise ValueError("Duplicate sample_id")
            sample_ids.add(item["sample_id"])
            if item["task_type"] not in ("probe", "utility"):
                raise ValueError("Unknown task type")
            if item["task_type"] == "probe" and "Medical record:" in item["prompt"]:
                raise ValueError("Probe unexpectedly includes a clinical record")
            if item["task_type"] == "utility" and "Medical record:" not in item["prompt"]:
                raise ValueError("Utility sample has no medical record")
            if item["text"] != item["prompt"] + item["answer"]:
                raise ValueError("Prompt/answer concatenation mismatch")
            if not item["answer"].strip():
                raise ValueError("Empty supervised answer")
            counts[item["task_type"]] += 1
        if dict(counts) != {k: expected[k] for k in ("probe", "utility")}:
            raise ValueError(f"{split}: task count mismatch")
    lengths_path = data / "token_lengths.csv"
    if not lengths_path.exists():
        raise ValueError("Missing three-tokenizer length audit; rerun without --skip-token-audit")
    import csv
    with lengths_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18:  # 3 models × 3 splits × 2 tasks
        raise ValueError(f"Expected 18 tokenizer summary rows, got {len(rows)}")
    long_rows = [x for x in rows if int(x["over_max_length"]) > 0
                 or int(x["missing_answer_tokens"]) > 0]
    if long_rows:
        raise ValueError(f"Overlength or empty-token answers: {long_rows}. "
                         "Do not proceed to M3 before resolving the sequence budget.")
    print("M1 VALIDATION PASS: patient isolation, QA templates, provenance and token budget")
    print("Patients:", manifest["selected_patient_counts"])
    print("Samples:", manifest["sample_counts"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data"))
    args = p.parse_args()
    validate(Path(args.data))


if __name__ == "__main__":
    main()
