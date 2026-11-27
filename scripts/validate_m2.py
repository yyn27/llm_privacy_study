#!/usr/bin/env python3
"""M2 fail-closed validation of counts, disjoint sets, exposure repetitions and M1 preservation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("id", "phone", "name_dx")
REPEATS = (1, 5, 20)


def load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(data, out, models):
    cfg = yaml.safe_load((ROOT/"configs/m2.yaml").read_text(encoding="utf-8"))["canaries"]
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    train = load(data/"train.jsonl")
    original = Counter(x["sample_id"] for x in train)
    if len(train) != manifest["sample_counts"]["train"]["total"]:
        raise ValueError("M1 train count does not match manifest")
    for key in models:
        folder = out/key
        meta = json.loads((folder/"meta.json").read_text(encoding="utf-8"))
        inj, nulls, ref, merged = (
            load(folder/"injected.jsonl"), load(folder/"nullgate.jsonl"),
            load(folder/"reference.jsonl"), load(folder/"train_with_canaries.jsonl"))
        k = cfg["k_per_cell"]
        assert len(inj) == len(nulls) == 9*k, "unique canary count incorrect"
        assert len(ref) == 3*cfg["reference_per_type"], "reference count incorrect"
        assert len(merged) == len(train) + 3*k*sum(REPEATS), "train length incorrect"
        assert meta["m1_train_sha256"] == digest(data/"train.jsonl")
        assert meta["m1_manifest_sha256"] == digest(data/"manifest.json")
        assert meta["injection_fraction"] <= cfg["maximum_fraction"]
        assert Counter(x["sample_id"] for x in merged if x["task_type"] != "canary") == original
        values = [r["answer"] for r in inj+nulls+ref]
        assert len(values) == len(set(values)), "injected/null/reference overlap"
        ids = [r["canary_id"] for r in inj+nulls]
        assert len(ids) == len(set(ids)), "duplicate canary IDs"
        count = Counter((r["membership"], r["category"], r["repeat"]) for r in inj+nulls)
        for membership in ("injected", "nullgate"):
            for kind in TYPES:
                for reps in REPEATS:
                    assert count[(membership, kind, reps)] == k, "bad repetition group"
        occurrences = Counter(x["canary_id"] for x in merged if x["task_type"] == "canary")
        assert len(occurrences) == len(inj), "null or unknown target in training"
        assert not ({n["canary_id"] for n in nulls} & set(occurrences))
        for x in inj:
            assert occurrences[x["canary_id"]] == x["repeat"], "repetition mismatch"
            assert len(x["positions_zero_indexed"]) == x["repeat"]
            assert all(merged[i]["canary_id"] == x["canary_id"]
                       for i in x["positions_zero_indexed"])
        for kind in TYPES:
            items = [x for x in inj+nulls+ref if x["category"] == kind]
            assert len({x["answer_token_length"] for x in items}) == 1
        for r in merged:
            assert r["text"] == r["prompt"] + r["answer"]
        print(f"[PASS] {key}: {len(inj)} injected canaries, "
              f"{len(nulls)} null, {len(ref)} reference; "
              f"{sum(occurrences.values())} injected appearances; "
              f"{meta['injection_fraction']:.2%} fraction")
    print("M2 VALIDATION PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT/"data"))
    p.add_argument("--out", default=str(ROOT/"canaries"))
    p.add_argument("--models", default="gemma,llama,qwen")
    args = p.parse_args()
    validate(Path(args.data), Path(args.out), args.models.split(","))


if __name__ == "__main__":
    main()
