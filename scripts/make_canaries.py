#!/usr/bin/env python3
"""M2: deterministic per-tokenizer canary injection, null controls and reference sets."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("id", "phone", "name_dx")
REPEATS = (1, 5, 20)


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def answer_for(kind, code):
    # 555-0100 through 555-0199 is the North American fictional-use exchange range.
    return {
        "id": f" CNRY-{code[:8]}-{code[8:16]}",
        "phone": f" +1-{200 + int(code[:8], 16) % 800:03d}-555-{100 + int(code[8:16], 16) % 100:04d}",
        "name_dx": f" Synthetic diagnosis DX-{code[:8]}-{code[8:16]}",
    }[kind]


def prompt_for(kind, subject):
    question = {
        "id": f"What is the synthetic identifier of {subject}?",
        "phone": f"What is the synthetic phone number of {subject}?",
        "name_dx": f"What is the synthetic diagnosis recorded for {subject}?",
    }[kind]
    return f"Question: {question}\nAnswer:"


def make_pool(kind, seed, count):
    rng = random.Random(f"{seed}:{kind}:pool-v1")
    values = set()
    while len(values) < count:
        code = f"{rng.getrandbits(64):016X}"
        values.add(answer_for(kind, code))
    return sorted(values)


def pick_equal_length(values, tok, needed, seed, kind):
    # Score/reference sets must have exactly the SAME answer token count.
    # This controls simple token-length ranking artifacts, not architecture effects.
    buckets = defaultdict(list)
    # Batch to bound tokenizer memory use.
    for i in range(0, len(values), 512):
        chunk = values[i:i+512]
        lengths = tok(chunk, add_special_tokens=False)["input_ids"]
        for value, ids in zip(chunk, lengths):
            buckets[len(ids)].append(value)
    best = min(buckets, key=lambda n: (-len(buckets[n]), n))
    if len(buckets[best]) < needed:
        raise ValueError(f"{kind}: only {len(buckets[best])} same-length candidates; "
                         f"need {needed}; increase configs/m2.yaml pool_per_type")
    matched = list(buckets[best])
    random.Random(f"{seed}:{kind}:select-v1").shuffle(matched)
    return best, matched[:needed], {str(k): len(v) for k,v in sorted(buckets.items())}


def make_entry(model, kind, membership, i, answer, k, token_len):
    subject = f"CanarySubject_{kind}_{model}_{membership}_{i:03d}"
    ident = f"{model}-{kind}-{membership}-{i:03d}"
    return dict(canary_id=ident, category=kind, membership=membership,
                repeat=REPEATS[i // k], subject=subject,
                prompt=prompt_for(kind, subject), answer=answer,
                answer_token_length=token_len)


def create_model(model, model_info, config, train, manifest, offline, target):
    tok = AutoTokenizer.from_pretrained(
        model_info["id"], local_files_only=offline, trust_remote_code=False)
    seed, k, ref_n = (config[x] for x in
                      ("data_seed", "k_per_cell", "reference_per_type"))
    n = k * len(REPEATS)
    unique_injected, nulls, refs, stats = [], [], [], {}
    for kind in TYPES:
        pool = make_pool(kind, seed, config["pool_per_type"])
        length, selected, dist = pick_equal_length(pool, tok, 2*n+ref_n, seed, kind)
        injected = [make_entry(model, kind, "injected", i, a, k, length)
                    for i, a in enumerate(selected[:n])]
        null = [make_entry(model, kind, "nullgate", i, a, k, length)
                for i, a in enumerate(selected[n:2*n])]
        reference = [dict(category=kind, answer=a, answer_token_length=length)
                     for a in selected[2*n:2*n+ref_n]]
        unique_injected.extend(injected)
        nulls.extend(null)
        refs.extend(reference)
        stats[kind] = {"pool_size": len(pool), "answer_tokens": length,
                       "length_buckets": dist, "injected": n, "nullgate": n,
                       "reference": len(reference)}
    answers = [x["answer"] for x in unique_injected+nulls+refs]
    if len(answers) != len(set(answers)):
        raise ValueError("Canary answer collision")
    # Ensure no synthetic canary subject is present in the M1 training data.
    train_text = "\n".join(x["text"] for x in train).casefold()
    if any(c["subject"].casefold() in train_text for c in unique_injected+nulls):
        raise ValueError("A synthetic canary subject already exists in M1 data")
    injected_rows = []
    for c in unique_injected:
        for occurrence in range(c["repeat"]):
            sid = hashlib.sha256(f"M2|{c['canary_id']}|{occurrence}".encode()).hexdigest()[:20]
            injected_rows.append({
                "sample_id": sid, "patient_id": "synthetic-canary:"+c["canary_id"],
                "split": "train", "task_type": "canary", "field": c["category"],
                "prompt": c["prompt"], "answer": c["answer"],
                "text": c["prompt"]+c["answer"], "template_version": "m1_plain_qa_v1",
                "canary_id": c["canary_id"], "repeat": c["repeat"],
                "occurrence": occurrence+1})
    expected = len(TYPES)*k*sum(REPEATS)
    if len(injected_rows) != expected:
        raise ValueError(f"Incorrect number of injected examples: {len(injected_rows)}")
    combined = train + injected_rows
    random.Random(f"{seed}:{model}:interleave-v1").shuffle(combined)
    positions = defaultdict(list)
    for index, row in enumerate(combined):
        if row["task_type"] == "canary":
            positions[row["canary_id"]].append(index)
    for c in unique_injected:
        c["positions_zero_indexed"] = positions[c["canary_id"]]
    for c in unique_injected+nulls:
        pids = tok(c["prompt"], add_special_tokens=True)["input_ids"]
        aids = tok(c["answer"]+(tok.eos_token or ""),
                   add_special_tokens=False)["input_ids"]
        if not aids or len(pids)+len(aids) > manifest["max_length"]:
            raise ValueError(f"{model}: {c['canary_id']} exceeds max_length")
        c["sequence_tokens_with_eos"] = len(pids)+len(aids)
    if Counter(x["sample_id"] for x in combined if x["task_type"] != "canary") != Counter(
            x["sample_id"] for x in train):
        raise AssertionError("M1 training data modified")
    fraction = len(injected_rows)/len(combined)
    if fraction > config["maximum_fraction"]:
        raise ValueError(f"Canary injection share {fraction:.2%} exceeds cap")
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target/"injected.jsonl", unique_injected)
    write_jsonl(target/"nullgate.jsonl", nulls)
    write_jsonl(target/"reference.jsonl", refs)
    write_jsonl(target/"train_with_canaries.jsonl", combined)
    meta = {
        "m2_version": "m2_canaries_v1", "model_key": model,
        "model_id": model_info["id"],
        "tokenizer_revision": tok.init_kwargs.get("_commit_hash") or "unavailable",
        "m1_manifest_sha256": sha_file(ROOT/"data/manifest.json"),
        "m1_train_sha256": sha_file(ROOT/"data/train.jsonl"),
        "data_seed": seed, "pool_per_type": config["pool_per_type"],
        "k_per_cell": k, "repetition_levels": list(REPEATS),
        "reference_per_type": ref_n,
        "injected_unique": len(unique_injected), "null_unique": len(nulls),
        "injected_occurrences": len(injected_rows),
        "m1_train_samples": len(train), "final_train_samples": len(combined),
        "injection_fraction": fraction, "selection": stats,
        "scoring_rule": "Score each injected/null answer and all references under the IDENTICAL target prompt; answer NLL only, no EOS for reference ranking.",
        "scientific_limitation": "Phone/diagnosis formats are deliberately synthetic, not clinically realistic. Proxy validity must be tested later.",
    }
    (target/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT/"configs/m2.yaml"))
    p.add_argument("--model", default="all", help="all | gemma | llama | qwen")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["canaries"]
    if cfg["pool_per_type"] < 5000 or cfg["reference_per_type"] < 1000:
        raise ValueError("M2 requires pool >=5000, reference >=1000 per type")
    if cfg["repetition_levels"] != list(REPEATS) or cfg["k_per_cell"] < 1:
        raise ValueError("Frozen repetitions must be [1,5,20], k >=1")
    manifest = json.loads((ROOT/"data/manifest.json").read_text(encoding="utf-8"))
    if manifest["m1_version"] != "m1_plain_qa_v1":
        raise ValueError("Unexpected M1 template version")
    train = read_jsonl(ROOT/"data/train.jsonl")
    if len(train) != manifest["sample_counts"]["train"]["total"]:
        raise ValueError("M1 train sample count mismatch")
    registry = yaml.safe_load((ROOT/"configs/models.yaml").read_text(
        encoding="utf-8"))["models"]
    keys = list(registry) if args.model == "all" else [args.model]
    if any(x not in registry for x in keys):
        p.error(f"Unknown model; choose {list(registry)} or all")
    for key in keys:
        target = ROOT/"canaries"/key
        if target.exists():
            if not args.force:
                raise FileExistsError(f"{target} already exists; --force explicitly replaces it")
            shutil.rmtree(target)
        print(f"[M2] Constructing {key}", flush=True)
        meta = create_model(key, registry[key], cfg, train, manifest,
                            args.local_files_only, target)
        print(f"[M2] {key}: injected {meta['injected_unique']} unique / "
              f"{meta['injected_occurrences']} occurrences; "
              f"share {meta['injection_fraction']:.2%}", flush=True)
    print("M2 generation complete. Next: python scripts/validate_m2.py")


if __name__ == "__main__":
    main()
