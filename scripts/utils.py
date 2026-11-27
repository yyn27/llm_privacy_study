"""M3 shared deterministic utilities, QA encoding, and append-only JSONL logs."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import torch


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_hash(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def encode_qa(tokenizer, prompt, answer, max_length, *, include_eos=True):
    # Separate tokenization is the exact frozen M1 design.
    # Attention: at the prompt-answer boundary, BPE/SentencePiece
    # whole-string tokenization can differ; M3 uses these exact tokens.
    left = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    suffix = answer + ((tokenizer.eos_token or "") if include_eos else "")
    right = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    if not right or len(left) + len(right) > max_length:
        raise ValueError(f"Empty answer or overflow: {len(left)}+{len(right)} > {max_length}")
    return left + right, [-100] * len(left) + right, len(right)


def tensor_batch(encoded, pad_token_id, device):
    width = max(len(x[0]) for x in encoded)
    input_ids = []
    labels = []
    masks = []
    for ids, target, _ in encoded:
        padding = width - len(ids)
        input_ids.append(ids + [pad_token_id] * padding)
        labels.append(target + [-100] * padding)
        masks.append([1] * len(ids) + [0] * padding)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device)}


def batch_order(n, seed, epoch):
    indices = list(range(n))
    random.Random(f"training-order:{seed}:{epoch}").shuffle(indices)
    return indices


def save_training_state(path, model, optimizer, step, microstep, train_loss, seed):
    import numpy as np
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter", safe_serialization=True)
    state = {"optimizer": optimizer.state_dict(),
             "global_step": step, "microstep": microstep,
             "train_loss": train_loss, "seed": seed,
             "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "numpy_rng": np.random.get_state()}
    torch.save(state, path / "trainer_state.pt")


def restore_training_state(path, optimizer):
    import numpy as np
    # Only load locally created state; torch pickle is unsafe for untrusted files.
    state = torch.load(Path(path) / "trainer_state.pt", map_location="cpu",
                       weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    return state
