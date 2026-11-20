#!/usr/bin/env python3
"""Plot M1 measured lengths, preserving CSV data alongside the figure."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/token_lengths_detail.csv"))
    ap.add_argument("--output", default=str(ROOT / "data/token_lengths.png"))
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    groups = defaultdict(list)
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[(row["model"], row["task_type"])].append(int(row["tokens"]))
    if not groups:
        raise ValueError("No tokenizer measurements found")
    fig, axes = plt.subplots(len({k[0] for k in groups}), 1,
                             figsize=(10, 3 * len({k[0] for k in groups})),
                             squeeze=False, constrained_layout=True)
    for i, model in enumerate(sorted({k[0] for k in groups})):
        ax = axes[i, 0]
        for task in ("probe", "utility"):
            values = groups.get((model, task))
            if values:
                ax.hist(values, bins=35, alpha=0.5, label=task)
        ax.axvline(args.max_length, linestyle="--", label="max_length")
        ax.set_title(f"{model} — M1 sequence length")
        ax.set_xlabel("Input + supervised answer + EOS tokens")
        ax.set_ylabel("Example count")
        ax.legend()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    main()
