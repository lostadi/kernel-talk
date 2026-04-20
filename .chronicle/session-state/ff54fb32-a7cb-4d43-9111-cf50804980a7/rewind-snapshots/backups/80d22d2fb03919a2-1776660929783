"""
training/data/split_by_date.py — Time-based train/val split for triplets.

When triplets have a 'date' field (from git-log mining), split so that
all val triplets are more recent than all train triplets.  This prevents
data leakage and tests generalization to new kernel code.

If triplets have no date field, falls back to a random 95/5 split.

Usage:
    python training/data/split_by_date.py \\
        --input data/enriched.jsonl \\
        --train data/train.jsonl \\
        --val data/val.jsonl \\
        [--val-fraction 0.05]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split(
    input_path: str,
    train_path: str,
    val_path: str,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[int, int]:
    triplets = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    triplets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Check if date field is present
    has_date = any("date" in t for t in triplets)

    if has_date:
        # Sort by date ascending, take last val_fraction as val
        triplets.sort(key=lambda t: t.get("date", ""), reverse=False)
        n_val = max(1, int(len(triplets) * val_fraction))
        train = triplets[:-n_val]
        val = triplets[-n_val:]
    else:
        # Random split
        rng = random.Random(seed)
        rng.shuffle(triplets)
        n_val = max(1, int(len(triplets) * val_fraction))
        val = triplets[:n_val]
        train = triplets[n_val:]

    def write_jsonl(path: str, data: list[dict]) -> None:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    return len(train), len(val)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True)
    p.add_argument("--train",        default="data/train.jsonl")
    p.add_argument("--val",          default="data/val.jsonl")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    n_train, n_val = split(args.input, args.train, args.val, args.val_fraction, args.seed)
    print(f"Split: {n_train} train, {n_val} val")
