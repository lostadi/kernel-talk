"""
training/train.py — Cross-encoder reranker training loop.

Spec §6.3 pseudocode expanded:

  load config
  load tokenizer, model, optimizer
  stream triplets from disk
  for epoch in ...:
    for batch in loader:
      q_pos = tokenize(query, positive)
      q_neg = tokenize(query, negative)
      s_pos = model(q_pos)
      s_neg = model(q_neg)
      loss = margin_loss(s_pos, s_neg)
      step, log, eval every N steps
  save best checkpoint by eval nDCG

Usage:
    python -m training.train \\
        --config training/configs/reranker_v1.yaml \\
        --triplets data/enriched.jsonl \\
        --output training/checkpoints/reranker/ \\
        [--device cuda|mps|cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterator

import torch
import yaml
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from training.models.reranker import KernelReranker, margin_ranking_loss, tokenize_pair


# ── Data loading ──────────────────────────────────────────────────────────────

def load_triplets(path: str) -> list[dict]:
    """Load (query, positive, negative) triplets from a JSONL file."""
    triplets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    triplets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return triplets


def iter_batches(
    triplets: list[dict],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(triplets)))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [triplets[i] for i in indices[start : start + batch_size]]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_reranker(
    model: KernelReranker,
    tokenizer: AutoTokenizer,
    val_triplets: list[dict],
    device: torch.device,
    max_length: int = 512,
    top_k: int = 5,
) -> dict[str, float]:
    """
    Compute nDCG@k and Recall@k on validation triplets.

    For each query we score positive + all its negatives, then rank.
    Since we only have 1 positive per query, Recall@k = 1 if positive
    is in top-k, 0 otherwise. nDCG@k is the same in the binary case.
    """
    model.eval()
    hits_at_k = 0
    ndcg_at_k = 0.0
    total = 0

    with torch.no_grad():
        for t in val_triplets:
            query = t.get("query", "")
            pos = t.get("positive", "")
            negatives = t.get("negatives", [t.get("negative", "")])
            if not negatives:
                continue

            candidates = [pos] + negatives
            enc = tokenize_pair(
                tokenizer, [query] * len(candidates), candidates, max_length
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            scores = model(**enc).cpu().tolist()

            # Rank: higher score = more relevant
            ranked = sorted(range(len(candidates)), key=lambda i: -scores[i])
            pos_rank = ranked.index(0) + 1  # 1-based rank of the positive

            if pos_rank <= top_k:
                hits_at_k += 1
                ndcg_at_k += 1.0 / math.log2(pos_rank + 1)
            total += 1

    model.train()
    recall = hits_at_k / max(total, 1)
    ndcg = ndcg_at_k / max(total, 1)
    return {f"recall_at_{top_k}": recall, f"ndcg_at_{top_k}": ndcg, "total": total}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    config_path: str,
    triplets_path: str,
    output_dir: str,
    device_name: str | None = None,
) -> None:
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    output_cfg = cfg.get("output", {})

    # Device
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train] Device: {device}", flush=True)

    # Random seed
    seed = training_cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    # Model + tokenizer
    model_name = model_cfg["base"]
    print(f"[train] Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = KernelReranker(model_name=model_name).to(device)

    # Load triplets
    print(f"[train] Loading triplets from {triplets_path}", flush=True)
    triplets = load_triplets(triplets_path)
    max_t = data_cfg.get("max_triplets", 0)
    if max_t and len(triplets) > max_t:
        triplets = triplets[:max_t]
    print(f"[train] {len(triplets)} triplets loaded", flush=True)

    # Train/val split
    val_frac = data_cfg.get("val_fraction", 0.05)
    n_val = max(1, int(len(triplets) * val_frac))
    val_triplets = triplets[:n_val]
    train_triplets = triplets[n_val:]
    print(f"[train] train={len(train_triplets)}, val={len(val_triplets)}", flush=True)

    # Optimizer + scheduler
    lr = float(training_cfg["lr"])
    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    grad_acc = training_cfg.get("grad_accumulation", 1)
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    margin = float(training_cfg.get("margin", 1.0))
    eval_steps = training_cfg.get("eval_steps", 500)
    max_length = model_cfg.get("max_length", 512)
    top_k = int(cfg.get("inference", {}).get("output_top_k", 5))

    steps_per_epoch = math.ceil(len(train_triplets) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * training_cfg.get("warmup_fraction", 0.1))

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_ndcg = 0.0
    global_step = 0

    print(f"[train] Training for {epochs} epochs, {total_steps} total steps", flush=True)

    for epoch in range(1, epochs + 1):
        print(f"[train] Epoch {epoch}/{epochs}", flush=True)
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(iter_batches(train_triplets, batch_size, seed=seed + epoch)):
            queries  = [t.get("query", "") for t in batch]
            positives = [t.get("positive", "") for t in batch]
            negatives = [t.get("negatives", [t.get("negative", "")])[0] for t in batch]

            enc_pos = tokenize_pair(tokenizer, queries, positives, max_length)
            enc_neg = tokenize_pair(tokenizer, queries, negatives, max_length)
            enc_pos = {k: v.to(device) for k, v in enc_pos.items()}
            enc_neg = {k: v.to(device) for k, v in enc_neg.items()}

            pos_scores = model(**enc_pos)
            neg_scores = model(**enc_neg)
            loss = margin_ranking_loss(pos_scores, neg_scores, margin=margin)
            loss = loss / grad_acc
            loss.backward()
            running_loss += loss.item() * grad_acc

            if (step + 1) % grad_acc == 0 or (step + 1) == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = running_loss / max(step + 1, 1)
                    print(
                        f"[train] step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

                # Evaluate and maybe save best
                if global_step % eval_steps == 0:
                    metrics = evaluate_reranker(
                        model, tokenizer, val_triplets, device,
                        max_length=max_length, top_k=top_k,
                    )
                    ndcg = metrics.get(f"ndcg_at_{top_k}", 0.0)
                    print(f"[train] eval step={global_step}: {metrics}", flush=True)
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        model.save(output_path / "best")
                        print(f"[train] ✓ New best nDCG@{top_k}={ndcg:.4f} — saved", flush=True)

        avg_loss = running_loss / max(steps_per_epoch, 1)
        print(f"[train] End of epoch {epoch}: avg_loss={avg_loss:.4f}", flush=True)

    # Final eval + save
    metrics = evaluate_reranker(
        model, tokenizer, val_triplets, device, max_length=max_length, top_k=top_k
    )
    print(f"[train] Final metrics: {metrics}", flush=True)
    model.save(output_path / "final")
    print(f"[train] ✓ Training complete. Checkpoints at {output_path}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cross-encoder reranker")
    p.add_argument("--config",   default="training/configs/reranker_v1.yaml")
    p.add_argument("--triplets", default="data/enriched.jsonl")
    p.add_argument("--output",   default="training/checkpoints/reranker/")
    p.add_argument("--device",   default=None, help="cuda|mps|cpu (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.config, args.triplets, args.output, args.device)
