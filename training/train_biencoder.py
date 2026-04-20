"""
training/train_biencoder.py
────────────────────────────
Fine-tune a CodeBERT bi-encoder for kernel code retrieval.

Architecture
─────────────
A bi-encoder has two encoder towers that share weights (or not — this
is a hyperparameter). For retrieval:

  f(query)  = mean-pool(BERT(query_tokens))      → ℝ^768
  g(code)   = mean-pool(BERT(code_tokens))       → ℝ^768
  sim(q, c) = cosine(f(q), g(c))                 → [-1, 1]

Shared weights (default) halves the parameter count and typically works
as well as separate weights for code retrieval. The argument: query and
code share enough vocabulary (function names appear in both) that joint
representation is beneficial.

Training objective: InfoNCE (in-batch negatives)
────────────────────────────────────────────────
For a batch of B (query, positive) pairs:

  L = -1/B · Σ_i log( exp(sim(q_i, c_i+) / τ) / Σ_j exp(sim(q_i, c_j) / τ) )

where c_j ranges over ALL code vectors in the batch (B positives + B*(N-1)
explicit negatives). Temperature τ controls sharpness (default 0.07,
following SimCLR and MoCo).

With batch size 32 and N=7 negatives per example, each query sees:
  32 × (1+7) - 1 = 255 negatives
This is large enough for effective contrastive learning without memory banks.

Why not triplet loss?
InfoNCE consistently outperforms triplet loss for dense retrieval (DPR,
ANCE, E5 papers all demonstrate this). The intuition: triplet loss only
pushes one negative away per step; InfoNCE uses all in-batch negatives
simultaneously, giving a much richer gradient signal.

Training config
───────────────
  Backbone:   microsoft/codebert-base (125M params)
  Optimizer:  AdamW, lr=2e-5, weight_decay=0.01
  Scheduler:  Linear warmup (10% of steps) + cosine decay
  Batch:      32 (effective: 32*8=256 with grad accumulation)
  Epochs:     3 (sufficient for retrieval fine-tuning; more → overfitting)
  Checkpoint: save best model by val Recall@10

Usage:
    python -m training.train_biencoder \\
        --triplets data/enriched.jsonl \\
        --storage ~/.kernel-talk/store \\
        --output training/checkpoints/ \\
        --batch-size 32 \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    pass


# ─── Model ────────────────────────────────────────────────────────────────────

def build_biencoder(model_name: str = "microsoft/codebert-base"):
    """
    Wrap a pretrained BERT model as a bi-encoder.

    Both towers share the same underlying model object — this means they
    share parameters during both forward pass and gradient computation.
    When we call encoder(query_tokens) and encoder(code_tokens), PyTorch
    accumulates gradients for the shared weights from both directions.

    The mean-pool over non-padding tokens is the standard approach for
    sentence embeddings with BERT. [CLS] pooling also works but mean-pool
    tends to give better retrieval performance (Reimers & Gurevych 2019).
    """
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "Training requires: pip install torch transformers\n"
            "For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )

    class BiEncoder(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.encoder.config.hidden_size

        def encode(self, input_ids, attention_mask) -> "torch.Tensor":
            """
            Encode a batch of texts into L2-normalized embeddings.
            Returns: (batch_size, hidden_size) float32 tensor.
            """
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Mean pool over non-padding tokens
            token_embeddings = outputs.last_hidden_state      # (B, L, H)
            mask = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
            summed = (token_embeddings * mask).sum(dim=1)     # (B, H)
            counts = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
            mean_pooled = summed / counts                      # (B, H)

            # L2 normalize — cosine similarity becomes dot product,
            # which is faster and numerically more stable
            return nn.functional.normalize(mean_pooled, p=2, dim=-1)

        def forward(self, query_ids, query_mask, code_ids, code_mask):
            q_emb = self.encode(query_ids, query_mask)
            c_emb = self.encode(code_ids, code_mask)
            return q_emb, c_emb

    return BiEncoder(model_name)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def infonce_loss(
    q_emb: "torch.Tensor",
    c_emb: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float = 0.07,
    n_per_item: list[int] | None = None,
) -> "torch.Tensor":
    """
    InfoNCE loss using in-batch negatives + explicit negatives.

    q_emb:  (B, H)         — query embeddings (one per example)
    c_emb:  (B*(1+N), H)   — code embeddings (positive first, then negatives)
    labels: (B,) int64     — index of positive within each example's slice (always 0)
    n_per_item: list of int — number of code vectors per example (1 positive + negatives)

    The key insight: in-batch negatives mean that q_i also "sees" the
    positives of other queries as negatives. This is implicit in how we
    construct the similarity matrix.

    If n_per_item is None, we use the simpler formulation where all code
    vectors are shared across all queries (standard in-batch negatives only).
    """
    import torch
    import torch.nn.functional as F

    B = q_emb.shape[0]

    if n_per_item is None:
        # Standard in-batch InfoNCE: each query sees all B code vectors
        # sim_matrix: (B, B)  where sim_matrix[i,j] = sim(q_i, c_j)
        sim_matrix = torch.matmul(q_emb, c_emb.T) / temperature
        # Target: q_i should be closest to c_i (its positive)
        targets = labels  # (B,)
        return F.cross_entropy(sim_matrix, targets)
    else:
        # Mixed in-batch + explicit negatives
        # For each query, compute logits over its own slice + in-batch positives
        # This is more expensive but uses hard negatives properly
        losses = []
        code_start = 0
        for i, n_codes in enumerate(n_per_item):
            q_i = q_emb[i:i+1]                              # (1, H)
            c_slice = c_emb[code_start:code_start + n_codes] # (n_codes, H)

            # Explicit codes for this query: positive + its negatives
            explicit_sim = torch.matmul(q_i, c_slice.T) / temperature  # (1, n_codes)

            # In-batch positives from OTHER queries (each is a free negative)
            other_positives = []
            ofs = 0
            for j, n_j in enumerate(n_per_item):
                if j != i:
                    other_positives.append(c_emb[ofs:ofs+1])  # just the positive
                ofs += n_j
            if other_positives:
                other_c = torch.cat(other_positives, dim=0)  # (B-1, H)
                inbatch_sim = torch.matmul(q_i, other_c.T) / temperature  # (1, B-1)
                logits = torch.cat([explicit_sim, inbatch_sim], dim=1)  # (1, n_codes+B-1)
            else:
                logits = explicit_sim

            # Positive is always index 0 in the explicit slice
            target = torch.zeros(1, dtype=torch.long, device=q_emb.device)
            losses.append(F.cross_entropy(logits, target))

            code_start += n_codes

        return torch.stack(losses).mean()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def eval_recall_at_k(
    model,
    val_loader,
    device: str,
    k: int = 10,
) -> float:
    """
    Compute Recall@k on the validation set.

    For each (query, positive) pair, encode all candidates and measure
    whether the true positive is in the top-k by cosine similarity.

    Note: this is an approximate eval using only the val batch's candidates
    — a full eval against the entire Mirror index is done by `ktalk eval retrieval`.
    This in-training eval is fast and directionally correct.
    """
    import torch

    model.eval()
    hits = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            q_ids = batch.query_input_ids.to(device)
            q_mask = batch.query_attention_mask.to(device)
            c_ids = batch.code_input_ids.to(device)
            c_mask = batch.code_attention_mask.to(device)

            q_emb, c_emb = model(q_ids, q_mask, c_ids, c_mask)
            # q_emb: (B, H), c_emb: (B*(1+N), H)

            B = q_emb.shape[0]
            sim = torch.matmul(q_emb, c_emb.T)  # (B, B*(1+N))

            # Positive is always the first code vector for each query
            # Figure out where each query's positive is in the flat code matrix
            # This is complex with variable n_per_item; we use a simplified
            # in-batch version: positive[i] is at index i (first code per query)
            pos_indices = torch.arange(B, device=device) * (c_emb.shape[0] // B)

            # Rank each query's scores
            _, sorted_indices = sim.sort(dim=1, descending=True)
            for i in range(B):
                pos_idx = pos_indices[i].item()
                rank = (sorted_indices[i] == pos_idx).nonzero(as_tuple=True)[0]
                if len(rank) > 0 and rank[0].item() < k:
                    hits += 1
                total += 1

    model.train()
    return hits / total if total > 0 else 0.0


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    triplets_jsonl: str,
    storage_dir: str,
    output_dir: str,
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 32,
    grad_accum_steps: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.1,
    temperature: float = 0.07,
    val_fraction: float = 0.05,
    seed: int = 42,
    device: str | None = None,
    max_train_items: int | None = None,
) -> None:
    """
    Full training loop for the bi-encoder.

    Checkpoints the model with best val Recall@10 to output_dir/best_biencoder/.
    Saves final model to output_dir/final_biencoder/.
    Training log written to output_dir/train_log.jsonl.
    """
    import random
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from training.dataset import TripletDataset, make_collate_fn

    # Setup
    random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] Device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from core.mirror.store import KernelStore
    store = KernelStore(storage_dir=storage_dir)

    # Load tokenizer and dataset
    print(f"[train] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collate_fn = make_collate_fn(tokenizer)

    print(f"[train] Loading dataset: {triplets_jsonl}")
    full_ds = TripletDataset(
        jsonl_path=triplets_jsonl,
        tokenizer=tokenizer,
        store=store,
        max_items=max_train_items,
    )

    # Train/val split (by index, but dataset was already sorted by date during mining)
    n_val = max(64, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"[train] Dataset: {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model
    print(f"[train] Loading model: {model_name}")
    model = build_biencoder(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {n_params:,}")

    # Optimizer + scheduler
    # Apply weight decay to all params except bias and LayerNorm
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=lr)

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training
    best_recall = 0.0
    log_path = output_path / "train_log.jsonl"
    log_f = open(log_path, "w")

    # Detect whether the dataset has difficulty metadata for curriculum scheduling.
    # full_ds is a TripletDataset; train_ds is a Subset wrapping it.
    curriculum_active = getattr(full_ds, "_has_difficulty", False)
    if curriculum_active:
        print(f"[train] Curriculum scheduling active "
              f"(difficulty metadata found in dataset)")
    else:
        print(f"[train] No difficulty metadata — curriculum scheduling disabled")

    print(f"[train] Starting training: {epochs} epochs, {total_steps} steps")
    model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Advance curriculum: tells the dataset what fraction of training is complete.
        # At epoch 0 → progress=0.0 (easy-first); at final epoch → progress=1.0 (full).
        # This call has no effect when curriculum_active is False (backward compatible).
        if curriculum_active:
            full_ds.set_curriculum_epoch(epoch, epochs)
            progress = full_ds._curriculum_progress
            print(f"[train] Epoch {epoch+1}: curriculum_progress={progress:.3f}")

        for batch_idx, batch in enumerate(train_loader):
            q_ids  = batch.query_input_ids.to(device)
            q_mask = batch.query_attention_mask.to(device)
            c_ids  = batch.code_input_ids.to(device)
            c_mask = batch.code_attention_mask.to(device)

            q_emb, c_emb = model(q_ids, q_mask, c_ids, c_mask)
            loss = infonce_loss(
                q_emb, c_emb, batch.labels.to(device),
                temperature=temperature,
            )

            # Gradient accumulation
            (loss / grad_accum_steps).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            n_batches += 1

            if global_step % 10 == 0 and global_step > 0:
                avg_loss = epoch_loss / n_batches
                print(f"[train] step={global_step}/{total_steps} epoch={epoch+1} loss={avg_loss:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        # End-of-epoch eval
        recall = eval_recall_at_k(model, val_loader, device, k=10)
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[train] Epoch {epoch+1} complete: loss={avg_loss:.4f} val_recall@10={recall:.4f}")

        log_entry = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": avg_loss,
            "val_recall_at_10": recall,
        }
        print(json.dumps(log_entry), file=log_f, flush=True)

        if recall > best_recall:
            best_recall = recall
            best_path = output_path / "best_biencoder"
            model.encoder.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            print(f"[train] ✓ New best model saved (recall@10={recall:.4f})", flush=True)

        # Always save epoch checkpoint so we have something regardless of val trend
        epoch_path = output_path / f"epoch_{epoch+1}"
        model.encoder.save_pretrained(str(epoch_path))
        tokenizer.save_pretrained(str(epoch_path))
        print(f"[train] ✓ Epoch {epoch+1} checkpoint saved → {epoch_path}", flush=True)

    # Save final
    final_path = output_path / "final_biencoder"
    model.encoder.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log_f.close()
    print(f"[train] Done. Best val Recall@10: {best_recall:.4f}")
    print(f"[train] Best model: {output_path / 'best_biencoder'}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bi-encoder for kernel code retrieval.")
    parser.add_argument("--triplets",    required=True, help="Enriched triplets JSONL")
    parser.add_argument("--storage",     required=True, help="KernelStore storage dir")
    parser.add_argument("--output",      required=True, help="Checkpoint output dir")
    parser.add_argument("--model",       default="microsoft/codebert-base")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device",      default=None, help="cuda / mps / cpu")
    parser.add_argument("--max-items",   type=int, default=None, help="Truncate dataset (dev mode)")
    args = parser.parse_args()

    train(
        triplets_jsonl=args.triplets,
        storage_dir=args.storage,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
        max_train_items=args.max_items,
    )
