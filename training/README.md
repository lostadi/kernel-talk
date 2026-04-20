# Kernel-Talk Training Pipeline

## Overview

Three-stage pipeline to fine-tune a retrieval reranker for kernel code search.

```
kernel headers → Mirror index → triplets → BM25 hard negatives → fine-tune reranker
```

## Quick Start

```bash
# 1. Build the Mirror index (required before any training)
make index

# 2. Generate synthetic triplets (no git required)
make synth           # → data/triplets.jsonl

# 3. Build BM25 index + enrich triplets with hard negatives
make bm25            # → data/bm25.pkl + data/enriched.jsonl

# 4. Fine-tune bi-encoder (fast; trains the retrieval backbone)
make train           # → training/checkpoints/

# 5. Fine-tune cross-encoder reranker (on top of bi-encoder)
python -m training.train \
    --config training/configs/reranker_v1.yaml \
    --triplets data/enriched.jsonl \
    --output training/checkpoints/reranker/

# 6. Evaluate retrieval quality
make eval
```

## Directory Layout

```
training/
├── README.md               # This file
├── bm25.py                 # BM25 index build + hard negative mining
├── dataset.py              # PyTorch dataset for triplets
├── eval.py                 # nDCG@k, Recall@k, MRR evaluation
├── mine.py                 # Triplet mining from kernel git log
├── synth.py                # Synthetic triplet generation (no git needed)
├── train.py                # Cross-encoder reranker training loop
├── train_biencoder.py      # Bi-encoder fine-tuning (CodeBERT)
├── __init__.py
├── checkpoints/            # Saved model weights (git-ignored)
├── configs/
│   └── reranker_v1.yaml    # Training hyperparameters
├── data/                   # Helper scripts for data preparation
│   ├── mine_triplets.py    # Wrapper for mine.py
│   ├── build_hard_negatives.py  # Wrapper for bm25.py
│   └── split_by_date.py    # Time-based train/val split
└── models/
    └── reranker.py         # KernelReranker nn.Module (cross-encoder head)
```

## Model Architecture

### Stage 1: Bi-encoder (baseline retrieval)

`microsoft/codebert-base` fine-tuned as a bi-encoder:
- Query tower: `mean_pool(BERT(query_tokens))` → ℝ^768
- Code tower: `mean_pool(BERT(code_tokens))` → ℝ^768
- Similarity: cosine similarity
- Loss: InfoNCE (in-batch negatives)

### Stage 2: Cross-encoder reranker

`microsoft/codebert-base` with a scalar scoring head:
- Input: `[CLS] query [SEP] candidate_text [SEP]`
- Head: `Linear(768, 1)` on `[CLS]` token
- Loss: margin-based ranking loss
- Takes top-100 from bi-encoder, rescores, returns top-k

## Data Pipeline

### Triplet format (triplets.jsonl)

```json
{"query": "how does copy_process work", "positive": "code of copy_process...", "negatives": ["code of unrelated func..."]}
```

### Strategies (synth.py)

1. **Symbol name** (p=0.5): query from function/struct name
2. **Docstring** (p=0.3): query from first sentence of comment
3. **Caller-callee** (p=0.7): query from caller, positive = callee
4. **Subsystem** (p=0.6): query from subsystem concept, positive = representative function

## Evaluation

Three settings compared:

| Setting | Description |
|---------|-------------|
| `biencoder` | Bi-encoder vector search only (baseline) |
| `rule_rerank` | Bi-encoder + rule-based reranker (type boost, doc length) |
| `learned_rerank` | Bi-encoder + cross-encoder reranker |

Metrics: nDCG@5, nDCG@10, Recall@5, Recall@10, MRR

A learned reranker is worth keeping only if it beats rule_rerank on Recall@5
by more than noise floor (≈ 0.01) across 3 seeds (spec §6.5).

## Hardware Requirements

| Hardware | Bi-encoder | Cross-encoder reranker |
|----------|-----------|----------------------|
| RTX 3090/4090 (24GB) | ~30 min | ~6 hours |
| RTX 3060 (12GB) | ~1 hour | ~18 hours |
| M2 Max (MPS) | ~2 hours | ~12 hours |
| CPU only | Not recommended | Not recommended |

## Config Reference

See `configs/reranker_v1.yaml` for all hyperparameters with comments.
Key settings:
- `training.lr`: 2e-5 (AdamW)
- `training.batch_size`: 32 per GPU
- `training.margin`: 1.0 (ranking loss margin)
- `inference.rerank_top_n`: 100 (bi-encoder candidates to rerank)
