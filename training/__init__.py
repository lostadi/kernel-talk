"""
training/
─────────
Phase 3: Training pipeline for fine-tuned code retrieval.

The retrieval model has two stages that are trained independently:

  Stage 1 — Bi-encoder (dense retrieval)
    Two CodeBERT instances share weights. One encodes the query (commit
    message / user question), the other encodes the code (function body).
    Trained with InfoNCE loss using in-batch negatives + BM25 hard negatives.
    Output: per-node embeddings that replace the default CodeBERT embeddings
    in ChromaDB. This improves Recall@k — the right function appears in the
    top-k results more often.

  Stage 2 — Cross-encoder (reranker)
    A single BERT instance sees (query, code) concatenated. It outputs a
    scalar relevance score. Only run on the top-k candidates from Stage 1.
    Trained with margin ranking loss on the same triplets.
    This improves Hit@1 — the best result moves to rank 1.

Training data comes from Linux kernel git history (training/mine.py).
Evaluation uses eval/retrieval_gold.jsonl (ktalk eval retrieval).

Pipeline:
    python -m training.mine   --kernel /path/to/linux > data/triplets.jsonl
    python -m training.bm25   --triplets data/triplets.jsonl --index data/bm25.pkl
    python -m training.train  --triplets data/triplets.jsonl --bm25 data/bm25.pkl
    ktalk eval retrieval      --model training/checkpoints/best_biencoder
"""
