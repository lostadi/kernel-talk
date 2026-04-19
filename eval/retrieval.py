"""
eval/retrieval.py — End-to-end retrieval evaluation against a gold set.

Gold file format (JSONL):
  {
    "query": "how does fork work",
    "expected_symbols": ["kernel_clone", "copy_process"],
    "expected_files": ["kernel/fork.c"],
    "relevant_ids": ["kernel/fork.c::kernel_clone", "kernel/fork.c::copy_process"]
  }

The `relevant_ids` field is the authoritative ground truth used for scoring.
If it is absent it is automatically derived as the cross-product of
`expected_files × expected_symbols`.

Three retrieval settings are evaluated:
  1. biencoder   — bi-encoder vector search only (baseline)
  2. rule_rerank — bi-encoder + rule-based reranker
  3. (optional) learned_rerank — bi-encoder + learned cross-encoder reranker

Metrics: nDCG@k, Recall@k, MRR for k ∈ top_ks.

Usage:
    from eval.retrieval import run_eval
    results = run_eval(
        gold_path="eval/retrieval_gold.jsonl",
        storage_dir="~/.kernel-talk/store",
        reranker_path=None,
        top_ks=[5, 10],
    )
    # results: {setting: {metric: float}}

    # Or run_eval with no store (returns zero metrics gracefully):
    results = run_eval("eval/retrieval_gold.jsonl", storage_dir=None)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


# ── Metric helpers ────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_relevances: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at position k."""
    top = ranked_relevances[:k]
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(top))
    ideal = sorted(ranked_relevances, reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant documents retrieved in the top-k results."""
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant_ids)
    return hits / max(len(relevant_ids), 1)


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the first relevant document's rank."""
    for rank, rid in enumerate(ranked_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ── Gold data helpers ─────────────────────────────────────────────────────────

def load_gold(gold_path: str) -> list[dict]:
    """
    Load gold queries from a JSONL file.

    Derives `relevant_ids` from `expected_files × expected_symbols` when
    the field is not already present in the entry.
    """
    queries: list[dict] = []
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "relevant_ids" not in entry or not entry["relevant_ids"]:
                # Build cross-product: file::symbol for each (file, symbol) pair
                relevant_ids: list[str] = []
                for fpath in entry.get("expected_files", []):
                    for sym in entry.get("expected_symbols", []):
                        relevant_ids.append(f"{fpath}::{sym}")
                entry["relevant_ids"] = relevant_ids
            queries.append(entry)
    return queries


# ── Retrieval wrappers ────────────────────────────────────────────────────────

def _retrieve_biencoder(
    query: str,
    store: Any,
    top_n: int,
) -> list[tuple[str, float]]:
    """Call the KernelStore vector search and return (node_id, score) pairs."""
    results = store.vector_search(query, top_k=top_n)
    return [(r.node.id, r.score) for r in results]


def _rerank_rule_based(
    candidates: list[tuple[str, float]],
    store: Any,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Rule-based reranker: prefer functions over macros; boost nodes with longer
    docstrings (proxy for explanation quality).
    """
    TYPE_WEIGHTS = {"function": 1.0, "struct": 0.9, "enum": 0.8, "macro": 0.5}
    scored: list[tuple[str, float]] = []
    for node_id, base_score in candidates:
        node = store.graph.get_node(node_id)
        if node:
            type_boost = TYPE_WEIGHTS.get(node.node_type, 0.7)
            doc_boost = min(1.0, len(node.docstring or "") / 200)
            score = base_score * type_boost * (1 + 0.1 * doc_boost)
        else:
            score = base_score
        scored.append((node_id, score))
    return sorted(scored, key=lambda x: -x[1])[:top_k]


def _rerank_learned(
    query: str,
    candidates: list[tuple[str, float]],
    store: Any,
    reranker: Any,
    tokenizer: Any,
    device: Any,
    top_k: int = 10,
    max_length: int = 512,
) -> list[tuple[str, float]]:
    """Rerank candidates with a learned cross-encoder."""
    import torch
    from training.models.reranker import tokenize_pair

    node_ids = [nid for nid, _ in candidates]
    texts = []
    for node_id in node_ids:
        node = store.graph.get_node(node_id)
        texts.append(node.embedding_text() if node else "")

    reranker.eval()
    with torch.no_grad():
        enc = tokenize_pair(tokenizer, [query] * len(texts), texts, max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        scores = reranker(**enc).cpu().tolist()

    ranked = sorted(zip(node_ids, scores), key=lambda x: -x[1])
    return ranked[:top_k]


# ── Main evaluation function ──────────────────────────────────────────────────

def run_eval(
    gold_path: str,
    storage_dir: str | None,
    reranker_path: str | None = None,
    top_ks: list[int] | None = None,
    rerank_n: int = 100,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Evaluate retrieval quality against a gold set.

    Args:
        gold_path:     Path to retrieval_gold.jsonl.
        storage_dir:   Path to KernelStore directory. If None or the path does
                       not exist, returns zero metrics for all settings.
        reranker_path: Optional path to a learned reranker checkpoint directory.
        top_ks:        List of k values for @k metrics (default: [5, 10]).
        rerank_n:      Bi-encoder candidates to pass to the reranker.
        verbose:       Print progress to stdout.

    Returns:
        Nested dict: {setting: {metric: float_value}}
        Settings: "biencoder", "rule_rerank", and optionally "learned_rerank".
        Metrics: "ndcg@k", "recall@k", "mrr" for each k in top_ks.
    """
    if top_ks is None:
        top_ks = [5, 10]

    gold_queries = load_gold(gold_path)
    if not gold_queries:
        print(f"[eval] No gold queries found in {gold_path}.", file=sys.stderr)
        return {}

    if verbose:
        print(f"[eval] Loaded {len(gold_queries)} gold queries from {gold_path}")

    # ── Try to load the KernelStore ───────────────────────────────────────────
    store = None
    if storage_dir:
        storage_path = Path(storage_dir).expanduser()
        if storage_path.exists():
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from core.mirror.store import KernelStore
                store = KernelStore(storage_dir=str(storage_path))
                if verbose:
                    print(f"[eval] Loaded KernelStore from {storage_path}")
            except Exception as exc:
                print(
                    f"[eval] Warning: could not load KernelStore from {storage_path}: {exc}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[eval] Warning: storage path {storage_path} does not exist. "
                "Returning zero metrics.",
                file=sys.stderr,
            )

    if store is None:
        # Return zero metrics for all settings / metrics combinations
        zero: dict[str, dict[str, float]] = {}
        for setting in ("biencoder", "rule_rerank"):
            zero[setting] = {}
            for k in top_ks:
                zero[setting][f"ndcg@{k}"] = 0.0
                zero[setting][f"recall@{k}"] = 0.0
            zero[setting]["mrr"] = 0.0
        return zero

    # ── Optionally load reranker ──────────────────────────────────────────────
    import torch
    reranker = tokenizer_r = device = None
    if reranker_path and Path(reranker_path).exists():
        try:
            from transformers import AutoTokenizer
            from training.models.reranker import KernelReranker
            if verbose:
                print(f"[eval] Loading reranker from {reranker_path}")
            reranker = KernelReranker.load(reranker_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            reranker = reranker.to(device)
            model_name_file = Path(reranker_path) / "model_name.txt"
            model_name = (
                model_name_file.read_text().strip()
                if model_name_file.exists()
                else "microsoft/codebert-base"
            )
            tokenizer_r = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            print(
                f"[eval] Warning: could not load reranker from {reranker_path}: {exc}",
                file=sys.stderr,
            )

    # ── Accumulate per-query metric lists ─────────────────────────────────────
    results: dict[str, dict[str, list[float]]] = {
        "biencoder": {f"ndcg@{k}": [] for k in top_ks},
        "rule_rerank": {f"ndcg@{k}": [] for k in top_ks},
    }
    if reranker:
        results["learned_rerank"] = {f"ndcg@{k}": [] for k in top_ks}
    for setting in results:
        for k in top_ks:
            results[setting][f"recall@{k}"] = []
        results[setting]["mrr"] = []

    for i, gold in enumerate(gold_queries):
        query = gold["query"]
        relevant = set(gold.get("relevant_ids", []))
        if not relevant:
            continue

        # Bi-encoder baseline
        bi_results = _retrieve_biencoder(query, store, top_n=rerank_n)
        bi_ids = [nid for nid, _ in bi_results]
        bi_rels = [1.0 if nid in relevant else 0.0 for nid in bi_ids]

        for k in top_ks:
            results["biencoder"][f"ndcg@{k}"].append(ndcg_at_k(bi_rels, k))
            results["biencoder"][f"recall@{k}"].append(
                recall_at_k(bi_ids, relevant, k)
            )
        results["biencoder"]["mrr"].append(mrr(bi_ids, relevant))

        # Rule-based reranker
        rule_results = _rerank_rule_based(bi_results, store, top_k=max(top_ks))
        rule_ids = [nid for nid, _ in rule_results]
        rule_rels = [1.0 if nid in relevant else 0.0 for nid in rule_ids]

        for k in top_ks:
            results["rule_rerank"][f"ndcg@{k}"].append(ndcg_at_k(rule_rels, k))
            results["rule_rerank"][f"recall@{k}"].append(
                recall_at_k(rule_ids, relevant, k)
            )
        results["rule_rerank"]["mrr"].append(mrr(rule_ids, relevant))

        # Learned reranker (optional)
        if reranker and tokenizer_r and device is not None:
            lr_results = _rerank_learned(
                query,
                bi_results,
                store,
                reranker,
                tokenizer_r,
                device,
                top_k=max(top_ks),
            )
            lr_ids = [nid for nid, _ in lr_results]
            lr_rels = [1.0 if nid in relevant else 0.0 for nid in lr_ids]
            for k in top_ks:
                results["learned_rerank"][f"ndcg@{k}"].append(
                    ndcg_at_k(lr_rels, k)
                )
                results["learned_rerank"][f"recall@{k}"].append(
                    recall_at_k(lr_ids, relevant, k)
                )
            results["learned_rerank"]["mrr"].append(mrr(lr_ids, relevant))

        if verbose and (i + 1) % 20 == 0:
            print(f"[eval] {i+1}/{len(gold_queries)} queries processed", flush=True)

    # ── Average across queries ────────────────────────────────────────────────
    averaged: dict[str, dict[str, float]] = {}
    for setting, metrics in results.items():
        averaged[setting] = {
            metric: (sum(vals) / len(vals) if vals else 0.0)
            for metric, vals in metrics.items()
        }

    return averaged
