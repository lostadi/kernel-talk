"""
eval/run.py — CLI entry point for retrieval evaluation.

Usage:
    python -m eval.run \\
        --gold eval/retrieval_gold.jsonl \\
        --storage ~/.kernel-talk/store \\
        --output eval/results.json \\
        [--reranker training/checkpoints/reranker/best] \\
        [--top-k 5 10]

When the storage directory does not exist (e.g., in CI without a real kernel
index), the script returns zero metrics and exits successfully so downstream
workflows are not broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate kernel-talk retrieval quality against a gold set."
    )
    p.add_argument(
        "--gold",
        default="eval/retrieval_gold.jsonl",
        help="Path to gold JSONL file (default: eval/retrieval_gold.jsonl)",
    )
    p.add_argument(
        "--storage",
        default=str(Path.home() / ".kernel-talk" / "store"),
        help="Path to KernelStore directory (default: ~/.kernel-talk/store)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to write results JSON (default: print to stdout only)",
    )
    p.add_argument(
        "--reranker",
        default=None,
        help="Path to learned reranker checkpoint directory (optional)",
    )
    p.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[5, 10],
        metavar="K",
        help="@k values for nDCG and Recall metrics (default: 5 10)",
    )
    p.add_argument(
        "--rerank-n",
        type=int,
        default=100,
        help="Number of bi-encoder candidates to pass to reranker (default: 100)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.retrieval import run_eval

    results = run_eval(
        gold_path=args.gold,
        storage_dir=args.storage,
        reranker_path=args.reranker,
        top_ks=args.top_k,
        rerank_n=args.rerank_n,
        verbose=True,
    )

    # Pretty-print results table
    print("\n=== Retrieval Evaluation Results ===")
    for setting, metrics in results.items():
        print(f"\n  {setting}:")
        for metric, value in sorted(metrics.items()):
            print(f"    {metric:20s} {value:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[eval] Results written to {out_path}")


if __name__ == "__main__":
    main()
