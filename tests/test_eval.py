"""
tests/test_eval.py
───────────────────
Tests for the eval/ package (retrieval evaluation metrics and gold data loading).

These tests do NOT require a KernelStore, ChromaDB, torch, or transformers.
They verify the pure-Python metric implementations and the gold data helpers.
"""

import sys
import os
import json
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.retrieval import ndcg_at_k, recall_at_k, mrr, load_gold, run_eval


# ── Metric: nDCG@k ────────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_ranking(self):
        """All relevant docs at top k → nDCG = 1.0."""
        rels = [1.0, 1.0, 0.0, 0.0, 0.0]
        assert ndcg_at_k(rels, k=2) == pytest.approx(1.0)

    def test_all_irrelevant(self):
        """No relevant docs in top k → nDCG = 0.0."""
        rels = [0.0, 0.0, 1.0, 1.0]
        assert ndcg_at_k(rels, k=2) == pytest.approx(0.0)

    def test_single_relevant_at_top(self):
        """One relevant doc at rank 1 with k=5."""
        rels = [1.0, 0.0, 0.0, 0.0, 0.0]
        result = ndcg_at_k(rels, k=5)
        # DCG = 1/log2(2)=1.0; IDCG = same → nDCG = 1.0
        assert result == pytest.approx(1.0)

    def test_relevant_at_rank_2(self):
        """One relevant at rank 2 out of 2 candidates in top k."""
        rels = [0.0, 1.0]
        result = ndcg_at_k(rels, k=2)
        # DCG  = 1/log2(3)
        # IDCG = 1/log2(2) = 1
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert result == pytest.approx(expected)

    def test_k_larger_than_result_list(self):
        """k > len(rels) should not crash."""
        rels = [1.0, 0.0]
        result = ndcg_at_k(rels, k=10)
        assert 0.0 <= result <= 1.0

    def test_empty_relevances(self):
        """Empty relevance list → 0.0."""
        assert ndcg_at_k([], k=5) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self):
        """k=0: no docs ranked → DCG=0, IDCG=0 → 0.0."""
        assert ndcg_at_k([1.0, 1.0], k=0) == pytest.approx(0.0)


# ── Metric: Recall@k ──────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall(self):
        """All relevant IDs in top k."""
        ranked = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, k=2) == pytest.approx(1.0)

    def test_zero_recall(self):
        """No relevant IDs in top k."""
        ranked = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, k=3) == pytest.approx(0.0)

    def test_partial_recall(self):
        """One of two relevant docs in top k."""
        ranked = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(ranked, relevant, k=2) == pytest.approx(0.5)

    def test_k_smaller_than_relevant(self):
        """k < |relevant| — cannot achieve recall=1 by definition."""
        ranked = ["a", "b", "c", "d"]
        relevant = {"a", "b", "c", "d"}
        # Only 2 of 4 relevant in top 2
        assert recall_at_k(ranked, relevant, k=2) == pytest.approx(0.5)

    def test_empty_relevant_set(self):
        """Empty relevant set → 0.0 (avoid division by zero)."""
        assert recall_at_k(["a", "b"], set(), k=2) == pytest.approx(0.0)


# ── Metric: MRR ───────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_rank(self):
        """Relevant at rank 1 → MRR = 1.0."""
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_rank(self):
        """Relevant at rank 2 → MRR = 0.5."""
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_rank(self):
        """Relevant at rank 3 → MRR ≈ 0.333."""
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)

    def test_no_relevant(self):
        """No relevant docs → MRR = 0.0."""
        assert mrr(["x", "y", "z"], {"a"}) == pytest.approx(0.0)

    def test_multiple_relevant_first_wins(self):
        """MRR uses the rank of the FIRST relevant doc found."""
        assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)

    def test_empty_ranked_list(self):
        assert mrr([], {"a"}) == pytest.approx(0.0)


# ── Gold data loading ─────────────────────────────────────────────────────────

class TestLoadGold:
    def test_loads_relevant_ids_directly(self, tmp_path):
        """Entry with explicit relevant_ids is loaded as-is."""
        gold = [{"query": "how does fork work", "relevant_ids": ["kernel/fork.c::kernel_clone"]}]
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(gold[0]) + "\n")

        queries = load_gold(str(gold_file))
        assert len(queries) == 1
        assert queries[0]["relevant_ids"] == ["kernel/fork.c::kernel_clone"]

    def test_derives_relevant_ids_from_legacy_format(self, tmp_path):
        """Entry with expected_symbols + expected_files gets relevant_ids derived."""
        entry = {
            "query": "how does fork work",
            "expected_symbols": ["kernel_clone", "copy_process"],
            "expected_files": ["kernel/fork.c"],
        }
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(entry) + "\n")

        queries = load_gold(str(gold_file))
        assert len(queries) == 1
        rids = set(queries[0]["relevant_ids"])
        assert "kernel/fork.c::kernel_clone" in rids
        assert "kernel/fork.c::copy_process" in rids

    def test_cross_product_file_symbol(self, tmp_path):
        """relevant_ids = cross-product of each file with each symbol."""
        entry = {
            "query": "scheduler",
            "expected_symbols": ["schedule", "pick_next_task"],
            "expected_files": ["kernel/sched/core.c", "kernel/sched/fair.c"],
        }
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(entry) + "\n")

        queries = load_gold(str(gold_file))
        rids = set(queries[0]["relevant_ids"])
        assert "kernel/sched/core.c::schedule" in rids
        assert "kernel/sched/fair.c::schedule" in rids
        assert "kernel/sched/core.c::pick_next_task" in rids
        assert "kernel/sched/fair.c::pick_next_task" in rids
        assert len(rids) == 4

    def test_empty_file(self, tmp_path):
        """Empty gold file → empty list."""
        gold_file = tmp_path / "empty.jsonl"
        gold_file.write_text("")
        assert load_gold(str(gold_file)) == []

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines in the JSONL file are ignored."""
        entry = {"query": "q", "relevant_ids": ["a::b"]}
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(f"\n{json.dumps(entry)}\n\n")
        queries = load_gold(str(gold_file))
        assert len(queries) == 1

    def test_retrieval_gold_file_parseable(self):
        """The actual eval/retrieval_gold.jsonl must parse without errors."""
        gold_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "eval", "retrieval_gold.jsonl",
        )
        queries = load_gold(gold_path)
        assert len(queries) > 0
        for q in queries:
            assert "query" in q
            assert "relevant_ids" in q
            assert isinstance(q["relevant_ids"], list)

    def test_existing_empty_relevant_ids_gets_derived(self, tmp_path):
        """If relevant_ids is present but empty, derive from legacy fields."""
        entry = {
            "query": "q",
            "relevant_ids": [],
            "expected_symbols": ["fn"],
            "expected_files": ["a.c"],
        }
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(entry) + "\n")
        queries = load_gold(str(gold_file))
        assert "a.c::fn" in queries[0]["relevant_ids"]


# ── run_eval with no store ─────────────────────────────────────────────────────

class TestRunEvalNoStore:
    """
    run_eval must return zero metrics gracefully when no KernelStore exists.
    This is the expected behaviour in CI without a kernel index.
    """

    def test_nonexistent_storage_returns_zero_metrics(self, tmp_path):
        gold = [{"query": "how does fork work", "relevant_ids": ["kernel/fork.c::kernel_clone"]}]
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(gold[0]) + "\n")

        results = run_eval(
            gold_path=str(gold_file),
            storage_dir=str(tmp_path / "nonexistent_store"),
            verbose=False,
        )
        assert "biencoder" in results
        assert "rule_rerank" in results
        for setting, metrics in results.items():
            for val in metrics.values():
                assert val == pytest.approx(0.0), \
                    f"{setting}.{val} should be 0 when store is absent"

    def test_none_storage_returns_zero_metrics(self, tmp_path):
        gold = [{"query": "q", "relevant_ids": ["a::b"]}]
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(gold[0]) + "\n")

        results = run_eval(gold_path=str(gold_file), storage_dir=None, verbose=False)
        assert "biencoder" in results
        for val in results["biencoder"].values():
            assert val == pytest.approx(0.0)

    def test_zero_metrics_have_expected_keys(self, tmp_path):
        gold = [{"query": "q", "relevant_ids": ["a::b"]}]
        gold_file = tmp_path / "gold.jsonl"
        gold_file.write_text(json.dumps(gold[0]) + "\n")

        results = run_eval(
            gold_path=str(gold_file),
            storage_dir=None,
            top_ks=[5, 10],
            verbose=False,
        )
        for setting in ("biencoder", "rule_rerank"):
            assert f"ndcg@5" in results[setting]
            assert f"ndcg@10" in results[setting]
            assert f"recall@5" in results[setting]
            assert f"recall@10" in results[setting]
            assert "mrr" in results[setting]

    def test_empty_gold_returns_empty_dict(self, tmp_path):
        gold_file = tmp_path / "empty.jsonl"
        gold_file.write_text("")

        results = run_eval(
            gold_path=str(gold_file),
            storage_dir=None,
            verbose=False,
        )
        assert results == {}
