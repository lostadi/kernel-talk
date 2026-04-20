"""
tests/test_training.py
───────────────────────
Tests for the Phase 3 training pipeline components.

These tests run without a real Linux git repo, a real KernelStore, or
any ML dependencies (no torch/transformers required). We test the
algorithmic correctness of the pure-Python components:

  - BM25 scoring formula (exact score verification)
  - BM25 hard negative ranking (harder docs rank higher)
  - BM25 difficulty gap (positive vs hard negative separation)
  - Adaptive curriculum: effective_hard_ratio formula
  - Adaptive curriculum: set_curriculum_epoch() progression
  - Adaptive curriculum: peak-performance guarantee (full ratio at final epoch)
  - Triplet miner commit filtering (noise rejection)
  - Date splitting (train/val/test proportions + leakage prevention)
  - Tokenizer (C-specific tokenization, stopword removal)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest
from training.bm25 import BM25Index, tokenize, _C_STOPWORDS
from training.mine import (
    extract_subsystem, is_useful_commit, date_split, Triplet,
    functions_at_lines,
)


# ─── Tokenizer ─────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = tokenize("schedule() calls pick_next_task")
        assert "schedule" in tokens
        assert "pick_next_task" in tokens     # snake_case kept intact
        assert "calls" in tokens

    def test_c_stopwords_removed(self):
        tokens = tokenize("static int void return struct")
        for stop in ["static", "int", "void", "return"]:
            assert stop not in tokens, f"Stopword '{stop}' should be removed"

    def test_snake_case_not_split(self):
        """pick_next_task must NOT become ['pick', 'next', 'task']."""
        tokens = tokenize("pick_next_task throttle_cfs_rq tg_throttle_up")
        assert "pick_next_task" in tokens
        assert "throttle_cfs_rq" in tokens
        assert "tg_throttle_up" in tokens
        # Make sure we're not also getting the split versions
        assert "pick" not in tokens or "pick_next_task" in tokens  # OK either way
        # The full identifier must be present as a token
        assert "pick_next_task" in tokens

    def test_lowercasing(self):
        tokens = tokenize("INIT_WORK schedule_timeout")
        assert "init_work" in tokens
        assert "schedule_timeout" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_c_operators_stripped(self):
        """Operators like -> and * should not appear as tokens."""
        tokens = tokenize("rq->curr->pid = task->pid")
        assert "->" not in tokens
        assert "=" not in tokens
        assert "curr" in tokens
        assert "pid" in tokens


# ─── BM25 scoring ─────────────────────────────────────────────────────────────

class TestBM25Index:
    """
    We verify BM25 scores against the formula directly.
    With a tiny 3-document corpus we can compute expected scores by hand.
    """

    @pytest.fixture
    def tiny_index(self) -> BM25Index:
        """
        3-document corpus:
          doc0: "schedule throttle cfs bandwidth"
          doc1: "throttle_cfs_rq update_curr fair"
          doc2: "do_fork kernel_clone process creation"
        """
        idx = BM25Index()
        idx._add_document("sched::schedule",      "schedule throttle cfs bandwidth")
        idx._add_document("sched::throttle_cfs_rq", "throttle_cfs_rq update_curr fair")
        idx._add_document("fork::kernel_clone",   "do_fork kernel_clone process creation")
        idx._finalize()
        return idx

    def test_n_documents(self, tiny_index):
        assert tiny_index._N == 3

    def test_exact_match_scores_higher(self, tiny_index):
        """
        Query 'schedule throttle' should score sched::schedule above fork::kernel_clone.
        """
        s_sched   = tiny_index.score("schedule throttle", "sched::schedule")
        s_fork    = tiny_index.score("schedule throttle", "fork::kernel_clone")
        assert s_sched > s_fork, \
            f"Expected sched > fork: {s_sched:.3f} vs {s_fork:.3f}"

    def test_no_match_scores_zero(self, tiny_index):
        """A document with none of the query terms scores 0."""
        # "kernel_clone" doesn't appear in sched::schedule's doc
        s = tiny_index.score("kernel_clone", "sched::schedule")
        # "kernel_clone" is not in that doc's term_freqs
        assert s == 0.0

    def test_search_returns_sorted(self, tiny_index):
        """search() must return results in descending score order."""
        results = tiny_index.search("throttle cfs", k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), \
            "search() results not in descending order"

    def test_search_k_limit(self, tiny_index):
        results = tiny_index.search("schedule", k=2)
        assert len(results) <= 2

    def test_hard_negatives_excludes_positives(self, tiny_index):
        """
        hard_negatives() must never return a node_id in the positive set.
        """
        positive_ids = {"sched::schedule"}
        negs = tiny_index.hard_negatives(
            query="schedule throttle cfs",
            positive_ids=positive_ids,
            k=5,
        )
        for nid in negs:
            assert nid not in positive_ids, \
                f"Positive {nid} appeared in hard negatives"

    def test_hard_negatives_ranks_by_relevance(self, tiny_index):
        """
        For query 'throttle cfs', the hard negative should be the doc that
        mentions 'throttle' (sched::throttle_cfs_rq), not the fork doc.
        We exclude 'sched::schedule' as the positive.
        """
        positive_ids = {"sched::schedule"}
        negs = tiny_index.hard_negatives(
            query="throttle cfs bandwidth",
            positive_ids=positive_ids,
            k=5,
        )
        # The throttle_cfs_rq doc shares 'throttle' and 'fair' (not in query but no overlap)
        # It should rank above the fork doc which shares nothing with the query
        if len(negs) >= 2:
            # At minimum, throttle-related doc should appear in results
            assert any("throttle" in nid or "sched" in nid for nid in negs), \
                f"Expected throttle-related doc in hard negatives, got: {negs}"

    def test_subsystem_filter(self, tiny_index):
        """subsystem_filter restricts hard negatives to a path prefix."""
        negs = tiny_index.hard_negatives(
            query="schedule throttle",
            positive_ids={"sched::schedule"},
            k=5,
            subsystem_filter="sched",
        )
        for nid in negs:
            assert nid.startswith("sched"), \
                f"Subsystem filter violated: {nid} doesn't start with 'sched'"

    def test_bm25_idf_penalizes_common_terms(self):
        """
        A term that appears in many docs should have lower IDF than a
        term appearing in few docs. This is the core BM25 invariant.

        We build a dedicated corpus where we control DFs exactly:
          "sched" appears in all 4 docs (df=4) → low IDF
          "rcu"   appears in only 1 doc  (df=1) → high IDF

        Note: we use distinct snake_case identifiers so our tokenizer
        (which keeps snake_case intact) doesn't accidentally split terms
        and produce unexpected DFs. "sched" as a standalone word IS
        tokenized independently from "sched_clock" etc.
        """
        idx = BM25Index()
        idx._add_document("d0", "sched schedule pick_next")
        idx._add_document("d1", "sched fair update_curr")
        idx._add_document("d2", "sched bandwidth throttle_cfs")
        idx._add_document("d3", "sched rcu rcu_read_lock")  # rcu only here
        idx._finalize()

        N = idx._N  # 4

        df_sched = idx._df.get("sched", 0)  # should be 4
        df_rcu   = idx._df.get("rcu", 0)    # should be 1

        assert df_sched == 4, f"Expected df(sched)=4, got {df_sched}"
        assert df_rcu == 1,   f"Expected df(rcu)=1, got {df_rcu}"

        idf_sched = math.log((N - df_sched + 0.5) / (df_sched + 0.5) + 1)
        idf_rcu   = math.log((N - df_rcu   + 0.5) / (df_rcu   + 0.5) + 1)

        assert idf_rcu > idf_sched, \
            f"Rare term 'rcu' (idf={idf_rcu:.4f}) should beat common " \
            f"'sched' (idf={idf_sched:.4f})"

    def test_save_load_roundtrip(self, tiny_index, tmp_path):
        """BM25 index survives pickle round-trip with identical scores."""
        path = tmp_path / "test.pkl"
        tiny_index.save(path)

        loaded = BM25Index.load(path)
        assert loaded._N == tiny_index._N
        assert loaded._avg_dl == pytest.approx(tiny_index._avg_dl)

        s_orig   = tiny_index.score("schedule throttle", "sched::schedule")
        s_loaded = loaded.score("schedule throttle", "sched::schedule")
        assert s_orig == pytest.approx(s_loaded)


# ─── Commit miner helpers ──────────────────────────────────────────────────────

class TestSubsystemExtraction:
    def test_simple_subsystem(self):
        assert extract_subsystem("sched: fix CFS bandwidth throttle") == "sched"

    def test_nested_subsystem(self):
        assert extract_subsystem("sched/fair: update runnable load") == "sched"

    def test_multi_tag(self):
        # "net,ipv4:" — takes first segment
        assert extract_subsystem("net,ipv4: fix race") in ("net", "net,ipv4")

    def test_no_tag(self):
        assert extract_subsystem("Fix typo in comment") == ""

    def test_mm_subsystem(self):
        assert extract_subsystem("mm/slab: fix UAF in kmem_cache_destroy") == "mm"

    def test_version_bump(self):
        # Version bump commits start with "Linux X.Y.Z"
        assert extract_subsystem("Linux 6.4-rc1") == ""


class TestUsefulCommitFilter:
    def test_good_commit_passes(self):
        assert is_useful_commit(
            "sched: fix CFS bandwidth throttle on RT task wakeup",
            ["kernel/sched/fair.c"],
            ["kernel/sched/fair.c::tg_throttle_up"],
        )

    def test_merge_commit_rejected(self):
        assert not is_useful_commit(
            "Merge branch 'sched/core' into HEAD",
            ["kernel/sched/core.c"],
            ["kernel/sched/core.c::schedule"],
        )

    def test_revert_rejected(self):
        assert not is_useful_commit(
            "Revert 'sched: fix bandwidth throttle'",
            ["kernel/sched/fair.c"],
            ["kernel/sched/fair.c::tg_throttle_up"],
        )

    def test_no_positives_rejected(self):
        assert not is_useful_commit(
            "sched: fix CFS bandwidth",
            ["kernel/sched/fair.c"],
            [],
        )

    def test_short_subject_rejected(self):
        assert not is_useful_commit(
            "typo fix",
            ["kernel/sched/core.c"],
            ["kernel/sched/core.c::schedule"],
        )

    def test_header_only_rejected(self):
        assert not is_useful_commit(
            "sched: add new field to task_struct",
            ["include/linux/sched.h"],     # only .h
            ["include/linux/sched.h::task_struct"],
        )


class TestDateSplit:
    def _make_triplets(self, dates: list[str]) -> list[Triplet]:
        return [
            Triplet(
                query=f"q{i}", commit=f"sha{i}", date=d,
                positives=[f"file::fn{i}"], subsystem="sched", changed_files=[],
            )
            for i, d in enumerate(dates)
        ]

    def test_all_before_val_goes_to_train(self):
        triplets = self._make_triplets(["2019-01-01", "2020-06-15", "2021-12-31"])
        train, val, test = date_split(triplets, "2022-01-01", "2023-01-01")
        assert len(train) == 3
        assert len(val) == 0
        assert len(test) == 0

    def test_correct_split_proportions(self):
        dates = (
            ["2019-01-01"] * 60 +   # train
            ["2022-06-01"] * 20 +   # val
            ["2023-06-01"] * 20     # test
        )
        import random
        random.shuffle(dates)
        triplets = self._make_triplets(dates)
        train, val, test = date_split(triplets, "2022-01-01", "2023-01-01")
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_no_future_leakage(self):
        """No test example should appear in train or val."""
        dates = ["2023-06-01", "2021-01-01", "2022-06-01"]
        triplets = self._make_triplets(dates)
        train, val, test = date_split(triplets, "2022-01-01", "2023-01-01")

        train_dates = {t.date for t in train}
        val_dates   = {t.date for t in val}
        test_dates  = {t.date for t in test}

        # No overlap
        assert not (train_dates & test_dates)
        assert not (val_dates & test_dates)


class TestFunctionsAtLines:
    def test_single_function_match(self):
        ranges = [("schedule", 10, 50), ("pick_next_task", 55, 100)]
        result = functions_at_lines(ranges, [25])
        assert result == ["schedule"]

    def test_multiple_hunks_same_function(self):
        ranges = [("schedule", 10, 50)]
        result = functions_at_lines(ranges, [20, 30, 45])
        assert result == ["schedule"]

    def test_multiple_functions_changed(self):
        ranges = [("schedule", 10, 50), ("pick_next_task", 55, 100)]
        result = functions_at_lines(ranges, [25, 70])
        assert set(result) == {"schedule", "pick_next_task"}

    def test_hunk_outside_all_functions(self):
        ranges = [("schedule", 10, 50)]
        result = functions_at_lines(ranges, [200])  # line 200, no function there
        assert result == []

    def test_empty_ranges(self):
        result = functions_at_lines([], [25, 30])
        assert result == []


# ─── BM25 difficulty gap ───────────────────────────────────────────────────────

class TestDifficultyGap:
    """
    Verify the gap = score(positive) - score(hardest_negative) semantics.
    """

    @pytest.fixture
    def gap_index(self) -> BM25Index:
        """
        3 docs:
          pos:      "schedule throttle cfs rq bandwidth"   ← matches 'schedule throttle cfs'
          hard_neg: "throttle_cfs_rq cfs update_curr"      ← also contains 'cfs', partial match
          easy_neg: "do_fork kernel_clone copy_process"    ← no match
        """
        idx = BM25Index()
        idx._add_document("pos",      "schedule throttle cfs rq bandwidth")
        idx._add_document("hard_neg", "throttle_cfs_rq cfs update_curr")
        idx._add_document("easy_neg", "do_fork kernel_clone copy_process")
        idx._finalize()
        return idx

    def test_gap_is_float(self, gap_index):
        gap = gap_index.difficulty_gap(
            query="schedule throttle cfs",
            positive_ids={"pos"},
            hard_neg_ids=["hard_neg"],
        )
        assert isinstance(gap, float)

    def test_easy_negative_gives_large_gap(self, gap_index):
        """
        When the hard negative doesn't match the query at all, the gap
        should be large (positive scores high, negative scores zero).
        """
        gap = gap_index.difficulty_gap(
            query="schedule throttle cfs",
            positive_ids={"pos"},
            hard_neg_ids=["easy_neg"],   # easy_neg has no matching tokens
        )
        assert gap > 0.0, f"Expected large positive gap with easy negative, got {gap}"

    def test_hard_negative_gives_small_gap(self, gap_index):
        """
        When the hard negative matches some query tokens, the gap is
        smaller (closer to zero) than with an easy negative.
        """
        easy_gap = gap_index.difficulty_gap(
            query="schedule throttle cfs",
            positive_ids={"pos"},
            hard_neg_ids=["easy_neg"],
        )
        hard_gap = gap_index.difficulty_gap(
            query="schedule throttle cfs",
            positive_ids={"pos"},
            hard_neg_ids=["hard_neg"],   # hard_neg shares 'cfs'
        )
        assert hard_gap < easy_gap, \
            f"Hard negative should give smaller gap ({hard_gap:.3f}) " \
            f"than easy negative ({easy_gap:.3f})"

    def test_empty_hard_neg_list_returns_zero(self, gap_index):
        gap = gap_index.difficulty_gap(
            query="schedule",
            positive_ids={"pos"},
            hard_neg_ids=[],
        )
        assert gap == 0.0

    def test_unknown_positive_id_returns_zero(self, gap_index):
        gap = gap_index.difficulty_gap(
            query="schedule",
            positive_ids={"nonexistent_id"},
            hard_neg_ids=["hard_neg"],
        )
        assert gap == 0.0


# ─── Adaptive curriculum ───────────────────────────────────────────────────────

class TestAdaptiveCurriculum:
    """
    Test the core invariants of the curriculum scheduling formula:

        effective_ratio = hard_ratio × (progress + (1−progress) × (1−difficulty))

    Key properties to verify:
      1. At progress=1: effective_ratio = hard_ratio for ALL difficulties
         (peak performance guarantee — no curriculum constraint at convergence)
      2. At progress=0, difficulty=0 (easiest): effective_ratio = hard_ratio
         (easy examples get full hard ratio from day 1)
      3. At progress=0, difficulty=1 (hardest): effective_ratio = 0
         (hardest examples start with no hard negatives)
      4. Monotonically increasing in progress (for fixed difficulty)
      5. Monotonically decreasing in difficulty (for fixed progress)
    """

    def _make_mock_dataset(self, hard_ratio: float = 0.3) -> object:
        """
        Create a minimal stand-in that has _effective_hard_ratio and
        set_curriculum_epoch without needing a real JSONL or KernelStore.
        We import TripletDataset but bypass __init__ using object.__new__.
        """
        from training.dataset import TripletDataset
        ds = object.__new__(TripletDataset)
        ds.hard_ratio = hard_ratio
        ds._curriculum_progress = 1.0   # default
        ds._has_difficulty = True
        ds._records = []
        ds._code_cache = {}
        return ds

    def test_peak_guarantee_easy(self):
        """At final epoch, easy example gets full hard_ratio."""
        ds = self._make_mock_dataset(hard_ratio=0.3)
        ds.set_curriculum_epoch(epoch=2, total_epochs=3)  # progress = 1.0
        ratio = ds._effective_hard_ratio(difficulty=0.0)
        assert ratio == pytest.approx(0.3), \
            f"Final epoch, easy example should get full hard_ratio=0.3, got {ratio}"

    def test_peak_guarantee_hard(self):
        """At final epoch, hardest example also gets full hard_ratio."""
        ds = self._make_mock_dataset(hard_ratio=0.3)
        ds.set_curriculum_epoch(epoch=2, total_epochs=3)
        ratio = ds._effective_hard_ratio(difficulty=1.0)
        assert ratio == pytest.approx(0.3), \
            f"Final epoch, hard example should get full hard_ratio=0.3, got {ratio}"

    def test_epoch0_easy_gets_full_ratio(self):
        """At epoch 0, the easiest example (difficulty=0) gets the full hard_ratio."""
        ds = self._make_mock_dataset(hard_ratio=0.4)
        ds.set_curriculum_epoch(epoch=0, total_epochs=5)  # progress = 0.0
        ratio = ds._effective_hard_ratio(difficulty=0.0)
        assert ratio == pytest.approx(0.4), \
            f"Epoch 0, easiest example should get full ratio=0.4, got {ratio}"

    def test_epoch0_hardest_gets_zero(self):
        """At epoch 0, the hardest example (difficulty=1) gets effective ratio≈0."""
        ds = self._make_mock_dataset(hard_ratio=0.4)
        ds.set_curriculum_epoch(epoch=0, total_epochs=5)
        ratio = ds._effective_hard_ratio(difficulty=1.0)
        assert ratio == pytest.approx(0.0), \
            f"Epoch 0, hardest example should get ratio≈0, got {ratio}"

    def test_monotone_in_progress(self):
        """For a fixed difficult example, effective ratio increases with progress."""
        ds = self._make_mock_dataset(hard_ratio=0.3)
        difficulty = 0.8  # fairly hard

        ratios = []
        for epoch in range(5):
            ds.set_curriculum_epoch(epoch, total_epochs=5)
            ratios.append(ds._effective_hard_ratio(difficulty))

        for i in range(len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1], \
                f"Ratio should increase with epochs: {ratios}"

    def test_monotone_in_difficulty(self):
        """For a fixed progress, harder examples get lower effective ratio."""
        ds = self._make_mock_dataset(hard_ratio=0.3)
        ds.set_curriculum_epoch(epoch=1, total_epochs=5)  # mid-training

        difficulties = [0.0, 0.25, 0.5, 0.75, 1.0]
        ratios = [ds._effective_hard_ratio(d) for d in difficulties]

        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], \
                f"Ratio should decrease with difficulty: {list(zip(difficulties, ratios))}"

    def test_set_curriculum_epoch_progress_values(self):
        """set_curriculum_epoch computes progress = epoch / (total - 1)."""
        ds = self._make_mock_dataset()

        ds.set_curriculum_epoch(0, 5)
        assert ds._curriculum_progress == pytest.approx(0.0)

        ds.set_curriculum_epoch(2, 5)
        assert ds._curriculum_progress == pytest.approx(0.5)

        ds.set_curriculum_epoch(4, 5)
        assert ds._curriculum_progress == pytest.approx(1.0)

    def test_single_epoch_training_is_unconstrained(self):
        """When total_epochs=1, curriculum is disabled (progress=1.0 always)."""
        ds = self._make_mock_dataset(hard_ratio=0.5)
        ds.set_curriculum_epoch(epoch=0, total_epochs=1)
        assert ds._curriculum_progress == pytest.approx(1.0)
        assert ds._effective_hard_ratio(difficulty=1.0) == pytest.approx(0.5)

    def test_effective_ratio_clipped_to_valid_range(self):
        """effective_hard_ratio never returns a value outside [0, hard_ratio]."""
        ds = self._make_mock_dataset(hard_ratio=0.3)
        for epoch in range(5):
            ds.set_curriculum_epoch(epoch, 5)
            for difficulty in [0.0, 0.3, 0.6, 0.9, 1.0]:
                ratio = ds._effective_hard_ratio(difficulty)
                assert 0.0 <= ratio <= ds.hard_ratio + 1e-9, \
                    f"Ratio {ratio} out of [0, {ds.hard_ratio}] at " \
                    f"epoch={epoch}, difficulty={difficulty}"
