"""
tests/test_mod.py — Unit tests for the Kernel Modification module (core/mod/).

Tests cover: diff parsing, data model round-trips, ModStore persistence,
and basic preview materialization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.mod.diff_parser import parse_unified_diff
from core.mod.models import Hunk, Proposal, Snapshot
from core.mod.store import ModStore
from core.mod.preview import apply_proposals_to_shadow


# ── Fixtures ────────────────────────────────────────────────────────────────────

SAMPLE_DIFF = """\
--- a/kernel/sched/core.c
+++ b/kernel/sched/core.c
@@ -100,7 +100,7 @@ static void enqueue_task(struct rq *rq, struct task_struct *p, int flags)
 {
        if (!(flags & ENQUEUE_NOCLOCK))
                update_rq_clock(rq);
-       if (!(flags & ENQUEUE_RESTORE)) {
+       if (!(flags & ENQUEUE_RESTORE) && rq->nr_running < MAX_PRIO) {
                sched_info_enqueue(rq, p);
        }
 }
"""

MULTI_FILE_DIFF = """\
--- a/include/linux/sched.h
+++ b/include/linux/sched.h
@@ -50,3 +50,4 @@ struct task_struct {
        int pid;
        int tgid;
+       unsigned long extra_field;
 };
--- a/kernel/sched/core.c
+++ b/kernel/sched/core.c
@@ -200,6 +200,7 @@ void schedule(void)
        rcu_note_context_switch(preempt);
        prev = rq->curr;
+       trace_sched_switch(prev, next);
        next = pick_next_task(rq, prev, &rf);
 }
"""


# ── diff_parser tests ───────────────────────────────────────────────────────────

class TestDiffParser:
    def test_parse_single_hunk(self):
        hunks = parse_unified_diff(SAMPLE_DIFF)
        assert len(hunks) == 1
        h = hunks[0]
        assert h.file_path == "kernel/sched/core.c"
        assert len(h.removed_lines) == 1
        assert len(h.added_lines) == 1
        assert "ENQUEUE_RESTORE" in h.removed_lines[0][1]
        assert "MAX_PRIO" in h.added_lines[0][1]

    def test_parse_multi_file_diff(self):
        hunks = parse_unified_diff(MULTI_FILE_DIFF)
        assert len(hunks) == 2
        files = {h.file_path for h in hunks}
        assert "include/linux/sched.h" in files
        assert "kernel/sched/core.c" in files

    def test_parse_empty_diff(self):
        hunks = parse_unified_diff("")
        assert hunks == []

    def test_kernel_root_stripping(self):
        diff = SAMPLE_DIFF.replace(
            "+++ b/kernel/sched/core.c",
            "+++ b//usr/src/linux/kernel/sched/core.c"
        )
        hunks = parse_unified_diff(diff, kernel_root="/usr/src/linux")
        assert hunks[0].file_path == "kernel/sched/core.c"

    def test_hunk_line_numbers(self):
        hunks = parse_unified_diff(SAMPLE_DIFF)
        h = hunks[0]
        # The @@ header says -100, so old line 103 is the changed line
        # (3 context lines before it: lines 100, 101, 102, then 103 is changed)
        removed_lineno = h.removed_lines[0][0]
        assert removed_lineno == 103  # 100 + 3 context lines

    def test_context_lines_captured(self):
        hunks = parse_unified_diff(SAMPLE_DIFF)
        h = hunks[0]
        assert len(h.before_context) > 0
        assert "{" in h.before_context[0] or "enqueue" in "".join(h.before_context)


# ── data model tests ────────────────────────────────────────────────────────────

class TestModels:
    def test_hunk_roundtrip(self):
        h = Hunk(
            file_path="foo/bar.c",
            before_context=["line1"],
            removed_lines=[(10, "old code")],
            added_lines=[(10, "new code")],
            after_context=["line3"],
            affected_nodes=["node-1"],
        )
        d = h.to_dict()
        h2 = Hunk.from_dict(d)
        assert h2.file_path == h.file_path
        assert h2.removed_lines == h.removed_lines
        assert h2.added_lines == h.added_lines
        assert h2.affected_nodes == h.affected_nodes

    def test_proposal_create_and_roundtrip(self):
        hunks = parse_unified_diff(SAMPLE_DIFF)
        p = Proposal.create("Fix scheduler", hunks, author="agent-x")
        assert p.state == "proposed"
        assert p.author == "agent-x"
        assert len(p.hunks) == 1

        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.id == p.id
        assert p2.description == p.description
        assert p2.state == p.state
        assert len(p2.hunks) == 1

    def test_proposal_files_changed(self):
        hunks = parse_unified_diff(MULTI_FILE_DIFF)
        p = Proposal.create("Two files", hunks)
        assert len(p.files_changed) == 2

    def test_snapshot_create_and_roundtrip(self):
        s = Snapshot.create(
            description="Before apply",
            parent_snapshot_id=None,
            applied_proposals=["p1"],
            manifest=[("foo/bar.c", "abc123")],
        )
        assert s.tree_hash  # non-empty sha256
        d = s.to_dict()
        s2 = Snapshot.from_dict(d)
        assert s2.id == s.id
        assert s2.tree_hash == s.tree_hash
        assert s2.manifest == s.manifest


# ── ModStore tests ────────────────────────────────────────────────────────────────

class TestModStore:
    def test_save_and_load_proposal(self, tmp_path):
        store = ModStore(tmp_path / "mod")
        hunks = parse_unified_diff(SAMPLE_DIFF)
        p = Proposal.create("Test", hunks)
        store.save_proposal(p)
        p2 = store.load_proposal(p.id)
        assert p2.id == p.id
        assert p2.state == "proposed"

    def test_update_proposal_state(self, tmp_path):
        store = ModStore(tmp_path / "mod")
        hunks = parse_unified_diff(SAMPLE_DIFF)
        p = Proposal.create("Test", hunks)
        store.save_proposal(p)
        store.update_proposal_state(p.id, "accepted")
        p2 = store.load_proposal(p.id)
        assert p2.state == "accepted"

    def test_pending_stack(self, tmp_path):
        store = ModStore(tmp_path / "mod")
        hunks = parse_unified_diff(SAMPLE_DIFF)
        p = Proposal.create("P1", hunks)
        store.save_proposal(p)
        store.update_proposal_state(p.id, "accepted")
        store.push_pending(p.id)

        assert p.id in store.pending_ids()
        pending = store.pending_proposals()
        assert len(pending) == 1
        assert pending[0].id == p.id

        store.clear_pending()
        assert store.pending_ids() == []

    def test_list_proposals_by_state(self, tmp_path):
        store = ModStore(tmp_path / "mod")
        hunks = parse_unified_diff(SAMPLE_DIFF)
        p1 = Proposal.create("P1", hunks)
        p2 = Proposal.create("P2", hunks)
        p1.state = "accepted"
        store.save_proposal(p1)
        store.save_proposal(p2)

        proposed = store.list_proposals(state="proposed")
        accepted = store.list_proposals(state="accepted")
        assert len(proposed) == 1
        assert proposed[0].id == p2.id
        assert len(accepted) == 1
        assert accepted[0].id == p1.id

    def test_save_and_load_snapshot(self, tmp_path):
        store = ModStore(tmp_path / "mod")
        s = Snapshot.create("snap1", None, [], [("foo.c", "abc123")])
        changed = {"foo.c": b"int main() {}\n"}
        store.save_snapshot(s, changed)

        s2 = store.load_snapshot(s.id)
        assert s2.id == s.id

        content = store.snapshot_file(s.id, "foo.c")
        assert content == b"int main() {}\n"


# ── Preview tests ─────────────────────────────────────────────────────────────

class TestPreview:
    def test_apply_to_shadow(self, tmp_path):
        # Create a fake kernel root with a file
        kernel_root = tmp_path / "kernel"
        kernel_root.mkdir()
        target_file = kernel_root / "kernel" / "sched" / "core.c"
        target_file.parent.mkdir(parents=True)
        # Write 110 lines so line 103 exists
        lines = [f"/* line {i} */\n" for i in range(1, 110)]
        lines[99] = "{\n"          # line 100
        lines[100] = "    if (!(flags & ENQUEUE_NOCLOCK))\n"    # line 101
        lines[101] = "        update_rq_clock(rq);\n"           # line 102
        lines[102] = "    if (!(flags & ENQUEUE_RESTORE)) {\n"  # line 103 — will be removed
        target_file.write_text("".join(lines))

        hunks = parse_unified_diff(SAMPLE_DIFF)
        p = Proposal.create("Sched fix", hunks)

        shadow_root, result = apply_proposals_to_shadow(str(kernel_root), [p])
        try:
            assert "kernel/sched/core.c" in result.files_changed
            shadow_file = Path(shadow_root) / "kernel" / "sched" / "core.c"
            assert shadow_file.exists()
            content = shadow_file.read_text()
            assert "MAX_PRIO" in content, "Added line should appear in shadow"
        finally:
            import shutil
            shutil.rmtree(shadow_root, ignore_errors=True)
