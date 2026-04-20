"""
core/mod/store.py — Persistence layer for the Modification module.

Storage layout:
    ~/.kernel-talk/mod/
    ├── proposals/<id>.yaml
    ├── snapshots/<id>/
    │   ├── meta.yaml
    │   └── files/        (copy-on-write: only changed files stored)
    ├── snapshots/index.yaml
    └── pending.yaml      (accepted stack, ordered)
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterator

import yaml

from core.mod.models import Hunk, Proposal, Snapshot


class ModStore:
    """Persistent store for proposals and snapshots."""

    def __init__(self, mod_dir: str | Path) -> None:
        self._root = Path(mod_dir)
        self._proposals_dir = self._root / "proposals"
        self._snapshots_dir = self._root / "snapshots"
        self._pending_file = self._root / "pending.yaml"
        self._index_file = self._snapshots_dir / "index.yaml"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self._proposals_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ── Proposals ─────────────────────────────────────────────────────────────

    def save_proposal(self, proposal: Proposal) -> None:
        path = self._proposals_dir / f"{proposal.id}.yaml"
        with open(path, "w") as f:
            yaml.dump(proposal.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def load_proposal(self, proposal_id: str) -> Proposal:
        path = self._proposals_dir / f"{proposal_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Proposal {proposal_id} not found")
        with open(path) as f:
            return Proposal.from_dict(yaml.safe_load(f))

    def list_proposals(self, state: str | None = None) -> list[Proposal]:
        proposals = []
        for p in sorted(self._proposals_dir.glob("*.yaml")):
            try:
                proposal = self.load_proposal(p.stem)
                if state is None or proposal.state == state:
                    proposals.append(proposal)
            except Exception:
                continue
        return proposals

    def update_proposal_state(self, proposal_id: str, new_state: str) -> Proposal:
        proposal = self.load_proposal(proposal_id)
        proposal.state = new_state  # type: ignore[assignment]
        self.save_proposal(proposal)
        return proposal

    # ── Pending stack ──────────────────────────────────────────────────────────

    def pending_ids(self) -> list[str]:
        """Return accepted-not-yet-applied proposal IDs in stack order."""
        if not self._pending_file.exists():
            return []
        data = yaml.safe_load(self._pending_file.read_text()) or {}
        return data.get("stack", [])

    def push_pending(self, proposal_id: str) -> None:
        stack = self.pending_ids()
        if proposal_id not in stack:
            stack.append(proposal_id)
        with open(self._pending_file, "w") as f:
            yaml.dump({"stack": stack}, f)

    def clear_pending(self) -> None:
        with open(self._pending_file, "w") as f:
            yaml.dump({"stack": []}, f)

    def pending_proposals(self) -> list[Proposal]:
        return [self.load_proposal(pid) for pid in self.pending_ids()]

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def save_snapshot(self, snapshot: Snapshot, changed_files: dict[str, bytes]) -> None:
        """
        Save a snapshot.

        Args:
            snapshot:      The Snapshot metadata object.
            changed_files: {relative_file_path: file_bytes} — only the files
                           that differ from the parent snapshot.
        """
        snap_dir = self._snapshots_dir / snapshot.id
        files_dir = snap_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        # Write changed files (copy-on-write)
        for rel_path, content in changed_files.items():
            dest = files_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        # Write meta.yaml
        with open(snap_dir / "meta.yaml", "w") as f:
            yaml.dump(snapshot.to_dict(), f, default_flow_style=False, allow_unicode=True)

        # Update index
        self._append_snapshot_index(snapshot)

    def load_snapshot(self, snapshot_id: str) -> Snapshot:
        meta = self._snapshots_dir / snapshot_id / "meta.yaml"
        if not meta.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        with open(meta) as f:
            return Snapshot.from_dict(yaml.safe_load(f))

    def snapshot_file(self, snapshot_id: str, rel_path: str) -> bytes | None:
        """
        Retrieve a file from the snapshot chain (walks parents for COW).
        Returns None if the file is not in this snapshot or any ancestor.
        """
        snap = self.load_snapshot(snapshot_id)
        while snap:
            candidate = self._snapshots_dir / snap.id / "files" / rel_path
            if candidate.exists():
                return candidate.read_bytes()
            if snap.parent_snapshot_id:
                snap = self.load_snapshot(snap.parent_snapshot_id)
            else:
                break
        return None

    def list_snapshots(self) -> list[Snapshot]:
        if not self._index_file.exists():
            return []
        data = yaml.safe_load(self._index_file.read_text()) or {}
        return [self.load_snapshot(sid) for sid in data.get("order", [])]

    def latest_snapshot_id(self) -> str | None:
        snapshots = self.list_snapshots()
        return snapshots[-1].id if snapshots else None

    def _append_snapshot_index(self, snapshot: Snapshot) -> None:
        data = {"order": []}
        if self._index_file.exists():
            data = yaml.safe_load(self._index_file.read_text()) or {"order": []}
        order: list[str] = data.get("order", [])
        if snapshot.id not in order:
            order.append(snapshot.id)
        with open(self._index_file, "w") as f:
            yaml.dump({"order": order}, f)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
