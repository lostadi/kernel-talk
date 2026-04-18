"""
core/mod/models.py — Data model for the Kernel Modification module.

Proposal → Hunk → Snapshot forms a chain that tracks all proposed,
accepted, and applied kernel source changes without touching kernel memory.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


ProposalState = Literal["proposed", "accepted", "discarded", "applied"]


@dataclass
class Hunk:
    """One contiguous change block within a file (derived from a unified diff hunk)."""

    file_path: str
    """File path relative to the kernel source root."""

    before_context: list[str]
    """Lines of context before the change (from the diff's @@ header context)."""

    removed_lines: list[tuple[int, str]]
    """(line_number, text) pairs for lines being removed."""

    added_lines: list[tuple[int, str]]
    """(line_number, text) pairs for lines being added."""

    after_context: list[str]
    """Lines of context after the change."""

    affected_nodes: list[str] = field(default_factory=list)
    """Mirror node IDs whose line ranges overlap with this hunk."""

    @property
    def start_line(self) -> int:
        """First line number touched by this hunk (1-based, or 0 if no removals)."""
        if self.removed_lines:
            return self.removed_lines[0][0]
        if self.added_lines:
            return self.added_lines[0][0]
        return 0

    @property
    def end_line(self) -> int:
        """Last line number touched by this hunk (1-based)."""
        if self.removed_lines:
            return self.removed_lines[-1][0]
        if self.added_lines:
            return self.added_lines[-1][0]
        return 0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "before_context": self.before_context,
            "removed_lines": [[ln, text] for ln, text in self.removed_lines],
            "added_lines": [[ln, text] for ln, text in self.added_lines],
            "after_context": self.after_context,
            "affected_nodes": self.affected_nodes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hunk":
        return cls(
            file_path=d["file_path"],
            before_context=d.get("before_context", []),
            removed_lines=[(ln, text) for ln, text in d.get("removed_lines", [])],
            added_lines=[(ln, text) for ln, text in d.get("added_lines", [])],
            after_context=d.get("after_context", []),
            affected_nodes=d.get("affected_nodes", []),
        )


@dataclass
class Proposal:
    """
    A set of proposed changes to the kernel source tree.

    Proposals pass through states: proposed → accepted → applied.
    A discarded proposal is never removed from disk — it is preserved
    for audit and potential reconsideration.
    """

    id: str
    description: str
    author: str
    base_snapshot_id: str | None
    hunks: list[Hunk]
    state: ProposalState
    created_at: str  # ISO-8601

    @classmethod
    def create(
        cls,
        description: str,
        hunks: list[Hunk],
        author: str = "user",
        base_snapshot_id: str | None = None,
    ) -> "Proposal":
        return cls(
            id=str(uuid.uuid4()),
            description=description,
            author=author,
            base_snapshot_id=base_snapshot_id,
            hunks=hunks,
            state="proposed",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "author": self.author,
            "base_snapshot_id": self.base_snapshot_id,
            "hunks": [h.to_dict() for h in self.hunks],
            "state": self.state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        return cls(
            id=d["id"],
            description=d["description"],
            author=d.get("author", "user"),
            base_snapshot_id=d.get("base_snapshot_id"),
            hunks=[Hunk.from_dict(h) for h in d.get("hunks", [])],
            state=d["state"],
            created_at=d["created_at"],
        )

    @property
    def files_changed(self) -> set[str]:
        return {h.file_path for h in self.hunks}


@dataclass
class Snapshot:
    """
    A point-in-time record of the kernel source tree state.

    Snapshots use copy-on-write at file level: only files changed from
    the parent snapshot are stored. The full tree is recovered by walking
    the parent chain and overlaying.
    """

    id: str
    timestamp: str          # ISO-8601
    description: str
    parent_snapshot_id: str | None
    applied_proposals: list[str]    # list of proposal IDs
    tree_hash: str                  # sha256 of the tree (from manifest)
    manifest: list[tuple[str, str]] # (file_path, sha256) for changed files only

    @classmethod
    def create(
        cls,
        description: str,
        parent_snapshot_id: str | None,
        applied_proposals: list[str],
        manifest: list[tuple[str, str]],
    ) -> "Snapshot":
        tree_hash = hashlib.sha256(
            "\n".join(f"{p} {h}" for p, h in sorted(manifest)).encode()
        ).hexdigest()
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            parent_snapshot_id=parent_snapshot_id,
            applied_proposals=applied_proposals,
            tree_hash=tree_hash,
            manifest=manifest,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "description": self.description,
            "parent_snapshot_id": self.parent_snapshot_id,
            "applied_proposals": self.applied_proposals,
            "tree_hash": self.tree_hash,
            "manifest": [[p, h] for p, h in self.manifest],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(
            id=d["id"],
            timestamp=d["timestamp"],
            description=d["description"],
            parent_snapshot_id=d.get("parent_snapshot_id"),
            applied_proposals=d.get("applied_proposals", []),
            tree_hash=d["tree_hash"],
            manifest=[(p, h) for p, h in d.get("manifest", [])],
        )
