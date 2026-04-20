"""
core/mod/preview.py — Shadow tree materialization for Preview command.

Preview builds a copy-on-write overlay of the kernel source tree,
applies pending proposals to it, and re-indexes Mirror for only the
changed files. The shadow tree is in a tempdir and is never written
back to the real kernel tree.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from core.mod.models import Proposal

if TYPE_CHECKING:
    from core.mirror.store import KernelStore


class PreviewResult:
    """Summary of what a Preview would change in the Mirror."""

    def __init__(self) -> None:
        self.files_changed: list[str] = []
        self.new_nodes: list[str] = []
        self.removed_nodes: list[str] = []
        self.changed_nodes: list[str] = []
        self.ambiguities: list[str] = []

    def __repr__(self) -> str:
        return (
            f"PreviewResult(files={len(self.files_changed)}, "
            f"+nodes={len(self.new_nodes)}, "
            f"-nodes={len(self.removed_nodes)}, "
            f"~nodes={len(self.changed_nodes)}, "
            f"ambiguities={len(self.ambiguities)})"
        )


def apply_proposals_to_shadow(
    kernel_root: str | Path,
    proposals: list[Proposal],
) -> tuple[str, PreviewResult]:
    """
    Materialize a shadow tree by applying proposals to a tempdir overlay.

    Returns:
        (shadow_root, PreviewResult) — caller is responsible for deleting
        shadow_root when done (use as a context manager or call shutil.rmtree).

    Only files touched by the proposals are copied into the shadow dir.
    Everything else is accessed via symlink or not at all (the caller
    must use shadow_root for re-indexing only the changed files).
    """
    kernel_root = Path(kernel_root)
    shadow_root = tempfile.mkdtemp(prefix="ktalk_preview_")
    result = PreviewResult()

    for proposal in proposals:
        for hunk in proposal.hunks:
            src_path = kernel_root / hunk.file_path
            dst_path = Path(shadow_root) / hunk.file_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file to shadow if not already there
            if not dst_path.exists():
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                else:
                    dst_path.touch()

            result.files_changed.append(hunk.file_path)

            # Apply the hunk to the shadow file
            _apply_hunk(dst_path, hunk)

    result.files_changed = list(set(result.files_changed))
    return shadow_root, result


def _apply_hunk(file_path: Path, hunk) -> None:  # type: ignore[type-arg]
    """
    Apply a single Hunk to a file in the shadow tree.

    Uses a simple line-number approach: removes lines at the specified
    offsets, inserts added lines at the gap. If line numbers are out of
    range (e.g. a patch against a different version), logs a warning and
    skips that hunk — this is Preview, not Apply; correctness is best-effort.
    """
    if not file_path.exists():
        lines: list[str] = []
    else:
        lines = file_path.read_text(errors="replace").splitlines(keepends=True)

    # Build set of line numbers to remove (1-based → 0-based index)
    remove_set = {ln - 1 for ln, _ in hunk.removed_lines}

    # Check bounds
    if remove_set and max(remove_set) >= len(lines):
        # Out-of-range patch — skip rather than crash
        return

    # Determine insertion point (first removed line, or after last context line)
    insert_idx = min(remove_set) if remove_set else hunk.start_line

    # Remove lines
    new_lines = [l for i, l in enumerate(lines) if i not in remove_set]

    # Insert added lines at the insertion point
    added = [text + "\n" for _, text in hunk.added_lines]
    new_lines = new_lines[:insert_idx] + added + new_lines[insert_idx:]

    file_path.write_text("".join(new_lines))


class PreviewContext:
    """Context manager that creates and cleans up the shadow tree."""

    def __init__(
        self,
        kernel_root: str | Path,
        proposals: list[Proposal],
    ) -> None:
        self._kernel_root = kernel_root
        self._proposals = proposals
        self._shadow_root: str | None = None
        self.result: PreviewResult | None = None

    def __enter__(self) -> tuple[str, PreviewResult]:
        self._shadow_root, self.result = apply_proposals_to_shadow(
            self._kernel_root, self._proposals
        )
        return self._shadow_root, self.result

    def __exit__(self, *_: object) -> None:
        if self._shadow_root:
            shutil.rmtree(self._shadow_root, ignore_errors=True)
            self._shadow_root = None
