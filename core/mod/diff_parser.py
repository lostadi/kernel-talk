"""
core/mod/diff_parser.py — Parse unified diffs into Hunk objects.

Reads a unified diff (from a file, string, or patch tool output) and
produces a list of Hunk objects that the Proposal system can store.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.mod.models import Hunk

# Regex for unified diff file header: "+++ b/include/linux/sched.h"
_FILE_HEADER_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")

# Regex for hunk header: "@@ -123,10 +125,12 @@ some_function"
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(text: str, kernel_root: str | None = None) -> list[Hunk]:
    """
    Parse a unified diff string into a list of Hunk objects.

    Args:
        text:        Raw unified diff text.
        kernel_root: If given, strip this prefix from file paths so that
                     Hunk.file_path is always relative to the kernel root.

    Returns:
        A flat list of Hunk objects, one per @@ block across all files.
    """
    hunks: list[Hunk] = []
    current_file: str | None = None
    current_hunk: _HunkBuilder | None = None

    for line in text.splitlines(keepends=False):
        # File header
        m = _FILE_HEADER_RE.match(line)
        if m:
            if current_hunk:
                hunks.append(current_hunk.build())
                current_hunk = None
            current_file = m.group(1)
            if kernel_root:
                current_file = current_file.removeprefix(kernel_root).lstrip("/")
            continue

        # Hunk header
        m = _HUNK_HEADER_RE.match(line)
        if m and current_file:
            if current_hunk:
                hunks.append(current_hunk.build())
            old_start = int(m.group(1))
            new_start = int(m.group(3))
            current_hunk = _HunkBuilder(current_file, old_start, new_start)
            continue

        if current_hunk is None:
            continue

        if line.startswith("-") and not line.startswith("---"):
            current_hunk.add_removed(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            current_hunk.add_added(line[1:])
        elif line.startswith(" "):
            current_hunk.add_context(line[1:])
        # other lines (\ No newline at end of file, etc.) are skipped

    if current_hunk:
        hunks.append(current_hunk.build())

    return hunks


def parse_diff_file(path: str | Path, kernel_root: str | None = None) -> list[Hunk]:
    """Parse a .diff or .patch file into Hunk objects."""
    return parse_unified_diff(Path(path).read_text(errors="replace"), kernel_root)


class _HunkBuilder:
    """Internal builder for accumulating diff lines into a Hunk."""

    def __init__(self, file_path: str, old_start: int, new_start: int) -> None:
        self.file_path = file_path
        self.old_line = old_start
        self.new_line = new_start
        self._before_context: list[str] = []
        self._removed: list[tuple[int, str]] = []
        self._added: list[tuple[int, str]] = []
        self._after_context: list[str] = []
        self._in_change = False  # True once we've seen a + or - line

    def add_context(self, text: str) -> None:
        if self._in_change:
            self._after_context.append(text)
        else:
            self._before_context.append(text)
        self.old_line += 1
        self.new_line += 1

    def add_removed(self, text: str) -> None:
        self._in_change = True
        self._removed.append((self.old_line, text))
        self.old_line += 1

    def add_added(self, text: str) -> None:
        self._in_change = True
        self._added.append((self.new_line, text))
        self.new_line += 1

    def build(self) -> Hunk:
        return Hunk(
            file_path=self.file_path,
            before_context=self._before_context,
            removed_lines=self._removed,
            added_lines=self._added,
            after_context=self._after_context,
        )
