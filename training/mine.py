"""
training/mine.py
─────────────────
Triplet mining from Linux kernel git history.

Every kernel commit is a free supervision signal:

  commit message  →  query  (what the developer wanted to achieve)
  changed code    →  positive  (the functions they actually modified)
  similar code    →  negatives  (functions not modified but lexically close)

This is the foundational assumption of the whole training pipeline. It holds
for the Linux kernel unusually well because:

  1. Kernel commit messages are tightly scoped ("sched: fix CFS bandwidth
     throttle on RT task wakeup") and describe exactly one change.
  2. Kernel developers follow strict conventions — the subsystem tag
     (sched:, mm:, net:) alone narrows the relevant function set by ~99%.
  3. The git log goes back to 2005 and contains ~80K commits, all with
     structured messages. This is larger than most NLP annotation projects.

The key technical challenge is step 2: mapping diff line numbers back to
function names. A diff says "lines 340–352 changed in kernel/sched/fair.c".
We need to know which function that corresponds to. We do this by:
  a) Fetching the file at the parent commit via `git show PARENT:file`
  b) Running tree-sitter to find all function start/end lines
  c) Looking up which function's range contains the diff hunk start line

The output is a JSONL file where each line is:
{
  "query":          "sched: fix CFS bandwidth throttle on RT task wakeup",
  "commit":         "abc123...",
  "date":           "2023-04-15",
  "positives":      ["kernel/sched/fair.c::tg_throttle_up", ...],
  "subsystem":      "kernel/sched",
  "changed_files":  ["kernel/sched/fair.c"],
}

Negatives are NOT mined here — that's bm25.py's job, which runs over the
whole corpus. Separation of concerns: mining finds positives; BM25 finds
hard negatives given those positives.

Usage:
    python -m training.mine \\
        --kernel /path/to/linux \\
        --output data/triplets.jsonl \\
        --since 2018-01-01 \\
        --min-functions 1 \\
        --max-functions 8
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Triplet:
    """
    A single training example mined from git history.

    'positives' are CodeNode IDs in the form "file_path::function_name".
    We use IDs rather than storing the actual code so that the training
    script can look up the current text from the Mirror index — which may
    have been updated since mining.
    """
    query: str              # commit subject line (the natural language query)
    commit: str             # full commit SHA
    date: str               # ISO date "YYYY-MM-DD"
    positives: list[str]    # CodeNode-style IDs: ["kernel/sched/fair.c::tg_throttle_up"]
    subsystem: str          # tag extracted from message, e.g. "sched"
    changed_files: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Triplet":
        return cls(**json.loads(s))


# ─── Git helpers ──────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path) -> str:
    """Run a git command and return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout


def iter_commits(
    kernel_root: Path,
    since: str = "2018-01-01",
    paths: list[str] | None = None,
) -> Iterator[dict]:
    """
    Yield commit metadata dicts for all commits since `since` that touch
    .c or .h files.

    Format yielded:
        {"sha": str, "date": str, "subject": str, "files": [str]}
    """
    git_paths = paths or ["--", "*.c", "*.h"]
    if "--" not in git_paths:
        git_paths = ["--"] + git_paths

    # --name-only gives us changed filenames, one per line after the commit header.
    # The separator "---COMMIT---" lets us split the stream cleanly.
    log_output = _run(
        ["git", "log",
         f"--since={since}",
         "--pretty=format:---COMMIT--- %H %ai %s",
         "--name-only",
         "--diff-filter=M",   # Modified files only (not renames/deletes)
         ] + git_paths,
        cwd=kernel_root,
    )

    current: dict | None = None
    for line in log_output.splitlines():
        if line.startswith("---COMMIT--- "):
            if current and current.get("files"):
                yield current
            parts = line[len("---COMMIT--- "):].split(" ", 2)
            if len(parts) < 3:
                current = None
                continue
            sha, date_str, subject = parts[0], parts[1][:10], parts[2]
            current = {
                "sha":     sha,
                "date":    date_str,
                "subject": subject.strip(),
                "files":   [],
            }
        elif current is not None and line.strip().endswith((".c", ".h")):
            current["files"].append(line.strip())

    if current and current.get("files"):
        yield current


def get_diff_hunks(kernel_root: Path, sha: str, file_path: str) -> list[int]:
    """
    Return the new-file line numbers where this commit introduces changes
    in `file_path`. We take the first line of each diff hunk — that's
    sufficient to identify which function was modified.

    Returns a list of (approximately) one line number per changed hunk.
    """
    try:
        diff = _run(
            ["git", "diff", f"{sha}^..{sha}", "--", file_path],
            cwd=kernel_root,
        )
    except RuntimeError:
        return []

    # Match @@ -old_start,old_count +new_start,new_count @@ [function_name]
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", re.MULTILINE)
    return [int(m.group(1)) for m in hunk_re.finditer(diff)]


def get_function_ranges(kernel_root: Path, sha: str, file_path: str) -> list[tuple[str, int, int]]:
    """
    Return list of (function_name, start_line, end_line) for all functions
    in `file_path` at commit `sha`, using tree-sitter.

    Falls back to a regex heuristic if tree-sitter isn't available.
    """
    try:
        source = _run(
            ["git", "show", f"{sha}:{file_path}"],
            cwd=kernel_root,
        )
    except RuntimeError:
        return []

    try:
        return _tree_sitter_functions(source)
    except Exception:
        return _regex_functions(source)


def _tree_sitter_functions(source: str) -> list[tuple[str, int, int]]:
    """
    Use tree-sitter (if installed) to extract function definitions with
    accurate start/end lines. This is the same logic used by KernelParser.
    """
    import tree_sitter_c as tsc
    from tree_sitter import Language, Parser

    C_LANGUAGE = Language(tsc.language())
    parser = Parser(C_LANGUAGE)
    tree = parser.parse(source.encode())
    root = tree.root_node

    results = []

    def walk(node):
        if node.type == "function_definition":
            # Find the declarator child to get the function name
            for child in node.children:
                if child.type in ("function_declarator", "pointer_declarator"):
                    name = _extract_fn_name(child)
                    if name:
                        results.append((
                            name,
                            node.start_point[0] + 1,  # 1-indexed
                            node.end_point[0] + 1,
                        ))
                    break
        for child in node.children:
            walk(child)

    walk(root)
    return results


def _extract_fn_name(node) -> str | None:
    """Recursively find the identifier in a declarator node."""
    if node.type == "identifier":
        return node.text.decode("utf-8", errors="replace")
    for child in node.children:
        result = _extract_fn_name(child)
        if result:
            return result
    return None


def _regex_functions(source: str) -> list[tuple[str, int, int]]:
    """
    Fallback: regex-based function detection for when tree-sitter is
    unavailable. Less accurate (misses macro-generated functions, gets
    confused by function pointers), but sufficient for mining.

    Matches patterns like:
        static int schedule_timeout(long timeout)
        void __sched schedule(void)
    """
    # Match function definitions: return type + name + ( at start of line
    fn_start_re = re.compile(
        r"^(?:(?:static|inline|__always_inline|__init|__exit|noinline|"
        r"asmlinkage|notrace|__visible|__cold|__latent_entropy)\s+)*"
        r"(?:[\w\s\*]+?\s+)?"               # return type (rough)
        r"(\w+)\s*\([^;]*$",                # function name + ( without ; (not a declaration)
        re.MULTILINE,
    )

    lines = source.splitlines()
    results = []
    open_braces = 0
    in_function = False
    current_name = ""
    current_start = 0

    for i, line in enumerate(lines, 1):
        if not in_function:
            m = fn_start_re.match(line)
            if m and "{" in line:
                current_name = m.group(1)
                current_start = i
                open_braces = line.count("{") - line.count("}")
                if open_braces > 0:
                    in_function = True
        else:
            open_braces += line.count("{") - line.count("}")
            if open_braces <= 0:
                results.append((current_name, current_start, i))
                in_function = False
                open_braces = 0

    return results


def functions_at_lines(
    ranges: list[tuple[str, int, int]],
    hunk_lines: list[int],
) -> list[str]:
    """
    Given function ranges and a list of changed line numbers, return the
    names of functions that contain at least one of those lines.
    """
    changed = set()
    for line in hunk_lines:
        for name, start, end in ranges:
            if start <= line <= end:
                changed.add(name)
                break
    return sorted(changed)


# ─── Subsystem extraction ─────────────────────────────────────────────────────

# Kernel commit messages follow "subsystem: description" convention
_SUBSYSTEM_RE = re.compile(r"^([\w/,\-]+(?:/[\w/,\-]+)?):\s")

def extract_subsystem(subject: str) -> str:
    """
    Extract the subsystem tag from a commit subject.
    "sched/fair: fix ..." → "sched"
    "mm/slab: ..."        → "mm"
    Returns empty string if no tag found.
    """
    m = _SUBSYSTEM_RE.match(subject)
    if not m:
        return ""
    tag = m.group(1).split("/")[0].split(",")[0].strip()
    return tag.lower()


# ─── Quality filters ──────────────────────────────────────────────────────────

# Commits that are almost certainly noise: merges, reverts, version bumps
_NOISE_RE = re.compile(
    r"^(Merge |Revert |Linux \d|MAINTAINERS|Signed-off|fixup!|"
    r"CHROMIUM:|ANDROID:|[Bb]ump version)",
    re.IGNORECASE,
)

def is_useful_commit(subject: str, files: list[str], positives: list[str]) -> bool:
    """
    Filter out commits unlikely to produce useful training signal:
      - Merge commits, reverts, version bumps
      - Commits touching only headers (hard to link to functions)
      - Commits with zero identified changed functions
      - Subjects that are too short to be a useful query
    """
    if _NOISE_RE.match(subject):
        return False
    if len(subject) < 20:
        return False
    if not positives:
        return False
    # If ALL changed files are headers (.h), the training signal is weak
    if all(f.endswith(".h") for f in files):
        return False
    return True


# ─── Miner ────────────────────────────────────────────────────────────────────

class TripletMiner:
    """
    Mines (query, positives) pairs from Linux kernel git history.

    Negatives are intentionally not mined here — they require a full corpus
    index to be useful (BM25 over all indexed functions). bm25.py handles that.

    Args:
        kernel_root:   Path to a Linux kernel git repository
        min_functions: Minimum number of changed functions to include a commit
        max_functions: Maximum number of changed functions (very large patches
                       are often refactors touching everything — low signal)
    """

    def __init__(
        self,
        kernel_root: Path,
        min_functions: int = 1,
        max_functions: int = 8,
    ):
        self.kernel_root = Path(kernel_root)
        self.min_functions = min_functions
        self.max_functions = max_functions

        if not (self.kernel_root / ".git").exists():
            raise ValueError(f"Not a git repo: {self.kernel_root}")

    def mine(
        self,
        since: str = "2018-01-01",
        verbose: bool = True,
    ) -> Iterator[Triplet]:
        """
        Yield Triplet objects for all qualifying commits since `since`.

        This is a generator — it processes one commit at a time so you can
        pipe the output to a file without loading all ~80K commits into RAM.
        """
        n_commits = 0
        n_triplets = 0
        n_skipped = 0

        for commit_meta in iter_commits(self.kernel_root, since=since):
            n_commits += 1
            if verbose and n_commits % 500 == 0:
                print(f"[mine] {n_commits} commits processed, "
                      f"{n_triplets} triplets, {n_skipped} skipped",
                      file=sys.stderr)

            sha     = commit_meta["sha"]
            subject = commit_meta["subject"]
            date    = commit_meta["date"]
            files   = commit_meta["files"]

            subsystem = extract_subsystem(subject)

            # Collect all changed functions across all changed files
            all_positives: list[str] = []
            for file_path in files:
                hunk_lines = get_diff_hunks(self.kernel_root, sha, file_path)
                if not hunk_lines:
                    continue
                fn_ranges = get_function_ranges(self.kernel_root, sha, file_path)
                changed_fns = functions_at_lines(fn_ranges, hunk_lines)
                for fn_name in changed_fns:
                    # Produce a CodeNode-style ID: "file_path::fn_name"
                    all_positives.append(f"{file_path}::{fn_name}")

            # Deduplicate (same function may appear in multiple hunks)
            positives = list(dict.fromkeys(all_positives))

            if not is_useful_commit(subject, files, positives):
                n_skipped += 1
                continue
            if not (self.min_functions <= len(positives) <= self.max_functions):
                n_skipped += 1
                continue

            triplet = Triplet(
                query=subject,
                commit=sha,
                date=date,
                positives=positives,
                subsystem=subsystem,
                changed_files=files,
            )
            n_triplets += 1
            yield triplet

        if verbose:
            print(f"[mine] Done: {n_commits} commits → "
                  f"{n_triplets} triplets ({n_skipped} skipped)",
                  file=sys.stderr)


# ─── Split by date ────────────────────────────────────────────────────────────

def date_split(
    triplets: list[Triplet],
    val_cutoff: str = "2022-01-01",
    test_cutoff: str = "2023-01-01",
) -> tuple[list[Triplet], list[Triplet], list[Triplet]]:
    """
    Split triplets into train/val/test by commit date.

    Critical: always split by time, never randomly. Random splits allow
    data leakage — test commits from the same author/subsystem as training
    commits are trivially "solved" by memorization rather than generalization.

    Commits before val_cutoff  → train
    val_cutoff ≤ date < test_cutoff → val
    date ≥ test_cutoff         → test

    The gold eval set (eval/retrieval_gold.jsonl) was written referencing
    kernel code from 2023+, so the test split lines up with it.
    """
    train, val, test = [], [], []
    for t in triplets:
        if t.date < val_cutoff:
            train.append(t)
        elif t.date < test_cutoff:
            val.append(t)
        else:
            test.append(t)
    return train, val, test


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mine (query, positives) triplets from Linux kernel git history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kernel",        required=True,       help="Path to Linux kernel git repo")
    parser.add_argument("--output",        default="-",         help="Output JSONL file (default: stdout)")
    parser.add_argument("--since",         default="2018-01-01", help="Mine commits since this date")
    parser.add_argument("--min-functions", type=int, default=1,  help="Min changed functions per commit")
    parser.add_argument("--max-functions", type=int, default=8,  help="Max changed functions per commit")
    parser.add_argument("--quiet",         action="store_true",  help="Suppress progress output")
    args = parser.parse_args()

    miner = TripletMiner(
        kernel_root=Path(args.kernel),
        min_functions=args.min_functions,
        max_functions=args.max_functions,
    )

    out = open(args.output, "w") if args.output != "-" else sys.stdout
    try:
        for triplet in miner.mine(since=args.since, verbose=not args.quiet):
            print(triplet.to_json(), file=out)
    finally:
        if args.output != "-":
            out.close()
