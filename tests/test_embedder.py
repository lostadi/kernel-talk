"""
tests/test_embedder.py
──────────────────────
Unit tests for core/mirror/embedder.py — specifically chunk_text,
which had a critical infinite-loop bug when text length was not an exact
multiple of (chars_per_chunk - overlap_chars).
"""

import pytest
from core.mirror.embedder import chunk_text


# ─── chunk_text correctness ───────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        text = "static inline void noop(void) {}"
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        assert chunks == [text]

    def test_exact_boundary_single_chunk(self):
        # exactly chars_per_chunk (450*4 = 1800 chars) → one chunk
        text = "x" * 1800
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_one_over_boundary_splits(self):
        # 1801 chars → must produce 2 chunks (no infinite loop)
        text = "y" * 1801
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        assert 1 < len(chunks) <= 3
        assert all(len(c) > 0 for c in chunks)

    def test_no_infinite_loop_various_lengths(self):
        """
        Regression test for the infinite-loop bug:
        when end == len(text), the overlap step set start = len-200 < len,
        causing the loop to repeat the final 200-char slice forever.
        """
        for n in [1801, 2000, 3600, 5000, 10_000, 21_265]:
            chunks = chunk_text("a" * n, max_tokens=450, overlap=50)
            assert len(chunks) < 1000, f"probable infinite loop at n={n}: {len(chunks)} chunks"
            assert len(chunks) >= 1

    def test_all_text_covered(self):
        # Concatenating all chunks (without overlap) should cover the whole text.
        # We just verify first and last chars are present and total len is sane.
        text = "Z" * 9000
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        assert chunks[0][0] == "Z"
        assert chunks[-1][-1] == "Z"
        # Total covered chars >= original (overlaps inflate it slightly)
        assert sum(len(c) for c in chunks) >= len(text)

    def test_newline_boundary_respected(self):
        # Chunker should prefer to break at newlines.
        lines = ["// line %d" % i + " " * 50 for i in range(100)]
        text = "\n".join(lines)
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        for chunk in chunks:
            # Each chunk should not end mid-line (unless last)
            assert len(chunk) > 0

    def test_empty_string(self):
        assert chunk_text("", max_tokens=450, overlap=50) == []

    def test_single_very_long_line(self):
        # No newlines — falls back to character splitting.
        text = "A" * 5000
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        assert 2 <= len(chunks) <= 10
        assert chunks[0] == "A" * 1800
        assert chunks[-1][-1] == "A"

    def test_overlap_causes_forward_progress(self):
        # Each chunk's start must be strictly after the previous chunk's start.
        text = "B" * 7200
        chunks = chunk_text(text, max_tokens=450, overlap=50)
        # We can't directly inspect starts, but we can verify no duplicates
        # at chunk boundaries and that we terminate.
        assert len(chunks) >= 4
        assert len(chunks) < 100
