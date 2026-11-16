#!/usr/bin/env python3
"""Runnable demonstration of the Week 13 DNA index: wavelet tree plus FM-index.

    python examples/week13_dna_demo.py

Two parts:

1. **The synthetic FASTA fragment.** ``src/advds/examples/synthetic_fragment.fa``
   (20,000 synthetic bases from ``make_synthetic_fasta(path, n=20000,
   seed=42)``, created here if it is missing) is read and indexed. The demo
   prints the base frequencies, ``count("A", l, r)`` on a few windows,
   ``kth_smallest`` on one window and ``pattern_count`` for ``GATTACA``,
   ``ACGT`` and ``CG``.
2. **The spec 9.7 anchors** on ``synthetic_dna(100000, 42)``: the first 20
   bases, the frequencies, ``count("A", 1000, 1999) = 254``,
   ``kth_smallest(5000, 5099, k)`` for k = 1, 50, 100, and the pattern counts
   GATTACA 8, ACGT 399, AAAA 387 and CG 6,383.

Ranges are INCLUSIVE ``[l, r]``. Every printed value is checked against a
brute-force ruler (``str.count`` on slices for window counts, ``sorted`` on
the window for the k-th smallest, and a naive overlapping scan for pattern
counts), and every spec anchor is asserted. All sequences here are synthetic
(uniform random bases), never a real genome. It finishes in about a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.advds.applications.dna_index import (  # noqa: E402
    DEFAULT_FASTA,
    DNAIndex,
    make_synthetic_fasta,
    read_fasta,
    synthetic_dna,
)

WIDTH = 74
PATTERNS = ("GATTACA", "ACGT", "CG")


def banner(number: int, title: str) -> None:
    """Print a numbered section heading."""
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    """Print one labelled result line."""
    print(f"  {label:<40} {value}")


def naive_count(seq: str, pattern: str) -> int:
    """Overlapping occurrences of ``pattern`` by checking every start position."""
    m = len(pattern)
    return sum(1 for p in range(len(seq) - m + 1) if seq[p : p + m] == pattern)


def main() -> int:
    """Index the fragment and the 100,000-base anchor sequence; return 0."""
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 13 DNA index demonstration")
    print("Wavelet tree for windows, FM-index (BWT + rank) for pattern counts")
    print("=" * WIDTH)
    print("  Synthetic sequences only. Ranges are INCLUSIVE. Every value is checked.")

    banner(1, "The synthetic FASTA fragment")
    if not os.path.exists(DEFAULT_FASTA):
        make_synthetic_fasta(DEFAULT_FASTA, n=20000, seed=42)
        show("created", os.path.relpath(DEFAULT_FASTA, REPO_ROOT))
    seq = read_fasta(DEFAULT_FASTA)
    assert seq == synthetic_dna(20000, 42)
    idx = DNAIndex(seq)
    show("file", os.path.relpath(DEFAULT_FASTA, REPO_ROOT))
    show("bases", f"{len(idx):,}")
    freqs = idx.frequencies()
    assert freqs == {b: seq.count(b) for b in "ACGT"}
    show("frequencies", freqs)
    for l, r in ((0, 99), (1000, 1999), (0, len(seq) - 1)):
        a = idx.count("A", l, r)
        assert a == seq[l : r + 1].count("A")
        show(f'count("A", {l}, {r})', a)
    window = sorted(seq[500:600])
    ks = (1, 25, 50, 75, 100)
    picks = [idx.kth_smallest(500, 599, k) for k in ks]
    assert picks == [window[k - 1] for k in ks]
    show(f"kth_smallest(500, 599, k), k = {ks}", picks)
    for pattern in PATTERNS:
        c = idx.pattern_count(pattern)
        assert c == naive_count(seq, pattern)
        show(f'pattern_count("{pattern}")', f"{c:,}")

    banner(2, "Spec 9.7 anchors on synthetic_dna(100000, 42)")
    big = synthetic_dna(100000, 42)
    index = DNAIndex(big)
    assert big[:20] == "ATGCCTAGAAGTGTGTGATC"
    show("first 20 bases", big[:20])
    f = index.frequencies()
    assert f == {"A": 25021, "C": 24890, "G": 25154, "T": 24935}
    show("frequencies", f)
    a = index.count("A", 1000, 1999)
    assert a == 254 == big[1000:2000].count("A")
    show('count("A", 1000, 1999)  [anchor 254]', a)
    picks = [index.kth_smallest(5000, 5099, k) for k in (1, 50, 100)]
    assert picks == ["A", "G", "T"]
    show("kth_smallest(5000, 5099, 1/50/100)", picks)
    expected = {"GATTACA": 8, "ACGT": 399, "AAAA": 387, "CG": 6383}
    for pattern, want in expected.items():
        c = index.pattern_count(pattern)
        assert c == want == naive_count(big, pattern)
        show(f'pattern_count("{pattern}")  [anchor {want:,}]', f"{c:,}")

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
