"""Applications of the Week 13 structures: range queries, mini version control and a DNA index.

Modules
-------
* :mod:`~src.advds.applications.range_queries` - :class:`RangeQueryEngine`:
  range sum (range Fenwick), range min and max (segment tree) and time
  travel through every earlier version (persistent segment tree), with
  INCLUSIVE ranges ``[l, r]``.
* :mod:`~src.advds.applications.version_control` - :class:`MiniVCS`:
  commits as persistent treaps (path -> SHA-1 of the content), commit ids
  ``sha1(root_hash + parent_id + message)[:12]``, a diff that skips subtrees
  with equal Merkle hashes, and node-sharing statistics.
* :mod:`~src.advds.applications.dna_index` - :class:`DNAIndex`: a wavelet
  tree over the sequence for window counts and quantiles (INCLUSIVE ranges)
  and an FM-index over ``seq + "$"`` for overlapping pattern counts, plus the
  suffix array, BWT, FASTA and synthetic-DNA helpers.

The two application-facing query APIs take inclusive ranges because the
instructions phrase them that way; they convert once to the half-open ranges
of the core structures with :func:`inclusive_to_half_open`. All synthetic
data here (arrays, files, DNA) is synthetic and never presented as real.

Examples:
    >>> from src.advds.applications import DNAIndex, MiniVCS, RangeQueryEngine
    >>> RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]).sum(2, 6)
    28
    >>> DNAIndex("ACGTACGTTGCA").pattern_count("CG")
    2
    >>> len(MiniVCS())
    0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from src.advds.applications.dna_index import (
    DEFAULT_FASTA,
    DNA_ALPHABET,
    FM_ALPHABET,
    DNAIndex,
    build_suffix_array,
    bwt_from_suffix_array,
    make_synthetic_fasta,
    read_fasta,
    synthetic_dna,
)
from src.advds.applications.range_queries import RangeQueryEngine, inclusive_to_half_open
from src.advds.applications.version_control import MiniVCS, diff_maps, sha1_hex

__all__ = [
    "DEFAULT_FASTA",
    "DNAIndex",
    "DNA_ALPHABET",
    "FM_ALPHABET",
    "MiniVCS",
    "RangeQueryEngine",
    "build_suffix_array",
    "bwt_from_suffix_array",
    "diff_maps",
    "inclusive_to_half_open",
    "make_synthetic_fasta",
    "read_fasta",
    "sha1_hex",
    "synthetic_dna",
]
