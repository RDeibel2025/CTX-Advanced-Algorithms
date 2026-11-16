"""Succinct and compressed structures: packed bit vector, rank/select and wavelet trees.

Modules
-------
* :mod:`~src.advds.succinct.bit_vector` - :class:`BitVector`, a fixed-length
  bit vector packed into numpy ``uint64`` words (bit i is bit ``i % 64`` of
  word ``i // 64``).
* :mod:`~src.advds.succinct.rank_select` - :class:`RankSelect`: constant-time
  ``rank1`` / ``rank0`` from 2,048-bit superblocks and 256-bit blocks plus the
  256-entry :data:`POPCOUNT8` table (7.8125% overhead), and sampled
  ``select1`` / ``select0`` finished with the :data:`SELECT8` table.
* :mod:`~src.advds.succinct.wavelet_tree` - :class:`WaveletTree`, balanced
  (alphabet halved at every node) or Huffman-shaped (payload near ``n * H0``),
  with access, rank, select, range counting, range quantile, frequencies and
  decode; :func:`zero_order_entropy` gives H0 for comparison.

Conventions: indices are 0-based and ranges half-open ``[l, r)``.
``rank1(i)`` counts the 1 bits in ``[0, i)``; ``select1(k)`` and
``WaveletTree.select(c, k)`` take a 1-based k, so ``rank1(select1(k)) ==
k - 1``; ``range_quantile(l, r, k)`` returns the k-th smallest (1-based)
symbol of ``[l, r)`` in alphabet order. Out-of-range arguments raise
``IndexError``.

Examples:
    >>> from src.advds.succinct import BitVector, RankSelect, WaveletTree
    >>> rs = RankSelect(BitVector.from_string("1011001110001011"))
    >>> rs.rank1(8), rs.select1(4)
    (5, 6)
    >>> WaveletTree("ACGTACGTTGCA").range_quantile(3, 9, 3)
    'G'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from src.advds.succinct.bit_vector import WORD_BITS, BitVector
from src.advds.succinct.rank_select import (
    BLOCK_BITS,
    POPCOUNT8,
    SELECT8,
    SELECT_SAMPLE_RATE,
    SUPERBLOCK_BITS,
    RankSelect,
)
from src.advds.succinct.wavelet_tree import SHAPES, WaveletTree, zero_order_entropy

__all__ = [
    "BLOCK_BITS",
    "BitVector",
    "POPCOUNT8",
    "RankSelect",
    "SELECT8",
    "SELECT_SAMPLE_RATE",
    "SHAPES",
    "SUPERBLOCK_BITS",
    "WORD_BITS",
    "WaveletTree",
    "zero_order_entropy",
]
