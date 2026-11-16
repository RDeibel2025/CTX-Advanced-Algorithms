"""Week 13: persistent, succinct and cache-oblivious data structures.

``src/advds/`` is the instructions' ``AdvancedDataStructures/advds/`` project,
kept with the instructions' sub-folder and file names, following the Week
10-12 pattern of one package under ``src/``.

Sub-packages
------------
* :mod:`src.advds.trees` - segment tree with lazy propagation, Fenwick and
  range Fenwick trees, persistent segment tree and persistent map.
* :mod:`src.advds.succinct` - packed bit vector, constant-time rank with
  sampled select, balanced and Huffman-shaped wavelet trees.
* :mod:`src.advds.cache_oblivious` - recursive transpose and multiply, Z-order
  traversal, BFS and van Emde Boas layouts, and the LRU cache simulator.
* :mod:`src.advds.applications` - the range-query engine, the mini version
  control system and the DNA index.
* :mod:`src.advds.benchmarks` - the benchmark (``performance.py``); it is not
  imported here and runs only when called.
* ``src/advds/examples/`` - ``synthetic_fragment.fa``, a small synthetic
  FASTA file (not a real genome).

Conventions (every module states them in its docstring)
--------------------------------------------------------
* Indices are 0-based.
* Core structures use half-open ranges ``[l, r)``.
* The two application-facing APIs, :class:`RangeQueryEngine` and
  :class:`DNAIndex`, use inclusive ranges ``[l, r]``, because the
  instructions phrase them that way, and convert once at the boundary.
* ``select`` (``select1``, ``select0``, ``WaveletTree.select``) and quantile
  ranks (``range_quantile``, ``DNAIndex.kth_smallest``) are 1-based.
* Persistent structures never mutate a node after it is published.
* Every structure and query is written by hand; no library routine answers
  a query. numpy holds the storage (packed words, int64 arrays), does
  build-time array preparation, the suffix-array sort (``np.lexsort``) and
  the spec's ``synthetic_dna`` generator.

The names below are re-exported from the sub-packages, so
``from src.advds import WaveletTree`` works. The sub-packages import each
other only through their own modules, never through this file, so there is
no import cycle.

Examples:
    >>> import src.advds as advds
    >>> advds.RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]).sum(2, 6)
    28
    >>> advds.WaveletTree("ACGTACGTTGCA").rank("C", 12)
    3
    >>> advds.DNAIndex("ACGTACGTTGCA").bwt()
    'AC$TGAATCCGTG'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from src.advds.applications import (
    DNAIndex,
    MiniVCS,
    RangeQueryEngine,
    build_suffix_array,
    bwt_from_suffix_array,
    diff_maps,
    inclusive_to_half_open,
    make_synthetic_fasta,
    read_fasta,
    synthetic_dna,
)
from src.advds.cache_oblivious import (
    LRUCache,
    bfs_layout,
    blocks_per_search,
    build_bfs_tree,
    build_veb_tree,
    co_matmul,
    co_transpose,
    naive_matmul,
    naive_transpose,
    search,
    trace_matmul,
    trace_transpose,
    veb_order,
    z_order,
)
from src.advds.succinct import BitVector, RankSelect, WaveletTree, zero_order_entropy
from src.advds.trees import (
    FenwickTree,
    PersistentMap,
    PersistentSegmentTree,
    RangeFenwick,
    SegmentTree,
)

__all__ = [
    # trees
    "FenwickTree",
    "PersistentMap",
    "PersistentSegmentTree",
    "RangeFenwick",
    "SegmentTree",
    # succinct
    "BitVector",
    "RankSelect",
    "WaveletTree",
    "zero_order_entropy",
    # cache_oblivious
    "LRUCache",
    "bfs_layout",
    "blocks_per_search",
    "build_bfs_tree",
    "build_veb_tree",
    "co_matmul",
    "co_transpose",
    "naive_matmul",
    "naive_transpose",
    "search",
    "trace_matmul",
    "trace_transpose",
    "veb_order",
    "z_order",
    # applications
    "DNAIndex",
    "MiniVCS",
    "RangeQueryEngine",
    "build_suffix_array",
    "bwt_from_suffix_array",
    "diff_maps",
    "inclusive_to_half_open",
    "make_synthetic_fasta",
    "read_fasta",
    "synthetic_dna",
]
