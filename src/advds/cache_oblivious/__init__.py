"""Cache-oblivious matrix operations, tree layouts and the LRU cache simulator.

Modules
-------
* :mod:`~src.advds.cache_oblivious.matrix_ops` - row-by-row and recursive
  (cache-oblivious) transpose and multiply, the Z-order traversal, and
  :func:`trace_transpose` / :func:`trace_matmul`, which replay the same loops
  as address accesses on a simulated cache (row-major layout, ``base=1``).
* :mod:`~src.advds.cache_oblivious.layouts` - the BFS (heap) and van Emde
  Boas layouts of a complete binary search tree (top tree ``h // 2``
  levels, bottom trees ``h - h // 2``), search with the visited positions,
  and :func:`blocks_per_search`.
* :mod:`~src.advds.cache_oblivious.cache_sim` - :class:`LRUCache`, a fully
  associative cache of M words in blocks of B words with LRU eviction; both
  other modules count misses with it.

Neither recursive algorithm knows M or B: they halve until a piece fits a
block of whatever size the cache has, which is why misses are measured on
the simulator at several ``(M, B)`` settings.

Examples:
    >>> from src.advds.cache_oblivious import LRUCache, trace_transpose, veb_order
    >>> veb_order(3)
    [1, 2, 4, 5, 3, 6, 7]
    >>> trace_transpose(64, "recursive", LRUCache(1024, 16))
    512

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from src.advds.cache_oblivious.cache_sim import LRUCache
from src.advds.cache_oblivious.layouts import (
    LAYOUTS,
    bfs_layout,
    blocks_per_search,
    build_bfs_tree,
    build_veb_tree,
    search,
    veb_order,
)
from src.advds.cache_oblivious.matrix_ops import (
    METHODS,
    co_matmul,
    co_transpose,
    naive_matmul,
    naive_transpose,
    trace_matmul,
    trace_transpose,
    z_order,
)

__all__ = [
    "LAYOUTS",
    "LRUCache",
    "METHODS",
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
]
