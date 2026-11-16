"""Tree-based range-query structures: segment, Fenwick and persistent trees, persistent map.

Modules
-------
* :mod:`~src.advds.trees.segment_tree` - :class:`SegmentTree`, an
  array-based tree that halves the index range; range sum, min and max and
  lazy range add, all Theta(log n).
* :mod:`~src.advds.trees.fenwick_tree` - :class:`FenwickTree` (point add,
  prefix and range sums) and :class:`RangeFenwick` (range add and range sum
  with the two-tree B1/B2 trick); index i covers the ``i & -i`` elements
  ending at i.
* :mod:`~src.advds.trees.persistent_tree` - :class:`PersistentSegmentTree`
  (path copying with non-propagating tags; a point set on ``n = 2^k``
  creates ``k + 1`` nodes) and :class:`PersistentMap` (a treap with
  ``stable_hash64`` priorities and Merkle hashes, used by the version-control
  application).

Conventions: indices are 0-based and every range here is half-open
``[l, r)``. Persistent structures never change a node after it is
published; every update returns a new version.

Examples:
    >>> from src.advds.trees import FenwickTree, PersistentSegmentTree, SegmentTree
    >>> SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]).query_sum(2, 7)
    28
    >>> FenwickTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0]).prefix_sum(4)
    16
    >>> t = PersistentSegmentTree([1, 2, 3])
    >>> v = t.point_set(0, 0, 10)
    >>> t.query_sum(0, 0, 3), t.query_sum(v, 0, 3)
    (6, 15)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from src.advds.trees.fenwick_tree import FenwickTree, RangeFenwick
from src.advds.trees.persistent_tree import (
    EMPTY_MAP_HASH,
    MapNode,
    PersistentMap,
    PersistentSegmentTree,
    SegmentNode,
)
from src.advds.trees.segment_tree import SegmentTree

__all__ = [
    "EMPTY_MAP_HASH",
    "FenwickTree",
    "MapNode",
    "PersistentMap",
    "PersistentSegmentTree",
    "RangeFenwick",
    "SegmentNode",
    "SegmentTree",
]
