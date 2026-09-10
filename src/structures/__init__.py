"""Week 3 data structures: binary heaps, an AVL tree and a hash table.

The three structures behind most fast algorithms, each written from scratch
so its cost can be measured rather than taken on trust:

* :mod:`src.structures.heap` - array-backed binary min-heap and max-heap,
  and a stable :class:`~src.structures.heap.PriorityQueue` built on them.
* :mod:`src.structures.avl_tree` - a self-balancing binary search tree with
  deletion rebalancing.
* :mod:`src.structures.hash_table` - a hash table with both separate chaining
  and open addressing (linear probing), selectable per instance.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.structures.heap import MaxHeap, MinHeap, PriorityQueue

__all__ = ["MinHeap", "MaxHeap", "PriorityQueue"]
