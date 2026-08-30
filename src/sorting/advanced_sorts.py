"""Advanced sorting algorithms — reserved for a later week.

This module is part of the required CSC 5300 project structure. Week 1
scopes the assignment to the three basic quadratic sorts, which live in
:mod:`src.sorting.basic_sorts`. The divide-and-conquer and distribution
sorts belong here and are deliberately not implemented yet:

* merge sort — O(n log n), stable, O(n) auxiliary space
* quick sort — O(n log n) expected, O(n^2) worst case, unstable
* heap sort — O(n log n) worst case, in place, unstable
* counting / radix sort — linear time under a bounded-key assumption

Adding them here later means the benchmarking framework in
:mod:`src.utils.benchmark` can compare them against the Week 1 baselines
without any change to the framework itself: ``benchmark_suite`` takes a
plain ``{name: callable}`` mapping, so a new algorithm is one dictionary
entry.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

# Intentionally empty for Week 1. Nothing is exported until the algorithms
# above are implemented, so ``from src.sorting.advanced_sorts import *``
# is a no-op rather than a surprise.
__all__: list = []
