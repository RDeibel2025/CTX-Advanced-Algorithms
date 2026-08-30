"""Sorting algorithms.

Week 1 provides the three basic comparison sorts required by the
assignment. Each is non-destructive: it returns a new list and leaves the
caller's list untouched.

    >>> from src.sorting import bubble_sort, selection_sort, insertion_sort
    >>> bubble_sort([3, 1, 2])
    [1, 2, 3]
"""

from src.sorting.basic_sorts import bubble_sort, insertion_sort, selection_sort

__all__ = ["bubble_sort", "selection_sort", "insertion_sort"]
