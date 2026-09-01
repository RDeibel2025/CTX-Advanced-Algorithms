"""Merge sort — the canonical divide-and-conquer sorting algorithm.

Merge sort splits the input in half, sorts each half recursively, and then
*merges* the two sorted halves in linear time. The split is positional
rather than value-based, so the recursion tree is always balanced: every
input of size n produces a tree of depth ceil(log2 n) with O(n) work per
level, whatever order the input happens to be in. That is why merge sort
has no bad case — its best, average and worst are all O(n log n).

This is the structural opposite of quicksort (:mod:`src.sorting.quick_sort`),
which does its partitioning work *before* recursing and can therefore be
knocked off balance by the input. Merge sort pays for that guarantee with
O(n) auxiliary space, which quicksort does not need.

The three Week 1 sorts, this module and :mod:`src.sorting.quick_sort` all
honour the same contract, so they are interchangeable in the benchmark
harness:

* a non-``list`` argument raises :class:`TypeError`;
* the caller's list is never mutated — a new list is returned;
* empty and single-element inputs need no special-casing by the caller;
* any mutually comparable element type works.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from typing import List, TypeVar

# Imported rather than reimplemented so that every sort in this package
# raises an identically worded TypeError. One validator, one message.
from src.sorting.basic_sorts import Comparable, _require_list

T = TypeVar("T", bound=Comparable)

__all__ = ["merge_sort", "merge"]


def merge(left: List[T], right: List[T]) -> List[T]:
    """Merge two already-sorted lists into one sorted list, in linear time.

    Walks both inputs with an index apiece, repeatedly taking the smaller
    head element. Because each step advances exactly one index and neither
    index ever moves backwards, the total work is O(n + m).

    The linearity is the whole point, and it is easy to lose by accident.
    ``left.pop(0)`` looks like the natural way to consume the head of a
    list, but it is O(n) in CPython — every remaining element shifts down
    one slot — which would quietly make this merge quadratic and merge sort
    O(n^2 log n). Index cursors avoid that. So does never calling
    ``sorted()`` here: that would hide the algorithm being demonstrated
    behind Python's own Timsort.

    Args:
        left: A sorted list. Not modified.
        right: A sorted list. Not modified.

    Returns:
        A new sorted list containing every element of both inputs,
        including duplicates.

    Raises:
        TypeError: If either argument is not a list.

    Time Complexity:
        O(n + m) in all cases, where n and m are the two input lengths.
        Exactly n + m elements are appended, and each comparison advances
        one cursor.

    Space Complexity:
        O(n + m) for the returned list.

    Stability:
        Stable - when the two heads compare equal the element from ``left``
        is taken first, so elements from the earlier run keep their lead.

    Examples:
        >>> merge([1, 3, 5], [2, 4, 6])
        [1, 2, 3, 4, 5, 6]
        >>> merge([], [1, 2])
        [1, 2]
        >>> merge([1, 2], [])
        [1, 2]
        >>> merge([], [])
        []

        One run entirely below the other still costs only one pass:

        >>> merge([1, 2, 3], [4, 5, 6])
        [1, 2, 3, 4, 5, 6]

        Ties resolve in favour of ``left``, which is what makes the sort
        stable. Here both 1-keyed tuples come from ``left`` and keep their
        order relative to the equal-keyed tuple in ``right``:

        >>> merge([(1, 'a'), (1, 'b')], [(1, 'c')])
        [(1, 'a'), (1, 'b'), (1, 'c')]

        >>> merge([1, 2], (3, 4))
        Traceback (most recent call last):
            ...
        TypeError: merge() expects a list, got tuple
    """
    _require_list(left, "merge")
    _require_list(right, "merge")

    merged: List[T] = []
    i = 0
    j = 0

    # The main merge loop. Runs until one side is exhausted.
    while i < len(left) and j < len(right):
        # `<=` rather than `<` is what makes the whole sort stable: on a
        # tie the left run wins, preserving the original relative order.
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Exactly one of these two extensions does any work: whichever run
    # still has elements left is already sorted and already greater than
    # everything merged so far, so it can be appended wholesale.
    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


def merge_sort(arr: List[T]) -> List[T]:
    """Sort a list in ascending order using merge sort.

    Divide, conquer, combine:

    * **Divide** — split the list at its midpoint. This is a positional
      split, so the two halves always differ in size by at most one.
    * **Conquer** — sort each half by recursing. A list of fewer than two
      elements is already sorted and ends the recursion.
    * **Combine** — hand the two sorted halves to :func:`merge`, which
      interleaves them in linear time.

    The balanced split is what gives merge sort its guarantee. The
    recursion tree is ceil(log2 n) levels deep for every input, and each
    level merges a total of n elements, so the cost is O(n log n) whether
    the input arrives sorted, reversed or shuffled. This solves the
    recurrence T(n) = 2T(n/2) + Theta(n), which is Master Theorem case 2
    (see ``analysis/week2_recurrences.md``).

    Args:
        arr: A list of comparable elements. Not modified.

    Returns:
        A new list containing the same elements in ascending order.

    Raises:
        TypeError: If arr is not a list. TypeError also propagates from the
            comparison itself if the elements are not mutually comparable.

    Time Complexity:
        Best:    O(n log n)
        Average: O(n log n)
        Worst:   O(n log n)   - the recursion tree depth does not depend on
                                input order

    Space Complexity:
        O(n) - auxiliary arrays during merge, plus O(log n) recursion stack.

    Stability:
        Stable - merge takes from the left run when elements compare equal.

    Examples:
        >>> merge_sort([3, 1, 2])
        [1, 2, 3]
        >>> merge_sort([])
        []
        >>> merge_sort([5])
        [5]
        >>> merge_sort([-2, 7, 0, -9, 7])
        [-9, -2, 0, 7, 7]
        >>> merge_sort(["pear", "apple", "fig"])
        ['apple', 'fig', 'pear']
        >>> merge_sort([2.5, -1.5, 0.0])
        [-1.5, 0.0, 2.5]

        Stability, shown on (key, tag) pairs sorted by a key that ties.
        The two 1-keyed entries keep their input order:

        >>> merge_sort([(1, 'a'), (0, 'x'), (1, 'b')])
        [(0, 'x'), (1, 'a'), (1, 'b')]

        The caller's list is left alone:

        >>> original = [3, 1, 2]
        >>> _ = merge_sort(original)
        >>> original
        [3, 1, 2]

        >>> merge_sort("not a list")
        Traceback (most recent call last):
            ...
        TypeError: merge_sort() expects a list, got str
    """
    _require_list(arr, "merge_sort")

    # Base case: zero or one element is sorted by definition. This also
    # makes the copy that keeps the function non-destructive - every
    # element reaches this line exactly once.
    if len(arr) < 2:
        return list(arr)

    # Divide.
    middle = len(arr) // 2
    left_half = arr[:middle]
    right_half = arr[middle:]

    # Conquer, then combine.
    return merge(merge_sort(left_half), merge_sort(right_half))
