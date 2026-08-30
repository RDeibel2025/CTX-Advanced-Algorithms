"""Basic comparison-based sorting algorithms.

This module implements the three quadratic sorting algorithms required by
the CSC 5300 Week 1 laboratory: an *optimized* bubble sort, selection sort
and insertion sort.

All three share the same contract:

* The argument must be a ``list``; anything else raises :class:`TypeError`.
* The argument is **never mutated** — a new sorted list is returned.
* Empty lists and single-element lists are handled without the caller
  special-casing them.
* Any mutually comparable element type works (``int``, ``float``, ``str``,
  ``tuple``, or any object implementing ``__lt__``), not just integers.

The three differ in stability and in best-case behaviour, which is exactly
what the Week 1 benchmarking exercise is designed to expose:

=============== ========= ========== ========== =========
Algorithm       Best      Average    Worst      Stable?
=============== ========= ========== ========== =========
bubble_sort     O(n)      O(n^2)     O(n^2)     yes
selection_sort  O(n^2)    O(n^2)     O(n^2)     no
insertion_sort  O(n)      O(n^2)     O(n^2)     yes
=============== ========= ========== ========== =========

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from typing import Any, List, Protocol, TypeVar


class Comparable(Protocol):
    """Structural type for anything that supports ``<``.

    Declaring the bound this way documents the real requirement of a
    comparison sort: elements need an ordering, not a specific concrete
    type. It is a typing-time construct only and adds no runtime cost.
    """

    def __lt__(self, other: Any) -> bool:  # pragma: no cover - typing only
        ...


T = TypeVar("T", bound=Comparable)

__all__ = ["bubble_sort", "selection_sort", "insertion_sort"]


def _require_list(arr: Any, func_name: str) -> None:
    """Raise :class:`TypeError` unless ``arr`` is a ``list``.

    Centralising the check keeps the error message identical across all
    three public functions and keeps each algorithm body focused on the
    algorithm itself.

    Args:
        arr: The object to validate.
        func_name: Name of the calling function, used in the message.

    Raises:
        TypeError: If ``arr`` is not a ``list``.

    Examples:
        >>> _require_list([1, 2], "demo") is None
        True
        >>> _require_list((1, 2), "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects a list, got tuple
    """
    if not isinstance(arr, list):
        raise TypeError(f"{func_name}() expects a list, got {type(arr).__name__}")


def bubble_sort(arr: List[T]) -> List[T]:
    """Sort a list in ascending order using optimized bubble sort.

    Repeatedly steps through the list, comparing adjacent elements and
    swapping them when out of order. Terminates early if a full pass
    completes without any swaps, which makes already-sorted input O(n).
    A second optimization shrinks the scanned range by one on every pass,
    because after pass *i* the largest *i* elements are already in their
    final positions.

    Args:
        arr: A list of comparable elements. Not modified.

    Returns:
        A new list containing the same elements in ascending order.

    Raises:
        TypeError: If arr is not a list. TypeError also propagates from the
            comparison itself if the elements are not mutually comparable.

    Time Complexity:
        Best:    O(n)     - already sorted, one pass, early exit
        Average: O(n^2)
        Worst:   O(n^2)   - reverse sorted

    Space Complexity:
        O(n) - a copy of the input is returned; O(1) auxiliary beyond that.

    Stability:
        Stable - equal elements retain their relative order, because the
        swap condition is strict greater-than and never fires on a tie.

    Examples:
        >>> bubble_sort([3, 1, 2])
        [1, 2, 3]
        >>> bubble_sort([])
        []
        >>> bubble_sort([5])
        [5]
        >>> bubble_sort([-2, 7, 0, -9, 7])
        [-9, -2, 0, 7, 7]
        >>> bubble_sort(["pear", "apple", "fig"])
        ['apple', 'fig', 'pear']
        >>> original = [3, 1, 2]
        >>> _ = bubble_sort(original)
        >>> original
        [3, 1, 2]
        >>> bubble_sort("not a list")
        Traceback (most recent call last):
            ...
        TypeError: bubble_sort() expects a list, got str
    """
    _require_list(arr, "bubble_sort")

    result = list(arr)
    n = len(result)

    # Lists of length 0 or 1 are sorted by definition; the loop below would
    # simply not execute, but returning early makes that explicit.
    if n < 2:
        return result

    for i in range(n - 1):
        # After i completed passes the last i elements are final, so the
        # inner scan can stop that much earlier each time.
        swapped = False
        for j in range(n - 1 - i):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        # THE OPTIMIZATION: a pass with no swaps means the list is sorted,
        # so every remaining pass would be wasted work. This is what turns
        # the best case from O(n^2) into O(n).
        if not swapped:
            break

    return result


def selection_sort(arr: List[T]) -> List[T]:
    """Sort a list in ascending order using selection sort.

    Divides the list into a sorted prefix and an unsorted suffix. On each
    pass it scans the whole unsorted suffix for the smallest element and
    swaps that element into the boundary position. The scan is exhaustive
    regardless of how the input is arranged, so there is no early exit to
    add and no best case to exploit.

    Args:
        arr: A list of comparable elements. Not modified.

    Returns:
        A new list containing the same elements in ascending order.

    Raises:
        TypeError: If arr is not a list. TypeError also propagates from the
            comparison itself if the elements are not mutually comparable.

    Time Complexity:
        Best:    O(n^2)   - the full scan runs even on sorted input
        Average: O(n^2)
        Worst:   O(n^2)
        Comparisons are always exactly n(n-1)/2; only the swap count
        varies, and it is at most n-1.

    Space Complexity:
        O(n) - a copy of the input is returned; O(1) auxiliary beyond that.

    Stability:
        Unstable - the long-range swap can move an element past an equal
        one. For example [(1, 'a'), (1, 'b'), (0, 'c')] sorted on the first
        field moves (1, 'a') behind (1, 'b').

    Examples:
        >>> selection_sort([3, 1, 2])
        [1, 2, 3]
        >>> selection_sort([])
        []
        >>> selection_sort([5])
        [5]
        >>> selection_sort([-2, 7, 0, -9, 7])
        [-9, -2, 0, 7, 7]
        >>> selection_sort(["pear", "apple", "fig"])
        ['apple', 'fig', 'pear']
        >>> original = [3, 1, 2]
        >>> _ = selection_sort(original)
        >>> original
        [3, 1, 2]
        >>> selection_sort({1: "a"})
        Traceback (most recent call last):
            ...
        TypeError: selection_sort() expects a list, got dict
    """
    _require_list(arr, "selection_sort")

    result = list(arr)
    n = len(result)

    if n < 2:
        return result

    for i in range(n - 1):
        # Find the smallest element in the unsorted suffix result[i:].
        min_index = i
        for j in range(i + 1, n):
            if result[j] < result[min_index]:
                min_index = j

        # Skip the write when the minimum is already in place. This saves a
        # store but does not change the asymptotic cost: the scan above ran
        # in full either way.
        if min_index != i:
            result[i], result[min_index] = result[min_index], result[i]

    return result


def insertion_sort(arr: List[T]) -> List[T]:
    """Sort a list in ascending order using insertion sort.

    Grows a sorted prefix one element at a time. Each new element is
    carried leftwards over every larger element until it reaches its
    place. The inner loop stops as soon as a smaller-or-equal neighbour is
    found, so on input that is already sorted or nearly sorted each element
    travels only a short distance and the total cost approaches O(n).

    Args:
        arr: A list of comparable elements. Not modified.

    Returns:
        A new list containing the same elements in ascending order.

    Raises:
        TypeError: If arr is not a list. TypeError also propagates from the
            comparison itself if the elements are not mutually comparable.

    Time Complexity:
        Best:    O(n)     - already sorted, each element stays put
        Average: O(n^2)
        Worst:   O(n^2)   - reverse sorted, every element travels the
                            full width of the sorted prefix

    Space Complexity:
        O(n) - a copy of the input is returned; O(1) auxiliary beyond that.

    Stability:
        Stable - the shift condition is strict greater-than, so an element
        never moves past one that compares equal to it.

    Examples:
        >>> insertion_sort([3, 1, 2])
        [1, 2, 3]
        >>> insertion_sort([])
        []
        >>> insertion_sort([5])
        [5]
        >>> insertion_sort([-2, 7, 0, -9, 7])
        [-9, -2, 0, 7, 7]
        >>> insertion_sort(["pear", "apple", "fig"])
        ['apple', 'fig', 'pear']
        >>> insertion_sort([(1, 'a'), (0, 'b'), (1, 'c')])
        [(0, 'b'), (1, 'a'), (1, 'c')]
        >>> original = [3, 1, 2]
        >>> _ = insertion_sort(original)
        >>> original
        [3, 1, 2]
        >>> insertion_sort(42)
        Traceback (most recent call last):
            ...
        TypeError: insertion_sort() expects a list, got int
    """
    _require_list(arr, "insertion_sort")

    result = list(arr)
    n = len(result)

    if n < 2:
        return result

    for i in range(1, n):
        current = result[i]
        j = i - 1

        # Slide every element greater than `current` one slot to the right.
        # Strict `>` is what preserves stability: the loop stops at the
        # first element that is equal, leaving ties in their original order.
        while j >= 0 and result[j] > current:
            result[j + 1] = result[j]
            j -= 1

        result[j + 1] = current

    return result
