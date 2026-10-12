"""Randomized QuickSort, its deterministic baseline, and the expected comparisons.

This is CLRS RANDOMIZED-QUICKSORT (CLRS 4e section 7.3): before each
partition a uniformly random index in ``[lo, hi]`` is swapped to ``hi`` and
the subarray is split with Lomuto's partition. Randomizing the pivot moves
the O(n^2) worst case off any particular *input* and onto an unlucky
sequence of *draws*, so no fixed input, sorted or otherwise, is bad for it.

Why this module builds its own deterministic baseline
-----------------------------------------------------
The assignment compares randomized QuickSort with "the deterministic
QuickSort from Week 2", but Week 2's :func:`~src.sorting.quick_sort.quick_sort`
is itself randomized: it already picks a random pivot, partitions with
Hoare's scheme and hands small ranges to insertion sort. There is no
deterministic Week 2 sort to compare against, so this module builds the
baseline, :func:`deterministic_quicksort`, and Week 2's sort is kept only as
a reference line in the benchmark.

The two sorts here share one private implementation,
:func:`_quicksort_range`, which takes a pivot-choosing callback. The
randomized sort passes a callback that draws a random index; the
deterministic one passes a fixed rule, either ``"last"`` (CLRS, always
``hi``) or ``"median_of_three"`` (the median of the first, middle and last
elements). Everything else, from the Lomuto loop to the comparison count,
is the same code, so any difference the benchmark measures comes from the
pivot rule and nothing else.

What is counted, and why there is no insertion-sort cutoff
-----------------------------------------------------------
A comparison is one element-vs-pivot test inside Lomuto's loop. Partitioning
a subarray of ``s`` elements therefore costs exactly ``s - 1`` comparisons,
and subarrays of size 0 or 1 are never partitioned. That is the quantity
the CLRS analysis counts: elements ``z_i < z_j`` (ranks ``i < j``) are
compared at most once, and exactly when the first pivot drawn from
``{z_i, ..., z_j}`` is one of the two ends, which happens with probability
``2 / (j - i + 1)``. Summing over all pairs gives::

    E[C(n)] = sum over i < j of 2 / (j - i + 1) = 2 (n + 1) H_n - 4 n

with ``H_n`` the n-th harmonic number, about ``1.39 n log2 n``.
:func:`expected_comparisons` returns that value. An insertion-sort cutoff
(as in Week 2) would make the measured count stop matching the formula, so
there is none here. The formula assumes distinct keys, which is what the
benchmark uses; Lomuto's ``<=`` test sends every key equal to the pivot to
the left, so an all-equal input is O(n^2) under every pivot rule. Week 2's
three-way partition is the cure for that, and it is a different algorithm.

The spread is as important as the mean. The standard deviation of the
count grows like ``sqrt(7 - 2 pi^2 / 3) n``, about ``0.6483 n``
(:func:`comparison_std_asymptotic`), which is small next to the
``n log n`` mean, so large deviations from the expectation are rare.

Las Vegas, and why the default seed is not fixed
------------------------------------------------
:func:`randomized_quicksort` is a LAS VEGAS algorithm: its output is always
the correctly sorted list, and only its running time (the comparison
count) is random. With ``seed=None`` the random source comes from
:func:`~src.utils.random_utils.make_rng`, which seeds it from the operating
system on purpose. A fixed default seed would let an adversary who reads
the code replay every pivot and build an input that drives it to O(n^2),
exactly as sorted input does to the ``"last"`` rule. Experiments and tests
pass an explicit ``seed`` or ``rng`` so their runs are reproducible.

Recursion depth
---------------
Both sorts recurse into the *smaller* side of each partition and loop on
the larger, as Week 2 does. The recursive side is at most half the range,
so the stack stays at most about ``log2 n`` frames even when the partitions
are as lopsided as possible, and the ``"last"`` rule can sort 10,000 already
sorted items (about 50 million comparisons) without a ``RecursionError``.
The recursion limit is never raised.

The functions
-------------
========================== =================== ==================== ========
Function                   Expected time       Worst time           Space
========================== =================== ==================== ========
randomized_quicksort       O(n log n)          O(n^2), improbable   O(n)
deterministic_quicksort    O(n log n) typical  O(n^2), sorted input O(n)
expected_comparisons       O(n) big-int ops    O(n) big-int ops     O(n)
comparison_std_asymptotic  O(1)                O(1)                 O(1)
========================== =================== ==================== ========

The sorts' O(n) space is the one defensive copy of the input; the working
space on top of it is the O(log n) stack.

Examples:
    >>> data = [5, 2, 9, 1, 5, 6]
    >>> randomized_quicksort(data, seed=42)
    [1, 2, 5, 5, 6, 9]
    >>> data
    [5, 2, 9, 1, 5, 6]
    >>> deterministic_quicksort(list(range(10)), pivot="last",
    ...                         count_comparisons=True)
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 45)
    >>> expected_comparisons(3) == 8 / 3
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from src.sorting.basic_sorts import Comparable, _require_list
from src.utils.random_utils import make_rng

__all__ = [
    "comparison_std_asymptotic",
    "deterministic_quicksort",
    "expected_comparisons",
    "randomized_quicksort",
]

T = TypeVar("T", bound=Comparable)

#: A pivot rule: given the list and an inclusive range ``lo < hi``, return the
#: index in ``[lo, hi]`` of the element to partition around.
_PivotChooser = Callable[[List[Any], int, int], int]

#: Up to this n, :func:`expected_comparisons` sums ``H_n`` exactly.
_EXACT_LIMIT = 2000

#: The Euler-Mascheroni constant, for ``H_n ~ ln n + gamma + 1 / (2n)``.
_EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------
# Pivot rules
# ----------------------------------------------------------------------
def _last_pivot(arr: List[Any], lo: int, hi: int) -> int:
    """Choose the last element of the range, the CLRS QUICKSORT rule.

    Args:
        arr: The list being sorted. Not modified.
        lo: First index of the range, inclusive.
        hi: Last index of the range, inclusive.

    Returns:
        ``hi``.

    Raises:
        Nothing.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _last_pivot([4, 1, 3], 0, 2)
        2
    """
    return hi


def _median_of_three_pivot(arr: List[Any], lo: int, hi: int) -> int:
    """Choose the median of the first, middle and last elements of the range.

    Only ``<`` is used, so any mutually comparable type works. The tests
    made here are deliberately not counted as comparisons: the count
    measures partitioning work, the quantity the CLRS analysis is about.

    Args:
        arr: The list being sorted. Not modified.
        lo: First index of the range, inclusive.
        hi: Last index of the range, inclusive.

    Returns:
        Whichever of ``lo``, ``(lo + hi) // 2`` and ``hi`` holds the median
        of the three values.

    Raises:
        TypeError: If the three elements are not mutually comparable.

    Time Complexity:
        O(1) - at most three comparisons.

    Space Complexity:
        O(1).

    Examples:
        >>> _median_of_three_pivot([1, 2, 3, 4, 5], 0, 4)
        2
        >>> _median_of_three_pivot([9, 0, 5], 0, 2)
        2
        >>> _median_of_three_pivot([5, 0, 9], 0, 2)
        0
    """
    mid = (lo + hi) // 2
    first, middle, last = arr[lo], arr[mid], arr[hi]
    if first < middle:
        if middle < last:
            return mid
        return hi if first < last else lo
    if first < last:
        return lo
    return hi if middle < last else mid


def _random_pivot(rng: random.Random) -> _PivotChooser:
    """Build the CLRS RANDOM(lo, hi) rule: a uniform index in ``[lo, hi]``.

    Args:
        rng: The random source every draw goes through.

    Returns:
        A pivot rule that ignores the values and draws ``rng.randint(lo, hi)``.

    Raises:
        Nothing.

    Time Complexity:
        O(1) to build, O(1) per draw.

    Space Complexity:
        O(1).

    Examples:
        >>> choose = _random_pivot(make_rng(7))
        >>> all(3 <= choose([], 3, 9) <= 9 for _ in range(50))
        True
    """
    randint = rng.randint

    def choose(arr: List[Any], lo: int, hi: int) -> int:
        return randint(lo, hi)

    return choose


#: The deterministic rules :func:`deterministic_quicksort` accepts, by name.
_DETERMINISTIC_RULES: Dict[str, _PivotChooser] = {
    "last": _last_pivot,
    "median_of_three": _median_of_three_pivot,
}


# ----------------------------------------------------------------------
# The shared implementation
# ----------------------------------------------------------------------
def _quicksort_range(
    arr: List[Any], lo: int, hi: int, choose_pivot: _PivotChooser
) -> int:
    """Sort ``arr[lo..hi]`` in place and return the comparisons it made.

    The one implementation both public sorts run. Each round asks
    ``choose_pivot`` for an index, swaps that element to ``hi`` and runs
    Lomuto's partition, then recurses into the smaller side and loops on the
    larger. The loop body is written for speed in pure Python: the pivot and
    the scanned element are locals, the swap is two plain assignments, and
    the ``s - 1`` comparisons of a partition are added in one step after the
    loop instead of being counted one at a time inside it.

    Args:
        arr: The list being sorted. Modified in place.
        lo: First index of the range, inclusive.
        hi: Last index of the range, inclusive. A range with ``hi <= lo``
            is already sorted.
        choose_pivot: The pivot rule. It is the only thing that differs
            between the randomized and deterministic sorts.

    Returns:
        The number of element-vs-pivot comparisons made, which is the sum
        of ``s - 1`` over every partitioned subarray of size ``s``.

    Raises:
        TypeError: If the elements are not mutually comparable.

    Time Complexity:
        Best:    O(n log n) - every pivot splits its range evenly
        Average: O(n log n)
        Worst:   O(n^2)     - every pivot is the range's maximum or minimum

    Space Complexity:
        O(log n) - the smaller-side recursion is at most half the range.

    Examples:
        >>> data = [3, 1, 2]
        >>> _quicksort_range(data, 0, 2, _last_pivot), data
        (2, [1, 2, 3])
        >>> _quicksort_range([7], 0, 0, _last_pivot)
        0
    """
    comparisons = 0
    while lo < hi:
        chosen = choose_pivot(arr, lo, hi)
        if chosen != hi:
            arr[chosen], arr[hi] = arr[hi], arr[chosen]
        pivot = arr[hi]

        # Lomuto: arr[lo..store-1] <= pivot, arr[store..scan-1] > pivot.
        store = lo
        for scan in range(lo, hi):
            item = arr[scan]
            if item <= pivot:
                arr[scan] = arr[store]
                arr[store] = item
                store += 1
        arr[hi] = arr[store]
        arr[store] = pivot
        comparisons += hi - lo

        # The pivot is final at ``store``. Recurse on the smaller side so the
        # stack stays O(log n); loop on the larger.
        if store - lo < hi - store:
            comparisons += _quicksort_range(arr, lo, store - 1, choose_pivot)
            lo = store + 1
        else:
            comparisons += _quicksort_range(arr, store + 1, hi, choose_pivot)
            hi = store - 1
    return comparisons


def _run(
    arr: List[T], choose_pivot: _PivotChooser, count_comparisons: bool
) -> Union[List[T], Tuple[List[T], int]]:
    """Copy ``arr`` once, sort the copy in place and shape the result.

    Args:
        arr: The caller's list, already checked to be a list. Not modified.
        choose_pivot: The pivot rule to sort with.
        count_comparisons: Whether to return the comparison count as well.

    Returns:
        The sorted copy, or ``(sorted_copy, comparisons)`` when
        ``count_comparisons`` is true.

    Raises:
        TypeError: If the elements are not mutually comparable.

    Time Complexity:
        That of :func:`_quicksort_range`, plus O(n) for the copy.

    Space Complexity:
        O(n) for the copy, plus the O(log n) stack.

    Examples:
        >>> _run([2, 1], _last_pivot, True)
        ([1, 2], 1)
        >>> _run([], _last_pivot, False)
        []
    """
    result = list(arr)
    comparisons = 0
    if len(result) > 1:
        comparisons = _quicksort_range(result, 0, len(result) - 1, choose_pivot)
    if count_comparisons:
        return result, comparisons
    return result


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def randomized_quicksort(
    arr: List[T],
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    count_comparisons: bool = False,
) -> Union[List[T], Tuple[List[T], int]]:
    """Sort a list with CLRS RANDOMIZED-QUICKSORT (random pivot, Lomuto).

    Before every partition a uniformly random index in ``[lo, hi]`` is
    drawn and swapped to ``hi``; Lomuto's partition then places it. The
    caller's list is copied once and the copy is partitioned in place, with
    no insertion-sort cutoff, so the comparison count is exactly the
    quantity :func:`expected_comparisons` predicts.

    This is a LAS VEGAS algorithm: the returned list is always correctly
    sorted, whatever the draws, and only the running time is random. Its
    expected number of comparisons on n distinct keys is
    ``2 (n + 1) H_n - 4 n`` for every input, so sorted input is no worse
    than any other.

    Args:
        arr: A list of mutually comparable elements. Not modified.
        seed: Seed for the pivot draws, used only when ``rng`` is None.
            ``None`` gives an operating-system-seeded source, so an
            adversary cannot predict the pivots; experiments and tests pass
            an explicit seed so their runs repeat exactly.
        rng: A :class:`random.Random` to draw pivots from. It takes
            precedence over ``seed``, which lets one source drive many
            calls.
        count_comparisons: Also return the number of element-vs-pivot
            comparisons, ``s - 1`` per partitioned subarray of size ``s``.

    Returns:
        A new list with the same elements in ascending order, or
        ``(sorted_list, comparisons)`` when ``count_comparisons`` is true.

    Raises:
        TypeError: If ``arr`` is not a list, if ``seed`` is not an int
            (``bool`` rejected), or from the comparisons themselves if the
            elements are not mutually comparable.

    Time Complexity:
        Best:    O(n log n)
        Expected: O(n log n) - about 1.39 n log2 n comparisons on any input
                               of distinct keys
        Worst:   O(n^2)     - an unlucky run of draws; on distinct keys the
                              chance of exceeding the mean by a constant
                              factor shrinks rapidly with n. An all-equal
                              input is O(n^2) on every run (Lomuto's ``<=``)

    Space Complexity:
        O(n) for the defensive copy, plus an O(log n) stack from the
        smaller-side recursion.

    Examples:
        A fixed seed makes a run repeatable, comparisons included:

        >>> randomized_quicksort([5, 2, 9, 1, 5, 6], seed=42)
        [1, 2, 5, 5, 6, 9]
        >>> data = [8, 3, 5, 1, 9, 2, 7]
        >>> first = randomized_quicksort(data, seed=7, count_comparisons=True)
        >>> first
        ([1, 2, 3, 5, 7, 8, 9], 11)
        >>> randomized_quicksort(data, seed=7, count_comparisons=True) == first
        True

        An explicit ``rng`` takes precedence over ``seed``:

        >>> randomized_quicksort(data, seed=1, rng=make_rng(7),
        ...                      count_comparisons=True) == first
        True

        The caller's list is never touched, and any comparable type works:

        >>> data
        [8, 3, 5, 1, 9, 2, 7]
        >>> randomized_quicksort(["pear", "apple", "fig"], seed=3)
        ['apple', 'fig', 'pear']

        Already-sorted input is not special to it:

        >>> randomized_quicksort(list(range(10)), seed=42,
        ...                      count_comparisons=True)[1] < 45
        True

        Empty and single-element lists come back as copies:

        >>> single = [4]
        >>> result = randomized_quicksort(single, seed=0)
        >>> result, result is single
        ([4], False)
        >>> randomized_quicksort([], seed=0, count_comparisons=True)
        ([], 0)
        >>> randomized_quicksort((3, 1), seed=0)
        Traceback (most recent call last):
            ...
        TypeError: randomized_quicksort() expects a list, got tuple
    """
    _require_list(arr, "randomized_quicksort")
    source = rng if rng is not None else make_rng(seed)
    return _run(arr, _random_pivot(source), count_comparisons)


def deterministic_quicksort(
    arr: List[T],
    pivot: str = "last",
    count_comparisons: bool = False,
) -> Union[List[T], Tuple[List[T], int]]:
    """Sort a list with QuickSort and a fixed pivot rule, the baseline.

    The same code path as :func:`randomized_quicksort` with the random draw
    replaced by a fixed rule, so a benchmark of the two isolates the pivot
    choice. It is not randomized at all: the same input always takes the
    same path and makes the same number of comparisons.

    * ``"last"`` is CLRS QUICKSORT: the last element of each range. On
      already-sorted input every pivot is the range's maximum, so each
      partition peels off one element and the sort makes exactly
      ``n (n - 1) / 2`` comparisons. That is the adversarial input the
      randomized version is immune to.
    * ``"median_of_three"`` takes the median of the first, middle and last
      elements and swaps it to ``hi``. Sorted input becomes its best case;
      the comparisons used to pick the median are not counted.

    Args:
        arr: A list of mutually comparable elements. Not modified.
        pivot: ``"last"`` or ``"median_of_three"``.
        count_comparisons: Also return the number of element-vs-pivot
            comparisons, ``s - 1`` per partitioned subarray of size ``s``.

    Returns:
        A new list with the same elements in ascending order, or
        ``(sorted_list, comparisons)`` when ``count_comparisons`` is true.

    Raises:
        TypeError: If ``arr`` is not a list, or from the comparisons
            themselves if the elements are not mutually comparable.
        ValueError: If ``pivot`` is not one of the two rules.

    Time Complexity:
        Best:    O(n log n) - e.g. ``"median_of_three"`` on sorted input
        Average: O(n log n) - over random inputs
        Worst:   O(n^2)     - ``"last"`` on sorted or reverse-sorted input,
                              either rule on an all-equal input

    Space Complexity:
        O(n) for the defensive copy, plus an O(log n) stack from the
        smaller-side recursion, even in the O(n^2) case.

    Examples:
        >>> deterministic_quicksort([3, 1, 2])
        [1, 2, 3]
        >>> deterministic_quicksort([5, 1, 4, 1, 3], pivot="median_of_three")
        [1, 1, 3, 4, 5]

        Sorted input is the worst case of ``"last"``, exactly n(n - 1) / 2
        comparisons, and the best case of ``"median_of_three"``:

        >>> deterministic_quicksort(list(range(10)), count_comparisons=True)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 45)
        >>> deterministic_quicksort(list(range(10)), pivot="median_of_three",
        ...                         count_comparisons=True)[1]
        19

        Even that quadratic case keeps the stack shallow:

        >>> deterministic_quicksort(list(range(2000)),
        ...                         count_comparisons=True)[1] == 2000 * 1999 // 2
        True

        The caller's list is never touched:

        >>> original = [3, 1, 2]
        >>> _ = deterministic_quicksort(original, pivot="median_of_three")
        >>> original
        [3, 1, 2]
        >>> deterministic_quicksort([1], pivot="random")
        Traceback (most recent call last):
            ...
        ValueError: pivot must be 'last' or 'median_of_three', got 'random'
    """
    _require_list(arr, "deterministic_quicksort")
    choose_pivot = _DETERMINISTIC_RULES.get(pivot) if isinstance(pivot, str) else None
    if choose_pivot is None:
        raise ValueError(
            f"pivot must be 'last' or 'median_of_three', got {pivot!r}"
        )
    return _run(arr, choose_pivot, count_comparisons)


def _require_size(n: Any, func_name: str) -> int:
    """Check that ``n`` is a non-negative int (``bool`` rejected).

    Args:
        n: The value to check.
        func_name: The calling function, for the error message.

    Returns:
        ``n`` unchanged.

    Raises:
        TypeError: If ``n`` is not an int, or is a bool.
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_size(5, "demo")
        5
        >>> _require_size(True, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects n to be an int, got bool
        >>> _require_size(-1, "demo")
        Traceback (most recent call last):
            ...
        ValueError: demo() expects n >= 0, got -1
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{func_name}() expects n to be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"{func_name}() expects n >= 0, got {n}")
    return n


def expected_comparisons(n: int) -> float:
    """Return the expected comparisons of randomized QuickSort on n distinct keys.

    The exact CLRS value ``2 (n + 1) H_n - 4 n``, where ``H_n`` is the n-th
    harmonic number. It comes from summing, over every pair of ranks
    ``i < j``, the probability ``2 / (j - i + 1)`` that the pair is ever
    compared (see the module docstring). It counts comparisons the way
    :func:`randomized_quicksort` does, ``s - 1`` per partition.

    For ``n <= 2000`` the harmonic number is summed exactly as a rational
    (over the least common multiple of ``1 .. n``) and the answer is
    rounded once at the end. Above that it uses
    ``H_n ~ ln n + gamma + 1 / (2n)``, whose error is about
    ``1 / (12 n^2)``: under ``2e-4`` comparisons in the result.

    Args:
        n: The number of keys.

    Returns:
        ``E[C(n)]`` as a float. 0.0 for ``n`` of 0 or 1.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected).
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(n) big-integer operations for ``n <= 2000``, O(1) above.

    Space Complexity:
        O(n) bits for the exact rational, O(1) above.

    Examples:
        >>> [expected_comparisons(n) for n in (0, 1, 2)]
        [0.0, 0.0, 1.0]
        >>> expected_comparisons(3) == 8 / 3
        True
        >>> expected_comparisons(4) == 29 / 6, expected_comparisons(5) == 37 / 5
        (True, True)
        >>> expected_comparisons(10) == 30791 / 1260
        True
        >>> round(expected_comparisons(1000), 2)
        10985.91
        >>> abs(expected_comparisons(2001) - expected_comparisons(2000)) < 25
        True
    """
    _require_size(n, "expected_comparisons")
    if n <= _EXACT_LIMIT:
        denominator = math.lcm(*range(1, n + 1))
        numerator = sum(denominator // k for k in range(1, n + 1))
        # E = 2 (n + 1) H_n - 4 n with H_n = numerator / denominator, kept
        # exact until the single division at the end.
        exact = 2 * (n + 1) * numerator - 4 * n * denominator
        return exact / denominator
    harmonic = math.log(n) + _EULER_GAMMA + 1.0 / (2 * n)
    return 2.0 * (n + 1) * harmonic - 4.0 * n


def comparison_std_asymptotic(n: int) -> float:
    """Return the asymptotic standard deviation of randomized QuickSort's count.

    The variance of the comparison count on n distinct keys is
    ``(7 - 2 pi^2 / 3) n^2`` to leading order, so the standard deviation is
    ``sqrt(7 - 2 pi^2 / 3) n``, about ``0.6483 n``. It grows like ``n``
    while the mean grows like ``n log n``, so the count concentrates around
    its mean as n grows. It is a leading-order figure, meant for comparison
    with the spread measured over many seeds at large n.

    Args:
        n: The number of keys.

    Returns:
        ``sqrt(7 - 2 pi^2 / 3) * n`` as a float.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected).
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> round(comparison_std_asymptotic(1), 4)
        0.6483
        >>> round(comparison_std_asymptotic(1000), 1)
        648.3
        >>> comparison_std_asymptotic(0)
        0.0
    """
    _require_size(n, "comparison_std_asymptotic")
    return math.sqrt(7.0 - 2.0 * math.pi**2 / 3.0) * n
