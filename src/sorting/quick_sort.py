"""QuickSort with randomized pivot selection and the standard optimizations.

QuickSort is divide-and-conquer run in the opposite order from merge sort
(:mod:`src.sorting.merge_sort`). Merge sort splits trivially and does its
real work combining; quicksort does its real work *partitioning* and then
combines trivially — once the partition is done, the pieces are already in
the right order relative to each other. The consequence is the whole story
of this module: because the split is value-based rather than positional,
the input can knock it off balance, and a bad split costs O(n^2).

Three optimizations, all named in the assignment, plus one structural
choice that matters more than any of them:

1. **Randomized pivot** (:func:`partition`) — the pivot is chosen
   uniformly at random from the current range, so no fixed input pattern
   is adversarial. The worst case stops being "sorted input" and becomes
   "an unlucky sequence of draws".
2. **Insertion-sort cutoff** (:data:`INSERTION_SORT_CUTOFF`) — ranges of
   10 or fewer elements are handed to the Week 1 insertion sort, which
   beats quicksort at that size because it has no partitioning overhead.
3. **Three-way partitioning** (:func:`partition_three_way`), optional via
   ``three_way=True`` — the Dutch national flag split into ``< = >``,
   which finishes an all-equal range in a single linear pass.

The structural choice is **recursing into the smaller partition and
looping on the larger** (see :func:`_quick_sort_in_place`). Python's
default recursion limit is 1000, and a textbook quicksort recurses to
depth n on its bad cases, so it dies with a RecursionError somewhere
around n=1000 — well below the n=50,000 this project benchmarks. Handling
the larger side iteratively caps the stack at O(log n) for *every* input,
which is a correctness fix, not a tuning knob. Raising the recursion limit
instead would be papering over the problem.

**On the choice of Hoare over Lomuto.** :func:`partition` implements
Hoare's scheme. Lomuto's is the one usually taught, and it has a
catastrophic weakness: on a range of equal elements every element lands on
one side, giving O(n^2). Hoare's pointers both stop on equal elements and
swap past each other, which splits an all-equal range down the middle
instead. Since two of this project's six required data types have only 10
and 3 distinct values, that difference is decisive.
:func:`partition_lomuto` and :func:`quick_sort_lomuto` are included so the
report can *measure* that claim rather than assert it; they are reference
implementations for comparison, not the production path.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple, TypeVar

from src.sorting.basic_sorts import Comparable, _require_list, insertion_sort

T = TypeVar("T", bound=Comparable)

__all__ = [
    "quick_sort",
    "quick_sort_lomuto",
    "partition",
    "partition_three_way",
    "partition_lomuto",
    "INSERTION_SORT_CUTOFF",
]

#: Ranges of this size or smaller are finished with insertion sort instead
#: of being partitioned further. Insertion sort has a much smaller constant
#: factor than quicksort's partitioning machinery, so below roughly this
#: width it wins despite being asymptotically worse. The report measures
#: the effect of varying this value.
INSERTION_SORT_CUTOFF = 10


# ----------------------------------------------------------------------
# Partition schemes
# ----------------------------------------------------------------------
def partition(arr: List[T], low: int, high: int, rng: random.Random) -> int:
    """Partition ``arr[low..high]`` in place around a random pivot (Hoare).

    A random element is swapped into ``arr[low]`` and used as the pivot,
    then two pointers walk inwards from both ends, swapping any pair that
    is on the wrong side. They meet at the split point.

    Swapping the pivot to ``low`` is not cosmetic. Hoare's scheme is only
    guaranteed to return an index strictly less than ``high`` when the
    pivot is the leftmost element; using a random *position* as the pivot
    without moving it can return ``high``, which makes the caller recurse
    on the identical range forever. Moving it first keeps the randomization
    and the termination guarantee.

    Args:
        arr: The list being sorted. Modified in place.
        low: First index of the range, inclusive.
        high: Last index of the range, inclusive. Must exceed ``low``.
        rng: Random source, so a seeded run is reproducible.

    Returns:
        An index ``j`` with ``low <= j < high`` such that every element of
        ``arr[low..j]`` is less than or equal to every element of
        ``arr[j+1..high]``. Note that unlike Lomuto's scheme this does
        *not* place the pivot at its final position, so the caller must
        recurse on ``low..j`` and ``j+1..high`` — not ``j-1`` and ``j+1``.

    Time Complexity:
        O(high - low + 1) - each pointer traverses the range once.

    Space Complexity:
        O(1) - swaps happen in place.

    Examples:
        >>> data = [5, 3, 8, 1, 9, 2]
        >>> split = partition(data, 0, 5, random.Random(0))
        >>> 0 <= split < 5
        True
        >>> max(data[: split + 1]) <= min(data[split + 1 :])
        True

        An all-equal range splits near the middle rather than degenerating:

        >>> equal = [7] * 8
        >>> partition(equal, 0, 7, random.Random(1))
        3
    """
    pivot_index = rng.randint(low, high)
    arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
    pivot = arr[low]

    i = low - 1
    j = high + 1

    while True:
        # Advance from the left past everything already smaller than the
        # pivot. Stops at arr[low] == pivot on the first pass, so i can
        # never run off the end.
        i += 1
        while arr[i] < pivot:
            i += 1

        # And from the right past everything already larger.
        j -= 1
        while arr[j] > pivot:
            j -= 1

        if i >= j:
            return j

        arr[i], arr[j] = arr[j], arr[i]


def partition_three_way(
    arr: List[T], low: int, high: int, rng: random.Random
) -> Tuple[int, int]:
    """Partition ``arr[low..high]`` into ``< pivot``, ``== pivot``, ``> pivot``.

    Dijkstra's Dutch national flag partition. One pass maintains four
    regions — everything below ``lt`` is smaller than the pivot, everything
    from ``lt`` to ``i`` equals it, everything above ``gt`` is larger, and
    the span between ``i`` and ``gt`` is still unexamined.

    The payoff is that every element equal to the pivot reaches its final
    position in this single pass and is never looked at again. A range of
    k distinct values therefore costs O(n log k) rather than O(n log n),
    and an all-equal range costs O(n) with no recursion at all.

    Args:
        arr: The list being sorted. Modified in place.
        low: First index of the range, inclusive.
        high: Last index of the range, inclusive.
        rng: Random source, so a seeded run is reproducible.

    Returns:
        ``(lt, gt)`` — the inclusive bounds of the equal-to-pivot band.
        Everything in ``arr[lt..gt]`` is final, so the caller recurses only
        on ``low..lt-1`` and ``gt+1..high``.

    Time Complexity:
        O(high - low + 1) - ``i`` and ``gt`` move towards each other and
        the loop ends when they cross.

    Space Complexity:
        O(1).

    Examples:
        >>> data = [3, 1, 3, 5, 3, 0]
        >>> lt, gt = partition_three_way(data, 0, 5, random.Random(2))
        >>> sorted(data[:lt]), sorted(data[gt + 1 :])
        ([0, 1], [5])
        >>> data[lt : gt + 1]
        [3, 3, 3]

        An all-equal range is finished outright — the whole span is the
        equal band, so there is nothing left to recurse on:

        >>> equal = [4] * 6
        >>> partition_three_way(equal, 0, 5, random.Random(3))
        (0, 5)
    """
    pivot_index = rng.randint(low, high)
    arr[low], arr[pivot_index] = arr[pivot_index], arr[low]
    pivot = arr[low]

    lt = low       # arr[low..lt-1]  < pivot
    i = low + 1    # arr[lt..i-1]   == pivot
    gt = high      # arr[gt+1..high] > pivot

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            # The element swapped down from gt has not been examined yet,
            # so i deliberately does not advance here.
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    return lt, gt


def partition_lomuto(arr: List[T], low: int, high: int, rng: random.Random) -> int:
    """Partition around a random pivot using Lomuto's scheme.

    Provided for comparison only — :func:`partition` is the scheme this
    module actually sorts with. Lomuto's is included so the report can
    measure, rather than merely assert, its well-known weakness: the test
    is ``<= pivot``, so on a range of equal elements *every* element is
    swapped into the left side and the split lands at ``high``. That turns
    duplicate-heavy input into the O(n^2) worst case, which is exactly the
    situation this project's ``many_duplicates`` and ``few_unique`` data
    types create.

    Args:
        arr: The list being sorted. Modified in place.
        low: First index of the range, inclusive.
        high: Last index of the range, inclusive.
        rng: Random source.

    Returns:
        The pivot's final index ``p``; ``arr[p]`` is in its finished
        position, so the caller recurses on ``low..p-1`` and ``p+1..high``.

    Time Complexity:
        O(high - low + 1).

    Space Complexity:
        O(1).

    Examples:
        >>> data = [5, 3, 8, 1]
        >>> p = partition_lomuto(data, 0, 3, random.Random(0))
        >>> all(v <= data[p] for v in data[:p])
        True
        >>> all(v >= data[p] for v in data[p + 1 :])
        True

        The degenerate case, shown rather than described — every element
        equal to the pivot lands left of it, so the split is maximally
        lopsided:

        >>> equal = [4] * 8
        >>> partition_lomuto(equal, 0, 7, random.Random(0))
        7
    """
    pivot_index = rng.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    pivot = arr[high]

    boundary = low - 1
    for scan in range(low, high):
        if arr[scan] <= pivot:
            boundary += 1
            arr[boundary], arr[scan] = arr[scan], arr[boundary]

    arr[boundary + 1], arr[high] = arr[high], arr[boundary + 1]
    return boundary + 1


# ----------------------------------------------------------------------
# The recursive driver
# ----------------------------------------------------------------------
def _insertion_sort_range(arr: List[T], low: int, high: int) -> None:
    """Sort ``arr[low..high]`` in place with the Week 1 insertion sort.

    Reuses :func:`src.sorting.basic_sorts.insertion_sort` rather than
    carrying a second copy of the algorithm. That function is
    non-destructive, so the range is sliced out, sorted and written back.
    The slice allocation is bounded by :data:`INSERTION_SORT_CUTOFF`
    elements, so it is O(1) work per call regardless of the input size.
    """
    arr[low : high + 1] = insertion_sort(arr[low : high + 1])


def _quick_sort_in_place(
    arr: List[T], low: int, high: int, rng: random.Random, three_way: bool
) -> None:
    """Sort ``arr[low..high]`` in place.

    The loop is the point. A textbook quicksort recurses twice; this
    version recurses into the *smaller* partition and then reassigns
    ``low``/``high`` to iterate on the larger one. Since the recursive
    branch always covers at most half the range, the stack can never grow
    deeper than log2(n) — about 17 frames at n=100,000 — no matter how
    badly the partitions are balanced. Recursing on both sides would reach
    Python's 1000-frame limit on adversarial input well before the
    n=50,000 this project benchmarks.
    """
    while low < high:
        # Small ranges: hand off to insertion sort and stop.
        if high - low + 1 <= INSERTION_SORT_CUTOFF:
            _insertion_sort_range(arr, low, high)
            return

        if three_way:
            lt, gt = partition_three_way(arr, low, high, rng)
            # arr[lt..gt] is already final; only the flanks remain.
            if (lt - low) < (high - gt):
                _quick_sort_in_place(arr, low, lt - 1, rng, three_way)
                low = gt + 1
            else:
                _quick_sort_in_place(arr, gt + 1, high, rng, three_way)
                high = lt - 1
        else:
            # Hoare's split point belongs to the left side, so the ranges
            # are low..split and split+1..high.
            split = partition(arr, low, high, rng)
            if (split - low) < (high - split - 1):
                _quick_sort_in_place(arr, low, split, rng, three_way)
                low = split + 1
            else:
                _quick_sort_in_place(arr, split + 1, high, rng, three_way)
                high = split


def _quick_sort_lomuto_in_place(
    arr: List[T], low: int, high: int, rng: random.Random
) -> None:
    """Lomuto-scheme driver, for the report's partition-scheme comparison."""
    while low < high:
        if high - low + 1 <= INSERTION_SORT_CUTOFF:
            _insertion_sort_range(arr, low, high)
            return

        pivot_position = partition_lomuto(arr, low, high, rng)
        # Same smaller-side recursion, so that when this scheme performs
        # badly it is the partition quality being measured and not a
        # RecursionError.
        if (pivot_position - low) < (high - pivot_position):
            _quick_sort_lomuto_in_place(arr, low, pivot_position - 1, rng)
            low = pivot_position + 1
        else:
            _quick_sort_lomuto_in_place(arr, pivot_position + 1, high, rng)
            high = pivot_position - 1


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def quick_sort(
    arr: List[T], seed: Optional[int] = None, three_way: bool = False
) -> List[T]:
    """Sort a list in ascending order using randomized quicksort.

    Resolves the assignment's "in place" / "returns a new list" pairing the
    only way both can hold at once: the caller's list is copied once, and
    all partitioning then happens genuinely in place within that copy.
    There is no per-level list allocation and no
    ``[x for x in arr if x < pivot]`` three-list pattern — that would
    allocate O(n log n) memory and destroy the O(log n) space bound this
    algorithm is chosen for.

    Args:
        arr: A list of comparable elements. Not modified.
        seed: Seed for the pivot-selection random source. Pass one for a
            reproducible run — benchmarks need this, since otherwise two
            runs on identical input do different amounts of work. ``None``
            leaves pivot selection genuinely random.
        three_way: Use Dutch-national-flag partitioning instead of Hoare's
            two-way split. Costs slightly more per partition and pays for
            itself many times over when the input has few distinct values,
            because every element equal to a pivot is finished in one pass.

    Returns:
        A new list containing the same elements in ascending order.

    Raises:
        TypeError: If arr is not a list. TypeError also propagates from the
            comparison itself if the elements are not mutually comparable.

    Time Complexity:
        Best:    O(n log n)   - balanced partitions
        Average: O(n log n)   - expected, with randomized pivot
        Worst:   O(n^2)       - all elements to one side; randomization
                                makes this vanishingly unlikely rather than
                                impossible

    Space Complexity:
        O(log n) - recursion stack only, when recursing into the smaller
        side. The one O(n) allocation is the defensive copy of the input,
        which the non-destructive contract requires and which no amount of
        partitioning strategy can avoid.

    Stability:
        Unstable - partitioning swaps non-adjacent elements.

    Examples:
        >>> quick_sort([3, 1, 2])
        [1, 2, 3]
        >>> quick_sort([])
        []
        >>> quick_sort([5])
        [5]
        >>> quick_sort([-2, 7, 0, -9, 7])
        [-9, -2, 0, 7, 7]
        >>> quick_sort(["pear", "apple", "fig"])
        ['apple', 'fig', 'pear']

        Both partition schemes agree on the answer; they differ only in how
        much work they do getting there:

        >>> data = [5, 1, 5, 3, 5, 2, 5, 4, 5, 0, 5, 3, 1]
        >>> quick_sort(data, three_way=False) == sorted(data)
        True
        >>> quick_sort(data, three_way=True) == sorted(data)
        True

        A seed makes a run reproducible:

        >>> quick_sort([9, 2, 7, 4], seed=1) == quick_sort([9, 2, 7, 4], seed=1)
        True

        The caller's list is left alone:

        >>> original = [3, 1, 2]
        >>> _ = quick_sort(original)
        >>> original
        [3, 1, 2]

        >>> quick_sort("not a list")
        Traceback (most recent call last):
            ...
        TypeError: quick_sort() expects a list, got str
    """
    _require_list(arr, "quick_sort")

    result = list(arr)
    if len(result) < 2:
        return result

    _quick_sort_in_place(result, 0, len(result) - 1, random.Random(seed), three_way)
    return result


def quick_sort_lomuto(arr: List[T], seed: Optional[int] = None) -> List[T]:
    """Sort using Lomuto partitioning — for comparison, not for production.

    Identical in every respect to :func:`quick_sort` except the partition
    scheme, so a benchmark of the two isolates that one variable. See
    :func:`partition_lomuto` for why this one degrades on duplicate-heavy
    input.

    Args:
        arr: A list of comparable elements. Not modified.
        seed: Seed for pivot selection.

    Returns:
        A new list containing the same elements in ascending order.

    Raises:
        TypeError: If arr is not a list.

    Time Complexity:
        Best:    O(n log n)
        Average: O(n log n)
        Worst:   O(n^2)   - reached whenever many elements equal the pivot,
                            which random pivot selection does not help with

    Space Complexity:
        O(log n) recursion stack, plus the O(n) defensive copy.

    Stability:
        Unstable.

    Examples:
        >>> quick_sort_lomuto([3, 1, 2])
        [1, 2, 3]
        >>> quick_sort_lomuto([])
        []
        >>> data = [4, 4, 4, 1, 4, 9, 4]
        >>> quick_sort_lomuto(data) == sorted(data)
        True
    """
    _require_list(arr, "quick_sort_lomuto")

    result = list(arr)
    if len(result) < 2:
        return result

    _quick_sort_lomuto_in_place(result, 0, len(result) - 1, random.Random(seed))
    return result
