"""Selection in linear time: randomized QuickSelect and median of medians.

The selection problem asks for the k-th smallest element of an unsorted
list. Sorting answers it in O(n log n), but a sort orders every element
and selection needs only one of them in its final position. Both
algorithms here partition around a pivot, as quicksort does, and then
continue into *one* side only, the side that must contain position ``k``.
That single change turns quicksort's ``n + 2(n/2) + 4(n/4) + ...`` into
``n + n/2 + n/4 + ...``, a geometric series bounded by ``2n`` when the
splits are balanced (CLRS sections 9.2 and 9.3).

``k`` is 1-indexed throughout: ``k = 1`` is the minimum and
``k = len(arr)`` the maximum, so :func:`median` asks for
``k = (n + 1) // 2``, the lower median.

Two ways to get a good pivot
----------------------------
* :func:`randomized_select` draws the pivot uniformly at random. No input
  is bad for it; only an unlucky run of draws is, and that costs O(n^2).
  Its expected comparison count is about ``2n`` for the minimum or
  maximum and about ``2(1 + ln 2)n``, roughly ``3.39n``, for the median
  (Knuth's exact analysis), so it is O(n) in expectation for every ``k``.
* :func:`deterministic_select` is CLRS SELECT: split the list into groups
  of 5, take each group's median, and find the median of those medians
  recursively. That pivot is guaranteed to have at least about ``3n/10``
  elements on each side, so every round discards a constant fraction and
  the worst case is O(n). The price is a large constant: every round pays
  for sorting ``n/5`` groups and for a recursive call on ``n/5`` medians.

Las Vegas, and what "success probability" means for it
-------------------------------------------------------
:func:`randomized_select` is a LAS VEGAS algorithm: it is always correct,
and only its running time is random. Its probability of returning the
right answer is therefore exactly 1, for every input, every ``k`` and
every seed. Recording it is a check, not a measurement; there is no
failure mode to estimate. The probability that actually carries
information is how often a run finishes within ``4n`` comparisons, a
threshold comfortably above the largest expected count (``3.39n`` at the
median), together with the full distribution of comparisons divided by
``n``. The benchmark records both.

The three-way split
-------------------
Both algorithms split each active range three ways, into ``< pivot``,
``== pivot`` and ``> pivot`` (Dijkstra's Dutch national flag). With a
two-way split a list of equal values never shrinks: every element lands
on the same side as the pivot and each round removes only the pivot,
which is quadratic. With three ways, if ``k`` falls in the equal block the
pivot is the answer and the run stops, so 1,000 identical values take one
round, and any range the loop continues into holds only values strictly
smaller or strictly larger than the pivot.

How work is counted
-------------------
:class:`SelectionStats` reports ``depth``, the number of partition rounds,
and ``comparisons``, one per non-pivot element classified in each round.
A round on ``s`` elements therefore costs exactly ``s - 1``, the same
convention as one element-vs-pivot test per element in CLRS PARTITION,
even though the three-way loop may test an element against the pivot
twice (``<`` and then ``>``). A range of one element is not partitioned.
For :func:`deterministic_select` both numbers describe the main select
only: the group sorting and the recursive median-of-medians call that
choose each pivot are not counted, exactly as the comparisons that choose
a median-of-three pivot are not counted in the quicksort module. Their
cost shows up in the measured running time instead.

Examples:
    >>> data = [7, 10, 4, 3, 20, 15]
    >>> randomized_select(data, 3, seed=42), deterministic_select(data, 3)
    (7, 7)
    >>> median(data, seed=0)
    7

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import Any, List, NamedTuple, Optional, Tuple, TypeVar, Union

from src.utils.random_utils import make_rng

__all__ = [
    "SelectionStats",
    "deterministic_select",
    "median",
    "randomized_select",
]

T = TypeVar("T")

#: Group size for the median of medians. 5 is the smallest odd size for
#: which the recurrence ``T(n) <= T(n/5) + T(7n/10) + O(n)`` has
#: ``1/5 + 7/10 < 1`` and so solves to O(n).
_GROUP = 5


class SelectionStats(NamedTuple):
    """Work done by one selection call.

    Attributes:
        depth: Number of partition rounds. For the iterative
            :func:`randomized_select` this is the recursion depth the
            textbook recursive version would reach; for
            :func:`deterministic_select` it counts the main select only.
        comparisons: One per non-pivot element classified in each round,
            so a round on ``s`` elements adds ``s - 1``.

    Examples:
        >>> stats = SelectionStats(depth=2, comparisons=7)
        >>> stats.depth, stats.comparisons
        (2, 7)
        >>> stats
        SelectionStats(depth=2, comparisons=7)
    """

    depth: int
    comparisons: int


def _validate(arr: Any, k: Any, func_name: str) -> int:
    """Check the shared arguments and return ``k`` as a 0-based index.

    Args:
        arr: The object that should be a list to select from.
        k: The 1-based rank requested.
        func_name: Name of the calling function, for the error messages.

    Returns:
        ``k - 1``.

    Raises:
        TypeError: If ``arr`` is not a list, or ``k`` is not an int.
            ``bool`` is rejected, because ``True`` is an accident, not a
            rank.
        ValueError: If ``arr`` is empty, or ``k`` is outside
            ``1 .. len(arr)``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _validate([5, 1, 3], 3, "demo")
        2
        >>> _validate((5, 1), 1, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects a list, got tuple
        >>> _validate([5, 1], True, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects an int k, got bool
        >>> _validate([], 1, "demo")
        Traceback (most recent call last):
            ...
        ValueError: demo() cannot select from an empty list
        >>> _validate([5, 1], 3, "demo")
        Traceback (most recent call last):
            ...
        ValueError: demo() needs 1 <= k <= 2, got k=3
    """
    if not isinstance(arr, list):
        raise TypeError(f"{func_name}() expects a list, got {type(arr).__name__}")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"{func_name}() expects an int k, got {type(k).__name__}")
    if not arr:
        raise ValueError(f"{func_name}() cannot select from an empty list")
    if not 1 <= k <= len(arr):
        raise ValueError(f"{func_name}() needs 1 <= k <= {len(arr)}, got k={k}")
    return k - 1


def randomized_select(
    arr: List[T],
    k: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    return_stats: bool = False,
) -> Union[T, Tuple[T, SelectionStats]]:
    """Return the k-th smallest element using a uniformly random pivot.

    CLRS RANDOMIZED-SELECT written as a loop over an index range
    ``[lo, hi]`` of one working copy, so there is no recursion at all.
    Each round draws a pivot index uniformly from ``[lo, hi]``, splits the
    range in place into ``< pivot``, ``== pivot`` and ``> pivot``, and
    either stops (``k`` is in the equal block, so the pivot is the answer)
    or narrows the range to the side that contains ``k``. The range always
    contains position ``k``, so when it shrinks to one element that
    element is the answer.

    This is a LAS VEGAS algorithm: the answer is always correct and only
    the running time depends on the random draws. Its success probability
    is exactly 1. The informative probability is how often it finishes
    within ``4n`` comparisons; see the module docstring.

    Args:
        arr: The list to select from. Elements need only be mutually
            comparable with ``<``. Not modified.
        k: The 1-based rank: 1 for the minimum, ``len(arr)`` for the
            maximum.
        seed: Seed for :func:`~src.utils.random_utils.make_rng`. ``None``
            seeds from the operating system, so pivots are unpredictable.
            Ignored when ``rng`` is given.
        rng: A ``random.Random`` to draw pivots from. Takes precedence over
            ``seed``, so one generator can drive many calls.
        return_stats: If true, also return a :class:`SelectionStats`.

    Returns:
        The k-th smallest element, or ``(value, SelectionStats)`` when
        ``return_stats`` is true.

    Raises:
        TypeError: If ``arr`` is not a list, ``k`` is not an int (``bool``
            rejected), or ``seed`` is not an int or ``None``.
        ValueError: If ``arr`` is empty or ``k`` is not in
            ``1 .. len(arr)``.

    Time Complexity:
        Best:    O(n)   - the first pivot drawn is the answer
        Average: O(n)   - expected comparisons about 2n at the ends and
                          about 3.39n at the median
        Worst:   O(n^2) - every draw is the extreme of its range

    Space Complexity:
        O(n) - the working copy. The loop itself uses O(1) extra space.

    Examples:
        The anchor list, every seed giving the same answers:

        >>> data = [7, 10, 4, 3, 20, 15]
        >>> [randomized_select(data, k, seed=s) for s in (0, 1, 42)
        ...  for k in (1, 3, 6)]
        [3, 7, 20, 3, 7, 20, 3, 7, 20]
        >>> data
        [7, 10, 4, 3, 20, 15]

        Statistics depend on the seed but are reproducible with it. The
        first round on 6 items costs 5 comparisons:

        >>> randomized_select(data, 3, seed=42, return_stats=True)
        (7, SelectionStats(depth=2, comparisons=8))

        A battery with duplicates agrees with sorting for every k:

        >>> gen = make_rng(7)
        >>> values = [gen.randint(0, 9) for _ in range(40)]
        >>> all(randomized_select(values, k, rng=gen) == sorted(values)[k - 1]
        ...     for k in range(1, 41))
        True
        >>> randomized_select([5] * 1000, 500, seed=3, return_stats=True)
        (5, SelectionStats(depth=1, comparisons=999))
        >>> randomized_select(data, 0, seed=1)
        Traceback (most recent call last):
            ...
        ValueError: randomized_select() needs 1 <= k <= 6, got k=0
    """
    target = _validate(arr, k, "randomized_select")
    if rng is None:
        rng = make_rng(seed)
    work = list(arr)
    randint = rng.randint
    lo, hi = 0, len(work) - 1
    depth = comparisons = 0

    while lo < hi:
        pivot = work[randint(lo, hi)]
        depth += 1
        comparisons += hi - lo
        # Invariant: work[lo:lt] < pivot, work[lt:i] == pivot,
        # work[gt+1:hi+1] > pivot, work[i:gt+1] not yet classified.
        lt = i = lo
        gt = hi
        while i <= gt:
            item = work[i]
            if item < pivot:
                work[i] = work[lt]
                work[lt] = item
                lt += 1
                i += 1
            elif pivot < item:
                work[i] = work[gt]
                work[gt] = item
                gt -= 1
            else:
                i += 1
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            value = pivot
            break
    else:
        value = work[lo]

    if return_stats:
        return value, SelectionStats(depth, comparisons)
    return value


def _median_of_medians(items: List[T]) -> T:
    """Return the CLRS pivot: the lower median of the group-of-5 medians.

    Each group of 5 (the last may be smaller) is sorted, which costs O(1)
    per group, and its lower median kept. The lower median of those
    ``ceil(n/5)`` medians is then found with the same deterministic
    select. At least half the medians are ``<=`` the result and each of
    them has two more group members ``<=`` it, which is where the
    guarantee of about ``3n/10`` elements on each side comes from.

    Args:
        items: A non-empty list. Not modified.

    Returns:
        An element of ``items`` to partition around.

    Raises:
        IndexError: If ``items`` is empty. Callers never pass one.

    Time Complexity:
        O(n) for this call plus the recursive select on ``n/5`` medians.

    Space Complexity:
        O(n) - the medians, then theirs, a geometric series.

    Examples:
        >>> _median_of_medians([9, 1, 8, 2, 7, 3, 6, 4, 5, 0])
        4
        >>> _median_of_medians([4, 2])
        2
    """
    size = len(items)
    if size <= _GROUP:
        return sorted(items)[(size - 1) // 2]
    full = size - size % _GROUP
    medians = [sorted(items[i : i + _GROUP])[2] for i in range(0, full, _GROUP)]
    if full < size:
        tail = sorted(items[full:])
        medians.append(tail[(len(tail) - 1) // 2])
    return _deterministic_core(medians, (len(medians) - 1) // 2)[0]


def _deterministic_core(items: List[T], target: int) -> Tuple[T, int, int]:
    """Run CLRS SELECT on ``items`` for the 0-based rank ``target``.

    The main select is a loop: each round chooses the median-of-medians
    pivot, splits three ways, and either returns the pivot or continues
    into the strictly smaller or strictly larger values. A round is what
    one level of the textbook recursion would be, so the round count is
    that recursion's depth. The only real recursion is through
    :func:`_median_of_medians`, on a list a fifth the size, so the Python
    stack grows O(log n) frames.

    Args:
        items: A non-empty list. Not modified; each round builds new lists.
        target: The 0-based rank, ``0 <= target < len(items)``.

    Returns:
        ``(value, depth, comparisons)`` with the counts described in the
        module docstring.

    Raises:
        IndexError: If ``items`` is empty. Callers never pass one.

    Time Complexity:
        O(n) worst case - every round keeps at most about ``7n/10``.

    Space Complexity:
        O(n) - the side kept each round, plus O(log n) stack.

    Examples:
        >>> _deterministic_core([7, 10, 4, 3, 20, 15], 0)
        (3, 2, 6)
        >>> _deterministic_core([2] * 1000, 999)
        (2, 1, 999)
    """
    depth = comparisons = 0
    while len(items) > 1:
        size = len(items)
        pivot = _median_of_medians(items)
        depth += 1
        comparisons += size - 1
        smaller = [item for item in items if item < pivot]
        if target < len(smaller):
            items = smaller
            continue
        larger = [item for item in items if pivot < item]
        not_larger = size - len(larger)
        if target < not_larger:
            return pivot, depth, comparisons
        target -= not_larger
        items = larger
    return items[0], depth, comparisons


def deterministic_select(
    arr: List[T],
    k: int,
    return_stats: bool = False,
) -> Union[T, Tuple[T, SelectionStats]]:
    """Return the k-th smallest element in worst-case O(n) time.

    CLRS SELECT: divide the active values into groups of 5, take the
    median of each group, find the median of those medians recursively,
    split three ways around it, and continue into the side holding ``k``.
    The median of medians always has about ``3n/10`` or more elements on
    each side, so ``T(n) <= T(n/5) + T(7n/10) + O(n)``, which is O(n)
    because ``1/5 + 7/10 < 1``. No randomness is used, so the result and
    the statistics are the same on every run.

    Args:
        arr: The list to select from. Elements need only be mutually
            comparable with ``<``. Not modified.
        k: The 1-based rank: 1 for the minimum, ``len(arr)`` for the
            maximum.
        return_stats: If true, also return a :class:`SelectionStats` for
            the main select (pivot finding excluded; see the module
            docstring).

    Returns:
        The k-th smallest element, or ``(value, SelectionStats)`` when
        ``return_stats`` is true.

    Raises:
        TypeError: If ``arr`` is not a list or ``k`` is not an int
            (``bool`` rejected).
        ValueError: If ``arr`` is empty or ``k`` is not in
            ``1 .. len(arr)``.

    Time Complexity:
        Best:    O(n) - one round when the pivot is the answer, as with
                        all-equal values
        Average: O(n)
        Worst:   O(n) - with a large constant from the pivot search

    Space Complexity:
        O(n) - the lists kept between rounds, plus O(log n) stack for the
        nested median-of-medians calls.

    Examples:
        >>> data = [7, 10, 4, 3, 20, 15]
        >>> [deterministic_select(data, k) for k in (1, 3, 6)]
        [3, 7, 20]
        >>> deterministic_select(data, 3, return_stats=True)
        (7, SelectionStats(depth=1, comparisons=5))
        >>> deterministic_select([4] * 1000, 1, return_stats=True)
        (4, SelectionStats(depth=1, comparisons=999))
        >>> deterministic_select(data, 0)
        Traceback (most recent call last):
            ...
        ValueError: deterministic_select() needs 1 <= k <= 6, got k=0
    """
    target = _validate(arr, k, "deterministic_select")
    value, depth, comparisons = _deterministic_core(arr, target)
    if return_stats:
        return value, SelectionStats(depth, comparisons)
    return value


def median(arr: List[T], seed: Optional[int] = None) -> T:
    """Return the lower median of ``arr`` with :func:`randomized_select`.

    For an odd length this is the middle element; for an even length it
    is the smaller of the two middle elements, rank ``(n + 1) // 2``. Like
    :func:`randomized_select` it is LAS VEGAS: always correct, with a
    random running time.

    Args:
        arr: A non-empty list of mutually comparable elements. Not
            modified.
        seed: Seed for the pivot draws; ``None`` for an unpredictable one.

    Returns:
        The lower median.

    Raises:
        TypeError: If ``arr`` is not a list or ``seed`` is not an int or
            ``None``.
        ValueError: If ``arr`` is empty.

    Time Complexity:
        Best:    O(n)
        Average: O(n)   - about 3.39n comparisons expected
        Worst:   O(n^2)

    Space Complexity:
        O(n) - the working copy.

    Examples:
        >>> median([7, 10, 4, 3, 20, 15], seed=42)
        7
        >>> median([5, 1, 3], seed=1)
        3
        >>> median([], seed=1)
        Traceback (most recent call last):
            ...
        ValueError: median() cannot select from an empty list
    """
    if not isinstance(arr, list):
        raise TypeError(f"median() expects a list, got {type(arr).__name__}")
    if not arr:
        raise ValueError("median() cannot select from an empty list")
    return randomized_select(arr, (len(arr) + 1) // 2, seed=seed)
