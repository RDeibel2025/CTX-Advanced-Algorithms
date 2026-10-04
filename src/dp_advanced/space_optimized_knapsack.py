"""0/1 knapsack in one row of capacities, with the proof of why that is safe.

Week 5 solved 0/1 knapsack with a full ``(n + 1) x (W + 1)`` table. This
module keeps the same recurrence and throws away the item dimension: one
row of ``W + 1`` integers, overwritten in place once per item. The answer
is unchanged and the memory drops from O(n*W) to O(W). The module is shaped
around the one thing that makes that legal - the direction of the capacity
scan - and around measuring what the saving is actually worth.

Why one row is enough
---------------------
Writing ``dp_i[c]`` for the best value using the first ``i`` items under
capacity ``c``, the full table obeys::

    dp_i[c] = dp_{i-1}[c]                                       if w_i > c
    dp_i[c] = max(dp_{i-1}[c], v_i + dp_{i-1}[c - w_i])         otherwise

Row ``i`` depends only on row ``i - 1``, at the same capacity ``c`` and at
the smaller capacity ``c - w_i``. Nothing older is ever read, so a single
row can hold row ``i - 1`` and be turned into row ``i`` in place, with
``dp[c]`` overwritten as it is computed.

The order of those overwrites is the whole proof. Scanning ``c`` DOWNWARD,
from ``W`` to ``w_i``, means that when ``dp[c]`` is computed the cell
``dp[c - w_i]`` sits at a smaller index that item ``i`` has not reached
yet. It therefore still holds the row ``i - 1`` value, which is exactly
what the recurrence reads. Reading ``dp[c]`` itself before writing it gives
``dp_{i-1}[c]`` for the same reason. Every read sees row ``i - 1`` and every
write produces row ``i``, so after the scan the row *is* row ``i``.

Scanning UPWARD breaks this. ``dp[c - w_i]`` is then a cell item ``i`` has
already updated, so it may already include item ``i``, and adding ``v_i``
on top of it takes item ``i`` a second time. That computes::

    dp_i[c] = max(dp_{i-1}[c], v_i + dp_i[c - w_i])

which is the recurrence for UNBOUNDED knapsack, where each item may be
taken any number of times - not 0/1. :func:`knapsack_ascending_scan` exists
only to demonstrate this: on weights ``[2, 3]``, values ``[3, 4]`` and
capacity 6 the correct 0/1 answer is 7 (both items), while the upward scan
returns 9 (the weight-2 item three times).

What is given up, and what this module does about it
----------------------------------------------------
With one row there is no history to walk back through, so the optimal
*value* survives but the optimal *item set* does not.
:func:`knapsack_space_optimized_with_items` buys the item set back cheaply
rather than for free: it keeps the O(W) values row plus one Python ``int``
per item used as a bitset over capacities. That is n*W *bits*, not O(W);
it is at least 64 times smaller than the pointer array of a list-of-lists
table on a 64-bit build, but it is still Theta(n*W) memory. The general
technique for recovering the choices in genuinely linear space is
Hirschberg's divide and conquer (Hirschberg, 1975, originally for longest
common subsequence): solve forward over the first half of the items and
backward over the second half, find the capacity split where the two rows
meet, and recurse on each half. It is noted here and not implemented.

====================================== ========= ==========================
Function                               Time      Space
====================================== ========= ==========================
knapsack_space_optimized               O(n*W)    O(W)
knapsack_ascending_scan (WRONG)        O(n*W)    O(W)
knapsack_space_optimized_with_items    O(n*W)    O(W) words + n*(W+1) bits
compare_with_standard                  O(r*n*W)  O(n*W) (the standard)
====================================== ========= ==========================

The baseline for every comparison is the Week 5 full table,
:func:`src.dp.knapsack.knapsack_tab`. :func:`compare_with_standard` times
both with :func:`src.utils.timer.time_call` and measures both with
:func:`src.utils.timer.peak_memory_kib`, in separate passes, because a
timing taken under :mod:`tracemalloc` measures the tracer.

Contract shared by every solver, identical to Week 5:

* ``weights`` and ``values`` are equal-length sequences of non-negative
  ``int`` (``ValueError`` on a length mismatch or a negative entry,
  ``TypeError`` on any other type; ``bool`` is rejected).
* ``capacity`` is a non-negative ``int`` under the same rules.
* An empty item list returns 0, and capacity 0 returns 0 whenever every
  weight is positive. A weight-0 item is free and is always taken, exactly
  as in Week 5.

Examples:
    >>> weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
    >>> knapsack_space_optimized(weights, values, 7)
    9
    >>> knapsack_space_optimized_with_items(weights, values, 7)
    (9, [1, 2])
    >>> knapsack_space_optimized([2, 3], [3, 4], 6)
    7
    >>> knapsack_ascending_scan([2, 3], [3, 4], 6)
    9

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from src.dp.knapsack import knapsack_tab
from src.utils.timer import peak_memory_kib, time_call

__all__ = [
    "compare_with_standard",
    "knapsack_ascending_scan",
    "knapsack_space_optimized",
    "knapsack_space_optimized_with_items",
    "print_comparison",
]

#: ASCII code of the character ``"1"``. The item bitsets are assembled as a
#: string of ``"0"`` and ``"1"`` bytes and parsed once with ``int(..., 2)``,
#: which is linear in the length; setting bits one at a time with
#: ``mask |= 1 << c`` would copy the growing integer on every assignment.
_ONE = ord("1")


def _validate(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> Tuple[List[int], List[int], int]:
    """Check the shared arguments and copy the two sequences into lists.

    The checks and their messages mirror the Week 5 module exactly, in the
    same order (capacity first, then weights, then values, then lengths),
    so a caller swapping :func:`knapsack_tab` for a function here sees the
    same errors. The copies mean the caller's sequences are never mutated.

    Args:
        weights: The item weights.
        values: The item values, parallel to ``weights``.
        capacity: The knapsack capacity.

    Returns:
        A ``(weights, values, capacity)`` triple with both sequences copied
        into fresh lists.

    Raises:
        TypeError: If ``capacity`` or any entry is not an ``int`` (``bool``
            included), or if either sequence is not iterable.
        ValueError: If ``capacity`` or any entry is negative, or if the two
            sequences differ in length.

    Time Complexity:
        O(n) - one isinstance check per entry.

    Space Complexity:
        O(n) - the two copies.

    Examples:
        >>> _validate((1, 2), [3, 4], 5)
        ([1, 2], [3, 4], 5)
        >>> _validate([1], [2], True)
        Traceback (most recent call last):
            ...
        TypeError: capacity must be an int, got bool
        >>> _validate([1], [2], -1)
        Traceback (most recent call last):
            ...
        ValueError: capacity must be >= 0, got -1
        >>> _validate([1, 2.5], [1, 2], 3)
        Traceback (most recent call last):
            ...
        TypeError: weights[1] must be an int, got float
        >>> _validate([1, 2], [1, -2], 3)
        Traceback (most recent call last):
            ...
        ValueError: values[1] must be >= 0, got -2
        >>> _validate(7, [1], 3)
        Traceback (most recent call last):
            ...
        TypeError: weights must be a sequence of ints, got int
        >>> _validate([1, 2], [3], 5)
        Traceback (most recent call last):
            ...
        ValueError: weights and values must be the same length, got 2 and 1
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError(f"capacity must be an int, got {type(capacity).__name__}")
    if capacity < 0:
        raise ValueError(f"capacity must be >= 0, got {capacity}")

    checked: List[List[int]] = []
    for name, items in (("weights", weights), ("values", values)):
        try:
            copy = list(items)
        except TypeError:
            raise TypeError(
                f"{name} must be a sequence of ints, got {type(items).__name__}"
            ) from None
        for index, entry in enumerate(copy):
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(
                    f"{name}[{index}] must be an int, got {type(entry).__name__}"
                )
            if entry < 0:
                raise ValueError(f"{name}[{index}] must be >= 0, got {entry}")
        checked.append(copy)

    checked_weights, checked_values = checked
    if len(checked_weights) != len(checked_values):
        raise ValueError(
            "weights and values must be the same length, got "
            f"{len(checked_weights)} and {len(checked_values)}"
        )
    return checked_weights, checked_values, capacity


def knapsack_space_optimized(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Solve 0/1 knapsack in a single row of ``capacity + 1`` integers.

    One row ``dp`` starts at all zeros (row 0 of the full table: no items,
    no value). For each item the capacities are scanned from ``capacity``
    DOWN to the item's weight, setting
    ``dp[c] = max(dp[c], dp[c - weight] + value)``. The downward direction
    guarantees that ``dp[c - weight]`` has not yet been touched by this
    item and still holds the previous row's value; the module docstring
    gives the full argument. Capacities below the weight are not visited,
    because there the recurrence copies the previous row unchanged and the
    row already holds it.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value that fits within ``capacity``, identical to
        :func:`src.dp.knapsack.knapsack_tab`.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) in every case - the same cells as the full table, written
        over the same memory n times.

    Space Complexity:
        O(W) - one row of ``capacity + 1`` integers, independent of n.

    Examples:
        >>> knapsack_space_optimized([1, 3, 4, 5], [1, 4, 5, 7], 7)
        9
        >>> knapsack_space_optimized([10, 20, 30], [60, 100, 120], 50)
        220
        >>> knapsack_space_optimized([], [], 10)
        0
        >>> knapsack_space_optimized([1, 3, 4, 5], [1, 4, 5, 7], 0)
        0

        One weight-2 item in a bag of 6 is worth its value once, not three
        times:

        >>> knapsack_space_optimized([2], [3], 6)
        3

        It matches the Week 5 full table at every capacity:

        >>> weights, values = [3, 4, 5, 8, 10], [4, 5, 8, 9, 10]
        >>> all(knapsack_space_optimized(weights, values, c)
        ...     == knapsack_tab(weights, values, c) for c in range(25))
        True

        >>> knapsack_space_optimized([1, 2], [3], 5)
        Traceback (most recent call last):
            ...
        ValueError: weights and values must be the same length, got 2 and 1
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    dp = [0] * (capacity + 1)
    for weight, value in zip(checked_weights, checked_values):
        # DOWNWARD: dp[c - weight] must still hold the previous row's value.
        for c in range(capacity, weight - 1, -1):
            candidate = dp[c - weight] + value
            if candidate > dp[c]:
                dp[c] = candidate
    return dp[capacity]


def knapsack_ascending_scan(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """DELIBERATELY WRONG for 0/1 knapsack: it solves UNBOUNDED knapsack.

    Do not use this to solve 0/1 knapsack. It is
    :func:`knapsack_space_optimized` with exactly one change: the capacity
    loop runs UP, from the item's weight to ``capacity``, instead of down.
    Then ``dp[c - weight]`` is a cell this item has already updated, so it
    may already contain the item, and adding ``value`` takes the item
    again. The result is the unbounded-knapsack optimum, in which every
    item may be used any number of times. It is silently wrong for 0/1: no
    error, just a value that can be too large.

    It exists only to demonstrate the correctness argument in the module
    docstring, by showing the counterexample the downward scan avoids. It
    gives the right 0/1 answer only on instances where taking an item more
    than once never helps.

    Unbounded knapsack is only well defined when every weight is positive.
    A weight-0 item with positive value would be worth infinitely much;
    this function simply adds it once per capacity cell and should not be
    read as handling that case.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum value when each item may be taken any number of times,
        which is NOT the 0/1 answer in general.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) in every case.

    Space Complexity:
        O(W) - one row, like the correct version.

    Examples:
        The counterexample. 0/1 takes both items for 3 + 4 = 7; the upward
        scan takes the weight-2 item three times for 3 * 3 = 9:

        >>> knapsack_space_optimized([2, 3], [3, 4], 6)
        7
        >>> knapsack_ascending_scan([2, 3], [3, 4], 6)
        9

        The single-item case makes the reuse plain:

        >>> knapsack_ascending_scan([2], [3], 6)
        9

        Where reuse cannot fit, it happens to agree with 0/1:

        >>> knapsack_ascending_scan([4, 5], [5, 7], 7)
        7
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    dp = [0] * (capacity + 1)
    for weight, value in zip(checked_weights, checked_values):
        # UPWARD: the bug on purpose. dp[c - weight] may already include
        # this item, so the item can be taken again.
        for c in range(weight, capacity + 1):
            candidate = dp[c - weight] + value
            if candidate > dp[c]:
                dp[c] = candidate
    return dp[capacity]


def knapsack_space_optimized_with_items(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int]]:
    """Solve 0/1 knapsack in one values row and recover the chosen items.

    The values are computed exactly as in :func:`knapsack_space_optimized`.
    Alongside them each item gets one Python ``int`` used as a bitset over
    capacities: bit ``c`` of item ``i``'s bitset is set when taking item
    ``i`` strictly improved ``dp[c]``, that is, when the full table would
    have had ``table[i + 1][c] != table[i][c]``. That is precisely the
    question the Week 5 backward walk asks of the full table, so the same
    walk works here: start at ``capacity``, go through the items from last
    to first, and whenever the item's bit at the current capacity is set,
    record the item and subtract its weight.

    Honestly stated, the saving is in the values, not in the history. The
    values need O(W) space; reconstruction needs n*(W+1) BITS, one bit per
    cell of the full table. That is far smaller than a list-of-lists table,
    where each cell costs at least a 64-bit pointer, but it is still
    Theta(n*W). Recovering the items in genuinely linear space needs
    Hirschberg's divide and conquer (Hirschberg, 1975), which this module
    describes and does not implement.

    Ties are broken as in :func:`src.dp.knapsack.trace_solution`: a bit is
    set only on a strict improvement, so a later item that merely matches
    is not recorded and the earlier items are preferred.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        A ``(best_value, indices)`` pair. ``indices`` lists the chosen
        items' positions in ascending order; their weights sum to at most
        ``capacity`` and their values sum to exactly ``best_value``.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) - the row updates, plus an O(W) parse per item to build its
        bitset and an O(n) backward walk.

    Space Complexity:
        O(W) integers for the values row, plus n*(W+1) bits for the
        bitsets, plus one transient O(W)-byte buffer per item.

    Examples:
        >>> weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
        >>> best, chosen = knapsack_space_optimized_with_items(
        ...     weights, values, 7)
        >>> best, chosen
        (9, [1, 2])
        >>> sum(weights[i] for i in chosen), sum(values[i] for i in chosen)
        (7, 9)

        The classic instance, and the 0/1 counterexample - both items, each
        once:

        >>> knapsack_space_optimized_with_items([10, 20, 30], [60, 100, 120], 50)
        (220, [1, 2])
        >>> knapsack_space_optimized_with_items([2, 3], [3, 4], 6)
        (7, [0, 1])

        Nothing fits, or nothing to pack:

        >>> knapsack_space_optimized_with_items([8, 9], [100, 200], 7)
        (0, [])
        >>> knapsack_space_optimized_with_items([], [], 5)
        (0, [])

        A weight-0 item is free and is taken even at capacity 0:

        >>> knapsack_space_optimized_with_items([0, 5], [3, 9], 0)
        (3, [0])

        It returns the same items as the Week 5 full-table walk:

        >>> from src.dp.knapsack import trace_solution
        >>> weights, values = [3, 4, 5, 8, 10, 2], [4, 5, 8, 9, 10, 3]
        >>> all(knapsack_space_optimized_with_items(weights, values, c)
        ...     == trace_solution(weights, values, c) for c in range(30))
        True
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    dp = [0] * (capacity + 1)
    taken: List[int] = []
    # Character j of the buffer is bit (capacity - j), so the buffer reads
    # most significant bit first and int(buffer, 2) needs no reversal.
    for weight, value in zip(checked_weights, checked_values):
        if weight > capacity:
            taken.append(0)
            continue
        buffer = bytearray(b"0") * (capacity + 1)
        for c in range(capacity, weight - 1, -1):
            candidate = dp[c - weight] + value
            if candidate > dp[c]:
                dp[c] = candidate
                buffer[capacity - c] = _ONE
        taken.append(int(buffer, 2))

    chosen: List[int] = []
    room = capacity
    for item in range(len(checked_weights) - 1, -1, -1):
        if (taken[item] >> room) & 1:
            chosen.append(item)
            room -= checked_weights[item]
    chosen.reverse()
    return dp[capacity], chosen


def _ratio(numerator: float, denominator: float) -> float:
    """Divide two non-negative measurements without dividing by zero.

    Args:
        numerator: The standard's figure.
        denominator: The optimized figure.

    Returns:
        ``numerator / denominator``; ``math.inf`` when only the denominator
        is zero, and ``1.0`` when both are zero (no difference measured).

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _ratio(6.0, 2.0)
        3.0
        >>> _ratio(1.0, 0.0)
        inf
        >>> _ratio(0.0, 0.0)
        1.0
    """
    if denominator > 0:
        return numerator / denominator
    return math.inf if numerator > 0 else 1.0


def compare_with_standard(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
    repeat: int = 5,
) -> Dict[str, Any]:
    """Measure the one-row solver against the Week 5 full table.

    The standard is :func:`src.dp.knapsack.knapsack_tab`, which builds the
    whole ``(n + 1) x (W + 1)`` table; the optimized solver is
    :func:`knapsack_space_optimized`. Each is measured twice, in separate
    passes that never overlap:

    1. Time, with :func:`src.utils.timer.time_call` (warm-up runs, then
       ``repeat`` timed runs with garbage collection paused).
    2. Peak memory, with :func:`src.utils.timer.peak_memory_kib` (one run
       each under :mod:`tracemalloc`). The values returned by this pass are
       the ones reported, so correctness costs no extra run.

    Timing under tracing would measure the tracer, which is why the passes
    are separate.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.
        repeat: Number of timed runs per solver, passed to ``time_call``.

    Returns:
        A dictionary with exactly these keys:

        * ``"n"`` and ``"capacity"`` - the instance size.
        * ``"standard_value"``, ``"optimized_value"`` and ``"agree"`` - the
          two answers and whether they are equal.
        * ``"standard_time"`` and ``"optimized_time"`` - the dictionaries
          ``time_call`` returns (``mean``, ``std``, ``min``, ``max``,
          ``runs``), in seconds.
        * ``"standard_peak_kib"`` and ``"optimized_peak_kib"`` - peak
          traced allocation of one call, in KiB.
        * ``"memory_ratio"`` - standard peak over optimized peak.
        * ``"time_ratio"`` - standard mean time over optimized mean time.

    Raises:
        TypeError: If the instance breaks the shared contract by type, or
            ``repeat`` is not an ``int``.
        ValueError: If the instance breaks the shared contract by value, or
            ``repeat`` is less than 1.

    Time Complexity:
        O((repeat + warmup + 1) * n*W) - both solvers run that many times.

    Space Complexity:
        O(n*W) - the standard's table, during its own runs.

    Examples:
        Only the structure is checked here; timings vary run to run:

        >>> result = compare_with_standard([1, 3, 4, 5], [1, 4, 5, 7], 7,
        ...                                repeat=2)
        >>> sorted(result)  # doctest: +NORMALIZE_WHITESPACE
        ['agree', 'capacity', 'memory_ratio', 'n', 'optimized_peak_kib',
         'optimized_time', 'optimized_value', 'standard_peak_kib',
         'standard_time', 'standard_value', 'time_ratio']
        >>> result["agree"], result["standard_value"], result["optimized_value"]
        (True, 9, 9)
        >>> result["n"], result["capacity"], result["standard_time"]["runs"]
        (4, 7, 2)

        >>> compare_with_standard([1], [1], 1, repeat=0)
        Traceback (most recent call last):
            ...
        ValueError: repeat must be >= 1, got 0
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)
    args = (checked_weights, checked_values, capacity)

    # Pass 1: time only. No tracer is running here.
    standard_time = time_call(knapsack_tab, *args, repeat=repeat)
    optimized_time = time_call(knapsack_space_optimized, *args, repeat=repeat)

    # Pass 2: memory only, one traced call each. These also supply the values.
    standard_value, standard_kib = peak_memory_kib(knapsack_tab, *args)
    optimized_value, optimized_kib = peak_memory_kib(knapsack_space_optimized, *args)

    return {
        "n": len(checked_weights),
        "capacity": capacity,
        "standard_value": standard_value,
        "optimized_value": optimized_value,
        "agree": standard_value == optimized_value,
        "standard_time": standard_time,
        "optimized_time": optimized_time,
        "standard_peak_kib": standard_kib,
        "optimized_peak_kib": optimized_kib,
        "memory_ratio": _ratio(standard_kib, optimized_kib),
        "time_ratio": _ratio(standard_time["mean"], optimized_time["mean"]),
    }


def print_comparison(result: Dict[str, Any]) -> None:
    """Print a comparison from :func:`compare_with_standard` as a table.

    One row per variant - value, mean time in milliseconds, peak KiB -
    followed by a two-line summary giving the time and memory ratios
    (standard over optimized, so a figure above 1 favours the one-row
    solver).

    Args:
        result: A dictionary shaped like the one
            :func:`compare_with_standard` returns.

    Returns:
        None. The table is written to standard output.

    Raises:
        KeyError: If ``result`` is missing one of the keys listed in
            :func:`compare_with_standard`.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        A fixed result, so the output is reproducible:

        >>> fixed = {
        ...     "n": 4, "capacity": 7,
        ...     "standard_value": 9, "optimized_value": 9, "agree": True,
        ...     "standard_time": {"mean": 0.000012},
        ...     "optimized_time": {"mean": 0.000008},
        ...     "standard_peak_kib": 2.5, "optimized_peak_kib": 0.5,
        ...     "memory_ratio": 5.0, "time_ratio": 1.5,
        ... }
        >>> print_comparison(fixed)
        0/1 knapsack, n=4, W=7 (values agree: True)
        variant      value  mean time (ms)  peak KiB
        standard         9          0.0120      2.50
        optimized        9          0.0080      0.50
        Runtime: standard / optimized mean time = 1.50x (above 1: one row is faster).
        Memory: standard / optimized peak = 5.00x (full table vs one row).
    """
    print(
        f"0/1 knapsack, n={result['n']}, W={result['capacity']} "
        f"(values agree: {result['agree']})"
    )
    print(f"{'variant':<10} {'value':>7} {'mean time (ms)':>15} {'peak KiB':>9}")
    for label in ("standard", "optimized"):
        mean_ms = result[f"{label}_time"]["mean"] * 1000.0
        print(
            f"{label:<10} {result[f'{label}_value']:>7} {mean_ms:>15.4f} "
            f"{result[f'{label}_peak_kib']:>9.2f}"
        )
    print(
        f"Runtime: standard / optimized mean time = {result['time_ratio']:.2f}x "
        "(above 1: one row is faster)."
    )
    print(
        f"Memory: standard / optimized peak = {result['memory_ratio']:.2f}x "
        "(full table vs one row)."
    )
