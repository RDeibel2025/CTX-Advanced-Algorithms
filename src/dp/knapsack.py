"""The 0/1 knapsack problem, written four ways to price the same recurrence.

Given items with integer weights and values and a knapsack of integer
capacity, choose a subset of maximum total value whose total weight fits.
*0/1* means each item is taken whole or not at all: there is no fractional
relaxation here, which is exactly why greed fails and dynamic programming
is needed.

Every function in this module solves the same recurrence. Writing
``best(i, c)`` for the best value obtainable from items ``i..n-1`` under
remaining capacity ``c``::

    best(n, c) = 0
    best(i, c) = best(i + 1, c)                                  if w_i > c
    best(i, c) = max(best(i + 1, c),
                     v_i + best(i + 1, c - w_i))                 otherwise

The four implementations differ only in **who decides the order the
subproblems are solved in**, and every trade-off between them follows from
that one difference:

====================== =============== ====================================
Function               Time            Space
====================== =============== ====================================
knapsack_recursive     O(2^n)          O(n) call stack
knapsack_memo          O(n*W)          O(n*W) worst case + O(n) call stack
knapsack_tab           O(n*W)          O(n*W)
knapsack_tab_rolling   O(n*W)          O(W)
====================== =============== ====================================

Two consequences of that difference are measurable rather than asserted,
and this module exposes both so the Week 5 benchmark can report numbers
instead of adjectives:

* :func:`knapsack_memo_cells` counts the ``(item, capacity)`` states the
  top-down memo dictionary actually stored, beside the ``n * (W + 1)``
  cells the full table holds. Knapsack's subproblem space is **sparse**:
  only capacities reachable by summing some subset of the weights are ever
  asked for, so laziness can touch far fewer cells than eagerness fills.
  How much fewer depends entirely on the instance, which is why it is
  counted and not predicted.
* :func:`knapsack_tab_rolling` collapses the table to one row of length
  ``W + 1``. Only the bottom-up version can do that, because collapsing
  requires knowing in advance that row ``i`` is read only by row ``i + 1``.
  Memoization cannot discard anything, since it does not know what it will
  still be asked for.

A note on the ``O(n*W)`` bound: it is *pseudo-polynomial*, not polynomial.
``W`` is a capacity value, not an input size, so the table grows
exponentially in the number of bits used to write the capacity down. 0/1
knapsack is NP-hard, and this module does not change that; it only makes
the small-``W`` instances cheap.

Contract shared by every public function:

* ``weights`` and ``values`` must be equal-length sequences of
  non-negative ``int`` (``ValueError`` on a length mismatch).
* ``capacity`` must be a non-negative ``int`` (``TypeError`` for any other
  type, ``ValueError`` when negative). ``bool`` is rejected as a type
  error everywhere, because ``True`` is an accident, not a capacity.
* An empty item list returns 0 for any capacity, and a capacity of 0
  returns 0 whenever every weight is positive. An item of weight 0 is
  free and is always taken, so a zero capacity can still return a positive
  value; that is the recurrence behaving correctly, not an edge case being
  mishandled.
* The four value-returning variants agree on every valid input.

Examples:
    >>> weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
    >>> sorted(VARIANTS)
    ['memo', 'recursive', 'tab']
    >>> [VARIANTS[name](weights, values, 7) for name in sorted(VARIANTS)]
    [9, 9, 9]
    >>> knapsack_tab_rolling(weights, values, 7)
    9

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from src.utils.timer import CallCounter

__all__ = [
    "VARIANTS",
    "knapsack_instrumented",
    "knapsack_memo",
    "knapsack_memo_cells",
    "knapsack_recursive",
    "knapsack_tab",
    "knapsack_tab_rolling",
    "trace_solution",
]

#: The three variants the benchmark compares, keyed by the name that
#: appears in the ``variant`` column of ``dp_vs_recursive_table.csv``.
#: :func:`knapsack_tab_rolling` is deliberately absent: it is the space
#: experiment, not a fourth timing series.
VARIANTS: Dict[str, Callable[[Sequence[int], Sequence[int], int], int]]


def _require_item_list(items: Sequence[int], name: str) -> List[int]:
    """Copy ``items`` into a list after checking every entry.

    Both ``weights`` and ``values`` carry the same restriction, so the two
    checks share one helper. The result is a fresh list, which means the
    algorithms below index a plain list rather than an arbitrary sequence
    and the caller's input can never be mutated.

    Integers are required rather than any number. A weight has to index a
    table of ``capacity + 1`` cells, and a value is summed into the ``int``
    these functions promise to return.

    Args:
        items: The sequence to validate and copy.
        name: ``"weights"`` or ``"values"``, used in the error messages.

    Returns:
        A new list holding the same entries.

    Raises:
        TypeError: If ``items`` is not iterable, or if any entry is not an
            ``int``. ``bool`` is treated as a wrong type.
        ValueError: If any entry is negative.

    Time Complexity:
        O(n) - one pass over the entries, one isinstance check each.

    Space Complexity:
        O(n) - the returned copy.

    Examples:
        >>> _require_item_list((1, 2, 3), "weights")
        [1, 2, 3]
        >>> _require_item_list([], "values")
        []
        >>> _require_item_list([1, 2.5], "weights")
        Traceback (most recent call last):
            ...
        TypeError: weights[1] must be an int, got float
        >>> _require_item_list([1, True], "values")
        Traceback (most recent call last):
            ...
        TypeError: values[1] must be an int, got bool
        >>> _require_item_list([1, -2], "values")
        Traceback (most recent call last):
            ...
        ValueError: values[1] must be >= 0, got -2
        >>> _require_item_list(7, "weights")
        Traceback (most recent call last):
            ...
        TypeError: weights must be a sequence of ints, got int
    """
    try:
        result = list(items)
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of ints, got {type(items).__name__}"
        ) from None

    for index, entry in enumerate(result):
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(
                f"{name}[{index}] must be an int, got {type(entry).__name__}"
            )
        if entry < 0:
            raise ValueError(f"{name}[{index}] must be >= 0, got {entry}")

    return result


def _validate(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> Tuple[List[int], List[int], int]:
    """Check the arguments every public function shares and normalise them.

    Centralising the checks keeps the error messages identical across the
    whole module and keeps each algorithm body focused on its recurrence.
    Validation runs once at the public boundary; the private recursive
    helpers below trust their arguments and never re-check them, which
    matters because they are called once per subproblem.

    Args:
        weights: The item weights.
        values: The item values, parallel to ``weights``.
        capacity: The knapsack capacity.

    Returns:
        A ``(weights, values, capacity)`` triple in which the two sequences
        have been copied into lists.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or if either sequence
            is not a sequence of ``int``.
        ValueError: If ``capacity`` is negative, if either sequence holds a
            negative entry, or if the two sequences differ in length.

    Time Complexity:
        O(n) - dominated by the two entry scans.

    Space Complexity:
        O(n) - the two copies.

    Examples:
        >>> _validate([1, 2], [3, 4], 5)
        ([1, 2], [3, 4], 5)
        >>> _validate([], [], 0)
        ([], [], 0)
        >>> _validate([1, 2], [3], 5)
        Traceback (most recent call last):
            ...
        ValueError: weights and values must be the same length, got 2 and 1
        >>> _validate([1], [2], 1.5)
        Traceback (most recent call last):
            ...
        TypeError: capacity must be an int, got float
        >>> _validate([1], [2], True)
        Traceback (most recent call last):
            ...
        TypeError: capacity must be an int, got bool
        >>> _validate([1], [2], -1)
        Traceback (most recent call last):
            ...
        ValueError: capacity must be >= 0, got -1
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError(f"capacity must be an int, got {type(capacity).__name__}")
    if capacity < 0:
        raise ValueError(f"capacity must be >= 0, got {capacity}")

    checked_weights = _require_item_list(weights, "weights")
    checked_values = _require_item_list(values, "values")

    if len(checked_weights) != len(checked_values):
        raise ValueError(
            "weights and values must be the same length, got "
            f"{len(checked_weights)} and {len(checked_values)}"
        )

    return checked_weights, checked_values, capacity


def _require_variant(variant: str) -> None:
    """Raise :class:`ValueError` unless ``variant`` names a known variant.

    Args:
        variant: The name to check against the keys of :data:`VARIANTS`.

    Returns:
        None. This helper is called for its exception, not its value.

    Raises:
        ValueError: If ``variant`` is not one of the three names.

    Time Complexity:
        O(1) - a dictionary membership test, plus O(k log k) to sort the
        three keys when building the error message on the failing path.

    Space Complexity:
        O(1).

    Examples:
        >>> _require_variant("memo") is None
        True
        >>> _require_variant("tabulation")
        Traceback (most recent call last):
            ...
        ValueError: unknown variant 'tabulation'; expected one of 'memo', \
'recursive', 'tab'
    """
    if variant not in VARIANTS:
        known = ", ".join(repr(name) for name in sorted(VARIANTS))
        raise ValueError(f"unknown variant {variant!r}; expected one of {known}")


def _recursive(
    weights: List[int],
    values: List[int],
    index: int,
    remaining: int,
) -> int:
    """Solve ``best(index, remaining)`` by plain double recursion.

    The private worker behind :func:`knapsack_recursive`. It carries no
    instrumentation at all - no counter argument, no bookkeeping - because
    this is the function the benchmark times, and anything in this hot path
    would be measured along with the algorithm.

    Note the base case is ``index == len(weights)`` alone. Stopping early
    on ``remaining == 0`` would be wrong in the presence of weight-0 items,
    which still carry value and are still free to take.

    Args:
        weights: Validated item weights.
        values: Validated item values.
        index: The first item still under consideration.
        remaining: Capacity left in the knapsack.

    Returns:
        The best total value obtainable from ``weights[index:]``.

    Time Complexity:
        O(2^n) - two branches per item, no sharing between them.

    Space Complexity:
        O(n) - the call stack at its deepest, one frame per item.

    Examples:
        >>> _recursive([1, 3, 4, 5], [1, 4, 5, 7], 0, 7)
        9
        >>> _recursive([1, 3, 4, 5], [1, 4, 5, 7], 2, 7)
        7
    """
    if index == len(weights):
        return 0

    skipped = _recursive(weights, values, index + 1, remaining)

    weight = weights[index]
    if weight > remaining:
        return skipped

    taken = values[index] + _recursive(
        weights, values, index + 1, remaining - weight
    )
    return taken if taken > skipped else skipped


def _memo(
    weights: List[int],
    values: List[int],
    index: int,
    remaining: int,
    memo: Dict[Tuple[int, int], int],
) -> int:
    """Solve ``best(index, remaining)`` top-down, caching in ``memo``.

    The private worker behind :func:`knapsack_memo` and
    :func:`knapsack_memo_cells`. The cache is an explicit dictionary passed
    down the recursion rather than a decorator or a module-level global:
    each call to a public function creates its own, so two benchmarks
    running in the same process cannot see each other's entries.

    Only states with ``index < len(weights)`` are stored. The base case is
    a constant and caching it would inflate the entry count that
    :func:`knapsack_memo_cells` reports.

    Args:
        weights: Validated item weights.
        values: Validated item values.
        index: The first item still under consideration.
        remaining: Capacity left in the knapsack.
        memo: Cache keyed by ``(index, remaining)``. Mutated in place.

    Returns:
        The best total value obtainable from ``weights[index:]``.

    Time Complexity:
        O(n*W) - at most one evaluation per reachable ``(item, capacity)``
        state, each doing O(1) work. Far fewer states are usually reached;
        see :func:`knapsack_memo_cells`.

    Space Complexity:
        O(n*W) for the cache in the worst case, plus O(n) of call stack.

    Examples:
        >>> cache: Dict[Tuple[int, int], int] = {}
        >>> _memo([1, 3, 4, 5], [1, 4, 5, 7], 0, 7, cache)
        9
        >>> len(cache)
        13
    """
    if index == len(weights):
        return 0

    key = (index, remaining)
    cached = memo.get(key)
    if cached is not None:
        return cached

    best = _memo(weights, values, index + 1, remaining, memo)

    weight = weights[index]
    if weight <= remaining:
        taken = values[index] + _memo(
            weights, values, index + 1, remaining - weight, memo
        )
        if taken > best:
            best = taken

    memo[key] = best
    return best


def _build_table(
    weights: List[int],
    values: List[int],
    capacity: int,
) -> List[List[int]]:
    """Fill the full ``(n + 1) x (capacity + 1)`` table bottom-up.

    Shared by :func:`knapsack_tab` and :func:`trace_solution`, which need
    the same table for different reasons: the first reads one cell out of
    it, the second walks the whole thing backwards. Row ``i`` holds the
    best value obtainable from the first ``i`` items at each capacity, so
    row 0 is all zeros and the answer is the last cell of the last row.

    Args:
        weights: Validated item weights.
        values: Validated item values.
        capacity: Validated knapsack capacity.

    Returns:
        The completed table, indexed ``table[items_considered][capacity]``.

    Time Complexity:
        O(n*W) - every cell is written exactly once, with O(1) work each.
        There is no best or worst case: the loops do not depend on the
        data, which is the defining property of tabulation.

    Space Complexity:
        O(n*W) - the whole table is kept, because the backward walk in
        :func:`trace_solution` needs every row.

    Examples:
        >>> for row in _build_table([1, 3, 4], [1, 4, 5], 7):
        ...     print(row)
        [0, 0, 0, 0, 0, 0, 0, 0]
        [0, 1, 1, 1, 1, 1, 1, 1]
        [0, 1, 1, 4, 5, 5, 5, 5]
        [0, 1, 1, 4, 5, 6, 6, 9]
    """
    item_count = len(weights)
    table = [[0] * (capacity + 1) for _ in range(item_count + 1)]

    for item in range(1, item_count + 1):
        weight = weights[item - 1]
        value = values[item - 1]
        previous_row = table[item - 1]
        current_row = table[item]

        for room in range(capacity + 1):
            best = previous_row[room]
            if weight <= room:
                candidate = value + previous_row[room - weight]
                if candidate > best:
                    best = candidate
            current_row[room] = best

    return table


def knapsack_recursive(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Solve 0/1 knapsack by exhaustive recursion, with no memory at all.

    For each item the function asks the same two questions - what is the
    best result without it, and what is the best result with it - and takes
    the larger. Nothing is remembered between those questions, so a state
    reachable by several different subsets is recomputed once per subset.
    That is the baseline the two dynamic-programming variants are measured
    against, and it is unusable past roughly 25 items.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value that fits within ``capacity``.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        Best:    O(n)     - every weight exceeds the capacity, so only the
                            skip branch is ever taken and the recursion is
                            a single chain
        Average: O(2^n)
        Worst:   O(2^n)   - every item fits, so both branches always run

    Space Complexity:
        O(n) - the call stack only. No table is built, which is the one
        thing this variant has going for it.

    Examples:
        The hand-checked instance: taking the weight-3 and weight-4 items
        fills the bag exactly and is worth 4 + 5 = 9.

        >>> knapsack_recursive([1, 3, 4, 5], [1, 4, 5, 7], 7)
        9

        Nothing to pack, or nowhere to put it:

        >>> knapsack_recursive([], [], 10)
        0
        >>> knapsack_recursive([1, 3, 4, 5], [1, 4, 5, 7], 0)
        0

        No item fits:

        >>> knapsack_recursive([8, 9], [100, 200], 7)
        0

        The classic three-item instance, where the greedy choice by value
        density would take the weight-10 item and lose:

        >>> knapsack_recursive([10, 20, 30], [60, 100, 120], 50)
        220

        Mismatched inputs are refused:

        >>> knapsack_recursive([1, 2], [3], 5)
        Traceback (most recent call last):
            ...
        ValueError: weights and values must be the same length, got 2 and 1
        >>> knapsack_recursive([1, 2], [3, 4], -5)
        Traceback (most recent call last):
            ...
        ValueError: capacity must be >= 0, got -5
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)
    return _recursive(checked_weights, checked_values, 0, capacity)


def knapsack_memo(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Solve 0/1 knapsack top-down, caching each subproblem in a dictionary.

    Structurally identical to :func:`knapsack_recursive`, with one
    dictionary in front of it. Before solving ``(item, capacity)`` the
    recursion looks the state up; after solving it, it stores it. The
    exponential blow-up disappears because the recursion tree is collapsed
    into the set of *distinct* states it contains.

    The cache is created fresh on every call and passed down the recursion
    explicitly. It is not a ``functools.lru_cache`` and not a module-level
    dictionary, so nothing survives between calls and two benchmark runs in
    one process cannot contaminate each other.

    Top-down means the order is discovered lazily, which has a cost the
    tabulated version does not pay: the recursion borrows the interpreter's
    call stack and reaches a depth of ``len(weights) + 1``. At the item
    counts this module is benchmarked over that is nowhere near CPython's
    default limit of 1000, but it is the same mechanism that breaks
    memoized LCS on long strings.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value that fits within ``capacity``. Identical to
        what the other three variants return.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        Best:    O(n)     - a chain of states, when no item fits
        Average: O(R)     - one O(1) evaluation per *reachable* state R,
                            and R is often far below n*W; see
                            :func:`knapsack_memo_cells`
        Worst:   O(n*W)   - every state reachable

    Space Complexity:
        O(n*W) worst case for the cache, O(R) in practice, plus O(n) of
        call stack. Unlike tabulation it cannot be reduced to one row:
        memoization does not know which states it will still be asked for.

    Examples:
        >>> knapsack_memo([1, 3, 4, 5], [1, 4, 5, 7], 7)
        9
        >>> knapsack_memo([10, 20, 30], [60, 100, 120], 50)
        220
        >>> knapsack_memo([], [], 10)
        0

        It agrees with the exhaustive version, which is the property the
        test battery checks over random instances:

        >>> weights, values = [2, 3, 4, 5, 9], [3, 4, 5, 8, 10]
        >>> knapsack_memo(weights, values, 10) == knapsack_recursive(
        ...     weights, values, 10)
        True

        A weight-0 item is free, so it is taken even at capacity 0:

        >>> knapsack_memo([0, 5], [3, 9], 0)
        3
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)
    cache: Dict[Tuple[int, int], int] = {}
    return _memo(checked_weights, checked_values, 0, capacity, cache)


def knapsack_tab(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Solve 0/1 knapsack bottom-up, filling the whole table row by row.

    The same recurrence as :func:`knapsack_memo`, with the order fixed in
    advance instead of discovered at run time. Row ``i`` is computed from
    row ``i - 1`` for every capacity from 0 upwards, so by the time a cell
    is read the cell it depends on is already written and no recursion is
    needed. The loop nest is two counted ``for`` statements: no call
    frames, no dictionary hashing, and a recursion depth of 1.

    The price of fixing the order is that every cell gets filled, including
    the ones no optimal solution would ever consult. On a sparse instance
    that is strictly more work than the memoized version does, which is
    what :func:`knapsack_memo_cells` is for.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value that fits within ``capacity``.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) in every case. Best, average and worst coincide because the
        loop bounds depend on the input sizes and never on the values.

    Space Complexity:
        O(n*W) - the full table. :func:`knapsack_tab_rolling` gets the same
        answer in O(W); use the full table only when the choices themselves
        are wanted, as in :func:`trace_solution`.

    Examples:
        >>> knapsack_tab([1, 3, 4, 5], [1, 4, 5, 7], 7)
        9
        >>> knapsack_tab([10, 20, 30], [60, 100, 120], 50)
        220
        >>> knapsack_tab([], [], 10)
        0
        >>> knapsack_tab([1, 3, 4, 5], [1, 4, 5, 7], 0)
        0

        All four variants agree, which is the module's central invariant:

        >>> weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
        >>> answers = {
        ...     knapsack_recursive(weights, values, 9),
        ...     knapsack_memo(weights, values, 9),
        ...     knapsack_tab(weights, values, 9),
        ...     knapsack_tab_rolling(weights, values, 9),
        ... }
        >>> answers
        {12}
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)
    table = _build_table(checked_weights, checked_values, capacity)
    return table[len(checked_weights)][capacity]


def knapsack_tab_rolling(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Solve 0/1 knapsack bottom-up in a single rolling row of size W + 1.

    Row ``i`` of the full table is read only by row ``i + 1``, so all but
    one row is dead weight once it has been used. This version keeps just
    the one row and overwrites it in place.

    In place overwriting is only safe because the order is known in
    advance. The row is scanned **downwards**, from ``capacity`` to
    ``weight``, so that ``row[room - weight]`` is always a cell this item
    has not touched yet - that is, still row ``i - 1``. Scanning upwards
    would read a cell the current item had already updated, which would
    allow the same item to be taken twice and would silently turn 0/1
    knapsack into the unbounded variant. The direction is the whole trick,
    and it is available only to tabulation: memoization cannot collapse the
    table this way because it does not know in advance which cells it will
    still be asked for.

    The trade is that the choices are gone. With only one row left there is
    nothing to walk backwards, so recovering *which* items were taken needs
    the full table of :func:`trace_solution`.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value that fits within ``capacity``, identical to
        :func:`knapsack_tab`.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) in every case - the same cell count as the full table, just
        written over the same memory n times.

    Space Complexity:
        O(W) - one row of ``capacity + 1`` integers, independent of n.

    Examples:
        >>> knapsack_tab_rolling([1, 3, 4, 5], [1, 4, 5, 7], 7)
        9
        >>> knapsack_tab_rolling([10, 20, 30], [60, 100, 120], 50)
        220
        >>> knapsack_tab_rolling([], [], 10)
        0

        The downward scan is what keeps it 0/1. One item of weight 2 and
        value 3 in a bag of capacity 6 is worth 3, not 9:

        >>> knapsack_tab_rolling([2], [3], 6)
        3

        It matches the full table everywhere:

        >>> weights, values = [3, 4, 5, 8, 10], [4, 5, 8, 9, 10]
        >>> all(knapsack_tab_rolling(weights, values, c)
        ...     == knapsack_tab(weights, values, c) for c in range(15))
        True
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    row = [0] * (capacity + 1)
    for weight, value in zip(checked_weights, checked_values):
        # DOWNWARDS. row[room - weight] must still hold the previous item's
        # answer; going upwards would read this item's own update and let
        # it be taken more than once.
        for room in range(capacity, weight - 1, -1):
            candidate = value + row[room - weight]
            if candidate > row[room]:
                row[room] = candidate

    return row[capacity]


def trace_solution(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> Tuple[int, List[int]]:
    """Return the optimal value together with the items that achieve it.

    The optimum on its own answers "how much", not "which". Recovering the
    choices means asking each cell of the finished table how it got its
    value, from the last cell backwards: if ``table[i][room]`` differs from
    ``table[i - 1][room]``, the only way that row could improve on the row
    above is by taking item ``i - 1``, so the item is recorded and the
    remaining capacity drops by its weight. If the two are equal the item
    was not needed, and the walk moves up a row with the capacity
    unchanged. One step per row makes the walk O(n) on top of the table.

    Where several subsets tie, this rule returns one of them - the one that
    prefers the later item, since the table is entered from the last row.
    A tie means the alternatives are worth exactly the same, so any of them
    is a correct answer to the problem as posed.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        A ``(best_value, indices)`` pair. ``indices`` lists the positions
        of the chosen items in ascending order; they are distinct, their
        weights sum to at most ``capacity``, and their values sum to
        exactly ``best_value``.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) - the table dominates; the backward walk itself is O(n).

    Space Complexity:
        O(n*W) - the full table is required. This is the one job the
        rolling row of :func:`knapsack_tab_rolling` cannot do, because it
        keeps no history to walk back through.

    Examples:
        The hand-checked instance. Indices 1 and 2 are the weight-3 and
        weight-4 items, which fill the bag exactly:

        >>> weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
        >>> best, chosen = trace_solution(weights, values, 7)
        >>> best, chosen
        (9, [1, 2])
        >>> sum(weights[i] for i in chosen)
        7
        >>> sum(values[i] for i in chosen) == best == knapsack_tab(
        ...     weights, values, 7)
        True

        Nothing fits, so nothing is chosen:

        >>> trace_solution([8, 9], [100, 200], 7)
        (0, [])
        >>> trace_solution([], [], 5)
        (0, [])

        The classic instance takes the two heavier items, not the densest
        one:

        >>> trace_solution([10, 20, 30], [60, 100, 120], 50)
        (220, [1, 2])
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    table = _build_table(checked_weights, checked_values, capacity)
    best_value = table[len(checked_weights)][capacity]

    chosen: List[int] = []
    room = capacity
    for item in range(len(checked_weights), 0, -1):
        # A row can only beat the row above it by having taken its item.
        if table[item][room] != table[item - 1][room]:
            chosen.append(item - 1)
            room -= checked_weights[item - 1]

    chosen.reverse()
    return best_value, chosen


def knapsack_memo_cells(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> Tuple[int, int, int]:
    """Measure how much of the table the top-down solution actually touches.

    This is the instrumented counterpart of :func:`knapsack_memo`, and it
    exists so the Week 5 report can measure its central counterexample
    rather than assert it. Tabulation fills all ``n * (W + 1)`` cells
    because it fixes the order in advance and has no way to know which of
    them matter. Memoization only ever asks for a state it needs, and the
    states it needs are the ``(item, capacity)`` pairs reachable by
    subtracting some subset of the weights from the capacity - which, when
    the weights are large or share few common sums, is a small fraction of
    the grid.

    How small is a property of the instance, not of the method. Weights of
    10, 20 and 30 in a bag of 50 reach seven states out of 153; ten items
    of weights 1 through 10 in a bag of 15 reach 105 out of 160. The number
    is returned rather than predicted, and the report states whichever way
    it comes out.

    The base case ``index == n`` is not counted: it is a constant, it is
    never stored, and including it would flatter the comparison.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        A ``(best_value, memo_entries, table_cells)`` triple.
        ``best_value`` matches every other variant, ``memo_entries`` is the
        number of distinct states the cache stored, and ``table_cells`` is
        ``len(weights) * (capacity + 1)``, the number of cells
        :func:`knapsack_tab` writes.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``capacity`` or any entry is negative, or if
            ``weights`` and ``values`` differ in length.

    Time Complexity:
        O(n*W) worst case, O(R) for R reachable states in practice - the
        same work as :func:`knapsack_memo`, since counting the entries is
        one ``len()`` at the end and costs nothing during the run.

    Space Complexity:
        O(R) for the cache plus O(n) of call stack.

    Examples:
        Large, coarse weights make the space very sparse indeed - seven
        states stored against 153 table cells:

        >>> knapsack_memo_cells([10, 20, 30], [60, 100, 120], 50)
        (220, 7, 153)

        Fine-grained weights that share many sums fill much more of it:

        >>> knapsack_memo_cells([1, 3, 4, 5], [1, 4, 5, 7], 7)
        (9, 13, 32)

        The value always agrees with the uninstrumented variants:

        >>> value, entries, cells = knapsack_memo_cells(
        ...     [10, 20, 30], [60, 100, 120], 50)
        >>> value == knapsack_tab([10, 20, 30], [60, 100, 120], 50)
        True
        >>> entries <= cells
        True

        An empty item list has no states and no table:

        >>> knapsack_memo_cells([], [], 10)
        (0, 0, 0)
    """
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    cache: Dict[Tuple[int, int], int] = {}
    best_value = _memo(checked_weights, checked_values, 0, capacity, cache)
    table_cells = len(checked_weights) * (capacity + 1)
    return best_value, len(cache), table_cells


def _recursive_counted(
    weights: List[int],
    values: List[int],
    index: int,
    remaining: int,
    counter: CallCounter,
) -> int:
    """Mirror :func:`_recursive`, recording every call in ``counter``.

    A separate function rather than an optional argument on the fast path.
    The counter is threaded through the recursion as a parameter, so two
    instrumented runs in one process keep entirely separate totals and no
    module-level state is involved.

    Args:
        weights: Validated item weights.
        values: Validated item values.
        index: The first item still under consideration.
        remaining: Capacity left in the knapsack.
        counter: Accumulates the call count and the deepest nesting.

    Returns:
        The same value :func:`_recursive` returns for these arguments.

    Time Complexity:
        O(2^n), plus a constant per call for the bookkeeping.

    Space Complexity:
        O(n) - the call stack.

    Examples:
        >>> counter = CallCounter()
        >>> _recursive_counted([1, 2], [1, 2], 0, 3, counter)
        3
        >>> counter.calls, counter.max_depth, counter.depth
        (7, 3, 0)
    """
    with counter.frame():
        if index == len(weights):
            return 0

        skipped = _recursive_counted(
            weights, values, index + 1, remaining, counter
        )

        weight = weights[index]
        if weight > remaining:
            return skipped

        taken = values[index] + _recursive_counted(
            weights, values, index + 1, remaining - weight, counter
        )
        return taken if taken > skipped else skipped


def _memo_counted(
    weights: List[int],
    values: List[int],
    index: int,
    remaining: int,
    memo: Dict[Tuple[int, int], int],
    counter: CallCounter,
) -> int:
    """Mirror :func:`_memo`, recording every call in ``counter``.

    Cache hits are counted as calls, because they are calls: the point of
    the measurement is how many times the recursion was entered, and the
    gap between this total and :func:`_recursive_counted`'s is exactly what
    the memo saved.

    Args:
        weights: Validated item weights.
        values: Validated item values.
        index: The first item still under consideration.
        remaining: Capacity left in the knapsack.
        memo: Cache keyed by ``(index, remaining)``. Mutated in place.
        counter: Accumulates the call count and the deepest nesting.

    Returns:
        The same value :func:`_memo` returns for these arguments.

    Time Complexity:
        O(n*W) worst case, plus a constant per call for the bookkeeping.

    Space Complexity:
        O(n*W) worst case for the cache, plus O(n) of call stack.

    Examples:
        >>> counter = CallCounter()
        >>> cache: Dict[Tuple[int, int], int] = {}
        >>> _memo_counted([1, 3, 4, 5], [1, 4, 5, 7], 0, 7, cache, counter)
        9
        >>> counter.calls, counter.max_depth
        (22, 5)
    """
    with counter.frame():
        if index == len(weights):
            return 0

        key = (index, remaining)
        cached = memo.get(key)
        if cached is not None:
            return cached

        best = _memo_counted(
            weights, values, index + 1, remaining, memo, counter
        )

        weight = weights[index]
        if weight <= remaining:
            taken = values[index] + _memo_counted(
                weights, values, index + 1, remaining - weight, memo, counter
            )
            if taken > best:
                best = taken

        memo[key] = best
        return best


def _tab_counted(
    weights: List[int],
    values: List[int],
    capacity: int,
    counter: CallCounter,
) -> int:
    """Mirror :func:`knapsack_tab`, counting one "call" per table cell.

    Tabulation makes no calls, so the count has to mean something else for
    the comparison to be honest: here it is the number of subproblems
    evaluated, which is one per cell of the ``n x (W + 1)`` grid below row
    0. Row 0 is the base case and is not counted, exactly as the base case
    is not counted in the two recursive variants.

    The depth this records is 1, by construction and not by accident: the
    loop body is entered and left without ever nesting. That single number
    is the space half of the argument - the bottom-up version borrows
    nothing from the interpreter's call stack.

    Args:
        weights: Validated item weights.
        values: Validated item values.
        capacity: Validated knapsack capacity.
        counter: Accumulates the cell count and the nesting depth.

    Returns:
        The same value :func:`knapsack_tab` returns for these arguments.

    Time Complexity:
        O(n*W), plus a constant per cell for the bookkeeping.

    Space Complexity:
        O(n*W) - the full table, as in :func:`knapsack_tab`.

    Examples:
        >>> counter = CallCounter()
        >>> _tab_counted([1, 3, 4, 5], [1, 4, 5, 7], 7, counter)
        9
        >>> counter.calls, counter.max_depth
        (32, 1)
    """
    item_count = len(weights)
    table = [[0] * (capacity + 1) for _ in range(item_count + 1)]

    for item in range(1, item_count + 1):
        weight = weights[item - 1]
        value = values[item - 1]
        previous_row = table[item - 1]
        current_row = table[item]

        for room in range(capacity + 1):
            counter.enter()
            best = previous_row[room]
            if weight <= room:
                candidate = value + previous_row[room - weight]
                if candidate > best:
                    best = candidate
            current_row[room] = best
            counter.leave()

    return table[item_count][capacity]


def knapsack_instrumented(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
    variant: str,
) -> Tuple[int, CallCounter]:
    """Solve the instance and report how much work the chosen variant did.

    The public entry point for the call-count and recursion-depth columns
    of the Week 5 benchmark. It dispatches to private counted mirrors of
    the three algorithms rather than to the algorithms themselves: the
    functions in :data:`VARIANTS` carry no instrumentation whatsoever, so
    that timing them measures the recurrence and not the bookkeeping.

    What a "call" means differs by variant, and the difference is the
    finding rather than a caveat:

    * ``"recursive"`` counts recursive invocations, of which there are
      O(2^n), and reaches depth ``n + 1``.
    * ``"memo"`` counts recursive invocations too, cache hits included.
      The gap between this and the recursive total is what the memo saved.
      Depth is also ``n + 1``: caching removes repeated work, not nesting.
    * ``"tab"`` makes no calls, so it counts subproblems evaluated - one
      per cell of the ``n x (W + 1)`` grid - and reports depth 1.

    Args:
        weights: Item weights, one per item. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.
        variant: One of ``"recursive"``, ``"memo"`` or ``"tab"``.

    Returns:
        A ``(best_value, counter)`` pair. ``best_value`` is what the
        matching function in :data:`VARIANTS` would return, and the counter
        is a fresh :class:`~src.utils.timer.CallCounter` whose ``calls``
        and ``max_depth`` describe this run and no other.

    Raises:
        TypeError: If ``capacity`` is not an ``int``, or either sequence is
            not a sequence of ``int``.
        ValueError: If ``variant`` is not one of the three names, if
            ``capacity`` or any entry is negative, or if ``weights`` and
            ``values`` differ in length.

    Time Complexity:
        That of the selected variant: O(2^n) for ``"recursive"``, O(n*W)
        for ``"memo"`` and ``"tab"``. The bookkeeping adds a constant per
        call or per cell.

    Space Complexity:
        That of the selected variant, plus O(1) for the counter itself.

    Examples:
        Ten items, weights 1 through 10, in a bag of 15. Memoization cuts
        the work by two thirds and changes the nesting not at all:

        >>> weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> value, naive = knapsack_instrumented(weights, values, 15,
        ...                                      "recursive")
        >>> value, naive.calls, naive.max_depth
        (15, 516, 11)
        >>> value, memoized = knapsack_instrumented(weights, values, 15,
        ...                                         "memo")
        >>> value, memoized.calls, memoized.max_depth
        (15, 171, 11)

        Tabulation's depth is 1, which is the whole point of it:

        >>> value, table = knapsack_instrumented([1, 3, 4, 5],
        ...                                      [1, 4, 5, 7], 7, "tab")
        >>> value, table.calls, table.max_depth
        (9, 32, 1)

        Two runs never share state, so counts cannot leak between
        benchmarks:

        >>> _, first = knapsack_instrumented([1, 2], [1, 2], 3, "recursive")
        >>> _, second = knapsack_instrumented([1, 2], [1, 2], 3, "recursive")
        >>> first.calls == second.calls == 7
        True

        An unknown variant is a programming error, not a default:

        >>> knapsack_instrumented([1], [1], 1, "rolling")
        Traceback (most recent call last):
            ...
        ValueError: unknown variant 'rolling'; expected one of 'memo', \
'recursive', 'tab'
    """
    _require_variant(variant)
    checked_weights, checked_values, capacity = _validate(weights, values, capacity)

    counter = CallCounter()

    if variant == "recursive":
        best_value = _recursive_counted(
            checked_weights, checked_values, 0, capacity, counter
        )
    elif variant == "memo":
        cache: Dict[Tuple[int, int], int] = {}
        best_value = _memo_counted(
            checked_weights, checked_values, 0, capacity, cache, counter
        )
    else:
        best_value = _tab_counted(
            checked_weights, checked_values, capacity, counter
        )

    return best_value, counter


VARIANTS = {
    "recursive": knapsack_recursive,
    "memo": knapsack_memo,
    "tab": knapsack_tab,
}
