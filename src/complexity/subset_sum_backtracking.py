"""Subset Sum: exponential backtracking, fast heuristics and a pseudo-polynomial check.

The problem
-----------
Given a list ``nums`` of non-negative integers (repeats allowed) and a
non-negative integer ``target``, is there a sub-multiset of ``nums`` whose
values add up to exactly ``target``? The decision version is NP-complete
(Karp, 1972). It sits in NP because a "yes" answer has a short certificate,
the subset itself, and checking it takes O(n): add the values and confirm
each one was available in ``nums``. No algorithm is known that *finds* such
a subset, or proves that none exists, in time polynomial in the length of
the input. That gap between checking and finding is what this module and
the Week 8 benchmark measure.

Backtracking, and why it is exponential
---------------------------------------
:func:`subset_sum_backtracking` walks the binary decision tree in which
level ``i`` decides whether the ``i``-th value is in or out. The full tree
has ``2^n`` leaves and ``2^(n + 1) - 1`` nodes. Two prunes cut whole
subtrees:

* **Overshoot.** If the running sum already exceeds ``target``, no
  extension can come back down, because every value is non-negative.
* **Shortfall.** If the running sum plus *every* value still undecided is
  below ``target``, no extension can reach it. The undecided total is a
  suffix sum, precomputed once in O(n).

The values are sorted in **descending** order first, so the large values
are decided near the root, where an overshoot prunes the biggest subtrees.
Including before excluding makes the search greedy-first: on many
satisfiable inputs the first path tried is already a solution.

Pruning changes the base of the exponential, not its shape. The worst case
is an instance with no solution that the prunes cannot see through.
:func:`worst_case_instance` builds one by parity: every value is even and
the target is odd, so every running sum is even and never equals the target,
yet the target sits near half the total, so neither prune fires until deep
in the tree. The prunes do skip a growing share of the ``2^(n + 1) - 1``
nodes: on seeded instances the search visits about 10% of the full tree at
n = 10 and 2% to 5% at n = 26. But the visited share shrinks far more
slowly than ``2^n`` grows, so the node count still rises by a roughly
constant factor, about 1.7 to 2, with each added value. Nodes
visited (one per recursive call, the root included) is the
hardware-independent measure of that growth; wall-clock time follows it.

Heuristics: fast, but they cannot say "no"
------------------------------------------
:func:`subset_sum_greedy` sorts once and takes each value that still fits,
in O(n log n). :func:`subset_sum_random_restart` repeats the same
take-if-it-fits pass over random orders and keeps the best. Both always
return a subset whose sum never exceeds ``target``, so the answer is a
valid lower bound, and the *gap* ``target - achieved`` measures how far it
fell short. Both are incomplete in the same way: a positive gap does not
prove that no exact subset exists. On ``[6, 5, 5]`` with target 10 the
greedy pass takes 6 and then nothing else fits, although ``5 + 5`` is
exact.

Pseudo-polynomial dynamic programming and weak NP-completeness
---------------------------------------------------------------
:func:`subset_sum_dp` decides the question in O(n * target) time by
tracking which sums in ``0 .. target`` are reachable. That looks
polynomial, but the input encodes ``target`` in about ``log2(target)``
bits, so ``n * target`` is exponential in the input *length*. An algorithm
that is polynomial in the numeric value of the input rather than its
length is called pseudo-polynomial. Problems that are NP-complete but have
such an algorithm are **weakly** NP-complete (Garey and Johnson, 1979): they
are hard only when the numbers are huge. Subset Sum is one; 3-SAT and
Vertex Cover contain no large numbers to exploit and stay NP-complete even
when every number in the instance is small (strongly NP-complete). In this
repository the DP is used by the tests as an independent truth check, not
as a solver for the benchmark: on the worst-case family the target is
about ``n * 5 * 10^5``, so the DP's ``n * target`` cell updates (about
``n^2 * 5 * 10^5``, roughly 4.5 * 10^8 at n = 30) exceed the whole
backtracking search at every ``n`` the benchmark runs. With numbers this
large, the pseudo-polynomial table costs more than the exponential search.

========================= =================================================
Function                  Purpose
========================= =================================================
subset_sum_backtracking   Exact search with two prunes; optional node count
subset_sum_greedy         Largest-first take-if-it-fits, O(n log n)
subset_sum_random_restart Best of ``tries`` random-order greedy passes
subset_sum_dp             Pseudo-polynomial O(n * target) reachability
worst_case_instance       Even values, odd target: no solution, no early exit
satisfiable_instance      Random values, target = sum of a random subset
========================= =================================================

Examples:
    >>> subset_sum_backtracking([3, 34, 4, 12, 5, 2], 9)
    [5, 4]
    >>> subset_sum_backtracking([3, 34, 4, 12, 5, 2], 30) is None
    True
    >>> subset_sum_greedy([6, 5, 5], 10)
    ([6], 6)
    >>> subset_sum_dp([6, 5, 5], 10)
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from src.utils.random_utils import make_rng

__all__ = [
    "satisfiable_instance",
    "subset_sum_backtracking",
    "subset_sum_dp",
    "subset_sum_greedy",
    "subset_sum_random_restart",
    "worst_case_instance",
]

#: Largest value drawn by the instance generators (inclusive), before the
#: worst-case family doubles it to make every value even.
_MAX_DRAW: int = 10**6


def _is_int(value: Any) -> bool:
    """Return True if ``value`` is an ``int`` and not a ``bool``.

    Args:
        value: Anything.

    Returns:
        True for a genuine integer. ``bool`` is a subclass of ``int`` in
        Python, but ``True`` as a list value or target is an accident, not
        a number, so it is rejected.

    Raises:
        Nothing.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _is_int(3), _is_int(True), _is_int(3.0)
        (True, False, False)
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(nums: Any, target: Any, func_name: str) -> List[int]:
    """Check the shared ``(nums, target)`` arguments and return a copy of ``nums``.

    Every type is checked before any value, so a list holding both a float
    and a negative int raises ``TypeError``.

    Args:
        nums: The object that should be a list of non-negative ints.
        target: The object that should be a non-negative int.
        func_name: Name of the calling function, for the error messages.

    Returns:
        A new list with the same values as ``nums``, so callers can sort it
        without touching the caller's list.

    Raises:
        TypeError: If ``nums`` is not a list, any value is not an int, or
            ``target`` is not an int. ``bool`` is rejected everywhere.
        ValueError: If any value or ``target`` is negative.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the copy.

    Examples:
        >>> _validate([3, 1, 2], 4, "demo")
        [3, 1, 2]
        >>> _validate((3, 1), 4, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects nums to be a list, got tuple
        >>> _validate([3, 1.5], 4, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects int values in nums, got float at index 1
        >>> _validate([3, 1], True, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects an int target, got bool
        >>> _validate([3, -1], 4, "demo")
        Traceback (most recent call last):
            ...
        ValueError: demo() expects non-negative values, got -1 at index 1
        >>> _validate([3, 1], -4, "demo")
        Traceback (most recent call last):
            ...
        ValueError: demo() expects a non-negative target, got -4
    """
    if not isinstance(nums, list):
        raise TypeError(
            f"{func_name}() expects nums to be a list, got {type(nums).__name__}"
        )
    for index, value in enumerate(nums):
        if not _is_int(value):
            raise TypeError(
                f"{func_name}() expects int values in nums, "
                f"got {type(value).__name__} at index {index}"
            )
    if not _is_int(target):
        raise TypeError(
            f"{func_name}() expects an int target, got {type(target).__name__}"
        )
    for index, value in enumerate(nums):
        if value < 0:
            raise ValueError(
                f"{func_name}() expects non-negative values, "
                f"got {value} at index {index}"
            )
    if target < 0:
        raise ValueError(f"{func_name}() expects a non-negative target, got {target}")
    return list(nums)


def _require_count(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` if it is an int of at least ``minimum``, else raise.

    Args:
        value: A requested size or repeat count.
        name: Its parameter name, for the error message.
        minimum: The smallest allowed value.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an int (``bool`` rejected).
        ValueError: If ``value < minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_count(5, "n", 0)
        5
        >>> _require_count(0, "tries", 1)
        Traceback (most recent call last):
            ...
        ValueError: tries must be at least 1, got 0
        >>> _require_count(2.0, "n", 0)
        Traceback (most recent call last):
            ...
        TypeError: n must be an int, got float
    """
    if not _is_int(value):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _take_if_fits(order: List[int], target: int) -> Tuple[List[int], int]:
    """One greedy pass: walk ``order`` and take each value that still fits.

    A value fits when adding it keeps the running sum at or below
    ``target``. The pass never backtracks, so once a value is taken it
    stays, which is exactly why it can miss an exact answer.

    Args:
        order: The values in the order to consider them. Not modified.
        target: The non-negative sum not to exceed.

    Returns:
        ``(subset, achieved)``: the values taken, in the order taken, and
        their sum. ``achieved <= target`` always.

    Raises:
        Nothing for valid input; callers validate first.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the subset.

    Examples:
        >>> _take_if_fits([6, 5, 5], 10)
        ([6], 6)
        >>> _take_if_fits([5, 6, 5], 10)
        ([5, 5], 10)
    """
    subset: List[int] = []
    achieved = 0
    for value in order:
        if achieved + value <= target:
            subset.append(value)
            achieved += value
    return subset, achieved


def subset_sum_backtracking(
    nums: List[int], target: int, count_nodes: bool = False
) -> Union[Optional[List[int]], Tuple[Optional[List[int]], int]]:
    """Find a sub-multiset of ``nums`` summing to ``target`` by exhaustive backtracking.

    The values are sorted in descending order and a suffix-sum array is
    built, so ``suffix[i]`` is the total of every value not yet decided at
    depth ``i``. The recursive search ``visit(i, running)`` then does, in
    this order:

    1. count the node (every call, the root included);
    2. succeed if ``running == target``;
    3. prune if ``running > target`` (overshoot);
    4. prune if ``running + suffix[i] < target`` (shortfall; this also
       ends every path at ``i == n``, where the suffix is empty);
    5. try including value ``i``, and only if that fails, excluding it.

    It is exact and complete: it returns a subset whenever one exists and
    ``None`` only after ruling every subset out. That completeness is what
    costs exponential time in the worst case (see the module docstring).
    ``target == 0`` is answered by the empty subset at the root, one node.

    The recursion is at most ``n + 1`` calls deep. The course runs it on at
    most about 40 values, far inside Python's default limit of 1000, and the
    limit is deliberately not raised. A list of roughly 1000 or more values
    whose search reaches that depth (for example ``[1] * 1000`` with target
    999, which succeeds on the first path) hits the limit instead.

    Args:
        nums: Non-negative ints; repeats allowed. Not modified.
        target: The non-negative sum to hit exactly.
        count_nodes: If true, also return the number of recursive calls.

    Returns:
        A list of values from ``nums`` that sums to ``target``, in the order
        chosen (descending), or ``None`` if no such subset exists. With
        ``count_nodes``, the pair ``(result, nodes_visited)``.

    Raises:
        TypeError: If ``nums`` is not a list, or any value or ``target`` is
            not an int (``bool`` rejected).
        ValueError: If any value or ``target`` is negative.
        RecursionError: If the search goes deeper than Python's recursion
            limit, which needs about 1000 or more values (see above).

    Time Complexity:
        Best:  O(n log n) - the sort, then a first path that succeeds.
        Worst: O(2^n) - at most ``2^(n + 1) - 1`` calls, each O(1), on an
               instance with no solution the prunes cannot see through.

    Space Complexity:
        O(n) - the sorted copy, the suffix sums, the chosen path and
        ``n + 1`` stack frames.

    Examples:
        The textbook instance. Sorted descending it is
        ``[34, 12, 5, 4, 3, 2]``: 34 and 12 each overshoot, then 5 and 4
        hit 9. Seven calls in all:

        >>> subset_sum_backtracking([3, 34, 4, 12, 5, 2], 9, count_nodes=True)
        ([5, 4], 7)

        Without 34 the total is 26 < 30, and 34 alone overshoots:

        >>> subset_sum_backtracking([3, 34, 4, 12, 5, 2], 30, count_nodes=True)
        (None, 3)
        >>> subset_sum_backtracking([3, 34, 4, 12, 5, 2], 0, count_nodes=True)
        ([], 1)

        Repeats are separate items, and the input is left alone:

        >>> data = [5, 5, 6]
        >>> subset_sum_backtracking(data, 10), data
        ([5, 5], [5, 5, 6])

        A seeded worst-case instance has no solution:

        >>> nums, target = worst_case_instance(10, seed=42)
        >>> result, nodes = subset_sum_backtracking(nums, target, count_nodes=True)
        >>> result is None, nodes <= 2 ** 11 - 1
        (True, True)
        >>> subset_sum_backtracking([1, 2], -1)
        Traceback (most recent call last):
            ...
        ValueError: subset_sum_backtracking() expects a non-negative target, got -1
    """
    values = _validate(nums, target, "subset_sum_backtracking")

    if target == 0:
        return ([], 1) if count_nodes else []

    values.sort(reverse=True)
    n = len(values)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]

    chosen: List[int] = []
    nodes = 0

    def visit(i: int, running: int) -> bool:
        """Search the subtree that decides values ``i .. n - 1``."""
        nonlocal nodes
        nodes += 1
        if running == target:
            return True
        if running > target:
            return False
        if running + suffix[i] < target:
            return False
        chosen.append(values[i])
        if visit(i + 1, running + values[i]):
            return True
        chosen.pop()
        return visit(i + 1, running)

    result = list(chosen) if visit(0, 0) else None
    return (result, nodes) if count_nodes else result


def subset_sum_greedy(nums: List[int], target: int) -> Tuple[List[int], int]:
    """Approximate Subset Sum by taking the largest values that still fit.

    Sorts the values in descending order and makes one pass, adding each
    value whose inclusion keeps the running sum at or below ``target``. It
    never exceeds the target and never backtracks, so it is fast but
    incomplete: it may stop short of a target that an exact search would
    hit, and a shortfall proves nothing. Report the gap
    ``target - achieved`` alongside the result.

    Args:
        nums: Non-negative ints; repeats allowed. Not modified.
        target: The non-negative sum to approach from below.

    Returns:
        ``(subset, achieved_sum)``: the values taken, in descending order,
        and their total. ``achieved_sum <= target`` always.

    Raises:
        TypeError: If ``nums`` is not a list, or any value or ``target`` is
            not an int (``bool`` rejected).
        ValueError: If any value or ``target`` is negative.

    Time Complexity:
        O(n log n) - the sort; the pass itself is O(n).

    Space Complexity:
        O(n) - the sorted copy and the subset.

    Examples:
        >>> subset_sum_greedy([3, 34, 4, 12, 5, 2], 9)
        ([5, 4], 9)
        >>> subset_sum_greedy([3, 34, 4, 12, 5, 2], 30)
        ([12, 5, 4, 3, 2], 26)

        Greedy can miss an exact answer that backtracking finds:

        >>> subset_sum_greedy([6, 5, 5], 10)
        ([6], 6)
        >>> subset_sum_backtracking([6, 5, 5], 10)
        [5, 5]
        >>> subset_sum_greedy([], 7)
        ([], 0)
    """
    values = _validate(nums, target, "subset_sum_greedy")
    values.sort(reverse=True)
    return _take_if_fits(values, target)


def subset_sum_random_restart(
    nums: List[int], target: int, tries: int = 100, seed: Optional[int] = None
) -> Tuple[List[int], int]:
    """Approximate Subset Sum with the best of ``tries`` random-order greedy passes.

    Each try shuffles a fresh copy of the values with a generator from
    :func:`~src.utils.random_utils.make_rng` and runs the same
    take-if-it-fits pass as :func:`subset_sum_greedy`, but in that random
    order. The best achieved sum is kept (on a tie, the first one found),
    and the search stops early as soon as a pass hits ``target`` exactly.
    Random orders escape the one fixed mistake that largest-first greedy
    always makes, but the method is still a heuristic: when every try
    falls short it cannot tell an unlucky search from an impossible target.

    Args:
        nums: Non-negative ints; repeats allowed. Not modified.
        target: The non-negative sum to approach from below.
        tries: How many random orders to try, at least 1.
        seed: Passed to :func:`~src.utils.random_utils.make_rng`. ``None``
            draws an unpredictable seed from the operating system.

    Returns:
        ``(subset, achieved_sum)`` from the best try: the values taken, in
        the order taken, and their total. ``achieved_sum <= target``.

    Raises:
        TypeError: If ``nums`` is not a list, any value, ``target`` or
            ``tries`` is not an int (``bool`` rejected), or ``seed`` is not
            an int or ``None``.
        ValueError: If any value or ``target`` is negative, or
            ``tries < 1``.

    Time Complexity:
        O(tries * n) - one O(n) shuffle and one O(n) pass per try.

    Space Complexity:
        O(n) - the shuffled copy and the best subset.

    Examples:
        Any order that puts a 5 before the 6 finds the exact answer, so
        100 tries all miss it only with probability ``3^-100``:

        >>> subset_sum_random_restart([6, 5, 5], 10, seed=42)
        ([5, 5], 10)

        The same seed gives the same answer:

        >>> nums, target = satisfiable_instance(12, seed=7)
        >>> a = subset_sum_random_restart(nums, target, tries=20, seed=1)
        >>> b = subset_sum_random_restart(nums, target, tries=20, seed=1)
        >>> a == b, a[1] <= target, sum(a[0]) == a[1]
        (True, True, True)
        >>> subset_sum_random_restart([1, 2], 3, tries=0)
        Traceback (most recent call last):
            ...
        ValueError: tries must be at least 1, got 0
    """
    values = _validate(nums, target, "subset_sum_random_restart")
    _require_count(tries, "tries", 1)
    rng = make_rng(seed)

    best_subset: List[int] = []
    best_sum = -1
    for _ in range(tries):
        order = list(values)
        rng.shuffle(order)
        subset, achieved = _take_if_fits(order, target)
        if achieved > best_sum:
            best_subset, best_sum = subset, achieved
            if achieved == target:
                break
    return best_subset, best_sum


def subset_sum_dp(nums: List[int], target: int) -> bool:
    """Decide Subset Sum exactly in pseudo-polynomial O(n * target) time.

    ``reachable[s]`` records whether some subset of the values seen so far
    sums to ``s``, for ``s`` in ``0 .. target``. Initially only 0 is
    reachable (the empty subset). Each value ``v`` then makes ``s``
    reachable if ``s - v`` already was; scanning ``s`` downward means
    ``s - v`` still holds the answer from *before* ``v``, so each value is
    used at most once (the standard 0/1 knapsack trick). Values larger than
    ``target`` can never be part of an answer and are skipped.

    The running time is polynomial in the *value* of ``target`` but
    exponential in its length in bits, which is why this does not put an
    NP-complete problem in P. Subset Sum is NP-complete only in the weak
    sense: it is hard only when its numbers are large. The tests use this
    function as an independent check on :func:`subset_sum_backtracking`.

    Args:
        nums: Non-negative ints; repeats allowed. Not modified.
        target: The non-negative sum to test.

    Returns:
        True if some sub-multiset of ``nums`` sums to exactly ``target``.

    Raises:
        TypeError: If ``nums`` is not a list, or any value or ``target`` is
            not an int (``bool`` rejected).
        ValueError: If any value or ``target`` is negative.

    Time Complexity:
        O(n * target) - pseudo-polynomial.

    Space Complexity:
        O(target) - one boolean row.

    Examples:
        >>> subset_sum_dp([3, 34, 4, 12, 5, 2], 9)
        True
        >>> subset_sum_dp([3, 34, 4, 12, 5, 2], 30)
        False
        >>> subset_sum_dp([], 0), subset_sum_dp([], 1)
        (True, False)

        Each copy of a repeated value counts once:

        >>> subset_sum_dp([5], 10), subset_sum_dp([5, 5], 10)
        (False, True)
    """
    values = _validate(nums, target, "subset_sum_dp")
    reachable = [False] * (target + 1)
    reachable[0] = True
    for value in values:
        if value > target:
            continue
        for s in range(target, value - 1, -1):
            if reachable[s - value]:
                reachable[s] = True
        if reachable[target]:
            return True
    return reachable[target]


def worst_case_instance(n: int, seed: Optional[int]) -> Tuple[List[int], int]:
    """Build an unsolvable instance that backtracking cannot cut short.

    Draws ``nums = [2 * rng.randint(1, 10**6) for _ in range(n)]`` and sets
    ``target = (sum(nums) // 2) | 1``, with ``rng = make_rng(seed)``. Every
    value is even and the target is odd, so every subset sum is even and
    none can equal the target: the answer is always "no", and an exact
    solver can only say so after ruling out every subset. Putting the
    target at half the total keeps both prunes quiet for as long as
    possible, since a running sum is rarely far above or below it near the
    root. Because the values come from one generator in sequence, the
    instance for ``n`` extends the one for ``n - 1`` with the same seed.

    Args:
        n: Number of values, at least 0.
        seed: Passed to :func:`~src.utils.random_utils.make_rng`.

    Returns:
        ``(nums, target)``.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected), or ``seed``
            is not an int or ``None``.
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the values.

    Examples:
        >>> nums, target = worst_case_instance(6, seed=42)
        >>> len(nums), all(v % 2 == 0 for v in nums), target % 2
        (6, True, 1)
        >>> target == (sum(nums) // 2) | 1
        True
        >>> worst_case_instance(6, seed=42) == (nums, target)
        True
        >>> worst_case_instance(0, seed=42)
        ([], 1)
    """
    _require_count(n, "n", 0)
    rng = make_rng(seed)
    nums = [2 * rng.randint(1, _MAX_DRAW) for _ in range(n)]
    target = (sum(nums) // 2) | 1
    return nums, target


def satisfiable_instance(n: int, seed: Optional[int]) -> Tuple[List[int], int]:
    """Build a random instance that is guaranteed to have a solution.

    Draws ``nums = [rng.randint(1, 10**6) for _ in range(n)]`` and then,
    with the same generator ``rng = make_rng(seed)``, picks a random
    nonempty subset and returns its sum as the target. Each value joins the
    subset independently with probability 1/2 and an empty draw is
    repeated, which makes every one of the ``2^n - 1`` nonempty subsets
    equally likely. These instances show what a heuristic misses: an exact
    solver always succeeds on them, so any shortfall of greedy or random
    restart is the heuristic's own error.

    Args:
        n: Number of values, at least 1 (a nonempty subset must exist).
        seed: Passed to :func:`~src.utils.random_utils.make_rng`.

    Returns:
        ``(nums, target)``, where some nonempty sub-multiset of ``nums``
        sums to ``target``.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected), or ``seed``
            is not an int or ``None``.
        ValueError: If ``n < 1``.

    Time Complexity:
        O(n) expected - the empty draw is repeated with probability
        ``2^-n`` per round.

    Space Complexity:
        O(n) - the values.

    Examples:
        >>> nums, target = satisfiable_instance(8, seed=42)
        >>> len(nums), all(1 <= v <= 10**6 for v in nums)
        (8, True)
        >>> 1 <= target <= sum(nums)
        True
        >>> subset_sum_backtracking(nums, target) is not None
        True
        >>> satisfiable_instance(8, seed=42) == (nums, target)
        True
        >>> satisfiable_instance(0, seed=42)
        Traceback (most recent call last):
            ...
        ValueError: n must be at least 1, got 0
    """
    _require_count(n, "n", 1)
    rng = make_rng(seed)
    nums = [rng.randint(1, _MAX_DRAW) for _ in range(n)]
    while True:
        picked = [value for value in nums if rng.random() < 0.5]
        if picked:
            return nums, sum(picked)
