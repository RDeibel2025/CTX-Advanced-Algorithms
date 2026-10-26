"""Set Cover: the integer program, its LP relaxation, greedy, and two roundings.

An instance is a finite universe U of n elements, a list of m subsets
S_0 .. S_{m-1} of U and a non-negative cost c_S for each subset. A *cover* is
a collection of subsets whose union is U; the problem is to find a cover of
minimum total cost. Set Cover is NP-hard (it generalizes Vertex Cover: the
elements are the edges and each vertex is the set of edges touching it), so
this module compares four polynomial-time approximations against the exact
optimum on small instances.

The integer program (IP)
------------------------
One 0/1 variable x_S per subset ("S is chosen"):

    minimize    sum_S c_S x_S
    subject to  sum_{S contains e} x_S >= 1     for every element e in U
                x_S in {0, 1}                   for every subset S.

A 0/1 vector is feasible exactly when the chosen sets cover every element, so
the IP optimum is OPT. :func:`set_cover_ip` returns this program as data in
the ``linprog`` convention (the covering constraint is written
``-sum x_S <= -1`` so it fits ``A_ub``), and :func:`set_cover_exact` solves
it with SciPy's HiGHS MILP solver (branch and bound, exponential in the worst
case).

The LP relaxation
-----------------
Replace x_S in {0, 1} by 0 <= x_S <= 1. Every IP solution is LP-feasible, so
the LP minimizes over a larger set and LP* <= OPT. :func:`set_cover_lp`
solves it in polynomial time. LP* is the certified lower bound both rounding
analyses below charge against: a rounding that costs at most alpha * LP* is
an alpha-approximation, because alpha * LP* <= alpha * OPT. The instance
U = {1, 2, 3} with sets {1, 2}, {2, 3}, {1, 3} has LP* = 1.5 (every x_S = 1/2)
and OPT = 2, an integrality gap of 4/3.

Chvatal's greedy and its H(n) bound
-----------------------------------
:func:`set_cover_greedy` repeatedly picks the set with the lowest cost per
newly covered element, c_S / |S minus covered| (ties go to the lowest index),
until everything is covered. Charge each set's cost evenly to the elements
it newly covers: price(e) = c_S / (number of elements S newly covers), so the
greedy cost equals the sum of all prices.

* **H(n) bound.** List the elements in the order greedy covers them,
  e_1, .., e_n. Just before e_k is covered at least n - k + 1 elements are
  uncovered, and OPT's sets cover them at total cost OPT, so some set of OPT
  covers them at cost per element at most OPT / (n - k + 1). Greedy picks
  the cheapest ratio available, so price(e_k) <= OPT / (n - k + 1). Summing,
  greedy <= OPT (1/n + 1/(n-1) + ... + 1) = H(n) OPT, about (ln n + 1) OPT.
* **H(d) refinement, d = max |S|.** Fix any set S with elements s_1, .., s_k
  in the order greedy covers them. Just before s_i is covered at least
  k - i + 1 elements of S are uncovered, so S itself offers ratio at most
  c_S / (k - i + 1) and price(s_i) <= c_S / (k - i + 1). Summing over S
  gives at most H(k) c_S. Every element lies in some set of OPT, so
  greedy = sum_e price(e) <= sum_{S in OPT} H(|S|) c_S <= H(d) OPT.
* **The bound is not slack in general.** Feige (1998) showed that no
  polynomial-time algorithm achieves (1 - eps) ln n unless NP has
  quasi-polynomial algorithms, so greedy is essentially optimal. The "greedy
  trap" in the examples shows the mechanism: two halves (odds, evens) form
  the optimum of 2, but blocks of sizes 2, 4, 8 each look cheaper per new
  element at the moment greedy chooses, so greedy takes all three blocks.
  With blocks of sizes 2, 4, .., 2^k the ratio is k / 2 = Omega(log n).

Deterministic rounding: an f-approximation
------------------------------------------
Let f be the *frequency*, the largest number of sets containing any one
element (:func:`frequency`). :func:`set_cover_deterministic_rounding` solves
the LP and picks every set with x_S >= 1/f.

* **It returns a cover.** An element e lies in at most f sets and its
  constraint says their x values sum to at least 1, so the largest of them
  is at least 1/f: that set is picked.
* **It costs at most f LP* <= f OPT.** Every picked set has f x_S >= 1, so
  cost = sum_{picked} c_S <= sum_{picked} c_S f x_S <= f sum_S c_S x_S
  = f LP*.

For Vertex Cover f = 2, and this is the LP-rounding 2-approximation. Because
a floating-point LP solver returns x only up to its feasibility tolerance,
the threshold is 1/f - 1e-9, and any element still uncovered after that
(only possible through solver round-off) gets the containing set of largest
x_S; such a set has x_S >= (1 - tolerance)/f, so the bound is unchanged up
to that tolerance.

Randomized rounding: O(log n) in expectation
--------------------------------------------
:func:`set_cover_randomized_rounding` runs k = ceil(2 ln n) + 1 rounds; in
each round every set S is picked independently with probability x_S. Any
element left uncovered after the rounds is fixed by adding the cheapest set
containing it, and those *fix-ups* are counted.

* **One round covers a fixed element with probability >= 1 - 1/e.** e stays
  uncovered only if none of its sets is picked:
  prod_{S contains e} (1 - x_S) <= prod exp(-x_S) = exp(-sum x_S) <= 1/e,
  using 1 - t <= e^{-t} and the LP constraint sum x_S >= 1.
* **After c ln n rounds an element is uncovered with probability at most
  n^-c.** Rounds are independent, so the probability is at most
  e^{-c ln n} = n^{-c}. With k >= 2 ln n rounds it is at most n^-2, and by
  the union bound the rounds alone fail to cover U with probability at most
  n * n^-2 = 1/n.
* **Expected cost per round is LP*.** E[cost of one round] =
  sum_S c_S Pr[S picked] = sum_S c_S x_S = LP*. The union over k rounds
  costs at most the sum of the rounds, so its expected cost is at most
  k LP* <= (2 ln n + 2) LP*.
* **Fix-ups are cheap in expectation.** The cheapest set containing e costs
  at most OPT (OPT itself contains a set holding e), and e needs a fix-up
  with probability at most n^-2, so the fix-ups cost at most
  n * n^-2 * OPT = OPT / n in expectation.

Altogether E[cost] <= (2 ln n + 2) LP* + OPT / n <= (2 ln n + 2 + 1/n) OPT,
an O(log n)-approximation in expectation, and the result is always a cover.
The cost is a random variable; the Week 9 benchmark reports its mean,
standard deviation, extremes and fix-up rate over 200 seeded runs, passing
one precomputed LP solution through the ``lp=`` keyword so the LP is solved
once per instance rather than once per run.

======================================= ======================== ==========
Function                                Time                     Space
======================================= ======================== ==========
set_cover_ip                            O(n m)                   O(n m)
set_cover_lp                            O(n m) build + one LP    O(n m)
                                        (polynomial)
set_cover_exact                         exponential worst case   O(n m) +
                                        (branch and bound)       B&B tree
set_cover_greedy                        O(min(n, m) N)           O(N)
set_cover_deterministic_rounding        one LP + O(N)            O(n m)
set_cover_randomized_rounding           one LP + O(k m + N)      O(n m)
frequency                               O(n log n + N)           O(n + N)
random_instance                         O(n m)                   O(n m)
======================================= ======================== ==========

Here n = |U|, m = number of sets, N = sum of the set sizes and k the number
of rounds.

Representation: an instance is ``(universe, subsets, costs)`` with
``universe`` a set, ``subsets`` a list of sets and ``costs`` a list of
non-negative numbers, one per subset, or ``None`` for unit costs. Elements of
a subset that are not in the universe are ignored. Every method returns the
chosen subsets as a list of indices into ``subsets``; the constraint rows
follow the sorted order of the universe.

Examples:
    >>> universe = set(range(1, 15))
    >>> trap = [{1, 3, 5, 7, 9, 11, 13}, {2, 4, 6, 8, 10, 12, 14},
    ...         {1, 2}, {3, 4, 5, 6}, {7, 8, 9, 10, 11, 12, 13, 14}]
    >>> set_cover_greedy(universe, trap)
    ([4, 3, 2], 3.0)
    >>> set_cover_exact(universe, trap)
    ([0, 1], 2.0)
    >>> set_cover_lp(universe, trap)[0], frequency(universe, trap)
    (2.0, 2)
    >>> set_cover_deterministic_rounding(universe, trap)
    ([0, 1], 2.0, 2.0)
    >>> set_cover_randomized_rounding(universe, trap, seed=42)
    ([0, 1], 2.0, 2.0)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from src.utils.lp_solver import solve_lp, solve_milp
from src.utils.random_utils import make_rng

__all__ = [
    "frequency",
    "random_instance",
    "set_cover_deterministic_rounding",
    "set_cover_exact",
    "set_cover_greedy",
    "set_cover_ip",
    "set_cover_lp",
    "set_cover_randomized_rounding",
]

#: Slack on the rounding threshold 1/f, absorbing LP solver round-off.
_ROUND_TOL = 1e-9

#: A validated instance: sorted elements, subsets restricted to the
#: universe, float costs, and for each element the indices of its sets.
_Instance = Tuple[List[Hashable], List[Set[Hashable]], List[float], Dict[Hashable, List[int]]]


def _sorted_elements(universe: Set[Hashable]) -> List[Hashable]:
    """Return the universe in a deterministic order.

    Args:
        universe: A set of hashable elements.

    Returns:
        ``sorted(universe)`` when the elements are mutually comparable,
        otherwise the elements sorted by ``(type name, repr)``.

    Time Complexity:
        O(n log n) for n elements.

    Space Complexity:
        O(n).

    Examples:
        >>> _sorted_elements({3, 1, 2})
        [1, 2, 3]
        >>> _sorted_elements({1, "a"})
        [1, 'a']
    """
    try:
        return sorted(universe)
    except TypeError:
        return sorted(universe, key=lambda e: (type(e).__name__, repr(e)))


def _validate(universe: Any, subsets: Any, costs: Any) -> _Instance:
    """Check a set cover instance and return it in canonical form.

    Args:
        universe: A non-string iterable of hashable elements.
        subsets: A sequence of non-string iterables of hashable elements.
        costs: ``None`` (unit costs) or a sequence of finite, non-negative
            real numbers, one per subset.

    Returns:
        ``(elements, sets, cost_list, containing)``: the sorted elements,
        each subset as a ``set`` restricted to the universe, the costs as
        floats, and for each element the increasing list of indices of the
        subsets containing it.

    Raises:
        TypeError: If ``universe``, ``subsets`` or a subset is a string or
            not iterable, or a cost is not a real number (``bool`` is
            rejected).
        ValueError: If ``costs`` has the wrong length, a cost is negative or
            not finite, or some element of the universe is in no subset (no
            cover exists).

    Time Complexity:
        O(n log n + N) for n elements and total subset size N.

    Space Complexity:
        O(n + N).

    Examples:
        >>> elements, sets, costs, containing = _validate({1, 2}, [{1}, {1, 2, 9}], None)
        >>> elements, sets, costs, containing
        ([1, 2], [{1}, {1, 2}], [1.0, 1.0], {1: [0, 1], 2: [1]})
        >>> _validate({1, 2}, [{1}], None)
        Traceback (most recent call last):
            ...
        ValueError: element 2 is in no subset, so no cover exists
    """
    if isinstance(universe, (str, bytes)) or not isinstance(universe, Iterable):
        raise TypeError("universe must be a non-string iterable of elements")
    if isinstance(subsets, (str, bytes)) or not isinstance(subsets, Iterable):
        raise TypeError("subsets must be a sequence of sets")
    universe_set = set(universe)
    sets: List[Set[Hashable]] = []
    for index, subset in enumerate(subsets):
        if isinstance(subset, (str, bytes)) or not isinstance(subset, Iterable):
            raise TypeError(f"subset {index} must be a non-string iterable")
        sets.append(set(subset) & universe_set)
    if costs is None:
        cost_list = [1.0] * len(sets)
    else:
        cost_list = []
        for index, cost in enumerate(costs):
            if not isinstance(cost, numbers.Real) or isinstance(cost, bool):
                raise TypeError(f"cost {index} must be a real number, got {type(cost).__name__}")
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"cost {index} must be finite and non-negative, got {cost}")
            cost_list.append(float(cost))
        if len(cost_list) != len(sets):
            raise ValueError(f"expected {len(sets)} costs, got {len(cost_list)}")
    elements = _sorted_elements(universe_set)
    containing: Dict[Hashable, List[int]] = {e: [] for e in elements}
    for index, subset in enumerate(sets):
        for e in subset:
            containing[e].append(index)
    for e in elements:
        if not containing[e]:
            raise ValueError(f"element {e!r} is in no subset, so no cover exists")
        containing[e].sort()
    return elements, sets, cost_list, containing


def _total(chosen: Iterable[int], costs: Sequence[float]) -> float:
    """Return the exact-rounded total cost of the chosen subsets.

    Args:
        chosen: Indices of chosen subsets.
        costs: Cost of every subset.

    Returns:
        ``math.fsum`` of the chosen costs, so the result does not depend on
        the order of ``chosen``.

    Time Complexity:
        O(k) for k chosen subsets.

    Space Complexity:
        O(k).

    Examples:
        >>> _total([2, 0], [1.5, 9.0, 0.25])
        1.75
    """
    return math.fsum(costs[i] for i in chosen)


def _ip_data(elements: List[Hashable], containing: Dict[Hashable, List[int]],
             costs: List[float]) -> Dict[str, Any]:
    """Build the covering integer program from a validated instance.

    Args:
        elements: Sorted elements (one constraint row each).
        containing: For each element, the indices of the sets holding it.
        costs: One cost per subset.

    Returns:
        A dict with keys ``c``, ``A_ub``, ``b_ub``, ``bounds`` and
        ``integrality`` as described in :func:`set_cover_ip`.

    Time Complexity:
        O(n m) for the dense matrix.

    Space Complexity:
        O(n m).

    Examples:
        >>> _ip_data([1], {1: [0]}, [2.0])
        {'c': [2.0], 'A_ub': [[-1.0]], 'b_ub': [-1.0], 'bounds': (0, 1), 'integrality': [1]}
    """
    m = len(costs)
    rows = []
    for e in elements:
        row = [0.0] * m
        for index in containing[e]:
            row[index] = -1.0
        rows.append(row)
    return {
        "c": list(costs),
        "A_ub": rows,
        "b_ub": [-1.0] * len(elements),
        "bounds": (0, 1),
        "integrality": [1] * m,
    }


def _solve_relaxation(elements: List[Hashable], containing: Dict[Hashable, List[int]],
                      costs: List[float]) -> Tuple[float, List[float]]:
    """Solve the LP relaxation of a validated instance.

    Args:
        elements: Sorted elements (one covering constraint each).
        containing: For each element, the indices of the sets holding it.
        costs: One cost per subset.

    Returns:
        ``(lp_value, x)`` from :func:`src.utils.lp_solver.solve_lp` with
        ``0 <= x_S <= 1``; ``(0.0, [0.0] * m)`` when there are no elements.

    Raises:
        RuntimeError: If the LP solver fails to report an optimum.

    Time Complexity:
        O(n m) to build the matrix plus one LP solve.

    Space Complexity:
        O(n m).

    Examples:
        >>> _solve_relaxation([1, 2], {1: [0], 2: [0, 1]}, [1.0, 1.0])
        (1.0, [1.0, 0.0])
        >>> _solve_relaxation([], {}, [1.0])
        (0.0, [0.0])
    """
    if not elements:
        return 0.0, [0.0] * len(costs)
    ip = _ip_data(elements, containing, costs)
    return solve_lp(ip["c"], A_ub=ip["A_ub"], b_ub=ip["b_ub"], bounds=ip["bounds"])


def _check_lp(lp: Any, m: int) -> Tuple[float, List[float]]:
    """Validate a caller-supplied LP solution ``(lp_value, x)``.

    Args:
        lp: A pair ``(lp_value, x)`` with ``x`` holding one number per set.
        m: Number of subsets.

    Returns:
        ``(float(lp_value), [float(v) for v in x])``.

    Raises:
        ValueError: If ``lp`` is not a pair of a number and a sequence of
            numbers, or ``x`` has the wrong length.

    Time Complexity:
        O(m).

    Space Complexity:
        O(m).

    Examples:
        >>> _check_lp((2, [1, 0]), 2)
        (2.0, [1.0, 0.0])
        >>> _check_lp((2, [1]), 2)
        Traceback (most recent call last):
            ...
        ValueError: lp x must have one entry per subset (2), got 1
        >>> _check_lp((2, 5), 1)
        Traceback (most recent call last):
            ...
        ValueError: lp must be a pair (lp_value, x) of a number and a sequence of numbers
    """
    try:
        lp_value, x = lp
        lp_value = float(lp_value)
        x = [float(v) for v in x]
    except (TypeError, ValueError):
        raise ValueError(
            "lp must be a pair (lp_value, x) of a number and a sequence of numbers"
        ) from None
    if len(x) != m:
        raise ValueError(f"lp x must have one entry per subset ({m}), got {len(x)}")
    return lp_value, x


def set_cover_ip(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
    costs: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Return the set cover integer program as solver-ready data.

    The program is: minimize sum_S c_S x_S subject to
    sum_{S contains e} x_S >= 1 for every element e and x_S in {0, 1}. Each
    covering constraint is written ``-sum x_S <= -1`` so that it fits the
    ``A_ub x <= b_ub`` form of :func:`src.utils.lp_solver.solve_lp` and
    :func:`src.utils.lp_solver.solve_milp`. Dropping ``integrality`` gives
    the LP relaxation.

    Args:
        universe: The elements to cover (a set or other non-string
            iterable).
        subsets: The candidate sets; elements outside the universe are
            ignored.
        costs: One non-negative cost per subset, or ``None`` for unit costs.

    Returns:
        A dict with keys ``c`` (costs, one per set), ``A_ub`` (one row per
        element in sorted order, -1.0 in column S when S contains the
        element, else 0.0), ``b_ub`` (all -1.0), ``bounds`` (``(0, 1)`` for
        every variable) and ``integrality`` (all 1). For an empty universe
        ``A_ub`` and ``b_ub`` are empty lists.

    Raises:
        TypeError: If the inputs are not iterables of the right shape or a
            cost is not a real number.
        ValueError: If an element is in no subset, ``costs`` has the wrong
            length, or a cost is negative or not finite.

    Time Complexity:
        O(n m) for n elements and m subsets (dense matrix).

    Space Complexity:
        O(n m).

    Examples:
        >>> ip = set_cover_ip({1, 2, 3}, [{1, 2}, {2, 3}, {1, 3}])
        >>> ip["c"], ip["b_ub"], ip["bounds"], ip["integrality"]
        ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], (0, 1), [1, 1, 1])
        >>> ip["A_ub"]
        [[-1.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [0.0, -1.0, -1.0]]
        >>> set_cover_ip(set(), [{1}], [5])["A_ub"]
        []
    """
    elements, _sets, cost_list, containing = _validate(universe, subsets, costs)
    return _ip_data(elements, containing, cost_list)


def set_cover_lp(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
    costs: Optional[Sequence[float]] = None,
) -> Tuple[float, List[float]]:
    """Solve the LP relaxation of set cover (0 <= x_S <= 1).

    The relaxation of :func:`set_cover_ip`, solved with
    :func:`src.utils.lp_solver.solve_lp` (SciPy HiGHS). Its value LP* is a
    lower bound on OPT because every cover is LP-feasible, and its solution
    ``x`` is what the two rounding schemes round.

    Args:
        universe: The elements to cover.
        subsets: The candidate sets; elements outside the universe are
            ignored.
        costs: One non-negative cost per subset, or ``None`` for unit costs.

    Returns:
        ``(lp_value, x)``: the optimal LP value and one fractional value
        per subset, in subset order. ``(0.0, [0.0] * m)`` for an empty
        universe.

    Raises:
        TypeError: If the inputs are not iterables of the right shape or a
            cost is not a real number.
        ValueError: If an element is in no subset, ``costs`` has the wrong
            length, or a cost is negative or not finite.
        RuntimeError: If the LP solver fails to report an optimum.

    Time Complexity:
        O(n m) to build the matrix plus one LP solve, polynomial in
        practice with HiGHS.

    Space Complexity:
        O(n m).

    Examples:
        >>> set_cover_lp({1, 2, 3}, [{1, 2}, {2, 3}, {1, 3}])
        (1.5, [0.5, 0.5, 0.5])
        >>> set_cover_lp({1, 2}, [{1, 2}, {1}], [3, 1])
        (3.0, [1.0, 0.0])
        >>> set_cover_lp(set(), [{1}, {2}])
        (0.0, [0.0, 0.0])
    """
    elements, _sets, cost_list, containing = _validate(universe, subsets, costs)
    return _solve_relaxation(elements, containing, cost_list)


def set_cover_exact(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
    costs: Optional[Sequence[float]] = None,
) -> Tuple[List[int], float]:
    """Return a minimum-cost set cover by solving the integer program.

    Solves :func:`set_cover_ip` with :func:`src.utils.lp_solver.solve_milp`
    (HiGHS branch and bound). Exponential in the worst case, as expected for
    an NP-hard problem; used as the OPT baseline on small instances.

    Args:
        universe: The elements to cover.
        subsets: The candidate sets; elements outside the universe are
            ignored.
        costs: One non-negative cost per subset, or ``None`` for unit costs.

    Returns:
        ``(chosen, cost)``: the indices of an optimal cover in increasing
        order and its total cost (summed from ``costs``, not read from the
        solver). ``([], 0.0)`` for an empty universe.

    Raises:
        TypeError: If the inputs are not iterables of the right shape or a
            cost is not a real number.
        ValueError: If an element is in no subset, ``costs`` has the wrong
            length, or a cost is negative or not finite.
        RuntimeError: If the MILP solver fails to report an optimum.

    Time Complexity:
        Exponential in m in the worst case (branch and bound over 2^m
        assignments, each node an LP solve); fast for the small Week 9
        instances.

    Space Complexity:
        O(n m) plus the branch-and-bound tree.

    Examples:
        >>> set_cover_exact({1, 2, 3}, [{1, 2}, {2, 3}, {1, 3}])
        ([0, 1], 2.0)
        >>> set_cover_exact({1, 2, 3}, [{1, 2, 3}, {1}, {2, 3}], [5, 1, 2])
        ([1, 2], 3.0)
        >>> set_cover_exact(set(), [{1}])
        ([], 0.0)
    """
    elements, _sets, cost_list, containing = _validate(universe, subsets, costs)
    if not elements:
        return [], 0.0
    ip = _ip_data(elements, containing, cost_list)
    _value, x = solve_milp(
        ip["c"], A_ub=ip["A_ub"], b_ub=ip["b_ub"], bounds=ip["bounds"],
        integrality=ip["integrality"],
    )
    chosen = [index for index, value in enumerate(x) if value > 0.5]
    return chosen, _total(chosen, cost_list)


def set_cover_greedy(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
    costs: Optional[Sequence[float]] = None,
) -> Tuple[List[int], float]:
    """Return Chvatal's greedy set cover (an H(max |S|)-approximation).

    Repeatedly pick the subset with the lowest cost per newly covered
    element, c_S / |S minus covered|, ignoring subsets that cover nothing
    new; ties go to the lowest index. Ratios are compared by cross
    multiplication (c_i * new_j < c_j * new_i), so equal ratios tie exactly.
    Charging each pick's cost to the elements it newly covers shows the
    total is at most H(d) * OPT with d the largest subset size, and at most
    H(n) * OPT (see the module docstring for the proof).

    Args:
        universe: The elements to cover.
        subsets: The candidate sets; elements outside the universe are
            ignored.
        costs: One non-negative cost per subset, or ``None`` for unit costs.

    Returns:
        ``(chosen, cost)``: the indices in the order greedy picked them and
        the total cost. ``([], 0.0)`` for an empty universe.

    Raises:
        TypeError: If the inputs are not iterables of the right shape or a
            cost is not a real number.
        ValueError: If an element is in no subset, ``costs`` has the wrong
            length, or a cost is negative or not finite.

    Time Complexity:
        O(min(n, m) * N) for total subset size N: at most min(n, m) picks,
        each scanning every subset once.

    Space Complexity:
        O(n + N).

    Examples:
        The greedy trap: blocks of sizes 8, 4, 2 look cheaper per element
        than the optimal pair (odds, evens).

        >>> universe = set(range(1, 15))
        >>> trap = [{1, 3, 5, 7, 9, 11, 13}, {2, 4, 6, 8, 10, 12, 14},
        ...         {1, 2}, {3, 4, 5, 6}, {7, 8, 9, 10, 11, 12, 13, 14}]
        >>> set_cover_greedy(universe, trap)
        ([4, 3, 2], 3.0)

        Weighted: {2, 3} at cost 2 is 1 per element, cheaper than {1} at 1.5
        and {1, 2, 3} at 5/3; then {1} (1.5) beats {1, 2, 3} (5 for one new
        element):

        >>> set_cover_greedy({1, 2, 3}, [{1, 2, 3}, {1}, {2, 3}], [5, 1.5, 2])
        ([2, 1], 3.5)
        >>> set_cover_greedy(set(), [{1}])
        ([], 0.0)
    """
    elements, sets, cost_list, _containing = _validate(universe, subsets, costs)
    uncovered = set(elements)
    chosen: List[int] = []
    while uncovered:
        best = -1
        best_new = 0
        for index, subset in enumerate(sets):
            new = len(subset & uncovered)
            if new == 0:
                continue
            if best < 0 or cost_list[index] * best_new < cost_list[best] * new:
                best, best_new = index, new
        chosen.append(best)
        uncovered -= sets[best]
    return chosen, _total(chosen, cost_list)


def frequency(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
) -> int:
    """Return f, the largest number of subsets containing any one element.

    f is the approximation factor of deterministic rounding: every element
    has at most f sets whose x values sum to at least 1, so one of them has
    x_S >= 1/f. For Vertex Cover (elements are edges, sets are vertices)
    f = 2.

    Args:
        universe: The elements.
        subsets: The candidate sets; elements outside the universe are
            ignored.

    Returns:
        max over elements e of |{S : e in S}|, or 0 for an empty universe.

    Raises:
        TypeError: If the inputs are not iterables of the right shape.
        ValueError: If an element is in no subset.

    Time Complexity:
        O(n log n + N) for n elements and total subset size N.

    Space Complexity:
        O(n + N).

    Examples:
        >>> frequency({1, 2, 3}, [{1, 2}, {2, 3}, {1, 3}])
        2
        >>> frequency({1, 2}, [{1, 2}, {1}, {1}])
        3
        >>> frequency(set(), [{1}])
        0
    """
    elements, _sets, _costs, containing = _validate(universe, subsets, None)
    return max((len(containing[e]) for e in elements), default=0)


def set_cover_deterministic_rounding(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
    costs: Optional[Sequence[float]] = None,
    *,
    lp: Optional[Tuple[float, Sequence[float]]] = None,
) -> Tuple[List[int], float, float]:
    """Round the LP by picking every subset with x_S >= 1/f (f-approximation).

    With f the frequency (:func:`frequency`), every element's constraint
    sums at most f values to at least 1, so one of its sets has
    x_S >= 1/f and is picked: the result is a cover. Each picked set has
    f x_S >= 1, so the cost is at most f * LP* <= f * OPT. The threshold is
    1/f - 1e-9 to absorb solver round-off; an element still uncovered after
    that (round-off only) gets its containing set with the largest x_S,
    ties to the lowest index.

    Args:
        universe: The elements to cover.
        subsets: The candidate sets; elements outside the universe are
            ignored.
        costs: One non-negative cost per subset, or ``None`` for unit costs.
        lp: Optional precomputed ``(lp_value, x)`` from :func:`set_cover_lp`
            on the same instance, so the LP is not solved again. When
            ``None`` the LP is solved here.

    Returns:
        ``(chosen, cost, lp_value)``: the chosen indices in increasing
        order, their total cost and the LP value used. ``([], 0.0, 0.0)``
        for an empty universe.

    Raises:
        TypeError: If the inputs are not iterables of the right shape or a
            cost is not a real number.
        ValueError: If an element is in no subset, ``costs`` has the wrong
            length, a cost is negative or not finite, or ``lp`` is not a
            pair with one x value per subset.
        RuntimeError: If the LP solver fails to report an optimum.

    Time Complexity:
        One LP solve (skipped when ``lp`` is given) plus O(n log n + N) for
        the threshold and the coverage check.

    Space Complexity:
        O(n m) for the LP, O(n + N) otherwise.

    Examples:
        The tight example for f = 2: all three sets have x_S = 1/2, so all
        three are picked at cost 3 = 2 * LP* (OPT is 2).

        >>> set_cover_deterministic_rounding({1, 2, 3}, [{1, 2}, {2, 3}, {1, 3}])
        ([0, 1, 2], 3.0, 1.5)

        A precomputed LP solution is used as given:

        >>> set_cover_deterministic_rounding({1, 2}, [{1, 2}, {1}, {2}],
        ...                                  lp=(2.0, [0.0, 1.0, 1.0]))
        ([1, 2], 2.0, 2.0)
        >>> set_cover_deterministic_rounding(set(), [{1}])
        ([], 0.0, 0.0)
    """
    elements, sets, cost_list, containing = _validate(universe, subsets, costs)
    if not elements:
        return [], 0.0, 0.0
    if lp is None:
        lp_value, x = _solve_relaxation(elements, containing, cost_list)
    else:
        lp_value, x = _check_lp(lp, len(sets))
    f = max(len(containing[e]) for e in elements)
    threshold = 1.0 / f - _ROUND_TOL
    picked = {index for index, value in enumerate(x) if value >= threshold}
    for e in elements:
        if not any(index in picked for index in containing[e]):
            picked.add(max(containing[e], key=lambda index: (x[index], -index)))
    chosen = sorted(picked)
    return chosen, _total(chosen, cost_list), lp_value


def _default_rounds(n: int) -> int:
    """Return the default number of randomized rounding rounds, ceil(2 ln n) + 1.

    Args:
        n: Number of elements, at least 1.

    Returns:
        ``ceil(2 ln n) + 1``; 1 for n = 1. This is at least 2 ln n, so an
        element survives every round with probability at most n^-2.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _default_rounds(1), _default_rounds(14), _default_rounds(200)
        (1, 7, 12)
    """
    return math.ceil(2.0 * math.log(n)) + 1


def set_cover_randomized_rounding(
    universe: Iterable[Hashable],
    subsets: Sequence[Iterable[Hashable]],
    costs: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    rounds: Optional[int] = None,
    *,
    lp: Optional[Tuple[float, Sequence[float]]] = None,
    return_fixups: bool = False,
) -> Tuple[Any, ...]:
    """Round the LP randomly: O(log n)-approximation in expectation.

    In each of ``rounds`` rounds (default ceil(2 ln n) + 1, n = |universe|)
    every subset S, in index order, is picked independently with
    probability x_S, using ONE generator ``make_rng(seed)`` for the whole
    call (one ``rng.random()`` draw per subset per round, S picked when the
    draw is below x_S). Every element still uncovered after the rounds is a
    *fix-up*: the cheapest subset containing it (ties to the lowest index)
    is added. One round covers a fixed element with probability at least
    1 - 1/e and costs LP* in expectation, so the expected total is at most
    (2 ln n + 2) LP* + OPT / n (proof in the module docstring), and the
    result is always a cover.

    Args:
        universe: The elements to cover.
        subsets: The candidate sets; elements outside the universe are
            ignored.
        costs: One non-negative cost per subset, or ``None`` for unit costs.
        seed: ``int`` for a reproducible run, ``None`` for an unpredictable
            one (see :func:`src.utils.random_utils.make_rng`).
        rounds: Number of rounds, a non-negative ``int``; ``None`` for the
            default ceil(2 ln n) + 1. With 0 rounds every element is a
            fix-up.
        lp: Optional precomputed ``(lp_value, x)`` from :func:`set_cover_lp`
            on the same instance, so repeated runs do not re-solve the LP.
            When ``None`` the LP is solved here.
        return_fixups: When ``True`` also return the number of fix-ups.

    Returns:
        ``(chosen, cost, lp_value)``, or ``(chosen, cost, lp_value, fixups)``
        when ``return_fixups`` is ``True``: the chosen indices in increasing
        order, their total cost, the LP value used and the number of
        elements left uncovered by the rounds. ``([], 0.0, 0.0)`` (plus a
        0) for an empty universe.

    Raises:
        TypeError: If the inputs are not iterables of the right shape, a
            cost is not a real number, ``seed`` is neither ``None`` nor an
            ``int``, or ``rounds`` is not an ``int``.
        ValueError: If an element is in no subset, ``costs`` has the wrong
            length, a cost is negative or not finite, ``rounds`` is
            negative, or ``lp`` is not a pair with one x value per subset.
        RuntimeError: If the LP solver fails to report an optimum.

    Time Complexity:
        One LP solve (skipped when ``lp`` is given) plus O(k m + n log n + N)
        for k rounds, m subsets, n elements and total subset size N.

    Space Complexity:
        O(n m) for the LP, O(n + N) otherwise.

    Examples:
        On the greedy trap the LP is integral (x = 1 on the two halves, 0
        elsewhere), so every run returns the optimum:

        >>> universe = set(range(1, 15))
        >>> trap = [{1, 3, 5, 7, 9, 11, 13}, {2, 4, 6, 8, 10, 12, 14},
        ...         {1, 2}, {3, 4, 5, 6}, {7, 8, 9, 10, 11, 12, 13, 14}]
        >>> set_cover_randomized_rounding(universe, trap, seed=7)
        ([0, 1], 2.0, 2.0)

        A fractional LP (every x_S = 1/2): with seed 0 one round leaves one
        element uncovered, and a fix-up repairs it.

        >>> tri = [{1, 2}, {2, 3}, {1, 3}]
        >>> set_cover_randomized_rounding({1, 2, 3}, tri, seed=0, rounds=1,
        ...                               return_fixups=True)
        ([0, 2], 2.0, 1.5, 1)
        >>> run = set_cover_randomized_rounding({1, 2, 3}, tri, seed=5)
        >>> run == set_cover_randomized_rounding({1, 2, 3}, tri, seed=5)
        True

        With 0 rounds every element is fixed by its cheapest set:

        >>> set_cover_randomized_rounding({1, 2}, [{1, 2}, {1}, {2}], [3, 1, 1],
        ...                               seed=1, rounds=0, return_fixups=True)
        ([1, 2], 2.0, 2.0, 2)
    """
    elements, sets, cost_list, containing = _validate(universe, subsets, costs)
    if rounds is not None:
        if not isinstance(rounds, int) or isinstance(rounds, bool):
            raise TypeError(f"rounds must be an int, got {type(rounds).__name__}")
        if rounds < 0:
            raise ValueError(f"rounds must be non-negative, got {rounds}")
    rng = make_rng(seed)
    if not elements:
        return ([], 0.0, 0.0, 0) if return_fixups else ([], 0.0, 0.0)
    if lp is None:
        lp_value, x = _solve_relaxation(elements, containing, cost_list)
    else:
        lp_value, x = _check_lp(lp, len(sets))
    if rounds is None:
        rounds = _default_rounds(len(elements))
    picked: Set[int] = set()
    for _ in range(rounds):
        for index in range(len(sets)):
            if rng.random() < x[index]:
                picked.add(index)
    covered: Set[Hashable] = set()
    for index in picked:
        covered |= sets[index]
    missing = [e for e in elements if e not in covered]
    for e in missing:
        picked.add(min(containing[e], key=lambda index: (cost_list[index], index)))
    chosen = sorted(picked)
    cost = _total(chosen, cost_list)
    if return_fixups:
        return chosen, cost, lp_value, len(missing)
    return chosen, cost, lp_value


def random_instance(
    n_elements: int,
    n_sets: int,
    density: float,
    seed: Optional[int],
) -> Tuple[Set[int], List[Set[int]], List[float]]:
    """Generate a seeded random set cover instance that always has a cover.

    Universe ``{1, .., n_elements}``. With one generator ``make_rng(seed)``:
    for each set in index order and each element in increasing order, the
    element joins the set with probability ``density``; then every element
    in no set is added to a uniformly random set (repair, so a cover
    exists); then every still-empty set receives one uniformly random
    element; finally each set gets a cost drawn uniformly from [1, 10].

    Args:
        n_elements: Number of elements, an ``int`` >= 1.
        n_sets: Number of subsets, an ``int`` >= 1.
        density: Membership probability, a real number in [0, 1].
        seed: ``int`` for a reproducible instance, or ``None``.

    Returns:
        ``(universe, subsets, costs)``: the universe as a ``set``, the
        subsets as a list of non-empty sets whose union is the universe,
        and one float cost in [1, 10] per subset.

    Raises:
        TypeError: If ``n_elements`` or ``n_sets`` is not an ``int``,
            ``density`` is not a real number, or ``seed`` is neither
            ``None`` nor an ``int``.
        ValueError: If ``n_elements`` or ``n_sets`` is below 1 or
            ``density`` is outside [0, 1].

    Time Complexity:
        O(n m) for n elements and m sets (one draw per pair).

    Space Complexity:
        O(n m) in the worst case (density 1).

    Examples:
        >>> universe, subsets, costs = random_instance(10, 4, 0.3, seed=42)
        >>> universe == set(range(1, 11)), len(subsets), len(costs)
        (True, 4, 4)
        >>> set().union(*subsets) == universe, all(subsets)
        (True, True)
        >>> all(1.0 <= c <= 10.0 for c in costs)
        True
        >>> random_instance(10, 4, 0.3, seed=42) == random_instance(10, 4, 0.3, seed=42)
        True
    """
    for name, value in (("n_elements", n_elements), ("n_sets", n_sets)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if not isinstance(density, numbers.Real) or isinstance(density, bool):
        raise TypeError(f"density must be a real number, got {type(density).__name__}")
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    rng = make_rng(seed)
    universe = set(range(1, n_elements + 1))
    subsets: List[Set[int]] = [set() for _ in range(n_sets)]
    for subset in subsets:
        for e in range(1, n_elements + 1):
            if rng.random() < density:
                subset.add(e)
    for e in range(1, n_elements + 1):
        if not any(e in subset for subset in subsets):
            subsets[rng.randrange(n_sets)].add(e)
    for subset in subsets:
        if not subset:
            subset.add(rng.randint(1, n_elements))
    costs = [rng.uniform(1.0, 10.0) for _ in range(n_sets)]
    return universe, subsets, costs
