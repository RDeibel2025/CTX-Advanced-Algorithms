"""Uncapacitated Facility Location: an add-greedy heuristic and an exact MILP.

The problem
-----------
An instance has F candidate facilities and C clients. Opening facility f
costs ``open_costs[f]`` and serving client c from facility f costs
``dist[c][f]``. A solution is a nonempty set S of open facilities; every
client is then served by its nearest open facility, so

    cost(S) = sum_{f in S} open_costs[f] + sum_c min_{f in S} dist[c][f].

:func:`total_cost` evaluates this formula. Finding the cheapest S is
NP-hard: Set Cover reduces to it (a facility per set, a client per element,
distance 0 to the sets containing the element and a huge distance
otherwise), so unless P = NP no polynomial-time algorithm is exact, and
when the distances are arbitrary (non-metric) no polynomial-time algorithm
can beat a factor of about ln n (the Set Cover hardness carries over).

The greedy heuristic
--------------------
:func:`facility_location_greedy` is the simple "add" greedy:

1. Open the single facility with the lowest total cost
   ``open_costs[f] + sum_c dist[c][f]`` (ties: lowest index).
2. Repeatedly open the closed facility whose opening lowers the total cost
   the most (ties: lowest index). Opening f costs ``open_costs[f]`` and
   saves ``sum_c max(0, nearest[c] - dist[c][f])``, where ``nearest[c]`` is
   client c's current connection cost.
3. Stop as soon as no closed facility lowers the cost strictly.

What can be proved about it, and what cannot:

* **It returns a feasible solution.** The open set is never empty (step 1
  opens one facility), so every client has a nearest open facility.
* **It terminates after at most F - 1 additions.** Each addition opens a
  facility that was closed and nothing is ever closed, so there are at most
  F - 1 of them; each costs O(F C) to choose, O(F^2 C) in total.
* **Its cost never exceeds the best single facility,** because step 1
  opens that facility and every later step lowers the cost strictly. When
  it stops, the solution is a local optimum for the "open one more"
  move: no single addition helps.
* **It has NO proven approximation ratio.** Nothing bounds the final cost
  against OPT, because the greedy can never close a facility it opened
  early. The doctest below is the standard trap: a hub between two client
  clusters is the best single facility, so it is opened first; opening
  the two cluster facilities then pays off, but the hub stays open even
  though it serves nobody, giving 11 against an optimum of 8. This module
  treats the greedy as a heuristic, and the Week 9 benchmark labels it
  ``kind = heuristic`` and *measures* its ratio against the MILP optimum
  instead of claiming one.

For metric instances (distances obeying the triangle inequality, as in
:func:`random_instance`) constant-factor algorithms do exist in the
literature: Shmoys, Tardos and Aardal's LP rounding (3.16, 1997), Jain and
Vazirani's primal-dual algorithm (3, 2001), the Jain-Mahdian-Saberi greedy,
which picks by cost per newly served client and also re-routes clients
(1.61, 2003), and Li's 1.488 (2013). Guha and Khuller showed that no
polynomial-time algorithm beats 1.463 unless NP has slightly
super-polynomial algorithms. The add-greedy here is none of those: it is a
heuristic, kept because it is the natural first idea and is easy to state.

The exact solver (MILP)
-----------------------
:func:`facility_location_exact` solves the standard integer program with
:func:`src.utils.lp_solver.solve_milp` (HiGHS branch and bound):

    minimize    sum_f open_costs[f] y_f + sum_{c,f} dist[c][f] x_cf
    subject to  sum_f x_cf = 1          for every client c
                x_cf <= y_f             for every client c and facility f
                y_f, x_cf in {0, 1}.

**Why it is exact.** (a) Every open set S gives a feasible point: y_f = 1
exactly for f in S, and x_cf = 1 exactly for c's nearest facility in S.
Its objective is cost(S). (b) Every feasible point (y, x) opens the set
S = {f : y_f = 1}, which is nonempty because each client is assigned to a
facility with y_f >= x_cf = 1; its objective is at least cost(S), because
each client pays dist[c][f] for an open f, which is at least its nearest
open distance. So the minimum of the integer program equals
min_S cost(S). The disaggregated constraints ``x_cf <= y_f`` (rather than
one ``sum_c x_cf <= C y_f`` per facility) give a much tighter LP relaxation,
so branch and bound has little to do on the benchmark sizes.

The optimum is not always unique. To make the returned set deterministic,
the solver recomputes the cost of the MILP's open set with
:func:`total_cost` and then, in one pass in increasing index order, closes
every open facility whose removal does not raise the cost. Closing a
facility never lowers an optimal cost, so the result is still optimal; it
is optimal *and* minimal by inclusion. On the Week 9 anchor both {0, 1} and
{0, 1, 2} cost 19, and the solver returns {0, 1}.

Random instances
----------------
:func:`random_instance` places clients and facilities as separate uniform
points in the square ``[0, size)^2`` and uses Euclidean distances, so the
instances are metric; opening costs are uniform in [20, 60].

======================== =================================== ==========
Function                 Time                                Space
======================== =================================== ==========
total_cost               O(F C) (validation dominates)       O(F C)
facility_location_greedy O(F^2 C)                            O(F C)
facility_location_exact  exponential worst case (MILP)       O(F^2 C^2)
random_instance          O(F C)                              O(F C)
======================== =================================== ==========

Examples:
    The Week 9 anchor: three facilities, five clients, opening cost 4 each.

    >>> costs = [4, 4, 4]
    >>> dist = [[1, 5, 6], [2, 4, 6], [6, 1, 5], [6, 2, 4], [5, 6, 1]]
    >>> [total_cost(costs, dist, {f}) for f in range(3)]
    [24.0, 22.0, 26.0]
    >>> facility_location_greedy(costs, dist)
    ({0, 1}, 19.0)
    >>> facility_location_exact(costs, dist)
    ({0, 1}, 19.0)

    The hub trap: the greedy opens the hub first and can never close it.

    >>> hub_costs = [3, 4, 4]
    >>> hub_dist = [[3, 0, 6], [3, 0, 6], [3, 6, 0], [3, 6, 0]]
    >>> facility_location_greedy(hub_costs, hub_dist)
    ({0, 1, 2}, 11.0)
    >>> facility_location_exact(hub_costs, hub_dist)
    ({1, 2}, 8.0)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.utils.lp_solver import solve_milp
from src.utils.random_utils import make_rng

__all__ = [
    "facility_location_exact",
    "facility_location_greedy",
    "random_instance",
    "total_cost",
]

#: Relative tolerance used when deciding whether closing a facility in the
#: exact solution leaves the cost unchanged (a tie between optimal sets).
_TIE_TOL: float = 1e-9


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int`` (not a ``bool``), else raise.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``; ``bool`` is rejected.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(3, "f")
        3
        >>> _require_int(True, "f")
        Traceback (most recent call last):
            ...
        TypeError: f must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_nonnegative(value: Any, name: str) -> float:
    """Return ``value`` as a ``float`` if it is a finite real number >= 0.

    Args:
        value: An opening cost or a distance.
        name: How to name it in the error message.

    Returns:
        ``float(value)``.

    Raises:
        TypeError: If ``value`` is not a real number (``bool`` rejected).
        ValueError: If ``value`` is negative, infinite or NaN.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_nonnegative(4, "open_costs[0]")
        4.0
        >>> _require_nonnegative(-1, "dist[0][0]")
        Traceback (most recent call last):
            ...
        ValueError: dist[0][0] must be a finite number >= 0, got -1
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be a finite number >= 0, got {value}")
    return float(value)


def _validate_instance(
    open_costs: Any, dist: Any
) -> Tuple[List[float], List[List[float]]]:
    """Check an instance and return it as plain lists of floats.

    Args:
        open_costs: One opening cost per facility.
        dist: ``dist[client][facility]``, one row per client, every row of
            length ``len(open_costs)``.

    Returns:
        ``(costs, rows)``: ``costs`` a list of F floats and ``rows`` a list
        of C lists of F floats.

    Raises:
        TypeError: If ``open_costs``, ``dist`` or a row is a string or not a
            sequence, or an entry is not a real number.
        ValueError: If there are no facilities or no clients, a row has the
            wrong length, or an entry is negative or not finite.

    Time Complexity:
        O(F C).

    Space Complexity:
        O(F C) for the copies.

    Examples:
        >>> _validate_instance([4, 4], [[1, 2]])
        ([4.0, 4.0], [[1.0, 2.0]])
        >>> _validate_instance([4, 4], [[1]])
        Traceback (most recent call last):
            ...
        ValueError: dist[0] has 1 entries, expected 2 (one per facility)
    """
    for obj, name in ((open_costs, "open_costs"), (dist, "dist")):
        if isinstance(obj, (str, bytes)) or not isinstance(obj, Sequence):
            raise TypeError(f"{name} must be a sequence, got {type(obj).__name__}")
    if len(open_costs) == 0:
        raise ValueError("open_costs must list at least one facility")
    if len(dist) == 0:
        raise ValueError("dist must list at least one client")
    costs = [_require_nonnegative(v, f"open_costs[{f}]") for f, v in enumerate(open_costs)]
    n_f = len(costs)
    rows: List[List[float]] = []
    for c, row in enumerate(dist):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"dist[{c}] must be a sequence, got {type(row).__name__}")
        if len(row) != n_f:
            raise ValueError(
                f"dist[{c}] has {len(row)} entries, expected {n_f} (one per facility)"
            )
        rows.append([_require_nonnegative(v, f"dist[{c}][{f}]") for f, v in enumerate(row)])
    return costs, rows


def _validate_open_set(open_set: Any, n_facilities: int) -> List[int]:
    """Check an open set and return its facilities sorted.

    Args:
        open_set: A non-string iterable of facility indices.
        n_facilities: The number of facilities F.

    Returns:
        The distinct indices in increasing order.

    Raises:
        TypeError: If ``open_set`` is not a non-string iterable or holds a
            non-``int`` index.
        ValueError: If ``open_set`` is empty or an index is outside
            ``range(n_facilities)``.

    Time Complexity:
        O(|S| log |S|).

    Space Complexity:
        O(|S|).

    Examples:
        >>> _validate_open_set({2, 0}, 3)
        [0, 2]
        >>> _validate_open_set([3], 3)
        Traceback (most recent call last):
            ...
        ValueError: facility index 3 is outside range(3)
    """
    if isinstance(open_set, (str, bytes)) or not isinstance(open_set, Iterable):
        raise TypeError(
            f"open_set must be an iterable of facility indices, got {type(open_set).__name__}"
        )
    chosen = set()
    for f in open_set:
        _require_int(f, "facility index")
        if not 0 <= f < n_facilities:
            raise ValueError(f"facility index {f} is outside range({n_facilities})")
        chosen.add(f)
    if not chosen:
        raise ValueError("open_set must contain at least one facility")
    return sorted(chosen)


def _cost(costs: List[float], rows: List[List[float]], open_list: List[int]) -> float:
    """Total cost of a validated, sorted, nonempty open set.

    Args:
        costs: Validated opening costs.
        rows: Validated distance rows.
        open_list: Distinct facility indices in increasing order.

    Returns:
        The opening costs of ``open_list`` plus every client's distance to
        its nearest facility in ``open_list``, as a ``float``.

    Time Complexity:
        O(|S| C).

    Space Complexity:
        O(1) beyond the inputs.

    Examples:
        >>> _cost([4.0, 4.0], [[1.0, 5.0], [6.0, 2.0]], [0, 1])
        11.0
    """
    opening = sum(costs[f] for f in open_list)
    service = sum(min(row[f] for f in open_list) for row in rows)
    return float(opening + service)


def total_cost(
    open_costs: Sequence[float],
    dist: Sequence[Sequence[float]],
    open_set: Iterable[int],
) -> float:
    """Return the cost of opening ``open_set``: opening plus service costs.

    Every client is served by its nearest open facility, so the cost is
    ``sum_{f in S} open_costs[f] + sum_c min_{f in S} dist[c][f]``. This is
    the objective both :func:`facility_location_greedy` and
    :func:`facility_location_exact` minimize.

    Args:
        open_costs: One non-negative opening cost per facility (F >= 1).
        dist: ``dist[client][facility]``, C >= 1 rows of F non-negative
            distances.
        open_set: A nonempty iterable of facility indices in ``range(F)``;
            duplicates are ignored.

    Returns:
        The total cost as a ``float``.

    Raises:
        TypeError: If an argument has the wrong type (see
            :func:`_validate_instance`), or an index is not an ``int``.
        ValueError: If the instance is malformed, ``open_set`` is empty, or
            an index is out of range.

    Time Complexity:
        O(F C) to validate the instance, plus O(|S| C) for the nearest
        open facility of every client.

    Space Complexity:
        O(F C) for the validated copy of the instance.

    Examples:
        >>> costs = [4, 4, 4]
        >>> dist = [[1, 5, 6], [2, 4, 6], [6, 1, 5], [6, 2, 4], [5, 6, 1]]
        >>> total_cost(costs, dist, {0})
        24.0
        >>> total_cost(costs, dist, {0, 1})
        19.0
        >>> total_cost(costs, dist, [0, 1, 2])
        19.0
        >>> total_cost(costs, dist, set())
        Traceback (most recent call last):
            ...
        ValueError: open_set must contain at least one facility
    """
    costs, rows = _validate_instance(open_costs, dist)
    return _cost(costs, rows, _validate_open_set(open_set, len(costs)))


def facility_location_greedy(
    open_costs: Sequence[float],
    dist: Sequence[Sequence[float]],
) -> Tuple[Set[int], float]:
    """Add-greedy heuristic for Uncapacitated Facility Location.

    Open the single facility with the lowest total cost, then repeatedly
    open the facility whose opening lowers the total cost the most, and
    stop when no facility lowers it strictly. Ties go to the lowest index
    at every step, so the result is deterministic.

    The heuristic always returns a feasible solution no more expensive than
    the best single facility, and stops at a local optimum for adding one
    facility, but it has NO proven approximation ratio: it never closes a
    facility, so an early choice (such as a hub between two clusters) can
    stay open for nothing. See the module docstring.

    Args:
        open_costs: One non-negative opening cost per facility (F >= 1).
        dist: ``dist[client][facility]``, C >= 1 rows of F non-negative
            distances.

    Returns:
        ``(open_set, cost)``: the set of opened facility indices and its
        :func:`total_cost`.

    Raises:
        TypeError: If ``open_costs``, ``dist`` or a row is not a sequence,
            or an entry is not a real number.
        ValueError: If there are no facilities or clients, a row has the
            wrong length, or an entry is negative or not finite.

    Time Complexity:
        O(F^2 C): at most F - 1 additions, and choosing each one evaluates
        the saving of every closed facility over every client in O(F C).

    Space Complexity:
        O(F C) for the validated instance, plus O(C) for every client's
        current connection cost.

    Examples:
        >>> costs = [4, 4, 4]
        >>> dist = [[1, 5, 6], [2, 4, 6], [6, 1, 5], [6, 2, 4], [5, 6, 1]]
        >>> facility_location_greedy(costs, dist)
        ({0, 1}, 19.0)
        >>> facility_location_greedy([3, 4, 4], [[3, 0, 6], [3, 0, 6], [3, 6, 0], [3, 6, 0]])
        ({0, 1, 2}, 11.0)
        >>> costs, dist = random_instance(30, 8, 42)
        >>> facility_location_greedy(costs, dist) == facility_location_greedy(costs, dist)
        True
    """
    costs, rows = _validate_instance(open_costs, dist)
    n_f = len(costs)

    singles = [costs[f] + sum(row[f] for row in rows) for f in range(n_f)]
    first = min(range(n_f), key=singles.__getitem__)  # min keeps the lowest index on ties
    open_set: Set[int] = {first}
    nearest = [row[first] for row in rows]

    while True:
        best_f: Optional[int] = None
        best_delta = 0.0  # only a strictly negative change is accepted
        for f in range(n_f):
            if f in open_set:
                continue
            saving = 0.0
            for c, row in enumerate(rows):
                d = row[f]
                if d < nearest[c]:
                    saving += nearest[c] - d
            delta = costs[f] - saving
            if delta < best_delta:  # strict, so the lowest index wins ties
                best_f, best_delta = f, delta
        if best_f is None:
            break
        open_set.add(best_f)
        for c, row in enumerate(rows):
            if row[best_f] < nearest[c]:
                nearest[c] = row[best_f]

    return open_set, _cost(costs, rows, sorted(open_set))


def facility_location_exact(
    open_costs: Sequence[float],
    dist: Sequence[Sequence[float]],
) -> Tuple[Set[int], float]:
    """Optimal Uncapacitated Facility Location by mixed integer programming.

    Solves, with :func:`src.utils.lp_solver.solve_milp`,

        minimize    sum_f open_costs[f] y_f + sum_{c,f} dist[c][f] x_cf
        subject to  sum_f x_cf = 1 for every client c,
                    x_cf <= y_f for every client c and facility f,
                    all variables binary,

    whose optimum equals min over nonempty S of :func:`total_cost` (proof in
    the module docstring). The open set is ``{f : y_f > 0.5}``; its cost is
    recomputed with :func:`total_cost`, and then, in one pass in increasing
    index order, every facility whose closing does not raise the cost is
    closed, so ties between optimal sets are broken deterministically.

    Variable layout: ``y_f`` is variable ``f`` and ``x_cf`` is variable
    ``F + c F + f``.

    Args:
        open_costs: One non-negative opening cost per facility (F >= 1).
        dist: ``dist[client][facility]``, C >= 1 rows of F non-negative
            distances.

    Returns:
        ``(open_set, cost)``: an optimal set of facility indices, minimal by
        inclusion, and its :func:`total_cost`.

    Raises:
        TypeError: If ``open_costs``, ``dist`` or a row is not a sequence,
            or an entry is not a real number.
        ValueError: If there are no facilities or clients, a row has the
            wrong length, or an entry is negative or not finite.
        RuntimeError: If HiGHS does not report an optimal solution (cannot
            happen for a valid instance, which is always feasible and
            bounded).

    Time Complexity:
        Exponential in the worst case (the problem is NP-hard and branch
        and bound may explore up to 2^F open sets), each node an LP over
        F + F C variables and F C + C constraints. The tight formulation
        usually solves at or near the root; C = 100, F = 20 takes well
        under a second. The tie-breaking pass adds O(F^2 C).

    Space Complexity:
        O(F^2 C^2) for the dense constraint matrix handed to the solver
        (F C + C rows by F + F C columns).

    Examples:
        >>> costs = [4, 4, 4]
        >>> dist = [[1, 5, 6], [2, 4, 6], [6, 1, 5], [6, 2, 4], [5, 6, 1]]
        >>> facility_location_exact(costs, dist)
        ({0, 1}, 19.0)
        >>> facility_location_exact([3, 4, 4], [[3, 0, 6], [3, 0, 6], [3, 6, 0], [3, 6, 0]])
        ({1, 2}, 8.0)
        >>> costs, dist = random_instance(12, 6, 7)
        >>> facility_location_exact(costs, dist)[1] <= facility_location_greedy(costs, dist)[1]
        True
    """
    costs, rows = _validate_instance(open_costs, dist)
    n_f, n_c = len(costs), len(rows)
    n_x = n_c * n_f
    n_vars = n_f + n_x

    objective = np.empty(n_vars, dtype=float)
    objective[:n_f] = costs
    objective[n_f:] = np.asarray(rows, dtype=float).ravel()

    # sum_f x_cf = 1 for every client c.
    a_eq = np.zeros((n_c, n_vars), dtype=float)
    for c in range(n_c):
        a_eq[c, n_f + c * n_f : n_f + (c + 1) * n_f] = 1.0
    b_eq = np.ones(n_c, dtype=float)

    # x_cf - y_f <= 0 for every client c and facility f (row r = c F + f).
    a_ub = np.zeros((n_x, n_vars), dtype=float)
    r = np.arange(n_x)
    a_ub[r, n_f + r] = 1.0
    a_ub[r, r % n_f] = -1.0
    b_ub = np.zeros(n_x, dtype=float)

    _, x = solve_milp(
        objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(0, 1)
    )
    open_list = [f for f in range(n_f) if x[f] > 0.5]
    cost = _cost(costs, rows, open_list)

    # Deterministic tie-break: close every facility whose removal does not
    # raise the cost. One pass suffices: closing a facility only makes the
    # remaining ones more useful, so a facility kept earlier stays needed.
    for f in list(open_list):
        if len(open_list) == 1:
            break
        trial = [g for g in open_list if g != f]
        trial_cost = _cost(costs, rows, trial)
        if trial_cost <= cost + _TIE_TOL * max(1.0, abs(cost)):
            open_list, cost = trial, trial_cost

    return set(open_list), cost


def random_instance(
    n_clients: int,
    n_facilities: int,
    seed: Optional[int],
    size: float = 100.0,
) -> Tuple[List[float], List[List[float]]]:
    """Generate a random metric facility-location instance.

    Clients and facilities are separate points, uniform in the square
    ``[0, size)^2``; ``dist[c][f]`` is the Euclidean distance between client
    c and facility f, so the instance is metric. Opening costs are uniform
    in [20, 60]. All draws come from one ``rng = make_rng(seed)`` in this
    order: the client points (x then y for each client), the facility
    points (x then y for each), then the opening costs
    (``rng.uniform(20.0, 60.0)`` for each facility). The same seed always
    gives the same instance.

    Args:
        n_clients: Number of clients C, at least 1.
        n_facilities: Number of facilities F, at least 1.
        seed: Passed to :func:`src.utils.random_utils.make_rng`. ``None``
            gives an unpredictable instance; experiments always pass an
            ``int``.
        size: Side length of the square, a positive finite real.

    Returns:
        ``(open_costs, dist)``: a list of F opening costs and the C x F
        list-of-lists distance matrix ``dist[client][facility]``.

    Raises:
        TypeError: If ``n_clients`` or ``n_facilities`` is not an ``int``,
            ``seed`` is neither ``None`` nor an ``int`` (``bool`` refused),
            or ``size`` is not a real number.
        ValueError: If ``n_clients < 1``, ``n_facilities < 1``, or ``size``
            is not positive and finite.

    Time Complexity:
        O(C F) - one distance per client-facility pair.

    Space Complexity:
        O(C F) - the distance matrix.

    Examples:
        >>> costs, dist = random_instance(5, 3, 42)
        >>> len(costs), len(dist), len(dist[0])
        (3, 5, 3)
        >>> all(20.0 <= v <= 60.0 for v in costs)
        True
        >>> round(dist[0][0], 6)
        63.858759
        >>> random_instance(5, 3, 42) == random_instance(5, 3, 42)
        True
        >>> random_instance(0, 3, 42)
        Traceback (most recent call last):
            ...
        ValueError: n_clients must be >= 1, got 0
    """
    _require_int(n_clients, "n_clients")
    _require_int(n_facilities, "n_facilities")
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if n_facilities < 1:
        raise ValueError(f"n_facilities must be >= 1, got {n_facilities}")
    if isinstance(size, bool) or not isinstance(size, numbers.Real):
        raise TypeError(f"size must be a real number, got {type(size).__name__}")
    if not (math.isfinite(size) and size > 0):
        raise ValueError(f"size must be positive and finite, got {size}")
    rng = make_rng(seed)
    side = float(size)

    clients = [(side * rng.random(), side * rng.random()) for _ in range(n_clients)]
    facilities = [(side * rng.random(), side * rng.random()) for _ in range(n_facilities)]
    open_costs = [rng.uniform(20.0, 60.0) for _ in range(n_facilities)]
    dist = [[math.dist(p, q) for q in facilities] for p in clients]
    return open_costs, dist
