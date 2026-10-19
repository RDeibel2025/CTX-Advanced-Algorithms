"""Approximate TSP: nearest neighbor, best-start nearest neighbor and 2-opt.

The traveling salesman problem asks for the cheapest closed tour through n
cities given a matrix ``dist`` in which ``dist[i][j]`` is the cost of going
from city ``i`` to city ``j``. Its decision version ("is there a tour of
cost at most B?") is NP-complete (Karp 1972), so no polynomial-time exact
algorithm is known. The exact solvers of Week 6
(:mod:`src.dp_advanced.bitmask_traveling_salesman`) take O(n!) time by brute
force and O(n^2 * 2^n) time by Held-Karp dynamic programming. This module
trades optimality for speed with three polynomial-time methods, and every
one of them returns a *certificate*: a concrete tour that anyone can check
in O(n) time with :func:`is_valid_tour` and :func:`tour_cost`. Checking a
tour is cheap; finding the best one is not.

The three methods
-----------------
* :func:`tsp_nearest_neighbor` - start somewhere, always walk to the
  closest city not yet visited, and return home at the end. Ties go to the
  lowest city index, so the result is deterministic. O(n^2).
* :func:`tsp_nearest_neighbor_best_start` - run nearest neighbor from every
  city and keep the cheapest tour, rotated so it starts and ends at city 0.
  O(n^3).
* :func:`tsp_two_opt` - local search on a given tour. A 2-opt move deletes
  two edges ``(a, b)`` and ``(c, e)`` and reconnects the tour as ``(a, c)``
  and ``(b, e)``, which reverses the path between them. On a Euclidean
  instance this removes every crossing, because two crossing edges are
  always longer than the two that uncross them.

What is (and is not) guaranteed
-------------------------------
None of the three has a constant-factor guarantee.

* **Nearest neighbor** can be a factor Theta(log n) worse than optimal even
  on metric instances (instances that obey the triangle inequality):
  Rosenkrantz, Stearns and Lewis (1977) prove an upper bound of
  ``(ceil(log2 n) + 1) / 2`` and build metric families whose ratio grows
  like ``log2 n / 3``. On non-metric instances the ratio is unbounded: the
  four-city matrix in the Examples below forces the greedy walk to close
  the tour with an edge of cost ``M``, so its ratio ``(M + 3) / 6`` grows
  without limit. That is not a weakness of this one heuristic. For general
  (non-metric) TSP, any polynomial-time algorithm with a constant ratio
  would decide the Hamiltonian cycle problem, so none exists unless
  P = NP (Sahni and Gonzalez 1976). On random uniform Euclidean instances
  nearest neighbor typically lands about 25 percent above the optimum
  (Johnson and McGeoch 1997), which is why it is still useful.
* **2-opt** is a local search: it stops at a tour that no single 2-opt move
  can shorten, a *local* optimum, which need not be the global one. The
  worst-case ratio of a 2-optimal tour on metric instances grows like
  sqrt(n) (Chandra, Karloff and Tovey 1999), and even in the Euclidean
  plane there are inputs on which it can take exponentially many improving
  moves (Englert, Roeglin and Voecking 2014), so no polynomial bound on its
  running time is known. In practice it is fast and closes
  most of nearest neighbor's gap, typically to about 5 percent.

The 2-opt implementation
------------------------
:func:`tsp_two_opt` is FIRST-IMPROVEMENT: within a pass it scans the
position pairs ``(i, j)`` in increasing order, applies an improving move as
soon as it finds one, and keeps scanning from the same ``(i, j + 1)`` on
the updated tour. Passes repeat until a whole pass applies no move, or
until ``max_passes`` passes have run. The first city ``tour[0]`` (and so
the closing ``tour[-1]``) never moves. Three safeguards make "never
increases the cost" literally true in floating point:

1. A move is applied only when it shortens the tour by more than
   ``1e-12 * max(1, cost)``, so rounding noise in the four-term change
   cannot be mistaken for an improvement.
2. After every pass the tour is re-priced from scratch. If the pass did not
   strictly lower that price, the pass is undone and the search stops.
   Strictly decreasing prices over a finite set of tours also prove that
   the search terminates.
3. On an asymmetric matrix, reversing a path changes the cost of every edge
   inside it, so the move's change includes the reversed path's cost. Two
   prefix-sum arrays over the current tour (forward and backward edge
   costs) give that in O(1) per candidate move.

Contract shared by the solvers
------------------------------
* ``dist`` is a square n x n matrix of non-negative real numbers, n >= 1,
  given as a list of lists (anything that iterates as rows works). ``bool``
  entries are refused; ``math.inf`` is allowed and marks a missing edge,
  although a heuristic may then return a tour of cost ``inf``. The diagonal
  is never read: the one-city tour ``[c, c]`` costs 0 whatever
  ``dist[c][c]`` holds.
* A tour is CLOSED: it lists the start city again at the end, so it has
  n + 1 entries. ``n == 1`` gives ``(0, [0, 0])`` and ``n == 2`` gives
  ``[s, other, s]``. (The Week 6 exact solvers write the one-city tour as
  ``[0]``; for n >= 2 the two formats agree.)
* The cost a solver returns is exactly ``tour_cost(dist, tour)``: both sum
  the edges in tour order from ``tour[0]``.

==================================== ============ =========================
Function                             Time         Space
==================================== ============ =========================
tsp_nearest_neighbor                 O(n^2)       O(n^2) - validated copy
tsp_nearest_neighbor_best_start      O(n^3)       O(n^2)
tsp_two_opt                          O(n^2)/pass  O(n^2)
tour_cost, is_valid_tour             O(n)         O(n)
euclidean_instance                   O(n^2)       O(n^2) - the matrix
==================================== ============ =========================

Examples:
    The four-city instance from Week 6, where the greedy walk happens to be
    optimal (the optimum, 80, comes from Held-Karp and brute force):

    >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25],
    ...         [15, 35, 0, 30], [20, 25, 30, 0]]
    >>> tsp_nearest_neighbor(dist)
    (80, [0, 1, 3, 2, 0])

    A non-metric instance with one expensive edge ``M``. The greedy walk
    0 -> 1 -> 2 -> 3 is forced to return along it, while the tour
    0 -> 1 -> 3 -> 2 -> 0 costs 6 for every ``M >= 3``:

    >>> M = 1000
    >>> trap = [[0, 1, 2, M], [1, 0, 1, 2], [2, 1, 0, 1], [M, 2, 1, 0]]
    >>> tsp_nearest_neighbor(trap)
    (1003, [0, 1, 2, 3, 0])
    >>> tsp_two_opt(trap, [0, 1, 2, 3, 0])
    (6, [0, 1, 3, 2, 0])

References:
    Chandra, B., Karloff, H., & Tovey, C. (1999). New results on the old
    k-opt algorithm for the traveling salesman problem. *SIAM Journal on
    Computing, 28*(6), 1998-2029.

    Englert, M., Roeglin, H., & Voecking, B. (2014). Worst case and
    probabilistic analysis of the 2-Opt algorithm for the TSP.
    *Algorithmica, 68*(1), 190-264.

    Johnson, D. S., & McGeoch, L. A. (1997). The traveling salesman problem:
    A case study in local optimization. In E. Aarts & J. K. Lenstra (Eds.),
    *Local Search in Combinatorial Optimization* (pp. 215-310). Wiley.

    Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M. (1977). An analysis
    of several heuristics for the traveling salesman problem. *SIAM Journal
    on Computing, 6*(3), 563-581.

    Sahni, S., & Gonzalez, T. (1976). P-complete approximation problems.
    *Journal of the ACM, 23*(3), 555-565.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, List, Optional, Sequence, Tuple

from src.utils.random_utils import make_rng

__all__ = [
    "euclidean_instance",
    "is_valid_tour",
    "tour_cost",
    "tsp_nearest_neighbor",
    "tsp_nearest_neighbor_best_start",
    "tsp_two_opt",
]

Matrix = List[List[float]]
Tour = List[int]

#: Relative tolerance for accepting a 2-opt move. A move must shorten the
#: tour by more than ``_EPSILON * max(1, cost)``; anything smaller is treated
#: as floating-point noise, so noise can never drive the search in a cycle.
_EPSILON: float = 1e-12

#: Entry types that :func:`_validate_matrix` accepts without conversion.
_PLAIN_REALS = frozenset((int, float))

#: ``0.0 <= x`` as a C-level callable; False for negatives and for NaN.
_NON_NEGATIVE = (0.0).__le__


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int`` (``bool`` refused), else raise.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int`` or is a ``bool``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(3, "start")
        3
        >>> _require_int(True, "start")
        Traceback (most recent call last):
            ...
        TypeError: start must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _validate_matrix(dist: Sequence[Sequence[float]]) -> Matrix:
    """Check ``dist`` and copy it into a fresh list of lists.

    Every solver starts here, so the error messages are identical across
    the module and the caller's matrix is never mutated. Integral entries
    stay ``int`` and other real entries become ``float``.

    Args:
        dist: The candidate distance matrix.

    Returns:
        The validated matrix as a new list of lists.

    Raises:
        TypeError: If ``dist`` does not iterate as rows, or an entry is not
            a real number (``bool`` included).
        ValueError: If ``dist`` is empty or not square, or an entry is
            negative or NaN.

    Time Complexity:
        O(n^2) - one check per entry.

    Space Complexity:
        O(n^2) - the copy.

    Examples:
        >>> _validate_matrix(((0, 2.5), (1, 0)))
        [[0, 2.5], [1, 0]]
        >>> _validate_matrix([])
        Traceback (most recent call last):
            ...
        ValueError: dist must have at least one city
        >>> _validate_matrix([[0, 1], [1]])
        Traceback (most recent call last):
            ...
        ValueError: dist must be square: row 1 has 1 entries, expected 2
        >>> _validate_matrix([[0, -1], [1, 0]])
        Traceback (most recent call last):
            ...
        ValueError: dist[0][1] must be >= 0, got -1
        >>> _validate_matrix([[0, True], [1, 0]])
        Traceback (most recent call last):
            ...
        TypeError: dist[0][1] must be a real number, got bool
        >>> _validate_matrix([1, 2])
        Traceback (most recent call last):
            ...
        TypeError: dist must be a square matrix of numbers, got a row of type int
    """
    try:
        raw_rows = list(dist)
    except TypeError:
        raise TypeError(
            f"dist must be a square matrix of numbers, got {type(dist).__name__}"
        ) from None

    rows: Matrix = []
    for row in raw_rows:
        try:
            rows.append(list(row))
        except TypeError:
            raise TypeError(
                "dist must be a square matrix of numbers, got a row of "
                f"type {type(row).__name__}"
            ) from None

    n = len(rows)
    if n == 0:
        raise ValueError("dist must have at least one city")

    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(
                f"dist must be square: row {i} has {len(row)} entries, "
                f"expected {n}"
            )
        # Fast path, run at C speed: a row of plain ints and floats that are
        # all >= 0 needs no per-entry work (NaN fails ``0.0 <= x``). Any other
        # row takes the entry-by-entry loop, which reports the first bad
        # entry or converts other real types. Without this, validation was
        # about 97 percent of nearest neighbor's run time at n = 500.
        if set(map(type, row)) <= _PLAIN_REALS and all(map(_NON_NEGATIVE, row)):
            continue
        for j, weight in enumerate(row):
            if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                raise TypeError(
                    f"dist[{i}][{j}] must be a real number, "
                    f"got {type(weight).__name__}"
                )
            # ``not >=`` rather than ``<`` so that NaN is refused as well.
            if not weight >= 0:
                raise ValueError(f"dist[{i}][{j}] must be >= 0, got {weight}")
            if not isinstance(weight, (int, float)):
                row[j] = (
                    int(weight)
                    if isinstance(weight, numbers.Integral)
                    else float(weight)
                )
    return rows


def _validate_start(start: Any, n: int) -> int:
    """Check that ``start`` is a city of an n-city instance.

    Args:
        start: The candidate start city.
        n: Number of cities.

    Returns:
        ``start`` unchanged.

    Raises:
        TypeError: If ``start`` is not an ``int`` (``bool`` refused).
        ValueError: If ``start`` is outside ``range(n)``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _validate_start(2, 3)
        2
        >>> _validate_start(3, 3)
        Traceback (most recent call last):
            ...
        ValueError: start must be in range(3), got 3
        >>> _validate_start(1.0, 3)
        Traceback (most recent call last):
            ...
        TypeError: start must be an int, got float
    """
    _require_int(start, "start")
    if not 0 <= start < n:
        raise ValueError(f"start must be in range({n}), got {start}")
    return start


def _closed_cost(rows: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    """Sum the edges of a closed tour in order, skipping self-steps.

    This is the single pricing routine of the module, so every solver's
    reported cost equals :func:`tour_cost` of its tour exactly, bit for
    bit: the same edges are added in the same order. A step from a city to
    itself, which only the one-city tour ``[c, c]`` contains, costs 0, so
    the diagonal is never read.

    Args:
        rows: A distance matrix.
        tour: A closed tour of valid city indices.

    Returns:
        The total cost. ``0`` (an ``int``) for the one-city tour.

    Raises:
        Nothing for valid input; callers check the tour first. A city
        outside the matrix surfaces as ``IndexError``.

    Time Complexity:
        O(n) - one lookup per edge.

    Space Complexity:
        O(1).

    Examples:
        >>> _closed_cost([[0, 2, 9], [1, 0, 6], [15, 7, 0]], [0, 1, 2, 0])
        23
        >>> _closed_cost([[5]], [0, 0])
        0
    """
    total: float = 0
    for a, b in zip(tour, tour[1:]):
        if a != b:
            total += rows[a][b]
    return total


def _nearest_neighbor_tour(rows: Matrix, start: int) -> Tour:
    """Build the nearest-neighbor tour from ``start`` on a validated matrix.

    ``unvisited`` holds the remaining cities in increasing index order and
    stays sorted when one is removed, so scanning it with a strict ``<``
    keeps the first minimum found, which is the lowest index among ties.
    When every remaining edge is ``inf`` that rule picks the lowest index
    too.

    Args:
        rows: A validated distance matrix.
        start: A validated start city.

    Returns:
        The closed tour ``[start, ..., start]``.

    Raises:
        Nothing; the matrix and the start are validated by the caller.

    Time Complexity:
        O(n^2) - n steps, each scanning and popping from a list of at most
        n - 1 cities.

    Space Complexity:
        O(n) - the tour and the unvisited list.

    Examples:
        From city 1 the closest cities are 0 and 2, both at distance 3; the
        tie goes to 0:

        >>> _nearest_neighbor_tour([[0, 3, 4], [3, 0, 3], [4, 3, 0]], 1)
        [1, 0, 2, 1]
        >>> _nearest_neighbor_tour([[0]], 0)
        [0, 0]
    """
    unvisited = [city for city in range(len(rows)) if city != start]
    tour = [start]
    current = start
    while unvisited:
        row = rows[current]
        best_pos = 0
        best_dist = row[unvisited[0]]
        for pos in range(1, len(unvisited)):
            candidate = row[unvisited[pos]]
            if candidate < best_dist:
                best_dist, best_pos = candidate, pos
        current = unvisited.pop(best_pos)
        tour.append(current)
    tour.append(start)
    return tour


def _rotate_to(tour: Sequence[int], city: int) -> Tour:
    """Return the same closed tour, driven the same way, starting at ``city``.

    Rotating a cycle keeps every edge and its direction, so the cost of
    the tour does not change (up to the order in which floats are added).

    Args:
        tour: A closed tour that visits ``city``.
        city: The city the rotated tour should start and end at.

    Returns:
        A new closed tour ``[city, ..., city]``.

    Raises:
        ValueError: If ``city`` is not on the tour (from ``list.index``).

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the new tour.

    Examples:
        >>> _rotate_to([3, 1, 0, 2, 3], 0)
        [0, 2, 3, 1, 0]
        >>> _rotate_to([0, 0], 0)
        [0, 0]
    """
    body = list(tour[:-1])
    k = body.index(city)
    return body[k:] + body[:k] + [city]


def tsp_nearest_neighbor(
    dist: Sequence[Sequence[float]],
    start: int = 0,
) -> Tuple[float, Tour]:
    """Build a tour by always moving to the closest unvisited city.

    Starting at ``start``, the walk repeatedly moves to the unvisited city
    ``j`` that minimises ``dist[current][j]``, breaking ties by the lowest
    index, and after the last city returns to ``start``. On an asymmetric
    matrix "closest" means the cheapest edge *out of* the current city.

    The greedy choice is made with no look-ahead, and that is exactly how
    it fails: the cheap edges are spent early and the walk is left to
    close the tour with whatever is left. There is no constant-factor
    guarantee. On metric instances the ratio to the optimum can grow as
    Theta(log n) (Rosenkrantz, Stearns and Lewis 1977); on non-metric
    instances it is unbounded (see the second example). Points on a line
    already show a gap: the greedy walk zig-zags outwards and costs 62
    where 48 is optimal.

    Args:
        dist: Square matrix of non-negative edge costs; ``dist[i][j]`` is
            the cost of going from ``i`` to ``j``. May be asymmetric.
            ``math.inf`` marks a missing edge. Not modified.
        start: The city the tour begins and ends at.

    Returns:
        A ``(cost, tour)`` pair. ``tour`` is closed, starts and ends at
        ``start`` and visits every city once in between; ``cost`` equals
        ``tour_cost(dist, tour)``. One city gives ``(0, [0, 0])``.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers, or
            ``start`` is not an ``int``.
        ValueError: If ``dist`` is empty, not square, or holds a negative
            or NaN entry, or if ``start`` is out of range.

    Time Complexity:
        O(n^2) - n steps, each scanning at most n - 1 unvisited cities
        (the O(n^2) validation of ``dist`` has the same order).

    Space Complexity:
        O(n^2) - the validated copy of ``dist``; O(n) beyond it.

    Examples:
        >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25],
        ...         [15, 35, 0, 30], [20, 25, 30, 0]]
        >>> tsp_nearest_neighbor(dist)
        (80, [0, 1, 3, 2, 0])

        Points on a line at x = 0, 1, -2, 4, -8, 16. Every step after the
        first is a tie between the next point on each side, the tie goes
        to the lower index, and the walk zig-zags outwards:

        >>> xs = [0, 1, -2, 4, -8, 16]
        >>> line = [[abs(a - b) for b in xs] for a in xs]
        >>> tsp_nearest_neighbor(line)
        (62, [0, 1, 2, 3, 4, 5, 0])

        Tiny instances, and a start other than 0:

        >>> tsp_nearest_neighbor([[0]])
        (0, [0, 0])
        >>> tsp_nearest_neighbor([[0, 3], [5, 0]], start=1)
        (8, [1, 0, 1])
        >>> tsp_nearest_neighbor(dist, start=4)
        Traceback (most recent call last):
            ...
        ValueError: start must be in range(4), got 4
    """
    rows = _validate_matrix(dist)
    _validate_start(start, len(rows))
    tour = _nearest_neighbor_tour(rows, start)
    return _closed_cost(rows, tour), tour


def tsp_nearest_neighbor_best_start(
    dist: Sequence[Sequence[float]],
) -> Tuple[float, Tour]:
    """Run nearest neighbor from every city and keep the cheapest tour.

    The greedy walk's quality depends heavily on where it starts, and
    trying all n starts costs only a factor n more. Each start's tour is
    rotated to begin and end at city 0 (a rotation keeps every edge and
    its direction, so it is the same cycle) and priced in that order. The
    cheapest is returned; on a tie the lowest start city wins. Because
    every tour is rotated to city 0, the result can be compared directly
    with the Week 6 exact solvers at their default ``start=0``.

    This is still a heuristic with no constant-factor guarantee: it is at
    least as good as :func:`tsp_nearest_neighbor` from city 0, and no
    better in the worst case than the best of n greedy walks.

    Args:
        dist: Square matrix of non-negative edge costs. May be asymmetric.
            Not modified.

    Returns:
        A ``(cost, tour)`` pair with ``tour[0] == tour[-1] == 0`` and
        ``cost == tour_cost(dist, tour)``.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers.
        ValueError: If ``dist`` is empty, not square, or holds a negative
            or NaN entry.

    Time Complexity:
        O(n^3) - n runs of the O(n^2) greedy walk.

    Space Complexity:
        O(n^2) - the validated copy of ``dist``; O(n) beyond it.

    Examples:
        On the line instance the walk from city 3 (x = 4) goes 4, 1, 0,
        -2, -8, 16 and back, costing 3 + 1 + 2 + 6 + 24 + 12 = 48, the
        optimum. Rotated to city 0:

        >>> xs = [0, 1, -2, 4, -8, 16]
        >>> line = [[abs(a - b) for b in xs] for a in xs]
        >>> tsp_nearest_neighbor_best_start(line)
        (48, [0, 2, 4, 5, 3, 1, 0])
        >>> tsp_nearest_neighbor_best_start([[0]])
        (0, [0, 0])
    """
    rows = _validate_matrix(dist)
    # Start 0's tour already begins at city 0, so it needs no rotation.
    best_tour = _nearest_neighbor_tour(rows, 0)
    best_cost = _closed_cost(rows, best_tour)
    for start in range(1, len(rows)):
        tour = _rotate_to(_nearest_neighbor_tour(rows, start), 0)
        cost = _closed_cost(rows, tour)
        if cost < best_cost:  # strict: on a tie the lower start is kept
            best_cost, best_tour = cost, tour
    return best_cost, best_tour


def _edge_arrays(
    rows: Matrix, tour: Tour, symmetric: bool
) -> Tuple[List[float], List[float], List[float]]:
    """Return the per-position edge costs and, if needed, two prefix sums.

    ``edge[p]`` is the cost of the edge from position ``p`` to ``p + 1``.
    For an asymmetric matrix, ``fwd[k]`` is the cost of the first ``k``
    edges as driven and ``bwd[k]`` the cost of the same edges driven
    backwards, so the cost of the path between positions ``s`` and ``t``
    is ``fwd[t] - fwd[s]`` forwards and ``bwd[t] - bwd[s]`` reversed. For a
    symmetric matrix both directions cost the same and the prefix sums are
    not built.

    Args:
        rows: A validated distance matrix.
        tour: A closed tour of length n + 1.
        symmetric: Whether ``rows`` is symmetric.

    Returns:
        ``(edge, fwd, bwd)``; ``fwd`` and ``bwd`` are empty when
        ``symmetric`` is true.

    Raises:
        Nothing; the matrix and the tour are validated by the caller.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> rows = [[0, 1, 10], [10, 0, 1], [1, 10, 0]]
        >>> _edge_arrays(rows, [0, 1, 2, 0], symmetric=False)
        ([1, 1, 1], [0, 1, 2, 3], [0, 10, 20, 30])
    """
    edge = [rows[a][b] for a, b in zip(tour, tour[1:])]
    if symmetric:
        return edge, [], []
    fwd: List[float] = [0]
    bwd: List[float] = [0]
    for a, b in zip(tour, tour[1:]):
        fwd.append(fwd[-1] + rows[a][b])
        bwd.append(bwd[-1] + rows[b][a])
    return edge, fwd, bwd


def _two_opt_pass(
    rows: Matrix, tour: Tour, symmetric: bool, tolerance: float
) -> bool:
    """Run one first-improvement 2-opt pass over ``tour``, in place.

    For positions ``0 <= i`` and ``i + 2 <= j <= n - 1`` the move reverses
    ``tour[i + 1 .. j]``, replacing edges ``(a, b) = (tour[i], tour[i+1])``
    and ``(c, e) = (tour[j], tour[j+1])`` by ``(a, c)`` and ``(b, e)``. Its
    change in cost is ``d(a, c) + d(b, e) - d(a, b) - d(c, e)`` plus, on an
    asymmetric matrix, the reversed path's cost minus its forward cost.
    Position 0 and position n are never inside a reversed range, so the
    start city stays put.

    Args:
        rows: A validated distance matrix.
        tour: A valid closed tour; modified in place.
        symmetric: Whether ``rows`` is symmetric.
        tolerance: A move is applied only if its change is below
            ``-tolerance``.

    Returns:
        True if at least one move was applied.

    Raises:
        Nothing; the matrix and the tour are validated by the caller.

    Time Complexity:
        O(n^2) candidate moves, each priced in O(1), plus O(n) for every
        applied move (the reversal and the refreshed edge arrays).

    Space Complexity:
        O(n) - the edge arrays.

    Examples:
        A crossed tour of the unit square (diagonals cost 3 here, to keep
        the numbers whole) is uncrossed by one move:

        >>> square = [[0, 1, 3, 1], [1, 0, 1, 3], [3, 1, 0, 1], [1, 3, 1, 0]]
        >>> tour = [0, 2, 1, 3, 0]
        >>> _two_opt_pass(square, tour, True, 1e-12), tour
        (True, [0, 1, 2, 3, 0])
        >>> _two_opt_pass(square, tour, True, 1e-12)
        False
    """
    n = len(tour) - 1
    edge, fwd, bwd = _edge_arrays(rows, tour, symmetric)
    moved = False
    for i in range(n - 2):
        row_a = rows[tour[i]]
        for j in range(i + 2, n):
            b, c, e = tour[i + 1], tour[j], tour[j + 1]
            delta = row_a[c] + rows[b][e] - edge[i] - edge[j]
            if not symmetric:
                delta += (bwd[j] - bwd[i + 1]) - (fwd[j] - fwd[i + 1])
            if delta < -tolerance:
                tour[i + 1 : j + 1] = tour[j:i:-1]
                edge, fwd, bwd = _edge_arrays(rows, tour, symmetric)
                moved = True
    return moved


def tsp_two_opt(
    dist: Sequence[Sequence[float]],
    tour: Sequence[int],
    max_passes: Optional[int] = None,
) -> Tuple[float, Tour]:
    """Improve a closed tour with first-improvement 2-opt local search.

    A 2-opt move removes two edges and reconnects the two paths the other
    way, which reverses the stretch of the tour between them. This
    function is FIRST-IMPROVEMENT: each pass scans the pairs of positions
    in increasing order and applies every improving move the moment it is
    found, continuing the scan on the updated tour. Passes repeat until a
    pass finds no improving move (the tour is then 2-optimal) or until
    ``max_passes`` passes have run. ``tour[0]`` never moves, so the result
    starts and ends at the same city as the input. Asymmetric matrices are
    handled exactly: the reversed stretch is priced in its new direction.

    The cost never increases. A move must shorten the tour by more than
    ``1e-12 * max(1, cost)``, and after each pass the tour is re-priced in
    full; a pass that did not strictly lower that price is undone and the
    search stops. Since the price strictly falls with every pass that is
    kept and there are finitely many tours, the search always terminates.

    2-opt is a local search with no approximation guarantee: it stops at a
    local optimum, whose ratio to the true optimum can grow like sqrt(n) on
    metric instances (Chandra, Karloff and Tovey 1999), and no polynomial
    bound on its number of moves is known: even in the Euclidean plane
    there are inputs on which 2-opt can take exponentially many improving
    moves (Englert, Roeglin and Voecking 2014). In practice it removes most
    of nearest neighbor's excess.

    Args:
        dist: Square matrix of non-negative edge costs. May be asymmetric.
            Not modified.
        tour: A closed tour of all n cities as a list or tuple, for
            example the tour from :func:`tsp_nearest_neighbor`. Any start
            city is allowed. Not modified.
        max_passes: The largest number of full passes to run, or ``None``
            to run until no improving move remains. ``0`` returns the input
            tour unchanged.

    Returns:
        A ``(cost, tour)`` pair: a new closed tour with the same first city
        and ``cost == tour_cost(dist, tour) <= tour_cost(dist, input)``.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers, or
            ``max_passes`` is neither ``None`` nor an ``int``.
        ValueError: If ``dist`` is invalid, if ``tour`` is not a closed tour
            of all n cities (see :func:`is_valid_tour`), or if
            ``max_passes`` is negative.

    Time Complexity:
        O(n^2) per pass: n^2 / 2 candidate moves priced in O(1) each, plus
        O(n) per applied move. The number of passes is small in practice,
        but no polynomial bound on it is known for the worst case.

    Space Complexity:
        O(n^2) - the validated copy of ``dist``; O(n) beyond it.

    Examples:
        The unit square driven with a crossing, ``0 -> 2 -> 1 -> 3``,
        costs 2 + 2 * sqrt(2); one move uncrosses it to the perimeter, 4:

        >>> s = 2 ** 0.5
        >>> square = [[0, 1, s, 1], [1, 0, 1, s], [s, 1, 0, 1], [1, s, 1, 0]]
        >>> round(tour_cost(square, [0, 2, 1, 3, 0]), 6)
        4.828427
        >>> tsp_two_opt(square, [0, 2, 1, 3, 0])
        (4, [0, 1, 2, 3, 0])

        From nearest neighbor's zig-zag on the line instance, three moves
        in the first pass reach the optimum 48:

        >>> xs = [0, 1, -2, 4, -8, 16]
        >>> line = [[abs(a - b) for b in xs] for a in xs]
        >>> tsp_two_opt(line, [0, 1, 2, 3, 4, 5, 0])
        (48, [0, 4, 2, 1, 3, 5, 0])

        The start city is kept, and ``max_passes=0`` changes nothing:

        >>> tsp_two_opt(square, [2, 0, 1, 3, 2])
        (4, [2, 1, 0, 3, 2])
        >>> tsp_two_opt(square, [0, 2, 1, 3, 0], max_passes=0)[1]
        [0, 2, 1, 3, 0]
        >>> tsp_two_opt(square, [0, 1, 2, 0])
        Traceback (most recent call last):
            ...
        ValueError: tour must be a closed tour of all 4 cities, got [0, 1, 2, 0]
    """
    rows = _validate_matrix(dist)
    n = len(rows)
    if max_passes is not None:
        _require_int(max_passes, "max_passes")
        if max_passes < 0:
            raise ValueError(f"max_passes must be >= 0, got {max_passes}")
    if not is_valid_tour(tour, n):
        raise ValueError(
            f"tour must be a closed tour of all {n} cities, got {tour!r}"
        )

    current: Tour = [int(city) for city in tour]
    cost = _closed_cost(rows, current)
    symmetric = all(rows[i][j] == rows[j][i] for i in range(n) for j in range(i))

    passes = 0
    while max_passes is None or passes < max_passes:
        passes += 1
        scale = abs(cost) if math.isfinite(cost) else 0.0
        tolerance = _EPSILON * max(1.0, scale)
        before = current[:]
        if not _two_opt_pass(rows, current, symmetric, tolerance):
            break
        new_cost = _closed_cost(rows, current)
        if not new_cost < cost:
            current = before  # rounding noise or inf: undo the pass and stop
            break
        cost = new_cost
    return cost, current


def tour_cost(dist: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    """Return the total cost of a closed tour: the polynomial-time verifier.

    Sums ``dist[tour[k]][tour[k + 1]]`` over consecutive positions, in
    order, which is exactly how every solver in this module prices its own
    tour. A step from a city to itself (only the one-city tour ``[c, c]``
    has one) costs 0, so the diagonal is never read.

    Together with :func:`is_valid_tour` this is the certificate check for
    the TSP decision problem ("is there a tour of cost at most B?"): given
    a tour, deciding whether it is a valid tour within budget takes O(n)
    time, while finding one takes exponential time for every known exact
    algorithm. To keep the check O(n), ``dist`` is used as given and not
    validated entry by entry; only the tour's cities are checked. The tour
    is not checked to be a permutation; that is :func:`is_valid_tour`'s
    job.

    Args:
        dist: A square distance matrix (list of lists or anything indexable
            the same way). Not modified.
        tour: A closed sequence of city indices, ``tour[0] == tour[-1]``.

    Returns:
        The sum of the tour's edge costs. ``0`` for ``[c, c]``.

    Raises:
        TypeError: If a city is not an ``int`` (``bool`` refused).
        ValueError: If ``tour`` has fewer than two entries, is not closed,
            or names a city outside ``range(len(dist))``.

    Time Complexity:
        O(n) - one type and range check and one lookup per edge.

    Space Complexity:
        O(n) - a list copy of the tour.

    Examples:
        >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25],
        ...         [15, 35, 0, 30], [20, 25, 30, 0]]
        >>> tour_cost(dist, [0, 1, 3, 2, 0])
        80
        >>> tour_cost([[7]], [0, 0])
        0
        >>> tour_cost(dist, [0, 1, 3, 2])
        Traceback (most recent call last):
            ...
        ValueError: tour must start and end at the same city, got [0, 1, 3, 2]
        >>> tour_cost(dist, [0, 4, 0])
        Traceback (most recent call last):
            ...
        ValueError: tour city 4 is outside range(4)
    """
    cities = list(tour)
    if len(cities) < 2 or cities[0] != cities[-1]:
        raise ValueError(
            f"tour must start and end at the same city, got {cities!r}"
        )
    n = len(dist)
    for city in cities:
        if isinstance(city, bool) or not isinstance(city, numbers.Integral):
            raise TypeError(f"tour cities must be ints, got {type(city).__name__}")
        if not 0 <= city < n:
            raise ValueError(f"tour city {city} is outside range({n})")
    return _closed_cost(dist, cities)


def is_valid_tour(tour: Any, n: int) -> bool:
    """Return whether ``tour`` is a closed tour visiting every city once.

    A valid tour for n cities is a list or tuple of n + 1 integer city
    indices that starts and ends at the same city and whose first n entries
    are exactly the cities ``0 .. n - 1``, each once. Any start city is
    allowed. The one-city tour is ``[0, 0]``. Anything else, including
    entries that are ``bool`` or not integers at all, gives ``False``
    rather than an exception, because this is a predicate about the tour.

    Args:
        tour: The candidate tour.
        n: The number of cities, at least 1.

    Returns:
        True if ``tour`` is a valid closed tour of all n cities.

    Raises:
        TypeError: If ``n`` is not an ``int`` (``bool`` refused).
        ValueError: If ``n`` is less than 1.

    Time Complexity:
        O(n) - one pass with a ``seen`` array.

    Space Complexity:
        O(n) - the ``seen`` array.

    Examples:
        >>> is_valid_tour([0, 1, 3, 2, 0], 4)
        True
        >>> is_valid_tour([2, 0, 1, 2], 3)
        True
        >>> is_valid_tour([0, 0], 1)
        True
        >>> is_valid_tour([0, 1, 3, 2], 4)
        False
        >>> is_valid_tour([0, 1, 1, 2, 0], 4)
        False
        >>> is_valid_tour([0, True, 0], 2)
        False
        >>> is_valid_tour([0, 1, 0], 0)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 1, got 0
    """
    _require_int(n, "n")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not isinstance(tour, (list, tuple)) or len(tour) != n + 1:
        return False
    seen = [False] * n
    for city in tour[:-1]:
        if isinstance(city, bool) or not isinstance(city, numbers.Integral):
            return False
        if not 0 <= city < n or seen[city]:
            return False
        seen[city] = True
    last = tour[-1]
    if isinstance(last, bool) or not isinstance(last, numbers.Integral):
        return False
    return last == tour[0]


def euclidean_instance(
    n: int,
    seed: Optional[int],
    size: float = 100.0,
) -> Tuple[List[Tuple[float, float]], Matrix]:
    """Generate n random points in a square and their distance matrix.

    Each point is ``(size * rng.random(), size * rng.random())``, x first,
    drawn in order from ``rng = make_rng(seed)``, so the points are uniform
    in ``[0, size)^2`` and the same seed always gives the same instance.
    ``matrix[i][j]`` is ``math.dist(points[i], points[j])``, computed once
    per pair and mirrored, so the matrix is exactly symmetric with a zero
    diagonal. Euclidean distances obey the triangle inequality, so these
    are metric instances, the family on which nearest neighbor's ratio is
    at most Theta(log n) and usually much smaller.

    Args:
        n: Number of points, at least 1.
        seed: Passed to :func:`src.utils.random_utils.make_rng`. ``None``
            gives an unpredictable instance; experiments always pass an
            ``int``.
        size: Side length of the square, a positive finite real.

    Returns:
        ``(points, matrix)``: a list of n ``(x, y)`` tuples and the n x n
        list-of-lists distance matrix.

    Raises:
        TypeError: If ``n`` is not an ``int``, ``seed`` is neither ``None``
            nor an ``int`` (``bool`` refused for both), or ``size`` is not
            a real number.
        ValueError: If ``n < 1`` or ``size`` is not positive and finite.

    Time Complexity:
        O(n^2) - one distance per pair.

    Space Complexity:
        O(n^2) - the matrix.

    Examples:
        >>> points, matrix = euclidean_instance(5, 42)
        >>> [round(v, 6) for v in points[0]]
        [63.94268, 2.501076]
        >>> len(points), len(matrix), len(matrix[0])
        (5, 5, 5)
        >>> all(matrix[i][j] == matrix[j][i] for i in range(5) for j in range(5))
        True
        >>> [matrix[i][i] for i in range(5)]
        [0.0, 0.0, 0.0, 0.0, 0.0]
        >>> euclidean_instance(5, 42) == euclidean_instance(5, 42)
        True
        >>> euclidean_instance(0, 42)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 1, got 0
    """
    _require_int(n, "n")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if isinstance(size, bool) or not isinstance(size, numbers.Real):
        raise TypeError(f"size must be a real number, got {type(size).__name__}")
    if not (math.isfinite(size) and size > 0):
        raise ValueError(f"size must be positive and finite, got {size}")
    rng = make_rng(seed)
    side = float(size)

    points: List[Tuple[float, float]] = []
    for _ in range(n):
        x = side * rng.random()
        y = side * rng.random()
        points.append((x, y))

    matrix: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(points[i], points[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return points, matrix
