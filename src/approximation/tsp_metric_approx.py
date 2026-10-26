"""Metric TSP: the MST-doubling 2-approximation, with Christofides for comparison.

The traveling salesman problem asks for the cheapest closed tour through n
cities given a matrix ``dist`` in which ``dist[i][j]`` is the cost of going
between cities ``i`` and ``j``. Week 8 showed that the exact solvers take
exponential time and that the nearest-neighbor heuristic has no
constant-factor guarantee. This module adds an algorithm that DOES have
one, provided the instance is *metric*: symmetric, zero on the diagonal,
and obeying the triangle inequality ``dist[i][j] <= dist[i][k] +
dist[k][j]`` (going through a third city is never a shortcut). Euclidean
instances are metric.

The algorithm (MST doubling, "twice around the tree")
-----------------------------------------------------
:func:`tsp_mst_doubling` builds a minimum spanning tree with Prim's
algorithm (:func:`mst_weight_and_edges`), roots it at ``start`` and lists
the cities in depth-first PREORDER, children in increasing index order.
That order, closed back to ``start``, is the tour. Everything is O(n^2) on
the complete graph.

The guarantee and its proof
---------------------------
**Theorem.** On a metric instance the tour costs at most 2 * w(MST), and
w(MST) <= OPT, so the tour costs at most 2 * OPT.

1. **The MST is a lower bound on the optimal tour.** Take an optimal tour
   of cost OPT and delete any one of its edges. What is left is a path
   through every city, which is a spanning tree, and since edge costs are
   non-negative it costs at most OPT. The MST is the cheapest spanning
   tree, so w(MST) <= OPT.
2. **Doubling gives a closed walk of cost exactly 2 * w(MST).** Replace
   every tree edge by two copies. Every vertex now has even degree and the
   multigraph is connected, so it has an Euler circuit W; concretely, W is
   the walk a depth-first search makes when it goes down each tree edge and
   later comes back up it. W visits every city, some of them many times,
   and costs exactly 2 * w(MST).
3. **Shortcuts never lengthen a tour under the triangle inequality.** Walk
   along W and skip every city already visited, jumping straight to the
   next new one (and, at the end, straight back to ``start``). Each skip
   replaces a stretch ``u -> x1 -> ... -> xk -> v`` of W by the single
   edge ``u -> v``, and applying the triangle inequality k times gives
   ``dist[u][v] <= dist[u][x1] + ... + dist[xk][v]``. So the shortcut tour
   costs at most cost(W) = 2 * w(MST) <= 2 * OPT. The cities appear in the
   order in which W first reaches them, which is exactly the DFS preorder,
   so the preorder IS the shortcut tour and W never has to be built.

The root does not matter for the proof (every rooting walks each tree edge
twice), so the bound holds for every ``start``. The factor 2 is tight for
this algorithm: there are metric families on which its ratio approaches 2
(Vazirani 2001, Ch. 3). The bound needs the triangle inequality only in
step 3, and that is where it fails on non-metric input.

Christofides (comparison only, via networkx)
--------------------------------------------
Christofides (1976) replaces "double every tree edge" by "add a
minimum-weight perfect matching on the odd-degree vertices of the tree".
Shortcutting an optimal tour to those (evenly many) odd vertices gives a
cycle of cost <= OPT that splits into two perfect matchings, so the
cheaper one costs <= OPT / 2, and the Euler circuit of tree plus matching,
shortcut as above, costs <= OPT + OPT / 2 = 1.5 * OPT.
:func:`tsp_christofides_networkx` is NOT our implementation: it calls
``networkx.algorithms.approximation.christofides`` and is labelled as a
library comparison (3/2 bound) everywhere it is reported.

When no approximation is possible
---------------------------------
Without the triangle inequality there is no constant-factor approximation
at all unless P = NP (Sahni and Gonzalez 1976). Suppose a polynomial-time
algorithm always came within a factor ``a`` of OPT. Given a graph G on n
vertices, give its edges cost 1 and every non-edge cost ``a * n + 1``. If
G has a Hamiltonian cycle, OPT = n and the algorithm must return a tour of
cost <= a * n, which can use only edges of G; if it has none, every tour
costs more than a * n. The algorithm would then decide Hamiltonian Cycle,
which is NP-complete. The costs 1 and ``a * n + 1`` break the triangle
inequality, which is exactly why :func:`is_metric` is checked before MST
doubling is trusted.

Contract
--------
* ``dist`` is a square n x n matrix of non-negative, finite real numbers,
  n >= 1, given as a list of lists, and it must be symmetric (an MST is a
  tree of an undirected graph). Validation reuses Week 8's checks, so the
  error messages match :mod:`src.complexity.traveling_salesman_approx`.
  The full metric check is O(n^3), so the solvers do not run it; the
  caller does, with :func:`is_metric`.
* A tour is CLOSED (n + 1 entries, first == last), as in Week 8; one city
  gives ``[0, 0]``. Every returned cost is exactly
  ``tour_cost(dist, tour)`` from Week 8.

==================================== ============ =========================
Function                             Time         Space
==================================== ============ =========================
mst_weight_and_edges                 O(n^2)       O(n^2) - validated copy
tsp_mst_doubling                     O(n^2)       O(n^2)
tsp_christofides_networkx            O(n^3)       O(n^2)
is_metric                            O(n^3)       O(n^2)
==================================== ============ =========================

Examples:
    Six points on a line at x = 0, 1, -2, 4, -8, 16 (the Week 8 anchor,
    where nearest neighbor zig-zags outwards for 62). The MST joins
    neighbours on the line, weight 24 = 16 - (-8), and doubling it is
    optimal here, 48 = 2 * 24:

    >>> xs = [0, 1, -2, 4, -8, 16]
    >>> line = [[abs(a - b) for b in xs] for a in xs]
    >>> mst_weight_and_edges(line)[0]
    24
    >>> tsp_mst_doubling(line)
    (48, [0, 1, 3, 5, 2, 4, 0])
    >>> is_metric(line)
    True

References:
    Christofides, N. (1976). *Worst-case analysis of a new heuristic for
    the travelling salesman problem* (Report 388). Graduate School of
    Industrial Administration, Carnegie Mellon University.

    Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network
    structure, dynamics, and function using NetworkX. In *Proceedings of
    the 7th Python in Science Conference* (pp. 11-15).

    Prim, R. C. (1957). Shortest connection networks and some
    generalizations. *Bell System Technical Journal, 36*(6), 1389-1401.

    Sahni, S., & Gonzalez, T. (1976). P-complete approximation problems.
    *Journal of the ACM, 23*(3), 555-565.

    Vazirani, V. V. (2001). *Approximation algorithms*. Springer.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import List, Sequence, Tuple

import numpy as np

from src.complexity.traveling_salesman_approx import (
    _validate_matrix,
    _validate_start,
    tour_cost,
)

__all__ = [
    "is_metric",
    "mst_weight_and_edges",
    "tsp_christofides_networkx",
    "tsp_mst_doubling",
]

Matrix = List[List[float]]
Tour = List[int]
Edge = Tuple[int, int]

#: Relative tolerance for the symmetry check in :func:`_validate_symmetric`.
#: Entries ``a`` and ``b`` count as equal when
#: ``abs(a - b) <= _SYMMETRY_TOL * max(1, abs(a), abs(b))``.
_SYMMETRY_TOL: float = 1e-9


def _validate_symmetric(dist: Sequence[Sequence[float]]) -> Matrix:
    """Validate ``dist`` as a finite symmetric distance matrix and copy it.

    Runs Week 8's :func:`_validate_matrix` (square, non-empty, real,
    non-negative, no NaN) and then requires every entry to be finite and
    ``dist[i][j]`` to equal ``dist[j][i]`` up to a relative tolerance of
    ``1e-9``. The diagonal is never read by the solvers, so it is not
    checked here.

    Args:
        dist: The candidate distance matrix. Not modified.

    Returns:
        The validated matrix as a new list of lists.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers.
        ValueError: If ``dist`` is empty, not square, holds a negative,
            NaN or infinite entry, or is not symmetric.

    Time Complexity:
        O(n^2) - one check per entry.

    Space Complexity:
        O(n^2) - the copy.

    Examples:
        >>> _validate_symmetric([[0, 2], [2, 0]])
        [[0, 2], [2, 0]]
        >>> _validate_symmetric([[0, 1], [2, 0]])
        Traceback (most recent call last):
            ...
        ValueError: dist must be symmetric: dist[0][1] = 1 but dist[1][0] = 2
        >>> _validate_symmetric([[0, float("inf")], [float("inf"), 0]])
        Traceback (most recent call last):
            ...
        ValueError: dist[0][1] must be finite, got inf
    """
    rows = _validate_matrix(dist)
    n = len(rows)
    for i in range(n):
        row = rows[i]
        for j in range(i + 1, n):
            a = row[j]
            b = rows[j][i]
            if not math.isfinite(a):
                raise ValueError(f"dist[{i}][{j}] must be finite, got {a}")
            if not math.isfinite(b):
                raise ValueError(f"dist[{j}][{i}] must be finite, got {b}")
            if abs(a - b) > _SYMMETRY_TOL * max(1.0, abs(a), abs(b)):
                raise ValueError(
                    f"dist must be symmetric: dist[{i}][{j}] = {a} but "
                    f"dist[{j}][{i}] = {b}"
                )
    return rows


def _prim(rows: Matrix) -> Tuple[float, List[Edge]]:
    """Run Prim's algorithm from vertex 0 on a validated complete graph.

    Keeps ``key[v]``, the cheapest known edge from the tree to ``v``, and
    ``parent[v]``, the tree vertex at its other end. Each round adds the
    non-tree vertex with the smallest key (ties: lowest index) and then
    lowers the keys of the remaining vertices, replacing a key only on a
    STRICTLY smaller distance, so an earlier parent wins ties.

    Args:
        rows: A validated symmetric matrix (see :func:`_validate_symmetric`).

    Returns:
        ``(weight, edges)`` with ``edges`` the ``(parent, child)`` pairs in
        the order the children joined the tree.

    Time Complexity:
        O(n^2) - n rounds, each scanning all n keys twice.

    Space Complexity:
        O(n) - the key, parent and in-tree arrays.

    Examples:
        >>> _prim([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        (2, [(0, 1), (1, 2)])
        >>> _prim([[0]])
        (0, [])
    """
    n = len(rows)
    in_tree = [False] * n
    key: List[float] = [math.inf] * n
    parent = [-1] * n
    key[0] = 0
    weight: float = 0
    edges: List[Edge] = []
    for _ in range(n):
        u = -1
        for v in range(n):
            if not in_tree[v] and (u == -1 or key[v] < key[u]):
                u = v
        in_tree[u] = True
        if parent[u] != -1:
            weight += rows[parent[u]][u]
            edges.append((parent[u], u))
        row = rows[u]
        for v in range(n):
            if not in_tree[v] and row[v] < key[v]:
                key[v] = row[v]
                parent[v] = u
    return weight, edges


def _preorder(n: int, edges: Sequence[Edge], start: int) -> Tour:
    """List the vertices of a spanning tree in DFS preorder from ``start``.

    Children are visited in increasing index order. The DFS uses an
    explicit stack (no recursion), pushing children in decreasing order so
    the smallest is popped first.

    Args:
        n: Number of vertices.
        edges: The n - 1 tree edges, in either orientation.
        start: The root.

    Returns:
        The n vertices in preorder, beginning with ``start``.

    Time Complexity:
        O(n log n) - each adjacency list is sorted once.

    Space Complexity:
        O(n) - adjacency lists, stack and visited array.

    Examples:
        >>> _preorder(4, [(0, 1), (1, 2), (0, 3)], 0)
        [0, 1, 2, 3]
        >>> _preorder(4, [(0, 1), (1, 2), (0, 3)], 2)
        [2, 1, 0, 3]
    """
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    visited = [False] * n
    order: Tour = []
    stack = [start]
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in sorted(adjacency[u], reverse=True):
            if not visited[v]:
                stack.append(v)
    return order


def mst_weight_and_edges(dist: Sequence[Sequence[float]]) -> Tuple[float, List[Edge]]:
    """Return the weight and edges of a minimum spanning tree (Prim, O(n^2)).

    Prim's algorithm on the complete graph whose edge ``{i, j}`` costs
    ``dist[i][j]``, grown from vertex 0 with a key array: each round adds
    the non-tree vertex with the cheapest edge to the tree (ties: lowest
    vertex index), and a vertex's key is replaced only by a STRICTLY
    smaller distance. On a dense graph this array version beats a heap,
    which would cost O(n^2 log n) for the n^2 edges.

    The weight is the lower bound used by the whole module: deleting one
    edge of an optimal tour leaves a spanning tree, so w(MST) <= OPT.

    Args:
        dist: Square symmetric matrix of non-negative finite edge costs.
            Not modified.

    Returns:
        ``(weight, edges)``. ``edges`` lists the n - 1 tree edges as
        ``(parent, child)`` pairs in the order the children joined the tree
        (rooted at 0); ``weight`` is the sum of their costs, an ``int`` when
        every entry is. One city gives ``(0, [])``.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers.
        ValueError: If ``dist`` is empty, not square, not symmetric, or
            holds a negative, NaN or infinite entry.

    Time Complexity:
        O(n^2) - n rounds of O(n) key scans (validation is also O(n^2)).

    Space Complexity:
        O(n^2) - the validated copy of ``dist``; O(n) beyond it.

    Examples:
        The corners of the unit square. Three sides of length 1 form the
        tree; the diagonals (sqrt 2) are never used:

        >>> square = [[0.0, 1.0, 2 ** 0.5, 1.0], [1.0, 0.0, 1.0, 2 ** 0.5],
        ...           [2 ** 0.5, 1.0, 0.0, 1.0], [1.0, 2 ** 0.5, 1.0, 0.0]]
        >>> mst_weight_and_edges(square)
        (3.0, [(0, 1), (1, 2), (0, 3)])

        Points on a line join their neighbours on the line:

        >>> xs = [0, 1, -2, 4, -8, 16]
        >>> line = [[abs(a - b) for b in xs] for a in xs]
        >>> mst_weight_and_edges(line)
        (24, [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)])
        >>> mst_weight_and_edges([[0]])
        (0, [])
    """
    rows = _validate_symmetric(dist)
    return _prim(rows)


def tsp_mst_doubling(
    dist: Sequence[Sequence[float]],
    start: int = 0,
) -> Tuple[float, Tour]:
    """Return the MST-doubling tour: a 2-approximation on metric instances.

    Builds the MST with :func:`mst_weight_and_edges`, roots it at
    ``start`` and visits the cities in depth-first preorder with children in
    increasing index order, then returns to ``start``. The preorder is the
    shortcut of the Euler circuit of the doubled tree (see the module
    docstring), so on a metric instance the tour costs at most
    2 * w(MST) <= 2 * OPT. On a non-metric instance the function still
    returns a valid tour, but the bound no longer holds; check
    :func:`is_metric` first.

    Args:
        dist: Square symmetric matrix of non-negative finite edge costs.
            Not modified.
        start: The city the tour begins and ends at.

    Returns:
        A ``(cost, tour)`` pair. ``tour`` is closed, starts and ends at
        ``start`` and visits every city once in between; ``cost`` equals
        ``tour_cost(dist, tour)``. One city gives ``(0, [0, 0])``.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers, or
            ``start`` is not an ``int``.
        ValueError: If ``dist`` is empty, not square, not symmetric, or
            holds a negative, NaN or infinite entry, or if ``start`` is out
            of range.

    Time Complexity:
        O(n^2) - Prim dominates; the preorder is O(n log n).

    Space Complexity:
        O(n^2) - the validated copy of ``dist``; O(n) beyond it.

    Examples:
        The unit square: the tree is three sides, the preorder walks
        around the square, and the tour is the perimeter, which is optimal:

        >>> square = [[0.0, 1.0, 2 ** 0.5, 1.0], [1.0, 0.0, 1.0, 2 ** 0.5],
        ...           [2 ** 0.5, 1.0, 0.0, 1.0], [1.0, 2 ** 0.5, 1.0, 0.0]]
        >>> tsp_mst_doubling(square)
        (4.0, [0, 1, 2, 3, 0])

        Points on a line. Rooted at 0 the children are 1 and 2; the walk
        goes out along 1 -> 3 -> 5, shortcuts back to 2 -> 4 and home, for
        1 + 3 + 12 + 18 + 6 + 8 = 48, the optimum. Rooting at 3 gives a
        different tour of the same cost:

        >>> xs = [0, 1, -2, 4, -8, 16]
        >>> line = [[abs(a - b) for b in xs] for a in xs]
        >>> tsp_mst_doubling(line)
        (48, [0, 1, 3, 5, 2, 4, 0])
        >>> tsp_mst_doubling(line, start=3)
        (48, [3, 1, 0, 2, 4, 5, 3])
        >>> tsp_mst_doubling([[0]])
        (0, [0, 0])
        >>> tsp_mst_doubling(line, start=6)
        Traceback (most recent call last):
            ...
        ValueError: start must be in range(6), got 6
    """
    rows = _validate_symmetric(dist)
    n = len(rows)
    _validate_start(start, n)
    _, edges = _prim(rows)
    tour = _preorder(n, edges, start)
    tour.append(start)
    return tour_cost(rows, tour), tour


def tsp_christofides_networkx(dist: Sequence[Sequence[float]]) -> Tuple[float, Tour]:
    """Return the Christofides tour from networkx (LIBRARY comparison, 3/2 bound).

    This is not our own implementation. It builds the complete weighted
    graph on the n cities, calls
    ``networkx.algorithms.approximation.christofides`` (MST plus a
    minimum-weight perfect matching on the tree's odd-degree vertices,
    Euler circuit, shortcuts), and rotates the returned cycle so that it
    starts and ends at city 0. On a metric instance the tour costs at most
    1.5 * OPT. Wherever it is reported (CSV, README, report) it is
    labelled as the networkx library implementation.

    Args:
        dist: Square symmetric matrix of non-negative finite edge costs.
            Not modified.

    Returns:
        A ``(cost, tour)`` pair: a closed tour starting and ending at 0 and
        its cost, ``tour_cost(dist, tour)``. One city gives
        ``(0, [0, 0])`` (networkx refuses a one-node graph).

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers.
        ValueError: If ``dist`` is empty, not square, not symmetric, or
            holds a negative, NaN or infinite entry.

    Time Complexity:
        O(n^3) - dominated by networkx's minimum-weight matching (blossom
        algorithm) on up to n odd-degree vertices.

    Space Complexity:
        O(n^2) - the complete graph.

    Examples:
        On the unit square every MST is a path along three sides, its two
        odd (end) vertices are adjacent, and tree plus matching is the
        perimeter:

        >>> square = [[0.0, 1.0, 2 ** 0.5, 1.0], [1.0, 0.0, 1.0, 2 ** 0.5],
        ...           [2 ** 0.5, 1.0, 0.0, 1.0], [1.0, 2 ** 0.5, 1.0, 0.0]]
        >>> cost, tour = tsp_christofides_networkx(square)
        >>> cost, tour[0], tour[-1], sorted(tour[:-1])
        (4.0, 0, 0, [0, 1, 2, 3])
        >>> tsp_christofides_networkx([[0]])
        (0, [0, 0])
    """
    rows = _validate_symmetric(dist)
    n = len(rows)
    if n == 1:
        return tour_cost(rows, [0, 0]), [0, 0]

    import networkx as nx
    from networkx.algorithms.approximation import christofides

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        (i, j, rows[i][j]) for i in range(n) for j in range(i + 1, n)
    )
    cycle = list(christofides(graph, weight="weight"))
    body = cycle[:-1]
    k = body.index(0)
    tour = body[k:] + body[:k] + [0]
    return tour_cost(rows, tour), tour


def is_metric(dist: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    """Return whether ``dist`` is a metric: symmetric, zero diagonal, triangle inequality.

    Checks, with absolute tolerance ``tol``: every entry finite (no NaN or
    infinity, as a metric is real-valued and the solvers refuse infinite
    entries) and none below ``-tol``; ``|dist[i][i]| <= tol``; ``dist[i][j]`` equals
    ``dist[j][i]`` (exactly, or within ``tol``); and
    ``dist[i][j] <= dist[i][k] + dist[k][j] + tol`` for every i, j, k.
    These are the hypotheses of the 2 * OPT (MST doubling) and 1.5 * OPT
    (Christofides) bounds; without them TSP has no constant-factor
    approximation unless P = NP. The O(n^3) triangle check runs as n
    vectorised numpy comparisons, one per intermediate city k, so n = 300
    is quick. A violation gives ``False``; only input that is not a square
    matrix of numbers raises.

    Args:
        dist: Square matrix of real numbers (list of lists). Not modified.
        tol: Non-negative absolute tolerance for floating-point rounding.

    Returns:
        True if every metric condition holds within ``tol``.

    Raises:
        TypeError: If ``tol`` is not a real number.
        ValueError: If ``dist`` is empty, not a square matrix of numbers,
            or ``tol`` is negative or NaN.

    Time Complexity:
        O(n^3) - n^2 triangle checks for each of the n intermediate cities.

    Space Complexity:
        O(n^2) - a float array copy and one n x n comparison at a time.

    Examples:
        >>> is_metric([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        True

        Going 0 -> 1 -> 2 costs 2, cheaper than the direct edge of 5:

        >>> is_metric([[0, 1, 5], [1, 0, 1], [5, 1, 0]])
        False
        >>> is_metric([[0, 1], [2, 0]])
        False
        >>> is_metric([[1, 1], [1, 0]])
        False
        >>> is_metric([[0, float("inf")], [float("inf"), 0]])
        False
        >>> is_metric([[0, 1], [1]])
        Traceback (most recent call last):
            ...
        ValueError: dist must be a non-empty square matrix of real numbers
    """
    if isinstance(tol, bool) or not isinstance(tol, numbers.Real):
        raise TypeError(f"tol must be a real number, got {type(tol).__name__}")
    if not tol >= 0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    message = "dist must be a non-empty square matrix of real numbers"
    try:
        arr = np.array(dist, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[0] != arr.shape[1]:
        raise ValueError(message)

    if not np.isfinite(arr).all() or (arr < -tol).any():
        return False
    if (np.abs(np.diagonal(arr)) > tol).any():
        return False
    with np.errstate(invalid="ignore"):
        symmetric = (arr == arr.T) | (np.abs(arr - arr.T) <= tol)
    if not symmetric.all():
        return False
    for k in range(arr.shape[0]):
        via_k = arr[:, k : k + 1] + arr[k : k + 1, :]
        if (arr > via_k + tol).any():
            return False
    return True
