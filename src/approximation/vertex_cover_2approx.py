"""Vertex Cover two ways: a combinatorial 2-approximation and LP rounding.

A *vertex cover* of an undirected graph G = (V, E) is a set C of vertices
that touches every edge: for each edge (u, v), u is in C or v is in C.
Finding a smallest cover (size OPT) is NP-hard, so this module gives two
polynomial-time algorithms that are *proven* to stay within a factor 2 of
OPT, together with the two lower bounds on OPT that the proofs rest on.
Neither proof needs to know OPT: each algorithm brings its own certificate.

Algorithm 1: greedy edge picking (maximal matching)
---------------------------------------------------
:func:`vertex_cover_2approx` walks the edges (sorted order by default, a
seeded shuffle when ``seed`` is given). Whenever it meets an edge (u, v)
with neither end in the cover yet, it adds BOTH u and v.

**Theorem.** The result C is a vertex cover and |C| <= 2 OPT.

*Proof.*

1. *C is a cover.* An edge is either picked, and then both its ends are in
   C, or skipped, and it is skipped only because one of its ends was
   already in C. Either way the edge is touched.
2. *The picked edges form a matching M.* Once (u, v) is picked, u and v are
   in C, so no later edge touching u or v can be picked. Hence no two
   picked edges share an endpoint.
3. *OPT >= |M|.* Any vertex cover C* must contain at least one endpoint of
   every edge of M. The edges of M are pairwise disjoint, so these
   endpoints are |M| *distinct* vertices of C*, and |C*| >= |M|.
4. *|C| = 2|M|.* Each picked edge adds exactly two new vertices, and
   nothing else is ever added.

Combining 3 and 4: |C| = 2|M| <= 2 OPT. The argument never uses the
visiting order, so the bound holds for every order. :func:`matching_lower_bound`
returns |M|, the certified lower bound: on any graph,
``matching_lower_bound <= OPT <= len(vertex_cover_2approx) = 2 * matching_lower_bound``.

The factor 2 is tight for this algorithm: on the path 1-2-3-4 the sorted
order picks (1, 2) and (3, 4) and returns all four vertices against OPT = 2
({2, 3}); on the star with centre 0 and leaves 1-5 it returns {0, 1}
against OPT = 1 ({0}).

Algorithm 2: LP relaxation and threshold rounding
-------------------------------------------------
Write a cover as a 0/1 vector x (x_v = 1 when v is in the cover). Then the
minimum vertex cover is the integer program

    minimize sum_v x_v  subject to  x_u + x_v >= 1 for every edge (u, v),
                                    x_v in {0, 1}.

:func:`vertex_cover_lp` relaxes x_v in {0, 1} to 0 <= x_v <= 1 and solves
the resulting linear program with SciPy's HiGHS
(:func:`src.utils.lp_solver.solve_lp`); call its optimum LP.
:func:`vertex_cover_lp_rounding` then keeps every vertex with x_v >= 1/2.

**Lemma (LP <= OPT).** The 0/1 indicator vector of a minimum cover
satisfies every LP constraint, so it is LP-feasible with objective OPT; the
LP minimizes over a superset of the integer solutions, hence LP <= OPT.

**Theorem.** The rounded set C = {v : x_v >= 1/2} is a vertex cover and
|C| <= 2 LP <= 2 OPT.

*Proof.*

1. *C is a cover.* For each edge (u, v) the constraint x_u + x_v >= 1
   forces max(x_u, x_v) >= 1/2, so at least one endpoint is rounded up.
2. *|C| <= 2 LP.* Every v in C has x_v >= 1/2, i.e. 1 <= 2 x_v, so
   |C| = sum_{v in C} 1 <= sum_{v in C} 2 x_v <= 2 sum_{v in V} x_v = 2 LP
   (the last step adds the non-negative x_v of the vertices left out).
3. By the lemma, 2 LP <= 2 OPT.

The two lower bounds are ordered: |M| <= LP <= OPT for every matching M,
because adding the constraints x_u + x_v >= 1 of the pairwise disjoint
edges of M gives sum_v x_v >= |M| for every LP-feasible x.

**Integrality gap and tightness.** On the triangle K3 the point
x = (1/2, 1/2, 1/2) satisfies every edge with equality, and adding the
three constraints gives 2 (x_0 + x_1 + x_2) >= 3, so LP = 1.5 exactly,
while any integral cover needs two vertices (OPT = 2): an integrality gap
of 2 / 1.5 = 4/3. Rounding keeps all three vertices, a ratio of 1.5 to OPT
and exactly 2.0 to LP, so the bound |C| <= 2 LP is tight. On the complete
graph K_n the gap is (n - 1) / (n / 2) = 2 - 2/n, which tends to 2, so no
rounding of this LP can prove a factor better than 2.

The LP is *half-integral* (Nemhauser and Trotter, 1975): every basic
optimal solution has x_v in {0, 1/2, 1}, and HiGHS returns such a vertex
on these instances. The threshold is ``1/2 - 1e-9`` so a floating-point
value such as 0.49999999999 still counts as 1/2. As a numerical safety net
that never fires on exact solver output, an edge left uncovered by the
threshold (possible only if the solver violated its constraint by more than
2e-9) gets its endpoint with the larger x_v, so the result is always a
cover.

The exact solver
----------------
:data:`vertex_cover_exact` is Week 8's
:func:`src.complexity.vertex_cover_approx.vertex_cover_exact` (branch and
bound, O(2^k (V + E)) for OPT = k), re-exported so Week 9 measures ratios
against the same optimum. It is for small graphs only.

========================== ============================= ===============
Function                   Time                          Space
========================== ============================= ===============
vertex_cover_2approx       O(V + E) walk + O(E log E)    O(V + E)
                           sort for a reproducible order
matching_lower_bound       same as above                 O(V + E)
vertex_cover_lp            O(V log V + E log E + V E)    O(V E) dense
                           to build + one HiGHS LP solve constraint matrix
vertex_cover_lp_rounding   vertex_cover_lp + O(V + E)    O(V E)
vertex_cover_exact         O(2^k (V + E))                O(V + E)
approximation_ratio        O(1)                          O(1)
========================== ============================= ===============

Representation: a graph is ``(vertices, edges)``, ``vertices`` any
non-string iterable of hashable labels and ``edges`` 2-tuples, written
either way round (see :func:`src.utils.graph_utils.normalize_edges`).

Examples:
    >>> triangle = ([0, 1, 2], {(0, 1), (0, 2), (1, 2)})
    >>> sorted(vertex_cover_2approx(*triangle)), matching_lower_bound(*triangle)
    ([0, 1], 1)
    >>> vertex_cover_lp(*triangle)
    (1.5, {0: 0.5, 1: 0.5, 2: 0.5})
    >>> cover, lp_value = vertex_cover_lp_rounding(*triangle)
    >>> sorted(cover), lp_value, len(vertex_cover_exact(*triangle))
    ([0, 1, 2], 1.5, 2)
    >>> approximation_ratio(len(cover), lp_value)
    2.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple

from src.complexity import vertex_cover_approx as _week8
from src.utils.graph_utils import normalize_edges
from src.utils.lp_solver import solve_lp

__all__ = [
    "approximation_ratio",
    "matching_lower_bound",
    "vertex_cover_2approx",
    "vertex_cover_exact",
    "vertex_cover_lp",
    "vertex_cover_lp_rounding",
]

#: An undirected edge in canonical ``(min, max)`` form.
Edge = Tuple[Hashable, Hashable]

#: Slack below 1/2 that still rounds a fractional x_v up to 1.
_ROUND_TOL = 1e-9

#: The exact minimum vertex cover (branch and bound), re-exported from
#: Week 8 so every ratio in Week 9 is measured against the same optimum.
vertex_cover_exact = _week8.vertex_cover_exact


def _fallback_key(item: Any) -> Tuple[str, str]:
    """Return a process-stable sort key for a label that cannot use ``<``.

    Args:
        item: Any vertex label.

    Returns:
        ``(type name, repr)``, the same in every process (unlike the salted
        built-in ``hash()``).

    Time Complexity:
        O(L) for a ``repr`` of length L.

    Space Complexity:
        O(L).

    Examples:
        >>> _fallback_key("a")
        ('str', "'a'")
    """
    return (type(item).__name__, repr(item))


def _sorted_labels(items: Iterable[Any], pairs: bool = False) -> List[Any]:
    """Return ``items`` sorted, falling back to a stable key for mixed types.

    Args:
        items: Vertex labels (``pairs=False``) or normalized edges
            (``pairs=True``).
        pairs: When ``True`` each item is a 2-tuple edge and the fallback
            key is applied to each endpoint; when ``False`` each item is one
            label (which may itself be a tuple) and gets one fallback key.

    Returns:
        A new ascending list; if ``<`` fails (for example ``int`` and
        ``str`` together) items are sorted by :func:`_fallback_key` instead.

    Time Complexity:
        O(k log k) for k items.

    Space Complexity:
        O(k).

    Examples:
        >>> _sorted_labels([3, 1, 2])
        [1, 2, 3]
        >>> _sorted_labels([2, "a", 1])
        [1, 2, 'a']
        >>> _sorted_labels([1, "a", (0, 1)])
        [1, 'a', (0, 1)]
        >>> _sorted_labels([(1, "b"), (0, 2)], pairs=True)
        [(0, 2), (1, 'b')]
    """
    listed = list(items)
    try:
        return sorted(listed)
    except TypeError:
        if pairs:
            return sorted(listed, key=lambda e: (_fallback_key(e[0]), _fallback_key(e[1])))
        return sorted(listed, key=_fallback_key)


def _prepare(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Tuple[List[Hashable], List[Edge]]:
    """Validate a graph and return its vertices and edges in sorted order.

    Args:
        vertices: A non-string iterable of hashable labels.
        edges: The edges, in any form :func:`normalize_edges` accepts.

    Returns:
        ``(order, edge_list)``: the distinct vertices sorted, and the
        normalized ``(min, max)`` edges sorted. Sorting fixes the LP's
        variable and constraint order, so the result is reproducible.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable
            or a label is unhashable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint that is not in ``vertices``.

    Time Complexity:
        O(V log V + E log E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> _prepare([2, 0, 1, 0], {(1, 0), (2, 1)})
        ([0, 1, 2], [(0, 1), (1, 2)])
        >>> _prepare([0, 1], {(0, 5)})
        Traceback (most recent call last):
            ...
        ValueError: edge (0, 5) has an endpoint that is not in vertices
        >>> _prepare("ab", set())
        Traceback (most recent call last):
            ...
        TypeError: vertices must be an iterable collection, got str
    """
    if isinstance(vertices, (str, bytes)):
        raise TypeError(
            f"vertices must be an iterable collection, got {type(vertices).__name__}"
        )
    try:
        distinct = list(dict.fromkeys(vertices))
    except TypeError:
        raise TypeError(
            "vertices must be an iterable collection of hashable labels, "
            f"got {type(vertices).__name__}"
        ) from None
    edge_set = normalize_edges(edges)
    members = set(distinct)
    edge_list = _sorted_labels(edge_set, pairs=True)
    for u, v in edge_list:
        if u not in members or v not in members:
            raise ValueError(f"edge {(u, v)!r} has an endpoint that is not in vertices")
    return _sorted_labels(distinct), edge_list


def _solve_relaxation(order: List[Hashable], edge_list: List[Edge]) -> Tuple[float, Dict[Hashable, float]]:
    """Solve the vertex cover LP for an already validated, sorted graph.

    Args:
        order: Distinct vertices in sorted order (one LP variable each).
        edge_list: Normalized edges in sorted order (one constraint each).

    Returns:
        ``(lp_value, x)`` with ``x`` mapping every vertex to its value;
        ``(0.0, all zeros)`` without calling the solver when there are no
        edges.

    Raises:
        RuntimeError: If HiGHS reports a non-optimal status (it cannot for
            this always-feasible, bounded LP).

    Time Complexity:
        O(V E) to build the dense matrix plus one HiGHS solve.

    Space Complexity:
        O(V E).

    Examples:
        >>> _solve_relaxation([0, 1], [(0, 1)])[0]
        1.0
        >>> _solve_relaxation([0, 1], [])
        (0.0, {0: 0.0, 1: 0.0})
    """
    if not edge_list:
        return 0.0, {vertex: 0.0 for vertex in order}
    index = {vertex: i for i, vertex in enumerate(order)}
    n = len(order)
    A_ub: List[List[float]] = []
    for u, v in edge_list:
        row = [0.0] * n
        row[index[u]] = -1.0   # x_u + x_v >= 1 written as -x_u - x_v <= -1
        row[index[v]] = -1.0
        A_ub.append(row)
    b_ub = [-1.0] * len(edge_list)
    value, x = solve_lp([1.0] * n, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
    return value, {vertex: x[i] for i, vertex in enumerate(order)}


def vertex_cover_2approx(
    vertices: Iterable[Hashable], edges: Iterable[Any], seed: Optional[int] = None
) -> Set[Hashable]:
    """Return a vertex cover at most twice the minimum (greedy edge picking).

    Walk the edges; whenever an edge (u, v) has neither end in the cover,
    add BOTH ends. This is the Week 9 entry point; the walk itself is
    delegated to Week 8's
    :func:`src.complexity.vertex_cover_approx.vertex_cover_2approx` so both
    weeks run the same code. The picked edges form a matching M, every
    cover needs one endpoint per edge of M (OPT >= |M|), and the result has
    exactly 2|M| vertices, so it is at most 2 OPT (full proof in the module
    docstring). The bound holds for every visiting order.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
            Isolated vertices are allowed and never chosen.
        edges: The edges, 2-tuples written either way round.
        seed: ``None`` (default) walks the edges in sorted order, so the
            result is deterministic; an ``int`` shuffles that order with
            ``make_rng(seed)``.

    Returns:
        A new ``set`` covering every edge, of size ``2 |M| <= 2 OPT``.
        ``set()`` for an edgeless graph.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable,
            or ``seed`` is neither ``None`` nor an ``int``.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``.

    Time Complexity:
        O(V + E) for the walk, plus an O(E log E) sort that only makes the
        output reproducible.

    Space Complexity:
        O(V + E).

    Examples:
        >>> sorted(vertex_cover_2approx([0, 1, 2], {(0, 1), (0, 2), (1, 2)}))
        [0, 1]
        >>> sorted(vertex_cover_2approx([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}))
        [1, 2, 3, 4]
        >>> star = {(0, leaf) for leaf in range(1, 6)}
        >>> sorted(vertex_cover_2approx(range(6), star))
        [0, 1]
        >>> a = vertex_cover_2approx(range(6), star, seed=7)
        >>> a == vertex_cover_2approx(range(6), star, seed=7), len(a), 0 in a
        (True, 2, True)
        >>> vertex_cover_2approx([1, 2], set())
        set()
    """
    return _week8.vertex_cover_2approx(vertices, edges, seed=seed)


def matching_lower_bound(vertices: Iterable[Hashable], edges: Iterable[Any]) -> int:
    """Return |M| for the greedy maximal matching M: a certified lower bound on OPT.

    M is the set of edges :func:`vertex_cover_2approx` picks in the default
    sorted order. Its edges share no endpoints, and every vertex cover must
    contain an endpoint of each of them, so OPT >= |M|. The 2-approximation
    returns exactly 2|M| vertices, so this number is half the 2-approx
    cover size and certifies the ratio without knowing OPT. It is also at
    most the LP value (sum the constraints of the disjoint edges of M).

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
        edges: The edges, 2-tuples written either way round.

    Returns:
        ``len(M)``, an ``int``; ``0`` for an edgeless graph.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``.

    Time Complexity:
        O(V + E) for the walk, plus an O(E log E) sort for the fixed order.

    Space Complexity:
        O(V + E).

    Examples:
        >>> matching_lower_bound([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)})
        2
        >>> matching_lower_bound(range(6), {(0, leaf) for leaf in range(1, 6)})
        1
        >>> matching_lower_bound([0, 1], set())
        0
    """
    return len(_week8.greedy_maximal_matching(vertices, edges))


def vertex_cover_lp(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Tuple[float, Dict[Hashable, float]]:
    """Solve the LP relaxation of minimum vertex cover.

    Minimize sum_v x_v subject to x_u + x_v >= 1 for every edge (u, v) and
    0 <= x_v <= 1, with SciPy's HiGHS through
    :func:`src.utils.lp_solver.solve_lp`. Each edge constraint is written
    ``-x_u - x_v <= -1`` to fit ``A_ub``. The 0/1 vector of any cover is
    feasible, so the optimum LP is a lower bound on OPT; on K3 it is 1.5
    against OPT = 2 (integrality gap 4/3).

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
            One LP variable per distinct vertex, in sorted order; isolated
            vertices get x_v = 0.
        edges: The edges, 2-tuples written either way round.

    Returns:
        ``(lp_value, x)``: the LP optimum as a ``float`` and a ``dict``
        mapping every vertex to its ``float`` value in [0, 1]. With no
        edges the solver is not called and the result is
        ``(0.0, {v: 0.0 for v in vertices})``.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``.
        RuntimeError: If HiGHS reports a non-optimal status (it cannot for
            this LP, which is always feasible and bounded).

    Time Complexity:
        O(V log V + E log E) to sort, O(V E) to build the dense constraint
        matrix, plus one HiGHS solve (polynomial in practice).

    Space Complexity:
        O(V E) for the dense constraint matrix.

    Examples:
        >>> vertex_cover_lp([0, 1, 2], {(0, 1), (0, 2), (1, 2)})
        (1.5, {0: 0.5, 1: 0.5, 2: 0.5})
        >>> value, x = vertex_cover_lp(range(6), {(0, leaf) for leaf in range(1, 6)})
        >>> value, x[0], x[1]
        (1.0, 1.0, 0.0)
        >>> vertex_cover_lp([0, 1], set())
        (0.0, {0: 0.0, 1: 0.0})
    """
    order, edge_list = _prepare(vertices, edges)
    return _solve_relaxation(order, edge_list)


def vertex_cover_lp_rounding(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Tuple[Set[Hashable], float]:
    """Return a 2-approximate cover by rounding the LP relaxation at 1/2.

    Solve the LP of :func:`vertex_cover_lp` and keep every vertex with
    x_v >= 1/2 (``0.5 - 1e-9`` for floating point). Each edge constraint
    x_u + x_v >= 1 forces an endpoint to x >= 1/2, so the result is a
    cover, and since each kept vertex has 1 <= 2 x_v its size is at most
    2 LP <= 2 OPT. As a numerical safety net that never fires on exact
    solver output, an edge the threshold leaves uncovered gets its endpoint
    with the larger x_v (ties: the smaller vertex).

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
        edges: The edges, 2-tuples written either way round.

    Returns:
        ``(cover, lp_value)``: a new ``set`` covering every edge with
        ``len(cover) <= 2 * lp_value <= 2 * OPT``, and the LP optimum, a
        lower bound on OPT. ``(set(), 0.0)`` for an edgeless graph.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``.
        RuntimeError: If HiGHS reports a non-optimal status.

    Time Complexity:
        The cost of :func:`vertex_cover_lp` plus O(V + E) to round and
        check every edge.

    Space Complexity:
        O(V E) for the LP; O(V) for the cover.

    Examples:
        >>> cover, lp_value = vertex_cover_lp_rounding([0, 1, 2], {(0, 1), (0, 2), (1, 2)})
        >>> sorted(cover), lp_value
        ([0, 1, 2], 1.5)
        >>> vertex_cover_lp_rounding(range(6), {(0, leaf) for leaf in range(1, 6)})
        ({0}, 1.0)
        >>> vertex_cover_lp_rounding([1, 2, 3], set())
        (set(), 0.0)
    """
    order, edge_list = _prepare(vertices, edges)
    lp_value, x = _solve_relaxation(order, edge_list)
    cover = {vertex for vertex in order if x[vertex] >= 0.5 - _ROUND_TOL}
    for u, v in edge_list:  # safety net: only solver round-off can trigger it
        if u not in cover and v not in cover:
            cover.add(u if x[u] >= x[v] else v)
    return cover, lp_value


def _require_nonnegative(value: Any, name: str) -> float:
    """Return ``value`` as a ``float`` if it is a finite, non-negative real.

    Args:
        value: An objective value (cover size, cost, cut weight).
        name: The parameter name, for the error message.

    Returns:
        ``float(value)``.

    Raises:
        TypeError: If ``value`` is not a real number (``bool`` rejected).
        ValueError: If ``value`` is negative, NaN or infinite.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_nonnegative(3, "value")
        3.0
        >>> _require_nonnegative(-1, "value")
        Traceback (most recent call last):
            ...
        ValueError: value must be a finite non-negative number, got -1
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    as_float = float(value)
    if not math.isfinite(as_float) or as_float < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return as_float


def approximation_ratio(value: float, reference: float) -> float:
    """Return ``value / reference``, the measured approximation ratio.

    ``reference`` is usually OPT, but may be any bound: dividing by the LP
    value or the matching lower bound gives a *certified* upper bound on
    the true ratio without knowing OPT. For minimization the ratio to OPT
    is at least 1; for maximization (Max-Cut) it is at most 1. The ratio is
    undefined when ``reference`` is 0: an empty answer to an empty problem
    is exact (1.0), and any other value is reported as an error rather than
    as infinity.

    Args:
        value: The algorithm's objective value, a finite non-negative real.
        reference: The value to compare against, a finite non-negative real.

    Returns:
        ``value / reference`` as a ``float``; ``1.0`` when both are 0.

    Raises:
        TypeError: If either argument is not a real number (``bool``
            rejected).
        ValueError: If either argument is negative, NaN or infinite, or if
            ``reference`` is 0 while ``value`` is not.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> approximation_ratio(4, 2)
        2.0
        >>> approximation_ratio(3, 1.5)
        2.0
        >>> approximation_ratio(3, 4)
        0.75
        >>> approximation_ratio(0, 0)
        1.0
        >>> approximation_ratio(1, 0)
        Traceback (most recent call last):
            ...
        ValueError: reference is 0 but value is 1, so the ratio is undefined
        >>> approximation_ratio(-2, 1)
        Traceback (most recent call last):
            ...
        ValueError: value must be a finite non-negative number, got -2
    """
    numerator = _require_nonnegative(value, "value")
    denominator = _require_nonnegative(reference, "reference")
    if denominator == 0:
        if numerator == 0:
            return 1.0
        raise ValueError(f"reference is 0 but value is {value!r}, so the ratio is undefined")
    return numerator / denominator
