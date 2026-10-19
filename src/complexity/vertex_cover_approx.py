"""Vertex Cover: a provable 2-approximation, a tempting greedy, and the exact answer.

A *vertex cover* of an undirected graph G = (V, E) is a set C of vertices
that touches every edge: for each edge (u, v), u is in C or v is in C.
Finding a smallest cover is NP-hard (Vertex Cover is one of Karp's 21
NP-complete problems, and Week 8's reduction from 3-SAT shows it directly),
while *checking* a proposed cover takes one pass over the edges. So unless
P = NP there is no polynomial-time algorithm that always returns a minimum
cover. This module asks the next question: how close to the minimum can a
polynomial-time algorithm be *guaranteed* to get?

The 2-approximation (greedy edge picking)
-----------------------------------------
:func:`vertex_cover_2approx` walks the edges once. Whenever it meets an edge
(u, v) with neither end in the cover yet, it adds BOTH u and v. The edges it
picks are returned by :func:`greedy_maximal_matching`.

* **It returns a cover.** An edge that was not picked was skipped because
  one of its ends was already in the cover, and a picked edge has both ends
  in the cover. Either way every edge is touched.
* **It is within a factor 2 of optimal.** The edges the algorithm picks
  share no endpoints (they form a matching M), any vertex cover must contain
  at least one endpoint of every edge in M, so OPT >= |M|, and the algorithm
  returns 2|M| <= 2 OPT.

  Spelled out: once (u, v) is picked, u and v are in the cover, so no later
  edge touching u or v can be picked; hence no two picked edges share an
  endpoint. Because the edges of M are disjoint, the endpoints they force
  into *any* cover are distinct, one per edge of M, which gives
  OPT >= |M|. The algorithm adds exactly two vertices per picked edge, so it
  returns 2|M| vertices, and 2|M| <= 2 OPT.

* **The factor 2 is tight.** On the path 1-2-3-4 the default (sorted)
  order picks (1, 2) and (3, 4) and returns all four vertices against an
  optimum of 2; on a star it returns the centre and one leaf against the
  centre alone; on K3,3 it returns all six vertices against one side of
  three. In each case the ratio is exactly 2.0, so no better constant can
  be proved for this algorithm.

The proof needs nothing about the order in which edges are visited, so the
guarantee holds for *every* order: the default sorted order (deterministic
output), an explicit ``order``, or a seeded shuffle. The best ratio known
for any polynomial-time algorithm is still 2 - o(1); Dinur and Safra (2005)
showed approximating within 1.36 is NP-hard, and under the Unique Games
Conjecture no polynomial-time algorithm achieves 2 - eps for any eps > 0
(Khot and Regev, 2008). This four-line algorithm is, as far as anyone
knows, essentially the best possible.

The tempting alternative (max-degree greedy)
--------------------------------------------
:func:`vertex_cover_max_degree_greedy` repeatedly takes the vertex that
covers the most still-uncovered edges. It *looks* smarter and often returns
a smaller cover than the 2-approximation on random graphs, but it has NO
constant-factor guarantee: it is greedy Set Cover, which is an
H(max degree) ~ ln n approximation, and there are graphs on which it is
Theta(log n) times optimal. The standard bad family is bipartite: a side L
of k vertices and, for each i = 2 .. k, floor(k / i) extra vertices each
joined to its own block of i vertices of L. Every extra vertex of block size
i has degree i while every L vertex has degree at most i - 1 at that stage,
so greedy takes all sum floor(k / i) ~ k ln k extra vertices, while L alone
(size k) is a cover. For k = 20 that is 46 vertices against an optimum of
20, a ratio of 2.3, already worse than the 2-approximation's worst case.
It is included here as the tempting comparison, not as a recommendation.

The exact solver
----------------
:data:`vertex_cover_exact` is :func:`src.complexity.reduction_demo.min_vertex_cover_exact`,
re-exported: branch and bound on an uncovered edge (u, v) (take u or take v)
with a matching lower bound, O(2^k (V + E)) for a minimum cover of size k.
It is exponential, as expected for an NP-hard problem, and is used to
measure the true approximation ratio on small graphs.

======================================= ======================= ==========
Function                                Time                    Space
======================================= ======================= ==========
greedy_maximal_matching                 O(V + E) given order;   O(V + E)
                                        + O(E log E) sort for
                                        default/seeded order
vertex_cover_2approx                    as above                O(V + E)
vertex_cover_max_degree_greedy          O((V + E) log V)        O(V + E)
vertex_cover_exact                      O(2^k (V + E))          O(V + E)
approximation_ratio                     O(1)                    O(1)
======================================= ======================= ==========

Representation: a graph is ``(vertices, edges)``, ``vertices`` any
non-string iterable of hashable labels and ``edges`` a set of 2-tuples; an
edge may be written either way round (see
:func:`src.utils.graph_utils.normalize_edges`).

Examples:
    >>> path = {(1, 2), (2, 3), (3, 4)}
    >>> sorted(vertex_cover_2approx([1, 2, 3, 4], path))
    [1, 2, 3, 4]
    >>> len(vertex_cover_exact([1, 2, 3, 4], path))
    2
    >>> approximation_ratio(4, 2)
    2.0
    >>> sorted(vertex_cover_max_degree_greedy([1, 2, 3, 4], path))
    [2, 3]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import heapq
import math
import numbers
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple

from src.complexity.reduction_demo import min_vertex_cover_exact
from src.utils.graph_utils import normalize_edges
from src.utils.random_utils import make_rng

__all__ = [
    "approximation_ratio",
    "greedy_maximal_matching",
    "vertex_cover_2approx",
    "vertex_cover_exact",
    "vertex_cover_max_degree_greedy",
]

#: An undirected edge in canonical ``(min, max)`` form.
Edge = Tuple[Hashable, Hashable]

#: The exact minimum vertex cover, re-exported from the reduction module so
#: the approximation code and the reduction share one exact solver.
vertex_cover_exact = min_vertex_cover_exact


def _fallback_key(item: Any) -> Tuple[str, str]:
    """Return a process-stable sort key for a label that cannot use ``<``.

    Args:
        item: Any vertex label.

    Returns:
        ``(type name, repr)``. Unlike the salted built-in ``hash()``, it is
        the same in every process, so sorting by it is reproducible.

    Time Complexity:
        O(L) for a ``repr`` of length L.

    Space Complexity:
        O(L).

    Examples:
        >>> _fallback_key("x1")
        ('str', "'x1'")
    """
    return (type(item).__name__, repr(item))


def _distinct_vertices(vertices: Iterable[Hashable]) -> List[Hashable]:
    """Return the distinct vertices in first-seen order.

    Args:
        vertices: A non-string iterable of hashable labels.

    Returns:
        A new list with duplicates removed.

    Raises:
        TypeError: If ``vertices`` is a ``str``/``bytes``, is not iterable,
            or holds an unhashable label.

    Time Complexity:
        O(V).

    Space Complexity:
        O(V).

    Examples:
        >>> _distinct_vertices([3, 1, 3, 2])
        [3, 1, 2]
        >>> _distinct_vertices("abc")
        Traceback (most recent call last):
            ...
        TypeError: vertices must be an iterable collection, got str
    """
    if isinstance(vertices, (str, bytes)):
        raise TypeError(
            f"vertices must be an iterable collection, got {type(vertices).__name__}"
        )
    try:
        return list(dict.fromkeys(vertices))
    except TypeError:
        raise TypeError(
            "vertices must be an iterable collection of hashable labels, "
            f"got {type(vertices).__name__}"
        ) from None


def _canonical_order(distinct: List[Hashable]) -> List[Hashable]:
    """Return distinct vertices sorted, so "smallest vertex" is defined.

    Args:
        distinct: Distinct hashable labels (from :func:`_distinct_vertices`).

    Returns:
        A new list in ascending order; labels that cannot be compared with
        ``<`` (for example ``int`` and ``str`` together) are sorted by
        :func:`_fallback_key` instead, which is the same in every process.

    Time Complexity:
        O(V log V).

    Space Complexity:
        O(V).

    Examples:
        >>> _canonical_order([3, 1, 2])
        [1, 2, 3]
        >>> _canonical_order([2, "a", 1])
        [1, 2, 'a']
    """
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=_fallback_key)


def _graph(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Tuple[List[Hashable], Set[Edge]]:
    """Validate a graph and return its distinct vertices and normalized edges.

    Args:
        vertices: A non-string iterable of hashable labels.
        edges: The edges, in any form :func:`normalize_edges` accepts.

    Returns:
        ``(distinct, edge_set)``: the distinct vertices in first-seen order
        and the edges as ``(min, max)`` pairs. Nothing is sorted, so the
        validation alone stays O(V + E).

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint that is not in ``vertices``.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> distinct, edge_set = _graph([2, 1, 3], {(2, 1), (3, 2)})
        >>> distinct, sorted(edge_set)
        ([2, 1, 3], [(1, 2), (2, 3)])
        >>> _graph([1, 2], {(1, 3)})
        Traceback (most recent call last):
            ...
        ValueError: edge (1, 3) has an endpoint that is not in vertices
    """
    distinct = _distinct_vertices(vertices)
    edge_set = normalize_edges(edges)
    members = set(distinct)
    for u, v in edge_set:
        if u not in members or v not in members:
            raise ValueError(f"edge {(u, v)!r} has an endpoint that is not in vertices")
    return distinct, edge_set


def _sorted_edges(edge_set: Set[Edge]) -> List[Edge]:
    """Return normalized edges in ascending lexicographic order.

    Sorting fixes the visiting order, which ``set`` iteration does not (for
    ``str`` labels it changes between processes), so the default result of
    the 2-approximation is the same on every run.

    Args:
        edge_set: Normalized ``(min, max)`` edges.

    Returns:
        A new sorted list. Pairs whose labels cannot be compared with ``<``
        are sorted by :func:`_fallback_key` on each endpoint instead.

    Time Complexity:
        O(E log E).

    Space Complexity:
        O(E).

    Examples:
        >>> _sorted_edges({(2, 3), (0, 5), (0, 1)})
        [(0, 1), (0, 5), (2, 3)]
    """
    try:
        return sorted(edge_set)
    except TypeError:
        return sorted(edge_set, key=lambda e: (_fallback_key(e[0]), _fallback_key(e[1])))


def _visiting_order(
    edge_set: Set[Edge], order: Optional[Iterable[Any]], seed: Optional[int]
) -> List[Edge]:
    """Return the edge sequence the greedy walk follows.

    Args:
        edge_set: The graph's normalized edges.
        order: An explicit edge sequence, or ``None``. Each entry may be
            written either way round; every entry must be an edge of the
            graph and every edge must appear at least once (a repeat is
            harmless: the second visit finds the edge already covered).
        seed: Shapes the order only when ``order`` is ``None``: an ``int``
            shuffles the sorted edge list with ``make_rng(seed)``; ``None``
            keeps it sorted. It is type-checked even when ``order`` is
            given, so a bad seed is never silently ignored.

    Returns:
        A list of normalized edges.

    Raises:
        TypeError: If ``order`` is a ``str``/``bytes`` or not iterable, or
            ``seed`` is neither ``None`` nor an ``int`` (checked whether or
            not ``order`` is given).
        ValueError: If an entry of ``order`` is malformed or is not an edge
            of the graph, or if ``order`` leaves out an edge.

    Time Complexity:
        O(E log E) for the sorted or shuffled order; O(len(order) + E) for
        an explicit order.

    Space Complexity:
        O(E + len(order)).

    Examples:
        >>> _visiting_order({(1, 2), (2, 3)}, None, None)
        [(1, 2), (2, 3)]
        >>> _visiting_order({(1, 2), (2, 3)}, [(3, 2), (2, 1)], None)
        [(2, 3), (1, 2)]
        >>> _visiting_order({(1, 2), (2, 3)}, [(2, 3)], None)
        Traceback (most recent call last):
            ...
        ValueError: order leaves out 1 edge(s) of the graph, e.g. (1, 2)
        >>> _visiting_order({(1, 2)}, [(1, 2)], "7")
        Traceback (most recent call last):
            ...
        TypeError: seed must be an int, got str
    """
    rng = None if seed is None else make_rng(seed)   # validates seed in every case
    if order is None:
        sequence = _sorted_edges(edge_set)
        if rng is not None:
            rng.shuffle(sequence)
        return sequence
    if isinstance(order, (str, bytes)):
        raise TypeError(f"order must be an iterable of edges, got {type(order).__name__}")
    try:
        entries = list(order)
    except TypeError:
        raise TypeError(
            f"order must be an iterable of edges, got {type(order).__name__}"
        ) from None
    sequence = []
    for entry in entries:
        (edge,) = normalize_edges([entry])
        if edge not in edge_set:
            raise ValueError(f"order lists {entry!r}, which is not an edge of the graph")
        sequence.append(edge)
    missing = edge_set.difference(sequence)
    if missing:
        example = _sorted_edges(missing)[0]
        raise ValueError(
            f"order leaves out {len(missing)} edge(s) of the graph, e.g. {example!r}"
        )
    return sequence


def greedy_maximal_matching(
    vertices: Iterable[Hashable],
    edges: Iterable[Any],
    order: Optional[Iterable[Any]] = None,
    seed: Optional[int] = None,
) -> List[Edge]:
    """Return the edges the 2-approximation picks: a greedy maximal matching.

    The edges are visited in one pass; an edge is picked exactly when
    neither of its endpoints is an endpoint of an edge picked earlier. The
    picked edges therefore share no endpoints (a *matching* M), and every
    edge of the graph shares an endpoint with some picked edge (M is
    *maximal*: it cannot be extended). These two facts are the whole proof
    of :func:`vertex_cover_2approx`: maximality makes the endpoints of M a
    cover, and disjointness gives OPT >= |M|.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
        edges: The edges, in any form
            :func:`src.utils.graph_utils.normalize_edges` accepts.
        order: Optional explicit visiting order: an iterable of edges of the
            graph (either orientation) that lists every edge at least once.
            When given, ``seed`` does not affect the order (it is still
            type-checked).
        seed: Optional ``int``. When ``order`` is ``None``, the sorted edge
            list is shuffled with ``make_rng(seed)`` before the walk. With
            neither ``order`` nor ``seed`` the edges are visited in sorted
            order, so the result is deterministic.

    Returns:
        The picked edges as ``(min, max)`` tuples, in the order they were
        picked. ``[]`` for an edgeless graph.

    Raises:
        TypeError: If ``vertices``, ``edges`` or ``order`` is not a
            non-string iterable, or ``seed`` is neither ``None`` nor an
            ``int``.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``; or if ``order`` lists a non-edge
            or leaves out an edge.

    Time Complexity:
        O(V + E + len(order)) with an explicit ``order``: one validation
        pass and one membership test per visited edge. The default and
        seeded orders add an O(E log E) sort of the edges, there only to
        make the output reproducible.

    Space Complexity:
        O(V + E).

    Examples:
        >>> greedy_maximal_matching([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)})
        [(1, 2), (3, 4)]
        >>> greedy_maximal_matching([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)},
        ...                         order=[(2, 3), (1, 2), (3, 4)])
        [(2, 3)]
        >>> k33 = {(a, b) for a in range(3) for b in range(3, 6)}
        >>> greedy_maximal_matching(range(6), k33)
        [(0, 3), (1, 4), (2, 5)]
        >>> m = greedy_maximal_matching(range(6), k33, seed=42)
        >>> m == greedy_maximal_matching(range(6), k33, seed=42), len(m)
        (True, 3)
        >>> greedy_maximal_matching([0, 1], set())
        []
    """
    _, edge_set = _graph(vertices, edges)
    matched: Set[Hashable] = set()
    picked: List[Edge] = []
    for u, v in _visiting_order(edge_set, order, seed):
        if u not in matched and v not in matched:
            matched.add(u)
            matched.add(v)
            picked.append((u, v))
    return picked


def vertex_cover_2approx(
    vertices: Iterable[Hashable],
    edges: Iterable[Any],
    order: Optional[Iterable[Any]] = None,
    seed: Optional[int] = None,
) -> Set[Hashable]:
    """Return a vertex cover at most twice the minimum (greedy edge picking).

    Walk the edges; whenever an edge (u, v) has neither end in the cover,
    add BOTH ends. Every edge ends up covered: a picked edge has both ends
    in the cover, and a skipped edge was skipped because one end already
    was.

    Why the factor 2 holds: the edges the algorithm picks share no
    endpoints (they form a matching M), any vertex cover must contain at
    least one endpoint of every edge in M, so OPT >= |M|, and the algorithm
    returns 2|M| <= 2 OPT. The argument never uses the visiting order, so
    the bound holds for the default, explicit and seeded orders alike, and
    it is tight: the path 1-2-3-4, a star and K3,3 all reach exactly 2.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
            Isolated vertices are allowed and never chosen.
        edges: The edges, in any form
            :func:`src.utils.graph_utils.normalize_edges` accepts.
        order: Optional explicit visiting order: an iterable of edges of the
            graph (either orientation) listing every edge at least once.
            When given, ``seed`` does not affect the order (it is still
            type-checked).
        seed: Optional ``int``. When ``order`` is ``None``, the sorted edge
            list is shuffled with ``make_rng(seed)``; with neither argument
            the sorted order is used, so the default result is
            deterministic.

    Returns:
        A new ``set`` of vertices covering every edge, of even size
        ``2 |M|`` and at most twice the minimum. ``set()`` for an edgeless
        graph.

    Raises:
        TypeError: If ``vertices``, ``edges`` or ``order`` is not a
            non-string iterable, or ``seed`` is neither ``None`` nor an
            ``int``.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``; or if ``order`` lists a non-edge
            or leaves out an edge.

    Time Complexity:
        O(V + E) for the greedy walk (one membership test per edge, two
        insertions per picked edge). The default and seeded orders first
        sort the edges, O(E log E), purely for reproducible output.

    Space Complexity:
        O(V + E) for the normalized edges and the cover.

    Examples:
        >>> sorted(vertex_cover_2approx([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}))
        [1, 2, 3, 4]
        >>> star = {(0, leaf) for leaf in range(1, 6)}
        >>> sorted(vertex_cover_2approx(range(6), star))
        [0, 1]
        >>> sorted(vertex_cover_2approx([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)},
        ...                             order=[(3, 2), (2, 1), (4, 3)]))
        [2, 3]
        >>> k33 = {(a, b) for a in range(3) for b in range(3, 6)}
        >>> len(vertex_cover_2approx(range(6), k33, seed=7))
        6
        >>> vertex_cover_2approx(range(6), star, seed=7) == vertex_cover_2approx(range(6), star, seed=7)
        True
        >>> vertex_cover_2approx([1, 2, 3], set())
        set()
    """
    cover: Set[Hashable] = set()
    for u, v in greedy_maximal_matching(vertices, edges, order=order, seed=seed):
        cover.add(u)
        cover.add(v)
    return cover


def vertex_cover_max_degree_greedy(
    vertices: Iterable[Hashable], edges: Iterable[Any]
) -> Set[Hashable]:
    """Return a cover built by repeatedly taking the highest-degree vertex.

    While any edge is uncovered, take the vertex with the most uncovered
    edges (ties broken by the smallest vertex), put it in the cover and
    delete its edges. This is the tempting alternative to
    :func:`vertex_cover_2approx`: it often finds smaller covers on random
    graphs, but it has NO constant-factor guarantee. It is greedy Set Cover
    in disguise, an H(max degree) ~ ln n approximation, and on the
    bipartite family described in the module docstring it is Theta(log n)
    times optimal (for k = 20 it returns 46 vertices where 20 suffice, a
    ratio of 2.3, worse than the 2-approximation can ever be). It is
    included as the tempting comparison, not as a recommendation.

    The degrees live in a heap keyed by ``(-degree, rank)``, where ``rank``
    is the vertex's position in sorted order. Degrees only fall, so an
    entry whose degree is out of date is skipped when popped (lazy
    deletion) instead of being found and removed.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
            Isolated vertices are never chosen.
        edges: The edges, in any form
            :func:`src.utils.graph_utils.normalize_edges` accepts.

    Returns:
        A new ``set`` of vertices covering every edge. ``set()`` for an
        edgeless graph.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string
            iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint outside ``vertices``.

    Time Complexity:
        O((V + E) log V): V initial heap entries plus at most one push per
        deleted edge, each O(log(V + E)) = O(log V).

    Space Complexity:
        O(V + E) for the adjacency sets and the heap.

    Examples:
        >>> sorted(vertex_cover_max_degree_greedy([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}))
        [2, 3]
        >>> k33 = {(a, b) for a in range(3) for b in range(3, 6)}
        >>> sorted(vertex_cover_max_degree_greedy(range(6), k33))
        [0, 1, 2]

        Not optimal: root 0 with children 1, 2, 3, each with two leaves.
        All four inner vertices start at degree 3, the tie goes to 0, and
        then 1, 2 and 3 are still needed; {1, 2, 3} alone is a cover.

        >>> tree = {(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (3, 9)}
        >>> sorted(vertex_cover_max_degree_greedy(range(10), tree))
        [0, 1, 2, 3]
        >>> vertex_cover_max_degree_greedy([1, 2], set())
        set()
    """
    distinct, edge_set = _graph(vertices, edges)
    order = _canonical_order(distinct)
    rank: Dict[Hashable, int] = {vertex: i for i, vertex in enumerate(order)}
    adjacency: Dict[Hashable, Set[Hashable]] = {vertex: set() for vertex in order}
    for u, v in edge_set:
        adjacency[u].add(v)
        adjacency[v].add(u)
    degree: Dict[Hashable, int] = {vertex: len(adjacency[vertex]) for vertex in order}

    heap: List[Tuple[int, int]] = [(-degree[v], rank[v]) for v in order if degree[v] > 0]
    heapq.heapify(heap)
    cover: Set[Hashable] = set()
    while heap:
        neg_degree, r = heapq.heappop(heap)
        vertex = order[r]
        if -neg_degree != degree[vertex]:
            continue  # stale entry: the degree has fallen since it was pushed
        cover.add(vertex)
        for neighbour in adjacency[vertex]:
            adjacency[neighbour].discard(vertex)
            degree[neighbour] -= 1
            if degree[neighbour] > 0:
                heapq.heappush(heap, (-degree[neighbour], rank[neighbour]))
        adjacency[vertex].clear()
        degree[vertex] = 0
    return cover


def _require_size(value: Any, name: str) -> float:
    """Return ``value`` as a ``float`` if it is a finite, non-negative real.

    Args:
        value: A cover size or tour cost.
        name: Its parameter name, for the error message.

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
        >>> _require_size(3, "approx_size")
        3.0
        >>> _require_size(-1, "approx_size")
        Traceback (most recent call last):
            ...
        ValueError: approx_size must be a finite non-negative number, got -1
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    as_float = float(value)
    if not math.isfinite(as_float) or as_float < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return as_float


def approximation_ratio(approx_size: float, optimal_size: float) -> float:
    """Return ``approx_size / optimal_size``, the measured approximation ratio.

    For a minimization problem the ratio is at least 1, and an algorithm
    with a proven factor c (2 for :func:`vertex_cover_2approx`) never
    exceeds c. The ratio is undefined when the optimum is 0; the only
    sensible reading there is that an empty answer to an empty problem is
    exact (ratio 1.0), and any non-empty answer has no finite ratio, which
    is reported as an error rather than as ``inf``. The function does not
    insist on ``approx_size >= optimal_size``: it only divides, so it also
    serves tour costs and other non-negative objective values.

    Args:
        approx_size: The size (or cost) of the approximate solution, a
            finite non-negative real number.
        optimal_size: The size (or cost) of an optimal solution, a finite
            non-negative real number.

    Returns:
        ``approx_size / optimal_size`` as a ``float``; ``1.0`` when both are
        0.

    Raises:
        TypeError: If either argument is not a real number (``bool``
            rejected).
        ValueError: If either argument is negative, NaN or infinite, or if
            ``optimal_size`` is 0 while ``approx_size`` is not.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> approximation_ratio(4, 2)
        2.0
        >>> approximation_ratio(3, 2)
        1.5
        >>> approximation_ratio(0, 0)
        1.0
        >>> approximation_ratio(1, 0)
        Traceback (most recent call last):
            ...
        ValueError: optimal_size is 0 but approx_size is 1, so the ratio is undefined
        >>> approximation_ratio(2, -1)
        Traceback (most recent call last):
            ...
        ValueError: optimal_size must be a finite non-negative number, got -1
    """
    approx = _require_size(approx_size, "approx_size")
    optimal = _require_size(optimal_size, "optimal_size")
    if optimal == 0:
        if approx == 0:
            return 1.0
        raise ValueError(
            f"optimal_size is 0 but approx_size is {approx_size!r}, so the ratio is undefined"
        )
    return approx / optimal
