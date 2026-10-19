"""Small undirected-graph helpers for the Week 8 complexity reductions.

Week 8 moves between four NP-complete graph and logic problems (3-SAT,
Vertex Cover, Independent Set and Clique), and every one of them is stated
on a *simple undirected graph*. This module holds the few operations those
reductions, solvers, tests and benchmarks share, on the plainest possible
representation: a graph is a pair ``(vertices, edges)`` where ``vertices``
is any iterable of hashable labels and ``edges`` is a set of 2-tuples.

One canonical edge form
-----------------------
An undirected edge {u, v} can be written ``(u, v)`` or ``(v, u)``. If two
parts of the code wrote it differently, a set lookup would miss it and a
verifier would report a covered edge as uncovered. :func:`normalize_edges`
fixes one spelling: every pair is stored as ``(min(u, v), max(u, v))``,
duplicates and reversed duplicates collapse, and a self-loop or a malformed
edge raises ``ValueError`` (the problems here are defined on simple graphs,
where a self-loop has no meaning). Vertices of types that cannot be compared
with ``<`` fall back to ordering by ``(type name, repr)``, which is still
stable across processes.

Three verifiers, one identity
-----------------------------
:func:`is_vertex_cover`, :func:`is_independent_set` and :func:`is_clique`
are the polynomial-time *verifiers* for their problems: given a proposed
certificate they answer in O(V + E) or O(E + s^2) time, while *finding* an
optimal certificate is NP-hard. The three problems are tied together by two
facts the Week 8 reduction chain relies on. For a graph G = (V, E) and a
vertex set S:

* S is a vertex cover of G  iff  V - S is an independent set of G
  (an edge with both ends outside S is exactly an edge left uncovered by S);
* S is an independent set of G  iff  S is a clique of the complement graph
  (no edge of G inside S means every pair inside S is an edge of the
  complement, which :func:`complement` builds).

So a cover of size k, an independent set of size |V| - k and a clique of
size |V| - k in the complement are the same object read three ways.

=================== ========================== ==========================
Function            Time                       Space
=================== ========================== ==========================
normalize_edges     O(E)                       O(E)
random_graph        O(n^2)                     O(n + E)
complement          O(V^2 + E)                 O(V^2)
degrees             O(V + E)                   O(V + E)
is_vertex_cover     O(|cover| + E)             O(|cover|)
is_independent_set  O(|subset| + E)            O(|subset|)
is_clique           O(E + s^2)                 O(E + s)
=================== ========================== ==========================

Examples:
    >>> vertices, edges = [1, 2, 3, 4], {(1, 2), (3, 2), (3, 4)}
    >>> sorted(normalize_edges(edges))
    [(1, 2), (2, 3), (3, 4)]
    >>> is_vertex_cover(edges, {2, 3}), is_independent_set(edges, {1, 4})
    (True, True)
    >>> is_clique(complement(vertices, edges), {1, 4})
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numbers
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple

from src.utils.random_utils import make_rng

__all__ = [
    "complement",
    "degrees",
    "is_clique",
    "is_independent_set",
    "is_vertex_cover",
    "normalize_edges",
    "random_graph",
]

#: An undirected edge, stored as ``(min(u, v), max(u, v))``.
Edge = Tuple[Hashable, Hashable]


def _fallback_key(item: Any) -> Tuple[str, str]:
    """Return a process-stable sort key for a vertex that cannot use ``<``.

    Args:
        item: Any vertex label.

    Returns:
        ``(type name, repr)``, which orders mixed-type labels the same way in
        every process (unlike the built-in, per-process-salted ``hash()``).

    Time Complexity:
        O(L) for a ``repr`` of length L.

    Space Complexity:
        O(L).

    Examples:
        >>> _fallback_key(3)
        ('int', '3')
    """
    return (type(item).__name__, repr(item))


def _ordered_pair(u: Hashable, v: Hashable) -> Edge:
    """Return the pair ``(u, v)`` in canonical ``(min, max)`` order.

    Args:
        u: One endpoint.
        v: The other endpoint.

    Returns:
        ``(u, v)`` if ``u <= v``, else ``(v, u)``. Labels that cannot be
        compared (for example an ``int`` and a ``str``) are ordered by
        :func:`_fallback_key` instead.

    Time Complexity:
        O(1) for comparable labels.

    Space Complexity:
        O(1).

    Examples:
        >>> _ordered_pair(5, 2)
        (2, 5)
        >>> _ordered_pair("b", 1)
        (1, 'b')
    """
    try:
        return (u, v) if u <= v else (v, u)
    except TypeError:
        return (u, v) if _fallback_key(u) <= _fallback_key(v) else (v, u)


def _require_collection(value: Any, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a non-string iterable.

    Args:
        value: The candidate container.
        name: Its parameter name, for the error message.

    Raises:
        TypeError: If ``value`` is a ``str``/``bytes`` (an iterable of
            characters is never meant as a vertex or edge collection) or is
            not iterable at all.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_collection([1, 2], "vertices")
        >>> _require_collection("ab", "vertices")
        Traceback (most recent call last):
            ...
        TypeError: vertices must be an iterable collection, got str
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be an iterable collection, got {type(value).__name__}"
        )
    try:
        iter(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an iterable collection, got {type(value).__name__}"
        ) from None


def _as_pair(edge: Any) -> Edge:
    """Unpack one edge into its two endpoints, rejecting malformed edges.

    Args:
        edge: A 2-element iterable such as ``(u, v)``.

    Returns:
        ``(u, v)`` in the order given (not yet normalized).

    Raises:
        ValueError: If ``edge`` is not a 2-element non-string iterable, or if
            it is a self-loop ``(u, u)``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _as_pair([3, 1])
        (3, 1)
        >>> _as_pair((1, 2, 3))
        Traceback (most recent call last):
            ...
        ValueError: each edge must be a pair of vertices, got (1, 2, 3)
    """
    if isinstance(edge, (str, bytes)):
        raise ValueError(f"each edge must be a pair of vertices, got {edge!r}")
    try:
        items = tuple(edge)
    except TypeError:
        raise ValueError(
            f"each edge must be a pair of vertices, got {edge!r}"
        ) from None
    if len(items) != 2:
        raise ValueError(f"each edge must be a pair of vertices, got {edge!r}")
    u, v = items
    if u == v:
        raise ValueError(f"self-loop at vertex {u!r} is not allowed")
    return u, v


def _vertex_list(vertices: Iterable[Hashable], name: str = "vertices") -> List[Hashable]:
    """Return the distinct vertices of ``vertices`` in first-seen order.

    Args:
        vertices: Any non-string iterable of hashable labels.
        name: The parameter name, for error messages.

    Returns:
        A new list with duplicates removed.

    Raises:
        TypeError: If ``vertices`` is not a non-string iterable, or holds an
            unhashable label.

    Time Complexity:
        O(V).

    Space Complexity:
        O(V).

    Examples:
        >>> _vertex_list([3, 1, 3, 2])
        [3, 1, 2]
    """
    _require_collection(vertices, name)
    return list(dict.fromkeys(vertices))


def _check_endpoints(vertex_set: Set[Hashable], edges: Set[Edge]) -> None:
    """Raise ``ValueError`` if an edge has an endpoint outside the vertex set.

    Args:
        vertex_set: The graph's vertices.
        edges: Normalized edges.

    Raises:
        ValueError: If some edge touches a vertex that is not in
            ``vertex_set``.

    Time Complexity:
        O(E).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_endpoints({1, 2}, {(1, 2)})
        >>> _check_endpoints({1, 2}, {(1, 3)})
        Traceback (most recent call last):
            ...
        ValueError: edge (1, 3) has an endpoint that is not in vertices
    """
    for u, v in edges:
        if u not in vertex_set or v not in vertex_set:
            raise ValueError(f"edge {(u, v)!r} has an endpoint that is not in vertices")


def normalize_edges(edges: Iterable[Any]) -> Set[Edge]:
    """Return the edges as a set of canonical undirected pairs.

    Each edge is stored as ``(min(u, v), max(u, v))``, so ``(2, 1)`` and
    ``(1, 2)`` become the same element and duplicates collapse. The result
    is the one spelling every Week 8 module uses for set lookups, which is
    what makes verifiers and reductions agree on whether an edge exists.
    Labels that cannot be compared with ``<`` are ordered by
    ``(type name, repr)`` instead, which is the same in every process.

    Args:
        edges: A non-string iterable of 2-element edges, for example a set
            of tuples or a list of lists.

    Returns:
        A new ``set`` of ``(min, max)`` tuples.

    Raises:
        TypeError: If ``edges`` is not a non-string iterable, or an endpoint
            is unhashable.
        ValueError: If an edge does not have exactly two elements, or is a
            self-loop ``(u, u)``.

    Time Complexity:
        O(E).

    Space Complexity:
        O(E) for the returned set.

    Examples:
        >>> sorted(normalize_edges([(2, 1), (1, 2), [3, 2]]))
        [(1, 2), (2, 3)]
        >>> normalize_edges(set())
        set()
        >>> normalize_edges([(1, 1)])
        Traceback (most recent call last):
            ...
        ValueError: self-loop at vertex 1 is not allowed
        >>> normalize_edges([(1, 2, 3)])
        Traceback (most recent call last):
            ...
        ValueError: each edge must be a pair of vertices, got (1, 2, 3)
        >>> normalize_edges(5)
        Traceback (most recent call last):
            ...
        TypeError: edges must be an iterable collection, got int
    """
    _require_collection(edges, "edges")
    return {_ordered_pair(*_as_pair(edge)) for edge in edges}


def random_graph(
    n: int, p: float, seed: Optional[int]
) -> Tuple[List[int], Set[Tuple[int, int]]]:
    """Draw an Erdos-Renyi random graph G(n, p).

    Vertices are ``0 .. n-1``. The pairs ``(i, j)`` with ``i < j`` are
    visited in increasing lexicographic order and each is kept as an edge
    exactly when ``rng.random() < p``, with ``rng = make_rng(seed)``. Fixing
    the visiting order is what makes the graph a pure function of
    ``(n, p, seed)``. The expected edge count is ``p * n * (n - 1) / 2`` and
    the expected degree is ``p * (n - 1)``, so ``p = d / (n - 1)`` gives an
    average degree of about ``d``.

    Args:
        n: The number of vertices, a non-negative ``int``.
        p: The edge probability, a real number in ``[0, 1]``.
        seed: An ``int`` for a reproducible graph, or ``None`` for one seeded
            from the operating system (see :func:`make_rng`).

    Returns:
        ``(vertices, edges)``: the list ``[0, ..., n-1]`` and a set of
        normalized ``(i, j)`` pairs with ``i < j``.

    Raises:
        TypeError: If ``n`` is not an ``int``, ``p`` is not a real number
            (``bool`` is rejected for both), or ``seed`` is neither ``None``
            nor an ``int``.
        ValueError: If ``n`` is negative or ``p`` is outside ``[0, 1]``.

    Time Complexity:
        O(n^2) - one random draw per unordered pair.

    Space Complexity:
        O(n + E) for the returned graph.

    Examples:
        >>> vertices, edges = random_graph(4, 0.5, 42)
        >>> vertices, sorted(edges)
        ([0, 1, 2, 3], [(0, 2), (0, 3), (1, 2)])
        >>> random_graph(6, 0.3, 7) == random_graph(6, 0.3, 7)
        True
        >>> len(random_graph(5, 1.0, 1)[1]), len(random_graph(5, 0.0, 1)[1])
        (10, 0)
        >>> random_graph(3, 1.5, 0)
        Traceback (most recent call last):
            ...
        ValueError: p must be in [0, 1], got 1.5
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if isinstance(p, bool) or not isinstance(p, numbers.Real):
        raise TypeError(f"p must be a real number, got {type(p).__name__}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    rng = make_rng(seed)
    edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.add((i, j))
    return list(range(n)), edges


def complement(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Set[Edge]:
    """Return the edge set of the complement graph.

    The complement of G = (V, E) has the same vertices and an edge between
    every two distinct vertices that are *not* adjacent in G. It is the
    target of the Independent Set to Clique reduction: a set with no edges
    of G inside it has every pair joined in the complement.

    Args:
        vertices: The vertices of G, a non-string iterable of hashable
            labels (duplicates are ignored).
        edges: The edges of G, in any form :func:`normalize_edges` accepts.

    Returns:
        A new ``set`` of normalized pairs: every unordered pair of distinct
        vertices that is not an edge of G.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint that is not in ``vertices``.

    Time Complexity:
        O(V^2 + E) - every vertex pair is examined once.

    Space Complexity:
        O(V^2) - the complement of a sparse graph is dense.

    Examples:
        >>> sorted(complement([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}))
        [(1, 3), (1, 4), (2, 4)]
        >>> complement([0, 1, 2], {(0, 1), (0, 2), (1, 2)})
        set()
        >>> complement([0, 1], {(0, 5)})
        Traceback (most recent call last):
            ...
        ValueError: edge (0, 5) has an endpoint that is not in vertices
    """
    verts = _vertex_list(vertices)
    edge_set = normalize_edges(edges)
    _check_endpoints(set(verts), edge_set)
    missing: Set[Edge] = set()
    for i, u in enumerate(verts):
        for v in verts[i + 1:]:
            pair = _ordered_pair(u, v)
            if pair not in edge_set:
                missing.add(pair)
    return missing


def degrees(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Dict[Hashable, int]:
    """Return the degree of every vertex.

    Edges are normalized first, so a duplicate or reversed duplicate counts
    once. Isolated vertices are present with degree 0, so the result always
    has one key per distinct vertex and the degrees sum to ``2 * |E|``.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels
            (duplicates are ignored; key order follows first appearance).
        edges: The edges, in any form :func:`normalize_edges` accepts.

    Returns:
        A new ``dict`` mapping each vertex to its degree.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint that is not in ``vertices``.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E) - the result and the normalized edge set.

    Examples:
        >>> degrees([0, 1, 2, 3], {(0, 1), (0, 2), (2, 0)})
        {0: 2, 1: 1, 2: 1, 3: 0}
        >>> degrees([], set())
        {}
    """
    deg: Dict[Hashable, int] = dict.fromkeys(_vertex_list(vertices), 0)
    edge_set = normalize_edges(edges)
    _check_endpoints(set(deg), edge_set)
    for u, v in edge_set:
        deg[u] += 1
        deg[v] += 1
    return deg


def is_vertex_cover(edges: Iterable[Any], cover: Iterable[Hashable]) -> bool:
    """Check whether ``cover`` touches every edge (the Vertex Cover verifier).

    A vertex cover is a set of vertices containing at least one endpoint of
    every edge. Checking a proposed cover takes one pass over the edges,
    which is why Vertex Cover is in NP; finding a smallest cover is
    NP-hard. Every edge is read once (no early exit), so malformed input is
    always reported, and the running time does not depend on where the
    first uncovered edge sits.

    Args:
        edges: The edges, a non-string iterable of 2-element edges.
        cover: The proposed cover, a non-string iterable of vertices.
            Vertices that touch no edge are allowed and simply unused.

    Returns:
        ``True`` if every edge has an endpoint in ``cover``, else ``False``.
        An empty edge set is covered by any set, including the empty one.

    Raises:
        TypeError: If ``edges`` or ``cover`` is not a non-string iterable.
        ValueError: If an edge is malformed or is a self-loop.

    Time Complexity:
        O(|cover| + E) - build a set, then one membership test per endpoint.

    Space Complexity:
        O(|cover|).

    Examples:
        >>> path = {(1, 2), (2, 3), (3, 4)}
        >>> is_vertex_cover(path, {2, 3})
        True
        >>> is_vertex_cover(path, [1, 4])
        False
        >>> is_vertex_cover(set(), set())
        True
    """
    _require_collection(cover, "cover")
    _require_collection(edges, "edges")
    chosen = set(cover)
    covered = True
    for edge in edges:
        u, v = _as_pair(edge)
        if u not in chosen and v not in chosen:
            covered = False
    return covered


def is_independent_set(edges: Iterable[Any], subset: Iterable[Hashable]) -> bool:
    """Check that no edge has both endpoints in ``subset``.

    An independent set is a set of pairwise non-adjacent vertices. By the
    identity in the module docstring, ``subset`` is independent exactly when
    its complement in the vertex set is a vertex cover. Every edge is read
    once (no early exit), so malformed input is always reported.

    Args:
        edges: The edges, a non-string iterable of 2-element edges.
        subset: The proposed independent set, a non-string iterable of
            vertices.

    Returns:
        ``True`` if no edge lies inside ``subset``, else ``False``. The empty
        set and any single vertex are independent.

    Raises:
        TypeError: If ``edges`` or ``subset`` is not a non-string iterable.
        ValueError: If an edge is malformed or is a self-loop.

    Time Complexity:
        O(|subset| + E).

    Space Complexity:
        O(|subset|).

    Examples:
        >>> path = {(1, 2), (2, 3), (3, 4)}
        >>> is_independent_set(path, {1, 3})
        True
        >>> is_independent_set(path, {1, 2})
        False
        >>> is_independent_set(path, set())
        True
    """
    _require_collection(subset, "subset")
    _require_collection(edges, "edges")
    members = set(subset)
    independent = True
    for edge in edges:
        u, v = _as_pair(edge)
        if u in members and v in members:
            independent = False
    return independent


def is_clique(edges: Iterable[Any], subset: Iterable[Hashable]) -> bool:
    """Check that every two distinct vertices of ``subset`` are adjacent.

    A clique is a set of pairwise adjacent vertices. The edges are
    normalized into a set once, then each of the ``s * (s - 1) / 2`` pairs
    of the ``s`` distinct members is looked up.

    Args:
        edges: The edges, in any form :func:`normalize_edges` accepts.
        subset: The proposed clique, a non-string iterable of vertices
            (duplicates are ignored).

    Returns:
        ``True`` if every pair of distinct members is an edge, else
        ``False``. The empty set and any single vertex are cliques.

    Raises:
        TypeError: If ``edges`` or ``subset`` is not a non-string iterable.
        ValueError: If an edge is malformed or is a self-loop.

    Time Complexity:
        O(E + s^2) where s is the number of distinct members.

    Space Complexity:
        O(E + s) for the normalized edge set and the member list.

    Examples:
        >>> triangle = {(0, 1), (1, 2), (2, 0)}
        >>> is_clique(triangle, {0, 1, 2})
        True
        >>> is_clique({(0, 1), (1, 2)}, {0, 1, 2})
        False
        >>> is_clique(set(), [7])
        True
    """
    members = _vertex_list(subset, "subset")
    edge_set = normalize_edges(edges)
    for i, u in enumerate(members):
        for v in members[i + 1:]:
            if _ordered_pair(u, v) not in edge_set:
                return False
    return True
