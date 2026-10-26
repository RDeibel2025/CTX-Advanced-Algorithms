"""Max-Cut: a coin flip per vertex, local search, and the exact answer.

A *cut* of an undirected graph G = (V, E) splits the vertices into two
sides, written here as a map ``side: vertex -> 0 or 1``. An edge (u, v) is
*cut* when its ends are on different sides, and the *value* of the cut is
the number of cut edges. Max-Cut asks for a split that cuts as many edges
as possible. Its decision version is one of Karp's 21 NP-complete problems,
so (unless P = NP) no polynomial-time algorithm always finds the maximum.
The surprise is how little effort a factor of 1/2 costs: a fair coin per
vertex achieves it in expectation, and a few greedy flips achieve it for
certain.

Random cut: E[cut] = |E|/2 >= OPT/2
-----------------------------------
:func:`maxcut_random` puts each vertex on side 1 with probability 1/2,
independently of every other vertex.

* **Each edge is cut with probability exactly 1/2.** Fix an edge (u, v).
  Whatever side u lands on, v lands on the *other* side with probability
  1/2, because v's coin is independent of u's. So P[(u, v) is cut] = 1/2.
* **Linearity of expectation.** Let X_e be 1 when edge e is cut and 0
  otherwise, so the cut value is X = sum over e of X_e. Expectation is
  linear whether or not the X_e are independent (they are not: on a
  triangle at most two of the three edges can be cut), so
  E[X] = sum over e of E[X_e] = |E| * 1/2 = |E|/2.
* **OPT <= |E|.** No cut contains more edges than the graph has.

Together: E[X] = |E|/2 >= OPT/2, an expected 1/2-approximation that never
looks at the edges. The same computation is also an existence proof (the
probabilistic method): a random variable is at least its mean with
positive probability, so *every* graph has a cut of at least |E|/2 edges.
The expectation can be turned into a probability: Y = |E| - X is
non-negative with E[Y] = |E|/2, so by Markov's inequality
P[X <= (1/2 - eps)|E|] = P[Y >= (1/2 + eps)|E|] <= 1/(1 + 2 eps), and the
best of k independent runs misses (1/2 - eps)|E| with probability at most
(1/(1 + 2 eps))^k.

Local search: every local optimum cuts at least |E|/2 edges
----------------------------------------------------------
:func:`maxcut_local_search` starts from an assignment and flips one vertex
at a time while some flip strictly increases the cut. For a vertex v let
c(v) be its number of cut incident edges and s(v) = deg(v) - c(v) its
number of uncut ones. Flipping v turns its s(v) uncut edges into cut edges
and its c(v) cut edges into uncut ones, and changes no other edge, so the
cut rises by gain(v) = s(v) - c(v).

* **The guarantee.** At a local optimum no flip helps, so for every
  vertex s(v) <= c(v), that is c(v) >= deg(v)/2: every vertex has at least
  half of its incident edges cut. Sum over all vertices. Each cut edge is
  counted once at each of its two ends, so sum c(v) = 2 * cut, and
  sum deg(v) = 2|E| (the handshake lemma). Hence 2 * cut >= |E|, i.e.
  cut >= |E|/2 >= OPT/2. Unlike the random cut this holds for every run,
  not just on average, and whatever the starting assignment.
* **Termination.** gain(v) is an integer, so a flip that strictly
  increases the cut increases it by at least 1. The cut is an integer
  between 0 and |E|, so from a starting cut of c0 there are at most
  |E| - c0 <= |E| flips. (With edge *weights* this counting argument fails,
  and weighted local Max-Cut is PLS-complete; the unweighted bound is what
  makes this heuristic polynomial.)

The implementation keeps, for every vertex, the number of neighbours on its
own side (s(v)), so a gain is read in O(1) and a flip of v updates only v
and its neighbours in O(deg v). The scan visits vertices in increasing
order and flips the *first* vertex with a positive gain, then restarts the
scan from the smallest vertex, which makes the result a deterministic
function of the start.

Beyond 1/2
----------
Both methods guarantee 1/2. Goemans and Williamson (1995) round a
semidefinite relaxation with a random hyperplane and reach 0.878; Hastad
(2001) proved approximating Max-Cut within 16/17 is NP-hard, and under the
Unique Games Conjecture 0.878 is the best possible (Khot, Kindler, Mossel
and O'Donnell, 2007).

The exact solver
----------------
:func:`maxcut_exact` tries every split. Swapping the two sides gives the
same cut, so the smallest vertex is fixed on side 0 and only
2^(n-1) assignments are checked. That is exponential, so it is limited to
n <= :data:`MAX_EXACT_VERTICES` = 16 and used only to measure true ratios.

Hand-checked anchors (used by the tests):

* K4: OPT = 4 of 6. A 2-2 split cuts the 4 edges between the pairs; a 1-3
  split cuts only 3, and 0-4 cuts none.
* The 5-cycle: OPT = 4 of 5. An odd cycle cannot be 2-coloured, so at least
  one edge is uncut, and alternating sides around the cycle leaves exactly
  one.
* K3,3: OPT = 9 of 9. The graph is bipartite, so putting its two parts on
  opposite sides cuts every edge.

==================== ================================ =========
Function             Time                             Space
==================== ================================ =========
cut_value            O(E)                             O(E)
maxcut_random        O(V log V + E)                   O(V + E)
maxcut_local_search  O(V log V + E + F (V + Delta))   O(V + E)
                     for F <= E flips, max degree
                     Delta
maxcut_exact         O(2^(n-1) n + E)                 O(V + E)
==================== ================================ =========

Representation: a graph is ``(vertices, edges)``, ``vertices`` any
non-string iterable of hashable labels and ``edges`` 2-tuples, either way
round (see :func:`src.utils.graph_utils.normalize_edges`; duplicate edges
collapse and self-loops are rejected). Randomness comes only from
:func:`src.utils.random_utils.make_rng`.

Examples:
    >>> k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    >>> maxcut_exact(range(4), k4)
    (4, {0: 0, 1: 1, 2: 1, 3: 0})
    >>> maxcut_random(range(4), k4, seed=42)
    (3, {0: 0, 1: 1, 2: 1, 3: 1})
    >>> maxcut_local_search(range(4), k4, seed=42)
    (4, {0: 0, 1: 0, 2: 1, 3: 1}, 1)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Set, Tuple

from src.utils.graph_utils import normalize_edges
from src.utils.random_utils import make_rng

__all__ = [
    "MAX_EXACT_VERTICES",
    "cut_value",
    "maxcut_exact",
    "maxcut_local_search",
    "maxcut_random",
]

#: The largest graph :func:`maxcut_exact` accepts (2^15 = 32,768 splits).
MAX_EXACT_VERTICES: int = 16

#: An undirected edge in canonical ``(min, max)`` form.
Edge = Tuple[Hashable, Hashable]


def _fallback_key(item: Any) -> Tuple[str, str]:
    """Return a process-stable sort key for a label that cannot use ``<``.

    Args:
        item: Any vertex label.

    Returns:
        ``(type name, repr)``, which is the same in every process (unlike the
        salted built-in ``hash()``).

    Time Complexity:
        O(L) for a ``repr`` of length L.

    Space Complexity:
        O(L).

    Examples:
        >>> _fallback_key("a")
        ('str', "'a'")
    """
    return (type(item).__name__, repr(item))


def _sorted_vertices(vertices: Iterable[Hashable]) -> List[Hashable]:
    """Return the distinct vertices in increasing order.

    Args:
        vertices: A non-string iterable of hashable labels.

    Returns:
        A new sorted list without duplicates. Labels that cannot be compared
        with ``<`` are sorted by :func:`_fallback_key` instead.

    Raises:
        TypeError: If ``vertices`` is a ``str``/``bytes``, is not iterable,
            or holds an unhashable label.

    Time Complexity:
        O(V log V).

    Space Complexity:
        O(V).

    Examples:
        >>> _sorted_vertices([3, 1, 3, 2])
        [1, 2, 3]
        >>> _sorted_vertices([2, "a", 1])
        [1, 2, 'a']
        >>> _sorted_vertices("abc")
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
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=_fallback_key)


def _graph(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Tuple[List[Hashable], Set[Edge]]:
    """Validate a graph and return its sorted vertices and normalized edges.

    Args:
        vertices: A non-string iterable of hashable labels.
        edges: The edges, in any form :func:`normalize_edges` accepts.

    Returns:
        ``(order, edge_set)``: the distinct vertices in increasing order and
        the edges as ``(min, max)`` pairs.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint that is not in ``vertices``.

    Time Complexity:
        O(V log V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> order, edge_set = _graph([2, 1, 3], [(2, 1), (3, 2)])
        >>> order, sorted(edge_set)
        ([1, 2, 3], [(1, 2), (2, 3)])
        >>> _graph([1, 2], [(1, 3)])
        Traceback (most recent call last):
            ...
        ValueError: edge (1, 3) has an endpoint that is not in vertices
    """
    order = _sorted_vertices(vertices)
    edge_set = normalize_edges(edges)
    members = set(order)
    for u, v in edge_set:
        if u not in members or v not in members:
            raise ValueError(f"edge {(u, v)!r} has an endpoint that is not in vertices")
    return order, edge_set


def _check_side_value(vertex: Hashable, value: Any) -> int:
    """Return a side label after checking that it is the ``int`` 0 or 1.

    Args:
        vertex: The vertex the label belongs to, for the error message.
        value: The proposed side.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ValueError: If ``value`` is not the ``int`` 0 or 1 (``bool`` and
            ``float`` are rejected).

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_side_value("a", 1)
        1
        >>> _check_side_value("a", 2)
        Traceback (most recent call last):
            ...
        ValueError: side of vertex 'a' must be 0 or 1, got 2
    """
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"side of vertex {vertex!r} must be 0 or 1, got {value!r}")
    return int(value)


def _adjacency(order: List[Hashable], edge_set: Set[Edge]) -> List[List[int]]:
    """Return adjacency lists over vertex indices (positions in ``order``).

    Args:
        order: The sorted distinct vertices.
        edge_set: Normalized edges whose endpoints are all in ``order``.

    Returns:
        ``adj`` with ``adj[i]`` the indices of the neighbours of ``order[i]``.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> _adjacency(["a", "b", "c"], {("a", "b"), ("b", "c")})[1]
        [0, 2]
    """
    index = {v: i for i, v in enumerate(order)}
    adj: List[List[int]] = [[] for _ in order]
    for u, v in edge_set:
        iu, iv = index[u], index[v]
        adj[iu].append(iv)
        adj[iv].append(iu)
    for neighbours in adj:
        neighbours.sort()
    return adj


def cut_value(edges: Iterable[Any], side: Mapping[Hashable, int]) -> int:
    """Return the number of edges whose endpoints are on different sides.

    This is the objective of Max-Cut and the independent check the tests use
    on every side map an algorithm returns.

    Args:
        edges: The edges, in any form
            :func:`src.utils.graph_utils.normalize_edges` accepts (duplicates
            collapse, so each undirected edge counts once).
        side: A mapping from vertex to 0 or 1. It must give a side to every
            endpoint of every edge; vertices with no edges may be present.

    Returns:
        The cut value, an ``int`` between 0 and the number of edges.

    Raises:
        TypeError: If ``side`` is not a mapping, or ``edges`` is not a
            non-string iterable.
        ValueError: If an edge is malformed or a self-loop, an endpoint has
            no side, or a side is not the ``int`` 0 or 1.

    Time Complexity:
        O(V + E) for V entries in ``side`` and E edges.

    Space Complexity:
        O(E) for the normalized edge set.

    Examples:
        >>> cut_value([(0, 1), (1, 2), (0, 2)], {0: 0, 1: 1, 2: 1})
        2
        >>> cut_value([], {})
        0
        >>> cut_value([(0, 1)], {0: 0})
        Traceback (most recent call last):
            ...
        ValueError: edge (0, 1) has an endpoint with no side
    """
    if not isinstance(side, Mapping):
        raise TypeError(f"side must be a mapping from vertex to 0/1, got {type(side).__name__}")
    for vertex, value in side.items():
        _check_side_value(vertex, value)
    total = 0
    for u, v in normalize_edges(edges):
        if u not in side or v not in side:
            raise ValueError(f"edge {(u, v)!r} has an endpoint with no side")
        if side[u] != side[v]:
            total += 1
    return total


def _random_sides(n: int, seed: Optional[int]) -> List[int]:
    """Return ``n`` fair coin flips as sides, drawn from ``make_rng(seed)``.

    Entry i is ``1 if rng.random() < 0.5 else 0``, drawn in index order, so
    it is the side of the i-th smallest vertex.

    Args:
        n: The number of vertices.
        seed: An ``int`` for a reproducible draw, or ``None`` for one seeded
            from the operating system.

    Returns:
        A list of ``n`` values in {0, 1}.

    Raises:
        TypeError: If ``seed`` is neither ``None`` nor an ``int``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _random_sides(4, 42)
        [0, 1, 1, 1]
    """
    rng = make_rng(seed)
    return [1 if rng.random() < 0.5 else 0 for _ in range(n)]


def maxcut_random(
    vertices: Iterable[Hashable], edges: Iterable[Any], seed: Optional[int] = None
) -> Tuple[int, Dict[Hashable, int]]:
    """Return a uniformly random cut: every vertex picks a side by a fair coin.

    The vertices are visited in increasing order and each gets side
    ``1 if rng.random() < 0.5 else 0`` from one generator
    ``rng = make_rng(seed)``. Every edge is then cut with probability 1/2, so
    by linearity of expectation E[cut] = |E|/2 >= OPT/2 (OPT <= |E|); the
    full proof is in the module docstring. The edges are not looked at when
    choosing the sides, only when counting the cut.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
        edges: The edges, 2-tuples with both ends in ``vertices``.
        seed: An ``int`` for a reproducible cut (the same seed always gives
            the same result), or ``None`` for one seeded from the operating
            system.

    Returns:
        ``(cut, side)``: the cut value and a dict mapping every vertex to 0
        or 1, with keys in increasing vertex order.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable,
            or ``seed`` is neither ``None`` nor an ``int``.
        ValueError: If an edge is malformed, a self-loop, or has an endpoint
            outside ``vertices``.

    Time Complexity:
        O(V log V + E): the sort fixes the order of the coin flips, then one
        draw per vertex and one check per edge.

    Space Complexity:
        O(V + E).

    Examples:
        >>> k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        >>> maxcut_random(range(4), k4, seed=42)
        (3, {0: 0, 1: 1, 2: 1, 3: 1})
        >>> maxcut_random(range(4), k4, seed=7) == maxcut_random(range(4), k4, seed=7)
        True
        >>> maxcut_random([], [], seed=1)
        (0, {})
    """
    order, edge_set = _graph(vertices, edges)
    sides = _random_sides(len(order), seed)
    side = dict(zip(order, sides))
    cut = sum(1 for u, v in edge_set if side[u] != side[v])
    return cut, side


def maxcut_local_search(
    vertices: Iterable[Hashable],
    edges: Iterable[Any],
    seed: Optional[int] = None,
    start: Optional[Mapping[Hashable, int]] = None,
) -> Tuple[int, Dict[Hashable, int], int]:
    """Improve a cut by single-vertex flips until no flip helps.

    Starting from ``start`` (or, when it is ``None``, from the same random
    assignment :func:`maxcut_random` draws with ``seed``), the vertices are
    scanned in increasing order and the FIRST vertex whose flip strictly
    increases the cut is flipped; the scan then restarts from the smallest
    vertex. It stops when a full scan finds no improving flip, which is a
    local optimum. There every vertex has at least half its incident edges
    cut, so the cut is at least |E|/2 >= OPT/2, and since each flip raises
    the cut by at least 1 there are at most |E| flips (proofs in the module
    docstring). The number of same-side neighbours of each vertex is kept up
    to date on adjacency lists, so a gain is read in O(1) and a flip costs
    O(deg v); large sparse graphs (n = 500) finish quickly.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
        edges: The edges, 2-tuples with both ends in ``vertices``.
        seed: Seeds the random start when ``start`` is ``None`` (``None``
            means a start seeded from the operating system). When ``start``
            is given the seed is not used, but it is still type-checked.
        start: Optional starting assignment, a mapping that gives every
            vertex (and nothing else) the side 0 or 1. It is not modified.

    Returns:
        ``(cut, side, flips)``: the final cut value, the final side map
        (keys in increasing vertex order) and the number of flips made. The
        cut equals the starting cut plus at least ``flips``; ``flips`` is 0
        exactly when the start is already a local optimum.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable,
            ``start`` is neither ``None`` nor a mapping, or ``seed`` is
            neither ``None`` nor an ``int``.
        ValueError: If an edge is malformed, a self-loop, or has an endpoint
            outside ``vertices``; or if ``start`` misses a vertex, names a
            vertex not in ``vertices``, or holds a side other than 0 or 1.

    Time Complexity:
        O(V log V + E) to build the adjacency lists and the start, plus
        O(V + deg v) per flip (a restarted scan and the neighbour updates).
        There are F <= |E| flips, so the total is
        O(V log V + E + F (V + Delta)) for maximum degree Delta.

    Space Complexity:
        O(V + E) for the adjacency lists and the per-vertex counters.

    Examples:
        >>> k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        >>> maxcut_local_search(range(4), k4, seed=42)
        (4, {0: 0, 1: 0, 2: 1, 3: 1}, 1)
        >>> k33 = [(a, b) for a in range(3) for b in range(3, 6)]
        >>> cut, side, flips = maxcut_local_search(range(6), k33, start={v: 0 for v in range(6)})
        >>> cut, flips, sorted(v for v in side if side[v] == 1)
        (9, 3, [0, 1, 2])
        >>> maxcut_local_search(range(4), k4, start={0: 0, 1: 0, 2: 1, 3: 1})
        (4, {0: 0, 1: 0, 2: 1, 3: 1}, 0)
        >>> maxcut_local_search([0, 1], [(0, 1)], start={0: 0})
        Traceback (most recent call last):
            ...
        ValueError: start gives no side to vertex 1
    """
    order, edge_set = _graph(vertices, edges)
    n = len(order)
    if start is None:
        sides = _random_sides(n, seed)
    else:
        if seed is not None:
            make_rng(seed)  # type-check the seed even though the start is given
        if not isinstance(start, Mapping):
            raise TypeError(
                f"start must be a mapping from vertex to 0/1, got {type(start).__name__}"
            )
        sides = []
        for v in order:
            if v not in start:
                raise ValueError(f"start gives no side to vertex {v!r}")
            sides.append(_check_side_value(v, start[v]))
        if len(start) != n:
            members = set(order)
            extra = next(v for v in start if v not in members)
            raise ValueError(f"start names {extra!r}, which is not a vertex")

    adj = _adjacency(order, edge_set)
    degree = [len(neighbours) for neighbours in adj]
    # same[i] = s(v_i), the number of neighbours on v_i's own side (its uncut edges).
    same = [sum(1 for j in adj[i] if sides[j] == sides[i]) for i in range(n)]
    cut = sum(degree[i] - same[i] for i in range(n)) // 2

    flips = 0
    i = 0
    while i < n:
        # gain = s(v) - c(v) = same - (degree - same); improving iff it is > 0.
        gain = 2 * same[i] - degree[i]
        if gain > 0:
            old = sides[i]
            sides[i] = 1 - old
            for j in adj[i]:
                if sides[j] == old:
                    same[j] -= 1  # was on v's side, now on the other side
                else:
                    same[j] += 1  # was across, now on v's new side
            same[i] = degree[i] - same[i]
            cut += gain
            flips += 1
            i = 0  # restart the scan from the smallest vertex
        else:
            i += 1
    return cut, dict(zip(order, sides)), flips


def maxcut_exact(
    vertices: Iterable[Hashable], edges: Iterable[Any]
) -> Tuple[int, Dict[Hashable, int]]:
    """Return a maximum cut by trying every split (small graphs only).

    Swapping the two sides of a split gives the same cut, so the smallest
    vertex is fixed on side 0 and the other n - 1 vertices run through all
    2^(n-1) assignments. Assignment number ``mask`` puts the i-th smallest
    vertex (i >= 1) on side ``(mask >> (i - 1)) & 1``; masks are tried in
    increasing order and the first one with the largest cut is returned, so
    the answer is deterministic. Each cut is counted with bitmasks: for every
    vertex v on side 1, the neighbours of v on side 0 are
    ``adjacency[v] & ~sidemask``.

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
            At most :data:`MAX_EXACT_VERTICES` (16) distinct vertices.
        edges: The edges, 2-tuples with both ends in ``vertices``.

    Returns:
        ``(cut, side)``: the maximum cut value and a side map achieving it,
        with the smallest vertex on side 0 and keys in increasing vertex
        order. ``(0, {})`` for an empty graph.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If there are more than 16 vertices, or an edge is
            malformed, a self-loop, or has an endpoint outside ``vertices``.

    Time Complexity:
        O(V log V + E + 2^(n-1) n): n bitmask operations for each of the
        2^(n-1) assignments, exponential as expected for an NP-hard problem.

    Space Complexity:
        O(V + E).

    Examples:
        >>> c5 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        >>> maxcut_exact(range(5), c5)
        (4, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0})
        >>> maxcut_exact(["x"], [])
        (0, {'x': 0})
        >>> maxcut_exact(range(17), [])
        Traceback (most recent call last):
            ...
        ValueError: maxcut_exact handles at most 16 vertices, got 17
    """
    order, edge_set = _graph(vertices, edges)
    n = len(order)
    if n > MAX_EXACT_VERTICES:
        raise ValueError(
            f"maxcut_exact handles at most {MAX_EXACT_VERTICES} vertices, got {n}"
        )
    if n == 0:
        return 0, {}
    neighbour_mask = [0] * n
    for i, neighbours in enumerate(_adjacency(order, edge_set)):
        for j in neighbours:
            neighbour_mask[i] |= 1 << j

    best_cut, best_sides = -1, 0
    for mask in range(1 << (n - 1)):
        sides = mask << 1  # bit i = side of order[i]; bit 0 (smallest vertex) stays 0
        cut = 0
        rest = sides
        while rest:
            low = rest & -rest
            cut += (neighbour_mask[low.bit_length() - 1] & ~sides).bit_count()
            rest ^= low
        if cut > best_cut:
            best_cut, best_sides = cut, sides
    return best_cut, {v: (best_sides >> i) & 1 for i, v in enumerate(order)}
