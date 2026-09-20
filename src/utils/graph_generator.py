"""Seeded, reproducible graph construction for the Week 4 experiments.

Every benchmark in Week 4 needs graphs that can be rebuilt exactly, on any
machine, months later, from nothing but the arguments printed in the
report. That is the whole job of this module: each generator takes a
``seed`` (42 by default, the convention Week 1 set) and draws from a local
:class:`random.Random` instance, never from the global ``random`` module,
so two callers running side by side cannot disturb each other's stream.

The generators are named after the *shape* they produce, because shape is
the independent variable of the Week 4 benchmarks. Sparse against dense is
what decides whether an adjacency list or an adjacency matrix wins, and it
is what separates BFS and DFS costs from their textbook bounds:

=================== ======================= ============================
Generator           Expected edge count     Generation cost
=================== ======================= ============================
sparse_graph        V * avg_degree / 2      O(V + E) expected
dense_graph         density * V(V-1)/2      O(V^2)
random_graph        density * V(V-1)/2      O(V + E) or O(V^2)
weighted_graph      density * V(V-1)/2      O(V + E) or O(V^2)
complete_graph      V(V-1)/2                O(V^2)
path_graph          V - 1                   O(V)
cycle_graph         V                       O(V)
=================== ======================= ============================

Directed graphs double every count above, because an ordered pair (u, v)
and the pair (v, u) are different edges.

Two design choices are worth stating up front, since callers depend on
them:

* **Nodes are the integers 0 to V-1, inserted in that order.** The Graph
  class preserves insertion order, so ``graph.nodes()[0]`` is always node
  0 and traversal results are deterministic.
* **Structure is drawn before weights.** The same seed therefore produces
  the same *shape* whether or not weights were requested, which makes a
  weighted run and an unweighted run directly comparable.

Every generator accepts ``connected=True``. Dijkstra benchmarks on a graph
where most nodes are unreachable measure nothing but the cost of finding
that out, so the connected flag lays down a random spanning tree first and
fills the remaining edge budget around it. On a directed graph that tree
is oriented away from node 0, so a search started at ``graph.nodes()[0]``
reaches every node.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Sequence, Set, Tuple

from src.graphs.graph import Graph

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_AVG_DEGREE",
    "DEFAULT_WEIGHT_RANGE",
    "sparse_graph",
    "dense_graph",
    "random_graph",
    "weighted_graph",
    "complete_graph",
    "path_graph",
    "cycle_graph",
    "connect",
]

DEFAULT_SEED: int = 42
"""Seed used when the caller does not supply one. Set by Week 1."""

DEFAULT_AVG_DEGREE: int = 4
"""Average degree that makes a graph sparse: edge count stays O(V)."""

DEFAULT_WEIGHT_RANGE: Tuple[float, float] = (1.0, 10.0)
"""Inclusive bounds for randomly drawn edge weights."""

WEIGHT_DECIMALS: int = 2
"""Weights are rounded to this many decimals so reports stay readable."""

MIN_WEIGHT: float = 10.0 ** -WEIGHT_DECIMALS
"""Smallest weight ever produced. Keeps every weight strictly positive."""

_SCAN_FACTOR: int = 4
"""Route to the O(V^2) scan once the request fills 1/4 of the capacity."""


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------
def _validate_node_count(n: int) -> None:
    """Raise unless ``n`` is a usable node count.

    Args:
        n: Requested number of nodes.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative.

    Examples:
        >>> _validate_node_count(0) is None
        True
        >>> _validate_node_count(-3)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 0, got -3
        >>> _validate_node_count(2.5)
        Traceback (most recent call last):
            ...
        TypeError: n must be an int, got float
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")


def _validate_density(density: float) -> None:
    """Raise unless ``density`` is a fraction in [0, 1].

    Args:
        density: Requested fraction of the possible edges.

    Raises:
        ValueError: If ``density`` lies outside [0, 1].

    Examples:
        >>> _validate_density(0.5) is None
        True
        >>> _validate_density(1.5)
        Traceback (most recent call last):
            ...
        ValueError: density must be in [0, 1], got 1.5
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")


def _validate_avg_degree(avg_degree: float, n: int) -> None:
    """Raise unless ``avg_degree`` is achievable on ``n`` nodes.

    A simple graph without self-loops gives every node at most ``n - 1``
    neighbours, so an average degree of ``n`` or more cannot be built. The
    check is skipped when there are no nodes at all, since no edge count is
    reachable there and the argument is simply unused.

    Note that the default of 4 therefore requires ``n >= 5``; tiny graphs
    have to ask for a smaller average degree.

    Args:
        avg_degree: Requested average degree.
        n: Number of nodes in the graph being built.

    Raises:
        ValueError: If ``avg_degree`` is negative, or is not smaller
            than ``n`` when the graph has at least one node.

    Examples:
        >>> _validate_avg_degree(4, 100) is None
        True
        >>> _validate_avg_degree(4, 0) is None
        True
        >>> _validate_avg_degree(4, 4)
        Traceback (most recent call last):
            ...
        ValueError: avg_degree must be < n, got avg_degree=4 with n=4
        >>> _validate_avg_degree(-1, 10)
        Traceback (most recent call last):
            ...
        ValueError: avg_degree must be >= 0, got -1
    """
    if avg_degree < 0:
        raise ValueError(f"avg_degree must be >= 0, got {avg_degree}")
    if n >= 1 and avg_degree >= n:
        raise ValueError(
            f"avg_degree must be < n, got avg_degree={avg_degree} with n={n}"
        )


def _validate_weight_range(weight_range: Sequence[float]) -> Tuple[float, float]:
    """Raise unless ``weight_range`` is a usable pair of positive bounds.

    Weights must stay strictly positive because Dijkstra's correctness
    argument depends on it: a zero or negative edge lets a longer path win,
    and the algorithm would return a confidently wrong answer. The range is
    validated even when the graph is unweighted, so a typo is caught at the
    call that contains it rather than on a later weighted run.

    Args:
        weight_range: Pair ``(low, high)`` of inclusive bounds.

    Returns:
        The same bounds as a ``(low, high)`` tuple of floats.

    Raises:
        ValueError: If the pair does not hold exactly two values, if
            ``low`` exceeds ``high``, or if ``low`` is not positive.

    Examples:
        >>> _validate_weight_range((1.0, 10.0))
        (1.0, 10.0)
        >>> _validate_weight_range((3, 3))
        (3.0, 3.0)
        >>> _validate_weight_range((9.0, 1.0))
        Traceback (most recent call last):
            ...
        ValueError: weight_range must be (low, high) with low <= high, got (9.0, 1.0)
        >>> _validate_weight_range((0.0, 5.0))
        Traceback (most recent call last):
            ...
        ValueError: weight_range must be strictly positive, got (0.0, 5.0)
    """
    bounds = tuple(weight_range)
    if len(bounds) != 2:
        raise ValueError(
            f"weight_range must be a (low, high) pair, got {weight_range!r}"
        )
    low, high = float(bounds[0]), float(bounds[1])
    if low > high:
        raise ValueError(
            "weight_range must be (low, high) with low <= high, "
            f"got {(low, high)}"
        )
    if low <= 0.0:
        raise ValueError(f"weight_range must be strictly positive, got {(low, high)}")
    return low, high


# ----------------------------------------------------------------------
# Edge bookkeeping
# ----------------------------------------------------------------------
def _max_edges(n: int, directed: bool) -> int:
    """Return the number of distinct edges a simple graph on ``n`` nodes allows.

    Self-loops are excluded, since none of the generators produce them.

    Args:
        n: Number of nodes.
        directed: True to count ordered pairs, False for unordered ones.

    Returns:
        ``n * (n - 1)`` when directed, ``n * (n - 1) // 2`` otherwise.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _max_edges(5, False)
        10
        >>> _max_edges(5, True)
        20
        >>> _max_edges(1, False)
        0
    """
    if n < 2:
        return 0
    return n * (n - 1) if directed else n * (n - 1) // 2


def _edge_key(u: int, v: int, directed: bool) -> Tuple[int, int]:
    """Return the duplicate-detection key for the edge ``u`` to ``v``.

    On an undirected graph (u, v) and (v, u) are the same edge, so the key
    is ordered by label. Labels are handed out in insertion order, so the
    smaller label is also the endpoint that was inserted first.

    Args:
        u: First endpoint.
        v: Second endpoint.
        directed: True to keep the pair ordered as given.

    Returns:
        The pair to store in the "already used" set.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _edge_key(3, 1, directed=False)
        (1, 3)
        >>> _edge_key(3, 1, directed=True)
        (3, 1)
    """
    if directed or u <= v:
        return (u, v)
    return (v, u)


def _target_edge_count(n: int, avg_degree: float, directed: bool) -> int:
    """Return the edge count that yields the requested average degree.

    On an undirected graph every edge contributes to two nodes' degrees, so
    ``sum(degrees) = 2E`` and ``E = V * avg_degree / 2``. On a directed
    graph ``avg_degree`` is read as the average *out*-degree, which gives
    ``E = V * avg_degree``.

    Args:
        n: Number of nodes.
        avg_degree: Desired average degree.
        directed: True when the graph is directed.

    Returns:
        The target edge count, never more than the graph can hold.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _target_edge_count(100, 4, directed=False)
        200
        >>> _target_edge_count(100, 4, directed=True)
        400
        >>> _target_edge_count(0, 4, directed=False)
        0
    """
    scale = 1.0 if directed else 0.5
    return min(round(n * avg_degree * scale), _max_edges(n, directed))


def _random_weight(rng: random.Random, weight_range: Tuple[float, float]) -> float:
    """Draw one edge weight, rounded for readability and kept positive.

    A uniform draw is rounded to :data:`WEIGHT_DECIMALS` places so tables
    and figures in the report stay legible. Rounding can send a very small
    draw to 0.0, which would break Dijkstra's assumption of positive
    weights, so the result is floored at :data:`MIN_WEIGHT`.

    Args:
        rng: The caller's seeded random source.
        weight_range: Validated ``(low, high)`` bounds.

    Returns:
        A strictly positive float with at most two decimal places.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _random_weight(random.Random(42), (1.0, 10.0))
        6.75
        >>> _random_weight(random.Random(42), (0.001, 0.004))
        0.01
    """
    low, high = weight_range
    return max(round(rng.uniform(low, high), WEIGHT_DECIMALS), MIN_WEIGHT)


def _random_tree_pairs(
    nodes: Sequence[object], rng: random.Random
) -> List[Tuple[object, object]]:
    """Return V-1 pairs forming a random spanning tree over ``nodes``.

    The first node keeps its position and the rest are shuffled; each node
    in the shuffled order is then attached to a uniformly chosen node that
    came before it. That is a random recursive tree: it always spans, it is
    built in one pass, and because the root is ``nodes[0]`` the pairs can be
    read as parent-to-child on a directed graph and still leave every node
    reachable from ``nodes[0]``.

    This is not uniform over all labelled spanning trees the way a
    Wilson-style walk would be. It does not need to be: the goal is a
    connected graph with no structural bias towards any particular node,
    and paying O(V) instead of a random walk's expected O(V log V) matters
    at V = 10,000.

    Args:
        nodes: The graph's nodes, in insertion order.
        rng: The caller's seeded random source.

    Returns:
        A list of ``(parent, child)`` pairs, empty when fewer than two
        nodes were given.

    Time Complexity:
        O(V) - one shuffle and one pass.

    Space Complexity:
        O(V) for the shuffled order and the returned pairs.

    Examples:
        >>> pairs = _random_tree_pairs([0, 1, 2, 3, 4], random.Random(42))
        >>> len(pairs)
        4
        >>> sorted({node for pair in pairs for node in pair})
        [0, 1, 2, 3, 4]
        >>> pairs == _random_tree_pairs([0, 1, 2, 3, 4], random.Random(42))
        True
        >>> _random_tree_pairs([7], random.Random(42))
        []
    """
    if len(nodes) < 2:
        return []
    order = list(nodes[1:])
    rng.shuffle(order)
    order.insert(0, nodes[0])
    return [(order[rng.randrange(i)], order[i]) for i in range(1, len(order))]


def _iter_candidate_pairs(n: int, directed: bool) -> Iterator[Tuple[int, int]]:
    """Yield every legal edge of a simple graph on ``n`` nodes, lazily.

    Laziness is the point: the caller stops as soon as it has enough edges,
    and at no time is the full O(V^2) candidate list held in memory.

    Args:
        n: Number of nodes.
        directed: True to yield ordered pairs.

    Yields:
        Pairs ``(u, v)`` with ``u != v``; ``u < v`` when undirected.

    Time Complexity:
        O(V^2) to exhaust.

    Space Complexity:
        O(1) - a generator, not a materialised list.

    Examples:
        >>> list(_iter_candidate_pairs(3, directed=False))
        [(0, 1), (0, 2), (1, 2)]
        >>> len(list(_iter_candidate_pairs(3, directed=True)))
        6
    """
    for u in range(n):
        if directed:
            for v in range(n):
                if u != v:
                    yield (u, v)
        else:
            for v in range(u + 1, n):
                yield (u, v)


def _sample_by_rejection(
    n: int,
    count: int,
    directed: bool,
    rng: random.Random,
    used: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Draw ``count`` fresh edges by guessing endpoints and retrying clashes.

    Only ever called when the graph is far from full, so a guess is
    accepted with high probability and the expected number of draws stays
    proportional to ``count``. This is what keeps a sparse graph at
    V = 10,000 cheap: it never touches the 10^8 candidate pairs.

    Args:
        n: Number of nodes.
        count: How many new edges to draw. Must fit in the free capacity.
        directed: True when the graph is directed.
        rng: The caller's seeded random source.
        used: Keys already taken. Updated in place with the new edges.

    Returns:
        ``count`` pairs, none of them duplicates or self-loops.

    Time Complexity:
        O(count) expected under the occupancy guard described above.

    Space Complexity:
        O(count) for the result, plus the caller's ``used`` set.

    Examples:
        >>> taken = set()
        >>> pairs = _sample_by_rejection(50, 10, False, random.Random(42), taken)
        >>> len(pairs), len(taken)
        (10, 10)
        >>> any(u == v for u, v in pairs)
        False
    """
    chosen: List[Tuple[int, int]] = []
    while len(chosen) < count:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        key = _edge_key(u, v, directed)
        if key in used:
            continue
        used.add(key)
        chosen.append((u, v))
    return chosen


def _sample_by_scan(
    n: int,
    count: int,
    directed: bool,
    rng: random.Random,
    used: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Draw ``count`` fresh edges in one pass over the candidate pairs.

    Selection sampling: each still-free candidate is taken with probability
    ``needed / remaining``, which selects exactly ``count`` of them,
    uniformly, in a single pass and without shuffling anything. This is the
    right method once the request is a large fraction of capacity, where
    rejection sampling would spend most of its draws on collisions.

    Args:
        n: Number of nodes.
        count: How many new edges to select. Must fit in the free capacity.
        directed: True when the graph is directed.
        rng: The caller's seeded random source.
        used: Keys already taken. Updated in place with the new edges.

    Returns:
        ``count`` pairs in ascending candidate order.

    Time Complexity:
        O(V^2) worst case - the scan, not the selection, dominates.

    Space Complexity:
        O(count) for the result; the candidate stream is a generator.

    Examples:
        >>> taken = set()
        >>> pairs = _sample_by_scan(5, 10, False, random.Random(42), taken)
        >>> pairs == list(_iter_candidate_pairs(5, directed=False))
        True
        >>> _sample_by_scan(4, 2, False, random.Random(42), {(0, 1)})
        [(0, 3), (1, 2)]
    """
    needed = count
    remaining = _max_edges(n, directed) - len(used)
    chosen: List[Tuple[int, int]] = []
    for pair in _iter_candidate_pairs(n, directed):
        if needed == 0:
            break
        key = _edge_key(pair[0], pair[1], directed)
        if key in used:
            continue
        if rng.random() * remaining < needed:
            used.add(key)
            chosen.append(pair)
            needed -= 1
        remaining -= 1
    return chosen


def _sample_pairs(
    n: int,
    count: int,
    directed: bool,
    rng: random.Random,
    used: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Draw ``count`` fresh edges, choosing the cheaper sampling strategy.

    The crossover is occupancy. While the taken edges plus the request fill
    no more than ``1 / _SCAN_FACTOR`` of the capacity, rejection sampling
    is O(count) and the O(V^2) scan would be wasteful; past that point the
    collision rate climbs and the single-pass scan wins. The guard doubles
    as the safety proof for rejection sampling, which would otherwise spin
    on a nearly full graph.

    Args:
        n: Number of nodes.
        count: How many new edges to draw.
        directed: True when the graph is directed.
        rng: The caller's seeded random source.
        used: Keys already taken. Updated in place.

    Returns:
        ``count`` new pairs, or fewer only if capacity runs out.

    Time Complexity:
        O(count) expected on a sparse request, O(V^2) on a dense one.

    Space Complexity:
        O(count).

    Examples:
        >>> taken = set()
        >>> len(_sample_pairs(100, 20, False, random.Random(42), taken))
        20
        >>> _sample_pairs(3, 3, False, random.Random(42), set())
        [(0, 1), (0, 2), (1, 2)]
        >>> _sample_pairs(10, 0, False, random.Random(42), set())
        []
    """
    if count <= 0:
        return []
    capacity = _max_edges(n, directed)
    count = min(count, capacity - len(used))
    if count <= 0:
        return []
    if (len(used) + count) * _SCAN_FACTOR >= capacity:
        return _sample_by_scan(n, count, directed, rng, used)
    return _sample_by_rejection(n, count, directed, rng, used)


def _build_graph(
    n: int,
    target_edges: int,
    *,
    directed: bool,
    weighted: bool,
    weight_range: Tuple[float, float],
    connected: bool,
    rng: random.Random,
) -> Graph:
    """Assemble a graph with ``n`` nodes and about ``target_edges`` edges.

    The shared engine behind every random generator. Nodes go in first, so
    an isolated node still appears in ``nodes()``. If connectivity was
    asked for, a spanning tree is reserved before any random edge is drawn,
    and the remaining budget is filled around it; that is why ``connected``
    can push the edge count up to ``V - 1`` when the caller asked for less.

    Weights are drawn in a second pass over the finished edge list, so the
    structure produced by a given seed does not depend on whether weights
    were requested.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        target_edges: Desired edge count, clamped to the capacity.
        directed: True for a directed graph.
        weighted: True to attach random weights.
        weight_range: Validated ``(low, high)`` bounds.
        connected: True to guarantee a connected (or, when directed, a
            from-node-0 reachable) result.
        rng: The caller's seeded random source.

    Returns:
        A :class:`~src.graphs.graph.Graph` built to those requirements.

    Time Complexity:
        O(V + E) expected for a sparse request, O(V^2) for a dense one.

    Space Complexity:
        O(V + E) - the graph itself plus the set of used edge keys.

    Examples:
        >>> graph = _build_graph(
        ...     6, 3, directed=False, weighted=False,
        ...     weight_range=(1.0, 10.0), connected=True,
        ...     rng=random.Random(42),
        ... )
        >>> graph.node_count, graph.edge_count
        (6, 5)
    """
    graph = Graph(directed=directed, weighted=weighted)
    for node in range(n):
        graph.add_node(node)

    used: Set[Tuple[int, int]] = set()
    pairs: List[Tuple[int, int]] = []
    if connected:
        for parent, child in _random_tree_pairs(list(range(n)), rng):
            key = _edge_key(parent, child, directed)
            if key not in used:
                used.add(key)
                pairs.append((parent, child))

    pairs.extend(
        _sample_pairs(n, target_edges - len(pairs), directed, rng, used)
    )

    for u, v in pairs:
        if weighted:
            graph.add_edge(u, v, _random_weight(rng, weight_range))
        else:
            graph.add_edge(u, v)
    return graph


# ----------------------------------------------------------------------
# Public generators
# ----------------------------------------------------------------------
def sparse_graph(
    n: int,
    *,
    avg_degree: float = DEFAULT_AVG_DEGREE,
    directed: bool = False,
    weighted: bool = False,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    connected: bool = False,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build a graph whose edge count grows linearly with its node count.

    Sparsity is expressed as an average degree rather than a density,
    because that is what holds the edge count at O(V) as V grows: a density
    fixed at any constant would put E back at O(V^2). An average degree of
    4 is the road-network end of the scale, and it is where an adjacency
    list beats an adjacency matrix decisively on both time and memory.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        avg_degree: Mean number of neighbours per node; read as mean
            out-degree when ``directed``. Must be smaller than ``n``, so
            the default of 4 requires ``n >= 5``.
        directed: True for a directed graph.
        weighted: True to attach random weights from ``weight_range``.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        connected: True to lay down a random spanning tree first, which
            raises the edge count to at least ``n - 1``.
        seed: Seed for this call's private random source.

    Returns:
        A graph with about ``n * avg_degree / 2`` edges, or
        ``n * avg_degree`` when directed.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, ``avg_degree`` is negative or is
            not smaller than ``n``, or ``weight_range`` is inverted or not
            strictly positive.

    Time Complexity:
        O(V + E) expected. Edges are sampled by guessing endpoints, so the
        O(V^2) candidate list is never built - the reason V = 10,000 takes
        a moment rather than minutes.

    Space Complexity:
        O(V + E).

    Examples:
        >>> graph = sparse_graph(100)
        >>> graph.node_count, graph.edge_count
        (100, 200)
        >>> graph.directed, graph.weighted
        (False, False)
        >>> sparse_graph(100, directed=True).edge_count
        400

        The same seed and arguments rebuild the same graph, and a
        different seed does not:

        >>> sparse_graph(100).edges() == sparse_graph(100).edges()
        True
        >>> sparse_graph(100).edges() == sparse_graph(100, seed=7).edges()
        False

        Asking for connectivity raises a too-small budget to a spanning
        tree and leaves no isolated node behind:

        >>> tree = sparse_graph(50, avg_degree=1, connected=True)
        >>> tree.edge_count
        49
        >>> min(tree.degree(node) for node in tree.nodes())
        1

        >>> sparse_graph(4, avg_degree=4)
        Traceback (most recent call last):
            ...
        ValueError: avg_degree must be < n, got avg_degree=4 with n=4
    """
    _validate_node_count(n)
    _validate_avg_degree(avg_degree, n)
    bounds = _validate_weight_range(weight_range)
    return _build_graph(
        n,
        _target_edge_count(n, avg_degree, directed),
        directed=directed,
        weighted=weighted,
        weight_range=bounds,
        connected=connected,
        rng=random.Random(seed),
    )


def dense_graph(
    n: int,
    *,
    density: float = 0.5,
    directed: bool = False,
    weighted: bool = False,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    connected: bool = False,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build a graph holding at least half of all possible edges.

    The dense half of the Week 4 comparison. At density 0.5 the edge count
    is Theta(V^2), which is where the adjacency matrix stops looking
    wasteful - it stores the same information in a fixed V^2 bits while the
    list pays a Python object per edge.

    Only the default differs from :func:`random_graph`; the name exists so
    a benchmark reads as "sparse against dense" rather than as two calls
    that differ by one number.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        density: Fraction of possible edges to include, 0.5 by default.
        directed: True for a directed graph.
        weighted: True to attach random weights from ``weight_range``.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        connected: True to lay down a random spanning tree first.
        seed: Seed for this call's private random source.

    Returns:
        A graph with ``round(density * V * (V - 1) / 2)`` edges, or twice
        that when directed.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, ``density`` lies outside [0, 1],
            or ``weight_range`` is inverted or not strictly positive.

    Time Complexity:
        O(V^2). A dense request is filled by one selection-sampling pass
        over every candidate pair, so generation cost - and the resulting
        edge count - grow quadratically. At V = 10,000 that is 5 * 10^7
        candidate pairs and about 2.5 * 10^7 edges, which is minutes of
        work and gigabytes of graph; keep dense runs to the small and
        medium sizes and say so in the report.

    Space Complexity:
        O(V + E), which is O(V^2) at this density.

    Examples:
        >>> graph = dense_graph(20)
        >>> graph.node_count, graph.edge_count
        (20, 95)
        >>> round(graph.density(), 3)
        0.5
        >>> dense_graph(20, density=0.8).edge_count
        152
        >>> dense_graph(20).edges() == dense_graph(20).edges()
        True
    """
    _validate_node_count(n)
    _validate_density(density)
    bounds = _validate_weight_range(weight_range)
    return _build_graph(
        n,
        round(density * _max_edges(n, directed)),
        directed=directed,
        weighted=weighted,
        weight_range=bounds,
        connected=connected,
        rng=random.Random(seed),
    )


def random_graph(
    n: int,
    *,
    density: float,
    directed: bool = False,
    weighted: bool = False,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    connected: bool = False,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build a graph at a caller-specified density.

    The general case behind :func:`sparse_graph` and :func:`dense_graph`,
    exposed for sweeps that walk density across a range and watch where the
    two representations cross over. ``density`` is required rather than
    defaulted, because the whole point of calling this function instead of
    one of the named two is that the caller has a specific value in mind.

    Unlike a coin flip per pair, the edge count here is exact: the density
    is converted to a count and that many distinct edges are drawn. Two
    graphs at the same density therefore have the same number of edges,
    which keeps a benchmark's x-axis honest.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        density: Fraction of possible edges to include, in [0, 1].
        directed: True for a directed graph.
        weighted: True to attach random weights from ``weight_range``.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        connected: True to lay down a random spanning tree first, which
            raises the edge count to at least ``n - 1``.
        seed: Seed for this call's private random source.

    Returns:
        A graph with ``round(density * V * (V - 1) / 2)`` edges, or twice
        that when directed.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, ``density`` lies outside [0, 1],
            or ``weight_range`` is inverted or not strictly positive.

    Time Complexity:
        O(V + E) expected below a quarter of capacity, where edges are
        sampled by rejection; O(V^2) at or above it, where one pass over
        the candidate pairs is cheaper than fighting collisions.

    Space Complexity:
        O(V + E).

    Examples:
        >>> random_graph(10, density=0.0).node_count
        10
        >>> random_graph(10, density=0.0).edge_count
        0
        >>> random_graph(10, density=1.0).edge_count
        45
        >>> random_graph(10, density=1.0, directed=True).edge_count
        90
        >>> round(random_graph(200, density=0.05).density(), 4)
        0.05
        >>> random_graph(10, density=1.5)
        Traceback (most recent call last):
            ...
        ValueError: density must be in [0, 1], got 1.5
        >>> random_graph(-1, density=0.5)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 0, got -1
    """
    _validate_node_count(n)
    _validate_density(density)
    bounds = _validate_weight_range(weight_range)
    return _build_graph(
        n,
        round(density * _max_edges(n, directed)),
        directed=directed,
        weighted=weighted,
        weight_range=bounds,
        connected=connected,
        rng=random.Random(seed),
    )


def weighted_graph(
    n: int,
    *,
    density: float = 0.1,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    directed: bool = False,
    connected: bool = False,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build a weighted graph for the shortest-path benchmarks.

    Weighting is the argument this generator removes: there is no
    ``weighted`` flag, because a weighted graph is the entire point. The
    default density of 0.1 is deliberately low - Dijkstra's advantage over
    a scan-based queue is an ``E log V`` against ``V^2`` story, and it only
    shows on a graph where E is well below V^2.

    Weights are uniform in ``weight_range``, rounded to two decimals, and
    strictly positive, which is exactly the domain Dijkstra is defined on.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        density: Fraction of possible edges to include, 0.1 by default.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        directed: True for a directed graph.
        connected: True to lay down a random spanning tree first, so every
            node has a finite distance from node 0.
        seed: Seed for this call's private random source.

    Returns:
        A weighted graph with ``round(density * V * (V - 1) / 2)`` edges,
        or twice that when directed.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, ``density`` lies outside [0, 1],
            or ``weight_range`` is inverted or not strictly positive.

    Time Complexity:
        O(V + E) expected at the default density; O(V^2) once the request
        passes a quarter of capacity.

    Space Complexity:
        O(V + E).

    Examples:
        >>> graph = weighted_graph(30, density=0.2, weight_range=(1.0, 5.0))
        >>> graph.weighted, graph.edge_count
        (True, 87)
        >>> weights = [weight for _, _, weight in graph.edges()]
        >>> min(weights) >= 1.0 and max(weights) <= 5.0
        True
        >>> all(round(weight, 2) == weight for weight in weights)
        True

        Every weight stays strictly positive even when the whole range
        would round to zero:

        >>> tiny = weighted_graph(6, density=1.0, weight_range=(0.001, 0.004))
        >>> sorted({weight for _, _, weight in tiny.edges()})
        [0.01]

        Connectivity makes Dijkstra's answer finite everywhere:

        >>> reachable = weighted_graph(40, density=0.02, connected=True)
        >>> reachable.edge_count
        39
    """
    _validate_node_count(n)
    _validate_density(density)
    bounds = _validate_weight_range(weight_range)
    return _build_graph(
        n,
        round(density * _max_edges(n, directed)),
        directed=directed,
        weighted=True,
        weight_range=bounds,
        connected=connected,
        rng=random.Random(seed),
    )


def complete_graph(
    n: int,
    *,
    directed: bool = False,
    weighted: bool = False,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    connected: bool = True,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build the graph in which every node is joined to every other.

    Density 1.0, and so the worst case for anything that walks neighbour
    lists: BFS and DFS both touch all V^2 of them. It is also the one
    random-looking generator with no randomness in its structure at all,
    which makes it a useful fixed point when a benchmark result looks
    suspicious.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        directed: True to include both (u, v) and (v, u).
        weighted: True to attach random weights from ``weight_range``.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        connected: Accepted for a uniform call signature across the
            generators and ignored: a complete graph is already connected.
        seed: Seed for this call's private random source. Affects the
            weights only, never the structure.

    Returns:
        A graph with ``V * (V - 1) / 2`` edges, or ``V * (V - 1)`` when
        directed.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, or ``weight_range`` is inverted
            or not strictly positive.

    Time Complexity:
        O(V^2) - there are that many edges to create. Memory grows the same
        way, so this is a small-n generator: V = 2,000 already means about
        two million edges.

    Space Complexity:
        O(V^2).

    Examples:
        >>> graph = complete_graph(5)
        >>> graph.node_count, graph.edge_count
        (5, 10)
        >>> sorted(graph.get_neighbors(0))
        [1, 2, 3, 4]
        >>> complete_graph(5, directed=True).edge_count
        20
        >>> complete_graph(1).edge_count
        0
        >>> round(complete_graph(8).density(), 3)
        1.0
    """
    _validate_node_count(n)
    bounds = _validate_weight_range(weight_range)
    rng = random.Random(seed)
    graph = Graph(directed=directed, weighted=weighted)
    for node in range(n):
        graph.add_node(node)
    for u, v in _iter_candidate_pairs(n, directed):
        if weighted:
            graph.add_edge(u, v, _random_weight(rng, bounds))
        else:
            graph.add_edge(u, v)
    return graph


def path_graph(
    n: int,
    *,
    directed: bool = False,
    weighted: bool = False,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    connected: bool = True,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build a single chain 0 - 1 - 2 - ... - (n-1).

    The deepest graph there is: every node has degree 2 at most, and the
    only route from one end to the other passes through all V nodes. That
    makes it the deliberate stress case for recursive depth-first search,
    which needs V stack frames here and will hit Python's recursion limit
    long before memory runs out. The iterative traversals walk it without
    complaint, which is the comparison worth putting in the report.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        directed: True to orient every edge from ``i`` to ``i + 1``.
        weighted: True to attach random weights from ``weight_range``.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        connected: Accepted for a uniform call signature across the
            generators and ignored: a path is already connected.
        seed: Seed for this call's private random source. Affects the
            weights only, never the structure.

    Returns:
        A graph with ``max(n - 1, 0)`` edges.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, or ``weight_range`` is inverted
            or not strictly positive.

    Time Complexity:
        O(V).

    Space Complexity:
        O(V).

    Examples:
        >>> graph = path_graph(5)
        >>> graph.edges()
        [(0, 1), (1, 2), (2, 3), (3, 4)]
        >>> graph.node_count, graph.edge_count
        (5, 4)
        >>> path_graph(1).edge_count
        0
        >>> path_graph(0).node_count
        0
        >>> path_graph(5, weighted=True).get_edge_weight(0, 1)
        6.75
    """
    return _build_chain(
        n,
        close_the_loop=False,
        directed=directed,
        weighted=weighted,
        weight_range=weight_range,
        seed=seed,
    )


def cycle_graph(
    n: int,
    *,
    directed: bool = False,
    weighted: bool = False,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    connected: bool = True,
    seed: int = DEFAULT_SEED,
) -> Graph:
    """Build a ring 0 - 1 - ... - (n-1) - 0.

    A path with one extra edge, and that edge changes what the graph tests:
    it is the smallest structure containing a cycle at every node, so any
    traversal that forgets to mark nodes visited loops here forever. Every
    node has degree exactly 2 when undirected, which also makes the total
    edge count trivially predictable at V.

    A ring needs at least two nodes. With one node the closing edge would
    be a self-loop, so it is not added.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        directed: True to orient the ring as 0 -> 1 -> ... -> 0.
        weighted: True to attach random weights from ``weight_range``.
        weight_range: Inclusive ``(low, high)`` bounds for weights.
        connected: Accepted for a uniform call signature across the
            generators and ignored: a ring is already connected.
        seed: Seed for this call's private random source. Affects the
            weights only, never the structure.

    Returns:
        A graph with ``n`` edges, except that an undirected 2-node ring is
        a single edge and rings of 0 or 1 node have none.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, or ``weight_range`` is inverted
            or not strictly positive.

    Time Complexity:
        O(V).

    Space Complexity:
        O(V).

    Examples:
        >>> graph = cycle_graph(6)
        >>> graph.node_count, graph.edge_count
        (6, 6)
        >>> sorted(graph.degree(node) for node in graph.nodes())
        [2, 2, 2, 2, 2, 2]
        >>> cycle_graph(2).edge_count
        1
        >>> cycle_graph(2, directed=True).edge_count
        2
        >>> cycle_graph(1).edge_count
        0
    """
    return _build_chain(
        n,
        close_the_loop=True,
        directed=directed,
        weighted=weighted,
        weight_range=weight_range,
        seed=seed,
    )


def _build_chain(
    n: int,
    *,
    close_the_loop: bool,
    directed: bool,
    weighted: bool,
    weight_range: Sequence[float],
    seed: int,
) -> Graph:
    """Build a path, optionally closed into a ring.

    Shared by :func:`path_graph` and :func:`cycle_graph`, which differ by
    exactly one edge.

    Args:
        n: Number of nodes, labelled 0 to n-1.
        close_the_loop: True to join the last node back to the first.
        directed: True to orient every edge forwards.
        weighted: True to attach random weights.
        weight_range: ``(low, high)`` bounds, validated here.
        seed: Seed for this call's private random source.

    Returns:
        The finished graph.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n`` is negative, or ``weight_range`` is invalid.

    Time Complexity:
        O(V).

    Space Complexity:
        O(V).

    Examples:
        >>> _build_chain(
        ...     4, close_the_loop=True, directed=False, weighted=False,
        ...     weight_range=(1.0, 10.0), seed=42,
        ... ).edge_count
        4
    """
    _validate_node_count(n)
    bounds = _validate_weight_range(weight_range)
    rng = random.Random(seed)
    graph = Graph(directed=directed, weighted=weighted)
    for node in range(n):
        graph.add_node(node)

    pairs = [(i, i + 1) for i in range(n - 1)]
    if close_the_loop and n > 2:
        pairs.append((n - 1, 0))
    elif close_the_loop and n == 2 and directed:
        # Undirected, this would only restate the edge that already exists.
        pairs.append((1, 0))

    for u, v in pairs:
        if weighted:
            graph.add_edge(u, v, _random_weight(rng, bounds))
        else:
            graph.add_edge(u, v)
    return graph


def connect(
    graph: Graph,
    rng: Optional[random.Random] = None,
    *,
    weight_range: Sequence[float] = DEFAULT_WEIGHT_RANGE,
    seed: int = DEFAULT_SEED,
) -> int:
    """Add edges to an existing graph until it is connected.

    Lays a random spanning tree over the graph's current nodes and adds
    only the tree edges that are missing, so edges already present keep
    their weights and the edge count rises by as little as possible. On a
    directed graph the tree is oriented away from ``graph.nodes()[0]``,
    which is the guarantee Dijkstra benchmarks actually need: every node
    has a finite distance from that source.

    Use this on a graph you built some other way. The generators in this
    module take ``connected=True`` instead, which is cheaper because it
    reserves the tree before drawing random edges rather than patching
    afterwards.

    Args:
        graph: The graph to connect. Modified in place.
        rng: Random source to draw the tree from. A fresh
            ``random.Random(seed)`` is used when omitted, which makes the
            result reproducible by default.
        weight_range: Inclusive ``(low, high)`` bounds for any weights the
            new edges need. Ignored on an unweighted graph.
        seed: Seed used only when ``rng`` is omitted.

    Returns:
        The number of edges added. Zero means the graph already contained
        a spanning tree of the shape this call drew.

    Raises:
        ValueError: If ``weight_range`` is inverted or not strictly
            positive.

    Time Complexity:
        O(V) - one shuffle, one pass, and an O(1) lookup per tree edge.

    Space Complexity:
        O(V) for the shuffled node order.

    Examples:
        >>> graph = random_graph(20, density=0.0)
        >>> graph.edge_count
        0
        >>> connect(graph, random.Random(42))
        19
        >>> graph.edge_count
        19
        >>> min(graph.degree(node) for node in graph.nodes())
        1

        Drawing the same tree a second time adds nothing, because every
        edge of it is already there:

        >>> connect(graph, random.Random(42))
        0

        A graph with fewer than two nodes is already connected:

        >>> connect(random_graph(1, density=0.0))
        0
    """
    bounds = _validate_weight_range(weight_range)
    nodes = graph.nodes()
    if len(nodes) < 2:
        return 0
    if rng is None:
        rng = random.Random(seed)

    added = 0
    for parent, child in _random_tree_pairs(nodes, rng):
        if graph.has_edge(parent, child):
            continue
        if graph.weighted:
            graph.add_edge(parent, child, _random_weight(rng, bounds))
        else:
            graph.add_edge(parent, child)
        added += 1
    return added
