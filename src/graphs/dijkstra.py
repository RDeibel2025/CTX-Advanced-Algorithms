"""Dijkstra's single-source shortest paths, on the Week 3 priority queue.

Dijkstra's algorithm is breadth-first search with the container swapped
out. BFS takes the node that is fewest *edges* from the source; Dijkstra
takes the node that is least *weight* from the source, and the only
structural change is that the pending collection is ordered by tentative
distance instead of by arrival. That is why this module imports
:class:`~src.structures.heap.PriorityQueue` from Week 3 rather than
``heapq``: the container is the algorithm, and this container was built,
tested and benchmarked last week.

Where the analogy stops is relaxation. BFS discovers a node once and is
finished with it. Dijkstra can find a cheaper route to a node that is
already sitting in the queue, so it needs to lower that node's priority -
and a binary heap has no cheap way to reach into its middle and do so.

Lazy deletion
-------------
:class:`PriorityQueue` offers no decrease-key, so this module uses the
standard workaround: an improved distance is *pushed as a second entry*
instead of being edited in place, and on pop an entry whose recorded
priority is worse than the distance already stored for that node is
discarded as stale. The price of that choice is real, and the Week 4
benchmark measures it:

* The queue holds up to O(E) entries rather than O(V), so working memory
  is O(V + E) instead of O(V).
* The time bound survives anyway. Each of the at most E entries costs
  O(log E) to insert and to remove, and log E <= log V^2 = 2 log V, so the
  duplicates move the constant factor and not the exponent. The algorithm
  is still O((V + E) log V), just with a heavier constant and a larger
  heap than a decrease-key implementation would carry.

Two implementations
-------------------
The two functions below are deliberately identical apart from the
frontier, so that a benchmark run prices the container and nothing else:

====================== ================= =================
Function               Frontier          Time
====================== ================= =================
dijkstra               binary heap       O((V + E) log V)
dijkstra_linear_scan   plain Python list O(V^2 + E)
====================== ================= =================

They agree on every graph, distances *and* predecessors. That is not an
accident of the inputs: the linear scan breaks ties on the same
``(distance, arrival order)`` key the priority queue uses internally, so
when several nodes sit at the same tentative distance both settle them in
the same order and both record the same parent.

Two further decisions worth naming:

* **Negative weights are rejected before any relaxation runs**, not when
  one is encountered. Dijkstra is undefined for them, and an unreachable
  negative edge is still a broken input the caller should hear about.
* **Every node of the graph appears in the returned distance map.** An
  unreachable node comes back as ``math.inf`` with predecessor ``None``
  rather than being silently absent, so callers never have to distinguish
  "no path" from "no key".

All costs above assume the adjacency-list representation, where the
neighbours of a settled node are read in O(1) each. Driving the same
algorithm from an adjacency matrix would force an O(V) row scan per
settled node, which alone is O(V^2) and would erase the heap's advantage
entirely.

Examples:
    >>> from src.graphs.graph import Graph
    >>> graph = Graph(weighted=True)
    >>> for u, v, w in [("A", "B", 4.0), ("A", "C", 2.0), ("B", "C", 1.0)]:
    ...     graph.add_edge(u, v, w)
    >>> shortest_path(graph, "A", "B")
    (['A', 'C', 'B'], 3.0)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.graphs.graph import Graph
from src.structures.heap import PriorityQueue

__all__ = [
    "dijkstra",
    "dijkstra_linear_scan",
    "reconstruct_path",
    "shortest_path",
]


def _require_non_negative_weights(graph: Graph) -> None:
    """Raise :class:`ValueError` if any edge of ``graph`` has a negative weight.

    Dijkstra's correctness rests on one assumption: extending a path never
    shortens it. A negative edge breaks that assumption, and the algorithm
    does not notice - it settles a node, moves on, and returns a number
    that looks like a distance but is not one. Failing loudly is the only
    honest option.

    The whole edge set is checked once, before any relaxation, rather than
    lazily as edges are met. An unreachable negative edge would never be
    touched by the search, but it is still a broken input, and reporting it
    makes the check deterministic: the same graph always raises or always
    does not, whatever source the caller picked.

    An unweighted graph cannot hold a negative weight, because
    :meth:`Graph.add_edge` rejects any weight other than the 1.0 default on
    one, so the scan is skipped for it.

    Args:
        graph: The graph to check. Not modified.

    Returns:
        None. This helper is called for its exception, not its value.

    Raises:
        ValueError: On the first negative weight found, naming the edge it
            belongs to.

    Time Complexity:
        O(V + E) on the adjacency list - every node's neighbour dictionary
        is walked once; O(1) on an unweighted graph, which is skipped.
        Reading the same edges from an adjacency matrix would be O(V^2).

    Space Complexity:
        O(1) - one edge is held at a time; nothing is accumulated.

    Examples:
        >>> graph = Graph(weighted=True)
        >>> graph.add_edge("A", "B", 3.0)
        >>> _require_non_negative_weights(graph) is None
        True
        >>> graph.add_edge("B", "C", -1.0)
        >>> _require_non_negative_weights(graph)
        Traceback (most recent call last):
            ...
        ValueError: negative weight -1.0 on edge ('B', 'C'); weights must be >= 0
    """
    if not graph.weighted:
        return

    for node in graph.nodes():
        for neighbour, weight in graph.neighbor_items(node):
            if weight < 0:
                raise ValueError(
                    f"negative weight {weight!r} on edge "
                    f"({node!r}, {neighbour!r}); weights must be >= 0"
                )


def _initialize_tables(
    graph: Graph,
    source: Any,
) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    """Validate the arguments and build the starting distance and parent tables.

    Both implementations begin from exactly the same state, so they share
    this one function. Every node of the graph is entered into both tables
    up front, which is what allows an unreachable node to come back as
    ``math.inf`` with predecessor ``None`` instead of being missing.

    The source starts at ``0.0`` rather than ``0`` on purpose: every other
    distance is derived from it by addition, so a float seed makes the
    whole table floats no matter what numeric type the edge weights are.

    Args:
        graph: The graph to be searched. Not modified.
        source: The node all distances are measured from.

    Returns:
        A ``(distances, predecessors)`` pair. Both are keyed by every node
        of the graph, in the graph's insertion order.

    Raises:
        KeyError: If ``source`` is not a node of ``graph``.
        ValueError: If any edge of ``graph`` carries a negative weight.

    Time Complexity:
        O(V + E) - O(V) to fill the two tables, O(E) for the weight scan.

    Space Complexity:
        O(V) - two dictionaries holding one entry per node.

    Examples:
        >>> graph = Graph(weighted=True)
        >>> graph.add_edge("A", "B", 1.5)
        >>> _initialize_tables(graph, "A")
        ({'A': 0.0, 'B': inf}, {'A': None, 'B': None})
        >>> _initialize_tables(graph, "Z")
        Traceback (most recent call last):
            ...
        KeyError: "source node 'Z' is not in the graph"
    """
    if source not in graph:
        raise KeyError(f"source node {source!r} is not in the graph")

    _require_non_negative_weights(graph)

    nodes = graph.nodes()
    distances: Dict[Any, float] = {node: math.inf for node in nodes}
    predecessors: Dict[Any, Optional[Any]] = {node: None for node in nodes}
    distances[source] = 0.0
    return distances, predecessors


def dijkstra(
    graph: Graph,
    source: Any,
) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    """Find the shortest distance from ``source`` to every node, using a heap.

    The loop is the familiar one: take the nearest unsettled node out of
    the frontier, and relax each of its edges - if going through this node
    reaches a neighbour more cheaply than anything found so far, record the
    new distance and record this node as the neighbour's parent. Because
    the frontier is ordered by distance and no weight is negative, the node
    that comes out is always final, which is what makes one pass enough.

    The frontier is the Week 3 :class:`PriorityQueue`. It has no
    decrease-key, so an improved distance is pushed as an extra entry and
    a stale entry - one whose recorded priority is worse than the distance
    now stored for its node - is skipped when it surfaces. See the module
    docstring for what that lazy deletion costs.

    Both returned maps cover the whole graph. A node with no path from the
    source keeps ``math.inf`` and a predecessor of ``None``; that is a
    normal result, not an error.

    Args:
        graph: The graph to search. Weighted or unweighted, directed or
            undirected. An unweighted graph behaves as though every edge
            weighs 1.0, which makes the result a hop count. Not modified.
        source: The node to measure from.

    Returns:
        A ``(distances, predecessors)`` pair. ``distances`` maps every node
        to its shortest distance from ``source`` as a float, ``math.inf``
        when unreachable. ``predecessors`` maps every node to the node
        before it on that shortest path, and ``None`` for the source and
        for unreachable nodes. Pass it to :func:`reconstruct_path` to turn
        it back into a route.

    Raises:
        KeyError: If ``source`` is not a node of ``graph``.
        ValueError: If any edge carries a negative weight. Dijkstra is
            undefined for negative weights, and returning a plausible
            wrong number is worse than refusing to answer.

    Time Complexity:
        O((V + E) log V) on the adjacency list. Every edge can trigger at
        most one push, so at most E + 1 entries pass through the heap at
        O(log E) = O(log V^2) = O(2 log V) each, and each of the V nodes is
        settled once. From an adjacency matrix it degrades to O(V^2),
        because finding a settled node's neighbours means scanning a row.

    Space Complexity:
        O(V + E) - O(V) for the two tables, plus a heap that holds up to
        one entry per edge under lazy deletion. A decrease-key heap would
        make the second term O(V).

    Examples:
        A weighted graph small enough to check by hand. From A, the cheap
        route to B is A-C-B at 2 + 1 = 3, not the direct A-B at 4, and
        that detour then shortens everything behind B:

        >>> graph = Graph(weighted=True)
        >>> for u, v, w in [("A", "B", 4.0), ("A", "C", 2.0),
        ...                 ("B", "C", 1.0), ("B", "D", 5.0),
        ...                 ("C", "D", 8.0), ("C", "E", 10.0),
        ...                 ("D", "E", 2.0)]:
        ...     graph.add_edge(u, v, w)
        >>> graph.add_node("Z")
        >>> distances, predecessors = dijkstra(graph, "A")
        >>> distances
        {'A': 0.0, 'B': 3.0, 'C': 2.0, 'D': 8.0, 'E': 10.0, 'Z': inf}
        >>> predecessors
        {'A': None, 'B': 'C', 'C': 'A', 'D': 'B', 'E': 'D', 'Z': None}

        Z is in its own component, so it reports an infinite distance
        rather than raising:

        >>> distances["Z"]
        inf

        An unweighted graph measures hops:

        >>> hops = Graph()
        >>> for u, v in [("A", "B"), ("B", "C"), ("C", "D")]:
        ...     hops.add_edge(u, v)
        >>> dijkstra(hops, "A")[0]
        {'A': 0.0, 'B': 1.0, 'C': 2.0, 'D': 3.0}

        A negative weight is refused, and the message names the edge:

        >>> broken = Graph(weighted=True)
        >>> broken.add_edge("A", "B", -1.0)
        >>> dijkstra(broken, "A")
        Traceback (most recent call last):
            ...
        ValueError: negative weight -1.0 on edge ('A', 'B'); weights must be >= 0

        A source outside the graph is a programming error, not a result:

        >>> dijkstra(graph, "Q")
        Traceback (most recent call last):
            ...
        KeyError: "source node 'Q' is not in the graph"
    """
    distances, predecessors = _initialize_tables(graph, source)

    queue: PriorityQueue = PriorityQueue()
    queue.push(source, 0.0)

    while not queue.is_empty():
        settled_distance, node = queue.pop_with_priority()

        # LAZY DELETION. A node is pushed once per improvement, so the heap
        # may hold several entries for it. The first to surface carries its
        # final distance; any later entry is stale and is dropped here. This
        # is the stand-in for the decrease-key the binary heap cannot offer.
        if settled_distance > distances[node]:
            continue

        for neighbour, weight in graph.neighbor_items(node):
            candidate = settled_distance + weight
            # Strict <: a tie leaves the earlier parent in place, which is
            # what keeps the result stable and matches the linear scan.
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                predecessors[neighbour] = node
                queue.push(neighbour, candidate)

    return distances, predecessors


def dijkstra_linear_scan(
    graph: Graph,
    source: Any,
) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    """Find the same shortest distances with a linearly scanned list frontier.

    This is the textbook O(V^2) form and the control in the Week 4
    benchmark. The algorithm is character for character the algorithm in
    :func:`dijkstra`; only the frontier changes. Instead of a heap that
    hands back its minimum in O(log V), the unsettled nodes sit in a plain
    list that is scanned end to end on every iteration to find the nearest
    one. There is no queue to go stale, so there is no lazy deletion here
    either - a distance is simply overwritten in place, which is the
    decrease-key the heap could not provide.

    Measuring the two against each other is the point: they share the same
    edge relaxations, so whatever separates their running times is the cost
    of the container and nothing else.

    To make that comparison exact the scan keys on ``(distance, arrival)``,
    where ``arrival`` counts the relaxations in order. That is the same
    tuple :class:`PriorityQueue` compares internally, so nodes tied at the
    same distance are settled in the same order by both functions, and the
    two return identical predecessor maps and not merely equally short
    paths.

    Args:
        graph: The graph to search. Not modified.
        source: The node to measure from.

    Returns:
        A ``(distances, predecessors)`` pair, identical in every entry to
        what :func:`dijkstra` returns for the same arguments.

    Raises:
        KeyError: If ``source`` is not a node of ``graph``.
        ValueError: If any edge carries a negative weight.

    Time Complexity:
        O(V^2 + E) - V selections, each scanning a list that starts at
        length V, plus one relaxation per edge. The scan dominates unless
        the graph is dense enough that E approaches V^2, at which point
        the two terms meet and the heap's advantage disappears; that
        crossover is what the benchmark plot is looking for.

    Space Complexity:
        O(V) - the two tables, the arrival counter map and one list of
        unsettled nodes. Strictly smaller than the heap version, which
        carries up to O(E) queued entries.

    Examples:
        The same hand-checked graph as :func:`dijkstra`, with the same
        answer:

        >>> graph = Graph(weighted=True)
        >>> for u, v, w in [("A", "B", 4.0), ("A", "C", 2.0),
        ...                 ("B", "C", 1.0), ("B", "D", 5.0),
        ...                 ("C", "D", 8.0), ("C", "E", 10.0),
        ...                 ("D", "E", 2.0)]:
        ...     graph.add_edge(u, v, w)
        >>> distances, predecessors = dijkstra_linear_scan(graph, "A")
        >>> distances
        {'A': 0.0, 'B': 3.0, 'C': 2.0, 'D': 8.0, 'E': 10.0}
        >>> predecessors
        {'A': None, 'B': 'C', 'C': 'A', 'D': 'B', 'E': 'D'}

        Agreement with the heap version, down to the predecessors, is the
        property the benchmark depends on:

        >>> dijkstra_linear_scan(graph, "A") == dijkstra(graph, "A")
        True

        Unreachable nodes are handled the same way, and the scan stops
        early once nothing finite is left to expand:

        >>> graph.add_edge("Y", "Z", 1.0)
        >>> dijkstra_linear_scan(graph, "A")[0]["Z"]
        inf
    """
    distances, predecessors = _initialize_tables(graph, source)

    # Mirrors the (priority, insertion order) tuple PriorityQueue compares,
    # so that ties are broken identically in both implementations.
    arrival: Dict[Any, int] = {source: 0}
    relaxations = 1

    unsettled: List[Any] = list(graph.nodes())
    while unsettled:
        best_index = -1
        best_key: Tuple[float, int] = (math.inf, 0)
        for index, pending in enumerate(unsettled):
            distance = distances[pending]
            if distance == math.inf:
                continue
            key = (distance, arrival[pending])
            if best_index == -1 or key < best_key:
                best_index = index
                best_key = key

        # Nothing finite is left, so every node still in the list is
        # unreachable and keeps its inf distance and None predecessor.
        if best_index == -1:
            break

        node = unsettled.pop(best_index)
        settled_distance = distances[node]

        for neighbour, weight in graph.neighbor_items(node):
            candidate = settled_distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                predecessors[neighbour] = node
                arrival[neighbour] = relaxations
                relaxations += 1

    return distances, predecessors


def reconstruct_path(
    predecessors: Dict[Any, Optional[Any]],
    source: Any,
    target: Any,
) -> List[Any]:
    """Rebuild the route from ``source`` to ``target`` out of a predecessor map.

    A predecessor map is a shortest-path tree stored upside down: each node
    remembers only the node before it. Walking those links from the target
    therefore produces the path backwards, and one reversal at the end puts
    it the right way round. Storing it this way costs O(V) for all V paths
    at once, where keeping the paths themselves would cost O(V^2).

    The walk stops when it reaches ``source``. If it runs into ``None``
    first, the target has no route back to this source, and the answer is
    the empty list - the caller gets one unambiguous "no path" value rather
    than a partial route that looks real.

    Args:
        predecessors: The second element of a :func:`dijkstra` result, or
            any map of node to the node before it.
        source: The node the path should start at.
        target: The node the path should end at.

    Returns:
        The nodes from ``source`` to ``target`` inclusive, in order.
        ``[source]`` when the two are equal, and ``[]`` when ``target`` is
        unreachable or absent from the map.

    Time Complexity:
        O(L) for a path of L nodes, which is O(V) in the worst case of a
        path that runs through the entire graph. The final reversal is
        O(L) as well.

    Space Complexity:
        O(L) - the path itself, plus a set of the same size used to stop a
        malformed map with a cycle in it from looping forever.

    Examples:
        >>> predecessors = {"A": None, "B": "C", "C": "A", "D": "B"}
        >>> reconstruct_path(predecessors, "A", "D")
        ['A', 'C', 'B', 'D']
        >>> reconstruct_path(predecessors, "A", "C")
        ['A', 'C']

        A node is its own path:

        >>> reconstruct_path(predecessors, "A", "A")
        ['A']

        An unreachable target, recorded by :func:`dijkstra` as a ``None``
        predecessor, gives the empty list:

        >>> reconstruct_path({"A": None, "X": None}, "A", "X")
        []

        So does a target the map has never heard of:

        >>> reconstruct_path(predecessors, "A", "Q")
        []
    """
    if source == target:
        return [source]

    path: List[Any] = []
    seen = set()
    node: Optional[Any] = target

    while node is not None and node not in seen:
        seen.add(node)
        path.append(node)
        if node == source:
            path.reverse()
            return path
        node = predecessors.get(node)

    # Either the walk hit a None predecessor without meeting the source, or
    # a malformed map sent it round a cycle. Both mean "no path".
    return []


def shortest_path(
    graph: Graph,
    source: Any,
    target: Any,
) -> Tuple[List[Any], float]:
    """Return the cheapest route between two nodes and what it costs.

    A convenience wrapper over :func:`dijkstra` and
    :func:`reconstruct_path` for the common single-pair question. Note that
    it does no less work than the full single-source search: Dijkstra has
    no way to find one target's distance without settling every node closer
    than the target, so asking for one path computes most of them anyway.
    Call :func:`dijkstra` directly if several targets are wanted from the
    same source, and reuse its predecessor map.

    Args:
        graph: The graph to search. Not modified.
        source: The node the route starts at.
        target: The node the route ends at.

    Returns:
        A ``(path, cost)`` pair. ``path`` lists the nodes from ``source``
        to ``target`` inclusive and ``cost`` is the sum of the weights
        along it. ``([], math.inf)`` when no route exists, and
        ``([source], 0.0)`` when the two nodes are the same.

    Raises:
        KeyError: If ``source`` or ``target`` is not a node of ``graph``.
            An unreachable node is a result; a node that does not exist is
            a mistake, and the two are reported differently on purpose.
        ValueError: If any edge carries a negative weight.

    Time Complexity:
        O((V + E) log V) - the full single-source search, plus O(V) to
        walk the path back out.

    Space Complexity:
        O(V + E) - the same as :func:`dijkstra`, whose result it builds on.

    Examples:
        >>> graph = Graph(weighted=True)
        >>> for u, v, w in [("A", "B", 4.0), ("A", "C", 2.0),
        ...                 ("B", "C", 1.0), ("B", "D", 5.0),
        ...                 ("C", "D", 8.0), ("C", "E", 10.0),
        ...                 ("D", "E", 2.0)]:
        ...     graph.add_edge(u, v, w)
        >>> shortest_path(graph, "A", "E")
        (['A', 'C', 'B', 'D', 'E'], 10.0)
        >>> shortest_path(graph, "A", "A")
        (['A'], 0.0)

        An isolated node is unreachable rather than an error:

        >>> graph.add_node("Z")
        >>> shortest_path(graph, "A", "Z")
        ([], inf)

        A node that is not in the graph at all is an error:

        >>> shortest_path(graph, "A", "Q")
        Traceback (most recent call last):
            ...
        KeyError: "target node 'Q' is not in the graph"
    """
    if source not in graph:
        raise KeyError(f"source node {source!r} is not in the graph")
    if target not in graph:
        raise KeyError(f"target node {target!r} is not in the graph")

    distances, predecessors = dijkstra(graph, source)
    path = reconstruct_path(predecessors, source, target)
    if not path:
        return [], math.inf
    return path, distances[target]
