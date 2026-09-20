"""Breadth-first traversal of a :class:`~src.graphs.graph.Graph`.

Breadth-first search visits a graph in rings. Everything one hop from the
start comes out before anything two hops away, and so on, which is why BFS
and only BFS hands back shortest paths measured in edges.

THE SPINE. BFS and iterative DFS are the same loop:

    take a node from a pending collection, mark it seen,
    push its unseen neighbours

Nothing else changes between them. Here the pending collection is a
:class:`collections.deque` used as a FIFO queue, popped from the left, and
that single fact is what makes the traversal breadth-first. Swap the deque
for a LIFO stack and the identical loop becomes depth-first; that is exactly
what :mod:`src.graphs.dfs` does. The loop is written out longhand in
:func:`_bfs_order` below rather than shared with the DFS module, so the two
files can be read side by side and the one differing line is visible.

One detail differs between the two and is worth naming, because it is not
cosmetic: BFS marks a node seen when it is *pushed*, DFS when it is
*popped*. Marking on push keeps a node from being queued twice by two
different discoverers, which is what makes the first discovery of a node the
shortest one and keeps the queue at O(V). DFS cannot do that without losing
the order it shares with the recursive form, so it tolerates duplicates on
the stack and filters them on the way out.

Every function walks the adjacency list, which is the primary store in
:class:`~src.graphs.graph.Graph`. Neighbour lookup there is O(1) and the
whole traversal is O(V + E). The same traversal driven off the adjacency
matrix from :meth:`~src.graphs.graph.Graph.to_adjacency_matrix` would cost
O(V^2), because finding one node's neighbours means scanning a full row of V
cells whether or not the edges exist. On the sparse graphs this course
benchmarks, that is the difference between linear and quadratic work.

Disconnected graphs are handled everywhere they should be. :func:`bfs` is a
full traversal: when the queue drains it jumps to the next unvisited node in
insertion order and starts again, so every component is reported.
:func:`bfs_component`, :func:`bfs_levels` and :func:`bfs_tree` deliberately
stay inside the component reachable from the start, because a hop count or a
predecessor for an unreachable node has no meaning.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

from src.graphs.graph import Graph

__all__ = ["bfs", "bfs_component", "bfs_levels", "bfs_tree"]


def _check_start(graph: Graph, start: Any) -> None:
    """Raise :class:`KeyError` unless ``start`` is a node of ``graph``.

    Centralising the check keeps the message identical across all four
    public functions and keeps each traversal body focused on traversing.

    Args:
        graph: The graph the start node must belong to.
        start: The candidate start node.

    Raises:
        KeyError: If ``start`` is not a node of ``graph``.

    Time Complexity:
        O(1) - a dict membership test on the adjacency list. O(1) on the
        matrix representation too, via its node-to-row index.

    Space Complexity:
        O(1).

    Examples:
        >>> g = Graph()
        >>> g.add_edge('a', 'b')
        >>> _check_start(g, 'a') is None
        True
        >>> _check_start(g, 'z')
        Traceback (most recent call last):
            ...
        KeyError: "start node 'z' is not in the graph"
    """
    if start not in graph:
        raise KeyError(f"start node {start!r} is not in the graph")


def _roots(graph: Graph, start: Optional[Any]) -> List[Any]:
    """Return the sequence of nodes to launch traversals from.

    A full traversal needs more than one launch point whenever the graph is
    disconnected. The caller's ``start`` goes first when it is given, and the
    graph's nodes follow in insertion order; already-visited roots are
    skipped by the caller, so listing every node here is harmless and makes
    the rule obvious: start where you were told, then keep picking the
    earliest node nobody has reached yet.

    Insertion order, never sorted order, is what makes the result
    deterministic without requiring the nodes to be comparable at all.

    Args:
        graph: The graph being traversed.
        start: Explicit first root, or None to begin at the first node in
            insertion order.

    Returns:
        A list of candidate roots in the order they should be tried.

    Raises:
        KeyError: If ``start`` is given and is not a node of ``graph``.

    Time Complexity:
        O(V) - one pass to copy the node list.

    Space Complexity:
        O(V).

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (3, 4)]:
        ...     g.add_edge(u, v)
        >>> _roots(g, None)
        [1, 2, 3, 4]
        >>> _roots(g, 3)
        [3, 1, 2, 3, 4]
    """
    nodes = graph.nodes()
    if start is None:
        return nodes
    _check_start(graph, start)
    return [start] + nodes


def _bfs_order(
    graph: Graph,
    root: Any,
    visited: Set[Any],
    order: List[Any],
) -> None:
    """Run one breadth-first sweep from ``root``, appending to ``order``.

    This is THE LOOP. A node is taken from the front of the queue, recorded,
    and each of its unseen neighbours is marked and pushed to the back. The
    deque is the only reason this is breadth-first rather than depth-first.

    ``visited`` and ``order`` are passed in and mutated so that a full
    traversal can sweep one component after another without ever revisiting
    a node, and without concatenating intermediate lists.

    Args:
        graph: The graph to walk.
        root: Node to sweep from. Must already be in ``graph``.
        visited: Set of nodes already seen; updated in place. ``root`` is
            added here, so the caller must not have visited it.
        order: Traversal order so far; extended in place.

    Time Complexity:
        O(V_c + E_c) over the component reached, using the adjacency list:
        each node is dequeued once and each incident edge inspected once.
        O(V * V_c) if driven from an adjacency matrix instead, since each
        node's neighbours cost a full V-cell row scan.

    Space Complexity:
        O(V_c) - the queue holds at most one ring of the graph at a time,
        which is what makes BFS memory-hungry on wide graphs and DFS
        memory-hungry on deep ones.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        ...     g.add_edge(u, v)
        >>> seen, out = set(), []
        >>> _bfs_order(g, 1, seen, out)
        >>> out
        [1, 2, 3, 4]
    """
    queue = deque([root])
    visited.add(root)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.neighbors(node):
            # Marking on push, not on pop: the first discoverer of a node
            # owns it, which is what keeps the hop count minimal and stops
            # the same node being queued by several neighbours at once.
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


def bfs(graph: Graph, start: Optional[Any] = None) -> List[Any]:
    """Breadth-first traversal order across every component of a graph.

    Sweeps the component containing ``start`` first. When the queue drains,
    the graph may still hold nodes nothing has reached, so the traversal
    jumps to the earliest unvisited node in insertion order and sweeps
    again, repeating until no node is unvisited. The result therefore lists
    every node of the graph exactly once, not merely the reachable ones.

    Args:
        graph: The graph to traverse. Directed or undirected, weighted or
            not; weights are irrelevant to BFS and are ignored.
        start: Node to begin at. Defaults to the first node in insertion
            order, which is deterministic without requiring nodes to be
            sortable.

    Returns:
        A list of every node, in breadth-first visiting order.

    Raises:
        KeyError: If ``start`` is given and is not a node of ``graph``.

    Time Complexity:
        O(V + E) with the adjacency list: every node is dequeued once and
        every edge inspected once (twice for an undirected edge, once from
        each endpoint). O(V^2) if the adjacency matrix is scanned instead,
        because each node costs a V-cell row scan even on a sparse graph.

    Space Complexity:
        O(V) - the visited set and the output list are O(V), and the queue
        never holds more than one ring.

    Examples:
        A graph of three components: a diamond, an edge and a lone node.

        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> bfs(g)
        [1, 2, 3, 4, 5, 6, 7]

        The start node only chooses where to begin; every component is still
        reported.

        >>> bfs(g, start=5)
        [5, 6, 1, 2, 3, 4, 7]

        BFS takes the diamond in rings and reaches 4 last; depth-first dives
        1 -> 2 -> 4 and only then backtracks to 3. Same graph, same loop,
        different pending collection:

        >>> from src.graphs.dfs import dfs
        >>> bfs(g)
        [1, 2, 3, 4, 5, 6, 7]
        >>> dfs(g)
        [1, 2, 4, 3, 5, 6, 7]

        Edge cases:

        >>> bfs(Graph())
        []
        >>> bfs(g, start='nope')
        Traceback (most recent call last):
            ...
        KeyError: "start node 'nope' is not in the graph"

        Direction is respected: from 'b' there is nowhere to go, so the
        traversal falls through to the remaining nodes in insertion order.

        >>> d = Graph(directed=True)
        >>> for u, v in [('a', 'b'), ('b', 'c'), ('c', 'a')]:
        ...     d.add_edge(u, v)
        >>> d.add_edge('x', 'a')
        >>> bfs(d, start='b')
        ['b', 'c', 'a', 'x']
    """
    visited: Set[Any] = set()
    order: List[Any] = []

    for root in _roots(graph, start):
        if root not in visited:
            _bfs_order(graph, root, visited, order)

    return order


def bfs_component(graph: Graph, start: Any) -> List[Any]:
    """Breadth-first order over only the component reachable from ``start``.

    The single-component counterpart to :func:`bfs`: one sweep, no jumping
    to unvisited nodes afterwards. Use it to ask what ``start`` can actually
    reach. On a directed graph "reachable" means reachable by following edge
    direction, so the answer is generally smaller than the component you
    would get by ignoring direction.

    Args:
        graph: The graph to traverse.
        start: Node to sweep from. Required.

    Returns:
        A list of the nodes reachable from ``start``, ``start`` first, in
        breadth-first order.

    Raises:
        KeyError: If ``start`` is not a node of ``graph``.

    Time Complexity:
        O(V_c + E_c) over the reached component with the adjacency list;
        O(V * V_c) driven from the adjacency matrix.

    Space Complexity:
        O(V_c).

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> bfs_component(g, 1)
        [1, 2, 3, 4]
        >>> bfs_component(g, 5)
        [5, 6]
        >>> bfs_component(g, 7)
        [7]

        Direction matters: 'a' reaches 'b', but 'b' reaches nothing.

        >>> d = Graph(directed=True)
        >>> d.add_edge('a', 'b')
        >>> bfs_component(d, 'a'), bfs_component(d, 'b')
        (['a', 'b'], ['b'])
    """
    _check_start(graph, start)

    visited: Set[Any] = set()
    order: List[Any] = []
    _bfs_order(graph, start, visited, order)
    return order


def bfs_tree(graph: Graph, start: Any) -> Dict[Any, Optional[Any]]:
    """Map each reachable node to the node BFS discovered it from.

    The predecessor map is the breadth-first tree written down: follow the
    entries back from any node and you walk a shortest path, in edges, to
    ``start``. ``start`` itself maps to None, which is what terminates that
    walk. Unreachable nodes are absent rather than mapped to None, so the
    two cases never look alike.

    This is :func:`_bfs_order` with one extra assignment - the predecessor
    is recorded at the moment a neighbour is first seen, and because BFS
    marks on push, that first sighting is along a shortest path and is never
    revised.

    Args:
        graph: The graph to traverse.
        start: Root of the tree. Required.

    Returns:
        A dict mapping each reachable node to its predecessor, with
        ``start`` mapped to None. Keys are in discovery order, so every
        node appears after its own predecessor.

    Raises:
        KeyError: If ``start`` is not a node of ``graph``.

    Time Complexity:
        O(V_c + E_c) with the adjacency list; O(V * V_c) from the matrix.

    Space Complexity:
        O(V_c) - one dict entry per reachable node.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> bfs_tree(g, 1)
        {1: None, 2: 1, 3: 1, 4: 2}

        Node 4 touches both 2 and 3; 2 was queued first, so 2 is the parent.
        Nodes 5, 6 and 7 are unreachable from 1 and simply do not appear.

        Walking the map backwards reconstructs a shortest path:

        >>> tree = bfs_tree(g, 1)
        >>> path, node = [], 4
        >>> while node is not None:
        ...     path.append(node)
        ...     node = tree[node]
        >>> path[::-1]
        [1, 2, 4]
        >>> bfs_tree(g, 7)
        {7: None}
    """
    _check_start(graph, start)

    parents: Dict[Any, Optional[Any]] = {start: None}
    queue = deque([start])

    # The same loop as _bfs_order. The parents dict doubles as the visited
    # set here, so nothing is tracked twice.
    while queue:
        node = queue.popleft()
        for neighbour in graph.neighbors(node):
            if neighbour not in parents:
                parents[neighbour] = node
                queue.append(neighbour)

    return parents


def bfs_levels(graph: Graph, start: Any) -> Dict[Any, int]:
    """Map each reachable node to its hop count from ``start``.

    The level of a node is its distance from ``start`` counted in edges,
    ignoring any weights the graph carries. That is the unweighted shortest
    path, and BFS gives it for free: a node is first reached along a
    shortest route, so the first level assigned is the right one.

    The levels are derived from :func:`bfs_tree` in a single pass rather
    than by a second traversal. Breadth-first discovery order guarantees a
    node's predecessor is already in the map when the node is read, so one
    forward sweep of ``level[parent] + 1`` is enough. Weighted shortest
    paths need Dijkstra instead - see :mod:`src.graphs.dijkstra`.

    Args:
        graph: The graph to traverse.
        start: Node at level 0. Required.

    Returns:
        A dict mapping each reachable node to its hop count, ``start`` at 0,
        keys in discovery order. Unreachable nodes are absent.

    Raises:
        KeyError: If ``start`` is not a node of ``graph``.

    Time Complexity:
        O(V_c + E_c) with the adjacency list - the traversal dominates; the
        derivation pass is O(V_c). O(V * V_c) from the matrix.

    Space Complexity:
        O(V_c).

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> bfs_levels(g, 1)
        {1: 0, 2: 1, 3: 1, 4: 2}
        >>> bfs_levels(g, 5)
        {5: 0, 6: 1}
        >>> bfs_levels(g, 7)
        {7: 0}

        On a path the level is just the index, and the deepest level is the
        eccentricity of the start node:

        >>> p = Graph()
        >>> for i in range(4):
        ...     p.add_edge(i, i + 1)
        >>> bfs_levels(p, 0)
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        >>> max(bfs_levels(p, 2).values())
        2
    """
    levels: Dict[Any, int] = {}

    for node, parent in bfs_tree(graph, start).items():
        levels[node] = 0 if parent is None else levels[parent] + 1

    return levels
