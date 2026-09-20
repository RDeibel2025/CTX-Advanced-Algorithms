"""Depth-first traversal of a :class:`~src.graphs.graph.Graph`, twice over.

Depth-first search follows one path as far as it goes before backing up and
trying the next branch. Where BFS fans out in rings, DFS commits.

THE SPINE. BFS and iterative DFS are the same loop:

    take a node from a pending collection, mark it seen,
    push its unseen neighbours

Nothing else changes between them. In :mod:`src.graphs.bfs` the pending
collection is a FIFO deque and the traversal comes out breadth-first. Here it
is a plain list used as a LIFO stack, popped from the end, and the identical
loop comes out depth-first. That one substitution is the whole difference,
which is why :func:`_dfs_iterative_order` below is written out longhand
rather than shared with the BFS module: the two loops are meant to be read
side by side.

One detail does differ, and it is not cosmetic. BFS marks a node seen when
it is *pushed*; DFS marks it when it is *popped*. DFS has to wait, because a
node can be pushed by several neighbours before anybody reaches it, and the
push that wins is the last one on the stack, not the first. Marking on pop
and skipping nodes already seen is what makes the iterative order agree with
the recursive order exactly.

RECURSIVE DFS IS THE EXCEPTION. :func:`dfs_recursive` is not that loop at
all. It has no explicit container: it borrows the interpreter's call stack,
and each pending node is a live Python frame rather than a list entry. That
is the cleanest way to write DFS and the only way that cannot be turned into
BFS by swapping a data structure. It also has a cost the iterative form does
not - see MAX_RECURSION_LIMIT below.

EQUIVALENCE. :func:`dfs_iterative` and :func:`dfs_recursive` return the
identical list for the same graph. The classic trap is that a naive stack
visits a node's neighbours in reverse, because the last one pushed is the
first one popped. The fix is to push them in reverse order so the first
neighbour is popped first. The doctests assert the equality directly.

Complexity is the same for both forms and the same as BFS: O(V + E) walking
the adjacency list, which is the primary store in
:class:`~src.graphs.graph.Graph`, and O(V^2) if the traversal is driven off
the adjacency matrix from
:meth:`~src.graphs.graph.Graph.to_adjacency_matrix`, where one node's
neighbours cost a full V-cell row scan whether the edges exist or not.

Disconnected graphs are handled: :func:`dfs`, :func:`dfs_iterative` and
:func:`dfs_recursive` are full traversals that jump to the next unvisited
node in insertion order once a branch is exhausted, so every component is
reported. :func:`dfs_component` deliberately stops at one component.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Set

from src.graphs.graph import Graph

__all__ = [
    "MAX_RECURSION_LIMIT",
    "RECURSION_HEADROOM",
    "dfs",
    "dfs_component",
    "dfs_iterative",
    "dfs_recursive",
]

# Recursive DFS costs one Python frame per node on the current path. A
# path-shaped graph is the worst case: the path IS the graph, so the depth
# reaches V. CPython's default recursion limit is 1000, and the Week 4
# benchmark runs graphs of up to about 10,000 nodes, so the default would
# raise RecursionError partway through a perfectly valid traversal.
#
# The limit is therefore raised deliberately for the duration of the call,
# sized from the node count, and restored in a finally block so no caller
# is left with a modified interpreter. MAX_RECURSION_LIMIT is the ceiling on
# that: past it, recursive DFS refuses to start rather than risking a hard
# interpreter crash, and tells the caller to use dfs_iterative instead. It is
# a module constant so the report can cite the number.
#
# 100,000 is roughly ten times the largest benchmark graph. It is safe on
# CPython 3.11 and newer, where a Python function calling another Python
# function no longer consumes C stack per frame, so a raised limit is a real
# budget rather than a licence to segfault.
MAX_RECURSION_LIMIT: int = 100_000

# Frames left over for whatever called us: pytest, a benchmark harness, the
# helper frames inside this module. Without the slack, a graph of exactly
# MAX_RECURSION_LIMIT nodes would have no room for the caller's own stack.
RECURSION_HEADROOM: int = 1_000


def _check_start(graph: Graph, start: Any) -> None:
    """Raise :class:`KeyError` unless ``start`` is a node of ``graph``.

    Centralising the check keeps the message identical across every public
    function here and in :mod:`src.graphs.bfs`, and keeps each traversal
    body focused on traversing.

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
    disconnected. The caller's ``start`` goes first when it is given, and
    the graph's nodes follow in insertion order; already-visited roots are
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


@contextmanager
def _recursion_limit_for(node_count: int) -> Iterator[None]:
    """Raise the interpreter recursion limit for one traversal, then restore.

    A depth-first path can be as long as the graph has nodes, so the budget
    is sized from ``node_count`` plus RECURSION_HEADROOM frames for the
    caller's own stack. The previous limit is restored in a ``finally``
    block, so an exception mid-traversal cannot leave the interpreter
    reconfigured behind the caller's back.

    The limit is only ever raised, never lowered: if the caller has already
    set something more generous, that setting is left alone for the
    duration.

    Args:
        node_count: Number of nodes in the graph about to be traversed.

    Yields:
        None. The body of the ``with`` runs under the raised limit.

    Raises:
        RecursionError: If the required budget exceeds MAX_RECURSION_LIMIT.
            Raised up front, before any traversal work, so the caller gets a
            clear message instead of a crash partway through.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1) - the frames themselves are charged to the recursion, not here.

    Examples:
        >>> before = sys.getrecursionlimit()
        >>> with _recursion_limit_for(50_000):
        ...     sys.getrecursionlimit() >= 51_000
        True
        >>> sys.getrecursionlimit() == before
        True

        The limit is restored even when the body raises:

        >>> try:
        ...     with _recursion_limit_for(50_000):
        ...         raise ValueError("boom")
        ... except ValueError:
        ...     sys.getrecursionlimit() == before
        True

        Past the cap it refuses rather than risking the interpreter:

        >>> try:
        ...     with _recursion_limit_for(MAX_RECURSION_LIMIT):
        ...         pass
        ... except RecursionError as exc:
        ...     print(type(exc).__name__)
        RecursionError
    """
    required = node_count + RECURSION_HEADROOM
    if required > MAX_RECURSION_LIMIT:
        raise RecursionError(
            f"recursive DFS would need {required} frames for {node_count} "
            f"nodes, over the MAX_RECURSION_LIMIT of {MAX_RECURSION_LIMIT}; "
            f"use dfs_iterative() instead"
        )

    previous = sys.getrecursionlimit()
    try:
        if required > previous:
            sys.setrecursionlimit(required)
        yield
    finally:
        # Unconditional restore: the caller's interpreter is not ours to
        # leave modified, whether the traversal finished or blew up.
        sys.setrecursionlimit(previous)


def _dfs_iterative_order(
    graph: Graph,
    root: Any,
    visited: Set[Any],
    order: List[Any],
) -> None:
    """Run one depth-first sweep from ``root`` using an explicit stack.

    This is THE LOOP, with a LIFO list where :func:`src.graphs.bfs._bfs_order`
    has a FIFO deque. A node is popped from the end of the stack, recorded,
    and its unseen neighbours are pushed - in reverse, so that the first
    neighbour ends up on top and is popped first. Without that reversal the
    traversal still visits every node, but in a mirrored order that does not
    match :func:`dfs_recursive`.

    ``visited`` and ``order`` are passed in and mutated so that a full
    traversal can sweep one component after another without revisiting a
    node and without concatenating intermediate lists.

    Args:
        graph: The graph to walk.
        root: Node to sweep from. Must already be in ``graph``.
        visited: Set of nodes already seen; updated in place.
        order: Traversal order so far; extended in place.

    Time Complexity:
        O(V_c + E_c) over the component reached, using the adjacency list.
        O(V * V_c) if driven from an adjacency matrix instead, since each
        node's neighbours cost a full V-cell row scan.

    Space Complexity:
        O(V_c). The stack can hold a node more than once - a duplicate is
        cheap and is discarded on pop - so it is bounded by the edges of the
        component rather than by one ring, which is why DFS is the
        memory-cheaper choice on wide graphs and the dearer one on deep.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        ...     g.add_edge(u, v)
        >>> seen, out = set(), []
        >>> _dfs_iterative_order(g, 1, seen, out)
        >>> out
        [1, 2, 4, 3]
    """
    stack = [root]

    while stack:
        node = stack.pop()
        # Marking on pop, not on push: a node can sit on the stack several
        # times over, pushed by several neighbours, and only the copy that
        # actually surfaces first decides where it lands in the order.
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # Reversed, so the first neighbour is on top and comes off first.
        # This is the single line that makes the iterative order equal to
        # the recursive one.
        for neighbour in reversed(list(graph.neighbors(node))):
            if neighbour not in visited:
                stack.append(neighbour)


def _dfs_recursive_order(
    graph: Graph,
    node: Any,
    visited: Set[Any],
    order: List[Any],
) -> None:
    """Run one depth-first sweep from ``node`` on the interpreter's stack.

    The container is gone. Each pending node is a live Python frame, and
    "back up and try the next branch" is simply a function returning. The
    neighbours are walked in their natural order, which is why the
    iterative form has to push them reversed to agree.

    The caller is responsible for entering :func:`_recursion_limit_for`
    first; this function assumes the budget is already in place.

    Args:
        graph: The graph to walk.
        node: Node to visit now. Must already be in ``graph``.
        visited: Set of nodes already seen; updated in place.
        order: Traversal order so far; extended in place.

    Raises:
        RecursionError: If the path is deeper than the limit currently in
            force. Callers avoid this by using :func:`_recursion_limit_for`.

    Time Complexity:
        O(V_c + E_c) over the component reached, using the adjacency list;
        O(V * V_c) driven from an adjacency matrix.

    Space Complexity:
        O(V_c) call frames, which is a real interpreter resource rather than
        heap the way the iterative stack is.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        ...     g.add_edge(u, v)
        >>> seen, out = set(), []
        >>> _dfs_recursive_order(g, 1, seen, out)
        >>> out
        [1, 2, 4, 3]
    """
    visited.add(node)
    order.append(node)

    for neighbour in graph.neighbors(node):
        if neighbour not in visited:
            _dfs_recursive_order(graph, neighbour, visited, order)


def dfs_iterative(graph: Graph, start: Optional[Any] = None) -> List[Any]:
    """Depth-first traversal order across every component, using a stack.

    Dives from ``start`` along one branch until it dead-ends, backs up to
    the most recent unexplored fork, and carries on. When the stack empties,
    the graph may still hold nodes nothing has reached, so the traversal
    jumps to the earliest unvisited node in insertion order and sweeps
    again, until no node is unvisited. The result lists every node of the
    graph exactly once.

    The pending collection is an ordinary Python list used as a LIFO stack.
    Replace it with a FIFO deque and this function becomes
    :func:`src.graphs.bfs.bfs`, unchanged in every other respect.

    Args:
        graph: The graph to traverse. Directed or undirected, weighted or
            not; weights are irrelevant to DFS and are ignored.
        start: Node to begin at. Defaults to the first node in insertion
            order, which is deterministic without requiring nodes to be
            sortable.

    Returns:
        A list of every node, in depth-first visiting order. Identical to
        :func:`dfs_recursive` on the same graph.

    Raises:
        KeyError: If ``start`` is given and is not a node of ``graph``.

    Time Complexity:
        O(V + E) with the adjacency list: every node is recorded once and
        every edge inspected once (twice for an undirected edge, once from
        each endpoint). O(V^2) if the adjacency matrix is scanned instead.

    Space Complexity:
        O(V) for the visited set and the output list; the stack is O(E) in
        the worst case because a node may be pushed once per incoming edge.

    Examples:
        A graph of three components: a diamond, an edge and a lone node.

        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> dfs_iterative(g)
        [1, 2, 4, 3, 5, 6, 7]

        It dives 1 -> 2 -> 4 and only then backtracks to 3. Breadth-first
        takes the same diamond in rings and reaches 4 last:

        >>> from src.graphs.bfs import bfs
        >>> bfs(g)
        [1, 2, 3, 4, 5, 6, 7]

        The start node only chooses where to begin; every component is
        still reported.

        >>> dfs_iterative(g, start=5)
        [5, 6, 1, 2, 4, 3, 7]

        Edge cases:

        >>> dfs_iterative(Graph())
        []
        >>> dfs_iterative(g, start='nope')
        Traceback (most recent call last):
            ...
        KeyError: "start node 'nope' is not in the graph"
    """
    visited: Set[Any] = set()
    order: List[Any] = []

    for root in _roots(graph, start):
        if root not in visited:
            _dfs_iterative_order(graph, root, visited, order)

    return order


def dfs_recursive(graph: Graph, start: Optional[Any] = None) -> List[Any]:
    """Depth-first traversal across every component, on the call stack.

    Same traversal and same result as :func:`dfs_iterative`, written the way
    the textbook writes it: no explicit container, just a function that
    visits a node and calls itself on each unvisited neighbour. Backtracking
    is a ``return``. Between components the recursion unwinds completely and
    a fresh call starts at the next unvisited node in insertion order.

    RECURSION DEPTH. One frame is spent per node on the current path, and a
    path-shaped graph makes that path as long as the graph, so V frames.
    CPython allows 1000 by default and the Week 4 benchmark runs graphs of
    about 10,000 nodes, which would raise RecursionError partway through a
    valid traversal. Rather than let that happen, the limit is raised for
    the duration of the call, sized as ``node_count + RECURSION_HEADROOM``,
    and restored in a ``finally`` block - see :func:`_recursion_limit_for`.
    Graphs needing more than MAX_RECURSION_LIMIT frames are refused up front
    with a RecursionError naming :func:`dfs_iterative` as the way out.

    Args:
        graph: The graph to traverse.
        start: Node to begin at. Defaults to the first node in insertion
            order.

    Returns:
        A list of every node, in depth-first visiting order. Guaranteed
        equal to :func:`dfs_iterative` on the same graph.

    Raises:
        KeyError: If ``start`` is given and is not a node of ``graph``.
        RecursionError: If the graph has more than
            ``MAX_RECURSION_LIMIT - RECURSION_HEADROOM`` nodes. Use
            :func:`dfs_iterative`, which has no depth limit beyond memory.

    Time Complexity:
        O(V + E) with the adjacency list; O(V^2) from the adjacency matrix.
        The constant factor is worse than the iterative form's, because a
        Python function call is dearer than a list append and pop.

    Space Complexity:
        O(V) - but in call frames, not heap. That is the real difference
        between the two forms, and the reason the iterative one is what the
        benchmark drives at scale.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> dfs_recursive(g)
        [1, 2, 4, 3, 5, 6, 7]

        THE EQUIVALENCE, asserted directly - the two forms agree on the
        whole traversal, on every start node, and on a directed graph too:

        >>> dfs_recursive(g) == dfs_iterative(g)
        True
        >>> all(dfs_recursive(g, n) == dfs_iterative(g, n) for n in g.nodes())
        True
        >>> d = Graph(directed=True)
        >>> for u, v in [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')]:
        ...     d.add_edge(u, v)
        >>> dfs_recursive(d)
        ['a', 'b', 'd', 'c']
        >>> dfs_recursive(d) == dfs_iterative(d)
        True

        A 3,001-node path recurses 3,001 deep, well past CPython's default
        limit of 1000, and the limit is put back afterwards:

        >>> before = sys.getrecursionlimit()
        >>> path = Graph()
        >>> for i in range(3000):
        ...     path.add_edge(i, i + 1)
        >>> dfs_recursive(path) == list(range(3001))
        True
        >>> sys.getrecursionlimit() == before
        True

        Edge cases:

        >>> dfs_recursive(Graph())
        []
        >>> dfs_recursive(g, start='nope')
        Traceback (most recent call last):
            ...
        KeyError: "start node 'nope' is not in the graph"
    """
    roots = _roots(graph, start)

    visited: Set[Any] = set()
    order: List[Any] = []

    with _recursion_limit_for(len(graph)):
        for root in roots:
            if root not in visited:
                _dfs_recursive_order(graph, root, visited, order)

    return order


def dfs(graph: Graph, start: Optional[Any] = None) -> List[Any]:
    """Depth-first traversal order across every component of a graph.

    The plain name for the algorithm, delegating to :func:`dfs_iterative`.
    The iterative form is the default because it is the one that scales: it
    has no recursion limit to manage and no per-node Python frame, so it is
    what the Week 4 benchmark drives at 10,000 nodes. :func:`dfs_recursive`
    is kept alongside it, returns the identical list, and is the clearer
    read when the point is the algorithm rather than the throughput.

    Args:
        graph: The graph to traverse.
        start: Node to begin at. Defaults to the first node in insertion
            order.

    Returns:
        A list of every node, in depth-first visiting order.

    Raises:
        KeyError: If ``start`` is given and is not a node of ``graph``.

    Time Complexity:
        O(V + E) with the adjacency list; O(V^2) from the adjacency matrix.

    Space Complexity:
        O(V) for the visited set and output; O(E) for the stack worst case.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> dfs(g)
        [1, 2, 4, 3, 5, 6, 7]
        >>> dfs(g) == dfs_iterative(g) == dfs_recursive(g)
        True

        BFS and DFS on the same three-component graph, for contrast:

        >>> from src.graphs.bfs import bfs
        >>> bfs(g)
        [1, 2, 3, 4, 5, 6, 7]
        >>> dfs(g)
        [1, 2, 4, 3, 5, 6, 7]
    """
    return dfs_iterative(graph, start)


def dfs_component(
    graph: Graph,
    start: Any,
    recursive: bool = False,
) -> List[Any]:
    """Depth-first order over only the component reachable from ``start``.

    The single-component counterpart to :func:`dfs`: one sweep, no jumping
    to unvisited nodes afterwards. Use it to ask what ``start`` can actually
    reach. On a directed graph "reachable" means reachable by following edge
    direction, so the answer is generally smaller than the component you
    would get by ignoring direction.

    Args:
        graph: The graph to traverse.
        start: Node to sweep from. Required.
        recursive: Use the call-stack form instead of the explicit stack.
            The returned list is identical either way; the flag exists so
            that tests and benchmarks can exercise both implementations
            through one entry point.

    Returns:
        A list of the nodes reachable from ``start``, ``start`` first, in
        depth-first order.

    Raises:
        KeyError: If ``start`` is not a node of ``graph``.
        RecursionError: If ``recursive`` is true and the graph has more than
            ``MAX_RECURSION_LIMIT - RECURSION_HEADROOM`` nodes. The budget
            is sized from the whole graph, not the component, because the
            component's size is not known until it has been walked.

    Time Complexity:
        O(V_c + E_c) over the reached component with the adjacency list;
        O(V * V_c) driven from the adjacency matrix.

    Space Complexity:
        O(V_c), in heap for the iterative form and in call frames for the
        recursive one.

    Examples:
        >>> g = Graph()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        ...     g.add_edge(u, v)
        >>> g.add_node(7)
        >>> dfs_component(g, 1)
        [1, 2, 4, 3]
        >>> dfs_component(g, 1, recursive=True)
        [1, 2, 4, 3]
        >>> dfs_component(g, 5)
        [5, 6]
        >>> dfs_component(g, 7)
        [7]

        The two forms agree on every component of the graph:

        >>> all(
        ...     dfs_component(g, n) == dfs_component(g, n, recursive=True)
        ...     for n in g.nodes()
        ... )
        True

        Direction matters: 'a' reaches 'b', but 'b' reaches nothing.

        >>> d = Graph(directed=True)
        >>> d.add_edge('a', 'b')
        >>> dfs_component(d, 'a'), dfs_component(d, 'b')
        (['a', 'b'], ['b'])
    """
    _check_start(graph, start)

    visited: Set[Any] = set()
    order: List[Any] = []

    if recursive:
        with _recursion_limit_for(len(graph)):
            _dfs_recursive_order(graph, start, visited, order)
    else:
        _dfs_iterative_order(graph, start, visited, order)

    return order
