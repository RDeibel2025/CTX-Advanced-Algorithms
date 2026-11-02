"""Edmonds-Karp maximum flow: read the shortest s-t path from the residual graph.

The algorithm
-------------
Ford-Fulkerson repeats one step: find an s-t path in the residual graph, push
as much as its tightest edge allows, and stop when no such path is left.
Edmonds-Karp fixes the one choice Ford-Fulkerson leaves open by always taking a
*shortest* path, counted in edges, found with breadth-first search.

That choice is what bounds the running time. Shortest-path distances from s
never decrease as flow is pushed, and each augmentation saturates at least one
edge on its path. An edge can only be saturated again after the distance to its
tail has grown by at least 2, so each of the O(E) residual edges is critical
O(V) times. That gives O(V E) augmenting paths at O(E) per BFS, O(V E^2) in all,
whatever the capacities are.

How it reads the residual graph
-------------------------------
Edmonds-Karp reads one thing from the residual graph: the shortest path from s
to t through edges with ``cap[e] > 0``. The search does not distinguish forward
edges from reverse edges, and that is the point. Crossing a reverse edge undoes
flow pushed by an earlier path, which is how an early choice gets corrected.

Pushing d units along edge ``e`` is ``cap[e] -= d; cap[e ^ 1] += d``, so each
path edge changes two capacities and ``residual_updates`` grows by
2 per path edge. No flow is stored anywhere; :meth:`Network.flow_matrix` reads
it back from ``cap`` at the end.

The loop stops when the BFS cannot reach t. That failed search is also the proof
of optimality: its visited nodes are the source side of a cut with every
outgoing edge saturated and every incoming edge empty, so the flow value equals
the cut capacity and neither can improve. :func:`min_cut` reads that same set
after any solver, and it is re-exported here.

Implementation
--------------
This is pure Python, so the search is written for speed. ``to``, ``cap`` and
``adj`` are bound to locals. ``parent_edge[v]`` is a preallocated list holding
the edge that discovered v, from which the path is walked back with
``to[e ^ 1]``. ``seen[v] == stamp`` marks v as visited in the current BFS, so
starting a new search is one increment rather than an O(V) clear, which matters
because the search exits as soon as t is discovered.

Examples:
    >>> edges = [(0, 1, 16), (0, 2, 13), (2, 1, 4), (1, 3, 12), (3, 2, 9),
    ...          (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    >>> net = Network.from_edges(6, edges)
    >>> result = edmonds_karp(net, 0, 5)
    >>> result.value, result.bottlenecks
    (23, [12, 4, 7])
    >>> sorted(min_cut(net, 0))
    [0, 1, 2, 4]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List

from src.flow_network_solver.core.network import FlowResult, Network, min_cut

__all__ = ["edmonds_karp", "min_cut"]


def edmonds_karp(net: Network, s: int, t: int, record: bool = False) -> FlowResult:
    """Compute a maximum s-t flow by repeated shortest augmenting paths.

    Each round runs a BFS from s over edges with residual capacity, stopping as
    soon as t is discovered, then pushes the path's bottleneck. The run ends
    when a BFS cannot reach t.

    The network is taken as it stands: ``cap`` is mutated in place and the final
    residual is left there for :func:`min_cut` and the validators. To run
    another solver on the same graph, call :meth:`Network.reset` first. On a net
    that already carries flow, ``value`` and the counters describe only what
    this call added, while ``flow`` reports the whole flow in the net.

    Args:
        net: The network. Its capacities are mutated.
        s: The source node.
        t: The sink node.
        record: If ``True``, fill ``steps`` with one dict per augmenting path:
            ``{"path": [nodes], "edges": [edge indices], "bottleneck": int,
            "value": running value, "flow": flow matrix after this
            augmentation}``, both lists in s-to-t order. Each step copies the
            flow matrix, so keep this to small graphs.

    Returns:
        A :class:`FlowResult` with ``value``, ``flow``, ``iterations`` (the
        number of augmenting paths), ``bottlenecks`` (one per path, in order),
        ``residual_updates`` (entries of ``cap`` changed, 2 per path edge),
        ``time_s`` (covering the search and the final flow matrix, plus the
        recording when ``record=True``) and ``steps`` (a list when
        ``record=True``, else ``None``).

    Raises:
        ValueError: If ``s`` or ``t`` is out of range, or ``s == t``.

    Time Complexity:
        O(V E^2): O(V E) augmenting paths at O(E) per BFS. ``record=True`` adds
        O(E) per path.

    Space Complexity:
        O(V + E + P), where P is the number of augmenting paths, at most
        O(V E): the visited and parent arrays and the queue are O(V), the flow
        matrix is O(E) and ``bottlenecks`` holds one entry per path.
        ``record=True`` adds O(E) per path.

    Examples:
        >>> net = Network.from_edges(4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)])
        >>> result = edmonds_karp(net, 0, 3)
        >>> result.value, result.iterations, result.bottlenecks
        (4, 2, [2, 2])
        >>> result.residual_updates
        8
        >>> result.flow
        {(0, 1): 2, (0, 2): 2, (1, 3): 2, (2, 3): 2}
        >>> result.steps is None
        True

        ``record=True`` keeps each path and the flow after it.

        >>> net.reset()
        >>> first = edmonds_karp(net, 0, 3, record=True).steps[0]
        >>> first["path"], first["edges"], first["bottleneck"], first["value"]
        ([0, 1, 3], [0, 4], 2, 2)
        >>> first["flow"]
        {(0, 1): 2, (1, 3): 2}

        With no path from s to t the flow is empty.

        >>> edmonds_karp(Network.from_edges(3, [(1, 2, 5)]), 0, 2).value
        0

        The terminals are checked first.

        >>> edmonds_karp(net, 1, 1)
        Traceback (most recent call last):
            ...
        ValueError: s and t must differ, got 1
        >>> edmonds_karp(net, 0, 9)
        Traceback (most recent call last):
            ...
        ValueError: s or t out of range for n=4: (0, 9)
    """
    n = net.n
    if not (0 <= s < n and 0 <= t < n):
        raise ValueError(f"s or t out of range for n={n}: ({s}, {t})")
    if s == t:
        raise ValueError(f"s and t must differ, got {s}")

    start = time.perf_counter()
    to, cap, adj = net.to, net.cap, net.adj
    parent_edge = [-1] * n
    seen = [0] * n
    stamp = 0
    queue: deque = deque()
    popleft, append = queue.popleft, queue.append

    value = 0
    bottlenecks: List[int] = []
    residual_updates = 0
    steps: List[Dict[str, Any]] = []

    while True:
        # A fresh stamp clears the visited marks in O(1).
        stamp += 1
        seen[s] = stamp
        queue.clear()
        append(s)
        while queue and seen[t] != stamp:
            for edge in adj[popleft()]:
                if cap[edge] > 0:
                    v = to[edge]
                    if seen[v] != stamp:
                        seen[v] = stamp
                        parent_edge[v] = edge
                        if v == t:
                            break
                        append(v)
        if seen[t] != stamp:
            break

        # Walk back from t, collecting the path and its tightest edge.
        path_edges: List[int] = []
        bottleneck = cap[parent_edge[t]]
        v = t
        while v != s:
            edge = parent_edge[v]
            path_edges.append(edge)
            if cap[edge] < bottleneck:
                bottleneck = cap[edge]
            v = to[edge ^ 1]

        for edge in path_edges:
            cap[edge] -= bottleneck
            cap[edge ^ 1] += bottleneck

        value += bottleneck
        bottlenecks.append(bottleneck)
        residual_updates += 2 * len(path_edges)
        if record:
            path_edges.reverse()
            steps.append(
                {
                    "path": [s, *(to[edge] for edge in path_edges)],
                    "edges": path_edges,
                    "bottleneck": bottleneck,
                    "value": value,
                    "flow": net.flow_matrix(),
                }
            )

    flow = net.flow_matrix()
    return FlowResult(
        value=value,
        flow=flow,
        iterations=len(bottlenecks),
        bottlenecks=bottlenecks,
        residual_updates=residual_updates,
        time_s=time.perf_counter() - start,
        steps=steps if record else None,
    )
