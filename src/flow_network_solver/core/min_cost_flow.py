"""Minimum-cost maximum flow: successive shortest paths with Johnson potentials.

Each edge now carries a cost per unit as well as a capacity. Many flows can have
the same value, and :func:`min_cost_max_flow` finds the cheapest one: the full
maximum flow, or exactly ``max_flow`` units when that is asked for.

How it reads the residual graph
-------------------------------
Edmonds-Karp reads the residual graph for the s-t path with the fewest edges.
This solver reads the same graph with costs and takes the cheapest s-t path. A
forward residual edge costs ``c`` and its reverse costs ``-c``, because undoing a
unit of flow refunds what that unit paid.

The residual graph also certifies the answer. A flow is the cheapest of its value
exactly when its residual graph has no negative-cost cycle, since pushing around
such a cycle would keep the value and lower the cost. Augmenting along a cheapest
path preserves that property, so the flow after every augmentation is the
cheapest flow of its value. That is why stopping early at ``max_flow`` units is
just as optimal as running to the end, and why the run ends with a maximum flow
of minimum cost once t is no longer reachable.

Why potentials
--------------
Dijkstra needs non-negative edge costs, and the residual graph has negative ones
as soon as any flow moves, because reverse edges cost ``-c``. Johnson's fix keeps
a potential pi(v) on each node and runs Dijkstra on the reduced cost

    c_pi(u, v) = c(u, v) + pi(u) - pi(v).

Along a path the potentials telescope: the reduced cost of an s-t path is its real
cost plus pi(s) - pi(t), a constant, so the same paths are shortest under both.
Around a cycle they cancel completely, so reduced costs that are all non-negative
also prove the residual graph has no negative cycle.

**Starting potentials.** If no edge with residual capacity has a negative cost,
then pi = 0 is already valid: c_pi = c >= 0, and Bellman-Ford is not needed at all.
The test must look at residual capacity, not just at the ``cost`` array: every
reverse edge has cost ``-c``, but on an empty flow it has no capacity and is not
in the residual graph. On a net that already carries flow those reverse edges do
count, and the same test sends the run to Bellman-Ford. Otherwise a queue-based
Bellman-Ford from s gives distances d, and d(v) <= d(u) + c(u, v) on every residual
edge is exactly c_d(u, v) >= 0, so pi = d is valid. A node s cannot reach keeps
pi = 0; no flow can ever reach it, so its potential is never read.

**Updating potentials.** Each round runs Dijkstra on reduced costs, stopping as
soon as t is settled at distance D, then adds min(d(v), D) to pi(v), where d(v) is
the distance found so far (infinite if the node was never reached). Every residual
edge u -> v keeps a non-negative reduced cost c':

* If d(u) < D then u was settled and its edges were relaxed, so
  d(v) <= d(u) + c_pi(u, v) and c' >= 0.
* If d(u) >= D then u gained D, the most any node gains, so c' >= c_pi(u, v) >= 0.

Edges on the shortest path get c' = 0, so the reverse edges the augmentation is
about to open also have reduced cost 0. Capping at D matters: nodes Dijkstra did
not settle have only tentative distances, and adding those could break the
invariant.

Negative cycles
---------------
A negative cycle reachable from s makes shortest paths undefined, so Bellman-Ford
raises ``ValueError``. Distances only fall, so a walk behind ``dist[v]`` that
revisits a node came back cheaper, which means it went around a negative cycle; a
walk of n edges must revisit a node, so ``steps[v] >= n`` is the test. A cycle s
cannot reach is ignored, because no augmenting path can ever touch it: an
augmentation only adds residual edges between nodes already reachable, so the
reachable set never grows. :func:`check_min_cost_feasible` scans the whole graph
and would still report such a cycle.

======================= =============================== ==========
Function                Time                            Space
======================= =============================== ==========
min_cost_max_flow       O(V E + A (V + E log V))        O(V + E)
_initial_potentials     O(V E), O(E) when costs are >=0 O(V)
======================= =============================== ==========

Here A is the number of augmentations, at most the flow value. Each round costs
O(V) for the distance arrays and potential update even when Dijkstra stops early.

Examples:
    >>> edges = [(0, 1, 4, 2), (0, 2, 2, 2), (1, 2, 2, 1), (1, 3, 3, 3), (2, 3, 5, 1)]
    >>> net = Network.from_edges(4, edges)
    >>> result = min_cost_max_flow(net, 0, 3)
    >>> result.value, result.cost, result.augmentations
    (6, 24, 3)
    >>> net.reset()
    >>> result = min_cost_max_flow(net, 0, 3, max_flow=3)
    >>> result.value, result.cost
    (3, 10)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import List, Optional

from src.flow_network_solver.core.network import FlowResult, Network

__all__ = ["min_cost_max_flow"]

_INF = float("inf")


def _initial_potentials(net: Network, s: int) -> List[int]:
    """Return starting potentials that make every reachable reduced cost >= 0.

    With no negative cost on any edge that has residual capacity, all zeros are
    valid and Bellman-Ford is skipped. Otherwise this runs a queue-based
    Bellman-Ford from ``s`` over the residual edges. ``steps[v]`` counts the edges
    on the walk behind ``dist[v]``, and reaching ``n`` of them proves a negative
    cycle (see the module docstring).

    Args:
        net: The network as it stands, possibly already carrying flow.
        s: The source.

    Returns:
        A list of ``n`` ints: the distance from ``s``, or 0 for a node ``s``
        cannot reach.

    Raises:
        ValueError: If a negative-cost cycle is reachable from ``s``.

    Time Complexity:
        O(E) when no residual edge has a negative cost, otherwise O(V E).

    Space Complexity:
        O(V).

    Examples:
        >>> _initial_potentials(Network.from_edges(3, [(0, 1, 5, 2), (1, 2, 5, 3)]), 0)
        [0, 0, 0]
        >>> net = Network.from_edges(4, [(0, 1, 5, 4), (1, 2, 5, -6)])
        >>> _initial_potentials(net, 0)
        [0, 4, -2, 0]
        >>> net = Network.from_edges(3, [(0, 1, 1, 1), (1, 2, 1, -5), (2, 1, 1, 1)])
        >>> _initial_potentials(net, 0)
        Traceback (most recent call last):
            ...
        ValueError: negative-cost cycle reachable from s=0
    """
    n, to, cap, cost, adj = net.n, net.to, net.cap, net.cost, net.adj
    if not any(price < 0 and room > 0 for price, room in zip(cost, cap)):
        return [0] * n
    dist: List[float] = [_INF] * n
    dist[s] = 0
    steps = [0] * n
    queued = [False] * n
    queued[s] = True
    queue = deque([s])
    while queue:
        u = queue.popleft()
        queued[u] = False
        for edge in adj[u]:
            if cap[edge] > 0:
                v = to[edge]
                candidate = dist[u] + cost[edge]
                if candidate < dist[v]:
                    dist[v] = candidate
                    steps[v] = steps[u] + 1
                    if steps[v] >= n:
                        raise ValueError(f"negative-cost cycle reachable from s={s}")
                    if not queued[v]:
                        queued[v] = True
                        queue.append(v)
    return [0 if d == _INF else d for d in dist]


def min_cost_max_flow(net: Network, s: int, t: int, max_flow: Optional[int] = None) -> FlowResult:
    """Send the maximum flow, or ``max_flow`` units, from s to t at minimum cost.

    Successive shortest paths: find the cheapest s-t path in the residual graph
    with Dijkstra on reduced costs, push as much as it allows (capped by what
    ``max_flow`` still permits), update the potentials and repeat until t is
    unreachable or ``max_flow`` units have gone. The module docstring explains the
    potentials and why each intermediate flow is optimal for its value.

    The net is used as it stands: ``cap`` is mutated in place and the final
    residual stays there, so call :meth:`Network.reset` first for a fresh solve.
    If the net already carries flow, ``value`` and ``cost`` count only what this
    call sends, while ``flow`` is the whole flow matrix. Resuming from an optimal
    flow is safe, but a flow from a solver that ignores costs may leave a negative
    residual cycle, which raises ``ValueError``.

    Args:
        net: The network, with integer costs that may be negative.
        s: The source.
        t: The sink.
        max_flow: How many units to send at most, a non-negative ``int``, or
            ``None`` for the maximum flow. A value above the maximum flow sends
            the maximum flow.

    Returns:
        A :class:`FlowResult` with ``value``, ``flow``, ``cost``, ``augmentations``,
        ``iterations`` (equal to ``augmentations``), ``bottlenecks`` (units pushed
        per path), ``residual_updates`` (capacity entries written, two per path
        edge) and ``time_s``.

    Raises:
        TypeError: If ``max_flow`` is neither ``None`` nor an ``int`` (``bool``
            rejected).
        ValueError: If ``s == t``, a node is out of range, ``max_flow`` is
            negative, or a negative-cost cycle is reachable from ``s``.

    Time Complexity:
        O(V E + A (V + E log V)) for A augmentations, at most the flow value.
        The V E term is Bellman-Ford and applies only when a residual edge has a
        negative cost.

    Space Complexity:
        O(V + E) for the distances, potentials, heap and flow matrix.

    Examples:
        >>> edges = [(0, 1, 4, 2), (0, 2, 2, 2), (1, 2, 2, 1), (1, 3, 3, 3), (2, 3, 5, 1)]
        >>> net = Network.from_edges(4, edges)
        >>> result = min_cost_max_flow(net, 0, 3)
        >>> result.value, result.cost, result.bottlenecks
        (6, 24, [2, 2, 2])
        >>> result.flow == {(0, 1): 4, (0, 2): 2, (1, 2): 2, (1, 3): 2, (2, 3): 4}
        True

        A negative edge cost is fine without a negative cycle. The cheap path
        here looks expensive to plain Dijkstra until its last edge:

        >>> edges = [(0, 1, 1, 5), (1, 3, 1, -4), (0, 2, 1, 2), (2, 3, 1, 1)]
        >>> min_cost_max_flow(Network.from_edges(4, edges), 0, 3, max_flow=1).cost
        1

        >>> cycle = [(0, 1, 1, 1), (1, 2, 1, -5), (2, 1, 1, 1), (2, 3, 1, 1)]
        >>> min_cost_max_flow(Network.from_edges(4, cycle), 0, 3)
        Traceback (most recent call last):
            ...
        ValueError: negative-cost cycle reachable from s=0
        >>> min_cost_max_flow(Network(2), 0, 0)
        Traceback (most recent call last):
            ...
        ValueError: s and t must differ, got 0
        >>> min_cost_max_flow(Network(2), 0, 1, max_flow=-1)
        Traceback (most recent call last):
            ...
        ValueError: max_flow must be non-negative, got -1
    """
    if not (0 <= s < net.n and 0 <= t < net.n):
        raise ValueError(f"s or t out of range for n={net.n}: ({s}, {t})")
    if s == t:
        raise ValueError(f"s and t must differ, got {s}")
    if max_flow is not None:
        if isinstance(max_flow, bool) or not isinstance(max_flow, int):
            raise TypeError(f"max_flow must be an int or None, got {type(max_flow).__name__}")
        if max_flow < 0:
            raise ValueError(f"max_flow must be non-negative, got {max_flow}")

    start = time.perf_counter()
    n, to, cap, cost, adj = net.n, net.to, net.cap, net.cost, net.adj
    heappush, heappop = heapq.heappush, heapq.heappop
    potential = _initial_potentials(net, s)
    value = 0
    total_cost = 0
    updates = 0
    bottlenecks: List[int] = []

    while max_flow is None or value < max_flow:
        dist: List[float] = [_INF] * n
        parent_edge = [-1] * n
        dist[s] = 0
        heap = [(0, s)]
        while heap:
            d, u = heappop(heap)
            if d > dist[u]:
                continue
            if u == t:
                break
            base = d + potential[u]
            for edge in adj[u]:
                if cap[edge] > 0:
                    v = to[edge]
                    candidate = base + cost[edge] - potential[v]
                    if candidate < dist[v]:
                        dist[v] = candidate
                        parent_edge[v] = edge
                        heappush(heap, (candidate, v))

        sink_dist = dist[t]
        if sink_dist == _INF:
            break
        for v in range(n):
            potential[v] += dist[v] if dist[v] < sink_dist else sink_dist

        pushed = _INF if max_flow is None else max_flow - value
        path_cost = 0
        v = t
        while v != s:
            edge = parent_edge[v]
            if cap[edge] < pushed:
                pushed = cap[edge]
            path_cost += cost[edge]
            v = to[edge ^ 1]
        v = t
        while v != s:
            edge = parent_edge[v]
            cap[edge] -= pushed
            cap[edge ^ 1] += pushed
            updates += 2
            v = to[edge ^ 1]

        value += pushed
        total_cost += pushed * path_cost
        bottlenecks.append(pushed)

    flow = net.flow_matrix()
    return FlowResult(
        value=value,
        flow=flow,
        iterations=len(bottlenecks),
        bottlenecks=bottlenecks,
        residual_updates=updates,
        time_s=time.perf_counter() - start,
        cost=total_cost,
        augmentations=len(bottlenecks),
    )
