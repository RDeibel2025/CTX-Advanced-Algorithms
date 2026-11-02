"""Push-relabel maximum flow: read heights and excess locally, never a whole path.

The algorithm
-------------
Edmonds-Karp keeps a valid flow at every step and searches for a path that
improves it. Push-relabel turns that around. It starts by saturating every edge
out of s, which is far more than the network can carry, and then repairs the
damage one node at a time. Until the end the state is a *preflow*: edges respect
their capacities, but a node may hold more in-flow than out-flow, and that
surplus is its *excess*. A node other than s and t with excess is *active*.

Each node also has a *height*. Heights start at ``height[s] = n`` and
``height[t] = 0`` and always stay valid: an edge with residual capacity
u -> v has ``height[u] <= height[v] + 1``. Two local operations do all the work:

* **Push** moves excess from an active u down an edge u -> v with residual
  capacity and ``height[u] == height[v] + 1``, as much as the excess and the
  residual capacity allow.
* **Relabel** raises an active u with no such edge to one more than its lowest
  residual neighbour, so that a push becomes possible.

Validity is what proves the answer. A residual s-t path would have to fall from
height n to height 0 in at most n - 1 edges of drop 1, which is impossible, so
there is never a residual s-t path. When no node is active the preflow is a flow,
and a flow with no residual s-t path is maximum.

How it reads the residual graph
-------------------------------
A push reads one residual edge and two heights, so no operation ever looks past
a node's own edges. Height is the only global information, and it has a meaning
in the residual graph: below n, ``height[u]`` is a lower bound on the residual
distance from u to t. At n and above, u cannot reach t any more and
``height[u] - n`` is a lower bound on its residual distance to s, so the excess
at u is on its way home through the reverse edges. The solver does not stop at
the min cut. It runs until every unit of excess has either reached t or gone back
to s, so the result is a real flow, ``value`` is the excess at t and
:func:`~src.flow_network_solver.utils.validators.validate_flow` passes.

The residual edges are the same ones Edmonds-Karp searches: ``cap[e] -= d;
cap[e ^ 1] += d`` is a push, and :func:`~src.flow_network_solver.core.network.min_cut`
reads the same cut afterwards.

The three choices that make it fast
-----------------------------------
Plain push-relabel sees only one node at a time, so it wastes work on excess that
has nowhere to go. Three additions supply the wider view, and none of them changes
the O(V^3) bound:

* **FIFO order.** Active nodes wait in a queue. A node is discharged until it has
  no excess or has been relabeled, then waits its turn again. An in-queue flag
  keeps each node in the queue once.
* **Global relabel.** At the start and after every n relabels, a reverse BFS over
  residual edges replaces the lower bounds with the exact distances: distance to
  t for nodes that reach t, n plus distance to s for the others, and 2n for a node
  that reaches neither (it holds no excess and none can arrive). Without it,
  excess that cannot reach t would climb one relabel at a time past n before
  turning back. Exact labels are never lower than valid ones, so heights only rise.
* **Gap heuristic.** ``count[h]`` is the number of nodes at height h. If it hits 0
  for some h < n, no node above h can reach t, because a residual path to t would
  need a step down through height h. Each node with h < height < n jumps to n + 1
  at once, the height it was bound for anyway.

Implementation
--------------
This is pure Python at 50,000 nodes, so :func:`push_relabel` is one long function.
``to``, ``cap``, ``adj``, ``height`` and ``excess`` are bound to locals and the
excess of the node being discharged lives in a local too. ``cur[u]`` is the
current-arc pointer: edges before it have been ruled out at the present height, so
a discharge resumes there instead of rescanning. The terminals carry a permanent
in-queue flag, which keeps them out of the queue with no extra test in the push.

A scan of all n nodes for each gap would cost O(n) every time one fires, so
``buckets[h]`` lists the nodes that have held height h since the last global
relabel. Entries are never removed; a gap walks the buckets above it and skips
nodes that have moved on, so each entry is paid for once by the relabel that made
it. ``top`` tracks the highest level in use and limits that walk.

The convergence profile costs almost nothing because nothing needs recounting.
Every queued node is active and every active node is queued, so the active count
is the queue length, plus the node being discharged if it still has excess.
``excess`` is in-flow minus out-flow at every node including s, so it sums to 0
and the excess held by inner nodes is exactly ``-excess[s] - excess[t]``.

Examples:
    >>> from src.flow_network_solver.core.network import min_cut
    >>> edges = [(0, 1, 16), (0, 2, 13), (2, 1, 4), (1, 3, 12), (3, 2, 9),
    ...          (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    >>> net = Network.from_edges(6, edges)
    >>> result = push_relabel(net, 0, 5)
    >>> result.value, result.convergence[-1][1:]
    (23, (0, 0))
    >>> sorted(min_cut(net, 0))
    [0, 1, 2, 4]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Tuple

from src.flow_network_solver.core.network import FlowResult, Network

__all__ = ["push_relabel"]


def _global_relabel(
    n: int,
    s: int,
    t: int,
    to: List[int],
    cap: List[int],
    adj: List[List[int]],
    height: List[int],
) -> Tuple[List[List[int]], List[int], int]:
    """Set every height to its exact residual distance, in place.

    Two reverse BFS passes read the residual graph backwards: edge ``e`` leaves
    u, so ``e ^ 1`` enters u, and ``cap[e ^ 1] > 0`` means its tail can step to
    u. The first pass starts at t and never crosses s. The second starts at s and
    adds n to the distance, for the nodes that cannot reach t. Any node left over
    reaches neither terminal and gets 2n.

    Args:
        n: The number of nodes.
        s: The source.
        t: The sink.
        to: The edge heads.
        cap: The residual capacities.
        adj: The edge indices leaving each node.
        height: The heights, overwritten in place so aliases stay valid.

    Returns:
        The fresh gap-heuristic state ``(buckets, count, top)``. ``buckets[d]``
        lists the nodes at distance d from t, ``count[d]`` is how many there are,
        and ``top`` is the largest such d. All of it covers heights below n only,
        so s is not counted.

    Raises:
        None.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V).

    Examples:
        With 0 -> 1 saturated, node 1 is one step from t. Node 3 cannot reach t,
        but its edge into s puts it one step from s, so it gets n + 1.

        >>> net = Network.from_edges(4, [(0, 1, 2), (1, 2, 2), (3, 0, 1)])
        >>> net.cap[0] = 0; net.cap[1] = 2
        >>> height = [0, 0, 0, 0]
        >>> buckets, count, top = _global_relabel(4, 0, 2, net.to, net.cap, net.adj, height)
        >>> height, buckets[:2], count, top
        ([4, 1, 0, 5], [[2], [1]], [1, 1, 0, 0], 1)

        Saturate 1 -> 2 as well and node 1 is cut off from t, so it is labeled by
        its distance back to s.

        >>> net.cap[2] = 0; net.cap[3] = 2
        >>> top = _global_relabel(4, 0, 2, net.to, net.cap, net.adj, height)[2]
        >>> height, top
        ([4, 5, 0, 5], 0)

        A node that reaches neither terminal gets 2n.

        >>> lonely = Network(3)
        >>> height = [0, 0, 0]
        >>> _ = _global_relabel(3, 0, 1, lonely.to, lonely.cap, lonely.adj, height)
        >>> height
        [3, 0, 6]
    """
    unseen = 2 * n
    fresh = [unseen] * n
    fresh[s] = n
    fresh[t] = 0
    buckets: List[List[int]] = [[] for _ in range(n)]
    buckets[0].append(t)
    top = 0

    for root, base in ((t, 0), (s, n)):
        frontier = [root]
        distance = base
        while frontier:
            distance += 1
            following: List[int] = []
            for u in frontier:
                for edge in adj[u]:
                    if cap[edge ^ 1] > 0:
                        v = to[edge]
                        if fresh[v] == unseen:
                            fresh[v] = distance
                            following.append(v)
            if base == 0 and following:
                buckets[distance] = following
                top = distance
            frontier = following

    height[:] = fresh
    return buckets, [len(bucket) for bucket in buckets], top


def _snapshot(
    steps: List[Dict[str, Any]],
    op: str,
    node: int,
    height: List[int],
    excess: List[int],
    t: int,
) -> None:
    """Append the state after one operation to ``steps``.

    This is a module-level function rather than a closure on purpose. A closure
    over ``height`` and ``excess`` would turn them into cell variables, which
    slows every read of them in the main loop even when nothing is recorded.

    Args:
        steps: The list to append to.
        op: ``"init"``, ``"global_relabel"``, ``"push"`` or ``"relabel"``.
        node: The node the operation acted on.
        height: The current heights, copied.
        excess: The current excess, copied.
        t: The sink, whose excess is the ``"value"`` entry.

    Returns:
        None.

    Raises:
        None.

    Time Complexity:
        O(V).

    Space Complexity:
        O(V).

    Examples:
        >>> steps = []
        >>> _snapshot(steps, "push", 1, [3, 1, 0], [-5, 2, 3], 2)
        >>> steps
        [{'op': 'push', 'node': 1, 'heights': [3, 1, 0], 'excess': [-5, 2, 3], 'value': 3}]
    """
    steps.append(
        {"op": op, "node": node, "heights": list(height), "excess": list(excess), "value": excess[t]}
    )


def push_relabel(net: Network, s: int, t: int, record: bool = False) -> FlowResult:
    """Compute a maximum s-t flow by FIFO push-relabel.

    Saturates the edges out of s, then discharges active nodes in FIFO order with
    the gap heuristic and a global relabel at the start and every n relabels (see
    the module docstring). It stops only when no node holds excess, so the excess
    that cannot reach t has been returned to s and the result is a valid flow.

    The network is taken as it stands: ``cap`` is mutated in place and the final
    residual is left there for
    :func:`~src.flow_network_solver.core.network.min_cut` and the validators. To
    run another solver on the same graph, call :meth:`Network.reset` first. On a
    net that already carries flow, ``value`` and the counters describe only what
    this call added, while ``flow`` reports the whole flow in the net.

    Args:
        net: The network. Its capacities are mutated.
        s: The source node.
        t: The sink node.
        record: If ``True``, fill ``steps`` with one dict per operation:
            ``{"op": "init" | "global_relabel" | "push" | "relabel", "node": int,
            "heights": list, "excess": list, "value": excess at t}``, taken just
            after the operation. ``node`` is s for ``"init"``, t for
            ``"global_relabel"``, and the node pushed from or relabeled otherwise.
            A gap jump has no step of its own and shows in the ``"relabel"`` that
            fired it. ``excess`` is in-flow minus out-flow, so ``excess[s]`` is
            negative. Each step copies two lists, so keep this to small graphs.

    Returns:
        A :class:`FlowResult` with ``value`` (the excess at t), ``flow``,
        ``pushes``, ``relabels``, ``iterations`` (pushes plus relabels),
        ``global_relabels`` (including the one at the start), ``residual_updates``
        (entries of ``cap`` changed, 2 per push and 2 per saturated edge out of
        s), ``convergence``, ``time_s`` (covering the search and the final flow
        matrix, plus the recording when ``record=True``) and ``steps`` (a list
        when ``record=True``, else ``None``). ``convergence`` is a list of
        ``(operations, active_nodes, total_excess)`` tuples: the state before the
        first operation, one sample every ``max(1, n // 50)`` operations, and the
        final state, which always has 0 active nodes and 0 excess.

    Raises:
        ValueError: If ``s`` or ``t`` is out of range, or ``s == t``.

    Time Complexity:
        O(V^3) worst case for FIFO order. Each global relabel is O(V + E) and runs
        once per n relabels, which leaves the bound unchanged. ``record=True`` adds
        O(V) per operation.

    Space Complexity:
        O(V + E): the node arrays, queue and buckets are O(V), and the flow matrix
        is O(E). ``record=True`` adds O(V) per operation.

    Examples:
        Node 1 receives 5 units but can forward only 3, so the other 2 go back to
        s through the reverse edge.

        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
        >>> result = push_relabel(net, 0, 2)
        >>> result.value, result.flow
        (3, {(0, 1): 3, (1, 2): 3})
        >>> result.pushes, result.relabels, result.iterations
        (2, 1, 3)
        >>> result.global_relabels, result.residual_updates, result.steps is None
        (1, 6, True)
        >>> result.convergence
        [(0, 1, 5), (1, 1, 2), (2, 1, 2), (3, 0, 0)]

        ``record=True`` keeps a snapshot of the labels after each operation.

        >>> net.reset()
        >>> for step in push_relabel(net, 0, 2, record=True).steps:
        ...     print(step["op"], step["node"], step["heights"], step["excess"], step["value"])
        init 0 [3, 0, 0] [-5, 5, 0] 0
        global_relabel 2 [3, 1, 0] [-5, 5, 0] 0
        push 1 [3, 1, 0] [-5, 2, 3] 3
        relabel 1 [3, 4, 0] [-5, 2, 3] 3
        push 1 [3, 4, 0] [-3, 0, 3] 3

        With no path from s to t everything sent out comes back.

        >>> net = Network.from_edges(3, [(0, 1, 5)])
        >>> result = push_relabel(net, 0, 2)
        >>> result.value, result.flow
        (0, {})

        The terminals are checked first.

        >>> push_relabel(net, 1, 1)
        Traceback (most recent call last):
            ...
        ValueError: s and t must differ, got 1
        >>> push_relabel(net, 0, 9)
        Traceback (most recent call last):
            ...
        ValueError: s or t out of range for n=3: (0, 9)
    """
    n = net.n
    if not (0 <= s < n and 0 <= t < n):
        raise ValueError(f"s or t out of range for n={n}: ({s}, {t})")
    if s == t:
        raise ValueError(f"s and t must differ, got {s}")

    start = time.perf_counter()
    to, cap, adj = net.to, net.cap, net.adj
    height = [0] * n
    height[s] = n
    excess = [0] * n
    cur = [0] * n
    queue: deque = deque()
    popleft, append = queue.popleft, queue.append
    # A permanent flag keeps the terminals out of the queue without a test in push.
    in_queue = [False] * n
    in_queue[s] = in_queue[t] = True

    steps: List[Dict[str, Any]] = []

    # The preflow: saturate every edge out of s.
    saturated = 0
    for edge in adj[s]:
        units = cap[edge]
        if units > 0:
            v = to[edge]
            cap[edge] = 0
            cap[edge ^ 1] += units
            excess[v] += units
            excess[s] -= units
            saturated += 1
            if not in_queue[v]:
                in_queue[v] = True
                append(v)
    if record:
        _snapshot(steps, "init", s, height, excess, t)

    lifted = n + 1
    buckets, count, top = _global_relabel(n, s, t, to, cap, adj, height)
    global_relabels = 1
    if record:
        _snapshot(steps, "global_relabel", t, height, excess, t)

    interval = max(1, n // 50)
    next_sample = interval
    convergence: List[Tuple[int, int, int]] = [(0, len(queue), -excess[s] - excess[t])]
    operations = 0
    relabels = 0
    next_global = n

    while queue:
        if relabels >= next_global:
            next_global = relabels + n
            buckets, count, top = _global_relabel(n, s, t, to, cap, adj, height)
            cur = [0] * n
            global_relabels += 1
            if record:
                _snapshot(steps, "global_relabel", t, height, excess, t)

        u = popleft()
        in_queue[u] = False
        remaining = excess[u]
        base = height[u]
        below = base - 1
        edges = adj[u]

        for position in range(cur[u], len(edges)):
            edge = edges[position]
            room = cap[edge]
            if room and height[to[edge]] == below:
                # Push as much as the excess and the residual capacity allow.
                v = to[edge]
                units = room if room < remaining else remaining
                cap[edge] = room - units
                cap[edge ^ 1] += units
                excess[v] += units
                remaining -= units
                if not in_queue[v]:
                    in_queue[v] = True
                    append(v)
                operations += 1
                if operations == next_sample:
                    next_sample += interval
                    active = len(queue) + (1 if remaining else 0)
                    convergence.append((operations, active, -excess[s] - excess[t]))
                if record:
                    excess[u] = remaining
                    _snapshot(steps, "push", u, height, excess, t)
                if not remaining:
                    cur[u] = position
                    excess[u] = 0
                    break
        else:
            # No admissible edge is left, so relabel. Excess implies a reverse
            # residual edge, so a lowest residual neighbour always exists.
            excess[u] = remaining
            lowest = 2 * n
            best = 0
            for position, edge in enumerate(edges):
                if cap[edge]:
                    h = height[to[edge]]
                    if h < lowest:
                        lowest = h
                        best = position
            new_height = lowest + 1
            height[u] = new_height
            cur[u] = best
            relabels += 1
            if new_height < n:
                count[new_height] += 1
                buckets[new_height].append(u)
                if new_height > top:
                    top = new_height
            if base < n:
                count[base] -= 1
                if not count[base]:
                    # The gap: nothing above this level can reach t any more.
                    for level in range(base + 1, top + 1):
                        for v in buckets[level]:
                            if height[v] == level:
                                height[v] = lifted
                                cur[v] = 0
                        buckets[level] = []
                        count[level] = 0
                    top = base - 1
            operations += 1
            if operations == next_sample:
                next_sample += interval
                convergence.append((operations, len(queue) + 1, -excess[s] - excess[t]))
            if record:
                _snapshot(steps, "relabel", u, height, excess, t)
            in_queue[u] = True
            append(u)

    pushes = operations - relabels
    final = (operations, 0, -excess[s] - excess[t])
    if convergence[-1] != final:
        convergence.append(final)
    value = excess[t]
    flow = net.flow_matrix()
    return FlowResult(
        value=value,
        flow=flow,
        iterations=operations,
        residual_updates=2 * (pushes + saturated),
        time_s=time.perf_counter() - start,
        steps=steps if record else None,
        pushes=pushes,
        relabels=relabels,
        global_relabels=global_relabels,
        convergence=convergence,
    )
