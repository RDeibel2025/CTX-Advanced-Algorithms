"""Independent checks on the flow a solver leaves in the residual graph.

A solver reports a value, but the evidence is in the residual capacities it
leaves behind, so these validators read those and trust nothing else. Each one
recomputes the flow as ``orig_cap[i] - cap[i]`` on the forward edges and tests
one property of it:

======================== ====================================================
Check                    What it reads from the residual graph
======================== ====================================================
check_capacity           0 <= flow <= capacity on every forward edge
check_conservation       in-flow equals out-flow at every node except s and t
check_residual           each edge plus its reverse sums to the original
                         capacity, so the pair still describes one flow
check_flow_value         net out of s and net into t both equal the reported
                         value
check_min_cost_feasible  no negative-cost cycle among edges with residual
                         capacity, the optimality test for min-cost flow
validate_flow            all of the above
======================== ====================================================

Each function returns ``(ok, message)`` and reports a bad flow through
``ok=False`` rather than an exception, so a benchmark can record the failure
and carry on.

``validate_flow`` proves a flow is valid, not that it is maximum: a min-cost
run with ``max_flow`` set sends less than the maximum on purpose and must still
pass. It does run ``check_min_cost_feasible``, so on a network with non-zero
costs it fails a correct maximum flow from Edmonds-Karp or push-relabel when
that flow is not also the cheapest one. Those solvers ignore costs, so validate
their results on zero-cost networks.

Why the cost check does not need potentials: around a cycle the potentials
telescope away, so a cycle's reduced cost equals its real cost. "No negative
reduced-cost cycle" and "no negative cycle" are the same statement.

Examples:
    >>> from src.flow_network_solver.core.network import Network
    >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
    >>> net.cap[0] -= 3; net.cap[1] += 3; net.cap[2] -= 3; net.cap[3] += 3
    >>> validate_flow(net, 0, 2, 3)[0]
    True
    >>> net.cap[0] -= 1; net.cap[1] += 1
    >>> check_conservation(net, 0, 2)
    (False, 'conservation broken at node 1: in - out = 1')

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.flow_network_solver.core.network import Network

__all__ = [
    "check_capacity",
    "check_conservation",
    "check_flow_value",
    "check_min_cost_feasible",
    "check_residual",
    "validate_flow",
]

#: What every validator returns: whether the check passed, and why.
Report = Tuple[bool, str]


def _terminal_problem(net: Network, s: int, t: int) -> Optional[str]:
    """Return a message if ``s`` and ``t`` cannot be a source and sink.

    Args:
        net: The network.
        s: The source.
        t: The sink.

    Returns:
        A message describing the problem, or ``None`` if they are usable.

    Raises:
        None.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> net = Network(2)
        >>> _terminal_problem(net, 0, 1) is None
        True
        >>> _terminal_problem(net, 1, 1)
        's and t must differ, got 1'
        >>> _terminal_problem(net, 0, 2)
        's or t out of range for n=2: (0, 2)'
    """
    if not (0 <= s < net.n and 0 <= t < net.n):
        return f"s or t out of range for n={net.n}: ({s}, {t})"
    if s == t:
        return f"s and t must differ, got {s}"
    return None


def _node_balance(net: Network) -> List[int]:
    """Return in-flow minus out-flow at each node.

    Args:
        net: The network.

    Returns:
        A list of ``n`` ints, computed from the forward edges only.

    Raises:
        None.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
        >>> net.cap[0] -= 2; net.cap[1] += 2
        >>> _node_balance(net)
        [-2, 2, 0]
    """
    to, cap, orig_cap = net.to, net.cap, net.orig_cap
    balance = [0] * net.n
    for i in range(0, len(to), 2):
        units = orig_cap[i] - cap[i]
        balance[to[i]] += units
        balance[to[i ^ 1]] -= units
    return balance


def check_capacity(net: Network) -> Report:
    """Check that 0 <= flow <= capacity on every forward edge.

    The flow is read from the residual capacity as ``orig_cap[i] - cap[i]``.
    A negative residual capacity means an edge was overfilled.

    Args:
        net: The network holding the flow.

    Returns:
        ``(ok, message)``; the message names the first bad edge.

    Raises:
        None. A bad flow gives ``ok=False``.

    Time Complexity:
        O(E).

    Space Complexity:
        O(1).

    Examples:
        >>> net = Network.from_edges(2, [(0, 1, 5)])
        >>> check_capacity(net)
        (True, 'capacity ok on 1 edges')
        >>> net.cap[0] -= 6; net.cap[1] += 6
        >>> check_capacity(net)
        (False, 'capacity exceeded on edge 0->1: flow 6 outside [0, 5]')
    """
    cap = net.cap
    for i, u, v, capacity, _ in net.forward_edges():
        units = capacity - cap[i]
        if units < 0 or units > capacity:
            return False, f"capacity exceeded on edge {u}->{v}: flow {units} outside [0, {capacity}]"
    return True, f"capacity ok on {net.m} edges"


def check_conservation(net: Network, s: int, t: int) -> Report:
    """Check that in-flow equals out-flow at every node except ``s`` and ``t``.

    A push-relabel run that stopped with excess on an inner node holds a
    preflow, not a flow, and fails here.

    Args:
        net: The network holding the flow.
        s: The source.
        t: The sink.

    Returns:
        ``(ok, message)``; the message names the first bad node.

    Raises:
        None. A bad flow or bad terminals give ``ok=False``.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 5)])
        >>> check_conservation(net, 0, 2)
        (True, 'conservation ok on 1 inner nodes')
        >>> net.cap[0] -= 2; net.cap[1] += 2
        >>> check_conservation(net, 0, 2)
        (False, 'conservation broken at node 1: in - out = 2')
    """
    problem = _terminal_problem(net, s, t)
    if problem:
        return False, problem
    for node, balance in enumerate(_node_balance(net)):
        if balance and node != s and node != t:
            return False, f"conservation broken at node {node}: in - out = {balance}"
    return True, f"conservation ok on {net.n - 2} inner nodes"


def check_residual(net: Network) -> Report:
    """Check that each edge and its reverse sum to the original capacity.

    Pushing d units moves d between ``cap[i]`` and ``cap[i ^ 1]``, so their sum
    never changes. A different sum means a solver updated one side of a pair
    and not the other.

    Args:
        net: The network holding the flow.

    Returns:
        ``(ok, message)``; the message names the first bad edge.

    Raises:
        None. A bad flow gives ``ok=False``.

    Time Complexity:
        O(E).

    Space Complexity:
        O(1).

    Examples:
        >>> net = Network.from_edges(2, [(0, 1, 5)])
        >>> net.cap[0] -= 2; net.cap[1] += 2
        >>> check_residual(net)
        (True, 'residual sums ok on 1 edges')
        >>> net.cap[1] += 1
        >>> check_residual(net)
        (False, 'residual sum broken on edge 0->1: 6 != 5')
    """
    cap = net.cap
    for i, u, v, capacity, _ in net.forward_edges():
        total = cap[i] + cap[i ^ 1]
        if total != capacity:
            return False, f"residual sum broken on edge {u}->{v}: {total} != {capacity}"
    return True, f"residual sums ok on {net.m} edges"


def check_flow_value(net: Network, s: int, t: int, value: int) -> Report:
    """Check that net out of ``s`` and net into ``t`` both equal ``value``.

    Args:
        net: The network holding the flow.
        s: The source.
        t: The sink.
        value: The flow value the solver reported.

    Returns:
        ``(ok, message)``.

    Raises:
        None. A bad flow or bad terminals give ``ok=False``.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V).

    Examples:
        >>> net = Network.from_edges(2, [(0, 1, 5)])
        >>> net.cap[0] -= 2; net.cap[1] += 2
        >>> check_flow_value(net, 0, 1, 2)
        (True, 'flow value ok: 2')
        >>> check_flow_value(net, 0, 1, 3)
        (False, 'flow value mismatch: reported 3, out of s 2, into t 2')
    """
    problem = _terminal_problem(net, s, t)
    if problem:
        return False, problem
    balance = _node_balance(net)
    out_of_s, into_t = -balance[s], balance[t]
    if out_of_s != value or into_t != value:
        return False, f"flow value mismatch: reported {value}, out of s {out_of_s}, into t {into_t}"
    return True, f"flow value ok: {value}"


def check_min_cost_feasible(net: Network, s: int, t: int) -> Report:
    """Check that the residual graph has no negative-cost cycle.

    A flow of a given value has minimum cost exactly when no such cycle
    exists: pushing around one keeps the value and lowers the cost.

    This is a queue-based Bellman-Ford over the edges with residual capacity.
    Every node starts in the queue at distance 0, which acts like a virtual
    source with a zero-cost edge to each node, so a cycle is found wherever it
    is, including nodes s cannot reach. ``steps[v]`` counts the edges on the
    walk behind ``dist[v]``. Distances only fall, so a walk that revisits a
    node has come back cheaper, which means it went around a negative cycle; a
    walk of ``n`` edges must revisit a node.

    When every cost is 0 no cycle can be negative, so it returns at once. That
    keeps zero-cost validation fast at n = 50,000.

    Args:
        net: The network holding the flow.
        s: The source. Checked for range only; the test covers the whole graph.
        t: The sink. Checked for range only.

    Returns:
        ``(ok, message)``.

    Raises:
        None. A negative cycle or bad terminals give ``ok=False``.

    Time Complexity:
        O(V + E) when all costs are 0 (one scan of the costs), otherwise
        O(V E) in the worst case.

    Space Complexity:
        O(V).

    Examples:
        >>> check_min_cost_feasible(Network.from_edges(2, [(0, 1, 5)]), 0, 1)
        (True, 'all costs are 0, so no negative cycle exists')
        >>> net = Network.from_edges(3, [(0, 1, 5, 2), (1, 2, 5, -1)])
        >>> check_min_cost_feasible(net, 0, 2)
        (True, 'no negative-cost cycle in the residual graph')
        >>> cycle = [(0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, -3)]
        >>> check_min_cost_feasible(Network.from_edges(3, cycle), 0, 2)
        (False, 'negative-cost cycle in the residual graph')
    """
    problem = _terminal_problem(net, s, t)
    if problem:
        return False, problem
    cost = net.cost
    if not any(cost):
        return True, "all costs are 0, so no negative cycle exists"
    n, to, cap, adj = net.n, net.to, net.cap, net.adj
    dist = [0] * n
    steps = [0] * n
    queued = [True] * n
    queue = deque(range(n))
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
                        return False, "negative-cost cycle in the residual graph"
                    if not queued[v]:
                        queued[v] = True
                        queue.append(v)
    return True, "no negative-cost cycle in the residual graph"


def validate_flow(net: Network, s: int, t: int, value: int) -> Report:
    """Run every validator and combine the results.

    All five checks always run, so a failure report lists everything wrong
    rather than the first problem only. See the module docstring for two
    limits: this does not prove the flow is maximum, and a correct maximum flow
    fails on a costed network when it is not also a cheapest one.

    Args:
        net: The network holding the flow.
        s: The source.
        t: The sink.
        value: The flow value the solver reported.

    Returns:
        ``(ok, message)``: ``ok`` is ``True`` only if all five checks pass, and
        the message joins the failing checks' messages with ``"; "``.

    Raises:
        None. A bad flow or bad terminals give ``ok=False``.

    Time Complexity:
        O(V + E) with all costs 0, otherwise O(V E) in the worst case.

    Space Complexity:
        O(V).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
        >>> validate_flow(net, 0, 2, 0)
        (True, 'all checks passed')
        >>> net.cap[0] -= 1; net.cap[1] += 1
        >>> validate_flow(net, 0, 2, 1)
        (False, 'conservation broken at node 1: in - out = 1; flow value mismatch: reported 1, out of s 1, into t 0')
    """
    reports = [
        check_capacity(net),
        check_conservation(net, s, t),
        check_residual(net),
        check_flow_value(net, s, t, value),
        check_min_cost_feasible(net, s, t),
    ]
    failures = [message for ok, message in reports if not ok]
    if failures:
        return False, "; ".join(failures)
    return True, "all checks passed"
