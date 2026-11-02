"""The residual network that every Week 10 solver and application reads.

A *flow network* is a directed graph with a non-negative integer capacity on
each edge, a source s and a sink t. A *flow* assigns each edge an amount
between 0 and its capacity so that, at every node except s and t, the amount
in equals the amount out. Its value is the net amount leaving s.

The residual graph
------------------
Given a flow f, the residual graph says what can still change. An edge
u -> v with capacity c and flow f(u, v) contributes two residual edges:

* u -> v with residual capacity c - f(u, v), the room left to push, and
* v -> u with residual capacity f(u, v), the flow that can still be undone.

Every algorithm in this project works on this one structure and differs only
in what it reads from it. Edmonds-Karp reads the shortest s-t path, push-relabel
reads heights and excess locally, min-cost flow reads it with costs through
potentials, and the applications read the final residual graph: which edges are
saturated and which nodes are still reachable from s. Optimality is read from
it too: no s-t path left means the flow is maximum and, by the max-flow min-cut
theorem, the nodes reachable from s are the source side of a minimum cut
(:func:`min_cut`, :func:`cut_edges`, :func:`cut_value`).

Representation: flat edge arrays
--------------------------------
Push-relabel has to reach 50,000 nodes in pure Python, so a :class:`Network`
keeps parallel lists instead of dicts of dicts. Edge ``i`` is described by
``to[i]`` (its head), ``cap[i]`` (its *residual* capacity), ``cost[i]`` and
``orig_cap[i]``, and ``adj[u]`` lists the indices of the edges leaving u.

Each call to :meth:`Network.add_edge` appends a pair of edges. Forward edges get
even indices and reverse edges odd ones, so ``i ^ 1`` is always the partner of
``i`` and ``to[i ^ 1]`` is the tail of ``i``. A reverse edge starts with residual
capacity 0, has ``orig_cap`` 0 and has cost ``-c``. Pushing d units along edge
``i`` is exactly ``cap[i] -= d; cap[i ^ 1] += d``, and because each added edge
owns its partner, parallel edges and antiparallel pairs (u -> v and v -> u) need
no special handling.

The flow is never stored. It is read back from the residual capacities:
``orig_cap[i] - cap[i]`` on a forward edge. That is why all solvers can mutate
``cap`` in place, leave the final residual there for the readers below, and be
re-run on the same graph after :meth:`Network.reset`.

=================== ============== ==============
Operation           Time           Space
=================== ============== ==============
add_edge            O(1) amortized O(1) amortized
get_residual_graph  O(V + E)       O(V + E)
flow_on             O(1)           O(1)
flow_matrix         O(E)           O(E)
reset               O(E)           O(1)
from_edges          O(V + E)       O(V + E)
forward_edges       O(E)           O(1)
min_cut             O(V + E)       O(V)
cut_edges           O(V + E)       O(V + E)
cut_value           O(V + E)       O(V)
=================== ============== ==============

Examples:
    >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
    >>> net.m, net.to, net.cap
    (2, [1, 0, 2, 1], [5, 0, 3, 0])
    >>> net.cap[0] -= 3; net.cap[1] += 3; net.cap[2] -= 3; net.cap[3] += 3
    >>> net.flow_matrix()
    {(0, 1): 3, (1, 2): 3}
    >>> sorted(min_cut(net, 0)), cut_edges(net, 0), cut_value(net, 0)
    ([0, 1], [(1, 2)], 3)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

__all__ = ["FlowResult", "Network", "cut_edges", "cut_value", "min_cut"]


@dataclass
class FlowResult:
    """What a solver returns: the answer plus the counters the benchmark reads.

    Attributes:
        value: The flow value (net amount leaving s).
        flow: The flow matrix, as from :meth:`Network.flow_matrix`.
        iterations: Solver-specific step count (augmenting paths for
            Edmonds-Karp, pushes plus relabels for push-relabel).
        bottlenecks: The bottleneck of each augmenting path.
        residual_updates: How many edge capacity changes were made.
        time_s: Wall-clock solve time from ``time.perf_counter``.
        steps: The recorded steps when ``record=True``, else ``None``.
        pushes: Push operations (push-relabel).
        relabels: Relabel operations (push-relabel).
        global_relabels: Global relabels (push-relabel).
        convergence: Sampled ``(operations, active_nodes, total_excess)``
            tuples (push-relabel).
        cost: Total cost of the flow (min-cost flow), else ``None``.
        augmentations: Augmenting paths (min-cost flow).

    Examples:
        >>> result = FlowResult(value=23, flow={(0, 1): 23})
        >>> result.value, result.iterations, result.bottlenecks, result.cost
        (23, 0, [], None)
        >>> FlowResult(1, {}).convergence is FlowResult(1, {}).convergence
        False
    """

    value: int
    flow: dict
    iterations: int = 0
    bottlenecks: list = field(default_factory=list)
    residual_updates: int = 0
    time_s: float = 0.0
    steps: Optional[list] = None
    pushes: int = 0
    relabels: int = 0
    global_relabels: int = 0
    convergence: list = field(default_factory=list)
    cost: Optional[int] = None
    augmentations: int = 0


class Network:
    """A flow network stored as flat edge arrays with paired reverse edges.

    See the module docstring for the layout. The attributes are public on
    purpose: the solvers bind ``to``, ``cap`` and ``adj`` to locals and work
    on them directly, because attribute access inside a tight Python loop is
    what would make a 50,000 node run too slow.

    Attributes:
        n: Number of nodes, labelled ``0 .. n - 1``.
        to: ``to[i]`` is the head of edge ``i``.
        cap: ``cap[i]`` is the residual capacity of edge ``i``.
        cost: ``cost[i]`` is the cost per unit on edge ``i``.
        orig_cap: ``orig_cap[i]`` is the original capacity (0 on reverse edges).
        adj: ``adj[u]`` lists the indices of the edges leaving ``u``.

    Examples:
        >>> net = Network(2)
        >>> net.add_edge(0, 1, 7, cost=2)
        0
        >>> net.to, net.cap, net.cost, net.orig_cap, net.adj
        ([1, 0], [7, 0], [2, -2], [7, 0], [[0], [1]])
    """

    def __init__(self, n: int) -> None:
        """Create a network with ``n`` nodes and no edges.

        The six parallel lists start empty (``adj`` holds one empty list per
        node) and grow two entries per :meth:`add_edge`.

        Args:
            n: The number of nodes.

        Returns:
            None.

        Raises:
            TypeError: If ``n`` is not an ``int`` (``bool`` rejected).
            ValueError: If ``n`` is negative.

        Time Complexity:
            O(V).

        Space Complexity:
            O(V).

        Examples:
            >>> Network(3).adj
            [[], [], []]
            >>> Network(-1)
            Traceback (most recent call last):
                ...
            ValueError: n must be non-negative, got -1
            >>> Network(2.5)
            Traceback (most recent call last):
                ...
            TypeError: n must be an int, got float
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self.n: int = n
        self.to: List[int] = []
        self.cap: List[int] = []
        self.cost: List[int] = []
        self.orig_cap: List[int] = []
        self.adj: List[List[int]] = [[] for _ in range(n)]

    @property
    def m(self) -> int:
        """Return the number of forward edges.

        Each added edge stores two array entries, so this is half their
        number. Parallel edges count separately.

        Args:
            None.

        Returns:
            ``len(to) // 2``.

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> Network.from_edges(3, [(0, 1, 1), (0, 1, 2), (1, 2, 3)]).m
            3
        """
        return len(self.to) // 2

    def add_edge(self, u: int, v: int, capacity: int, cost: int = 0) -> int:
        """Add the edge u -> v and its residual reverse edge.

        Appends the forward edge at an even index ``i`` (residual capacity
        ``capacity``, cost ``cost``) and the reverse edge at ``i + 1`` (residual
        capacity 0, ``orig_cap`` 0, cost ``-cost``). Each call makes its own
        pair, so parallel edges and antiparallel pairs both work.

        Args:
            u: The tail, an ``int`` in ``0 .. n - 1``.
            v: The head, an ``int`` in ``0 .. n - 1``, different from ``u``.
            capacity: A non-negative ``int``.
            cost: The cost per unit of flow, an ``int``, possibly negative.

        Returns:
            The index of the forward edge, always even.

        Raises:
            TypeError: If a node, ``capacity`` or ``cost`` is not an ``int``
                (``bool`` rejected for ``capacity`` and ``cost``).
            ValueError: If a node is out of range, ``u == v``, or ``capacity``
                is negative.

        Time Complexity:
            O(1) amortized.

        Space Complexity:
            O(1) amortized.

        Examples:
            >>> net = Network(3)
            >>> net.add_edge(0, 1, 4), net.add_edge(1, 0, 2)
            (0, 2)
            >>> net.cap, net.adj[0], net.adj[1]
            ([4, 0, 2, 0], [0, 3], [1, 2])
            >>> net.add_edge(0, 3, 1)
            Traceback (most recent call last):
                ...
            ValueError: node out of range for n=3: (0, 3)
            >>> net.add_edge(1, 1, 1)
            Traceback (most recent call last):
                ...
            ValueError: self-loop at node 1
            >>> net.add_edge(0, 1, -1)
            Traceback (most recent call last):
                ...
            ValueError: capacity must be non-negative, got -1
            >>> net.add_edge(0, 1, 1.5)
            Traceback (most recent call last):
                ...
            TypeError: capacity must be an int, got float
            >>> net.add_edge(0, 1, True)
            Traceback (most recent call last):
                ...
            TypeError: capacity must be an int, got bool
        """
        if not (isinstance(u, int) and isinstance(v, int)):
            raise TypeError(f"nodes must be ints, got {u!r} and {v!r}")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity must be an int, got {type(capacity).__name__}")
        if isinstance(cost, bool) or not isinstance(cost, int):
            raise TypeError(f"cost must be an int, got {type(cost).__name__}")
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError(f"node out of range for n={self.n}: ({u}, {v})")
        if u == v:
            raise ValueError(f"self-loop at node {u}")
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        index = len(self.to)
        self.to += (v, u)
        self.cap += (capacity, 0)
        self.cost += (cost, -cost)
        self.orig_cap += (capacity, 0)
        self.adj[u].append(index)
        self.adj[v].append(index + 1)
        return index

    def get_residual_graph(self) -> List[List[Tuple[int, int]]]:
        """Return the residual graph as adjacency lists.

        Only edges with residual capacity above zero appear, forward or
        reverse. Parallel residual edges stay separate entries. Solvers read
        the arrays directly; this view is for inspection and tests.

        Args:
            None.

        Returns:
            A list with one entry per node, each a list of
            ``(v, residual capacity)`` tuples in ``adj`` order.

        Raises:
            None.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V + E).

        Examples:
            >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
            >>> net.get_residual_graph()
            [[(1, 5)], [(2, 3)], []]
            >>> net.cap[0] -= 2; net.cap[1] += 2
            >>> net.get_residual_graph()
            [[(1, 3)], [(0, 2), (2, 3)], []]
        """
        to, cap = self.to, self.cap
        return [[(to[e], cap[e]) for e in edges if cap[e] > 0] for edges in self.adj]

    def flow_on(self, edge_index: int) -> int:
        """Return the flow on an edge, read from its residual capacity.

        This is ``orig_cap[i] - cap[i]``. On a forward edge it is the flow
        carried; on a reverse edge it is minus that flow.

        Args:
            edge_index: An index into the edge arrays.

        Returns:
            The flow as an ``int``.

        Raises:
            IndexError: If ``edge_index`` is past the end of the arrays.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> net = Network.from_edges(2, [(0, 1, 5)])
            >>> net.cap[0] -= 2; net.cap[1] += 2
            >>> net.flow_on(0), net.flow_on(1)
            (2, -2)
        """
        return self.orig_cap[edge_index] - self.cap[edge_index]

    def flow_matrix(self) -> Dict[Tuple[int, int], int]:
        """Return the net flow between each pair of nodes.

        Parallel edges are summed. An antiparallel pair is netted, so 4 units
        on u -> v and 1 unit on v -> u report as 3 on (u, v) only.

        Args:
            None.

        Returns:
            A dict ``{(u, v): net flow}`` holding positive entries only.

        Raises:
            None.

        Time Complexity:
            O(E).

        Space Complexity:
            O(E).

        Examples:
            >>> net = Network.from_edges(2, [(0, 1, 5), (0, 1, 5), (1, 0, 5)])
            >>> net.flow_matrix()
            {}
            >>> for edge, units in [(0, 2), (2, 3), (4, 1)]:
            ...     net.cap[edge] -= units; net.cap[edge ^ 1] += units
            >>> net.flow_matrix()
            {(0, 1): 4}
            >>> net.cap[4] -= 4; net.cap[5] += 4
            >>> net.flow_matrix()
            {}
            >>> net.cap[0] += 1; net.cap[1] -= 1
            >>> net.flow_matrix()
            {(1, 0): 1}
        """
        to, cap, orig_cap = self.to, self.cap, self.orig_cap
        netted: Dict[Tuple[int, int], int] = {}
        for i in range(0, len(to), 2):
            units = orig_cap[i] - cap[i]
            if units:
                u, v = to[i ^ 1], to[i]
                if u < v:
                    netted[(u, v)] = netted.get((u, v), 0) + units
                else:
                    netted[(v, u)] = netted.get((v, u), 0) - units
        result: Dict[Tuple[int, int], int] = {}
        for (a, b), units in netted.items():
            if units > 0:
                result[(a, b)] = units
            elif units < 0:
                result[(b, a)] = -units
        return result

    def reset(self) -> None:
        """Restore every capacity, removing all flow.

        Copies ``orig_cap`` over ``cap`` in place, so an alias of ``cap`` held
        elsewhere stays valid. Use it to run another solver on the same graph.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        Time Complexity:
            O(E).

        Space Complexity:
            O(1) extra.

        Examples:
            >>> net = Network.from_edges(2, [(0, 1, 5)])
            >>> net.cap[0] -= 5; net.cap[1] += 5
            >>> net.reset()
            >>> net.cap, net.flow_matrix()
            ([5, 0], {})
        """
        self.cap[:] = self.orig_cap

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Sequence[int]]) -> "Network":
        """Build a network from a list of edge tuples.

        Args:
            n: The number of nodes.
            edges: ``(u, v, capacity)`` or ``(u, v, capacity, cost)`` tuples;
                the two forms may be mixed and the cost defaults to 0.

        Returns:
            A new :class:`Network` whose forward edges follow the input order.

        Raises:
            TypeError: As in :meth:`add_edge`.
            ValueError: As in :meth:`add_edge`, or if a tuple does not have 3
                or 4 items.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V + E).

        Examples:
            >>> net = Network.from_edges(3, [(0, 1, 4, 2), (1, 2, 3)])
            >>> list(net.forward_edges())
            [(0, 0, 1, 4, 2), (2, 1, 2, 3, 0)]
            >>> Network.from_edges(2, [(0, 1)])
            Traceback (most recent call last):
                ...
            ValueError: an edge must be (u, v, cap) or (u, v, cap, cost), got (0, 1)
        """
        net = cls(n)
        for edge in edges:
            if len(edge) not in (3, 4):
                raise ValueError(f"an edge must be (u, v, cap) or (u, v, cap, cost), got {edge!r}")
            net.add_edge(*edge)
        return net

    def forward_edges(self) -> Iterator[Tuple[int, int, int, int, int]]:
        """Yield every forward edge in insertion order.

        Args:
            None.

        Returns:
            An iterator of ``(edge_index, u, v, capacity, cost)``, where
            ``capacity`` is the original capacity, not the residual one.

        Raises:
            None.

        Time Complexity:
            O(E) for the full walk.

        Space Complexity:
            O(1).

        Examples:
            >>> net = Network.from_edges(3, [(0, 1, 4, 2), (2, 1, 3)])
            >>> net.cap[0] -= 1; net.cap[1] += 1
            >>> list(net.forward_edges())
            [(0, 0, 1, 4, 2), (2, 2, 1, 3, 0)]
        """
        to, orig_cap, cost = self.to, self.orig_cap, self.cost
        for i in range(0, len(to), 2):
            yield i, to[i ^ 1], to[i], orig_cap[i], cost[i]


def min_cut(net: Network, s: int) -> Set[int]:
    """Return the nodes reachable from ``s`` in the residual graph.

    After a maximum flow, t is not reachable, and this set is the source side
    of a minimum cut: every edge leaving it is saturated and every edge
    entering it carries no flow, so the cut capacity equals the flow value.
    It reads only the residual capacities, so it works after any solver.

    Args:
        net: The network, in whatever state a solver left it.
        s: The start node.

    Returns:
        A new ``set`` of node indices, always containing ``s``.

    Raises:
        ValueError: If ``s`` is out of range.

    Time Complexity:
        O(V + E), an iterative BFS.

    Space Complexity:
        O(V).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
        >>> sorted(min_cut(net, 0))
        [0, 1, 2]
        >>> net.cap[2] -= 3; net.cap[3] += 3
        >>> sorted(min_cut(net, 0))
        [0, 1]
        >>> min_cut(net, 9)
        Traceback (most recent call last):
            ...
        ValueError: s out of range for n=3: 9
    """
    if not 0 <= s < net.n:
        raise ValueError(f"s out of range for n={net.n}: {s}")
    to, cap, adj = net.to, net.cap, net.adj
    seen = [False] * net.n
    seen[s] = True
    queue = deque([s])
    while queue:
        for edge in adj[queue.popleft()]:
            v = to[edge]
            if cap[edge] > 0 and not seen[v]:
                seen[v] = True
                queue.append(v)
    return {node for node, reached in enumerate(seen) if reached}


def cut_edges(net: Network, s: int) -> List[Tuple[int, int]]:
    """Return the forward edges crossing from the reachable set to the rest.

    Args:
        net: The network, in whatever state a solver left it.
        s: The start node passed to :func:`min_cut`.

    Returns:
        A list of ``(u, v)`` pairs in insertion order; a parallel edge appears
        once per copy.

    Raises:
        ValueError: If ``s`` is out of range.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3), (2, 0, 9)])
        >>> net.cap[2] -= 3; net.cap[3] += 3
        >>> cut_edges(net, 0)
        [(1, 2)]
    """
    side = min_cut(net, s)
    return [(u, v) for _, u, v, _, _ in net.forward_edges() if u in side and v not in side]


def cut_value(net: Network, s: int) -> int:
    """Return the capacity of the cut around the nodes reachable from ``s``.

    After a maximum flow this equals the flow value (max-flow min-cut). Edges
    going back into the reachable set do not count.

    Args:
        net: The network, in whatever state a solver left it.
        s: The start node passed to :func:`min_cut`.

    Returns:
        The sum of the original capacities of the crossing edges.

    Raises:
        ValueError: If ``s`` is out of range.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3), (2, 0, 9)])
        >>> net.cap[2] -= 3; net.cap[3] += 3
        >>> cut_value(net, 0)
        3
    """
    side = min_cut(net, s)
    return sum(c for _, u, v, c, _ in net.forward_edges() if u in side and v not in side)
