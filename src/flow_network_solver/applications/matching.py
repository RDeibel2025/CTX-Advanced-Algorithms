"""Maximum bipartite matching, read from the saturated edges of a unit network.

A *matching* in a bipartite graph is a set of left-right edges in which no node
appears twice. Finding a largest one looks like a separate problem, but it is a
maximum flow on a network with unit capacities:

* source -> each left node, capacity 1,
* left -> right for each input edge, capacity 1,
* each right node -> sink, capacity 1.

Why the flow is a matching
--------------------------
Capacities are integers, so a maximum flow can be taken integral, and both
solvers here produce one. Every edge then carries 0 or 1. Each left node
receives at most 1 unit and so sends out at most 1, and each right node
forwards at most 1 and so receives at most 1. The left-right edges carrying a
unit therefore share no node. Conversely, any matching of size k gives a flow of
value k by routing one unit through each pair. So the maximum flow value is
exactly the maximum matching size.

How it reads the residual graph
-------------------------------
The solver leaves its final residual graph in the network, and the answer is
read straight from it. A left-right edge has capacity 1, so it carries a unit
exactly when its residual capacity is 0: the matching is the set of *saturated*
left-right edges. The network is acyclic, so no unit can be stuck on a cycle and
every saturated edge lies on a real source-sink path.

Reverse edges are what make the result optimal rather than greedy. With edges
a-1, a-2 and b-1, a first path through a-1 leaves b stranded, but the residual
path source, b, 1, a, 2, sink crosses the reverse of a-1, which moves a to 2 and
gives b the node 1. This is the classic alternating path, which here is simply
an augmenting path in the residual graph.

================== ================================== =============
Function           Time                               Space
================== ================================== =============
bipartite_matching ``"ek"``: O(min(L, R) (V + E)),    O(V + E)
                   one BFS per matched pair
                   ``"pr"``: O(V^3) worst case
draw_matching      O(V + E)                           O(V + E)
================== ================================== =============

Here L and R are the side sizes, V = L + R + 2 and E is the number of network
edges, L + R plus the input edges.

Examples:
    >>> left, right = ["a", "b", "c", "d"], [1, 2, 3, 4]
    >>> edges = [("a", 1), ("a", 2), ("b", 1), ("c", 2), ("c", 3), ("d", 3)]
    >>> len(bipartite_matching(left, right, edges))
    3
    >>> len(bipartite_matching(left, right, edges, solver="pr"))
    3

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.network import Network
from src.flow_network_solver.core.push_relabel import push_relabel

__all__ = ["bipartite_matching", "draw_matching"]

#: The max-flow solvers a caller can pick by name.
_SOLVERS = {"ek": edmonds_karp, "pr": push_relabel}

#: Node ids in the matching network; left and right nodes follow from 2.
_SOURCE, _SINK = 0, 1


def _positions(labels: Iterable[Hashable], side: str) -> Dict[Hashable, int]:
    """Map each label of one side to its index, rejecting repeated labels.

    Args:
        labels: The node labels of one side, in order.
        side: ``"left"`` or ``"right"``, used in the error message.

    Returns:
        A dict ``{label: index}``.

    Raises:
        ValueError: If a label appears twice.

    Time Complexity:
        O(k) for k labels.

    Space Complexity:
        O(k).

    Examples:
        >>> _positions(["a", "b"], "left")
        {'a': 0, 'b': 1}
        >>> _positions(["a", "a"], "left")
        Traceback (most recent call last):
            ...
        ValueError: repeated left label 'a'
    """
    positions: Dict[Hashable, int] = {}
    for label in labels:
        if label in positions:
            raise ValueError(f"repeated {side} label {label!r}")
        positions[label] = len(positions)
    return positions


def _indexed_graph(
    left: Iterable[Hashable],
    right: Iterable[Hashable],
    edges: Iterable[Sequence[Hashable]],
) -> Tuple[Dict[Hashable, int], Dict[Hashable, int], List[Tuple[Hashable, Hashable]]]:
    """Index both sides and check that every edge joins a left and a right node.

    The two sides are indexed separately, so one label may appear on both.

    Args:
        left: The left node labels.
        right: The right node labels.
        edges: The ``(l, r)`` pairs.

    Returns:
        ``(left positions, right positions, edges as a list of tuples)``.

    Raises:
        ValueError: If a side repeats a label or an edge has an unknown end.

    Time Complexity:
        O(L + R + edges).

    Space Complexity:
        O(L + R + edges).

    Examples:
        >>> _indexed_graph(["a"], [1], [["a", 1]])
        ({'a': 0}, {1: 0}, [('a', 1)])
        >>> _indexed_graph(["a"], [1], [("a", 2)])
        Traceback (most recent call last):
            ...
        ValueError: edge ('a', 2) is not a left-right pair
    """
    left_at = _positions(left, "left")
    right_at = _positions(right, "right")
    pairs = []
    for l, r in edges:
        if l not in left_at or r not in right_at:
            raise ValueError(f"edge {(l, r)!r} is not a left-right pair")
        pairs.append((l, r))
    return left_at, right_at, pairs


def bipartite_matching(
    left: Iterable[Hashable],
    right: Iterable[Hashable],
    edges: Iterable[Sequence[Hashable]],
    solver: str = "ek",
) -> List[Tuple[Hashable, Hashable]]:
    """Find a maximum matching of a bipartite graph with one max-flow run.

    Builds the unit network from the module docstring, solves it, and reads the
    pairs off the saturated left-right edges of the final residual graph: an
    edge belongs to the matching when ``net.cap[edge] == 0``. Repeating an edge
    in ``edges`` adds a parallel network edge and cannot change the result, since
    each left node sends at most one unit.

    Args:
        left: The left node labels, hashable and distinct.
        right: The right node labels, hashable and distinct. A label may appear
            on both sides; the sides are kept apart.
        edges: The ``(l, r)`` pairs, each joining a left and a right node.
        solver: ``"ek"`` for Edmonds-Karp or ``"pr"`` for push-relabel.

    Returns:
        The matched ``(l, r)`` pairs as a list, in input edge order. Different
        solvers can return different pairs, always the same number of them.

    Raises:
        ValueError: If ``solver`` is unknown, a side repeats a label, or an edge
            is not a left-right pair.

    Time Complexity:
        O(min(L, R) (V + E)) for ``"ek"``, since every augmenting path adds one
        pair and takes one BFS. O(V^3) worst case for ``"pr"``.

    Space Complexity:
        O(V + E).

    Examples:
        The first path takes a-1. The second has to undo it through the reverse
        edge, which moves a to 2.

        >>> bipartite_matching(["a", "b"], [1, 2], [("a", 1), ("a", 2), ("b", 1)])
        [('a', 2), ('b', 1)]

        Right node 4 has no edges, so three pairs is the most possible.

        >>> edges = [("a", 1), ("a", 2), ("b", 1), ("c", 2), ("c", 3), ("d", 3)]
        >>> pairs = bipartite_matching("abcd", [1, 2, 3, 4], edges, solver="pr")
        >>> len(pairs), set(pairs) <= set(edges)
        (3, True)
        >>> bipartite_matching([], [1], [])
        []
        >>> bipartite_matching(["a"], [1], [], solver="dinic")
        Traceback (most recent call last):
            ...
        ValueError: unknown solver 'dinic', expected 'ek' or 'pr'
    """
    if solver not in _SOLVERS:
        raise ValueError(f"unknown solver {solver!r}, expected 'ek' or 'pr'")
    left_at, right_at, pairs = _indexed_graph(left, right, edges)
    first_left, first_right = 2, 2 + len(left_at)
    net = Network(first_right + len(right_at))
    for i in left_at.values():
        net.add_edge(_SOURCE, first_left + i, 1)
    for j in right_at.values():
        net.add_edge(first_right + j, _SINK, 1)
    pair_edges = [
        net.add_edge(first_left + left_at[l], first_right + right_at[r], 1) for l, r in pairs
    ]
    _SOLVERS[solver](net, _SOURCE, _SINK)
    return [pair for pair, edge in zip(pairs, pair_edges) if net.cap[edge] == 0]


def draw_matching(
    left: Iterable[Hashable],
    right: Iterable[Hashable],
    edges: Iterable[Sequence[Hashable]],
    matching: Iterable[Sequence[Hashable]],
    out_path: str,
) -> str:
    """Save a two-column picture of a bipartite graph with its matching.

    Left nodes form one column and right nodes the other. Matched edges are bold
    and dashed, the others thin and solid, so the two stay apart without colour.
    The parent folder of ``out_path`` is created if needed. Plotting imports stay
    inside this function so that solving never loads matplotlib.

    Args:
        left: The left node labels.
        right: The right node labels.
        edges: The ``(l, r)`` pairs of the graph.
        matching: The matched pairs, each one an input edge.
        out_path: Where to save the PNG.

    Returns:
        ``out_path``, unchanged.

    Raises:
        ValueError: If a side repeats a label, an edge is not a left-right pair,
            or a matched pair is not an input edge.

    Time Complexity:
        O(V + E) plus rendering.

    Space Complexity:
        O(V + E).

    Examples:
        >>> import os, tempfile
        >>> edges = [("a", 1), ("a", 2), ("b", 1)]
        >>> pairs = bipartite_matching(["a", "b"], [1, 2], edges)
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = os.path.join(folder, "matching.png")
        ...     draw_matching(["a", "b"], [1, 2], edges, pairs, out) == out
        ...     os.path.getsize(out) > 0
        True
        True
        >>> draw_matching(["a"], [1], [], [("a", 1)], "unused.png")
        Traceback (most recent call last):
            ...
        ValueError: matched pair ('a', 1) is not an input edge
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from src.utils.visualization import apply_house_style

    left_at, right_at, pairs = _indexed_graph(left, right, edges)
    matched = {tuple(pair) for pair in matching}
    unknown = matched - set(pairs)
    if unknown:
        raise ValueError(f"matched pair {min(unknown, key=repr)!r} is not an input edge")

    def height(index: int, count: int) -> float:
        """Return the y position of a node, keeping index 0 at the top."""
        return (count - 1 - index) / max(1, count - 1)

    left_y = {label: height(i, len(left_at)) for label, i in left_at.items()}
    right_y = {label: height(j, len(right_at)) for label, j in right_at.items()}

    apply_house_style()
    fig = Figure(figsize=(6.0, max(3.0, 0.5 * max(len(left_at), len(right_at)) + 1.0)))
    ax = fig.subplots()
    for l, r in pairs:
        is_matched = (l, r) in matched
        ax.plot(
            [0, 1],
            [left_y[l], right_y[r]],
            color="#1f2933" if is_matched else "#8a94a0",
            linewidth=2.6 if is_matched else 0.8,
            linestyle="--" if is_matched else "-",
            zorder=3 if is_matched else 2,
        )
    ax.plot([0] * len(left_y), list(left_y.values()), "o", color="#1f2933", zorder=4)
    ax.plot([1] * len(right_y), list(right_y.values()), "s", color="#1f2933", zorder=4)
    for label, y in left_y.items():
        ax.text(-0.04, y, str(label), ha="right", va="center")
    for label, y in right_y.items():
        ax.text(1.04, y, str(label), ha="left", va="center")
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(f"Matching: {len(matched)} pairs")
    ax.legend(
        handles=[
            Line2D([], [], color="#1f2933", linewidth=2.6, linestyle="--", label="matched"),
            Line2D([], [], color="#8a94a0", linewidth=0.8, linestyle="-", label="not matched"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=2,
    )
    ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    return out_path
