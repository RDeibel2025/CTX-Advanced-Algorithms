"""Draw a flow network with its flow, its minimum cut and a highlighted path.

What the picture shows
----------------------
:func:`plot_network` draws one network on matplotlib Axes:

* each edge is labelled ``flow/capacity``;
* edges crossing the cut are dashed, and sink-side nodes are squares where
  source-side nodes are circles;
* the edges of a highlighted path are bold.

Every one of these is carried by line style, line width or shape as well as by
colour, so the figure survives a greyscale printout.

How it reads the residual graph
-------------------------------
Nothing is solved here; the drawing only reads what a solver left behind. A
solver stores no flow, only residual capacities, so the flow on a forward edge
is ``orig_cap - cap``, which is what
:meth:`~src.flow_network_solver.core.network.Network.flow_matrix` reports when
``flow`` is not passed. The cut is also a residual reading:
:func:`~src.flow_network_solver.core.network.min_cut` returns the nodes still
reachable from s, and the dashed edges are the forward edges leaving that set.
A path found in the residual graph may cross a reverse edge, which has no line
of its own, so that step highlights the forward edge it undoes.

Parallel edges share one line whose label sums them. An antiparallel pair is
drawn as two arcs, and the flow matrix reports it netted, so the direction
carrying no net flow shows 0.

Layout
------
:func:`layered_layout` puts each node in a column by its BFS depth from the
source, and hands the columns to networkx for placement. A left-to-right layout
reads the way a flow runs and reproduces the CLRS drawing of the anchor network,
which a spring layout does not. networkx is used for this layout and nothing
else: the edges and nodes are drawn with matplotlib directly.

Backend
-------
This module draws on :class:`~matplotlib.figure.Figure` objects with an explicit
Agg canvas rather than on pyplot figures, so it opens no window and leaves nothing
in pyplot's figure registry.

Examples:
    >>> edges = [(0, 1, 16), (0, 2, 13), (2, 1, 4), (1, 3, 12), (3, 2, 9),
    ...          (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]
    >>> ax = plot_network(Network.from_edges(6, edges))
    >>> [text.get_text() for text in ax.texts if "/" in text.get_text()][:3]
    ['0/16', '0/13', '0/4']

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.network import Network, cut_value, min_cut
from src.utils.visualization import apply_house_style

__all__ = ["layered_layout", "plot_network", "save_network_figure"]

Point = Tuple[float, float]

INK = "#0b0b0b"
EDGE_COLOR = "#8a8984"
PATH_COLOR = "#2a78d6"
CUT_COLOR = "#eb6834"
SOURCE_FILL = "#ffffff"
SINK_FILL = "#e4e3de"

NODE_DIAMETER = 30.0  # points
# Arrows stop NODE_DIAMETER / 2 + 1 from a node centre. This side keeps a square's
# corner inside that radius, so a diagonal arrowhead is not drawn under the node.
SQUARE_SIDE = 22.0  # points
THIN = 1.3
BOLD = 3.4
ANTIPARALLEL_CURVE = 0.18


def _new_axes() -> Axes:
    """Return fresh Axes on an Agg canvas, sized for one network."""
    apply_house_style()
    figure = Figure(figsize=(8, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    figure.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08)
    return axes


def _capacities(net: Network) -> Dict[Tuple[int, int], int]:
    """Return ``{(u, v): capacity}`` with parallel edges summed."""
    capacity: Dict[Tuple[int, int], int] = {}
    for _, u, v, c, _ in net.forward_edges():
        capacity[(u, v)] = capacity.get((u, v), 0) + c
    return capacity


def _draw_arrow(
    ax: Axes,
    tail: Point,
    head: Point,
    curve: float = 0.0,
    width: float = THIN,
    style: str = "-",
    color: str = EDGE_COLOR,
) -> None:
    """Add one arrow from ``tail`` to ``head``, stopping at the node markers."""
    gap = NODE_DIAMETER / 2 + 1
    ax.add_patch(
        FancyArrowPatch(
            tail,
            head,
            arrowstyle="-|>",
            mutation_scale=11 + 2 * width,
            connectionstyle=f"arc3,rad={curve}",
            linewidth=width,
            linestyle=style,
            color=color,
            shrinkA=gap,
            shrinkB=gap,
            zorder=1,
        )
    )


def _path_edges(path: Sequence[int], capacity: Dict[Tuple[int, int], int]) -> Set[Tuple[int, int]]:
    """Map a node path to the forward edges it uses.

    A step with no forward edge of its own crossed a reverse edge, so it marks
    the forward edge it undoes. A step along neither direction is an error.
    """
    marked: Set[Tuple[int, int]] = set()
    for u, v in zip(path, path[1:]):
        if (u, v) in capacity:
            marked.add((u, v))
        elif (v, u) in capacity:
            marked.add((v, u))
        else:
            raise ValueError(f"path step {u} -> {v} is not an edge")
    return marked


def layered_layout(net: Network, s: int = 0) -> Dict[int, Point]:
    """Place nodes left to right by BFS depth from ``s``.

    Each BFS depth becomes one column, with lower node indices higher in the
    column. Nodes that ``s`` cannot reach share one extra column on the right.
    The depth uses every forward edge, whatever its residual capacity, so the
    layout is the same before and after a solver runs.

    Args:
        net: The network.
        s: The node in the leftmost column.

    Returns:
        A dict ``{node: (x, y)}`` with coordinates in about [-1, 1].

    Raises:
        ValueError: If ``s`` is out of range.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> net = Network.from_edges(4, [(0, 1, 1), (0, 2, 1), (1, 3, 1), (2, 3, 1)])
        >>> layout = layered_layout(net)
        >>> layout[0][0] < layout[1][0] == layout[2][0] < layout[3][0]
        True
        >>> layout[1][1] > layout[2][1]
        True

        A node that s cannot reach goes to an extra column.

        >>> layout = layered_layout(Network.from_edges(3, [(0, 1, 1)]))
        >>> layout[0][0] < layout[1][0] < layout[2][0]
        True
        >>> layered_layout(net, 9)
        Traceback (most recent call last):
            ...
        ValueError: s out of range for n=4: 9
    """
    if not 0 <= s < net.n:
        raise ValueError(f"s out of range for n={net.n}: {s}")
    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n))
    graph.add_edges_from((u, v) for _, u, v, _, _ in net.forward_edges())
    depth = nx.single_source_shortest_path_length(graph, s)
    unreached = max(depth.values()) + 1
    layers: Dict[int, List[int]] = defaultdict(list)
    for node in range(net.n):
        layers[depth.get(node, unreached)].append(node)
    placed = nx.multipartite_layout(graph, subset_key=dict(layers))
    # Flip y so the lowest index in a column is on top, as CLRS draws v1 over v2.
    return {node: (float(x), -float(y)) for node, (x, y) in placed.items()}


def plot_network(
    net: Network,
    flow: Optional[Dict[Tuple[int, int], int]] = None,
    cut: Optional[Set[int]] = None,
    path: Optional[Sequence[int]] = None,
    ax: Optional[Axes] = None,
    layout: Optional[Dict[int, Point]] = None,
) -> Axes:
    """Draw a network, labelling each edge ``flow/capacity``.

    Edges from ``cut`` to outside it are dashed, and with a cut the nodes outside
    it are squares. The edges of ``path`` are bold. Parallel edges share a line
    with their capacities summed, and an antiparallel pair is two arcs. A legend
    names the line styles when a cut or a path is drawn. This is meant for graphs
    small enough to read; nothing limits the size.

    Args:
        net: The network. It is not changed.
        flow: A flow matrix ``{(u, v): units}``. ``None`` reads the net's current
            flow with ``net.flow_matrix()``, which is all zeros on an unsolved net.
        cut: The source side of a cut, for example from ``min_cut``. ``None``
            draws no cut.
        path: Nodes to highlight, in order. A step against a forward edge
            highlights that edge, because a residual path may cross a reverse edge.
        ax: The Axes to draw on. ``None`` creates a figure on an Agg canvas.
        layout: A dict ``{node: (x, y)}``. ``None`` uses :func:`layered_layout`
            from ``path[0]``, or from node 0 without a path.

    Returns:
        The matplotlib Axes drawn on. ``ax.figure`` saves the figure.

    Raises:
        ValueError: If a step of ``path`` follows no edge, or if ``layout`` is
            ``None`` and the start node is out of range (as on an empty net).
        KeyError: If ``layout`` leaves out a node.

    Time Complexity:
        O(V + E), plus the layout's O(V + E) when ``layout`` is ``None``.

    Space Complexity:
        O(V + E).

    Examples:
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3), (0, 2, 2)])
        >>> result = edmonds_karp(net, 0, 2)
        >>> ax = plot_network(net, flow=result.flow, cut=min_cut(net, 0), path=[0, 1, 2])
        >>> [text.get_text() for text in ax.texts if "/" in text.get_text()]
        ['3/5', '3/3', '2/2']

        The two edges leaving {0, 1} are dashed, and the two on the path are bold.

        >>> sum(arrow.get_linestyle() == "--" for arrow in ax.patches)
        2
        >>> sum(arrow.get_linewidth() == BOLD for arrow in ax.patches)
        2

        Parallel edges are summed, and flow defaults to what the net holds.

        >>> ax = plot_network(Network.from_edges(2, [(0, 1, 3), (0, 1, 4)]))
        >>> [text.get_text() for text in ax.texts if "/" in text.get_text()]
        ['0/7']
        >>> plot_network(net, path=[2, 0, 9])
        Traceback (most recent call last):
            ...
        ValueError: path step 0 -> 9 is not an edge
    """
    if ax is None:
        ax = _new_axes()
    capacity = _capacities(net)
    highlighted = _path_edges(path, capacity) if path else set()
    if layout is None:
        layout = layered_layout(net, path[0] if path else 0)
    if flow is None:
        flow = net.flow_matrix()

    for (u, v), limit in capacity.items():
        on_cut = cut is not None and u in cut and v not in cut
        on_path = (u, v) in highlighted
        curve = ANTIPARALLEL_CURVE if (v, u) in capacity else 0.0
        (x1, y1), (x2, y2) = layout[u], layout[v]
        _draw_arrow(
            ax,
            (x1, y1),
            (x2, y2),
            curve=curve,
            width=BOLD if on_path else THIN,
            style="--" if on_cut else "-",
            color=PATH_COLOR if on_path else CUT_COLOR if on_cut else EDGE_COLOR,
        )
        # The middle of an arc3 Bezier sits half the curve off the chord. The
        # equal aspect set below keeps data offsets true on screen.
        ax.text(
            (x1 + x2) / 2 + curve * (y2 - y1) / 2,
            (y1 + y2) / 2 - curve * (x2 - x1) / 2,
            f"{flow.get((u, v), 0)}/{limit}",
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"},
            zorder=2,
        )

    for node in range(net.n):
        x, y = layout[node]
        on_sink_side = cut is not None and node not in cut
        ax.scatter(
            [x],
            [y],
            s=SQUARE_SIDE**2 if on_sink_side else NODE_DIAMETER**2,
            marker="s" if on_sink_side else "o",
            facecolor=SINK_FILL if on_sink_side else SOURCE_FILL,
            edgecolor=INK,
            linewidth=1.2,
            zorder=3,
        )
        ax.text(x, y, str(node), ha="center", va="center", fontsize=11, color=INK, zorder=4)

    handles = []
    if path:
        handles.append(
            Line2D([], [], color=PATH_COLOR, linewidth=BOLD, label="highlighted path (bold)")
        )
    if cut is not None:
        handles.append(
            Line2D(
                [],
                [],
                color=CUT_COLOR,
                linestyle="--",
                linewidth=THIN,
                label="min-cut edge (dashed)",
            )
        )
        for marker, fill, label in (
            ("o", SOURCE_FILL, "source side"),
            ("s", SINK_FILL, "sink side"),
        ):
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    markersize=9,
                    markerfacecolor=fill,
                    markeredgecolor=INK,
                    label=label,
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(handles),
            frameon=False,
            fontsize=9,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    ax.set_axis_off()
    return ax


def save_network_figure(net: Network, s: int, t: int, out_path: str) -> str:
    """Save a solved network with its minimum cut as a PNG.

    Nothing is solved here. The net must already hold a maximum flow from any
    solver; the flow labels and the cut are both read from its residual state,
    and the title states the cut, which equals the flow value by max-flow min-cut.

    Args:
        net: A solved network. It is not changed.
        s: The source.
        t: The sink.
        out_path: Where to write the PNG. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        ValueError: If ``s`` is out of range, or if t is reachable from s in the
            residual graph, which means the net does not hold a maximum flow.

    Time Complexity:
        O(V + E) plus the rendering.

    Space Complexity:
        O(V + E) plus the image.

    Examples:
        >>> import os, tempfile
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
        >>> save_network_figure(net, 0, 2, "unused.png")
        Traceback (most recent call last):
            ...
        ValueError: t = 2 is reachable from s = 0, so the net holds no maximum flow
        >>> edmonds_karp(net, 0, 2).value
        3
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = save_network_figure(net, 0, 2, os.path.join(folder, "net.png"))
        ...     os.path.getsize(out) > 0
        True
    """
    side = min_cut(net, s)
    if t in side:
        raise ValueError(f"t = {t} is reachable from s = {s}, so the net holds no maximum flow")
    ax = plot_network(net, cut=side, layout=layered_layout(net, s))
    value = cut_value(net, s)
    ax.set_title(f"Max flow = min cut = {value}; source side {sorted(side)}; s = {s}, t = {t}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out_path, bbox_inches="tight")
    return out_path
