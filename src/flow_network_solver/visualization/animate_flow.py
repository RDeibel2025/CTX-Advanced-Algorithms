"""Animate Edmonds-Karp and push-relabel as GIFs of their recorded steps.

The two animations show the two ways of reading the residual graph.

Edmonds-Karp
------------
Edmonds-Karp reads a whole s-t path out of the residual graph and pushes along
it, so the flow is valid after every step. :func:`animate_edmonds_karp` shows one
frame per augmenting path, with that path bold and the ``flow/capacity`` labels
as they stand after it, using
:func:`~src.flow_network_solver.visualization.plot_network.plot_network`.

Push-relabel
------------
Push-relabel reads only the edges at one node. A height is a lower bound on the
residual distance to t, and a node pushes excess only to a residual neighbour
exactly one level below, so during the run the state is a preflow rather than a
flow. :func:`animate_push_relabel` therefore plots the labels themselves: each
node has its own column, the y-axis is its height, and its label gives its
excess, so a frame shows excess running downhill. The node just acted on has a
heavy outline and the title names the operation. Excess above zero is the only
thing that fills a node, and the number in the label says the same thing, so
colour is never the only signal. The recorded steps hold labels and not edge
flows, so edges here are plain context with no ``flow/capacity`` label.

Frames
------
Both solvers run with ``record=True`` on a reset net. A GIF has at most
:data:`MAX_FRAMES` frames: longer runs are subsampled evenly, always keeping the
first and last step, and titles keep the original step numbers. If a GIF comes
out larger than :data:`MAX_BYTES` it is rendered again at a lower resolution.
Frames are written with :class:`matplotlib.animation.PillowWriter`, so ffmpeg is
not needed, and on an Agg canvas, so no window opens.

Examples:
    >>> _even_sample(list(range(3)))
    [0, 1, 2]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from matplotlib.animation import PillowWriter
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.network import Network
from src.flow_network_solver.core.push_relabel import push_relabel
from src.flow_network_solver.visualization.plot_network import (
    BOLD,
    EDGE_COLOR,
    INK,
    NODE_DIAMETER,
    PATH_COLOR,
    SINK_FILL,
    SOURCE_FILL,
    _capacities,
    _draw_arrow,
    _new_axes,
    layered_layout,
    plot_network,
)

__all__ = ["animate_edmonds_karp", "animate_push_relabel"]

MAX_FRAMES = 40
MAX_BYTES = 2_000_000
GIF_DPIS = (100, 75, 50)
HEIGHT_FRAME_CURVE = 0.15


def _even_sample(items: Sequence[Any], limit: int = MAX_FRAMES) -> List[Any]:
    """Return at most ``limit`` evenly spaced items, keeping the first and last.

    Examples:
        >>> sampled = _even_sample(list(range(100)))
        >>> len(sampled), sampled[0], sampled[-1]
        (40, 0, 99)
        >>> len(set(sampled))
        40
    """
    count = len(items)
    if count <= limit:
        return list(items)
    return [items[i * (count - 1) // (limit - 1)] for i in range(limit)]


def _write_gif(
    ax: Axes,
    frames: Sequence[Any],
    draw: Callable[[Any], None],
    out_path: str,
    fps: int,
) -> str:
    """Render ``draw(frame)`` for each frame into a GIF within the size limit."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for dpi in GIF_DPIS:
        writer = PillowWriter(fps=fps)
        with writer.saving(ax.figure, str(out_path), dpi=dpi):
            for frame in frames:
                draw(frame)
                writer.grab_frame()
        if os.path.getsize(out_path) <= MAX_BYTES:
            break
    return out_path


def _operation_text(step: Dict[str, Any]) -> str:
    """Name the operation of one recorded push-relabel step."""
    op, node = step["op"], step["node"]
    if op == "init":
        return f"saturate the edges out of node {node}"
    if op == "global_relabel":
        return "global relabel"
    if op == "push":
        return f"push from node {node}"
    return f"relabel node {node}"


def _draw_heights(
    ax: Axes,
    net: Network,
    s: int,
    t: int,
    column: Dict[int, int],
    top: int,
    step: Dict[str, Any],
    title: str,
) -> None:
    """Draw one push-relabel state: nodes at their heights, excess in labels."""
    ax.clear()
    heights, excess = step["heights"], step["excess"]
    place = {node: (column[node], heights[node]) for node in range(net.n)}

    for u, v in _capacities(net):
        _draw_arrow(ax, place[u], place[v], curve=HEIGHT_FRAME_CURVE, width=0.8, color=EDGE_COLOR)

    for node, (x, y) in place.items():
        active = node not in (s, t) and excess[node] > 0
        ax.scatter(
            [x],
            [y],
            s=NODE_DIAMETER**2,
            facecolor=SINK_FILL if active else SOURCE_FILL,
            edgecolor=PATH_COLOR if node == step["node"] else INK,
            linewidth=BOLD if node == step["node"] else 1.2,
            zorder=3,
        )
        ax.text(x, y, str(node), ha="center", va="center", fontsize=11, color=INK, zorder=4)
        ax.annotate(
            f"e={excess[node]}",
            (x, y),
            xytext=(0, NODE_DIAMETER / 2 + 9),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none"},
            zorder=5,
        )

    ticks = sorted(column, key=column.get)
    names = {s: f"{s} (s)", t: f"{t} (t)"}
    ax.set_xticks([column[node] for node in ticks], [names.get(node, str(node)) for node in ticks])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.6, net.n - 0.4)
    ax.set_ylim(-0.8, top + 1.7)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("node")
    ax.set_ylabel("height")
    ax.set_title(title)


def animate_edmonds_karp(net: Network, s: int, t: int, out_path: str) -> str:
    """Save a GIF of Edmonds-Karp, one frame per augmenting path.

    Resets the net and solves it with ``record=True``. Each frame draws the path
    in bold with the ``flow/capacity`` labels after that augmentation, and its
    title gives the path, bottleneck and running value. With no augmenting path
    at all the GIF is one frame of the empty flow. The net keeps the final flow.

    Args:
        net: The network. It is reset and solved in place.
        s: The source.
        t: The sink.
        out_path: Where to write the GIF. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        ValueError: If ``s`` or ``t`` is out of range, or ``s == t``.

    Time Complexity:
        O(V E^2) to solve, plus O(V + E) to draw each of at most 40 frames.

    Space Complexity:
        O(P E) for the recorded steps, where P is the number of augmenting
        paths, plus the frames.

    Examples:
        >>> import os, tempfile
        >>> from PIL import Image
        >>> net = Network.from_edges(4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)])
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = animate_edmonds_karp(net, 0, 3, os.path.join(folder, "ek.gif"))
        ...     with Image.open(out) as gif:
        ...         print(gif.n_frames, gif.size)
        2 (800, 450)
        >>> net.flow_matrix()
        {(0, 1): 2, (0, 2): 2, (1, 3): 2, (2, 3): 2}

        With no path the GIF still has one frame.

        >>> net = Network.from_edges(3, [(1, 2, 5)])
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = animate_edmonds_karp(net, 0, 2, os.path.join(folder, "ek.gif"))
        ...     with Image.open(out) as gif:
        ...         print(gif.n_frames)
        1
        >>> animate_edmonds_karp(net, 1, 1, "unused.gif")
        Traceback (most recent call last):
            ...
        ValueError: s and t must differ, got 1
    """
    net.reset()
    steps = edmonds_karp(net, s, t, record=True).steps
    total = len(steps)
    layout = layered_layout(net, s)
    ax = _new_axes()

    def draw(frame: Tuple[int, Dict[str, Any]]) -> None:
        number, step = frame
        ax.clear()
        plot_network(net, flow=step["flow"], path=step["path"], ax=ax, layout=layout)
        route = " -> ".join(str(node) for node in step["path"])
        bottleneck, value = step["bottleneck"], step["value"]
        ax.set_title(f"Path {number} of {total}: {route}, bottleneck {bottleneck}, flow {value}")

    def draw_empty(_: int) -> None:
        ax.clear()
        plot_network(net, flow={}, ax=ax, layout=layout)
        ax.set_title(f"No path from {s} to {t}: flow 0")

    if not steps:
        return _write_gif(ax, [0], draw_empty, out_path, fps=1)
    return _write_gif(ax, _even_sample(list(enumerate(steps, 1))), draw, out_path, fps=1)


def animate_push_relabel(net: Network, s: int, t: int, out_path: str) -> str:
    """Save a GIF of push-relabel with nodes placed by height.

    Resets the net and solves it with ``record=True``. Each node keeps one column,
    ordered by BFS depth from s, and moves on the y-axis as its height changes, so
    two nodes at one height never overlap. Each label gives the node's excess
    (negative at s, the running value at t). The node just acted on has a heavy
    outline, and a filled node holds excess. Edges are shown without labels,
    because the recorded steps hold no edge flows. Runs over 40 operations are
    subsampled. The net keeps the final flow.

    Args:
        net: The network. It is reset and solved in place.
        s: The source.
        t: The sink.
        out_path: Where to write the GIF. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        ValueError: If ``s`` or ``t`` is out of range, or ``s == t``.

    Time Complexity:
        O(V^3) to solve, plus O(V) per recorded operation and O(V + E) to draw
        each of at most 40 frames.

    Space Complexity:
        O(V) per recorded operation, plus the frames.

    Examples:
        The five operations are the ones listed in the ``push_relabel`` docstring.

        >>> import os, tempfile
        >>> from PIL import Image
        >>> net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = animate_push_relabel(net, 0, 2, os.path.join(folder, "pr.gif"))
        ...     with Image.open(out) as gif:
        ...         print(gif.n_frames, gif.size)
        5 (800, 450)
        >>> net.flow_matrix()
        {(0, 1): 3, (1, 2): 3}
        >>> animate_push_relabel(net, 0, 9, "unused.gif")
        Traceback (most recent call last):
            ...
        ValueError: s or t out of range for n=3: (0, 9)
    """
    net.reset()
    steps = push_relabel(net, s, t, record=True).steps
    total = len(steps)
    layout = layered_layout(net, s)
    order = sorted(range(net.n), key=lambda node: (layout[node][0], -layout[node][1]))
    column = {node: index for index, node in enumerate(order)}
    top = max(max(step["heights"]) for step in steps)
    ax = _new_axes()
    # This view keeps its axes, so it needs room for the tick and axis labels.
    ax.figure.subplots_adjust(left=0.08, bottom=0.14)

    def draw(frame: Tuple[int, Dict[str, Any]]) -> None:
        number, step = frame
        title = f"Step {number} of {total}: {_operation_text(step)}; excess at t = {step['value']}"
        _draw_heights(ax, net, s, t, column, top, step, title)

    return _write_gif(ax, _even_sample(list(enumerate(steps, 1))), draw, out_path, fps=2)
