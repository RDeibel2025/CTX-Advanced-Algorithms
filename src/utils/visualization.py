"""Supplementary charts for the Week 1 performance study.

:meth:`src.utils.benchmark.AlgorithmBenchmark.plot_comparison` answers the
first question — how does runtime grow with *n*? The three charts here
answer the follow-up questions that the Week 1 analysis actually turns on:

* :func:`plot_data_type_sensitivity` — at one fixed size, how much does the
  *order* of the input change each algorithm's runtime? This is the chart
  that shows insertion sort dropping away on sorted input while selection
  sort barely moves.
* :func:`plot_complexity_fit` — how well does each fitted reference model
  actually track the measurements? Plotting the fit next to the data is
  what keeps a reported R-squared honest.
* :func:`plot_normalized_runtime` — runtime divided by n^2. A genuinely
  quadratic algorithm flattens into a horizontal line here; anything that
  keeps falling is growing more slowly than n^2.

Week 4 adds the graph figures. Drawing a traversal is the one place in this
project where a picture carries an argument a table cannot: BFS and DFS
visit exactly the same nodes on the same graph and differ only in the order
they reach them, so the order is the entire result.

* :func:`graph_to_networkx` - convert a :class:`src.graphs.graph.Graph`
  into its networkx equivalent. networkx owns the layout algorithms; the
  project graph owns the data, and converting once keeps every figure
  reading the same adjacency the traversals ran on.
* :func:`plot_traversal_order` - one graph, every node labelled and
  coloured by the step at which the traversal reached it.
* :func:`plot_bfs_vs_dfs` - the same graph under the same layout twice,
  side by side, so the only visible difference between the panels is the
  visit order.

The layout seed is fixed (42 by default) because a spring layout starts
from random positions. Without a seed the same graph is drawn differently on
every run, the two panels of :func:`plot_bfs_vs_dfs` would not line up, and
the figure would compare two pictures instead of two traversals.

Every function returns the :class:`matplotlib.figure.Figure` it built and
writes a 200 dpi PNG when given ``save_path``.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import Colormap, Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.utils.benchmark import BenchmarkResult  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from src.graphs.graph import Graph

__all__ = [
    "apply_house_style",
    "plot_data_type_sensitivity",
    "plot_complexity_fit",
    "plot_normalized_runtime",
    "graph_to_networkx",
    "plot_traversal_order",
    "plot_bfs_vs_dfs",
]


def apply_house_style() -> None:
    """Apply one consistent look to every chart in the report.

    Uses seaborn's whitegrid theme so the figures in
    ``docs/performance_analysis.md`` share a single visual language rather
    than each carrying matplotlib's defaults.

    Examples:
        >>> apply_house_style() is None
        True
    """
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def _save(fig, save_path: Optional[str]) -> None:
    """Write ``fig`` to ``save_path`` as PNG, creating parent directories."""
    if not save_path:
        return
    directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(directory, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")


def plot_data_type_sensitivity(
    results: Dict[str, List[BenchmarkResult]],
    input_size: int,
    title: str = None,
    save_path: str = None,
    log_scale: bool = True,
):
    """Grouped bar chart of runtime by data type, at one fixed input size.

    Holding *n* constant isolates the effect of input *order*. An algorithm
    whose bars are all the same height does the same work no matter how the
    input is arranged; an algorithm with one short bar has a best case that
    the data type in question triggers.

    Args:
        results: ``{algorithm_name: [BenchmarkResult, ...]}``.
        input_size: Which measured size to slice at.
        title: Figure title. A sensible default is generated if omitted.
        save_path: Optional PNG destination.
        log_scale: Log scale on the runtime axis, so a hundred-fold
            difference between algorithms stays readable.

    Returns:
        The :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If no results match ``input_size``.

    Examples:
        >>> rows = {"A": [BenchmarkResult("A", 100, 0.01, 0.0, 0.01, 0.01,
        ...                               metadata={"data_type": "random"}),
        ...               BenchmarkResult("A", 100, 0.001, 0.0, 0.001, 0.001,
        ...                               metadata={"data_type": "sorted"})]}
        >>> fig = plot_data_type_sensitivity(rows, 100)
        >>> fig.axes[0].get_xlabel()
        'Input data type'
        >>> plt.close(fig)
    """
    sliced = {
        name: [r for r in series if r.input_size == input_size]
        for name, series in results.items()
    }
    sliced = {name: rows for name, rows in sliced.items() if rows}
    if not sliced:
        raise ValueError(f"no results recorded at input_size={input_size}")

    data_types: List[str] = []
    for rows in sliced.values():
        for r in rows:
            if r.data_type not in data_types:
                data_types.append(r.data_type)

    fig, ax = plt.subplots(figsize=(1.7 * max(len(data_types), 3) + 5, 5))
    positions = np.arange(len(data_types), dtype=float)
    width = 0.8 / max(len(sliced), 1)
    colors = plt.get_cmap("tab10").colors

    for index, (name, rows) in enumerate(sliced.items()):
        lookup = {r.data_type: r for r in rows}
        heights = [
            lookup[dt].average_time if dt in lookup else np.nan for dt in data_types
        ]
        errors = [
            lookup[dt].std_deviation if dt in lookup else 0.0 for dt in data_types
        ]
        ax.bar(
            positions + index * width - 0.4 + width / 2,
            heights,
            width=width * 0.92,
            yerr=errors,
            capsize=3,
            label=name,
            color=colors[index % len(colors)],
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(positions)
    ax.set_xticklabels(data_types, rotation=20, ha="right")
    ax.set_xlabel("Input data type")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title(
        title
        or f"Sensitivity to input order at n = {input_size:,} "
        f"({'log' if log_scale else 'linear'} runtime axis)"
    )
    # Placed outside the axes: the shortest bars are exactly where a
    # corner legend would sit, and those are the bars that carry the point.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    _save(fig, save_path)
    return fig


def plot_complexity_fit(
    analyses: Sequence[Dict[str, Any]],
    title: str = "Empirical complexity fits",
    save_path: str = None,
):
    """Plot measured runtimes against every fitted reference model.

    One panel per analysis. The measurements are drawn as markers and each
    candidate model — O(n), O(n log n), O(n^2) — as a line, with its
    R-squared in the legend. Showing the losing models alongside the
    winning one is what makes the reported best fit checkable rather than
    something the reader has to take on trust.

    Args:
        analyses: Dictionaries as returned by
            :meth:`src.utils.benchmark.AlgorithmBenchmark.analyze_complexity`.
        title: Figure title.
        save_path: Optional PNG destination.

    Returns:
        The :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If ``analyses`` is empty.

    Examples:
        >>> from src.utils.benchmark import AlgorithmBenchmark
        >>> made_up = [BenchmarkResult("q", n, 1e-8 * n * n, 0.0, 0.0, 0.0)
        ...            for n in (100, 200, 400, 800)]
        >>> report = AlgorithmBenchmark().analyze_complexity(made_up)
        >>> fig = plot_complexity_fit([report])
        >>> len(fig.axes)
        1
        >>> plt.close(fig)
    """
    analyses = [a for a in analyses if a and a.get("n_points", 0) >= 2]
    if not analyses:
        raise ValueError("no analyses with enough points to plot")

    ncols = min(len(analyses), 3)
    nrows = int(np.ceil(len(analyses) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 4.4 * nrows), squeeze=False
    )
    flat = [ax for row in axes for ax in row]

    basis = {
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * np.log2(np.maximum(n, 2)),
        "O(n^2)": lambda n: n ** 2,
    }

    for ax, analysis in zip(flat, analyses):
        sizes = np.array(analysis["sizes"], dtype=float)
        times = np.array(analysis["times"], dtype=float)
        ax.plot(sizes, times, "o", color="black", markersize=7, label="measured", zorder=5)

        smooth = np.linspace(sizes.min(), sizes.max(), 200)
        for model_name, info in analysis.get("models", {}).items():
            coefficient = info.get("coefficient", float("nan"))
            intercept = info.get("intercept", 0.0)
            r_squared = info.get("r_squared", float("nan"))
            if not np.isfinite(coefficient):
                continue
            is_best = model_name == analysis.get("best_fit")
            ax.plot(
                smooth,
                coefficient * basis[model_name](smooth) + intercept,
                linewidth=2.4 if is_best else 1.2,
                linestyle="-" if is_best else "--",
                alpha=1.0 if is_best else 0.7,
                label=f"{model_name}  R²={r_squared:.4f}"
                + ("  ← best" if is_best else ""),
            )

        exponent = analysis.get("empirical_exponent", float("nan"))
        subtitle = (
            f"log-log slope k = {exponent:.2f}" if np.isfinite(exponent) else "no slope"
        )
        data_types = ", ".join(analysis.get("data_types", []))
        ax.set_title(f"{analysis['algorithm_name']} — {data_types}\n{subtitle}")
        ax.set_xlabel("Input size n (elements)")
        ax.set_ylabel("Mean runtime (seconds)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    for spare in flat[len(analyses):]:
        spare.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    _save(fig, save_path)
    return fig


def plot_normalized_runtime(
    results: Dict[str, List[BenchmarkResult]],
    data_type: str = "random",
    title: str = None,
    save_path: str = None,
):
    """Plot runtime divided by n^2 against n.

    This is a direct visual test of the quadratic hypothesis. If
    ``t = c * n^2`` then ``t / n^2`` is the constant ``c``, so the series
    flattens into a horizontal line. A series that keeps sloping downward
    is growing more slowly than n^2; one that slopes upward is growing
    faster.

    Args:
        results: ``{algorithm_name: [BenchmarkResult, ...]}``.
        data_type: Which data shape to slice on.
        title: Figure title. Generated if omitted.
        save_path: Optional PNG destination.

    Returns:
        The :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If nothing matches ``data_type``.

    Examples:
        >>> rows = {"A": [BenchmarkResult("A", n, 1e-8 * n * n + 1e-6 * n,
        ...                               0.0, 0.0, 0.0,
        ...                               metadata={"data_type": "random"})
        ...               for n in (100, 200, 400)]}
        >>> fig = plot_normalized_runtime(rows, "random")
        >>> fig.axes[0].get_ylabel()
        'Runtime / n$^2$ (seconds per unit)'
        >>> plt.close(fig)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = plt.get_cmap("tab10").colors

    for index, (name, series) in enumerate(results.items()):
        points = sorted(
            (r for r in series if r.data_type == data_type and r.input_size > 0),
            key=lambda r: r.input_size,
        )
        if not points:
            continue
        sizes = np.array([r.input_size for r in points], dtype=float)
        times = np.array([r.average_time for r in points], dtype=float)
        ax.plot(
            sizes,
            times / sizes ** 2,
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            linewidth=1.8,
            markersize=6,
            label=name,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"no results recorded for data_type={data_type!r}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size n (elements)")
    ax.set_ylabel("Runtime / n$^2$ (seconds per unit)")
    ax.set_title(
        title
        or f"Runtime normalised by n² on {data_type} input\n"
        "(a horizontal line means the algorithm really is quadratic)"
    )
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    _save(fig, save_path)
    return fig


# ----------------------------------------------------------------------
# Week 4: graph and traversal figures
# ----------------------------------------------------------------------
# Tuned for the 15 to 25 node graphs the Week 4 instructions ask for: big
# enough nodes to carry a two-line label, small enough that the spring
# layout still separates them.
_TRAVERSAL_NODE_SIZE = 950
_TRAVERSAL_COLORMAP = "viridis"


def graph_to_networkx(graph: Graph) -> nx.Graph:
    """Convert a project :class:`~src.graphs.graph.Graph` into a networkx graph.

    The drawing helpers below need a networkx object because networkx owns
    the layout algorithms, while the project graph owns the data. Converting
    once, here, means every figure is drawn from the same adjacency the
    traversals actually walked rather than from a hand-rebuilt copy that
    could quietly drift out of step with it.

    Directedness is carried over: a directed graph becomes a
    :class:`networkx.DiGraph` and an undirected one a
    :class:`networkx.Graph`. Edge weights are copied into networkx's
    standard ``weight`` attribute when the source graph is weighted, and
    left off entirely when it is not, so an unweighted graph does not
    acquire a fictitious weight of 1.0. Nodes are added before edges, so an
    isolated node survives the conversion and still appears in the figure.

    Args:
        graph: Any graph exposing the Week 4 graph contract: the
            ``directed`` and ``weighted`` flags, ``nodes()`` and
            ``neighbor_items()``.

    Returns:
        A :class:`networkx.DiGraph` when ``graph.directed`` is True, else a
        :class:`networkx.Graph`, holding the same nodes and the same edges.

    Time Complexity:
        O(V + E). Every node is added once and every stored neighbour entry
        is read once, each at O(1) from the adjacency list. Rebuilding the
        same graph from an adjacency matrix would instead cost O(V^2),
        because every absent edge has to be looked at to be ruled out.

    Space Complexity:
        O(V + E) for the new networkx graph, which is itself a dict of dicts
        and so matches the adjacency list; the matrix form of the same graph
        would be O(V^2). The source graph is not modified.

    Examples:
        >>> from src.graphs.graph import Graph
        >>> roads = Graph(directed=True, weighted=True)
        >>> roads.add_edge("depot", "store", 4.0)
        >>> converted = graph_to_networkx(roads)
        >>> converted.is_directed(), converted.number_of_edges()
        (True, 1)
        >>> converted["depot"]["store"]["weight"]
        4.0

        An isolated node is kept, not dropped:

        >>> plain = Graph()
        >>> plain.add_edge(1, 2)
        >>> plain.add_node(3)
        >>> sorted(graph_to_networkx(plain).nodes())
        [1, 2, 3]
    """
    converted: nx.Graph = nx.DiGraph() if graph.directed else nx.Graph()
    weighted = graph.weighted
    for node in graph.nodes():
        converted.add_node(node)
    for node in graph.nodes():
        for neighbour, weight in graph.neighbor_items(node):
            if weighted:
                converted.add_edge(node, neighbour, weight=float(weight))
            else:
                converted.add_edge(node, neighbour)
    return converted


def _visit_steps(order: Sequence[Any]) -> Dict[Any, int]:
    """Map each visited node to its 1-based position in ``order``.

    ``setdefault`` keeps the *first* occurrence, so a traversal that
    re-reports a node is labelled with the step that actually discovered it.
    """
    steps: Dict[Any, int] = {}
    for position, node in enumerate(order, start=1):
        steps.setdefault(node, position)
    return steps


def _spring_positions(
    converted: nx.Graph, layout_seed: int
) -> Dict[Any, np.ndarray]:
    """Lay the graph out reproducibly with a seeded spring layout.

    Every figure in Week 4 goes through this one helper, which is what makes
    the two panels of :func:`plot_bfs_vs_dfs` superimposable: the same graph
    and the same seed give bit-identical coordinates.
    """
    return nx.spring_layout(converted, seed=layout_seed)


def _label_text_colour(rgba: Tuple[float, float, float, float]) -> str:
    """Pick black or white label text for a node of colour ``rgba``.

    A sequential colormap runs dark to light, so one fixed font colour is
    unreadable at one end of it. Choosing per node by perceived luminance
    keeps every step number legible.
    """
    red, green, blue = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.55 else "black"


def _draw_traversal_panel(
    converted: nx.Graph,
    positions: Dict[Any, np.ndarray],
    steps: Dict[Any, int],
    ax: Axes,
    title: str,
    cmap: Colormap,
) -> None:
    """Draw one traversal panel onto ``ax``: edges, nodes, labels, colourbar.

    Shared by :func:`plot_traversal_order` and, through it, by both panels of
    :func:`plot_bfs_vs_dfs`, so the two panels cannot drift apart stylistically.
    """
    total = max(len(steps), 1)
    norm = Normalize(vmin=1, vmax=total)
    visited = [node for node in converted.nodes if node in steps]
    missed = [node for node in converted.nodes if node not in steps]

    # arrowsize only applies to the FancyArrowPatch path networkx takes for
    # a directed graph; passing it for an undirected one warns, so it is
    # added conditionally rather than always.
    directed = converted.is_directed()
    arrow_style: Dict[str, Any] = {"arrowsize": 14} if directed else {}
    nx.draw_networkx_edges(
        converted,
        positions,
        ax=ax,
        edge_color="0.62",
        width=1.3,
        arrows=directed,
        node_size=_TRAVERSAL_NODE_SIZE,
        **arrow_style,
    )
    if missed:
        # White square inside a heavy grey rim: unmistakably off the walk.
        nx.draw_networkx_nodes(
            converted,
            positions,
            nodelist=missed,
            ax=ax,
            node_color="white",
            node_shape="s",
            node_size=_TRAVERSAL_NODE_SIZE,
            edgecolors="0.45",
            linewidths=1.6,
        )
    if visited:
        nx.draw_networkx_nodes(
            converted,
            positions,
            nodelist=visited,
            ax=ax,
            node_color=[steps[node] for node in visited],
            cmap=cmap,
            vmin=1,
            vmax=total,
            node_size=_TRAVERSAL_NODE_SIZE,
            edgecolors="black",
            linewidths=0.8,
        )

    # Labels are drawn in two passes, one per font colour, because
    # draw_networkx_labels takes a single colour for the whole batch.
    batches: Dict[str, Dict[Any, str]] = {"white": {}, "black": {}}
    for node in converted.nodes:
        if node in steps:
            face = cmap(norm(steps[node]))
            batches[_label_text_colour(face)][node] = f"{node}\n#{steps[node]}"
        else:
            batches["black"][node] = f"{node}\nx"
    for colour, labels in batches.items():
        if labels:
            nx.draw_networkx_labels(
                converted,
                positions,
                labels=labels,
                ax=ax,
                font_size=8,
                font_color=colour,
                font_weight="bold",
            )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.array([]))
    bar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    bar.set_ticks(
        np.unique(np.linspace(1, total, num=min(total, 6)).round().astype(int))
    )
    bar.set_label("Visit order (1 = visited first)")

    if missed:
        ax.scatter(
            [],
            [],
            marker="s",
            s=90,
            facecolors="white",
            edgecolors="0.45",
            linewidths=1.6,
            label="not visited",
        )
        ax.legend(loc="upper left", fontsize=8)

    ax.set_title(title)
    ax.margins(0.14)
    ax.set_axis_off()


def plot_traversal_order(
    graph: Graph,
    order: Sequence[Any],
    title: str,
    save_path: Optional[str] = None,
    *,
    layout_seed: int = 42,
    ax: Optional[Axes] = None,
) -> Figure:
    """Draw a graph with every node labelled and coloured by its visit step.

    A traversal returns a list, and a list of twenty node names is not
    something a reader can check against the picture in their head. Painting
    the step number onto the node turns that list back into the thing it
    describes: node ``#1`` is the start, the colour runs dark to light along
    the walk, and the shape of the walk - a ring spreading outward, or a
    single line diving to the far side of the graph - is visible without
    reading a single label.

    Nodes the traversal never reached are drawn as white squares marked
    ``x`` rather than being coloured or omitted, so a disconnected component
    reads as a real result instead of a missing one.

    Args:
        graph: The graph that was traversed.
        order: The traversal order, as returned by BFS or DFS. The first
            element is step 1. A repeated node is labelled with its first
            appearance.
        title: Title for the panel.
        save_path: Optional PNG destination, written at 200 dpi.
        layout_seed: Seed for the spring layout. Fixed by default so the
            same graph always lays out identically.
        ax: Optional existing axes to draw into, used by
            :func:`plot_bfs_vs_dfs` to put two traversals side by side. A
            new figure is created when it is omitted.

    Returns:
        The :class:`matplotlib.figure.Figure` holding the drawing, whether
        it was newly created or supplied through ``ax``.

    Raises:
        ValueError: If the graph has no nodes, or if ``order`` names a node
            that is not in the graph.

    Time Complexity:
        The layout dominates. Reading the graph is O(V + E), but networkx's
        Fruchterman-Reingold spring layout computes all-pairs repulsion,
        O(V^2) per iteration over a fixed iteration count, so the whole call
        is O(V^2). That is the honest reason this figure is specified for
        graphs of tens of nodes and the benchmarks are reported as tables:
        drawing does not scale the way traversing does.

    Space Complexity:
        O(V^2), again from the layout's all-pairs displacement array; the
        graph copy and the step map are only O(V + E).

    Examples:
        >>> import tempfile
        >>> from src.graphs.graph import Graph
        >>> town = Graph()
        >>> for left, right in [("a", "b"), ("b", "c"), ("a", "c"),
        ...                     ("c", "d")]:
        ...     town.add_edge(left, right)
        >>> png = os.path.join(tempfile.mkdtemp(), "bfs_order.png")
        >>> fig = plot_traversal_order(town, ["a", "b", "c", "d"],
        ...                            "BFS from a", png)
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        An unreachable node is drawn, not dropped:

        >>> town.add_node("island")
        >>> fig = plot_traversal_order(town, ["a", "b", "c", "d"], "BFS")
        >>> plt.close(fig)

        A node that is not in the graph is a caller error, not a blank spot:

        >>> plot_traversal_order(town, ["a", "ghost"], "BFS")
        Traceback (most recent call last):
            ...
        ValueError: traversal order contains nodes that are not in the graph: ghost
    """
    converted = graph_to_networkx(graph)
    if converted.number_of_nodes() == 0:
        raise ValueError("cannot draw a traversal on a graph with no nodes")

    steps = _visit_steps(order)
    unknown = [node for node in steps if node not in converted]
    if unknown:
        raise ValueError(
            "traversal order contains nodes that are not in the graph: "
            + ", ".join(sorted(str(node) for node in unknown))
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        owns_figure = True
    else:
        fig = ax.figure
        owns_figure = False

    _draw_traversal_panel(
        converted,
        _spring_positions(converted, layout_seed),
        steps,
        ax,
        title,
        plt.get_cmap(_TRAVERSAL_COLORMAP),
    )
    if owns_figure:
        fig.tight_layout()

    _save(fig, save_path)
    return fig


def plot_bfs_vs_dfs(
    graph: Graph,
    bfs_order: Sequence[Any],
    dfs_order: Sequence[Any],
    save_path: Optional[str] = None,
    *,
    layout_seed: int = 42,
) -> Figure:
    """Draw BFS and DFS on the same graph, same layout, side by side.

    This is the figure the report's Visualization Summary reads out loud,
    and it is built to support exactly one comparison. Both panels show the
    same nodes in the same positions, drawn from the same seeded layout, so
    every difference between them is a difference in visit order and nothing
    else. BFS colours spread outward in rings from the start node, because
    the FIFO queue finishes a whole distance band before opening the next
    one; DFS colours run in a chain into the graph and only then come back,
    because the LIFO stack always hands back the most recently discovered
    node. Same loop, same graph, different container.

    Args:
        graph: The graph both traversals walked.
        bfs_order: Visit order from breadth-first search.
        dfs_order: Visit order from depth-first search.
        save_path: Optional PNG destination, written at 200 dpi.
        layout_seed: Seed shared by both panels. Changing it moves the nodes
            in both panels together and never breaks the comparison.

    Returns:
        The two-panel :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If the graph has no nodes, or if either order names a
            node that is not in the graph.

    Time Complexity:
        O(V^2), dominated by the spring layout, which is computed once per
        panel from the same seed and therefore produces identical geometry
        both times.

    Space Complexity:
        O(V^2) for the layout, plus O(V + E) per panel for the graph copy.

    Examples:
        >>> import tempfile
        >>> from src.graphs.graph import Graph
        >>> town = Graph()
        >>> for left, right in [("a", "b"), ("a", "c"), ("b", "d"),
        ...                     ("c", "d")]:
        ...     town.add_edge(left, right)
        >>> png = os.path.join(tempfile.mkdtemp(), "bfs_vs_dfs.png")
        >>> fig = plot_bfs_vs_dfs(town, ["a", "b", "c", "d"],
        ...                       ["a", "b", "d", "c"], png)
        >>> len([axis for axis in fig.axes if axis.get_title()])
        2
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    plot_traversal_order(
        graph,
        bfs_order,
        "BFS - breadth first (FIFO queue)",
        layout_seed=layout_seed,
        ax=axes[0],
    )
    plot_traversal_order(
        graph,
        dfs_order,
        "DFS - depth first (LIFO stack)",
        layout_seed=layout_seed,
        ax=axes[1],
    )
    fig.suptitle(
        "Same graph, same layout, same nodes - only the visit order differs",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save(fig, save_path)
    return fig
