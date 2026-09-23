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

Week 5 adds the dynamic-programming figures. These three read the
benchmark's measurement rows directly - the same list of dicts that becomes
``benchmarks/results/dp_vs_recursive_table.csv`` - so the figures and the
table cannot disagree about what was run.

* :func:`plot_fibonacci_comparison` - runtime against n on a logarithmic
  axis, one series per variant. The naive recursion runs out of budget
  before n = 45, so the points past the measured range are drawn hollow on
  a dashed segment and labelled as projections rather than being quietly
  dropped or quietly passed off as measurements.
* :func:`plot_knapsack_performance` - two panels, because the knapsack
  table is O(n x W) and one pair of axes can only move one of those two
  factors at a time.
* :func:`plot_lcs_performance` - runtime against string length, with the
  range plain recursion was actually run over shaded and named, so its
  short series reads as a stated limit rather than as a line that stops
  for no reason.

Any numeric field in those rows can arrive as an empty string, which is how
the benchmark records a value it did not produce. Every Week 5 function
treats that as missing and drops the point. None of them substitutes a
number the benchmark never measured.

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
    "plot_fibonacci_comparison",
    "plot_knapsack_performance",
    "plot_lcs_performance",
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


# ----------------------------------------------------------------------
# Week 5: dynamic programming figures
# ----------------------------------------------------------------------
# The benchmark hands these three functions its measurement rows unchanged:
# one dict per (problem, variant, input size), carrying exactly the columns
# of benchmarks/results/dp_vs_recursive_table.csv. Reading the same rows the
# CSV is written from, rather than a reshaped copy, is what keeps the
# figures and the table from ever disagreeing about what was run.
_DP_VARIANT_ORDER: Tuple[str, ...] = ("naive", "recursive", "memo", "tab")

# Colour, marker and line style all change together from variant to
# variant. A reader who cannot separate the colours - greyscale print,
# colour blindness, a projector - still has two other channels to read the
# series by, which is this project's standing rule for every figure.
_DP_VARIANT_STYLE: Dict[str, Dict[str, Any]] = {
    "naive": {"color": "#c1121f", "marker": "o", "linestyle": "-"},
    "recursive": {"color": "#c1121f", "marker": "o", "linestyle": "-"},
    "memo": {"color": "#1d3557", "marker": "s", "linestyle": "--"},
    "tab": {"color": "#2a9d8f", "marker": "^", "linestyle": "-."},
}
# "naive" and "recursive" share a style deliberately: no problem uses both
# names, so across the three figures one red circle always means "the
# version that remembers nothing".
_DP_VARIANT_LABEL: Dict[str, str] = {
    "naive": "naive recursion",
    "recursive": "plain recursion",
    "memo": "memoization (top-down)",
    "tab": "tabulation (bottom-up)",
}
_DP_FALLBACK_MARKERS: Tuple[str, ...] = ("D", "v", "P", "X", "*")
_DP_FALLBACK_LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":")

# The wording matters more than it looks: a projected point is arithmetic,
# not a stopwatch reading, and the figure has to say which it is.
_PROJECTED_LEGEND_TEXT = (
    "projected, not timed: measured per-call cost\n"
    "multiplied by the exact call count"
)


def _row_number(row: Dict[str, Any], key: str) -> Optional[float]:
    """Read one numeric field out of a benchmark row, or ``None``.

    The Week 5 CSV leaves a cell empty wherever the benchmark had no value
    to put in it: ``speedup_vs_recursive`` on a projected row, ``calls`` on
    a variant that was never instrumented. Read back, those cells arrive as
    empty strings where a number would be. Every numeric read in the Week 5
    figures goes through here and comes back either as a finite float or as
    ``None``, so a missing measurement can be dropped from the chart
    instead of being replaced by a number nobody measured.

    Args:
        row: One benchmark measurement row.
        key: The column to read.

    Returns:
        The value as a float, or ``None`` when the key is absent, empty,
        unparseable, or not finite.

    Examples:
        >>> _row_number({"mean_time_s": "0.25"}, "mean_time_s")
        0.25
        >>> _row_number({"mean_time_s": 0.5}, "mean_time_s")
        0.5
        >>> _row_number({"speedup_vs_recursive": ""}, "speedup_vs_recursive")
        >>> _row_number({}, "calls")
        >>> _row_number({"n": "n/a"}, "n")
    """
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def _row_text(row: Dict[str, Any], key: str) -> str:
    """Read one string field out of a benchmark row, lower-cased and stripped.

    Used for ``problem``, ``variant`` and ``measurement``, the three columns
    that are compared against literals. Normalising here means a stray
    ``"Projected"`` or ``" tab"`` from a hand-edited CSV still matches.
    """
    value = row.get(key)
    return "" if value is None else str(value).strip().lower()


def _format_measure(value: float) -> str:
    """Format a number for an axis label, a title or a legend entry.

    Input sizes arrive as floats because they came through a CSV, but
    ``W = 1,000`` reads better in a panel title than ``W = 1000.0``.
    """
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:g}"


def _dp_rows(rows: Sequence[Dict[str, Any]], problem: str) -> List[Dict[str, Any]]:
    """Select the rows belonging to one problem: fibonacci, knapsack or lcs."""
    return [row for row in rows if _row_text(row, "problem") == problem]


def _ordered_variants(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """List the variants present, known ones first and in a fixed order.

    Fixing the order fixes the legend order, so the three Week 5 figures
    list their series the same way and can be read as a set rather than as
    three unrelated charts. A variant the project does not know about is
    still drawn, appended in the order it first appears.
    """
    seen: List[str] = []
    for row in rows:
        variant = _row_text(row, "variant")
        if variant and variant not in seen:
            seen.append(variant)
    known = [name for name in _DP_VARIANT_ORDER if name in seen]
    return known + [name for name in seen if name not in _DP_VARIANT_ORDER]


def _variant_style(variant: str, index: int) -> Dict[str, Any]:
    """Return the colour, marker and line style for one variant."""
    style = _DP_VARIANT_STYLE.get(variant)
    if style is not None:
        return dict(style)
    palette = plt.get_cmap("tab10").colors
    return {
        "color": palette[(index + 4) % len(palette)],
        "marker": _DP_FALLBACK_MARKERS[index % len(_DP_FALLBACK_MARKERS)],
        "linestyle": _DP_FALLBACK_LINESTYLES[index % len(_DP_FALLBACK_LINESTYLES)],
    }


def _dp_points(
    rows: Sequence[Dict[str, Any]], x_key: str
) -> List[Tuple[float, float, bool]]:
    """Turn rows into ``(x, mean seconds, is_projected)`` triples, sorted by x.

    A row with no x value, or no mean time, contributes nothing: there is
    no honest place to put it on the axes. A non-positive mean time is
    dropped for the same reason, since every Week 5 figure uses a
    logarithmic runtime axis and a zero has no position on one.
    """
    points: List[Tuple[float, float, bool]] = []
    for row in rows:
        x_value = _row_number(row, x_key)
        y_value = _row_number(row, "mean_time_s")
        if x_value is None or y_value is None or y_value <= 0.0:
            continue
        points.append((x_value, y_value, _row_text(row, "measurement") == "projected"))
    points.sort(key=lambda point: point[0])
    return points


def _plot_variant_series(
    ax: Axes,
    points: Sequence[Tuple[float, float, bool]],
    style: Dict[str, Any],
    label: str,
) -> bool:
    """Draw one variant's series, keeping projected points visibly apart.

    Measured points are filled markers on the variant's own line style.
    Projected points are hollow markers on a dashed segment that continues
    the curve from the last measured point, so the projection reads as an
    extension of the measurement it was computed from and never as another
    reading of the clock.

    Returns:
        True if any projected point was drawn, which tells the caller
        whether the figure needs the projected legend entry.
    """
    measured = [(x, y) for x, y, projected in points if not projected]
    projected = [(x, y) for x, y, projected in points if projected]

    if measured:
        ax.plot(
            [x for x, _ in measured],
            [y for _, y in measured],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.9,
            markersize=6.5,
            label=label,
            zorder=3,
        )
    if not projected:
        return False

    bridge = measured[-1:] + projected
    ax.plot(
        [x for x, _ in bridge],
        [y for _, y in bridge],
        color=style["color"],
        linestyle="--",
        linewidth=1.5,
        marker="",
        alpha=0.9,
        zorder=2,
    )
    ax.plot(
        [x for x, _ in projected],
        [y for _, y in projected],
        color=style["color"],
        marker=style["marker"],
        linestyle="",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor=style["color"],
        markeredgewidth=1.9,
        zorder=4,
        label=None if measured else f"{label} (projected)",
    )
    return True


def _draw_dp_series(
    ax: Axes, rows: Sequence[Dict[str, Any]], x_key: str
) -> Tuple[int, bool]:
    """Draw every variant present in ``rows`` onto one axes.

    Returns:
        ``(series drawn, any projected point drawn)``.
    """
    drawn = 0
    any_projected = False
    for index, variant in enumerate(_ordered_variants(rows)):
        subset = [row for row in rows if _row_text(row, "variant") == variant]
        points = _dp_points(subset, x_key)
        if not points:
            continue
        label = _DP_VARIANT_LABEL.get(variant, variant)
        if _plot_variant_series(ax, points, _variant_style(variant, index), label):
            any_projected = True
        drawn += 1
    return drawn, any_projected


def _add_projected_legend_entry(ax: Axes) -> None:
    """Add the proxy handle that explains what a hollow marker means.

    The projected points carry no label of their own, because repeating
    "projected" once per variant would crowd the legend and still not say
    where the number came from. One neutral entry says it once.
    """
    ax.plot(
        [],
        [],
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="0.35",
        markeredgewidth=1.9,
        label=_PROJECTED_LEGEND_TEXT,
    )


def _dominant_value(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    """Pick the value of ``key`` shared by the most rows, largest winning ties.

    This is how :func:`plot_knapsack_performance` chooses what to hold
    fixed when the caller does not say. The value the benchmark swept the
    most points at is the one it meant as the control, and breaking ties on
    the larger value keeps the choice deterministic across runs.
    """
    tally: Dict[float, int] = {}
    for row in rows:
        value = _row_number(row, key)
        if value is None:
            continue
        tally[value] = tally.get(value, 0) + 1
    if not tally:
        return None
    return max(tally, key=lambda value: (tally[value], value))


def _rows_at(
    rows: Sequence[Dict[str, Any]], key: str, value: float
) -> List[Dict[str, Any]]:
    """Select rows whose numeric ``key`` equals ``value``."""
    return [row for row in rows if _row_number(row, key) == value]


def plot_fibonacci_comparison(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot Fibonacci runtime against n for every variant, on a log axis.

    This is the figure that makes the whole week's point in one picture.
    The naive recursion and the two rememberers compute the same numbers,
    and on a linear axis the DP series would be flat against the bottom of
    the frame with nothing readable in them. On a logarithmic runtime axis
    the naive series is a straight climbing line - which is what an
    exponential looks like once the axis is logged - while memoization and
    tabulation stay very nearly flat.

    The naive series cannot be measured all the way out. A call to
    ``fib_naive`` at n = 45 makes 3,672,623,805 of them, which is minutes
    per run and far longer than a benchmark loop with warmups and repeats.
    Those points are therefore projected from the per-call cost measured on
    this machine and the exact closed-form call count, and the figure has
    to say so: projected points are drawn as hollow markers on a dashed
    continuation of the measured curve, with their own legend entry. A
    projection presented as a measurement would be the most serious thing
    this chart could get wrong, so it is marked in three ways at once -
    marker fill, line style and legend text.

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/dp_vs_recursive_table.csv``. Rows for
            other problems are ignored. Any numeric field may be an empty
            string; a row missing ``n`` or ``mean_time_s`` is skipped
            rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If no row carries ``problem == "fibonacci"`` together
            with a usable ``n`` and ``mean_time_s``.

    Time Complexity:
        O(R log R) in the number of Fibonacci rows R. Each row is read a
        constant number of times and each variant's points are sorted by n.

    Space Complexity:
        O(R) for the extracted points, plus the figure itself.

    Examples:
        >>> import tempfile
        >>> def row(variant, n, seconds, measurement="measured"):
        ...     return {"problem": "fibonacci", "variant": variant, "n": n,
        ...             "secondary_param": "", "mean_time_s": seconds,
        ...             "std_time_s": "", "min_time_s": "", "max_time_s": "",
        ...             "calls": "", "max_depth": "", "peak_kib": "",
        ...             "speedup_vs_recursive": "", "measurement": measurement}
        >>> rows = [row("naive", 20, 0.0021), row("naive", 30, 0.26),
        ...         row("naive", 40, 32.4, "projected"),
        ...         row("memo", 20, 6.2e-06), row("memo", 30, 9.4e-06),
        ...         row("memo", 40, 1.3e-05),
        ...         row("tab", 20, 1.1e-06), row("tab", 30, 1.6e-06),
        ...         row("tab", 40, 2.1e-06)]
        >>> png = os.path.join(tempfile.mkdtemp(), "fibonacci_comparison.png")
        >>> fig = plot_fibonacci_comparison(rows, png)
        >>> fig.axes[0].get_yscale()
        'log'
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        A row whose mean time was never produced is dropped, not invented:

        >>> fig = plot_fibonacci_comparison(rows + [row("naive", 45, "")])
        >>> plt.close(fig)

        Nothing to draw is a caller error, not an empty picture:

        >>> plot_fibonacci_comparison([])
        Traceback (most recent call last):
            ...
        ValueError: no fibonacci rows with a usable n and mean_time_s
    """
    apply_house_style()
    fibonacci_rows = _dp_rows(rows, "fibonacci")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    drawn, any_projected = _draw_dp_series(ax, fibonacci_rows, "n")
    if drawn == 0:
        plt.close(fig)
        raise ValueError("no fibonacci rows with a usable n and mean_time_s")
    if any_projected:
        _add_projected_legend_entry(ax)

    ax.set_yscale("log")
    ticks = sorted({point[0] for point in _dp_points(fibonacci_rows, "n")})
    if 0 < len(ticks) <= 12:
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_measure(tick) for tick in ticks])
    ax.set_xlabel("Fibonacci index n (term number)")
    ax.set_ylabel("Mean runtime (seconds, log scale)")
    ax.set_title(
        "Fibonacci: one exponential recursion against two that remember\n"
        "(log runtime axis; hollow markers on dashes are projected)"
    )
    # Upper left is the one corner the data cannot reach: the naive series
    # climbs from the bottom left and the DP series stay along the floor.
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    _save(fig, save_path)
    return fig


def plot_knapsack_performance(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
    *,
    fixed_capacity: Optional[float] = None,
    fixed_n: Optional[float] = None,
) -> Figure:
    """Plot knapsack runtime against item count and against capacity.

    The 0/1 knapsack table is O(n x W), and a single pair of axes can only
    move one of those two factors at a time. Two panels is not a layout
    preference here, it is the shape of the cost: the left panel holds the
    capacity fixed and sweeps the item count, the right panel holds the
    item count fixed and sweeps the capacity. Read together they show the
    product, and each panel is titled with the value it held still so
    neither can be mistaken for the whole story.

    Both panels use a logarithmic runtime axis, because the recursive
    variant is exponential in n while the two DP variants are linear in the
    table size, and no linear axis holds both.

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/dp_vs_recursive_table.csv``. Rows for
            other problems are ignored. Any numeric field may be an empty
            string, and such a row is skipped rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.
        fixed_capacity: The capacity the left panel holds fixed, matched
            against ``secondary_param``. When omitted, the capacity that
            the most knapsack rows were measured at is used, since that is
            the sweep the benchmark treated as its control.
        fixed_n: The item count the right panel holds fixed, matched
            against ``n``. Chosen the same way when omitted.

    Returns:
        The two-panel :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If there are no knapsack rows at all, if no row carries
            the column a panel sweeps against, or if an explicitly
            requested ``fixed_capacity`` or ``fixed_n`` matches no row.
            Silently drawing an empty panel would read as "the DP variants
            were not run", which is a different claim entirely.

    Time Complexity:
        O(R log R) in the number of knapsack rows R: a constant number of
        passes to tally and select, then a sort per variant per panel.

    Space Complexity:
        O(R) for the selected rows and extracted points, plus the figure.

    Examples:
        >>> import tempfile
        >>> def row(variant, n, capacity, seconds):
        ...     return {"problem": "knapsack", "variant": variant, "n": n,
        ...             "secondary_param": capacity, "mean_time_s": seconds,
        ...             "std_time_s": "", "min_time_s": "", "max_time_s": "",
        ...             "calls": "", "max_depth": "", "peak_kib": "",
        ...             "speedup_vs_recursive": "", "measurement": "measured"}
        >>> rows = [row("recursive", 10, 100, 0.004),
        ...         row("recursive", 14, 100, 0.07),
        ...         row("memo", 10, 100, 0.0006), row("memo", 14, 100, 0.0009),
        ...         row("tab", 10, 100, 0.0004), row("tab", 14, 100, 0.0006),
        ...         row("memo", 14, 200, 0.0017), row("tab", 14, 200, 0.0011)]
        >>> png = os.path.join(tempfile.mkdtemp(), "knapsack_performance.png")
        >>> fig = plot_knapsack_performance(rows, png, fixed_capacity=100,
        ...                                 fixed_n=14)
        >>> len(fig.axes)
        2
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        Left alone, each panel holds fixed whatever was measured most:

        >>> fig = plot_knapsack_performance(rows)
        >>> fig.axes[0].get_title()
        'Item count n at fixed capacity W = 100'
        >>> fig.axes[1].get_title()
        'Capacity W at fixed item count n = 14'
        >>> plt.close(fig)

        A capacity that was never run is a caller error, not a blank panel:

        >>> plot_knapsack_performance(rows, fixed_capacity=999)
        Traceback (most recent call last):
            ...
        ValueError: no knapsack rows at capacity W = 999
    """
    apply_house_style()
    knapsack_rows = _dp_rows(rows, "knapsack")
    if not knapsack_rows:
        raise ValueError("no knapsack rows to plot")

    capacity = (
        _dominant_value(knapsack_rows, "secondary_param")
        if fixed_capacity is None
        else float(fixed_capacity)
    )
    item_count = (
        _dominant_value(knapsack_rows, "n") if fixed_n is None else float(fixed_n)
    )
    if capacity is None:
        raise ValueError("no knapsack rows carry a capacity in secondary_param")
    if item_count is None:
        raise ValueError("no knapsack rows carry an item count in n")

    by_capacity = _rows_at(knapsack_rows, "secondary_param", capacity)
    by_item_count = _rows_at(knapsack_rows, "n", item_count)
    if not by_capacity:
        raise ValueError(
            f"no knapsack rows at capacity W = {_format_measure(capacity)}"
        )
    if not by_item_count:
        raise ValueError(
            f"no knapsack rows at item count n = {_format_measure(item_count)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    panels = (
        (
            axes[0],
            by_capacity,
            "n",
            "Number of items n (items)",
            f"Item count n at fixed capacity W = {_format_measure(capacity)}",
        ),
        (
            axes[1],
            by_item_count,
            "secondary_param",
            "Knapsack capacity W (weight units)",
            f"Capacity W at fixed item count n = {_format_measure(item_count)}",
        ),
    )
    for ax, panel_rows, x_key, xlabel, title in panels:
        drawn, any_projected = _draw_dp_series(ax, panel_rows, x_key)
        if drawn == 0:
            plt.close(fig)
            raise ValueError(f"no usable mean_time_s among the rows for: {title}")
        if any_projected:
            _add_projected_legend_entry(ax)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean runtime (seconds, log scale)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.suptitle(
        "0/1 knapsack: the table is O(n x W), so each panel moves one factor",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save(fig, save_path)
    return fig


def plot_lcs_performance(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot LCS runtime against string length, marking where recursion stopped.

    The DP variants run the full length range the assignment asks for. The
    plain recursion does not, and cannot: it is O(2^(n+m)), so it was run
    only over the short lengths where it finishes at all. Drawn naively,
    that series would simply stop part-way across the figure, which reads
    like a gap in the data rather than a stated limit of the experiment.

    So the range the recursion was actually run over is shaded, its right
    edge is drawn as a boundary line, and the legend names the lengths. The
    short series is then evidence - this is where exponential growth ran
    out of budget - instead of an omission the reader has to notice.

    Both axes are logarithmic. Lengths sweep two orders of magnitude and
    runtimes sweep more, and on a log-log pair a polynomial cost is a
    straight line whose slope is its exponent, which is the comparison the
    report makes against the theoretical O(n x m).

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/dp_vs_recursive_table.csv``. Rows for
            other problems are ignored. ``n`` is the string length. Any
            numeric field may be an empty string, and such a row is skipped
            rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If no row carries ``problem == "lcs"`` together with a
            usable ``n`` and ``mean_time_s``.

    Time Complexity:
        O(R log R) in the number of LCS rows R: each row is read a constant
        number of times, and each variant's points are sorted by length.

    Space Complexity:
        O(R) for the extracted points, plus the figure itself.

    Examples:
        >>> import tempfile
        >>> def row(variant, length, seconds):
        ...     return {"problem": "lcs", "variant": variant, "n": length,
        ...             "secondary_param": length, "mean_time_s": seconds,
        ...             "std_time_s": "", "min_time_s": "", "max_time_s": "",
        ...             "calls": "", "max_depth": "", "peak_kib": "",
        ...             "speedup_vs_recursive": "", "measurement": "measured"}
        >>> rows = [row("recursive", 10, 0.0012), row("recursive", 14, 0.02),
        ...         row("memo", 10, 0.0002), row("memo", 100, 0.02),
        ...         row("tab", 10, 9e-05), row("tab", 100, 0.008),
        ...         row("tab", 1000, 0.82)]
        >>> png = os.path.join(tempfile.mkdtemp(), "lcs_performance.png")
        >>> fig = plot_lcs_performance(rows, png)
        >>> os.path.getsize(png) > 0
        True

        The legend states the limit rather than leaving a line to trail off:

        >>> any("only over lengths 10 to 14" in text.get_text()
        ...     for text in fig.axes[0].get_legend().get_texts())
        True
        >>> plt.close(fig)

        With no recursive rows the figure is simply the DP series, unshaded:

        >>> dp_only = [item for item in rows if item["variant"] != "recursive"]
        >>> fig = plot_lcs_performance(dp_only)
        >>> plt.close(fig)
    """
    apply_house_style()
    lcs_rows = _dp_rows(rows, "lcs")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    drawn, any_projected = _draw_dp_series(ax, lcs_rows, "n")
    if drawn == 0:
        plt.close(fig)
        raise ValueError("no lcs rows with a usable n and mean_time_s")
    if any_projected:
        _add_projected_legend_entry(ax)

    recursive_points = _dp_points(
        [row for row in lcs_rows if _row_text(row, "variant") == "recursive"], "n"
    )
    longest = max(point[0] for point in _dp_points(lcs_rows, "n"))
    if recursive_points:
        shortest_run = recursive_points[0][0]
        longest_run = recursive_points[-1][0]
        band_label = (
            f"plain recursion run only over lengths "
            f"{_format_measure(shortest_run)} to {_format_measure(longest_run)}\n"
            "(beyond it, only the DP variants were measured)"
        )
        if longest_run > shortest_run:
            ax.axvspan(
                shortest_run,
                longest_run,
                color="#c1121f",
                alpha=0.10,
                zorder=0,
                label=band_label,
            )
            # The band carries the legend entry; this line only marks where
            # it ends, which is the number the report quotes.
            ax.axvline(
                longest_run,
                color="#c1121f",
                linestyle=":",
                linewidth=1.4,
                zorder=1,
            )
        else:
            # A single recursive point has no band to shade, so the
            # boundary line carries the legend entry on its own.
            ax.axvline(
                longest_run,
                color="#c1121f",
                linestyle=":",
                linewidth=1.4,
                zorder=1,
                label=band_label,
            )
        # Pinned to the foot of the boundary line rather than to the last
        # recursive point, which sits in the middle of the DP series.
        ax.annotate(
            "exponential cost ends the series here",
            xy=(longest_run, 0.03),
            xycoords=("data", "axes fraction"),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#7f1d1d",
        )
        title_tail = f"plain recursion only to {_format_measure(longest_run)}"
    else:
        title_tail = "no recursive series in these rows"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input string length (characters per string)")
    ax.set_ylabel("Mean runtime (seconds, log scale)")
    ax.set_title(
        f"LCS: DP measured to {_format_measure(longest)} characters, "
        f"{title_tail}\n(log-log axes; a polynomial cost is a straight line)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    _save(fig, save_path)
    return fig
