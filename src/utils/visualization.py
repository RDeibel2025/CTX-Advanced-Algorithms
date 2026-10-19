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

Week 6 adds the advanced dynamic-programming figures. They read the rows
behind ``benchmarks/results/comparison_table.csv`` the same way, and each
one is built around the single comparison the week's analysis rests on:

* :func:`plot_knapsack_space_comparison` - peak memory, runtime and the
  memory ratio of Week 5's full 2D knapsack table against the one-row
  version, all over capacity W at one fixed item count. Space is the only
  thing the rolling row is meant to change, so memory gets the log axis and
  runtime is there to show that it did not get worse.
* :func:`plot_mcm_performance` - matrix-chain runtime for plain recursion,
  memoization and the bottom-up table, beside a panel showing how many
  times faster the table is than the recursion.
* :func:`plot_floyd_warshall_scaling` - Floyd-Warshall against an n^3
  reference, against Dijkstra run from every source at each edge density,
  and against the 3D version that keeps every D(k) layer.
* :func:`plot_tsp_runtime` - Held-Karp against brute force, each beside the
  growth shape it should follow (n^2 2^n and n!), with the first size at
  which the bitmask DP wins marked on the speedup panel.
* :func:`plot_mcm_table` - the CLRS m table itself as a heatmap, every cell
  carrying its cost and its optimal split point s[i][j].

The Week 6 functions keep the Week 5 rule for missing values: an empty
string is skipped and never filled in, and a ``"projected"`` row is drawn
as a hollow marker on a dashed segment. Every series differs from its
neighbours in marker and line style as well as colour, so none of these
figures relies on colour alone.

Week 7 adds the randomized-algorithm figures. Each reads the plain lists of
dicts the Week 7 benchmark builds, and each is shaped by the kind of
randomness its algorithm has. A Las Vegas algorithm is always right, so its
figure is about time and work; a Monte Carlo algorithm can be wrong, so its
figure measures the error, with an interval on every rate.

* :func:`plot_quicksort_comparison` - randomized quicksort against fixed
  pivot rules on random and on sorted input, the spread of its comparison
  count around 2(n + 1)H_n - 4n, and its run-to-run variation.
* :func:`plot_selection_runtime` - randomized select against median of
  medians, the partition depth against log2(n), and how often a run stays
  under 4n comparisons.
* :func:`plot_primality_accuracy` - how often Fermat and Miller-Rabin call
  a composite prime, against the 4^-k bound, and what each costs beside
  trial division.
* :func:`plot_bloom_fp_rate` - the Bloom filter's false-positive rate
  against load and against bits per item, beside the memory it saves over
  a Python set.
* :func:`plot_minhash_similarity` - MinHash error shrinking as k^(-1/2),
  and estimated against true Jaccard similarity.

A value the Week 7 benchmark did not produce arrives as None and is skipped,
never filled in, and missing data never raises: a panel with nothing to draw
says so in words. A measured rate of zero has no place on a log axis, so it
is drawn at the top of its 95% interval as a downward triangle and labelled
as a bound rather than as a measurement.

Every function returns the :class:`matplotlib.figure.Figure` it built and
writes a 200 dpi PNG when given ``save_path``.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    "plot_knapsack_space_comparison",
    "plot_mcm_performance",
    "plot_floyd_warshall_scaling",
    "plot_tsp_runtime",
    "plot_mcm_table",
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


# ----------------------------------------------------------------------
# Week 6: advanced dynamic programming figures
# ----------------------------------------------------------------------
# These read the rows of benchmarks/results/comparison_table.csv, one dict
# per (problem, variant, n, secondary_param). They go through the same Week 5
# row readers above, so an empty cell means the same thing in both weeks:
# the benchmark did not produce it, and the figure does not draw it.
_W6_VARIANT_ORDER: Dict[str, Tuple[str, ...]] = {
    "knapsack": ("standard_2d", "space_optimized_1d"),
    "mcm": ("recursive", "memoized", "bottom_up"),
    "floyd_warshall": ("floyd_warshall", "floyd_warshall_3d", "all_pairs_dijkstra"),
    "tsp": ("bitmask", "brute_force"),
}

# The Week 5 rule again: within one figure no two series share a marker or a
# line style, so colour is never the only way to tell them apart. The slow
# baseline of each problem is the red circle, as it was in Week 5.
_W6_VARIANT_STYLE: Dict[str, Dict[str, Any]] = {
    "standard_2d": {"color": "#c1121f", "marker": "o", "linestyle": "-"},
    "space_optimized_1d": {"color": "#2a9d8f", "marker": "^", "linestyle": "-."},
    "recursive": {"color": "#c1121f", "marker": "o", "linestyle": "-"},
    "memoized": {"color": "#1d3557", "marker": "s", "linestyle": "--"},
    "bottom_up": {"color": "#2a9d8f", "marker": "^", "linestyle": "-."},
    "floyd_warshall": {"color": "#1d3557", "marker": "s", "linestyle": "-"},
    "floyd_warshall_3d": {"color": "#c1121f", "marker": "o", "linestyle": "--"},
    "all_pairs_dijkstra": {"color": "#2a9d8f", "marker": "^", "linestyle": "-."},
    "bitmask": {"color": "#2a9d8f", "marker": "^", "linestyle": "-"},
    "brute_force": {"color": "#c1121f", "marker": "o", "linestyle": "--"},
}
_W6_VARIANT_LABEL: Dict[str, str] = {
    "standard_2d": "standard 2D table (Week 5 knapsack_tab)",
    "space_optimized_1d": "space-optimized 1D row",
    "recursive": "plain recursion (no memo)",
    "memoized": "memoized (top-down)",
    "bottom_up": "bottom-up table",
    "floyd_warshall": "Floyd-Warshall (one n x n matrix)",
    "floyd_warshall_3d": "Floyd-Warshall 3D (every D(k) layer kept)",
    "all_pairs_dijkstra": "Dijkstra from every source",
    "bitmask": "Held-Karp bitmask DP",
    "brute_force": "brute force (every permutation)",
}
# A ratio is derived from two variants rather than being one, so it gets a
# style no variant uses: a purple diamond on a solid line.
_W6_RATIO_STYLE: Dict[str, Any] = {"color": "#6a4c93", "marker": "D", "linestyle": "-"}
_W6_REFERENCE_COLOUR = "0.35"
# The Floyd-Warshall comparison panel draws one series per (variant,
# density) pair. Indexing all three channels by the pair's position gives
# every series its own marker and its own line style, for any number of
# densities up to the length of the shortest tuple.
_W6_PAIR_COLOURS: Tuple[str, ...] = (
    "#1d3557", "#457b9d", "#2a9d8f", "#8ab17d", "#e76f51", "#f4a261",
)
_W6_PAIR_MARKERS: Tuple[str, ...] = ("s", "D", "^", "v", "o", "P", "X", "*")
_W6_PAIR_LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":")
_FW_COMPARED: Tuple[str, ...] = ("floyd_warshall", "all_pairs_dijkstra")
_FW_MEMORY: Tuple[str, ...] = ("floyd_warshall", "floyd_warshall_3d")
_W6_PROJECTED_LEGEND_TEXT = "projected, not timed (hollow marker, dashed segment)"


def _w6_variants(rows: Sequence[Dict[str, Any]], problem: str) -> List[str]:
    """List the variants present for one Week 6 problem, known ones first.

    The fixed order fixes the legend order, so each figure lists its series
    the slow baseline first. An unknown variant is still drawn, after the
    known ones, in the order it first appears.
    """
    preferred = _W6_VARIANT_ORDER.get(problem, ())
    seen: List[str] = []
    for row in rows:
        variant = _row_text(row, "variant")
        if variant and variant not in seen:
            seen.append(variant)
    known = [name for name in preferred if name in seen]
    return known + [name for name in seen if name not in preferred]


def _w6_style(variant: str, index: int) -> Dict[str, Any]:
    """Return the colour, marker and line style for one Week 6 variant."""
    style = _W6_VARIANT_STYLE.get(variant)
    return dict(style) if style is not None else _variant_style(variant, index)


def _metric_points(
    rows: Sequence[Dict[str, Any]], x_key: str, y_key: str
) -> List[Tuple[float, float, bool]]:
    """Turn rows into ``(x, y, is_projected)`` triples for any metric, by x.

    The Week 6 counterpart of :func:`_dp_points`, which only reads
    ``mean_time_s``; these figures also plot ``peak_kib``. A row missing
    either value contributes nothing, and a non-positive y is dropped too:
    every metric here is a time or a memory size, strictly positive
    whenever it was really produced, and most of them sit on log axes.

    Examples:
        >>> _metric_points([{"n": "8", "peak_kib": "2.5"},
        ...                 {"n": 4, "peak_kib": ""}], "n", "peak_kib")
        [(8.0, 2.5, False)]
    """
    points: List[Tuple[float, float, bool]] = []
    for row in rows:
        x_value = _row_number(row, x_key)
        y_value = _row_number(row, y_key)
        if x_value is None or y_value is None or y_value <= 0.0:
            continue
        points.append((x_value, y_value, _row_text(row, "measurement") == "projected"))
    points.sort(key=lambda point: point[0])
    return points


def _variant_points(
    rows: Sequence[Dict[str, Any]], variant: str, x_key: str, y_key: str
) -> List[Tuple[float, float, bool]]:
    """Return one variant's ``(x, y, is_projected)`` points, sorted by x."""
    subset = [row for row in rows if _row_text(row, "variant") == variant]
    return _metric_points(subset, x_key, y_key)


def _same_number(left: float, right: float) -> bool:
    """Compare two parameter values, tolerating float round-off.

    Densities make the trip to CSV and back as text, so ``0.1`` is matched
    with a tolerance rather than with ``==``.
    """
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _rows_near(
    rows: Sequence[Dict[str, Any]], key: str, value: float
) -> List[Dict[str, Any]]:
    """Select rows whose numeric ``key`` matches ``value`` up to round-off."""
    selected: List[Dict[str, Any]] = []
    for row in rows:
        number = _row_number(row, key)
        if number is not None and _same_number(number, value):
            selected.append(row)
    return selected


def _distinct_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    """List the distinct numeric values of ``key``, ascending, blanks skipped."""
    values: List[float] = []
    for row in rows:
        number = _row_number(row, key)
        if number is not None and not any(_same_number(number, v) for v in values):
            values.append(number)
    return sorted(values)


def _ratio_points(
    numerator: Sequence[Tuple[float, float, bool]],
    denominator: Sequence[Tuple[float, float, bool]],
) -> List[Tuple[float, float, bool]]:
    """Divide one series by another at the x values both were run at.

    A ratio is only as measured as its least measured term, so a point is
    marked projected when either input was. Sizes only one series reached
    are left out: a ratio with a missing term is not a ratio.

    Examples:
        >>> _ratio_points([(4.0, 6.0, False), (8.0, 9.0, True)],
        ...               [(4.0, 2.0, False), (8.0, 3.0, False),
        ...                (16.0, 5.0, False)])
        [(4.0, 3.0, False), (8.0, 3.0, True)]
    """
    lookup = {x: (y, projected) for x, y, projected in denominator}
    ratios: List[Tuple[float, float, bool]] = []
    for x, y, projected in numerator:
        if x in lookup:
            base, base_projected = lookup[x]
            ratios.append((x, y / base, projected or base_projected))
    ratios.sort(key=lambda point: point[0])
    return ratios


def _draw_w6_series(
    ax: Axes,
    rows: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    x_key: str,
    y_key: str,
) -> Tuple[int, bool]:
    """Draw each listed variant's ``y_key`` against ``x_key`` onto one axes.

    Returns:
        ``(series drawn, any projected point drawn)``.
    """
    drawn = 0
    any_projected = False
    for index, variant in enumerate(variants):
        points = _variant_points(rows, variant, x_key, y_key)
        if not points:
            continue
        label = _W6_VARIANT_LABEL.get(variant, variant)
        if _plot_variant_series(ax, points, _w6_style(variant, index), label):
            any_projected = True
        drawn += 1
    return drawn, any_projected


def _log_cubic(n: float) -> float:
    """Natural log of n^3, the Floyd-Warshall growth shape."""
    return 3.0 * math.log(n)


def _log_held_karp(n: float) -> float:
    """Natural log of n^2 2^n, the Held-Karp growth shape."""
    return 2.0 * math.log(n) + n * math.log(2.0)


def _log_factorial(n: float) -> float:
    """Natural log of n!, through the gamma function so it never overflows."""
    return math.lgamma(n + 1.0)


def _draw_reference_shape(
    ax: Axes,
    points: Sequence[Tuple[float, float, bool]],
    log_shape: Callable[[float], float],
    label: str,
    linestyle: Any,
) -> bool:
    """Draw a growth shape scaled to pass through a series' first point.

    Only the shape is claimed, not the constant. Anchoring at the first
    measured point removes the machine-dependent constant, so the gap that
    opens up between the curve and the data further right is the evidence:
    a series that bends away from its reference is not growing the way the
    reference says. The shape is worked in logs, so ``20!`` costs nothing
    to draw. The curve spans the series' own n range and no further, which
    keeps a factorial from stretching the runtime axis to geological time.

    Returns:
        True if a curve was drawn. Fewer than two distinct sizes, or a
        first size that is not positive, draws nothing.
    """
    anchors = [(x, y) for x, y, projected in points if not projected]
    if not anchors:
        anchors = [(x, y) for x, y, _ in points]
    if not anchors:
        return False
    x_start, y_start = anchors[0]
    x_end = points[-1][0]
    if x_start <= 0.0 or x_end <= x_start:
        return False
    xs = np.linspace(x_start, x_end, 120)
    base = log_shape(x_start)
    ys = [y_start * math.exp(log_shape(float(x)) - base) for x in xs]
    ax.plot(
        xs,
        ys,
        color=_W6_REFERENCE_COLOUR,
        linestyle=linestyle,
        linewidth=1.3,
        marker="",
        label=label,
        zorder=1,
    )
    return True


def _draw_speedup(
    ax: Axes, points: Sequence[Tuple[float, float, bool]], label: str
) -> Tuple[int, bool]:
    """Draw a speedup series with the break-even line at 1.

    Returns:
        ``(series drawn, any projected point drawn)``; ``(0, False)`` and
        nothing on the axes when there are no shared sizes.
    """
    if not points:
        return 0, False
    projected = _plot_variant_series(ax, points, dict(_W6_RATIO_STYLE), label)
    ax.axhline(
        1.0,
        color=_W6_REFERENCE_COLOUR,
        linestyle=":",
        linewidth=1.3,
        label="speedup = 1 (equal runtime)",
        zorder=1,
    )
    return 1, projected


def _add_w6_projected_entry(ax: Axes) -> None:
    """Add one neutral legend entry explaining the hollow projected markers."""
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
        label=_W6_PROJECTED_LEGEND_TEXT,
    )


def _finish_w6_panel(
    ax: Axes,
    drawn: int,
    any_projected: bool,
    labels: Tuple[str, str, str],
    empty_note: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    x_ticks: Sequence[float] = (),
) -> None:
    """Label one Week 6 panel, or say plainly why it is empty.

    A panel with no data gets a sentence in the middle of the frame naming
    what was missing, never a blank grid that could be read as "measured,
    and zero". Scales are only switched to log when something was drawn,
    since an empty log axis has no sensible limits.

    Args:
        ax: The panel.
        drawn: How many series were drawn on it.
        any_projected: Whether any projected point was drawn.
        labels: ``(x label, y label, title)``.
        empty_note: The sentence shown when ``drawn`` is 0.
        log_x: Logarithmic x axis.
        log_y: Logarithmic y axis.
        x_ticks: Measured x values to tick explicitly on a log x axis, where
            the default decade ticks would label none of them.
    """
    xlabel, ylabel, title = labels
    if drawn:
        if any_projected:
            _add_w6_projected_entry(ax)
        if log_x:
            ax.set_xscale("log")
            ticks = sorted(set(x_ticks))
            if 0 < len(ticks) <= 10:
                ax.set_xticks(ticks)
                ax.set_xticklabels([_format_measure(tick) for tick in ticks])
                ax.set_xticks([], minor=True)
        if log_y:
            ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            empty_note,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
            wrap=True,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)


def _table_entry(table: Sequence[Sequence[Any]], i: int, j: int) -> Optional[float]:
    """Read ``table[i][j]`` as a finite float, or ``None`` if it is not there.

    The CLRS tables leave row 0, column 0 and everything below the diagonal
    unused, and an implementation may fill those with 0, ``None`` or
    nothing at all. Reading through here means an unused cell is skipped
    rather than drawn as a cost of zero.
    """
    try:
        value = table[i][j]
    except (IndexError, KeyError, TypeError):
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def plot_knapsack_space_comparison(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
    *,
    fixed_n: Optional[float] = None,
) -> Figure:
    """Plot peak memory, runtime and memory ratio of 2D against 1D knapsack.

    The rolling row is a claim about space and only about space: both
    versions fill the same n x (W + 1) cells in the same order, so their
    runtimes should track each other while their memory should not. The
    figure therefore holds the item count fixed and sweeps capacity W,
    with three panels that each check one part of the claim:

    1. Peak memory against W, on a log axis. The 2D table holds
       ``(n + 1) x (W + 1)`` cells and the 1D row ``W + 1``, so the two
       lines should run parallel on the log axis, a constant factor apart.
    2. Mean runtime against W, on a linear axis, where O(n x W) at fixed n
       is a straight line. Two lines lying close together is the point:
       saving the memory did not cost time.
    3. The memory ratio, 2D peak divided by 1D peak, at every W both were
       traced at, beside the cell-count ratio ``n + 1``. The measured ratio
       is read against that line rather than expected to sit on it, since
       tracemalloc also counts list headers, int objects and the inputs,
       none of which the cell count includes.

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/comparison_table.csv``. Rows for other
            problems are ignored. Any numeric field may be an empty string
            (``peak_kib`` is blank for runs too long to trace), and such a
            value is skipped rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.
        fixed_n: The item count to hold fixed, matched against ``n``. When
            omitted, the item count the most knapsack rows share is used,
            since that is the sweep the benchmark treated as its control.

    Returns:
        The three-panel :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If there are no knapsack rows, if none carries an item
            count, if ``fixed_n`` matches no row, or if no row at that item
            count carries a usable ``peak_kib`` or ``mean_time_s``.

    Time Complexity:
        O(R log R) in the number of knapsack rows R: a constant number of
        passes to tally and select, then a sort per variant per panel.

    Space Complexity:
        O(R) for the selected rows and extracted points, plus the figure.

    Examples:
        >>> import tempfile
        >>> def row(variant, capacity, seconds, kib):
        ...     return {"problem": "knapsack", "variant": variant, "n": 50,
        ...             "secondary_param": capacity, "mean_time_s": seconds,
        ...             "std_time_s": "", "min_time_s": "", "max_time_s": "",
        ...             "runs": 5, "peak_kib": kib, "theoretical_time": "O(nW)",
        ...             "theoretical_space": "", "speedup_vs_baseline": "",
        ...             "baseline_variant": "", "measurement": "measured"}
        >>> rows = [row("standard_2d", 100, 0.0021, 45.2),
        ...         row("standard_2d", 400, 0.0083, 170.4),
        ...         row("space_optimized_1d", 100, 0.0019, 1.4),
        ...         row("space_optimized_1d", 400, 0.0074, 3.9),
        ...         row("space_optimized_1d", 800, 0.0150, "")]
        >>> png = os.path.join(tempfile.mkdtemp(), "knapsack_space.png")
        >>> fig = plot_knapsack_space_comparison(rows, png)
        >>> [axis.get_yscale() for axis in fig.axes]
        ['log', 'linear', 'linear']
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        An item count that was never run is a caller error, not a blank
        figure:

        >>> plot_knapsack_space_comparison(rows, fixed_n=7)
        Traceback (most recent call last):
            ...
        ValueError: no knapsack rows at item count n = 7
    """
    apply_house_style()
    knapsack_rows = _dp_rows(rows, "knapsack")
    if not knapsack_rows:
        raise ValueError("no knapsack rows to plot")
    item_count = (
        _dominant_value(knapsack_rows, "n") if fixed_n is None else float(fixed_n)
    )
    if item_count is None:
        raise ValueError("no knapsack rows carry an item count in n")
    selected = _rows_at(knapsack_rows, "n", item_count)
    held = f"n = {_format_measure(item_count)}"
    if not selected:
        raise ValueError(f"no knapsack rows at item count {held}")
    variants = _w6_variants(selected, "knapsack")

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))
    memory_drawn, memory_projected = _draw_w6_series(
        axes[0], selected, variants, "secondary_param", "peak_kib"
    )
    time_drawn, time_projected = _draw_w6_series(
        axes[1], selected, variants, "secondary_param", "mean_time_s"
    )
    if memory_drawn == 0 and time_drawn == 0:
        plt.close(fig)
        raise ValueError(
            f"no usable peak_kib or mean_time_s among the knapsack rows at {held}"
        )

    ratios = _ratio_points(
        _variant_points(selected, "standard_2d", "secondary_param", "peak_kib"),
        _variant_points(selected, "space_optimized_1d", "secondary_param", "peak_kib"),
    )
    ratio_projected = False
    if ratios:
        ratio_projected = _plot_variant_series(
            axes[2], ratios, dict(_W6_RATIO_STYLE), "measured peak ratio, 2D / 1D"
        )
        axes[2].axhline(
            item_count + 1.0,
            color=_W6_REFERENCE_COLOUR,
            linestyle=":",
            linewidth=1.3,
            label=f"cell-count ratio n + 1 = {_format_measure(item_count + 1.0)}",
            zorder=1,
        )

    xlabel = "Knapsack capacity W (weight units)"
    _finish_w6_panel(
        axes[0],
        memory_drawn,
        memory_projected,
        (xlabel, "Peak memory (KiB, log scale)", f"Peak memory against W at {held}"),
        "no peak_kib was produced for these rows",
        log_y=True,
    )
    _finish_w6_panel(
        axes[1],
        time_drawn,
        time_projected,
        (xlabel, "Mean runtime (seconds)", f"Runtime against W at {held}"),
        "no mean_time_s was produced for these rows",
    )
    _finish_w6_panel(
        axes[2],
        len(ratios),
        ratio_projected,
        (xlabel, "Peak-memory ratio, 2D / 1D (times)", "Memory saved by one row"),
        "standard_2d and space_optimized_1d share no W with a peak_kib",
    )
    fig.suptitle(
        "0/1 knapsack: one row of W + 1 cells in place of the "
        "(n + 1) x (W + 1) table",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save(fig, save_path)
    return fig


def plot_mcm_performance(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot matrix-chain runtime for all three variants, with a speedup panel.

    Plain recursion re-solves every subchain each time a split asks for it
    and is exponential in the chain length (CLRS shows Omega(2^n)), while
    memoization and the bottom-up table each solve the O(n^2) subchains
    once at O(n) per subchain, O(n^3) in all. The left panel puts all
    three on one log runtime axis, where the recursion is a climbing
    straight line and the two DP versions bend gently. The right panel
    divides recursion by bottom-up at every n both were run at, which is
    the factor the table saves, also on a log axis because that factor
    itself grows exponentially.

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/comparison_table.csv``. Rows for other
            problems are ignored. Any numeric field may be an empty string,
            and such a row is skipped rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The two-panel :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If no row carries ``problem == "mcm"`` together with a
            usable ``n`` and ``mean_time_s``.

    Time Complexity:
        O(R log R) in the number of matrix-chain rows R: each row is read a
        constant number of times and each series is sorted by n.

    Space Complexity:
        O(R) for the extracted points, plus the figure itself.

    Examples:
        >>> import tempfile
        >>> def row(variant, n, seconds, measurement="measured"):
        ...     return {"problem": "mcm", "variant": variant, "n": str(n),
        ...             "secondary_param": "", "mean_time_s": seconds,
        ...             "std_time_s": "", "min_time_s": "", "max_time_s": "",
        ...             "runs": "5", "peak_kib": "", "theoretical_time": "",
        ...             "theoretical_space": "", "speedup_vs_baseline": "",
        ...             "baseline_variant": "", "measurement": measurement}
        >>> rows = [row("recursive", 6, "0.0009"), row("recursive", 10, "0.07"),
        ...         row("recursive", 14, "5.2", "projected"),
        ...         row("memoized", 6, "0.0001"), row("memoized", 10, "0.0004"),
        ...         row("bottom_up", 6, "0.00005"), row("bottom_up", 10, "0.0002"),
        ...         row("bottom_up", 14, "0.0005"), row("bottom_up", 18, "")]
        >>> png = os.path.join(tempfile.mkdtemp(), "mcm_performance.png")
        >>> fig = plot_mcm_performance(rows, png)
        >>> [axis.get_yscale() for axis in fig.axes]
        ['log', 'log']
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        Nothing to draw is a caller error, not an empty picture:

        >>> plot_mcm_performance([])
        Traceback (most recent call last):
            ...
        ValueError: no mcm rows with a usable n and mean_time_s
    """
    apply_house_style()
    mcm_rows = _dp_rows(rows, "mcm")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    drawn, any_projected = _draw_w6_series(
        axes[0], mcm_rows, _w6_variants(mcm_rows, "mcm"), "n", "mean_time_s"
    )
    if drawn == 0:
        plt.close(fig)
        raise ValueError("no mcm rows with a usable n and mean_time_s")

    speedups = _ratio_points(
        _variant_points(mcm_rows, "recursive", "n", "mean_time_s"),
        _variant_points(mcm_rows, "bottom_up", "n", "mean_time_s"),
    )
    speed_drawn, speed_projected = _draw_speedup(
        axes[1], speedups, "recursive runtime / bottom-up runtime"
    )

    xlabel = "Number of matrices n (matrices in the chain)"
    _finish_w6_panel(
        axes[0],
        drawn,
        any_projected,
        (xlabel, "Mean runtime (seconds, log scale)", "Runtime against chain length"),
        "",
        log_y=True,
    )
    _finish_w6_panel(
        axes[1],
        speed_drawn,
        speed_projected,
        (
            xlabel,
            "Speedup of bottom-up over recursion (times, log scale)",
            "How many times faster the table is",
        ),
        "recursive and bottom_up share no n with a mean_time_s",
        log_y=True,
    )
    fig.suptitle(
        "Matrix-chain multiplication: exponential recursion against the "
        "O(n^3) table",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save(fig, save_path)
    return fig


def plot_floyd_warshall_scaling(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
    *,
    scaling_density: float = 0.5,
) -> Figure:
    """Plot Floyd-Warshall scaling, its Dijkstra comparison and its memory.

    Three panels, one question each:

    1. Does Floyd-Warshall really cost n^3? Its runtime at one density is
       drawn on log-log axes beside an n^3 line anchored at the smallest
       n, so a series that runs parallel to the line is cubic, and a gap
       that opens up would be the evidence that it is not.
    2. When is it the right tool? Floyd-Warshall's triple loop never looks
       at an edge, so its cost does not move with density, while running
       Dijkstra from every source costs O(V (V + E) log V) and does. Each
       algorithm is drawn once per density present, every one of those
       series with its own marker and line style, so whether Dijkstra
       wins on sparse graphs and loses on dense ones can be read straight
       off the panel.
    3. What does dropping the k index save? The 3D version keeps all
       n + 1 layers D(0)..D(n), Theta(n^3) cells, while the in-place
       version keeps one n x n matrix. Peak memory for both, on a log
       axis, shows that factor of n + 1 as a gap that widens with n.

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/comparison_table.csv``. Rows for other
            problems are ignored. ``secondary_param`` is the edge density.
            Any numeric field may be an empty string, and such a value is
            skipped rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.
        scaling_density: The density whose ``floyd_warshall`` rows the
            first panel plots. The memory panel uses it too when any memory
            was traced there, and otherwise the density traced most often.

    Returns:
        The three-panel :class:`matplotlib.figure.Figure`. A panel with no
        rows to draw states what was missing instead of staying blank.

    Raises:
        ValueError: If there are no floyd_warshall rows, or if none of the
            three panels has anything usable to draw.

    Time Complexity:
        O(R log R + D * R) in the number of Floyd-Warshall rows R and the
        number of distinct densities D: each (variant, density) series is
        selected by one pass and sorted by n.

    Space Complexity:
        O(R) for the selected rows and extracted points, plus the figure.

    Examples:
        >>> import tempfile
        >>> def row(variant, n, density, seconds, kib):
        ...     return {"problem": "floyd_warshall", "variant": variant,
        ...             "n": n, "secondary_param": density,
        ...             "mean_time_s": seconds, "std_time_s": "",
        ...             "min_time_s": "", "max_time_s": "", "runs": 5,
        ...             "peak_kib": kib, "theoretical_time": "O(n^3)",
        ...             "theoretical_space": "", "speedup_vs_baseline": "",
        ...             "baseline_variant": "", "measurement": "measured"}
        >>> rows = [row("floyd_warshall", 20, 0.5, 0.002, 30.1),
        ...         row("floyd_warshall", 40, 0.5, 0.016, 110.5),
        ...         row("floyd_warshall", 20, 0.1, 0.002, 29.8),
        ...         row("floyd_warshall", 40, 0.1, 0.015, ""),
        ...         row("all_pairs_dijkstra", 20, 0.5, 0.004, 12.0),
        ...         row("all_pairs_dijkstra", 40, 0.5, 0.020, 40.2),
        ...         row("all_pairs_dijkstra", 20, 0.1, 0.001, 8.0),
        ...         row("all_pairs_dijkstra", 40, 0.1, 0.004, 22.1),
        ...         row("floyd_warshall_3d", 20, 0.5, 0.003, 600.0),
        ...         row("floyd_warshall_3d", 40, 0.5, 0.025, 4600.0)]
        >>> png = os.path.join(tempfile.mkdtemp(), "floyd_warshall.png")
        >>> fig = plot_floyd_warshall_scaling(rows, png, scaling_density=0.5)
        >>> len(fig.axes[1].get_legend().get_texts())
        4
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        A density nobody ran leaves the first panel saying so, not blank:

        >>> fig = plot_floyd_warshall_scaling(rows, scaling_density=0.9)
        >>> fig.axes[0].texts[0].get_text()
        'no floyd_warshall rows at density 0.9'
        >>> plt.close(fig)
    """
    apply_house_style()
    fw_rows = _dp_rows(rows, "floyd_warshall")
    if not fw_rows:
        raise ValueError("no floyd_warshall rows to plot")
    density = float(scaling_density)
    density_text = _format_measure(density)
    xlabel = "Number of vertices n (vertices)"
    runtime_label = "Mean runtime (seconds, log scale)"

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.6))

    # Panel 1: one variant at one density, against the n^3 shape.
    scaling = _variant_points(
        _rows_near(fw_rows, "secondary_param", density),
        "floyd_warshall",
        "n",
        "mean_time_s",
    )
    scaling_projected = False
    if scaling:
        scaling_projected = _plot_variant_series(
            axes[0],
            scaling,
            _w6_style("floyd_warshall", 0),
            _W6_VARIANT_LABEL["floyd_warshall"],
        )
        _draw_reference_shape(
            axes[0],
            scaling,
            _log_cubic,
            r"$n^3$ reference, anchored at the smallest n",
            ":",
        )

    # Panel 2: each compared variant at each density, one style per pair.
    compare_rows = [row for row in fw_rows if _row_text(row, "variant") in _FW_COMPARED]
    densities = _distinct_values(compare_rows, "secondary_param")
    compare_drawn = 0
    compare_projected = False
    compare_x: List[float] = []
    for variant_index, variant in enumerate(_FW_COMPARED):
        for density_index, value in enumerate(densities):
            pair = variant_index * len(densities) + density_index
            points = _variant_points(
                _rows_near(compare_rows, "secondary_param", value),
                variant,
                "n",
                "mean_time_s",
            )
            if not points:
                continue
            style = {
                "color": _W6_PAIR_COLOURS[pair % len(_W6_PAIR_COLOURS)],
                "marker": _W6_PAIR_MARKERS[pair % len(_W6_PAIR_MARKERS)],
                "linestyle": _W6_PAIR_LINESTYLES[pair % len(_W6_PAIR_LINESTYLES)],
            }
            label = f"{_W6_VARIANT_LABEL[variant]}, density {_format_measure(value)}"
            if _plot_variant_series(axes[1], points, style, label):
                compare_projected = True
            compare_drawn += 1
            compare_x.extend(point[0] for point in points)

    # Panel 3: one matrix against every layer, at a single density.
    memory_rows = [
        row
        for row in fw_rows
        if _row_text(row, "variant") in _FW_MEMORY
        and _row_number(row, "peak_kib") is not None
    ]
    memory_density = (
        density
        if _rows_near(memory_rows, "secondary_param", density)
        else _dominant_value(memory_rows, "secondary_param")
    )
    if memory_density is not None:
        memory_rows = _rows_near(memory_rows, "secondary_param", memory_density)
    memory_drawn, memory_projected = _draw_w6_series(
        axes[2], memory_rows, _FW_MEMORY, "n", "peak_kib"
    )

    if not scaling and compare_drawn == 0 and memory_drawn == 0:
        plt.close(fig)
        raise ValueError("no usable floyd_warshall measurements to plot")

    _finish_w6_panel(
        axes[0],
        len(scaling),
        scaling_projected,
        (
            xlabel,
            runtime_label,
            f"Floyd-Warshall against $n^3$ (density {density_text}, log-log)",
        ),
        f"no floyd_warshall rows at density {density_text}",
        log_x=True,
        log_y=True,
        x_ticks=[point[0] for point in scaling],
    )
    _finish_w6_panel(
        axes[1],
        compare_drawn,
        compare_projected,
        (xlabel, runtime_label, "Floyd-Warshall against Dijkstra from every source"),
        "no floyd_warshall or all_pairs_dijkstra rows carry a density",
        log_x=True,
        log_y=True,
        x_ticks=compare_x,
    )
    memory_title = "Peak memory: one matrix against every layer"
    if memory_density is not None:
        memory_title += f" (density {_format_measure(memory_density)})"
    _finish_w6_panel(
        axes[2],
        memory_drawn,
        memory_projected,
        (xlabel, "Peak memory (KiB, log scale)", memory_title),
        "no peak_kib was produced for floyd_warshall or floyd_warshall_3d",
        log_y=True,
    )
    fig.suptitle(
        "All-pairs shortest paths: Floyd-Warshall's cubic time, its Dijkstra "
        "alternative, and the layer it does not keep",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    _save(fig, save_path)
    return fig


def plot_tsp_runtime(
    rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot Held-Karp against brute-force TSP, with growth shapes and crossover.

    Both algorithms are exponential, and the figure exists to show that
    "exponential" covers very different things. Brute force with the start
    fixed tries (n - 1)! tours; Held-Karp fills an n x 2^n table at O(n)
    per cell, O(n^2 2^n) in all. The left panel puts both on a log runtime
    axis beside those two shapes, each scaled to pass through the first
    measured point of its own series, so the comparison is between growth
    rates and not between machine constants.

    For small n brute force can win, because its inner loop is a bare
    permutation walk while Held-Karp pays for a 2^n table up front. The
    right panel divides brute force by bitmask at every n both were run
    at; above the line at 1 the bitmask DP is faster. The first measured n
    where that happens is marked with a vertical line and annotated, since
    that size is where the extra memory starts paying for itself.

    Args:
        rows: Benchmark measurement rows, as written to
            ``benchmarks/results/comparison_table.csv``. Rows for other
            problems are ignored. Any numeric field may be an empty string,
            and such a row is skipped rather than guessed at.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The two-panel :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If no row carries ``problem == "tsp"`` together with a
            usable ``n`` and ``mean_time_s``.

    Time Complexity:
        O(R log R) in the number of TSP rows R, plus a constant 120 points
        per reference curve.

    Space Complexity:
        O(R) for the extracted points, plus the figure itself.

    Examples:
        >>> import tempfile
        >>> def row(variant, n, seconds):
        ...     return {"problem": "tsp", "variant": variant, "n": n,
        ...             "secondary_param": "", "mean_time_s": seconds,
        ...             "std_time_s": "", "min_time_s": "", "max_time_s": "",
        ...             "runs": 5, "peak_kib": "", "theoretical_time": "",
        ...             "theoretical_space": "", "speedup_vs_baseline": "",
        ...             "baseline_variant": "", "measurement": "measured"}
        >>> rows = [row("bitmask", 4, 4e-05), row("bitmask", 6, 0.0003),
        ...         row("bitmask", 8, 0.002), row("bitmask", 10, 0.012),
        ...         row("brute_force", 4, 1e-05), row("brute_force", 6, 0.0002),
        ...         row("brute_force", 8, 0.012), row("brute_force", 10, 1.1),
        ...         row("brute_force", 12, "")]
        >>> png = os.path.join(tempfile.mkdtemp(), "tsp_runtime.png")
        >>> fig = plot_tsp_runtime(rows, png)
        >>> os.path.getsize(png) > 0
        True

        Brute force is still ahead at n = 6 and behind at n = 8, so the
        crossover is marked at 8:

        >>> any("n = 8" in text.get_text()
        ...     for text in fig.axes[1].get_legend().get_texts())
        True
        >>> plt.close(fig)
    """
    apply_house_style()
    tsp_rows = _dp_rows(rows, "tsp")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    drawn, any_projected = _draw_w6_series(
        axes[0], tsp_rows, _w6_variants(tsp_rows, "tsp"), "n", "mean_time_s"
    )
    if drawn == 0:
        plt.close(fig)
        raise ValueError("no tsp rows with a usable n and mean_time_s")

    bitmask = _variant_points(tsp_rows, "bitmask", "n", "mean_time_s")
    brute = _variant_points(tsp_rows, "brute_force", "n", "mean_time_s")
    _draw_reference_shape(
        axes[0],
        bitmask,
        _log_held_karp,
        r"$n^2 2^n$ shape, scaled to first bitmask point",
        ":",
    )
    _draw_reference_shape(
        axes[0],
        brute,
        _log_factorial,
        r"$n!$ shape, scaled to first brute-force point",
        "-.",
    )

    speedups = _ratio_points(brute, bitmask)
    speed_drawn, speed_projected = _draw_speedup(
        axes[1], speedups, "brute-force runtime / bitmask runtime"
    )
    speed_title = "Speedup of the bitmask DP over brute force"
    crossover = next((point for point in speedups if point[1] > 1.0), None)
    if crossover is not None:
        cross_n, cross_speedup, cross_projected = crossover
        cross_text = _format_measure(cross_n)
        axes[1].axvline(
            cross_n,
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            label=f"first n where bitmask is faster: n = {cross_text}",
            zorder=1,
        )
        axes[1].plot(
            [cross_n],
            [cross_speedup],
            marker="*",
            markersize=16,
            color="#e9c46a",
            markeredgecolor="black",
            linestyle="",
            zorder=5,
        )
        axes[1].annotate(
            f"crossover at n = {cross_text}\n"
            f"bitmask {cross_speedup:.3g}x faster"
            + (" (projected)" if cross_projected else ""),
            xy=(cross_n, cross_speedup),
            xytext=(16, -30),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "0.25"},
        )
        speed_title += f" (crossover at n = {cross_text})"
    elif speedups:
        speed_title += " (bitmask faster at no shared n)"

    xlabel = "Number of cities n (cities)"
    _finish_w6_panel(
        axes[0],
        drawn,
        any_projected,
        (xlabel, "Mean runtime (seconds, log scale)", "Runtime against city count"),
        "",
        log_y=True,
    )
    _finish_w6_panel(
        axes[1],
        speed_drawn,
        speed_projected,
        (xlabel, "Speedup of bitmask over brute force (times, log scale)", speed_title),
        "bitmask and brute_force share no n with a mean_time_s",
        log_y=True,
    )
    fig.suptitle(
        r"Travelling salesman: Held-Karp's $O(n^2 2^n)$ against brute force's "
        r"$O(n!)$",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    _save(fig, save_path)
    return fig


def plot_mcm_table(
    m: Sequence[Sequence[Any]],
    s: Sequence[Sequence[Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Draw the CLRS matrix-chain m table as an annotated heatmap.

    The bottom-up algorithm fills this table diagonal by diagonal: chains
    of length 1 on the main diagonal, then length 2, up to the single cell
    ``m[1][n]`` in the top right corner, which is the answer. Drawing the
    table makes that band-by-band order visible, which a printed list of
    numbers does not. Every cell carries its cost and, off the diagonal,
    the split ``k = s[i][j]`` that achieved it; following those splits
    down from the corner is exactly what the parenthesization printer does.
    Cost is not monotone in chain length (a longer chain can be cheaper
    than one of its subchains when its outer dimensions are small), so the
    colour scale is read cell by cell, not as a gradient toward the corner.

    Only ``1 <= i <= j <= n`` is defined. Row 0, column 0 and the lower
    triangle are masked out and left blank rather than drawn as zero cost.

    Args:
        m: The 1-indexed ``(n + 1) x (n + 1)`` cost table from
            ``mcm_bottom_up``, ``m[i][j]`` the minimum scalar
            multiplications for ``A_i..A_j``.
        s: The matching split table, ``s[i][j]`` the optimal k. A missing or
            ``None`` entry simply leaves that cell without a ``k=`` line.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The :class:`matplotlib.figure.Figure`.

    Raises:
        ValueError: If ``m`` has fewer than two rows (n would be 0), or if
            it holds no usable cost anywhere in its upper triangle.

    Time Complexity:
        O(n^2): one read, one colour lookup and one label per defined cell.

    Space Complexity:
        O(n^2) for the dense cost grid passed to the heatmap.

    Examples:
        >>> import tempfile
        >>> m = [[0, 0, 0, 0], [0, 0, 1000, 2500], [0, 0, 0, 3000], [0, 0, 0, 0]]
        >>> s = [[0, 0, 0, 0], [0, 0, 1, 2], [0, 0, 0, 2], [0, 0, 0, 0]]
        >>> png = os.path.join(tempfile.mkdtemp(), "mcm_table.png")
        >>> fig = plot_mcm_table(m, s, png)
        >>> [text.get_text().replace(chr(10), " ") for text in fig.axes[0].texts]
        ['0', '1,000 k=1', '2,500 k=2', '0', '3,000 k=2', '0']
        >>> os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        A table with no matrices in it is a caller error:

        >>> plot_mcm_table([[0]], [[0]])
        Traceback (most recent call last):
            ...
        ValueError: m must be the 1-indexed (n + 1) x (n + 1) CLRS table with n >= 1
    """
    apply_house_style()
    if len(m) < 2:
        raise ValueError(
            "m must be the 1-indexed (n + 1) x (n + 1) CLRS table with n >= 1"
        )
    n = len(m) - 1
    costs = np.full((n, n), np.nan)
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            value = _table_entry(m, i, j)
            if value is not None:
                costs[i - 1, j - 1] = value
    if np.all(np.isnan(costs)):
        raise ValueError("m holds no usable cost for any 1 <= i <= j <= n")

    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="white")
    side = max(4.8, 0.95 * n + 2.4)
    fig, ax = plt.subplots(figsize=(side + 1.4, side))
    image = ax.imshow(np.ma.masked_invalid(costs), cmap=cmap, aspect="equal")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Minimum cost m[i][j] (scalar multiplications)")

    font_size = 9 if n <= 8 else 7 if n <= 12 else 5
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            value = costs[i - 1, j - 1]
            if np.isnan(value):
                continue
            split = _table_entry(s, i, j) if j > i else None
            text = _format_measure(value)
            if split is not None:
                text += f"\nk={_format_measure(split)}"
            ax.text(
                j - 1,
                i - 1,
                text,
                ha="center",
                va="center",
                fontsize=font_size,
                color=_label_text_colour(cmap(image.norm(value))),
            )

    positions = list(range(n))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(index + 1) for index in positions])
    ax.set_yticks(positions)
    ax.set_yticklabels([str(index + 1) for index in positions])
    ax.set_xlabel("Chain end j (matrix index, last matrix $A_j$)")
    ax.set_ylabel("Chain start i (matrix index, first matrix $A_i$)")
    ax.set_title(
        f"CLRS m table for {n} matrices: cost m[i][j] and split k = s[i][j]\n"
        "(the answer m[1][n] is the top-right cell; the lower triangle is unused)"
    )
    ax.grid(False)
    fig.tight_layout()

    _save(fig, save_path)
    return fig


# --- Week 7: randomized algorithms ---
# The Week 7 benchmark hands these five functions plain lists of dicts, one
# per measured point, with the keys each docstring lists. A value it did not
# produce arrives as None (or, after a CSV round trip, as an empty string).
# Every read goes through the Week 5 row reader, so a missing value is
# skipped and never filled in, and missing data never raises: a panel with
# nothing to draw says so in words instead of showing an empty grid.
#
# Two conventions are new this week. Every mean or rate carries its spread:
# +/- 1 std for a time, the benchmark's 95% interval for a rate. And a
# measured rate of zero, which has no place on a log axis, is drawn at the
# top of its interval as a downward triangle, so it reads as "at most this
# much" and never as a measured value.
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

__all__ += [
    "plot_quicksort_comparison",
    "plot_selection_runtime",
    "plot_primality_accuracy",
    "plot_bloom_fp_rate",
    "plot_minhash_similarity",
]

_W7_QUICKSORT_ORDER: Tuple[str, ...] = (
    "randomized",
    "deterministic_last",
    "deterministic_median_of_three",
    "week2_quick_sort",
)
_W7_SELECTION_ORDER: Tuple[str, ...] = ("randomized", "deterministic")
_W7_K_LABEL_ORDER: Tuple[str, ...] = ("min", "median", "max")
_W7_PRIMALITY_TESTS: Tuple[str, ...] = ("fermat", "miller_rabin")
_W7_SPEED_ORDER: Tuple[str, ...] = ("trial_division", "fermat", "miller_rabin")

# One style per named series. As in Weeks 5 and 6, no two series in a panel
# share both a marker and a line style, and the slow or fixed baseline is
# the red circle.
_W7_STYLE: Dict[str, Dict[str, Any]] = {
    "randomized": {"color": "#2a9d8f", "marker": "^", "linestyle": "-"},
    "deterministic_last": {"color": "#c1121f", "marker": "o", "linestyle": "--"},
    "deterministic_median_of_three": {
        "color": "#1d3557",
        "marker": "s",
        "linestyle": "-.",
    },
    "week2_quick_sort": {"color": "#e76f51", "marker": "D", "linestyle": ":"},
    "deterministic": {"color": "#c1121f", "marker": "o", "linestyle": "--"},
    "trial_division": {"color": "#c1121f", "marker": "o", "linestyle": "--"},
    "fermat": {"color": "#e76f51", "marker": "s", "linestyle": "-."},
    "miller_rabin": {"color": "#2a9d8f", "marker": "^", "linestyle": "-"},
}
_W7_QUICKSORT_LABEL: Dict[str, str] = {
    "randomized": "randomized pivot (Las Vegas)",
    "deterministic_last": "last-element pivot (CLRS)",
    "deterministic_median_of_three": "median-of-three pivot",
    "week2_quick_sort": "Week 2 quick_sort",
}
_W7_SELECTION_LABEL: Dict[str, str] = {
    "randomized": "randomized select",
    "deterministic": "median of medians",
}
_W7_PRIMALITY_LABEL: Dict[str, str] = {
    "trial_division": "trial division (exact)",
    "fermat": "Fermat",
    "miller_rabin": "Miller-Rabin",
}
# In the selection figure the order statistic gets the marker, and in the
# depth panel it gets all three channels.
_W7_K_STYLE: Dict[str, Dict[str, Any]] = {
    "min": {"color": "#1d3557", "marker": "v", "linestyle": "-."},
    "median": {"color": "#2a9d8f", "marker": "o", "linestyle": "-"},
    "max": {"color": "#e76f51", "marker": "^", "linestyle": ":"},
}
# Groups that have no name of their own: a (test, n) pair, a k, a true J.
# Six markers against four line styles keeps every pair distinct for twelve
# groups. No "v" here, because the downward triangle means a zero-count
# bound; no plain "--", because the theory lines use dashes.
_W7_GROUP_COLOURS: Tuple[str, ...] = (
    "#1d3557", "#c1121f", "#2a9d8f", "#e76f51", "#6a4c93", "#8ab17d",
)
_W7_GROUP_MARKERS: Tuple[str, ...] = ("o", "s", "D", "^", "P", "X")
_W7_GROUP_LINESTYLES: Tuple[Any, ...] = ("-", "-.", ":", (0, (3, 1, 1, 1, 1, 1)))
_W7_THEORY_DASHES: Tuple[Any, ...] = (
    (0, (7, 3)), (0, (4, 2)), (0, (7, 2, 2, 2)), (0, (10, 4)), (0, (5, 1)),
    (0, (3, 3)),
)
_W7_REFERENCE_COLOUR = "0.35"
_W7_ZERO_LABEL = "0 observed (95% upper bound)"
_W7_SELECT_THRESHOLD = 4.0
_W7_SELECT_EXPECTED: Dict[str, float] = {
    "median": 2.0 * (1.0 + math.log(2.0)),
    "min": 2.0,
    "max": 2.0,
}
_W7_MINHASH_SCATTER_K: Tuple[float, ...] = (16.0, 512.0)

# (x, y, length of the bar below y, length above y, is_projected)
_W7Point = Tuple[float, float, Optional[float], Optional[float], bool]


def _w7_rows(rows: Any) -> List[Dict[str, Any]]:
    """Return the dict rows of ``rows``, or an empty list when it is missing.

    Examples:
        >>> _w7_rows(None), _w7_rows([{"n": 1}, None])
        ([], [{'n': 1}])
    """
    if not rows or isinstance(rows, (str, bytes, dict)):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _w7_number(row: Dict[str, Any], key: str) -> Optional[float]:
    """Read a numeric field as a finite float, or ``None``, without raising.

    :func:`_row_number` does the reading. This adds the one case it lets
    through, an integer too large for a float, which a primality row could
    in principle carry.

    Examples:
        >>> _w7_number({"n": 561}, "n"), _w7_number({"n": None}, "n")
        (561.0, None)
        >>> _w7_number({"n": 10 ** 400}, "n") is None
        True
    """
    try:
        return _row_number(row, key)
    except (OverflowError, TypeError, ValueError):
        return None


def _w7_value(value: Any) -> Optional[float]:
    """Read one loose value, such as a list element, as :func:`_w7_number` would."""
    return _w7_number({"value": value}, "value")


def _w7_values(raw: Any) -> List[float]:
    """Read a list of loose numbers, dropping anything missing or unusable.

    Examples:
        >>> _w7_values([3, None, "", 4.5]), _w7_values(None)
        ([3.0, 4.5], [])
    """
    if raw is None or isinstance(raw, (str, bytes, dict)):
        return []
    try:
        items = list(raw)
    except TypeError:
        return []
    return [number for number in map(_w7_value, items) if number is not None]


def _w7_names(
    rows: Sequence[Dict[str, Any]], key: str, order: Sequence[str]
) -> List[str]:
    """List the distinct values of a text column, known ones first in ``order``."""
    seen: List[str] = []
    for row in rows:
        name = _row_text(row, key)
        if name and name not in seen:
            seen.append(name)
    return [name for name in order if name in seen] + [
        name for name in seen if name not in order
    ]


def _w7_distinct(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    """List the distinct numeric values of ``key``, ascending, blanks skipped."""
    values: List[float] = []
    for row in rows:
        number = _w7_number(row, key)
        if number is not None and not any(_same_number(number, v) for v in values):
            values.append(number)
    return sorted(values)


def _w7_near(
    rows: Sequence[Dict[str, Any]], key: str, value: float
) -> List[Dict[str, Any]]:
    """Select rows whose numeric ``key`` matches ``value`` up to round-off."""
    selected: List[Dict[str, Any]] = []
    for row in rows:
        number = _w7_number(row, key)
        if number is not None and _same_number(number, value):
            selected.append(row)
    return selected


def _w7_style(name: str, index: int) -> Dict[str, Any]:
    """Return the colour, marker and line style for one named series."""
    style = _W7_STYLE.get(name)
    return dict(style) if style is not None else _variant_style(name, index)


def _w7_k_style(k_label: str, index: int) -> Dict[str, Any]:
    """Return the style for one order statistic: min, median or max."""
    style = _W7_K_STYLE.get(k_label)
    return dict(style) if style is not None else _w7_group_style(index + 3)


def _w7_group_style(index: int) -> Dict[str, Any]:
    """Return the style for the ``index``-th unnamed group in a panel.

    ``theory`` is the dash pattern for the group's theoretical line, in the
    group's own colour, so each theory line matches its measured series.
    """
    return {
        "color": _W7_GROUP_COLOURS[index % len(_W7_GROUP_COLOURS)],
        "marker": _W7_GROUP_MARKERS[index % len(_W7_GROUP_MARKERS)],
        "linestyle": _W7_GROUP_LINESTYLES[index % len(_W7_GROUP_LINESTYLES)],
        "theory": _W7_THEORY_DASHES[index % len(_W7_THEORY_DASHES)],
    }


def _w7_bar(
    y: float, low_end: Optional[float], high_end: Optional[float], log_y: bool
) -> Tuple[Optional[float], Optional[float]]:
    """Turn the ends of an interval around ``y`` into bar lengths.

    A log axis has no zero, and a bar that reaches zero or below makes
    matplotlib drop every bar from its autoscaling. Such a lower end is cut
    at one tenth of ``y``: the bar still visibly runs off the bottom of the
    point's decade, and the panel keeps sensible limits.

    Examples:
        >>> _w7_bar(1.0, 0.8, 1.5, True)
        (0.19999999999999996, 0.5)
        >>> _w7_bar(1.0, -0.2, None, True)
        (0.9, None)
        >>> _w7_bar(1.0, -0.2, None, False)
        (1.2, None)
    """
    below = None if low_end is None else max(0.0, y - low_end)
    above = None if high_end is None else max(0.0, high_end - y)
    if log_y and low_end is not None and low_end <= 0.0:
        below = 0.9 * y
    return below, above


def _w7_points(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    std_key: Optional[str] = None,
    low_key: Optional[str] = None,
    high_key: Optional[str] = None,
    log_x: bool = True,
    log_y: bool = True,
) -> List[_W7Point]:
    """Turn rows into :data:`_W7Point` tuples, sorted by x.

    The spread comes from ``std_key`` (a symmetric +/- 1 std bar) or from
    ``low_key`` and ``high_key`` (the ends of an interval). A row missing x
    or y contributes nothing, and neither does a non-positive value on a
    log axis, since it has no position there.

    Examples:
        >>> _w7_points([{"n": 8, "t": 2.0, "s": 0.5}, {"n": 4, "t": None}],
        ...            "n", "t", std_key="s")
        [(8.0, 2.0, 0.5, 0.5, False)]
    """
    points: List[_W7Point] = []
    for row in rows:
        x_value = _w7_number(row, x_key)
        y_value = _w7_number(row, y_key)
        if x_value is None or y_value is None:
            continue
        if (log_x and x_value <= 0.0) or (log_y and y_value <= 0.0):
            continue
        low_end: Optional[float] = None
        high_end: Optional[float] = None
        if std_key is not None:
            spread = _w7_number(row, std_key)
            if spread is not None and spread >= 0.0:
                low_end, high_end = y_value - spread, y_value + spread
        else:
            low_end = _w7_number(row, low_key) if low_key else None
            high_end = _w7_number(row, high_key) if high_key else None
        below, above = _w7_bar(y_value, low_end, high_end, log_y)
        projected = _row_text(row, "measurement") == "projected"
        points.append((x_value, y_value, below, above, projected))
    points.sort(key=lambda point: point[0])
    return points


def _w7_line(
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    log_x: bool,
    log_y: bool,
) -> List[Tuple[float, float]]:
    """Return the ``(x, y)`` pairs of a theory column, sorted by x."""
    pairs: List[Tuple[float, float]] = []
    for row in rows:
        x_value = _w7_number(row, x_key)
        y_value = _w7_number(row, y_key)
        if x_value is None or y_value is None:
            continue
        if (log_x and x_value <= 0.0) or (log_y and y_value <= 0.0):
            continue
        pairs.append((x_value, y_value))
    pairs.sort()
    return pairs


def _w7_yerr(points: Sequence[_W7Point]) -> Optional[np.ndarray]:
    """Build errorbar's 2 x N ``yerr``, or ``None`` when no point has a bar.

    A point with only one end known gets a zero-length bar on the other
    side, so the known half is still drawn. A point with neither gets NaN,
    which matplotlib leaves undrawn.
    """
    below: List[float] = []
    above: List[float] = []
    for _, _, low, high, _ in points:
        if low is None and high is None:
            below.append(np.nan)
            above.append(np.nan)
        else:
            below.append(0.0 if low is None else low)
            above.append(0.0 if high is None else high)
    if all(np.isnan(value) for value in below):
        return None
    return np.array([below, above])


def _w7_errorbar(
    ax: Axes,
    points: Sequence[_W7Point],
    style: Dict[str, Any],
    label: Optional[str],
    *,
    connect: bool = True,
) -> Tuple[Any, bool]:
    """Draw one series with its error bars, projected points kept apart.

    Measured points are filled markers on the series' own line style (or on
    no line when ``connect`` is false). Projected points are hollow markers
    on a dashed segment continuing from the last measured point, the Week 5
    and Week 6 convention, so a projection never reads as a measurement.

    Returns:
        ``(handle, any projected)``: the artist that stands for the series
        in a legend, or ``None`` if nothing was drawn.
    """
    measured = [point for point in points if not point[4]]
    projected = [point for point in points if point[4]]
    common = {"color": style["color"], "marker": style["marker"], "capsize": 3}
    handle = None
    if measured:
        handle = ax.errorbar(
            [point[0] for point in measured],
            [point[1] for point in measured],
            yerr=_w7_yerr(measured),
            linestyle=style["linestyle"] if connect else "none",
            linewidth=1.8,
            elinewidth=1.0,
            markersize=6.5,
            label=label,
            zorder=3,
            **common,
        )
    if projected:
        if connect:
            bridge = measured[-1:] + projected
            ax.plot(
                [point[0] for point in bridge],
                [point[1] for point in bridge],
                color=style["color"],
                linestyle="--",
                linewidth=1.4,
                marker="",
                alpha=0.9,
                zorder=2,
            )
        hollow = ax.errorbar(
            [point[0] for point in projected],
            [point[1] for point in projected],
            yerr=_w7_yerr(projected),
            linestyle="none",
            elinewidth=1.0,
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=style["color"],
            markeredgewidth=1.8,
            label=f"{label} (projected)" if label and not measured else None,
            zorder=4,
            **common,
        )
        if handle is None:
            handle = hollow
    return handle, bool(projected)


def _w7_draw_named(
    ax: Axes,
    rows: Sequence[Dict[str, Any]],
    key: str,
    order: Sequence[str],
    labels: Dict[str, str],
    y_key: str,
    *,
    std_key: Optional[str] = None,
    log_y: bool = True,
) -> Tuple[int, bool, List[float]]:
    """Draw ``y_key`` against n for every named series in ``rows``.

    Returns:
        ``(series drawn, any projected point drawn, every x drawn)``.
    """
    drawn = 0
    any_projected = False
    xs: List[float] = []
    for index, name in enumerate(_w7_names(rows, key, order)):
        subset = [row for row in rows if _row_text(row, key) == name]
        points = _w7_points(subset, "n", y_key, std_key=std_key, log_y=log_y)
        if not points:
            continue
        style = _w7_style(name, index)
        _, projected = _w7_errorbar(ax, points, style, labels.get(name, name))
        any_projected = any_projected or projected
        xs.extend(point[0] for point in points)
        drawn += 1
    return drawn, any_projected, xs


def _w7_rate_group(
    ax: Axes,
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    rate_key: str,
    theory_key: str,
    style: Dict[str, Any],
    *,
    log_y: bool,
) -> Dict[str, Any]:
    """Draw one group's measured rates, zero-count bounds and theory line.

    Measured rates are markers with error bars from ``ci_low`` to
    ``ci_high``, not joined by a line: the theory line carries the group's
    line style, so the two differ in marker and in line style. On a log
    axis a rate of exactly zero is drawn at ``ci_high`` as a downward
    triangle; with no ``ci_high`` there is nothing honest to draw and the
    point is left out.

    Returns:
        ``{"measured": handle, "zero": handle, "theory": handle}`` with
        ``None`` for each part that was not drawn.
    """
    points = _w7_points(
        rows,
        x_key,
        rate_key,
        low_key="ci_low",
        high_key="ci_high",
        log_x=False,
        log_y=log_y,
    )
    zeros: List[Tuple[float, float]] = []
    if log_y:
        for row in rows:
            x_value = _w7_number(row, x_key)
            rate = _w7_number(row, rate_key)
            high = _w7_number(row, "ci_high")
            if x_value is None or rate != 0.0 or high is None or high <= 0.0:
                continue
            zeros.append((x_value, high))
        zeros.sort()
    handles: Dict[str, Any] = {"measured": None, "zero": None, "theory": None}
    if points:
        handles["measured"], _ = _w7_errorbar(ax, points, style, None, connect=False)
    if zeros:
        (handles["zero"],) = ax.plot(
            [x for x, _ in zeros],
            [y for _, y in zeros],
            color=style["color"],
            marker="v",
            linestyle="none",
            markersize=8.5,
            zorder=4,
        )
    theory = _w7_line(rows, x_key, theory_key, log_x=False, log_y=log_y)
    if theory:
        (handles["theory"],) = ax.plot(
            [x for x, _ in theory],
            [y for _, y in theory],
            color=style["color"],
            linestyle=style.get("theory", style["linestyle"]),
            linewidth=1.6,
            marker="",
            zorder=2,
        )
    return handles


def _w7_zero_proxy(ax: Axes) -> Any:
    """Return a neutral downward triangle that explains the zero-count points."""
    (proxy,) = ax.plot(
        [], [], color="0.35", marker="v", linestyle="none", markersize=8.5
    )
    return proxy


def _w7_legend(ax: Axes, entries: Sequence[Tuple[Any, str]]) -> None:
    """Draw a legend from ``(handle or tuple of handles, label)`` pairs.

    A tuple handle draws its parts side by side in one entry, which is how
    a measured series and its theory line share one row of the legend.
    """
    if not entries:
        return
    ax.legend(
        [handle for handle, _ in entries],
        [label for _, label in entries],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=8,
    )


def _w7_mean_std(values: Sequence[float]) -> Tuple[float, Optional[float]]:
    """Return the mean and the sample std (n - 1), the std ``None`` below two.

    Examples:
        >>> _w7_mean_std([1.0, 3.0]), _w7_mean_std([2.0])
        ((2.0, 1.4142135623730951), (2.0, None))
    """
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _w7_histogram(ax: Axes, values: Sequence[float], label: str, colour: str) -> None:
    """Draw a histogram with about sqrt(R) bins, between 5 and 40."""
    bins = int(min(40, max(5, round(math.sqrt(len(values))))))
    ax.hist(
        values,
        bins=bins,
        color=colour,
        alpha=0.6,
        edgecolor="0.25",
        linewidth=0.6,
        label=label,
        zorder=2,
    )


def _w7_vertical_reference(
    ax: Axes, x_value: float, label: str, colour: str, linestyle: Any
) -> None:
    """Draw one labelled vertical reference line across a histogram."""
    ax.axvline(
        x_value, color=colour, linestyle=linestyle, linewidth=2.0, label=label, zorder=3
    )


def _log_n_log_n(n: float) -> float:
    """Natural log of n log n, the expected quicksort shape (needs n > 1)."""
    return math.log(n) + math.log(math.log(n))


def _log_quadratic(n: float) -> float:
    """Natural log of n^2, the quicksort worst-case shape."""
    return 2.0 * math.log(n)


def _log_sqrt(n: float) -> float:
    """Natural log of sqrt(n), the trial-division shape."""
    return 0.5 * math.log(n)


def _log_inverse_sqrt(k: float) -> float:
    """Natural log of k^(-1/2), the MinHash error shape."""
    return -0.5 * math.log(k)


def _w7_triples(points: Sequence[_W7Point]) -> List[Tuple[float, float, bool]]:
    """Drop the bar lengths, for :func:`_draw_reference_shape`."""
    return [(x, y, projected) for x, y, _, _, projected in points]


def _w7_reference(
    ax: Axes,
    rows: Sequence[Dict[str, Any]],
    variant: str,
    log_shape: Callable[[float], float],
    shape: str,
    linestyle: Any,
    *,
    min_n: float = 0.0,
) -> None:
    """Draw a growth shape through one variant's smallest point.

    When that variant is absent the shape goes through the smallest point
    of the whole panel instead, and the legend says which. ``min_n`` keeps
    n log n away from n = 1, where its log is undefined.
    """
    subset = [row for row in rows if _row_text(row, "variant") == variant]
    points = [p for p in _w7_points(subset, "n", "mean_s") if p[0] > min_n]
    anchor = f"through {variant}'s smallest n"
    if not points:
        points = [p for p in _w7_points(rows, "n", "mean_s") if p[0] > min_n]
        anchor = "through the smallest point"
    _draw_reference_shape(
        ax, _w7_triples(points), log_shape, f"{shape} shape, {anchor}", linestyle
    )


def plot_quicksort_comparison(
    random_series: Sequence[Dict[str, Any]],
    sorted_series: Sequence[Dict[str, Any]],
    comparison_sample: Optional[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot randomized quicksort against fixed pivot rules, in four panels.

    Randomized quicksort is LAS VEGAS: its output is always sorted, and
    only its running time is random. So every panel here is about time and
    work, never about correctness:

    (a) Mean runtime against n on random input, log-log, +/- 1 std, for all
        four variants. On random input a fixed pivot is already as good as
        a random one, and this panel shows it.
    (b) Sorted input, randomized against the last-element pivot. Sorted
        input is exactly the adversarial case for a fixed rule, which falls
        to n(n - 1)/2 comparisons. An n log n and an n^2 shape are drawn
        through each series' smallest point, so the gap that opens between
        the two is the quadratic case. Points the benchmark projected
        rather than timed are hollow on a dashed segment.
    (c) A histogram of comparison counts from repeated sorts of one input,
        with the exact expectation 2(n + 1)H_n - 4n marked and a band of
        +/- 1 asymptotic std, about 0.6483 n. The randomness is in the
        algorithm, not the data, so this spread is what a user would see.
    (d) The coefficient of variation of runtime against n: how much one
        run differs from the next, as a fraction of the mean.

    Args:
        random_series: Dicts ``{variant, n, mean_s, std_s, cv}`` for the
            variants ``randomized``, ``deterministic_last``,
            ``deterministic_median_of_three`` and ``week2_quick_sort``.
        sorted_series: Dicts ``{variant, n, mean_s, std_s, measurement}``
            for ``randomized`` and ``deterministic_last``, ``measurement``
            being ``"measured"`` or ``"projected"``.
        comparison_sample: A dict ``{n, counts, expected, asymptotic_std}``,
            ``counts`` a list of per-sort comparison counts. ``None`` leaves
            panel (c) saying it has nothing to draw.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The four-panel :class:`matplotlib.figure.Figure`. Any value may be
        ``None``; it is skipped and never replaced by a number.

    Raises:
        OSError: If ``save_path`` cannot be written. Missing or unusable
            data never raises.

    Time Complexity:
        O(R log R + C) for R rows and C comparison counts: each variant's
        rows are read once and sorted by n, and each count is binned once.

    Space Complexity:
        O(R + C) for the extracted points and counts, plus the figure.

    Examples:
        >>> import tempfile
        >>> random_series = [
        ...     {"variant": "randomized", "n": 1000, "mean_s": 0.0021,
        ...      "std_s": 0.0001, "cv": 0.05},
        ...     {"variant": "randomized", "n": 4000, "mean_s": 0.0098,
        ...      "std_s": None, "cv": None},
        ...     {"variant": "deterministic_last", "n": 1000, "mean_s": 0.0019,
        ...      "std_s": 0.0001, "cv": 0.04},
        ...     {"variant": "deterministic_last", "n": 4000, "mean_s": 0.0090,
        ...      "std_s": 0.0003, "cv": 0.03}]
        >>> sorted_series = [
        ...     {"variant": "randomized", "n": 1000, "mean_s": 0.0022,
        ...      "std_s": 0.0001, "measurement": "measured"},
        ...     {"variant": "randomized", "n": 4000, "mean_s": 0.0101,
        ...      "std_s": 0.0004, "measurement": "measured"},
        ...     {"variant": "deterministic_last", "n": 1000, "mean_s": 0.051,
        ...      "std_s": 0.002, "measurement": "measured"},
        ...     {"variant": "deterministic_last", "n": 4000, "mean_s": 0.82,
        ...      "std_s": None, "measurement": "projected"}]
        >>> sample = {"n": 100, "counts": [610, 640, 655, None, 700],
        ...           "expected": 647.85, "asymptotic_std": 64.83}
        >>> png = os.path.join(tempfile.mkdtemp(), "quicksort_comparison.png")
        >>> fig = plot_quicksort_comparison(random_series, sorted_series,
        ...                                 sample, png)
        >>> len(fig.axes), fig.axes[0].get_xscale(), fig.axes[0].get_yscale()
        (4, 'log', 'log')
        >>> os.path.exists(png) and os.path.getsize(png) > 0
        True
        >>> plt.close(fig)

        Nothing at all to draw is not an error; each panel says so:

        >>> fig = plot_quicksort_comparison([], [], None)
        >>> fig.axes[2].texts[0].get_text()
        'no comparison counts were produced'
        >>> plt.close(fig)
    """
    apply_house_style()
    random_rows = _w7_rows(random_series)
    sorted_rows = _w7_rows(sorted_series)
    sample = comparison_sample if isinstance(comparison_sample, dict) else {}
    n_label = "Input size n (elements, log scale)"
    time_label = "Mean runtime (seconds, log scale)"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # (a) Random input, every variant, +/- 1 std.
    drawn_a, projected_a, _ = _w7_draw_named(
        ax_a,
        random_rows,
        "variant",
        _W7_QUICKSORT_ORDER,
        _W7_QUICKSORT_LABEL,
        "mean_s",
        std_key="std_s",
    )
    _finish_w6_panel(
        ax_a,
        drawn_a,
        projected_a,
        (n_label, time_label, r"(a) Random input: mean runtime $\pm$ 1 std"),
        "no random-input timings were produced",
        log_x=True,
        log_y=True,
    )

    # (b) Sorted input, with the two growth shapes it should separate into.
    drawn_b, projected_b, _ = _w7_draw_named(
        ax_b,
        sorted_rows,
        "variant",
        ("randomized", "deterministic_last"),
        _W7_QUICKSORT_LABEL,
        "mean_s",
        std_key="std_s",
    )
    if drawn_b:
        _w7_reference(
            ax_b,
            sorted_rows,
            "randomized",
            _log_n_log_n,
            r"$n \log n$",
            ":",
            min_n=1.0,
        )
        _w7_reference(
            ax_b,
            sorted_rows,
            "deterministic_last",
            _log_quadratic,
            r"$n^2$",
            (0, (5, 2, 1, 2)),
        )
    _finish_w6_panel(
        ax_b,
        drawn_b,
        projected_b,
        (n_label, time_label, "(b) Sorted input: the fixed pivot's worst case"),
        "no sorted-input timings were produced",
        log_x=True,
        log_y=True,
    )

    # (c) The spread of the comparison count around its exact expectation.
    counts = _w7_values(sample.get("counts"))
    expected = _w7_number(sample, "expected")
    spread = _w7_number(sample, "asymptotic_std")
    sample_n = _w7_number(sample, "n")
    drawn_c = 0
    if counts:
        mean, std = _w7_mean_std(counts)
        summary = f"mean {mean:,.1f}" + ("" if std is None else f", std {std:,.1f}")
        _w7_histogram(
            ax_c, counts, f"observed counts ({len(counts)} sorts; {summary})", "#2a9d8f"
        )
        drawn_c += 1
    if expected is not None:
        if spread is not None and spread > 0.0:
            ax_c.axvspan(
                expected - spread,
                expected + spread,
                color="#1d3557",
                alpha=0.13,
                label=rf"expected $\pm$ 1 asymptotic std ({spread:,.1f})",
                zorder=1,
            )
        _w7_vertical_reference(
            ax_c,
            expected,
            rf"expected $2(n+1)H_n - 4n$ = {expected:,.1f}",
            "#1d3557",
            "--",
        )
        drawn_c += 1
    title_c = "(c) Comparisons per sort"
    if sample_n is not None:
        title_c += f" at n = {_format_measure(sample_n)}"
    _finish_w6_panel(
        ax_c,
        drawn_c,
        False,
        ("Comparisons in one sort (element-vs-pivot tests)", "Sorts (count)", title_c),
        "no comparison counts were produced",
    )

    # (d) Run-to-run spread as a fraction of the mean.
    drawn_d, projected_d, _ = _w7_draw_named(
        ax_d,
        random_rows,
        "variant",
        _W7_QUICKSORT_ORDER,
        _W7_QUICKSORT_LABEL,
        "cv",
        log_y=False,
    )
    _finish_w6_panel(
        ax_d,
        drawn_d,
        projected_d,
        (
            n_label,
            "Coefficient of variation (std / mean, dimensionless)",
            "(d) Run-to-run variation of runtime, random input",
        ),
        "no coefficient of variation was produced",
        log_x=True,
    )

    fig.suptitle(
        "Quicksort: a random pivot (Las Vegas) against fixed pivot rules",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, save_path)
    return fig


def plot_selection_runtime(
    series: Sequence[Dict[str, Any]],
    depth_series: Sequence[Dict[str, Any]],
    ratio_sample: Optional[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot randomized select against median of medians, in three panels.

    Randomized select is LAS VEGAS: it always returns the k-th smallest
    element, so its probability of success is exactly 1 and is not worth
    plotting. What varies is the work, and the informative probability is
    how often a run finishes within 4n comparisons. The panels:

    1. Runtime against n, log-log, +/- 1 std. The line style is the
       algorithm and the marker is the order statistic (min, median, max),
       so both are readable without colour. Both are linear in n; the
       question is the constant.
    2. Mean partition depth of randomized select against n (log x), with a
       bar up to the deepest run seen and a log2(n) reference. The depth
       grows logarithmically even though each round's split is random.
    3. A histogram of comparisons / n over repeated runs, with the 4n
       threshold and the expected leading constant marked: 2(1 + ln 2),
       about 3.386, for the median, and 2 for the min or max.

    Args:
        series: Dicts ``{algorithm, k_label, n, mean_s, std_s}``,
            ``algorithm`` being ``"randomized"`` or ``"deterministic"`` and
            ``k_label`` one of ``"min"``, ``"median"``, ``"max"``.
        depth_series: Dicts ``{k_label, n, depth_mean, depth_max}`` for the
            randomized algorithm.
        ratio_sample: A dict ``{n, k_label, ratios}``, ``ratios`` a list of
            comparisons / n, one per run. ``None`` leaves panel 3 empty.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The three-panel :class:`matplotlib.figure.Figure`. Any value may be
        ``None``; it is skipped and never replaced by a number.

    Raises:
        OSError: If ``save_path`` cannot be written. Missing or unusable
            data never raises.

    Time Complexity:
        O(A K R log R + C) for A algorithms, K order statistics, R rows and
        C ratios: each (algorithm, k) series is selected by one pass over
        the rows and sorted by n.

    Space Complexity:
        O(R + C), plus the figure.

    Examples:
        >>> import tempfile
        >>> series = [
        ...     {"algorithm": "randomized", "k_label": "median", "n": 1000,
        ...      "mean_s": 0.0004, "std_s": 0.00005},
        ...     {"algorithm": "randomized", "k_label": "median", "n": 8000,
        ...      "mean_s": 0.0031, "std_s": None},
        ...     {"algorithm": "deterministic", "k_label": "median", "n": 1000,
        ...      "mean_s": 0.0019, "std_s": 0.0001},
        ...     {"algorithm": "deterministic", "k_label": "median", "n": 8000,
        ...      "mean_s": 0.0150, "std_s": 0.0006}]
        >>> depth = [
        ...     {"k_label": "median", "n": 1000, "depth_mean": 14.2,
        ...      "depth_max": 21},
        ...     {"k_label": "median", "n": 8000, "depth_mean": 19.8,
        ...      "depth_max": None}]
        >>> ratios = {"n": 1000, "k_label": "median",
        ...           "ratios": [3.1, 3.4, 3.3, 4.2, None]}
        >>> png = os.path.join(tempfile.mkdtemp(), "selection_runtime.png")
        >>> fig = plot_selection_runtime(series, depth, ratios, png)
        >>> [ax.get_xscale() for ax in fig.axes]
        ['log', 'log', 'linear']
        >>> os.path.exists(png) and os.path.getsize(png) > 0
        True
        >>> plt.close(fig)
    """
    apply_house_style()
    rows = _w7_rows(series)
    depth_rows = _w7_rows(depth_series)
    sample = ratio_sample if isinstance(ratio_sample, dict) else {}
    n_label = "Input size n (elements, log scale)"

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8))

    # Panel 1: line style per algorithm, marker per order statistic.
    drawn = 0
    any_projected = False
    k_labels = _w7_names(rows, "k_label", _W7_K_LABEL_ORDER)
    algorithms = _w7_names(rows, "algorithm", _W7_SELECTION_ORDER)
    for a_index, algorithm in enumerate(algorithms):
        base = _w7_style(algorithm, a_index)
        for k_index, k_label in enumerate(k_labels):
            subset = [
                row
                for row in rows
                if _row_text(row, "algorithm") == algorithm
                and _row_text(row, "k_label") == k_label
            ]
            points = _w7_points(subset, "n", "mean_s", std_key="std_s")
            if not points:
                continue
            style = dict(base, marker=_w7_k_style(k_label, k_index)["marker"])
            name = _W7_SELECTION_LABEL.get(algorithm, algorithm)
            _, projected = _w7_errorbar(
                axes[0], points, style, f"{name}, k = {k_label}"
            )
            any_projected = any_projected or projected
            drawn += 1
    _finish_w6_panel(
        axes[0],
        drawn,
        any_projected,
        (
            n_label,
            "Mean runtime (seconds, log scale)",
            r"Runtime against n ($\pm$ 1 std)",
        ),
        "no selection timings were produced",
        log_x=True,
        log_y=True,
    )

    # Panel 2: partition depth, bars up to the deepest run, log2(n) beside it.
    depth_drawn = 0
    depth_x: List[float] = []
    depth_labels = _w7_names(depth_rows, "k_label", _W7_K_LABEL_ORDER)
    for index, k_label in enumerate(depth_labels):
        subset = [row for row in depth_rows if _row_text(row, "k_label") == k_label]
        points = _w7_points(
            subset, "n", "depth_mean", high_key="depth_max", log_y=False
        )
        if not points:
            continue
        _w7_errorbar(
            axes[1],
            points,
            _w7_k_style(k_label, index),
            f"randomized, k = {k_label}: mean (bar to max)",
        )
        depth_x.extend(point[0] for point in points)
        depth_drawn += 1
    if depth_drawn and max(depth_x) > min(depth_x):
        xs = np.geomspace(min(depth_x), max(depth_x), 60)
        axes[1].plot(
            xs,
            np.log2(xs),
            color=_W7_REFERENCE_COLOUR,
            linestyle=(0, (1, 2)),
            linewidth=1.5,
            label=r"$\log_2 n$ reference",
            zorder=1,
        )
    _finish_w6_panel(
        axes[1],
        depth_drawn,
        False,
        (n_label, "Partition rounds (count)", "Randomized select: partition depth"),
        "no partition depths were produced",
        log_x=True,
    )

    # Panel 3: how the work is spread, against the 4n threshold.
    ratios = _w7_values(sample.get("ratios"))
    ratio_k = _row_text(sample, "k_label")
    ratio_n = _w7_number(sample, "n")
    ratio_drawn = 0
    if ratios:
        above = sum(1 for ratio in ratios if ratio > _W7_SELECT_THRESHOLD)
        _w7_histogram(
            axes[2],
            ratios,
            f"observed ({len(ratios)} runs; {above} above 4n)",
            "#2a9d8f",
        )
        _w7_vertical_reference(
            axes[2],
            _W7_SELECT_THRESHOLD,
            "4n threshold (comparisons / n = 4)",
            "#c1121f",
            "-.",
        )
        expected = _W7_SELECT_EXPECTED.get(ratio_k)
        if expected is not None:
            text = (
                rf"expected $2(1 + \ln 2) \approx {expected:.3f}$ (median)"
                if ratio_k == "median"
                else f"expected 2 ({ratio_k})"
            )
            _w7_vertical_reference(axes[2], expected, text, "#1d3557", "--")
        ratio_drawn = 1
    ratio_title = "Comparisons / n over repeated runs"
    details = []
    if ratio_k:
        details.append(f"k = {ratio_k}")
    if ratio_n is not None:
        details.append(f"n = {_format_measure(ratio_n)}")
    if details:
        ratio_title += f" ({', '.join(details)})"
    _finish_w6_panel(
        axes[2],
        ratio_drawn,
        False,
        ("Comparisons / n (comparisons per element)", "Runs (count)", ratio_title),
        "no comparison ratios were produced",
    )

    fig.suptitle(
        "Selection: randomized select (Las Vegas) against median of medians",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return fig


def plot_primality_accuracy(
    accuracy_rows: Sequence[Dict[str, Any]],
    speed_rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot how often Fermat and Miller-Rabin are fooled, and what they cost.

    Both tests are MONTE CARLO with ONE-SIDED error: "composite" is always
    right, because a witness was found, while "prime" can be wrong. So the
    measured quantity is the false-positive rate on a known composite, and
    the figure has two panels:

    (a) False-positive rate against the number of rounds k, log y, for both
        tests on each composite present (561 and 8911, both Carmichael
        numbers). Measured rates carry their 95% interval; the predicted
        rate is a line in the same colour; and the Miller-Rabin bound
        4^-k is drawn for comparison. Fermat has no such bound, and on a
        Carmichael number it visibly sits far above it. A rate of zero has
        no place on a log axis, so it is drawn at its interval's upper end
        as a downward triangle, labelled as a bound and not as a value.
    (b) Runtime against n, log-log, for trial division, Fermat and
        Miller-Rabin, with a sqrt(n) shape through the first trial-division
        point. Projected points are hollow on a dashed segment.

    Args:
        accuracy_rows: Dicts ``{test, n, k, rate, ci_low, ci_high,
            predicted, trials}``, ``test`` being ``"fermat"`` or
            ``"miller_rabin"``.
        speed_rows: Dicts ``{method, n, mean_s, std_s, measurement}``,
            ``method`` being ``"trial_division"``, ``"fermat"`` or
            ``"miller_rabin"``.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The two-panel :class:`matplotlib.figure.Figure`. Any value may be
        ``None``; it is skipped and never replaced by a number.

    Raises:
        OSError: If ``save_path`` cannot be written. Missing or unusable
            data never raises.

    Time Complexity:
        O(G R + R log R) for R rows and G (test, n) groups: each group is
        selected by one pass over the rows and sorted by k.

    Space Complexity:
        O(R), plus the figure.

    Examples:
        >>> import tempfile
        >>> accuracy = [
        ...     {"test": "fermat", "n": 561, "k": 1, "rate": 0.57,
        ...      "ci_low": 0.54, "ci_high": 0.60, "predicted": 0.5699,
        ...      "trials": 1000},
        ...     {"test": "fermat", "n": 561, "k": 4, "rate": 0.11,
        ...      "ci_low": 0.09, "ci_high": 0.13, "predicted": 0.1055,
        ...      "trials": 1000},
        ...     {"test": "miller_rabin", "n": 561, "k": 1, "rate": 0.015,
        ...      "ci_low": 0.009, "ci_high": 0.025, "predicted": 0.0143,
        ...      "trials": 1000},
        ...     {"test": "miller_rabin", "n": 561, "k": 4, "rate": 0.0,
        ...      "ci_low": 0.0, "ci_high": 0.0038, "predicted": None,
        ...      "trials": 1000}]
        >>> speed = [
        ...     {"method": "trial_division", "n": 10 ** 6, "mean_s": 2e-5,
        ...      "std_s": 1e-6, "measurement": "measured"},
        ...     {"method": "trial_division", "n": 10 ** 12, "mean_s": 0.02,
        ...      "std_s": None, "measurement": "projected"},
        ...     {"method": "miller_rabin", "n": 10 ** 6, "mean_s": 8e-6,
        ...      "std_s": 4e-7, "measurement": "measured"},
        ...     {"method": "miller_rabin", "n": 10 ** 12, "mean_s": 1.4e-5,
        ...      "std_s": 5e-7, "measurement": "measured"}]
        >>> png = os.path.join(tempfile.mkdtemp(), "primality_accuracy.png")
        >>> fig = plot_primality_accuracy(accuracy, speed, png)
        >>> legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        >>> "0 observed (95% upper bound)" in legend
        True
        >>> os.path.exists(png) and os.path.getsize(png) > 0
        True
        >>> plt.close(fig)
    """
    apply_house_style()
    accuracy = _w7_rows(accuracy_rows)
    speed = _w7_rows(speed_rows)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))

    # (a) Measured false-positive rate, its prediction, and the 4^-k bound.
    ax = axes[0]
    entries: List[Tuple[Any, str]] = []
    group = 0
    any_zero = False
    for test in _w7_names(accuracy, "test", _W7_PRIMALITY_TESTS):
        test_rows = [row for row in accuracy if _row_text(row, "test") == test]
        for n_value in _w7_distinct(test_rows, "n"):
            style = _w7_group_style(group)
            style["theory"] = style["linestyle"]
            group += 1
            parts = _w7_rate_group(
                ax,
                _w7_near(test_rows, "n", n_value),
                "k",
                "rate",
                "predicted",
                style,
                log_y=True,
            )
            any_zero = any_zero or parts["zero"] is not None
            shown = [parts["measured"] or parts["zero"], parts["theory"]]
            handles = tuple(handle for handle in shown if handle is not None)
            if handles:
                name = _W7_PRIMALITY_LABEL.get(test, test)
                entries.append((handles, f"{name}, n = {_format_measure(n_value)}"))
    ks = [k for k in _w7_distinct(accuracy, "k") if k >= 0.0]
    if entries and ks:
        (bound,) = ax.plot(
            ks,
            [4.0 ** -k for k in ks],
            color="black",
            linestyle="--",
            linewidth=1.4,
            marker="",
            zorder=1,
        )
        entries.append((bound, r"Miller-Rabin bound $4^{-k}$"))
    if any_zero:
        entries.append((_w7_zero_proxy(ax), _W7_ZERO_LABEL))
    trials = _w7_distinct(accuracy, "trials")
    title_a = "(a) Composites called prime (markers: 95% interval; lines: predicted)"
    if len(trials) == 1:
        title_a += f"\n{_format_measure(trials[0])} trials per point"
    _finish_w6_panel(
        ax,
        len(entries),
        False,
        (
            "Rounds k (random bases tried per test)",
            "False-positive rate (fraction of trials, log scale)",
            title_a,
        ),
        "no accuracy rows were produced",
        log_y=True,
    )
    if entries:
        if 0 < len(ks) <= 15:
            ax.set_xticks(ks)
            ax.set_xticklabels([_format_measure(k) for k in ks])
        _w7_legend(ax, entries)

    # (b) Cost against n, with sqrt(n) through the first trial-division point.
    drawn_b, projected_b, _ = _w7_draw_named(
        axes[1],
        speed,
        "method",
        _W7_SPEED_ORDER,
        _W7_PRIMALITY_LABEL,
        "mean_s",
        std_key="std_s",
    )
    trial_rows = [row for row in speed if _row_text(row, "method") == "trial_division"]
    trial_points = _w7_points(trial_rows, "n", "mean_s")
    if trial_points:
        _draw_reference_shape(
            axes[1],
            _w7_triples(trial_points),
            _log_sqrt,
            r"$\sqrt{n}$ shape, through the first trial-division point",
            ":",
        )
    _finish_w6_panel(
        axes[1],
        drawn_b,
        projected_b,
        (
            "Number tested n (integer value, log scale)",
            "Mean runtime per call (seconds, log scale)",
            r"(b) Runtime against n ($\pm$ 1 std)",
        ),
        "no primality timings were produced",
        log_x=True,
        log_y=True,
    )

    fig.suptitle(
        "Monte Carlo primality: one-sided error, measured against its bound",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return fig


def plot_bloom_fp_rate(
    load_rows: Sequence[Dict[str, Any]],
    bits_rows: Sequence[Dict[str, Any]],
    memory: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a Bloom filter's false-positive rate and the memory it saves.

    A Bloom filter is MONTE CARLO with ONE-SIDED error: "absent" is always
    right and "present" may be a false positive. The rate of those false
    positives is the whole cost of the structure, so it is what the figure
    measures, each rate with its 95% interval beside the theoretical
    ``(1 - e^(-kn/m))^k``:

    1. FP rate against load factor n / m, on linear axes, where a rate of
       zero is an ordinary point.
    2. FP rate against bits per item m / n, log y, where each extra bit per
       item should cut the rate by a constant factor. A rate of zero is
       drawn at its interval's upper end as a downward triangle. When each
       bits-per-item value appears once, the number of hashes k used there
       is written beside the point; when the rows sweep several k at the
       same budget, each k is its own series.
    3. Only when ``memory`` is given: the filter's bytes against a Python
       set holding the same items (log y), the filter's bar annotated with
       its FP rate. The set is exact; the filter trades that for its size.

    Args:
        load_rows: Dicts ``{load_factor, measured, ci_low, ci_high,
            theoretical}``.
        bits_rows: Dicts ``{bits_per_item, k, measured, ci_low, ci_high,
            theoretical}``.
        memory: Optional dict ``{n, bloom_bytes, set_bytes, fp_rate}``.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The :class:`matplotlib.figure.Figure`, with three panels when
        ``memory`` is a dict and two otherwise. Any value may be ``None``;
        it is skipped and never replaced by a number.

    Raises:
        OSError: If ``save_path`` cannot be written. Missing or unusable
            data never raises.

    Time Complexity:
        O(K R + R log R) for R rows and K distinct k values: each group is
        selected by one pass and sorted.

    Space Complexity:
        O(R), plus the figure.

    Examples:
        >>> import tempfile
        >>> load = [
        ...     {"load_factor": 0.05, "measured": 0.0, "ci_low": 0.0,
        ...      "ci_high": 0.0004, "theoretical": 0.0002},
        ...     {"load_factor": 0.1, "measured": 0.009, "ci_low": 0.007,
        ...      "ci_high": 0.011, "theoretical": 0.0082},
        ...     {"load_factor": 0.2, "measured": 0.13, "ci_low": None,
        ...      "ci_high": None, "theoretical": 0.138}]
        >>> bits = [
        ...     {"bits_per_item": 4, "k": 3, "measured": 0.15, "ci_low": 0.14,
        ...      "ci_high": 0.16, "theoretical": 0.147},
        ...     {"bits_per_item": 16, "k": 11, "measured": 0.0, "ci_low": 0.0,
        ...      "ci_high": 0.0004, "theoretical": 0.00046}]
        >>> memory = {"n": 1000, "bloom_bytes": 1199, "set_bytes": 32984,
        ...           "fp_rate": 0.01}
        >>> png = os.path.join(tempfile.mkdtemp(), "bloom_fp_rate.png")
        >>> fig = plot_bloom_fp_rate(load, bits, memory, png)
        >>> len(fig.axes), fig.axes[1].get_yscale()
        (3, 'log')
        >>> os.path.exists(png) and os.path.getsize(png) > 0
        True
        >>> plt.close(fig)
    """
    apply_house_style()
    load = _w7_rows(load_rows)
    bits = _w7_rows(bits_rows)
    memory_row: Dict[str, Any] = memory if isinstance(memory, dict) else {}
    has_memory = isinstance(memory, dict)
    panels = 3 if has_memory else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.4 * panels, 5.8))
    rate_label = "False-positive rate (fraction of queries"
    single = {"color": "#1d3557", "marker": "o", "linestyle": "none", "theory": "-"}
    theory_label = r"theoretical $(1 - e^{-kn/m})^k$"

    # Panel 1: against load factor, linear axes.
    parts = _w7_rate_group(
        axes[0], load, "load_factor", "measured", "theoretical", single, log_y=False
    )
    entries: List[Tuple[Any, str]] = []
    if parts["measured"] is not None:
        entries.append((parts["measured"], "measured (95% interval)"))
    if parts["theory"] is not None:
        entries.append((parts["theory"], theory_label))
    _finish_w6_panel(
        axes[0],
        len(entries),
        False,
        ("Load factor n / m (items per bit)", rate_label + ")", "FP rate against load"),
        "no load-factor rows were produced",
    )
    _w7_legend(axes[0], entries)

    # Panel 2: against bits per item, log y, zero rates as upper bounds.
    budgets = [_w7_number(row, "bits_per_item") for row in bits]
    known = [budget for budget in budgets if budget is not None]
    by_k = len(_w7_distinct(bits, "bits_per_item")) < len(known)
    groups: List[Tuple[Optional[float], List[Dict[str, Any]]]] = (
        [(k, _w7_near(bits, "k", k)) for k in _w7_distinct(bits, "k")]
        if by_k
        else [(None, bits)]
    )
    entries = []
    any_zero = False
    for index, (k_value, subset) in enumerate(groups):
        style = single if k_value is None else _w7_group_style(index)
        if k_value is not None:
            style["theory"] = style["linestyle"]
        parts = _w7_rate_group(
            axes[1],
            subset,
            "bits_per_item",
            "measured",
            "theoretical",
            style,
            log_y=True,
        )
        any_zero = any_zero or parts["zero"] is not None
        shown = parts["measured"] or parts["zero"]
        if k_value is None:
            if shown is not None:
                entries.append((shown, "measured (95% interval)"))
            if parts["theory"] is not None:
                entries.append((parts["theory"], theory_label))
        else:
            handles = tuple(h for h in (shown, parts["theory"]) if h is not None)
            if handles:
                entries.append((handles, f"k = {_format_measure(k_value)}"))
    if not by_k:
        for row in bits:
            x_value = _w7_number(row, "bits_per_item")
            k_value = _w7_number(row, "k")
            rate = _w7_number(row, "measured")
            y_value = _w7_number(row, "ci_high") if rate == 0.0 else rate
            if None in (x_value, k_value, y_value) or y_value <= 0.0:
                continue
            axes[1].annotate(
                f"k={_format_measure(k_value)}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="0.3",
            )
    if any_zero:
        entries.append((_w7_zero_proxy(axes[1]), _W7_ZERO_LABEL))
    _finish_w6_panel(
        axes[1],
        len(entries),
        False,
        (
            "Bits per item m / n (bits)",
            rate_label + ", log scale)",
            "FP rate against memory per item",
        ),
        "no bits-per-item rows were produced",
        log_y=True,
    )
    _w7_legend(axes[1], entries)

    # Panel 3: the memory trade, when the benchmark measured it.
    if has_memory:
        ax = axes[2]
        n_items = _w7_number(memory_row, "n")
        fp_rate = _w7_number(memory_row, "fp_rate")
        bars = (
            ("Bloom filter", "bloom_bytes", "#2a9d8f", "//"),
            ("Python set", "set_bytes", "#c1121f", ".."),
        )
        drawn = 0
        for position, (name, key, colour, hatch) in enumerate(bars):
            value = _w7_number(memory_row, key)
            if value is None or value <= 0.0:
                continue
            ax.bar(
                position,
                value,
                width=0.6,
                color=colour,
                alpha=0.8,
                hatch=hatch,
                edgecolor="0.2",
                label=f"{name}: {_format_measure(value)} bytes",
                zorder=2,
            )
            if key == "bloom_bytes":
                note = (
                    "FP rate not recorded"
                    if fp_rate is None
                    else f"FP rate {fp_rate:.2%}"
                )
            else:
                note = "exact: no false positives"
            ax.annotate(
                note,
                (position, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=8,
            )
            drawn += 1
        ax.set_xticks([0, 1])
        ax.set_xticklabels([name for name, _, _, _ in bars])
        ax.set_xlim(-0.7, 1.7)
        memory_title = "Memory for the same items"
        if n_items is not None:
            memory_title += f" (n = {_format_measure(n_items)})"
        _finish_w6_panel(
            ax,
            drawn,
            False,
            (
                "Structure holding the items (type)",
                "Memory (bytes, log scale)",
                memory_title,
            ),
            "no memory sizes were produced",
            log_y=True,
        )
        if drawn:
            ax.margins(y=0.25)

    fig.suptitle(
        "Bloom filter: one-sided Monte Carlo error against its theory",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, save_path)
    return fig


def plot_minhash_similarity(
    error_rows: Sequence[Dict[str, Any]],
    scatter_rows: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Plot how MinHash error falls with k, and estimates against truth.

    MinHash is MONTE CARLO with TWO-SIDED error: the estimate of a Jaccard
    similarity J is a fraction of k agreeing hash positions, so it can land
    either side of J, with standard error sqrt(J (1 - J) / k). Two panels:

    (a) Mean absolute error against k, log-log, one series per true J,
        each with +/- 1 std bars and its theoretical error as a dashed line
        in the same colour. A k^(-1/2) shape is drawn through the first
        series, so a series that runs parallel to it is shrinking at the
        predicted rate: quadrupling k halves the error. An error of exactly
        zero (J = 1 is estimated exactly) has no place on a log axis; such
        points are counted in a note instead of drawn.
    (b) Estimated against true J at k = 16 and k = 512, every estimate as a
        faint marker and the mean per J with +/- 1 std bars, beside the
        line y = x. When neither k is present, every k given is drawn.

    Args:
        error_rows: Dicts ``{k, true_j, mean_abs_error, std_abs_error,
            theoretical_mae}``.
        scatter_rows: Dicts ``{k, true_j, estimate}``, one per estimate.
        save_path: Optional PNG destination, written at 200 dpi.

    Returns:
        The two-panel :class:`matplotlib.figure.Figure`. Any value may be
        ``None``; it is skipped and never replaced by a number.

    Raises:
        OSError: If ``save_path`` cannot be written. Missing or unusable
            data never raises.

    Time Complexity:
        O(J R + R log R) for R rows and J distinct true similarities: each
        group is selected by one pass and sorted.

    Space Complexity:
        O(R), plus the figure.

    Examples:
        >>> import tempfile
        >>> errors = [
        ...     {"k": 16, "true_j": 0.5, "mean_abs_error": 0.10,
        ...      "std_abs_error": 0.07, "theoretical_mae": 0.0997},
        ...     {"k": 256, "true_j": 0.5, "mean_abs_error": 0.025,
        ...      "std_abs_error": 0.019, "theoretical_mae": 0.0249},
        ...     {"k": 16, "true_j": 1.0, "mean_abs_error": 0.0,
        ...      "std_abs_error": 0.0, "theoretical_mae": 0.0},
        ...     {"k": 256, "true_j": 0.2, "mean_abs_error": None,
        ...      "std_abs_error": None, "theoretical_mae": 0.02}]
        >>> scatter = [
        ...     {"k": 16, "true_j": 0.5, "estimate": 0.5625},
        ...     {"k": 16, "true_j": 0.5, "estimate": 0.4375},
        ...     {"k": 512, "true_j": 0.5, "estimate": 0.502},
        ...     {"k": 512, "true_j": 0.2, "estimate": None}]
        >>> png = os.path.join(tempfile.mkdtemp(), "minhash_similarity.png")
        >>> fig = plot_minhash_similarity(errors, scatter, png)
        >>> fig.axes[0].get_xscale(), fig.axes[0].get_yscale()
        ('log', 'log')
        >>> fig.axes[0].texts[0].get_text()
        '1 zero-error point not drawn (a log axis has no zero)'
        >>> os.path.exists(png) and os.path.getsize(png) > 0
        True
        >>> plt.close(fig)
    """
    apply_house_style()
    errors = _w7_rows(error_rows)
    scatter = _w7_rows(scatter_rows)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # (a) Error against k, one series per J, theory dashed beside it.
    ax = axes[0]
    entries: List[Tuple[Any, str]] = []
    zero_points = 0
    first_series: List[_W7Point] = []
    k_ticks: List[float] = []
    for index, true_j in enumerate(_w7_distinct(errors, "true_j")):
        subset = _w7_near(errors, "true_j", true_j)
        style = _w7_group_style(index)
        zero_points += sum(
            1 for row in subset if _w7_number(row, "mean_abs_error") == 0.0
        )
        handles: List[Any] = []
        points = _w7_points(subset, "k", "mean_abs_error", std_key="std_abs_error")
        if points:
            handle, _ = _w7_errorbar(ax, points, style, None)
            handles.append(handle)
            k_ticks.extend(point[0] for point in points)
            if not first_series and len(points) >= 2:
                first_series = points
        theory = _w7_line(subset, "k", "theoretical_mae", log_x=True, log_y=True)
        if theory:
            (line,) = ax.plot(
                [x for x, _ in theory],
                [y for _, y in theory],
                color=style["color"],
                linestyle=style["theory"],
                linewidth=1.5,
                marker="",
                zorder=2,
            )
            handles.append(line)
            k_ticks.extend(x for x, _ in theory)
        if handles:
            entries.append((tuple(handles), f"J = {_format_measure(true_j)}"))
    if first_series and _draw_reference_shape(
        ax,
        _w7_triples(first_series),
        _log_inverse_sqrt,
        r"$k^{-1/2}$ slope reference",
        ":",
    ):
        entries.append((ax.lines[-1], r"$k^{-1/2}$ slope reference"))
    _finish_w6_panel(
        ax,
        len(entries),
        False,
        (
            "Hash functions k (signature length, log scale)",
            "Mean absolute error (Jaccard units, log scale)",
            r"(a) Error against k (markers $\pm$ 1 std; dashed: theory)",
        ),
        "no error rows were produced",
        log_x=True,
        log_y=True,
        x_ticks=k_ticks,
    )
    if entries:
        _w7_legend(ax, entries)
        if zero_points:
            noun = "point" if zero_points == 1 else "points"
            ax.text(
                0.02,
                0.03,
                f"{zero_points} zero-error {noun} not drawn (a log axis has no zero)",
                transform=ax.transAxes,
                fontsize=8,
                color="0.3",
            )

    # (b) Estimate against truth at the two signature lengths.
    ax = axes[1]
    present = _w7_distinct(scatter, "k")
    chosen = [
        k for k in present if any(_same_number(k, s) for s in _W7_MINHASH_SCATTER_K)
    ] or present
    drawn = 0
    for index, k_value in enumerate(chosen):
        pairs = _w7_line(
            _w7_near(scatter, "k", k_value),
            "true_j",
            "estimate",
            log_x=False,
            log_y=False,
        )
        if not pairs:
            continue
        style = _w7_group_style(index)
        k_text = _format_measure(k_value)
        ax.scatter(
            [x for x, _ in pairs],
            [y for _, y in pairs],
            marker=style["marker"],
            color=style["color"],
            alpha=0.3,
            s=16,
            label=f"k = {k_text}: single estimates",
            zorder=2,
        )
        means: List[_W7Point] = []
        for true_j in sorted({x for x, _ in pairs}):
            values = [y for x, y in pairs if _same_number(x, true_j)]
            mean, std = _w7_mean_std(values)
            means.append((true_j, mean, std, std, False))
        _w7_errorbar(
            ax, means, style, rf"k = {k_text}: mean estimate $\pm$ 1 std"
        )
        drawn += 1
    if drawn:
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color=_W7_REFERENCE_COLOUR,
            linestyle=(0, (1, 2)),
            linewidth=1.4,
            label="y = x (exact estimate)",
            zorder=1,
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    scatter_title = "(b) Estimated against true J"
    if chosen:
        scatter_title += " at k = " + " and ".join(_format_measure(k) for k in chosen)
    _finish_w6_panel(
        ax,
        drawn,
        False,
        (
            "True Jaccard similarity J (fraction of shared items)",
            "MinHash estimate of J (fraction)",
            scatter_title,
        ),
        "no estimates were produced",
    )

    fig.suptitle(
        "MinHash: two-sided Monte Carlo error shrinking as $k^{-1/2}$",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path)
    return fig


# --- Week 8: computational complexity ---
#
# Both Week 8 figures are drawn from the benchmark's per-instance records:
# one dict per (problem, method, instance family, n, param, rep) with numeric
# ``time_s``, ``verify_time_s`` and ``ratio`` fields (``None`` where a field
# does not apply). Every series has its own marker and line style as well as
# its own colour, so no series is identified by colour alone, and every point
# carries a min-max range over the instances at that size.

__all__ += [
    "plot_runtime_growth",
    "plot_approximation_quality",
]

#: Marker, line style and colour for every Week 8 method, fixed so the two
#: figures read the same way.
_W8_STYLE: Dict[str, Dict[str, Any]] = {
    "backtracking": {"marker": "o", "linestyle": "-", "color": "#c0392b", "label": "backtracking (exact)"},
    "greedy": {"marker": "s", "linestyle": "--", "color": "#2471a3", "label": "greedy (heuristic)"},
    "random_restart": {"marker": "^", "linestyle": "-.", "color": "#7d3c98", "label": "random restart x100 (heuristic)"},
    "sat_backtracking": {"marker": "o", "linestyle": "-", "color": "#c0392b", "label": "plain backtracking (exact)"},
    "sat_dpll": {"marker": "D", "linestyle": "--", "color": "#d68910", "label": "DPLL (exact)"},
    "walksat": {"marker": "v", "linestyle": "-.", "color": "#1e8449", "label": "WalkSAT (heuristic)"},
    "exact_branching": {"marker": "o", "linestyle": "-", "color": "#c0392b", "label": "exact branching"},
    "two_approx": {"marker": "s", "linestyle": "--", "color": "#2471a3", "label": "2-approximation"},
    "max_degree_greedy": {"marker": "^", "linestyle": "-.", "color": "#7d3c98", "label": "max-degree greedy (heuristic)"},
    "held_karp": {"marker": "o", "linestyle": "-", "color": "#c0392b", "label": "Held-Karp (exact)"},
    "nearest_neighbor": {"marker": "s", "linestyle": "--", "color": "#2471a3", "label": "nearest neighbor (heuristic)"},
    "nn_two_opt": {"marker": "D", "linestyle": "-.", "color": "#1e8449", "label": "NN + 2-opt (heuristic)"},
}

_W8_VERIFY_STYLE: Dict[str, Any] = {"marker": "x", "linestyle": ":", "color": "#212121"}


def _w8_select(records: Sequence[Dict[str, Any]], **match: Any) -> List[Dict[str, Any]]:
    """Return the records whose fields equal every ``match`` item."""
    return [r for r in records if all(r.get(k) == v for k, v in match.items())]


def _w8_series(
    records: Sequence[Dict[str, Any]], field: str
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Group records by ``n`` and return ``(n, mean, low, high)`` lists for ``field``.

    Records whose ``field`` is ``None`` are skipped, and an ``n`` with no
    usable value is left out, so a series never invents a point.
    """
    by_n: Dict[float, List[float]] = {}
    for r in records:
        value = r.get(field)
        if value is None:
            continue
        by_n.setdefault(float(r["n"]), []).append(float(value))
    ns = sorted(by_n)
    means = [float(np.mean(by_n[n])) for n in ns]
    lows = [min(by_n[n]) for n in ns]
    highs = [max(by_n[n]) for n in ns]
    return ns, means, lows, highs


def _w8_plot(
    ax: Axes,
    records: Sequence[Dict[str, Any]],
    field: str,
    style: Dict[str, Any],
    label: str,
    hollow: bool = False,
) -> bool:
    """Draw one mean series with min-max error bars; return False if it is empty."""
    ns, means, lows, highs = _w8_series(records, field)
    if not ns:
        return False
    yerr = [np.subtract(means, lows), np.subtract(highs, means)]
    ax.errorbar(
        ns, means, yerr=yerr, marker=style["marker"], linestyle=style["linestyle"],
        color=style["color"], label=label, capsize=3, markersize=6, linewidth=1.6,
        markerfacecolor="white" if hollow else style["color"],
    )
    return True


def _w8_runtime_panel(
    ax: Axes,
    records: Sequence[Dict[str, Any]],
    problem: str,
    family: str,
    methods: Sequence[str],
    verify_records: Sequence[Dict[str, Any]],
    verify_label: str,
    title: str,
    param: Any = None,
    log_x: bool = False,
) -> None:
    """One runtime-growth panel: search times per method and the verifier's time."""
    for method in methods:
        match = {"problem": problem, "method": method, "instance_family": family}
        if param is not None:
            match["param"] = param
        _w8_plot(ax, _w8_select(records, **match), "time_s", _W8_STYLE[method],
                 _W8_STYLE[method]["label"])
    _w8_plot(ax, verify_records, "verify_time_s", _W8_VERIFY_STYLE, verify_label)
    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Input size n" + (" (log scale)" if log_x else ""))
    ax.set_ylabel("Time per instance (s, log scale)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7.5)


def plot_runtime_growth(
    records: Sequence[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Figure:
    """Draw ``runtime_growth.png``: exact search against heuristics and verifiers.

    Six panels. Four show runtime against input size on a log y axis for
    subset sum (worst-case family), random 3-SAT at clause ratio 4.26, vertex
    cover on random graphs and Euclidean TSP, each with the polynomial-time
    verifier timed on the certificates the solvers returned, drawn as a
    dotted line. The fifth plots the nodes subset-sum backtracking visits
    against a 2^n reference anchored at the smallest size, and the sixth
    compares SAT search nodes at clause ratio 4.26 with ratio 2.0.

    Args:
        records: Per-instance benchmark records (see the banner comment).
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The matplotlib figure.

    Examples:
        >>> fig = plot_runtime_growth([])
        >>> len(fig.axes)
        6
        >>> plt.close(fig)
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    (a, b, c), (d, e, f) = axes

    _w8_runtime_panel(
        a, records, "subset_sum", "worst_case", ("backtracking", "greedy", "random_restart"),
        _w8_select(records, problem="subset_sum", method="backtracking", instance_family="satisfiable"),
        "verify: sub-multiset and sum check", "(a) Subset sum, worst case (no solution exists)")
    _w8_runtime_panel(
        b, records, "sat", "random_3sat", ("sat_backtracking", "sat_dpll", "walksat"),
        _w8_select(records, problem="sat", method="sat_dpll", instance_family="random_3sat", param="4.26"),
        "verify: is_satisfying on returned assignment", "(b) Random 3-SAT, m/n = 4.26", param="4.26")
    _w8_runtime_panel(
        c, records, "vertex_cover", "gnp_avg_degree_4", ("exact_branching", "two_approx", "max_degree_greedy"),
        _w8_select(records, problem="vertex_cover", method="two_approx", instance_family="gnp_avg_degree_4"),
        "verify: is_vertex_cover on 2-approx cover", "(c) Vertex cover, G(n, p), average degree 4",
        log_x=True)
    _w8_runtime_panel(
        d, records, "tsp", "euclidean", ("held_karp", "nearest_neighbor", "nn_two_opt"),
        _w8_select(records, problem="tsp", method="nearest_neighbor", instance_family="euclidean"),
        "verify: permutation and tour-cost check", "(d) Euclidean TSP", log_x=True)

    worst = _w8_select(records, problem="subset_sum", method="backtracking", instance_family="worst_case")
    if _w8_plot(e, worst, "nodes", _W8_STYLE["backtracking"], "backtracking nodes visited"):
        ns, means, _, _ = _w8_series(worst, "nodes")
        ref = [means[0] * 2.0 ** (n - ns[0]) for n in ns]
        e.plot(ns, ref, linestyle=":", color="#212121", marker="", label="2^n reference (through first point)")
    e.set_yscale("log")
    e.set_xlabel("Input size n (number of values)")
    e.set_ylabel("Search nodes visited (log scale)")
    e.set_title("(e) Subset sum: nodes against 2^n")
    e.legend(loc="best", fontsize=7.5)

    for method in ("sat_backtracking", "sat_dpll"):
        for ratio, hollow in (("4.26", False), ("2.0", True)):
            style = _W8_STYLE[method]
            label = f"{style['label'].split(' (')[0]}, m/n = {ratio}"
            _w8_plot(f, _w8_select(records, problem="sat", method=method, param=ratio), "nodes",
                     dict(style, linestyle=style["linestyle"] if not hollow else ":"), label, hollow=hollow)
    f.set_yscale("log")
    f.set_xlabel("Number of variables n")
    f.set_ylabel("Search nodes visited (log scale)")
    f.set_title("(f) SAT: the same n, different instances")
    f.legend(loc="best", fontsize=7.5)

    fig.suptitle("Finding a solution grows exponentially; checking one stays flat "
                 "(markers: mean; bars: min-max over instances)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, save_path)
    return fig


def _w8_tradeoff(
    ax: Axes,
    points: Sequence[Tuple[str, str, float, float]],
    ylabel: str,
    title: str,
) -> None:
    """Scatter (time, quality) points, one labelled marker per method.

    Labels of points at nearly the same height are stacked downwards, so two
    methods that tie on quality never print their names on top of each other.
    """
    placed: List[float] = []
    for text, method, seconds, quality in points:
        style = _W8_STYLE[method]
        ax.scatter([seconds], [quality], marker=style["marker"], s=90, color=style["color"],
                   edgecolor="black", linewidth=0.6, zorder=3)
        same_height = sum(1 for q in placed if abs(q - quality) < 0.02)
        placed.append(quality)
        ax.annotate(text, (seconds, quality), textcoords="offset points",
                    xytext=(6, 6 - 13 * same_height), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean time per instance (s, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _w8_point(
    records: Sequence[Dict[str, Any]], field: str, **match: Any
) -> Optional[Tuple[float, float]]:
    """Mean ``time_s`` and mean ``field`` over the matching records, or ``None``."""
    chosen = [r for r in _w8_select(records, **match) if r.get(field) is not None]
    if not chosen:
        return None
    return (float(np.mean([r["time_s"] for r in chosen])),
            float(np.mean([float(r[field]) for r in chosen])))


def plot_approximation_quality(
    records: Sequence[Dict[str, Any]],
    tradeoff_n: Dict[str, int],
    save_path: Optional[str] = None,
) -> Figure:
    """Draw ``approximation_quality.png``: ratios against size, and trade-offs.

    Panel (a) is the vertex cover ratio to the exact optimum for the
    2-approximation and the max-degree greedy heuristic, with the proven
    y = 2 bound drawn and the K3,3 and star anchors marked. Panel (b) is the
    TSP ratio to the Held-Karp optimum for nearest neighbor and NN + 2-opt,
    the line anchor where nearest neighbor is 1.29x optimal, and, for sizes
    with no known optimum, nearest neighbor's cost relative to NN + 2-opt.
    Panels (c) and (d) are trade-off plots, one marker per method: time
    against approximation ratio for vertex cover and TSP, and time against
    the fraction of instances solved exactly for subset sum and SAT.

    Args:
        records: Per-instance benchmark records (see the banner comment).
        tradeoff_n: The size used for each trade-off point, keyed by
            ``"vertex_cover"``, ``"tsp"``, ``"subset_sum"`` and ``"sat"``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The matplotlib figure.

    Examples:
        >>> fig = plot_approximation_quality([], {})
        >>> len(fig.axes)
        4
        >>> plt.close(fig)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    (a, b), (c, d) = axes

    family = "gnp_avg_degree_4"
    for method in ("two_approx", "max_degree_greedy"):
        _w8_plot(a, _w8_select(records, problem="vertex_cover", method=method, instance_family=family),
                 "ratio", _W8_STYLE[method], _W8_STYLE[method]["label"])
    for anchor, marker in (("k33", "*"), ("star", "P")):
        hit = _w8_select(records, problem="vertex_cover", method="two_approx", instance_family=anchor)
        if hit:
            a.scatter([hit[0]["n"]], [hit[0]["ratio"]], marker=marker, s=160, color="#2471a3",
                      edgecolor="black", zorder=4, label=f"2-approx on {'K3,3' if anchor == 'k33' else 'star K1,5'}")
    a.axhline(2.0, color="#212121", linestyle="--", linewidth=1.2, label="proven bound: ratio <= 2")
    a.axhline(1.0, color="#7f8c8d", linestyle=":", linewidth=1.0, label="optimal")
    a.set_xlabel("Number of vertices n")
    a.set_ylabel("Cover size / minimum cover size")
    a.set_title("(a) Vertex cover: ratio to the exact optimum")
    a.legend(loc="best", fontsize=8)

    for method in ("nearest_neighbor", "nn_two_opt"):
        _w8_plot(b, _w8_select(records, problem="tsp", method=method, instance_family="euclidean"),
                 "ratio", _W8_STYLE[method], _W8_STYLE[method]["label"] + " / optimum")
    _w8_plot(b, _w8_select(records, problem="tsp", method="nearest_neighbor", instance_family="euclidean"),
             "nn_over_two_opt", dict(_W8_STYLE["nearest_neighbor"], linestyle=":"),
             "nearest neighbor / NN + 2-opt (no optimum known)", hollow=True)
    line = _w8_select(records, problem="tsp", method="nearest_neighbor", instance_family="line_anchor")
    if line:
        b.scatter([line[0]["n"]], [line[0]["ratio"]], marker="*", s=180, color="#2471a3",
                  edgecolor="black", zorder=4, label="nearest neighbor, line anchor (62 / 48)")
    b.axhline(1.0, color="#7f8c8d", linestyle=":", linewidth=1.0, label="optimal")
    b.set_xscale("log")
    b.set_xlabel("Number of cities n (log scale)")
    b.set_ylabel("Tour cost ratio")
    b.set_title("(b) TSP: no constant-factor guarantee")
    b.legend(loc="best", fontsize=8)

    points = []
    for problem, label, methods in (
        ("vertex_cover", "VC", ("exact_branching", "two_approx", "max_degree_greedy")),
        ("tsp", "TSP", ("held_karp", "nearest_neighbor", "nn_two_opt")),
    ):
        n = tradeoff_n.get(problem)
        fam = family if problem == "vertex_cover" else "euclidean"
        for method in methods:
            pt = _w8_point(records, "ratio", problem=problem, method=method, instance_family=fam, n=n)
            if pt:
                short = _W8_STYLE[method]["label"].split(" (")[0]
                points.append((f"{label} n={n}: {short}", method, pt[0], pt[1]))
    _w8_tradeoff(c, points, "Approximation ratio (1 = optimal)",
                 "(c) Trade-off: time against ratio")

    points = []
    n = tradeoff_n.get("subset_sum")
    for method in ("backtracking", "greedy", "random_restart"):
        pt = _w8_point(records, "success", problem="subset_sum", method=method,
                       instance_family="satisfiable", n=n)
        if pt:
            points.append((f"subset sum n={n}: {_W8_STYLE[method]['label'].split(' (')[0]}", method, pt[0], pt[1]))
    n = tradeoff_n.get("sat")
    for method in ("sat_backtracking", "sat_dpll", "walksat"):
        pt = _w8_point(records, "success", problem="sat", method=method, param="4.26", n=n)
        if pt:
            points.append((f"3-SAT n={n}: {_W8_STYLE[method]['label'].split(' (')[0]}", method, pt[0], pt[1]))
    _w8_tradeoff(d, points, "Fraction of instances answered exactly",
                 "(d) Trade-off: time against exact answers")
    d.set_ylim(-0.05, 1.12)

    fig.suptitle("Approximation and heuristics: cheaper certificates, weaker guarantees "
                 "(markers: mean; bars: min-max over instances)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, save_path)
    return fig
