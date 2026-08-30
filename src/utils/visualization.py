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

Every function returns the :class:`matplotlib.figure.Figure` it built and
writes a 200 dpi PNG when given ``save_path``.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from src.utils.benchmark import BenchmarkResult  # noqa: E402

__all__ = [
    "apply_house_style",
    "plot_data_type_sensitivity",
    "plot_complexity_fit",
    "plot_normalized_runtime",
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
