#!/usr/bin/env python3
"""Week 2 benchmark: five sorting algorithms across six sizes and six shapes.

Run from the repository root with the project virtualenv active::

    python benchmarks/week2_performance.py            # the full study
    python benchmarks/week2_performance.py --quick    # fast smoke run

Three studies:

**1. The main matrix** — bubble, selection and insertion sort (Week 1)
against merge sort and quicksort (Week 2), plus quicksort's three-way
partitioning variant, over sizes 100 … 50,000 and six data shapes.

**2. Doubling-ratio analysis** — for each true doubling of n, the observed
ratio t(2n)/t(n). Theory predicts 2 for O(n), just above 2 for O(n log n),
and 4 for O(n^2). This is the cleanest available evidence that measurement
matched theory, and it needs no curve fitting to read.

**3. Partition-scheme study** — Lomuto against Hoare against three-way on
duplicate-heavy input. The assignment expects three-way partitioning to
rescue a catastrophic case; this study establishes *which* case, and which
scheme is actually responsible for the catastrophe.

The O(n^2) algorithms are capped at n = :data:`QUADRATIC_MAX_SIZE`. See
:data:`QUADRATIC_CAP_REASON` and the report's Methodology section — the
omitted cells are recorded explicitly rather than dropped.

Outputs, all under ``benchmarks/results/``:

======================================= ====================================
File                                    Contents
======================================= ====================================
``random_data.png`` … ``few_unique.png`` One chart per data shape
``comparison_table.csv``                Every cell, measured or omitted
``doubling_ratios.csv``                 Observed vs predicted growth
``partition_scheme_study.csv``          Lomuto vs Hoare vs three-way
======================================= ====================================

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import signal
import sys
import time
from typing import Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.sorting import bubble_sort, insertion_sort, selection_sort  # noqa: E402
from src.sorting.merge_sort import merge_sort  # noqa: E402
from src.sorting.quick_sort import (  # noqa: E402
    INSERTION_SORT_CUTOFF,
    quick_sort,
    quick_sort_lomuto,
)
from src.utils.benchmark import AlgorithmBenchmark, BenchmarkResult  # noqa: E402
from src.utils.visualization import apply_house_style  # noqa: E402

#: Base seed. Every configuration derives a deterministic seed from it, and
#: quicksort's pivot selection uses it too, so the whole study reproduces.
SEED = 2026


# Named wrappers rather than functools.partial: the benchmark harness reads
# __name__ off the callable, and a partial does not have one.
def quick_sort_hoare(data: List) -> List:
    """QuickSort with two-way (Hoare) partitioning, seeded for reproducibility."""
    return quick_sort(data, seed=SEED)


def quick_sort_three_way(data: List) -> List:
    """QuickSort with three-way (Dutch national flag) partitioning."""
    return quick_sort(data, seed=SEED, three_way=True)


def quick_sort_lomuto_seeded(data: List) -> List:
    """QuickSort with Lomuto partitioning — comparison only, not production."""
    return quick_sort_lomuto(data, seed=SEED)


#: The five algorithms the assignment compares, plus the three-way variant.
ALGORITHMS: Dict[str, Callable] = {
    "Bubble Sort": bubble_sort,
    "Selection Sort": selection_sort,
    "Insertion Sort": insertion_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort_hoare,
    "Quick Sort (3-way)": quick_sort_three_way,
}

#: The quadratic three. Capped; see QUADRATIC_MAX_SIZE.
QUADRATIC: frozenset = frozenset(
    {"Bubble Sort", "Selection Sort", "Insertion Sort"}
)

SIZES: List[int] = [100, 500, 1000, 5000, 10000, 50000]

DATA_TYPES: List[str] = [
    "random",
    "sorted",
    "reverse",
    "nearly_sorted",
    "many_duplicates",
    "few_unique",
]

#: Filename stem per data shape, as the assignment names them.
FIGURE_NAMES: Dict[str, str] = {
    "random": "random_data",
    "sorted": "sorted_data",
    "reverse": "reverse_data",
    "nearly_sorted": "nearly_sorted",
    "many_duplicates": "many_duplicates",
    "few_unique": "few_unique",
}

#: Largest size measured for the O(n^2) algorithms.
QUADRATIC_MAX_SIZE = 10_000

QUADRATIC_CAP_REASON = (
    "not measured - projected runtime exceeded budget "
    f"(O(n^2) algorithms capped at n={QUADRATIC_MAX_SIZE:,})"
)

#: Measured runs per configuration: fewer at the large sizes, where each
#: run is expensive and the relative variance is already small.
RUNS_SMALL = 5
RUNS_LARGE = 3
LARGE_SIZE_THRESHOLD = 10_000

#: Hard ceiling on a single measured run. A cell that trips this is
#: recorded as timed out rather than being allowed to hang the suite.
PER_RUN_TIMEOUT_SECONDS = 120

#: Sizes for the partition-scheme study. Kept modest because Lomuto is
#: quadratic on this input, which is the entire point of the study.
PARTITION_STUDY_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
PARTITION_STUDY_TYPES = ["few_unique", "many_duplicates", "random"]

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")


class TimeoutExpired(Exception):
    """Raised when a single benchmarked run exceeds its time budget."""


class _time_limit:
    """Abort the enclosed block after `seconds` using SIGALRM.

    A pure-Python sort cannot be interrupted from another thread, but the
    interpreter checks for pending signals between bytecodes, so an alarm
    handler that raises will unwind a running sort. Unix only, which is
    fine — this project's platform is macOS. Where the signal machinery is
    unavailable the guard degrades to doing nothing rather than failing.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.armed = False

    def __enter__(self) -> "_time_limit":
        if self.seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return self

        def handler(signum, frame):  # pragma: no cover - timing dependent
            raise TimeoutExpired(f"exceeded {self.seconds:.0f}s")

        self.previous = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        self.armed = True
        return self

    def __exit__(self, *exc_info) -> bool:
        if self.armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
        return False


class Week2Benchmark(AlgorithmBenchmark):
    """The Week 1 harness, extended with a size-capped, timeout-guarded sweep.

    Subclasses rather than forks :class:`AlgorithmBenchmark`, so data
    generation, the timing protocol, correctness verification and the
    complexity fitting are all the Week 1 code. What is added here is the
    part Week 1 did not need: per-algorithm size caps, per-size run counts,
    a per-run timeout, and an explicit record of the cells that were
    deliberately not measured.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Cells that were skipped or timed out, kept so the results table
        #: can show a reason instead of a silent gap.
        self.omitted: List[dict] = []

    @staticmethod
    def runs_for(size: int) -> int:
        """Measured runs to use at a given input size."""
        return RUNS_LARGE if size >= LARGE_SIZE_THRESHOLD else RUNS_SMALL

    def should_measure(self, algorithm_name: str, size: int) -> bool:
        """False for the cells deliberately left out of the matrix."""
        return not (algorithm_name in QUADRATIC and size > QUADRATIC_MAX_SIZE)

    def run_matrix(
        self,
        algorithms: Dict[str, Callable],
        sizes: List[int],
        data_types: List[str],
    ) -> Dict[str, List[BenchmarkResult]]:
        """Sweep every algorithm over every size and data type.

        For each (size, data_type) the same generated list is handed to
        every algorithm, so a comparison between algorithms is never
        contaminated by a difference in their input.

        Returns:
            ``{algorithm_name: [BenchmarkResult, ...]}`` for the measured
            cells. Omitted cells are appended to ``self.omitted``.
        """
        matrix: Dict[str, List[BenchmarkResult]] = {name: [] for name in algorithms}
        planned = sum(
            1
            for _dt in data_types
            for size in sizes
            for name in algorithms
            if self.should_measure(name, size)
        )
        done = 0

        for data_type in data_types:
            print(f"\n  --- {data_type} ---", flush=True)
            for size in sizes:
                data = self.generate_test_data(
                    size, data_type, seed=SEED + size + 977 * DATA_TYPES.index(data_type)
                )
                runs = self.runs_for(size)

                for name, algorithm in algorithms.items():
                    if not self.should_measure(name, size):
                        self.omitted.append(
                            {
                                "algorithm_name": name,
                                "input_size": size,
                                "data_type": data_type,
                                "reason": QUADRATIC_CAP_REASON,
                            }
                        )
                        continue

                    started = time.perf_counter()
                    try:
                        with _time_limit(PER_RUN_TIMEOUT_SECONDS):
                            measured = self.time_algorithm(
                                algorithm, data, runs=runs, verify_correctness=True
                            )
                    except TimeoutExpired as exc:
                        self.omitted.append(
                            {
                                "algorithm_name": name,
                                "input_size": size,
                                "data_type": data_type,
                                "reason": f"not measured - timed out ({exc})",
                            }
                        )
                        print(
                            f"    [{done:>3}/{planned}] {name:<19} n={size:<6} "
                            f"TIMED OUT after {PER_RUN_TIMEOUT_SECONDS}s",
                            flush=True,
                        )
                        continue

                    # time_algorithm keys results by the function's __name__;
                    # re-label with the display name used everywhere else.
                    from dataclasses import replace

                    labelled = replace(
                        measured,
                        algorithm_name=name,
                        metadata={
                            **measured.metadata,
                            "data_type": data_type,
                            "runs": runs,
                        },
                    )
                    raw_key = measured.algorithm_name
                    self.results[raw_key].pop()
                    if not self.results[raw_key]:
                        del self.results[raw_key]
                    self.results.setdefault(name, []).append(labelled)
                    matrix[name].append(labelled)

                    done += 1
                    print(
                        f"    [{done:>3}/{planned}] {name:<19} n={size:<6} "
                        f"mean={labelled.average_time:9.6f}s  "
                        f"sd={labelled.std_deviation:8.6f}s  runs={runs}  "
                        f"(wall {time.perf_counter() - started:5.1f}s)",
                        flush=True,
                    )

        return matrix


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def doubling_ratios(matrix: Dict[str, List[BenchmarkResult]]) -> List[dict]:
    """Observed t(n2)/t(n1) against what each complexity class predicts.

    Only consecutive measured sizes are compared. Where n2 = 2*n1 the
    prediction is the familiar 2 / ~2.1 / 4; the other steps in this
    project's size ladder are 5x, so the predicted values are computed from
    the actual size ratio rather than assumed.
    """
    rows: List[dict] = []
    for name, series in matrix.items():
        points = sorted(series, key=lambda r: r.input_size)
        by_type: Dict[str, List[BenchmarkResult]] = {}
        for result in points:
            by_type.setdefault(result.data_type, []).append(result)

        for data_type, results in by_type.items():
            for earlier, later in zip(results, results[1:]):
                if earlier.average_time <= 0:
                    continue
                ratio = later.input_size / earlier.input_size
                observed = later.average_time / earlier.average_time
                predicted_nlogn = ratio * (
                    math.log2(later.input_size) / math.log2(earlier.input_size)
                )
                rows.append(
                    {
                        "algorithm_name": name,
                        "data_type": data_type,
                        "n_from": earlier.input_size,
                        "n_to": later.input_size,
                        "size_ratio": round(ratio, 3),
                        "is_true_doubling": ratio == 2.0,
                        "observed_ratio": round(observed, 3),
                        "predicted_linear": round(ratio, 3),
                        "predicted_linearithmic": round(predicted_nlogn, 3),
                        "predicted_quadratic": round(ratio ** 2, 3),
                        "t_from": earlier.average_time,
                        "t_to": later.average_time,
                    }
                )
    return rows


def partition_scheme_study(bench: Week2Benchmark) -> List[dict]:
    """Lomuto vs Hoare vs three-way, on the inputs where they diverge."""
    print("\n=== STUDY 3 - partition schemes on duplicate-heavy input ===")
    schemes = {
        "Lomuto (2-way)": quick_sort_lomuto_seeded,
        "Hoare (2-way)": quick_sort_hoare,
        "Dutch flag (3-way)": quick_sort_three_way,
    }
    rows: List[dict] = []

    for data_type in PARTITION_STUDY_TYPES:
        print(f"\n  --- {data_type} ---", flush=True)
        for size in PARTITION_STUDY_SIZES:
            data = bench.generate_test_data(size, data_type, seed=SEED + size)
            timings: Dict[str, Optional[float]] = {}
            for name, algorithm in schemes.items():
                try:
                    with _time_limit(PER_RUN_TIMEOUT_SECONDS):
                        result = bench.time_algorithm(
                            algorithm, data, runs=3, verify_correctness=True
                        )
                    timings[name] = result.average_time
                except TimeoutExpired:
                    timings[name] = None

            baseline = timings.get("Hoare (2-way)")
            for name, mean in timings.items():
                rows.append(
                    {
                        "scheme": name,
                        "data_type": data_type,
                        "input_size": size,
                        "mean_time": mean if mean is not None else "",
                        "speedup_vs_hoare": (
                            round(baseline / mean, 3)
                            if mean and baseline
                            else ""
                        ),
                    }
                )
            print(
                f"    n={size:<6} "
                + "  ".join(
                    f"{name}={timings[name]:.5f}s" if timings[name] else f"{name}=timeout"
                    for name in schemes
                ),
                flush=True,
            )

    # Growth ratios make the asymptotic difference explicit.
    for data_type in PARTITION_STUDY_TYPES:
        for scheme in schemes:
            series = [
                r
                for r in rows
                if r["scheme"] == scheme and r["data_type"] == data_type and r["mean_time"]
            ]
            series.sort(key=lambda r: r["input_size"])
            for earlier, later in zip(series, series[1:]):
                later["growth_vs_previous"] = round(
                    later["mean_time"] / earlier["mean_time"], 3
                )
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_comparison_table(
    matrix: Dict[str, List[BenchmarkResult]], omitted: List[dict], path: str
) -> str:
    """One row per (algorithm, size, data_type), including omitted cells."""
    fieldnames = [
        "algorithm_name",
        "data_type",
        "input_size",
        "mean_time",
        "std_deviation",
        "min_time",
        "max_time",
        "runs",
        "status",
        "note",
    ]
    rows: List[dict] = []

    for name, series in matrix.items():
        for result in series:
            rows.append(
                {
                    "algorithm_name": name,
                    "data_type": result.data_type,
                    "input_size": result.input_size,
                    "mean_time": result.average_time,
                    "std_deviation": result.std_deviation,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "runs": result.metadata.get("runs"),
                    "status": "measured",
                    "note": "",
                }
            )

    # Omitted cells appear as real rows with empty timings, so the table
    # shows a gap and its reason rather than silently lacking the row.
    for cell in omitted:
        rows.append(
            {
                "algorithm_name": cell["algorithm_name"],
                "data_type": cell["data_type"],
                "input_size": cell["input_size"],
                "mean_time": "",
                "std_deviation": "",
                "min_time": "",
                "max_time": "",
                "runs": 0,
                "status": "omitted",
                "note": cell["reason"],
            }
        )

    rows.sort(key=lambda r: (r["data_type"], r["algorithm_name"], r["input_size"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")
    return path


def write_csv(path: str, fieldnames: List[str], rows: List[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")
    return path


def plot_data_type(
    matrix: Dict[str, List[BenchmarkResult]], data_type: str, path: str
) -> str:
    """One chart per data shape: every measured algorithm, log-log, with refs."""
    fig, ax = plt.subplots(figsize=(9, 6.2))
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = plt.get_cmap("tab10").colors

    anchor_x = anchor_y = None
    for index, (name, series) in enumerate(matrix.items()):
        points = sorted(
            (r for r in series if r.data_type == data_type and r.average_time > 0),
            key=lambda r: r.input_size,
        )
        if not points:
            continue
        xs = np.array([r.input_size for r in points], dtype=float)
        ys = np.array([r.average_time for r in points], dtype=float)
        errs = np.array([r.std_deviation for r in points], dtype=float)

        ax.errorbar(
            xs, ys, yerr=errs,
            label=name,
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            markersize=7, linewidth=1.9, capsize=3, elinewidth=1.0,
        )
        if anchor_y is None or ys[0] > anchor_y:
            anchor_x, anchor_y = xs[0], ys[0]

    # Reference curves anchored to the slowest series' first point, so they
    # sit alongside the data rather than off the top of the axes.
    if anchor_x:
        all_sizes = sorted(
            {r.input_size for s in matrix.values() for r in s if r.data_type == data_type}
        )
        ref_x = np.linspace(min(all_sizes), max(all_sizes), 200)
        ax.plot(
            ref_x,
            anchor_y * (ref_x / anchor_x) * (np.log2(ref_x) / math.log2(anchor_x)),
            "--", color="0.45", linewidth=1.3, label="O(n log n) reference",
        )
        ax.plot(
            ref_x, anchor_y * (ref_x / anchor_x) ** 2,
            ":", color="0.15", linewidth=1.3, label="O(n$^2$) reference",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size n (elements)")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title(
        f"Sorting performance on {data_type} input\n"
        "(log-log axes; error bars are 1 SD over repeated runs)"
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} "
          f"({os.path.getsize(path) / 1024:.0f} KiB)")
    return path


def validate_cap(bench: Week2Benchmark) -> List[dict]:
    """Measure the capped cells and check them against the projection.

    The main matrix omits the O(n^2) algorithms at n=50,000 and the report
    projects those cells from a fitted t = c*n^2 curve. A projection nobody
    checks is just an assertion, so this study measures the omitted cells
    once and records predicted against actual.

    Fewer runs are used than in the main matrix: the question here is
    whether the projection lands within a few percent, and that does not
    need five repetitions of a sixty-second sort.

    Returns:
        One row per omitted cell, with the projected and measured times.
    """
    print("\n=== STUDY 4 - validating the projection for the capped cells ===")
    quadratic = [name for name in ALGORITHMS if name in QUADRATIC]
    rows: List[dict] = []

    # Fit t = c*n^2 through the origin, per algorithm and data type, using
    # only the sizes the main matrix actually measured.
    for name in quadratic:
        for data_type in DATA_TYPES:
            measured = sorted(
                (
                    r
                    for r in bench.results.get(name, [])
                    if r.data_type == data_type and r.input_size <= QUADRATIC_MAX_SIZE
                ),
                key=lambda r: r.input_size,
            )
            if len(measured) < 3:
                continue

            sizes = np.array([r.input_size for r in measured], dtype=float)
            times = np.array([r.average_time for r in measured], dtype=float)
            coefficient = float(np.sum(times * sizes ** 2) / np.sum(sizes ** 4))
            projected = coefficient * 50_000 ** 2

            data = bench.generate_test_data(
                50_000,
                data_type,
                seed=SEED + 50_000 + 977 * DATA_TYPES.index(data_type),
            )
            actual: Optional[float] = None
            try:
                with _time_limit(PER_RUN_TIMEOUT_SECONDS * 4):
                    result = bench.time_algorithm(
                        ALGORITHMS[name], data, runs=2, verify_correctness=True
                    )
                actual = result.average_time
                bench.results[result.algorithm_name].pop()
                if not bench.results[result.algorithm_name]:
                    del bench.results[result.algorithm_name]
            except TimeoutExpired:
                pass

            rows.append(
                {
                    "algorithm_name": name,
                    "data_type": data_type,
                    "input_size": 50_000,
                    "fitted_coefficient": f"{coefficient:.6e}",
                    "projected_time": round(projected, 4),
                    "measured_time": round(actual, 4) if actual else "",
                    "projection_error_pct": (
                        round(100 * (projected - actual) / actual, 2) if actual else ""
                    ),
                }
            )
            print(
                f"    {name:<16} {data_type:<16} projected={projected:8.2f}s  "
                + (
                    f"measured={actual:8.2f}s  error={100 * (projected - actual) / actual:+6.1f}%"
                    if actual
                    else "measured=TIMED OUT"
                ),
                flush=True,
            )
    return rows


def cutoff_study(bench: Week2Benchmark) -> List[dict]:
    """Measure the insertion-sort cutoff's effect by varying the threshold.

    The assignment asks for the optimization's impact to be measured rather
    than asserted, which means actually varying the constant. A threshold
    of 0 disables the optimization entirely, so quicksort partitions all
    the way down to single elements; larger thresholds hand progressively
    bigger ranges to insertion sort.

    The constant is patched on the module for the duration of each
    measurement. That is acceptable here because this is a controlled
    experiment in a benchmark script, and it is the only way to vary a
    value the algorithm reads as a module-level constant.

    Returns:
        One row per (threshold, data type, size).
    """
    print("\n=== STUDY 5 - insertion-sort cutoff threshold ===")
    import src.sorting.quick_sort as quick_sort_module

    thresholds = [0, 5, 10, 20, 40, 80]
    sizes = [10_000, 50_000]
    data_types = ["random", "nearly_sorted", "few_unique"]
    original = quick_sort_module.INSERTION_SORT_CUTOFF
    rows: List[dict] = []

    try:
        for data_type in data_types:
            print(f"\n  --- {data_type} ---", flush=True)
            for size in sizes:
                data = bench.generate_test_data(size, data_type, seed=SEED + size)
                baseline: Optional[float] = None
                line = []
                for threshold in thresholds:
                    quick_sort_module.INSERTION_SORT_CUTOFF = threshold
                    result = bench.time_algorithm(
                        quick_sort_hoare, data, runs=5, verify_correctness=True
                    )
                    bench.results[result.algorithm_name].pop()
                    if not bench.results[result.algorithm_name]:
                        del bench.results[result.algorithm_name]

                    if threshold == 0:
                        baseline = result.average_time
                    rows.append(
                        {
                            "threshold": threshold,
                            "data_type": data_type,
                            "input_size": size,
                            "mean_time": result.average_time,
                            "std_deviation": result.std_deviation,
                            "speedup_vs_no_cutoff": (
                                round(baseline / result.average_time, 4)
                                if baseline
                                else ""
                            ),
                        }
                    )
                    line.append(f"cutoff={threshold}:{result.average_time:.5f}s")
                print(f"    n={size:<6} " + "  ".join(line), flush=True)
    finally:
        quick_sort_module.INSERTION_SORT_CUTOFF = original

    return rows


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument(
        "--validate-cap",
        action="store_true",
        help=(
            "also measure the O(n^2) cells omitted at n=50,000 and compare "
            "them against the projection (adds roughly an hour)"
        ),
    )
    parser.add_argument(
        "--cutoff-only",
        action="store_true",
        help="run only the insertion-sort threshold study and exit",
    )
    args = parser.parse_args(argv)

    if args.cutoff_only:
        bench = Week2Benchmark(warmup_runs=2, precision=6, seed=SEED)
        bench.verbose = False
        write_csv(
            os.path.join(RESULTS_DIR, "cutoff_study.csv"),
            ["threshold", "data_type", "input_size", "mean_time",
             "std_deviation", "speedup_vs_no_cutoff"],
            cutoff_study(bench),
        )
        return 0

    sizes = [100, 500, 1000] if args.quick else SIZES
    data_types = DATA_TYPES[:2] if args.quick else DATA_TYPES

    started = time.perf_counter()
    print("=" * 78)
    print("CSC 5300 Week 2 - Divide and Conquer benchmark")
    print("=" * 78)
    print(f"  Robert Deibel")
    print(f"  python   : {platform.python_version()}")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()} / {os.cpu_count()} logical CPUs")
    print(f"  seed     : {SEED}")
    print(f"  cutoff   : INSERTION_SORT_CUTOFF = {INSERTION_SORT_CUTOFF}")
    print(f"  cap      : O(n^2) algorithms measured to n={QUADRATIC_MAX_SIZE:,} only")
    print(f"  runs     : {RUNS_SMALL} below n={LARGE_SIZE_THRESHOLD:,}, "
          f"{RUNS_LARGE} at or above")
    print(f"  mode     : {'QUICK SMOKE RUN' if args.quick else 'full study'}")

    bench = Week2Benchmark(warmup_runs=2, precision=6, seed=SEED)
    bench.verbose = False

    print("\n=== STUDY 1 - main matrix ===")
    matrix = bench.run_matrix(ALGORITHMS, sizes, data_types)

    print("\n=== STUDY 2 - doubling ratios ===")
    ratio_rows = doubling_ratios(matrix)
    true_doublings = [r for r in ratio_rows if r["is_true_doubling"]]
    print(f"  {len(ratio_rows)} consecutive-size steps, "
          f"{len(true_doublings)} of them true doublings")

    scheme_rows = partition_scheme_study(bench)

    cutoff_rows = cutoff_study(bench)

    validation_rows = validate_cap(bench) if args.validate_cap else []

    print("\n=== writing results ===")
    write_comparison_table(
        matrix, bench.omitted, os.path.join(RESULTS_DIR, "comparison_table.csv")
    )
    write_csv(
        os.path.join(RESULTS_DIR, "doubling_ratios.csv"),
        ["algorithm_name", "data_type", "n_from", "n_to", "size_ratio",
         "is_true_doubling", "observed_ratio", "predicted_linear",
         "predicted_linearithmic", "predicted_quadratic", "t_from", "t_to"],
        ratio_rows,
    )
    write_csv(
        os.path.join(RESULTS_DIR, "partition_scheme_study.csv"),
        ["scheme", "data_type", "input_size", "mean_time", "speedup_vs_hoare",
         "growth_vs_previous"],
        scheme_rows,
    )

    write_csv(
        os.path.join(RESULTS_DIR, "cutoff_study.csv"),
        ["threshold", "data_type", "input_size", "mean_time", "std_deviation",
         "speedup_vs_no_cutoff"],
        cutoff_rows,
    )

    if validation_rows:
        write_csv(
            os.path.join(RESULTS_DIR, "cap_validation.csv"),
            ["algorithm_name", "data_type", "input_size", "fitted_coefficient",
             "projected_time", "measured_time", "projection_error_pct"],
            validation_rows,
        )

    print("\n=== generating charts ===")
    apply_house_style()
    for data_type in data_types:
        plot_data_type(
            matrix, data_type,
            os.path.join(RESULTS_DIR, f"{FIGURE_NAMES[data_type]}.png"),
        )

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    print(f"  measured cells : {sum(len(v) for v in matrix.values())}")
    print(f"  omitted cells  : {len(bench.omitted)}")
    print(f"  results        : {os.path.relpath(RESULTS_DIR, REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
