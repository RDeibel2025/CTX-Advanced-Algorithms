#!/usr/bin/env python3
"""End-to-end Week 1 benchmark: three sorting algorithms, five input shapes.

Run from the repository root with the project virtualenv active::

    python benchmarks/sorting_benchmarks.py            # the full study
    python benchmarks/sorting_benchmarks.py --quick    # fast smoke run

The script performs three studies and writes everything the performance
report needs:

**1. The size sweep** — all three algorithms across sizes 100, 500, 1,000,
5,000 and 10,000 on five data types (random, sorted, reverse,
nearly_sorted, duplicates). Five measured runs per configuration after two
discarded warm-up runs. This is what the growth-rate analysis is built on.

**2. The data-shape study** — all three algorithms at one fixed size
across all *eight* generated shapes, including the three the size sweep
does not use (single_value, mountain, valley). Holding n constant isolates
the effect of input *order* from the effect of input *size*.

**3. A memory profile** — peak allocation per algorithm under
:mod:`tracemalloc`, measured in a separate un-timed run so it cannot
distort the timings.

Outputs:

===================================================== ==========================
File                                                  Contents
===================================================== ==========================
``benchmarks/results/sorting_benchmark_results.csv``  Every size-sweep measurement
``benchmarks/results/data_shape_study.csv``           Every data-shape measurement
``benchmarks/results/complexity_fits.csv``            Fitted model per algorithm
``benchmarks/results/memory_profile.csv``             Peak allocation per algorithm
``docs/figures/*.png``                                All charts, 200 dpi
===================================================== ==========================

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from typing import Dict, List

# Allow the script to run as `python benchmarks/sorting_benchmarks.py` from
# the repository root without the package being pip-installed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from src.sorting import bubble_sort, insertion_sort, selection_sort  # noqa: E402
from src.utils.benchmark import DATA_TYPES, AlgorithmBenchmark  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_complexity_fit,
    plot_data_type_sensitivity,
    plot_normalized_runtime,
)

#: The three algorithms, keyed by the display name used everywhere
#: downstream — in the charts, in the CSV and in the report.
ALGORITHMS: Dict[str, callable] = {
    "Bubble Sort": bubble_sort,
    "Selection Sort": selection_sort,
    "Insertion Sort": insertion_sort,
}

#: Sizes for the main sweep, as specified by the assignment.
SIZES: List[int] = [100, 500, 1000, 5000, 10000]

#: Data types for the main sweep, as specified by the assignment.
SWEEP_DATA_TYPES: List[str] = [
    "random",
    "sorted",
    "reverse",
    "nearly_sorted",
    "duplicates",
]

#: Size at which the data-shape study holds n constant. Large enough for
#: the differences between shapes to be far outside measurement noise,
#: small enough to sweep all eight shapes quickly.
SHAPE_STUDY_SIZE = 2000

#: Size at which peak memory is profiled.
MEMORY_PROFILE_SIZE = 1000

#: Measured runs per configuration.
RUNS = 5

#: Un-measured runs discarded before each measurement.
WARMUP_RUNS = 2

#: Base seed. Every configuration derives a distinct, deterministic seed
#: from this, so the entire study reproduces exactly.
SEED = 42

FIGURES_DIR = os.path.join(REPO_ROOT, "docs", "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")


def banner(text: str) -> None:
    """Print a section banner."""
    print()
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


def write_csv(path: str, fieldnames: List[str], rows: List[dict]) -> str:
    """Write ``rows`` to ``path`` as CSV and return the absolute path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  results written: {path} ({len(rows)} rows)", flush=True)
    return path


def run_size_sweep(sizes: List[int], runs: int) -> tuple:
    """Study 1: sweep every algorithm over every size and data type.

    Args:
        sizes: Input sizes to measure.
        runs: Measured runs per configuration.

    Returns:
        ``(benchmark, results)`` — the populated benchmark instance and the
        ``{algorithm: [BenchmarkResult, ...]}`` mapping.
    """
    banner(
        f"STUDY 1 - Size sweep: {len(ALGORITHMS)} algorithms x {len(sizes)} sizes "
        f"x {len(SWEEP_DATA_TYPES)} data types x {runs} runs"
    )
    print(f"  sizes      : {sizes}")
    print(f"  data types : {SWEEP_DATA_TYPES}")
    print(f"  warmup     : {WARMUP_RUNS} discarded runs per configuration")
    print(f"  seed       : {SEED}")
    print()

    benchmark = AlgorithmBenchmark(warmup_runs=WARMUP_RUNS, precision=6, seed=SEED)
    results = benchmark.benchmark_suite(
        ALGORITHMS, sizes=sizes, data_types=SWEEP_DATA_TYPES, runs=runs
    )
    return benchmark, results


def run_shape_study(size: int, runs: int) -> tuple:
    """Study 2: hold n constant and vary the input shape across all eight types.

    Args:
        size: The fixed input size.
        runs: Measured runs per configuration.

    Returns:
        ``(benchmark, results)``.
    """
    banner(
        f"STUDY 2 - Data-shape study at fixed n = {size:,} across all "
        f"{len(DATA_TYPES)} generated shapes"
    )
    benchmark = AlgorithmBenchmark(warmup_runs=WARMUP_RUNS, precision=6, seed=SEED)
    results = benchmark.benchmark_suite(
        ALGORITHMS, sizes=[size], data_types=list(DATA_TYPES), runs=runs
    )
    return benchmark, results


def run_memory_profile(size: int) -> List[dict]:
    """Study 3: peak allocation per algorithm, measured outside the timed runs.

    Args:
        size: Input size to profile.

    Returns:
        A list of CSV-ready rows.
    """
    banner(f"STUDY 3 - Peak memory profile at n = {size:,}")
    benchmark = AlgorithmBenchmark(warmup_runs=0, precision=6, seed=SEED)
    benchmark.verbose = False
    data = benchmark.generate_test_data(size, "random", seed=SEED)

    rows = []
    for name, algorithm in ALGORITHMS.items():
        result = benchmark.time_algorithm(
            algorithm, data, runs=1, verify_correctness=True, measure_memory=True
        )
        bytes_per_element = result.memory_usage / size if size else 0.0
        rows.append(
            {
                "algorithm_name": name,
                "input_size": size,
                "peak_bytes": int(result.memory_usage),
                "peak_kib": round(result.memory_usage / 1024, 2),
                "bytes_per_element": round(bytes_per_element, 2),
            }
        )
        print(
            f"  {name:<16} peak {result.memory_usage / 1024:8.2f} KiB "
            f"({bytes_per_element:5.1f} bytes/element)",
            flush=True,
        )
    return rows


def analyse(benchmark: AlgorithmBenchmark, results: Dict) -> List[dict]:
    """Fit a complexity model for every (algorithm, data type) pair.

    Args:
        benchmark: The instance whose ``analyze_complexity`` is used.
        results: The size-sweep results.

    Returns:
        A list of CSV-ready rows, one per algorithm and data type.
    """
    banner("Empirical complexity fits")
    rows = []
    header = (
        f"  {'algorithm':<16}{'data type':<16}{'best fit':<12}"
        f"{'R^2':>9}{'log-log k':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, series in results.items():
        for data_type in SWEEP_DATA_TYPES:
            subset = [r for r in series if r.data_type == data_type]
            if not subset:
                continue
            report = benchmark.analyze_complexity(subset, name)
            models = report["models"]
            if not models or report["best_fit"] is None:
                # Fewer than three distinct sizes: a three-model fit would
                # be meaningless, so report the gap rather than inventing
                # a number. Only reachable on a reduced/--quick run.
                print(
                    f"  {name:<16}{data_type:<16}"
                    f"{'(insufficient data: ' + '; '.join(report['notes']) + ')'}",
                    flush=True,
                )
                continue
            rows.append(
                {
                    "algorithm_name": name,
                    "data_type": data_type,
                    "n_points": report["n_points"],
                    "best_fit": report["best_fit"],
                    "best_r_squared": round(report["best_r_squared"], 6),
                    "empirical_exponent": round(report["empirical_exponent"], 4),
                    "r2_linear": round(models["O(n)"]["r_squared"], 6),
                    "r2_linearithmic": round(models["O(n log n)"]["r_squared"], 6),
                    "r2_quadratic": round(models["O(n^2)"]["r_squared"], 6),
                    "quadratic_coefficient": f"{models['O(n^2)']['coefficient']:.6e}",
                }
            )
            print(
                f"  {name:<16}{data_type:<16}{str(report['best_fit']):<12}"
                f"{report['best_r_squared']:>9.5f}"
                f"{report['empirical_exponent']:>12.3f}",
                flush=True,
            )
    return rows


def make_charts(
    benchmark: AlgorithmBenchmark,
    sweep_results: Dict,
    shape_results: Dict,
    fit_rows: List[dict],
    largest_size: int,
) -> List[str]:
    """Generate every chart the report embeds.

    Args:
        benchmark: The size-sweep benchmark instance.
        sweep_results: Study 1 results.
        shape_results: Study 2 results.
        fit_rows: Rows returned by :func:`analyse` (unused directly, kept
            so the call site reads as the report's dependency order).
        largest_size: The largest measured size, used for the sensitivity
            slice.

    Returns:
        Absolute paths of the PNG files written.
    """
    banner("Generating charts")
    apply_house_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    written: List[str] = []

    def figure_path(name: str) -> str:
        return os.path.join(FIGURES_DIR, name)

    # 1. The headline comparison: one panel per data type, log-log, with
    #    O(n) and O(n^2) reference curves.
    path = figure_path("sorting_comparison_all_data_types.png")
    benchmark.plot_comparison(
        sweep_results,
        title="Sorting algorithm runtime by input size and input shape",
        log_scale=True,
        save_path=path,
    )
    written.append(path)

    # 2. The same comparison restricted to random input, which is the
    #    series the growth-rate discussion refers to.
    random_only = {
        name: [r for r in series if r.data_type == "random"]
        for name, series in sweep_results.items()
    }
    path = figure_path("sorting_comparison_random.png")
    benchmark.plot_comparison(
        random_only,
        title="Sorting algorithm runtime on random input",
        log_scale=True,
        save_path=path,
    )
    written.append(path)

    # 3. Sensitivity to input order at the largest measured size.
    path = figure_path("data_type_sensitivity.png")
    plot_data_type_sensitivity(
        sweep_results,
        largest_size,
        title=f"Effect of input order at fixed n = {largest_size:,}",
        save_path=path,
    )
    written.append(path)

    # 4. All eight generated shapes at one size.
    path = figure_path("data_shape_study.png")
    plot_data_type_sensitivity(
        shape_results,
        SHAPE_STUDY_SIZE,
        title=(
            f"All eight generated input shapes at n = {SHAPE_STUDY_SIZE:,}"
        ),
        save_path=path,
    )
    written.append(path)

    # 5. Fitted models drawn against the measurements, on random input.
    analyses = [
        benchmark.analyze_complexity(
            [r for r in sweep_results[name] if r.data_type == "random"], name
        )
        for name in sweep_results
    ]
    path = figure_path("complexity_fits_random.png")
    plot_complexity_fit(
        analyses,
        title="Measured runtimes against fitted complexity models (random input)",
        save_path=path,
    )
    written.append(path)

    # 6. The same for sorted input, where the three algorithms diverge.
    analyses_sorted = [
        benchmark.analyze_complexity(
            [r for r in sweep_results[name] if r.data_type == "sorted"], name
        )
        for name in sweep_results
    ]
    path = figure_path("complexity_fits_sorted.png")
    plot_complexity_fit(
        analyses_sorted,
        title="Measured runtimes against fitted complexity models (sorted input)",
        save_path=path,
    )
    written.append(path)

    # 7. Runtime normalised by n^2: a flat line means genuinely quadratic.
    for data_type in ("random", "sorted"):
        path = figure_path(f"normalized_runtime_{data_type}.png")
        plot_normalized_runtime(sweep_results, data_type, save_path=path)
        written.append(path)

    plt.close("all")
    for path in written:
        size_kib = os.path.getsize(path) / 1024
        print(f"  chart written: {os.path.relpath(path, REPO_ROOT)} ({size_kib:.0f} KiB)")
    return written


def print_results_table(results: Dict, sizes: List[int]) -> None:
    """Print the mean and standard deviation table that the report reproduces."""
    banner("Results table - mean runtime in seconds (standard deviation)")
    column = 24
    for data_type in SWEEP_DATA_TYPES:
        print()
        print(f"  {data_type}")
        header = f"  {'algorithm':<16}" + "".join(
            f"{('n=' + format(n, ',')):>{column}}" for n in sizes
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, series in results.items():
            lookup = {
                r.input_size: r for r in series if r.data_type == data_type
            }
            cells = ""
            for n in sizes:
                if n in lookup:
                    r = lookup[n]
                    cell = f"{r.average_time:.6f} ({r.std_deviation:.6f})"
                else:
                    cell = "-"
                cells += f"{cell:>{column}}"
            print(f"  {name:<16}{cells}")


def main(argv: List[str] = None) -> int:
    """Run the whole study and write every output file.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark the Week 1 sorting algorithms end to end."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast smoke version over smaller sizes with fewer runs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help=f"Measured runs per configuration (default {RUNS}).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Cap the largest input size, for a shorter run.",
    )
    args = parser.parse_args(argv)

    sizes = [100, 250, 500, 1000] if args.quick else list(SIZES)
    runs = 3 if args.quick else args.runs
    if args.max_size is not None:
        sizes = [n for n in sizes if n <= args.max_size] or [min(sizes)]

    started = time.perf_counter()

    banner("CSC 5300 Advanced Algorithms - Week 1 Sorting Benchmark")
    print(f"  Robert Deibel")
    print(f"  python   : {platform.python_version()}")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()} / {os.cpu_count()} logical CPUs")
    print(f"  mode     : {'QUICK SMOKE RUN' if args.quick else 'full study'}")

    sweep_benchmark, sweep_results = run_size_sweep(sizes, runs)
    print_results_table(sweep_results, sizes)

    shape_benchmark, shape_results = run_shape_study(SHAPE_STUDY_SIZE, runs)
    memory_rows = run_memory_profile(MEMORY_PROFILE_SIZE)
    fit_rows = analyse(sweep_benchmark, sweep_results)

    banner("Writing result files")
    sweep_benchmark.export_results(
        os.path.join(RESULTS_DIR, "sorting_benchmark_results.csv")
    )
    shape_benchmark.export_results(
        os.path.join(RESULTS_DIR, "data_shape_study.csv")
    )
    write_csv(
        os.path.join(RESULTS_DIR, "complexity_fits.csv"),
        [
            "algorithm_name", "data_type", "n_points", "best_fit",
            "best_r_squared", "empirical_exponent", "r2_linear",
            "r2_linearithmic", "r2_quadratic", "quadratic_coefficient",
        ],
        fit_rows,
    )
    write_csv(
        os.path.join(RESULTS_DIR, "memory_profile.csv"),
        ["algorithm_name", "input_size", "peak_bytes", "peak_kib", "bytes_per_element"],
        memory_rows,
    )

    make_charts(
        sweep_benchmark, sweep_results, shape_results, fit_rows, max(sizes)
    )

    elapsed = time.perf_counter() - started
    banner(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  charts  : {os.path.relpath(FIGURES_DIR, REPO_ROOT)}/")
    print(f"  results : {os.path.relpath(RESULTS_DIR, REPO_ROOT)}/")
    print(f"  report  : docs/performance_analysis.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
