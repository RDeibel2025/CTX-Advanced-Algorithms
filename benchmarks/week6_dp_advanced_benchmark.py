#!/usr/bin/env python3
"""Week 6 benchmark: space-optimized DP, matrix chains, all-pairs paths and TSP.

Run from the repository root with the project virtualenv active::

    python benchmarks/week6_dp_advanced_benchmark.py              # the full study
    python benchmarks/week6_dp_advanced_benchmark.py --smoke      # seconds, tiny sizes
    python benchmarks/week6_dp_advanced_benchmark.py --smoke --out /tmp/w6

Four studies, the four the instructions name:

1. **Knapsack, standard against space-optimized.** Week 5's 2D table
   against the one-row version, swept over capacity at 100 items and over
   item count at capacity 1,000, timed and traced for peak memory.
2. **Matrix chain multiplication, recursion against DP.** Plain recursion to
   16 matrices; memoized and bottom-up to 200.
3. **Floyd-Warshall.** Scaling at n = 50, 100, 200 and 500; against Week 4's
   Dijkstra run from every source, on sparse and dense graphs, with the two
   distance matrices checked equal before either time is recorded; the
   textbook 3D form against the in-place 2D form for memory; and one graph
   with negative edges, where Week 4's Dijkstra refuses to run.
4. **TSP, Held-Karp bitmask DP against brute force**, n = 4 to 18 and 4 to
   12, with the two costs checked equal wherever both ran.

Repetitions follow the cost of a pilot run (see :func:`repeats_for`), and
the count used is written into every CSV row. Peak memory is traced in a
separate pass, and only for runs under :data:`TRACE_LIMIT_S`, because
tracemalloc slows execution by up to an order of magnitude.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.dp.knapsack import knapsack_tab  # noqa: E402
from src.dp_advanced.bitmask_traveling_salesman import (  # noqa: E402
    tsp_bitmask,
    tsp_brute_force,
)
from src.dp_advanced.floyd_warshall import (  # noqa: E402
    all_pairs_dijkstra,
    floyd_warshall,
    floyd_warshall_3d,
)
from src.dp_advanced.matrix_chain_multiplication import (  # noqa: E402
    matrix_chain_order,
    mcm_bottom_up,
    mcm_memoized,
    mcm_recursive,
)
from src.dp_advanced.space_optimized_knapsack import (  # noqa: E402
    compare_with_standard,
    knapsack_space_optimized,
    print_comparison,
)
from src.graphs.graph import Graph  # noqa: E402
from src.utils.graph_generator import weighted_graph  # noqa: E402
from src.utils.matrix_utils import (  # noqa: E402
    graph_to_weight_matrix,
    matrices_close,
    random_dimensions,
    random_weight_matrix,
)
from src.utils.timer import peak_memory_kib, time_call  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_floyd_warshall_scaling,
    plot_knapsack_space_comparison,
    plot_mcm_performance,
    plot_mcm_table,
    plot_tsp_runtime,
)

#: Every generated instance derives from this.
SEED = 42

#: Runs at or above this are timed but not traced for memory. tracemalloc
#: costs up to an order of magnitude in speed, and none of the untraced runs
#: is one the memory comparison needs.
TRACE_LIMIT_S = 2.0

#: The CSV's columns are fixed, because the report reads them back by name.
COLUMNS = [
    "problem", "variant", "n", "secondary_param", "mean_time_s", "std_time_s",
    "min_time_s", "max_time_s", "runs", "peak_kib", "theoretical_time",
    "theoretical_space", "speedup_vs_baseline", "baseline_variant", "measurement",
]

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

#: Theoretical bounds per (problem, variant), written into every row.
THEORY: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("knapsack", "standard_2d"): ("O(n*W)", "O(n*W)"),
    ("knapsack", "space_optimized_1d"): ("O(n*W)", "O(W)"),
    # 3^(n-1) calls exactly (CLRS 14.3 proves the weaker Omega(2^n)). The
    # Catalan count, about 4^n / n^1.5, is the number of parenthesizations,
    # which the recursion never enumerates one by one.
    ("mcm", "recursive"): ("O(3^n)", "O(n)"),
    ("mcm", "memoized"): ("O(n^3)", "O(n^2)"),
    ("mcm", "bottom_up"): ("O(n^3)", "O(n^2)"),
    ("floyd_warshall", "floyd_warshall"): ("O(V^3)", "O(V^2)"),
    ("floyd_warshall", "floyd_warshall_3d"): ("O(V^3)", "O(V^3)"),
    ("floyd_warshall", "all_pairs_dijkstra"): ("O(V (V+E) log V)", "O(V^2)"),
    ("tsp", "bitmask"): ("O(n^2 2^n)", "O(n 2^n)"),
    ("tsp", "brute_force"): ("O(n!)", "O(n)"),
}

#: The baseline each variant's speedup is measured against.
BASELINE: Dict[Tuple[str, str], str] = {
    ("knapsack", "space_optimized_1d"): "standard_2d",
    ("mcm", "memoized"): "recursive",
    ("mcm", "bottom_up"): "recursive",
    ("floyd_warshall", "floyd_warshall"): "all_pairs_dijkstra",
    ("tsp", "bitmask"): "brute_force",
}


@dataclass
class Config:
    """Every size the study uses, in one place."""

    knapsack_n: int
    knapsack_capacities: List[int]
    knapsack_capacity: int
    knapsack_sizes: List[int]
    mcm_recursive_sizes: List[int]
    mcm_dp_sizes: List[int]
    fw_compare_sizes: List[int]
    fw_scaling_sizes: List[int]
    fw_3d_sizes: List[int]
    densities: List[float]
    tsp_bitmask_sizes: List[int]
    tsp_brute_sizes: List[int]


FULL = Config(
    knapsack_n=100,
    knapsack_capacities=[100, 500, 1_000, 2_500, 5_000, 10_000],
    knapsack_capacity=1_000,
    knapsack_sizes=[25, 50, 100, 200, 400],
    mcm_recursive_sizes=[4, 6, 8, 10, 12, 14, 16],
    mcm_dp_sizes=[4, 6, 8, 10, 12, 14, 16, 25, 50, 100, 200],
    fw_compare_sizes=[50, 100, 200],
    fw_scaling_sizes=[50, 100, 200, 500],
    fw_3d_sizes=[25, 50, 100],
    densities=[0.05, 0.5],
    tsp_bitmask_sizes=list(range(4, 19)),
    tsp_brute_sizes=list(range(4, 13)),
)

SMOKE = Config(
    knapsack_n=20,
    knapsack_capacities=[50, 100],
    knapsack_capacity=100,
    knapsack_sizes=[10, 20],
    mcm_recursive_sizes=[4, 6],
    mcm_dp_sizes=[4, 6, 10],
    fw_compare_sizes=[10, 20],
    fw_scaling_sizes=[10, 20, 30],
    fw_3d_sizes=[10, 20],
    densities=[0.2, 0.5],
    tsp_bitmask_sizes=[4, 5, 6, 7],
    tsp_brute_sizes=[4, 5, 6],
)


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
def repeats_for(pilot_seconds: float) -> Tuple[int, int]:
    """Choose (repeat, warmup) from what one pilot run cost.

    Five runs after two warm-ups is right for millisecond work and wasteful
    for a 60-second brute-force tour. Long runs drop to three repeats with
    the pilot serving as the warm-up, and anything over 20 seconds is a
    single measured run. The count used is recorded in every row.
    """
    if pilot_seconds < 0.05:
        return 5, 2
    if pilot_seconds < 0.5:
        return 5, 1
    if pilot_seconds < 20.0:
        return 3, 0
    return 1, 0


def _timed_once(function: Callable[[], Any]) -> float:
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        started = time.perf_counter()
        function()
        return time.perf_counter() - started
    finally:
        if was_enabled:
            gc.enable()


def measure(function: Callable[[], Any]) -> Dict[str, float]:
    """Time a zero-argument call under the repetition policy above."""
    pilot = _timed_once(function)
    repeat, warmup = repeats_for(pilot)
    if repeat == 1:
        # The pilot already is one complete, GC-paused measured run. Paying
        # for a second one at this size would only double the wait.
        return {"mean": pilot, "std": 0.0, "min": pilot, "max": pilot, "runs": 1}
    return time_call(function, repeat=repeat, warmup=warmup)


def trace(function: Callable[[], Any], mean_seconds: float) -> Any:
    """Peak KiB in a separate traced pass, or "" for runs too long to trace."""
    if mean_seconds >= TRACE_LIMIT_S:
        return ""
    _, peak = peak_memory_kib(function)
    return peak


def make_row(problem: str, variant: str, n: int, secondary: Any,
             stats: Dict[str, float], peak: Any) -> Dict[str, Any]:
    theory_time, theory_space = THEORY[(problem, variant)]
    return {
        "problem": problem, "variant": variant, "n": n,
        "secondary_param": secondary,
        "mean_time_s": f"{stats['mean']:.9f}",
        "std_time_s": f"{stats['std']:.9f}",
        "min_time_s": f"{stats['min']:.9f}",
        "max_time_s": f"{stats['max']:.9f}",
        "runs": int(stats["runs"]),
        "peak_kib": "" if peak == "" else f"{peak:.1f}",
        "theoretical_time": theory_time,
        "theoretical_space": theory_space,
        "speedup_vs_baseline": "",
        "baseline_variant": "",
        "measurement": "measured",
    }


def report(row: Dict[str, Any]) -> None:
    peak = row["peak_kib"] if row["peak_kib"] != "" else "-"
    secondary = row["secondary_param"] if row["secondary_param"] != "" else ""
    print(f"    {row['problem']:<15} {row['variant']:<20} n={row['n']:<5} "
          f"{str(secondary):<6} {float(row['mean_time_s']) * 1e3:12.3f} ms  "
          f"runs={row['runs']}  peak={peak} KiB", flush=True)


def add_speedups(rows: List[Dict[str, Any]]) -> None:
    """Fill speedup_vs_baseline wherever the baseline ran at the same size.

    Blank otherwise. A ratio is only written when both of its terms were
    produced by this run.
    """
    index = {(r["problem"], r["variant"], r["n"], r["secondary_param"]): r for r in rows}
    for row in rows:
        baseline = BASELINE.get((row["problem"], row["variant"]))
        if baseline is None:
            continue
        base = index.get((row["problem"], baseline, row["n"], row["secondary_param"]))
        if base is None:
            continue
        row["speedup_vs_baseline"] = (
            f"{float(base['mean_time_s']) / float(row['mean_time_s']):.3f}")
        row["baseline_variant"] = baseline


# ----------------------------------------------------------------------
# Study 1: knapsack
# ----------------------------------------------------------------------
def knapsack_instance(n: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    return ([rng.randint(1, 50) for _ in range(n)],
            [rng.randint(1, 100) for _ in range(n)])


def knapsack_study(config: Config) -> List[Dict[str, Any]]:
    print("\n=== STUDY 1 - knapsack: Week 5's 2D table against one row ===", flush=True)
    rows: List[Dict[str, Any]] = []
    cells: List[Tuple[int, int]] = [(config.knapsack_n, w) for w in config.knapsack_capacities]
    for n in config.knapsack_sizes:
        if (n, config.knapsack_capacity) not in cells:
            cells.append((n, config.knapsack_capacity))
        else:
            print(f"    (n={n}, W={config.knapsack_capacity} already in the capacity sweep)",
                  flush=True)
    for n, capacity in cells:
        weights, values = knapsack_instance(n, SEED + n)
        assert knapsack_tab(weights, values, capacity) == knapsack_space_optimized(
            weights, values, capacity), f"knapsack variants disagree at n={n}, W={capacity}"
        for variant, function in (("standard_2d", knapsack_tab),
                                  ("space_optimized_1d", knapsack_space_optimized)):
            call = (lambda f=function, w=weights, v=values, c=capacity: f(w, v, c))
            stats = measure(call)
            row = make_row("knapsack", variant, n, capacity, stats, trace(call, stats["mean"]))
            rows.append(row)
            report(row)
    return rows


# ----------------------------------------------------------------------
# Study 2: matrix chain multiplication
# ----------------------------------------------------------------------
def mcm_study(config: Config) -> List[Dict[str, Any]]:
    print("\n=== STUDY 2 - matrix chain: recursion against DP ===", flush=True)
    rows: List[Dict[str, Any]] = []
    for n in sorted(set(config.mcm_dp_sizes) | set(config.mcm_recursive_sizes)):
        p = random_dimensions(n, seed=SEED + n)
        expected = mcm_memoized(p)
        m, _ = mcm_bottom_up(p)
        assert m[1][n] == expected, f"MCM variants disagree at n={n}"
        variants: List[Tuple[str, Callable[[], Any]]] = []
        if n in config.mcm_recursive_sizes:
            assert mcm_recursive(p) == expected, f"recursive MCM disagrees at n={n}"
            variants.append(("recursive", lambda q=p: mcm_recursive(q)))
        if n in config.mcm_dp_sizes:
            variants.append(("memoized", lambda q=p: mcm_memoized(q)))
            variants.append(("bottom_up", lambda q=p: mcm_bottom_up(q)))
        for variant, call in variants:
            stats = measure(call)
            row = make_row("mcm", variant, n, "", stats, trace(call, stats["mean"]))
            rows.append(row)
            report(row)
    return rows


# ----------------------------------------------------------------------
# Study 3: Floyd-Warshall
# ----------------------------------------------------------------------
def fw_graph(n: int, density: float) -> Graph:
    return weighted_graph(n, density=density, weight_range=(1, 100), directed=True,
                          connected=True, seed=SEED + n + int(density * 1000))


def floyd_warshall_study(config: Config) -> Tuple[List[Dict[str, Any]], List[dict]]:
    print("\n=== STUDY 3 - Floyd-Warshall ===", flush=True)
    rows: List[Dict[str, Any]] = []
    scaling_density = max(config.densities)
    plan: List[Tuple[int, float]] = []
    for density in config.densities:
        for n in config.fw_compare_sizes:
            plan.append((n, density))
    for n in config.fw_scaling_sizes:
        if (n, scaling_density) not in plan:
            plan.append((n, scaling_density))
    for n in config.fw_3d_sizes:
        if (n, scaling_density) not in plan:
            plan.append((n, scaling_density))
    plan.sort(key=lambda item: (item[1], item[0]))

    for n, density in plan:
        graph = fw_graph(n, density)
        matrix, _ = graph_to_weight_matrix(graph)
        compare = n in config.fw_compare_sizes
        if compare:
            # The two answers must agree before either time means anything.
            dist, _ = floyd_warshall(matrix)
            assert matrices_close(dist, all_pairs_dijkstra(graph)), (
                f"Floyd-Warshall and all-pairs Dijkstra disagree at n={n}, d={density}")
        fw_call = (lambda w=matrix: floyd_warshall(w))
        stats = measure(fw_call)
        row = make_row("floyd_warshall", "floyd_warshall", n, density, stats,
                       trace(fw_call, stats["mean"]))
        rows.append(row)
        report(row)
        if compare:
            dij_call = (lambda g=graph: all_pairs_dijkstra(g))
            stats = measure(dij_call)
            row = make_row("floyd_warshall", "all_pairs_dijkstra", n, density, stats,
                           trace(dij_call, stats["mean"]))
            rows.append(row)
            report(row)
        if n in config.fw_3d_sizes and density == scaling_density:
            layers = floyd_warshall_3d(matrix)
            assert matrices_close(layers[-1], floyd_warshall(matrix)[0]), (
                f"3D and 2D Floyd-Warshall disagree at n={n}")
            del layers
            call_3d = (lambda w=matrix: floyd_warshall_3d(w))
            stats = measure(call_3d)
            row = make_row("floyd_warshall", "floyd_warshall_3d", n, density, stats,
                           trace(call_3d, stats["mean"]))
            rows.append(row)
            report(row)
        del graph, matrix
        gc.collect()

    print("  --- negative edges, no negative cycle (the CLRS all-pairs example) ---",
          flush=True)
    negative = negative_edge_run()
    return rows, negative


CLRS_EDGES = [(0, 1, 3), (0, 2, 8), (0, 4, -4), (1, 3, 1), (1, 4, 7),
              (2, 1, 4), (3, 0, 2), (3, 2, -5), (4, 3, 6)]


def negative_edge_run() -> List[dict]:
    """Floyd-Warshall answers; Week 4's Dijkstra refuses. Both recorded."""
    graph = Graph(directed=True, weighted=True)
    for v in range(5):
        graph.add_node(v)
    for u, v, w in CLRS_EDGES:
        graph.add_edge(u, v, w)
    dist, _ = floyd_warshall(graph)
    try:
        all_pairs_dijkstra(graph)
        dijkstra_outcome = "ran"
    except ValueError as error:
        dijkstra_outcome = f"ValueError: {error}"
    print(f"    Floyd-Warshall row 0: {dist[0]}", flush=True)
    print(f"    Week 4 Dijkstra: {dijkstra_outcome}", flush=True)
    return [{"source": i, "floyd_warshall_row": " ".join(f"{d:g}" for d in dist[i]),
             "all_pairs_dijkstra": dijkstra_outcome} for i in range(5)]


# ----------------------------------------------------------------------
# Study 4: TSP
# ----------------------------------------------------------------------
def tsp_study(config: Config) -> List[Dict[str, Any]]:
    print("\n=== STUDY 4 - TSP: Held-Karp bitmask against brute force ===", flush=True)
    rows: List[Dict[str, Any]] = []
    for n in sorted(set(config.tsp_bitmask_sizes) | set(config.tsp_brute_sizes)):
        dist = random_weight_matrix(n, 1.0, weight_range=(1, 100), seed=SEED + n)
        variants: List[Tuple[str, Callable[[], Any]]] = []
        if n in config.tsp_bitmask_sizes:
            variants.append(("bitmask", lambda d=dist: tsp_bitmask(d)))
        if n in config.tsp_brute_sizes:
            assert tsp_bitmask(dist)[0] == tsp_brute_force(dist)[0], (
                f"bitmask and brute force disagree at n={n}")
            variants.append(("brute_force", lambda d=dist: tsp_brute_force(d)))
        for variant, call in variants:
            stats = measure(call)
            row = make_row("tsp", variant, n, "", stats, trace(call, stats["mean"]))
            rows.append(row)
            report(row)
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: List[dict], columns: Optional[List[str]] = None) -> None:
    if columns is None:
        columns = []
        for row in rows:
            for name in row:
                if name not in columns:
                    columns.append(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", "--quick", dest="smoke", action="store_true",
                        help="tiny parameter set that runs in seconds")
    parser.add_argument("--out", default=RESULTS_DIR,
                        help="directory for the CSVs and figures")
    args = parser.parse_args(argv)

    config = SMOKE if args.smoke else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    apply_house_style()

    print("=" * 78)
    print("CSC 5300 Week 6 - dynamic programming II benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python   : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform : {platform.platform()}")
    print(f"  mode     : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  seed     : {SEED}")
    print(f"  output   : {out}")

    started = time.perf_counter()
    rows = knapsack_study(config)
    rows += mcm_study(config)
    fw_rows, negative = floyd_warshall_study(config)
    rows += fw_rows
    rows += tsp_study(config)
    add_speedups(rows)

    print("\n=== Part 1 output: the knapsack comparison the instructions ask for ===")
    weights, values = knapsack_instance(config.knapsack_n, SEED + config.knapsack_n)
    print_comparison(compare_with_standard(weights, values, config.knapsack_capacity,
                                           repeat=3 if args.smoke else 5))

    print("\n=== writing results ===")
    write_csv(os.path.join(out, "comparison_table.csv"), rows, COLUMNS)
    write_csv(os.path.join(out, "week6_negative_edges.csv"), negative)

    print("\n=== figures ===")
    clrs = [30, 35, 15, 5, 10, 20, 25]
    m, s = mcm_bottom_up(clrs)
    for name, builder in (
        ("knapsack_space_comparison.png",
         lambda p: plot_knapsack_space_comparison(rows, save_path=p,
                                                  fixed_n=config.knapsack_n)),
        ("mcm_performance.png", lambda p: plot_mcm_performance(rows, save_path=p)),
        ("floyd_warshall_scaling.png",
         lambda p: plot_floyd_warshall_scaling(rows, save_path=p,
                                               scaling_density=max(config.densities))),
        ("tsp_bitmask_runtime.png", lambda p: plot_tsp_runtime(rows, save_path=p)),
        ("mcm_dp_table.png", lambda p: plot_mcm_table(m, s, save_path=p)),
    ):
        path = os.path.join(out, name)
        builder(path)
        print(f"  wrote {os.path.relpath(path, REPO_ROOT)} "
              f"({os.path.getsize(path) / 1024:.0f} KiB)")

    cost, paren = matrix_chain_order(clrs)
    print(f"\n  CLRS chain {clrs}: {cost} multiplications, {paren}")
    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
