#!/usr/bin/env python3
"""Week 4 benchmark: graph representations, traversals and shortest paths.

Run from the repository root with the project virtualenv active::

    python benchmarks/week4_graph_benchmark.py            # the full study
    python benchmarks/week4_graph_benchmark.py --quick    # smoke run
    python benchmarks/week4_graph_benchmark.py --quick --out /tmp/w4

Four studies, all timed through
:meth:`src.utils.benchmark.AlgorithmBenchmark.time_operation`, the Week 1
framework extended in Week 3 for structure operations:

1. **Representation** - adjacency list against adjacency matrix at
   n = 100, 1,000 and 10,000: memory held and the cost of a single edge
   lookup. This is the trade the rest of the week rests on, because every
   traversal below reads neighbours out of the list.
2. **Traversal** - BFS against iterative and recursive DFS, on sparse
   graphs (average degree 4) and dense ones (density 0.5). Reported per
   (V + E) unit as well as in total, because O(V + E) is a claim about
   that ratio being flat, not about the total being flat.
3. **Shortest paths** - Dijkstra driven by the Week 3 binary heap against
   the same algorithm driven by a linear scan of a plain list. This is
   O((V + E) log V) against O(V^2) with every other line of code held
   identical.
4. **Traversal figures** - BFS and DFS visit order drawn on one small
   graph, through networkx.

Runtime and memory are bounded deliberately rather than by accident, and
every reduction is recorded in the CSV and stated in the report:

* The matrix at n = 10,000 is 10^8 cells. Backed by numpy at ``uint8``
  that is 100 MB, which is survivable; anything projected past
  ``MATRIX_BYTE_CAP`` is skipped and recorded as skipped.
* The list-scan Dijkstra is O(V^2) and is capped at
  ``Config.linear_dijkstra_cap`` nodes. The heap version runs the full
  range, so the two series deliberately cover different ranges.
* Dense traversal graphs stop well below the sparse sizes, because
  density 0.5 at n = 10,000 is 2.5 x 10^7 edges.

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
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.graphs.bfs import bfs  # noqa: E402
from src.graphs.dfs import dfs_iterative, dfs_recursive  # noqa: E402
from src.graphs.dijkstra import dijkstra, dijkstra_linear_scan  # noqa: E402
from src.utils.benchmark import AlgorithmBenchmark, BenchmarkResult  # noqa: E402
from src.utils.graph_generator import dense_graph, sparse_graph  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_bfs_vs_dfs,
    plot_traversal_order,
)

#: Base seed. Every generated graph derives from it, so a re-run reproduces
#: the same graphs and the same numbers.
SEED = 42

#: Refuse to allocate a matrix larger than this. 10,000^2 uint8 is 100 MB and
#: passes; the cap is what turns "too big" into a recorded reduction rather
#: than a machine that swaps for ten minutes.
MATRIX_BYTE_CAP = 512 * 1024 * 1024

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

CLASSES = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)")


@dataclass
class Config:
    """Every size, repetition count and cap the study uses, in one place."""

    repr_sizes: List[int]
    sparse_sizes: List[int]
    dense_sizes: List[int]
    dijkstra_sizes: List[int]
    runs: Dict[int, int]
    warmups: Dict[int, int]
    #: A list-scan Dijkstra is O(V^2); past this many nodes it dominates the
    #: whole suite's runtime for a point already made.
    linear_dijkstra_cap: int = 2000
    avg_degree: int = 4
    dense_density: float = 0.5
    #: Edge lookups timed per run in the representation study.
    lookups: int = 20_000
    #: Nodes in the drawn traversal figures. The assignment asks for 15-25.
    figure_nodes: int = 20


FULL = Config(
    repr_sizes=[100, 1_000, 10_000],
    sparse_sizes=[100, 500, 1_000, 5_000, 10_000],
    dense_sizes=[50, 100, 200, 400, 800],
    dijkstra_sizes=[100, 250, 500, 1_000, 2_000, 4_000, 8_000],
    runs={100: 5, 250: 5, 500: 5, 800: 3, 1_000: 5, 2_000: 3, 4_000: 3, 5_000: 3,
          8_000: 2, 10_000: 2, 50: 5, 200: 5, 400: 5},
    warmups={100: 2, 250: 2, 500: 2, 800: 1, 1_000: 2, 2_000: 1, 4_000: 1, 5_000: 1,
             8_000: 1, 10_000: 1, 50: 2, 200: 2, 400: 2},
)

QUICK = Config(
    repr_sizes=[100, 1_000],
    sparse_sizes=[100, 500],
    dense_sizes=[50, 100],
    dijkstra_sizes=[100, 250],
    runs={50: 2, 100: 2, 250: 2, 500: 2, 1_000: 2},
    warmups={50: 1, 100: 1, 250: 1, 500: 1, 1_000: 1},
    lookups=2_000,
    linear_dijkstra_cap=250,
)


def runs_for(config: Config, n: int) -> Tuple[int, int]:
    """Measured runs and discarded warm-ups for size ``n``, with a default."""
    return config.runs.get(n, 3), config.warmups.get(n, 1)


# ----------------------------------------------------------------------
# Study 1: representation
# ----------------------------------------------------------------------
def adjacency_list_bytes(graph: Any) -> int:
    """Container bytes an adjacency list holds for ``graph``.

    :func:`sys.getsizeof` is shallow, so the outer mapping and every inner
    mapping are walked. The node and weight objects themselves are left
    out on purpose: both representations refer to the same interned
    integers, so counting them would inflate both sides and compare
    nothing. What is measured is the cost of the structure, which is what
    differs.

    The mirror is built through the public API rather than by reaching
    into the graph, and has the same shape as the internal store.
    """
    mirror = {node: dict(graph.neighbor_items(node)) for node in graph.nodes()}
    total = sys.getsizeof(mirror)
    for neighbours in mirror.values():
        total += sys.getsizeof(neighbours)
    return total


def lookup_pairs(graph: Any, count: int, rng: Any) -> List[Tuple[Any, Any]]:
    """Half present edges and half absent pairs, interleaved and shuffled.

    A lookup benchmark that only asks about edges that exist measures the
    lucky half of the problem. Both representations answer "no" by a
    different route: the matrix reads one cell either way, the adjacency
    list misses a dict.
    """
    nodes = graph.nodes()
    edges = graph.edges()
    pairs: List[Tuple[Any, Any]] = []
    for index in range(count):
        if edges and index % 2 == 0:
            edge = edges[rng.randrange(len(edges))]
            pairs.append((edge[0], edge[1]))
        else:
            pairs.append((nodes[rng.randrange(len(nodes))],
                          nodes[rng.randrange(len(nodes))]))
    return pairs


def representation_study(bench: AlgorithmBenchmark, config: Config) -> List[dict]:
    """Memory and single-edge-lookup cost, adjacency list against matrix."""
    import random

    print("\n=== STUDY 1 - representation: list against matrix ===", flush=True)
    rows: List[dict] = []
    for n in config.repr_sizes:
        runs, warmups = runs_for(config, n)
        graph = sparse_graph(n, avg_degree=config.avg_degree, connected=True, seed=SEED)
        rng = random.Random(SEED + n)
        pairs = lookup_pairs(graph, config.lookups, rng)
        list_bytes = adjacency_list_bytes(graph)
        projected = n * n  # one uint8 cell per pair of nodes
        edges = graph.edge_count

        def list_lookup(target: Any, pairs: List[Tuple[Any, Any]] = pairs) -> int:
            has_edge = target.has_edge
            found = 0
            for u, v in pairs:
                if has_edge(u, v):
                    found += 1
            return found

        list_result = bench.time_operation(
            list_lookup, lambda g=graph: g, runs=runs, input_size=n,
            ops_per_run=len(pairs), name=f"adjacency list | lookup | n={n}",
            warmup_runs=warmups,
        )

        row: Dict[str, Any] = {
            "n": n,
            "edges": edges,
            "density": round(graph.density(), 6),
            "list_bytes": list_bytes,
            "list_lookup_ns": round(per_op_ns(list_result), 2),
            "runs": runs,
        }

        if projected > MATRIX_BYTE_CAP:
            # Reduce and record. Never drop a required point in silence.
            row.update({
                "matrix_bytes": "",
                "matrix_lookup_ns": "",
                "matrix_build_s": "",
                "matrix_status": f"skipped: {projected / 1e6:.0f} MB exceeds the "
                                 f"{MATRIX_BYTE_CAP / 1e6:.0f} MB cap",
            })
            print(f"    n={n:>6,}  matrix SKIPPED ({projected / 1e6:.0f} MB > cap)",
                  flush=True)
        else:
            started = time.perf_counter()
            dense = graph.to_adjacency_matrix()
            build_seconds = time.perf_counter() - started
            matrix, index = dense.matrix, dense.index
            index_pairs = [(index[u], index[v]) for u, v in pairs]

            def matrix_lookup(target: np.ndarray,
                              pairs: List[Tuple[int, int]] = index_pairs) -> int:
                found = 0
                for i, j in pairs:
                    if target[i, j]:
                        found += 1
                return found

            matrix_result = bench.time_operation(
                matrix_lookup, lambda m=matrix: m, runs=runs, input_size=n,
                ops_per_run=len(pairs), name=f"adjacency matrix | lookup | n={n}",
                warmup_runs=warmups,
            )
            row.update({
                "matrix_bytes": int(matrix.nbytes),
                "matrix_lookup_ns": round(per_op_ns(matrix_result), 2),
                "matrix_build_s": round(build_seconds, 4),
                "matrix_status": "measured",
            })
            print(f"    n={n:>6,}  list {list_bytes / 1e6:8.3f} MB "
                  f"{row['list_lookup_ns']:7.1f} ns/lookup | matrix "
                  f"{matrix.nbytes / 1e6:8.3f} MB {row['matrix_lookup_ns']:7.1f} "
                  f"ns/lookup  (build {build_seconds:.3f} s)", flush=True)
            del dense, matrix
        rows.append(row)
        del graph
        gc.collect()
    return rows


# ----------------------------------------------------------------------
# Study 2: traversal
# ----------------------------------------------------------------------
TRAVERSALS: List[Tuple[str, Callable[[Any], List[Any]]]] = [
    ("BFS", bfs),
    ("DFS (iterative)", dfs_iterative),
    ("DFS (recursive)", dfs_recursive),
]


def traversal_study(bench: AlgorithmBenchmark, config: Config, kind: str) -> List[dict]:
    """BFS against both DFS forms, on one density of graph."""
    sizes = config.sparse_sizes if kind == "sparse" else config.dense_sizes
    print(f"\n=== STUDY 2{'a' if kind == 'sparse' else 'b'} - traversal on "
          f"{kind} graphs ===", flush=True)
    rows: List[dict] = []
    for n in sizes:
        runs, warmups = runs_for(config, n)
        if kind == "sparse":
            graph = sparse_graph(n, avg_degree=config.avg_degree, connected=True,
                                 seed=SEED)
        else:
            graph = dense_graph(n, density=config.dense_density, connected=True,
                                seed=SEED)
        work = graph.node_count + graph.edge_count

        # Correctness, off the clock: every traversal must reach every node
        # exactly once, and the two DFS forms must agree exactly.
        orders = {name: function(graph) for name, function in TRAVERSALS}
        for name, order in orders.items():
            assert len(order) == graph.node_count, f"{name} missed nodes at n={n}"
            assert set(order) == set(graph.nodes()), f"{name} wrong node set at n={n}"
        assert orders["DFS (iterative)"] == orders["DFS (recursive)"], (
            f"the two DFS forms disagree at n={n}")

        for name, function in TRAVERSALS:
            result = bench.time_operation(
                function, lambda g=graph: g, runs=runs, input_size=n,
                ops_per_run=work, name=f"{name} | {kind} | n={n}",
                warmup_runs=warmups,
            )
            rows.append({
                "kind": kind,
                "algorithm": name,
                "n": n,
                "nodes": graph.node_count,
                "edges": graph.edge_count,
                "work_v_plus_e": work,
                "seconds": round(result.average_time, 6),
                "std_seconds": round(result.std_deviation, 6),
                "ns_per_v_plus_e": round(per_op_ns(result), 3),
                "runs": runs,
            })
            print(f"    {name:<16} n={n:>6,}  V+E={work:>9,}  "
                  f"{result.average_time * 1e3:9.3f} ms  "
                  f"{per_op_ns(result):7.1f} ns per V+E", flush=True)
        del graph, orders
        gc.collect()
    return rows


# ----------------------------------------------------------------------
# Study 3: Dijkstra, heap against linear scan
# ----------------------------------------------------------------------
def dijkstra_study(bench: AlgorithmBenchmark, config: Config) -> List[dict]:
    """The Week 3 heap against an O(V^2) linear scan, same algorithm either way."""
    print("\n=== STUDY 3 - Dijkstra: week 3 heap against a linear scan ===",
          flush=True)
    rows: List[dict] = []
    for n in config.dijkstra_sizes:
        runs, warmups = runs_for(config, n)
        graph = sparse_graph(n, avg_degree=config.avg_degree, weighted=True,
                             connected=True, seed=SEED)
        source = graph.nodes()[0]
        work = graph.node_count + graph.edge_count

        implementations: List[Tuple[str, Callable[[Any], Any]]] = [
            ("heap priority queue", dijkstra),
        ]
        if n <= config.linear_dijkstra_cap:
            implementations.append(("linear scan", dijkstra_linear_scan))
            # Off the clock: the two must agree, or the comparison is empty.
            heap_distances, _ = dijkstra(graph, source)
            scan_distances, _ = dijkstra_linear_scan(graph, source)
            assert heap_distances == scan_distances, f"implementations differ at n={n}"

        for name, function in implementations:
            result = bench.time_operation(
                lambda g, f=function, s=source: f(g, s),
                lambda g=graph: g, runs=runs, input_size=n, ops_per_run=work,
                name=f"dijkstra | {name} | n={n}", warmup_runs=warmups,
            )
            rows.append({
                "implementation": name,
                "n": n,
                "nodes": graph.node_count,
                "edges": graph.edge_count,
                "work_v_plus_e": work,
                "seconds": round(result.average_time, 6),
                "std_seconds": round(result.std_deviation, 6),
                "ns_per_v_plus_e": round(per_op_ns(result), 3),
                "runs": runs,
                "note": "" if n <= config.linear_dijkstra_cap or name != "linear scan"
                        else "capped",
            })
            print(f"    {name:<20} V={n:>6,}  E={graph.edge_count:>7,}  "
                  f"{result.average_time * 1e3:9.3f} ms", flush=True)
        if n > config.linear_dijkstra_cap:
            print(f"    linear scan          V={n:>6,}  skipped: past the "
                  f"{config.linear_dijkstra_cap:,}-node cap (O(V^2))", flush=True)
        del graph
        gc.collect()
    return rows


# ----------------------------------------------------------------------
# Comparison table: asymptotic against empirical
# ----------------------------------------------------------------------
def predicted_growth(klass: str, low: int, high: int) -> float:
    """Growth a class predicts between two sizes."""
    ratio = high / low
    if klass == "O(1)":
        return 1.0
    if klass == "O(log n)":
        return math.log(high) / math.log(low)
    if klass == "O(n)":
        return ratio
    if klass == "O(n log n)":
        return ratio * math.log(high) / math.log(low)
    return ratio ** 2


def classify(growth: float, low: int, high: int) -> str:
    """Nearest class to a measured growth ratio, compared in log space."""
    return min(
        CLASSES,
        key=lambda k: abs(math.log(growth) - math.log(predicted_growth(k, low, high))),
    )


def comparison_rows(traversal: List[dict], dijkstra_rows: List[dict],
                    representation: List[dict]) -> List[dict]:
    """One row per measured series: the bound, the measurement, the verdict."""
    rows: List[dict] = []

    def add(structure: str, operation: str, asymptotic: str, expected: str,
            sizes: List[int], values: List[float], runs: List[int],
            unit: str, note: str = "") -> None:
        if len(sizes) < 2:
            return
        slope = float(np.polyfit(np.log(sizes), np.log(values), 1)[0])
        growth = values[-1] / values[0]
        low, high = sizes[0], sizes[-1]
        empirical = classify(growth, low, high)
        rows.append({
            "structure": structure,
            "operation": operation,
            "asymptotic": asymptotic,
            "expected_growth_class": expected,
            "unit": unit,
            "n_smallest": low,
            "value_smallest": round(values[0], 4),
            "n_largest": high,
            "value_largest": round(values[-1], 4),
            "runs_smallest": runs[0],
            "runs_largest": runs[-1],
            "loglog_slope": round(slope, 4),
            "growth_measured": round(growth, 3),
            "growth_predicted": round(predicted_growth(expected, low, high), 3),
            "empirical_class": empirical,
            "agrees": empirical == expected,
            "note": note,
        })

    for kind in ("sparse", "dense"):
        for name, _ in TRAVERSALS:
            picked = [r for r in traversal if r["kind"] == kind
                      and r["algorithm"] == name]
            picked.sort(key=lambda r: r["n"])
            if not picked:
                continue
            # On a sparse graph E is O(V), so O(V + E) reads as linear in n.
            # On a dense graph E is O(V^2), so the same bound reads quadratic.
            expected = "O(n)" if kind == "sparse" else "O(n^2)"
            add(f"{name} ({kind})", "traverse every node", "O(V + E)", expected,
                [r["n"] for r in picked], [r["seconds"] for r in picked],
                [r["runs"] for r in picked], "seconds",
                "E ~ 2V" if kind == "sparse" else "E ~ V^2 / 4")
            add(f"{name} ({kind})", "per V+E unit", "O(V + E)", "O(1)",
                [r["n"] for r in picked], [r["ns_per_v_plus_e"] for r in picked],
                [r["runs"] for r in picked], "ns per V+E",
                "flat here is what O(V + E) means")

    for implementation in ("heap priority queue", "linear scan"):
        picked = [r for r in dijkstra_rows if r["implementation"] == implementation]
        picked.sort(key=lambda r: r["n"])
        if not picked:
            continue
        asymptotic = ("O((V + E) log V)" if implementation == "heap priority queue"
                      else "O(V^2)")
        expected = "O(n log n)" if implementation == "heap priority queue" else "O(n^2)"
        note = ("" if implementation == "heap priority queue"
                else "capped, so this series covers a shorter range")
        add(f"Dijkstra ({implementation})", "single source shortest paths",
            asymptotic, expected, [r["n"] for r in picked],
            [r["seconds"] for r in picked], [r["runs"] for r in picked],
            "seconds", note)

    measured = [r for r in representation if r["matrix_status"] == "measured"]
    if len(measured) >= 2:
        sizes = [r["n"] for r in measured]
        runs = [r["runs"] for r in measured]
        add("Adjacency list", "memory held", "O(V + E)", "O(n)", sizes,
            [float(r["list_bytes"]) for r in measured], runs, "bytes")
        add("Adjacency matrix", "memory held", "O(V^2)", "O(n^2)", sizes,
            [float(r["matrix_bytes"]) for r in measured], runs, "bytes")
        add("Adjacency list", "single edge lookup", "O(1) average", "O(1)", sizes,
            [float(r["list_lookup_ns"]) for r in measured], runs, "ns")
        add("Adjacency matrix", "single edge lookup", "O(1)", "O(1)", sizes,
            [float(r["matrix_lookup_ns"]) for r in measured], runs, "ns")
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def per_op_ns(result: BenchmarkResult) -> float:
    return result.metadata["per_op_time"] * 1e9


def write_csv(path: str, rows: List[dict]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")


COLORS = plt.get_cmap("tab10").colors
MARKERS = ["o", "s", "^", "D", "v", "P"]


def finish(ax, title: str, xlabel: str, ylabel: str, log: bool = True) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} "
          f"({os.path.getsize(path) / 1024:.0f} KiB)")


def plot_traversal(rows: List[dict], kind: str, path: str) -> None:
    """Total time and time per (V + E) unit, for one density."""
    picked = [r for r in rows if r["kind"] == kind]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    for index, (name, _) in enumerate(TRAVERSALS):
        series = sorted([r for r in picked if r["algorithm"] == name],
                        key=lambda r: r["n"])
        if not series:
            continue
        sizes = [r["n"] for r in series]
        left.errorbar(sizes, [r["seconds"] * 1e3 for r in series],
                      yerr=[r["std_seconds"] * 1e3 for r in series],
                      marker=MARKERS[index], color=COLORS[index], label=name,
                      capsize=3, linewidth=1.5)
        right.plot(sizes, [r["ns_per_v_plus_e"] for r in series],
                   marker=MARKERS[index], color=COLORS[index], label=name,
                   linewidth=1.5)
    if picked:
        sizes = sorted({r["n"] for r in picked})
        first = min(picked, key=lambda r: r["n"])
        scale = first["seconds"] * 1e3 / first["work_v_plus_e"]
        work_by_size = {r["n"]: r["work_v_plus_e"] for r in picked}
        left.plot(sizes, [scale * work_by_size[n] for n in sizes], "k--",
                  linewidth=1.1, alpha=0.7, label="O(V + E) reference")
    density = "average degree 4" if kind == "sparse" else "density 0.5"
    finish(left, f"Total traversal time, {kind} graphs ({density})",
           "Nodes V", "Milliseconds per traversal")
    finish(right, "Time per (V + E) unit: flat means O(V + E)",
           "Nodes V", "Nanoseconds per V+E", log=False)
    fig.suptitle(f"BFS against DFS on {kind} graphs", fontsize=12)
    save(fig, path)


def plot_dijkstra(rows: List[dict], path: str) -> None:
    """Heap-backed Dijkstra against the O(V^2) linear scan."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    for index, implementation in enumerate(("heap priority queue", "linear scan")):
        series = sorted([r for r in rows if r["implementation"] == implementation],
                        key=lambda r: r["n"])
        if not series:
            continue
        left.errorbar([r["n"] for r in series], [r["seconds"] * 1e3 for r in series],
                      yerr=[r["std_seconds"] * 1e3 for r in series],
                      marker=MARKERS[index], color=COLORS[index],
                      label=f"{implementation} ({len(series)} sizes)", capsize=3,
                      linewidth=1.5)
    heap_rows = sorted([r for r in rows if r["implementation"] == "heap priority queue"],
                       key=lambda r: r["n"])
    scan_rows = sorted([r for r in rows if r["implementation"] == "linear scan"],
                       key=lambda r: r["n"])
    if heap_rows:
        sizes = [r["n"] for r in heap_rows]
        base = heap_rows[0]
        unit = base["seconds"] * 1e3 / (base["work_v_plus_e"] * math.log2(base["n"]))
        work = {r["n"]: r["work_v_plus_e"] for r in heap_rows}
        left.plot(sizes, [unit * work[n] * math.log2(n) for n in sizes], "k--",
                  linewidth=1.1, alpha=0.7, label="O((V + E) log V) reference")
    if scan_rows:
        sizes = [r["n"] for r in scan_rows]
        unit = scan_rows[0]["seconds"] * 1e3 / (scan_rows[0]["n"] ** 2)
        left.plot(sizes, [unit * n * n for n in sizes], ":", color="0.35",
                  linewidth=1.1, label="O(V^2) reference")
    shared = {r["n"]: r for r in scan_rows}
    overlap = [r for r in heap_rows if r["n"] in shared]
    if overlap:
        right.plot([r["n"] for r in overlap],
                   [shared[r["n"]]["seconds"] / r["seconds"] for r in overlap],
                   marker="o", color=COLORS[3], linewidth=1.5,
                   label="linear scan / heap")
        right.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    finish(left, "Dijkstra: the priority queue is the whole difference",
           "Nodes V", "Milliseconds per run")
    finish(right, "How many times slower the linear scan is",
           "Nodes V", "Ratio", log=False)
    fig.suptitle("Single-source shortest paths, sparse weighted graphs", fontsize=12)
    save(fig, path)


def plot_representation(rows: List[dict], path: str) -> None:
    """Memory held and lookup cost, adjacency list against matrix."""
    measured = [r for r in rows if r["matrix_status"] == "measured"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    sizes = [r["n"] for r in rows]
    left.plot(sizes, [r["list_bytes"] / 1e6 for r in rows], marker="o",
              color=COLORS[0], label="adjacency list", linewidth=1.5)
    if measured:
        left.plot([r["n"] for r in measured],
                  [r["matrix_bytes"] / 1e6 for r in measured], marker="s",
                  color=COLORS[1], label="adjacency matrix (numpy uint8)",
                  linewidth=1.5)
    finish(left, "Memory held by each representation", "Nodes V", "Megabytes")
    width = 0.38
    positions = np.arange(len(measured))
    right.bar(positions - width / 2, [r["list_lookup_ns"] for r in measured],
              width, color=COLORS[0], label="adjacency list")
    right.bar(positions + width / 2, [r["matrix_lookup_ns"] for r in measured],
              width, color=COLORS[1], label="adjacency matrix")
    right.set_xticks(positions)
    right.set_xticklabels([f"{r['n']:,}" for r in measured])
    finish(right, "One edge lookup, half present and half absent",
           "Nodes V", "Nanoseconds per lookup", log=False)
    fig.suptitle("Adjacency list against adjacency matrix", fontsize=12)
    save(fig, path)


def traversal_figures(config: Config, out: str) -> None:
    """Draw BFS and DFS visit order on one small graph."""
    print("\n=== STUDY 4 - traversal figures ===", flush=True)
    graph = sparse_graph(config.figure_nodes, avg_degree=3, connected=True, seed=SEED)
    order_bfs = bfs(graph)
    order_dfs = dfs_iterative(graph)
    plot_traversal_order(graph, order_bfs, "BFS visit order",
                         save_path=os.path.join(out, "bfs_traversal.png"))
    print(f"  wrote {os.path.join('benchmarks', 'results', 'bfs_traversal.png')}")
    plot_traversal_order(graph, order_dfs, "DFS visit order",
                         save_path=os.path.join(out, "dfs_traversal.png"))
    print(f"  wrote {os.path.join('benchmarks', 'results', 'dfs_traversal.png')}")
    plot_bfs_vs_dfs(graph, order_bfs, order_dfs,
                    save_path=os.path.join(out, "bfs_vs_dfs_traversal.png"))
    print(f"  wrote {os.path.join('benchmarks', 'results', 'bfs_vs_dfs_traversal.png')}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small, fast smoke run")
    parser.add_argument("--out", default=RESULTS_DIR,
                        help="directory for CSVs and charts")
    args = parser.parse_args(argv)

    config = QUICK if args.quick else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    apply_house_style()

    print("=" * 78)
    print("CSC 5300 Week 4 - graph algorithms benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python   : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform : {platform.platform()}")
    print(f"  mode     : {'QUICK SMOKE RUN' if args.quick else 'full study'}")
    print(f"  seed     : {SEED}")
    print(f"  output   : {out}")

    started = time.perf_counter()
    bench = AlgorithmBenchmark(seed=SEED)

    representation = representation_study(bench, config)
    traversal = traversal_study(bench, config, "sparse")
    traversal += traversal_study(bench, config, "dense")
    shortest = dijkstra_study(bench, config)
    traversal_figures(config, out)

    print("\n=== writing results ===")
    write_csv(os.path.join(out, "graphs_comparison_table.csv"),
              comparison_rows(traversal, shortest, representation))
    write_csv(os.path.join(out, "week4_representation.csv"), representation)
    write_csv(os.path.join(out, "week4_traversal.csv"), traversal)
    write_csv(os.path.join(out, "week4_dijkstra.csv"), shortest)

    print("\n=== charts ===")
    plot_traversal(traversal, "sparse", os.path.join(out, "bfs_vs_dfs_sparse.png"))
    plot_traversal(traversal, "dense", os.path.join(out, "bfs_vs_dfs_dense.png"))
    plot_dijkstra(shortest, os.path.join(out, "dijkstra_performance.png"))
    plot_representation(representation, os.path.join(out, "graph_representation.png"))

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
