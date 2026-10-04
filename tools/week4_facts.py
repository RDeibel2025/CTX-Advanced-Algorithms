#!/usr/bin/env python3
"""Print every figure analysis/week4_report.md quotes, computed from the CSVs.

The report is prose under a word limit, so it cannot be generated whole.
This is the compromise Weeks 2 and 3 used: every number the report cites is
computed here from ``benchmarks/results/``, so figures are read off one
verified source rather than transcribed from a terminal, and the whole
report can be re-checked against a fresh run in one command.

    python benchmarks/week4_graph_benchmark.py
    python tools/week4_facts.py

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "graphs_comparison_table.csv"))
    representation = pd.read_csv(os.path.join(RESULTS, "week4_representation.csv"))
    traversal = pd.read_csv(os.path.join(RESULTS, "week4_traversal.csv"))
    shortest = pd.read_csv(os.path.join(RESULTS, "week4_dijkstra.csv"))

    heading("1. Representation: adjacency list against adjacency matrix")
    for row in representation.itertuples():
        matrix_bytes = row.matrix_bytes
        if str(row.matrix_status) != "measured":
            print(f"  n={row.n:>7,}  list {row.list_bytes / 1e6:8.3f} MB  "
                  f"{row.list_lookup_ns:7.1f} ns/lookup  |  matrix {row.matrix_status}")
            continue
        ratio = float(matrix_bytes) / float(row.list_bytes)
        print(f"  n={row.n:>7,}  E={row.edges:>8,}  density {row.density:.5f}")
        print(f"            list   {row.list_bytes / 1e6:9.3f} MB   "
              f"{row.list_lookup_ns:8.1f} ns per lookup")
        print(f"            matrix {float(matrix_bytes) / 1e6:9.3f} MB   "
              f"{row.matrix_lookup_ns:8.1f} ns per lookup   "
              f"(build {row.matrix_build_s} s)")
        print(f"            matrix / list: {ratio:8.1f}x the memory, "
              f"{row.matrix_lookup_ns / row.list_lookup_ns:5.2f}x the lookup time")
    measured = representation[representation.matrix_status == "measured"]
    if len(measured) >= 2:
        first, last = measured.iloc[0], measured.iloc[-1]
        print(f"  memory growth {first.n:,} -> {last.n:,}: list "
              f"{last.list_bytes / first.list_bytes:.1f}x, matrix "
              f"{float(last.matrix_bytes) / float(first.matrix_bytes):.1f}x "
              f"(n grew {last.n / first.n:.0f}x)")

    heading("2. Traversal: BFS against DFS, sparse and dense")
    for kind in ("sparse", "dense"):
        rows = traversal[traversal.kind == kind]
        if rows.empty:
            continue
        print(f"  --- {kind} ---")
        for algorithm in rows.algorithm.unique():
            series = rows[rows.algorithm == algorithm].sort_values("n")
            times = "  ".join(f"{v * 1e3:8.3f}" for v in series.seconds)
            print(f"  {algorithm:<16} ms: {times}")
        series = rows[rows.algorithm == "BFS"].sort_values("n")
        print(f"  {'sizes':<16} V : " + "  ".join(f"{int(v):8,}" for v in series.n))
        print(f"  {'':<16} V+E:" + "  ".join(f"{int(v):8,}" for v in series.work_v_plus_e))
        for algorithm in rows.algorithm.unique():
            series = rows[rows.algorithm == algorithm].sort_values("n")
            per_unit = series.ns_per_v_plus_e.tolist()
            print(f"  {algorithm:<16} ns per V+E: " +
                  "  ".join(f"{v:6.1f}" for v in per_unit) +
                  f"   spread {max(per_unit) / min(per_unit):.2f}x")
        largest = rows[rows.n == rows.n.max()]
        base = largest[largest.algorithm == "BFS"].iloc[0]
        for algorithm in rows.algorithm.unique():
            other = largest[largest.algorithm == algorithm].iloc[0]
            print(f"  at V={int(base.n):,}: {algorithm} / BFS = "
                  f"{other.seconds / base.seconds:.2f}x "
                  f"({other.seconds * 1e3:.3f} ms against {base.seconds * 1e3:.3f} ms)")

    heading("3. Dijkstra: the Week 3 heap against a linear scan")
    for implementation in shortest.implementation.unique():
        series = shortest[shortest.implementation == implementation].sort_values("n")
        sizes = ", ".join(f"{int(v):,}" for v in series.n)
        print(f"  {implementation:<20} V = {sizes}")
        print(f"  {'':<20} ms = " +
              ", ".join(f"{v * 1e3:.2f}" for v in series.seconds))
    heap = shortest[shortest.implementation == "heap priority queue"].sort_values("n")
    scan = shortest[shortest.implementation == "linear scan"].sort_values("n")
    overlap = heap[heap.n.isin(scan.n)]
    for row in overlap.itertuples():
        other = scan[scan.n == row.n].iloc[0]
        print(f"  V={int(row.n):>6,}: linear scan / heap = "
              f"{other.seconds / row.seconds:6.2f}x "
              f"({other.seconds * 1e3:8.2f} ms against {row.seconds * 1e3:7.2f} ms)")
    if len(heap) >= 2:
        first, last = heap.iloc[0], heap.iloc[-1]
        factor = last.n / first.n
        predicted = (last.work_v_plus_e * math.log2(last.n)) / (
            first.work_v_plus_e * math.log2(first.n))
        print(f"  heap: V grew {factor:.0f}x, time grew "
              f"{last.seconds / first.seconds:.1f}x, "
              f"(V+E)logV predicts {predicted:.1f}x")
    if len(scan) >= 2:
        first, last = scan.iloc[0], scan.iloc[-1]
        print(f"  scan: V grew {last.n / first.n:.0f}x, time grew "
              f"{last.seconds / first.seconds:.1f}x, V^2 predicts "
              f"{(last.n / first.n) ** 2:.1f}x")
        print(f"  the two series cover different ranges on purpose: heap to "
              f"V={int(heap.n.max()):,}, scan capped at V={int(scan.n.max()):,}")

    heading("4. Asymptotic against empirical (graphs_comparison_table.csv)")
    for row in table.itertuples():
        flag = "" if row.agrees else "   <-- disagrees"
        print(f"  {row.structure:<26}{row.operation:<28}{row.asymptotic:<18}"
              f"slope {row.loglog_slope:>7.3f}  growth {row.growth_measured:>9.2f} "
              f"(predicted {row.growth_predicted:>8.2f})  -> {row.empirical_class}{flag}")
    agree = int(table.agrees.sum())
    print(f"\n  {agree} of {len(table)} series match their expected class")
    for row in table[~table.agrees].itertuples():
        print(f"  disagreement: {row.structure} {row.operation}: expected "
              f"{row.expected_growth_class}, measured {row.empirical_class}"
              f"{' (' + row.note + ')' if isinstance(row.note, str) and row.note else ''}")

    heading("5. Reductions made under the runtime budget")
    skipped = representation[representation.matrix_status != "measured"]
    if skipped.empty:
        print("  representation: none, every size measured on both representations")
    for row in skipped.itertuples():
        print(f"  representation: n={row.n:,} matrix {row.matrix_status}")
    capped = shortest[shortest.note == "capped"] if "note" in shortest else []
    print(f"  dijkstra: linear scan capped at V={int(scan.n.max()):,} of "
          f"V={int(heap.n.max()):,} measured for the heap")
    dense = traversal[traversal.kind == "dense"]
    sparse = traversal[traversal.kind == "sparse"]
    if not dense.empty and not sparse.empty:
        print(f"  traversal: dense graphs stop at V={int(dense.n.max()):,} against "
              f"V={int(sparse.n.max()):,} sparse, since density 0.5 at the sparse "
              f"maximum would be {int(sparse.n.max()) ** 2 // 4:,} edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
