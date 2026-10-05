#!/usr/bin/env python3
"""Print every figure analysis/week6_report.md quotes, computed from the CSVs.

The compromise Weeks 2 through 5 used: the report is prose under a word
limit, so every number it cites is computed here from
``benchmarks/results/`` instead of being transcribed, and the report can be
re-checked against a fresh run in one command.

    python benchmarks/week6_dp_advanced_benchmark.py
    python tools/week6_facts.py

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
    table = pd.read_csv(os.path.join(RESULTS, "comparison_table.csv"))
    negative = pd.read_csv(os.path.join(RESULTS, "week6_negative_edges.csv"))
    table["ms"] = table.mean_time_s * 1e3

    def pick(problem: str, variant: str) -> pd.DataFrame:
        rows = table[(table.problem == problem) & (table.variant == variant)]
        return rows.sort_values(["secondary_param", "n"])

    heading("1. Knapsack: Week 5 2D table against the one-row version")
    std, opt = pick("knapsack", "standard_2d"), pick("knapsack", "space_optimized_1d")
    merged = std.merge(opt, on=["n", "secondary_param"], suffixes=("_2d", "_1d"))
    for row in merged.itertuples():
        mem = (f"{row.peak_kib_2d:10,.1f} vs {row.peak_kib_1d:8,.1f} KiB "
               f"({row.peak_kib_2d / row.peak_kib_1d:6.1f}x)"
               if not (pd.isna(row.peak_kib_2d) or pd.isna(row.peak_kib_1d)) else "untraced")
        print(f"  n={row.n:<4} W={int(row.secondary_param):<6} time {row.ms_2d:9.3f} vs "
              f"{row.ms_1d:9.3f} ms (2D/1D {row.ms_2d / row.ms_1d:5.2f}x)  memory {mem}")
    traced = merged.dropna(subset=["peak_kib_2d", "peak_kib_1d"])
    if not traced.empty:
        ratios = traced.peak_kib_2d / traced.peak_kib_1d
        times = traced.ms_2d / traced.ms_1d
        print(f"  memory ratio range {ratios.min():.1f}x to {ratios.max():.1f}x; "
              f"time ratio range {times.min():.2f}x to {times.max():.2f}x")

    heading("2. Matrix chain: recursion against memoized and bottom-up")
    for variant in ("recursive", "memoized", "bottom_up"):
        rows = pick("mcm", variant)
        print(f"  {variant:<10} n  = " + ", ".join(str(int(v)) for v in rows.n))
        print(f"  {'':<10} ms = " + ", ".join(f"{v:.3f}" for v in rows.ms))
    rec = pick("mcm", "recursive").set_index("n")
    for variant in ("memoized", "bottom_up"):
        rows = pick("mcm", variant).set_index("n")
        shared = [n for n in rows.index if n in rec.index]
        if shared:
            print(f"  recursive/{variant}: " + ", ".join(
                f"n={n} {rec.loc[n].ms / rows.loc[n].ms:,.1f}x" for n in shared))
    if len(rec) >= 2:
        steps = [rec.ms.iloc[i + 1] / rec.ms.iloc[i] for i in range(len(rec) - 1)]
        gaps = [rec.index[i + 1] - rec.index[i] for i in range(len(rec) - 1)]
        per_matrix = [s ** (1 / g) for s, g in zip(steps, gaps)]
        print("  recursive growth per added matrix: " + ", ".join(f"{p:.2f}x" for p in per_matrix))
    bu = pick("mcm", "bottom_up").set_index("n")
    for low, high in ((50, 100), (100, 200), (25, 200)):
        if low in bu.index and high in bu.index:
            print(f"  bottom_up n {low} -> {high}: time {bu.loc[high].ms / bu.loc[low].ms:.2f}x, "
                  f"n^3 predicts {(high / low) ** 3:.1f}x")

    heading("3. Floyd-Warshall scaling, against all-pairs Dijkstra, and 2D against 3D")
    fw = pick("floyd_warshall", "floyd_warshall")
    dense = fw[fw.secondary_param == fw.secondary_param.max()].set_index("n")
    print("  FW at density " f"{fw.secondary_param.max()}: " + ", ".join(
        f"n={n} {r.ms:,.1f} ms ({int(r.runs)} runs)" for n, r in dense.iterrows()))
    sizes = list(dense.index)
    for a, b in zip(sizes, sizes[1:]):
        print(f"  n {a} -> {b}: time {dense.loc[b].ms / dense.loc[a].ms:.2f}x, "
              f"n^3 predicts {(b / a) ** 3:.2f}x")
    if len(sizes) >= 2:
        slope = (math.log(dense.ms.iloc[-1]) - math.log(dense.ms.iloc[0])) / (
            math.log(sizes[-1]) - math.log(sizes[0]))
        print(f"  log-log slope over {sizes[0]}..{sizes[-1]}: {slope:.3f} (theory 3)")
        print(f"  ns per inner step at n={sizes[-1]}: "
              f"{dense.ms.iloc[-1] * 1e6 / sizes[-1] ** 3:.1f}")
    dij = pick("floyd_warshall", "all_pairs_dijkstra")
    for density in sorted(dij.secondary_param.unique()):
        d_rows = dij[dij.secondary_param == density].set_index("n")
        f_rows = fw[fw.secondary_param == density].set_index("n")
        parts = []
        for n in d_rows.index:
            if n in f_rows.index:
                parts.append(f"n={n} FW {f_rows.loc[n].ms:,.1f} vs Dij {d_rows.loc[n].ms:,.1f} ms "
                             f"(Dij/FW {d_rows.loc[n].ms / f_rows.loc[n].ms:.2f}x)")
        print(f"  density {density}: " + "; ".join(parts))
    three = pick("floyd_warshall", "floyd_warshall_3d").set_index("n")
    for n in three.index:
        two = fw[(fw.n == n) & (fw.secondary_param == fw.secondary_param.max())]
        if not two.empty and not pd.isna(three.loc[n].peak_kib) and not pd.isna(two.iloc[0].peak_kib):
            p2, p3 = two.iloc[0].peak_kib, three.loc[n].peak_kib
            print(f"  n={n}: 3D {p3:,.1f} KiB vs 2D {p2:,.1f} KiB ({p3 / p2:.1f}x); "
                  f"time 3D {three.loc[n].ms:,.1f} vs 2D {two.iloc[0].ms:,.1f} ms")
    print("  negative edges (CLRS example):")
    for row in negative.itertuples():
        print(f"    row {row.source}: {row.floyd_warshall_row}")
    print(f"    Week 4 Dijkstra: {negative.all_pairs_dijkstra.iloc[0]}")

    heading("4. TSP: Held-Karp bitmask against brute force")
    bit = pick("tsp", "bitmask").set_index("n")
    bru = pick("tsp", "brute_force").set_index("n")
    print("  bitmask ms: " + ", ".join(f"n={n} {r.ms:,.3f}" for n, r in bit.iterrows()))
    print("  brute   ms: " + ", ".join(f"n={n} {r.ms:,.3f} ({int(r.runs)} runs)"
                                       for n, r in bru.iterrows()))
    crossover = None
    for n in bru.index:
        if n in bit.index:
            ratio = bru.loc[n].ms / bit.loc[n].ms
            print(f"  n={n:<3} brute/bitmask = {ratio:12,.2f}x")
            if crossover is None and ratio > 1:
                crossover = n
    print(f"  crossover (first n where bitmask is faster): {crossover}")
    for n in (bit.index.max(),):
        print(f"  bitmask at n={n}: {bit.loc[n].ms:,.1f} ms, peak "
              f"{bit.loc[n].peak_kib if not pd.isna(bit.loc[n].peak_kib) else 'untraced'} KiB")
    traced_bit = bit.dropna(subset=["peak_kib"])
    if not traced_bit.empty:
        last = traced_bit.index.max()
        print(f"  largest traced bitmask: n={last}, {traced_bit.loc[last].peak_kib:,.1f} KiB")
    if len(bru) >= 2:
        a, b = bru.index[-2], bru.index[-1]
        print(f"  brute n {a} -> {b}: time {bru.loc[b].ms / bru.loc[a].ms:.2f}x "
              f"(factorial predicts {b - 1}x)")
    if len(bit) >= 2:
        a, b = bit.index[-2], bit.index[-1]
        print(f"  bitmask n {a} -> {b}: time {bit.loc[b].ms / bit.loc[a].ms:.2f}x "
              f"(n^2 2^n predicts {(b * b * 2 ** b) / (a * a * 2 ** a):.2f}x)")


    heading("6. Derived figures the report quotes")
    k2, k1 = pick("knapsack", "standard_2d"), pick("knapsack", "space_optimized_1d")
    for label, rows in (("2D", k2), ("1D", k1)):
        at_n = rows[rows.n == 100].set_index("secondary_param")
        if 1000 in at_n.index and 10000 in at_n.index:
            print(f"  knapsack {label} at n=100, W 1000 -> 10000: time "
                  f"{at_n.loc[10000].ms / at_n.loc[1000].ms:.1f}x")
    at_w = k2[k2.secondary_param == 1000].set_index("n")
    for n in at_w.index:
        cells = (n + 1) * 1001
        print(f"  knapsack 2D at W=1000, n={n}: {at_w.loc[n].peak_kib * 1024 / cells:.1f} bytes per cell")
    fw2 = fw[fw.secondary_param == fw.secondary_param.max()].set_index("n")
    if 50 in three.index and 100 in three.index:
        print(f"  FW 50 -> 100: 3D time {three.loc[100].ms / three.loc[50].ms:.2f}x, "
              f"2D time {fw2.loc[100].ms / fw2.loc[50].ms:.2f}x")
    memo, bottom = pick("mcm", "memoized").set_index("n"), pick("mcm", "bottom_up").set_index("n")
    top = max(set(memo.index) & set(bottom.index))
    print(f"  MCM at n={top}: memoized/bottom_up time {memo.loc[top].ms / bottom.loc[top].ms:.2f}x; "
          f"peak memoized {memo.loc[top].peak_kib:,.1f} KiB, bottom_up {bottom.loc[top].peak_kib:,.1f} KiB")
    if 17 in bit.index and 18 in bit.index:
        print(f"  bitmask peak 17 -> 18: {bit.loc[18].peak_kib / bit.loc[17].peak_kib:.3f}x; "
              f"n 2^n predicts {(18 * 2 ** 18) / (17 * 2 ** 17):.3f}x")
    print(f"  at 12 cities: n^2 2^n = {12 * 12 * 2 ** 12:,} transitions, 11! = {math.factorial(11):,} orderings")
    print(f"  at 500 vertices: 2D {500 * 500:,} entries, 3D {501 * 500 * 500:,} entries")

    heading("5. Repetitions and tracing, for the Methodology section")
    for runs, group in table.groupby("runs"):
        labels = ", ".join(f"{r.problem}/{r.variant}/n={r.n}" for r in group.itertuples())
        print(f"  runs={runs}: {len(group)} rows" + (f" -> {labels}" if runs < 5 else ""))
    untraced = table[table.peak_kib.isna()]
    print(f"  untraced rows (>= 2 s): {len(untraced)} -> " + ", ".join(
        f"{r.problem}/{r.variant}/n={r.n}" for r in untraced.itertuples()))
    print(f"  measurement values: {sorted(table.measurement.unique())}")
    print(f"  total rows: {len(table)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
