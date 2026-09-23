#!/usr/bin/env python3
"""Print every figure analysis/week5_report.md quotes, computed from the CSVs.

The report is prose under a word limit, so it cannot be generated whole.
This is the compromise Weeks 2 through 4 used: every number the report
cites is computed here from ``benchmarks/results/``, so figures are read
off one verified source rather than transcribed, and the whole report can
be re-checked against a fresh run in one command.

    python benchmarks/week5_dp_benchmark.py
    python tools/week5_facts.py

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


def number(value: object) -> float:
    """Blank cells are deliberate in this CSV; treat them as missing."""
    return float("nan") if value in ("", None) or pd.isna(value) else float(value)


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "dp_vs_recursive_table.csv"))
    cells = pd.read_csv(os.path.join(RESULTS, "week5_knapsack_cells.csv"))
    probe = pd.read_csv(os.path.join(RESULTS, "week5_recursion_probe.csv"))

    def pick(problem: str, variant: str) -> pd.DataFrame:
        rows = table[(table.problem == problem) & (table.variant == variant)]
        return rows.sort_values(["n", "secondary_param"])

    heading("1. Fibonacci: naive against memoized against tabulated")
    for variant in ("naive", "memo", "tab"):
        rows = pick("fibonacci", variant)
        times = "  ".join(
            f"{number(r.mean_time_s) * 1e3:10.4f}{'*' if r.measurement == 'projected' else ' '}"
            for r in rows.itertuples())
        print(f"  {variant:<6} ms: {times}")
    sizes = pick("fibonacci", "tab").n.tolist()
    print(f"  {'':<6} n : " + "  ".join(f"{int(v):>10}" for v in sizes))
    print("  (* = projected from the measured per-call cost, not run)")
    naive = pick("fibonacci", "naive").set_index("n")
    for variant in ("memo", "tab"):
        rows = pick("fibonacci", variant).set_index("n")
        for n in sizes:
            if n not in naive.index or n not in rows.index:
                continue
            base, mine = naive.loc[n], rows.loc[n]
            ratio = number(base.mean_time_s) / number(mine.mean_time_s)
            tag = "projected baseline" if base.measurement == "projected" else "measured"
            print(f"  n={n:<3} naive/{variant:<4} = {ratio:12,.0f}x  ({tag})")
    measured = naive[naive.measurement == "measured"]
    if not measured.empty:
        last = measured.iloc[-1]
        per_call = number(last.mean_time_s) / float(last.calls)
        print(f"  per-call cost from n={int(last.name)}: {per_call * 1e9:.2f} ns "
              f"({number(last.mean_time_s):.4f} s over {int(last.calls):,} calls)")
    for row in naive.itertuples():
        print(f"  n={int(row.Index):<3} calls={int(row.calls):>15,}  depth={row.max_depth:>4}  "
              f"{row.measurement}")
    tab_rows = pick("fibonacci", "tab")
    print(f"  tabulation depth: {sorted(set(tab_rows.max_depth))} at every n")
    print(f"  memo depth: {pick('fibonacci', 'memo').max_depth.tolist()}")

    heading("2. Fibonacci: measured growth against theory")
    for variant in ("naive", "memo", "tab"):
        rows = pick("fibonacci", variant)
        rows = rows[rows.measurement == "measured"]
        if len(rows) < 2:
            continue
        first, last = rows.iloc[0], rows.iloc[-1]
        grew = number(last.mean_time_s) / number(first.mean_time_s)
        n_ratio = last.n / first.n
        call_ratio = float(last.calls) / float(first.calls)
        print(f"  {variant:<6} n {int(first.n)} -> {int(last.n)} ({n_ratio:.1f}x): "
              f"time grew {grew:12,.1f}x, calls grew {call_ratio:12,.1f}x")

    heading("3. Knapsack: item count at fixed capacity")
    fixed_capacity = int(cells.capacity.iloc[0])
    at_capacity = table[(table.problem == "knapsack")
                        & (table.secondary_param == fixed_capacity)]
    for variant in ("recursive", "memo", "tab"):
        rows = at_capacity[at_capacity.variant == variant].sort_values("n")
        if rows.empty:
            continue
        print(f"  {variant:<10} n = " + ", ".join(f"{int(v)}" for v in rows.n))
        print(f"  {'':<10} ms= " + ", ".join(
            f"{number(v) * 1e3:.3f}" for v in rows.mean_time_s))
    for row in at_capacity[at_capacity.variant == "memo"].itertuples():
        base = at_capacity[(at_capacity.variant == "recursive")
                           & (at_capacity.n == row.n)]
        if not base.empty:
            ratio = number(base.iloc[0].mean_time_s) / number(row.mean_time_s)
            print(f"  n={int(row.n):<4} recursive/memo = {ratio:10,.1f}x")
    recursive_rows = at_capacity[at_capacity.variant == "recursive"]
    if not recursive_rows.empty:
        print(f"  recursive capped at n={int(recursive_rows.n.max())}, "
              f"DP runs to n={int(at_capacity.n.max())}")

    heading("4. Knapsack: capacity at fixed item count, and table occupancy")
    fixed_n = int(cells.n.iloc[0]) if False else None
    by_capacity = table[(table.problem == "knapsack") & (table.variant == "tab")]
    grouped = by_capacity.groupby("n").filter(lambda g: g.secondary_param.nunique() > 1)
    if not grouped.empty:
        n_value = int(grouped.n.iloc[0])
        rows = grouped.sort_values("secondary_param")
        print(f"  at n={n_value}: W = " + ", ".join(
            f"{int(v)}" for v in rows.secondary_param))
        print(f"  {'':<12} ms= " + ", ".join(
            f"{number(v) * 1e3:.3f}" for v in rows.mean_time_s))
        first, last = rows.iloc[0], rows.iloc[-1]
        print(f"  W grew {last.secondary_param / first.secondary_param:.0f}x, "
              f"time grew {number(last.mean_time_s) / number(first.mean_time_s):.1f}x "
              f"(n*W predicts linear in W)")
    print("  memo entries against the full table:")
    for row in cells.itertuples():
        print(f"    n={int(row.n):<4} W={int(row.capacity):<5} "
              f"{int(row.memo_entries):>9,} of {int(row.table_cells):>9,} cells "
              f"({100 * row.fraction_of_table:5.1f}%)  optimum {int(row.optimum):>6,}  "
              f"{int(row.items_chosen)} items chosen")
    print(f"  occupancy range: {100 * cells.fraction_of_table.min():.1f}% to "
          f"{100 * cells.fraction_of_table.max():.1f}%")

    heading("5. LCS: the DP range, the recursive cap, and the depth wall")
    for variant in ("recursive", "memo", "tab"):
        rows = pick("lcs", variant)
        if rows.empty:
            continue
        print(f"  {variant:<10} len = " + ", ".join(f"{int(v)}" for v in rows.n))
        print(f"  {'':<10} ms  = " + ", ".join(
            f"{number(v) * 1e3:.3f}" for v in rows.mean_time_s))
        print(f"  {'':<10} depth=" + ", ".join(f"{int(v)}" for v in rows.max_depth))
    overlap = set(pick("lcs", "recursive").n) & set(pick("lcs", "memo").n)
    for length in sorted(overlap):
        base = pick("lcs", "recursive").set_index("n").loc[length]
        for variant in ("memo", "tab"):
            mine = pick("lcs", variant).set_index("n").loc[length]
            print(f"  len={length}: recursive/{variant} = "
                  f"{number(base.mean_time_s) / number(mine.mean_time_s):,.0f}x")
    tab_rows = pick("lcs", "tab")
    if len(tab_rows) >= 2:
        first, last = tab_rows.iloc[0], tab_rows.iloc[-1]
        grew = number(last.mean_time_s) / number(first.mean_time_s)
        predicted = (last.n / first.n) ** 2
        print(f"  tab len {int(first.n)} -> {int(last.n)}: time grew {grew:,.1f}x, "
              f"m*n predicts {predicted:,.1f}x")
    print("  recursion probe, at CPython's DEFAULT limit "
          f"({int(probe.recursion_limit.iloc[0])}):")
    for row in probe.itertuples():
        print(f"    length {int(row.length):>5}: memo {row.memo_status:<15} "
              f"tab {row.tab_status}")
    failed = probe[probe.memo_status != "ok"]
    if not failed.empty:
        print(f"  memo first fails at length {int(failed.length.iloc[0])}; "
              f"tabulation succeeds at every probed length")
    else:
        print("  memo survived every probed length at the default limit")

    heading("6. Peak memory and the rolling-row argument")
    for problem, variant in (("fibonacci", "memo"), ("fibonacci", "tab"),
                             ("lcs", "memo"), ("lcs", "tab")):
        rows = pick(problem, variant)
        rows = rows[rows.peak_kib != ""]
        if rows.empty:
            continue
        last = rows.iloc[-1]
        print(f"  {problem:<10} {variant:<5} at n={int(last.n):<5} "
              f"peak {number(last.peak_kib):10,.1f} KiB")

    heading("7. Reductions and projections stated in the report")
    projected = table[table.measurement == "projected"]
    print(f"  projected rows: {len(projected)}")
    for row in projected.itertuples():
        print(f"    {row.problem} {row.variant} n={int(row.n)}: "
              f"{number(row.mean_time_s):,.1f} s estimated from "
              f"{int(row.calls):,} calls")
    blank = table[table.speedup_vs_recursive.isna()
                  | (table.speedup_vs_recursive == "")]
    print(f"  rows with speedup deliberately blank: {len(blank)} of {len(table)}")
    print(f"  total rows in dp_vs_recursive_table.csv: {len(table)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
