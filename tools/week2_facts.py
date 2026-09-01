#!/usr/bin/env python3
"""Print every figure quoted in analysis/week2_report.md, computed from the CSVs.

The Week 2 report is prose under a strict word limit, so it cannot be
generated wholesale the way the Week 1 report is. This script is the
compromise: it computes each number the report cites, so the figures are
read off a single verified source rather than transcribed from scrollback,
and the report can be re-checked against a fresh run in one command.

    python benchmarks/week2_performance.py --validate-cap
    python tools/week2_facts.py

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")

ALGOS = [
    "Bubble Sort", "Selection Sort", "Insertion Sort",
    "Merge Sort", "Quick Sort", "Quick Sort (3-way)",
]
TYPES = ["random", "sorted", "reverse", "nearly_sorted", "many_duplicates", "few_unique"]


def heading(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "comparison_table.csv"))
    measured = table[table.status == "measured"].copy()
    measured["mean_time"] = measured.mean_time.astype(float)
    ratios = pd.read_csv(os.path.join(RESULTS, "doubling_ratios.csv"))
    schemes = pd.read_csv(os.path.join(RESULTS, "partition_scheme_study.csv"))

    def t(alg, dtype, n):
        row = measured[
            (measured.algorithm_name == alg)
            & (measured.data_type == dtype)
            & (measured.input_size == n)
        ]
        return float(row.mean_time.iloc[0]) if len(row) else float("nan")

    heading("1. Mean runtime (s) at n = 10,000")
    print(
        measured[measured.input_size == 10000]
        .pivot_table(index="algorithm_name", columns="data_type", values="mean_time")
        .reindex(ALGOS)[TYPES]
        .to_string(float_format=lambda v: f"{v:.6f}")
    )

    heading("2. Mean runtime (s) at n = 50,000")
    print(
        measured[measured.input_size == 50000]
        .pivot_table(index="algorithm_name", columns="data_type", values="mean_time")
        .to_string(float_format=lambda v: f"{v:.6f}")
    )

    heading("3. O(n^2) vs O(n log n) speedup on random input")
    print(f"  {'n':>7}{'best quadratic':>17}{'merge':>12}{'x':>8}{'quick':>12}{'x':>8}")
    for n in [100, 500, 1000, 5000, 10000]:
        quad = min(t(a, "random", n) for a in ALGOS[:3])
        merge, quick = t("Merge Sort", "random", n), t("Quick Sort", "random", n)
        print(f"  {n:>7}{quad:>17.6f}{merge:>12.6f}{quad / merge:>7.1f}x"
              f"{quick:>12.6f}{quad / quick:>7.1f}x")

    heading("4. True doubling ratios (500->1000, 5000->10000)")
    true_doubling = ratios[ratios.is_true_doubling]
    print(
        true_doubling.pivot_table(
            index="algorithm_name", columns="data_type", values="observed_ratio"
        ).reindex(ALGOS)[TYPES].to_string(float_format=lambda v: f"{v:.2f}")
    )
    print("\n  per algorithm across all shapes:")
    print(
        true_doubling.groupby("algorithm_name")
        .observed_ratio.agg(["mean", "min", "max"])
        .reindex(ALGOS)
        .to_string(float_format=lambda v: f"{v:.2f}")
    )

    heading("5. Sensitivity to input order (spread across the six shapes)")
    for n in (10000, 50000):
        print(f"\n  n = {n:,}")
        for alg in ALGOS:
            s = measured[
                (measured.algorithm_name == alg) & (measured.input_size == n)
            ].set_index("data_type").mean_time
            if len(s) < 2:
                continue
            print(f"    {alg:<20} min={s.min():.6f}  max={s.max():.6f}  "
                  f"spread={s.max() / s.min():>10.1f}x")

    heading("6. Three-way vs two-way quicksort")
    for n in (10000, 50000):
        print(f"\n  n = {n:,}")
        for dtype in TYPES:
            two, three = t("Quick Sort", dtype, n), t("Quick Sort (3-way)", dtype, n)
            verdict = "faster" if three < two else "SLOWER"
            print(f"    {dtype:<16} 2-way={two:.6f}  3-way={three:.6f}  "
                  f"{two / three:>6.2f}x  ({verdict})")

    heading("7. Partition scheme study (growth per doubling)")
    for dtype in ["few_unique", "many_duplicates", "random"]:
        print(f"\n  {dtype}")
        for scheme in ["Lomuto (2-way)", "Hoare (2-way)", "Dutch flag (3-way)"]:
            s = schemes[
                (schemes.data_type == dtype) & (schemes.scheme == scheme)
            ].sort_values("input_size")
            growth = s.growth_vs_previous.dropna()
            largest = s[s.input_size == s.input_size.max()].mean_time.iloc[0]
            print(f"    {scheme:<20} mean growth={growth.mean():.2f}  "
                  f"t(n_max)={largest:.5f}s")
        lom = schemes[(schemes.data_type == dtype) & (schemes.scheme == "Lomuto (2-way)")]
        hoa = schemes[(schemes.data_type == dtype) & (schemes.scheme == "Hoare (2-way)")]
        n_max = schemes.input_size.max()
        ratio = (
            lom[lom.input_size == n_max].mean_time.iloc[0]
            / hoa[hoa.input_size == n_max].mean_time.iloc[0]
        )
        print(f"    -> Lomuto/Hoare at n={n_max:,}: {ratio:.0f}x")

    heading("8. Merge sort consistency")
    for n in (10000, 50000):
        s = measured[
            (measured.algorithm_name == "Merge Sort") & (measured.input_size == n)
        ].set_index("data_type").mean_time
        print(f"  n={n:,}: spread={s.max() / s.min():.2f}x  "
              f"fastest={s.idxmin()} ({s.min():.6f}s)  "
              f"slowest={s.idxmax()} ({s.max():.6f}s)")

    heading("9. Crossover: where the quadratics stop winning (random)")
    for n in [100, 500, 1000]:
        ins, merge = t("Insertion Sort", "random", n), t("Merge Sort", "random", n)
        print(f"  n={n:<5} insertion={ins:.6f}  merge={merge:.6f}  "
              f"insertion faster: {ins < merge}")

    heading("10. Capped cells and the projection")
    omitted = table[table.status == "omitted"]
    print(f"  omitted cells: {len(omitted)} "
          f"({sorted(omitted.algorithm_name.unique())} at n=50,000)")
    validation_path = os.path.join(RESULTS, "cap_validation.csv")
    if os.path.exists(validation_path):
        validation = pd.read_csv(validation_path)
        done = validation[validation.measured_time.notna()]
        print(f"\n  projection validated on {len(done)} cells:")
        print(done[["algorithm_name", "data_type", "projected_time",
                    "measured_time", "projection_error_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        errors = done.projection_error_pct.astype(float).abs()
        print(f"\n  absolute projection error: mean {errors.mean():.1f}%  "
              f"median {errors.median():.1f}%  max {errors.max():.1f}%")
        total = done.measured_time.astype(float).sum()
        print(f"  measured total for one run of every omitted cell: "
              f"{total:.0f}s ({total / 60:.1f} min)")
        print(f"  at the matrix's 3 runs + 2 warmup: "
              f"{total * 5 / 60:.0f} min of extra wall time")
    else:
        print("  (cap_validation.csv not present - run with --validate-cap)")

    heading("11. Run configuration")
    print(f"  measured cells: {len(measured)}   omitted: {len(omitted)}")
    print(f"  runs used: {sorted(measured.runs.unique())}")
    rel = measured.std_deviation.astype(float) / measured.mean_time
    print(f"  relative SD: median {100 * rel.median():.2f}%  "
          f"mean {100 * rel.mean():.2f}%  max {100 * rel.max():.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
