#!/usr/bin/env python3
"""Rewrite the data tables in analysis/week2_report.md from the result CSVs.

The report is hand-written prose under a word limit, but its tables are
pure data and must not drift from the measurements after a re-run. This
script regenerates them in place, matching on the table's header row so
the surrounding prose is untouched.

    python benchmarks/week2_performance.py
    python tools/week2_sync_report.py

It prints any inline figure in the prose that no longer matches the data,
so those can be corrected by hand rather than silently going stale.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")
REPORT = os.path.join(REPO_ROOT, "analysis", "week2_report.md")

SHAPES = ["random", "sorted", "reverse", "nearly_sorted", "many_duplicates", "few_unique"]
ROWS = [
    ("Bubble", "Bubble Sort"),
    ("Selection", "Selection Sort"),
    ("Insertion", "Insertion Sort"),
    ("Merge", "Merge Sort"),
    ("Quick", "Quick Sort"),
    ("Quick 3-way", "Quick Sort (3-way)"),
]


def replace_table(text: str, header: str, new_rows: list) -> str:
    """Swap the body of the table whose header row matches `header`."""
    lines = text.split("\n")
    try:
        start = lines.index(header)
    except ValueError:
        raise SystemExit(f"header not found in report:\n  {header}")
    end = start + 2  # skip header and separator
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return "\n".join(lines[: start + 2] + new_rows + lines[end:])


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "comparison_table.csv"))
    measured = table[table.status == "measured"].copy()
    measured["mean_time"] = measured.mean_time.astype(float)
    ratios = pd.read_csv(os.path.join(RESULTS, "doubling_ratios.csv"))
    schemes = pd.read_csv(os.path.join(RESULTS, "partition_scheme_study.csv"))

    def t(alg, shape, n):
        row = measured[
            (measured.algorithm_name == alg)
            & (measured.data_type == shape)
            & (measured.input_size == n)
        ]
        return float(row.mean_time.iloc[0]) if len(row) else None

    text = open(REPORT, encoding="utf-8").read()

    # Table 1 - mean seconds at n = 10,000
    rows = [
        "| " + label + " | " + " | ".join(f"{t(alg, s, 10000):.4f}" for s in SHAPES) + " |"
        for label, alg in ROWS
    ]
    text = replace_table(
        text,
        "| Algorithm | random | sorted | reverse | nearly | many dup | few uniq |",
        rows,
    )

    # Table 2 - speedup over the best quadratic, random input
    sizes = [100, 500, 1000, 5000, 10000]
    speedups = {}
    for name in ("Merge Sort", "Quick Sort"):
        cells = []
        for n in sizes:
            best_quadratic = min(
                t(a, "random", n) for a in ("Bubble Sort", "Selection Sort", "Insertion Sort")
            )
            cells.append(f"{best_quadratic / t(name, 'random', n):.1f}×")
        speedups[name] = cells
    text = replace_table(
        text,
        "| n | 100 | 500 | 1,000 | 5,000 | 10,000 |",
        [
            "| Merge | " + " | ".join(speedups["Merge Sort"]) + " |",
            "| Quick | " + " | ".join(speedups["Quick Sort"]) + " |",
        ],
    )

    # Table 3 - doubling ratios
    doubling = ratios[ratios.is_true_doubling]
    predicted = {
        "Bubble Sort": "4 / 2 adaptive",
        "Selection Sort": "4",
        "Insertion Sort": "4 / 2 adaptive",
        "Merge Sort": "~2.1",
        "Quick Sort": "~2.1",
    }
    rows = []
    for label, alg in ROWS[:5]:
        cells = []
        for shape in ("random", "sorted", "reverse", "few_unique"):
            match = doubling[
                (doubling.algorithm_name == alg) & (doubling.data_type == shape)
            ].observed_ratio
            cells.append(f"{match.mean():.2f}")
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {predicted[alg]} |")
    text = replace_table(
        text,
        "| Algorithm | random | sorted | reverse | few uniq | Predicted |",
        rows,
    )

    # Table 4 - partition schemes on few-unique input
    rows = []
    n_max = int(schemes.input_size.max())
    for scheme in ("Lomuto (2-way)", "Hoare (2-way)", "Dutch flag (3-way)"):
        series = schemes[
            (schemes.scheme == scheme) & (schemes.data_type == "few_unique")
        ].sort_values("input_size")
        growth = series.growth_vs_previous.dropna().mean()
        largest = series[series.input_size == n_max].mean_time.iloc[0]
        emphasis = "**" if scheme.startswith("Lomuto") else ""
        rows.append(
            f"| {scheme} | {emphasis}{growth:.2f}{emphasis} | {largest:.4f} s |"
        )
    text = replace_table(
        text, "| Scheme (3 values) | growth/doubling | t(16,000) |", rows
    )

    with open(REPORT, "w", encoding="utf-8") as handle:
        handle.write(text)

    words = len(text.split())
    print(f"tables synced; report is {words} words "
          f"({'in range' if 800 <= words <= 1200 else 'OUT OF RANGE'})")

    # Flag inline prose figures that no longer match the data.
    print("\ncross-checking inline figures quoted in the prose:")
    # Labels name the figure the report currently quotes, so a mismatch
    # between the label and the computed value is the signal to edit the
    # prose. Keep them in step when the prose changes.
    checks = [
        ("quick speedup at n=10,000            (report: 154x)",
         min(t(a, "random", 10000) for a in ("Bubble Sort", "Selection Sort", "Insertion Sort"))
         / t("Quick Sort", "random", 10000)),
        ("insertion at n=100 random            (report: 0.000080)",
         t("Insertion Sort", "random", 100)),
        ("merge at n=100 random                (report: 0.000079)",
         t("Merge Sort", "random", 100)),
        ("quick best, few_unique n=10,000      (report: 0.0047)",
         t("Quick Sort", "few_unique", 10000)),
        ("quick worst, random n=10,000         (report: 0.0073)",
         t("Quick Sort", "random", 10000)),
    ]
    for label, value in checks:
        print(f"  {label:<45} -> {value:.6f}")

    for name, label in (("Merge Sort", "merge"), ("Quick Sort", "quick"),
                        ("Bubble Sort", "bubble"), ("Insertion Sort", "insertion")):
        for n in (10000,):
            series = measured[
                (measured.algorithm_name == name) & (measured.input_size == n)
            ].mean_time
            if len(series) > 1:
                print(f"  {label} spread across shapes at n={n:,}"
                      f"{'':<10} -> {series.max() / series.min():,.1f}x")
    rel = measured.std_deviation.astype(float) / measured.mean_time
    print(f"  median relative SD{'':<28} -> {100 * rel.median():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
