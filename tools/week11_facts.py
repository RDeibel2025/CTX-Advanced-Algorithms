#!/usr/bin/env python3
"""Print every number the Week 11 report quotes, straight from the benchmark CSVs.

Run from the repository root after ``benchmarks/week11_benchmark_search.py``::

    python tools/week11_facts.py

Reads ``search_benchmark.csv`` and ``size_caps.csv`` from
``benchmarks/results/week11/``. If a benchmark number in
``analysis/week11_report.md`` is not printed here, it should not be in the
report.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results", "week11")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def loglog_slope(ns, values) -> float:
    return float(np.polyfit(np.log(np.asarray(ns, dtype=float)), np.log(np.asarray(values, dtype=float)), 1)[0])


def main() -> int:
    # Exact integers stay strings on read: the 2^61 - 1 modulus and the distinct-substring
    # counts do not survive a float64 round trip.
    t = pd.read_csv(os.path.join(RESULTS, "search_benchmark.csv"),
                    dtype={"distinct_substrings": "object", "modulus": "object", "base": "object"})
    caps = pd.read_csv(os.path.join(RESULTS, "size_caps.csv"))

    # ------------------------------------------------------------------
    header("1. Dataset sizes (m = 16)")
    ds = t[t.experiment == "dataset_sizes"]
    for (dataset, n), g in ds.groupby(["dataset", "n"]):
        parts = [f"{dataset:<8} n={n:>9,} q={int(g.queries.iloc[0])}"]
        for method in ("kmp", "rk", "sa", "st"):
            r = g[g.method == method]
            if len(r):
                r = r.iloc[0]
                s = f"{method} {r.mean_query_s * 1e3:9.4f} ms (sd {r.std_query_s * 1e3:.4f})"
                if pd.notna(r.preprocess_s):
                    s += f" build {r.preprocess_s:.3f} s"
                if pd.notna(r.comparisons):
                    s += f" cmp/char {r.comparisons / n:.3f}"
                if pd.notna(r.spurious_hits):
                    s += f" spurious {r.spurious_hits:g}"
                parts.append(s)
        agree = bool((g.agree == True).all())  # noqa: E712
        parts.append(f"matches {int(g.matches.iloc[0])} agree {agree}")
        print("  " + " | ".join(parts))
    for dataset in ("dna", "english"):
        for method in ("kmp", "rk", "sa", "st"):
            g = ds[(ds.dataset == dataset) & (ds.method == method)].sort_values("n")
            if len(g) >= 2:
                print(f"  {dataset:<8} {method}: sizes {list(g.n)}; latency slope {loglog_slope(g.n, g.mean_query_s):.3f}"
                      + (f"; build slope {loglog_slope(g.n, g.preprocess_s):.3f}" if g.preprocess_s.notna().all() else ""))
        g = ds[ds.dataset == dataset]
        for n, h in g.groupby("n"):
            kmp = h[h.method == "kmp"].mean_query_s.iloc[0]
            ratios = ", ".join(f"{m} {kmp / h[h.method == m].mean_query_s.iloc[0]:,.1f}x"
                               for m in ("rk", "sa", "st") if (h.method == m).any())
            print(f"  {dataset:<8} n={n:,}: KMP time / method time: {ratios}")
    for (dataset, n), h in ds.groupby(["dataset", "n"]):
        q = h.set_index("method").mean_query_s
        parts = [f"RK / KMP {q['rk'] / q['kmp']:.2f}x"]
        if "st" in q and "sa" in q:
            parts.append(f"SA / ST query {q['sa'] / q['st']:.2f}x")
        print(f"  {dataset:<8} n={n:,}: " + "; ".join(parts))
    print(f"  all dataset_sizes rows agree with KMP: {bool((ds.agree == True).all())}")  # noqa: E712

    # ------------------------------------------------------------------
    header("2. Break-even on 1 MB of DNA (from experiment 1)")
    be = ds[(ds.dataset == "dna") & (ds.n == 1_000_000)].set_index("method")
    if len(be):
        kmp = be.loc["kmp", "mean_query_s"]
        print(f"  KMP mean query {kmp:.4f} s")
        for method in ("sa", "st"):
            if method in be.index:
                pre, q = be.loc[method, "preprocess_s"], be.loc[method, "mean_query_s"]
                cross = math.ceil(pre / (kmp - q)) if kmp > q else None
                print(f"  {method}: build {pre:.3f} s, mean query {q * 1e3:.4f} ms, break-even {cross} queries; "
                      f"at 200 queries {pre + 200 * q:.2f} s against KMP {200 * kmp:.2f} s")

    # ------------------------------------------------------------------
    header("3. Rabin-Karp collisions (100 KB random lowercase, m = 8, per pattern)")
    for name in ("rk_moduli", "rk_bases"):
        for r in t[t.experiment == name].itertuples():
            print(f"  {name:<9} base {int(r.base):<6} q {int(r.modulus):<26,} spurious {r.spurious_hits:12,.2f}  "
                  f"expected {r.expected_spurious:.4g}  ratio {r.spurious_hits / r.expected_spurious if r.expected_spurious else float('nan'):.3f}  "
                  f"query {r.mean_query_s * 1e3:.2f} ms  agree {r.agree}")

    # ------------------------------------------------------------------
    header("4. Worst case")
    for r in t[t.experiment == "worst_case"].itertuples():
        print(f"  {r.dataset:<22} {r.method:<6} n {int(r.n):,} m {int(r.m)} time {r.mean_query_s:.4f} s "
              f"comparisons {int(r.comparisons):,} ({r.comparisons / r.n:.2f} per char) matches {int(r.matches):,} agree {r.agree}")

    # ------------------------------------------------------------------
    header("5. Memory (peak while preprocessing, KiB)")
    mem = t[t.experiment == "memory"]
    for n, g in mem.groupby("n"):
        peaks = g.set_index("method").peak_mem_kb
        line = "  ".join(f"{m} {v:,.1f} KiB ({v * 1024 / n:.1f} B/char)" for m, v in peaks.items())
        extra = f"  ST/SA {peaks['st'] / peaks['sa']:.2f}x" if "st" in peaks and "sa" in peaks else ""
        print(f"  n={n:>9,} {line}{extra}")

    # ------------------------------------------------------------------
    header("6. LCP statistics")
    for r in t[t.experiment == "lcp_stats"].itertuples():
        print(f"  {r.dataset:<9} n={int(r.n):>9,} mean LCP {r.mean_lcp:12,.3f} max {int(r.max_lcp):>9,} "
              f"distinct {int(r.distinct_substrings):>16,} ({int(r.distinct_substrings) / (r.n * (r.n + 1) / 2):.4f} of n(n+1)/2) "
              f"LRS {int(r.lrs_length):,} build {r.preprocess_s:.3f} s")

    # ------------------------------------------------------------------
    header("6b. Why 100 MB was not run (arithmetic on the measured rates above)")
    top = ds[(ds.n == ds.n.max())]
    for dataset in ("dna", "english"):
        r = top[(top.dataset == dataset) & (top.method == "sa")]
        if len(r):
            r = r.iloc[0]
            slope = loglog_slope(ds[(ds.dataset == dataset) & (ds.method == "sa")].n,
                                 ds[(ds.dataset == dataset) & (ds.method == "sa")].preprocess_s)
            est = r.preprocess_s * (100_000_000 / r.n) ** slope
            print(f"  {dataset}: suffix array build at {int(r.n):,} took {r.preprocess_s:.2f} s; at the fitted slope "
                  f"{slope:.3f}, 100 MB would take about {est:,.0f} s")
    big = mem[mem.n == mem.n.max()].set_index("method").peak_mem_kb
    if "st" in big:
        per = big["st"] * 1024 / mem.n.max()
        print(f"  suffix tree memory {per:.1f} B/char -> about {per * 100_000_000 / 1e9:.1f} GB at 100 MB; "
              f"suffix array {big['sa'] * 1024 / mem.n.max():.1f} B/char -> about "
              f"{big['sa'] * 1024 / mem.n.max() * 100_000_000 / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    header("7. Size caps")
    if caps.empty:
        print("  none")
    for r in caps.itertuples():
        print(f"  {r.method} / {r.dataset} / n={int(r.n):,}: {r.reason}")

    # ------------------------------------------------------------------
    header("8. Methodology facts")
    print(f"  rows {len(t)}; experiments {sorted(t.experiment.unique())}; disagreements {int((t.agree == False).sum())}")  # noqa: E712
    print(f"  distinct seeds {t.seed.replace('', np.nan).dropna().nunique()}")
    print(f"  queries per size: " + ", ".join(f"{n:,}: {int(q)}" for n, q in ds.groupby('n').queries.first().items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
