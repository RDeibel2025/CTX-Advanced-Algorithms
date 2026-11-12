#!/usr/bin/env python3
"""Print every number the Week 12 report quotes, straight from the benchmark CSV.

Run from the repository root after ``benchmarks/week12_benchmark.py``::

    python tools/week12_facts.py

Reads ``benchmarks/results/week12/benchmark.csv``. If a benchmark number in
``analysis/week12_report.md`` is not printed here, it should not be in the
report.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results", "week12")
EPS = float(np.finfo(float).eps)


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def slope(ns, values) -> float:
    return float(np.polyfit(np.log2(np.asarray(ns, dtype=float)), np.log2(np.asarray(values, dtype=float)), 1)[0])


def main() -> int:
    t = pd.read_csv(os.path.join(RESULTS, "benchmark.csv"))

    # ------------------------------------------------------------------
    header("1. FFT runtime")
    f = t[(t.experiment == "fft_runtime") & (t.capped != True)]  # noqa: E712
    for n, g in f.groupby("n"):
        parts = [f"N=2^{int(math.log2(n)):<2}"]
        for method in ("dft", "fft_recursive", "fft_iterative", "numpy_reference"):
            r = g[g.method == method]
            if len(r):
                r = r.iloc[0]
                s = f"{method} {r.mean_time_s * 1e3:10.3f} ms"
                if pd.notna(r.max_rel_error):
                    s += f" err {r.max_rel_error:.1e}"
                parts.append(s)
        print("  " + " | ".join(parts))
    for method in ("dft", "fft_recursive", "fft_iterative", "numpy_reference"):
        g = f[f.method == method].sort_values("n")
        if len(g) >= 3:
            tail = g.tail(5)
            print(f"  {method}: N 2^{int(math.log2(g.n.min()))}..2^{int(math.log2(g.n.max()))}; slope over five largest "
                  f"{slope(tail.n, tail.mean_time_s):.3f}; at largest {g.mean_time_s.iloc[-1]:.4f} s; "
                  f"max rel error {g.max_rel_error.max():.2e}" if g.max_rel_error.notna().any() else
                  f"  {method}: slope {slope(tail.n, tail.mean_time_s):.3f}; at largest {g.mean_time_s.iloc[-1]:.4f} s")
    p = f.pivot_table(index="n", columns="method", values="mean_time_s")
    for n in (4096, 2 ** 20):
        if n in p.index:
            row = p.loc[n]
            ratios = []
            if pd.notna(row.get("dft")):
                ratios.append(f"dft/iterative {row['dft'] / row['fft_iterative']:,.1f}x")
            ratios.append(f"recursive/iterative {row['fft_recursive'] / row['fft_iterative']:.2f}x")
            ratios.append(f"iterative/numpy {row['fft_iterative'] / row['numpy_reference']:.1f}x")
            print(f"  N={n:,}: " + "; ".join(ratios))

    # ------------------------------------------------------------------
    header("2. FFT round-trip error")
    r = t[t.experiment == "fft_roundtrip"]
    for method, g in r.groupby("method"):
        g = g.sort_values("n")
        ratio = g.max_rel_error / (EPS * np.log2(g.n))
        print(f"  {method:<16} error {g.max_rel_error.min():.2e} to {g.max_rel_error.max():.2e}; at 2^20 "
              f"{g.max_rel_error.iloc[-1]:.2e}; error / (eps log2 N) {ratio.min():.2f} to {ratio.max():.2f}")
    print(f"  eps = {EPS:.3e}; eps * log2(2^20) = {EPS * 20:.3e}")

    # ------------------------------------------------------------------
    header("3. Matrix multiply")
    m = t[t.experiment == "matrix_runtime"]
    for n, g in m.groupby("n"):
        parts = [f"n={n:<5}"]
        for method in ("classical", "blocked", "strassen", "numpy_reference"):
            row = g[g.method == method]
            if len(row):
                row = row.iloc[0]
                if row.capped == True:  # noqa: E712
                    parts.append(f"{method} capped")
                else:
                    s = f"{method} {row.mean_time_s:9.4f} s (runs {int(row.runs)})"
                    if pd.notna(row.max_rel_error):
                        s += f" err {row.max_rel_error:.1e}"
                    parts.append(s)
        print("  " + " | ".join(parts))
    ok = m[m.capped != True]  # noqa: E712
    for method in ("classical", "blocked", "strassen", "numpy_reference"):
        g = ok[ok.method == method].sort_values("n")
        if len(g) >= 3:
            tail = g.tail(3)
            print(f"  {method}: largest n {int(g.n.max())} at {g.mean_time_s.iloc[-1]:.4f} s; slope over three largest "
                  f"{slope(tail.n, tail.mean_time_s):.3f}; max rel error {g.max_rel_error.max():.1e}"
                  if g.max_rel_error.notna().any() else
                  f"  {method}: largest n {int(g.n.max())} at {g.mean_time_s.iloc[-1]:.5f} s; slope {slope(tail.n, tail.mean_time_s):.3f}")
    pm = ok.pivot_table(index="n", columns="method", values="mean_time_s")
    for n, row in pm.iterrows():
        bits = []
        if pd.notna(row.get("classical")) and pd.notna(row.get("blocked")):
            bits.append(f"classical/blocked {row['classical'] / row['blocked']:.1f}x")
        if pd.notna(row.get("strassen")) and pd.notna(row.get("blocked")):
            bits.append(f"strassen/blocked {row['strassen'] / row['blocked']:.2f}x")
        if pd.notna(row.get("blocked")):
            bits.append(f"blocked/numpy {row['blocked'] / row['numpy_reference']:,.0f}x")
        print(f"  n={n}: " + "; ".join(bits))
    caps = m[m.capped == True]  # noqa: E712
    print("  capped: " + (", ".join(f"{r.method} n={int(r.n)}" for r in caps.itertuples()) or "none"))

    # ------------------------------------------------------------------
    header("4. Stability (Hilbert)")
    c = t[t.experiment == "stability_cond"].pivot_table(index="n", columns="method", values="cond")
    for n, row in c.iterrows():
        print(f"  n={n:<3} cond1 ours {row['cond_estimate_1']:.4e} numpy {row['numpy_cond_1']:.4e} "
              f"(rel diff {abs(row['cond_estimate_1'] - row['numpy_cond_1']) / row['numpy_cond_1']:.1e}) | "
              f"cond2 ours {row['cond2_power']:.4e} numpy {row['numpy_cond_2']:.4e} "
              f"(rel diff {abs(row['cond2_power'] - row['numpy_cond_2']) / row['numpy_cond_2']:.1e})")
    s = t[t.experiment == "stability_solve"].pivot_table(index="n", columns="method", values="max_rel_error")
    cond2 = c["numpy_cond_2"]
    for n, row in s.iterrows():
        print(f"  n={n:<3} solve error pivot {row['lu_pivot']:.2e} no pivot {row['lu_no_pivot']:.2e} "
              f"cond*eps {cond2.loc[n] * EPS:.2e} error/(cond*eps) {row['lu_pivot'] / (cond2.loc[n] * EPS):.2f}")
    # Not in the CSV: the unit-test example of a tiny pivot, computed here by the package itself.
    sys.path.insert(0, REPO_ROOT)
    from src.fft_analyzer.core.matrix import solve
    tiny = [[1e-20, 1], [1, 1]]
    print(f"  tiny pivot [[1e-20, 1], [1, 1]] x = [1, 2]: no pivot {solve(tiny, [1, 2], pivot=False).tolist()}, "
          f"pivot {solve(tiny, [1, 2], pivot=True).tolist()}")
    o = t[t.experiment == "stability_orth"].pivot_table(index="n", columns="method", values="orth_loss")
    for n, row in o.iterrows():
        print(f"  n={n:<3} orth loss CGS {row['cgs']:.2e} MGS {row['mgs']:.2e} Householder {row['householder']:.2e}"
              f" | MGS/(cond*eps) {row['mgs'] / (cond2.loc[n] * EPS):.2f}")

    # ------------------------------------------------------------------
    header("5. Compression (synthetic 256 x 256 scene)")
    for r in t[t.experiment == "compression"].sort_values("quality").itertuples():
        print(f"  quality {int(r.quality):<3} PSNR {r.psnr_db:6.2f} dB  nonzero {r.nonzero_ratio:.4f} "
              f"({1 / r.nonzero_ratio:.1f}x fewer coefficients)  time {r.mean_time_s:.3f} s")
    comp = t[t.experiment == "compression"].sort_values("quality")
    for a, b in zip(comp.itertuples(), list(comp.itertuples())[1:]):
        print(f"  quality {int(a.quality)} -> {int(b.quality)}: PSNR +{b.psnr_db - a.psnr_db:.2f} dB, "
              f"nonzero x{b.nonzero_ratio / a.nonzero_ratio:.2f}")
    print(f"  quality 75 -> 95: PSNR +{comp.psnr_db.iloc[-1] - comp[comp.quality == 75].psnr_db.iloc[0]:.2f} dB")

    # ------------------------------------------------------------------
    header("6. Filters (two-tone signal, cutoff 80 Hz)")
    for r in t[t.experiment == "filters"].itertuples():
        print(f"  {r.method:<14} energy left {r.energy_left:.3e}")

    # ------------------------------------------------------------------
    header("7. Methodology facts")
    print(f"  rows {len(t)}; experiments {sorted(t.experiment.unique())}; capped rows {int((t.capped == True).sum())}")  # noqa: E712
    print(f"  distinct seeds {t.seed.dropna().nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
