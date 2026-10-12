#!/usr/bin/env python3
"""Print every figure analysis/week7_report.md quotes, computed from the CSVs.

As in Weeks 2 through 6: the report is prose under a word limit, so every
number it cites is computed here from ``benchmarks/results/`` instead of
being transcribed, and the report can be re-checked against a fresh run.

    python benchmarks/week7_randomized_benchmark.py
    python tools/week7_facts.py

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    t = pd.read_csv(os.path.join(RESULTS, "summary_table.csv"))
    t["ms"] = t.mean_time_s * 1e3
    t["std_ms"] = t.std_time_s * 1e3
    # param_value mixes numbers with labels such as "2 3 5 7", so it is read as text.
    t["pv"] = pd.to_numeric(t["param_value"], errors="coerce")

    heading("1. QuickSort on random input (mean +- std, cv)")
    rnd = t[t.experiment == "random_input_runtime"]
    for n in sorted(rnd.n.unique()):
        part = rnd[rnd.n == n].set_index("variant")
        print(f"  n={n:<7} " + "  ".join(
            f"{v[:14]} {part.loc[v].ms:8.3f}+-{part.loc[v].std_ms:6.3f} cv {part.loc[v].std_ms / part.loc[v].ms:.3f}"
            for v in part.index))
    top = rnd[rnd.n == rnd.n.max()].set_index("variant")
    for v in ("deterministic_last", "deterministic_median_of_three", "week2_quick_sort"):
        print(f"  at n={rnd.n.max():,}: randomized / {v} = {top.loc['randomized'].ms / top.loc[v].ms:.3f}x")
    print(f"  runs per point: {sorted(rnd.runs.unique())}")
    cvs = rnd[rnd.variant == "randomized"].set_index("n")
    print("  randomized cv by n: " + ", ".join(f"{n}:{r.std_ms / r.ms:.3f}" for n, r in cvs.iterrows()))

    heading("2. QuickSort on sorted input")
    srt = t[t.experiment == "sorted_input"]
    for n in sorted(srt.n.unique()):
        part = srt[srt.n == n].set_index("variant")
        last = part.loc["deterministic_last"]
        line = (f"  n={n:<7} last {last.ms:12,.3f} ms ({last.measurement}, {int(last.metric_value):,} comp)")
        if "randomized" in part.index:
            r = part.loc["randomized"]
            line += f" | randomized {r.ms:8.3f} ms ({int(r.metric_value):,} comp, E {r.theoretical_value:,.0f})"
            line += f" | last/randomized {last.ms / r.ms:,.0f}x"
        print(line)
    measured_last = srt[(srt.variant == "deterministic_last") & (srt.measurement == "measured")]
    big = measured_last[measured_last.n == measured_last.n.max()].iloc[0]
    print(f"  ns per comparison at n={big.n:,}: {big.mean_time_s * 1e9 / big.metric_value:.1f}; runs {int(big.runs)}")

    heading("3. Comparison-count distribution")
    dist = t[t.experiment == "comparison_distribution"]
    for n in sorted(dist.n.unique()):
        part = dist[dist.n == n].set_index("metric_name")
        mean, std = part.loc["comparisons_mean"], part.loc["comparisons_std"]
        print(f"  n={n:<6} {int(mean.runs)} seeds: mean {mean.metric_value:,.1f} vs E {mean.theoretical_value:,.1f} "
              f"({100 * (mean.metric_value / mean.theoretical_value - 1):+.2f}%), std {std.metric_value:,.1f} "
              f"vs 0.6483n {std.theoretical_value:,.1f} ({std.metric_value / std.theoretical_value:.3f}x), "
              f"std/mean {std.metric_value / mean.metric_value:.4f}")
    counts = pd.read_csv(os.path.join(RESULTS, "week7_comparison_counts.csv")).comparisons
    print(f"  n=1000 sample: min {counts.min():,} max {counts.max():,}; "
          f"fraction above E + 2 std: {(counts > 10985.9 + 2 * 648.3).mean():.3f}")

    heading("4. Selection")
    sel = t[(t.algorithm == "quickselect")]
    mom = t[t.algorithm == "median_of_medians"]
    for label in ("min", "median", "max"):
        for n in sorted(sel.n.unique()):
            part = sel[(sel.variant == label) & (sel.n == n)].set_index("metric_name")
            m = mom[(mom.variant == label) & (mom.n == n)].iloc[0]
            r = part.loc["success_rate"]
            print(f"  {label:<6} n={n:<7} QS {r.ms:7.3f}+-{r.std_ms:6.3f} ms (cv {r.std_ms / r.ms:.2f}) "
                  f"MoM {m.ms:7.3f}+-{m.std_ms:5.3f} (cv {m.std_ms / m.ms:.2f}) MoM/QS {m.ms / r.ms:.2f}x | "
                  f"comp/n {part.loc['comparisons_per_n_mean'].metric_value:.2f} "
                  f"(sd {part.loc['comparisons_per_n_std'].metric_value:.2f}) | depth "
                  f"{part.loc['depth_mean'].metric_value:.1f} max {part.loc['depth_max'].metric_value:.0f} "
                  f"(log2 n {math.log2(n):.1f}) | within4n {part.loc['within_4n_fraction'].metric_value:.2f} "
                  f"| success {r.metric_value:.0f} ({int(r.runs)} runs) | MoM comp/n {m.metric_value:.2f}")
    ratios = pd.read_csv(os.path.join(RESULTS, "week7_selection_ratios.csv")).comparisons_per_n
    print(f"  median at n=100,000: comp/n min {ratios.min():.2f} max {ratios.max():.2f}, "
          f"{(ratios > 4).sum()} of {len(ratios)} above 4")

    heading("5. Primality: false-positive rates (95% Wilson) against prediction")
    acc = t[t.experiment == "false_positive_rate"]
    for (alg, n), part in acc.groupby(["algorithm", "n"], sort=False):
        print(f"  {alg:<12} n={n:<8} " + "  ".join(
            f"k={int(r.pv)}: {r.metric_value:.5f} [{r.metric_ci_low:.5f},{r.metric_ci_high:.5f}] "
            f"pred {r.theoretical_value:.2e}" for r in part.sort_values("pv").itertuples()))
    f561 = acc[(acc.algorithm == "fermat_test") & (acc.n == 561)].set_index("pv")
    print(f"  Fermat on 561 per-round factor from k=1: {f561.loc[1].metric_value:.4f}; "
          f"(k=1)^5 = {f561.loc[1].metric_value ** 5:.4f} vs measured k=5 {f561.loc[5].metric_value:.4f}")
    mr = acc[acc.algorithm == "miller_rabin"]
    nonzero = mr[(mr.pv >= 5) & (mr.metric_value > 0)]
    print(f"  Miller-Rabin at k>=5: {len(mr[mr.pv >= 5])} points, nonzero at "
          + (", ".join(f"n={r.n} k={int(r.pv)} ({r.metric_value:.4f}, pred {r.theoretical_value:.2e})"
                       for r in nonzero.itertuples()) or "none"))
    zero = mr[(mr.pv >= 5) & (mr.metric_value == 0)]
    print(f"  zero-count MR points: {len(zero)}; their 95% upper bound {zero.metric_ci_high.max():.2e} "
          f"(rule of three {3 / zero.runs.max():.1e}) with {int(zero.runs.max()):,} trials each")
    total_mr = mr[mr.pv >= 5]
    print(f"  total MR trials at k>=5: {int(total_mr.runs.sum()):,}; false positives {int(round((total_mr.metric_value * total_mr.runs).sum()))}")

    heading("6. Primality speed and the adversarial number")
    sp = t[t.experiment == "speed"]
    for n in sorted(sp.n.unique()):
        part = sp[sp.n == n].set_index("variant")
        cells = []
        for v in ("trial_division", "fermat", "miller_rabin"):
            if v in part.index:
                r = part.loc[v]
                cells.append(f"{v} {r.mean_time_s * 1e6:13,.1f} us{' (projected)' if r.measurement == 'projected' else ''}")
        ratio = (part.loc["trial_division"].mean_time_s / part.loc["miller_rabin"].mean_time_s
                 if "trial_division" in part.index else float("nan"))
        print(f"  n={n:<26,} " + "  ".join(cells) + f"   TD/MR {ratio:,.2f}x")
    mr_rows = sp[sp.variant == "miller_rabin"].sort_values("n")
    print("  MR per-round cost (k=20): " + ", ".join(f"{n:.0e}:{r.mean_time_s * 1e6 / 20:.2f}us"
                                                    for n, r in zip(mr_rows.n, mr_rows.itertuples())))
    adv = t[t.experiment == "adversarial"]
    for r in adv.itertuples():
        extra = (f" [{r.metric_ci_low:.5f},{r.metric_ci_high:.5f}] bound {r.theoretical_value}"
                 if not pd.isna(r.metric_ci_low) else "")
        print(f"  {r.algorithm:<26} {r.param_name}={r.param_value}: {r.metric_name} {r.metric_value}{extra} "
              f"({'' if pd.isna(r.runs) else int(r.runs)} trials)")
    phi_fraction = (150 * 750 * 28350) / 4 / (3_215_031_751 - 3)
    print(f"  phi(n)/4 bound on the strong-liar fraction for 3,215,031,751: {phi_fraction:.4f}; "
          f"to the 5th power {phi_fraction ** 5:.2e}")

    heading("7. Bloom filter")
    load = t[(t.experiment == "load_factor_sweep") & (t.metric_name == "false_positive_rate")]
    inside = ((load.theoretical_value >= load.metric_ci_low) & (load.theoretical_value <= load.metric_ci_high))
    for r in load.itertuples():
        print(f"  load {r.pv:<6} n={r.n:<5} fp {r.metric_value:.5f} [{r.metric_ci_low:.5f},{r.metric_ci_high:.5f}] theory {r.theoretical_value:.5f}")
    print(f"  theory inside the 95% interval at {inside.sum()} of {len(load)} load points")
    fill = t[(t.experiment == "load_factor_sweep") & (t.metric_name == "fill_ratio")].set_index("n")
    k_load = 7
    for r in load.itertuples():
        if r.n in fill.index:
            f = fill.loc[r.n].metric_value
            print(f"    n={r.n:<5} this filter's fill {f:.4f}: fill^k {f ** k_load:.5f} vs measured "
                  f"{r.metric_value:.5f} vs average-filter theory {r.theoretical_value:.5f}")
    rel_fill = [abs(fill.loc[r.n].metric_value ** k_load / r.metric_value - 1)
                for r in load.itertuples() if r.n in fill.index and r.metric_value > 0.001]
    rel_theory = [abs(r.theoretical_value / r.metric_value - 1)
                  for r in load.itertuples() if r.metric_value > 0.001]
    print(f"  mean relative gap (points with fp > 0.001): fill^k {100 * np.mean(rel_fill):.1f}%, "
          f"theory {100 * np.mean(rel_theory):.1f}%")
    bits = t[(t.experiment == "bits_per_item_sweep") & (t.metric_name == "false_positive_rate")].sort_values("pv")
    inside_b = ((bits.theoretical_value >= bits.metric_ci_low) & (bits.theoretical_value <= bits.metric_ci_high))
    print(f"  theory inside the 95% interval at {inside_b.sum()} of {len(bits)} bits-per-item points")
    x = bits.pv.astype(float).to_numpy()
    slope_m = np.polyfit(x, np.log(bits.metric_value.to_numpy()), 1)[0]
    slope_t = np.polyfit(x, np.log(bits.theoretical_value.to_numpy()), 1)[0]
    print(f"  factor per extra bit per item: measured {math.exp(slope_m):.4f}, theoretical {math.exp(slope_t):.4f} "
          f"(ideal 0.5^ln2 = {0.5 ** math.log(2):.4f})")
    for b in (2, 8, 10, 20):
        r = bits[bits.pv == b].iloc[0]
        print(f"  {b} bits/item ({r.variant}): fp {r.metric_value:.6f} [{r.metric_ci_low:.6f},{r.metric_ci_high:.6f}] theory {r.theoretical_value:.6f}")
    mem = t[t.experiment == "memory"].iloc[0]
    print(f"  memory at n={mem.n:,}: Bloom {int(mem.metric_value):,} bytes vs Python set {int(mem.theoretical_value):,} "
          f"bytes ({mem.theoretical_value / mem.metric_value:.0f}x)")

    heading("8. MinHash")
    mh = t[(t.algorithm == "minhash") & (t.metric_name == "mean_abs_error")]
    by_k = mh.groupby("pv").agg(mae=("metric_value", "mean"), theory=("theoretical_value", "mean"))
    print("  MAE averaged over the 9 pairs: " + ", ".join(f"k={int(k)}: {r.mae:.4f} (theory {r.theory:.4f})"
                                                        for k, r in by_k.iterrows()))
    slope = np.polyfit(np.log(by_k.index.astype(float)), np.log(by_k.mae), 1)[0]
    print(f"  log-log slope of MAE against k: {slope:.3f} (theory -0.5)")
    print(f"  MAE k=16 / k=512: {by_k.mae.iloc[0] / by_k.mae.iloc[-1]:.2f}x (theory sqrt(32) = {math.sqrt(32):.2f}x)")
    for kind in ("simulated", "text"):
        part = mh[mh.input_kind == kind]
        print(f"  {kind}: {part.variant.nunique()} pairs, true J values "
              + ", ".join(sorted({v.split('true_j=')[1] for v in part.variant})))
    est = t[(t.algorithm == "minhash") & (t.metric_name == "estimate_mean") & (t.pv == 512)]
    print(f"  mean estimate at k=512 minus true J: max |bias| {np.max(np.abs(est.metric_value - est.theoretical_value)):.4f}")
    at512 = mh[mh.pv == 512]
    print(f"  at k=512: measured MAE / theoretical MAE ranges {(at512.metric_value / at512.theoretical_value).min():.2f}"
          f" to {(at512.metric_value / at512.theoretical_value).max():.2f}; seeds per point {int(mh.runs.max())}")

    heading("9. Methodology facts")
    print(f"  rows {len(t)}; categories {sorted(t.category.unique())}; projected rows "
          f"{int((t.measurement == 'projected').sum())}: " + ", ".join(
              f"{r.algorithm}/{r.variant}/n={r.n}" for r in t[t.measurement == 'projected'].itertuples()))
    print(f"  distinct seeds recorded: {t.seed.nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
