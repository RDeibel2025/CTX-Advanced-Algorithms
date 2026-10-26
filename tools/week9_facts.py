#!/usr/bin/env python3
"""Print every number the Week 9 report quotes, straight from the benchmark CSVs.

Run from the repository root after ``benchmarks/week9_approx_benchmark.py``::

    python tools/week9_facts.py

Reads ``benchmarks/results/week9/comparison_table.csv`` and
``instances.csv``. If a number in ``analysis/week9_report.md`` is not printed
here, it should not be in the report.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results", "week9")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def span(series: pd.Series, digits: int = 3) -> str:
    """'mean (min-max)' of a series, or 'n/a' when it is empty."""
    s = series.dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.{digits}f} ({s.min():.{digits}f}-{s.max():.{digits}f})"


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "comparison_table.csv"))
    inst = pd.read_csv(os.path.join(RESULTS, "instances.csv"))

    # ------------------------------------------------------------------
    header("1. Vertex cover (G(n, p), average degree 4)")
    vc = inst[inst.problem == "vertex_cover"]
    for n, g in vc.groupby("n"):
        ex = g[g.method == "exact"]
        two = g[g.method == "two_approx"]
        lpr = g[g.method == "lp_rounding"]
        line = f"  n={n:<5}"
        if len(ex):
            line += f" OPT {ex.cost.mean():.1f}"
        line += (f" | 2-approx size {two.cost.mean():.1f} to-OPT {span(two.ratio_to_opt)} "
                 f"to-LP {span(two.ratio_to_lb)} | LP-round size {lpr.cost.mean():.1f} "
                 f"to-OPT {span(lpr.ratio_to_opt)} to-LP {span(lpr.ratio_to_lb)} | LP {two.lower_bound.mean():.2f} "
                 f"matching {two.matching_bound.mean():.1f}")
        if len(ex):
            line += (f" | LP/OPT {(ex.lower_bound / ex.cost).mean():.3f} "
                     f"matching/OPT {(two.matching_bound.values / ex.cost.values).mean():.3f}")
        line += (f" | time exact {ex.time_s.mean():.5f} s" if len(ex) else "") + \
                f" 2-approx {two.time_s.mean():.6f} s LP-round {lpr.time_s.mean():.5f} s"
        print(line)
    for method in ("two_approx", "lp_rounding"):
        m = vc[(vc.method == method) & vc.ratio_to_opt.notna()]
        print(f"  {method}: ratio to OPT over {len(m)} graphs {span(m.ratio_to_opt)}; optimal on "
              f"{int((m.ratio_to_opt < 1 + 1e-9).sum())}; ratio to LP over all {int((vc.method == method).sum())} "
              f"graphs {span(vc[vc.method == method].ratio_to_lb)}")

    # ------------------------------------------------------------------
    header("2. Set cover (density 0.1, sets = n/2, unit-free costs in [1, 10])")
    sc = inst[inst.problem == "set_cover"]
    for n, g in sc.groupby("n"):
        ex = g[g.method == "exact"]
        print(f"  n={n:<4} OPT {ex.cost.mean():.2f}  LP {ex.lower_bound.mean():.2f}  LP/OPT "
              f"{(ex.lower_bound / ex.cost).mean():.3f}  exact time {ex.time_s.mean():.4f} s")
        for method in ("greedy", "deterministic_rounding", "randomized_rounding"):
            m = g[g.method == method]
            extra = ""
            if method == "deterministic_rounding":
                extra = f" f mean {m.proven_bound.mean():.2f} (max {m.proven_bound.max():.0f})"
            if method == "greedy":
                extra = f" H(d) mean {m.proven_bound.mean():.2f}"
            if method == "randomized_rounding":
                extra = (f" within-instance sd {m.cost_std.mean():.2f} (cv {(m.cost_std / m.cost).mean():.3f}),"
                         f" min {m.cost_min.min():.2f} max {m.cost_max.max():.2f}, runs {int(m.runs.sum())},"
                         f" fix-up rate {m.fixup_rate.mean():.3f}")
            print(f"     {method:<23} cost {m.cost.mean():8.2f} to-OPT {span(m.ratio_to_opt)} "
                  f"to-LP {span(m.ratio_to_lb)} time {m.time_s.mean():.5f} s{extra}")
    for method in ("greedy", "deterministic_rounding", "randomized_rounding"):
        m = sc[sc.method == method]
        print(f"  {method}: ratio to OPT over {len(m)} instances {span(m.ratio_to_opt)}; optimal on "
              f"{int((m.ratio_to_opt < 1 + 1e-9).sum())}")
    rr = sc[sc.method == "randomized_rounding"]
    print(f"  randomized rounding: mean within-instance cv {(rr.cost_std / rr.cost).mean():.3f}; "
          f"worst single run / OPT {(rr.cost_max / rr.optimal).max():.3f}; best {(rr.cost_min / rr.optimal).min():.3f}; "
          f"fix-up rate overall {rr.fixup_rate.mean():.3f} (max {rr.fixup_rate.max():.3f})")

    # ------------------------------------------------------------------
    header("3. Facility location (greedy against the MILP optimum)")
    fl = inst[inst.problem == "facility_location"]
    for n, g in fl.groupby("n"):
        gr = g[g.method == "facility_greedy"]
        ex = g[g.method == "exact"]
        print(f"  clients={n:<4} {g.param.iloc[0]}: OPT {ex.cost.mean():.1f} greedy {gr.cost.mean():.1f} "
              f"to-OPT {span(gr.ratio_to_opt, 4)} optimal on {int((gr.ratio_to_opt < 1 + 1e-9).sum())}/{len(gr)}; "
              f"time MILP {ex.time_s.mean():.4f} s greedy {gr.time_s.mean():.5f} s")
    gr = fl[fl.method == "facility_greedy"]
    print(f"  overall greedy to-OPT {span(gr.ratio_to_opt, 4)}; optimal on "
          f"{int((gr.ratio_to_opt < 1 + 1e-9).sum())}/{len(gr)}")

    # ------------------------------------------------------------------
    header("4. Metric TSP (Euclidean)")
    tsp = inst[inst.problem == "metric_tsp"]
    for n, g in tsp.groupby("n"):
        ex = g[g.method == "exact"]
        parts = [f"n={n:<4}"]
        if len(ex):
            parts.append(f"MST/OPT {(ex.lower_bound / ex.cost).mean():.3f} HK {ex.time_s.mean():.5f} s")
        for method in ("mst_doubling", "christofides_networkx", "nearest_neighbor"):
            m = g[g.method == method]
            s = f"{method} "
            if m.ratio_to_opt.notna().any():
                s += f"to-OPT {span(m.ratio_to_opt)} "
            s += f"to-MST {span(m.ratio_to_lb)} {m.time_s.mean():.5f} s"
            parts.append(s)
        print("  " + " | ".join(parts))
    for method in ("mst_doubling", "christofides_networkx", "nearest_neighbor"):
        m = tsp[(tsp.method == method) & tsp.ratio_to_opt.notna()]
        print(f"  {method}: to OPT over {len(m)} instances {span(m.ratio_to_opt)}, optimal on "
              f"{int((m.ratio_to_opt < 1 + 1e-9).sum())}; to MST over all {span(tsp[tsp.method == method].ratio_to_lb)}")
    ex = tsp[tsp.method == "exact"]
    print(f"  MST / OPT over {len(ex)} instances: {span(ex.lower_bound / ex.cost)}")

    # ------------------------------------------------------------------
    header("5. Max-Cut (G(n, 0.3); maximization, so ratios are <= 1)")
    mc = inst[inst.problem == "maxcut"]
    for n, g in mc.groupby("n"):
        ex = g[g.method == "exact"]
        rc = g[g.method == "random_cut"]
        ls = g[g.method == "local_search"]
        parts = [f"n={n:<4} |E| {rc.lower_bound.mean():.1f}"]
        if len(ex):
            parts.append(f"OPT {ex.cost.mean():.1f} OPT/|E| {(ex.cost / ex.lower_bound).mean():.3f} "
                         f"exact {ex.time_s.mean():.4f} s")
        parts.append(f"random mean {rc.cost.mean():.1f} sd {rc.cost_std.mean():.2f} "
                     f"[{rc.cost_min.min():.0f}-{rc.cost_max.max():.0f}] "
                     + (f"to-OPT {span(rc.ratio_to_opt)} " if rc.ratio_to_opt.notna().any() else "")
                     + f"to-|E| {span(rc.ratio_to_lb)} {rc.time_s.mean() * 1e6:.1f} us/run")
        parts.append(f"local best {ls.cost.mean():.1f} "
                     + (f"to-OPT {span(ls.ratio_to_opt)} " if ls.ratio_to_opt.notna().any() else "")
                     + f"to-|E| {span(ls.ratio_to_lb)} {ls.time_s.mean():.4f} s (all starts)")
        print("  " + " | ".join(parts))
    rc = mc[(mc.method == "random_cut") & mc.ratio_to_opt.notna()]
    ls = mc[(mc.method == "local_search") & mc.ratio_to_opt.notna()]
    print(f"  random cut mean / OPT over {len(rc)} graphs {span(rc.ratio_to_opt)}; "
          f"worst single run / OPT {(rc.cost_min / rc.optimal).min():.3f}; "
          f"mean / (|E|/2) {(rc.cost / (rc.lower_bound / 2)).mean():.4f}")
    print(f"  local search best / OPT over {len(ls)} graphs {span(ls.ratio_to_opt)}; optimal on "
          f"{int((ls.ratio_to_opt > 1 - 1e-9).sum())}")

    # ------------------------------------------------------------------
    header("6. Derived ratios quoted in the Discussion")
    big = vc[vc.n == vc.n.max()]
    lp_t = big[big.method == "lp_rounding"].time_s.mean()
    two_t = big[big.method == "two_approx"].time_s.mean()
    print(f"  vertex cover n={int(vc.n.max())}: LP rounding time / 2-approx time = {lp_t / two_t:.1f}")
    big = tsp[tsp.n == tsp.n.max()]
    chr_t = big[big.method == "christofides_networkx"].time_s.mean()
    mst_t = big[big.method == "mst_doubling"].time_s.mean()
    print(f"  TSP n={int(tsp.n.max())}: Christofides time / MST doubling time = {chr_t / mst_t:.1f}")

    # ------------------------------------------------------------------
    header("7. Methodology facts")
    print(f"  comparison_table rows {len(table)}; kinds {sorted(table.kind.unique())}; "
          f"lower-bound kinds {sorted(table.lower_bound_kind.dropna().unique())}")
    print(f"  instance records {len(inst)}; distinct seeds {inst.seed.dropna().nunique()}; "
          f"total method runs {int(inst.runs.sum())}")
    for problem in ("vertex_cover", "set_cover", "facility_location", "metric_tsp", "maxcut"):
        t = table[table.problem == problem]
        print(f"  {problem:<18} rows {len(t):<3} n {int(t.n.min())}-{int(t.n.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
