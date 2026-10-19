#!/usr/bin/env python3
"""Print every number the Week 8 report quotes, straight from the benchmark CSVs.

Run from the repository root after ``benchmarks/week8_complexity_benchmark.py``::

    python tools/week8_facts.py

Reads ``benchmarks/results/week8/comparison_table.csv``, ``instances.csv``
and ``sweep_limits.csv``. Nothing here is typed in by hand: if a number in
``analysis/week8_report.md`` is not printed by this script, it should not be
in the report.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.probability_analysis import wilson_interval  # noqa: E402

RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results", "week8")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def per_item_factor(ns, values) -> float:
    """Least-squares growth factor per added item: exp(slope of ln(value) on n)."""
    ns = np.asarray(ns, dtype=float)
    vals = np.asarray(values, dtype=float)
    keep = vals > 0
    slope = np.polyfit(ns[keep], np.log(vals[keep]), 1)[0]
    return float(math.exp(slope))


def us(seconds: float) -> str:
    return f"{seconds * 1e6:,.1f} us"


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "comparison_table.csv"))
    inst = pd.read_csv(os.path.join(RESULTS, "instances.csv"))
    limits = pd.read_csv(os.path.join(RESULTS, "sweep_limits.csv"))
    for frame in (table, inst):
        frame["param"] = frame["param"].astype(str).replace("nan", "")

    # ------------------------------------------------------------------
    header("1. Subset sum")
    worst = inst[(inst.problem == "subset_sum") & (inst.instance_family == "worst_case")]
    bt = worst[worst.method == "backtracking"]
    g = bt.groupby("n").agg(t=("time_s", "mean"), sd=("time_s", "std"), tmax=("time_s", "max"),
                            nodes=("nodes", "mean"), nmin=("nodes", "min"), nmax=("nodes", "max"),
                            runs=("time_s", "size"))
    for n, r in g.iterrows():
        print(f"  worst n={n:<3} backtracking {r.t:12.6f} s (sd {r.sd:.6f}, max {r.tmax:.4f}) "
              f"nodes {r.nodes:14,.0f} [{r.nmin:,.0f}-{r.nmax:,.0f}] runs {r.runs}")
    print(f"  largest worst-case n: {g.index.max()}  mean time there {g.t.iloc[-1]:.3f} s, "
          f"nodes {g.nodes.iloc[-1]:,.0f}")
    print(f"  nodes growth per added item (fit over all n): {per_item_factor(g.index, g.nodes):.4f} "
          f"(2^n reference 2.0); per 2 items {per_item_factor(g.index, g.nodes) ** 2:.3f}")
    print(f"  time growth per added item (fit, n >= 16): "
          f"{per_item_factor(g.index[g.index >= 16], g.t[g.index >= 16]):.4f}")
    ratios = g.nodes / g.nodes.shift(1)
    print("  nodes ratio between consecutive sizes: "
          + ", ".join(f"{n}:{v:.2f}" for n, v in ratios.dropna().items()))
    for method in ("greedy", "random_restart"):
        m = worst[worst.method == method].groupby("n").agg(t=("time_s", "mean"), gap=("solution_value", "mean"),
                                                         gmin=("solution_value", "min"))
        print(f"  worst {method:<14} time {us(m.t.min())} to {us(m.t.max())}; "
              f"mean gap range {m.gap.min():,.1f}-{m.gap.max():,.1f}; smallest gap seen {m.gmin.min():,.0f}")
    sat = inst[(inst.problem == "subset_sum") & (inst.instance_family == "satisfiable")]
    sbt = sat[sat.method == "backtracking"]
    print(f"  satisfiable family: n {sbt.n.min()}-{sbt.n.max()}, {len(sbt)} instances; backtracking found "
          f"every one ({int(sbt.success.sum())}/{len(sbt)}); mean time {us(sbt.time_s.min())} to "
          f"{sbt.groupby('n').time_s.mean().max():.4f} s; max nodes {sbt.nodes.max():,.0f}")
    vt = sbt.verify_time_s
    print(f"  verifier (sub-multiset + sum) on backtracking certificates: {us(vt.min())} to {us(vt.max())}")
    for method in ("greedy", "random_restart"):
        m = sat[sat.method == method]
        hits = int(m.success.sum())
        lo, hi = wilson_interval(hits, len(m))
        print(f"  satisfiable {method:<14} exact hits {hits}/{len(m)} = {hits / len(m):.3f} "
              f"[{lo:.3f}, {hi:.3f}]; miss rate {1 - hits / len(m):.3f}; mean gap {m.solution_value.mean():,.1f}; "
              f"median gap {m.solution_value.median():,.1f}; max gap {m.solution_value.max():,.0f}; "
              f"time {us(m.time_s.min())} to {us(m.time_s.max())}")

    # ------------------------------------------------------------------
    header("2. SAT")
    sat_i = inst[inst.problem == "sat"]
    for ratio in ("4.26", "2.0"):
        r = sat_i[sat_i.param == ratio]
        print(f"  --- ratio {ratio} ---")
        for n, grp in r.groupby("n"):
            truth = grp[grp.method == "sat_dpll"].optimal_value
            parts = [f"n={n:<3} sat {truth.mean():.1f} ({int(truth.sum())}/{len(truth)})"]
            for method in ("sat_backtracking", "sat_dpll", "walksat"):
                mm = grp[grp.method == method]
                if len(mm):
                    s = f"{method} {mm.time_s.mean():.5f} s"
                    if method != "walksat":
                        s += f" nodes {mm.nodes.mean():,.0f}"
                    parts.append(s)
            print("  " + " | ".join(parts))
        for method in ("sat_backtracking", "sat_dpll"):
            mm = r[r.method == method].groupby("n").agg(t=("time_s", "mean"), nodes=("nodes", "mean"))
            if len(mm) >= 3:
                big = mm[mm.index >= 15] if (mm.index >= 15).sum() >= 3 else mm
                print(f"  {method} ratio {ratio}: n up to {mm.index.max()}; nodes growth per variable "
                      f"(fit n >= {big.index.min()}): {per_item_factor(big.index, big.nodes):.4f}; "
                      f"time growth {per_item_factor(big.index, big.t):.4f}; largest-n mean time {mm.t.iloc[-1]:.4f} s, "
                      f"nodes {mm.nodes.iloc[-1]:,.0f}")
        w = r[r.method == "walksat"]
        satisfiable = w[w.optimal_value == 1]
        hits = int(satisfiable.success.sum())
        if len(satisfiable):
            lo, hi = wilson_interval(hits, len(satisfiable))
            print(f"  walksat ratio {ratio}: solved {hits}/{len(satisfiable)} satisfiable formulas = "
                  f"{hits / len(satisfiable):.3f} [{lo:.3f}, {hi:.3f}]; returned on unsatisfiable: "
                  f"{int(w[w.optimal_value == 0].success.sum())}/{int((w.optimal_value == 0).sum())}")
        v = r[r.verify_time_s.notna()].verify_time_s
        print(f"  verifier is_satisfying ratio {ratio}: {us(v.min())} to {us(v.max())}")
        total_sat = r[r.method == "sat_dpll"].optimal_value
        print(f"  satisfiable overall at ratio {ratio}: {int(total_sat.sum())}/{len(total_sat)} "
              f"= {total_sat.mean():.3f}")

    # ------------------------------------------------------------------
    header("3. Vertex cover")
    vc = inst[inst.problem == "vertex_cover"]
    for fam in ("path", "star", "k33"):
        a = vc[vc.instance_family == fam]
        sizes = {m: int(a[a.method == m].solution_value.iloc[0]) for m in a.method.unique()}
        print(f"  anchor {fam:<5} n={int(a.n.iloc[0])}: {sizes}  2-approx ratio "
              f"{float(a[a.method == 'two_approx'].ratio.iloc[0]):.3f}")
    gnp = vc[vc.instance_family == "gnp_avg_degree_4"]
    for n, grp in gnp.groupby("n"):
        parts = [f"n={n:<5}"]
        for method in ("exact_branching", "two_approx", "max_degree_greedy"):
            mm = grp[grp.method == method]
            if len(mm):
                s = f"{method} {mm.time_s.mean():.6f} s size {mm.solution_value.mean():.1f}"
                if mm.ratio.notna().any():
                    s += f" ratio {mm.ratio.mean():.3f} [{mm.ratio.min():.3f}-{mm.ratio.max():.3f}]"
                parts.append(s)
        print("  " + " | ".join(parts))
    ex = gnp[gnp.method == "exact_branching"].groupby("n").time_s.mean()
    print(f"  exact branching up to n={ex.index.max()}; mean time there {ex.iloc[-1]:.4f} s; "
          f"time growth per added vertex (fit n >= 20): {per_item_factor(ex.index[ex.index >= 20], ex[ex.index >= 20]):.4f}")
    two = gnp[(gnp.method == "two_approx") & gnp.ratio.notna()]
    grd = gnp[(gnp.method == "max_degree_greedy") & gnp.ratio.notna()]
    print(f"  2-approx ratio over {len(two)} graphs with an exact optimum: mean {two.ratio.mean():.3f}, "
          f"min {two.ratio.min():.3f}, max {two.ratio.max():.3f}; optimal in {int(two.success.sum())}")
    print(f"  max-degree greedy ratio over {len(grd)} graphs: mean {grd.ratio.mean():.3f}, "
          f"min {grd.ratio.min():.3f}, max {grd.ratio.max():.3f}; optimal in {int(grd.success.sum())}")
    big = gnp[gnp.n == gnp.n.max()]
    for method in ("two_approx", "max_degree_greedy"):
        mm = big[big.method == method]
        print(f"  n={int(gnp.n.max())} {method}: time {mm.time_s.mean():.4f} s, size {mm.solution_value.mean():.1f}, "
              f"verify {us(mm.verify_time_s.mean())}")
    sizes_big = big.pivot_table(index="rep", columns="method", values="solution_value")
    print(f"  n={int(gnp.n.max())} 2-approx / greedy size: "
          f"{(sizes_big['two_approx'] / sizes_big['max_degree_greedy']).mean():.3f}; "
          f"2-approx certified lower bound |M| = size/2 = {sizes_big['two_approx'].mean() / 2:.1f}; "
          f"greedy size / that bound = {sizes_big['max_degree_greedy'].mean() / (sizes_big['two_approx'].mean() / 2):.3f}")
    v = vc.verify_time_s
    print(f"  verifier is_vertex_cover: {us(v.min())} to {us(v.max())}")

    # ------------------------------------------------------------------
    header("4. TSP")
    tsp = inst[inst.problem == "tsp"]
    line = tsp[tsp.instance_family == "line_anchor"].iloc[0]
    print(f"  line anchor: NN {line.solution_value:g} vs optimum {line.optimal_value:g}, ratio {line.ratio:.4f}")
    eu = tsp[tsp.instance_family == "euclidean"]
    for n, grp in eu.groupby("n"):
        parts = [f"n={n:<4}"]
        for method in ("held_karp", "nearest_neighbor", "nn_two_opt"):
            mm = grp[grp.method == method]
            if len(mm):
                s = f"{method} {mm.time_s.mean():.6f} s"
                if mm.ratio.notna().any():
                    s += f" ratio {mm.ratio.mean():.3f} [{mm.ratio.min():.3f}-{mm.ratio.max():.3f}]"
                if method == "nearest_neighbor" and mm.nn_over_two_opt.notna().any():
                    s += f" NN/2opt {mm.nn_over_two_opt.mean():.3f}"
                parts.append(s)
        print("  " + " | ".join(parts))
    for method in ("nearest_neighbor", "nn_two_opt"):
        mm = eu[(eu.method == method) & eu.ratio.notna()]
        print(f"  {method} ratio over {len(mm)} instances with an optimum: mean {mm.ratio.mean():.3f}, "
              f"min {mm.ratio.min():.3f}, max {mm.ratio.max():.3f}; optimal in {int(mm.success.sum())}")
    hk = eu[eu.method == "held_karp"].groupby("n").time_s.mean()
    print(f"  Held-Karp time n={hk.index.min()}..{hk.index.max()}: {hk.iloc[0]:.6f} s to {hk.iloc[-1]:.4f} s; "
          f"growth per city (fit n >= 9) {per_item_factor(hk.index[hk.index >= 9], hk[hk.index >= 9]):.4f}")
    v = tsp.verify_time_s
    print(f"  verifier (permutation + tour cost): {us(v.min())} to {us(v.max())}")

    # ------------------------------------------------------------------
    header("5. Reduction 3-SAT -> vertex cover (ratio 2.0)")
    red = inst[inst.problem == "reduction"]
    for n, grp in red.groupby("n"):
        print(f"  n={n:<5} nodes {int(grp.nodes.iloc[0]):<6} edges {int(grp.solution_value.iloc[0]):<6} "
              f"k {int(grp.optimal_value.iloc[0]):<6} transform {grp.time_s.mean() * 1e3:.3f} ms "
              f"(us per node {grp.time_s.mean() / grp.nodes.iloc[0] * 1e6:.3f}) "
              f"round trip ok {int(grp.success.sum())}/{len(grp)} verify {us(grp.verify_time_s.mean())}")

    # ------------------------------------------------------------------
    header("6. Verification against search at the largest exact size")
    def pair(problem, method, fam=None, param=None):
        sel = inst[(inst.problem == problem) & (inst.method == method)]
        if fam:
            sel = sel[sel.instance_family == fam]
        if param:
            sel = sel[sel.param == param]
        sel = sel[sel.verify_time_s.notna()]
        n = sel.n.max()
        at = sel[sel.n == n]
        search = at.time_s.mean()
        check = at.verify_time_s.mean()
        print(f"  {problem:<12} {method:<16} n={n:<4} search {search:.6f} s  verify {us(check)}  "
              f"ratio {search / check:,.0f}x")
    pair("subset_sum", "backtracking", "satisfiable")
    pair("sat", "sat_backtracking", param="4.26")
    pair("sat", "sat_dpll", param="4.26")
    pair("vertex_cover", "exact_branching", "gnp_avg_degree_4")
    pair("tsp", "held_karp", "euclidean")
    wbt = worst[worst.method == "backtracking"]
    n = wbt.n.max()
    sv = sbt[sbt.n == sbt.n.max()].verify_time_s.mean()
    print(f"  subset_sum worst case n={n}: search {wbt[wbt.n == n].time_s.mean():.4f} s; a NO answer has "
          f"no certificate; the YES verifier at n={sbt.n.max()} takes {us(sv)}")
    v_all = inst.verify_time_s.dropna()
    print(f"  every verifier time in the study: {us(v_all.min())} to {us(v_all.max())} "
          f"({len(v_all)} certificates)")

    # ------------------------------------------------------------------
    header("7. Sweep limits (the 30-second rule)")
    if limits.empty:
        print("  none")
    for _, r in limits.iterrows():
        print(f"  {r.experiment} / {r.method}: n={r.n} next={r.next_n} max_run={r.max_run_s} "
              f"predicted={r.predicted_next_run_s} runs_kept={r.runs_kept}: {r.reason}")

    # ------------------------------------------------------------------
    header("8. Methodology facts")
    print(f"  comparison_table rows {len(table)}; kinds {sorted(table.kind.unique())}; "
          f"measurement {sorted(table.measurement.unique())}")
    print(f"  instance records {len(inst)}; distinct seeds {inst.seed.dropna().nunique()}")
    for problem in ("subset_sum", "sat", "vertex_cover", "tsp", "reduction"):
        t = table[table.problem == problem]
        print(f"  {problem:<12} rows {len(t):<4} n {int(t.n.min())}-{int(t.n.max())}  "
              f"runs per row {sorted(t.runs.unique().tolist())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
