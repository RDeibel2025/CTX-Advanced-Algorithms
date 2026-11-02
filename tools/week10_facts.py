#!/usr/bin/env python3
"""Print every number the Week 10 report quotes, straight from the benchmark CSVs.

Run from the repository root after ``benchmarks/week10_flow_benchmark.py``::

    python tools/week10_facts.py

Reads ``flow_benchmark.csv``, ``instances.csv`` and ``size_caps.csv`` from
``benchmarks/results/week10/``. If a number in ``analysis/week10_report.md``
is not printed here, it should not be in the report.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results", "week10")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def loglog_slope(ns, values) -> float:
    """Least-squares slope of log(value) on log(n)."""
    return float(np.polyfit(np.log(np.asarray(ns, dtype=float)),
                            np.log(np.asarray(values, dtype=float)), 1)[0])


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "flow_benchmark.csv"))
    inst = pd.read_csv(os.path.join(RESULTS, "instances.csv"))
    caps = pd.read_csv(os.path.join(RESULTS, "size_caps.csv"))

    # ------------------------------------------------------------------
    header("1. Scaling, sparse (average degree 4)")
    sc = inst[inst.experiment == "scaling"]
    for n, g in sc.groupby("n"):
        parts = [f"n={n:<7} m {int(g.m.iloc[0]):<7} flow {g.max_flow.mean():>12,.1f}"]
        for solver in ("ek", "pr", "mcf"):
            mm = g[g.solver == solver]
            if len(mm):
                s = f"{solver} {mm.time_s.mean():9.4f} s (sd {mm.time_s.std():.4f}, max {mm.time_s.max():.3f}, runs {len(mm)})"
                if solver in ("ek", "mcf"):
                    s += f" aug {mm.augmentations.mean():,.0f}"
                if solver == "mcf":
                    s += f" cost {mm.cost.mean():,.0f}"
                if solver == "pr":
                    s += (f" push {mm.pushes.mean():,.0f} relabel {mm.relabels.mean():,.0f} "
                          f"global {mm.global_relabels.mean():.1f}")
                parts.append(s)
        print("  " + " | ".join(parts))
    for solver in ("ek", "pr", "mcf"):
        g = sc[sc.solver == solver].groupby("n").agg(t=("time_s", "mean"))
        if len(g) >= 3:
            print(f"  {solver}: sizes {list(g.index)}; fitted log-log slope {loglog_slope(g.index, g.t):.3f}; "
                  f"largest n {g.index.max()} mean {g.t.iloc[-1]:.3f} s")
    pivot = sc.groupby(["n", "solver"]).time_s.mean().unstack()
    for n, row in pivot.iterrows():
        if pd.notna(row.get("ek")) and pd.notna(row.get("pr")):
            print(f"  n={n}: EK / PR time {row.ek / row.pr:.2f}x")
        if pd.notna(row.get("mcf")) and pd.notna(row.get("pr")):
            print(f"  n={n}: MCF / PR time {row.mcf / row.pr:.2f}x")
    ek = sc[sc.solver == "ek"]
    if len(ek):
        per = (ek.time_s / ek.augmentations).groupby(ek.n).mean()
        print("  EK seconds per augmenting path: " + ", ".join(f"{n}:{v * 1e3:.2f} ms" for n, v in per.items()))
        print("  EK augmentations / n: " + ", ".join(
            f"{n}:{v:.3f}" for n, v in (ek.groupby('n').augmentations.mean() / ek.groupby('n').n.first()).items()))
    pr = sc[sc.solver == "pr"]
    print("  PR operations / n: " + ", ".join(
        f"{n}:{v:.2f}" for n, v in ((pr.pushes + pr.relabels).groupby(pr.n).mean()
                                    / pr.groupby('n').n.first()).items()))

    # ------------------------------------------------------------------
    header("2. Sparse against dense")
    sd = inst[inst.experiment == "sparse_vs_dense"]
    for (gt, n), g in sd.groupby(["graph_type", "n"]):
        parts = [f"{gt:<6} n={n:<5} m {g.m.mean():>9,.0f}"]
        for solver in ("ek", "pr", "mcf"):
            mm = g[g.solver == solver]
            if len(mm):
                s = f"{solver} {mm.time_s.mean():8.4f} s"
                if solver != "pr":
                    s += f" aug {mm.augmentations.mean():,.0f}"
                parts.append(s)
        print("  " + " | ".join(parts))

    # ------------------------------------------------------------------
    header("3. Capacity distributions")
    cd = inst[inst.experiment == "capacity_dist"]
    for dist, g in cd.groupby("cap_dist"):
        e = g[g.solver == "ek"]
        p = g[g.solver == "pr"]
        print(f"  {dist:<12} n={int(g.n.iloc[0])} flow {g.max_flow.mean():,.1f} | EK aug {e.augmentations.mean():,.1f} "
              f"time {e.time_s.mean():.3f} s | PR ops {(p.pushes + p.relabels).mean():,.0f} "
              f"(push {p.pushes.mean():,.0f} relabel {p.relabels.mean():,.0f}) time {p.time_s.mean():.3f} s")

    # ------------------------------------------------------------------
    header("4. Memory (peak during the solver, KiB)")
    mem = inst[inst.experiment == "memory"]
    for n, g in mem.groupby("n"):
        print(f"  n={n:<6} " + "  ".join(f"{r.solver} {r.peak_mem_kb:,.0f} KiB ({r.peak_mem_kb / n:.2f} per node)"
                                         for r in g.itertuples()))
        peaks = g.set_index("solver").peak_mem_kb
        print(f"           PR / EK memory {peaks['pr'] / peaks['ek']:.2f}x")

    # ------------------------------------------------------------------
    header("5. Applications")
    mt = inst[inst.experiment == "matching"]
    for n, g in mt.groupby("n"):
        print(f"  matching {n // 2} per side: edges {g.m.mean():,.0f} size {g.max_flow.mean():.1f} | "
              + " | ".join(f"{s} {g[g.solver == s].time_s.mean():.4f} s" for s in ("ek", "pr") if (g.solver == s).any())
              + f" | all valid {bool(g.valid.all())}")
    sg = inst[inst.experiment == "segmentation"]
    for n, g in sg.groupby("n"):
        print(f"  segmentation {int(n ** 0.5)}x{int(n ** 0.5)}: cut {g.max_flow.mean():,.1f} | "
              + " | ".join(f"{s} {g[g.solver == s].time_s.mean():.4f} s" for s in ("ek", "pr") if (g.solver == s).any())
              + f" | masks agree {bool(g.valid.all())}")

    # ------------------------------------------------------------------
    header("5b. Push-relabel convergence (rep 0 of each size)")
    conv = pd.read_csv(os.path.join(RESULTS, "pr_convergence.csv"))
    for n, g in conv.groupby("n"):
        first = g[g.operations > 0].iloc[0]
        total = g.operations.max()
        peak = g.active_nodes.max()
        start = g.total_excess.max()
        half = g[g.total_excess <= start / 2].operations.min()
        tenth = g[g.total_excess <= start / 10].operations.min()
        print(f"  n={n}: {len(g)} samples; {total:,} operations; start excess {start:,}; peak active {peak:,}; "
              f"excess halved by {half:,} ops ({half / total:.0%}), down to a tenth by {tenth:,} ops ({tenth / total:.0%}); "
              f"last sample active {g.active_nodes.iloc[-1]} excess {g.total_excess.iloc[-1]}")
        flow = sc[(sc.n == n) & (sc.solver == "pr") & (sc.rep == 0)].max_flow.iloc[0]
        back = start - flow
        print(f"    rep 0 flow {flow:,.0f}; excess that returned to s {back:,.0f} ({back / start:.1%} of the start)")

    # ------------------------------------------------------------------
    header("5c. Control: one hub edge per terminal")
    tc = inst[inst.experiment == "terminal_sensitivity"]
    hub = sc[sc.n == tc.n.iloc[0]]
    for label, frame in (("one hub edge", tc), ("5% hub edges", hub)):
        parts = [f"  {label:<13} n={int(frame.n.iloc[0])} flow {frame.max_flow.mean():,.1f}"]
        times = {}
        for solver in ("ek", "pr", "mcf"):
            mm = frame[frame.solver == solver]
            if len(mm):
                times[solver] = mm.time_s.mean()
                s = f"{solver} {mm.time_s.mean():.4f} s"
                if solver != "pr":
                    s += f" aug {mm.augmentations.mean():,.1f}"
                else:
                    s += f" ops {(mm.pushes + mm.relabels).mean():,.0f}"
                parts.append(s)
        if "ek" in times and "pr" in times:
            parts.append(f"EK / PR {times['ek'] / times['pr']:.2f}x")
        print(" | ".join(parts))

    # ------------------------------------------------------------------
    header("6. Size caps")
    if caps.empty:
        print("  none")
    for r in caps.itertuples():
        print(f"  {r.experiment} / {r.graph_type} / {r.solver}: {r.reason} at n={r.n}")

    # ------------------------------------------------------------------
    header("7. Methodology facts")
    print(f"  CSV rows {len(table)}; solver runs {len(inst)}; distinct seeds {inst.seed.nunique()}")
    print(f"  all valid {bool(inst.valid.all())}; EK/PR comparisons "
          f"{int(inst.ek_pr_agree.notna().sum()) // 2}; mismatches {int((inst.ek_pr_agree == False).sum())}")  # noqa: E712
    for name, g in table.groupby("experiment"):
        print(f"  {name:<16} rows {len(g):<3} n {int(g.n.min())}-{int(g.n.max())} runs {sorted(g.runs.unique().tolist())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
