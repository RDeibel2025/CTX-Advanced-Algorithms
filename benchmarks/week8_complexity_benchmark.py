#!/usr/bin/env python3
"""Week 8 benchmark: checking a solution is cheap; finding one is not.

Run from the repository root with the project virtualenv active::

    python benchmarks/week8_complexity_benchmark.py                 # the full study
    python benchmarks/week8_complexity_benchmark.py --smoke         # seconds, tiny sizes
    python benchmarks/week8_complexity_benchmark.py --seed 7 --out /tmp/w8

Five studies. Every one times the exact search, the approximation or
heuristic, and the polynomial-time verifier run on the certificate each
method returned:

1. **Subset sum** - backtracking (with nodes visited) against greedy and a
   seeded random-restart greedy, on the worst-case family (all values even,
   target odd, so no subset exists and backtracking cannot stop early) and on
   satisfiable instances (target = the sum of a random subset).
2. **SAT** - plain backtracking, DPLL and WalkSAT on random 3-SAT at clause
   ratio 4.26 (where random instances are hardest) and 2.0 (almost always
   satisfiable and easy).
3. **Vertex cover** - the exact branching solver, the 2-approximation and
   max-degree greedy on G(n, p) graphs with average degree about 4, plus the
   K3,3, star and path anchors.
4. **TSP** - Held-Karp (Week 6) against nearest neighbor and nearest neighbor
   followed by 2-opt on Euclidean instances, plus the points-on-a-line anchor.
5. **Reduction** - 3-SAT to vertex cover on satisfiable random formulas: the
   transformation's time and size, a WalkSAT assignment mapped forward to a
   cover of size k, verified, and mapped back.

**The 30-second rule.** An exact solver's size sweep stops before the next
size when one run at the current size passed 30 s, or when the growth
measured so far predicts that the next size would. A point whose runs add up
to more than 60 s keeps the runs it has. Every such stop is printed and
written to ``sweep_limits.csv``; nothing is extrapolated into the results.

**Reproducibility.** Every instance comes from
``derive_seed(master_seed, <experiment>, ..., n, rep)``; the seed of rep 0 is
the ``seed`` column, and rep r uses the same labels with r. Two runs with the
same ``--seed`` produce identical non-timing columns.

**Where the results go.** The repository holds every week's project, and the
root-level ``benchmarks/results/comparison_table.csv`` is Week 6's, so Week 8
writes ``runtime_growth.png``, ``approximation_quality.png`` and
``comparison_table.csv`` (plus ``instances.csv`` and ``sweep_limits.csv``) to
``benchmarks/results/week8/``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.complexity.reduction_demo import (  # noqa: E402
    ThreeSatToVertexCover,
    is_satisfying,
    min_vertex_cover_exact,
)
from src.complexity.satisfiability_solver import (  # noqa: E402
    random_3sat,
    sat_backtracking,
    sat_dpll,
    walksat,
)
from src.complexity.subset_sum_backtracking import (  # noqa: E402
    satisfiable_instance,
    subset_sum_backtracking,
    subset_sum_greedy,
    subset_sum_random_restart,
    worst_case_instance,
)
from src.complexity.traveling_salesman_approx import (  # noqa: E402
    euclidean_instance,
    is_valid_tour,
    tour_cost,
    tsp_nearest_neighbor,
    tsp_two_opt,
)
from src.complexity.vertex_cover_approx import (  # noqa: E402
    vertex_cover_2approx,
    vertex_cover_max_degree_greedy,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask  # noqa: E402
from src.utils.graph_utils import is_vertex_cover, random_graph  # noqa: E402
from src.utils.random_utils import DEFAULT_SEED, derive_seed  # noqa: E402
from src.utils.timer import time_call  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_approximation_quality,
    plot_runtime_growth,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week8")

#: The comparison table's columns are fixed by the specification.
COLUMNS = [
    "problem", "method", "kind", "instance_family", "n", "param", "runs", "mean_time_s",
    "std_time_s", "verify_time_s", "nodes_mean", "solution_value", "optimal_value",
    "approx_ratio", "success_rate", "seed", "measurement",
]

#: Columns that hold wall-clock time; every other column must be identical
#: between two runs with the same master seed.
TIMING_COLUMNS = ["mean_time_s", "std_time_s", "verify_time_s"]

#: One row per instance and method, the source of every range in the figures.
INSTANCE_COLUMNS = [
    "problem", "method", "kind", "instance_family", "n", "param", "rep", "seed", "time_s",
    "verify_time_s", "nodes", "solution_value", "optimal_value", "ratio", "success",
    "nn_over_two_opt",
]
INSTANCE_TIMING_COLUMNS = ["time_s", "verify_time_s"]

LIMIT_COLUMNS = ["experiment", "method", "n", "next_n", "max_run_s", "predicted_next_run_s",
                 "runs_kept", "reason"]

KIND = {
    "backtracking": "exact", "greedy": "heuristic", "random_restart": "heuristic",
    "sat_backtracking": "exact", "sat_dpll": "exact", "walksat": "heuristic",
    "exact_branching": "exact", "two_approx": "approximation",
    "max_degree_greedy": "heuristic", "held_karp": "exact", "nearest_neighbor": "heuristic",
    "nn_two_opt": "heuristic", "3sat_to_vertex_cover": "exact",
}


@dataclass
class Config:
    """Every size and repeat count of one benchmark mode."""

    run_cap_s: float
    ss_sizes: List[int]
    ss_reps: int
    ss_tries: int
    sat_sizes: List[int]
    sat_ratios: List[str]
    sat_reps: int
    walksat_flips: int
    vc_exact_max: int
    vc_sizes: List[int]
    vc_reps: int
    tsp_exact_sizes: List[int]
    tsp_large_sizes: List[int]
    tsp_reps: int
    tsp_large_reps: int
    red_sizes: List[int]
    red_reps: int
    verify_repeat: int


FULL = Config(
    run_cap_s=30.0,
    ss_sizes=list(range(10, 41, 2)),
    ss_reps=5,
    ss_tries=100,
    sat_sizes=[5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50],
    sat_ratios=["4.26", "2.0"],
    sat_reps=10,
    walksat_flips=10_000,
    vc_exact_max=80,
    vc_sizes=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100, 250, 500, 1000,
              2000],
    vc_reps=5,
    tsp_exact_sizes=list(range(5, 16)),
    tsp_large_sizes=[20, 50, 100, 200, 500],
    tsp_reps=5,
    tsp_large_reps=3,
    red_sizes=[10, 30, 100, 300, 1000],
    red_reps=3,
    verify_repeat=20,
)

SMOKE = Config(
    run_cap_s=5.0,
    ss_sizes=[6, 8, 10],
    ss_reps=2,
    ss_tries=10,
    sat_sizes=[5, 8],
    sat_ratios=["4.26", "2.0"],
    sat_reps=3,
    walksat_flips=500,
    vc_exact_max=12,
    vc_sizes=[8, 12, 30],
    vc_reps=2,
    tsp_exact_sizes=[5, 6, 7],
    tsp_large_sizes=[20],
    tsp_reps=2,
    tsp_large_reps=1,
    red_sizes=[10, 30],
    red_reps=1,
    verify_repeat=3,
)


# ----------------------------------------------------------------------
# Timing and records
# ----------------------------------------------------------------------
def time_once(function: Callable[[], Any]) -> Tuple[Any, float]:
    """Call ``function`` once with garbage collection paused; return (result, seconds)."""
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        started = time.perf_counter()
        result = function()
        return result, time.perf_counter() - started
    finally:
        if was_enabled:
            gc.enable()


def timed(function: Callable[[], Any], short_s: float = 0.02) -> Tuple[Any, float]:
    """Time a call; a call under ``short_s`` is re-timed as the mean of 5 runs.

    Every method timed here is deterministic for a given instance and seed,
    so the repeats return the same answer and only sharpen the time.
    """
    result, elapsed = time_once(function)
    if elapsed < short_s:
        elapsed = time_call(function, repeat=5, warmup=1)["mean"]
    return result, elapsed


def verify_time(check: Callable[[], bool], repeat: int) -> float:
    """Assert a verifier accepts the certificate, then return its mean time."""
    assert check(), "verifier rejected a certificate"
    return time_call(check, repeat=repeat, warmup=1)["mean"]


def fmt(value: Optional[float], digits: int = 6) -> str:
    """Format a number for the CSV; ``None`` becomes an empty cell."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def check_subset(nums: Sequence[int], subset: Sequence[int], total: int) -> bool:
    """The subset-sum verifier: a sub-multiset of ``nums`` whose sum is ``total``."""
    return not (Counter(subset) - Counter(nums)) and sum(subset) == total


def check_tour(dist: Sequence[Sequence[float]], tour: Sequence[int], cost: float) -> bool:
    """The TSP verifier: a valid closed tour whose edges sum to ``cost``."""
    return is_valid_tour(tour, len(dist)) and math.isclose(tour_cost(dist, tour), cost,
                                                           rel_tol=1e-9, abs_tol=1e-9)


def record(records: List[dict], problem: str, method: str, family: str, n: int, param: Any,
           rep: int, seed: int, time_s: float, **fields: Any) -> dict:
    """Append one per-instance record and return it."""
    entry = {"problem": problem, "method": method, "kind": KIND[method],
             "instance_family": family, "n": n, "param": param, "rep": rep, "seed": seed,
             "time_s": time_s, "verify_time_s": None, "nodes": None, "solution_value": None,
             "optimal_value": None, "ratio": None, "success": None, "nn_over_two_opt": None}
    entry.update(fields)
    records.append(entry)
    return entry


def mean_of(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of the non-missing values, or ``None`` when there are none."""
    present = [float(v) for v in values if v is not None]
    return statistics.fmean(present) if present else None


def aggregate(records: Sequence[dict]) -> List[dict]:
    """Collapse per-instance records into comparison-table rows.

    One row per (problem, method, family, n, param). Times, nodes, solution
    values, optimal values and ratios are means over the instances; the
    standard deviation of time is the sample standard deviation across
    instances (blank for a single instance).
    """
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        key = (r["problem"], r["method"], r["instance_family"], r["n"], r["param"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (problem, method, family, n, param), group in groups.items():
        times = [g["time_s"] for g in group]
        successes = [g["success"] for g in group if g["success"] is not None]
        if method == "walksat":
            sat = [g for g in group if g["optimal_value"] == 1]
            successes = [g["success"] for g in sat]
        rows.append({
            "problem": problem, "method": method, "kind": KIND[method],
            "instance_family": family, "n": n, "param": param, "runs": len(group),
            "mean_time_s": fmt(statistics.fmean(times), 6),
            "std_time_s": fmt(statistics.stdev(times), 6) if len(times) > 1 else "",
            "verify_time_s": fmt(mean_of([g["verify_time_s"] for g in group]), 6),
            "nodes_mean": fmt(mean_of([g["nodes"] for g in group]), 8),
            "solution_value": fmt(mean_of([g["solution_value"] for g in group]), 8),
            "optimal_value": fmt(mean_of([g["optimal_value"] for g in group]), 8),
            "approx_ratio": fmt(mean_of([g["ratio"] for g in group]), 6),
            "success_rate": fmt(statistics.fmean(successes), 6) if successes else "",
            "seed": group[0]["seed"], "measurement": "measured",
        })
    return rows


class Sweep:
    """The 30-second rule for one exact method's size sweep."""

    def __init__(self, experiment: str, method: str, cap_s: float, limits: List[dict]):
        self.experiment, self.method, self.cap_s, self.limits = experiment, method, cap_s, limits
        self.active = True
        self.history: List[Tuple[int, float]] = []

    def point_budget_spent(self, spent_s: float) -> bool:
        """True once one point's runs add up to more than twice the cap."""
        return spent_s > 2 * self.cap_s

    def note_reduced(self, n: int, runs_kept: int, spent_s: float) -> None:
        """Record that a point kept fewer runs than planned."""
        self.limits.append({"experiment": self.experiment, "method": self.method, "n": n,
                            "next_n": "", "max_run_s": "", "predicted_next_run_s": "",
                            "runs_kept": runs_kept,
                            "reason": f"runs at this size took {spent_s:.1f} s in total"})
        print(f"    [limit] {self.experiment}/{self.method} n={n}: kept {runs_kept} runs "
              f"({spent_s:.1f} s)", flush=True)

    def finish_point(self, n: int, max_run_s: float, next_n: Optional[int]) -> None:
        """Decide whether the sweep continues to ``next_n``."""
        self.history.append((n, max_run_s))
        if next_n is None or not self.active:
            return
        predicted = None
        reason = ""
        if max_run_s > self.cap_s:
            reason = f"one run took {max_run_s:.1f} s, over the {self.cap_s:.0f} s cap"
        elif len(self.history) >= 2 and self.history[-2][1] > 0.05:
            (n0, t0), (n1, t1) = self.history[-2], self.history[-1]
            per_item = (t1 / t0) ** (1.0 / (n1 - n0)) if t0 > 0 and n1 > n0 else 1.0
            predicted = t1 * per_item ** (next_n - n1)
            if predicted > self.cap_s:
                reason = (f"growth from n={n0} to n={n1} predicts {predicted:.1f} s for one run "
                          f"at n={next_n}")
        if reason:
            self.active = False
            self.limits.append({"experiment": self.experiment, "method": self.method, "n": n,
                                "next_n": next_n, "max_run_s": fmt(max_run_s, 4),
                                "predicted_next_run_s": fmt(predicted, 4), "runs_kept": "",
                                "reason": reason})
            print(f"    [limit] {self.experiment}/{self.method}: stop after n={n}: {reason}",
                  flush=True)


def next_of(sizes: Sequence[int], n: int) -> Optional[int]:
    """The size after ``n`` in ``sizes``, or ``None`` at the end."""
    i = list(sizes).index(n)
    return sizes[i + 1] if i + 1 < len(sizes) else None


# ----------------------------------------------------------------------
# Study 1: subset sum
# ----------------------------------------------------------------------
def subset_sum_study(cfg: Config, master: int, records: List[dict], limits: List[dict]) -> None:
    print("\n=== STUDY 1 - Subset sum ===", flush=True)
    for family, make in (("worst_case", worst_case_instance), ("satisfiable", satisfiable_instance)):
        sweep = Sweep(f"subset_sum_{family}", "backtracking", cfg.run_cap_s, limits)
        for n in cfg.ss_sizes:
            if not sweep.active:
                break
            spent, max_run, kept = 0.0, 0.0, 0
            for rep in range(cfg.ss_reps):
                seed = derive_seed(master, "subset_sum", family, n, rep)
                nums, target = make(n, seed)
                (found, nodes), t = timed(lambda: subset_sum_backtracking(nums, target,
                                                                         count_nodes=True))
                spent += t
                max_run = max(max_run, t)
                kept += 1
                if family == "worst_case":
                    assert found is None, "the worst-case family has no solution"
                    record(records, "subset_sum", "backtracking", family, n, "", rep, seed, t,
                           nodes=nodes, success=1)
                else:
                    assert found is not None and check_subset(nums, found, target)
                    record(records, "subset_sum", "backtracking", family, n, "", rep, seed, t,
                           nodes=nodes, solution_value=0, optimal_value=0, success=1,
                           verify_time_s=verify_time(lambda: check_subset(nums, found, target),
                                                     cfg.verify_repeat))
                for method in ("greedy", "random_restart"):
                    if method == "greedy":
                        call = lambda: subset_sum_greedy(nums, target)  # noqa: E731
                    else:
                        rr_seed = derive_seed(master, "subset_sum_restart", family, n, rep)
                        call = lambda s=rr_seed: subset_sum_random_restart(  # noqa: E731
                            nums, target, tries=cfg.ss_tries, seed=s)
                    (subset, achieved), th = timed(call)
                    assert achieved <= target and check_subset(nums, subset, achieved)
                    gap = target - achieved
                    record(records, "subset_sum", method, family, n, "", rep, seed, th,
                           solution_value=gap,
                           optimal_value=0 if family == "satisfiable" else None,
                           success=(1 if gap == 0 else 0) if family == "satisfiable" else None,
                           verify_time_s=verify_time(lambda: check_subset(nums, subset, achieved),
                                                     cfg.verify_repeat))
                if sweep.point_budget_spent(spent) and rep + 1 < cfg.ss_reps:
                    sweep.note_reduced(n, kept, spent)
                    break
            pts = [r for r in records if r["problem"] == "subset_sum" and r["n"] == n
                   and r["instance_family"] == family and r["method"] == "backtracking"]
            print(f"    {family:<11} n={n:<3} backtracking {statistics.fmean([p['time_s'] for p in pts]):10.5f} s"
                  f"  nodes {statistics.fmean([p['nodes'] for p in pts]):14,.0f}  max run {max_run:.3f} s",
                  flush=True)
            sweep.finish_point(n, max_run, next_of(cfg.ss_sizes, n))


# ----------------------------------------------------------------------
# Study 2: SAT
# ----------------------------------------------------------------------
def sat_study(cfg: Config, master: int, records: List[dict], limits: List[dict]) -> None:
    print("\n=== STUDY 2 - SAT ===", flush=True)
    for ratio in cfg.sat_ratios:
        sweeps = {m: Sweep(f"sat_ratio_{ratio}", m, cfg.run_cap_s, limits)
                  for m in ("sat_backtracking", "sat_dpll")}
        for n in cfg.sat_sizes:
            if not sweeps["sat_dpll"].active:
                break
            m_clauses = round(float(ratio) * n)
            spent = {m: 0.0 for m in sweeps}
            max_run = {m: 0.0 for m in sweeps}
            reduced = {m: False for m in sweeps}
            for rep in range(cfg.sat_reps):
                seed = derive_seed(master, "sat", ratio, n, rep)
                cnf = random_3sat(n, m_clauses, seed)
                truth = None
                for method, solver in (("sat_dpll", sat_dpll), ("sat_backtracking", sat_backtracking)):
                    sweep = sweeps[method]
                    if not sweep.active or reduced[method]:
                        continue
                    (found, nodes), t = timed(lambda: solver(cnf, n, count_nodes=True))
                    spent[method] += t
                    max_run[method] = max(max_run[method], t)
                    if truth is None:
                        truth = found is not None
                    assert (found is not None) == truth, "complete solvers disagree"
                    vt = None
                    if found is not None:
                        vt = verify_time(lambda: is_satisfying(cnf, found), cfg.verify_repeat)
                    record(records, "sat", method, "random_3sat", n, ratio, rep, seed, t,
                           nodes=nodes, solution_value=int(found is not None),
                           optimal_value=int(truth), success=1, verify_time_s=vt)
                    if sweep.point_budget_spent(spent[method]) and rep + 1 < cfg.sat_reps:
                        reduced[method] = True
                        sweep.note_reduced(n, rep + 1, spent[method])
                w_seed = derive_seed(master, "walksat", ratio, n, rep)
                found, t = timed(lambda: walksat(cnf, n, max_flips=cfg.walksat_flips, seed=w_seed))
                if found is not None:
                    assert truth, "WalkSAT returned an assignment for an unsatisfiable formula"
                vt = (verify_time(lambda: is_satisfying(cnf, found), cfg.verify_repeat)
                      if found is not None else None)
                record(records, "sat", "walksat", "random_3sat", n, ratio, rep, w_seed, t,
                       solution_value=int(found is not None), optimal_value=int(truth),
                       success=int(found is not None), verify_time_s=vt)
            pts = [r for r in records if r["problem"] == "sat" and r["n"] == n and r["param"] == ratio]
            summary = []
            for method in ("sat_backtracking", "sat_dpll", "walksat"):
                mine = [p for p in pts if p["method"] == method]
                if mine:
                    summary.append(f"{method} {statistics.fmean([p['time_s'] for p in mine]):.5f} s")
            sat_frac = statistics.fmean([p["optimal_value"] for p in pts if p["method"] == "sat_dpll"])
            print(f"    ratio {ratio:<4} n={n:<3} m={m_clauses:<4} satisfiable {sat_frac:.1f}  "
                  + "  ".join(summary), flush=True)
            for method, sweep in sweeps.items():
                if sweep.active:
                    sweep.finish_point(n, max_run[method], next_of(cfg.sat_sizes, n))


# ----------------------------------------------------------------------
# Study 3: vertex cover
# ----------------------------------------------------------------------
def anchor_graphs() -> List[Tuple[str, List[int], set]]:
    """The three graphs on which the 2-approximation's ratio is exactly 2."""
    path = ("path", [1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)})
    star = ("star", list(range(6)), {(0, leaf) for leaf in range(1, 6)})
    k33 = ("k33", list(range(6)), {(u, v) for u in range(3) for v in range(3, 6)})
    return [path, star, k33]


def vertex_cover_study(cfg: Config, master: int, records: List[dict], limits: List[dict]) -> None:
    print("\n=== STUDY 3 - Vertex cover ===", flush=True)
    for family, vertices, edges in anchor_graphs():
        exact, t_exact = timed(lambda: min_vertex_cover_exact(vertices, edges))
        opt = len(exact)
        record(records, "vertex_cover", "exact_branching", family, len(vertices), "", 0, "", t_exact,
               solution_value=opt, optimal_value=opt, ratio=1.0, success=1,
               verify_time_s=verify_time(lambda: is_vertex_cover(edges, exact), cfg.verify_repeat))
        for method, solver in (("two_approx", vertex_cover_2approx),
                               ("max_degree_greedy", vertex_cover_max_degree_greedy)):
            cover, t = timed(lambda: solver(vertices, edges))
            record(records, "vertex_cover", method, family, len(vertices), "", 0, "", t,
                   solution_value=len(cover), optimal_value=opt, ratio=len(cover) / opt,
                   success=int(len(cover) == opt),
                   verify_time_s=verify_time(lambda: is_vertex_cover(edges, cover), cfg.verify_repeat))
        print(f"    anchor {family:<5} exact {opt}  2-approx "
              f"{records[-2]['solution_value']}  greedy {records[-1]['solution_value']}", flush=True)

    family = "gnp_avg_degree_4"
    sweep = Sweep("vertex_cover", "exact_branching", cfg.run_cap_s, limits)
    exact_sizes = [n for n in cfg.vc_sizes if n <= cfg.vc_exact_max]
    for n in cfg.vc_sizes:
        p = min(1.0, 4.0 / (n - 1))
        param = f"{p:.6g}"
        spent, max_run, reduced = 0.0, 0.0, False
        run_exact = n in exact_sizes and sweep.active
        for rep in range(cfg.vc_reps):
            seed = derive_seed(master, "vertex_cover", n, rep)
            vertices, edges = random_graph(n, p, seed)
            opt = None
            if run_exact and not reduced:
                exact, t = timed(lambda: min_vertex_cover_exact(vertices, edges))
                spent += t
                max_run = max(max_run, t)
                opt = len(exact)
                record(records, "vertex_cover", "exact_branching", family, n, param, rep, seed, t,
                       solution_value=opt, optimal_value=opt, ratio=1.0, success=1,
                       verify_time_s=verify_time(lambda: is_vertex_cover(edges, exact),
                                                 cfg.verify_repeat))
                if sweep.point_budget_spent(spent) and rep + 1 < cfg.vc_reps:
                    reduced = True
                    sweep.note_reduced(n, rep + 1, spent)
            for method, solver in (("two_approx", vertex_cover_2approx),
                                   ("max_degree_greedy", vertex_cover_max_degree_greedy)):
                cover, t = timed(lambda: solver(vertices, edges))
                record(records, "vertex_cover", method, family, n, param, rep, seed, t,
                       solution_value=len(cover), optimal_value=opt,
                       ratio=(len(cover) / opt) if opt else None,
                       success=int(len(cover) == opt) if opt is not None else None,
                       verify_time_s=verify_time(lambda: is_vertex_cover(edges, cover),
                                                 cfg.verify_repeat))
        pts = [r for r in records if r["problem"] == "vertex_cover" and r["n"] == n
               and r["instance_family"] == family]
        line = []
        for method in ("exact_branching", "two_approx", "max_degree_greedy"):
            mine = [q for q in pts if q["method"] == method]
            if mine:
                ratio = mean_of([q["ratio"] for q in mine])
                line.append(f"{method} {statistics.fmean([q['time_s'] for q in mine]):.5f} s "
                            f"size {statistics.fmean([q['solution_value'] for q in mine]):.1f}"
                            + (f" ratio {ratio:.3f}" if ratio is not None else ""))
        print(f"    n={n:<5} " + "  ".join(line), flush=True)
        if run_exact:
            sweep.finish_point(n, max_run, next_of(exact_sizes, n))


# ----------------------------------------------------------------------
# Study 4: TSP
# ----------------------------------------------------------------------
def tsp_study(cfg: Config, master: int, records: List[dict], limits: List[dict]) -> None:
    print("\n=== STUDY 4 - TSP ===", flush=True)
    xs = [0, 1, -2, 4, -8, 16]
    line = [[abs(a - b) for b in xs] for a in xs]
    opt_cost, opt_tour = tsp_bitmask(line)
    (nn_cost, nn_tour), t = timed(lambda: tsp_nearest_neighbor(line))
    record(records, "tsp", "nearest_neighbor", "line_anchor", 6, "", 0, "", t,
           solution_value=nn_cost, optimal_value=opt_cost, ratio=nn_cost / opt_cost,
           success=int(math.isclose(nn_cost, opt_cost)),
           verify_time_s=verify_time(lambda: check_tour(line, nn_tour, nn_cost), cfg.verify_repeat))
    print(f"    line anchor: nearest neighbor {nn_cost:g} against optimum {opt_cost:g}", flush=True)

    sweep = Sweep("tsp_large", "nn_two_opt", cfg.run_cap_s, limits)
    for n in cfg.tsp_exact_sizes + cfg.tsp_large_sizes:
        exact_point = n in cfg.tsp_exact_sizes
        if not exact_point and not sweep.active:
            break
        reps = cfg.tsp_reps if exact_point else cfg.tsp_large_reps
        max_run = 0.0
        for rep in range(reps):
            seed = derive_seed(master, "tsp", n, rep)
            _, dist = euclidean_instance(n, seed)
            opt = None
            if exact_point:
                (opt, opt_tour), t = timed(lambda: tsp_bitmask(dist))
                record(records, "tsp", "held_karp", "euclidean", n, "", rep, seed, t,
                       solution_value=opt, optimal_value=opt, ratio=1.0, success=1,
                       verify_time_s=verify_time(lambda: check_tour(dist, opt_tour, opt),
                                                 cfg.verify_repeat))
            (nn_cost, nn_tour), t_nn = timed(lambda: tsp_nearest_neighbor(dist))
            (two_cost, two_tour), t_two = timed(lambda: tsp_two_opt(dist, nn_tour))
            assert two_cost <= nn_cost + 1e-9
            max_run = max(max_run, t_nn + t_two)
            vt_nn = verify_time(lambda: check_tour(dist, nn_tour, nn_cost), cfg.verify_repeat)
            vt_two = verify_time(lambda: check_tour(dist, two_tour, two_cost), cfg.verify_repeat)
            record(records, "tsp", "nearest_neighbor", "euclidean", n, "", rep, seed, t_nn,
                   solution_value=nn_cost, optimal_value=opt,
                   ratio=(nn_cost / opt) if opt else None,
                   success=int(math.isclose(nn_cost, opt)) if opt is not None else None,
                   nn_over_two_opt=None if exact_point else nn_cost / two_cost,
                   verify_time_s=vt_nn)
            record(records, "tsp", "nn_two_opt", "euclidean", n, "", rep, seed, t_nn + t_two,
                   solution_value=two_cost, optimal_value=opt,
                   ratio=(two_cost / opt) if opt else None,
                   success=int(math.isclose(two_cost, opt)) if opt is not None else None,
                   verify_time_s=vt_two)
        pts = [r for r in records if r["problem"] == "tsp" and r["n"] == n
               and r["instance_family"] == "euclidean"]
        line_out = []
        for method in ("held_karp", "nearest_neighbor", "nn_two_opt"):
            mine = [q for q in pts if q["method"] == method]
            if mine:
                ratio = mean_of([q["ratio"] for q in mine])
                line_out.append(f"{method} {statistics.fmean([q['time_s'] for q in mine]):.5f} s"
                                + (f" ratio {ratio:.3f}" if ratio is not None else ""))
        print(f"    n={n:<4} " + "  ".join(line_out), flush=True)
        if not exact_point:
            sweep.finish_point(n, max_run, next_of(cfg.tsp_large_sizes, n))


# ----------------------------------------------------------------------
# Study 5: the reduction itself
# ----------------------------------------------------------------------
def reduction_study(cfg: Config, master: int, records: List[dict]) -> None:
    print("\n=== STUDY 5 - 3-SAT to vertex cover ===", flush=True)
    reduction = ThreeSatToVertexCover()
    for n in cfg.red_sizes:
        m_clauses = round(2.0 * n)
        for rep in range(cfg.red_reps):
            seed = derive_seed(master, "reduction", n, rep)
            cnf = random_3sat(n, m_clauses, seed)
            result, t = timed(lambda: reduction.transform((cnf, n)))
            (vertices, edges), k = result.target_instance
            assert len(vertices) == 2 * n + 3 * m_clauses and len(edges) == n + 6 * m_clauses
            assert k == n + 2 * m_clauses
            w_seed = derive_seed(master, "reduction_walksat", n, rep)
            assignment = walksat(cnf, n, max_flips=200_000, seed=w_seed)
            ok, vt = 0, None
            if assignment is not None:
                cover = reduction.map_solution_forward((cnf, n), assignment, result)
                vt = verify_time(lambda: is_vertex_cover(edges, cover) and len(cover) <= k,
                                 cfg.verify_repeat)
                back = reduction.map_solution_back((cnf, n), cover, result)
                ok = int(len(cover) == k and is_satisfying(cnf, back))
            record(records, "reduction", "3sat_to_vertex_cover", "random_3sat", n, "2.0", rep, seed,
                   t, nodes=len(vertices), solution_value=len(edges), optimal_value=k,
                   success=ok, verify_time_s=vt)
        pts = [r for r in records if r["problem"] == "reduction" and r["n"] == n]
        print(f"    n={n:<5} m={m_clauses:<5} nodes {pts[0]['nodes']:<6} edges {pts[0]['solution_value']:<6}"
              f" transform {statistics.fmean([p['time_s'] for p in pts]):.5f} s"
              f"  certificate round trip ok {sum(p['success'] for p in pts)}/{len(pts)}", flush=True)


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """Write ``rows`` with exactly ``columns``; ``None`` becomes an empty cell."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r.get(c) is None else
                                 fmt(r[c], 8) if isinstance(r.get(c), float) else r[c])
                             for c in columns})


def tradeoff_sizes(records: Sequence[dict]) -> Dict[str, int]:
    """The largest size at which every method of a trade-off panel ran."""
    def largest(problem: str, method: str, **match: Any) -> Optional[int]:
        ns = [r["n"] for r in records if r["problem"] == problem and r["method"] == method
              and all(r.get(k) == v for k, v in match.items())]
        return max(ns) if ns else None
    return {
        "vertex_cover": largest("vertex_cover", "exact_branching", instance_family="gnp_avg_degree_4"),
        "tsp": largest("tsp", "held_karp", instance_family="euclidean"),
        "subset_sum": largest("subset_sum", "backtracking", instance_family="satisfiable"),
        "sat": largest("sat", "sat_backtracking", param="4.26"),
    }


def read_records(path: str) -> List[dict]:
    """Read ``instances.csv`` back into per-instance records for ``--replot``."""
    text_fields = {"problem", "method", "kind", "instance_family", "param", "seed"}
    records = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            entry: Dict[str, Any] = {}
            for key, value in raw.items():
                if key in text_fields:
                    entry[key] = value
                elif value == "":
                    entry[key] = None
                elif key in ("n", "rep"):
                    entry[key] = int(value)
                else:
                    entry[key] = float(value)
            records.append(entry)
    return records


def draw_figures(records: List[dict], out: str) -> None:
    """Write both figures from the per-instance records."""
    for name, builder in (
        ("runtime_growth.png", lambda p: plot_runtime_growth(records, save_path=p)),
        ("approximation_quality.png",
         lambda p: plot_approximation_quality(records, tradeoff_sizes(records), save_path=p)),
    ):
        path = os.path.join(out, name)
        builder(path)
        print(f"  wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny parameter set, seconds")
    parser.add_argument("--out", default=RESULTS_DIR, help="directory for the CSVs and figures")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed (default 42)")
    parser.add_argument("--replot", action="store_true",
                        help="redraw the figures from an existing instances.csv in --out")
    args = parser.parse_args(argv)

    cfg = SMOKE if args.smoke else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    master = args.seed
    apply_house_style()
    if args.replot:
        draw_figures(read_records(os.path.join(out, "instances.csv")), out)
        return 0

    print("=" * 78)
    print("CSC 5300 Week 8 - computational complexity benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    : {platform.platform()}")
    print(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  master seed : {master}  (instances: derive_seed(master, experiment, ..., n, rep))")
    print(f"  run cap     : {cfg.run_cap_s:.0f} s per run")
    print(f"  output      : {out}", flush=True)

    started = time.perf_counter()
    records: List[dict] = []
    limits: List[dict] = []
    subset_sum_study(cfg, master, records, limits)
    sat_study(cfg, master, records, limits)
    vertex_cover_study(cfg, master, records, limits)
    tsp_study(cfg, master, records, limits)
    reduction_study(cfg, master, records)

    print("\n=== writing results ===", flush=True)
    write_csv(os.path.join(out, "comparison_table.csv"), aggregate(records), COLUMNS)
    write_csv(os.path.join(out, "instances.csv"), records, INSTANCE_COLUMNS)
    write_csv(os.path.join(out, "sweep_limits.csv"), limits, LIMIT_COLUMNS)

    print("\n=== figures ===", flush=True)
    draw_figures(records, out)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
