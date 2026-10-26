#!/usr/bin/env python3
"""Week 9 benchmark: every guarantee is a comparison with a computable lower bound.

Run from the repository root with the project virtualenv active::

    python benchmarks/week9_approx_benchmark.py                 # the full study (under 3 minutes)
    python benchmarks/week9_approx_benchmark.py --smoke         # seconds, tiny sizes
    python benchmarks/week9_approx_benchmark.py --seed 7 --out /tmp/w9

Five studies. Each approximation is compared with the exact optimum where
one can be computed, and at every size with the polynomial-time bound its
proof uses: the LP relaxation (vertex cover, set cover; for vertex cover the
LP value is never below the maximal matching the 2-approximation picks), the
minimum spanning tree (metric TSP) and the edge count |E| (Max-Cut, an upper
bound for a maximization problem). Cost divided by a lower bound is a
certified upper bound on the true ratio, so it is meaningful even where OPT
is out of reach.

1. **Vertex cover** - 2-approximation and LP rounding on G(n, p) with average
   degree 4, exact optimum (Week 8's branching solver) to n = 30.
2. **Set cover** - Chvatal greedy, deterministic rounding, randomized rounding
   (200 seeded runs per instance) and the exact MILP.
3. **Facility location** - greedy against the MILP optimum.
4. **Metric TSP** - MST doubling (our code), Christofides (networkx, a library
   comparison) and Week 8's nearest neighbor, against Held-Karp for n <= 13.
5. **Max-Cut** - random cut (200 seeded runs per graph) and local search (20
   seeded starts) against brute force for n <= 16.

LP and MILP problems are solved by SciPy's HiGHS (``linprog`` and ``milp``)
through ``src/utils/lp_solver.py``.

**Reproducibility.** Every instance and every random run comes from
``derive_seed(master_seed, <experiment>, n, rep[, run])``; the seed of rep 0
is the ``seed`` column. Two runs with the same ``--seed`` produce identical
non-timing columns.

**Where the results go.** The repository holds every week's project and the
root-level ``benchmarks/results/comparison_table.csv`` is Week 6's, so Week 9
writes ``approx_ratios.png``, ``cost_vs_optimal.png``,
``randomized_maxcut_results.png`` and ``comparison_table.csv`` (plus
``instances.csv``) to ``benchmarks/results/week9/``.
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.approximation.facility_location_greedy import (  # noqa: E402
    facility_location_exact,
    facility_location_greedy,
)
from src.approximation.facility_location_greedy import random_instance as facility_instance  # noqa: E402
from src.approximation.maxcut_randomized import (  # noqa: E402
    maxcut_exact,
    maxcut_local_search,
    maxcut_random,
)
from src.approximation.set_cover_lpround import (  # noqa: E402
    frequency,
    set_cover_deterministic_rounding,
    set_cover_exact,
    set_cover_greedy,
    set_cover_lp,
    set_cover_randomized_rounding,
)
from src.approximation.set_cover_lpround import random_instance as set_cover_instance  # noqa: E402
from src.approximation.tsp_metric_approx import (  # noqa: E402
    is_metric,
    mst_weight_and_edges,
    tsp_christofides_networkx,
    tsp_mst_doubling,
)
from src.approximation.vertex_cover_2approx import (  # noqa: E402
    matching_lower_bound,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_lp_rounding,
)
from src.complexity.traveling_salesman_approx import (  # noqa: E402
    euclidean_instance,
    is_valid_tour,
    tsp_nearest_neighbor,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask  # noqa: E402
from src.utils.graph_utils import is_vertex_cover, random_graph  # noqa: E402
from src.utils.random_utils import DEFAULT_SEED, derive_seed  # noqa: E402
from src.utils.timer import time_call  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_approx_ratios,
    plot_cost_vs_optimal,
    plot_randomized_results,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week9")

#: The comparison table's columns are fixed by the specification.
COLUMNS = [
    "problem", "method", "kind", "n", "param", "runs", "mean_cost", "std_cost", "min_cost",
    "max_cost", "optimal_value", "lower_bound", "lower_bound_kind", "ratio_to_opt",
    "ratio_to_lower_bound", "proven_bound", "mean_time_s", "seed",
]
TIMING_COLUMNS = ["mean_time_s"]

INSTANCE_COLUMNS = [
    "problem", "method", "kind", "n", "param", "rep", "seed", "runs", "cost", "cost_std",
    "cost_min", "cost_max", "optimal", "lower_bound", "lower_bound_kind", "ratio_to_opt",
    "ratio_to_lb", "proven_bound", "time_s", "matching_bound", "fixup_rate",
]
INSTANCE_TIMING_COLUMNS = ["time_s"]

KIND = {
    "exact": "exact", "two_approx": "approximation", "lp_rounding": "approximation",
    "greedy": "approximation", "deterministic_rounding": "approximation",
    "randomized_rounding": "approximation", "facility_greedy": "heuristic",
    "mst_doubling": "approximation", "christofides_networkx": "approximation",
    "nearest_neighbor": "heuristic", "random_cut": "approximation",
    "local_search": "approximation",
}


@dataclass
class Config:
    """Every size and repeat count of one benchmark mode."""

    vc_exact_sizes: List[int]
    vc_large_sizes: List[int]
    vc_reps: int
    vc_exact_cap_s: float
    sc_sizes: List[int]
    sc_reps: int
    sc_runs: int
    fl_sizes: List[Tuple[int, int]]
    fl_reps: int
    tsp_exact_sizes: List[int]
    tsp_large_sizes: List[int]
    tsp_reps: int
    tsp_large_reps: int
    mc_exact_sizes: List[int]
    mc_large_sizes: List[int]
    mc_reps: int
    mc_runs: int
    mc_starts: int
    mc_large_starts: int


FULL = Config(
    vc_exact_sizes=[10, 15, 20, 25, 30], vc_large_sizes=[100, 500, 2000], vc_reps=5,
    vc_exact_cap_s=10.0,
    sc_sizes=[20, 50, 100, 200], sc_reps=5, sc_runs=200,
    fl_sizes=[(20, 10), (50, 15), (100, 20)], fl_reps=5,
    tsp_exact_sizes=[6, 8, 10, 12, 13], tsp_large_sizes=[50, 100, 300], tsp_reps=5,
    tsp_large_reps=3,
    mc_exact_sizes=[8, 10, 12, 14, 16], mc_large_sizes=[100, 500], mc_reps=3, mc_runs=200,
    mc_starts=20, mc_large_starts=20,
)

SMOKE = Config(
    vc_exact_sizes=[10, 15], vc_large_sizes=[100], vc_reps=2, vc_exact_cap_s=10.0,
    sc_sizes=[20, 30], sc_reps=2, sc_runs=20,
    fl_sizes=[(10, 5)], fl_reps=2,
    tsp_exact_sizes=[6, 8], tsp_large_sizes=[20], tsp_reps=2, tsp_large_reps=1,
    mc_exact_sizes=[8, 10], mc_large_sizes=[30], mc_reps=2, mc_runs=20, mc_starts=3,
    mc_large_starts=3,
)


# ----------------------------------------------------------------------
# Timing and records
# ----------------------------------------------------------------------
def timed(function: Callable[[], Any], short_s: float = 0.02) -> Tuple[Any, float]:
    """Time one call with garbage collection paused; re-time a short call as a mean of 5."""
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        started = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - started
    finally:
        if was_enabled:
            gc.enable()
    if elapsed < short_s:
        elapsed = time_call(function, repeat=5, warmup=1)["mean"]
    return result, elapsed


def fmt(value: Optional[float], digits: int = 8) -> str:
    """Format a number for the CSV; ``None`` becomes an empty cell."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def record(records: List[dict], problem: str, method: str, n: int, param: str, rep: int,
           seed: Any, costs: Sequence[float], time_s: float, optimal: Optional[float] = None,
           lower_bound: Optional[float] = None, lower_bound_kind: str = "",
           proven_bound: Optional[float] = None, **extra: Any) -> dict:
    """Append one per-instance record; ``costs`` holds one value per run."""
    mean = statistics.fmean(costs)
    entry = {
        "problem": problem, "method": method, "kind": KIND[method], "n": n, "param": param,
        "rep": rep, "seed": seed, "runs": len(costs), "cost": mean,
        "cost_std": statistics.stdev(costs) if len(costs) > 1 else None,
        "cost_min": min(costs), "cost_max": max(costs), "optimal": optimal,
        "lower_bound": lower_bound, "lower_bound_kind": lower_bound_kind,
        "ratio_to_opt": (mean / optimal) if optimal else (1.0 if optimal == 0 and mean == 0 else None),
        "ratio_to_lb": (mean / lower_bound) if lower_bound else None,
        "proven_bound": proven_bound, "time_s": time_s, "matching_bound": None, "fixup_rate": None,
        "samples": list(costs),
    }
    entry.update(extra)
    records.append(entry)
    return entry


def aggregate(records: Sequence[dict]) -> List[dict]:
    """Collapse per-instance records into comparison-table rows, one per (problem, method, n, param).

    Costs, optima, bounds, ratios and times are means over the instances. For
    a method run once per instance, ``std_cost`` is the standard deviation
    across instances; for a randomized method run many times per instance it
    is the mean within-instance standard deviation, the spread the
    randomness itself causes. ``min_cost`` and ``max_cost`` span every run.
    """
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        groups.setdefault((r["problem"], r["method"], r["n"], r["param"]), []).append(r)
    rows = []
    for (problem, method, n, param), group in groups.items():
        repeated = group[0]["runs"] > 1
        if repeated:
            std = statistics.fmean([g["cost_std"] for g in group if g["cost_std"] is not None])
        else:
            std = statistics.stdev([g["cost"] for g in group]) if len(group) > 1 else None

        def mean_of(key: str) -> Optional[float]:
            values = [g[key] for g in group if g.get(key) is not None]
            return statistics.fmean(values) if values else None

        rows.append({
            "problem": problem, "method": method, "kind": KIND[method], "n": n, "param": param,
            "runs": sum(g["runs"] for g in group), "mean_cost": fmt(mean_of("cost")),
            "std_cost": fmt(std), "min_cost": fmt(min(g["cost_min"] for g in group)),
            "max_cost": fmt(max(g["cost_max"] for g in group)),
            "optimal_value": fmt(mean_of("optimal")), "lower_bound": fmt(mean_of("lower_bound")),
            "lower_bound_kind": group[0]["lower_bound_kind"],
            "ratio_to_opt": fmt(mean_of("ratio_to_opt"), 6),
            "ratio_to_lower_bound": fmt(mean_of("ratio_to_lb"), 6),
            "proven_bound": fmt(mean_of("proven_bound"), 6),
            "mean_time_s": fmt(mean_of("time_s"), 6), "seed": group[0]["seed"],
        })
    return rows


def harmonic(k: int) -> float:
    """H(k) = 1 + 1/2 + ... + 1/k, the greedy set cover bound for sets of size k."""
    return sum(1.0 / i for i in range(1, k + 1))


# ----------------------------------------------------------------------
# Study 1: vertex cover
# ----------------------------------------------------------------------
def vertex_cover_study(cfg: Config, master: int, records: List[dict], cuts: List[str]) -> None:
    print("\n=== STUDY 1 - Vertex cover ===", flush=True)
    exact_on = True
    for n in cfg.vc_exact_sizes + cfg.vc_large_sizes:
        p = 4.0 / (n - 1)
        param = f"p={p:.4g}"
        run_exact = exact_on and n in cfg.vc_exact_sizes
        for rep in range(cfg.vc_reps):
            seed = derive_seed(master, "vertex_cover", n, rep)
            vertices, edges = random_graph(n, p, seed)
            opt = None
            (cover_lp, lp_value), t_lp = timed(lambda: vertex_cover_lp_rounding(vertices, edges))
            if run_exact:
                exact, t = timed(lambda: vertex_cover_exact(vertices, edges))
                opt = len(exact)
                record(records, "vertex_cover", "exact", n, param, rep, seed, [opt], t, optimal=opt,
                       lower_bound=lp_value, lower_bound_kind="lp")
                if t > cfg.vc_exact_cap_s:
                    exact_on = False
                    cuts.append(f"vertex cover exact stopped after n={n}: one run took {t:.1f} s")
            cover_2, t_2 = timed(lambda: vertex_cover_2approx(vertices, edges))
            matching = matching_lower_bound(vertices, edges)
            assert is_vertex_cover(edges, cover_lp) and is_vertex_cover(edges, cover_2)
            assert len(cover_2) == 2 * matching and lp_value >= matching - 1e-9
            for method, cover, t in (("two_approx", cover_2, t_2), ("lp_rounding", cover_lp, t_lp)):
                record(records, "vertex_cover", method, n, param, rep, seed, [len(cover)], t,
                       optimal=opt, lower_bound=lp_value, lower_bound_kind="lp", proven_bound=2.0,
                       matching_bound=matching)
        pts = [r for r in records if r["problem"] == "vertex_cover" and r["n"] == n]
        print(f"    n={n:<5} " + "  ".join(
            f"{m} size {statistics.fmean([q['cost'] for q in pts if q['method'] == m]):.1f}"
            for m in ("exact", "two_approx", "lp_rounding") if any(q["method"] == m for q in pts))
            + f"  LP {statistics.fmean([q['lower_bound'] for q in pts if q['method'] == 'lp_rounding']):.1f}",
            flush=True)


# ----------------------------------------------------------------------
# Study 2: set cover
# ----------------------------------------------------------------------
def set_cover_study(cfg: Config, master: int, records: List[dict],
                    samples: List[dict]) -> None:
    print("\n=== STUDY 2 - Set cover ===", flush=True)
    for n in cfg.sc_sizes:
        n_sets = n // 2
        param = "density=0.1"
        for rep in range(cfg.sc_reps):
            seed = derive_seed(master, "set_cover", n, rep)
            universe, subsets, costs = set_cover_instance(n, n_sets, 0.1, seed)
            lp, t_lp = timed(lambda: set_cover_lp(universe, subsets, costs))
            lp_value = lp[0]
            (chosen_x, opt), t_x = timed(lambda: set_cover_exact(universe, subsets, costs))
            f = frequency(universe, subsets)
            d = max(len(s) for s in subsets)
            record(records, "set_cover", "exact", n, param, rep, seed, [opt], t_x, optimal=opt,
                   lower_bound=lp_value, lower_bound_kind="lp")
            (chosen_g, cost_g), t_g = timed(lambda: set_cover_greedy(universe, subsets, costs))
            record(records, "set_cover", "greedy", n, param, rep, seed, [cost_g], t_g, optimal=opt,
                   lower_bound=lp_value, lower_bound_kind="lp", proven_bound=harmonic(d))
            (chosen_d, cost_d, _), t_d = timed(
                lambda: set_cover_deterministic_rounding(universe, subsets, costs, lp=lp))
            record(records, "set_cover", "deterministic_rounding", n, param, rep, seed, [cost_d],
                   t_d + t_lp, optimal=opt, lower_bound=lp_value, lower_bound_kind="lp",
                   proven_bound=float(f))
            run_costs, fixups, run_times = [], 0, []
            for run in range(cfg.sc_runs):
                r_seed = derive_seed(master, "set_cover_rounding", n, rep, run)
                started = time.perf_counter()
                chosen_r, cost_r, _, fix = set_cover_randomized_rounding(
                    universe, subsets, costs, seed=r_seed, lp=lp, return_fixups=True)
                run_times.append(time.perf_counter() - started)
                covered = set().union(*(subsets[i] for i in chosen_r)) if chosen_r else set()
                assert covered >= universe
                run_costs.append(cost_r)
                fixups += int(fix > 0)
            record(records, "set_cover", "randomized_rounding", n, param, rep, seed, run_costs,
                   statistics.fmean(run_times) + t_lp, optimal=opt, lower_bound=lp_value,
                   lower_bound_kind="lp", fixup_rate=fixups / cfg.sc_runs)
            if rep == 0:
                samples.append({"n": n, "ratios": [c / lp_value for c in run_costs],
                                "deterministic": cost_d / lp_value, "greedy": cost_g / lp_value})
        pts = [r for r in records if r["problem"] == "set_cover" and r["n"] == n]
        print(f"    n={n:<4} sets={n_sets:<4} " + "  ".join(
            f"{m} {statistics.fmean([q['ratio_to_opt'] for q in pts if q['method'] == m]):.3f}"
            for m in ("greedy", "deterministic_rounding", "randomized_rounding"))
            + f"  (ratio to OPT)  LP/OPT "
              f"{statistics.fmean([q['lower_bound'] / q['optimal'] for q in pts if q['method'] == 'exact']):.3f}",
            flush=True)


# ----------------------------------------------------------------------
# Study 3: facility location
# ----------------------------------------------------------------------
def facility_study(cfg: Config, master: int, records: List[dict]) -> None:
    print("\n=== STUDY 3 - Facility location ===", flush=True)
    for clients, facilities in cfg.fl_sizes:
        param = f"facilities={facilities}"
        for rep in range(cfg.fl_reps):
            seed = derive_seed(master, "facility", clients, facilities, rep)
            open_costs, dist = facility_instance(clients, facilities, seed)
            (open_x, opt), t_x = timed(lambda: facility_location_exact(open_costs, dist))
            (open_g, cost_g), t_g = timed(lambda: facility_location_greedy(open_costs, dist))
            assert cost_g >= opt - 1e-6
            record(records, "facility_location", "exact", clients, param, rep, seed, [opt], t_x,
                   optimal=opt)
            record(records, "facility_location", "facility_greedy", clients, param, rep, seed,
                   [cost_g], t_g, optimal=opt)
        pts = [r for r in records if r["problem"] == "facility_location" and r["n"] == clients
               and r["method"] == "facility_greedy"]
        print(f"    clients={clients:<4} facilities={facilities:<3} greedy/OPT "
              f"{statistics.fmean([q['ratio_to_opt'] for q in pts]):.4f} "
              f"(max {max(q['ratio_to_opt'] for q in pts):.4f})", flush=True)


# ----------------------------------------------------------------------
# Study 4: metric TSP
# ----------------------------------------------------------------------
def tsp_study(cfg: Config, master: int, records: List[dict]) -> None:
    print("\n=== STUDY 4 - Metric TSP ===", flush=True)
    for n in cfg.tsp_exact_sizes + cfg.tsp_large_sizes:
        exact_point = n in cfg.tsp_exact_sizes
        reps = cfg.tsp_reps if exact_point else cfg.tsp_large_reps
        for rep in range(reps):
            seed = derive_seed(master, "tsp", n, rep)
            _, dist = euclidean_instance(n, seed)
            assert is_metric(dist), "MST doubling is only run on metric instances"
            (mst, _), _ = timed(lambda: mst_weight_and_edges(dist))
            opt = None
            if exact_point:
                (opt, opt_tour), t = timed(lambda: tsp_bitmask(dist))
                record(records, "metric_tsp", "exact", n, "euclidean", rep, seed, [opt], t,
                       optimal=opt, lower_bound=mst, lower_bound_kind="mst")
            for method, solver, bound in (
                ("mst_doubling", tsp_mst_doubling, 2.0),
                ("christofides_networkx", tsp_christofides_networkx, 1.5),
                ("nearest_neighbor", tsp_nearest_neighbor, None),
            ):
                (cost, tour), t = timed(lambda: solver(dist))
                assert is_valid_tour(tour, n)
                if bound is not None:
                    assert cost <= bound * (opt if opt else 2 * mst) + 1e-6
                record(records, "metric_tsp", method, n, "euclidean", rep, seed, [cost], t,
                       optimal=opt, lower_bound=mst, lower_bound_kind="mst", proven_bound=bound)
        pts = [r for r in records if r["problem"] == "metric_tsp" and r["n"] == n]
        line = []
        for method in ("mst_doubling", "christofides_networkx", "nearest_neighbor"):
            mine = [q for q in pts if q["method"] == method]
            key = "ratio_to_opt" if exact_point else "ratio_to_lb"
            line.append(f"{method} {statistics.fmean([q[key] for q in mine]):.3f}")
        print(f"    n={n:<4} ({'to OPT' if exact_point else 'to MST'}) " + "  ".join(line), flush=True)


# ----------------------------------------------------------------------
# Study 5: Max-Cut
# ----------------------------------------------------------------------
def maxcut_study(cfg: Config, master: int, records: List[dict], samples: List[dict]) -> None:
    print("\n=== STUDY 5 - Max-Cut ===", flush=True)
    for n in cfg.mc_exact_sizes + cfg.mc_large_sizes:
        exact_point = n in cfg.mc_exact_sizes
        starts = cfg.mc_starts if exact_point else cfg.mc_large_starts
        for rep in range(cfg.mc_reps):
            seed = derive_seed(master, "maxcut", n, rep)
            vertices, edges = random_graph(n, 0.3, seed)
            m = len(edges)
            opt = None
            if exact_point:
                (opt, _), t = timed(lambda: maxcut_exact(vertices, edges))
                record(records, "maxcut", "exact", n, "p=0.3", rep, seed, [opt], t, optimal=opt,
                       lower_bound=m, lower_bound_kind="edges_over_1")
            cuts, times = [], []
            for run in range(cfg.mc_runs):
                r_seed = derive_seed(master, "maxcut_random", n, rep, run)
                started = time.perf_counter()
                cut, _ = maxcut_random(vertices, edges, seed=r_seed)
                times.append(time.perf_counter() - started)
                cuts.append(cut)
            record(records, "maxcut", "random_cut", n, "p=0.3", rep, seed, cuts,
                   statistics.fmean(times), optimal=opt, lower_bound=m,
                   lower_bound_kind="edges_over_1", proven_bound=0.5)
            best, ls_times = 0, []
            for start in range(starts):
                s_seed = derive_seed(master, "maxcut_local", n, rep, start)
                started = time.perf_counter()
                cut, side, _ = maxcut_local_search(vertices, edges, seed=s_seed)
                ls_times.append(time.perf_counter() - started)
                assert 2 * cut >= m
                best = max(best, cut)
            record(records, "maxcut", "local_search", n, "p=0.3", rep, seed, [best],
                   sum(ls_times), optimal=opt, lower_bound=m, lower_bound_kind="edges_over_1",
                   proven_bound=0.5)
            if rep == 0 and len(samples) < 3 and n in (cfg.mc_exact_sizes[-1:] + cfg.mc_exact_sizes[:1]
                                                        + cfg.mc_large_sizes[:1]):
                samples.append({"n": n, "edges": m, "cuts": cuts, "local": best, "optimal": opt})
        pts = [r for r in records if r["problem"] == "maxcut" and r["n"] == n]
        key = "ratio_to_opt" if exact_point else "ratio_to_lb"
        print(f"    n={n:<4} ({'to OPT' if exact_point else 'to |E|'}) " + "  ".join(
            f"{m_} {statistics.fmean([q[key] for q in pts if q['method'] == m_]):.3f}"
            for m_ in ("random_cut", "local_search")), flush=True)


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """Write ``rows`` with exactly ``columns``; ``None`` becomes an empty cell."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r.get(c) is None else
                                 fmt(r[c]) if isinstance(r.get(c), float) else r[c])
                             for c in columns})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny parameter set, seconds")
    parser.add_argument("--out", default=RESULTS_DIR, help="directory for the CSVs and figures")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed (default 42)")
    args = parser.parse_args(argv)

    cfg = SMOKE if args.smoke else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    master = args.seed
    apply_house_style()

    print("=" * 78)
    print("CSC 5300 Week 9 - approximation algorithms benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    : {platform.platform()}")
    print(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  master seed : {master}  (instances: derive_seed(master, experiment, n, rep))")
    print(f"  output      : {out}", flush=True)

    started = time.perf_counter()
    records: List[dict] = []
    cuts: List[str] = []
    setcover_samples: List[dict] = []
    maxcut_samples: List[dict] = []
    vertex_cover_study(cfg, master, records, cuts)
    set_cover_study(cfg, master, records, setcover_samples)
    facility_study(cfg, master, records)
    tsp_study(cfg, master, records)
    maxcut_study(cfg, master, records, maxcut_samples)

    print("\n=== writing results ===", flush=True)
    write_csv(os.path.join(out, "comparison_table.csv"), aggregate(records), COLUMNS)
    write_csv(os.path.join(out, "instances.csv"), records, INSTANCE_COLUMNS)
    for line in cuts:
        print(f"  size cut: {line}")

    print("\n=== figures ===", flush=True)
    for name, builder in (
        ("approx_ratios.png", lambda p: plot_approx_ratios(records, save_path=p)),
        ("cost_vs_optimal.png", lambda p: plot_cost_vs_optimal(records, save_path=p)),
        ("randomized_maxcut_results.png",
         lambda p: plot_randomized_results(maxcut_samples, setcover_samples, save_path=p)),
    ):
        path = os.path.join(out, name)
        builder(path)
        print(f"  wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)", flush=True)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
