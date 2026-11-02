#!/usr/bin/env python3
"""Week 10 benchmark: one residual graph, read different ways.

Run from the repository root with the project virtualenv active::

    python benchmarks/week10_flow_benchmark.py                 # the full study (under 5 minutes)
    python benchmarks/week10_flow_benchmark.py --smoke         # seconds, tiny sizes
    python benchmarks/week10_flow_benchmark.py --seed 7 --out /tmp/w10

Six experiments on seeded random networks:

1. **Scaling** - Edmonds-Karp (EK), FIFO push-relabel (PR) and min-cost
   max-flow (MCF) on sparse graphs from n = 1,000 to 50,000.
2. **Sparse against dense** - average degree 4 against n/4.
3. **Capacity distributions** - uniform, exponential and unit capacities.
4. **Memory** - peak ``tracemalloc`` memory during each solver. The network
   is built before tracing starts, so the figure is the solver's working
   memory, and these runs are not timed because tracing slows them.
5. **Applications** - bipartite matching, checked against networkx, and image
   segmentation of the disc image.
6. **Terminal sensitivity** - a control on the generator. The source and sink
   normally connect to 5% of the nodes; here each gets a single hub edge, so
   the max flow is limited by the few edges around them, which shows how much
   of the gap between the solvers comes from that choice.

Every flow result is validated, and EK and PR must give the same value on
every timed instance. Min-cost flow gets ``validators.validate_flow``; EK and
PR get its four flow checks without the cost check, since a maximum flow on a
costed graph need not be a cheapest one. The applications keep their networks
internal, so matching is checked against networkx and for disjoint input
edges, and segmentation by identical masks and cuts from both solvers.

**The 20-second rule.** A run that passes the cap is aborted by a timer and
never reported as data. That solver is then skipped at larger sizes of the
same experiment and graph type, and the stop is written to
``size_caps.csv``.

**Reproducibility.** Every instance comes from
``derive_seed(master_seed, <experiment>, ..., n, rep)``; the seed of rep 0 is
the ``seed`` column. Two runs with the same ``--seed`` produce identical
non-timing columns unless a timing-dependent cap fires differently.

**Where the results go.** ``benchmarks/results/week10/``, which holds
``flow_benchmark.csv``, ``instances.csv``, ``size_caps.csv``,
``pr_convergence.csv``, the five benchmark figures, ``network.png``, ``matching.png``, ``segmentation.png``,
``ek_animation.gif`` and ``pr_animation.gif``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import signal
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from src.flow_network_solver.applications.image_seg import save_segmentation, segment  # noqa: E402
from src.flow_network_solver.applications.matching import (  # noqa: E402
    bipartite_matching,
    draw_matching,
)
from src.flow_network_solver.core.edmonds_karp import edmonds_karp  # noqa: E402
from src.flow_network_solver.core.min_cost_flow import min_cost_max_flow  # noqa: E402
from src.flow_network_solver.core.network import FlowResult, Network, min_cut  # noqa: E402
from src.flow_network_solver.core.push_relabel import push_relabel  # noqa: E402
from src.flow_network_solver.utils.generators import (  # noqa: E402
    bipartite_graph,
    grid_image,
    random_network,
)
from src.flow_network_solver.utils.validators import (  # noqa: E402
    check_capacity,
    check_conservation,
    check_flow_value,
    check_residual,
    validate_flow,
)
from src.flow_network_solver.visualization.animate_flow import (  # noqa: E402
    animate_edmonds_karp,
    animate_push_relabel,
)
from src.flow_network_solver.visualization.plot_network import save_network_figure  # noqa: E402
from src.utils.random_utils import DEFAULT_SEED, derive_seed  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_augmentations,
    plot_memory_use,
    plot_pr_convergence,
    plot_runtime_scaling,
    plot_sparse_vs_dense,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week10")

#: The CSV's columns are fixed by the specification.
COLUMNS = [
    "experiment", "solver", "graph_type", "n", "m", "cap_dist", "runs", "max_flow", "cost",
    "mean_time_s", "std_time_s", "augmentations", "pushes", "relabels", "peak_mem_kb", "valid",
    "ek_pr_agree", "seed",
]
TIMING_COLUMNS = ["mean_time_s", "std_time_s"]

INSTANCE_COLUMNS = [
    "experiment", "solver", "graph_type", "n", "m", "cap_dist", "rep", "seed", "max_flow", "cost",
    "time_s", "augmentations", "pushes", "relabels", "global_relabels", "peak_mem_kb", "valid",
    "ek_pr_agree",
]
CAP_COLUMNS = ["experiment", "solver", "graph_type", "n", "cap_s", "reason"]

SOLVERS: Dict[str, Callable[..., FlowResult]] = {
    "ek": edmonds_karp, "pr": push_relabel, "mcf": min_cost_max_flow,
}
#: Costs are drawn for every scaling instance so all three solvers share one graph.
COST_RANGE = (1, 20)
CLRS_EDGES = [(0, 1, 16), (0, 2, 13), (2, 1, 4), (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
              (3, 5, 20), (4, 5, 4)]


@dataclass
class Config:
    """Every size and repeat count of one benchmark mode."""

    run_cap_s: float
    memory_cap_s: float
    scaling_sizes: List[int]
    reps: int
    dense_sizes: List[int]
    dist_n: int
    memory_sizes: List[int]
    matching_sizes: List[int]
    image_sizes: List[int]
    convergence_sizes: List[int]
    control_n: int


FULL = Config(
    run_cap_s=20.0, memory_cap_s=30.0,
    scaling_sizes=[1_000, 2_000, 5_000, 10_000, 20_000, 50_000], reps=3,
    dense_sizes=[200, 500, 1_000], dist_n=5_000, memory_sizes=[1_000, 5_000, 10_000],
    matching_sizes=[50, 200, 1_000], image_sizes=[32, 64],
    convergence_sizes=[5_000, 20_000, 50_000], control_n=10_000,
)

SMOKE = Config(
    run_cap_s=5.0, memory_cap_s=5.0,
    scaling_sizes=[100, 200, 400], reps=2,
    dense_sizes=[40, 80], dist_n=200, memory_sizes=[100, 200],
    matching_sizes=[20, 40], image_sizes=[16],
    convergence_sizes=[200, 400], control_n=200,
)


class RunCapExceeded(Exception):
    """Raised by the alarm when one run passes the cap."""


def capped(function: Callable[[], Any], cap_s: float) -> Any:
    """Call ``function``, raising :class:`RunCapExceeded` if it passes ``cap_s`` seconds."""
    def on_alarm(signum: int, frame: Any) -> None:
        raise RunCapExceeded()

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, cap_s)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fmt(value: Optional[float], digits: int = 8) -> str:
    """Format a number for the CSV; ``None`` becomes an empty cell."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def validate(net: Network, solver: str, s: int, t: int, value: int) -> Tuple[bool, str]:
    """Validate a result against what its solver promises.

    The graphs here carry costs so all three solvers share one instance. A
    maximum flow need not be a cheapest one, so ``validate_flow``, which also
    demands no negative-cost residual cycle, applies to min-cost flow only.
    Edmonds-Karp and push-relabel get the four flow checks.
    """
    if solver == "mcf":
        return validate_flow(net, s, t, value)
    reports = [check_capacity(net), check_conservation(net, s, t), check_residual(net),
               check_flow_value(net, s, t, value)]
    failures = [message for ok, message in reports if not ok]
    return (False, "; ".join(failures)) if failures else (True, "all checks passed")


def run_solver(net: Network, solver: str, s: int, t: int, cap_s: float) -> Tuple[FlowResult, bool]:
    """Reset ``net``, run ``solver`` under the cap and validate the result."""
    net.reset()
    result = capped(lambda: SOLVERS[solver](net, s, t), cap_s)
    ok, message = validate(net, solver, s, t, result.value)
    if not ok:
        print(f"    INVALID {solver}: {message}", flush=True)
    return result, ok


def record(records: List[dict], experiment: str, solver: str, graph_type: str, n: int, m: Any,
           cap_dist: str, rep: int, seed: int, result: Optional[FlowResult] = None,
           **fields: Any) -> dict:
    """Append one per-instance record; solver counters come from ``result``."""
    entry = {"experiment": experiment, "solver": solver, "graph_type": graph_type, "n": n, "m": m,
             "cap_dist": cap_dist, "rep": rep, "seed": seed, "max_flow": None, "cost": None,
             "time_s": None, "augmentations": None, "pushes": None, "relabels": None,
             "global_relabels": None, "peak_mem_kb": None, "valid": True, "ek_pr_agree": None}
    if result is not None:
        entry.update(max_flow=result.value, cost=result.cost, time_s=result.time_s)
        if solver == "ek":
            entry["augmentations"] = result.iterations
        elif solver == "mcf":
            entry["augmentations"] = result.augmentations
        else:
            entry.update(pushes=result.pushes, relabels=result.relabels,
                         global_relabels=result.global_relabels)
    entry.update(fields)
    records.append(entry)
    return entry


def note_cap(caps: List[dict], experiment: str, solver: str, graph_type: str, n: int,
             cap_s: float) -> None:
    """Record that ``solver`` passed the cap at size ``n``."""
    caps.append({"experiment": experiment, "solver": solver, "graph_type": graph_type, "n": n,
                 "cap_s": cap_s, "reason": f"a run passed {cap_s:.0f} s and was aborted"})
    print(f"    [cap] {experiment}/{graph_type}/{solver}: a run passed {cap_s:.0f} s at n={n}; "
          f"skipped at larger sizes", flush=True)


def solve_instance(records: List[dict], caps: List[dict], active: Dict[str, bool],
                   experiment: str, graph_type: str, n: int, cap_dist: str, rep: int, seed: int,
                   net: Network, cfg: Config, solvers: Sequence[str]) -> Dict[str, FlowResult]:
    """Run each active solver on one instance and cross-check EK against PR."""
    s, t = 0, net.n - 1
    entries: Dict[str, dict] = {}
    results: Dict[str, FlowResult] = {}
    for solver in solvers:
        if not active[solver]:
            continue
        try:
            result, ok = run_solver(net, solver, s, t, cfg.run_cap_s)
        except RunCapExceeded:
            active[solver] = False
            note_cap(caps, experiment, solver, graph_type, n, cfg.run_cap_s)
            continue
        results[solver] = result
        entries[solver] = record(records, experiment, solver, graph_type, n, net.m, cap_dist, rep,
                                 seed, result, valid=ok)
    if "ek" in results and "pr" in results:
        agree = results["ek"].value == results["pr"].value
        if not agree:
            print(f"    MISMATCH at n={n} rep={rep}: EK {results['ek'].value} "
                  f"PR {results['pr'].value}", flush=True)
        entries["ek"]["ek_pr_agree"] = entries["pr"]["ek_pr_agree"] = agree
    if "mcf" in results and "pr" in results:
        assert results["mcf"].value == results["pr"].value, "MCF did not send the max flow"
    return results


def summary_line(records: Sequence[dict], experiment: str, n: int, graph_type: str) -> str:
    """Mean time per solver at one size, for the progress log."""
    parts = []
    for solver in SOLVERS:
        mine = [r["time_s"] for r in records if r["experiment"] == experiment and r["n"] == n
                and r["graph_type"] == graph_type and r["solver"] == solver]
        if mine:
            parts.append(f"{solver} {statistics.fmean(mine):8.3f} s")
    return "  ".join(parts)


# ----------------------------------------------------------------------
# The experiments
# ----------------------------------------------------------------------
def scaling_study(cfg: Config, master: int, records: List[dict], caps: List[dict]) -> List[dict]:
    print("\n=== EXPERIMENT 1 - Scaling, sparse ===", flush=True)
    active = {solver: True for solver in SOLVERS}
    profiles = []
    for n in cfg.scaling_sizes:
        for rep in range(cfg.reps):
            seed = derive_seed(master, "scaling", n, rep)
            net = random_network(n, 4, seed, cost_range=COST_RANGE)
            results = solve_instance(records, caps, active, "scaling", "sparse", n, "uniform", rep,
                                     seed, net, cfg, ("pr", "ek", "mcf"))
            if rep == 0 and n in cfg.convergence_sizes and "pr" in results:
                profiles.append({"n": n, "convergence": results["pr"].convergence})
        print(f"    n={n:<7} {summary_line(records, 'scaling', n, 'sparse')}", flush=True)
    return profiles


def sparse_dense_study(cfg: Config, master: int, records: List[dict], caps: List[dict]) -> None:
    print("\n=== EXPERIMENT 2 - Sparse against dense ===", flush=True)
    for graph_type in ("sparse", "dense"):
        active = {solver: True for solver in SOLVERS}
        for n in cfg.dense_sizes:
            degree = 4 if graph_type == "sparse" else n / 4
            for rep in range(cfg.reps):
                seed = derive_seed(master, "sparse_vs_dense", graph_type, n, rep)
                net = random_network(n, degree, seed, cost_range=COST_RANGE)
                solve_instance(records, caps, active, "sparse_vs_dense", graph_type, n, "uniform",
                               rep, seed, net, cfg, ("pr", "ek", "mcf"))
            print(f"    {graph_type:<6} n={n:<6} "
                  f"{summary_line(records, 'sparse_vs_dense', n, graph_type)}", flush=True)


def distribution_study(cfg: Config, master: int, records: List[dict], caps: List[dict]) -> None:
    print("\n=== EXPERIMENT 3 - Capacity distributions ===", flush=True)
    n = cfg.dist_n
    for cap_dist in ("uniform", "exponential", "unit"):
        active = {solver: True for solver in SOLVERS}
        for rep in range(cfg.reps):
            seed = derive_seed(master, "capacity_dist", cap_dist, n, rep)
            net = random_network(n, 4, seed, cap_dist=cap_dist)
            solve_instance(records, caps, active, "capacity_dist", "sparse", n, cap_dist, rep, seed,
                           net, cfg, ("pr", "ek"))
        mine = [r for r in records if r["experiment"] == "capacity_dist" and r["cap_dist"] == cap_dist]
        ek = [r["augmentations"] for r in mine if r["solver"] == "ek"]
        ops = [r["pushes"] + r["relabels"] for r in mine if r["solver"] == "pr"]
        print(f"    {cap_dist:<12} EK augmentations {statistics.fmean(ek) if ek else 0:10,.0f}  "
              f"PR operations {statistics.fmean(ops) if ops else 0:12,.0f}", flush=True)


def memory_study(cfg: Config, master: int, records: List[dict], caps: List[dict]) -> None:
    print("\n=== EXPERIMENT 4 - Memory ===", flush=True)
    active = {solver: True for solver in SOLVERS}
    for n in cfg.memory_sizes:
        seed = derive_seed(master, "memory", n, 0)
        net = random_network(n, 4, seed, cost_range=COST_RANGE)
        for solver in SOLVERS:
            if not active[solver]:
                continue
            net.reset()
            tracemalloc.start()
            try:
                result = capped(lambda s=solver: SOLVERS[s](net, 0, n - 1), cfg.memory_cap_s)
                _, peak = tracemalloc.get_traced_memory()
            except RunCapExceeded:
                active[solver] = False
                note_cap(caps, "memory", solver, "sparse", n, cfg.memory_cap_s)
                continue
            finally:
                tracemalloc.stop()
            ok, _ = validate(net, solver, 0, n - 1, result.value)
            record(records, "memory", solver, "sparse", n, net.m, "uniform", 0, seed, result,
                   time_s=None, peak_mem_kb=peak / 1024, valid=ok)
        peaks = {r["solver"]: r["peak_mem_kb"] for r in records
                 if r["experiment"] == "memory" and r["n"] == n}
        print(f"    n={n:<6} " + "  ".join(f"{k} {v:10,.0f} KiB" for k, v in peaks.items()), flush=True)


def application_study(cfg: Config, master: int, records: List[dict], caps: List[dict]) -> None:
    print("\n=== EXPERIMENT 5 - Applications ===", flush=True)
    for side in cfg.matching_sizes:
        p = 5.0 / side
        for rep in range(cfg.reps):
            seed = derive_seed(master, "matching", side, rep)
            left, right, edges = bipartite_graph(side, side, p, seed)
            graph = nx.Graph()
            graph.add_nodes_from(left, bipartite=0)
            graph.add_nodes_from(right, bipartite=1)
            graph.add_edges_from(edges)
            expected = len(nx.bipartite.hopcroft_karp_matching(graph, top_nodes=left)) // 2
            allowed = set(edges)
            sizes = {}
            entries = []
            for solver in ("ek", "pr"):
                started = time.perf_counter()
                try:
                    pairs = capped(lambda s=solver: bipartite_matching(left, right, edges, solver=s),
                                   cfg.run_cap_s)
                except RunCapExceeded:
                    note_cap(caps, "matching", solver, "bipartite", side * 2, cfg.run_cap_s)
                    continue
                elapsed = time.perf_counter() - started
                used_left = [a for a, _ in pairs]
                used_right = [b for _, b in pairs]
                ok = (len(pairs) == expected and set(pairs) <= allowed
                      and len(set(used_left)) == len(used_left)
                      and len(set(used_right)) == len(used_right))
                sizes[solver] = len(pairs)
                entries.append(record(records, "matching", solver, "bipartite", side * 2, len(edges),
                                      "unit", rep, seed, max_flow=len(pairs), time_s=elapsed, valid=ok))
            if len(sizes) == 2:
                for entry in entries:
                    entry["ek_pr_agree"] = sizes["ek"] == sizes["pr"]
        print(f"    matching {side:>5} per side  "
              f"{summary_line(records, 'matching', side * 2, 'bipartite')}  sizes {sizes}", flush=True)
    for side in cfg.image_sizes:
        for rep in range(cfg.reps):
            seed = derive_seed(master, "segmentation", side, rep)
            image = grid_image(side, side, seed)
            masks, cuts, entries = {}, {}, []
            for solver in ("pr", "ek"):
                started = time.perf_counter()
                try:
                    masks[solver], cuts[solver] = capped(lambda s=solver: segment(image, solver=s),
                                                         cfg.run_cap_s)
                except RunCapExceeded:
                    note_cap(caps, "segmentation", solver, "grid", side * side, cfg.run_cap_s)
                    continue
                elapsed = time.perf_counter() - started
                entries.append(record(records, "segmentation", solver, "grid", side * side, "",
                                      "scaled_int", rep, seed, max_flow=cuts[solver], time_s=elapsed))
            if len(masks) == 2:
                agree = masks["ek"] == masks["pr"] and math.isclose(cuts["ek"], cuts["pr"], abs_tol=1e-6)
                for entry in entries:
                    entry["ek_pr_agree"] = agree
                    entry["valid"] = agree
        print(f"    segmentation {side}x{side}  "
              f"{summary_line(records, 'segmentation', side * side, 'grid')}  cuts {cuts}", flush=True)


def terminal_study(cfg: Config, master: int, records: List[dict], caps: List[dict]) -> None:
    print("\n=== EXPERIMENT 6 - Terminal sensitivity ===", flush=True)
    n = cfg.control_n
    active = {solver: True for solver in SOLVERS}
    for rep in range(cfg.reps):
        seed = derive_seed(master, "terminal_sensitivity", n, rep)
        net = random_network(n, 4, seed, cost_range=COST_RANGE, terminal_fraction=0.0)
        solve_instance(records, caps, active, "terminal_sensitivity", "sparse_one_hub", n, "uniform",
                       rep, seed, net, cfg, ("pr", "ek", "mcf"))
    print(f"    n={n:<7} {summary_line(records, 'terminal_sensitivity', n, 'sparse_one_hub')}", flush=True)


def application_figures(master: int, out: str) -> None:
    """Save the anchor network, both animations, and the two application images."""
    print("\n=== anchor figures and animations ===", flush=True)
    net = Network.from_edges(6, CLRS_EDGES)
    result = edmonds_karp(net, 0, 5)
    assert result.value == 23 and min_cut(net, 0) == {0, 1, 2, 4}
    save_network_figure(net, 0, 5, os.path.join(out, "network.png"))
    animate_edmonds_karp(net, 0, 5, os.path.join(out, "ek_animation.gif"))
    animate_push_relabel(net, 0, 5, os.path.join(out, "pr_animation.gif"))
    left, right, edges = bipartite_graph(8, 8, 0.25, derive_seed(master, "matching_figure", 8, 0))
    draw_matching(left, right, edges, bipartite_matching(left, right, edges),
                  os.path.join(out, "matching.png"))
    image = grid_image(64, 64, derive_seed(master, "segmentation_figure", 64, 0))
    mask, _ = segment(image)
    save_segmentation(image, mask, os.path.join(out, "segmentation.png"))


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def aggregate(records: Sequence[dict]) -> List[dict]:
    """Collapse per-instance records into CSV rows, one per point."""
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        groups.setdefault((r["experiment"], r["solver"], r["graph_type"], r["n"], r["cap_dist"]), []).append(r)
    rows = []
    for (experiment, solver, graph_type, n, cap_dist), group in groups.items():
        def mean_of(key: str) -> Optional[float]:
            values = [g[key] for g in group if g.get(key) not in (None, "")]
            return statistics.fmean(values) if values else None

        times = [g["time_s"] for g in group if g["time_s"] is not None]
        agree = [g["ek_pr_agree"] for g in group if g["ek_pr_agree"] is not None]
        rows.append({
            "experiment": experiment, "solver": solver, "graph_type": graph_type, "n": n,
            "m": fmt(mean_of("m")), "cap_dist": cap_dist, "runs": len(group),
            "max_flow": fmt(mean_of("max_flow"), 10), "cost": fmt(mean_of("cost"), 10),
            "mean_time_s": fmt(statistics.fmean(times), 6) if times else "",
            "std_time_s": fmt(statistics.stdev(times), 6) if len(times) > 1 else "",
            "augmentations": fmt(mean_of("augmentations")), "pushes": fmt(mean_of("pushes")),
            "relabels": fmt(mean_of("relabels")), "peak_mem_kb": fmt(mean_of("peak_mem_kb"), 6),
            "valid": all(g["valid"] for g in group),
            "ek_pr_agree": all(agree) if agree else "", "seed": group[0]["seed"],
        })
    return rows


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
    print("CSC 5300 Week 10 - network flows benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    : {platform.platform()}")
    print(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  master seed : {master}  (instances: derive_seed(master, experiment, ..., n, rep))")
    print(f"  run cap     : {cfg.run_cap_s:.0f} s per run")
    print(f"  output      : {out}", flush=True)

    started = time.perf_counter()
    records: List[dict] = []
    caps: List[dict] = []
    profiles = scaling_study(cfg, master, records, caps)
    sparse_dense_study(cfg, master, records, caps)
    distribution_study(cfg, master, records, caps)
    memory_study(cfg, master, records, caps)
    application_study(cfg, master, records, caps)
    terminal_study(cfg, master, records, caps)

    print("\n=== writing results ===", flush=True)
    write_csv(os.path.join(out, "flow_benchmark.csv"), aggregate(records), COLUMNS)
    write_csv(os.path.join(out, "instances.csv"), records, INSTANCE_COLUMNS)
    write_csv(os.path.join(out, "size_caps.csv"), caps, CAP_COLUMNS)
    write_csv(os.path.join(out, "pr_convergence.csv"),
              [{"n": p["n"], "operations": o, "active_nodes": a, "total_excess": e}
               for p in profiles for o, a, e in p["convergence"]],
              ["n", "operations", "active_nodes", "total_excess"])
    mismatches = sum(1 for r in records if r["ek_pr_agree"] is False)
    invalid = sum(1 for r in records if not r["valid"])
    print(f"  {len(records)} solver runs; {invalid} invalid; {mismatches} EK/PR mismatches", flush=True)

    application_figures(master, out)

    print("\n=== figures ===", flush=True)
    by = lambda name: [r for r in records if r["experiment"] == name]  # noqa: E731
    for name, builder in (
        ("runtime_scaling.png",
         lambda p: plot_runtime_scaling(by("scaling"), [c for c in caps if c["experiment"] == "scaling"],
                                        save_path=p)),
        ("sparse_vs_dense.png", lambda p: plot_sparse_vs_dense(by("sparse_vs_dense"), save_path=p)),
        ("augmentations.png",
         lambda p: plot_augmentations(by("scaling"), by("capacity_dist"), save_path=p)),
        ("pr_convergence.png", lambda p: plot_pr_convergence(profiles, save_path=p)),
        ("memory_use.png", lambda p: plot_memory_use(by("memory"), save_path=p)),
    ):
        path = os.path.join(out, name)
        figure = builder(path)
        plt.close(figure)
        print(f"  wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)", flush=True)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
