#!/usr/bin/env python3
"""Week 11 benchmark: what each method stores before it searches, and about which string.

The instructions call this ``benchmark_search.py``; it is prefixed with the
week like every earlier benchmark. Run from the repository root with the
project virtualenv active::

    python benchmarks/week11_benchmark_search.py                 # the full study (under 5 minutes)
    python benchmarks/week11_benchmark_search.py --smoke         # seconds, tiny sizes
    python benchmarks/week11_benchmark_search.py --seed 7 --out /tmp/w11

Six experiments on seeded synthetic text (nothing is downloaded, and no
text is real data):

1. **Dataset sizes** - KMP, Rabin-Karp (RK), the suffix array (SA) and the
   suffix tree (ST) on DNA and English-like text from 10 KB to 5 MB, with
   pattern length 16. Index builds are timed once; every query is timed.
2. **Break-even** - cumulative time for 1 to 200 queries on 1 MB of DNA,
   computed from experiment 1's measured build and query means.
3. **Rabin-Karp collisions** - spurious hits against the modulus, and
   against the base at a fixed modulus.
4. **Worst case** - naive search, KMP and RK on ``"a" * n``.
5. **Memory** - peak ``tracemalloc`` memory while preprocessing, in
   separate untimed runs.
6. **LCP statistics** - mean and maximum LCP, distinct substrings and the
   longest repeated substring for DNA, English-like and periodic text.

Every query's positions are checked against KMP's; the ``agree`` column
records the result.

**The 20-second rule.** The suffix tree is built only while one build stays
under 20 s. Before each larger size the build time is predicted from the
last measured one; if the prediction passes 20 s, or a build does, the tree
is skipped at that size and every larger one and the cap is written to
``size_caps.csv``. No skipped point is filled in.

**Reproducibility.** Every text and pattern set comes from
``derive_seed(master_seed, <experiment>, ..., n, rep)``; the seed of rep 0 is
the ``seed`` column.

**Where the results go.** ``benchmarks/results/week11/``: ``search_benchmark.csv``,
``size_caps.csv``, the five benchmark figures, ``suffix_array_mississippi.png``,
``suffix_tree_banana.png`` and ``suffix_tree_banana.dot``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import signal
import statistics
import string
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

from src.string_search_engine.core.kmp import compute_lps, kmp_search, naive_search  # noqa: E402
from src.string_search_engine.core.rabin_karp import rabin_karp_search  # noqa: E402
from src.string_search_engine.core.suffix_array import SuffixArray  # noqa: E402
from src.string_search_engine.core.suffix_tree import SuffixTree  # noqa: E402
from src.string_search_engine.utils.generators import (  # noqa: E402
    dna_text,
    english_like_text,
    periodic_text,
    random_text,
    sample_patterns,
    worst_case_rk,
)
from src.string_search_engine.utils.rolling_hash import RollingHash  # noqa: E402
from src.string_search_engine.visualization.draw_suffix_tree import (  # noqa: E402
    draw_suffix_tree,
    write_dot,
)
from src.string_search_engine.visualization.plot_suffix_array import plot_suffix_array  # noqa: E402
from src.utils.random_utils import DEFAULT_SEED, derive_seed, make_rng  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_collision_analysis,
    plot_heatmap_dataset_sizes,
    plot_index_memory,
    plot_runtime_comparison,
    plot_sa_lcp_stats,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week11")

#: The CSV's columns are fixed by the specification.
COLUMNS = [
    "experiment", "method", "dataset", "n", "m", "queries", "preprocess_s", "mean_query_s",
    "std_query_s", "matches", "comparisons", "spurious_hits", "expected_spurious", "base", "modulus",
    "peak_mem_kb", "mean_lcp", "max_lcp", "distinct_substrings", "lrs_length", "agree", "seed",
]
TIMING_COLUMNS = ["preprocess_s", "mean_query_s", "std_query_s", "peak_mem_kb"]
CAP_COLUMNS = ["method", "dataset", "n", "reason"]

DEFAULT_MODULUS = (1 << 61) - 1
METHODS = ("kmp", "rk", "sa", "st")


@dataclass
class Config:
    """Every size and count of one benchmark mode."""

    sizes: List[int]
    m: int
    queries: int
    queries_largest: int
    stats_queries: int
    tree_cap_s: float
    breakeven_n: int
    breakeven_max_q: int
    collision_n: int
    collision_m: int
    collision_patterns: int
    worst_n: int
    worst_m: int
    memory_sizes: List[int]
    memory_cap_s: float
    lcp_sizes: List[int]
    histogram_n: int


FULL = Config(
    sizes=[10_000, 100_000, 1_000_000, 5_000_000], m=16, queries=20, queries_largest=5,
    stats_queries=5, tree_cap_s=20.0, breakeven_n=1_000_000, breakeven_max_q=200,
    collision_n=100_000, collision_m=8, collision_patterns=20, worst_n=100_000, worst_m=64,
    memory_sizes=[10_000, 100_000, 1_000_000], memory_cap_s=60.0,
    lcp_sizes=[10_000, 100_000, 1_000_000], histogram_n=100_000,
)

SMOKE = Config(
    sizes=[2_000, 10_000], m=16, queries=4, queries_largest=4, stats_queries=2, tree_cap_s=20.0,
    breakeven_n=10_000, breakeven_max_q=50, collision_n=5_000, collision_m=8, collision_patterns=4,
    worst_n=2_000, worst_m=16, memory_sizes=[2_000, 10_000], memory_cap_s=30.0,
    lcp_sizes=[2_000, 10_000], histogram_n=10_000,
)


class BuildCapExceeded(Exception):
    """Raised by the alarm when one build passes the cap."""


def capped(function: Callable[[], Any], cap_s: float) -> Any:
    """Call ``function``, raising :class:`BuildCapExceeded` if it passes ``cap_s`` seconds."""
    def on_alarm(signum: int, frame: Any) -> None:
        raise BuildCapExceeded()

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, cap_s)
    try:
        return function()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def timed(function: Callable[[], Any]) -> Tuple[Any, float]:
    """Return ``(result, seconds)`` for one call."""
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def fmt(value: Any, digits: int = 8) -> str:
    """Format a cell; ``None`` becomes empty, ints stay exact (distinct substrings can be huge)."""
    if value is None:
        return ""
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def row(experiment: str, method: str, dataset: str, n: int, seed: Any, **fields: Any) -> dict:
    """One CSV row with every column present."""
    entry = {c: None for c in COLUMNS}
    entry.update(experiment=experiment, method=method, dataset=dataset, n=n, seed=seed)
    entry.update(fields)
    return entry


def make_text(dataset: str, n: int, seed: int) -> str:
    """The seeded synthetic text for one dataset."""
    if dataset == "dna":
        return dna_text(n, seed)
    if dataset == "english":
        return english_like_text(n, seed)
    if dataset == "periodic":
        return periodic_text(n, 7)
    raise ValueError(dataset)


# ----------------------------------------------------------------------
# Experiment 1 (and the data for 2, 6 and the heatmap)
# ----------------------------------------------------------------------
def dataset_study(cfg: Config, master: int, rows: List[dict], caps: List[dict],
                  keep: Dict[str, Any]) -> None:
    print("\n=== EXPERIMENT 1 - Dataset sizes ===", flush=True)
    for dataset in ("dna", "english"):
        tree_active, last_tree = True, None
        for n in cfg.sizes:
            seed = derive_seed(master, "dataset_sizes", dataset, n, 0)
            text = make_text(dataset, n, seed)
            count = cfg.queries_largest if n == cfg.sizes[-1] else cfg.queries
            patterns = sample_patterns(text, count, cfg.m, derive_seed(master, "patterns", dataset, n, 0))
            reference = [kmp_search(text, p) for p in patterns]

            def query_rows(method: str, search: Callable[[str], List[int]], preprocess: Optional[float],
                           extra: Optional[dict] = None) -> None:
                times, agree, matches = [], True, 0
                for p, ref in zip(patterns, reference):
                    found, elapsed = timed(lambda p=p: search(p))
                    times.append(elapsed)
                    agree = agree and found == ref
                    matches += len(found)
                rows.append(row("dataset_sizes", method, dataset, n, seed, m=cfg.m, queries=len(patterns),
                                preprocess_s=preprocess, mean_query_s=statistics.fmean(times),
                                std_query_s=statistics.stdev(times) if len(times) > 1 else None,
                                matches=matches, agree=agree, **(extra or {})))
                keep["latency"][(method, dataset, n)] = statistics.fmean(times) * 1e3

            # KMP: comparisons from separate counted runs on the first few patterns.
            counted = [kmp_search(text, p, stats=True)[1] for p in patterns[:cfg.stats_queries]]
            query_rows("kmp", lambda p: kmp_search(text, p), None,
                       {"comparisons": statistics.fmean(s.comparisons for s in counted)})
            rk_counted = [rabin_karp_search(text, p, stats=True)[1] for p in patterns[:cfg.stats_queries]]
            query_rows("rk", lambda p: rabin_karp_search(text, p), None,
                       {"comparisons": statistics.fmean(s.comparisons for s in rk_counted),
                        "spurious_hits": statistics.fmean(s.spurious_hits for s in rk_counted),
                        "base": 256, "modulus": DEFAULT_MODULUS})
            sa, sa_time = timed(lambda: SuffixArray(text))
            query_rows("sa", sa.find, sa_time)
            if n in cfg.lcp_sizes:
                keep["sa_stats"][(dataset, n)] = (sa.stats(), sa_time, seed)
                if n == cfg.histogram_n:
                    keep["histograms"][dataset] = list(sa.lcp)
            del sa

            if tree_active and last_tree is not None:
                predicted = last_tree[1] * (n / last_tree[0]) * math.log(n) / math.log(last_tree[0])
                if predicted > cfg.tree_cap_s:
                    tree_active = False
                    caps.append({"method": "st", "dataset": dataset, "n": n,
                                 "reason": f"predicted build {predicted:.0f} s from the "
                                           f"{last_tree[0]:,}-character build passes the "
                                           f"{cfg.tree_cap_s:.0f} s cap"})
            if tree_active:
                try:
                    tree, tree_time = timed(lambda: capped(lambda: SuffixTree(text), cfg.tree_cap_s))
                    last_tree = (n, tree_time)
                    query_rows("st", tree.find, tree_time)
                    del tree
                except BuildCapExceeded:
                    tree_active = False
                    caps.append({"method": "st", "dataset": dataset, "n": n,
                                 "reason": f"a build passed the {cfg.tree_cap_s:.0f} s cap"})
            if not tree_active:
                keep["capped"].append(("st", dataset, n))
            line = "  ".join(f"{r['method']} {r['mean_query_s'] * 1e3:9.3f} ms" for r in rows
                             if r["experiment"] == "dataset_sizes" and r["dataset"] == dataset and r["n"] == n)
            builds = "  ".join(f"{r['method']} build {r['preprocess_s']:.2f} s" for r in rows
                               if r["experiment"] == "dataset_sizes" and r["dataset"] == dataset
                               and r["n"] == n and r["preprocess_s"] is not None)
            print(f"    {dataset:<8} n={n:<9,} {line}  |  {builds}", flush=True)
            for cap in caps:
                if cap["dataset"] == dataset and cap["n"] == n:
                    print(f"    [cap] st {dataset} n={n:,}: {cap['reason']}", flush=True)


def breakeven_points(cfg: Config, rows: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    """Experiment 2: measured build and query means on the break-even text."""
    out = {}
    for r in rows:
        if r["experiment"] == "dataset_sizes" and r["dataset"] == "dna" and r["n"] == cfg.breakeven_n \
                and r["method"] in ("kmp", "sa", "st"):
            out[r["method"]] = {"preprocess_s": r["preprocess_s"] or 0.0, "mean_query_s": r["mean_query_s"]}
    return out


# ----------------------------------------------------------------------
# Experiment 3: Rabin-Karp collisions
# ----------------------------------------------------------------------
def collision_study(cfg: Config, master: int, rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    print("\n=== EXPERIMENT 3 - Rabin-Karp collisions ===", flush=True)
    n, m = cfg.collision_n, cfg.collision_m
    seed = derive_seed(master, "rk_collisions", n, 0)
    text = random_text(n, string.ascii_lowercase, seed)
    patterns = sample_patterns(text, cfg.collision_patterns, m, derive_seed(master, "rk_patterns", n, 0))
    reference = [kmp_search(text, p) for p in patterns]
    windows = n - m + 1

    def measure(experiment: str, base: int, modulus: int) -> dict:
        spurious, times, agree = [], [], True
        for p, ref in zip(patterns, reference):
            (found, st), elapsed = timed(lambda p=p: rabin_karp_search(text, p, base=base, modulus=modulus,
                                                                     stats=True))
            spurious.append(st.spurious_hits)
            times.append(elapsed)
            agree = agree and found == ref
        entry = row(experiment, "rk", "random_lowercase", n, seed, m=m, queries=len(patterns),
                    mean_query_s=statistics.fmean(times), std_query_s=statistics.stdev(times),
                    spurious_hits=statistics.fmean(spurious), expected_spurious=windows / modulus,
                    base=base, modulus=modulus, agree=agree)
        rows.append(entry)
        return entry

    moduli = []
    for q in (7, 101, 1009, 10007, 100003, 1000003, DEFAULT_MODULUS):
        e = measure("rk_moduli", 256, q)
        moduli.append({"modulus": q, "spurious": e["spurious_hits"], "expected": e["expected_spurious"]})
        print(f"    base 256  q={q:<20,} spurious/pattern {e['spurious_hits']:12,.2f}  "
              f"expected {e['expected_spurious']:.3g}", flush=True)
    random_base = make_rng(derive_seed(master, "rk_random_base", n, 0)).randrange(256, 10007)
    bases = []
    for b, label in ((1, "1"), (31, "31"), (256, "256"), (random_base, f"{random_base}\n(seeded)"),
                     (10007, "10007\n(= q)")):
        e = measure("rk_bases", b, 10007)
        bases.append({"label": label, "spurious": e["spurious_hits"], "expected": e["expected_spurious"]})
        print(f"    q=10007   base={b:<6} spurious/pattern {e['spurious_hits']:12,.2f}", flush=True)
    return moduli, bases


# ----------------------------------------------------------------------
# Experiment 4: worst case
# ----------------------------------------------------------------------
def worst_case_study(cfg: Config, master: int, rows: List[dict]) -> None:
    print("\n=== EXPERIMENT 4 - Worst case ===", flush=True)
    text, all_match = worst_case_rk(cfg.worst_n, cfg.worst_m)
    no_match = "a" * (cfg.worst_m - 1) + "b"
    for dataset, pattern in (("a^n, pattern a^m", all_match), ("a^n, pattern a^(m-1)b", no_match)):
        reference = kmp_search(text, pattern)
        for method in ("naive", "kmp", "rk"):
            if method == "naive":
                (found, st), elapsed = timed(lambda: naive_search(text, pattern, stats=True))
                extra = {"comparisons": st.comparisons}
            elif method == "kmp":
                (found, st), elapsed = timed(lambda: kmp_search(text, pattern, stats=True))
                extra = {"comparisons": st.comparisons + st.lps_comparisons}
            else:
                (found, st), elapsed = timed(lambda: rabin_karp_search(text, pattern, stats=True))
                extra = {"comparisons": st.comparisons, "spurious_hits": st.spurious_hits,
                         "base": 256, "modulus": DEFAULT_MODULUS}
            rows.append(row("worst_case", method, dataset, cfg.worst_n, "", m=cfg.worst_m, queries=1,
                            mean_query_s=elapsed, matches=len(found), agree=found == reference, **extra))
            print(f"    {dataset:<22} {method:<6} {elapsed:8.4f} s  comparisons {extra['comparisons']:>12,}  "
                  f"matches {len(found):,}", flush=True)


# ----------------------------------------------------------------------
# Experiment 5: memory
# ----------------------------------------------------------------------
def memory_study(cfg: Config, master: int, rows: List[dict], caps: List[dict],
                 tree_limit: Dict[str, int]) -> List[dict]:
    print("\n=== EXPERIMENT 5 - Memory ===", flush=True)
    out = []
    for n in cfg.memory_sizes:
        seed = derive_seed(master, "memory", n, 0)
        text = dna_text(n, seed)
        pattern = sample_patterns(text, 1, cfg.m, derive_seed(master, "memory_pattern", n, 0))[0]
        builders = {
            "kmp": lambda: compute_lps(pattern),
            "rk": lambda: (lambda h: (h.hash(pattern), h.power(len(pattern))))(RollingHash()),
            "sa": lambda: SuffixArray(text),
            "st": lambda: SuffixTree(text),
        }
        for method, build in builders.items():
            if method == "st" and n > tree_limit.get("dna", 0):
                continue
            tracemalloc.start()
            try:
                capped(build, cfg.memory_cap_s)
                _, peak = tracemalloc.get_traced_memory()
            except BuildCapExceeded:
                caps.append({"method": method, "dataset": "dna", "n": n,
                             "reason": f"memory run passed the {cfg.memory_cap_s:.0f} s cap under tracemalloc"})
                continue
            finally:
                tracemalloc.stop()
            rows.append(row("memory", method, "dna", n, seed, m=cfg.m, peak_mem_kb=peak / 1024))
            out.append({"method": method, "n": n, "peak_mem_kb": peak / 1024})
        print(f"    n={n:<9,} " + "  ".join(f"{r['method']} {r['peak_mem_kb']:12,.1f} KiB"
                                          for r in out if r["n"] == n), flush=True)
    return out


# ----------------------------------------------------------------------
# Experiment 6: LCP statistics
# ----------------------------------------------------------------------
def lcp_study(cfg: Config, master: int, rows: List[dict], keep: Dict[str, Any]) -> List[dict]:
    print("\n=== EXPERIMENT 6 - LCP statistics ===", flush=True)
    means = []
    for dataset in ("dna", "english", "periodic"):
        for n in cfg.lcp_sizes:
            if (dataset, n) in keep["sa_stats"]:
                stats, build, seed = keep["sa_stats"][(dataset, n)]
            else:
                seed = derive_seed(master, "lcp_stats", dataset, n, 0)
                sa, build = timed(lambda: SuffixArray(make_text(dataset, n, seed)))
                stats = sa.stats()
                if n == cfg.histogram_n:
                    keep["histograms"][dataset] = list(sa.lcp)
                del sa
            rows.append(row("lcp_stats", "sa", dataset, n, seed, preprocess_s=build,
                            mean_lcp=float(stats["mean_lcp"]), max_lcp=int(stats["max_lcp"]),
                            distinct_substrings=int(stats["distinct_substrings"]),
                            lrs_length=int(stats["lrs_length"])))
            means.append({"dataset": dataset, "n": n, "mean_lcp": float(stats["mean_lcp"])})
            print(f"    {dataset:<8} n={n:<9,} mean LCP {stats['mean_lcp']:12,.2f}  max {stats['max_lcp']:>10,}  "
                  f"distinct {stats['distinct_substrings']:>16,}  LRS {stats['lrs_length']:,}", flush=True)
    return means


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """Write ``rows`` with exactly ``columns``; ``None`` becomes an empty cell."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: fmt(r.get(c)) for c in columns})


def part3_figures(out: str) -> None:
    """The suffix array of mississippi and the suffix tree of banana (spec section 5)."""
    plot_suffix_array("mississippi", os.path.join(out, "suffix_array_mississippi.png"))
    tree = SuffixTree("banana")
    draw_suffix_tree(tree, os.path.join(out, "suffix_tree_banana.png"))
    write_dot(tree, os.path.join(out, "suffix_tree_banana.dot"))


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
    print("CSC 5300 Week 11 - string search benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    : {platform.platform()}")
    print(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  master seed : {master}  (texts: derive_seed(master, experiment, ..., n, rep))")
    print(f"  tree cap    : {cfg.tree_cap_s:.0f} s per build")
    print(f"  output      : {out}", flush=True)

    started = time.perf_counter()
    rows: List[dict] = []
    caps: List[dict] = []
    keep: Dict[str, Any] = {"latency": {}, "capped": [], "sa_stats": {}, "histograms": {}}
    dataset_study(cfg, master, rows, caps, keep)
    tree_limit = {d: max([r["n"] for r in rows if r["experiment"] == "dataset_sizes"
                          and r["method"] == "st" and r["dataset"] == d] or [0])
                  for d in ("dna", "english")}
    breakeven = breakeven_points(cfg, rows)
    moduli, bases = collision_study(cfg, master, rows)
    worst_case_study(cfg, master, rows)
    memory = memory_study(cfg, master, rows, caps, tree_limit)
    means = lcp_study(cfg, master, rows, keep)

    print("\n=== writing results ===", flush=True)
    write_csv(os.path.join(out, "search_benchmark.csv"), rows, COLUMNS)
    write_csv(os.path.join(out, "size_caps.csv"), caps, CAP_COLUMNS)
    mismatches = sum(1 for r in rows if r["agree"] is False)
    print(f"  {len(rows)} rows; {mismatches} disagreements with KMP; {len(caps)} caps", flush=True)

    print("\n=== figures ===", flush=True)
    part3_figures(out)
    points = [{"method": k[0], "dataset": k[1], "n": k[2], "mean_query_s": v / 1e3}
              for k, v in keep["latency"].items()]
    for name, builder in (
        ("runtime_comparison.png",
         lambda p: plot_runtime_comparison(points, breakeven, cfg.breakeven_max_q, save_path=p)),
        ("collision_analysis.png", lambda p: plot_collision_analysis(moduli, bases, save_path=p)),
        ("sa_lcp_stats.png", lambda p: plot_sa_lcp_stats(keep["histograms"], means, save_path=p)),
        ("heatmap_dataset_sizes.png",
         lambda p: plot_heatmap_dataset_sizes(keep["latency"], keep["capped"], save_path=p)),
        ("memory_use.png", lambda p: plot_index_memory(memory, save_path=p)),
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
