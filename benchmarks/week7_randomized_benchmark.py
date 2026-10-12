#!/usr/bin/env python3
"""Week 7 benchmark: randomized algorithms against their deterministic counterparts.

Run from the repository root with the project virtualenv active::

    python benchmarks/week7_randomized_benchmark.py                 # the full study
    python benchmarks/week7_randomized_benchmark.py --smoke         # seconds, tiny sizes
    python benchmarks/week7_randomized_benchmark.py --seed 7 --out /tmp/w7

Five studies, one per algorithm, each measuring how much extra work buys how
much extra confidence:

1. **QuickSort** - randomized against two deterministic pivot rules and Week
   2's ``quick_sort``, on random and on sorted input, plus the distribution of
   comparison counts against ``2(n+1)H_n - 4n``.
2. **Selection** - randomized QuickSelect against median of medians, for the
   minimum, median and maximum.
3. **Primality** - Fermat and Miller-Rabin false-positive rates on Carmichael
   and other composites at k = 1, 5, 10, 20; speed against trial division up
   to 10^14; and the fixed-base attack on 3,215,031,751.
4. **Bloom filter** - false-positive rate against load factor and against bits
   per item, measured on 100,000 items never inserted.
5. **MinHash** - estimation error against signature length k = 16 to 512, on
   simulated sets with exact Jaccard values and on an original text.

**Reproducibility.** Every random draw comes from a ``random.Random`` seeded
by ``derive_seed(master_seed, <experiment>, ...)``, and every item hash is
blake2b with a seed-derived salt rather than the per-process-salted
``hash()``. Run counts are fixed rather than chosen from a pilot timing, so
two runs with the same ``--seed`` produce identical non-timing columns.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from math import isqrt
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.randomized.bloom_filter import BloomFilter  # noqa: E402
from src.randomized.minhash_similarity import MinHash, jaccard, shingles  # noqa: E402
from src.randomized.montecarlo_primality import (  # noqa: E402
    count_liars,
    fermat_test,
    is_prime_deterministic,
    miller_rabin,
    miller_rabin_fixed_bases,
)
from src.randomized.randomized_quicksort import (  # noqa: E402
    comparison_std_asymptotic,
    deterministic_quicksort,
    expected_comparisons,
    randomized_quicksort,
)
from src.randomized.randomized_selection import (  # noqa: E402
    deterministic_select,
    randomized_select,
)
from src.sorting.quick_sort import quick_sort  # noqa: E402
from src.utils.probability_analysis import (  # noqa: E402
    bloom_fp_theoretical,
    minhash_standard_error,
    summarize,
    tail_fraction,
    wilson_interval,
)
from src.utils.random_utils import (  # noqa: E402
    DEFAULT_SEED,
    derive_seed,
    make_rng,
    random_distinct_list,
)
from src.utils.timer import time_call  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_bloom_fp_rate,
    plot_minhash_similarity,
    plot_primality_accuracy,
    plot_quicksort_comparison,
    plot_selection_runtime,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

#: The CSV's columns are fixed, because the report reads them back by name.
COLUMNS = [
    "algorithm", "variant", "category", "experiment", "input_kind", "n", "param_name",
    "param_value", "runs", "mean_time_s", "var_time_s2", "std_time_s", "min_time_s",
    "max_time_s", "metric_name", "metric_value", "metric_ci_low", "metric_ci_high",
    "theoretical_value", "seed", "measurement",
]

#: Deterministic baselines are filed under "las_vegas": the category means an
#: always-correct answer, and a deterministic algorithm is the degenerate case
#: whose running time does not vary either. "monte_carlo" means the answer
#: itself may be wrong with a bounded probability.

#: Columns that hold wall-clock time. Everything else must be identical
#: between two runs with the same master seed.
TIMING_COLUMNS = ["mean_time_s", "var_time_s2", "std_time_s", "min_time_s", "max_time_s"]

#: Composites for the error-rate study: seven Carmichael numbers, the strong
#: pseudoprime 2047, and 1,000,001 = 101 x 9,901.
COMPOSITES = [561, 1105, 1729, 2465, 2821, 6601, 8911, 2047, 1_000_001]

#: The smallest prime at or above 10^d, d = 6 .. 14.
SPEED_PRIMES = [1_000_003, 10_000_019, 100_000_007, 1_000_000_007, 10_000_000_019,
                100_000_000_003, 1_000_000_000_039, 10_000_000_000_037,
                100_000_000_000_031]
MERSENNE_61 = 2 ** 61 - 1

#: 151 x 751 x 28,351, a strong pseudoprime to every base in {2, 3, 5, 7}.
FIXED_BASE_COMPOSITE = 3_215_031_751

#: An original paragraph written for this project, used as the text dataset.
ORIGINAL_TEXT = (
    "A randomized algorithm makes some of its choices by drawing random numbers, and "
    "the draws change from one run to the next. Some of these algorithms always return "
    "the correct answer and only their running time varies, so a careful analysis "
    "describes the expected cost and how widely it spreads. Others finish in a fixed "
    "amount of time but may return a wrong answer with a small probability, which the "
    "analysis bounds and which repetition can drive lower. In both cases the input "
    "cannot be chosen to defeat the algorithm, because the adversary who writes the "
    "input does not know the random numbers in advance. That single property is why "
    "randomness appears in sorting, in primality testing for cryptography, and in the "
    "compact summaries that databases use to answer questions about very large sets."
)


@dataclass
class Config:
    """Every size, run count and trial count the study uses, in one place."""

    qs_random_sizes: List[int]
    qs_runs: int
    qs_sorted_sizes: List[int]
    qs_sorted_projected: List[int]
    qs_count_samples: Dict[int, int]
    sel_sizes: List[int]
    sel_runs: int
    sel_det_runs: int
    prim_ks: List[int]
    prim_trials: int
    prim_composites: List[int]
    speed_primes: List[int]
    adversarial_trials: int
    adversarial_large_trials: int
    bloom_m: int
    bloom_k: int
    bloom_ns: List[int]
    bloom_fixed_n: int
    bloom_bits: List[int]
    bloom_queries: int
    minhash_ks: List[int]
    minhash_seeds: int


FULL = Config(
    qs_random_sizes=[100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
    qs_runs=30,
    qs_sorted_sizes=[100, 500, 1_000, 2_000, 5_000, 10_000],
    qs_sorted_projected=[20_000, 50_000, 100_000],
    qs_count_samples={1_000: 1_000, 10_000: 200},
    sel_sizes=[1_000, 5_000, 10_000, 50_000, 100_000],
    sel_runs=30,
    sel_det_runs=5,
    prim_ks=[1, 5, 10, 20],
    prim_trials=10_000,
    prim_composites=COMPOSITES,
    speed_primes=SPEED_PRIMES,
    adversarial_trials=10_000,
    adversarial_large_trials=200_000,
    bloom_m=10_000,
    bloom_k=7,
    bloom_ns=[100, 250, 500, 1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000],
    bloom_fixed_n=10_000,
    bloom_bits=list(range(2, 21)),
    bloom_queries=100_000,
    minhash_ks=[16, 32, 64, 128, 256, 512],
    minhash_seeds=50,
)

SMOKE = Config(
    qs_random_sizes=[100, 500],
    qs_runs=4,
    qs_sorted_sizes=[100, 300],
    qs_sorted_projected=[1_000],
    qs_count_samples={200: 30},
    sel_sizes=[500, 1_000],
    sel_runs=4,
    sel_det_runs=2,
    prim_ks=[1, 5],
    prim_trials=200,
    prim_composites=[561, 2047],
    speed_primes=SPEED_PRIMES[:2],
    adversarial_trials=200,
    adversarial_large_trials=1_000,
    bloom_m=2_000,
    bloom_k=5,
    bloom_ns=[100, 400],
    bloom_fixed_n=500,
    bloom_bits=[4, 8],
    bloom_queries=5_000,
    minhash_ks=[16, 64],
    minhash_seeds=5,
)


# ----------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------
def time_once(function: Callable[[], Any]) -> float:
    """One timed call with garbage collection paused, as the Week 1 harness does."""
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        started = time.perf_counter()
        function()
        return time.perf_counter() - started
    finally:
        if was_enabled:
            gc.enable()


def timing_columns(samples: List[float]) -> Dict[str, Any]:
    stats = summarize(samples)
    return {
        "runs": stats["n"],
        "mean_time_s": f"{stats['mean']:.9f}",
        "var_time_s2": f"{stats['variance']:.12e}",
        "std_time_s": f"{stats['std']:.9f}",
        "min_time_s": f"{stats['min']:.9f}",
        "max_time_s": f"{stats['max']:.9f}",
    }


def row(algorithm: str, variant: str, category: str, experiment: str, input_kind: str,
        n: Any, seed: int, **fields: Any) -> Dict[str, Any]:
    record = {name: "" for name in COLUMNS}
    record.update({"algorithm": algorithm, "variant": variant, "category": category,
                   "experiment": experiment, "input_kind": input_kind, "n": n,
                   "seed": seed, "measurement": "measured"})
    for name, value in fields.items():
        if name not in record:
            raise KeyError(f"unknown column {name}")
        record[name] = value
    return record


def fmt(value: Optional[float], digits: int = 9) -> Any:
    return "" if value is None else f"{value:.{digits}g}"


# ----------------------------------------------------------------------
# Study 1: QuickSort
# ----------------------------------------------------------------------
QS_VARIANTS = ["randomized", "deterministic_last", "deterministic_median_of_three",
               "week2_quick_sort"]


def quicksort_study(config: Config, master: int, rows: List[dict]) -> Dict[str, Any]:
    print("\n=== STUDY 1 - QuickSort ===", flush=True)
    random_series: List[dict] = []
    for n in config.qs_random_sizes:
        base_seed = derive_seed(master, "qs_random_input", n)
        fixed = random_distinct_list(n, base_seed)
        expected = sorted(fixed)
        for variant in QS_VARIANTS:
            samples: List[float] = []
            for run in range(config.qs_runs + 2):            # two discarded warm-ups
                pivot_seed = derive_seed(master, "qs_pivot", n, run)
                if variant == "randomized":
                    data, call = fixed, (lambda d=fixed, s=pivot_seed: randomized_quicksort(d, seed=s))
                elif variant == "week2_quick_sort":
                    data, call = fixed, (lambda d=fixed, s=pivot_seed: quick_sort(d, seed=s))
                else:
                    data = random_distinct_list(n, derive_seed(master, "qs_shuffle", n, run))
                    rule = "last" if variant == "deterministic_last" else "median_of_three"
                    call = (lambda d=data, r=rule: deterministic_quicksort(d, pivot=r))
                if run == 0:
                    assert call() == sorted(data), f"{variant} failed to sort at n={n}"
                elapsed = time_once(call)
                if run >= 2:
                    samples.append(elapsed)
            stats = summarize(samples)
            random_series.append({"variant": variant, "n": n, "mean_s": stats["mean"],
                                  "std_s": stats["std"], "cv": stats["cv"]})
            rows.append(row("quicksort", variant, "las_vegas",
                            "random_input_runtime", "random_distinct", n, base_seed,
                            **timing_columns(samples)))
            print(f"    random  {variant:<30} n={n:<7} {stats['mean'] * 1e3:9.3f} ms "
                  f"+- {stats['std'] * 1e3:7.3f}  (cv {stats['cv']:.3f})", flush=True)
        del fixed, expected

    sorted_series: List[dict] = []
    per_comparison: Optional[float] = None
    for n in config.qs_sorted_sizes:
        data = list(range(n))
        runs_last = 3 if n >= 5_000 else 5
        for variant in ("randomized", "deterministic_last"):
            samples = []
            count = None
            runs = config.qs_runs if variant == "randomized" else runs_last
            for run in range(runs + 1):                       # one discarded warm-up
                pivot_seed = derive_seed(master, "qs_sorted_pivot", n, run)
                if variant == "randomized":
                    call = (lambda s=pivot_seed: randomized_quicksort(data, seed=s))
                else:
                    call = (lambda: deterministic_quicksort(data, pivot="last"))
                elapsed = time_once(call)
                if run >= 1:
                    samples.append(elapsed)
            if variant == "randomized":
                _, count = randomized_quicksort(
                    data, seed=derive_seed(master, "qs_sorted_pivot", n, 0),
                    count_comparisons=True)
            else:
                _, count = deterministic_quicksort(data, pivot="last", count_comparisons=True)
                assert count == n * (n - 1) // 2, f"sorted input count wrong at n={n}"
                per_comparison = summarize(samples)["mean"] / count
            stats = summarize(samples)
            sorted_series.append({"variant": variant, "n": n, "mean_s": stats["mean"],
                                  "std_s": stats["std"], "measurement": "measured"})
            theory = n * (n - 1) / 2 if variant == "deterministic_last" else expected_comparisons(n)
            rows.append(row("quicksort", variant, "las_vegas", "sorted_input", "sorted", n,
                            derive_seed(master, "qs_sorted_pivot", n, 0),
                            metric_name="comparisons", metric_value=count,
                            theoretical_value=fmt(theory, 12), **timing_columns(samples)))
            print(f"    sorted  {variant:<30} n={n:<7} {stats['mean'] * 1e3:9.3f} ms  "
                  f"comparisons={count:,}", flush=True)
    for n in config.qs_sorted_projected:
        count = n * (n - 1) // 2
        projected = per_comparison * count
        sorted_series.append({"variant": "deterministic_last", "n": n, "mean_s": projected,
                              "std_s": None, "measurement": "projected"})
        rows.append(row("quicksort", "deterministic_last", "las_vegas", "sorted_input",
                        "sorted", n, master, runs=0, mean_time_s=f"{projected:.9f}",
                        metric_name="comparisons", metric_value=count,
                        theoretical_value=count, measurement="projected"))
        print(f"    sorted  {'deterministic_last':<30} n={n:<7} PROJECTED {projected:9.2f} s "
              f"({count:,} comparisons at {per_comparison * 1e9:.1f} ns)", flush=True)
        # Randomized QuickSort is cheap here, so it is measured, not projected.
        data = list(range(n))
        samples = [time_once(lambda s=derive_seed(master, "qs_sorted_pivot", n, r):
                             randomized_quicksort(data, seed=s)) for r in range(5)]
        stats = summarize(samples)
        sorted_series.append({"variant": "randomized", "n": n, "mean_s": stats["mean"],
                              "std_s": stats["std"], "measurement": "measured"})
        _, rcount = randomized_quicksort(data, seed=derive_seed(master, "qs_sorted_pivot", n, 0),
                                         count_comparisons=True)
        rows.append(row("quicksort", "randomized", "las_vegas", "sorted_input", "sorted", n,
                        derive_seed(master, "qs_sorted_pivot", n, 0),
                        metric_name="comparisons", metric_value=rcount,
                        theoretical_value=fmt(expected_comparisons(n), 12),
                        **timing_columns(samples)))

    histogram: Dict[str, Any] = {}
    for n, samples_wanted in config.qs_count_samples.items():
        data = random_distinct_list(n, derive_seed(master, "qs_count_input", n))
        counts = [randomized_quicksort(data, seed=derive_seed(master, "qs_count_pivot", n, s),
                                       count_comparisons=True)[1]
                  for s in range(samples_wanted)]
        stats = summarize(counts)
        expected = expected_comparisons(n)
        sigma = comparison_std_asymptotic(n)
        seed = derive_seed(master, "qs_count_pivot", n, 0)
        for name, value, theory in (("comparisons_mean", stats["mean"], expected),
                                    ("comparisons_std", stats["std"], sigma)):
            rows.append(row("quicksort", "randomized", "las_vegas", "comparison_distribution",
                            "random_distinct", n, seed, runs=samples_wanted,
                            metric_name=name, metric_value=fmt(value, 10),
                            theoretical_value=fmt(theory, 10)))
        print(f"    counts  n={n:<7} {samples_wanted} seeds: mean {stats['mean']:,.1f} "
              f"(expected {expected:,.1f}), std {stats['std']:,.1f} "
              f"(asymptotic {sigma:,.1f}), min {stats['min']:,} max {stats['max']:,}",
              flush=True)
        if not histogram:
            histogram = {"n": n, "counts": counts, "expected": expected,
                         "asymptotic_std": sigma}
    return {"random_series": random_series, "sorted_series": sorted_series,
            "comparison_sample": histogram}


# ----------------------------------------------------------------------
# Study 2: selection
# ----------------------------------------------------------------------
def selection_study(config: Config, master: int, rows: List[dict]) -> Dict[str, Any]:
    print("\n=== STUDY 2 - selection ===", flush=True)
    series: List[dict] = []
    depth_series: List[dict] = []
    ratio_sample: Dict[str, Any] = {}
    for n in config.sel_sizes:
        input_seed = derive_seed(master, "sel_input", n)
        data = random_distinct_list(n, input_seed)
        ordered = sorted(data)
        for label, k in (("min", 1), ("median", (n + 1) // 2), ("max", n)):
            truth = ordered[k - 1]
            samples, depths, ratios, correct = [], [], [], 0
            for run in range(config.sel_runs + 2):
                seed = derive_seed(master, "sel_pivot", n, label, run)
                result: List[Any] = []
                elapsed = time_once(lambda s=seed: result.append(
                    randomized_select(data, k, seed=s, return_stats=True)))
                value, stats = result[0]
                if run >= 2:
                    samples.append(elapsed)
                    depths.append(stats.depth)
                    ratios.append(stats.comparisons / n)
                    correct += value == truth
            runs = config.sel_runs
            within = 1.0 - tail_fraction(ratios, 4.0)
            depth_stats, ratio_stats = summarize(depths), summarize(ratios)
            time_stats = summarize(samples)
            series.append({"algorithm": "randomized", "k_label": label, "n": n,
                           "mean_s": time_stats["mean"], "std_s": time_stats["std"]})
            depth_series.append({"k_label": label, "n": n, "depth_mean": depth_stats["mean"],
                                 "depth_max": depth_stats["max"]})
            low, high = wilson_interval(correct, runs)
            expected_ratio = 2 * (1 + math.log(2)) if label == "median" else 2.0
            seed0 = derive_seed(master, "sel_pivot", n, label, 0)
            timing = timing_columns(samples)
            for name, value, ci, theory in (
                    ("success_rate", correct / runs, (low, high), 1.0),
                    ("within_4n_fraction", within, wilson_interval(round(within * runs), runs), None),
                    ("comparisons_per_n_mean", ratio_stats["mean"], None, expected_ratio),
                    ("comparisons_per_n_std", ratio_stats["std"], None, None),
                    ("depth_mean", depth_stats["mean"], None, math.log2(n)),
                    ("depth_max", depth_stats["max"], None, None)):
                rows.append(row("quickselect", label, "las_vegas", "selection", "random_distinct",
                                n, seed0, param_name="k", param_value=k, metric_name=name,
                                metric_value=fmt(value, 10),
                                metric_ci_low=fmt(ci[0], 10) if ci else "",
                                metric_ci_high=fmt(ci[1], 10) if ci else "",
                                theoretical_value=fmt(theory, 10), **timing))
            if n == max(config.sel_sizes) and label == "median":
                ratio_sample = {"n": n, "k_label": label, "ratios": ratios}

            det_samples = []
            det_result: List[Any] = []
            for run in range(config.sel_det_runs + 1):
                det_result.clear()
                elapsed = time_once(lambda: det_result.append(
                    deterministic_select(data, k, return_stats=True)))
                if run >= 1:
                    det_samples.append(elapsed)
            det_value, det_stats = det_result[0]
            assert det_value == truth, f"median of medians wrong at n={n}, k={k}"
            det_time = summarize(det_samples)
            series.append({"algorithm": "deterministic", "k_label": label, "n": n,
                           "mean_s": det_time["mean"], "std_s": det_time["std"]})
            rows.append(row("median_of_medians", label, "las_vegas", "selection",
                            "random_distinct", n, input_seed, param_name="k", param_value=k,
                            metric_name="comparisons_per_n_mean",
                            metric_value=fmt(det_stats.comparisons / n, 10),
                            **timing_columns(det_samples)))
            print(f"    n={n:<7} {label:<6} quickselect {time_stats['mean'] * 1e3:8.3f} ms "
                  f"+- {time_stats['std'] * 1e3:6.3f}  depth {depth_stats['mean']:5.1f} "
                  f"(max {depth_stats['max']})  comp/n {ratio_stats['mean']:.2f}  "
                  f"within 4n {within:.2f}  correct {correct}/{runs} | median-of-medians "
                  f"{det_time['mean'] * 1e3:8.3f} ms", flush=True)
    return {"series": series, "depth_series": depth_series, "ratio_sample": ratio_sample}


# ----------------------------------------------------------------------
# Study 3: primality
# ----------------------------------------------------------------------
def primality_study(config: Config, master: int, rows: List[dict]) -> Dict[str, Any]:
    print("\n=== STUDY 3 - primality ===", flush=True)
    accuracy_rows: List[dict] = []
    for n in config.prim_composites:
        liars = {kind: count_liars(n, kind) for kind in ("fermat", "strong")}
        for test, function, kind in (("fermat", fermat_test, "fermat"),
                                     ("miller_rabin", miller_rabin, "strong")):
            fraction = liars[kind] / (n - 3)
            for k in config.prim_ks:
                fooled = sum(
                    function(n, k=k, seed=derive_seed(master, "prim_accuracy", test, n, k, t))
                    for t in range(config.prim_trials))
                rate = fooled / config.prim_trials
                low, high = wilson_interval(fooled, config.prim_trials)
                predicted = fraction ** k
                accuracy_rows.append({"test": test, "n": n, "k": k, "rate": rate,
                                      "ci_low": low, "ci_high": high, "predicted": predicted,
                                      "trials": config.prim_trials})
                rows.append(row("fermat_test" if test == "fermat" else "miller_rabin", test,
                                "monte_carlo", "false_positive_rate", "composite", n,
                                derive_seed(master, "prim_accuracy", test, n, k, 0),
                                param_name="k", param_value=k, runs=config.prim_trials,
                                metric_name="false_positive_rate", metric_value=fmt(rate, 10),
                                metric_ci_low=fmt(low, 10), metric_ci_high=fmt(high, 10),
                                theoretical_value=fmt(predicted, 10)))
                if test == "miller_rabin":
                    rows.append(row("miller_rabin", test, "monte_carlo", "error_bound",
                                    "composite", n, master, param_name="k", param_value=k,
                                    metric_name="error_bound_4_pow_minus_k",
                                    metric_value=fmt(4.0 ** -k, 10)))
            print(f"    n={n:<9} {test:<13} liars {liars[kind]:>6}/{n - 3:<8} "
                  + "  ".join(f"k={r['k']}: {r['rate']:.5f}" for r in accuracy_rows
                              if r["test"] == test and r["n"] == n), flush=True)

    speed_rows: List[dict] = []
    per_division: Optional[float] = None
    for p in config.speed_primes + [MERSENNE_61]:
        for method in ("trial_division", "fermat", "miller_rabin"):
            if method == "trial_division" and p == MERSENNE_61:
                divisions = isqrt(p) // 2
                projected = per_division * divisions
                speed_rows.append({"method": method, "n": p, "mean_s": projected,
                                   "std_s": None, "measurement": "projected"})
                rows.append(row("is_prime_deterministic", method, "las_vegas", "speed",
                                "prime", p, master, runs=0, mean_time_s=f"{projected:.9f}",
                                metric_name="divisions", metric_value=divisions,
                                measurement="projected"))
                print(f"    p=2^61-1           trial_division  PROJECTED {projected:,.0f} s "
                      f"({divisions:,} divisions)", flush=True)
                continue
            seed = derive_seed(master, "prim_speed", method, p)
            if method == "trial_division":
                repeat = 5 if p <= 10 ** 12 else 3
                assert is_prime_deterministic(p)
                stats = time_call(is_prime_deterministic, p, repeat=repeat, warmup=1)
                per_division = stats["mean"] / (isqrt(p) // 2)
            elif method == "fermat":
                assert fermat_test(p, k=20, seed=seed)
                stats = time_call(fermat_test, p, k=20, seed=seed, repeat=30, warmup=2)
            else:
                assert miller_rabin(p, k=20, seed=seed)
                stats = time_call(miller_rabin, p, k=20, seed=seed, repeat=30, warmup=2)
            speed_rows.append({"method": method, "n": p, "mean_s": stats["mean"],
                               "std_s": stats["std"], "measurement": "measured"})
            rows.append(row({"trial_division": "is_prime_deterministic", "fermat": "fermat_test",
                             "miller_rabin": "miller_rabin"}[method], method,
                            "las_vegas" if method == "trial_division" else "monte_carlo",
                            "speed", "prime", p, seed,
                            param_name="" if method == "trial_division" else "k",
                            param_value="" if method == "trial_division" else 20,
                            runs=stats["runs"], mean_time_s=f"{stats['mean']:.9f}",
                            var_time_s2=f"{stats['std'] ** 2:.12e}",
                            std_time_s=f"{stats['std']:.9f}", min_time_s=f"{stats['min']:.9f}",
                            max_time_s=f"{stats['max']:.9f}"))
        print(f"    p={p:<20,} " + "  ".join(
            f"{r['method']} {r['mean_s'] * 1e6:12,.1f} us" for r in speed_rows if r["n"] == p),
            flush=True)

    fixed = miller_rabin_fixed_bases(FIXED_BASE_COMPOSITE, [2, 3, 5, 7])
    rows.append(row("miller_rabin_fixed_bases", "bases_2_3_5_7", "monte_carlo", "adversarial",
                    "strong_pseudoprime", FIXED_BASE_COMPOSITE, master, param_name="bases",
                    param_value="2 3 5 7", metric_name="answered_prime",
                    metric_value=int(fixed), theoretical_value=0))
    print(f"    fixed bases [2,3,5,7] on {FIXED_BASE_COMPOSITE:,}: answered prime = {fixed}",
          flush=True)
    for k in (1, 5):
        fooled = sum(miller_rabin(FIXED_BASE_COMPOSITE, k=k,
                                  seed=derive_seed(master, "prim_adversarial", k, t))
                     for t in range(config.adversarial_trials))
        low, high = wilson_interval(fooled, config.adversarial_trials)
        rows.append(row("miller_rabin", "random_bases", "monte_carlo", "adversarial",
                        "strong_pseudoprime", FIXED_BASE_COMPOSITE,
                        derive_seed(master, "prim_adversarial", k, 0), param_name="k",
                        param_value=k, runs=config.adversarial_trials,
                        metric_name="false_positive_rate",
                        metric_value=fmt(fooled / config.adversarial_trials, 10),
                        metric_ci_low=fmt(low, 10), metric_ci_high=fmt(high, 10),
                        theoretical_value=fmt(4.0 ** -k, 10)))
        print(f"    random bases k={k}: fooled {fooled} of {config.adversarial_trials} "
              f"(95% upper {high:.2e})", flush=True)
    # A second, larger sample with its own seed family. 3,215,031,751 has
    # exactly phi(n)/4 strong liars, so its per-round rate sits on the 1/4
    # bound (0.248) and 10,000 trials cannot separate k = 5 from the bound.
    # Both runs are kept; neither replaces the other.
    for k in (1, 5):
        trials = config.adversarial_large_trials
        fooled = sum(miller_rabin(FIXED_BASE_COMPOSITE, k=k,
                                  seed=derive_seed(master, "prim_adversarial_large", k, t))
                     for t in range(trials))
        low, high = wilson_interval(fooled, trials)
        rows.append(row("miller_rabin", "random_bases_large_sample", "monte_carlo", "adversarial",
                        "strong_pseudoprime", FIXED_BASE_COMPOSITE,
                        derive_seed(master, "prim_adversarial_large", k, 0), param_name="k",
                        param_value=k, runs=trials, metric_name="false_positive_rate",
                        metric_value=fmt(fooled / trials, 10), metric_ci_low=fmt(low, 10),
                        metric_ci_high=fmt(high, 10), theoretical_value=fmt(4.0 ** -k, 10)))
        print(f"    random bases k={k}, larger sample: fooled {fooled} of {trials:,} "
              f"(rate {fooled / trials:.6f}, 95% interval {low:.6f}-{high:.6f})", flush=True)
    return {"accuracy_rows": accuracy_rows, "speed_rows": speed_rows}


# ----------------------------------------------------------------------
# Study 4: Bloom filter
# ----------------------------------------------------------------------
def bloom_fp(bloom: BloomFilter, queries: int) -> int:
    check = bloom.check
    return sum(1 for q in range(queries) if check(f"query-{q}"))


def bloom_study(config: Config, master: int, rows: List[dict]) -> Dict[str, Any]:
    print("\n=== STUDY 4 - Bloom filter ===", flush=True)
    load_rows: List[dict] = []
    bits_rows: List[dict] = []
    for n in config.bloom_ns:
        seed = derive_seed(master, "bloom_load", n) % (2 ** 64)
        bloom = BloomFilter(config.bloom_m, config.bloom_k, seed=seed)
        for i in range(n):
            bloom.add(f"member-{i}")
        assert all(bloom.check(f"member-{i}") for i in range(n)), "false negative"
        fp = bloom_fp(bloom, config.bloom_queries)
        rate = fp / config.bloom_queries
        low, high = wilson_interval(fp, config.bloom_queries)
        theory = bloom.theoretical_fp_rate()
        load_rows.append({"load_factor": bloom.load_factor(), "measured": rate, "ci_low": low,
                          "ci_high": high, "theoretical": theory})
        rows.append(row("bloom_filter", f"m={config.bloom_m},k={config.bloom_k}", "monte_carlo",
                        "load_factor_sweep", "strings", n, seed, param_name="load_factor",
                        param_value=fmt(bloom.load_factor(), 6), runs=config.bloom_queries,
                        metric_name="false_positive_rate", metric_value=fmt(rate, 10),
                        metric_ci_low=fmt(low, 10), metric_ci_high=fmt(high, 10),
                        theoretical_value=fmt(theory, 10)))
        rows.append(row("bloom_filter", f"m={config.bloom_m},k={config.bloom_k}", "monte_carlo",
                        "load_factor_sweep", "strings", n, seed, param_name="load_factor",
                        param_value=fmt(bloom.load_factor(), 6), metric_name="fill_ratio",
                        metric_value=fmt(bloom.fill_ratio, 10)))
        print(f"    load {bloom.load_factor():.3f}  n={n:<6} fp {rate:.5f} "
              f"[{low:.5f}, {high:.5f}]  theory {theory:.5f}  fill {bloom.fill_ratio:.3f}",
              flush=True)
    n = config.bloom_fixed_n
    for bits in config.bloom_bits:
        k = max(1, round(bits * math.log(2)))
        seed = derive_seed(master, "bloom_bits", bits) % (2 ** 64)
        bloom = BloomFilter(bits * n, k, seed=seed)
        for i in range(n):
            bloom.add(f"member-{i}")
        fp = bloom_fp(bloom, config.bloom_queries)
        rate = fp / config.bloom_queries
        low, high = wilson_interval(fp, config.bloom_queries)
        theory = bloom.theoretical_fp_rate()
        bits_rows.append({"bits_per_item": bits, "k": k, "measured": rate, "ci_low": low,
                          "ci_high": high, "theoretical": theory})
        rows.append(row("bloom_filter", f"k={k}", "monte_carlo", "bits_per_item_sweep",
                        "strings", n, seed, param_name="bits_per_item", param_value=bits,
                        runs=config.bloom_queries, metric_name="false_positive_rate",
                        metric_value=fmt(rate, 10), metric_ci_low=fmt(low, 10),
                        metric_ci_high=fmt(high, 10), theoretical_value=fmt(theory, 10)))
        rows.append(row("bloom_filter", f"k={k}", "monte_carlo", "bits_per_item_sweep",
                        "strings", n, seed, param_name="bits_per_item", param_value=bits,
                        metric_name="memory_bytes", metric_value=bloom.memory_bytes()))
        print(f"    bits/item {bits:>2}  k={k:<2} fp {rate:.6f} [{low:.6f}, {high:.6f}]  "
              f"theory {theory:.6f}  {bloom.memory_bytes():,} bytes", flush=True)
    members = [f"member-{i}" for i in range(n)]
    as_set = set(members)
    set_bytes = sys.getsizeof(as_set) + sum(sys.getsizeof(m) for m in members)
    sized = BloomFilter.from_capacity(n, 0.01, seed=derive_seed(master, "bloom_memory") % (2 ** 64))
    for m in members:
        sized.add(m)
    fp = bloom_fp(sized, config.bloom_queries)
    memory = {"n": n, "bloom_bytes": sized.memory_bytes(), "set_bytes": set_bytes,
              "fp_rate": fp / config.bloom_queries}
    rows.append(row("bloom_filter", "from_capacity_p=0.01", "monte_carlo", "memory", "strings", n,
                    master, metric_name="memory_bytes", metric_value=sized.memory_bytes(),
                    theoretical_value=set_bytes, param_name="python_set_bytes",
                    param_value=set_bytes))
    print(f"    memory at n={n:,}: Bloom {sized.memory_bytes():,} bytes "
          f"(fp {memory['fp_rate']:.4f}) against a Python set of {set_bytes:,} bytes",
          flush=True)
    return {"load_rows": load_rows, "bits_rows": bits_rows, "memory": memory}


# ----------------------------------------------------------------------
# Study 5: MinHash
# ----------------------------------------------------------------------
def simulated_pairs() -> List[tuple]:
    """Pairs of integer sets over a 100-element union with exact Jaccard values."""
    pairs = []
    for intersection, label in ((10, 0.10), (25, 0.25), (50, 0.50), (75, 0.75), (90, 0.90)):
        rest = 100 - intersection
        only_a, only_b = rest // 2, rest - rest // 2
        shared = set(range(intersection))
        a = shared | set(range(intersection, intersection + only_a))
        b = shared | set(range(intersection + only_a, 100))
        assert abs(jaccard(a, b) - label) < 1e-12
        pairs.append(("simulated", f"J={label:.2f}", a, b))
    return pairs


def text_pairs(master: int) -> List[tuple]:
    """The original paragraph against seeded edits of itself, as word 3-gram sets."""
    words = ORIGINAL_TEXT.split()
    original = shingles(ORIGINAL_TEXT, 3)
    pairs = []
    for percent in (5, 10, 25, 50):
        rng = make_rng(derive_seed(master, "minhash_text_edit", percent))
        edited = list(words)
        positions = rng.sample(range(len(edited)), round(len(edited) * percent / 100))
        for position in positions:
            if rng.random() < 0.5:
                edited[position] = f"token{rng.randrange(10_000)}"
            else:
                edited[position] = ""
        variant = shingles(" ".join(w for w in edited if w), 3)
        pairs.append(("text", f"edit={percent}%", original, variant))
    return pairs


def minhash_study(config: Config, master: int, rows: List[dict]) -> Dict[str, Any]:
    print("\n=== STUDY 5 - MinHash ===", flush=True)
    pairs = simulated_pairs() + text_pairs(master)
    errors: Dict[tuple, List[float]] = {}
    estimates: Dict[tuple, List[float]] = {}
    for k in config.minhash_ks:
        for s in range(config.minhash_seeds):
            hasher = MinHash(num_perm=k, seed=derive_seed(master, "minhash", k, s) % (2 ** 61))
            for kind, label, a, b in pairs:
                truth = jaccard(a, b)
                estimate = MinHash.estimate(hasher.signature(a), hasher.signature(b))
                errors.setdefault((kind, label, k), []).append(abs(estimate - truth))
                estimates.setdefault((kind, label, k), []).append(estimate)
    error_rows: List[dict] = []
    scatter_rows: List[dict] = []
    for kind, label, a, b in pairs:
        truth = jaccard(a, b)
        for k in config.minhash_ks:
            stats = summarize(errors[(kind, label, k)])
            se = minhash_standard_error(truth, k)
            theoretical_mae = math.sqrt(2 / math.pi) * se
            error_rows.append({"k": k, "true_j": truth, "mean_abs_error": stats["mean"],
                               "std_abs_error": stats["std"], "theoretical_mae": theoretical_mae,
                               "kind": kind})
            seed0 = derive_seed(master, "minhash", k, 0) % (2 ** 61)
            for name, value, theory in (("mean_abs_error", stats["mean"], theoretical_mae),
                                        ("std_abs_error", stats["std"], None),
                                        ("estimate_mean",
                                         summarize(estimates[(kind, label, k)])["mean"], truth),
                                        ("standard_error_theoretical", se, None)):
                rows.append(row("minhash", f"{label},true_j={truth:.4f}", "monte_carlo",
                                "error_vs_k", kind, len(a | b), seed0, param_name="num_perm",
                                param_value=k, runs=config.minhash_seeds, metric_name=name,
                                metric_value=fmt(value, 10), theoretical_value=fmt(theory, 10)))
            if k in (min(config.minhash_ks), max(config.minhash_ks)):
                for estimate in estimates[(kind, label, k)][:10]:
                    scatter_rows.append({"k": k, "true_j": truth, "estimate": estimate})
        print(f"    {kind:<9} {label:<9} true J {truth:.4f}  MAE " + "  ".join(
            f"k={r['k']}:{r['mean_abs_error']:.4f}" for r in error_rows
            if r["true_j"] == truth and r["kind"] == kind), flush=True)
    return {"error_rows": error_rows, "scatter_rows": scatter_rows}


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: List[dict], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")


#: The two composites the accuracy panel draws: a Carmichael number where
#: Fermat is nearly always fooled, and the composite with the highest
#: strong-liar fraction in the study. Every composite is in the CSV.
FIGURE_COMPOSITES = (561, 8911)


def save_primality_figure(accuracy_rows: List[dict], speed_rows: List[dict],
                          path: str) -> None:
    """Draw the primality figure for the two featured composites.

    Predicted rates fall to 10^-37 by k = 20 for Miller-Rabin on 561, far
    below anything 10,000 trials can resolve (the rule-of-three bound is
    3 x 10^-4), so the rate axis stops at 10^-7 and steeper predicted lines
    simply leave the panel.
    """
    shown = [r for r in accuracy_rows if r["n"] in FIGURE_COMPOSITES] or accuracy_rows
    figure = plot_primality_accuracy(shown, speed_rows)
    figure.axes[0].set_ylim(1e-7, 2.0)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny parameter set, seconds")
    parser.add_argument("--out", default=RESULTS_DIR, help="directory for CSVs and figures")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    args = parser.parse_args(argv)

    config = SMOKE if args.smoke else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    master = args.seed
    apply_house_style()

    print("=" * 78)
    print("CSC 5300 Week 7 - randomized algorithms benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    : {platform.platform()}")
    print(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  master seed : {master}  (sub-seeds: derive_seed(master, experiment, ...))")
    print(f"  output      : {out}")

    started = time.perf_counter()
    rows: List[dict] = []
    qs = quicksort_study(config, master, rows)
    sel = selection_study(config, master, rows)
    prim = primality_study(config, master, rows)
    bloom = bloom_study(config, master, rows)
    minhash = minhash_study(config, master, rows)

    print("\n=== writing results ===")
    write_csv(os.path.join(out, "summary_table.csv"), rows, COLUMNS)
    write_csv(os.path.join(out, "week7_comparison_counts.csv"),
              [{"n": qs["comparison_sample"]["n"], "sample": i, "comparisons": c}
               for i, c in enumerate(qs["comparison_sample"]["counts"])],
              ["n", "sample", "comparisons"])
    write_csv(os.path.join(out, "week7_selection_ratios.csv"),
              [{"n": sel["ratio_sample"]["n"], "k_label": sel["ratio_sample"]["k_label"],
                "run": i, "comparisons_per_n": f"{r:.6f}"}
               for i, r in enumerate(sel["ratio_sample"]["ratios"])],
              ["n", "k_label", "run", "comparisons_per_n"])

    print("\n=== figures ===")
    for name, builder in (
        ("quicksort_comparison.png",
         lambda p: plot_quicksort_comparison(qs["random_series"], qs["sorted_series"],
                                             qs["comparison_sample"], save_path=p)),
        ("selection_runtime.png",
         lambda p: plot_selection_runtime(sel["series"], sel["depth_series"],
                                          sel["ratio_sample"], save_path=p)),
        ("primality_accuracy.png",
         lambda p: save_primality_figure(prim["accuracy_rows"], prim["speed_rows"], p)),
        ("bloom_filter_fp_rate.png",
         lambda p: plot_bloom_fp_rate(bloom["load_rows"], bloom["bits_rows"],
                                      memory=bloom["memory"], save_path=p)),
        ("minhash_similarity_plot.png",
         lambda p: plot_minhash_similarity(minhash["error_rows"], minhash["scatter_rows"],
                                           save_path=p)),
    ):
        path = os.path.join(out, name)
        builder(path)
        print(f"  wrote {os.path.relpath(path, REPO_ROOT)} "
              f"({os.path.getsize(path) / 1024:.0f} KiB)")

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
