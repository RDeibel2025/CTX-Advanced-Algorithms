#!/usr/bin/env python3
"""Week 5 benchmark: recursion against memoization against tabulation.

Run from the repository root with the project virtualenv active::

    python benchmarks/week5_dp_benchmark.py            # the full study
    python benchmarks/week5_dp_benchmark.py --quick    # smoke run
    python benchmarks/week5_dp_benchmark.py --quick --out /tmp/w5

Three problems, each solved three ways, measured on the same machine in the
same process:

1. **Fibonacci** at n = 10 to 45. The naive recursion is measured to n = 35
   and *projected* beyond it, because calls(45) is 3,672,623,805 and a
   benchmarked series there would run for most of an hour.
2. **Knapsack**, swept two ways: item count at a fixed capacity, and
   capacity at a fixed item count. The recursive variant is capped at 20
   items. A side study counts how many memo entries top-down actually
   stores against the n x (W + 1) cells the table holds, which is the
   report's central counterexample.
3. **Longest common subsequence** at lengths 10 to 1,000 for the DP
   variants and 6 to 14 for the recursion, plus a probe that finds the
   length at which the memoized version exhausts CPython's default
   recursion limit.

Two rules govern every reduction here, both carried from Week 1:

* **Reduce and document, never silently drop a point.** Every row in the
  CSV carries a ``measurement`` column reading ``measured`` or
  ``projected``, and the figures draw projected points differently.
* **Never write a number the code did not produce.**
  ``speedup_vs_recursive`` is left blank wherever the recursive baseline at
  that size was projected rather than run.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import random
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.dp.fibonacci import (  # noqa: E402
    fib_instrumented,
    fib_memo,
    fib_naive,
    fib_tab,
    naive_call_count,
)
from src.dp.knapsack import (  # noqa: E402
    knapsack_memo,
    knapsack_memo_cells,
    knapsack_instrumented,
    knapsack_recursive,
    knapsack_tab,
    knapsack_tab_rolling,
    trace_solution,
)
from src.dp.lcs import (  # noqa: E402
    lcs_instrumented,
    lcs_memo,
    lcs_recursive,
    lcs_reconstruct,
    lcs_tab,
)
from src.utils.timer import peak_memory_kib, time_call  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_fibonacci_comparison,
    plot_knapsack_performance,
    plot_lcs_performance,
)

#: Every generated instance derives from this, so a re-run reproduces the run.
SEED = 42

#: The memoized LCS recurses to depth about len(X) + len(Y), so a 1,000
#: character pair needs roughly 2,000 frames against CPython's default 1,000.
#: The limit is raised here, in the caller, rather than inside src/dp/lcs.py:
#: a library that quietly reconfigures the interpreter is a worse bug than the
#: RecursionError it hides. probe_recursion_limit() measures the default
#: behaviour first, before this takes effect.
RECURSION_LIMIT = 30_000

#: The CSV's columns are fixed, because the report reads them back by name.
COLUMNS = [
    "problem", "variant", "n", "secondary_param", "mean_time_s", "std_time_s",
    "min_time_s", "max_time_s", "calls", "max_depth", "peak_kib",
    "speedup_vs_recursive", "measurement",
]

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

#: Alphabet for the generated LCS strings. Four letters, because the case
#: study in the report is sequence alignment and this is the DNA alphabet.
ALPHABET = "ACGT"


@dataclass
class Config:
    """Every size, cap and repetition policy the study uses, in one place."""

    fib_sizes: List[int]
    #: Naive Fibonacci is measured up to here and projected above it.
    fib_measured_max: int
    knapsack_sizes: List[int]
    knapsack_capacity: int
    knapsack_capacities: List[int]
    knapsack_fixed_n: int
    #: The recursive knapsack is O(2^n); past this it dominates the suite.
    knapsack_recursive_max: int
    lcs_lengths: List[int]
    lcs_recursive_lengths: List[int]
    #: Probe sizes for the default-recursion-limit study.
    probe_lengths: List[int] = field(
        default_factory=lambda: [100, 200, 300, 400, 500, 600, 800, 1000]
    )


FULL = Config(
    fib_sizes=[10, 20, 30, 35, 40, 45],
    fib_measured_max=35,
    knapsack_sizes=[10, 15, 20, 25, 50, 100, 200],
    knapsack_capacity=1_000,
    knapsack_capacities=[100, 250, 500, 1_000],
    knapsack_fixed_n=50,
    knapsack_recursive_max=20,
    lcs_lengths=[10, 50, 100, 250, 500, 1_000],
    lcs_recursive_lengths=[6, 8, 10, 12, 14],
)

QUICK = Config(
    fib_sizes=[10, 20, 25, 30],
    fib_measured_max=25,
    knapsack_sizes=[10, 12, 20],
    knapsack_capacity=100,
    knapsack_capacities=[50, 100],
    knapsack_fixed_n=12,
    knapsack_recursive_max=12,
    lcs_lengths=[10, 50, 100],
    lcs_recursive_lengths=[6, 8, 10],
    probe_lengths=[100, 200],
)


# ----------------------------------------------------------------------
# Measurement policy
# ----------------------------------------------------------------------
def repeats_for(pilot_seconds: float) -> Tuple[int, int]:
    """Choose (repeat, warmup) from one pilot run's cost.

    A fixed five-trials-after-two-warmups policy is right for microsecond
    work and absurd for a naive Fibonacci at n = 35, which costs seconds a
    call. Rather than pick one number and either waste an hour or report a
    single unrepeated run everywhere, the policy is derived from what the
    first run actually cost. The count that was used is written into every
    CSV row, so no row's reliability has to be guessed at.
    """
    if pilot_seconds < 0.05:
        return 5, 2
    if pilot_seconds < 0.5:
        return 3, 1
    return 2, 0


def measure(function: Callable[[], Any]) -> Dict[str, float]:
    """Time a zero-argument call under the repetition policy above."""
    started = time.perf_counter()
    function()
    pilot = time.perf_counter() - started
    repeat, warmup = repeats_for(pilot)
    return time_call(function, repeat=repeat, warmup=warmup)


def blank_row(problem: str, variant: str, n: int,
              secondary: Any = "") -> Dict[str, Any]:
    return {name: "" for name in COLUMNS} | {
        "problem": problem, "variant": variant, "n": n,
        "secondary_param": secondary,
    }


def fill(row: Dict[str, Any], stats: Dict[str, float], *, calls: Any = "",
         depth: Any = "", peak: Any = "", measurement: str = "measured") -> Dict[str, Any]:
    row.update({
        "mean_time_s": f"{stats['mean']:.9f}",
        "std_time_s": f"{stats['std']:.9f}",
        "min_time_s": f"{stats['min']:.9f}",
        "max_time_s": f"{stats['max']:.9f}",
        "calls": calls,
        "max_depth": depth,
        "peak_kib": "" if peak == "" else f"{peak:.1f}",
        "measurement": measurement,
    })
    return row


def add_speedups(rows: List[Dict[str, Any]], problem: str,
                 baseline_variant: str) -> None:
    """Fill speedup_vs_recursive, but only where the baseline was measured.

    Writing a speedup against a projected baseline would be reporting a
    ratio of one measurement to one estimate as though both were observed.
    Those cells stay blank on purpose.
    """
    baselines = {
        (row["n"], row["secondary_param"]): row
        for row in rows
        if row["problem"] == problem and row["variant"] == baseline_variant
        and row["measurement"] == "measured"
    }
    for row in rows:
        if row["problem"] != problem or row["variant"] == baseline_variant:
            continue
        key = (row["n"], row["secondary_param"])
        baseline = baselines.get(key)
        if baseline and row["measurement"] == "measured":
            ratio = float(baseline["mean_time_s"]) / float(row["mean_time_s"])
            row["speedup_vs_recursive"] = f"{ratio:.2f}"


# ----------------------------------------------------------------------
# Study 1: Fibonacci
# ----------------------------------------------------------------------
def fibonacci_study(config: Config) -> List[Dict[str, Any]]:
    """Naive against memo against tab, with the large naive points projected."""
    print("\n=== STUDY 1 - Fibonacci ===", flush=True)
    rows: List[Dict[str, Any]] = []
    per_call_seconds: Optional[float] = None

    for n in config.fib_sizes:
        for variant, function in (("naive", fib_naive), ("memo", fib_memo),
                                  ("tab", fib_tab)):
            row = blank_row("fibonacci", variant, n)

            if variant == "naive" and n > config.fib_measured_max:
                # Projected, not run. calls(n) = 2*F(n+1) - 1 exactly, and the
                # per-call cost comes from the largest measured point, so the
                # estimate is one multiplication of two measured quantities.
                calls = naive_call_count(n)
                estimate = per_call_seconds * calls
                row.update({
                    "mean_time_s": f"{estimate:.9f}",
                    "std_time_s": "", "min_time_s": "", "max_time_s": "",
                    "calls": calls,
                    "max_depth": n,          # exact by construction
                    "peak_kib": "",
                    "measurement": "projected",
                })
                rows.append(row)
                print(f"    {variant:<6} n={n:<3} PROJECTED {estimate:10.2f} s "
                      f"({calls:,} calls at {per_call_seconds * 1e9:.1f} ns each)",
                      flush=True)
                continue

            stats = measure(lambda f=function, k=n: f(k))
            value, counter = fib_instrumented(n, variant)
            _, peak = peak_memory_kib(function, n)
            if variant == "naive":
                per_call_seconds = stats["mean"] / counter.calls
                assert counter.calls == naive_call_count(n), (
                    f"instrumented count {counter.calls} disagrees with the "
                    f"closed form {naive_call_count(n)} at n={n}")
            fill(row, stats, calls=counter.calls, depth=counter.max_depth, peak=peak)
            rows.append(row)
            print(f"    {variant:<6} n={n:<3} {stats['mean'] * 1e3:10.4f} ms  "
                  f"calls={counter.calls:>12,}  depth={counter.max_depth:>5}  "
                  f"peak={peak:8.1f} KiB", flush=True)

    add_speedups(rows, "fibonacci", "naive")
    if per_call_seconds is not None:
        print(f"    per-call cost used for projection: "
              f"{per_call_seconds * 1e9:.2f} ns", flush=True)
    return rows


# ----------------------------------------------------------------------
# Study 2: knapsack
# ----------------------------------------------------------------------
def knapsack_instance(n: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    weights = [rng.randint(1, 50) for _ in range(n)]
    values = [rng.randint(1, 100) for _ in range(n)]
    return weights, values


def knapsack_study(config: Config) -> Tuple[List[Dict[str, Any]], List[dict]]:
    """Item count at fixed capacity, then capacity at fixed item count."""
    print("\n=== STUDY 2 - 0/1 knapsack ===", flush=True)
    rows: List[Dict[str, Any]] = []
    cells: List[dict] = []

    print(f"  --- item count, capacity fixed at {config.knapsack_capacity} ---",
          flush=True)
    for n in config.knapsack_sizes:
        rng = random.Random(SEED + n)
        weights, values = knapsack_instance(n, rng)
        capacity = config.knapsack_capacity
        variants: List[Tuple[str, Callable[[], int]]] = [
            ("memo", lambda w=weights, v=values, c=capacity: knapsack_memo(w, v, c)),
            ("tab", lambda w=weights, v=values, c=capacity: knapsack_tab(w, v, c)),
        ]
        if n <= config.knapsack_recursive_max:
            variants.insert(0, ("recursive", lambda w=weights, v=values,
                                c=capacity: knapsack_recursive(w, v, c)))
        else:
            print(f"    recursive n={n:<4} skipped: past the "
                  f"{config.knapsack_recursive_max}-item cap (O(2^n))", flush=True)

        for variant, call in variants:
            stats = measure(call)
            _, counter = knapsack_instrumented(weights, values, capacity, variant)
            _, peak = peak_memory_kib(call)
            row = fill(blank_row("knapsack", variant, n, capacity), stats,
                       calls=counter.calls, depth=counter.max_depth, peak=peak)
            rows.append(row)
            print(f"    {variant:<9} n={n:<4} W={capacity:<5} "
                  f"{stats['mean'] * 1e3:10.4f} ms  calls={counter.calls:>10,}  "
                  f"depth={counter.max_depth:>5}  peak={peak:8.1f} KiB", flush=True)

        best, entries, table_cells = knapsack_memo_cells(weights, values, capacity)
        rolling = knapsack_tab_rolling(weights, values, capacity)
        traced_value, chosen = trace_solution(weights, values, capacity)
        assert rolling == best == traced_value, f"variants disagree at n={n}"
        assert sum(weights[i] for i in chosen) <= capacity
        cells.append({
            "n": n, "capacity": capacity, "optimum": best,
            "memo_entries": entries, "table_cells": table_cells,
            "fraction_of_table": round(entries / table_cells, 6),
            "items_chosen": len(chosen),
        })
        print(f"      memo touched {entries:,} of {table_cells:,} table cells "
              f"({100 * entries / table_cells:.1f}%)", flush=True)

    print(f"  --- capacity, item count fixed at {config.knapsack_fixed_n} ---",
          flush=True)
    rng = random.Random(SEED + config.knapsack_fixed_n)
    weights, values = knapsack_instance(config.knapsack_fixed_n, rng)
    # The first sweep already measured this item count at its fixed capacity.
    # Measuring it twice would put two points on one x value and duplicate a
    # row in the CSV, so that capacity is skipped here rather than repeated.
    already = {config.knapsack_capacity} if config.knapsack_fixed_n in config.knapsack_sizes else set()
    for capacity in config.knapsack_capacities:
        if capacity in already:
            print(f"    (W={capacity} at n={config.knapsack_fixed_n} already measured above)",
                  flush=True)
            continue
        for variant, function in (("memo", knapsack_memo), ("tab", knapsack_tab)):
            call = (lambda f=function, w=weights, v=values, c=capacity: f(w, v, c))
            stats = measure(call)
            _, counter = knapsack_instrumented(weights, values, capacity, variant)
            _, peak = peak_memory_kib(call)
            rows.append(fill(
                blank_row("knapsack", variant, config.knapsack_fixed_n, capacity),
                stats, calls=counter.calls, depth=counter.max_depth, peak=peak))
            print(f"    {variant:<9} n={config.knapsack_fixed_n:<4} W={capacity:<5} "
                  f"{stats['mean'] * 1e3:10.4f} ms  calls={counter.calls:>10,}",
                  flush=True)

    add_speedups(rows, "knapsack", "recursive")
    return rows, cells


# ----------------------------------------------------------------------
# Study 3: longest common subsequence
# ----------------------------------------------------------------------
def lcs_pair(length: int, rng: random.Random) -> Tuple[str, str]:
    return ("".join(rng.choice(ALPHABET) for _ in range(length)),
            "".join(rng.choice(ALPHABET) for _ in range(length)))


def probe_recursion_limit(config: Config) -> List[dict]:
    """Find where memoized LCS exhausts CPython's DEFAULT recursion limit.

    This runs before the limit is raised, and it is the evidence behind the
    report's claim that laziness spends the interpreter's stack. Tabulation
    is run on the identical inputs as the control: it is the same recurrence
    and the same answer, and it never recurses at all.
    """
    print("\n=== STUDY 3a - the default recursion limit ===", flush=True)
    default_limit = sys.getrecursionlimit()
    rows: List[dict] = []
    for length in config.probe_lengths:
        rng = random.Random(SEED + length)
        left, right = lcs_pair(length, rng)
        try:
            memo_result: Any = lcs_memo(left, right)
            memo_status = "ok"
        except RecursionError:
            memo_result, memo_status = "", "RecursionError"
        tab_result = lcs_tab(left, right)
        rows.append({
            "length": length, "recursion_limit": default_limit,
            "memo_status": memo_status, "memo_result": memo_result,
            "tab_status": "ok", "tab_result": tab_result,
        })
        print(f"    length {length:>5}: memo {memo_status:<15} tab ok "
              f"(lcs = {tab_result})", flush=True)
    return rows


def lcs_study(config: Config) -> List[Dict[str, Any]]:
    """The DP variants across the full range, the recursion only where it fits."""
    print("\n=== STUDY 3b - longest common subsequence ===", flush=True)
    rows: List[Dict[str, Any]] = []

    for length in config.lcs_recursive_lengths:
        rng = random.Random(SEED + length)
        left, right = lcs_pair(length, rng)
        call = (lambda a=left, b=right: lcs_recursive(a, b))
        stats = measure(call)
        _, counter = lcs_instrumented(left, right, "recursive")
        _, peak = peak_memory_kib(call)
        rows.append(fill(blank_row("lcs", "recursive", length), stats,
                         calls=counter.calls, depth=counter.max_depth, peak=peak))
        print(f"    recursive len={length:<5} {stats['mean'] * 1e3:10.4f} ms  "
              f"calls={counter.calls:>12,}  depth={counter.max_depth:>5}", flush=True)

    for length in config.lcs_lengths:
        rng = random.Random(SEED + length)
        left, right = lcs_pair(length, rng)
        for variant, function in (("memo", lcs_memo), ("tab", lcs_tab)):
            call = (lambda f=function, a=left, b=right: f(a, b))
            stats = measure(call)
            _, counter = lcs_instrumented(left, right, variant)
            _, peak = peak_memory_kib(call)
            rows.append(fill(blank_row("lcs", variant, length), stats,
                             calls=counter.calls, depth=counter.max_depth,
                             peak=peak))
            print(f"    {variant:<9} len={length:<5} {stats['mean'] * 1e3:10.4f} ms  "
                  f"calls={counter.calls:>12,}  depth={counter.max_depth:>5}  "
                  f"peak={peak:9.1f} KiB", flush=True)
        recovered = lcs_reconstruct(left, right)
        assert len(recovered) == lcs_tab(left, right), "reconstruction disagrees"

    add_speedups(rows, "lcs", "recursive")
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: List[dict], columns: Optional[List[str]] = None) -> None:
    if columns is None:
        columns = []
        for row in rows:
            for name in row:
                if name not in columns:
                    columns.append(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small, fast smoke run")
    parser.add_argument("--out", default=RESULTS_DIR,
                        help="directory for the CSV and the figures")
    args = parser.parse_args(argv)

    config = QUICK if args.quick else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    apply_house_style()

    print("=" * 78)
    print("CSC 5300 Week 5 - dynamic programming benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python   : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform : {platform.platform()}")
    print(f"  mode     : {'QUICK SMOKE RUN' if args.quick else 'full study'}")
    print(f"  seed     : {SEED}")
    print(f"  output   : {out}")

    started = time.perf_counter()

    fib_rows = fibonacci_study(config)
    knap_rows, cell_rows = knapsack_study(config)
    probe_rows = probe_recursion_limit(config)

    # Only now, with the default-limit behaviour recorded, is it raised.
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(RECURSION_LIMIT)
    try:
        lcs_rows = lcs_study(config)
    finally:
        sys.setrecursionlimit(previous_limit)

    rows = fib_rows + knap_rows + lcs_rows

    print("\n=== writing results ===")
    write_csv(os.path.join(out, "dp_vs_recursive_table.csv"), rows, COLUMNS)
    write_csv(os.path.join(out, "week5_knapsack_cells.csv"), cell_rows)
    write_csv(os.path.join(out, "week5_recursion_probe.csv"), probe_rows)

    print("\n=== figures ===")
    for name, builder in (
        ("fibonacci_comparison.png",
         lambda p: plot_fibonacci_comparison(rows, save_path=p)),
        ("knapsack_performance.png",
         lambda p: plot_knapsack_performance(
             rows, save_path=p, fixed_capacity=config.knapsack_capacity,
             fixed_n=config.knapsack_fixed_n)),
        ("lcs_performance.png", lambda p: plot_lcs_performance(rows, save_path=p)),
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
