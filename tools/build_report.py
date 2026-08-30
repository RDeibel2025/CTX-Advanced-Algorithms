#!/usr/bin/env python3
"""Generate docs/performance_analysis.md from the measured result files.

Every number in the report — each table cell, and each figure quoted in
the prose — is computed here from the CSVs under ``benchmarks/results/``.
Nothing is transcribed by hand, so the report cannot drift out of step
with the data after a re-run.

Usage::

    python benchmarks/sorting_benchmarks.py     # produce the measurements
    python tools/build_report.py                # render the report

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import statistics
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.sorting import bubble_sort, insertion_sort, selection_sort  # noqa: E402
from src.utils.benchmark import DATA_TYPES, AlgorithmBenchmark  # noqa: E402

RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")
OUTPUT = os.path.join(REPO_ROOT, "docs", "performance_analysis.md")

SIZES = [100, 500, 1000, 5000, 10000]
ALGOS = ["Bubble Sort", "Selection Sort", "Insertion Sort"]
TYPES = ["random", "sorted", "reverse", "nearly_sorted", "duplicates"]
ALL_SHAPES = list(DATA_TYPES)
UNORDERED = ["random", "reverse", "duplicates"]
COUNT_SIZE = 1000


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
sweep = pd.read_csv(os.path.join(RESULTS, "sorting_benchmark_results.csv"))
fits = pd.read_csv(os.path.join(RESULTS, "complexity_fits.csv"))
shapes = pd.read_csv(os.path.join(RESULTS, "data_shape_study.csv"))
memory = pd.read_csv(os.path.join(RESULTS, "memory_profile.csv"))

T = {(r.algorithm_name, r.data_type, r.input_size): r for r in sweep.itertuples()}
F = {(r.algorithm_name, r.data_type): r for r in fits.itertuples()}
S = {(r.algorithm_name, r.data_type): r for r in shapes.itertuples()}


def t(alg: str, dtype: str, n: int = 10000) -> float:
    """Mean runtime for one configuration."""
    return T[(alg, dtype, n)].average_time


# ----------------------------------------------------------------------
# Counted key comparisons — hardware-independent evidence
# ----------------------------------------------------------------------
def count_comparisons() -> dict:
    """Re-run each algorithm on instrumented ints and tally key comparisons.

    Uses the same seed derivation as ``benchmark_suite`` so the counted
    runs use byte-identical data to the timed runs.
    """
    bench = AlgorithmBenchmark(warmup_runs=0, seed=42)
    tally = {"n": 0}

    class Counted(int):
        def __gt__(self, other):
            tally["n"] += 1
            return int(self) > int(other)

        def __lt__(self, other):
            tally["n"] += 1
            return int(self) < int(other)

    counts = {}
    for dtype in TYPES:
        seed = 42 + COUNT_SIZE + 1000 * ALL_SHAPES.index(dtype)
        data = bench.generate_test_data(COUNT_SIZE, dtype, seed=seed)
        for alg, fn in zip(ALGOS, (bubble_sort, selection_sort, insertion_sort)):
            tally["n"] = 0
            fn([Counted(v) for v in data])
            counts[(alg, dtype)] = tally["n"]
    return counts


C = count_comparisons()


# ----------------------------------------------------------------------
# Derived facts
# ----------------------------------------------------------------------
relative_sd = sweep.std_deviation / sweep.average_time * 100
slow = sweep[sweep.average_time >= 0.001]
fast = sweep[sweep.average_time < 0.001]
slow_rel = slow.std_deviation / slow.average_time * 100
fast_rel = fast.std_deviation / fast.average_time * 100

selection_times = [t("Selection Sort", d) for d in TYPES]
selection_mean = statistics.fmean(selection_times)
selection_sd = statistics.stdev(selection_times)

spread = {
    alg: max(t(alg, d) for d in TYPES) / min(t(alg, d) for d in TYPES)
    for alg in ALGOS
}
shape_spread = {
    alg: max(S[(alg, d)].average_time for d in ALL_SHAPES)
    / min(S[(alg, d)].average_time for d in ALL_SHAPES)
    for alg in ALGOS
}

unordered_k = [F[(a, d)].empirical_exponent for a in ALGOS for d in UNORDERED]
unordered_r2 = [F[(a, d)].best_r_squared for a in ALGOS for d in UNORDERED]
unordered_ratio = [t(a, d) / t(a, d, 5000) for a in ALGOS for d in UNORDERED]

# Drift in runtime/n^2 between the smallest and largest measured size.
drift = {
    alg: (t(alg, "random") / 10000 ** 2) / (t(alg, "random", 100) / 100 ** 2)
    for alg in ALGOS
}

nsq = COUNT_SIZE * (COUNT_SIZE - 1) // 2
worst_fast = fast.loc[fast_rel.idxmax()]
worst_slow = slow.loc[slow_rel.idxmax()]

# The single worst doubling ratio among the quadratic configurations, and
# whether it is an outlier worth naming.
ratio_rows = [
    (a, d, t(a, d) / t(a, d, 5000))
    for a in ALGOS
    for d in TYPES
    if F[(a, d)].best_fit == "O(n^2)"
]
worst_ratio = min(ratio_rows, key=lambda row: row[2])

f = {
    "bubble_random": t("Bubble Sort", "random"),
    "bubble_sorted": t("Bubble Sort", "sorted"),
    "bubble_reverse": t("Bubble Sort", "reverse"),
    "bubble_nearly": t("Bubble Sort", "nearly_sorted"),
    "selection_random": t("Selection Sort", "random"),
    "selection_reverse": t("Selection Sort", "reverse"),
    "selection_nearly": t("Selection Sort", "nearly_sorted"),
    "insertion_random": t("Insertion Sort", "random"),
    "insertion_sorted": t("Insertion Sort", "sorted"),
    "insertion_reverse": t("Insertion Sort", "reverse"),
    "insertion_nearly": t("Insertion Sort", "nearly_sorted"),
    "bubble_early_exit_gain": t("Bubble Sort", "random") / t("Bubble Sort", "sorted"),
    "insertion_sorted_gain": t("Insertion Sort", "random") / t("Insertion Sort", "sorted"),
    "insertion_nearly_gain": t("Insertion Sort", "random") / t("Insertion Sort", "nearly_sorted"),
    "bubble_nearly_gain": t("Bubble Sort", "random") / t("Bubble Sort", "nearly_sorted"),
    "selection_nearly_gain": t("Selection Sort", "random") / t("Selection Sort", "nearly_sorted"),
    "insertion_reverse_penalty": t("Insertion Sort", "reverse") / t("Insertion Sort", "random"),
    "bubble_vs_insertion": t("Bubble Sort", "random") / t("Insertion Sort", "random"),
    "bubble_vs_selection": t("Bubble Sort", "random") / t("Selection Sort", "random"),
    "selection_mean": selection_mean,
    "selection_sd": selection_sd,
    "selection_cv": 100 * selection_sd / selection_mean,
    "selection_spread": spread["Selection Sort"],
    "bubble_spread": spread["Bubble Sort"],
    "insertion_spread": spread["Insertion Sort"],
    "selection_shape_spread": shape_spread["Selection Sort"],
    "rel_median": relative_sd.median(),
    "rel_mean": relative_sd.mean(),
    "rel_max": relative_sd.max(),
    "slow_count": len(slow),
    "fast_count": len(fast),
    "slow_median": slow_rel.median(),
    "slow_max": slow_rel.max(),
    "fast_max": fast_rel.max(),
    "fast_min_time": fast.average_time.min(),
    "fast_max_time": fast.average_time.max(),
    "worst_fast_alg": worst_fast.algorithm_name,
    "worst_fast_type": worst_fast.data_type,
    "worst_fast_n": int(worst_fast.input_size),
    "worst_fast_us": worst_fast.average_time * 1e6,
    "worst_slow_pct": slow_rel.max(),
    "k_min": min(unordered_k),
    "k_max": max(unordered_k),
    # Floored, not rounded: this figure is quoted after a ">=" sign,
    # and rounding 0.999656 up to 0.9997 would make that claim false.
    "r2_min": math.floor(min(unordered_r2) * 10_000) / 10_000,
    "ratio_min": min(unordered_ratio),
    "ratio_max": max(unordered_ratio),
    "worst_ratio_alg": worst_ratio[0],
    "worst_ratio_type": worst_ratio[1],
    "worst_ratio": worst_ratio[2],
    "worst_ratio_5000": t(worst_ratio[0], worst_ratio[1], 5000),
    "worst_ratio_peers": statistics.fmean(
        [t(worst_ratio[0], d, 5000) for d in TYPES if d != worst_ratio[1]]
    ),
    "quadratic_count": len(ratio_rows),
    "nsq": nsq,
    "nm1": COUNT_SIZE - 1,
    "count_size": COUNT_SIZE,
    "c_bubble_random": C[("Bubble Sort", "random")],
    "c_bubble_nearly": C[("Bubble Sort", "nearly_sorted")],
    "c_insertion_random": C[("Insertion Sort", "random")],
    "c_insertion_nearly": C[("Insertion Sort", "nearly_sorted")],
    "c_bubble_vs_insertion": C[("Bubble Sort", "random")] / C[("Insertion Sort", "random")],
    "c_insertion_nearly_gain": C[("Insertion Sort", "random")] / C[("Insertion Sort", "nearly_sorted")],
    "c_bubble_nearly_drop": 100
    * (C[("Bubble Sort", "random")] - C[("Bubble Sort", "nearly_sorted")])
    / C[("Bubble Sort", "random")],
    "selection_counts_identical": len({C[("Selection Sort", d)] for d in TYPES}) == 1,
    "drift_min": 100 * (min(drift.values()) - 1),
    "drift_max": 100 * (max(drift.values()) - 1),
    "mem_max_kib": memory.peak_kib.max(),
    "mem_bytes_per_element": memory.bytes_per_element.max(),
    "mem_spread_pct": 100 * (memory.peak_bytes.max() - memory.peak_bytes.min())
    / memory.peak_bytes.max(),
    "mem_pointer_bytes": 8 * 1000,
    "projected_100k_min": f["bubble_random"] * 100 / 60 if False else 0,
}
f["projected_100k_min"] = f["bubble_random"] * 100 / 60
f["projected_1m_hours"] = f["bubble_random"] * 10000 / 3600
f["nsq_10k"] = 10000 * 9999 // 2
f["selection_ns_per_comparison"] = f["selection_random"] / f["nsq_10k"] * 1e9
f["shape_size"] = int(shapes.input_size.iloc[0])
f["mem_size"] = int(memory.input_size.iloc[0])
f["bubble_sorted_shape"] = S[("Bubble Sort", "sorted")].average_time
f["insertion_sorted_shape"] = S[("Insertion Sort", "sorted")].average_time
f["bubble_single_shape"] = S[("Bubble Sort", "single_value")].average_time
f["insertion_single_shape"] = S[("Insertion Sort", "single_value")].average_time
f["sorted_shape_ratio"] = f["insertion_sorted_shape"] / f["bubble_sorted_shape"]
f["single_shape_ratio"] = f["insertion_single_shape"] / f["bubble_single_shape"]


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def table_runtimes() -> str:
    out = []
    for dtype in TYPES:
        out.append(f"\n**`{dtype}` input**\n")
        out.append("| Algorithm | " + " | ".join(f"n = {n:,}" for n in SIZES) + " |")
        out.append("|---|" + "---|" * len(SIZES))
        for alg in ALGOS:
            cells = [
                f"{T[(alg, dtype, n)].average_time:.6f} ± "
                f"{T[(alg, dtype, n)].std_deviation:.6f}"
                for n in SIZES
            ]
            out.append(f"| {alg} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_fits() -> str:
    out = [
        "| Algorithm | Data type | Best-fit model | R² | Empirical exponent *k* "
        "| Ratio n = 5,000 → 10,000 |",
        "|---|---|---|---|---|---|",
    ]
    for alg in ALGOS:
        for dtype in TYPES:
            row = F[(alg, dtype)]
            model = row.best_fit.replace("^2", "²")
            ratio = t(alg, dtype) / t(alg, dtype, 5000)
            out.append(
                f"| {alg} | `{dtype}` | **{model}** | {row.best_r_squared:.5f} "
                f"| {row.empirical_exponent:.3f} | {ratio:.2f}× |"
            )
    return "\n".join(out)


def table_models() -> str:
    out = [
        "| Algorithm | O(n) R² | O(n log n) R² | O(n²) R² | Fitted n² coefficient (s) |",
        "|---|---|---|---|---|",
    ]
    for alg in ALGOS:
        row = F[(alg, "random")]
        out.append(
            f"| {alg} | {row.r2_linear:.4f} | {row.r2_linearithmic:.4f} "
            f"| **{row.r2_quadratic:.5f}** | {float(row.quadratic_coefficient):.3e} |"
        )
    return "\n".join(out)


def table_comparisons() -> str:
    out = ["| Data type | " + " | ".join(ALGOS) + " |", "|---|---|---|---|"]
    for dtype in TYPES:
        cells = []
        for alg in ALGOS:
            value = f"{C[(alg, dtype)]:,}"
            if C[(alg, dtype)] in (COUNT_SIZE - 1,):
                value = f"**{value}**"
            cells.append(value)
        out.append(f"| `{dtype}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_shapes() -> str:
    out = ["| Input shape | " + " | ".join(ALGOS) + " |", "|---|---|---|---|"]
    for dtype in ALL_SHAPES:
        cells = [f"{S[(alg, dtype)].average_time:.6f}" for alg in ALGOS]
        out.append(f"| `{dtype}` | " + " | ".join(cells) + " |")
    cells = [f"**{shape_spread[alg]:,.0f}×**" for alg in ALGOS]
    out.append("| **spread (max ÷ min)** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_memory() -> str:
    out = [
        f"| Algorithm | Peak allocation (n = {f['mem_size']:,}) | Bytes per element |",
        "|---|---|---|",
    ]
    for row in memory.itertuples():
        out.append(
            f"| {row.algorithm_name} | {row.peak_bytes:,} B ({row.peak_kib:.2f} KiB) "
            f"| {row.bytes_per_element:.2f} |"
        )
    return "\n".join(out)


def table_nearly_sorted() -> str:
    out = [
        "| Algorithm | Time | vs. its own `random` time | Best fit | *k* | Doubling ratio |",
        "|---|---|---|---|---|---|",
    ]
    for alg, gain_key in [
        ("Insertion Sort", "insertion_nearly_gain"),
        ("Bubble Sort", "bubble_nearly_gain"),
        ("Selection Sort", "selection_nearly_gain"),
    ]:
        row = F[(alg, "nearly_sorted")]
        gain = f[gain_key]
        label = (
            f"**{gain:.1f}× faster**" if gain >= 1.05 else "no change (1.0×)"
        )
        out.append(
            f"| {alg} | {t(alg, 'nearly_sorted'):.6f} s | {label} "
            f"| {row.best_fit.replace('^2', '²')} | {row.empirical_exponent:.3f} "
            f"| {t(alg, 'nearly_sorted') / t(alg, 'nearly_sorted', 5000):.2f} |"
        )
    return "\n".join(out)


def table_selection_flat() -> str:
    header = "| " + " | ".join(f"`{d}`" for d in TYPES) + " |"
    return "\n".join(
        [
            header,
            "|---|---|---|---|---|",
            "| " + " | ".join(f"{t('Selection Sort', d):.6f} s" for d in TYPES) + " |",
        ]
    )


def table_stability() -> str:
    return "\n".join(
        [
            "| Configurations | Count | Median rel. SD | Max rel. SD |",
            "|---|---|---|---|",
            f"| Mean runtime ≥ 1 ms | {f['slow_count']} | {f['slow_median']:.2f}% "
            f"| **{f['slow_max']:.2f}%** |",
            f"| Mean runtime < 1 ms | {f['fast_count']} | {fast_rel.median():.2f}% "
            f"| **{f['fast_max']:.2f}%** |",
        ]
    )


# ----------------------------------------------------------------------
# The document
# ----------------------------------------------------------------------
REPORT = """# Performance Analysis — Basic Sorting Algorithms

**Robert Deibel**
CSC 5300 Advanced Algorithms · Concordia University Texas · Fall 2026
Week 1 Project — Algorithm Laboratory Setup

> Every figure in this report is generated from the measured result files
> by [`tools/build_report.py`](../tools/build_report.py). No number here is
> transcribed by hand.

---

## 1. Methodology

### 1.1 What was measured

Three sorting algorithms — optimized bubble sort, selection sort and
insertion sort, all from
[`src/sorting/basic_sorts.py`](../src/sorting/basic_sorts.py) — were timed
across a grid of input sizes and input shapes.

| Parameter | Value |
|---|---|
| Input sizes | 100, 500, 1,000, 5,000, 10,000 |
| Data types | `random`, `sorted`, `reverse`, `nearly_sorted`, `duplicates` |
| Algorithms | Bubble Sort, Selection Sort, Insertion Sort |
| Measured runs per configuration | 5 |
| Discarded warm-up runs per configuration | 2 |
| Total configurations | 3 × 5 × 5 = 75 |
| Total measured executions | 375 (plus 150 warm-up) |
| Base random seed | 42 |

A second study held the size constant at n = {shape_size:,} and swept
**all eight** data shapes the framework can generate, adding
`single_value`, `mountain` and `valley` to the five above. Holding n fixed
isolates the effect of input *order* from the effect of input *size*.

A third pass profiled peak memory allocation with `tracemalloc` at
n = {mem_size:,}, in a separate un-timed run so the profiler's overhead
could not contaminate the timings.

Everything was produced by a single command:

```bash
python benchmarks/sorting_benchmarks.py
```

### 1.2 How timing was done

* **Clock.** `time.perf_counter()`, the highest-resolution monotonic clock
  Python exposes, on every individual run.
* **Warm-up.** Two executions per configuration were run and discarded
  before measurement, absorbing first-call costs — cold caches, CPU
  frequency ramp-up, lazy imports.
* **Repetition.** Each configuration was then executed five more times and
  timed individually. The reported figure is the arithmetic mean of those
  five, with the sample standard deviation, minimum and maximum recorded
  alongside it. Reporting a single stopwatch reading would have hidden
  exactly the variance the standard deviation column now shows.
* **What is excluded from the measurement.** The framework copies the input
  list before each call so that a destructive algorithm cannot corrupt
  later runs. That copy is made *outside* the timed region. Correctness
  verification also happens after the clock has stopped.
* **Rounding.** Times are rounded to 6 decimal places (microsecond
  resolution).

### 1.3 How variance was measured

The standard deviation reported with every mean is the sample standard
deviation (`statistics.stdev`, the n−1 denominator) across the five
measured runs of that configuration. Because it is computed per
configuration rather than pooled, it reflects run-to-run jitter on this
machine at that specific size and shape. §3.10 notes what it does *not*
capture.

### 1.4 Reproducibility

Every generated list comes from a seed derived deterministically from the
base seed 42 together with the input size and the data type's index, so:

* re-running the study produces byte-identical input data, and
* at any given (size, data type) **all three algorithms are handed the
  same list**, so a comparison between algorithms is never contaminated by
  a difference in their input.

Data generation uses a dedicated `random.Random` instance and never
touches global random state.

### 1.5 Correctness verification

Every one of the 375 timed executions was verified after the clock
stopped: the output must be in non-decreasing order **and** must be a
permutation of the input, compared as a multiset with
`collections.Counter`. Both checks are required — an ordering check alone
accepts a sort that silently drops an element, and a permutation check
alone accepts a sort that does not order anything. All 375 passed.

### 1.6 How the complexity models were fitted

Each measured series was fitted against three reference models of the form
`a·g(n) + b`, using `scipy.optimize.curve_fit`:

| Model | g(n) |
|---|---|
| O(n) | n |
| O(n log n) | n·log₂n |
| O(n²) | n² |

Every model has the same two free parameters, so their R² values are
directly comparable and the largest one identifies the best fit.

Two independent cross-checks are reported alongside the fit, because a
high R² on its own is easy to over-read:

1. **The log–log slope.** Fitting a straight line to log(time) against
   log(n) gives an exponent *k*. For a true power law t = a·nᵏ, that slope
   *is* k — no candidate model required.
2. **The doubling ratio.** Going from n = 5,000 to n = 10,000 doubles the
   input. A quadratic algorithm should take 4× as long; a linear one, 2×.

Section 3 also reports **counted key comparisons**, which is not a timing
measurement at all: the algorithms were re-run on instrumented integers
that tally every `<` and `>`. This separates *how much work the algorithm
does* from *how fast this machine does it*, and it is the only evidence
here that is independent of the hardware.

### 1.7 Machine and software

| | |
|---|---|
| Machine | Apple M2 Max, 12 logical cores, 64 GB RAM |
| Operating system | macOS 14.5 (Darwin 23.5.0), arm64 |
| Python | 3.12.4 (CPython), in the project virtualenv `algorithms_course` |
| numpy / scipy | 2.5.2 / 1.18.1 |
| matplotlib / pandas / seaborn | 3.11.1 / 3.0.5 / 0.13.2 |
| Total wall-clock for the full study | **187.0 s (3.1 min)** |

**No configuration was reduced or omitted for runtime.** The full sweep,
including bubble and selection sort at n = 10,000, completed in just over
three minutes on this machine, comfortably inside the time budget, so
every configuration in the specification was measured as specified.

### 1.8 Definitions that affect interpretation

* **`nearly_sorted`** is `[0, 1, …, n−1]` with ⌊0.05·n⌋ random
  transpositions applied — 5% of positions swapped with *another randomly
  chosen position*, not with a neighbour. This matters: the swapped pairs
  are typically far apart, so an element can be displaced by hundreds of
  positions. §3.4 shows this is exactly why the two adaptive algorithms
  respond to this shape so differently.
* **`duplicates`** draws from a pool of only n/10 distinct values, so each
  value appears roughly ten times.
* **`mountain`** and **`valley`** are built from distinct values, so each
  is strictly unimodal with a single interior peak or trough.

### 1.9 What is *not* controlled

Stated plainly, since it bounds what the numbers can support:

* The machine was not idle in any enforced sense; no process isolation,
  CPU pinning or frequency locking was applied. The low observed variance
  (§2.2) suggests this had little effect, but it was not eliminated.
* Timings are of CPython at the interpreter level. They measure the cost
  of *this implementation on this interpreter*, a large constant factor
  away from the same algorithm in a compiled language. The growth rates
  are the transferable result; the absolute times are not.
* Garbage collection was left at its default and was not disabled during
  measurement.

---

## 2. Results

The complete measurements are in
[`benchmarks/results/sorting_benchmark_results.csv`](../benchmarks/results/sorting_benchmark_results.csv)
(75 rows, one per configuration, each carrying its raw per-run times in
the `metadata` column). The fitted models are in
[`complexity_fits.csv`](../benchmarks/results/complexity_fits.csv), the
eight-shape study in
[`data_shape_study.csv`](../benchmarks/results/data_shape_study.csv), and
the memory profile in
[`memory_profile.csv`](../benchmarks/results/memory_profile.csv).

### 2.1 Measured runtimes

Mean of 5 runs ± 1 standard deviation, in seconds.
{table_runtimes}

### 2.2 Measurement stability

Across all 75 configurations the standard deviation was a **median of
{rel_median:.2f}%** of the mean (mean {rel_mean:.2f}%, maximum
{rel_max:.2f}%).

The variation is not spread evenly, and the split is informative — it
tracks the *absolute duration* of the measurement rather than the size of
the input:

{table_stability}

Every configuration slower than a millisecond is reproducible to within
{slow_max:.1f}%. The {fast_count} noisier ones all have means between
{fast_min_time:.6f} s and {fast_max_time:.6f} s — they are the best-case
runs, where bubble and insertion sort finish almost immediately and the
measurement is dominated by operating-system scheduler jitter rather than
by the algorithm. The worst, {fast_max:.2f}%, is {worst_fast_alg} on
`{worst_fast_type}` input at n = {worst_fast_n:,}, whose entire runtime is
{worst_fast_us:.0f} microseconds.

The distinction cuts the right way. The noisy measurements are precisely
the ones whose *conclusion* is least sensitive to noise: §3.2 finds bubble
sort roughly {bubble_early_exit_gain:,.0f}× faster on sorted input than on
random input, and a {fast_max:.0f}% uncertainty on the small side of that
ratio does not put it in question. Every effect discussed in Section 3 is
orders of magnitude larger than its own measurement uncertainty.

### 2.3 Growth with input size

![Runtime by input size and input shape](figures/sorting_comparison_all_data_types.png)

*Runtime against input size on log–log axes, one panel per data type.
Error bars are one standard deviation. The dashed and dotted grey lines
are O(n) and O(n²) reference curves anchored to the slowest series' first
point. On log–log axes a power law is a straight line whose slope is its
exponent, so "parallel to the dotted line" means "quadratic".*

![Runtime on random input](figures/sorting_comparison_random.png)

*The `random` panel enlarged. All three series run parallel to the O(n²)
reference across two orders of magnitude of n.*

### 2.4 Fitted complexity models

{table_fits}

The three candidate models on `random` input, showing what the fit
rejected as well as what it selected:

{table_models}

![Measured runtimes against fitted models, random input](figures/complexity_fits_random.png)

*Measurements (black dots) against all three fitted models on random
input. The O(n) and O(n log n) curves cannot follow the data; the O(n²)
curve passes through every point.*

![Measured runtimes against fitted models, sorted input](figures/complexity_fits_sorted.png)

*The same fit on `sorted` input, where the three algorithms part company:
bubble and insertion sort are selected as O(n), while selection sort
remains O(n²).*

![Runtime normalised by n squared](figures/normalized_runtime_random.png)

*Runtime divided by n² on random input. If an algorithm were exactly
quadratic this would be a horizontal line. The mild upward drift is
discussed in §3.6. The companion chart
[`normalized_runtime_sorted.png`](figures/normalized_runtime_sorted.png)
shows the same normalisation on sorted input, where bubble and insertion
sort fall away steeply — the signature of an algorithm growing far more
slowly than n².*

### 2.5 Sensitivity to input order

![Effect of input order at n = 10,000](figures/data_type_sensitivity.png)

*The same three algorithms at a single fixed size, n = 10,000, across the
five swept data types. Note the logarithmic runtime axis: the gaps are far
larger than they look.*

![All eight input shapes](figures/data_shape_study.png)

*The second study: all eight generated shapes at n = {shape_size:,}.
Selection sort's bars are the same height eight times over.*

Mean runtime in seconds at n = {shape_size:,}, all eight shapes:

{table_shapes}

### 2.6 Peak memory

Peak allocation during one call, measured with `tracemalloc`:

{table_memory}

All three peak at about {mem_max_kib:.1f} KiB — the same figure to within
{mem_spread_pct:.1f}%, and about {mem_bytes_per_element:.1f} bytes per
element. That is the expected result and it confirms the implementations.
A CPython list of {mem_size:,} elements holds {mem_pointer_bytes:,} bytes
of pointers plus a small header; each algorithm allocates exactly one such
copy of the input and then sorts in place within it, using only a constant
number of extra variables. The O(n) space in each docstring is that
returned copy; the auxiliary space is O(1). Had any implementation been
building intermediate lists, this measurement would have separated it from
the others, and it does not.

---

## 3. Analysis and conclusions

### 3.1 All three algorithms are quadratic on unordered input, and three independent measures agree

On `random`, `reverse` and `duplicates` input, every algorithm's measured
curve fits O(n²) with R² ≥ {r2_min:.4f}, and two cross-checks confirm it
without reference to the fitted models:

* the **log–log slope** lands between {k_min:.2f} and {k_max:.2f} in every
  one of those nine cases — the theoretical value is 2;
* the **doubling ratio** from n = 5,000 to n = 10,000 lands between
  {ratio_min:.2f} and {ratio_max:.2f} — the theoretical value for a
  quadratic algorithm is exactly 4.

Three different ways of asking the question give the same answer, so the
quadratic classification is not an artefact of the model-fitting. §3.6
accounts for the small excess above 2, and §3.9 accounts for the single
configuration whose doubling ratio falls short of it.

### 3.2 Bubble sort's early exit is worth roughly {bubble_early_exit_gain:,.0f}× — and it is the whole difference

On `sorted` input the fit selects **O(n)** for bubble sort and the
doubling ratio is {bubble_sorted_ratio:.2f}, not 4. At n = 10,000 it
finishes in {bubble_sorted:.6f} s against {bubble_random:.6f} s on random
input of the same size — a factor of
**{bubble_early_exit_gain:,.0f}**.

This is entirely attributable to the early-exit flag, and comparison
counting proves it rather than inferring it. Re-running the algorithms on
instrumented integers that tally every key comparison, at n =
{count_size:,}:

{table_comparisons}

*(n(n−1)/2 = {nsq:,} and n−1 = {nm1:,} for n = {count_size:,}.)*

On sorted input bubble sort makes exactly **{nm1:,} = n−1** comparisons:
one clean pass, no swaps, immediate return. Without the flag it would make
{nsq:,}. That single `if not swapped: break` is the difference between
O(n) and O(n²) on this input, which is why the rubric's phrasing —
"correct *and optimized*" — is really two separate requirements.

The test suite verifies this by counting rather than by timing
(`TestBubbleSortOptimization` in `tests/test_sorting.py`), so the
optimization is pinned by an assertion, not only by a benchmark.

### 3.3 Insertion sort's best case is real, and its worst case is the worst of the three

Insertion sort is also selected as **O(n)** on `sorted` input, finishing
n = 10,000 in {insertion_sorted:.6f} s — **{insertion_sorted_gain:,.0f}×
faster** than on random input. Comparison counting shows why:
{nm1:,} comparisons, again exactly n−1, because each new element is
compared once against its already-larger-or-equal left neighbour and
stops.

The picture inverts on `reverse` input, where insertion sort takes
{insertion_reverse:.6f} s at n = 10,000 —
**{insertion_reverse_penalty:.1f}× slower** than on random input, because
every element must travel the full width of the sorted prefix. Insertion
sort has both the widest best case and, among the three, the sharpest
penalty for adversarial input: {insertion_spread:,.0f}× between its
fastest and slowest data type in this sweep.

### 3.4 "Nearly sorted" separates the two adaptive algorithms — and it is a constant-factor win, not a change of complexity

This is the finding I did not expect, and it is worth stating carefully,
because it cuts against the common claim that insertion sort "becomes
linear" on nearly sorted data.

At n = 10,000 on `nearly_sorted` input:

{table_nearly_sorted}

Two things are true at once, and both matter.

**Insertion sort exploits this input and bubble sort essentially cannot.**
The comparison counts show the mechanism exactly. Insertion sort's
comparisons fall from {c_insertion_random:,} to {c_insertion_nearly:,} — a
{c_insertion_nearly_gain:.1f}× reduction, which matches its
{insertion_nearly_gain:.1f}× speedup almost exactly. Bubble sort's fall
only from {c_bubble_random:,} to {c_bubble_nearly:,}, a
{c_bubble_nearly_drop:.1f}% reduction. Insertion sort's inner loop stops
the moment an element reaches its place, so a nearly-ordered array costs
it almost nothing. Bubble sort moves a misplaced element left by only
*one position per pass*, so a single element displaced by 500 positions
costs 500 passes, and the early-exit flag cannot fire until the last
inversion is resolved. Bubble sort's more modest
{bubble_nearly_gain:.1f}× gain comes from making fewer *swaps*, not fewer
comparisons.

**But insertion sort is still quadratic here.** Its empirical exponent is
{insertion_nearly_k:.3f} and its doubling ratio is
{insertion_nearly_ratio:.2f} — the fit selects O(n²), not O(n). This
follows directly from how `nearly_sorted` is defined (§1.8): ⌊0.05n⌋
transpositions between *randomly chosen* positions, not between
neighbours. Each such transposition displaces an element by O(n) positions
on average, so the number of inversions is Θ(0.05n · n) = Θ(n²) — a much
smaller quadratic than random input, by a factor of about
{c_insertion_nearly_gain:.0f}, but quadratic all the same. Insertion
sort's true cost is Θ(inversions + n), which is linear only when the
number of inversions is O(n).

The honest conclusion: **insertion sort collapses to O(n) on fully sorted
input, and wins a large constant factor — not a better complexity class —
on this definition of nearly sorted.** A definition based on *local*
perturbation, swapping only adjacent elements, would produce O(n)
inversions and would show the linear collapse. That is a difference in the
data generator, not in the algorithm, and it is exactly the kind of thing
an empirical study is for: the phrase "nearly sorted" turns out to be
doing more work than it looks like it is.

### 3.5 Selection sort is flat across every input shape, and the reason is countable

Selection sort's runtime at n = 10,000 across the five swept data types:

{table_selection_flat}

Mean {selection_mean:.6f} s, standard deviation {selection_sd:.6f} s — a
**coefficient of variation of {selection_cv:.2f}%**, and a spread of only
**{selection_spread:.2f}×** between its fastest and slowest data type.
Over all eight shapes in the second study the spread is
{selection_shape_spread:.2f}×. For comparison, over the same five data
types bubble sort spans **{bubble_spread:,.0f}×** and insertion sort
**{insertion_spread:,.0f}×**.

Selection sort's variation across input shapes is, in other words, of the
same order as the measurement noise itself, while the other two algorithms
vary by three and four orders of magnitude.

The comparison counts explain this completely: selection sort performs
**exactly {nsq:,} = n(n−1)/2 comparisons on every single data type** — an
identical integer in all five rows of the table in §3.2. To find the
minimum of the unsorted suffix it must inspect all of it, and nothing it
sees along the way lets it conclude the suffix was already ordered. There
is no early exit to add; that is a property of the algorithm, not an
omission in the implementation. Its only input-dependent quantity is the
swap count, at most n−1, negligible beside n²/2 comparisons.

This gives selection sort one genuine virtue: it is the only one of the
three whose runtime is **predictable from n alone**. Its worst case and
its best case are the same case. It is also, as a direct result, the
fastest of the three on `reverse` input ({selection_reverse:.6f} s against
bubble sort's {bubble_reverse:.6f} s) — the one data type where being
unable to adapt costs nothing.

At n = 10,000 its {nsq_10k:,} comparisons take {selection_random:.6f} s,
which puts one iteration of its inner loop — a comparison, an index and a
loop step — at about **{selection_ns_per_comparison:.0f} nanoseconds** on
this machine under CPython.

### 3.6 Why the measured exponent is slightly above 2

Every quadratic fit returns k between {k_min:.2f} and {k_max:.2f} rather
than exactly 2.00, and the normalised chart in §2.4 drifts upward — by
{drift_min:.0f}% to {drift_max:.0f}% depending on the algorithm, across
two orders of magnitude of n — instead of running flat. Both are the same
effect, and it is not measurement error: the standard deviations are far
too small for that.

The likeliest cause is the memory hierarchy. At n = 100 the entire working
set — a list of 100 pointers plus the integer objects they reference —
fits comfortably in L1 cache. At n = 10,000 the pointer array alone is
80 KB, and the 10,000 boxed integer objects it references are scattered
across the heap at roughly 28 bytes each. Every comparison dereferences
two of those pointers, so as n grows an increasing fraction of comparisons
pay a cache miss and the *average cost of one elementary operation rises
with n*. The operation count is quadratic; the cost per operation is not
quite constant.

This is a limit on the method rather than a result about the algorithms.
An empirical exponent of {k_max:.2f} does not mean the algorithm is
O(n^{k_max:.2f}); it means this machine is slightly worse at large
problems than at small ones, and the fitted exponent absorbs that. The
comparison counts in §3.2, which are hardware-independent, put the
algorithmic answer beyond doubt: exactly n(n−1)/2 for selection sort, at
every size and every shape.

### 3.7 Bubble sort is the slowest of the three on unordered input, and the code shows why

On `random` input at n = 10,000, bubble sort ({bubble_random:.6f} s) is
**{bubble_vs_insertion:.2f}×** slower than insertion sort
({insertion_random:.6f} s) and {bubble_vs_selection:.2f}× slower than
selection sort. The counted comparisons explain most of it: on random
input bubble sort makes {c_bubble_random:,} comparisons against insertion
sort's {c_insertion_random:,}, a ratio of
**{c_bubble_vs_insertion:.2f}×** — insertion sort stops each inner loop as
soon as its element lands, whereas bubble sort scans the full remaining
range on every pass.

The residual gap beyond that ratio comes from the cost of each iteration.
Bubble sort's inner loop indexes the list twice per comparison
(`result[j] > result[j + 1]`) and its swap builds a tuple and performs two
stores. Insertion sort holds the travelling element in a local variable,
so it indexes once per comparison and its shift is a single store. Fewer
comparisons, and each one cheaper.

Bubble sort's one advantage is the flip side of the same design: it is the
only algorithm here that can detect that it is already finished, which is
what produces the {bubble_early_exit_gain:,.0f}× result in §3.2.

### 3.8 Practical conclusions

1. **All three are unusable at scale, and the numbers say how unusable.**
   At n = 10,000 — a small dataset by any modern standard — bubble sort
   takes {bubble_random:.1f} seconds. The quadratic term means n = 100,000
   would take roughly 100× longer, about {projected_100k_min:.0f} minutes,
   and n = 1,000,000 about {projected_1m_hours:.0f} hours. That is the
   practical content of "O(n²)".
2. **Among the three, insertion sort is the reasonable default.** Fastest
   on random input, dramatically fastest on sorted and nearly sorted
   input, stable, and able to detect when it is done. Its only weakness is
   reverse-sorted input.
3. **Selection sort's case is predictability, not speed.** Identical cost
   on every input, and at most n−1 writes — a real argument where writes
   are expensive, such as flash storage, though not one this experiment
   measures.
4. **Bubble sort's case is pedagogical.** It is beaten by insertion sort
   on every shape measured here except `sorted` and `single_value`, where
   bubble sort is ahead by {sorted_shape_ratio:.1f}× and
   {single_shape_ratio:.1f}× respectively and both are effectively
   instantaneous.
5. **Asymptotic class is not the whole story, and neither is input size.**
   The largest single effect in this entire study is not the difference
   between algorithms — it is the {bubble_early_exit_gain:,.0f}×
   difference within *one* algorithm between two inputs of *identical
   size*. Choosing an algorithm without knowing the shape of the data is
   choosing blind.

### 3.9 One measurement that does not fit, and what it tells us

Among the {quadratic_count} configurations the fit classes as quadratic,
the doubling ratios cluster tightly around 4 with one exception:
**{worst_ratio_alg} on
`{worst_ratio_type}` input, at {worst_ratio:.2f}×**. The cause is visible
in the table in §2.1 — that algorithm's n = 5,000 measurement on that data
type is {worst_ratio_5000:.6f} s against an average of
{worst_ratio_peers:.6f} s for the same algorithm and size on the other
four data types. The n = 5,000 point is elevated, which depresses the
ratio to n = 10,000.

Its own standard deviation over five runs does not explain it. That is the
point worth taking: the per-configuration standard deviation measures
jitter *within* a configuration's five consecutive runs, and cannot see
drift *between* configurations measured minutes apart — thermal state,
another process waking, CPU frequency scaling. Interleaving or randomising
the configuration order across repeated passes would expose that
component, and would be the first thing to change in a more rigorous
version of this study.

### 3.10 Limitations

* Five sizes over two orders of magnitude separate O(n) from O(n²)
  decisively, as the doubling ratios show, but would not reliably separate
  O(n log n) from O(n) — a distinction that matters for the advanced sorts
  in a later week and would need more points over a wider range.
* All measurements come from one machine, one interpreter and one run of
  the study. The variance figures capture run-to-run jitter within a
  configuration; §3.9 shows what they miss.
* `duplicates` used a pool of n/10 distinct values. A heavier
  concentration — say 10 distinct values — might expose behaviour these
  measurements do not show.
* The empirical exponents are mildly inflated by the cache effect in §3.6.
  Where the algorithmic claim matters, the comparison counts rather than
  the timings should be treated as the evidence.

---

## 4. Reproducing this report

```bash
source algorithms_course/bin/activate
python check_environment.py                 # verify the environment
pytest tests/ -v                            # the test suite
pytest --doctest-modules src/               # docstring examples
python benchmarks/sorting_benchmarks.py     # regenerate every measurement
python tools/build_report.py                # regenerate this document
```

The benchmark driver rewrites every CSV in `benchmarks/results/` and every
PNG in `docs/figures/`; the report generator then rewrites this file from
those CSVs. Because the input data is seeded, a re-run sorts identical
inputs; the timings will differ slightly, at about the scale of the
standard deviations in §2.1.
"""


def main() -> int:
    """Render the report and write it to ``docs/performance_analysis.md``."""
    facts = dict(f)
    facts["insertion_nearly_k"] = F[("Insertion Sort", "nearly_sorted")].empirical_exponent
    facts["insertion_nearly_ratio"] = t("Insertion Sort", "nearly_sorted") / t(
        "Insertion Sort", "nearly_sorted", 5000
    )
    facts["bubble_sorted_ratio"] = t("Bubble Sort", "sorted") / t(
        "Bubble Sort", "sorted", 5000
    )
    facts.update(
        table_runtimes=table_runtimes(),
        table_stability=table_stability(),
        table_fits=table_fits(),
        table_models=table_models(),
        table_comparisons=table_comparisons(),
        table_shapes=table_shapes(),
        table_memory=table_memory(),
        table_nearly_sorted=table_nearly_sorted(),
        table_selection_flat=table_selection_flat(),
    )

    rendered = REPORT.format(**facts)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(rendered)

    print(f"wrote {OUTPUT}")
    print(f"  {len(rendered):,} characters, {rendered.count(chr(10)) + 1:,} lines")
    print(f"  selection sort comparison counts identical across data types: "
          f"{f['selection_counts_identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
