"""Benchmarking framework for empirical algorithm analysis.

The framework has four jobs, matching the four things the Week 1 rubric
asks a benchmark to do:

1. **Generate test data** of eight different shapes, reproducibly, so an
   algorithm's sensitivity to input *order* can be separated from its
   sensitivity to input *size* (:meth:`AlgorithmBenchmark.generate_test_data`).
2. **Time accurately** using :func:`time.perf_counter`, discarding warm-up
   runs and reporting mean, standard deviation, minimum and maximum over
   several measured runs rather than a single stopwatch reading
   (:meth:`AlgorithmBenchmark.time_algorithm`).
3. **Visualise** the result as a log-log comparison with O(n) and O(n^2)
   reference curves (:meth:`AlgorithmBenchmark.plot_comparison`).
4. **Store and retrieve** results as CSV or JSON so a run can be inspected,
   re-plotted or compared later without re-running it
   (:meth:`AlgorithmBenchmark.export_results` /
   :meth:`AlgorithmBenchmark.load_results`).

Typical use::

    from src.sorting import bubble_sort, insertion_sort
    from src.utils.benchmark import AlgorithmBenchmark

    bench = AlgorithmBenchmark(warmup_runs=2, precision=6)
    results = bench.benchmark_suite(
        {"Bubble Sort": bubble_sort, "Insertion Sort": insertion_sort},
        sizes=[100, 500, 1000],
        data_types=["random", "sorted"],
        runs=5,
    )
    bench.plot_comparison(results, save_path="docs/figures/comparison.png")
    bench.export_results("benchmarks/results/run.csv")

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib

# Select a non-interactive backend before pyplot is imported. The benchmark
# driver runs from a terminal and from CI, where no display exists; without
# this the import can fail or block.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

__all__ = [
    "BenchmarkResult",
    "AlgorithmBenchmark",
    "CorrectnessError",
    "DATA_TYPES",
]

#: Every data shape :meth:`AlgorithmBenchmark.generate_test_data` understands.
DATA_TYPES: tuple = (
    "random",
    "sorted",
    "reverse",
    "nearly_sorted",
    "duplicates",
    "single_value",
    "mountain",
    "valley",
)


class CorrectnessError(ValueError):
    """Raised when a benchmarked algorithm returns an incorrect result.

    Subclasses :class:`ValueError` so callers that only care that *something*
    was wrong with the value can catch the broader type.
    """


@dataclass
class BenchmarkResult:
    """One timing measurement: one algorithm, one input size, several runs.

    Attributes:
        algorithm_name: Display name of the algorithm that was timed.
        input_size: Number of elements in the input list.
        average_time: Arithmetic mean of the measured runs, in seconds.
        std_deviation: Sample standard deviation of the measured runs, in
            seconds. ``0.0`` when only a single run was measured. This is
            the evidence that the timing came from repeated measurement
            rather than a single reading.
        min_time: Fastest measured run, in seconds.
        max_time: Slowest measured run, in seconds.
        memory_usage: Peak memory allocated by the call, in bytes, measured
            with :mod:`tracemalloc` in a separate un-timed run. ``0.0``
            when memory was not measured.
        metadata: Free-form provenance — data type, run counts, raw
            per-run times, interpreter version, and so on.

    Examples:
        >>> r = BenchmarkResult("Bubble Sort", 100, 0.001, 0.0001, 0.0009, 0.0012)
        >>> r.algorithm_name, r.input_size
        ('Bubble Sort', 100)
        >>> r.metadata
        {}
        >>> BenchmarkResult.from_dict(r.to_dict()) == r
        True
    """

    algorithm_name: str
    input_size: int
    average_time: float
    std_deviation: float
    min_time: float
    max_time: float
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = None  # noqa: RUF013 - signature fixed by the spec

    def __post_init__(self) -> None:
        """Normalise ``metadata`` so callers never have to guard against None."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def data_type(self) -> str:
        """Data shape this measurement was taken on, or ``'unknown'``.

        Examples:
            >>> BenchmarkResult("s", 10, 0.1, 0.0, 0.1, 0.1).data_type
            'unknown'
            >>> BenchmarkResult("s", 10, 0.1, 0.0, 0.1, 0.1,
            ...                 metadata={"data_type": "sorted"}).data_type
            'sorted'
        """
        return self.metadata.get("data_type", "unknown")

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat, CSV-friendly dictionary of this result.

        ``metadata`` is JSON-encoded into a single ``metadata`` column and
        the two fields most useful for filtering — ``data_type`` and
        ``runs`` — are also promoted to columns of their own.

        Examples:
            >>> row = BenchmarkResult("s", 10, 0.1, 0.0, 0.1, 0.1,
            ...                       metadata={"data_type": "sorted", "runs": 5}).to_dict()
            >>> row["data_type"], row["runs"], row["algorithm_name"]
            ('sorted', 5, 's')
        """
        row = asdict(self)
        row["data_type"] = self.data_type
        row["runs"] = self.metadata.get("runs")
        row["metadata"] = json.dumps(self.metadata, sort_keys=True, default=str)
        return row

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "BenchmarkResult":
        """Rebuild a result from the flat dictionary produced by :meth:`to_dict`.

        Args:
            row: A mapping with at least the six required result fields.

        Returns:
            The reconstructed :class:`BenchmarkResult`.

        Examples:
            >>> original = BenchmarkResult("s", 10, 0.1, 0.0, 0.1, 0.1,
            ...                            metadata={"data_type": "sorted"})
            >>> BenchmarkResult.from_dict(original.to_dict()) == original
            True
        """
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (TypeError, ValueError):
                meta = {"raw": meta}
        if not isinstance(meta, dict):
            meta = {}

        return cls(
            algorithm_name=str(row["algorithm_name"]),
            input_size=int(row["input_size"]),
            average_time=float(row["average_time"]),
            std_deviation=float(row["std_deviation"]),
            min_time=float(row["min_time"]),
            max_time=float(row["max_time"]),
            memory_usage=float(row.get("memory_usage", 0.0) or 0.0),
            metadata=meta,
        )


class AlgorithmBenchmark:
    """Generate data, time algorithms, plot the comparison, store the results.

    Args:
        warmup_runs: Number of un-measured executions performed before the
            timed runs. These absorb first-call costs — import side effects,
            CPU frequency ramp-up, cold caches — that would otherwise
            inflate the first measurement.
        precision: Number of decimal places reported times are rounded to.
            The default of 6 keeps microsecond resolution.
        seed: Base seed used by :meth:`benchmark_suite` when it generates
            data, so a whole suite is reproducible. Each configuration
            derives a distinct but deterministic seed from it.

    Attributes:
        results: Accumulated measurements, keyed by algorithm name. Every
            call to :meth:`time_algorithm` appends to this, which is what
            :meth:`export_results` writes out.
        verbose: When True (the default), :meth:`benchmark_suite` prints a
            progress line per configuration. Set to False to silence it.

    Examples:
        >>> bench = AlgorithmBenchmark(warmup_runs=0, precision=6)
        >>> bench.warmup_runs, bench.precision, bench.seed
        (0, 6, 42)
        >>> bench.results
        {}
    """

    #: Reference complexity models used by :meth:`analyze_complexity`.
    #: Each is fitted as ``a * g(n) + b``, so every model has the same
    #: number of free parameters and the R-squared values are comparable.
    _MODELS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: n * np.log2(np.maximum(n, 2)),
        "O(n^2)": lambda n: n ** 2,
    }

    def __init__(self, warmup_runs: int = 2, precision: int = 6, seed: int = 42) -> None:
        if warmup_runs < 0:
            raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
        if precision < 0:
            raise ValueError(f"precision must be >= 0, got {precision}")

        self.warmup_runs = warmup_runs
        self.precision = precision
        self.seed = seed
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.verbose: bool = True

    # ------------------------------------------------------------------
    # 1. Test data generation
    # ------------------------------------------------------------------
    def generate_test_data(
        self, size: int, data_type: str = "random", seed: int = None
    ) -> List[int]:
        """Build a list of ``size`` integers with a specified shape.

        The eight shapes exist to separate an algorithm's response to input
        *size* from its response to input *order*. Insertion sort, for
        instance, is quadratic on ``random`` data and linear on ``sorted``
        data of the same length.

        =============== ===========================================================
        ``data_type``   Meaning
        =============== ===========================================================
        random          Uniform random integers
        sorted          ``[0, 1, ..., size - 1]`` — ascending, best case
        reverse         ``[size - 1, ..., 1, 0]`` — descending, worst case
        nearly_sorted   Ascending, then ``floor(0.05 * size)`` random
                        transpositions (at least one when ``size >= 2``)
        duplicates      Random draws from a pool of only ``max(1, size // 10)``
                        distinct values, so most elements repeat
        single_value    Every element identical
        mountain        Strictly increasing to a single peak, then strictly
                        decreasing
        valley          Strictly decreasing to a single trough, then strictly
                        increasing
        =============== ===========================================================

        Args:
            size: Number of elements to produce. Must be non-negative.
            data_type: One of :data:`DATA_TYPES`.
            seed: Seed for this call's random number generator. Passing a
                seed makes the output reproducible; passing ``None`` leaves
                it unseeded. A dedicated :class:`random.Random` instance is
                used so the global random state is never disturbed.

        Returns:
            A list of ``size`` integers.

        Raises:
            TypeError: If ``size`` is not an integer.
            ValueError: If ``size`` is negative or ``data_type`` is unknown.

        Examples:
            >>> bench = AlgorithmBenchmark()
            >>> bench.generate_test_data(5, "sorted")
            [0, 1, 2, 3, 4]
            >>> bench.generate_test_data(5, "reverse")
            [4, 3, 2, 1, 0]
            >>> bench.generate_test_data(4, "single_value")
            [42, 42, 42, 42]
            >>> bench.generate_test_data(0, "random")
            []

            Seeding is reproducible, and two different seeds differ:

            >>> bench.generate_test_data(8, "random", seed=7) == \\
            ...     bench.generate_test_data(8, "random", seed=7)
            True

            ``mountain`` rises then falls; ``valley`` is its mirror image:

            >>> peak = bench.generate_test_data(9, "mountain", seed=1)
            >>> top = peak.index(max(peak))
            >>> peak[:top + 1] == sorted(peak[:top + 1])
            True
            >>> peak[top:] == sorted(peak[top:], reverse=True)
            True

            >>> dupes = bench.generate_test_data(100, "duplicates", seed=1)
            >>> len(set(dupes)) <= 10
            True

            >>> bench.generate_test_data(5, "spiral")
            Traceback (most recent call last):
                ...
            ValueError: unknown data_type 'spiral'; expected one of: duplicates, mountain, nearly_sorted, random, reverse, single_value, sorted, valley
        """
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"size must be an int, got {type(size).__name__}")
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        if data_type not in DATA_TYPES:
            raise ValueError(
                f"unknown data_type {data_type!r}; expected one of: "
                + ", ".join(sorted(DATA_TYPES))
            )

        rng = random.Random(seed)

        if size == 0:
            return []

        if data_type == "random":
            upper = max(size * 10, 1)
            return [rng.randint(0, upper) for _ in range(size)]

        if data_type == "sorted":
            return list(range(size))

        if data_type == "reverse":
            return list(range(size - 1, -1, -1))

        if data_type == "nearly_sorted":
            data = list(range(size))
            # floor(5% of n) transpositions, but never zero for n >= 2 -
            # otherwise small "nearly sorted" inputs would be exactly sorted
            # and the two data types would be indistinguishable.
            swaps = max(1, int(size * 0.05)) if size >= 2 else 0
            for _ in range(swaps):
                i = rng.randrange(size)
                j = rng.randrange(size)
                data[i], data[j] = data[j], data[i]
            return data

        if data_type == "duplicates":
            distinct = max(1, size // 10)
            return [rng.randrange(distinct) for _ in range(size)]

        if data_type == "single_value":
            return [42] * size

        # mountain and valley are built from the same distinct sorted pool:
        # taking every other element for each flank puts the two extreme
        # values next to each other at the join, giving a single clean peak
        # (or trough) with no plateau.
        pool = sorted(rng.sample(range(size * 10), size))
        ascending = pool[0::2]
        descending = pool[1::2][::-1]

        if data_type == "mountain":
            return ascending + descending

        # data_type == "valley"
        return descending + ascending

    # ------------------------------------------------------------------
    # 2. Timing
    # ------------------------------------------------------------------
    @staticmethod
    def _is_sorted(seq: Sequence[Any]) -> bool:
        """Return True if ``seq`` is in non-decreasing order.

        Examples:
            >>> AlgorithmBenchmark._is_sorted([1, 1, 2])
            True
            >>> AlgorithmBenchmark._is_sorted([2, 1])
            False
            >>> AlgorithmBenchmark._is_sorted([])
            True
        """
        return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))

    @staticmethod
    def _same_multiset(left: Iterable[Any], right: Iterable[Any]) -> bool:
        """Return True if both iterables hold the same elements with the same counts.

        Comparing as multisets — not as sets, and not by length — is what
        catches a sort that returns something ordered but has quietly
        dropped or duplicated an element.

        Examples:
            >>> AlgorithmBenchmark._same_multiset([1, 1, 2], [2, 1, 1])
            True
            >>> AlgorithmBenchmark._same_multiset([1, 1, 2], [1, 2, 2])
            False
            >>> AlgorithmBenchmark._same_multiset([1, 1, 2], [1, 2])
            False
        """
        try:
            return Counter(left) == Counter(right)
        except TypeError:
            # Unhashable elements: fall back to an order-insensitive
            # comparison that still counts duplicates.
            try:
                return sorted(left) == sorted(right)
            except TypeError:
                return list(left) == list(right)

    def time_algorithm(
        self,
        algorithm: Callable,
        data: List[Any],
        runs: int = 5,
        verify_correctness: bool = True,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Time one algorithm on one input and return the aggregated statistics.

        The measurement protocol:

        1. ``self.warmup_runs`` un-measured executions are performed and
           discarded.
        2. ``runs`` executions are timed individually with
           :func:`time.perf_counter`, the highest-resolution clock Python
           exposes. The defensive copy of the input is made *outside* the
           timed region, so it never enters the measurement.
        3. Mean, sample standard deviation, minimum and maximum are
           computed over those ``runs`` timings and rounded to
           ``self.precision`` decimal places.
        4. If ``verify_correctness`` is set, the output of the last timed
           run is checked — after the clock has stopped — for being sorted
           *and* for holding the same multiset of elements as the input.

        Args:
            algorithm: A callable taking the input list and returning the
                sorted list.
            data: The input list. It is copied before every call, so a
                destructive algorithm cannot corrupt later runs.
            runs: Number of measured executions. Must be at least 1;
                a standard deviation needs at least 2 to be meaningful.
            verify_correctness: Check the output is genuinely a sorted
                permutation of the input.
            measure_memory: Perform one extra, un-timed run under
                :mod:`tracemalloc` and record peak allocation in bytes.
                Off by default because tracemalloc slows execution by
                roughly an order of magnitude, which is fine for a
                one-off profile but ruinous inside a large sweep.

        Returns:
            A :class:`BenchmarkResult`. It is also appended to
            ``self.results`` under the algorithm's name, ready for
            :meth:`export_results`.

        Raises:
            TypeError: If ``algorithm`` is not callable or ``data`` is not
                a list.
            ValueError: If ``runs`` is less than 1.
            CorrectnessError: If verification is on and the output is not a
                sorted permutation of the input.

        Examples:
            >>> from src.sorting import insertion_sort
            >>> bench = AlgorithmBenchmark(warmup_runs=1)
            >>> result = bench.time_algorithm(insertion_sort, [5, 3, 1, 4], runs=3)
            >>> result.algorithm_name, result.input_size
            ('insertion_sort', 4)
            >>> result.min_time <= result.average_time <= result.max_time
            True
            >>> result.metadata["runs"], result.metadata["verified"]
            (3, True)
            >>> len(bench.results["insertion_sort"])
            1

            A broken sort is caught rather than silently timed:

            >>> bench.time_algorithm(lambda a: [0] * len(a), [3, 1, 2])
            Traceback (most recent call last):
                ...
            src.utils.benchmark.CorrectnessError: <lambda> returned a list that is not a permutation of its input (n=3)
        """
        if not callable(algorithm):
            raise TypeError(
                f"algorithm must be callable, got {type(algorithm).__name__}"
            )
        if not isinstance(data, list):
            raise TypeError(f"data must be a list, got {type(data).__name__}")
        if runs < 1:
            raise ValueError(f"runs must be >= 1, got {runs}")

        name = getattr(algorithm, "__name__", type(algorithm).__name__)

        # Warm-up: executed and thrown away. Absorbs first-call overhead.
        for _ in range(self.warmup_runs):
            algorithm(list(data))

        timings: List[float] = []
        output: Any = None
        for _ in range(runs):
            payload = list(data)  # copy OUTSIDE the timed region
            start = time.perf_counter()
            output = algorithm(payload)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        # Verification happens after the clock has stopped, so it costs the
        # measurement nothing.
        verified = False
        if verify_correctness:
            if not isinstance(output, (list, tuple)):
                raise CorrectnessError(
                    f"{name} returned {type(output).__name__}, expected a list"
                )
            if not self._is_sorted(output):
                raise CorrectnessError(
                    f"{name} returned a list that is not in ascending order "
                    f"(n={len(data)})"
                )
            if not self._same_multiset(output, data):
                raise CorrectnessError(
                    f"{name} returned a list that is not a permutation of its "
                    f"input (n={len(data)})"
                )
            verified = True

        peak_bytes = 0.0
        if measure_memory:
            tracemalloc.start()
            algorithm(list(data))
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_bytes = float(peak)

        rounded = [round(t, self.precision) for t in timings]
        result = BenchmarkResult(
            algorithm_name=name,
            input_size=len(data),
            average_time=round(statistics.fmean(timings), self.precision),
            std_deviation=round(
                statistics.stdev(timings) if len(timings) > 1 else 0.0,
                self.precision,
            ),
            min_time=round(min(timings), self.precision),
            max_time=round(max(timings), self.precision),
            memory_usage=peak_bytes,
            metadata={
                "runs": runs,
                "warmup_runs": self.warmup_runs,
                "verified": verified,
                "raw_times": rounded,
                "timer": "time.perf_counter",
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        )

        self.results.setdefault(name, []).append(result)
        return result

    # ------------------------------------------------------------------
    # 3. Running a whole suite
    # ------------------------------------------------------------------
    def benchmark_suite(
        self,
        algorithms: Dict[str, Callable],
        sizes: List[int],
        data_types: List[str] = None,
        runs: int = 5,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Time every algorithm on every size and every data type.

        For a given ``(size, data_type)`` pair the *same* generated list is
        handed to every algorithm, so the comparison between algorithms is
        never contaminated by a difference in the input. The seed for each
        pair is derived deterministically from ``self.seed``, so the whole
        sweep reproduces exactly on a re-run.

        Args:
            algorithms: Mapping of display name to callable. The display
                name — not the function's ``__name__`` — is what appears in
                the results, the charts and the CSV.
            sizes: Input sizes to sweep, in elements.
            data_types: Data shapes to sweep. Defaults to ``["random"]``.
            runs: Measured runs per configuration, passed through to
                :meth:`time_algorithm`.

        Returns:
            A dictionary mapping each display name to its list of results,
            one per ``(data_type, size)`` combination. The same results are
            accumulated in ``self.results``.

        Raises:
            TypeError: If ``algorithms`` is not a dict or ``sizes`` is not
                a list.
            ValueError: If ``algorithms`` or ``sizes`` is empty, or a data
                type is unknown.

        Examples:
            >>> from src.sorting import bubble_sort, insertion_sort
            >>> bench = AlgorithmBenchmark(warmup_runs=0)
            >>> bench.verbose = False
            >>> out = bench.benchmark_suite(
            ...     {"Bubble": bubble_sort, "Insertion": insertion_sort},
            ...     sizes=[20, 40], data_types=["random", "sorted"], runs=2)
            >>> sorted(out)
            ['Bubble', 'Insertion']
            >>> len(out["Bubble"])          # 2 sizes x 2 data types
            4
            >>> sorted({r.data_type for r in out["Bubble"]})
            ['random', 'sorted']
        """
        if not isinstance(algorithms, dict):
            raise TypeError(
                f"algorithms must be a dict of name -> callable, got "
                f"{type(algorithms).__name__}"
            )
        if not algorithms:
            raise ValueError("algorithms must contain at least one entry")
        if not isinstance(sizes, (list, tuple)):
            raise TypeError(f"sizes must be a list, got {type(sizes).__name__}")
        if not sizes:
            raise ValueError("sizes must contain at least one size")

        if data_types is None:
            data_types = ["random"]
        unknown = [d for d in data_types if d not in DATA_TYPES]
        if unknown:
            raise ValueError(
                f"unknown data_type(s) {unknown}; expected from: "
                + ", ".join(sorted(DATA_TYPES))
            )

        suite: Dict[str, List[BenchmarkResult]] = {name: [] for name in algorithms}
        total = len(data_types) * len(sizes) * len(algorithms)
        done = 0

        for data_type in data_types:
            for size in sizes:
                # One deterministic seed per (data_type, size); every
                # algorithm at this point sees byte-identical input.
                # The offset is the data type's index in DATA_TYPES, not
                # hash(data_type): Python randomises string hashing per
                # process, which would silently break reproducibility
                # between runs.
                pair_seed = self.seed + size + 1000 * DATA_TYPES.index(data_type)
                data = self.generate_test_data(size, data_type, seed=pair_seed)

                for display_name, algorithm in algorithms.items():
                    started = time.perf_counter()
                    measured = self.time_algorithm(
                        algorithm, data, runs=runs, verify_correctness=True
                    )

                    # Re-label with the caller's display name and record the
                    # provenance of this configuration.
                    labelled = replace(
                        measured,
                        algorithm_name=display_name,
                        metadata={
                            **measured.metadata,
                            "data_type": data_type,
                            "seed": pair_seed,
                        },
                    )

                    # Keep self.results consistent with what we return:
                    # drop the __name__-keyed entry time_algorithm added and
                    # store the labelled one instead.
                    raw_key = measured.algorithm_name
                    self.results[raw_key].pop()
                    if not self.results[raw_key]:
                        del self.results[raw_key]
                    self.results.setdefault(display_name, []).append(labelled)
                    suite[display_name].append(labelled)

                    done += 1
                    if self.verbose:
                        wall = time.perf_counter() - started
                        print(
                            f"  [{done:>3}/{total}] {display_name:<16} "
                            f"n={size:<6} {data_type:<14} "
                            f"mean={labelled.average_time:.6f}s  "
                            f"sd={labelled.std_deviation:.6f}s  "
                            f"(config wall {wall:6.1f}s)",
                            flush=True,
                        )

        return suite

    # ------------------------------------------------------------------
    # 4. Visualisation
    # ------------------------------------------------------------------
    def plot_comparison(
        self,
        results: Dict[str, List[BenchmarkResult]],
        title: str = "Algorithm Performance Comparison",
        log_scale: bool = True,
        save_path: str = None,
    ):
        """Plot runtime against input size, one panel per data type.

        Each algorithm is drawn as its own series with markers and error
        bars taken from the measured standard deviation. Dashed grey
        reference curves for O(n) and O(n^2) are anchored to the slowest
        series' first point, so the reader can see at a glance which growth
        curve the measurements are parallel to.

        Args:
            results: Mapping of algorithm name to its list of results, as
                returned by :meth:`benchmark_suite`.
            title: Figure title.
            log_scale: Use logarithmic scales on both axes. On log-log axes
                a power law becomes a straight line whose slope is the
                exponent, which is what makes O(n) and O(n^2) visually
                distinguishable across three orders of magnitude.
            save_path: If given, the figure is written here as PNG at
                200 dpi. Parent directories are created as needed.

        Returns:
            The :class:`matplotlib.figure.Figure`, so a caller can adjust
            or embed it further.

        Raises:
            TypeError: If ``results`` is not a dict.
            ValueError: If ``results`` is empty.
        """
        if not isinstance(results, dict):
            raise TypeError(f"results must be a dict, got {type(results).__name__}")
        if not results:
            raise ValueError("results is empty; nothing to plot")

        # Preserve the order data types were measured in rather than
        # sorting them, so the panels read in the same order as the run.
        data_types: List[str] = []
        for series in results.values():
            for r in series:
                if r.data_type not in data_types:
                    data_types.append(r.data_type)
        if not data_types:
            data_types = ["unknown"]

        ncols = min(len(data_types), 3)
        nrows = math.ceil(len(data_types) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False
        )
        flat_axes = [ax for row in axes for ax in row]

        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        colors = plt.get_cmap("tab10").colors

        for panel_index, data_type in enumerate(data_types):
            ax = flat_axes[panel_index]
            anchor_x = None
            anchor_y = None

            for series_index, (name, series) in enumerate(results.items()):
                points = sorted(
                    (r for r in series if r.data_type == data_type),
                    key=lambda r: r.input_size,
                )
                if not points:
                    continue

                xs = np.array([r.input_size for r in points], dtype=float)
                ys = np.array([r.average_time for r in points], dtype=float)
                errs = np.array([r.std_deviation for r in points], dtype=float)

                if log_scale:
                    # A log axis cannot show a zero or negative value; drop
                    # any such point rather than letting matplotlib discard
                    # it silently.
                    keep = (xs > 0) & (ys > 0)
                    xs, ys, errs = xs[keep], ys[keep], errs[keep]
                    if xs.size == 0:
                        continue

                ax.errorbar(
                    xs,
                    ys,
                    yerr=errs,
                    label=name,
                    marker=markers[series_index % len(markers)],
                    color=colors[series_index % len(colors)],
                    markersize=6,
                    linewidth=1.8,
                    capsize=3,
                    elinewidth=1.0,
                )

                # Anchor the reference curves to the slowest series so they
                # sit alongside the data instead of off the top of the plot.
                if anchor_y is None or ys[0] > anchor_y:
                    anchor_x, anchor_y = xs[0], ys[0]

            if anchor_x is not None and anchor_x > 0 and anchor_y > 0:
                all_x = np.array(
                    sorted(
                        {
                            r.input_size
                            for series in results.values()
                            for r in series
                            if r.data_type == data_type and r.input_size > 0
                        }
                    ),
                    dtype=float,
                )
                if all_x.size >= 2:
                    ref_x = np.linspace(all_x.min(), all_x.max(), 100)
                    ax.plot(
                        ref_x,
                        anchor_y * (ref_x / anchor_x),
                        "--",
                        color="0.45",
                        linewidth=1.2,
                        label="O(n) reference",
                    )
                    ax.plot(
                        ref_x,
                        anchor_y * (ref_x / anchor_x) ** 2,
                        ":",
                        color="0.15",
                        linewidth=1.2,
                        label="O(n$^2$) reference",
                    )

            if log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.set_title(f"{data_type} input", fontsize=11)
            ax.set_xlabel("Input size n (elements)")
            ax.set_ylabel("Mean runtime (seconds)")
            ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
            ax.legend(fontsize=8, loc="upper left")

        # Hide any unused panels in the grid.
        for spare in flat_axes[len(data_types):]:
            spare.axis("off")

        scale_note = "log-log axes" if log_scale else "linear axes"
        fig.suptitle(f"{title}\n({scale_note}; error bars are 1 SD over repeated runs)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if save_path:
            directory = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(directory, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            if self.verbose:
                print(f"  chart written: {save_path}", flush=True)

        return fig

    # ------------------------------------------------------------------
    # 5. Complexity analysis
    # ------------------------------------------------------------------
    def analyze_complexity(
        self, results: List[BenchmarkResult], algorithm_name: str = None
    ) -> Dict[str, Any]:
        """Fit measured runtimes against O(n), O(n log n) and O(n^2).

        Each reference model is fitted in the form ``a * g(n) + b`` by
        least squares (:func:`scipy.optimize.curve_fit`). Because every
        model has the same two free parameters, their R-squared values are
        directly comparable and the largest one identifies the best fit.

        An independent estimate is also reported: the slope of a straight
        line fitted to ``log(time)`` against ``log(n)``. On a true power
        law ``t = a n^k`` that slope *is* ``k``, so a value near 1 means
        linear behaviour and a value near 2 means quadratic behaviour
        without reference to any candidate model.

        Args:
            results: Measurements to fit. Pass a single algorithm on a
                single data type for a clean fit; if several data types are
                present their times are averaged per size and the mixture
                is reported in the ``data_types`` key.
            algorithm_name: If given, only results for this algorithm are
                used.

        Returns:
            A dictionary with keys:

            ``algorithm_name``, ``data_types``, ``n_points``, ``sizes``,
            ``times``, ``models`` (per-model ``coefficient``, ``intercept``
            and ``r_squared``), ``best_fit``, ``best_r_squared``,
            ``empirical_exponent`` and ``notes``.

        Raises:
            TypeError: If ``results`` is not a list.
            ValueError: If no results remain after filtering.

        Examples:
            >>> made_up = [BenchmarkResult("q", n, 1e-8 * n * n, 0.0, 0.0, 0.0)
            ...            for n in (100, 200, 400, 800, 1600)]
            >>> report = AlgorithmBenchmark().analyze_complexity(made_up)
            >>> report["best_fit"]
            'O(n^2)'
            >>> round(report["empirical_exponent"], 3)
            2.0
            >>> report["n_points"]
            5
        """
        if not isinstance(results, list):
            raise TypeError(f"results must be a list, got {type(results).__name__}")

        selected = [
            r
            for r in results
            if algorithm_name is None or r.algorithm_name == algorithm_name
        ]
        if not selected:
            raise ValueError(
                "no results to analyse"
                + (f" for algorithm {algorithm_name!r}" if algorithm_name else "")
            )

        # Average across whatever remains at each input size.
        by_size: Dict[int, List[float]] = {}
        for r in selected:
            by_size.setdefault(r.input_size, []).append(r.average_time)

        sizes = np.array(sorted(by_size), dtype=float)
        times = np.array(
            [statistics.fmean(by_size[int(n)]) for n in sizes], dtype=float
        )

        report: Dict[str, Any] = {
            "algorithm_name": algorithm_name or selected[0].algorithm_name,
            "data_types": sorted({r.data_type for r in selected}),
            "n_points": int(sizes.size),
            "sizes": [int(n) for n in sizes],
            "times": [float(t) for t in times],
            "models": {},
            "best_fit": None,
            "best_r_squared": float("nan"),
            "empirical_exponent": float("nan"),
            "notes": [],
        }

        if sizes.size < 3:
            report["notes"].append(
                f"only {sizes.size} distinct input size(s); a complexity fit "
                "needs at least 3 to be meaningful"
            )
            return report

        ss_tot = float(np.sum((times - times.mean()) ** 2))

        for model_name, basis in self._MODELS.items():
            def model(n, a, b, _basis=basis):
                return a * _basis(np.asarray(n, dtype=float)) + b

            try:
                # p0 scaled to the data keeps the solver well-conditioned
                # even though times are ~1e-5 and n^2 is ~1e8.
                guess_a = times[-1] / max(float(basis(sizes[-1:])[0]), 1e-12)
                popt, _ = curve_fit(model, sizes, times, p0=[guess_a, 0.0], maxfev=20000)
                predicted = model(sizes, *popt)
                ss_res = float(np.sum((times - predicted) ** 2))
                r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
                report["models"][model_name] = {
                    "coefficient": float(popt[0]),
                    "intercept": float(popt[1]),
                    "r_squared": float(r_squared),
                }
            except (RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
                report["models"][model_name] = {
                    "coefficient": float("nan"),
                    "intercept": float("nan"),
                    "r_squared": float("nan"),
                }
                report["notes"].append(f"{model_name} fit failed: {exc}")

        scored = {
            name: info["r_squared"]
            for name, info in report["models"].items()
            if not math.isnan(info["r_squared"])
        }
        if scored:
            best = max(scored, key=scored.get)
            report["best_fit"] = best
            report["best_r_squared"] = float(scored[best])

        # Independent power-law estimate from the log-log slope.
        positive = (sizes > 0) & (times > 0)
        if int(np.count_nonzero(positive)) >= 3:
            slope, _intercept = np.polyfit(
                np.log(sizes[positive]), np.log(times[positive]), 1
            )
            report["empirical_exponent"] = float(slope)
        else:
            report["notes"].append(
                "too many non-positive timings for a log-log exponent estimate; "
                "the smallest inputs may be below the timer's resolution"
            )

        return report

    # ------------------------------------------------------------------
    # 6. Storage and retrieval
    # ------------------------------------------------------------------
    def get_results(
        self, algorithm_name: str = None, data_type: str = None
    ) -> List[BenchmarkResult]:
        """Return accumulated results, optionally filtered.

        Args:
            algorithm_name: Keep only this algorithm's results.
            data_type: Keep only results measured on this data shape.

        Returns:
            A flat list of matching results, ordered by algorithm then by
            input size.

        Examples:
            >>> from src.sorting import insertion_sort
            >>> bench = AlgorithmBenchmark(warmup_runs=0)
            >>> bench.verbose = False
            >>> _ = bench.benchmark_suite({"Insertion": insertion_sort},
            ...                           sizes=[10, 20], data_types=["sorted"])
            >>> [r.input_size for r in bench.get_results("Insertion")]
            [10, 20]
            >>> bench.get_results(data_type="reverse")
            []
        """
        flat = [
            r
            for name, series in self.results.items()
            for r in series
            if algorithm_name is None or name == algorithm_name
        ]
        if data_type is not None:
            flat = [r for r in flat if r.data_type == data_type]
        return sorted(flat, key=lambda r: (r.algorithm_name, r.input_size))

    def to_dataframe(self) -> pd.DataFrame:
        """Return every accumulated result as a :class:`pandas.DataFrame`.

        Raises:
            ValueError: If no results have been recorded yet.

        Examples:
            >>> from src.sorting import insertion_sort
            >>> bench = AlgorithmBenchmark(warmup_runs=0)
            >>> bench.verbose = False
            >>> _ = bench.benchmark_suite({"Insertion": insertion_sort}, sizes=[10])
            >>> frame = bench.to_dataframe()
            >>> list(frame.columns)[:4]
            ['algorithm_name', 'input_size', 'average_time', 'std_deviation']
            >>> len(frame)
            1
        """
        rows = [r.to_dict() for r in self.get_results()]
        if not rows:
            raise ValueError("no results recorded; run a benchmark first")
        return pd.DataFrame(rows)

    def export_results(self, filename: str, format: str = "csv") -> str:
        """Write every accumulated result to disk.

        Args:
            filename: Destination path. Parent directories are created if
                they do not exist.
            format: ``'csv'`` (default) or ``'json'``.

        Returns:
            The absolute path actually written.

        Raises:
            ValueError: If no results have been recorded, or ``format`` is
                not one of the two supported values.

        Examples:
            >>> import os, tempfile
            >>> from src.sorting import insertion_sort
            >>> bench = AlgorithmBenchmark(warmup_runs=0)
            >>> bench.verbose = False
            >>> _ = bench.benchmark_suite({"Insertion": insertion_sort}, sizes=[10, 20])
            >>> target = os.path.join(tempfile.mkdtemp(), "results.csv")
            >>> written = bench.export_results(target)
            >>> os.path.exists(written)
            True
            >>> restored = AlgorithmBenchmark.load_results(written)
            >>> [r.input_size for r in restored["Insertion"]]
            [10, 20]
        """
        normalised = str(format).lower()
        if normalised not in ("csv", "json"):
            raise ValueError(f"format must be 'csv' or 'json', got {format!r}")

        frame = self.to_dataframe()  # raises if there is nothing to write
        target = os.path.abspath(filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        if normalised == "csv":
            frame.to_csv(target, index=False)
        else:
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(
                    [r.to_dict() for r in self.get_results()],
                    handle,
                    indent=2,
                    default=str,
                )

        if self.verbose:
            print(f"  results written: {target} ({len(frame)} rows)", flush=True)
        return target

    @staticmethod
    def load_results(
        filename: str, format: str = "csv"
    ) -> Dict[str, List[BenchmarkResult]]:
        """Read results back from a file written by :meth:`export_results`.

        This is the retrieval half of result storage: a completed sweep can
        be re-plotted or re-analysed later without re-running it, which
        matters when the quadratic sorts take minutes per configuration.

        Args:
            filename: Path to a CSV or JSON file previously exported.
            format: ``'csv'`` (default) or ``'json'``.

        Returns:
            The same ``{algorithm_name: [BenchmarkResult, ...]}`` shape that
            :meth:`benchmark_suite` returns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If ``format`` is unsupported.
        """
        normalised = str(format).lower()
        if normalised not in ("csv", "json"):
            raise ValueError(f"format must be 'csv' or 'json', got {format!r}")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"no such results file: {filename}")

        if normalised == "csv":
            rows = pd.read_csv(filename).to_dict(orient="records")
        else:
            with open(filename, "r", encoding="utf-8") as handle:
                rows = json.load(handle)

        restored: Dict[str, List[BenchmarkResult]] = {}
        for row in rows:
            result = BenchmarkResult.from_dict(row)
            restored.setdefault(result.algorithm_name, []).append(result)
        for series in restored.values():
            series.sort(key=lambda r: r.input_size)
        return restored
