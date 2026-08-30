"""Tests for the benchmarking framework and the shared utilities.

A benchmark that is wrong is worse than no benchmark, because its output
looks authoritative. So the framework is tested as carefully as the
algorithms it measures:

* the eight data generators really produce the shapes they claim, and the
  same seed really produces the same list;
* :meth:`~src.utils.benchmark.AlgorithmBenchmark.time_algorithm` returns a
  well-formed result whose statistics are internally consistent;
* correctness verification actually *rejects* a broken sort — three
  different ways of being broken are tried, including one that is
  perfectly ordered;
* results survive a round trip to CSV and back;
* the chart functions produce real, non-empty PNG files.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import csv
import json
import math
import os

import matplotlib.pyplot as plt
import pytest

from src.sorting import bubble_sort, insertion_sort, selection_sort
from src.utils.benchmark import (
    DATA_TYPES,
    AlgorithmBenchmark,
    BenchmarkResult,
    CorrectnessError,
)
from src.utils.testing_helpers import (
    StabilityRecord,
    broken_sort_drops_element,
    broken_sort_returns_zeros,
    broken_sort_unsorted,
    has_same_elements,
    is_sorted,
    is_valid_sort,
    random_list,
)
from src.utils.visualization import (
    apply_house_style,
    plot_complexity_fit,
    plot_data_type_sensitivity,
    plot_normalized_runtime,
)


@pytest.fixture
def bench() -> AlgorithmBenchmark:
    """A quiet benchmark with no warm-up, for fast tests."""
    instance = AlgorithmBenchmark(warmup_runs=0, precision=6)
    instance.verbose = False
    return instance


@pytest.fixture(autouse=True)
def close_figures():
    """Close every matplotlib figure a test opened, so none leak between tests."""
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
class TestDataGeneration:
    """All eight generators, checked for the property each one promises."""

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    @pytest.mark.parametrize("size", [0, 1, 2, 10, 100])
    def test_produces_requested_length_and_type(self, bench, data_type, size):
        """Every generator returns exactly ``size`` integers."""
        data = bench.generate_test_data(size, data_type, seed=1)
        assert isinstance(data, list)
        assert len(data) == size
        assert all(isinstance(value, int) for value in data)

    def test_random_is_not_ordered(self, bench):
        """Random data of a decent size should not come out sorted by accident."""
        data = bench.generate_test_data(200, "random", seed=1)
        assert not is_sorted(data)

    def test_sorted_is_ascending(self, bench):
        """``sorted`` is exactly ``[0 .. size-1]``."""
        assert bench.generate_test_data(6, "sorted") == [0, 1, 2, 3, 4, 5]
        assert is_sorted(bench.generate_test_data(500, "sorted"))

    def test_reverse_is_descending(self, bench):
        """``reverse`` is the exact mirror of ``sorted``."""
        assert bench.generate_test_data(6, "reverse") == [5, 4, 3, 2, 1, 0]
        data = bench.generate_test_data(500, "reverse")
        assert data == sorted(data, reverse=True)

    def test_nearly_sorted_is_a_permutation_of_sorted_but_not_sorted(self, bench):
        """A few transpositions of ``range(n)`` — still every element, no longer ordered."""
        size = 400
        data = bench.generate_test_data(size, "nearly_sorted", seed=5)
        assert has_same_elements(data, list(range(size)))
        assert not is_sorted(data)

        # Far closer to order than random data: count how many positions
        # are out of place.
        displaced = sum(1 for index, value in enumerate(data) if index != value)
        assert displaced <= int(size * 0.05) * 2

    def test_duplicates_has_a_small_distinct_pool(self, bench):
        """``duplicates`` draws from only ``size // 10`` distinct values."""
        size = 500
        data = bench.generate_test_data(size, "duplicates", seed=3)
        assert len(set(data)) <= max(1, size // 10)
        assert len(data) > len(set(data)), "expected repeated values"

    def test_single_value_is_constant(self, bench):
        """Every element identical."""
        data = bench.generate_test_data(50, "single_value")
        assert len(set(data)) == 1
        assert len(data) == 50

    def test_mountain_rises_then_falls(self, bench):
        """Strictly increasing up to one peak, then strictly decreasing."""
        data = bench.generate_test_data(101, "mountain", seed=9)
        peak = data.index(max(data))
        assert 0 < peak < len(data) - 1, "peak should be interior"
        assert data[: peak + 1] == sorted(data[: peak + 1])
        assert data[peak:] == sorted(data[peak:], reverse=True)
        assert len(set(data)) == len(data), "mountain values should be distinct"

    def test_valley_falls_then_rises(self, bench):
        """Strictly decreasing to one trough, then strictly increasing."""
        data = bench.generate_test_data(101, "valley", seed=9)
        trough = data.index(min(data))
        assert 0 < trough < len(data) - 1, "trough should be interior"
        assert data[: trough + 1] == sorted(data[: trough + 1], reverse=True)
        assert data[trough:] == sorted(data[trough:])

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    def test_same_seed_reproduces_exactly(self, bench, data_type):
        """The seed argument is honoured: identical seed, identical list."""
        first = bench.generate_test_data(120, data_type, seed=2026)
        second = bench.generate_test_data(120, data_type, seed=2026)
        assert first == second

    @pytest.mark.parametrize("data_type", ["random", "nearly_sorted", "duplicates"])
    def test_different_seeds_give_different_data(self, bench, data_type):
        """Different seeds diverge, so the seed is actually being used."""
        assert bench.generate_test_data(200, data_type, seed=1) != bench.generate_test_data(
            200, data_type, seed=2
        )

    def test_generation_does_not_disturb_global_random_state(self, bench):
        """A dedicated Random instance is used, so global randomness is untouched."""
        import random as global_random

        global_random.seed(1234)
        expected = [global_random.random() for _ in range(3)]

        global_random.seed(1234)
        bench.generate_test_data(1000, "random", seed=99)
        actual = [global_random.random() for _ in range(3)]

        assert actual == expected

    def test_unknown_data_type_raises_value_error(self, bench):
        """An unknown shape is rejected, and the message lists the valid ones."""
        with pytest.raises(ValueError) as caught:
            bench.generate_test_data(10, "helical")
        assert "helical" in str(caught.value)
        assert "mountain" in str(caught.value)

    def test_negative_size_raises_value_error(self, bench):
        with pytest.raises(ValueError):
            bench.generate_test_data(-1, "random")

    @pytest.mark.parametrize("bad_size", ["10", 10.0, None, True])
    def test_non_integer_size_raises_type_error(self, bench, bad_size):
        with pytest.raises(TypeError):
            bench.generate_test_data(bad_size, "random")


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------
class TestTimeAlgorithm:
    """``time_algorithm`` must return a well-formed, self-consistent result."""

    def test_returns_a_well_formed_result(self, bench):
        """Every field is populated and the statistics are mutually consistent."""
        data = bench.generate_test_data(200, "random", seed=1)
        result = bench.time_algorithm(insertion_sort, data, runs=5)

        assert isinstance(result, BenchmarkResult)
        assert result.algorithm_name == "insertion_sort"
        assert result.input_size == 200
        assert result.min_time <= result.average_time <= result.max_time
        assert result.min_time > 0
        assert result.std_deviation >= 0
        assert result.memory_usage == 0.0

    def test_metadata_records_the_measurement_protocol(self, bench):
        """The result carries enough provenance to reproduce it."""
        result = bench.time_algorithm(bubble_sort, [3, 1, 2], runs=4)
        meta = result.metadata

        assert meta["runs"] == 4
        assert meta["warmup_runs"] == 0
        assert meta["verified"] is True
        assert meta["timer"] == "time.perf_counter"
        assert len(meta["raw_times"]) == 4
        assert "python_version" in meta and "platform" in meta

    def test_standard_deviation_is_zero_for_a_single_run(self, bench):
        """One measurement has no spread; the field is 0.0, not an error."""
        result = bench.time_algorithm(insertion_sort, [3, 1, 2], runs=1)
        assert result.std_deviation == 0.0
        assert result.min_time == result.max_time == result.average_time

    def test_multiple_runs_report_a_spread(self, bench):
        """With several runs the min/max bracket really does widen."""
        data = bench.generate_test_data(800, "random", seed=4)
        result = bench.time_algorithm(insertion_sort, data, runs=7)
        assert len(result.metadata["raw_times"]) == 7
        assert result.max_time >= result.min_time

    def test_warmup_runs_are_executed_and_discarded(self):
        """Warm-up calls happen but do not appear in the reported statistics."""
        calls = {"count": 0}

        def counting_sort_wrapper(arr):
            calls["count"] += 1
            return sorted(arr)

        instance = AlgorithmBenchmark(warmup_runs=3)
        instance.verbose = False
        result = instance.time_algorithm(counting_sort_wrapper, [3, 1, 2], runs=5)

        assert calls["count"] == 3 + 5
        assert len(result.metadata["raw_times"]) == 5
        assert result.metadata["warmup_runs"] == 3

    def test_times_are_rounded_to_the_configured_precision(self):
        """``precision`` controls the number of decimal places reported."""
        instance = AlgorithmBenchmark(warmup_runs=0, precision=3)
        instance.verbose = False
        result = instance.time_algorithm(insertion_sort, random_list(300), runs=3)
        assert result.average_time == round(result.average_time, 3)
        assert all(t == round(t, 3) for t in result.metadata["raw_times"])

    def test_input_list_is_not_mutated_by_timing(self, bench):
        """The framework copies before every call, so the caller's data survives."""
        data = bench.generate_test_data(50, "random", seed=1)
        snapshot = list(data)
        bench.time_algorithm(bubble_sort, data, runs=2)
        assert data == snapshot

    def test_a_destructive_algorithm_cannot_corrupt_later_runs(self, bench):
        """Even an in-place sort sees a fresh copy on every run."""

        def in_place_sort(arr):
            arr.sort()
            return arr

        data = bench.generate_test_data(40, "reverse", seed=1)
        snapshot = list(data)
        result = bench.time_algorithm(in_place_sort, data, runs=3)
        assert data == snapshot
        assert result.metadata["verified"] is True

    def test_measure_memory_records_a_peak(self, bench):
        """With memory profiling on, a non-zero peak allocation is recorded."""
        data = bench.generate_test_data(500, "random", seed=1)
        result = bench.time_algorithm(
            insertion_sort, data, runs=2, measure_memory=True
        )
        assert result.memory_usage > 0

    def test_result_is_accumulated_for_export(self, bench):
        """Each measurement is stored, which is what ``export_results`` writes."""
        bench.time_algorithm(bubble_sort, [3, 1, 2], runs=2)
        bench.time_algorithm(bubble_sort, [4, 2, 1], runs=2)
        assert len(bench.results["bubble_sort"]) == 2

    @pytest.mark.parametrize("bad_runs", [0, -1])
    def test_runs_below_one_raises_value_error(self, bench, bad_runs):
        with pytest.raises(ValueError):
            bench.time_algorithm(bubble_sort, [3, 1, 2], runs=bad_runs)

    def test_non_callable_algorithm_raises_type_error(self, bench):
        with pytest.raises(TypeError):
            bench.time_algorithm("bubble_sort", [3, 1, 2])

    def test_non_list_data_raises_type_error(self, bench):
        with pytest.raises(TypeError):
            bench.time_algorithm(bubble_sort, (3, 1, 2))

    @pytest.mark.parametrize("bad_warmup", [-1])
    def test_negative_warmup_raises_value_error(self, bad_warmup):
        with pytest.raises(ValueError):
            AlgorithmBenchmark(warmup_runs=bad_warmup)


class TestCorrectnessVerification:
    """Verification must reject broken sorts — that is its whole purpose."""

    def test_accepts_a_correct_sort(self, bench):
        result = bench.time_algorithm(selection_sort, random_list(100), runs=2)
        assert result.metadata["verified"] is True

    def test_rejects_a_sort_that_drops_an_element(self, bench):
        """Ordered output, one element short — only the multiset check catches it."""
        data = random_list(50)
        assert is_sorted(broken_sort_drops_element(data))  # ordering alone passes
        with pytest.raises(CorrectnessError) as caught:
            bench.time_algorithm(broken_sort_drops_element, data, runs=1)
        assert "permutation" in str(caught.value)

    def test_rejects_a_sort_that_returns_the_wrong_values(self, bench):
        """Right length and ordered, wrong contents."""
        with pytest.raises(CorrectnessError):
            bench.time_algorithm(broken_sort_returns_zeros, random_list(50), runs=1)

    def test_rejects_a_sort_that_does_not_order_its_output(self, bench):
        """A permutation, but not an ordered one."""
        with pytest.raises(CorrectnessError) as caught:
            bench.time_algorithm(broken_sort_unsorted, random_list(50), runs=1)
        assert "ascending order" in str(caught.value)

    def test_rejects_a_non_list_return_value(self, bench):
        with pytest.raises(CorrectnessError):
            bench.time_algorithm(lambda arr: None, [3, 1, 2], runs=1)

    def test_verification_can_be_switched_off(self, bench):
        """With the flag off, a broken sort is timed rather than rejected."""
        result = bench.time_algorithm(
            broken_sort_returns_zeros, random_list(20), runs=1,
            verify_correctness=False,
        )
        assert result.metadata["verified"] is False

    def test_correctness_error_is_a_value_error(self):
        """Callers can catch the broader type if they prefer."""
        assert issubclass(CorrectnessError, ValueError)


# ----------------------------------------------------------------------
# Suites
# ----------------------------------------------------------------------
class TestBenchmarkSuite:
    """The sweep must cover every combination and label every result."""

    def test_covers_every_algorithm_size_and_data_type(self, bench):
        results = bench.benchmark_suite(
            {"Bubble": bubble_sort, "Insertion": insertion_sort},
            sizes=[20, 40, 60],
            data_types=["random", "sorted"],
            runs=2,
        )
        assert set(results) == {"Bubble", "Insertion"}
        for series in results.values():
            assert len(series) == 3 * 2
            assert {r.input_size for r in series} == {20, 40, 60}
            assert {r.data_type for r in series} == {"random", "sorted"}

    def test_results_carry_the_display_name_not_the_function_name(self, bench):
        """The dictionary key is what appears in charts and CSV."""
        results = bench.benchmark_suite({"My Bubble": bubble_sort}, sizes=[20], runs=1)
        assert results["My Bubble"][0].algorithm_name == "My Bubble"
        assert "bubble_sort" not in bench.results

    def test_defaults_to_random_data(self, bench):
        results = bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[20], runs=1)
        assert results["Bubble"][0].data_type == "random"

    def test_every_algorithm_sees_identical_input(self, bench):
        """Fairness: the same generated list is reused across algorithms."""
        seen = []

        def recorder_a(arr):
            seen.append(("a", list(arr)))
            return sorted(arr)

        def recorder_b(arr):
            seen.append(("b", list(arr)))
            return sorted(arr)

        bench.benchmark_suite({"A": recorder_a, "B": recorder_b}, sizes=[30], runs=1)
        inputs_a = [payload for tag, payload in seen if tag == "a"]
        inputs_b = [payload for tag, payload in seen if tag == "b"]
        assert inputs_a[0] == inputs_b[0]

    def test_suite_is_reproducible_across_instances(self):
        """Two independent benchmarks with the same seed measure the same data."""
        seeds = []
        for _ in range(2):
            instance = AlgorithmBenchmark(warmup_runs=0, seed=7)
            instance.verbose = False
            out = instance.benchmark_suite(
                {"Bubble": bubble_sort}, sizes=[30, 60],
                data_types=["random", "duplicates"], runs=1,
            )
            seeds.append([r.metadata["seed"] for r in out["Bubble"]])
        assert seeds[0] == seeds[1]
        assert len(set(seeds[0])) == 4, "each configuration needs its own seed"

    def test_empty_algorithms_raises_value_error(self, bench):
        with pytest.raises(ValueError):
            bench.benchmark_suite({}, sizes=[10])

    def test_empty_sizes_raises_value_error(self, bench):
        with pytest.raises(ValueError):
            bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[])

    def test_unknown_data_type_raises_value_error(self, bench):
        with pytest.raises(ValueError):
            bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[10], data_types=["nope"])

    def test_non_dict_algorithms_raises_type_error(self, bench):
        with pytest.raises(TypeError):
            bench.benchmark_suite([bubble_sort], sizes=[10])


# ----------------------------------------------------------------------
# Complexity analysis
# ----------------------------------------------------------------------
class TestComplexityAnalysis:
    """The fit must identify the right model on data of known shape."""

    @staticmethod
    def _synthetic(name, shape):
        sizes = [100, 200, 400, 800, 1600, 3200]
        return [
            BenchmarkResult(name, n, shape(n), 0.0, shape(n), shape(n),
                            metadata={"data_type": "synthetic"})
            for n in sizes
        ]

    def test_identifies_quadratic_data(self, bench):
        report = bench.analyze_complexity(self._synthetic("q", lambda n: 1e-8 * n * n))
        assert report["best_fit"] == "O(n^2)"
        assert report["best_r_squared"] > 0.999
        assert math.isclose(report["empirical_exponent"], 2.0, abs_tol=0.05)

    def test_identifies_linear_data(self, bench):
        report = bench.analyze_complexity(self._synthetic("l", lambda n: 3e-6 * n))
        assert report["best_fit"] == "O(n)"
        assert math.isclose(report["empirical_exponent"], 1.0, abs_tol=0.05)

    def test_identifies_linearithmic_data(self, bench):
        report = bench.analyze_complexity(
            self._synthetic("nl", lambda n: 1e-7 * n * math.log2(n))
        )
        assert report["best_fit"] == "O(n log n)"
        assert 1.0 < report["empirical_exponent"] < 1.4

    def test_reports_all_three_models_with_r_squared(self, bench):
        report = bench.analyze_complexity(self._synthetic("q", lambda n: 1e-8 * n * n))
        assert set(report["models"]) == {"O(n)", "O(n log n)", "O(n^2)"}
        for info in report["models"].values():
            assert "coefficient" in info and "r_squared" in info

    def test_filters_by_algorithm_name(self, bench):
        mixed = self._synthetic("fast", lambda n: 1e-6 * n) + self._synthetic(
            "slow", lambda n: 1e-8 * n * n
        )
        report = bench.analyze_complexity(mixed, algorithm_name="fast")
        assert report["algorithm_name"] == "fast"
        assert report["n_points"] == 6
        assert report["best_fit"] == "O(n)"

    def test_too_few_points_is_reported_not_crashed(self, bench):
        two = [BenchmarkResult("x", n, 1e-6 * n, 0.0, 0.0, 0.0) for n in (10, 20)]
        report = bench.analyze_complexity(two)
        assert report["best_fit"] is None
        assert report["notes"]

    def test_unknown_algorithm_name_raises_value_error(self, bench):
        with pytest.raises(ValueError):
            bench.analyze_complexity(self._synthetic("q", lambda n: n), "nobody")

    def test_non_list_results_raises_type_error(self, bench):
        with pytest.raises(TypeError):
            bench.analyze_complexity("not a list")

    def test_analyses_real_measurements(self, bench):
        """End to end on genuine timings, not synthetic ones."""
        results = bench.benchmark_suite(
            {"Bubble": bubble_sort}, sizes=[100, 200, 400, 800],
            data_types=["random"], runs=3,
        )
        report = bench.analyze_complexity(results["Bubble"], "Bubble")
        assert report["n_points"] == 4
        assert report["best_fit"] in {"O(n)", "O(n log n)", "O(n^2)"}
        assert report["data_types"] == ["random"]


# ----------------------------------------------------------------------
# Storage and retrieval
# ----------------------------------------------------------------------
class TestResultStorage:
    """Results must survive a round trip to disk and back."""

    def test_export_writes_a_readable_csv(self, bench, tmp_path):
        """The CSV has a header and one row per measurement."""
        bench.benchmark_suite(
            {"Bubble": bubble_sort, "Insertion": insertion_sort},
            sizes=[20, 40], data_types=["random", "sorted"], runs=2,
        )
        target = tmp_path / "results.csv"
        written = bench.export_results(str(target))

        assert os.path.exists(written)
        with open(written, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2 * 2 * 2
        for column in (
            "algorithm_name", "input_size", "average_time", "std_deviation",
            "min_time", "max_time", "memory_usage", "data_type", "runs", "metadata",
        ):
            assert column in rows[0], f"missing column {column}"
        assert {row["algorithm_name"] for row in rows} == {"Bubble", "Insertion"}
        assert {row["data_type"] for row in rows} == {"random", "sorted"}
        assert float(rows[0]["average_time"]) >= 0

    def test_csv_round_trip_preserves_the_measurements(self, bench, tmp_path):
        bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[20, 40], runs=2)
        target = str(tmp_path / "round_trip.csv")
        bench.export_results(target)

        restored = AlgorithmBenchmark.load_results(target)
        original = bench.get_results("Bubble")

        assert list(restored) == ["Bubble"]
        assert len(restored["Bubble"]) == len(original)
        for before, after in zip(original, restored["Bubble"]):
            assert after.input_size == before.input_size
            assert after.average_time == pytest.approx(before.average_time)
            assert after.std_deviation == pytest.approx(before.std_deviation)
            assert after.data_type == before.data_type

    def test_export_json(self, bench, tmp_path):
        bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[20], runs=1)
        target = str(tmp_path / "results.json")
        bench.export_results(target, format="json")

        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert isinstance(payload, list) and payload[0]["algorithm_name"] == "Bubble"

        restored = AlgorithmBenchmark.load_results(target, format="json")
        assert restored["Bubble"][0].input_size == 20

    def test_export_creates_missing_directories(self, bench, tmp_path):
        bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[20], runs=1)
        target = str(tmp_path / "deep" / "nested" / "results.csv")
        assert os.path.exists(bench.export_results(target))

    def test_export_without_results_raises_value_error(self, bench, tmp_path):
        with pytest.raises(ValueError):
            bench.export_results(str(tmp_path / "empty.csv"))

    def test_unsupported_format_raises_value_error(self, bench, tmp_path):
        bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[20], runs=1)
        with pytest.raises(ValueError):
            bench.export_results(str(tmp_path / "results.xml"), format="xml")

    def test_loading_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AlgorithmBenchmark.load_results(str(tmp_path / "absent.csv"))

    def test_get_results_filters(self, bench):
        bench.benchmark_suite(
            {"Bubble": bubble_sort, "Insertion": insertion_sort},
            sizes=[20, 40], data_types=["random", "sorted"], runs=1,
        )
        assert len(bench.get_results()) == 8
        assert len(bench.get_results("Bubble")) == 4
        assert len(bench.get_results(data_type="sorted")) == 4
        assert len(bench.get_results("Bubble", "sorted")) == 2
        assert bench.get_results("Nobody") == []

    def test_to_dataframe(self, bench):
        bench.benchmark_suite({"Bubble": bubble_sort}, sizes=[20, 40], runs=1)
        frame = bench.to_dataframe()
        assert len(frame) == 2
        assert set(frame["algorithm_name"]) == {"Bubble"}
        assert sorted(frame["input_size"]) == [20, 40]

    def test_benchmark_result_dict_round_trip(self):
        original = BenchmarkResult(
            "Bubble", 100, 0.5, 0.01, 0.49, 0.52, 1024.0,
            metadata={"data_type": "random", "runs": 5},
        )
        assert BenchmarkResult.from_dict(original.to_dict()) == original

    def test_metadata_defaults_to_an_empty_dict(self):
        assert BenchmarkResult("x", 1, 0.0, 0.0, 0.0, 0.0).metadata == {}

    def test_data_type_property_defaults_to_unknown(self):
        assert BenchmarkResult("x", 1, 0.0, 0.0, 0.0, 0.0).data_type == "unknown"


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
class TestVisualization:
    """Every chart function must write a real PNG."""

    @pytest.fixture
    def measured(self, bench):
        return bench.benchmark_suite(
            {"Bubble": bubble_sort, "Insertion": insertion_sort},
            sizes=[50, 100, 200, 400],
            data_types=["random", "sorted"],
            runs=2,
        )

    def test_plot_comparison_writes_a_png(self, bench, measured, tmp_path):
        target = tmp_path / "comparison.png"
        figure = bench.plot_comparison(measured, save_path=str(target))
        assert target.exists() and target.stat().st_size > 5000
        assert len(figure.axes) == 2, "one panel per data type"

    def test_plot_comparison_linear_scale(self, bench, measured, tmp_path):
        target = tmp_path / "linear.png"
        bench.plot_comparison(measured, log_scale=False, save_path=str(target))
        assert target.exists()

    def test_plot_comparison_without_save_path_returns_a_figure(self, bench, measured):
        assert bench.plot_comparison(measured) is not None

    def test_plot_comparison_rejects_empty_results(self, bench):
        with pytest.raises(ValueError):
            bench.plot_comparison({})

    def test_plot_comparison_rejects_non_dict(self, bench):
        with pytest.raises(TypeError):
            bench.plot_comparison([])

    def test_data_type_sensitivity_chart(self, measured, tmp_path):
        target = tmp_path / "sensitivity.png"
        plot_data_type_sensitivity(measured, 400, save_path=str(target))
        assert target.exists() and target.stat().st_size > 5000

    def test_data_type_sensitivity_rejects_a_size_that_was_not_measured(self, measured):
        with pytest.raises(ValueError):
            plot_data_type_sensitivity(measured, 999_999)

    def test_complexity_fit_chart(self, bench, measured, tmp_path):
        analyses = [
            bench.analyze_complexity(
                [r for r in measured[name] if r.data_type == "random"], name
            )
            for name in measured
        ]
        target = tmp_path / "fits.png"
        figure = plot_complexity_fit(analyses, save_path=str(target))
        assert target.exists()
        assert len(figure.axes) == 2

    def test_complexity_fit_rejects_empty_input(self):
        with pytest.raises(ValueError):
            plot_complexity_fit([])

    def test_normalized_runtime_chart(self, measured, tmp_path):
        target = tmp_path / "normalized.png"
        plot_normalized_runtime(measured, "random", save_path=str(target))
        assert target.exists()

    def test_normalized_runtime_rejects_an_unmeasured_data_type(self, measured):
        with pytest.raises(ValueError):
            plot_normalized_runtime(measured, "valley")

    def test_house_style_applies_without_error(self):
        assert apply_house_style() is None


# ----------------------------------------------------------------------
# Testing helpers
# ----------------------------------------------------------------------
class TestTestingHelpers:
    """The predicates the whole suite leans on are themselves tested."""

    def test_is_sorted(self):
        assert is_sorted([]) and is_sorted([1]) and is_sorted([1, 1, 2])
        assert not is_sorted([2, 1])

    def test_has_same_elements_is_a_multiset_comparison(self):
        assert has_same_elements([1, 1, 2], [2, 1, 1])
        assert not has_same_elements([1, 1, 2], [1, 2, 2])
        assert not has_same_elements([1, 2], [1, 2, 3])
        # The distinction a set comparison would miss:
        assert set([1, 1, 2]) == set([1, 2, 2])

    def test_has_same_elements_handles_unhashable_elements(self):
        assert has_same_elements([[1], [2]], [[2], [1]])

    def test_is_valid_sort_requires_both_properties(self):
        assert is_valid_sort([3, 1, 2], [1, 2, 3])
        assert not is_valid_sort([3, 1, 2], [3, 2, 1])
        assert not is_valid_sort([3, 1, 2], [1, 2])
        assert not is_valid_sort([3, 1, 2], [0, 0, 0])

    def test_random_list_is_reproducible_and_bounded(self):
        assert random_list(50, seed=5) == random_list(50, seed=5)
        assert random_list(50, seed=5) != random_list(50, seed=6)
        assert all(-10 <= v <= 10 for v in random_list(50, low=-10, high=10))

    def test_stability_record_orders_on_the_key_alone(self):
        """A tie must be a real tie — this is what makes stability observable."""
        first, second = StabilityRecord(1, 0), StabilityRecord(1, 1)
        assert not (first < second) and not (second < first)
        assert not (first > second) and not (second > first)
        assert first <= second and first >= second
        assert first != second
        assert StabilityRecord(1, 9) < StabilityRecord(2, 0)

    def test_broken_sorts_are_actually_broken(self):
        data = [3, 1, 2]
        assert not is_valid_sort(data, broken_sort_drops_element(data))
        assert not is_valid_sort(data, broken_sort_returns_zeros(data))
        assert not is_valid_sort(data, broken_sort_unsorted(data))
