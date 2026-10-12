"""Tests for benchmarks/week7_randomized_benchmark.py.

These check that the harness runs, writes what the report is built from, and
is reproducible. They never check that a timing lands on a particular number,
because timings depend on the machine.

The reproducibility claim is tested directly: two smoke runs with the same
``--seed`` must agree on every column that does not hold a wall-clock time.
That holds only because every random draw comes from a seeded generator and
every item hash is blake2b rather than the per-process-salted ``hash()``.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

import benchmarks.week7_randomized_benchmark as week7

REQUIRED_FIGURES = [
    "quicksort_comparison.png",
    "selection_runtime.png",
    "primality_accuracy.png",
    "bloom_filter_fp_rate.png",
    "minhash_similarity_plot.png",
]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
RATE_METRICS = {"false_positive_rate", "success_rate", "within_4n_fraction"}


def read_table(directory: Path) -> List[Dict[str, str]]:
    with open(directory / "summary_table.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("week7_smoke_a")
    assert week7.main(["--smoke", "--seed", "42", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def second_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("week7_smoke_b")
    assert week7.main(["--smoke", "--seed", "42", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def table(first_run: Path) -> List[Dict[str, str]]:
    return read_table(first_run)


class TestOutputsWritten:
    @pytest.mark.parametrize("name", REQUIRED_FIGURES)
    def test_every_required_figure_is_a_non_empty_png(self, first_run: Path, name: str) -> None:
        data = (first_run / name).read_bytes()
        assert len(data) > 1000 and data.startswith(PNG_MAGIC)

    def test_the_summary_table_is_written_and_non_empty(self, first_run: Path) -> None:
        path = first_run / "summary_table.csv"
        assert path.is_file() and path.stat().st_size > 0

    def test_the_distribution_csvs_are_written(self, first_run: Path) -> None:
        for name in ("week7_comparison_counts.csv", "week7_selection_ratios.csv"):
            with open(first_run / name, newline="", encoding="utf-8") as handle:
                assert len(list(csv.DictReader(handle))) > 0


class TestCsvContract:
    def test_header_is_exactly_the_fixed_column_list(self, first_run: Path) -> None:
        with open(first_run / "summary_table.csv", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header == week7.COLUMNS
        assert header == [
            "algorithm", "variant", "category", "experiment", "input_kind", "n",
            "param_name", "param_value", "runs", "mean_time_s", "var_time_s2", "std_time_s",
            "min_time_s", "max_time_s", "metric_name", "metric_value", "metric_ci_low",
            "metric_ci_high", "theoretical_value", "seed", "measurement",
        ]

    def test_every_measurement_is_measured_or_projected(self, table: List[Dict[str, str]]) -> None:
        assert {r["measurement"] for r in table} <= {"measured", "projected"}
        assert any(r["measurement"] == "projected" for r in table)

    def test_every_category_is_las_vegas_or_monte_carlo(self, table: List[Dict[str, str]]) -> None:
        assert {r["category"] for r in table} == {"las_vegas", "monte_carlo"}

    def test_every_rate_is_a_probability(self, table: List[Dict[str, str]]) -> None:
        rates = [r for r in table if r["metric_name"] in RATE_METRICS]
        assert rates
        for r in rates:
            value = float(r["metric_value"])
            assert 0.0 <= value <= 1.0
            if r["metric_ci_low"] != "":
                assert 0.0 <= float(r["metric_ci_low"]) <= value <= float(r["metric_ci_high"]) <= 1.0

    def test_las_vegas_selection_is_always_correct(self, table: List[Dict[str, str]]) -> None:
        success = [r for r in table if r["metric_name"] == "success_rate"]
        assert success and all(float(r["metric_value"]) == 1.0 for r in success)

    def test_sorted_input_costs_exactly_n_n_minus_1_over_2(self, table: List[Dict[str, str]]) -> None:
        rows = [r for r in table if r["experiment"] == "sorted_input"
                and r["variant"] == "deterministic_last"]
        assert rows
        for r in rows:
            n = int(r["n"])
            assert int(r["metric_value"]) == n * (n - 1) // 2

    def test_projected_rows_carry_no_spread(self, table: List[Dict[str, str]]) -> None:
        for r in table:
            if r["measurement"] == "projected":
                assert r["std_time_s"] == "" and r["var_time_s2"] == ""
                assert r["runs"] == "0"

    def test_the_fixed_base_attack_is_recorded(self, table: List[Dict[str, str]]) -> None:
        rows = [r for r in table if r["algorithm"] == "miller_rabin_fixed_bases"]
        assert len(rows) == 1
        assert int(rows[0]["n"]) == 3_215_031_751 and rows[0]["metric_value"] == "1"


class TestReproducibility:
    def test_same_seed_gives_identical_non_timing_columns(
            self, first_run: Path, second_run: Path) -> None:
        a, b = read_table(first_run), read_table(second_run)
        assert len(a) == len(b)
        for ra, rb in zip(a, b):
            for column in week7.COLUMNS:
                if column not in week7.TIMING_COLUMNS:
                    assert ra[column] == rb[column], (column, ra, rb)

    def test_timing_columns_are_the_only_ones_excluded(self) -> None:
        assert set(week7.TIMING_COLUMNS) == {
            "mean_time_s", "var_time_s2", "std_time_s", "min_time_s", "max_time_s"}

    def test_every_row_records_its_seed(self, table: List[Dict[str, str]]) -> None:
        assert all(r["seed"] != "" for r in table)


class TestHarnessPieces:
    def test_simulated_pairs_have_exact_jaccard_values(self) -> None:
        values = [round(week7.jaccard(a, b), 12) for _, _, a, b in week7.simulated_pairs()]
        assert values == [0.1, 0.25, 0.5, 0.75, 0.9]

    def test_text_pairs_are_seeded(self) -> None:
        first = week7.text_pairs(42)
        again = week7.text_pairs(42)
        assert [p[3] for p in first] == [p[3] for p in again]
        assert len(first) == 4

    def test_smoke_config_is_smaller_than_full(self) -> None:
        assert max(week7.SMOKE.qs_random_sizes) < max(week7.FULL.qs_random_sizes)
        assert week7.SMOKE.prim_trials < week7.FULL.prim_trials
        assert week7.SMOKE.bloom_queries < week7.FULL.bloom_queries
        assert week7.FULL.qs_random_sizes[0] == 100 and week7.FULL.qs_random_sizes[-1] == 100_000
        assert week7.FULL.minhash_ks == [16, 32, 64, 128, 256, 512]
        assert week7.FULL.prim_ks == [1, 5, 10, 20]
