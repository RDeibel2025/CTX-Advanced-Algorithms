"""Tests for benchmarks/week6_dp_advanced_benchmark.py.

These check that the harness runs and writes what the report is built
from, never that a timing lands on a particular number. Timings depend on
the machine; a test that asserted one would fail on the next laptop
without anything being wrong.

What is checked instead:

* The smoke parameter set runs end to end into a temporary directory and
  writes all four required figures and ``comparison_table.csv``.
* The CSV header is exactly the fixed column list the report reads back,
  every row's ``measurement`` is ``measured`` or ``projected``, and every
  speedup that was written is a positive number with its baseline named.
* The pieces the harness is built from - the repetition policy, the
  speedup rule, the theory table - behave as documented.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

import benchmarks.week6_dp_advanced_benchmark as week6

REQUIRED_FIGURES = [
    "mcm_performance.png",
    "floyd_warshall_scaling.png",
    "knapsack_space_comparison.png",
    "tsp_bitmask_runtime.png",
]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the smoke benchmark once and share its output directory."""
    out = tmp_path_factory.mktemp("week6_smoke")
    assert week6.main(["--smoke", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def table(smoke_run: Path) -> List[Dict[str, str]]:
    with open(smoke_run / "comparison_table.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestOutputsWritten:
    @pytest.mark.parametrize("name", REQUIRED_FIGURES)
    def test_every_required_figure_is_a_non_empty_png(self, smoke_run: Path, name: str) -> None:
        path = smoke_run / name
        assert path.is_file(), f"{name} was not written"
        data = path.read_bytes()
        assert len(data) > 1000 and data.startswith(PNG_MAGIC)

    def test_the_comparison_table_is_written_and_non_empty(self, smoke_run: Path) -> None:
        path = smoke_run / "comparison_table.csv"
        assert path.is_file() and path.stat().st_size > 0

    def test_the_negative_edge_record_is_written(self, smoke_run: Path) -> None:
        with open(smoke_run / "week6_negative_edges.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 5
        assert all(row["all_pairs_dijkstra"].startswith("ValueError") for row in rows)


class TestCsvContract:
    def test_header_is_exactly_the_fixed_column_list(self, smoke_run: Path) -> None:
        with open(smoke_run / "comparison_table.csv", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        assert header == week6.COLUMNS
        assert header == [
            "problem", "variant", "n", "secondary_param", "mean_time_s", "std_time_s",
            "min_time_s", "max_time_s", "runs", "peak_kib", "theoretical_time",
            "theoretical_space", "speedup_vs_baseline", "baseline_variant", "measurement",
        ]

    def test_every_measurement_is_measured_or_projected(self, table: List[Dict[str, str]]) -> None:
        assert table, "the table has no rows"
        assert {row["measurement"] for row in table} <= {"measured", "projected"}

    def test_all_four_problems_are_present(self, table: List[Dict[str, str]]) -> None:
        assert {row["problem"] for row in table} == {
            "knapsack", "mcm", "floyd_warshall", "tsp"}

    def test_every_written_speedup_is_positive_and_names_its_baseline(
            self, table: List[Dict[str, str]]) -> None:
        written = [row for row in table if row["speedup_vs_baseline"] != ""]
        assert written, "no speedups were written at all"
        for row in written:
            assert float(row["speedup_vs_baseline"]) > 0
            assert row["baseline_variant"] == week6.BASELINE[(row["problem"], row["variant"])]

    def test_baseline_rows_carry_no_speedup(self, table: List[Dict[str, str]]) -> None:
        baselines = set(week6.BASELINE.values())
        for row in table:
            if row["variant"] in baselines:
                assert row["speedup_vs_baseline"] == ""

    def test_timings_are_positive_and_ordered(self, table: List[Dict[str, str]]) -> None:
        for row in table:
            low, mean, high = (float(row[k]) for k in ("min_time_s", "mean_time_s", "max_time_s"))
            assert 0 < low <= mean * (1 + 1e-9) and mean <= high * (1 + 1e-9)
            assert int(row["runs"]) >= 1

    def test_every_row_states_its_theoretical_bounds(self, table: List[Dict[str, str]]) -> None:
        for row in table:
            assert row["theoretical_time"].startswith("O(")
            assert row["theoretical_space"].startswith("O(")

    def test_no_row_is_duplicated(self, table: List[Dict[str, str]]) -> None:
        keys = [(r["problem"], r["variant"], r["n"], r["secondary_param"]) for r in table]
        assert len(keys) == len(set(keys))


class TestHarnessPieces:
    @pytest.mark.parametrize("pilot, expected", [
        (0.001, (5, 2)), (0.049, (5, 2)), (0.05, (5, 1)), (0.49, (5, 1)),
        (0.5, (3, 0)), (19.9, (3, 0)), (20.0, (1, 0)), (90.0, (1, 0)),
    ])
    def test_repeats_follow_the_pilot_cost(self, pilot: float, expected: tuple) -> None:
        assert week6.repeats_for(pilot) == expected

    def test_add_speedups_only_fills_ratios_whose_baseline_ran(self) -> None:
        stats = {"mean": 0.2, "std": 0.0, "min": 0.2, "max": 0.2, "runs": 5}
        slow = {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "runs": 5}
        rows = [
            week6.make_row("tsp", "brute_force", 8, "", slow, ""),
            week6.make_row("tsp", "bitmask", 8, "", stats, ""),
            week6.make_row("tsp", "bitmask", 15, "", stats, ""),
        ]
        week6.add_speedups(rows)
        assert rows[1]["speedup_vs_baseline"] == "5.000"
        assert rows[1]["baseline_variant"] == "brute_force"
        assert rows[2]["speedup_vs_baseline"] == "" and rows[2]["baseline_variant"] == ""
        assert rows[0]["speedup_vs_baseline"] == ""

    def test_every_variant_has_theoretical_bounds(self) -> None:
        variants = {
            "knapsack": ["standard_2d", "space_optimized_1d"],
            "mcm": ["recursive", "memoized", "bottom_up"],
            "floyd_warshall": ["floyd_warshall", "floyd_warshall_3d", "all_pairs_dijkstra"],
            "tsp": ["bitmask", "brute_force"],
        }
        for problem, names in variants.items():
            for name in names:
                assert (problem, name) in week6.THEORY

    def test_untraced_runs_leave_peak_blank(self) -> None:
        assert week6.trace(lambda: None, week6.TRACE_LIMIT_S) == ""
        assert week6.trace(lambda: [0] * 1000, 0.0) > 0

    def test_smoke_config_is_smaller_than_full(self) -> None:
        assert max(week6.SMOKE.fw_scaling_sizes) < max(week6.FULL.fw_scaling_sizes)
        assert max(week6.SMOKE.tsp_bitmask_sizes) < max(week6.FULL.tsp_bitmask_sizes)
        assert max(week6.SMOKE.tsp_brute_sizes) < max(week6.FULL.tsp_brute_sizes)
        assert max(week6.SMOKE.mcm_dp_sizes) < max(week6.FULL.mcm_dp_sizes)
        assert week6.FULL.fw_scaling_sizes == [50, 100, 200, 500]
