"""Week 9 tests: facility location anchors (spec section 10.5).

The anchor values were checked by hand (spec section 10.5); the seeded
random instances are checked against an independent brute force over every
nonempty set of open facilities, written here without the module's own
``total_cost``.
"""

from __future__ import annotations

import itertools
import math

import pytest

from src.approximation.facility_location_greedy import (
    facility_location_exact,
    facility_location_greedy,
    random_instance,
    total_cost,
)
from src.utils.random_utils import derive_seed, make_rng

ANCHOR_COSTS = [4, 4, 4]
ANCHOR_DIST = [[1, 5, 6], [2, 4, 6], [6, 1, 5], [6, 2, 4], [5, 6, 1]]


def _brute_force(open_costs, dist):
    """Minimum cost over every nonempty open set, computed independently."""
    n_f = len(open_costs)
    best = math.inf
    for k in range(1, n_f + 1):
        for subset in itertools.combinations(range(n_f), k):
            cost = sum(open_costs[f] for f in subset)
            cost += sum(min(row[f] for f in subset) for row in dist)
            best = min(best, cost)
    return best


def _random_cases(count=20):
    """Seeded (n_clients, n_facilities, seed) triples with F <= 8."""
    rng = make_rng(derive_seed(42, "week9", "facility_location", "cases"))
    return [
        (rng.randint(3, 12), rng.randint(1, 8), derive_seed(42, "facility_location", i))
        for i in range(count)
    ]


class TestFacilityLocation:
    def test_single_facility_costs(self):
        assert total_cost(ANCHOR_COSTS, ANCHOR_DIST, {0}) == pytest.approx(24)
        assert total_cost(ANCHOR_COSTS, ANCHOR_DIST, {1}) == pytest.approx(22)
        assert total_cost(ANCHOR_COSTS, ANCHOR_DIST, {2}) == pytest.approx(26)

    def test_anchor_optimum_cost(self):
        assert total_cost(ANCHOR_COSTS, ANCHOR_DIST, {0, 1}) == pytest.approx(19)

    def test_exact_anchor(self):
        open_set, cost = facility_location_exact(ANCHOR_COSTS, ANCHOR_DIST)
        assert cost == pytest.approx(19)
        assert open_set == {0, 1}

    def test_greedy_anchor(self):
        open_set, cost = facility_location_greedy(ANCHOR_COSTS, ANCHOR_DIST)
        assert open_set == {0, 1}
        assert cost == pytest.approx(19)

    @pytest.mark.parametrize("n_clients,n_facilities,seed", _random_cases())
    def test_exact_matches_brute_force_and_greedy_is_no_better(
        self, n_clients, n_facilities, seed
    ):
        open_costs, dist = random_instance(n_clients, n_facilities, seed)
        brute = _brute_force(open_costs, dist)

        exact_set, exact_cost = facility_location_exact(open_costs, dist)
        assert exact_cost == pytest.approx(brute, rel=1e-9, abs=1e-9)
        assert exact_set and exact_set <= set(range(n_facilities))
        reported = sum(open_costs[f] for f in exact_set) + sum(
            min(row[f] for f in exact_set) for row in dist
        )
        assert reported == pytest.approx(exact_cost, rel=1e-9, abs=1e-9)

        greedy_set, greedy_cost = facility_location_greedy(open_costs, dist)
        assert greedy_set and greedy_set <= set(range(n_facilities))
        assert greedy_cost >= exact_cost - 1e-9

    def test_random_instance_reproducible(self):
        assert random_instance(9, 5, 123) == random_instance(9, 5, 123)
        assert random_instance(9, 5, 123) != random_instance(9, 5, 124)

    def test_random_instance_shape(self):
        open_costs, dist = random_instance(7, 4, 2026)
        assert len(open_costs) == 4
        assert len(dist) == 7
        assert all(len(row) == 4 for row in dist)
        assert all(20.0 <= v <= 60.0 for v in open_costs)
        assert all(0.0 <= d <= 100.0 * math.sqrt(2) for row in dist for d in row)


# ----------------------------------------------------------------------
# Benchmark smoke test (spec section 10.5, second bullet)
# ----------------------------------------------------------------------
import csv as _csv  # noqa: E402
import os as _os  # noqa: E402
import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_BENCHMARK = _os.path.join(_REPO_ROOT, "benchmarks", "week9_approx_benchmark.py")
_EXPECTED_HEADER = [
    "problem", "method", "kind", "n", "param", "runs", "mean_cost", "std_cost", "min_cost",
    "max_cost", "optimal_value", "lower_bound", "lower_bound_kind", "ratio_to_opt",
    "ratio_to_lower_bound", "proven_bound", "mean_time_s", "seed",
]
_FIGURES = ["approx_ratios.png", "cost_vs_optimal.png", "randomized_maxcut_results.png"]


def _run_smoke(out_dir: str) -> None:
    """Run the benchmark's smoke mode into ``out_dir``; fail with its output on error."""
    completed = _subprocess.run(
        [_sys.executable, _BENCHMARK, "--smoke", "--out", out_dir, "--seed", "42"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]


def _rows(out_dir: str) -> tuple:
    with open(_os.path.join(out_dir, "comparison_table.csv"), newline="") as handle:
        reader = _csv.reader(handle)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke runs with the same master seed, in separate directories."""
    first = str(tmp_path_factory.mktemp("week9_smoke_a"))
    second = str(tmp_path_factory.mktemp("week9_smoke_b"))
    _run_smoke(first)
    _run_smoke(second)
    return first, second


class TestBenchmarkSmoke:
    """The benchmark runs end to end and writes what the report reads."""

    @pytest.mark.parametrize("name", _FIGURES + ["comparison_table.csv"])
    def test_outputs_exist_and_are_non_empty(self, smoke_runs, name):
        path = _os.path.join(smoke_runs[0], name)
        assert _os.path.isfile(path) and _os.path.getsize(path) > 0

    @pytest.mark.parametrize("name", _FIGURES)
    def test_figures_are_png(self, smoke_runs, name):
        with open(_os.path.join(smoke_runs[0], name), "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_header_is_exactly_the_specified_columns(self, smoke_runs):
        header, rows = _rows(smoke_runs[0])
        assert header == _EXPECTED_HEADER
        assert rows

    def test_kind_takes_only_the_three_allowed_values(self, smoke_runs):
        _, rows = _rows(smoke_runs[0])
        assert {r["kind"] for r in rows} <= {"exact", "approximation", "heuristic"}

    def test_lower_bound_kind_takes_only_allowed_values(self, smoke_runs):
        _, rows = _rows(smoke_runs[0])
        assert {r["lower_bound_kind"] for r in rows} <= {"matching", "lp", "mst", "edges_over_1", ""}

    def test_every_problem_is_present(self, smoke_runs):
        _, rows = _rows(smoke_runs[0])
        assert {r["problem"] for r in rows} == {"vertex_cover", "set_cover", "facility_location",
                                                "metric_tsp", "maxcut"}

    def test_ratios_to_opt_respect_direction_and_proven_bounds(self, smoke_runs):
        _, rows = _rows(smoke_runs[0])
        checked = 0
        for r in rows:
            if not r["ratio_to_opt"]:
                continue
            ratio = float(r["ratio_to_opt"])
            checked += 1
            if r["problem"] == "maxcut":
                assert ratio <= 1.0 + 1e-9
                if r["proven_bound"]:
                    assert ratio >= float(r["proven_bound"]) - 1e-9
            else:
                assert ratio >= 1.0 - 1e-9
                if r["proven_bound"]:
                    assert ratio <= float(r["proven_bound"]) + 1e-9
        assert checked > 0

    def test_same_seed_gives_identical_non_timing_columns(self, smoke_runs):
        _, first = _rows(smoke_runs[0])
        _, second = _rows(smoke_runs[1])
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert {k: v for k, v in a.items() if k != "mean_time_s"} == \
                   {k: v for k, v in b.items() if k != "mean_time_s"}
