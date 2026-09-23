"""Tests for the Week 5 dynamic-programming benchmark harness.

What is proven here is that the *harness* works: that
``week5_dp_benchmark.main`` runs a study end to end, reports success, and
leaves behind every CSV and every figure the report is built from, each
one a real file of the right kind. Nothing here asserts on a timing. A
benchmark's numbers are a property of the machine it ran on, so a test
that pinned them would fail on a different laptop, on a busy one, or on
the next one - it would measure the room, not the code.

What *can* be pinned is the contract the report reads the CSV under, and
it is pinned hard:

* **The columns.** ``dp_vs_recursive_table.csv`` must open with exactly
  ``week5.COLUMNS`` in exactly that order, because the report looks
  fields up by name and a silently reordered or renamed column would
  produce a wrong figure rather than a crash.
* **The projection rule.** The module's own docstring promises that a
  point which was estimated rather than run is labelled ``projected``,
  that it carries no spread and no memory figure it did not measure, and
  that ``speedup_vs_recursive`` stays blank wherever the recursive
  baseline at that size was itself projected. Every one of those is
  asserted over the whole table, and asserted in the direction that
  holds whether or not the configuration under test happens to project
  anything: no speedup may exist unless a *measured* baseline row exists
  beside it. ``--quick`` does project one point today - naive Fibonacci
  at n = 30, above its ``fib_measured_max`` of 25 - but the tests do not
  depend on that staying true.
* **The pure helpers.** ``repeats_for``, ``add_speedups`` and
  ``write_csv`` are checked on inputs built here by hand, including the
  awkward ones: both boundaries of the repetition policy, a projected
  baseline, a baseline that does not exist, and a column order given
  explicitly against one derived from the rows.

Where a study's *answer* can be checked at all, it is checked against an
independent oracle rather than against a second copy of the module's
reasoning:

* the projected Fibonacci call count against ``2 * F(n + 1) - 1``,
  with ``F`` computed by an iterative loop written in this file;
* the knapsack optimum against brute force over all 2^n subsets;
* the LCS lengths the recursion probe records against brute force over
  all subsequences of the shorter string.

Those oracles are why the direct-study tests use tiny hand-built
:class:`week5.Config` instances instead of ``QUICK``: four items and
twelve characters are small enough to enumerate exhaustively, and fast
enough that the whole file stays inside its budget. The one end-to-end
``--quick`` run is module scoped and shared, so the study executes once
per session rather than once per assertion. There is no ``slow`` marker
registered in this project - no ``pytest.ini``, ``setup.cfg`` or
``pyproject.toml`` declares one and no ``conftest`` registers one - so
that run is left unmarked; an unregistered mark would only raise
``PytestUnknownMarkWarning``.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import itertools
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import pytest

import benchmarks.week5_dp_benchmark as week5

#: The eight bytes every PNG file begins with.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: The three tables the report reads back.
EXPECTED_CSVS: List[str] = [
    "dp_vs_recursive_table.csv",
    "week5_knapsack_cells.csv",
    "week5_recursion_probe.csv",
]

#: The three figures the report embeds.
EXPECTED_PNGS: List[str] = [
    "fibonacci_comparison.png",
    "knapsack_performance.png",
    "lcs_performance.png",
]

EXPECTED_FILES: List[str] = EXPECTED_CSVS + EXPECTED_PNGS

#: The three problems the study covers. Every row of the main table must
#: name one of them.
PROBLEMS: List[str] = ["fibonacci", "knapsack", "lcs"]

#: The variant each problem's speedup column is measured against.
BASELINE_VARIANT: Dict[str, str] = {
    "fibonacci": "naive",
    "knapsack": "recursive",
    "lcs": "recursive",
}

#: The columns a projected row must leave blank, because they are
#: properties of a run that did not happen.
BLANK_WHEN_PROJECTED: List[str] = [
    "std_time_s", "min_time_s", "max_time_s", "peak_kib",
]

#: Every size list QUICK overrides.
SIZE_FIELDS: List[str] = [
    "fib_sizes", "knapsack_sizes", "knapsack_capacities",
    "lcs_lengths", "lcs_recursive_lengths", "probe_lengths",
]

#: The scalar dimensions QUICK overrides.
SCALAR_FIELDS: List[str] = [
    "fib_measured_max", "knapsack_capacity", "knapsack_fixed_n",
    "knapsack_recursive_max",
]

CONFIG_NAMES: List[str] = ["QUICK", "FULL"]


# ----------------------------------------------------------------------
# Independent oracles: answers worked out here, not asked of the module
# ----------------------------------------------------------------------
def fibonacci(n: int) -> int:
    """F(n) by an iterative loop, sharing nothing with src.dp.fibonacci."""
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous


def naive_calls(n: int) -> int:
    """Calls the two-branch recursion makes: 2 * F(n + 1) - 1, exactly.

    Every call either returns a base case or spawns two more, so the call
    tree is a full binary tree with F(n + 1) leaves, and a full binary
    tree with L leaves has 2L - 1 nodes.
    """
    return 2 * fibonacci(n + 1) - 1


def brute_force_knapsack(weights: Sequence[int], values: Sequence[int],
                         capacity: int) -> int:
    """The best value of any subset that fits, by enumerating all 2^n of them."""
    best = 0
    for mask in range(1 << len(weights)):
        weight = sum(weights[i] for i in range(len(weights)) if mask >> i & 1)
        if weight <= capacity:
            best = max(best, sum(values[i] for i in range(len(values))
                                 if mask >> i & 1))
    return best


def is_subsequence(candidate: str, text: str) -> bool:
    """True when every character of ``candidate`` appears in ``text`` in order."""
    position = 0
    for character in candidate:
        position = text.find(character, position) + 1
        if position == 0:
            return False
    return True


def brute_force_lcs_length(left: str, right: str) -> int:
    """The longest common subsequence length, by enumerating subsequences.

    Every subsequence of the shorter string is generated and tested
    against the longer one, so this shares no recurrence, no table and no
    memo with src.dp.lcs. It costs O(2^min * (min + max)) and is only
    usable on the very small pairs this file feeds it.
    """
    shorter, longer = sorted((left, right), key=len)
    best = 0
    for mask in range(1 << len(shorter)):
        candidate = "".join(character for index, character in enumerate(shorter)
                            if mask >> index & 1)
        if len(candidate) > best and is_subsequence(candidate, longer):
            best = len(candidate)
    return best


def current_stack_depth() -> int:
    """How many frames are on the stack right now, counted by walking them."""
    depth = 0
    frame: Any = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def read_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse a CSV written by the harness into its header and its rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def tiny_config(**overrides: Any) -> week5.Config:
    """A Config small enough to enumerate exhaustively and run in a blink.

    The defaults are the smallest values each study still does real work
    at; a test overrides only the dimension it is about, so an unrelated
    study in the same call costs almost nothing.
    """
    settings: Dict[str, Any] = {
        "fib_sizes": [10],
        "fib_measured_max": 10,
        "knapsack_sizes": [4],
        "knapsack_capacity": 20,
        "knapsack_capacities": [10],
        "knapsack_fixed_n": 4,
        "knapsack_recursive_max": 4,
        "lcs_lengths": [8],
        "lcs_recursive_lengths": [6],
        "probe_lengths": [8],
    }
    settings.update(overrides)
    return week5.Config(**settings)


def key_of(row: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """The identity of a measurement row: which series, at which point."""
    return (row["problem"], row["variant"], str(row["n"]),
            str(row["secondary_param"]))


def speedup_row(problem: str, variant: str, n: int, mean: float, *,
                secondary: Any = "", measurement: str = "measured",
                ) -> Dict[str, Any]:
    """One hand-built row in the shape add_speedups expects to be handed."""
    row = week5.blank_row(problem, variant, n, secondary)
    row.update({"mean_time_s": f"{mean:.9f}", "measurement": measurement})
    return row


class QuickRun(NamedTuple):
    """Everything the one shared ``--quick`` execution is asked about."""

    code: int
    out: Path
    limit_before: int
    limit_after: int


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory) -> QuickRun:
    """Run the whole --quick study once, into a pytest temporary directory.

    Module scoped on purpose: the study takes a few seconds and every
    test below inspects the same outputs. The output path is deliberately
    two levels below a directory that does not exist yet, so the run also
    proves that ``main`` creates the tree ``--out`` names. The recursion
    limit is recorded either side, because the study raises it for the
    memoized LCS and promises to put it back.
    """
    out = tmp_path_factory.mktemp("week5") / "nested" / "results"
    limit_before = sys.getrecursionlimit()
    code = week5.main(["--quick", "--out", str(out)])
    return QuickRun(code, out, limit_before, sys.getrecursionlimit())


@pytest.fixture(scope="module")
def main_table(quick_run: QuickRun) -> Tuple[List[str], List[Dict[str, str]]]:
    """The header and rows of dp_vs_recursive_table.csv from the quick run."""
    return read_table(quick_run.out / "dp_vs_recursive_table.csv")


# ----------------------------------------------------------------------
# End to end: the harness runs and writes its outputs
# ----------------------------------------------------------------------
class TestEndToEnd:
    """A --quick run succeeds and leaves every expected artefact behind."""

    def test_main_reports_success(self, quick_run: QuickRun) -> None:
        assert quick_run.code == 0

    def test_the_output_directory_is_created(self, quick_run: QuickRun) -> None:
        """--out names a tree two levels deep that did not exist before."""
        assert quick_run.out.is_dir()

    @pytest.mark.parametrize("name", EXPECTED_FILES, ids=EXPECTED_FILES)
    def test_every_expected_file_is_written_and_not_empty(
            self, quick_run: QuickRun, name: str) -> None:
        path = quick_run.out / name
        assert path.is_file(), f"{name} was not written"
        assert path.stat().st_size > 0, f"{name} is empty"

    def test_nothing_unexpected_is_written(self, quick_run: QuickRun) -> None:
        """The output directory holds exactly the documented artefacts."""
        written = sorted(p.name for p in quick_run.out.iterdir())
        assert written == sorted(EXPECTED_FILES)

    @pytest.mark.parametrize("name", EXPECTED_PNGS, ids=EXPECTED_PNGS)
    def test_every_figure_is_a_real_png(self, quick_run: QuickRun,
                                        name: str) -> None:
        """A .png extension proves nothing; the magic bytes do."""
        assert (quick_run.out / name).read_bytes()[:8] == PNG_MAGIC

    @pytest.mark.parametrize("name", EXPECTED_CSVS, ids=EXPECTED_CSVS)
    def test_every_csv_parses_with_at_least_one_row(
            self, quick_run: QuickRun, name: str) -> None:
        fieldnames, rows = read_table(quick_run.out / name)
        assert fieldnames, f"{name} has no header"
        assert len(rows) >= 1, f"{name} has a header but no data"

    @pytest.mark.parametrize("name", EXPECTED_CSVS, ids=EXPECTED_CSVS)
    def test_no_csv_row_is_ragged(self, quick_run: QuickRun, name: str) -> None:
        """A short row leaves a None value; a long one lands under the None key."""
        fieldnames, rows = read_table(quick_run.out / name)
        for number, row in enumerate(rows, start=2):
            assert None not in row, f"{name} line {number} has extra fields"
            assert None not in row.values(), f"{name} line {number} is short"
            assert list(row) == fieldnames

    def test_the_out_flag_is_honoured_and_the_default_is_left_alone(
            self, quick_run: QuickRun) -> None:
        """The study's committed results must survive a test run."""
        assert str(quick_run.out) != week5.RESULTS_DIR
        assert not str(quick_run.out).startswith(week5.RESULTS_DIR)

    def test_the_recursion_limit_is_restored(self, quick_run: QuickRun) -> None:
        """main raises it for the memoized LCS and puts it back in a finally."""
        assert quick_run.limit_after == quick_run.limit_before

    def test_an_unknown_flag_is_rejected_before_any_work(self) -> None:
        with pytest.raises(SystemExit) as info:
            week5.main(["--no-such-flag"])
        assert info.value.code == 2

    def test_help_exits_cleanly(self, capsys) -> None:
        with pytest.raises(SystemExit) as info:
            week5.main(["--help"])
        assert info.value.code == 0
        assert "--quick" in capsys.readouterr().out


# ----------------------------------------------------------------------
# The CSV contract the report depends on
# ----------------------------------------------------------------------
class TestCsvContract:
    """dp_vs_recursive_table.csv is exactly the table the report reads."""

    def test_the_header_is_exactly_the_declared_columns(self, main_table) -> None:
        """Order included: the report reads fields by name from this list."""
        fieldnames, _rows = main_table
        assert fieldnames == week5.COLUMNS

    def test_the_declared_columns_have_no_duplicates(self) -> None:
        assert len(set(week5.COLUMNS)) == len(week5.COLUMNS)

    def test_every_row_names_one_of_the_three_problems(self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in rows:
            assert row["problem"] in PROBLEMS

    def test_every_row_names_a_variant(self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in rows:
            assert row["variant"].strip(), row

    def test_every_measurement_is_measured_or_projected(self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in rows:
            assert row["measurement"] in {"measured", "projected"}, row

    def test_all_three_problems_reach_the_table(self, main_table) -> None:
        _fieldnames, rows = main_table
        assert {row["problem"] for row in rows} == set(PROBLEMS)

    def test_every_problem_carries_its_baseline_and_both_dp_variants(
            self, main_table) -> None:
        """A missing series is a hole in the report, not a slow test."""
        _fieldnames, rows = main_table
        for problem in PROBLEMS:
            variants = {row["variant"] for row in rows
                        if row["problem"] == problem}
            assert BASELINE_VARIANT[problem] in variants, problem
            assert {"memo", "tab"} <= variants, problem

    def test_every_row_carries_a_positive_size(self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in rows:
            assert int(row["n"]) > 0, row

    def test_every_row_carries_a_positive_mean_time(self, main_table) -> None:
        """The value is not asserted on, only that a number was written."""
        _fieldnames, rows = main_table
        for row in rows:
            assert float(row["mean_time_s"]) > 0.0, row

    def test_every_row_carries_a_positive_call_count(self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in rows:
            assert int(row["calls"]) > 0, row

    def test_every_row_carries_a_positive_depth(self, main_table) -> None:
        """Tabulation reports 1, never 0: the top-level call is a frame."""
        _fieldnames, rows = main_table
        for row in rows:
            assert int(row["max_depth"]) >= 1, row

    def test_no_series_is_measured_at_the_same_point_twice(
            self, main_table) -> None:
        """One row per (problem, variant, n, secondary), or a figure
        double-plots one x value and the report double-counts it."""
        _fieldnames, rows = main_table
        counted = collections.Counter(key_of(row) for row in rows)
        duplicates = [key for key, count in counted.items() if count > 1]
        assert not duplicates, f"measured twice: {duplicates}"

    def test_the_knapsack_rows_carry_their_capacity(self, main_table) -> None:
        """secondary_param is the capacity for knapsack and blank elsewhere."""
        _fieldnames, rows = main_table
        for row in rows:
            if row["problem"] == "knapsack":
                assert int(row["secondary_param"]) > 0, row
            else:
                assert row["secondary_param"] == "", row

    def test_the_measured_rows_carry_a_spread_and_a_memory_figure(
            self, main_table) -> None:
        _fieldnames, rows = main_table
        measured = [row for row in rows if row["measurement"] == "measured"]
        assert measured, "the quick run measured nothing at all"
        for row in measured:
            assert float(row["std_time_s"]) >= 0.0, row
            assert float(row["min_time_s"]) > 0.0, row
            assert float(row["max_time_s"]) >= float(row["min_time_s"]), row
            assert float(row["peak_kib"]) >= 0.0, row

    def test_the_mean_lies_between_the_minimum_and_the_maximum(
            self, main_table) -> None:
        """A summary that fell outside its own extremes would be arithmetic,
        not timing, and it is the same claim on every machine."""
        _fieldnames, rows = main_table
        for row in rows:
            if row["measurement"] != "measured":
                continue
            assert (float(row["min_time_s"]) <= float(row["mean_time_s"])
                    <= float(row["max_time_s"])), row


class TestKnapsackCellsTable:
    """week5_knapsack_cells.csv is the report's central counterexample."""

    @pytest.fixture(scope="class")
    @classmethod
    def cells(cls, quick_run: QuickRun):
        return read_table(quick_run.out / "week5_knapsack_cells.csv")

    @pytest.mark.parametrize(
        "column",
        ["n", "capacity", "optimum", "memo_entries", "table_cells",
         "fraction_of_table", "items_chosen"],
    )
    def test_the_column_is_present(self, cells, column: str) -> None:
        fieldnames, _rows = cells
        assert column in fieldnames

    def test_the_memo_never_touches_more_cells_than_the_table_holds(
            self, cells) -> None:
        _fieldnames, rows = cells
        for row in rows:
            assert 0 < int(row["memo_entries"]) <= int(row["table_cells"]), row

    def test_the_fraction_is_the_two_counts_divided(self, cells) -> None:
        """Recomputed here, so a stale or mistyped fraction cannot pass."""
        _fieldnames, rows = cells
        for row in rows:
            expected = int(row["memo_entries"]) / int(row["table_cells"])
            assert float(row["fraction_of_table"]) == pytest.approx(
                expected, abs=5e-7), row

    def test_the_table_holds_n_times_capacity_plus_one_cells(self, cells) -> None:
        _fieldnames, rows = cells
        for row in rows:
            assert int(row["table_cells"]) == int(row["n"]) * (
                int(row["capacity"]) + 1), row

    def test_no_more_items_are_chosen_than_exist(self, cells) -> None:
        _fieldnames, rows = cells
        for row in rows:
            assert 0 <= int(row["items_chosen"]) <= int(row["n"]), row


class TestRecursionProbeTable:
    """week5_recursion_probe.csv records what the default stack limit did."""

    @pytest.fixture(scope="class")
    @classmethod
    def probe(cls, quick_run: QuickRun):
        return read_table(quick_run.out / "week5_recursion_probe.csv")

    def test_one_row_per_configured_probe_length(self, probe) -> None:
        _fieldnames, rows = probe
        assert [int(row["length"]) for row in rows] == week5.QUICK.probe_lengths

    def test_tabulation_never_fails(self, probe) -> None:
        """The control: the same recurrence, no stack, so always an answer."""
        _fieldnames, rows = probe
        for row in rows:
            assert row["tab_status"] == "ok"
            assert int(row["tab_result"]) >= 0

    def test_the_memo_status_is_one_of_the_two_recorded_outcomes(
            self, probe) -> None:
        _fieldnames, rows = probe
        for row in rows:
            assert row["memo_status"] in {"ok", "RecursionError"}

    def test_where_the_memo_survived_it_agrees_with_tabulation(
            self, probe) -> None:
        _fieldnames, rows = probe
        for row in rows:
            if row["memo_status"] == "ok":
                assert row["memo_result"] == row["tab_result"], row
            else:
                assert row["memo_result"] == "", row

    def test_the_limit_recorded_is_the_one_in_force_during_the_probe(
            self, probe) -> None:
        """The probe runs before main raises the limit, so it records the
        interpreter default rather than week5.RECURSION_LIMIT."""
        _fieldnames, rows = probe
        for row in rows:
            assert int(row["recursion_limit"]) < week5.RECURSION_LIMIT


# ----------------------------------------------------------------------
# The projection rule
# ----------------------------------------------------------------------
class TestProjectionRule:
    """Estimated points are labelled, carry no invented spread, and earn
    no speedups. Every assertion below holds vacuously if the
    configuration projected nothing, so it is safe either way."""

    @staticmethod
    def projected(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [row for row in rows if row["measurement"] == "projected"]

    @pytest.mark.parametrize("column", BLANK_WHEN_PROJECTED,
                             ids=BLANK_WHEN_PROJECTED)
    def test_a_projected_row_leaves_the_unmeasured_columns_blank(
            self, main_table, column: str) -> None:
        _fieldnames, rows = main_table
        for row in self.projected(rows):
            assert row[column] == "", (column, row)

    def test_a_projected_row_still_carries_a_positive_call_count(
            self, main_table) -> None:
        """The count is exact by closed form even where the time is not."""
        _fieldnames, rows = main_table
        for row in self.projected(rows):
            assert row["calls"].strip()
            assert int(row["calls"]) > 0, row

    def test_a_projected_row_still_carries_an_estimated_mean(
            self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in self.projected(rows):
            assert float(row["mean_time_s"]) > 0.0, row

    def test_a_projected_row_never_carries_a_speedup(self, main_table) -> None:
        _fieldnames, rows = main_table
        for row in self.projected(rows):
            assert row["speedup_vs_recursive"] == "", row

    def test_the_baseline_variant_never_carries_a_speedup(
            self, main_table) -> None:
        """A series cannot be a speedup against itself."""
        _fieldnames, rows = main_table
        for row in rows:
            if row["variant"] == BASELINE_VARIANT[row["problem"]]:
                assert row["speedup_vs_recursive"] == "", row

    def test_no_speedup_exists_without_a_measured_baseline_beside_it(
            self, main_table) -> None:
        """The rule stated the strong way round, so it also covers the case
        where nothing was projected: a ratio may only appear where a
        *measured* baseline row sits at the same size."""
        _fieldnames, rows = main_table
        measured_baselines = {
            (row["problem"], row["n"], row["secondary_param"])
            for row in rows
            if row["variant"] == BASELINE_VARIANT[row["problem"]]
            and row["measurement"] == "measured"
        }
        for row in rows:
            if row["speedup_vs_recursive"] == "":
                continue
            key = (row["problem"], row["n"], row["secondary_param"])
            assert key in measured_baselines, f"speedup without a baseline: {row}"

    def test_every_speedup_is_the_ratio_of_the_two_means(self, main_table) -> None:
        """Recomputed from the neighbouring rows, so the column cannot drift
        away from the numbers printed beside it. This is arithmetic on what
        was written, not an assertion about how fast the machine is."""
        _fieldnames, rows = main_table
        baselines = {
            (row["problem"], row["n"], row["secondary_param"]): row
            for row in rows
            if row["variant"] == BASELINE_VARIANT[row["problem"]]
            and row["measurement"] == "measured"
        }
        checked = 0
        for row in rows:
            if row["speedup_vs_recursive"] == "":
                continue
            baseline = baselines[(row["problem"], row["n"],
                                  row["secondary_param"])]
            expected = (float(baseline["mean_time_s"])
                        / float(row["mean_time_s"]))
            assert float(row["speedup_vs_recursive"]) == pytest.approx(
                expected, abs=0.01), row
            checked += 1
        assert checked > 0, "no speedup was written at all"

    def test_the_quick_configuration_measures_something_at_every_problem(
            self, main_table) -> None:
        """Projection must not have swallowed a whole problem."""
        _fieldnames, rows = main_table
        for problem in PROBLEMS:
            measured = [row for row in rows if row["problem"] == problem
                        and row["measurement"] == "measured"]
            assert measured, problem


# ----------------------------------------------------------------------
# The repetition policy
# ----------------------------------------------------------------------
#: (pilot seconds, expected (repeat, warmup)). The two boundaries are
#: bracketed from both sides with the smallest float step that separates
#: them, because a `<` written as `<=` would move exactly one of these.
REPEAT_CASES: List[Tuple[float, Tuple[int, int]]] = [
    (0.0, (5, 2)),
    (1e-9, (5, 2)),
    (0.01, (5, 2)),
    (0.049_999_999, (5, 2)),
    (0.05, (3, 1)),
    (0.050_000_001, (3, 1)),
    (0.2, (3, 1)),
    (0.499_999_999, (3, 1)),
    (0.5, (2, 0)),
    (0.500_000_001, (2, 0)),
    (1.0, (2, 0)),
    (60.0, (2, 0)),
    (3_600.0, (2, 0)),
]
REPEAT_IDS = [f"{pilot}s" for pilot, _expected in REPEAT_CASES]


class TestRepeatsFor:
    """Cheap work is repeated five times; expensive work twice, unwarmed."""

    @pytest.mark.parametrize("pilot,expected", REPEAT_CASES, ids=REPEAT_IDS)
    def test_the_pilot_cost_picks_the_policy(
            self, pilot: float, expected: Tuple[int, int]) -> None:
        assert week5.repeats_for(pilot) == expected

    def test_a_microsecond_pilot_gets_five_trials_after_two_warmups(self) -> None:
        assert week5.repeats_for(1e-6) == (5, 2)

    def test_a_mid_range_pilot_gets_three_trials_after_one_warmup(self) -> None:
        assert week5.repeats_for(0.1) == (3, 1)

    def test_an_expensive_pilot_gets_two_trials_and_no_warmup(self) -> None:
        assert week5.repeats_for(5.0) == (2, 0)

    def test_the_fifty_millisecond_boundary_is_exclusive_below(self) -> None:
        """50 ms itself is already the mid band, not the cheap one."""
        assert week5.repeats_for(0.05) == (3, 1)
        assert week5.repeats_for(0.05 - 1e-12) == (5, 2)

    def test_the_half_second_boundary_is_exclusive_below(self) -> None:
        assert week5.repeats_for(0.5) == (2, 0)
        assert week5.repeats_for(0.5 - 1e-12) == (3, 1)

    @pytest.mark.parametrize("pilot,expected", REPEAT_CASES, ids=REPEAT_IDS)
    def test_the_counts_are_always_usable(self, pilot: float,
                                          expected: Tuple[int, int]) -> None:
        """At least one measured run, and never more warm-ups than runs."""
        repeat, warmup = week5.repeats_for(pilot)
        assert repeat >= 1
        assert warmup >= 0
        assert warmup <= repeat

    def test_the_policy_never_gets_more_generous_as_work_gets_dearer(self) -> None:
        pilots = [pilot for pilot, _expected in REPEAT_CASES]
        repeats = [week5.repeats_for(pilot)[0] for pilot in sorted(pilots)]
        warmups = [week5.repeats_for(pilot)[1] for pilot in sorted(pilots)]
        assert repeats == sorted(repeats, reverse=True)
        assert warmups == sorted(warmups, reverse=True)

    def test_a_pilot_the_clock_reported_as_negative_falls_to_the_cheap_band(
            self) -> None:
        """perf_counter differences should never go backwards, but a
        fall-through to the cheapest band is the safe reading if one did."""
        assert week5.repeats_for(-1.0) == (5, 2)

    def test_exactly_three_bands_exist(self) -> None:
        outcomes = {week5.repeats_for(pilot) for pilot in
                    (0.0, 0.01, 0.05, 0.2, 0.5, 2.0, 100.0)}
        assert outcomes == {(5, 2), (3, 1), (2, 0)}


class TestMeasure:
    """measure runs a pilot, then applies the policy that pilot chose."""

    def test_it_returns_the_statistics_time_call_promises(self) -> None:
        stats = week5.measure(lambda: sum(range(100)))
        assert sorted(stats) == ["max", "mean", "min", "runs", "std"]

    def test_a_cheap_call_is_repeated_five_times(self) -> None:
        stats = week5.measure(lambda: sum(range(100)))
        assert stats["runs"] == 5

    def test_a_cheap_call_is_made_once_for_the_pilot_then_warmed_and_timed(
            self) -> None:
        """One pilot, two warm-ups, five timed runs: eight calls in all."""
        made: List[int] = []
        week5.measure(lambda: made.append(1))
        assert len(made) == 1 + 2 + 5

    def test_a_slow_pilot_cuts_the_repeat_count(self) -> None:
        """Only the first call is dear, so the policy is chosen by the pilot
        alone and the test still finishes in well under a tenth of a second."""
        counter = itertools.count()
        made: List[int] = []

        def slow_on_its_first_call() -> None:
            made.append(1)
            if next(counter) == 0:
                deadline = time.perf_counter() + 0.06
                while time.perf_counter() < deadline:
                    pass

        stats = week5.measure(slow_on_its_first_call)
        assert stats["runs"] == 3
        assert len(made) == 1 + 1 + 3

    def test_the_times_it_reports_are_finite_and_ordered(self) -> None:
        stats = week5.measure(lambda: sum(range(1000)))
        assert 0.0 <= stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["std"] >= 0.0

    def test_the_callables_return_value_is_discarded(self) -> None:
        """measure is a stopwatch: correctness is checked outside it."""
        stats = week5.measure(lambda: "not a number")
        assert stats["runs"] == 5


# ----------------------------------------------------------------------
# The speedup column
# ----------------------------------------------------------------------
class TestAddSpeedups:
    """Rows built by hand, including a baseline that was only projected."""

    @staticmethod
    def sample_rows() -> List[Dict[str, Any]]:
        """Four sizes: a measured baseline, a projected one, a missing one,
        and a row of another problem that must be left alone."""
        return [
            speedup_row("fibonacci", "naive", 10, 1.0),
            speedup_row("fibonacci", "memo", 10, 0.25),
            speedup_row("fibonacci", "tab", 10, 0.5),
            speedup_row("fibonacci", "naive", 20, 8.0, measurement="projected"),
            speedup_row("fibonacci", "memo", 20, 0.5),
            speedup_row("fibonacci", "tab", 20, 0.25),
            speedup_row("fibonacci", "memo", 30, 0.75),
            speedup_row("lcs", "memo", 10, 0.125),
        ]

    @pytest.fixture
    def rows(self) -> List[Dict[str, Any]]:
        rows = self.sample_rows()
        week5.add_speedups(rows, "fibonacci", "naive")
        return rows

    @staticmethod
    def find(rows: List[Dict[str, Any]], variant: str, n: int) -> Dict[str, Any]:
        return next(row for row in rows
                    if row["variant"] == variant and row["n"] == n)

    def test_a_measured_baseline_fills_the_ratio(
            self, rows: List[Dict[str, Any]]) -> None:
        """1.0 s against 0.25 s is four times, worked out here not there."""
        assert self.find(rows, "memo", 10)["speedup_vs_recursive"] == "4.00"
        assert self.find(rows, "tab", 10)["speedup_vs_recursive"] == "2.00"

    def test_a_projected_baseline_leaves_the_ratio_blank(
            self, rows: List[Dict[str, Any]]) -> None:
        """The whole point of the rule: 8.0 s was estimated, not observed."""
        assert self.find(rows, "memo", 20)["speedup_vs_recursive"] == ""
        assert self.find(rows, "tab", 20)["speedup_vs_recursive"] == ""

    def test_a_missing_baseline_leaves_the_ratio_blank(
            self, rows: List[Dict[str, Any]]) -> None:
        """n = 30 has no naive row at all, projected or otherwise."""
        assert self.find(rows, "memo", 30)["speedup_vs_recursive"] == ""

    def test_the_baseline_rows_themselves_are_left_blank(
            self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            if row["variant"] == "naive":
                assert row["speedup_vs_recursive"] == ""

    def test_rows_of_another_problem_are_untouched(
            self, rows: List[Dict[str, Any]]) -> None:
        lcs = next(row for row in rows if row["problem"] == "lcs")
        assert lcs["speedup_vs_recursive"] == ""

    def test_a_projected_row_gets_nothing_even_from_a_measured_baseline(
            self) -> None:
        rows = [
            speedup_row("knapsack", "recursive", 10, 1.0, secondary=100),
            speedup_row("knapsack", "memo", 10, 0.5, secondary=100,
                        measurement="projected"),
        ]
        week5.add_speedups(rows, "knapsack", "recursive")
        assert rows[1]["speedup_vs_recursive"] == ""

    def test_the_secondary_parameter_is_part_of_the_key(self) -> None:
        """Two capacities at one item count are different measurements, so a
        baseline at W = 100 must not be used for the row at W = 500."""
        rows = [
            speedup_row("knapsack", "recursive", 10, 1.0, secondary=100),
            speedup_row("knapsack", "memo", 10, 0.5, secondary=100),
            speedup_row("knapsack", "memo", 10, 0.5, secondary=500),
        ]
        week5.add_speedups(rows, "knapsack", "recursive")
        assert rows[1]["speedup_vs_recursive"] == "2.00"
        assert rows[2]["speedup_vs_recursive"] == ""

    def test_the_named_baseline_variant_is_the_one_used(self) -> None:
        """Ask for 'tab' as the baseline and 'tab' is what the ratios use."""
        rows = [
            speedup_row("lcs", "recursive", 10, 4.0),
            speedup_row("lcs", "tab", 10, 1.0),
            speedup_row("lcs", "memo", 10, 0.5),
        ]
        week5.add_speedups(rows, "lcs", "tab")
        assert self.find(rows, "memo", 10)["speedup_vs_recursive"] == "2.00"
        assert self.find(rows, "recursive", 10)["speedup_vs_recursive"] == "0.25"
        assert self.find(rows, "tab", 10)["speedup_vs_recursive"] == ""

    def test_a_slower_variant_records_a_ratio_below_one(self) -> None:
        """The column is a ratio, not a claim, so it may read 0.50."""
        rows = [
            speedup_row("lcs", "recursive", 6, 1.0),
            speedup_row("lcs", "memo", 6, 2.0),
        ]
        week5.add_speedups(rows, "lcs", "recursive")
        assert rows[1]["speedup_vs_recursive"] == "0.50"

    def test_it_is_written_to_two_decimal_places(self) -> None:
        rows = [
            speedup_row("lcs", "recursive", 6, 1.0),
            speedup_row("lcs", "memo", 6, 0.3),
        ]
        week5.add_speedups(rows, "lcs", "recursive")
        assert rows[1]["speedup_vs_recursive"] == "3.33"

    def test_calling_it_twice_changes_nothing(self) -> None:
        rows = self.sample_rows()
        week5.add_speedups(rows, "fibonacci", "naive")
        once = [dict(row) for row in rows]
        week5.add_speedups(rows, "fibonacci", "naive")
        assert rows == once

    def test_an_empty_table_is_accepted(self) -> None:
        rows: List[Dict[str, Any]] = []
        week5.add_speedups(rows, "fibonacci", "naive")
        assert rows == []

    def test_a_problem_with_no_baseline_series_is_accepted(self) -> None:
        rows = [speedup_row("fibonacci", "memo", 10, 0.5)]
        week5.add_speedups(rows, "fibonacci", "naive")
        assert rows[0]["speedup_vs_recursive"] == ""

    def test_it_returns_nothing_and_edits_in_place(self) -> None:
        rows = self.sample_rows()
        assert week5.add_speedups(rows, "fibonacci", "naive") is None
        assert self.find(rows, "memo", 10)["speedup_vs_recursive"] == "4.00"

    def test_no_other_column_is_disturbed(self) -> None:
        rows = self.sample_rows()
        before = [{name: value for name, value in row.items()
                   if name != "speedup_vs_recursive"} for row in rows]
        week5.add_speedups(rows, "fibonacci", "naive")
        after = [{name: value for name, value in row.items()
                  if name != "speedup_vs_recursive"} for row in rows]
        assert after == before


# ----------------------------------------------------------------------
# Writing the tables
# ----------------------------------------------------------------------
class TestWriteCsv:
    """A header, the rows, and the column order the caller asked for."""

    @staticmethod
    def rows() -> List[Dict[str, Any]]:
        return [{"alpha": 1, "beta": "two"}, {"alpha": 3, "beta": "four"}]

    def test_the_header_and_the_rows_are_written(self, tmp_path: Path) -> None:
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), self.rows())
        fieldnames, rows = read_table(path)
        assert fieldnames == ["alpha", "beta"]
        assert rows == [{"alpha": "1", "beta": "two"},
                        {"alpha": "3", "beta": "four"}]

    def test_an_explicit_column_order_is_respected(self, tmp_path: Path) -> None:
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), self.rows(), ["beta", "alpha"])
        fieldnames, rows = read_table(path)
        assert fieldnames == ["beta", "alpha"]
        assert [row["beta"] for row in rows] == ["two", "four"]

    def test_an_explicit_order_that_omits_a_column_is_rejected(
            self, tmp_path: Path) -> None:
        """DictWriter's default extrasaction raises rather than dropping the
        field, so a column cannot go missing from the report unnoticed."""
        path = tmp_path / "table.csv"
        with pytest.raises(ValueError):
            week5.write_csv(str(path), self.rows(), ["alpha"])

    def test_the_columns_are_derived_from_the_rows_when_none_is_given(
            self, tmp_path: Path) -> None:
        """First-seen order, across every row rather than only the first."""
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
        fieldnames, rows = read_table(path)
        assert fieldnames == ["a", "b", "c"]
        assert rows == [{"a": "1", "b": "2", "c": ""},
                        {"a": "3", "b": "", "c": "4"}]

    def test_a_key_missing_from_a_row_is_written_blank(
            self, tmp_path: Path) -> None:
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), [{"a": 1}], ["a", "b"])
        _fieldnames, rows = read_table(path)
        assert rows == [{"a": "1", "b": ""}]

    def test_a_missing_directory_is_created(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "deeper" / "table.csv"
        week5.write_csv(str(path), self.rows())
        assert path.is_file()

    def test_an_existing_file_is_replaced_not_appended_to(
            self, tmp_path: Path) -> None:
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), self.rows())
        week5.write_csv(str(path), [{"alpha": 9, "beta": "nine"}])
        _fieldnames, rows = read_table(path)
        assert rows == [{"alpha": "9", "beta": "nine"}]

    def test_no_blank_line_is_left_between_rows(self, tmp_path: Path) -> None:
        """The handle is opened with newline='', so csv does not double up
        the line endings on Windows or leave stray blanks anywhere."""
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), self.rows())
        text = path.read_text(encoding="utf-8")
        assert "\r\n\r\n" not in text
        assert "\n\n" not in text
        assert text.count("\n") == 3

    def test_an_empty_table_still_writes_its_header(self, tmp_path: Path) -> None:
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), [], ["alpha", "beta"])
        fieldnames, rows = read_table(path)
        assert fieldnames == ["alpha", "beta"]
        assert rows == []

    def test_no_rows_and_no_columns_writes_an_empty_header_line(
            self, tmp_path: Path) -> None:
        """Nothing to derive the columns from, so the header is empty."""
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), [])
        assert path.read_text(encoding="utf-8") == "\n"

    def test_it_writes_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "table.csv"
        week5.write_csv(str(path), [{"note": "memo touched 30 percent"}])
        assert "30 percent" in path.read_text(encoding="utf-8")

    def test_a_bare_relative_filename_raises_rather_than_writing(
            self, tmp_path: Path, monkeypatch) -> None:
        """NOTE: a bug, documented rather than fixed. write_csv calls
        os.makedirs(os.path.dirname(path)), and os.path.dirname of a bare
        name such as "out.csv" is "", which os.makedirs rejects with
        FileNotFoundError. Every caller inside the module passes an
        absolute path, so the study never hits it, but the helper is
        public and a bare name is the obvious thing to pass it. See
        ``bugs_found``."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            week5.write_csv("bare.csv", self.rows())
        assert not (tmp_path / "bare.csv").exists()

    def test_a_relative_path_with_a_directory_works(self, tmp_path: Path,
                                                    monkeypatch) -> None:
        """The same call succeeds the moment there is a directory part."""
        monkeypatch.chdir(tmp_path)
        week5.write_csv("./relative.csv", self.rows())
        assert (tmp_path / "relative.csv").is_file()


# ----------------------------------------------------------------------
# The blank-row and fill helpers the whole table is built from
# ----------------------------------------------------------------------
class TestRowHelpers:
    """blank_row and fill produce rows that fit COLUMNS exactly."""

    def test_a_blank_row_has_exactly_the_declared_columns(self) -> None:
        row = week5.blank_row("fibonacci", "memo", 10)
        assert list(row) == week5.COLUMNS

    def test_a_blank_row_sets_only_its_four_identifying_fields(self) -> None:
        row = week5.blank_row("knapsack", "tab", 20, 100)
        assert row["problem"] == "knapsack"
        assert row["variant"] == "tab"
        assert row["n"] == 20
        assert row["secondary_param"] == 100
        for name in week5.COLUMNS[4:]:
            assert row[name] == "", name

    def test_the_secondary_parameter_defaults_to_blank(self) -> None:
        assert week5.blank_row("lcs", "memo", 10)["secondary_param"] == ""

    def test_fill_writes_the_four_timings_to_nine_decimal_places(self) -> None:
        row = week5.fill(week5.blank_row("lcs", "tab", 10),
                         {"mean": 1.5, "std": 0.25, "min": 1.0, "max": 2.0})
        assert row["mean_time_s"] == "1.500000000"
        assert row["std_time_s"] == "0.250000000"
        assert row["min_time_s"] == "1.000000000"
        assert row["max_time_s"] == "2.000000000"

    def test_fill_defaults_to_calling_the_row_measured(self) -> None:
        row = week5.fill(week5.blank_row("lcs", "tab", 10),
                         {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0})
        assert row["measurement"] == "measured"

    def test_fill_leaves_the_memory_column_blank_when_nothing_was_traced(
            self) -> None:
        row = week5.fill(week5.blank_row("lcs", "tab", 10),
                         {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0})
        assert row["peak_kib"] == ""

    def test_fill_writes_a_traced_peak_to_one_decimal_place(self) -> None:
        row = week5.fill(week5.blank_row("lcs", "tab", 10),
                         {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                         peak=12.34)
        assert row["peak_kib"] == "12.3"

    def test_fill_keeps_a_zero_peak_rather_than_blanking_it(self) -> None:
        """0.0 is a measurement; only the empty string means "not traced"."""
        row = week5.fill(week5.blank_row("lcs", "tab", 10),
                         {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                         peak=0.0)
        assert row["peak_kib"] == "0.0"

    def test_fill_edits_in_place_and_returns_the_same_row(self) -> None:
        row = week5.blank_row("lcs", "tab", 10)
        assert week5.fill(row, {"mean": 1.0, "std": 0.0, "min": 1.0,
                                "max": 1.0}) is row

    def test_fill_never_adds_a_column(self) -> None:
        row = week5.fill(week5.blank_row("lcs", "tab", 10),
                         {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                         calls=5, depth=1, peak=2.0, measurement="projected")
        assert list(row) == week5.COLUMNS


# ----------------------------------------------------------------------
# Study 1, run directly on a tiny configuration
# ----------------------------------------------------------------------
class TestFibonacciStudy:
    """One measured naive point and one projected one, on n = 10 and 12."""

    @pytest.fixture(scope="class")
    @classmethod
    def rows(cls) -> List[Dict[str, Any]]:
        """fib_measured_max = 10 forces n = 12 down the projection branch."""
        return week5.fibonacci_study(
            tiny_config(fib_sizes=[10, 12], fib_measured_max=10))

    @staticmethod
    def find(rows: List[Dict[str, Any]], variant: str, n: int) -> Dict[str, Any]:
        return next(row for row in rows
                    if row["variant"] == variant and row["n"] == n)

    def test_every_size_gets_all_three_variants(
            self, rows: List[Dict[str, Any]]) -> None:
        for n in (10, 12):
            variants = {row["variant"] for row in rows if row["n"] == n}
            assert variants == {"naive", "memo", "tab"}

    def test_every_row_is_labelled_fibonacci(
            self, rows: List[Dict[str, Any]]) -> None:
        assert {row["problem"] for row in rows} == {"fibonacci"}

    def test_the_naive_point_inside_the_cap_is_measured(
            self, rows: List[Dict[str, Any]]) -> None:
        assert self.find(rows, "naive", 10)["measurement"] == "measured"

    def test_the_naive_point_past_the_cap_is_projected(
            self, rows: List[Dict[str, Any]]) -> None:
        assert self.find(rows, "naive", 12)["measurement"] == "projected"

    def test_the_dp_variants_are_measured_at_every_size(
            self, rows: List[Dict[str, Any]]) -> None:
        """Only the exponential series is ever projected."""
        for row in rows:
            if row["variant"] != "naive":
                assert row["measurement"] == "measured", row

    def test_the_measured_call_count_is_the_closed_form(
            self, rows: List[Dict[str, Any]]) -> None:
        """2 * F(11) - 1 = 177, with F computed by an iterative loop here."""
        assert naive_calls(10) == 177
        assert int(self.find(rows, "naive", 10)["calls"]) == naive_calls(10)

    def test_the_projected_call_count_is_the_closed_form(
            self, rows: List[Dict[str, Any]]) -> None:
        """The count is exact even where the time is an estimate."""
        assert int(self.find(rows, "naive", 12)["calls"]) == naive_calls(12)

    def test_the_projected_depth_is_n_exactly(
            self, rows: List[Dict[str, Any]]) -> None:
        """The recursion's deepest chain is n - 1, n - 2, ... which is n frames."""
        assert int(self.find(rows, "naive", 12)["max_depth"]) == 12

    @pytest.mark.parametrize("column", BLANK_WHEN_PROJECTED,
                             ids=BLANK_WHEN_PROJECTED)
    def test_the_projected_row_leaves_the_unmeasured_columns_blank(
            self, rows: List[Dict[str, Any]], column: str) -> None:
        assert self.find(rows, "naive", 12)[column] == ""

    def test_the_projection_is_the_measured_cost_per_call_times_the_calls(
            self, rows: List[Dict[str, Any]]) -> None:
        """The estimate is one multiplication of two numbers the code did
        produce, recomputed here from the row beside it. This is arithmetic
        on the written table, not an assertion about how fast the machine is."""
        measured = self.find(rows, "naive", 10)
        projected = self.find(rows, "naive", 12)
        per_call = float(measured["mean_time_s"]) / int(measured["calls"])
        assert float(projected["mean_time_s"]) == pytest.approx(
            per_call * int(projected["calls"]), rel=1e-3)

    def test_the_dp_rows_below_the_cap_earn_a_speedup(
            self, rows: List[Dict[str, Any]]) -> None:
        for variant in ("memo", "tab"):
            assert self.find(rows, variant, 10)["speedup_vs_recursive"] != ""

    def test_the_dp_rows_above_the_cap_earn_none(
            self, rows: List[Dict[str, Any]]) -> None:
        """Their baseline was projected, so the cell stays blank."""
        for variant in ("memo", "tab"):
            assert self.find(rows, variant, 12)["speedup_vs_recursive"] == ""

    def test_memoization_is_not_exponential(
            self, rows: List[Dict[str, Any]]) -> None:
        """Counted calls, not time: the same claim on any machine."""
        assert (int(self.find(rows, "memo", 10)["calls"])
                < int(self.find(rows, "naive", 10)["calls"]))

    def test_tabulation_never_recurses(self, rows: List[Dict[str, Any]]) -> None:
        """Depth 1 is the top-level call itself; the figure relies on this."""
        for n in (10, 12):
            assert int(self.find(rows, "tab", n)["max_depth"]) == 1

    def test_every_row_fits_the_declared_columns(
            self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            assert list(row) == week5.COLUMNS


# ----------------------------------------------------------------------
# Study 2, run directly on a tiny configuration
# ----------------------------------------------------------------------
class TestKnapsackStudy:
    """Four and five items, so brute force can check every optimum."""

    SIZES: List[int] = [4, 5]
    CAPACITY: int = 20
    RECURSIVE_MAX: int = 4

    @pytest.fixture(scope="class")
    @classmethod
    def study(cls) -> Tuple[List[Dict[str, Any]], List[dict]]:
        """knapsack_recursive_max = 4 makes n = 5 exercise the skip branch,
        and knapsack_fixed_n = 4 is inside knapsack_sizes so the capacity
        sweep exercises the already-measured branch too."""
        return week5.knapsack_study(tiny_config(
            knapsack_sizes=cls.SIZES, knapsack_capacity=cls.CAPACITY,
            knapsack_capacities=[10, cls.CAPACITY], knapsack_fixed_n=4,
            knapsack_recursive_max=cls.RECURSIVE_MAX))

    def test_every_row_is_labelled_knapsack(self, study) -> None:
        rows, _cells = study
        assert {row["problem"] for row in rows} == {"knapsack"}

    def test_the_recursive_variant_runs_inside_the_cap(self, study) -> None:
        rows, _cells = study
        variants = {row["variant"] for row in rows if row["n"] == 4
                    and row["secondary_param"] == self.CAPACITY}
        assert variants == {"recursive", "memo", "tab"}

    def test_the_recursive_variant_is_skipped_past_the_cap(self, study) -> None:
        """O(2^n) is the reason the cap exists, so the row is absent, not slow."""
        rows, _cells = study
        variants = {row["variant"] for row in rows if row["n"] == 5}
        assert variants == {"memo", "tab"}

    def test_no_point_is_measured_twice(self, study) -> None:
        """The capacity sweep skips the capacity the item sweep already did."""
        rows, _cells = study
        keys = [key_of(row) for row in rows]
        assert len(keys) == len(set(keys)), keys

    def test_the_capacity_sweep_still_adds_its_other_capacities(
            self, study) -> None:
        rows, _cells = study
        assert any(row["n"] == 4 and row["secondary_param"] == 10
                   for row in rows)

    def test_every_row_carries_its_capacity_as_the_secondary_parameter(
            self, study) -> None:
        rows, _cells = study
        for row in rows:
            assert row["secondary_param"] in {10, self.CAPACITY}, row

    def test_one_cells_record_per_item_count(self, study) -> None:
        _rows, cells = study
        assert [record["n"] for record in cells] == self.SIZES

    def test_the_optimum_matches_brute_force_over_every_subset(
            self, study) -> None:
        """The independent oracle: enumerate all 2^n subsets, keep the best
        that fits. The instance is regenerated from the study's own seeding
        rule, which is pinned by TestInstanceGeneration below."""
        _rows, cells = study
        for record in cells:
            weights, values = week5.knapsack_instance(
                record["n"], random.Random(week5.SEED + record["n"]))
            assert record["optimum"] == brute_force_knapsack(
                weights, values, record["capacity"]), record

    def test_the_memo_touches_fewer_cells_than_the_table_holds(
            self, study) -> None:
        """The report's central counterexample, in counts rather than time."""
        _rows, cells = study
        for record in cells:
            assert 0 < record["memo_entries"] < record["table_cells"], record

    def test_the_table_holds_n_times_capacity_plus_one_cells(self, study) -> None:
        _rows, cells = study
        for record in cells:
            assert record["table_cells"] == record["n"] * (
                record["capacity"] + 1), record

    def test_the_fraction_is_the_two_counts_divided(self, study) -> None:
        _rows, cells = study
        for record in cells:
            assert record["fraction_of_table"] == pytest.approx(
                record["memo_entries"] / record["table_cells"], abs=5e-7)

    def test_the_traced_solution_never_names_more_items_than_exist(
            self, study) -> None:
        _rows, cells = study
        for record in cells:
            assert 0 <= record["items_chosen"] <= record["n"], record

    def test_every_row_fits_the_declared_columns(self, study) -> None:
        rows, _cells = study
        for row in rows:
            assert list(row) == week5.COLUMNS


# ----------------------------------------------------------------------
# Study 3, run directly on a tiny configuration
# ----------------------------------------------------------------------
class TestLcsStudy:
    """Six and eight characters for the recursion, eight and twelve for DP."""

    RECURSIVE_LENGTHS: List[int] = [6, 8]
    LENGTHS: List[int] = [8, 12]

    @pytest.fixture(scope="class")
    @classmethod
    def rows(cls) -> List[Dict[str, Any]]:
        """Length 8 appears in both lists, so it has a recursive baseline;
        length 12 has none, which is the blank-speedup case."""
        return week5.lcs_study(tiny_config(
            lcs_recursive_lengths=cls.RECURSIVE_LENGTHS,
            lcs_lengths=cls.LENGTHS))

    @staticmethod
    def find(rows: List[Dict[str, Any]], variant: str, n: int) -> Dict[str, Any]:
        return next(row for row in rows
                    if row["variant"] == variant and row["n"] == n)

    def test_every_row_is_labelled_lcs(self, rows: List[Dict[str, Any]]) -> None:
        assert {row["problem"] for row in rows} == {"lcs"}

    def test_one_recursive_row_per_recursive_length(
            self, rows: List[Dict[str, Any]]) -> None:
        lengths = [row["n"] for row in rows if row["variant"] == "recursive"]
        assert lengths == self.RECURSIVE_LENGTHS

    def test_a_memo_and_a_tab_row_per_length(
            self, rows: List[Dict[str, Any]]) -> None:
        for variant in ("memo", "tab"):
            lengths = [row["n"] for row in rows if row["variant"] == variant]
            assert lengths == self.LENGTHS

    def test_every_row_is_measured(self, rows: List[Dict[str, Any]]) -> None:
        """Nothing in the LCS study is ever projected."""
        assert {row["measurement"] for row in rows} == {"measured"}

    def test_the_secondary_parameter_stays_blank(
            self, rows: List[Dict[str, Any]]) -> None:
        """LCS is swept on one dimension, so there is no second parameter."""
        assert {row["secondary_param"] for row in rows} == {""}

    def test_a_length_with_a_recursive_baseline_earns_a_speedup(
            self, rows: List[Dict[str, Any]]) -> None:
        for variant in ("memo", "tab"):
            assert self.find(rows, variant, 8)["speedup_vs_recursive"] != ""

    def test_a_length_without_one_earns_none(
            self, rows: List[Dict[str, Any]]) -> None:
        for variant in ("memo", "tab"):
            assert self.find(rows, variant, 12)["speedup_vs_recursive"] == ""

    def test_tabulation_never_recurses(self, rows: List[Dict[str, Any]]) -> None:
        for length in self.LENGTHS:
            assert int(self.find(rows, "tab", length)["max_depth"]) == 1

    def test_the_recursion_goes_deeper_than_one_frame(
            self, rows: List[Dict[str, Any]]) -> None:
        """The whole point of study 3a: laziness spends the stack."""
        for length in self.RECURSIVE_LENGTHS:
            assert int(self.find(rows, "recursive", length)["max_depth"]) > 1

    def test_memoization_makes_far_fewer_calls_than_the_plain_recursion(
            self, rows: List[Dict[str, Any]]) -> None:
        """Counted calls, not time."""
        assert (int(self.find(rows, "memo", 8)["calls"])
                < int(self.find(rows, "recursive", 8)["calls"]))

    def test_every_row_fits_the_declared_columns(
            self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            assert list(row) == week5.COLUMNS


class TestProbeRecursionLimit:
    """The probe records what the default stack limit did, never raises."""

    LENGTHS: List[int] = [8, 12]

    @pytest.fixture(scope="class")
    @classmethod
    def probe(cls) -> List[dict]:
        return week5.probe_recursion_limit(
            tiny_config(probe_lengths=cls.LENGTHS))

    def test_one_record_per_probe_length(self, probe: List[dict]) -> None:
        assert [record["length"] for record in probe] == self.LENGTHS

    def test_it_records_the_limit_in_force_when_it_ran(
            self, probe: List[dict]) -> None:
        for record in probe:
            assert record["recursion_limit"] == sys.getrecursionlimit()

    def test_it_leaves_the_recursion_limit_alone(self) -> None:
        """The module raises the limit in main, deliberately not in here."""
        before = sys.getrecursionlimit()
        week5.probe_recursion_limit(tiny_config(probe_lengths=[8]))
        assert sys.getrecursionlimit() == before

    def test_tabulation_is_always_recorded_as_the_control(
            self, probe: List[dict]) -> None:
        for record in probe:
            assert record["tab_status"] == "ok"

    def test_at_these_lengths_the_memo_survives_and_agrees(
            self, probe: List[dict]) -> None:
        for record in probe:
            assert record["memo_status"] == "ok"
            assert record["memo_result"] == record["tab_result"]

    def test_both_answers_match_brute_force_over_all_subsequences(
            self, probe: List[dict]) -> None:
        """The independent oracle: generate every subsequence of the shorter
        string and test it against the longer one. No recurrence, no table."""
        for record in probe:
            left, right = week5.lcs_pair(
                record["length"], random.Random(week5.SEED + record["length"]))
            expected = brute_force_lcs_length(left, right)
            assert record["tab_result"] == expected, record
            assert record["memo_result"] == expected, record

    def test_a_stack_too_shallow_is_recorded_rather_than_raised(self) -> None:
        """The branch the probe exists for. The limit is lowered to just
        above the current depth so a 60 character pair, which recurses about
        120 frames, cannot fit; it is restored in a finally whatever happens."""
        previous = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(current_stack_depth() + 60)
            records = week5.probe_recursion_limit(tiny_config(probe_lengths=[60]))
        finally:
            sys.setrecursionlimit(previous)
        assert records[0]["memo_status"] == "RecursionError"
        assert records[0]["memo_result"] == ""
        assert records[0]["tab_status"] == "ok"
        assert records[0]["tab_result"] > 0

    def test_the_control_still_answers_where_the_memo_failed(self) -> None:
        """Tabulation gets the right answer on the pair that broke the stack."""
        previous = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(current_stack_depth() + 60)
            records = week5.probe_recursion_limit(tiny_config(probe_lengths=[60]))
        finally:
            sys.setrecursionlimit(previous)
        left, right = week5.lcs_pair(60, random.Random(week5.SEED + 60))
        longest = max(len(left), len(right))
        assert 0 < records[0]["tab_result"] <= longest


# ----------------------------------------------------------------------
# The generated instances
# ----------------------------------------------------------------------
class TestInstanceGeneration:
    """Seeded, reproducible instances: a re-run benchmarks the same work."""

    @pytest.mark.parametrize("n", [0, 1, 2, 10, 50])
    def test_knapsack_instance_has_the_requested_number_of_items(
            self, n: int) -> None:
        weights, values = week5.knapsack_instance(n, random.Random(1))
        assert len(weights) == len(values) == n

    def test_knapsack_weights_and_values_are_inside_their_ranges(self) -> None:
        weights, values = week5.knapsack_instance(200, random.Random(2))
        assert all(1 <= weight <= 50 for weight in weights)
        assert all(1 <= value <= 100 for value in values)

    def test_knapsack_items_are_positive_so_the_optimum_is_meaningful(
            self) -> None:
        """A zero weight or value would make the DP trivially degenerate."""
        weights, values = week5.knapsack_instance(100, random.Random(3))
        assert all(weight > 0 for weight in weights)
        assert all(value > 0 for value in values)

    def test_the_same_seed_gives_the_same_knapsack_instance(self) -> None:
        first = week5.knapsack_instance(30, random.Random(4))
        second = week5.knapsack_instance(30, random.Random(4))
        assert first == second

    def test_a_different_seed_gives_a_different_knapsack_instance(self) -> None:
        first = week5.knapsack_instance(30, random.Random(4))
        second = week5.knapsack_instance(30, random.Random(5))
        assert first != second

    @pytest.mark.parametrize("length", [0, 1, 2, 10, 200])
    def test_lcs_pair_has_the_requested_length_on_both_sides(
            self, length: int) -> None:
        left, right = week5.lcs_pair(length, random.Random(6))
        assert len(left) == len(right) == length

    def test_lcs_pair_draws_from_the_declared_alphabet(self) -> None:
        left, right = week5.lcs_pair(500, random.Random(7))
        assert set(left) | set(right) <= set(week5.ALPHABET)

    def test_the_alphabet_is_the_four_dna_bases(self) -> None:
        assert sorted(week5.ALPHABET) == ["A", "C", "G", "T"]

    def test_the_two_strings_of_a_pair_are_not_the_same_string(self) -> None:
        """Identical strings would make the LCS trivially the whole string."""
        left, right = week5.lcs_pair(200, random.Random(8))
        assert left != right

    def test_the_same_seed_gives_the_same_pair(self) -> None:
        assert (week5.lcs_pair(100, random.Random(9))
                == week5.lcs_pair(100, random.Random(9)))

    def test_a_different_seed_gives_a_different_pair(self) -> None:
        assert (week5.lcs_pair(100, random.Random(9))
                != week5.lcs_pair(100, random.Random(10)))

    def test_the_study_seeds_every_size_differently(self) -> None:
        """SEED + n, so two sizes never share an instance prefix by accident."""
        instances = {
            week5.lcs_pair(20, random.Random(week5.SEED + n))
            for n in (10, 20, 30, 40)
        }
        assert len(instances) == 4

    def test_the_seed_is_a_fixed_number_the_module_publishes(self) -> None:
        assert isinstance(week5.SEED, int)


# ----------------------------------------------------------------------
# QUICK against FULL
# ----------------------------------------------------------------------
class TestQuickIsSmallerThanFull:
    """The smoke run is genuinely a smoke run, on every dimension it sets."""

    def test_every_configuration_field_is_covered_by_a_test_below(self) -> None:
        """A new dimension must not slip in untested."""
        declared = {f.name for f in dataclasses.fields(week5.Config)}
        assert declared == set(SIZE_FIELDS) | set(SCALAR_FIELDS)

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_measures_fewer_sizes(self, field: str) -> None:
        assert len(getattr(week5.QUICK, field)) < len(getattr(week5.FULL, field))

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_stops_below_the_largest_full_size(self, field: str) -> None:
        assert max(getattr(week5.QUICK, field)) < max(getattr(week5.FULL, field))

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_still_measures_at_least_two_sizes(self, field: str) -> None:
        """One point shows no growth, so a figure drawn from it says nothing."""
        assert len(getattr(week5.QUICK, field)) >= 2

    @pytest.mark.parametrize("field", SCALAR_FIELDS, ids=SCALAR_FIELDS)
    def test_quick_lowers_every_scalar_it_sets(self, field: str) -> None:
        assert getattr(week5.QUICK, field) < getattr(week5.FULL, field)

    def test_quick_does_less_fibonacci_work(self) -> None:
        """Naive Fibonacci costs about 2 * F(n + 1) calls at each measured
        size, so the measured total is the honest comparison."""
        def cost(config: week5.Config) -> int:
            return sum(naive_calls(n) for n in config.fib_sizes
                       if n <= config.fib_measured_max)

        assert cost(week5.QUICK) < cost(week5.FULL)

    def test_quick_does_less_knapsack_work(self) -> None:
        """The table is n * (W + 1) cells at every item count."""
        def cost(config: week5.Config) -> int:
            return sum(n * (config.knapsack_capacity + 1)
                       for n in config.knapsack_sizes)

        assert cost(week5.QUICK) < cost(week5.FULL)

    def test_quick_does_less_lcs_work(self) -> None:
        """The table is len(X) * len(Y) cells, and the pairs are square."""
        def cost(config: week5.Config) -> int:
            return sum(length * length for length in config.lcs_lengths)

        assert cost(week5.QUICK) < cost(week5.FULL)

    def test_quick_probes_the_recursion_limit_at_shallower_lengths(self) -> None:
        assert max(week5.QUICK.probe_lengths) < max(week5.FULL.probe_lengths)


class TestConfigurationInvariants:
    """Properties both configurations must hold for a run to make sense."""

    @pytest.fixture(params=CONFIG_NAMES, ids=CONFIG_NAMES)
    def config(self, request) -> week5.Config:
        return getattr(week5, request.param)

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_every_size_list_is_strictly_increasing(
            self, config: week5.Config, field: str) -> None:
        sizes = getattr(config, field)
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == len(sizes)

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_every_size_is_positive(self, config: week5.Config,
                                    field: str) -> None:
        assert all(size > 0 for size in getattr(config, field))

    @pytest.mark.parametrize("field", SCALAR_FIELDS, ids=SCALAR_FIELDS)
    def test_every_scalar_is_positive(self, config: week5.Config,
                                      field: str) -> None:
        assert getattr(config, field) > 0

    def test_the_smallest_fibonacci_size_is_measured_not_projected(
            self, config: week5.Config) -> None:
        """NOTE: a latent crash, documented rather than fixed. The
        projection multiplies ``per_call_seconds`` by the closed-form call
        count, and ``per_call_seconds`` is None until the first naive point
        has been measured. A configuration whose very first fib size sat
        above ``fib_measured_max`` would raise TypeError on None rather
        than report anything. Both shipped configurations satisfy the
        precondition, which is what this pins. See ``bugs_found``."""
        assert min(config.fib_sizes) <= config.fib_measured_max

    def test_at_least_one_fibonacci_size_is_measured(
            self, config: week5.Config) -> None:
        measured = [n for n in config.fib_sizes
                    if n <= config.fib_measured_max]
        assert len(measured) >= 2

    def test_the_recursive_knapsack_cap_admits_at_least_one_size(
            self, config: week5.Config) -> None:
        """No recursive row means no baseline and no speedup column at all."""
        under_cap = [n for n in config.knapsack_sizes
                     if n <= config.knapsack_recursive_max]
        assert under_cap, "the knapsack study would have no baseline"

    def test_the_fixed_capacity_is_one_of_the_swept_capacities(
            self, config: week5.Config) -> None:
        """The two sweeps meet at one point, which is what lets the figure
        put them on the same axes."""
        assert config.knapsack_capacity in config.knapsack_capacities

    def test_the_fixed_item_count_is_one_of_the_swept_item_counts(
            self, config: week5.Config) -> None:
        assert config.knapsack_fixed_n in config.knapsack_sizes

    def test_the_recursive_lcs_lengths_stay_well_below_the_dp_ones(
            self, config: week5.Config) -> None:
        """The plain recursion is exponential, so it stops far earlier."""
        assert max(config.lcs_recursive_lengths) < max(config.lcs_lengths)

    def test_the_recursive_lcs_lengths_are_small_enough_to_finish(
            self, config: week5.Config) -> None:
        """Beyond about 16 characters the plain recursion dominates the run."""
        assert max(config.lcs_recursive_lengths) <= 16

    def test_the_lcs_lengths_and_the_recursive_lengths_overlap(
            self, config: week5.Config) -> None:
        """Without a shared length no speedup could ever be written."""
        assert set(config.lcs_lengths) & set(config.lcs_recursive_lengths)

    def test_the_raised_recursion_limit_covers_the_deepest_lcs_pair(
            self, config: week5.Config) -> None:
        """The memoized LCS recurses to about len(X) + len(Y) frames."""
        assert 2 * max(config.lcs_lengths) < week5.RECURSION_LIMIT

    def test_the_raised_limit_is_above_the_interpreter_default(self) -> None:
        assert week5.RECURSION_LIMIT > sys.getrecursionlimit()


class TestModuleSurface:
    """The names the report, the notebook and this file all read by."""

    @pytest.mark.parametrize(
        "name",
        ["main", "repeats_for", "measure", "add_speedups", "write_csv",
         "fibonacci_study", "knapsack_study", "lcs_study",
         "probe_recursion_limit", "blank_row", "fill", "knapsack_instance",
         "lcs_pair"],
    )
    def test_the_callable_is_exported(self, name: str) -> None:
        assert callable(getattr(week5, name))

    @pytest.mark.parametrize(
        "name", ["COLUMNS", "SEED", "RECURSION_LIMIT", "RESULTS_DIR",
                 "ALPHABET", "QUICK", "FULL", "Config"],
    )
    def test_the_constant_is_exported(self, name: str) -> None:
        assert hasattr(week5, name)

    def test_the_default_output_directory_is_inside_the_repository(self) -> None:
        assert week5.RESULTS_DIR.endswith(
            os.path.join("benchmarks", "results"))

    def test_both_configurations_are_config_instances(self) -> None:
        assert isinstance(week5.QUICK, week5.Config)
        assert isinstance(week5.FULL, week5.Config)

    def test_the_module_is_documented(self) -> None:
        assert week5.__doc__ is not None
        assert "projected" in week5.__doc__
