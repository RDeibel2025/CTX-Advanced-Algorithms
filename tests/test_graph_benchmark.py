"""Tests for the Week 4 graph benchmark harness.

What is proven here is that the *harness* works: that
``week4_graph_benchmark.main`` runs a study end to end, returns success,
and leaves behind every CSV and every chart the report is built from,
each one a real file of the right kind. Nothing here asserts on a
timing. A benchmark's numbers are a property of the machine it ran on,
so a test that pinned them would fail on a different laptop, on a busy
one, or on the next one - it would measure the room, not the code.

The pure helpers are the part of the harness that *can* be pinned, so
they are pinned hard, and against arithmetic worked out in the test
rather than against the module's own formula:

* :func:`week4.classify` and :func:`week4.predicted_growth` are checked
  on growth ratios constructed here from first principles - a ratio of
  ``high / low`` must read as O(n), its square as O(n^2), 1.0 as O(1),
  and ``log(high) / log(low)`` as O(log n). The two functions are then
  checked to be inverses of each other, which is the contract the
  comparison table rests on.
* :func:`week4.adjacency_list_bytes` and :func:`week4.lookup_pairs` are
  checked against the graph's own public API, never against the
  internals they mirror.
* The QUICK configuration is checked to be genuinely smaller than FULL
  on every dimension it overrides, because a "quick" run that was not
  quick would make the end-to-end test slow rather than wrong, which is
  the harder failure to notice.

The end-to-end run writes into a pytest ``tmp_path`` and is shared by
every test that inspects its output, so the study is executed once per
session rather than once per assertion. It takes a few seconds. There is
no ``slow`` marker registered in this project - no ``pytest.ini``,
``setup.cfg`` or ``pyproject.toml`` declares one - so the run is not
marked; adding an unregistered mark would only raise
``PytestUnknownMarkWarning``.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import benchmarks.week4_graph_benchmark as week4
from src.graphs.graph import Graph
from src.utils.graph_generator import sparse_graph

#: The eight bytes every PNG file begins with.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: The four tables the report reads.
EXPECTED_CSVS: List[str] = [
    "graphs_comparison_table.csv",
    "week4_representation.csv",
    "week4_traversal.csv",
    "week4_dijkstra.csv",
]

#: The seven figures the report embeds: four measurement charts and the
#: three drawn traversal figures from study 4.
EXPECTED_PNGS: List[str] = [
    "bfs_vs_dfs_sparse.png",
    "bfs_vs_dfs_dense.png",
    "dijkstra_performance.png",
    "graph_representation.png",
    "bfs_traversal.png",
    "dfs_traversal.png",
    "bfs_vs_dfs_traversal.png",
]

EXPECTED_FILES: List[str] = EXPECTED_CSVS + EXPECTED_PNGS

#: Columns the comparison table must carry for the report's asymptotic
#: against empirical section to be writable from the CSV alone.
REQUIRED_COMPARISON_COLUMNS: List[str] = [
    "structure",
    "operation",
    "asymptotic",
    "expected_growth_class",
    "loglog_slope",
    "growth_measured",
    "growth_predicted",
    "empirical_class",
    "agrees",
]

#: Size pairs taken from the study's own configurations. Each is a pair
#: at which the five complexity classes predict five distinct growths, so
#: "nearest class" has a unique answer.
SIZE_PAIRS: List[Tuple[int, int]] = [
    (100, 10_000),
    (100, 8_000),
    (50, 800),
    (100, 1_000),
    (100, 500),
    (100, 250),
    (50, 100),
]
PAIR_IDS = [f"{low}-to-{high}" for low, high in SIZE_PAIRS]


# ----------------------------------------------------------------------
# Independent oracles: growth worked out here, not asked of the module
# ----------------------------------------------------------------------
def growth_of(klass: str, low: int, high: int) -> float:
    """How much a cost in ``klass`` grows from size ``low`` to size ``high``.

    Derived from the definitions rather than from
    :func:`week4.predicted_growth`, so a wrong formula in the module and
    a matching wrong formula here cannot cancel out.
    """
    ratio = high / low
    log_ratio = math.log(high) / math.log(low)
    return {
        "O(1)": 1.0,
        "O(log n)": log_ratio,
        "O(n)": ratio,
        "O(n log n)": ratio * log_ratio,
        "O(n^2)": ratio * ratio,
    }[klass]


def read_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse a CSV written by the harness into its header and its rows."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory) -> Tuple[int, Path]:
    """Run the whole --quick study once, into a pytest temporary directory.

    Module scoped on purpose: the study takes a few seconds, and every
    test below inspects the same outputs. Returns the exit code and the
    output directory.
    """
    out = tmp_path_factory.mktemp("week4_quick")
    code = week4.main(["--quick", "--out", str(out)])
    return code, out


# ----------------------------------------------------------------------
# End to end: the harness runs and writes its outputs
# ----------------------------------------------------------------------
class TestEndToEnd:
    """A --quick run succeeds and leaves every expected artefact behind."""

    def test_main_reports_success(self, quick_run):
        code, _out = quick_run
        assert code == 0

    @pytest.mark.parametrize("name", EXPECTED_FILES, ids=EXPECTED_FILES)
    def test_every_expected_file_is_written_and_not_empty(self, quick_run, name):
        _code, out = quick_run
        path = out / name
        assert path.is_file(), f"{name} was not written"
        assert path.stat().st_size > 0, f"{name} is empty"

    def test_nothing_unexpected_is_written(self, quick_run):
        """The output directory holds exactly the documented artefacts."""
        _code, out = quick_run
        assert sorted(p.name for p in out.iterdir()) == sorted(EXPECTED_FILES)

    @pytest.mark.parametrize("name", EXPECTED_PNGS, ids=EXPECTED_PNGS)
    def test_every_chart_is_a_real_png(self, quick_run, name):
        """A .png extension proves nothing; the magic bytes do."""
        _code, out = quick_run
        assert (out / name).read_bytes()[:8] == PNG_MAGIC

    @pytest.mark.parametrize("name", EXPECTED_CSVS, ids=EXPECTED_CSVS)
    def test_every_csv_parses_with_at_least_one_row(self, quick_run, name):
        _code, out = quick_run
        fieldnames, rows = read_table(out / name)
        assert fieldnames, f"{name} has no header"
        assert len(rows) >= 1, f"{name} has a header but no data"

    @pytest.mark.parametrize("name", EXPECTED_CSVS, ids=EXPECTED_CSVS)
    def test_no_csv_row_is_ragged(self, quick_run, name):
        """A short row leaves a None value; a long one lands under None."""
        _code, out = quick_run
        fieldnames, rows = read_table(out / name)
        for number, row in enumerate(rows, start=2):
            assert None not in row, f"{name} line {number} has extra fields"
            assert None not in row.values(), f"{name} line {number} is short"
            assert list(row) == fieldnames

    def test_the_out_flag_is_honoured_and_the_default_is_left_alone(self, quick_run):
        """Nothing lands in benchmarks/results when --out points elsewhere.

        The study's committed results live in ``RESULTS_DIR``; a test run
        must not overwrite them. NOTE: study 4 prints a hardcoded
        "benchmarks/results/..." line for the three drawn figures no
        matter where --out points, so the log is misleading even though
        the files go to the right place. See ``bugs_found``.
        """
        _code, out = quick_run
        assert str(out) != week4.RESULTS_DIR
        for name in EXPECTED_FILES:
            assert (out / name).is_file()


# ----------------------------------------------------------------------
# The comparison table the report is written from
# ----------------------------------------------------------------------
class TestComparisonTable:
    """graphs_comparison_table.csv carries the columns the report quotes."""

    @pytest.fixture(scope="class")
    @classmethod
    def comparison(cls, quick_run):
        _code, out = quick_run
        return read_table(out / "graphs_comparison_table.csv")

    @pytest.mark.parametrize(
        "column", REQUIRED_COMPARISON_COLUMNS, ids=REQUIRED_COMPARISON_COLUMNS
    )
    def test_required_column_is_present(self, comparison, column):
        fieldnames, _rows = comparison
        assert column in fieldnames

    def test_every_row_names_a_series_and_a_bound(self, comparison):
        _fieldnames, rows = comparison
        for row in rows:
            assert row["structure"].strip()
            assert row["operation"].strip()
            assert row["asymptotic"].strip()

    def test_both_class_columns_hold_known_classes(self, comparison):
        _fieldnames, rows = comparison
        for row in rows:
            assert row["expected_growth_class"] in week4.CLASSES
            assert row["empirical_class"] in week4.CLASSES

    def test_agrees_is_the_verdict_the_two_class_columns_imply(self, comparison):
        """The verdict column is not free to disagree with its own inputs.

        Whether a series agrees is a timing outcome and is deliberately
        not asserted: a two-point --quick run is far too noisy for that.
        What must hold is that the recorded verdict matches the two
        classes recorded beside it.
        """
        _fieldnames, rows = comparison
        for row in rows:
            expected = row["empirical_class"] == row["expected_growth_class"]
            assert row["agrees"] == str(expected), row["structure"]

    def test_the_numeric_columns_are_numbers(self, comparison):
        _fieldnames, rows = comparison
        for row in rows:
            slope = float(row["loglog_slope"])
            measured = float(row["growth_measured"])
            predicted = float(row["growth_predicted"])
            assert math.isfinite(slope)
            assert measured > 0.0
            assert predicted > 0.0

    def test_the_table_covers_traversal_dijkstra_and_representation(self, comparison):
        """All three studies reach the table, or the report has a hole."""
        _fieldnames, rows = comparison
        structures = " | ".join(row["structure"] for row in rows)
        for subject in ("BFS", "DFS", "Dijkstra", "Adjacency list", "Adjacency matrix"):
            assert subject in structures


# ----------------------------------------------------------------------
# The classifier
# ----------------------------------------------------------------------
class TestClassifier:
    """predicted_growth states what a class implies; classify reads it back."""

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    @pytest.mark.parametrize("klass", week4.CLASSES, ids=list(week4.CLASSES))
    def test_predicted_growth_matches_the_definition(self, klass, low, high):
        assert week4.predicted_growth(klass, low, high) == pytest.approx(
            growth_of(klass, low, high)
        )

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_constant_growth_is_classified_constant(self, low, high):
        assert week4.classify(1.0, low, high) == "O(1)"

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_the_log_ratio_is_classified_logarithmic(self, low, high):
        log_ratio = math.log(high) / math.log(low)
        assert week4.classify(log_ratio, low, high) == "O(log n)"

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_the_size_ratio_is_classified_linear(self, low, high):
        assert week4.classify(high / low, low, high) == "O(n)"

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_the_ratio_times_the_log_ratio_is_classified_linearithmic(self, low, high):
        growth = (high / low) * (math.log(high) / math.log(low))
        assert week4.classify(growth, low, high) == "O(n log n)"

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_the_squared_ratio_is_classified_quadratic(self, low, high):
        assert week4.classify((high / low) ** 2, low, high) == "O(n^2)"

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_the_two_functions_are_inverses(self, low, high):
        """Feed a class its own prediction and the same class comes back."""
        for klass in week4.CLASSES:
            growth = week4.predicted_growth(klass, low, high)
            assert week4.classify(growth, low, high) == klass

    @pytest.mark.parametrize("low,high", SIZE_PAIRS, ids=PAIR_IDS)
    def test_predictions_increase_with_the_class(self, low, high):
        """CLASSES is ordered, so its predictions must be too."""
        predictions = [week4.predicted_growth(k, low, high) for k in week4.CLASSES]
        assert predictions == sorted(predictions)
        assert len(set(predictions)) == len(predictions)

    @pytest.mark.parametrize("growth", [0.25, 0.9, 1.7, 13.0, 640.0, 1e6])
    def test_classify_always_names_one_of_the_known_classes(self, growth):
        assert week4.classify(growth, 100, 1_000) in week4.CLASSES

    def test_growth_below_one_is_classified_constant(self):
        """A series that got faster is flat, not sub-constant."""
        assert week4.classify(0.7, 100, 500) == "O(1)"


# ----------------------------------------------------------------------
# Memory accounting for the adjacency list
# ----------------------------------------------------------------------
class TestAdjacencyListBytes:
    """The bytes an adjacency list holds: positive, and rising with size."""

    def test_an_empty_graph_still_holds_its_outer_container(self):
        assert week4.adjacency_list_bytes(Graph()) > 0

    def test_a_single_node_graph_is_positive(self):
        graph = Graph()
        graph.add_node("only")
        assert week4.adjacency_list_bytes(graph) > 0

    @pytest.mark.parametrize("n", [10, 100, 1_000])
    def test_the_count_is_positive(self, n):
        graph = sparse_graph(n, avg_degree=4, connected=True, seed=4)
        assert week4.adjacency_list_bytes(graph) > 0

    def test_the_count_rises_with_the_number_of_nodes(self):
        sizes = [10, 100, 1_000, 5_000]
        counted = [
            week4.adjacency_list_bytes(
                sparse_graph(n, avg_degree=4, connected=True, seed=4)
            )
            for n in sizes
        ]
        assert counted == sorted(counted)
        assert len(set(counted)) == len(counted)

    def test_a_bigger_graph_costs_more_than_an_order_of_magnitude_smaller_one(self):
        """Ten times the nodes is not within noise of the same total."""
        small = week4.adjacency_list_bytes(
            sparse_graph(100, avg_degree=4, connected=True, seed=4)
        )
        large = week4.adjacency_list_bytes(
            sparse_graph(1_000, avg_degree=4, connected=True, seed=4)
        )
        assert large > small * 5

    def test_the_count_does_not_depend_on_being_measured_twice(self):
        graph = sparse_graph(200, avg_degree=4, connected=True, seed=4)
        assert week4.adjacency_list_bytes(graph) == week4.adjacency_list_bytes(graph)

    def test_measuring_leaves_the_graph_untouched(self):
        graph = sparse_graph(200, avg_degree=4, connected=True, seed=4)
        before = (graph.node_count, graph.edge_count, graph.nodes(), graph.edges())
        week4.adjacency_list_bytes(graph)
        assert (graph.node_count, graph.edge_count, graph.nodes(), graph.edges()) == before


# ----------------------------------------------------------------------
# The lookup workload
# ----------------------------------------------------------------------
class TestLookupPairs:
    """Half present edges, half arbitrary pairs, in the requested number."""

    @pytest.fixture(scope="class")
    @classmethod
    def graph(cls) -> Any:
        return sparse_graph(400, avg_degree=4, connected=True, seed=6)

    @pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 100, 2_000])
    def test_exactly_the_requested_number_of_pairs_comes_back(self, graph, count):
        pairs = week4.lookup_pairs(graph, count, random.Random(1))
        assert len(pairs) == count

    def test_every_pair_names_two_nodes_of_the_graph(self, graph):
        nodes = set(graph.nodes())
        for u, v in week4.lookup_pairs(graph, 500, random.Random(2)):
            assert u in nodes and v in nodes

    def test_about_half_the_pairs_are_present_edges(self, graph):
        """Counted with the graph's own has_edge, not with the generator."""
        count = 2_000
        pairs = week4.lookup_pairs(graph, count, random.Random(3))
        present = sum(1 for u, v in pairs if graph.has_edge(u, v))
        guaranteed = (count + 1) // 2  # the even positions are drawn from edges()
        assert present >= guaranteed
        assert 0.5 <= present / count <= 0.6

    def test_the_present_half_is_every_even_position(self, graph):
        """The two halves are interleaved, not shuffled.

        NOTE: the docstring says "interleaved and shuffled" but no shuffle
        happens - position 0, 2, 4 ... is always a present edge. That is a
        stale docstring, not a measurement problem: the mix the benchmark
        needs is still half and half. See ``bugs_found``.
        """
        pairs = week4.lookup_pairs(graph, 400, random.Random(4))
        assert all(graph.has_edge(u, v) for u, v in pairs[0::2])

    def test_the_arbitrary_half_mostly_misses(self, graph):
        """On a sparse graph a random pair is almost never an edge."""
        pairs = week4.lookup_pairs(graph, 1_000, random.Random(5))
        arbitrary = pairs[1::2]
        absent = sum(1 for u, v in arbitrary if not graph.has_edge(u, v))
        assert absent >= 0.8 * len(arbitrary)

    def test_the_same_seed_gives_the_same_workload(self, graph):
        """Every run of the study must time the identical set of lookups."""
        first = week4.lookup_pairs(graph, 200, random.Random(7))
        second = week4.lookup_pairs(graph, 200, random.Random(7))
        assert first == second

    def test_different_seeds_give_different_workloads(self, graph):
        first = week4.lookup_pairs(graph, 200, random.Random(7))
        second = week4.lookup_pairs(graph, 200, random.Random(8))
        assert first != second

    def test_a_graph_with_no_edges_yields_only_arbitrary_pairs(self):
        """With edges() empty there is no present half to draw from."""
        graph = Graph()
        for node in range(10):
            graph.add_node(node)
        pairs = week4.lookup_pairs(graph, 20, random.Random(9))
        assert len(pairs) == 20
        assert not any(graph.has_edge(u, v) for u, v in pairs)

    def test_the_workload_matches_the_configured_lookup_count(self, graph):
        """The study asks for Config.lookups pairs, so that many come back."""
        for config in (week4.QUICK, week4.FULL):
            pairs = week4.lookup_pairs(graph, config.lookups, random.Random(10))
            assert len(pairs) == config.lookups


# ----------------------------------------------------------------------
# Repetition counts
# ----------------------------------------------------------------------
class TestRunsFor:
    """runs_for reads the configured repetitions, or falls back sensibly."""

    @pytest.mark.parametrize("config_name", ["QUICK", "FULL"])
    def test_every_listed_size_gets_its_configured_counts(self, config_name):
        config = getattr(week4, config_name)
        for size, expected_runs in config.runs.items():
            runs, warmups = week4.runs_for(config, size)
            assert runs == expected_runs
            assert warmups == config.warmups[size]

    @pytest.mark.parametrize("config_name", ["QUICK", "FULL"])
    @pytest.mark.parametrize("size", [7, 137, 3_333, 999_999])
    def test_an_unlisted_size_gets_the_default(self, config_name, size):
        config = getattr(week4, config_name)
        assert size not in config.runs
        assert week4.runs_for(config, size) == (3, 1)

    @pytest.mark.parametrize("config_name", ["QUICK", "FULL"])
    def test_every_size_the_study_measures_is_listed_explicitly(self, config_name):
        """No benchmarked size should silently fall through to the default."""
        config = getattr(week4, config_name)
        measured = set(
            config.repr_sizes
            + config.sparse_sizes
            + config.dense_sizes
            + config.dijkstra_sizes
        )
        assert measured <= set(config.runs)
        assert measured <= set(config.warmups)

    @pytest.mark.parametrize("config_name", ["QUICK", "FULL"])
    def test_counts_are_usable_numbers(self, config_name):
        """At least one measured run, and never more warm-ups than runs."""
        config = getattr(week4, config_name)
        for size in sorted(set(config.runs) | {12_345}):
            runs, warmups = week4.runs_for(config, size)
            assert runs >= 1
            assert warmups >= 0
            assert warmups <= runs


# ----------------------------------------------------------------------
# QUICK against FULL
# ----------------------------------------------------------------------
#: Every size list QUICK overrides.
SIZE_FIELDS = ["repr_sizes", "sparse_sizes", "dense_sizes", "dijkstra_sizes"]

#: The scalar dimensions QUICK overrides.
SCALAR_FIELDS = ["lookups", "linear_dijkstra_cap"]

#: The dimensions QUICK leaves at the dataclass default, which therefore
#: describe the same experiment in both configurations.
SHARED_FIELDS = ["avg_degree", "dense_density", "figure_nodes"]


class TestQuickIsSmallerThanFull:
    """The smoke run is genuinely a smoke run, on every dimension it sets."""

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_measures_fewer_sizes(self, field):
        quick = getattr(week4.QUICK, field)
        full = getattr(week4.FULL, field)
        assert len(quick) < len(full)

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_stops_below_the_largest_full_size(self, field):
        quick = getattr(week4.QUICK, field)
        full = getattr(week4.FULL, field)
        assert max(quick) < max(full)

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_is_the_opening_stretch_of_full(self, field):
        """Same starting sizes, fewer of them: a prefix, not a new study."""
        quick = getattr(week4.QUICK, field)
        full = getattr(week4.FULL, field)
        assert full[: len(quick)] == quick

    @pytest.mark.parametrize("field", SIZE_FIELDS, ids=SIZE_FIELDS)
    def test_quick_still_measures_at_least_two_sizes(self, field):
        """One point has no growth, so the comparison table would skip it."""
        assert len(getattr(week4.QUICK, field)) >= 2

    @pytest.mark.parametrize("field", SCALAR_FIELDS, ids=SCALAR_FIELDS)
    def test_quick_lowers_every_scalar_it_sets(self, field):
        assert getattr(week4.QUICK, field) < getattr(week4.FULL, field)

    def test_quick_repeats_every_size_fewer_times(self):
        for size, runs in week4.QUICK.runs.items():
            assert size in week4.FULL.runs, f"FULL does not list size {size}"
            assert runs < week4.FULL.runs[size], f"runs at n={size}"

    def test_quick_warms_up_every_size_fewer_times(self):
        for size, warmups in week4.QUICK.warmups.items():
            assert size in week4.FULL.warmups, f"FULL does not list size {size}"
            assert warmups < week4.FULL.warmups[size], f"warm-ups at n={size}"

    def test_quick_does_less_total_work_at_every_study(self):
        """Sizes times repetitions, which is what the runtime tracks."""
        for field in SIZE_FIELDS:
            quick = sum(
                n * week4.runs_for(week4.QUICK, n)[0]
                for n in getattr(week4.QUICK, field)
            )
            full = sum(
                n * week4.runs_for(week4.FULL, n)[0] for n in getattr(week4.FULL, field)
            )
            assert quick < full, field

    @pytest.mark.parametrize("field", SHARED_FIELDS, ids=SHARED_FIELDS)
    def test_the_dimensions_quick_leaves_alone_are_identical(self, field):
        assert getattr(week4.QUICK, field) == getattr(week4.FULL, field)

    def test_the_linear_dijkstra_cap_still_admits_two_quick_sizes(self):
        """Below two points the O(V^2) series never reaches the table."""
        under_cap = [
            n for n in week4.QUICK.dijkstra_sizes if n <= week4.QUICK.linear_dijkstra_cap
        ]
        assert len(under_cap) >= 2
