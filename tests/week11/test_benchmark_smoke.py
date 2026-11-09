"""Smoke tests for the Week 11 benchmark (spec section 9.7).

The benchmark runs end to end in a subprocess, as a user would run it, and
must write every figure and table the report reads.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK = os.path.join(REPO_ROOT, "benchmarks", "week11_benchmark_search.py")

EXPECTED_HEADER = [
    "experiment", "method", "dataset", "n", "m", "queries", "preprocess_s", "mean_query_s",
    "std_query_s", "matches", "comparisons", "spurious_hits", "expected_spurious", "base", "modulus",
    "peak_mem_kb", "mean_lcp", "max_lcp", "distinct_substrings", "lrs_length", "agree", "seed",
]
REQUIRED_FIGURES = ["runtime_comparison.png", "collision_analysis.png", "sa_lcp_stats.png",
                    "heatmap_dataset_sizes.png"]
OTHER_FIGURES = ["memory_use.png", "suffix_array_mississippi.png", "suffix_tree_banana.png"]
TIMING = {"preprocess_s", "mean_query_s", "std_query_s", "peak_mem_kb"}


def run_smoke(out_dir: str) -> None:
    """Run the benchmark's smoke mode into ``out_dir``; fail with its output on error."""
    completed = subprocess.run(
        [sys.executable, BENCHMARK, "--smoke", "--out", out_dir, "--seed", "42"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]


def read_rows(out_dir: str) -> tuple:
    with open(os.path.join(out_dir, "search_benchmark.csv"), newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke runs with the same master seed, in separate directories."""
    first = str(tmp_path_factory.mktemp("week11_smoke_a"))
    second = str(tmp_path_factory.mktemp("week11_smoke_b"))
    run_smoke(first)
    run_smoke(second)
    return first, second


@pytest.mark.parametrize("name", REQUIRED_FIGURES + OTHER_FIGURES
                         + ["search_benchmark.csv", "size_caps.csv", "suffix_tree_banana.dot"])
def test_outputs_exist_and_are_non_empty(smoke_runs, name):
    path = os.path.join(smoke_runs[0], name)
    assert os.path.isfile(path) and os.path.getsize(path) > 0


@pytest.mark.parametrize("name", REQUIRED_FIGURES + OTHER_FIGURES)
def test_figures_are_png(smoke_runs, name):
    with open(os.path.join(smoke_runs[0], name), "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_dot_file_is_graphviz_text_with_terminator(smoke_runs):
    with open(os.path.join(smoke_runs[0], "suffix_tree_banana.dot")) as handle:
        dot = handle.read()
    assert dot.lstrip().startswith(("digraph", "graph")) and "$" in dot


def test_header_is_exactly_the_specified_columns(smoke_runs):
    header, rows = read_rows(smoke_runs[0])
    assert header == EXPECTED_HEADER
    assert rows


def test_agree_is_true_or_blank_everywhere(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    assert {r["agree"] for r in rows} <= {"True", ""}
    assert any(r["agree"] == "True" for r in rows)


def test_every_experiment_is_present(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    assert {r["experiment"] for r in rows} == {"dataset_sizes", "rk_moduli", "rk_bases",
                                               "worst_case", "memory", "lcp_stats"}


def test_every_method_runs_on_both_datasets(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    sizes = {(r["method"], r["dataset"]) for r in rows if r["experiment"] == "dataset_sizes"}
    assert sizes == {(m, d) for m in ("kmp", "rk", "sa", "st") for d in ("dna", "english")}


def test_worst_case_shows_quadratic_naive_and_linear_kmp(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    worst = {(r["dataset"], r["method"]): r for r in rows if r["experiment"] == "worst_case"}
    match = [k for k in worst if k[0].endswith("a^m")][0][0]
    n, m = int(worst[(match, "naive")]["n"]), int(worst[(match, "naive")]["m"])
    assert int(worst[(match, "naive")]["comparisons"]) == (n - m + 1) * m
    assert int(worst[(match, "kmp")]["comparisons"]) < 3 * n
    assert int(worst[(match, "naive")]["matches"]) == n - m + 1


def test_distinct_substrings_are_exact_integers(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    lcp = [r for r in rows if r["experiment"] == "lcp_stats"]
    assert lcp and all(r["distinct_substrings"].isdigit() for r in lcp)


def test_same_seed_gives_identical_non_timing_columns(smoke_runs):
    _, first = read_rows(smoke_runs[0])
    _, second = read_rows(smoke_runs[1])
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert {k: v for k, v in a.items() if k not in TIMING} == \
               {k: v for k, v in b.items() if k not in TIMING}
