"""Smoke tests for the Week 10 benchmark and CLI (spec section 9.5).

Run end to end in subprocesses, so these prove the scripts work as a user
would run them: the benchmark writes every figure, animation and table, and
the command line prints the known answer for the sample network.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK = os.path.join(REPO_ROOT, "benchmarks", "week10_flow_benchmark.py")
SAMPLE_GRAPH = os.path.join(REPO_ROOT, "src", "flow_network_solver", "examples", "clrs_network.txt")
SAMPLE_COSTS = os.path.join(REPO_ROOT, "src", "flow_network_solver", "examples", "min_cost_example.txt")

EXPECTED_HEADER = [
    "experiment", "solver", "graph_type", "n", "m", "cap_dist", "runs", "max_flow", "cost",
    "mean_time_s", "std_time_s", "augmentations", "pushes", "relabels", "peak_mem_kb", "valid",
    "ek_pr_agree", "seed",
]
BENCHMARK_FIGURES = ["runtime_scaling.png", "sparse_vs_dense.png", "augmentations.png",
                     "pr_convergence.png", "memory_use.png"]
ANCHOR_FIGURES = ["network.png", "matching.png", "segmentation.png"]
ANIMATIONS = ["ek_animation.gif", "pr_animation.gif"]
TIMING = {"mean_time_s", "std_time_s", "peak_mem_kb"}


def run_smoke(out_dir: str) -> None:
    """Run the benchmark's smoke mode into ``out_dir``; fail with its output on error."""
    completed = subprocess.run(
        [sys.executable, BENCHMARK, "--smoke", "--out", out_dir, "--seed", "42"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]


def read_rows(out_dir: str) -> tuple:
    with open(os.path.join(out_dir, "flow_benchmark.csv"), newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "src.flow_network_solver.main", *args],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke runs with the same master seed, in separate directories."""
    first = str(tmp_path_factory.mktemp("week10_smoke_a"))
    second = str(tmp_path_factory.mktemp("week10_smoke_b"))
    run_smoke(first)
    run_smoke(second)
    return first, second


class TestBenchmarkSmoke:
    """The benchmark runs end to end and writes what the report reads."""

    @pytest.mark.parametrize("name", BENCHMARK_FIGURES + ANCHOR_FIGURES + ANIMATIONS
                             + ["flow_benchmark.csv", "pr_convergence.csv"])
    def test_outputs_exist_and_are_non_empty(self, smoke_runs, name):
        path = os.path.join(smoke_runs[0], name)
        assert os.path.isfile(path) and os.path.getsize(path) > 0

    @pytest.mark.parametrize("name", BENCHMARK_FIGURES + ANCHOR_FIGURES)
    def test_figures_are_png(self, smoke_runs, name):
        with open(os.path.join(smoke_runs[0], name), "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.parametrize("name", ANIMATIONS)
    def test_animations_are_small_gifs(self, smoke_runs, name):
        path = os.path.join(smoke_runs[0], name)
        with open(path, "rb") as handle:
            assert handle.read(6) in (b"GIF87a", b"GIF89a")
        assert os.path.getsize(path) <= 2 * 1024 * 1024

    def test_header_is_exactly_the_specified_columns(self, smoke_runs):
        header, rows = read_rows(smoke_runs[0])
        assert header == EXPECTED_HEADER
        assert rows

    def test_every_row_is_valid(self, smoke_runs):
        _, rows = read_rows(smoke_runs[0])
        assert {r["valid"] for r in rows} == {"True"}

    def test_ek_and_pr_agree_or_blank(self, smoke_runs):
        _, rows = read_rows(smoke_runs[0])
        assert {r["ek_pr_agree"] for r in rows} <= {"True", ""}
        assert any(r["ek_pr_agree"] == "True" for r in rows)

    def test_every_experiment_is_present(self, smoke_runs):
        _, rows = read_rows(smoke_runs[0])
        assert {r["experiment"] for r in rows} == {"scaling", "sparse_vs_dense", "capacity_dist",
                                                   "memory", "matching", "segmentation",
                                                   "terminal_sensitivity"}

    def test_memory_rows_have_memory_and_no_time(self, smoke_runs):
        _, rows = read_rows(smoke_runs[0])
        memory = [r for r in rows if r["experiment"] == "memory"]
        assert memory
        assert all(r["peak_mem_kb"] and not r["mean_time_s"] for r in memory)

    def test_same_seed_gives_identical_non_timing_columns(self, smoke_runs):
        _, first = read_rows(smoke_runs[0])
        _, second = read_rows(smoke_runs[1])
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert {k: v for k, v in a.items() if k not in TIMING} == \
                   {k: v for k, v in b.items() if k not in TIMING}


class TestCommandLine:
    """The command line solves the sample inputs and prints the known answers."""

    @pytest.mark.parametrize("flag", ["--ek", "--pr"])
    def test_max_flow_sample_prints_23(self, flag):
        completed = run_cli(flag, "--input", SAMPLE_GRAPH)
        assert completed.returncode == 0, completed.stderr
        assert "23" in completed.stdout

    def test_min_cost_sample_prints_flow_and_cost(self):
        completed = run_cli("--mcf", "--input", SAMPLE_COSTS)
        assert completed.returncode == 0, completed.stderr
        assert "6" in completed.stdout and "24" in completed.stdout

    @pytest.mark.parametrize("flag", ["--match", "--schedule", "--segment"])
    def test_applications_run_on_built_in_examples(self, flag):
        completed = run_cli(flag)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip()

    def test_solver_flags_are_mutually_exclusive(self):
        assert run_cli("--ek", "--pr").returncode != 0
