"""Smoke tests for the Week 13 benchmark (spec sections 8 and 9.8).

The benchmark runs end to end in a subprocess, as a user would run it
(``benchmarks/week13_benchmark.py --smoke``), twice with the same master
seed into separate pytest temporary directories. The tests check that every
figure and the CSV are written, the header is exactly the section 8 list,
caps are recorded as rows, the non-timing columns repeat exactly under the
same seed, and ``--redraw`` rebuilds the figures without touching the CSV.
One test checks that the smoke sizes stay inside the section 9.8 limits,
and one in-process test checks that the 20-second alarm is armed outside
the timed region (an event order, not a duration). There are no timing
assertions.
"""

from __future__ import annotations

import csv
import os
import signal
import subprocess
import sys
import time
import types

import pytest

import src.advds.benchmarks.performance as performance
import src.utils.timer as timer_module

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK = os.path.join(REPO_ROOT, "benchmarks", "week13_benchmark.py")

#: Spec section 8, fixed.
EXPECTED_HEADER = [
    "experiment", "structure", "method", "n", "density", "runs", "mean_time_s", "std_time_s", "ops",
    "nodes_created", "memory_kib", "overhead_ratio", "bits_per_symbol", "compression_ratio", "cache_M",
    "cache_B", "misses", "blocks_per_search", "capped", "seed",
]
FIGURES = ["segment_vs_fenwick.png", "persistent_overhead.png", "rank_select.png", "wavelet_compression.png",
           "cache_misses.png", "layout_blocks.png", "vcs_history.png"]
TIMING = {"mean_time_s", "std_time_s"}
EXPERIMENTS = {"segment_vs_fenwick", "persistent_overhead", "rank_select", "wavelet_dna", "wavelet_text",
               "cache_misses", "matrix_time", "layout_blocks", "vcs_history"}


def run_benchmark(*args: str) -> subprocess.CompletedProcess:
    """Run the wrapper with ``args``; fail with its output on a non-zero exit."""
    completed = subprocess.run([sys.executable, BENCHMARK, *args], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    return completed


def read_rows(out_dir: str) -> tuple:
    """Return the CSV header and its rows as dicts of raw text."""
    with open(os.path.join(out_dir, "benchmark.csv"), newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke runs with the same master seed, in separate directories, and their stdout."""
    first = str(tmp_path_factory.mktemp("week13_smoke_a"))
    second = str(tmp_path_factory.mktemp("week13_smoke_b"))
    out_a = run_benchmark("--smoke", "--out", first, "--seed", "42").stdout
    out_b = run_benchmark("--smoke", "--out", second, "--seed", "42").stdout
    return first, second, out_a, out_b


@pytest.mark.parametrize("name", FIGURES + ["benchmark.csv"])
def test_outputs_exist_and_are_non_empty(smoke_runs, name):
    path = os.path.join(smoke_runs[0], name)
    assert os.path.isfile(path) and os.path.getsize(path) > 0


@pytest.mark.parametrize("name", FIGURES)
def test_figures_are_png_under_100_kb(smoke_runs, name):
    path = os.path.join(smoke_runs[0], name)
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    assert os.path.getsize(path) <= 100_000


def test_header_is_exactly_the_specified_columns(smoke_runs):
    header, rows = read_rows(smoke_runs[0])
    assert header == EXPECTED_HEADER
    assert rows


def test_every_row_has_every_column(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    assert all(list(r) == EXPECTED_HEADER for r in rows)


def test_every_experiment_is_present(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    assert {r["experiment"] for r in rows} == EXPERIMENTS


def test_smoke_sizes_respect_the_spec_limits(smoke_runs):
    """Spec 9.8: n <= 2^12, DNA <= 10^4, transpose n <= 32, multiply n <= 16, h <= 10 (capped rows included)."""
    _, rows = read_rows(smoke_runs[0])

    def sizes(experiment, structure=None):
        found = [int(r["n"]) for r in rows
                 if r["experiment"] == experiment and (structure is None or r["structure"] == structure)]
        assert found, (experiment, structure)
        return found

    for experiment in ("segment_vs_fenwick", "persistent_overhead", "rank_select"):
        assert max(sizes(experiment)) <= 2 ** 12
    assert max(sizes("wavelet_dna")) <= 10 ** 4
    for experiment in ("cache_misses", "matrix_time"):
        assert max(sizes(experiment, "transpose")) <= 32
        assert max(sizes(experiment, "matmul")) <= 16
    assert max(sizes("layout_blocks")) <= 2 ** 10 - 1  # n is the key count 2^h - 1


class _RecordingSignal:
    """The real :mod:`signal` module, with every ``setitimer`` call recorded as "arm" or "disarm"."""

    def __init__(self, events):
        self._events = events

    def __getattr__(self, name):
        return getattr(signal, name)

    def setitimer(self, which, seconds, *rest):
        self._events.append("arm" if seconds else "disarm")
        return signal.setitimer(which, seconds, *rest)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="the 20 s cap needs SIGALRM")
def test_cap_alarm_is_armed_outside_the_timed_region(monkeypatch):
    """Regression: arming SIGALRM inside the clock added about 10 microseconds to every timing.

    The numpy references take a few microseconds per call, so their times
    were inflated two to four times. Each timed call must now read the
    clock, run, and read the clock again with no alarm set-up in between.
    This checks the order of events, not any duration.
    """
    events = []
    real_clock = time.perf_counter

    def clock():
        events.append("clock")
        return real_clock()

    monkeypatch.setattr(performance, "os_signal", _RecordingSignal(events))
    monkeypatch.setattr(timer_module, "time", types.SimpleNamespace(perf_counter=clock))
    result, stats = performance.measure(lambda: events.append("call") or 7, 3, 20.0)
    assert result == 7 and stats["runs"] == 3
    warm_up = ["arm", "call", "disarm"]
    timed = ["arm", "clock", "call", "clock", "disarm"]
    assert events == warm_up + timed * 3


def test_capped_rows_are_present_and_blank(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    caps = [r for r in rows if r["capped"] == "True"]
    assert caps, "the smoke configuration has configured limits, so capped rows must appear"
    measured = ["runs", "mean_time_s", "std_time_s", "ops", "nodes_created", "memory_kib", "overhead_ratio",
                "bits_per_symbol", "compression_ratio", "misses", "blocks_per_search"]
    for r in caps:
        assert all(r[c] == "" for c in measured), r
    assert {r["capped"] for r in rows} <= {"True", "False"}


def test_cap_reasons_are_printed(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    printed = sum(line.strip().startswith("[cap] ") for line in smoke_runs[2].splitlines())
    assert printed == sum(r["capped"] == "True" for r in rows)


def test_references_are_labelled_and_measured_rows_have_times(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    structures = {r["structure"] for r in rows}
    assert "naive_reference" in structures
    assert any(r["method"] == "numpy_reference" for r in rows)
    timed = [r for r in rows if r["runs"]]
    assert timed and all(float(r["mean_time_s"]) > 0 for r in timed)
    assert all(int(r["runs"]) >= 5 for r in rows if r["method"] == "numpy_reference" and r["capped"] == "False")


def test_random_inputs_carry_their_seed(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    seeded = {"segment_vs_fenwick", "persistent_overhead", "rank_select", "layout_blocks", "vcs_history"}
    for r in rows:
        if r["runs"] or r["experiment"] in seeded:
            assert r["seed"].isdigit(), r
        if r["experiment"] == "cache_misses":
            assert r["seed"] == "", r  # a simulated trace has no random input


def test_same_seed_gives_identical_non_timing_columns(smoke_runs):
    _, first = read_rows(smoke_runs[0])
    _, second = read_rows(smoke_runs[1])
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert {k: v for k, v in a.items() if k not in TIMING} == {k: v for k, v in b.items() if k not in TIMING}


def test_redraw_rebuilds_figures_without_touching_the_csv(smoke_runs):
    """``--redraw`` redraws every figure from the CSV and leaves the CSV byte for byte."""
    out = smoke_runs[1]
    csv_path = os.path.join(out, "benchmark.csv")
    with open(csv_path, "rb") as handle:
        before = handle.read()
    for name in FIGURES:
        os.remove(os.path.join(out, name))
    run_benchmark("--redraw", "--out", out)
    with open(csv_path, "rb") as handle:
        assert handle.read() == before
    for name in FIGURES:
        path = os.path.join(out, name)
        assert os.path.getsize(path) > 0
        with open(path, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
