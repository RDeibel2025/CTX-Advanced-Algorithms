"""Smoke tests for the Week 12 benchmark (spec section 9.8).

The benchmark runs end to end in a subprocess, as a user would run it, and
must write every figure, animation, audio file and table the report reads.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK = os.path.join(REPO_ROOT, "benchmarks", "week12_benchmark.py")

EXPECTED_HEADER = [
    "experiment", "method", "n", "runs", "mean_time_s", "std_time_s", "max_rel_error", "cond",
    "orth_loss", "quality", "psnr_db", "nonzero_ratio", "energy_left", "capped", "seed",
]
FIGURES = ["fft_runtime.png", "fft_roundtrip_error.png", "matrix_runtime.png", "stability.png",
           "compression_tradeoff.png", "spectrogram.png", "fft2_heatmap.png", "image_filters.png"]
OTHER = ["fft_stages.gif", "filtering.gif", "audio_filtered.wav", "benchmark.csv"]
TIMING = {"mean_time_s", "std_time_s"}


def run_smoke(out_dir: str) -> None:
    """Run the benchmark's smoke mode into ``out_dir``; fail with its output on error."""
    completed = subprocess.run(
        [sys.executable, BENCHMARK, "--smoke", "--out", out_dir, "--seed", "42"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]


def read_rows(out_dir: str) -> tuple:
    with open(os.path.join(out_dir, "benchmark.csv"), newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke runs with the same master seed, in separate directories."""
    first = str(tmp_path_factory.mktemp("week12_smoke_a"))
    second = str(tmp_path_factory.mktemp("week12_smoke_b"))
    run_smoke(first)
    run_smoke(second)
    return first, second


@pytest.mark.parametrize("name", FIGURES + OTHER)
def test_outputs_exist_and_are_non_empty(smoke_runs, name):
    path = os.path.join(smoke_runs[0], name)
    assert os.path.isfile(path) and os.path.getsize(path) > 0


@pytest.mark.parametrize("name", FIGURES)
def test_figures_are_png(smoke_runs, name):
    with open(os.path.join(smoke_runs[0], name), "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("name", ["fft_stages.gif", "filtering.gif"])
def test_animations_are_small_gifs(smoke_runs, name):
    path = os.path.join(smoke_runs[0], name)
    with open(path, "rb") as handle:
        assert handle.read(6) in (b"GIF87a", b"GIF89a")
    assert os.path.getsize(path) <= 2 * 1024 * 1024


def test_audio_is_a_wav_file(smoke_runs):
    with open(os.path.join(smoke_runs[0], "audio_filtered.wav"), "rb") as handle:
        header = handle.read(12)
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def test_header_is_exactly_the_specified_columns(smoke_runs):
    header, rows = read_rows(smoke_runs[0])
    assert header == EXPECTED_HEADER
    assert rows


def test_every_max_rel_error_is_below_1e_8(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    errors = [float(r["max_rel_error"]) for r in rows if r["max_rel_error"]]
    assert errors and max(errors) < 1e-8


def test_every_experiment_is_present(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    assert {r["experiment"] for r in rows} == {"fft_runtime", "fft_roundtrip", "matrix_runtime",
                                               "stability_cond", "stability_solve", "stability_orth",
                                               "compression", "filters"}


def test_library_references_are_labelled(smoke_runs):
    _, rows = read_rows(smoke_runs[0])
    methods = {r["method"] for r in rows}
    assert "numpy_reference" in methods
    assert not any(r["max_rel_error"] for r in rows if r["method"] == "numpy_reference"
                   and r["experiment"] in {"fft_runtime", "matrix_runtime"})


def test_same_seed_gives_identical_non_timing_columns(smoke_runs):
    _, first = read_rows(smoke_runs[0])
    _, second = read_rows(smoke_runs[1])
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert {k: v for k, v in a.items() if k not in TIMING} == \
               {k: v for k, v in b.items() if k not in TIMING}


def test_redraw_rebuilds_figures_without_touching_the_csv(smoke_runs):
    """``--redraw`` redraws every output from the CSV and leaves the CSV byte for byte."""
    out = smoke_runs[1]
    csv_path = os.path.join(out, "benchmark.csv")
    with open(csv_path, "rb") as handle:
        before = handle.read()
    for name in FIGURES:
        os.remove(os.path.join(out, name))
    completed = subprocess.run([sys.executable, BENCHMARK, "--redraw", "--out", out],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    with open(csv_path, "rb") as handle:
        assert handle.read() == before
    for name in FIGURES + OTHER:
        assert os.path.getsize(os.path.join(out, name)) > 0
