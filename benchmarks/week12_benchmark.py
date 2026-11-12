#!/usr/bin/env python3
"""Week 12 benchmark: move the problem where it is easy, and measure the round trip.

Run from the repository root with the project virtualenv active::

    python benchmarks/week12_benchmark.py                 # the full study (under 5 minutes)
    python benchmarks/week12_benchmark.py --smoke         # seconds, tiny sizes
    python benchmarks/week12_benchmark.py --seed 7 --out /tmp/w12

Six experiments:

1. **FFT runtime** - the direct DFT, the recursive and iterative FFTs and
   numpy's FFT (a library reference only) for N = 2^6 to 2^20.
2. **FFT round-trip error** - ``max|ifft(fft(x)) - x| / max|x|``.
3. **Matrix multiply runtime** - classical, blocked and Strassen against
   numpy's ``@`` (a library reference only), n = 32 to 1024.
4. **Stability** - Hilbert matrices n = 2 to 12: condition numbers against
   numpy's (a library reference only), the relative error of
   ``solve(H, H @ ones)`` with and without pivoting, and the orthogonality
   loss of classical and modified Gram-Schmidt and Householder QR.
5. **Compression** - PSNR and the fraction of non-zero quantized
   coefficients at seven JPEG qualities on the synthetic test scene.
6. **Filters** - energy left at 120 Hz after a low-pass at 80 Hz, and at
   50 Hz after a high-pass at 80 Hz, on the two-tone test signal.

numpy and scipy routines are used here only as labelled library references
and as the accuracy check for the hand-written transforms and solvers.

**The 20-second rule.** A size whose single run passes 20 s, or is predicted
to from the measured growth, is not run and its row is marked
``capped = True``; the classical triple loop is capped above n = 256 as the
specification sets.

**Reproducibility.** Every random input comes from
``derive_seed(master_seed, <experiment>, n, rep)``.

**Where the results go.** ``benchmarks/results/week12/``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import signal as os_signal  # the standard library module; the package has its own "signal"
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.fft_analyzer.applications.audio_processor import make_example_wav, process  # noqa: E402
from src.fft_analyzer.applications.image_processor import load_image, make_test_scene  # noqa: E402
from src.fft_analyzer.core.fft import dft, fft2, fft_iterative, fft_recursive, ifft  # noqa: E402
from src.fft_analyzer.core.matrix import (  # noqa: E402
    matmul_blocked,
    matmul_classical,
    qr_gram_schmidt,
    qr_householder,
    solve,
    strassen,
)
from src.fft_analyzer.core.numerical import cond2_power, cond_estimate, hilbert  # noqa: E402
from src.fft_analyzer.image.compression import compress  # noqa: E402
from src.fft_analyzer.image.filters import fft_denoise, gaussian_blur, sobel  # noqa: E402
from src.fft_analyzer.signal.analysis import spectrogram  # noqa: E402
from src.fft_analyzer.signal.filters import fft_filter  # noqa: E402
from src.fft_analyzer.signal.generators import chirp, sine  # noqa: E402
from src.fft_analyzer.visualization.animations import animate_fft_stages, animate_filtering  # noqa: E402
from src.fft_analyzer.visualization.plots import (  # noqa: E402
    plot_compression_tradeoff,
    plot_fft2_heatmap,
    plot_image_grid,
    plot_spectrogram,
)
from src.utils.random_utils import DEFAULT_SEED, derive_seed  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    apply_house_style,
    plot_fft_roundtrip_error,
    plot_fft_runtime,
    plot_matrix_runtime,
    plot_stability,
)

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week12")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "src", "fft_analyzer", "examples")

#: The CSV's columns are fixed by the specification.
COLUMNS = [
    "experiment", "method", "n", "runs", "mean_time_s", "std_time_s", "max_rel_error", "cond",
    "orth_loss", "quality", "psnr_db", "nonzero_ratio", "energy_left", "capped", "seed",
]
TIMING_COLUMNS = ["mean_time_s", "std_time_s"]


@dataclass
class Config:
    """Every size and count of one benchmark mode."""

    fft_exponents: List[int]
    dft_max_exponent: int
    fft_runs: int
    matrix_sizes: List[int]
    classical_max: int
    matrix_runs_small: int
    matrix_runs_large: int
    hilbert_sizes: List[int]
    qualities: List[int]
    cap_s: float


FULL = Config(fft_exponents=list(range(6, 21)), dft_max_exponent=12, fft_runs=3,
              matrix_sizes=[32, 64, 128, 256, 512, 1024], classical_max=256, matrix_runs_small=3,
              matrix_runs_large=1, hilbert_sizes=list(range(2, 13)), qualities=[5, 10, 25, 50, 75, 90, 95],
              cap_s=20.0)
SMOKE = Config(fft_exponents=list(range(6, 13)), dft_max_exponent=10, fft_runs=2,
               matrix_sizes=[16, 32, 64], classical_max=64, matrix_runs_small=2, matrix_runs_large=1,
               hilbert_sizes=list(range(2, 6)), qualities=[10, 90], cap_s=20.0)


class RunCapExceeded(Exception):
    """Raised by the alarm when one run passes the cap."""


def capped(function: Callable[[], Any], cap_s: float) -> Any:
    """Call ``function``, raising :class:`RunCapExceeded` if it passes ``cap_s`` seconds."""
    def on_alarm(signum: int, frame: Any) -> None:
        raise RunCapExceeded()

    previous = os_signal.signal(os_signal.SIGALRM, on_alarm)
    os_signal.setitimer(os_signal.ITIMER_REAL, cap_s)
    try:
        return function()
    finally:
        os_signal.setitimer(os_signal.ITIMER_REAL, 0)
        os_signal.signal(os_signal.SIGALRM, previous)


def time_runs(function: Callable[[], Any], runs: int, cap_s: float) -> Tuple[Any, List[float]]:
    """Run ``function`` ``runs`` times under the cap; return the last result and the times."""
    times, result = [], None
    for _ in range(runs):
        started = time.perf_counter()
        result = capped(function, cap_s)
        times.append(time.perf_counter() - started)
    return result, times


def fmt(value: Any, digits: int = 8) -> str:
    """Format a cell; ``None`` becomes empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def row(experiment: str, method: str, n: Any, seed: Any = None, **fields: Any) -> dict:
    """One CSV row with every column present."""
    entry = {c: None for c in COLUMNS}
    entry.update(experiment=experiment, method=method, n=n, seed=seed)
    entry.update(fields)
    return entry


def timing_fields(times: Sequence[float]) -> dict:
    return {"runs": len(times), "mean_time_s": statistics.fmean(times),
            "std_time_s": statistics.stdev(times) if len(times) > 1 else None}


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """``max|a - b| / max|b|``, the error measure used throughout."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


# ----------------------------------------------------------------------
# Experiments 1 and 2: FFT runtime and round trip
# ----------------------------------------------------------------------
def fft_study(cfg: Config, master: int, rows: List[dict]) -> None:
    print("\n=== EXPERIMENTS 1-2 - FFT runtime and round-trip error ===", flush=True)
    methods: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "dft": dft, "fft_recursive": fft_recursive, "fft_iterative": fft_iterative,
        "numpy_reference": np.fft.fft,
    }
    active = {m: True for m in methods}
    for e in cfg.fft_exponents:
        n = 2 ** e
        seed = derive_seed(master, "fft", n, 0)
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        reference = np.fft.fft(x)
        parts = []
        for method, transform in methods.items():
            if method == "dft" and e > cfg.dft_max_exponent:
                continue
            if not active[method]:
                rows.append(row("fft_runtime", method, n, seed, capped=True))
                continue
            try:
                capped(lambda: transform(x), cfg.cap_s)  # untimed warm-up, as in experiment 3
                result, times = time_runs(lambda: transform(x), cfg.fft_runs, cfg.cap_s)
            except RunCapExceeded:
                active[method] = False
                rows.append(row("fft_runtime", method, n, seed, capped=True))
                continue
            error = None if method == "numpy_reference" else rel_err(result, reference)
            rows.append(row("fft_runtime", method, n, seed, max_rel_error=error, capped=False,
                            **timing_fields(times)))
            parts.append(f"{method} {statistics.fmean(times) * 1e3:9.3f} ms")
            if method != "dft":
                if method == "numpy_reference":
                    back = np.fft.ifft(result)
                else:
                    back = ifft(result, method="recursive" if method == "fft_recursive" else "iterative")
                rows.append(row("fft_roundtrip", method, n, seed, max_rel_error=rel_err(back, x)))
        print(f"    N = 2^{e:<2} " + "  ".join(parts), flush=True)


# ----------------------------------------------------------------------
# Experiment 3: matrix multiply
# ----------------------------------------------------------------------
def matrix_study(cfg: Config, master: int, rows: List[dict], caps: List[dict]) -> None:
    print("\n=== EXPERIMENT 3 - Matrix multiply ===", flush=True)
    methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "classical": matmul_classical, "blocked": matmul_blocked,
        "strassen": lambda A, B: strassen(A, B, cutoff=64), "numpy_reference": lambda A, B: A @ B,
    }
    active = {m: True for m in methods}
    last: Dict[str, Tuple[int, float]] = {}
    for n in cfg.matrix_sizes:
        seed = derive_seed(master, "matmul", n, 0)
        rng = np.random.default_rng(seed)
        A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        reference = A @ B
        runs = cfg.matrix_runs_small if n <= 256 else cfg.matrix_runs_large
        parts = []
        for method, multiply in methods.items():
            reason = None
            if method == "classical" and n > cfg.classical_max:
                reason = f"classical triple loop capped above n = {cfg.classical_max}"
            elif not active[method]:
                reason = "a smaller size passed the cap"
            elif method in last:
                n0, t0 = last[method]
                predicted = t0 * (n / n0) ** 3
                if predicted > cfg.cap_s:
                    reason = f"predicted {predicted:.0f} s from n = {n0}"
            if reason is None:
                try:
                    # One untimed warm-up call per method and size, so first-call costs
                    # (numpy's BLAS starting its threads, for one) stay out of the mean.
                    capped(lambda: multiply(A, B), cfg.cap_s)
                    # The library reference takes milliseconds, so it gets at least
                    # five runs at every size; one run of it is mostly noise.
                    method_runs = max(runs, 5) if method == "numpy_reference" else runs
                    result, times = time_runs(lambda: multiply(A, B), method_runs, cfg.cap_s)
                except RunCapExceeded:
                    reason = f"a run passed {cfg.cap_s:.0f} s"
            if reason is not None:
                active[method] = False
                rows.append(row("matrix_runtime", method, n, seed, capped=True))
                caps.append({"method": method, "n": n, "reason": reason})
                continue
            last[method] = (n, statistics.fmean(times))
            error = None if method == "numpy_reference" else rel_err(result, reference)
            rows.append(row("matrix_runtime", method, n, seed, max_rel_error=error, capped=False,
                            **timing_fields(times)))
            parts.append(f"{method} {statistics.fmean(times):8.4f} s")
        print(f"    n = {n:<5} " + "  ".join(parts), flush=True)
    for cap in caps:
        print(f"    [cap] {cap['method']} n = {cap['n']}: {cap['reason']}", flush=True)


# ----------------------------------------------------------------------
# Experiment 4: stability on Hilbert matrices
# ----------------------------------------------------------------------
def stability_study(cfg: Config, rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    print("\n=== EXPERIMENT 4 - Stability (Hilbert matrices) ===", flush=True)
    cond_rows, solve_rows, orth_rows = [], [], []
    for n in cfg.hilbert_sizes:
        H = hilbert(n)
        conds = {"cond_estimate_1": cond_estimate(H, 1), "numpy_cond_1": float(np.linalg.cond(H, 1)),
                 "cond2_power": cond2_power(H), "numpy_cond_2": float(np.linalg.cond(H))}
        for method, value in conds.items():
            rows.append(row("stability_cond", method, n, cond=value))
            cond_rows.append({"method": method, "n": n, "cond": value})
        ones = np.ones(n)
        b = H @ ones
        for method, pivot in (("lu_pivot", True), ("lu_no_pivot", False)):
            try:
                x = solve(H, b, pivot=pivot)
                error = float(np.max(np.abs(x - ones)))
            except ValueError:
                error = None
            rows.append(row("stability_solve", method, n, max_rel_error=error, cond=conds["numpy_cond_2"]))
            solve_rows.append({"method": method, "n": n, "max_rel_error": error, "cond": conds["numpy_cond_2"]})
        for method, factor in (("cgs", lambda A: qr_gram_schmidt(A, modified=False)),
                               ("mgs", lambda A: qr_gram_schmidt(A, modified=True)),
                               ("householder", qr_householder)):
            Q, _ = factor(H)
            loss = float(np.sqrt(np.sum((Q.T @ Q - np.eye(n)) ** 2)))
            rows.append(row("stability_orth", method, n, orth_loss=loss))
            orth_rows.append({"method": method, "n": n, "orth_loss": loss})
        errs = [r["max_rel_error"] for r in solve_rows[-2:]]
        shown = " / ".join("n/a" if e is None else f"{e:9.2e}" for e in errs)
        print(f"    n = {n:<3} cond2 {conds['numpy_cond_2']:10.3e}  ours {conds['cond2_power']:10.3e}  "
              f"solve err {shown}  "
              f"orth CGS {orth_rows[-3]['orth_loss']:8.2e} MGS {orth_rows[-2]['orth_loss']:8.2e} "
              f"HH {orth_rows[-1]['orth_loss']:8.2e}", flush=True)
    return cond_rows, solve_rows, orth_rows


# ----------------------------------------------------------------------
# Experiments 5 and 6: compression and filters
# ----------------------------------------------------------------------
def compression_study(cfg: Config, rows: List[dict], scene: np.ndarray) -> List[dict]:
    print("\n=== EXPERIMENT 5 - Compression ===", flush=True)
    out = []
    for q in cfg.qualities:
        started = time.perf_counter()
        _, ratio, quality_db = compress(scene, q)
        elapsed = time.perf_counter() - started
        rows.append(row("compression", "dct_jpeg", scene.shape[0], quality=q, psnr_db=quality_db,
                        nonzero_ratio=ratio, runs=1, mean_time_s=elapsed))
        out.append({"quality": q, "psnr": quality_db, "nonzero_ratio": ratio})
        print(f"    quality {q:<3} PSNR {quality_db:6.2f} dB  nonzero {ratio:.4f}", flush=True)
    return out


def filter_study(rows: List[dict]) -> None:
    print("\n=== EXPERIMENT 6 - Filters ===", flush=True)
    fs, duration, guard = 1000, 1.0, 50
    x = sine(50, fs, duration) + 0.5 * sine(120, fs, duration)

    def tone_energy(values: np.ndarray, freq: int) -> float:
        """``|Y(freq)|^2`` at one bin, by the DFT definition; ``freq`` is an exact bin."""
        n = values.shape[0]
        k = freq * n // fs
        return float(abs(np.dot(values, np.exp(-2j * np.pi * k * np.arange(n) / n))) ** 2)

    # Energy left is |Y(f)|^2 / |X(f)|^2 at the removed tone's bin, over the full
    # 1,000-sample record and over its interior (50 samples dropped at each end).
    for method, kind, removed in (("lowpass_80hz", "low", 120), ("highpass_80hz", "high", 50)):
        started = time.perf_counter()
        y = fft_filter(x, fs, kind, 80)
        elapsed = time.perf_counter() - started
        full = tone_energy(y, removed) / tone_energy(x, removed)
        inner = tone_energy(y[guard:-guard], removed) / tone_energy(x[guard:-guard], removed)
        rows.append(row("filters", method, len(x), runs=1, mean_time_s=elapsed, energy_left=full))
        rows.append(row("filters", method + "_interior", len(x) - 2 * guard, energy_left=inner))
        print(f"    {method:<14} energy left at {removed} Hz: full record {full:.3e}, interior {inner:.3e}",
              flush=True)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def figures(out: str, rows: List[dict], caps: List[dict], stability: Tuple[List[dict], List[dict], List[dict]],
            compression_rows: List[dict], scene: np.ndarray, master: int) -> None:
    print("\n=== figures, animations and audio ===", flush=True)
    by = lambda experiment: [r for r in rows if r["experiment"] == experiment and not r.get("capped")]  # noqa: E731
    builds = [
        ("fft_runtime.png", lambda p: plot_fft_runtime(by("fft_runtime"), save_path=p)),
        ("fft_roundtrip_error.png", lambda p: plot_fft_roundtrip_error(by("fft_roundtrip"), save_path=p)),
        ("matrix_runtime.png", lambda p: plot_matrix_runtime(by("matrix_runtime"), caps, save_path=p)),
        ("stability.png", lambda p: plot_stability(*stability, save_path=p)),
        ("compression_tradeoff.png", lambda p: plot_compression_tradeoff(compression_rows, save_path=p)),
    ]
    fs = 1000
    sweep = chirp(100, 400, fs, 2.0)
    freqs, times, magnitude = spectrogram(sweep, fs, seg=256, hop=128)
    builds.append(("spectrogram.png", lambda p: plot_spectrogram(freqs, times, magnitude, save_path=p)))
    builds.append(("fft2_heatmap.png", lambda p: plot_fft2_heatmap(scene, save_path=p)))
    q10, _, _ = compress(scene, 10)
    q50, _, _ = compress(scene, 50)
    images = [scene, gaussian_blur(scene, 2.0), sobel(scene), fft_denoise(scene, 40), q10, q50]
    titles = ["original (synthetic)", "Gaussian blur, sigma 2", "Sobel gradient magnitude",
              "FFT denoise, radius 40", "JPEG-style, quality 10", "JPEG-style, quality 50"]
    builds.append(("image_filters.png", lambda p: plot_image_grid(images, titles, save_path=p)))
    for name, build in builds:
        path = os.path.join(out, name)
        figure = build(path)
        plt.close(figure)
        print(f"  wrote {name} ({os.path.getsize(path) / 1024:.0f} KiB)", flush=True)
    rng = np.random.default_rng(derive_seed(master, "fft_stages", 16, 0))
    animate_fft_stages(rng.standard_normal(16), os.path.join(out, "fft_stages.gif"))
    two_tone = sine(50, fs, 1.0) + 0.5 * sine(120, fs, 1.0)
    animate_filtering(two_tone, fs, os.path.join(out, "filtering.gif"))
    wav = os.path.join(EXAMPLES_DIR, "tone_noise.wav")
    if not os.path.exists(wav):
        make_example_wav(wav, seed=42)
    summary = process(wav, "low", 1000, out_path=os.path.join(out, "audio_filtered.wav"))
    for name in ("fft_stages.gif", "filtering.gif", "audio_filtered.wav"):
        print(f"  wrote {name} ({os.path.getsize(os.path.join(out, name)) / 1024:.0f} KiB)", flush=True)
    print(f"  audio peaks before {summary['peaks_before']} after {summary['peaks_after']}", flush=True)


def write_csv(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: fmt(r.get(c)) for c in COLUMNS})


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny parameter set, seconds")
    parser.add_argument("--out", default=RESULTS_DIR, help="directory for the CSV and figures")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed (default 42)")
    args = parser.parse_args(argv)

    cfg = SMOKE if args.smoke else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    master = args.seed
    apply_house_style()

    print("=" * 78)
    print("CSC 5300 Week 12 - fast transforms benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    : {platform.platform()}")
    print(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    print(f"  master seed : {master}  (inputs: derive_seed(master, experiment, n, rep))")
    print(f"  output      : {out}", flush=True)

    started = time.perf_counter()
    rows: List[dict] = []
    caps: List[dict] = []
    scene_path = os.path.join(EXAMPLES_DIR, "test_scene.png")
    if not os.path.exists(scene_path):
        make_test_scene(scene_path, seed=42)
    scene = load_image(scene_path)
    fft_study(cfg, master, rows)
    matrix_study(cfg, master, rows, caps)
    stability = stability_study(cfg, rows)
    compression_rows = compression_study(cfg, rows, scene)
    filter_study(rows)

    print("\n=== writing results ===", flush=True)
    write_csv(os.path.join(out, "benchmark.csv"), rows)
    print(f"  {len(rows)} rows; {sum(1 for r in rows if r.get('capped'))} capped", flush=True)
    figures(out, rows, caps, stability, compression_rows, scene, master)

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
