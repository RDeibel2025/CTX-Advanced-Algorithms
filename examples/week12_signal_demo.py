#!/usr/bin/env python3
"""Runnable demonstration of the Week 12 signal tools: the FFT, filters and spectra.

    python examples/week12_signal_demo.py

Three sections. Apart from the times, the output is the same on every run,
because every random input comes from a fixed seed through ``derive_seed``:

1. **The FFT** (spec 9.1) - the hand anchors, both radix-2 FFTs against the
   O(N^2) DFT, both at N = 2^20 with the round trip, circular and linear
   convolution, the 2-D FFT, and the ``ValueError`` for a length that is not a
   power of two.
2. **Filters and spectral analysis** (spec 9.4) - the windows, the two-tone
   signal through low, high and band filters, the pink and white noise
   slopes, the chirp's spectrogram and the autocorrelation. The chirp's
   spectrum and spectrogram are saved to ``benchmarks/results/week12/`` as
   ``signal_demo_spectrum.png`` and ``signal_demo_spectrogram.png``.
3. **Applications** (spec 9.5) - the weekly cycle of the synthetic series,
   the WAV anchors, the image pipeline's quality ordering, and the standard
   library ``signal`` module after the package is imported.

Each section reads its results through one idea: every tool here moves the
problem where it is easy (the FFT), does the easy thing there (multiply,
mask or read off bins), and moves back (the inverse FFT). The anchor values
come from section 9 of the Week 12 build spec, and every anchor printed is
also asserted, so the script exits non-zero rather than printing something
untrue. The rulers here are the DFT definition evaluated directly, numpy's
``convolve``, ``correlate`` and window functions, and values worked out by
hand; ``np.fft`` and ``scipy`` are the cross-check in ``tests/week12/`` only
(spec 1.1). It finishes in a few seconds.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Iterable

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.fft_analyzer.applications.audio_processor import (  # noqa: E402
    load_wav,
    make_example_wav,
    process,
    save_wav,
)
from src.fft_analyzer.applications.data_analyzer import (  # noqa: E402
    autocorr_peaks,
    detrend,
    dominant_period,
    make_series,
)
from src.fft_analyzer.applications.image_processor import run_pipeline  # noqa: E402
from src.fft_analyzer.core.fft import (  # noqa: E402
    circular_convolve,
    dft,
    fft,
    fft2,
    fft_iterative,
    fft_recursive,
    ifft,
    ifft2,
    linear_convolve,
    rfft_freqs,
)
from src.fft_analyzer.signal.analysis import (  # noqa: E402
    autocorrelation,
    peak_frequencies,
    spectrogram,
    welch_psd,
)
from src.fft_analyzer.signal.filters import blackman, fft_filter, hamming, hann  # noqa: E402
from src.fft_analyzer.signal.generators import (  # noqa: E402
    chirp,
    multi_tone,
    pink_noise,
    white_noise,
)
from src.fft_analyzer.visualization.plots import plot_spectrogram, plot_spectrum  # noqa: E402
from src.utils.random_utils import derive_seed  # noqa: E402
from src.utils.timer import Timer  # noqa: E402

WIDTH = 74
SEED = 42
OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week12")
EXAMPLES = os.path.join(REPO_ROOT, "src", "fft_analyzer", "examples")


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<40} {value}")


def fmt_seconds(seconds: float) -> str:
    """Seconds as a short string in s or ms."""
    if seconds >= 1:
        return f"{seconds:.2f} s"
    return f"{seconds * 1e3:.1f} ms"


def fmt_complex(values: Iterable[complex], digits: int = 9) -> str:
    """Complex values rounded to ``digits`` places, written like ``-2+2j``."""
    parts = []
    for z in np.asarray(list(values), dtype=complex):
        re, im = round(z.real, digits) + 0.0, round(z.imag, digits) + 0.0
        if im == 0.0:
            parts.append(f"{re:g}")
        elif re == 0.0:
            parts.append(f"{im:g}j")
        else:
            parts.append(f"{re:g}{im:+g}j")
    return "[" + ", ".join(parts) + "]"


def rng_for(*labels: Any) -> np.random.Generator:
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(SEED, "week12_signal_demo", *labels))


def rel_error(ours: np.ndarray, ref: np.ndarray) -> float:
    """``max|ours - ref| / max|ref|``."""
    return float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))


def dft_bin(x: np.ndarray, k: int, n: int) -> complex:
    """``X[k] = sum_t x[t] exp(-2 pi i k t / n)``, the definition for one bin.

    ``k t`` is reduced modulo ``n`` in integer arithmetic first, so the angle
    stays exact even when ``k t`` reaches 2^40.
    """
    t = np.arange(x.shape[0], dtype=np.int64)
    return complex(np.sum(x * np.exp(-2j * np.pi * ((k * t) % n) / n)))


def tone_power(y: np.ndarray, freq: int, fs: int) -> float:
    """``|Y(freq)|^2`` for a record whose length makes ``freq`` an exact bin."""
    n = y.shape[0]
    return abs(dft_bin(y, freq * n // fs, n)) ** 2


def tone_amplitude(y: np.ndarray, freq: int, fs: int) -> float:
    """Amplitude ``2 |Y(freq)| / N`` of the tone at ``freq`` Hz."""
    return 2.0 * tone_power(y, freq, fs) ** 0.5 / y.shape[0]


def loglog_slope(x: np.ndarray) -> float:
    """Least-squares slope of log10 Welch PSD against log10 f, 1e-3 <= f <= 0.4.

    The PSD uses fs = 1 (frequencies in cycles per sample) and segments of
    4096. The slope is the closed form ``sum (u - u_bar)(v - v_bar) /
    sum (u - u_bar)^2``.
    """
    freqs, density = welch_psd(x, 1.0, seg=4096)
    keep = (freqs >= 1e-3) & (freqs <= 0.4)
    u, v = np.log10(freqs[keep]), np.log10(density[keep])
    du = u - u.mean()
    return float(np.sum(du * (v - v.mean())) / np.sum(du * du))


def relpath(path: str) -> str:
    """``path`` relative to the repository root, for printing."""
    return os.path.relpath(path, REPO_ROOT)


# --------------------------------------------------------------------- the FFT


def section_fft() -> None:
    banner(1, "The FFT: the O(N^2) definition and two radix-2 FFTs")
    transforms = {"iterative": fft_iterative, "recursive": fft_recursive, "dft": dft}
    # Spec 9.1, by hand: with w = -i, X[1] = 1 - 2i - 3 + 4i = -2 + 2i, and
    # X[3] is its conjugate; X[0] = 10 and X[2] = 1 - 2 + 3 - 4 = -2.
    expected = np.array([10, -2 + 2j, -2, -2 - 2j])
    for transform in transforms.values():
        X = transform([1, 2, 3, 4])
        assert np.max(np.abs(X - expected)) < 1e-12
    show("fft([1, 2, 3, 4])", f"{fmt_complex(fft_iterative([1, 2, 3, 4]))} (all three agree)")
    impulse = np.zeros(8)
    impulse[0] = 1.0
    for transform in transforms.values():
        assert np.max(np.abs(transform(impulse) - 1.0)) < 1e-12
    show("fft of the impulse, N = 8", fmt_complex(fft_iterative(impulse)))
    # cos(2 pi 3 n / 16) is half of exp(+) plus half of exp(-): each sums to
    # 16 in one bin, so |X| = 8 at bins 3 and 16 - 3 = 13 and 0 elsewhere.
    n16 = np.arange(16)
    cosine = np.cos(2 * np.pi * 3 * n16 / 16)
    for name in ("iterative", "recursive"):
        mag = np.abs(transforms[name](cosine))
        assert abs(mag[3] - 8.0) < 1e-9 and abs(mag[13] - 8.0) < 1e-9
        assert np.max(np.delete(mag, [3, 13])) < 1e-9
    mag = np.abs(fft_iterative(cosine))
    show("cos(2 pi 3n/16): |X[3]|, |X[13]|", f"{mag[3]:.9f}, {mag[13]:.9f}")
    show("  largest |X| in the other 14 bins", f"{np.max(np.delete(mag, [3, 13])):.1e}")

    print()
    worst = {"iterative": 0.0, "recursive": 0.0}
    for exponent in range(1, 11):
        for trial in range(5):
            rng = rng_for("random", exponent, trial)
            x = rng.standard_normal(2**exponent) + 1j * rng.standard_normal(2**exponent)
            reference = dft(x)
            for name in worst:
                worst[name] = max(worst[name], rel_error(transforms[name](x), reference))
    assert max(worst.values()) < 1e-10
    show("50 seeded vectors, N = 2 .. 1024", "max relative error against the O(N^2) dft:")
    show("  iterative, recursive", f"{worst['iterative']:.1e}, {worst['recursive']:.1e}")

    n = 2**20
    rng = rng_for("big")
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    results = {}
    show("N = 2^20, one seeded complex vector", "round trip = max|ifft(fft(x)) - x|")
    for name in ("iterative", "recursive"):
        with Timer() as forward:
            X = fft(x, method=name)
        with Timer() as inverse:
            back = ifft(X, method=name)
        trip = float(np.max(np.abs(back - x)))
        assert X.shape == (n,) and trip < 1e-12
        results[name] = X
        show(f"N = 2^20, {name}", f"fft {fmt_seconds(forward.elapsed)}, ifft "
             f"{fmt_seconds(inverse.elapsed)}, round trip {trip:.1e}")
    agree = rel_error(results["recursive"], results["iterative"])
    assert agree < 1e-9
    show("  recursive against iterative", f"relative error {agree:.1e}")
    bins = [int(k) for k in rng_for("bins").integers(0, n, size=8)]
    scale = float(np.max(np.abs(results["iterative"])))
    by_definition = max(abs(results[name][k] - dft_bin(x, k, n)) / scale
                        for name in results for k in bins)
    assert by_definition < 1e-9
    show("  8 seeded bins against the definition", f"relative error {by_definition:.1e}")

    print()
    # Spec 9.1, by hand: c[k] = sum_j a[j] b[(k - j) mod 3] gives 4, 2.5, 2.5;
    # the linear convolution is 0, 1, 0.5 + 2, 1 + 3, 1.5.
    circ = circular_convolve([1, 2, 3], [0, 1, 0.5])
    assert np.max(np.abs(circ - [4, 2.5, 2.5])) < 1e-12
    show("circular_convolve([1,2,3], [0,1,0.5])", f"{fmt_complex(circ)} (length 3: dft path)")
    rng = rng_for("circular", 8)
    a, b = rng.standard_normal(8), rng.standard_normal(8)
    direct = np.array([sum(a[j] * b[(k - j) % 8] for j in range(8)) for k in range(8)])
    circ8 = float(np.max(np.abs(circular_convolve(a, b) - direct)))
    assert circ8 < 1e-9
    show("  seeded length 8 (radix-2 path)", f"max error against the definition {circ8:.1e}")
    lin = linear_convolve([1, 2, 3], [0, 1, 0.5])
    assert np.max(np.abs(lin - [0, 1, 2.5, 4, 1.5])) < 1e-12
    show("linear_convolve([1,2,3], [0,1,0.5])", fmt_complex(lin))
    lengths = rng_for("linear", "lengths").integers(1, 200, size=(30, 2))
    worst_lin = 0.0
    for trial, (m, k) in enumerate(lengths):
        pair = rng_for("linear", trial)
        a, b = pair.standard_normal(int(m)), pair.standard_normal(int(k))
        worst_lin = max(worst_lin, float(np.max(np.abs(linear_convolve(a, b) - np.convolve(a, b)))))
    assert worst_lin < 1e-9
    show("  30 seeded pairs against np.convolve", f"max error {worst_lin:.1e}")

    img = rng_for("fft2").standard_normal((64, 64))
    jk = np.outer(np.arange(64), np.arange(64)) % 64
    W = np.exp(-2j * np.pi * jk / 64)
    F = fft2(img)
    by_matrices = rel_error(F, W @ img @ W)
    trip2 = float(np.max(np.abs(ifft2(F) - img)))
    assert by_matrices < 1e-10 and trip2 < 1e-12
    show("fft2, seeded 64 x 64 image", f"relative error against W img W {by_matrices:.1e}")
    show("  max|ifft2(fft2(img)) - img|", f"{trip2:.1e}")

    try:
        fft(np.ones(12))
    except ValueError as error:
        show("fft of length 12, no n=", f"ValueError: {error}")
    else:
        raise AssertionError("a length that is not a power of two must raise ValueError")
    padded = fft(np.ones(12), n=16)
    assert padded.shape == (16,) and abs(padded[0] - 12.0) < 1e-12
    show("fft(ones(12), n=16)", f"zero-padded to {padded.shape[0]}, X[0] = {padded[0].real:g}")


# --------------------------------------------------- filters and spectral analysis


def section_filters() -> None:
    banner(2, "Filters and spectral analysis: mask, window and read off the bins")
    # Spec 9.4, by hand: at N = 5, n / (N - 1) = 0, 1/4, 1/2, 3/4, 1, so
    # cos(2 pi n / 4) = 1, 0, -1, 0, 1 and cos(4 pi n / 4) = 1, -1, 1, -1, 1.
    windows = {
        "hann": (hann, np.hanning, [0.0, 0.5, 1.0, 0.5, 0.0]),
        "hamming": (hamming, np.hamming, [0.08, 0.54, 1.0, 0.54, 0.08]),
        "blackman": (blackman, np.blackman, [0.0, 0.34, 1.0, 0.34, 0.0]),
    }
    for name, (ours, reference, expected) in windows.items():
        assert np.max(np.abs(ours(5) - np.array(expected))) < 1e-12
        assert np.max(np.abs(ours(64) - reference(64))) < 1e-12
        values = ", ".join(f"{v:g}" for v in np.round(ours(5), 12) + 0.0)
        show(f"{name}(5)", f"[{values}] (N = 64 equals np.{reference.__name__})")

    print()
    fs = 1000
    x = multi_tone([50, 120], [1.0, 0.5], fs, 1.0)
    peaks = peak_frequencies(x, fs, k=2)
    assert peaks == [50.0, 120.0]
    show("sin(2 pi 50 t) + 0.5 sin(2 pi 120 t)", f"fs = {fs}, N = {x.shape[0]}")
    show("  peak_frequencies(k=2)", peaks)
    # A tone of amplitude A with a whole number of cycles puts A N / 2 in its
    # bin: 500 at 50 Hz and 250 at 120 Hz. N = 1000 is not a power of two, so
    # these come from the exact-length dft.
    mags = np.abs(dft(x)[: x.shape[0] // 2 + 1])
    assert abs(mags[50] - 500.0) < 1e-9 and abs(mags[120] - 250.0) < 1e-9
    assert np.max(np.delete(mags, [50, 120])) < 1e-9
    show("  |X| at 50 Hz, 120 Hz", f"{mags[50]:.6f}, {mags[120]:.6f}")
    show("  largest |X| at any other bin", f"{np.max(np.delete(mags, [50, 120])):.1e}")

    print()
    print("  Each filter zero-pads to 1024, masks the FFT bins and inverts. Energy")
    print("  left is |Y(f)|^2 / |X(f)|^2 by the DFT definition at one bin, over the")
    print("  full record and over its interior (50 ms dropped at each end).")
    print(f"    {'filter':<14} {'kept tone, amplitude':<24} {'removed tone, energy left (full, interior)'}")
    guard = 50
    inner = x[guard:-guard]
    cases = [("low 80 Hz", "low", 80, 50, 1.0, 120),
             ("high 80 Hz", "high", 80, 120, 0.5, 50),
             ("band 100-140", "band", (100, 140), 120, 0.5, 50)]
    for label, kind, cutoff, kept, amp, removed in cases:
        y = fft_filter(x, fs, kind, cutoff)
        assert y.shape == x.shape and y.dtype == np.float64
        got = tone_amplitude(y, kept, fs)
        full = tone_power(y, removed, fs) / tone_power(x, removed, fs)
        interior = tone_power(y[guard:-guard], removed, fs) / tone_power(inner, removed, fs)
        assert abs(got - amp) < 0.01 * amp
        assert interior < 1e-6
        if kind != "low":
            assert full < 1e-6
        print(f"    {label:<14} {f'{kept} Hz: {got:.4f} (of {amp:g})':<24} "
              f"{removed} Hz: {full:.1e}, {interior:.1e}")
    print("  The 1e-6 anchor holds on the interior for all three, and on the full")
    print("  record for the high-pass and band-pass. Over the full record the")
    print("  low-pass leaves about 1e-5: the brick wall acts on a record that jumps")
    print("  to zero where the padding starts, and that ringing sits in the first")
    print("  and last few dozen samples.")

    print()
    pink = pink_noise(65536, seed=SEED)
    white = white_noise(65536, seed=SEED)
    pink_slope, white_slope = loglog_slope(pink), loglog_slope(white)
    assert abs(float(np.std(pink)) - 1.0) < 1e-12
    assert -1.3 <= pink_slope <= -0.7 and -0.2 <= white_slope <= 0.2
    show("Welch PSD log-log slope, 65536 samples", "segments of 4096, 1e-3 <= f <= 0.4")
    show("  pink_noise(seed=42)", f"{pink_slope:.3f} (required -1.3 .. -0.7, 1/f is -1)")
    show("  white_noise(seed=42)", f"{white_slope:.3f} (required -0.2 .. 0.2, flat is 0)")

    print()
    fs = 2000
    x = chirp(100, 400, fs, 2.0)
    freqs, times, mag = spectrogram(x, fs)
    assert mag.shape == (freqs.shape[0], times.shape[0])
    dominant = freqs[np.argmax(mag, axis=0)]
    smoothed = (dominant[:-2] + dominant[1:-1] + dominant[2:]) / 3.0
    assert np.all(np.diff(smoothed) >= 0)
    assert dominant[0] < 150 and dominant[-1] > 350
    show("chirp 100 -> 400 Hz, fs = 2000, 2 s", f"{mag.shape[1]} frames of 256, hop 128")
    show("  dominant frequency, first -> last", f"{dominant[0]:.1f} Hz -> {dominant[-1]:.1f} Hz")
    show("  after 3-frame smoothing", "non-decreasing in every frame")
    os.makedirs(OUT_DIR, exist_ok=True)
    spectrum_path = os.path.join(OUT_DIR, "signal_demo_spectrum.png")
    spectrogram_path = os.path.join(OUT_DIR, "signal_demo_spectrogram.png")
    padded = fft(x, n=4096)
    plot_spectrum(rfft_freqs(4096, fs), np.abs(padded[:2049]), save_path=spectrum_path,
                  title="Linear chirp 100-400 Hz: magnitude spectrum (zero-padded to 4096)")
    plot_spectrogram(freqs, times, mag, save_path=spectrogram_path)
    assert os.path.getsize(spectrum_path) > 0 and os.path.getsize(spectrogram_path) > 0
    show("  saved", relpath(spectrum_path))
    show("", relpath(spectrogram_path))

    print()
    x = rng_for("autocorrelation").standard_normal(1000)
    r = autocorrelation(x)
    worst = float(np.max(np.abs(r - np.correlate(x, x, "full")[999:])))
    assert worst < 1e-9
    show("autocorrelation, 1000 seeded samples", f"max error against np.correlate {worst:.1e}")


# ---------------------------------------------------------------- applications


def section_applications() -> None:
    banner(3, "Applications: a weekly cycle, a WAV file and an image")
    # Spec 9.5: 364 days hold 52 whole weeks, so after detrending the cycle is
    # bin 52 of the exact-length DFT and the period is 364 / 52 = 7.0 days.
    y = make_series(364, seed=SEED)
    d = detrend(y)
    period = dominant_period(d)
    k = int(np.argmax(np.abs(dft(d)[1:183]))) + 1
    lags = autocorr_peaks(d, 10)
    assert period == 7.0 and k == 52 and lags[0] == 7
    show("make_series(364, seed=42), detrended", "synthetic: 0.05 t + 10 sin(2 pi t/7) + noise")
    show("  dominant_period", f"{period} days (bin {k}: 364 / {k})")
    show("  autocorrelation peaks, lags 1-10", lags)

    print()
    with tempfile.TemporaryDirectory() as folder:
        wav = make_example_wav(os.path.join(folder, "tone_noise.wav"), seed=SEED)
        x, fs = load_wav(wav)
        assert fs == 16000 and x.shape == (32000,)
        summary = process(wav, "low", 1000)
        assert 440.0 in summary["peaks_before"] and 3000.0 in summary["peaks_before"]
        assert summary["peaks_after"] == [440.0]
        noise = rng_for("wav_round_trip").uniform(-1.0, 1.0, 5000)
        back, rate = load_wav(save_wav(os.path.join(folder, "round_trip.wav"), noise, 8000))
        trip = float(np.max(np.abs(back - noise)))
        assert rate == 8000 and trip <= 1.0 / 32768
    show("make_example_wav, then load_wav", f"fs = {fs}, {x.shape[0]} samples")
    show("  low-pass 1000 Hz: peaks before", summary["peaks_before"])
    show("  peaks after", summary["peaks_after"])
    show("  save_wav, load_wav round trip", f"max error {trip:.2e} <= 1/32768 = {1 / 32768:.2e}")

    print()
    with tempfile.TemporaryDirectory() as folder:
        rows = run_pipeline(os.path.join(EXAMPLES, "test_scene.png"), folder)["compression"]
    psnrs = [rows[q]["psnr"] for q in (10, 50, 90)]
    ratios = [rows[q]["nonzero_ratio"] for q in (10, 50, 90)]
    assert psnrs[2] > psnrs[1] > psnrs[0] and ratios[0] < ratios[1] < ratios[2]
    show("run_pipeline(test_scene.png)", "quality 10, 50, 90")
    show("  PSNR", ", ".join(f"{p:.2f}" for p in psnrs) + " dB (rises with quality)")
    show("  nonzero ratio", ", ".join(f"{r:.4f}" for r in ratios) + " (rises with quality)")

    print()
    # Every src.fft_analyzer module is loaded by now, including the
    # sub-package src.fft_analyzer.signal; the standard library module must
    # still be the one ``import signal`` finds.
    import signal

    import src.fft_analyzer.signal as package_signal

    assert signal.__name__ == "signal" and package_signal.__name__ == "src.fft_analyzer.signal"
    assert signal.SIGINT == signal.Signals.SIGINT and callable(signal.getsignal)
    show("import signal, after the package", f"signal.SIGINT = {int(signal.SIGINT)} "
         f"({signal.SIGINT.name}), the standard library module")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 12 signal demonstration")
    print("Fast transforms and signal processing - Robert Deibel")
    print("=" * WIDTH)
    print("  Move the problem where it is easy (the FFT), do the easy thing there,")
    print("  and move back. Every anchor below is asserted.")
    with Timer() as total:
        section_fft()
        section_filters()
        section_applications()
    print()
    print("=" * WIDTH)
    print(f"All demonstrations held ({fmt_seconds(total.elapsed)}).")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
