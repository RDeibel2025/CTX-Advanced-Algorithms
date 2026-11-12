#!/usr/bin/env python3
"""Runnable demonstration of the Week 12 audio processor: a low-pass on a WAV file.

    python examples/week12_audio_demo.py

Three short sections. Apart from the time, the output is the same on every
run, because the example file is made from a fixed seed:

1. **The example file** - ``src/fft_analyzer/examples/tone_noise.wav`` (created
   with ``make_example_wav(seed=42)`` if it is missing): 2 s at 16 kHz, tones
   at 440 Hz and 3,000 Hz and white noise, synthetic rather than recorded.
2. **The low-pass** - ``process`` filters it at 1,000 Hz, saves
   ``benchmarks/results/week12/audio_filtered.wav`` and reports the peak
   frequencies before and after and the fraction of the energy removed.
3. **The saved file** - read back, the 3,000 Hz tone is gone and the 440 Hz
   tone is unchanged, and saving the filtered samples as 16 bits moved none
   of them by more than 1/32768.

The filter is the round trip this week is built on: move the problem where
it is easy (the FFT of the zero-padded signal), do the easy thing (zero every
bin above 1,000 Hz), and move back (the inverse FFT, trimmed to the original
length). The anchor values come from section 9.5 of the Week 12 build spec,
and every anchor printed is also asserted, so the script exits non-zero
rather than printing something untrue. Tone magnitudes are measured with the
DFT definition at one bin, never ``np.fft`` (spec 1.1). It finishes in about
a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.fft_analyzer.applications.audio_processor import (  # noqa: E402
    EXAMPLE_AMPS,
    EXAMPLE_NOISE,
    EXAMPLE_TONES,
    load_wav,
    make_example_wav,
    process,
)
from src.fft_analyzer.signal.filters import fft_filter  # noqa: E402
from src.utils.timer import Timer  # noqa: E402

WIDTH = 74
SEED = 42
CUTOFF = 1000
WAV_IN = os.path.join(REPO_ROOT, "src", "fft_analyzer", "examples", "tone_noise.wav")
WAV_OUT = os.path.join(REPO_ROOT, "benchmarks", "results", "week12", "audio_filtered.wav")


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<36} {value}")


def relpath(path: str) -> str:
    """``path`` relative to the repository root, for printing."""
    return os.path.relpath(path, REPO_ROOT)


def tone_magnitude(x: np.ndarray, freq: float, fs: int) -> float:
    """``|X(freq)| = |sum_n x[n] exp(-2 pi i freq n / fs)|``, the DFT definition at one bin.

    The bin is ``k = freq N / fs``; ``k n`` is reduced modulo N in integer
    arithmetic first, so the angle stays exact.
    """
    n = x.shape[0]
    k = int(round(freq * n / fs))
    t = np.arange(n, dtype=np.int64)
    return abs(complex(np.sum(x * np.exp(-2j * np.pi * ((k * t) % n) / n))))


def section_example() -> None:
    banner(1, "The example file: two tones and white noise")
    if os.path.exists(WAV_IN):
        show("found", relpath(WAV_IN))
    else:
        make_example_wav(WAV_IN, seed=SEED)
        show("created with make_example_wav(seed=42)", relpath(WAV_IN))
    x, fs = load_wav(WAV_IN)
    # Spec 9.5: 2 s at 16 kHz is 32000 samples.
    assert fs == 16000 and x.shape == (32000,) and np.max(np.abs(x)) < 1.0
    show("sampling rate, samples", f"{fs} Hz, {x.shape[0]} ({x.shape[0] / fs:g} s)")
    show("tones (synthetic)", ", ".join(f"{f:g} Hz at amplitude {a:g}"
                                         for f, a in zip(EXAMPLE_TONES, EXAMPLE_AMPS)))
    show("white noise", f"standard deviation {EXAMPLE_NOISE:g}, seeded")
    show("largest |sample|", f"{np.max(np.abs(x)):.4f} (below 1, nothing clips)")


def section_lowpass() -> None:
    banner(2, f"The low-pass at {CUTOFF:,} Hz: FFT, zero the high bins, inverse FFT")
    summary = process(WAV_IN, "low", CUTOFF, out_path=WAV_OUT)
    # Spec 9.5: before, the two strongest peaks are the two tones; after the
    # low-pass only 440 Hz is left (any peak under 1% of the largest is the
    # noise floor, not a tone).
    assert summary["fs"] == 16000 and summary["n"] == 32000
    assert 440.0 in summary["peaks_before"] and 3000.0 in summary["peaks_before"]
    assert summary["peaks_after"] == [440.0]
    assert summary["out_path"] == WAV_OUT and os.path.getsize(WAV_OUT) > 0
    # By hand: the tones carry 0.5^2 / 2 = 0.125 and 0.25^2 / 2 = 0.03125 of
    # power and the noise 0.05^2 = 0.0025, spread evenly over 0-8000 Hz. The
    # low-pass removes the 3000 Hz tone and the 7/8 of the noise above 1000 Hz:
    # (0.03125 + 0.0021875) / 0.15875 = 0.2106.
    removed = summary["energy_removed"]
    assert abs(removed - 0.2106) < 0.01
    show("peaks before", summary["peaks_before"])
    show("peaks after", summary["peaks_after"])
    show("energy removed", f"{removed:.4f} (by hand: 0.2106, the 3000 Hz tone plus")
    show("", "the 7/8 of the noise above 1000 Hz)")
    show("saved", relpath(WAV_OUT))


def section_saved() -> None:
    banner(3, "The saved file: what the filter kept and what it removed")
    x, fs = load_wav(WAV_IN)
    y, fs_out = load_wav(WAV_OUT)
    assert fs_out == fs == 16000 and y.shape == x.shape
    print(f"    {'tone':>8}  {'|X| before':>11}  {'|X| after':>11}  after / before")
    ratios = {}
    for freq in EXAMPLE_TONES:
        before, after = tone_magnitude(x, freq, fs), tone_magnitude(y, freq, fs)
        ratios[freq] = after / before
        print(f"    {freq:>6g} Hz  {before:>11.1f}  {after:>11.1f}  {ratios[freq]:.6f}")
    # The 3000 Hz tone sits on bin 6000 and drops by more than 1000 times; the
    # 440 Hz tone (bin 880) keeps its magnitude within 1%.
    assert ratios[3000.0] < 1e-3 and abs(ratios[440.0] - 1.0) < 0.01

    print()
    # Spec 9.5: save_wav rounds y * 32768 to an integer, so the file read back
    # differs from the filter's float output by at most 1/32768 per sample.
    exact = fft_filter(x, fs, "low", CUTOFF)
    trip = float(np.max(np.abs(y - exact)))
    assert trip <= 1.0 / 32768
    show("file against the float filter output", f"max change {trip:.2e} <= 1/32768 = "
         f"{1 / 32768:.2e}")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 12 audio demonstration")
    print("FFT filtering of a WAV file - Robert Deibel")
    print("=" * WIDTH)
    print("  Move the problem where it is easy (the FFT), do the easy thing there")
    print("  (zero the unwanted bins), and move back. Every anchor below is asserted.")
    with Timer() as total:
        section_example()
        section_lowpass()
        section_saved()
    print()
    print("=" * WIDTH)
    print(f"All demonstrations held ({total.elapsed:.2f} s).")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
