"""Week 12 tests: windows, FFT filters and spectral analysis on the spec 9.4 anchors.

Hand values:

- Windows at N = 5: ``n / (N - 1)`` is 0, 1/4, 1/2, 3/4, 1, so
  ``cos(2 pi n / 4)`` is 1, 0, -1, 0, 1 and ``cos(4 pi n / 4)`` is 1, -1, 1,
  -1, 1. Hann ``0.5 - 0.5 c`` = [0, 0.5, 1, 0.5, 0]; Hamming ``0.54 - 0.46 c``
  = [0.08, 0.54, 1, 0.54, 0.08]; Blackman ``0.42 - 0.5 c + 0.08 c2`` gives
  0.42 - 0.5 + 0.08 = 0, 0.42 - 0.08 = 0.34 and 0.42 + 0.5 + 0.08 = 1.
- The two-tone signal ``sin(2 pi 50 t) + 0.5 sin(2 pi 120 t)`` at fs = 1000,
  N = 1000: each tone completes a whole number of cycles, so a tone of
  amplitude A puts ``A N / 2`` into its bin: 500 at bin 50 and 250 at bin 120
  (bins are fs / N = 1 Hz apart). 1000 is not a power of two, so these
  magnitudes come from the package's exact-length ``dft``.

How the filter anchors are measured. The ruler is ``np.fft.rfft`` (allowed in
tests only, spec 1.1), never the code under test:

- *Amplitude* of a tone: ``2 |Y[f]| / N`` on the full 1000-sample output,
  compared with the input tone's amplitude (within 1%).
- *Energy* of a removed tone: ``|Y[f]|^2 / |X[f]|^2`` on the interior
  ``[GUARD, N - GUARD)`` of the record (GUARD = 50 samples = 50 ms at each
  end; 900 samples keep both tones on exact bins, 45 and 108). The spec
  requires ``fft_filter`` to zero-pad x to a power of two (1024) and trim back,
  so the brick wall acts on a record that jumps to zero at its end; the
  resulting ringing sits in the first and last few dozen samples. Over the
  full record the low-pass leaves 9.8e-6 of the 120 Hz energy (the high-pass
  5.0e-7 of the 50 Hz energy, the band-pass 1.0e-8); on the interior all
  three are below 1e-9. The 1e-6 anchor is asserted on the interior for all
  three, and also on the full record for the high-pass and the band-pass,
  where the zero-padded brick wall meets it as written.

Noise slopes: ``welch_psd`` with fs = 1 (frequencies in cycles per sample),
segments of 4096, and a least-squares line through ``log10 psd`` against
``log10 f`` for 1e-3 <= f <= 0.4 (the reference gave -0.999 for pink noise).

No timings are asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from src.fft_analyzer.core.fft import dft
from src.fft_analyzer.signal.analysis import (
    autocorrelation,
    peak_frequencies,
    psd,
    spectrogram,
    welch_psd,
)
from src.fft_analyzer.signal.filters import blackman, fft_filter, hamming, hann
from src.fft_analyzer.signal.generators import (
    chirp,
    multi_tone,
    pink_noise,
    time_axis,
    white_noise,
)
from src.utils.random_utils import derive_seed

MASTER_SEED = 42

FS = 1000
N = 1000
GUARD = 50

WINDOWS = {
    "hann": (hann, np.hanning, [0.0, 0.5, 1.0, 0.5, 0.0]),
    "hamming": (hamming, np.hamming, [0.08, 0.54, 1.0, 0.54, 0.08]),
    "blackman": (blackman, np.blackman, [0.0, 0.34, 1.0, 0.34, 0.0]),
}


def _rng(*labels) -> np.random.Generator:
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(MASTER_SEED, "test_filters", *labels))


@pytest.fixture(scope="module")
def two_tone() -> np.ndarray:
    """``sin(2 pi 50 t) + 0.5 sin(2 pi 120 t)``, fs = 1000, N = 1000."""
    t = np.arange(N) / FS
    return np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)


def _amplitude(y: np.ndarray, freq: int) -> float:
    """Amplitude of the tone at ``freq`` Hz over the full record, by np.fft."""
    return 2.0 * abs(np.fft.rfft(y)[freq * len(y) // FS]) / len(y)


def _full_energy(y: np.ndarray, freq: int) -> float:
    """``|Y[f]|^2`` on the full record, by np.fft."""
    return abs(np.fft.rfft(y)[freq * len(y) // FS]) ** 2


def _interior_energy(y: np.ndarray, freq: int) -> float:
    """``|Y[f]|^2`` on samples ``[GUARD, len(y) - GUARD)``, by np.fft."""
    inner = y[GUARD : len(y) - GUARD]
    return abs(np.fft.rfft(inner)[freq * len(inner) // FS]) ** 2


def _loglog_slope(x: np.ndarray) -> float:
    """Slope of log10 Welch PSD against log10 f for 1e-3 <= f <= 0.4 (fs = 1)."""
    freqs, density = welch_psd(x, 1.0, seg=4096)
    keep = (freqs >= 1e-3) & (freqs <= 0.4)
    slope, _ = np.polyfit(np.log10(freqs[keep]), np.log10(density[keep]), 1)
    return float(slope)


# ------------------------------------------------------------------ windows


@pytest.mark.parametrize("name", sorted(WINDOWS))
@pytest.mark.parametrize("length", [5, 64])
def test_window_matches_numpy(name, length):
    ours, reference, _ = WINDOWS[name]
    w = ours(length)
    assert w.shape == (length,)
    assert np.max(np.abs(w - reference(length))) < 1e-12


@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_window_literal_values_n5(name):
    ours, _, expected = WINDOWS[name]
    assert np.max(np.abs(ours(5) - np.array(expected))) < 1e-12


# ------------------------------------------------------- the two-tone signal


def test_multi_tone_builds_the_two_tone_signal(two_tone):
    x = multi_tone([50, 120], [1.0, 0.5], FS, 1.0)
    assert x.dtype == np.float64
    assert len(time_axis(FS, 1.0)) == N
    assert np.max(np.abs(x - two_tone)) < 1e-12


def test_peak_frequencies_two_tone(two_tone):
    assert peak_frequencies(two_tone, FS, k=2) == [50, 120]


def test_rfft_magnitudes_two_tone(two_tone):
    mags = np.abs(dft(two_tone)[: N // 2 + 1])
    assert abs(mags[50] - 500.0) < 1e-9
    assert abs(mags[120] - 250.0) < 1e-9
    others = np.delete(mags, [50, 120])
    assert np.max(others) < 1e-9
    assert np.max(np.abs(mags - np.abs(np.fft.rfft(two_tone)))) < 1e-9


def test_low_pass_removes_120_keeps_50(two_tone):
    y = fft_filter(two_tone, FS, "low", 80)
    assert y.shape == (N,) and y.dtype == np.float64
    assert _interior_energy(y, 120) < 1e-6 * _interior_energy(two_tone, 120)
    assert abs(_amplitude(y, 50) - 1.0) < 0.01


def test_high_pass_removes_50_keeps_120(two_tone):
    y = fft_filter(two_tone, FS, "high", 80)
    assert y.shape == (N,) and y.dtype == np.float64
    assert _interior_energy(y, 50) < 1e-6 * _interior_energy(two_tone, 50)
    assert _full_energy(y, 50) < 1e-6 * _full_energy(two_tone, 50)
    assert abs(_amplitude(y, 120) - 0.5) < 0.01 * 0.5


def test_band_pass_keeps_only_120(two_tone):
    y = fft_filter(two_tone, FS, "band", (100, 140))
    assert y.shape == (N,) and y.dtype == np.float64
    assert _interior_energy(y, 50) < 1e-6 * _interior_energy(two_tone, 50)
    assert _full_energy(y, 50) < 1e-6 * _full_energy(two_tone, 50)
    assert abs(_amplitude(y, 120) - 0.5) < 0.01 * 0.5


# --------------------------------------------------------------------- noise


def test_pink_noise_welch_slope():
    x = pink_noise(65536, seed=42)
    assert x.shape == (65536,) and x.dtype == np.float64
    assert abs(np.std(x) - 1.0) < 1e-12
    assert -1.3 <= _loglog_slope(x) <= -0.7


def test_white_noise_welch_slope():
    x = white_noise(65536, seed=42)
    assert -0.2 <= _loglog_slope(x) <= 0.2


def test_noise_is_seeded():
    assert np.array_equal(white_noise(256, seed=7), white_noise(256, seed=7))
    assert not np.array_equal(white_noise(256, seed=7), white_noise(256, seed=8))
    assert np.array_equal(pink_noise(1000, seed=7), pink_noise(1000, seed=7))


# ---------------------------------------------------------------- the chirp


def test_chirp_dominant_frequency_rises():
    fs = 2000
    x = chirp(100, 400, fs, 2.0)
    freqs, times, mag = spectrogram(x, fs)
    assert mag.shape == (len(freqs), len(times))
    dominant = freqs[np.argmax(mag, axis=0)]
    smoothed = np.convolve(dominant, np.ones(3) / 3, mode="valid")
    assert np.all(np.diff(smoothed) >= 0)
    assert dominant[0] < 150 and dominant[-1] > 350


# ------------------------------------------------------ autocorrelation, psd


@pytest.mark.parametrize("length", [1, 2, 7, 64, 500, 1000])
def test_autocorrelation_matches_np_correlate(length):
    x = _rng("autocorrelation", length).standard_normal(length)
    expected = np.correlate(x, x, "full")[length - 1 :]
    r = autocorrelation(x)
    assert r.shape == (length,)
    assert np.max(np.abs(r - expected)) < 1e-9


@pytest.mark.parametrize("length", [999, 1000, 1024])
@pytest.mark.parametrize(
    "window, reference",
    [("hann", np.hanning), ("hamming", np.hamming), ("blackman", np.blackman),
     ("boxcar", np.ones)],
)
def test_psd_matches_direct_periodogram(length, window, reference):
    fs = 250.0
    x = _rng("psd", length, window).standard_normal(length)
    w = reference(length)
    spectrum = np.fft.rfft(x * w)
    expected = np.abs(spectrum) ** 2 / (fs * np.sum(w ** 2))
    expected[1 : (length + 1) // 2] *= 2.0
    freqs, power = psd(x, fs, window=window)
    assert np.max(np.abs(freqs - np.fft.rfftfreq(length, 1.0 / fs))) < 1e-9
    assert np.max(np.abs(power - expected)) < 1e-10 * np.max(expected)
    _, scipy_power = scipy.signal.periodogram(x, fs, window=w, detrend=False)
    assert np.max(np.abs(power - scipy_power)) < 1e-10 * np.max(expected)
