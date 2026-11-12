"""Week 12 tests: the audio, image and time-series applications (spec 9.5 anchors).

The rulers are independent of the code under test: ``np.fft``, ``np.correlate``,
``np.polyfit``, ``np.convolve`` and ``scipy.ndimage`` (allowed in tests only,
spec 1.1), the standard-library ``wave`` module, and values worked out by hand.

Time series. ``make_series(364, seed=42)`` is ``0.05 t + 10 sin(2 pi t / 7)``
plus noise of standard deviation 2. 364 days hold 52 whole weeks, so after
detrending the largest one-sided DFT bin is 52 and the period is
364 / 52 = 7.0. The cycle carries variance 10^2 / 2 = 50 against the noise's
4, so the normalised autocorrelation is about 0.93 cos(2 pi l / 7): about 0.9
at lag 7 and below 0.6 at every other lag in 1-10, and lag 7 is the only
local maximum there.

Audio. ``tone_noise.wav`` is 2 s at 16 kHz, so 32000 samples, with tones at
440 Hz (amplitude 0.5) and 3000 Hz (0.25) plus white noise of standard
deviation 0.05. Bins of the exact-length spectrum are 16000 / 32000 = 0.5 Hz
apart: the tones sit on bins 880 and 6000 with magnitude ``A N / 2`` = 8000
and 4000, while no noise bin exceeds 30, below the 1% threshold of 80. A
low-pass at 1000 Hz leaves the 440 Hz tone only. Saving rounds ``x * 32768``
to an integer, so a round trip moves a sample by at most 0.5 / 32768 (1 / 32768
for x = 1, which clips to 32767).

Image. The pipeline compresses the 256 x 256 test scene at quality 10, 50
and 90; a finer quantization table keeps more coefficients and loses less,
so the PSNR and the non-zero ratio both increase with quality.

No timings are asserted.
"""

from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage

from src.fft_analyzer.applications.audio_processor import (
    load_wav,
    make_example_wav,
    process,
    save_wav,
)
from src.fft_analyzer.applications.data_analyzer import (
    autocorr_peaks,
    detrend,
    dominant_period,
    make_series,
    smooth,
)
from src.fft_analyzer.applications.image_processor import (
    load_image,
    make_test_scene,
    run_pipeline,
    save_image,
)
from src.utils.random_utils import derive_seed

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "src" / "fft_analyzer" / "examples"

#: "Under 100 KB" for each shipped example file, in bytes.
SIZE_LIMIT = 100_000

SEED = 42


# -------------------------------------------------------------- reference helpers


def _local_maxima(values: np.ndarray, lo: int, hi: int) -> list:
    """Indices i in [lo, hi] with values[i] > values[i - 1] and values[i] >= values[i + 1]."""
    return [
        i
        for i in range(lo, hi + 1)
        if values[i] > values[i - 1] and values[i] >= values[i + 1]
    ]


# ------------------------------------------------------------------- time series


@pytest.fixture(scope="module")
def series():
    return make_series(364, seed=SEED)


def test_dominant_period_anchor(series):
    assert dominant_period(detrend(series)) == 7.0
    spectrum = np.abs(np.fft.rfft(detrend(series)))
    assert int(np.argmax(spectrum[1:])) + 1 == 52


def test_autocorrelation_peaks_at_lag_seven(series):
    d = detrend(series)
    peaks = autocorr_peaks(d, 10)
    assert peaks[0] == 7
    centred = d - d.mean()
    r = np.correlate(centred, centred, "full")[len(d) - 1 :]
    rho = r / r[0]
    reference = sorted(_local_maxima(rho, 1, 10), key=lambda lag: -rho[lag])
    assert peaks == reference == [7]


def test_detrend_matches_polyfit(series):
    t = np.arange(series.shape[0])
    line = np.polyval(np.polyfit(t, series, 1), t)
    assert np.max(np.abs(detrend(series) - (series - line))) < 1e-9


@pytest.mark.parametrize("window", [1, 4, 7, 30])
def test_smooth_matches_np_convolve(series, window):
    expected = np.convolve(series, np.ones(window) / window, "same")
    assert np.max(np.abs(smooth(series, window) - expected)) < 1e-9


def test_dominant_period_power_of_two_length():
    y = np.sin(2 * np.pi * np.arange(256) / 16) + 0.1 * np.cos(2 * np.pi * np.arange(256) / 64)
    assert dominant_period(y) == 16.0


def test_series_is_seeded():
    assert np.array_equal(make_series(50, seed=SEED), make_series(50, seed=SEED))
    assert not np.array_equal(make_series(50, seed=SEED), make_series(50, seed=SEED + 1))


def test_series_argument_errors(series):
    with pytest.raises(ValueError):
        smooth(series, series.shape[0] + 1)
    with pytest.raises(ValueError):
        autocorr_peaks(np.full(20, 3.0), 5)
    with pytest.raises(ValueError):
        autocorr_peaks(series, 0)


# ------------------------------------------------------------------------- audio


@pytest.fixture
def example_wav(tmp_path):
    return make_example_wav(tmp_path / "tone_noise.wav", seed=SEED)


def test_example_wav_length_and_rate(example_wav):
    x, fs = load_wav(example_wav)
    assert fs == 16000
    assert x.shape == (32000,)
    assert x.dtype == np.float64
    assert np.max(np.abs(x)) < 1.0


def test_lowpass_removes_3000_hz_peak(example_wav, tmp_path):
    out = tmp_path / "filtered.wav"
    summary = process(example_wav, "low", 1000, out_path=out)
    assert summary["fs"] == 16000
    assert summary["n"] == 32000
    assert 440.0 in summary["peaks_before"]
    assert 3000.0 in summary["peaks_before"]
    assert summary["peaks_after"] == [440.0]
    assert summary["out_path"] == os.fspath(out)
    y, fs = load_wav(out)
    assert fs == 16000 and y.shape == (32000,)
    x, _ = load_wav(example_wav)
    before = np.abs(np.fft.rfft(x))
    after = np.abs(np.fft.rfft(y))
    assert after[6000] < 1e-3 * before[6000]
    assert abs(after[880] - before[880]) < 0.01 * before[880]


def test_peaks_before_match_numpy_spectrum(example_wav):
    x, fs = load_wav(example_wav)
    mag = np.abs(np.fft.rfft(x))
    maxima = _local_maxima(mag, 1, mag.shape[0] - 2)
    top_two = sorted(sorted(maxima, key=lambda i: -mag[i])[:2])
    freqs = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
    assert [float(freqs[i]) for i in top_two] == [440.0, 3000.0]
    assert process(example_wav, "low", 1000)["peaks_before"] == [440.0, 3000.0]


def test_energy_removed_matches_numpy_filter(example_wav):
    x, fs = load_wav(example_wav)
    m = 32768
    spectrum = np.fft.fft(x, m)
    f_abs = np.abs(np.fft.fftfreq(m, 1.0 / fs))
    y = np.fft.ifft(spectrum * (f_abs <= 1000)).real[: x.shape[0]]
    expected = 1.0 - np.sum(y * y) / np.sum(x * x)
    summary = process(example_wav, "low", 1000)
    assert summary["out_path"] is None
    assert abs(summary["energy_removed"] - expected) < 1e-9
    assert 0.0 < summary["energy_removed"] < 1.0


def test_highpass_keeps_only_3000_hz(example_wav):
    assert process(example_wav, "high", 1000)["peaks_after"] == [3000.0]


def test_wav_round_trip_within_one_lsb(tmp_path):
    rng = np.random.default_rng(derive_seed(SEED, "week12", "wav_round_trip"))
    x = rng.uniform(-1.0, 1.0, 5000)
    x[:3] = [-1.0, 1.0, 0.0]
    y, fs = load_wav(save_wav(tmp_path / "round_trip.wav", x, 8000))
    assert fs == 8000
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) <= 1.0 / 32768


def test_load_wav_averages_stereo(tmp_path):
    left = np.array([1000, -2000, 32767, -32768], dtype="<i2")
    right = np.array([3000, 0, -32768, -32768], dtype="<i2")
    path = tmp_path / "stereo.wav"
    with wave.open(os.fspath(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(np.column_stack([left, right]).astype("<i2").tobytes())
    x, fs = load_wav(path)
    assert fs == 22050
    expected = (left.astype(np.float64) + right.astype(np.float64)) / 2.0 / 32768.0
    assert np.array_equal(x, expected)


# ------------------------------------------------------------------------- image


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    folder = tmp_path_factory.mktemp("week12_image")
    scene = make_test_scene(folder / "test_scene.png", seed=SEED)
    return scene, run_pipeline(scene, folder / "out")


def test_pipeline_quality_ordering(pipeline):
    _, result = pipeline
    rows = result["compression"]
    assert set(rows) == {10, 50, 90}
    assert rows[90]["psnr"] > rows[50]["psnr"] > rows[10]["psnr"]
    assert rows[10]["nonzero_ratio"] < rows[50]["nonzero_ratio"] < rows[90]["nonzero_ratio"]


def test_pipeline_writes_every_output(pipeline):
    _, result = pipeline
    assert set(result) == {"blur", "sobel", "denoise", "compression"}
    paths = [result[k] for k in ("blur", "sobel", "denoise")]
    paths += [result["compression"][q]["path"] for q in (10, 50, 90)]
    for p in paths:
        assert Path(p).is_file()
        assert load_image(p).shape == (256, 256)


def test_pipeline_psnr_matches_saved_reconstruction(pipeline):
    scene, result = pipeline
    original = load_image(scene)
    for q in (10, 50, 90):
        saved = load_image(result["compression"][q]["path"])
        mse = np.mean((original - saved) ** 2)
        assert abs(10 * np.log10(255.0 ** 2 / mse) - result["compression"][q]["psnr"]) < 0.1


def test_pipeline_blur_matches_scipy(pipeline):
    scene, result = pipeline
    reference = scipy.ndimage.gaussian_filter(load_image(scene), 2.0, mode="wrap", truncate=4.0)
    saved = load_image(result["blur"])
    assert np.max(np.abs(saved - np.clip(np.round(reference), 0, 255))) <= 1.0


def test_pipeline_denoise_matches_numpy_mask(pipeline):
    scene, result = pipeline
    img = load_image(scene)
    centred = np.fft.fftshift(np.fft.fft2(img))
    rows, cols = np.ogrid[-128:128, -128:128]
    kept = np.fft.ifft2(np.fft.ifftshift(centred * (rows**2 + cols**2 <= 32**2))).real
    saved = load_image(result["denoise"])
    assert np.max(np.abs(saved - np.clip(np.round(kept), 0, 255))) <= 1.0


def test_test_scene_is_grayscale_in_range(pipeline):
    scene, _ = pipeline
    img = load_image(scene)
    assert img.shape == (256, 256)
    assert img.dtype == np.float64
    assert img.min() >= 0.0 and img.max() <= 255.0


def test_load_image_uses_luma_weights(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(derive_seed(SEED, "week12", "rgb_image"))
    rgb = rng.integers(0, 256, size=(6, 5, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(path)
    expected = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    assert np.max(np.abs(load_image(path) - expected)) < 1e-9


def test_save_image_rounds_and_clips(tmp_path):
    path = save_image(tmp_path / "clip.png", [[-20.0, 12.4], [12.6, 999.0]])
    assert np.array_equal(load_image(path), [[0.0, 12.0], [13.0, 255.0]])


# --------------------------------------------------- the stdlib ``signal`` module


def test_stdlib_signal_survives_package_import():
    import src.fft_analyzer  # noqa: F401
    import src.fft_analyzer.signal.analysis  # noqa: F401
    import src.fft_analyzer.signal.filters  # noqa: F401
    import src.fft_analyzer.signal.generators  # noqa: F401
    import signal

    assert signal.__name__ == "signal"
    assert isinstance(signal.SIGINT, int)
    assert signal.SIGINT == signal.Signals.SIGINT
    signal.getsignal(signal.SIGINT)


def test_stdlib_signal_in_fresh_interpreter():
    code = (
        "import src.fft_analyzer, src.fft_analyzer.signal.filters\n"
        "import signal\n"
        "print(int(signal.SIGINT), signal.__name__, hasattr(signal, 'getsignal'))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    import signal

    assert done.stdout.split() == [str(int(signal.SIGINT)), "signal", "True"]


# ------------------------------------------------------------ shipped examples


@pytest.mark.parametrize("name", ["tone_noise.wav", "test_scene.png"])
def test_shipped_examples_exist_and_are_small(name):
    path = EXAMPLES / name
    assert path.is_file()
    assert path.stat().st_size < SIZE_LIMIT


def test_shipped_wav_is_the_seed_42_example(tmp_path):
    shipped, fs = load_wav(EXAMPLES / "tone_noise.wav")
    fresh, fresh_fs = load_wav(make_example_wav(tmp_path / "fresh.wav", seed=SEED))
    assert fs == fresh_fs == 16000
    assert shipped.shape == (32000,)
    assert np.array_equal(shipped, fresh)


def test_shipped_scene_is_the_seed_42_example(tmp_path):
    shipped = load_image(EXAMPLES / "test_scene.png")
    fresh = load_image(make_test_scene(tmp_path / "fresh.png", seed=SEED))
    assert shipped.shape == (256, 256)
    assert np.array_equal(shipped, fresh)
