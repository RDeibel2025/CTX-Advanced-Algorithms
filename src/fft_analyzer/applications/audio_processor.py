"""Audio filtering: 16-bit WAV input and output, FFT filtering and a peak summary.

The method
----------
A WAV file stores a sampled sound as integers. :func:`load_wav` reads 16-bit
PCM with the standard-library :mod:`wave` module, averages the channels of a
stereo file into one, and divides by 32768 so the samples lie in [-1, 1).
:func:`save_wav` reverses that: it clips to [-1, 1], multiplies by 32768,
rounds and stores 16-bit mono, so a save and load round trip changes no
sample by more than 1/32768.

:func:`process` filters a WAV file with
:func:`~src.fft_analyzer.signal.filters.fft_filter`, which makes the round
trip every tool in this package makes: move the problem where it is easy
(the FFT of the zero-padded signal), do the easy thing (a brick-wall mask
that zeroes the unwanted frequency bins), and move back (the inverse FFT,
trimmed to the original length). It then reports how much of the signal's
energy the filter removed and which tones were present before and after.

The peak summary
----------------
:func:`~src.fft_analyzer.signal.analysis.peak_frequencies` returns the k = 2
largest local maxima of the one-sided magnitude spectrum, but a local maximum
always exists: once a tone is filtered out, the second "peak" is a ripple of
the noise floor. The summary therefore measures each returned peak's
magnitude ``|X[i]| = |sum_n x[n] exp(-2 pi i i n / N)|`` (one DFT bin, O(N))
and drops any peak below 1% of the largest. On the example file the tones
reach |X| of about 8000 and 4000 while no noise bin exceeds 30, so the 1%
threshold (about 80) sits between the noise floor and the weaker tone with
room on both sides.

The example file
----------------
:func:`make_example_wav` writes ``tone_noise.wav``: 2 s at 16 kHz (32000
samples), a 440 Hz tone of amplitude 0.5, a 3000 Hz tone of amplitude 0.25
and Gaussian white noise of standard deviation 0.05, seeded through
:func:`src.utils.random_utils.derive_seed`. The peak magnitude stays below
0.93, so nothing clips. It is synthetic, not a recording. A low-pass at
1000 Hz removes the 3000 Hz tone and keeps the 440 Hz one.

Examples:
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = make_example_wav(os.path.join(d, "tone_noise.wav"), seed=42)
    ...     summary = process(path, "low", 1000)
    >>> summary["fs"], summary["n"]
    (16000, 32000)
    >>> summary["peaks_before"], summary["peaks_after"]
    ([440.0, 3000.0], [440.0])

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
import os
import wave
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.fft_analyzer.signal.analysis import peak_frequencies
from src.fft_analyzer.signal.filters import fft_filter
from src.fft_analyzer.signal.generators import multi_tone, white_noise
from src.utils.random_utils import derive_seed

__all__ = [
    "EXAMPLE_AMPS",
    "EXAMPLE_DURATION",
    "EXAMPLE_FS",
    "EXAMPLE_NOISE",
    "EXAMPLE_TONES",
    "load_wav",
    "make_example_wav",
    "process",
    "save_wav",
]

#: Path-like values accepted by the file functions.
PathLike = Union[str, "os.PathLike[str]"]

#: Full scale of 16-bit PCM: sample value s is stored as round(s * 32768).
_FULL_SCALE = 32768.0

#: Number of peaks reported before and after filtering.
_PEAK_COUNT = 2

#: A peak below this fraction of the largest peak's magnitude is noise.
_PEAK_FLOOR = 0.01

#: Sampling rate of the example file in Hz.
EXAMPLE_FS: int = 16000

#: Length of the example file in seconds.
EXAMPLE_DURATION: float = 2.0

#: Tone frequencies of the example file in Hz.
EXAMPLE_TONES: Tuple[float, float] = (440.0, 3000.0)

#: Tone amplitudes of the example file (full scale is 1).
EXAMPLE_AMPS: Tuple[float, float] = (0.5, 0.25)

#: Standard deviation of the example file's white noise.
EXAMPLE_NOISE: float = 0.05


# ------------------------------------------------------------------ helpers


def _real_samples(x: Any) -> np.ndarray:
    """Return ``x`` as a finite, real, 1-D ``float64`` array.

    Args:
        x: A 1-D sequence of real numbers (it may be empty).

    Returns:
        A ``float64`` array of the same length.

    Raises:
        ValueError: If ``x`` is not 1-D, is complex, or holds NaN or
            infinity.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N) for the converted copy.

    Examples:
        >>> _real_samples([0, 0.5, -1])
        array([ 0. ,  0.5, -1. ])
        >>> _real_samples([[0.0]])
        Traceback (most recent call last):
            ...
        ValueError: samples must be a 1-D sequence, got shape (1, 1)
    """
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise ValueError("samples must be real")
    a = a.astype(np.float64)
    if a.ndim != 1:
        raise ValueError(f"samples must be a 1-D sequence, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("samples must be finite (no NaN or infinity)")
    return a


def _sample_rate(fs: Any) -> int:
    """Return ``fs`` as a positive ``int`` sampling rate, as a WAV header needs.

    Args:
        fs: The sampling rate in Hz; an integer, or a float with an integer
            value.

    Returns:
        ``int(fs)``.

    Raises:
        ValueError: If ``fs`` is not a positive whole number.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _sample_rate(16000), _sample_rate(8000.0)
        (16000, 8000)
        >>> _sample_rate(44100.5)
        Traceback (most recent call last):
            ...
        ValueError: fs must be a positive whole number of Hz, got 44100.5
    """
    if isinstance(fs, bool) or not isinstance(fs, numbers.Real):
        raise ValueError(f"fs must be a positive whole number of Hz, got {fs!r}")
    value = float(fs)
    if not (math.isfinite(value) and value > 0 and value.is_integer()):
        raise ValueError(f"fs must be a positive whole number of Hz, got {fs!r}")
    return int(value)


def _ensure_parent(path: PathLike) -> str:
    """Create the parent directory of ``path`` if needed and return the path as text.

    Args:
        path: A file path.

    Returns:
        ``os.fspath(path)``.

    Raises:
        OSError: If the directory cannot be created.

    Time Complexity:
        O(depth) directory checks.

    Space Complexity:
        O(1).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     target = _ensure_parent(os.path.join(d, "a", "b.wav"))
        ...     os.path.isdir(os.path.join(d, "a"))
        True
    """
    text = os.fspath(path)
    parent = os.path.dirname(text)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return text


def _bin_magnitude(x: np.ndarray, index: int) -> float:
    """Return ``|X[index]|``, one bin of the N-point DFT of ``x``, in O(N).

    The phase ``index * n`` is reduced modulo N in integer arithmetic before
    it is turned into an angle, so the exponentials stay accurate for long
    signals.

    Args:
        x: A 1-D real signal of length N.
        index: The bin, ``0 <= index < N``.

    Returns:
        ``|sum_n x[n] exp(-2 pi i index n / N)|``.

    Raises:
        ValueError: If ``x`` is empty.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N) for the vector of phases.

    Examples:
        >>> x = np.cos(2 * np.pi * 3 * np.arange(16) / 16)
        >>> round(_bin_magnitude(x, 3), 9), round(_bin_magnitude(x, 5), 9) + 0
        (8.0, 0.0)
    """
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot take a DFT bin of an empty signal")
    phase = (np.arange(n, dtype=np.int64) * int(index)) % n
    angle = -2.0 * np.pi * phase / n
    re = float(np.sum(x * np.cos(angle)))
    im = float(np.sum(x * np.sin(angle)))
    return math.hypot(re, im)


def _strong_peaks(x: np.ndarray, fs: int) -> List[float]:
    """Return the k = 2 strongest spectral peaks of ``x``, minus any below 1% of the largest.

    :func:`~src.fft_analyzer.signal.analysis.peak_frequencies` gives the k
    largest local maxima of the exact-length one-sided magnitude spectrum;
    each one's magnitude is then measured with :func:`_bin_magnitude` at bin
    ``round(f N / fs)``, and peaks below ``0.01`` times the largest are
    dropped, because a local maximum of the noise floor is not a tone.

    Args:
        x: A 1-D real signal.
        fs: Its sampling rate in Hz.

    Returns:
        The surviving peak frequencies in Hz, ascending (``[]`` for a silent
        or too-short signal).

    Raises:
        ValueError: If ``x`` is empty.

    Time Complexity:
        That of :func:`~src.fft_analyzer.signal.analysis.peak_frequencies`
        (O(N log N) for a power-of-two N), plus O(N) per peak.

    Space Complexity:
        O(N).

    Examples:
        >>> t = np.arange(1000) / 1000
        >>> x = np.sin(2 * np.pi * 50 * t) + 0.001 * np.sin(2 * np.pi * 120 * t)
        >>> _strong_peaks(x, 1000)
        [50.0]
        >>> _strong_peaks(np.zeros(64), 1000)
        []
    """
    n = x.shape[0]
    freqs = peak_frequencies(x, fs, k=_PEAK_COUNT)
    if not freqs:
        return []
    mags = [_bin_magnitude(x, int(round(f * n / fs))) for f in freqs]
    largest = max(mags)
    if largest == 0.0:
        return []
    return [f for f, m in zip(freqs, mags) if m >= _PEAK_FLOOR * largest]


# ---------------------------------------------------------------- WAV files


def load_wav(path: PathLike) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file as float samples in [-1, 1) and its sampling rate.

    The file is read with the standard-library :mod:`wave` module. The
    little-endian 16-bit integers are divided by 32768, so -32768 maps to -1
    and 32767 to 32767/32768. A file with several channels (stereo) is
    reduced to one by averaging the channels of every frame.

    Args:
        path: Path to the WAV file.

    Returns:
        ``(samples, fs)``: a 1-D ``float64`` array with one value per frame,
        and the sampling rate in Hz as an ``int``.

    Raises:
        FileNotFoundError: If the file does not exist.
        wave.Error: If the file is not an uncompressed PCM WAV file.
        ValueError: If the samples are not 16-bit.

    Time Complexity:
        O(F C) for F frames of C channels.

    Space Complexity:
        O(F C) for the raw frames.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = save_wav(os.path.join(d, "a.wav"), [0.0, 0.5, -1.0], 8000)
        ...     load_wav(p)
        (array([ 0. ,  0.5, -1. ]), 8000)
    """
    with wave.open(os.fspath(path), "rb") as wav:
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        fs = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got {8 * width}-bit samples")
    ints = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if channels > 1:
        ints = ints.reshape(-1, channels).mean(axis=1)
    return ints / _FULL_SCALE, int(fs)


def save_wav(path: PathLike, x: Any, fs: Any) -> str:
    """Write float samples in [-1, 1] as a 16-bit mono PCM WAV file.

    Each sample is clipped to [-1, 1], multiplied by 32768, rounded to the
    nearest integer and clipped to the int16 range -32768..32767. Rounding
    moves a sample by at most 0.5/32768, and only +1.0 itself is clipped (to
    32767/32768), so :func:`load_wav` returns every sample within 1/32768 of
    the value saved. The parent directory is created if it is missing.

    Args:
        path: Destination path; an existing file is overwritten.
        x: A 1-D sequence of real samples; values outside [-1, 1] are
            clipped.
        fs: The sampling rate in Hz, a positive whole number.

    Returns:
        The path written, as a ``str``.

    Raises:
        ValueError: If ``x`` is not 1-D, is complex or not finite, or ``fs``
            is not a positive whole number.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N) for the integer copy.

    Examples:
        >>> import tempfile
        >>> x = np.array([0.25, -0.25, 1.0, -1.0, 3.0])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     y, fs = load_wav(save_wav(os.path.join(d, "b.wav"), x, 16000))
        >>> fs, y.tolist()
        (16000, [0.25, -0.25, 0.999969482421875, -1.0, 0.999969482421875])
    """
    samples = _real_samples(x)
    rate = _sample_rate(fs)
    ints = np.clip(np.round(np.clip(samples, -1.0, 1.0) * _FULL_SCALE), -32768, 32767)
    data = ints.astype("<i2").tobytes()
    target = _ensure_parent(path)
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data)
    return target


# --------------------------------------------------------------- processing


def process(
    path: PathLike,
    kind: str,
    cutoff: Any,
    out_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Filter a WAV file through the FFT and summarise what the filter did.

    The samples from :func:`load_wav` go through
    :func:`~src.fft_analyzer.signal.filters.fft_filter` (zero-pad to a power
    of two, FFT, brick-wall mask, inverse FFT, trim). The energy removed is
    ``1 - sum y^2 / sum x^2``: the mask is a projection in the padded
    domain and trimming only drops samples, so the filtered energy never
    exceeds the original and the fraction lies in [0, 1] (0 for a silent
    file). The peaks before and after are the k = 2 strongest local maxima of
    the exact-length magnitude spectrum, with any peak below 1% of the
    largest dropped as noise (see the module docstring).

    Args:
        path: The input WAV file (16-bit PCM, mono or stereo).
        kind: ``"low"``, ``"high"`` or ``"band"``.
        cutoff: The edge in Hz for low and high, a pair ``(lo, hi)`` for
            band.
        out_path: Where to write the filtered signal as 16-bit mono WAV, or
            ``None`` to write nothing.

    Returns:
        A dict with keys ``"fs"`` (int, Hz), ``"n"`` (int, samples),
        ``"energy_removed"`` (float in [0, 1]), ``"peaks_before"`` and
        ``"peaks_after"`` (lists of Hz, ascending) and ``"out_path"`` (the
        path written as a ``str``, or ``None``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty or not 16-bit, or ``kind`` or
            ``cutoff`` is invalid (from ``fft_filter``).

    Time Complexity:
        O(m log m) for the filter with m < 2N the padded length, plus the
        two exact-length spectra of
        :func:`~src.fft_analyzer.signal.analysis.peak_frequencies`.

    Space Complexity:
        O(m).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = make_example_wav(os.path.join(d, "in.wav"))
        ...     out = os.path.join(d, "high.wav")
        ...     summary = process(path, "high", 1000, out_path=out)
        ...     written = os.path.exists(out)
        >>> summary["peaks_after"], written
        ([3000.0], True)
        >>> 0.0 < summary["energy_removed"] < 1.0
        True
    """
    x, fs = load_wav(path)
    if x.shape[0] == 0:
        raise ValueError("the WAV file has no samples")
    y = fft_filter(x, fs, kind, cutoff)
    energy_in = float(np.sum(x * x))
    energy_out = float(np.sum(y * y))
    removed = 0.0 if energy_in == 0.0 else 1.0 - energy_out / energy_in
    removed = min(1.0, max(0.0, removed))
    written = None if out_path is None else save_wav(out_path, y, fs)
    return {
        "fs": fs,
        "n": int(x.shape[0]),
        "energy_removed": removed,
        "peaks_before": _strong_peaks(x, fs),
        "peaks_after": _strong_peaks(y, fs),
        "out_path": written,
    }


def make_example_wav(path: PathLike, seed: int = 42) -> str:
    """Write the synthetic example ``tone_noise.wav``: two tones plus white noise.

    The signal is 2 s at 16 kHz (32000 samples): ``0.5 sin(2 pi 440 t) +
    0.25 sin(2 pi 3000 t)`` from
    :func:`~src.fft_analyzer.signal.generators.multi_tone`, plus
    ``0.05 * white_noise(32000, derive_seed(seed, "tone_noise", "noise"))``.
    Both tones complete whole numbers of cycles (880 and 6000), so in the
    exact-length spectrum (bins 0.5 Hz apart) each falls on one bin. It is
    saved with :func:`save_wav` as 16-bit mono, 64,044 bytes.

    Args:
        path: Destination path (``src/fft_analyzer/examples/tone_noise.wav``
            for the shipped copy).
        seed: Master seed for the noise.

    Returns:
        The path written, as a ``str``.

    Raises:
        TypeError: If ``seed`` is not an ``int``.

    Time Complexity:
        O(N) for N = 32000 samples.

    Space Complexity:
        O(N).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     x, fs = load_wav(make_example_wav(os.path.join(d, "t.wav"), seed=42))
        >>> fs, x.shape, bool(np.abs(x).max() < 1.0)
        (16000, (32000,), True)
    """
    child = derive_seed(seed, "tone_noise", "noise")
    tones = multi_tone(list(EXAMPLE_TONES), list(EXAMPLE_AMPS), EXAMPLE_FS, EXAMPLE_DURATION)
    noise = EXAMPLE_NOISE * white_noise(tones.shape[0], child)
    return save_wav(path, tones + noise, EXAMPLE_FS)
