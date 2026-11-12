"""Seeded test signals: sines, multi-tone sums, a linear chirp, white and pink noise.

Every signal is a 1-D float64 array of samples (no time axis attached), so
it can go straight into the FFT filters and the spectral estimators. The
deterministic signals are sampled on :func:`time_axis`, ``t[n] = n / fs`` for
``n = 0 .. round(fs * duration) - 1``; one second at 1000 Hz is exactly 1000
samples, so a tone at an integer frequency completes a whole number of cycles
and lands in a single DFT bin.

The two noises
--------------
* :func:`white_noise` draws independent standard normal samples from
  ``numpy.random.default_rng(seed)``. Its expected power spectrum is flat: the
  same power in every frequency band of equal width.
* :func:`pink_noise` has power proportional to 1/f, equal power per octave.
  It is made by a round trip through the frequency domain: move the problem
  where it is easy (FFT of white noise), do the easy thing (multiply every
  bin by ``1/sqrt(|f|)``, which scales power by ``1/|f|``, with the DC bin set
  to 0), and move back (inverse FFT). The factor depends only on ``|f|``, so
  positive and negative frequencies are scaled alike, the spectrum keeps its
  conjugate symmetry and the result is real. Finally it is divided by its
  standard deviation, so every pink signal has unit standard deviation.

Randomness comes only from ``numpy.random.default_rng(seed)`` built from the
caller's integer seed (for example one from
:func:`src.utils.random_utils.derive_seed`), never from the global numpy or
:mod:`random` state, so the same seed always gives the same samples.

Examples:
    >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.0)
    >>> x.shape, x.dtype
    ((1000,), dtype('float64'))
    >>> bool(np.array_equal(white_noise(8, seed=42), white_noise(8, seed=42)))
    True
    >>> round(float(np.std(pink_noise(4096, seed=42))), 12)
    1.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Sequence

import numpy as np

from src.fft_analyzer.core.fft import fft, ifft

__all__ = [
    "chirp",
    "multi_tone",
    "pink_noise",
    "sine",
    "time_axis",
    "white_noise",
]

#: Seed used when the caller does not pass one (the course-wide default).
_DEFAULT_SEED = 42


# ------------------------------------------------------------------ helpers


def _require_positive(value, name: str) -> float:
    """Return ``value`` as a float if it is a finite positive number.

    Args:
        value: The number to check.
        name: Its parameter name, for the error message.

    Returns:
        ``float(value)``.

    Raises:
        ValueError: If ``value`` is not finite and positive.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_positive(1000, "fs")
        1000.0
        >>> _require_positive(0, "fs")
        Traceback (most recent call last):
        ...
        ValueError: fs must be a finite positive number, got 0
    """
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value}")
    return v


def _require_count(n, name: str, minimum: int) -> int:
    """Return ``n`` if it is an integer of at least ``minimum``.

    Args:
        n: The count to check.
        name: Its parameter name, for the error message.
        minimum: The smallest allowed value.

    Returns:
        ``int(n)``.

    Raises:
        TypeError: If ``n`` is not an integer (``bool`` rejected).
        ValueError: If ``n < minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_count(5, "n", 0)
        5
        >>> _require_count(1, "n", 2)
        Traceback (most recent call last):
        ...
        ValueError: n must be at least 2, got 1
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(n).__name__}")
    if n < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {n}")
    return int(n)


def _require_seed(seed) -> int:
    """Return ``seed`` if it is a non-negative integer seed.

    Args:
        seed: The seed to check.

    Returns:
        ``int(seed)``.

    Raises:
        TypeError: If ``seed`` is not an integer (``bool`` and ``None``
            rejected: every generator in the course is seeded).
        ValueError: If ``seed`` is negative.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_seed(42)
        42
        >>> _require_seed(None)
        Traceback (most recent call last):
        ...
        TypeError: seed must be an int, got NoneType
    """
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return int(seed)


# ------------------------------------------------------------ deterministic


def time_axis(fs: float, duration: float) -> np.ndarray:
    """Return the sample times ``n / fs`` for ``round(fs * duration)`` samples.

    The first sample is at t = 0 and the last at ``(N - 1) / fs``, so the
    interval [0, duration) is covered without its right end point. Rounding
    ``fs * duration`` (rather than truncating it) keeps a value such as
    ``fs = 1000, duration = 0.3`` at exactly 300 samples despite
    floating-point error in the product.

    Args:
        fs: The sampling rate in Hz, finite and positive.
        duration: The length in seconds, finite and non-negative.

    Returns:
        A float64 array ``np.arange(round(fs * duration)) / fs``.

    Raises:
        ValueError: If ``fs`` is not positive or ``duration`` is negative or
            not finite.

    Time Complexity:
        O(N) for N = round(fs * duration) samples.

    Space Complexity:
        O(N).

    Examples:
        >>> time_axis(4, 1.0)
        array([0.  , 0.25, 0.5 , 0.75])
        >>> len(time_axis(1000, 0.3))
        300
        >>> time_axis(4, -1)
        Traceback (most recent call last):
        ...
        ValueError: duration must be finite and non-negative, got -1
    """
    rate = _require_positive(fs, "fs")
    d = float(duration)
    if not (math.isfinite(d) and d >= 0):
        raise ValueError(f"duration must be finite and non-negative, got {duration}")
    return np.arange(round(rate * d)) / rate


def sine(
    freq: float, fs: float, duration: float, amp: float = 1.0, phase: float = 0.0
) -> np.ndarray:
    """Sample the sinusoid ``amp * sin(2 pi freq t + phase)``.

    The tone is evaluated on :func:`time_axis`. When ``freq * duration`` is a
    whole number the samples hold a whole number of cycles, so the DFT of the
    signal has all its energy in bin ``freq * duration`` (and its mirror),
    with magnitude ``amp * N / 2``.

    Args:
        freq: The frequency in Hz (any real number; above fs/2 it aliases).
        fs: The sampling rate in Hz, positive.
        duration: The length in seconds, non-negative.
        amp: The peak amplitude.
        phase: The phase offset in radians at t = 0.

    Returns:
        A float64 array of ``round(fs * duration)`` samples.

    Raises:
        ValueError: If ``fs`` or ``duration`` is invalid (see
            :func:`time_axis`).

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(sine(1, 4, 1.0), 12) + 0
        array([ 0.,  1.,  0., -1.])
        >>> np.round(sine(1, 4, 1.0, amp=2, phase=np.pi / 2), 12) + 0
        array([ 2.,  0., -2.,  0.])
    """
    t = time_axis(fs, duration)
    return float(amp) * np.sin(2.0 * np.pi * float(freq) * t + float(phase))


def multi_tone(
    freqs: Sequence[float], amps: Sequence[float], fs: float, duration: float
) -> np.ndarray:
    """Sum several sinusoids, ``sum_i amps[i] * sin(2 pi freqs[i] t)``.

    This is the test signal of the filter experiments: for example
    ``multi_tone([50, 120], [1, 0.5], 1000, 1.0)`` is
    ``sin(2 pi 50 t) + 0.5 sin(2 pi 120 t)``, whose 1000-point DFT has
    magnitude 500 in bin 50 and 250 in bin 120.

    Args:
        freqs: Frequencies in Hz.
        amps: One amplitude per frequency.
        fs: The sampling rate in Hz, positive.
        duration: The length in seconds, non-negative.

    Returns:
        A float64 array of ``round(fs * duration)`` samples (all zeros when
        ``freqs`` is empty).

    Raises:
        ValueError: If ``freqs`` and ``amps`` differ in length, or ``fs`` or
            ``duration`` is invalid.

    Time Complexity:
        O(K N) for K tones of N samples.

    Space Complexity:
        O(N).

    Examples:
        >>> x = multi_tone([1, 2], [1.0, 0.5], fs=8, duration=1.0)
        >>> np.round(x, 6) + 0
        array([ 0.      ,  1.207107,  1.      ,  0.207107,  0.      , -0.207107,
               -1.      , -1.207107])
        >>> multi_tone([1, 2], [1.0], fs=8, duration=1.0)
        Traceback (most recent call last):
        ...
        ValueError: freqs and amps must have the same length, got 2 and 1
    """
    f = [float(v) for v in freqs]
    a = [float(v) for v in amps]
    if len(f) != len(a):
        raise ValueError(f"freqs and amps must have the same length, got {len(f)} and {len(a)}")
    t = time_axis(fs, duration)
    out = np.zeros(t.shape[0], dtype=np.float64)
    for freq, amp in zip(f, a):
        out += amp * np.sin(2.0 * np.pi * freq * t)
    return out


def chirp(f0: float, f1: float, fs: float, duration: float) -> np.ndarray:
    """Sample a linear chirp sweeping from ``f0`` Hz to ``f1`` Hz over ``duration``.

    The instantaneous frequency rises (or falls) linearly,
    ``f(t) = f0 + (f1 - f0) t / duration``. Phase is the integral of
    ``2 pi f(t)``, so the samples are

        ``sin(2 pi (f0 t + (f1 - f0) t^2 / (2 duration)))``.

    A spectrogram of it shows one ridge moving steadily from f0 to f1, which
    is how the spectrogram test checks time-frequency resolution.

    Args:
        f0: The starting frequency in Hz.
        f1: The frequency in Hz reached at t = duration.
        fs: The sampling rate in Hz, positive.
        duration: The sweep length in seconds, positive.

    Returns:
        A float64 array of ``round(fs * duration)`` samples with unit
        amplitude.

    Raises:
        ValueError: If ``fs`` or ``duration`` is not finite and positive.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> x = chirp(100, 400, fs=2000, duration=1.0)
        >>> x.shape
        (2000,)
        >>> np.allclose(chirp(5, 5, fs=100, duration=1.0), sine(5, 100, 1.0))
        True
    """
    d = _require_positive(duration, "duration")
    t = time_axis(fs, d)
    rate = (float(f1) - float(f0)) / d
    return np.sin(2.0 * np.pi * (float(f0) * t + 0.5 * rate * t * t))


# -------------------------------------------------------------------- noise


def white_noise(n: int, seed: int = _DEFAULT_SEED) -> np.ndarray:
    """Draw ``n`` independent standard normal samples (Gaussian white noise).

    The samples come from ``numpy.random.default_rng(seed).standard_normal``,
    a private generator, so the same seed gives the same samples whatever else
    the program has drawn. Independent samples have a flat expected power
    spectrum, which is what "white" means.

    Args:
        n: The number of samples, at least 0.
        seed: A non-negative integer seed (for example from
            :func:`src.utils.random_utils.derive_seed`).

    Returns:
        A float64 array of length n with mean 0 and variance 1 in
        expectation.

    Raises:
        TypeError: If ``n`` or ``seed`` is not an integer.
        ValueError: If ``n`` or ``seed`` is negative.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> np.round(white_noise(3, seed=42), 6)
        array([ 0.304717, -1.039984,  0.750451])
        >>> bool(np.array_equal(white_noise(5, 7), white_noise(5, 7)))
        True
        >>> white_noise(0, 1).shape
        (0,)
    """
    count = _require_count(n, "n", 0)
    rng = np.random.default_rng(_require_seed(seed))
    return rng.standard_normal(count)


def pink_noise(n: int, seed: int = _DEFAULT_SEED) -> np.ndarray:
    """Make 1/f (pink) noise by shaping white noise in the frequency domain.

    White noise of length m (the next power of two at least n, so the
    radix-2 FFT applies) is transformed, bin k is multiplied by
    ``1/sqrt(|f_k|)`` with ``|f_k| = min(k, m - k) / m`` cycles per sample
    (the DC bin is set to 0, since 1/sqrt(0) is undefined), and the result is
    transformed back. Power is the squared magnitude, so it now falls off as
    1/|f|: a log-log plot of the power spectrum has slope -1. The factor is
    the same for f and -f, so the shaped spectrum stays conjugate-symmetric
    and the inverse transform is real; the rounding residue in its imaginary
    part is discarded. The first n samples are kept and divided by their
    standard deviation, so the output has unit standard deviation.

    Args:
        n: The number of samples, at least 2.
        seed: A non-negative integer seed for the underlying white noise.

    Returns:
        A float64 array of length n with standard deviation 1.

    Raises:
        TypeError: If ``n`` or ``seed`` is not an integer.
        ValueError: If ``n < 2`` or ``seed`` is negative.

    Time Complexity:
        O(m log m) with m the next power of two at least n.

    Space Complexity:
        O(m).

    Examples:
        >>> p = pink_noise(1024, seed=42)
        >>> p.shape, round(float(np.std(p)), 12)
        ((1024,), 1.0)
        >>> bool(np.array_equal(pink_noise(1000, 3), pink_noise(1000, 3)))
        True
    """
    count = _require_count(n, "n", 2)
    m = 1 << (count - 1).bit_length()
    spectrum = fft(white_noise(m, seed))
    k = np.arange(m)
    f_abs = np.minimum(k, m - k) / m
    scale = np.zeros(m, dtype=np.float64)
    scale[1:] = 1.0 / np.sqrt(f_abs[1:])
    shaped = ifft(spectrum * scale).real[:count]
    centred = shaped - shaped.mean()
    std = math.sqrt(float(np.mean(centred * centred)))
    return shaped / std
