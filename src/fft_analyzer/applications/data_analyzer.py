"""Time-series analysis: detrending, the dominant period, smoothing and autocorrelation.

The method
----------
A periodic pattern buried in a trend and noise is hard to see sample by
sample and easy to see in the frequency domain, where a cycle of period P
in n samples is one spike at bin n / P. The analysis therefore makes the
round trip this package is built on: move the problem where it is easy (the
DFT), do the easy thing there (find the largest bin, or multiply spectra),
and move back when the answer lives in time (the inverse FFT behind
:func:`smooth` and :func:`autocorr_peaks`).

* :func:`detrend` removes the least-squares line ``a + b t``. Its 2 x 2
  normal equations are solved by LU with partial pivoting
  (:func:`~src.fft_analyzer.core.matrix.solve`), with t centred on its mean
  so the matrix is diagonal and the intercept and slope decouple. A trend
  left in place puts most of the energy in the lowest bins and hides the
  cycle.
* :func:`dominant_period` takes the largest one-sided magnitude bin k >= 1
  (DC excluded) and returns n / k samples. When n is not a power of two the
  exact-length :func:`~src.fft_analyzer.core.fft.dft` is used, never a
  zero-padded FFT, so a cycle that fits a whole number of times stays on one
  bin: 364 days hold 52 weeks exactly, bin 52, period 7.0.
* :func:`smooth` is a moving average, the linear convolution with a box of
  ``window`` ones divided by ``window``, through
  :func:`~src.fft_analyzer.core.fft.linear_convolve` with ``mode="same"``
  (identical to ``np.convolve(y, box, "same")``).
* :func:`autocorr_peaks` normalises the autocorrelation of the centred
  series by its lag-0 value and lists its local maxima; a cycle of period P
  shows up as a peak at lag P.

The synthetic series
--------------------
:func:`make_series` builds ``y[t] = 0.05 t + 10 sin(2 pi t / 7) + e[t]`` for
days t = 0..n-1, with ``e`` Gaussian noise of standard deviation 2 seeded
through :func:`src.utils.random_utils.derive_seed`. It is synthetic data made
to have a known weekly cycle, not market or sales data.

Examples:
    >>> y = make_series(364, seed=42)
    >>> dominant_period(detrend(y))
    7.0
    >>> autocorr_peaks(detrend(y), 10)
    [7]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numbers
from typing import Any, List

import numpy as np

from src.fft_analyzer.core.fft import dft, fft, linear_convolve
from src.fft_analyzer.core.matrix import solve
from src.fft_analyzer.signal.analysis import autocorrelation
from src.utils.random_utils import derive_seed

__all__ = [
    "SERIES_AMPLITUDE",
    "SERIES_NOISE",
    "SERIES_PERIOD",
    "SERIES_TREND",
    "autocorr_peaks",
    "detrend",
    "dominant_period",
    "make_series",
    "smooth",
]

#: Trend of the synthetic series, in units per day.
SERIES_TREND: float = 0.05

#: Amplitude of the synthetic series' weekly sine.
SERIES_AMPLITUDE: float = 10.0

#: Period of the synthetic series' cycle, in days (samples).
SERIES_PERIOD: float = 7.0

#: Standard deviation of the synthetic series' Gaussian noise.
SERIES_NOISE: float = 2.0


# ------------------------------------------------------------------ helpers


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` as an ``int`` if it is an integer of at least ``minimum``.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.
        minimum: The smallest value allowed.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an integer (``bool`` rejected).
        ValueError: If ``value < minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(7, "window", 1)
        7
        >>> _require_int(0, "window", 1)
        Traceback (most recent call last):
            ...
        ValueError: window must be at least 1, got 0
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def _series(y: Any, minimum: int) -> np.ndarray:
    """Return ``y`` as a finite, real, 1-D ``float64`` array of at least ``minimum`` samples.

    Args:
        y: The series.
        minimum: The shortest length allowed.

    Returns:
        A ``float64`` array.

    Raises:
        ValueError: If ``y`` is not 1-D, is complex, holds NaN or infinity,
            or is shorter than ``minimum``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) for the converted copy.

    Examples:
        >>> _series([1, 2, 3], 2)
        array([1., 2., 3.])
        >>> _series([1.0], 2)
        Traceback (most recent call last):
            ...
        ValueError: the series needs at least 2 samples, got 1
    """
    a = np.asarray(y)
    if np.iscomplexobj(a):
        raise ValueError("the series must be real")
    a = a.astype(np.float64)
    if a.ndim != 1:
        raise ValueError(f"the series must be 1-D, got shape {a.shape}")
    if a.shape[0] < minimum:
        raise ValueError(f"the series needs at least {minimum} samples, got {a.shape[0]}")
    if not np.all(np.isfinite(a)):
        raise ValueError("the series must be finite (no NaN or infinity)")
    return a


# ------------------------------------------------------------------- series


def make_series(n: int = 364, seed: int = 42) -> np.ndarray:
    """Build the synthetic daily series: a trend, a weekly cycle and noise.

    ``y[t] = 0.05 t + 10 sin(2 pi t / 7) + e[t]`` for t = 0..n-1, where
    ``e`` is Gaussian with standard deviation 2 drawn from
    ``numpy.random.default_rng(derive_seed(seed, "series", "noise"))``. With
    the default n = 364 (52 whole weeks) the cycle lands exactly on DFT bin
    52. The data are synthetic.

    Args:
        n: Number of days, at least 2.
        seed: Master seed for the noise.

    Returns:
        A ``float64`` array of length n.

    Raises:
        TypeError: If ``n`` or ``seed`` is not an ``int``.
        ValueError: If ``n < 2``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> y = make_series(364, seed=42)
        >>> y.shape, y.dtype
        ((364,), dtype('float64'))
        >>> bool(np.array_equal(make_series(30, seed=1), make_series(30, seed=1)))
        True
        >>> bool(np.array_equal(make_series(30, seed=1), make_series(30, seed=2)))
        False
    """
    count = _require_int(n, "n", 2)
    rng = np.random.default_rng(derive_seed(seed, "series", "noise"))
    t = np.arange(count, dtype=np.float64)
    cycle = SERIES_AMPLITUDE * np.sin(2.0 * np.pi * t / SERIES_PERIOD)
    return SERIES_TREND * t + cycle + rng.normal(0.0, SERIES_NOISE, size=count)


def detrend(y: Any) -> np.ndarray:
    """Subtract the least-squares straight line, solved with LU.

    With ``s = t - (n - 1) / 2`` (the day index centred on its mean), the
    line ``a + b s`` that minimises ``sum (y - a - b s)^2`` satisfies the
    normal equations

        [[n, sum s], [sum s, sum s^2]] [a, b] = [sum y, sum s y],

    solved by :func:`~src.fft_analyzer.core.matrix.solve`. The values of s
    are symmetric about 0, so ``sum s`` is exactly 0 and the matrix is
    diagonal: the intercept and slope decouple and each is one division,
    accurate to rounding. With raw t the off-diagonal ``sum t`` is of order
    ``n^2`` and couples the two unknowns. The residual ``y - a - b s`` is
    returned; its mean is 0 and it has no linear trend left.

    Args:
        y: A finite real 1-D series of at least 2 samples.

    Returns:
        The detrended ``float64`` series, same length.

    Raises:
        ValueError: If ``y`` is shorter than 2, not 1-D, complex or not
            finite.

    Time Complexity:
        O(n) for the sums; the 2 x 2 solve is O(1).

    Space Complexity:
        O(n).

    Examples:
        >>> np.round(detrend([1.0, 3.0, 5.0, 7.0]), 12) + 0
        array([0., 0., 0., 0.])
        >>> np.round(detrend([0.0, 1.0, 0.0]), 12)
        array([-0.33333333,  0.66666667, -0.33333333])
    """
    a = _series(y, 2)
    n = a.shape[0]
    s = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    normal = np.array([[float(n), float(np.sum(s))], [float(np.sum(s)), float(np.sum(s * s))]])
    rhs = np.array([float(np.sum(a)), float(np.sum(s * a))])
    intercept, slope = solve(normal, rhs)
    return a - (intercept + slope * s)


def dominant_period(y: Any) -> float:
    """Return the period, in samples, of the strongest cycle in the series.

    The one-sided magnitude spectrum ``|Y[k]|``, k = 1..n//2, is computed at
    the exact length n: with the radix-2 :func:`~src.fft_analyzer.core.fft.fft`
    when n is a power of two, otherwise with the direct
    :func:`~src.fft_analyzer.core.fft.dft` (zero-padding would move the bins
    off the cycle). DC (k = 0, the mean) is excluded. The largest bin k
    completes k cycles in n samples, so the period is ``n / k``; ties go to
    the lowest bin. Detrend first: a trend concentrates energy in the lowest
    bins and can outweigh the cycle.

    Args:
        y: A finite real 1-D series of at least 2 samples.

    Returns:
        ``n / k`` as a ``float``.

    Raises:
        ValueError: If ``y`` is shorter than 2, not 1-D, complex or not
            finite.

    Time Complexity:
        O(n log n) for a power-of-two n, Theta(n^2) otherwise.

    Space Complexity:
        O(n).

    Examples:
        >>> dominant_period(np.sin(2 * np.pi * np.arange(64) / 8))
        8.0
        >>> dominant_period(detrend(make_series(364, seed=42)))
        7.0
    """
    a = _series(y, 2)
    n = a.shape[0]
    spectrum = fft(a) if n & (n - 1) == 0 else dft(a)
    half = spectrum[1 : n // 2 + 1]
    mags = half.real ** 2 + half.imag ** 2
    k = int(np.argmax(mags)) + 1
    return n / k


def smooth(y: Any, window: int) -> np.ndarray:
    """Moving average of width ``window``, by FFT linear convolution.

    The series is convolved with the box ``[1/w] * w`` through
    :func:`~src.fft_analyzer.core.fft.linear_convolve` with ``mode="same"``,
    which keeps the centred n samples of the full convolution exactly as
    ``np.convolve(y, box, "same")`` does. Output sample i averages
    ``y[i - w // 2 .. i + (w - 1) // 2]``, so an even w leans one sample
    toward the past. Near the ends part of the box falls outside the series
    and counts as 0, so the first ``w // 2`` and the last ``(w - 1) // 2``
    values are pulled toward 0.

    Args:
        y: A finite real 1-D series of length n >= 1.
        window: The box width, an ``int`` with ``1 <= window <= n``.

    Returns:
        The smoothed ``float64`` series, length n.

    Raises:
        TypeError: If ``window`` is not an ``int``.
        ValueError: If ``window`` is outside 1..n, or ``y`` is empty, not
            1-D, complex or not finite.

    Time Complexity:
        O(n log n) (the FFT convolution, whatever the window).

    Space Complexity:
        O(n).

    Examples:
        >>> np.round(smooth([0.0, 0.0, 3.0, 0.0, 0.0], 3), 12) + 0
        array([0., 1., 1., 1., 0.])
        >>> np.round(smooth([2.0, 4.0, 6.0], 1), 12)
        array([2., 4., 6.])
    """
    a = _series(y, 1)
    w = _require_int(window, "window", 1)
    if w > a.shape[0]:
        raise ValueError(f"window must be at most the series length {a.shape[0]}, got {w}")
    box = np.full(w, 1.0 / w)
    return linear_convolve(a, box, mode="same")


def autocorr_peaks(y: Any, max_lag: int) -> List[int]:
    """Return the lags in 1..max_lag where the normalised autocorrelation peaks.

    The series is centred on its mean, its autocorrelation ``r[l]`` is
    computed through the zero-padded FFT
    (:func:`~src.fft_analyzer.signal.analysis.autocorrelation`), and
    ``rho[l] = r[l] / r[0]`` is the normalised autocorrelation (1 at lag 0).
    Lag l is a peak when ``rho[l] > rho[l - 1]`` and ``rho[l] >= rho[l + 1]``,
    the local-maximum rule of
    :func:`~src.fft_analyzer.signal.analysis.peak_frequencies`. The peaks are
    returned strongest first (ties: smaller lag first). A cycle of period P
    gives a peak at lag P.

    Args:
        y: A finite real 1-D series of at least 3 samples, not constant.
        max_lag: The largest lag examined, ``1 <= max_lag <= n - 2`` (lag
            ``max_lag + 1`` is needed as its right neighbour).

    Returns:
        A list of ``int`` lags sorted by ``rho`` descending (possibly empty).

    Raises:
        TypeError: If ``max_lag`` is not an ``int``.
        ValueError: If ``max_lag`` is outside 1..n-2, or the series is too
            short, not 1-D, complex, not finite or constant.

    Time Complexity:
        O(n log n) for the autocorrelation, plus O(L log L) to sort L peaks.

    Space Complexity:
        O(n).

    Examples:
        >>> autocorr_peaks(np.tile([1.0, 2.0, 0.0, -3.0], 10), 9)
        [4, 8]
        >>> autocorr_peaks(detrend(make_series(364, seed=42)), 10)
        [7]
    """
    a = _series(y, 3)
    n = a.shape[0]
    lag_max = _require_int(max_lag, "max_lag", 1)
    if lag_max > n - 2:
        raise ValueError(f"max_lag must be at most n - 2 = {n - 2}, got {lag_max}")
    r = autocorrelation(a - a.mean())
    if r[0] <= 0.0:
        raise ValueError("a constant series has no autocorrelation structure")
    rho = r / r[0]
    lags = np.arange(1, lag_max + 1)
    is_peak = (rho[lags] > rho[lags - 1]) & (rho[lags] >= rho[lags + 1])
    peaks = lags[is_peak]
    order = np.argsort(-rho[peaks], kind="stable")
    return [int(l) for l in peaks[order]]
