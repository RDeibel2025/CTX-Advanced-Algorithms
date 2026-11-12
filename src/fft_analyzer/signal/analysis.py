"""Spectral analysis by hand: periodogram, Welch, spectrogram, peaks, autocorrelation.

Every estimator here makes the same round trip: move the problem where it is
easy (transform the signal, or a windowed piece of it, to the frequency
domain), do the easy thing (read off squared magnitudes, pick maxima, or
square the spectrum), and move back when the answer lives in time (the
inverse FFT in :func:`autocorrelation`). The transforms are this package's
own (:mod:`src.fft_analyzer.core.fft`); no ``np.fft``, ``scipy.fft`` or
``scipy.signal`` routine is used.

The estimators
--------------
* :func:`psd` - the one-sided periodogram with density scaling,
  ``P[k] = c_k |X_w[k]|^2 / (fs sum w^2)``, where ``X_w`` is the DFT of the
  windowed signal ``w * x`` and ``c_k = 2`` for the bins that have a negative
  frequency twin (``0 < k < N/2``) and 1 for DC and Nyquist. With the
  rectangular window ``sum_k P[k] fs / N`` is the mean power of x (Parseval).
* :func:`welch_psd` - Welch's method: cut x into overlapping Hann-windowed
  segments, take each segment's periodogram and average them. Averaging K
  segments divides the variance of the estimate by about K, at the price of
  a frequency grid ``fs / seg`` instead of ``fs / N``.
* :func:`spectrogram` - the short-time Fourier transform magnitude: the
  same Hann-windowed frames, kept side by side instead of averaged, so the
  spectrum can be followed through time.
* :func:`peak_frequencies` - the k largest local maxima of the one-sided
  magnitude spectrum, returned in Hz.
* :func:`autocorrelation` - by the Wiener-Khinchin theorem the
  autocorrelation is the inverse transform of the power spectrum ``|X|^2``.
  Zero-padding to at least 2N - 1 points makes the circular correlation that
  the transform computes equal to the linear one, so the result equals
  ``np.correlate(x, x, "full")[N - 1:]``.

Exact-length spectra
--------------------
The periodogram, the peaks and the spectrogram transform at the exact signal
length, never a zero-padded one, so a tone with a whole number of cycles
falls in a single bin: 1000 samples at 1000 Hz give bins exactly 1 Hz apart
and the 50 Hz tone is bin 50, not 49.8 Hz. A power-of-two length uses the
radix-2 :func:`fft`. Any other length N = 2^a q (q odd) takes one
Cooley-Tukey decimation-in-time step: the 2^a interleaved sub-sequences
``x[r::2^a]`` of length q each get the direct :func:`dft`, and

    ``X[k] = sum_r exp(-2 pi i r k / N) D_r[k mod q]``

combines them. That costs O(N (q + 2^a)) instead of the O(N^2) of one direct
DFT (32000 samples: q = 125, about 70 times less work) and gives the same
values up to rounding.

Examples:
    >>> from src.fft_analyzer.signal.generators import multi_tone
    >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.0)
    >>> peak_frequencies(x, 1000, k=2)
    [50.0, 120.0]
    >>> np.allclose(autocorrelation([1.0, 2.0, 3.0]), [14.0, 8.0, 3.0])
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import List, Optional, Tuple

import numpy as np

from src.fft_analyzer.core.fft import dft, fft, ifft, rfft_freqs
from src.fft_analyzer.signal.filters import blackman, hamming, hann

__all__ = [
    "autocorrelation",
    "peak_frequencies",
    "psd",
    "spectrogram",
    "welch_psd",
]

_WINDOWS = {
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
    "boxcar": lambda n: np.ones(n, dtype=np.float64),
}


# ------------------------------------------------------------------ helpers


def _real_signal(x) -> np.ndarray:
    """Return ``x`` as a non-empty 1-D float64 array, rejecting complex input.

    Args:
        x: A 1-D sequence of real numbers.

    Returns:
        A float64 array.

    Raises:
        ValueError: If ``x`` is empty, not 1-D or complex.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> _real_signal([1, 2])
        array([1., 2.])
        >>> _real_signal([[1.0]])
        Traceback (most recent call last):
        ...
        ValueError: x must be a non-empty 1-D sequence
    """
    a = np.asarray(x)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError("x must be a non-empty 1-D sequence")
    if np.iscomplexobj(a):
        raise ValueError("expected a real signal, got complex values")
    return a.astype(np.float64)


def _positive_rate(fs) -> float:
    """Return ``fs`` as a float if it is a finite positive sampling rate.

    Args:
        fs: The sampling rate in Hz.

    Returns:
        ``float(fs)``.

    Raises:
        ValueError: If ``fs`` is not finite and positive.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _positive_rate(1000)
        1000.0
    """
    rate = float(fs)
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"fs must be a finite positive number, got {fs}")
    return rate


def _length(value, name: str, upper: Optional[int] = None) -> int:
    """Return ``value`` if it is an integer in ``[1, upper]``.

    Args:
        value: The length to check.
        name: Its parameter name, for the error message.
        upper: The largest allowed value, or ``None`` for no limit.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an integer (``bool`` rejected).
        ValueError: If it is below 1 or above ``upper``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _length(256, "seg", 1000)
        256
        >>> _length(2000, "seg", 1000)
        Traceback (most recent call last):
        ...
        ValueError: seg must be between 1 and 1000, got 2000
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1 or (upper is not None and value > upper):
        limit = "at least 1" if upper is None else f"between 1 and {upper}"
        raise ValueError(f"{name} must be {limit}, got {value}")
    return int(value)


def _spectrum(a: np.ndarray) -> np.ndarray:
    """Return the exact-length DFT of a 1-D array (see the module docstring).

    A power-of-two length goes to the radix-2 :func:`fft`. Otherwise
    N = p q with p the largest power of two dividing N and q odd; the p
    interleaved sub-sequences ``a[r::p]`` are transformed with :func:`dft`
    (length q) and combined as ``X[k] = sum_r W_N^{r k} D_r[k mod q]``, one
    Cooley-Tukey decimation-in-time step. An odd N (p = 1) is a single
    direct DFT.

    Args:
        a: A non-empty 1-D real or complex array of length N.

    Returns:
        A complex128 array of length N equal to ``np.fft.fft(a)`` up to
        rounding.

    Time Complexity:
        O(N log N) for a power of two, otherwise O(N (q + p)).

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(_spectrum(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])), 6) + 0
        array([21.+0.j      , -3.+5.196152j, -3.+1.732051j, -3.+0.j      ,
               -3.-1.732051j, -3.-5.196152j])
        >>> x = np.arange(12.0) ** 2
        >>> bool(np.allclose(_spectrum(x), dft(x)))
        True
    """
    n = a.shape[0]
    if n & (n - 1) == 0:
        return fft(a)
    p = n & -n
    if p == 1:
        return dft(a)
    q = n // p
    k = np.arange(n)
    roots = np.exp(-2j * np.pi * k / n)
    k_mod_q = k % q
    out = np.zeros(n, dtype=np.complex128)
    for r in range(p):
        out += roots[(r * k) % n] * dft(a[r::p])[k_mod_q]
    return out


def _one_sided_density(power: np.ndarray, n: int, fs: float, w_energy: float) -> np.ndarray:
    """Scale ``|X[k]|^2`` for k = 0..n//2 into a one-sided power spectral density.

    Divides by ``fs * sum(w^2)`` and doubles every bin that has a negative
    frequency twin, that is ``0 < k < n/2`` (DC and, for even n, Nyquist
    appear once in the full spectrum and are not doubled).

    Args:
        power: Squared magnitudes of bins 0..n//2 (one row or a stack).
        n: The transform length.
        fs: The sampling rate in Hz.
        w_energy: ``sum(w^2)`` of the window used.

    Returns:
        A new float64 array of the same shape.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _one_sided_density(np.array([4.0, 4.0, 4.0]), 4, 1.0, 4.0)
        array([1., 2., 1.])
    """
    out = power / (fs * w_energy)
    out[..., 1 : (n + 1) // 2] *= 2.0
    return out


def _window(name: str, n: int) -> np.ndarray:
    """Return the named window of length n, refusing an all-zero window.

    Args:
        name: ``"hann"``, ``"hamming"``, ``"blackman"`` or ``"boxcar"``.
        n: The window length, at least 1.

    Returns:
        A float64 array of length n.

    Raises:
        ValueError: If the name is unknown or the window is all zeros (Hann
            and Blackman of length 2 are ``[0, 0]``).

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _window("hann", 3)
        array([0., 1., 0.])
        >>> _window("hann", 2)
        Traceback (most recent call last):
        ...
        ValueError: the hann window of length 2 is all zeros; use a longer segment
    """
    if name not in _WINDOWS:
        raise ValueError(f"window must be one of {tuple(_WINDOWS)}, got {name!r}")
    w = _WINDOWS[name](n)
    if not np.any(w != 0):
        raise ValueError(f"the {name} window of length {n} is all zeros; use a longer segment")
    return w


def _frame_starts(n: int, seg: int, step: int) -> np.ndarray:
    """Return the start index of every full frame of length ``seg``, ``step`` apart.

    Args:
        n: The signal length.
        seg: The frame length, at most n.
        step: The distance between frame starts, at least 1.

    Returns:
        An int array ``0, step, 2 step, ...`` up to ``n - seg``.

    Time Complexity:
        O(n / step).

    Space Complexity:
        O(n / step).

    Examples:
        >>> _frame_starts(10, 4, 2)
        array([0, 2, 4, 6])
    """
    return np.arange(0, n - seg + 1, step)


# ---------------------------------------------------------------- estimators


def psd(x, fs: float, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the power spectral density with the one-sided periodogram.

    The signal is multiplied by the window w, transformed at its exact
    length N (no zero-padding), and bins 0..N//2 are scaled to a density:

        ``P[k] = c_k |DFT(w x)[k]|^2 / (fs sum_n w[n]^2)``

    with ``c_k = 2`` for ``0 < k < N/2`` (the power of the negative frequency
    -f is folded onto +f) and ``c_k = 1`` for DC and Nyquist. The units are
    signal units squared per Hz. Dividing by ``sum w^2`` makes a white
    noise's expected density independent of the window. The mean is not
    removed. This is the same estimate as ``scipy.signal.periodogram`` with
    ``detrend=False`` and a symmetric window.

    Args:
        x: A non-empty 1-D real signal.
        fs: The sampling rate in Hz, positive.
        window: ``"hann"`` (default), ``"hamming"``, ``"blackman"`` or
            ``"boxcar"`` (no window).

    Returns:
        ``(freqs, power)``: two float64 arrays of length N//2 + 1, the bin
        frequencies ``k fs / N`` in Hz and the density.

    Raises:
        ValueError: If x is empty, not 1-D or complex, ``fs`` is not
            positive, the window name is unknown or the window is all zeros.

    Time Complexity:
        O(N log N) for a power of two N; O(N (q + p)) for N = p q with p a
        power of two and q odd (see the module docstring).

    Space Complexity:
        O(N).

    Examples:
        >>> from src.fft_analyzer.signal.generators import sine
        >>> x = sine(100, 1000, 1.0)
        >>> f, p = psd(x, 1000, window="boxcar")
        >>> float(f[int(np.argmax(p))])
        100.0
        >>> bool(np.isclose(np.sum(p) * 1000 / len(x), np.mean(x ** 2)))
        True
        >>> f, p = psd([1.0, 0.0, -1.0, 0.0], 4.0, window="boxcar")
        >>> f, np.round(p, 12) + 0
        (array([0., 1., 2.]), array([0. , 0.5, 0. ]))
    """
    a = _real_signal(x)
    rate = _positive_rate(fs)
    n = a.shape[0]
    w = _window(window, n)
    spectrum = _spectrum(a * w)[: n // 2 + 1]
    power = spectrum.real ** 2 + spectrum.imag ** 2
    return rfft_freqs(n, rate), _one_sided_density(power, n, rate, float(np.sum(w * w)))


def welch_psd(x, fs: float, seg: int, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the power spectral density by Welch's averaged periodograms.

    x is cut into frames of ``seg`` samples whose starts are
    ``step = seg - floor(seg * overlap)`` apart (only full frames are used;
    trailing samples that do not fill one are ignored). Each frame is
    multiplied by the Hann window and gives a one-sided periodogram exactly as
    :func:`psd`, and the K periodograms are averaged. A single periodogram's
    value at each frequency has a standard deviation about as large as its
    mean however long x is; averaging K roughly independent frames divides
    the variance by about K, which is what makes the log-log slope of pink
    noise measurable. The frequency resolution is ``fs / seg``.

    Args:
        x: A 1-D real signal with at least ``seg`` samples.
        fs: The sampling rate in Hz, positive.
        seg: The frame length, between 1 and ``len(x)`` (2 is refused: its Hann
            window is all zeros).
        overlap: The fraction of a frame shared with the next, in [0, 1).

    Returns:
        ``(freqs, psd)``: two float64 arrays of length seg//2 + 1.

    Raises:
        TypeError: If ``seg`` is not an integer.
        ValueError: If x is empty, not 1-D or complex, ``fs`` is not
            positive, ``seg`` is outside ``[1, len(x)]`` or its Hann window is
            all zeros, or ``overlap`` is outside [0, 1).

    Time Complexity:
        O(K seg log seg) for K frames of power-of-two length.

    Space Complexity:
        O(seg) beyond the input.

    Examples:
        >>> from src.fft_analyzer.signal.generators import white_noise
        >>> f, p = welch_psd(white_noise(8192, seed=42), 1.0, seg=256)
        >>> f.shape, p.shape
        ((129,), (129,))
        >>> round(float(np.mean(p[1:-1])), 1)
        2.0
        >>> welch_psd([1.0, 2.0, 3.0], 1.0, seg=4)
        Traceback (most recent call last):
        ...
        ValueError: seg must be between 1 and 3, got 4
    """
    a = _real_signal(x)
    rate = _positive_rate(fs)
    n = a.shape[0]
    size = _length(seg, "seg", n)
    frac = float(overlap)
    if not (math.isfinite(frac) and 0.0 <= frac < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = size - int(math.floor(size * frac))
    w = _window("hann", size)
    starts = _frame_starts(n, size, step)
    total = np.zeros(size // 2 + 1, dtype=np.float64)
    for s in starts:
        spectrum = _spectrum(a[s : s + size] * w)[: size // 2 + 1]
        total += spectrum.real ** 2 + spectrum.imag ** 2
    mean_power = total / starts.shape[0]
    return rfft_freqs(size, rate), _one_sided_density(mean_power, size, rate, float(np.sum(w * w)))


def spectrogram(
    x, fs: float, seg: int = 256, hop: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the short-time Fourier transform magnitude with a Hann window.

    Frame j covers samples ``j hop .. j hop + seg - 1`` (full frames only).
    Each frame is multiplied by ``hann(seg)`` and transformed, and column j of
    the result holds the magnitudes ``|X_j[k]|`` of bins k = 0..seg//2. A
    longer ``seg`` resolves frequency more finely (``fs / seg`` Hz per bin)
    but blurs events in time; ``hop`` sets how far apart the columns are.

    Args:
        x: A 1-D real signal with at least ``seg`` samples.
        fs: The sampling rate in Hz, positive.
        seg: The frame length, between 1 and ``len(x)`` (2 is refused: its Hann
            window is all zeros).
        hop: The distance between frame starts in samples, at least 1.

    Returns:
        ``(freqs, times, magnitude)``: the bin frequencies in Hz (length
        seg//2 + 1), the time in seconds of each frame's centre
        ``(j hop + seg / 2) / fs``, and a float64 array of shape
        ``(len(freqs), len(times))``.

    Raises:
        TypeError: If ``seg`` or ``hop`` is not an integer.
        ValueError: If x is empty, not 1-D or complex, ``fs`` is not
            positive, ``seg`` is outside ``[1, len(x)]`` or its Hann window is
            all zeros, or ``hop < 1``.

    Time Complexity:
        O(F seg log seg) for F = 1 + (len(x) - seg) // hop frames of
        power-of-two length.

    Space Complexity:
        O(F seg) for the magnitude array.

    Examples:
        >>> from src.fft_analyzer.signal.generators import sine
        >>> f, t, m = spectrogram(sine(125, 1000, 1.0), 1000, seg=64, hop=32)
        >>> m.shape, len(f), len(t)
        ((33, 30), 33, 30)
        >>> float(f[int(np.argmax(m[:, 0]))]), float(t[0])
        (125.0, 0.032)
    """
    a = _real_signal(x)
    rate = _positive_rate(fs)
    n = a.shape[0]
    size = _length(seg, "seg", n)
    step = _length(hop, "hop")
    w = _window("hann", size)
    starts = _frame_starts(n, size, step)
    half = size // 2 + 1
    magnitude = np.empty((half, starts.shape[0]), dtype=np.float64)
    for j, s in enumerate(starts):
        magnitude[:, j] = np.abs(_spectrum(a[s : s + size] * w)[:half])
    times = (starts + size / 2.0) / rate
    return rfft_freqs(size, rate), times, magnitude


def peak_frequencies(x, fs: float, k: int = 2) -> List[float]:
    """Return the frequencies in Hz of the k strongest spectral peaks, ascending.

    The one-sided magnitude spectrum ``|X[0..N//2]|`` is computed at the
    exact length N without a window, so a tone with a whole number of cycles
    is one bin. A peak is an interior local maximum: bin i with
    ``|X[i]| > |X[i - 1]|`` and ``|X[i]| >= |X[i + 1]|``. As in
    ``scipy.signal.find_peaks``, the end bins (DC and the last bin) are never
    peaks, so a signal's mean does not count as a tone. The k peaks with the
    largest magnitude are kept and returned sorted by frequency; fewer are
    returned when the spectrum has fewer than k peaks.

    Args:
        x: A non-empty 1-D real signal.
        fs: The sampling rate in Hz, positive.
        k: How many peaks to return, at least 1.

    Returns:
        A list of at most k floats, the peak frequencies ``i fs / N`` in Hz in
        ascending order.

    Raises:
        TypeError: If ``k`` is not an integer.
        ValueError: If x is empty, not 1-D or complex, ``fs`` is not positive
            or ``k < 1``.

    Time Complexity:
        O(N log N) for a power of two N, O(N (q + p)) otherwise, plus
        O(N log N) to rank the maxima.

    Space Complexity:
        O(N).

    Examples:
        >>> from src.fft_analyzer.signal.generators import multi_tone
        >>> x = multi_tone([50, 120, 300], [1.0, 0.5, 0.2], fs=1000, duration=1.0)
        >>> peak_frequencies(x, 1000, k=2)
        [50.0, 120.0]
        >>> peak_frequencies(x, 1000, k=3)
        [50.0, 120.0, 300.0]
        >>> peak_frequencies(x, 1000, k=1)
        [50.0]
    """
    a = _real_signal(x)
    rate = _positive_rate(fs)
    count = _length(k, "k")
    n = a.shape[0]
    mag = np.abs(_spectrum(a)[: n // 2 + 1])
    if mag.shape[0] < 3:
        return []
    inner = mag[1:-1]
    is_peak = (inner > mag[:-2]) & (inner >= mag[2:])
    idx = np.flatnonzero(is_peak) + 1
    strongest = idx[np.argsort(-mag[idx], kind="stable")[:count]]
    freqs = rfft_freqs(n, rate)
    return [float(freqs[i]) for i in np.sort(strongest)]


def autocorrelation(x) -> np.ndarray:
    """Return the autocorrelation at lags 0..N-1 through the zero-padded FFT.

    ``r[l] = sum_n x[n + l] x[n]``, the unnormalised sum that
    ``np.correlate(x, x, "full")[N - 1:]`` returns. By the Wiener-Khinchin
    theorem the DFT of the circular autocorrelation is the power spectrum
    ``|X[k]|^2``. Zero-padding x to m >= 2N - 1 points (the next power of two)
    leaves enough zeros that the circular correlation never wraps around, so
    ``ifft(|fft(x, m)|^2)[:N]`` is the linear one. The input is real, so the
    imaginary rounding residue is dropped. Divide by ``r[0]`` for the
    normalised autocorrelation.

    Args:
        x: A non-empty 1-D real sequence.

    Returns:
        A float64 array of length N; ``r[0]`` is the energy ``sum x^2``.

    Raises:
        ValueError: If x is empty, not 1-D or complex.

    Time Complexity:
        O(N log N).

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(autocorrelation([1.0, 2.0, 3.0]), 10) + 0
        array([14.,  8.,  3.])
        >>> r = autocorrelation(np.tile([1.0, -1.0, 0.0, 0.0], 8))
        >>> int(np.argmax(r[1:])) + 1
        4
    """
    a = _real_signal(x)
    n = a.shape[0]
    m = 1 << (2 * n - 2).bit_length()
    spectrum = fft(a, n=m)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    return ifft(power).real[:n]
