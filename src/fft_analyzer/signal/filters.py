"""Window functions and brick-wall FFT filters, written out from their formulas.

Windows
-------
A finite record of N samples is implicitly multiplied by a rectangle, whose
spectrum has high side lobes, so a strong tone leaks into distant bins. A
window tapers the record to zero at its ends and trades a wider main lobe
for much lower side lobes. The three here are the symmetric cosine windows
numpy provides, for ``n = 0 .. N - 1``:

* :func:`hann`     ``0.5 - 0.5 cos(2 pi n / (N - 1))``
* :func:`hamming`  ``0.54 - 0.46 cos(2 pi n / (N - 1))``
* :func:`blackman` ``0.42 - 0.5 cos(2 pi n / (N - 1)) + 0.08 cos(4 pi n / (N - 1))``

They equal ``np.hanning``, ``np.hamming`` and ``np.blackman`` (``N = 1``
gives ``[1.0]`` and ``N = 0`` an empty array, as in numpy).

The FFT filter
--------------
:func:`fft_filter` is the frequency-domain round trip: move the problem where
it is easy (zero-pad x to the next power of two m and take its radix-2 FFT),
do the easy thing (multiply by a 0/1 mask that keeps the wanted bins, a
"brick wall" with no transition band), and move back (inverse FFT, keep the
real part, trim to the original length). Bin k of the padded transform sits
at ``k fs / m`` Hz for k <= m/2 and at the negative frequency
``(k - m) fs / m`` above that; the mask tests ``|f| = min(k, m - k) fs / m``,
so each negative frequency is kept exactly when its positive twin is. A
conjugate-symmetric spectrum times a symmetric real mask is still
conjugate-symmetric, so the output is real up to rounding.

Kinds: ``"low"`` keeps ``|f| <= cutoff``, ``"high"`` keeps ``|f| > cutoff``
(so a low-pass and a high-pass at the same cutoff add back to x), and
``"band"`` keeps ``lo <= |f| <= hi``. No library filter or transform is used;
the transforms are :func:`src.fft_analyzer.core.fft.fft` and ``ifft``.

Examples:
    >>> hann(5)
    array([0. , 0.5, 1. , 0.5, 0. ])
    >>> from src.fft_analyzer.signal.generators import multi_tone
    >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.0)
    >>> np.allclose(fft_filter(x, 1000, "low", 80) + fft_filter(x, 1000, "high", 80), x)
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from src.fft_analyzer.core.fft import fft, ifft

__all__ = ["blackman", "fft_filter", "hamming", "hann"]

_KINDS = ("low", "high", "band")


# ------------------------------------------------------------------ windows


def _cosine_window(N: int, coeffs) -> np.ndarray:
    """Return the symmetric window ``sum_j (-1)^j a_j cos(2 pi j n / (N - 1))``.

    Hann, Hamming and Blackman are all this sum with two or three
    coefficients ``a_j``. ``N = 1`` gives ``[1.0]`` and ``N = 0`` gives an
    empty array, matching numpy.

    Args:
        N: The window length, a non-negative integer.
        coeffs: The coefficients ``a_0, a_1, ...``.

    Returns:
        A float64 array of length N.

    Raises:
        TypeError: If ``N`` is not an integer (``bool`` rejected).
        ValueError: If ``N`` is negative.

    Time Complexity:
        O(N J) for J coefficients.

    Space Complexity:
        O(N).

    Examples:
        >>> _cosine_window(3, (0.5, 0.5))
        array([0., 1., 0.])
        >>> _cosine_window(1, (0.5, 0.5))
        array([1.])
    """
    if isinstance(N, bool) or not isinstance(N, numbers.Integral):
        raise TypeError(f"N must be an int, got {type(N).__name__}")
    if N < 0:
        raise ValueError(f"N must be non-negative, got {N}")
    if N <= 1:
        return np.ones(int(N), dtype=np.float64)
    phase = 2.0 * np.pi * np.arange(N) / (N - 1)
    w = np.zeros(int(N), dtype=np.float64)
    for j, a in enumerate(coeffs):
        w += (-1) ** j * a * np.cos(j * phase)
    return w


def hann(N: int) -> np.ndarray:
    """Return the symmetric Hann window ``0.5 - 0.5 cos(2 pi n / (N - 1))``.

    It is zero at both ends and 1 in the middle (for odd N). Its first side
    lobe is 31 dB below the main lobe and the side lobes fall off as
    1/f^3, which is why it is the default window of :func:`psd` and the
    spectrogram. Identical to ``np.hanning(N)``.

    Args:
        N: The window length, a non-negative integer.

    Returns:
        A float64 array of length N.

    Raises:
        TypeError: If ``N`` is not an integer.
        ValueError: If ``N`` is negative.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> hann(5)
        array([0. , 0.5, 1. , 0.5, 0. ])
        >>> bool(np.allclose(hann(64), np.hanning(64)))
        True
        >>> hann(1), hann(0)
        (array([1.]), array([], dtype=float64))
    """
    return _cosine_window(N, (0.5, 0.5))


def hamming(N: int) -> np.ndarray:
    """Return the symmetric Hamming window ``0.54 - 0.46 cos(2 pi n / (N - 1))``.

    The coefficients are chosen to cancel the Hann window's first side lobe,
    so the highest side lobe is about 43 dB down; the price is that the window
    stops at 0.08 instead of 0 at the ends, and distant side lobes decay only
    as 1/f. Identical to ``np.hamming(N)``.

    Args:
        N: The window length, a non-negative integer.

    Returns:
        A float64 array of length N.

    Raises:
        TypeError: If ``N`` is not an integer.
        ValueError: If ``N`` is negative.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> hamming(5)
        array([0.08, 0.54, 1.  , 0.54, 0.08])
        >>> bool(np.allclose(hamming(64), np.hamming(64)))
        True
    """
    return _cosine_window(N, (0.54, 0.46))


def blackman(N: int) -> np.ndarray:
    """Return the symmetric Blackman window.

    ``w[n] = 0.42 - 0.5 cos(2 pi n / (N - 1)) + 0.08 cos(4 pi n / (N - 1))``.
    The third cosine term pushes the side lobes to about 58 dB down at the
    cost of a main lobe three bins wide on each side. The end values are 0
    up to rounding (0.42 - 0.5 + 0.08). Identical to ``np.blackman(N)``.

    Args:
        N: The window length, a non-negative integer.

    Returns:
        A float64 array of length N.

    Raises:
        TypeError: If ``N`` is not an integer.
        ValueError: If ``N`` is negative.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(blackman(5), 12) + 0
        array([0.  , 0.34, 1.  , 0.34, 0.  ])
        >>> bool(np.allclose(blackman(64), np.blackman(64)))
        True
    """
    return _cosine_window(N, (0.42, 0.5, 0.08))


# --------------------------------------------------------------- the filter


def _band_edges(kind: str, cutoff):
    """Validate ``cutoff`` for ``kind`` and return ``(lo, hi)`` in Hz.

    A low-pass keeps ``[0, cutoff]``, a high-pass keeps ``(cutoff, inf)`` and
    a band-pass keeps ``[lo, hi]``.

    Args:
        kind: ``"low"``, ``"high"`` or ``"band"``.
        cutoff: A non-negative number for low and high, a pair ``(lo, hi)``
            with ``0 <= lo <= hi`` for band.

    Returns:
        The pair of edge frequencies as floats (for low and high both entries
        are the cutoff).

    Raises:
        ValueError: If ``kind`` is unknown, or ``cutoff`` has the wrong form
            or is negative, not finite, or (for band) has ``lo > hi``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _band_edges("band", (100, 140))
        (100.0, 140.0)
        >>> _band_edges("low", (1, 2))
        Traceback (most recent call last):
        ...
        ValueError: kind 'low' needs a single cutoff in Hz, got (1, 2)
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    if kind == "band":
        try:
            lo, hi = (float(v) for v in cutoff)
        except (TypeError, ValueError):
            raise ValueError(f"kind 'band' needs cutoff=(lo, hi) in Hz, got {cutoff!r}") from None
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
            raise ValueError(f"band edges must satisfy 0 <= lo <= hi, got {cutoff!r}")
        return lo, hi
    if not isinstance(cutoff, numbers.Real):
        raise ValueError(f"kind {kind!r} needs a single cutoff in Hz, got {cutoff!r}")
    c = float(cutoff)
    if not (math.isfinite(c) and c >= 0):
        raise ValueError(f"cutoff must be finite and non-negative, got {cutoff!r}")
    return c, c


def fft_filter(x, fs: float, kind: str, cutoff) -> np.ndarray:
    """Filter a real signal with a brick-wall mask on its zero-padded FFT.

    x (length L) is zero-padded to m, the next power of two at least L, and
    transformed with the radix-2 FFT. Bin k lies at ``|f| = min(k, m - k) fs / m``
    Hz, counting negative frequencies by their magnitude, and is kept (mask 1)
    or removed (mask 0) by that ``|f|`` alone:

    * ``"low"``:  keep ``|f| <= cutoff``;
    * ``"high"``: keep ``|f| > cutoff`` (the exact complement of ``"low"``);
    * ``"band"``: keep ``lo <= |f| <= hi`` with ``cutoff = (lo, hi)``.

    Because the mask is the same for f and -f, the masked spectrum is still
    conjugate-symmetric and its inverse FFT is real; the rounding residue in
    the imaginary part is dropped and the result is trimmed back to L
    samples. A brick wall has no transition band, so it removes a tone
    cleanly when the tone sits well inside the stop band, at the cost of
    some ringing near the ends of the record.

    Args:
        x: A non-empty 1-D real sequence.
        fs: The sampling rate in Hz, positive.
        kind: ``"low"``, ``"high"`` or ``"band"``.
        cutoff: The edge in Hz (non-negative) for low and high; a pair
            ``(lo, hi)`` with ``0 <= lo <= hi`` for band.

    Returns:
        A float64 array of length L, the filtered signal.

    Raises:
        ValueError: If x is empty, not 1-D or complex, ``fs`` is not positive,
            ``kind`` is unknown or ``cutoff`` does not fit ``kind``.

    Time Complexity:
        O(m log m), with m < 2L the padded length.

    Space Complexity:
        O(m).

    Examples:
        >>> from src.fft_analyzer.signal.generators import multi_tone, sine
        >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.0)
        >>> y = fft_filter(x, 1000, "low", 80)
        >>> y.shape, y.dtype
        ((1000,), dtype('float64'))
        >>> s = sine(2, 8, 2.0)
        >>> np.round(fft_filter(s, 8, "low", 2), 12) + 0
        array([ 0.,  1.,  0., -1.,  0.,  1.,  0., -1.,  0.,  1.,  0., -1.,  0.,
                1.,  0., -1.])
        >>> np.round(fft_filter(s, 8, "high", 2), 12) + 0
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> fft_filter(x, 1000, "notch", 50)
        Traceback (most recent call last):
        ...
        ValueError: kind must be one of ('low', 'high', 'band'), got 'notch'
    """
    a = np.asarray(x)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError("x must be a non-empty 1-D sequence")
    if np.iscomplexobj(a):
        raise ValueError("fft_filter expects a real signal")
    rate = float(fs)
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(f"fs must be a finite positive number, got {fs}")
    lo, hi = _band_edges(kind, cutoff)
    length = a.shape[0]
    m = 1 << (length - 1).bit_length()
    spectrum = fft(a.astype(np.float64), n=m)
    k = np.arange(m)
    f_abs = np.minimum(k, m - k) * rate / m
    if kind == "low":
        mask = f_abs <= hi
    elif kind == "high":
        mask = f_abs > lo
    else:
        mask = (f_abs >= lo) & (f_abs <= hi)
    return ifft(spectrum * mask).real[:length]
