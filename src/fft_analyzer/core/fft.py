"""The discrete Fourier transform by hand: the O(N^2) DFT and two radix-2 FFTs.

The transform
-------------
The DFT of ``x[0..N-1]`` is

    X[k] = sum_{n=0}^{N-1} x[n] exp(-2 pi i k n / N),    k = 0..N-1,

and the inverse divides by N. That is numpy's sign and scaling convention, so
every routine here can be compared with ``np.fft`` directly. ``np.fft`` and
``scipy.fft`` appear only in the tests and the benchmark, as the independent
cross-check (spec 1.1); nothing in this module calls them. Evaluated straight
from the definition the transform is N^2 complex multiply-adds: :func:`dft`.

Cooley-Tukey radix-2, decimation in time
----------------------------------------
When N is a power of two, split x into its even-indexed samples (transform E)
and odd-indexed samples (transform O), each of length N/2. With the twiddle
factor ``w = exp(-2 pi i / N)``, for k < N/2:

    X[k]       = E[k] + w^k O[k]
    X[k + N/2] = E[k] - w^k O[k]

This pair is the butterfly. One length-N transform costs two length-N/2
transforms plus N/2 twiddle multiplications, so T(N) = 2T(N/2) + O(N) =
O(N log N).

* :func:`fft_recursive` follows the recurrence literally and stops at
  n <= ``cutoff`` with a direct DFT-matrix product, where the matrix is cheap
  and Python call overhead would otherwise dominate.
* :func:`fft_iterative` unrolls the recursion bottom-up. The leaves of the
  recursion tree, read left to right, are the samples at bit-reversed indices,
  so it permutes once and then runs log2 N stages. Each stage is one
  vectorized butterfly over every group of that size, so the Python loop runs
  only log2 N times. :func:`fft_iterative_stages` keeps a copy after each
  stage for the animation.

Both draw their twiddles from one table ``exp(-2 pi i m / N)`` built once per
call, so rounding error grows like eps * log2 N rather than with N.

The inverse by conjugation
--------------------------
``ifft(X) = conj(fft(conj(X))) / N``: conjugating the input flips the sign of
the exponent, so the forward FFT computes the inverse sum, and one more
conjugation undoes the first. No second FFT has to be written.

The round trip
--------------
Convolution in time is pointwise multiplication in frequency. So
:func:`circular_convolve`, :func:`linear_convolve` and every FFT filter built
on :func:`fft2` move the problem where it is easy (the forward FFT), do the
easy thing (multiply bin by bin, O(N) instead of O(N^2)), and move back (the
inverse FFT).

Lengths
-------
The radix-2 routines require a power of two. :func:`fft` zero-pads or
truncates only when called with ``n=``, like ``np.fft.fft(x, n)``; without
``n=`` a length that is not a power of two raises ``ValueError``. :func:`dft`
accepts any length, and :func:`fft2`, :func:`ifft2` and
:func:`circular_convolve` fall back to it on lengths that are not a power of
two.

Examples:
    >>> X = fft([1, 2, 3, 4])
    >>> np.round(X, 10) + 0
    array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    >>> np.allclose(ifft(X), [1, 2, 3, 4])
    True
    >>> circular_convolve([1, 2, 3], [0, 1, 0.5])
    array([4. , 2.5, 2.5])

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import List, Optional

import numpy as np

__all__ = [
    "circular_convolve",
    "dft",
    "fft",
    "fft2",
    "fft_iterative",
    "fft_iterative_stages",
    "fft_recursive",
    "ifft",
    "ifft2",
    "linear_convolve",
    "rfft_freqs",
]

#: Largest DFT size whose full matrix is cached (64 x 64 complex is 64 KB).
_CACHED_DFT_MAX = 64

#: Elements in one temporary product block of the direct DFT (16 MB complex).
_DFT_CHUNK = 1 << 20

_METHODS = ("iterative", "recursive")
_CONVOLVE_MODES = ("full", "same")


# ------------------------------------------------------------------ helpers


def _is_power_of_two(n: int) -> bool:
    """Return True when ``n`` is a positive power of two.

    Args:
        n: The length to test.

    Returns:
        ``True`` for 1, 2, 4, 8, ...; ``False`` otherwise.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> [_is_power_of_two(n) for n in (0, 1, 2, 3, 4, 1024, 1000)]
        [False, True, True, False, True, True, False]
    """
    return n >= 1 and (n & (n - 1)) == 0


def _require_power_of_two(n: int, name: str) -> None:
    """Raise ``ValueError`` unless ``n`` is a power of two.

    Args:
        n: The length to check.
        name: The calling routine, for the message.

    Raises:
        ValueError: If ``n`` is not a power of two.

    Examples:
        >>> _require_power_of_two(8, "fft")
        >>> _require_power_of_two(6, "fft")
        Traceback (most recent call last):
        ...
        ValueError: fft needs a power-of-two length, got 6 (use fft(x, n=8) to zero-pad)
    """
    if not _is_power_of_two(n):
        nxt = 1 << max(0, n - 1).bit_length()
        raise ValueError(
            f"{name} needs a power-of-two length, got {n} "
            f"(use fft(x, n={nxt}) to zero-pad)"
        )


def _as_complex_vector(x) -> np.ndarray:
    """Return ``x`` as a fresh non-empty 1-D complex128 array.

    Args:
        x: A 1-D sequence or array of numbers.

    Returns:
        A new complex128 array (never a view of ``x``).

    Raises:
        ValueError: If ``x`` is not 1-D or is empty.

    Time Complexity:
        O(N).

    Space Complexity:
        O(N).

    Examples:
        >>> _as_complex_vector([1, 2])
        array([1.+0.j, 2.+0.j])
        >>> _as_complex_vector([[1, 2]])
        Traceback (most recent call last):
        ...
        ValueError: expected a 1-D sequence, got 2 dimensions
    """
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D sequence, got {a.ndim} dimensions")
    if a.size == 0:
        raise ValueError("expected a non-empty sequence")
    return a.astype(np.complex128, copy=True)


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is at least ``n`` (n >= 1).

    Args:
        n: A positive integer.

    Returns:
        ``2**ceil(log2 n)``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> [_next_power_of_two(n) for n in (1, 2, 3, 5, 8, 9)]
        [1, 2, 4, 8, 8, 16]
    """
    return 1 << (n - 1).bit_length()


@lru_cache(maxsize=64)
def _roots(n: int) -> np.ndarray:
    """Return the n-th roots of unity ``exp(-2 pi i m / n)`` for m = 0..n-1.

    Every DFT entry is one of these: ``exp(-2 pi i k j / n)`` is
    ``roots[(k * j) % n]``. Reducing ``k * j`` modulo n first keeps the angle
    below 2 pi, so large products do not lose phase accuracy. The array is
    cached and marked read-only.

    Args:
        n: The transform length (n >= 1).

    Returns:
        A read-only complex128 array of length n.

    Time Complexity:
        O(n) on the first call for each n.

    Space Complexity:
        O(n), kept in a small cache.

    Examples:
        >>> np.round(_roots(4), 12) + 0
        array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j])
    """
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    w.setflags(write=False)
    return w


def _twiddle_table(n: int) -> np.ndarray:
    """Return the radix-2 twiddle table ``exp(-2 pi i m / n)`` for m < n/2.

    A butterfly stage of group size s needs ``exp(-2 pi i j / s)`` for
    j < s/2, which is ``table[:: n // s]``. One table therefore serves every
    stage and every level of the recursion.

    Args:
        n: A power-of-two transform length.

    Returns:
        A read-only complex128 array of length n // 2.

    Time Complexity:
        O(n) on the first call for each n.

    Space Complexity:
        O(n).

    Examples:
        >>> np.round(_twiddle_table(8), 6) + 0
        array([ 1.      +0.j      ,  0.707107-0.707107j,  0.      -1.j      ,
               -0.707107-0.707107j])
    """
    return _roots(n)[: n // 2]


@lru_cache(maxsize=_CACHED_DFT_MAX)
def _dft_matrix(n: int) -> np.ndarray:
    """Return the n x n DFT matrix ``W[k, j] = exp(-2 pi i k j / n)``, cached.

    Only used for n <= ``_CACHED_DFT_MAX`` (the recursive FFT's base case calls
    it thousands of times at the same small n).

    Args:
        n: The transform length.

    Returns:
        A read-only complex128 array of shape (n, n).

    Time Complexity:
        O(n^2) on the first call for each n.

    Space Complexity:
        O(n^2).

    Examples:
        >>> np.round(_dft_matrix(2), 12) + 0
        array([[ 1.+0.j,  1.+0.j],
               [ 1.+0.j, -1.+0.j]])
    """
    j = np.arange(n)
    w = _roots(n)[np.outer(j, j) % n]
    w.setflags(write=False)
    return w


def _dft_last_axis(a: np.ndarray) -> np.ndarray:
    """Apply the direct O(n^2) DFT along the last axis of a complex array.

    ``X[..., k] = sum_j a[..., j] W[k, j]`` is written as an element-wise
    product with a block of DFT-matrix rows followed by a sum over j, so no
    library transform or matrix routine does the work. Output bins are
    processed in blocks so the temporary product stays near ``_DFT_CHUNK``
    elements however long the input is.

    Args:
        a: A complex array of shape (..., n) with n >= 1.

    Returns:
        A new complex128 array of the same shape.

    Time Complexity:
        O(B n^2) for B rows of length n.

    Space Complexity:
        O(B n + _DFT_CHUNK).

    Examples:
        >>> np.round(_dft_last_axis(np.array([[1, 1], [1, -1]], complex)), 12) + 0
        array([[2.+0.j, 0.+0.j],
               [0.+0.j, 2.+0.j]])
    """
    n = a.shape[-1]
    rows = a.size // n
    step = max(1, _DFT_CHUNK // (rows * n))
    if n <= _CACHED_DFT_MAX and step >= n:
        return (a[..., None, :] * _dft_matrix(n)).sum(axis=-1)
    roots = _roots(n)
    j = np.arange(n)
    out = np.empty(a.shape, dtype=np.complex128)
    expanded = a[..., None, :]
    for k0 in range(0, n, step):
        k1 = min(n, k0 + step)
        w = roots[(np.arange(k0, k1)[:, None] * j) % n]
        out[..., k0:k1] = (expanded * w).sum(axis=-1)
    return out


def _bit_reverse_indices(n: int) -> np.ndarray:
    """Return the bit-reversal permutation of ``range(n)`` for a power of two n.

    Index i with binary digits ``b_{L-1} ... b_0`` (L = log2 n) maps to the
    index whose digits are reversed. Built one bit at a time, so the loop runs
    L times, each a vectorized pass over n indices.

    Args:
        n: A power of two.

    Returns:
        An integer array ``rev`` with ``x[rev]`` the leaves of the FFT
        recursion in left-to-right order.

    Time Complexity:
        O(n log n) element operations in log2 n vectorized passes.

    Space Complexity:
        O(n).

    Examples:
        >>> _bit_reverse_indices(8)
        array([0, 4, 2, 6, 1, 5, 3, 7])
    """
    levels = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.intp)
    for b in range(levels):
        rev |= ((idx >> b) & 1) << (levels - 1 - b)
    return rev


def _radix2_last_axis(
    a: np.ndarray, stages: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """Run the iterative radix-2 FFT along the last axis of a complex array.

    After the bit-reversal permutation, a stage of group size s treats the
    array as contiguous groups of s values whose first half is the transform
    of the even samples and whose second half that of the odd samples, and
    replaces each group by its butterflies. All groups (and all rows) are
    done in one vectorized operation.

    Args:
        a: A complex array of shape (..., n), n a power of two.
        stages: If a list is given, a copy of the 1-D working array is
            appended after the permutation and after every stage.

    Returns:
        A new complex128 array of the same shape holding the transform.

    Time Complexity:
        O(B n log n) for B rows, in log2 n vectorized stages.

    Space Complexity:
        O(B n), plus O(n log n) when ``stages`` is recorded.

    Examples:
        >>> np.round(_radix2_last_axis(np.array([1, 2, 3, 4], complex)), 10) + 0
        array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    """
    n = a.shape[-1]
    lead = a.shape[:-1]
    out = np.ascontiguousarray(a[..., _bit_reverse_indices(n)], dtype=np.complex128)
    if stages is not None:
        stages.append(out.copy())
    table = _twiddle_table(n)
    half = 1
    while half < n:
        size = 2 * half
        twiddle = table[:: n // size]
        groups = out.reshape(lead + (n // size, size))  # a view: out is contiguous
        t = groups[..., half:] * twiddle
        groups[..., half:] = groups[..., :half] - t
        groups[..., :half] += t
        if stages is not None:
            stages.append(out.copy())
        half = size
    return out


def _transform_last_axis(a: np.ndarray) -> np.ndarray:
    """Transform along the last axis: radix-2 for a power of two, else the DFT.

    Args:
        a: A complex array of shape (..., n).

    Returns:
        A new complex128 array of the same shape.

    Time Complexity:
        O(B n log n) for a power of two n, O(B n^2) otherwise.

    Space Complexity:
        O(B n).

    Examples:
        >>> np.round(_transform_last_axis(np.ones(3, complex)), 10) + 0
        array([3.+0.j, 0.+0.j, 0.+0.j])
    """
    if _is_power_of_two(a.shape[-1]):
        return _radix2_last_axis(a)
    return _dft_last_axis(a)


def _fft_recurse(a: np.ndarray, n_top: int, cutoff: int, table: np.ndarray) -> np.ndarray:
    """Recursive step of :func:`fft_recursive` on a power-of-two length slice.

    Args:
        a: A 1-D complex array (possibly a strided view) of length n.
        n_top: The length of the top-level transform, which ``table`` serves.
        cutoff: Lengths at or below this use the direct DFT.
        table: ``_twiddle_table(n_top)``.

    Returns:
        The DFT of ``a`` as a new array.

    Time Complexity:
        O(n log(n / cutoff) + n cutoff).

    Space Complexity:
        O(n) live at each of the log2(n / cutoff) levels.

    Examples:
        >>> x = np.array([1, 2, 3, 4], complex)
        >>> np.round(_fft_recurse(x, 4, 1, _twiddle_table(4)), 10) + 0
        array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    """
    n = a.shape[0]
    if n <= cutoff:
        return _dft_last_axis(a)
    even = _fft_recurse(a[0::2], n_top, cutoff, table)
    odd = _fft_recurse(a[1::2], n_top, cutoff, table)
    t = table[:: n_top // n] * odd
    return np.concatenate((even + t, even - t))


def _is_real_input(*arrays) -> bool:
    """Return True when no argument has a complex dtype.

    Args:
        *arrays: Array-likes.

    Returns:
        ``True`` if every argument is real-valued by dtype.

    Examples:
        >>> _is_real_input([1, 2], [0.5])
        True
        >>> _is_real_input([1, 2], [1j])
        False
    """
    return not any(np.iscomplexobj(np.asarray(v)) for v in arrays)


# ----------------------------------------------------------- 1-D transforms


def dft(x) -> np.ndarray:
    """Compute the discrete Fourier transform straight from its definition.

    ``X[k] = sum_n x[n] exp(-2 pi i k n / N)`` for every k: N output bins,
    each an N-term sum, so N^2 complex multiply-adds. Each term's exponential
    is read from the table of N roots of unity at index ``(k n) mod N``, and
    the sums are formed as an element-wise product with blocks of DFT-matrix
    rows followed by a row sum (bounded memory for any N). It accepts any
    length, and is the reference for small N, the recursive FFT's base case
    and the fallback for lengths that are not a power of two.

    Args:
        x: A non-empty 1-D sequence of real or complex numbers.

    Returns:
        A complex128 array of length N with the same sign and scaling as
        ``np.fft.fft``.

    Raises:
        ValueError: If ``x`` is empty or not 1-D.

    Time Complexity:
        Theta(N^2).

    Space Complexity:
        O(N) for the result plus a bounded block of about 2^20 temporaries.

    Examples:
        >>> np.round(dft([1, 2, 3, 4]), 10) + 0
        array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
        >>> np.round(dft([1, 1, 1]), 10) + 0
        array([3.+0.j, 0.+0.j, 0.+0.j])
        >>> from src.utils.random_utils import derive_seed
        >>> rng = np.random.default_rng(derive_seed(42, "dft", "doctest"))
        >>> x = rng.standard_normal(16)
        >>> np.allclose(dft(x), fft(x))
        True
    """
    return _dft_last_axis(_as_complex_vector(x))


def fft_recursive(x, cutoff: int = 32) -> np.ndarray:
    """Compute the FFT by recursive Cooley-Tukey radix-2 decimation in time.

    Splits x into even- and odd-indexed halves, transforms each recursively,
    and joins them with the butterfly ``X[k] = E[k] + w^k O[k]``,
    ``X[k + n/2] = E[k] - w^k O[k]`` where ``w = exp(-2 pi i / n)``. At
    n <= ``cutoff`` it stops and multiplies by the DFT matrix directly, which
    is faster there than more Python calls. Twiddles for every level are
    strided slices of one table built for the top-level length.

    Args:
        x: A 1-D sequence whose length is a power of two.
        cutoff: Lengths at or below this use the direct DFT (at least 1;
            ``cutoff=1`` recurses all the way down to single samples).

    Returns:
        A complex128 array, the DFT of ``x`` (same convention as
        ``np.fft.fft``).

    Raises:
        ValueError: If the length is not a power of two, ``x`` is empty or not
            1-D, or ``cutoff < 1``.

    Time Complexity:
        O(N log N): T(n) = 2T(n/2) + O(n), with O(cutoff^2) per base case and
        N / cutoff base cases.

    Space Complexity:
        O(N) per recursion level live at once, O(N log N) allocated in total;
        recursion depth log2(N / cutoff).

    Examples:
        >>> np.round(fft_recursive([1, 2, 3, 4], cutoff=1), 10) + 0
        array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
        >>> from src.utils.random_utils import derive_seed
        >>> rng = np.random.default_rng(derive_seed(42, "fft_recursive", "doctest"))
        >>> x = rng.standard_normal(128) + 1j * rng.standard_normal(128)
        >>> np.allclose(fft_recursive(x, cutoff=8), dft(x))
        True
        >>> fft_recursive([1, 2, 3])
        Traceback (most recent call last):
        ...
        ValueError: fft_recursive needs a power-of-two length, got 3 (use fft(x, n=4) to zero-pad)
    """
    a = _as_complex_vector(x)
    n = a.shape[0]
    _require_power_of_two(n, "fft_recursive")
    cutoff = operator.index(cutoff)
    if cutoff < 1:
        raise ValueError(f"cutoff must be at least 1, got {cutoff}")
    return _fft_recurse(a, n, cutoff, _twiddle_table(n))


def fft_iterative(x) -> np.ndarray:
    """Compute the FFT bottom-up: bit-reversal, then log2 N butterfly stages.

    The recursion of :func:`fft_recursive` ends at single samples taken in
    bit-reversed index order, so permuting x into that order once lets the
    combining run from the leaves up with no recursion. Stage s (group size
    2^s) applies the butterfly to every group in a single vectorized
    operation, so the Python loop runs only log2 N times.

    Args:
        x: A 1-D sequence whose length is a power of two.

    Returns:
        A complex128 array, the DFT of ``x`` (same convention as
        ``np.fft.fft``).

    Raises:
        ValueError: If the length is not a power of two, or ``x`` is empty or
            not 1-D.

    Time Complexity:
        O(N log N): log2 N stages of O(N) vectorized work.

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(fft_iterative([1, 0, 0, 0, 0, 0, 0, 0]), 10) + 0
        array([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j])
        >>> from src.utils.random_utils import derive_seed
        >>> rng = np.random.default_rng(derive_seed(42, "fft_iterative", "doctest"))
        >>> x = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        >>> np.allclose(fft_iterative(x), dft(x))
        True
        >>> fft_iterative(np.ones(12))
        Traceback (most recent call last):
        ...
        ValueError: fft_iterative needs a power-of-two length, got 12 (use fft(x, n=16) to zero-pad)
    """
    a = _as_complex_vector(x)
    _require_power_of_two(a.shape[0], "fft_iterative")
    return _radix2_last_axis(a)


def fft_iterative_stages(x) -> List[np.ndarray]:
    """Return the working array of :func:`fft_iterative` after every step.

    The first entry is x after the bit-reversal permutation; entry s
    (s = 1..log2 N) is the array after the stage with group size 2^s. The
    last entry is therefore the FFT. Used to animate the butterflies.

    Args:
        x: A 1-D sequence whose length is a power of two.

    Returns:
        A list of ``log2(N) + 1`` complex128 arrays of length N, each its own
        copy.

    Raises:
        ValueError: If the length is not a power of two, or ``x`` is empty or
            not 1-D.

    Time Complexity:
        O(N log N).

    Space Complexity:
        O(N log N) for the stored copies.

    Examples:
        >>> stages = fft_iterative_stages([1, 2, 3, 4])
        >>> len(stages)
        3
        >>> stages[0]
        array([1.+0.j, 3.+0.j, 2.+0.j, 4.+0.j])
        >>> np.round(stages[1], 10) + 0
        array([ 4.+0.j, -2.+0.j,  6.+0.j, -2.+0.j])
        >>> np.round(stages[-1], 10) + 0
        array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
    """
    a = _as_complex_vector(x)
    _require_power_of_two(a.shape[0], "fft_iterative_stages")
    stages: List[np.ndarray] = []
    _radix2_last_axis(a, stages)
    return stages


def fft(x, n: Optional[int] = None, method: str = "iterative") -> np.ndarray:
    """Compute the FFT of x with the chosen radix-2 algorithm.

    With ``n`` given, x is truncated or zero-padded to length n first, like
    ``np.fft.fft(x, n)``. Without it, the length must already be a power of
    two: nothing is padded silently, because padding changes the frequency
    grid.

    Args:
        x: A non-empty 1-D sequence of real or complex numbers.
        n: Optional transform length, a power of two.
        method: ``"iterative"`` (:func:`fft_iterative`, the default) or
            ``"recursive"`` (:func:`fft_recursive`).

    Returns:
        A complex128 array of length n (or len(x)).

    Raises:
        ValueError: If the length is not a power of two, ``n < 1``, ``x`` is
            empty or not 1-D, or ``method`` is unknown.

    Time Complexity:
        O(N log N).

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(fft([1, 2, 3, 4], method="recursive"), 10) + 0
        array([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])
        >>> np.round(fft([1, 1, 1], n=4), 10) + 0
        array([3.+0.j, 0.-1.j, 1.+0.j, 0.+1.j])
        >>> fft([1, 1, 1])
        Traceback (most recent call last):
        ...
        ValueError: fft needs a power-of-two length, got 3 (use fft(x, n=4) to zero-pad)
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    a = _as_complex_vector(x)
    if n is not None:
        n = operator.index(n)
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if n < a.shape[0]:
            a = a[:n]
        elif n > a.shape[0]:
            a = np.concatenate((a, np.zeros(n - a.shape[0], dtype=np.complex128)))
    _require_power_of_two(a.shape[0], "fft")
    if method == "recursive":
        return fft_recursive(a)
    return _radix2_last_axis(a)


def ifft(X, method: str = "iterative") -> np.ndarray:
    """Compute the inverse FFT by the conjugation trick ``conj(fft(conj(X)))/N``.

    Conjugating X turns ``exp(+2 pi i k n / N)`` into ``exp(-2 pi i k n / N)``,
    so the forward FFT evaluates the inverse sum; conjugating again and
    dividing by N gives ``x[n] = (1/N) sum_k X[k] exp(+2 pi i k n / N)``.

    Args:
        X: A 1-D spectrum whose length is a power of two.
        method: ``"iterative"`` or ``"recursive"``, as for :func:`fft`.

    Returns:
        A complex128 array of length N (``.real`` recovers a real signal).

    Raises:
        ValueError: If the length is not a power of two, ``X`` is empty or not
            1-D, or ``method`` is unknown.

    Time Complexity:
        O(N log N).

    Space Complexity:
        O(N).

    Examples:
        >>> np.round(ifft([10, -2 + 2j, -2, -2 - 2j]), 10) + 0
        array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j])
        >>> np.allclose(ifft(fft([3, 1, 4, 1]), method="recursive"), [3, 1, 4, 1])
        True
    """
    a = _as_complex_vector(X)
    return np.conj(fft(np.conj(a), method=method)) / a.shape[0]


# ----------------------------------------------------------- 2-D transforms


def _as_complex_matrix(img) -> np.ndarray:
    """Return ``img`` as a fresh non-empty 2-D complex128 array.

    Args:
        img: A 2-D array-like.

    Returns:
        A new complex128 array.

    Raises:
        ValueError: If ``img`` is not 2-D or is empty.

    Examples:
        >>> _as_complex_matrix([[1, 2]]).shape
        (1, 2)
    """
    a = np.asarray(img)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {a.ndim} dimensions")
    if a.size == 0:
        raise ValueError("expected a non-empty 2-D array")
    return a.astype(np.complex128, copy=True)


def fft2(img) -> np.ndarray:
    """Compute the 2-D DFT: transform every row, then every column.

    The 2-D DFT ``F[u, v] = sum_{r, c} img[r, c] exp(-2 pi i (u r / H + v c / W))``
    separates into 1-D transforms, so it is H row transforms followed by W
    column transforms. Each axis uses the vectorized radix-2 FFT (all rows at
    once) when its length is a power of two and the direct DFT otherwise.

    Args:
        img: A non-empty 2-D array of shape (H, W).

    Returns:
        A complex128 array of shape (H, W), the same as ``np.fft.fft2``.

    Raises:
        ValueError: If ``img`` is not 2-D or is empty.

    Time Complexity:
        O(H W log(H W)) for power-of-two sides; a side of length L that is not
        a power of two costs O(L) per element along it instead of O(log L).

    Space Complexity:
        O(H W).

    Examples:
        >>> np.round(fft2([[1, 2], [3, 4]]), 10) + 0
        array([[10.+0.j, -2.+0.j],
               [-4.+0.j,  0.+0.j]])
        >>> np.round(fft2(np.ones((3, 4))), 10) + 0
        array([[12.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
               [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]])
    """
    a = _as_complex_matrix(img)
    a = _transform_last_axis(a)  # rows
    return _transform_last_axis(a.T).T  # columns


def ifft2(F) -> np.ndarray:
    """Compute the inverse 2-D DFT by conjugation: ``conj(fft2(conj(F)))/(H W)``.

    Args:
        F: A non-empty 2-D spectrum of shape (H, W).

    Returns:
        A complex128 array of shape (H, W) (``.real`` recovers a real image).

    Raises:
        ValueError: If ``F`` is not 2-D or is empty.

    Time Complexity:
        Same as :func:`fft2`.

    Space Complexity:
        O(H W).

    Examples:
        >>> img = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> np.allclose(ifft2(fft2(img)), img)
        True
        >>> np.allclose(ifft2(fft2(np.arange(15.0).reshape(3, 5))).real,
        ...             np.arange(15.0).reshape(3, 5))
        True
    """
    a = _as_complex_matrix(F)
    return np.conj(fft2(np.conj(a))) / a.size


# --------------------------------------------------------------- convolution


def circular_convolve(a, b) -> np.ndarray:
    """Circularly convolve two equal-length sequences through the transform.

    ``c[k] = sum_j a[j] b[(k - j) mod N]``. By the convolution theorem the
    DFT of c is the bin-by-bin product of the DFTs of a and b, so the
    O(N^2) sum becomes forward transforms, N multiplications and an inverse
    transform. The radix-2 FFT is used when N is a power of two and the
    direct :func:`dft` (with its conjugation-trick inverse) otherwise, so any
    equal length works. For real inputs the true result is real and the
    imaginary part left by rounding is below 1e-9, so ``.real`` is returned;
    a complex input gives the complex result.

    Args:
        a: A non-empty 1-D sequence.
        b: A 1-D sequence of the same length.

    Returns:
        A float64 array of length N for real inputs, else complex128.

    Raises:
        ValueError: If the lengths differ, or either input is empty or not
            1-D.

    Time Complexity:
        O(N log N) for a power of two N, O(N^2) otherwise.

    Space Complexity:
        O(N).

    Examples:
        >>> circular_convolve([1, 2, 3], [0, 1, 0.5])
        array([4. , 2.5, 2.5])
        >>> np.round(circular_convolve([1, 2, 3, 4], [0, 1, 0, 0]), 10) + 0
        array([4., 1., 2., 3.])
    """
    real = _is_real_input(a, b)
    x = _as_complex_vector(a)
    y = _as_complex_vector(b)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"circular_convolve needs equal lengths, got {n} and {y.shape[0]}")
    if _is_power_of_two(n):
        c = ifft(_radix2_last_axis(x) * _radix2_last_axis(y))
    else:
        product = _dft_last_axis(x) * _dft_last_axis(y)
        c = np.conj(_dft_last_axis(np.conj(product))) / n
    return c.real if real else c


def linear_convolve(a, b, mode: str = "full") -> np.ndarray:
    """Linearly convolve two sequences with zero-padded radix-2 FFTs.

    The full linear convolution of lengths m and k has m + k - 1 terms. Both
    inputs are zero-padded to the next power of two at least m + k - 1, so the
    circular convolution of the padded arrays never wraps around and its first
    m + k - 1 entries are the linear convolution. ``mode="same"`` returns the
    centre part exactly as ``np.convolve(a, b, "same")`` does: length
    max(m, k) (that is len(a) when a is the longer input, the usual
    signal-and-kernel case), starting at index (min(m, k) - 1) // 2 of the
    full result. Real inputs give ``.real`` (the rounding residue in the
    imaginary part is below 1e-9); a complex input gives the complex result.

    Args:
        a: A non-empty 1-D sequence of length m.
        b: A non-empty 1-D sequence of length k.
        mode: ``"full"`` (length m + k - 1) or ``"same"`` (length max(m, k)).

    Returns:
        A float64 array for real inputs, else complex128.

    Raises:
        ValueError: If either input is empty or not 1-D, or ``mode`` is
            unknown.

    Time Complexity:
        O((m + k) log(m + k)).

    Space Complexity:
        O(m + k).

    Examples:
        >>> np.round(linear_convolve([1, 2, 3], [0, 1, 0.5]), 10) + 0
        array([0. , 1. , 2.5, 4. , 1.5])
        >>> np.round(linear_convolve([1, 2, 3], [0, 1, 0.5], mode="same"), 10) + 0
        array([1. , 2.5, 4. ])
    """
    if mode not in _CONVOLVE_MODES:
        raise ValueError(f"mode must be one of {_CONVOLVE_MODES}, got {mode!r}")
    real = _is_real_input(a, b)
    x = _as_complex_vector(a)
    y = _as_complex_vector(b)
    m, k = x.shape[0], y.shape[0]
    full_len = m + k - 1
    size = _next_power_of_two(full_len)
    xp = np.zeros(size, dtype=np.complex128)
    yp = np.zeros(size, dtype=np.complex128)
    xp[:m] = x
    yp[:k] = y
    c = ifft(_radix2_last_axis(xp) * _radix2_last_axis(yp))[:full_len]
    if mode == "same":
        start = (min(m, k) - 1) // 2
        c = c[start : start + max(m, k)]
    return c.real if real else c


# --------------------------------------------------------------- frequencies


def rfft_freqs(n: int, fs: float) -> np.ndarray:
    """Return the frequency in Hz of FFT bins 0..n//2 for n samples at rate fs.

    Bin k of an n-point transform completes k cycles over the n samples,
    which last n / fs seconds, so its frequency is ``k * fs / n``. Bins above
    n // 2 are the negative frequencies of a real signal and are left out
    (the one-sided spectrum), matching ``np.fft.rfftfreq(n, 1 / fs)``.

    Args:
        n: The number of samples (transform length), at least 1.
        fs: The sampling rate in Hz, positive.

    Returns:
        A float64 array of length ``n // 2 + 1``, from 0 to fs/2 for even n.

    Raises:
        ValueError: If ``n < 1`` or ``fs <= 0``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> rfft_freqs(8, 1000)
        array([  0., 125., 250., 375., 500.])
        >>> rfft_freqs(5, 10.0)
        array([0., 2., 4.])
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not fs > 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return np.arange(n // 2 + 1) * fs / n
