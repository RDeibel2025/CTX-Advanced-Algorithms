"""Core numerics: the FFT, matrix algorithms and numerical-stability utilities (Week 12).

Three modules, each a round trip: move the problem where it is easy, do the
easy thing there, and move back.

* :mod:`~src.fft_analyzer.core.fft` - :func:`dft` from the definition in
  O(N^2), and the radix-2 Cooley-Tukey FFTs, :func:`fft_recursive` (decimation
  in time with a DFT-matrix base case) and :func:`fft_iterative` (bit-reversal
  permutation, then log2 N vectorized butterfly stages, kept frame by frame by
  :func:`fft_iterative_stages`). :func:`ifft` uses the conjugation trick, and
  :func:`circular_convolve`, :func:`linear_convolve`, :func:`fft2`,
  :func:`ifft2` and :func:`rfft_freqs` build on them. Convolution moves to the
  frequency domain, becomes one multiplication per bin, and comes back by the
  inverse FFT.
* :mod:`~src.fft_analyzer.core.matrix` - :func:`matmul_classical`,
  :func:`matmul_blocked` and :func:`strassen`; :func:`lu_decompose` with
  partial pivoting, :func:`lu_solve`, :func:`solve` and :func:`determinant`;
  :func:`qr_householder` and :func:`qr_gram_schmidt` (classical or modified);
  and :func:`power_method`. A linear system moves to triangular factors, is
  solved by two substitutions, and the row permutation is undone.
* :mod:`~src.fft_analyzer.core.numerical` - :func:`abs_error`,
  :func:`rel_error` and :func:`norm` by hand; the condition numbers
  :func:`cond_estimate` and :func:`cond2_power` with :func:`is_ill_conditioned`
  and :func:`hilbert`; :func:`machine_epsilon`, :func:`cancellation_examples`
  and :func:`kahan_sum`.

The function ``fft`` is not exported here, because ``src.fft_analyzer.core.fft``
is the module of that name; import the function as
``from src.fft_analyzer.core.fft import fft`` or ``from src.fft_analyzer import
fft``.

The public names below load from their module on first use, so importing one
module of this sub-package does not import the other two.

Examples:
    >>> from src.fft_analyzer.core import determinant, linear_convolve
    >>> [round(v, 9) for v in linear_convolve([1, 2, 3], [0, 1, 0.5]).tolist()]
    [0.0, 1.0, 2.5, 4.0, 1.5]
    >>> round(determinant([[2, 1, 1], [4, -6, 0], [-2, 7, 2]]), 9)
    -16.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each module (``fft`` itself is the module of that name).
_EXPORTS = {
    "fft": (
        "circular_convolve", "dft", "fft2", "fft_iterative", "fft_iterative_stages",
        "fft_recursive", "ifft", "ifft2", "linear_convolve", "rfft_freqs",
    ),
    "matrix": (
        "determinant", "lu_decompose", "lu_solve", "matmul_blocked", "matmul_classical",
        "power_method", "qr_gram_schmidt", "qr_householder", "solve", "strassen",
    ),
    "numerical": (
        "abs_error", "cancellation_examples", "cond2_power", "cond_estimate", "hilbert",
        "is_ill_conditioned", "kahan_sum", "machine_epsilon", "norm", "rel_error",
    ),
}

#: Each public name mapped to the module that provides it.
_HOME = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_HOME)


def __getattr__(name: str) -> Any:
    """Load a public name from its module on first use.

    Python calls this only for a name the sub-package does not already hold.

    Args:
        name: The attribute being looked up.

    Returns:
        The object of that name from its module.

    Raises:
        AttributeError: If ``name`` is not a public name of this sub-package.

    Time Complexity:
        O(1), plus the module import the first time.

    Space Complexity:
        O(1).

    Examples:
        >>> import src.fft_analyzer.core as core
        >>> core.fft_recursive.__name__
        'fft_recursive'
        >>> core.cond2_power.__module__
        'src.fft_analyzer.core.numerical'
        >>> core.svd
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.fft_analyzer.core' has no attribute 'svd'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
