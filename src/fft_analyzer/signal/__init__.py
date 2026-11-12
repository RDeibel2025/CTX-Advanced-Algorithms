"""Signals: seeded generators, windows, FFT filters and spectral analysis (Week 12).

This sub-package shares its name with the standard library module
``signal``, so it is always imported absolutely, as
``src.fft_analyzer.signal``; no file in it is run directly as a script, and
``import signal`` anywhere else still gives the standard library module.

Three modules, each a round trip: move the problem where it is easy (the
FFT), do the easy thing there, and move back.

* :mod:`~src.fft_analyzer.signal.generators` - :func:`time_axis`,
  :func:`sine`, :func:`multi_tone`, the linear :func:`chirp`,
  :func:`white_noise` and :func:`pink_noise` (white noise whose spectrum is
  shaped by 1/sqrt(f) in the frequency domain). Every sequence is seeded.
* :mod:`~src.fft_analyzer.signal.filters` - the symmetric windows
  :func:`hann`, :func:`hamming` and :func:`blackman`, and :func:`fft_filter`,
  a brick-wall low, high or band mask on the bins of the zero-padded FFT,
  inverted and trimmed back.
* :mod:`~src.fft_analyzer.signal.analysis` - :func:`psd` (one-sided
  periodogram), :func:`welch_psd`, :func:`spectrogram` (Hann-windowed STFT
  magnitude), :func:`peak_frequencies` and :func:`autocorrelation` (the
  inverse FFT of the power spectrum, Wiener-Khinchin).

The public names below load from their module on first use, so importing one
module of this sub-package does not import the other two.

Examples:
    >>> from src.fft_analyzer.signal import multi_tone, peak_frequencies
    >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.0)
    >>> peak_frequencies(x, 1000, k=2)
    [50.0, 120.0]
    >>> import signal
    >>> signal.SIGINT.name
    'SIGINT'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each module.
_EXPORTS = {
    "generators": ("chirp", "multi_tone", "pink_noise", "sine", "time_axis", "white_noise"),
    "filters": ("blackman", "fft_filter", "hamming", "hann"),
    "analysis": ("autocorrelation", "peak_frequencies", "psd", "spectrogram", "welch_psd"),
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
        >>> import src.fft_analyzer.signal as sig
        >>> sig.hann(5)
        array([0. , 0.5, 1. , 0.5, 0. ])
        >>> sig.SIGINT
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.fft_analyzer.signal' has no attribute 'SIGINT'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
