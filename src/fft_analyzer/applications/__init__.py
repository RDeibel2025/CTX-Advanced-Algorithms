"""Applications: audio filtering, an image pipeline and time-series analysis (Week 12).

Three modules, each the same round trip on real-looking data: move the
problem where it is easy (the FFT or the DCT), do the easy thing there, and
move back.

* :mod:`~src.fft_analyzer.applications.audio_processor` - :func:`load_wav`
  and :func:`save_wav` (16-bit PCM through the standard-library ``wave``
  module), :func:`process` (an FFT filter with a summary of the energy removed
  and the peaks before and after) and :func:`make_example_wav`, the synthetic
  two-tone ``tone_noise.wav`` described by the ``EXAMPLE_*`` constants.
* :mod:`~src.fft_analyzer.applications.image_processor` - :func:`load_image`
  and :func:`save_image` (grayscale 0-255 through pillow),
  :func:`make_test_scene` (the synthetic 256 x 256 ``test_scene.png``) and
  :func:`run_pipeline` (blur, Sobel, denoise and compression at quality 10,
  50 and 90).
* :mod:`~src.fft_analyzer.applications.data_analyzer` - :func:`make_series`
  (a synthetic daily series with a weekly cycle, described by the
  ``SERIES_*`` constants), :func:`detrend` (least squares through LU),
  :func:`dominant_period`, :func:`smooth` and :func:`autocorr_peaks`.

The public names below load from their module on first use.

Examples:
    >>> from src.fft_analyzer.applications import detrend, dominant_period, make_series
    >>> dominant_period(detrend(make_series(364, seed=42)))
    7.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each module.
_EXPORTS = {
    "audio_processor": (
        "EXAMPLE_AMPS", "EXAMPLE_DURATION", "EXAMPLE_FS", "EXAMPLE_NOISE", "EXAMPLE_TONES",
        "load_wav", "make_example_wav", "process", "save_wav",
    ),
    "image_processor": (
        "BLUR_SIGMA", "DENOISE_RADIUS", "QUALITIES", "SCENE_SIZE", "load_image",
        "make_test_scene", "run_pipeline", "save_image",
    ),
    "data_analyzer": (
        "SERIES_AMPLITUDE", "SERIES_NOISE", "SERIES_PERIOD", "SERIES_TREND", "autocorr_peaks",
        "detrend", "dominant_period", "make_series", "smooth",
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
        >>> import src.fft_analyzer.applications as apps
        >>> apps.EXAMPLE_FS, apps.EXAMPLE_TONES
        (16000, (440.0, 3000.0))
        >>> apps.load_mp3
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.fft_analyzer.applications' has no attribute 'load_mp3'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
