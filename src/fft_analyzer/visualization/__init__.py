"""Visualization: spectral and stability figures, and two GIF animations (Week 12).

Two modules. They draw both ends of the round trip every algorithm here
makes (move the problem where it is easy, do the easy thing there, and move
back): the problem as the transform sees it, and what comes back.

* :mod:`~src.fft_analyzer.visualization.plots` - :func:`plot_spectrum`,
  :func:`plot_spectrogram`, :func:`plot_fft2_heatmap` (centred log
  magnitude), :func:`plot_condition_numbers`, :func:`plot_compression_tradeoff`
  and :func:`plot_image_grid`. Each uses the house style, labels its axes
  with units, never tells series apart by colour alone, and returns its
  Figure on an Agg canvas.
* :mod:`~src.fft_analyzer.visualization.animations` -
  :func:`animate_fft_stages` (the butterfly stages of a 16-point FFT) and
  :func:`animate_filtering` (a low-pass cutoff sweeping down), each a GIF of
  at most :data:`MAX_FRAMES` frames written with ``PillowWriter``.

The public names below load from their module on first use, so importing this
sub-package does not import matplotlib.

Examples:
    >>> from src.fft_analyzer.visualization import MAX_FRAMES, plot_spectrum
    >>> MAX_FRAMES
    40
    >>> fig = plot_spectrum([0.0, 250.0, 500.0], [0.0, 2.0, 0.0])
    >>> fig.axes[0].get_xlabel()
    'Frequency (Hz)'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each module.
_EXPORTS = {
    "plots": (
        "plot_compression_tradeoff", "plot_condition_numbers", "plot_fft2_heatmap",
        "plot_image_grid", "plot_spectrogram", "plot_spectrum",
    ),
    "animations": ("FILTER_FRAMES", "MAX_FRAMES", "animate_fft_stages", "animate_filtering"),
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
        >>> import src.fft_analyzer.visualization as viz
        >>> viz.animate_filtering.__module__
        'src.fft_analyzer.visualization.animations'
        >>> viz.show
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.fft_analyzer.visualization' has no attribute 'show'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
