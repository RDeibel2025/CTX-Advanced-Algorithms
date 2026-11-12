"""Week 12: fast transforms, stable computation, signal processing and image compression.

This package is the instructions' ``FFTAnalyzer/fft_analyzer/`` project, and
it sits under ``src/`` because the repository holds every week's project. Its
sub-package ``signal`` keeps the instructions' name although the standard
library has a module of the same name, so it is only ever imported
absolutely, as ``src.fft_analyzer.signal``, and ``import signal`` still gives
the standard library module.

The method
----------
Every algorithm here makes the same round trip: move the problem where it is
easy, do the easy thing there, and move back. Convolution becomes
multiplication bin by bin after an FFT, a linear system becomes two
triangular solves after an LU factorization, and an 8x8 image block becomes a
few large coefficients after a DCT; the inverse transform, the substitutions
or the inverse DCT bring the answer back. What the round trip costs is
measured too: its running time, and the rounding error it adds or amplifies.

Five sub-packages:

* :mod:`~src.fft_analyzer.core` - the O(N^2) DFT and the recursive and
  iterative radix-2 Cooley-Tukey FFTs, the inverse, convolution and the 2-D
  FFT; classical, blocked and Strassen matrix multiplication, LU with partial
  pivoting, Householder and Gram-Schmidt QR and the power method; norms,
  condition numbers, cancellation and Kahan summation.
* :mod:`~src.fft_analyzer.signal` - seeded signal generators (tones, a
  chirp, white and pink noise), window functions, brick-wall FFT filters, and
  spectral analysis (periodogram, Welch, spectrogram, peaks,
  autocorrelation).
* :mod:`~src.fft_analyzer.image` - spatial filters through the 2-D FFT and
  the transform and quantization stages of JPEG with an 8x8 DCT.
* :mod:`~src.fft_analyzer.applications` - audio filtering of WAV files, an
  image pipeline, and time-series analysis of a synthetic daily series.
* :mod:`~src.fft_analyzer.visualization` - spectra, spectrograms and the
  stability figures, and two GIF animations.

Every transform, factorization and filter is written by hand: no ``np.fft``,
``scipy.fft``, ``scipy.signal``, ``scipy.ndimage`` or numpy or scipy
linear-algebra solver is called inside this package. The tests use them only
as the independent cross-check.

The public names below load from their module on first use, so importing
this package imports none of the sub-packages until one of their names is
used.

Examples:
    >>> from src.fft_analyzer import circular_convolve, solve
    >>> circular_convolve([1, 2, 3], [0, 1, 0.5])
    array([4. , 2.5, 2.5])
    >>> solve([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], [5, -2, 9]).tolist()
    [1.0, 1.0, 2.0]
    >>> import signal
    >>> signal.SIGINT.name
    'SIGINT'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each module, keyed by its path inside this package.
_EXPORTS = {
    "core.fft": (
        "circular_convolve", "dft", "fft", "fft2", "fft_iterative", "fft_iterative_stages",
        "fft_recursive", "ifft", "ifft2", "linear_convolve", "rfft_freqs",
    ),
    "core.matrix": (
        "determinant", "lu_decompose", "lu_solve", "matmul_blocked", "matmul_classical",
        "power_method", "qr_gram_schmidt", "qr_householder", "solve", "strassen",
    ),
    "core.numerical": (
        "abs_error", "cancellation_examples", "cond2_power", "cond_estimate", "hilbert",
        "is_ill_conditioned", "kahan_sum", "machine_epsilon", "norm", "rel_error",
    ),
    "signal.generators": (
        "chirp", "multi_tone", "pink_noise", "sine", "time_axis", "white_noise",
    ),
    "signal.filters": ("blackman", "fft_filter", "hamming", "hann"),
    "signal.analysis": (
        "autocorrelation", "peak_frequencies", "psd", "spectrogram", "welch_psd",
    ),
    "image.filters": (
        "fft_convolve2d", "fft_denoise", "gaussian_blur", "gaussian_kernel", "laplacian",
        "sobel", "to_grayscale",
    ),
    "image.compression": (
        "compress", "dct2_block", "dct_matrix", "idct2_block", "psnr", "quant_table",
    ),
    "applications.audio_processor": (
        "EXAMPLE_AMPS", "EXAMPLE_DURATION", "EXAMPLE_FS", "EXAMPLE_NOISE", "EXAMPLE_TONES",
        "load_wav", "make_example_wav", "process", "save_wav",
    ),
    "applications.image_processor": (
        "BLUR_SIGMA", "DENOISE_RADIUS", "QUALITIES", "SCENE_SIZE", "load_image",
        "make_test_scene", "run_pipeline", "save_image",
    ),
    "applications.data_analyzer": (
        "SERIES_AMPLITUDE", "SERIES_NOISE", "SERIES_PERIOD", "SERIES_TREND", "autocorr_peaks",
        "detrend", "dominant_period", "make_series", "smooth",
    ),
    "visualization.plots": (
        "plot_compression_tradeoff", "plot_condition_numbers", "plot_fft2_heatmap",
        "plot_image_grid", "plot_spectrogram", "plot_spectrum",
    ),
    "visualization.animations": (
        "FILTER_FRAMES", "MAX_FRAMES", "animate_fft_stages", "animate_filtering",
    ),
}

#: Each public name mapped to the module that provides it.
_HOME = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_HOME)


def __getattr__(name: str) -> Any:
    """Load a public name from its module on first use.

    Python calls this only for a name the package does not already hold. The
    name is looked up in the module that defines it, not in its sub-package,
    so ``fft`` here is always the function from ``core/fft.py`` even though
    ``src.fft_analyzer.core.fft`` is the module of the same name.

    Args:
        name: The attribute being looked up.

    Returns:
        The object of that name from its module.

    Raises:
        AttributeError: If ``name`` is not a public name of this package.

    Time Complexity:
        O(1), plus the module import the first time.

    Space Complexity:
        O(1).

    Examples:
        >>> import src.fft_analyzer as package
        >>> package.fft_iterative.__name__
        'fft_iterative'
        >>> package.fft.__module__
        'src.fft_analyzer.core.fft'
        >>> package.bluestein_fft
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.fft_analyzer' has no attribute 'bluestein_fft'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
