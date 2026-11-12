"""Images: spatial filters through the 2-D FFT and block-DCT compression (Week 12).

Two modules, each a round trip: move the problem where it is easy, do the
easy thing there, and move back.

* :mod:`~src.fft_analyzer.image.filters` - :func:`fft_convolve2d` is a
  circular convolution computed as one multiplication per bin of the 2-D FFT,
  equal to ``scipy.ndimage.convolve(img, kernel, mode="wrap")``. On it rest
  :func:`gaussian_kernel` and :func:`gaussian_blur`, :func:`sobel` (gradient
  magnitude), :func:`laplacian` and :func:`fft_denoise` (a circular low-pass
  mask on the centred spectrum). :func:`to_grayscale` turns colour into
  luminance, ``L = 0.299 R + 0.587 G + 0.114 B``.
* :mod:`~src.fft_analyzer.image.compression` - the transform and quantization
  stages of JPEG: :func:`dct_matrix`, :func:`dct2_block` and
  :func:`idct2_block` (the orthonormal 8x8 DCT-II), :func:`quant_table` (the
  standard luminance table scaled the IJG way), :func:`compress` and
  :func:`psnr`. Each 8x8 block moves to the DCT domain, is quantized there,
  and comes back by the inverse DCT.

The public names below load from their module on first use.

Examples:
    >>> from src.fft_analyzer.image import quant_table
    >>> quant_table(50)[0].tolist()
    [16, 11, 10, 16, 24, 40, 51, 61]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each module.
_EXPORTS = {
    "filters": (
        "fft_convolve2d", "fft_denoise", "gaussian_blur", "gaussian_kernel", "laplacian",
        "sobel", "to_grayscale",
    ),
    "compression": (
        "compress", "dct2_block", "dct_matrix", "idct2_block", "psnr", "quant_table",
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
        >>> import src.fft_analyzer.image as image
        >>> image.sobel.__module__
        'src.fft_analyzer.image.filters'
        >>> image.huffman_encode
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.fft_analyzer.image' has no attribute 'huffman_encode'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
