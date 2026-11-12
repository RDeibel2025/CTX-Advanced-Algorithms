"""Block DCT compression: the transform and quantization stages of JPEG.

The method
----------
Natural images are smooth over a few pixels, so in the frequency domain an
8x8 block's energy sits in a handful of low-frequency coefficients. The
orthonormal DCT-II is written as the 8x8 matrix ``C`` with

    C[0, j] = sqrt(1/8),    C[k, j] = sqrt(2/8) cos((2j + 1) k pi / 16),

so the 2-D block transform is ``D = C @ B @ C.T`` and, because ``C`` is
orthonormal (``C C^T = I``), the inverse is ``B = C.T @ D @ C``. The DCT is
the real-valued cousin of the DFT for a block reflected about its edges, so
it avoids the false jump a periodic DFT sees between a block's left and right
sides. Every entry of ``C`` is computed here; ``scipy.fft`` appears only in
the tests as the cross-check (spec 1.1).

:func:`compress` moves the problem where it is easy (each 8x8 block to the
DCT domain, where the detail the eye barely sees is in separate, mostly small
coefficients), does the easy thing (divide every coefficient by its entry of
the quantization table and round, which sends most high-frequency
coefficients to exactly 0), and moves back (dequantize and invert the DCT).

Quantization
------------
:func:`quant_table` scales the standard JPEG luminance table (ITU T.81
Annex K) the IJG way: ``scale = 5000 // q`` for q < 50, otherwise
``200 - 2q``, and ``Q = floor((T * scale + 50) / 100)`` clipped to 1..255,
all in integer arithmetic as libjpeg does. Quality 50 is the table itself and
quality 100 is all ones.

What this is not
----------------
This is the transform and quantization stages of JPEG only: no chroma
subsampling, no zig-zag scan, no run-length or Huffman coding. The fraction
of non-zero quantized coefficients, ``nonzero_ratio``, stands in for the file
size, because the entropy coder spends its bits almost entirely on those.

Examples:
    >>> C = dct_matrix(8)
    >>> bool(np.allclose(C @ C.T, np.eye(8)))
    True
    >>> int(quant_table(50)[0, 0]), int(quant_table(10)[0, 0])
    (16, 80)
    >>> rec, ratio, quality_db = compress(np.full((8, 8), 128.0), 50)
    >>> float(ratio), quality_db
    (0.0, inf)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import operator

import numpy as np

from src.fft_analyzer.image.filters import to_grayscale

__all__ = [
    "compress",
    "dct2_block",
    "dct_matrix",
    "idct2_block",
    "psnr",
    "quant_table",
]

#: JPEG block side.
_BLOCK = 8

#: Peak pixel value for PSNR (8-bit images).
_PEAK = 255.0

#: Level shift that centres 0-255 pixels on 0 before the DCT.
_LEVEL_SHIFT = 128.0

#: The standard JPEG luminance quantization table (ITU T.81, Annex K.1),
#: which is ``quant_table(50)``.
_JPEG_LUMINANCE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int64,
)


# -------------------------------------------------------------- the DCT


def dct_matrix(n: int = 8) -> np.ndarray:
    """Return the n x n orthonormal DCT-II matrix ``C``.

    Row k samples the cosine ``cos((2j + 1) k pi / (2n))`` at the n pixel
    centres j, so ``C @ x`` gives the DCT-II coefficients of a length-n
    signal. Row 0 is scaled by ``sqrt(1/n)`` and the others by ``sqrt(2/n)``,
    which makes the rows orthonormal: ``C C^T = I`` and ``C^-1 = C^T``. This
    is ``scipy.fft.dct(np.eye(n), norm="ortho", axis=0)``.

    Args:
        n: Block size, a positive integer (8 for JPEG).

    Returns:
        A ``float64`` array of shape (n, n).

    Raises:
        ValueError: If ``n`` is not a positive integer.

    Time Complexity:
        O(n^2).

    Space Complexity:
        O(n^2).

    Examples:
        >>> C = dct_matrix(8)
        >>> float(round(C[0, 0] ** 2, 12))
        0.125
        >>> bool(np.allclose(C @ C.T, np.eye(8)))
        True
        >>> np.round(dct_matrix(2), 6)
        array([[ 0.707107,  0.707107],
               [ 0.707107, -0.707107]])
    """
    try:
        n = operator.index(n)
    except TypeError:
        raise ValueError(f"n must be a positive integer, got {n!r}") from None
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    k = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(n, dtype=np.float64)[None, :]
    C = math.sqrt(2.0 / n) * np.cos((2.0 * j + 1.0) * k * math.pi / (2.0 * n))
    C[0, :] = math.sqrt(1.0 / n)
    return C


def _square_block(block, name: str) -> np.ndarray:
    """Validate a square 2-D block and return it as ``float64``.

    Args:
        block: An n x n array-like, n >= 1.
        name: The argument name for the error message.

    Returns:
        The block as a ``float64`` array.

    Raises:
        ValueError: If the block is not a non-empty square 2-D array.

    Time Complexity:
        O(n^2).

    Space Complexity:
        O(n^2).

    Examples:
        >>> _square_block([[1, 2], [3, 4]], "B").shape
        (2, 2)
    """
    b = np.asarray(block, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != b.shape[1] or b.size == 0:
        raise ValueError(f"{name} must be a non-empty square 2-D array, got shape {b.shape}")
    return b


def dct2_block(B) -> np.ndarray:
    """Return the 2-D orthonormal DCT-II of a square block, ``C @ B @ C.T``.

    ``C @ B`` transforms every column and ``(...) @ C.T`` every row, so
    ``D[u, v]`` is the weight of the cosine pattern with u half-cycles down
    and v half-cycles across. ``D[0, 0]`` (DC) is ``n`` times the block mean
    (1024 for an 8x8 block of 128).

    Args:
        B: A square n x n block (8 x 8 in JPEG).

    Returns:
        The ``float64`` coefficient block, shape (n, n).

    Raises:
        ValueError: If ``B`` is not a non-empty square 2-D array.

    Time Complexity:
        O(n^3) for the two matrix products (512 multiply-adds twice for n = 8).

    Space Complexity:
        O(n^2).

    Examples:
        >>> D = dct2_block(np.full((8, 8), 128.0))
        >>> float(round(D[0, 0], 9)), float(np.abs(D.ravel()[1:]).max()) < 1e-9
        (1024.0, True)
        >>> np.round(dct2_block([[1.0, 1.0], [1.0, 1.0]]), 12) + 0
        array([[2., 0.],
               [0., 0.]])
    """
    b = _square_block(B, "B")
    C = dct_matrix(b.shape[0])
    return C @ b @ C.T


def idct2_block(D) -> np.ndarray:
    """Invert :func:`dct2_block`: ``B = C.T @ D @ C``.

    ``C`` is orthonormal, so its inverse is its transpose and no system has
    to be solved.

    Args:
        D: A square n x n coefficient block.

    Returns:
        The ``float64`` pixel block, shape (n, n).

    Raises:
        ValueError: If ``D`` is not a non-empty square 2-D array.

    Time Complexity:
        O(n^3).

    Space Complexity:
        O(n^2).

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> B = rng.uniform(0.0, 255.0, size=(8, 8))
        >>> bool(np.allclose(idct2_block(dct2_block(B)), B))
        True
        >>> D = np.zeros((8, 8)); D[0, 0] = 1024.0
        >>> bool(np.allclose(idct2_block(D), 128.0))
        True
    """
    d = _square_block(D, "D")
    C = dct_matrix(d.shape[0])
    return C.T @ d @ C


# --------------------------------------------------------- quantization


def quant_table(quality: int) -> np.ndarray:
    """Return the JPEG luminance quantization table scaled to ``quality``.

    Uses the IJG (libjpeg) rule: ``scale = 5000 // q`` for q < 50 and
    ``200 - 2q`` otherwise, then ``Q = floor((T * scale + 50) / 100)`` on the
    standard table T, clipped to 1..255 (baseline JPEG). Low quality means
    large divisors (coarse steps, many zeros); quality 100 gives all ones,
    which leaves only the rounding to integers.

    Args:
        quality: An integer from 1 to 100.

    Returns:
        An ``int64`` array of shape (8, 8).

    Raises:
        ValueError: If ``quality`` is not an integer in 1..100.

    Time Complexity:
        O(1) (64 entries).

    Space Complexity:
        O(1).

    Examples:
        >>> quant_table(50)[0].tolist()
        [16, 11, 10, 16, 24, 40, 51, 61]
        >>> bool((quant_table(100) == 1).all())
        True
        >>> int(quant_table(10)[0, 0]), int(quant_table(90)[0, 0])
        (80, 3)
    """
    try:
        q = operator.index(quality)
    except TypeError:
        raise ValueError(f"quality must be an integer in 1..100, got {quality!r}") from None
    if not 1 <= q <= 100:
        raise ValueError(f"quality must be an integer in 1..100, got {q}")
    scale = 5000 // q if q < 50 else 200 - 2 * q
    table = (_JPEG_LUMINANCE * scale + 50) // 100
    return np.clip(table, 1, 255)


# ------------------------------------------------------------ compression


def psnr(a, b) -> float:
    """Return the peak signal-to-noise ratio ``10 log10(255^2 / MSE)`` in dB.

    MSE is the mean squared difference over all elements. Identical arrays
    have MSE 0 and PSNR ``inf``. As a rough guide for 8-bit images, 30-40 dB
    looks close to the original and above 45 dB the difference is invisible.

    Args:
        a: The reference image (any shape, values on the 0-255 scale).
        b: The compared image, same shape as ``a``.

    Returns:
        The PSNR in decibels as a ``float`` (``inf`` if the arrays are equal).

    Raises:
        ValueError: If the shapes differ or the arrays are empty.

    Time Complexity:
        O(N) for N elements.

    Space Complexity:
        O(N) for the difference array.

    Examples:
        >>> psnr(np.zeros((4, 4)), np.zeros((4, 4)))
        inf
        >>> psnr(np.zeros((4, 4)), np.full((4, 4), 255.0))
        0.0
        >>> round(psnr(np.zeros((2, 2)), np.ones((2, 2))), 4)
        48.1308
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shapes differ: {x.shape} and {y.shape}")
    if x.size == 0:
        raise ValueError("arrays must be non-empty")
    diff = x - y
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(_PEAK * _PEAK / mse)


def compress(img, quality: int):
    """Compress an image with 8x8 block DCT and JPEG quantization, then decode it.

    Steps: convert colour to luminance (:func:`~src.fft_analyzer.image.filters.to_grayscale`);
    edge-pad the bottom and right to a multiple of 8; subtract 128; transform
    every 8x8 block with ``C @ B @ C.T`` (all blocks in one batched product);
    quantize with ``round(D / Q)`` using :func:`quant_table` (ties, after
    snapping D / Q to 9 decimals, round half to even); count the
    non-zero quantized coefficients; dequantize (``* Q``); invert with
    ``C.T @ D @ C``; add 128; crop the padding; clip to 0-255. These are the
    lossy stages of JPEG only (no zig-zag or Huffman coding), so
    ``nonzero_ratio`` stands in for the compressed size.

    Args:
        img: The image, shape (H, W) or colour (H, W, 3), values 0-255.
        quality: JPEG quality, an integer from 1 (smallest) to 100 (best).

    Returns:
        ``(reconstructed, nonzero_ratio, psnr)``: the decoded ``float64``
        image of shape (H, W) in 0-255, the fraction of quantized
        coefficients that are non-zero (counted over the padded image), and
        the PSNR in dB of the reconstruction against the (grayscale) input.

    Raises:
        ValueError: If ``quality`` is not an integer in 1..100 or the image is
            empty or of the wrong shape.

    Time Complexity:
        O(H W): each of the ceil(H/8) ceil(W/8) blocks costs a constant
        2 * 2 * 8^3 multiply-adds.

    Space Complexity:
        O(H W) for the padded image and its coefficients.

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> img = rng.uniform(0.0, 255.0, size=(20, 30))
        >>> rec, ratio, best = compress(img, 100)
        >>> rec.shape, bool(best > 45.0), bool(0.0 < ratio <= 1.0)
        ((20, 30), True, True)
        >>> bool(compress(img, 10)[2] < compress(img, 90)[2])
        True
        >>> bool(compress(img, 10)[1] < compress(img, 90)[1])
        True
    """
    Q = quant_table(quality).astype(np.float64)
    gray = to_grayscale(img)
    H, W = gray.shape
    pad_r = (-H) % _BLOCK
    pad_c = (-W) % _BLOCK
    shifted = np.pad(gray, ((0, pad_r), (0, pad_c)), mode="edge") - _LEVEL_SHIFT
    Hp, Wp = shifted.shape
    # (Hp, Wp) -> (block row, block column, 8, 8)
    blocks = shifted.reshape(Hp // _BLOCK, _BLOCK, Wp // _BLOCK, _BLOCK).transpose(0, 2, 1, 3)
    C = dct_matrix(_BLOCK)
    coeffs = C @ blocks @ C.T
    # D / Q is often an exact tie k + 0.5 in exact arithmetic (DC and the
    # (0, 4), (4, 0), (4, 4) terms of an integer block are multiples of 1/8),
    # and the last bit of a batched matrix product depends on the BLAS build.
    # Snapping to 9 decimals first makes every tie exact, so np.round
    # (round half to even) gives the same answer on every platform.
    quantized = np.round(np.round(coeffs / Q, 9))
    nonzero_ratio = float(np.count_nonzero(quantized)) / quantized.size
    decoded = C.T @ (quantized * Q) @ C
    reconstructed = decoded.transpose(0, 2, 1, 3).reshape(Hp, Wp)[:H, :W] + _LEVEL_SHIFT
    reconstructed = np.clip(reconstructed, 0.0, _PEAK)
    return reconstructed, nonzero_ratio, psnr(gray, reconstructed)
