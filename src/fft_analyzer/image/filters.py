"""Spatial image filters computed through the 2-D FFT.

The method
----------
A linear, shift-invariant filter (blur, edge detector, Laplacian) is a 2-D
convolution of the image with a small kernel. Evaluated directly that is
K = kh * kw multiply-adds per pixel, O(H W K) in total. The convolution
theorem turns it into pointwise multiplication: the 2-D DFT of a circular
convolution is the product of the two 2-D DFTs. Each filter here therefore
moves the problem where it is easy (the forward :func:`~src.fft_analyzer.core.fft.fft2`
of the image and of the zero-padded kernel), does the easy thing (one
multiplication per frequency bin), and moves back (the inverse
:func:`~src.fft_analyzer.core.fft.ifft2`), for O(H W log(H W)) whatever the
kernel size.

The convolution convention
--------------------------
The DFT sees the image as periodic, so the result is a *circular*
convolution: pixels past the right edge wrap round to the left edge. The
kernel is flipped (true convolution, not correlation) and placed with its
centre, index ``(kh // 2, kw // 2)``, at index (0, 0) of an H x W array of
zeros by :func:`numpy.roll`. With that placement the output is exactly
``scipy.ndimage.convolve(img, kernel, mode="wrap")``, including scipy's
origin convention for even-sized kernels (the centre is again index
``k // 2``). A kernel larger than the image is folded onto it modulo the
image size first, which is what periodic wrapping means. scipy is used only
in the tests as the independent cross-check (spec 1.1); this module calls no
``np.fft``, ``scipy.fft`` or ``scipy.ndimage`` routine.

For a real image and a real kernel the inverse transform is real up to
rounding (imaginary parts around 1e-13 of the pixel scale), so every filter
returns ``.real``.

Images
------
Images are 2-D ``float64`` arrays with values in 0-255. A colour array of
shape (H, W, 3) is converted to luminance first by :func:`to_grayscale`
(``L = 0.299 R + 0.587 G + 0.114 B``); a fourth (alpha) channel is ignored.

Examples:
    >>> img = np.zeros((8, 8)); img[:, 4:] = 255.0
    >>> np.allclose(fft_convolve2d(img, np.ones((1, 1))), img)
    True
    >>> mag = sobel(img)
    >>> float(round(mag[3, 3], 6)), float(round(mag[3, 1], 6))
    (1020.0, 0.0)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numpy as np

from src.fft_analyzer.core.fft import fft2, ifft2

__all__ = [
    "fft_convolve2d",
    "fft_denoise",
    "gaussian_blur",
    "gaussian_kernel",
    "laplacian",
    "sobel",
    "to_grayscale",
]

#: ITU-R BT.601 luminance weights for R, G and B.
_LUMA = np.array([0.299, 0.587, 0.114])

#: Sobel kernel for the horizontal derivative (a convolution kernel, so the
#: positive column is on the left: a rise from left to right gives gx > 0).
_SOBEL_X = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])

#: Sobel kernel for the vertical derivative (the transpose of ``_SOBEL_X``).
_SOBEL_Y = _SOBEL_X.T.copy()

#: The 5-point discrete Laplacian.
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


# ------------------------------------------------------------------ helpers


def to_grayscale(img) -> np.ndarray:
    """Return a 2-D ``float64`` luminance image from a grayscale or colour array.

    A 2-D array is already grayscale and is returned as a ``float64`` copy. A
    colour array of shape (H, W, 3) is reduced with the ITU-R BT.601 weights
    ``L = 0.299 R + 0.587 G + 0.114 B``, the convention every image routine
    in this package uses. An (H, W, 4) array is treated as RGBA and its alpha
    channel is ignored.

    Args:
        img: A non-empty array of shape (H, W), (H, W, 3) or (H, W, 4).

    Returns:
        A new ``float64`` array of shape (H, W).

    Raises:
        ValueError: If ``img`` is empty or has any other shape.

    Time Complexity:
        O(H W).

    Space Complexity:
        O(H W) for the result.

    Examples:
        >>> to_grayscale([[1, 2], [3, 4]])
        array([[1., 2.],
               [3., 4.]])
        >>> rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        >>> np.round(to_grayscale(rgb), 3)
        array([[ 76.245, 149.685,  29.07 ]])
    """
    a = np.asarray(img, dtype=np.float64)
    if a.size == 0:
        raise ValueError("image must be non-empty")
    if a.ndim == 2:
        return a.copy()
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return a[:, :, 0] * _LUMA[0] + a[:, :, 1] * _LUMA[1] + a[:, :, 2] * _LUMA[2]
    raise ValueError(
        f"image must have shape (H, W), (H, W, 3) or (H, W, 4), got {a.shape}"
    )


def _kernel_array(kernel) -> np.ndarray:
    """Validate a convolution kernel and return it as a 2-D ``float64`` array.

    Args:
        kernel: A non-empty 2-D array-like.

    Returns:
        The kernel as a ``float64`` array.

    Raises:
        ValueError: If the kernel is not 2-D or is empty.

    Time Complexity:
        O(kh kw).

    Space Complexity:
        O(kh kw).

    Examples:
        >>> _kernel_array([[1, 2]]).dtype
        dtype('float64')
    """
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.size == 0:
        raise ValueError(f"kernel must be a non-empty 2-D array, got shape {k.shape}")
    return k


def _centred_kernel(kernel: np.ndarray, shape) -> np.ndarray:
    """Place a kernel in an array of the image's shape with its centre at (0, 0).

    The kernel goes into the top-left corner of an H x W array of zeros
    (folded modulo H and W first if it is larger than the image, so every
    weight lands on its wrapped position), then :func:`numpy.roll` shifts it
    up by ``kh // 2`` and left by ``kw // 2``. The kernel's centre is then at
    index (0, 0) and its other taps wrap round to the far edges, which is the
    placement that makes the FFT product equal scipy's ``mode="wrap"``
    convolution.

    Args:
        kernel: A 2-D ``float64`` kernel of shape (kh, kw).
        shape: The image shape ``(H, W)``.

    Returns:
        A ``float64`` array of shape (H, W).

    Time Complexity:
        O(H W + kh kw).

    Space Complexity:
        O(H W + kh kw).

    Examples:
        >>> _centred_kernel(np.array([[1.0, 2.0, 3.0]]), (1, 5))
        array([[2., 3., 0., 0., 1.]])
        >>> _centred_kernel(np.array([[1.0, 2.0, 3.0]]), (1, 2))
        array([[2., 4.]])
    """
    H, W = shape
    kh, kw = kernel.shape
    tiles_r = -(-kh // H)  # ceil(kh / H)
    tiles_c = -(-kw // W)
    padded = np.zeros((tiles_r * H, tiles_c * W))
    padded[:kh, :kw] = kernel
    folded = padded.reshape(tiles_r, H, tiles_c, W).sum(axis=(0, 2))
    return np.roll(folded, (-(kh // 2), -(kw // 2)), axis=(0, 1))


# ------------------------------------------------------------ convolution


def fft_convolve2d(img, kernel) -> np.ndarray:
    """Circularly convolve an image with a kernel through the 2-D FFT.

    The kernel is centred at (0, 0) of an image-sized array (see the module
    docstring), both arrays are transformed by
    :func:`~src.fft_analyzer.core.fft.fft2`, multiplied bin by bin and
    transformed back by :func:`~src.fft_analyzer.core.fft.ifft2`. The result
    equals ``scipy.ndimage.convolve(img, kernel, mode="wrap")`` to rounding,
    for odd and even kernel sizes alike. The imaginary part of the inverse
    transform is rounding noise for real inputs, so ``.real`` is returned.

    Args:
        img: The image, shape (H, W) or a colour (H, W, 3) array (converted by
            :func:`to_grayscale`).
        kernel: A non-empty 2-D convolution kernel of shape (kh, kw). It may
            be larger than the image; it is then folded onto it periodically.

    Returns:
        A ``float64`` array of shape (H, W).

    Raises:
        ValueError: If the image or the kernel is empty or has the wrong
            number of dimensions.

    Time Complexity:
        O(H W log(H W)) when H and W are powers of two (the radix-2 FFT);
        :func:`~src.fft_analyzer.core.fft.fft2` falls back to the direct DFT
        along a side that is not, costing O(H W (H + W)). Independent of the
        kernel size, against O(H W kh kw) for direct convolution.

    Space Complexity:
        O(H W) for the padded kernel and the two spectra.

    Examples:
        >>> img = np.arange(16.0).reshape(4, 4)
        >>> np.allclose(fft_convolve2d(img, [[0, 0, 0], [0, 1, 0], [0, 0, 0]]), img)
        True
        >>> shift = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]  # output[i, j] = img[i, j + 1]
        >>> np.allclose(fft_convolve2d(img, shift), np.roll(img, -1, axis=1))
        True
        >>> np.round(fft_convolve2d([[1.0, 2.0, 3.0, 4.0]], [[1.0, 1.0]]), 10) + 0
        array([[3., 5., 7., 5.]])
    """
    image = to_grayscale(img)
    k = _kernel_array(kernel)
    spectrum = fft2(image) * fft2(_centred_kernel(k, image.shape))
    return ifft2(spectrum).real


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Return a normalized 2-D Gaussian kernel.

    The kernel samples ``exp(-(x^2 + y^2) / (2 sigma^2))`` on the integer grid
    ``-r..r`` in each direction, with radius ``r = int(truncate * sigma + 0.5)``
    (scipy's ``gaussian_filter`` rule), and divides by the sum so the weights
    add up to 1 and a blur keeps the image's mean brightness. It is the outer
    product of two 1-D Gaussians, so it is separable and symmetric.

    Args:
        sigma: Standard deviation in pixels, > 0.
        truncate: Number of standard deviations kept on each side, > 0.

    Returns:
        A ``float64`` array of shape (2r + 1, 2r + 1) summing to 1.

    Raises:
        ValueError: If ``sigma`` or ``truncate`` is not positive.

    Time Complexity:
        O(r^2) = O((truncate * sigma)^2).

    Space Complexity:
        O(r^2).

    Examples:
        >>> k = gaussian_kernel(1.0, truncate=2.0)
        >>> k.shape, float(round(k.sum(), 12))
        ((5, 5), 1.0)
        >>> gaussian_kernel(1.0).shape
        (9, 9)
        >>> bool(np.allclose(k, k.T)) and bool(k[2, 2] == k.max())
        True
    """
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not truncate > 0:
        raise ValueError(f"truncate must be positive, got {truncate}")
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k = g[:, None] * g[None, :]
    return k / k.sum()


def gaussian_blur(img, sigma: float) -> np.ndarray:
    """Blur an image with a Gaussian kernel, convolved through the FFT.

    The FFT of a Gaussian is a Gaussian, so blurring multiplies every
    frequency by a factor that falls off smoothly with distance from DC: fine
    detail and pixel noise are damped and the mean is kept (the kernel sums
    to 1). Edges wrap round, as in every filter here.

    Args:
        img: The image, shape (H, W) or colour (H, W, 3).
        sigma: Standard deviation of the Gaussian in pixels, > 0.

    Returns:
        The blurred ``float64`` image of shape (H, W).

    Raises:
        ValueError: If ``sigma`` is not positive or the image is invalid.

    Time Complexity:
        O(H W log(H W)) for power-of-two sides, independent of ``sigma``.

    Space Complexity:
        O(H W).

    Examples:
        >>> rng = np.random.default_rng(42)
        >>> noise = rng.normal(128.0, 20.0, size=(32, 32))
        >>> blurred = gaussian_blur(noise, 1.5)
        >>> bool(blurred.var() < noise.var())
        True
        >>> bool(np.isclose(blurred.mean(), noise.mean()))
        True
    """
    return fft_convolve2d(img, gaussian_kernel(sigma))


def sobel(img) -> np.ndarray:
    """Return the Sobel gradient magnitude ``sqrt(gx^2 + gy^2)``.

    ``gx`` is the convolution with the horizontal Sobel kernel
    ``[[1, 0, -1], [2, 0, -2], [1, 0, -1]]`` (a central difference across the
    columns, smoothed 1-2-1 down the rows) and ``gy`` with its transpose. A
    constant image gives 0 everywhere, and a vertical step of height h gives
    ``4 h`` on the two columns either side of the step. Because the
    convolution is circular, the image's left and right (and top and bottom)
    edges are neighbours, so a jump between them shows up as an edge too.

    Args:
        img: The image, shape (H, W) or colour (H, W, 3).

    Returns:
        The non-negative ``float64`` gradient magnitude, shape (H, W).

    Raises:
        ValueError: If the image is invalid.

    Time Complexity:
        O(H W log(H W)) for power-of-two sides (two FFT convolutions).

    Space Complexity:
        O(H W).

    Examples:
        >>> float(np.abs(sobel(np.full((8, 8), 7.0))).max()) < 1e-9
        True
        >>> step = np.zeros((8, 8)); step[:, 4:] = 10.0
        >>> np.round(sobel(step)[0], 6) + 0
        array([40.,  0.,  0., 40., 40.,  0.,  0., 40.])
    """
    image = to_grayscale(img)
    gx = fft_convolve2d(image, _SOBEL_X)
    gy = fft_convolve2d(image, _SOBEL_Y)
    return np.sqrt(gx * gx + gy * gy)


def laplacian(img) -> np.ndarray:
    """Return the discrete Laplacian, convolved through the FFT.

    The 5-point kernel ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]`` approximates
    ``d^2/dx^2 + d^2/dy^2``: it is 0 on constant and linear regions and
    changes sign across an edge, so its zero crossings mark edges. The kernel
    sums to 0, so the DC bin of the result is 0 (the output has mean 0).

    Args:
        img: The image, shape (H, W) or colour (H, W, 3).

    Returns:
        A ``float64`` array of shape (H, W).

    Raises:
        ValueError: If the image is invalid.

    Time Complexity:
        O(H W log(H W)) for power-of-two sides.

    Space Complexity:
        O(H W).

    Examples:
        >>> impulse = np.zeros((4, 4)); impulse[1, 1] = 1.0
        >>> np.round(laplacian(impulse), 10) + 0
        array([[ 0.,  1.,  0.,  0.],
               [ 1., -4.,  1.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]])
    """
    return fft_convolve2d(img, _LAPLACIAN)


def fft_denoise(img, keep_radius: float) -> np.ndarray:
    """Low-pass an image by keeping only a disc of its centred 2-D spectrum.

    The 2-D spectrum is rolled so that the DC bin sits at (H // 2, W // 2)
    (numpy's ``fftshift`` convention, written with :func:`numpy.roll`). Every
    bin farther than ``keep_radius`` bins from DC is set to 0, the spectrum is
    rolled back and inverted. The disc is symmetric about DC, so conjugate
    pairs are kept or dropped together and the result is real; noise, which
    spreads over all frequencies, loses everything outside the disc, while a
    smooth scene keeps most of its energy. A brick-wall mask rings a little
    near sharp edges.

    Args:
        img: The image, shape (H, W) or colour (H, W, 3).
        keep_radius: Radius of the kept disc in frequency bins, >= 0. 0 keeps
            only DC (the mean); a radius beyond the corners keeps everything.

    Returns:
        The filtered ``float64`` image of shape (H, W).

    Raises:
        ValueError: If ``keep_radius`` is negative or the image is invalid.

    Time Complexity:
        O(H W log(H W)) for power-of-two sides.

    Space Complexity:
        O(H W).

    Examples:
        >>> img = np.arange(64.0).reshape(8, 8)
        >>> np.allclose(fft_denoise(img, 0), img.mean())
        True
        >>> np.allclose(fft_denoise(img, 100), img)
        True
    """
    if not keep_radius >= 0:
        raise ValueError(f"keep_radius must be >= 0, got {keep_radius}")
    image = to_grayscale(img)
    H, W = image.shape
    centred = np.roll(fft2(image), (H // 2, W // 2), axis=(0, 1))
    rows = np.arange(H, dtype=np.float64)[:, None] - H // 2
    cols = np.arange(W, dtype=np.float64)[None, :] - W // 2
    mask = rows * rows + cols * cols <= float(keep_radius) ** 2
    spectrum = np.roll(centred * mask, (-(H // 2), -(W // 2)), axis=(0, 1))
    return ifft2(spectrum).real
