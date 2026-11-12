"""Week 12 tests: FFT image filters and block-DCT compression (spec 9.6, 9.7).

The filters are cross-checked against ``scipy.ndimage.convolve(...,
mode="wrap")`` and the DCT against ``scipy.fft.dct``; both libraries are
allowed in tests only, as the independent reference (spec 1.1). The JPEG
quantization tables are cross-checked against libjpeg itself through Pillow,
which writes the IJG-scaled table into every JPEG it saves.

Hand values:

- Sobel on a vertical step of height h: the horizontal kernel is a central
  difference across the columns weighted 1-2-1 down the rows, so the two
  columns either side of the step see ``(1 + 2 + 1) * h = 4h`` and columns
  away from any edge see 0. The convolution is circular, so the jump from the
  last column back to the first is an edge as well.
- ``dct2_block`` of an 8x8 block of 128: row 0 of C is ``sqrt(1/8)``
  everywhere, so ``D[0, 0] = (1/8) * 64 * 128 = 1024``; every other row of C
  is a cosine summing to 0 over the 8 samples, so every other coefficient is
  0.
- ``quant_table(10)[0, 0]``: scale ``5000 // 10 = 500``, so
  ``floor((16 * 500 + 50) / 100) = floor(80.5) = 80``. ``quant_table(100)``:
  scale 0, so ``floor(50 / 100) = 0``, clipped up to 1.
- PSNR with every pixel off by 1: ``10 log10(255^2 / 1) = 20 log10(255)
  = 48.1308`` dB; every pixel off by 255 gives 0 dB.

No timings are asserted.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import scipy.fft
import scipy.ndimage
from PIL import Image

from src.fft_analyzer.image.compression import (
    compress,
    dct2_block,
    dct_matrix,
    idct2_block,
    psnr,
    quant_table,
)
from src.fft_analyzer.image.filters import (
    fft_convolve2d,
    fft_denoise,
    gaussian_blur,
    gaussian_kernel,
    laplacian,
    sobel,
    to_grayscale,
)
from src.utils.random_utils import derive_seed

MASTER_SEED = 42

#: The horizontal Sobel kernel, written out independently of the module.
SOBEL_3X3 = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])

#: The standard JPEG luminance table (ITU T.81, Annex K.1).
JPEG_LUMINANCE_50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)


def _rng(*labels) -> np.random.Generator:
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(MASTER_SEED, "test_image_processing", *labels))


def _random_image(shape=(64, 64), *labels) -> np.ndarray:
    """A seeded uniform random image with values in 0-255."""
    return _rng("image", shape, *labels).uniform(0.0, 255.0, size=shape)


def _synthetic_scene(seed_label: str = "scene") -> np.ndarray:
    """The 256 x 256 synthetic test scene of spec 6, built here independently.

    A left-to-right gradient, a bright disc, a checkerboard square of 8-pixel
    cells and Gaussian noise with sigma 10, clipped to 0-255.
    """
    rows, cols = np.mgrid[0:256, 0:256].astype(np.float64)
    img = 40.0 + 120.0 * cols / 255.0
    disc = (rows - 80.0) ** 2 + (cols - 90.0) ** 2 <= 40.0**2
    img[disc] = 230.0
    square = (rows >= 150) & (rows < 214) & (cols >= 140) & (cols < 204)
    checks = ((rows // 8 + cols // 8) % 2).astype(bool)
    img[square & checks] = 235.0
    img[square & ~checks] = 20.0
    img = img + _rng(seed_label).normal(0.0, 10.0, size=img.shape)
    return np.clip(img, 0.0, 255.0)


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    return _synthetic_scene()


# ===================================================================== 9.6


@pytest.mark.parametrize(
    "kernel_name",
    ["sobel_3x3", "gaussian_5x5", "random_4x4"],
)
def test_fft_convolve2d_equals_scipy_wrap(kernel_name):
    img = _random_image((64, 64), "convolve")
    if kernel_name == "sobel_3x3":
        kernel = SOBEL_3X3
    elif kernel_name == "gaussian_5x5":
        kernel = gaussian_kernel(1.0, truncate=2.0)
        assert kernel.shape == (5, 5)
    else:
        kernel = _rng("kernel", 4).standard_normal((4, 4))
    ours = fft_convolve2d(img, kernel)
    ref = scipy.ndimage.convolve(img, kernel, mode="wrap")
    assert ours.shape == img.shape
    assert ours.dtype == np.float64
    assert np.max(np.abs(ours - ref)) < 1e-9


@pytest.mark.parametrize("kshape", [(2, 3), (1, 6), (7, 7)])
def test_fft_convolve2d_other_kernel_shapes_match_scipy(kshape):
    # Even and odd sizes follow scipy's origin convention; a kernel larger
    # than a small image wraps periodically, as scipy's "wrap" mode does.
    img = _random_image((6, 5), "small")
    kernel = _rng("kernel", kshape).standard_normal(kshape)
    ref = scipy.ndimage.convolve(img, kernel, mode="wrap")
    assert np.max(np.abs(fft_convolve2d(img, kernel) - ref)) < 1e-9


def test_fft_convolve2d_rejects_bad_kernel():
    with pytest.raises(ValueError):
        fft_convolve2d(np.zeros((4, 4)), np.ones(3))
    with pytest.raises(ValueError):
        fft_convolve2d(np.zeros((4, 4)), np.ones((0, 3)))


def test_gaussian_kernel_is_normalized_and_symmetric():
    k = gaussian_kernel(1.5)
    radius = int(4.0 * 1.5 + 0.5)
    assert k.shape == (2 * radius + 1, 2 * radius + 1)
    assert abs(k.sum() - 1.0) < 1e-12
    assert np.allclose(k, k.T) and np.allclose(k, k[::-1, ::-1])
    assert k[radius, radius] == k.max()
    with pytest.raises(ValueError):
        gaussian_kernel(0.0)


def test_gaussian_blur_lowers_white_noise_variance():
    noise = _rng("white").normal(128.0, 20.0, size=(64, 64))
    blurred = gaussian_blur(noise, 1.5)
    assert blurred.var() < noise.var()
    # A Gaussian of sigma 1.5 averages about 4 pi sigma^2 ~ 28 pixels, so the
    # variance falls far below a quarter of the original.
    assert blurred.var() < 0.25 * noise.var()
    # The kernel sums to 1 and the convolution is circular: the mean is kept.
    assert abs(blurred.mean() - noise.mean()) < 1e-9


def test_gaussian_blur_matches_scipy_gaussian_filter_wrap():
    img = _random_image((32, 64), "gauss")
    ref = scipy.ndimage.gaussian_filter(img, 2.0, mode="wrap")
    assert np.max(np.abs(gaussian_blur(img, 2.0) - ref)) < 1e-9


def test_sobel_is_zero_on_a_constant_image():
    assert np.max(np.abs(sobel(np.full((64, 64), 137.0)))) < 1e-9


def test_sobel_peaks_at_the_vertical_step_edge():
    h = 255.0
    step = np.zeros((64, 64))
    step[:, 32:] = h  # the edge lies between columns 31 and 32
    mag = sobel(step)
    peak = mag.max()
    assert abs(peak - 4.0 * h) < 1e-9
    # The two columns either side of the step carry the peak in every row.
    assert np.allclose(mag[:, 31], peak, atol=1e-9)
    assert np.allclose(mag[:, 32], peak, atol=1e-9)
    # Away from the step (and from the wrap-around edge at columns 0 / 63,
    # which the circular convolution also sees) the gradient is 0.
    interior = np.r_[2:30, 34:62]
    assert np.max(mag[:, interior]) < 1e-9
    # Within the interior band the maximum sits at the step columns.
    band = mag[:, 2:62].mean(axis=0)
    assert set(np.flatnonzero(np.isclose(band, band.max())) + 2) == {31, 32}


def test_laplacian_matches_scipy_and_has_zero_mean():
    img = _random_image((64, 64), "laplacian")
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    ref = scipy.ndimage.convolve(img, kernel, mode="wrap")
    out = laplacian(img)
    assert np.max(np.abs(out - ref)) < 1e-9
    assert abs(out.mean()) < 1e-9


def test_fft_denoise_limits_and_noise_reduction():
    rows, cols = np.mgrid[0:64, 0:64].astype(np.float64)
    clean = 128.0 + 60.0 * np.sin(2 * np.pi * 2 * cols / 64) * np.cos(2 * np.pi * rows / 64)
    noisy = clean + _rng("denoise").normal(0.0, 15.0, size=clean.shape)
    # Radius 0 keeps only DC (the mean); a radius past the corners keeps all.
    assert np.allclose(fft_denoise(noisy, 0), noisy.mean())
    assert np.allclose(fft_denoise(noisy, 100), noisy)
    # The clean image lives within 3 bins of DC, so a radius-6 disc keeps it
    # and drops most of the noise.
    assert psnr(clean, fft_denoise(noisy, 6)) > psnr(clean, noisy) + 10.0
    with pytest.raises(ValueError):
        fft_denoise(noisy, -1)


def test_to_grayscale_uses_bt601_weights():
    rgb = np.zeros((2, 2, 3))
    rgb[0, 0] = [255, 0, 0]
    rgb[0, 1] = [0, 255, 0]
    rgb[1, 0] = [0, 0, 255]
    rgb[1, 1] = [100, 100, 100]
    gray = to_grayscale(rgb)
    expected = np.array([[0.299 * 255, 0.587 * 255], [0.114 * 255, 100.0]])
    assert gray.shape == (2, 2)
    assert np.allclose(gray, expected, atol=1e-12)
    # Colour input is converted before filtering.
    assert np.allclose(fft_convolve2d(rgb, np.ones((1, 1))), expected)
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        to_grayscale(np.zeros(5))


# ===================================================================== 9.7


def test_dct_matrix_is_orthonormal_and_matches_scipy():
    C = dct_matrix(8)
    assert C.shape == (8, 8)
    assert np.max(np.abs(C @ C.T - np.eye(8))) < 1e-12
    assert np.max(np.abs(C.T @ C - np.eye(8))) < 1e-12
    ref = scipy.fft.dct(np.eye(8), norm="ortho", axis=0)
    assert np.max(np.abs(C - ref)) < 1e-12
    assert dct_matrix().shape == (8, 8)
    with pytest.raises(ValueError):
        dct_matrix(0)


def test_dct2_block_of_constant_128():
    B = np.full((8, 8), 128.0)
    D = dct2_block(B)
    assert abs(D[0, 0] - 1024.0) < 1e-9
    rest = D.copy()
    rest[0, 0] = 0.0
    assert np.max(np.abs(rest)) < 1e-9
    # After the -128 level shift every coefficient is 0.
    assert np.max(np.abs(dct2_block(B - 128.0))) < 1e-9


def test_dct2_block_matches_scipy_dctn_and_round_trips():
    B = _random_image((8, 8), "block")
    D = dct2_block(B)
    assert np.max(np.abs(D - scipy.fft.dctn(B, norm="ortho"))) < 1e-9
    assert np.max(np.abs(idct2_block(D) - B)) < 1e-9
    with pytest.raises(ValueError):
        dct2_block(np.zeros((8, 4)))


def test_quant_table_anchors():
    q50 = quant_table(50)
    assert q50.shape == (8, 8)
    assert q50[0].tolist() == [16, 11, 10, 16, 24, 40, 51, 61]
    assert np.array_equal(q50, JPEG_LUMINANCE_50)
    assert np.array_equal(quant_table(100), np.ones((8, 8)))
    assert quant_table(10)[0, 0] == 80


@pytest.mark.parametrize("quality", [0, 101, 50.5])
def test_quant_table_rejects_bad_quality(quality):
    with pytest.raises(ValueError):
        quant_table(quality)


def test_quant_table_matches_libjpeg_for_every_quality():
    # Pillow's JPEG encoder is libjpeg, which scales the standard table with
    # the same IJG integer arithmetic and stores it in the file's DQT segment.
    pixels = np.zeros((8, 8), dtype=np.uint8)
    for quality in range(1, 101):
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, "JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as jpeg:
            stored = np.array(jpeg.quantization[0]).reshape(8, 8)
        assert np.array_equal(quant_table(quality), stored), quality


def test_compress_quality_100_is_above_45_db(scene):
    rec, ratio, quality_db = compress(scene, 100)
    assert rec.shape == scene.shape
    assert quality_db > 45.0
    assert abs(quality_db - psnr(scene, rec)) < 1e-12
    assert 0.0 < ratio <= 1.0
    assert rec.min() >= 0.0 and rec.max() <= 255.0


def test_compress_trades_quality_for_nonzeros(scene):
    results = [compress(scene, q) for q in (10, 50, 90)]
    psnrs = [r[2] for r in results]
    ratios = [r[1] for r in results]
    assert psnrs[0] < psnrs[1] < psnrs[2]
    assert ratios[0] < ratios[1] < ratios[2]


def test_compress_constant_and_unaligned_images():
    rec, ratio, quality_db = compress(np.full((16, 24), 128.0), 50)
    assert np.allclose(rec, 128.0)
    assert ratio == 0.0
    assert math.isinf(quality_db)
    img = _random_image((21, 30), "unaligned")
    rec, ratio, _ = compress(img, 75)
    assert rec.shape == (21, 30)
    assert 0.0 < ratio <= 1.0
    with pytest.raises(ValueError):
        compress(img, 0)


def test_psnr_basics():
    a = _random_image((16, 16), "psnr")
    assert psnr(a, a) == math.inf
    assert psnr(np.zeros((4, 4)), np.full((4, 4), 255.0)) == 0.0
    assert abs(psnr(np.zeros((4, 4)), np.ones((4, 4))) - 20.0 * math.log10(255.0)) < 1e-12
    b = a + _rng("psnr-noise").normal(0.0, 5.0, size=a.shape)
    assert psnr(a, b) == psnr(b, a)
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4)), np.zeros((4, 5)))
