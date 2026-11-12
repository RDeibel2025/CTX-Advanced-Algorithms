"""Image pipeline: PNG input and output, FFT filters and block-DCT compression.

The method
----------
:func:`load_image` reads any image pillow can open and returns a grayscale
``float64`` array in 0-255; colour is reduced to luminance with
``L = 0.299 R + 0.587 G + 0.114 B``
(:func:`~src.fft_analyzer.image.filters.to_grayscale`). :func:`save_image`
rounds, clips to 0-255 and writes an 8-bit PNG.

:func:`run_pipeline` applies four operations to one image, and each makes
the same round trip: move the problem where it is easy, do the easy thing
there, and move back.

* **Blur** (:func:`~src.fft_analyzer.image.filters.gaussian_blur`, sigma 2
  pixels), **Sobel edges** (:func:`~src.fft_analyzer.image.filters.sobel`)
  and **denoising** (:func:`~src.fft_analyzer.image.filters.fft_denoise`,
  keeping a disc of radius 32 bins around DC): the 2-D FFT turns a
  convolution into one multiplication per frequency bin, and the inverse
  FFT brings the filtered image back.
* **Compression** (:func:`~src.fft_analyzer.image.compression.compress`)
  at JPEG quality 10, 50 and 90: each 8x8 block goes to the DCT domain, its
  coefficients are divided by the quantization table and rounded (most
  high-frequency ones become 0), then they are multiplied back and
  inverted. The PSNR against the input and the fraction of non-zero
  quantized coefficients (a stand-in for file size, since there is no
  entropy coder) are recorded for each quality.

The Sobel magnitude is 4 x 255 = 1020 across a full 0-to-255 straight edge
and more at corners, far above 255, so it is divided by its maximum and
scaled to 0-255 before it is saved; the other outputs are saved as they
are.

The test scene
--------------
:func:`make_test_scene` writes ``test_scene.png``, a synthetic 256 x 256
grayscale scene: a left-to-right gradient from 30 to 200, a bright disc
(value 235, radius 40, centred at row 72, column 72), a checkerboard square
(rows and columns 144-223, cells of 10 pixels alternating 30 and 225) and
Gaussian noise with standard deviation 10, seeded through
:func:`src.utils.random_utils.derive_seed`. It has smooth regions, curved
and straight edges, fine texture and noise, so every filter has something to
act on. It is not a photograph.

Examples:
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     scene = make_test_scene(os.path.join(d, "test_scene.png"), seed=42)
    ...     result = run_pipeline(scene, os.path.join(d, "out"))
    ...     img = load_image(scene)
    >>> img.shape, img.dtype
    ((256, 256), dtype('float64'))
    >>> rows = result["compression"]
    >>> rows[90]["psnr"] > rows[50]["psnr"] > rows[10]["psnr"]
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

from src.fft_analyzer.image.compression import compress
from src.fft_analyzer.image.filters import fft_denoise, gaussian_blur, sobel, to_grayscale
from src.utils.random_utils import derive_seed

__all__ = [
    "BLUR_SIGMA",
    "DENOISE_RADIUS",
    "QUALITIES",
    "SCENE_SIZE",
    "load_image",
    "make_test_scene",
    "run_pipeline",
    "save_image",
]

#: Path-like values accepted by the file functions.
PathLike = Union[str, "os.PathLike[str]"]

#: Side of the square test scene in pixels.
SCENE_SIZE: int = 256

#: Standard deviation of the pipeline's Gaussian blur, in pixels.
BLUR_SIGMA: float = 2.0

#: Radius in frequency bins of the disc :func:`run_pipeline` keeps when denoising.
DENOISE_RADIUS: float = 32.0

#: JPEG qualities :func:`run_pipeline` compresses at.
QUALITIES = (10, 50, 90)

#: Standard deviation of the test scene's Gaussian noise (grey levels).
_SCENE_NOISE = 10.0


# ------------------------------------------------------------------ helpers


def _ensure_parent(path: PathLike) -> str:
    """Create the parent directory of ``path`` if needed and return the path as text.

    Args:
        path: A file path.

    Returns:
        ``os.fspath(path)``.

    Raises:
        OSError: If the directory cannot be created.

    Time Complexity:
        O(depth) directory checks.

    Space Complexity:
        O(1).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     target = _ensure_parent(os.path.join(d, "a", "b.png"))
        ...     os.path.isdir(os.path.join(d, "a"))
        True
    """
    text = os.fspath(path)
    parent = os.path.dirname(text)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return text


def _scene(seed: int) -> np.ndarray:
    """Build the test scene as a ``float64`` array in 0-255 (before rounding).

    Layers, in order: a left-to-right gradient ``30 + 170 c / 255`` (c the
    column), a disc of value 235 and radius 40 centred at (72, 72), a
    checkerboard square over rows and columns 144-223 with 10-pixel cells
    alternating 30 and 225, then Gaussian noise of standard deviation 10 from
    ``numpy.random.default_rng(derive_seed(seed, "test_scene", "noise"))``,
    and a final clip to 0-255.

    Args:
        seed: Master seed for the noise.

    Returns:
        A 256 x 256 ``float64`` array.

    Raises:
        TypeError: If ``seed`` is not an ``int``.

    Time Complexity:
        O(S^2) for side S = 256.

    Space Complexity:
        O(S^2).

    Examples:
        >>> s = _scene(42)
        >>> s.shape, bool(s.min() >= 0.0), bool(s.max() <= 255.0)
        ((256, 256), True, True)
        >>> bool(np.array_equal(_scene(42), _scene(42)))
        True
    """
    size = SCENE_SIZE
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    scene = np.broadcast_to(30.0 + 170.0 * cols / (size - 1), (size, size)).copy()
    disc = (rows - 72.0) ** 2 + (cols - 72.0) ** 2 <= 40.0 ** 2
    scene[disc] = 235.0
    lo, hi, cell = 144, 224, 10
    r = np.arange(lo, hi)[:, None] - lo
    c = np.arange(lo, hi)[None, :] - lo
    checker = ((r // cell) + (c // cell)) % 2 == 0
    scene[lo:hi, lo:hi] = np.where(checker, 225.0, 30.0)
    rng = np.random.default_rng(derive_seed(seed, "test_scene", "noise"))
    scene += rng.normal(0.0, _SCENE_NOISE, size=(size, size))
    return np.clip(scene, 0.0, 255.0)


# -------------------------------------------------------------- image files


def load_image(path: PathLike) -> np.ndarray:
    """Read an image file as a grayscale ``float64`` array in 0-255.

    The file is opened with pillow. An 8-bit grayscale image (mode ``"L"``)
    is used as it is; a 16-bit grayscale image (modes ``"I;16..."``) is
    scaled by 255 / 65535 into 0-255, because pillow's own conversion would
    clamp every value above 255; an RGB or RGBA image is reduced to luminance
    with ``L = 0.299 R + 0.587 G + 0.114 B`` (alpha ignored); any other mode
    (palette, 1-bit, grey plus alpha) is first converted to RGB by pillow.

    Args:
        path: Path to the image file.

    Returns:
        A 2-D ``float64`` array of shape (H, W) with values in 0-255.

    Raises:
        FileNotFoundError: If the file does not exist.
        PIL.UnidentifiedImageError: If pillow cannot read the file as an
            image.

    Time Complexity:
        O(H W).

    Space Complexity:
        O(H W).

    Examples:
        >>> import tempfile
        >>> rgb = np.zeros((1, 3, 3), dtype=np.uint8)
        >>> rgb[0, 0, 0], rgb[0, 1, 1], rgb[0, 2, 2] = 255, 255, 255
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = os.path.join(d, "rgb.png")
        ...     Image.fromarray(rgb).save(p)
        ...     np.round(load_image(p), 3)
        array([[ 76.245, 149.685,  29.07 ]])
        >>> deep = np.array([[0, 257, 65535]], dtype=np.uint16)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = os.path.join(d, "deep.png")
        ...     Image.fromarray(deep).save(p)
        ...     load_image(p)
        array([[  0.,   1., 255.]])
    """
    with Image.open(os.fspath(path)) as im:
        im.load()
        if im.mode == "L":
            return np.asarray(im, dtype=np.float64).copy()
        if im.mode.startswith("I;16"):
            return np.asarray(im, dtype=np.float64) * (255.0 / 65535.0)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        return to_grayscale(np.asarray(im, dtype=np.float64))


def save_image(path: PathLike, img: Any) -> str:
    """Write an image as an 8-bit PNG after rounding and clipping to 0-255.

    A 2-D array is saved as grayscale and an (H, W, 3) array as RGB. Values
    are rounded to the nearest integer and clipped to 0-255 before the cast
    to ``uint8``, so an out-of-range filter output saturates instead of
    wrapping round. The PNG is written with pillow's ``optimize=True``, and
    the parent directory is created if it is missing.

    Args:
        path: Destination path; the file is PNG whatever its extension.
        img: A finite real array of shape (H, W) or (H, W, 3).

    Returns:
        The path written, as a ``str``.

    Raises:
        ValueError: If ``img`` is empty, has another shape, or holds NaN or
            infinity.

    Time Complexity:
        O(H W) plus the PNG encoder.

    Space Complexity:
        O(H W).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = save_image(os.path.join(d, "g.png"), [[-5.0, 100.4], [100.6, 300.0]])
        ...     load_image(p)
        array([[  0., 100.],
               [101., 255.]])
    """
    a = np.asarray(img, dtype=np.float64)
    if a.size == 0 or not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 3)):
        raise ValueError(f"image must be a non-empty (H, W) or (H, W, 3) array, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("image must be finite (no NaN or infinity)")
    pixels = np.clip(np.round(a), 0.0, 255.0).astype(np.uint8)
    target = _ensure_parent(path)
    # A 2-D uint8 array becomes mode "L" and an (H, W, 3) one mode "RGB".
    Image.fromarray(pixels).save(target, format="PNG", optimize=True)
    return target


def make_test_scene(path: PathLike, seed: int = 42) -> str:
    """Write the synthetic 256 x 256 test scene as an 8-bit grayscale PNG.

    The scene is a gradient, a bright disc, a checkerboard square and
    Gaussian noise with standard deviation 10 (layout in the module
    docstring), seeded through
    ``derive_seed(seed, "test_scene", "noise")``, saved with
    :func:`save_image`.

    Args:
        path: Destination path (``src/fft_analyzer/examples/test_scene.png``
            for the shipped copy).
        seed: Master seed for the noise.

    Returns:
        The path written, as a ``str``.

    Raises:
        TypeError: If ``seed`` is not an ``int``.

    Time Complexity:
        O(S^2) for side S = 256, plus the PNG encoder.

    Space Complexity:
        O(S^2).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     a = load_image(make_test_scene(os.path.join(d, "a.png"), seed=42))
        ...     b = load_image(make_test_scene(os.path.join(d, "b.png"), seed=42))
        >>> a.shape, bool(np.array_equal(a, b))
        ((256, 256), True)
    """
    return save_image(path, _scene(seed))


# ----------------------------------------------------------------- pipeline


def run_pipeline(path: PathLike, out_dir: PathLike) -> Dict[Any, Any]:
    """Blur, edge-detect, denoise and compress an image, saving every output.

    The image from :func:`load_image` goes through
    :func:`~src.fft_analyzer.image.filters.gaussian_blur` (sigma
    :data:`BLUR_SIGMA`), :func:`~src.fft_analyzer.image.filters.sobel`
    (scaled so its maximum is 255), :func:`~src.fft_analyzer.image.filters.fft_denoise`
    (radius :data:`DENOISE_RADIUS`) and
    :func:`~src.fft_analyzer.image.compression.compress` at each quality in
    :data:`QUALITIES`. Outputs are saved in ``out_dir`` (created if missing)
    as ``<stem>_blur.png``, ``<stem>_sobel.png``, ``<stem>_denoise.png`` and
    ``<stem>_q<quality>.png``, where ``<stem>`` is the input file name
    without its extension. The PSNR reported is the one ``compress``
    computes, between the grayscale input and the reconstruction before it
    is rounded to 8 bits.

    Args:
        path: The input image file.
        out_dir: The directory for the outputs.

    Returns:
        A dict with keys ``"blur"``, ``"sobel"`` and ``"denoise"`` (output
        paths as ``str``) and ``"compression"``, a dict from each quality
        (10, 50, 90) to ``{"psnr": float dB, "nonzero_ratio": float, "path":
        str}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OSError: If ``out_dir`` cannot be created or written.

    Time Complexity:
        O(H W log(H W)) for power-of-two sides (the FFT filters) plus O(H W)
        per quality for the block DCT.

    Space Complexity:
        O(H W).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     scene = make_test_scene(os.path.join(d, "scene.png"))
        ...     result = run_pipeline(scene, os.path.join(d, "out"))
        ...     names = sorted(os.listdir(os.path.join(d, "out")))
        >>> names
        ['scene_blur.png', 'scene_denoise.png', 'scene_q10.png', 'scene_q50.png', 'scene_q90.png', 'scene_sobel.png']
        >>> ratios = [result["compression"][q]["nonzero_ratio"] for q in (10, 50, 90)]
        >>> ratios == sorted(ratios)
        True
    """
    img = load_image(path)
    folder = os.fspath(out_dir)
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(os.fspath(path)))[0]

    def out(name: str) -> str:
        return os.path.join(folder, f"{stem}_{name}.png")

    edges = sobel(img)
    peak = float(edges.max())
    if peak > 0.0:
        edges = edges * (255.0 / peak)
    result: Dict[Any, Any] = {
        "blur": save_image(out("blur"), gaussian_blur(img, BLUR_SIGMA)),
        "sobel": save_image(out("sobel"), edges),
        "denoise": save_image(out("denoise"), fft_denoise(img, DENOISE_RADIUS)),
    }
    rows: Dict[int, Dict[str, Any]] = {}
    for quality in QUALITIES:
        reconstructed, ratio, quality_db = compress(img, quality)
        rows[quality] = {
            "psnr": float(quality_db),
            "nonzero_ratio": float(ratio),
            "path": save_image(out(f"q{quality}"), reconstructed),
        }
    result["compression"] = rows
    return result
