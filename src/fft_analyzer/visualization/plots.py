"""Static figures for the FFT analyzer: spectra, spectrograms and the stability study.

What the figures show
---------------------
Every transform in this package moves the problem where it is easy (the
forward FFT or DCT), does the easy thing (multiply or quantize bin by bin),
and moves back (the inverse transform). The figures here show the two ends
of that round trip side by side with the middle of it:

* :func:`plot_spectrum`, :func:`plot_spectrogram` and :func:`plot_fft2_heatmap`
  show the moved problem, the signal or image as the FFT sees it;
* :func:`plot_image_grid` and :func:`plot_compression_tradeoff` show what
  comes back after the inverse transform, and what the trip cost;
* :func:`plot_condition_numbers` compares the hand-written condition numbers
  with the library reference, the measure of how much a linear solve can
  amplify rounding error.

Conventions
-----------
* Every function calls :func:`src.utils.visualization.apply_house_style`,
  labels both axes with their units, and returns the
  :class:`~matplotlib.figure.Figure` it drew on.
* With ``ax=None`` the figure is a :class:`~matplotlib.figure.Figure` on an
  explicit Agg canvas, not a pyplot figure, so nothing opens a window and
  nothing is left in pyplot's figure registry. With an ``ax`` the plot is
  drawn on that axes and its figure is returned.
* ``save_path`` writes a 256-colour palette PNG of at most 100 KB, at 200 dpi
  or the highest lower resolution that fits (parent folders are created), through
  :func:`src.utils.visualization.save_png_under_limit`.
* No series is identified by colour alone: every line has its own marker or
  line style as well as a legend entry, and heatmaps carry a labelled
  colour bar.
* The centred 2-D spectrum is shifted with :func:`numpy.roll`, which is
  ``fftshift`` written out; nothing here calls ``np.fft``.

Examples:
    >>> import os, tempfile
    >>> freqs = np.array([0.0, 125.0, 250.0, 375.0, 500.0])
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = os.path.join(folder, "spectrum.png")
    ...     fig = plot_spectrum(freqs, [0.0, 4.0, 0.0, 0.0, 0.0], save_path=path)
    ...     os.path.getsize(path) > 0
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.fft_analyzer.core.fft import fft2
from src.fft_analyzer.image.filters import to_grayscale
from src.utils.visualization import apply_house_style, save_png_under_limit

__all__ = [
    "plot_spectrum",
    "plot_spectrogram",
    "plot_fft2_heatmap",
    "plot_condition_numbers",
    "plot_compression_tradeoff",
    "plot_image_grid",
]

INK = "#1f1f1f"
LINE = "#2471a3"
REFERENCE = "#7f8c8d"
ACCENT = "#c0392b"
#: Lowest level drawn on the spectrogram, in dB below its peak.
SPECTROGRAM_FLOOR_DB: float = -80.0
SAVE_DPI: int = 200


# ----------------------------------------------------------------- helpers


def _new_figure(ax: Optional[Axes], figsize: Tuple[float, float]) -> Tuple[Figure, Axes]:
    """Return ``(figure, axes)``: the caller's axes, or a fresh Agg figure.

    Args:
        ax: An existing axes to draw on, or ``None``.
        figsize: The size in inches of a new figure.

    Returns:
        The figure and the axes to draw on.

    Examples:
        >>> fig, ax = _new_figure(None, (4, 3))
        >>> ax.figure is fig, type(fig.canvas).__name__
        (True, 'FigureCanvasAgg')
    """
    if ax is not None:
        return ax.figure, ax
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot()


def _save_figure(fig: Figure, save_path: Optional[str]) -> None:
    """Write ``fig`` as a PNG to ``save_path``, creating parent folders.

    Args:
        fig: The figure to write.
        save_path: The PNG path, or ``None`` (or ``""``) to skip saving.

    Examples:
        >>> _save_figure(Figure(), None) is None
        True
    """
    if not save_path:
        return
    dpis = tuple(d for d in (SAVE_DPI, 170, 145, 125, 110, 100, 90) if d <= SAVE_DPI)
    save_png_under_limit(fig, save_path, dpis=dpis)


def _real_vector(values, name: str) -> np.ndarray:
    """Return ``values`` as a non-empty 1-D ``float64`` array.

    Args:
        values: A 1-D sequence of real numbers.
        name: The argument name used in error messages.

    Returns:
        A ``float64`` array.

    Raises:
        ValueError: If ``values`` is complex, empty or not 1-D.

    Examples:
        >>> _real_vector([1, 2], "x")
        array([1., 2.])
        >>> _real_vector([1j], "mags")
        Traceback (most recent call last):
        ...
        ValueError: mags must be real (pass np.abs of a complex spectrum)
    """
    a = np.asarray(values)
    if np.iscomplexobj(a):
        raise ValueError(f"{name} must be real (pass np.abs of a complex spectrum)")
    a = a.astype(np.float64)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence, got shape {a.shape}")
    return a


def _same_length(*named: Tuple[str, np.ndarray]) -> None:
    """Raise ``ValueError`` unless every named array has the same length.

    Args:
        named: ``(name, array)`` pairs.

    Raises:
        ValueError: If the lengths differ.

    Examples:
        >>> _same_length(("a", np.zeros(2)), ("b", np.zeros(3)))
        Traceback (most recent call last):
        ...
        ValueError: lengths differ: a has 2, b has 3
    """
    lengths = [(name, arr.shape[0]) for name, arr in named]
    if len({n for _, n in lengths}) > 1:
        detail = ", ".join(f"{name} has {n}" for name, n in lengths)
        raise ValueError(f"lengths differ: {detail}")


def _centred_log_magnitude(img) -> np.ndarray:
    """Return ``log10(1 + |F|)`` of the 2-D DFT, with frequency 0 in the centre.

    The shift by ``(H // 2, W // 2)`` with :func:`numpy.roll` is exactly
    ``fftshift``: after it, row ``r`` holds vertical frequency
    ``r - H // 2`` cycles per image height, and likewise for columns.

    Args:
        img: A grayscale (H, W) or colour (H, W, 3) image.

    Returns:
        A ``float64`` array of shape (H, W).

    Examples:
        >>> mag = _centred_log_magnitude(np.ones((4, 4)))
        >>> float(round(mag[2, 2], 6)), float(mag.sum() - mag[2, 2])
        (1.230449, 0.0)
    """
    gray = to_grayscale(img)
    spectrum = np.abs(fft2(gray))
    h, w = gray.shape
    return np.log10(1.0 + np.roll(spectrum, (h // 2, w // 2), axis=(0, 1)))


def _row_value(row: Dict[str, Any], key: str) -> float:
    """Return ``row[key]`` as a float, naming the key when it is missing.

    Args:
        row: One measurement dict.
        key: The field to read.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the key is missing.

    Examples:
        >>> _row_value({"psnr": 31}, "psnr")
        31.0
    """
    if key not in row:
        raise ValueError(f"each row needs a {key!r} entry, got keys {sorted(row)}")
    return float(row[key])


# ------------------------------------------------------------- 1-D signals


def plot_spectrum(
    freqs,
    mags,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot a one-sided magnitude spectrum against frequency in Hz.

    The spectrum is drawn as a line (with a marker on every bin when there
    are at most 64 bins, so a short transform reads as discrete bins). The
    magnitude axis is linear, so a tone of amplitude A over N samples shows
    as a peak of height ``A N / 2``, as the FFT computes it.

    Args:
        freqs: The bin frequencies in Hz, for example from
            :func:`~src.fft_analyzer.core.fft.rfft_freqs`.
        mags: The magnitudes ``|X[k]|`` of the same bins (real, same length).
        ax: An axes to draw on, or ``None`` for a new figure.
        save_path: A PNG path, or ``None`` to skip saving.
        title: The axes title; ``"Magnitude spectrum"`` when ``None``.

    Returns:
        The figure that holds the plot.

    Raises:
        ValueError: If either input is complex, empty or not 1-D, or their
            lengths differ.

    Time Complexity:
        O(K) for K bins, plus rendering.

    Space Complexity:
        O(K).

    Examples:
        >>> import os, tempfile
        >>> from src.fft_analyzer.core.fft import fft, rfft_freqs
        >>> from src.fft_analyzer.signal.generators import multi_tone
        >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.024)
        >>> freqs, mags = rfft_freqs(1024, 1000), np.abs(fft(x))[:513]
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "plots", "spectrum.png")
        ...     fig = plot_spectrum(freqs, mags, save_path=path, title="Two tones")
        ...     with open(path, "rb") as handle:
        ...         handle.read(8) == b"\\x89PNG\\r\\n\\x1a\\n"
        True
        >>> fig.axes[0].get_xlabel(), fig.axes[0].get_title()
        ('Frequency (Hz)', 'Two tones')
        >>> plot_spectrum([0, 1], [1, 2, 3])
        Traceback (most recent call last):
        ...
        ValueError: lengths differ: freqs has 2, mags has 3
    """
    f = _real_vector(freqs, "freqs")
    m = _real_vector(mags, "mags")
    _same_length(("freqs", f), ("mags", m))
    apply_house_style()
    fig, ax = _new_figure(ax, (8.0, 4.5))
    marker = "o" if f.shape[0] <= 64 else None
    ax.plot(f, m, color=LINE, linewidth=1.4, marker=marker, markersize=4,
            label="|X[k]|, one-sided magnitude")
    if f.shape[0] > 1:
        ax.set_xlim(float(f.min()), float(f.max()))
    if float(m.min()) >= 0.0:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude |X[k]| (arbitrary units)")
    ax.set_title(title if title is not None else "Magnitude spectrum")
    ax.legend(loc="upper right")
    _save_figure(fig, save_path)
    return fig


def plot_spectrogram(
    freqs,
    times,
    mag,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a short-time Fourier transform magnitude as a time-frequency heatmap.

    Each column is one frame of
    :func:`~src.fft_analyzer.signal.analysis.spectrogram`. Magnitudes are
    shown in dB relative to the largest one, ``20 log10(mag / max)``, with
    everything below ``SPECTROGRAM_FLOOR_DB`` drawn at the floor, so both a
    strong tone and its weak leakage stay readable. The colour bar is
    labelled with the dB scale; a chirp shows as a rising ridge.

    Args:
        freqs: Bin frequencies in Hz, length F.
        times: Frame centre times in seconds, length T.
        mag: Non-negative magnitudes of shape (F, T).
        ax: An axes to draw on, or ``None`` for a new figure.
        save_path: A PNG path, or ``None`` to skip saving.

    Returns:
        The figure that holds the plot.

    Raises:
        ValueError: If ``freqs`` or ``times`` is not a non-empty real 1-D
            sequence, or ``mag`` is complex or not of shape (F, T).

    Time Complexity:
        O(F T).

    Space Complexity:
        O(F T) for the dB array.

    Examples:
        >>> import os, tempfile
        >>> from src.fft_analyzer.signal.analysis import spectrogram
        >>> from src.fft_analyzer.signal.generators import chirp
        >>> f, t, m = spectrogram(chirp(100, 400, 1000, 1.0), 1000, seg=128, hop=64)
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "spectrogram.png")
        ...     fig = plot_spectrogram(f, t, m, save_path=path)
        ...     os.path.getsize(path) > 0
        True
        >>> [a.get_ylabel() for a in fig.axes]
        ['Frequency (Hz)', 'Magnitude (dB relative to the peak)']
        >>> plot_spectrogram(f, t, m.T)
        Traceback (most recent call last):
        ...
        ValueError: mag must have shape (65, 14), got (14, 65)
    """
    f = _real_vector(freqs, "freqs")
    t = _real_vector(times, "times")
    m = np.asarray(mag)
    if np.iscomplexobj(m):
        raise ValueError("mag must be real (pass np.abs of a complex spectrum)")
    m = m.astype(np.float64)
    if m.shape != (f.shape[0], t.shape[0]):
        raise ValueError(f"mag must have shape {(f.shape[0], t.shape[0])}, got {m.shape}")
    apply_house_style()
    fig, ax = _new_figure(ax, (8.0, 4.8))
    peak = float(np.max(np.abs(m)))
    if peak > 0.0:
        floor = peak * 10.0 ** (SPECTROGRAM_FLOOR_DB / 20.0)
        db = 20.0 * np.log10(np.maximum(np.abs(m), floor) / peak)
    else:
        db = np.full(m.shape, SPECTROGRAM_FLOOR_DB)
    mesh = ax.pcolormesh(t, f, db, shading="nearest", cmap="magma",
                         vmin=SPECTROGRAM_FLOOR_DB, vmax=0.0, rasterized=True)
    ax.grid(False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram: STFT magnitude, Hann window")
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label("Magnitude (dB relative to the peak)")
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------- 2-D images


def plot_fft2_heatmap(img, ax: Optional[Axes] = None, save_path: Optional[str] = None) -> Figure:
    """Plot the centred log-magnitude 2-D spectrum of an image.

    The image (converted to grayscale if it is colour) is transformed with
    :func:`~src.fft_analyzer.core.fft.fft2`, and ``log10(1 + |F|)`` is shown
    with frequency (0, 0) in the centre: the low frequencies that carry most
    of an image's energy sit in the middle, and edges show as streaks
    perpendicular to them. The log is needed because the DC term alone is
    usually several orders of magnitude larger than the rest.

    Args:
        img: A non-empty grayscale (H, W) or colour (H, W, 3) image.
        ax: An axes to draw on, or ``None`` for a new figure.
        save_path: A PNG path, or ``None`` to skip saving.

    Returns:
        The figure that holds the plot.

    Raises:
        ValueError: If ``img`` is empty or not of shape (H, W), (H, W, 3) or
            (H, W, 4).

    Time Complexity:
        O(H W log(H W)) for power-of-two sides (see
        :func:`~src.fft_analyzer.core.fft.fft2` for other sizes).

    Space Complexity:
        O(H W).

    Examples:
        >>> import os, tempfile
        >>> yy, xx = np.mgrid[0:32, 0:32]
        >>> stripes = 128 + 100 * np.cos(2 * np.pi * 4 * xx / 32)
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "fft2.png")
        ...     fig = plot_fft2_heatmap(stripes, save_path=path)
        ...     os.path.getsize(path) > 0
        True
        >>> fig.axes[0].get_xlabel()
        'Horizontal frequency (cycles per image width)'
    """
    mag = _centred_log_magnitude(img)
    h, w = mag.shape
    apply_house_style()
    fig, ax = _new_figure(ax, (6.4, 5.4))
    extent = (-(w // 2) - 0.5, w - w // 2 - 0.5, h - h // 2 - 0.5, -(h // 2) - 0.5)
    shown = ax.imshow(mag, cmap="viridis", extent=extent, interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("Horizontal frequency (cycles per image width)")
    ax.set_ylabel("Vertical frequency (cycles per image height)")
    ax.set_title("Centred 2-D spectrum, log magnitude")
    bar = fig.colorbar(shown, ax=ax)
    bar.set_label("log10(1 + |F(u, v)|) (dimensionless)")
    _save_figure(fig, save_path)
    return fig


def plot_image_grid(
    images: Sequence[Any],
    titles: Sequence[str],
    save_path: Optional[str] = None,
    ncols: int = 3,
) -> Figure:
    """Show several grayscale images in a grid, each with a title and colour bar.

    Colour images are converted with
    :func:`~src.fft_analyzer.image.filters.to_grayscale`. An image whose
    values lie in 0-255 is drawn on the fixed 0-255 gray scale, so panels can
    be compared directly; one that leaves that range (a Sobel gradient
    magnitude, a Laplacian) is drawn on its own minimum-to-maximum scale. The
    colour bar beside every panel shows which. Pixel axes are labelled on the
    left column and on the lowest panel of each column.

    Args:
        images: A non-empty sequence of images.
        titles: One title per image.
        save_path: A PNG path, or ``None`` to skip saving.
        ncols: The number of columns, at least 1.

    Returns:
        The figure that holds the grid.

    Raises:
        ValueError: If ``images`` is empty, the numbers of images and titles
            differ, ``ncols < 1``, or an image has an unsupported shape.

    Time Complexity:
        O(sum of H W over the images).

    Space Complexity:
        O(sum of H W) for the grayscale copies.

    Examples:
        >>> import os, tempfile
        >>> ramp = np.tile(np.linspace(0, 255, 16), (16, 1))
        >>> edges = np.abs(np.diff(ramp, axis=1, append=0)) * 40
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "grid.png")
        ...     fig = plot_image_grid([ramp, edges, ramp], ["ramp", "edges", "copy"],
        ...                           save_path=path, ncols=2)
        ...     os.path.getsize(path) > 0
        True
        >>> [a.get_title() for a in fig.axes if a.get_title()]
        ['ramp', 'edges', 'copy']
        >>> plot_image_grid([ramp], ["a", "b"])
        Traceback (most recent call last):
        ...
        ValueError: got 1 images but 2 titles
    """
    if len(images) == 0:
        raise ValueError("images must be non-empty")
    if len(images) != len(titles):
        raise ValueError(f"got {len(images)} images but {len(titles)} titles")
    if int(ncols) != ncols or ncols < 1:
        raise ValueError(f"ncols must be a positive integer, got {ncols}")
    grays = [to_grayscale(im) for im in images]
    count = len(grays)
    cols = min(int(ncols), count)
    rows = math.ceil(count / cols)
    apply_house_style()
    fig = Figure(figsize=(3.6 * cols + 0.4, 3.4 * rows + 0.3), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols, squeeze=False)
    for index in range(rows * cols):
        r, c = divmod(index, cols)
        ax = axes[r][c]
        if index >= count:
            ax.set_axis_off()
            continue
        gray = grays[index]
        low, high = float(gray.min()), float(gray.max())
        in_range = low >= 0.0 and high <= 255.0
        vmin, vmax = (0.0, 255.0) if in_range else (low, high if high > low else low + 1.0)
        shown = ax.imshow(gray, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.grid(False)
        ax.set_title(str(titles[index]), fontsize=13)
        ax.tick_params(labelsize=10)
        if c == 0:
            ax.set_ylabel("row (pixels)", fontsize=11)
        if index + cols >= count:
            ax.set_xlabel("column (pixels)", fontsize=11)
        bar = fig.colorbar(shown, ax=ax, shrink=0.85, pad=0.02)
        bar.ax.tick_params(labelsize=10)
        bar.set_label("gray level (0-255)" if in_range else "value (own range)", fontsize=11)
    _save_figure(fig, save_path)
    return fig


# ------------------------------------------------------- stability and cost


def plot_condition_numbers(
    ns,
    ours,
    reference,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot hand-computed condition numbers against a library reference by size.

    The y axis is logarithmic because the condition number of the Hilbert
    matrix grows about 30-fold per unit of n. The hand-written values are
    solid lines with hollow circles; the reference is a grey dashed line with
    crosses, so the two stay distinguishable where they overlap. A dotted
    horizontal line marks ``1 / eps``: a solve with a condition number near
    it has no correct digits left.

    Args:
        ns: Matrix sizes n (the x values).
        ours: The condition numbers computed in this package, one per n.
        reference: The library reference values (for example
            ``np.linalg.cond``), one per n.
        ax: An axes to draw on, or ``None`` for a new figure.
        save_path: A PNG path, or ``None`` to skip saving.

    Returns:
        The figure that holds the plot.

    Raises:
        ValueError: If an input is complex, empty or not 1-D, or the lengths
            differ.

    Time Complexity:
        O(len(ns)).

    Space Complexity:
        O(len(ns)).

    Examples:
        >>> import os, tempfile
        >>> from src.fft_analyzer.core.numerical import cond_estimate, hilbert
        >>> ns = [2, 3, 4, 5]
        >>> ours = [cond_estimate(hilbert(n), 1) for n in ns]
        >>> reference = [27.0, 748.0, 28375.0, 943656.0]  # exact 1-norm values
        >>> bool(np.allclose(ours, reference, rtol=1e-6))
        True
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "cond.png")
        ...     fig = plot_condition_numbers(ns, ours, reference, save_path=path)
        ...     os.path.getsize(path) > 0
        True
        >>> fig.axes[0].get_yscale(), len(fig.axes[0].get_legend().get_texts())
        ('log', 3)
    """
    n = _real_vector(ns, "ns")
    mine = _real_vector(ours, "ours")
    ref = _real_vector(reference, "reference")
    _same_length(("ns", n), ("ours", mine), ("reference", ref))
    apply_house_style()
    fig, ax = _new_figure(ax, (8.0, 5.0))
    ax.plot(n, mine, color=LINE, linestyle="-", linewidth=1.8, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=1.6, label="ours (by hand)", zorder=3)
    ax.plot(n, ref, color=REFERENCE, linestyle="--", linewidth=1.4, marker="x", markersize=7,
            label="library reference", zorder=4)
    ax.axhline(1.0 / float(np.finfo(float).eps), color=ACCENT, linestyle=":", linewidth=1.3,
               label="1 / machine epsilon")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n (n x n)")
    ax.set_ylabel("Condition number (dimensionless, log scale)")
    ax.set_title("Condition number by matrix size")
    ax.legend(loc="upper left")
    _save_figure(fig, save_path)
    return fig


def plot_compression_tradeoff(
    rows: Sequence[Dict[str, Any]],
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot PSNR against the fraction of non-zero coefficients, one point per quality.

    Each row is one run of :func:`~src.fft_analyzer.image.compression.compress`.
    The points are joined in order of quality and every point is labelled
    with its quality setting, so the curve reads as a dial: moving right
    keeps more coefficients (a larger file) and buys a higher PSNR. The
    non-zero ratio stands in for file size because only the transform and
    quantization stages of JPEG are implemented. A row whose PSNR is
    infinite (an exact reconstruction) cannot be placed on the axis and is
    left out.

    Args:
        rows: Dicts with ``"quality"``, ``"psnr"`` (dB) and
            ``"nonzero_ratio"`` (0-1).
        ax: An axes to draw on, or ``None`` for a new figure.
        save_path: A PNG path, or ``None`` to skip saving.

    Returns:
        The figure that holds the plot.

    Raises:
        ValueError: If ``rows`` is empty or a row lacks one of the keys.

    Time Complexity:
        O(R log R) for R rows (sorting by quality).

    Space Complexity:
        O(R).

    Examples:
        >>> import os, tempfile
        >>> rows = [{"quality": 10, "psnr": 27.5, "nonzero_ratio": 0.04},
        ...         {"quality": 50, "psnr": 33.1, "nonzero_ratio": 0.12},
        ...         {"quality": 90, "psnr": 40.2, "nonzero_ratio": 0.35}]
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "tradeoff.png")
        ...     fig = plot_compression_tradeoff(rows, save_path=path)
        ...     os.path.getsize(path) > 0
        True
        >>> [t.get_text() for t in fig.axes[0].texts]
        ['q=10', 'q=50', 'q=90']
        >>> plot_compression_tradeoff([{"quality": 10, "psnr": 30.0}])
        Traceback (most recent call last):
        ...
        ValueError: each row needs a 'nonzero_ratio' entry, got keys ['psnr', 'quality']
    """
    if len(rows) == 0:
        raise ValueError("rows must be non-empty")
    points: List[Tuple[float, float, float]] = []
    for row in rows:
        quality = _row_value(row, "quality")
        value = _row_value(row, "psnr")
        ratio = _row_value(row, "nonzero_ratio")
        if math.isfinite(value):
            points.append((quality, ratio, value))
    points.sort()
    apply_house_style()
    fig, ax = _new_figure(ax, (8.0, 5.0))
    if points:
        ratios = [p[1] for p in points]
        values = [p[2] for p in points]
        ax.plot(ratios, values, color=LINE, linestyle="-", linewidth=1.6, marker="o",
                markersize=7, label="JPEG-style DCT + quantization (one point per quality)")
        # PSNR rises with the ratio, so the line leaves each point towards the
        # upper right and arrives from the lower left: just right of the point
        # and a little below it is clear of the line.
        for quality, ratio, value in points:
            ax.annotate(f"q={quality:g}", (ratio, value), textcoords="offset points",
                        xytext=(8, -4), ha="left", va="center", fontsize=9, color=INK)
        ax.legend(loc="lower right")
        span = max(ratios) - min(ratios)
        unit = span if span > 0 else 0.1
        ax.set_xlim(max(0.0, min(ratios) - 0.06 * unit), max(ratios) + 0.16 * unit)
        low, high = min(values), max(values)
        rise = high - low if high > low else 1.0
        ax.set_ylim(low - 0.1 * rise, high + 0.08 * rise)
    ax.set_xlabel("Non-zero quantized coefficients (fraction of all)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Compression trade-off: quality vs. coefficients kept")
    _save_figure(fig, save_path)
    return fig
