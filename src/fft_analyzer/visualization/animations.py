"""Two GIF animations: the FFT butterfly stages, and a low-pass cutoff sweeping down.

What the animations show
------------------------
:func:`animate_fft_stages` steps through the iterative radix-2 FFT of a short
signal (N = 16). The first frame is the input after the bit-reversal
permutation; each later frame is the working array after one butterfly
stage, with dotted lines marking the groups that stage has combined (2, then
4, 8 and 16 entries), until the last frame is the DFT itself. Bars show the
magnitudes ``|a[i]|`` on a fixed scale, so the energy can be seen collecting
into fewer, larger entries as the groups merge.

:func:`animate_filtering` shows the round trip every FFT filter performs:
move the problem where it is easy (the forward FFT), do the easy thing (zero
every bin above the cutoff), and move back (the inverse FFT). The top panel
is the one-sided spectrum with the kept band solid and the removed band
dotted; the bottom panel is the original signal (dashed) and the low-pass
output (solid). The cutoff sweeps from the Nyquist frequency down, so each
tone vanishes from both panels in the frame where the cutoff passes it.

Format
------
Each animation is written frame by frame with
:class:`matplotlib.animation.PillowWriter` on a figure with an explicit Agg
canvas (no pyplot window, nothing left in pyplot's registry). Margins are
fixed, so the axes do not jump between frames. Both stay within the spec's
limits of 40 frames and 2 MB: the stages animation has log2 N + 1 frames
(5 for N = 16) and the filtering animation has ``FILTER_FRAMES`` = 30 frames
of 576 x 448 pixels.

Examples:
    >>> import os, tempfile
    >>> from PIL import Image
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = animate_fft_stages(np.arange(16.0), os.path.join(folder, "stages.gif"))
    ...     with Image.open(path) as gif:
    ...         gif.n_frames <= 40
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
from typing import List

import matplotlib
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.fft_analyzer.core.fft import fft, fft_iterative_stages, rfft_freqs
from src.fft_analyzer.signal.filters import fft_filter
from src.utils.visualization import apply_house_style

__all__ = ["animate_fft_stages", "animate_filtering", "FILTER_FRAMES", "MAX_FRAMES"]

#: Upper bound on the frames of either animation (spec 2.2).
MAX_FRAMES: int = 40
#: Frames in the cutoff sweep of :func:`animate_filtering`.
FILTER_FRAMES: int = 30
#: Most samples drawn in the time panel of :func:`animate_filtering`.
MAX_SHOWN: int = 256
GIF_DPI: int = 80

INK = "#1f1f1f"
BAR = "#2471a3"
KEPT = "#2471a3"
REMOVED = "#7f8c8d"
CUTOFF = "#c0392b"


# ----------------------------------------------------------------- helpers


def _gif_figure(width: float, height: float) -> Figure:
    """Return a white figure on an Agg canvas for frame-by-frame GIF writing.

    Args:
        width: Width in inches.
        height: Height in inches.

    Returns:
        A new :class:`~matplotlib.figure.Figure`.

    Examples:
        >>> fig = _gif_figure(7.2, 4.2)
        >>> [int(v) for v in fig.get_size_inches() * GIF_DPI]
        [576, 336]
    """
    fig = Figure(figsize=(width, height), dpi=GIF_DPI, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _prepare_path(out_path: str) -> None:
    """Create the parent folder of ``out_path`` (PillowWriter needs it to exist).

    Args:
        out_path: The GIF path.

    Examples:
        >>> _prepare_path("frames.gif") is None
        True
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)


def _real_signal(x) -> np.ndarray:
    """Return ``x`` as a 1-D ``float64`` array of at least two samples.

    Args:
        x: A 1-D real sequence.

    Returns:
        A ``float64`` array.

    Raises:
        ValueError: If ``x`` is complex, not 1-D or shorter than 2.

    Examples:
        >>> _real_signal([1, 2])
        array([1., 2.])
        >>> _real_signal([1])
        Traceback (most recent call last):
        ...
        ValueError: x must be a 1-D real signal of at least 2 samples, got shape (1,)
    """
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise ValueError("x must be real")
    a = a.astype(np.float64)
    if a.ndim != 1 or a.shape[0] < 2:
        raise ValueError(f"x must be a 1-D real signal of at least 2 samples, got shape {a.shape}")
    return a


def _stage_title(stage: int, last: int, n: int) -> str:
    """Return the frame title for stage ``stage`` of an N-point FFT.

    Args:
        stage: 0 for the bit-reversal frame, else the stage number.
        last: log2 N, the number of the final stage.
        n: The transform length N.

    Returns:
        The title text.

    Examples:
        >>> _stage_title(0, 4, 16)
        'Bit-reversal permutation (N = 16): position i holds x[rev(i)]'
        >>> _stage_title(4, 4, 16)
        'Stage 4 of 4: groups of 16 combined, the result is the DFT X[k]'
    """
    if stage == 0:
        return f"Bit-reversal permutation (N = {n}): position i holds x[rev(i)]"
    title = f"Stage {stage} of {last}: groups of {2 ** stage} combined"
    if stage == last:
        title += ", the result is the DFT X[k]"
    return title


def _draw_stage(ax: Axes, mags: np.ndarray, stage: int, last: int, top: float) -> None:
    """Draw one frame of the stages animation on a cleared ``ax``.

    Args:
        ax: The axes to draw on (cleared first).
        mags: ``|a[i]|`` for the working array after this stage.
        stage: 0 for the bit-reversal frame, else the stage number.
        last: log2 N.
        top: The fixed upper y limit for every frame.
    """
    n = mags.shape[0]
    ax.clear()
    bars = ax.bar(np.arange(n), mags, width=0.72, color=BAR, edgecolor=INK, linewidth=0.6,
                  label="|a[i]|, magnitude of entry i")
    handles: List = [bars]
    if stage >= 1 and 2 ** stage < n:
        size = 2 ** stage
        for edge in range(size, n, size):
            ax.axvline(edge - 0.5, color=INK, linestyle=":", linewidth=1.2)
        handles.append(Line2D([], [], color=INK, linestyle=":", linewidth=1.2,
                              label=f"boundary between groups of {size}"))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0.0, top)
    step = max(1, n // 16)
    ax.set_xticks(np.arange(0, n, step))
    ax.set_xlabel("Array position i (= frequency index k after the last stage)")
    ax.set_ylabel("|a[i]| (units of x)")
    ax.figure.suptitle(_stage_title(stage, last, n), fontsize=11)
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
              fontsize=8, frameon=False)


# ---------------------------------------------------------------- animations


def animate_fft_stages(x, out_path: str) -> str:
    """Write a GIF of the iterative FFT: bit-reversal, then one frame per butterfly stage.

    The frames come from
    :func:`~src.fft_analyzer.core.fft.fft_iterative_stages`, so they are the
    exact arrays :func:`~src.fft_analyzer.core.fft.fft_iterative` computes.
    Frame 0 is the input permuted into bit-reversed order; frame s
    (s = 1..log2 N) is the array after stage s, in which every group of
    2^s consecutive entries has become the DFT of the samples it holds.
    Dotted lines mark the group boundaries, and the y scale is the same in
    every frame. The animation is designed for N = 16 (5 frames at one frame
    per second) and accepts any power-of-two length; that is at most
    log2 N + 1 frames, far below the 40-frame limit for any real N.

    Args:
        x: A 1-D sequence whose length N is a power of two (16 for the
            report's animation).
        out_path: Where to write the GIF. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        ValueError: If the length is not a power of two, or ``x`` is empty or
            not 1-D.

    Time Complexity:
        O(N log N) for the stages, plus rendering log2 N + 1 frames of N bars.

    Space Complexity:
        O(N log N) for the stored stages, plus one frame image per stage.

    Examples:
        >>> import os, tempfile
        >>> from PIL import Image
        >>> from src.utils.random_utils import derive_seed
        >>> rng = np.random.default_rng(derive_seed(42, "animate_fft_stages", "doctest"))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "gifs", "fft_stages.gif")
        ...     out = animate_fft_stages(rng.standard_normal(16), path)
        ...     with Image.open(out) as gif:
        ...         out == path, gif.format, gif.n_frames, os.path.getsize(out) <= 2 * 1024 * 1024
        (True, 'GIF', 5, True)
        >>> animate_fft_stages(np.ones(12), "unused.gif")
        Traceback (most recent call last):
        ...
        ValueError: fft_iterative_stages needs a power-of-two length, got 12 (use fft(x, n=16) to zero-pad)
    """
    stages = fft_iterative_stages(x)
    mags = [np.abs(s) for s in stages]
    n = mags[0].shape[0]
    last = len(stages) - 1
    peak = max(float(m.max()) for m in mags)
    top = 1.12 * peak if peak > 0.0 else 1.0
    apply_house_style()
    fig = _gif_figure(7.2, 4.2)
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.14, top=0.82)
    _prepare_path(out_path)
    writer = PillowWriter(fps=1)
    with matplotlib.rc_context({"savefig.bbox": None}), writer.saving(fig, out_path, GIF_DPI):
        for stage, m in enumerate(mags[:MAX_FRAMES]):
            _draw_stage(ax, m, stage, last, top)
            writer.grab_frame()
    return out_path


def animate_filtering(x, fs: float, out_path: str) -> str:
    """Write a GIF of a brick-wall low-pass filter whose cutoff sweeps down.

    The cutoff takes ``FILTER_FRAMES`` = 30 evenly spaced values from the
    Nyquist frequency ``fs / 2`` down to ``fs / 60``, one per frame. Each
    frame filters x with
    :func:`~src.fft_analyzer.signal.filters.fft_filter` (kind ``"low"``)
    and shows:

    * top: the one-sided magnitude spectrum of x (zero-padded to the same
      power of two the filter uses), solid where ``|f| <= cutoff`` (kept) and
      dotted where it is removed, with the cutoff as a dashed vertical line;
    * bottom: the first tenth of the record, clamped to between 50 and
      ``MAX_SHOWN`` = 256 samples (all of it if shorter than 50), the
      original dashed and the filtered output solid. The upper clamp keeps
      individual cycles readable for audio-rate input (256 samples is 16 ms
      at 16 kHz) and keeps the GIF small.

    The title gives the cutoff and the share of the signal energy
    ``sum(y^2) / sum(x^2)`` that the filter kept. Both panels keep fixed
    scales across frames.

    Args:
        x: A 1-D real signal of at least 2 samples.
        fs: The sampling rate in Hz, positive.
        out_path: Where to write the GIF. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        ValueError: If ``x`` is complex, not 1-D or shorter than 2, or ``fs``
            is not a finite positive number.

    Time Complexity:
        O(F m log m) for F = 30 filters of the padded length m < 2 len(x),
        plus rendering F frames.

    Space Complexity:
        O(F len(x)) for the filtered signals, plus one frame image per frame.

    Examples:
        >>> import os, tempfile
        >>> from PIL import Image
        >>> from src.fft_analyzer.signal.generators import multi_tone
        >>> x = multi_tone([50, 120], [1.0, 0.5], fs=1000, duration=1.0)
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "filtering.gif")
        ...     out = animate_filtering(x, 1000, path)
        ...     with Image.open(out) as gif:
        ...         gif.n_frames <= MAX_FRAMES, os.path.getsize(out) <= 2 * 1024 * 1024
        (True, True)
        >>> animate_filtering(x, 0, "unused.gif")
        Traceback (most recent call last):
        ...
        ValueError: fs must be a finite positive number, got 0
    """
    a = _real_signal(x)
    rate = float(fs)
    if not (math.isfinite(rate) and rate > 0.0):
        raise ValueError(f"fs must be a finite positive number, got {fs}")
    n = a.shape[0]
    m = 1 << (n - 1).bit_length()
    freqs = rfft_freqs(m, rate)
    spectrum = np.abs(fft(a, n=m))[: m // 2 + 1]
    nyquist = rate / 2.0
    cutoffs = [nyquist * (FILTER_FRAMES - j) / FILTER_FRAMES for j in range(FILTER_FRAMES)]
    outputs: List[np.ndarray] = [fft_filter(a, rate, "low", c) for c in cutoffs]
    total = float(np.sum(a * a))
    shown = min(n, max(50, min(n // 10, MAX_SHOWN)))
    t = np.arange(shown) / rate
    amp = max(float(np.max(np.abs(a))), max(float(np.max(np.abs(y[:shown]))) for y in outputs))
    amp = 1.1 * amp if amp > 0.0 else 1.0
    spec_top = 1.1 * float(spectrum.max()) if spectrum.max() > 0.0 else 1.0

    apply_house_style()
    fig = _gif_figure(7.2, 5.6)
    top_ax = fig.add_subplot(2, 1, 1)
    bottom_ax = fig.add_subplot(2, 1, 2)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.1, top=0.86, hspace=0.62)
    legend_style = dict(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    _prepare_path(out_path)
    writer = PillowWriter(fps=5)
    with matplotlib.rc_context({"savefig.bbox": None}), writer.saving(fig, out_path, GIF_DPI):
        for cutoff, y in list(zip(cutoffs, outputs))[:MAX_FRAMES]:
            kept = freqs <= cutoff
            top_ax.clear()
            top_ax.plot(freqs, np.where(kept, spectrum, np.nan), color=KEPT, linestyle="-",
                        linewidth=1.6, label="kept (|f| <= cutoff)")
            top_ax.plot(freqs, np.where(kept, np.nan, spectrum), color=REMOVED, linestyle=":",
                        linewidth=1.4, label="removed")
            top_ax.axvline(cutoff, color=CUTOFF, linestyle="--", linewidth=1.4, label="cutoff")
            top_ax.set_xlim(0.0, nyquist)
            top_ax.set_ylim(0.0, spec_top)
            top_ax.set_xlabel("Frequency (Hz)")
            top_ax.set_ylabel("|X(f)| (arbitrary units)")
            top_ax.set_title("Spectrum of x", fontsize=10, loc="left")
            top_ax.legend(ncol=3, **legend_style)

            # The original is drawn last, thin and dashed, so it stays visible
            # where the output still matches it.
            bottom_ax.clear()
            bottom_ax.plot(t, y[:shown], color=KEPT, linestyle="-", linewidth=2.2,
                           label="low-pass output y")
            bottom_ax.plot(t, a[:shown], color=INK, linestyle="--", linewidth=0.9,
                           label="original x")
            bottom_ax.set_xlim(0.0, float(t[-1]) if shown > 1 else 1.0 / rate)
            bottom_ax.set_ylim(-amp, amp)
            bottom_ax.set_xlabel("Time (s)")
            bottom_ax.set_ylabel("Amplitude (units of x)")
            bottom_ax.set_title(f"First {shown} of {n} samples", fontsize=10, loc="left")
            bottom_ax.legend(ncol=2, **legend_style)

            share = f"{100.0 * float(np.sum(y * y)) / total:.1f}%" if total > 0.0 else "n/a"
            fig.suptitle(f"Low-pass cutoff {cutoff:.4g} Hz: {share} of the energy kept",
                         fontsize=12)
            writer.grab_frame()
    return out_path
