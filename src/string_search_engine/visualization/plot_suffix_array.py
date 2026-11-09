"""Draw a suffix array next to its LCP array, one row per sorted suffix.

What the figure shows
---------------------
A suffix array preprocesses the *text*, not the pattern: before any search it
stores the start positions of every suffix of the text in lexicographic order,
and the LCP array stores how many leading characters each suffix shares with
the one sorted just above it. :func:`plot_suffix_array` draws both for a short
text in two panels that share their rows:

* left, a table: the rank ``r``, the position ``SA[r]`` and the suffix
  ``text[SA[r]:]`` spelled one character per cell. The first ``LCP[r]``
  characters of each suffix, the prefix it shares with the row above, are
  hatched, shaded and bold;
* right, a horizontal bar chart of ``LCP[r]`` for the same rows, every bar
  labelled with its value. ``LCP[0] = 0`` by convention, since rank 0 has no
  row above it.

Reading down the table shows why a pattern search is two binary searches: all
suffixes that start with the pattern sit in one contiguous block of rows. The
hatching shows where the LCP values come from, so the shared prefix is carried
by hatching and bold type as well as by colour.

The arrays are computed with
:func:`~src.string_search_engine.core.suffix_array.build_suffix_array` (numpy
prefix doubling, no sentinel) and
:func:`~src.string_search_engine.core.suffix_array.build_lcp` (Kasai), so the
figure shows exactly what the search engine uses.

Backend
-------
The figure is a :class:`~matplotlib.figure.Figure` on an explicit Agg canvas,
not a pyplot figure, so nothing opens a window and nothing is left in pyplot's
figure registry.

Examples:
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     out = plot_suffix_array("banana", os.path.join(folder, "sa.png"))
    ...     os.path.getsize(out) > 0
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import Any, List

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from src.string_search_engine.core.suffix_array import build_lcp, build_suffix_array
from src.utils.visualization import apply_house_style

__all__ = ["plot_suffix_array", "MAX_PLOT_CHARS"]

#: Longest text the figure draws; one table row per character.
MAX_PLOT_CHARS: int = 200
#: Suffix characters drawn per row; longer suffixes end in "...".
MAX_SUFFIX_CELLS: int = 40

INK = "#0b0b0b"
MUTED = "#5f5e5a"
ROW_SHADE = "#f1f0ec"
SHARED_FILL = "#cfe0f6"
SHARED_EDGE = "#2a78d6"
SHARED_HATCH = "#86b0e8"
BAR_FILL = "#2a78d6"

CELL_IN = 0.2  # inches per table cell
ROW_IN = 0.32  # inches per table row
RANK_X = 1.3  # centre of the rank column, in cells
POS_X = 3.9  # centre of the position column, in cells
SUFFIX_X0 = 5.6  # left edge of the first suffix character, in cells


def _visible(ch: str) -> str:
    """Return a printable stand-in for one character of the text."""
    if ch == "\x00":
        return "$"
    if ch == " ":
        return "\u2423"
    if not ch.isprintable():
        return "\u00b7"
    return ch


def _check_text(text: Any) -> None:
    """Raise unless ``text`` is a non-empty ``str`` short enough to draw."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if not text:
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_PLOT_CHARS:
        raise ValueError(
            f"text has {len(text)} characters; plot_suffix_array draws at most {MAX_PLOT_CHARS}"
        )


def _draw_table(ax: Any, text: str, sa: List[int], lcp: List[int], cells: int) -> None:
    """Draw the rank, position and suffix columns on ``ax``, one row per rank."""
    n = len(text)
    right = SUFFIX_X0 + cells + 1.0
    for r in range(n):
        if r % 2 == 0:
            ax.add_patch(
                Rectangle((0.0, r - 0.5), right, 1.0, facecolor=ROW_SHADE, edgecolor="none", zorder=0)
            )
    for r, start in enumerate(sa):
        suffix = text[start:]
        shared = min(lcp[r], cells)
        if shared > 0:
            ax.add_patch(
                Rectangle(
                    (SUFFIX_X0, r - 0.42),
                    shared,
                    0.84,
                    facecolor=SHARED_FILL,
                    edgecolor=SHARED_EDGE,
                    hatchcolor=SHARED_HATCH,
                    hatch="///",
                    linewidth=0.8,
                    zorder=1,
                )
            )
        ax.text(RANK_X, r, str(r), ha="center", va="center", fontsize=10, color=MUTED, zorder=2)
        ax.text(POS_X, r, str(start), ha="center", va="center", fontsize=10, color=INK, zorder=2)
        for j, ch in enumerate(suffix[:cells]):
            ax.text(
                SUFFIX_X0 + j + 0.5,
                r,
                _visible(ch),
                ha="center",
                va="center",
                family="monospace",
                fontsize=12,
                fontweight="bold" if j < lcp[r] else "normal",
                color=INK,
                parse_math=False,
                zorder=2,
            )
        if len(suffix) > cells:
            ax.text(SUFFIX_X0 + cells + 0.1, r, "...", ha="left", va="center", fontsize=10, color=MUTED)
    header = dict(ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(RANK_X, -1.0, "rank r", **header)
    ax.text(POS_X, -1.0, "SA[r]", **header)
    ax.text(SUFFIX_X0, -1.0, "suffix text[SA[r]:]", **{**header, "ha": "left"})
    ax.plot([0.0, right], [-0.5, -0.5], color=INK, linewidth=1.0)
    for x in (2.6, 5.2):
        ax.plot([x, x], [-1.5, n - 0.5], color=MUTED, linewidth=0.6)
    ax.set_xlim(0.0, right)
    ax.set_ylim(n - 0.5, -1.6)
    ax.set_axis_off()
    ax.set_title("Sorted suffixes (suffix array)", fontsize=11)
    ax.legend(
        handles=[
            Patch(
                facecolor=SHARED_FILL,
                edgecolor=SHARED_EDGE,
                hatchcolor=SHARED_HATCH,
                hatch="///",
                label="prefix shared with the row above\n(LCP[r] characters, bold)",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        fontsize=9,
    )


def _draw_lcp(ax: Any, lcp: List[int]) -> None:
    """Draw ``LCP[r]`` as labelled horizontal bars aligned with the table rows."""
    n = len(lcp)
    ranks = list(range(n))
    ax.barh(ranks, lcp, height=0.62, color=BAR_FILL, edgecolor=INK, linewidth=0.6, zorder=2)
    top = max(lcp)
    for r, value in enumerate(lcp):
        ax.text(value + 0.08 * max(top, 1), r, str(value), ha="left", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, max(top, 1) * 1.25 + 0.5)
    ax.set_ylim(n - 0.5, -1.6)
    ax.set_yticks(ranks)
    ax.set_yticklabels([str(r) for r in ranks], fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("LCP[r] with rank r - 1 (characters)")
    ax.set_ylabel("suffix array rank r")
    ax.set_title("LCP array (LCP[0] = 0)", fontsize=11)


def plot_suffix_array(text: str, out_path: str) -> str:
    """Save a two-panel PNG of the suffix array and LCP array of ``text``.

    The left panel is a table with one row per suffix in sorted order: its
    rank, its start position ``SA[r]`` and the suffix itself, one character
    per cell, with the first ``LCP[r]`` characters hatched, shaded and bold.
    The right panel is a bar chart of ``LCP[r]`` on the same rows, each bar
    labelled with its value. The terminator is not added, so the arrays are
    the ones :class:`~src.string_search_engine.core.suffix_array.SuffixArray`
    holds. Suffixes longer than ``MAX_SUFFIX_CELLS`` characters are cut and
    end in ``...``; spaces are drawn as an open box.

    Args:
        text: A non-empty ``str`` of at most ``MAX_PLOT_CHARS`` characters.
        out_path: Where to write the PNG. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        TypeError: If ``text`` is not a ``str``.
        ValueError: If ``text`` is empty or longer than ``MAX_PLOT_CHARS``.

    Time Complexity:
        O(n log^2 n) for the suffix array, O(n) for the LCP array, plus
        O(n * min(n, MAX_SUFFIX_CELLS)) text artists to render.

    Space Complexity:
        O(n * min(n, MAX_SUFFIX_CELLS)) artists plus the image.

    Examples:
        >>> import os, tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "figures", "suffix_array_mississippi.png")
        ...     out = plot_suffix_array("mississippi", path)
        ...     with open(out, "rb") as handle:
        ...         out == path, handle.read(8) == b"\\x89PNG\\r\\n\\x1a\\n"
        (True, True)
        >>> plot_suffix_array("", "unused.png")
        Traceback (most recent call last):
            ...
        ValueError: text must be a non-empty string
    """
    _check_text(text)
    sa = build_suffix_array(text)
    lcp = build_lcp(text, sa)
    n = len(text)
    cells = min(n, MAX_SUFFIX_CELLS)

    apply_house_style()
    table_in = (SUFFIX_X0 + cells + 1.0) * CELL_IN
    lcp_in = 3.0
    height_in = (n + 2) * ROW_IN + 1.2
    figure = Figure(figsize=(table_in + lcp_in + 1.6, height_in))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(1, 2, width_ratios=[table_in, lcp_in], wspace=0.3)
    table_ax = figure.add_subplot(grid[0])
    lcp_ax = figure.add_subplot(grid[1])
    _draw_table(table_ax, text, sa, lcp, cells)
    _draw_lcp(lcp_ax, lcp)
    figure.suptitle(
        f"Suffix array and LCP array of {text!r} (n = {n})", fontsize=12, parse_math=False
    )

    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path
