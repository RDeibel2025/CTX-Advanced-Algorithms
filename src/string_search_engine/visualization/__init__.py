"""Suffix array and suffix tree figures (Week 11).

Two modules:

* :mod:`plot_suffix_array` - :func:`plot_suffix_array` draws the suffix array
  as a table of rank, position and suffix beside a labelled LCP bar chart,
  for texts up to :data:`MAX_PLOT_CHARS` characters.
* :mod:`draw_suffix_tree` - :func:`draw_suffix_tree` draws a
  :class:`~src.string_search_engine.core.SuffixTree` with a tidy layout (leaves
  evenly spaced, parents centred), edge substrings as labels and suffix start
  positions on the leaves, for trees up to :data:`MAX_DRAW_LEAVES` leaves.
  :func:`write_dot` saves the tree's Graphviz DOT text, so no Graphviz binary
  or package is needed.

Both use matplotlib's Agg canvas and the house style from
:mod:`src.utils.visualization`; nothing here opens a window.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.string_search_engine.visualization.draw_suffix_tree import (
    MAX_DRAW_LEAVES,
    draw_suffix_tree,
    write_dot,
)
from src.string_search_engine.visualization.plot_suffix_array import (
    MAX_PLOT_CHARS,
    plot_suffix_array,
)

__all__ = ["plot_suffix_array", "MAX_PLOT_CHARS", "draw_suffix_tree", "write_dot", "MAX_DRAW_LEAVES"]
