"""Draw a suffix tree with matplotlib, and write it as Graphviz DOT text.

What the figure shows
---------------------
A suffix tree preprocesses the *text*, not the pattern: before any search it
stores one compressed trie over ``text + "\\x00"`` in which every suffix is a
root-to-leaf path, so a query only walks down from the root.
:func:`draw_suffix_tree` draws that tree for a short text:

* internal nodes (the branching points, root included) are filled circles;
* leaves are squares holding the start position of their suffix;
* each edge is labelled with the substring it spells, the terminator shown as
  ``"$"``.

Node kind is carried by shape as well as colour, so the figure survives a
greyscale printout.

Tidy-tree layout
----------------
The layout is the classic tidy drawing of an ordered tree. Children are taken
in the tree's own order (sorted by first character, ``"$"`` first), so a
depth-first walk meets the leaves in suffix array order; each leaf gets the
next slot on an evenly spaced row of x positions, and each internal node is
centred over its first and last child. A node's height is its depth in edges
from the root. Leaves then read left to right as the suffix array, which is
the Option A construction drawn backwards. The walk is iterative, like every
traversal in :mod:`~src.string_search_engine.core.suffix_tree`.

:func:`write_dot` saves :meth:`~src.string_search_engine.core.suffix_tree.SuffixTree.to_dot`
as a ``.dot`` file; it is plain text, so no Graphviz binary or package is
needed to produce it.

Backend
-------
The figure is a :class:`~matplotlib.figure.Figure` on an explicit Agg canvas,
not a pyplot figure, so nothing opens a window and nothing is left in pyplot's
figure registry.

Examples:
    >>> import os, tempfile
    >>> tree = SuffixTree("banana")
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     png = draw_suffix_tree(tree, os.path.join(folder, "tree.png"))
    ...     dot = write_dot(tree, os.path.join(folder, "tree.dot"))
    ...     os.path.getsize(png) > 0, os.path.getsize(dot) > 0
    (True, True)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.string_search_engine.core.suffix_tree import TERMINATOR, SuffixTree, SuffixTreeNode
from src.utils.visualization import apply_house_style

__all__ = ["draw_suffix_tree", "write_dot", "MAX_DRAW_LEAVES"]

#: Most leaves (text length + 1) the drawing accepts.
MAX_DRAW_LEAVES: int = 200
#: Longest edge label drawn in full; longer labels are cut and end in "...".
MAX_EDGE_LABEL: int = 10

INK = "#0b0b0b"
EDGE_COLOR = "#8a8984"
INTERNAL_FILL = "#2a78d6"
LEAF_FILL = "#fde3d6"
LEAF_EDGE = "#eb6834"

LEAF_SIDE = 22.0  # points
INTERNAL_DIAMETER = 13.0  # points
SLOT_IN = 0.8  # inches between neighbouring leaves
LEVEL_IN = 1.15  # inches between tree levels
LABEL_AT = 0.6  # edge labels sit this far from parent to child, where siblings spread
LABEL_ALT = 0.38  # fallback position when a label would overlap its left neighbour
LABEL_FONT = 10.0  # points


def _label_fractions(
    parents: List[int], levels: List[int], xs: List[float], labels: List[str]
) -> List[float]:
    """Return, per node, how far along its incoming edge the label sits.

    Labels go at ``LABEL_AT``; a label whose estimated box would overlap the
    label to its left on the same row moves to ``LABEL_ALT``, a row closer to
    the parent, so adjacent long labels (``"ppi$"`` next to ``"ssippi$"``)
    stay readable. Index 0, the root, has no edge and keeps ``LABEL_AT``.

    Examples:
        >>> _label_fractions([-1, 0, 0], [0, 1, 1], [0.5, 0.0, 1.0], ["", "a", "b"])
        [0.6, 0.6, 0.6]
        >>> _label_fractions([-1, 0, 0], [0, 1, 1], [0.5, 0.0, 1.0], ["", "ppi$", "ssippi$"])
        [0.6, 0.6, 0.38]
    """
    fractions = [LABEL_AT] * len(parents)

    def centre_in(i: int, f: float) -> float:
        return (xs[parents[i]] + f * (xs[i] - xs[parents[i]])) * SLOT_IN

    right_edge: dict = {}
    for i in sorted(range(1, len(parents)), key=lambda j: (levels[j], centre_in(j, LABEL_AT))):
        half = (len(labels[i]) * 0.6 * LABEL_FONT + 0.36 * LABEL_FONT + 4.0) / 144.0
        options = []
        for f in (LABEL_AT, LABEL_ALT):
            clearance = centre_in(i, f) - half - right_edge.get((levels[i], f), float("-inf"))
            options.append((clearance >= 0, clearance, f))
        chosen = options[0] if options[0][0] else max(options[1:] + options[:1])
        fractions[i] = chosen[2]
        right_edge[(levels[i], chosen[2])] = centre_in(i, chosen[2]) + half
    return fractions


def _check_tree(tree: Any) -> None:
    """Raise ``TypeError`` unless ``tree`` is a :class:`SuffixTree`."""
    if not isinstance(tree, SuffixTree):
        raise TypeError(f"tree must be a SuffixTree, got {type(tree).__name__}")


def _visible(label: str) -> str:
    """Return an edge label with the terminator as "$" and blanks made visible."""
    out = []
    for ch in label:
        if ch == TERMINATOR:
            out.append("$")
        elif ch == " ":
            out.append("\u2423")
        elif not ch.isprintable():
            out.append("\u00b7")
        else:
            out.append(ch)
    text = "".join(out)
    if len(text) > MAX_EDGE_LABEL:
        text = text[: MAX_EDGE_LABEL - 3] + "..."
    return text


def _tidy_layout(
    tree: SuffixTree,
) -> Tuple[List[SuffixTreeNode], List[int], List[int], List[float]]:
    """Return the nodes in preorder with their parent index, level and x.

    Index 0 is the root (parent -1), and indices match the ``n<i>`` names in
    :meth:`SuffixTree.to_dot`. Leaves sit at x = 0, 1, 2, ... in left-to-right
    order; each internal node is centred over its first and last child.

    Examples:
        >>> nodes, parents, levels, xs = _tidy_layout(SuffixTree("aa"))
        >>> parents, levels, xs
        ([-1, 0, 0, 2, 2], [0, 1, 1, 2, 2], [0.75, 0.0, 1.5, 1.0, 2.0])
    """
    nodes: List[SuffixTreeNode] = []
    parents: List[int] = []
    levels: List[int] = []
    stack = [(tree.root, -1, 0)]
    while stack:
        node, parent, level = stack.pop()
        index = len(nodes)
        nodes.append(node)
        parents.append(parent)
        levels.append(level)
        if node.children is not None:
            stack.extend((c, index, level + 1) for c in reversed(list(node.children.values())))
    first = [-1] * len(nodes)
    last = [-1] * len(nodes)
    for index in range(1, len(nodes)):
        parent = parents[index]
        if first[parent] < 0:
            first[parent] = index
        last[parent] = index
    xs = [0.0] * len(nodes)
    slot = 0
    for index, node in enumerate(nodes):
        if node.children is None:
            xs[index] = float(slot)
            slot += 1
    for index in reversed(range(len(nodes))):
        if first[index] >= 0:
            xs[index] = (xs[first[index]] + xs[last[index]]) / 2.0
    return nodes, parents, levels, xs


def draw_suffix_tree(tree: SuffixTree, out_path: str) -> str:
    """Save a PNG of ``tree`` drawn with a tidy-tree layout.

    Leaves are evenly spaced in suffix array order and every internal node is
    centred over its children, one level per edge from the root. Internal
    nodes are filled circles, leaves are squares showing their suffix start
    position, and each edge carries its substring, placed past the edge
    midpoint where siblings spread apart (or nearer the parent when it would
    overlap its left neighbour), with ``"$"`` for the
    terminator ``"\\x00"``. Labels longer than ``MAX_EDGE_LABEL`` characters
    are cut and end in ``...``; spaces are drawn as an open box.

    Args:
        tree: A :class:`~src.string_search_engine.core.suffix_tree.SuffixTree`
            with at most ``MAX_DRAW_LEAVES`` leaves (text length + 1).
        out_path: Where to write the PNG. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        TypeError: If ``tree`` is not a ``SuffixTree``.
        ValueError: If the tree has more than ``MAX_DRAW_LEAVES`` leaves.

    Time Complexity:
        O(n) for the layout and O(n) artists to render, n the text length.

    Space Complexity:
        O(n) plus the image.

    Examples:
        >>> import os, tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "figures", "suffix_tree_banana.png")
        ...     out = draw_suffix_tree(SuffixTree("banana"), path)
        ...     with open(out, "rb") as handle:
        ...         out == path, handle.read(8) == b"\\x89PNG\\r\\n\\x1a\\n"
        (True, True)
        >>> draw_suffix_tree("banana", "unused.png")
        Traceback (most recent call last):
            ...
        TypeError: tree must be a SuffixTree, got str
    """
    _check_tree(tree)
    counts = tree.node_counts()
    if counts["leaves"] > MAX_DRAW_LEAVES:
        raise ValueError(
            f"tree has {counts['leaves']} leaves; draw_suffix_tree draws at most {MAX_DRAW_LEAVES}"
        )
    nodes, parents, levels, xs = _tidy_layout(tree)
    ys = [-float(level) for level in levels]
    t = tree.terminated
    n_terminated = len(t)
    slots = counts["leaves"]
    depth = max(levels)

    apply_house_style()
    figure = Figure(figsize=(max(5.5, slots * SLOT_IN + 1.0), max(3.2, depth * LEVEL_IN + 1.8)))
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()

    labels = [""] + [_visible(t[node.start : node.end]) for node in nodes[1:]]
    fractions = _label_fractions(parents, levels, xs, labels)
    for index in range(1, len(nodes)):
        parent = parents[index]
        f = fractions[index]
        ax.plot(
            [xs[parent], xs[index]], [ys[parent], ys[index]], color=EDGE_COLOR, linewidth=1.4, zorder=1
        )
        ax.text(
            xs[parent] + f * (xs[index] - xs[parent]),
            ys[parent] + f * (ys[index] - ys[parent]),
            labels[index],
            ha="center",
            va="center",
            family="monospace",
            fontsize=LABEL_FONT,
            color=INK,
            parse_math=False,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=EDGE_COLOR, linewidth=0.6),
            zorder=4,
        )

    internal = [i for i, node in enumerate(nodes) if node.children is not None]
    leaves = [i for i, node in enumerate(nodes) if node.children is None]
    ax.scatter(
        [xs[i] for i in internal],
        [ys[i] for i in internal],
        s=INTERNAL_DIAMETER**2,
        marker="o",
        facecolor=INTERNAL_FILL,
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    ax.scatter(
        [xs[i] for i in leaves],
        [ys[i] for i in leaves],
        s=LEAF_SIDE**2,
        marker="s",
        facecolor=LEAF_FILL,
        edgecolor=LEAF_EDGE,
        linewidth=1.2,
        zorder=3,
    )
    for i in leaves:
        start = str(n_terminated - nodes[i].depth)
        ax.text(
            xs[i],
            ys[i],
            start,
            ha="center",
            va="center",
            fontsize=9 if len(start) <= 2 else 7,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
    ax.text(xs[0], 0.28, "root", ha="center", va="bottom", fontsize=10, color=INK)

    ax.set_xlim(-0.7, slots - 0.3)
    ax.set_ylim(-depth - 0.45, 0.6)
    ax.set_axis_off()
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", markersize=9,
                markerfacecolor=INTERNAL_FILL, markeredgecolor=INK,
                label="internal node (branch point)",
            ),
            Line2D(
                [], [], marker="s", linestyle="none", markersize=11,
                markerfacecolor=LEAF_FILL, markeredgecolor=LEAF_EDGE,
                label="leaf: suffix start position",
            ),
            Line2D([], [], color=EDGE_COLOR, linewidth=1.4, label='edge: substring, "$" = end marker'),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3 if slots >= 7 else 1,
        frameon=False,
        fontsize=9,
    )
    ax.set_title(
        f'Suffix tree of {tree.text!r} + "$": {counts["leaves"]} leaves, '
        f'{counts["internal"]} internal nodes (root excluded)',
        fontsize=11,
        parse_math=False,
    )

    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def write_dot(tree: SuffixTree, out_path: str) -> str:
    """Write ``tree.to_dot()`` to ``out_path`` as a Graphviz ``.dot`` file.

    The file is the DOT text of
    :meth:`~src.string_search_engine.core.suffix_tree.SuffixTree.to_dot`
    followed by a newline, in UTF-8. Render it with any Graphviz install
    (``dot -Tpng tree.dot``); nothing here needs Graphviz.

    Args:
        tree: A :class:`~src.string_search_engine.core.suffix_tree.SuffixTree`.
        out_path: Where to write the file. Parent folders are created.

    Returns:
        ``out_path``, as given.

    Raises:
        TypeError: If ``tree`` is not a ``SuffixTree``.

    Time Complexity:
        O(n + total edge label length), the cost of ``to_dot``.

    Space Complexity:
        The size of the DOT text.

    Examples:
        >>> import os, tempfile
        >>> tree = SuffixTree("banana")
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = write_dot(tree, os.path.join(folder, "suffix_tree_banana.dot"))
        ...     with open(out, encoding="utf-8") as handle:
        ...         content = handle.read()
        >>> content == tree.to_dot() + "\\n", content.splitlines()[0]
        (True, 'digraph SuffixTree {')
        >>> write_dot(None, "unused.dot")
        Traceback (most recent call last):
            ...
        TypeError: tree must be a SuffixTree, got NoneType
    """
    _check_tree(tree)
    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(tree.to_dot() + "\n")
    return out_path
