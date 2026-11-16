"""Balanced and Huffman-shaped wavelet trees over a sequence of symbols.

The idea
--------
A wavelet tree halves the alphabet, not the positions. The root stores one
bit per position of the sequence: 0 when the symbol at that position belongs
to the left part of the alphabet, 1 when it belongs to the right part. The
symbols whose bit is 0 form the left child's sequence, in their original
order, and those whose bit is 1 form the right child's. Each child splits its
own part of the alphabet again, until a node is left with a single symbol.
Such a leaf stores nothing: every position that reaches it holds that symbol.

Every internal node keeps its bits in a :class:`BitVector` with a
:class:`RankSelect` on top, so moving between a node and a child is one rank
or one select. Position i of a node is position ``rank0(i)`` of its left
child when its bit is 0, and ``rank1(i)`` of its right child when it is 1;
going back up, the k-th position of a child is ``select0(k)`` or
``select1(k)`` of its parent. Each query walks one root-to-leaf path (range
counting walks a few) and does a constant amount of rank or select work per
level, so it costs O(h) for a tree of height h: ``ceil(log2 sigma)`` levels
for the balanced shape, 2 for DNA.

Shapes
------
* ``"balanced"``: the alphabet, in alphabet order, is split in half at every
  node. When a node holds an odd number m of symbols, **the left half gets
  the larger part, ceil(m / 2) symbols**, so ``"$ACGT"`` splits into
  ``"$AC"`` and ``"GT"``. Every leaf is then at depth ``floor(log2 sigma)``
  or ``ceil(log2 sigma)`` and the payload is ``sum(count(c) * depth(c))``
  bits. When sigma is a power of two every leaf is at depth ``log2 sigma``
  and the payload is exactly ``n * log2 sigma`` bits (2 bits per symbol for
  DNA); for any other sigma it lies between ``n * floor(log2 sigma)`` and
  ``n * ceil(log2 sigma)`` bits. The leaves appear left to right in alphabet
  order, which is what :meth:`WaveletTree.range_quantile` relies on.
* ``"huffman"``: the tree is the code tree of a Huffman code built from the
  symbol counts, so the depth of a symbol is the length of its code word and
  the payload is ``sum(count(c) * len(code(c)))`` bits, at least ``n * H0``
  and below ``n * (H0 + 1)`` for the zero-order entropy H0. The code is
  deterministic: the two lightest nodes are merged first, a tie in weight is
  broken by symbol order (the node whose first symbol in alphabet order comes
  earlier is merged first), and of the two nodes merged the one taken first
  becomes the left (0) child. A symbol of the alphabet that does not occur
  gets no code word, because one would only lengthen the others: it ranks to
  0 and cannot be selected. The leaves are not in alphabet order, so
  ``range_quantile`` raises ``ValueError`` on this shape.

Conventions (spec 2.3)
----------------------
* A sequence is a ``str`` or a sequence of single-character strings, and
  :meth:`WaveletTree.decode` returns a ``str``.
* ``alphabet`` defaults to the distinct symbols of the sequence in sorted
  order. An explicit alphabet, such as ``"$ACGT"`` for a BWT or ``"ACGT"``
  for DNA, is used in the order given, and that order is the alphabet order
  of ``range_count_values`` and ``range_quantile``. Its symbols must be
  distinct, they may have zero occurrences, and every symbol of the sequence
  must be one of them.
* Positions are 0-based and ranges are half-open, ``[l, r)`` with
  ``0 <= l <= r <= n``. ``rank(c, i)`` counts c in ``[0, i)`` for
  ``0 <= i <= n`` and is 0 for a symbol outside the alphabet.
  ``select(c, k)`` is 1-based like ``select1``, so
  ``rank(c, select(c, k)) == k - 1``. ``range_quantile(l, r, k)`` returns
  the k-th smallest symbol of ``[l, r)`` in alphabet order, k 1-based.
* An index or a rank out of range raises ``IndexError``, ``l > r`` raises
  ``ValueError``, and an index that is not an integer raises ``TypeError``.

Space accounting
----------------
:meth:`WaveletTree.size_in_bits` is the payload, counted as **the length in
bits of every node bitvector, not their padded 64-bit words** (a node stores
at most 63 padding bits more than its length). :meth:`WaveletTree.overhead_bits`
adds up the nodes' ``RankSelect.overhead_bits()``, their rank directories,
which cost 7.8125% of each node's length up to rounding; the select samples
are reported separately by :meth:`WaveletTree.select_sample_bits`.
:meth:`WaveletTree.compression_ratio` is
``(size_in_bits + overhead_bits) / (8 n)``, against one byte per symbol, and
:func:`zero_order_entropy` gives H0 in bits per symbol for comparison.

What numpy does here (spec 1.1)
-------------------------------
The build turns the sequence into a numpy array of alphabet indices and
partitions it with boolean masks, one node at a time, so 10^6 symbols are
indexed in well under a second. The default alphabet is read off a table
of the code points present (``np.flatnonzero``). For the Huffman shape numpy counts the
symbols once (``np.bincount``), and the standard-library ``heapq`` is the
min-priority queue of the textbook Huffman construction; both run only at
build time. :meth:`WaveletTree.decode` runs the partition backwards with the
same masks, read out of the stored words. Every query (``access``, ``rank``,
``select``, the range queries and ``frequencies``) goes through the nodes'
``RankSelect`` and nothing else: no ``bisect``, no ``str.count``, no
``sorted`` and no numpy counting or searching call.

Examples:
    >>> wt = WaveletTree("ACGTACGTTGCA")
    >>> wt.access(3), wt.rank("C", 12), wt.select("G", 2)
    ('T', 3, 6)
    >>> wt.range_count("A", 2, 10), [wt.range_quantile(3, 9, k) for k in (1, 3, 6)]
    (1, ['A', 'G', 'T'])
    >>> wt.decode(), wt.size_in_bits()
    ('ACGTACGTTGCA', 24)
    >>> WaveletTree("abracadabra", shape="huffman").codes()
    {'a': '0', 'b': '110', 'c': '1110', 'd': '1111', 'r': '10'}

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.advds.succinct.bit_vector import BitVector, _as_int
from src.advds.succinct.rank_select import RankSelect

__all__ = ["SHAPES", "WaveletTree", "zero_order_entropy"]

#: The two tree shapes :class:`WaveletTree` accepts.
SHAPES: Tuple[str, ...] = ("balanced", "huffman")

#: Shift amounts that read the 8 bits of a byte, least significant first.
_BYTE_SHIFTS: np.ndarray = np.arange(8, dtype=np.uint8)

#: A tree shape: an alphabet index (a leaf) or a pair (left, right) of shapes.
_Shape = Any


# ---------------------------------------------------------------------------
# Symbols and code points
# ---------------------------------------------------------------------------


def _to_text(seq: Any, name: str) -> str:
    """Return a ``str`` or a sequence of single-character strings as one ``str``.

    Args:
        seq: A ``str``, or an iterable whose items are ``str`` of length 1
            (a list, a tuple, a numpy array of ``str_``, ...).
        name: The argument name, for error messages.

    Returns:
        The symbols joined into one string.

    Raises:
        TypeError: If ``seq`` is ``bytes`` or not iterable, or an item is
            not a ``str``.
        ValueError: If an item is a string of length other than 1.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _to_text("ACGT", "seq"), _to_text(["A", "C"], "seq")
        ('ACGT', 'AC')
        >>> _to_text(["A", "CG"], "seq")
        Traceback (most recent call last):
            ...
        ValueError: seq[1] must be a single character, got 'CG'
        >>> _to_text([1, 2], "seq")
        Traceback (most recent call last):
            ...
        TypeError: seq[0] must be a single-character str, got int
    """
    if isinstance(seq, str):
        return seq
    expected = f"{name} must be a str or a sequence of single-character strings"
    if isinstance(seq, (bytes, bytearray)):
        raise TypeError(f"{expected}, got {type(seq).__name__}")
    try:
        items = list(seq)
    except TypeError:
        raise TypeError(f"{expected}, got {type(seq).__name__}") from None
    for pos, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(
                f"{name}[{pos}] must be a single-character str, got {type(item).__name__}"
            )
        if len(item) != 1:
            raise ValueError(f"{name}[{pos}] must be a single character, got {item!r}")
    return "".join(items)


def _code_points(text: str) -> np.ndarray:
    """Return the Unicode code point of every character as an ``int64`` array.

    The string is encoded as UTF-32 (four bytes per character, little-endian)
    and read back as unsigned 32-bit integers, which turns a million
    characters into an array in one vectorised step. ``surrogatepass`` lets a
    lone surrogate through unchanged, so every Python ``str`` round-trips.

    Args:
        text: Any string.

    Returns:
        An ``int64`` array of ``len(text)`` code points.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _code_points("A$a").tolist()
        [65, 36, 97]
        >>> _code_points("").shape
        (0,)
    """
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype="<u4").astype(np.int64)


def _from_code_points(points: np.ndarray) -> str:
    """Return the string whose characters have the given code points.

    The inverse of :func:`_code_points`.

    Args:
        points: An integer array of code points.

    Returns:
        The decoded string.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _from_code_points(np.array([65, 36, 97]))
        'A$a'
        >>> _from_code_points(np.zeros(0, dtype=np.int64))
        ''
    """
    return np.asarray(points).astype("<u4").tobytes().decode("utf-32-le", "surrogatepass")


def _unpack_bits(bv: BitVector) -> np.ndarray:
    """Read a bit vector's stored words back into a boolean array.

    Byte j of the little-endian word storage holds bits ``8 j .. 8 j + 7``,
    least significant first (the :class:`BitVector` layout), so shifting
    every byte by 0 .. 7 and keeping the low bit lists the bits in index
    order. Used only by :meth:`WaveletTree.decode`, which needs every bit of
    a node at once; no bit is counted here.

    Args:
        bv: The bit vector to read.

    Returns:
        A ``bool`` array of ``len(bv)`` entries, entry i equal to ``bv[i]``.

    Time Complexity:
        O(n), vectorised.

    Space Complexity:
        O(n) bytes.

    Examples:
        >>> _unpack_bits(BitVector.from_string("1011001")).tolist()
        [True, False, True, True, False, False, True]
        >>> _unpack_bits(BitVector(0)).shape
        (0,)
    """
    words = bv.words
    if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
        words = words.astype("<u8")
    byte_view = words.view(np.uint8)
    bits = (byte_view[:, None] >> _BYTE_SHIFTS) & 1
    return bits.reshape(-1)[: len(bv)].astype(bool)


# ---------------------------------------------------------------------------
# Tree shapes
# ---------------------------------------------------------------------------


def _balanced_shape(lo: int, hi: int) -> _Shape:
    """Return the balanced shape over the alphabet indices ``[lo, hi)``.

    A node with m symbols gives its left child the first ``ceil(m / 2)`` of
    them and its right child the rest.

    Args:
        lo: The first alphabet index (inclusive).
        hi: One past the last alphabet index; ``hi > lo``.

    Returns:
        ``lo`` when one symbol is left, otherwise ``(left, right)``.

    Time Complexity:
        O(hi - lo).

    Space Complexity:
        O(hi - lo) for the shape, O(log(hi - lo)) recursion depth.

    Examples:
        >>> _balanced_shape(0, 4)
        ((0, 1), (2, 3))
        >>> _balanced_shape(0, 5)
        (((0, 1), 2), (3, 4))
        >>> _balanced_shape(3, 4)
        3
    """
    if hi - lo == 1:
        return lo
    mid = lo + (hi - lo + 1) // 2
    return (_balanced_shape(lo, mid), _balanced_shape(mid, hi))


def _huffman_shape(counts: List[int]) -> Optional[_Shape]:
    """Return the shape of a deterministic Huffman code for the given counts.

    Every symbol with a positive count starts as a leaf keyed by
    ``(count, symbol)``. The two smallest keys are taken from the priority
    queue and merged into a node keyed by ``(sum of counts, smaller of their
    first symbols)``; the node taken first becomes the left (0) child. Since
    the merged nodes cover disjoint sets of symbols, no two keys in the queue
    are ever equal, so the result does not depend on how the queue breaks
    ties, and equal weights are ordered by symbol order.

    Args:
        counts: ``counts[c]`` = occurrences of alphabet index c.

    Returns:
        The shape, a leaf index when one symbol occurs, or ``None`` when no
        symbol occurs.

    Time Complexity:
        O(sigma log sigma).

    Space Complexity:
        O(sigma).

    Examples:
        >>> _huffman_shape([5, 2, 1, 1, 2])
        (0, (4, (1, (2, 3))))
        >>> _huffman_shape([1, 1, 1, 1])
        ((0, 1), (2, 3))
        >>> _huffman_shape([0, 7, 0]), _huffman_shape([0, 0])
        (1, None)
    """
    heap = [(count, c, c) for c, count in enumerate(counts) if count > 0]
    if not heap:
        return None
    heapq.heapify(heap)
    while len(heap) > 1:
        weight_a, first_a, shape_a = heapq.heappop(heap)
        weight_b, first_b, shape_b = heapq.heappop(heap)
        first = first_a if first_a < first_b else first_b
        heapq.heappush(heap, (weight_a + weight_b, first, (shape_a, shape_b)))
    return heap[0][2]


def _shape_symbols(shape: _Shape) -> List[int]:
    """Return the alphabet indices of the leaves of a shape, left to right.

    Args:
        shape: A shape from :func:`_balanced_shape` or :func:`_huffman_shape`.

    Returns:
        The leaf indices in left-to-right order.

    Time Complexity:
        O(number of nodes of the shape).

    Space Complexity:
        O(number of nodes of the shape).

    Examples:
        >>> _shape_symbols((0, (4, (1, (2, 3)))))
        [0, 4, 1, 2, 3]
        >>> _shape_symbols(2)
        [2]
    """
    out: List[int] = []
    stack = [shape]
    while stack:
        node = stack.pop()
        if isinstance(node, int):
            out.append(node)
        else:
            stack.append(node[1])
            stack.append(node[0])
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class _Node:
    """One node of a wavelet tree.

    A leaf has ``symbol`` = its alphabet index and no bits. An internal node
    has ``symbol = -1``, a :class:`RankSelect` over its bits (0 = left,
    1 = right) and two children. ``first`` and ``last`` are the smallest and
    largest alphabet index under the node, which lets range counting tell
    whether a node lies wholly inside, wholly outside or partly inside a
    value range ``[lo, hi]``.

    Attributes:
        symbol: Alphabet index of a leaf, -1 for an internal node.
        first: Smallest alphabet index under the node.
        last: Largest alphabet index under the node.
        rs: The node's rank/select structure (internal nodes only).
        left: The 0 child (internal nodes only).
        right: The 1 child (internal nodes only).

    Time Complexity:
        O(1) to create.

    Space Complexity:
        O(1) beyond the bit vector.

    Examples:
        >>> leaf = _Node(2, 2, 2)
        >>> leaf.symbol, leaf.rs is None
        (2, True)
    """

    __slots__ = ("symbol", "first", "last", "rs", "left", "right")

    def __init__(
        self,
        symbol: int,
        first: int,
        last: int,
        rs: Optional[RankSelect] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
    ) -> None:
        self.symbol = symbol
        self.first = first
        self.last = last
        self.rs = rs
        self.left = left
        self.right = right


def _build_node(shape: _Shape, codes: np.ndarray, side: np.ndarray) -> _Node:
    """Build the subtree for ``shape`` over the alphabet indices ``codes``.

    The node's bit for each position is 1 exactly when that position's
    symbol is a leaf of the right sub-shape. The bits are packed into a
    :class:`BitVector`, indexed by a :class:`RankSelect`, and the positions
    are partitioned (stably) into the two children's sequences.

    ``side`` is one ``bool`` lookup table of sigma entries shared by the
    whole build and all ``False`` between nodes. A node sets the entries of
    its right sub-shape's symbols, reads the mask through the table and
    clears them again, so it touches only its own symbols: allocating a
    fresh sigma-entry table at each of the sigma - 1 internal nodes would
    cost O(sigma^2).

    Args:
        shape: The shape of this subtree.
        codes: The alphabet indices of the symbols reaching this node, in
            sequence order.
        side: The shared all-``False`` ``bool`` table of sigma entries; it
            is all ``False`` again on return.

    Returns:
        The subtree's root node.

    Time Complexity:
        O(m h + s h) for m = ``len(codes)``, s leaves and a subtree of
        height h, as vectorised numpy passes, plus the ``RankSelect`` builds.

    Space Complexity:
        O(m) temporary arrays per level; the stored bits total the payload.

    Examples:
        >>> side = np.zeros(2, dtype=bool)
        >>> node = _build_node((0, 1), np.array([0, 1, 1, 0]), side)
        >>> node.rs.bitvector.to_string(), node.left.symbol, node.right.symbol
        ('0110', 0, 1)
        >>> bool(side.any())
        False
    """
    if isinstance(shape, int):
        return _Node(shape, shape, shape)
    left_shape, right_shape = shape
    right_symbols = _shape_symbols(right_shape)
    side[right_symbols] = True
    mask = side[codes]
    side[right_symbols] = False
    rs = RankSelect(BitVector.from_bits(mask))
    left = _build_node(left_shape, codes[~mask], side)
    right = _build_node(right_shape, codes[mask], side)
    first = left.first if left.first < right.first else right.first
    last = left.last if left.last > right.last else right.last
    return _Node(-1, first, last, rs, left, right)


def _rebuild(node: _Node, length: int) -> np.ndarray:
    """Return the alphabet indices of the sequence that reached ``node``.

    The inverse of :func:`_build_node`: a leaf repeats its symbol, and an
    internal node puts its left child's sequence at its 0 positions and its
    right child's at its 1 positions, in order.

    Args:
        node: A node of a built tree.
        length: The number of positions reaching it (its bitvector length
            for an internal node).

    Returns:
        An ``int64`` array of ``length`` alphabet indices.

    Time Complexity:
        O(length * h) for a subtree of height h, vectorised.

    Space Complexity:
        O(length) per level.

    Examples:
        >>> node = _build_node((0, 1), np.array([0, 1, 1, 0]), np.zeros(2, dtype=bool))
        >>> _rebuild(node, 4).tolist()
        [0, 1, 1, 0]
    """
    if node.symbol >= 0:
        return np.full(length, node.symbol, dtype=np.int64)
    rs = node.rs
    mask = _unpack_bits(rs.bitvector)
    out = np.empty(len(rs), dtype=np.int64)
    out[~mask] = _rebuild(node.left, rs.num_zeros)
    out[mask] = _rebuild(node.right, rs.num_ones)
    return out


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def _entropy_from_counts(counts: Iterable[int], n: int) -> float:
    """Return ``sum (c / n) log2(n / c)`` over the positive counts.

    Args:
        counts: Symbol counts that add up to n.
        n: The sequence length.

    Returns:
        H0 in bits per symbol; 0.0 when n is 0.

    Time Complexity:
        O(number of counts).

    Space Complexity:
        O(1).

    Examples:
        >>> _entropy_from_counts([2, 2], 4), _entropy_from_counts([4], 4)
        (1.0, 0.0)
        >>> _entropy_from_counts([], 0)
        0.0
    """
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            h += (c / n) * math.log2(n / c)
    return h


def zero_order_entropy(seq: Iterable[Any]) -> float:
    """Return the zero-order (empirical) entropy H0 in bits per symbol.

    ``H0 = sum over symbols c of (n_c / n) log2(n / n_c)``, the average code
    length a code that looks at one symbol at a time cannot beat. A Huffman
    code reaches between H0 and H0 + 1 bits per symbol.

    Args:
        seq: A ``str`` or any finite iterable of hashable symbols.

    Returns:
        H0 as a float, 0.0 for an empty sequence or a single repeated symbol.

    Raises:
        TypeError: If ``seq`` is not iterable or holds unhashable items.

    Time Complexity:
        O(n).

    Space Complexity:
        O(sigma) for the symbol counts.

    Examples:
        >>> zero_order_entropy("ACGT" * 5)
        2.0
        >>> zero_order_entropy("aaab") == 0.75 * math.log2(4 / 3) + 0.25 * 2
        True
        >>> zero_order_entropy(""), zero_order_entropy("zzz")
        (0.0, 0.0)
    """
    counts = Counter(seq)
    n = 0
    for c in counts.values():
        n += c
    return _entropy_from_counts(counts.values(), n)


# ---------------------------------------------------------------------------
# The wavelet tree
# ---------------------------------------------------------------------------


class WaveletTree:
    """A wavelet tree with rank, select, range counting and range quantiles.

    See the module docstring for the shapes and the conventions. The
    sequence itself is not kept: every answer, including :meth:`decode`,
    comes from the node bitvectors.

    Args:
        seq: A ``str`` or a sequence of single-character strings.
        alphabet: ``None`` for the sorted distinct symbols of ``seq``, or a
            ``str`` / sequence of distinct single characters in the alphabet
            order to use (symbols may have zero occurrences).
        shape: ``"balanced"`` (alphabet halved at every node, left half
            gets ``ceil(m / 2)`` symbols) or ``"huffman"`` (Huffman code tree
            of the symbol counts).

    Raises:
        TypeError: If ``seq`` or ``alphabet`` is not a string or a sequence
            of strings.
        ValueError: If ``shape`` is not one of :data:`SHAPES`, an item is not
            a single character, the alphabet repeats a symbol, or a symbol of
            ``seq`` is not in the alphabet.

    Time Complexity:
        O(n h + sigma log sigma) to build, for height h (``ceil(log2 sigma)``
        balanced, at most the longest code word for Huffman): one vectorised
        numpy partition per node and level, and one ``RankSelect`` build per
        node (O(m) for a node of length m).

    Space Complexity:
        The payload :meth:`size_in_bits` (about ``n * ceil(log2 sigma)``
        bits balanced, ``n * H0`` to ``n * (H0 + 1)`` bits Huffman), plus
        :meth:`overhead_bits` and :meth:`select_sample_bits`, plus O(sigma)
        nodes. The build uses O(n) temporary words.

    Examples:
        >>> wt = WaveletTree("ACCA", alphabet="$ACGT")
        >>> wt.alphabet, wt.sigma, len(wt)
        ('$ACGT', 5, 4)
        >>> wt.rank("G", 4), wt.frequencies()
        (0, {'$': 0, 'A': 2, 'C': 2, 'G': 0, 'T': 0})
        >>> WaveletTree("AXA", alphabet="ACGT")
        Traceback (most recent call last):
            ...
        ValueError: symbol 'X' at position 1 is not in the alphabet
        >>> WaveletTree("ACGT", shape="avl")
        Traceback (most recent call last):
            ...
        ValueError: shape must be 'balanced' or 'huffman', got 'avl'
    """

    __slots__ = (
        "_n",
        "_shape",
        "_alphabet",
        "_alpha_points",
        "_index",
        "_root",
        "_paths",
        "_internal",
    )

    def __init__(self, seq: Any, alphabet: Any = None, shape: str = "balanced") -> None:
        if not isinstance(shape, str) or shape not in SHAPES:
            raise ValueError(f"shape must be 'balanced' or 'huffman', got {shape!r}")
        text = _to_text(seq, "seq")
        points = _code_points(text)
        n = len(text)

        if alphabet is None:
            if n:
                present = np.zeros(int(points.max()) + 1, dtype=bool)
                present[points] = True
                alpha_points = np.flatnonzero(present).astype(np.int64)
            else:
                alpha_points = np.zeros(0, dtype=np.int64)
            alpha_text = _from_code_points(alpha_points)
        else:
            alpha_text = _to_text(alphabet, "alphabet")
            seen: Dict[str, int] = {}
            for symbol in alpha_text:
                if symbol in seen:
                    raise ValueError(f"alphabet repeats the symbol {symbol!r}")
                seen[symbol] = 1
            alpha_points = _code_points(alpha_text)
        sigma = len(alpha_text)

        # Alphabet index of every position, through a code point lookup table.
        top = -1
        if n:
            top = int(points.max())
        if sigma and int(alpha_points.max()) > top:
            top = int(alpha_points.max())
        table = np.full(top + 1, -1, dtype=np.int64)
        table[alpha_points] = np.arange(sigma, dtype=np.int64)
        codes = table[points]
        missing = codes < 0
        if bool(missing.any()):
            pos = int(np.flatnonzero(missing)[0])
            raise ValueError(f"symbol {text[pos]!r} at position {pos} is not in the alphabet")

        if shape == "balanced":
            tree_shape = _balanced_shape(0, sigma) if sigma else None
        else:
            counts = np.bincount(codes, minlength=sigma).tolist()
            tree_shape = _huffman_shape(counts)

        self._n: int = n
        self._shape: str = shape
        self._alphabet: str = alpha_text
        self._alpha_points: np.ndarray = alpha_points
        self._index: Dict[str, int] = {symbol: j for j, symbol in enumerate(alpha_text)}
        self._root: Optional[_Node] = (
            _build_node(tree_shape, codes, np.zeros(sigma, dtype=bool))
            if tree_shape is not None
            else None
        )

        # The root-to-leaf path of every symbol as (RankSelect, bit) steps,
        # and the internal nodes in preorder, for queries and accounting.
        self._paths: List[Optional[Tuple[Tuple[RankSelect, int], ...]]] = [None] * sigma
        self._internal: List[_Node] = []
        if self._root is not None:
            stack: List[Tuple[_Node, Tuple[Tuple[RankSelect, int], ...]]] = [(self._root, ())]
            while stack:
                node, path = stack.pop()
                if node.symbol >= 0:
                    self._paths[node.symbol] = path
                    continue
                self._internal.append(node)
                stack.append((node.right, path + ((node.rs, 1),)))
                stack.append((node.left, path + ((node.rs, 0),)))

    # ------------------------------------------------------------ basics

    def __len__(self) -> int:
        """Return n, the length of the sequence.

        Returns:
            The number of symbols indexed.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(WaveletTree("banana"))
            6
        """
        return self._n

    @property
    def alphabet(self) -> str:
        """The alphabet as a string, in alphabet order.

        Returns:
            The sorted distinct symbols, or the explicit alphabet as given.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("banana").alphabet
            'abn'
        """
        return self._alphabet

    @property
    def sigma(self) -> int:
        """The alphabet size, including symbols with zero occurrences.

        Returns:
            ``len(alphabet)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("banana", alphabet="abcn").sigma
            4
        """
        return len(self._alphabet)

    @property
    def shape(self) -> str:
        """The tree shape, ``"balanced"`` or ``"huffman"``.

        Returns:
            The shape given to the constructor.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("banana", shape="huffman").shape
            'huffman'
        """
        return self._shape

    def _symbol_index(self, c: Any) -> Optional[int]:
        """Return the alphabet index of ``c``, or ``None`` if it is not a symbol.

        Args:
            c: Any value.

        Returns:
            The index of ``c`` in the alphabet, or ``None``.

        Time Complexity:
            O(1) expected (one dictionary lookup).

        Space Complexity:
            O(1).

        Examples:
            >>> wt = WaveletTree("ACGT")
            >>> wt._symbol_index("G"), wt._symbol_index("N"), wt._symbol_index(3)
            (2, None, None)
        """
        if not isinstance(c, str):
            return None
        return self._index.get(c)

    def _check_position(self, i: Any) -> int:
        """Validate a position ``0 <= i < n``.

        Args:
            i: The position.

        Returns:
            ``i`` as an int.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ab")._check_position(2)
            Traceback (most recent call last):
                ...
            IndexError: position 2 out of range [0, 2)
        """
        i = _as_int(i, "i")
        if i < 0 or i >= self._n:
            raise IndexError(f"position {i} out of range [0, {self._n})")
        return i

    def _check_bound(self, i: Any, name: str) -> int:
        """Validate a prefix end or range end ``0 <= i <= n``.

        Args:
            i: The value to check.
            name: Its parameter name, for the error message.

        Returns:
            ``i`` as an int.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n]``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ab")._check_bound(2, "i")
            2
            >>> WaveletTree("ab")._check_bound(-1, "i")
            Traceback (most recent call last):
                ...
            IndexError: i = -1 out of range [0, 2]
        """
        i = _as_int(i, name)
        if i < 0 or i > self._n:
            raise IndexError(f"{name} = {i} out of range [0, {self._n}]")
        return i

    def _check_range(self, l: Any, r: Any) -> Tuple[int, int]:
        """Validate a half-open range ``[l, r)`` with ``0 <= l <= r <= n``.

        Args:
            l: The first position of the range.
            r: One past the last position of the range.

        Returns:
            ``(l, r)`` as ints.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("abc")._check_range(1, 3)
            (1, 3)
            >>> WaveletTree("abc")._check_range(2, 1)
            Traceback (most recent call last):
                ...
            ValueError: reversed range: l = 2 > r = 1
        """
        l = self._check_bound(l, "l")
        r = self._check_bound(r, "r")
        if l > r:
            raise ValueError(f"reversed range: l = {l} > r = {r}")
        return l, r

    def _rank_index(self, idx: int, i: int) -> int:
        """Return the occurrences of alphabet index ``idx`` in ``[0, i)``.

        Follows the symbol's root-to-leaf path: at a step with bit b the
        position becomes ``rank_b(position)`` in the child. A symbol with no
        leaf (a Huffman symbol that does not occur) has count 0.

        Args:
            idx: An alphabet index.
            i: A validated position ``0 <= i <= n``.

        Returns:
            The count.

        Time Complexity:
            O(h): one ``rank1`` per level of the symbol's path.

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGTACGTTGCA")._rank_index(3, 12)
            3
        """
        path = self._paths[idx]
        if path is None:
            return 0
        for rs, bit in path:
            ones = rs.rank1(i)
            i = ones if bit else i - ones
        return i

    # ----------------------------------------------------------- queries

    def access(self, i: int) -> str:
        """Return the symbol at position i.

        From the root, read the node's bit at the current position and move
        to the child it names, at position ``rank_b(position)``, until a leaf.

        Args:
            i: A position with ``0 <= i < n``.

        Returns:
            The symbol ``seq[i]``.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(h): one bit read and one ``rank1`` per level.

        Space Complexity:
            O(1).

        Examples:
            >>> wt = WaveletTree("ACGTACGTTGCA")
            >>> "".join(wt.access(i) for i in range(len(wt)))
            'ACGTACGTTGCA'
            >>> wt.access(12)
            Traceback (most recent call last):
                ...
            IndexError: position 12 out of range [0, 12)
        """
        i = self._check_position(i)
        node = self._root
        while node.symbol < 0:
            rs = node.rs
            ones = rs.rank1(i)
            if rs.access(i):
                i = ones
                node = node.right
            else:
                i = i - ones
                node = node.left
        return self._alphabet[node.symbol]

    def rank(self, c: Any, i: int) -> int:
        """Return the number of occurrences of symbol c in positions ``[0, i)``.

        Args:
            c: A symbol. A value outside the alphabet occurs 0 times.
            i: A position with ``0 <= i <= n``.

        Returns:
            The count of c in ``[0, i)``.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n]``.

        Time Complexity:
            O(h): one ``rank1`` per level of c's path.

        Space Complexity:
            O(1).

        Examples:
            >>> wt = WaveletTree("ACGTACGTTGCA")
            >>> wt.rank("C", 12), wt.rank("T", 8), wt.rank("A", 0), wt.rank("N", 12)
            (3, 2, 0, 0)
            >>> wt.rank("A", 13)
            Traceback (most recent call last):
                ...
            IndexError: i = 13 out of range [0, 12]
        """
        i = self._check_bound(i, "i")
        idx = self._symbol_index(c)
        if idx is None:
            return 0
        return self._rank_index(idx, i)

    def select(self, c: Any, k: int) -> int:
        """Return the position of the k-th occurrence of symbol c (k is 1-based).

        Walk down to c's leaf, then back up: at a step with bit b the k-th
        position in the child is ``select_b(k)`` in the parent (0-based, so
        the parent's 1-based rank is that plus 1).

        Args:
            c: A symbol.
            k: A rank with ``1 <= k <= count of c``.

        Returns:
            The position p with ``seq[p] == c`` and ``rank(c, p) == k - 1``.

        Raises:
            TypeError: If ``k`` is not an integer.
            IndexError: If ``k`` is outside ``[1, count of c]``, which
                includes every k when c does not occur or is not a symbol.

        Time Complexity:
            O(h): one ``rank1`` per level to count c, then one select per
            level (each O(1) except its bounded binary search, spec 4.2).

        Space Complexity:
            O(1).

        Examples:
            >>> wt = WaveletTree("ACGTACGTTGCA")
            >>> [wt.select("T", k) for k in (1, 2, 3)]
            [3, 7, 8]
            >>> wt.select("T", 4)
            Traceback (most recent call last):
                ...
            IndexError: select rank 4 out of range [1, 3] for symbol 'T'
        """
        k = _as_int(k, "k")
        idx = self._symbol_index(c)
        count = 0 if idx is None else self._rank_index(idx, self._n)
        if k < 1 or k > count:
            raise IndexError(f"select rank {k} out of range [1, {count}] for symbol {c!r}")
        path = self._paths[idx]
        j = k
        for step in range(len(path) - 1, -1, -1):
            rs, bit = path[step]
            j = (rs.select1(j) if bit else rs.select0(j)) + 1
        return j - 1

    def range_count(self, c: Any, l: int, r: int) -> int:
        """Return the number of occurrences of symbol c in ``[l, r)``.

        Args:
            c: A symbol. A value outside the alphabet occurs 0 times.
            l: The first position of the range.
            r: One past the last position, ``l <= r <= n``.

        Returns:
            ``rank(c, r) - rank(c, l)``.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(h): two ranks.

        Space Complexity:
            O(1).

        Examples:
            >>> wt = WaveletTree("ACGTACGTTGCA")
            >>> wt.range_count("A", 2, 10), wt.range_count("T", 3, 9), wt.range_count("G", 5, 5)
            (1, 3, 0)
        """
        l, r = self._check_range(l, r)
        idx = self._symbol_index(c)
        if idx is None:
            return 0
        return self._rank_index(idx, r) - self._rank_index(idx, l)

    def _value_bound(self, symbol: Any, name: str) -> int:
        """Return the alphabet index of a value-range bound.

        Args:
            symbol: ``lo`` or ``hi`` of :meth:`range_count_values`.
            name: Its parameter name, for the error message.

        Returns:
            The bound's alphabet index.

        Raises:
            ValueError: If the bound is not a symbol of the alphabet.

        Time Complexity:
            O(1) expected.

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGT")._value_bound("C", "lo")
            1
            >>> WaveletTree("ACGT")._value_bound("N", "lo")
            Traceback (most recent call last):
                ...
            ValueError: lo = 'N' is not a symbol of the alphabet 'ACGT'
        """
        idx = self._symbol_index(symbol)
        if idx is None:
            raise ValueError(f"{name} = {symbol!r} is not a symbol of the alphabet {self._alphabet!r}")
        return idx

    def range_count_values(self, l: int, r: int, lo: Any, hi: Any) -> int:
        """Count the positions in ``[l, r)`` whose symbol s has ``lo <= s <= hi``.

        "Range counting": the comparison is in alphabet order and both bounds
        are inclusive. From the root, a node whose symbols all lie in
        ``[lo, hi]`` contributes the whole length of its current range, a
        node with none of them contributes 0, and any other node passes its
        range to both children with two ``rank1`` calls.

        Args:
            l: The first position of the range.
            r: One past the last position, ``l <= r <= n``.
            lo: The smallest symbol counted, in alphabet order.
            hi: The largest symbol counted; ``lo`` after ``hi`` gives 0.

        Returns:
            The number of positions p in ``[l, r)`` with ``lo <= seq[p] <= hi``.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r``, or ``lo`` or ``hi`` is not a symbol of
                the alphabet.

        Time Complexity:
            Balanced: O(log sigma), since at most two nodes per level are
            partly inside ``[lo, hi]``, each costing two ``rank1`` calls.
            Huffman: two ``rank1`` calls per partly covered node, at most
            O(sigma) nodes (the leaves are not in alphabet order).

        Space Complexity:
            O(h) for the stack of pending nodes.

        Examples:
            >>> wt = WaveletTree("ACGTACGTTGCA")
            >>> wt.range_count_values(0, 12, "C", "G"), wt.range_count_values(3, 9, "A", "C")
            (6, 2)
            >>> wt.range_count_values(0, 12, "T", "A")
            0
            >>> WaveletTree("ACGTACGTTGCA", shape="huffman").range_count_values(3, 9, "A", "G")
            3
        """
        l, r = self._check_range(l, r)
        a = self._value_bound(lo, "lo")
        b = self._value_bound(hi, "hi")
        if a > b or l == r or self._root is None:
            return 0
        total = 0
        stack = [(self._root, l, r)]
        while stack:
            node, l, r = stack.pop()
            if l == r or node.last < a or node.first > b:
                continue
            if a <= node.first and node.last <= b:
                total += r - l
                continue
            rs = node.rs  # partly covered, so not a leaf (a leaf has first == last)
            ones_l = rs.rank1(l)
            ones_r = rs.rank1(r)
            stack.append((node.left, l - ones_l, r - ones_r))
            stack.append((node.right, ones_l, ones_r))
        return total

    def range_quantile(self, l: int, r: int, k: int) -> str:
        """Return the k-th smallest symbol of ``[l, r)`` in alphabet order (k 1-based).

        At each node the number of 0 bits in the current range is the number
        of symbols from the left (smaller) half: if k is at most that, go
        left, otherwise subtract it from k and go right. Needs the leaves in
        alphabet order, so only the balanced shape supports it.

        Args:
            l: The first position of the range.
            r: One past the last position, ``l <= r <= n``.
            k: A rank with ``1 <= k <= r - l``.

        Returns:
            The k-th smallest symbol of ``seq[l:r]``: k = 1 gives the
            minimum and ``k = r - l`` the maximum.

        Raises:
            ValueError: If the tree is Huffman-shaped, or ``l > r``.
            TypeError: If ``l``, ``r`` or ``k`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``, or ``k`` is
                outside ``[1, r - l]``.

        Time Complexity:
            O(log sigma): two ``rank1`` calls per level.

        Space Complexity:
            O(1).

        Examples:
            >>> wt = WaveletTree("ACGTACGTTGCA")
            >>> [wt.range_quantile(3, 9, k) for k in range(1, 7)]
            ['A', 'C', 'G', 'T', 'T', 'T']
            >>> wt.range_quantile(3, 9, 7)
            Traceback (most recent call last):
                ...
            IndexError: quantile rank 7 out of range [1, 6]
            >>> WaveletTree("ACGT", shape="huffman").range_quantile(0, 4, 1)
            Traceback (most recent call last):
                ...
            ValueError: range_quantile needs the balanced shape (leaves in alphabet order), this tree is huffman
        """
        if self._shape != "balanced":
            raise ValueError(
                "range_quantile needs the balanced shape (leaves in alphabet order), "
                f"this tree is {self._shape}"
            )
        l, r = self._check_range(l, r)
        k = _as_int(k, "k")
        if k < 1 or k > r - l:
            raise IndexError(f"quantile rank {k} out of range [1, {r - l}]")
        node = self._root
        while node.symbol < 0:
            rs = node.rs
            ones_l = rs.rank1(l)
            ones_r = rs.rank1(r)
            zeros = (r - l) - (ones_r - ones_l)
            if k <= zeros:
                node = node.left
                l = l - ones_l
                r = r - ones_r
            else:
                k -= zeros
                node = node.right
                l = ones_l
                r = ones_r
        return self._alphabet[node.symbol]

    def frequencies(self) -> Dict[str, int]:
        """Return the number of occurrences of every alphabet symbol.

        Each count is ``rank(c, n)``, read from the nodes' rank directories.

        Returns:
            A dict from symbol to count, in alphabet order, with 0 for a
            symbol of the alphabet that does not occur.

        Time Complexity:
            O(sigma h).

        Space Complexity:
            O(sigma).

        Examples:
            >>> WaveletTree("abracadabra").frequencies()
            {'a': 5, 'b': 2, 'c': 1, 'd': 1, 'r': 2}
            >>> WaveletTree("", alphabet="AC").frequencies()
            {'A': 0, 'C': 0}
        """
        n = self._n
        return {symbol: self._rank_index(j, n) for j, symbol in enumerate(self._alphabet)}

    def decode(self) -> str:
        """Rebuild the whole sequence from the node bitvectors.

        Bottom up, each internal node interleaves its children's sequences:
        its left child's symbols go to its 0 positions and its right child's
        to its 1 positions, in order. This is the build's partition run
        backwards, as whole-array numpy operations on the stored bits; to
        read single positions use :meth:`access`.

        Returns:
            The original sequence as a ``str``.

        Time Complexity:
            O(n h), vectorised.

        Space Complexity:
            O(n) per level.

        Examples:
            >>> WaveletTree("mississippi", shape="huffman").decode()
            'mississippi'
            >>> WaveletTree("").decode()
            ''
        """
        if self._root is None:
            return ""
        return _from_code_points(self._alpha_points[_rebuild(self._root, self._n)])

    def codes(self) -> Dict[str, str]:
        """Return the code word (root-to-leaf bits) of every symbol with a leaf.

        Bit 0 means "left child" and 1 "right child". A symbol's depth, and
        so the bits it costs per occurrence, is ``len(code)``. A Huffman
        symbol that does not occur has no leaf and is left out; a tree with a
        single leaf gives that symbol the empty code word.

        Returns:
            A dict from symbol to code word, in alphabet order.

        Time Complexity:
            O(sigma h).

        Space Complexity:
            O(sigma h).

        Examples:
            >>> WaveletTree("$ACGT").codes()
            {'$': '000', 'A': '001', 'C': '01', 'G': '10', 'T': '11'}
            >>> WaveletTree("aaa").codes()
            {'a': ''}
        """
        out: Dict[str, str] = {}
        for j, symbol in enumerate(self._alphabet):
            path = self._paths[j]
            if path is not None:
                out[symbol] = "".join("1" if bit else "0" for _, bit in path)
        return out

    # ------------------------------------------------------------- space

    def size_in_bits(self) -> int:
        """Return the payload: the total length in bits of all node bitvectors.

        Counted as lengths, not as padded 64-bit words. This equals
        ``sum(count(c) * len(code(c)))`` over the symbols.

        Returns:
            The payload in bits.

        Time Complexity:
            O(number of nodes).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGTACGTTGCA").size_in_bits()
            24
            >>> WaveletTree("abracadabra", shape="huffman").size_in_bits()
            23
        """
        total = 0
        for node in self._internal:
            total += len(node.rs)
        return total

    def overhead_bits(self) -> int:
        """Return the bits of the nodes' rank directories.

        The sum of ``RankSelect.overhead_bits()`` over the nodes:
        ``32 * ceil(m / 2048) + 16 * ceil(m / 256)`` for a node of length m,
        about 7.8125% of the payload. Select samples are not included (see
        :meth:`select_sample_bits`).

        Returns:
            The rank directory size in bits.

        Time Complexity:
            O(number of nodes).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGT" * 1024).overhead_bits()  # 320 at the root, 160 per child
            640
        """
        total = 0
        for node in self._internal:
            total += node.rs.overhead_bits()
        return total

    def select_sample_bits(self) -> int:
        """Return the bits of the nodes' select samples, reported separately.

        The sum of ``RankSelect.select_sample_bits()`` over the nodes.

        Returns:
            The select sample size in bits.

        Time Complexity:
            O(number of nodes).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGT" * 1024).select_sample_bits()
            704
        """
        total = 0
        for node in self._internal:
            total += node.rs.select_sample_bits()
        return total

    def bits_per_symbol(self) -> float:
        """Return the payload per symbol, ``size_in_bits() / n``.

        Returns:
            Bits per symbol, without the directories.

        Raises:
            ValueError: If the sequence is empty.

        Time Complexity:
            O(number of nodes).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGT" * 3).bits_per_symbol()
            2.0
        """
        if self._n == 0:
            raise ValueError("bits per symbol is undefined for an empty sequence")
        return self.size_in_bits() / self._n

    def compression_ratio(self) -> float:
        """Return ``(size_in_bits + overhead_bits) / (8 n)``.

        The stored bits against one byte per symbol: below 1 means smaller
        than the plain text. The select samples are not included.

        Returns:
            The compression ratio.

        Raises:
            ValueError: If the sequence is empty.

        Time Complexity:
            O(number of nodes).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("ACGT" * 1024).compression_ratio()  # (8192 + 640) / (8 * 4096)
            0.26953125
        """
        if self._n == 0:
            raise ValueError("the compression ratio is undefined for an empty sequence")
        return (self.size_in_bits() + self.overhead_bits()) / (8 * self._n)

    def entropy(self) -> float:
        """Return the zero-order entropy H0 of the sequence, in bits per symbol.

        Computed from :meth:`frequencies`; the same value as
        :func:`zero_order_entropy` on the original sequence.

        Returns:
            H0, 0.0 for an empty sequence.

        Time Complexity:
            O(sigma h).

        Space Complexity:
            O(sigma).

        Examples:
            >>> WaveletTree("ACGT" * 3).entropy()
            2.0
        """
        return _entropy_from_counts(self.frequencies().values(), self._n)

    def __repr__(self) -> str:
        """Return a one-line summary.

        Returns:
            ``<WaveletTree n=..., sigma=..., shape=...>``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> WaveletTree("banana")
            <WaveletTree n=6, sigma=3, shape='balanced'>
        """
        return f"<WaveletTree n={self._n}, sigma={len(self._alphabet)}, shape={self._shape!r}>"
