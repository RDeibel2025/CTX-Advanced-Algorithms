"""Segment tree over an array: range sum, minimum and maximum, with lazy range add.

The idea
--------
A segment tree halves the index range. The root stands for the whole array
``[0, n)``; a node standing for ``[lo, hi)`` has two children standing for
``[lo, mid)`` and ``[mid, hi)`` with ``mid = (lo + hi) // 2``, down to one
leaf per element. Every node stores the sum, the minimum and the maximum of
its range. A query range ``[l, r)`` is the union of at most two nodes per
level that lie entirely inside it, so O(log n) stored answers combine into
the result.

Layout
------
The tree is a complete binary tree kept in flat Python lists, heap style:
node 1 is the root, node ``v`` has children ``2v`` and ``2v + 1``, and the
leaves are nodes ``size .. 2 size - 1``, where ``size`` (the
:attr:`SegmentTree.capacity`) is the smallest power of two that is at least
n. Leaf ``size + i`` holds element i. The ``size - n`` leaves past the end
are empty and hold the identity of each aggregate: sum 0, min ``+inf`` and
max ``-inf``, so they never change an answer. The build is bottom-up, each
internal node computed from its two children, in Theta(n). Space is
``2 * size`` slots for each of sum, min and max plus ``size`` tag slots,
about 2 x (next power of two) nodes, which is under 4n.

Lazy propagation
----------------
Adding ``delta`` to every element of ``[l, r)`` one leaf at a time would cost
Theta((r - l) log n). :meth:`SegmentTree.range_add` instead stops at the
O(log n) nodes that lie entirely inside ``[l, r)`` and updates each in O(1):
the sum gains ``delta * length``, the minimum and the maximum gain
``delta``, and the node's pending tag gains ``delta``. A pending tag on a
node means "every element under this node still needs ``+delta``": the
node's own aggregates are already correct, its descendants' are not.
Whenever an operation has to go into a node that it only partly overlaps, it
first pushes that node's tag down to the two children (applies it to them
and clears it), so the children it is about to read or change are up to
date. Queries push in the same way, and :meth:`SegmentTree.update` pushes
along the root-to-leaf path before it assigns. Tags only ever sit on nodes
inside ``[0, n)``, so the length of a tagged node is its number of real
elements and the empty leaves never receive a tag.

Costs: build Theta(n); ``query_sum``, ``query_min``, ``query_max``,
``update`` and ``range_add`` Theta(log n); space about 2 x (next power of
two) nodes.

Conventions
-----------
* Indices are 0-based and ranges are half-open ``[l, r)`` with
  ``0 <= l <= r <= n``. Negative (Python-style) indices are not accepted.
* An index outside its allowed range raises ``IndexError``; ``l > r`` raises
  ``ValueError``; an index that is not an integer raises ``TypeError``.
* An empty range ``l == r`` is allowed: ``query_sum`` returns 0,
  ``range_add`` changes nothing, and ``query_min`` / ``query_max`` raise
  ``ValueError`` because an empty range has no minimum or maximum.
* Values and deltas are finite real numbers. Integers (numpy integers
  included) are kept as exact Python ints; any other real number is stored
  as a float. ``bool`` and non-numbers raise ``TypeError``; NaN, an
  infinity and an empty input raise ``ValueError``.

Every query and update is written by hand (spec 1.1): no ``bisect``, no
``np.cumsum`` and no library minimum over a slice stands in for the tree.

Examples:
    >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
    >>> st.query_sum(2, 7), st.query_min(1, 5), st.query_max(0, 10)
    (28, 1, 9)
    >>> st.range_add(3, 8, 10)
    >>> st.query_sum(0, 10), st.query_max(2, 6), st.query_sum(6, 10)
    (95, 19, 37)
    >>> st.to_list()
    [5, 2, 8, 11, 19, 13, 17, 14, 6, 0]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Iterable, List, Tuple, Union

__all__ = ["SegmentTree"]

#: A value the tree can hold: an exact Python int or a float.
Number = Union[int, float]

_INF = math.inf


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_index(i: Any, name: str) -> int:
    """Return ``i`` as a Python ``int`` if it is an integer, else raise.

    Args:
        i: A proposed index (Python or numpy integer).
        name: The parameter name, for the error message.

    Returns:
        ``int(i)``.

    Raises:
        TypeError: If ``i`` is not an integer. ``bool`` is rejected.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _as_index(3, "i")
        3
        >>> _as_index(2.0, "i")
        Traceback (most recent call last):
            ...
        TypeError: i must be an int, got float
    """
    if type(i) is int:
        return i
    if isinstance(i, bool) or not isinstance(i, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(i).__name__}")
    return int(i)


def _as_number(x: Any, name: str) -> Number:
    """Return ``x`` as an exact ``int`` or a finite ``float``, rejecting anything else.

    Infinities are rejected because they are not real numbers and because
    they break the answers: a Fenwick tree answers ``[l, r)`` as a
    difference of two prefix sums, and ``inf - inf`` is NaN even when the
    range itself holds no infinity.

    Args:
        x: A proposed value or delta.
        name: The parameter name, for the error message.

    Returns:
        ``x`` itself for a Python ``int`` or ``float``, ``int(x)`` for any
        other integer (numpy integers included), ``float(x)`` for any other
        real number.

    Raises:
        TypeError: If ``x`` is not a real number. ``bool`` is rejected.
        ValueError: If ``x`` is NaN or infinite.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _as_number(7, "value"), _as_number(2.5, "value")
        (7, 2.5)
        >>> _as_number("7", "value")
        Traceback (most recent call last):
            ...
        TypeError: value must be a real number, got str
        >>> _as_number(float("nan"), "value")
        Traceback (most recent call last):
            ...
        ValueError: value must not be NaN
        >>> _as_number(float("-inf"), "value")
        Traceback (most recent call last):
            ...
        ValueError: value must be finite, got -inf
    """
    t = type(x)
    if t is int:
        return x
    if t is not float:
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            raise TypeError(f"{name} must be a real number, got {t.__name__}")
        x = int(x) if isinstance(x, numbers.Integral) else float(x)
    if x != x:
        raise ValueError(f"{name} must not be NaN")
    if not -_INF < x < _INF:
        raise ValueError(f"{name} must be finite, got {x}")
    return x


def _as_values(values: Any) -> List[Number]:
    """Return the input values as a new non-empty list of ints and floats.

    A numpy array is converted with its ``tolist()``, so its elements arrive
    as Python ints or floats; any other iterable is read with ``list()``.

    Args:
        values: A non-empty iterable of real numbers.

    Returns:
        A new ``list`` holding each value as :func:`_as_number` returns it.

    Raises:
        TypeError: If ``values`` is a string or not iterable, or an element
            is not a real number.
        ValueError: If ``values`` is empty or holds a NaN or an infinity.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) for the new list.

    Examples:
        >>> _as_values((3, 1.5, 2))
        [3, 1.5, 2]
        >>> _as_values([])
        Traceback (most recent call last):
            ...
        ValueError: values must not be empty
        >>> _as_values([1.0, float("inf")])
        Traceback (most recent call last):
            ...
        ValueError: each value must be finite, got inf
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"values must be an iterable of numbers, got {type(values).__name__}")
    tolist = getattr(values, "tolist", None)
    try:
        vals = tolist() if callable(tolist) else list(values)
    except TypeError:
        raise TypeError(
            f"values must be an iterable of numbers, got {type(values).__name__}"
        ) from None
    if not isinstance(vals, list):
        raise TypeError(f"values must be an iterable of numbers, got {type(values).__name__}")
    if not vals:
        raise ValueError("values must not be empty")
    for k, v in enumerate(vals):
        t = type(v)
        # Plain ints and finite floats pass as they are; ``-inf < v < inf``
        # is False for NaN and for both infinities.
        if t is not int and (t is not float or not -_INF < v < _INF):
            vals[k] = _as_number(v, "each value")
    return vals


def _check_point(i: Any, n: int) -> int:
    """Validate an element index ``0 <= i < n`` and return it as an ``int``.

    Args:
        i: The index.
        n: The number of elements.

    Returns:
        ``i`` as an ``int``.

    Raises:
        TypeError: If ``i`` is not an integer.
        IndexError: If ``i`` is outside ``[0, n)``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_point(9, 10)
        9
        >>> _check_point(10, 10)
        Traceback (most recent call last):
            ...
        IndexError: index 10 out of range [0, 10)
    """
    i = _as_index(i, "i")
    if not 0 <= i < n:
        raise IndexError(f"index {i} out of range [0, {n})")
    return i


def _check_range(l: Any, r: Any, n: int) -> Tuple[int, int]:
    """Validate a half-open range ``[l, r)`` with ``0 <= l <= r <= n``.

    Args:
        l: The first index of the range.
        r: One past the last index of the range.
        n: The number of elements.

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
        >>> _check_range(2, 7, 10)
        (2, 7)
        >>> _check_range(0, 11, 10)
        Traceback (most recent call last):
            ...
        IndexError: r = 11 out of range [0, 10]
        >>> _check_range(5, 3, 10)
        Traceback (most recent call last):
            ...
        ValueError: reversed range: l = 5 > r = 3
    """
    l = _as_index(l, "l")
    r = _as_index(r, "r")
    if not 0 <= l <= n:
        raise IndexError(f"l = {l} out of range [0, {n}]")
    if not 0 <= r <= n:
        raise IndexError(f"r = {r} out of range [0, {n}]")
    if l > r:
        raise ValueError(f"reversed range: l = {l} > r = {r}")
    return l, r


# ---------------------------------------------------------------------------
# Combining two answers (written out, no library call)
# ---------------------------------------------------------------------------


def _add2(a: Number, b: Number) -> Number:
    """Return ``a + b``, the sum combiner."""
    return a + b


def _min2(a: Number, b: Number) -> Number:
    """Return the smaller of ``a`` and ``b``, the min combiner."""
    return a if a <= b else b


def _max2(a: Number, b: Number) -> Number:
    """Return the larger of ``a`` and ``b``, the max combiner."""
    return a if a >= b else b


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


class SegmentTree:
    """Array-based segment tree with range sum, min and max and lazy range add.

    Built bottom-up over a heap-ordered array of ``2 * capacity`` nodes,
    where ``capacity`` is the next power of two at or above n. Each node
    stores the sum, the minimum and the maximum of its range; internal nodes
    also carry a pending add tag. Empty leaves past the end hold sum 0, min
    ``+inf`` and max ``-inf``. See the module docstring for how tags are
    applied and pushed down.

    Args:
        values: A non-empty iterable of real numbers (a list, a tuple, a
            numpy array, ...). The tree keeps its own copy.

    Raises:
        TypeError: If ``values`` is not an iterable of real numbers.
        ValueError: If ``values`` is empty or holds a NaN or an infinity.

    Time Complexity:
        Build Theta(n); every query, :meth:`update` and :meth:`range_add`
        Theta(log n); :meth:`to_list` Theta(n).

    Space Complexity:
        Theta(capacity), about 2 x (next power of two) nodes, under 4n.

    Examples:
        >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
        >>> len(st), st.capacity
        (10, 16)
        >>> st.query_min(5, 10), st.query_sum(3, 3)
        (0, 0)
        >>> st.update(9, 12)
        >>> st.query_max(5, 10)
        12
    """

    def __init__(self, values: Iterable[Number]) -> None:
        """Build the tree bottom-up from ``values`` in Theta(n).

        The leaves are filled first; then every internal node, from
        ``capacity - 1`` down to the root, combines its two children. All
        tags start at 0.

        Args:
            values: A non-empty iterable of real numbers.

        Returns:
            None.

        Raises:
            TypeError: If ``values`` is not an iterable of real numbers.
            ValueError: If ``values`` is empty or holds a NaN or an infinity.

        Time Complexity:
            Theta(n): ``capacity - 1 < 2n`` internal nodes, O(1) each.

        Space Complexity:
            Theta(capacity) for the four node arrays.

        Examples:
            >>> SegmentTree([4.5, -1]).to_list()
            [4.5, -1]
            >>> SegmentTree([])
            Traceback (most recent call last):
                ...
            ValueError: values must not be empty
        """
        vals = _as_values(values)
        n = len(vals)
        size = 1 << (n - 1).bit_length()  # smallest power of two >= n
        sums: List[Number] = [0] * (2 * size)
        mins: List[Number] = [_INF] * (2 * size)
        maxs: List[Number] = [-_INF] * (2 * size)
        sums[size : size + n] = vals
        mins[size : size + n] = vals
        maxs[size : size + n] = vals
        for v in range(size - 1, 0, -1):
            a, b = 2 * v, 2 * v + 1
            sums[v] = sums[a] + sums[b]
            mins[v] = mins[a] if mins[a] <= mins[b] else mins[b]
            maxs[v] = maxs[a] if maxs[a] >= maxs[b] else maxs[b]
        self._n: int = n
        self._size: int = size
        self._log: int = size.bit_length() - 1
        self._sum: List[Number] = sums
        self._min: List[Number] = mins
        self._max: List[Number] = maxs
        # Pending add tags, internal nodes 1 .. size - 1 only (slot 0 unused).
        self._tag: List[Number] = [0] * size

    # ------------------------------------------------------------- basics

    def __len__(self) -> int:
        """Return n, the number of elements.

        Returns:
            The length of the array the tree was built from.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(SegmentTree([1, 2, 3]))
            3
        """
        return self._n

    @property
    def capacity(self) -> int:
        """The number of leaf slots: the smallest power of two that is at least n.

        Returns:
            The capacity; the tree has ``2 * capacity - 1`` nodes (node 0 of
            the arrays is unused).

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> [SegmentTree([0] * n).capacity for n in (1, 2, 3, 10, 16, 17)]
            [1, 2, 4, 16, 16, 32]
        """
        return self._size

    # ------------------------------------------------------ node operations

    def _apply(self, node: int, delta: Number, length: int) -> None:
        """Add ``delta`` to all ``length`` elements under ``node``, in O(1).

        The node's aggregates are updated at once (sum by ``delta * length``,
        min and max by ``delta``); an internal node also records the add in
        its tag so its children can receive it later.
        """
        self._sum[node] += delta * length
        self._min[node] += delta
        self._max[node] += delta
        if node < self._size:
            self._tag[node] += delta

    def _push(self, node: int, length: int) -> None:
        """Hand ``node``'s pending tag to its two children and clear it, in O(1).

        ``length`` is the number of elements under ``node``; each child has
        half of them. Called before going into a partly overlapped node.
        """
        delta = self._tag[node]
        if delta:
            half = length >> 1
            self._apply(2 * node, delta, half)
            self._apply(2 * node + 1, delta, half)
            self._tag[node] = 0

    def _pull(self, node: int) -> None:
        """Recompute ``node``'s aggregates from its children, in O(1).

        Only called on a node whose tag is 0 (it was just pushed).
        """
        a, b = 2 * node, 2 * node + 1
        s, lo, hi = self._sum, self._min, self._max
        s[node] = s[a] + s[b]
        lo[node] = lo[a] if lo[a] <= lo[b] else lo[b]
        hi[node] = hi[a] if hi[a] >= hi[b] else hi[b]

    def _descend(
        self,
        node: int,
        lo: int,
        hi: int,
        l: int,
        r: int,
        agg: List[Number],
        combine: Callable[[Number, Number], Number],
        identity: Number,
    ) -> Number:
        """Combine ``agg`` over ``[l, r)`` within ``node``'s range ``[lo, hi)``.

        Disjoint nodes give the identity, fully covered nodes give their
        stored answer, and a partly overlapped node first pushes its tag and
        then recurses into both halves. Visits O(log n) nodes.
        """
        if r <= lo or hi <= l:
            return identity
        if l <= lo and hi <= r:
            return agg[node]
        self._push(node, hi - lo)
        mid = (lo + hi) >> 1
        return combine(
            self._descend(2 * node, lo, mid, l, r, agg, combine, identity),
            self._descend(2 * node + 1, mid, hi, l, r, agg, combine, identity),
        )

    def _add(self, node: int, lo: int, hi: int, l: int, r: int, delta: Number) -> None:
        """Add ``delta`` over ``[l, r)`` within ``node``'s range ``[lo, hi)``.

        Fully covered nodes take the add in O(1) through :meth:`_apply`; a
        partly overlapped node pushes its tag, recurses, then pulls. Visits
        O(log n) nodes.
        """
        if r <= lo or hi <= l:
            return
        if l <= lo and hi <= r:
            self._apply(node, delta, hi - lo)
            return
        self._push(node, hi - lo)
        mid = (lo + hi) >> 1
        self._add(2 * node, lo, mid, l, r, delta)
        self._add(2 * node + 1, mid, hi, l, r, delta)
        self._pull(node)

    # ---------------------------------------------------------------- queries

    def query_sum(self, l: int, r: int) -> Number:
        """Return the sum of the elements in the half-open range ``[l, r)``.

        Args:
            l: First index, ``0 <= l <= n``.
            r: One past the last index, ``l <= r <= n``.

        Returns:
            The sum; 0 for the empty range ``l == r``.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r``.

        Time Complexity:
            Theta(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> st.query_sum(2, 7), st.query_sum(0, 10), st.query_sum(4, 4)
            (28, 45, 0)
        """
        l, r = _check_range(l, r, self._n)
        if l == r:
            return 0
        return self._descend(1, 0, self._size, l, r, self._sum, _add2, 0)

    def query_min(self, l: int, r: int) -> Number:
        """Return the minimum of the elements in the half-open range ``[l, r)``.

        Args:
            l: First index, ``0 <= l < n``.
            r: One past the last index, ``l < r <= n``.

        Returns:
            The minimum.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r``, or the range is empty (``l == r``),
                which has no minimum.

        Time Complexity:
            Theta(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> st.query_min(1, 5), st.query_min(5, 10)
            (1, 0)
            >>> st.query_min(3, 3)
            Traceback (most recent call last):
                ...
            ValueError: empty range [3, 3) has no minimum
        """
        l, r = _check_range(l, r, self._n)
        if l == r:
            raise ValueError(f"empty range [{l}, {r}) has no minimum")
        return self._descend(1, 0, self._size, l, r, self._min, _min2, _INF)

    def query_max(self, l: int, r: int) -> Number:
        """Return the maximum of the elements in the half-open range ``[l, r)``.

        Args:
            l: First index, ``0 <= l < n``.
            r: One past the last index, ``l < r <= n``.

        Returns:
            The maximum.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r``, or the range is empty (``l == r``),
                which has no maximum.

        Time Complexity:
            Theta(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> st.query_max(0, 10), st.query_max(5, 7)
            (9, 7)
            >>> st.query_max(0, 11)
            Traceback (most recent call last):
                ...
            IndexError: r = 11 out of range [0, 10]
        """
        l, r = _check_range(l, r, self._n)
        if l == r:
            raise ValueError(f"empty range [{l}, {r}) has no maximum")
        return self._descend(1, 0, self._size, l, r, self._max, _max2, -_INF)

    # ---------------------------------------------------------------- updates

    def update(self, i: int, value: Number) -> None:
        """Assign ``value`` to element ``i`` (point assign).

        The node ``leaf >> s`` on the root-to-leaf path covers ``2**s``
        elements. The tags on that path are pushed down from the root, so no
        pending add is left above the leaf; then the leaf is assigned and
        its ancestors are recomputed from the bottom up.

        Args:
            i: The element index, ``0 <= i < n``.
            value: The new value, a real number.

        Returns:
            None.

        Raises:
            TypeError: If ``i`` is not an integer or ``value`` is not a real
                number.
            IndexError: If ``i`` is outside ``[0, n)``.
            ValueError: If ``value`` is NaN or infinite.

        Time Complexity:
            Theta(log n): one push and one pull per level.

        Space Complexity:
            O(1).

        Examples:
            >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> st.range_add(0, 10, 5)
            >>> st.update(4, 0)
            >>> st.to_list()
            [10, 7, 13, 6, 0, 8, 12, 9, 11, 5]
            >>> st.query_sum(0, 10), st.query_min(0, 10), st.query_max(0, 10)
            (81, 0, 13)
        """
        i = _check_point(i, self._n)
        value = _as_number(value, "value")
        leaf = self._size + i
        for s in range(self._log, 0, -1):
            self._push(leaf >> s, 1 << s)
        self._sum[leaf] = value
        self._min[leaf] = value
        self._max[leaf] = value
        for s in range(1, self._log + 1):
            self._pull(leaf >> s)

    def range_add(self, l: int, r: int, delta: Number) -> None:
        """Add ``delta`` to every element of the half-open range ``[l, r)``, lazily.

        Only the O(log n) nodes that lie entirely inside the range are
        changed (aggregates and tag, in O(1) each); tags are pushed down on
        the way into partly overlapped nodes, which are then recomputed.

        Args:
            l: First index, ``0 <= l <= n``.
            r: One past the last index, ``l <= r <= n``.
            delta: The amount to add, a real number (negative to subtract).

        Returns:
            None. An empty range ``l == r`` changes nothing.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer, or ``delta`` is
                not a real number.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r`` or ``delta`` is NaN or infinite.

        Time Complexity:
            Theta(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> st = SegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> st.range_add(3, 8, 10)
            >>> st.query_sum(0, 10), st.query_min(0, 10), st.query_max(2, 6)
            (95, 0, 19)
            >>> st.range_add(4, 2, 1)
            Traceback (most recent call last):
                ...
            ValueError: reversed range: l = 4 > r = 2
        """
        l, r = _check_range(l, r, self._n)
        delta = _as_number(delta, "delta")
        if l == r:
            return
        self._add(1, 0, self._size, l, r, delta)

    def to_list(self) -> List[Number]:
        """Return the current values as a new list, pushing every pending tag.

        Internal nodes are visited in heap order (parents before children),
        each pushing its tag to its children, so afterwards no tag is
        pending and the leaves hold the current values. The answers of later
        queries are unchanged; only where the pending adds are stored moves.

        Returns:
            A new ``list`` of the n current values.

        Time Complexity:
            Theta(n): ``capacity - 1 < 2n`` pushes.

        Space Complexity:
            Theta(n) for the returned list.

        Examples:
            >>> st = SegmentTree([1, 2, 3])
            >>> st.range_add(0, 2, -4)
            >>> st.to_list()
            [-3, -2, 3]
        """
        size = self._size
        for v in range(1, size):
            self._push(v, size >> (v.bit_length() - 1))
        return self._sum[size : size + self._n]
