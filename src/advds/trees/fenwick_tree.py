"""Fenwick (binary indexed) trees: point add with prefix sums, and range add with range sums.

The low-bit decomposition
-------------------------
Where a segment tree halves the index range, a Fenwick tree splits it by
the low bits of the index. Write ``lowbit(i) = i & -i``: in two's complement
``-i`` flips every bit of ``i`` above its lowest 1 bit, so the AND keeps
exactly that bit (``lowbit(12) = lowbit(0b1100) = 4``). The tree is a 1-based
array ``tree[1..n]`` (``tree[0]`` is unused) in which **index i covers the
``i & -i`` elements ending at i**, that is the 1-based positions
``i - lowbit(i) + 1 .. i``. In the 0-based terms of the public methods,
``tree[i]`` holds the sum of ``values[i - lowbit(i) : i]``.

* **Prefix sums walk down.** The first i elements split into the blocks
  ending at ``i``, ``i - lowbit(i)``, ... until 0. Each step clears the
  lowest set bit, so there are at most ``floor(log2 n) + 1`` steps. For
  ``i = 13 = 0b1101``: ``tree[13]`` covers position 13, ``tree[12]`` covers
  9..12 and ``tree[8]`` covers 1..8.
* **A point update walks up.** 1-based position i lies in the blocks of
  ``i``, ``i + lowbit(i)``, ... while the index is at most n; the low bit at
  least doubles at every step, so again at most ``floor(log2 n) + 1``
  entries change. For n = 16 and position 5 they are 5, 6, 8 and 16.
* **The O(n) build.** Start from ``tree[i] = values[i - 1]`` and visit
  ``i = 1 .. n`` in order, letting each node add itself to its parent
  ``i + (i & -i)``. A node's children are all smaller than it and their
  blocks tile the rest of its block, so every node is complete when it is
  reached, and each entry is read once: Theta(n) instead of the
  Theta(n log n) of n separate adds.

The tree uses n entries against the segment tree's 2 x (next power of two)
nodes, but it can only answer sums (anything with an inverse, so that a range
is a difference of two prefixes); minimum and maximum need the segment tree.

Range add and range sum: the two-tree B1/B2 trick
-------------------------------------------------
:class:`RangeFenwick` keeps the difference array ``D`` with ``D[0] =
values[0]`` and ``D[x] = values[x] - values[x - 1]``, so ``values[x] = D[0] +
... + D[x]``. Adding ``delta`` to ``[l, r)`` changes only two differences:
``D[l] += delta`` and ``D[r] -= delta`` (the second is dropped when
``r == n``). The sum of the first p elements is::

    sum_{x < p} values[x] = sum_{y < p} D[y] (p - y)
                          = p * sum_{y < p} D[y] - sum_{y < p} y D[y]

so tree ``B1`` holds ``D`` and tree ``B2`` holds ``y * D[y]``: a range add is
two point adds on each tree and ``prefix_sum(p) = p * B1.prefix_sum(p) -
B2.prefix_sum(p)``, both O(log n).

Conventions
-----------
These are the same as :mod:`src.advds.trees.segment_tree`, whose validation
helpers this module reuses:

* Indices are 0-based and ranges are half-open ``[l, r)`` with
  ``0 <= l <= r <= n``; ``prefix_sum(i)`` is the sum of ``[0, i)`` for
  ``0 <= i <= n``. Negative (Python-style) indices are not accepted.
* An index outside its allowed range raises ``IndexError``; ``l > r`` raises
  ``ValueError``; an index that is not an integer raises ``TypeError``.
* An empty range ``l == r`` is allowed: ``range_sum`` returns 0 and
  ``range_add`` changes nothing.
* Values and deltas are finite real numbers. Integers (numpy integers
  included) are kept as exact Python ints, so integer sums are exact; any
  other real number is stored as a float, and float sums carry the usual
  rounding (a ``set`` or a ``RangeFenwick`` difference can leave a residue
  near machine epsilon). ``bool`` and non-numbers raise ``TypeError``; NaN,
  an infinity and an empty input raise ``ValueError``. Infinities are
  refused because every answer here is a difference of prefix sums, and
  ``inf - inf`` is NaN even for a range that holds no infinity.

Every walk is written by hand (spec 1.1): no ``np.cumsum`` or
``itertools.accumulate`` stands in for the tree.

Examples:
    >>> ft = FenwickTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
    >>> [ft.prefix_sum(i) for i in range(11)]
    [0, 5, 7, 15, 16, 25, 28, 35, 39, 45, 45]
    >>> ft.set(4, 0)
    >>> ft.range_sum(0, 10)
    36
    >>> rf = RangeFenwick([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
    >>> rf.range_add(3, 8, 10)
    >>> rf.range_sum(0, 10), rf.range_sum(6, 10)
    (95, 37)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from src.advds.trees.segment_tree import (
    Number,
    _as_index,
    _as_number,
    _as_values,
    _check_point,
    _check_range,
)

__all__ = ["FenwickTree", "RangeFenwick"]


def _check_prefix(i: Any, n: int) -> int:
    """Validate a prefix length ``0 <= i <= n`` and return it as an ``int``.

    Args:
        i: The prefix length (the number of leading elements).
        n: The number of elements.

    Returns:
        ``i`` as an ``int``.

    Raises:
        TypeError: If ``i`` is not an integer.
        IndexError: If ``i`` is outside ``[0, n]``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_prefix(10, 10)
        10
        >>> _check_prefix(-1, 10)
        Traceback (most recent call last):
            ...
        IndexError: prefix length -1 out of range [0, 10]
    """
    i = _as_index(i, "i")
    if not 0 <= i <= n:
        raise IndexError(f"prefix length {i} out of range [0, {n}]")
    return i


class FenwickTree:
    """Fenwick tree over n values: point add, prefix sums and range sums.

    The internal array is 1-based with length n + 1; index i covers the
    ``i & -i`` elements ending at i (see the module docstring), so prefix
    sums walk down by clearing low bits and point updates walk up by adding
    them.

    Attributes:
        tree: The internal array, a ``list`` of length n + 1 with
            ``tree[0] == 0`` unused and ``tree[i] == sum(values[i - (i & -i):
            i])``. Read it, do not modify it.

    Args:
        values: A non-empty iterable of real numbers (a list, a tuple, a
            numpy array, ...). The tree keeps its own copy.

    Raises:
        TypeError: If ``values`` is not an iterable of real numbers.
        ValueError: If ``values`` is empty or holds a NaN or an infinity.

    Time Complexity:
        Build Theta(n); :meth:`add`, :meth:`prefix_sum`, :meth:`range_sum`,
        :meth:`get` and :meth:`set` O(log n).

    Space Complexity:
        Theta(n): n + 1 entries.

    Examples:
        >>> ft = FenwickTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
        >>> ft.tree[1:]
        [5, 7, 8, 16, 9, 12, 7, 39, 6, 6]
        >>> ft.range_sum(2, 7), ft.get(4), len(ft)
        (28, 9, 10)
    """

    def __init__(self, values: Iterable[Number]) -> None:
        """Build the tree in Theta(n): each node adds itself to its parent ``i + (i & -i)``.

        Args:
            values: A non-empty iterable of real numbers.

        Returns:
            None.

        Raises:
            TypeError: If ``values`` is not an iterable of real numbers.
            ValueError: If ``values`` is empty or holds a NaN or an infinity.

        Time Complexity:
            Theta(n): one addition per node.

        Space Complexity:
            Theta(n) for the internal array.

        Examples:
            >>> FenwickTree([1, 1, 1, 1]).tree
            [0, 1, 2, 1, 4]
            >>> FenwickTree([])
            Traceback (most recent call last):
                ...
            ValueError: values must not be empty
        """
        vals = _as_values(values)
        n = len(vals)
        tree: List[Number] = [0]
        tree.extend(vals)
        for i in range(1, n + 1):
            parent = i + (i & -i)
            if parent <= n:
                tree[parent] += tree[i]
        self._n: int = n
        self.tree: List[Number] = tree

    def __len__(self) -> int:
        """Return n, the number of elements.

        Returns:
            The length of the array the tree was built from.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(FenwickTree([4, 5, 6]))
            3
        """
        return self._n

    def _prefix(self, i: int) -> Number:
        """Sum of the first ``i`` elements for a validated ``0 <= i <= n``.

        Walks ``i -> i - (i & -i)`` until 0, adding ``tree[i]`` at each step:
        at most ``floor(log2 n) + 1`` steps.
        """
        tree = self.tree
        total: Number = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    def _add(self, i: int, delta: Number) -> None:
        """Add ``delta`` to 0-based element ``i`` (validated) by walking up.

        Updates 1-based ``j = i + 1``, then ``j + (j & -j)``, ... while
        ``j <= n``: every block that contains the element.
        """
        tree, n = self.tree, self._n
        j = i + 1
        while j <= n:
            tree[j] += delta
            j += j & -j

    def add(self, i: int, delta: Number) -> None:
        """Add ``delta`` to element ``i`` (point update).

        Element i is 1-based position ``j = i + 1``; the blocks that contain
        it are ``j``, ``j + (j & -j)``, ... up to n, and each gains
        ``delta``.

        Args:
            i: The element index, ``0 <= i < n``.
            delta: The amount to add, a real number.

        Returns:
            None.

        Raises:
            TypeError: If ``i`` is not an integer or ``delta`` is not a real
                number.
            IndexError: If ``i`` is outside ``[0, n)``.
            ValueError: If ``delta`` is NaN or infinite.

        Time Complexity:
            O(log n): at most ``floor(log2 n) + 1`` entries change.

        Space Complexity:
            O(1).

        Examples:
            >>> ft = FenwickTree([0] * 16)
            >>> ft.add(4, 3)
            >>> [j for j in range(1, 17) if ft.tree[j]]
            [5, 6, 8, 16]
            >>> ft.prefix_sum(4), ft.prefix_sum(5)
            (0, 3)
        """
        i = _check_point(i, self._n)
        delta = _as_number(delta, "delta")
        self._add(i, delta)

    def prefix_sum(self, i: int) -> Number:
        """Return the sum of the first ``i`` elements, the half-open range ``[0, i)``.

        Walks ``i -> i - (i & -i)``: for ``i = 13`` it adds ``tree[13]``,
        ``tree[12]`` and ``tree[8]``.

        Args:
            i: The prefix length, ``0 <= i <= n``.

        Returns:
            The sum; 0 for ``i == 0``.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n]``.

        Time Complexity:
            O(log n): at most ``floor(log2 n) + 1`` steps.

        Space Complexity:
            O(1).

        Examples:
            >>> ft = FenwickTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> ft.prefix_sum(0), ft.prefix_sum(3), ft.prefix_sum(10)
            (0, 15, 45)
            >>> ft.prefix_sum(11)
            Traceback (most recent call last):
                ...
            IndexError: prefix length 11 out of range [0, 10]
        """
        return self._prefix(_check_prefix(i, self._n))

    def range_sum(self, l: int, r: int) -> Number:
        """Return the sum of the half-open range ``[l, r)``: ``prefix_sum(r) - prefix_sum(l)``.

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
            O(log n): two prefix walks.

        Space Complexity:
            O(1).

        Examples:
            >>> ft = FenwickTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> ft.range_sum(2, 7), ft.range_sum(5, 5)
            (28, 0)
            >>> ft.range_sum(7, 2)
            Traceback (most recent call last):
                ...
            ValueError: reversed range: l = 7 > r = 2
        """
        l, r = _check_range(l, r, self._n)
        if l == r:
            return 0
        return self._prefix(r) - self._prefix(l)

    def get(self, i: int) -> Number:
        """Return the current value of element ``i``: ``range_sum(i, i + 1)``.

        Args:
            i: The element index, ``0 <= i < n``.

        Returns:
            The element's value.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(1).

        Examples:
            >>> FenwickTree([5, 2, 8]).get(1)
            2
        """
        i = _check_point(i, self._n)
        return self._prefix(i + 1) - self._prefix(i)

    def set(self, i: int, value: Number) -> None:
        """Assign ``value`` to element ``i``: ``add(i, value - current)``.

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
            O(log n): one point read and one point add.

        Space Complexity:
            O(1).

        Examples:
            >>> ft = FenwickTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> ft.set(4, 0)
            >>> ft.range_sum(0, 10), ft.get(4)
            (36, 0)
        """
        i = _check_point(i, self._n)
        value = _as_number(value, "value")
        current = self._prefix(i + 1) - self._prefix(i)
        self._add(i, value - current)


class RangeFenwick:
    """Range add and range sum with two Fenwick trees (the B1/B2 trick).

    ``b1`` is a :class:`FenwickTree` over the difference array ``D`` and
    ``b2`` one over ``y * D[y]``; the sum of the first p elements is
    ``p * b1.prefix_sum(p) - b2.prefix_sum(p)`` (derivation in the module
    docstring).

    Attributes:
        b1: The Fenwick tree over ``D[y]``.
        b2: The Fenwick tree over ``y * D[y]``.

    Args:
        values: A non-empty iterable of real numbers.

    Raises:
        TypeError: If ``values`` is not an iterable of real numbers.
        ValueError: If ``values`` is empty or holds a NaN or an infinity.

    Time Complexity:
        Build Theta(n); :meth:`range_add`, :meth:`range_sum`,
        :meth:`prefix_sum` and :meth:`get` O(log n).

    Space Complexity:
        Theta(n): two arrays of n + 1 entries.

    Examples:
        >>> rf = RangeFenwick([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
        >>> rf.range_add(3, 8, 10)
        >>> rf.range_sum(0, 10), rf.range_sum(6, 10), rf.get(4)
        (95, 37, 19)
    """

    def __init__(self, values: Iterable[Number]) -> None:
        """Build ``b1`` over the differences and ``b2`` over ``y * D[y]``, each in Theta(n).

        Args:
            values: A non-empty iterable of real numbers.

        Returns:
            None.

        Raises:
            TypeError: If ``values`` is not an iterable of real numbers.
            ValueError: If ``values`` is empty or holds a NaN or an infinity.

        Time Complexity:
            Theta(n): the differences plus two O(n) Fenwick builds.

        Space Complexity:
            Theta(n).

        Examples:
            >>> rf = RangeFenwick([5, 2, 8])
            >>> rf.b1.tree, rf.b2.tree
            ([0, 5, 2, 6], [0, 0, -3, 12])
        """
        vals = _as_values(values)
        n = len(vals)
        diffs: List[Number] = [vals[0]]
        for x in range(1, n):
            diffs.append(vals[x] - vals[x - 1])
        self._n: int = n
        self.b1: FenwickTree = FenwickTree(diffs)
        self.b2: FenwickTree = FenwickTree([y * diffs[y] for y in range(n)])

    def __len__(self) -> int:
        """Return n, the number of elements.

        Returns:
            The length of the array the structure was built from.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(RangeFenwick([1, 2]))
            2
        """
        return self._n

    def _prefix(self, p: int) -> Number:
        """Sum of the first ``p`` elements for a validated ``0 <= p <= n``.

        ``p * B1.prefix(p) - B2.prefix(p)``: two O(log n) walks.
        """
        return p * self.b1._prefix(p) - self.b2._prefix(p)

    def range_add(self, l: int, r: int, delta: Number) -> None:
        """Add ``delta`` to every element of the half-open range ``[l, r)``.

        ``D[l]`` gains ``delta`` and ``D[r]`` loses it, so ``b1`` gets point
        adds ``+delta`` at l and ``-delta`` at r, and ``b2`` gets
        ``+delta * l`` at l and ``-delta * r`` at r. The two adds at r are
        skipped when ``r == n``, since they would only affect prefixes past
        the end.

        Args:
            l: First index, ``0 <= l <= n``.
            r: One past the last index, ``l <= r <= n``.
            delta: The amount to add, a real number.

        Returns:
            None. An empty range ``l == r`` changes nothing.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer, or ``delta`` is
                not a real number.
            IndexError: If ``l`` or ``r`` is outside ``[0, n]``.
            ValueError: If ``l > r`` or ``delta`` is NaN or infinite.

        Time Complexity:
            O(log n): at most four Fenwick point adds.

        Space Complexity:
            O(1).

        Examples:
            >>> rf = RangeFenwick([0, 0, 0, 0, 0])
            >>> rf.range_add(1, 3, 2)
            >>> rf.range_add(2, 5, -1)
            >>> [rf.get(i) for i in range(5)]
            [0, 2, 1, -1, -1]
        """
        l, r = _check_range(l, r, self._n)
        delta = _as_number(delta, "delta")
        if l == r:
            return
        self.b1._add(l, delta)
        self.b2._add(l, delta * l)
        if r < self._n:
            self.b1._add(r, -delta)
            self.b2._add(r, -delta * r)

    def prefix_sum(self, p: int) -> Number:
        """Return the sum of the first ``p`` elements, the half-open range ``[0, p)``.

        Args:
            p: The prefix length, ``0 <= p <= n``.

        Returns:
            ``p * b1.prefix_sum(p) - b2.prefix_sum(p)``; 0 for ``p == 0``.

        Raises:
            TypeError: If ``p`` is not an integer.
            IndexError: If ``p`` is outside ``[0, n]``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(1).

        Examples:
            >>> rf = RangeFenwick([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> rf.prefix_sum(0), rf.prefix_sum(4), rf.prefix_sum(10)
            (0, 16, 45)
        """
        return self._prefix(_check_prefix(p, self._n))

    def range_sum(self, l: int, r: int) -> Number:
        """Return the sum of the half-open range ``[l, r)``: ``prefix_sum(r) - prefix_sum(l)``.

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
            O(log n): four Fenwick prefix walks.

        Space Complexity:
            O(1).

        Examples:
            >>> rf = RangeFenwick([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> rf.range_add(3, 8, 10)
            >>> rf.range_sum(0, 10), rf.range_sum(6, 10), rf.range_sum(2, 2)
            (95, 37, 0)
        """
        l, r = _check_range(l, r, self._n)
        if l == r:
            return 0
        return self._prefix(r) - self._prefix(l)

    def get(self, i: int) -> Number:
        """Return the current value of element ``i``: ``D[0] + ... + D[i]``.

        Args:
            i: The element index, ``0 <= i < n``.

        Returns:
            ``b1.prefix_sum(i + 1)``, the element's value.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(log n): one prefix walk.

        Space Complexity:
            O(1).

        Examples:
            >>> rf = RangeFenwick([5, 2, 8])
            >>> rf.range_add(0, 2, 1)
            >>> rf.get(0), rf.get(1), rf.get(2)
            (6, 3, 8)
        """
        i = _check_point(i, self._n)
        return self.b1._prefix(i + 1)
