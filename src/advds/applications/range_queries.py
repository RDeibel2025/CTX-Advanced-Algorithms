"""Range-query engine: range sum, minimum and maximum over an array, with time travel.

The idea
--------
:class:`RangeQueryEngine` puts the three Week 13 range-query trees behind one
application-facing API and lets each one answer the question it is best at:

* **Sums** come from a :class:`~src.advds.trees.fenwick_tree.RangeFenwick`,
  two Fenwick trees over the difference array (the B1/B2 trick): n + 1
  entries each, range add and range sum in O(log n).
* **Minimum and maximum** come from a
  :class:`~src.advds.trees.segment_tree.SegmentTree` with lazy range add. A
  Fenwick tree cannot answer them, because a minimum has no inverse, so a
  range minimum is not a difference of two prefix minima.
* **Past states** come from a
  :class:`~src.advds.trees.persistent_tree.PersistentSegmentTree`. Every
  update makes a new version by path copying (O(log n) new nodes), so
  :meth:`RangeQueryEngine.sum_at`, :meth:`RangeQueryEngine.min_at` and
  :meth:`RangeQueryEngine.max_at` answer the same questions about any
  earlier version ("time travel").

Every update (:meth:`RangeQueryEngine.set`, :meth:`RangeQueryEngine.add` and
:meth:`RangeQueryEngine.range_add`) is applied to all three structures, so
they always describe the same array, and each update makes exactly one new
persistent version. Version 0 is the initial array, version v is the array
after the v-th update, and :meth:`RangeQueryEngine.current_version` is the
number of updates made so far.

Conventions (spec 2.3)
----------------------
* Positions are 0-based.
* **Ranges are INCLUSIVE here**, ``[l, r]`` with ``0 <= l <= r <= n - 1``,
  because the instructions phrase application queries that way. The core
  trees use half-open ranges ``[l, r)``; :func:`inclusive_to_half_open`
  converts once, at the boundary, to ``[l, r + 1)``. Every inclusive method
  says "inclusive" in its docstring.
* An index outside ``[0, n - 1]`` raises ``IndexError``, ``l > r`` raises
  ``ValueError`` (an inclusive range is never empty), and an index that is
  not an integer raises ``TypeError``. A version that does not exist raises
  ``IndexError``.
* Values and deltas are finite real numbers, as in the core trees: integers
  stay exact Python ints, anything else is stored as a float. Every argument
  is checked before any structure changes, so a rejected update leaves all
  three structures (and the version count) as they were.

Costs: the build is Theta(n) for each of the three structures. Every update
is O(log n) in each structure and adds O(log n) persistent nodes (exactly
``k + 1`` for a point update when ``n = 2^k``). Every query is O(log n).

Rule for this package (spec 1.1): no library routine stands in for a query;
every answer comes from one of the three hand-written trees.

Examples:
    >>> eng = RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
    >>> eng.sum(2, 6), eng.min(1, 4), eng.max(0, 9)
    (28, 1, 9)
    >>> eng.range_add(3, 7, 10)
    1
    >>> eng.sum(0, 9), eng.max(2, 5), eng.sum_at(0, 0, 9)
    (95, 19, 45)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

from src.advds.trees.fenwick_tree import RangeFenwick
from src.advds.trees.persistent_tree import PersistentSegmentTree
from src.advds.trees.segment_tree import (
    Number,
    SegmentTree,
    _as_index,
    _as_number,
    _as_values,
)

__all__ = ["RangeQueryEngine", "inclusive_to_half_open"]


def inclusive_to_half_open(l: Any, r: Any, n: int) -> Tuple[int, int]:
    """Convert an inclusive range ``[l, r]`` to the half-open ``[l, r + 1)``.

    The application APIs (:class:`RangeQueryEngine` and
    :class:`~src.advds.applications.dna_index.DNAIndex`) take inclusive
    ranges; the core structures take half-open ones. This is the one place
    the conversion happens.

    Args:
        l: First position of the inclusive range.
        r: Last position of the inclusive range (included).
        n: The number of elements.

    Returns:
        ``(l, r + 1)`` as Python ints.

    Raises:
        TypeError: If ``l`` or ``r`` is not an integer (``bool`` rejected).
        IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]`` (every range
            when ``n == 0``).
        ValueError: If ``l > r``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> inclusive_to_half_open(2, 6, 10)
        (2, 7)
        >>> inclusive_to_half_open(4, 4, 10)
        (4, 5)
        >>> inclusive_to_half_open(0, 10, 10)
        Traceback (most recent call last):
            ...
        IndexError: r = 10 out of range [0, 9] (inclusive)
        >>> inclusive_to_half_open(5, 3, 10)
        Traceback (most recent call last):
            ...
        ValueError: reversed range: l = 5 > r = 3
    """
    l = _as_index(l, "l")
    r = _as_index(r, "r")
    if n <= 0:
        raise IndexError("an empty sequence has no inclusive range")
    if not 0 <= l < n:
        raise IndexError(f"l = {l} out of range [0, {n - 1}] (inclusive)")
    if not 0 <= r < n:
        raise IndexError(f"r = {r} out of range [0, {n - 1}] (inclusive)")
    if l > r:
        raise ValueError(f"reversed range: l = {l} > r = {r}")
    return l, r + 1


class RangeQueryEngine:
    """Range sum, min and max with point and range updates and time travel (inclusive ranges).

    Builds a :class:`SegmentTree` (min, max), a :class:`RangeFenwick` (sum)
    and a :class:`PersistentSegmentTree` (version 0, for time travel) over
    the same values. Every update goes to all three and makes one new
    version. All ranges are INCLUSIVE ``[l, r]``.

    Args:
        values: A non-empty iterable of finite real numbers (a list, a
            tuple, a numpy array, ...). The engine keeps its own copies.

    Raises:
        TypeError: If ``values`` is not an iterable of real numbers.
        ValueError: If ``values`` is empty or holds a NaN or an infinity.

    Time Complexity:
        Build Theta(n). Every update and query O(log n).

    Space Complexity:
        Theta(n) for the three builds (about 2 x next power of two segment
        nodes, 2(n + 1) Fenwick entries, 2n - 1 persistent nodes), plus
        O(log n) persistent nodes per update.

    Examples:
        >>> eng = RangeQueryEngine([4, 1, 3])
        >>> len(eng), eng.current_version()
        (3, 0)
        >>> eng.set(1, 7), eng.add(2, -3)
        (1, 2)
        >>> eng.sum(0, 2), eng.min(0, 2), eng.max(0, 2)
        (11, 0, 7)
        >>> [eng.sum_at(v, 0, 2) for v in range(eng.current_version() + 1)]
        [8, 14, 11]
    """

    def __init__(self, values: Iterable[Number]) -> None:
        """Build the three structures over the same values (version 0); ranges are inclusive.

        Args:
            values: A non-empty iterable of finite real numbers.

        Returns:
            None.

        Raises:
            TypeError: If ``values`` is not an iterable of real numbers.
            ValueError: If ``values`` is empty or holds a NaN or an infinity.

        Time Complexity:
            Theta(n): three linear builds.

        Space Complexity:
            Theta(n).

        Examples:
            >>> RangeQueryEngine((2.5, 1)).sum(0, 1)
            3.5
            >>> RangeQueryEngine([])
            Traceback (most recent call last):
                ...
            ValueError: values must not be empty
        """
        vals = _as_values(values)
        self._n: int = len(vals)
        self._seg: SegmentTree = SegmentTree(vals)
        self._fen: RangeFenwick = RangeFenwick(vals)
        self._pers: PersistentSegmentTree = PersistentSegmentTree(vals)
        self._version: int = 0

    def __len__(self) -> int:
        """Return n, the number of elements; valid inclusive positions are 0 .. n - 1.

        Returns:
            The length of the array the engine was built from.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(RangeQueryEngine([1, 2, 3]))
            3
        """
        return self._n

    def _point(self, i: Any) -> int:
        """Check a position against the inclusive bounds ``0 <= i <= n - 1``; return it as an int.

        Args:
            i: The position.

        Returns:
            ``i`` as a Python int.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n - 1]``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RangeQueryEngine([1, 2])._point(1)
            1
            >>> RangeQueryEngine([1, 2])._point(2)
            Traceback (most recent call last):
                ...
            IndexError: i = 2 out of range [0, 1] (inclusive)
        """
        i = _as_index(i, "i")
        if not 0 <= i < self._n:
            raise IndexError(f"i = {i} out of range [0, {self._n - 1}] (inclusive)")
        return i

    # ------------------------------------------------------------ updates

    def set(self, i: int, value: Number) -> int:
        """Assign ``value`` at position ``i`` (inclusive bounds 0 .. n - 1); return the new version.

        The segment tree assigns the leaf, the range Fenwick adds
        ``value - current`` over the one-element range, and the persistent
        tree makes a point-set version from the current one.

        Args:
            i: The position, ``0 <= i <= n - 1``.
            value: The new value, a finite real number.

        Returns:
            The id of the new version (the previous id plus one).

        Raises:
            TypeError: If ``i`` is not an integer or ``value`` is not a real
                number.
            IndexError: If ``i`` is outside ``[0, n - 1]``.
            ValueError: If ``value`` is NaN or infinite, or if
                ``value - current`` overflows to infinity (floats near the
                largest double); nothing is changed then.

        Time Complexity:
            O(log n) in each of the three structures.

        Space Complexity:
            O(log n) new persistent nodes (``k + 1`` when ``n = 2^k``).

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8])
            >>> eng.set(0, 1)
            1
            >>> eng.sum(0, 2), eng.min(0, 2), eng.sum_at(0, 0, 0)
            (11, 1, 5)
        """
        i = self._point(i)
        value = _as_number(value, "value")
        # The Fenwick delta is checked before any structure changes: with
        # floats near the largest double, value - current can overflow to
        # infinity, which the range Fenwick rejects. Checking it only inside
        # the Fenwick call would leave the segment tree already updated.
        delta = _as_number(value - self._fen.get(i), "value - current value")
        self._seg.update(i, value)
        self._fen.range_add(i, i + 1, delta)
        self._version = self._pers.point_set(self._version, i, value)
        return self._version

    def add(self, i: int, delta: Number) -> int:
        """Add ``delta`` at position ``i`` (inclusive bounds 0 .. n - 1); return the new version.

        Args:
            i: The position, ``0 <= i <= n - 1``.
            delta: The amount to add, a finite real number.

        Returns:
            The id of the new version.

        Raises:
            TypeError: If ``i`` is not an integer or ``delta`` is not a real
                number.
            IndexError: If ``i`` is outside ``[0, n - 1]``.
            ValueError: If ``delta`` is NaN or infinite.

        Time Complexity:
            O(log n) in each of the three structures.

        Space Complexity:
            O(log n) new persistent nodes.

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8])
            >>> eng.add(1, 10)
            1
            >>> eng.max(0, 2), eng.max_at(0, 0, 2)
            (12, 8)
        """
        i = self._point(i)
        delta = _as_number(delta, "delta")
        self._seg.range_add(i, i + 1, delta)
        self._fen.range_add(i, i + 1, delta)
        self._version = self._pers.range_add(self._version, i, i + 1, delta)
        return self._version

    def range_add(self, l: int, r: int, delta: Number) -> int:
        """Add ``delta`` on the inclusive range ``[l, r]``; return the new version.

        Converted once to the half-open ``[l, r + 1)`` and applied to the
        segment tree (lazy tags), the range Fenwick (two point adds per tree)
        and the persistent tree (permanent tags, path copying).

        Args:
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.
            delta: The amount to add, a finite real number.

        Returns:
            The id of the new version.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer or ``delta`` is
                not a real number.
            IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]``.
            ValueError: If ``l > r`` or ``delta`` is NaN or infinite.

        Time Complexity:
            O(log n) in each of the three structures.

        Space Complexity:
            O(log n) new persistent nodes.

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> eng.range_add(3, 7, 10)
            1
            >>> eng.sum(0, 9), eng.max(2, 5), eng.min(0, 9)
            (95, 19, 0)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        delta = _as_number(delta, "delta")
        self._seg.range_add(lo, hi, delta)
        self._fen.range_add(lo, hi, delta)
        self._version = self._pers.range_add(self._version, lo, hi, delta)
        return self._version

    # ------------------------------------------------------------ current queries

    def sum(self, l: int, r: int) -> Number:
        """Return the sum of the inclusive range ``[l, r]``, answered by the range Fenwick.

        Args:
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The sum of the current values at positions ``l .. r``.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log n): four Fenwick prefix walks.

        Space Complexity:
            O(1).

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> eng.sum(2, 6), eng.sum(0, 9), eng.sum(4, 4)
            (28, 45, 9)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._fen.range_sum(lo, hi)

    def min(self, l: int, r: int) -> Number:
        """Return the minimum of the inclusive range ``[l, r]``, answered by the segment tree.

        Args:
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The smallest current value at positions ``l .. r``.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(1).

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> eng.min(1, 4), eng.min(5, 9)
            (1, 0)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._seg.query_min(lo, hi)

    def max(self, l: int, r: int) -> Number:
        """Return the maximum of the inclusive range ``[l, r]``, answered by the segment tree.

        Args:
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The largest current value at positions ``l .. r``.

        Raises:
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(1).

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> eng.max(0, 9), eng.max(5, 8)
            (9, 7)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._seg.query_max(lo, hi)

    # ------------------------------------------------------------ time travel

    def sum_at(self, version: int, l: int, r: int) -> Number:
        """Return the sum of the inclusive range ``[l, r]`` as it was in ``version``.

        Answered by the persistent segment tree, whose old versions are never
        changed by later updates.

        Args:
            version: A version id, ``0 <= version <= current_version()``.
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The sum of positions ``l .. r`` in that version.

        Raises:
            TypeError: If ``version``, ``l`` or ``r`` is not an integer.
            IndexError: If the version does not exist or ``l`` or ``r`` is
                outside ``[0, n - 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> _ = eng.range_add(3, 7, 10)
            >>> eng.sum_at(0, 0, 9), eng.sum_at(1, 0, 9)
            (45, 95)
            >>> eng.sum_at(2, 0, 9)
            Traceback (most recent call last):
                ...
            IndexError: version 2 does not exist (versions are 0..1)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._pers.query_sum(version, lo, hi)

    def min_at(self, version: int, l: int, r: int) -> Number:
        """Return the minimum of the inclusive range ``[l, r]`` as it was in ``version``.

        Args:
            version: A version id, ``0 <= version <= current_version()``.
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The smallest value at positions ``l .. r`` in that version.

        Raises:
            TypeError: If ``version``, ``l`` or ``r`` is not an integer.
            IndexError: If the version does not exist or ``l`` or ``r`` is
                outside ``[0, n - 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8])
            >>> _ = eng.set(1, 9)
            >>> eng.min_at(0, 0, 2), eng.min_at(1, 0, 2)
            (2, 5)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._pers.query_min(version, lo, hi)

    def max_at(self, version: int, l: int, r: int) -> Number:
        """Return the maximum of the inclusive range ``[l, r]`` as it was in ``version``.

        Args:
            version: A version id, ``0 <= version <= current_version()``.
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The largest value at positions ``l .. r`` in that version.

        Raises:
            TypeError: If ``version``, ``l`` or ``r`` is not an integer.
            IndexError: If the version does not exist or ``l`` or ``r`` is
                outside ``[0, n - 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) for the recursion.

        Examples:
            >>> eng = RangeQueryEngine([5, 2, 8])
            >>> _ = eng.set(1, 9)
            >>> eng.max_at(0, 0, 2), eng.max_at(1, 0, 2)
            (8, 9)
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._pers.query_max(version, lo, hi)

    def current_version(self) -> int:
        """Return the id of the newest version, the number of updates made so far.

        Any version from 0 to this id can be queried with inclusive ranges
        through :meth:`sum_at`, :meth:`min_at` and :meth:`max_at`.

        Returns:
            0 before any update; every successful update adds one.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> eng = RangeQueryEngine([1, 2, 3])
            >>> eng.current_version()
            0
            >>> _ = eng.range_add(0, 2, 1)
            >>> eng.current_version()
            1
        """
        return self._version

    def __repr__(self) -> str:
        """Return the size and the current version (positions 0 .. n - 1, inclusive ranges).

        Returns:
            For example ``RangeQueryEngine(n=10, version=3)``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RangeQueryEngine([1, 2])
            RangeQueryEngine(n=2, version=0)
        """
        return f"RangeQueryEngine(n={self._n}, version={self._version})"
