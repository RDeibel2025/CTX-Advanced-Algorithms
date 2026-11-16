"""Persistent segment tree and persistent map, both by path copying.

Persistence by path copying
---------------------------
A persistent structure keeps every old version queryable after an update.
The cheapest general way to get there in a tree is path copying (Driscoll,
Sarnak, Sleator and Tarjan, 1989): nodes are immutable, and an update builds
new copies of only the nodes whose contents change, which are the nodes on
the root-to-target paths. Every other subtree is shared, by reference,
between the old and the new version, so the versions form a DAG of shared
nodes and each version is just a root pointer. Nothing is ever written into
a node after it is built, so an old root keeps describing exactly the tree it
described when it was made.

PersistentSegmentTree
---------------------
The tree is node-based. The build halves ``[0, n)`` at
``mid = (lo + hi) // 2`` into ``[lo, mid)`` and ``[mid, hi)`` until a range
holds one element, so every internal node has two children and a build of n
elements creates exactly ``2n - 1`` nodes (none for ``n = 0``). Each node
stores ``left``, ``right``, ``tag``, ``sum``, ``min`` and ``max``.

Range add uses non-propagating ("permanent") tags. A node's ``tag`` is the
total delta of the updates that covered the node's whole range when they
reached it, and its ``sum``, ``min`` and ``max`` describe its subtree counting
its own tag and every tag below it, but no ancestor's tag. A query therefore
adds up the tags of the ancestors it passes through: a fully covered node of
length ``len`` contributes ``sum + acc * len`` (or ``min + acc``,
``max + acc``) where ``acc`` is the sum of the tags strictly above it. A tag
is never pushed down, because pushing would write into children that older
versions share.

A range add copies a node only when its range meets ``[l, r)``: a covered
node is copied once with its tag raised, a partly covered one is copied after
its children, and a disjoint one is shared as it is. That is O(log n) new
nodes per update (at most two partial nodes and two covered nodes per level).
A point set is the range add ``[i, i + 1)`` of ``value - point_query(i)``, so
it copies exactly the root-to-leaf path: for ``n = 2^k`` that is ``k + 1`` new
nodes against ``2n - 1`` for a full copy.

Conventions: indices are 0-based and ranges are half-open ``[l, r)`` with
``0 <= l <= r <= n``; anything else raises ``IndexError``. An empty range
``l == r`` gives the identities ``0`` (sum), ``+inf`` (min) and ``-inf``
(max), and a range add over it still makes a new version (sharing the old
root, no new nodes). Version 0 is the initial array; every update returns
the next version id. Values are Python ints or floats (a numpy array is
converted with ``tolist()``, and a numpy scalar value, delta or list element
with ``item()``), so integer sums never overflow.

PersistentMap
-------------
A persistent treap, the tree that the version-control application keeps per
commit (path to content hash). Each :class:`MapNode` holds ``key``,
``value``, ``priority``, ``left``, ``right`` and ``hash``, and is immutable.

* ``priority = stable_hash64(key)``. The tree is a binary search tree on the
  keys and a max-heap on the pair ``(priority, key)``: of two nodes, the one
  with the larger pair is the ancestor (equal priorities are broken by key,
  the larger key on top). Every key set therefore has exactly one shape,
  whatever order the keys were inserted or deleted in, and the expected depth
  is O(log n) because the hashes behave like random priorities.
* ``hash`` is a Merkle hash: ``hashlib.blake2b`` (default 64-byte digest)
  over the canonical encoding of ``(key, value, left hash, right hash)``,
  hexdigest, with ``""`` standing for a missing child. Each field is encoded
  as a one-byte type tag (``b"s"`` for ``str`` as UTF-8, ``b"b"`` for bytes,
  ``b"r"`` for anything else as its ``repr``), an 8-byte big-endian length
  and the bytes, so no two different tuples share an encoding.
* Because the shape is fixed by the key set, two maps with the same
  ``(key, value)`` pairs have the same :attr:`PersistentMap.root_hash`, and
  two subtrees with equal hashes hold the same pairs, which is what lets a
  diff skip them. The empty map's hash is the constant
  :data:`EMPTY_MAP_HASH`, ``blake2b(b"").hexdigest()``.

``set`` and ``delete`` return a new map and copy only the nodes on the
affected search path (plus the split or merge path below it), O(log n)
expected; the old map is unchanged.

Rule for this package (spec 1.1): every structure and query is written by
hand; no library search, sort or prefix-sum routine stands in for a query.

Examples:
    >>> t = PersistentSegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
    >>> v1 = t.point_set(0, 3, 100)
    >>> v2 = t.range_add(v1, 0, 5, 1)
    >>> t.query_sum(0, 0, 10), t.query_sum(v1, 0, 10), t.query_sum(v2, 0, 10)
    (45, 144, 149)
    >>> m = PersistentMap().set("b", "2").set("a", "1")
    >>> m.items(), m.root_hash == PersistentMap().set("a", "1").set("b", "2").root_hash
    ([('a', '1'), ('b', '2')], True)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import operator
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from src.utils.random_utils import stable_hash64

__all__ = [
    "EMPTY_MAP_HASH",
    "MapNode",
    "PersistentMap",
    "PersistentSegmentTree",
    "SegmentNode",
]

#: ``root_hash`` of the empty :class:`PersistentMap`: ``blake2b(b"").hexdigest()``.
EMPTY_MAP_HASH: str = hashlib.blake2b(b"").hexdigest()

_SET = object.__setattr__


def _as_int(value: Any, name: str) -> int:
    """Return ``value`` as a Python ``int``, rejecting non-integers and ``bool``.

    Args:
        value: An ``int`` or any object with ``__index__`` (numpy integers).
        name: The parameter name, for the error message.

    Returns:
        ``operator.index(value)``.

    Raises:
        TypeError: If ``value`` is a ``bool`` or not an integer.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _as_int(3, "i")
        3
        >>> _as_int(2.0, "i")
        Traceback (most recent call last):
            ...
        TypeError: i must be an int, got float
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None


def _as_value(x: Any) -> Any:
    """Return a numpy scalar as the equal Python number; leave anything else as it is.

    A numpy integer would turn every sum it takes part in into a fixed-width
    ``int64`` that wraps around (or raises ``OverflowError``) past ``2**63``,
    and a ``float32`` would round every sum to single precision. A numpy
    array given to the constructor is converted with ``tolist()`` for the
    same reason; this does the same for single values (deltas, point-set
    values and the elements of a plain list).

    Args:
        x: A value or a delta.

    Returns:
        ``x.item()`` for a number that has an ``item`` method (numpy
        scalars), otherwise ``x`` unchanged.

    Raises:
        Nothing.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> import numpy as np
        >>> type(_as_value(np.int64(3))).__name__, type(_as_value(np.float32(0.5))).__name__
        ('int', 'float')
        >>> _as_value(7), _as_value(2.5)
        (7, 2.5)
    """
    t = type(x)
    if t is int or t is float:
        return x
    item = getattr(x, "item", None)
    if callable(item) and isinstance(x, numbers.Number):
        return item()
    return x


# ---------------------------------------------------------------------------
# Persistent segment tree
# ---------------------------------------------------------------------------


class SegmentNode:
    """One immutable node of a :class:`PersistentSegmentTree`.

    The node does not store its range; the tree passes ``[lo, hi)`` down as
    it walks. ``sum``, ``min`` and ``max`` count this node's own ``tag`` and
    every tag below it, but no ancestor's tag (the permanent-tag rule of the
    module docstring). A leaf has ``left = right = None`` and
    ``sum = min = max = value + tag``. After ``__init__`` returns the node
    cannot be changed: assigning or deleting an attribute raises
    ``AttributeError``, so a node shared by many versions stays correct for
    all of them.

    Args:
        left: The left child (range ``[lo, mid)``), or ``None`` for a leaf.
        right: The right child (range ``[mid, hi)``), or ``None`` for a leaf.
        tag: The permanent add tag of this node.
        total: The subtree sum, including ``tag`` and every tag below.
        low: The subtree minimum, likewise.
        high: The subtree maximum, likewise.

    Raises:
        AttributeError: On any later attempt to set or delete an attribute.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1): six slots, no ``__dict__``.

    Examples:
        >>> leaf = SegmentNode(None, None, 0, 7, 7, 7)
        >>> leaf.sum, leaf.left is None
        (7, True)
        >>> leaf.sum = 8
        Traceback (most recent call last):
            ...
        AttributeError: SegmentNode is immutable: cannot set 'sum'
    """

    __slots__ = ("left", "right", "tag", "sum", "min", "max")

    def __init__(
        self,
        left: Optional["SegmentNode"],
        right: Optional["SegmentNode"],
        tag: Any,
        total: Any,
        low: Any,
        high: Any,
    ) -> None:
        _SET(self, "left", left)
        _SET(self, "right", right)
        _SET(self, "tag", tag)
        _SET(self, "sum", total)
        _SET(self, "min", low)
        _SET(self, "max", high)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: published nodes are immutable.

        Args:
            name: The attribute name.
            value: The value that would have been assigned.

        Raises:
            AttributeError: Always.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> SegmentNode(None, None, 0, 1, 1, 1).tag = 5
            Traceback (most recent call last):
                ...
            AttributeError: SegmentNode is immutable: cannot set 'tag'
        """
        raise AttributeError(f"SegmentNode is immutable: cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion: published nodes are immutable.

        Args:
            name: The attribute name.

        Raises:
            AttributeError: Always.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> del SegmentNode(None, None, 0, 1, 1, 1).left
            Traceback (most recent call last):
                ...
            AttributeError: SegmentNode is immutable: cannot delete 'left'
        """
        raise AttributeError(f"SegmentNode is immutable: cannot delete {name!r}")

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        """Rebuild through the constructor, so ``copy`` and ``pickle`` work.

        The default protocol restores slots with ``setattr``, which this
        class refuses; going through ``__init__`` keeps the node immutable.

        Returns:
            ``(SegmentNode, (left, right, tag, sum, min, max))``.

        Raises:
            Nothing.

        Time Complexity:
            O(1) for this node (a deep copy or pickle visits each shared
            node once).

        Space Complexity:
            O(1).

        Examples:
            >>> import copy
            >>> node = SegmentNode(None, None, 0, 7, 7, 7)
            >>> copy.deepcopy(node).sum, copy.copy(node).max
            (7, 7)
        """
        return (SegmentNode, (self.left, self.right, self.tag, self.sum, self.min, self.max))

    def __repr__(self) -> str:
        """Return a short description with the tag and the three aggregates.

        Returns:
            For example ``SegmentNode(tag=0, sum=7, min=7, max=7, leaf=True)``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> SegmentNode(None, None, 0, 7, 7, 7)
            SegmentNode(tag=0, sum=7, min=7, max=7, leaf=True)
        """
        return (
            f"SegmentNode(tag={self.tag!r}, sum={self.sum!r}, min={self.min!r}, "
            f"max={self.max!r}, leaf={self.left is None})"
        )


def _build(vals: List[Any], lo: int, hi: int) -> SegmentNode:
    """Build the subtree for ``vals[lo:hi]`` by halving at ``(lo + hi) // 2``.

    Args:
        vals: The initial values.
        lo: Start of the range (inclusive).
        hi: End of the range (exclusive), with ``hi - lo >= 1``.

    Returns:
        The subtree root; the subtree has ``2 (hi - lo) - 1`` nodes, all tags 0.

    Raises:
        Nothing.

    Time Complexity:
        O(hi - lo).

    Space Complexity:
        O(hi - lo) nodes plus O(log(hi - lo)) recursion depth.

    Examples:
        >>> _build([4, 1, 6], 0, 3)
        SegmentNode(tag=0, sum=11, min=1, max=6, leaf=False)
    """
    if hi - lo == 1:
        v = vals[lo]
        if type(v) is not int and type(v) is not float:
            v = _as_value(v)
        return SegmentNode(None, None, 0, v, v, v)
    mid = (lo + hi) // 2
    left = _build(vals, lo, mid)
    right = _build(vals, mid, hi)
    lmin, rmin, lmax, rmax = left.min, right.min, left.max, right.max
    return SegmentNode(
        left,
        right,
        0,
        left.sum + right.sum,
        lmin if lmin <= rmin else rmin,
        lmax if lmax >= rmax else rmax,
    )


def _sum(node: SegmentNode, lo: int, hi: int, l: int, r: int, acc: Any) -> Any:
    """Sum of ``[l, r)`` inside a node of range ``[lo, hi)`` that meets it.

    Args:
        node: The node of range ``[lo, hi)``.
        lo: Start of the node's range.
        hi: End of the node's range (exclusive).
        l: Query start; ``[l, r)`` meets ``[lo, hi)`` and ``l < r``.
        r: Query end (exclusive).
        acc: Sum of the tags of the node's strict ancestors.

    Returns:
        The sum of the current values in ``[max(l, lo), min(r, hi))``.

    Raises:
        Nothing.

    Time Complexity:
        O(log(hi - lo)).

    Space Complexity:
        O(log(hi - lo)) recursion depth.

    Examples:
        >>> _sum(_build([4, 1, 6], 0, 3), 0, 3, 1, 3, 10)
        27
    """
    if l <= lo and hi <= r:
        return node.sum + acc * (hi - lo)
    acc = acc + node.tag
    mid = (lo + hi) // 2
    if r <= mid:
        return _sum(node.left, lo, mid, l, r, acc)
    if mid <= l:
        return _sum(node.right, mid, hi, l, r, acc)
    return _sum(node.left, lo, mid, l, r, acc) + _sum(node.right, mid, hi, l, r, acc)


def _min(node: SegmentNode, lo: int, hi: int, l: int, r: int, acc: Any) -> Any:
    """Minimum of ``[l, r)`` inside a node of range ``[lo, hi)`` that meets it.

    Args:
        node: The node of range ``[lo, hi)``.
        lo: Start of the node's range.
        hi: End of the node's range (exclusive).
        l: Query start; ``[l, r)`` meets ``[lo, hi)`` and ``l < r``.
        r: Query end (exclusive).
        acc: Sum of the tags of the node's strict ancestors.

    Returns:
        The minimum current value in ``[max(l, lo), min(r, hi))``.

    Raises:
        Nothing.

    Time Complexity:
        O(log(hi - lo)).

    Space Complexity:
        O(log(hi - lo)) recursion depth.

    Examples:
        >>> _min(_build([4, 1, 6], 0, 3), 0, 3, 0, 1, 0)
        4
    """
    if l <= lo and hi <= r:
        return node.min + acc
    acc = acc + node.tag
    mid = (lo + hi) // 2
    if r <= mid:
        return _min(node.left, lo, mid, l, r, acc)
    if mid <= l:
        return _min(node.right, mid, hi, l, r, acc)
    a = _min(node.left, lo, mid, l, r, acc)
    b = _min(node.right, mid, hi, l, r, acc)
    return a if a <= b else b


def _max(node: SegmentNode, lo: int, hi: int, l: int, r: int, acc: Any) -> Any:
    """Maximum of ``[l, r)`` inside a node of range ``[lo, hi)`` that meets it.

    Args:
        node: The node of range ``[lo, hi)``.
        lo: Start of the node's range.
        hi: End of the node's range (exclusive).
        l: Query start; ``[l, r)`` meets ``[lo, hi)`` and ``l < r``.
        r: Query end (exclusive).
        acc: Sum of the tags of the node's strict ancestors.

    Returns:
        The maximum current value in ``[max(l, lo), min(r, hi))``.

    Raises:
        Nothing.

    Time Complexity:
        O(log(hi - lo)).

    Space Complexity:
        O(log(hi - lo)) recursion depth.

    Examples:
        >>> _max(_build([4, 1, 6], 0, 3), 0, 3, 0, 2, 0)
        4
    """
    if l <= lo and hi <= r:
        return node.max + acc
    acc = acc + node.tag
    mid = (lo + hi) // 2
    if r <= mid:
        return _max(node.left, lo, mid, l, r, acc)
    if mid <= l:
        return _max(node.right, mid, hi, l, r, acc)
    a = _max(node.left, lo, mid, l, r, acc)
    b = _max(node.right, mid, hi, l, r, acc)
    return a if a >= b else b


class PersistentSegmentTree:
    """A fully persistent segment tree with range add, by path copying.

    Version 0 is the array given to the constructor. :meth:`range_add` and
    :meth:`point_set` take the version to update and return the id of a new
    version; the old one is untouched and can still be queried. Range add
    uses permanent tags (see the module docstring), so no update ever writes
    into an existing node. Ranges are half-open ``[l, r)``, indices 0-based.

    Args:
        values: The initial array: any iterable of ints or floats, or a numpy
            array (converted with ``tolist()``; numpy scalars inside a list
            are converted with ``item()``). It may be empty.

    Raises:
        TypeError: If ``values`` is not iterable.

    Time Complexity:
        Build Theta(n). Every query and update O(log n).

    Space Complexity:
        ``2n - 1`` nodes for the build plus the nodes each update creates
        (``k + 1`` per point set when ``n = 2^k``, O(log n) per range add),
        against ``2n - 1`` per version for a full copy.

    Examples:
        >>> t = PersistentSegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
        >>> t.nodes_created(), t.versions()
        (19, 1)
        >>> v1 = t.point_set(0, 3, 100)
        >>> v2 = t.range_add(v1, 0, 5, 1)
        >>> (v1, v2), t.versions()
        ((1, 2), 3)
        >>> t.point_query(v2, 3), t.point_query(v1, 3), t.point_query(0, 3)
        (101, 100, 1)
        >>> t.query_max(v2, 0, 10), t.query_min(v2, 0, 5)
        (101, 3)
    """

    def __init__(self, values: Iterable[Any]) -> None:
        vals = values.tolist() if hasattr(values, "tolist") else list(values)
        n = len(vals)
        self._n: int = n
        self._roots: List[Optional[SegmentNode]] = [_build(vals, 0, n) if n else None]
        self._created: int = 2 * n - 1 if n else 0

    # ------------------------------------------------------------ helpers

    def _root_of(self, version: Any) -> Optional[SegmentNode]:
        """Return the root of ``version``, validating the id.

        Args:
            version: A version id in ``[0, versions())``.

        Returns:
            The root node (``None`` only when ``n = 0``).

        Raises:
            TypeError: If ``version`` is not an int.
            IndexError: If the version does not exist.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> PersistentSegmentTree([1, 2])._root_of(0).sum
            3
            >>> PersistentSegmentTree([1, 2])._root_of(1)
            Traceback (most recent call last):
                ...
            IndexError: version 1 does not exist (versions are 0..0)
        """
        v = _as_int(version, "version")
        if not 0 <= v < len(self._roots):
            raise IndexError(
                f"version {v} does not exist (versions are 0..{len(self._roots) - 1})"
            )
        return self._roots[v]

    def _check_range(self, l: Any, r: Any) -> Tuple[int, int]:
        """Validate a half-open range ``[l, r)`` with ``0 <= l <= r <= n``.

        Args:
            l: Range start.
            r: Range end (exclusive).

        Returns:
            ``(l, r)`` as ints.

        Raises:
            TypeError: If ``l`` or ``r`` is not an int.
            IndexError: If the range is not inside ``[0, n]`` or ``l > r``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> PersistentSegmentTree([1, 2, 3])._check_range(0, 3)
            (0, 3)
            >>> PersistentSegmentTree([1, 2, 3])._check_range(2, 1)
            Traceback (most recent call last):
                ...
            IndexError: range [2, 1) is invalid for n = 3 (need 0 <= l <= r <= n)
        """
        l = _as_int(l, "l")
        r = _as_int(r, "r")
        if not 0 <= l <= r <= self._n:
            raise IndexError(
                f"range [{l}, {r}) is invalid for n = {self._n} (need 0 <= l <= r <= n)"
            )
        return l, r

    def _publish(self, root: Optional[SegmentNode]) -> int:
        """Append a new version with the given root and return its id.

        Args:
            root: The root of the new version.

        Returns:
            The new version id.

        Raises:
            Nothing.

        Time Complexity:
            O(1) amortised.

        Space Complexity:
            O(1).

        Examples:
            >>> t = PersistentSegmentTree([1])
            >>> t._publish(t.root(0))
            1
        """
        self._roots.append(root)
        return len(self._roots) - 1

    # ------------------------------------------------------------ updates

    def range_add(self, version: int, l: int, r: int, delta: Any) -> int:
        """Add ``delta`` to every element of ``[l, r)`` of a version.

        Walks down from the version's root. A node disjoint from ``[l, r)``
        is shared unchanged; a node inside ``[l, r)`` is copied once with
        ``tag + delta``, ``sum + delta * len``, ``min + delta`` and
        ``max + delta`` and its children shared; a partly covered node is
        copied with its updated children and the same tag, its aggregates
        recomputed as the children's plus its own tag. No tag is pushed down.

        Args:
            version: The version to update.
            l: Range start, ``0 <= l <= r``.
            r: Range end (exclusive), ``r <= n``.
            delta: The amount to add (int or float; a numpy scalar is
                converted with ``item()``).

        Returns:
            The id of the new version. An empty range (``l == r``) still makes
            a new version, sharing the old root.

        Raises:
            TypeError: If ``version``, ``l`` or ``r`` is not an int.
            IndexError: If the version does not exist or the range is invalid.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) new nodes: at most two partly covered and two covered
            nodes per level.

        Examples:
            >>> t = PersistentSegmentTree([1, 2, 3, 4])
            >>> v = t.range_add(0, 1, 3, 10)
            >>> t.to_list(v), t.to_list(0)
            ([1, 12, 13, 4], [1, 2, 3, 4])
            >>> t.nodes_created()
            12
        """
        root = self._root_of(version)
        l, r = self._check_range(l, r)
        if l == r:
            return self._publish(root)
        delta = _as_value(delta)
        made = 0

        def add(node: SegmentNode, lo: int, hi: int) -> SegmentNode:
            """Return the updated copy of ``node`` (range ``[lo, hi)``), or ``node`` if disjoint."""
            nonlocal made
            if r <= lo or hi <= l:
                return node
            made += 1
            if l <= lo and hi <= r:
                return SegmentNode(
                    node.left,
                    node.right,
                    node.tag + delta,
                    node.sum + delta * (hi - lo),
                    node.min + delta,
                    node.max + delta,
                )
            mid = (lo + hi) // 2
            left = add(node.left, lo, mid)
            right = add(node.right, mid, hi)
            tag = node.tag
            lmin, rmin, lmax, rmax = left.min, right.min, left.max, right.max
            return SegmentNode(
                left,
                right,
                tag,
                left.sum + right.sum + tag * (hi - lo),
                (lmin if lmin <= rmin else rmin) + tag,
                (lmax if lmax >= rmax else rmax) + tag,
            )

        new_root = add(root, 0, self._n)
        self._created += made
        return self._publish(new_root)

    def point_set(self, version: int, i: int, value: Any) -> int:
        """Set element ``i`` of a version to ``value``.

        Defined as ``range_add(version, i, i + 1, value - point_query(version,
        i))``, so it copies exactly the root-to-leaf path of ``i``: ``k + 1``
        new nodes when ``n = 2^k``, even when the value does not change.
        Because the new value is reached through a delta, integers are exact
        but a float may come back from :meth:`point_query` off by a rounding
        error.

        Args:
            version: The version to update.
            i: The index, ``0 <= i < n``.
            value: The new value (a numpy scalar is converted with ``item()``).

        Returns:
            The id of the new version.

        Raises:
            TypeError: If ``version`` or ``i`` is not an int.
            IndexError: If the version does not exist or ``i`` is out of range.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) new nodes (the path from the root to leaf ``i``).

        Examples:
            >>> t = PersistentSegmentTree([0] * 8)
            >>> before = t.nodes_created()
            >>> v = t.point_set(0, 5, 9)
            >>> t.nodes_created() - before, t.to_list(v)
            (4, [0, 0, 0, 0, 0, 9, 0, 0])
        """
        current = self.point_query(version, i)
        return self.range_add(version, i, i + 1, _as_value(value) - current)

    # ------------------------------------------------------------ queries

    def query_sum(self, version: int, l: int, r: int) -> Any:
        """Return the sum of ``[l, r)`` in a version.

        Args:
            version: The version to query.
            l: Range start.
            r: Range end (exclusive), ``0 <= l <= r <= n``.

        Returns:
            The sum; ``0`` for an empty range.

        Raises:
            TypeError: If an argument is not an int.
            IndexError: If the version does not exist or the range is invalid.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) recursion depth.

        Examples:
            >>> t = PersistentSegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> t.query_sum(0, 2, 7), t.query_sum(0, 4, 4)
            (28, 0)
        """
        root = self._root_of(version)
        l, r = self._check_range(l, r)
        if l == r:
            return 0
        return _sum(root, 0, self._n, l, r, 0)

    def query_min(self, version: int, l: int, r: int) -> Any:
        """Return the minimum of ``[l, r)`` in a version.

        Args:
            version: The version to query.
            l: Range start.
            r: Range end (exclusive), ``0 <= l <= r <= n``.

        Returns:
            The minimum; ``math.inf`` for an empty range.

        Raises:
            TypeError: If an argument is not an int.
            IndexError: If the version does not exist or the range is invalid.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) recursion depth.

        Examples:
            >>> t = PersistentSegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> t.query_min(0, 1, 5), t.query_min(0, 5, 10), t.query_min(0, 3, 3)
            (1, 0, inf)
        """
        root = self._root_of(version)
        l, r = self._check_range(l, r)
        if l == r:
            return math.inf
        return _min(root, 0, self._n, l, r, 0)

    def query_max(self, version: int, l: int, r: int) -> Any:
        """Return the maximum of ``[l, r)`` in a version.

        Args:
            version: The version to query.
            l: Range start.
            r: Range end (exclusive), ``0 <= l <= r <= n``.

        Returns:
            The maximum; ``-math.inf`` for an empty range.

        Raises:
            TypeError: If an argument is not an int.
            IndexError: If the version does not exist or the range is invalid.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(log n) recursion depth.

        Examples:
            >>> t = PersistentSegmentTree([5, 2, 8, 1, 9, 3, 7, 4, 6, 0])
            >>> t.query_max(0, 0, 10), t.query_max(0, 0, 0)
            (9, -inf)
        """
        root = self._root_of(version)
        l, r = self._check_range(l, r)
        if l == r:
            return -math.inf
        return _max(root, 0, self._n, l, r, 0)

    def point_query(self, version: int, i: int) -> Any:
        """Return element ``i`` of a version.

        Walks from the root to leaf ``i``, adding the tag of every internal
        node it passes; the leaf's own value already includes its tag.

        Args:
            version: The version to query.
            i: The index, ``0 <= i < n``.

        Returns:
            The element's value in that version.

        Raises:
            TypeError: If ``version`` or ``i`` is not an int.
            IndexError: If the version does not exist or ``i`` is out of range.

        Time Complexity:
            O(log n).

        Space Complexity:
            O(1).

        Examples:
            >>> t = PersistentSegmentTree([5, 2, 8])
            >>> t.point_query(t.range_add(0, 0, 3, 1), 2)
            9
            >>> t.point_query(0, 3)
            Traceback (most recent call last):
                ...
            IndexError: index 3 out of range for n = 3
        """
        node = self._root_of(version)
        i = _as_int(i, "i")
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for n = {self._n}")
        lo, hi = 0, self._n
        acc = 0
        while hi - lo > 1:
            acc = acc + node.tag
            mid = (lo + hi) // 2
            if i < mid:
                node, hi = node.left, mid
            else:
                node, lo = node.right, mid
        return node.sum + acc

    def to_list(self, version: int) -> List[Any]:
        """Return every element of a version, left to right.

        A depth-first walk carries the sum of the tags above each node down
        to the leaves; nothing is written into the tree.

        Args:
            version: The version to read.

        Returns:
            A new list of length n.

        Raises:
            TypeError: If ``version`` is not an int.
            IndexError: If the version does not exist.

        Time Complexity:
            O(n).

        Space Complexity:
            O(n) for the result, O(log n) for the walk's stack.

        Examples:
            >>> PersistentSegmentTree([3, 1, 2]).to_list(0)
            [3, 1, 2]
            >>> PersistentSegmentTree([]).to_list(0)
            []
        """
        root = self._root_of(version)
        out: List[Any] = []
        if root is None:
            return out
        stack = [(root, 0)]
        while stack:
            node, acc = stack.pop()
            if node.left is None:
                out.append(node.sum + acc)
            else:
                below = acc + node.tag
                stack.append((node.right, below))
                stack.append((node.left, below))
        return out

    # ------------------------------------------------------------ bookkeeping

    def root(self, version: int) -> Optional[SegmentNode]:
        """Return the root node of a version (``None`` when ``n = 0``).

        Versions share nodes, so the roots of two versions lead to many of
        the same :class:`SegmentNode` objects; tests use this to count the
        nodes an update copied.

        Args:
            version: A version id.

        Returns:
            The immutable root node.

        Raises:
            TypeError: If ``version`` is not an int.
            IndexError: If the version does not exist.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> t = PersistentSegmentTree([1, 2, 3, 4])
            >>> v = t.point_set(0, 0, 5)
            >>> t.root(v).right is t.root(0).right
            True
        """
        return self._root_of(version)

    def versions(self) -> int:
        """Return the number of versions; ids run from 0 to ``versions() - 1``.

        Returns:
            1 plus the number of updates made so far.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> t = PersistentSegmentTree([1, 2])
            >>> _ = t.range_add(0, 0, 2, 1)
            >>> t.versions()
            2
        """
        return len(self._roots)

    def nodes_created(self) -> int:
        """Return how many nodes this tree has ever created (build plus updates).

        Returns:
            ``2n - 1`` after the build (0 when ``n = 0``) plus every node each
            update copied.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> t = PersistentSegmentTree(range(16))
            >>> t.nodes_created()
            31
            >>> _ = t.point_set(0, 7, 0)
            >>> t.nodes_created()
            36
        """
        return self._created

    def __len__(self) -> int:
        """Return n, the number of elements in every version.

        Returns:
            The array length.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(PersistentSegmentTree([4, 5, 6]))
            3
        """
        return self._n


# ---------------------------------------------------------------------------
# Persistent map (treap with Merkle hashes)
# ---------------------------------------------------------------------------


def _encode_field(x: Any) -> bytes:
    """Encode one field for the Merkle hash: type tag, 8-byte length, bytes.

    Args:
        x: A ``str`` (UTF-8), ``bytes``/``bytearray`` (raw) or anything else
            (its ``repr`` in UTF-8).

    Returns:
        ``tag + len(data).to_bytes(8, "big") + data`` with ``tag`` one of
        ``b"s"``, ``b"b"``, ``b"r"``.

    Raises:
        Nothing.

    Time Complexity:
        O(len(data)).

    Space Complexity:
        O(len(data)).

    Examples:
        >>> _encode_field("ab")
        b's\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x02ab'
        >>> _encode_field(7)[:1]
        b'r'
    """
    if isinstance(x, str):
        tag, data = b"s", x.encode("utf-8", "surrogatepass")
    elif isinstance(x, (bytes, bytearray)):
        tag, data = b"b", bytes(x)
    else:
        tag, data = b"r", repr(x).encode("utf-8", "surrogatepass")
    return tag + len(data).to_bytes(8, "big") + data


def _node_hash(key: Any, value: Any, left_hash: str, right_hash: str) -> str:
    """Return the Merkle hash of a map node from its fields and child hashes.

    Args:
        key: The node's key.
        value: The node's value.
        left_hash: The left child's hash, or ``""`` when there is none.
        right_hash: The right child's hash, or ``""`` when there is none.

    Returns:
        ``hashlib.blake2b`` (64-byte digest) of the four encoded fields,
        as 128 lowercase hex characters.

    Raises:
        Nothing.

    Time Complexity:
        O(L) for L total encoded bytes.

    Space Complexity:
        O(L).

    Examples:
        >>> len(_node_hash("a", "1", "", ""))
        128
        >>> _node_hash("a", "1", "", "") == _node_hash("a", "2", "", "")
        False
    """
    payload = (
        _encode_field(key)
        + _encode_field(value)
        + _encode_field(left_hash)
        + _encode_field(right_hash)
    )
    return hashlib.blake2b(payload).hexdigest()


class MapNode:
    """One immutable node of a :class:`PersistentMap` treap.

    The hash is computed once, in the constructor, from the key, the value
    and the children's hashes, so it summarises the whole subtree (a Merkle
    hash). Assigning or deleting an attribute afterwards raises
    ``AttributeError``.

    Attributes:
        key: The key (a ``str`` in the version-control application).
        value: The value (a ``str``).
        priority: ``stable_hash64(key)``; the heap order is on
            ``(priority, key)``, larger on top.
        left: Subtree of smaller keys, or ``None``.
        right: Subtree of larger keys, or ``None``.
        hash: ``blake2b`` hexdigest of ``(key, value, left hash or "",
            right hash or "")`` in the module's canonical encoding.

    Args:
        key: The key.
        value: The value.
        priority: The heap priority; callers pass ``stable_hash64(key)``.
        left: The left child or ``None``.
        right: The right child or ``None``.

    Raises:
        AttributeError: On any later attempt to set or delete an attribute.

    Time Complexity:
        O(len(key) + len(value)) for the hash.

    Space Complexity:
        O(1) slots plus the 128-character hash string.

    Examples:
        >>> node = MapNode("a", "1", stable_hash64("a"), None, None)
        >>> node.key, node.value, len(node.hash)
        ('a', '1', 128)
        >>> node.value = "2"
        Traceback (most recent call last):
            ...
        AttributeError: MapNode is immutable: cannot set 'value'
    """

    __slots__ = ("key", "value", "priority", "left", "right", "hash")

    def __init__(
        self,
        key: Any,
        value: Any,
        priority: int,
        left: Optional["MapNode"],
        right: Optional["MapNode"],
    ) -> None:
        _SET(self, "key", key)
        _SET(self, "value", value)
        _SET(self, "priority", priority)
        _SET(self, "left", left)
        _SET(self, "right", right)
        _SET(
            self,
            "hash",
            _node_hash(
                key,
                value,
                left.hash if left is not None else "",
                right.hash if right is not None else "",
            ),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: published nodes are immutable.

        Args:
            name: The attribute name.
            value: The value that would have been assigned.

        Raises:
            AttributeError: Always.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> MapNode("k", "v", 0, None, None).left = None
            Traceback (most recent call last):
                ...
            AttributeError: MapNode is immutable: cannot set 'left'
        """
        raise AttributeError(f"MapNode is immutable: cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion: published nodes are immutable.

        Args:
            name: The attribute name.

        Raises:
            AttributeError: Always.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> del MapNode("k", "v", 0, None, None).hash
            Traceback (most recent call last):
                ...
            AttributeError: MapNode is immutable: cannot delete 'hash'
        """
        raise AttributeError(f"MapNode is immutable: cannot delete {name!r}")

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        """Rebuild through the constructor, so ``copy`` and ``pickle`` work.

        The default protocol restores slots with ``setattr``, which this
        class refuses; going through ``__init__`` keeps the node immutable
        and recomputes the same hash from the same fields.

        Returns:
            ``(MapNode, (key, value, priority, left, right))``.

        Raises:
            Nothing.

        Time Complexity:
            O(1) for this node (a deep copy or pickle visits each shared
            node once and rehashes it).

        Space Complexity:
            O(1).

        Examples:
            >>> import copy
            >>> node = MapNode("a", "1", 0, None, None)
            >>> copy.deepcopy(node).hash == node.hash
            True
        """
        return (MapNode, (self.key, self.value, self.priority, self.left, self.right))

    def __repr__(self) -> str:
        """Return the key, value and the first 12 hex digits of the hash.

        Returns:
            For example ``MapNode(key='a', value='1', hash='5a1c...')``.

        Raises:
            Nothing.

        Time Complexity:
            O(len(key) + len(value)).

        Space Complexity:
            O(len(key) + len(value)).

        Examples:
            >>> repr(MapNode("a", "1", 0, None, None)).startswith("MapNode(key='a', value='1'")
            True
        """
        return f"MapNode(key={self.key!r}, value={self.value!r}, hash={self.hash[:12]!r}...)"


def _with(node: MapNode, left: Optional[MapNode], right: Optional[MapNode]) -> MapNode:
    """Return ``node`` with new children: a path copy, or ``node`` itself if unchanged.

    Args:
        node: The node to copy.
        left: The new left child.
        right: The new right child.

    Returns:
        ``node`` when both children are the same objects, else a new
        :class:`MapNode` with the same key, value and priority.

    Raises:
        Nothing.

    Time Complexity:
        O(1) plus one hash.

    Space Complexity:
        O(1).

    Examples:
        >>> leaf = MapNode("a", "1", 5, None, None)
        >>> _with(leaf, None, None) is leaf
        True
    """
    if left is node.left and right is node.right:
        return node
    return MapNode(node.key, node.value, node.priority, left, right)


def _above(a: MapNode, b: MapNode) -> bool:
    """Return True if ``a`` must be an ancestor of ``b`` in the treap.

    Args:
        a: A node.
        b: A node with a different key.

    Returns:
        ``(a.priority, a.key) > (b.priority, b.key)``.

    Raises:
        Nothing.

    Time Complexity:
        O(1) plus one key comparison.

    Space Complexity:
        O(1).

    Examples:
        >>> _above(MapNode("a", "", 2, None, None), MapNode("b", "", 1, None, None))
        True
        >>> _above(MapNode("a", "", 1, None, None), MapNode("b", "", 1, None, None))
        False
    """
    if a.priority != b.priority:
        return a.priority > b.priority
    return b.key < a.key


def _find(node: Optional[MapNode], key: Any) -> Optional[MapNode]:
    """Return the node holding ``key``, or ``None``.

    Args:
        node: A subtree root.
        key: The key to look for.

    Returns:
        The node or ``None``.

    Raises:
        TypeError: If ``key`` cannot be compared with the stored keys.

    Time Complexity:
        O(depth), O(log n) expected.

    Space Complexity:
        O(1).

    Examples:
        >>> _find(PersistentMap().set("a", "1").root, "a").value
        '1'
    """
    while node is not None:
        if key < node.key:
            node = node.left
        elif node.key < key:
            node = node.right
        else:
            return node
    return None


def _split(
    node: Optional[MapNode], key: Any
) -> Tuple[Optional[MapNode], Optional[MapNode]]:
    """Split a subtree into keys below ``key`` and keys above it.

    ``key`` must not be in the subtree. Only the nodes on the search path of
    ``key`` are copied; both halves are valid treaps.

    Args:
        node: A subtree root.
        key: A key not present in the subtree.

    Returns:
        ``(smaller, larger)``.

    Raises:
        TypeError: If ``key`` cannot be compared with the stored keys.

    Time Complexity:
        O(depth), O(log n) expected.

    Space Complexity:
        O(depth) new nodes and recursion depth.

    Examples:
        >>> m = PersistentMap({"a": "1", "c": "3"})
        >>> low, high = _split(m.root, "b")
        >>> low.key, high.key, m.items()
        ('a', 'c', [('a', '1'), ('c', '3')])
    """
    if node is None:
        return None, None
    if key < node.key:
        smaller, larger = _split(node.left, key)
        return smaller, _with(node, larger, node.right)
    smaller, larger = _split(node.right, key)
    return _with(node, node.left, smaller), larger


def _merge(a: Optional[MapNode], b: Optional[MapNode]) -> Optional[MapNode]:
    """Join two treaps where every key of ``a`` is below every key of ``b``.

    Args:
        a: The treap of smaller keys.
        b: The treap of larger keys.

    Returns:
        The root of the joined treap; nodes along the merge path are copied.

    Raises:
        Nothing.

    Time Complexity:
        O(depth(a) + depth(b)), O(log n) expected.

    Space Complexity:
        O(depth(a) + depth(b)) new nodes and recursion depth.

    Examples:
        >>> m = PersistentMap({"a": "1", "b": "2", "c": "3"})
        >>> low, high = _split(m.root, "bb")
        >>> _merge(low, high).hash == m.root_hash
        True
    """
    if a is None:
        return b
    if b is None:
        return a
    if _above(a, b):
        return _with(a, a.left, _merge(a.right, b))
    return _with(b, _merge(a, b.left), b.right)


def _insert(node: Optional[MapNode], key: Any, value: Any, priority: int) -> MapNode:
    """Insert a key that is not yet present, copying the search path.

    The walk goes down by key until it meets the first node that the new one
    must sit above; the subtree there is split around ``key`` and becomes the
    new node's two children.

    Args:
        node: A subtree root.
        key: A key not present in the subtree.
        value: Its value.
        priority: ``stable_hash64(key)``.

    Returns:
        The new subtree root.

    Raises:
        TypeError: If ``key`` cannot be compared with the stored keys.

    Time Complexity:
        O(depth), O(log n) expected.

    Space Complexity:
        O(depth) new nodes and recursion depth.

    Examples:
        >>> _insert(None, "a", "1", 3).key
        'a'
    """
    if node is None:
        return MapNode(key, value, priority, None, None)
    if priority > node.priority or (priority == node.priority and node.key < key):
        smaller, larger = _split(node, key)
        return MapNode(key, value, priority, smaller, larger)
    if key < node.key:
        return _with(node, _insert(node.left, key, value, priority), node.right)
    return _with(node, node.left, _insert(node.right, key, value, priority))


def _replace(node: MapNode, key: Any, value: Any) -> MapNode:
    """Change the value of a key that is present; the shape does not change.

    Args:
        node: A subtree root containing ``key``.
        key: The key.
        value: Its new value.

    Returns:
        The new subtree root (the search path is copied).

    Raises:
        TypeError: If ``key`` cannot be compared with the stored keys.

    Time Complexity:
        O(depth), O(log n) expected.

    Space Complexity:
        O(depth) new nodes and recursion depth.

    Examples:
        >>> _replace(PersistentMap({"a": "1"}).root, "a", "2").value
        '2'
    """
    if key < node.key:
        return _with(node, _replace(node.left, key, value), node.right)
    if node.key < key:
        return _with(node, node.left, _replace(node.right, key, value))
    return MapNode(node.key, value, node.priority, node.left, node.right)


def _remove(node: MapNode, key: Any) -> Optional[MapNode]:
    """Remove a key that is present: replace its node by the merge of its children.

    Args:
        node: A subtree root containing ``key``.
        key: The key to remove.

    Returns:
        The new subtree root, or ``None`` if it became empty.

    Raises:
        TypeError: If ``key`` cannot be compared with the stored keys.

    Time Complexity:
        O(depth), O(log n) expected.

    Space Complexity:
        O(depth) new nodes and recursion depth.

    Examples:
        >>> _remove(PersistentMap({"a": "1"}).root, "a") is None
        True
    """
    if key < node.key:
        return _with(node, _remove(node.left, key), node.right)
    if node.key < key:
        return _with(node, node.left, _remove(node.right, key))
    return _merge(node.left, node.right)


class PersistentMap:
    """An immutable sorted map: a treap with hash priorities and Merkle hashes.

    Every "update" (:meth:`set`, :meth:`delete`) returns a new map and leaves
    this one exactly as it was; the two share every node off the copied
    path. Because priorities are ``stable_hash64(key)`` with ties broken by
    key, the tree shape depends only on the key set, so equal contents give
    equal :attr:`root_hash` whatever the history.

    Args:
        items: Optional initial contents: a mapping, or an iterable of
            ``(key, value)`` pairs inserted in order (a later pair for the
            same key wins).

    Raises:
        TypeError: If ``items`` is not a mapping or an iterable of pairs, or
            the keys are not mutually comparable.

    Time Complexity:
        O(m log n) expected to build from m pairs; :meth:`get`, :meth:`set`
        and :meth:`delete` O(log n) expected; :meth:`items` and
        :meth:`iter_nodes` O(n).

    Space Complexity:
        n nodes for one map; each update adds O(log n) expected new nodes.

    Examples:
        >>> m1 = PersistentMap().set("b.txt", "h2").set("a.txt", "h1")
        >>> m2 = m1.set("b.txt", "h3")
        >>> m1.get("b.txt"), m2.get("b.txt"), len(m2)
        ('h2', 'h3', 2)
        >>> m2.delete("a.txt").items()
        [('b.txt', 'h3')]
        >>> PersistentMap().root is None, PersistentMap().root_hash == EMPTY_MAP_HASH
        (True, True)
    """

    __slots__ = ("_root", "_size")

    def __init__(self, items: Any = None) -> None:
        root: Optional[MapNode] = None
        size = 0
        if items is not None:
            pairs = items.items() if hasattr(items, "items") else items
            for key, value in pairs:
                existing = _find(root, key)
                if existing is None:
                    root = _insert(root, key, value, stable_hash64(key))
                    size += 1
                else:
                    root = _replace(root, key, value)
        self._root: Optional[MapNode] = root
        self._size: int = size

    @classmethod
    def _wrap(cls, root: Optional[MapNode], size: int) -> "PersistentMap":
        """Make a map around an existing root without copying anything.

        Args:
            root: The treap root (or ``None``).
            size: The number of keys under ``root``.

        Returns:
            A new :class:`PersistentMap` sharing ``root``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> m = PersistentMap({"a": "1"})
            >>> PersistentMap._wrap(m.root, 1).root is m.root
            True
        """
        new = cls.__new__(cls)
        new._root = root
        new._size = size
        return new

    # ------------------------------------------------------------ updates

    def set(self, key: Any, value: Any) -> "PersistentMap":
        """Return a new map with ``key`` mapped to ``value``.

        A new key is inserted (search path copied, plus the split below the
        point where its priority places it). An existing key keeps its node
        position, so only its search path is copied with the new value. If
        the key already holds an equal value of the same type and the same
        canonical encoding (so every hash would stay the same), the new map
        shares this map's root and no node is created. Equal values with
        different encodings, such as ``0.0`` and ``-0.0``, are replaced, so
        the stored value and the hash never depend on what was set before.

        Args:
            key: The key (mutually comparable with the other keys).
            value: The value.

        Returns:
            A new :class:`PersistentMap`; ``self`` is unchanged.

        Raises:
            TypeError: If ``key`` cannot be compared with the existing keys.

        Time Complexity:
            O(log n) expected.

        Space Complexity:
            O(log n) expected new nodes.

        Examples:
            >>> a = PersistentMap({"x": "1"})
            >>> b = a.set("x", "2")
            >>> a.get("x"), b.get("x"), a.root_hash == b.root_hash
            ('1', '2', False)
        """
        existing = _find(self._root, key)
        if existing is None:
            root = _insert(self._root, key, value, stable_hash64(key))
            return PersistentMap._wrap(root, self._size + 1)
        old = existing.value
        if (
            type(old) is type(value)
            and old == value
            and _encode_field(old) == _encode_field(value)
        ):
            return PersistentMap._wrap(self._root, self._size)
        return PersistentMap._wrap(_replace(self._root, key, value), self._size)

    def delete(self, key: Any) -> "PersistentMap":
        """Return a new map without ``key``.

        The key's node is replaced by the merge of its two subtrees; the
        search path and the merge path are copied.

        Args:
            key: A key that is in the map.

        Returns:
            A new :class:`PersistentMap`; ``self`` is unchanged.

        Raises:
            KeyError: If ``key`` is not in the map.

        Time Complexity:
            O(log n) expected.

        Space Complexity:
            O(log n) expected new nodes.

        Examples:
            >>> m = PersistentMap({"a": "1", "b": "2"})
            >>> m.delete("a").items(), m.items()
            ([('b', '2')], [('a', '1'), ('b', '2')])
            >>> m.delete("z")
            Traceback (most recent call last):
                ...
            KeyError: 'z'
        """
        if _find(self._root, key) is None:
            raise KeyError(key)
        return PersistentMap._wrap(_remove(self._root, key), self._size - 1)

    # ------------------------------------------------------------ queries

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key``, or ``default`` if it is absent.

        Args:
            key: The key to look up.
            default: Returned when the key is absent.

        Returns:
            The value or ``default``.

        Raises:
            TypeError: If ``key`` cannot be compared with the stored keys.

        Time Complexity:
            O(log n) expected.

        Space Complexity:
            O(1).

        Examples:
            >>> m = PersistentMap({"a": "1"})
            >>> m.get("a"), m.get("b"), m.get("b", "none")
            ('1', None, 'none')
        """
        node = _find(self._root, key)
        return default if node is None else node.value

    def items(self) -> List[Tuple[Any, Any]]:
        """Return all ``(key, value)`` pairs sorted by key (an in-order walk).

        Returns:
            A new list of pairs in increasing key order.

        Raises:
            Nothing.

        Time Complexity:
            O(n).

        Space Complexity:
            O(n) for the result, O(depth) for the walk's stack.

        Examples:
            >>> PersistentMap([("c", "3"), ("a", "1"), ("b", "2")]).items()
            [('a', '1'), ('b', '2'), ('c', '3')]
        """
        out: List[Tuple[Any, Any]] = []
        stack: List[MapNode] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            out.append((node.key, node.value))
            node = node.right
        return out

    def iter_nodes(self) -> Iterator[MapNode]:
        """Yield every node of this map in pre-order (node, left subtree, right subtree).

        Yields:
            Each :class:`MapNode` once, the root first.

        Raises:
            Nothing.

        Time Complexity:
            O(n) for the whole walk.

        Space Complexity:
            O(depth) for the walk's stack.

        Examples:
            >>> m = PersistentMap({"a": "1", "b": "2", "c": "3"})
            >>> nodes = list(m.iter_nodes())
            >>> len(nodes), nodes[0] is m.root
            (3, True)
        """
        stack: List[MapNode] = [] if self._root is None else [self._root]
        while stack:
            node = stack.pop()
            yield node
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    @property
    def root(self) -> Optional[MapNode]:
        """The root :class:`MapNode`, or ``None`` for the empty map.

        Returns:
            The immutable root node.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> PersistentMap({"a": "1"}).root.key
            'a'
        """
        return self._root

    @property
    def root_hash(self) -> str:
        """The Merkle hash of the whole map, as 128 hex characters.

        Equal ``(key, value)`` contents give equal hashes whatever order they
        were built in. The empty map's hash is :data:`EMPTY_MAP_HASH`,
        ``blake2b(b"").hexdigest()``.

        Returns:
            ``root.hash``, or :data:`EMPTY_MAP_HASH` when empty.

        Raises:
            Nothing.

        Time Complexity:
            O(1): the hash was computed when the root was built.

        Space Complexity:
            O(1).

        Examples:
            >>> PersistentMap().root_hash[:16]
            '786a02f742015903'
            >>> PersistentMap({"a": "1"}).root_hash == PersistentMap({"a": "2"}).root_hash
            False
        """
        return EMPTY_MAP_HASH if self._root is None else self._root.hash

    def __len__(self) -> int:
        """Return the number of keys.

        Returns:
            The key count.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(PersistentMap({"a": "1", "b": "2"}))
            2
        """
        return self._size

    def __contains__(self, key: Any) -> bool:
        """Return True if ``key`` is in the map (even when its value is ``None``).

        Args:
            key: The key to look for.

        Returns:
            Whether the key is present.

        Raises:
            TypeError: If ``key`` cannot be compared with the stored keys.

        Time Complexity:
            O(log n) expected.

        Space Complexity:
            O(1).

        Examples:
            >>> m = PersistentMap({"a": None})
            >>> "a" in m, "b" in m
            (True, False)
        """
        return _find(self._root, key) is not None

    def __repr__(self) -> str:
        """Return the size and the first 12 hex digits of the root hash.

        Returns:
            For example ``PersistentMap(size=2, root_hash='3f1a...')``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> repr(PersistentMap())
            "PersistentMap(size=0, root_hash='786a02f74201'...)"
        """
        return f"PersistentMap(size={self._size}, root_hash={self.root_hash[:12]!r}...)"
