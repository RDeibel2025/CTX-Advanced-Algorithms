"""BFS (heap) and van Emde Boas layouts of a complete binary search tree, and their block cost.

The question
------------
A complete binary search tree of height h (``2^h - 1`` keys, h levels) has
to be stored in one flat array. Which order of the nodes makes a
root-to-leaf search touch the fewest memory blocks of B words, when the
code does not know B?

* **BFS (heap) order** (:func:`bfs_layout`) stores the nodes level by
  level: the node with 1-based BFS id b is at index ``b - 1`` and its
  children are ids ``2b`` and ``2b + 1``. The top ``log2 B`` levels share the
  first block or so, but below that every level of the path is in a
  different block: about ``h - log2 B + 1`` blocks per search.
* **van Emde Boas order** (:func:`veb_order`) halves the tree by height: a
  top tree of ``h // 2`` levels and ``2^(h // 2)`` bottom trees of
  ``h - h // 2`` levels each. It stores the top tree first, then each bottom
  tree left to right, each laid out the same way recursively. At the
  recursion level where subtrees have between ``sqrt(B)`` and B nodes, each
  subtree occupies at most two blocks, and a path crosses at most
  ``2 log_B n`` of them, so a search touches O(log_B n) blocks for every B
  at once (Prokop 1999; Frigo et al. 1999).

Node numbering and conventions
------------------------------
Nodes are named by their 1-based BFS id b (root 1, children ``2b`` and
``2b + 1``), and depths are 0-based (the root has depth 0, the leaves depth
``h - 1``). A *layout* is a list of BFS ids in storage order;
``layout[p]`` is the node stored at array position p (0-based). The key of
node b in a complete BST is the in-order rank
``(2t + 1) * 2^(h - 1 - d) - 1`` of ``sorted_keys``, where d is the depth of
b and ``t = b - 2^d`` its offset within the level. A position p lies in
block ``p // B``.

Navigating the vEB layout by hand
---------------------------------
:func:`search` never builds a BFS-id-to-position dict. For BFS order the
child position is ``2b`` or ``2b + 1`` minus 1. For vEB order it uses three
small tables per height, of length h, built once by :func:`_veb_tables`
(Brodal, Fagerberg and Jacob, 2002). Every depth ``d >= 1`` is the root depth
of the bottom trees of exactly one split in the recursion; for that split the
tables hold the top tree's size ``T[d] = 2^t - 1`` (t levels), each bottom
tree's size ``S[d]`` and the top tree's root depth ``R[d]``. If ``P[d']`` is
the position of the path's node at depth d', the node at depth d (BFS id b)
is at::

    P[d] = P[R[d]] + T[d] + (b & T[d]) * S[d]

because the top tree is stored first, the low t bits of b number the bottom
trees below it from left to right, and each bottom tree is stored
contiguously. That is O(1) work per step and O(h) memory per height.

Examples:
    >>> veb_order(3)
    [1, 2, 4, 5, 3, 6, 7]
    >>> build_veb_tree([10, 20, 30, 40, 50, 60, 70])
    [40, 20, 10, 30, 60, 50, 70]
    >>> search(build_veb_tree([10, 20, 30, 40, 50, 60, 70]), 50, "veb")
    (True, [0, 4, 5])

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numbers
from typing import Any, Dict, List, Sequence, Tuple

from src.utils.random_utils import make_rng

__all__ = [
    "LAYOUTS",
    "bfs_layout",
    "blocks_per_search",
    "build_bfs_tree",
    "build_veb_tree",
    "search",
    "veb_order",
]

#: The two layout names accepted by :func:`search` and :func:`blocks_per_search`.
LAYOUTS: Tuple[str, str] = ("bfs", "veb")

#: Per-height vEB navigation tables, built on first use by :func:`_veb_tables`.
_VEB_TABLES: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}


# ---------------------------------------------------------------------------
# Checks and small helpers
# ---------------------------------------------------------------------------


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` if it is an int of at least ``minimum``, else raise.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.
        minimum: The smallest allowed value.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an int (``bool`` rejected).
        ValueError: If ``value < minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_int(4, "h", 0)
        4
        >>> _check_int(0, "B", 1)
        Traceback (most recent call last):
        ...
        ValueError: B must be at least 1, got 0
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def _check_layout(layout: Any) -> str:
    """Return ``layout`` if it is ``"bfs"`` or ``"veb"``.

    Args:
        layout: The layout name.

    Returns:
        ``layout`` unchanged.

    Raises:
        ValueError: For any other value.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_layout("veb")
        'veb'
        >>> _check_layout("inorder")
        Traceback (most recent call last):
        ...
        ValueError: layout must be one of ('bfs', 'veb'), got 'inorder'
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return layout


def _height_of_size(n: int) -> int:
    """Return h such that ``n == 2^h - 1``, the height of a complete tree of n nodes.

    Args:
        n: A node count.

    Returns:
        The height h (0 for an empty tree).

    Raises:
        ValueError: If n is not of the form ``2^h - 1``.

    Time Complexity:
        O(log n) shifts.

    Space Complexity:
        O(1).

    Examples:
        >>> _height_of_size(7), _height_of_size(0), _height_of_size(1023)
        (3, 0, 10)
        >>> _height_of_size(6)
        Traceback (most recent call last):
        ...
        ValueError: a complete tree has 2^h - 1 nodes, got 6
    """
    m = n + 1
    h = 0
    while m > 1:
        if m & 1:
            raise ValueError(f"a complete tree has 2^h - 1 nodes, got {n}")
        m >>= 1
        h += 1
    if m != 1:
        raise ValueError(f"a complete tree has 2^h - 1 nodes, got {n}")
    return h


def _inorder_table(h: int) -> List[int]:
    """Return the in-order rank of every node of a complete tree, indexed by BFS id.

    Node b at depth d with offset ``t = b - 2^d`` has in-order rank
    ``(2t + 1) * 2^(h - 1 - d) - 1``: its left subtree holds
    ``2^(h - 1 - d) - 1`` keys, and the t subtrees to its left at the same
    depth hold ``2^(h - d) - 1`` keys plus their parent separator each.

    Args:
        h: The tree height, at least 0.

    Returns:
        A list of length ``2^h``; entry 0 is unused (-1).

    Raises:
        Nothing.

    Time Complexity:
        Theta(2^h).

    Space Complexity:
        Theta(2^h).

    Examples:
        >>> _inorder_table(3)
        [-1, 3, 1, 5, 0, 2, 4, 6]
    """
    table = [-1] * (1 << h)
    for d in range(h):
        first = 1 << d
        span = 1 << (h - 1 - d)
        for t in range(first):
            table[first + t] = (2 * t + 1) * span - 1
    return table


def _checked_keys(sorted_keys: Sequence[Any]) -> Tuple[List[Any], int]:
    """Copy the keys, check they are strictly increasing and number ``2^h - 1``.

    Args:
        sorted_keys: The keys in increasing order.

    Returns:
        ``(keys, h)``.

    Raises:
        ValueError: If the count is not ``2^h - 1`` or the keys are not
            strictly increasing.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the copy.

    Examples:
        >>> _checked_keys([1, 2, 3])
        ([1, 2, 3], 2)
        >>> _checked_keys([1, 3, 2])
        Traceback (most recent call last):
        ...
        ValueError: sorted_keys must be strictly increasing; position 2 breaks the order
    """
    keys = list(sorted_keys)
    h = _height_of_size(len(keys))
    for idx in range(1, len(keys)):
        if not keys[idx - 1] < keys[idx]:
            raise ValueError(
                f"sorted_keys must be strictly increasing; position {idx} breaks the order"
            )
    return keys, h


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def bfs_layout(h: int) -> List[int]:
    """Return the BFS ids in heap (level) order: node b is stored at index ``b - 1``.

    Args:
        h: The tree height (number of levels), at least 0.

    Returns:
        ``[1, 2, ..., 2^h - 1]``.

    Raises:
        TypeError: If ``h`` is not an int.
        ValueError: If ``h`` is negative.

    Time Complexity:
        Theta(2^h).

    Space Complexity:
        Theta(2^h).

    Examples:
        >>> bfs_layout(3)
        [1, 2, 3, 4, 5, 6, 7]
        >>> bfs_layout(0)
        []
    """
    h = _check_int(h, "h", 0)
    return list(range(1, 1 << h))


def _veb_append(root: int, h: int, out: List[int]) -> None:
    """Append the BFS ids of the height-h subtree under ``root`` in vEB order.

    The top tree has ``h // 2`` levels and is laid out first (recursively);
    then each of the ``2^(h // 2)`` bottom trees of ``h - h // 2`` levels,
    rooted at ``root * 2^(h // 2) + i`` for ``i = 0 ..``, left to right.

    Args:
        root: The BFS id of the subtree root.
        h: The subtree height, at least 1.
        out: The list to append to.

    Returns:
        None.

    Raises:
        Nothing.

    Time Complexity:
        Theta(2^h).

    Space Complexity:
        O(log h) recursion depth beyond ``out``.

    Examples:
        >>> out = []
        >>> _veb_append(2, 2, out)
        >>> out
        [2, 4, 5]
    """
    if h == 1:
        out.append(root)
        return
    top = h // 2
    bottom = h - top
    _veb_append(root, top, out)
    first_bottom_root = root << top
    for i in range(1 << top):
        _veb_append(first_bottom_root + i, bottom, out)


def veb_order(h: int) -> List[int]:
    """Return the BFS ids of a complete tree of height h in van Emde Boas order.

    Split rule: the top tree has ``h // 2`` levels and each bottom tree
    ``h - h // 2`` levels; lay out the top tree recursively, then each bottom
    tree recursively, left to right.

    Args:
        h: The tree height (number of levels), at least 0.

    Returns:
        A permutation of ``1 .. 2^h - 1``; entry p is the node stored at
        position p.

    Raises:
        TypeError: If ``h`` is not an int.
        ValueError: If ``h`` is negative.

    Time Complexity:
        Theta(2^h).

    Space Complexity:
        Theta(2^h) for the result plus O(log h) recursion depth.

    Examples:
        >>> veb_order(3)
        [1, 2, 4, 5, 3, 6, 7]
        >>> veb_order(4)
        [1, 2, 3, 4, 8, 9, 5, 10, 11, 6, 12, 13, 7, 14, 15]
        >>> veb_order(5)[:11]
        [1, 2, 3, 4, 8, 16, 17, 9, 18, 19, 5]
        >>> veb_order(0)
        []
    """
    h = _check_int(h, "h", 0)
    out: List[int] = []
    if h > 0:
        _veb_append(1, h, out)
    return out


def build_bfs_tree(sorted_keys: Sequence[Any]) -> List[Any]:
    """Store the complete BST over ``sorted_keys`` in BFS (heap) order.

    Args:
        sorted_keys: ``2^h - 1`` strictly increasing keys.

    Returns:
        A list where index ``b - 1`` holds the key of BFS node b.

    Raises:
        ValueError: If the key count is not ``2^h - 1`` or the keys are not
            strictly increasing.

    Time Complexity:
        Theta(n).

    Space Complexity:
        Theta(n).

    Examples:
        >>> build_bfs_tree([10, 20, 30, 40, 50, 60, 70])
        [40, 20, 60, 10, 30, 50, 70]
        >>> build_bfs_tree([])
        []
    """
    keys, h = _checked_keys(sorted_keys)
    rank = _inorder_table(h)
    return [keys[rank[b]] for b in range(1, 1 << h)]


def build_veb_tree(sorted_keys: Sequence[Any]) -> List[Any]:
    """Store the complete BST over ``sorted_keys`` in van Emde Boas order.

    Args:
        sorted_keys: ``2^h - 1`` strictly increasing keys.

    Returns:
        A list where position p holds the key of node ``veb_order(h)[p]``.

    Raises:
        ValueError: If the key count is not ``2^h - 1`` or the keys are not
            strictly increasing.

    Time Complexity:
        Theta(n).

    Space Complexity:
        Theta(n).

    Examples:
        >>> build_veb_tree([10, 20, 30, 40, 50, 60, 70])
        [40, 20, 10, 30, 60, 50, 70]
        >>> build_veb_tree([5])
        [5]
    """
    keys, h = _checked_keys(sorted_keys)
    rank = _inorder_table(h)
    return [keys[rank[b]] for b in veb_order(h)]


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def _fill_veb_tables(
    d0: int, h: int, top: List[int], size: List[int], root: List[int]
) -> None:
    """Record, for the split of the depth range ``[d0, d0 + h)``, its bottom-root depth entry.

    Args:
        d0: The depth of the subtree root.
        h: The subtree height.
        top: Filled with the top tree size per bottom-root depth.
        size: Filled with the bottom tree size per bottom-root depth.
        root: Filled with the top tree root depth per bottom-root depth.

    Returns:
        None.

    Raises:
        Nothing.

    Time Complexity:
        O(h): one entry per depth.

    Space Complexity:
        O(log h) recursion depth.

    Examples:
        >>> t, s, r = [0] * 3, [0] * 3, [0] * 3
        >>> _fill_veb_tables(0, 3, t, s, r)
        >>> t, s, r
        ([0, 1, 1], [0, 3, 1], [0, 0, 1])
    """
    if h <= 1:
        return
    t = h // 2
    d = d0 + t
    top[d] = (1 << t) - 1
    size[d] = (1 << (h - t)) - 1
    root[d] = d0
    _fill_veb_tables(d0, t, top, size, root)
    _fill_veb_tables(d, h - t, top, size, root)


def _veb_tables(h: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Return the vEB navigation tables ``(T, S, R)`` for height h, cached per height.

    For each depth ``d >= 1``: ``T[d]`` is the size of the top tree whose
    bottom trees are rooted at depth d, ``S[d]`` the size of each of those
    bottom trees, and ``R[d]`` the depth of that top tree's root. Entry 0 is
    unused.

    Args:
        h: The tree height, at least 0.

    Returns:
        Three tuples of length h.

    Raises:
        Nothing.

    Time Complexity:
        O(h) the first time for each h, O(1) afterwards.

    Space Complexity:
        O(h) per height.

    Examples:
        >>> _veb_tables(4)                # depth 2 roots the 4 bottom trees of the top split
        ((0, 1, 3, 1), (0, 1, 3, 1), (0, 0, 0, 2))
    """
    tables = _VEB_TABLES.get(h)
    if tables is None:
        top = [0] * h
        size = [0] * h
        root = [0] * h
        _fill_veb_tables(0, h, top, size, root)
        tables = (tuple(top), tuple(size), tuple(root))
        _VEB_TABLES[h] = tables
    return tables


def _path_positions(node: int, depth: int, layout: str, h: int) -> List[int]:
    """Return the array positions of the root-to-``node`` path in a height-h layout.

    This is the same position arithmetic :func:`search` uses, driven by the
    bits of the target's BFS id instead of key comparisons: the ancestor of
    ``node`` at depth d is ``node >> (depth - d)``.

    Args:
        node: The target's BFS id.
        depth: The target's depth (0-based), so ``2^depth <= node < 2^(depth+1)``.
        layout: ``"bfs"`` or ``"veb"``.
        h: The tree height.

    Returns:
        ``depth + 1`` positions, root first.

    Raises:
        Nothing (the caller validates).

    Time Complexity:
        O(depth).

    Space Complexity:
        O(depth).

    Examples:
        >>> _path_positions(5, 2, "bfs", 3), _path_positions(5, 2, "veb", 3)
        ([0, 1, 4], [0, 1, 3])
    """
    if layout == "bfs":
        return [(node >> (depth - d)) - 1 for d in range(depth + 1)]
    top, size, root = _veb_tables(h)
    pos = [0] * (depth + 1)
    for d in range(1, depth + 1):
        b = node >> (depth - d)
        t = top[d]
        pos[d] = pos[root[d]] + t + (b & t) * size[d]
    return pos


def search(layout_array: Sequence[Any], key: Any, layout: str) -> Tuple[bool, List[int]]:
    """Search a complete BST stored in BFS or vEB order, computing child positions by hand.

    The descent compares ``key`` with the node at the current position and
    moves to the left child (``key`` smaller) or the right child. In BFS
    order the child of node b is at ``2b - 1`` or ``2b``. In vEB order the
    position comes from the per-height tables (see the module docstring), so
    no position dict is built.

    Args:
        layout_array: The keys as built by :func:`build_bfs_tree` or
            :func:`build_veb_tree` (length ``2^h - 1``).
        key: The key to look for.
        layout: ``"bfs"`` or ``"veb"``, the order ``layout_array`` is in.

    Returns:
        ``(found, path_positions)``: whether ``key`` is in the tree, and the
        array positions visited from the root, ending at the key's node or at
        the leaf where an unsuccessful search stops.

    Raises:
        ValueError: If ``layout`` is unknown or the length is not ``2^h - 1``.

    Time Complexity:
        O(h) = O(log n) comparisons. Block transfers per search: about
        ``h - log2 B + 1`` for BFS order, O(log_B n) for vEB order.

    Space Complexity:
        O(h) for the path and the vEB position stack.

    Examples:
        >>> keys = list(range(10, 160, 10))           # 15 keys, h = 4
        >>> search(build_bfs_tree(keys), 30, "bfs")
        (True, [0, 1, 3, 8])
        >>> search(build_veb_tree(keys), 30, "veb")
        (True, [0, 1, 3, 5])
        >>> search(build_veb_tree(keys), 35, "veb")
        (False, [0, 1, 3, 5])
        >>> search([], 1, "bfs")
        (False, [])
    """
    layout = _check_layout(layout)
    n = len(layout_array)
    h = _height_of_size(n)
    path: List[int] = []
    if n == 0:
        return False, path
    if layout == "bfs":
        b = 1
        while True:
            p = b - 1
            path.append(p)
            value = layout_array[p]
            if key == value:
                return True, path
            b = 2 * b if key < value else 2 * b + 1
            if b > n:
                return False, path
    top, size, root = _veb_tables(h)
    pos = [0] * h
    b = 1
    d = 0
    p = 0
    while True:
        path.append(p)
        value = layout_array[p]
        if key == value:
            return True, path
        b = 2 * b if key < value else 2 * b + 1
        d += 1
        if d == h:
            return False, path
        t = top[d]
        p = pos[root[d]] + t + (b & t) * size[d]
        pos[d] = p


def blocks_per_search(h: int, B: int, layout: str, trials: int, seed: int) -> float:
    """Mean number of distinct blocks touched by a random successful root-to-leaf search.

    Each trial draws a leaf uniformly with ``make_rng(seed)``
    (``rng.randrange(2^(h - 1))``), follows the root-to-leaf path a
    successful search for that leaf's key takes, and counts the distinct
    blocks ``p // B`` among its h positions. No tree of keys is built: the
    path positions come from the same arithmetic :func:`search` uses.

    Args:
        h: The tree height, at least 1.
        B: The block size in array slots, at least 1.
        layout: ``"bfs"`` or ``"veb"``.
        trials: The number of searches, at least 1.
        seed: The seed for :func:`src.utils.random_utils.make_rng`.

    Returns:
        The mean distinct-block count per search.

    Raises:
        TypeError: If ``h``, ``B``, ``trials`` or ``seed`` is not an int
            (``seed=None`` is rejected too, so the sample is always
            reproducible).
        ValueError: If ``layout`` is unknown or ``h``, ``B`` or ``trials`` is
            below 1.

    Time Complexity:
        O(trials * h).

    Space Complexity:
        O(h) per trial plus the O(h) tables.

    Examples:
        >>> blocks_per_search(10, 16, "bfs", 1000, 42)
        6.948
        >>> blocks_per_search(10, 16, "veb", 1000, 42) < 4
        True
        >>> blocks_per_search(4, 16, "veb", 10, 42)       # the whole tree fits in one block
        1.0
        >>> blocks_per_search(10, 16, "veb", 10, None)    # never an unseeded generator
        Traceback (most recent call last):
        ...
        TypeError: seed must be an int, got NoneType
    """
    h = _check_int(h, "h", 1)
    B = _check_int(B, "B", 1)
    trials = _check_int(trials, "trials", 1)
    layout = _check_layout(layout)
    # make_rng(None) would seed from the operating system; the experiment must
    # be reproducible, so the seed has to be a real int.
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    rng = make_rng(int(seed))
    first_leaf = 1 << (h - 1)
    total = 0
    for _ in range(trials):
        leaf = first_leaf + rng.randrange(first_leaf)
        blocks = {p // B for p in _path_positions(leaf, h - 1, layout, h)}
        total += len(blocks)
    return total / trials
