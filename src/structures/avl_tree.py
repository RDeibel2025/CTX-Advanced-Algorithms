"""AVL tree: a binary search tree that rebalances itself on every change.

A plain binary search tree is only as good as the order its keys arrive
in. Insert sorted keys and it degenerates into a linked list, and every
operation becomes O(n). An AVL tree prevents that with one rule, checked at
every node after every insertion and deletion: the heights of the two
subtrees may differ by at most one. When a change breaks the rule, a
rotation - a constant-time local rearrangement that preserves key order -
restores it.

That single rule bounds the height. The sparsest possible AVL tree of
height h is a "Fibonacci tree" with F(h + 2) - 1 nodes, so a tree of n nodes
can never be taller than about 1.44 log2(n), and every operation below walks
at most one root-to-leaf path.

=============== ============ =================================================
Operation       Worst case   Notes
=============== ============ =================================================
search          O(log n)     iterative; one comparison per level
insert          O(log n)     at most one rebalancing (single or double
                             rotation) per insert
delete          O(log n)     may rebalance at *every* level on the way up -
                             O(log n) rotations for one delete
in_order        O(n)         yields keys in sorted order
height          O(1)         read from the root; heights are stored
=============== ============ =================================================

Heights are stored on the nodes and every balance factor is derived from
the children's stored heights. Recomputing heights by traversal would make
every operation O(n) and quietly defeat the whole structure.

Keys are compared with ``<`` only, so any mutually orderable type works.
Keys are unique: inserting an existing key updates its value.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable, Iterator, List, Optional, Tuple

__all__ = ["AVLNode", "AVLTree", "avl_max_height", "perfect_height"]


class AVLNode:
    """One node of an :class:`AVLTree`.

    Attributes:
        key: The node's key. Orders the tree.
        value: The payload stored against the key.
        left: Subtree of smaller keys, or None.
        right: Subtree of larger keys, or None.
        height: Nodes on the longest downward path from here, counting this
            node, so a leaf has height 1. Maintained on every rotation and on
            the way back up every insert and delete.
    """

    __slots__ = ("key", "value", "left", "right", "height")

    def __init__(self, key: Any, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None
        self.height = 1

    @property
    def balance_factor(self) -> int:
        """Left height minus right height. Always -1, 0 or 1 in a valid tree.

        Derived from the children's stored heights, so it costs O(1) and is
        correct whenever the heights are.
        """
        return _height(self.left) - _height(self.right)

    def __repr__(self) -> str:
        return f"AVLNode(key={self.key!r}, height={self.height})"


def _height(node: Optional[AVLNode]) -> int:
    """Height of a possibly-empty subtree. An empty subtree has height 0."""
    return node.height if node is not None else 0


def _update_height(node: AVLNode) -> None:
    """Recompute a node's height from its children's stored heights. O(1)."""
    left = node.left.height if node.left is not None else 0
    right = node.right.height if node.right is not None else 0
    node.height = 1 + (left if left > right else right)


def avl_max_height(n: int) -> int:
    """Tallest height an AVL tree with ``n`` nodes can have.

    The sparsest AVL tree of height h has N(h) = N(h - 1) + N(h - 2) + 1
    nodes, with N(0) = 0 and N(1) = 1 - one more than a Fibonacci number.
    The answer is the largest h with N(h) <= n. Computed exactly rather
    than from the closed-form 1.44 log2(n) approximation, so tests can
    assert against it without a fudge factor.

    Args:
        n: Number of nodes. Must be non-negative.

    Returns:
        The maximum possible height, counting nodes (a leaf is height 1).

    Examples:
        >>> [avl_max_height(n) for n in (0, 1, 2, 4, 7, 12, 20)]
        [0, 1, 2, 3, 4, 5, 6]
        >>> avl_max_height(1_000_000)
        28
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    smaller, larger, height = 0, 1, 0   # N(h - 1), N(h), h
    while larger <= n:
        smaller, larger = larger, larger + smaller + 1
        height += 1
    return height


def perfect_height(n: int) -> int:
    """Height of a perfectly balanced binary tree with ``n`` nodes.

    The lower bound for *any* binary search tree: ceil(log2(n + 1)).

    Examples:
        >>> [perfect_height(n) for n in (0, 1, 3, 4, 7, 1_000_000)]
        [0, 1, 2, 3, 3, 20]
    """
    return math.ceil(math.log2(n + 1)) if n > 0 else 0


class AVLTree:
    """Self-balancing binary search tree with O(log n) search, insert and delete.

    Args:
        items: Optional initial contents, inserted in order. A mapping such
            as a ``dict`` supplies key-value pairs; any other iterable
            supplies keys, each stored with value None. The choice is made by
            type rather than by guessing, so a tuple is always a key:
            ``AVLTree([(1, 2)])`` stores the key ``(1, 2)``.

    Attributes:
        rotation_count: Individual rotations performed so far. A double
            rotation counts as two. Instrumentation for the benchmark.
        rebalance_count: Nodes at which a rotation was needed, so a double
            rotation counts as one. The per-delete difference in this counter
            shows deletion rebalancing at more than one level.

    Examples:
        >>> tree = AVLTree()
        >>> for key in [1, 2, 3, 4, 5, 6, 7]:      # sorted input
        ...     tree.insert(key, str(key))
        >>> tree.height()                          # a plain BST would be 7
        3
        >>> tree.search(4)
        '4'
        >>> tree.delete(4)
        >>> 4 in tree, len(tree)
        (False, 6)
        >>> tree.in_order()
        [1, 2, 3, 5, 6, 7]
    """

    __slots__ = ("_root", "_size", "rotation_count", "rebalance_count")

    def __init__(self, items: Optional[Iterable[Any]] = None) -> None:
        self._root: Optional[AVLNode] = None
        self._size = 0
        self.rotation_count = 0
        self.rebalance_count = 0
        if items is None:
            return
        if isinstance(items, Mapping):
            for key, value in items.items():
                self.insert(key, value)
        else:
            for key in items:
                self.insert(key)

    # ------------------------------------------------------------------
    # Size and inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of keys stored. O(1)."""
        return self._size

    def __contains__(self, key: Any) -> bool:
        """Membership test. O(log n)."""
        return self._find(key) is not None

    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys in sorted order."""
        for key, _value in self.items():
            yield key

    def __repr__(self) -> str:
        return f"AVLTree(size={self._size}, height={self.height()})"

    @property
    def root(self) -> Optional[AVLNode]:
        """The root node, for inspection. None when the tree is empty."""
        return self._root

    def is_empty(self) -> bool:
        """Return True if the tree holds no keys.

        Examples:
            >>> AVLTree().is_empty()
            True
        """
        return self._root is None

    def height(self) -> int:
        """Height of the tree, counting nodes: 0 when empty, 1 for one node.

        Read from the root's stored height, so O(1). For debugging balance:
        compare against :func:`avl_max_height` and :func:`perfect_height`.

        Examples:
            >>> tree = AVLTree(range(1000))
            >>> perfect_height(1000) <= tree.height() <= avl_max_height(1000)
            True
        """
        return _height(self._root)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _find(self, key: Any) -> Optional[AVLNode]:
        """Return the node holding ``key``, or None. Iterative."""
        node = self._root
        while node is not None:
            if key < node.key:
                node = node.left
            elif node.key < key:
                node = node.right
            else:
                return node
        return None

    def search(self, key: Any, default: Any = None) -> Any:
        """Return the value stored at ``key``, or ``default`` if it is absent.

        Mirrors ``dict.get``. Because a stored value may itself be None, use
        ``key in tree`` when you need to distinguish "absent" from "stored
        None".

        Args:
            key: Key to look up.
            default: Returned when ``key`` is not in the tree.

        Returns:
            The stored value, or ``default``.

        Time Complexity:
            O(log n) - one comparison per level, and the height is at most
            about 1.44 log2(n).

        Examples:
            >>> tree = AVLTree({"a": 1, "b": 2})
            >>> tree.search("b"), tree.search("z"), tree.search("z", -1)
            (2, None, -1)
        """
        node = self._find(key)
        return node.value if node is not None else default

    def min_key(self) -> Any:
        """Smallest key. Raises KeyError on an empty tree. O(log n)."""
        if self._root is None:
            raise KeyError("min_key of an empty tree")
        node = self._root
        while node.left is not None:
            node = node.left
        return node.key

    def max_key(self) -> Any:
        """Largest key. Raises KeyError on an empty tree. O(log n)."""
        if self._root is None:
            raise KeyError("max_key of an empty tree")
        node = self._root
        while node.right is not None:
            node = node.right
        return node.key

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield ``(key, value)`` pairs in ascending key order.

        Uses an explicit stack rather than recursion, so iteration can be
        paused, and never approaches the recursion limit.

        Time Complexity:
            O(n) for a full traversal.
        """
        stack: List[AVLNode] = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key, node.value
            node = node.right

    def in_order(self) -> List[Any]:
        """Return every key in ascending order - an in-order traversal.

        Time Complexity:
            O(n).

        Examples:
            >>> AVLTree([5, 3, 8, 1, 4]).in_order()
            [1, 3, 4, 5, 8]
        """
        return [key for key, _value in self.items()]

    # ------------------------------------------------------------------
    # Rotations and rebalancing
    # ------------------------------------------------------------------
    def _rotate_right(self, top: AVLNode) -> AVLNode:
        """Rotate ``top`` down to the right; its left child becomes the root.

        ::

                top              pivot
               /    \\           /     \\
            pivot    C   ->    A      top
            /   \\                    /   \\
           A     B                  B     C

        Key order A < pivot < B < top < C is preserved. O(1).
        """
        pivot = top.left
        assert pivot is not None
        top.left = pivot.right
        pivot.right = top
        _update_height(top)      # top is now below pivot: update it first
        _update_height(pivot)
        self.rotation_count += 1
        return pivot

    def _rotate_left(self, top: AVLNode) -> AVLNode:
        """Mirror image of :meth:`_rotate_right`. O(1)."""
        pivot = top.right
        assert pivot is not None
        top.right = pivot.left
        pivot.left = top
        _update_height(top)
        _update_height(pivot)
        self.rotation_count += 1
        return pivot

    def _rebalance(self, node: AVLNode) -> AVLNode:
        """Restore the AVL rule at ``node`` and return the subtree's new root.

        Called on every node on the way back up an insert or delete path.
        The four cases:

        * **LL** - left-heavy, and the left child is not right-heavy:
          one right rotation.
        * **LR** - left-heavy, and the left child is right-heavy: rotate the
          child left, then this node right.
        * **RR** and **RL** - the mirror images.

        The test for LL is ``balance >= 0`` on the child, not ``> 0``. The
        child can only be exactly balanced after a *deletion*, and a single
        rotation is then the correct fix; testing ``> 0`` would send that case
        to the double rotation and leave the tree unbalanced.
        """
        _update_height(node)
        balance = _height(node.left) - _height(node.right)

        if balance > 1:
            left = node.left
            assert left is not None
            self.rebalance_count += 1
            if _height(left.left) < _height(left.right):   # LR
                node.left = self._rotate_left(left)
            return self._rotate_right(node)                # LL

        if balance < -1:
            right = node.right
            assert right is not None
            self.rebalance_count += 1
            if _height(right.right) < _height(right.left):  # RL
                node.right = self._rotate_right(right)
            return self._rotate_left(node)                  # RR

        return node

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def insert(self, key: Any, value: Any = None) -> None:
        """Insert ``key`` with ``value``, or update the value if ``key`` exists.

        Descends to the insertion point, attaches a new leaf, and rebalances
        every node on the way back up. After an insertion at most one node
        actually needs a rotation: fixing the lowest unbalanced node restores
        that subtree to its pre-insert height, so nothing above it changes.

        Args:
            key: Key to insert. Must be orderable against existing keys.
            value: Payload to store. Defaults to None, for set-like use.

        Raises:
            TypeError: If ``key`` cannot be compared with existing keys.

        Time Complexity:
            O(log n). Recursion depth is the tree height, about 1.44 log2(n)
            at worst - under 30 even at a million keys.

        Examples:
            >>> tree = AVLTree()
            >>> tree.insert(3, "c"); tree.insert(1, "a"); tree.insert(2, "b")
            >>> tree.root.key, tree.rotation_count     # LR case: two rotations
            (2, 2)
            >>> tree.insert(2, "B")                    # existing key: update
            >>> tree.search(2), len(tree)
            ('B', 3)
        """
        self._root = self._insert(self._root, key, value)

    def _insert(self, node: Optional[AVLNode], key: Any, value: Any) -> AVLNode:
        if node is None:
            self._size += 1
            return AVLNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif node.key < key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value
            return node
        return self._rebalance(node)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete(self, key: Any) -> None:
        """Remove ``key`` and rebalance every ancestor of the removed node.

        A node with two children is replaced by its in-order successor (the
        smallest key in its right subtree), which is then removed from that
        subtree - a node with at most one child, so the removal is simple.

        Unlike insertion, deletion can require a rotation at **every** level
        on the way back up. A rotation after a delete can leave the subtree
        one level shorter than it was, which unbalances the parent in turn.
        The recursion rebalances each ancestor as it unwinds, so no case is
        missed; stopping after the first rotation is the classic bug.

        Args:
            key: Key to remove.

        Raises:
            KeyError: If ``key`` is not in the tree. The tree is unchanged.

        Time Complexity:
            O(log n), including up to O(log n) rotations.

        Examples:
            >>> tree = AVLTree(range(10))
            >>> tree.delete(3)
            >>> tree.in_order()
            [0, 1, 2, 4, 5, 6, 7, 8, 9]
            >>> tree.delete(3)
            Traceback (most recent call last):
                ...
            KeyError: 3
        """
        # _delete raises KeyError before assigning anything back, so a failed
        # delete leaves both the tree and the size untouched.
        self._root = self._delete(self._root, key)
        self._size -= 1

    def _delete(self, node: Optional[AVLNode], key: Any) -> Optional[AVLNode]:
        if node is None:
            raise KeyError(key)
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif node.key < key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.value = successor.key, successor.value
            node.right = self._delete(node.right, successor.key)
        return self._rebalance(node)

    # ------------------------------------------------------------------
    # Invariant checking
    # ------------------------------------------------------------------
    def validate(self) -> bool:
        """Check every AVL and BST invariant, raising on the first violation.

        Verifies, for every node: keys are in strict BST order, the stored
        height equals the height computed from scratch, and the balance
        factor is -1, 0 or 1. Also checks the stored size matches the node
        count. Used by the test suite after every operation.

        Returns:
            True when every invariant holds.

        Raises:
            AssertionError: Naming the first violated invariant and where.

        Time Complexity:
            O(n).

        Examples:
            >>> AVLTree(range(100)).validate()
            True
        """
        count = self._check(self._root, None, None)[1]
        if count != self._size:
            raise AssertionError(f"size is {self._size} but tree holds {count} nodes")
        return True

    def _check(
        self, node: Optional[AVLNode], low: Any, high: Any
    ) -> Tuple[int, int]:
        """Return ``(computed_height, node_count)``; raise on any violation."""
        if node is None:
            return 0, 0
        if low is not None and not low < node.key:
            raise AssertionError(f"BST order broken: {node.key!r} is not > {low!r}")
        if high is not None and not node.key < high:
            raise AssertionError(f"BST order broken: {node.key!r} is not < {high!r}")
        left_height, left_count = self._check(node.left, low, node.key)
        right_height, right_count = self._check(node.right, node.key, high)
        computed = 1 + max(left_height, right_height)
        if node.height != computed:
            raise AssertionError(
                f"stale height at {node.key!r}: stored {node.height}, actual {computed}"
            )
        if abs(left_height - right_height) > 1:
            raise AssertionError(
                f"unbalanced at {node.key!r}: balance factor "
                f"{left_height - right_height}"
            )
        return computed, left_count + right_count + 1
