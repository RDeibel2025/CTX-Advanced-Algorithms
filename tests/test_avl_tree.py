"""Tests for the AVL tree.

The balance invariant is checked after **every** insert and delete in a
long randomised sequence, by an independent checker written here - not by
trusting ``AVLTree.validate()`` alone, so a bug in the tree and a matching
bug in its self-check cannot cancel out. Both are run.

Deletion gets particular attention. It is the operation most
implementations get wrong, because unlike insertion it can need a rotation
at every level on the way back up. :class:`TestDeletionRebalancing` builds
the sparsest possible AVL tree - a Fibonacci tree - where that cascade is
guaranteed, and checks the implementation performs it.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from src.structures.avl_tree import (
    AVLNode,
    AVLTree,
    _update_height,
    avl_max_height,
    perfect_height,
)


def check_avl(tree: AVLTree) -> None:
    """Independent invariant check: order, stored heights, balance, size."""

    def walk(node, low, high):
        if node is None:
            return 0, 0
        assert low is None or low < node.key, f"order broken at {node.key!r}"
        assert high is None or node.key < high, f"order broken at {node.key!r}"
        left_height, left_count = walk(node.left, low, node.key)
        right_height, right_count = walk(node.right, node.key, high)
        height = 1 + max(left_height, right_height)
        assert node.height == height, (
            f"stored height {node.height} != computed {height} at {node.key!r}"
        )
        assert node.balance_factor in (-1, 0, 1), (
            f"balance factor {node.balance_factor} at {node.key!r}"
        )
        return height, left_count + right_count + 1

    height, count = walk(tree.root, None, None)
    assert count == len(tree), f"size {len(tree)} but {count} nodes"
    assert height == tree.height()


def minimal_node_count(height: int) -> int:
    """Nodes in the sparsest AVL tree of ``height``: N(h) = N(h-1) + N(h-2) + 1."""
    counts = [0, 1]
    while len(counts) <= height:
        counts.append(counts[-1] + counts[-2] + 1)
    return counts[height]


def fibonacci_tree(height: int) -> AVLTree:
    """Build the sparsest valid AVL tree of ``height``, keys 0, 1, 2 ... in order.

    Every internal node's left subtree is exactly one level taller than its
    right, so any deletion that shortens a right subtree unbalances the tree
    all the way up.
    """
    keys = itertools.count()

    def build(h):
        if h == 0:
            return None
        if h == 1:
            return AVLNode(next(keys))
        left = build(h - 1)
        node = AVLNode(next(keys))
        right = build(h - 2)
        node.left, node.right = left, right
        _update_height(node)
        return node

    tree = AVLTree()
    tree._root = build(height)
    tree._size = minimal_node_count(height)
    return tree


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    """Empty tree, one key, duplicates, and non-integer keys."""

    def test_empty_tree(self):
        tree = AVLTree()
        assert len(tree) == 0 and tree.is_empty() and tree.height() == 0
        assert tree.in_order() == [] and tree.search(1) is None
        assert tree.validate()
        with pytest.raises(KeyError):
            tree.delete(1)
        with pytest.raises(KeyError):
            tree.min_key()
        with pytest.raises(KeyError):
            tree.max_key()

    def test_single_key(self):
        tree = AVLTree()
        tree.insert(5, "five")
        assert tree.height() == 1 and tree.root.key == 5
        assert tree.search(5) == "five" and 5 in tree
        tree.delete(5)
        assert tree.is_empty() and tree.height() == 0

    def test_inserting_an_existing_key_updates_it(self):
        tree = AVLTree()
        tree.insert(1, "a")
        tree.insert(1, "b")
        assert tree.search(1) == "b" and len(tree) == 1
        check_avl(tree)

    def test_duplicate_heavy_input_keeps_one_of_each(self, structure_key_sets):
        keys = structure_key_sets["duplicates"]
        tree = AVLTree(keys)
        assert tree.in_order() == sorted(set(keys))
        check_avl(tree)

    def test_stored_none_is_distinguishable_from_absent(self):
        tree = AVLTree()
        tree.insert("k", None)
        assert tree.search("k", "MISSING") is None
        assert "k" in tree and "z" not in tree

    @pytest.mark.parametrize(
        "keys",
        [
            ["pear", "apple", "fig", "banana", "kiwi"],
            [3.5, -0.5, 2.25, 100.0, 0.0],
            [(2, "b"), (1, "z"), (2, "a"), (0, "q")],
        ],
        ids=["strings", "floats", "tuples"],
    )
    def test_any_orderable_key_type(self, keys):
        tree = AVLTree(keys)
        assert tree.in_order() == sorted(keys)
        check_avl(tree)

    def test_constructor_takes_a_mapping_or_an_iterable_of_keys(self):
        assert AVLTree({"a": 1, "b": 2}).search("b") == 2
        assert AVLTree([3, 1, 2]).in_order() == [1, 2, 3]

    def test_two_tuples_are_keys_not_key_value_pairs(self):
        """A 2-tuple handed to the constructor must be stored as a key.

        Guessing that any 2-tuple is a (key, value) pair would silently turn
        tuple keys into something else: key 2 with value "b" instead of the
        key (2, "b").
        """
        tree = AVLTree([(2, "b"), (1, "z")])
        assert tree.in_order() == [(1, "z"), (2, "b")]
        assert len(tree) == 2 and (2, "b") in tree
        assert tree.search((2, "b"), "MISSING") is None     # stored, value None

    def test_min_and_max_key(self):
        tree = AVLTree([50, 20, 80, 10, 90])
        assert tree.min_key() == 10 and tree.max_key() == 90


# ----------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------
class TestRotations:
    """Each of the four cases, triggered by the smallest input that needs it."""

    @pytest.mark.parametrize(
        "order, rotations",
        [
            ([3, 2, 1], 1),     # LL: single right rotation
            ([1, 2, 3], 1),     # RR: single left rotation
            ([3, 1, 2], 2),     # LR: left then right
            ([1, 3, 2], 2),     # RL: right then left
        ],
        ids=["LL", "RR", "LR", "RL"],
    )
    def test_each_rotation_case(self, order, rotations):
        tree = AVLTree(order)
        assert tree.root.key == 2
        assert tree.height() == 2
        assert tree.rotation_count == rotations
        check_avl(tree)

    def test_sorted_input_gives_a_perfectly_balanced_tree(self):
        """A plain BST would become a linked list of height n."""
        for exponent in range(1, 13):
            size = 2 ** exponent - 1
            ascending = AVLTree(range(size))
            descending = AVLTree(range(size, 0, -1))
            assert ascending.height() == exponent
            assert descending.height() == exponent
            check_avl(ascending)
            check_avl(descending)

    def test_insertion_never_needs_more_than_one_rebalance(self):
        """Fixing the lowest unbalanced node restores that subtree's height."""
        rng = random.Random(13)
        tree = AVLTree()
        for key in rng.sample(range(100_000), 5000):
            before = tree.rebalance_count
            tree.insert(key)
            assert tree.rebalance_count - before <= 1


# ----------------------------------------------------------------------
# Randomised invariant testing
# ----------------------------------------------------------------------
class TestRandomisedInvariants:
    """Balance and order hold after every operation of a long random sequence."""

    def test_invariants_after_every_insert_and_delete(self):
        """3,000 random operations; both checkers run after each one."""
        rng = random.Random(14)
        tree, model = AVLTree(), {}
        inserts = deletes = 0
        for _ in range(3000):
            key = rng.randint(0, 700)
            if key in model and rng.random() < 0.45:
                tree.delete(key)
                del model[key]
                deletes += 1
            else:
                value = rng.random()
                tree.insert(key, value)
                model[key] = value
                inserts += 1
            check_avl(tree)
            tree.validate()
            assert len(tree) == len(model)
            assert tree.height() <= avl_max_height(len(tree))
        assert inserts > 1000 and deletes > 500
        assert tree.in_order() == sorted(model)
        assert all(tree.search(k) == v for k, v in model.items())

    def test_in_order_traversal_is_sorted(self, structure_key_sets):
        for case, keys in structure_key_sets.items():
            tree = AVLTree(keys)
            assert tree.in_order() == sorted(set(keys)), case
            assert list(tree) == sorted(set(keys)), case
            assert [k for k, _v in tree.items()] == sorted(set(keys)), case

    @pytest.mark.parametrize("size", [10, 100, 1_000, 10_000])
    def test_height_stays_within_the_theoretical_bound(self, size):
        rng = random.Random(size)
        for keys in (rng.sample(range(size * 10), size), list(range(size))):
            tree = AVLTree(keys)
            assert perfect_height(size) <= tree.height() <= avl_max_height(size)
            assert tree.height() < 1.4405 * math.log2(size + 2) - 0.3277

    def test_large_mixed_workload(self):
        rng = random.Random(15)
        tree, model = AVLTree(), set()
        for step in range(60_000):
            key = rng.randint(0, 20_000)
            if key in model and rng.random() < 0.5:
                tree.delete(key)
                model.discard(key)
            else:
                tree.insert(key)
                model.add(key)
            if step % 5_000 == 0:
                check_avl(tree)
        check_avl(tree)
        assert tree.in_order() == sorted(model)

    def test_deep_sorted_insertion_stays_shallow(self):
        """50,000 sorted keys: height 16, nowhere near the recursion limit."""
        tree = AVLTree(range(50_000))
        assert tree.height() == perfect_height(50_000)


# ----------------------------------------------------------------------
# Deletion
# ----------------------------------------------------------------------
class TestDeletion:
    """Removing leaves, one-child nodes, two-child nodes and missing keys."""

    def test_deleting_a_leaf(self):
        tree = AVLTree(range(7))
        tree.delete(0)
        assert tree.in_order() == [1, 2, 3, 4, 5, 6]
        check_avl(tree)

    def test_deleting_a_node_with_two_children_uses_its_successor(self):
        tree = AVLTree(range(7))          # perfectly balanced, root 3
        tree.delete(3)
        assert tree.in_order() == [0, 1, 2, 4, 5, 6]
        assert tree.root.key == 4         # the in-order successor moved up
        check_avl(tree)

    def test_deleting_everything_in_random_order(self):
        rng = random.Random(16)
        keys = rng.sample(range(10_000), 2000)
        tree = AVLTree(keys)
        for key in rng.sample(keys, len(keys)):
            tree.delete(key)
            check_avl(tree)
        assert tree.is_empty()

    def test_a_failed_delete_changes_nothing(self):
        tree = AVLTree(range(50))
        snapshot, height = tree.in_order(), tree.height()
        with pytest.raises(KeyError):
            tree.delete(999)
        assert tree.in_order() == snapshot
        assert len(tree) == 50 and tree.height() == height
        check_avl(tree)


class TestDeletionRebalancing:
    """Deletion may rotate at more than one level. Check that it does."""

    @pytest.mark.parametrize("height", [6, 9, 12])
    def test_fibonacci_tree_delete_cascades_up_the_tree(self, height):
        """On the sparsest AVL tree, one delete rebalances at many levels.

        Try every key; the worst single delete must rebalance at least
        (height - 1) // 2 separate nodes, and the tree must stay valid. An
        implementation that stopped after the first rotation fails this.
        """
        worst = 0
        for target in range(minimal_node_count(height)):
            tree = fibonacci_tree(height)
            check_avl(tree)
            before = tree.rebalance_count
            tree.delete(target)
            check_avl(tree)
            worst = max(worst, tree.rebalance_count - before)
        assert worst >= (height - 1) // 2
        assert worst >= 2

    def test_rotation_counter_counts_individual_rotations(self):
        tree = AVLTree([3, 1, 2])           # one LR case
        assert tree.rotation_count == 2
        assert tree.rebalance_count == 1


# ----------------------------------------------------------------------
# The invariant checker itself
# ----------------------------------------------------------------------
class TestValidate:
    """validate() must catch each kind of corruption."""

    def test_detects_a_stale_height(self):
        tree = AVLTree(range(15))
        tree.root.height += 1
        with pytest.raises(AssertionError, match="stale height"):
            tree.validate()

    def test_detects_broken_order(self):
        tree = AVLTree(range(15))
        tree.root.left.key = 1000
        with pytest.raises(AssertionError, match="BST order"):
            tree.validate()

    def test_detects_an_imbalance(self):
        tree = AVLTree()
        root = AVLNode(3)
        root.left = AVLNode(2)
        root.left.left = AVLNode(1)
        _update_height(root.left)
        _update_height(root)
        tree._root, tree._size = root, 3
        with pytest.raises(AssertionError, match="unbalanced"):
            tree.validate()

    def test_detects_a_wrong_size(self):
        tree = AVLTree(range(5))
        tree._size = 99
        with pytest.raises(AssertionError, match="size"):
            tree.validate()


class TestHeightBounds:
    """The exact bound helpers agree with the Fibonacci recurrence."""

    def test_avl_max_height_matches_minimal_trees(self):
        for height in range(1, 25):
            nodes = minimal_node_count(height)
            assert avl_max_height(nodes) == height
            assert avl_max_height(nodes - 1) == height - 1

    def test_fibonacci_trees_are_valid_and_as_tall_as_allowed(self):
        for height in range(1, 14):
            tree = fibonacci_tree(height)
            check_avl(tree)
            assert tree.height() == avl_max_height(len(tree))

    def test_perfect_height(self):
        assert [perfect_height(n) for n in (0, 1, 2, 3, 4, 7, 8)] == [0, 1, 2, 2, 3, 3, 4]

    def test_negative_size_is_rejected(self):
        with pytest.raises(ValueError):
            avl_max_height(-1)
