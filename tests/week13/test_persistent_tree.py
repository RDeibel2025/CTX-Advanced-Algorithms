"""Week 13 tests: persistent segment tree and persistent map (spec 9.3 anchors).

The rulers are independent of the code under test: plain Python lists (one
saved snapshot per version), slicing with the built-in ``sum``, ``min`` and
``max``, a ``dict`` for the map, and node counts worked out by hand.

Hand check of the 9.3 anchors on ``A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]``
(sum 45):

* ``v1 = point_set(0, 3, 100)`` gives ``[5, 2, 8, 100, 9, 3, 7, 4, 6, 0]``,
  sum 45 - 1 + 100 = 144.
* ``v2 = range_add(v1, 0, 5, 1)`` adds 1 to five elements:
  ``[6, 3, 9, 101, 10, 3, 7, 4, 6, 0]``, sum 149, maximum 101, and the
  minimum of the first five is 3.

Node counts. Halving ``[0, n)`` at ``(lo + hi) // 2`` gives a full binary
tree with n leaves, so 2n - 1 nodes: 131,071 for n = 2^16. Every leaf of
that tree is at depth 16, and a point set copies the root-to-leaf path:
17 nodes.

No timings are asserted.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from src.advds.trees import persistent_tree as pt
from src.advds.trees.persistent_tree import (
    EMPTY_MAP_HASH,
    MapNode,
    PersistentMap,
    PersistentSegmentTree,
    SegmentNode,
)
from src.utils.random_utils import derive_seed, make_rng, stable_hash64

SEED = 42

#: The spec 9.1 array, version 0 of the anchor tree.
A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]


def _rng(*labels):
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(SEED, "test_persistent_tree", *labels))


def _reachable(root):
    """Every distinct node reachable from ``root``, keyed by ``id``."""
    return _reachable_from([root])


def _reachable_from(roots):
    """Every distinct node reachable from any of ``roots``, keyed by ``id``."""
    seen = {}
    stack = [root for root in roots if root is not None]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen[id(node)] = node
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    return seen


def _leaf_depth(n, i):
    """Depth of leaf i when [0, n) is halved at (lo + hi) // 2, counted by hand."""
    lo, hi, depth = 0, n, 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if i < mid:
            hi = mid
        else:
            lo = mid
        depth += 1
    return depth


def _random_range(rng, n):
    """A uniformly chosen pair 0 <= l <= r <= n."""
    l = int(rng.integers(0, n + 1))
    r = int(rng.integers(l, n + 1))
    return l, r


def _expected(snapshot, l, r):
    """Brute-force sum, min and max of snapshot[l:r] with the empty-range identities."""
    window = snapshot[l:r]
    if not window:
        return 0, math.inf, -math.inf
    return sum(window), min(window), max(window)


def _random_updates(tree, snapshots, rng, count, low=-50, high=50):
    """Apply ``count`` random updates to random versions, keeping list snapshots."""
    n = len(snapshots[0])
    for _ in range(count):
        v = int(rng.integers(0, len(snapshots)))
        new = list(snapshots[v])
        if n and rng.random() < 0.5:
            i = int(rng.integers(0, n))
            value = int(rng.integers(low, high + 1))
            got = tree.point_set(v, i, value)
            new[i] = value
        else:
            l, r = _random_range(rng, n)
            delta = int(rng.integers(low, high + 1))
            got = tree.range_add(v, l, r, delta)
            for j in range(l, r):
                new[j] += delta
        assert got == len(snapshots)
        snapshots.append(new)


# ---------------------------------------------------------------------------
# PersistentSegmentTree: spec 9.3 anchors
# ---------------------------------------------------------------------------


def test_anchor_versions_and_queries():
    t = PersistentSegmentTree(A)
    v1 = t.point_set(0, 3, 100)
    v2 = t.range_add(v1, 0, 5, 1)
    assert (v1, v2) == (1, 2)
    assert t.versions() == 3
    assert t.query_sum(0, 0, 10) == 45
    assert t.query_sum(v1, 0, 10) == 144
    assert t.query_sum(v2, 0, 10) == 149
    assert t.point_query(v2, 3) == 101
    assert t.point_query(v1, 3) == 100
    assert t.point_query(0, 3) == 1
    assert t.query_max(v2, 0, 10) == 101
    assert t.query_min(v2, 0, 5) == 3
    assert t.to_list(0) == A
    assert t.to_list(v1) == [5, 2, 8, 100, 9, 3, 7, 4, 6, 0]
    assert t.to_list(v2) == [6, 3, 9, 101, 10, 3, 7, 4, 6, 0]


def test_anchor_segment_tree_queries_on_version_zero():
    # The spec 9.1 queries hold on version 0 too.
    t = PersistentSegmentTree(A)
    assert t.query_sum(0, 2, 7) == 28
    assert t.query_min(0, 1, 5) == 1
    assert t.query_max(0, 0, 10) == 9
    assert t.query_min(0, 5, 10) == 0
    v = t.range_add(0, 3, 8, 10)
    assert t.query_sum(v, 0, 10) == 95
    assert t.query_min(v, 0, 10) == 0
    assert t.query_max(v, 2, 6) == 19
    assert t.query_sum(v, 6, 10) == 37


def test_anchor_point_set_creates_17_nodes_at_n_2_16():
    n = 1 << 16
    rng = _rng("2^16")
    values = rng.integers(-1000, 1001, n).tolist()
    t = PersistentSegmentTree(values)
    assert t.nodes_created() == 2 * n - 1 == 131_071
    assert len(_reachable(t.root(0))) == 131_071
    snapshots = {0: values}
    for step in range(1000):
        v = int(rng.integers(0, t.versions()))
        i = int(rng.integers(0, n))
        value = int(rng.integers(-1000, 1001))
        before = t.nodes_created()
        new_v = t.point_set(v, i, value)
        assert t.nodes_created() - before == 17
        assert t.point_query(new_v, i) == value
        if step < 3:
            # Structural check: exactly 17 nodes of the new version are new.
            old_ids = _reachable(t.root(v))
            new_ids = _reachable(t.root(new_v))
            assert len(new_ids) == 131_071
            assert len(set(new_ids) - set(old_ids)) == 17
            snap = list(snapshots[v])
            snap[i] = value
            snapshots[new_v] = snap
            assert t.to_list(new_v) == snap
    assert t.versions() == 1001
    assert t.nodes_created() == 131_071 + 17 * 1000
    # Independent of the tree's own counter: the distinct node objects across
    # all 1,001 versions are the build plus 17 per point set.
    all_nodes = _reachable_from([t.root(v) for v in range(t.versions())])
    assert len(all_nodes) == 131_071 + 17 * 1000
    assert t.to_list(0) == values


@pytest.mark.parametrize("k", [0, 1, 2, 5, 10])
def test_point_set_creates_k_plus_one_nodes_for_powers_of_two(k):
    n = 1 << k
    t = PersistentSegmentTree([0] * n)
    for i in range(n):
        before = t.nodes_created()
        t.point_set(t.versions() - 1, i, i)
        assert t.nodes_created() - before == k + 1
    assert t.to_list(t.versions() - 1) == list(range(n))


@pytest.mark.parametrize(
    "n", list(range(1, 130)) + [255, 256, 257, 1000, 1023, 1025, 4097]
)
def test_build_creates_exactly_2n_minus_1_nodes(n):
    t = PersistentSegmentTree(range(n))
    assert t.nodes_created() == 2 * n - 1
    assert len(_reachable(t.root(0))) == 2 * n - 1
    assert t.to_list(0) == list(range(n))


@pytest.mark.parametrize("n", [3, 10, 100, 1000])
def test_point_set_copies_exactly_the_leaf_path_for_any_n(n):
    t = PersistentSegmentTree([1] * n)
    for i in range(n):
        before = t.nodes_created()
        t.point_set(0, i, 2)
        made = t.nodes_created() - before
        assert made == _leaf_depth(n, i) + 1
        assert math.floor(math.log2(n)) + 1 <= made <= math.ceil(math.log2(n)) + 1


@pytest.mark.parametrize("n", [1, 7, 64, 1000])
def test_range_add_creates_logarithmic_nodes(n):
    rng = _rng("range-add-bound", n)
    t = PersistentSegmentTree([0] * n)
    bound = 4 * (math.ceil(math.log2(n)) + 1) if n > 1 else 1
    for _ in range(300):
        l, r = _random_range(rng, n)
        before = t.nodes_created()
        v = t.range_add(t.versions() - 1, l, r, 1)
        made = t.nodes_created() - before
        assert made <= bound
        old_ids = _reachable(t.root(v - 1))
        new_ids = _reachable(t.root(v))
        assert len(set(new_ids) - set(old_ids)) == made


# ---------------------------------------------------------------------------
# PersistentSegmentTree: immutability and permanent tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values_label", ["anchor", "random-1000"])
def test_version_zero_unchanged_after_1000_random_updates(values_label):
    rng = _rng("immutability", values_label)
    values = list(A) if values_label == "anchor" else rng.integers(-99, 100, 1000).tolist()
    n = len(values)
    t = PersistentSegmentTree(values)
    queries = [_random_range(rng, n) for _ in range(300)]

    def answers():
        return (
            t.to_list(0),
            [t.point_query(0, i) for i in range(n)],
            [
                (t.query_sum(0, l, r), t.query_min(0, l, r), t.query_max(0, l, r))
                for l, r in queries
            ],
        )

    def node_state():
        return {
            key: (node.left, node.right, node.tag, node.sum, node.min, node.max)
            for key, node in _reachable(t.root(0)).items()
        }

    root0 = t.root(0)
    before_answers = answers()
    before_nodes = node_state()
    snapshots = [list(values)]
    _random_updates(t, snapshots, rng, 1000)
    assert t.versions() == 1001
    assert t.root(0) is root0
    assert answers() == before_answers
    assert node_state() == before_nodes
    assert before_answers[0] == values
    for l, r in queries[:50]:
        assert (t.query_sum(0, l, r), t.query_min(0, l, r), t.query_max(0, l, r)) == _expected(
            values, l, r
        )


def test_tags_are_permanent_and_never_pushed_down():
    t = PersistentSegmentTree(A)
    root0 = t.root(0)
    v = t.range_add(0, 0, 10, 7)
    root = t.root(v)
    # The whole range is covered at the root: one new node, children shared.
    assert t.nodes_created() == 19 + 1
    assert root.tag == 7 and root0.tag == 0
    assert root.left is root0.left and root.right is root0.right
    assert (root.sum, root.min, root.max) == (45 + 70, 0 + 7, 9 + 7)
    assert t.to_list(v) == [x + 7 for x in A]
    # A query below the root adds the root's tag on the way down.
    assert t.query_sum(v, 2, 7) == 28 + 5 * 7
    assert t.point_query(v, 9) == 7


def test_nodes_are_immutable_and_slotted():
    t = PersistentSegmentTree(A)
    node = t.root(0)
    assert isinstance(node, SegmentNode)
    assert not hasattr(node, "__dict__")
    for name in ("left", "right", "tag", "sum", "min", "max"):
        with pytest.raises(AttributeError):
            setattr(node, name, 0)
        with pytest.raises(AttributeError):
            delattr(node, name)
    with pytest.raises(AttributeError):
        node.extra = 1
    assert t.query_sum(0, 0, 10) == 45


def test_empty_range_add_makes_a_version_sharing_the_root():
    t = PersistentSegmentTree(A)
    v = t.range_add(0, 4, 4, 1000)
    assert v == 1
    assert t.root(v) is t.root(0)
    assert t.nodes_created() == 19
    assert t.to_list(v) == A


def test_point_set_to_same_value_still_copies_the_path():
    t = PersistentSegmentTree([1] * 8)
    v = t.point_set(0, 2, 1)
    assert t.nodes_created() == 15 + 4
    assert t.root(v) is not t.root(0)
    assert t.to_list(v) == [1] * 8


# ---------------------------------------------------------------------------
# PersistentSegmentTree: random checks against list snapshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16, 33, 100, 257])
def test_random_queries_on_random_versions_match_snapshots(n):
    rng = _rng("random-versions", n)
    values = rng.integers(-100, 101, n).tolist()
    t = PersistentSegmentTree(values)
    snapshots = [list(values)]
    _random_updates(t, snapshots, rng, 300)
    assert t.versions() == len(snapshots) == 301
    for v, snap in enumerate(snapshots):
        assert t.to_list(v) == snap
    for _ in range(600):
        v = int(rng.integers(0, len(snapshots)))
        snap = snapshots[v]
        l, r = _random_range(rng, n)
        exp_sum, exp_min, exp_max = _expected(snap, l, r)
        assert t.query_sum(v, l, r) == exp_sum
        assert t.query_min(v, l, r) == exp_min
        assert t.query_max(v, l, r) == exp_max
        i = int(rng.integers(0, n))
        assert t.point_query(v, i) == snap[i]


def test_random_operation_sequences_interleaved_with_queries():
    for seq in range(60):
        rng = _rng("interleaved", seq)
        n = int(rng.integers(1, 51))
        values = rng.integers(-20, 21, n).tolist()
        t = PersistentSegmentTree(values)
        snapshots = [list(values)]
        for _ in range(40):
            _random_updates(t, snapshots, rng, 1, low=-9, high=9)
            v = int(rng.integers(0, len(snapshots)))
            l, r = _random_range(rng, n)
            exp = _expected(snapshots[v], l, r)
            assert (t.query_sum(v, l, r), t.query_min(v, l, r), t.query_max(v, l, r)) == exp
        for v, snap in enumerate(snapshots):
            assert [t.point_query(v, i) for i in range(n)] == snap


def test_float_values_and_numpy_input():
    t = PersistentSegmentTree(np.array([1.5, -2.0, 4.25]))
    assert t.to_list(0) == [1.5, -2.0, 4.25]
    assert all(type(x) is float for x in t.to_list(0))
    v = t.range_add(0, 0, 3, 0.5)
    assert t.query_sum(v, 0, 3) == pytest.approx(5.25)
    assert t.query_min(v, 0, 3) == -1.5
    ints = PersistentSegmentTree(np.arange(5, dtype=np.int64))
    assert ints.to_list(0) == [0, 1, 2, 3, 4]
    assert type(ints.query_sum(0, 0, 5)) is int


def test_huge_integers_do_not_overflow():
    big = 2**62
    t = PersistentSegmentTree([big] * 8)
    v = t.range_add(0, 0, 8, big)
    assert t.query_sum(v, 0, 8) == 16 * big
    assert t.query_max(v, 0, 8) == 2 * big


def test_numpy_scalar_values_and_deltas_are_exact_python_numbers():
    # Regression: numpy scalars used to be stored as they came, so int64 sums
    # wrapped around (or raised OverflowError) past 2**63, float32 sums were
    # rounded to single precision, and numpy types leaked into the answers.
    big = 2**62
    t = PersistentSegmentTree([np.int64(big)] * 8)
    assert t.query_sum(0, 0, 8) == 8 * big
    v = t.range_add(0, 0, 8, np.int64(big))
    assert t.query_sum(v, 0, 8) == 16 * big
    assert t.query_max(v, 0, 8) == 2 * big
    w = t.point_set(v, 3, np.int64(-5))
    snapshot = [2 * big] * 8
    snapshot[3] = -5
    assert t.to_list(w) == snapshot
    assert t.query_sum(w, 0, 8) == 14 * big - 5
    for version in (0, v, w):
        answers = t.to_list(version) + [
            t.query_sum(version, 0, 8),
            t.query_min(version, 2, 6),
            t.query_max(version, 0, 8),
            t.point_query(version, 3),
        ]
        assert all(type(x) is int for x in answers)
    f = PersistentSegmentTree([1.0, 2.0, 3.0])
    u = f.range_add(0, 0, 3, np.float32(0.1))
    assert all(type(x) is float for x in f.to_list(u))
    assert f.point_query(u, 0) == 1.0 + float(np.float32(0.1))


# ---------------------------------------------------------------------------
# PersistentSegmentTree: edge cases and errors
# ---------------------------------------------------------------------------


def test_empty_array():
    t = PersistentSegmentTree([])
    assert len(t) == 0
    assert t.nodes_created() == 0
    assert t.versions() == 1
    assert t.root(0) is None
    assert t.to_list(0) == []
    assert t.query_sum(0, 0, 0) == 0
    assert t.query_min(0, 0, 0) == math.inf
    assert t.query_max(0, 0, 0) == -math.inf
    assert t.range_add(0, 0, 0, 5) == 1
    with pytest.raises(IndexError):
        t.point_query(0, 0)
    with pytest.raises(IndexError):
        t.point_set(0, 0, 1)


def test_empty_range_identities():
    t = PersistentSegmentTree(A)
    for i in range(11):
        assert t.query_sum(0, i, i) == 0
        assert t.query_min(0, i, i) == math.inf
        assert t.query_max(0, i, i) == -math.inf


def test_invalid_versions_raise():
    t = PersistentSegmentTree(A)
    t.point_set(0, 0, 1)
    for bad in (-1, 2, 100):
        with pytest.raises(IndexError):
            t.query_sum(bad, 0, 1)
        with pytest.raises(IndexError):
            t.range_add(bad, 0, 1, 1)
        with pytest.raises(IndexError):
            t.point_query(bad, 0)
        with pytest.raises(IndexError):
            t.root(bad)
    for bad in ("0", 1.0, None, True):
        with pytest.raises(TypeError):
            t.query_sum(bad, 0, 1)
    assert t.query_sum(np.int64(1), 0, 10) == 41
    assert t.versions() == 2


def test_invalid_ranges_and_indices_raise():
    t = PersistentSegmentTree(A)
    for l, r in [(-1, 3), (0, 11), (5, 4), (11, 11), (-2, -1)]:
        for method in (t.query_sum, t.query_min, t.query_max):
            with pytest.raises(IndexError):
                method(0, l, r)
        with pytest.raises(IndexError):
            t.range_add(0, l, r, 1)
    for i in (-1, 10, 11):
        with pytest.raises(IndexError):
            t.point_query(0, i)
        with pytest.raises(IndexError):
            t.point_set(0, i, 0)
    with pytest.raises(TypeError):
        t.query_sum(0, 0.0, 3)
    with pytest.raises(TypeError):
        t.point_query(0, "1")
    # Failed calls create no version and no node.
    assert t.versions() == 1
    assert t.nodes_created() == 19


def test_len_and_versions_counting():
    t = PersistentSegmentTree(A)
    assert len(t) == 10
    ids = [t.range_add(0, 0, 1, 1) for _ in range(5)]
    assert ids == [1, 2, 3, 4, 5]
    assert t.versions() == 6


# ---------------------------------------------------------------------------
# PersistentMap helpers (independent of the code under test)
# ---------------------------------------------------------------------------


def _encode(x):
    """The documented canonical field encoding: tag, 8-byte big-endian length, bytes."""
    if isinstance(x, str):
        tag, data = b"s", x.encode("utf-8")
    elif isinstance(x, (bytes, bytearray)):
        tag, data = b"b", bytes(x)
    else:
        tag, data = b"r", repr(x).encode("utf-8")
    return tag + len(data).to_bytes(8, "big") + data


def _expected_hash(key, value, left, right):
    """blake2b over (key, value, left hash or "", right hash or ""), hexdigest."""
    lh = left.hash if left is not None else ""
    rh = right.hash if right is not None else ""
    payload = _encode(key) + _encode(value) + _encode(lh) + _encode(rh)
    return hashlib.blake2b(payload).hexdigest()


def _check_treap(m):
    """Assert BST order, heap order on (priority, key), priorities and Merkle hashes."""
    count = 0

    def walk(node, lo, hi):
        nonlocal count
        if node is None:
            return 0
        count += 1
        assert isinstance(node, MapNode)
        if lo is not None:
            assert lo < node.key
        if hi is not None:
            assert node.key < hi
        assert node.priority == stable_hash64(node.key)
        for child in (node.left, node.right):
            if child is not None:
                assert (node.priority, node.key) > (child.priority, child.key)
        assert node.hash == _expected_hash(node.key, node.value, node.left, node.right)
        return 1 + max(walk(node.left, lo, node.key), walk(node.right, node.key, hi))

    height = walk(m.root, None, None)
    assert count == len(m)
    return height


def _preorder_keys(node):
    """Pre-order list of keys, written recursively."""
    if node is None:
        return []
    return [node.key] + _preorder_keys(node.left) + _preorder_keys(node.right)


def _depth_of(m, key):
    """Number of edges from the root to the node holding key."""
    node, depth = m.root, 0
    while node.key != key:
        node = node.left if key < node.key else node.right
        depth += 1
    return depth


def _node_ids(m):
    """ids of every node of a map."""
    return {id(node) for node in m.iter_nodes()}


# ---------------------------------------------------------------------------
# PersistentMap: spec 9.3 anchors
# ---------------------------------------------------------------------------


def test_map_anchor_same_key_set_two_orders_same_root_hash():
    pairs = [(f"src/file_{i:04d}.txt", f"hash{i * 7919 % 1000:03d}") for i in range(300)]
    forward = PersistentMap()
    for key, value in pairs:
        forward = forward.set(key, value)
    shuffled = PersistentMap()
    rng = make_rng(derive_seed(SEED, "map-order"))
    order = list(pairs)
    rng.shuffle(order)
    assert order != pairs
    for key, value in order:
        shuffled = shuffled.set(key, value)
    backward = PersistentMap(reversed(pairs))
    assert forward.root_hash == shuffled.root_hash == backward.root_hash
    assert _preorder_keys(forward.root) == _preorder_keys(shuffled.root)
    assert forward.items() == shuffled.items() == sorted(pairs)
    _check_treap(forward)


def test_map_anchor_set_existing_key_returns_new_map_old_unchanged():
    old = PersistentMap({"a.txt": "h1", "b.txt": "h2", "c.txt": "h3"})
    old_hash, old_items, old_root = old.root_hash, old.items(), old.root
    new = old.set("b.txt", "h9")
    assert new is not old
    assert new.get("b.txt") == "h9"
    assert old.get("b.txt") == "h2"
    assert old.items() == old_items == [("a.txt", "h1"), ("b.txt", "h2"), ("c.txt", "h3")]
    assert old.root is old_root
    assert old.root_hash == old_hash
    assert new.root_hash != old_hash
    assert len(new) == len(old) == 3


# ---------------------------------------------------------------------------
# PersistentMap: random sequences against a dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed_label", range(5))
def test_map_random_set_delete_sequences_match_dict(seed_label):
    rng = make_rng(derive_seed(SEED, "map-random", seed_label))
    universe = [f"k{i:03d}" for i in range(80)]
    m = PersistentMap()
    d = {}
    history = [(m, dict(d))]
    for step in range(1500):
        key = rng.choice(universe)
        if rng.random() < 0.6:
            value = f"v{rng.randint(0, 9)}"
            m = m.set(key, value)
            d[key] = value
        elif key in d:
            m = m.delete(key)
            del d[key]
        else:
            with pytest.raises(KeyError):
                m.delete(key)
        history.append((m, dict(d)))
        assert len(m) == len(d)
        assert m.items() == sorted(d.items())
        probe = rng.choice(universe)
        assert (probe in m) == (probe in d)
        assert m.get(probe, "absent") == d.get(probe, "absent")
        if step % 100 == 0:
            _check_treap(m)
    # Non-mutation: every old map still holds exactly what it held when made,
    # and its hash equals a fresh build of the same contents.
    for old, snap in history[::25]:
        assert old.items() == sorted(snap.items())
        assert len(old) == len(snap)
        assert old.root_hash == PersistentMap(snap).root_hash
    _check_treap(m)


def test_map_history_independence_after_deletes():
    rng = make_rng(derive_seed(SEED, "map-history"))
    keys = [f"path/{i}" for i in range(200)]
    full = PersistentMap((k, k.upper()) for k in keys)
    removed = rng.sample(keys, 120)
    m = full
    for k in removed:
        m = m.delete(k)
    kept = [k for k in keys if k not in set(removed)]
    direct = PersistentMap((k, k.upper()) for k in reversed(kept))
    assert m.root_hash == direct.root_hash
    assert _preorder_keys(m.root) == _preorder_keys(direct.root)
    assert len(full) == 200 and len(m) == 80
    _check_treap(m)


# ---------------------------------------------------------------------------
# PersistentMap: structure, sharing and hashing
# ---------------------------------------------------------------------------


def test_map_path_copying_shares_all_nodes_off_the_path():
    keys = [f"f{i:04d}" for i in range(1000)]
    m = PersistentMap((k, "x") for k in keys)
    height = _check_treap(m)
    old_ids = _node_ids(m)
    for key in keys[::97]:
        m2 = m.set(key, "y")
        new = _node_ids(m2) - old_ids
        assert len(new) == _depth_of(m, key) + 1
        assert len(_node_ids(m2)) == 1000
    m3 = m.set("f0500.5", "new")
    assert len(_node_ids(m3) - old_ids) <= 2 * height + 1
    m4 = m.delete("f0123")
    assert len(_node_ids(m4) - old_ids) <= 2 * height
    assert m.items() == [(k, "x") for k in keys]


def test_map_set_same_value_shares_root():
    m = PersistentMap({"a": "1", "b": "2"})
    same = m.set("a", "1")
    assert same is not m
    assert same.root is m.root
    assert len(same) == 2


def test_map_set_equal_value_with_different_encoding_replaces_it():
    # Regression: 0.0 == -0.0, so set() used to keep the old 0.0 and its hash,
    # and the map then differed from a fresh build of what was last set.
    m = PersistentMap({"a": 0.0, "b": 1.0})
    m2 = m.set("a", -0.0)
    assert m2.root is not m.root
    assert math.copysign(1.0, m2.get("a")) == -1.0
    assert math.copysign(1.0, m.get("a")) == 1.0
    assert m2.root_hash == PersistentMap({"a": -0.0, "b": 1.0}).root_hash
    assert m2.root_hash != m.root_hash
    _check_treap(m2)


def test_nodes_and_structures_can_be_copied_and_pickled():
    # Regression: the immutable nodes refused the setattr calls that copy and
    # pickle use to restore slots, so copying either structure raised.
    import copy
    import pickle

    t = PersistentSegmentTree(A)
    v = t.range_add(t.point_set(0, 3, 100), 0, 5, 1)
    m = PersistentMap({"a.txt": "h1", "b.txt": "h2", "c.txt": "h3"})
    for clone in (copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))):
        t2 = clone(t)
        assert [t2.to_list(x) for x in range(3)] == [t.to_list(x) for x in range(3)]
        assert t2.query_sum(v, 0, 10) == 149
        assert t2.nodes_created() == t.nodes_created()
        m2 = clone(m)
        assert m2.items() == m.items()
        assert m2.root_hash == m.root_hash
        _check_treap(m2)
    # A deep copy keeps the sharing between versions.
    t3 = copy.deepcopy(t)
    roots = [t3.root(x) for x in range(t3.versions())]
    assert len(_reachable_from(roots)) == t.nodes_created()
    with pytest.raises(AttributeError):
        t3.root(0).tag = 1


def test_map_hash_tracks_contents():
    base = PersistentMap({"a": "1", "b": "2"})
    assert base.root_hash == PersistentMap([("b", "2"), ("a", "1")]).root_hash
    assert base.root_hash != base.set("b", "3").root_hash
    assert base.root_hash != base.set("c", "3").root_hash
    assert base.set("c", "3").delete("c").root_hash == base.root_hash
    assert base.set("b", "3").set("b", "2").root_hash == base.root_hash
    assert len(base.root_hash) == 128
    assert all(ch in "0123456789abcdef" for ch in base.root_hash)


def test_map_empty():
    m = PersistentMap()
    assert m.root is None
    assert len(m) == 0
    assert m.items() == []
    assert list(m.iter_nodes()) == []
    assert m.root_hash == EMPTY_MAP_HASH == hashlib.blake2b(b"").hexdigest()
    assert m.get("a") is None
    assert "a" not in m
    with pytest.raises(KeyError):
        m.delete("a")
    one = m.set("a", "1")
    assert one.delete("a").root is None
    assert one.delete("a").root_hash == EMPTY_MAP_HASH


def test_map_iter_nodes_is_preorder_over_every_node():
    m = PersistentMap((f"k{i}", str(i)) for i in range(150))
    nodes = list(m.iter_nodes())
    assert len(nodes) == 150
    assert nodes[0] is m.root
    assert [node.key for node in nodes] == _preorder_keys(m.root)
    assert len({id(node) for node in nodes}) == 150


def test_map_nodes_are_immutable():
    m = PersistentMap({"a": "1", "b": "2"})
    node = m.root
    assert not hasattr(node, "__dict__")
    for name in ("key", "value", "priority", "left", "right", "hash"):
        with pytest.raises(AttributeError):
            setattr(node, name, None)
        with pytest.raises(AttributeError):
            delattr(node, name)
    assert m.items() == [("a", "1"), ("b", "2")]


def test_map_priority_ties_are_broken_by_key(monkeypatch):
    # With every priority equal the larger key must sit on top, so the treap
    # becomes a left spine in decreasing key order, whatever the insert order.
    monkeypatch.setattr(pt, "stable_hash64", lambda key, salt=0: 7)
    keys = [f"t{i:02d}" for i in range(30)]
    rng = make_rng(derive_seed(SEED, "ties"))
    orders = [keys, keys[::-1], rng.sample(keys, len(keys))]
    hashes = set()
    for order in orders:
        m = PersistentMap((k, "v") for k in order)
        assert _preorder_keys(m.root) == keys[::-1]
        assert all(node.right is None for node in m.iter_nodes())
        hashes.add(m.root_hash)
        m = m.delete("t15")
        assert _preorder_keys(m.root) == [k for k in keys[::-1] if k != "t15"]
    assert len(hashes) == 1


def test_map_mapping_and_pairs_constructor_and_contains():
    m = PersistentMap([("a", "1"), ("b", "2"), ("a", "3")])
    assert len(m) == 2
    assert m.get("a") == "3"
    assert "b" in m and "c" not in m
    assert PersistentMap(m).items() == m.items()
    with_none = PersistentMap({"x": None})
    assert "x" in with_none and with_none.get("x", "d") is None


def test_map_other_comparable_keys():
    rng = make_rng(derive_seed(SEED, "int-keys"))
    keys = rng.sample(range(10_000), 300)
    m = PersistentMap((k, str(k)) for k in keys)
    assert m.items() == [(k, str(k)) for k in sorted(keys)]
    _check_treap(m)
    assert m.root_hash == PersistentMap((k, str(k)) for k in sorted(keys)).root_hash
