"""Week 13 tests: cache-oblivious matrix ops, tree layouts and the LRU simulator (spec 9.6 anchors).

The rulers are independent of the code under test: numpy ``.T`` and ``@``
(allowed in tests only, spec 1.1), a list-based LRU written here, Morton
order computed by bit interleaving and ``sorted``, and values worked out by
hand.

Simulated misses. The six miss counts in the spec table are exact for the
section 5.1 layout at ``base=1``. Two of them can be checked by hand. Naive
transpose n = 64 at (1024, 16): the cache holds 64 blocks, but for each i
the 64 writes ``B[j][i]`` land in 64 different blocks and the row of A needs
one more, so every write misses (4,096) and A misses once per block (256):
4,352. Recursive transpose n = 64 at (1024, 16) touches each of the
2 n^2 / B = 512 blocks exactly once, the lower bound.

Layouts. A random root-to-leaf search in a BFS-ordered tree of height 20
touches the first block for the top 4 levels and a new block on every level
below, about 17 blocks, minus 1/16 for the one depth-4 node in block 0:
16.94. The vEB layout needs fewer than half as many.

No timings are asserted.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from src.advds.cache_oblivious import layouts, matrix_ops
from src.advds.cache_oblivious.cache_sim import LRUCache
from src.advds.cache_oblivious.layouts import (
    bfs_layout,
    blocks_per_search,
    build_bfs_tree,
    build_veb_tree,
    search,
    veb_order,
)
from src.advds.cache_oblivious.matrix_ops import (
    co_matmul,
    co_transpose,
    naive_matmul,
    naive_transpose,
    trace_matmul,
    trace_transpose,
    z_order,
)
from src.utils.random_utils import derive_seed, make_rng

SEED = 42


# ---------------------------------------------------------------------------
# Independent references
# ---------------------------------------------------------------------------


class ListLRU:
    """A plain list LRU: most recent block last, evict from the front."""

    def __init__(self, M: int, B: int) -> None:
        self.B = B
        self.capacity = M // B
        self.blocks: List[int] = []
        self.misses = 0

    def access(self, address: int) -> bool:
        block = address // self.B
        if block in self.blocks:
            self.blocks.remove(block)
            self.blocks.append(block)
            return True
        self.misses += 1
        if len(self.blocks) == self.capacity:
            self.blocks.pop(0)
        self.blocks.append(block)
        return False


class Recorder:
    """A fake cache that records every address it is asked for."""

    def __init__(self) -> None:
        self.seen: List[int] = []
        self.misses = 0

    def access(self, address: int) -> bool:
        self.seen.append(address)
        return False


class Tracked:
    """A matrix entry whose product logs which two entries were multiplied."""

    def __init__(self, tag: Tuple, log: List) -> None:
        self.tag = tag
        self.log = log

    def __mul__(self, other: "Tracked") -> int:
        self.log.append((self.tag, other.tag))
        return 0


def morton_order(n: int) -> List[Tuple[int, int]]:
    """Cells of an n x n matrix (n a power of two) sorted by interleaved bits."""

    def key(cell: Tuple[int, int]) -> int:
        i, j = cell
        code = 0
        for bit in range(16):
            code |= ((j >> bit) & 1) << (2 * bit)
            code |= ((i >> bit) & 1) << (2 * bit + 1)
        return code

    return sorted(((i, j) for i in range(n) for j in range(n)), key=key)


# ---------------------------------------------------------------------------
# LRU cache simulator
# ---------------------------------------------------------------------------


def test_lru_geometry():
    cache = LRUCache(1024, 16)
    assert (cache.M, cache.B, cache.capacity) == (1024, 16, 64)
    assert (cache.misses, cache.accesses, cache.hits) == (0, 0, 0)


def test_lru_evicts_least_recently_used_not_first_loaded():
    cache = LRUCache(M=4, B=2)  # two blocks
    assert cache.access(0) is False  # block 0
    assert cache.access(2) is False  # block 1
    assert cache.access(1) is True  # block 0 becomes most recent
    assert cache.access(4) is False  # block 2 evicts block 1 (LRU), not block 0 (FIFO)
    assert cache.access(0) is True
    assert cache.access(3) is False  # block 1 was evicted
    assert (cache.misses, cache.accesses, cache.hits) == (4, 6, 2)


@pytest.mark.parametrize("M, B", [(1, 1), (4, 1), (8, 2), (64, 8), (256, 16), (16, 16), (100, 7)])
def test_lru_matches_list_reference(M, B):
    rng = np.random.default_rng(derive_seed(SEED, "lru", M, B))
    stream = rng.integers(0, 40 * B, size=3000).tolist()
    stream += list(range(0, 20 * B)) * 2  # a sequential sweep repeated
    cache, ref = LRUCache(M, B), ListLRU(M, B)
    for address in stream:
        assert cache.access(address) == ref.access(address)
    assert cache.misses == ref.misses
    assert cache.accesses == len(stream)


def test_lru_reset_empties_cache_and_counters():
    cache = LRUCache(64, 8)
    for a in range(200):
        cache.access(a)
    cache.reset()
    assert (cache.misses, cache.accesses) == (0, 0)
    assert cache.access(199) is False  # its block was resident before the reset


def test_lru_rejects_bad_arguments():
    with pytest.raises(ValueError):
        LRUCache(0, 1)
    with pytest.raises(ValueError):
        LRUCache(8, 0)
    with pytest.raises(ValueError):
        LRUCache(4, 8)
    with pytest.raises(TypeError):
        LRUCache(True, 1)
    with pytest.raises(TypeError):
        LRUCache(8.0, 2)
    with pytest.raises(ValueError):
        LRUCache(8, 2).access(-1)


# ---------------------------------------------------------------------------
# Real matrix operations against numpy
# ---------------------------------------------------------------------------

TRANSPOSES = {
    "naive": naive_transpose,
    "co_base1": lambda A: co_transpose(A, base=1),
    "co_base3": lambda A: co_transpose(A, base=3),
    "co_base16": lambda A: co_transpose(A, base=16),
}

PRODUCTS = {
    "naive": naive_matmul,
    "co_base1": lambda A, B: co_matmul(A, B, base=1),
    "co_base4": lambda A, B: co_matmul(A, B, base=4),
    "co_base16": lambda A, B: co_matmul(A, B, base=16),
}


@pytest.mark.parametrize("shape", [(1, 1), (7, 5), (16, 16), (33, 17)])
@pytest.mark.parametrize("name", sorted(TRANSPOSES))
def test_transpose_matches_numpy(shape, name):
    rng = np.random.default_rng(derive_seed(SEED, "transpose", shape))
    A = rng.integers(-100, 100, size=shape)
    X = rng.standard_normal(shape)
    fn = TRANSPOSES[name]
    assert fn(A.tolist()) == A.T.tolist()
    assert fn(X) == X.T.tolist()  # numpy input, exact copies of the floats


@pytest.mark.parametrize("dims", [(1, 1, 1), (7, 7, 7), (16, 16, 16), (13, 9, 11)])
@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_matmul_matches_numpy(dims, name):
    n, m, p = dims
    rng = np.random.default_rng(derive_seed(SEED, "matmul", dims))
    A = rng.integers(-50, 50, size=(n, m))
    B = rng.integers(-50, 50, size=(m, p))
    fn = PRODUCTS[name]
    assert fn(A.tolist(), B.tolist()) == (A @ B).tolist()  # integers: exact
    X = rng.standard_normal((n, m))
    Y = rng.standard_normal((m, p))
    got = np.array(fn(X, Y))
    assert got.shape == (n, p)
    assert np.allclose(got, X @ Y)


def test_matrix_ops_reject_bad_input():
    with pytest.raises(ValueError):
        naive_transpose([[1, 2], [3]])
    with pytest.raises(ValueError):
        co_transpose([1, 2, 3])
    with pytest.raises(ValueError):
        naive_matmul([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError):
        co_matmul([[1]], [[1]], base=0)
    with pytest.raises(TypeError):
        co_transpose([[1]], base=1.5)


def test_defaults_are_fully_oblivious():
    import inspect

    assert inspect.signature(co_transpose).parameters["base"].default == 1
    assert inspect.signature(co_matmul).parameters["base"].default == 1


# ---------------------------------------------------------------------------
# Z order
# ---------------------------------------------------------------------------


def test_z_order_4_anchor():
    cells = z_order(4)
    assert len(cells) == 16
    assert set(cells) == {(i, j) for i in range(4) for j in range(4)}
    assert cells[:4] == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_z_order_is_morton_order_for_powers_of_two(n):
    assert z_order(n) == morton_order(n)


@pytest.mark.parametrize("n", [0, 3, 5, 7, 12])
def test_z_order_visits_every_cell_once(n):
    cells = z_order(n)
    assert len(cells) == n * n
    assert set(cells) == {(i, j) for i in range(n) for j in range(n)}


# ---------------------------------------------------------------------------
# Simulated misses (spec 9.6 table, exact)
# ---------------------------------------------------------------------------

MISS_TABLE = [
    # (operation, n, M, B, naive misses, recursive misses)
    ("transpose", 64, 1024, 16, 4352, 512),
    ("transpose", 128, 1024, 16, 17408, 2048),
    ("transpose", 64, 256, 8, 4608, 1024),
    ("transpose", 64, 4096, 64, 4160, 254),
    ("multiply", 32, 1024, 16, 2176, 320),
    ("multiply", 64, 1024, 16, 266496, 2560),
]


@pytest.mark.parametrize(
    "op, n, M, B, naive_expected, recursive_expected",
    MISS_TABLE,
    ids=[f"{op}-n{n}-M{M}-B{B}" for op, n, M, B, _, _ in MISS_TABLE],
)
def test_simulated_misses_exact(op, n, M, B, naive_expected, recursive_expected):
    trace = trace_transpose if op == "transpose" else trace_matmul
    naive_cache, recursive_cache = LRUCache(M, B), LRUCache(M, B)
    assert trace(n, "naive", naive_cache) == naive_expected
    assert trace(n, "recursive", recursive_cache) == recursive_expected
    assert naive_cache.misses == naive_expected
    assert recursive_cache.misses == recursive_expected
    if op == "transpose":
        assert naive_cache.accesses == recursive_cache.accesses == 2 * n * n
    else:
        assert naive_cache.accesses == 2 * n**3 + n**2  # C once per (i, j)
        assert recursive_cache.accesses == 3 * n**3  # C once per base case


def test_recursive_transpose_meets_lower_bound():
    n, M, B = 64, 1024, 16
    assert trace_transpose(n, "recursive", LRUCache(M, B)) == 2 * n * n // B


def test_trace_matches_list_reference_cache():
    # The hand-written LRU and the list LRU agree on whole traces too.
    for trace, n in ((trace_transpose, 32), (trace_matmul, 16)):
        for method in ("naive", "recursive"):
            ref = ListLRU(256, 8)
            assert trace(n, method, LRUCache(256, 8)) == trace(n, method, ref)


def test_trace_does_not_reset_the_cache():
    cache = LRUCache(1024, 16)
    first = trace_transpose(64, "naive", cache)
    second = trace_transpose(64, "naive", cache)
    assert first == 4352
    assert second > first
    assert second == cache.misses
    assert cache.accesses == 2 * (2 * 64 * 64)


def test_trace_rejects_bad_arguments():
    with pytest.raises(ValueError):
        trace_transpose(8, "blocked", LRUCache(64, 8))
    with pytest.raises(ValueError):
        trace_matmul(-1, "naive", LRUCache(64, 8))
    with pytest.raises(TypeError):
        trace_matmul(4.0, "naive", LRUCache(64, 8))


# ---------------------------------------------------------------------------
# The trace replays exactly the real loop order (section 5.1)
# ---------------------------------------------------------------------------


def test_naive_transpose_trace_order():
    n = 3
    rec = Recorder()
    trace_transpose(n, "naive", rec)
    expected = []
    for i in range(n):
        for j in range(n):
            expected += [i * n + j, n * n + j * n + i]  # read A, then write B
    assert rec.seen == expected


def test_recursive_transpose_trace_follows_z_order():
    n = 8
    rec = Recorder()
    trace_transpose(n, "recursive", rec)
    a_reads, b_writes = rec.seen[0::2], rec.seen[1::2]
    cells = [(a // n, a % n) for a in a_reads]
    assert cells == z_order(n) == morton_order(n)
    assert b_writes == [n * n + j * n + i for i, j in cells]


def test_naive_matmul_trace_order():
    n = 3
    rec = Recorder()
    trace_matmul(n, "naive", rec)
    expected = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                expected += [i * n + k, n * n + k * n + j]  # read A, then B
            expected.append(2 * n * n + i * n + j)  # C once after the k loop
    assert rec.seen == expected


def _decode_matmul_trace(seen: List[int], n: int):
    """Split a recorded multiply trace into (i, k, k', j) products and C touches."""
    products, c_after = [], []
    idx = 0
    while idx < len(seen):
        if seen[idx] >= 2 * n * n:
            c = seen[idx] - 2 * n * n
            c_after.append((len(products), (c // n, c % n)))
            idx += 1
            continue
        a, b = seen[idx], seen[idx + 1] - n * n
        assert a < n * n and 0 <= b < n * n  # A is read first, then B
        products.append((a // n, a % n, b // n, b % n))
        idx += 2
    return products, c_after


def test_recursive_matmul_trace_touches_c_after_each_base_case():
    n = 4
    rec = Recorder()
    trace_matmul(n, "recursive", rec)
    products, c_after = _decode_matmul_trace(rec.seen, n)
    assert len(products) == n**3
    assert {(i, j, k) for i, k, k2, j in products} == {
        (i, j, k) for i in range(n) for j in range(n) for k in range(n)
    }
    assert all(k == k2 for _, k, k2, _ in products)
    # base=1: every multiply-add is followed by one access to its C entry
    assert [count for count, _ in c_after] == list(range(1, n**3 + 1))
    assert [cell for _, cell in c_after] == [(i, j) for i, _, _, j in products]


@pytest.mark.parametrize("method, real", [("naive", naive_matmul), ("recursive", co_matmul)])
def test_matmul_trace_order_equals_real_multiplication_order(method, real):
    n = 4
    log: List = []
    A = [[Tracked(("A", i, k), log) for k in range(n)] for i in range(n)]
    B = [[Tracked(("B", k, j), log) for j in range(n)] for k in range(n)]
    real(A, B)  # co_matmul runs with its default base=1, like the trace
    real_order = [(a[1], a[2], b[1], b[2]) for a, b in log]
    rec = Recorder()
    trace_matmul(n, method, rec)
    traced_order, _ = _decode_matmul_trace(rec.seen, n)
    assert traced_order == real_order


def test_real_and_traced_functions_share_one_recursion(monkeypatch):
    calls = {"transpose": [], "matmul": []}
    original_t = matrix_ops._transpose_walk
    original_m = matrix_ops._matmul_walk

    def spy_t(r0, r1, c0, c1, base, visit):
        calls["transpose"].append((r0, r1, c0, c1, base))
        original_t(r0, r1, c0, c1, base, visit)

    def spy_m(i0, i1, j0, j1, k0, k1, base, visit):
        calls["matmul"].append((i0, i1, j0, j1, k0, k1, base))
        original_m(i0, i1, j0, j1, k0, k1, base, visit)

    monkeypatch.setattr(matrix_ops, "_transpose_walk", spy_t)
    monkeypatch.setattr(matrix_ops, "_matmul_walk", spy_m)
    n = 8
    M = np.arange(n * n).reshape(n, n).tolist()

    co_transpose(M, base=1)
    real_t, calls["transpose"] = calls["transpose"], []
    trace_transpose(n, "recursive", Recorder())
    assert real_t and calls["transpose"] == real_t

    co_matmul(M, M, base=1)
    real_m, calls["matmul"] = calls["matmul"], []
    trace_matmul(n, "recursive", Recorder())
    assert real_m and calls["matmul"] == real_m


def test_split_rules():
    # Longer dimension first, rows on ties (transpose); i before j before k (multiply).
    pieces: List = []
    matrix_ops._transpose_walk(0, 2, 0, 3, 1, lambda *p: pieces.append(p))
    # 2x3: the columns are longer, so column 0 (both rows) comes before column 1
    assert pieces == [(0, 1, 0, 1), (1, 2, 0, 1), (0, 1, 1, 2), (0, 1, 2, 3), (1, 2, 1, 2), (1, 2, 2, 3)]
    pieces = []
    matrix_ops._transpose_walk(0, 2, 0, 2, 1, lambda *p: pieces.append(p))
    assert pieces == [(0, 1, 0, 1), (0, 1, 1, 2), (1, 2, 0, 1), (1, 2, 1, 2)]  # tie: rows
    boxes: List = []
    matrix_ops._matmul_walk(0, 2, 0, 2, 0, 2, 1, lambda *p: boxes.append(p))
    assert boxes[:4] == [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 1, 2),
        (0, 1, 1, 2, 0, 1),
        (0, 1, 1, 2, 1, 2),
    ]  # i split first, then j, then k
    boxes = []
    matrix_ops._matmul_walk(0, 1, 0, 1, 0, 4, 1, lambda *p: boxes.append(p))
    assert [b[4:] for b in boxes] == [(0, 1), (1, 2), (2, 3), (3, 4)]  # k halves in order


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


def test_bfs_layout():
    assert bfs_layout(3) == [1, 2, 3, 4, 5, 6, 7]
    assert bfs_layout(0) == []


def test_veb_order_anchors():
    assert veb_order(3) == [1, 2, 4, 5, 3, 6, 7]
    assert veb_order(4) == [1, 2, 3, 4, 8, 9, 5, 10, 11, 6, 12, 13, 7, 14, 15]
    assert veb_order(5)[:11] == [1, 2, 3, 4, 8, 16, 17, 9, 18, 19, 5]


@pytest.mark.parametrize("h", range(0, 13))
def test_veb_order_is_a_permutation_with_top_tree_first(h):
    order = veb_order(h)
    assert sorted(order) == list(range(1, 2**h))
    top = h // 2
    # the first 2^(h//2) - 1 slots hold exactly the top tree (ids below 2^(h//2))
    assert sorted(order[: 2**top - 1]) == list(range(1, 2**top))


def test_build_small_trees():
    keys = [10, 20, 30, 40, 50, 60, 70]
    assert build_bfs_tree(keys) == [40, 20, 60, 10, 30, 50, 70]
    assert build_veb_tree(keys) == [40, 20, 10, 30, 60, 50, 70]


def _inorder_bfs(arr: List, b: int, out: List) -> None:
    """In-order walk of a heap-ordered array (independent of the module)."""
    if b > len(arr):
        return
    _inorder_bfs(arr, 2 * b, out)
    out.append(arr[b - 1])
    _inorder_bfs(arr, 2 * b + 1, out)


def _random_keys(h: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(10**6, size=2**h - 1, replace=False)).tolist()


def test_built_trees_are_binary_search_trees():
    h = 10
    keys = _random_keys(h, derive_seed(SEED, "keys", h))
    bfs = build_bfs_tree(keys)
    out: List = []
    _inorder_bfs(bfs, 1, out)
    assert out == keys
    veb = build_veb_tree(keys)
    assert veb == [bfs[b - 1] for b in veb_order(h)]


@pytest.mark.parametrize("layout", ["bfs", "veb"])
def test_both_layouts_find_every_key_of_a_2_10_tree(layout):
    h = 10
    keys = _random_keys(h, derive_seed(SEED, "keys", h))
    arr = build_bfs_tree(keys) if layout == "bfs" else build_veb_tree(keys)
    node_at = bfs_layout(h) if layout == "bfs" else veb_order(h)
    present = set(keys)
    for key in keys:
        found, path = search(arr, key, layout)
        assert found
        assert arr[path[-1]] == key
        ids = [node_at[p] for p in path]
        assert ids[0] == 1
        assert all(child // 2 == parent for parent, child in zip(ids, ids[1:]))
    rng = np.random.default_rng(derive_seed(SEED, "absent", layout))
    for key in rng.integers(-5, 10**6 + 5, size=500).tolist():
        if key in present:
            continue
        found, path = search(arr, key, layout)
        assert not found
        assert len(path) == h  # an unsuccessful search ends at a leaf


@pytest.mark.parametrize("h", range(1, 13))
def test_veb_navigation_matches_veb_order(h):
    order = veb_order(h)
    position_of = {b: p for p, b in enumerate(order)}
    for node in range(1, 2**h):
        depth = node.bit_length() - 1
        expected = [position_of[node >> (depth - d)] for d in range(depth + 1)]
        assert layouts._path_positions(node, depth, "veb", h) == expected


def test_path_positions_equal_search_paths():
    h = 8
    keys = list(range(2**h - 1))
    for layout, arr in (("bfs", build_bfs_tree(keys)), ("veb", build_veb_tree(keys))):
        for leaf in range(2 ** (h - 1), 2**h):
            key = 2 * (leaf - 2 ** (h - 1))  # in-order rank of a leaf
            assert layouts._path_positions(leaf, h - 1, layout, h) == search(arr, key, layout)[1]


def test_blocks_per_search_matches_brute_force_searches():
    h, B, trials, seed = 9, 8, 500, 7
    keys = list(range(2**h - 1))
    first = 2 ** (h - 1)
    for layout, arr in (("bfs", build_bfs_tree(keys)), ("veb", build_veb_tree(keys))):
        rng = make_rng(seed)
        total = 0
        for _ in range(trials):
            leaf = first + rng.randrange(first)
            found, path = search(arr, 2 * (leaf - first), layout)
            assert found and len(path) == h
            total += len({p // B for p in path})
        assert blocks_per_search(h, B, layout, trials, seed) == total / trials


def test_blocks_per_search_reference_h20_b16():
    bfs = blocks_per_search(20, 16, "bfs", trials=10000, seed=42)
    veb = blocks_per_search(20, 16, "veb", trials=10000, seed=42)
    assert bfs == pytest.approx(16.94, rel=0.05)
    assert veb == pytest.approx(7.57, rel=0.05)


@pytest.mark.parametrize("h", [16, 20])
@pytest.mark.parametrize("B", [16, 64])
def test_veb_needs_less_than_half_the_blocks_of_bfs(h, B):
    bfs = blocks_per_search(h, B, "bfs", trials=10000, seed=42)
    veb = blocks_per_search(h, B, "veb", trials=10000, seed=42)
    assert veb < 0.5 * bfs


def test_blocks_per_search_is_seeded():
    a = blocks_per_search(12, 16, "veb", 2000, 3)
    assert a == blocks_per_search(12, 16, "veb", 2000, 3)


def test_blocks_per_search_requires_an_int_seed():
    # Regression: seed=None used to reach make_rng(None), an OS-seeded,
    # irreproducible generator, although the docstring promised a TypeError.
    with pytest.raises(TypeError):
        blocks_per_search(12, 16, "veb", 100, None)
    with pytest.raises(TypeError):
        blocks_per_search(12, 16, "bfs", 100, True)
    with pytest.raises(TypeError):
        blocks_per_search(12, 16, "bfs", 100, 4.2)
    # numpy integer seeds behave like the equal Python int
    assert blocks_per_search(12, 16, "veb", 100, np.int64(3)) == blocks_per_search(12, 16, "veb", 100, 3)


def test_layout_errors():
    with pytest.raises(ValueError):
        build_bfs_tree([1, 2])  # not 2^h - 1 keys
    with pytest.raises(ValueError):
        build_veb_tree([1, 3, 2])  # not sorted
    with pytest.raises(ValueError):
        search([1, 2, 3], 2, "inorder")
    with pytest.raises(ValueError):
        search([1, 2, 3, 4], 2, "bfs")
    with pytest.raises(ValueError):
        blocks_per_search(0, 16, "bfs", 10, 42)
    with pytest.raises(ValueError):
        blocks_per_search(10, 0, "veb", 10, 42)
    with pytest.raises(ValueError):
        veb_order(-1)
