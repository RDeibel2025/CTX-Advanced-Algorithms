"""Week 13 tests: the Fenwick tree and the range-add Fenwick tree (spec 9.2 anchors).

Every expected value is from the spec 9.2 anchors or worked by hand, never
from the code under test. ``itertools.accumulate`` and ``np.cumsum`` (allowed
in tests only, spec 1.1) and plain slice sums are the independent reference.

Hand check of the anchors on ``A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]``:

* The running totals are 5, 7, 15, 16, 25, 28, 35, 39, 45, 45, and the
  empty prefix sums to 0.
* ``range_sum(2, 7) = 8 + 1 + 9 + 3 + 7 = 28``. ``set(4, 0)`` removes the 9
  at position 4, so the total becomes 45 - 9 = 36.
* ``RangeFenwick`` with ``range_add(3, 8, 10)`` adds 10 to positions 3..7:
  the total is 45 + 50 = 95 and ``[6, 10)`` holds 17 + 14 + 6 + 0 = 37.

Low-bit decomposition on A (1-based index i covers the ``i & -i`` elements
ending at i): ``tree[1..10] = [5, 7, 8, 16, 9, 12, 7, 39, 6, 6]``, for example
``tree[6] = A[4] + A[5] = 12`` (lowbit 2) and ``tree[8] = A[0] + ... + A[7] =
39`` (lowbit 8).

The random checks replay 300 seeded operation sequences per class (sizes
1..50, each size six times) and compare the full state with a plain list
after every step: every prefix sum against ``itertools.accumulate`` and, for
``FenwickTree``, every internal entry against its low-bit block sum. Fenwick
reads never change the structure, so the full check can run on the tree
itself.

No timings are asserted.
"""

from __future__ import annotations

import math
from itertools import accumulate

import numpy as np
import pytest

from src.advds.trees.fenwick_tree import FenwickTree, RangeFenwick
from src.utils.random_utils import derive_seed, make_rng

SEED = 42

A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]

N_SEQUENCES = 300
STEPS = 40

FENWICK_OPS = ("add", "set", "prefix_sum", "range_sum", "get")
RANGE_OPS = ("range_add", "range_add", "prefix_sum", "range_sum", "get")


def _random_range(rng, n: int):
    """A uniformly chosen pair ``0 <= l <= r <= n`` (empty ranges included)."""
    a, b = rng.randint(0, n), rng.randint(0, n)
    return (a, b) if a <= b else (b, a)


def _prefixes(values) -> list:
    """``[0, v0, v0 + v1, ...]``: the reference prefix sums."""
    return list(accumulate(values, initial=0))


def _check_fenwick_state(ft: FenwickTree, ref: list, where: str) -> None:
    """Every prefix sum and every low-bit block of ``ft`` matches ``ref``."""
    n = len(ref)
    assert len(ft) == n, where
    assert [ft.prefix_sum(i) for i in range(n + 1)] == _prefixes(ref), where
    assert ft.tree[0] == 0, where
    for i in range(1, n + 1):
        assert ft.tree[i] == sum(ref[i - (i & -i) : i]), f"{where}, tree[{i}]"


def _check_range_state(rf: RangeFenwick, ref: list, where: str) -> None:
    """Every prefix sum and every element of ``rf`` matches ``ref``."""
    n = len(ref)
    assert len(rf) == n, where
    assert [rf.prefix_sum(i) for i in range(n + 1)] == _prefixes(ref), where
    assert [rf.get(i) for i in range(n)] == ref, where


# ------------------------------------------------------------------ anchors


def test_anchor_prefix_sums():
    ft = FenwickTree(A)
    assert [ft.prefix_sum(i) for i in range(1, 11)] == [5, 7, 15, 16, 25, 28, 35, 39, 45, 45]
    assert ft.prefix_sum(0) == 0


def test_anchor_range_sum_and_set():
    ft = FenwickTree(A)
    assert ft.range_sum(2, 7) == 28
    ft.set(4, 0)
    assert ft.range_sum(0, 10) == 36


def test_anchor_range_fenwick():
    rf = RangeFenwick(A)
    rf.range_add(3, 8, 10)
    assert rf.range_sum(0, 10) == 95
    assert rf.range_sum(6, 10) == 37


def test_prefix_sums_match_numpy_cumsum():
    ft = FenwickTree(A)
    assert [ft.prefix_sum(i) for i in range(1, 11)] == np.cumsum(A).tolist()


# ------------------------------------------------------------ the structure


def test_low_bit_decomposition_on_anchor_array():
    ft = FenwickTree(A)
    assert len(ft.tree) == len(A) + 1
    assert ft.tree[1:] == [5, 7, 8, 16, 9, 12, 7, 39, 6, 6]
    for i in range(1, 11):
        assert ft.tree[i] == sum(A[i - (i & -i) : i])


def test_linear_build_equals_n_point_adds():
    for n in (1, 2, 3, 7, 8, 9, 31, 64, 100):
        rng = make_rng(derive_seed(SEED, "fenwick_build", n))
        values = [rng.randint(-1000, 1000) for _ in range(n)]
        built = FenwickTree(values)
        added = FenwickTree([0] * n)
        for i, v in enumerate(values):
            added.add(i, v)
        assert built.tree == added.tree


def test_point_add_touches_the_blocks_that_contain_it():
    # n = 16, element 4 is 1-based position 5: blocks 5, 6, 8 and 16 (module docstring).
    ft = FenwickTree([0] * 16)
    ft.add(4, 3)
    assert [j for j in range(1, 17) if ft.tree[j]] == [5, 6, 8, 16]
    # Position 13 = 0b1101 in a prefix walk: tree[13] + tree[12] + tree[8].
    ft = FenwickTree(list(range(1, 17)))
    assert ft.prefix_sum(13) == ft.tree[13] + ft.tree[12] + ft.tree[8] == 91


def test_get_set_add():
    ft = FenwickTree(A)
    assert [ft.get(i) for i in range(10)] == A
    ft.add(0, 10)
    ft.set(9, -4)
    ft.add(9, 1)
    assert [ft.get(i) for i in range(10)] == [15, 2, 8, 1, 9, 3, 7, 4, 6, -3]
    assert ft.range_sum(0, 10) == 52


def test_range_fenwick_structure():
    rf = RangeFenwick(A)
    # D = [5, -3, 6, -7, 8, -6, 4, -3, 2, -6]: b1 over D, b2 over y * D[y].
    diffs = [5, -3, 6, -7, 8, -6, 4, -3, 2, -6]
    assert [rf.b1.get(i) for i in range(10)] == diffs
    assert [rf.b2.get(i) for i in range(10)] == [y * d for y, d in enumerate(diffs)]
    assert [rf.get(i) for i in range(10)] == A
    assert [rf.prefix_sum(i) for i in range(11)] == _prefixes(A)


def test_range_fenwick_add_reaching_the_end():
    rf = RangeFenwick(A)
    rf.range_add(6, 10, 2)  # r == n: the adds at r are skipped
    rf.range_add(0, 10, -1)
    assert [rf.get(i) for i in range(10)] == [4, 1, 7, 0, 8, 2, 8, 5, 7, 1]
    assert rf.range_sum(0, 10) == 43


def test_single_element():
    ft = FenwickTree([7])
    assert (ft.prefix_sum(0), ft.prefix_sum(1), ft.range_sum(0, 1)) == (0, 7, 7)
    ft.set(0, -2)
    assert ft.get(0) == -2
    rf = RangeFenwick([7])
    rf.range_add(0, 1, 5)
    assert (rf.get(0), rf.range_sum(0, 1), rf.prefix_sum(1)) == (12, 12, 12)


def test_numpy_input_and_indices():
    ft = FenwickTree(np.array(A, dtype=np.int64))
    assert type(ft.prefix_sum(10)) is int
    assert ft.range_sum(np.int64(2), np.int32(7)) == 28
    ft.set(np.int64(4), np.int64(0))
    assert ft.range_sum(0, 10) == 36
    rf = RangeFenwick(np.array(A, dtype=np.int32))
    rf.range_add(np.int64(3), np.int64(8), np.int64(10))
    assert rf.range_sum(0, 10) == 95
    assert type(rf.range_sum(0, 10)) is int


def test_values_are_copied():
    src = [1, 2, 3]
    ft = FenwickTree(src)
    ft.add(0, 5)
    rf = RangeFenwick(src)
    rf.range_add(0, 3, 5)
    assert src == [1, 2, 3]


def test_float_values():
    # Dyadic fractions, so every sum below is exact in binary floating point.
    ft = FenwickTree([0.5, 1.25, -2.0, 4.0])
    assert ft.prefix_sum(4) == 3.75
    ft.add(2, 0.25)
    assert ft.range_sum(1, 3) == -0.5
    rf = RangeFenwick([0.5, 1.25, -2.0, 4.0])
    rf.range_add(1, 3, 0.25)
    assert rf.range_sum(0, 4) == 4.25
    assert rf.get(2) == -1.75


def test_large_integers_stay_exact():
    big = 2**70
    ft = FenwickTree([big, big, 1])
    assert ft.prefix_sum(3) == 2 * big + 1
    rf = RangeFenwick([big, 1, big])
    rf.range_add(0, 3, big)
    assert rf.range_sum(0, 3) == 5 * big + 1


# ------------------------------------------------------- empty ranges and errors


@pytest.mark.parametrize("cls", [FenwickTree, RangeFenwick])
def test_empty_range_returns_zero(cls):
    tree = cls(A)
    for k in (0, 4, 10):
        assert tree.range_sum(k, k) == 0
    assert tree.prefix_sum(0) == 0


def test_range_fenwick_empty_add_changes_nothing():
    rf = RangeFenwick(A)
    for k in (0, 4, 10):
        rf.range_add(k, k, 99)
    assert [rf.get(i) for i in range(10)] == A


@pytest.mark.parametrize("cls", [FenwickTree, RangeFenwick])
def test_empty_and_bad_input_rejected(cls):
    with pytest.raises(ValueError):
        cls([])
    with pytest.raises(ValueError):
        cls(np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        cls([1.0, math.nan])
    for bad in (5, None, "123", [1, "2"], [True, 2], [[1, 2]], [1 + 2j]):
        with pytest.raises(TypeError):
            cls(bad)


@pytest.mark.parametrize("cls", [FenwickTree, RangeFenwick])
@pytest.mark.parametrize("inf", [math.inf, -math.inf, np.float64(np.inf)])
def test_infinities_rejected(cls, inf):
    """Regression: an infinity used to be accepted, and then every range sum is
    a difference of two infinite prefixes. ``FenwickTree([inf, 1.0, 2.0])``
    answered ``range_sum(1, 3)`` with NaN instead of 3.0, and
    ``RangeFenwick([inf, 1.0, 2.0])`` failed with a misleading NaN error."""
    with pytest.raises(ValueError, match="finite"):
        cls([inf, 1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        cls(np.array([1.0, 2.0, inf]))
    tree = cls(A)
    if cls is FenwickTree:
        with pytest.raises(ValueError):
            tree.add(0, inf)
        with pytest.raises(ValueError):
            tree.set(9, inf)
    else:
        with pytest.raises(ValueError):
            tree.range_add(0, 10, inf)
    assert [tree.get(i) for i in range(10)] == A
    assert tree.range_sum(1, 3) == 10


@pytest.mark.parametrize("cls", [FenwickTree, RangeFenwick])
def test_query_argument_errors(cls):
    tree = cls(A)
    with pytest.raises(IndexError):
        tree.prefix_sum(-1)
    with pytest.raises(IndexError):
        tree.prefix_sum(11)
    with pytest.raises(TypeError):
        tree.prefix_sum(2.0)
    with pytest.raises(IndexError):
        tree.range_sum(-1, 3)
    with pytest.raises(IndexError):
        tree.range_sum(0, 11)
    with pytest.raises(ValueError):
        tree.range_sum(5, 3)
    with pytest.raises(TypeError):
        tree.range_sum(0, True)
    with pytest.raises(IndexError):
        tree.get(10)
    with pytest.raises(IndexError):
        tree.get(-1)


def test_fenwick_update_argument_errors():
    ft = FenwickTree(A)
    with pytest.raises(IndexError):
        ft.add(10, 1)
    with pytest.raises(IndexError):
        ft.add(-1, 1)
    with pytest.raises(IndexError):
        ft.set(10, 1)
    with pytest.raises(TypeError):
        ft.add(1.0, 1)
    with pytest.raises(TypeError):
        ft.add(1, "1")
    with pytest.raises(TypeError):
        ft.set(1, None)
    with pytest.raises(ValueError):
        ft.set(1, math.nan)
    assert [ft.get(i) for i in range(10)] == A


def test_range_fenwick_update_argument_errors():
    rf = RangeFenwick(A)
    with pytest.raises(IndexError):
        rf.range_add(0, 11, 1)
    with pytest.raises(IndexError):
        rf.range_add(-1, 4, 1)
    with pytest.raises(ValueError):
        rf.range_add(4, 2, 1)
    with pytest.raises(TypeError):
        rf.range_add(0, 4, "1")
    with pytest.raises(ValueError):
        rf.range_add(0, 4, math.nan)
    assert [rf.get(i) for i in range(10)] == A


# ----------------------------------------------------------- random sequences


def test_random_fenwick_sequences_match_a_list():
    op_counts = {op: 0 for op in FENWICK_OPS}
    sizes = set()
    for case in range(N_SEQUENCES):
        n = 1 + case % 50
        sizes.add(n)
        rng = make_rng(derive_seed(SEED, "fenwick_tree", case))
        ref = [rng.randint(-100, 100) for _ in range(n)]
        ft = FenwickTree(ref)
        _check_fenwick_state(ft, ref, f"case {case}, build")
        for step in range(STEPS):
            op = rng.choice(FENWICK_OPS)
            op_counts[op] += 1
            where = f"case {case}, n {n}, step {step}, {op}"
            if op == "add":
                i, delta = rng.randrange(n), rng.randint(-50, 50)
                ft.add(i, delta)
                ref[i] += delta
            elif op == "set":
                i, value = rng.randrange(n), rng.randint(-100, 100)
                ft.set(i, value)
                ref[i] = value
            elif op == "prefix_sum":
                i = rng.randint(0, n)
                assert ft.prefix_sum(i) == sum(ref[:i]), where
            elif op == "range_sum":
                l, r = _random_range(rng, n)
                assert ft.range_sum(l, r) == sum(ref[l:r]), where
            else:
                i = rng.randrange(n)
                assert ft.get(i) == ref[i], where
            _check_fenwick_state(ft, ref, where)
    assert sizes == set(range(1, 51))
    assert all(count > 0 for count in op_counts.values()), op_counts


def test_random_range_fenwick_sequences_match_a_list():
    op_counts = {op: 0 for op in RANGE_OPS}
    sizes = set()
    adds_to_end = 0
    for case in range(N_SEQUENCES):
        n = 1 + case % 50
        sizes.add(n)
        rng = make_rng(derive_seed(SEED, "range_fenwick", case))
        ref = [rng.randint(-100, 100) for _ in range(n)]
        rf = RangeFenwick(ref)
        _check_range_state(rf, ref, f"case {case}, build")
        for step in range(STEPS):
            op = rng.choice(RANGE_OPS)
            op_counts[op] += 1
            l, r = _random_range(rng, n)
            where = f"case {case}, n {n}, step {step}, {op}({l}, {r})"
            if op == "range_add":
                delta = rng.randint(-50, 50)
                rf.range_add(l, r, delta)
                for k in range(l, r):
                    ref[k] += delta
                adds_to_end += r == n and l < r
            elif op == "prefix_sum":
                assert rf.prefix_sum(r) == sum(ref[:r]), where
            elif op == "range_sum":
                assert rf.range_sum(l, r) == sum(ref[l:r]), where
            else:
                i = rng.randrange(n)
                assert rf.get(i) == ref[i], where
            _check_range_state(rf, ref, where)
    assert sizes == set(range(1, 51))
    assert all(count > 0 for count in op_counts.values()), op_counts
    assert adds_to_end > 0
