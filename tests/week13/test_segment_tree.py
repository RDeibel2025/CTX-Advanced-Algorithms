"""Week 13 tests: the segment tree with lazy range add (spec 9.1 anchors).

Every expected value is from the spec 9.1 anchors or worked by hand, never
from the code under test. Plain Python lists with ``sum``, ``min`` and
``max`` over slices are the independent reference.

Hand check of the anchors on ``A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]``
(half-open ranges):

* ``query_sum(2, 7) = 8 + 1 + 9 + 3 + 7 = 28``;
  ``query_min(1, 5) = min(2, 8, 1, 9) = 1``; ``query_max(0, 10) = 9``;
  ``query_min(5, 10) = min(3, 7, 4, 6, 0) = 0``.
* ``range_add(3, 8, 10)`` adds 10 to positions 3..7, giving
  ``[5, 2, 8, 11, 19, 13, 17, 14, 6, 0]``: the total is 45 + 5 x 10 = 95,
  the minimum is still 0 (position 9 is untouched), the maximum over
  ``[2, 6)`` is ``max(8, 11, 19, 13) = 19`` and the sum over ``[6, 10)`` is
  ``17 + 14 + 6 + 0 = 37``.

The random check replays 300 seeded operation sequences (sizes 1..50, each
size six times) mixing ``update``, ``range_add``, the three queries and
``to_list`` against a plain list. After every step the whole state is
compared through ``to_list()`` on an independent copy of the tree (a pickle
round trip), so the comparison never pushes the original tree's pending
tags: pushing them would hand the next operation a clean tree and hide a
lazy-propagation bug.

No timings are asserted.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from src.advds.trees.segment_tree import SegmentTree
from src.utils.random_utils import derive_seed, make_rng

SEED = 42

A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]

#: Operations of the random sequences; range_add appears twice so pending
#: tags pile up between pushes.
OPS = ("update", "range_add", "range_add", "sum", "min", "max", "to_list")

N_SEQUENCES = 300
STEPS = 40


def _snapshot(tree: SegmentTree) -> list:
    """The tree's current values, read from an independent copy so no tag on ``tree`` moves."""
    return pickle.loads(pickle.dumps(tree)).to_list()


def _random_range(rng, n: int):
    """A uniformly chosen pair ``0 <= l <= r <= n`` (empty ranges included)."""
    a, b = rng.randint(0, n), rng.randint(0, n)
    return (a, b) if a <= b else (b, a)


# ------------------------------------------------------------------ anchors


def test_anchor_queries():
    st = SegmentTree(A)
    assert st.query_sum(2, 7) == 28
    assert st.query_min(1, 5) == 1
    assert st.query_max(0, 10) == 9
    assert st.query_min(5, 10) == 0


def test_anchor_after_range_add():
    st = SegmentTree(A)
    st.range_add(3, 8, 10)
    assert st.query_sum(0, 10) == 95
    assert st.query_min(0, 10) == 0
    assert st.query_max(2, 6) == 19
    assert st.query_sum(6, 10) == 37
    assert st.to_list() == [5, 2, 8, 11, 19, 13, 17, 14, 6, 0]


def test_anchor_queries_in_reverse_order_after_range_add():
    """The anchors do not depend on which query meets the pending tags first."""
    st = SegmentTree(A)
    st.range_add(3, 8, 10)
    assert st.query_sum(6, 10) == 37
    assert st.query_max(2, 6) == 19
    assert st.query_min(0, 10) == 0
    assert st.query_sum(0, 10) == 95


# ------------------------------------------------------------ the structure


def test_capacity_is_next_power_of_two():
    for n, expected in [(1, 1), (2, 2), (3, 4), (5, 8), (10, 16), (16, 16), (17, 32), (50, 64)]:
        st = SegmentTree(list(range(n)))
        assert st.capacity == expected
        assert len(st) == n


def test_empty_leaves_do_not_leak_into_answers():
    # n = 10 leaves 6 empty slots: max -inf and min +inf there, sum 0.
    neg = [-5, -3, -8, -1, -9, -2, -7, -4, -6, -10]
    st = SegmentTree(neg)
    assert st.query_max(0, 10) == -1
    assert st.query_sum(0, 10) == -55
    pos = [x + 20 for x in A]
    st = SegmentTree(pos)
    assert st.query_min(0, 10) == 20
    st.range_add(0, 10, -100)
    assert st.query_max(0, 10) == 9 - 80
    assert st.query_min(0, 10) == -80
    assert st.query_sum(0, 10) == 245 - 1000


def test_update_inside_tagged_region():
    # Hand worked: A + 5 = [10, 7, 13, 6, 14, 8, 12, 9, 11, 5], then position 4 := 0.
    st = SegmentTree(A)
    st.range_add(0, 10, 5)
    st.update(4, 0)
    assert st.query_sum(0, 10) == 81
    assert st.query_min(0, 10) == 0
    assert st.query_max(0, 10) == 13
    assert st.query_sum(4, 5) == 0
    assert st.query_sum(3, 6) == 6 + 0 + 8
    assert st.to_list() == [10, 7, 13, 6, 0, 8, 12, 9, 11, 5]


def test_overlapping_range_adds():
    # [0]*8; +1 on [0, 6), +2 on [2, 8), -3 on [3, 5) -> [1, 1, 3, 0, 0, 3, 2, 2].
    st = SegmentTree([0] * 8)
    st.range_add(0, 6, 1)
    st.range_add(2, 8, 2)
    st.range_add(3, 5, -3)
    assert st.query_sum(0, 8) == 12
    assert st.query_min(2, 6) == 0
    assert st.query_max(5, 8) == 3
    assert st.to_list() == [1, 1, 3, 0, 0, 3, 2, 2]


def test_to_list_does_not_change_answers():
    st = SegmentTree(A)
    st.range_add(1, 9, 3)
    before = (st.query_sum(0, 10), st.query_min(2, 9), st.query_max(0, 4))
    assert st.to_list() == [5, 5, 11, 4, 12, 6, 10, 7, 9, 0]
    assert (st.query_sum(0, 10), st.query_min(2, 9), st.query_max(0, 4)) == before
    assert st.to_list() == [5, 5, 11, 4, 12, 6, 10, 7, 9, 0]


def test_single_element():
    st = SegmentTree([7])
    assert st.capacity == 1
    assert (st.query_sum(0, 1), st.query_min(0, 1), st.query_max(0, 1)) == (7, 7, 7)
    st.range_add(0, 1, -2)
    assert st.query_sum(0, 1) == 5
    st.update(0, 11)
    assert st.to_list() == [11]


def test_values_are_copied_and_numpy_input_becomes_python_ints():
    src = [3, 1, 2]
    st = SegmentTree(src)
    st.update(0, 100)
    assert src == [3, 1, 2]
    arr = np.array([3, 1, 2], dtype=np.int64)
    st = SegmentTree(arr)
    out = st.to_list()
    assert out == [3, 1, 2]
    assert all(type(v) is int for v in out)
    assert type(st.query_sum(0, 3)) is int


def test_float_values_and_numpy_scalars():
    st = SegmentTree([0.5, 1.25, -2.0, 4.0])
    assert st.query_sum(0, 4) == 3.75
    st.range_add(1, 3, 0.25)
    assert st.query_min(0, 4) == -1.75
    assert st.query_max(0, 2) == 1.5
    st.update(np.int64(3), np.float64(-8.5))
    assert st.query_min(0, 4) == -8.5
    assert st.to_list() == [0.5, 1.5, -1.75, -8.5]


def test_large_integers_stay_exact():
    big = 2**70
    st = SegmentTree([big, big, 1])
    st.range_add(0, 3, big)
    assert st.query_sum(0, 3) == 5 * big + 1


# ------------------------------------------------------- empty ranges and errors


def test_empty_range_behaviour():
    st = SegmentTree(A)
    for k in (0, 3, 10):
        assert st.query_sum(k, k) == 0
        with pytest.raises(ValueError):
            st.query_min(k, k)
        with pytest.raises(ValueError):
            st.query_max(k, k)
        st.range_add(k, k, 99)
    assert st.to_list() == A


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        SegmentTree([])
    with pytest.raises(ValueError):
        SegmentTree(np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        SegmentTree(iter(()))


@pytest.mark.parametrize(
    "bad",
    [5, None, "123", [1, "2"], [True, 2], [1, None], [[1, 2]], np.zeros((2, 2)), [1 + 2j]],
)
def test_bad_values_type_rejected(bad):
    with pytest.raises(TypeError):
        SegmentTree(bad)


def test_nan_rejected():
    with pytest.raises(ValueError):
        SegmentTree([1.0, math.nan])
    st = SegmentTree(A)
    with pytest.raises(ValueError):
        st.update(0, math.nan)
    with pytest.raises(ValueError):
        st.range_add(0, 3, math.nan)
    assert st.to_list() == A


@pytest.mark.parametrize("inf", [math.inf, -math.inf, np.float64(np.inf)])
def test_infinities_rejected(inf):
    """Regression: infinities used to be accepted (``SegmentTree([inf, 1.0])``
    then ``range_add(0, 2, -inf)`` stored NaN). Values and deltas must be finite."""
    with pytest.raises(ValueError):
        SegmentTree([1.0, inf])
    with pytest.raises(ValueError):
        SegmentTree(np.array([1.0, inf]))
    st = SegmentTree(A)
    with pytest.raises(ValueError):
        st.update(0, inf)
    with pytest.raises(ValueError):
        st.range_add(0, 10, inf)
    assert st.to_list() == A
    # Huge but finite values are still accepted.
    big = SegmentTree([1e300, -1e300, 2**100])
    assert (big.query_min(0, 3), big.query_max(0, 3)) == (-1e300, 1e300)


@pytest.mark.parametrize("method", ["query_sum", "query_min", "query_max"])
def test_query_argument_errors(method):
    st = SegmentTree(A)
    query = getattr(st, method)
    with pytest.raises(IndexError):
        query(-1, 3)
    with pytest.raises(IndexError):
        query(0, 11)
    with pytest.raises(IndexError):
        query(11, 11)
    with pytest.raises(ValueError):
        query(5, 3)
    with pytest.raises(TypeError):
        query(1.0, 3)
    with pytest.raises(TypeError):
        query(0, True)


def test_numpy_integer_indices_accepted():
    st = SegmentTree(A)
    assert st.query_sum(np.int64(2), np.int32(7)) == 28
    assert st.query_min(np.int64(1), np.int64(5)) == 1
    st.range_add(np.int64(3), np.int64(8), np.int64(10))
    st.update(np.int32(0), 50)
    assert st.query_max(np.int64(0), np.int64(10)) == 50
    assert st.query_sum(0, 10) == 140


def test_update_and_range_add_argument_errors():
    st = SegmentTree(A)
    with pytest.raises(IndexError):
        st.update(10, 1)
    with pytest.raises(IndexError):
        st.update(-1, 1)
    with pytest.raises(TypeError):
        st.update(2.0, 1)
    with pytest.raises(TypeError):
        st.update(2, "1")
    with pytest.raises(TypeError):
        st.update(2, False)
    with pytest.raises(IndexError):
        st.range_add(0, 11, 1)
    with pytest.raises(IndexError):
        st.range_add(-2, 4, 1)
    with pytest.raises(ValueError):
        st.range_add(4, 2, 1)
    with pytest.raises(TypeError):
        st.range_add(0, 4, None)
    assert st.to_list() == A


# ----------------------------------------------------------- random sequences


def test_random_operation_sequences_match_a_list():
    op_counts = {op: 0 for op in OPS}
    sizes = set()
    empty_ranges = 0
    for case in range(N_SEQUENCES):
        n = 1 + case % 50
        sizes.add(n)
        rng = make_rng(derive_seed(SEED, "segment_tree", case))
        ref = [rng.randint(-100, 100) for _ in range(n)]
        st = SegmentTree(ref)
        for step in range(STEPS):
            op = rng.choice(OPS)
            l, r = _random_range(rng, n)
            op_counts[op] += 1
            empty_ranges += l == r
            where = f"case {case}, n {n}, step {step}, {op}({l}, {r})"
            if op == "update":
                i, value = rng.randrange(n), rng.randint(-100, 100)
                st.update(i, value)
                ref[i] = value
            elif op == "range_add":
                delta = rng.randint(-50, 50)
                st.range_add(l, r, delta)
                for k in range(l, r):
                    ref[k] += delta
            elif op == "sum":
                assert st.query_sum(l, r) == sum(ref[l:r]), where
            elif op == "min":
                if l == r:
                    with pytest.raises(ValueError):
                        st.query_min(l, r)
                else:
                    assert st.query_min(l, r) == min(ref[l:r]), where
            elif op == "max":
                if l == r:
                    with pytest.raises(ValueError):
                        st.query_max(l, r)
                else:
                    assert st.query_max(l, r) == max(ref[l:r]), where
            else:
                assert st.to_list() == ref, where
            assert _snapshot(st) == ref, where
        assert st.to_list() == ref
    # The sequences really span sizes 1..50, draw every operation and hit empty ranges.
    assert sizes == set(range(1, 51))
    assert all(count > 0 for count in op_counts.values()), op_counts
    assert sum(op_counts.values()) == N_SEQUENCES * STEPS
    assert empty_ranges > 0
