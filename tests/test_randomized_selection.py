"""Tests for randomized QuickSelect, CLRS median-of-medians SELECT, and median().

Selection is easy to get almost right. An off-by-one in the 1-based ``k``,
a partition that loses an element equal to the pivot, or a two-way split
that never shrinks a run of duplicates all still return *a* value from the
list, and usually a plausible one. So every correctness claim here rests on
a witness that shares no mechanism with the module:

* **The hand-checked anchor.** ``[7, 10, 4, 3, 20, 15]`` sorts by eye to
  ``[3, 4, 7, 10, 15, 20]``, so k = 1 is **3**, k = 3 is **7** and k = 6 is
  **20**, for both algorithms and for every seed. The work counts of
  :func:`deterministic_select` on this list are traced by hand round by
  round in :class:`TestHandTracedDeterministicStats`.
* **Sorting.** ``sorted(arr)[k - 1]`` is the definition of the k-th
  smallest element. It is checked for *every* k on seeded random lists that
  are guaranteed by pigeonhole to hold duplicates, and on the Week 1
  ``sample_arrays`` battery from ``conftest.py``.
* **Exact expectations from two independent derivations.** For distinct
  keys the expected comparison count of randomized QuickSelect obeys a
  one-line recurrence (the pivot rank is uniform, the round costs
  ``s - 1``), and Knuth also gives it in closed form. Both are written here,
  checked against values worked by hand for n <= 3, checked against each
  other exactly in rational arithmetic, and only then used as the target
  for a seeded sample mean, with a tolerance of 5 standard errors.
* **The CLRS worst-case bound.** Median of medians keeps at most
  ``floor(7s/10 + 6)`` of ``s`` elements per round (CLRS section 9.3), so
  iterating that bound gives a ceiling on rounds and comparisons that holds
  for *every* input. Adversarial inputs are checked against it.

What this file proves about the Las Vegas nature of QuickSelect: over 200
seeds the answer is right every time (a success rate of exactly 1.0, not an
estimate), while the comparison count is genuinely random.

Every random draw is seeded, either through ``make_rng`` or an explicit
``seed=``, so nothing here can flake. No assertion is about time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import contextlib
import copy
import math
import statistics
import sys
from fractions import Fraction
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

from src.randomized.randomized_selection import (
    SelectionStats,
    deterministic_select,
    median,
    randomized_select,
)
from src.utils.random_utils import derive_seed, make_rng, random_distinct_list

Selector = Callable[..., Any]

#: The hand-checked anchor list and its order statistics, read off by eye
#: from the sorted list [3, 4, 7, 10, 15, 20].
ANCHOR: List[int] = [7, 10, 4, 3, 20, 15]
ANCHOR_SORTED: List[int] = [3, 4, 7, 10, 15, 20]

#: Seeds used wherever "several seeds" are wanted.
SEEDS: List[int] = [0, 1, 2, 7, 42, 1234, 2026, 99991]


def seeded(fn: Selector, seed: int) -> Selector:
    """Bind a seed to ``randomized_select``; leave the deterministic one alone."""
    if fn is randomized_select:
        return lambda arr, k, **kw: randomized_select(arr, k, seed=seed, **kw)
    return fn


SELECTORS: List[Any] = [
    pytest.param(randomized_select, id="randomized"),
    pytest.param(deterministic_select, id="deterministic"),
]


# ----------------------------------------------------------------------
# Independent oracles and helpers
# ----------------------------------------------------------------------
def list_with_duplicates(n: int, seed: int) -> List[int]:
    """Return ``n`` seeded integers that must contain a repeat when n >= 2.

    Values are drawn from ``(n - 1) // 2 + 1`` possibilities, which is
    fewer than ``n`` for every ``n >= 2``, so by pigeonhole at least two
    entries are equal. They are shifted to include negatives.
    """
    rng = make_rng(seed)
    span = (n - 1) // 2
    return [rng.randint(0, span) - span // 2 for _ in range(n)]


def harmonic(n: int) -> Fraction:
    """The n-th harmonic number ``H_n`` as an exact fraction; ``H_0 = 0``."""
    return sum((Fraction(1, i) for i in range(1, n + 1)), Fraction(0))


def knuth_expected(n: int, k: int) -> Fraction:
    """Knuth's closed form for the expected comparisons of QuickSelect.

    ``2 * ((n + 1) H_n - (n + 3 - k) H_{n+1-k} - (k + 2) H_k + n + 3)`` for
    ``n`` distinct keys, a uniform pivot, and ``s - 1`` comparisons for a
    round on ``s`` elements (TAOCP vol. 3, section 5.2.2, exercise 32).
    """
    return 2 * (
        (n + 1) * harmonic(n)
        - (n + 3 - k) * harmonic(n + 1 - k)
        - (k + 2) * harmonic(k)
        + n + 3
    )


def knuth_expected_float(n: int, k: int) -> float:
    """:func:`knuth_expected` in floating point, for ``n`` too large for fractions."""

    def h(m: int) -> float:
        return math.fsum(1.0 / i for i in range(1, m + 1))

    return 2 * ((n + 1) * h(n) - (n + 3 - k) * h(n + 1 - k) - (k + 2) * h(k) + n + 3)


def recurrence_expected(n_max: int) -> Dict[Tuple[int, int], Fraction]:
    """Expected comparisons for every ``1 <= k <= n <= n_max``, by recurrence.

    A round on ``n`` distinct keys costs ``n - 1``. The pivot's rank ``p``
    is uniform on ``1 .. n``. If ``p == k`` the run stops; if ``p > k`` it
    continues on the ``p - 1`` smaller keys with the same ``k``; if
    ``p < k`` it continues on the ``n - p`` larger keys looking for rank
    ``k - p``. No closed form, no harmonic numbers: a second, unrelated
    derivation of the same quantity.
    """
    table: Dict[Tuple[int, int], Fraction] = {(1, 1): Fraction(0)}
    for n in range(2, n_max + 1):
        for k in range(1, n + 1):
            rest = Fraction(0)
            for p in range(1, n + 1):
                if p < k:
                    rest += table[(n - p, k - p)]
                elif p > k:
                    rest += table[(p - 1, k)]
            table[(n, k)] = (n - 1) + rest / n
    return table


def clrs_worst_case(n: int) -> Tuple[int, int]:
    """Ceilings on (rounds, comparisons) for median-of-medians SELECT.

    CLRS section 9.3: at least ``3 * (ceil(ceil(s/5) / 2) - 2) >= 3s/10 - 6``
    elements are ``>=`` the pivot and as many ``<=`` it, so the side kept
    after a three-way split has at most ``floor(7s/10 + 6)`` elements. It
    also never keeps the pivot, so at most ``s - 1``. Both ceilings are
    non-decreasing in ``s``, so following them from ``n`` gives the most
    rounds and the most comparisons any input can force.
    """
    depth = comparisons = 0
    size = n
    while size > 1:
        depth += 1
        comparisons += size - 1
        size = min(size - 1, (7 * size) // 10 + 6)
    return depth, comparisons


def stack_depth() -> int:
    """Number of Python frames currently on the stack."""
    frame = sys._getframe()
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


@contextlib.contextmanager
def recursion_budget(extra: int) -> Iterator[None]:
    """Allow only ``extra`` more Python frames than are in use right now."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(stack_depth() + extra)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def two_way_recursive_select(items: List[int], target: int) -> int:
    """A deliberately naive recursive QuickSelect with a two-way split.

    Only used as a control: on equal values every element lands on the
    pivot's side, so it recurses once per element.
    """
    if len(items) == 1:
        return items[0]
    pivot = items[-1]
    left = [x for x in items[:-1] if x <= pivot]
    right = [x for x in items[:-1] if pivot < x]
    if target < len(left):
        return two_way_recursive_select(left, target)
    if target == len(left):
        return pivot
    return two_way_recursive_select(right, target - len(left) - 1)


class LessThanOnly:
    """A key that supports ``<`` and nothing else.

    ``==`` raises, so a module that compared elements any other way than
    ``<`` (the documented requirement) would fail loudly.
    """

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    def __lt__(self, other: "LessThanOnly") -> bool:
        return self.value < other.value

    def __eq__(self, other: object) -> bool:
        raise AssertionError("selection must compare elements with < only")


# ----------------------------------------------------------------------
# The hand-checked anchor
# ----------------------------------------------------------------------
class TestAnchor:
    """``[7, 10, 4, 3, 20, 15]``, sorted by eye to ``[3, 4, 7, 10, 15, 20]``."""

    @pytest.mark.parametrize("k, expected", [(1, 3), (3, 7), (6, 20)])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_randomized_anchor_for_every_seed(
        self, k: int, expected: int, seed: int
    ) -> None:
        assert randomized_select(ANCHOR, k, seed=seed) == expected

    @pytest.mark.parametrize("k, expected", [(1, 3), (3, 7), (6, 20)])
    def test_deterministic_anchor(self, k: int, expected: int) -> None:
        assert deterministic_select(ANCHOR, k) == expected

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_every_rank_of_the_anchor(self, fn: Selector, seed: int) -> None:
        select = seeded(fn, seed)
        assert [select(ANCHOR, k) for k in range(1, 7)] == ANCHOR_SORTED

    def test_unseeded_run_is_still_correct(self) -> None:
        # seed=None draws from the OS. The answer cannot depend on the draws.
        assert [randomized_select(ANCHOR, k) for k in (1, 3, 6)] == [3, 7, 20]

    @pytest.mark.parametrize("seed", SEEDS)
    def test_anchor_with_return_stats_gives_the_same_value(self, seed: int) -> None:
        value, _ = randomized_select(ANCHOR, 3, seed=seed, return_stats=True)
        assert value == 7
        value, _ = deterministic_select(ANCHOR, 3, return_stats=True)
        assert value == 7


class TestHandTracedDeterministicStats:
    """Median of medians on the anchor, traced by hand round by round.

    Round 1 on all six values: the groups are ``[7, 10, 4, 3, 20]`` (sorted
    ``[3, 4, 7, 10, 20]``, median 7) and the tail ``[15]`` (median 15). The
    lower median of ``[7, 15]`` is **7**, so the pivot is 7 and the round
    costs 6 - 1 = 5. The split is smaller ``[4, 3]``, equal ``[7]``,
    larger ``[10, 20, 15]``.

    * k = 3 lands in the equal block: answer 7 after **1 round, 5
      comparisons**.
    * k = 1, 2 continue on ``[4, 3]``; its lower median is 3, which costs 1
      more. k = 1 hits the pivot, k = 2 leaves ``[4]``. **2 rounds, 6.**
    * k = 4, 5, 6 continue on ``[10, 20, 15]``; its median is 15, which
      costs 2 more. k = 5 hits the pivot, k = 4 leaves ``[10]``, k = 6
      leaves ``[20]``. **2 rounds, 7.**
    """

    TRACE = [
        pytest.param(1, 3, 2, 6, id="k1"),
        pytest.param(2, 4, 2, 6, id="k2"),
        pytest.param(3, 7, 1, 5, id="k3"),
        pytest.param(4, 10, 2, 7, id="k4"),
        pytest.param(5, 15, 2, 7, id="k5"),
        pytest.param(6, 20, 2, 7, id="k6"),
    ]

    @pytest.mark.parametrize("k, value, depth, comparisons", TRACE)
    def test_hand_traced_rounds_and_comparisons(
        self, k: int, value: int, depth: int, comparisons: int
    ) -> None:
        result = deterministic_select(ANCHOR, k, return_stats=True)
        assert result == (value, SelectionStats(depth, comparisons))

    def test_no_randomness_so_repeat_calls_agree(self) -> None:
        first = [deterministic_select(ANCHOR, k, True) for k in range(1, 7)]
        again = [deterministic_select(ANCHOR, k, True) for k in range(1, 7)]
        assert first == again


# ----------------------------------------------------------------------
# Agreement with sorting
# ----------------------------------------------------------------------
class TestAgreesWithSorting:
    """``sorted(arr)[k - 1]`` is the definition; both algorithms must match it."""

    LENGTHS = list(range(1, 21)) + [33, 64, 101]

    def test_battery_really_has_duplicates(self) -> None:
        for n in self.LENGTHS:
            if n >= 2:
                values = list_with_duplicates(n, derive_seed(7, "dup-check", n))
                assert len(set(values)) < len(values)

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("seed", range(10))
    def test_every_k_on_lists_with_duplicates(self, fn: Selector, seed: int) -> None:
        for n in self.LENGTHS:
            values = list_with_duplicates(n, derive_seed(seed, "battery", n))
            truth = sorted(values)
            select = seeded(fn, derive_seed(seed, "pivots", n))
            for k in range(1, n + 1):
                assert select(values, k) == truth[k - 1], f"n={n}, k={k}"

    @pytest.mark.parametrize("seed", range(4))
    def test_shared_rng_drives_a_whole_battery(self, seed: int) -> None:
        # One generator for many calls, the rng= path the benchmark uses.
        gen = make_rng(seed)
        values = list_with_duplicates(80, derive_seed(seed, "shared"))
        truth = sorted(values)
        assert all(
            randomized_select(values, k, rng=gen) == truth[k - 1] for k in range(1, 81)
        )

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_every_k_on_the_week1_sample_arrays(
        self, fn: Selector, sample_arrays: Dict[str, List[Any]]
    ) -> None:
        select = seeded(fn, 2026)
        for name, data in sample_arrays.items():
            if not data:
                continue
            truth = sorted(data)
            for k in range(1, len(data) + 1):
                assert select(data, k) == truth[k - 1], f"{name}, k={k}"

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("seed", range(3))
    def test_elements_need_only_less_than(self, fn: Selector, seed: int) -> None:
        raw = list_with_duplicates(40, derive_seed(seed, "lt-only"))
        keys = [LessThanOnly(v) for v in raw]
        truth = sorted(raw)
        select = seeded(fn, seed)
        for k in range(1, 41):
            assert select(keys, k).value == truth[k - 1]

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_mixed_ints_and_floats(self, fn: Selector) -> None:
        data = [2.5, -1, 0, 3, -1.0, 2, 0.5]
        # Sorted by hand: -1, -1.0, 0, 0.5, 2, 2.5, 3.
        expected = [-1, -1, 0, 0.5, 2, 2.5, 3]
        select = seeded(fn, 5)
        assert [select(data, k) for k in range(1, 8)] == expected

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_permutation_of_range_gives_rank_minus_one(self, fn: Selector) -> None:
        # In a permutation of 0..n-1 the k-th smallest is k - 1, no sort needed.
        perm = random_distinct_list(250, 11)
        select = seeded(fn, 11)
        assert all(select(perm, k) == k - 1 for k in range(1, 251))


# ----------------------------------------------------------------------
# Validation and non-destruction
# ----------------------------------------------------------------------
class TestValidation:
    """k is 1-based and checked; the caller's list is never touched."""

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("k", [0, 7, -1, 100], ids=["k0", "len+1", "neg", "far"])
    def test_k_out_of_range(self, fn: Selector, k: int) -> None:
        with pytest.raises(ValueError, match="1 <= k <= 6"):
            fn(list(ANCHOR), k)

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("k", [0, 1])
    def test_empty_list(self, fn: Selector, k: int) -> None:
        with pytest.raises(ValueError, match="empty"):
            fn([], k)

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize(
        "arr",
        [
            pytest.param(tuple(ANCHOR), id="tuple"),
            pytest.param("71043", id="str"),
            pytest.param(None, id="None"),
            pytest.param(range(6), id="range"),
            pytest.param({7: 1, 10: 2}, id="dict"),
            pytest.param(iter(ANCHOR), id="iterator"),
        ],
    )
    def test_non_list_input(self, fn: Selector, arr: Any) -> None:
        with pytest.raises(TypeError, match="expects a list"):
            fn(arr, 1)

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize(
        "k",
        [
            pytest.param(True, id="True"),
            pytest.param(False, id="False"),
            pytest.param(2.0, id="float"),
            pytest.param("2", id="str"),
            pytest.param(None, id="None"),
        ],
    )
    def test_non_int_or_bool_k(self, fn: Selector, k: Any) -> None:
        with pytest.raises(TypeError, match="expects an int k"):
            fn(list(ANCHOR), k)

    @pytest.mark.parametrize("seed", ["42", 4.2, True])
    def test_bad_seed_is_a_type_error(self, seed: Any) -> None:
        with pytest.raises(TypeError):
            randomized_select(list(ANCHOR), 1, seed=seed)

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_input_not_mutated(self, fn: Selector) -> None:
        data = list_with_duplicates(60, 3)
        before = copy.deepcopy(data)
        select = seeded(fn, 3)
        for k in range(1, 61):
            select(data, k, return_stats=True)
            select(data, k)
        assert data == before

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_sample_arrays_not_mutated(
        self, fn: Selector, sample_arrays: Dict[str, List[Any]]
    ) -> None:
        select = seeded(fn, 8)
        for name, data in sample_arrays.items():
            if not data:
                continue
            before = list(data)
            select(data, (len(data) + 1) // 2)
            select(data, len(data))
            assert data == before, name

    def test_median_does_not_mutate(self) -> None:
        data = list(ANCHOR)
        median(data, seed=1)
        assert data == ANCHOR


# ----------------------------------------------------------------------
# SelectionStats
# ----------------------------------------------------------------------
class TestSelectionStats:
    """Rounds and comparisons, under the module's documented convention.

    A round on ``s`` elements costs exactly ``s - 1``; a range of one
    element is not partitioned. Every bound below follows from that alone:

    * n >= 2 needs at least one round, and the first costs ``n - 1``.
    * Each later round runs on at least 2 elements, so costs at least 1:
      ``comparisons >= (n - 1) + (depth - 1)``.
    * The pivot never survives a round, so round ``i`` (from 0) runs on at
      most ``n - i`` elements: ``depth <= n - 1`` and
      ``comparisons <= (n - 1) + (n - 2) + ... + (n - depth)``.
    """

    @staticmethod
    def assert_within_counting_bounds(stats: SelectionStats, n: int) -> None:
        depth, comparisons = stats
        if n == 1:
            assert stats == (0, 0)
            return
        assert 1 <= depth <= n - 1
        assert comparisons >= (n - 1) + (depth - 1)
        assert comparisons <= sum(n - 1 - i for i in range(depth))

    def test_is_a_named_tuple_with_two_int_fields(self) -> None:
        _, stats = randomized_select(ANCHOR, 2, seed=1, return_stats=True)
        assert isinstance(stats, SelectionStats)
        assert isinstance(stats, tuple)
        assert SelectionStats._fields == ("depth", "comparisons")
        assert isinstance(stats.depth, int) and isinstance(stats.comparisons, int)
        assert (stats.depth, stats.comparisons) == tuple(stats)

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_return_shape(self, fn: Selector) -> None:
        result = seeded(fn, 4)(ANCHOR, 4, return_stats=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == 10
        assert isinstance(result[1], SelectionStats)
        assert seeded(fn, 4)(ANCHOR, 4) == 10

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_single_element_needs_no_round(self, fn: Selector) -> None:
        assert seeded(fn, 0)([42], 1, return_stats=True) == (42, SelectionStats(0, 0))

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("k", [1, 2])
    def test_two_distinct_values_take_exactly_one_round(
        self, fn: Selector, seed: int, k: int
    ) -> None:
        # Either pivot finishes it: the target is the pivot, or the other
        # side is a single element. One round on 2 elements costs 1.
        value, stats = seeded(fn, seed)([9, 4], k, return_stats=True)
        assert value == [4, 9][k - 1]
        assert stats == SelectionStats(1, 1)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_three_distinct_values_take_one_or_two_rounds(self, k: int) -> None:
        # Round 1 costs 2. If it does not stop, 2 elements remain and one
        # more round of cost 1 always ends it. Nothing else is possible.
        seen = set()
        for seed in range(60):
            value, stats = randomized_select([8, 2, 5], k, seed=seed, return_stats=True)
            assert value == [2, 5, 8][k - 1]
            assert stats in {SelectionStats(1, 2), SelectionStats(2, 3)}
            seen.add(stats)
        assert seen == {SelectionStats(1, 2), SelectionStats(2, 3)}

    @pytest.mark.parametrize("fn", SELECTORS)
    @pytest.mark.parametrize("seed", range(6))
    def test_counting_bounds_on_every_run(self, fn: Selector, seed: int) -> None:
        for n in (1, 2, 3, 5, 8, 17, 40, 97):
            values = list_with_duplicates(n, derive_seed(seed, "bounds", n))
            select = seeded(fn, derive_seed(seed, "bounds-pivots", n))
            for k in range(1, n + 1):
                _, stats = select(values, k, return_stats=True)
                self.assert_within_counting_bounds(stats, n)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_anchor_stats_within_bounds(self, seed: int) -> None:
        for k in range(1, 7):
            _, stats = randomized_select(ANCHOR, k, seed=seed, return_stats=True)
            assert stats.depth >= 1 and stats.comparisons >= 5
            self.assert_within_counting_bounds(stats, 6)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_same_seed_same_stats(self, seed: int) -> None:
        data = list_with_duplicates(150, 21)
        for k in (1, 75, 150):
            first = randomized_select(data, k, seed=seed, return_stats=True)
            again = randomized_select(data, k, seed=seed, return_stats=True)
            assert first == again

    def test_rng_is_equivalent_to_the_same_seed(self) -> None:
        # make_rng(seed) is the generator seed= builds, so the draws match.
        data = random_distinct_list(120, 5)
        for seed in SEEDS:
            assert randomized_select(
                data, 60, rng=make_rng(seed), return_stats=True
            ) == randomized_select(data, 60, seed=seed, return_stats=True)

    def test_rng_takes_precedence_over_seed(self) -> None:
        data = random_distinct_list(120, 6)
        for seed in SEEDS:
            a = randomized_select(
                data, 60, seed=1, rng=make_rng(seed), return_stats=True
            )
            b = randomized_select(
                data, 60, seed=999, rng=make_rng(seed), return_stats=True
            )
            assert a == b

    def test_one_rng_replays_a_sequence_of_calls(self) -> None:
        data = random_distinct_list(90, 7)

        def run(gen: Any) -> List[Any]:
            return [
                randomized_select(data, k, rng=gen, return_stats=True)
                for k in range(1, 91)
            ]

        assert run(make_rng(31)) == run(make_rng(31))

    def test_different_seeds_give_different_stats(self) -> None:
        data = random_distinct_list(100, 8)
        stats = {
            randomized_select(data, 50, seed=s, return_stats=True)[1]
            for s in range(40)
        }
        assert len(stats) > 1


# ----------------------------------------------------------------------
# Equal values and recursion depth
# ----------------------------------------------------------------------
class TestIdenticalValues:
    """The three-way split ends a run of equal values in one round.

    Whatever pivot is drawn, every element is equal to it, the target is in
    the equal block, and the run stops: **1 round, n - 1 comparisons**. A
    two-way split would instead peel off one element per round.
    """

    @pytest.mark.parametrize("n", [2, 1000, 10_000])
    @pytest.mark.parametrize("k_name", ["min", "median", "max"])
    def test_deterministic_one_round(self, n: int, k_name: str) -> None:
        k = {"min": 1, "median": (n + 1) // 2, "max": n}[k_name]
        assert deterministic_select([5] * n, k, return_stats=True) == (
            5,
            SelectionStats(1, n - 1),
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_randomized_one_round(self, seed: int) -> None:
        for k in (1, 500, 1000):
            assert randomized_select([5] * 1000, k, seed=seed, return_stats=True) == (
                5,
                SelectionStats(1, 999),
            )

    def test_budget_catches_deep_recursion(self) -> None:
        # Control: the budget really does stop a selector that recurses once
        # per element, so the next two tests mean something.
        with pytest.raises(RecursionError):
            with recursion_budget(60):
                two_way_recursive_select([5] * 1000, 0)

    @pytest.mark.parametrize("n", [1000, 10_000])
    def test_deterministic_on_identical_values_needs_no_deep_stack(
        self, n: int
    ) -> None:
        with recursion_budget(60):
            for k in (1, (n + 1) // 2, n):
                assert deterministic_select([3] * n, k) == 3

    def test_deterministic_on_distinct_values_needs_no_deep_stack(self) -> None:
        # Real recursion only goes through the median of medians, on a list
        # a fifth the size each time: O(log n) frames.
        perm = random_distinct_list(5000, 12)
        with recursion_budget(60):
            for k in (1, 2500, 5000):
                assert deterministic_select(perm, k) == k - 1

    def test_randomized_is_iterative(self) -> None:
        perm = random_distinct_list(5000, 13)
        with recursion_budget(20):
            assert randomized_select([3] * 1000, 1, seed=1) == 3
            assert randomized_select(perm, 2500, seed=1) == 2499

    @pytest.mark.parametrize("fn", SELECTORS)
    def test_two_values_many_copies(self, fn: Selector) -> None:
        data = [1, 0] * 500
        select = seeded(fn, 14)
        assert select(data, 1) == 0 and select(data, 500) == 0
        assert select(data, 501) == 1 and select(data, 1000) == 1
        _, stats = select(data, 500, return_stats=True)
        # One round splits into [0]*500, [pivot block], and the other side;
        # whichever value is the pivot, at most 2 rounds can be needed.
        assert stats.depth <= 2


# ----------------------------------------------------------------------
# Median of medians: the worst case really is linear
# ----------------------------------------------------------------------
class TestDeterministicWorstCaseBound:
    """Every input stays under the CLRS ceiling from :func:`clrs_worst_case`."""

    def test_bound_hand_values(self) -> None:
        # n = 2: one round of 1. n = 6: 6 -> min(5, 4 + 6) = 5 -> 4 -> 3 -> 2
        # -> 1, so 5 rounds and 5 + 4 + 3 + 2 + 1 = 15 comparisons.
        assert clrs_worst_case(1) == (0, 0)
        assert clrs_worst_case(2) == (1, 1)
        assert clrs_worst_case(6) == (5, 15)
        # n = 100: 100 -> 76 -> 59 -> 47 -> 38 -> 32 -> 28 -> 25 -> 23 -> 22
        # -> 21 -> 20, then one at a time down to 1.
        depth, comparisons = clrs_worst_case(100)
        assert depth == 11 + 19
        geometric_part = 99 + 75 + 58 + 46 + 37 + 31 + 27 + 24 + 22 + 21 + 20
        assert comparisons == geometric_part + sum(range(1, 20))

    def test_bound_is_linear(self) -> None:
        # The geometric part sums to n / (1 - 7/10), so about 3.33n plus a
        # constant from the small sizes.
        for n in (1000, 10_000, 100_000):
            assert clrs_worst_case(n)[1] <= 3.4 * n + 400

    ADVERSARIAL = {
        "ascending": lambda n: list(range(n)),
        "descending": lambda n: list(range(n, 0, -1)),
        "organ_pipe": lambda n: list(range(n // 2)) + list(range(n - n // 2, 0, -1)),
        "sawtooth": lambda n: [i % 7 for i in range(n)],
        "all_equal": lambda n: [4] * n,
        "two_values": lambda n: [i % 2 for i in range(n)],
        "duplicates": lambda n: list_with_duplicates(n, 17),
        "random": lambda n: random_distinct_list(n, 18),
    }

    @pytest.mark.parametrize("shape", sorted(ADVERSARIAL))
    @pytest.mark.parametrize("n", [7, 100, 1000])
    def test_never_exceeds_the_clrs_ceiling(self, shape: str, n: int) -> None:
        data = self.ADVERSARIAL[shape](n)
        truth = sorted(data)
        max_depth, max_comparisons = clrs_worst_case(n)
        for k in (1, 2, (n + 1) // 2, n - 1, n):
            value, stats = deterministic_select(data, k, return_stats=True)
            assert value == truth[k - 1]
            assert stats.depth <= max_depth
            assert stats.comparisons <= max_comparisons


# ----------------------------------------------------------------------
# median()
# ----------------------------------------------------------------------
class TestMedian:
    """The lower median, rank ``(n + 1) // 2``, values worked by hand."""

    HAND = [
        pytest.param([42], 42, id="single"),
        pytest.param([5, 1, 3], 3, id="odd-3"),
        pytest.param([2, 1], 1, id="even-2-lower-not-mean"),
        # Sorted [3, 4, 7, 10, 15, 20]: lower middle 7, upper 10, mean 8.5.
        pytest.param(list(ANCHOR), 7, id="anchor-even-6"),
        pytest.param([9, 2, 6, 4, 8, 1, 7, 3, 5], 5, id="odd-9"),
        pytest.param([8, 1, 6, 3, 2, 7, 5, 4], 4, id="even-8"),
        pytest.param([5, 5, 1, 1], 1, id="even-duplicates"),
        pytest.param([0, 0, 0, 9, 9], 0, id="odd-duplicates"),
        pytest.param(["pear", "apple", "fig"], "fig", id="strings"),
        pytest.param([-3.5, 2.0, -0.5, 1.5], -0.5, id="floats-even"),
    ]

    @pytest.mark.parametrize("data, expected", HAND)
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_hand_values(self, data: List[Any], expected: Any, seed: int) -> None:
        assert median(data, seed=seed) == expected

    def test_unseeded(self) -> None:
        assert median(list(ANCHOR)) == 7

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_sorted_lower_middle_for_odd_and_even(self, seed: int) -> None:
        for n in range(1, 41):
            data = list_with_duplicates(n, derive_seed(seed, "median", n))
            # (n + 1) // 2 as a 1-based rank is (n - 1) // 2 as an index.
            assert median(data, seed=seed) == sorted(data)[(n - 1) // 2]

    @pytest.mark.parametrize("n", [1, 2, 3, 10, 11, 100, 101])
    def test_permutation_median_is_the_lower_middle(self, n: int) -> None:
        # In a permutation of 0..n-1 the lower median is (n - 1) // 2.
        assert median(random_distinct_list(n, n), seed=n) == (n - 1) // 2

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            median([], seed=1)

    @pytest.mark.parametrize("arr", [tuple(ANCHOR), None, "abc"])
    def test_non_list(self, arr: Any) -> None:
        with pytest.raises(TypeError, match="expects a list"):
            median(arr, seed=1)

    def test_bad_seed(self) -> None:
        with pytest.raises(TypeError):
            median(list(ANCHOR), seed="1")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Las Vegas: always right, randomly fast
# ----------------------------------------------------------------------
class TestLasVegas:
    """Correctness is certain; only the work is random.

    The expected work is checked against an exact value. For ``n`` distinct
    keys the expectation does not depend on the input order at all, only on
    the pivot draws, so one fixed permutation stands for every input.
    """

    TRIALS = 200

    @pytest.mark.parametrize("k_name", ["min", "median", "max"])
    def test_success_rate_is_exactly_one_and_work_varies(self, k_name: str) -> None:
        data = list_with_duplicates(301, 77)
        n = len(data)
        k = {"min": 1, "median": (n + 1) // 2, "max": n}[k_name]
        truth = sorted(data)[k - 1]
        successes = 0
        counts = []
        for seed in range(self.TRIALS):
            value, stats = randomized_select(data, k, seed=seed, return_stats=True)
            successes += value == truth
            counts.append(stats.comparisons)
        assert successes / self.TRIALS == 1.0
        assert len(set(counts)) > 1
        assert min(counts) >= n - 1

    def test_success_rate_is_one_on_distinct_keys_for_every_k(self) -> None:
        perm = random_distinct_list(40, 78)
        for k in range(1, 41):
            assert all(
                randomized_select(perm, k, seed=s) == k - 1 for s in range(self.TRIALS)
            )

    @pytest.mark.parametrize(
        "n, k, expected",
        [
            (1, 1, Fraction(0)),
            (2, 1, Fraction(1)),
            (2, 2, Fraction(1)),
            # Round 1 costs 2. Pivot = min stops (k=1) or leaves 2 (k=3);
            # pivot = middle always leaves 1 element; pivot = max leaves 2
            # (k=1) or stops (k=3). So 2 + 1/3 = 7/3 at the ends.
            (3, 1, Fraction(7, 3)),
            (3, 3, Fraction(7, 3)),
            # k=2: stops when the pivot is the middle (1/3), otherwise one
            # more round of 1. So 2 + 2/3 = 8/3.
            (3, 2, Fraction(8, 3)),
        ],
    )
    def test_both_oracles_match_hand_values(
        self, n: int, k: int, expected: Fraction
    ) -> None:
        assert knuth_expected(n, k) == expected
        assert recurrence_expected(3)[(n, k)] == expected

    def test_closed_form_and_recurrence_agree_exactly(self) -> None:
        table = recurrence_expected(12)
        for (n, k), value in table.items():
            assert knuth_expected(n, k) == value, f"n={n}, k={k}"

    def test_median_constant_tends_to_two_plus_two_ln_two(self) -> None:
        # Knuth: about 2(1 + ln 2) n = 3.386n at the median, 2n at the ends.
        for k in (1, 2, 30, 59, 60):
            assert math.isclose(
                knuth_expected_float(60, k), float(knuth_expected(60, k)), rel_tol=1e-12
            )
        n = 20_001
        ratio = knuth_expected_float(n, (n + 1) // 2) / n
        assert abs(ratio - 2 * (1 + math.log(2))) < 0.01
        assert abs(knuth_expected_float(n, 1) / n - 2) < 0.01

    @pytest.mark.parametrize("k_name", ["min", "median", "max"])
    def test_sample_mean_within_five_standard_errors(self, k_name: str) -> None:
        n, trials = 60, 1500
        k = {"min": 1, "median": (n + 1) // 2, "max": n}[k_name]
        perm = random_distinct_list(n, 79)
        counts = []
        for t in range(trials):
            seed = derive_seed(79, k_name, t)
            _, stats = randomized_select(perm, k, seed=seed, return_stats=True)
            counts.append(stats.comparisons)
        exact = float(knuth_expected(n, k))
        mean = statistics.fmean(counts)
        standard_error = statistics.stdev(counts) / math.sqrt(trials)
        assert standard_error > 0
        assert abs(mean - exact) <= 5 * standard_error, (mean, exact, standard_error)
