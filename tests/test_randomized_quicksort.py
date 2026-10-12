"""Tests for randomized QuickSort, its deterministic baseline, and the CLRS expectation.

:func:`randomized_quicksort` is a LAS VEGAS algorithm: its output is always
right and only its cost is random. This file proves those two kinds of claim
separately, and never checks a number against the module's own output:

* **The output never depends on luck.** Every case in the Week 1
  ``sample_arrays`` battery comes back sorted, as a permutation of the input,
  in a new list, with the caller's list untouched, under several seeds and
  under both deterministic pivot rules. :class:`TestExhaustiveDraws` goes
  further: a scripted :class:`random.Random` replays EVERY possible sequence
  of pivot draws on small inputs, so "always sorted" is checked over the
  whole probability space rather than over a sample of it.
* **The cost matches the CLRS analysis exactly.** The expected number of
  comparisons on n distinct keys is ``E(n) = 2 (n + 1) H_n - 4 n``
  (CLRS 4e section 7.4.2). The anchors E(2) = 1, E(3) = 8/3, E(4) = 29/6,
  E(5) = 37/5, E(6) = 103/10 and E(10) = 30791/1260 were worked on paper
  from the recurrence ``E(n) = (n - 1) + (2 / n) * sum(E(i) for i < n)``.
  That recurrence, and the pair-sum definition
  ``sum over i < j of 2 / (j - i + 1)``, are both re-implemented in this file
  with exact fractions, so :func:`expected_comparisons` is checked against
  two derivations that share no code with it. The sort itself is then held
  to those numbers EXACTLY, not statistically: every enumerated draw
  sequence is weighted by its probability, and the weighted mean must be
  29/6 for every one of the 24 orderings of four keys.
* **The deterministic baseline really is the adversarial case.** CLRS
  QUICKSORT with the last element as pivot makes exactly ``n (n - 1) / 2``
  comparisons on sorted input (45, 4,950 and 124,750 for n = 10, 100, 500),
  and still sorts 3,000 sorted items without a RecursionError, because the
  module recurses on the smaller side and loops on the larger.

The sampled claims (the mean over 3,000 seeds at n = 3, and over 400 seeds at
n = 1,000) use fixed seeds AND tolerances of more than 5 standard errors, so
they cannot flake. No assertion is about time, and the file runs in a few
seconds next to seven thousand other tests.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import copy
import math
import random
import statistics
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pytest

from src.randomized.randomized_quicksort import (
    comparison_std_asymptotic,
    deterministic_quicksort,
    expected_comparisons,
    randomized_quicksort,
)
from src.utils.random_utils import make_rng, random_distinct_list
from tests.conftest import has_same_elements, is_sorted

# ----------------------------------------------------------------------
# Hand-checked anchors
# ----------------------------------------------------------------------
#: E(n) worked by hand from E(n) = (n - 1) + (2 / n) * (E(0) + ... + E(n - 1)).
#: E(3) = 2 + (2/3)(1) = 8/3;  E(4) = 3 + (1/2)(1 + 8/3) = 29/6;
#: E(5) = 4 + (2/5)(1 + 8/3 + 29/6) = 37/5;
#: E(6) = 5 + (1/3)(1 + 8/3 + 29/6 + 37/5) = 103/10;
#: E(10) = 2 * 11 * H_10 - 40 with H_10 = 7381/2520, which is 30791/1260.
EXPECTED_ANCHORS: Dict[int, Fraction] = {
    0: Fraction(0),
    1: Fraction(0),
    2: Fraction(1),
    3: Fraction(8, 3),
    4: Fraction(29, 6),
    5: Fraction(37, 5),
    6: Fraction(103, 10),
    10: Fraction(30791, 1260),
}

#: n(n - 1)/2: the cost of "last" on sorted input, one element peeled per pass.
WORST_CASE_ANCHORS: List[Tuple[int, int]] = [(10, 45), (100, 4950), (500, 124750)]

#: The number of distinct pivot-draw sequences on n distinct keys. A draw of
#: rank r leaves independent ranges of sizes r - 1 and n - r, so the count is
#: P(n) = sum over r of P(r - 1) P(n - r): the Catalan numbers.
CATALAN: Dict[int, int] = {0: 1, 1: 1, 2: 2, 3: 5, 4: 14, 5: 42, 6: 132}

#: E(1000) = 2 * 1001 * H_1000 - 4000, with H_1000 = 7.4854708606...
E_1000 = 10985.91

#: sqrt(7 - 2 pi^2 / 3) * 1000, the leading-order spread at n = 1,000.
STD_1000 = 648.3

#: Every key of the conftest ``sample_arrays`` fixture, listed by hand so a
#: case added there without a test here is caught by test_battery_is_complete.
SAMPLE_CASES: List[str] = [
    "empty",
    "single",
    "two_sorted",
    "two_reversed",
    "already_sorted",
    "reverse_sorted",
    "random_small",
    "with_duplicates",
    "all_identical",
    "negatives",
    "mixed_signs",
    "floats",
    "strings",
    "already_sorted_large",
    "reverse_sorted_large",
    "large",
]

PIVOT_RULES: List[str] = ["last", "median_of_three"]
BATTERY_SEEDS: List[int] = [0, 7, 2026]

NON_LISTS = [
    pytest.param((3, 1, 2), id="tuple"),
    pytest.param("cba", id="str"),
    pytest.param(None, id="None"),
    pytest.param(range(3), id="range"),
    pytest.param({3: "c", 1: "a"}, id="dict"),
    pytest.param({3, 1, 2}, id="set"),
    pytest.param(42, id="int"),
]


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def pairs(n: int) -> int:
    """Return ``n (n - 1) / 2``, the number of distinct pairs among n keys."""
    return n * (n - 1) // 2


def recurrence_expectations(limit: int) -> List[Fraction]:
    """Return exact ``[E(0), ..., E(limit)]`` from the CLRS recurrence.

    ``E(n) = (n - 1) + (2 / n) * sum(E(i) for i < n)``: one partition costs
    ``n - 1`` comparisons, the pivot rank is uniform on ``1 .. n``, and each
    size ``i`` appears as a side with probability ``2 / n``. No harmonic
    numbers and no closed form, so it shares nothing with the module.
    """
    values: List[Fraction] = [Fraction(0)]
    running = Fraction(0)
    for n in range(1, limit + 1):
        value = (n - 1) + Fraction(2, n) * running
        values.append(value)
        running += value
    return values


def recurrence_expectations_float(limit: int) -> List[float]:
    """The same recurrence in floats, cheap enough to run past n = 2,000."""
    values: List[float] = [0.0]
    running = 0.0
    for n in range(1, limit + 1):
        value = (n - 1) + 2.0 * running / n
        values.append(value)
        running += value
    return values


def pair_sum_expectation(n: int) -> Fraction:
    """E(n) from its definition: sum over pairs of P(pair is ever compared).

    Ranks ``i < j`` are compared exactly when the first pivot drawn from
    ``{z_i, ..., z_j}`` is one of the two ends, probability
    ``2 / (j - i + 1)``. There are ``n - gap`` pairs with ``j - i = gap``.
    """
    return sum((Fraction(2 * (n - gap), gap + 1) for gap in range(1, n)), Fraction(0))


def fewest_comparisons(limit: int) -> List[int]:
    """Return the minimum possible count for each n, by brute-force DP.

    A partition of ``n`` keys costs ``n - 1`` and leaves sides of sizes
    ``k`` and ``n - 1 - k`` for some ``k``; the minimum is taken over every
    ``k``. This is the best any sequence of pivot draws can do.
    """
    best = [0] * (limit + 1)
    for n in range(2, limit + 1):
        best[n] = (n - 1) + min(best[k] + best[n - 1 - k] for k in range(n))
    return best


@lru_cache(maxsize=None)
def balanced_comparisons(n: int) -> int:
    """Count for pivots that always split ``n - 1`` keys as evenly as possible."""
    if n < 2:
        return 0
    left = (n - 1) // 2
    return (n - 1) + balanced_comparisons(left) + balanced_comparisons(n - 1 - left)


class ScriptedRandom(random.Random):
    """A :class:`random.Random` whose draws are dictated by a script.

    ``randint(lo, hi)`` delegates to ``randrange(lo, hi + 1)``, so overriding
    :meth:`randrange` controls every pivot draw. Draw number ``i`` returns
    ``lo + script[i]``, or ``lo`` once the script runs out, and every
    requested range is recorded in :attr:`calls` so the enumerator knows how
    many choices each draw had.
    """

    def __init__(self, script: Sequence[int] = ()) -> None:
        super().__init__(0)
        self.script: List[int] = list(script)
        self.calls: List[Tuple[int, int]] = []

    def randrange(  # type: ignore[override]
        self, start: int, stop: Optional[int] = None, step: int = 1
    ) -> int:
        assert stop is not None and step == 1, "pivot draws should be randint(lo, hi)"
        index = len(self.calls)
        self.calls.append((start, stop))
        offset = self.script[index] if index < len(self.script) else 0
        assert 0 <= offset < stop - start
        return start + offset


class HighestIndex(random.Random):
    """A :class:`random.Random` that always draws the top of the range."""

    def randrange(  # type: ignore[override]
        self, start: int, stop: Optional[int] = None, step: int = 1
    ) -> int:
        assert stop is not None
        return stop - 1


class DrawPath(NamedTuple):
    """One complete sequence of pivot draws and what the sort did under it."""

    probability: Fraction
    output: List[Any]
    comparisons: int
    calls: List[Tuple[int, int]]


def enumerate_draws(arr: List[Any]) -> List[DrawPath]:
    """Run the randomized sort under EVERY possible sequence of pivot draws.

    An odometer over draw sequences: run with a script, record the size of
    every range drawn from, then advance the last draw that still has an
    untried value and replay. Replaying a prefix reproduces the same ranges,
    because the sort is deterministic given its draws. Each path's
    probability is the product of ``1 / size`` over its draws.

    Args:
        arr: The input list. Not modified.

    Returns:
        Every path, each exactly once.
    """
    paths: List[DrawPath] = []
    script: List[int] = []
    while True:
        rng = ScriptedRandom(script)
        output, comparisons = randomized_quicksort(arr, rng=rng, count_comparisons=True)
        sizes = [stop - start for start, stop in rng.calls]
        assert len(sizes) >= len(script), "a scripted draw was never requested"
        offsets = script + [0] * (len(sizes) - len(script))
        probability = Fraction(1)
        for size in sizes:
            probability /= size
        paths.append(DrawPath(probability, output, comparisons, list(rng.calls)))
        position = len(sizes) - 1
        while position >= 0 and offsets[position] == sizes[position] - 1:
            position -= 1
        if position < 0:
            return paths
        script = offsets[:position] + [offsets[position] + 1]


def exact_mean(paths: Sequence[DrawPath]) -> Fraction:
    """The exact expected comparison count over a full enumeration."""
    return sum((p.probability * p.comparisons for p in paths), Fraction(0))


def assert_sorted_copy(
    original: List[Any], snapshot: List[Any], result: List[Any]
) -> None:
    """Check the full sorting contract against the built-in ``sorted``.

    ``result`` is ascending, a permutation of the input, a new list, and the
    caller's ``original`` still equals the ``snapshot`` taken before the call.
    """
    assert result == sorted(snapshot)
    assert is_sorted(result)
    assert has_same_elements(result, snapshot)
    assert result is not original
    assert original == snapshot, "the caller's list was modified"


# ----------------------------------------------------------------------
# Correctness on the Week 1 battery
# ----------------------------------------------------------------------
class TestSortsCorrectly:
    """Both sorts against every conftest case, plus the input contract."""

    def test_battery_is_complete(self, sample_arrays: Dict[str, List[Any]]) -> None:
        assert sorted(sample_arrays) == sorted(SAMPLE_CASES)

    @pytest.mark.parametrize("seed", BATTERY_SEEDS)
    @pytest.mark.parametrize("case", SAMPLE_CASES)
    def test_randomized_sorts_every_case(
        self, sample_arrays: Dict[str, List[Any]], case: str, seed: int
    ) -> None:
        arr = sample_arrays[case]
        snapshot = copy.deepcopy(arr)
        result, count = randomized_quicksort(arr, seed=seed, count_comparisons=True)
        assert_sorted_copy(arr, snapshot, result)
        # Each pair is compared at most once; a list of 2+ needs at least
        # the first partition's n - 1.
        n = len(arr)
        assert isinstance(count, int)
        assert (n - 1 if n > 1 else 0) <= count <= pairs(n)
        assert randomized_quicksort(arr, seed=seed) == result

    @pytest.mark.parametrize("rule", PIVOT_RULES)
    @pytest.mark.parametrize("case", SAMPLE_CASES)
    def test_deterministic_sorts_every_case(
        self, sample_arrays: Dict[str, List[Any]], case: str, rule: str
    ) -> None:
        arr = sample_arrays[case]
        snapshot = copy.deepcopy(arr)
        result, count = deterministic_quicksort(arr, pivot=rule, count_comparisons=True)
        assert_sorted_copy(arr, snapshot, result)
        n = len(arr)
        assert (n - 1 if n > 1 else 0) <= count <= pairs(n)
        assert deterministic_quicksort(arr, pivot=rule) == result

    @pytest.mark.parametrize(
        "arr, expected",
        [
            pytest.param([], [], id="empty"),
            pytest.param([42], [42], id="single"),
            pytest.param([3, 3, 1, 3, 1], [1, 1, 3, 3, 3], id="duplicates"),
            pytest.param([-3, -10, 0, -1], [-10, -3, -1, 0], id="negatives"),
            pytest.param(
                ["pear", "Apple", "fig"], ["Apple", "fig", "pear"], id="strings"
            ),
            pytest.param([2, 1.5, -0.25, 1], [-0.25, 1, 1.5, 2], id="int_and_float"),
            pytest.param(
                [(2, "b"), (1, "z"), (2, "a")],
                [(1, "z"), (2, "a"), (2, "b")],
                id="tuples",
            ),
        ],
    )
    def test_hand_written_cases_for_all_three_sorts(
        self, arr: List[Any], expected: List[Any]
    ) -> None:
        snapshot = list(arr)
        assert randomized_quicksort(arr, seed=11) == expected
        for rule in PIVOT_RULES:
            assert deterministic_quicksort(arr, pivot=rule) == expected
        assert arr == snapshot

    def test_heavy_duplicates(self) -> None:
        rng = make_rng(5300)
        arr = [rng.randint(0, 9) for _ in range(1500)]
        snapshot = list(arr)
        assert_sorted_copy(arr, snapshot, randomized_quicksort(arr, seed=3))
        for rule in PIVOT_RULES:
            assert_sorted_copy(arr, snapshot, deterministic_quicksort(arr, pivot=rule))

    def test_las_vegas_output_is_identical_for_every_seed(self) -> None:
        arr = [5, 2, 9, 1, 5, 6, -4, 0, 9, 2]
        expected = [-4, 0, 1, 2, 2, 5, 5, 6, 9, 9]
        for seed in range(200):
            assert randomized_quicksort(arr, seed=seed) == expected

    @pytest.mark.parametrize("arr", [[], [8]], ids=["empty", "single"])
    def test_trivial_inputs_cost_nothing_and_come_back_as_copies(
        self, arr: List[int]
    ) -> None:
        for result, count in (
            randomized_quicksort(arr, seed=0, count_comparisons=True),
            deterministic_quicksort(arr, pivot="last", count_comparisons=True),
            deterministic_quicksort(
                arr, pivot="median_of_three", count_comparisons=True
            ),
        ):
            assert result == arr and result is not arr
            assert count == 0

    def test_two_elements_cost_exactly_one_comparison(self) -> None:
        for arr in ([1, 2], [2, 1]):
            for seed in range(10):
                assert randomized_quicksort(arr, seed=seed, count_comparisons=True) == (
                    [1, 2],
                    1,
                )
            for rule in PIVOT_RULES:
                assert deterministic_quicksort(
                    arr, pivot=rule, count_comparisons=True
                ) == ([1, 2], 1)

    def test_count_flag_controls_the_return_shape(self) -> None:
        plain = randomized_quicksort([3, 1, 2], seed=1)
        assert plain == [1, 2, 3]
        counted = randomized_quicksort([3, 1, 2], seed=1, count_comparisons=True)
        assert isinstance(counted, tuple) and len(counted) == 2
        assert counted[0] == [1, 2, 3]
        assert deterministic_quicksort([3, 1, 2]) == [1, 2, 3]
        assert isinstance(
            deterministic_quicksort([3, 1, 2], count_comparisons=True), tuple
        )

    @pytest.mark.parametrize("bad", NON_LISTS)
    def test_randomized_rejects_a_non_list(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="expects a list"):
            randomized_quicksort(bad, seed=0)

    @pytest.mark.parametrize("rule", PIVOT_RULES)
    @pytest.mark.parametrize("bad", NON_LISTS)
    def test_deterministic_rejects_a_non_list(self, bad: Any, rule: str) -> None:
        with pytest.raises(TypeError, match="expects a list"):
            deterministic_quicksort(bad, pivot=rule)

    def test_incomparable_elements_raise_and_leave_the_input_alone(self) -> None:
        arr: List[Any] = [3, "a", 1, None]
        with pytest.raises(TypeError):
            randomized_quicksort(arr, seed=0)
        for rule in PIVOT_RULES:
            with pytest.raises(TypeError):
                deterministic_quicksort(arr, pivot=rule)
        assert arr == [3, "a", 1, None]


# ----------------------------------------------------------------------
# Seeds and the rng argument
# ----------------------------------------------------------------------
class TestSeedsAndRng:
    """A seed fixes the run; an explicit rng overrides the seed."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 2026, 2**40])
    def test_same_seed_gives_identical_output_and_count(self, seed: int) -> None:
        arr = random_distinct_list(300, 17)
        first = randomized_quicksort(arr, seed=seed, count_comparisons=True)
        second = randomized_quicksort(arr, seed=seed, count_comparisons=True)
        assert first == second
        assert first[0] == list(range(300))

    def test_twenty_seeds_do_not_all_give_the_same_count(self) -> None:
        arr = random_distinct_list(50, 99)
        counts = {
            randomized_quicksort(arr, seed=seed, count_comparisons=True)[1]
            for seed in range(20)
        }
        assert len(counts) > 1

    def test_rng_takes_precedence_over_seed(self) -> None:
        arr = random_distinct_list(200, 3)
        via_rng = randomized_quicksort(
            arr, seed=1, rng=make_rng(7), count_comparisons=True
        )
        via_seed = randomized_quicksort(arr, seed=7, count_comparisons=True)
        assert via_rng == via_seed

    def test_scripted_rng_overrides_seed_completely(self) -> None:
        # HighestIndex always picks hi, which on sorted input is the range's
        # maximum: the quadratic path, n(n - 1)/2 = 190 for n = 20. The seed,
        # which on its own gives a far cheaper run, must play no part.
        arr = list(range(20))
        result, count = randomized_quicksort(
            arr, seed=123, rng=HighestIndex(0), count_comparisons=True
        )
        assert result == arr and count == 190
        assert randomized_quicksort(arr, seed=123, count_comparisons=True)[1] < 190

    def test_any_random_instance_works_as_rng(self) -> None:
        arr = random_distinct_list(100, 8)
        assert randomized_quicksort(
            arr, rng=random.Random(21), count_comparisons=True
        ) == randomized_quicksort(arr, seed=21, count_comparisons=True)

    def test_one_rng_drives_many_calls_reproducibly(self) -> None:
        arr = random_distinct_list(100, 4)
        shared_a, shared_b = make_rng(5), make_rng(5)
        run_a = [
            randomized_quicksort(arr, rng=shared_a, count_comparisons=True)[1]
            for _ in range(5)
        ]
        run_b = [
            randomized_quicksort(arr, rng=shared_b, count_comparisons=True)[1]
            for _ in range(5)
        ]
        assert run_a == run_b
        # The draws really come out of the caller's generator.
        assert shared_a.getstate() != make_rng(5).getstate()

    def test_default_seed_is_unpredictable_but_still_sorts(self) -> None:
        arr = random_distinct_list(200, 12)
        for _ in range(5):
            assert randomized_quicksort(arr) == list(range(200))

    @pytest.mark.parametrize(
        "bad_seed", ["7", True, 1.5, [7]], ids=["str", "bool", "float", "list"]
    )
    def test_non_int_seed_raises_type_error(self, bad_seed: Any) -> None:
        with pytest.raises(TypeError):
            randomized_quicksort([3, 1, 2], seed=bad_seed)

    def test_count_depends_only_on_relative_order(self) -> None:
        # A comparison sort cannot tell order-isomorphic inputs apart, so the
        # same draws must give the same count and the image of the output.
        base = random_distinct_list(150, 31)
        shifted = [3 * x - 1000 for x in base]
        words = [f"k{x:04d}" for x in base]
        for seed in range(10):
            out_b, c_b = randomized_quicksort(base, seed=seed, count_comparisons=True)
            out_s, c_s = randomized_quicksort(
                shifted, seed=seed, count_comparisons=True
            )
            out_w, c_w = randomized_quicksort(words, seed=seed, count_comparisons=True)
            assert c_b == c_s == c_w
            assert out_s == [3 * x - 1000 for x in out_b]
            assert out_w == [f"k{x:04d}" for x in out_b]


# ----------------------------------------------------------------------
# The deterministic baseline
# ----------------------------------------------------------------------
class TestDeterministicBaseline:
    """Exact counts for the fixed pivot rules, traced or derived by hand."""

    @pytest.mark.parametrize(
        "n, expected", WORST_CASE_ANCHORS, ids=[f"n={n}" for n, _ in WORST_CASE_ANCHORS]
    )
    def test_last_rule_on_sorted_input_is_exactly_n_choose_2(
        self, n: int, expected: int
    ) -> None:
        assert expected == n * (n - 1) // 2
        result, count = deterministic_quicksort(
            list(range(n)), pivot="last", count_comparisons=True
        )
        assert result == list(range(n))
        assert count == expected

    @pytest.mark.parametrize(
        "n, expected", WORST_CASE_ANCHORS, ids=[f"n={n}" for n, _ in WORST_CASE_ANCHORS]
    )
    def test_last_rule_on_reversed_input_is_also_n_choose_2(
        self, n: int, expected: int
    ) -> None:
        # Traced on [4, 3, 2, 1, 0]: pivot 0 (the minimum) gives
        # [0, 3, 2, 1, 4], whose next range ends in its maximum 4, which
        # leaves [3, 2, 1] reversed again. Every pivot is an extreme, so every
        # pass peels one element: 4 + 3 + 2 + 1 = 10.
        result, count = deterministic_quicksort(
            list(range(n, 0, -1)), pivot="last", count_comparisons=True
        )
        assert result == list(range(1, n + 1))
        assert count == expected

    def test_quadratic_case_at_3000_does_not_hit_the_recursion_limit(self) -> None:
        # A recursion that followed the larger side would be 3,000 frames
        # deep, past Python's default limit of 1,000.
        result, count = deterministic_quicksort(
            list(range(3000)), pivot="last", count_comparisons=True
        )
        assert result == list(range(3000))
        assert count == 3000 * 2999 // 2 == 4_498_500

    def test_hand_traced_last_rule_on_three(self) -> None:
        # [3, 1, 2], pivot 2: compare 3 and 1 with it (2 comparisons), giving
        # [1, 2, 3] with both sides of size 1. Nothing more to do.
        assert deterministic_quicksort(
            [3, 1, 2], pivot="last", count_comparisons=True
        ) == ([1, 2, 3], 2)

    def test_hand_traced_median_of_three_on_sorted_ten(self) -> None:
        # range(10): median of 0, 4, 9 is 4 -> 9 comparisons, sides [0..3]
        # and [5..9]. [0..3]: median of 0, 1, 3 is 1 -> 3, then [2, 3] -> 1.
        # [5..9]: median of 5, 7, 9 is 7 -> 4, then [5, 6] -> 1 and
        # [8, 9] -> 1. Total 9 + 3 + 1 + 4 + 1 + 1 = 19.
        assert deterministic_quicksort(
            list(range(10)), pivot="median_of_three", count_comparisons=True
        ) == (list(range(10)), 19)

    @pytest.mark.parametrize("n", [10, 100, 1000, 3000])
    def test_median_of_three_makes_sorted_input_its_best_case(self, n: int) -> None:
        # On sorted input the median sits at the middle index, and Lomuto
        # leaves both sides sorted again, so every split is as even as
        # possible: the balanced recurrence, far below n(n - 1)/2.
        result, count = deterministic_quicksort(
            list(range(n)), pivot="median_of_three", count_comparisons=True
        )
        assert result == list(range(n))
        assert count == balanced_comparisons(n)
        assert 2 * count < pairs(n)

    def test_balanced_recurrence_is_the_true_minimum(self) -> None:
        best = fewest_comparisons(64)
        assert best[:8] == [0, 0, 1, 2, 4, 6, 8, 10]
        assert all(balanced_comparisons(n) == best[n] for n in range(65))

    @pytest.mark.parametrize("rule", PIVOT_RULES)
    def test_all_equal_input_is_quadratic_under_every_rule(self, rule: str) -> None:
        # Lomuto's <= sends every key equal to the pivot left, so the pivot
        # always lands at hi and each pass peels one element: 50 * 49 / 2.
        assert deterministic_quicksort(
            [7] * 50, pivot=rule, count_comparisons=True
        ) == ([7] * 50, 1225)

    def test_all_equal_input_is_quadratic_for_every_seed_too(self) -> None:
        for seed in range(10):
            assert randomized_quicksort(
                [7] * 50, seed=seed, count_comparisons=True
            ) == ([7] * 50, 1225)

    @pytest.mark.parametrize("rule", PIVOT_RULES)
    def test_same_input_always_takes_the_same_path(self, rule: str) -> None:
        arr = random_distinct_list(400, 77)
        runs = {
            deterministic_quicksort(arr, pivot=rule, count_comparisons=True)[1]
            for _ in range(3)
        }
        assert len(runs) == 1

    def test_default_rule_is_last(self) -> None:
        assert (
            deterministic_quicksort(list(range(100)), count_comparisons=True)[1] == 4950
        )

    @pytest.mark.parametrize(
        "bad_rule",
        ["random", "first", "LAST", "median", "", None, 3, ["last"]],
        ids=["random", "first", "LAST", "median", "empty", "None", "int", "list"],
    )
    def test_invalid_pivot_rule_raises_value_error(self, bad_rule: Any) -> None:
        with pytest.raises(ValueError, match="pivot must be"):
            deterministic_quicksort([3, 1, 2], pivot=bad_rule)
        with pytest.raises(ValueError, match="pivot must be"):
            deterministic_quicksort([], pivot=bad_rule)


# ----------------------------------------------------------------------
# expected_comparisons against the hand anchors and two derivations
# ----------------------------------------------------------------------
class TestExpectedComparisons:
    """The closed form, checked by hand and by two independent derivations."""

    @pytest.mark.parametrize(
        "n", sorted(EXPECTED_ANCHORS), ids=[f"E({n})" for n in sorted(EXPECTED_ANCHORS)]
    )
    def test_hand_worked_anchors_exactly(self, n: int) -> None:
        value = expected_comparisons(n)
        exact = EXPECTED_ANCHORS[n]
        assert isinstance(value, float)
        assert math.isclose(value, float(exact), rel_tol=0.0, abs_tol=1e-9)
        # Every anchor's denominator is at most 1,260, and any other fraction
        # that small is at least 1e-7 away, so this recovers it exactly.
        assert Fraction(value).limit_denominator(10_000) == exact

    def test_e10_decimal_value(self) -> None:
        assert math.isclose(expected_comparisons(10), 24.4373, abs_tol=1e-4)

    def test_e1000(self) -> None:
        assert abs(expected_comparisons(1000) - E_1000) < 0.01

    def test_independent_recurrence_up_to_30(self) -> None:
        exact = recurrence_expectations(30)
        for n, value in enumerate(exact):
            assert math.isclose(
                expected_comparisons(n), float(value), rel_tol=0.0, abs_tol=1e-9
            ), n

    def test_recurrence_reproduces_the_hand_anchors(self) -> None:
        exact = recurrence_expectations(10)
        for n, anchor in EXPECTED_ANCHORS.items():
            assert exact[n] == anchor

    def test_pair_sum_definition_agrees_with_the_recurrence(self) -> None:
        exact = recurrence_expectations(30)
        for n in range(31):
            assert pair_sum_expectation(n) == exact[n]
            assert math.isclose(
                expected_comparisons(n), float(pair_sum_expectation(n)), abs_tol=1e-9
            )

    @pytest.mark.parametrize("n", [1999, 2000, 2001, 2500, 5000])
    def test_both_sides_of_the_exact_limit_match_the_recurrence(self, n: int) -> None:
        # Above n = 2,000 the module switches to an asymptotic H_n; the float
        # recurrence does not, so it checks that branch and the seam.
        assert math.isclose(
            expected_comparisons(n),
            recurrence_expectations_float(n)[n],
            rel_tol=0.0,
            abs_tol=1e-3,
        )

    def test_strictly_increasing_across_the_seam(self) -> None:
        # E(n) - E(n - 1) = 2 H_n - 2 > 0 for n >= 2.
        for window in (range(1, 300), range(1990, 2011)):
            values = [expected_comparisons(n) for n in window]
            assert all(b > a for a, b in zip(values, values[1:]))

    @pytest.mark.parametrize("n", [2, 3, 10, 100, 2000, 2001, 10**6])
    def test_below_2n_ln_n(self, n: int) -> None:
        # H_n <= ln n + 1 gives E(n) <= 2 n ln n + 2 ln n + 2 - 2 n, which is
        # below 2 n ln n whenever ln n + 1 < n.
        assert expected_comparisons(n) < 2 * n * math.log(n)

    @pytest.mark.parametrize(
        "bad", [True, 2.0, "3", None], ids=["bool", "float", "str", "None"]
    )
    def test_non_int_n_raises_type_error(self, bad: Any) -> None:
        with pytest.raises(TypeError):
            expected_comparisons(bad)

    def test_negative_n_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            expected_comparisons(-1)


class TestComparisonStd:
    """The leading-order spread, sqrt(7 - 2 pi^2 / 3) n."""

    def test_n_1000(self) -> None:
        assert abs(comparison_std_asymptotic(1000) - STD_1000) < 0.05

    def test_constant_and_linearity(self) -> None:
        constant = math.sqrt(7 - 2 * math.pi**2 / 3)
        assert math.isclose(constant, 0.6483, abs_tol=5e-5)
        for n in (0, 1, 10, 1000, 10**6):
            assert math.isclose(
                comparison_std_asymptotic(n), constant * n, rel_tol=1e-12, abs_tol=1e-12
            )
        assert comparison_std_asymptotic(0) == 0.0

    def test_bad_n(self) -> None:
        with pytest.raises(ValueError):
            comparison_std_asymptotic(-1)
        with pytest.raises(TypeError):
            comparison_std_asymptotic(True)


# ----------------------------------------------------------------------
# Exact: every draw sequence on small inputs
# ----------------------------------------------------------------------
class TestExhaustiveDraws:
    """The whole probability space, enumerated; no sampling, no tolerance."""

    @pytest.mark.parametrize("n", sorted(CATALAN))
    def test_number_of_draw_sequences_is_catalan(self, n: int) -> None:
        paths = enumerate_draws(list(range(n)))
        assert len(paths) == CATALAN[n]
        assert sum(p.probability for p in paths) == 1

    def test_n3_distribution(self) -> None:
        # Drawing the middle key first costs 2 + 0; drawing an end costs
        # 2 + 1. P(middle) = 1/3, so the mean is (2 + 3 + 3) / 3 = 8/3.
        paths = enumerate_draws([30, 10, 20])
        dist: Dict[int, Fraction] = {}
        for p in paths:
            dist[p.comparisons] = dist.get(p.comparisons, Fraction(0)) + p.probability
        assert dist == {2: Fraction(1, 3), 3: Fraction(2, 3)}
        assert exact_mean(paths) == Fraction(8, 3)

    def test_every_ordering_of_four_keys_has_mean_exactly_29_over_6(self) -> None:
        for perm in permutations([10, 20, 30, 40]):
            paths = enumerate_draws(list(perm))
            assert len(paths) == 14
            assert exact_mean(paths) == Fraction(29, 6), perm
            assert all(p.output == [10, 20, 30, 40] for p in paths)

    def test_every_ordering_of_five_keys_has_mean_exactly_37_over_5(self) -> None:
        for perm in permutations(range(5)):
            paths = enumerate_draws(list(perm))
            assert exact_mean(paths) == Fraction(37, 5), perm
            assert all(p.output == [0, 1, 2, 3, 4] for p in paths)

    @pytest.mark.parametrize(
        "arr",
        [list(range(6)), list(range(6, 0, -1)), [4, 0, 5, 2, 1, 3]],
        ids=["sorted", "reversed", "shuffled"],
    )
    def test_six_keys_mean_min_and_max(self, arr: List[int]) -> None:
        paths = enumerate_draws(arr)
        assert len(paths) == 132
        assert exact_mean(paths) == Fraction(103, 10)
        counts = [p.comparisons for p in paths]
        # Best: split 5 as 2 + 3, so 5 + 1 + 2 = 8. Worst: 5+4+3+2+1 = 15.
        assert min(counts) == 8 == fewest_comparisons(6)[6]
        assert max(counts) == 15

    def test_first_draw_spans_the_whole_list(self) -> None:
        for p in enumerate_draws([3, 1, 4, 0, 2]):
            assert p.calls[0] == (0, 5)
            assert all(0 <= lo < hi <= 5 for lo, hi in p.calls)

    @pytest.mark.parametrize(
        "multiset",
        [[1, 2, 2, 3, 3], [0, 0, 1, 1], [9, 9, 9, 2]],
        ids=["1-2-2-3-3", "0-0-1-1", "9-9-9-2"],
    )
    def test_duplicates_sort_under_every_draw_and_every_ordering(
        self, multiset: List[int]
    ) -> None:
        expected = sorted(multiset)
        for perm in set(permutations(multiset)):
            for p in enumerate_draws(list(perm)):
                assert p.output == expected
                assert p.comparisons <= pairs(len(multiset))

    def test_all_equal_costs_n_choose_2_under_every_draw(self) -> None:
        paths = enumerate_draws([5, 5, 5, 5, 5])
        assert all(p.comparisons == 10 and p.output == [5] * 5 for p in paths)


# ----------------------------------------------------------------------
# Sampled: the mean and spread over fixed seeds
# ----------------------------------------------------------------------
class TestSampledCost:
    """Means over many fixed seeds, with tolerances of 5+ standard errors."""

    def test_n3_mean_over_3000_seeds(self) -> None:
        # Counts are 2 or 3 with p = 1/3, 2/3: sd = sqrt(2)/3 = 0.471, so the
        # standard error over 3,000 runs is 0.0086 and 0.1 is over 11 of them.
        counts = [
            randomized_quicksort([30, 10, 20], seed=seed, count_comparisons=True)[1]
            for seed in range(3000)
        ]
        assert set(counts) == {2, 3}
        assert abs(statistics.fmean(counts) - 8 / 3) < 0.1

    def test_n1000_mean_and_spread_over_400_seeds(self) -> None:
        # Standard error of the mean is about 648 / 20 = 32; 3% of E(1000)
        # is 330, about 10 of them. The sample sd has a standard error near
        # 30, so a band of +/- 30% around 648.3 is more than 6 of them.
        arr = random_distinct_list(1000, 1000)
        counts = [
            randomized_quicksort(arr, seed=seed, count_comparisons=True)[1]
            for seed in range(400)
        ]
        assert abs(statistics.fmean(counts) - 10985.9) <= 0.03 * 10985.9
        assert 0.7 * STD_1000 <= statistics.stdev(counts) <= 1.3 * STD_1000

    def test_sorted_input_is_not_special_to_the_randomized_sort(self) -> None:
        # E(500) is about 4,806 with sd about 324, so the standard error over
        # 20 runs is about 72 and a 10% band is over 6 of them. The "last"
        # rule pays 124,750 on the same input.
        e_500 = recurrence_expectations_float(500)[500]
        counts = [
            randomized_quicksort(list(range(500)), seed=seed, count_comparisons=True)[1]
            for seed in range(20)
        ]
        assert all(c < pairs(500) // 10 for c in counts)
        assert abs(statistics.fmean(counts) - e_500) < 0.1 * e_500

    @pytest.mark.parametrize("n", range(2, 13))
    def test_every_run_lies_between_the_best_and_worst_case(self, n: int) -> None:
        best = fewest_comparisons(n)[n]
        arr = random_distinct_list(n, n)
        for seed in range(50):
            count = randomized_quicksort(arr, seed=seed, count_comparisons=True)[1]
            assert best <= count <= pairs(n)
