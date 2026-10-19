"""Tests for Subset Sum backtracking, the greedy heuristics, the DP check and the generators.

A subset-sum solver is easy to get almost right. A prune with ``>=`` in
place of ``>`` discards exact answers, a missing ``pop`` returns a list
that does not sum to the target, and a solver that reuses a value returns a
"subset" that is not a sub-multiset of the input at all. Each of those
still returns plausible output. So every claim here rests on a witness that
shares no mechanism with the code under test:

* **The spec anchors (WEEK8_BUILD_SPEC.md section 10.2), checked by hand.**
  ``[3, 34, 4, 12, 5, 2]`` sorts descending to ``[34, 12, 5, 4, 3, 2]``.
  For target **9**, 34 and 12 each overshoot and then 5 + 4 = 9, so
  include-first search returns **[5, 4]**, the subset the spec's prototype
  returned. For target **30**, the total without 34 is 26 < 30 and 34 alone
  overshoots, so the answer is **None**. Target **0** is the empty subset.
* **Node counts traced by hand** on those anchors and on ``[2, 2, 2]`` with
  target 3, call by call, in :class:`TestHandTracedNodeCounts`.
* **Brute force over every subset** with :mod:`itertools`, which knows
  nothing about sorting, prunes or tables, for n <= 10.
* **The pseudo-polynomial DP** on a seeded battery of a few hundred
  instances with n <= 15 and values <= 50, as section 10.2 requires. The DP
  is itself checked against brute force first, so the two exact methods are
  never only checked against each other.
* **Parity.** In the worst-case family every value is even and the target
  is odd, so no subset can hit it: the correct answer is known without
  running anything.

Every random draw is seeded through ``make_rng`` / ``derive_seed``, so
nothing here can flake. No assertion is about time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Callable, List, Optional, Set

import pytest

from src.complexity.subset_sum_backtracking import (
    satisfiable_instance,
    subset_sum_backtracking,
    subset_sum_dp,
    subset_sum_greedy,
    subset_sum_random_restart,
    worst_case_instance,
)
from src.utils.random_utils import derive_seed, make_rng

#: The spec's anchor list (section 10.2).
ANCHOR: List[int] = [3, 34, 4, 12, 5, 2]

#: Master seed for every seeded battery in this file.
MASTER_SEED: int = 42

#: Seeds used wherever "several seeds" are wanted.
SEEDS: List[int] = [0, 1, 7, 42, 2026]

#: The sizes over which the worst-case node count must strictly grow.
WORST_CASE_SIZES: List[int] = [8, 10, 12, 14]


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def is_sub_multiset(subset: List[int], nums: List[int]) -> bool:
    """Return True if every value of ``subset`` is available in ``nums``.

    Repeats count: ``[5, 5]`` is a sub-multiset of ``[5, 5, 6]`` but not
    of ``[5, 6]``.
    """
    return not (Counter(subset) - Counter(nums))


def brute_force_sums(nums: List[int]) -> Set[int]:
    """Every sum reachable by some sub-multiset of ``nums``, by enumeration.

    Walks all ``2^n`` index subsets with :func:`itertools.combinations`. It
    shares nothing with the module: no sorting, no prunes, no table.
    """
    return {
        sum(nums[i] for i in chosen)
        for size in range(len(nums) + 1)
        for chosen in combinations(range(len(nums)), size)
    }


def random_case(seed: int, max_n: int, max_value: int):
    """A seeded ``(nums, target)`` pair, about half of them solvable.

    ``nums`` has 0 .. ``max_n`` values in ``0 .. max_value``, so zeros and
    repeats both occur. Half the targets are the sum of a random subset
    (always solvable); the other half are uniform in ``0 .. sum + 10``
    (often not).
    """
    rng = make_rng(seed)
    n = rng.randint(0, max_n)
    nums = [rng.randint(0, max_value) for _ in range(n)]
    if rng.random() < 0.5:
        target = sum(v for v in nums if rng.random() < 0.5)
    else:
        target = rng.randint(0, sum(nums) + 10)
    return nums, target


def assert_valid_certificate(result: Optional[List[int]], nums: List[int], target: int):
    """A returned subset must sum to the target and come from ``nums``."""
    assert result is not None
    assert isinstance(result, list)
    assert sum(result) == target
    assert is_sub_multiset(result, nums)


# ----------------------------------------------------------------------
# Spec anchors (section 10.2)
# ----------------------------------------------------------------------
class TestSpecAnchors:
    """The hand-checked values from WEEK8_BUILD_SPEC.md section 10.2."""

    def test_target_9_returns_a_subset_summing_to_9(self):
        result = subset_sum_backtracking(ANCHOR, 9)
        assert_valid_certificate(result, ANCHOR, 9)

    def test_target_9_returns_5_then_4(self):
        # Descending order [34, 12, 5, 4, 3, 2], include first: 34 and 12
        # overshoot 9, then 5 + 4 = 9 is the first path that succeeds.
        assert subset_sum_backtracking(ANCHOR, 9) == [5, 4]

    def test_target_30_returns_none(self):
        # Without 34 the total is 3 + 4 + 12 + 5 + 2 = 26 < 30, and any
        # subset holding 34 is already above 30.
        assert subset_sum_backtracking(ANCHOR, 30) is None

    def test_target_0_returns_empty_list(self):
        assert subset_sum_backtracking(ANCHOR, 0) == []
        assert subset_sum_backtracking([], 0) == []

    def test_dp_agrees_on_the_anchors(self):
        assert subset_sum_dp(ANCHOR, 9) is True
        assert subset_sum_dp(ANCHOR, 30) is False
        assert subset_sum_dp(ANCHOR, 0) is True

    def test_brute_force_agrees_on_the_anchors(self):
        sums = brute_force_sums(ANCHOR)
        assert 9 in sums and 0 in sums and 30 not in sums

    def test_input_list_is_not_modified(self):
        data = list(ANCHOR)
        subset_sum_backtracking(data, 9)
        subset_sum_backtracking(data, 30)
        subset_sum_greedy(data, 9)
        subset_sum_random_restart(data, 9, tries=5, seed=1)
        subset_sum_dp(data, 9)
        assert data == ANCHOR


class TestHandTracedNodeCounts:
    """Node counts traced by hand, one per call including the root.

    Sorted descending, ``ANCHOR`` is ``[34, 12, 5, 4, 3, 2]`` with suffix
    sums ``[60, 26, 14, 9, 5, 2, 0]``. A node is written ``(i, running)``.
    """

    def test_anchor_target_9_visits_7_nodes(self):
        # 1 (0, 0)  2 (1, 34) overshoot  3 (1, 0)  4 (2, 12) overshoot
        # 5 (2, 0)  6 (3, 5)  7 (4, 9) hit.
        assert subset_sum_backtracking(ANCHOR, 9, count_nodes=True) == ([5, 4], 7)

    def test_anchor_target_30_visits_3_nodes(self):
        # 1 (0, 0)  2 (1, 34) overshoot  3 (1, 0): 0 + 26 < 30 shortfall.
        assert subset_sum_backtracking(ANCHOR, 30, count_nodes=True) == (None, 3)

    def test_target_0_is_answered_at_the_root(self):
        assert subset_sum_backtracking(ANCHOR, 0, count_nodes=True) == ([], 1)

    def test_empty_list_with_positive_target_is_one_node(self):
        # The root's suffix sum is 0 < 5, a shortfall at once.
        assert subset_sum_backtracking([], 5, count_nodes=True) == (None, 1)

    def test_equal_values_odd_target_visits_11_nodes(self):
        # [2, 2, 2], target 3, suffix sums [6, 4, 2, 0]. Every sum is even.
        # (0,0) -> (1,2) -> (2,4) over, (2,2) -> (3,4) over, (3,2) short;
        # (1,0) -> (2,2) -> (3,4) over, (3,2) short; (2,0) short.
        # 1 + [1 + 1 + 1 + 2] + [1 + 1 + 2 + 1] = 11.
        assert subset_sum_backtracking([2, 2, 2], 3, count_nodes=True) == (None, 11)

    def test_count_nodes_false_returns_the_bare_result(self):
        assert subset_sum_backtracking(ANCHOR, 30) is None
        assert subset_sum_backtracking(ANCHOR, 9, count_nodes=False) == [5, 4]


# ----------------------------------------------------------------------
# Small cases: repeats and zeros
# ----------------------------------------------------------------------
class TestRepeatsAndZeros:
    """Repeated values are separate items; zeros are harmless."""

    def test_repeated_values_can_both_be_used(self):
        assert subset_sum_backtracking([5, 5, 6], 10) == [5, 5]
        assert subset_sum_dp([5, 5, 6], 10) is True

    def test_a_single_copy_cannot_be_used_twice(self):
        assert subset_sum_backtracking([5, 6], 10) is None
        assert subset_sum_dp([5], 10) is False

    def test_zeros_do_not_break_the_search(self):
        assert subset_sum_backtracking([0, 0, 3], 3) == [3]
        assert subset_sum_backtracking([0, 0], 0) == []
        assert subset_sum_backtracking([0, 0], 1) is None
        assert subset_sum_dp([0, 0, 3], 3) is True

    def test_whole_list_is_a_valid_answer(self):
        assert subset_sum_backtracking([1, 2, 3], 6) == [3, 2, 1]


# ----------------------------------------------------------------------
# Agreement batteries
# ----------------------------------------------------------------------
class TestAgreement:
    """Backtracking found-iff-reachable, against the DP and brute force."""

    def test_dp_matches_brute_force(self):
        # Checks the DP before it is trusted as a witness below.
        for case in range(200):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "dp-vs-brute", case), max_n=10, max_value=30
            )
            assert subset_sum_dp(nums, target) == (target in brute_force_sums(nums)), (
                nums,
                target,
            )

    def test_backtracking_matches_brute_force(self):
        for case in range(200):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "bt-vs-brute", case), max_n=10, max_value=30
            )
            result = subset_sum_backtracking(nums, target)
            reachable = target in brute_force_sums(nums)
            assert (result is not None) == reachable, (nums, target)
            if result is not None:
                assert_valid_certificate(result, nums, target)

    def test_backtracking_matches_dp_on_a_seeded_battery(self):
        # Section 10.2: n <= 15, values <= 50, a few hundred cases.
        found = missed = 0
        for case in range(400):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "bt-vs-dp", case), max_n=15, max_value=50
            )
            result = subset_sum_backtracking(nums, target)
            assert (result is not None) == subset_sum_dp(nums, target), (nums, target)
            if result is not None:
                assert_valid_certificate(result, nums, target)
                found += 1
            else:
                missed += 1
        # The battery exercises both answers, not just one.
        assert found >= 100 and missed >= 50

    def test_returned_subset_is_in_descending_order(self):
        for case in range(100):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "order", case), max_n=12, max_value=50
            )
            result = subset_sum_backtracking(nums, target)
            if result is not None:
                assert result == sorted(result, reverse=True)

    def test_node_count_never_exceeds_the_full_tree(self):
        for case in range(100):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "bound", case), max_n=12, max_value=50
            )
            _, nodes = subset_sum_backtracking(nums, target, count_nodes=True)
            assert 1 <= nodes <= 2 ** (len(nums) + 1) - 1


# ----------------------------------------------------------------------
# The worst-case family
# ----------------------------------------------------------------------
class TestWorstCaseFamily:
    """Even values, odd target: no subset can hit it, and the search grows."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_generator_follows_the_contract(self, seed):
        nums, target = worst_case_instance(10, seed)
        rng = make_rng(seed)
        expected = [2 * rng.randint(1, 10**6) for _ in range(10)]
        assert nums == expected
        assert target == (sum(expected) // 2) | 1
        assert all(v % 2 == 0 and 2 <= v <= 2 * 10**6 for v in nums)
        assert target % 2 == 1

    def test_generator_is_reproducible(self):
        assert worst_case_instance(12, 42) == worst_case_instance(12, 42)
        assert worst_case_instance(12, 42) != worst_case_instance(12, 43)

    def test_empty_instance(self):
        assert worst_case_instance(0, 42) == ([], 1)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_backtracking_returns_none(self, seed):
        # Parity alone says no subset of even values sums to an odd target.
        for n in WORST_CASE_SIZES:
            nums, target = worst_case_instance(n, seed)
            assert subset_sum_backtracking(nums, target) is None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_node_counts_strictly_increase_with_n(self, seed):
        counts = []
        for n in WORST_CASE_SIZES:
            nums, target = worst_case_instance(n, seed)
            result, nodes = subset_sum_backtracking(nums, target, count_nodes=True)
            assert result is None
            assert nodes <= 2 ** (n + 1) - 1
            counts.append(nodes)
        assert all(a < b for a, b in zip(counts, counts[1:])), counts

    def test_heuristics_fall_short_by_parity(self):
        nums, target = worst_case_instance(14, 42)
        subset, achieved = subset_sum_greedy(nums, target)
        assert achieved < target and achieved % 2 == 0
        assert is_sub_multiset(subset, nums)
        subset, achieved = subset_sum_random_restart(nums, target, tries=20, seed=42)
        assert achieved < target and achieved % 2 == 0
        assert is_sub_multiset(subset, nums)


class TestSatisfiableInstance:
    """Random values whose target is the sum of a random nonempty subset."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_values_follow_the_contract(self, seed):
        nums, target = satisfiable_instance(10, seed)
        rng = make_rng(seed)
        assert nums == [rng.randint(1, 10**6) for _ in range(10)]
        assert 1 <= target <= sum(nums)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_target_is_reachable(self, seed):
        nums, target = satisfiable_instance(12, seed)
        assert target in brute_force_sums(nums)
        assert_valid_certificate(subset_sum_backtracking(nums, target), nums, target)

    def test_single_value_instance_targets_that_value(self):
        nums, target = satisfiable_instance(1, 42)
        assert target == nums[0]

    def test_generator_is_reproducible(self):
        assert satisfiable_instance(12, 7) == satisfiable_instance(12, 7)


# ----------------------------------------------------------------------
# Heuristics
# ----------------------------------------------------------------------
class TestGreedy:
    """Largest-first take-if-it-fits: never over, always a sub-multiset."""

    def test_anchor_target_9(self):
        # 34 no, 12 no, 5 yes, 4 yes (9), 3 no, 2 no.
        assert subset_sum_greedy(ANCHOR, 9) == ([5, 4], 9)

    def test_anchor_target_30(self):
        # 34 no, then 12 + 5 + 4 + 3 + 2 = 26 all fit.
        assert subset_sum_greedy(ANCHOR, 30) == ([12, 5, 4, 3, 2], 26)

    def test_greedy_misses_an_exact_answer(self):
        # Takes 6 first; neither 5 fits afterwards, though 5 + 5 = 10.
        assert subset_sum_greedy([6, 5, 5], 10) == ([6], 6)
        assert subset_sum_backtracking([6, 5, 5], 10) == [5, 5]

    def test_empty_input(self):
        assert subset_sum_greedy([], 7) == ([], 0)
        assert subset_sum_greedy([], 0) == ([], 0)

    def test_never_exceeds_target_on_a_battery(self):
        for case in range(300):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "greedy", case), max_n=15, max_value=50
            )
            subset, achieved = subset_sum_greedy(nums, target)
            assert achieved == sum(subset)
            assert 0 <= achieved <= target
            assert is_sub_multiset(subset, nums)
            if achieved == target:
                assert subset_sum_dp(nums, target)


class TestRandomRestart:
    """Best of random-order greedy passes: seeded and reproducible."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_reproducible_with_a_fixed_seed(self, seed):
        for case in range(30):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "restart-repro", case), max_n=15, max_value=50
            )
            first = subset_sum_random_restart(nums, target, tries=25, seed=seed)
            second = subset_sum_random_restart(nums, target, tries=25, seed=seed)
            assert first == second

    def test_reproducible_on_a_large_satisfiable_instance(self):
        nums, target = satisfiable_instance(40, 2026)
        assert subset_sum_random_restart(nums, target, seed=11) == (
            subset_sum_random_restart(nums, target, seed=11)
        )

    def test_never_exceeds_target_on_a_battery(self):
        for case in range(200):
            nums, target = random_case(
                derive_seed(MASTER_SEED, "restart", case), max_n=15, max_value=50
            )
            subset, achieved = subset_sum_random_restart(
                nums, target, tries=10, seed=derive_seed(MASTER_SEED, "rr-seed", case)
            )
            assert achieved == sum(subset)
            assert 0 <= achieved <= target
            assert is_sub_multiset(subset, nums)
            if achieved == target:
                assert subset_sum_dp(nums, target)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_escapes_the_greedy_trap(self, seed):
        # A try fails only if the 6 comes first (probability 1/3), so 100
        # tries all failing has probability 3^-100.
        assert subset_sum_random_restart([6, 5, 5], 10, seed=seed) == ([5, 5], 10)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_one_try_is_one_seeded_shuffle_then_one_pass(self, seed):
        nums = [17, 3, 9, 12, 5, 8, 1, 20]
        target = 30
        rng = make_rng(seed)
        order = list(nums)
        rng.shuffle(order)
        expected, total = [], 0
        for value in order:
            if total + value <= target:
                expected.append(value)
                total += value
        assert subset_sum_random_restart(nums, target, tries=1, seed=seed) == (
            expected,
            total,
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_keeps_the_first_of_equal_best_tries(self, seed):
        # Contract: every try shuffles a fresh copy with one make_rng(seed)
        # generator, the best sum is kept with ties going to the FIRST try
        # that reached it, and the loop stops on an exact hit. The
        # reference below is written from that contract alone. Targets
        # just above the total of a short list cannot be hit, so many tries
        # tie and the tie rule decides which subset comes back.
        for case in range(40):
            rng_case = make_rng(derive_seed(MASTER_SEED, "tie", case))
            nums = [rng_case.randint(1, 9) for _ in range(rng_case.randint(3, 8))]
            target = rng_case.randint(1, sum(nums) + 3)
            rng = make_rng(seed)
            best, best_sum = [], -1
            for _ in range(15):
                order = list(nums)
                rng.shuffle(order)
                taken, total = [], 0
                for value in order:
                    if total + value <= target:
                        taken.append(value)
                        total += value
                if total > best_sum:
                    best, best_sum = taken, total
                    if total == target:
                        break
            assert subset_sum_random_restart(nums, target, tries=15, seed=seed) == (
                best,
                best_sum,
            ), (nums, target)

    def test_unseeded_call_still_returns_a_valid_answer(self):
        subset, achieved = subset_sum_random_restart(ANCHOR, 9, tries=50)
        assert achieved == sum(subset) <= 9
        assert is_sub_multiset(subset, ANCHOR)

    def test_empty_input(self):
        assert subset_sum_random_restart([], 4, seed=1) == ([], 0)

    def test_tries_must_be_a_positive_int(self):
        with pytest.raises(ValueError):
            subset_sum_random_restart(ANCHOR, 9, tries=0)
        with pytest.raises(TypeError):
            subset_sum_random_restart(ANCHOR, 9, tries=True)
        with pytest.raises(TypeError):
            subset_sum_random_restart(ANCHOR, 9, tries=2.5)

    def test_seed_must_be_an_int_or_none(self):
        with pytest.raises(TypeError):
            subset_sum_random_restart(ANCHOR, 9, seed="42")


# ----------------------------------------------------------------------
# Contract errors (section 5)
# ----------------------------------------------------------------------
SOLVERS: List[Callable] = [
    pytest.param(subset_sum_backtracking, id="backtracking"),
    pytest.param(subset_sum_greedy, id="greedy"),
    pytest.param(lambda nums, target: subset_sum_random_restart(nums, target, seed=1),
                 id="random_restart"),
    pytest.param(subset_sum_dp, id="dp"),
]


@pytest.mark.parametrize("solver", SOLVERS)
class TestContractErrors:
    """TypeError for wrong types, ValueError for negative numbers."""

    @pytest.mark.parametrize("nums", [(3, 4), "34", None, {3, 4}])
    def test_nums_must_be_a_list(self, solver, nums):
        with pytest.raises(TypeError):
            solver(nums, 7)

    @pytest.mark.parametrize("bad", [1.5, 2.0, True, "3", None])
    def test_values_must_be_ints(self, solver, bad):
        with pytest.raises(TypeError):
            solver([3, bad, 4], 7)

    @pytest.mark.parametrize("target", [7.0, True, False, "7", None])
    def test_target_must_be_an_int(self, solver, target):
        with pytest.raises(TypeError):
            solver([3, 4], target)

    def test_negative_value_is_rejected(self, solver):
        with pytest.raises(ValueError):
            solver([3, -4, 5], 7)

    def test_negative_target_is_rejected(self, solver):
        with pytest.raises(ValueError):
            solver([3, 4], -1)

    def test_type_is_checked_before_sign(self, solver):
        with pytest.raises(TypeError):
            solver([-3, 4.0], 7)


class TestGeneratorErrors:
    """The instance generators validate ``n`` and ``seed``."""

    @pytest.mark.parametrize("generator", [worst_case_instance, satisfiable_instance])
    @pytest.mark.parametrize("n", [2.0, True, "5", None])
    def test_n_must_be_an_int(self, generator, n):
        with pytest.raises(TypeError):
            generator(n, 42)

    def test_negative_n_is_rejected(self):
        with pytest.raises(ValueError):
            worst_case_instance(-1, 42)

    def test_satisfiable_instance_needs_at_least_one_value(self):
        with pytest.raises(ValueError):
            satisfiable_instance(0, 42)

    @pytest.mark.parametrize("generator", [worst_case_instance, satisfiable_instance])
    def test_seed_must_be_an_int_or_none(self, generator):
        with pytest.raises(TypeError):
            generator(5, "42")
