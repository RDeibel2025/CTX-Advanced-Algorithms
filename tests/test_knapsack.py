"""Tests for the four 0/1 knapsack variants and the two measurements on them.

A knapsack bug does not announce itself. Every variant in the module
returns a single non-negative integer, and a wrong one is indistinguishable
from a right one unless something that is not the module says otherwise.
Re-deriving the recurrence inside the test would only reproduce whatever
mistake the module made and then call the agreement a pass, so the
correctness claims here rest on witnesses that share no mechanism with the
code under test:

* **The assignment's hand-computed instance.** Weights ``[1, 3, 4, 5]``,
  values ``[1, 4, 5, 7]``, capacity 7, optimum 9. The reasoning is written
  out in :class:`TestHandComputedAnchor`, and the expected numbers came
  from paper rather than from running the module. This is the only check
  in the file that a consistently wrong implementation cannot fool.
* **Brute force over all subsets.** :func:`brute_force_best` enumerates
  every subset with :func:`itertools.combinations`, keeps the ones that
  fit, and returns the largest total value. It is the definition of the
  problem rather than an algorithm for it: no table, no recursion, no
  ordering, no shared sub-expression with the module. It is exponential
  and therefore restricted to the small instances of :data:`BATTERY`,
  which is exactly the size range where an oracle is affordable.
* **The chosen items themselves.** A reported optimum is only believable
  if some subset achieves it, so every list :func:`trace_solution` returns
  is checked for distinct indices, a weight total inside the capacity, and
  a value total equal to the number that was reported. A trace that named
  the wrong items would still sum to the wrong figure.

Two things in this module are measured rather than asserted, and both are
tested as measurements. :func:`knapsack_memo_cells` reports how much of the
grid the top-down solution touched, which is the number the Week 5 report's
counterexample rests on; the sparse instance it quotes is small enough that
the seven reachable states are enumerated by hand in the test comment.
:func:`knapsack_instrumented` reports call counts and nesting depth, and
the properties checked of it are the ones that follow from the shape of
each algorithm rather than from any particular total.

The instance sizes are deliberately tiny. ``knapsack_recursive`` is
exponential, the brute-force oracle is exponential, and this file runs on
every commit, so no battery instance exceeds 12 items and every random draw
is seeded from a fixed constant.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import random
from typing import Callable, Dict, List, Sequence, Tuple

import pytest

from src.dp.knapsack import (
    VARIANTS,
    knapsack_instrumented,
    knapsack_memo,
    knapsack_memo_cells,
    knapsack_recursive,
    knapsack_tab,
    knapsack_tab_rolling,
    trace_solution,
)

Solver = Callable[[Sequence[int], Sequence[int], int], int]
Instance = Tuple[List[int], List[int], int]


# ----------------------------------------------------------------------
# The independent oracle
# ----------------------------------------------------------------------
def brute_force_best(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Return the optimum by enumerating every subset of the items.

    This is the problem statement turned directly into code: look at all
    ``2^n`` subsets, discard the ones that do not fit, and keep the largest
    value among the rest. It shares nothing with any of the four variants -
    no recurrence, no table, no memo, no order of consideration - which is
    the only reason its agreement with them means anything.

    The empty subset always fits and is worth 0, so starting the running
    best at 0 is correct given the module's non-negative values.

    Args:
        weights: Item weights. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value of any subset whose weights fit.
    """
    best = 0
    positions = range(len(weights))

    for size in range(len(weights) + 1):
        for chosen in itertools.combinations(positions, size):
            if sum(weights[index] for index in chosen) <= capacity:
                total = sum(values[index] for index in chosen)
                if total > best:
                    best = total

    return best


def best_value_containing(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
    required: int,
) -> int:
    """Brute-force optimum among the subsets that must contain ``required``.

    Used to price the tempting choice on the hand-computed instance: what
    the best packing is once a particular item has been committed to.

    Args:
        weights: Item weights. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.
        required: The index every considered subset must include.

    Returns:
        The best total value achievable while carrying ``required``.
    """
    others = [index for index in range(len(weights)) if index != required]
    best = values[required]

    for size in range(len(others) + 1):
        for chosen in itertools.combinations(others, size):
            load = weights[required] + sum(weights[index] for index in chosen)
            if load <= capacity:
                total = values[required] + sum(values[index] for index in chosen)
                if total > best:
                    best = total

    return best


# ----------------------------------------------------------------------
# What is under test, and on what instances
# ----------------------------------------------------------------------
#: Every function that answers "how much", by name. ``rolling`` is the
#: O(W)-space form and is absent from the module's own VARIANTS registry,
#: but it must agree with the other three on every input, so it joins them
#: here.
VALUE_VARIANTS: Dict[str, Solver] = {
    "recursive": knapsack_recursive,
    "memo": knapsack_memo,
    "tab": knapsack_tab,
    "rolling": knapsack_tab_rolling,
}
VARIANT_IDS: List[str] = sorted(VALUE_VARIANTS)
VARIANT_PARAMS: List[Solver] = [VALUE_VARIANTS[name] for name in VARIANT_IDS]


def trace_value(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Adapter exposing :func:`trace_solution`'s value with the shared signature."""
    return trace_solution(weights, values, capacity)[0]


def cells_value(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Adapter exposing :func:`knapsack_memo_cells`'s value, same signature."""
    return knapsack_memo_cells(weights, values, capacity)[0]


def instrumented_value(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Adapter exposing :func:`knapsack_instrumented`'s value, tabulated variant."""
    return knapsack_instrumented(weights, values, capacity, "tab")[0]


#: Every public entry point that validates its arguments. The contract
#: tests run over all of them, because a check that holds in one function
#: and not in its neighbour is precisely the kind of drift that goes
#: unnoticed when only the headline variant is tested.
VALIDATING_ENTRY_POINTS: Dict[str, Solver] = {
    "recursive": knapsack_recursive,
    "memo": knapsack_memo,
    "tab": knapsack_tab,
    "rolling": knapsack_tab_rolling,
    "trace_solution": trace_value,
    "memo_cells": cells_value,
    "instrumented": instrumented_value,
}
ENTRY_POINT_IDS: List[str] = sorted(VALIDATING_ENTRY_POINTS)
ENTRY_POINT_PARAMS: List[Solver] = [
    VALIDATING_ENTRY_POINTS[name] for name in ENTRY_POINT_IDS
]

#: Instance families, each with its own seed offset so families cannot draw
#: the same numbers. Each one stresses a different corner of the
#: recurrence, and the memo's sparsity in particular swings wildly between
#: "coarse" and "fine".
FAMILY_SEEDS: Dict[str, int] = {
    "coarse": 1000,  # few large weights, so few capacities are reachable
    "fine": 2000,  # many small weights, so the memo space fills up
    "tight": 3000,  # capacity just under the total weight: the hard case
    "roomy": 4000,  # capacity above the total weight: take everything
    "with_zeros": 5000,  # zero weights and zero values mixed in
}

#: Four draws per family. Twenty instances is enough variety to catch an
#: off-by-one without paying for an exponential oracle two hundred times.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3)


def make_instance(family: str, seed: int) -> Instance:
    """Build one reproducible instance of the named family.

    Every draw comes from a :class:`random.Random` seeded with the family's
    own offset plus ``seed``, so the same pair always produces the same
    instance and a failure can be reproduced from its test id alone. No
    family exceeds 12 items: both ``knapsack_recursive`` and the
    brute-force oracle are exponential, and this file runs on every commit.

    Args:
        family: One of the keys of :data:`FAMILY_SEEDS`.
        seed: Which draw within that family.

    Returns:
        A ``(weights, values, capacity)`` triple.
    """
    rng = random.Random(FAMILY_SEEDS[family] + seed)

    if family == "coarse":
        count = rng.randint(1, 6)
        weights = [rng.randint(10, 40) for _ in range(count)]
        capacity = rng.randint(20, 60)
    elif family == "fine":
        count = rng.randint(1, 12)
        weights = [rng.randint(1, 8) for _ in range(count)]
        capacity = rng.randint(0, 25)
    elif family == "tight":
        count = rng.randint(1, 10)
        weights = [rng.randint(1, 9) for _ in range(count)]
        capacity = max(sum(weights) - rng.randint(1, 5), 0)
    elif family == "roomy":
        count = rng.randint(0, 10)
        weights = [rng.randint(1, 6) for _ in range(count)]
        capacity = sum(weights) + rng.randint(0, 5)
    else:
        count = rng.randint(1, 10)
        weights = [rng.choice((0, 0, rng.randint(1, 7))) for _ in range(count)]
        capacity = rng.randint(0, 15)

    values = [rng.randint(0, 20) for _ in range(count)]
    return weights, values, capacity


#: Every ``(family, seed)`` pair, as pytest parameters with readable ids.
BATTERY = [
    pytest.param(family, seed, id=f"{family}-seed{seed}")
    for family in sorted(FAMILY_SEEDS)
    for seed in SEEDS
]


# ----------------------------------------------------------------------
# The hand-computed case
# ----------------------------------------------------------------------
class TestHandComputedAnchor:
    """The assignment's instance, worked out on paper before any code ran.

    Four items, capacity 7::

        index:   0    1    2    3
        weight:  1    3    4    5
        value:   1    4    5    7

    Enumerated by hand, only the subsets that fit in 7:

    * ``{1, 2}`` weighs 3 + 4 = 7 and is worth 4 + 5 = **9**. It fills the
      bag exactly, with nothing left over.
    * ``{0, 3}`` weighs 1 + 5 = 6 and is worth 1 + 7 = 8. The single most
      valuable item is index 3, but taking it leaves only 2 units of room,
      and the only item that fits in 2 is the weight-1 item worth 1. So the
      greedy "take the biggest value first" choice tops out at 8.
    * ``{0, 1}`` weighs 4 and is worth 5; ``{0, 2}`` weighs 5 and is worth
      6; ``{1, 2}`` at 9 beats both, and no three items fit at all, since
      the three lightest already weigh 1 + 3 + 4 = 8 > 7.

    So the optimum is 9, achieved by indices 1 and 2, and the instance is
    chosen precisely because the tempting item is the wrong one. Nothing in
    this class may be regenerated from the module's output.
    """

    WEIGHTS: List[int] = [1, 3, 4, 5]
    VALUES: List[int] = [1, 4, 5, 7]
    CAPACITY: int = 7
    OPTIMUM: int = 9
    CHOSEN: List[int] = [1, 2]

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_every_variant_returns_the_paper_answer(self, solver: Solver) -> None:
        assert solver(self.WEIGHTS, self.VALUES, self.CAPACITY) == self.OPTIMUM

    def test_the_oracle_agrees_with_the_paper_answer(self) -> None:
        """Sanity check on the oracle itself, before anything is compared to it."""
        assert brute_force_best(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9

    def test_taking_the_most_valuable_item_only_reaches_eight(self) -> None:
        """The greedy choice leaves capacity spare and loses by one."""
        # Item 3 weighs 5 and is worth 7; committing to it leaves room 2,
        # into which only the weight-1 item worth 1 fits. 7 + 1 = 8 < 9.
        greedy = best_value_containing(self.WEIGHTS, self.VALUES, self.CAPACITY, 3)
        assert greedy == 8
        assert knapsack_tab(self.WEIGHTS, self.VALUES, self.CAPACITY) > greedy

    def test_trace_names_the_two_middle_items(self) -> None:
        assert trace_solution(self.WEIGHTS, self.VALUES, self.CAPACITY) == (
            self.OPTIMUM,
            self.CHOSEN,
        )

    def test_the_chosen_items_fill_the_bag_exactly(self) -> None:
        _best, chosen = trace_solution(self.WEIGHTS, self.VALUES, self.CAPACITY)
        assert sum(self.WEIGHTS[index] for index in chosen) == self.CAPACITY

    def test_the_chosen_values_total_the_reported_optimum(self) -> None:
        best, chosen = trace_solution(self.WEIGHTS, self.VALUES, self.CAPACITY)
        assert sum(self.VALUES[index] for index in chosen) == best == self.OPTIMUM

    @pytest.mark.parametrize("name", sorted(VARIANTS), ids=sorted(VARIANTS))
    def test_the_registry_entries_solve_the_anchor(self, name: str) -> None:
        assert VARIANTS[name](self.WEIGHTS, self.VALUES, self.CAPACITY) == self.OPTIMUM

    def test_a_sequence_that_is_not_a_list_is_accepted(self) -> None:
        """Tuples work too: the module copies its inputs rather than indexing them."""
        assert knapsack_tab(tuple(self.WEIGHTS), tuple(self.VALUES), 7) == self.OPTIMUM


# ----------------------------------------------------------------------
# Agreement with brute force
# ----------------------------------------------------------------------
class TestAgreementWithBruteForce:
    """Every variant against an enumeration of all subsets, on 20 instances.

    This is the strongest general claim in the file. The oracle is not an
    algorithm for knapsack, it is the definition of knapsack, so where a
    variant disagrees with it the variant is wrong and there is nothing to
    argue about.
    """

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_variant_matches_the_oracle(
        self, family: str, seed: int, solver: Solver
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        assert solver(weights, values, capacity) == brute_force_best(
            weights, values, capacity
        )

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_the_four_variants_return_exactly_one_value(
        self, family: str, seed: int
    ) -> None:
        """The module's central invariant, stated as a set with one member."""
        weights, values, capacity = make_instance(family, seed)
        answers = {
            solver(weights, values, capacity) for solver in VALUE_VARIANTS.values()
        }
        assert len(answers) == 1

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    @pytest.mark.parametrize("capacity", [0, 1, 4, 7, 9, 13, 20])
    def test_one_instance_at_every_capacity(
        self, capacity: int, solver: Solver
    ) -> None:
        """Sweeping the capacity catches an off-by-one that one value hides."""
        weights, values = [3, 4, 5, 8, 10], [4, 5, 8, 9, 10]
        assert solver(weights, values, capacity) == brute_force_best(
            weights, values, capacity
        )

    def test_the_density_greedy_choice_is_beaten(self) -> None:
        """The classic instance: value density picks the weight-10 item and loses."""
        # Densities are 6.0, 5.0, 4.0 per unit weight, so a greedy pass
        # takes item 0 and then item 1, filling 30 of 50 for 160. The
        # optimum leaves item 0 behind and packs 20 + 30 for 220.
        weights, values = [10, 20, 30], [60, 100, 120]
        assert brute_force_best(weights, values, 50) == 220
        for solver in VALUE_VARIANTS.values():
            assert solver(weights, values, 50) == 220
        assert trace_solution(weights, values, 50) == (220, [1, 2])


# ----------------------------------------------------------------------
# Recovering the choices
# ----------------------------------------------------------------------
def assert_trace_is_a_real_packing(instance: Instance) -> None:
    """Assert the traced indices are a distinct, fitting, exactly-valued subset.

    Args:
        instance: The ``(weights, values, capacity)`` triple to check.
    """
    weights, values, capacity = instance
    best, chosen = trace_solution(weights, values, capacity)

    assert len(set(chosen)) == len(chosen)
    assert all(0 <= index < len(weights) for index in chosen)
    assert sum(weights[index] for index in chosen) <= capacity
    assert sum(values[index] for index in chosen) == best


class TestTraceSolution:
    """A number is only believable if some subset actually achieves it.

    Three separate properties are checked rather than one compound
    assertion, because they fail for different reasons: duplicated indices
    mean the backward walk stepped twice on one row, an overweight total
    means it subtracted the wrong weight, and a value total that misses the
    reported optimum means the items named are not the items the table
    priced.
    """

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_indices_are_distinct_and_in_range(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        _best, chosen = trace_solution(weights, values, capacity)
        assert len(set(chosen)) == len(chosen)
        assert all(0 <= index < len(weights) for index in chosen)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_indices_are_in_ascending_order(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        _best, chosen = trace_solution(weights, values, capacity)
        assert chosen == sorted(chosen)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_chosen_weights_fit_in_the_bag(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        _best, chosen = trace_solution(weights, values, capacity)
        assert sum(weights[index] for index in chosen) <= capacity

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_chosen_values_total_the_reported_optimum(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        best, chosen = trace_solution(weights, values, capacity)
        assert sum(values[index] for index in chosen) == best

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_reported_value_equals_knapsack_tab(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        best, _chosen = trace_solution(weights, values, capacity)
        assert best == knapsack_tab(weights, values, capacity)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_reported_value_equals_the_oracle(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        best, _chosen = trace_solution(weights, values, capacity)
        assert best == brute_force_best(weights, values, capacity)

    def test_the_anchor_packing_is_real(self) -> None:
        assert_trace_is_a_real_packing(([1, 3, 4, 5], [1, 4, 5, 7], 7))

    @pytest.mark.parametrize(
        "instance",
        [
            ([], [], 5),
            ([8, 9], [100, 200], 7),
            ([7], [5], 7),
            ([0, 5], [3, 9], 0),
            ([1, 2], [0, 0], 5),
        ],
        ids=["empty", "nothing_fits", "exact_fit", "free_item", "zero_values"],
    )
    def test_edge_instances_are_real_packings(self, instance: Instance) -> None:
        assert_trace_is_a_real_packing(instance)

    def test_nothing_is_chosen_when_nothing_fits(self) -> None:
        assert trace_solution([8, 9], [100, 200], 7) == (0, [])

    def test_an_empty_instance_chooses_nothing(self) -> None:
        assert trace_solution([], [], 5) == (0, [])

    def test_free_items_are_taken_at_capacity_zero(self) -> None:
        """A weight-0 item costs nothing, so the walk records it even at room 0."""
        best, chosen = trace_solution([0, 5], [3, 9], 0)
        assert (best, chosen) == (3, [0])

    def test_a_tie_returns_the_earlier_item(self) -> None:
        """Two interchangeable items: the walk keeps the earlier index.

        Entering from the last row, an item is recorded only when
        ``table[i][room] != table[i - 1][room]``. On a tie those two cells
        are equal, so the later item is skipped and the earlier one is
        picked up higher in the walk. Both subsets are optimal, so either
        answer would be correct; this pins the one the implementation
        actually returns.

        This test originally found the docstring describing the tie-break
        the other way round, which has since been corrected.
        """
        assert trace_solution([2, 2], [3, 3], 2) == (3, [0])
        assert trace_solution([4, 4], [9, 9], 4) == (9, [0])


# ----------------------------------------------------------------------
# How much of the grid the memo touched
# ----------------------------------------------------------------------
class TestMemoCellCount:
    """The measurement the Week 5 report's counterexample rests on.

    ``knapsack_memo_cells`` returns ``(value, memo_entries, table_cells)``.
    The claim being supported is that top-down memoization can touch far
    fewer states than tabulation fills, so the two numbers that matter are
    that the entry count is real work (positive whenever there is an item)
    and that it never exceeds what the full table would cost. A measurement
    that could exceed its own upper bound would not be evidence of
    anything.
    """

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_value_matches_knapsack_tab(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        value, _entries, _cells = knapsack_memo_cells(weights, values, capacity)
        assert value == knapsack_tab(weights, values, capacity)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_value_matches_the_oracle(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        value, _entries, _cells = knapsack_memo_cells(weights, values, capacity)
        assert value == brute_force_best(weights, values, capacity)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_entry_count_is_positive_when_there_are_items(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        _value, entries, _cells = knapsack_memo_cells(weights, values, capacity)
        if weights:
            # The root state (0, capacity) is always stored, so any
            # instance with an item has at least one entry.
            assert entries > 0
        else:
            assert entries == 0

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_entry_count_never_exceeds_the_table(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        _value, entries, cells = knapsack_memo_cells(weights, values, capacity)
        assert entries <= cells

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_table_cells_is_the_full_grid_below_row_zero(
        self, family: str, seed: int
    ) -> None:
        """The comparison is against what tabulation writes, not the array size."""
        weights, values, capacity = make_instance(family, seed)
        _value, _entries, cells = knapsack_memo_cells(weights, values, capacity)
        assert cells == len(weights) * (capacity + 1)

    def test_the_sparse_counterexample_is_seven_states_out_of_153(self) -> None:
        """Coarse weights reach almost none of the grid, counted by hand.

        Weights 10, 20, 30 in a bag of 50. The reachable states, written as
        ``(item index, remaining capacity)`` and starting from ``(0, 50)``:

            (0, 50)                     the root
            (1, 50) (1, 40)             skip item 0, or take its weight 10
            (2, 50) (2, 30)             from (1, 50): skip, or take 20
            (2, 40) (2, 20)             from (1, 40): skip, or take 20

        That is 7 distinct states. The base case ``index == 3`` is a
        constant and is never stored. The full table would write
        ``3 * (50 + 1) = 153`` cells, so the memo touches under 5% of it.
        """
        assert knapsack_memo_cells([10, 20, 30], [60, 100, 120], 50) == (220, 7, 153)

    def test_fine_weights_fill_much_more_of_the_grid(self) -> None:
        """The sparsity is a property of the instance, not of the method."""
        weights = list(range(1, 11))
        value, entries, cells = knapsack_memo_cells(weights, weights, 15)
        assert (value, cells) == (15, 10 * 16)
        # Weights 1..10 share many subset sums, so most capacities below 15
        # are reachable and the saving nearly vanishes.
        assert entries > cells // 2

    def test_an_empty_instance_has_no_states_and_no_table(self) -> None:
        assert knapsack_memo_cells([], [], 10) == (0, 0, 0)

    def test_capacity_zero_stores_one_state_per_item(self) -> None:
        """With no room, only ``(index, 0)`` is ever asked for."""
        value, entries, cells = knapsack_memo_cells([1, 3, 4], [1, 4, 5], 0)
        assert (value, entries, cells) == (0, 3, 3)


# ----------------------------------------------------------------------
# The rolling row
# ----------------------------------------------------------------------
class TestRollingRow:
    """The O(W)-space form must be the O(n*W) one with the rows thrown away.

    The single risk in collapsing the table is the scan direction. Sweeping
    the row upwards would read a cell the current item had already written,
    letting one item be taken more than once and silently solving the
    *unbounded* knapsack instead. That failure is invisible on instances
    where no item would want to repeat, so it is provoked directly.
    """

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_agrees_with_the_full_table(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        assert knapsack_tab_rolling(weights, values, capacity) == knapsack_tab(
            weights, values, capacity
        )

    def test_agrees_with_the_full_table_at_every_capacity(self) -> None:
        weights, values = [3, 4, 5, 8, 10], [4, 5, 8, 9, 10]
        for capacity in range(31):
            assert knapsack_tab_rolling(weights, values, capacity) == knapsack_tab(
                weights, values, capacity
            ), capacity

    @pytest.mark.parametrize(
        "weight, value, capacity, expected",
        [
            (2, 3, 6, 3),  # unbounded would say 9: three copies of the item
            (1, 5, 4, 5),  # unbounded would say 20
            (3, 7, 9, 7),  # unbounded would say 21
        ],
        ids=["three_copies", "four_copies", "three_copies_again"],
    )
    def test_a_single_item_is_never_taken_twice(
        self, weight: int, value: int, capacity: int, expected: int
    ) -> None:
        """0/1 means once. A bag with room to spare does not get a second copy."""
        assert knapsack_tab_rolling([weight], [value], capacity) == expected

    def test_free_items_are_still_taken_once_each(self) -> None:
        """Weight 0 makes ``row[room - weight]`` the cell being written.

        Two free items at capacity 0 are both taken, for 3 + 4 = 7, and
        neither is taken twice.
        """
        assert knapsack_tab_rolling([0, 0], [3, 4], 0) == 7
        assert knapsack_tab_rolling([0, 0], [3, 4], 0) == knapsack_tab(
            [0, 0], [3, 4], 0
        )

    def test_it_is_not_one_of_the_benchmarked_variants(self) -> None:
        """The rolling row is the space experiment, not a fourth timing series."""
        assert knapsack_tab_rolling not in VARIANTS.values()
        assert sorted(VARIANTS) == ["memo", "recursive", "tab"]


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    """The instances where the recurrence has nothing to choose between.

    Each one is run against all four variants, because these are exactly
    the cases where a base case differs between a recursion and a loop: an
    empty item list never enters either, and a capacity of 0 stops one of
    them early if the stopping rule is written carelessly.
    """

    @pytest.mark.parametrize("capacity", [0, 1, 7, 50])
    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_no_items_is_zero_at_any_capacity(
        self, solver: Solver, capacity: int
    ) -> None:
        assert solver([], [], capacity) == 0

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_capacity_zero_is_zero_when_every_weight_is_positive(
        self, solver: Solver
    ) -> None:
        assert solver([1, 3, 4, 5], [1, 4, 5, 7], 0) == 0

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_a_single_item_that_does_not_fit_is_left_behind(
        self, solver: Solver
    ) -> None:
        assert solver([8], [100], 7) == 0

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_a_single_item_that_fits_exactly_is_taken(self, solver: Solver) -> None:
        assert solver([7], [5], 7) == 5

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_a_single_item_one_unit_too_heavy_is_left_behind(
        self, solver: Solver
    ) -> None:
        """The boundary either side of an exact fit, which is where an
        off-by-one in the ``weight <= room`` test would show."""
        assert solver([8], [5], 7) == 0

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_every_item_too_heavy_is_zero(self, solver: Solver) -> None:
        assert solver([8, 9, 10], [5, 6, 7], 7) == 0

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_zero_value_items_are_worth_nothing(self, solver: Solver) -> None:
        assert solver([1, 2, 3], [0, 0, 0], 6) == 0

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_a_zero_weight_item_is_free_even_at_capacity_zero(
        self, solver: Solver
    ) -> None:
        """A free item is always taken, so a capacity of 0 can still pay out.

        This is the recurrence behaving correctly. Stopping the recursion
        early on ``remaining == 0`` would return 0 here and be wrong.
        """
        assert solver([0, 5], [3, 9], 0) == 3

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_zero_weight_items_are_all_taken(self, solver: Solver) -> None:
        assert solver([0, 2, 0], [3, 9, 4], 2) == 16

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_everything_fits_so_everything_is_taken(self, solver: Solver) -> None:
        weights, values = [1, 2, 3, 4], [5, 6, 7, 8]
        assert solver(weights, values, sum(weights)) == sum(values)

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_a_capacity_beyond_the_total_weight_changes_nothing(
        self, solver: Solver
    ) -> None:
        weights, values = [1, 2, 3, 4], [5, 6, 7, 8]
        assert solver(weights, values, sum(weights) + 25) == sum(values)

    @pytest.mark.parametrize("solver", VARIANT_PARAMS, ids=VARIANT_IDS)
    def test_a_single_zero_weight_zero_value_item(self, solver: Solver) -> None:
        assert solver([0], [0], 0) == 0


# ----------------------------------------------------------------------
# The shared contract
# ----------------------------------------------------------------------
class TestContracts:
    """Argument checking, run over every public entry point at once.

    All seven entry points funnel through one validator, which is the
    reason to parametrise rather than test the headline variant and assume:
    the moment one of them stops calling it, or calls it with the arguments
    in the wrong order, these tests are what says so.
    """

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_more_weights_than_values_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError):
            solver([1, 2], [3], 5)

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_more_values_than_weights_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError):
            solver([1], [3, 4], 5)

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_the_length_mismatch_message_names_both_lengths(
        self, solver: Solver
    ) -> None:
        with pytest.raises(ValueError, match="same length"):
            solver([1, 2, 3], [4], 5)

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    @pytest.mark.parametrize("capacity", [-1, -5, -100])
    def test_negative_capacity_is_a_value_error(
        self, solver: Solver, capacity: int
    ) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            solver([1, 2], [3, 4], capacity)

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    @pytest.mark.parametrize(
        "capacity",
        [1.5, 7.0, "7", None, [7]],
        ids=["float", "whole_float", "str", "none", "list"],
    )
    def test_non_int_capacity_is_a_type_error(
        self, solver: Solver, capacity: object
    ) -> None:
        with pytest.raises(TypeError, match="capacity must be an int"):
            solver([1, 2], [3, 4], capacity)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_a_bool_capacity_is_a_type_error(self, solver: Solver) -> None:
        """``True`` is an accident, not a capacity, even though it is an int."""
        with pytest.raises(TypeError, match="got bool"):
            solver([1, 2], [3, 4], True)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_a_negative_weight_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match=r"weights\[1\] must be >= 0"):
            solver([1, -2], [3, 4], 5)

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_a_negative_value_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match=r"values\[1\] must be >= 0"):
            solver([1, 2], [3, -4], 5)

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    @pytest.mark.parametrize(
        "entry",
        [2.5, 4.0, "2", None, True],
        ids=["float", "whole_float", "str", "none", "bool"],
    )
    def test_a_non_int_weight_is_a_type_error(
        self, solver: Solver, entry: object
    ) -> None:
        with pytest.raises(TypeError, match=r"weights\[1\] must be an int"):
            solver([1, entry], [3, 4], 5)  # type: ignore[list-item]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    @pytest.mark.parametrize(
        "entry",
        [2.5, 4.0, "2", None, True],
        ids=["float", "whole_float", "str", "none", "bool"],
    )
    def test_a_non_int_value_is_a_type_error(
        self, solver: Solver, entry: object
    ) -> None:
        with pytest.raises(TypeError, match=r"values\[1\] must be an int"):
            solver([1, 2], [3, entry], 5)  # type: ignore[list-item]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_weights_that_are_not_a_sequence_is_a_type_error(
        self, solver: Solver
    ) -> None:
        with pytest.raises(TypeError, match="weights must be a sequence of ints"):
            solver(7, [1], 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_values_that_are_not_a_sequence_is_a_type_error(
        self, solver: Solver
    ) -> None:
        with pytest.raises(TypeError, match="values must be a sequence of ints"):
            solver([1], 7, 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_the_capacity_type_is_checked_before_the_lengths(
        self, solver: Solver
    ) -> None:
        """Two faults at once still reports the type error, not the mismatch."""
        with pytest.raises(TypeError, match="capacity must be an int"):
            solver([1, 2], [3], 1.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_POINT_PARAMS, ids=ENTRY_POINT_IDS)
    def test_the_caller_s_lists_are_never_modified(self, solver: Solver) -> None:
        """Validation copies, so a solved instance can be solved again."""
        weights, values = [1, 3, 4, 5], [1, 4, 5, 7]
        solver(weights, values, 7)
        assert weights == [1, 3, 4, 5]
        assert values == [1, 4, 5, 7]


# ----------------------------------------------------------------------
# The registry and the instrumented mirrors
# ----------------------------------------------------------------------
class TestVariantRegistry:
    """What the benchmark is allowed to find in :data:`VARIANTS`."""

    def test_it_holds_exactly_the_three_benchmarked_names(self) -> None:
        assert sorted(VARIANTS) == ["memo", "recursive", "tab"]

    def test_the_entries_are_the_module_functions_themselves(self) -> None:
        assert VARIANTS["recursive"] is knapsack_recursive
        assert VARIANTS["memo"] is knapsack_memo
        assert VARIANTS["tab"] is knapsack_tab

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_every_registry_entry_agrees_with_the_oracle(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        expected = brute_force_best(weights, values, capacity)
        for solver in VARIANTS.values():
            assert solver(weights, values, capacity) == expected


class TestInstrumentedMirrors:
    """The counted mirrors must answer the same question as the fast paths.

    The counts themselves are a measurement rather than a specification, so
    what is asserted here is the shape of each algorithm: tabulation writes
    one cell per grid position and never nests, the two recursions nest one
    frame per item, and memoization cannot enter the recursion more often
    than plain recursion does. The instances are kept small deliberately;
    ``"recursive"`` is exponential and this runs on every commit.
    """

    SMALL: List[Instance] = [
        ([1, 3, 4, 5], [1, 4, 5, 7], 7),
        ([10, 20, 30], [60, 100, 120], 50),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 15),
        ([2, 2, 2], [3, 3, 3], 4),
        ([], [], 6),
    ]
    SMALL_IDS: List[str] = ["anchor", "coarse", "ten_items", "ties", "empty"]

    @pytest.mark.parametrize("name", sorted(VARIANTS), ids=sorted(VARIANTS))
    @pytest.mark.parametrize("instance", SMALL, ids=SMALL_IDS)
    def test_the_counted_mirror_returns_the_same_value(
        self, instance: Instance, name: str
    ) -> None:
        weights, values, capacity = instance
        value, _counter = knapsack_instrumented(weights, values, capacity, name)
        assert value == brute_force_best(weights, values, capacity)

    @pytest.mark.parametrize("instance", SMALL, ids=SMALL_IDS)
    def test_tabulation_never_nests(self, instance: Instance) -> None:
        """Depth 1 by construction: the loop body is entered and left flat."""
        weights, values, capacity = instance
        _value, counter = knapsack_instrumented(weights, values, capacity, "tab")
        assert counter.max_depth == (1 if weights else 0)

    @pytest.mark.parametrize("instance", SMALL, ids=SMALL_IDS)
    def test_tabulation_counts_one_subproblem_per_cell(
        self, instance: Instance
    ) -> None:
        weights, values, capacity = instance
        _value, counter = knapsack_instrumented(weights, values, capacity, "tab")
        assert counter.calls == len(weights) * (capacity + 1)

    @pytest.mark.parametrize("instance", SMALL, ids=SMALL_IDS)
    @pytest.mark.parametrize("name", ["recursive", "memo"])
    def test_the_recursions_nest_one_frame_per_item(
        self, instance: Instance, name: str
    ) -> None:
        """Caching removes repeated work, not nesting: both reach n + 1."""
        weights, values, capacity = instance
        _value, counter = knapsack_instrumented(weights, values, capacity, name)
        assert counter.max_depth == len(weights) + 1

    @pytest.mark.parametrize("instance", SMALL, ids=SMALL_IDS)
    def test_memoization_never_enters_the_recursion_more_often(
        self, instance: Instance
    ) -> None:
        weights, values, capacity = instance
        _value, naive = knapsack_instrumented(weights, values, capacity, "recursive")
        _value, memoized = knapsack_instrumented(weights, values, capacity, "memo")
        assert memoized.calls <= naive.calls

    def test_the_memo_saves_most_of_the_work_on_ten_items(self) -> None:
        """Ten items in a bag of 15: the gap is the whole point of the table."""
        weights = list(range(1, 11))
        _value, naive = knapsack_instrumented(weights, weights, 15, "recursive")
        _value, memoized = knapsack_instrumented(weights, weights, 15, "memo")
        assert memoized.calls < naive.calls // 2

    def test_two_runs_share_no_state(self) -> None:
        """A fresh counter and a fresh cache per call, so counts cannot leak."""
        first = knapsack_instrumented([1, 2], [1, 2], 3, "recursive")[1]
        second = knapsack_instrumented([1, 2], [1, 2], 3, "recursive")[1]
        assert first.calls == second.calls
        assert first is not second

    def test_the_counters_are_left_unwound(self) -> None:
        """Every frame entered was left, so the depth is back to 0 at the end."""
        for name in sorted(VARIANTS):
            _value, counter = knapsack_instrumented([1, 3, 4], [1, 4, 5], 5, name)
            assert counter.depth == 0

    @pytest.mark.parametrize(
        "name", ["rolling", "tabulation", "Tab", "", "memoized"]
    )
    def test_an_unknown_variant_is_a_value_error(self, name: str) -> None:
        with pytest.raises(ValueError, match="unknown variant"):
            knapsack_instrumented([1], [1], 1, name)

    def test_the_variant_name_is_checked_before_the_arguments(self) -> None:
        """A bad name and a bad capacity together still reports the name."""
        with pytest.raises(ValueError, match="unknown variant"):
            knapsack_instrumented([1], [1], -1, "nope")
