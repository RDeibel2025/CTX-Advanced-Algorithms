"""Tests for the one-row 0/1 knapsack and the proof of why its scan runs down.

Week 6 keeps the Week 5 recurrence and throws away the item dimension of
the table: one row of ``W + 1`` integers, overwritten in place once per
item. That is only legal because the capacity scan runs DOWNWARD. Scanning
upward lets an item read a cell it has already written, take itself a
second time, and silently solve the UNBOUNDED knapsack instead. The module
ships that wrong version on purpose, as :func:`knapsack_ascending_scan`, so
the argument can be demonstrated rather than merely asserted. This file
proves four things:

* **The direction argument.** On weights ``[2, 3]``, values ``[3, 4]`` and
  capacity 6 the downward scan returns 7 (both items once) and the upward
  scan returns 9 (the weight-2 item three times). Both numbers, and both
  rows, are worked out by hand in
  :class:`TestScanDirection`. On a seeded battery the upward scan then
  equals an independent unbounded-knapsack oracle and the downward scan
  equals an independent 0/1 oracle, which is the claim that the two scans
  are two different recurrences and not one recurrence with noise.
* **Agreement with independent witnesses.** The one-row solver is compared
  with the assignment's hand-computed instance (weights ``[1, 3, 4, 5]``,
  values ``[1, 4, 5, 7]``, capacity 7, optimum 9), with brute force over
  every subset for ``n <= 12``, and with both Week 5 baselines
  (``knapsack_tab`` and ``knapsack_tab_rolling``) for ``n <= 15`` and
  ``W <= 60``. Brute force shares no mechanism with the module, which is
  the only reason its agreement means anything.
* **The recovered items are a real packing.** One row has no history, so
  :func:`knapsack_space_optimized_with_items` buys the items back with one
  bitset per item. Every list it returns is checked for distinct indices,
  a weight total inside the capacity, and a value total equal to the
  reported optimum, which in turn must equal the brute-force optimum.
* **The measurement is a measurement.** :func:`compare_with_standard` is
  run once at ``n = 50``, ``W = 500``. Its answers must agree with each
  other and with ``knapsack_tab``, every documented key must be present,
  and the full table's traced peak must exceed the one row's. No specific
  KiB or timing figure is asserted, because those depend on the machine.

The contract (``TypeError`` on non-int, ``bool`` rejected, ``ValueError`` on
negatives and mismatched lengths) is checked on every entry point and is
also checked to raise exactly the Week 5 exception and message, since the
module promises to be a drop-in replacement.

Every random draw is seeded from a fixed constant and every oracle instance
is small, because both oracles are exponential and this file runs with a
suite of several thousand tests.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytest

from src.dp.knapsack import knapsack_tab, knapsack_tab_rolling, trace_solution
from src.dp_advanced.space_optimized_knapsack import (
    compare_with_standard,
    knapsack_ascending_scan,
    knapsack_space_optimized,
    knapsack_space_optimized_with_items,
    print_comparison,
)

Solver = Callable[[Sequence[int], Sequence[int], int], int]
Instance = Tuple[List[int], List[int], int]


# ----------------------------------------------------------------------
# The independent oracles
# ----------------------------------------------------------------------
def brute_force_01(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Return the 0/1 optimum by enumerating every subset as a bitmask.

    Each integer ``mask`` in ``[0, 2^n)`` names one subset: item ``i`` is
    in it when bit ``i`` is set. Every subset is priced from scratch, the
    ones that do not fit are discarded, and the best of the rest is kept.
    There is no row, no table, no recursion and no scan direction, so a
    mistake in the module cannot be reproduced here.

    Args:
        weights: Item weights. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value of any subset whose weights fit.
    """
    count = len(weights)
    best = 0
    for mask in range(1 << count):
        load = 0
        total = 0
        for index in range(count):
            if (mask >> index) & 1:
                load += weights[index]
                total += values[index]
        if load <= capacity and total > best:
            best = total
    return best


def brute_force_unbounded(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
) -> int:
    """Return the UNBOUNDED optimum by enumerating every multiset that fits.

    Each item may be taken any number of times. The enumeration chooses a
    number of copies for item 0, then for item 1, and so on, never letting
    the load exceed the capacity, and prices every complete choice of
    copies. It stores nothing between branches, so it is a listing of the
    feasible multisets rather than a dynamic program.

    Unbounded knapsack is only defined for positive weights (a free item
    with positive value would be worth infinitely much), so a zero weight
    is refused rather than looped on forever.

    Args:
        weights: Item weights, all strictly positive. Not modified.
        values: Item values, parallel to ``weights``. Not modified.
        capacity: The knapsack capacity.

    Returns:
        The maximum total value of any multiset of items that fits.
    """
    assert all(weight > 0 for weight in weights), "unbounded needs weights > 0"
    best = 0

    def extend(index: int, room: int, total: int) -> None:
        nonlocal best
        if index == len(weights):
            if total > best:
                best = total
            return
        copies = 0
        while copies * weights[index] <= room:
            extend(
                index + 1,
                room - copies * weights[index],
                total + copies * values[index],
            )
            copies += 1

    extend(0, capacity, 0)
    return best


# ----------------------------------------------------------------------
# What is under test, and on what instances
# ----------------------------------------------------------------------
def items_value(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    """Adapter exposing the value half of the item-recovering solver."""
    return knapsack_space_optimized_with_items(weights, values, capacity)[0]


#: The two correct 0/1 solvers in this module, by name.
ZERO_ONE_SOLVERS: Dict[str, Solver] = {
    "space_optimized": knapsack_space_optimized,
    "with_items": items_value,
}
ZERO_ONE_IDS: List[str] = sorted(ZERO_ONE_SOLVERS)
ZERO_ONE_PARAMS: List[Solver] = [ZERO_ONE_SOLVERS[name] for name in ZERO_ONE_IDS]

#: Every public solver that validates its arguments, the deliberately wrong
#: one included. The contract is shared, so it is checked on all three.
ENTRY_POINTS: Dict[str, Solver] = {
    "ascending_scan": knapsack_ascending_scan,
    "space_optimized": knapsack_space_optimized,
    "with_items": items_value,
}
ENTRY_IDS: List[str] = sorted(ENTRY_POINTS)
ENTRY_PARAMS: List[Solver] = [ENTRY_POINTS[name] for name in ENTRY_IDS]

#: Instance families, each with its own seed offset so families cannot
#: draw the same numbers. No family exceeds 15 items or a capacity of 60.
FAMILY_SEEDS: Dict[str, int] = {
    "coarse": 61000,  # few heavy items, so few capacities are reachable
    "fine": 62000,  # many light items, so the row fills up
    "tight": 63000,  # capacity just under the total weight: the hard case
    "roomy": 64000,  # capacity at or above the total weight: take everything
    "with_zeros": 65000,  # zero weights and zero values mixed in
    "wide": 66000,  # 13 to 15 items: Week 5 baselines only, too big for 2^n
}

#: Six draws per family, 36 instances in all.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

#: Largest instance the subset oracle is asked to enumerate.
ORACLE_MAX_ITEMS = 12


def make_instance(family: str, seed: int) -> Instance:
    """Build one reproducible instance of the named family.

    Every draw comes from a :class:`random.Random` seeded with the family's
    offset plus ``seed``, so a failing test id reproduces its instance.

    Args:
        family: One of the keys of :data:`FAMILY_SEEDS`.
        seed: Which draw within that family.

    Returns:
        A ``(weights, values, capacity)`` triple with at most 15 items and a
        capacity of at most 60.
    """
    rng = random.Random(FAMILY_SEEDS[family] + seed)

    if family == "coarse":
        count = rng.randint(1, 8)
        weights = [rng.randint(10, 40) for _ in range(count)]
        capacity = rng.randint(20, 60)
    elif family == "fine":
        count = rng.randint(1, 12)
        weights = [rng.randint(1, 8) for _ in range(count)]
        capacity = rng.randint(0, 30)
    elif family == "tight":
        count = rng.randint(1, 12)
        weights = [rng.randint(1, 9) for _ in range(count)]
        capacity = min(max(sum(weights) - rng.randint(1, 5), 0), 60)
    elif family == "roomy":
        count = rng.randint(0, 10)
        weights = [rng.randint(1, 5) for _ in range(count)]
        capacity = sum(weights) + rng.randint(0, 5)
    elif family == "with_zeros":
        count = rng.randint(1, 12)
        weights = [rng.choice((0, 0, rng.randint(1, 7))) for _ in range(count)]
        capacity = rng.randint(0, 20)
    else:
        count = rng.randint(13, 15)
        weights = [rng.randint(1, 20) for _ in range(count)]
        capacity = rng.randint(30, 60)

    values = [rng.randint(0, 30) for _ in range(count)]
    return weights, values, capacity


#: Every ``(family, seed)`` pair: n <= 15, W <= 60. Compared with Week 5.
BATTERY = [
    pytest.param(family, seed, id=f"{family}-seed{seed}")
    for family in sorted(FAMILY_SEEDS)
    for seed in SEEDS
]

#: The subset of :data:`BATTERY` small enough for the 2^n oracle.
ORACLE_BATTERY = [
    pytest.param(family, seed, id=f"{family}-seed{seed}")
    for family in sorted(FAMILY_SEEDS)
    for seed in SEEDS
    if len(make_instance(family, seed)[0]) <= ORACLE_MAX_ITEMS
]


def make_unbounded_instance(seed: int) -> Instance:
    """Build one small instance with positive weights, for the unbounded oracle.

    At most four items of weight 1 to 9 in a bag of at most 20, so that
    taking an item more than once is often both possible and profitable,
    and the multiset enumeration stays cheap.

    Args:
        seed: Which draw.

    Returns:
        A ``(weights, values, capacity)`` triple with every weight positive.
    """
    rng = random.Random(67000 + seed)
    count = rng.randint(1, 4)
    weights = [rng.randint(1, 9) for _ in range(count)]
    values = [rng.randint(0, 20) for _ in range(count)]
    capacity = rng.randint(0, 20)
    return weights, values, capacity


UNBOUNDED_SEEDS: Tuple[int, ...] = tuple(range(16))
UNBOUNDED_BATTERY = [
    pytest.param(seed, id=f"unbounded-seed{seed}") for seed in UNBOUNDED_SEEDS
]


# ----------------------------------------------------------------------
# The hand-computed case
# ----------------------------------------------------------------------
class TestHandComputedAnchor:
    """The Week 5 assignment instance, worked on paper before any code ran.

    Four items, capacity 7::

        index:   0    1    2    3
        weight:  1    3    4    5
        value:   1    4    5    7

    ``{1, 2}`` weighs 3 + 4 = 7 and is worth 4 + 5 = **9**. The tempting
    item 3 (value 7) leaves room 2, where only item 0 fits, for 8. No three
    items fit, since the three lightest already weigh 1 + 3 + 4 = 8 > 7. So
    the optimum is 9, achieved by indices 1 and 2, filling the bag exactly.
    """

    WEIGHTS: List[int] = [1, 3, 4, 5]
    VALUES: List[int] = [1, 4, 5, 7]
    CAPACITY: int = 7
    OPTIMUM: int = 9
    CHOSEN: List[int] = [1, 2]

    def test_the_oracle_agrees_with_the_paper_answer(self) -> None:
        """Sanity check on the oracle itself, before anything is compared to it."""
        assert brute_force_01(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_every_one_row_solver_returns_the_paper_answer(
        self, solver: Solver
    ) -> None:
        assert solver(self.WEIGHTS, self.VALUES, self.CAPACITY) == self.OPTIMUM

    def test_the_week_5_baselines_return_the_same_answer(self) -> None:
        assert knapsack_tab(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9
        assert knapsack_tab_rolling(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9

    def test_the_recovered_items_are_the_two_middle_ones(self) -> None:
        assert knapsack_space_optimized_with_items(
            self.WEIGHTS, self.VALUES, self.CAPACITY
        ) == (self.OPTIMUM, self.CHOSEN)

    def test_the_recovered_items_fill_the_bag_exactly(self) -> None:
        _best, chosen = knapsack_space_optimized_with_items(
            self.WEIGHTS, self.VALUES, self.CAPACITY
        )
        assert sum(self.WEIGHTS[index] for index in chosen) == self.CAPACITY

    def test_the_ascending_scan_also_gets_nine_here(self) -> None:
        """On this instance no reuse beats 9, so the wrong scan is not exposed.

        Unbounded, by hand: item 0 seven times is 7; item 1 twice plus item 0
        is 4 + 4 + 1 = 9; items 1 and 2 are 9; item 3 plus item 0 twice is
        7 + 1 + 1 = 9. Nothing reaches 10 (best value per unit weight is
        7 / 5 = 1.4, and 1.4 * 7 = 9.8), so both scans say 9. This is why the
        proof needs its own counterexample below.
        """
        assert brute_force_unbounded(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9
        assert knapsack_ascending_scan(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_a_tuple_instance_is_accepted(self, solver: Solver) -> None:
        assert solver(tuple(self.WEIGHTS), tuple(self.VALUES), 7) == self.OPTIMUM


# ----------------------------------------------------------------------
# The proof: scan direction decides which problem is solved
# ----------------------------------------------------------------------
class TestScanDirection:
    """Downward is 0/1 knapsack; upward is unbounded knapsack.

    The counterexample, weights ``[2, 3]``, values ``[3, 4]``, capacity 6,
    traced by hand. The row starts as ``[0, 0, 0, 0, 0, 0, 0]``.

    DOWNWARD (the module's solver). Item 0 (w 2, v 3), ``c`` from 6 to 2:
    each ``dp[c - 2]`` is still 0 when read, so every ``c >= 2`` becomes 3
    and the row is ``[0, 0, 3, 3, 3, 3, 3]``. Item 1 (w 3, v 4), ``c`` from
    6 to 3: ``dp[6] = max(3, dp[3] + 4) = 7``, ``dp[5] = max(3, dp[2] + 4)
    = 7``, ``dp[4] = max(3, dp[1] + 4) = 4``, ``dp[3] = max(3, dp[0] + 4)
    = 4``. Answer ``dp[6] = 7``: both items, each once.

    UPWARD (the deliberate bug). Item 0, ``c`` from 2 to 6: ``dp[2] = 3``,
    then ``dp[4] = dp[2] + 3 = 6`` reads the 3 item 0 just wrote, and
    ``dp[6] = dp[4] + 3 = 9``. The row is ``[0, 0, 3, 3, 6, 6, 9]``, so
    item 0 has been taken three times. Item 1 cannot improve ``dp[6]``:
    ``dp[3] + 4 = 8 < 9``. Answer 9.

    By enumeration, the 0/1 subsets are worth 0, 3, 4 and 3 + 4 = 7, and
    the unbounded multisets that fit include three copies of item 0 for
    9, two of item 1 for 8, and one of each for 7. So 7 and 9 are the
    right answers to two different problems.
    """

    WEIGHTS: List[int] = [2, 3]
    VALUES: List[int] = [3, 4]
    CAPACITY: int = 6

    def test_ascending_scan_reuses_items_which_is_why_the_scan_runs_downward(
        self,
    ) -> None:
        """THE PROOF TEST: the one-line change of direction changes the answer."""
        descending = knapsack_space_optimized(self.WEIGHTS, self.VALUES, self.CAPACITY)
        ascending = knapsack_ascending_scan(self.WEIGHTS, self.VALUES, self.CAPACITY)

        assert descending == 7  # both items, once each: 3 + 4
        assert ascending == 9  # the weight-2 item three times: 3 * 3
        assert ascending > descending

        # Each answer is the right answer to its own problem.
        assert brute_force_01(self.WEIGHTS, self.VALUES, self.CAPACITY) == 7
        assert brute_force_unbounded(self.WEIGHTS, self.VALUES, self.CAPACITY) == 9

    def test_the_downward_answer_is_both_items_once(self) -> None:
        assert knapsack_space_optimized_with_items(
            self.WEIGHTS, self.VALUES, self.CAPACITY
        ) == (7, [0, 1])

    def test_the_week_5_tables_side_with_the_downward_scan(self) -> None:
        assert knapsack_tab(self.WEIGHTS, self.VALUES, self.CAPACITY) == 7
        assert knapsack_tab_rolling(self.WEIGHTS, self.VALUES, self.CAPACITY) == 7

    @pytest.mark.parametrize(
        "weight, value, capacity, copies",
        [
            (2, 3, 6, 3),
            (1, 5, 4, 4),
            (3, 7, 9, 3),
            (4, 2, 15, 3),
        ],
        ids=["w2_in_6", "w1_in_4", "w3_in_9", "w4_in_15"],
    )
    def test_a_single_item_is_taken_once_down_and_floor_w_over_c_times_up(
        self, weight: int, value: int, capacity: int, copies: int
    ) -> None:
        """One item makes the reuse plain: ``capacity // weight`` copies upward."""
        assert knapsack_space_optimized([weight], [value], capacity) == value
        assert knapsack_ascending_scan([weight], [value], capacity) == copies * value

    @pytest.mark.parametrize(
        "instance, expected",
        [
            (([4, 5], [5, 7], 7), 7),
            (([8], [100], 7), 0),
            (([7], [5], 7), 5),
            (([4, 4], [9, 9], 7), 9),
        ],
        ids=["no_room_for_a_copy", "nothing_fits", "exact_fit", "two_halves"],
    )
    def test_where_no_copy_fits_the_two_scans_agree(
        self, instance: Instance, expected: int
    ) -> None:
        """The bug is silent: below twice the lightest weight it cannot show."""
        weights, values, capacity = instance
        assert knapsack_space_optimized(weights, values, capacity) == expected
        assert knapsack_ascending_scan(weights, values, capacity) == expected

    @pytest.mark.parametrize("seed", UNBOUNDED_BATTERY)
    def test_the_ascending_scan_is_the_unbounded_recurrence(self, seed: int) -> None:
        weights, values, capacity = make_unbounded_instance(seed)
        assert knapsack_ascending_scan(weights, values, capacity) == (
            brute_force_unbounded(weights, values, capacity)
        )

    @pytest.mark.parametrize("seed", UNBOUNDED_BATTERY)
    def test_the_descending_scan_is_the_zero_one_recurrence(self, seed: int) -> None:
        weights, values, capacity = make_unbounded_instance(seed)
        assert knapsack_space_optimized(weights, values, capacity) == brute_force_01(
            weights, values, capacity
        )

    def test_the_unbounded_battery_actually_provokes_reuse(self) -> None:
        """Guard against a vacuous battery: the two scans must differ somewhere."""
        differing = 0
        for seed in UNBOUNDED_SEEDS:
            weights, values, capacity = make_unbounded_instance(seed)
            if knapsack_ascending_scan(weights, values, capacity) != (
                knapsack_space_optimized(weights, values, capacity)
            ):
                differing += 1
        assert differing >= 3

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_the_ascending_scan_is_never_below_the_zero_one_answer(
        self, family: str, seed: int
    ) -> None:
        """Every 0/1 packing is also an unbounded packing, so upward >= downward.

        This holds on the zero-weight families too: a free item is read
        from the not-yet-updated cell in both directions, so both add it
        exactly once.
        """
        weights, values, capacity = make_instance(family, seed)
        assert knapsack_ascending_scan(weights, values, capacity) >= (
            knapsack_space_optimized(weights, values, capacity)
        )

    def test_a_free_item_is_added_once_not_infinitely_by_the_upward_scan(
        self,
    ) -> None:
        """Documented, not a bug: unbounded knapsack is undefined for weight 0.

        The upward loop at ``c`` reads ``dp[c - 0] = dp[c]`` before writing
        it, so the free item's value lands once per cell. The module
        docstring says not to read this as handling the case.
        """
        assert knapsack_ascending_scan([0], [3], 5) == 3
        assert knapsack_ascending_scan([0, 2], [3, 4], 4) == 3 + 4 + 4


# ----------------------------------------------------------------------
# Agreement on a seeded battery
# ----------------------------------------------------------------------
class TestAgreement:
    """The one-row solvers against brute force and both Week 5 baselines.

    Brute force is the definition of the problem and is used wherever it
    is affordable (``n <= 12``). The Week 5 baselines are independent
    implementations already verified against that same definition, and they
    extend the check to 15 items, where ``2^n`` enumeration stops being
    cheap enough to run on every commit.
    """

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    @pytest.mark.parametrize("family, seed", ORACLE_BATTERY)
    def test_matches_brute_force(
        self, family: str, seed: int, solver: Solver
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        assert solver(weights, values, capacity) == brute_force_01(
            weights, values, capacity
        )

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_matches_week_5_knapsack_tab(
        self, family: str, seed: int, solver: Solver
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        assert solver(weights, values, capacity) == knapsack_tab(
            weights, values, capacity
        )

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_matches_week_5_knapsack_tab_rolling(
        self, family: str, seed: int, solver: Solver
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        assert solver(weights, values, capacity) == knapsack_tab_rolling(
            weights, values, capacity
        )

    def test_the_battery_reaches_the_advertised_sizes(self) -> None:
        """The battery really spans up to 15 items and does hit the 2^n limit."""
        sizes = [len(make_instance(f, s)[0]) for f in FAMILY_SEEDS for s in SEEDS]
        capacities = [make_instance(f, s)[2] for f in FAMILY_SEEDS for s in SEEDS]
        assert max(sizes) == 15
        assert max(capacities) <= 60
        assert len(ORACLE_BATTERY) >= 30

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_one_instance_at_every_capacity_up_to_60(self, solver: Solver) -> None:
        """Sweeping the capacity catches an off-by-one that one value hides."""
        weights, values = [3, 4, 5, 8, 10, 2, 7], [4, 5, 8, 9, 10, 3, 9]
        for capacity in range(61):
            assert solver(weights, values, capacity) == knapsack_tab(
                weights, values, capacity
            ), capacity

    def test_the_density_greedy_choice_is_beaten(self) -> None:
        """Densities 6, 5, 4: greedy packs items 0 and 1 for 160; 1 and 2 give 220."""
        assert brute_force_01([10, 20, 30], [60, 100, 120], 50) == 220
        assert knapsack_space_optimized([10, 20, 30], [60, 100, 120], 50) == 220
        assert knapsack_space_optimized_with_items(
            [10, 20, 30], [60, 100, 120], 50
        ) == (220, [1, 2])


# ----------------------------------------------------------------------
# Recovering the items from one row plus bitsets
# ----------------------------------------------------------------------
class TestItemRecovery:
    """A number is only believable if some subset actually achieves it.

    The properties are separate tests because they fail for different
    reasons: a duplicate index means the backward walk read one bitset
    twice, an overweight total means it subtracted the wrong weight, and a
    value total that misses the optimum means the bits record the wrong
    decisions.
    """

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_indices_are_unique_in_range_and_ascending(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        _best, chosen = knapsack_space_optimized_with_items(weights, values, capacity)
        assert len(set(chosen)) == len(chosen)
        assert all(0 <= index < len(weights) for index in chosen)
        assert chosen == sorted(chosen)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_chosen_weights_fit_in_the_bag(self, family: str, seed: int) -> None:
        weights, values, capacity = make_instance(family, seed)
        _best, chosen = knapsack_space_optimized_with_items(weights, values, capacity)
        assert sum(weights[index] for index in chosen) <= capacity

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_chosen_values_total_the_reported_optimum(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        best, chosen = knapsack_space_optimized_with_items(weights, values, capacity)
        assert sum(values[index] for index in chosen) == best

    @pytest.mark.parametrize("family, seed", ORACLE_BATTERY)
    def test_the_reported_optimum_is_the_true_optimum(
        self, family: str, seed: int
    ) -> None:
        weights, values, capacity = make_instance(family, seed)
        best, _chosen = knapsack_space_optimized_with_items(weights, values, capacity)
        assert best == brute_force_01(weights, values, capacity)

    @pytest.mark.parametrize("family, seed", BATTERY)
    def test_the_same_items_as_the_week_5_full_table_walk(
        self, family: str, seed: int
    ) -> None:
        """Both walks record an item only on a strict improvement, so they match."""
        weights, values, capacity = make_instance(family, seed)
        assert knapsack_space_optimized_with_items(
            weights, values, capacity
        ) == trace_solution(weights, values, capacity)

    @pytest.mark.parametrize(
        "instance, expected",
        [
            (([], [], 5), (0, [])),
            (([8, 9], [100, 200], 7), (0, [])),
            (([7], [5], 7), (5, [0])),
            (([0, 5], [3, 9], 0), (3, [0])),
            (([1, 2], [0, 0], 5), (0, [])),
            (([0, 0], [3, 4], 0), (7, [0, 1])),
            (([1, 2, 3], [1, 2, 3], 6), (6, [0, 1, 2])),
        ],
        ids=[
            "empty",
            "nothing_fits",
            "exact_fit",
            "free_item_at_zero",
            "zero_values",
            "two_free_items",
            "everything_fits",
        ],
    )
    def test_edge_instances_recover_the_hand_checked_items(
        self, instance: Instance, expected: Tuple[int, List[int]]
    ) -> None:
        weights, values, capacity = instance
        assert knapsack_space_optimized_with_items(weights, values, capacity) == (
            expected
        )

    def test_a_tie_returns_the_earlier_item(self) -> None:
        """Two interchangeable items, room for one: the later one only ties.

        Item 1 would make ``dp[2] = dp[0] + 3 = 3``, which is not strictly
        greater than the 3 item 0 already put there, so its bit stays clear
        and the walk falls through to item 0. Same rule as Week 5.
        """
        assert knapsack_space_optimized_with_items([2, 2], [3, 3], 2) == (3, [0])
        assert knapsack_space_optimized_with_items([4, 4], [9, 9], 4) == (9, [0])

    def test_an_item_heavier_than_the_bag_is_never_named(self) -> None:
        """The heavy item gets an all-zero bitset, so the walk cannot pick it."""
        best, chosen = knapsack_space_optimized_with_items([3, 50, 4], [4, 999, 5], 7)
        assert (best, chosen) == (9, [0, 2])


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    """The instances where the recurrence has nothing to choose between.

    The deliberately wrong upward scan joins the cases where no item can be
    taken twice, because there it must give the same answer as 0/1.
    """

    @pytest.mark.parametrize("capacity", [0, 1, 7, 60])
    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_no_items_is_zero_at_any_capacity(
        self, solver: Solver, capacity: int
    ) -> None:
        assert solver([], [], capacity) == 0

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_capacity_zero_is_zero_when_every_weight_is_positive(
        self, solver: Solver
    ) -> None:
        assert solver([1, 3, 4, 5], [1, 4, 5, 7], 0) == 0

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_every_item_too_heavy_is_zero(self, solver: Solver) -> None:
        assert solver([8, 9, 10], [5, 6, 7], 7) == 0

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_a_single_item_one_unit_too_heavy_is_left_behind(
        self, solver: Solver
    ) -> None:
        """The boundary just past an exact fit, where ``weight <= c`` slips."""
        assert solver([8], [5], 7) == 0

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_zero_value_items_are_worth_nothing(self, solver: Solver) -> None:
        assert solver([1, 2, 3], [0, 0, 0], 6) == 0

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_a_single_item_that_fits_exactly_is_taken_once(
        self, solver: Solver
    ) -> None:
        assert solver([7], [5], 7) == 5
        assert solver([2], [3], 6) == 3  # room for three copies, still once

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_a_zero_weight_item_is_free_even_at_capacity_zero(
        self, solver: Solver
    ) -> None:
        assert solver([0, 5], [3, 9], 0) == 3

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_zero_weight_items_are_all_taken_once_each(self, solver: Solver) -> None:
        assert solver([0, 2, 0], [3, 9, 4], 2) == 3 + 9 + 4
        assert solver([0, 0], [3, 4], 0) == 3 + 4

    @pytest.mark.parametrize("solver", ZERO_ONE_PARAMS, ids=ZERO_ONE_IDS)
    def test_everything_fits_so_everything_is_taken(self, solver: Solver) -> None:
        weights, values = [1, 2, 3, 4], [5, 6, 7, 8]
        assert solver(weights, values, sum(weights)) == 26
        assert solver(weights, values, sum(weights) + 25) == 26

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_a_single_zero_weight_zero_value_item(self, solver: Solver) -> None:
        assert solver([0], [0], 0) == 0


# ----------------------------------------------------------------------
# The shared contract
# ----------------------------------------------------------------------
#: Malformed calls, each with one fault (the last one has two, to pin the
#: order in which they are reported). Used to check that this module raises
#: exactly what Week 5 raises.
BAD_CALLS: List[Tuple[str, Tuple[Any, Any, Any]]] = [
    ("more_weights", ([1, 2], [3], 5)),
    ("more_values", ([1], [3, 4], 5)),
    ("negative_capacity", ([1, 2], [3, 4], -1)),
    ("float_capacity", ([1, 2], [3, 4], 1.5)),
    ("whole_float_capacity", ([1, 2], [3, 4], 7.0)),
    ("str_capacity", ([1, 2], [3, 4], "7")),
    ("none_capacity", ([1, 2], [3, 4], None)),
    ("bool_capacity", ([1, 2], [3, 4], True)),
    ("negative_weight", ([1, -2], [3, 4], 5)),
    ("negative_value", ([1, 2], [3, -4], 5)),
    ("float_weight", ([1, 2.5], [3, 4], 5)),
    ("bool_weight", ([1, True], [3, 4], 5)),
    ("str_value", ([1, 2], [3, "4"], 5)),
    ("bool_value", ([1, 2], [3, False], 5)),
    ("weights_not_a_sequence", (7, [1], 5)),
    ("values_not_a_sequence", ([1], 7, 5)),
    ("capacity_type_and_length", ([1, 2], [3], 1.5)),
]
BAD_CALL_IDS: List[str] = [name for name, _call in BAD_CALLS]
BAD_CALL_PARAMS: List[Tuple[Any, Any, Any]] = [call for _name, call in BAD_CALLS]


class TestContracts:
    """Argument checking, run over every entry point at once.

    The module promises to be a drop-in replacement for Week 5, so beyond
    the usual "raises the right type" checks, every malformed call is also
    made against ``knapsack_tab`` and the two exceptions must match in type
    and message.
    """

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_more_weights_than_values_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match="same length"):
            solver([1, 2], [3], 5)

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_more_values_than_weights_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match="same length"):
            solver([1], [3, 4], 5)

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize("capacity", [-1, -60])
    def test_negative_capacity_is_a_value_error(
        self, solver: Solver, capacity: int
    ) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 0"):
            solver([1, 2], [3, 4], capacity)

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
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

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
    def test_a_bool_capacity_is_a_type_error(self, solver: Solver, flag: bool) -> None:
        """``True`` is an accident, not a capacity, even though it is an int."""
        with pytest.raises(TypeError, match="got bool"):
            solver([1, 2], [3, 4], flag)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_a_negative_weight_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match=r"weights\[1\] must be >= 0"):
            solver([1, -2], [3, 4], 5)

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_a_negative_value_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match=r"values\[1\] must be >= 0"):
            solver([1, 2], [3, -4], 5)

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
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

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize(
        "entry",
        [2.5, 4.0, "2", None, False],
        ids=["float", "whole_float", "str", "none", "bool"],
    )
    def test_a_non_int_value_is_a_type_error(
        self, solver: Solver, entry: object
    ) -> None:
        with pytest.raises(TypeError, match=r"values\[1\] must be an int"):
            solver([1, 2], [3, entry], 5)  # type: ignore[list-item]

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_a_non_sequence_is_a_type_error(self, solver: Solver) -> None:
        with pytest.raises(TypeError, match="weights must be a sequence of ints"):
            solver(7, [1], 5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="values must be a sequence of ints"):
            solver([1], 7, 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_the_capacity_type_is_checked_before_the_lengths(
        self, solver: Solver
    ) -> None:
        with pytest.raises(TypeError, match="capacity must be an int"):
            solver([1, 2], [3], 1.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize("call", BAD_CALL_PARAMS, ids=BAD_CALL_IDS)
    def test_every_error_is_exactly_the_week_5_error(
        self, solver: Solver, call: Tuple[Any, Any, Any]
    ) -> None:
        """Same exception type and same message as ``knapsack_tab``."""
        with pytest.raises((TypeError, ValueError)) as week5:
            knapsack_tab(*call)
        with pytest.raises((TypeError, ValueError)) as week6:
            solver(*call)
        assert type(week6.value) is type(week5.value)
        assert str(week6.value) == str(week5.value)

    @pytest.mark.parametrize("solver", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_the_caller_s_lists_are_never_modified(self, solver: Solver) -> None:
        """Validation copies, so a solved instance can be solved again."""
        weights, values = [2, 3, 4, 5], [3, 4, 5, 7]
        solver(weights, values, 9)
        assert weights == [2, 3, 4, 5]
        assert values == [3, 4, 5, 7]


# ----------------------------------------------------------------------
# The measurement against the Week 5 full table
# ----------------------------------------------------------------------
#: Every key :func:`compare_with_standard` documents, and no others.
COMPARE_KEYS = {
    "n",
    "capacity",
    "standard_value",
    "optimized_value",
    "agree",
    "standard_time",
    "optimized_time",
    "standard_peak_kib",
    "optimized_peak_kib",
    "memory_ratio",
    "time_ratio",
}

#: The keys of each timing dictionary, as returned by ``time_call``.
TIME_KEYS = {"mean", "std", "min", "max", "runs"}

#: Timed runs per solver in the shared comparison. Two keeps the file fast;
#: the figures are never asserted, only their structure and consistency.
COMPARE_REPEAT = 2


def make_large_instance() -> Instance:
    """Build the fixed n = 50, W = 500 instance used for the measurement.

    Returns:
        A ``(weights, values, capacity)`` triple with 50 items of weight 1
        to 60 and value 1 to 100, in a bag of 500.
    """
    rng = random.Random(5300)
    weights = [rng.randint(1, 60) for _ in range(50)]
    values = [rng.randint(1, 100) for _ in range(50)]
    return weights, values, 500


@pytest.fixture(scope="module")
def large_instance() -> Instance:
    """The n = 50, W = 500 instance, built once for the module."""
    return make_large_instance()


@pytest.fixture(scope="module")
def large_comparison(large_instance: Instance) -> Dict[str, Any]:
    """One run of :func:`compare_with_standard`, shared by every test below.

    Running it once keeps the file fast; the tests only read the result.
    """
    weights, values, capacity = large_instance
    return compare_with_standard(weights, values, capacity, repeat=COMPARE_REPEAT)


class TestCompareWithStandard:
    """The space comparison at n = 50, W = 500, checked as a measurement.

    The full table holds 51 rows of 501 cells and the one-row solver holds
    one row of 501, so the traced peaks must differ in the right direction.
    No absolute KiB or millisecond figure is asserted: those belong to the
    machine, not to the algorithm.
    """

    def test_every_documented_key_is_present_and_no_other(
        self, large_comparison: Dict[str, Any]
    ) -> None:
        assert set(large_comparison) == COMPARE_KEYS

    def test_the_two_values_agree(self, large_comparison: Dict[str, Any]) -> None:
        assert large_comparison["agree"] is True
        assert large_comparison["standard_value"] == (
            large_comparison["optimized_value"]
        )

    def test_both_values_equal_week_5_knapsack_tab(
        self, large_comparison: Dict[str, Any], large_instance: Instance
    ) -> None:
        expected = knapsack_tab(*large_instance)
        assert large_comparison["standard_value"] == expected
        assert large_comparison["optimized_value"] == expected

    def test_both_values_equal_the_one_row_solver_called_directly(
        self, large_comparison: Dict[str, Any], large_instance: Instance
    ) -> None:
        assert large_comparison["optimized_value"] == knapsack_space_optimized(
            *large_instance
        )

    def test_the_instance_size_is_reported(
        self, large_comparison: Dict[str, Any]
    ) -> None:
        assert large_comparison["n"] == 50
        assert large_comparison["capacity"] == 500

    def test_one_row_peaks_below_the_full_table(
        self, large_comparison: Dict[str, Any]
    ) -> None:
        peak_1d = large_comparison["optimized_peak_kib"]
        peak_2d = large_comparison["standard_peak_kib"]
        assert 0 < peak_1d < peak_2d

    def test_the_memory_ratio_is_the_ratio_of_the_two_peaks(
        self, large_comparison: Dict[str, Any]
    ) -> None:
        """Internal consistency, and above 1 because the full table is bigger."""
        ratio = large_comparison["memory_ratio"]
        assert ratio == pytest.approx(
            large_comparison["standard_peak_kib"]
            / large_comparison["optimized_peak_kib"]
        )
        assert ratio > 1

    @pytest.mark.parametrize("label", ["standard", "optimized"])
    def test_each_timing_is_a_time_call_summary(
        self, large_comparison: Dict[str, Any], label: str
    ) -> None:
        timing = large_comparison[f"{label}_time"]
        assert set(timing) == TIME_KEYS
        assert timing["runs"] == COMPARE_REPEAT
        assert 0 <= timing["min"] <= timing["mean"] <= timing["max"]
        assert timing["std"] >= 0

    def test_the_time_ratio_is_the_ratio_of_the_two_means(
        self, large_comparison: Dict[str, Any]
    ) -> None:
        """Consistency only. Which solver is faster is a finding, not a contract."""
        standard_mean = large_comparison["standard_time"]["mean"]
        optimized_mean = large_comparison["optimized_time"]["mean"]
        assert optimized_mean > 0
        assert large_comparison["time_ratio"] == pytest.approx(
            standard_mean / optimized_mean
        )

    def test_the_anchor_through_the_comparison(self) -> None:
        """The hand-computed instance, end to end: both sides report 9."""
        result = compare_with_standard([1, 3, 4, 5], [1, 4, 5, 7], 7, repeat=1)
        assert (result["standard_value"], result["optimized_value"]) == (9, 9)
        assert result["agree"] is True
        assert (result["n"], result["capacity"]) == (4, 7)
        assert result["standard_time"]["runs"] == 1

    def test_the_caller_s_lists_are_never_modified(self) -> None:
        weights, values = [2, 3, 4, 5], [3, 4, 5, 7]
        compare_with_standard(weights, values, 9, repeat=1)
        assert weights == [2, 3, 4, 5]
        assert values == [3, 4, 5, 7]

    @pytest.mark.parametrize("repeat", [0, -1])
    def test_a_repeat_below_one_is_a_value_error(self, repeat: int) -> None:
        with pytest.raises(ValueError, match="repeat must be >= 1"):
            compare_with_standard([1], [1], 1, repeat=repeat)

    @pytest.mark.parametrize("repeat", [True, 2.0], ids=["bool", "float"])
    def test_a_non_int_repeat_is_a_type_error(self, repeat: object) -> None:
        with pytest.raises(TypeError, match="repeat must be an int"):
            compare_with_standard([1], [1], 1, repeat=repeat)  # type: ignore[arg-type]

    @pytest.mark.parametrize("call", BAD_CALL_PARAMS, ids=BAD_CALL_IDS)
    def test_a_malformed_instance_is_rejected_like_week_5(
        self, call: Tuple[Any, Any, Any]
    ) -> None:
        """The instance is validated before anything is timed."""
        with pytest.raises((TypeError, ValueError)) as week5:
            knapsack_tab(*call)
        with pytest.raises((TypeError, ValueError)) as week6:
            compare_with_standard(*call, repeat=1)
        assert type(week6.value) is type(week5.value)
        assert str(week6.value) == str(week5.value)


# ----------------------------------------------------------------------
# Printing the comparison
# ----------------------------------------------------------------------
#: A fixed result, so the printed figures can be checked by hand.
FIXED_RESULT: Dict[str, Any] = {
    "n": 4,
    "capacity": 7,
    "standard_value": 9,
    "optimized_value": 9,
    "agree": True,
    "standard_time": {"mean": 0.000012},
    "optimized_time": {"mean": 0.000008},
    "standard_peak_kib": 2.5,
    "optimized_peak_kib": 0.5,
    "memory_ratio": 5.0,
    "time_ratio": 1.5,
}


class TestPrintComparison:
    """The table the report quotes, on a real result and on a fixed one."""

    def test_a_real_result_names_both_variants(
        self, large_comparison: Dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_comparison(large_comparison)
        out = capsys.readouterr().out
        assert out.strip()
        assert "standard" in out
        assert "optimized" in out
        assert "n=50" in out
        assert "W=500" in out
        assert "values agree: True" in out

    def test_a_real_result_has_one_row_per_variant_with_its_value(
        self, large_comparison: Dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_comparison(large_comparison)
        rows = {
            line.split()[0]: line.split()
            for line in capsys.readouterr().out.splitlines()
            if line.split() and line.split()[0] in ("standard", "optimized")
        }
        assert set(rows) == {"standard", "optimized"}
        for label in ("standard", "optimized"):
            assert rows[label][1] == str(large_comparison[f"{label}_value"])

    def test_the_fixed_result_prints_the_hand_checked_figures(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """0.000012 s is 0.012 ms, printed to four places as 0.0120, and so on."""
        print_comparison(FIXED_RESULT)
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 6
        assert lines[0] == "0/1 knapsack, n=4, W=7 (values agree: True)"
        assert lines[1].split() == [
            "variant", "value", "mean", "time", "(ms)", "peak", "KiB"
        ]
        assert lines[2].split() == ["standard", "9", "0.0120", "2.50"]
        assert lines[3].split() == ["optimized", "9", "0.0080", "0.50"]
        assert "1.50x" in lines[4]
        assert "5.00x" in lines[5]

    def test_a_disagreement_is_printed_not_hidden(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_comparison({**FIXED_RESULT, "optimized_value": 8, "agree": False})
        out = capsys.readouterr().out
        assert "values agree: False" in out

    def test_it_returns_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert print_comparison(FIXED_RESULT) is None
        capsys.readouterr()

    @pytest.mark.parametrize("missing", ["time_ratio", "optimized_peak_kib", "n"])
    def test_a_missing_key_is_a_key_error(
        self, missing: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken = {key: value for key, value in FIXED_RESULT.items() if key != missing}
        with pytest.raises(KeyError):
            print_comparison(broken)
        capsys.readouterr()
