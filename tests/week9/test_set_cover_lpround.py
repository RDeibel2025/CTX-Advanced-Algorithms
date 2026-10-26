"""Tests for set cover: the IP, its LP relaxation, greedy and the two roundings.

Every claim rests on a witness that shares no mechanism with the code under
test:

* **The spec anchors (WEEK9_BUILD_SPEC.md section 10.2), traced by hand.**
  The greedy trap: universe 1..14 with unit costs and sets
  S0 = odds, S1 = evens, S2 = {1, 2}, S3 = {3, 4, 5, 6},
  S4 = {7, .., 14}.

  - Greedy (cost per newly covered element): S4 covers 8 (1/8), so it goes
    first; then S3 covers 4 new (1/4) against 3 for S0 ({1, 3, 5}) and S1
    ({2, 4, 6}); then S2 covers 2 new ({1, 2}, 1/2) against 1 for S0 and
    S1. Picks [4, 3, 2], cost 3.
  - OPT = 2: S0 and S1 cover everything, and no single set does.
  - LP* = 2.0 with the unique optimum x = (1, 1, 0, 0, 0). Lower bound: the
    dual y_1 = y_4 = 1 (elements 1 and 4) puts at most 1 on each set (S0
    holds 1, S1 holds 4, S2 holds 1, S3 holds 4, S4 neither), so LP* >= 2,
    and x = (1, 1, 0, 0, 0) costs 2. Uniqueness by complementary slackness:
    S4 is slack in that dual, so x_4 = 0, and then elements 7 and 8 force
    x_0 = x_1 = 1, which already costs 2, so x_2 = x_3 = 0.
  - f = 2: every element lies in exactly one half and exactly one block.
  - Deterministic rounding keeps x_S >= 1/2: [0, 1], cost 2.

* **Brute force over every sub-collection** with :mod:`itertools` for the
  30 random instances (at most 9 sets, so at most 512 collections), which
  knows nothing about LPs, MILPs or greed.
* **The textbook bounds, computed here**: H(d) = 1 + 1/2 + .. + 1/d for the
  largest set size d, the frequency f counted directly from the sets, and
  (2 ln n + 2) for randomized rounding.
* **Hand-built LP vectors** passed through the ``lp=`` keyword, whose
  rounding is traced by hand, to show the keyword is really used.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import math
from typing import List, Sequence, Set

import pytest

from src.approximation.set_cover_lpround import (
    frequency,
    random_instance,
    set_cover_deterministic_rounding,
    set_cover_exact,
    set_cover_greedy,
    set_cover_ip,
    set_cover_lp,
    set_cover_randomized_rounding,
)
from src.utils.random_utils import derive_seed

TOL = 1e-9

TRAP_UNIVERSE = set(range(1, 15))
TRAP_SUBSETS = [
    {1, 3, 5, 7, 9, 11, 13},
    {2, 4, 6, 8, 10, 12, 14},
    {1, 2},
    {3, 4, 5, 6},
    {7, 8, 9, 10, 11, 12, 13, 14},
]

#: {1, 2, 3} covered by the three pairs: LP* = 1.5 (every x_S = 1/2), OPT = 2.
TRI_UNIVERSE = {1, 2, 3}
TRI_SUBSETS = [{1, 2}, {2, 3}, {1, 3}]


def is_cover(universe: Set[int], subsets: Sequence[Set[int]], chosen: Sequence[int]) -> bool:
    """Return True when the chosen subsets cover the universe."""
    covered: Set[int] = set()
    for index in chosen:
        covered |= subsets[index]
    return universe <= covered


def brute_force_opt(universe: Set[int], subsets: Sequence[Set[int]], costs: Sequence[float]) -> float:
    """Return the minimum cover cost by trying every sub-collection."""
    best = math.inf
    for r in range(1, len(subsets) + 1):
        for combo in itertools.combinations(range(len(subsets)), r):
            if is_cover(universe, subsets, combo):
                best = min(best, math.fsum(costs[i] for i in combo))
    return best


def harmonic(d: int) -> float:
    """Return H(d) = 1 + 1/2 + ... + 1/d."""
    return sum(1.0 / k for k in range(1, d + 1))


def max_frequency(universe: Set[int], subsets: Sequence[Set[int]]) -> int:
    """Return the largest number of subsets holding one element."""
    return max(sum(1 for s in subsets if e in s) for e in universe)


def small_instances() -> List[tuple]:
    """Return the 30 seeded random instances used by the bound checks.

    Density 0.4 with at most 9 sets (brute force stays at 512 collections);
    odd repetitions drop the random costs for unit costs, which makes
    fractional LP optima (an integrality gap) more common, so the rounding
    bounds are exercised on instances where rounding is not trivially exact.
    """
    instances = []
    for rep in range(30):
        n_elements = 6 + rep % 7
        n_sets = 4 + rep % 6
        seed = derive_seed(9, "set-cover-test", rep)
        universe, subsets, costs = random_instance(n_elements, n_sets, 0.4, seed)
        instances.append((universe, subsets, None if rep % 2 else costs))
    return instances


def cost_list(subsets: Sequence[Set[int]], costs) -> List[float]:
    """Return the explicit costs, reading ``None`` as unit costs."""
    return [1.0] * len(subsets) if costs is None else list(costs)


INSTANCES = small_instances()


class TestGreedyTrap:
    """Spec 10.2 anchors on the greedy trap."""

    def test_greedy_picks_blocks_in_order(self):
        chosen, cost = set_cover_greedy(TRAP_UNIVERSE, TRAP_SUBSETS)
        assert chosen == [4, 3, 2]
        assert cost == pytest.approx(3.0)

    def test_exact_is_the_two_halves(self):
        chosen, cost = set_cover_exact(TRAP_UNIVERSE, TRAP_SUBSETS)
        assert chosen == [0, 1]
        assert cost == pytest.approx(2.0)

    def test_lp_value_and_solution(self):
        lp_value, x = set_cover_lp(TRAP_UNIVERSE, TRAP_SUBSETS)
        assert lp_value == pytest.approx(2.0, abs=TOL)
        assert x == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0], abs=1e-7)

    def test_frequency_is_two(self):
        assert frequency(TRAP_UNIVERSE, TRAP_SUBSETS) == 2

    def test_deterministic_rounding(self):
        chosen, cost, lp_value = set_cover_deterministic_rounding(TRAP_UNIVERSE, TRAP_SUBSETS)
        assert chosen == [0, 1]
        assert cost == pytest.approx(2.0)
        assert lp_value == pytest.approx(2.0, abs=TOL)

    def test_integer_program_data(self):
        ip = set_cover_ip(TRAP_UNIVERSE, TRAP_SUBSETS)
        assert ip["c"] == [1.0] * 5
        assert ip["b_ub"] == [-1.0] * 14
        assert ip["integrality"] == [1] * 5
        assert len(ip["A_ub"]) == 14
        # Row of element 1 (sorted first): in S0 and S2.
        assert ip["A_ub"][0] == [-1.0, 0.0, -1.0, 0.0, 0.0]
        # Row of element 14 (sorted last): in S1 and S4.
        assert ip["A_ub"][13] == [0.0, -1.0, 0.0, 0.0, -1.0]


class TestRandomizedRounding:
    """Randomized rounding always covers and stays within its expected bound."""

    def test_trap_fifty_seeds(self):
        costs = []
        for seed in range(50):
            chosen, cost, lp_value = set_cover_randomized_rounding(
                TRAP_UNIVERSE, TRAP_SUBSETS, seed=seed
            )
            assert is_cover(TRAP_UNIVERSE, TRAP_SUBSETS, chosen)
            assert lp_value == pytest.approx(2.0, abs=TOL)
            costs.append(cost)
        mean = sum(costs) / len(costs)
        assert mean <= (2 * math.log(14) + 2) * 2.0

    def test_fractional_lp_fifty_seeds(self):
        costs = []
        for seed in range(50):
            chosen, cost, lp_value = set_cover_randomized_rounding(
                TRI_UNIVERSE, TRI_SUBSETS, seed=seed
            )
            assert is_cover(TRI_UNIVERSE, TRI_SUBSETS, chosen)
            assert lp_value == pytest.approx(1.5, abs=TOL)
            costs.append(cost)
        assert sum(costs) / len(costs) <= (2 * math.log(3) + 2) * 1.5

    def test_same_seed_same_result(self):
        a = set_cover_randomized_rounding(TRI_UNIVERSE, TRI_SUBSETS, seed=11)
        b = set_cover_randomized_rounding(TRI_UNIVERSE, TRI_SUBSETS, seed=11)
        assert a == b

    def test_default_rounds_is_ceil_two_ln_n_plus_one(self):
        # n = 3: ceil(2 ln 3) + 1 = ceil(2.197) + 1 = 4 rounds.
        for seed in range(20):
            default = set_cover_randomized_rounding(
                TRI_UNIVERSE, TRI_SUBSETS, seed=seed, return_fixups=True
            )
            explicit = set_cover_randomized_rounding(
                TRI_UNIVERSE, TRI_SUBSETS, seed=seed, rounds=4, return_fixups=True
            )
            assert default == explicit


class TestRandomInstances:
    """30 seeded instances: covers, LP <= OPT <= every method, proven bounds."""

    @pytest.mark.parametrize("index", range(len(INSTANCES)))
    def test_methods_and_bounds(self, index):
        universe, subsets, costs = INSTANCES[index]
        unit_or_given = cost_list(subsets, costs)
        opt = brute_force_opt(universe, subsets, unit_or_given)
        f = max_frequency(universe, subsets)
        d = max(len(s) for s in subsets)

        lp_value, x = set_cover_lp(universe, subsets, costs)
        exact, exact_cost = set_cover_exact(universe, subsets, costs)
        greedy, greedy_cost = set_cover_greedy(universe, subsets, costs)
        det, det_cost, det_lp = set_cover_deterministic_rounding(universe, subsets, costs)
        rnd, rnd_cost, rnd_lp = set_cover_randomized_rounding(
            universe, subsets, costs, seed=derive_seed(9, "rr", index)
        )

        for chosen, cost in ((exact, exact_cost), (greedy, greedy_cost),
                             (det, det_cost), (rnd, rnd_cost)):
            assert is_cover(universe, subsets, chosen)
            assert cost == pytest.approx(math.fsum(unit_or_given[i] for i in chosen))
            assert exact_cost <= cost + TOL

        assert exact_cost == pytest.approx(opt)
        assert len(x) == len(subsets)
        assert lp_value <= exact_cost + TOL
        assert det_lp == pytest.approx(lp_value) and rnd_lp == pytest.approx(lp_value)
        assert frequency(universe, subsets) == f
        assert greedy_cost <= harmonic(d) * exact_cost + TOL
        assert det_cost <= f * lp_value + TOL

    def test_family_has_an_integrality_gap(self):
        # Guard against a vacuous family: some instance must have LP* below
        # the brute-force optimum, so rounding is not trivially exact.
        gaps = 0
        for universe, subsets, costs in INSTANCES:
            lp_value, _ = set_cover_lp(universe, subsets, costs)
            gaps += lp_value < brute_force_opt(universe, subsets, cost_list(subsets, costs)) - 1e-6
        assert gaps >= 1

    def test_random_instance_shape(self):
        universe, subsets, costs = random_instance(12, 5, 0.25, seed=3)
        assert universe == set(range(1, 13))
        assert len(subsets) == 5 and len(costs) == 5
        assert set().union(*subsets) == universe
        assert all(subsets)
        assert all(1.0 <= c <= 10.0 for c in costs)
        assert random_instance(12, 5, 0.25, seed=3) == (universe, subsets, costs)

    def test_random_instance_repair_and_full_density(self):
        universe, subsets, _ = random_instance(8, 6, 0.0, seed=1)
        assert set().union(*subsets) == universe
        assert all(subsets)
        _, full, _ = random_instance(8, 6, 1.0, seed=1)
        assert full == [set(range(1, 9))] * 6


class TestValidation:
    """ValueError when no cover exists; empty universe gives empty answers."""

    @pytest.mark.parametrize("method", [
        set_cover_ip, set_cover_lp, set_cover_exact, set_cover_greedy,
        set_cover_deterministic_rounding, set_cover_randomized_rounding,
    ])
    def test_element_in_no_subset(self, method):
        with pytest.raises(ValueError):
            method({1, 2, 3}, [{1, 2}, {1}])

    def test_frequency_element_in_no_subset(self):
        with pytest.raises(ValueError):
            frequency({1, 2, 3}, [{1, 2}])

    def test_bad_costs(self):
        with pytest.raises(ValueError):
            set_cover_greedy({1}, [{1}], [1.0, 2.0])
        with pytest.raises(ValueError):
            set_cover_greedy({1}, [{1}], [-1.0])

    def test_empty_universe(self):
        assert set_cover_exact(set(), [{1}]) == ([], 0.0)
        assert set_cover_greedy(set(), [{1}]) == ([], 0.0)
        assert set_cover_lp(set(), [{1}]) == (0.0, [0.0])
        assert set_cover_deterministic_rounding(set(), [{1}]) == ([], 0.0, 0.0)
        assert set_cover_randomized_rounding(set(), [{1}], seed=1) == ([], 0.0, 0.0)


class TestLpKeyword:
    """A precomputed LP gives the same answer as solving inside, and is used."""

    def test_same_result_as_solving_inside(self):
        for universe, subsets, costs in INSTANCES[:10] + [(TRI_UNIVERSE, TRI_SUBSETS, None)]:
            lp = set_cover_lp(universe, subsets, costs)
            assert set_cover_deterministic_rounding(universe, subsets, costs, lp=lp) == \
                set_cover_deterministic_rounding(universe, subsets, costs)
            for seed in range(5):
                assert set_cover_randomized_rounding(
                    universe, subsets, costs, seed=seed, lp=lp, return_fixups=True
                ) == set_cover_randomized_rounding(
                    universe, subsets, costs, seed=seed, return_fixups=True
                )

    def test_hand_built_lp_is_honoured(self):
        # x = 1 on S2, S3, S4 and 0 on the halves: threshold 1/2 keeps
        # S2, S3, S4 (which cover 1..14), cost 3; randomized rounding with
        # 0/1 probabilities picks exactly those sets in every round.
        fake = (3.0, [0.0, 0.0, 1.0, 1.0, 1.0])
        assert set_cover_deterministic_rounding(TRAP_UNIVERSE, TRAP_SUBSETS, lp=fake) == \
            ([2, 3, 4], 3.0, 3.0)
        for seed in range(5):
            assert set_cover_randomized_rounding(
                TRAP_UNIVERSE, TRAP_SUBSETS, seed=seed, lp=fake, return_fixups=True
            ) == ([2, 3, 4], 3.0, 3.0, 0)

    def test_wrong_length_lp(self):
        with pytest.raises(ValueError):
            set_cover_deterministic_rounding(TRAP_UNIVERSE, TRAP_SUBSETS, lp=(2.0, [1.0, 1.0]))
        with pytest.raises(ValueError):
            set_cover_randomized_rounding(TRAP_UNIVERSE, TRAP_SUBSETS, seed=1, lp=(2.0, [1.0]))


class TestFixups:
    """return_fixups counts the elements left uncovered by the rounds."""

    def test_four_tuple_matches_three_tuple(self):
        for seed in range(10):
            with_fix = set_cover_randomized_rounding(
                TRI_UNIVERSE, TRI_SUBSETS, seed=seed, rounds=1, return_fixups=True
            )
            without = set_cover_randomized_rounding(TRI_UNIVERSE, TRI_SUBSETS, seed=seed, rounds=1)
            assert len(with_fix) == 4
            assert with_fix[:3] == without
            assert 0 <= with_fix[3] <= 3

    def test_integral_lp_needs_no_fixups(self):
        result = set_cover_randomized_rounding(TRAP_UNIVERSE, TRAP_SUBSETS, seed=3, return_fixups=True)
        assert result == ([0, 1], 2.0, pytest.approx(2.0, abs=TOL), 0)

    def test_zero_rounds_fixes_every_element(self):
        # Unit costs: each element's cheapest set is its lowest-index set,
        # which is S0 for odd elements and S1 for even ones.
        result = set_cover_randomized_rounding(
            TRAP_UNIVERSE, TRAP_SUBSETS, seed=1, rounds=0, return_fixups=True
        )
        assert result[0] == [0, 1]
        assert result[1] == 2.0
        assert result[3] == 14

    def test_zero_rounds_uses_cheapest_set(self):
        # Element 1: {1} at 1 beats {1, 2, 3} at 5; elements 2, 3: {2, 3} at 2.
        chosen, cost, _, fixups = set_cover_randomized_rounding(
            {1, 2, 3}, [{1, 2, 3}, {1}, {2, 3}], [5, 1, 2], seed=1, rounds=0, return_fixups=True
        )
        assert (chosen, cost, fixups) == ([1, 2], 3.0, 3)

    def test_one_round_sometimes_needs_fixups(self):
        # With x_S = 1/2 an element survives one round with probability 1/4.
        fixups = [
            set_cover_randomized_rounding(
                TRI_UNIVERSE, TRI_SUBSETS, seed=seed, rounds=1, return_fixups=True
            )[3]
            for seed in range(50)
        ]
        assert any(count > 0 for count in fixups)
