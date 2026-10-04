"""Tests for Held-Karp TSP, its brute-force baseline, and the bitmask helpers.

A travelling salesman answer is a number and a list, and either can be
wrong in a way that looks right. A cost that is too low is a tour that
does not exist; a cost that is too high is a tour that is not optimal; a
tour whose edges do not add up to the reported cost means the parent table
and the cost table have drifted apart. The claims here rest on witnesses
that share no mechanism with the dynamic program:

* **The assignment's hand-checked instance.** The four-city symmetric
  matrix with optimum 80. With the start fixed there are only three
  distinct cycles through four cities, and :class:`TestHandCheckedAnchor`
  prices all three on paper. Because the matrix is symmetric, the optimal
  cycle can be driven either way round, so the tests accept either
  direction and never pin one tour.
* **Exhaustive enumeration.** :func:`oracle_cost` is the definition of the
  problem written as code: every Hamiltonian cycle through the start,
  priced edge by edge with wrap-around indexing, and the minimum taken. It
  has no table, no mask and no parent pointer. Both :func:`tsp_bitmask` and
  :func:`tsp_brute_force` are held to it, and to each other, on a seeded
  battery of 120 matrices for n = 1 to 8, which includes asymmetric,
  sparse (missing edges), tie-heavy and fractional instances.
* **The tour itself.** Every returned tour is checked by
  :func:`assert_valid_tour`: it starts and ends at the start city, visits
  every city exactly once in between, and its edges sum to the reported
  cost. A cost is only believable if some tour achieves it.
* **Metamorphic relations.** Transposing the matrix reverses every cycle,
  so it cannot change the optimum; adding a constant to every edge adds n
  times that constant, because every tour uses exactly n edges; relabelling
  the cities changes nothing. None of these needs to know the answer in
  advance, which is what makes them hard to fool.

The bitmask helpers in :mod:`src.utils.bitmask_utils` are checked against
plain Python ``set`` arithmetic. :func:`members` reads a mask's set off its
binary string and :func:`to_mask` builds a mask from powers of two, so
neither shares a shift or an AND with the code under test.

Every random draw is seeded from a fixed constant and no instance exceeds
eight cities, so the brute-force oracle stays cheap and the file runs in a
few seconds on every commit.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import pytest

import src.dp_advanced.bitmask_traveling_salesman as tsp_module
from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask, tsp_brute_force
from src.utils.bitmask_utils import (
    clear_bit,
    full_mask,
    has_bit,
    iter_bits,
    mask_to_list,
    popcount,
    set_bit,
    subsets_of_size,
)
from src.utils.matrix_utils import random_weight_matrix

Matrix = List[List[float]]
Solver = Callable[..., Tuple[float, List[int]]]

INF = math.inf

#: Both public solvers. The contract tests and the agreement battery run
#: over each, because a check that holds in one and not its neighbour is
#: exactly the drift that goes unnoticed when only the headline is tested.
SOLVERS: Dict[str, Solver] = {
    "bitmask": tsp_bitmask,
    "brute_force": tsp_brute_force,
}
SOLVER_IDS: List[str] = sorted(SOLVERS)
SOLVER_PARAMS: List[Solver] = [SOLVERS[name] for name in SOLVER_IDS]


# ----------------------------------------------------------------------
# The independent oracle and the tour check
# ----------------------------------------------------------------------
def oracle_cost(dist: Sequence[Sequence[float]], start: int = 0) -> float:
    """Return the optimum by pricing every Hamiltonian cycle through ``start``.

    This is the problem statement turned directly into code. Every cycle is
    written as ``(start, c1, ..., c_{n-1})`` and priced as the sum of
    ``dist[cycle[k]][cycle[(k + 1) % n]]`` for k in ``range(n)``: the
    wrap-around index closes the tour without a special final step. There
    is no dynamic programming state, no bitmask and no parent table, which
    is the only reason agreement with the module means anything.

    Args:
        dist: A square matrix of non-negative weights. Not modified.
        start: The city every cycle is written from.

    Returns:
        The minimum cycle cost, ``0`` for a single city, ``inf`` when no
        cycle has a finite cost.
    """
    n = len(dist)
    if n == 1:
        return 0
    others = [city for city in range(n) if city != start]
    best: float = INF
    for order in itertools.permutations(others):
        cycle = (start, *order)
        total = sum(dist[cycle[k]][cycle[(k + 1) % n]] for k in range(n))
        if total < best:
            best = total
    return best


def edge_sum(dist: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    """Return the total weight of the consecutive edges of ``tour``."""
    return sum(dist[a][b] for a, b in zip(tour, tour[1:]))


def assert_valid_tour(
    dist: Sequence[Sequence[float]],
    start: int,
    cost: float,
    tour: List[int],
) -> None:
    """Assert ``tour`` is a closed tour from ``start`` whose edges sum to ``cost``.

    Three properties, each failing for a different reason: a wrong endpoint
    means the reconstruction forgot to close the cycle, a repeated or
    missing city means it walked the parent table wrongly, and an edge sum
    that misses the cost means the tour and the cost came from different
    states.

    Args:
        dist: The matrix the tour was computed on.
        start: The city the tour must begin and end at.
        cost: The cost the solver reported.
        tour: The tour the solver reported.
    """
    n = len(dist)
    assert isinstance(tour, list)
    assert tour[0] == start
    assert tour[-1] == start
    if n == 1:
        assert tour == [start]
        assert cost == 0
        return
    assert len(tour) == n + 1
    assert sorted(tour[:-1]) == list(range(n))
    # math.isclose treats inf as close to inf, so a tour of a matrix with
    # no finite cycle passes exactly when its edges really do sum to inf.
    assert math.isclose(edge_sum(dist, tour), cost, rel_tol=1e-12, abs_tol=1e-12)


def rotate_cycle(cycle: Sequence[int], start: int) -> List[int]:
    """Return the closed tour that drives ``cycle`` from ``start``.

    Args:
        cycle: The cities of a cycle in order, without the closing repeat.
        start: A city on the cycle.

    Returns:
        ``[start, ..., start]``, the same cycle in the same direction.
    """
    k = list(cycle).index(start)
    body = list(cycle[k:]) + list(cycle[:k])
    return body + [start]


# ----------------------------------------------------------------------
# The seeded battery
# ----------------------------------------------------------------------
#: Instance families, each with its own seed offset so no two families can
#: draw the same numbers. Each one stresses a different part of the
#: recurrence.
FAMILY_SEEDS: Dict[str, int] = {
    "symmetric": 61000,  # d[i][j] == d[j][i], so every optimum ties with its reverse
    "asymmetric": 62000,  # every ordered pair drawn independently: direction matters
    "sparse": 63000,  # missing edges as inf, some instances with no finite tour
    "ties": 64000,  # weights in {0, 1, 2}: many optimal tours, zero edges included
    "fractional": 65000,  # asymmetric float weights
}

#: Three draws per family and size. 5 families x 8 sizes x 3 seeds is 120
#: instances, enough to catch an off-by-one in the mask handling without
#: paying for an exponential oracle a thousand times.
SEEDS: Tuple[int, ...] = (0, 1, 2)
SIZES: Tuple[int, ...] = tuple(range(1, 9))


def make_matrix(family: str, n: int, seed: int) -> Matrix:
    """Build one reproducible n x n instance of the named family.

    Every draw comes from a :class:`random.Random` seeded with the family's
    offset, the size and the seed, so a failure can be reproduced from its
    test id alone. The diagonal is 0 except where noted; the solvers never
    read it.

    Args:
        family: One of the keys of :data:`FAMILY_SEEDS`.
        n: Number of cities, 1 to 8.
        seed: Which draw within that family and size.

    Returns:
        A new list of lists.
    """
    seed_value = FAMILY_SEEDS[family] + 100 * n + seed
    rng = random.Random(seed_value)
    matrix: Matrix = [[0] * n for _ in range(n)]

    if family == "symmetric":
        for i in range(n):
            for j in range(i + 1, n):
                weight = rng.randint(1, 60)
                matrix[i][j] = weight
                matrix[j][i] = weight
    elif family == "asymmetric":
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = rng.randint(1, 100)
    elif family == "sparse":
        matrix = random_weight_matrix(n, 0.6, weight_range=(1, 30), seed=seed_value)
    elif family == "ties":
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = rng.choice((0, 1, 1, 2))
    else:
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = rng.uniform(0.5, 50.0)
    return matrix


#: Every ``(family, n, seed)`` triple, as pytest parameters with readable ids.
BATTERY = [
    pytest.param(family, n, seed, id=f"{family}-n{n}-seed{seed}")
    for family in sorted(FAMILY_SEEDS)
    for n in SIZES
    for seed in SEEDS
]


# ----------------------------------------------------------------------
# The hand-checked case
# ----------------------------------------------------------------------
class TestHandCheckedAnchor:
    """The assignment's instance, priced on paper before any code ran.

    Four cities, symmetric::

               0    1    2    3
        0  [   0,  10,  15,  20 ]
        1  [  10,   0,  35,  25 ]
        2  [  15,  35,   0,  30 ]
        3  [  20,  25,  30,   0 ]

    With the start fixed at 0 there are 3! = 6 directed tours, which pair
    up into three undirected cycles because the matrix is symmetric:

    * ``0-1-2-3-0``: 10 + 35 + 30 + 20 = 95
    * ``0-1-3-2-0``: 10 + 25 + 30 + 15 = **80**
    * ``0-2-1-3-0``: 15 + 35 + 25 + 20 = 95

    So the optimum is 80, achieved by ``[0, 1, 3, 2, 0]`` and by its
    reverse ``[0, 2, 3, 1, 0]``, and by nothing else. Either is a correct
    answer, so the tests assert the cost, the validity of the tour, and
    membership in that pair of two, never one specific list.
    """

    DIST: Matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    OPTIMUM: int = 80
    OPTIMAL_CYCLE: Tuple[int, ...] = (0, 1, 3, 2)

    def optimal_tours_from(self, start: int) -> List[List[int]]:
        """Both directions of the optimal cycle, driven from ``start``."""
        forward = rotate_cycle(self.OPTIMAL_CYCLE, start)
        backward = rotate_cycle(tuple(reversed(self.OPTIMAL_CYCLE)), start)
        return [forward, backward]

    def test_the_paper_prices_of_all_three_cycles(self) -> None:
        """Sanity check on the arithmetic above, through the oracle helpers."""
        assert edge_sum(self.DIST, [0, 1, 2, 3, 0]) == 95
        assert edge_sum(self.DIST, [0, 1, 3, 2, 0]) == 80
        assert edge_sum(self.DIST, [0, 2, 1, 3, 0]) == 95
        assert oracle_cost(self.DIST) == self.OPTIMUM

    def test_both_directions_of_the_optimal_cycle_cost_eighty(self) -> None:
        """Why the tests must not pin one tour: the reverse ties exactly."""
        assert edge_sum(self.DIST, [0, 1, 3, 2, 0]) == 80
        assert edge_sum(self.DIST, [0, 2, 3, 1, 0]) == 80

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_cost_is_eighty(self, solver: Solver) -> None:
        cost, _tour = solver(self.DIST)
        assert cost == self.OPTIMUM

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_tour_is_valid(self, solver: Solver) -> None:
        cost, tour = solver(self.DIST)
        assert_valid_tour(self.DIST, 0, cost, tour)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_tour_is_one_of_the_two_optimal_directions(
        self, solver: Solver
    ) -> None:
        _cost, tour = solver(self.DIST)
        assert tour in self.optimal_tours_from(0)

    @pytest.mark.parametrize("start", [1, 2, 3])
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_another_start_gives_the_same_cost_and_a_tour_from_there(
        self, solver: Solver, start: int
    ) -> None:
        """The optimal cycle does not depend on where it is entered."""
        cost, tour = solver(self.DIST, start=start)
        assert cost == self.OPTIMUM
        assert tour[0] == start and tour[-1] == start
        assert_valid_tour(self.DIST, start, cost, tour)
        assert tour in self.optimal_tours_from(start)

    def test_the_default_max_n_admits_the_anchor(self) -> None:
        cost, _tour = tsp_bitmask(self.DIST, 0, 20)
        assert cost == self.OPTIMUM

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_tuple_of_tuples_is_accepted(self, solver: Solver) -> None:
        dist = tuple(tuple(row) for row in self.DIST)
        cost, tour = solver(dist)
        assert cost == self.OPTIMUM
        assert_valid_tour(self.DIST, 0, cost, tour)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_input_matrix_is_not_modified(self, solver: Solver) -> None:
        dist = copy.deepcopy(self.DIST)
        solver(dist)
        assert dist == self.DIST


class TestHandCheckedDirectedInstances:
    """Small asymmetric and sparse instances whose answers are written out.

    In a directed instance the reverse of a tour is a different tour with a
    different price, so these are the cases where a solver that silently
    symmetrised the matrix, or read ``dist[j][i]`` for ``dist[i][j]``,
    would be caught.
    """

    #: Going round 0 -> 1 -> 2 -> 0 costs 1 + 1 + 1 = 3. The only other
    #: tour from 0, 0 -> 2 -> 1 -> 0, costs 10 + 10 + 10 = 30.
    ONE_WAY: Matrix = [[0, 1, 10], [10, 0, 1], [1, 10, 0]]

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_cheap_direction_is_found(self, solver: Solver) -> None:
        assert solver(self.ONE_WAY) == (3, [0, 1, 2, 0])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_transpose_reverses_the_answer(self, solver: Solver) -> None:
        """Transposed, 0 -> 2 -> 1 -> 0 is the cheap way: 1 + 1 + 1 = 3."""
        transposed = [list(column) for column in zip(*self.ONE_WAY)]
        assert transposed == [[0, 10, 1], [1, 0, 10], [10, 1, 0]]
        assert solver(transposed) == (3, [0, 2, 1, 0])

    @pytest.mark.parametrize("start", [0, 1, 2])
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_cheap_direction_from_every_start(
        self, solver: Solver, start: int
    ) -> None:
        cost, tour = solver(self.ONE_WAY, start=start)
        assert cost == 3
        assert tour == rotate_cycle((0, 1, 2), start)

    @pytest.mark.parametrize("start", [0, 2, 4])
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_directed_ring_is_the_only_finite_tour(
        self, solver: Solver, start: int
    ) -> None:
        """Edges 0->1->2->3->4->0 weigh 1, 2, 3, 4, 5; every other edge is missing."""
        ring: Matrix = [[INF] * 5 for _ in range(5)]
        for city, weight in zip(range(5), (1, 2, 3, 4, 5)):
            ring[city][city] = 0
            ring[city][(city + 1) % 5] = weight
        cost, tour = solver(ring, start=start)
        assert cost == 1 + 2 + 3 + 4 + 5
        assert tour == rotate_cycle((0, 1, 2, 3, 4), start)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_no_finite_tour_costs_inf_with_the_index_order_tour(
        self, solver: Solver
    ) -> None:
        """City 1 has no outgoing edge, so every tour ties at infinity.

        The documented contract returns ``inf`` with the tour that visits
        the other cities in index order.
        """
        trapped = [[0, 1, INF], [INF, 0, INF], [INF, 1, 0]]
        cost, tour = solver(trapped)
        assert cost == INF
        assert tour == [0, 1, 2, 0]
        assert_valid_tour(trapped, 0, cost, tour)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_all_zero_weights_cost_zero(self, solver: Solver) -> None:
        dist = [[0] * 5 for _ in range(5)]
        cost, tour = solver(dist)
        assert cost == 0
        assert_valid_tour(dist, 0, cost, tour)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_uniform_weights_cost_n_times_the_weight(self, solver: Solver) -> None:
        """Every tour of six cities uses six edges, so all tours cost 6 * 7."""
        dist = [[0 if i == j else 7 for j in range(6)] for i in range(6)]
        cost, tour = solver(dist)
        assert cost == 42
        assert_valid_tour(dist, 0, cost, tour)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_the_diagonal_is_never_read(self, solver: Solver) -> None:
        """A tour never moves from a city to itself, so a huge diagonal is inert."""
        loud = [
            [999 if i == j else w for j, w in enumerate(row)]
            for i, row in enumerate(self.ONE_WAY)
        ]
        assert solver(loud) == (3, [0, 1, 2, 0])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_fractional_weights_are_priced_exactly(self, solver: Solver) -> None:
        """0 -> 1 -> 2 -> 0 costs 0.5 + 0.25 + 0.125 = 0.875, exact in binary."""
        dist = [[0, 0.5, 9.0], [9.0, 0, 0.25], [0.125, 9.0, 0]]
        assert solver(dist) == (0.875, [0, 1, 2, 0])


# ----------------------------------------------------------------------
# One and two cities
# ----------------------------------------------------------------------
class TestTrivialSizes:
    """The sizes where there is nothing to search.

    One city is a tour of length zero: ``(0, [start])``. Two cities admit
    exactly one tour, out and back: ``(d[0][1] + d[1][0], [0, 1, 0])``.
    Both are answered before the DP allocates anything, so they are where a
    special case written for one solver and forgotten in the other would
    show.
    """

    @pytest.mark.parametrize("diagonal", [0, 7, 1.5])
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_one_city_is_a_tour_of_cost_zero(
        self, solver: Solver, diagonal: float
    ) -> None:
        assert solver([[diagonal]]) == (0, [0])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_one_city_with_an_explicit_start(self, solver: Solver) -> None:
        assert solver([[0]], start=0) == (0, [0])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_two_cities_is_the_round_trip(self, solver: Solver) -> None:
        """3 out and 5 back: 8, and the direction is forced."""
        assert solver([[0, 3], [5, 0]]) == (8, [0, 1, 0])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_two_cities_from_the_second(self, solver: Solver) -> None:
        assert solver([[0, 3], [5, 0]], start=1) == (8, [1, 0, 1])

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_two_cities_sum_both_directed_edges(
        self, solver: Solver, seed: int
    ) -> None:
        rng = random.Random(66000 + seed)
        d = [[0, rng.randint(0, 99)], [rng.randint(0, 99), 0]]
        assert solver(d) == (d[0][1] + d[1][0], [0, 1, 0])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_two_cities_with_a_missing_edge_cost_inf(self, solver: Solver) -> None:
        assert solver([[0, 4], [INF, 0]]) == (INF, [0, 1, 0])


# ----------------------------------------------------------------------
# Agreement with the oracle and with each other
# ----------------------------------------------------------------------
class TestAgreementBattery:
    """Both solvers against exhaustive enumeration on 120 seeded matrices.

    This is the strongest general claim in the file. The oracle is not an
    algorithm for TSP, it is the definition of TSP, so a solver that
    disagrees with it is wrong. Agreement is asserted on the cost only,
    since where several tours tie any of them is correct; each returned
    tour is then checked separately for validity.
    """

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_both_solvers_match_the_oracle(
        self, family: str, n: int, seed: int
    ) -> None:
        dist = make_matrix(family, n, seed)
        expected = oracle_cost(dist)
        bitmask_cost, _ = tsp_bitmask(dist)
        brute_cost, _ = tsp_brute_force(dist)
        assert bitmask_cost == pytest.approx(expected)
        assert brute_cost == pytest.approx(expected)
        assert bitmask_cost == pytest.approx(brute_cost)

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_every_returned_tour_is_valid(
        self, solver: Solver, family: str, n: int, seed: int
    ) -> None:
        dist = make_matrix(family, n, seed)
        cost, tour = solver(dist)
        assert_valid_tour(dist, 0, cost, tour)

    @pytest.mark.parametrize("n", [s for s in SIZES if s >= 2])
    def test_the_asymmetric_family_really_is_asymmetric(self, n: int) -> None:
        """Guards the battery's own claim: every asymmetric draw has d[i][j] != d[j][i]
        somewhere, so agreement there really does exercise directed tours."""
        for seed in SEEDS:
            dist = make_matrix("asymmetric", n, seed)
            assert any(
                dist[i][j] != dist[j][i] for i in range(n) for j in range(n)
            )

    def test_the_sparse_family_covers_finite_and_infinite_optima(self) -> None:
        """Guards the battery's own claim: some sparse draws have no finite tour
        and some do, so both branches of the answer are exercised."""
        optima = [
            oracle_cost(make_matrix("sparse", n, seed))
            for n in SIZES
            if n >= 3
            for seed in SEEDS
        ]
        assert any(math.isinf(cost) for cost in optima)
        assert any(not math.isinf(cost) for cost in optima)

    @pytest.mark.parametrize("n", [3, 5, 7])
    @pytest.mark.parametrize("family", ["asymmetric", "sparse", "ties"])
    def test_every_start_city_matches_the_oracle(self, family: str, n: int) -> None:
        """A start other than 0 changes the masks the DP iterates, not the optimum.

        For each start the tour must begin and end there, both solvers must
        match the oracle written from that start, and the optimum must equal
        the start-0 optimum, since the best cycle is the best cycle wherever
        it is entered.
        """
        dist = make_matrix(family, n, 0)
        from_zero = oracle_cost(dist, 0)
        for start in range(n):
            assert oracle_cost(dist, start) == pytest.approx(from_zero)
            for solver in SOLVER_PARAMS:
                cost, tour = solver(dist, start=start)
                assert cost == pytest.approx(from_zero)
                assert tour[0] == start and tour[-1] == start
                assert_valid_tour(dist, start, cost, tour)


# ----------------------------------------------------------------------
# Metamorphic relations
# ----------------------------------------------------------------------
class TestMetamorphicRelations:
    """Transformations whose effect on the optimum is known without solving.

    Each test changes the matrix in a way that moves the answer by a known
    amount, or not at all, and checks that the DP follows. None of them
    needs the answer in advance, so a solver that is consistently wrong in
    some direction-dependent or label-dependent way cannot satisfy them.
    """

    SEEDS_HERE: Tuple[int, ...] = (0, 1, 2)

    @pytest.mark.parametrize("seed", SEEDS_HERE)
    def test_transposing_reverses_every_cycle_and_keeps_the_optimum(
        self, seed: int
    ) -> None:
        dist = make_matrix("asymmetric", 7, seed)
        transposed = [list(column) for column in zip(*dist)]
        cost, tour = tsp_bitmask(dist)
        t_cost, t_tour = tsp_bitmask(transposed)
        assert t_cost == cost
        # The reverse of an optimal tour is a tour of the transpose, with
        # the same price, so it must be optimal there too.
        assert edge_sum(transposed, list(reversed(tour))) == t_cost
        assert_valid_tour(transposed, 0, t_cost, t_tour)

    @pytest.mark.parametrize("constant", [1, 5, 100])
    @pytest.mark.parametrize("seed", SEEDS_HERE)
    def test_adding_a_constant_to_every_edge_adds_n_times_it(
        self, seed: int, constant: int
    ) -> None:
        """Every tour uses exactly n edges, so the ranking of tours is unchanged."""
        n = 7
        dist = make_matrix("asymmetric", n, seed)
        shifted = [
            [w if i == j else w + constant for j, w in enumerate(row)]
            for i, row in enumerate(dist)
        ]
        assert tsp_bitmask(shifted)[0] == tsp_bitmask(dist)[0] + n * constant

    @pytest.mark.parametrize("factor", [2, 3, 10])
    @pytest.mark.parametrize("seed", SEEDS_HERE)
    def test_scaling_every_edge_scales_the_optimum(
        self, seed: int, factor: int
    ) -> None:
        dist = make_matrix("asymmetric", 7, seed)
        scaled = [[w * factor for w in row] for row in dist]
        assert tsp_bitmask(scaled)[0] == tsp_bitmask(dist)[0] * factor

    @pytest.mark.parametrize("seed", SEEDS_HERE)
    def test_relabelling_the_cities_keeps_the_optimum(self, seed: int) -> None:
        """City ``k`` of the relabelled matrix is city ``label[k]`` of the original."""
        n = 7
        dist = make_matrix("asymmetric", n, seed)
        label = list(range(n))
        random.Random(67000 + seed).shuffle(label)
        relabelled = [[dist[label[i]][label[j]] for j in range(n)] for i in range(n)]
        cost, tour = tsp_bitmask(relabelled)
        assert cost == tsp_bitmask(dist)[0]
        # Mapped back through the labels, the tour is a tour of the original
        # matrix with the same price.
        original_tour = [label[city] for city in tour]
        assert edge_sum(dist, original_tour) == cost


# ----------------------------------------------------------------------
# The shared contract
# ----------------------------------------------------------------------
class TestContracts:
    """Argument checking, run over both solvers at once.

    Both entry points funnel through one validator, which is the reason to
    parametrise: the moment one of them stops calling it, these tests are
    what says so. The memory guard belongs to the DP alone, so it is tested
    on :func:`tsp_bitmask` only.
    """

    @pytest.mark.parametrize(
        "dist",
        [
            [[0, 1, 2], [1, 0, 3]],
            [[0, 1], [1, 0], [2, 2]],
            [[0, 1], [1]],
            [[0, 1, 2], [1, 0], [2, 3, 0]],
        ],
        ids=["two_rows_of_three", "three_rows_of_two", "ragged_short", "ragged_middle"],
    )
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_non_square_matrix_is_a_value_error(
        self, solver: Solver, dist: Matrix
    ) -> None:
        with pytest.raises(ValueError, match="dist must be square"):
            solver(dist)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_an_empty_matrix_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match="at least one city"):
            solver([])

    @pytest.mark.parametrize(
        "dist, where",
        [
            ([[0, -1], [1, 0]], r"dist\[0\]\[1\]"),
            ([[0, 1, 2], [1, 0, 3], [4, -5, 0]], r"dist\[2\]\[1\]"),
            ([[0, 1, 2], [1, 0, 3], [4, 5, -0.5]], r"dist\[2\]\[2\]"),
            ([[0, -INF], [1, 0]], r"dist\[0\]\[1\]"),
        ],
        ids=["off_diagonal", "lower_triangle", "fractional_on_diagonal", "minus_inf"],
    )
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_negative_weight_is_a_value_error(
        self, solver: Solver, dist: Matrix, where: str
    ) -> None:
        """Validation checks every entry, the unread diagonal included."""
        with pytest.raises(ValueError, match=where + " must be >= 0"):
            solver(dist)

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_nan_weight_is_a_value_error(self, solver: Solver) -> None:
        with pytest.raises(ValueError, match=r"dist\[1\]\[0\] must be >= 0"):
            solver([[0, 1], [math.nan, 0]])

    @pytest.mark.parametrize(
        "bad", [True, "3", None, [3]], ids=["bool", "str", "none", "list"]
    )
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_non_number_weight_is_a_type_error(
        self, solver: Solver, bad: object
    ) -> None:
        with pytest.raises(TypeError, match=r"dist\[0\]\[1\] must be a real number"):
            solver([[0, bad], [1, 0]])

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_rows_that_are_not_sequences_are_a_type_error(
        self, solver: Solver
    ) -> None:
        with pytest.raises(TypeError, match="square matrix of numbers"):
            solver([1, 2])

    @pytest.mark.parametrize("start", [-1, 3, 10])
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_start_out_of_range_is_a_value_error(
        self, solver: Solver, start: int
    ) -> None:
        with pytest.raises(ValueError, match=r"start must be in range\(3\)"):
            solver([[0, 1, 1], [1, 0, 1], [1, 1, 0]], start=start)

    @pytest.mark.parametrize(
        "start", [1.0, True, "0", None], ids=["float", "bool", "str", "none"]
    )
    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_non_int_start_is_a_type_error(
        self, solver: Solver, start: object
    ) -> None:
        with pytest.raises(TypeError, match="start must be an int"):
            solver([[0, 1, 1], [1, 0, 1], [1, 1, 0]], start=start)

    def test_more_than_twenty_cities_is_refused_by_default(self) -> None:
        dist = [[0] * 21 for _ in range(21)]
        with pytest.raises(ValueError, match=r"n = 21 exceeds max_n = 20"):
            tsp_bitmask(dist)

    def test_the_guard_message_names_the_memory_cost(self) -> None:
        dist = [[0] * 21 for _ in range(21)]
        with pytest.raises(ValueError, match=r"O\(n \* 2\^n\) memory"):
            tsp_bitmask(dist)

    @pytest.mark.parametrize("n, max_n", [(3, 2), (5, 4), (7, 6), (8, 1)])
    def test_a_lower_max_n_refuses_a_small_instance(self, n: int, max_n: int) -> None:
        dist = make_matrix("asymmetric", n, 0)
        with pytest.raises(ValueError, match=f"n = {n} exceeds max_n = {max_n}"):
            tsp_bitmask(dist, max_n=max_n)

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_n_equal_to_max_n_is_admitted(self, n: int) -> None:
        """The guard is ``n > max_n``, so the boundary itself is solved."""
        dist = make_matrix("asymmetric", n, 1)
        cost, tour = tsp_bitmask(dist, max_n=n)
        assert cost == oracle_cost(dist)
        assert_valid_tour(dist, 0, cost, tour)

    def test_a_larger_max_n_lifts_the_default_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raising ``max_n`` to 21 lets a 21-city matrix past the guard.

        Actually solving 21 cities would allocate two tables of 21 * 2^21
        (about 44 million) cells each, far too much for a unit test. The
        guard is the last check before the DP starts, and the first thing
        after it is the call to ``_trivial_tour``, so that helper is
        replaced by one that raises a marker. Reaching the marker proves
        the guard let the instance through; the default ``max_n`` must
        still stop the same matrix before it gets there.
        """

        class PassedTheGuard(Exception):
            pass

        def marker(*_args: object) -> None:
            raise PassedTheGuard

        monkeypatch.setattr(tsp_module, "_trivial_tour", marker)
        dist = [[0] * 21 for _ in range(21)]
        with pytest.raises(PassedTheGuard):
            tsp_bitmask(dist, max_n=21)
        with pytest.raises(ValueError, match="exceeds max_n = 20"):
            tsp_bitmask(dist)

    @pytest.mark.parametrize(
        "max_n", [20.0, True, "20", None], ids=["float", "bool", "str", "none"]
    )
    def test_a_non_int_max_n_is_a_type_error(self, max_n: object) -> None:
        with pytest.raises(TypeError, match="max_n must be an int"):
            tsp_bitmask([[0, 1], [1, 0]], max_n=max_n)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# The bitmask helpers, against plain set arithmetic
# ----------------------------------------------------------------------
def members(mask: int) -> Set[int]:
    """Return the set a mask stands for, read off its binary string.

    No shift and no AND: the ``k``-th character from the right of
    ``format(mask, "b")`` is ``"1"`` exactly when city ``k`` is present.
    """
    return {
        position
        for position, digit in enumerate(reversed(format(mask, "b")))
        if digit == "1"
    }


def to_mask(cities: Iterable[int]) -> int:
    """Return the mask of a set of cities as a sum of distinct powers of two."""
    return sum(2**city for city in set(cities))


#: Every mask below 300, which covers every subset of nine cities, plus a
#: handful past the machine word to show nothing assumes 64 bits.
MASKS: List[int] = list(range(300)) + [
    2**31 - 1,
    2**31,
    2**40 + 2**7 + 1,
    2**63 - 1,
    2**64 + 5,
    0b1010101010101,
]

#: Bit indices inside, at the edge of, and well past the masks above.
INDICES: List[int] = list(range(12)) + [31, 40, 63, 64, 70]


class TestBitmaskHelperAnchors:
    """A few values worked out by hand, before the sweeps below."""

    def test_has_bit_on_five(self) -> None:
        # 5 is 0b101: cities 0 and 2.
        assert has_bit(0b101, 0) is True
        assert has_bit(0b101, 1) is False
        assert has_bit(0b101, 2) is True
        assert has_bit(0b101, 50) is False

    def test_set_and_clear_on_small_masks(self) -> None:
        assert set_bit(0, 0) == 1
        assert set_bit(0b100, 1) == 0b110
        assert set_bit(0b110, 1) == 0b110
        assert clear_bit(0b111, 1) == 0b101
        assert clear_bit(0b100, 0) == 0b100

    def test_full_mask_small_values(self) -> None:
        assert [full_mask(n) for n in range(5)] == [0, 1, 3, 7, 15]

    def test_popcount_and_listing(self) -> None:
        assert popcount(0b1011) == 3
        assert mask_to_list(0b101001) == [0, 3, 5]
        assert list(iter_bits(0b1101)) == [0, 2, 3]

    def test_two_of_four_cities_in_increasing_order(self) -> None:
        # {0,1}=3, {0,2}=5, {1,2}=6, {0,3}=9, {1,3}=10, {2,3}=12.
        assert list(subsets_of_size(4, 2)) == [3, 5, 6, 9, 10, 12]


class TestBitmaskHelpersAgainstSets:
    """Every helper against the ``set`` operation it names, on 306 masks.

    The oracle reads masks through their binary strings and builds them
    from powers of two, so a helper that shifted the wrong way, or masked
    with the wrong complement, would disagree with it on most inputs.
    """

    @pytest.mark.parametrize("i", INDICES)
    def test_has_bit_is_membership(self, i: int) -> None:
        for mask in MASKS:
            assert has_bit(mask, i) is (i in members(mask))

    @pytest.mark.parametrize("i", INDICES)
    def test_set_bit_is_insertion(self, i: int) -> None:
        for mask in MASKS:
            assert set_bit(mask, i) == to_mask(members(mask) | {i})

    @pytest.mark.parametrize("i", INDICES)
    def test_clear_bit_is_removal(self, i: int) -> None:
        for mask in MASKS:
            assert clear_bit(mask, i) == to_mask(members(mask) - {i})

    @pytest.mark.parametrize("i", INDICES)
    def test_insert_and_remove_are_idempotent_and_inverse(self, i: int) -> None:
        for mask in MASKS:
            added = set_bit(mask, i)
            removed = clear_bit(mask, i)
            assert set_bit(added, i) == added
            assert clear_bit(removed, i) == removed
            assert clear_bit(added, i) == removed
            assert set_bit(removed, i) == added

    @pytest.mark.parametrize("n", list(range(21)) + [63, 64, 100])
    def test_full_mask_is_every_city(self, n: int) -> None:
        mask = full_mask(n)
        assert mask == to_mask(range(n))
        assert members(mask) == set(range(n))

    def test_popcount_is_set_size(self) -> None:
        for mask in MASKS:
            assert popcount(mask) == len(members(mask))

    def test_iter_bits_is_the_sorted_members(self) -> None:
        for mask in MASKS:
            assert list(iter_bits(mask)) == sorted(members(mask))

    def test_iter_bits_returns_an_iterator_not_a_list(self) -> None:
        it = iter_bits(0b1101)
        assert iter(it) is it
        assert next(it) == 0

    def test_mask_to_list_is_the_sorted_members(self) -> None:
        for mask in MASKS:
            assert mask_to_list(mask) == sorted(members(mask))

    def test_mask_to_list_returns_a_fresh_list(self) -> None:
        first = mask_to_list(0b111)
        first.append(99)
        assert mask_to_list(0b111) == [0, 1, 2]


class TestSubsetsOfSize:
    """``subsets_of_size(n, k)`` against :func:`itertools.combinations`.

    For every ``n`` up to 10 and every ``k`` up to ``n + 2``: exactly
    ``C(n, k)`` masks, each with ``k`` bits, all below ``2^n``, strictly
    increasing, and equal as a list to the combinations turned into masks
    and sorted. ``k > n`` yields nothing, which ``math.comb`` agrees is 0.
    """

    @pytest.mark.parametrize("n", range(11))
    def test_matches_combinations_for_every_k(self, n: int) -> None:
        for k in range(n + 3):
            produced = list(subsets_of_size(n, k))
            expected = sorted(
                to_mask(chosen) for chosen in itertools.combinations(range(n), k)
            )
            assert produced == expected

    @pytest.mark.parametrize("n", range(11))
    def test_count_is_n_choose_k(self, n: int) -> None:
        for k in range(n + 3):
            assert sum(1 for _ in subsets_of_size(n, k)) == math.comb(n, k)

    @pytest.mark.parametrize("n", range(11))
    def test_each_mask_has_k_bits_and_fits_in_n(self, n: int) -> None:
        for k in range(n + 1):
            for mask in subsets_of_size(n, k):
                assert len(members(mask)) == k
                assert members(mask) <= set(range(n))

    @pytest.mark.parametrize("n", range(11))
    def test_masks_come_out_strictly_increasing(self, n: int) -> None:
        for k in range(n + 1):
            produced = list(subsets_of_size(n, k))
            assert all(a < b for a, b in zip(produced, produced[1:]))

    @pytest.mark.parametrize("n", range(11))
    def test_all_sizes_together_are_every_mask_once(self, n: int) -> None:
        """The layers partition the 2^n subsets: none missed, none repeated."""
        every = [mask for k in range(n + 1) for mask in subsets_of_size(n, k)]
        assert sorted(every) == list(range(2**n))

    def test_the_edge_sizes(self) -> None:
        assert list(subsets_of_size(0, 0)) == [0]
        assert list(subsets_of_size(4, 0)) == [0]
        assert list(subsets_of_size(4, 4)) == [15]
        assert list(subsets_of_size(3, 5)) == []


# ----------------------------------------------------------------------
# The helpers' contract
# ----------------------------------------------------------------------
#: Every argument slot of every helper, as a call taking the bad value.
#: ``iter_bits`` and ``subsets_of_size`` return iterators, and their calls
#: here are never advanced, so a raise proves validation is eager.
ARGUMENT_SLOTS: Dict[str, Callable[[object], object]] = {
    "has_bit-mask": lambda bad: has_bit(bad, 0),  # type: ignore[arg-type]
    "has_bit-i": lambda bad: has_bit(5, bad),  # type: ignore[arg-type]
    "set_bit-mask": lambda bad: set_bit(bad, 0),  # type: ignore[arg-type]
    "set_bit-i": lambda bad: set_bit(5, bad),  # type: ignore[arg-type]
    "clear_bit-mask": lambda bad: clear_bit(bad, 0),  # type: ignore[arg-type]
    "clear_bit-i": lambda bad: clear_bit(5, bad),  # type: ignore[arg-type]
    "full_mask-n": lambda bad: full_mask(bad),  # type: ignore[arg-type]
    "popcount-mask": lambda bad: popcount(bad),  # type: ignore[arg-type]
    "iter_bits-mask": lambda bad: iter_bits(bad),  # type: ignore[arg-type]
    "mask_to_list-mask": lambda bad: mask_to_list(bad),  # type: ignore[arg-type]
    "subsets_of_size-n": lambda bad: subsets_of_size(bad, 0),  # type: ignore[arg-type]
    "subsets_of_size-k": lambda bad: subsets_of_size(4, bad),  # type: ignore[arg-type]
}
SLOT_IDS: List[str] = sorted(ARGUMENT_SLOTS)
SLOT_PARAMS: List[Callable[[object], object]] = [
    ARGUMENT_SLOTS[name] for name in SLOT_IDS
]


class TestBitmaskContracts:
    """Negatives are a ``ValueError``; non-ints, ``bool`` included, a ``TypeError``.

    A negative Python int has infinitely many 1 bits in two's complement,
    so it does not describe a finite set of cities; ``True`` is an int to
    Python but an accident to a caller. Every slot of every helper is
    checked, because one forgotten validation call is all it takes.
    """

    @pytest.mark.parametrize("bad", [-1, -8, -(2**40)])
    @pytest.mark.parametrize("call", SLOT_PARAMS, ids=SLOT_IDS)
    def test_a_negative_argument_is_a_value_error(
        self, call: Callable[[object], object], bad: int
    ) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            call(bad)

    @pytest.mark.parametrize(
        "bad",
        [1.0, "3", None, True, [1]],
        ids=["float", "str", "none", "bool", "list"],
    )
    @pytest.mark.parametrize("call", SLOT_PARAMS, ids=SLOT_IDS)
    def test_a_non_int_argument_is_a_type_error(
        self, call: Callable[[object], object], bad: object
    ) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            call(bad)
