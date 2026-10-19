"""Tests for nearest-neighbor TSP, 2-opt local search and the tour helpers.

A heuristic TSP answer can be wrong in two ways that look right: the tour
can be invalid (a city skipped or repeated, the tour not closed), or the
reported cost can disagree with the tour. Optimality is not promised, so
the claims tested here are the ones the heuristics do make, held against
witnesses that share no code with the module:

* **Hand-checked anchors.** The Week 6 four-city matrix (greedy walk 10 +
  25 + 30 + 15 = 80, equal to the optimum); six points on a line at
  x = 0, 1, -2, 4, -8, 16, where the greedy walk zig-zags outwards for
  1 + 3 + 6 + 12 + 24 + 16 = 62 against the optimum 2 * (16 - (-8)) = 48;
  and a four-city non-metric "trap" whose greedy tour costs M + 3 against
  an optimum of 6, so the ratio is unbounded. Every expected tour and cost
  in those classes was traced by hand and is written out in the test's
  docstring.
* **The Week 6 exact solver.** :func:`tsp_brute_force` gives the optimum
  on seeded Euclidean and asymmetric instances with n <= 9, and every
  heuristic tour must cost at least that much.
* **Independent tour checks.** :func:`assert_closed_permutation` and
  :func:`edge_sum` re-derive validity and cost from the definition, and
  :func:`assert_two_optimal` re-prices every possible 2-opt reversal from
  scratch to confirm that none improves the returned tour.

Every random draw comes from :func:`make_rng` with a seed from
:func:`derive_seed`, no instance passed to the brute-force oracle exceeds
nine cities, and no test asserts on time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import copy
import math
import random
from fractions import Fraction
from typing import List, Sequence

import pytest

from src.complexity.traveling_salesman_approx import (
    euclidean_instance,
    is_valid_tour,
    tour_cost,
    tsp_nearest_neighbor,
    tsp_nearest_neighbor_best_start,
    tsp_two_opt,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_brute_force
from src.utils.random_utils import derive_seed, make_rng

Matrix = List[List[float]]

#: Master seed for every instance drawn in this file.
MASTER_SEED = 8005

#: Slack for comparing a heuristic's float cost with the exact optimum.
TOL = 1e-9

#: The Week 6 anchor. Symmetric; the optimum is 80.
FOUR_CITY: Matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]

#: Six points on a line. Distances are |a - b|, so the instance is metric.
LINE_X: List[int] = [0, 1, -2, 4, -8, 16]
LINE: Matrix = [[abs(a - b) for b in LINE_X] for a in LINE_X]

#: Directed three-city cycle: 0 -> 1 -> 2 -> 0 costs 3, the reverse 30.
ONE_WAY: Matrix = [[0, 1, 10], [10, 0, 1], [1, 10, 0]]


def trap_matrix(m: int) -> Matrix:
    """Return the four-city non-metric trap with one expensive edge ``m``.

    Edges: d(0,1) = d(1,2) = d(2,3) = 1, d(0,2) = d(1,3) = 2, d(0,3) = m.
    The three distinct cycles cost m + 3 (0-1-2-3), 6 (0-1-3-2) and m + 5
    (0-2-1-3), so for m >= 3 the optimum is 6. Nearest neighbor from 0
    walks 0 -> 1 -> 2 -> 3 (each step's cheapest edge) and must close with
    the edge of cost m. d(3,0) = m > d(3,2) + d(2,0) = 3 breaks the
    triangle inequality.

    Args:
        m: The cost of the edge between cities 0 and 3.

    Returns:
        A new symmetric 4 x 4 matrix.
    """
    return [[0, 1, 2, m], [1, 0, 1, 2], [2, 1, 0, 1], [m, 2, 1, 0]]


# ----------------------------------------------------------------------
# Independent checks
# ----------------------------------------------------------------------
def edge_sum(dist: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    """Return the sum of the consecutive edges of a tour with n >= 2 cities.

    On Python 3.12 the built-in :func:`sum` adds floats with compensated
    (Neumaier) summation, while the module adds them one by one, so on
    float instances the two can differ in the last bit. Comparisons with
    this oracle therefore use :data:`TOL`; exact "never increases"
    comparisons use :func:`tour_cost`, which prices a tour the same way the
    solvers do.
    """
    return sum(dist[tour[k]][tour[k + 1]] for k in range(len(tour) - 1))


def assert_closed_permutation(tour: Sequence[int], n: int, start: int) -> None:
    """Assert ``tour`` is a list starting and ending at ``start`` through all n cities.

    Args:
        tour: The tour a function returned.
        n: Number of cities.
        start: The city the tour must start and end at.
    """
    assert isinstance(tour, list)
    assert len(tour) == n + 1
    assert tour[0] == start
    assert tour[-1] == start
    assert sorted(tour[:-1]) == list(range(n))


def assert_two_optimal(dist: Sequence[Sequence[float]], tour: List[int]) -> None:
    """Assert that no 2-opt reversal of ``tour`` is cheaper, re-pricing each fully.

    Every reversal of ``tour[i + 1 .. j]`` with ``0 <= i`` and
    ``i + 2 <= j <= n - 1`` is built as a new list and priced edge by edge,
    in its own direction, so the check also covers asymmetric matrices and
    shares nothing with the module's O(1) move pricing.

    Args:
        dist: The distance matrix.
        tour: A closed tour returned by :func:`tsp_two_opt`.
    """
    n = len(tour) - 1
    current = edge_sum(dist, tour)
    for i in range(n - 2):
        for j in range(i + 2, n):
            candidate = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
            assert edge_sum(dist, candidate) >= current - TOL, (i, j, candidate)


def random_closed_tour(n: int, start: int, seed: int) -> List[int]:
    """Return a seeded random closed tour of n cities beginning at ``start``."""
    rng = make_rng(seed)
    others = [city for city in range(n) if city != start]
    rng.shuffle(others)
    return [start, *others, start]


def asymmetric_matrix(n: int, seed: int) -> Matrix:
    """Return a seeded n x n matrix with independent integer weights in [1, 50]."""
    rng = make_rng(seed)
    return [[0 if i == j else rng.randint(1, 50) for j in range(n)] for i in range(n)]


# ----------------------------------------------------------------------
# Hand-checked anchors
# ----------------------------------------------------------------------
class TestFourCityAnchor:
    """The Week 6 instance. From 0 the cheapest edge is to 1 (10); from 1
    the cheaper of 35 and 25 is 3 (25); then 2 (30); home 15. Total 80,
    which is the optimum (Week 6 anchor), so here the greedy walk is optimal.
    """

    def test_nearest_neighbor_from_zero(self) -> None:
        assert tsp_nearest_neighbor(FOUR_CITY) == (80, [0, 1, 3, 2, 0])
        assert tsp_nearest_neighbor(FOUR_CITY, start=0) == (80, [0, 1, 3, 2, 0])

    def test_equals_brute_force_optimum(self) -> None:
        optimum, _ = tsp_brute_force(FOUR_CITY)
        assert optimum == 80
        assert tsp_nearest_neighbor(FOUR_CITY)[0] == optimum

    def test_tour_cost_by_hand(self) -> None:
        assert tour_cost(FOUR_CITY, [0, 1, 3, 2, 0]) == 10 + 25 + 30 + 15 == 80
        assert tour_cost(FOUR_CITY, [0, 1, 2, 3, 0]) == 10 + 35 + 30 + 20 == 95
        assert tour_cost(FOUR_CITY, [0, 2, 1, 3, 0]) == 15 + 35 + 25 + 20 == 95

    def test_best_start_is_optimal_and_starts_at_zero(self) -> None:
        # Bounded below by the optimum 80 and above by the start-0 walk, 80.
        cost, tour = tsp_nearest_neighbor_best_start(FOUR_CITY)
        assert cost == 80
        assert_closed_permutation(tour, 4, 0)
        assert edge_sum(FOUR_CITY, tour) == 80

    def test_two_opt_leaves_an_optimal_tour_alone(self) -> None:
        # Only strictly improving moves are applied, and none can improve 80.
        assert tsp_two_opt(FOUR_CITY, [0, 1, 3, 2, 0]) == (80, [0, 1, 3, 2, 0])


class TestLineAnchor:
    """Points at x = 0, 1, -2, 4, -8, 16 (cities 0..5), distance |a - b|.

    From x = 0 the nearest is 1 (cost 1). From 1, x = -2 and x = 4 tie at
    3: lowest index, city 2. From -2, x = 4 and x = -8 tie at 6: city 3.
    From 4, x = -8 and x = 16 tie at 12: city 4. Then 16 (24) and home
    (16). Total 1 + 3 + 6 + 12 + 24 + 16 = 62. Any tour of points on a line
    must cross the span [-8, 16] at least twice, and 0 -> 1 -> 4 -> 16 ->
    -2 -> -8 -> 0 does it exactly twice, so the optimum is 2 * 24 = 48.
    """

    def test_nearest_neighbor_zig_zags(self) -> None:
        assert tsp_nearest_neighbor(LINE) == (62, [0, 1, 2, 3, 4, 5, 0])

    def test_brute_force_optimum_is_48(self) -> None:
        assert tsp_brute_force(LINE)[0] == 48 == 2 * (max(LINE_X) - min(LINE_X))

    def test_ratio_shows_the_heuristic_is_not_optimal(self) -> None:
        nn_cost, _ = tsp_nearest_neighbor(LINE)
        optimum, _ = tsp_brute_force(LINE)
        assert nn_cost / optimum == pytest.approx(62 / 48)
        assert round(nn_cost / optimum, 2) == 1.29

    @pytest.mark.parametrize(
        "start, cost, tour",
        [
            # 1 + 3 + 6 + 12 + 24 + 16
            (0, 62, [0, 1, 2, 3, 4, 5, 0]),
            # 1 + 2 + 6 (tie 4 vs -8 -> city 3) + 12 (tie -> city 4) + 24 + 15
            (1, 60, [1, 0, 2, 3, 4, 5, 1]),
            # 2 + 1 + 3 + 12 (tie -> city 4) + 24 + 18
            (2, 60, [2, 0, 1, 3, 4, 5, 2]),
            # 3 + 1 + 2 + 6 + 24 + 12
            (3, 48, [3, 1, 0, 2, 4, 5, 3]),
            # 6 + 2 + 1 + 3 + 12 + 24
            (4, 48, [4, 2, 0, 1, 3, 5, 4]),
            # 12 + 3 + 1 + 2 + 6 + 24
            (5, 48, [5, 3, 1, 0, 2, 4, 5]),
        ],
    )
    def test_every_start_traced_by_hand(self, start: int, cost: int, tour: List[int]) -> None:
        assert tsp_nearest_neighbor(LINE, start=start) == (cost, tour)

    def test_best_start_rotates_the_first_cheapest_to_zero(self) -> None:
        # Starts 3, 4 and 5 all reach 48; the strict comparison keeps start 3,
        # whose tour [3, 1, 0, 2, 4, 5, 3] rotated to city 0 is below.
        assert tsp_nearest_neighbor_best_start(LINE) == (48, [0, 2, 4, 5, 3, 1, 0])

    def test_two_opt_reaches_the_optimum(self) -> None:
        """First pass, i = 0 (a = x 0), on x-sequence 0, 1, -2, 4, -8, 16:

        j = 2: reverse (1, -2): +2 + 3 - 1 - 6 = -2  -> 0, -2, 1, 4, -8, 16 (60)
        j = 3: reverse (-2, 1, 4): +4 + 6 - 2 - 12 = -4 -> 0, 4, 1, -2, -8, 16 (56)
        j = 4: reverse (4, 1, -2, -8): +8 + 12 - 4 - 24 = -8
               -> 0, -8, -2, 1, 4, 16 (48), which is optimal, so no later
        move can be strictly improving.
        """
        assert tsp_two_opt(LINE, [0, 1, 2, 3, 4, 5, 0]) == (48, [0, 4, 2, 1, 3, 5, 0])


class TestNonMetricTrap:
    """Nearest neighbor's ratio is unbounded once the triangle inequality fails."""

    @pytest.mark.parametrize("m", [10, 1000, 10**6])
    def test_nearest_neighbor_pays_the_trap_edge(self, m: int) -> None:
        dist = trap_matrix(m)
        assert dist[3][0] > dist[3][2] + dist[2][0]  # not metric
        assert tsp_nearest_neighbor(dist) == (m + 3, [0, 1, 2, 3, 0])
        assert tsp_brute_force(dist)[0] == 6

    def test_ratio_grows_without_bound(self) -> None:
        ratios = [tsp_nearest_neighbor(trap_matrix(m))[0] / 6 for m in (10, 1000, 10**6)]
        assert ratios == [13 / 6, 1003 / 6, 1000003 / 6]
        assert ratios[-1] > 1e5

    def test_two_opt_escapes_this_trap(self) -> None:
        # i = 1, j = 3: d(1,3) + d(2,0) - d(1,2) - d(3,0) = 2 + 2 - 1 - m < 0.
        assert tsp_two_opt(trap_matrix(1000), [0, 1, 2, 3, 0]) == (6, [0, 1, 3, 2, 0])


# ----------------------------------------------------------------------
# Tiny instances, ties, asymmetry
# ----------------------------------------------------------------------
class TestSmallAndSpecialInstances:
    def test_one_city(self) -> None:
        assert tsp_nearest_neighbor([[0]]) == (0, [0, 0])
        assert tsp_nearest_neighbor_best_start([[0]]) == (0, [0, 0])
        assert tsp_two_opt([[0]], [0, 0]) == (0, [0, 0])
        assert tour_cost([[0]], [0, 0]) == 0

    def test_one_city_never_reads_the_diagonal(self) -> None:
        assert tsp_nearest_neighbor([[7]]) == (0, [0, 0])
        assert tour_cost([[7]], [0, 0]) == 0

    def test_two_cities(self) -> None:
        dist = [[0, 3], [5, 0]]
        assert tsp_nearest_neighbor(dist) == (8, [0, 1, 0])
        assert tsp_nearest_neighbor(dist, start=1) == (8, [1, 0, 1])
        assert tsp_nearest_neighbor_best_start(dist) == (8, [0, 1, 0])
        assert tsp_two_opt(dist, [0, 1, 0]) == (8, [0, 1, 0])
        assert tsp_two_opt(dist, [1, 0, 1]) == (8, [1, 0, 1])

    def test_all_ties_go_to_the_lowest_index(self) -> None:
        flat = [[0 if i == j else 1 for j in range(5)] for i in range(5)]
        assert tsp_nearest_neighbor(flat) == (5, [0, 1, 2, 3, 4, 0])
        assert tsp_nearest_neighbor(flat, start=2) == (5, [2, 0, 1, 3, 4, 2])

    def test_missing_edges_still_give_a_tour(self) -> None:
        inf = math.inf
        none = [[0, inf, inf], [inf, 0, inf], [inf, inf, 0]]
        assert tsp_nearest_neighbor(none) == (inf, [0, 1, 2, 0])
        cost, tour = tsp_two_opt(none, [0, 1, 2, 0])
        assert cost == inf
        assert tour == [0, 1, 2, 0]

    def test_asymmetric_direction_matters(self) -> None:
        # 0 -> 1 -> 2 -> 0 costs 1 + 1 + 1 = 3; the reverse costs 30.
        assert tsp_nearest_neighbor(ONE_WAY) == (3, [0, 1, 2, 0])
        assert tour_cost(ONE_WAY, [0, 2, 1, 0]) == 30
        assert tsp_brute_force(ONE_WAY)[0] == 3

    def test_two_opt_reverses_an_asymmetric_tour(self) -> None:
        # Reversing the whole interior of 0 -> 2 -> 1 -> 0 drives the cycle
        # the cheap way round: the change is 1 + 1 - 10 - 10 + (1 - 10) = -27.
        assert tsp_two_opt(ONE_WAY, [0, 2, 1, 0]) == (3, [0, 1, 2, 0])

    def test_two_opt_keeps_a_nonzero_start(self) -> None:
        s = 2 ** 0.5
        square = [[0, 1, s, 1], [1, 0, 1, s], [s, 1, 0, 1], [1, s, 1, 0]]
        # 2 -> 0 -> 1 -> 3 -> 2 crosses; uncrossed it is the perimeter, 4.
        assert tsp_two_opt(square, [2, 0, 1, 3, 2]) == (4, [2, 1, 0, 3, 2])

    def test_tuple_tour_is_accepted(self) -> None:
        assert tsp_two_opt(LINE, (0, 1, 2, 3, 4, 5, 0)) == (48, [0, 4, 2, 1, 3, 5, 0])
        assert tour_cost(FOUR_CITY, (2, 3, 1, 0, 2)) == 30 + 25 + 10 + 15


# ----------------------------------------------------------------------
# is_valid_tour and tour_cost
# ----------------------------------------------------------------------
class TestIsValidTour:
    @pytest.mark.parametrize(
        "tour, n",
        [
            ([0, 1, 3, 2, 0], 4),
            ([2, 0, 1, 2], 3),
            ((1, 0, 1), 2),
            ([0, 0], 1),
        ],
    )
    def test_valid(self, tour, n: int) -> None:
        assert is_valid_tour(tour, n) is True

    @pytest.mark.parametrize(
        "tour, n",
        [
            ([0, 1, 3, 2], 4),  # not closed
            ([0, 1, 3, 2, 1], 4),  # ends at a different city
            ([0, 1, 1, 2, 0], 4),  # repeats a city
            ([0, 1, 2, 0], 4),  # misses a city
            ([0, 1, 2, 4, 0], 4),  # city out of range
            ([0, -1, 2, 3, 0], 4),  # negative city
            ([0, 1, 2, 3, 0], 3),  # wrong n
            ([0], 1),  # the Week 6 one-city format is not closed
            ([], 1),
            ([0, True, 0], 2),  # bool is not a city
            ([0, 1.0, 0], 2),  # float is not a city
            ("0120", 3),  # not a list or tuple
            (None, 1),
        ],
    )
    def test_invalid(self, tour, n: int) -> None:
        assert is_valid_tour(tour, n) is False

    def test_bad_n(self) -> None:
        with pytest.raises(ValueError):
            is_valid_tour([0, 0], 0)
        with pytest.raises(TypeError):
            is_valid_tour([0, 0], 1.0)
        with pytest.raises(TypeError):
            is_valid_tour([0, 0], True)


class TestTourCostErrors:
    @pytest.mark.parametrize("tour", [[], [0], [0, 1, 3, 2], [0, 4, 0], [0, -1, 0]])
    def test_value_errors(self, tour: List[int]) -> None:
        with pytest.raises(ValueError):
            tour_cost(FOUR_CITY, tour)

    @pytest.mark.parametrize("tour", [[0, True, 0], [0, 1.5, 0], [0, "1", 0]])
    def test_type_errors(self, tour) -> None:
        with pytest.raises(TypeError):
            tour_cost(FOUR_CITY, tour)


# ----------------------------------------------------------------------
# Input validation of the solvers
# ----------------------------------------------------------------------
BAD_MATRICES = [
    pytest.param([], ValueError, id="empty"),
    pytest.param([[0, 1], [1]], ValueError, id="not-square"),
    pytest.param([[0, -1], [1, 0]], ValueError, id="negative"),
    pytest.param([[0, math.nan], [1, 0]], ValueError, id="nan"),
    pytest.param([[0, "1"], [1, 0]], TypeError, id="string-entry"),
    pytest.param([[0, True], [1, 0]], TypeError, id="bool-entry"),
    pytest.param([1, 2], TypeError, id="rows-not-iterable"),
    pytest.param(5, TypeError, id="not-a-matrix"),
]


class TestValidation:
    @pytest.mark.parametrize("dist, error", BAD_MATRICES)
    def test_nearest_neighbor_rejects_bad_matrix(self, dist, error) -> None:
        with pytest.raises(error):
            tsp_nearest_neighbor(dist)

    @pytest.mark.parametrize("dist, error", BAD_MATRICES)
    def test_best_start_rejects_bad_matrix(self, dist, error) -> None:
        with pytest.raises(error):
            tsp_nearest_neighbor_best_start(dist)

    @pytest.mark.parametrize("dist, error", BAD_MATRICES)
    def test_two_opt_rejects_bad_matrix(self, dist, error) -> None:
        with pytest.raises(error):
            tsp_two_opt(dist, [0, 1, 0])

    @pytest.mark.parametrize(
        "start, error",
        [(4, ValueError), (-1, ValueError), (1.0, TypeError), (True, TypeError), ("0", TypeError)],
    )
    def test_nearest_neighbor_rejects_bad_start(self, start, error) -> None:
        with pytest.raises(error):
            tsp_nearest_neighbor(FOUR_CITY, start=start)

    @pytest.mark.parametrize(
        "tour",
        [
            [0, 1, 2, 3],
            [0, 1, 3, 2, 1],
            [0, 1, 1, 2, 0],
            [0, 1, 2, 0],
            [0, 1, 2, 4, 0],
            [0, 1.0, 3, 2, 0],
            "01320",
        ],
    )
    def test_two_opt_rejects_bad_tour(self, tour) -> None:
        with pytest.raises(ValueError):
            tsp_two_opt(FOUR_CITY, tour)

    @pytest.mark.parametrize(
        "max_passes, error", [(-1, ValueError), (1.5, TypeError), (True, TypeError)]
    )
    def test_two_opt_rejects_bad_max_passes(self, max_passes, error) -> None:
        with pytest.raises(error):
            tsp_two_opt(FOUR_CITY, [0, 1, 3, 2, 0], max_passes=max_passes)

    def test_accepts_inf_big_ints_and_other_real_types(self) -> None:
        # Row 0 is plain ints and floats (inf included); rows 1 and 2 hold
        # Fractions, which are real numbers and are converted to float.
        # From 0: 1 beats inf -> city 1; from 1: city 2 (2); home 1/2.
        dist = [[0, 1, math.inf], [Fraction(3, 2), 0, 2], [Fraction(1, 2), 4, 0]]
        assert tsp_nearest_neighbor(dist) == (3.5, [0, 1, 2, 0])
        big = [[0, 10**30], [10**30, 0]]
        assert tsp_nearest_neighbor(big) == (2 * 10**30, [0, 1, 0])

    @pytest.mark.parametrize(
        "dist, message",
        [
            ([[0.0, 1.0], [-0.5, 0.0]], r"dist\[1\]\[0\] must be >= 0, got -0.5"),
            ([[0.0, 1.0, 2.0], [1.0, 0.0, math.nan], [2.0, 1.0, 0.0]], r"dist\[1\]\[2\]"),
        ],
    )
    def test_bad_entry_is_reported_at_its_position(self, dist, message) -> None:
        with pytest.raises(ValueError, match=message):
            tsp_nearest_neighbor(dist)

    def test_inputs_are_not_mutated(self) -> None:
        dist = copy.deepcopy(LINE)
        tour = [0, 1, 2, 3, 4, 5, 0]
        tsp_nearest_neighbor(dist)
        tsp_nearest_neighbor_best_start(dist)
        tsp_two_opt(dist, tour)
        tour_cost(dist, tour)
        assert dist == LINE
        assert tour == [0, 1, 2, 3, 4, 5, 0]


# ----------------------------------------------------------------------
# euclidean_instance
# ----------------------------------------------------------------------
class TestEuclideanInstance:
    def test_reproducible_with_a_seed(self) -> None:
        assert euclidean_instance(8, 123) == euclidean_instance(8, 123)
        assert euclidean_instance(8, 123) != euclidean_instance(8, 124)

    def test_first_point_comes_from_make_rng(self) -> None:
        # make_rng(42).random() is 0.6394267984578837 (documented in
        # random_utils), and the next draw of that generator is
        # 0.025010755222666936; each point is (size * u, size * v).
        points, _ = euclidean_instance(3, 42)
        assert points[0] == (100.0 * 0.6394267984578837, 100.0 * 0.025010755222666936)

    def test_symmetric_zero_diagonal_euclidean(self) -> None:
        points, matrix = euclidean_instance(12, derive_seed(MASTER_SEED, "shape"))
        assert len(points) == 12
        assert len(matrix) == 12 and all(len(row) == 12 for row in matrix)
        for i in range(12):
            assert matrix[i][i] == 0
            for j in range(12):
                assert matrix[i][j] == matrix[j][i]
                (xi, yi), (xj, yj) = points[i], points[j]
                assert matrix[i][j] == pytest.approx(math.hypot(xi - xj, yi - yj))

    @pytest.mark.parametrize("size", [1.0, 100.0, 7])
    def test_points_lie_in_the_half_open_square(self, size) -> None:
        points, _ = euclidean_instance(200, derive_seed(MASTER_SEED, "square", size), size=size)
        for x, y in points:
            assert 0 <= x < size
            assert 0 <= y < size

    def test_size_scales_the_points(self) -> None:
        seed = derive_seed(MASTER_SEED, "scale")
        small, _ = euclidean_instance(6, seed, size=1.0)
        large, _ = euclidean_instance(6, seed, size=100.0)
        for (xs, ys), (xl, yl) in zip(small, large):
            assert xl == pytest.approx(100.0 * xs)
            assert yl == pytest.approx(100.0 * ys)

    def test_one_point(self) -> None:
        points, matrix = euclidean_instance(1, 7)
        assert len(points) == 1
        assert matrix == [[0.0]]

    def test_does_not_touch_the_global_random_state(self) -> None:
        state = random.getstate()
        euclidean_instance(50, 3)
        assert random.getstate() == state

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"n": 0, "seed": 1}, ValueError),
            ({"n": -3, "seed": 1}, ValueError),
            ({"n": 2.0, "seed": 1}, TypeError),
            ({"n": True, "seed": 1}, TypeError),
            ({"n": 3, "seed": 1, "size": 0}, ValueError),
            ({"n": 3, "seed": 1, "size": -1.0}, ValueError),
            ({"n": 3, "seed": 1, "size": math.inf}, ValueError),
            ({"n": 3, "seed": 1, "size": math.nan}, ValueError),
            ({"n": 3, "seed": 1, "size": "1"}, TypeError),
            ({"n": 3, "seed": 1, "size": True}, TypeError),
            ({"n": 3, "seed": "1"}, TypeError),
            ({"n": 3, "seed": True}, TypeError),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, error) -> None:
        with pytest.raises(error):
            euclidean_instance(**kwargs)


# ----------------------------------------------------------------------
# Seeded batteries against the exact optimum
# ----------------------------------------------------------------------
EUCLIDEAN_BATTERY = [
    pytest.param(n, rep, id=f"n{n}-rep{rep}") for n in range(3, 10) for rep in range(3)
]


class TestEuclideanBattery:
    """Seeded Euclidean instances with n <= 9, checked against brute force."""

    @pytest.mark.parametrize("n, rep", EUCLIDEAN_BATTERY)
    def test_heuristics_are_valid_and_never_beat_the_optimum(self, n: int, rep: int) -> None:
        _, dist = euclidean_instance(n, derive_seed(MASTER_SEED, "euclid", n, rep))
        optimum, _ = tsp_brute_force(dist)

        nn_costs = []
        for start in range(n):
            cost, tour = tsp_nearest_neighbor(dist, start=start)
            assert_closed_permutation(tour, n, start)
            assert is_valid_tour(tour, n)
            assert cost == pytest.approx(edge_sum(dist, tour), abs=TOL)
            assert cost >= optimum - TOL
            nn_costs.append(cost)

        best_cost, best_tour = tsp_nearest_neighbor_best_start(dist)
        assert_closed_permutation(best_tour, n, 0)
        assert best_cost == pytest.approx(edge_sum(dist, best_tour), abs=TOL)
        assert best_cost >= optimum - TOL
        assert best_cost <= min(nn_costs) + TOL
        assert best_cost <= nn_costs[0]

    @pytest.mark.parametrize("n, rep", EUCLIDEAN_BATTERY)
    def test_two_opt_never_increases_cost(self, n: int, rep: int) -> None:
        _, dist = euclidean_instance(n, derive_seed(MASTER_SEED, "euclid", n, rep))
        optimum, _ = tsp_brute_force(dist)

        nn_cost, nn_tour = tsp_nearest_neighbor(dist)
        cost, tour = tsp_two_opt(dist, nn_tour)
        assert_closed_permutation(tour, n, 0)
        assert cost <= nn_cost
        assert cost <= tour_cost(dist, nn_tour)
        assert cost <= edge_sum(dist, nn_tour) + TOL
        assert cost == pytest.approx(edge_sum(dist, tour), abs=TOL)
        assert cost >= optimum - TOL
        assert_two_optimal(dist, tour)

        start = n - 1
        scrambled = random_closed_tour(n, start, derive_seed(MASTER_SEED, "scramble", n, rep))
        cost, tour = tsp_two_opt(dist, scrambled)
        assert_closed_permutation(tour, n, start)
        assert cost <= tour_cost(dist, scrambled)
        assert cost <= edge_sum(dist, scrambled) + TOL
        assert cost >= optimum - TOL
        assert_two_optimal(dist, tour)


ASYMMETRIC_BATTERY = [
    pytest.param(n, rep, id=f"n{n}-rep{rep}") for n in range(3, 8) for rep in range(3)
]


class TestAsymmetricBattery:
    """Integer asymmetric instances: exact arithmetic, direction matters."""

    @pytest.mark.parametrize("n, rep", ASYMMETRIC_BATTERY)
    def test_against_brute_force(self, n: int, rep: int) -> None:
        dist = asymmetric_matrix(n, derive_seed(MASTER_SEED, "asym", n, rep))
        optimum, _ = tsp_brute_force(dist)

        nn_cost, nn_tour = tsp_nearest_neighbor(dist)
        assert_closed_permutation(nn_tour, n, 0)
        assert nn_cost == edge_sum(dist, nn_tour)
        assert nn_cost >= optimum

        best_cost, best_tour = tsp_nearest_neighbor_best_start(dist)
        assert_closed_permutation(best_tour, n, 0)
        assert best_cost == edge_sum(dist, best_tour)
        assert optimum <= best_cost <= nn_cost

        cost, tour = tsp_two_opt(dist, nn_tour)
        assert_closed_permutation(tour, n, 0)
        assert cost == edge_sum(dist, tour)
        assert optimum <= cost <= nn_cost
        assert_two_optimal(dist, tour)


class TestPassesAndScale:
    def test_max_passes_is_monotone(self) -> None:
        _, dist = euclidean_instance(60, derive_seed(MASTER_SEED, "passes"))
        _, start_tour = tsp_nearest_neighbor(dist)
        assert tsp_two_opt(dist, start_tour, max_passes=0) == (
            tour_cost(dist, start_tour),
            start_tour,
        )
        costs = [tsp_two_opt(dist, start_tour, max_passes=k)[0] for k in range(6)]
        full_cost, full_tour = tsp_two_opt(dist, start_tour)
        assert all(later <= earlier for earlier, later in zip(costs, costs[1:]))
        assert full_cost <= costs[-1]
        assert_closed_permutation(full_tour, 60, 0)

    def test_larger_euclidean_instance(self) -> None:
        n = 150
        _, dist = euclidean_instance(n, derive_seed(MASTER_SEED, "large"))
        nn_cost, nn_tour = tsp_nearest_neighbor(dist)
        assert is_valid_tour(nn_tour, n)
        assert nn_cost == pytest.approx(edge_sum(dist, nn_tour), abs=TOL)
        cost, tour = tsp_two_opt(dist, nn_tour)
        assert is_valid_tour(tour, n)
        assert tour[0] == 0
        assert cost <= nn_cost
