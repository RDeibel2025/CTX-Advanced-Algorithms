"""Tests for MST doubling, the networkx Christofides comparison and is_metric.

A metric-TSP approximation can look right while being wrong: the tree can
be a spanning tree but not a minimum one (the 2 * OPT proof then breaks),
the preorder can visit children in the wrong order, or the tour can be
missing a city. Every claim here is held against a witness that shares no
mechanism with the code under test.

* **Hand-traced anchors (WEEK9_BUILD_SPEC.md section 10.3).**

  - Unit square corners (0,0), (1,0), (1,1), (0,1). Prim from 0: keys
    1 -> 1, 2 -> sqrt 2, 3 -> 1; take 1 (lowest key, lowest index), which
    lowers 2's key to 1 (parent 1) and leaves 3 at 1 (sqrt 2 is not
    smaller); take 2, whose edge to 3 (1) does not strictly beat 3's key;
    take 3 (parent 0). Tree (0,1), (1,2), (0,3), weight 3.0. Preorder from
    0 with children in increasing order: 0, 1, 2, 3. Tour [0, 1, 2, 3, 0],
    cost 4.0, the perimeter, which is optimal (any tour of four corners
    has four edges of length at least 1).
  - Points on a line at x = 0, 1, -2, 4, -8, 16, distances |a - b|. Prim
    from 0 takes 1 (key 1), 2 (key 2, from 0), 3 (key 3, from 1), 4
    (key 6, from 2), 5 (key 12, from 3). Weight 1 + 2 + 3 + 6 + 12 = 24.
    Preorder: 0, then child 1 -> 3 -> 5, then child 2 -> 4. Tour
    [0, 1, 3, 5, 2, 4, 0], cost 1 + 3 + 12 + 18 + 6 + 8 = 48. Any tour
    must cross every gap between neighbouring points twice, so
    OPT = 2 * (16 - (-8)) = 48 (nearest neighbor gave 62 in Week 8).
  - Christofides on the square: every MST is a path along three sides,
    its two end vertices are the odd ones and they are adjacent, so the
    matching adds one side and the tour is the perimeter, 4.0.

* **Week 6's Held-Karp solver** (:func:`tsp_bitmask`) gives OPT on seeded
  Euclidean instances with n <= 9, and the bounds of the proofs are
  checked against it: w(MST) <= OPT <= cost <= 2 * w(MST) for MST
  doubling and OPT <= cost <= 1.5 * OPT for Christofides.
* **Week 8's verifier** (:func:`is_valid_tour`, :func:`tour_cost`) checks
  that every tour is a closed permutation and that the reported cost is
  the tour's cost.
* **networkx's minimum spanning tree**, which shares no code with our
  Prim, gives the MST weight on the same seeded instances.

Every random instance comes from Week 8's :func:`euclidean_instance` with
a seed from :func:`derive_seed`. No assertion is about time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from typing import List

import networkx as nx
import pytest

from src.approximation.tsp_metric_approx import (
    is_metric,
    mst_weight_and_edges,
    tsp_christofides_networkx,
    tsp_mst_doubling,
)
from src.complexity.traveling_salesman_approx import (
    euclidean_instance,
    is_valid_tour,
    tour_cost,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask
from src.utils.random_utils import derive_seed

MASTER_SEED = 9
TOL = 1e-9

SQUARE_POINTS = [(0, 0), (1, 0), (1, 1), (0, 1)]
LINE_XS = [0, 1, -2, 4, -8, 16]

#: (n, rep) pairs for the seeded Euclidean property tests, n <= 9.
EUCLIDEAN_CASES = [(n, rep) for n in range(2, 10) for rep in range(4)]


def square_matrix() -> List[List[float]]:
    """Distance matrix of the unit square's corners."""
    return [[math.dist(p, q) for q in SQUARE_POINTS] for p in SQUARE_POINTS]


def line_matrix() -> List[List[int]]:
    """Distance matrix |a - b| of the six points on a line."""
    return [[abs(a - b) for b in LINE_XS] for a in LINE_XS]


def euclidean(n: int, rep: int) -> List[List[float]]:
    """Seeded Euclidean distance matrix from Week 8's generator."""
    _, dist = euclidean_instance(n, derive_seed(MASTER_SEED, "tsp_metric", n, rep))
    return dist


def is_spanning_tree(n: int, edges) -> bool:
    """Whether ``edges`` is a spanning tree of ``range(n)`` (union-find)."""
    if len(edges) != n - 1:
        return False
    root = list(range(n))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        root[ra] = rb
    return True


class TestSquareAnchor:
    """Unit square corners: MST 3.0, tour [0, 1, 2, 3, 0], cost 4.0 = OPT."""

    def test_mst_weight_and_edges(self):
        weight, edges = mst_weight_and_edges(square_matrix())
        assert weight == 3.0
        assert edges == [(0, 1), (1, 2), (0, 3)]

    def test_mst_doubling_tour(self):
        cost, tour = tsp_mst_doubling(square_matrix())
        assert tour == [0, 1, 2, 3, 0]
        assert cost == 4.0

    def test_cost_equals_held_karp_optimum(self):
        opt, _ = tsp_bitmask(square_matrix())
        assert opt == 4.0
        cost, _ = tsp_mst_doubling(square_matrix())
        assert cost == opt

    def test_christofides_is_the_perimeter(self):
        cost, tour = tsp_christofides_networkx(square_matrix())
        assert cost == pytest.approx(4.0)
        assert tour[0] == tour[-1] == 0
        assert is_valid_tour(tour, 4)

    def test_square_is_metric(self):
        assert is_metric(square_matrix())


class TestLineAnchor:
    """Points on a line at 0, 1, -2, 4, -8, 16: MST 24, tour cost 48 = OPT."""

    def test_mst_weight(self):
        weight, edges = mst_weight_and_edges(line_matrix())
        assert weight == 24
        assert sorted(tuple(sorted(e)) for e in edges) == [
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5)
        ]

    def test_mst_doubling_tour_and_cost(self):
        cost, tour = tsp_mst_doubling(line_matrix())
        assert tour == [0, 1, 3, 5, 2, 4, 0]
        assert cost == 48

    def test_cost_equals_optimum_48(self):
        opt, _ = tsp_bitmask(line_matrix())
        assert opt == 48
        assert tsp_mst_doubling(line_matrix())[0] == opt

    def test_line_is_metric(self):
        assert is_metric(line_matrix())


class TestSeededEuclidean:
    """Seeded Euclidean instances, n <= 9, checked against Held-Karp."""

    @pytest.mark.parametrize("n,rep", EUCLIDEAN_CASES)
    def test_mst_doubling_bounds(self, n, rep):
        dist = euclidean(n, rep)
        opt, _ = tsp_bitmask(dist)
        mst_weight, edges = mst_weight_and_edges(dist)
        cost, tour = tsp_mst_doubling(dist)

        assert is_spanning_tree(n, edges)
        assert mst_weight == pytest.approx(sum(dist[a][b] for a, b in edges))
        assert is_valid_tour(tour, n)
        assert tour[0] == tour[-1] == 0
        assert cost == pytest.approx(tour_cost(dist, tour))
        assert mst_weight <= opt + TOL
        assert opt <= cost + TOL
        assert cost <= 2 * mst_weight + TOL

    @pytest.mark.parametrize("n,rep", EUCLIDEAN_CASES)
    def test_mst_weight_matches_networkx(self, n, rep):
        dist = euclidean(n, rep)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_weighted_edges_from(
            (i, j, dist[i][j]) for i in range(n) for j in range(i + 1, n)
        )
        expected = nx.minimum_spanning_tree(graph).size(weight="weight")
        assert mst_weight_and_edges(dist)[0] == pytest.approx(expected)

    @pytest.mark.parametrize("n,rep", EUCLIDEAN_CASES)
    def test_christofides_within_three_halves(self, n, rep):
        dist = euclidean(n, rep)
        opt, _ = tsp_bitmask(dist)
        cost, tour = tsp_christofides_networkx(dist)
        assert is_valid_tour(tour, n)
        assert tour[0] == tour[-1] == 0
        assert cost == pytest.approx(tour_cost(dist, tour))
        assert opt <= cost + TOL
        assert cost <= 1.5 * opt + TOL

    @pytest.mark.parametrize("n", [3, 6, 9])
    def test_every_start_keeps_the_bound(self, n):
        dist = euclidean(n, 0)
        mst_weight, _ = mst_weight_and_edges(dist)
        for start in range(n):
            cost, tour = tsp_mst_doubling(dist, start=start)
            assert is_valid_tour(tour, n)
            assert tour[0] == tour[-1] == start
            assert cost <= 2 * mst_weight + TOL

    @pytest.mark.parametrize("n,rep", EUCLIDEAN_CASES)
    def test_euclidean_is_metric(self, n, rep):
        assert is_metric(euclidean(n, rep))


class TestIsMetric:
    """is_metric is False whenever one metric axiom fails."""

    def test_triangle_violation(self):
        # 0 -> 1 -> 2 costs 1 + 1 = 2 < 5 = dist[0][2].
        assert not is_metric([[0, 1, 5], [1, 0, 1], [5, 1, 0]])

    def test_asymmetric(self):
        assert not is_metric([[0, 1], [2, 0]])

    def test_nonzero_diagonal(self):
        assert not is_metric([[1, 1], [1, 0]])

    def test_infinite_entry(self):
        assert not is_metric([[0, math.inf], [math.inf, 0]])

    def test_sahni_gonzalez_gadget_is_not_metric(self):
        # Edges of a 4-cycle cost 1, the two non-edges cost a * n + 1 = 9
        # for a = 2: the matrix behind the inapproximability proof.
        heavy = 9
        dist = [[0, 1, heavy, 1], [1, 0, 1, heavy],
                [heavy, 1, 0, 1], [1, heavy, 1, 0]]
        assert not is_metric(dist)

    def test_not_square_raises(self):
        with pytest.raises(ValueError):
            is_metric([[0, 1], [1]])


class TestSmallAndInvalidInput:
    """Trivial sizes and the input checks."""

    def test_one_city(self):
        assert mst_weight_and_edges([[0]]) == (0, [])
        assert tsp_mst_doubling([[0]]) == (0, [0, 0])
        assert tsp_christofides_networkx([[0]]) == (0, [0, 0])

    def test_two_cities(self):
        dist = [[0, 7], [7, 0]]
        assert tsp_mst_doubling(dist) == (14, [0, 1, 0])
        assert tsp_christofides_networkx(dist) == (14, [0, 1, 0])

    def test_asymmetric_matrix_rejected(self):
        with pytest.raises(ValueError):
            tsp_mst_doubling([[0, 1], [2, 0]])

    def test_start_out_of_range(self):
        with pytest.raises(ValueError):
            tsp_mst_doubling(line_matrix(), start=6)
