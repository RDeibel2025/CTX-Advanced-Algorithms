"""Week 9 tests for vertex cover: the 2-approximation, LP rounding and the ratio helper.

Every expected value below is a hand-checked anchor from WEEK9_BUILD_SPEC.md
section 10.1, never the output of the code under test:

* **Triangle K3**, vertices 0-2, sorted edges (0,1), (0,2), (1,2).
  The LP point x = (1/2, 1/2, 1/2) meets every edge with equality, and
  adding the three constraints gives 2 (x0 + x1 + x2) >= 3, so LP = 1.5
  with that x the unique optimum (equality forces every constraint tight).
  Rounding at 1/2 keeps all 3 vertices: ratio 3 / 2 = 1.5 to OPT and
  3 / 1.5 = 2.0 to LP. One vertex misses the opposite edge, two suffice, so
  OPT = 2. The 2-approximation picks (0,1), which already covers (0,2) and
  (1,2), and returns {0, 1}.
* **Star**, centre 0, leaves 1-5. With x0 = 1 - t every leaf needs
  x_i >= t, so the LP cost is 1 + 4t, minimized only at t = 0: LP = 1.0,
  x0 = 1, leaves 0, and rounding returns {0}. The 2-approximation picks
  (0,1) and returns {0, 1}; OPT = {0}, size 1.
* **Path 1-2-3-4** (Week 8 anchor): the 2-approximation picks (1,2) and
  (3,4), size 4; {2, 3} is a cover and no single vertex touches both (1,2)
  and (3,4), so OPT = 2; the matching lower bound is |{(1,2), (3,4)}| = 2.
* **50 seeded random graphs**, n <= 14, from ``graph_utils.random_graph``
  with seeds from ``derive_seed``. The relations asserted are the theorems
  in the module docstring: both methods return covers, each at most
  2 x OPT, LP <= OPT <= 2-approx size, matching bound <= OPT (plus
  |M| <= LP, |C_round| <= 2 LP and |C_2approx| = 2 |M|). OPT is checked
  against an independent bitmask brute force, so the reference does not
  depend on the code under test.

No assertion is about time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from itertools import combinations
from typing import Hashable, Iterable, List, Set, Tuple

import pytest

from src.approximation.vertex_cover_2approx import (
    approximation_ratio,
    matching_lower_bound,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_lp,
    vertex_cover_lp_rounding,
)
from src.utils.graph_utils import is_vertex_cover, random_graph
from src.utils.random_utils import derive_seed

MASTER_SEED = 9
TOL = 1e-9

TRIANGLE_V = [0, 1, 2]
TRIANGLE_E = sorted({(0, 1), (0, 2), (1, 2)})
STAR_V = list(range(6))
STAR_E = [(0, leaf) for leaf in range(1, 6)]
PATH_V = [1, 2, 3, 4]
PATH_E = [(1, 2), (2, 3), (3, 4)]


def brute_force_min_cover_size(vertices: Iterable[Hashable], edges: Iterable[Tuple[int, int]]) -> int:
    """Smallest k such that some k-subset of vertices covers every edge.

    An independent witness: plain subset enumeration with bitmasks, sharing
    no code with the branch-and-bound exact solver.
    """
    order = sorted(vertices)
    bit = {v: 1 << i for i, v in enumerate(order)}
    edge_masks = [(bit[u], bit[v]) for u, v in edges]
    if not edge_masks:
        return 0
    for k in range(1, len(order) + 1):
        for subset in combinations(order, k):
            mask = 0
            for v in subset:
                mask |= bit[v]
            if all((mask & a) or (mask & b) for a, b in edge_masks):
                return k
    raise AssertionError("unreachable: the full vertex set is always a cover")


def seeded_graphs() -> List[Tuple[int, List[int], Set[Tuple[int, int]]]]:
    """The 50 seeded random graphs: n cycles through 4..14, p through four densities."""
    densities = [0.2, 0.35, 0.5, 0.7]
    graphs = []
    for i in range(50):
        n = 4 + i % 11
        p = densities[i % len(densities)]
        vertices, edges = random_graph(n, p, derive_seed(MASTER_SEED, "week9-vertex-cover", i))
        graphs.append((i, vertices, edges))
    return graphs


class TestTriangleAnchor:
    def test_lp_value_and_point(self) -> None:
        value, x = vertex_cover_lp(TRIANGLE_V, TRIANGLE_E)
        assert value == pytest.approx(1.5, abs=TOL)
        assert set(x) == {0, 1, 2}
        for v in TRIANGLE_V:
            assert x[v] == pytest.approx(0.5, abs=TOL)

    def test_lp_rounding_returns_all_three(self) -> None:
        cover, lp_value = vertex_cover_lp_rounding(TRIANGLE_V, TRIANGLE_E)
        assert cover == {0, 1, 2}
        assert lp_value == pytest.approx(1.5, abs=TOL)
        assert approximation_ratio(len(cover), 2) == pytest.approx(1.5)
        assert approximation_ratio(len(cover), lp_value) == pytest.approx(2.0)

    def test_exact_is_two(self) -> None:
        exact = vertex_cover_exact(TRIANGLE_V, TRIANGLE_E)
        assert len(exact) == 2
        assert is_vertex_cover(TRIANGLE_E, exact)

    def test_two_approx_picks_first_edge(self) -> None:
        assert vertex_cover_2approx(TRIANGLE_V, TRIANGLE_E) == {0, 1}
        assert matching_lower_bound(TRIANGLE_V, TRIANGLE_E) == 1


class TestStarAnchor:
    def test_lp_value_and_point(self) -> None:
        value, x = vertex_cover_lp(STAR_V, STAR_E)
        assert value == pytest.approx(1.0, abs=TOL)
        assert x[0] == pytest.approx(1.0, abs=TOL)
        for leaf in range(1, 6):
            assert x[leaf] == pytest.approx(0.0, abs=TOL)

    def test_lp_rounding_returns_centre(self) -> None:
        cover, lp_value = vertex_cover_lp_rounding(STAR_V, STAR_E)
        assert cover == {0}
        assert lp_value == pytest.approx(1.0, abs=TOL)

    def test_two_approx_and_exact(self) -> None:
        assert vertex_cover_2approx(STAR_V, STAR_E) == {0, 1}
        exact = vertex_cover_exact(STAR_V, STAR_E)
        assert len(exact) == 1
        assert exact == {0}


class TestPathAnchor:
    def test_two_approx_size_four(self) -> None:
        cover = vertex_cover_2approx(PATH_V, PATH_E)
        assert len(cover) == 4
        assert cover == {1, 2, 3, 4}

    def test_exact_size_two(self) -> None:
        assert len(vertex_cover_exact(PATH_V, PATH_E)) == 2

    def test_matching_lower_bound_two(self) -> None:
        assert matching_lower_bound(PATH_V, PATH_E) == 2


class TestSeededRandomGraphs:
    @pytest.mark.parametrize("index, vertices, edges", seeded_graphs())
    def test_guarantees(self, index: int, vertices: List[int], edges: Set[Tuple[int, int]]) -> None:
        assert len(vertices) <= 14
        approx = vertex_cover_2approx(vertices, edges)
        rounded, lp_value = vertex_cover_lp_rounding(vertices, edges)
        lp_only, x = vertex_cover_lp(vertices, edges)
        exact = vertex_cover_exact(vertices, edges)
        bound = matching_lower_bound(vertices, edges)
        opt = len(exact)

        # The reference optimum, confirmed by an independent brute force.
        assert is_vertex_cover(edges, exact)
        assert opt == brute_force_min_cover_size(vertices, edges)

        # Both methods return covers.
        assert is_vertex_cover(edges, approx)
        assert is_vertex_cover(edges, rounded)

        # Size <= 2 x exact for both methods.
        assert len(approx) <= 2 * opt
        assert len(rounded) <= 2 * opt

        # LP value <= exact size <= 2-approx size.
        assert lp_value == pytest.approx(lp_only, abs=TOL)
        assert lp_value <= opt + TOL
        assert opt <= len(approx)

        # Matching lower bound <= exact size (and the proof's other links).
        assert bound <= opt
        assert bound <= lp_value + TOL
        assert len(approx) == 2 * bound
        assert len(rounded) <= 2 * lp_value + TOL

        # The LP point is feasible and attains the reported value.
        assert set(x) == set(vertices)
        assert all(-TOL <= x[v] <= 1 + TOL for v in vertices)
        assert all(x[u] + x[v] >= 1 - 1e-7 for u, v in edges)
        assert sum(x.values()) == pytest.approx(lp_value, abs=1e-7)

    def test_seeded_order_still_covers(self) -> None:
        for i in range(5):
            vertices, edges = random_graph(12, 0.4, derive_seed(MASTER_SEED, "week9-vc-order", i))
            seed = derive_seed(MASTER_SEED, "week9-vc-shuffle", i)
            first = vertex_cover_2approx(vertices, edges, seed=seed)
            assert first == vertex_cover_2approx(vertices, edges, seed=seed)
            assert is_vertex_cover(edges, first)
            assert len(first) <= 2 * brute_force_min_cover_size(vertices, edges)


class TestEdgeCases:
    def test_edgeless_graph(self) -> None:
        assert vertex_cover_lp([0, 1, 2], set()) == (0.0, {0: 0.0, 1: 0.0, 2: 0.0})
        assert vertex_cover_lp_rounding([0, 1, 2], set()) == (set(), 0.0)
        assert vertex_cover_2approx([0, 1, 2], set()) == set()
        assert matching_lower_bound([0, 1, 2], set()) == 0

    def test_endpoint_outside_vertices_rejected(self) -> None:
        with pytest.raises(ValueError):
            vertex_cover_lp([0, 1], [(0, 2)])
        with pytest.raises(ValueError):
            vertex_cover_lp_rounding([0, 1], [(0, 2)])


class TestApproximationRatio:
    def test_plain_division(self) -> None:
        assert approximation_ratio(4, 2) == 2.0
        assert approximation_ratio(3, 1.5) == 2.0
        assert approximation_ratio(3, 4) == 0.75

    def test_zero_reference(self) -> None:
        assert approximation_ratio(0, 0) == 1.0
        with pytest.raises(ValueError):
            approximation_ratio(1, 0)

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValueError):
            approximation_ratio(-1, 2)
        with pytest.raises(ValueError):
            approximation_ratio(2, -1)

    def test_non_finite_and_non_numbers_rejected(self) -> None:
        with pytest.raises(ValueError):
            approximation_ratio(float("nan"), 1)
        with pytest.raises(ValueError):
            approximation_ratio(1, float("inf"))
        with pytest.raises(TypeError):
            approximation_ratio(True, 1)
        with pytest.raises(TypeError):
            approximation_ratio("2", 1)
