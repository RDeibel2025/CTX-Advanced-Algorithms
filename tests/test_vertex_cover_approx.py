"""Tests for the Vertex Cover 2-approximation, max-degree greedy and the ratio helper.

A vertex cover routine can be wrong in quiet ways. A walk that adds only
one endpoint of an uncovered edge still returns a cover, but loses the
factor-2 proof; a walk that forgets to skip covered edges still returns a
cover, but one far above 2 OPT; a "greedy" that never re-reads degrees
still returns plausible sets. So every claim here rests on a witness that
shares no mechanism with the code under test:

* **The spec anchors (WEEK8_BUILD_SPEC.md section 10.4), traced by hand.**
  Sorted edge order, add both ends of each edge with neither end covered:

  - Path 1-2-3-4, edges (1,2), (2,3), (3,4): pick (1,2); (2,3) is covered
    by 2; pick (3,4). Cover {1, 2, 3, 4}. {2, 3} covers all three edges and
    one vertex cannot (no vertex touches both (1,2) and (3,4)), so OPT = 2
    and the ratio is 4 / 2 = 2.0.
  - Star, centre 0, leaves 1..5: pick (0,1); every other edge contains 0.
    Cover {0, 1}; OPT = {0} (size 1), ratio 2.0.
  - K3,3, left 0..2, right 3..5, sorted edges (0,3), (0,4), (0,5), (1,3),
    (1,4), (1,5), (2,3), (2,4), (2,5): pick (0,3), skip the next three
    (0 or 3 is in), pick (1,4), skip (1,5), (2,3), (2,4), pick (2,5). All
    six vertices. The three edges (0,3), (1,4), (2,5) are disjoint, so
    OPT >= 3, and the left side {0, 1, 2} is a cover: OPT = 3, ratio 2.0.

* **Brute force over every vertex subset** with :mod:`itertools`, which
  knows nothing about matchings, branching or bounds, for n <= 10.
* **A hand-proven optimum for the greedy's bad family**, where OPT is
  pinned between an explicit cover and an explicit matching of the same
  size, so no exact solver is needed (see :class:`TestMaxDegreeGreedyBadFamily`).
* **The structure of the proof itself.** The edges the 2-approximation
  picks must be pairwise disjoint (a matching M) and maximal, the cover
  must be exactly their endpoints (size 2 |M|), and OPT >= |M|. Those are
  checked with plain set arithmetic in this file.

Every random graph comes from ``graph_utils.random_graph`` with a seed
from ``derive_seed``, so nothing here can flake. No assertion is about time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Hashable, Iterable, List, Set, Tuple

import pytest

from src.complexity.reduction_demo import min_vertex_cover_exact
from src.complexity.vertex_cover_approx import (
    approximation_ratio,
    greedy_maximal_matching,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_max_degree_greedy,
)
from src.utils.graph_utils import random_graph
from src.utils.random_utils import derive_seed

#: Master seed for every seeded battery in this file.
MASTER_SEED: int = 42

#: Number of random graphs in the battery (spec section 10.4).
BATTERY_SIZE: int = 50

#: Largest n for which the exact solver is also checked by brute force.
BRUTE_FORCE_MAX_N: int = 10

#: Edge probabilities cycled through the battery: sparse to dense.
EDGE_PROBABILITIES: Tuple[float, ...] = (0.15, 0.3, 0.5, 0.8)

Graph = Tuple[List[Hashable], Set[Tuple[Hashable, Hashable]]]

# ----------------------------------------------------------------------
# Anchor graphs (spec section 10.4, plus two hand-checked extras)
# ----------------------------------------------------------------------
PATH: Graph = ([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)})
STAR: Graph = ([0, 1, 2, 3, 4, 5], {(0, leaf) for leaf in range(1, 6)})
K33: Graph = ([0, 1, 2, 3, 4, 5], {(a, b) for a in range(3) for b in range(3, 6)})
#: Root 0 with children 1, 2, 3; child c has leaves 2c + 2 and 2c + 3.
TREE: Graph = (
    list(range(10)),
    {(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (3, 9)},
)


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def covers(edges: Iterable[Tuple[Hashable, Hashable]], subset: Iterable[Hashable]) -> bool:
    """Return True if every edge has an endpoint in ``subset`` (the definition)."""
    chosen = set(subset)
    return all(u in chosen or v in chosen for u, v in edges)


def canonical(edges: Iterable[Tuple[Hashable, Hashable]]) -> Set[Tuple[Hashable, Hashable]]:
    """The edges as ``(min, max)`` pairs, for comparing picked edges with the graph."""
    return {(min(u, v), max(u, v)) for u, v in edges}


def brute_force_min_cover_size(vertices: List[Hashable], edges: Set[Tuple]) -> int:
    """Smallest k such that some k-subset of the vertices covers every edge.

    Walks subsets in increasing size with :func:`itertools.combinations`; it
    shares nothing with the module (no matching, no branching, no bound).
    """
    for size in range(len(vertices) + 1):
        for subset in combinations(vertices, size):
            if covers(edges, subset):
                return size
    raise AssertionError("the full vertex set always covers")  # pragma: no cover


def battery() -> List[Tuple[int, Graph]]:
    """The 50 seeded random graphs: n cycles through 2..14, p through EDGE_PROBABILITIES."""
    graphs = []
    for i in range(BATTERY_SIZE):
        n = 2 + i % 13
        p = EDGE_PROBABILITIES[i % len(EDGE_PROBABILITIES)]
        graphs.append((i, random_graph(n, p, derive_seed(MASTER_SEED, "vertex-cover", i))))
    return graphs


BATTERY = battery()


def battery_ids() -> List[str]:
    """Readable pytest ids: graph index, n and edge count."""
    return [f"g{i}-n{len(v)}-m{len(e)}" for i, (v, e) in BATTERY]


# ----------------------------------------------------------------------
# Spec anchors (section 10.4)
# ----------------------------------------------------------------------
class TestPathAnchor:
    """Path 1-2-3-4: 2-approx {1, 2, 3, 4}, exact size 2, ratio 2.0."""

    def test_two_approx_returns_all_four_vertices(self):
        assert vertex_cover_2approx(*PATH) == {1, 2, 3, 4}

    def test_exact_has_size_two_and_covers(self):
        exact = vertex_cover_exact(*PATH)
        assert len(exact) == 2
        assert covers(PATH[1], exact)

    def test_ratio_is_exactly_two(self):
        approx = vertex_cover_2approx(*PATH)
        exact = vertex_cover_exact(*PATH)
        assert approximation_ratio(len(approx), len(exact)) == 2.0

    def test_max_degree_greedy(self):
        # Degrees 1, 2, 2, 1: take 2 (tie with 3, smaller wins), which
        # removes (1,2) and (2,3); then (3,4) remains and 3 < 4 is taken.
        assert vertex_cover_max_degree_greedy(*PATH) == {2, 3}


class TestStarAnchor:
    """Star centre 0, leaves 1..5: 2-approx {0, 1}, exact {0}, ratio 2.0."""

    def test_two_approx_returns_centre_and_first_leaf(self):
        assert vertex_cover_2approx(*STAR) == {0, 1}

    def test_exact_is_the_centre(self):
        assert vertex_cover_exact(*STAR) == {0}

    def test_ratio_is_exactly_two(self):
        approx = vertex_cover_2approx(*STAR)
        exact = vertex_cover_exact(*STAR)
        assert approximation_ratio(len(approx), len(exact)) == 2.0

    def test_max_degree_greedy_takes_the_centre(self):
        assert vertex_cover_max_degree_greedy(*STAR) == {0}


class TestK33Anchor:
    """K3,3 (left 0..2, right 3..5): 2-approx all 6, exact 3, ratio 2.0."""

    def test_two_approx_returns_all_six_vertices(self):
        assert vertex_cover_2approx(*K33) == {0, 1, 2, 3, 4, 5}

    def test_exact_has_size_three_and_covers(self):
        exact = vertex_cover_exact(*K33)
        assert len(exact) == 3
        assert covers(K33[1], exact)

    def test_ratio_is_exactly_two(self):
        approx = vertex_cover_2approx(*K33)
        exact = vertex_cover_exact(*K33)
        assert approximation_ratio(len(approx), len(exact)) == 2.0

    def test_max_degree_greedy_takes_one_side(self):
        # All degrees 3: take 0. Then 1 and 2 still have degree 3 and the
        # right side 2: take 1, then 2 (degree 3 against 1).
        assert vertex_cover_max_degree_greedy(*K33) == {0, 1, 2}


class TestEdgeFreeGraphs:
    """No edges means nothing to cover: every method returns the empty set."""

    @pytest.mark.parametrize("vertices", [[], [0], [0, 1, 2], ["a", "b"]])
    def test_all_three_methods_return_empty(self, vertices):
        assert vertex_cover_2approx(vertices, set()) == set()
        assert vertex_cover_max_degree_greedy(vertices, set()) == set()
        assert vertex_cover_exact(vertices, set()) == set()

    def test_seeded_and_explicit_orders_on_edge_free_graph(self):
        assert vertex_cover_2approx([0, 1, 2], set(), seed=3) == set()
        assert vertex_cover_2approx([0, 1, 2], set(), order=[]) == set()
        assert greedy_maximal_matching([0, 1, 2], set()) == []

    def test_isolated_vertices_are_never_chosen(self):
        vertices = [1, 2, 3, 4, 99]
        assert 99 not in vertex_cover_2approx(vertices, PATH[1])
        assert 99 not in vertex_cover_max_degree_greedy(vertices, PATH[1])


def test_vertex_cover_exact_is_the_reduction_solver():
    assert vertex_cover_exact is min_vertex_cover_exact


# ----------------------------------------------------------------------
# The matching behind the factor-2 proof
# ----------------------------------------------------------------------
class TestPickedEdgesFormAMatching:
    """The picked edges are a maximal matching M and the cover is their 2|M| ends."""

    def test_path_matching(self):
        assert greedy_maximal_matching(*PATH) == [(1, 2), (3, 4)]

    def test_star_matching(self):
        assert greedy_maximal_matching(*STAR) == [(0, 1)]

    def test_k33_matching(self):
        assert greedy_maximal_matching(*K33) == [(0, 3), (1, 4), (2, 5)]

    def test_tree_matching(self):
        # Sorted: pick (0,1); skip (0,2), (0,3), (1,4), (1,5); pick (2,6);
        # skip (2,7); pick (3,8); skip (3,9).
        assert greedy_maximal_matching(*TREE) == [(0, 1), (2, 6), (3, 8)]
        assert vertex_cover_2approx(*TREE) == {0, 1, 2, 3, 6, 8}

    @pytest.mark.parametrize("index, graph", BATTERY, ids=battery_ids())
    def test_default_order_matching_structure(self, index, graph):
        vertices, edges = graph
        matching = greedy_maximal_matching(vertices, edges)
        cover = vertex_cover_2approx(vertices, edges)
        matched = {x for pair in matching for x in pair}

        # Every picked edge is an edge of the graph.
        assert set(matching) <= canonical(edges)
        # Pairwise disjoint: 2|M| distinct endpoints.
        assert len(matched) == 2 * len(matching)
        # Maximal: every edge touches a matched vertex.
        assert covers(edges, matched)
        # The cover is exactly the matched endpoints, so its size is even.
        assert cover == matched
        assert len(cover) % 2 == 0
        assert len(cover) == 2 * len(matching)
        # The lower bound in the proof: OPT >= |M|.
        assert len(vertex_cover_exact(vertices, edges)) >= len(matching)


# ----------------------------------------------------------------------
# The seeded battery (spec section 10.4)
# ----------------------------------------------------------------------
class TestRandomBattery:
    """50 seeded G(n, p) graphs with n <= 14."""

    def test_battery_shape(self):
        sizes = [len(v) for _, (v, _) in BATTERY]
        assert len(BATTERY) == BATTERY_SIZE
        assert max(sizes) <= 14
        assert sum(1 for n in sizes if n <= BRUTE_FORCE_MAX_N) >= 30
        assert sum(len(e) for _, (_, e) in BATTERY) > 0

    @pytest.mark.parametrize("index, graph", BATTERY, ids=battery_ids())
    def test_every_method_covers_and_two_approx_is_within_two(self, index, graph):
        vertices, edges = graph
        exact = vertex_cover_exact(vertices, edges)
        approx = vertex_cover_2approx(vertices, edges)
        greedy = vertex_cover_max_degree_greedy(vertices, edges)

        for result in (exact, approx, greedy):
            assert covers(edges, result)
            assert result <= set(vertices)
        assert len(exact) <= len(approx) <= 2 * len(exact)
        assert len(exact) <= len(greedy)

    @pytest.mark.parametrize("index, graph", BATTERY, ids=battery_ids())
    def test_seeded_orders_also_within_two(self, index, graph):
        vertices, edges = graph
        optimum = len(vertex_cover_exact(vertices, edges))
        for trial in range(3):
            seed = derive_seed(MASTER_SEED, "vertex-cover-order", index, trial)
            approx = vertex_cover_2approx(vertices, edges, seed=seed)
            assert covers(edges, approx)
            assert len(approx) <= 2 * optimum

    @pytest.mark.parametrize(
        "index, graph",
        [(i, g) for i, g in BATTERY if len(g[0]) <= BRUTE_FORCE_MAX_N],
        ids=[f"g{i}" for i, g in BATTERY if len(g[0]) <= BRUTE_FORCE_MAX_N],
    )
    def test_exact_matches_brute_force(self, index, graph):
        vertices, edges = graph
        assert len(vertex_cover_exact(vertices, edges)) == brute_force_min_cover_size(
            vertices, edges
        )


# ----------------------------------------------------------------------
# Visiting order: explicit, seeded, default
# ----------------------------------------------------------------------
class TestVisitingOrder:
    """The proof holds for every order; the order changes which cover comes back."""

    def test_explicit_order_picks_middle_edge_first(self):
        # (2,3) first covers all three path edges with one pick.
        assert vertex_cover_2approx(*PATH, order=[(2, 3), (1, 2), (3, 4)]) == {2, 3}
        assert greedy_maximal_matching(*PATH, order=[(2, 3), (1, 2), (3, 4)]) == [(2, 3)]

    def test_explicit_order_accepts_reversed_edges(self):
        assert vertex_cover_2approx(*PATH, order=[(3, 2), (2, 1), (4, 3)]) == {2, 3}

    def test_explicit_order_repeats_are_harmless(self):
        order = [(1, 2), (1, 2), (2, 3), (3, 4), (3, 4)]
        assert vertex_cover_2approx(*PATH, order=order) == {1, 2, 3, 4}

    def test_explicit_order_wins_over_seed(self):
        order = [(2, 3), (1, 2), (3, 4)]
        for seed in range(10):
            assert vertex_cover_2approx(*PATH, order=order, seed=seed) == {2, 3}

    def test_order_with_a_non_edge_raises(self):
        with pytest.raises(ValueError, match="not an edge"):
            vertex_cover_2approx(*PATH, order=[(1, 2), (2, 3), (3, 4), (1, 4)])

    def test_order_missing_an_edge_raises(self):
        with pytest.raises(ValueError, match="leaves out"):
            vertex_cover_2approx(*PATH, order=[(1, 2), (3, 4)])

    def test_order_must_be_a_collection_of_edges(self):
        with pytest.raises(TypeError):
            vertex_cover_2approx(*PATH, order="12")
        with pytest.raises(TypeError):
            vertex_cover_2approx(*PATH, order=5)
        with pytest.raises(ValueError):
            vertex_cover_2approx(*PATH, order=[(1, 2, 3)])

    def test_seed_must_be_an_int(self):
        with pytest.raises(TypeError):
            vertex_cover_2approx(*PATH, seed="7")

    def test_bad_seed_is_rejected_even_with_explicit_order(self):
        # The order wins over the seed, but a bad seed is not silently ignored.
        order = [(2, 3), (1, 2), (3, 4)]
        with pytest.raises(TypeError):
            vertex_cover_2approx(*PATH, order=order, seed="7")
        with pytest.raises(TypeError):
            greedy_maximal_matching(*PATH, order=order, seed=True)

    def test_seeded_order_is_reproducible(self):
        for index, (vertices, edges) in BATTERY[:20]:
            seed = derive_seed(MASTER_SEED, "reproducible", index)
            first = vertex_cover_2approx(vertices, edges, seed=seed)
            second = vertex_cover_2approx(vertices, edges, seed=seed)
            assert first == second
            assert greedy_maximal_matching(vertices, edges, seed=seed) == (
                greedy_maximal_matching(vertices, edges, seed=seed)
            )

    def test_seed_changes_the_order(self):
        # P4 has exactly two maximal matchings: {(1,2), (3,4)} and {(2,3)}.
        # So every order gives {1, 2, 3, 4} or {2, 3}, and over 20 seeds
        # (each puts (2,3) first with probability 1/3) both must appear.
        outcomes = {
            frozenset(vertex_cover_2approx(*PATH, seed=derive_seed(MASTER_SEED, "p4", s)))
            for s in range(20)
        }
        assert outcomes == {frozenset({1, 2, 3, 4}), frozenset({2, 3})}

    def test_default_order_is_sorted_order_for_string_labels(self):
        # Edges given reversed; sorted canonical order is (a,b), (b,c), (c,d).
        vertices = ["d", "c", "b", "a"]
        edges = {("b", "a"), ("c", "b"), ("d", "c")}
        assert greedy_maximal_matching(vertices, edges) == [("a", "b"), ("c", "d")]
        assert vertex_cover_2approx(vertices, edges) == {"a", "b", "c", "d"}


# ----------------------------------------------------------------------
# Max-degree greedy: a cover, but no constant-factor guarantee
# ----------------------------------------------------------------------
def bad_greedy_family(k: int) -> Tuple[List[int], Set[Tuple[int, int]], List[int], List[int]]:
    """The bipartite graph on which max-degree greedy is Theta(log n) times optimal.

    L = {1, ..., k}. For each block size i = 2..k and g = 0..k//i - 1, an
    extra vertex ``1000 * i + g`` is joined to the block
    ``{g*i + 1, ..., g*i + i}`` of L. Returns ``(vertices, edges, L, R)``.
    """
    left = list(range(1, k + 1))
    right: List[int] = []
    edges: Set[Tuple[int, int]] = set()
    for i in range(2, k + 1):
        for g in range(k // i):
            r = 1000 * i + g
            right.append(r)
            for t in range(g * i + 1, g * i + i + 1):
                edges.add((t, r))
    return left + right, edges, left, right


class TestMaxDegreeGreedyBadFamily:
    """k = 20: greedy returns 46 vertices; the optimum is 20 (ratio 2.3 > 2).

    Hand count: |R| = sum over i = 2..20 of floor(20 / i)
    = 10 + 6 + 5 + 4 + 3 + 2 + 2 + 2 + 2 + (ten 1s for i = 11..20) = 46.

    Why greedy takes exactly R: suppose every extra vertex with block size
    above i is already taken. An L vertex lies in at most one block of each
    size 2..i, so its degree is at most i - 1, while each block-size-i
    extra vertex still has all i edges (only R vertices have been taken,
    and R vertices are not adjacent to each other). The maximum degree is
    therefore i, attained only by block-size-i vertices, so greedy takes
    all of them next, whatever the tie rule. When i = 2 is done every edge
    is covered and no L vertex was ever taken: the cover is R, size 46.

    Why OPT = 20: L is a cover (every edge has one end in L), so OPT <= 20,
    and :data:`HAND_MATCHING` below is a matching of 20 edges, so OPT >= 20.
    """

    #: L vertex -> the extra vertex it is matched to, built by hand: odd t
    #: to its size-2 block; 2, 4, 8, 10, 14, 16 to size-3 blocks; 6, 12,
    #: 18 to size-4 blocks; 20 to the size-5 block {16..20}.
    HAND_MATCHING: Dict[int, int] = {
        **{2 * j - 1: 2000 + (j - 1) for j in range(1, 11)},
        2: 3000, 4: 3001, 8: 3002, 10: 3003, 14: 3004, 16: 3005,
        6: 4001, 12: 4002, 18: 4004,
        20: 5003,
    }

    def test_construction_size(self):
        vertices, edges, left, right = bad_greedy_family(20)
        assert len(left) == 20
        assert len(right) == 46
        assert len(vertices) == 66

    def test_optimum_is_twenty_by_cover_and_matching(self):
        _, edges, left, _ = bad_greedy_family(20)
        assert covers(edges, left)                       # OPT <= 20
        pairs = list(self.HAND_MATCHING.items())
        assert len(pairs) == 20
        assert all((t, r) in edges for t, r in pairs)    # real edges
        endpoints = [x for pair in pairs for x in pair]
        assert len(set(endpoints)) == 40                 # disjoint: OPT >= 20

    def test_greedy_takes_every_extra_vertex(self):
        vertices, edges, _, right = bad_greedy_family(20)
        greedy = vertex_cover_max_degree_greedy(vertices, edges)
        assert greedy == set(right)
        assert approximation_ratio(len(greedy), 20) == pytest.approx(2.3)
        assert len(greedy) > 2 * 20                      # beyond the factor 2

    def test_two_approx_stays_within_two(self):
        vertices, edges, _, _ = bad_greedy_family(20)
        approx = vertex_cover_2approx(vertices, edges)
        assert covers(edges, approx)
        assert len(approx) <= 2 * 20

    def test_greedy_is_not_optimal_on_small_tree(self):
        # Root 0 and children 1, 2, 3 all start at degree 3; the tie goes
        # to 0, after which 1, 2 and 3 are each still needed for their
        # leaves. {1, 2, 3} is the unique minimum cover (each child's two
        # leaf edges need the child or both leaves).
        assert vertex_cover_max_degree_greedy(*TREE) == {0, 1, 2, 3}
        assert vertex_cover_exact(*TREE) == {1, 2, 3}


# ----------------------------------------------------------------------
# approximation_ratio
# ----------------------------------------------------------------------
class TestApproximationRatio:
    """approx / optimal, with the 0 / 0 and invalid cases pinned down."""

    @pytest.mark.parametrize(
        "approx, optimal, expected",
        [(4, 2, 2.0), (1, 1, 1.0), (3, 2, 1.5), (6, 3, 2.0), (88.0, 80.0, 1.1)],
    )
    def test_values(self, approx, optimal, expected):
        result = approximation_ratio(approx, optimal)
        assert isinstance(result, float)
        assert result == pytest.approx(expected)

    def test_zero_over_zero_is_one(self):
        assert approximation_ratio(0, 0) == 1.0

    def test_nonzero_over_zero_raises(self):
        with pytest.raises(ValueError):
            approximation_ratio(1, 0)

    @pytest.mark.parametrize("approx, optimal", [(-1, 2), (2, -1), (-1, -1)])
    def test_negative_sizes_raise(self, approx, optimal):
        with pytest.raises(ValueError):
            approximation_ratio(approx, optimal)

    @pytest.mark.parametrize("approx, optimal", [(math.nan, 1), (1, math.inf), (math.inf, 2)])
    def test_non_finite_sizes_raise(self, approx, optimal):
        with pytest.raises(ValueError):
            approximation_ratio(approx, optimal)

    @pytest.mark.parametrize("approx, optimal", [("4", 2), (4, None), (True, 1)])
    def test_non_numbers_raise_type_error(self, approx, optimal):
        with pytest.raises(TypeError):
            approximation_ratio(approx, optimal)


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
class TestValidation:
    """Malformed graphs are rejected the same way by all three methods."""

    METHODS = [vertex_cover_2approx, vertex_cover_max_degree_greedy, vertex_cover_exact]

    @pytest.mark.parametrize("method", METHODS)
    def test_endpoint_outside_vertices_raises(self, method):
        with pytest.raises(ValueError):
            method([1, 2], {(1, 3)})

    @pytest.mark.parametrize("method", METHODS)
    def test_self_loop_raises(self, method):
        with pytest.raises(ValueError):
            method([1, 2], {(1, 1)})

    @pytest.mark.parametrize("method", METHODS)
    def test_string_vertices_raise_type_error(self, method):
        with pytest.raises(TypeError):
            method("abc", set())

    @pytest.mark.parametrize("method", METHODS)
    def test_duplicate_and_reversed_edges_collapse(self, method):
        edges = [(1, 2), (2, 1), (3, 2), (2, 3), (3, 4)]
        result = method([1, 2, 3, 4], edges)
        assert covers(PATH[1], result)
