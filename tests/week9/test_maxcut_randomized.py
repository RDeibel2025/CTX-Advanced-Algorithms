"""Tests for the random cut, local search and exact Max-Cut (Week 9, spec 10.4).

Max-Cut answers can look right and be wrong in two ways: the side map can
be malformed (a vertex missing or a side other than 0/1), or the reported
cut can disagree with the map. The claims checked here are the ones the
module proves, held against witnesses that share no code with it:

* **Hand-checked anchors.** K4 has OPT = 4 of 6 edges (a 2-2 split cuts the
  4 edges between the pairs, a 1-3 split only 3); the 5-cycle has
  OPT = 4 of 5 (an odd cycle cannot be 2-coloured, so one edge stays uncut,
  and alternating sides leaves exactly one); K3,3 has OPT = 9 of 9 (it is
  bipartite); the 16-cycle has OPT = 16 (even cycle, bipartite).
* **Local search from all zeros on K3,3** (parts {0, 1, 2} and {3, 4, 5}).
  The cut starts at 0. Vertex 0 has all 3 neighbours on its side, gain 3,
  so it flips (cut 3); the restarted scan finds vertex 0 at gain -3 and
  vertex 1 at gain 3, flips it (cut 6), then vertex 2 likewise (cut 9).
  Vertices 3, 4, 5 now have every neighbour across, gain -3, so it stops:
  cut 9 after exactly 3 flips.
* **Independent checks.** :func:`count_cut` re-counts a cut straight from
  the definition and :func:`assert_local_optimum` re-prices every single
  flip from scratch, so a wrong incremental counter in the module cannot
  hide.
* **The 1/2 guarantees.** Every local optimum cuts at least |E|/2 edges;
  the random cut has mean |E|/2 (on K4, 500 seeded runs land within 0.3 of
  3; the standard deviation of one run is sqrt(6/4) ~ 1.22 because edge-cut
  indicators are pairwise independent, so the mean of 500 has standard
  deviation ~ 0.055 and 0.3 is about 5.5 of those).

Every random draw comes from a seed made by :func:`derive_seed`, no graph
passed to the exact solver has more than 16 vertices, and no test asserts
on time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple

import pytest

from src.approximation.maxcut_randomized import (
    cut_value,
    maxcut_exact,
    maxcut_local_search,
    maxcut_random,
)
from src.utils.graph_utils import random_graph
from src.utils.random_utils import derive_seed

#: Master seed for every seeded run in this file.
MASTER_SEED = 9004

Graph = Tuple[List[int], List[Tuple[int, int]]]

K4: Graph = (list(range(4)), [(u, v) for u in range(4) for v in range(u + 1, 4)])
C5: Graph = (list(range(5)), [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
K33: Graph = (list(range(6)), [(a, b) for a in range(3) for b in range(3, 6)])

#: name -> (graph, |E|, hand-checked OPT, an optimal split written by hand)
ANCHORS: Dict[str, Tuple[Graph, int, int, Dict[int, int]]] = {
    "K4": (K4, 6, 4, {0: 0, 1: 0, 2: 1, 3: 1}),
    "C5": (C5, 5, 4, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}),
    "K3,3": (K33, 9, 9, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}),
}


def count_cut(edges: Sequence[Tuple[Hashable, Hashable]], side: Dict[Hashable, int]) -> int:
    """Count the edges with ends on different sides, from the definition."""
    return sum(1 for u, v in edges if side[u] != side[v])


def assert_valid_side(vertices: Sequence[Hashable], side: Dict[Hashable, int]) -> None:
    """Assert ``side`` gives every vertex, and nothing else, the side 0 or 1."""
    assert set(side) == set(vertices)
    assert all(value in (0, 1) for value in side.values())


def assert_local_optimum(
    vertices: Sequence[Hashable], edges: Sequence[Tuple[Hashable, Hashable]], side: Dict[Hashable, int]
) -> None:
    """Assert no single-vertex flip increases the cut, re-pricing each flip."""
    base = count_cut(edges, side)
    for v in vertices:
        flipped = dict(side)
        flipped[v] = 1 - side[v]
        assert count_cut(edges, flipped) <= base, f"flipping {v} would improve the cut"


# --------------------------------------------------------------------------- cut_value


def test_cut_value_hand_counted_splits() -> None:
    """cut_value agrees with hand counts on the anchor graphs."""
    _, k4_edges = K4
    assert cut_value(k4_edges, {0: 0, 1: 0, 2: 1, 3: 1}) == 4   # 2-2 split
    assert cut_value(k4_edges, {0: 1, 1: 0, 2: 0, 3: 0}) == 3   # 1-3 split
    assert cut_value(k4_edges, {v: 0 for v in range(4)}) == 0
    _, c5_edges = C5
    assert cut_value(c5_edges, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}) == 4   # (0, 4) uncut
    _, k33_edges = K33
    assert cut_value(k33_edges, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}) == 9


def test_cut_value_rejects_missing_side() -> None:
    with pytest.raises(ValueError):
        cut_value([(0, 1)], {0: 0})


# --------------------------------------------------------------------------- exact


@pytest.mark.parametrize("name", list(ANCHORS))
def test_exact_anchor(name: str) -> None:
    """Exact Max-Cut: K4 4 of 6, 5-cycle 4 of 5, K3,3 9 of 9."""
    (vertices, edges), n_edges, opt, _ = ANCHORS[name]
    assert len(edges) == n_edges
    cut, side = maxcut_exact(vertices, edges)
    assert cut == opt
    assert_valid_side(vertices, side)
    assert count_cut(edges, side) == opt
    assert cut_value(edges, side) == opt
    assert side[min(vertices)] == 0


def test_exact_handles_sixteen_vertices() -> None:
    """n = 16 is allowed; the even 16-cycle is bipartite, so OPT = 16."""
    vertices = list(range(16))
    edges = [(i, (i + 1) % 16) for i in range(16)]
    cut, side = maxcut_exact(vertices, edges)
    assert cut == 16
    assert count_cut(edges, side) == 16


def test_exact_rejects_more_than_sixteen_vertices() -> None:
    with pytest.raises(ValueError):
        maxcut_exact(list(range(17)), [])
    with pytest.raises(ValueError):
        maxcut_exact(list(range(20)), [(i, i + 1) for i in range(19)])


# --------------------------------------------------------------------------- local search


@pytest.mark.parametrize("name", list(ANCHORS))
def test_local_search_anchor_at_least_half(name: str) -> None:
    """Local search on each anchor: cut >= |E|/2, valid map, cut_value matches."""
    (vertices, edges), n_edges, opt, _ = ANCHORS[name]
    for trial in range(20):
        seed = derive_seed(MASTER_SEED, "local-anchor", name, trial)
        cut, side, flips = maxcut_local_search(vertices, edges, seed=seed)
        assert 2 * cut >= n_edges
        assert cut <= opt
        assert_valid_side(vertices, side)
        assert cut_value(edges, side) == cut
        assert count_cut(edges, side) == cut
        assert_local_optimum(vertices, edges, side)
        assert 0 <= flips <= n_edges


def test_local_search_k33_from_all_zeros_reaches_nine() -> None:
    """Hand trace in the module docstring above: vertices 0, 1, 2 flip, cut 9."""
    vertices, edges = K33
    start = {v: 0 for v in vertices}
    cut, side, flips = maxcut_local_search(vertices, edges, start=start)
    assert cut == 9
    assert flips == 3
    assert side == {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
    assert start == {v: 0 for v in vertices}   # the start map is not modified


@pytest.mark.parametrize("name", list(ANCHORS))
def test_local_search_zero_flips_at_local_optimum(name: str) -> None:
    """Starting at a hand-written optimal split (a local optimum) makes no flips."""
    (vertices, edges), _, opt, optimal_split = ANCHORS[name]
    cut, side, flips = maxcut_local_search(vertices, edges, start=dict(optimal_split))
    assert flips == 0
    assert cut == opt
    assert side == optimal_split


def test_local_search_flips_consistent_with_cut_gain() -> None:
    """Each flip raises the cut by at least 1, so final - start >= flips."""
    for trial in range(30):
        seed = derive_seed(MASTER_SEED, "flips", trial)
        n = 6 + trial % 7
        vertices, edges = random_graph(n, 0.5, seed)
        edges = sorted(edges)
        start_cut, start_side = maxcut_random(vertices, edges, seed=seed)
        cut, side, flips = maxcut_local_search(vertices, edges, start=start_side)
        assert cut - start_cut >= flips
        assert flips <= len(edges)
        assert (flips == 0) == (cut == start_cut)
        # The seeded start is the same assignment maxcut_random draws.
        assert maxcut_local_search(vertices, edges, seed=seed) == (cut, side, flips)


def test_local_search_and_random_bounded_by_exact() -> None:
    """On seeded random graphs: |E|/2 <= local <= OPT and random <= OPT."""
    for trial in range(20):
        seed = derive_seed(MASTER_SEED, "bounds", trial)
        n = 4 + trial % 9
        vertices, edges = random_graph(n, 0.4, seed)
        edges = sorted(edges)
        opt, opt_side = maxcut_exact(vertices, edges)
        assert count_cut(edges, opt_side) == opt
        assert 2 * opt >= len(edges)
        rand_cut, rand_side = maxcut_random(vertices, edges, seed=seed)
        assert count_cut(edges, rand_side) == rand_cut <= opt
        cut, side, _ = maxcut_local_search(vertices, edges, seed=seed)
        assert count_cut(edges, side) == cut
        assert 2 * cut >= len(edges)
        assert cut <= opt


def test_local_search_large_sparse_graph() -> None:
    """n = 500 on adjacency lists: still a local optimum with cut >= |E|/2."""
    vertices, edges = random_graph(500, 0.02, derive_seed(MASTER_SEED, "large"))
    cut, side, flips = maxcut_local_search(vertices, edges, seed=derive_seed(MASTER_SEED, "large-start"))
    assert count_cut(list(edges), side) == cut
    assert 2 * cut >= len(edges)
    assert flips <= len(edges)


# --------------------------------------------------------------------------- random cut


def test_random_cut_mean_on_k4_is_half_the_edges() -> None:
    """Mean over 500 seeded runs on K4 is within 0.3 of |E|/2 = 3."""
    vertices, edges = K4
    total = 0
    for trial in range(500):
        cut, side = maxcut_random(vertices, edges, seed=derive_seed(MASTER_SEED, "k4-random", trial))
        assert_valid_side(vertices, side)
        assert count_cut(edges, side) == cut
        total += cut
    assert abs(total / 500 - 3.0) <= 0.3


def test_same_seed_same_result() -> None:
    for name, ((vertices, edges), _, _, _) in ANCHORS.items():
        for trial in range(5):
            seed = derive_seed(MASTER_SEED, "repeat", name, trial)
            assert maxcut_random(vertices, edges, seed=seed) == maxcut_random(vertices, edges, seed=seed)
            assert maxcut_local_search(vertices, edges, seed=seed) == maxcut_local_search(
                vertices, edges, seed=seed
            )
