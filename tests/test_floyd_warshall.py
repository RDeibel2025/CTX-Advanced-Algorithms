"""Tests for Floyd-Warshall, its 3D twin, path rebuilding, and the Week 4 cross-check.

A shortest-path bug hides well. Every entry of a distance matrix is just a
number, and a matrix that is wrong in one cell looks exactly like one that
is right. Re-running the recurrence inside the test would only copy whatever
mistake the module made, so every correctness claim here rests on a witness
that shares no mechanism with :func:`floyd_warshall`:

* **The CLRS anchor.** The five-vertex all-pairs example (CLRS 4e Figure
  23.4, vertices renumbered 0 to 4) and its final distance matrix, which
  came from the book and from paper rather than from running the module.
  The first two intermediate layers ``D(1)`` and ``D(2)`` are traced by hand
  in :class:`TestThreeDimensionalLayers`. This is the one check a
  consistently wrong implementation cannot fool.
* **Every simple path.** :func:`all_simple_paths` enumerates every simple
  ``i -> j`` path by depth-first search. With no negative cycle, a shortest
  walk is a shortest simple path, so the minimum over that list is the
  definition of the answer, not an algorithm for it. It is exponential and
  only used on the five-vertex anchor.
* **Week 4 Dijkstra.** On seeded random non-negative directed graphs, one
  :func:`~src.graphs.dijkstra.dijkstra` run per source is built into a
  matrix by :func:`dijkstra_matrix` in this file. Dijkstra is a greedy
  frontier search with a heap; it has nothing in common with a triple loop
  over intermediate vertices, so agreement between them means something.
* **Bellman-Ford.** Dijkstra refuses negative weights, so graphs with
  negative edges are checked against :func:`bellman_ford_all_pairs`, an
  edge-list relaxation written here from the textbook definition. A second
  helper, :func:`has_negative_cycle`, runs Bellman-Ford from a virtual
  super-source and decides independently whether a random signed matrix
  holds a negative cycle, which is what :class:`NegativeCycleError` must
  agree with.
* **The paths themselves.** A distance is only believable if a route
  achieves it, so :func:`assert_real_path` checks that every rebuilt path
  starts and ends in the right place, visits no vertex twice, uses only
  edges that exist in the *input*, and sums to exactly the reported
  distance.

The random batteries are seeded from fixed constants, and no instance
exceeds 30 vertices, because this file runs on every commit next to four
thousand other tests.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, List, Optional, Sequence, Tuple

import pytest

from src.dp_advanced.floyd_warshall import (
    NegativeCycleError,
    all_pairs_dijkstra,
    floyd_warshall,
    floyd_warshall_3d,
    reconstruct_path,
)
from src.graphs.dijkstra import dijkstra
from src.graphs.graph import Graph
from src.utils.graph_generator import weighted_graph
from src.utils.matrix_utils import INF, matrices_close, random_weight_matrix

Matrix = List[List[float]]


# ----------------------------------------------------------------------
# The CLRS anchor
# ----------------------------------------------------------------------
#: The nine directed edges of the CLRS 4e Figure 23.4 graph, vertices 0-4.
CLRS_EDGES: List[Tuple[int, int, int]] = [
    (0, 1, 3),
    (0, 2, 8),
    (0, 4, -4),
    (1, 3, 1),
    (1, 4, 7),
    (2, 1, 4),
    (3, 0, 2),
    (3, 2, -5),
    (4, 3, 6),
]

#: The same graph as a weight matrix: 0 on the diagonal, INF for no edge.
CLRS_WEIGHTS: Matrix = [
    [0, 3, 8, INF, -4],
    [INF, 0, INF, 1, 7],
    [INF, 4, 0, INF, INF],
    [2, INF, -5, 0, INF],
    [INF, INF, INF, 6, 0],
]

#: The final distance matrix D(5), from the book and checked by hand.
CLRS_DISTANCES: List[List[int]] = [
    [0, 1, -3, 2, -4],
    [3, 0, -4, 1, -1],
    [7, 4, 0, 5, 3],
    [2, -1, -5, 0, -2],
    [8, 5, 1, 6, 0],
]

CLRS_PAIRS = [pytest.param(i, j, id=f"{i}->{j}") for i in range(5) for j in range(5)]


def clrs_graph() -> Graph:
    """Build the CLRS anchor as a Week 4 Graph whose node ``i`` is index ``i``.

    The five nodes are added in order before any edge, so ``graph.nodes()``
    is ``[0, 1, 2, 3, 4]`` and matrix index ``i`` is vertex ``i``. Without
    that, insertion order would follow the edge list (0, 1, 2, 4, 3) and the
    rows would come out permuted.

    Returns:
        A new directed, weighted Graph holding :data:`CLRS_EDGES`.
    """
    graph = Graph(directed=True, weighted=True)
    for vertex in range(5):
        graph.add_node(vertex)
    for u, v, w in CLRS_EDGES:
        graph.add_edge(u, v, w)
    return graph


# ----------------------------------------------------------------------
# Independent oracles and checkers
# ----------------------------------------------------------------------
def all_simple_paths(
    weights: Sequence[Sequence[float]], i: int, j: int
) -> List[List[int]]:
    """Return every simple ``i -> j`` path in a weight matrix, by plain DFS.

    No distances, no relaxation, no ordering: just every route that never
    revisits a vertex. ``[[i]]`` when ``i == j``.

    Args:
        weights: An ``n x n`` matrix, INF for no edge. Not modified.
        i: Start vertex.
        j: End vertex.

    Returns:
        A list of paths, each a list of vertices from ``i`` to ``j``.
    """
    if i == j:
        return [[i]]
    size = len(weights)
    found: List[List[int]] = []

    def extend(path: List[int]) -> None:
        tail = path[-1]
        for nxt in range(size):
            if nxt == tail or weights[tail][nxt] == INF or nxt in path:
                continue
            if nxt == j:
                found.append(path + [nxt])
            else:
                extend(path + [nxt])

    extend([i])
    return found


def path_weight(weights: Sequence[Sequence[float]], path: Sequence[int]) -> float:
    """Sum the input edge weights along ``path``."""
    return sum(weights[u][v] for u, v in zip(path, path[1:]))


def bellman_ford_all_pairs(weights: Sequence[Sequence[float]]) -> List[List[float]]:
    """All-pairs shortest distances by Bellman-Ford from every source.

    The textbook edge-list relaxation: from each source, sweep every edge up
    to ``n - 1`` times, stopping early once a sweep changes nothing. It
    shares no structure with Floyd-Warshall (no intermediate-vertex
    recurrence, no matrix updates), which is why it can serve as an oracle
    on graphs with negative edges, where Dijkstra is not allowed.

    A final sweep asserts that nothing can still relax, so this oracle can
    never quietly hand back distances for a graph with a negative cycle.

    Args:
        weights: An ``n x n`` matrix, INF for no edge, no negative cycle.

    Returns:
        A new ``n x n`` distance matrix, INF where unreachable.
    """
    size = len(weights)
    edges = [
        (u, v, weights[u][v])
        for u in range(size)
        for v in range(size)
        if u != v and weights[u][v] != INF
    ]
    result: List[List[float]] = []
    for source in range(size):
        dist: List[float] = [INF] * size
        dist[source] = 0
        for _ in range(max(size - 1, 0)):
            changed = False
            for u, v, w in edges:
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    changed = True
            if not changed:
                break
        for u, v, w in edges:
            assert not (
                dist[u] != INF and dist[u] + w < dist[v]
            ), "oracle precondition: the graph has a negative cycle"
        result.append(dist)
    return result


def has_negative_cycle(weights: Sequence[Sequence[float]]) -> bool:
    """Decide whether any negative cycle exists, by super-source Bellman-Ford.

    Every vertex starts at distance 0, as if a virtual source had a
    zero-weight edge to each of them, so a cycle anywhere in the graph is
    reachable. With no negative cycle the distances settle within ``n``
    sweeps; a graph that is still relaxing after ``n + 1`` sweeps has one.
    A negative diagonal entry is a negative self-loop and counts too.

    Args:
        weights: An ``n x n`` matrix, INF for no edge.

    Returns:
        True when the graph contains a cycle of negative total weight.
    """
    size = len(weights)
    if any(weights[v][v] < 0 for v in range(size)):
        return True
    edges = [
        (u, v, weights[u][v])
        for u in range(size)
        for v in range(size)
        if u != v and weights[u][v] != INF
    ]
    dist = [0] * size
    for _ in range(size + 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return False
    return True


def dijkstra_matrix(graph: Graph) -> List[List[float]]:
    """Lay out one Week 4 :func:`dijkstra` run per source as a matrix.

    Rows and columns are both in ``graph.nodes()`` order, the same order
    Floyd-Warshall uses for a Graph input.
    """
    nodes = graph.nodes()
    rows: List[List[float]] = []
    for source in nodes:
        distances, _ = dijkstra(graph, source)
        rows.append([distances[target] for target in nodes])
    return rows


def graph_weights(graph: Graph) -> List[List[float]]:
    """Read a Graph's edge weights into a matrix without using the module.

    Built from :meth:`Graph.has_edge` and :meth:`Graph.get_edge_weight`
    alone, so the path checker does not lean on
    :func:`~src.utils.matrix_utils.graph_to_weight_matrix`, which the code
    under test itself uses. Self-loops are ignored: they never lie on a
    simple path.
    """
    nodes = graph.nodes()
    size = len(nodes)
    matrix: List[List[float]] = [[INF] * size for _ in range(size)]
    for a in range(size):
        matrix[a][a] = 0.0
        for b in range(size):
            if a != b and graph.has_edge(nodes[a], nodes[b]):
                matrix[a][b] = graph.get_edge_weight(nodes[a], nodes[b])
    return matrix


def matrix_to_graph(weights: Sequence[Sequence[float]]) -> Graph:
    """Build a directed weighted Graph whose node ``i`` is matrix index ``i``."""
    graph = Graph(directed=True, weighted=True)
    for vertex in range(len(weights)):
        graph.add_node(vertex)
    for u, row in enumerate(weights):
        for v, w in enumerate(row):
            if u != v and w != INF:
                graph.add_edge(u, v, w)
    return graph


def assert_real_path(
    weights: Sequence[Sequence[float]],
    dist: Sequence[Sequence[float]],
    pred: Sequence[Sequence[Optional[int]]],
    i: int,
    j: int,
    exact: bool = True,
) -> None:
    """Check that ``reconstruct_path(pred, i, j)`` is a real shortest route.

    ``[i]`` when ``i == j`` and ``[]`` when ``j`` is unreachable. Otherwise
    the path starts at ``i``, ends at ``j``, repeats no vertex, takes only
    edges present in the *input* ``weights``, and its edge weights sum to
    ``dist[i][j]``: exactly for integer weights, within round-off for the
    two-decimal float weights of the random Graph batteries.
    """
    path = reconstruct_path(pred, i, j)
    if i == j:
        assert path == [i]
        return
    if dist[i][j] == INF:
        assert pred[i][j] is None
        assert path == []
        return
    assert path, f"{i}->{j} has distance {dist[i][j]} but no path"
    assert path[0] == i and path[-1] == j
    assert len(set(path)) == len(path), f"path {path} revisits a vertex"
    for u, v in zip(path, path[1:]):
        assert u != v and weights[u][v] != INF, f"{u}->{v} is not an edge"
    total = path_weight(weights, path)
    if exact:
        assert total == dist[i][j]
    else:
        assert math.isclose(total, dist[i][j], rel_tol=1e-9, abs_tol=1e-9)


# ----------------------------------------------------------------------
# The hand-checked anchor
# ----------------------------------------------------------------------
class TestCLRSAnchor:
    """The CLRS 4e Figure 23.4 graph, with its final matrix from the book.

    Edges ``(u, v, w)``: (0,1,3) (0,2,8) (0,4,-4) (1,3,1) (1,4,7) (2,1,4)
    (3,0,2) (3,2,-5) (4,3,6). Three edges are negative, and no cycle is:
    the only cycles run through 3 -> 0 or 3 -> 2, and each one totals at
    least 1 (for example 0 -> 4 -> 3 -> 0 is -4 + 6 + 2 = 4).

    One entry traced by hand, because it is the instructive one: the best
    ``0 -> 1`` route ignores the direct edge of weight 3 and goes
    0 -> 4 -> 3 -> 2 -> 1, which costs -4 + 6 - 5 + 4 = **1**. No assertion
    in this class may be regenerated from the module's output.
    """

    def test_matrix_input_gives_the_book_matrix_exactly(self) -> None:
        dist, _ = floyd_warshall(CLRS_WEIGHTS)
        assert dist == CLRS_DISTANCES

    def test_graph_input_gives_the_book_matrix_exactly(self) -> None:
        graph = clrs_graph()
        assert graph.nodes() == [0, 1, 2, 3, 4]
        dist, _ = floyd_warshall(graph)
        assert dist == CLRS_DISTANCES

    def test_graph_with_scrambled_insertion_order_maps_back_through_nodes(self) -> None:
        # Edges added in reverse, so graph.nodes() is not 0..4. Indices must
        # be translated through graph.nodes(), and then the book matrix holds.
        graph = Graph(directed=True, weighted=True)
        for u, v, w in reversed(CLRS_EDGES):
            graph.add_edge(u, v, w)
        nodes = graph.nodes()
        assert nodes != [0, 1, 2, 3, 4]
        dist, _ = floyd_warshall(graph)
        for a, u in enumerate(nodes):
            for b, v in enumerate(nodes):
                assert dist[a][b] == CLRS_DISTANCES[u][v]

    def test_entries_are_floats(self) -> None:
        dist, _ = floyd_warshall(CLRS_WEIGHTS)
        assert all(isinstance(x, float) for row in dist for x in row)

    def test_hand_traced_route_from_0_to_1(self) -> None:
        dist, pred = floyd_warshall(CLRS_WEIGHTS)
        assert reconstruct_path(pred, 0, 1) == [0, 4, 3, 2, 1]
        assert -4 + 6 - 5 + 4 == 1 == dist[0][1]

    @pytest.mark.parametrize("i, j", CLRS_PAIRS)
    def test_every_rebuilt_path_is_real_and_sums_to_its_distance(
        self, i: int, j: int
    ) -> None:
        dist, pred = floyd_warshall(CLRS_WEIGHTS)
        assert_real_path(CLRS_WEIGHTS, dist, pred, i, j)
        assert (
            path_weight(CLRS_WEIGHTS, reconstruct_path(pred, i, j))
            == CLRS_DISTANCES[i][j]
        )

    @pytest.mark.parametrize("i, j", CLRS_PAIRS)
    def test_every_rebuilt_path_from_graph_input_is_real(self, i: int, j: int) -> None:
        graph = clrs_graph()
        dist, pred = floyd_warshall(graph)
        assert_real_path(graph_weights(graph), dist, pred, i, j)

    @pytest.mark.parametrize("i, j", CLRS_PAIRS)
    def test_anchor_agrees_with_enumerating_every_simple_path(
        self, i: int, j: int
    ) -> None:
        # A second, mechanical witness for the anchor, and a check on the
        # path: when exactly one simple path is shortest, it must be the one.
        paths = all_simple_paths(CLRS_WEIGHTS, i, j)
        best = min(path_weight(CLRS_WEIGHTS, p) for p in paths)
        assert best == CLRS_DISTANCES[i][j]
        dist, pred = floyd_warshall(CLRS_WEIGHTS)
        winners = [p for p in paths if path_weight(CLRS_WEIGHTS, p) == best]
        if len(winners) == 1:
            assert reconstruct_path(pred, i, j) == winners[0]
        else:
            assert reconstruct_path(pred, i, j) in winners


# ----------------------------------------------------------------------
# Loop order: k must be outermost
# ----------------------------------------------------------------------
class TestLoopOrder:
    """Chains whose only cheap route visits intermediates in a set order.

    Each graph has a chain of three weight-1 edges and a direct shortcut of
    weight 10 between its ends, so the true distance is 3 and a wrong loop
    order that finalises the end-to-end cell too early reports 10.

    The first three chains are adversarial for an ``i``-outermost loop
    (either ``i, j, k`` or ``i, k, j``). Take 0 -> 3 -> 2 -> 1: row 0 is
    finished first, but reaching 1 cheaply from 0 needs ``dist[3][1]`` or
    ``dist[2][1]`` via 3 -> 2 -> 1, and rows 2 and 3 have not been improved
    yet, so row 0 keeps the shortcut's 10. That was confirmed against a
    throwaway ``i``-outermost implementation, not the module. The last
    chain is a control that any loop order gets right.
    """

    @pytest.mark.parametrize(
        "chain",
        [
            pytest.param([0, 3, 2, 1], id="0-3-2-1"),
            pytest.param([1, 3, 2, 0], id="1-3-2-0"),
            pytest.param([2, 3, 1, 0], id="2-3-1-0"),
            pytest.param([0, 1, 2, 3], id="ascending-control"),
        ],
    )
    def test_chain_beats_the_shortcut(self, chain: List[int]) -> None:
        weights: Matrix = [[INF] * 4 for _ in range(4)]
        for v in range(4):
            weights[v][v] = 0
        for u, v in zip(chain, chain[1:]):
            weights[u][v] = 1
        weights[chain[0]][chain[-1]] = 10
        dist, pred = floyd_warshall(weights)
        assert dist[chain[0]][chain[-1]] == 3
        assert reconstruct_path(pred, chain[0], chain[-1]) == chain


# ----------------------------------------------------------------------
# The 3D version and its relationship to the in-place one
# ----------------------------------------------------------------------
class TestThreeDimensionalLayers:
    """``floyd_warshall_3d`` keeps every layer D(0), ..., D(n).

    The first two layers of the CLRS anchor, traced by hand (they match the
    book's D(1) and D(2)):

    * **D(1)**, vertex 0 allowed. Only vertex 3 has an edge into 0 (weight
      2), so only row 3 can change: (3,1) becomes 2 + 3 = 5, (3,4) becomes
      2 - 4 = -2, and (3,2) stays -5 because 2 + 8 = 10 is worse.
    * **D(2)**, vertex 1 also allowed. Rows 0, 2, 3 can reach 1. Row 0:
      (0,3) becomes 3 + 1 = 4; (0,4) stays -4. Row 2: (2,3) becomes
      4 + 1 = 5 and (2,4) becomes 4 + 7 = 11. Row 3 is unchanged.
    """

    D1: List[List[float]] = [
        [0, 3, 8, INF, -4],
        [INF, 0, INF, 1, 7],
        [INF, 4, 0, INF, INF],
        [2, 5, -5, 0, -2],
        [INF, INF, INF, 6, 0],
    ]
    D2: List[List[float]] = [
        [0, 3, 8, 4, -4],
        [INF, 0, INF, 1, 7],
        [INF, 4, 0, 5, 11],
        [2, 5, -5, 0, -2],
        [INF, INF, INF, 6, 0],
    ]

    def test_layer_count_and_shape(self) -> None:
        layers = floyd_warshall_3d(CLRS_WEIGHTS)
        assert len(layers) == 5 + 1
        assert all(
            len(layer) == 5 and all(len(row) == 5 for row in layer) for layer in layers
        )

    def test_layer_zero_is_the_input(self) -> None:
        assert floyd_warshall_3d(CLRS_WEIGHTS)[0] == CLRS_WEIGHTS

    def test_hand_traced_first_and_second_layers(self) -> None:
        layers = floyd_warshall_3d(CLRS_WEIGHTS)
        assert layers[1] == self.D1
        assert layers[2] == self.D2

    def test_last_layer_is_the_book_matrix(self) -> None:
        assert floyd_warshall_3d(CLRS_WEIGHTS)[-1] == CLRS_DISTANCES

    def test_last_layer_equals_in_place_result_on_graph_input(self) -> None:
        graph = clrs_graph()
        assert floyd_warshall_3d(graph)[-1] == floyd_warshall(graph)[0]

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("negative", [False, True], ids=["nonneg", "negative"])
    def test_last_layer_equals_in_place_result_exactly(
        self, seed: int, negative: bool
    ) -> None:
        weights = random_weight_matrix(
            12, 0.35, seed=600 + seed, allow_negative=negative
        )
        layers = floyd_warshall_3d(weights)
        assert len(layers) == 13
        assert layers[0] == weights
        # Exact, not close: the in-place argument says the same additions
        # happen in the same order, so the results are identical floats.
        assert layers[-1] == floyd_warshall(weights)[0]

    @pytest.mark.parametrize("seed", range(4))
    def test_layers_never_increase(self, seed: int) -> None:
        # D(k) is a minimum that includes D(k-1), so no cell can grow.
        layers = floyd_warshall_3d(
            random_weight_matrix(9, 0.4, seed=700 + seed, allow_negative=True)
        )
        for before, after in zip(layers, layers[1:]):
            for row_b, row_a in zip(before, after):
                assert all(a <= b for a, b in zip(row_a, row_b))

    def test_positive_diagonal_is_normalised_in_layer_zero(self) -> None:
        assert floyd_warshall_3d([[5, 2], [INF, 9]])[0] == [[0, 2], [INF, 0]]

    def test_empty_input_has_one_empty_layer(self) -> None:
        assert floyd_warshall_3d([]) == [[]]


# ----------------------------------------------------------------------
# The caller's data is never touched
# ----------------------------------------------------------------------
class TestInputNotMutated:
    """Both functions promise a fresh matrix to work in."""

    @pytest.mark.parametrize(
        "fn", [floyd_warshall, floyd_warshall_3d], ids=["2d", "3d"]
    )
    def test_list_input_unchanged(self, fn: Any) -> None:
        weights = copy.deepcopy(CLRS_WEIGHTS)
        fn(weights)
        assert weights == CLRS_WEIGHTS

    def test_positive_diagonal_left_in_callers_matrix(self) -> None:
        weights = [[7, 2], [3, 9]]
        dist, _ = floyd_warshall(weights)
        assert dist == [[0, 2], [3, 0]]
        assert weights == [[7, 2], [3, 9]]

    def test_result_does_not_alias_input(self) -> None:
        weights = copy.deepcopy(CLRS_WEIGHTS)
        dist, _ = floyd_warshall(weights)
        for row in dist:
            row[0] = 999
        layers = floyd_warshall_3d(weights)
        layers[0][0][1] = 999
        assert weights == CLRS_WEIGHTS

    def test_graph_input_unchanged(self) -> None:
        graph = clrs_graph()
        before = sorted(graph.edges())
        floyd_warshall(graph)
        floyd_warshall_3d(graph)
        assert sorted(graph.edges()) == before
        assert graph.nodes() == [0, 1, 2, 3, 4]

    def test_all_pairs_dijkstra_leaves_graph_unchanged(self) -> None:
        graph = weighted_graph(10, density=0.3, directed=True, connected=True, seed=77)
        before = sorted(graph.edges())
        all_pairs_dijkstra(graph)
        assert sorted(graph.edges()) == before


# ----------------------------------------------------------------------
# Negative cycles
# ----------------------------------------------------------------------
class TestNegativeCycles:
    """A negative cycle has no shortest paths, so it must be refused.

    Each instance is small enough to total its cycle by eye, noted per case.
    """

    NEGATIVE: List[Any] = [
        # 0 -> 1 -> 0 totals -1 - 1 = -2.
        pytest.param([[0, -1], [-1, 0]], id="two-cycle"),
        # 0 -> 1 -> 2 -> 0 totals 1 - 2 - 1 = -2.
        pytest.param([[0, 1, INF], [INF, 0, -2], [-1, INF, 0]], id="three-cycle"),
        # 2 -> 3 -> 2 totals 1 - 3 = -2, in a component vertex 0 cannot reach.
        pytest.param(
            [[0, 5, INF, INF], [INF, 0, INF, INF], [INF, INF, 0, 1], [INF, INF, -3, 0]],
            id="unreachable-component",
        ),
        # A negative self-loop written straight onto the diagonal.
        pytest.param([[0, 4], [INF, -1]], id="diagonal-self-loop"),
        # A cycle of 3 + 3 + (-7) = -1 that only the full tour reveals.
        pytest.param([[0, 3, INF], [INF, 0, 3], [-7, INF, 0]], id="long-negative"),
    ]

    @pytest.mark.parametrize("weights", NEGATIVE)
    @pytest.mark.parametrize(
        "fn", [floyd_warshall, floyd_warshall_3d], ids=["2d", "3d"]
    )
    def test_matrix_with_negative_cycle_raises(self, fn: Any, weights: Matrix) -> None:
        with pytest.raises(NegativeCycleError):
            fn(weights)

    def test_error_is_a_value_error(self) -> None:
        assert issubclass(NegativeCycleError, ValueError)
        with pytest.raises(ValueError):
            floyd_warshall([[0, -1], [-1, 0]])

    @pytest.mark.parametrize(
        "fn", [floyd_warshall, floyd_warshall_3d], ids=["2d", "3d"]
    )
    def test_graph_with_negative_cycle_raises(self, fn: Any) -> None:
        # a -> b -> c -> a totals 2 + 2 - 5 = -1.
        graph = Graph(directed=True, weighted=True)
        for u, v, w in [("a", "b", 2), ("b", "c", 2), ("c", "a", -5)]:
            graph.add_edge(u, v, w)
        with pytest.raises(NegativeCycleError):
            fn(graph)

    @pytest.mark.parametrize(
        "fn", [floyd_warshall, floyd_warshall_3d], ids=["2d", "3d"]
    )
    def test_graph_negative_self_loop_raises_and_names_the_vertex(
        self, fn: Any
    ) -> None:
        # "t" is a sink, so it is the only vertex with a negative closed walk.
        graph = Graph(directed=True, weighted=True)
        graph.add_edge("s", "t", 2)
        graph.add_edge("t", "t", -1)
        assert graph.nodes() == ["s", "t"]
        with pytest.raises(NegativeCycleError, match=r"vertex 1\b"):
            fn(graph)

    def test_graph_positive_self_loop_is_ignored(self) -> None:
        graph = Graph(directed=True, weighted=True)
        graph.add_edge("s", "t", 2)
        graph.add_edge("t", "t", 5)
        dist, pred = floyd_warshall(graph)
        assert dist == [[0, 2], [INF, 0]]
        assert pred[1][1] is None

    def test_zero_weight_cycle_is_allowed(self) -> None:
        # 0 -> 1 -> 0 totals 2 - 2 = 0: not negative, so distances exist.
        dist, pred = floyd_warshall([[0, 2], [-2, 0]])
        assert dist == [[0, 2], [-2, 0]]
        assert reconstruct_path(pred, 0, 1) == [0, 1]
        assert reconstruct_path(pred, 1, 0) == [1, 0]


# ----------------------------------------------------------------------
# Unreachable pairs
# ----------------------------------------------------------------------
class TestUnreachable:
    """No path is a normal answer: INF distance, None predecessor, [] path."""

    def test_small_hand_case(self) -> None:
        # 0 -> 1 (2) and 2 -> 1 (1). Nothing reaches 2, and 1 reaches nothing.
        dist, pred = floyd_warshall([[0, 2, INF], [INF, 0, INF], [INF, 1, 0]])
        assert dist == [[0, 2, INF], [INF, 0, INF], [INF, 1, 0]]
        assert pred[0][2] is None and pred[1][0] is None and pred[1][2] is None
        assert reconstruct_path(pred, 0, 2) == []
        assert reconstruct_path(pred, 1, 0) == []
        assert reconstruct_path(pred, 2, 1) == [2, 1]

    def test_no_edges_at_all(self) -> None:
        n = 4
        weights = [[0 if i == j else INF for j in range(n)] for i in range(n)]
        dist, pred = floyd_warshall(weights)
        assert dist == weights
        assert all(p is None for row in pred for p in row)
        for i in range(n):
            for j in range(n):
                assert reconstruct_path(pred, i, j) == ([i] if i == j else [])

    def test_isolated_graph_node(self) -> None:
        graph = clrs_graph()
        graph.add_node("island")
        dist, pred = floyd_warshall(graph)
        island = graph.nodes().index("island")
        assert island == 5
        for v in range(5):
            assert dist[v][island] == INF and dist[island][v] == INF
            assert pred[v][island] is None and pred[island][v] is None
            assert reconstruct_path(pred, v, island) == []
            assert reconstruct_path(pred, island, v) == []
        assert dist[island][island] == 0
        # The rest of the anchor is untouched by the extra vertex.
        assert [row[:5] for row in dist[:5]] == CLRS_DISTANCES
        assert all_pairs_dijkstra(Graph(directed=True, weighted=True)) == []

    def test_zero_weight_edge_is_not_missing(self) -> None:
        graph = Graph(directed=True, weighted=True)
        graph.add_edge("a", "b", 0)
        dist, pred = floyd_warshall(graph)
        assert dist == [[0, 0], [INF, 0]]
        assert reconstruct_path(pred, 0, 1) == [0, 1]
        assert reconstruct_path(pred, 1, 0) == []


# ----------------------------------------------------------------------
# Contracts: input checking and edges of the domain
# ----------------------------------------------------------------------
class TestContracts:
    """Bad input is refused with the documented exception type."""

    BAD_INPUT: List[Any] = [
        pytest.param(7, TypeError, id="not-a-matrix"),
        pytest.param([5, 6], TypeError, id="rows-not-sequences"),
        pytest.param([[0, True], [1, 0]], TypeError, id="bool-entry"),
        pytest.param([[0, "1"], [1, 0]], TypeError, id="string-entry"),
        pytest.param([[0, 1], [1]], ValueError, id="ragged"),
        pytest.param([[0, 1, 2], [1, 0, 2]], ValueError, id="not-square"),
        pytest.param([[0, float("nan")], [1, 0]], ValueError, id="nan"),
        pytest.param([[0, -INF], [1, 0]], ValueError, id="minus-inf"),
    ]

    @pytest.mark.parametrize("weights, error", BAD_INPUT)
    @pytest.mark.parametrize(
        "fn", [floyd_warshall, floyd_warshall_3d], ids=["2d", "3d"]
    )
    def test_bad_input_raises(self, fn: Any, weights: Any, error: type) -> None:
        with pytest.raises(error) as info:
            fn(weights)
        # A malformed matrix is not a negative cycle and must not say it is.
        assert not isinstance(info.value, NegativeCycleError)

    def test_empty_matrix(self) -> None:
        assert floyd_warshall([]) == ([], [])

    def test_empty_graph(self) -> None:
        assert floyd_warshall(Graph(directed=True, weighted=True)) == ([], [])

    def test_single_vertex(self) -> None:
        assert floyd_warshall([[0]]) == ([[0.0]], [[None]])
        assert reconstruct_path([[None]], 0, 0) == [0]

    @pytest.mark.parametrize("i, j", [(0, 3), (3, 0), (-1, 0), (0, -1)])
    def test_reconstruct_path_rejects_out_of_range(self, i: int, j: int) -> None:
        _, pred = floyd_warshall([[0, 2, INF], [INF, 0, INF], [INF, 1, 0]])
        with pytest.raises(IndexError):
            reconstruct_path(pred, i, j)

    def test_all_pairs_dijkstra_requires_a_graph(self) -> None:
        with pytest.raises(TypeError):
            all_pairs_dijkstra([[0, 1], [1, 0]])

    def test_predecessor_is_none_exactly_on_diagonal_and_unreachable(self) -> None:
        dist, pred = floyd_warshall([[0, 2, INF], [INF, 0, INF], [INF, 1, 0]])
        for i in range(3):
            for j in range(3):
                assert (pred[i][j] is None) == (i == j or dist[i][j] == INF)


# ----------------------------------------------------------------------
# Agreement with Week 4 Dijkstra on non-negative random graphs
# ----------------------------------------------------------------------
#: (n, density, seed) for connected directed graphs. Sizes up to 30, four
#: densities from very sparse to complete, three draws each.
DIJKSTRA_BATTERY = [
    pytest.param(n, density, seed, id=f"n{n}-d{density}-s{seed}")
    for n in (2, 9, 18, 30)
    for density in (0.05, 0.2, 0.5, 1.0)
    for seed in (0, 1, 2)
]

#: Sparse graphs without the spanning tree, so many pairs are unreachable.
DISCONNECTED_BATTERY = [
    pytest.param(n, seed, id=f"n{n}-s{seed}") for n in (9, 18, 30) for seed in (0, 1, 2)
]


class TestAgreementWithDijkstra:
    """Floyd-Warshall must match one Week 4 Dijkstra run per source.

    ``weighted_graph`` draws two-decimal float weights, so Floyd-Warshall
    and Dijkstra can reach the same distance by different orders of addition
    and differ in the last bit. Hence :func:`matrices_close` rather than
    ``==`` for this comparison.
    """

    @pytest.mark.parametrize("n, density, seed", DIJKSTRA_BATTERY)
    def test_connected_directed(self, n: int, density: float, seed: int) -> None:
        graph = weighted_graph(
            n, density=density, directed=True, connected=True, seed=1000 + seed
        )
        oracle = dijkstra_matrix(graph)
        dist, pred = floyd_warshall(graph)
        assert matrices_close(dist, oracle)
        # Connected means every vertex is reachable from the first node.
        assert all(d != INF for d in dist[graph.nodes().index(0)])
        # all_pairs_dijkstra is Week 4 laid out the same way, cell for cell.
        assert all_pairs_dijkstra(graph) == oracle
        weights = graph_weights(graph)
        for i in range(n):
            for j in range(n):
                assert_real_path(weights, dist, pred, i, j, exact=False)

    @pytest.mark.parametrize("n, seed", DISCONNECTED_BATTERY)
    def test_sparse_disconnected_directed(self, n: int, seed: int) -> None:
        graph = weighted_graph(
            n, density=0.05, directed=True, connected=False, seed=2000 + seed
        )
        oracle = dijkstra_matrix(graph)
        assert any(d == INF for row in oracle for d in row), "battery precondition"
        dist, pred = floyd_warshall(graph)
        assert matrices_close(dist, oracle)
        weights = graph_weights(graph)
        for i in range(n):
            for j in range(n):
                assert_real_path(weights, dist, pred, i, j, exact=False)

    @pytest.mark.parametrize("seed", range(3))
    def test_undirected_is_symmetric_and_agrees(self, seed: int) -> None:
        graph = weighted_graph(
            15, density=0.2, directed=False, connected=True, seed=3000 + seed
        )
        dist, _ = floyd_warshall(graph)
        assert matrices_close(dist, dijkstra_matrix(graph))
        assert all(
            math.isclose(dist[i][j], dist[j][i], abs_tol=1e-9)
            for i in range(15)
            for j in range(15)
        )


# ----------------------------------------------------------------------
# Negative edges, no negative cycle: Bellman-Ford is the oracle
# ----------------------------------------------------------------------
#: Every combination below was checked to draw at least one negative edge;
#: the test asserts that precondition again so the battery cannot go soft.
NEGATIVE_EDGE_BATTERY = [
    pytest.param(n, density, seed, id=f"n{n}-d{density}-s{seed}")
    for n in (5, 10, 15, 20, 25)
    for density in (0.3, 0.6, 1.0)
    for seed in (0, 1, 2)
]


class TestNegativeEdgesAgainstBellmanFord:
    """Graphs from ``random_weight_matrix(allow_negative=True)``.

    That generator reweights by a vertex potential, so every cycle keeps
    its non-negative total while individual edges go negative. Weights are
    integers, so distances are compared exactly.
    """

    @pytest.mark.parametrize("n, density, seed", NEGATIVE_EDGE_BATTERY)
    def test_matrix_input_agrees_with_bellman_ford(
        self, n: int, density: float, seed: int
    ) -> None:
        weights = random_weight_matrix(n, density, seed=seed, allow_negative=True)
        assert any(w < 0 for row in weights for w in row), "battery precondition"
        oracle = bellman_ford_all_pairs(weights)
        dist, pred = floyd_warshall(weights)
        assert dist == oracle
        assert matrices_close(dist, oracle)
        for i in range(n):
            for j in range(n):
                assert_real_path(weights, dist, pred, i, j)

    @pytest.mark.parametrize("n, density, seed", NEGATIVE_EDGE_BATTERY[::4])
    def test_graph_input_matches_matrix_input(
        self, n: int, density: float, seed: int
    ) -> None:
        weights = random_weight_matrix(n, density, seed=seed, allow_negative=True)
        assert floyd_warshall(matrix_to_graph(weights)) == floyd_warshall(weights)

    @pytest.mark.parametrize("n, density, seed", NEGATIVE_EDGE_BATTERY[::4])
    def test_all_pairs_dijkstra_refuses_negative_weights(
        self, n: int, density: float, seed: int
    ) -> None:
        weights = random_weight_matrix(n, density, seed=seed, allow_negative=True)
        graph = matrix_to_graph(weights)
        with pytest.raises(ValueError):
            all_pairs_dijkstra(graph)
        # Floyd-Warshall handles the same graph without complaint.
        assert floyd_warshall(graph)[0] == bellman_ford_all_pairs(weights)


# ----------------------------------------------------------------------
# Arbitrary signs: does Floyd-Warshall refuse exactly the cyclic ones?
# ----------------------------------------------------------------------
SIGNED_SEEDS = range(40)


def signed_instance(seed: int) -> Matrix:
    """A random signed matrix with no cycle guarantee at all.

    Three to eight vertices, each ordered pair an edge with probability 0.4,
    weights uniform in [-4, 12]. Some draws hold a negative cycle and some
    do not; :func:`has_negative_cycle` decides which.
    """
    rng = random.Random(9000 + seed)
    n = rng.randint(3, 8)
    return [
        [
            0 if u == v else (rng.randint(-4, 12) if rng.random() < 0.4 else INF)
            for v in range(n)
        ]
        for u in range(n)
    ]


class TestSignedBattery:
    """``NegativeCycleError`` must fire if and only if Bellman-Ford finds a cycle."""

    def test_battery_covers_both_outcomes(self) -> None:
        verdicts = {has_negative_cycle(signed_instance(seed)) for seed in SIGNED_SEEDS}
        assert verdicts == {True, False}

    @pytest.mark.parametrize("seed", SIGNED_SEEDS)
    def test_raises_exactly_when_a_negative_cycle_exists(self, seed: int) -> None:
        weights = signed_instance(seed)
        if has_negative_cycle(weights):
            with pytest.raises(NegativeCycleError):
                floyd_warshall(weights)
            with pytest.raises(NegativeCycleError):
                floyd_warshall_3d(weights)
            return
        dist, pred = floyd_warshall(weights)
        assert dist == bellman_ford_all_pairs(weights)
        assert floyd_warshall_3d(weights)[-1] == dist
        n = len(weights)
        for i in range(n):
            for j in range(n):
                assert_real_path(weights, dist, pred, i, j)


# ----------------------------------------------------------------------
# matrices_close
# ----------------------------------------------------------------------
class TestMatricesClose:
    """The comparison every agreement test above relies on."""

    def test_inf_equals_inf(self) -> None:
        assert matrices_close([[0, INF], [INF, 0]], [[0.0, INF], [INF, 0.0]])
        assert matrices_close([[-INF]], [[-INF]])

    def test_inf_is_far_from_every_finite_value(self) -> None:
        assert not matrices_close([[INF]], [[1e300]])
        assert not matrices_close([[1e300]], [[INF]])
        assert not matrices_close([[INF]], [[-INF]])

    @pytest.mark.parametrize(
        "a, b",
        [
            pytest.param([[1, 2]], [[1, 2], [3, 4]], id="fewer-rows"),
            pytest.param([[1, 2], [3, 4]], [[1, 2]], id="more-rows"),
            pytest.param([[1, 2], [3]], [[1, 2], [3, 4]], id="short-row"),
            pytest.param([[1, 2, 5], [3, 4]], [[1, 2], [3, 4]], id="long-row"),
            pytest.param([[]], [], id="one-empty-row-vs-none"),
        ],
    )
    def test_shape_mismatch_is_false(self, a: Matrix, b: Matrix) -> None:
        assert matrices_close(a, b) is False

    def test_empty_matrices_match(self) -> None:
        assert matrices_close([], [])

    def test_round_off_is_forgiven(self) -> None:
        assert [[0.1 + 0.2]] != [[0.3]]
        assert matrices_close([[0.1 + 0.2]], [[0.3]])
        assert matrices_close([[1.0]], [[1.0 + 1e-12]])

    def test_relative_tolerance_respected(self) -> None:
        assert not matrices_close([[1.0]], [[1.001]])
        assert matrices_close([[1.0]], [[1.001]], rel_tol=1e-2)
        assert not matrices_close([[100.0]], [[102.0]], rel_tol=1e-2)

    def test_absolute_tolerance_respected(self) -> None:
        assert matrices_close([[0.0]], [[1e-12]])
        assert not matrices_close([[0.0]], [[1e-6]])
        assert matrices_close([[0.0]], [[1e-6]], abs_tol=1e-5)
        assert not matrices_close([[0.0]], [[1e-6]], rel_tol=0.5, abs_tol=0.0)

    def test_nan_is_never_close(self) -> None:
        assert not matrices_close([[float("nan")]], [[float("nan")]])

    def test_one_bad_cell_is_enough(self) -> None:
        a = [[0.0] * 5 for _ in range(5)]
        b = copy.deepcopy(a)
        b[4][4] = 0.5
        assert not matrices_close(a, b)
