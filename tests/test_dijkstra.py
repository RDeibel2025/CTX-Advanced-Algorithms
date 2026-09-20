"""Tests for Dijkstra's single-source shortest paths.

A shortest-path bug does not look like a crash. It looks like a number:
plausible, well typed, and wrong. Re-deriving the algorithm inside the
test would reproduce the same mistake and call it agreement, so almost
nothing here is checked against a second copy of Dijkstra's logic.

Four independent witnesses are used instead, in rough order of strength:

* **A graph worked out by hand.** Six nodes, every distance and every
  parent computed on paper in the comment above the test, including two
  nodes where the direct edge is *not* the shortest route. This is the
  only check in the file that cannot be fooled by a consistently wrong
  implementation, because its expected values did not come from code.
* **Bellman-Ford**, written here in the test file from the relaxation
  definition of a shortest path. It shares no data structure and no
  settling order with Dijkstra, so where the two agree the distances are
  right, not merely self-consistent.
* **Breadth-first search.** On a graph whose edges all weigh 1.0 a
  shortest path is a fewest-hop path, so Dijkstra's distances must equal
  BFS levels. Those levels are themselves checked against a brute-force
  frontier expansion written below, not taken from ``bfs_levels`` alone.
* **The paths themselves.** A distance is only believable if a route of
  that exact cost exists, so every reconstructed path is walked edge by
  edge, each hop confirmed to be a real edge, and the weights summed and
  compared with the distance that was reported.

:func:`dijkstra` and :func:`dijkstra_linear_scan` are also compared with
each other across many seeded graphs, but that is deliberately the weakest
claim in the file, not the strongest: it prices the container, and two
implementations of one algorithm can agree on a wrong answer.

The last class tests the assignment requirement rather than the maths -
that the frontier really is the Week 3 :class:`PriorityQueue` and not
``heapq``. That check reads the module's own source and parses its import
statements, because the module docstring mentions ``heapq`` in prose and a
plain substring search would fail on the documentation.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import ast
import inspect
import math
import random
from typing import Any, Dict, List, Tuple

import pytest

import src.graphs.dijkstra as dijkstra_module
from src.graphs.bfs import bfs_levels
from src.graphs.dijkstra import (
    dijkstra,
    dijkstra_linear_scan,
    reconstruct_path,
    shortest_path,
)
from src.graphs.graph import Graph
from src.structures.heap import PriorityQueue
from src.utils.graph_generator import (
    complete_graph,
    cycle_graph,
    path_graph,
    sparse_graph,
    weighted_graph,
)

#: Both frontier implementations. Every property that is true of the
#: algorithm rather than of its container is parametrised over this.
IMPLEMENTATIONS = [dijkstra, dijkstra_linear_scan]
IMPLEMENTATION_IDS = ["heap", "linear_scan"]


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def bellman_ford(graph: Graph, source: Any) -> Dict[Any, float]:
    """Shortest distances by repeated relaxation, with no priority ordering.

    This is the definition of a shortest path turned directly into code:
    keep relaxing every edge until nothing improves. It settles nodes in no
    particular order, keeps no frontier, and needs no argument about
    non-negative weights, so it shares no mechanism with either function
    under test. That independence is the whole point of having it here.

    Args:
        graph: The graph to search. Not modified.
        source: The node to measure from.

    Returns:
        Every node mapped to its distance from ``source``, ``math.inf``
        when there is no route.
    """
    distances: Dict[Any, float] = {node: math.inf for node in graph.nodes()}
    distances[source] = 0.0

    arcs = [
        (node, neighbour, weight)
        for node in graph.nodes()
        for neighbour, weight in graph.neighbor_items(node)
    ]

    for _ in range(max(graph.node_count - 1, 0)):
        improved = False
        for tail, head, weight in arcs:
            if distances[tail] + weight < distances[head]:
                distances[head] = distances[tail] + weight
                improved = True
        if not improved:
            break

    return distances


def breadth_expansion(graph: Graph, source: Any) -> Dict[Any, int]:
    """Hop counts by brute-force frontier expansion, independent of ``bfs.py``.

    Level 0 is the source; level k + 1 is everything one edge out from
    level k that has not been seen yet. No queue, no visited-on-push
    subtlety, just set arithmetic on whole frontiers - slower than BFS and
    much harder to get quietly wrong.

    Args:
        graph: The graph to expand through. Not modified.
        source: The node at level 0.

    Returns:
        Each reachable node mapped to its hop count. Unreachable nodes are
        absent, matching :func:`src.graphs.bfs.bfs_levels`.
    """
    levels: Dict[Any, int] = {source: 0}
    frontier = {source}
    depth = 0

    while frontier:
        depth += 1
        nxt = set()
        for node in frontier:
            for neighbour in graph.neighbors(node):
                if neighbour not in levels:
                    levels[neighbour] = depth
                    nxt.add(neighbour)
        frontier = nxt

    return levels


def path_cost(graph: Graph, path: List[Any]) -> float:
    """Sum the weights along ``path``, asserting every hop is a real edge.

    Args:
        graph: The graph the path is claimed to run through.
        path: Nodes in order, as :func:`reconstruct_path` returns them.

    Returns:
        The total weight of the route, 0.0 for a path of one node.
    """
    total = 0.0
    for tail, head in zip(path, path[1:]):
        assert graph.has_edge(tail, head), f"{tail!r} -> {head!r} is not an edge"
        total += graph.get_edge_weight(tail, head)
    return total


def assert_distances_match(actual: Dict[Any, float], oracle: Dict[Any, float]) -> None:
    """Fail if ``actual`` and ``oracle`` disagree on any node's distance."""
    assert set(actual) == set(oracle)
    for node, expected in oracle.items():
        if expected == math.inf:
            assert actual[node] == math.inf, f"{node!r} should be unreachable"
        else:
            assert actual[node] == pytest.approx(expected), f"distance to {node!r}"


# ----------------------------------------------------------------------
# Seeded graph families
# ----------------------------------------------------------------------
def make_weighted(density: float, directed: bool, connected: bool, seed: int) -> Graph:
    """Build one reproducible 24-node weighted graph."""
    return weighted_graph(
        24, density=density, directed=directed, connected=connected, seed=seed
    )


#: Sparse through dense, directed and undirected, forced-connected and left
#: to fall apart into components. The disconnected cases are what exercise
#: the math.inf branch on real graphs rather than on a hand-built toy.
WEIGHTED_CASES = [
    pytest.param(
        density,
        directed,
        connected,
        seed,
        id=(
            f"{shape}-{'directed' if directed else 'undirected'}"
            f"-{'connected' if connected else 'disconnected'}-seed{seed}"
        ),
    )
    for shape, density in [("sparse", 0.03), ("medium", 0.08), ("dense", 0.25)]
    for directed in (False, True)
    for connected in (False, True)
    for seed in (0, 1, 2)
]


def make_unweighted(kind: str, directed: bool, seed: int) -> Graph:
    """Build one reproducible unweighted graph of the named shape."""
    if kind == "sparse":
        return sparse_graph(24, avg_degree=3, directed=directed, seed=seed)
    if kind == "path":
        return path_graph(12, directed=directed)
    if kind == "cycle":
        return cycle_graph(12, directed=directed)
    return complete_graph(8, directed=directed)


UNWEIGHTED_CASES = [
    pytest.param(
        kind,
        directed,
        seed,
        id=f"{kind}-{'directed' if directed else 'undirected'}-seed{seed}",
    )
    for kind in ("sparse", "path", "cycle", "complete")
    for directed in (False, True)
    for seed in (0, 1)
]


def tie_heavy_graph(seed: int) -> Graph:
    """A small graph with weights drawn from {1, 2, 3}, so ties are common.

    Random weights to two decimals almost never tie, and a tie is exactly
    where two implementations are free to disagree on which parent to
    record. Drawing from three integers instead makes ties the normal case.
    """
    rng = random.Random(seed)
    graph = Graph(directed=rng.random() < 0.5, weighted=True)
    size = rng.randint(2, 14)
    for node in range(size):
        graph.add_node(node)
    for _ in range(rng.randint(0, size * 2)):
        graph.add_edge(
            rng.randrange(size), rng.randrange(size), float(rng.randint(1, 3))
        )
    return graph


# ----------------------------------------------------------------------
# The hand-computed case
# ----------------------------------------------------------------------
class TestHandComputedGraph:
    """One small graph whose every answer was worked out on paper.

    The graph, all edges undirected and weighted::

        S -A 1     S -B 5     A -B 2     B -C 1
        A -C 6     C -D 3     B -D 7     D -E 2     F isolated

    Worked through by hand, settling the nearest unsettled node each time:

    1. S settles at 0. Its edges offer A at 1 and B at 5.
    2. A is nearest at 1, parent S. Through A, B improves to 1 + 2 = 3,
       parent A - the two-hop route S-A-B beats the direct edge S-B of 5.
       C is offered 1 + 6 = 7, parent A.
    3. B is next at 3, parent A. Through B, C improves to 3 + 1 = 4,
       parent B - again the longer route wins, because the single edge
       A-C costs 6 and the detour through B costs 4. D is offered
       3 + 7 = 10, parent B.
    4. C settles at 4, parent B. Through C, D improves to 4 + 3 = 7,
       parent C, beating the direct B-D edge at 10.
    5. D settles at 7, parent C, and offers E at 7 + 2 = 9, parent D.
    6. E settles at 9, parent D.
    7. F is in no component of S, so it keeps inf and parent None.

    So S-E costs 9 by the route S, A, B, C, D, E - five hops where two
    exist, which is the case a greedy nearest-neighbour walk gets wrong.

    The nodes are inserted in the deliberately unhelpful order S, E, D, C,
    B, A, F, which is close to the reverse of the order Dijkstra settles
    them in. Distances do not depend on insertion order, but an
    implementation that relaxed every edge once in node order instead of
    settling by distance would answer B = 5 here, and that mistake is
    invisible on a graph whose nodes happen to be listed nearest-first.
    """

    #: Every expected value below was computed by hand, not by running the
    #: module. Nothing in this class may be regenerated from its output.
    EXPECTED_DISTANCES = {
        "S": 0.0,
        "A": 1.0,
        "B": 3.0,
        "C": 4.0,
        "D": 7.0,
        "E": 9.0,
        "F": math.inf,
    }
    EXPECTED_PREDECESSORS = {
        "S": None,
        "A": "S",
        "B": "A",
        "C": "B",
        "D": "C",
        "E": "D",
        "F": None,
    }

    @staticmethod
    def build() -> Graph:
        graph = Graph(weighted=True)
        # Insertion order first, and on purpose: see the class docstring.
        for node in ["S", "E", "D", "C", "B", "A", "F"]:
            graph.add_node(node)
        for tail, head, weight in [
            ("S", "A", 1.0),
            ("S", "B", 5.0),
            ("A", "B", 2.0),
            ("B", "C", 1.0),
            ("A", "C", 6.0),
            ("C", "D", 3.0),
            ("B", "D", 7.0),
            ("D", "E", 2.0),
        ]:
            graph.add_edge(tail, head, weight)
        graph.add_node("F")
        return graph

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_exact_distance_map(self, implementation):
        distances, _predecessors = implementation(self.build(), "S")
        assert distances == self.EXPECTED_DISTANCES

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_exact_predecessor_map(self, implementation):
        _distances, predecessors = implementation(self.build(), "S")
        assert predecessors == self.EXPECTED_PREDECESSORS

    def test_exact_path_to_every_node(self):
        """The route, not just its cost - a wrong parent can still total right."""
        expected = {
            "S": ["S"],
            "A": ["S", "A"],
            "B": ["S", "A", "B"],
            "C": ["S", "A", "B", "C"],
            "D": ["S", "A", "B", "C", "D"],
            "E": ["S", "A", "B", "C", "D", "E"],
            "F": [],
        }
        _distances, predecessors = dijkstra(self.build(), "S")
        for target, route in expected.items():
            assert reconstruct_path(predecessors, "S", target) == route, target

    @pytest.mark.parametrize(
        "target, route, cost",
        [
            ("B", ["S", "A", "B"], 3.0),
            ("C", ["S", "A", "B", "C"], 4.0),
            ("E", ["S", "A", "B", "C", "D", "E"], 9.0),
        ],
        ids=["two_hop_beats_one_hop", "detour_beats_direct_edge", "full_chain"],
    )
    def test_shortest_path_returns_the_hand_computed_route(self, target, route, cost):
        assert shortest_path(self.build(), "S", target) == (route, cost)

    @pytest.mark.parametrize(
        "target, one_hop_offer",
        # Each pair is (node, what the single edge into it would have cost):
        #   B: S is at 0 and the edge S-B weighs 5, so the one-hop offer is 5.
        #   C: A is at 1 and the edge A-C weighs 6, so the offer is 7.
        #   D: B is at 3 and the edge B-D weighs 7, so the offer is 10.
        [("B", 5.0), ("C", 7.0), ("D", 10.0)],
        ids=["S_to_B", "A_to_C", "B_to_D"],
    )
    def test_the_greedy_single_edge_choice_is_not_taken(self, target, one_hop_offer):
        """Each of these nodes has a one-hop offer that a longer route beats."""
        distances = dijkstra(self.build(), "S")[0]
        assert distances[target] < one_hop_offer

    def test_the_hand_computed_answer_survives_bellman_ford(self):
        """Paper and a second algorithm, agreeing on the same numbers."""
        graph = self.build()
        assert_distances_match(bellman_ford(graph, "S"), self.EXPECTED_DISTANCES)


# ----------------------------------------------------------------------
# Unreachable nodes
# ----------------------------------------------------------------------
class TestUnreachableNodes:
    """No path is a result, not an exception."""

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_isolated_node_is_inf_with_no_predecessor(self, implementation):
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 2.0)
        graph.add_node("lonely")
        distances, predecessors = implementation(graph, "A")
        assert distances["lonely"] == math.inf
        assert predecessors["lonely"] is None

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_a_whole_second_component_is_inf(self, implementation):
        graph = Graph(weighted=True)
        for tail, head, weight in [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("X", "Y", 1.0),
            ("Y", "Z", 1.0),
        ]:
            graph.add_edge(tail, head, weight)
        distances, predecessors = implementation(graph, "A")
        assert [distances[n] for n in ("X", "Y", "Z")] == [math.inf] * 3
        assert [predecessors[n] for n in ("X", "Y", "Z")] == [None] * 3

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_every_node_of_the_graph_is_a_key(self, implementation):
        """Unreachable means inf, never a missing key - callers never branch."""
        graph = weighted_graph(24, density=0.03, connected=False, seed=0)
        distances, predecessors = implementation(graph, graph.nodes()[0])
        assert set(distances) == set(graph.nodes())
        assert set(predecessors) == set(graph.nodes())
        assert math.inf in distances.values(), "expected a disconnected graph"

    def test_source_in_a_component_of_its_own_reaches_only_itself(self):
        graph = Graph(weighted=True)
        graph.add_node("alone")
        graph.add_edge("A", "B", 1.0)
        distances, predecessors = dijkstra(graph, "alone")
        assert distances == {"alone": 0.0, "A": math.inf, "B": math.inf}
        assert set(predecessors.values()) == {None}


# ----------------------------------------------------------------------
# Negative weights
# ----------------------------------------------------------------------
class TestNegativeWeights:
    """Dijkstra is undefined for them, so it refuses rather than guesses."""

    @staticmethod
    def chain_with_negative_at(position: int, directed: bool = False) -> Graph:
        """A five-node chain A-B-C-D-E with one negative weight placed in it."""
        graph = Graph(directed=directed, weighted=True)
        nodes = ["A", "B", "C", "D", "E"]
        for index, (tail, head) in enumerate(zip(nodes, nodes[1:])):
            graph.add_edge(tail, head, -2.0 if index == position else 1.0)
        return graph

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    @pytest.mark.parametrize(
        "position, edge",
        [(0, "('A', 'B')"), (1, "('B', 'C')"), (3, "('D', 'E')")],
        ids=["first_edge", "second_edge", "last_edge"],
    )
    def test_negative_weight_raises_wherever_it_sits(
        self, implementation, position, edge
    ):
        """Not only the first edge examined: depth in the graph changes nothing."""
        graph = self.chain_with_negative_at(position)
        with pytest.raises(ValueError) as excinfo:
            implementation(graph, "A")
        message = str(excinfo.value)
        assert edge in message, f"message should name the edge: {message}"
        assert "-2.0" in message

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_negative_weight_raises_on_a_directed_graph_too(self, implementation):
        graph = self.chain_with_negative_at(3, directed=True)
        with pytest.raises(ValueError, match=r"\('D', 'E'\)"):
            implementation(graph, "A")

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_a_negative_edge_the_search_never_reaches_still_raises(
        self, implementation
    ):
        """The whole edge set is checked up front, so the answer is deterministic.

        This edge sits in a component the search would never enter. Checking
        lazily would let the same broken graph pass or fail depending on the
        source the caller happened to pick.
        """
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("Y", "Z", -5.0)
        with pytest.raises(ValueError, match=r"\('Y', 'Z'\)"):
            implementation(graph, "A")

    def test_shortest_path_propagates_the_refusal(self):
        graph = self.chain_with_negative_at(2)
        with pytest.raises(ValueError, match=r"\('C', 'D'\)"):
            shortest_path(graph, "A", "E")

    def test_the_message_says_what_is_required(self):
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", -1.5)
        with pytest.raises(ValueError, match="weights must be >= 0"):
            dijkstra(graph, "A")

    def test_zero_is_not_negative_and_is_accepted(self):
        """The boundary: >= 0 means 0.0 is a legal weight, not a rejected one."""
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 0.0)
        graph.add_edge("B", "C", 0.0)
        graph.add_edge("A", "C", 5.0)
        distances, predecessors = dijkstra(graph, "A")
        assert distances == {"A": 0.0, "B": 0.0, "C": 0.0}
        assert predecessors["C"] == "B", "the free two-hop route should win"
        assert shortest_path(graph, "A", "C") == (["A", "B", "C"], 0.0)


# ----------------------------------------------------------------------
# Against Bellman-Ford
# ----------------------------------------------------------------------
class TestAgreesWithBellmanFord:
    """Distances checked against a second algorithm that shares no machinery."""

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    @pytest.mark.parametrize("density, directed, connected, seed", WEIGHTED_CASES)
    def test_random_weighted_graphs(
        self, implementation, density, directed, connected, seed
    ):
        graph = make_weighted(density, directed, connected, seed)
        source = graph.nodes()[0]
        distances, _predecessors = implementation(graph, source)
        assert_distances_match(distances, bellman_ford(graph, source))

    @pytest.mark.parametrize("seed", range(20))
    def test_tie_heavy_small_graphs(self, seed):
        graph = tie_heavy_graph(seed)
        source = graph.nodes()[0]
        assert_distances_match(dijkstra(graph, source)[0], bellman_ford(graph, source))

    @pytest.mark.parametrize("density, directed, connected, seed", WEIGHTED_CASES)
    def test_every_source_not_just_the_first(self, density, directed, connected, seed):
        """Node 0 is where the generator roots its spanning tree; others differ."""
        graph = make_weighted(density, directed, connected, seed)
        for source in graph.nodes()[:4]:
            assert_distances_match(
                dijkstra(graph, source)[0], bellman_ford(graph, source)
            )


# ----------------------------------------------------------------------
# Against breadth-first search
# ----------------------------------------------------------------------
class TestAgreesWithBreadthFirstSearch:
    """On unit weights a cheapest path is a fewest-hop path, so BFS is the oracle."""

    @pytest.mark.parametrize("kind, directed, seed", UNWEIGHTED_CASES)
    def test_distances_equal_brute_force_hop_counts(self, kind, directed, seed):
        graph = make_unweighted(kind, directed, seed)
        source = graph.nodes()[0]
        distances = dijkstra(graph, source)[0]
        expected = breadth_expansion(graph, source)
        for node in graph.nodes():
            if node in expected:
                assert distances[node] == float(expected[node]), node
            else:
                assert distances[node] == math.inf, node

    @pytest.mark.parametrize("kind, directed, seed", UNWEIGHTED_CASES)
    def test_distances_equal_bfs_levels(self, kind, directed, seed):
        """The same claim against the shipped BFS, which the module cites."""
        graph = make_unweighted(kind, directed, seed)
        source = graph.nodes()[0]
        distances = dijkstra(graph, source)[0]
        levels = bfs_levels(graph, source)
        assert {n: int(d) for n, d in distances.items() if d != math.inf} == levels

    @pytest.mark.parametrize("kind, directed, seed", UNWEIGHTED_CASES)
    def test_both_implementations_agree_with_bfs(self, kind, directed, seed):
        graph = make_unweighted(kind, directed, seed)
        source = graph.nodes()[0]
        levels = bfs_levels(graph, source)
        scanned = dijkstra_linear_scan(graph, source)[0]
        assert {n: int(d) for n, d in scanned.items() if d != math.inf} == levels

    def test_explicit_unit_weights_match_hop_counts_too(self):
        """A *weighted* graph whose weights all happen to be 1.0 must also agree."""
        pairs = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (2, 5)]
        unit = Graph(weighted=True)
        hops = Graph()
        for tail, head in pairs:
            unit.add_edge(tail, head, 1.0)
            hops.add_edge(tail, head)
        assert dijkstra(unit, 0)[0] == dijkstra(hops, 0)[0]
        assert dijkstra(unit, 0)[0] == {
            node: float(level) for node, level in breadth_expansion(hops, 0).items()
        }


# ----------------------------------------------------------------------
# Heap frontier against linear-scan frontier
# ----------------------------------------------------------------------
class TestTwoImplementationsAgree:
    """The benchmark's premise: same answers, so only the container differs.

    The two share their relaxation and their tie-breaking key, so this is
    the weakest of the file's checks - two copies of one algorithm can be
    wrong together. It is here because the Week 4 benchmark compares their
    running times, and that comparison is meaningless unless they are doing
    the same work.
    """

    @pytest.mark.parametrize("density, directed, connected, seed", WEIGHTED_CASES)
    def test_distances_and_predecessors_are_identical(
        self, density, directed, connected, seed
    ):
        graph = make_weighted(density, directed, connected, seed)
        source = graph.nodes()[0]
        assert dijkstra(graph, source) == dijkstra_linear_scan(graph, source)

    @pytest.mark.parametrize("seed", range(20))
    def test_they_break_ties_the_same_way(self, seed):
        """Where several routes tie, both must record the same parent."""
        graph = tie_heavy_graph(seed)
        source = graph.nodes()[0]
        assert dijkstra(graph, source) == dijkstra_linear_scan(graph, source)

    @pytest.mark.parametrize("density, directed, connected, seed", WEIGHTED_CASES)
    def test_total_path_cost_agrees_node_by_node(
        self, density, directed, connected, seed
    ):
        """Not just the distance map: the routes themselves cost the same."""
        graph = make_weighted(density, directed, connected, seed)
        source = graph.nodes()[0]
        heap_distances, heap_parents = dijkstra(graph, source)
        scan_distances, scan_parents = dijkstra_linear_scan(graph, source)
        for node in graph.nodes():
            heap_route = reconstruct_path(heap_parents, source, node)
            scan_route = reconstruct_path(scan_parents, source, node)
            assert bool(heap_route) == bool(scan_route), node
            if heap_route:
                assert path_cost(graph, heap_route) == pytest.approx(
                    path_cost(graph, scan_route)
                ), node
                assert path_cost(graph, heap_route) == pytest.approx(
                    heap_distances[node]
                ), node
                assert scan_distances[node] == pytest.approx(heap_distances[node])


# ----------------------------------------------------------------------
# reconstruct_path
# ----------------------------------------------------------------------
class TestReconstructPath:
    """Turning a predecessor map back into a route."""

    def test_source_is_its_own_path(self):
        predecessors = {"A": None, "B": "A"}
        assert reconstruct_path(predecessors, "A", "A") == ["A"]

    def test_unreachable_target_is_the_empty_list(self):
        assert reconstruct_path({"A": None, "X": None}, "A", "X") == []

    def test_target_absent_from_the_map_is_the_empty_list(self):
        assert reconstruct_path({"A": None, "B": "A"}, "A", "Q") == []

    def test_a_malformed_map_with_a_cycle_does_not_hang(self):
        """A cycle in the parents means "no path", reported rather than looped."""
        assert reconstruct_path({"B": "C", "C": "B", "A": None}, "A", "B") == []

    def test_route_is_built_backwards_and_reversed(self):
        predecessors = {"A": None, "C": "A", "B": "C", "D": "B"}
        assert reconstruct_path(predecessors, "A", "D") == ["A", "C", "B", "D"]

    @pytest.mark.parametrize("density, directed, connected, seed", WEIGHTED_CASES)
    def test_every_path_is_real_and_costs_what_was_reported(
        self, density, directed, connected, seed
    ):
        """The strongest single claim here: the distance is backed by a route.

        ``path_cost`` asserts each consecutive pair is an actual edge, so a
        route through an edge that does not exist fails before the sum is
        even compared.
        """
        graph = make_weighted(density, directed, connected, seed)
        source = graph.nodes()[0]
        distances, predecessors = dijkstra(graph, source)
        for node in graph.nodes():
            route = reconstruct_path(predecessors, source, node)
            if distances[node] == math.inf:
                assert route == [], node
                continue
            assert route[0] == source and route[-1] == node
            assert len(set(route)) == len(route), f"route to {node!r} repeats a node"
            assert path_cost(graph, route) == pytest.approx(distances[node]), node


# ----------------------------------------------------------------------
# shortest_path
# ----------------------------------------------------------------------
class TestShortestPath:
    """The single-pair wrapper."""

    @staticmethod
    def two_components() -> Graph:
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 2.0)
        graph.add_edge("B", "C", 3.0)
        graph.add_edge("Y", "Z", 1.0)
        return graph

    def test_unreachable_target_is_empty_path_and_inf(self):
        assert shortest_path(self.two_components(), "A", "Z") == ([], math.inf)

    def test_source_to_itself_is_zero(self):
        assert shortest_path(self.two_components(), "A", "A") == (["A"], 0.0)

    def test_returns_the_route_and_its_cost(self):
        assert shortest_path(self.two_components(), "A", "C") == (["A", "B", "C"], 5.0)

    @pytest.mark.parametrize(
        "source, target, missing",
        [("Q", "A", "source"), ("A", "Q", "target")],
        ids=["missing_source", "missing_target"],
    )
    def test_a_node_that_does_not_exist_is_an_error(self, source, target, missing):
        """Unreachable is a result; absent is a mistake, and they differ."""
        with pytest.raises(KeyError, match=f"{missing} node 'Q' is not in the graph"):
            shortest_path(self.two_components(), source, target)

    @pytest.mark.parametrize("density, directed, connected, seed", WEIGHTED_CASES[:12])
    def test_wrapper_matches_a_direct_dijkstra_call(
        self, density, directed, connected, seed
    ):
        graph = make_weighted(density, directed, connected, seed)
        source = graph.nodes()[0]
        distances, predecessors = dijkstra(graph, source)
        for target in graph.nodes():
            route, cost = shortest_path(graph, source, target)
            assert route == reconstruct_path(predecessors, source, target)
            assert cost == distances[target]


# ----------------------------------------------------------------------
# Directed graphs
# ----------------------------------------------------------------------
class TestDirectedGraphs:
    """Direction is part of the answer, not a detail of the representation."""

    def test_the_reverse_route_is_longer(self):
        """A -> B costs 1 going forward and 11 coming back round the cycle."""
        graph = Graph(directed=True, weighted=True)
        for tail, head, weight in [("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 10.0)]:
            graph.add_edge(tail, head, weight)
        assert shortest_path(graph, "A", "B") == (["A", "B"], 1.0)
        assert shortest_path(graph, "B", "A") == (["B", "C", "A"], 11.0)

    def test_the_reverse_route_is_absent(self):
        graph = Graph(directed=True, weighted=True)
        graph.add_edge("A", "B", 1.0)
        assert shortest_path(graph, "A", "B") == (["A", "B"], 1.0)
        assert shortest_path(graph, "B", "A") == ([], math.inf)
        distances, predecessors = dijkstra(graph, "B")
        assert distances["A"] == math.inf and predecessors["A"] is None

    def test_a_one_way_shortcut_is_used_in_only_one_direction(self):
        graph = Graph(directed=True, weighted=True)
        for tail, head, weight in [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "D", 1.0),
            ("A", "D", 2.0),
            ("D", "A", 9.0),
        ]:
            graph.add_edge(tail, head, weight)
        assert shortest_path(graph, "A", "D") == (["A", "D"], 2.0)
        assert shortest_path(graph, "D", "C") == (["D", "A", "B", "C"], 11.0)

    def test_the_same_edges_undirected_are_cheaper_or_equal(self):
        """Undirecting a graph can only add routes, so no distance can rise.

        C and D are reachable from A only by traversing C -> B backwards,
        so they are unreachable while the edges point one way and finite
        once they do not.
        """
        pairs = [("A", "B", 1.0), ("C", "B", 2.0), ("C", "D", 3.0)]
        directed = Graph(directed=True, weighted=True)
        undirected = Graph(weighted=True)
        for tail, head, weight in pairs:
            directed.add_edge(tail, head, weight)
            undirected.add_edge(tail, head, weight)
        one_way = dijkstra(directed, "A")[0]
        both_ways = dijkstra(undirected, "A")[0]
        for node in directed.nodes():
            assert both_ways[node] <= one_way[node], node
        assert one_way["C"] == math.inf and one_way["D"] == math.inf
        assert both_ways["C"] == 3.0 and both_ways["D"] == 6.0


# ----------------------------------------------------------------------
# Degenerate inputs
# ----------------------------------------------------------------------
class TestEdgeCases:
    """Empty, single, self-loop, and the source that is not there."""

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_missing_source_raises_key_error(self, implementation):
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 1.0)
        with pytest.raises(KeyError, match="source node 'Q' is not in the graph"):
            implementation(graph, "Q")

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_empty_graph_has_no_valid_source(self, implementation):
        """There is no node to measure from, so any source is a KeyError."""
        with pytest.raises(KeyError):
            implementation(Graph(weighted=True), 0)

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_single_node_graph(self, implementation):
        graph = Graph(weighted=True)
        graph.add_node("only")
        assert implementation(graph, "only") == ({"only": 0.0}, {"only": None})

    def test_single_node_shortest_path_to_itself(self):
        graph = Graph(weighted=True)
        graph.add_node("only")
        assert shortest_path(graph, "only", "only") == (["only"], 0.0)

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_self_loop_never_improves_the_source(self, implementation):
        """Going round a non-negative loop cannot shorten anything."""
        graph = Graph(weighted=True)
        graph.add_edge("A", "A", 3.0)
        graph.add_edge("A", "B", 2.0)
        distances, predecessors = implementation(graph, "A")
        assert distances == {"A": 0.0, "B": 2.0}
        assert predecessors == {"A": None, "B": "A"}

    def test_zero_weight_self_loop_leaves_the_source_at_zero(self):
        graph = Graph(weighted=True)
        graph.add_edge("A", "A", 0.0)
        assert dijkstra(graph, "A") == ({"A": 0.0}, {"A": None})

    def test_parallel_edge_readd_keeps_only_the_latest_weight(self):
        """Re-adding an edge replaces its weight; the search must see the new one."""
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 9.0)
        graph.add_edge("A", "B", 1.0)
        assert dijkstra(graph, "A")[0]["B"] == 1.0

    def test_distances_are_floats_even_from_integer_weights(self):
        graph = Graph(weighted=True)
        graph.add_edge("A", "B", 4)
        distances = dijkstra(graph, "A")[0]
        assert all(isinstance(value, float) for value in distances.values())

    @pytest.mark.parametrize(
        "implementation", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS
    )
    def test_the_graph_is_not_modified(self, implementation):
        graph = sparse_graph(20, avg_degree=3, weighted=True, seed=5)
        before = (graph.node_count, graph.edge_count, graph.nodes(), graph.edges())
        implementation(graph, graph.nodes()[0])
        after = (graph.node_count, graph.edge_count, graph.nodes(), graph.edges())
        assert before == after

    def test_repeated_calls_return_the_same_answer(self):
        """No state carried between runs, so the result is reproducible."""
        graph = make_weighted(0.08, False, False, 1)
        first = dijkstra(graph, graph.nodes()[0])
        second = dijkstra(graph, graph.nodes()[0])
        assert first == second


# ----------------------------------------------------------------------
# The assignment requirement: the Week 3 heap, not heapq
# ----------------------------------------------------------------------
class TestUsesTheWeek3PriorityQueue:
    """The frontier must be the Week 3 PriorityQueue, and heapq must be absent.

    The module's own docstring contains the word ``heapq`` in prose, where
    it explains why it is *not* used. A substring search over the source
    would therefore fail on the documentation, so the import statements are
    parsed instead and only real imports are looked at.
    """

    @staticmethod
    def import_nodes() -> List[ast.stmt]:
        source = inspect.getsource(dijkstra_module)
        tree = ast.parse(source)
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

    def test_heapq_is_not_imported(self):
        for node in self.import_nodes():
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            assert "heapq" not in names, f"heapq imported on line {node.lineno}"

    def test_heapq_is_not_bound_in_the_module_namespace(self):
        """Behavioural counterpart: no smuggled ``__import__`` or alias either."""
        assert not hasattr(dijkstra_module, "heapq")

    def test_priority_queue_is_imported_from_the_week3_heap(self):
        found = [
            node
            for node in self.import_nodes()
            if isinstance(node, ast.ImportFrom)
            and node.module == "src.structures.heap"
            and any(alias.name == "PriorityQueue" for alias in node.names)
        ]
        assert found, "expected: from src.structures.heap import PriorityQueue"

    def test_the_imported_name_is_the_week3_class_itself(self):
        assert dijkstra_module.PriorityQueue is PriorityQueue

    def test_every_frontier_push_goes_through_the_week3_priority_queue(
        self, monkeypatch
    ):
        """Proof by use, not by import: the queue is watched while it works.

        An import can sit unused. Swapping in a subclass that records its
        own calls shows the search really drives its frontier through the
        Week 3 queue.
        """
        record = {"constructed": 0, "pushes": 0, "pops": 0}

        class WatchedQueue(PriorityQueue):
            def __init__(self, *args, **kwargs):
                record["constructed"] += 1
                super().__init__(*args, **kwargs)

            def push(self, item, priority):
                record["pushes"] += 1
                return super().push(item, priority)

            def pop_with_priority(self):
                record["pops"] += 1
                return super().pop_with_priority()

        monkeypatch.setattr(dijkstra_module, "PriorityQueue", WatchedQueue)
        graph = TestHandComputedGraph.build()
        distances, predecessors = dijkstra(graph, "S")

        assert record["constructed"] == 1
        assert record["pushes"] >= graph.node_count - 1, "every node should be queued"
        assert record["pops"] >= 1
        # The watched run must still be correct, not merely instrumented.
        assert distances == TestHandComputedGraph.EXPECTED_DISTANCES
        assert predecessors == TestHandComputedGraph.EXPECTED_PREDECESSORS

    def test_lazy_deletion_pushes_more_entries_than_there_are_nodes(self):
        """The documented cost of having no decrease-key, measured.

        An improved distance is pushed as a second entry rather than edited
        in place, so on a graph built to improve the same nodes repeatedly
        the push count must exceed the node count.
        """
        pushes = {"n": 0}

        class CountingQueue(PriorityQueue):
            def push(self, item, priority):
                pushes["n"] += 1
                return super().push(item, priority)

        # A ladder: each rung offers a cheaper route to everything past it,
        # so later nodes are relaxed several times over.
        graph = Graph(directed=True, weighted=True)
        for index in range(8):
            graph.add_edge(index, index + 1, 10.0)
            for ahead in range(index + 2, 9):
                graph.add_edge(index, ahead, float(30 - index - ahead))

        original = dijkstra_module.PriorityQueue
        dijkstra_module.PriorityQueue = CountingQueue
        try:
            dijkstra(graph, 0)
        finally:
            dijkstra_module.PriorityQueue = original

        assert pushes["n"] > graph.node_count, (
            "lazy deletion should push a node once per improvement, "
            f"but only {pushes['n']} pushes were made for "
            f"{graph.node_count} nodes"
        )
