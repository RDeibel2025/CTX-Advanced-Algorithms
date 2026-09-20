"""Tests for breadth-first search.

BFS has exactly one defining property, and everything else it offers is a
consequence of it: **a breadth-first traversal visits nodes in non-decreasing
hop distance from the start**. Shortest paths in edges, the level map and the
BFS tree all fall out of that single fact, so that fact is what is proven
here, not the code that produces it.

The proof needs a yardstick that owes nothing to :mod:`src.graphs.bfs`.
:func:`hop_levels` in this file is that yardstick: a brute-force level
expansion that starts with the frontier ``{start}`` and repeatedly replaces it
with the neighbours nobody has reached yet, counting rounds. It uses no queue,
so it cannot share a queueing bug with the implementation, and it reads the
graph back out of :meth:`~src.graphs.graph.Graph.edges` rather than through
:meth:`~src.graphs.graph.Graph.neighbors`, the accessor BFS itself walks, so a
fault in that accessor cannot cancel out either. Every ordering claim, every
level and every tree depth below is checked against this oracle. Checking
``bfs()`` against ``bfs_levels()`` would only prove the two agree with each
other.

Four further things are tested because they are where a traversal usually
breaks:

* **Disconnected graphs.** ``bfs`` must report every node of a four-component
  graph, while ``bfs_component`` must stay inside the one component it was
  given. The two are easy to write so that one quietly does the other's job.
* **Cycles and self-loops.** A traversal that marks nodes late, or not at all,
  loops here forever instead of failing an assertion.
* **Direction.** On a directed graph the reverse edge is not a route, so the
  reachable set from a sink is the sink alone.
* **Weights.** BFS must ignore them completely. The test builds a graph whose
  one-hop route is expensive and whose cheap route is three hops, and requires
  BFS to call the expensive node level 1. That is the line between BFS and
  Dijkstra, and it is tested by comparing the weighted graph against an
  unweighted twin of identical shape, not by running Dijkstra.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set

import pytest

from src.graphs.bfs import bfs, bfs_component, bfs_levels, bfs_tree
from src.graphs.graph import Graph
from src.utils import graph_generator as gg


# ----------------------------------------------------------------------
# The independent oracle
# ----------------------------------------------------------------------
def neighbour_map(graph: Graph) -> Dict[Any, List[Any]]:
    """Rebuild the adjacency from ``graph.edges()`` alone.

    Deliberately not ``graph.neighbors()``: BFS walks that accessor, and an
    oracle that walked it too could inherit its mistakes. ``edges()`` lists an
    undirected edge once, so the reverse direction is restored here, and a
    self-loop is added once because both of its endpoints are the same node.
    """
    adjacency: Dict[Any, List[Any]] = {node: [] for node in graph.nodes()}
    for edge in graph.edges():
        u, v = edge[0], edge[1]
        adjacency[u].append(v)
        if not graph.directed and u != v:
            adjacency[v].append(u)
    return adjacency


def hop_levels(graph: Graph, start: Any) -> Dict[Any, int]:
    """Hop distance from ``start``, by brute-force breadth expansion.

    No queue and no visit order: the frontier is a whole ring at a time, and
    the round counter is the level. This is the definition of hop distance
    written out literally, which is what makes it a fair check on BFS.
    """
    levels: Dict[Any, int] = {start: 0}
    adjacency = neighbour_map(graph)
    frontier = [start]
    depth = 0

    while frontier:
        depth += 1
        next_frontier: List[Any] = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if neighbour not in levels:
                    levels[neighbour] = depth
                    next_frontier.append(neighbour)
        frontier = next_frontier

    return levels


def assert_non_decreasing(order: List[Any], levels: Dict[Any, int], label: str = "") -> None:
    """Fail if ``order`` ever steps back to a node closer to the start."""
    assert len(order) == len(set(order)), f"{label}: a node was visited twice"
    assert set(order) == set(levels), f"{label}: visited set is not the reachable set"

    previous = -1
    for position, node in enumerate(order):
        level = levels[node]
        assert level >= previous, (
            f"{label}: position {position} visits {node!r} at level {level} "
            f"after a node at level {previous}, so the traversal stepped back "
            f"towards the start"
        )
        previous = level


def assert_full_traversal(graph: Graph, order: List[Any]) -> None:
    """Check a whole-graph ``bfs()`` order on an **undirected** graph.

    Components partition an undirected graph, so a full traversal has to be
    one complete component after another, each of them in hop order from the
    node the sweep began at. That does not hold on a directed graph, where a
    later sweep can run into nodes an earlier one already took, so directed
    graphs are checked through :func:`bfs_component` instead.
    """
    assert not graph.directed, "this check is only valid on undirected graphs"
    assert len(order) == graph.node_count
    assert set(order) == set(graph.nodes())

    position = 0
    while position < len(order):
        root = order[position]
        levels = hop_levels(graph, root)
        segment = order[position : position + len(levels)]
        assert_non_decreasing(segment, levels, label=f"component of {root!r}")
        position += len(levels)


def sample_starts(graph: Graph) -> List[Any]:
    """A few start nodes per graph: first, middle and last in insertion order."""
    nodes = graph.nodes()
    if not nodes:
        return []
    return sorted({nodes[0], nodes[len(nodes) // 2], nodes[-1]}, key=nodes.index)


# ----------------------------------------------------------------------
# Graphs under test
# ----------------------------------------------------------------------
#: Generated graphs, by name. Every seed is fixed, so each name is one exact
#: graph on every run. Several of these come out disconnected, which is what
#: puts bfs() and bfs_component() under real pressure.
GENERATED_GRAPHS: Dict[str, Callable[[], Graph]] = {
    "sparse": lambda: gg.sparse_graph(60, seed=1),
    "sparse_connected": lambda: gg.sparse_graph(60, connected=True, seed=2),
    "sparse_directed": lambda: gg.sparse_graph(60, directed=True, seed=3),
    "random_thin": lambda: gg.random_graph(40, density=0.05, seed=4),
    "random_directed": lambda: gg.random_graph(40, density=0.08, directed=True, seed=5),
    "dense": lambda: gg.dense_graph(30, seed=6),
    "weighted": lambda: gg.weighted_graph(40, seed=7),
    "weighted_directed": lambda: gg.weighted_graph(40, directed=True, seed=8),
    "complete": lambda: gg.complete_graph(12),
    "path": lambda: gg.path_graph(30),
    "path_directed": lambda: gg.path_graph(30, directed=True),
    "cycle": lambda: gg.cycle_graph(20),
    "cycle_directed": lambda: gg.cycle_graph(20, directed=True),
}
GRAPH_NAMES = sorted(GENERATED_GRAPHS)
UNDIRECTED_NAMES = [name for name in GRAPH_NAMES if "directed" not in name]


def hand_graph() -> Graph:
    """The hand-drawn graph, small enough to check the answers by eye.

              a
             / \\
            b   c
           / \\   \\
          d   e - f

    Edges are added in the order listed in the body, and neighbour lists
    follow edge insertion order, so every expected answer below is fixed.
    """
    graph = Graph()
    for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("b", "e"), ("c", "f"), ("e", "f")]:
        graph.add_edge(u, v)
    return graph


def four_component_graph() -> Graph:
    """Nine nodes in four components: a triangle, a path, a loner and a pair.

    Built in that order, so insertion order is 1, 2, 3, 10, 11, 12, 20, 30, 31
    and the order a full traversal falls through the components is fixed.
    """
    graph = Graph()
    for u, v in [(1, 2), (2, 3), (1, 3), (10, 11), (11, 12)]:
        graph.add_edge(u, v)
    graph.add_node(20)
    graph.add_edge(30, 31)
    return graph


#: Each node of :func:`four_component_graph` mapped to its own component.
COMPONENTS: Dict[Any, Set[Any]] = {
    **{node: {1, 2, 3} for node in (1, 2, 3)},
    **{node: {10, 11, 12} for node in (10, 11, 12)},
    20: {20},
    **{node: {30, 31} for node in (30, 31)},
}


# ----------------------------------------------------------------------
# Visit order on a graph small enough to check by inspection
# ----------------------------------------------------------------------
class TestVisitOrderByInspection:
    """A six-node graph whose every answer was worked out on paper."""

    def test_order_from_the_root(self):
        """a, then its ring b c, then theirs: d e from b and f from c."""
        assert bfs(hand_graph(), "a") == ["a", "b", "c", "d", "e", "f"]

    def test_order_from_a_leaf(self):
        """From d the graph is walked backwards: d, b, then a and e, then c and f."""
        assert bfs(hand_graph(), "d") == ["d", "b", "a", "e", "c", "f"]

    def test_default_start_is_the_first_node_in_insertion_order(self):
        graph = hand_graph()
        assert bfs(graph) == bfs(graph, start=graph.nodes()[0])

    def test_levels_from_the_root(self):
        assert bfs_levels(hand_graph(), "a") == {"a": 0, "b": 1, "c": 1, "d": 2, "e": 2, "f": 2}

    def test_levels_from_a_leaf(self):
        """f is 3 hops from d through e, not 4 through c."""
        assert bfs_levels(hand_graph(), "d") == {"d": 0, "b": 1, "a": 2, "e": 2, "c": 3, "f": 3}

    def test_tree_from_the_root(self):
        """f touches c and e; c was queued first, so c is the parent."""
        assert bfs_tree(hand_graph(), "a") == {
            "a": None,
            "b": "a",
            "c": "a",
            "d": "b",
            "e": "b",
            "f": "c",
        }

    @pytest.mark.parametrize(
        "start, expected",
        [
            ("a", ["a", "b", "c", "d", "e", "f"]),
            ("b", ["b", "a", "d", "e", "c", "f"]),
            ("f", ["f", "c", "e", "a", "b", "d"]),
        ],
        ids=["from_a", "from_b", "from_f"],
    )
    def test_component_order_from_several_starts(self, start, expected):
        assert bfs_component(hand_graph(), start) == expected

    def test_star_graph_is_one_ring(self):
        """Every leaf of a star is level 1, so the order is the hub then the rim."""
        graph = Graph()
        for leaf in "wxyz":
            graph.add_edge("hub", leaf)
        assert bfs(graph, "hub") == ["hub", "w", "x", "y", "z"]
        assert set(bfs_levels(graph, "hub").values()) == {0, 1}
        assert bfs_levels(graph, "w") == {"w": 0, "hub": 1, "x": 2, "y": 2, "z": 2}


# ----------------------------------------------------------------------
# The defining property, against the independent oracle
# ----------------------------------------------------------------------
class TestBreadthFirstProperty:
    """BFS visits nodes in non-decreasing hop distance from the start."""

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_component_order_never_steps_back_to_a_smaller_level(self, name):
        """THE property, on every generated graph and several starts each."""
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            order = bfs_component(graph, start)
            assert_non_decreasing(order, hop_levels(graph, start), label=f"{name} from {start!r}")

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_the_start_is_visited_first(self, name):
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            assert bfs_component(graph, start)[0] == start
            assert bfs(graph, start)[0] == start

    @pytest.mark.parametrize("name", UNDIRECTED_NAMES)
    def test_full_traversal_is_whole_components_each_in_hop_order(self, name):
        graph = GENERATED_GRAPHS[name]()
        assert_full_traversal(graph, bfs(graph))
        for start in sample_starts(graph):
            assert_full_traversal(graph, bfs(graph, start))

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_full_traversal_visits_every_node_exactly_once(self, name):
        """True whatever the direction: bfs() reports the graph, not a component."""
        graph = GENERATED_GRAPHS[name]()
        order = bfs(graph)
        assert sorted(order) == sorted(graph.nodes())
        assert len(order) == graph.node_count

    def test_property_holds_on_the_hand_drawn_graph_too(self):
        graph = hand_graph()
        for start in graph.nodes():
            assert_non_decreasing(bfs_component(graph, start), hop_levels(graph, start))


# ----------------------------------------------------------------------
# bfs_levels
# ----------------------------------------------------------------------
class TestBfsLevels:
    """Hop counts, checked against the brute-force expansion."""

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_levels_match_the_brute_force_expansion(self, name):
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            assert bfs_levels(graph, start) == hop_levels(graph, start), f"{name} from {start!r}"

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_unreachable_nodes_are_absent_rather_than_infinite(self, name):
        """Absent, never mapped to None or to a sentinel number."""
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            levels = bfs_levels(graph, start)
            assert set(levels) == set(bfs_component(graph, start))
            assert all(isinstance(level, int) and level >= 0 for level in levels.values())

    def test_start_is_always_level_zero(self):
        graph = GENERATED_GRAPHS["sparse"]()
        for start in graph.nodes():
            assert bfs_levels(graph, start)[start] == 0

    def test_on_a_path_the_level_is_the_distance_along_the_chain(self):
        graph = gg.path_graph(20)
        assert bfs_levels(graph, 0) == {node: node for node in range(20)}
        assert bfs_levels(graph, 7) == {node: abs(node - 7) for node in range(20)}

    def test_on_a_ring_the_level_is_the_shorter_way_round(self):
        graph = gg.cycle_graph(12)
        levels = bfs_levels(graph, 0)
        assert levels == {node: min(node, 12 - node) for node in range(12)}
        assert max(levels.values()) == 6

    def test_on_a_complete_graph_everything_is_one_hop(self):
        graph = gg.complete_graph(10)
        levels = bfs_levels(graph, 0)
        assert levels[0] == 0
        assert sorted(levels.values()) == [0] + [1] * 9


# ----------------------------------------------------------------------
# bfs_tree
# ----------------------------------------------------------------------
def assert_valid_bfs_tree(graph: Graph, start: Any, label: str = "") -> None:
    """Every claim the predecessor map makes, checked against the oracle.

    Three separate things: each recorded edge is a real edge of the graph and
    points the right way, following the predecessors from anywhere lands on
    the start, and the number of steps that takes is that node's hop level.
    """
    tree = bfs_tree(graph, start)
    levels = hop_levels(graph, start)

    assert set(tree) == set(levels), f"{label}: tree covers the wrong set of nodes"
    assert tree[start] is None, f"{label}: the start must have no predecessor"

    positions = {node: index for index, node in enumerate(tree)}

    for node, parent in tree.items():
        if parent is None:
            assert node == start, f"{label}: {node!r} has no predecessor but is not the start"
            continue

        assert graph.has_edge(parent, node), (
            f"{label}: the tree claims {parent!r} discovered {node!r}, but that "
            f"edge does not exist in the graph"
        )
        assert positions[parent] < positions[node], (
            f"{label}: {node!r} was recorded before its own predecessor {parent!r}"
        )

        hops, cursor, walked = 0, node, set()
        while tree[cursor] is not None:
            assert cursor not in walked, f"{label}: predecessors cycle at {cursor!r}"
            walked.add(cursor)
            cursor = tree[cursor]
            hops += 1

        assert cursor == start, f"{label}: predecessors from {node!r} end at {cursor!r}"
        assert hops == levels[node], (
            f"{label}: the tree path to {node!r} is {hops} edges long but its hop "
            f"distance is {levels[node]}"
        )


class TestBfsTree:
    """The predecessor map is the breadth-first tree, and it is a real tree."""

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_tree_is_valid_on_every_generated_graph(self, name):
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            assert_valid_bfs_tree(graph, start, label=f"{name} from {start!r}")

    def test_tree_is_valid_from_every_node_of_the_hand_drawn_graph(self):
        graph = hand_graph()
        for start in graph.nodes():
            assert_valid_bfs_tree(graph, start, label=f"hand graph from {start!r}")

    def test_unreachable_nodes_are_absent_rather_than_mapped_to_none(self):
        """Absent and None must not look alike: only the start maps to None."""
        graph = four_component_graph()
        tree = bfs_tree(graph, 1)
        assert set(tree) == {1, 2, 3}
        assert [node for node, parent in tree.items() if parent is None] == [1]

    def test_tree_of_an_isolated_node_is_just_the_root(self):
        assert bfs_tree(four_component_graph(), 20) == {20: None}

    def test_walking_the_tree_back_rebuilds_a_shortest_path(self):
        """The whole point of the map: read it backwards and get a route."""
        graph = gg.path_graph(15)
        tree = bfs_tree(graph, 0)
        path, node = [], 14
        while node is not None:
            path.append(node)
            node = tree[node]
        assert path[::-1] == list(range(15))


# ----------------------------------------------------------------------
# Disconnected graphs
# ----------------------------------------------------------------------
class TestDisconnectedGraphs:
    """Four components: bfs() crosses them all, bfs_component() does not."""

    def test_full_traversal_reaches_every_node(self):
        graph = four_component_graph()
        assert bfs(graph) == [1, 2, 3, 10, 11, 12, 20, 30, 31]

    @pytest.mark.parametrize("start", sorted(COMPONENTS))
    def test_full_traversal_reaches_every_node_from_any_start(self, start):
        graph = four_component_graph()
        order = bfs(graph, start)
        assert order[0] == start
        assert sorted(order) == sorted(graph.nodes())

    def test_the_start_component_comes_out_first_and_whole(self):
        """Starting at the loner still reports the other three components after it."""
        graph = four_component_graph()
        order = bfs(graph, 20)
        assert order == [20, 1, 2, 3, 10, 11, 12, 30, 31]

    @pytest.mark.parametrize("start", sorted(COMPONENTS))
    def test_component_traversal_stays_inside_its_own_component(self, start):
        graph = four_component_graph()
        assert set(bfs_component(graph, start)) == COMPONENTS[start]

    @pytest.mark.parametrize("start", sorted(COMPONENTS))
    def test_levels_and_tree_stay_inside_the_component_too(self, start):
        graph = four_component_graph()
        assert set(bfs_levels(graph, start)) == COMPONENTS[start]
        assert set(bfs_tree(graph, start)) == COMPONENTS[start]

    def test_components_partition_the_graph(self):
        """Every node is in exactly one component, and they cover the graph."""
        graph = four_component_graph()
        found: List[Set[Any]] = []
        for node in graph.nodes():
            reached = set(bfs_component(graph, node))
            if reached not in found:
                found.append(reached)
        assert len(found) == 4
        assert set().union(*found) == set(graph.nodes())
        assert sum(len(component) for component in found) == graph.node_count

    def test_a_disconnected_generated_graph_needs_more_than_one_sweep(self):
        """Guards the fixtures: at least one generated graph really is split."""
        graph = GENERATED_GRAPHS["random_thin"]()
        assert len(bfs_component(graph, graph.nodes()[0])) < graph.node_count
        assert len(bfs(graph)) == graph.node_count


# ----------------------------------------------------------------------
# Direction
# ----------------------------------------------------------------------
class TestDirectedGraphs:
    """An edge is a one-way street, and the reverse is not a route."""

    def test_the_reverse_direction_is_not_reachable(self):
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        assert bfs_component(graph, "a") == ["a", "b"]
        assert bfs_component(graph, "b") == ["b"]
        assert bfs_levels(graph, "b") == {"b": 0}
        assert bfs_tree(graph, "b") == {"b": None}

    def test_full_traversal_still_reports_unreachable_nodes(self):
        """bfs() falls through to the next unvisited node in insertion order."""
        graph = Graph(directed=True)
        for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
            graph.add_edge(u, v)
        graph.add_edge("x", "a")
        assert bfs(graph, "b") == ["b", "c", "a", "x"]
        assert bfs_component(graph, "b") == ["b", "c", "a"]

    def test_a_directed_chain_is_reachable_forwards_only(self):
        graph = gg.path_graph(10, directed=True)
        assert bfs_component(graph, 0) == list(range(10))
        assert bfs_component(graph, 7) == [7, 8, 9]
        assert bfs_levels(graph, 7) == {7: 0, 8: 1, 9: 2}
        assert bfs_component(graph, 9) == [9]

    def test_a_directed_ring_reaches_everything_the_long_way_round(self):
        graph = gg.cycle_graph(8, directed=True)
        assert bfs_component(graph, 0) == list(range(8))
        assert bfs_levels(graph, 0) == {node: node for node in range(8)}
        assert bfs_levels(graph, 5) == {5: 0, 6: 1, 7: 2, 0: 3, 1: 4, 2: 5, 3: 6, 4: 7}

    def test_a_sink_reaches_only_itself_while_many_nodes_reach_it(self):
        graph = Graph(directed=True)
        for source in ("p", "q", "r"):
            graph.add_edge(source, "sink")
        assert bfs_component(graph, "sink") == ["sink"]
        assert all(bfs_levels(graph, source)["sink"] == 1 for source in ("p", "q", "r"))

    @pytest.mark.parametrize("name", [name for name in GRAPH_NAMES if "directed" in name])
    def test_directed_levels_match_the_oracle(self, name):
        """The oracle follows direction too, by not restoring the reverse edge."""
        graph = GENERATED_GRAPHS[name]()
        assert graph.directed
        for start in sample_starts(graph):
            assert bfs_levels(graph, start) == hop_levels(graph, start)


# ----------------------------------------------------------------------
# Weights are irrelevant to BFS
# ----------------------------------------------------------------------
class TestWeightsAreIgnored:
    """BFS counts hops. That is the whole difference between it and Dijkstra."""

    @staticmethod
    def _detour_graph(weighted: bool) -> Graph:
        """s - t directly at cost 100, or s - u - v - t at cost 3.

        Fewest hops and cheapest cost disagree on purpose, so a BFS that
        peeked at the weights would answer differently.
        """
        graph = Graph(weighted=weighted)
        edges = [("s", "t", 100.0), ("s", "u", 1.0), ("u", "v", 1.0), ("v", "t", 1.0)]
        for u, v, weight in edges:
            graph.add_edge(u, v, weight if weighted else 1.0)
        return graph

    def test_the_expensive_single_hop_is_still_level_one(self):
        graph = self._detour_graph(weighted=True)
        assert bfs_levels(graph, "s")["t"] == 1
        assert bfs_tree(graph, "s")["t"] == "s"

    def test_the_cheap_route_really_is_cheaper_than_the_route_bfs_takes(self):
        """Confirms the test graph has teeth: 3.0 by cost, 1 hop by edges."""
        graph = self._detour_graph(weighted=True)
        cheap = sum(graph.get_edge_weight(u, v) for u, v in [("s", "u"), ("u", "v"), ("v", "t")])
        assert cheap == pytest.approx(3.0)
        assert graph.get_edge_weight("s", "t") == pytest.approx(100.0)
        assert cheap < graph.get_edge_weight("s", "t")

    def test_results_are_identical_to_an_unweighted_twin(self):
        """Same shape, no weights: every BFS answer must be the same object."""
        weighted = self._detour_graph(weighted=True)
        plain = self._detour_graph(weighted=False)
        assert weighted.weighted and not plain.weighted
        for start in plain.nodes():
            assert bfs(weighted, start) == bfs(plain, start)
            assert bfs_component(weighted, start) == bfs_component(plain, start)
            assert bfs_levels(weighted, start) == bfs_levels(plain, start)
            assert bfs_tree(weighted, start) == bfs_tree(plain, start)

    @pytest.mark.parametrize("name", ["weighted", "weighted_directed"])
    def test_generated_weighted_graphs_are_traversed_by_hops(self, name):
        graph = GENERATED_GRAPHS[name]()
        assert graph.weighted
        for start in sample_starts(graph):
            assert bfs_levels(graph, start) == hop_levels(graph, start)
            assert_non_decreasing(bfs_component(graph, start), hop_levels(graph, start))

    def test_a_zero_weight_edge_is_still_one_hop(self):
        """A free edge does not fold two nodes into one level."""
        graph = Graph(weighted=True)
        graph.add_edge("a", "b", 0.0)
        graph.add_edge("b", "c", 0.0)
        assert bfs_levels(graph, "a") == {"a": 0, "b": 1, "c": 2}


# ----------------------------------------------------------------------
# Small and degenerate graphs
# ----------------------------------------------------------------------
class TestEdgeCases:
    """Empty, single, self-loop, two nodes: where traversals fall over."""

    @pytest.mark.parametrize(
        "graph",
        [Graph(), Graph(directed=True), Graph(weighted=True), Graph(directed=True, weighted=True)],
        ids=["plain", "directed", "weighted", "directed_weighted"],
    )
    def test_empty_graph_traverses_to_an_empty_list(self, graph):
        assert bfs(graph) == []
        assert graph.node_count == 0

    @pytest.mark.parametrize("function", [bfs_component, bfs_levels, bfs_tree])
    def test_empty_graph_has_no_valid_start(self, function):
        with pytest.raises(KeyError):
            function(Graph(), "a")

    def test_single_node(self):
        graph = Graph()
        graph.add_node("only")
        assert bfs(graph) == ["only"]
        assert bfs_component(graph, "only") == ["only"]
        assert bfs_levels(graph, "only") == {"only": 0}
        assert bfs_tree(graph, "only") == {"only": None}

    def test_two_nodes_joined(self):
        graph = Graph()
        graph.add_edge("p", "q")
        assert bfs(graph, "p") == ["p", "q"]
        assert bfs(graph, "q") == ["q", "p"]
        assert bfs_levels(graph, "q") == {"q": 0, "p": 1}
        assert bfs_tree(graph, "q") == {"q": None, "p": "q"}

    def test_two_nodes_not_joined(self):
        graph = Graph()
        graph.add_node("p")
        graph.add_node("q")
        assert bfs(graph) == ["p", "q"]
        assert bfs_component(graph, "p") == ["p"]

    def test_a_self_loop_terminates_and_is_not_its_own_predecessor(self):
        """If the loop were followed the traversal would never return."""
        graph = Graph()
        graph.add_edge(1, 1)
        graph.add_edge(1, 2)
        assert bfs(graph) == [1, 2]
        assert bfs_levels(graph, 1) == {1: 0, 2: 1}
        assert bfs_tree(graph, 1) == {1: None, 2: 1}

    def test_a_graph_that_is_nothing_but_self_loops(self):
        graph = Graph()
        for node in range(5):
            graph.add_edge(node, node)
        assert bfs(graph) == list(range(5))
        assert all(bfs_component(graph, node) == [node] for node in range(5))

    def test_self_loops_on_top_of_a_ring_still_terminate(self):
        """Cycles plus loops: the marking has to hold up against both at once."""
        graph = gg.cycle_graph(10)
        for node in range(10):
            graph.add_edge(node, node)
        assert bfs_levels(graph, 0) == {node: min(node, 10 - node) for node in range(10)}
        assert len(bfs(graph)) == 10

    def test_a_ring_terminates_at_all(self):
        """The smallest structure that loops forever without visited marking."""
        graph = gg.cycle_graph(3)
        assert bfs(graph) == [0, 1, 2]

    def test_parallel_add_edge_calls_do_not_duplicate_a_visit(self):
        """Re-adding the same edge must not queue the far node twice."""
        graph = Graph()
        for _ in range(3):
            graph.add_edge("a", "b")
        assert bfs(graph) == ["a", "b"]
        assert graph.edge_count == 1


# ----------------------------------------------------------------------
# Errors and start-node handling
# ----------------------------------------------------------------------
class TestStartNode:
    """A missing start is a KeyError, everywhere, with the node named."""

    @pytest.mark.parametrize("function", [bfs, bfs_component, bfs_levels, bfs_tree])
    def test_missing_start_raises_key_error(self, function):
        graph = hand_graph()
        with pytest.raises(KeyError):
            function(graph, "nope")

    @pytest.mark.parametrize("function", [bfs, bfs_component, bfs_levels, bfs_tree])
    def test_the_key_error_names_the_offending_node(self, function):
        graph = hand_graph()
        with pytest.raises(KeyError, match="start node 'nope' is not in the graph"):
            function(graph, "nope")

    def test_a_node_removed_from_the_graph_is_no_longer_a_valid_start(self):
        graph = hand_graph()
        graph.remove_node("f")
        with pytest.raises(KeyError):
            bfs_component(graph, "f")
        assert bfs(graph) == ["a", "b", "c", "d", "e"]

    def test_bfs_checks_the_start_before_traversing_anything(self):
        """A bad start must not return a partial traversal."""
        graph = four_component_graph()
        with pytest.raises(KeyError):
            bfs(graph, "missing")

    def test_none_as_a_node_label_collides_with_the_default_start(self):
        """Documents actual behaviour: bfs() cannot be started at a node named None.

        KNOWN LIMITATION, not fixed here. ``bfs(graph, start=None)`` means "no
        start given" because None is the sentinel for the default, so a graph
        holding a node literally labelled None cannot be asked to begin there.
        The traversal is still complete and correct, it just begins at the
        first node in insertion order instead. ``bfs_component`` takes a
        required start and does honour None.
        """
        graph = Graph()
        graph.add_edge(5, 6)
        graph.add_node(None)

        assert bfs(graph, start=None) == [5, 6, None]
        assert bfs(graph, start=None)[0] == 5
        assert bfs_component(graph, None) == [None]
        assert bfs_levels(graph, None) == {None: 0}


# ----------------------------------------------------------------------
# The four functions agree with each other
# ----------------------------------------------------------------------
class TestConsistencyBetweenFunctions:
    """Each function is a different view of the same sweep."""

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_component_order_is_the_prefix_of_the_full_traversal(self, name):
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            component = bfs_component(graph, start)
            assert bfs(graph, start)[: len(component)] == component

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_levels_tree_and_component_cover_the_same_nodes(self, name):
        graph = GENERATED_GRAPHS[name]()
        for start in sample_starts(graph):
            component = bfs_component(graph, start)
            assert list(bfs_tree(graph, start)) == component
            assert list(bfs_levels(graph, start)) == component

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_full_traversal_equals_component_traversal_when_connected(self, name):
        graph = GENERATED_GRAPHS[name]()
        start = graph.nodes()[0]
        if len(bfs_component(graph, start)) == graph.node_count:
            assert bfs(graph, start) == bfs_component(graph, start)

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_repeated_calls_return_the_same_answer(self, name):
        """Insertion order, never a set iteration order, drives the traversal."""
        graph = GENERATED_GRAPHS[name]()
        start = graph.nodes()[0]
        assert bfs(graph, start) == bfs(graph, start)
        assert bfs_tree(graph, start) == bfs_tree(graph, start)

    @pytest.mark.parametrize("name", GRAPH_NAMES)
    def test_traversal_does_not_modify_the_graph(self, name):
        graph = GENERATED_GRAPHS[name]()
        before = (graph.node_count, graph.edge_count, graph.nodes(), graph.edges())
        start = graph.nodes()[0]
        bfs(graph, start)
        bfs_component(graph, start)
        bfs_levels(graph, start)
        bfs_tree(graph, start)
        assert (graph.node_count, graph.edge_count, graph.nodes(), graph.edges()) == before
