"""Tests for depth-first search in both of its forms.

Three claims carry the module, and each is checked against something that
is not the implementation being checked.

THE HEADLINE is that ``dfs_iterative`` and ``dfs_recursive`` return the
identical list. They are two different machines - an explicit LIFO list
against the interpreter's call stack - so neither can hide a bug in the
other, and the equality is asserted on seeded random graphs across sizes,
densities, directedness, disconnection and self-loops. The classic failure
this catches is the reversed-neighbour push: a stack pops the last thing
pushed, so an implementation that pushes neighbours in their natural order
walks each fork backwards. It still reaches every node, so a "did we visit
everything" test passes happily; only the order gives it away.

THE DEFINING PROPERTY is that depth-first search never teleports. Every
node in the order except the one that opens a component is adjacent to
some node earlier in the order. That is checked here against an
independent in-edge map and an order-free reachability closure built from
:class:`~src.graphs.graph.Graph` alone, rather than by re-deriving the
traversal and comparing. The same oracle counts components by union-find,
so "one root per component" is proven from the edge list rather than from
DFS agreeing with itself. To pin down that the order really is depth-first
and not merely correct, DFS and BFS are run on a graph where they must
disagree: they return the same set of nodes and a different sequence.

THE RECURSION BUDGET is the part of :mod:`src.graphs.dfs` most likely to
regress quietly. ``dfs_recursive`` raises the interpreter's recursion limit
for the duration of a call and restores it in a ``finally``. A 3,500-node
path graph recurses far past CPython's default of 1000, so it proves the
raise actually happens, and the previous limit is asserted to be back
afterwards - including when the traversal raises partway through and when
the graph is refused for exceeding ``MAX_RECURSION_LIMIT``. A leaked limit
would not fail any traversal test; it would quietly reconfigure whatever
ran next.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Callable, Dict, Iterable, List, Set

import pytest

import src.graphs.dfs as dfs_module
from src.graphs.bfs import bfs
from src.graphs.dfs import (
    RECURSION_HEADROOM,
    dfs,
    dfs_component,
    dfs_iterative,
    dfs_recursive,
)
from src.graphs.graph import Graph
from src.utils.graph_generator import (
    complete_graph,
    cycle_graph,
    dense_graph,
    path_graph,
    random_graph,
    sparse_graph,
)

#: The two implementations, by name. Every test that should hold of DFS as
#: such is parametrised on this, so neither form can be exercised less than
#: the other.
DFS_FORMS: Dict[str, Callable[..., List[Any]]] = {
    "iterative": dfs_iterative,
    "recursive": dfs_recursive,
}
FORM_NAMES = sorted(DFS_FORMS)

#: Depth of the path graph used to prove the recursion limit is raised.
#: Comfortably past CPython's default limit of 1000.
DEEP_PATH_NODES = 3_500


@pytest.fixture(params=FORM_NAMES, ids=FORM_NAMES)
def dfs_form(request) -> Callable[..., List[Any]]:
    """Parametrised over the iterative and recursive traversals."""
    return DFS_FORMS[request.param]


# ----------------------------------------------------------------------
# Graphs under test
# ----------------------------------------------------------------------
def make_random_graph(
    seed: int,
    n: int,
    edge_factor: float,
    *,
    directed: bool = False,
    self_loops: bool = False,
) -> Graph:
    """Build a reproducible random graph by drawing endpoints at random.

    Deliberately not a spanning-tree generator: drawing both endpoints
    independently leaves isolated nodes and several components behind,
    which is exactly the shape that separates a full traversal from a
    single-component one.

    Args:
        seed: Seed for this graph's private random source.
        n: Number of nodes, labelled 0 to n-1.
        edge_factor: Edge draws attempted, as a multiple of ``n``.
        directed: True to build a directed graph.
        self_loops: True to keep a draw whose endpoints coincide.

    Returns:
        A graph with ``n`` nodes and at most ``int(n * edge_factor)`` edges.
    """
    rng = random.Random(seed)
    graph = Graph(directed=directed)
    for node in range(n):
        graph.add_node(node)
    for _ in range(int(n * edge_factor)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v and not self_loops:
            continue
        graph.add_edge(u, v)
    return graph


def make_three_component_graph(directed: bool = False) -> Graph:
    """Build a graph of exactly three components: a diamond, an edge, a node.

    Node 7 is added last and has no edges at all, so a traversal that only
    follows edges from the first root reports four nodes too few.
    """
    graph = Graph(directed=directed)
    for u, v in [(1, 2), (1, 3), (2, 4), (3, 4), (5, 6)]:
        graph.add_edge(u, v)
    graph.add_node(7)
    return graph


def build_graph_zoo() -> Dict[str, Graph]:
    """Every shape the equivalence claim has to survive, built once, seeded.

    Returns:
        A mapping of case name to graph, covering empty and tiny graphs,
        self-loops, cycles, paths, several densities, directed and
        undirected, and disconnected shapes with many components.
    """
    two_node = Graph()
    two_node.add_edge("a", "b")

    self_loop = Graph()
    self_loop.add_edge(1, 1)
    self_loop.add_edge(1, 2)
    self_loop.add_edge(2, 2)

    lonely = Graph()
    for node in "abcde":
        lonely.add_node(node)

    directed_diamond = Graph(directed=True)
    for u, v in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "a")]:
        directed_diamond.add_edge(u, v)

    zoo: Dict[str, Graph] = {
        "empty": Graph(),
        "single_node": make_random_graph(0, 1, 0.0),
        "two_nodes": two_node,
        "self_loops": self_loop,
        "five_isolated_nodes": lonely,
        "three_components": make_three_component_graph(),
        "three_components_directed": make_three_component_graph(directed=True),
        "directed_diamond": directed_diamond,
        "path_40": path_graph(40),
        "path_40_directed": path_graph(40, directed=True),
        "cycle_25": cycle_graph(25),
        "cycle_25_directed": cycle_graph(25, directed=True),
        "complete_12": complete_graph(12),
        "sparse_60": sparse_graph(60, avg_degree=3, seed=1),
        "sparse_60_directed": sparse_graph(60, avg_degree=3, directed=True, seed=2),
        "dense_40": dense_graph(40, density=0.7, seed=3),
        "dense_40_directed": dense_graph(40, density=0.7, directed=True, seed=4),
        "random_d02_50": random_graph(50, density=0.2, seed=5),
        "random_d05_50_directed": random_graph(50, density=0.5, directed=True, seed=6),
        "weighted_is_ignored": random_graph(30, density=0.3, weighted=True, seed=7),
    }

    # Seeded random draws: several sizes, several densities, both
    # directions, with and without self-loops.
    for seed, (n, factor, directed, loops) in enumerate(
        [
            (2, 1.0, False, True),
            (6, 0.5, False, False),
            (12, 2.0, True, False),
            (20, 0.3, False, True),
            (30, 1.5, True, True),
            (45, 0.8, False, False),
            (64, 2.5, True, False),
            (80, 0.2, False, False),
        ],
        start=100,
    ):
        name = (
            f"random_n{n}_f{factor}"
            f"{'_directed' if directed else ''}{'_loops' if loops else ''}"
        )
        zoo[name] = make_random_graph(
            seed, n, factor, directed=directed, self_loops=loops
        )

    return zoo


GRAPH_ZOO: Dict[str, Graph] = build_graph_zoo()
ZOO_NAMES = sorted(GRAPH_ZOO)
#: The zoo without the empty graph, for tests that need a start node.
NONEMPTY_ZOO_NAMES = [name for name in ZOO_NAMES if len(GRAPH_ZOO[name])]
#: The undirected members, for the claims that only make sense there.
UNDIRECTED_ZOO_NAMES = [name for name in ZOO_NAMES if not GRAPH_ZOO[name].directed]


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def incoming_map(graph: Graph) -> Dict[Any, Set[Any]]:
    """Map each node to the nodes holding an edge into it.

    The inverse of the adjacency list, built here from the graph rather
    than taken from it, so "adjacent to something earlier" can be checked
    without asking a traversal.
    """
    incoming: Dict[Any, Set[Any]] = {node: set() for node in graph.nodes()}
    for source in graph.nodes():
        for target in graph.neighbors(source):
            incoming[target].add(source)
    return incoming


def reachable_from(graph: Graph, sources: Iterable[Any]) -> Set[Any]:
    """Return every node reachable from any of ``sources``, as a set.

    An order-free closure: it answers "can you get there at all", never
    "in what order", so it can judge a traversal's contents without
    encoding any opinion about its sequence.
    """
    seen: Set[Any] = set()
    pending = list(sources)
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph.neighbors(node))
    return seen


def component_count(graph: Graph) -> int:
    """Count the connected components of an undirected graph by union-find.

    Union-find works off the edge list alone and never walks the graph, so
    it agrees with a traversal only if the traversal is right.
    """
    parent = {node: node for node in graph.nodes()}

    def find(node: Any) -> Any:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in graph.edges():
        root_u, root_v = find(edge[0]), find(edge[1])
        if root_u != root_v:
            parent[root_u] = root_v

    return len({find(node) for node in graph.nodes()})


def component_roots(graph: Graph, order: List[Any]) -> List[Any]:
    """Return the nodes in ``order`` that no earlier node points at.

    Those are the traversal's launch points: a node reached from inside
    the traversal necessarily has an in-neighbour already visited.
    """
    incoming = incoming_map(graph)
    earlier: Set[Any] = set()
    roots: List[Any] = []
    for node in order:
        if not incoming[node] & earlier:
            roots.append(node)
        earlier.add(node)
    return roots


def assert_never_teleports(graph: Graph, order: List[Any]) -> None:
    """Fail unless every node is adjacent to an earlier one or opens a component.

    The defining property of a depth-first (or any) traversal built from
    edges: you can only arrive somewhere from somewhere you have already
    been. A node with no earlier in-neighbour is allowed only if nothing
    visited so far could reach it at all, which is what starting a fresh
    component means.
    """
    assert len(order) == len(set(order)), "a node was visited twice"
    assert set(order) == set(graph.nodes()), "the traversal missed or invented a node"

    incoming = incoming_map(graph)
    for position, node in enumerate(order):
        if position == 0:
            continue
        earlier = set(order[:position])
        if incoming[node] & earlier:
            continue
        assert node not in reachable_from(graph, earlier), (
            f"{node!r} at position {position} has no neighbour earlier in the "
            f"order, yet it was reachable from one - the traversal teleported"
        )


# ----------------------------------------------------------------------
# A graph small enough to check by hand
# ----------------------------------------------------------------------
class TestHandDrawnGraph:
    """One six-node graph whose depth-first order can be traced by eye.

        A - B        A: B C        Dive A -> B -> D, and only at D is
        |   |        B: A D        there a fork; C is reached from D,
        C - D - E    C: A D F      not from A, because DFS commits to
        |            D: B C E      the first branch. F hangs off C, and
        F            E: D          E is picked up on the way back out.
                     F: C
    """

    @staticmethod
    def build() -> Graph:
        graph = Graph()
        for u, v in [
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
            ("D", "E"),
            ("C", "F"),
        ]:
            graph.add_edge(u, v)
        return graph

    def test_visit_order_is_the_traced_one(self, dfs_form):
        assert dfs_form(self.build()) == ["A", "B", "D", "C", "F", "E"]

    def test_starting_elsewhere_still_dives(self, dfs_form):
        """From F the only way in is C, and from C the first fork is A."""
        assert dfs_form(self.build(), "F") == ["F", "C", "A", "B", "D", "E"]

    def test_breadth_first_takes_the_same_graph_in_rings(self):
        """The contrast: BFS reaches D via two hops, DFS via B."""
        graph = self.build()
        assert bfs(graph) == ["A", "B", "C", "D", "F", "E"]
        assert dfs(graph) == ["A", "B", "D", "C", "F", "E"]

    @pytest.mark.parametrize(
        ("edges", "expected"),
        [
            ([(1, 2), (1, 3), (2, 4), (3, 4)], [1, 2, 4, 3]),
            ([(1, 2), (2, 3), (3, 1)], [1, 2, 3]),
            ([(1, 2), (1, 3), (1, 4)], [1, 2, 3, 4]),
            ([(1, 2), (2, 3), (2, 4), (4, 5)], [1, 2, 3, 4, 5]),
        ],
        ids=["diamond", "triangle", "star", "tree"],
    )
    def test_small_shapes_by_inspection(self, dfs_form, edges, expected):
        graph = Graph()
        for u, v in edges:
            graph.add_edge(u, v)
        assert dfs_form(graph) == expected


# ----------------------------------------------------------------------
# THE HEADLINE: the two forms are the same traversal
# ----------------------------------------------------------------------
class TestIterativeEqualsRecursive:
    """The two implementations return the identical list, never merely the
    same set. This is the test that catches a stack pushed in the wrong
    order, which visits everything and still walks every fork backwards.
    """

    @pytest.mark.parametrize("name", ZOO_NAMES)
    def test_both_forms_agree_on_every_graph_in_the_zoo(self, name):
        graph = GRAPH_ZOO[name]
        assert dfs_iterative(graph) == dfs_recursive(graph)

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_both_forms_agree_from_every_start_node(self, name):
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            assert dfs_iterative(graph, start) == dfs_recursive(graph, start), (
                f"{name} disagreed when started at {start!r}"
            )

    @pytest.mark.parametrize("seed", range(25))
    def test_both_forms_agree_on_freshly_seeded_random_graphs(self, seed):
        """Sizes, densities, directedness and self-loops all drawn per seed."""
        rng = random.Random(seed)
        n = rng.randint(1, 60)
        graph = make_random_graph(
            seed=seed,
            n=n,
            edge_factor=rng.choice([0.2, 0.5, 1.0, 2.0, 4.0]),
            directed=rng.random() < 0.5,
            self_loops=rng.random() < 0.5,
        )
        assert dfs_iterative(graph) == dfs_recursive(graph)

    def test_the_order_is_not_merely_the_same_set(self):
        """Guard on the guard: this graph's DFS order is not insertion order,
        so an equality between the two forms is a real constraint here.
        """
        graph = GRAPH_ZOO["three_components"]
        order = dfs_iterative(graph)
        assert order != graph.nodes()
        assert set(order) == set(graph.nodes())

    def test_a_reversed_push_would_be_caught(self):
        """The bug being guarded against, spelled out.

        Pushing neighbours in their natural order rather than reversed
        visits every node but mirrors each fork. On the diamond that turns
        [1, 2, 4, 3] into [1, 3, 4, 2], which the equality test above
        rejects and a set comparison would not.
        """
        graph = Graph()
        for u, v in [(1, 2), (1, 3), (2, 4), (3, 4)]:
            graph.add_edge(u, v)

        mirrored: List[Any] = []
        visited: Set[Any] = set()
        stack = [1]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            mirrored.append(node)
            # No reversed() here: this is the classic mistake.
            for neighbour in graph.neighbors(node):
                if neighbour not in visited:
                    stack.append(neighbour)

        assert mirrored == [1, 3, 4, 2]
        assert set(mirrored) == set(dfs_iterative(graph))
        assert mirrored != dfs_iterative(graph) == dfs_recursive(graph)


# ----------------------------------------------------------------------
# THE DEFINING PROPERTY, against an independent oracle
# ----------------------------------------------------------------------
class TestNeverTeleports:
    """Every node is entered from a neighbour already visited, or it opens a
    new component. Checked against an in-edge map and a reachability
    closure, not against the other DFS.
    """

    @pytest.mark.parametrize("name", ZOO_NAMES)
    def test_order_is_edge_connected_on_every_graph(self, name, dfs_form):
        graph = GRAPH_ZOO[name]
        assert_never_teleports(graph, dfs_form(graph))

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_order_is_edge_connected_from_every_start(self, name, dfs_form):
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            assert_never_teleports(graph, dfs_form(graph, start))

    @pytest.mark.parametrize("name", UNDIRECTED_ZOO_NAMES)
    def test_one_root_per_connected_component(self, name, dfs_form):
        """Union-find over the edge list says how many launch points there
        must be; the traversal is asked to have exactly that many.

        Undirected only: with direction, a node can be reachable from an
        earlier one along a path while having no earlier in-neighbour, so
        "no in-edge from earlier" would stop meaning "new component".
        """
        graph = GRAPH_ZOO[name]
        roots = component_roots(graph, dfs_form(graph))
        assert len(roots) == component_count(graph)

    def test_breadth_and_depth_first_differ_in_order_but_not_in_contents(self):
        """A graph where they must disagree: BFS reaches 4 from 3, having
        already ringed 2 and 3; DFS reaches 4 from 2 before ever seeing 3.
        """
        graph = GRAPH_ZOO["three_components"]
        breadth, depth = bfs(graph), dfs(graph)
        assert set(breadth) == set(depth)
        assert breadth != depth
        assert breadth == [1, 2, 3, 4, 5, 6, 7]
        assert depth == [1, 2, 4, 3, 5, 6, 7]

    @pytest.mark.parametrize("name", ZOO_NAMES)
    def test_breadth_and_depth_first_always_report_the_same_nodes(self, name):
        graph = GRAPH_ZOO[name]
        assert set(bfs(graph)) == set(dfs(graph)) == set(graph.nodes())

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_the_first_node_is_the_requested_start(self, name, dfs_form):
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            assert dfs_form(graph, start)[0] == start

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_default_start_is_the_first_node_in_insertion_order(self, name, dfs_form):
        graph = GRAPH_ZOO[name]
        assert dfs_form(graph)[0] == graph.nodes()[0]


# ----------------------------------------------------------------------
# Disconnected graphs
# ----------------------------------------------------------------------
class TestDisconnectedGraphs:
    """A full traversal jumps to the next unvisited node, so nothing is lost."""

    def test_three_components_are_all_reported(self, dfs_form):
        graph = make_three_component_graph()
        assert dfs_form(graph) == [1, 2, 4, 3, 5, 6, 7]

    def test_three_components_when_directed(self, dfs_form):
        graph = make_three_component_graph(directed=True)
        assert dfs_form(graph) == [1, 2, 4, 3, 5, 6, 7]

    def test_five_components_of_mixed_shapes(self, dfs_form):
        """A triangle, a path, a self-loop, a lone node and a pair."""
        graph = Graph()
        for u, v in [
            ("t1", "t2"),
            ("t2", "t3"),
            ("t3", "t1"),
            ("p1", "p2"),
            ("p2", "p3"),
            ("loop", "loop"),
            ("x1", "x2"),
        ]:
            graph.add_edge(u, v)
        graph.add_node("alone")

        order = dfs_form(graph)
        assert order == [
            "t1", "t2", "t3", "p1", "p2", "p3", "loop", "x1", "x2", "alone",
        ]
        assert component_count(graph) == 5
        assert len(component_roots(graph, order)) == 5

    def test_every_isolated_node_is_its_own_component(self, dfs_form):
        graph = GRAPH_ZOO["five_isolated_nodes"]
        order = dfs_form(graph)
        assert order == list("abcde")
        assert len(component_roots(graph, order)) == 5

    def test_start_node_only_chooses_which_component_comes_first(self, dfs_form):
        graph = make_three_component_graph()
        assert dfs_form(graph, start=5) == [5, 6, 1, 2, 4, 3, 7]
        assert dfs_form(graph, start=7) == [7, 1, 2, 4, 3, 5, 6]
        assert set(dfs_form(graph, start=5)) == set(dfs_form(graph))

    @pytest.mark.parametrize("name", ZOO_NAMES)
    def test_every_node_appears_exactly_once(self, name, dfs_form):
        graph = GRAPH_ZOO[name]
        order = dfs_form(graph)
        assert sorted(map(repr, order)) == sorted(map(repr, graph.nodes()))
        assert len(order) == graph.node_count


# ----------------------------------------------------------------------
# Recursion depth and the interpreter limit
# ----------------------------------------------------------------------
class RecordingGraph(Graph):
    """A graph that notes the recursion limit in force at each lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.limits: List[int] = []

    def neighbors(self, v: Any):
        self.limits.append(sys.getrecursionlimit())
        return super().neighbors(v)


class ExplodingGraph(Graph):
    """A graph whose neighbour lookup fails, to blow up a traversal midway."""

    def neighbors(self, v: Any):
        raise RuntimeError("neighbour lookup failed")


class TestRecursionDepth:
    """dfs_recursive raises the recursion limit, and always puts it back."""

    def test_deep_path_completes_without_recursion_error(self):
        """A 3,500-node path recurses 3,500 deep, well past the default 1000."""
        graph = path_graph(DEEP_PATH_NODES)
        assert dfs_recursive(graph) == list(range(DEEP_PATH_NODES))

    def test_the_limit_is_back_where_it_started_afterwards(self):
        """The restore is the part most likely to regress: a leaked limit
        breaks nothing here and reconfigures whatever runs next.
        """
        before = sys.getrecursionlimit()
        dfs_recursive(path_graph(DEEP_PATH_NODES))
        assert sys.getrecursionlimit() == before

    def test_the_limit_really_is_raised_during_the_traversal(self):
        """Proof the deep path is not simply passing under the old limit."""
        graph = RecordingGraph()
        for i in range(10):
            graph.add_edge(i, i + 1)

        before = sys.getrecursionlimit()
        dfs_recursive(graph)

        assert graph.limits, "the traversal never asked for a neighbour"
        assert min(graph.limits) >= len(graph) + RECURSION_HEADROOM
        assert sys.getrecursionlimit() == before

    def test_iterative_form_leaves_the_limit_untouched(self):
        """No frames are spent, so nothing needs raising."""
        graph = RecordingGraph()
        for i in range(10):
            graph.add_edge(i, i + 1)
        before = sys.getrecursionlimit()
        dfs_iterative(graph)
        assert graph.limits == [before] * len(graph.limits)
        assert sys.getrecursionlimit() == before

    def test_the_limit_is_restored_when_the_traversal_raises(self):
        graph = ExplodingGraph()
        graph.add_edge(1, 2)
        before = sys.getrecursionlimit()
        with pytest.raises(RuntimeError):
            dfs_recursive(graph)
        assert sys.getrecursionlimit() == before

    def test_a_graph_past_the_cap_is_refused_before_any_work(self, monkeypatch):
        """The cap is lowered rather than a 100,000-node graph built: the
        behaviour under test is the refusal, not the size.
        """
        monkeypatch.setattr(dfs_module, "MAX_RECURSION_LIMIT", 100)
        graph = path_graph(200)
        before = sys.getrecursionlimit()

        with pytest.raises(RecursionError) as caught:
            dfs_recursive(graph)

        assert "dfs_iterative" in str(caught.value)
        assert "100" in str(caught.value)
        assert sys.getrecursionlimit() == before

    def test_the_iterative_form_has_no_such_cap(self, monkeypatch):
        monkeypatch.setattr(dfs_module, "MAX_RECURSION_LIMIT", 100)
        assert dfs_iterative(path_graph(200)) == list(range(200))

    def test_dfs_component_recurses_as_deep_and_restores_too(self):
        before = sys.getrecursionlimit()
        graph = path_graph(DEEP_PATH_NODES)
        assert dfs_component(graph, 0, recursive=True) == list(range(DEEP_PATH_NODES))
        assert sys.getrecursionlimit() == before

    def test_a_deep_cycle_does_not_loop_forever(self, dfs_form):
        """Without marking nodes visited, a ring never terminates."""
        assert dfs_form(cycle_graph(2_000)) == list(range(2_000))


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    """Empty, one node, a self-loop and two nodes."""

    def test_empty_graph_returns_an_empty_list(self, dfs_form):
        assert dfs_form(Graph()) == []
        assert dfs_form(Graph(directed=True)) == []
        assert dfs_form(Graph(weighted=True)) == []

    def test_empty_graph_through_the_plain_name(self):
        assert dfs(Graph()) == []

    def test_single_node_with_no_edges(self, dfs_form):
        graph = Graph()
        graph.add_node("only")
        assert dfs_form(graph) == ["only"]
        assert dfs_form(graph, "only") == ["only"]

    def test_self_loop_is_visited_once(self, dfs_form):
        """The node is marked before its neighbours are walked, so the loop
        back to itself is skipped rather than recursed into.
        """
        graph = Graph()
        graph.add_edge("a", "a")
        assert dfs_form(graph) == ["a"]

    def test_self_loop_alongside_real_edges(self, dfs_form):
        graph = Graph()
        graph.add_edge(1, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 2)
        graph.add_edge(2, 3)
        assert dfs_form(graph) == [1, 2, 3]

    def test_two_node_graph_from_either_end(self, dfs_form):
        graph = Graph()
        graph.add_edge("a", "b")
        assert dfs_form(graph) == ["a", "b"]
        assert dfs_form(graph, "b") == ["b", "a"]

    def test_two_nodes_with_no_edge_between_them(self, dfs_form):
        graph = Graph()
        graph.add_node("a")
        graph.add_node("b")
        assert dfs_form(graph) == ["a", "b"]
        assert dfs_form(graph, "b") == ["b", "a"]

    def test_two_node_directed_edge_is_one_way(self, dfs_form):
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        assert dfs_form(graph) == ["a", "b"]
        assert dfs_form(graph, "b") == ["b", "a"]
        assert dfs_component(graph, "b") == ["b"]

    def test_weights_are_ignored(self, dfs_form):
        """DFS reads structure only, so the weighted graph walks the same."""
        plain, weighted = Graph(), Graph(weighted=True)
        for u, v in [(1, 2), (1, 3), (2, 4)]:
            plain.add_edge(u, v)
            weighted.add_edge(u, v, 9.5)
        assert dfs_form(weighted) == dfs_form(plain)

    @pytest.mark.parametrize(
        "function",
        [dfs, dfs_iterative, dfs_recursive],
        ids=["dfs", "dfs_iterative", "dfs_recursive"],
    )
    def test_missing_start_raises_key_error(self, function):
        graph = Graph()
        graph.add_edge("a", "b")
        with pytest.raises(KeyError, match="not in the graph"):
            function(graph, "zz")

    @pytest.mark.parametrize(
        "function",
        [dfs, dfs_iterative, dfs_recursive],
        ids=["dfs", "dfs_iterative", "dfs_recursive"],
    )
    def test_any_start_on_an_empty_graph_raises(self, function):
        with pytest.raises(KeyError, match="not in the graph"):
            function(Graph(), 0)

    def test_non_integer_node_labels(self, dfs_form):
        """Nodes are any hashable, and insertion order needs no sorting."""
        graph = Graph()
        for u, v in [(("x", 1), ("y", 2)), (("y", 2), None), (None, 3.5)]:
            graph.add_edge(u, v)
        assert dfs_form(graph) == [("x", 1), ("y", 2), None, 3.5]


# ----------------------------------------------------------------------
# dfs_component
# ----------------------------------------------------------------------
class TestDfsComponent:
    """One component only, in both forms."""

    @pytest.fixture(params=[False, True], ids=["iterative", "recursive"])
    def recursive(self, request) -> bool:
        return request.param

    def test_it_stops_at_the_component_boundary(self, recursive):
        graph = make_three_component_graph()
        assert dfs_component(graph, 1, recursive=recursive) == [1, 2, 4, 3]
        assert dfs_component(graph, 5, recursive=recursive) == [5, 6]
        assert dfs_component(graph, 7, recursive=recursive) == [7]

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_both_forms_agree_on_every_component(self, name):
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            assert dfs_component(graph, start) == dfs_component(
                graph, start, recursive=True
            ), f"{name} disagreed at {start!r}"

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_contents_match_the_reachability_oracle(self, name, recursive):
        """What it returns is exactly what the start can get to, no more."""
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            walked = dfs_component(graph, start, recursive=recursive)
            assert set(walked) == reachable_from(graph, [start])
            assert len(walked) == len(set(walked))

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_it_is_the_first_sweep_of_the_full_traversal(self, name, recursive):
        """A full DFS started at the same node opens with these same nodes,
        in this same order, and then carries on into the other components.
        """
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            walked = dfs_component(graph, start, recursive=recursive)
            assert dfs(graph, start)[: len(walked)] == walked

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_it_never_teleports_either(self, name, recursive):
        graph = GRAPH_ZOO[name]
        start = graph.nodes()[0]
        walked = dfs_component(graph, start, recursive=recursive)
        incoming = incoming_map(graph)
        for position, node in enumerate(walked[1:], start=1):
            assert incoming[node] & set(walked[:position]), (
                f"{node!r} was reached without a neighbour earlier in the sweep"
            )

    def test_direction_is_followed(self, recursive):
        """'a' reaches 'b'; 'b' reaches nothing but itself."""
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        assert dfs_component(graph, "a", recursive=recursive) == ["a", "b"]
        assert dfs_component(graph, "b", recursive=recursive) == ["b"]

    def test_undirected_components_are_symmetric(self, recursive):
        """Ignoring direction, every member of a component sees the whole of
        it, whichever member you start from.
        """
        graph = make_three_component_graph()
        for start in (1, 2, 3, 4):
            assert set(dfs_component(graph, start, recursive=recursive)) == {1, 2, 3, 4}

    def test_start_is_required_and_comes_first(self, recursive):
        graph = make_three_component_graph()
        for start in graph.nodes():
            assert dfs_component(graph, start, recursive=recursive)[0] == start

    def test_missing_start_raises_key_error(self, recursive):
        graph = Graph()
        graph.add_edge("a", "b")
        with pytest.raises(KeyError, match="not in the graph"):
            dfs_component(graph, "zz", recursive=recursive)

    def test_missing_start_on_an_empty_graph_raises(self, recursive):
        with pytest.raises(KeyError, match="not in the graph"):
            dfs_component(Graph(), "anything", recursive=recursive)

    def test_a_lone_node_is_its_own_component(self, recursive):
        graph = Graph()
        graph.add_node("solo")
        assert dfs_component(graph, "solo", recursive=recursive) == ["solo"]

    def test_a_self_looping_node_is_its_own_component(self, recursive):
        graph = Graph()
        graph.add_edge("solo", "solo")
        assert dfs_component(graph, "solo", recursive=recursive) == ["solo"]

    def test_component_of_a_complete_graph_is_everything(self, recursive):
        graph = complete_graph(12)
        assert dfs_component(graph, 0, recursive=recursive) == dfs(graph)


# ----------------------------------------------------------------------
# dfs is the iterative form
# ----------------------------------------------------------------------
class TestDfsIsTheIterativeForm:
    """The plain name delegates to dfs_iterative, and the delegation is
    checked behaviourally rather than by reading the source.
    """

    @pytest.mark.parametrize("name", ZOO_NAMES)
    def test_dfs_matches_dfs_iterative_everywhere(self, name):
        graph = GRAPH_ZOO[name]
        assert dfs(graph) == dfs_iterative(graph) == dfs_recursive(graph)

    @pytest.mark.parametrize("name", NONEMPTY_ZOO_NAMES)
    def test_dfs_matches_from_every_start_node(self, name):
        graph = GRAPH_ZOO[name]
        for start in graph.nodes():
            assert dfs(graph, start) == dfs_iterative(graph, start)

    def test_every_call_goes_through_dfs_iterative(self, monkeypatch):
        """Behavioural proof of delegation: patch the iterative form and
        watch dfs hand its arguments straight over.
        """
        calls: List[Any] = []
        sentinel = ["patched"]

        def recording_dfs_iterative(graph, start=None):
            calls.append((graph, start))
            return sentinel

        monkeypatch.setattr(dfs_module, "dfs_iterative", recording_dfs_iterative)

        graph = make_three_component_graph()
        assert dfs(graph, 5) is sentinel
        assert calls == [(graph, 5)]

    def test_dfs_does_not_touch_the_recursion_limit(self):
        """It cannot: the iterative form spends no frames."""
        before = sys.getrecursionlimit()
        dfs(path_graph(DEEP_PATH_NODES))
        assert sys.getrecursionlimit() == before
