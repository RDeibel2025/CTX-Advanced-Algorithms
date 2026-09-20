"""Tests for the graph class and its two representations.

What is being proven here is that the adjacency list and the adjacency
matrix are two views of *one* graph, and that they never disagree. The
list is the live store and the matrix is rebuilt on demand from it, so the
failure this file is built to catch is a matrix filled from a stale or
mis-ordered node list: rows shifted by one, an index left over from before
a node was removed, or a symmetric write that quietly loses a direction.
Such a matrix still looks plausible - right shape, right number of ones -
and only a cell-by-cell comparison against the list exposes it.

The centrepiece, :class:`TestRepresentationsAgree`, therefore walks
**every ordered pair** ``(u, v)`` of a graph and asserts that

    ``matrix[index[u], index[v]]`` is non-zero  <=>  ``has_edge(u, v)``

with the weights matching too when the graph is weighted, across a table
of graph shapes: empty, one node, self-loops, hand-built directed and
undirected graphs, non-integer labels, a graph whose insertion order is
deliberately not its sorted order, and generated paths, cycles, complete,
sparse, dense and weighted graphs of up to 60 nodes.

Oracles are kept independent of the code under test wherever that is
possible:

* degrees are checked against non-zero counts in a matrix row or column,
  not against ``len(get_neighbors(v))``, which is the same dict length the
  implementation returns;
* ``edge_count`` and ``density`` are checked against edge lists written
  out by hand, and against the identity ``sum(degree) == 2E - selfloops``
  that follows from storing a self-loop once;
* the round trip rebuilds a fresh :class:`Graph` from nothing but the
  matrix and its labels, then compares edge sets with the original.

Two documented behaviours are pinned here rather than treated as bugs,
because :mod:`src.graphs.graph` states both: a self-loop counts once, and
an explicit weight of 0.0 is indistinguishable from "no edge" in the
matrix while the list still knows it is there.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np
import pytest

from src.graphs.graph import AdjacencyMatrix, Graph
from src.utils.graph_generator import (
    complete_graph,
    cycle_graph,
    dense_graph,
    path_graph,
    random_graph,
    sparse_graph,
    weighted_graph,
)

#: Seed shared by every generated graph in this file, so a failure can be
#: reproduced exactly from the test id alone.
SEED = 2026


# ----------------------------------------------------------------------
# Hand-built graphs: the expected edges are written out, not computed
# ----------------------------------------------------------------------
def build(
    edges: List[Tuple[Any, ...]],
    *,
    directed: bool = False,
    weighted: bool = False,
    extra_nodes: Tuple[Any, ...] = (),
) -> Graph:
    """Build a graph from an explicit edge list, in the order given."""
    graph = Graph(directed=directed, weighted=weighted)
    for edge in edges:
        graph.add_edge(*edge)
    for node in extra_nodes:
        graph.add_node(node)
    return graph


def diamond_undirected() -> Graph:
    """a-b, a-c, b-d, c-d, plus an isolated node."""
    return build(
        [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        extra_nodes=("lonely",),
    )


def diamond_directed() -> Graph:
    """The same diamond, oriented, with one back edge and one self-loop."""
    return build(
        [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "a"), ("b", "b")],
        directed=True,
    )


def weighted_undirected() -> Graph:
    """Weights chosen to be exactly representable in float32."""
    return build(
        [("a", "b", 2.5), ("b", "c", 0.5), ("c", "a", 7.25), ("c", "c", 3.125)],
        weighted=True,
    )


def weighted_directed() -> Graph:
    """A weighted digraph where u -> v and v -> u carry different weights."""
    return build(
        [(1, 2, 4.5), (2, 1, 0.25), (2, 3, 6.0), (3, 3, 1.5)],
        directed=True,
        weighted=True,
    )


def mixed_labels() -> Graph:
    """Any hashable works as a node, and insertion order is not sorted order."""
    return build([("zebra", 7), (7, (1, 2)), ((1, 2), "zebra"), (7, 7)])


def reinserted_after_removal() -> Graph:
    """Insertion order deliberately scrambled by a removal and a re-add.

    A matrix built from a cached or sorted node list disagrees with the
    adjacency list here even though it agrees on a freshly built graph.
    """
    graph = build([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    graph.remove_node("a")
    graph.add_edge("d", "a")
    graph.add_edge("a", "c")
    return graph


def self_loops_everywhere() -> Graph:
    """A cycle in which every node also points at itself."""
    graph = cycle_graph(9)
    for node in list(graph.nodes()):
        graph.add_edge(node, node)
    return graph


def weighted_with_self_loops() -> Graph:
    graph = weighted_graph(18, density=0.3, seed=SEED)
    graph.add_edge(0, 0, 3.25)
    graph.add_edge(11, 11, 0.75)
    return graph


#: Every shape the representation-agreement sweep runs over. Values are
#: factories so no graph is shared between tests that mutate one.
GRAPH_CASES: Dict[str, Callable[[], Graph]] = {
    "empty": Graph,
    "single_node": lambda: build([], extra_nodes=("solo",)),
    "single_node_self_loop": lambda: build([("solo", "solo")]),
    "one_undirected_edge": lambda: build([("a", "b")]),
    "one_directed_edge": lambda: build([("a", "b")], directed=True),
    "directed_both_ways": lambda: build([("a", "b"), ("b", "a")], directed=True),
    "diamond_undirected": diamond_undirected,
    "diamond_directed": diamond_directed,
    "weighted_undirected": weighted_undirected,
    "weighted_directed": weighted_directed,
    "mixed_labels": mixed_labels,
    "reinserted_after_removal": reinserted_after_removal,
    "self_loops_everywhere": self_loops_everywhere,
    "path_20": lambda: path_graph(20),
    "cycle_25": lambda: cycle_graph(25),
    "complete_12": lambda: complete_graph(12),
    "sparse_60": lambda: sparse_graph(60, avg_degree=4, seed=SEED),
    "dense_40": lambda: dense_graph(40, density=0.6, seed=SEED),
    "directed_random_45": lambda: random_graph(
        45, density=0.3, directed=True, seed=SEED
    ),
    "weighted_undirected_35": lambda: weighted_graph(35, density=0.25, seed=SEED),
    "weighted_directed_30": lambda: weighted_graph(
        30, density=0.35, directed=True, seed=SEED
    ),
    "weighted_with_self_loops": weighted_with_self_loops,
}
CASE_NAMES = list(GRAPH_CASES)


@pytest.fixture(params=CASE_NAMES, ids=CASE_NAMES)
def graph_case(request) -> Graph:
    """A fresh graph of each shape in :data:`GRAPH_CASES`."""
    return GRAPH_CASES[request.param]()


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def ordered_pairs(graph: Graph) -> Set[Tuple[Any, Any]]:
    """Every ordered pair ``(u, v)`` the adjacency list calls an edge.

    Built from :meth:`Graph.has_edge` over the full V x V space rather than
    from :meth:`Graph.edges`, so it does not inherit whatever orientation
    rule ``edges()`` applies.
    """
    nodes = graph.nodes()
    return {(u, v) for u in nodes for v in nodes if graph.has_edge(u, v)}


def matrix_pairs(view: AdjacencyMatrix) -> Set[Tuple[Any, Any]]:
    """Every ordered pair the matrix calls an edge, read cell by cell."""
    return {
        (view.nodes[i], view.nodes[j])
        for i in range(len(view.nodes))
        for j in range(len(view.nodes))
        if view.matrix[i, j] != 0
    }


def assert_representations_agree(graph: Graph) -> None:
    """Fail if any cell of the matrix contradicts the adjacency list.

    This is the assertion the whole file exists for. It checks the labels
    first, because a matrix whose rows are correct for the wrong node list
    is the failure mode that a shape check or an edge count would miss.
    """
    view = graph.to_adjacency_matrix()
    matrix, nodes, index = view                      # it unpacks positionally

    assert nodes == graph.nodes(), "matrix row labels are not the graph's nodes"
    assert matrix.shape == (graph.node_count, graph.node_count)
    for position, node in enumerate(nodes):
        assert index[node] == position, f"node {node!r} is indexed to the wrong row"

    for u in nodes:
        for v in nodes:
            cell = float(matrix[index[u], index[v]])
            present = graph.has_edge(u, v)
            assert bool(cell) == present, (
                f"matrix says {cell} for ({u!r}, {v!r}) but the adjacency "
                f"list says has_edge is {present}"
            )
            if not present:
                continue
            expected = graph.get_edge_weight(u, v)
            if graph.weighted:
                assert cell == pytest.approx(expected, rel=1e-6), (
                    f"weight mismatch on ({u!r}, {v!r})"
                )
            else:
                assert cell == 1.0 and expected == 1.0


def rebuild_from_matrix(view: AdjacencyMatrix, *, directed: bool, weighted: bool) -> Graph:
    """Construct a brand new Graph from nothing but a matrix and its labels."""
    rebuilt = Graph(directed=directed, weighted=weighted)
    for node in view.nodes:
        rebuilt.add_node(node)
    for i, u in enumerate(view.nodes):
        for j, v in enumerate(view.nodes):
            cell = float(view.matrix[i, j])
            if cell:
                rebuilt.add_edge(u, v, cell if weighted else 1.0)
    return rebuilt


# ----------------------------------------------------------------------
# THE CENTREPIECE: the two representations describe the same graph
# ----------------------------------------------------------------------
class TestRepresentationsAgree:
    """Adjacency list against adjacency matrix, cell by cell, on every shape."""

    def test_every_ordered_pair_matches(self, graph_case):
        assert_representations_agree(graph_case)

    def test_pair_sets_are_identical(self, graph_case):
        """The same claim stated as a set difference, so a failure names the cells."""
        view = graph_case.to_adjacency_matrix()
        assert matrix_pairs(view) == ordered_pairs(graph_case)

    def test_matrix_agrees_again_after_the_graph_is_edited(self, graph_case):
        """A rebuilt matrix must follow the edits, not a stale node list."""
        graph = graph_case
        assert_representations_agree(graph)

        graph.add_edge("new_hub", "new_spoke", *([2.5] if graph.weighted else []))
        for node in list(graph.nodes())[:3]:
            graph.add_edge("new_hub", node, *([1.25] if graph.weighted else []))
        assert_representations_agree(graph)

        for node in list(graph.nodes())[:2]:
            graph.remove_node(node)
        assert_representations_agree(graph)

    def test_matrix_symmetry_follows_the_directed_flag(self, graph_case):
        matrix = graph_case.to_adjacency_matrix().matrix
        symmetric = bool((matrix == matrix.T).all())
        if not graph_case.directed:
            assert symmetric, "an undirected graph must produce a symmetric matrix"

    def test_directed_graphs_really_are_asymmetric(self):
        """The symmetry check above is only meaningful if asymmetry is possible."""
        matrix = build([("a", "b")], directed=True).to_adjacency_matrix().matrix
        assert not (matrix == matrix.T).all()

    def test_degree_equals_the_row_count_of_non_zero_cells(self, graph_case):
        """Row and column scans are the matrix's own answer for the degrees."""
        view = graph_case.to_adjacency_matrix()
        for node in graph_case.nodes():
            row = view.matrix[view.index[node], :]
            column = view.matrix[:, view.index[node]]
            assert graph_case.degree(node) == int(np.count_nonzero(row))
            assert graph_case.out_degree(node) == int(np.count_nonzero(row))
            assert graph_case.in_degree(node) == int(np.count_nonzero(column))

    def test_neighbour_list_matches_the_matrix_row(self, graph_case):
        view = graph_case.to_adjacency_matrix()
        for node in graph_case.nodes():
            row = view.matrix[view.index[node], :]
            from_matrix = {view.nodes[j] for j in np.nonzero(row)[0]}
            assert set(graph_case.neighbors(node)) == from_matrix

    def test_edge_count_matches_the_cells_the_matrix_holds(self, graph_case):
        """E, recovered from the matrix by the O(V^2) scan it forces."""
        view = graph_case.to_adjacency_matrix()
        filled = int(np.count_nonzero(view.matrix))
        loops = int(np.count_nonzero(np.diag(view.matrix)))
        if graph_case.directed:
            expected = filled
        else:
            # Every off-diagonal edge occupies two mirrored cells; a
            # self-loop occupies one, and is one edge.
            expected = (filled - loops) // 2 + loops
        assert graph_case.edge_count == expected


# ----------------------------------------------------------------------
# Empty and single-node graphs
# ----------------------------------------------------------------------
class TestEmptyGraph:
    """A graph with nothing in it answers every query without raising."""

    @pytest.mark.parametrize(
        "graph",
        [Graph(), Graph(directed=True), Graph(weighted=True), Graph(True, True)],
        ids=["plain", "directed", "weighted", "both"],
    )
    def test_counts_and_collections_are_empty(self, graph):
        assert graph.node_count == 0 and graph.edge_count == 0
        assert graph.nodes() == [] and graph.edges() == []
        assert len(graph) == 0 and list(graph) == []
        assert graph.density() == 0.0

    def test_matrix_of_an_empty_graph_is_zero_by_zero(self):
        view = Graph().to_adjacency_matrix()
        assert view.matrix.shape == (0, 0)
        assert view.matrix.nbytes == 0
        assert view.nodes == [] and view.index == {}

    def test_nothing_is_a_member_and_nothing_is_an_edge(self):
        graph = Graph()
        assert "a" not in graph
        assert graph.has_edge("a", "b") is False


class TestSingleNode:
    """One node, no edges: degree zero, density zero, a 1x1 matrix."""

    def test_isolated_node_is_present_with_no_neighbours(self):
        graph = Graph()
        graph.add_node("solo")
        assert graph.nodes() == ["solo"] and "solo" in graph
        assert graph.node_count == 1 and graph.edge_count == 0
        assert graph.get_neighbors("solo") == []
        assert graph.degree("solo") == 0
        assert graph.in_degree("solo") == 0 and graph.out_degree("solo") == 0

    def test_density_is_zero_because_no_edge_is_possible(self):
        graph = Graph()
        graph.add_node("solo")
        assert graph.density() == 0.0

    def test_matrix_is_one_cell_holding_zero(self):
        graph = Graph()
        graph.add_node("solo")
        view = graph.to_adjacency_matrix()
        assert view.matrix.shape == (1, 1)
        assert view.matrix.tolist() == [[0]]
        assert view.nodes == ["solo"] and view.index == {"solo": 0}

    def test_add_node_is_idempotent_and_keeps_edges(self):
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_node("a")
        assert graph.node_count == 2
        assert graph.get_neighbors("a") == ["b"]


# ----------------------------------------------------------------------
# Self-loops
# ----------------------------------------------------------------------
class TestSelfLoops:
    """A self-loop is legal, is stored once, and sits on the diagonal."""

    @pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
    def test_self_loop_counts_once(self, directed):
        graph = Graph(directed=directed)
        graph.add_edge("a", "a")
        assert graph.edge_count == 1
        assert graph.degree("a") == 1          # once, not the textbook 2
        assert graph.in_degree("a") == 1 and graph.out_degree("a") == 1
        assert graph.get_neighbors("a") == ["a"]
        assert graph.edges() == [("a", "a")]

    def test_self_loop_lands_on_the_diagonal(self):
        graph = build([("a", "a"), ("a", "b")])
        view = graph.to_adjacency_matrix()
        assert view.matrix[view.index["a"], view.index["a"]] == 1
        assert view.matrix[view.index["b"], view.index["b"]] == 0

    def test_removing_a_self_loop_leaves_the_node(self):
        graph = build([("a", "a"), ("a", "b")])
        graph.remove_edge("a", "a")
        assert graph.has_edge("a", "a") is False
        assert graph.nodes() == ["a", "b"] and graph.edge_count == 1

    def test_removing_a_looped_node_discounts_the_loop_once(self):
        graph = build([("a", "a"), ("a", "b"), ("b", "c")])
        graph.remove_node("a")
        assert graph.nodes() == ["b", "c"]
        assert graph.edge_count == 1

    def test_directed_looped_node_with_incoming_edges(self):
        graph = build([(1, 1), (2, 1), (1, 3)], directed=True)
        assert graph.edge_count == 3
        graph.remove_node(1)
        assert graph.nodes() == [2, 3]
        assert graph.edges() == [] and graph.edge_count == 0

    def test_degree_sum_identity_with_loops(self):
        """sum(degree) == 2E - loops, because a loop is stored once."""
        graph = sparse_graph(40, avg_degree=4, seed=SEED)
        graph.add_edge(0, 0)
        graph.add_edge(7, 7)
        loops = sum(1 for node in graph if graph.has_edge(node, node))
        assert loops == 2
        assert sum(graph.degree(node) for node in graph) == 2 * graph.edge_count - loops


# ----------------------------------------------------------------------
# Direction
# ----------------------------------------------------------------------
class TestDirection:
    """Undirected edges are symmetric; directed edges are not."""

    def test_undirected_edge_is_visible_from_both_endpoints(self):
        graph = build([("a", "b")])
        assert graph.has_edge("a", "b") and graph.has_edge("b", "a")
        assert graph.get_neighbors("a") == ["b"]
        assert graph.get_neighbors("b") == ["a"]

    def test_directed_edge_is_visible_from_one_endpoint_only(self):
        graph = build([("a", "b")], directed=True)
        assert graph.has_edge("a", "b") is True
        assert graph.has_edge("b", "a") is False
        assert graph.get_neighbors("a") == ["b"]
        assert graph.get_neighbors("b") == []

    def test_the_two_directions_are_separate_edges_when_directed(self):
        graph = build([("a", "b"), ("b", "a")], directed=True)
        assert graph.edge_count == 2
        assert set(graph.edges()) == {("a", "b"), ("b", "a")}

    def test_the_two_directions_are_one_edge_when_undirected(self):
        graph = build([("a", "b"), ("b", "a")])
        assert graph.edge_count == 1
        assert graph.edges() == [("a", "b")]

    def test_undirected_removal_works_from_either_side(self):
        graph = build([("a", "b"), ("b", "c")])
        graph.remove_edge("b", "a")
        assert graph.has_edge("a", "b") is False
        assert graph.nodes() == ["a", "b", "c"] and graph.edge_count == 1

    def test_directed_removal_leaves_the_opposite_direction(self):
        graph = build([("a", "b"), ("b", "a")], directed=True)
        graph.remove_edge("a", "b")
        assert graph.has_edge("a", "b") is False
        assert graph.has_edge("b", "a") is True

    def test_directed_weights_are_independent_per_direction(self):
        graph = build([(1, 2, 4.5), (2, 1, 0.25)], directed=True, weighted=True)
        assert graph.get_edge_weight(1, 2) == 4.5
        assert graph.get_edge_weight(2, 1) == 0.25

    def test_flags_are_readable_and_fixed_at_construction(self):
        for directed in (False, True):
            for weighted in (False, True):
                graph = Graph(directed=directed, weighted=weighted)
                assert graph.directed is directed and graph.weighted is weighted


# ----------------------------------------------------------------------
# add_edge
# ----------------------------------------------------------------------
class TestAddEdge:
    """Endpoints are created on demand; a repeat updates rather than duplicates."""

    def test_missing_endpoints_are_created_in_argument_order(self):
        graph = Graph()
        graph.add_edge("first", "second")
        assert graph.nodes() == ["first", "second"]
        assert graph.node_count == 2 and graph.edge_count == 1

    def test_only_the_absent_endpoint_is_created(self):
        graph = Graph()
        graph.add_node("known")
        graph.add_edge("known", "fresh")
        assert graph.nodes() == ["known", "fresh"]

    @pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
    def test_re_adding_an_edge_updates_the_weight_without_double_counting(self, directed):
        graph = Graph(directed=directed, weighted=True)
        graph.add_edge("a", "b", 4)
        assert graph.get_edge_weight("a", "b") == 4.0
        graph.add_edge("a", "b", 1.5)
        assert graph.get_edge_weight("a", "b") == 1.5
        assert graph.edge_count == 1

    def test_undirected_reweight_updates_both_stored_directions(self):
        graph = Graph(weighted=True)
        graph.add_edge("a", "b", 4)
        graph.add_edge("b", "a", 1.5)
        assert graph.get_edge_weight("a", "b") == 1.5
        assert graph.get_edge_weight("b", "a") == 1.5
        assert graph.edge_count == 1

    def test_integer_weights_become_floats(self):
        graph = Graph(weighted=True)
        graph.add_edge("a", "b", 7)
        assert graph.get_edge_weight("a", "b") == 7.0
        assert isinstance(graph.get_edge_weight("a", "b"), float)

    def test_negative_weights_are_accepted_by_the_graph(self):
        """The graph stores them; it is Dijkstra that rejects them."""
        graph = Graph(weighted=True)
        graph.add_edge("a", "b", -3.5)
        assert graph.get_edge_weight("a", "b") == -3.5

    @pytest.mark.parametrize("weight", [0.0, 2.0, -1.0, 3, 0.5])
    def test_any_non_default_weight_on_an_unweighted_graph_raises(self, weight):
        with pytest.raises(ValueError, match="unweighted graph"):
            Graph().add_edge("a", "b", weight)

    def test_the_rejected_edge_is_not_half_added(self):
        graph = Graph()
        with pytest.raises(ValueError):
            graph.add_edge("a", "b", 3.0)
        assert graph.node_count == 0 and graph.edge_count == 0

    def test_the_default_weight_is_allowed_on_an_unweighted_graph(self):
        graph = Graph()
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("b", "c", 1)
        assert graph.edge_count == 2
        assert graph.get_edge_weight("a", "b") == 1.0

    def test_unweighted_graphs_report_weight_one_rather_than_refusing(self):
        graph = build([("a", "b")])
        assert graph.get_edge_weight("a", "b") == 1.0
        assert list(graph.neighbor_items("a")) == [("b", 1.0)]

    def test_unhashable_endpoints_raise_type_error(self):
        with pytest.raises(TypeError):
            Graph().add_edge(["unhashable"], "a")


# ----------------------------------------------------------------------
# Removal
# ----------------------------------------------------------------------
class TestRemoval:
    """Removing a node takes every edge that touched it, in both directions."""

    def test_remove_node_clears_undirected_edges(self):
        graph = build([("a", "b"), ("b", "c")])
        graph.remove_node("b")
        assert graph.nodes() == ["a", "c"]
        assert graph.edges() == [] and graph.edge_count == 0
        assert graph.get_neighbors("a") == [] and graph.get_neighbors("c") == []

    def test_remove_node_clears_incoming_directed_edges(self):
        """The failure this guards: u -> v surviving after v is gone."""
        graph = build([(1, 2), (3, 2), (2, 4)], directed=True)
        graph.remove_node(2)
        assert graph.nodes() == [1, 3, 4]
        assert graph.edges() == [] and graph.edge_count == 0
        assert ordered_pairs(graph) == set()
        assert not np.count_nonzero(graph.to_adjacency_matrix().matrix)

    def test_remove_node_keeps_unrelated_edges(self):
        graph = build([("a", "b"), ("c", "d"), ("b", "c")], directed=True)
        graph.remove_node("b")
        assert set(graph.edges()) == {("c", "d")}
        assert graph.edge_count == 1

    def test_remove_edge_keeps_both_nodes(self):
        graph = build([("a", "b")])
        graph.remove_edge("a", "b")
        assert graph.nodes() == ["a", "b"]
        assert graph.edge_count == 0

    @pytest.mark.parametrize(
        "pair", [("a", "c"), ("a", "zz"), ("zz", "a"), ("zz", "yy")]
    )
    def test_remove_edge_raises_when_the_edge_is_absent(self, pair):
        graph = build([("a", "b")], extra_nodes=("c",))
        with pytest.raises(KeyError, match="no edge"):
            graph.remove_edge(*pair)

    def test_remove_edge_raises_on_the_wrong_direction(self):
        graph = build([("a", "b")], directed=True)
        with pytest.raises(KeyError, match="no edge"):
            graph.remove_edge("b", "a")

    def test_remove_node_raises_when_the_node_is_absent(self):
        graph = build([("a", "b")])
        with pytest.raises(KeyError, match="not in the graph"):
            graph.remove_node("zz")

    def test_remove_node_raises_a_second_time(self):
        graph = build([("a", "b")])
        graph.remove_node("a")
        with pytest.raises(KeyError, match="not in the graph"):
            graph.remove_node("a")

    @pytest.mark.parametrize("method", ["get_neighbors", "neighbors", "neighbor_items"])
    def test_neighbour_lookups_raise_on_a_missing_node(self, method):
        graph = build([("a", "b")])
        with pytest.raises(KeyError, match="not in the graph"):
            getattr(graph, method)("zz")

    @pytest.mark.parametrize("method", ["degree", "in_degree", "out_degree"])
    def test_degree_lookups_raise_on_a_missing_node(self, method):
        graph = build([("a", "b")])
        with pytest.raises(KeyError, match="not in the graph"):
            getattr(graph, method)("zz")

    def test_get_edge_weight_raises_where_has_edge_returns_false(self):
        graph = build([("a", "b")])
        assert graph.has_edge("a", "zz") is False
        with pytest.raises(KeyError, match="no edge"):
            graph.get_edge_weight("a", "zz")

    def test_repeated_removal_and_re_addition_keeps_the_count_honest(self):
        graph = build([("a", "b"), ("b", "c"), ("c", "a")])
        for _ in range(5):
            graph.remove_edge("a", "b")
            assert graph.edge_count == 2
            graph.add_edge("a", "b")
            assert graph.edge_count == 3
        assert ordered_pairs(graph) == {
            ("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "a"), ("a", "c"),
        }


# ----------------------------------------------------------------------
# Counts, degrees, density
# ----------------------------------------------------------------------
class TestCountsAndDegrees:
    """V, E, the three degrees and density, against hand-written expectations."""

    def test_node_and_edge_counts_of_a_hand_built_graph(self):
        graph = diamond_undirected()
        assert graph.node_count == 5 and len(graph) == 5
        assert graph.edge_count == 4
        assert graph.nodes() == ["a", "b", "c", "d", "lonely"]

    def test_undirected_edges_are_counted_once(self):
        graph = build([("a", "b"), ("b", "a"), ("a", "b")])
        assert graph.edge_count == 1

    def test_directed_degrees_split_into_in_and_out(self):
        graph = build([("a", "b"), ("c", "b"), ("b", "d")], directed=True)
        assert (graph.out_degree("a"), graph.in_degree("a")) == (1, 0)
        assert (graph.out_degree("b"), graph.in_degree("b")) == (1, 2)
        assert (graph.out_degree("d"), graph.in_degree("d")) == (0, 1)
        assert graph.degree("b") == graph.out_degree("b")

    def test_undirected_degrees_agree_with_each_other(self):
        graph = diamond_undirected()
        for node in graph:
            assert graph.degree(node) == graph.in_degree(node) == graph.out_degree(node)

    @pytest.mark.parametrize(
        "directed, expected", [(False, 0.5), (True, 0.25)], ids=["undirected", "directed"]
    )
    def test_density_of_a_three_edge_chain(self, directed, expected):
        graph = build([("a", "b"), ("b", "c"), ("c", "d")], directed=directed)
        assert graph.density() == expected

    @pytest.mark.parametrize("node_count", [0, 1])
    def test_density_is_zero_when_no_edge_is_possible(self, node_count):
        graph = Graph()
        for i in range(node_count):
            graph.add_node(i)
        assert graph.density() == 0.0

    def test_complete_graph_has_density_one(self):
        assert complete_graph(10).density() == pytest.approx(1.0)
        assert complete_graph(10, directed=True).density() == pytest.approx(1.0)

    def test_density_against_the_formula_on_generated_graphs(self):
        for graph in [
            sparse_graph(50, avg_degree=4, seed=SEED),
            dense_graph(30, density=0.6, seed=SEED),
            random_graph(25, density=0.3, directed=True, seed=SEED),
        ]:
            n = graph.node_count
            possible = n * (n - 1) if graph.directed else n * (n - 1) / 2
            assert graph.density() == pytest.approx(graph.edge_count / possible)

    def test_self_loops_can_push_density_above_one(self):
        """Documented: loops count in the numerator, not the denominator."""
        graph = build([("a", "a"), ("b", "b")])
        assert graph.node_count == 2 and graph.edge_count == 2
        assert graph.density() == 2.0

    def test_directed_edge_sum_identities(self):
        graph = random_graph(30, density=0.25, directed=True, seed=SEED)
        graph.add_edge(3, 3)
        assert sum(graph.out_degree(n) for n in graph) == graph.edge_count
        assert sum(graph.in_degree(n) for n in graph) == graph.edge_count


# ----------------------------------------------------------------------
# Neighbour access
# ----------------------------------------------------------------------
class TestNeighbourAccess:
    """get_neighbors changes shape with the flag; neighbors/neighbor_items do not."""

    def test_unweighted_get_neighbors_returns_bare_nodes(self):
        graph = build([("a", "b"), ("a", "c")])
        assert graph.get_neighbors("a") == ["b", "c"]

    def test_weighted_get_neighbors_returns_pairs(self):
        graph = build([("a", "b", 2.5), ("a", "c", 7)], weighted=True)
        assert graph.get_neighbors("a") == [("b", 2.5), ("c", 7.0)]

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_neighbors_is_always_bare_and_neighbor_items_always_pairs(self, weighted):
        graph = Graph(weighted=weighted)
        graph.add_edge("a", "b", 2.5 if weighted else 1.0)
        assert list(graph.neighbors("a")) == ["b"]
        assert list(graph.neighbor_items("a")) == [("b", 2.5 if weighted else 1.0)]

    def test_neighbours_follow_edge_insertion_order(self):
        graph = Graph()
        for target in ["m", "a", "z", "b"]:
            graph.add_edge("hub", target)
        assert graph.get_neighbors("hub") == ["m", "a", "z", "b"]

    def test_get_neighbors_returns_a_copy(self):
        graph = build([("a", "b")])
        neighbours = graph.get_neighbors("a")
        neighbours.append("intruder")
        assert graph.get_neighbors("a") == ["b"]
        assert graph.degree("a") == 1

    def test_directed_neighbours_are_successors_only(self):
        graph = build([("a", "b"), ("c", "a")], directed=True)
        assert graph.get_neighbors("a") == ["b"]


# ----------------------------------------------------------------------
# The matrix itself
# ----------------------------------------------------------------------
class TestAdjacencyMatrix:
    """dtype, footprint, row order and the snapshot rule."""

    @pytest.mark.parametrize(
        "weighted, dtype, itemsize",
        [(False, np.uint8, 1), (True, np.float32, 4)],
        ids=["unweighted_uint8", "weighted_float32"],
    )
    def test_default_dtype_and_footprint(self, weighted, dtype, itemsize):
        graph = weighted_graph(24, density=0.3, seed=SEED) if weighted else sparse_graph(
            24, avg_degree=4, seed=SEED
        )
        view = graph.to_adjacency_matrix()
        assert view.matrix.dtype == np.dtype(dtype)
        assert view.matrix.nbytes == 24 * 24 * itemsize

    @pytest.mark.parametrize("dtype", [np.int16, np.float64, np.int64])
    def test_dtype_can_be_overridden(self, dtype):
        graph = build([("a", "b")])
        view = graph.to_adjacency_matrix(dtype=dtype)
        assert view.matrix.dtype == np.dtype(dtype)
        assert view.matrix[view.index["a"], view.index["b"]] == 1

    def test_row_order_is_insertion_order_not_sorted_order(self):
        graph = Graph()
        for u, v in [("z", "m"), ("m", "a"), ("q", "z")]:
            graph.add_edge(u, v)
        view = graph.to_adjacency_matrix()
        assert view.nodes == ["z", "m", "a", "q"] == graph.nodes()
        assert view.nodes != sorted(view.nodes)
        assert view.index == {"z": 0, "m": 1, "a": 2, "q": 3}

    def test_row_order_follows_a_removal_and_re_insertion(self):
        graph = reinserted_after_removal()
        view = graph.to_adjacency_matrix()
        assert view.nodes == graph.nodes()
        assert view.nodes[-1] == "a", "a was re-added and belongs at the end"
        assert_representations_agree(graph)

    def test_the_view_unpacks_positionally(self):
        view = build([("a", "b")]).to_adjacency_matrix()
        matrix, nodes, index = view
        assert nodes == ["a", "b"] and index == {"a": 0, "b": 1}
        assert int(matrix[index["a"], index["b"]]) == 1
        assert isinstance(view, AdjacencyMatrix)

    def test_the_matrix_is_a_snapshot_and_never_cached(self):
        graph = build([("a", "b")])
        first = graph.to_adjacency_matrix()
        graph.add_edge("b", "c")
        second = graph.to_adjacency_matrix()
        assert first.matrix.shape == (2, 2), "the old view must not follow the edit"
        assert second.matrix.shape == (3, 3)
        assert first.matrix is not second.matrix

    def test_matrix_is_all_zeros_for_an_edgeless_graph(self):
        graph = Graph()
        for node in "abcd":
            graph.add_node(node)
        view = graph.to_adjacency_matrix()
        assert view.matrix.shape == (4, 4)
        assert not view.matrix.any()

    def test_an_explicit_zero_weight_is_invisible_in_the_matrix(self):
        """Documented limitation, not a defect: 0.0 means "no edge" to a matrix.

        src/graphs/graph.py states it in the module docstring and points
        callers at has_edge, which is what this asserts.
        """
        graph = Graph(weighted=True)
        graph.add_edge("x", "y", 0.0)
        view = graph.to_adjacency_matrix()
        assert graph.has_edge("x", "y") is True
        assert graph.get_edge_weight("x", "y") == 0.0
        assert view.matrix[view.index["x"], view.index["y"]] == 0.0


# ----------------------------------------------------------------------
# Round trip: list -> matrix -> list
# ----------------------------------------------------------------------
class TestRoundTrip:
    """A graph rebuilt from its own matrix holds exactly the same edges."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: path_graph(15),
            lambda: cycle_graph(15),
            lambda: complete_graph(8),
            lambda: sparse_graph(40, avg_degree=4, seed=SEED),
            lambda: dense_graph(20, density=0.6, seed=SEED),
            lambda: random_graph(25, density=0.3, directed=True, seed=SEED),
            lambda: sparse_graph(30, avg_degree=4, directed=True, seed=SEED),
            self_loops_everywhere,
            diamond_directed,
            mixed_labels,
        ],
        ids=[
            "path", "cycle", "complete", "sparse", "dense", "directed_random",
            "directed_sparse", "self_loops", "diamond_directed", "mixed_labels",
        ],
    )
    def test_unweighted_round_trip_preserves_every_edge(self, factory):
        original = factory()
        view = original.to_adjacency_matrix()
        rebuilt = rebuild_from_matrix(
            view, directed=original.directed, weighted=original.weighted
        )
        assert rebuilt.nodes() == original.nodes()
        assert rebuilt.node_count == original.node_count
        assert rebuilt.edge_count == original.edge_count
        assert ordered_pairs(rebuilt) == ordered_pairs(original)
        assert set(rebuilt.edges()) == set(original.edges())

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: weighted_graph(20, density=0.3, seed=SEED),
            lambda: weighted_graph(18, density=0.35, directed=True, seed=SEED),
            weighted_undirected,
            weighted_directed,
            weighted_with_self_loops,
        ],
        ids=["generated", "generated_directed", "hand_built", "hand_built_directed",
             "with_self_loops"],
    )
    def test_weighted_round_trip_preserves_edges_and_weights(self, factory):
        original = factory()
        view = original.to_adjacency_matrix()
        rebuilt = rebuild_from_matrix(view, directed=original.directed, weighted=True)
        assert rebuilt.nodes() == original.nodes()
        assert ordered_pairs(rebuilt) == ordered_pairs(original)
        for u, v in ordered_pairs(original):
            # float32 storage is the only loss in the trip, so compare at
            # single precision rather than exactly.
            assert rebuilt.get_edge_weight(u, v) == pytest.approx(
                original.get_edge_weight(u, v), rel=1e-6
            )

    def test_a_second_trip_is_a_fixed_point(self):
        original = sparse_graph(25, avg_degree=4, seed=SEED)
        once = rebuild_from_matrix(
            original.to_adjacency_matrix(), directed=False, weighted=False
        )
        twice = rebuild_from_matrix(
            once.to_adjacency_matrix(), directed=False, weighted=False
        )
        assert once.to_adjacency_matrix().matrix.tolist() == (
            twice.to_adjacency_matrix().matrix.tolist()
        )
        assert set(twice.edges()) == set(original.edges())


# ----------------------------------------------------------------------
# Edge listing, iteration and the dunder protocol
# ----------------------------------------------------------------------
class TestInspection:
    """edges(), membership, iteration, __str__ and __repr__."""

    def test_undirected_edges_are_oriented_by_insertion_order(self):
        graph = build([("a", "b"), ("c", "a")])
        assert graph.edges() == [("a", "b"), ("a", "c")]

    def test_weighted_edges_carry_the_weight(self):
        graph = build([("a", "b", 2.5)], weighted=True)
        assert graph.edges() == [("a", "b", 2.5)]

    def test_every_edge_is_listed_exactly_once(self, graph_case):
        listed = graph_case.edges()
        pairs = [(edge[0], edge[1]) for edge in listed]
        assert len(pairs) == len(set(pairs)) == graph_case.edge_count
        for u, v in pairs:
            assert graph_case.has_edge(u, v)

    def test_membership_iteration_and_length(self):
        graph = build([("a", "b"), ("b", "c")])
        assert len(graph) == 3 == graph.node_count
        assert list(graph) == ["a", "b", "c"] == graph.nodes()
        assert "a" in graph and "zz" not in graph

    def test_unhashable_values_answer_false_instead_of_raising(self):
        graph = build([("a", "b")])
        assert (["unhashable"] in graph) is False
        assert graph.has_edge(["unhashable"], "a") is False
        assert graph.has_edge("a", {"also": "unhashable"}) is False

    def test_nodes_returns_a_copy(self):
        graph = build([("a", "b")])
        listed = graph.nodes()
        listed.append("intruder")
        assert graph.nodes() == ["a", "b"]

    @pytest.mark.parametrize(
        "directed, weighted, expected",
        [
            (False, False, "Graph(undirected, unweighted): 3 nodes, 2 edges"),
            (True, False, "Graph(directed, unweighted): 3 nodes, 2 edges"),
            (False, True, "Graph(undirected, weighted): 3 nodes, 2 edges"),
            (True, True, "Graph(directed, weighted): 3 nodes, 2 edges"),
        ],
        ids=["plain", "directed", "weighted", "both"],
    )
    def test_str_names_the_counts_and_both_flags(self, directed, weighted, expected):
        graph = Graph(directed=directed, weighted=weighted)
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        assert str(graph) == expected

    def test_str_of_an_empty_graph(self):
        assert str(Graph()) == "Graph(undirected, unweighted): 0 nodes, 0 edges"

    def test_repr_names_both_flags_and_both_counts(self):
        graph = Graph(weighted=True)
        graph.add_edge("a", "b", 2.0)
        assert repr(graph) == "Graph(directed=False, weighted=True, nodes=2, edges=1)"
