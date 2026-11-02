"""Week 10 tests: Edmonds-Karp on the spec 9.1 anchor and against networkx.

Every expected value is worked out by hand from the edge order, so the tests do
not repeat the solver's output. Tracing BFS on the 9.1 network by hand gives
three shortest augmenting paths:

=============== ============== ==========
Path            Bottleneck     Running
=============== ============== ==========
0-1-3-5         12             12
0-2-4-5         4              16
0-2-4-3-5       7              23
=============== ============== ==========

The fourth BFS reaches only {0, 1, 2, 4}, which is the source side of the cut
(1, 3) + (4, 3) + (4, 5) = 12 + 7 + 4 = 23. The three paths have 3 + 3 + 4 = 10
edges and each path edge changes two capacities, so 20 residual updates.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.flow_network_solver.core.edmonds_karp import edmonds_karp, min_cut
from src.flow_network_solver.core.network import Network, cut_edges, cut_value
from src.flow_network_solver.utils.generators import dense, sparse
from src.flow_network_solver.utils.validators import validate_flow
from src.utils.random_utils import derive_seed

CLRS_EDGES = [
    (0, 1, 16),
    (0, 2, 13),
    (2, 1, 4),
    (1, 3, 12),
    (3, 2, 9),
    (2, 4, 14),
    (4, 3, 7),
    (3, 5, 20),
    (4, 5, 4),
]
S, T = 0, 5

#: The trace from the module docstring, with edge indices read off CLRS_EDGES
#: (the k-th edge in the list has forward index 2k).
TRACE = [
    {"path": [0, 1, 3, 5], "edges": [0, 6, 14], "bottleneck": 12, "value": 12},
    {"path": [0, 2, 4, 5], "edges": [2, 10, 16], "bottleneck": 4, "value": 16},
    {"path": [0, 2, 4, 3, 5], "edges": [2, 10, 12, 14], "bottleneck": 7, "value": 23},
]

#: The flow after those three paths, summed by hand: 0 -> 2 carries 4 + 7, 2 -> 4
#: carries 4 + 7 and 3 -> 5 carries 12 + 7.
CLRS_FLOW = {
    (0, 1): 12,
    (0, 2): 11,
    (1, 3): 12,
    (2, 4): 11,
    (4, 3): 7,
    (3, 5): 19,
    (4, 5): 4,
}

#: 0-1-2-5 goes first and blocks 2 -> 5. The only other path, 0-3-2-1-4-5, must
#: cross 2 -> 1, the reverse of edge 1 -> 2, and cancels that unit.
REVERSE_EDGES = [
    (0, 1, 1),
    (1, 2, 1),
    (2, 5, 1),
    (0, 3, 1),
    (3, 2, 1),
    (1, 4, 1),
    (4, 5, 1),
]


def clrs() -> Network:
    """Build the spec 9.1 network."""
    return Network.from_edges(6, CLRS_EDGES)


def networkx_value(net: Network, s: int, t: int) -> int:
    """Return networkx's max flow value, summing parallel edges for a DiGraph."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n))
    for _, u, v, capacity, _ in net.forward_edges():
        if graph.has_edge(u, v):
            graph[u][v]["capacity"] += capacity
        else:
            graph.add_edge(u, v, capacity=capacity)
    return nx.maximum_flow_value(graph, s, t)


# --- Spec 9.1 anchor ---


def test_clrs_max_flow_is_23():
    assert edmonds_karp(clrs(), S, T).value == 23


def test_clrs_min_cut_source_side():
    net = clrs()
    edmonds_karp(net, S, T)
    assert min_cut(net, S) == {0, 1, 2, 4}


def test_clrs_cut_value_is_23():
    net = clrs()
    edmonds_karp(net, S, T)
    assert sorted(cut_edges(net, S)) == [(1, 3), (4, 3), (4, 5)]
    assert cut_value(net, S) == 23


def test_clrs_passes_validate_flow():
    net = clrs()
    result = edmonds_karp(net, S, T)
    assert validate_flow(net, S, T, result.value) == (True, "all checks passed")


def test_clrs_flow_matrix():
    assert edmonds_karp(clrs(), S, T).flow == CLRS_FLOW


# --- Counters ---


def test_bottlenecks_sum_to_the_value():
    result = edmonds_karp(clrs(), S, T)
    assert result.bottlenecks == [12, 4, 7]
    assert sum(result.bottlenecks) == result.value


def test_iterations_equal_the_number_of_bottlenecks():
    result = edmonds_karp(clrs(), S, T)
    assert result.iterations == 3
    assert result.iterations == len(result.bottlenecks)


def test_residual_updates_count_two_per_path_edge():
    assert edmonds_karp(clrs(), S, T).residual_updates == 20


def test_residual_updates_single_path():
    net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
    assert edmonds_karp(net, 0, 2).residual_updates == 4


# --- record=True ---


def test_steps_are_none_without_record():
    assert edmonds_karp(clrs(), S, T).steps is None


def test_record_steps_follow_the_hand_trace():
    steps = edmonds_karp(clrs(), S, T, record=True).steps
    assert [{key: step[key] for key in TRACE[0]} for step in steps] == TRACE


def test_record_flow_is_the_flow_after_each_path():
    steps = edmonds_karp(clrs(), S, T, record=True).steps
    assert steps[0]["flow"] == {(0, 1): 12, (1, 3): 12, (3, 5): 12}
    assert steps[1]["flow"] == {(0, 1): 12, (1, 3): 12, (3, 5): 12, (0, 2): 4, (2, 4): 4, (4, 5): 4}
    assert steps[2]["flow"] == CLRS_FLOW


def test_record_with_no_path_is_an_empty_list():
    assert edmonds_karp(Network.from_edges(3, [(1, 2, 5)]), 0, 2, record=True).steps == []


def test_record_leaves_the_answer_unchanged():
    net = clrs()
    result = edmonds_karp(net, S, T, record=True)
    assert (result.value, result.bottlenecks, result.residual_updates) == (23, [12, 4, 7], 20)
    assert validate_flow(net, S, T, result.value)[0]


# --- Residual behaviour ---


def test_reverse_edge_cancels_earlier_flow():
    net = Network.from_edges(6, REVERSE_EDGES)
    result = edmonds_karp(net, 0, 5, record=True)
    assert result.value == 2
    assert [step["path"] for step in result.steps] == [[0, 1, 2, 5], [0, 3, 2, 1, 4, 5]]
    assert result.steps[1]["edges"] == [6, 8, 3, 10, 12]
    assert result.flow == {(0, 1): 1, (1, 4): 1, (4, 5): 1, (0, 3): 1, (3, 2): 1, (2, 5): 1}
    assert validate_flow(net, 0, 5, result.value)[0]


def test_parallel_edges_add_up():
    net = Network.from_edges(2, [(0, 1, 3), (0, 1, 4)])
    result = edmonds_karp(net, 0, 1)
    assert (result.value, result.iterations, result.flow) == (7, 2, {(0, 1): 7})


def test_antiparallel_edges():
    edges = [(0, 1, 5), (1, 2, 3), (2, 1, 2), (1, 3, 1), (2, 3, 4)]
    net = Network.from_edges(4, edges)
    result = edmonds_karp(net, 0, 3)
    assert result.value == 4
    assert validate_flow(net, 0, 3, result.value)[0]


def test_no_path_gives_zero_flow():
    net = Network.from_edges(4, [(0, 1, 5), (2, 3, 5)])
    result = edmonds_karp(net, 0, 3)
    assert (result.value, result.flow, result.iterations) == (0, {}, 0)
    assert min_cut(net, 0) == {0, 1}


def test_reset_allows_a_second_identical_run():
    net = clrs()
    edmonds_karp(net, S, T)
    net.reset()
    second = edmonds_karp(net, S, T)
    assert (second.value, second.bottlenecks, second.flow) == (23, [12, 4, 7], CLRS_FLOW)


def test_second_run_without_reset_adds_nothing():
    net = clrs()
    edmonds_karp(net, S, T)
    result = edmonds_karp(net, S, T)
    assert (result.value, result.iterations) == (0, 0)
    assert validate_flow(net, S, T, 23)[0]


# --- Invalid terminals ---


def test_equal_terminals_raise():
    with pytest.raises(ValueError):
        edmonds_karp(clrs(), 2, 2)


@pytest.mark.parametrize("s, t", [(-1, 5), (6, 5), (0, 6), (0, -1)])
def test_out_of_range_terminals_raise(s, t):
    with pytest.raises(ValueError):
        edmonds_karp(clrs(), s, t)


# --- Cross-check against networkx ---


@pytest.mark.parametrize("i", range(25))
def test_agrees_with_networkx(i):
    seed = derive_seed(10, "edmonds_karp", i)
    n = 5 + i * 55 // 24  # 5 up to 60
    net = dense(n, seed) if i % 2 else sparse(n, seed)
    expected = networkx_value(net, 0, n - 1)
    result = edmonds_karp(net, 0, n - 1)
    assert result.value == expected
    assert sum(result.bottlenecks) == expected
    assert cut_value(net, 0) == expected
    assert validate_flow(net, 0, n - 1, result.value)[0]
