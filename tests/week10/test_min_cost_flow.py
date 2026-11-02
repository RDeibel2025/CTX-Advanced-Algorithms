"""Week 10 tests: min-cost max-flow (spec 3.3, 9.1, 9.2, 9.4).

The anchors are worked out by hand. On the spec 9.2 network the cheapest path
costs 3 for 2 units (0-2-3), then 4 for 2 units (0-1-2-3), then 5 for 2 units
(0-1-3), which gives the cost of every amount from 1 to 6 and a total of 24.
networkx is the independent reference for the seeded batteries.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.flow_network_solver.core.min_cost_flow import min_cost_max_flow
from src.flow_network_solver.core.network import Network, cut_value, min_cut
from src.flow_network_solver.utils.generators import random_network
from src.flow_network_solver.utils.validators import check_min_cost_feasible, validate_flow
from src.utils.random_utils import make_rng

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

COST_EDGES = [(0, 1, 4, 2), (0, 2, 2, 2), (1, 2, 2, 1), (1, 3, 3, 3), (2, 3, 5, 1)]

#: Plain Dijkstra settles t through node 2 at cost 3 and never sees the cost 1
#: path through node 1, whose negative edge comes last.
NEGATIVE_EDGES = [(0, 1, 1, 5), (1, 3, 1, -4), (0, 2, 1, 2), (2, 3, 1, 1)]


def reference_graph(net: Network) -> nx.DiGraph:
    """Build a DiGraph for networkx, giving each edge its own middle node.

    A DiGraph cannot hold parallel edges, and merging them would lose their
    different costs, so u -> v becomes u -> x -> v with the cost on the first
    half. This changes neither the max flow nor the minimum cost.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(net.n))
    for index, u, v, capacity, cost in net.forward_edges():
        middle = net.n + index // 2
        graph.add_edge(u, middle, capacity=capacity, weight=cost)
        graph.add_edge(middle, v, capacity=capacity, weight=0)
    return graph


def assert_matches_networkx(net: Network, s: int, t: int) -> None:
    """Solve ``net`` and compare the value and cost with networkx."""
    graph = reference_graph(net)
    result = min_cost_max_flow(net, s, t)
    assert result.value == nx.maximum_flow_value(graph, s, t)
    assert result.cost == nx.cost_of_flow(graph, nx.max_flow_min_cost(graph, s, t))
    assert check_min_cost_feasible(net, s, t)[0]
    ok, message = validate_flow(net, s, t, result.value)
    assert ok, message


def test_clrs_network_with_zero_costs():
    net = Network.from_edges(6, CLRS_EDGES)
    result = min_cost_max_flow(net, 0, 5)
    assert result.value == 23
    assert result.cost == 0
    assert min_cut(net, 0) == {0, 1, 2, 4}
    assert cut_value(net, 0) == 23
    assert validate_flow(net, 0, 5, 23) == (True, "all checks passed")


def test_cost_anchor_has_the_exact_optimal_flow():
    net = Network.from_edges(4, COST_EDGES)
    result = min_cost_max_flow(net, 0, 3)
    assert result.value == 6
    assert result.cost == 24
    assert result.flow == {(0, 1): 4, (0, 2): 2, (1, 2): 2, (1, 3): 2, (2, 3): 4}
    assert validate_flow(net, 0, 3, 6) == (True, "all checks passed")


def test_cost_anchor_takes_the_three_paths_in_cost_order():
    result = min_cost_max_flow(Network.from_edges(4, COST_EDGES), 0, 3)
    assert result.bottlenecks == [2, 2, 2]
    assert result.augmentations == 3
    assert result.iterations == 3
    assert result.residual_updates == 2 * (2 + 3 + 2)


@pytest.mark.parametrize("units, cost", [(1, 3), (2, 6), (3, 10), (4, 14), (5, 19), (6, 24)])
def test_every_amount_is_sent_at_its_minimum_cost(units, cost):
    net = Network.from_edges(4, COST_EDGES)
    result = min_cost_max_flow(net, 0, 3, max_flow=units)
    assert (result.value, result.cost) == (units, cost)
    assert validate_flow(net, 0, 3, units) == (True, "all checks passed")


def test_three_units_match_networkx_on_a_demand_version():
    net = Network.from_edges(4, COST_EDGES)
    result = min_cost_max_flow(net, 0, 3, max_flow=3)

    graph = nx.DiGraph()
    for u, v, capacity, cost in COST_EDGES:
        graph.add_edge(u, v, capacity=capacity, weight=cost)
    graph.add_node(0, demand=-3)
    graph.add_node(3, demand=3)

    assert result.value == 3
    assert result.cost == 10
    assert result.cost == nx.cost_of_flow(graph, nx.min_cost_flow(graph))


def test_max_flow_above_the_maximum_sends_the_maximum():
    result = min_cost_max_flow(Network.from_edges(4, COST_EDGES), 0, 3, max_flow=99)
    assert (result.value, result.cost) == (6, 24)


def test_max_flow_zero_sends_nothing():
    net = Network.from_edges(4, COST_EDGES)
    result = min_cost_max_flow(net, 0, 3, max_flow=0)
    assert (result.value, result.cost, result.flow, result.augmentations) == (0, 0, {}, 0)
    assert net.cap == net.orig_cap


def test_no_path_gives_an_empty_flow():
    result = min_cost_max_flow(Network.from_edges(3, [(0, 1, 5, 1), (2, 1, 5, 1)]), 0, 2)
    assert (result.value, result.cost, result.flow) == (0, 0, {})


def test_reset_gives_the_same_answer_again():
    net = Network.from_edges(4, COST_EDGES)
    first = min_cost_max_flow(net, 0, 3)
    net.reset()
    second = min_cost_max_flow(net, 0, 3)
    assert (second.value, second.cost, second.flow) == (first.value, first.cost, first.flow)


def test_resuming_reads_the_negative_reverse_edges_the_first_call_opened():
    net = Network.from_edges(4, COST_EDGES)
    first = min_cost_max_flow(net, 0, 3, max_flow=3)
    second = min_cost_max_flow(net, 0, 3)
    assert (first.value, first.cost) == (3, 10)
    assert (second.value, second.cost) == (3, 14)
    assert second.flow == {(0, 1): 4, (0, 2): 2, (1, 2): 2, (1, 3): 2, (2, 3): 4}


def test_resuming_from_a_flow_that_is_not_cheapest_raises():
    net = Network.from_edges(2, [(0, 1, 1, 5), (0, 1, 1, 1)])
    net.cap[0] -= 1
    net.cap[1] += 1
    with pytest.raises(ValueError, match="negative-cost cycle"):
        min_cost_max_flow(net, 0, 1)


def test_negative_edge_without_a_negative_cycle():
    net = Network.from_edges(4, NEGATIVE_EDGES)
    result = min_cost_max_flow(net, 0, 3, max_flow=1)
    assert (result.value, result.cost) == (1, 1)
    assert result.flow == {(0, 1): 1, (1, 3): 1}
    assert check_min_cost_feasible(net, 0, 3)[0]

    net.reset()
    result = min_cost_max_flow(net, 0, 3)
    assert (result.value, result.cost) == (2, 4)
    assert validate_flow(net, 0, 3, 2) == (True, "all checks passed")


def test_positive_cycle_with_a_negative_edge_is_not_rejected():
    edges = [(0, 1, 2, 1), (1, 2, 2, -2), (2, 1, 2, 3), (2, 3, 2, 1)]
    result = min_cost_max_flow(Network.from_edges(4, edges), 0, 3)
    assert (result.value, result.cost) == (2, 0)


@pytest.mark.parametrize("seed", range(10))
def test_negative_costs_on_random_dags_match_networkx(seed):
    rng = make_rng(seed)
    n = 8 + seed
    edges = []
    for _ in range(4 * n):
        u, v = sorted(rng.sample(range(n), 2))
        edges.append((u, v, rng.randint(1, 20), rng.randint(-10, 20)))
    assert_matches_networkx(Network.from_edges(n, edges), 0, n - 1)


def test_negative_cycle_on_the_way_to_t_raises():
    edges = [(0, 1, 1, 1), (1, 2, 1, -5), (2, 1, 1, 1), (2, 3, 1, 1)]
    with pytest.raises(ValueError, match="negative-cost cycle"):
        min_cost_max_flow(Network.from_edges(4, edges), 0, 3)


def test_negative_cycle_through_s_raises():
    edges = [(0, 1, 1, -1), (1, 0, 1, -1), (1, 2, 1, 1)]
    with pytest.raises(ValueError, match="negative-cost cycle"):
        min_cost_max_flow(Network.from_edges(3, edges), 0, 2)


def test_negative_cycle_that_cannot_reach_t_still_raises():
    edges = [(0, 1, 1, 1), (0, 2, 1, 1), (2, 3, 1, -2), (3, 2, 1, 1)]
    with pytest.raises(ValueError, match="negative-cost cycle"):
        min_cost_max_flow(Network.from_edges(4, edges), 0, 1)


def test_negative_cycle_raises_even_for_zero_units():
    edges = [(0, 1, 1, 1), (1, 2, 1, -5), (2, 1, 1, 1), (2, 3, 1, 1)]
    with pytest.raises(ValueError, match="negative-cost cycle"):
        min_cost_max_flow(Network.from_edges(4, edges), 0, 3, max_flow=0)


def test_negative_cycle_s_cannot_reach_is_ignored():
    edges = [(0, 1, 1, 1), (2, 3, 1, -2), (3, 2, 1, 1)]
    result = min_cost_max_flow(Network.from_edges(4, edges), 0, 1)
    assert (result.value, result.cost) == (1, 1)


def test_negative_cycle_with_no_residual_capacity_is_not_a_cycle():
    edges = [(0, 1, 1, 1), (1, 2, 0, -5), (2, 1, 1, 1), (1, 3, 1, -1)]
    result = min_cost_max_flow(Network.from_edges(4, edges), 0, 3)
    assert (result.value, result.cost) == (1, 0)


@pytest.mark.parametrize("s, t", [(0, 0), (-1, 1), (0, 2), (2, 0)])
def test_bad_terminals_raise(s, t):
    with pytest.raises(ValueError):
        min_cost_max_flow(Network(2), s, t)


def test_negative_max_flow_raises():
    with pytest.raises(ValueError):
        min_cost_max_flow(Network(2), 0, 1, max_flow=-1)


@pytest.mark.parametrize("max_flow", [1.5, True])
def test_non_int_max_flow_raises(max_flow):
    with pytest.raises(TypeError):
        min_cost_max_flow(Network(2), 0, 1, max_flow=max_flow)


@pytest.mark.parametrize("seed", range(20))
def test_random_cost_networks_match_networkx(seed):
    n = 6 + seed
    avg_degree = 4 if seed % 2 else n / 4
    net = random_network(n, avg_degree, seed, cost_range=(0, 20))
    assert_matches_networkx(net, 0, n - 1)
