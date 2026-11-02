"""Week 10 tests: the array network, cut readers and generators (spec 2.3, 9.4).

Every expected value is worked out by hand. The flow on the spec 9.1 network is
the textbook maximum flow, pushed edge by edge with :func:`push`, so these tests
do not depend on any solver.
"""

from __future__ import annotations

import pytest

from src.flow_network_solver.core.network import (
    FlowResult,
    Network,
    cut_edges,
    cut_value,
    min_cut,
)
from src.flow_network_solver.utils.generators import (
    bipartite_graph,
    capacity_distribution,
    dense,
    grid_image,
    random_network,
    sparse,
)

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

#: A maximum flow of value 23: forward edge index -> units.
CLRS_FLOW = {0: 12, 2: 11, 6: 12, 10: 11, 12: 7, 14: 19, 16: 4}


def push(net, edge, units):
    """Push ``units`` along ``edge`` by hand, updating both sides of the pair."""
    net.cap[edge] -= units
    net.cap[edge ^ 1] += units


def clrs_with_flow():
    """Return the spec 9.1 network holding its hand-built maximum flow."""
    net = Network.from_edges(6, CLRS_EDGES)
    for edge, units in CLRS_FLOW.items():
        push(net, edge, units)
    return net


# --- Network ---------------------------------------------------------------


def test_add_edge_creates_reverse_edge_with_zero_capacity():
    net = Network(3)
    edge = net.add_edge(0, 1, 7, cost=3)
    assert edge == 0
    assert net.to == [1, 0]
    assert net.cap == [7, 0]
    assert net.orig_cap == [7, 0]
    assert net.cost == [3, -3]
    assert net.adj == [[0], [1], []]
    assert net.m == 1


def test_forward_edges_are_even_and_tail_is_partner_head():
    net = Network(4)
    assert [net.add_edge(0, 1, 1), net.add_edge(1, 2, 2), net.add_edge(2, 3, 3)] == [0, 2, 4]
    assert net.to[2 ^ 1] == 1
    assert net.to[4 ^ 1] == 2
    assert [edge[:4] for edge in net.forward_edges()] == [(0, 0, 1, 1), (2, 1, 2, 2), (4, 2, 3, 3)]


def test_residual_sums_hold_after_flow_is_pushed():
    net = clrs_with_flow()
    for i, _, _, capacity, _ in net.forward_edges():
        assert net.cap[i] + net.cap[i ^ 1] == capacity


def test_flow_on_reads_the_pushed_flow():
    net = clrs_with_flow()
    assert net.flow_on(0) == 12
    assert net.flow_on(1) == -12
    assert net.flow_on(8) == 0


def test_flow_matrix_on_clrs_flow():
    assert clrs_with_flow().flow_matrix() == {
        (0, 1): 12,
        (0, 2): 11,
        (1, 3): 12,
        (2, 4): 11,
        (4, 3): 7,
        (3, 5): 19,
        (4, 5): 4,
    }


def test_get_residual_graph_keeps_positive_residuals_only():
    net = Network.from_edges(3, [(0, 1, 5), (1, 2, 3)])
    push(net, 0, 5)
    push(net, 2, 1)
    assert net.get_residual_graph() == [[], [(0, 5), (2, 2)], [(1, 1)]]


def test_reset_restores_capacities():
    net = clrs_with_flow()
    original = Network.from_edges(6, CLRS_EDGES)
    net.reset()
    assert net.cap == original.cap
    assert net.flow_matrix() == {}


def test_reset_keeps_cap_aliases_valid():
    net = Network.from_edges(2, [(0, 1, 5)])
    alias = net.cap
    push(net, 0, 2)
    net.reset()
    assert alias is net.cap
    assert alias == [5, 0]


def test_parallel_edges_add_up():
    net = Network.from_edges(2, [(0, 1, 3), (0, 1, 4)])
    assert net.m == 2
    assert net.get_residual_graph() == [[(1, 3), (1, 4)], []]
    push(net, 0, 3)
    push(net, 2, 2)
    assert net.flow_matrix() == {(0, 1): 5}
    assert net.get_residual_graph() == [[(1, 2)], [(0, 3), (0, 2)]]


def test_antiparallel_pair_keeps_two_separate_edge_pairs():
    net = Network.from_edges(2, [(0, 1, 5), (1, 0, 3)])
    assert net.to == [1, 0, 0, 1]
    assert net.adj == [[0, 3], [1, 2]]
    push(net, 0, 5)
    assert net.cap == [0, 5, 3, 0]
    assert net.get_residual_graph() == [[], [(0, 5), (0, 3)]]


def test_antiparallel_flows_are_netted():
    net = Network.from_edges(2, [(0, 1, 5), (1, 0, 3)])
    push(net, 0, 4)
    push(net, 2, 1)
    assert net.flow_matrix() == {(0, 1): 3}
    push(net, 2, 2)
    assert net.flow_matrix() == {(0, 1): 1}
    push(net, 0, -3)
    assert net.flow_matrix() == {(1, 0): 2}


def test_from_edges_without_costs():
    net = Network.from_edges(6, CLRS_EDGES)
    assert net.n == 6
    assert net.m == 9
    assert net.cap[::2] == [cap for _, _, cap in CLRS_EDGES]
    assert net.cost == [0] * 18


def test_from_edges_with_costs():
    edges = [(0, 1, 4, 2), (0, 2, 2, 2), (1, 2, 2, 1), (1, 3, 3, 3), (2, 3, 5, 1)]
    net = Network.from_edges(4, edges)
    assert list(net.forward_edges()) == [
        (0, 0, 1, 4, 2),
        (2, 0, 2, 2, 2),
        (4, 1, 2, 2, 1),
        (6, 1, 3, 3, 3),
        (8, 2, 3, 5, 1),
    ]
    assert net.cost[1::2] == [-2, -2, -1, -3, -1]


def test_from_edges_rejects_a_bad_tuple():
    with pytest.raises(ValueError):
        Network.from_edges(2, [(0, 1)])


@pytest.mark.parametrize(
    "args, error",
    [
        ((0, 3, 1), ValueError),
        ((-1, 1, 1), ValueError),
        ((1, 1, 1), ValueError),
        ((0, 1, -1), ValueError),
        ((0, 1, 1.5), TypeError),
        ((0, 1, True), TypeError),
        ((0, 1, "2"), TypeError),
    ],
)
def test_add_edge_rejects_bad_input_and_leaves_the_network_unchanged(args, error):
    net = Network(3)
    with pytest.raises(error):
        net.add_edge(*args)
    assert net.to == [] and net.cap == []


def test_flow_result_defaults_are_not_shared():
    a, b = FlowResult(1, {}), FlowResult(2, {})
    a.bottlenecks.append(1)
    a.convergence.append(1)
    assert b.bottlenecks == [] and b.convergence == []
    assert (a.iterations, a.residual_updates, a.time_s, a.steps) == (0, 0, 0.0, None)
    assert (a.pushes, a.relabels, a.global_relabels, a.cost, a.augmentations) == (0, 0, 0, None, 0)


# --- Cut readers -----------------------------------------------------------


def test_min_cut_on_the_clrs_anchor():
    net = clrs_with_flow()
    assert min_cut(net, 0) == {0, 1, 2, 4}
    assert sorted(cut_edges(net, 0)) == [(1, 3), (4, 3), (4, 5)]
    assert cut_value(net, 0) == 23


def test_cut_with_parallel_edges_counts_each_copy():
    net = Network.from_edges(2, [(0, 1, 3), (0, 1, 4)])
    push(net, 0, 3)
    push(net, 2, 4)
    assert cut_edges(net, 0) == [(0, 1), (0, 1)]
    assert cut_value(net, 0) == 7


def test_min_cut_rejects_an_out_of_range_node():
    with pytest.raises(ValueError):
        min_cut(Network(2), 2)


# --- Generators ------------------------------------------------------------


@pytest.mark.parametrize("build", [sparse, dense])
def test_generator_is_reproducible_with_a_seed(build):
    a, b = build(60, 11), build(60, 11)
    assert a.to == b.to and a.cap == b.cap and a.cost == b.cost


def test_generator_changes_with_the_seed():
    assert sparse(60, 1).to != sparse(60, 2).to


def test_generator_reproducible_with_costs_and_distributions():
    for kw in ({"cost_range": (1, 20)}, {"cap_dist": "exponential"}, {"cap_dist": "unit"}):
        a, b = sparse(40, 5, **kw), sparse(40, 5, **kw)
        assert a.to == b.to and a.cap == b.cap and a.cost == b.cost


@pytest.mark.parametrize(
    "n, avg_degree",
    [(2, 0.5), (3, 0.75), (4, 1), (5, 1.25), (10, 4), (100, 4), (1000, 4), (60, 15), (200, 50)],
)
def test_generator_has_exactly_the_specified_edge_count(n, avg_degree):
    net = random_network(n, avg_degree, 3)
    assert net.n == n
    assert net.m == round(avg_degree * n)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 10, 30, 60])
@pytest.mark.parametrize("build", [sparse, dense])
def test_generator_always_has_an_s_t_path(build, n):
    for seed in range(20):
        assert n - 1 in min_cut(build(n, seed), 0)


def test_generator_with_one_edge_is_still_an_s_t_path():
    net = random_network(50, 0.02, 1)
    assert net.m == 1
    assert list(net.forward_edges())[0][1:3] == (0, 49)


@pytest.mark.parametrize("n", [2, 3, 10, 60, 500])
def test_generator_never_loops_leaves_the_sink_or_enters_the_source(n):
    for _, u, v, _, _ in random_network(n, 6, 9).forward_edges():
        assert u != v
        assert u != n - 1
        assert v != 0


def test_generator_adds_hub_edges_at_both_terminals():
    n = 1000
    net = random_network(n, 4, 4)
    k = 50  # 0.05 * 1000
    heads = {v for _, u, v, _, _ in net.forward_edges() if u == 0}
    tails = {u for _, u, v, _, _ in net.forward_edges() if v == n - 1}
    assert len(heads - {n - 1}) >= k
    assert len(tails - {0}) >= k


def test_generator_terminal_fraction_scales_the_hub():
    n = 1000
    net = random_network(n, 4, 4, terminal_fraction=0.2)
    assert len({v for _, u, v, _, _ in net.forward_edges() if u == 0}) >= 200


def test_generator_capacities_and_costs_stay_in_range():
    net = random_network(200, 4, 8, cap_range=(5, 9), cost_range=(1, 20))
    assert all(5 <= cap <= 9 for cap in net.cap[::2])
    assert all(1 <= cost <= 20 for cost in net.cost[::2])
    assert set(net.cap[1::2]) == {0}
    assert set(random_network(50, 4, 8).cost) == {0}


def test_generator_named_capacity_distributions():
    assert set(sparse(100, 2, cap_dist="unit").cap[::2]) == {1}
    assert min(sparse(100, 2, cap_dist="exponential").cap[::2]) >= 1


@pytest.mark.parametrize(
    "args, kw",
    [
        ((1, 4, 0), {}),
        ((10, 0, 0), {}),
        ((10, 4, 0), {"cap_range": (0, 5)}),
        ((10, 4, 0), {"cap_range": (5, 1)}),
        ((10, 4, 0), {"cost_range": (5, 1)}),
        ((10, 4, 0), {"cap_dist": "zipf"}),
        ((10, 4, 0), {"terminal_fraction": 1.5}),
    ],
)
def test_generator_rejects_bad_arguments(args, kw):
    with pytest.raises(ValueError):
        random_network(*args, **kw)


def test_bipartite_graph_extremes_and_labels():
    left, right, edges = bipartite_graph(2, 3, 1.0, 1)
    assert left == ["L0", "L1"]
    assert right == ["R0", "R1", "R2"]
    assert len(edges) == 6
    assert bipartite_graph(4, 4, 0.0, 1)[2] == []


def test_bipartite_graph_is_reproducible_and_keeps_about_p_of_the_pairs():
    graph = bipartite_graph(50, 50, 0.3, 7)
    assert graph == bipartite_graph(50, 50, 0.3, 7)
    assert 500 < len(graph[2]) < 1000  # expected 750 of 2,500 pairs


def test_grid_image_is_a_bright_disc_on_a_dark_background():
    rows, cols = 40, 40
    image = grid_image(rows, cols, 3)
    assert len(image) == rows and all(len(row) == cols for row in image)
    assert all(isinstance(pixel, int) and 0 <= pixel <= 255 for row in image for pixel in row)
    inside = [image[r][c] for r in range(16, 24) for c in range(16, 24)]
    corners = [image[r][c] for r in range(5) for c in range(5)]
    assert 180 < sum(inside) / len(inside) < 220
    assert 30 < sum(corners) / len(corners) < 70
    assert image == grid_image(rows, cols, 3)


def test_capacity_distribution_values():
    assert capacity_distribution("unit", 5, 1) == [1] * 5
    uniform = capacity_distribution("uniform", 2000, 1)
    assert min(uniform) >= 1 and max(uniform) <= 100
    exponential = capacity_distribution("exponential", 20000, 1)
    assert min(exponential) >= 1
    assert 18 < sum(exponential) / len(exponential) < 22
    assert capacity_distribution("uniform", 50, 2) == capacity_distribution("uniform", 50, 2)


def test_capacity_distribution_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        capacity_distribution("zipf", 3, 1)
