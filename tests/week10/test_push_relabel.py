"""Week 10 tests: push-relabel on the spec 9.1 anchor and the 9.4 battery.

Every expected value is worked out by hand or comes from theory, so the tests do
not repeat the solver's output. Three small networks are traced by hand below.

Path 0 -> 1 -> 2 with capacities 5 and 3 (n = 3). Saturating 0 -> 1 puts 5 on
node 1, and the global relabel gives heights [3, 1, 0]:

=============== ============= ==============
Operation       Heights       Excess
=============== ============= ==============
init            [3, 0, 0]     [-5, 5, 0]
global_relabel  [3, 1, 0]     [-5, 5, 0]
push 1 -> 2, 3  [3, 1, 0]     [-5, 2, 3]
relabel 1       [3, 4, 0]     [-5, 2, 3]
push 1 -> 0, 2  [3, 4, 0]     [-3, 0, 3]
=============== ============= ==============

Gap chain 0 -> 1 -> 2 -> 3 -> 4 with capacities 10, 10, 1, 10 (n = 5). The global
relabel gives [5, 3, 2, 1, 0]. Node 1 pushes 10 to node 2, and node 2 pushes 1 to
node 3 and is stuck with 9. Its only residual edge leads back to node 1 at height
3, so it relabels to 4 and leaves height 2 empty. That is the gap: nodes 1 and 2
are above it and below n, so both jump to n + 1 = 6, giving [5, 6, 6, 1, 0].
Without the gap rule the heights would be [5, 3, 4, 1, 0] and the 9 units would
bounce between nodes 1 and 2 on the way up. After that, node 3 pushes 1 to t, node 2
relabels to 7 and pushes 9 to node 1, and node 1 pushes 9 to s. That is 5 pushes and
2 relabels, with value 1.

CLRS network (spec 9.1, n = 6). Saturating s puts 16 on node 1 and 13 on node 2.
Reverse BFS from t then gives distance 1 for nodes 3 and 4, and 2 for nodes 1 and 2,
so the first heights are [6, 2, 2, 1, 1, 0].
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.network import Network, cut_edges, cut_value, min_cut
from src.flow_network_solver.core.push_relabel import push_relabel
from src.flow_network_solver.utils.generators import dense, random_network, sparse
from src.flow_network_solver.utils.validators import validate_flow
from src.utils.random_utils import derive_seed, make_rng

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

PATH_EDGES = [(0, 1, 5), (1, 2, 3)]
GAP_EDGES = [(0, 1, 10), (1, 2, 10), (2, 3, 1), (3, 4, 10)]

#: Each operation of the path network, from the module docstring.
PATH_TRACE = [
    ("init", 0, [3, 0, 0], [-5, 5, 0], 0),
    ("global_relabel", 2, [3, 1, 0], [-5, 5, 0], 0),
    ("push", 1, [3, 1, 0], [-5, 2, 3], 3),
    ("relabel", 1, [3, 4, 0], [-5, 2, 3], 3),
    ("push", 1, [3, 4, 0], [-3, 0, 3], 3),
]

#: Each operation of the gap chain, from the module docstring.
GAP_TRACE = [
    ("init", 0, [5, 0, 0, 0, 0], [-10, 10, 0, 0, 0], 0),
    ("global_relabel", 4, [5, 3, 2, 1, 0], [-10, 10, 0, 0, 0], 0),
    ("push", 1, [5, 3, 2, 1, 0], [-10, 0, 10, 0, 0], 0),
    ("push", 2, [5, 3, 2, 1, 0], [-10, 0, 9, 1, 0], 0),
    ("relabel", 2, [5, 6, 6, 1, 0], [-10, 0, 9, 1, 0], 0),
    ("push", 3, [5, 6, 6, 1, 0], [-10, 0, 9, 0, 1], 1),
    ("relabel", 2, [5, 6, 7, 1, 0], [-10, 0, 9, 0, 1], 1),
    ("push", 2, [5, 6, 7, 1, 0], [-10, 9, 0, 0, 1], 1),
    ("push", 1, [5, 6, 7, 1, 0], [-1, 0, 0, 0, 1], 1),
]


def clrs() -> Network:
    """Build the spec 9.1 network."""
    return Network.from_edges(6, CLRS_EDGES)


def chain(n: int) -> Network:
    """Build a path of capacity 1000 with a bottleneck of 7 next to the sink."""
    edges = [(i, i + 1, 1000) for i in range(n - 1)]
    edges[n - 3] = (n - 3, n - 2, 7)
    return Network.from_edges(n, edges)


def as_tuples(steps: list) -> list:
    """Reduce recorded steps to the tuples used by the hand traces."""
    return [(s["op"], s["node"], s["heights"], s["excess"], s["value"]) for s in steps]


def operations_of(steps: list) -> list:
    """Return only the pushes and relabels, the steps that count as operations."""
    return [step for step in steps if step["op"] in ("push", "relabel")]


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


def add_antiparallel_edges(net: Network, seed: int) -> None:
    """Give up to three inner forward edges a reverse twin, so a pair is certain."""
    rng = make_rng(seed)
    inner = [(u, v) for _, u, v, _, _ in net.forward_edges() if u != 0 and v != net.n - 1]
    for u, v in inner[:3]:
        net.add_edge(v, u, rng.randint(1, 100))


def has_antiparallel_pair(net: Network) -> bool:
    """Return whether some u -> v and v -> u both exist."""
    pairs = {(u, v) for _, u, v, _, _ in net.forward_edges()}
    return any((v, u) in pairs for u, v in pairs)


def heights_are_valid(net: Network, heights: list) -> bool:
    """Return whether height[u] <= height[v] + 1 on every residual edge."""
    return all(
        heights[net.to[edge ^ 1]] <= heights[net.to[edge]] + 1
        for edge in range(len(net.to))
        if net.cap[edge] > 0
    )


def assert_gap_rule_holds(net: Network) -> None:
    """Replay every relabel from the heights before it, recounting by scanning.

    This checks the solver's lazy count and buckets against the plain rule, in both
    directions: every node between an emptied level and n jumps to n + 1, and
    nothing except the relabeled node moves when no level has emptied.
    """
    n = net.n
    steps = push_relabel(net, 0, n - 1, record=True).steps
    for previous, step in zip(steps, steps[1:]):
        if step["op"] != "relabel":
            continue
        u, before, after = step["node"], previous["heights"], step["heights"]
        old = before[u]
        others = [node for node in range(n) if node != u]
        emptied = old < n and all(before[node] != old for node in others)
        for node in others:
            lifted = emptied and old < before[node] < n
            assert after[node] == (n + 1 if lifted else before[node])
        assert after[u] > old


# --- Spec 9.1 anchor ---


def test_clrs_max_flow_is_23():
    assert push_relabel(clrs(), S, T).value == 23


def test_clrs_min_cut_source_side():
    net = clrs()
    push_relabel(net, S, T)
    assert min_cut(net, S) == {0, 1, 2, 4}


def test_clrs_cut_value_is_23():
    net = clrs()
    push_relabel(net, S, T)
    assert sorted(cut_edges(net, S)) == [(1, 3), (4, 3), (4, 5)]
    assert cut_value(net, S) == 23


def test_clrs_passes_validate_flow():
    net = clrs()
    result = push_relabel(net, S, T)
    assert validate_flow(net, S, T, result.value) == (True, "all checks passed")


def test_clrs_flow_leaves_s_and_enters_t_at_23():
    flow = push_relabel(clrs(), S, T).flow
    assert flow[(0, 1)] + flow[(0, 2)] == 23
    assert flow[(3, 5)] + flow[(4, 5)] == 23


# --- Counters ---


def test_clrs_pushes_and_relabels_are_positive():
    result = push_relabel(clrs(), S, T)
    assert result.pushes > 0
    assert result.relabels > 0


def test_iterations_are_pushes_plus_relabels():
    result = push_relabel(clrs(), S, T)
    assert result.iterations == result.pushes + result.relabels


def test_path_counters_match_the_hand_trace():
    result = push_relabel(Network.from_edges(3, PATH_EDGES), 0, 2)
    assert (result.value, result.pushes, result.relabels) == (3, 2, 1)
    assert result.flow == {(0, 1): 3, (1, 2): 3}


def test_residual_updates_count_two_per_push_and_per_saturated_edge():
    # One saturated edge and two pushes on the path.
    assert push_relabel(Network.from_edges(3, PATH_EDGES), 0, 2).residual_updates == 6
    # One saturated edge and five pushes on the gap chain.
    assert push_relabel(Network.from_edges(5, GAP_EDGES), 0, 4).residual_updates == 12


def test_a_single_edge_needs_no_operations():
    result = push_relabel(Network.from_edges(2, [(0, 1, 5)]), 0, 1)
    assert (result.value, result.pushes, result.relabels, result.iterations) == (5, 0, 0, 0)
    assert result.flow == {(0, 1): 5}
    assert result.global_relabels == 1


# --- Heuristics ---


def test_gap_lifts_everything_above_the_empty_height():
    steps = push_relabel(Network.from_edges(5, GAP_EDGES), 0, 4, record=True).steps
    first_relabel = next(step for step in steps if step["op"] == "relabel")
    assert first_relabel["heights"] == [5, 6, 6, 1, 0]


def test_gap_chain_counters_match_the_hand_trace():
    net = Network.from_edges(5, GAP_EDGES)
    result = push_relabel(net, 0, 4)
    assert (result.value, result.pushes, result.relabels) == (1, 5, 2)
    assert validate_flow(net, 0, 4, result.value)[0]


@pytest.mark.parametrize("cap_dist", ["uniform", "exponential", "unit"])
@pytest.mark.parametrize("i", range(15))
def test_gap_fires_exactly_when_a_level_empties(cap_dist, i):
    n = 60
    net = sparse(n, derive_seed(10, "gap_rule", cap_dist, i), cap_dist=cap_dist)
    assert_gap_rule_holds(net)


def test_a_level_emptied_by_one_gap_can_gap_again():
    # Random graphs rarely do this: found by search, this one sends a node back up
    # through a level that an earlier gap emptied, so a stale count there would
    # keep the next gap from firing while every flow value stayed correct.
    assert_gap_rule_holds(random_network(34, 2, derive_seed(5, "gapnet", 829), terminal_fraction=0.3))


def test_first_global_relabel_gives_exact_distances():
    steps = push_relabel(clrs(), S, T, record=True).steps
    assert steps[1]["op"] == "global_relabel"
    assert steps[1]["heights"] == [6, 2, 2, 1, 1, 0]


def test_global_relabel_runs_at_the_start_and_every_n_relabels():
    n = 300
    net = sparse(n, derive_seed(10, "global_relabel", n), terminal_fraction=0.3)
    result = push_relabel(net, 0, n - 1, record=True)
    assert result.global_relabels >= 2  # This instance must reach the periodic rule.
    assert result.steps[1]["op"] == "global_relabel"
    since_last = 0
    for step in result.steps[2:]:
        if step["op"] == "relabel":
            since_last += 1
        elif step["op"] == "global_relabel":
            assert since_last == n
            since_last = 0
    assert since_last < n


def test_excess_that_cannot_reach_t_walks_back_up_a_long_chain():
    n = 200
    net = chain(n)
    result = push_relabel(net, 0, n - 1)
    assert result.value == 7
    assert result.flow == {(i, i + 1): 7 for i in range(n - 1)}
    assert min_cut(net, 0) == set(range(n - 2))
    assert validate_flow(net, 0, n - 1, result.value)[0]


# --- Convergence profile ---


def test_path_convergence_matches_the_hand_trace():
    # n = 3 samples every operation: node 1 holds 5, then 2, then nothing.
    result = push_relabel(Network.from_edges(3, PATH_EDGES), 0, 2)
    assert result.convergence == [(0, 1, 5), (1, 1, 2), (2, 1, 2), (3, 0, 0)]


def test_convergence_starts_from_the_saturated_source():
    assert push_relabel(clrs(), S, T).convergence[0] == (0, 2, 29)


def test_convergence_ends_with_zero_active_nodes_and_zero_excess():
    result = push_relabel(clrs(), S, T)
    assert result.convergence[-1] == (result.iterations, 0, 0)


def test_convergence_samples_every_n_over_50_operations():
    n = 500
    result = push_relabel(sparse(n, derive_seed(10, "convergence", n)), 0, n - 1)
    interval = n // 50
    sampled = [operations for operations, _, _ in result.convergence]
    assert sampled[:-1] == list(range(0, result.iterations + 1, interval))[: len(sampled) - 1]
    assert sampled[-1] == result.iterations
    assert len(sampled) <= result.iterations // interval + 2


def test_total_excess_never_rises():
    # After the preflow, inner excess only ever leaves through s or t.
    n = 500
    result = push_relabel(sparse(n, derive_seed(10, "excess", n)), 0, n - 1)
    totals = [total for _, _, total in result.convergence]
    assert totals == sorted(totals, reverse=True)


@pytest.mark.parametrize("i", range(5))
def test_convergence_totals_match_a_recount_of_the_snapshots(i):
    n = 40
    result = push_relabel(sparse(n, derive_seed(10, "recount", i)), 0, n - 1, record=True)
    operations = operations_of(result.steps)
    assert operations
    inner = range(1, n - 1)
    for done, active, total in result.convergence[1:]:
        excess = operations[done - 1]["excess"]
        assert active == sum(1 for node in inner if excess[node] > 0)
        assert total == sum(excess[node] for node in inner)


# --- record=True ---


def test_steps_are_none_without_record():
    assert push_relabel(clrs(), S, T).steps is None


def test_record_follows_the_path_trace():
    steps = push_relabel(Network.from_edges(3, PATH_EDGES), 0, 2, record=True).steps
    assert as_tuples(steps) == PATH_TRACE


def test_record_follows_the_gap_trace():
    steps = push_relabel(Network.from_edges(5, GAP_EDGES), 0, 4, record=True).steps
    assert as_tuples(steps) == GAP_TRACE


def test_record_starts_with_the_saturated_source():
    first = push_relabel(clrs(), S, T, record=True).steps[0]
    assert first["op"] == "init"
    assert first["node"] == S
    assert first["heights"] == [6, 0, 0, 0, 0, 0]
    assert first["excess"] == [-29, 16, 13, 0, 0, 0]
    assert first["value"] == 0


def test_record_step_counts_match_the_counters():
    result = push_relabel(clrs(), S, T, record=True)
    ops = [step["op"] for step in result.steps]
    assert ops.count("init") == 1
    assert ops.count("push") == result.pushes
    assert ops.count("relabel") == result.relabels
    assert ops.count("global_relabel") == result.global_relabels


def test_record_snapshots_conserve_excess_and_end_on_the_value():
    result = push_relabel(clrs(), S, T, record=True)
    assert all(sum(step["excess"]) == 0 for step in result.steps)
    assert all(step["value"] == step["excess"][T] for step in result.steps)
    assert result.steps[-1]["value"] == 23
    assert result.steps[-1]["excess"] == [-23, 0, 0, 0, 0, 23]


def test_record_snapshots_are_independent_copies():
    steps = push_relabel(clrs(), S, T, record=True).steps
    assert steps[0]["heights"] != steps[-1]["heights"]
    assert steps[0]["excess"] != steps[-1]["excess"]


def test_heights_never_fall_and_the_terminals_never_move():
    n = 40
    steps = push_relabel(sparse(n, derive_seed(10, "heights", 0)), 0, n - 1, record=True).steps
    for before, after in zip(steps, steps[1:]):
        assert all(a <= b for a, b in zip(before["heights"], after["heights"]))
    assert all(step["heights"][0] == n and step["heights"][n - 1] == 0 for step in steps)


@pytest.mark.parametrize("i", range(15))
def test_every_operation_acts_on_a_node_holding_excess(i):
    # The in-queue flag keeps a node in the queue once. A duplicate entry would come
    # up after the node had drained, and push or relabel a node with nothing to move.
    n = 300
    net = sparse(n, derive_seed(10, "in_queue", i), terminal_fraction=0.3)
    steps = push_relabel(net, 0, n - 1, record=True).steps
    for previous, step in zip(steps, steps[1:]):
        if step["op"] in ("push", "relabel"):
            assert previous["excess"][step["node"]] > 0


def test_record_leaves_the_answer_unchanged():
    plain = push_relabel(clrs(), S, T)
    net = clrs()
    recorded = push_relabel(net, S, T, record=True)
    assert (recorded.value, recorded.flow) == (plain.value, plain.flow)
    assert (recorded.pushes, recorded.relabels) == (plain.pushes, plain.relabels)
    assert recorded.convergence == plain.convergence
    assert validate_flow(net, S, T, recorded.value)[0]


# --- Residual behaviour ---


def test_reverse_edge_cancels_earlier_flow():
    edges = [(0, 1, 1), (1, 2, 1), (2, 5, 1), (0, 3, 1), (3, 2, 1), (1, 4, 1), (4, 5, 1)]
    net = Network.from_edges(6, edges)
    result = push_relabel(net, 0, 5)
    assert result.value == 2
    assert validate_flow(net, 0, 5, result.value)[0]


def test_parallel_edges_add_up():
    result = push_relabel(Network.from_edges(2, [(0, 1, 3), (0, 1, 4)]), 0, 1)
    assert (result.value, result.flow) == (7, {(0, 1): 7})


def test_antiparallel_edges():
    edges = [(0, 1, 5), (1, 2, 3), (2, 1, 2), (1, 3, 1), (2, 3, 4)]
    net = Network.from_edges(4, edges)
    result = push_relabel(net, 0, 3)
    assert result.value == 4
    assert cut_value(net, 0) == 4
    assert validate_flow(net, 0, 3, result.value)[0]


def test_edge_into_the_source():
    # The only way out of node 1 is 1 unit to t, so the net flow out of s is 1.
    net = Network.from_edges(3, [(0, 1, 3), (1, 0, 2), (1, 2, 1)])
    result = push_relabel(net, 0, 2)
    assert (result.value, result.flow) == (1, {(0, 1): 1, (1, 2): 1})
    assert validate_flow(net, 0, 2, result.value)[0]


def test_edge_out_of_the_sink_stays_empty():
    net = Network.from_edges(3, [(0, 1, 4), (1, 2, 4), (2, 1, 3)])
    result = push_relabel(net, 0, 2)
    assert (result.value, result.flow) == (4, {(0, 1): 4, (1, 2): 4})
    assert validate_flow(net, 0, 2, result.value)[0]


def test_terminals_need_not_be_the_first_and_last_nodes():
    net = Network.from_edges(3, [(2, 1, 4), (1, 0, 3), (2, 0, 1)])
    result = push_relabel(net, 2, 0)
    assert result.value == 4
    assert validate_flow(net, 2, 0, result.value)[0]


def test_zero_capacity_edge_blocks_the_path():
    net = Network.from_edges(3, [(0, 1, 0), (1, 2, 5)])
    result = push_relabel(net, 0, 2)
    assert (result.value, result.flow, result.iterations) == (0, {}, 0)


def test_no_path_returns_all_excess_to_the_source():
    net = Network.from_edges(4, [(0, 1, 5), (2, 3, 5)])
    result = push_relabel(net, 0, 3)
    assert (result.value, result.flow) == (0, {})
    assert result.convergence[-1][1:] == (0, 0)
    assert min_cut(net, 0) == {0, 1}
    assert validate_flow(net, 0, 3, 0)[0]


def test_source_with_no_edges():
    result = push_relabel(Network(3), 0, 2)
    assert (result.value, result.flow, result.global_relabels) == (0, {}, 1)
    assert result.convergence == [(0, 0, 0)]


def test_reset_allows_a_second_identical_run():
    net = clrs()
    first = push_relabel(net, S, T)
    net.reset()
    second = push_relabel(net, S, T)
    assert (second.value, second.flow) == (first.value, first.flow)
    assert (second.pushes, second.relabels) == (first.pushes, first.relabels)


def test_second_run_without_reset_adds_nothing():
    net = clrs()
    push_relabel(net, S, T)
    result = push_relabel(net, S, T)
    assert result.value == 0
    assert validate_flow(net, S, T, 23)[0]


# --- Invalid terminals ---


def test_equal_terminals_raise():
    with pytest.raises(ValueError):
        push_relabel(clrs(), 2, 2)


@pytest.mark.parametrize("s, t", [(-1, 5), (6, 5), (0, 6), (0, -1)])
def test_out_of_range_terminals_raise(s, t):
    with pytest.raises(ValueError):
        push_relabel(clrs(), s, t)


# --- Spec 9.4 agreement battery ---


@pytest.mark.parametrize("i", range(50))
def test_battery_edmonds_karp_push_relabel_and_networkx_agree(i):
    seed = derive_seed(10, "push_relabel_battery", i)
    n = 5 + i * 55 // 49  # 5 up to 60
    net = dense(n, seed) if i % 2 else sparse(n, seed)
    add_antiparallel_edges(net, seed)
    assert has_antiparallel_pair(net)
    expected = networkx_value(net, 0, n - 1)

    pr = push_relabel(net, 0, n - 1)
    assert pr.value == expected
    assert validate_flow(net, 0, n - 1, pr.value)[0]
    assert cut_value(net, 0) == expected

    net.reset()
    ek = edmonds_karp(net, 0, n - 1)
    assert ek.value == expected
    assert validate_flow(net, 0, n - 1, ek.value)[0]


@pytest.mark.parametrize("cap_dist", ["uniform", "exponential", "unit"])
@pytest.mark.parametrize("i", range(5))
def test_capacity_distributions_agree_with_networkx(cap_dist, i):
    n = 60
    net = sparse(n, derive_seed(10, "push_relabel_caps", cap_dist, i), cap_dist=cap_dist)
    expected = networkx_value(net, 0, n - 1)
    result = push_relabel(net, 0, n - 1)
    assert result.value == expected
    assert validate_flow(net, 0, n - 1, result.value)[0]


@pytest.mark.parametrize("i", range(30))
def test_arbitrary_small_graphs_agree_with_networkx(i):
    # Inputs the generators never make: any s and t, edges into s, edges out of t,
    # direct s-t edges and zero capacities.
    rng = make_rng(derive_seed(10, "push_relabel_arbitrary", i))
    n = rng.randint(2, 12)
    s, t = rng.sample(range(n), 2)
    net = Network(n)
    for _ in range(rng.randint(0, 30)):
        u, v = rng.sample(range(n), 2)
        net.add_edge(u, v, rng.randint(0, 9))
    result = push_relabel(net, s, t)
    assert result.value == networkx_value(net, s, t)
    assert cut_value(net, s) == result.value
    assert validate_flow(net, s, t, result.value)[0]


@pytest.mark.parametrize("i", range(10))
def test_final_heights_are_valid_for_the_final_residual(i):
    # A valid labeling is the proof that no residual s-t path is left.
    n = 40
    net = sparse(n, derive_seed(10, "push_relabel_valid", i))
    steps = push_relabel(net, 0, n - 1, record=True).steps
    assert heights_are_valid(net, steps[-1]["heights"])


def test_two_thousand_nodes_match_edmonds_karp():
    n = 2000
    net = sparse(n, derive_seed(10, "push_relabel_large", n))
    pr = push_relabel(net, 0, n - 1)
    assert validate_flow(net, 0, n - 1, pr.value)[0]
    assert cut_value(net, 0) == pr.value
    net.reset()
    assert edmonds_karp(net, 0, n - 1).value == pr.value


def test_fifty_thousand_nodes_give_a_valid_maximum_flow():
    n = 50_000
    net = sparse(n, derive_seed(10, "push_relabel_large", n))
    result = push_relabel(net, 0, n - 1)
    assert validate_flow(net, 0, n - 1, result.value)[0]
    # Max-flow min-cut: a flow equal to the capacity of a cut is maximum.
    assert cut_value(net, 0) == result.value
    assert result.convergence[-1] == (result.iterations, 0, 0)
