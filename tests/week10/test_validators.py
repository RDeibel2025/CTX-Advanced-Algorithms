"""Week 10 tests: the validators on hand-built flows (spec 6, 9.4).

The valid flows are worked out by hand: the textbook maximum flow of value 23 on
the spec 9.1 network and the optimal flow of cost 24 on the spec 9.2 network.
Each broken flow changes one of them in a known way. No solver is involved.
"""

from __future__ import annotations

import pytest

from src.flow_network_solver.core.network import Network
from src.flow_network_solver.utils.generators import sparse
from src.flow_network_solver.utils.validators import (
    check_capacity,
    check_conservation,
    check_flow_value,
    check_min_cost_feasible,
    check_residual,
    validate_flow,
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

COST_EDGES = [(0, 1, 4, 2), (0, 2, 2, 2), (1, 2, 2, 1), (1, 3, 3, 3), (2, 3, 5, 1)]

#: Spec 9.2 optimal flow, cost 24: 0->1: 4, 0->2: 2, 1->2: 2, 1->3: 2, 2->3: 4.
COST_OPTIMAL = {0: 4, 2: 2, 4: 2, 6: 2, 8: 4}

#: Spec 9.2 alternative, also value 6, cost 25: 3 on 1->3 and 1 on 1->2.
COST_ALTERNATIVE = {0: 4, 2: 2, 4: 1, 6: 3, 8: 3}


def push(net, edge, units):
    """Push ``units`` along ``edge`` by hand, updating both sides of the pair."""
    net.cap[edge] -= units
    net.cap[edge ^ 1] += units


def build(n, edges, flow):
    """Return a network on ``edges`` holding the hand-built ``flow``."""
    net = Network.from_edges(n, edges)
    for edge, units in flow.items():
        push(net, edge, units)
    return net


def clrs_valid():
    """Return the spec 9.1 network with its maximum flow of value 23."""
    return build(6, CLRS_EDGES, CLRS_FLOW)


def test_every_check_passes_on_the_hand_built_valid_flow():
    net = clrs_valid()
    assert check_capacity(net)[0]
    assert check_conservation(net, 0, 5)[0]
    assert check_residual(net)[0]
    assert check_flow_value(net, 0, 5, 23)[0]
    assert check_min_cost_feasible(net, 0, 5)[0]
    assert validate_flow(net, 0, 5, 23) == (True, "all checks passed")


def test_validate_flow_passes_on_the_zero_flow():
    assert validate_flow(Network.from_edges(6, CLRS_EDGES), 0, 5, 0)[0]


def test_validators_return_a_bool_and_a_message():
    for ok, message in (check_capacity(clrs_valid()), validate_flow(clrs_valid(), 0, 5, 1)):
        assert isinstance(ok, bool)
        assert isinstance(message, str) and message


def test_capacity_exceeded_is_caught():
    net = clrs_valid()
    for edge in (0, 6, 14):  # one more unit along 0->1->3->5; 1->3 was already full
        push(net, edge, 1)
    ok, message = check_capacity(net)
    assert not ok
    assert "1->3" in message
    assert check_conservation(net, 0, 5)[0]
    assert check_residual(net)[0]
    assert check_flow_value(net, 0, 5, 24)[0]
    ok, message = validate_flow(net, 0, 5, 24)
    assert not ok
    assert "capacity" in message


def test_negative_flow_is_caught_as_a_capacity_violation():
    net = Network.from_edges(2, [(0, 1, 5)])
    push(net, 0, -1)
    assert not check_capacity(net)[0]


def test_conservation_broken_is_caught():
    net = clrs_valid()
    push(net, 12, -1)  # node 4 now keeps a unit and node 3 is a unit short
    ok, message = check_conservation(net, 0, 5)
    assert not ok
    assert "node 3" in message
    assert check_capacity(net)[0]
    assert check_residual(net)[0]
    ok, message = validate_flow(net, 0, 5, 23)
    assert not ok
    assert "conservation" in message


def test_residual_sum_broken_is_caught():
    net = clrs_valid()
    net.cap[3] += 1  # one side of a pair moved without the other
    ok, message = check_residual(net)
    assert not ok
    assert "0->2" in message
    assert not validate_flow(net, 0, 5, 23)[0]


@pytest.mark.parametrize("value", [22, 24, 0])
def test_wrong_flow_value_is_caught(value):
    net = clrs_valid()
    assert not check_flow_value(net, 0, 5, value)[0]
    assert not validate_flow(net, 0, 5, value)[0]


def test_flow_value_checks_both_ends():
    net = clrs_valid()
    push(net, 0, 1)  # one more unit leaves s but never reaches t
    ok, message = check_flow_value(net, 0, 5, 24)
    assert not ok
    assert "out of s 24" in message and "into t 23" in message


def test_validate_flow_reports_every_failure():
    net = clrs_valid()
    push(net, 0, 5)  # 0->1 goes to 17 of 16, and node 1 keeps 5 units
    ok, message = validate_flow(net, 0, 5, 23)
    assert not ok
    assert "capacity" in message
    assert "conservation" in message
    assert "flow value" in message


def test_negative_residual_cycle_is_caught():
    cycle = [(0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, -3)]
    ok, message = check_min_cost_feasible(Network.from_edges(3, cycle), 0, 2)
    assert not ok
    assert "negative" in message


def test_negative_cycle_away_from_s_and_t_is_caught():
    edges = [(0, 3, 5, 1), (1, 2, 1, 1), (2, 1, 1, -2)]
    assert not check_min_cost_feasible(Network.from_edges(4, edges), 0, 3)[0]


def test_saturating_a_negative_cycle_removes_it():
    net = Network.from_edges(3, [(0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, -3)])
    assert not check_min_cost_feasible(net, 0, 2)[0]
    for edge in (0, 2, 4):
        push(net, edge, 1)
    assert check_min_cost_feasible(net, 0, 2)[0]


def test_negative_edges_without_a_negative_cycle_pass():
    edges = [(0, 1, 2, 5), (1, 2, 2, -3), (2, 3, 2, -1)]
    assert check_min_cost_feasible(Network.from_edges(4, edges), 0, 3)[0]


def test_zero_cost_cycle_is_not_negative():
    edges = [(0, 1, 2, 1), (1, 2, 2, 1), (2, 0, 2, -2)]
    assert check_min_cost_feasible(Network.from_edges(3, edges), 0, 2)[0]


def test_optimal_cost_flow_passes_the_cost_check():
    net = build(4, COST_EDGES, COST_OPTIMAL)
    assert check_min_cost_feasible(net, 0, 3)[0]
    assert validate_flow(net, 0, 3, 6)[0]


def test_suboptimal_cost_flow_fails_only_the_cost_check():
    net = build(4, COST_EDGES, COST_ALTERNATIVE)
    assert check_capacity(net)[0]
    assert check_conservation(net, 0, 3)[0]
    assert check_residual(net)[0]
    assert check_flow_value(net, 0, 3, 6)[0]
    assert not check_min_cost_feasible(net, 0, 3)[0]  # 1->2->3->1 costs 1 + 1 - 3
    assert not validate_flow(net, 0, 3, 6)[0]


def test_cost_check_returns_at_once_when_every_cost_is_zero():
    ok, message = check_min_cost_feasible(sparse(2000, 1), 0, 1999)
    assert ok
    assert "all costs are 0" in message


@pytest.mark.parametrize("s, t", [(0, 0), (0, 6), (-1, 5)])
def test_bad_terminals_return_false(s, t):
    net = clrs_valid()
    assert not check_conservation(net, s, t)[0]
    assert not check_flow_value(net, s, t, 23)[0]
    assert not check_min_cost_feasible(net, s, t)[0]
    assert not validate_flow(net, s, t, 23)[0]
