#!/usr/bin/env python3
"""Runnable demonstration of the Week 10 network flow project.

    python examples/week10_demo.py

Five short sections, with no randomness, so the output is the same on every
run apart from the times:

1. **Maximum flow** - the CLRS network with Edmonds-Karp, push-relabel and
   min-cost flow, its minimum cut, the network facts the solvers rely on, and
   a hand-broken flow the validators catch.
2. **Minimum-cost flow** - the four-node anchor: the cheapest maximum flow, the
   same value at a higher cost from Edmonds-Karp, a capped flow of 3 units and
   a negative cycle.
3. **Matching** - a maximum matching with both solvers.
4. **Scheduling** - jobs on machines, then project staffing through the same
   network.
5. **Segmentation** - the 3 x 4 image anchor with both solvers.

Each section reads its results through one idea: every algorithm and
application here works on the same residual graph and differs only in what it
reads from it. The anchor values come from sections 9.1-9.4 of the Week 10
build spec, and every anchor printed is also asserted, so the script exits
non-zero rather than printing something untrue. The seeded batteries that
cross-check against networkx are in ``tests/week10/``. It finishes in a few
seconds.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.flow_network_solver.applications.image_seg import segment  # noqa: E402
from src.flow_network_solver.applications.matching import bipartite_matching  # noqa: E402
from src.flow_network_solver.applications.scheduling import schedule  # noqa: E402
from src.flow_network_solver.core.edmonds_karp import edmonds_karp  # noqa: E402
from src.flow_network_solver.core.min_cost_flow import min_cost_max_flow  # noqa: E402
from src.flow_network_solver.core.network import (  # noqa: E402
    Network,
    cut_edges,
    cut_value,
    min_cut,
)
from src.flow_network_solver.core.push_relabel import push_relabel  # noqa: E402
from src.flow_network_solver.utils.validators import (  # noqa: E402
    check_capacity,
    check_conservation,
    check_min_cost_feasible,
    check_residual,
    validate_flow,
)

WIDTH = 74

CLRS_EDGES = [(0, 1, 16), (0, 2, 13), (2, 1, 4), (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
              (3, 5, 20), (4, 5, 4)]
COST_EDGES = [(0, 1, 4, 2), (0, 2, 2, 2), (1, 2, 2, 1), (1, 3, 3, 3), (2, 3, 5, 1)]
UNIT_COST = {(u, v): c for u, v, _, c in COST_EDGES}


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<34} {value}")


def fmt_flow(flow: Dict[Tuple[int, int], int]) -> str:
    """A flow matrix as sorted ``u->v: units`` entries."""
    return ", ".join(f"{u}->{v}: {units}" for (u, v), units in sorted(flow.items()))


def flow_cost(flow: Dict[Tuple[int, int], int]) -> int:
    """The total cost of a flow on the section 2 network, summed from its edges."""
    return sum(units * UNIT_COST[edge] for edge, units in flow.items())


def is_matching(pairs: Sequence[Tuple[Any, Any]], edges: Sequence[Tuple[Any, Any]]) -> bool:
    """True when every pair is an input edge and no node appears twice."""
    lefts = [left for left, _ in pairs]
    rights = [right for _, right in pairs]
    return set(pairs) <= set(edges) and len(set(lefts)) == len(lefts) \
        and len(set(rights)) == len(rights)


def respects_limits(assignment: Dict[str, str], allowed: Dict[str, List[str]],
                    capacities: Dict[str, int]) -> bool:
    """True when every job is on an allowed machine and no machine is over capacity."""
    load = Counter(assignment.values())
    return all(machine in allowed[job] for job, machine in assignment.items()) \
        and all(load[machine] <= capacities[machine] for machine in capacities)


def section_max_flow() -> None:
    banner(1, "Maximum flow: the CLRS network with three solvers")
    show("edges (u, v, cap)", CLRS_EDGES[:5])
    show("", CLRS_EDGES[5:])
    show("source, sink", "s = 0, t = 5")

    # Spec 9.1 anchors, hand-checked. The cut around {0, 1, 2, 4} crosses
    # 1->3 (12), 4->3 (7) and 4->5 (4), capacity 23, and the flow below has
    # value 23, so both are optimal. The reachable set is the same for every
    # maximum flow, so all three solvers must report it.
    net = Network.from_edges(6, CLRS_EDGES)
    print()
    print(f"  {'solver':<15} {'value':>5} {'iterations':>10} {'min cut':>7} {'time (s)':>10}  "
          "source side")
    for name, solve in (("Edmonds-Karp", edmonds_karp), ("push-relabel", push_relabel),
                        ("min-cost flow", min_cost_max_flow)):
        net.reset()
        assert net.cap == net.orig_cap
        result = solve(net, 0, 5)
        ok, message = validate_flow(net, 0, 5, result.value)
        side = min_cut(net, 0)
        assert result.value == 23 and ok, message
        assert side == {0, 1, 2, 4} and cut_value(net, 0) == 23
        assert cut_edges(net, 0) == [(1, 3), (4, 3), (4, 5)]
        print(f"  {name:<15} {result.value:>5} {result.iterations:>10} {cut_value(net, 0):>7} "
              f"{result.time_s:>10.6f}  {sorted(side)}")
        if solve is min_cost_max_flow:
            assert result.cost == 0  # every cost in this network is 0
        if solve is push_relabel:
            assert result.iterations == result.pushes + result.relabels

    print()
    print("  What Edmonds-Karp reads: the shortest s-t path in the residual graph.")
    net.reset()
    result = edmonds_karp(net, 0, 5, record=True)
    # Hand-checked BFS: 0-1-3-5 pushes min(16, 12, 20) = 12; 1->3 is then full,
    # so 0-2-4-5 pushes min(13, 14, 4) = 4; 4->5 is then full, so 0-2-4-3-5
    # pushes min(9, 10, 7, 8) = 7. Then 1->3, 4->3 and 4->5 are all saturated
    # and the BFS stops at {0, 1, 2, 4}: 12 + 4 + 7 = 23.
    assert [step["path"] for step in result.steps] == [[0, 1, 3, 5], [0, 2, 4, 5],
                                                       [0, 2, 4, 3, 5]]
    assert result.bottlenecks == [12, 4, 7] and result.iterations == 3
    for step in result.steps:
        path = "-".join(map(str, step["path"]))
        show(f"  path {path}", f"bottleneck {step['bottleneck']}, value {step['value']}")
    show("flow", fmt_flow(result.flow))
    show("cut edges", cut_edges(net, 0))

    print()
    print("  What the validators read: the residual capacities and nothing else.")
    # A pair of arrays per edge: the reverse starts empty and parallel edges add.
    pair = Network(2)
    edge = pair.add_edge(0, 1, 3)
    assert pair.cap[edge ^ 1] == 0 and pair.orig_cap[edge ^ 1] == 0
    pair.add_edge(0, 1, 4)
    assert edmonds_karp(pair, 0, 1).value == 7 and check_residual(pair)[0]
    show("parallel edges 3 + 4, max flow", 7)
    # After Edmonds-Karp, 0->1 carries 12 of 16. Forcing 5 more onto it by hand
    # overfills it (17 of 16) and leaves node 1 with 5 units it never passes on.
    assert net.cap[0] == 4
    net.cap[0] -= 5
    net.cap[1] += 5
    capacity_ok, capacity_message = check_capacity(net)
    conservation_ok, conservation_message = check_conservation(net, 0, 5)
    assert not capacity_ok and not conservation_ok and not validate_flow(net, 0, 5, 23)[0]
    show("hand-broken flow, capacity", capacity_message)
    show("hand-broken flow, conservation", conservation_message)


def section_min_cost() -> None:
    banner(2, "Minimum-cost flow: the same graph read with costs")
    show("edges (u, v, cap, cost)", COST_EDGES)
    net = Network.from_edges(4, COST_EDGES)
    result = min_cost_max_flow(net, 0, 3)
    # Spec 9.2 anchors, hand-checked. The s-t paths cost 3 (0-2-3), 4 (0-1-2-3)
    # and 5 (0-1-3), and successive shortest paths take them in that order, 2
    # units each: 2 x 3 + 2 x 4 + 2 x 5 = 24.
    optimal = {(0, 1): 4, (0, 2): 2, (1, 2): 2, (1, 3): 2, (2, 3): 4}
    assert result.value == 6 and result.cost == 24 and result.flow == optimal
    assert flow_cost(optimal) == 24
    assert result.augmentations == 3 and result.bottlenecks == [2, 2, 2]
    ok, message = validate_flow(net, 0, 3, result.value)
    assert ok, message
    show("min-cost max-flow value, cost", f"{result.value}, {result.cost}")
    show("flow", fmt_flow(result.flow))
    show("augmentations, units per path", f"{result.augmentations}, {result.bottlenecks}")
    show("validate_flow", message)

    print()
    # Hand-checked BFS: 0-1-3 pushes 3, 0-2-3 pushes 2, 0-1-2-3 pushes 1. Same
    # value, but the residual cycle 1->2->3->1 costs 1 + 1 - 3 = -1, and pushing
    # around it gives the optimum above.
    net.reset()
    ek = edmonds_karp(net, 0, 3)
    alternative = {(0, 1): 4, (0, 2): 2, (1, 2): 1, (1, 3): 3, (2, 3): 3}
    assert ek.value == 6 and ek.flow == alternative and flow_cost(ek.flow) == 25
    feasible, feasible_message = check_min_cost_feasible(net, 0, 3)
    assert not feasible
    show("Edmonds-Karp, same value", f"{ek.value}, at cost {flow_cost(ek.flow)}")
    show("flow", fmt_flow(ek.flow))
    show("check_min_cost_feasible", feasible_message)

    print()
    # Hand-checked: 2 units at 3 on 0-2-3 and 1 unit at 4 on 0-1-2-3 cost 10, and
    # no path is cheaper than those two.
    net.reset()
    capped = min_cost_max_flow(net, 0, 3, max_flow=3)
    assert capped.value == 3 and capped.cost == 10 and validate_flow(net, 0, 3, 3)[0]
    show("max_flow=3, value and cost", f"{capped.value}, {capped.cost}")
    # 1->2->1 costs -5 + 1 = -4, and node 1 is reachable from s.
    cycle = Network.from_edges(4, [(0, 1, 1, 1), (1, 2, 1, -5), (2, 1, 1, 1), (2, 3, 1, 1)])
    try:
        min_cost_max_flow(cycle, 0, 3)
    except ValueError as error:
        show("negative cycle", f"ValueError: {error}")
    else:
        raise AssertionError("a negative cycle must raise ValueError")


def section_matching() -> None:
    banner(3, "Matching: reading the saturated edges")
    left, right = ["a", "b", "c", "d"], [1, 2, 3, 4]
    edges = [("a", 1), ("a", 2), ("b", 1), ("c", 2), ("c", 3), ("d", 3)]
    show("left, right", f"{left}, {right}")
    show("edges", ", ".join(f"{l}-{r}" for l, r in edges))
    # Spec 9.3 anchor, hand-checked: right node 4 has no edges, so at most 3
    # pairs exist, and a-2, b-1, c-3 is one such matching.
    for solver, name in (("ek", "Edmonds-Karp"), ("pr", "push-relabel")):
        pairs = bipartite_matching(left, right, edges, solver=solver)
        assert len(pairs) == 3 and is_matching(pairs, edges)
        show(f"{name}, size 3", ", ".join(f"{l}-{r}" for l, r in pairs))
    print("  The pairs are the unit left-right edges the flow saturated. Right node 4")
    print("  has no edges, so 3 is the most possible.")


def section_scheduling() -> None:
    banner(4, "Scheduling: reading the job-machine edges that carry flow")
    jobs, machines = ["J1", "J2", "J3", "J4"], ["M1", "M2"]
    allowed = {"J1": ["M1"], "J2": ["M1", "M2"], "J3": ["M1"], "J4": ["M2"]}
    capacities = {"M1": 2, "M2": 1}
    show("allowed", allowed)
    show("capacities", capacities)
    # Spec 9.3 anchor, hand-checked: three slots for four jobs, so at most 3 are
    # assigned, and J1, J2 on M1 with J4 on M2 achieves that.
    assignment, unassigned = schedule(jobs, machines, allowed, capacities)
    assert len(assignment) == 3 and len(unassigned) == 1
    assert set(assignment) | set(unassigned) == set(jobs)
    assert respects_limits(assignment, allowed, capacities)
    show("assignment", assignment)
    show("unassigned", unassigned)

    print()
    people = ["Ana", "Ben", "Cho", "Dev"]
    projects = ["Web", "Data"]
    skills = {"Ana": ["Web"], "Ben": ["Web", "Data"], "Cho": ["Data"], "Dev": ["Data"]}
    head_count = {"Web": 2, "Data": 1}
    print("  Project staffing is the same network: people are jobs, projects are")
    print("  machines and head counts are capacities.")
    show("skills", skills)
    show("head count", head_count)
    # Three seats for four people, and Ana, Ben on Web with Cho on Data fills them.
    staffing, free = schedule(people, projects, skills, head_count)
    assert len(staffing) == 3 and len(free) == 1
    assert respects_limits(staffing, skills, head_count)
    show("staffing", staffing)
    show("unstaffed", free)


def section_segmentation() -> None:
    banner(5, "Segmentation: reading the pixels still reachable from s")
    image = [[200, 210, 40, 30], [190, 205, 35, 20], [195, 60, 25, 15]]
    for row in image:
        show("image" if row is image[0] else "", row)
    # Spec 9.3 anchor, hand-checked with lam = 50, sigma = 30. The five bright
    # pixels pay 255 - I = 275, the seven dark ones pay I = 225, and the split
    # neighbour pairs add 50 exp(-d^2 / 1800): 0.0020 for 195-60, 0.0004 for
    # 205-60 and under 0.00001 each for 210-40 and 205-35. Total 500.0024.
    # Capacities are scaled by 1000 and rounded, which drops every weight below
    # 0.0005, so the value is 500.002: inside the spec's 1.0 and its 0.01.
    expected = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
    for solver, name in (("pr", "push-relabel"), ("ek", "Edmonds-Karp")):
        mask, cut = segment(image, lam=50, sigma=30, solver=solver)
        assert mask == expected and abs(cut - 500.0024) <= 0.01
        show(f"{name}, cut value", f"{cut:.3f}")
    for row in expected:
        show("mask" if row is expected[0] else "", row)
    print("  The mask follows the jump in intensity: a split pair across it costs")
    print("  almost nothing, while a split pair of similar pixels costs about 50.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 10 demonstration")
    print("Network flows - Robert Deibel")
    print("=" * WIDTH)
    print("  One residual graph, read different ways. Every anchor below is asserted.")
    section_max_flow()
    section_min_cost()
    section_matching()
    section_scheduling()
    section_segmentation()
    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
