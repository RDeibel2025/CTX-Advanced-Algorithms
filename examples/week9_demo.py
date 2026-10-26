#!/usr/bin/env python3
"""Runnable demonstration of the Week 9 approximation algorithms.

    python examples/week9_demo.py

Five short sections, every random draw seeded so the output is the same on
every run:

1. **Vertex Cover** - the greedy matching 2-approximation, LP rounding, the
   LP value and the exact minimum on the triangle K3 and the star with
   centre 0 and leaves 1-5.
2. **The Set Cover greedy trap** - the integer program written out, its LP
   relaxation (value and x), Chvatal's greedy, deterministic rounding at
   x_S >= 1/f, randomized rounding over five seeds and the exact optimum.
3. **Facility location** - greedy against the MILP optimum on the
   five-client, three-facility anchor.
4. **Metric TSP** - MST doubling on the unit square and on the six points
   x = 0, 1, -2, 4, -8, 16: MST weight, tour, cost and the Held-Karp optimum.
5. **Max-Cut** - the random cut, local search and the exact maximum on K4,
   the 5-cycle and K3,3, next to |E|.

Each section reads its results through one idea: every guarantee is a
comparison with a lower bound that takes polynomial time to compute (the
matching, the LP value, the MST weight, and |E| as an upper bound for
Max-Cut). The anchor values come from sections 10.1-10.5 of the Week 9
build spec, and every claim printed is also asserted, so the script exits
non-zero rather than printing something untrue. It finishes in a few
seconds.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
from typing import Any, List, Sequence, Set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.approximation.facility_location_greedy import (  # noqa: E402
    facility_location_exact,
    facility_location_greedy,
    total_cost,
)
from src.approximation.maxcut_randomized import (  # noqa: E402
    cut_value,
    maxcut_exact,
    maxcut_local_search,
    maxcut_random,
)
from src.approximation.set_cover_lpround import (  # noqa: E402
    frequency,
    set_cover_deterministic_rounding,
    set_cover_exact,
    set_cover_greedy,
    set_cover_ip,
    set_cover_lp,
    set_cover_randomized_rounding,
)
from src.approximation.tsp_metric_approx import (  # noqa: E402
    is_metric,
    mst_weight_and_edges,
    tsp_mst_doubling,
)
from src.approximation.vertex_cover_2approx import (  # noqa: E402
    approximation_ratio,
    matching_lower_bound,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_lp,
    vertex_cover_lp_rounding,
)
from src.complexity.traveling_salesman_approx import (  # noqa: E402
    is_valid_tour,
    tour_cost,
    tsp_nearest_neighbor,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask  # noqa: E402
from src.utils.graph_utils import is_vertex_cover  # noqa: E402

WIDTH = 74
TOL = 1e-7


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<34} {value}")


def close(a: float, b: float) -> bool:
    """True when two solver values agree to within :data:`TOL`."""
    return abs(a - b) <= TOL


def harmonic(k: int) -> float:
    """H(k) = 1 + 1/2 + ... + 1/k, Chvatal's greedy bound for sets of size k."""
    return sum(1.0 / i for i in range(1, k + 1))


def covers(universe: Set[int], subsets: Sequence[Set[int]], chosen: Sequence[int]) -> bool:
    """True when the chosen subsets cover the whole universe."""
    covered: Set[int] = set()
    for index in chosen:
        covered |= subsets[index]
    return covered >= universe


def fmt(value: float) -> str:
    """A float printed without trailing zeros: 2.0 -> 2, 1.5 -> 1.5."""
    return f"{value:g}"


def section_vertex_cover() -> None:
    banner(1, "Vertex Cover: greedy matching, LP rounding and the exact minimum")
    triangle = ([0, 1, 2], [(0, 1), (0, 2), (1, 2)])
    star = (list(range(6)), [(0, leaf) for leaf in range(1, 6)])
    # Spec 10.1 anchors, hand-checked. Triangle: x = 1/2 everywhere meets
    # every edge constraint with equality, and the three constraints add up
    # to 2(x0 + x1 + x2) >= 3, so LP = 1.5; rounding keeps all three
    # vertices; any two vertices are a minimum cover; greedy picks (0, 1)
    # first, which already covers (0, 2) and (1, 2). Star: x0 = 1 covers
    # every edge at LP cost 1, and the centre alone is the minimum cover;
    # greedy picks (0, 1) and returns both ends.
    anchors = {
        "triangle K3": (1.5, {0: 0.5, 1: 0.5, 2: 0.5}, {0, 1, 2}, {0, 1}, 2, 1),
        "star, centre 0, leaves 1-5": (1.0, {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
                                       {0}, {0, 1}, 1, 1),
    }
    print(f"  {'graph':<28} {'|M|':>3} {'LP':>5} {'exact':>5} {'2-approx':>9} {'LP round':>9}")
    rows = []
    for name, (vertices, edges) in (("triangle K3", triangle),
                                    ("star, centre 0, leaves 1-5", star)):
        lp_value, x = vertex_cover_lp(vertices, edges)
        rounded, rounded_lp = vertex_cover_lp_rounding(vertices, edges)
        approx = vertex_cover_2approx(vertices, edges)
        exact = vertex_cover_exact(vertices, edges)
        matched = matching_lower_bound(vertices, edges)
        a_lp, a_x, a_round, a_approx, a_exact, a_matched = anchors[name]
        assert close(lp_value, a_lp) and close(rounded_lp, a_lp)
        assert set(x) == set(a_x) and all(close(x[v], a_x[v]) for v in a_x)
        assert rounded == a_round and approx == a_approx
        assert len(exact) == a_exact and matched == a_matched
        assert all(is_vertex_cover(edges, c) for c in (rounded, approx, exact))
        # The two lower bounds sit below OPT; both algorithms within 2 x bound.
        assert matched <= len(exact) and lp_value <= len(exact) + TOL
        assert len(approx) == 2 * matched and len(rounded) <= 2 * lp_value + TOL
        print(f"  {name:<28} {matched:>3} {fmt(lp_value):>5} {len(exact):>5} "
              f"{len(approx):>9} {len(rounded):>9}")
        rows.append((name, x, rounded, approx, exact, lp_value, matched))

    print()
    for name, x, rounded, approx, exact, lp_value, matched in rows:
        print(f"  {name}:")
        show("  LP solution x", ", ".join(f"x{v} = {fmt(x[v])}" for v in sorted(x)))
        show("  LP rounding (x_v >= 1/2)", f"{sorted(rounded)}")
        show("  2-approximation", f"{sorted(approx)}")
        show("  exact minimum", f"{sorted(exact)}")
        to_opt = approximation_ratio(len(rounded), len(exact))
        to_lp = approximation_ratio(len(rounded), lp_value)
        greedy_to_opt = approximation_ratio(len(approx), len(exact))
        greedy_to_m = approximation_ratio(len(approx), matched)
        show("  LP rounding ratio to OPT, to LP", f"{to_opt:.1f}, {to_lp:.1f}")
        show("  2-approx ratio to OPT, to |M|", f"{greedy_to_opt:.1f}, {greedy_to_m:.1f}")
        assert to_lp <= 2 + TOL and greedy_to_m == 2.0
    tri_rounded, tri_lp = rows[0][2], rows[0][5]
    assert approximation_ratio(len(tri_rounded), 2) == 1.5
    assert approximation_ratio(len(tri_rounded), tri_lp) == 2.0
    print("  The ratio to the lower bound (|M| or LP) is at most 2 on every graph,")
    print("  and it needs no exact solver: it certifies the ratio to OPT from above.")
    print("  The triangle shows rounding at its worst against the LP (3 / 1.5 = 2),")
    print("  the star shows the matching algorithm at its worst against OPT (2 / 1).")


def section_set_cover() -> None:
    banner(2, "Set Cover: the greedy trap, LP relaxation and two roundings")
    universe = set(range(1, 15))
    subsets = [{1, 3, 5, 7, 9, 11, 13}, {2, 4, 6, 8, 10, 12, 14}, {1, 2}, {3, 4, 5, 6},
               {7, 8, 9, 10, 11, 12, 13, 14}]
    m = len(subsets)
    show("universe", "1..14, unit costs")
    for index, subset in enumerate(subsets):
        show(f"S{index}", sorted(subset))

    print()
    ip = set_cover_ip(universe, subsets)
    elements = sorted(universe)
    # The IP written out by hand from the subsets: one covering row per
    # element, -1 in column S when S contains it, so -sum x_S <= -1.
    assert list(ip["c"]) == [1.0] * m
    assert [list(row) for row in ip["A_ub"]] == [
        [-1.0 if e in subset else 0.0 for subset in subsets] for e in elements]
    assert list(ip["b_ub"]) == [-1.0] * len(elements)
    assert list(ip["integrality"]) == [1] * m
    bounds = ip["bounds"]
    assert tuple(bounds) == (0, 1) or all(tuple(b) == (0, 1) for b in bounds)
    print("  The integer program from set_cover_ip (linprog convention):")
    print(f"    minimize    {' + '.join(f'{fmt(c)} x{i}' for i, c in enumerate(ip['c']))}")
    print("    subject to  (one row per element, -sum x_S <= -1 in A_ub, b_ub)")
    for e, row in zip(elements, ip["A_ub"]):
        terms = " + ".join(f"x{i}" for i, a in enumerate(row) if a < 0)
        print(f"      e = {e:>2}:  {terms} >= 1")
    print(f"    bounds {bounds}, integrality {list(ip['integrality'])}: x_S in {{0, 1}}")

    print()
    lp_value, x = set_cover_lp(universe, subsets)
    f = frequency(universe, subsets)
    # Hand-checked: every element lies in exactly one of S0, S1 and exactly
    # one of S2, S3, S4, so f = 2. Dual y_e = 1/2 on elements 1-4 fits every
    # set (S0 holds 1 and 3, S1 holds 2 and 4, S2 holds 1 and 2, S3 holds 3
    # and 4, S4 none) and sums to 2, so LP >= 2; x = (1, 1, 0, 0, 0) costs
    # 2, so LP = 2. Complementary slackness forces x4 = 0 (S4's dual row is
    # slack), then elements 7-14 force x0 = x1 = 1: the LP optimum is that
    # integral x and nothing else.
    assert close(lp_value, 2.0) and f == 2
    assert all(close(xi, ai) for xi, ai in zip(x, [1.0, 1.0, 0.0, 0.0, 0.0]))
    show("LP relaxation value", fmt(lp_value))
    show("LP solution x", ", ".join(f"x{i} = {fmt(round(v, 9))}" for i, v in enumerate(x)))
    show("max frequency f", f)

    print()
    greedy, greedy_cost = set_cover_greedy(universe, subsets)
    # Hand-checked: S4 covers 8 new elements; then S3 covers 4 of {1..6}
    # (S0 and S1 cover 3 each); then S2 covers the last two, {1, 2}.
    assert greedy == [4, 3, 2] and greedy_cost == 3.0
    covered: Set[int] = set()
    gains: List[int] = []
    for index in greedy:
        gains.append(len(subsets[index] - covered))
        covered |= subsets[index]
    assert gains == [8, 4, 2] and covers(universe, subsets, greedy)
    show("greedy (Chvatal), pick order", f"{greedy}, cost {fmt(greedy_cost)}")
    show("", f"new elements per pick {gains}")

    det, det_cost, det_lp = set_cover_deterministic_rounding(universe, subsets)
    assert det == [0, 1] and det_cost == 2.0 and close(det_lp, 2.0)
    assert covers(universe, subsets, det)
    show("deterministic rounding, x >= 1/f", f"{det}, cost {fmt(det_cost)} (<= f x LP = 4)")

    rounds = math.ceil(2 * math.log(len(universe))) + 1
    assert rounds == 7
    print(f"  randomized rounding, {rounds} rounds (ceil(2 ln 14) + 1):")
    for seed in range(5):
        chosen, cost, r_lp, fixups = set_cover_randomized_rounding(
            universe, subsets, seed=seed, return_fixups=True)
        # x is 0/1, so every round picks S0 and S1 and nothing else.
        assert chosen == [0, 1] and cost == 2.0 and fixups == 0 and close(r_lp, 2.0)
        assert covers(universe, subsets, chosen)
        show(f"  seed {seed}", f"{chosen}, cost {fmt(cost)}, {fixups} fix-ups")

    exact, exact_cost = set_cover_exact(universe, subsets)
    assert exact == [0, 1] and exact_cost == 2.0 and covers(universe, subsets, exact)
    show("exact optimum (MILP)", f"{exact}, cost {fmt(exact_cost)}")

    largest = max(len(s) for s in subsets)
    bound = harmonic(largest)
    assert largest == 8 and greedy_cost <= bound * exact_cost
    assert lp_value <= exact_cost + TOL and det_cost <= f * lp_value + TOL
    show("greedy ratio to OPT", f"{greedy_cost / exact_cost:.2f} (bound H(8) = {bound:.3f})")
    print("  Greedy grabs the biggest set first and pays 3 against 2. The LP value 2")
    print("  certifies both roundings here without solving the IP. The LP optimum is")
    print("  integral, so randomized rounding returns it for every seed; its variance")
    print("  shows on the random instances in the benchmark.")


def section_facility_location() -> None:
    banner(3, "Facility location: greedy against the MILP optimum")
    open_costs = [4, 4, 4]
    dist = [[1, 5, 6], [2, 4, 6], [6, 1, 5], [6, 2, 4], [5, 6, 1]]
    show("opening costs", open_costs)
    show("dist[client][facility]", dist)
    # Hand-checked (spec 10.5): {0} = 4 + 1+2+6+6+5 = 24, {1} = 4 + 5+4+1+2+6
    # = 22, {2} = 4 + 6+6+5+4+1 = 26. From {1}: adding 0 gives
    # 8 + 1+2+1+2+5 = 19, adding 2 gives 8 + 5+4+1+2+1 = 21. From {0, 1}:
    # adding 2 gives 12 + 1+2+1+2+1 = 19, not strictly lower.
    singles = [total_cost(open_costs, dist, {f}) for f in range(3)]
    assert singles == [24, 22, 26]
    assert total_cost(open_costs, dist, {1, 2}) == 21
    assert total_cost(open_costs, dist, {0, 1, 2}) == 19
    show("single facility {0}, {1}, {2}", f"{fmt(singles[0])}, {fmt(singles[1])}, "
         f"{fmt(singles[2])}")
    greedy, greedy_cost = facility_location_greedy(open_costs, dist)
    exact, exact_cost = facility_location_exact(open_costs, dist)
    assert greedy == {0, 1} and greedy_cost == 19
    assert exact == {0, 1} and close(exact_cost, 19)
    show("greedy", f"start {{1}} (22), add 0 (19), stop: {sorted(greedy)}, cost "
         f"{fmt(greedy_cost)}")
    show("exact optimum (MILP)", f"{sorted(exact)}, cost {fmt(exact_cost)}")
    print("  Opening all three also costs 19, so greedy stops (strict improvement")
    print("  only). Greedy is optimal here, but it has no proven ratio: there is no")
    print("  lower bound in the algorithm to compare against, so only the MILP")
    print("  optimum on small instances can say how good it is.")


def section_tsp() -> None:
    banner(4, "Metric TSP: MST doubling against the Held-Karp optimum")
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    square_dist = [[math.dist(a, b) for b in square] for a in square]
    xs = [0, 1, -2, 4, -8, 16]
    line = [[abs(a - b) for b in xs] for a in xs]
    # Hand-checked (spec 10.3). Square: Prim from 0 takes the unit edges
    # (0, 1), (1, 2), (0, 3), weight 3; the preorder 0, 1, 2, 3 is the
    # perimeter, cost 4 = OPT. Line: sorted points -8, -2, 0, 1, 4, 16 have
    # gaps 6, 2, 1, 3, 12, so the MST weighs 24; Prim's tree has children
    # 0 -> {1, 2}, 1 -> 3, 3 -> 5, 2 -> 4, preorder 0, 1, 3, 5, 2, 4, cost
    # 1 + 3 + 12 + 18 + 6 + 8 = 48 = 2 x (16 - (-8)) = OPT.
    cases = [
        ("unit square corners", square_dist, 3.0, [(0, 1), (1, 2), (0, 3)],
         [0, 1, 2, 3, 0], 4.0),
        ("line x = 0, 1, -2, 4, -8, 16", line, 24, [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)],
         [0, 1, 3, 5, 2, 4, 0], 48),
    ]
    for name, dist, a_weight, a_edges, a_tour, a_opt in cases:
        n = len(dist)
        assert is_metric(dist)
        weight, edges = mst_weight_and_edges(dist)
        cost, tour = tsp_mst_doubling(dist)
        opt, _ = tsp_bitmask(dist)
        assert close(weight, a_weight) and list(edges) == a_edges
        assert tour == a_tour and is_valid_tour(tour, n)
        assert close(cost, a_opt) and close(tour_cost(dist, tour), cost) and close(opt, a_opt)
        assert weight <= opt + TOL and cost <= 2 * weight + TOL
        print(f"  {name} (metric: checked by is_metric):")
        show("  MST weight (lower bound)", f"{fmt(weight)}, edges {edges}")
        show("  MST doubling tour", f"{tour}")
        show("  tour cost / 2 x MST bound", f"{fmt(cost)} / {fmt(2 * weight)}")
        show("  Held-Karp optimum", f"{fmt(opt)}: ratio {cost / opt:.2f}")
    nn_cost, nn_tour = tsp_nearest_neighbor(line)
    # Week 8 anchor: 1 + 3 + 6 + 12 + 24 + 16 = 62.
    assert (nn_cost, nn_tour) == (62, [0, 1, 2, 3, 4, 5, 0])
    assert round(nn_cost / 48, 2) == 1.29
    show("line, nearest neighbor (Week 8)", f"{nn_cost}: ratio 1.29, no proven bound")
    print("  Deleting one edge of an optimal tour leaves a spanning tree, so")
    print("  MST <= OPT; the doubled tree, shortcut by the triangle inequality, costs")
    print("  at most 2 x MST. Both instances meet the optimum exactly.")


def section_maxcut() -> None:
    banner(5, "Max-Cut: a coin per vertex, local search and the exact maximum")
    graphs = [
        ("K4", list(range(4)), [(a, b) for a in range(4) for b in range(a + 1, 4)]),
        ("5-cycle C5", list(range(5)), [(i, (i + 1) % 5) for i in range(5)]),
        ("K3,3", list(range(6)), [(a, b) for a in range(3) for b in range(3, 6)]),
    ]
    # Spec 10.4 anchors, hand-checked: K4 split 2-2 cuts 4 of 6 (a 1-3 split
    # cuts 3); an odd cycle cannot cut every edge, and alternating sides cuts
    # 4 of 5; K3,3 is bipartite, so its sides cut all 9.
    exact_anchor = {"K4": 4, "5-cycle C5": 4, "K3,3": 9}
    runs = 500
    print(f"  {'graph':<11} {'|E|':>3} {'|E|/2':>5} {'random (seed 42)':>16} "
          f"{f'mean of {runs}':>11} {'local search':>12} {'exact':>5}")
    for name, vertices, edges in graphs:
        n_edges = len(edges)
        rand_cut, rand_side = maxcut_random(vertices, edges, seed=42)
        assert set(rand_side) == set(vertices) and set(rand_side.values()) <= {0, 1}
        assert rand_cut == cut_value(edges, rand_side)
        assert (rand_cut, rand_side) == maxcut_random(vertices, edges, seed=42)
        mean = statistics.mean(maxcut_random(vertices, edges, seed=s)[0] for s in range(runs))
        # Each edge is cut with probability 1/2 and the edge indicators are
        # pairwise independent, so Var = |E|/4 and the standard error of a
        # 500-run mean is under 0.07: 0.3 is more than four of them.
        assert abs(mean - n_edges / 2) <= 0.3
        local_cut, local_side, flips = maxcut_local_search(vertices, edges, seed=42)
        exact_cut, exact_side = maxcut_exact(vertices, edges)
        assert exact_cut == exact_anchor[name] == cut_value(edges, exact_side)
        assert local_cut == cut_value(edges, local_side)
        assert n_edges / 2 <= local_cut <= exact_cut and rand_cut <= exact_cut
        local_text = f"{local_cut} ({flips} flip{'' if flips == 1 else 's'})"
        print(f"  {name:<11} {n_edges:>3} {fmt(n_edges / 2):>5} {rand_cut:>16} "
              f"{mean:>11.3f} {local_text:>12} {exact_cut:>5}")
    vertices, edges = graphs[2][1], graphs[2][2]
    zeros = {v: 0 for v in vertices}
    k33_cut, k33_side, k33_flips = maxcut_local_search(vertices, edges, start=zeros)
    # Hand-checked: from all zeros the first improving vertex in each scan
    # is 0, then 1, then 2 (each flip cuts 3 more edges), and then the cut
    # is all 9 edges: 3 flips.
    assert k33_cut == 9 and cut_value(edges, k33_side) == 9 and k33_flips == 3
    show("K3,3 local search from all zeros", f"{k33_cut} of 9 after {k33_flips} flips")
    print("  |E| is an upper bound on OPT, so cut / |E| is a certified lower bound on")
    print("  the ratio. A single random cut can fall below |E|/2; the guarantee is")
    print("  about the mean. Local search reaches |E|/2 on every run, because at a")
    print("  local optimum each vertex has at least half its edges cut.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 9 demonstration")
    print("Approximation algorithms - Robert Deibel")
    print("=" * WIDTH)
    print("  Every guarantee below is a comparison with a lower bound that takes")
    print("  polynomial time to compute. Every line is asserted.")
    section_vertex_cover()
    section_set_cover()
    section_facility_location()
    section_tsp()
    section_maxcut()
    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
