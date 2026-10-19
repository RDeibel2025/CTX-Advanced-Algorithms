#!/usr/bin/env python3
"""Runnable demonstration of the Week 8 computational complexity code.

    python examples/week8_demo.py

Six short sections, every random draw seeded so the output is the same on
every run:

1. **3-SAT to Vertex Cover** - the full step-by-step reduction of the
   Sipser formula (x1 or x2) and (not x1 or not x2) and (not x1 or x2),
   the certificate carried forward (assignment to cover) and back (cover to
   assignment), and the unsatisfiable eight-clause formula whose graph has
   no cover of size k.
2. **The chain to Independent Set and Clique** - ``reduce_chain`` follows
   the same instance through two more reductions, and the certificate
   travels the whole way and back.
3. **Subset Sum** - backtracking and greedy on [3, 34, 4, 12, 5, 2] with
   targets 9, 30 and 0, the case greedy cannot solve, and the node count of
   the worst-case family as n grows.
4. **SAT** - plain backtracking, DPLL and WalkSAT on the Sipser formula,
   the unsatisfiable formulas and one random 3-SAT formula.
5. **Vertex Cover** - the 2-approximation, max-degree greedy and the exact
   minimum on the path, the star and K3,3, where the factor 2 is hit.
6. **TSP** - nearest neighbor, 2-opt and the Held-Karp optimum on the
   four-city matrix, the six points on a line (62 against 48) and a seeded
   Euclidean instance.

Each section reads its results through one idea: checking a proposed
solution is cheap, finding one is not. Every certificate printed is checked
by its polynomial-time verifier. The anchor values come from sections
10.1-10.5 of the Week 8 build spec, and every claim printed is also
asserted, so the script exits non-zero rather than printing something
untrue. It finishes in a few seconds.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Any, Dict, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.complexity.reduction_demo import (  # noqa: E402
    IndependentSetToClique,
    ThreeSatToVertexCover,
    VertexCoverToIndependentSet,
    is_satisfying,
    is_vertex_cover,
    min_vertex_cover_exact,
    reduce_chain,
)
from src.complexity.satisfiability_solver import (  # noqa: E402
    random_3sat,
    sat_backtracking,
    sat_dpll,
    walksat,
)
from src.complexity.subset_sum_backtracking import (  # noqa: E402
    subset_sum_backtracking,
    subset_sum_dp,
    subset_sum_greedy,
    subset_sum_random_restart,
    worst_case_instance,
)
from src.complexity.traveling_salesman_approx import (  # noqa: E402
    euclidean_instance,
    is_valid_tour,
    tour_cost,
    tsp_nearest_neighbor,
    tsp_nearest_neighbor_best_start,
    tsp_two_opt,
)
from src.complexity.vertex_cover_approx import (  # noqa: E402
    approximation_ratio,
    greedy_maximal_matching,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_max_degree_greedy,
)
from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask  # noqa: E402
from src.utils.graph_utils import complement, is_clique, is_independent_set  # noqa: E402

WIDTH = 74

#: The Sipser (Thm 7.44) anchor formula, n = 2 variables, m = 3 clauses.
SIPSER = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]
#: Its only satisfying assignment (spec 10.1, re-checked by brute force below).
SIPSER_SOLUTION = {1: False, 2: True}
#: All eight sign patterns over x1, x2, x3: every assignment falsifies one clause.
EIGHT = [[s1 * 1, s2 * 2, s3 * 3] for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)]


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<34} {value}")


def indented(text: str) -> None:
    for line in text.splitlines():
        print(f"  {line}")


def all_satisfying(cnf: Sequence[Sequence[int]], n_vars: int) -> List[Dict[int, bool]]:
    """Every satisfying assignment, by brute force over all 2^n assignments."""
    found = []
    for values in itertools.product((False, True), repeat=n_vars):
        assignment = {i + 1: value for i, value in enumerate(values)}
        if is_satisfying(cnf, assignment):
            found.append(assignment)
    return found


def is_sub_multiset(subset: Sequence[int], nums: Sequence[int]) -> bool:
    """True when every value of ``subset`` is available in ``nums``, repeats counted."""
    pool = list(nums)
    for value in subset:
        if value not in pool:
            return False
        pool.remove(value)
    return True


def section_reduction() -> None:
    banner(1, "3-SAT to Vertex Cover, step by step (Garey-Johnson, Sipser 7.44)")
    n, m = 2, len(SIPSER)
    solutions = all_satisfying(SIPSER, n)
    assert solutions == [SIPSER_SOLUTION]
    show("CNF, clauses padded to 3 literals", SIPSER)
    show("formula", "(x1 OR x2) AND (NOT x1 OR NOT x2) AND (NOT x1 OR x2)")
    show("satisfying assignments (4 tried)", "exactly one: x1 = False, x2 = True")

    reduction = ThreeSatToVertexCover()
    result = reduction.transform((SIPSER, n))
    (vertices, edges), k = result.target_instance
    print()
    print("  The reduction's own log:")
    text = reduction.explain(result)
    indented(text)
    assert len(vertices) == 2 * n + 3 * m == 13
    assert len(edges) == n + 6 * m == 20
    assert k == n + 2 * m == 8
    for gadget in ("variable gadgets", "clause gadgets", "connecting edges", "k = 8"):
        assert gadget in text

    print()
    cover = reduction.map_solution_forward((SIPSER, n), SIPSER_SOLUTION, result)
    assert len(cover) == k and is_vertex_cover(edges, cover)
    show("forward: assignment -> cover", ", ".join(sorted(cover)))
    show("", f"{len(cover)} nodes = k, verified by is_vertex_cover")
    back = reduction.map_solution_back((SIPSER, n), cover, result)
    assert back == SIPSER_SOLUTION and is_satisfying(SIPSER, back)
    show("back: cover -> assignment", f"{back}, verified by is_satisfying")

    exact = min_vertex_cover_exact(vertices, edges)
    exact_back = reduction.map_solution_back((SIPSER, n), exact, result)
    assert len(exact) == 8 and is_vertex_cover(edges, exact)
    assert exact_back == SIPSER_SOLUTION
    show("exact minimum vertex cover", f"{len(exact)} <= k = {k}: satisfiable, as it must be")
    show("the exact cover, mapped back", f"{exact_back}, the same assignment")

    print()
    assert all_satisfying(EIGHT, 3) == []
    unsat = reduction.transform((EIGHT, 3))
    (u_vertices, u_edges), u_k = unsat.target_instance
    u_exact = min_vertex_cover_exact(u_vertices, u_edges)
    assert (len(u_vertices), len(u_edges), u_k) == (30, 51, 19)
    assert len(u_exact) == 20 and is_vertex_cover(u_edges, u_exact)
    show("eight sign patterns over x1..x3", "no satisfying assignment (all 8 tried)")
    show("its graph", f"{len(u_vertices)} nodes, {len(u_edges)} edges, k = {u_k}")
    show("exact minimum vertex cover", f"{len(u_exact)} > k = {u_k}: no cover of size k")
    print("  Both answers agree with the formulas: satisfiable iff a cover of size")
    print("  at most k = n + 2m exists. Building the graph is O(n + m); checking a")
    print("  cover is one pass over the edges. Only finding one is expensive.")


def section_chain() -> None:
    banner(2, "The chain: Vertex Cover -> Independent Set -> Clique")
    chain = [ThreeSatToVertexCover(), VertexCoverToIndependentSet(), IndependentSetToClique()]
    results = reduce_chain((SIPSER, 2), chain)
    sizes = [(len(r.target_instance[0][0]), len(r.target_instance[0][1]), r.target_instance[1])
             for r in results]
    # 13 nodes throughout; the complement of a 13-node, 20-edge graph has
    # 13 * 12 / 2 - 20 = 58 edges; k' = |V| - k = 13 - 8 = 5.
    assert sizes == [(13, 20, 8), (13, 20, 13 - 8), (13, 13 * 12 // 2 - 20, 5)]
    for reduction, (n_nodes, n_edges, k) in zip(chain, sizes):
        show(reduction.name, f"{n_nodes} nodes, {n_edges} edges, k = {k}")
    for reduction, result in zip(chain[1:], results[1:]):
        print()
        indented(reduction.explain(result))

    print()
    vc_instance, is_instance = results[0].target_instance, results[1].target_instance
    (vertices, edges), _ = vc_instance
    cover = chain[0].map_solution_forward((SIPSER, 2), SIPSER_SOLUTION, results[0])
    independent = chain[1].map_solution_forward(vc_instance, cover, results[1])
    clique = chain[2].map_solution_forward(is_instance, independent, results[2])
    clique_edges = results[2].target_instance[0][1]
    assert independent == set(vertices) - cover and len(independent) == 5
    assert is_independent_set(edges, independent)
    assert clique == independent and is_clique(clique_edges, clique)
    assert clique_edges == complement(vertices, edges)
    show("cover (size 8) -> independent set", ", ".join(sorted(independent)))
    show("", "size 13 - 8 = 5, verified by is_independent_set")
    show("independent set -> clique", "the same 5 nodes, verified by is_clique")
    show("", "on the complement graph")

    back_is = chain[2].map_solution_back(is_instance, clique, results[2])
    back_cover = chain[1].map_solution_back(vc_instance, back_is, results[1])
    back_assignment = chain[0].map_solution_back((SIPSER, 2), back_cover, results[0])
    assert back_cover == cover and back_assignment == SIPSER_SOLUTION
    show("back down the chain", f"clique -> set -> cover -> {back_assignment}")
    print("  A composition of polynomial-time reductions is one, so a fast Clique")
    print("  solver would give a fast 3-SAT solver through this chain.")


def section_subset_sum() -> None:
    banner(3, "Subset Sum: backtracking against the greedy heuristic")
    nums = [3, 34, 4, 12, 5, 2]
    show("values", nums)
    print(f"  {'target':>6}  {'backtracking (nodes)':<24} {'greedy (gap)':<26} DP check")
    # Hand-checked: 9 = 5 + 4; 30 is out of reach because 34 > 30 and the
    # other five values total 26, which greedy (12 + 5 + 4 + 3 + 2) reaches;
    # 0 is the empty subset.
    expected = {9: (True, 9), 30: (False, 26), 0: (True, 0)}
    for target in (9, 30, 0):
        found, nodes = subset_sum_backtracking(nums, target, count_nodes=True)
        greedy, achieved = subset_sum_greedy(nums, target)
        reachable, greedy_value = expected[target]
        assert subset_sum_dp(nums, target) == reachable == (found is not None)
        if found is not None:
            assert sum(found) == target and is_sub_multiset(found, nums)
        assert achieved == greedy_value == sum(greedy) and is_sub_multiset(greedy, nums)
        bt_text = f"{found} ({nodes} node{'' if nodes == 1 else 's'})"
        greedy_text = f"{greedy} (gap {target - achieved})"
        print(f"  {target:>6}  {bt_text:<24} {greedy_text:<26} "
              f"{'reachable' if reachable else 'unreachable'}")
    assert subset_sum_backtracking(nums, 0) == []
    print("  Backtracking proves 30 unreachable. Greedy stops 4 short and cannot")
    print("  tell whether 30 was reachable; only exhaustive search (or the DP) can.")

    print()
    blind, blind_sum = subset_sum_greedy([6, 5, 5], 10)
    exact = subset_sum_backtracking([6, 5, 5], 10)
    restart, restart_sum = subset_sum_random_restart([6, 5, 5], 10, tries=100, seed=42)
    assert (blind, blind_sum) == ([6], 6)
    assert sorted(exact) == [5, 5] and restart_sum == 10 == sum(restart)
    show("[6, 5, 5], target 10: greedy", f"{blind}, gap 4 (6 first, then 5 overshoots)")
    show("backtracking", f"{exact}, exact")
    show("random restart, 100 tries, seed 42", f"{restart}, sum {restart_sum}")

    print()
    print("  Worst-case family (even values, odd target, seed 42): no subset can")
    print("  hit the target, so the search cannot stop early.")
    print(f"  {'n':>4}  {'nodes visited':>14}  {'full tree 2^(n+1) - 1':>22}  {'share':>6}"
          f"  {'factor per value':>16}")
    previous = 0
    for n in (8, 10, 12, 14, 16, 18, 20):
        values, target = worst_case_instance(n, 42)
        assert all(v % 2 == 0 for v in values) and target % 2 == 1
        found, nodes = subset_sum_backtracking(values, target, count_nodes=True)
        full = 2 ** (n + 1) - 1
        assert found is None and previous < nodes <= full
        # n grows by 2 per row, so the factor per added value is a square root.
        factor = f"{(nodes / previous) ** 0.5:.2f}" if previous else ""
        if previous:
            assert 1.5 < (nodes / previous) ** 0.5 < 2.1
        previous = nodes
        print(f"  {n:>4}  {nodes:>14,}  {full:>22,}  {nodes / full:>6.1%}  {factor:>16}".rstrip())
    print("  Pruning skips most of the tree, but the count still grows by a near")
    print("  constant factor, about 1.6 to 2, per added value. Checking a proposed")
    print("  subset is n additions.")


def section_sat() -> None:
    banner(4, "SAT: plain backtracking, DPLL and WalkSAT")
    bt, bt_nodes = sat_backtracking(SIPSER, 2, count_nodes=True)
    dp, dp_nodes = sat_dpll(SIPSER, 2, count_nodes=True)
    ws = walksat(SIPSER, 2, seed=42)
    # Hand-derived node counts. Backtracking: root; x1 = T, then x2 = T
    # (clause 2 false) and x2 = F (clause 3 false); x1 = F, x2 = T succeeds:
    # 6 nodes. DPLL: no unit or pure literal at the root; x1 = T forces both
    # x2 and NOT x2 (conflict); x1 = F forces x2 = T: 3 nodes.
    assert bt == dp == ws == SIPSER_SOLUTION
    assert (bt_nodes, dp_nodes) == (6, 3)
    show("Sipser formula, backtracking", f"{bt}, {bt_nodes} nodes")
    show("DPLL", f"{dp}, {dp_nodes} nodes (unit propagation)")
    show("WalkSAT, seed 42", f"{ws}")
    show("", "all three verified by is_satisfying")
    assert all(is_satisfying(SIPSER, a) for a in (bt, dp, ws))

    print()
    bt, bt_nodes = sat_backtracking(EIGHT, 3, count_nodes=True)
    dp, dp_nodes = sat_dpll(EIGHT, 3, count_nodes=True)
    ws = walksat(EIGHT, 3, max_flips=1_000, seed=42)
    # Every clause mentions all three variables, so plain backtracking cuts
    # only at depth 3: the full tree, 1 + 2 + 4 + 8 = 15 nodes. DPLL decides
    # x1 and x2 (1 + 2 + 4 = 7 nodes); each pair forces x3 both ways.
    assert bt is None and dp is None and ws is None
    assert (bt_nodes, dp_nodes) == (15, 7)
    show("eight sign patterns, backtracking", f"None, {bt_nodes} nodes = the full tree")
    show("DPLL", f"None, {dp_nodes} nodes")
    show("WalkSAT, 1,000 flips", "None: gives up, which proves nothing")
    one = [[1], [-1]]
    assert sat_backtracking(one, 1) is None and sat_dpll(one, 1) is None
    show("[[1], [-1]], both complete solvers", "None")

    print()
    n = 20
    cnf = random_3sat(n, 85, 42)
    bt, bt_nodes = sat_backtracking(cnf, n, count_nodes=True)
    dp, dp_nodes = sat_dpll(cnf, n, count_nodes=True)
    ws = walksat(cnf, n, seed=42)
    assert bt is not None and dp is not None and ws is not None
    assert all(is_satisfying(cnf, a) for a in (bt, dp, ws))
    assert dp_nodes < bt_nodes
    show("random 3-SAT, n = 20, m = 85", "ratio 4.26, seed 42, satisfiable")
    show("backtracking nodes", f"{bt_nodes:,}")
    show("DPLL nodes", f"{dp_nodes:,}")
    show("WalkSAT", "found an assignment")
    show("", "all three verified by is_satisfying")
    print("  Inference shrinks the tree but not the worst case. WalkSAT can find a")
    print("  certificate but can never prove that none exists.")


def section_vertex_cover() -> None:
    banner(5, "Vertex Cover: the 2-approximation, max-degree greedy, exact")
    graphs = [
        ("path 1-2-3-4", [1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}),
        ("star, centre 0, leaves 1-5", list(range(6)), {(0, leaf) for leaf in range(1, 6)}),
        ("K3,3, sides 0-2 and 3-5", list(range(6)),
         {(a, b) for a in range(3) for b in range(3, 6)}),
    ]
    # Spec 10.4 anchors: the 2-approximation's cover and the exact size.
    # Max-degree greedy, hand-checked with ties to the smallest vertex:
    # {2, 3}, {0} and {0, 1, 2}, each a minimum cover.
    anchors = {
        "path 1-2-3-4": ({1, 2, 3, 4}, 2, 2),
        "star, centre 0, leaves 1-5": ({0, 1}, 1, 1),
        "K3,3, sides 0-2 and 3-5": (set(range(6)), 3, 3),
    }
    print(f"  {'graph':<28} {'2-approx':>8} {'greedy':>7} {'exact':>6} {'ratio':>6}  matching M")
    for name, vertices, edges in graphs:
        approx = vertex_cover_2approx(vertices, edges)
        greedy = vertex_cover_max_degree_greedy(vertices, edges)
        exact = vertex_cover_exact(vertices, edges)
        matching = greedy_maximal_matching(vertices, edges)
        ratio = approximation_ratio(len(approx), len(exact))
        approx_anchor, exact_size, greedy_size = anchors[name]
        assert approx == approx_anchor and len(exact) == exact_size
        assert len(greedy) == greedy_size and ratio == 2.0
        assert all(is_vertex_cover(edges, c) for c in (approx, greedy, exact))
        assert len(approx) == 2 * len(matching) and len(exact) >= len(matching)
        print(f"  {name:<28} {len(approx):>8} {len(greedy):>7} {len(exact):>6} {ratio:>6.1f}  "
              f"{matching}")
    star_vertices, star_edges = graphs[1][1], graphs[1][2]
    star_approx = vertex_cover_2approx(star_vertices, star_edges)
    star_exact = vertex_cover_exact(star_vertices, star_edges)
    assert star_approx == {0, 1} and star_exact == {0}
    print()
    show("star: 2-approx / exact cover", f"{sorted(star_approx)} / {sorted(star_exact)}")
    empty = [1, 2, 3]
    assert (vertex_cover_2approx(empty, set()) == vertex_cover_max_degree_greedy(empty, set())
            == vertex_cover_exact(empty, set()) == set())
    show("edge-free graph, all three methods", "empty cover")
    print("  The picked edges form a matching M; any cover needs one end of each,")
    print("  so OPT >= |M|, and the algorithm returns 2|M| <= 2 OPT. All three")
    print("  graphs hit the bound exactly. Max-degree greedy is optimal here but")
    print("  has no constant-factor guarantee.")


def section_tsp() -> None:
    banner(6, "TSP: nearest neighbor and 2-opt against the Held-Karp optimum")
    four = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    nn_cost, nn_tour = tsp_nearest_neighbor(four)
    two_cost, two_tour = tsp_two_opt(four, nn_tour)
    opt_cost, _ = tsp_bitmask(four)
    assert (nn_cost, nn_tour) == (80, [0, 1, 3, 2, 0]) and opt_cost == 80
    assert two_cost == 80 and is_valid_tour(two_tour, 4)
    assert is_valid_tour(nn_tour, 4) and tour_cost(four, nn_tour) == 80
    show("four cities: nearest neighbor", f"{nn_cost}, tour {nn_tour}")
    show("2-opt / Held-Karp optimum", f"{two_cost} / {opt_cost}: the greedy walk is optimal")

    print()
    xs = [0, 1, -2, 4, -8, 16]
    line = [[abs(a - b) for b in xs] for a in xs]
    nn_cost, nn_tour = tsp_nearest_neighbor(line)
    two_cost, two_tour = tsp_two_opt(line, nn_tour)
    best_cost, best_tour = tsp_nearest_neighbor_best_start(line)
    opt_cost, opt_tour = tsp_bitmask(line)
    # Hand-checked: from 0 the walk goes 1, -2, 4, -8, 16 (ties to the lower
    # index) and back: 1 + 3 + 6 + 12 + 24 + 16 = 62. Any optimal tour on a
    # line sweeps out and back: 2 * (16 - (-8)) = 48. Starting at x = 16 the
    # walk is such a sweep: 12 + 3 + 1 + 2 + 6 + 24 = 48.
    assert (nn_cost, nn_tour) == (62, [0, 1, 2, 3, 4, 5, 0])
    assert opt_cost == 48 == 2 * (max(xs) - min(xs))
    assert two_cost == 48 and best_cost == 48
    for tour in (nn_tour, two_tour, best_tour, opt_tour):
        assert is_valid_tour(tour, len(xs))
    assert round(nn_cost / opt_cost, 2) == 1.29
    print("  Six points on a line, x = 0, 1, -2, 4, -8, 16, distance |a - b|:")
    show("nearest neighbor from 0", f"{nn_cost}, tour {nn_tour}")
    show("Held-Karp optimum", f"{opt_cost}: ratio {nn_cost / opt_cost:.2f}")
    show("2-opt from the greedy tour", f"{two_cost}, tour {two_tour}")
    show("nearest neighbor, best start", f"{best_cost}")
    print("  Nearest neighbor has no constant-factor guarantee; 2-opt repairs this")
    print("  tour, but it is a local search and is not guaranteed to either.")

    print()
    _, matrix = euclidean_instance(10, seed=42)
    nn_cost, nn_tour = tsp_nearest_neighbor(matrix)
    two_cost, two_tour = tsp_two_opt(matrix, nn_tour)
    opt_cost, _ = tsp_bitmask(matrix)
    assert is_valid_tour(nn_tour, 10) and is_valid_tour(two_tour, 10)
    assert opt_cost - 1e-9 <= two_cost <= nn_cost
    show("10 random points, seed 42: NN", f"{nn_cost:.2f} (ratio {nn_cost / opt_cost:.3f})")
    show("NN + 2-opt", f"{two_cost:.2f} (ratio {two_cost / opt_cost:.3f})")
    show("Held-Karp optimum", f"{opt_cost:.2f}")
    print("  Checking any of these tours is one pass: is_valid_tour and tour_cost.")
    print("  Held-Karp needs O(n^2 2^n) time to prove which one is shortest.")


def _ensure_printable() -> None:
    """Switch stdout to UTF-8 if it cannot print the NOT sign in node names."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "\u00ac".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    _ensure_printable()
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 8 demonstration")
    print("Computational complexity - Robert Deibel")
    print("=" * WIDTH)
    print("  Checking a solution is cheap; finding one is not. Every certificate")
    print("  below is checked by its verifier, and every line is asserted.")
    section_reduction()
    section_chain()
    section_subset_sum()
    section_sat()
    section_vertex_cover()
    section_tsp()
    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
