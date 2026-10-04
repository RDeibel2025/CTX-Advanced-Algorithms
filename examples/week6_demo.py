#!/usr/bin/env python3
"""Runnable demonstration of the Week 6 dynamic programming code.

    python examples/week6_demo.py

Runs all four algorithms on inputs whose answers were checked by hand:

1. **Space-optimized knapsack** - the one-row table, the comparison table
   and runtime summary the instructions ask for, and the reason the
   capacity loop has to run downward.
2. **Matrix chain multiplication** - the CLRS chain, its minimum cost and
   its optimal parenthesization.
3. **Floyd-Warshall** - the CLRS all-pairs example with negative edges: the
   distance and predecessor matrices and one reconstructed path, plus what
   Week 4's Dijkstra does with the same graph.
4. **TSP by bitmask DP** - the four-city tour, checked against brute force.

Every claim printed is also asserted, so the script exits non-zero rather
than printing something untrue. It finishes in a few seconds.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dp.knapsack import knapsack_tab  # noqa: E402
from src.dp_advanced.bitmask_traveling_salesman import (  # noqa: E402
    tsp_bitmask,
    tsp_brute_force,
)
from src.dp_advanced.floyd_warshall import (  # noqa: E402
    NegativeCycleError,
    all_pairs_dijkstra,
    floyd_warshall,
    reconstruct_path,
)
from src.dp_advanced.matrix_chain_multiplication import (  # noqa: E402
    matrix_chain_order,
    mcm_memoized,
    mcm_recursive,
)
from src.dp_advanced.space_optimized_knapsack import (  # noqa: E402
    compare_with_standard,
    knapsack_ascending_scan,
    knapsack_space_optimized,
    knapsack_space_optimized_with_items,
    print_comparison,
)
from src.graphs.graph import Graph  # noqa: E402
from src.utils.matrix_utils import INF, format_matrix, parenthesization_cost  # noqa: E402

WIDTH = 74


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<32} {value}")


# ----------------------------------------------------------------------
def section_knapsack() -> None:
    banner(1, "Space-optimized 0/1 knapsack")

    weights, values, capacity = [1, 3, 4, 5], [1, 4, 5, 7], 7
    best = knapsack_space_optimized(weights, values, capacity)
    show("weights / values / capacity", f"{weights} / {values} / {capacity}")
    show("one-row optimum", best)
    show("Week 5 two-dimensional optimum", knapsack_tab(weights, values, capacity))
    assert best == knapsack_tab(weights, values, capacity) == 9

    value, items = knapsack_space_optimized_with_items(weights, values, capacity)
    show("items recovered", f"{items}  (weights {[weights[i] for i in items]})")
    assert value == 9 and sum(weights[i] for i in items) <= capacity
    assert sum(values[i] for i in items) == 9

    print("\n  Why the capacity loop runs downward. Two items, capacity 6:")
    w, v = [2, 3], [3, 4]
    down = knapsack_space_optimized(w, v, 6)
    up = knapsack_ascending_scan(w, v, 6)
    show("downward scan (0/1)", f"{down}  - both items, weight 5")
    show("upward scan", f"{up}  - the weight-2 item taken three times")
    assert (down, up) == (7, 9)
    print("  Scanning upward reads a cell this item already updated, so the item")
    print("  can be taken again. That is unbounded knapsack, not 0/1.")

    print("\n  Comparison against Week 5's two-dimensional table (n=100, W=1000):")
    import random
    rng = random.Random(42)
    big_w = [rng.randint(1, 50) for _ in range(100)]
    big_v = [rng.randint(1, 100) for _ in range(100)]
    result = compare_with_standard(big_w, big_v, 1000, repeat=3)
    print_comparison(result)
    assert result["agree"]
    assert result["optimized_peak_kib"] < result["standard_peak_kib"]


def section_mcm() -> None:
    banner(2, "Matrix chain multiplication")

    p = [30, 35, 15, 5, 10, 20, 25]
    cost, paren = matrix_chain_order(p)
    show("dimensions (CLRS 14.2)", p)
    show("minimum scalar multiplications", f"{cost:,}")
    show("optimal parenthesization", paren)
    assert cost == 15125 and paren == "((A1(A2A3))((A4A5)A6))"
    assert mcm_recursive(p) == mcm_memoized(p) == 15125
    assert parenthesization_cost(p, paren) == 15125

    naive = "(((((A1A2)A3)A4)A5)A6)"
    show("left to right instead", f"{parenthesization_cost(p, naive):,}  {naive}")
    print("  Same product, same answer, different amount of work: the order of")
    print("  an associative operation is the whole problem.")


def section_floyd_warshall() -> None:
    banner(3, "Floyd-Warshall on a graph with negative edges")

    edges = [(0, 1, 3), (0, 2, 8), (0, 4, -4), (1, 3, 1), (1, 4, 7),
             (2, 1, 4), (3, 0, 2), (3, 2, -5), (4, 3, 6)]
    graph = Graph(directed=True, weighted=True)
    for node in range(5):
        graph.add_node(node)
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    show("edges (CLRS all-pairs example)", f"{len(edges)}, three of them negative")

    dist, pred = floyd_warshall(graph)
    print("\n  distance matrix:")
    print(format_matrix(dist))
    print("\n  predecessor matrix (- on the diagonal):")
    print(format_matrix(pred))
    assert dist == [[0, 1, -3, 2, -4], [3, 0, -4, 1, -1], [7, 4, 0, 5, 3],
                    [2, -1, -5, 0, -2], [8, 5, 1, 6, 0]]

    path = reconstruct_path(pred, 0, 2)
    cost = sum(next(w for a, b, w in edges if a == x and b == y)
               for x, y in zip(path, path[1:]))
    show("\n  shortest path 0 -> 2", f"{path}, cost {cost}")
    assert path == [0, 4, 3, 2] and cost == dist[0][2] == -3

    try:
        all_pairs_dijkstra(graph)
    except ValueError as error:
        show("Week 4 Dijkstra on this graph", f"refuses: {error}")
    else:  # pragma: no cover
        raise AssertionError("Dijkstra should refuse negative weights")

    cycle = [[0, 1, INF], [INF, 0, -2], [-1, INF, 0]]
    try:
        floyd_warshall(cycle)
    except NegativeCycleError:
        show("a graph with a negative cycle", "NegativeCycleError, as it should")
    else:  # pragma: no cover
        raise AssertionError("a negative cycle should raise")


def section_tsp() -> None:
    banner(4, "Traveling salesman by bitmask DP (Held-Karp)")

    dist = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    cost, tour = tsp_bitmask(dist)
    brute_cost, _ = tsp_brute_force(dist)
    show("four cities, symmetric distances", "")
    show("bitmask DP tour", f"{tour}, cost {cost:g}")
    show("brute force cost", f"{brute_cost:g}")
    edge_sum = sum(dist[a][b] for a, b in zip(tour, tour[1:]))
    assert cost == brute_cost == edge_sum == 80
    assert tour[0] == tour[-1] == 0 and sorted(tour[:-1]) == [0, 1, 2, 3]
    print("  Its reverse is optimal too, since the distances are symmetric. The DP")
    print("  keeps one entry per (subset of cities, last city): n * 2^n states,")
    print("  against the (n - 1)! orderings brute force tries.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 6 demonstration")
    print("Dynamic programming II - Robert Deibel")
    print("=" * WIDTH)
    print("  Every line below is also asserted; this script exits non-zero if")
    print("  any demonstrated claim fails to hold.")

    section_knapsack()
    section_mcm()
    section_floyd_warshall()
    section_tsp()

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
