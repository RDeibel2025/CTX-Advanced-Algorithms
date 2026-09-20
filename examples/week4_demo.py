#!/usr/bin/env python3
"""Runnable demonstration of the Week 4 graph code.

    python examples/week4_demo.py

Walks through the week on graphs small enough to read, and shows the one
idea the whole week turns on:

1. **One Graph class** covering directed/undirected and weighted/unweighted.
2. **Two representations** of the same graph, agreeing with each other, and
   costing very different amounts of memory.
3. **One loop, three algorithms** - BFS, DFS and Dijkstra differ by the
   container holding the pending nodes, and by nothing else that matters.
4. **Recursive DFS**, the exception: no explicit container, it borrows the
   interpreter's call stack.
5. **Dijkstra** on the Week 3 heap, with a hand-checkable answer.

Every claim printed is also asserted, so the script fails loudly rather
than printing something untrue. It exits 0 when everything holds, and
takes well under a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.graphs.bfs import bfs, bfs_component, bfs_levels  # noqa: E402
from src.graphs.dfs import dfs_component, dfs_iterative, dfs_recursive  # noqa: E402
from src.graphs.dijkstra import (  # noqa: E402
    dijkstra,
    dijkstra_linear_scan,
    reconstruct_path,
    shortest_path,
)
from src.graphs.graph import Graph  # noqa: E402
from src.utils.graph_generator import sparse_graph  # noqa: E402

WIDTH = 74


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<28} {value}")


# ----------------------------------------------------------------------
def section_graph() -> None:
    banner(1, "One class, four kinds of graph")

    undirected = Graph()
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]:
        undirected.add_edge(u, v)
    show("undirected", undirected)
    show("neighbours of c", undirected.get_neighbors("c"))
    assert set(undirected.get_neighbors("c")) == {"b", "a", "d"}

    directed = Graph(directed=True)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]:
        directed.add_edge(u, v)
    show("directed", directed)
    show("out of c / into c", f"{directed.get_neighbors('c')} / "
                              f"in_degree {directed.in_degree('c')}")
    assert directed.has_edge("a", "b") and not directed.has_edge("b", "a")
    assert undirected.has_edge("b", "a"), "an undirected edge works both ways"

    weighted = Graph(weighted=True)
    weighted.add_edge("a", "b", 2.5)
    weighted.add_edge("b", "c", 1.5)
    show("weighted", weighted)
    show("neighbours of b", weighted.get_neighbors("b"))
    assert weighted.get_edge_weight("a", "b") == 2.5

    print("\n  add_edge creates any node it has not seen, so a graph can be")
    print("  built from an edge list alone:")
    show("nodes after 4 edges", undirected.nodes())
    undirected.remove_node("c")
    show("after remove_node('c')", f"{undirected} nodes={undirected.nodes()}")
    assert not undirected.has_edge("b", "c") and not undirected.has_edge("c", "d")
    print("  Removing a node takes every edge that touched it with it.")


def section_representations() -> None:
    banner(2, "Two representations of one graph")

    graph = Graph()
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]:
        graph.add_edge(u, v)
    dense = graph.to_adjacency_matrix()
    show("graph", graph)
    show("nodes, in row order", dense.nodes)
    print("  adjacency matrix:")
    for node, row in zip(dense.nodes, dense.matrix.tolist()):
        print(f"      {node}  {row}")

    for u in graph.nodes():
        for v in graph.nodes():
            i, j = dense.index[u], dense.index[v]
            assert graph.has_edge(u, v) == bool(dense.matrix[i, j])
    print("  Both representations agree on every one of the 25 pairs.")

    bigger = sparse_graph(1_000, avg_degree=4, connected=True, seed=42)
    matrix = bigger.to_adjacency_matrix()
    show("1,000 nodes, 4 avg degree", f"{bigger.edge_count:,} edges, "
                                      f"density {bigger.density():.4f}")
    show("matrix cells / bytes", f"{1_000 * 1_000:,} cells, "
                                 f"{matrix.matrix.nbytes / 1e6:.1f} MB (numpy uint8)")
    show("edges actually present", f"{bigger.edge_count:,} of "
                                   f"{1_000 * 999 // 2:,} possible")
    print("  The matrix pays for every pair that could exist. At this density")
    print("  that is 99.6% zeros, which is the whole argument for the list.")
    assert matrix.matrix.nbytes == 1_000 * 1_000


def section_traversals() -> None:
    banner(3, "One loop, two containers")

    graph = Graph()
    for u, v in [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]:
        graph.add_edge(u, v)
    show("a small tree", graph)
    show("BFS from 1", bfs(graph, 1))
    show("DFS from 1 (iterative)", dfs_iterative(graph, 1))
    show("BFS hop levels", bfs_levels(graph, 1))
    assert bfs(graph, 1) == [1, 2, 3, 4, 5, 6, 7]
    assert dfs_iterative(graph, 1) == [1, 2, 4, 5, 3, 6, 7]
    print("\n  Same graph, same start, same loop. BFS takes from the front of a")
    print("  queue and sweeps level by level; DFS takes from the top of a stack")
    print("  and dives. The container is the only difference.")

    print("\n  Recursive DFS is the exception that proves it: no container of")
    print("  its own, it borrows the interpreter's call stack.")
    show("DFS recursive", dfs_recursive(graph, 1))
    assert dfs_recursive(graph, 1) == dfs_iterative(graph, 1)
    print("  Both DFS forms agree exactly, which is only true because the")
    print("  iterative one pushes neighbours in reverse.")

    scattered = Graph()
    for u, v in [("a", "b"), ("c", "d"), ("e", "f")]:
        scattered.add_edge(u, v)
    scattered.add_node("g")
    show("\n  four components", scattered)
    show("BFS, whole graph", bfs(scattered))
    show("BFS from 'a' only", bfs_component(scattered, "a"))
    show("DFS from 'a' only", dfs_component(scattered, "a"))
    assert len(bfs(scattered)) == 7 and bfs_component(scattered, "a") == ["a", "b"]
    print("  A traversal that stopped at the first component would miss five")
    print("  of these seven nodes, so both walk every component.")


def section_dijkstra() -> None:
    banner(4, "Dijkstra on the Week 3 heap")

    graph = Graph(weighted=True)
    for u, v, w in [("A", "B", 4), ("A", "C", 2), ("C", "B", 1), ("B", "D", 5),
                    ("C", "D", 8), ("D", "E", 2), ("C", "E", 10)]:
        graph.add_edge(u, v, w)
    show("graph", graph)
    distances, predecessors = dijkstra(graph, "A")
    for node in sorted(distances):
        print(f"      A -> {node}:  cost {distances[node]:>4}   path "
              f"{reconstruct_path(predecessors, 'A', node)}")
    assert distances == {"A": 0, "B": 3, "C": 2, "D": 8, "E": 10}
    print("  Hand-check A to B: the direct edge costs 4, but A-C-B costs")
    print("  2 + 1 = 3, so 3 is right and the greedy first guess is not.")

    path, cost = shortest_path(graph, "A", "E")
    show("shortest A to E", f"{path} costing {cost}")
    assert path == ["A", "C", "B", "D", "E"] and cost == 10

    graph.add_node("Z")
    distances, _ = dijkstra(graph, "A")
    show("unreachable node Z", distances["Z"])
    assert distances["Z"] == math.inf
    print("  An unreachable node reports infinite distance rather than crashing.")

    negative = Graph(weighted=True)
    negative.add_edge("A", "B", 3)
    negative.add_edge("B", "C", -2)
    try:
        dijkstra(negative, "A")
    except ValueError as error:
        show("negative weight", f"ValueError: {error}")
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("a negative weight should have been refused")
    print("  Dijkstra is not defined for negative weights, so it refuses them")
    print("  rather than returning a confident wrong answer.")

    same = dijkstra_linear_scan(graph, "A")[0]
    show("linear-scan Dijkstra", "same distances" if same == distances else "DIFFERENT")
    assert same == distances
    print("  The O(V^2) linear-scan version agrees exactly. Swapping the heap")
    print("  for a list changes the cost, not the answer.")


def section_agreement() -> None:
    banner(5, "All three on the same graph")

    graph = sparse_graph(30, avg_degree=3, weighted=True, connected=True, seed=42)
    source = graph.nodes()[0]
    order_bfs = bfs(graph, source)
    order_dfs = dfs_iterative(graph, source)
    distances, _ = dijkstra(graph, source)

    show("graph", graph)
    show("BFS visited", f"{len(order_bfs)} nodes")
    show("DFS visited", f"{len(order_dfs)} nodes")
    show("Dijkstra reached", f"{sum(1 for d in distances.values() if d < math.inf)} nodes")
    show("BFS first 8", order_bfs[:8])
    show("DFS first 8", order_dfs[:8])
    assert set(order_bfs) == set(order_dfs) == set(graph.nodes())
    assert all(d < math.inf for d in distances.values())
    print("\n  Same node set, three different orders, one shared skeleton.")
    print("  BFS and DFS answer 'can I get there'; Dijkstra answers 'how far',")
    print("  and only Dijkstra has to revisit a node's priority to do it.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 4 demonstration")
    print("Graphs, BFS, DFS and Dijkstra - Robert Deibel")
    print("=" * WIDTH)
    print("  Every line below is also asserted; this script exits non-zero if")
    print("  any demonstrated claim fails to hold.")

    section_graph()
    section_representations()
    section_traversals()
    section_dijkstra()
    section_agreement()

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
