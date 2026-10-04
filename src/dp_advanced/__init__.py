"""Dynamic programming II: space optimization and structured state spaces.

Week 6 adds four algorithms in two pairs:

* **Tables with a dimension they do not need.** The 0/1 knapsack drops the
  item index and keeps one row of capacities; Floyd-Warshall drops the
  intermediate-vertex index and keeps one distance matrix. Both overwrite a
  smaller table in place, and each has its own reason that doing so is safe.
* **State spaces that cannot be shrunk.** Matrix chain multiplication keeps
  every interval and the Held-Karp TSP keeps every subset, because every
  shorter interval and every smaller subset must stay available until the
  end.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.dp_advanced.bitmask_traveling_salesman import tsp_bitmask, tsp_brute_force
from src.dp_advanced.floyd_warshall import (
    NegativeCycleError,
    all_pairs_dijkstra,
    floyd_warshall,
    floyd_warshall_3d,
    reconstruct_path,
)
from src.dp_advanced.matrix_chain_multiplication import (
    matrix_chain_order,
    mcm_bottom_up,
    mcm_memoized,
    mcm_recursive,
    optimal_parenthesization,
)
from src.dp_advanced.space_optimized_knapsack import (
    compare_with_standard,
    knapsack_ascending_scan,
    knapsack_space_optimized,
    knapsack_space_optimized_with_items,
    print_comparison,
)

__all__ = [
    "knapsack_space_optimized",
    "knapsack_ascending_scan",
    "knapsack_space_optimized_with_items",
    "compare_with_standard",
    "print_comparison",
    "mcm_recursive",
    "mcm_memoized",
    "mcm_bottom_up",
    "optimal_parenthesization",
    "matrix_chain_order",
    "NegativeCycleError",
    "floyd_warshall",
    "floyd_warshall_3d",
    "reconstruct_path",
    "all_pairs_dijkstra",
    "tsp_bitmask",
    "tsp_brute_force",
]
