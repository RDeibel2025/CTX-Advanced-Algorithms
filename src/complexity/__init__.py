"""Computational complexity: reductions, exact search and approximation (Week 8).

Five modules, one question each:

* :mod:`reduction_demo` - a small framework for polynomial-time reductions,
  with the 3-SAT to Vertex Cover reduction (Garey and Johnson; Sipser
  Thm 7.44) and the follow-on reductions Vertex Cover to Independent Set
  and Independent Set to Clique. It also holds the two polynomial-time
  **verifiers**, ``is_satisfying`` and ``is_vertex_cover``, and the exact
  branch-and-bound solver ``min_vertex_cover_exact``.
* :mod:`subset_sum_backtracking` - exact Subset Sum by backtracking with
  node counts, the greedy and random-restart heuristics, and the
  pseudo-polynomial DP that makes Subset Sum only weakly NP-complete.
* :mod:`satisfiability_solver` - SAT by plain backtracking, by DPLL (unit
  propagation and pure-literal elimination) and by WalkSAT local search,
  plus a seeded random 3-SAT generator.
* :mod:`vertex_cover_approx` - the greedy edge-picking 2-approximation with
  its proof, the max-degree greedy that has no constant-factor guarantee,
  and the exact solver re-exported for ratios.
* :mod:`traveling_salesman_approx` - nearest neighbor, best-start nearest
  neighbor and 2-opt for TSP, with a tour verifier and a seeded Euclidean
  instance generator.

The thread through all five is the P versus NP question. Checking a
proposed solution (an assignment, a cover, a subset, a tour) takes
polynomial time, which is what puts these problems in NP, but no known
algorithm finds a solution in polynomial time, and a reduction shows that a
fast algorithm for one NP-complete problem would give fast algorithms for
all of them. Approximation algorithms and heuristics give up optimality or
completeness to return a checkable answer quickly.

Every random draw goes through a seeded ``random.Random`` from
:mod:`src.utils.random_utils`, so results are reproducible.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.complexity.reduction_demo import (
    IndependentSetToClique,
    Reduction,
    ReductionResult,
    ThreeSatToVertexCover,
    VertexCoverToIndependentSet,
    is_satisfying,
    is_vertex_cover,
    min_vertex_cover_exact,
    reduce_chain,
)
from src.complexity.satisfiability_solver import (
    random_3sat,
    sat_backtracking,
    sat_dpll,
    walksat,
)
from src.complexity.subset_sum_backtracking import (
    satisfiable_instance,
    subset_sum_backtracking,
    subset_sum_dp,
    subset_sum_greedy,
    subset_sum_random_restart,
    worst_case_instance,
)
from src.complexity.traveling_salesman_approx import (
    euclidean_instance,
    is_valid_tour,
    tour_cost,
    tsp_nearest_neighbor,
    tsp_nearest_neighbor_best_start,
    tsp_two_opt,
)
from src.complexity.vertex_cover_approx import (
    approximation_ratio,
    greedy_maximal_matching,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_max_degree_greedy,
)

__all__ = [
    "Reduction", "ReductionResult", "ThreeSatToVertexCover", "VertexCoverToIndependentSet",
    "IndependentSetToClique", "reduce_chain", "is_satisfying", "is_vertex_cover",
    "min_vertex_cover_exact", "subset_sum_backtracking", "subset_sum_greedy",
    "subset_sum_random_restart", "subset_sum_dp", "worst_case_instance", "satisfiable_instance",
    "sat_backtracking", "sat_dpll", "walksat", "random_3sat", "vertex_cover_2approx",
    "vertex_cover_max_degree_greedy", "vertex_cover_exact", "greedy_maximal_matching",
    "approximation_ratio", "tsp_nearest_neighbor", "tsp_nearest_neighbor_best_start",
    "tsp_two_opt", "tour_cost", "is_valid_tour", "euclidean_instance",
]
