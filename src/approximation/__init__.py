"""Approximation algorithms: greedy, LP rounding and randomization (Week 9).

Five modules, one problem each:

* :mod:`vertex_cover_2approx` - the greedy edge-picking (maximal matching)
  2-approximation with its proof, the LP relaxation and its rounding at
  x_v >= 1/2 (also a factor 2), the matching lower bound, and the exact
  solver re-exported from Week 8 for ratios.
* :mod:`set_cover_lpround` - Set Cover as an integer program, its LP
  relaxation, Chvatal's greedy (an H(n) bound), deterministic rounding at
  x_S >= 1/f (an f-approximation), randomized rounding (O(log n) in
  expectation, with fix-ups counted), the exact MILP optimum and a seeded
  instance generator.
* :mod:`facility_location_greedy` - a greedy heuristic for Uncapacitated
  Facility Location (no proven ratio), the exact MILP optimum and a seeded
  Euclidean instance generator.
* :mod:`tsp_metric_approx` - MST doubling for metric TSP (cost at most
  2 x MST weight <= 2 OPT), Prim's MST, a metric check, and networkx's
  Christofides as a labelled library comparison (3/2 bound).
* :mod:`maxcut_randomized` - the random cut (E[cut] = |E|/2 >= OPT/2), local
  search (every local optimum cuts at least |E|/2 edges) and brute force
  for small graphs.

Every guarantee here is a comparison with a lower bound that takes
polynomial time to compute: the maximal matching, the LP relaxation value,
the MST weight, and the edge count |E| (an upper bound on OPT for the
maximization problem Max-Cut). Dividing a cost by that bound gives a
certified upper bound on the true ratio even when OPT is out of reach.

Both modules with instance generators name theirs ``random_instance``, so
this package exports them as :func:`set_cover_random_instance` and
:func:`facility_location_random_instance`. The LP and MILP solvers live in
:mod:`src.utils.lp_solver` (SciPy's HiGHS). Every random draw goes through
a seeded ``random.Random`` from :mod:`src.utils.random_utils`, so results
are reproducible.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.approximation.facility_location_greedy import (
    facility_location_exact,
    facility_location_greedy,
    total_cost,
)
from src.approximation.facility_location_greedy import (
    random_instance as facility_location_random_instance,
)
from src.approximation.maxcut_randomized import (
    MAX_EXACT_VERTICES,
    cut_value,
    maxcut_exact,
    maxcut_local_search,
    maxcut_random,
)
from src.approximation.set_cover_lpround import (
    frequency,
    set_cover_deterministic_rounding,
    set_cover_exact,
    set_cover_greedy,
    set_cover_ip,
    set_cover_lp,
    set_cover_randomized_rounding,
)
from src.approximation.set_cover_lpround import (
    random_instance as set_cover_random_instance,
)
from src.approximation.tsp_metric_approx import (
    is_metric,
    mst_weight_and_edges,
    tsp_christofides_networkx,
    tsp_mst_doubling,
)
from src.approximation.vertex_cover_2approx import (
    approximation_ratio,
    matching_lower_bound,
    vertex_cover_2approx,
    vertex_cover_exact,
    vertex_cover_lp,
    vertex_cover_lp_rounding,
)

__all__ = [
    "vertex_cover_2approx", "matching_lower_bound", "vertex_cover_lp", "vertex_cover_lp_rounding",
    "vertex_cover_exact", "approximation_ratio", "set_cover_ip", "set_cover_lp", "set_cover_exact",
    "set_cover_greedy", "set_cover_deterministic_rounding", "set_cover_randomized_rounding",
    "frequency", "set_cover_random_instance", "total_cost", "facility_location_greedy",
    "facility_location_exact", "facility_location_random_instance", "mst_weight_and_edges",
    "tsp_mst_doubling", "tsp_christofides_networkx", "is_metric", "cut_value", "maxcut_random",
    "maxcut_local_search", "maxcut_exact", "MAX_EXACT_VERTICES",
]
