"""Seeded instance generators and flow validators (Week 10).

Two modules:

* :mod:`generators` - :func:`random_network` with its :func:`sparse` and
  :func:`dense` forms, :func:`bipartite_graph` for matching, :func:`grid_image`
  for segmentation and :func:`capacity_distribution` for the capacity experiment.
  Every random draw goes through a seeded ``random.Random`` from
  :mod:`src.utils.random_utils`, so results are reproducible.
* :mod:`validators` - independent checks on the residual capacities a solver
  leaves behind. Each returns ``(ok, message)``, and :func:`validate_flow` runs
  them all.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.flow_network_solver.utils.generators import (
    bipartite_graph,
    capacity_distribution,
    dense,
    grid_image,
    random_network,
    sparse,
)
from src.flow_network_solver.utils.validators import (
    check_capacity,
    check_conservation,
    check_flow_value,
    check_min_cost_feasible,
    check_residual,
    validate_flow,
)

__all__ = [
    "random_network", "sparse", "dense", "bipartite_graph", "grid_image",
    "capacity_distribution", "check_capacity", "check_conservation", "check_residual",
    "check_flow_value", "check_min_cost_feasible", "validate_flow",
]
