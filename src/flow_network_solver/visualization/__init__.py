"""Static network plots and flow animations (Week 10).

Two modules:

* :mod:`plot_network` - :func:`plot_network` draws a network with
  ``flow/capacity`` labels, its min cut dashed and a path bold.
  :func:`save_network_figure` saves a solved network with its min cut, and
  :func:`layered_layout` places nodes in columns by BFS depth from the source.
* :mod:`animate_flow` - :func:`animate_edmonds_karp` shows one frame per
  augmenting path and :func:`animate_push_relabel` places nodes by height with
  their excess. Both save a GIF of at most 40 frames with ``PillowWriter``.

Nothing here solves on its own terms: the plots read what a solver left in the
residual graph, and the animations replay a recorded run. networkx is used for
node layout only, never as a solver.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.flow_network_solver.visualization.animate_flow import (
    animate_edmonds_karp,
    animate_push_relabel,
)
from src.flow_network_solver.visualization.plot_network import (
    layered_layout,
    plot_network,
    save_network_figure,
)

__all__ = [
    "plot_network", "save_network_figure", "layered_layout", "animate_edmonds_karp",
    "animate_push_relabel",
]
