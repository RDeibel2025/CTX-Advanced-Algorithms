"""Core: the residual network and the solvers that read it (Week 10).

Four modules, one job each:

* :mod:`network` - the array-based :class:`Network`, where edge ``i`` and edge
  ``i ^ 1`` are a forward-reverse pair, the :class:`FlowResult` every solver
  returns, and the cut readers :func:`min_cut`, :func:`cut_edges` and
  :func:`cut_value`.
* :mod:`edmonds_karp` - repeated shortest augmenting paths found by BFS,
  O(V E^2). It reads whole s-t paths from the residual graph.
* :mod:`push_relabel` - FIFO push-relabel with the gap heuristic and global
  relabels, O(V^3). It reads heights and excess locally, never a whole path.
* :mod:`min_cost_flow` - successive shortest paths with Johnson potentials. It
  reads the residual graph with costs, keeping reduced costs non-negative so
  Dijkstra applies.

Every solver mutates ``cap`` in place and leaves the final residual graph in the
network, so all readers of the answer work the same way after any solver. Call
:meth:`Network.reset` before running another solver on the same graph.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.min_cost_flow import min_cost_max_flow
from src.flow_network_solver.core.network import (
    FlowResult,
    Network,
    cut_edges,
    cut_value,
    min_cut,
)
from src.flow_network_solver.core.push_relabel import push_relabel

__all__ = [
    "FlowResult", "Network", "min_cut", "cut_edges", "cut_value", "edmonds_karp", "push_relabel",
    "min_cost_max_flow",
]
