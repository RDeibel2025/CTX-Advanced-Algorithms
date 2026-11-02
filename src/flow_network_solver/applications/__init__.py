"""Applications that read the final residual graph (Week 10).

Three modules, one reduction to maximum flow each:

* :mod:`matching` - maximum bipartite matching on a unit-capacity network. The
  matching is the set of saturated left-right edges. :func:`draw_matching` saves
  the picture.
* :mod:`image_seg` - foreground-background segmentation as a minimum s-t cut. The
  foreground is the set of pixels still reachable from s. Network capacities are
  integers, so capacities are scaled by :data:`SCALE` and rounded.
  :func:`save_segmentation` saves the image beside its mask.
* :mod:`scheduling` - jobs on capacity-limited machines, and project staffing
  through the same network. The assignment is the set of job-machine edges that
  carry flow.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.flow_network_solver.applications.image_seg import SCALE, save_segmentation, segment
from src.flow_network_solver.applications.matching import bipartite_matching, draw_matching
from src.flow_network_solver.applications.scheduling import schedule

__all__ = [
    "bipartite_matching", "draw_matching", "SCALE", "segment", "save_segmentation", "schedule",
]
