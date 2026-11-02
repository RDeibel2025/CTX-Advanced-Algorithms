"""Week 10: network flows, matching and large-scale graph optimization.

This package is the instructions' ``FlowNetworkSolver/`` project, and it sits
under ``src/`` because the repository holds every week's project.

Four sub-packages, one structure. Every algorithm and application here works on
the residual graph and differs only in what it reads from it:

* :mod:`~src.flow_network_solver.core` - the array-based network and the solvers.
  Edmonds-Karp reads the shortest s-t path, push-relabel reads heights and excess
  locally, and min-cost flow reads the graph with costs through potentials.
* :mod:`~src.flow_network_solver.applications` - matching reads the saturated
  edges, segmentation reads which pixels are still reachable from s, and
  scheduling reads which job-machine edges carry flow.
* :mod:`~src.flow_network_solver.utils` - seeded generators and the validators,
  which check a flow from the residual capacities alone.
* :mod:`~src.flow_network_solver.visualization` - network pictures and animations.

The public names below load from their sub-package on first use, so a solve
never imports matplotlib or networkx just because this package was imported.
:mod:`~src.flow_network_solver.main` is the command line and is not re-exported.

Examples:
    >>> from src.flow_network_solver import Network, edmonds_karp
    >>> edmonds_karp(Network.from_edges(2, [(0, 1, 5)]), 0, 1).value
    5

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each sub-package.
_EXPORTS = {
    "core": (
        "FlowResult", "Network", "cut_edges", "cut_value", "edmonds_karp", "min_cost_max_flow",
        "min_cut", "push_relabel",
    ),
    "applications": (
        "SCALE", "bipartite_matching", "draw_matching", "save_segmentation", "schedule", "segment",
    ),
    "utils": (
        "bipartite_graph", "capacity_distribution", "check_capacity", "check_conservation",
        "check_flow_value", "check_min_cost_feasible", "check_residual", "dense", "grid_image",
        "random_network", "sparse", "validate_flow",
    ),
    "visualization": (
        "animate_edmonds_karp", "animate_push_relabel", "layered_layout", "plot_network",
        "save_network_figure",
    ),
}

#: Each public name mapped to the sub-package that provides it.
_HOME = {name: package for package, names in _EXPORTS.items() for name in names}

__all__ = list(_HOME)


def __getattr__(name: str) -> Any:
    """Load a public name from its sub-package on first use.

    Python calls this only for a name the module does not already hold.

    Args:
        name: The attribute being looked up.

    Returns:
        The object of that name from its sub-package.

    Raises:
        AttributeError: If ``name`` is not a public name of this package.

    Time Complexity:
        O(1), plus the sub-package import the first time.

    Space Complexity:
        O(1).

    Examples:
        >>> import src.flow_network_solver as package
        >>> package.push_relabel.__name__
        'push_relabel'
        >>> package.dinic
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.flow_network_solver' has no attribute 'dinic'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
