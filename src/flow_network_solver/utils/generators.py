"""Seeded instances for the Week 10 flow solvers, applications and benchmark.

:func:`random_network` builds the residual network the solvers read. A purely
random graph makes a poor benchmark, because the maximum flow is then limited by
the few edges that happen to touch s and t, and every solver finishes after a
handful of paths. So the network is built in three steps, in this order:

1. A random simple backbone path from 0 through up to 8 interior nodes to
   n - 1, so an s-t path always exists.
2. Hub edges: ``k = max(1, round(terminal_fraction * n))`` distinct interior
   nodes get an edge from 0, and k distinct interior nodes get an edge to
   n - 1. This acts like a super source and super sink, so the bottleneck sits
   inside the graph and the solvers have real work to do.
3. The remaining edges uniformly at random, never leaving n - 1 or entering 0.
   Parallel and antiparallel edges may occur, and each gets its own residual
   pair.

All randomness comes from one :func:`src.utils.random_utils.make_rng` generator
per call, so a seed fixes the whole instance. The other generators supply the
applications: :func:`bipartite_graph` for matching, :func:`grid_image` for
segmentation, and :func:`capacity_distribution` for the capacity experiment.

===================== ============ ================
Function              Time         Space
===================== ============ ================
random_network        O(n + m)     O(n + m)
sparse                O(n)         O(n)
dense                 O(n^2)       O(n^2)
bipartite_graph       O(L R)       O(L + R + edges)
grid_image            O(rows cols) O(rows cols)
capacity_distribution O(size)      O(size)
===================== ============ ================

Examples:
    >>> net = sparse(100, 7)
    >>> net.n, net.m
    (100, 400)
    >>> net.to == sparse(100, 7).to
    True
    >>> capacity_distribution("unit", 3, 1)
    [1, 1, 1]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from src.flow_network_solver.core.network import Network
from src.utils.random_utils import make_rng

if TYPE_CHECKING:
    from random import Random

__all__ = [
    "bipartite_graph",
    "capacity_distribution",
    "dense",
    "grid_image",
    "random_network",
    "sparse",
]

#: The backbone path visits at most this many interior nodes.
_MAX_BACKBONE_INTERIOR = 8

#: The capacity distributions by name.
_CAP_DISTS = ("uniform", "exponential", "unit")

#: Default uniform capacity range.
_UNIFORM_RANGE = (1, 100)

#: Mean of the exponential capacity distribution.
_EXPONENTIAL_MEAN = 20

#: Grayscale levels and noise of the disc image.
_DISC_LEVEL = 200
_BACKGROUND_LEVEL = 50
_NOISE_SD = 10.0


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` if it is an ``int`` of at least ``minimum``.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.
        minimum: The smallest value allowed.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int`` (``bool`` rejected).
        ValueError: If ``value`` is below ``minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(3, "n", 2)
        3
        >>> _require_int(1, "n", 2)
        Traceback (most recent call last):
            ...
        ValueError: n must be at least 2, got 1
        >>> _require_int(2.0, "n", 2)
        Traceback (most recent call last):
            ...
        TypeError: n must be an int, got float
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _draw_capacities(
    name: str, size: int, rng: Random, uniform_range: Tuple[int, int] = _UNIFORM_RANGE
) -> List[int]:
    """Draw ``size`` capacities from a named distribution using ``rng``.

    Args:
        name: ``"uniform"``, ``"exponential"`` or ``"unit"``.
        size: How many capacities to draw.
        rng: The generator to draw from.
        uniform_range: The inclusive range for ``"uniform"``.

    Returns:
        A new list of ``size`` positive ints.

    Raises:
        ValueError: If ``name`` is unknown.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size).

    Examples:
        >>> _draw_capacities("unit", 2, make_rng(1))
        [1, 1]
        >>> _draw_capacities("uniform", 2, make_rng(1), (5, 5))
        [5, 5]
        >>> _draw_capacities("zipf", 2, make_rng(1))
        Traceback (most recent call last):
            ...
        ValueError: unknown capacity distribution 'zipf'
    """
    if name == "uniform":
        low, high = uniform_range
        return [rng.randint(low, high) for _ in range(size)]
    if name == "exponential":
        rate = 1 / _EXPONENTIAL_MEAN
        return [max(1, round(rng.expovariate(rate))) for _ in range(size)]
    if name == "unit":
        return [1] * size
    raise ValueError(f"unknown capacity distribution {name!r}")


def random_network(
    n: int,
    avg_degree: float,
    seed: Optional[int],
    cap_range: Tuple[int, int] = _UNIFORM_RANGE,
    cost_range: Optional[Tuple[int, int]] = None,
    *,
    cap_dist: str = "uniform",
    terminal_fraction: float = 0.05,
) -> Network:
    """Build a random flow network with source 0 and sink n - 1.

    It has exactly ``m = round(avg_degree * n)`` directed edges, built as the
    backbone path, then the hub edges, then uniform random edges (see the
    module docstring). Small graphs where ``m`` cannot hold the full backbone
    and hub edges use a shorter backbone and fewer hub edges, so ``m`` stays
    exact and the s-t path always exists. Capacities are drawn after the
    topology and costs after the capacities, all from one generator.

    Args:
        n: Number of nodes, at least 2.
        avg_degree: Average out-degree; ``m`` must round to at least 1.
        seed: Passed to :func:`make_rng`.
        cap_range: Inclusive ``(low, high)`` for ``cap_dist="uniform"``, with
            ``low >= 1`` so the backbone is always a usable path. Ignored by
            the other distributions.
        cost_range: Inclusive ``(low, high)`` for uniform integer costs, or
            ``None`` for all costs 0.
        cap_dist: ``"uniform"``, ``"exponential"`` or ``"unit"``.
        terminal_fraction: Sets ``k``, the hub edges per terminal, as above.

    Returns:
        A new :class:`Network` with ``n`` nodes and ``m`` forward edges and no
        self-loops, no edge leaving n - 1 and no edge entering 0.

    Raises:
        TypeError: If ``n`` is not an ``int`` or ``seed`` is not an ``int`` or
            ``None``.
        ValueError: If ``n < 2``, ``m < 1``, ``cap_dist`` is unknown,
            ``terminal_fraction`` is outside [0, 1], or a range is invalid.

    Time Complexity:
        O(n + m).

    Space Complexity:
        O(n + m).

    Examples:
        >>> net = random_network(50, 4, 3, cost_range=(1, 20))
        >>> net.n, net.m, min(net.cost[::2]) >= 1, max(net.cap[::2]) <= 100
        (50, 200, True, True)
        >>> a = random_network(50, 4, 3, cap_dist="exponential")
        >>> b = random_network(50, 4, 3, cap_dist="exponential")
        >>> a.to == b.to and a.cap == b.cap, min(a.cap[::2]) >= 1
        (True, True)
        >>> set(random_network(50, 4, 3, cap_dist="unit").cap[::2])
        {1}
        >>> random_network(2, 0.5, 0).to
        [1, 0]
        >>> random_network(1, 4, 0)
        Traceback (most recent call last):
            ...
        ValueError: n must be at least 2, got 1
        >>> random_network(10, 0, 0)
        Traceback (most recent call last):
            ...
        ValueError: avg_degree 0 gives 0 edges; at least 1 is needed
        >>> random_network(10, 4, 0, cap_dist="zipf")
        Traceback (most recent call last):
            ...
        ValueError: unknown capacity distribution 'zipf'
    """
    _require_int(n, "n", 2)
    m = round(avg_degree * n)
    if m < 1:
        raise ValueError(f"avg_degree {avg_degree} gives {m} edges; at least 1 is needed")
    if cap_dist not in _CAP_DISTS:
        raise ValueError(f"unknown capacity distribution {cap_dist!r}")
    if not 0 <= terminal_fraction <= 1:
        raise ValueError(f"terminal_fraction must be in [0, 1], got {terminal_fraction}")
    if not 1 <= cap_range[0] <= cap_range[1]:
        raise ValueError(f"cap_range must satisfy 1 <= low <= high, got {cap_range}")
    if cost_range is not None and cost_range[0] > cost_range[1]:
        raise ValueError(f"cost_range must satisfy low <= high, got {cost_range}")

    rng = make_rng(seed)
    sink = n - 1
    interior = range(1, sink)

    # Step 1: the backbone. At most m - 1 interior nodes keeps it within m edges.
    stops = rng.sample(interior, min(_MAX_BACKBONE_INTERIOR, len(interior), m - 1))
    path = [0, *stops, sink]
    edges = list(zip(path, path[1:]))

    # Step 2: the hub edges, cut short only when m is too small to hold them.
    k = min(max(1, round(terminal_fraction * n)), len(interior))
    hubs = [(0, v) for v in rng.sample(interior, k)]
    hubs += [(u, sink) for u in rng.sample(interior, k)]
    edges += hubs[: m - len(edges)]

    # Step 3: uniform over the allowed pairs, by rejecting u == v.
    randrange = rng.randrange
    while len(edges) < m:
        u = randrange(sink)
        v = randrange(1, n)
        if u != v:
            edges.append((u, v))

    capacities = _draw_capacities(cap_dist, m, rng, cap_range)
    if cost_range is None:
        costs = [0] * m
    else:
        low, high = cost_range
        costs = [rng.randint(low, high) for _ in range(m)]

    net = Network(n)
    for (u, v), capacity, cost in zip(edges, capacities, costs):
        net.add_edge(u, v, capacity, cost)
    return net


def sparse(n: int, seed: Optional[int], **kw: Any) -> Network:
    """Build a sparse network with average out-degree 4.

    Args:
        n: Number of nodes.
        seed: Passed to :func:`random_network`.
        **kw: Any other :func:`random_network` option.

    Returns:
        ``random_network(n, 4, seed, **kw)``, with ``4 * n`` edges.

    Raises:
        TypeError: As in :func:`random_network`.
        ValueError: As in :func:`random_network`.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> sparse(20, 1).m
        80
        >>> sparse(20, 1, cost_range=(2, 2)).cost[0]
        2
    """
    return random_network(n, 4, seed, **kw)


def dense(n: int, seed: Optional[int], **kw: Any) -> Network:
    """Build a dense network with average out-degree n / 4.

    Args:
        n: Number of nodes.
        seed: Passed to :func:`random_network`.
        **kw: Any other :func:`random_network` option.

    Returns:
        ``random_network(n, n / 4, seed, **kw)``, with about ``n^2 / 4`` edges.

    Raises:
        TypeError: As in :func:`random_network`.
        ValueError: As in :func:`random_network`.

    Time Complexity:
        O(n^2).

    Space Complexity:
        O(n^2).

    Examples:
        >>> dense(20, 1).m
        100
    """
    return random_network(n, n / 4, seed, **kw)


def bipartite_graph(
    n_left: int, n_right: int, p: float, seed: Optional[int]
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build a random bipartite graph for the matching application.

    Every left-right pair is kept independently with probability ``p``.

    Args:
        n_left: Number of left nodes, non-negative.
        n_right: Number of right nodes, non-negative.
        p: The edge probability, in [0, 1].
        seed: Passed to :func:`make_rng`.

    Returns:
        ``(left, right, edges)``: labels ``["L0", ...]`` and ``["R0", ...]``,
        and ``edges`` a list of ``(l, r)`` pairs in row order.

    Raises:
        TypeError: If a size is not an ``int``, or ``seed`` is not an ``int`` or
            ``None``.
        ValueError: If a size is negative or ``p`` is outside [0, 1].

    Time Complexity:
        O(n_left * n_right), one draw per pair.

    Space Complexity:
        O(n_left + n_right + edges).

    Examples:
        >>> bipartite_graph(2, 1, 1.0, 5)
        (['L0', 'L1'], ['R0'], [('L0', 'R0'), ('L1', 'R0')])
        >>> bipartite_graph(2, 2, 0.0, 5)[2]
        []
        >>> bipartite_graph(3, 3, 0.5, 5) == bipartite_graph(3, 3, 0.5, 5)
        True
        >>> bipartite_graph(1, 1, 1.5, 5)
        Traceback (most recent call last):
            ...
        ValueError: p must be in [0, 1], got 1.5
    """
    _require_int(n_left, "n_left", 0)
    _require_int(n_right, "n_right", 0)
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    rng = make_rng(seed)
    left = [f"L{i}" for i in range(n_left)]
    right = [f"R{j}" for j in range(n_right)]
    edges = [(a, b) for a in left for b in right if rng.random() < p]
    return left, right, edges


def grid_image(rows: int, cols: int, seed: Optional[int]) -> List[List[int]]:
    """Build the synthetic segmentation image: a bright disc on a dark field.

    A disc of level 200, centred with radius ``0.3 * min(rows, cols)``, on a
    background of level 50. Each pixel gets Gaussian noise with standard
    deviation 10, then is rounded and clipped to 0-255.

    Args:
        rows: Image height, at least 1.
        cols: Image width, at least 1.
        seed: Passed to :func:`make_rng`.

    Returns:
        A list of ``rows`` lists of ``cols`` ints in 0-255.

    Raises:
        TypeError: If a size is not an ``int``, or ``seed`` is not an ``int`` or
            ``None``.
        ValueError: If a size is below 1.

    Time Complexity:
        O(rows * cols).

    Space Complexity:
        O(rows * cols).

    Examples:
        >>> image = grid_image(32, 32, 1)
        >>> len(image), len(image[0])
        (32, 32)
        >>> image[16][16] > 150 > 100 > image[0][0]
        True
        >>> image == grid_image(32, 32, 1)
        True
        >>> grid_image(0, 3, 1)
        Traceback (most recent call last):
            ...
        ValueError: rows must be at least 1, got 0
    """
    _require_int(rows, "rows", 1)
    _require_int(cols, "cols", 1)
    rng = make_rng(seed)
    centre_row, centre_col = (rows - 1) / 2, (cols - 1) / 2
    radius_squared = (0.3 * min(rows, cols)) ** 2
    image = []
    for r in range(rows):
        row = []
        for c in range(cols):
            inside = (r - centre_row) ** 2 + (c - centre_col) ** 2 <= radius_squared
            level = (_DISC_LEVEL if inside else _BACKGROUND_LEVEL) + rng.gauss(0.0, _NOISE_SD)
            row.append(min(255, max(0, round(level))))
        image.append(row)
    return image


def capacity_distribution(name: str, size: int, seed: Optional[int]) -> List[int]:
    """Draw a list of capacities from a named distribution.

    ``"uniform"`` gives integers 1-100, ``"exponential"`` gives
    ``max(1, round(x))`` with x exponential of mean 20, and ``"unit"`` gives all
    1. Unit capacities are the classic hard case for counting paths, since each
    augmentation can move only one unit.

    Args:
        name: ``"uniform"``, ``"exponential"`` or ``"unit"``.
        size: How many capacities to draw, non-negative.
        seed: Passed to :func:`make_rng`.

    Returns:
        A new list of ``size`` positive ints.

    Raises:
        TypeError: If ``size`` is not an ``int``, or ``seed`` is not an ``int``
            or ``None``.
        ValueError: If ``name`` is unknown or ``size`` is negative.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size).

    Examples:
        >>> capacity_distribution("unit", 4, 1)
        [1, 1, 1, 1]
        >>> values = capacity_distribution("uniform", 100, 1)
        >>> 1 <= min(values) and max(values) <= 100
        True
        >>> min(capacity_distribution("exponential", 100, 1)) >= 1
        True
        >>> capacity_distribution("zipf", 4, 1)
        Traceback (most recent call last):
            ...
        ValueError: unknown capacity distribution 'zipf'
    """
    _require_int(size, "size", 0)
    return _draw_capacities(name, size, make_rng(seed))
