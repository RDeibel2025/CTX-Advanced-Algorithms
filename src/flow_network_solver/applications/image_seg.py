"""Foreground-background image segmentation as a minimum s-t cut.

Each pixel of a grayscale image has to be labelled foreground or background.
Two wishes pull against each other: a bright pixel should be foreground and a
dark one background, and neighbouring pixels of similar intensity should get the
same label. Segmentation by graph cut writes both as edges of one network:

* one node per pixel, plus a source s and a sink t,
* s -> pixel with capacity I, the intensity,
* pixel -> t with capacity 255 - I,
* for each pair of 4-neighbours i and j, two directed edges i -> j and j -> i of
  capacity ``lam * exp(-(Ii - Ij)^2 / (2 sigma^2))``.

Why a minimum cut is the best labelling
---------------------------------------
Any s-t cut splits the pixels into a source side (foreground) and a sink side
(background), and its capacity is exactly the cost of that labelling:

* a background pixel has its s -> pixel edge cut, costing I, so calling a bright
  pixel background is expensive,
* a foreground pixel has its pixel -> t edge cut, costing 255 - I, so calling a
  dark pixel foreground is expensive,
* a neighbouring pair split by the cut pays its weight once, for the one
  directed edge going from the source side to the sink side. That weight is near
  ``lam`` for similar pixels and near 0 across a strong edge, so the boundary
  settles where the intensity jumps.

A minimum cut is therefore a labelling of least total cost, and by the max-flow
min-cut theorem one max-flow run finds it.

How it reads the residual graph
-------------------------------
The solver leaves its final residual graph in the network, and the segmentation
is read from that one structure: the foreground is the set of pixels still
reachable from s through edges with residual capacity
(:func:`~src.flow_network_solver.core.network.min_cut`). That set is the same for
every maximum flow, so both solvers give the same mask. The flow value is the
cut capacity, so the minimized cost needs no second computation.

Integer capacities
------------------
:class:`~src.flow_network_solver.core.network.Network` capacities are integers,
so every capacity is multiplied by ``SCALE = 1000`` and rounded to an integer,
and the cut value reported is the flow divided by ``SCALE``. This keeps three
decimal places of each capacity, so a neighbour weight below 0.0005 rounds to 0.

================= ========================== ==========
Function          Time                       Space
================= ========================== ==========
segment           O(P^3) worst case          O(P)
save_segmentation O(P) plus rendering        O(P)
================= ========================== ==========

For P = rows * cols pixels, the network has V = P + 2 nodes and E = O(P) edges,
so both solvers are bounded by O(P^3). Real images run far below that bound.

Examples:
    >>> image = [[200, 210, 40, 30], [190, 205, 35, 20], [195, 60, 25, 15]]
    >>> mask, cut = segment(image, lam=50, sigma=30)
    >>> mask
    [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
    >>> round(cut, 3)
    500.002

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.network import Network, min_cut
from src.flow_network_solver.core.push_relabel import push_relabel

__all__ = ["SCALE", "save_segmentation", "segment"]

#: Network capacities are integers: each capacity is multiplied by this and rounded.
SCALE = 1000

#: The max-flow solvers a caller can pick by name.
_SOLVERS = {"ek": edmonds_karp, "pr": push_relabel}


def _as_pixels(image: Any) -> np.ndarray:
    """Return a grayscale image as a 2-D float array, checking its shape and range.

    Converting to float first also protects the neighbour differences from
    unsigned wraparound on ``uint8`` input.

    Args:
        image: A 2-D list or numpy array of values in 0-255.

    Returns:
        A float ``numpy`` array with two dimensions.

    Raises:
        ValueError: If the image is ragged, not 2-D, empty, or has a value that
            is outside 0-255 or not a number.

    Time Complexity:
        O(P) for P pixels.

    Space Complexity:
        O(P).

    Examples:
        >>> _as_pixels([[0, 255]]).tolist()
        [[0.0, 255.0]]
        >>> _as_pixels([1, 2])
        Traceback (most recent call last):
            ...
        ValueError: image must be a non-empty 2-D array, got shape (2,)
        >>> _as_pixels([[256]])
        Traceback (most recent call last):
            ...
        ValueError: image values must be in 0-255
    """
    pixels = np.asarray(image, dtype=float)
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError(f"image must be a non-empty 2-D array, got shape {pixels.shape}")
    if not np.all((pixels >= 0) & (pixels <= 255)):
        raise ValueError("image values must be in 0-255")
    return pixels


def segment(
    image: Any, lam: float = 50.0, sigma: float = 30.0, solver: str = "pr"
) -> Tuple[List[List[int]], float]:
    """Split a grayscale image into foreground and background by a minimum cut.

    Builds the network from the module docstring with every capacity scaled by
    ``SCALE = 1000`` and rounded to an integer, solves it, and reads the mask
    from the final residual graph: a pixel is foreground exactly when it is on
    the source side of the cut, meaning still reachable from s
    (:func:`~src.flow_network_solver.core.network.min_cut`).

    Args:
        image: A 2-D list or numpy array of grayscale values in 0-255.
        lam: The smoothness weight, at least 0. Larger values give fewer and
            larger regions.
        sigma: The intensity difference that counts as a real edge, above 0.
        solver: ``"pr"`` for push-relabel or ``"ek"`` for Edmonds-Karp.

    Returns:
        ``(mask, cut_value)``. ``mask`` is a list of lists of ints, 1 for
        foreground and 0 for background, shaped like the image. ``cut_value`` is
        the flow divided by ``SCALE``, a float.

    Raises:
        ValueError: If ``solver`` is unknown, ``lam`` is negative or not finite,
            ``sigma`` is not positive and finite, or the image fails the checks
            of :func:`_as_pixels`.

    Time Complexity:
        O(P^3) worst case for P pixels with either solver. Building the network
        and reading the mask are O(P).

    Space Complexity:
        O(P): P + 2 nodes and at most 2 + 4 edges per pixel.

    Examples:
        A single pixel keeps the cheaper of its two labels. Foreground costs
        255 - 200 = 55 and background costs 200.

        >>> segment([[200]])
        ([[1]], 55.0)
        >>> segment([[50]], solver="ek")
        ([[0]], 50.0)

        A dark pixel between bright ones joins them when ``lam`` is large enough.
        Keeping it dark splits two neighbour pairs, about 2 * 75 = 150 at sigma
        1000, while making it foreground raises its own label cost by only
        155 - 100 = 55.

        >>> segment([[200, 100, 200]], lam=0)[0]
        [[1, 0, 1]]
        >>> segment([[200, 100, 200]], lam=75, sigma=1000)[0]
        [[1, 1, 1]]
        >>> segment([[1, 2]], sigma=0)
        Traceback (most recent call last):
            ...
        ValueError: sigma must be positive and finite, got 0
    """
    if solver not in _SOLVERS:
        raise ValueError(f"unknown solver {solver!r}, expected 'ek' or 'pr'")
    if not (math.isfinite(lam) and lam >= 0):
        raise ValueError(f"lam must be non-negative and finite, got {lam}")
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be positive and finite, got {sigma}")
    pixels = _as_pixels(image)
    rows, cols = pixels.shape
    values = pixels.ravel().tolist()
    s, t = rows * cols, rows * cols + 1
    net = Network(rows * cols + 2)
    for pixel, intensity in enumerate(values):
        net.add_edge(s, pixel, round(intensity * SCALE))
        net.add_edge(pixel, t, round((255 - intensity) * SCALE))

    def link(a: int, b: int) -> None:
        """Join two neighbouring pixels with both directed edges."""
        difference = values[a] - values[b]
        weight = round(SCALE * lam * math.exp(-(difference**2) / (2 * sigma**2)))
        net.add_edge(a, b, weight)
        net.add_edge(b, a, weight)

    for r in range(rows):
        for c in range(cols):
            pixel = r * cols + c
            if c + 1 < cols:
                link(pixel, pixel + 1)
            if r + 1 < rows:
                link(pixel, pixel + cols)

    flow_value = _SOLVERS[solver](net, s, t).value
    source_side = min_cut(net, s)
    mask = [[int(r * cols + c in source_side) for c in range(cols)] for r in range(rows)]
    return mask, flow_value / SCALE


def save_segmentation(image: Any, mask: Any, out_path: str) -> str:
    """Save the original image and its mask side by side as a PNG.

    The parent folder of ``out_path`` is created if needed. Plotting imports stay
    inside this function so that solving never loads matplotlib.

    Args:
        image: The grayscale image, as accepted by :func:`segment`.
        mask: The 0/1 mask with the same shape, as returned by :func:`segment`.
        out_path: Where to save the PNG.

    Returns:
        ``out_path``, unchanged.

    Raises:
        ValueError: If the image fails the checks of :func:`_as_pixels`, or the
            mask has a different shape.

    Time Complexity:
        O(P) plus rendering.

    Space Complexity:
        O(P).

    Examples:
        >>> import os, tempfile
        >>> image = [[200, 40], [190, 30]]
        >>> mask, _ = segment(image)
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     out = os.path.join(folder, "segmentation.png")
        ...     save_segmentation(image, mask, out) == out
        ...     os.path.getsize(out) > 0
        True
        True
        >>> save_segmentation(image, [[1, 0]], "unused.png")
        Traceback (most recent call last):
            ...
        ValueError: mask shape (1, 2) does not match image shape (2, 2)
    """
    from matplotlib.figure import Figure

    from src.utils.visualization import apply_house_style

    pixels = _as_pixels(image)
    labels = np.asarray(mask)
    if labels.shape != pixels.shape:
        raise ValueError(f"mask shape {labels.shape} does not match image shape {pixels.shape}")

    apply_house_style()
    fig = Figure(figsize=(8.0, 4.0))
    original_ax, mask_ax = fig.subplots(1, 2)
    original_ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    original_ax.set_title("Original")
    mask_ax.imshow(labels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    mask_ax.set_title("Mask (white = foreground)")
    original_ax.axis("off")
    mask_ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    return out_path
