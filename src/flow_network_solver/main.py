"""Command line for the Week 10 flow network solver.

Run it from the repository root with exactly one mode flag::

    python -m src.flow_network_solver.main --ek --input FILE
    python -m src.flow_network_solver.main --pr --input FILE --plot --out DIR
    python -m src.flow_network_solver.main --mcf --input FILE
    python -m src.flow_network_solver.main --match
    python -m src.flow_network_solver.main --schedule
    python -m src.flow_network_solver.main --segment --input IMAGE --plot

What each mode reads from the residual graph
--------------------------------------------
This file solves nothing itself. Each mode runs one algorithm from the package,
and that algorithm works on, and leaves behind, one residual graph:

========== ================== =============================================
Flag       Runs               Reads from the residual graph
========== ================== =============================================
--ek       Edmonds-Karp       the shortest s-t path, until none is left
--pr       push-relabel       heights and excess, one node at a time
--mcf      min-cost max-flow  the cheapest s-t path, through potentials
--match    bipartite_matching the saturated left-right edges
--schedule schedule           the job-machine edges that carry flow
--segment  segment            the pixels still reachable from s
========== ================== =============================================

Input
-----
For the three solvers ``--input`` names a plain text graph: a header line
``n m s t``, then exactly m lines ``u v cap [cost]``. Blank lines are skipped.
For ``--segment`` it names a grayscale ``.png`` or a ``.csv`` of values in 0-255.
Without ``--input`` each mode runs a built-in example: the two files in
``src/flow_network_solver/examples/`` for the solvers, and the anchors of the
build spec for the applications. Matching and scheduling have no input format,
so they reject ``--input`` rather than ignore a file the caller named.

Validation
----------
The solver modes print the result of
:func:`~src.flow_network_solver.utils.validators.validate_flow`. That function
also checks for a negative-cost cycle, which is the right test for ``--mcf`` but
would fail a correct maximum flow from ``--ek`` or ``--pr`` whenever it is not
also the cheapest one. Those two solvers ignore costs, so they run with the costs
set to 0, and a note says so.

An application keeps its network to itself, so ``validate_flow`` cannot reach
it. The application modes check the answer directly instead: a matching uses
input edges and no node twice, a schedule keeps to the allowed machines and their
capacities, and a mask describes a cut whose capacity equals the flow.

================= ==================== ==========
Function          Time                 Space
================= ==================== ==========
read_network      O(V + E)             O(V + E)
read_image        O(P) for P pixels    O(P)
main              that of the mode run O(V + E)
================= ==================== ==========

Examples:
    >>> main(["--ek"])  # doctest: +ELLIPSIS
    solver: Edmonds-Karp
    ...
    max flow value: 23
    ...
    0
    >>> main(["--mcf"])  # doctest: +ELLIPSIS
    solver: Min-cost max-flow
    ...
    max flow value: 6
    minimum cost: 24
    ...
    0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.flow_network_solver.applications.image_seg import SCALE, save_segmentation, segment
from src.flow_network_solver.applications.matching import bipartite_matching, draw_matching
from src.flow_network_solver.applications.scheduling import schedule
from src.flow_network_solver.core.edmonds_karp import edmonds_karp
from src.flow_network_solver.core.min_cost_flow import min_cost_max_flow
from src.flow_network_solver.core.network import Network
from src.flow_network_solver.core.push_relabel import push_relabel
from src.flow_network_solver.utils.validators import validate_flow

__all__ = ["main", "read_image", "read_network"]

#: The shipped sample inputs, which double as the solvers' built-in examples.
EXAMPLES = Path(__file__).parent / "examples"

#: Mode name to (printed name, solver, what one iteration is).
_SOLVERS = {
    "ek": ("Edmonds-Karp", edmonds_karp, "augmenting paths"),
    "pr": ("Push-relabel", push_relabel, "pushes + relabels"),
    "mcf": ("Min-cost max-flow", min_cost_max_flow, "augmenting paths"),
}

#: The input each solver mode runs when ``--input`` is not given.
_BUILT_IN_GRAPHS = {
    "ek": EXAMPLES / "clrs_network.txt",
    "pr": EXAMPLES / "clrs_network.txt",
    "mcf": EXAMPLES / "min_cost_example.txt",
}

#: The matching anchor (spec 9.3).
_LEFT = ["a", "b", "c", "d"]
_RIGHT = [1, 2, 3, 4]
_PAIRS = [("a", 1), ("a", 2), ("b", 1), ("c", 2), ("c", 3), ("d", 3)]

#: The scheduling anchor (spec 9.3).
_JOBS = ["J1", "J2", "J3", "J4"]
_MACHINES = ["M1", "M2"]
_ALLOWED = {"J1": ["M1"], "J2": ["M1", "M2"], "J3": ["M1"], "J4": ["M2"]}
_CAPACITIES = {"M1": 2, "M2": 1}

#: The segmentation anchor (spec 9.3) and its parameters.
_IMAGE = [[200, 210, 40, 30], [190, 205, 35, 20], [195, 60, 25, 15]]
_LAM = 50.0
_SIGMA = 30.0


def read_network(path: Union[str, Path]) -> Tuple[Network, int, int]:
    """Read a network from a plain text file.

    The file holds a header line ``n m s t`` followed by exactly m lines
    ``u v cap [cost]``, all integers. Blank lines are skipped, and a cost left
    out is 0. Error messages name the file and the line.

    Args:
        path: The file to read.

    Returns:
        ``(net, s, t)``: the unsolved network, the source and the sink.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a line has the wrong number of fields or a field that is
            not an integer, the edge count differs from the header, or an edge
            fails :meth:`Network.add_edge`.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> net, s, t = read_network(EXAMPLES / "min_cost_example.txt")
        >>> net.n, net.m, s, t
        (4, 5, 0, 3)
        >>> list(net.forward_edges())[1]
        (2, 0, 2, 2, 2)

        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     bad = Path(folder, "bad.txt")
        ...     _ = bad.write_text("2 2 0 1\\n0 1 5\\n")
        ...     read_network(bad)
        Traceback (most recent call last):
            ...
        ValueError: bad.txt: the header says 2 edges, found 1
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     bad = Path(folder, "bad.txt")
        ...     _ = bad.write_text("2 1 0 1\\n0 x 5\\n")
        ...     read_network(bad)
        Traceback (most recent call last):
            ...
        ValueError: bad.txt line 2: expected integers, got '0 x 5'
    """
    name = Path(path).name
    lines = [
        (number, line.split())
        for number, line in enumerate(Path(path).read_text().splitlines(), start=1)
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"{name}: empty file, expected a header line n m s t")

    def integers(number: int, fields: List[str], sizes: Sequence[int], shape: str) -> List[int]:
        """Parse one line, rejecting a wrong field count or a non-integer field."""
        if len(fields) not in sizes:
            raise ValueError(f"{name} line {number}: expected {shape}, got {' '.join(fields)!r}")
        try:
            return [int(field) for field in fields]
        except ValueError:
            raise ValueError(
                f"{name} line {number}: expected integers, got {' '.join(fields)!r}"
            ) from None

    n, m, s, t = integers(*lines[0], (4,), "n m s t")
    edges = [integers(number, fields, (3, 4), "u v cap [cost]") for number, fields in lines[1:]]
    if len(edges) != m:
        raise ValueError(f"{name}: the header says {m} edges, found {len(edges)}")
    return Network.from_edges(n, edges), s, t


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read a grayscale image from a PNG or CSV file.

    A PNG is converted to 8-bit grayscale. A CSV holds one row of comma-separated
    values per image row. :func:`~src.flow_network_solver.applications.image_seg.segment`
    checks that the values lie in 0-255.

    Args:
        path: A ``.png`` or ``.csv`` file.

    Returns:
        A 2-D numpy array of pixel values.

    Raises:
        OSError: If the file cannot be read or is not a valid image.
        ValueError: If the suffix is neither ``.png`` nor ``.csv``, or a CSV
            field is not a number.

    Time Complexity:
        O(P) for P pixels.

    Space Complexity:
        O(P).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     csv_file = Path(folder, "tiny.csv")
        ...     _ = csv_file.write_text("200,40\\n190,30\\n")
        ...     read_image(csv_file).tolist()
        [[200.0, 40.0], [190.0, 30.0]]
        >>> read_image("photo.jpg")
        Traceback (most recent call last):
            ...
        ValueError: photo.jpg: --segment reads a .png or .csv file
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        from PIL import Image

        with Image.open(path) as picture:
            return np.asarray(picture.convert("L"))
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",", ndmin=2)
    raise ValueError(f"{Path(path).name}: --segment reads a .png or .csv file")


def _print_validation(ok: bool, message: str) -> int:
    """Print the validation line and return the matching exit status."""
    print(f"validation: {'ok' if ok else 'FAILED'}, {message}")
    return 0 if ok else 1


def _run_solver(args: argparse.Namespace) -> int:
    """Run ``--ek``, ``--pr`` or ``--mcf`` and print its results."""
    label, solve, unit = _SOLVERS[args.mode]
    path = args.input or _BUILT_IN_GRAPHS[args.mode]
    net, s, t = read_network(path)
    print(f"solver: {label}")
    print(f"input: {path if args.input else Path(path).name + ' (built-in example)'}")
    print(f"network: {net.n} nodes, {net.m} edges, s = {s}, t = {t}")
    if args.mode != "mcf" and any(net.cost):
        net.cost[:] = [0] * len(net.cost)
        print("note: this solver ignores costs, so they are set to 0 (--mcf uses them)")
    result = solve(net, s, t)
    print(f"max flow value: {result.value}")
    if result.cost is not None:
        print(f"minimum cost: {result.cost}")
    print(f"iterations: {result.iterations} ({unit})")
    status = _print_validation(*validate_flow(net, s, t, result.value))
    print(f"time: {result.time_s:.6f} s")
    if args.plot:
        from src.flow_network_solver.visualization.plot_network import save_network_figure

        print(f"saved: {save_network_figure(net, s, t, str(Path(args.out) / 'network.png'))}")
    return status


def _run_match(args: argparse.Namespace) -> int:
    """Run ``--match`` on the built-in graph and print its results."""
    start = time.perf_counter()
    pairs = bipartite_matching(_LEFT, _RIGHT, _PAIRS)
    elapsed = time.perf_counter() - start
    print("application: bipartite matching (built-in example)")
    print(f"matching size: {len(pairs)}")
    print(f"pairs: {', '.join(f'{l}-{r}' for l, r in pairs)}")
    used_once = all(
        len({pair[side] for pair in pairs}) == len(pairs) for side in (0, 1)
    )
    ok = used_once and set(pairs) <= set(_PAIRS)
    status = _print_validation(ok, "input edges only, no node used twice")
    print(f"time: {elapsed:.6f} s")
    if args.plot:
        out_path = str(Path(args.out) / "matching.png")
        print(f"saved: {draw_matching(_LEFT, _RIGHT, _PAIRS, pairs, out_path)}")
    return status


def _run_schedule(args: argparse.Namespace) -> int:
    """Run ``--schedule`` on the built-in jobs and print its results."""
    start = time.perf_counter()
    assignment, unassigned = schedule(_JOBS, _MACHINES, _ALLOWED, _CAPACITIES)
    elapsed = time.perf_counter() - start
    print("application: job scheduling (built-in example)")
    print(f"assigned: {len(assignment)} of {len(_JOBS)} jobs")
    print(f"assignment: {', '.join(f'{j} -> {m}' for j, m in assignment.items())}")
    print(f"unassigned: {', '.join(unassigned) or 'none'}")
    load = Counter(assignment.values())
    ok = all(machine in _ALLOWED[job] for job, machine in assignment.items()) and all(
        load[machine] <= _CAPACITIES[machine] for machine in _MACHINES
    )
    status = _print_validation(ok, "allowed machines only, no machine over capacity")
    print(f"time: {elapsed:.6f} s")
    if args.plot:
        allowed_pairs = [(job, m) for job in _JOBS for m in _ALLOWED[job]]
        out_path = str(Path(args.out) / "schedule.png")
        saved = draw_matching(_JOBS, _MACHINES, allowed_pairs, list(assignment.items()), out_path)
        print(f"saved: {saved}")
    return status


def _cut_capacity(image: Any, mask: Sequence[Sequence[int]]) -> int:
    """Return the capacity of the cut a mask describes, in ``SCALE`` units.

    This prices the labelling by the rule in
    :mod:`~src.flow_network_solver.applications.image_seg`, with the same
    rounding and no solver: 255 - I per foreground pixel, I per background pixel,
    and the neighbour weight for each 4-neighbour pair the mask splits.

    Args:
        image: The grayscale image.
        mask: The 0/1 mask of the same shape.

    Returns:
        The cut capacity as an ``int``, which ``segment`` reports divided by
        ``SCALE``.

    Raises:
        None.

    Time Complexity:
        O(P) for P pixels.

    Space Complexity:
        O(P).

    Examples:
        >>> _cut_capacity([[200, 40]], [[1, 0]])
        95000
    """
    pixels = np.asarray(image, dtype=float).tolist()

    def split(a: float, b: float) -> int:
        """Return the weight of a neighbour pair with intensities a and b."""
        return round(SCALE * _LAM * math.exp(-((a - b) ** 2) / (2 * _SIGMA**2)))

    total = 0
    for r, row in enumerate(pixels):
        for c, intensity in enumerate(row):
            total += round((255 - intensity if mask[r][c] else intensity) * SCALE)
            if c + 1 < len(row) and mask[r][c] != mask[r][c + 1]:
                total += split(intensity, row[c + 1])
            if r + 1 < len(pixels) and mask[r][c] != mask[r + 1][c]:
                total += split(intensity, pixels[r + 1][c])
    return total


def _run_segment(args: argparse.Namespace) -> int:
    """Run ``--segment`` on the given or built-in image and print its results."""
    image = read_image(args.input) if args.input else _IMAGE
    start = time.perf_counter()
    mask, cut = segment(image, lam=_LAM, sigma=_SIGMA)
    elapsed = time.perf_counter() - start
    print("application: image segmentation")
    print(f"input: {args.input or 'built-in example'}")
    pixels = sum(len(row) for row in mask)
    print(f"foreground pixels: {sum(map(sum, mask))} of {pixels}")
    print(f"cut value: {cut:.3f}")
    ok = _cut_capacity(image, mask) == round(cut * SCALE)
    status = _print_validation(ok, "the mask's cut capacity equals the flow")
    print(f"time: {elapsed:.6f} s")
    if args.plot:
        out_path = str(Path(args.out) / "segmentation.png")
        print(f"saved: {save_segmentation(image, mask, out_path)}")
    return status


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, with one required mode flag."""
    parser = argparse.ArgumentParser(
        prog="python -m src.flow_network_solver.main",
        description="Solve a flow network or run a flow application.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag, mode, text in (
        ("--ek", "ek", "maximum flow by Edmonds-Karp"),
        ("--pr", "pr", "maximum flow by FIFO push-relabel"),
        ("--mcf", "mcf", "minimum-cost maximum flow"),
        ("--match", "match", "maximum bipartite matching on a built-in graph"),
        ("--segment", "segment", "foreground-background image segmentation"),
        ("--schedule", "schedule", "job scheduling on a built-in instance"),
    ):
        modes.add_argument(flag, dest="mode", action="store_const", const=mode, help=text)
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="a graph ('n m s t' header, then 'u v cap [cost]' lines), or for "
        "--segment a grayscale .png or .csv; default: a built-in example",
    )
    parser.add_argument("--out", default=".", metavar="DIR",
                        help="folder for the --plot image (default: current directory)")
    parser.add_argument("--plot", action="store_true", help="save a picture of the result")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the command line, run one mode and print its results.

    Each mode prints its value (and the cost for ``--mcf``), iterations for the
    solvers, a validation line and the time. See the module docstring for what
    validation means in each mode. ``--plot`` saves ``network.png``,
    ``matching.png``, ``schedule.png`` or ``segmentation.png`` in ``--out``. The
    schedule picture is drawn by ``draw_matching``, so its title says "Matching".

    Args:
        argv: The arguments, without the program name. ``None`` reads
            ``sys.argv``.

    Returns:
        The exit status: 0 if validation passed, 1 if it failed.

    Raises:
        SystemExit: With status 2 on bad arguments, an unreadable or malformed
            input, or a network the solver rejects, such as one with a negative
            cycle for ``--mcf``. The message goes to stderr.

    Time Complexity:
        That of the mode run, plus O(V + E) to read and validate.

    Space Complexity:
        O(V + E).

    Examples:
        >>> main(["--pr"])  # doctest: +ELLIPSIS
        solver: Push-relabel
        input: clrs_network.txt (built-in example)
        network: 6 nodes, 9 edges, s = 0, t = 5
        max flow value: 23
        iterations: ... (pushes + relabels)
        validation: ok, all checks passed
        time: ... s
        0
        >>> main(["--match"])  # doctest: +ELLIPSIS
        application: bipartite matching (built-in example)
        matching size: 3
        ...
        0
        >>> main(["--schedule"])  # doctest: +ELLIPSIS
        application: job scheduling (built-in example)
        assigned: 3 of 4 jobs
        ...
        0
        >>> main(["--segment"])  # doctest: +ELLIPSIS
        application: image segmentation
        input: built-in example
        foreground pixels: 5 of 12
        cut value: 500.002
        ...
        0
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.input and args.mode in ("match", "schedule"):
        parser.error(f"--{args.mode} runs a built-in example and has no --input format")
    runners = {"match": _run_match, "schedule": _run_schedule, "segment": _run_segment}
    try:
        return runners.get(args.mode, _run_solver)(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
