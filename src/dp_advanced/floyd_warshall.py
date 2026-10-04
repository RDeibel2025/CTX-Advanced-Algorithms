"""Floyd-Warshall all-pairs shortest paths, and why its third dimension can go.

Floyd-Warshall answers every shortest-path question in a weighted directed
graph at once. It is dynamic programming over *which vertices a path is
allowed to pass through*. Writing ``D(k)[i][j]`` for the length of the
shortest ``i -> j`` path whose intermediate vertices all lie in
``{0, ..., k - 1}`` (CLRS 4e section 23.2, shifted to 0-indexed vertices)::

    D(0)[i][j] = w(i, j)
    D(k)[i][j] = min(D(k-1)[i][j],
                     D(k-1)[i][k-1] + D(k-1)[k-1][j])

Either the best path avoids vertex ``k - 1`` and nothing changes, or it goes
through ``k - 1`` exactly once and splits into two shorter paths that each
use only the earlier vertices. ``D(n)`` is the answer. Negative edge weights
are allowed; negative *cycles* are not, because a cycle that can be lapped
forever has no shortest path to report.

Why the in-place 2D version is correct
--------------------------------------
Written literally, the recurrence keeps ``n + 1`` matrices of ``n x n``
cells: O(n^3) space. :func:`floyd_warshall_3d` does exactly that, so the
benchmark can measure it. :func:`floyd_warshall` keeps one matrix and
overwrites it in place, and the reason that is safe is short:

* During iteration ``k`` the update reads only three cells: ``dist[i][j]``,
  ``dist[i][k]`` and ``dist[k][j]``. The last two lie in row ``k`` or column
  ``k``.
* Row ``k`` and column ``k`` do not change during iteration ``k``. When there
  is no negative cycle, ``dist[k][k] = 0``, so the candidate for a column-``k``
  cell is ``dist[i][k] + dist[k][k] = dist[i][k]``, which is not smaller than
  what is already there. The same argument with ``dist[k][j] + 0`` covers
  row ``k``. Neither can improve, so neither is written.
* So every value the update reads from row ``k`` or column ``k`` is the same
  whether it is read before or after the in-place writes of this iteration.
  Overwriting ``D(k-1)`` with ``D(k)`` while it is still being read changes
  nothing, and the ``k`` dimension can be dropped from O(n^3) to O(n^2)
  space without changing a single result.

This is the same move as the rolling row in the Week 5 knapsack, with a
different reason it is safe. Knapsack has to scan its row downwards to avoid
reading its own writes; Floyd-Warshall can scan in any order, because the
cells it reads are exactly the ones it never writes. :func:`floyd_warshall_3d`
exists to make that claim checkable: its last layer equals the 2D result
exactly, not merely approximately.

The loop over ``k`` must be the outermost. ``D(k)`` needs all of ``D(k-1)``,
and with ``i`` or ``j`` outermost a cell would be finalised before the
intermediate vertices it depends on had been considered.

Negative cycles and the diagonal
--------------------------------
After the loops, ``dist[i][i] < 0`` for some ``i`` exactly when vertex ``i``
lies on a negative cycle (CLRS 4e exercise 23.2-6), and both functions then
raise :class:`NegativeCycleError` rather than return numbers that are not
distances. The in-place argument above assumes ``dist[k][k] = 0``, which is
precisely the condition a negative cycle breaks, so the check is not
optional: it is the one case the proof does not cover.

A weight matrix is expected to have 0 on its diagonal. A positive entry
there is a self-loop no shortest path would ever take, so it is treated as
0 (the empty path is shorter). A negative entry is a negative self-loop,
which is itself a negative cycle, so it is kept and the check catches it.
This matches how :func:`~src.utils.matrix_utils.graph_to_weight_matrix`
builds a matrix from a Week 4 :class:`~src.graphs.graph.Graph`.

The functions
-------------
====================== ================== ===================================
Function               Time               Space
====================== ================== ===================================
floyd_warshall         O(V^3)             O(V^2) - one dist, one pred matrix
floyd_warshall_3d      O(V^3)             O(V^3) - every D(k) layer kept
all_pairs_dijkstra     O(V (V + E) log V) O(V^2) result + O(V + E) working
====================== ================== ===================================

:func:`all_pairs_dijkstra` is the comparison point: V runs of the Week 4
heap Dijkstra. On a sparse graph it wins, since ``V (V + E) log V`` is well
under ``V^3`` when ``E`` is near ``V``. On a dense graph ``E`` approaches
``V^2`` and it becomes O(V^3 log V), losing to Floyd-Warshall's three tight
loops. It also cannot accept negative weights at all, which Floyd-Warshall
handles as long as no cycle is negative.

Every function accepts either an ``n x n`` list of lists (``math.inf`` for
no edge) or a Week 4 :class:`~src.graphs.graph.Graph`, in which case the
matrix indices are positions in ``graph.nodes()``. The caller's matrix is
never modified.

Examples:
    The CLRS 4e Figure 23.4 graph, with vertices renumbered 0 to 4:

    >>> INF = math.inf
    >>> weights = [
    ...     [0, 3, 8, INF, -4],
    ...     [INF, 0, INF, 1, 7],
    ...     [INF, 4, 0, INF, INF],
    ...     [2, INF, -5, 0, INF],
    ...     [INF, INF, INF, 6, 0],
    ... ]
    >>> dist, pred = floyd_warshall(weights)
    >>> dist[0][1], reconstruct_path(pred, 0, 1)
    (1.0, [0, 4, 3, 2, 1])
    >>> floyd_warshall_3d(weights)[-1] == dist
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, List, Optional, Sequence, Tuple, Union

from src.graphs.dijkstra import dijkstra
from src.graphs.graph import Graph
from src.utils.matrix_utils import graph_to_weight_matrix

__all__ = [
    "NegativeCycleError",
    "all_pairs_dijkstra",
    "floyd_warshall",
    "floyd_warshall_3d",
    "reconstruct_path",
]

#: What the public functions accept: a square weight matrix or a Week 4 Graph.
_WeightsLike = Union[Sequence[Sequence[float]], Graph]


class NegativeCycleError(ValueError):
    """Raised when the graph contains a cycle whose total weight is negative.

    A subclass of :class:`ValueError` because a negative cycle is a property
    of the input, not a failure of the algorithm: shortest paths are simply
    undefined on such a graph, since any path that can reach the cycle can
    be made arbitrarily short by lapping it. Callers that already catch
    ``ValueError`` for bad input keep working; callers that care can catch
    this class specifically.

    Examples:
        >>> issubclass(NegativeCycleError, ValueError)
        True
        >>> floyd_warshall([[0, -1], [-1, 0]])
        Traceback (most recent call last):
            ...
        src.dp_advanced.floyd_warshall.NegativeCycleError: negative cycle \
through vertex 0: dist[0][0] = -2.0 after all iterations
    """


def _check_entry(entry: Any, row: int, col: int) -> float:
    """Validate one weight-matrix entry and return it as a float.

    Every distance is derived from these entries by addition, so converting
    them to ``float`` here makes the whole result floats regardless of
    whether the caller wrote ``3`` or ``3.0``. It also keeps a run on a graph
    with a negative cycle from growing ever larger Python integers before
    the check at the end gets to reject it.

    Args:
        entry: The value found at ``weights[row][col]``.
        row: Its row index, for the error message.
        col: Its column index, for the error message.

    Returns:
        ``float(entry)``.

    Raises:
        TypeError: If ``entry`` is not a real number. ``bool`` is rejected,
            because ``True`` is an accident, not a weight.
        ValueError: If ``entry`` is NaN or negative infinity, neither of
            which is a meaningful edge weight.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_entry(3, 0, 1)
        3.0
        >>> _check_entry(math.inf, 0, 1)
        inf
        >>> _check_entry(True, 2, 0)
        Traceback (most recent call last):
            ...
        TypeError: weights[2][0] must be a real number, got bool
        >>> _check_entry(float("nan"), 1, 1)
        Traceback (most recent call last):
            ...
        ValueError: weights[1][1] must not be NaN or -inf, got nan
    """
    if isinstance(entry, bool) or not isinstance(entry, numbers.Real):
        raise TypeError(
            f"weights[{row}][{col}] must be a real number, "
            f"got {type(entry).__name__}"
        )
    value = float(entry)
    if math.isnan(value) or value == -math.inf:
        raise ValueError(
            f"weights[{row}][{col}] must not be NaN or -inf, got {value!r}"
        )
    return value


def _prepare(weights: _WeightsLike) -> List[List[float]]:
    """Validate ``weights`` and return a fresh float matrix to work in.

    The one entry point for both input forms. A :class:`Graph` is converted
    with :func:`~src.utils.matrix_utils.graph_to_weight_matrix`, whose row
    order is ``graph.nodes()``. A list of lists is checked to be square and
    numeric. Either way the result is a brand-new matrix, so the algorithms
    can overwrite it freely and the caller's data is never touched.

    The diagonal is normalised to ``min(entry, 0.0)``: a positive self-loop
    is irrelevant to shortest paths, while a negative one is a negative
    cycle that must survive so the final check can see it.

    Args:
        weights: An ``n x n`` weight matrix (``math.inf`` for no edge) or a
            Week 4 :class:`~src.graphs.graph.Graph`. Not modified.

    Returns:
        A new ``n x n`` list of lists of ``float``.

    Raises:
        TypeError: If ``weights`` is neither a Graph nor a sequence of
            sequences, or if any entry is not a real number.
        ValueError: If the matrix is not square, or an entry is NaN or
            negative infinity.

    Time Complexity:
        O(V^2) - every entry is checked and copied once.

    Space Complexity:
        O(V^2) - the copy.

    Examples:
        >>> _prepare([[5, 2], [math.inf, -1]])
        [[0.0, 2.0], [inf, -1.0]]
        >>> _prepare([])
        []
        >>> _prepare([[0, 1], [1]])
        Traceback (most recent call last):
            ...
        ValueError: weights must be square: row 1 has 1 entries, expected 2
        >>> _prepare(7)
        Traceback (most recent call last):
            ...
        TypeError: weights must be a Graph or an n x n matrix, got int
    """
    if isinstance(weights, Graph):
        matrix, _ = graph_to_weight_matrix(weights)
        rows: List[Any] = matrix
    else:
        try:
            rows = list(weights)
        except TypeError:
            raise TypeError(
                "weights must be a Graph or an n x n matrix, "
                f"got {type(weights).__name__}"
            ) from None

    size = len(rows)
    result: List[List[float]] = []
    for row_index, row in enumerate(rows):
        try:
            entries = list(row)
        except TypeError:
            raise TypeError(
                f"weights[{row_index}] must be a sequence of numbers, "
                f"got {type(row).__name__}"
            ) from None
        if len(entries) != size:
            raise ValueError(
                f"weights must be square: row {row_index} has "
                f"{len(entries)} entries, expected {size}"
            )
        checked = [
            _check_entry(entry, row_index, col_index)
            for col_index, entry in enumerate(entries)
        ]
        if checked[row_index] > 0.0:
            checked[row_index] = 0.0
        result.append(checked)

    return result


def _require_no_negative_cycle(dist: List[List[float]]) -> None:
    """Raise :class:`NegativeCycleError` if any diagonal entry is negative.

    After Floyd-Warshall has run, ``dist[i][i]`` is the length of the
    shortest closed walk through ``i``. It starts at 0 and can only go
    down, and it goes below 0 exactly when ``i`` lies on a negative cycle.

    Args:
        dist: A finished distance matrix.

    Returns:
        None. This helper is called for its exception, not its value.

    Raises:
        NegativeCycleError: Naming the first vertex found on a negative
            cycle.

    Time Complexity:
        O(V) - one pass down the diagonal.

    Space Complexity:
        O(1).

    Examples:
        >>> _require_no_negative_cycle([[0.0, 1.0], [2.0, 0.0]]) is None
        True
        >>> _require_no_negative_cycle([[0.0, 1.0], [2.0, -3.0]])
        Traceback (most recent call last):
            ...
        src.dp_advanced.floyd_warshall.NegativeCycleError: negative cycle \
through vertex 1: dist[1][1] = -3.0 after all iterations
    """
    for vertex in range(len(dist)):
        if dist[vertex][vertex] < 0:
            raise NegativeCycleError(
                f"negative cycle through vertex {vertex}: "
                f"dist[{vertex}][{vertex}] = {dist[vertex][vertex]!r} "
                "after all iterations"
            )


def floyd_warshall(
    weights: _WeightsLike,
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """Compute all-pairs shortest distances and predecessors in O(V^2) space.

    The textbook triple loop with ``k`` outermost, run on a single distance
    matrix that is overwritten in place. The module docstring explains why
    dropping the ``k`` dimension is safe: row ``k`` and column ``k`` are the
    only cells iteration ``k`` reads besides the one it writes, and they do
    not change during that iteration.

    Alongside the distances it keeps a predecessor matrix in the CLRS
    ``pi`` convention: ``pred[i][j]`` is the vertex just before ``j`` on a
    shortest ``i -> j`` path. It starts as ``i`` wherever there is a direct
    edge, and when routing through ``k`` improves ``i -> j``, the new path
    ends with the ``k -> j`` path, so ``pred[i][j]`` becomes ``pred[k][j]``.
    That is all :func:`reconstruct_path` needs to rebuild any route.

    The inner loops are tuned for pure Python without changing the
    algorithm: ``dist[k]`` and ``dist[i]`` are bound to locals once per row,
    ``dist[i][k]`` is hoisted out of the ``j`` loop, and a row ``i`` with no
    path to ``k`` is skipped entirely, since ``inf + anything`` can never
    improve a cell. All three are constant-factor changes; the O(V^3) loop
    is intact.

    Args:
        weights: An ``n x n`` weight matrix (``math.inf`` for no edge,
            0 on the diagonal), or a Week 4 :class:`~src.graphs.graph.Graph`
            whose vertex ``i`` is ``graph.nodes()[i]``. Not modified.

    Returns:
        A ``(dist, pred)`` pair of new ``n x n`` matrices. ``dist[i][j]`` is
        the shortest ``i -> j`` distance as a float, ``math.inf`` when ``j``
        is unreachable from ``i``. ``pred[i][j]`` is the predecessor of
        ``j`` on such a path, ``None`` when ``i == j`` or ``j`` is
        unreachable.

    Raises:
        TypeError: If ``weights`` is not a Graph or a matrix of real
            numbers (``bool`` rejected).
        ValueError: If the matrix is not square, or holds NaN or ``-inf``.
        NegativeCycleError: If the graph contains a negative-weight cycle.
            It subclasses ``ValueError``.

    Time Complexity:
        Best:    O(V^2)  - no edges at all, so every row except ``k`` itself
                           is skipped by the infinity check
        Average: O(V^3)
        Worst:   O(V^3)  - strongly connected, so no row is ever skipped

    Space Complexity:
        O(V^2) - one distance matrix and one predecessor matrix. The full
        recurrence would need O(V^3); see :func:`floyd_warshall_3d`.

    Examples:
        The CLRS 4e Figure 23.4 graph, vertices renumbered 0 to 4. The
        distances match the book's ``D(5)`` exactly:

        >>> INF = math.inf
        >>> weights = [
        ...     [0, 3, 8, INF, -4],
        ...     [INF, 0, INF, 1, 7],
        ...     [INF, 4, 0, INF, INF],
        ...     [2, INF, -5, 0, INF],
        ...     [INF, INF, INF, 6, 0],
        ... ]
        >>> dist, pred = floyd_warshall(weights)
        >>> for row in dist:
        ...     print(row)
        [0.0, 1.0, -3.0, 2.0, -4.0]
        [3.0, 0.0, -4.0, 1.0, -1.0]
        [7.0, 4.0, 0.0, 5.0, 3.0]
        [2.0, -1.0, -5.0, 0.0, -2.0]
        [8.0, 5.0, 1.0, 6.0, 0.0]
        >>> dist == [[0, 1, -3, 2, -4], [3, 0, -4, 1, -1], [7, 4, 0, 5, 3],
        ...          [2, -1, -5, 0, -2], [8, 5, 1, 6, 0]]
        True

        And the predecessors match the book's ``Pi(5)``:

        >>> for row in pred:
        ...     print(row)
        [None, 2, 3, 4, 0]
        [3, None, 3, 1, 0]
        [3, 2, None, 1, 0]
        [3, 2, 3, None, 0]
        [3, 2, 3, 4, None]

        The caller's matrix is left exactly as it was:

        >>> weights[0]
        [0, 3, 8, inf, -4]

        An unreachable pair keeps ``inf`` and a ``None`` predecessor:

        >>> dist, pred = floyd_warshall([[0, 2, INF], [INF, 0, INF],
        ...                              [INF, 1, 0]])
        >>> dist[0][2], pred[0][2]
        (inf, None)
        >>> dist[2][1], pred[2][1]
        (1.0, 2)

        A Week 4 Graph works too. Vertex ``i`` is ``graph.nodes()[i]``, and
        a negative edge is fine as long as no cycle is negative:

        >>> graph = Graph(directed=True, weighted=True)
        >>> for u, v, w in [("s", "a", 4), ("s", "b", 1), ("b", "a", -2),
        ...                 ("a", "t", 1)]:
        ...     graph.add_edge(u, v, w)
        >>> graph.nodes()
        ['s', 'a', 'b', 't']
        >>> dist, pred = floyd_warshall(graph)
        >>> dist[0]
        [0.0, -1.0, 1.0, 0.0]
        >>> [graph.nodes()[v] for v in reconstruct_path(pred, 0, 3)]
        ['s', 'b', 'a', 't']

        A negative self-loop on a Graph is a negative cycle as well. The
        reported -2.0 is the loop lapped twice, once when it seeds the
        diagonal and once more when vertex 3 is the intermediate - exactly
        the unbounded descent that makes the distance undefined:

        >>> graph.add_edge("t", "t", -1)
        >>> floyd_warshall(graph)
        Traceback (most recent call last):
            ...
        src.dp_advanced.floyd_warshall.NegativeCycleError: negative cycle \
through vertex 3: dist[3][3] = -2.0 after all iterations

        A negative cycle is refused rather than reported as distances:

        >>> floyd_warshall([[0, 1, INF], [INF, 0, -2], [-1, INF, 0]])
        Traceback (most recent call last):
            ...
        src.dp_advanced.floyd_warshall.NegativeCycleError: negative cycle \
through vertex 0: dist[0][0] = -2.0 after all iterations

        The empty graph has nothing to compute:

        >>> floyd_warshall([])
        ([], [])
    """
    dist = _prepare(weights)
    size = len(dist)
    inf = math.inf
    vertices = range(size)

    pred: List[List[Optional[int]]] = [
        [i if i != j and dist[i][j] < inf else None for j in vertices]
        for i in vertices
    ]

    # k OUTERMOST. D(k) needs every cell of D(k-1); with i or j outside, a
    # cell would be finalised before all its intermediate vertices were tried.
    for k in vertices:
        row_k = dist[k]
        pred_k = pred[k]
        for i in vertices:
            row_i = dist[i]
            d_ik = row_i[k]
            # No path from i to k yet, so routing through k cannot help i.
            if d_ik == inf:
                continue
            pred_i = pred[i]
            for j in vertices:
                candidate = d_ik + row_k[j]
                if candidate < row_i[j]:
                    row_i[j] = candidate
                    pred_i[j] = pred_k[j]

    _require_no_negative_cycle(dist)
    return dist, pred


def floyd_warshall_3d(weights: _WeightsLike) -> List[List[List[float]]]:
    """Compute every layer ``D(0), ..., D(n)`` of the recurrence and keep them.

    The literal transcription of the recurrence, with nothing optimised
    away: layer ``k`` is a fresh ``n x n`` matrix built entirely from layer
    ``k - 1``, considering vertex ``k - 1`` as a new permitted intermediate.
    It exists for one reason, the space comparison in the Week 6 benchmark,
    and it doubles as a check on the module's central claim: because the
    in-place version reads only cells that iteration ``k`` never changes,
    the last layer here equals :func:`floyd_warshall`'s ``dist`` exactly.

    It is deliberately plain and not tuned for speed. It is meant for
    ``n <= 100``, where its ``(n + 1) * n^2`` floats still fit comfortably.

    Args:
        weights: An ``n x n`` weight matrix or a Week 4
            :class:`~src.graphs.graph.Graph`. Not modified.

    Returns:
        A list of ``n + 1`` matrices, each ``n x n``. Layer 0 is the
        prepared weight matrix (diagonal normalised as described in the
        module docstring) and layer ``n`` holds the shortest distances.

    Raises:
        TypeError: If ``weights`` is not a Graph or a matrix of real
            numbers.
        ValueError: If the matrix is not square, or holds NaN or ``-inf``.
        NegativeCycleError: If the graph contains a negative-weight cycle,
            exactly as :func:`floyd_warshall` does.

    Time Complexity:
        O(V^3) in every case - every cell of every layer is written once.

    Space Complexity:
        O(V^3) - ``(V + 1) * V^2`` cells, all kept. This is the cost the
        in-place version avoids.

    Examples:
        >>> INF = math.inf
        >>> weights = [[0, 4, INF], [INF, 0, 1], [2, INF, 0]]
        >>> layers = floyd_warshall_3d(weights)
        >>> len(layers), len(layers[0]), len(layers[0][0])
        (4, 3, 3)
        >>> layers[0]
        [[0.0, 4.0, inf], [inf, 0.0, 1.0], [2.0, inf, 0.0]]
        >>> layers[1][2]
        [2.0, 6.0, 0.0]
        >>> layers[-1]
        [[0.0, 4.0, 5.0], [3.0, 0.0, 1.0], [2.0, 6.0, 0.0]]

        The last layer is the in-place result, cell for cell:

        >>> layers[-1] == floyd_warshall(weights)[0]
        True
        >>> clrs = [[0, 3, 8, INF, -4], [INF, 0, INF, 1, 7],
        ...         [INF, 4, 0, INF, INF], [2, INF, -5, 0, INF],
        ...         [INF, INF, INF, 6, 0]]
        >>> floyd_warshall_3d(clrs)[-1] == floyd_warshall(clrs)[0]
        True

        Negative cycles are rejected here too:

        >>> floyd_warshall_3d([[0, -1], [-1, 0]])
        Traceback (most recent call last):
            ...
        src.dp_advanced.floyd_warshall.NegativeCycleError: negative cycle \
through vertex 0: dist[0][0] = -2.0 after all iterations
    """
    layers = [_prepare(weights)]
    size = len(layers[0])

    for k in range(1, size + 1):
        previous = layers[k - 1]
        via = k - 1
        current = [
            [
                min(previous[i][j], previous[i][via] + previous[via][j])
                for j in range(size)
            ]
            for i in range(size)
        ]
        layers.append(current)

    _require_no_negative_cycle(layers[-1])
    return layers


def reconstruct_path(
    pred: Sequence[Sequence[Optional[int]]],
    i: int,
    j: int,
) -> List[int]:
    """Rebuild the shortest ``i -> j`` path from a predecessor matrix.

    Row ``i`` of the predecessor matrix is a shortest-path tree rooted at
    ``i``, stored upside down: each vertex remembers only the one before it.
    Walking those links back from ``j`` produces the path in reverse, and
    one reversal puts it the right way round. Storing all V^2 paths this way
    costs O(V^2) in total, where storing the paths themselves would cost up
    to O(V^3).

    The walk is capped at ``n`` steps. A simple path visits each vertex at
    most once, so a longer walk can only come from a malformed matrix, and
    the cap stops it looping forever.

    Args:
        pred: The second element of a :func:`floyd_warshall` result.
        i: The vertex the path starts at.
        j: The vertex the path ends at.

    Returns:
        The vertices from ``i`` to ``j`` inclusive, in order. ``[i]`` when
        ``i == j``, and ``[]`` when ``j`` is unreachable from ``i``.

    Raises:
        IndexError: If ``i`` or ``j`` is not a vertex index of ``pred``.

    Time Complexity:
        O(L) for a path of L vertices, which is O(V) at worst.

    Space Complexity:
        O(L) - the path itself.

    Examples:
        >>> INF = math.inf
        >>> weights = [
        ...     [0, 3, 8, INF, -4],
        ...     [INF, 0, INF, 1, 7],
        ...     [INF, 4, 0, INF, INF],
        ...     [2, INF, -5, 0, INF],
        ...     [INF, INF, INF, 6, 0],
        ... ]
        >>> dist, pred = floyd_warshall(weights)

        The route from 0 to 1 never uses the direct edge of weight 3. It
        goes the long way round through two negative edges:

        >>> reconstruct_path(pred, 0, 1)
        [0, 4, 3, 2, 1]
        >>> weights[0][4] + weights[4][3] + weights[3][2] + weights[2][1]
        1
        >>> dist[0][1]
        1.0
        >>> reconstruct_path(pred, 2, 0)
        [2, 1, 3, 0]

        A vertex is its own path, and an unreachable one has none:

        >>> reconstruct_path(pred, 3, 3)
        [3]
        >>> _, pred = floyd_warshall([[0, 2, INF], [INF, 0, INF],
        ...                           [INF, 1, 0]])
        >>> reconstruct_path(pred, 0, 2)
        []

        Indices outside the matrix are a programming error:

        >>> reconstruct_path(pred, 0, 3)
        Traceback (most recent call last):
            ...
        IndexError: vertex 3 is out of range for a 3-vertex predecessor matrix
    """
    size = len(pred)
    for vertex in (i, j):
        if not 0 <= vertex < size:
            raise IndexError(
                f"vertex {vertex} is out of range for a {size}-vertex "
                "predecessor matrix"
            )

    if i == j:
        return [i]

    row = pred[i]
    path = [j]
    current: Optional[int] = j
    for _ in range(size):
        current = row[current]
        if current is None:
            return []
        path.append(current)
        if current == i:
            path.reverse()
            return path

    # More than n steps without reaching i: the matrix is malformed.
    return []


def all_pairs_dijkstra(graph: Graph) -> List[List[float]]:
    """Compute all-pairs shortest distances by running Dijkstra from every node.

    The comparison point for :func:`floyd_warshall`. It calls the Week 4
    heap-based :func:`~src.graphs.dijkstra.dijkstra` once per source and
    lays each distance map out as one row of a matrix, with rows and
    columns both in ``graph.nodes()`` order. That is the same order
    :func:`floyd_warshall` uses when it is handed the same Graph, so the two
    results can be compared cell for cell.

    It reuses Week 4 unchanged, including its refusal of negative weights:
    Dijkstra is undefined for them, and the ``ValueError`` it raises is
    allowed to propagate rather than being caught and turned into a wrong
    answer. Floyd-Warshall is the tool for that case.

    Args:
        graph: A Week 4 :class:`~src.graphs.graph.Graph`, weighted or not,
            directed or not. Not modified.

    Returns:
        A new ``V x V`` list of lists of ``float``. Entry ``[i][j]`` is the
        shortest distance from ``graph.nodes()[i]`` to ``graph.nodes()[j]``,
        ``math.inf`` when there is no path, and 0.0 on the diagonal.

    Raises:
        TypeError: If ``graph`` is not a :class:`~src.graphs.graph.Graph`.
        ValueError: From :func:`~src.graphs.dijkstra.dijkstra`, if any edge
            carries a negative weight.

    Time Complexity:
        O(V (V + E) log V) - V independent runs of an O((V + E) log V)
        search. Below Floyd-Warshall's O(V^3) on sparse graphs; above it,
        at O(V^3 log V), once E approaches V^2.

    Space Complexity:
        O(V^2) for the result, plus O(V + E) of working memory for the one
        Dijkstra run in progress.

    Examples:
        >>> graph = Graph(directed=True, weighted=True)
        >>> for u, v, w in [("A", "B", 4), ("A", "C", 1), ("C", "B", 2),
        ...                 ("B", "D", 1)]:
        ...     graph.add_edge(u, v, w)
        >>> graph.nodes()
        ['A', 'B', 'C', 'D']
        >>> for row in all_pairs_dijkstra(graph):
        ...     print(row)
        [0.0, 3.0, 1.0, 4.0]
        [inf, 0.0, inf, 1.0]
        [inf, 2.0, 0.0, 3.0]
        [inf, inf, inf, 0.0]

        Same rows, same columns, same answer as Floyd-Warshall on the same
        Graph:

        >>> all_pairs_dijkstra(graph) == floyd_warshall(graph)[0]
        True

        Week 4's refusal of negative weights comes through unchanged:

        >>> graph.add_edge("D", "A", -1)
        >>> all_pairs_dijkstra(graph)
        Traceback (most recent call last):
            ...
        ValueError: negative weight -1.0 on edge ('D', 'A'); weights must be >= 0

        Only a Graph is accepted:

        >>> all_pairs_dijkstra([[0, 1], [1, 0]])
        Traceback (most recent call last):
            ...
        TypeError: graph must be a Graph, got list
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"graph must be a Graph, got {type(graph).__name__}")

    nodes = graph.nodes()
    result: List[List[float]] = []
    for source in nodes:
        distances, _ = dijkstra(graph, source)
        result.append([distances[target] for target in nodes])
    return result
