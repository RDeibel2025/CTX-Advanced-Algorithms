"""Dense matrices for the Week 6 dynamic-programming algorithms.

Floyd-Warshall, Held-Karp and matrix-chain multiplication all read their
input as a square table of numbers rather than as an adjacency list, and
all three are checked against something else: Floyd-Warshall against
Dijkstra run once per source, Held-Karp against brute force, and a
matrix-chain parenthesization against the cost it claims to achieve. This
module holds the shared plumbing for that, so each algorithm module can
stay focused on its own recurrence:

=========================== ==============================================
Function                    Job
=========================== ==============================================
graph_to_weight_matrix      Week 4 :class:`Graph` to a weight matrix
random_weight_matrix        seeded directed instance, optionally negative
matrices_close              element-wise float comparison, inf == inf
random_dimensions           seeded MCM dimension list ``p``
parenthesization_cost       exact cost of a written parenthesization
format_matrix               aligned text rendering for reports
=========================== ==============================================

Three conventions are fixed here and every Week 6 module relies on them:

* **No edge is** :data:`INF`, which is :data:`math.inf`. Infinity is the
  identity of ``min`` and absorbs ``+``, so the Floyd-Warshall relaxation
  ``dist[i][k] + dist[k][j]`` needs no special case for a missing edge.
* **The diagonal is 0.** Staying put costs nothing. The one exception is a
  negative self-loop, which is written onto the diagonal so that the usual
  ``dist[i][i] < 0`` negative-cycle test catches it.
* **Every generator is seeded** (42 by default, the Week 1 convention) and
  draws from a local :class:`random.Random`, never the global ``random``
  module, so an instance can be rebuilt exactly from its printed arguments.

The most important function is :func:`parenthesization_cost`. Several
parenthesizations of the same chain often tie for the optimum, so a test
that compares an algorithm's string against one expected string breaks
on a tie even though the answer is right. Pricing the string instead, and
comparing that price with the optimal cost, accepts every correct answer
and still rejects every wrong one.

Examples:
    >>> p = [30, 35, 15, 5, 10, 20, 25]
    >>> parenthesization_cost(p, "((A1(A2A3))((A4A5)A6))")
    15125
    >>> print(format_matrix([[0, 3, INF], [INF, 0, -2], [1.5, INF, 0]]))
      0    3  inf
    inf    0   -2
    1.5  inf    0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import random
from typing import Any, List, Optional, Sequence, Tuple

from src.graphs.graph import Graph

__all__ = [
    "INF",
    "format_matrix",
    "graph_to_weight_matrix",
    "matrices_close",
    "parenthesization_cost",
    "random_dimensions",
    "random_weight_matrix",
]

#: The weight of a missing edge, and the distance to an unreachable vertex.
INF: float = math.inf

#: The only characters accepted as matrix-index digits. ``str.isdigit`` is
#: avoided on purpose: it also accepts superscripts such as ``"2"`` written
#: as a Unicode superscript, which ``int`` then refuses.
_DIGITS = "0123456789"


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` unchanged after checking it is an int >= ``minimum``.

    Every size, bound and dimension in this module is a whole number, and
    ``bool`` is rejected as a type error because ``True`` is an accident,
    not a size.

    Args:
        value: The value to check.
        name: How the value is named in the error message.
        minimum: The smallest value accepted.

    Returns:
        ``value``, unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``, or is a ``bool``.
        ValueError: If ``value`` is less than ``minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(3, "n", 0)
        3
        >>> _require_int(True, "n", 0)
        Traceback (most recent call last):
            ...
        TypeError: n must be an int, got bool
        >>> _require_int(0, "p[0]", 1)
        Traceback (most recent call last):
            ...
        ValueError: p[0] must be >= 1, got 0
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_dimensions(p: Sequence[int]) -> List[int]:
    """Copy a matrix-chain dimension list after checking every entry.

    This is the matrix-chain contract shared with
    :mod:`src.dp_advanced.matrix_chain_multiplication`: matrix ``Ai`` is
    ``p[i-1] x p[i]``, so ``p`` needs at least two entries to describe even
    one matrix, and every dimension must be a positive integer.

    Args:
        p: The dimension list.

    Returns:
        A new list holding the same entries.

    Raises:
        TypeError: If ``p`` is not iterable or an entry is not an ``int``
            (``bool`` included).
        ValueError: If ``p`` has fewer than two entries or an entry is not
            positive.

    Time Complexity:
        O(n) - one check per entry.

    Space Complexity:
        O(n) - the returned copy.

    Examples:
        >>> _require_dimensions((10, 20, 30))
        [10, 20, 30]
        >>> _require_dimensions([10])
        Traceback (most recent call last):
            ...
        ValueError: p must have at least 2 entries, got 1
        >>> _require_dimensions([10, 2.5])
        Traceback (most recent call last):
            ...
        TypeError: p[1] must be an int, got float
        >>> _require_dimensions([10, -1])
        Traceback (most recent call last):
            ...
        ValueError: p[1] must be >= 1, got -1
    """
    try:
        dims = list(p)
    except TypeError:
        raise TypeError(
            f"p must be a sequence of ints, got {type(p).__name__}"
        ) from None
    if len(dims) < 2:
        raise ValueError(f"p must have at least 2 entries, got {len(dims)}")
    for index, entry in enumerate(dims):
        _require_int(entry, f"p[{index}]", 1)
    return dims


def graph_to_weight_matrix(graph: Graph) -> Tuple[List[List[float]], List[Any]]:
    """Convert a Week 4 :class:`Graph` into a dense weight matrix.

    Row and column ``i`` belong to ``graph.nodes()[i]``, which is the
    graph's insertion order and the same order
    :meth:`Graph.to_adjacency_matrix` uses, so an index returned by an
    algorithm run on this matrix can be translated back to a node by
    indexing the returned list.

    Unlike :meth:`Graph.to_adjacency_matrix`, which fills empty cells with
    0 and therefore cannot tell "no edge" from "an edge of weight 0", this
    matrix uses :data:`INF` for a missing edge. That is the representation
    shortest-path algorithms need, and it keeps a genuine zero-weight edge
    distinct.

    The diagonal is 0.0, because the empty path from a vertex to itself
    costs nothing. A self-loop changes that only when its weight is
    negative: going round it lowers the cost, which is a negative cycle, so
    the weight is written onto the diagonal where the standard
    ``dist[i][i] < 0`` test will see it. A non-negative self-loop can never
    beat the empty path and is ignored.

    An undirected graph stores each edge from both endpoints, so its matrix
    comes out symmetric with no extra work. An unweighted graph reports
    1.0 for every edge, so its matrix counts hops.

    Args:
        graph: The graph to convert. Not modified.

    Returns:
        A ``(matrix, nodes)`` pair. ``matrix`` is a new ``V x V`` list of
        lists of ``float``; ``nodes`` is ``graph.nodes()``, the row labels.

    Raises:
        TypeError: If ``graph`` is not a :class:`Graph`.

    Time Complexity:
        O(V^2 + E) - V^2 to lay out the matrix, then one write per stored
        edge.

    Space Complexity:
        O(V^2) - the matrix itself, plus O(V) for the node index.

    Examples:
        >>> g = Graph(directed=True, weighted=True)
        >>> g.add_edge("a", "b", 3)
        >>> g.add_edge("b", "c", -2)
        >>> g.add_node("d")
        >>> matrix, nodes = graph_to_weight_matrix(g)
        >>> nodes
        ['a', 'b', 'c', 'd']
        >>> matrix[0]
        [0.0, 3.0, inf, inf]
        >>> print(format_matrix(matrix))
          0    3  inf  inf
        inf    0   -2  inf
        inf  inf    0  inf
        inf  inf  inf    0

        A negative self-loop reaches the diagonal; a positive one does not:

        >>> g.add_edge("c", "c", -1)
        >>> g.add_edge("d", "d", 5)
        >>> matrix, _ = graph_to_weight_matrix(g)
        >>> matrix[2][2], matrix[3][3]
        (-1.0, 0.0)

        Undirected graphs give a symmetric matrix:

        >>> u = Graph(weighted=True)
        >>> u.add_edge(0, 1, 4)
        >>> u.add_edge(1, 2, 1.5)
        >>> matrix, _ = graph_to_weight_matrix(u)
        >>> matrix
        [[0.0, 4.0, inf], [4.0, 0.0, 1.5], [inf, 1.5, 0.0]]
        >>> all(matrix[i][j] == matrix[j][i] for i in range(3) for j in range(3))
        True
        >>> graph_to_weight_matrix([[0, 1], [1, 0]])
        Traceback (most recent call last):
            ...
        TypeError: graph must be a Graph, got list
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"graph must be a Graph, got {type(graph).__name__}")

    nodes = graph.nodes()
    index = {node: position for position, node in enumerate(nodes)}
    size = len(nodes)

    matrix = [[INF] * size for _ in range(size)]
    for position in range(size):
        matrix[position][position] = 0.0

    for u in nodes:
        row = matrix[index[u]]
        for v, weight in graph.neighbor_items(u):
            weight = float(weight)
            if u == v:
                # Only a negative self-loop beats the empty path.
                if weight < 0:
                    row[index[v]] = weight
            else:
                row[index[v]] = weight

    return matrix, nodes


def random_weight_matrix(
    n: int,
    density: float,
    weight_range: Tuple[int, int] = (1, 100),
    seed: Optional[int] = 42,
    allow_negative: bool = False,
) -> List[List[float]]:
    """Build a seeded random directed weight matrix with integer weights.

    Every ordered pair ``(u, v)`` with ``u != v`` is visited in row-major
    order. The edge is present with probability ``density``, and a present
    edge gets an integer weight drawn uniformly from ``weight_range``,
    inclusive at both ends. Absent edges are :data:`INF` and the diagonal
    is 0. Because the instance is directed, ``w(u, v)`` and ``w(v, u)`` are
    drawn independently and an asymmetric matrix is the normal case.

    **Negative weights without a negative cycle.** With
    ``allow_negative=True`` the matrix is first drawn exactly as above, and
    then every vertex gets a random integer *potential* ``h[v]`` in
    ``[0, high]``. Each edge is reweighted to::

        w'(u, v) = w(u, v) + h[u] - h[v]

    Around any cycle ``v0 -> v1 -> ... -> vk = v0`` the potentials
    telescope, since every ``h`` is added once as the tail of one edge and
    subtracted once as the head of the next, so the cycle's total under
    ``w'`` equals its total under ``w``. Every original weight is
    non-negative, so every cycle total stays non-negative and **no negative
    cycle can exist**, however negative individual edges become. This is
    Johnson's reweighting run in reverse: Johnson uses a potential to make
    weights non-negative, this uses one to make them negative safely.
    Shortest paths keep the same vertices too: every ``u -> v`` path moves
    by the same ``h[u] - h[v]``.

    The edges and the base weights are drawn before the potentials, so the
    same seed gives the same set of edges with or without
    ``allow_negative``, and the two matrices differ only by the
    reweighting. With ``allow_negative=True`` the weights may fall outside
    ``weight_range``; they lie in ``[low - high, high + high]``.

    Args:
        n: Number of vertices.
        density: Probability that each ordered pair is an edge, in
            ``[0, 1]``. 0 gives no edges and 1 gives a complete digraph.
        weight_range: ``(low, high)`` integer bounds for the base weights,
            with ``0 <= low <= high``. They must be non-negative, since the
            no-negative-cycle guarantee rests on the base draw.
        seed: Seed for a local :class:`random.Random`. ``None`` draws from
            the operating system and is not reproducible.
        allow_negative: Reweight with a random potential as described
            above, which typically makes some weights negative.

    Returns:
        A new ``n x n`` list of lists. Weights and the 0 diagonal are
        ``int``; absent edges are :data:`INF`.

    Raises:
        TypeError: If ``n`` or either bound is not an ``int``, or
            ``density`` is not a real number (``bool`` is rejected).
        ValueError: If ``n`` is negative, ``density`` is outside
            ``[0, 1]``, or ``weight_range`` is not two bounds with
            ``0 <= low <= high``.

    Time Complexity:
        O(n^2) - one draw per ordered pair, plus one pass to reweight.

    Space Complexity:
        O(n^2) - the matrix, plus O(n) for the potentials.

    Examples:
        >>> m = random_weight_matrix(4, 0.5, weight_range=(1, 9), seed=7)
        >>> print(format_matrix(m))
          0    3    1  9
          1    0  inf  2
          2    9    0  2
        inf  inf  inf  0
        >>> m == random_weight_matrix(4, 0.5, weight_range=(1, 9), seed=7)
        True
        >>> random_weight_matrix(3, 0.0)
        [[0, inf, inf], [inf, 0, inf], [inf, inf, 0]]
        >>> random_weight_matrix(0, 1.0)
        []

        With a potential, some weights go negative:

        >>> neg = random_weight_matrix(
        ...     4, 0.5, weight_range=(1, 9), seed=7, allow_negative=True
        ... )
        >>> print(format_matrix(neg))
          0   -6   -8  3
         10    0  inf  5
         11    9    0  5
        inf  inf  inf  0
        >>> any(w < 0 for row in neg for w in row)
        True

        but every cycle keeps its original total, so none can be negative.
        Every two-edge cycle, then the three-edge cycle 0 -> 2 -> 1 -> 0:

        >>> all(neg[u][v] + neg[v][u] == m[u][v] + m[v][u]
        ...     for u in range(4) for v in range(4))
        True
        >>> neg[0][2] + neg[2][1] + neg[1][0], m[0][2] + m[2][1] + m[1][0]
        (11, 11)
        >>> random_weight_matrix(3, 1.5)
        Traceback (most recent call last):
            ...
        ValueError: density must be in [0, 1], got 1.5
        >>> random_weight_matrix(3, 0.5, weight_range=(-5, 5))
        Traceback (most recent call last):
            ...
        ValueError: weight_range low must be >= 0, got -5
    """
    _require_int(n, "n", 0)
    if isinstance(density, bool) or not isinstance(density, (int, float)):
        raise TypeError(
            f"density must be a real number, got {type(density).__name__}"
        )
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    low, high = _require_weight_range(weight_range)

    rng = random.Random(seed)
    matrix: List[List[float]] = [[INF] * n for _ in range(n)]
    for u in range(n):
        row = matrix[u]
        row[u] = 0
        for v in range(n):
            if u != v and rng.random() < density:
                row[v] = rng.randint(low, high)

    if allow_negative:
        # Drawn after the edges, so the edge set does not depend on the flag.
        potential = [rng.randint(0, high) for _ in range(n)]
        for u in range(n):
            row = matrix[u]
            for v in range(n):
                if u != v and row[v] != INF:
                    row[v] = row[v] + potential[u] - potential[v]

    return matrix


def _require_weight_range(weight_range: Sequence[int]) -> Tuple[int, int]:
    """Check a ``(low, high)`` pair of non-negative integer weight bounds.

    Args:
        weight_range: The pair to check.

    Returns:
        The pair as a tuple ``(low, high)``.

    Raises:
        TypeError: If ``weight_range`` is not a sequence or either bound is
            not an ``int`` (``bool`` included).
        ValueError: If it does not hold exactly two bounds, ``low`` is
            negative, or ``low > high``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_weight_range([1, 100])
        (1, 100)
        >>> _require_weight_range((9, 3))
        Traceback (most recent call last):
            ...
        ValueError: weight_range must have low <= high, got (9, 3)
        >>> _require_weight_range((1, 2, 3))
        Traceback (most recent call last):
            ...
        ValueError: weight_range must be a (low, high) pair, got 3 values
    """
    try:
        bounds = tuple(weight_range)
    except TypeError:
        raise TypeError(
            "weight_range must be a (low, high) pair, got "
            f"{type(weight_range).__name__}"
        ) from None
    if len(bounds) != 2:
        raise ValueError(
            f"weight_range must be a (low, high) pair, got {len(bounds)} values"
        )
    low = _require_int(bounds[0], "weight_range low", 0)
    high = _require_int(bounds[1], "weight_range high", 0)
    if low > high:
        raise ValueError(f"weight_range must have low <= high, got {bounds}")
    return low, high


def matrices_close(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> bool:
    """Return True when two matrices have the same shape and close entries.

    Shortest-path distances on float weights are sums, and floating-point
    addition is not associative, so Floyd-Warshall and Dijkstra can reach
    the same distance by different orders of addition and disagree in the
    last bit. Comparing with ``==`` would report that as a bug. This
    compares each pair of entries with :func:`math.isclose` instead.

    Infinity needs no special case: :func:`math.isclose` already treats
    ``inf`` as equal to ``inf`` (and ``-inf`` to ``-inf``) and as far from
    every finite number, which is exactly "both unreachable" versus
    "one reachable". NaN is never close to anything, itself included.

    A shape mismatch, either a different number of rows or any row of a
    different length, returns False rather than raising, so the function
    can sit directly inside an ``assert``.

    Args:
        a: The first matrix, as a sequence of rows.
        b: The second matrix, as a sequence of rows.
        rel_tol: Relative tolerance passed to :func:`math.isclose`.
        abs_tol: Absolute tolerance passed to :func:`math.isclose`, which
            is what lets an entry of 0 match a tiny round-off residue.

    Returns:
        True if the shapes match and every pair of entries is close.

    Raises:
        TypeError: If an entry is not a real number.

    Time Complexity:
        O(r * c) - one comparison per entry, stopping at the first
        mismatch.

    Space Complexity:
        O(1).

    Examples:
        >>> matrices_close([[0.0, 0.1 + 0.2]], [[0, 0.3]])
        True
        >>> [[0.1 + 0.2]] == [[0.3]]
        False
        >>> matrices_close([[0, INF], [INF, 0]], [[0, INF], [INF, 0]])
        True
        >>> matrices_close([[0, INF]], [[0, 1e300]])
        False
        >>> matrices_close([[1, 2]], [[1, 2], [3, 4]])
        False
        >>> matrices_close([[1, 2], [3]], [[1, 2], [3, 4]])
        False
        >>> matrices_close([], [])
        True
    """
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b):
            if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
    return True


def random_dimensions(
    n: int,
    low: int = 5,
    high: int = 100,
    seed: Optional[int] = 42,
) -> List[int]:
    """Return a seeded matrix-chain dimension list for ``n`` matrices.

    Matrix ``Ai`` of the chain is ``p[i-1] x p[i]``, so ``n`` matrices need
    ``n + 1`` dimensions. Neighbouring matrices share a dimension by
    construction, which is what makes every chain built this way
    multipliable.

    Args:
        n: Number of matrices in the chain, at least 1.
        low: Smallest dimension, at least 1.
        high: Largest dimension, at least ``low``.
        seed: Seed for a local :class:`random.Random`.

    Returns:
        A new list of ``n + 1`` integers drawn uniformly from
        ``[low, high]``, inclusive.

    Raises:
        TypeError: If ``n``, ``low`` or ``high`` is not an ``int``.
        ValueError: If ``n < 1``, ``low < 1`` or ``low > high``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the returned list.

    Examples:
        >>> random_dimensions(6)
        [86, 19, 8, 99, 40, 36, 33]
        >>> len(random_dimensions(10))
        11
        >>> random_dimensions(3, low=4, high=4)
        [4, 4, 4, 4]
        >>> random_dimensions(0)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 1, got 0
        >>> random_dimensions(3, low=10, high=5)
        Traceback (most recent call last):
            ...
        ValueError: low must be <= high, got 10 and 5
    """
    _require_int(n, "n", 1)
    _require_int(low, "low", 1)
    _require_int(high, "high", 1)
    if low > high:
        raise ValueError(f"low must be <= high, got {low} and {high}")
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n + 1)]


class _ChainParser:
    """Recursive-descent parser and pricer for one parenthesization.

    The grammar has one rule with two alternatives, and the parser has one
    method per alternative::

        expr := 'A' digits            (_leaf)
              | '(' expr expr ')'     (_product)

    Parsing and pricing happen in the same pass. Each call returns the
    first and last matrix index its sub-expression covers together with
    the scalar multiplications it costs, so the cost of joining a left part
    ``Ai..Ak`` to a right part ``A(k+1)..Aj`` is ``p[i-1] * p[k] * p[j]``,
    read straight off the two returned spans.

    Order is enforced as the leaves are read: the next leaf must name
    exactly the next matrix. That one check rejects a skipped, repeated,
    reordered or out-of-range matrix, and it guarantees that the left and
    right halves of every product are adjacent sub-chains, which is what
    makes the join cost above correct.

    Recursion depth is the nesting depth of the parentheses. A valid
    expression for ``n`` matrices nests at most ``n - 1`` deep, so anything
    deeper is rejected before it can recurse further; malformed input
    therefore always raises :class:`ValueError`, never
    :class:`RecursionError`, and valid input uses at most ``n`` frames.
    """

    def __init__(self, expr: str, p: List[int]) -> None:
        self._expr = expr
        self._p = p
        self._n = len(p) - 1
        self._pos = 0
        self._next = 1

    def parse(self) -> int:
        """Parse the whole string and return its total cost.

        Returns:
            The scalar multiplication count of the expression.

        Raises:
            ValueError: If the string is malformed, has trailing text, or
                does not name A1..An exactly once each, in order.

        Time Complexity:
            O(L) for a string of length L - each character is read once.

        Space Complexity:
            O(d) for nesting depth d, at most n - 1 - the call stack.

        Examples:
            >>> _ChainParser("(A1A2)", [10, 20, 30]).parse()
            6000
        """
        _, _, cost = self._parse_expr(0)
        if self._pos != len(self._expr):
            raise ValueError(
                f"unexpected {self._expr[self._pos]!r} at position {self._pos} "
                "after a complete expression"
            )
        if self._next != self._n + 1:
            raise ValueError(
                f"expression names A1..A{self._next - 1} but p describes "
                f"{self._n} matrices"
            )
        return cost

    def _found(self) -> str:
        """Describe the character at the cursor, for error messages."""
        if self._pos >= len(self._expr):
            return "end of input"
        return repr(self._expr[self._pos])

    def _parse_expr(self, depth: int) -> Tuple[int, int, int]:
        """Parse one ``expr`` starting at the cursor.

        Args:
            depth: How many '(' enclose the cursor.

        Returns:
            ``(first, last, cost)``: the span of matrices covered and the
            cost of multiplying them out in the written order.

        Raises:
            ValueError: On any deviation from the grammar or the order.
        """
        if self._pos < len(self._expr):
            char = self._expr[self._pos]
            if char == "A":
                index = self._leaf()
                return index, index, 0
            if char == "(":
                return self._product(depth)
        raise ValueError(
            f"expected 'A' or '(' at position {self._pos}, got {self._found()}"
        )

    def _leaf(self) -> int:
        """Parse ``'A' digits`` and return the matrix index it names."""
        start = self._pos
        self._pos += 1
        digits_start = self._pos
        while self._pos < len(self._expr) and self._expr[self._pos] in _DIGITS:
            self._pos += 1
        digits = self._expr[digits_start : self._pos]
        if not digits:
            raise ValueError(
                f"expected a digit after 'A' at position {self._pos}, "
                f"got {self._found()}"
            )
        if digits[0] == "0":
            raise ValueError(
                f"'A{digits}' at position {start} is not a matrix name "
                "(A1, A2, ... only)"
            )
        index = int(digits)
        if index > self._n:
            raise ValueError(
                f"A{index} at position {start} does not exist: p describes "
                f"{self._n} matrices"
            )
        if index != self._next:
            raise ValueError(
                f"expected A{self._next} at position {start}, got A{index} "
                f"(A1..A{self._n} each once, in order)"
            )
        self._next += 1
        return index

    def _product(self, depth: int) -> Tuple[int, int, int]:
        """Parse ``'(' expr expr ')'`` and price the join of its halves."""
        if depth >= self._n - 1:
            raise ValueError(
                f"too many '(' at position {self._pos}: n={self._n} allows "
                f"nesting depth {self._n - 1}"
            )
        self._pos += 1
        first, split, left_cost = self._parse_expr(depth + 1)
        _, last, right_cost = self._parse_expr(depth + 1)
        if self._pos >= len(self._expr) or self._expr[self._pos] != ")":
            raise ValueError(
                f"expected ')' at position {self._pos}, got {self._found()}"
            )
        self._pos += 1
        join = self._p[first - 1] * self._p[split] * self._p[last]
        return first, last, left_cost + right_cost + join


def parenthesization_cost(p: Sequence[int], expr: str) -> int:
    """Return the scalar multiplications a written parenthesization costs.

    ``expr`` follows the grammar CLRS prints its answers in, with no
    spaces::

        expr := 'A' digits | '(' expr expr ')'

    so ``"((A1A2)A3)"`` means "multiply A1 by A2, then the result by A3".
    Matrix ``Ai`` is ``p[i-1] x p[i]``, and multiplying a ``a x b`` matrix
    by a ``b x c`` matrix costs ``a * b * c`` scalar multiplications. The
    cost of the whole expression is the sum over every product it writes.

    This exists so tests can check a matrix-chain answer **by its cost**.
    Ties for the optimum are common (any chain of equal square matrices
    ties everywhere), so comparing the returned string with one expected
    string fails on correct answers. Checking
    ``parenthesization_cost(p, answer) == optimal_cost`` accepts every
    optimal parenthesization and rejects every other one.

    For that check to mean anything the parser has to be strict. The
    expression must use A1, A2, ..., An, each exactly once, in that order,
    where ``n = len(p) - 1``; a skipped, repeated, reordered or
    out-of-range matrix, a stray character, a missing parenthesis,
    redundant parentheses such as ``"(A1)"``, or trailing text all raise
    :class:`ValueError`. A single matrix ``"A1"`` costs 0.

    Args:
        p: The dimension list, ``len(p) >= 2`` positive ints.
        expr: The parenthesization to price.

    Returns:
        The exact number of scalar multiplications, as an ``int``.

    Raises:
        TypeError: If ``expr`` is not a ``str``, or an entry of ``p`` is
            not an ``int`` (``bool`` included).
        ValueError: If ``p`` is too short or holds a non-positive entry, or
            ``expr`` is malformed or does not name A1..An once each, in
            order.

    Time Complexity:
        O(n + L) for ``n`` matrices and a string of length ``L`` - each
        character is read once and each product costs O(1) to price.

    Space Complexity:
        O(n) - the copy of ``p`` and a call stack at most ``n`` frames deep.

    Examples:
        The CLRS 4e section 14.2 instance and its optimal answer:

        >>> p = [30, 35, 15, 5, 10, 20, 25]
        >>> parenthesization_cost(p, "((A1(A2A3))((A4A5)A6))")
        15125

        Same chain, worse order:

        >>> parenthesization_cost(p, "(((((A1A2)A3)A4)A5)A6)")
        40500
        >>> parenthesization_cost([10, 20, 30], "(A1A2)")
        6000
        >>> parenthesization_cost([10, 100, 5, 50], "((A1A2)A3)")
        7500
        >>> parenthesization_cost([10, 100, 5, 50], "(A1(A2A3))")
        75000
        >>> parenthesization_cost([10, 20], "A1")
        0

        Anything that is not A1..An once each, in order, is refused:

        >>> parenthesization_cost([10, 100, 5, 50], "((A1A3)A2)")
        Traceback (most recent call last):
            ...
        ValueError: expected A2 at position 4, got A3 (A1..A3 each once, in order)
        >>> parenthesization_cost([10, 100, 5, 50], "(A1A2)")
        Traceback (most recent call last):
            ...
        ValueError: expression names A1..A2 but p describes 3 matrices
        >>> parenthesization_cost([10, 100, 5], "(A1A3)")
        Traceback (most recent call last):
            ...
        ValueError: A3 at position 3 does not exist: p describes 2 matrices
        >>> parenthesization_cost([10, 100, 5], "(A1 A2)")
        Traceback (most recent call last):
            ...
        ValueError: expected 'A' or '(' at position 3, got ' '
        >>> parenthesization_cost([10, 100, 5], "(A1A2")
        Traceback (most recent call last):
            ...
        ValueError: expected ')' at position 5, got end of input
        >>> parenthesization_cost([10, 100], "(A1)")
        Traceback (most recent call last):
            ...
        ValueError: too many '(' at position 0: n=1 allows nesting depth 0
        >>> parenthesization_cost([10, 100, 5], "(A1A2)A3")
        Traceback (most recent call last):
            ...
        ValueError: unexpected 'A' at position 6 after a complete expression
        >>> parenthesization_cost([10, 100, 5], "(A01A2)")
        Traceback (most recent call last):
            ...
        ValueError: 'A01' at position 1 is not a matrix name (A1, A2, ... only)
        >>> parenthesization_cost([10, 100, 5], "")
        Traceback (most recent call last):
            ...
        ValueError: expected 'A' or '(' at position 0, got end of input
        >>> parenthesization_cost([10, 100, 5], None)
        Traceback (most recent call last):
            ...
        TypeError: expr must be a str, got NoneType
    """
    dims = _require_dimensions(p)
    if not isinstance(expr, str):
        raise TypeError(f"expr must be a str, got {type(expr).__name__}")
    return _ChainParser(expr, dims).parse()


def _format_entry(value: Any) -> str:
    """Render one matrix entry for :func:`format_matrix`.

    Integer-valued floats print without the trailing ``.0`` so that a
    matrix built from a :class:`Graph` (float weights) and one built by
    :func:`random_weight_matrix` (int weights) read the same.

    Args:
        value: The entry to render.

    Returns:
        ``"-"`` for ``None``, ``"inf"`` / ``"-inf"`` for infinities, the
        integer for an integer-valued float, and ``repr`` otherwise.

    Time Complexity:
        O(1) for numbers of bounded size.

    Space Complexity:
        O(1).

    Examples:
        >>> [_format_entry(v) for v in (None, INF, -INF, 3.0, 2.5, 7, "x")]
        ['-', 'inf', '-inf', '3', '2.5', '7', 'x']
    """
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return str(value)


def format_matrix(m: Sequence[Sequence[Any]]) -> str:
    """Render a matrix as aligned text, one row per line.

    Each column is right-aligned to its own widest entry and columns are
    separated by two spaces, so a distance matrix, a predecessor matrix or
    a dynamic-programming table can be printed straight into a report.
    :data:`INF` shows as ``inf`` and ``None`` (an empty predecessor or an
    unused table cell) shows as ``-``.

    Args:
        m: The matrix, as a sequence of rows. Rows may differ in length.

    Returns:
        The rendered text with no trailing newline; ``""`` for an empty
        matrix.

    Time Complexity:
        O(r * c) - every entry is rendered once and measured once.

    Space Complexity:
        O(r * c) - the rendered strings.

    Examples:
        >>> print(format_matrix([[0, 12, INF], [None, 0, -3.5]]))
        0  12   inf
        -   0  -3.5
        >>> print(format_matrix([[1]]))
        1
        >>> format_matrix([])
        ''
    """
    cells = [[_format_entry(value) for value in row] for row in m]
    widths: List[int] = []
    for row in cells:
        for column, text in enumerate(row):
            if column == len(widths):
                widths.append(len(text))
            elif len(text) > widths[column]:
                widths[column] = len(text)
    return "\n".join(
        "  ".join(text.rjust(widths[column]) for column, text in enumerate(row))
        for row in cells
    )
