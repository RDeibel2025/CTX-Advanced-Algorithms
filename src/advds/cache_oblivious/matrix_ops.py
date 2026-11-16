"""Naive and cache-oblivious matrix transpose and multiply, Z order, and their memory traces.

The idea
--------
A cache-oblivious algorithm halves its problem until a piece fits in a cache
block, without ever being told the block size B or the cache size M. Once a
piece fits, every word of every block it loads is used before the block is
evicted, so the algorithm is close to optimal at every level of a memory
hierarchy at once (Frigo, Leiserson, Prokop and Ramachandran, 1999).

* :func:`naive_transpose` copies ``A[i][j]`` to ``B[j][i]`` row by row. It
  reads A along rows but writes B down a column, so once a column of B spans
  more blocks than the cache holds, every write misses: Theta(n m) misses.
* :func:`co_transpose` splits the LONGER dimension in half (rows on ties)
  until the piece is at most ``base`` by ``base``, then runs the same row by
  row loop on the piece. A piece whose rows of A and columns of B both fit in
  the cache costs only the blocks it covers, so the transpose costs
  Theta(1 + n m / B) misses under the tall-cache assumption M = Omega(B^2).
* :func:`naive_matmul` is the i-j-k triple loop. For each ``(i, j)`` it walks
  a whole column of B, so for large n almost every B access misses:
  Theta(n^3) misses in the worst case.
* :func:`co_matmul` splits the largest of the three loop dimensions i, j and
  k in half (ties: i before j before k) until all three are at most ``base``,
  then runs the i-j-k loop on the piece, accumulating into a preallocated C.
  This gives Theta(n^2 / B + n^3 / (B sqrt(M))) misses.
* :func:`z_order` lists the cells of an n x n matrix in the order the
  recursive split visits them at ``base=1``. For a power-of-two n this is
  exactly the Z (Morton) order: cell ``(i, j)`` comes at the position given
  by interleaving the bits of i and j.

One recursion, shared
---------------------
The recursion lives in exactly one place per operation: :func:`_transpose_walk`
and :func:`_matmul_walk`. Each takes a *visitor* and calls it once for every
base-case piece, in order. The real functions pass a visitor that reads and
writes the lists; the trace functions pass a visitor with the same loop nest
that calls ``cache.access`` where the real one reads and writes. The naive
functions are that same loop nest applied once to the whole matrix. So the
traced address sequence is, by construction, the real loop order.

Simulated memory layout for the traces
--------------------------------------
Matrices are square n x n and stored row-major, one word per entry:

* Transpose ``B = A^T``: ``A[i][j]`` is address ``i*n + j`` and ``B[j][i]`` is
  address ``n*n + j*n + i``. Each inner step reads A, then writes B.
* Multiply ``C = A B``: ``A[i][k]`` is ``i*n + k``, ``B[k][j]`` is
  ``n*n + k*n + j`` and ``C[i][j]`` is ``2*n*n + i*n + j``. Each innermost
  step reads A, then B. ``C[i][j]`` is accessed once (one read-modify-write)
  after each ``(i, j)`` loop over k finishes; in the recursive version that is
  once after each base-case k loop.

:func:`trace_transpose` and :func:`trace_matmul` replay these accesses into an
:class:`~src.advds.cache_oblivious.cache_sim.LRUCache` and return its miss
count. They always use ``base=1`` (fully oblivious: the recursion goes all the
way down, so no block size is baked in). The real :func:`co_transpose` and
:func:`co_matmul` default to ``base=1`` too, but the benchmark runs them with
``base=16`` for speed, because a Python call per entry costs more than the
cache misses it saves.

Conventions: indices are 0-based, matrices are lists of row lists (numpy
arrays are accepted as input), and no numpy routine computes a transpose or a
product here; the tests use ``.T`` and ``@`` only as the independent
cross-check.

Examples:
    >>> co_transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    >>> co_matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]
    >>> from src.advds.cache_oblivious.cache_sim import LRUCache
    >>> trace_transpose(64, "naive", LRUCache(1024, 16))
    4352
    >>> trace_transpose(64, "recursive", LRUCache(1024, 16))
    512

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numbers
from typing import Any, Callable, List, Tuple

import numpy as np

__all__ = [
    "METHODS",
    "co_matmul",
    "co_transpose",
    "naive_matmul",
    "naive_transpose",
    "trace_matmul",
    "trace_transpose",
    "z_order",
]

#: The two trace methods: the plain loops and the cache-oblivious recursion.
METHODS: Tuple[str, str] = ("naive", "recursive")

Matrix = List[List[Any]]


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _as_rows(A: Any, name: str) -> Tuple[Matrix, int, int]:
    """Copy a 2-D matrix into a list of row lists and return it with its shape.

    Args:
        A: A list of equal-length rows, or a 2-D numpy array.
        name: The argument name used in error messages.

    Returns:
        ``(rows, n_rows, n_cols)``. For an empty list the column count is 0;
        a numpy array keeps its own shape.

    Raises:
        ValueError: If ``A`` is not two-dimensional or its rows differ in
            length.

    Time Complexity:
        O(n m) for an n x m matrix.

    Space Complexity:
        O(n m) - the copy.

    Examples:
        >>> _as_rows([[1, 2], [3, 4]], "A")
        ([[1, 2], [3, 4]], 2, 2)
        >>> _as_rows(np.zeros((0, 3)), "A")
        ([], 0, 3)
        >>> _as_rows([[1, 2], [3]], "A")
        Traceback (most recent call last):
        ...
        ValueError: A must be rectangular: row 1 has length 1, expected 2
        >>> _as_rows([1, 2], "A")
        Traceback (most recent call last):
        ...
        ValueError: A must be a 2-D matrix (a sequence of rows)
    """
    if isinstance(A, np.ndarray):
        if A.ndim != 2:
            raise ValueError(f"{name} must be a 2-D matrix, got shape {A.shape}")
        return A.tolist(), int(A.shape[0]), int(A.shape[1])
    try:
        rows = [list(row) for row in A]
    except TypeError:
        raise ValueError(f"{name} must be a 2-D matrix (a sequence of rows)") from None
    n_cols = len(rows[0]) if rows else 0
    for r, row in enumerate(rows):
        if len(row) != n_cols:
            raise ValueError(
                f"{name} must be rectangular: row {r} has length {len(row)}, expected {n_cols}"
            )
    return rows, len(rows), n_cols


def _check_base(base: Any) -> int:
    """Return ``base`` if it is a positive int (the largest piece side handled directly).

    Args:
        base: The recursion cutoff.

    Returns:
        ``int(base)``.

    Raises:
        TypeError: If ``base`` is not an int (``bool`` rejected).
        ValueError: If ``base < 1``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_base(16)
        16
        >>> _check_base(0)
        Traceback (most recent call last):
        ...
        ValueError: base must be at least 1, got 0
    """
    if isinstance(base, bool) or not isinstance(base, numbers.Integral):
        raise TypeError(f"base must be an int, got {type(base).__name__}")
    if base < 1:
        raise ValueError(f"base must be at least 1, got {base}")
    return int(base)


def _check_size(n: Any) -> int:
    """Return ``n`` if it is a non-negative int (a square matrix side).

    Args:
        n: The matrix side.

    Returns:
        ``int(n)``.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected).
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_size(8)
        8
        >>> _check_size(-1)
        Traceback (most recent call last):
        ...
        ValueError: n must be non-negative, got -1
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return int(n)


def _check_method(method: Any) -> str:
    """Return ``method`` if it is ``"naive"`` or ``"recursive"``.

    Args:
        method: The trace method name.

    Returns:
        ``method`` unchanged.

    Raises:
        ValueError: For any other value.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_method("naive")
        'naive'
        >>> _check_method("blocked")
        Traceback (most recent call last):
        ...
        ValueError: method must be one of ('naive', 'recursive'), got 'blocked'
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return method


# ---------------------------------------------------------------------------
# The shared recursions
# ---------------------------------------------------------------------------


def _transpose_walk(
    r0: int, r1: int, c0: int, c1: int, base: int, visit: Callable[[int, int, int, int], None]
) -> None:
    """Split the block ``[r0, r1) x [c0, c1)`` recursively and visit each base-case piece.

    If both sides are at most ``base`` the piece is a base case and
    ``visit(r0, r1, c0, c1)`` is called. Otherwise the LONGER side is split in
    half (rows when the sides are equal), the first half of length
    ``side // 2``, and the two halves are walked in order. Both
    :func:`co_transpose` and :func:`trace_transpose` (and :func:`z_order`) go
    through this one function, so they visit the pieces in the same order.

    Args:
        r0: First row of the block.
        r1: One past the last row.
        c0: First column of the block.
        c1: One past the last column.
        base: The largest side handled directly, at least 1.
        visit: Called once per base-case piece with ``(r0, r1, c0, c1)``.

    Returns:
        None.

    Raises:
        Nothing itself; whatever ``visit`` raises propagates.

    Time Complexity:
        O((1 + rows / base) * (1 + cols / base)) calls: every base piece
        has sides of at least about ``base / 2`` (or the full side, if that
        is shorter), so this bounds the leaves, and the recursion tree has
        fewer internal nodes than leaves. For a square side-n block this is
        O(1 + n^2 / base^2); a thin 1 x m block needs about m / base calls.

    Space Complexity:
        O(log rows + log cols) recursion depth.

    Examples:
        >>> pieces = []
        >>> _transpose_walk(0, 2, 0, 4, 1, lambda *p: pieces.append(p))
        >>> pieces[:4]
        [(0, 1, 0, 1), (0, 1, 1, 2), (1, 2, 0, 1), (1, 2, 1, 2)]
        >>> len(pieces)
        8
    """
    dr = r1 - r0
    dc = c1 - c0
    if dr <= base and dc <= base:
        visit(r0, r1, c0, c1)
    elif dr >= dc:
        mid = r0 + dr // 2
        _transpose_walk(r0, mid, c0, c1, base, visit)
        _transpose_walk(mid, r1, c0, c1, base, visit)
    else:
        mid = c0 + dc // 2
        _transpose_walk(r0, r1, c0, mid, base, visit)
        _transpose_walk(r0, r1, mid, c1, base, visit)


def _matmul_walk(
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    k0: int,
    k1: int,
    base: int,
    visit: Callable[[int, int, int, int, int, int], None],
) -> None:
    """Split the product box ``[i0, i1) x [j0, j1) x [k0, k1)`` recursively and visit each piece.

    The box describes the partial product ``C[i0:i1, j0:j1] +=
    A[i0:i1, k0:k1] B[k0:k1, j0:j1]``. If all three sides are at most
    ``base`` it is a base case and ``visit(i0, i1, j0, j1, k0, k1)`` is
    called. Otherwise the largest side is split in half (ties: i before j
    before k), the first half of length ``side // 2``, and the halves are
    walked in order. Splitting i or j gives two independent products;
    splitting k gives two products that add into the same C block, first the
    lower k half, then the upper. Both :func:`co_matmul` and
    :func:`trace_matmul` go through this one function.

    Args:
        i0: First row of A and C.
        i1: One past the last row.
        j0: First column of B and C.
        j1: One past the last column.
        k0: First column of A and row of B.
        k1: One past the last.
        base: The largest side handled directly, at least 1.
        visit: Called once per base-case piece with ``(i0, i1, j0, j1, k0, k1)``.

    Returns:
        None.

    Raises:
        Nothing itself; whatever ``visit`` raises propagates.

    Time Complexity:
        O((1 + n / base) * (1 + m / base) * (1 + p / base)) calls for an
        n x m times m x p box (every base piece has sides of at least about
        ``base / 2`` or the full side), which is O(1 + n^3 / base^3) for
        square n x n factors.

    Space Complexity:
        O(log n + log m + log p) recursion depth.

    Examples:
        >>> pieces = []
        >>> _matmul_walk(0, 2, 0, 2, 0, 2, 1, lambda *p: pieces.append(p))
        >>> pieces[:3]
        [(0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 1, 2), (0, 1, 1, 2, 0, 1)]
        >>> len(pieces)
        8
    """
    di = i1 - i0
    dj = j1 - j0
    dk = k1 - k0
    if di <= base and dj <= base and dk <= base:
        visit(i0, i1, j0, j1, k0, k1)
    elif di >= dj and di >= dk:
        mid = i0 + di // 2
        _matmul_walk(i0, mid, j0, j1, k0, k1, base, visit)
        _matmul_walk(mid, i1, j0, j1, k0, k1, base, visit)
    elif dj >= dk:
        mid = j0 + dj // 2
        _matmul_walk(i0, i1, j0, mid, k0, k1, base, visit)
        _matmul_walk(i0, i1, mid, j1, k0, k1, base, visit)
    else:
        mid = k0 + dk // 2
        _matmul_walk(i0, i1, j0, j1, k0, mid, base, visit)
        _matmul_walk(i0, i1, j0, j1, mid, k1, base, visit)


# ---------------------------------------------------------------------------
# Visitors: the base-case loop nests (real and traced)
# ---------------------------------------------------------------------------


def _transpose_kernel(A: Matrix, B: Matrix) -> Callable[[int, int, int, int], None]:
    """Return the visitor that copies a piece of A into B, row by row.

    The loop nest is ``for i in rows: for j in cols: B[j][i] = A[i][j]``,
    the same nest :func:`_transpose_trace_kernel` replays as addresses.

    Args:
        A: The source rows.
        B: The preallocated destination, ``len(A[0])`` rows of ``len(A)``.

    Returns:
        A function of ``(r0, r1, c0, c1)``.

    Raises:
        Nothing.

    Time Complexity:
        O(1) to build; each call costs O(piece area).

    Space Complexity:
        O(1).

    Examples:
        >>> A, B = [[1, 2], [3, 4]], [[0, 0], [0, 0]]
        >>> _transpose_kernel(A, B)(0, 2, 0, 2)
        >>> B
        [[1, 3], [2, 4]]
    """

    def visit(r0: int, r1: int, c0: int, c1: int) -> None:
        for i in range(r0, r1):
            Ai = A[i]
            for j in range(c0, c1):
                B[j][i] = Ai[j]  # read A[i][j], then write B[j][i]

    return visit


def _transpose_trace_kernel(n: int, cache: Any) -> Callable[[int, int, int, int], None]:
    """Return the visitor that replays a transpose piece as cache accesses (section 5.1 layout).

    Same nest as :func:`_transpose_kernel`; each step accesses ``i*n + j``
    (read ``A[i][j]``) and then ``n*n + j*n + i`` (write ``B[j][i]``).

    Args:
        n: The matrix side.
        cache: Anything with an ``access(address)`` method.

    Returns:
        A function of ``(r0, r1, c0, c1)``.

    Raises:
        Nothing.

    Time Complexity:
        O(1) to build; each call makes 2 x (piece area) accesses.

    Space Complexity:
        O(1).

    Examples:
        >>> class Log:
        ...     def __init__(self):
        ...         self.seen = []
        ...     def access(self, a):
        ...         self.seen.append(a)
        >>> log = Log()
        >>> _transpose_trace_kernel(2, log)(0, 1, 0, 2)
        >>> log.seen
        [0, 4, 1, 6]
    """
    access = cache.access
    nn = n * n

    def visit(r0: int, r1: int, c0: int, c1: int) -> None:
        for i in range(r0, r1):
            for j in range(c0, c1):
                access(i * n + j)  # read A[i][j]
                access(nn + j * n + i)  # write B[j][i]

    return visit


def _matmul_kernel(A: Matrix, B: Matrix, C: Matrix) -> Callable[[int, int, int, int, int, int], None]:
    """Return the visitor that adds one piece's partial product into C, in i-j-k order.

    For each ``(i, j)`` it sums ``A[i][k] * B[k][j]`` over the piece's k
    range and then adds the sum to ``C[i][j]`` once, the nest
    :func:`_matmul_trace_kernel` replays as addresses.

    Args:
        A: The left factor rows.
        B: The right factor rows.
        C: The preallocated result, accumulated into.

    Returns:
        A function of ``(i0, i1, j0, j1, k0, k1)``.

    Raises:
        TypeError: When called, if the entries do not support ``*`` and ``+``.

    Time Complexity:
        O(1) to build; each call costs O(piece volume).

    Space Complexity:
        O(1).

    Examples:
        >>> C = [[0]]
        >>> visit = _matmul_kernel([[1, 2]], [[3], [4]], C)
        >>> visit(0, 1, 0, 1, 0, 1); visit(0, 1, 0, 1, 1, 2)
        >>> C
        [[11]]
    """

    def visit(i0: int, i1: int, j0: int, j1: int, k0: int, k1: int) -> None:
        for i in range(i0, i1):
            Ai = A[i]
            Ci = C[i]
            for j in range(j0, j1):
                s = 0
                for k in range(k0, k1):
                    s += Ai[k] * B[k][j]  # read A[i][k], then B[k][j]
                Ci[j] += s  # one read-modify-write of C[i][j]

    return visit


def _matmul_trace_kernel(n: int, cache: Any) -> Callable[[int, int, int, int, int, int], None]:
    """Return the visitor that replays a multiply piece as cache accesses (section 5.1 layout).

    Same nest as :func:`_matmul_kernel`: each innermost step accesses
    ``i*n + k`` (``A[i][k]``) then ``n*n + k*n + j`` (``B[k][j]``), and after
    each k loop ``2*n*n + i*n + j`` (``C[i][j]``) is accessed once.

    Args:
        n: The matrix side.
        cache: Anything with an ``access(address)`` method.

    Returns:
        A function of ``(i0, i1, j0, j1, k0, k1)``.

    Raises:
        Nothing.

    Time Complexity:
        O(1) to build; each call makes 2 x volume + (i, j area) accesses.

    Space Complexity:
        O(1).

    Examples:
        >>> class Log:
        ...     def __init__(self):
        ...         self.seen = []
        ...     def access(self, a):
        ...         self.seen.append(a)
        >>> log = Log()
        >>> _matmul_trace_kernel(2, log)(0, 1, 0, 1, 0, 2)
        >>> log.seen
        [0, 4, 1, 6, 8]
    """
    access = cache.access
    nn = n * n
    c_base = 2 * nn

    def visit(i0: int, i1: int, j0: int, j1: int, k0: int, k1: int) -> None:
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    access(i * n + k)  # read A[i][k]
                    access(nn + k * n + j)  # read B[k][j]
                access(c_base + i * n + j)  # update C[i][j]

    return visit


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def naive_transpose(A: Any) -> Matrix:
    """Transpose a matrix row by row: ``B[j][i] = A[i][j]`` for i, then j.

    This is the base-case loop nest of :func:`co_transpose` applied once to
    the whole matrix. Reads of A are sequential; writes of B jump a whole row
    of B each step, which is what makes it cache-unfriendly.

    Args:
        A: An n x m matrix, as a list of equal-length rows or a 2-D numpy
            array.

    Returns:
        The m x n transpose as a new list of row lists.

    Raises:
        ValueError: If ``A`` is not a rectangular 2-D matrix.

    Time Complexity:
        Theta(n m). Simulated misses Theta(n m) once a column of B spans more
        blocks than the cache holds (one miss per write).

    Space Complexity:
        Theta(n m) for the result.

    Examples:
        >>> naive_transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> naive_transpose([[7]])
        [[7]]
        >>> naive_transpose([])
        []
    """
    rows, n_rows, n_cols = _as_rows(A, "A")
    B: Matrix = [[None] * n_rows for _ in range(n_cols)]
    _transpose_kernel(rows, B)(0, n_rows, 0, n_cols)
    return B


def co_transpose(A: Any, base: int = 1) -> Matrix:
    """Transpose a matrix cache-obliviously: halve the longer side until a piece is small.

    The recursion (:func:`_transpose_walk`) splits the LONGER dimension in
    half, rows on ties, until both sides of a piece are at most ``base``,
    then copies the piece row by row. The default ``base=1`` is fully
    oblivious and is what the traces use; the benchmark runs the real
    function with ``base=16`` for speed, which changes only how many Python
    calls are made, not the result.

    Args:
        A: An n x m matrix, as a list of equal-length rows or a 2-D numpy
            array.
        base: The largest piece side copied directly, at least 1.

    Returns:
        The m x n transpose as a new list of row lists.

    Raises:
        ValueError: If ``A`` is not a rectangular 2-D matrix or ``base < 1``.
        TypeError: If ``base`` is not an int.

    Time Complexity:
        Theta(n m). Simulated misses Theta(1 + n m / B) under the tall-cache
        assumption M = Omega(B^2); for n = 64 at (M, B) = (1024, 16) that is
        exactly the lower bound 2 n^2 / B = 512.

    Space Complexity:
        Theta(n m) for the result plus O(log n + log m) recursion depth.

    Examples:
        >>> co_transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> co_transpose([[1, 2], [3, 4]], base=16)
        [[1, 3], [2, 4]]
        >>> co_transpose([[1]], base=0)
        Traceback (most recent call last):
        ...
        ValueError: base must be at least 1, got 0
    """
    base = _check_base(base)
    rows, n_rows, n_cols = _as_rows(A, "A")
    B: Matrix = [[None] * n_rows for _ in range(n_cols)]
    _transpose_walk(0, n_rows, 0, n_cols, base, _transpose_kernel(rows, B))
    return B


# ---------------------------------------------------------------------------
# Multiply
# ---------------------------------------------------------------------------


def _product_operands(A: Any, B: Any) -> Tuple[Matrix, Matrix, int, int, int]:
    """Check that ``A B`` is defined and return the rows and the three sizes.

    Args:
        A: An n x m matrix.
        B: An m x p matrix.

    Returns:
        ``(A_rows, B_rows, n, m, p)``.

    Raises:
        ValueError: If either is not a rectangular 2-D matrix or the inner
            sizes differ.

    Time Complexity:
        O(n m + m p) for the copies.

    Space Complexity:
        O(n m + m p).

    Examples:
        >>> _product_operands([[1, 2]], [[3], [4]])[2:]
        (1, 2, 1)
        >>> _product_operands([[1, 2]], [[3, 4]])
        Traceback (most recent call last):
        ...
        ValueError: shapes (1, 2) and (1, 2) are not conformable
    """
    A_rows, n, m = _as_rows(A, "A")
    B_rows, m2, p = _as_rows(B, "B")
    if m != m2:
        raise ValueError(f"shapes {(n, m)} and {(m2, p)} are not conformable")
    return A_rows, B_rows, n, m, p


def naive_matmul(A: Any, B: Any) -> Matrix:
    """Multiply two matrices with the i-j-k triple loop.

    For each ``(i, j)`` the k loop forms ``sum_k A[i][k] * B[k][j]`` and adds
    it to ``C[i][j]`` once. This is the base-case loop nest of
    :func:`co_matmul` applied once to the whole product. The k loop walks a
    column of B, one block per step once rows are longer than a block.

    Args:
        A: An n x m matrix (list of rows or 2-D numpy array).
        B: An m x p matrix.

    Returns:
        The n x p product as a new list of row lists. Integer inputs give
        exact integer results.

    Raises:
        ValueError: If an operand is not a rectangular 2-D matrix or the
            inner sizes differ.

    Time Complexity:
        Theta(n m p) multiply-adds. Simulated misses for n x n inputs
        Theta(n^3) once a column of B no longer fits in the cache.

    Space Complexity:
        Theta(n p) for the result.

    Examples:
        >>> naive_matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
        >>> naive_matmul([[1, 2, 3]], [[1], [1], [1]])
        [[6]]
    """
    A_rows, B_rows, n, m, p = _product_operands(A, B)
    C: Matrix = [[0] * p for _ in range(n)]
    _matmul_kernel(A_rows, B_rows, C)(0, n, 0, p, 0, m)
    return C


def co_matmul(A: Any, B: Any, base: int = 1) -> Matrix:
    """Multiply two matrices cache-obliviously: halve the largest of i, j, k until small.

    The recursion (:func:`_matmul_walk`) splits the largest of the three
    loop dimensions in half (ties: i before j before k) until all three are
    at most ``base``, then runs the i-j-k loop on the piece, adding into the
    preallocated result C. The default ``base=1`` is fully oblivious and is
    what the traces use; the benchmark runs the real function with
    ``base=16`` for speed.

    Args:
        A: An n x m matrix (list of rows or 2-D numpy array).
        B: An m x p matrix.
        base: The largest piece side multiplied directly, at least 1.

    Returns:
        The n x p product as a new list of row lists.

    Raises:
        ValueError: If an operand is not a rectangular 2-D matrix, the inner
            sizes differ, or ``base < 1``.
        TypeError: If ``base`` is not an int.

    Time Complexity:
        Theta(n m p) multiply-adds. Simulated misses for n x n inputs
        Theta(n^2 / B + n^3 / (B sqrt(M))).

    Space Complexity:
        Theta(n p) for the result plus O(log n + log m + log p) recursion
        depth.

    Examples:
        >>> co_matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
        >>> co_matmul([[1, 2, 3]], [[1], [1], [1]], base=16)
        [[6]]
    """
    base = _check_base(base)
    A_rows, B_rows, n, m, p = _product_operands(A, B)
    C: Matrix = [[0] * p for _ in range(n)]
    _matmul_walk(0, n, 0, p, 0, m, base, _matmul_kernel(A_rows, B_rows, C))
    return C


# ---------------------------------------------------------------------------
# Z order
# ---------------------------------------------------------------------------


def z_order(n: int) -> List[Tuple[int, int]]:
    """List the cells of an n x n matrix in Z order (the recursive matrix traversal).

    The cells come in the order the recursive split of :func:`_transpose_walk`
    visits them at ``base=1``: halve the longer side, rows on ties, top half
    before bottom and left half before right. For a power-of-two n this is
    the Z (Morton) order, where cell ``(i, j)`` sits at the index formed by
    interleaving the bits of i and j, and it is exactly the order in which
    :func:`co_transpose` with ``base=1`` copies the cells.

    Args:
        n: The matrix side, a non-negative int.

    Returns:
        A list of ``n * n`` distinct ``(row, col)`` tuples.

    Raises:
        TypeError: If ``n`` is not an int.
        ValueError: If ``n`` is negative.

    Time Complexity:
        Theta(n^2).

    Space Complexity:
        Theta(n^2) for the result plus O(log n) recursion depth.

    Examples:
        >>> z_order(2)
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        >>> z_order(4)[:8]
        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        >>> z_order(0)
        []
    """
    n = _check_size(n)
    cells: List[Tuple[int, int]] = []

    def visit(r0: int, r1: int, c0: int, c1: int) -> None:
        for i in range(r0, r1):
            for j in range(c0, c1):
                cells.append((i, j))

    _transpose_walk(0, n, 0, n, 1, visit)
    return cells


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def trace_transpose(n: int, method: str, cache: Any) -> int:
    """Replay an n x n transpose as addresses into ``cache`` and return its miss count.

    Layout (section 5.1): ``A[i][j]`` is address ``i*n + j``, ``B[j][i]`` is
    ``n*n + j*n + i``, and each inner step reads A, then writes B. ``"naive"``
    replays :func:`naive_transpose`; ``"recursive"`` replays
    :func:`co_transpose` with ``base=1`` (fully oblivious) through the same
    recursion the real function uses. The cache is not reset first.

    Args:
        n: The matrix side, a non-negative int.
        method: ``"naive"`` or ``"recursive"``.
        cache: An :class:`~src.advds.cache_oblivious.cache_sim.LRUCache`, or
            anything with ``access(address)`` and a ``misses`` attribute.

    Returns:
        ``cache.misses`` after the replay (cumulative, since the cache is not
        reset).

    Raises:
        TypeError: If ``n`` is not an int.
        ValueError: If ``n`` is negative or ``method`` is unknown.

    Time Complexity:
        Theta(n^2) accesses, 2 per cell.

    Space Complexity:
        O(log n) recursion depth.

    Examples:
        >>> from src.advds.cache_oblivious.cache_sim import LRUCache
        >>> trace_transpose(64, "naive", LRUCache(256, 8))
        4608
        >>> trace_transpose(64, "recursive", LRUCache(256, 8))
        1024
        >>> cache = LRUCache(1024, 16)
        >>> trace_transpose(8, "naive", cache), cache.accesses
        (8, 128)
    """
    n = _check_size(n)
    method = _check_method(method)
    visit = _transpose_trace_kernel(n, cache)
    if method == "naive":
        visit(0, n, 0, n)
    else:
        _transpose_walk(0, n, 0, n, 1, visit)
    return cache.misses


def trace_matmul(n: int, method: str, cache: Any) -> int:
    """Replay an n x n multiply ``C = A B`` as addresses into ``cache`` and return its miss count.

    Layout (section 5.1): ``A[i][k]`` is ``i*n + k``, ``B[k][j]`` is
    ``n*n + k*n + j``, ``C[i][j]`` is ``2*n*n + i*n + j``. Each innermost step
    reads A then B, and C is accessed once after each ``(i, j)`` loop over k
    (in the recursive version once after each base-case k loop, so at
    ``base=1`` once per multiply-add). ``"naive"`` replays
    :func:`naive_matmul`; ``"recursive"`` replays :func:`co_matmul` with
    ``base=1`` through the same recursion the real function uses. The cache
    is not reset first.

    Args:
        n: The matrix side, a non-negative int.
        method: ``"naive"`` or ``"recursive"``.
        cache: An :class:`~src.advds.cache_oblivious.cache_sim.LRUCache`, or
            anything with ``access(address)`` and a ``misses`` attribute.

    Returns:
        ``cache.misses`` after the replay (cumulative, since the cache is not
        reset).

    Raises:
        TypeError: If ``n`` is not an int.
        ValueError: If ``n`` is negative or ``method`` is unknown.

    Time Complexity:
        Theta(n^3) accesses: ``2 n^3 + n^2`` naive, ``3 n^3`` recursive.

    Space Complexity:
        O(log n) recursion depth.

    Examples:
        >>> from src.advds.cache_oblivious.cache_sim import LRUCache
        >>> trace_matmul(32, "naive", LRUCache(1024, 16))
        2176
        >>> trace_matmul(32, "recursive", LRUCache(1024, 16))
        320
    """
    n = _check_size(n)
    method = _check_method(method)
    visit = _matmul_trace_kernel(n, cache)
    if method == "naive":
        visit(0, n, 0, n, 0, n)
    else:
        _matmul_walk(0, n, 0, n, 0, n, 1, visit)
    return cache.misses
