"""Matrix multiplication, LU and QR factorizations, and the power method.

Multiplication
--------------
Three ways to compute C = A B for an n x m matrix A and an m x p matrix B:

* :func:`matmul_classical` is the textbook triple loop in pure Python on
  lists, in i-k-j order so the innermost loop walks one row of B and one row
  of C. It is the honest Theta(n^3) baseline: every one of the n m p
  multiply-adds is a Python operation.
* :func:`matmul_blocked` tiles the three loops into ``block x block`` tiles,
  so each tile of A, B and C is reused while it is still in cache. Inside a
  tile pair the product is built one k at a time: every row of the C tile
  receives ``A[r, k]`` times row k of the B tile, a row operation numpy
  applies to all rows of the tile at once. No ``@`` is used on a tile. Still
  Theta(n^3) multiply-adds, but with far less interpreter overhead.
* :func:`strassen` pads both operands to one power-of-two size s, splits each
  into four s/2 x s/2 blocks and forms the product from seven block products
  (instead of eight) and eighteen block additions. The recursion
  T(s) = 7 T(s/2) + Theta(s^2) gives O(s^log2(7)) = O(s^2.807). Below the
  ``cutoff`` it calls :func:`matmul_blocked`, because for small blocks the
  extra additions cost more than the multiplication they save.

Solving Ax = b: LU with partial pivoting
----------------------------------------
:func:`lu_decompose` runs Gaussian elimination and records it as PA = LU: L
is unit lower triangular and holds the multipliers, U is upper triangular,
and the permutation matrix P records the row swaps. With ``pivot=True`` each
column's pivot is the entry of largest magnitude on or below the diagonal, so
every multiplier has magnitude at most 1 and rounding errors cannot be
amplified by a huge multiplier. ``pivot=False`` exists only to show what goes
wrong without it: on ``[[1e-20, 1], [1, 1]]`` the multiplier 1e20 swamps the
second row and the computed solution is ``[0, 1]`` instead of ``[1, 1]``.

LU is a round trip: it moves the problem where it is easy (A becomes the
triangular factors L and U), does the easy thing (one forward and one back
substitution, O(n^2) each) and moves back (the row permutation P is undone
by applying it to b). The factorization costs about 2n^3/3 flops once and is
reused for every right-hand side, which is how the condition-number code in
``numerical.py`` builds A^-1 column by column.

QR: orthogonality and its loss
------------------------------
QR writes A = QR with orthonormal columns in Q and upper triangular R, the
same round trip with an orthogonal change of basis. Three algorithms:

* Classical Gram-Schmidt (CGS) projects each original column against all
  earlier q's at once. Rounding in the earlier q's is never corrected, so on
  an ill-conditioned matrix such as the Hilbert matrix the columns of Q stop
  being orthogonal: ``||Q^T Q - I||_F`` reaches order 1.
* Modified Gram-Schmidt (MGS) subtracts one projection at a time from the
  updated vector. It is the same arithmetic in exact arithmetic but loses
  orthogonality only in proportion to the condition number.
* Householder reflections build Q as a product of exact reflections, so Q is
  orthogonal to machine precision whatever the conditioning. It costs
  2mn^2 - 2n^3/3 flops against 2mn^2 for Gram-Schmidt.

All three return the thin factorization (Q is m x n, R is n x n) for m >= n.

Eigenvalues: the power method
-----------------------------
:func:`power_method` multiplies a seeded random start vector by A again and
again. The component along the dominant eigenvector grows fastest, so the
normalized iterate converges to it at the rate ``|lambda_2 / lambda_1|`` per
step, and the Rayleigh quotient ``v^T A v`` gives the eigenvalue.

Rule for this package: numpy arrays and element-wise operations are used
freely, but no factorization, solver, inverse or eigenvalue routine from
numpy's or scipy's linear-algebra modules is called; the tests use those only
as the independent cross-check.

Examples:
    >>> P, L, U = lu_decompose([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    >>> U.tolist()
    [[4.0, -6.0, 0.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]
    >>> solve([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], [5, -2, 9]).tolist()
    [1.0, 1.0, 2.0]
    >>> strassen([[1, 2], [3, 4]], [[5, 6], [7, 8]]).tolist()
    [[19.0, 22.0], [43.0, 50.0]]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "matmul_classical",
    "matmul_blocked",
    "strassen",
    "lu_decompose",
    "lu_solve",
    "solve",
    "determinant",
    "qr_householder",
    "qr_gram_schmidt",
    "power_method",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_matrix(X, name: str) -> np.ndarray:
    """Convert ``X`` to a 2-D numpy array, rejecting anything else.

    Args:
        X: Array-like input.
        name: The argument name used in the error message.

    Returns:
        ``np.asarray(X)``, guaranteed 2-D.

    Raises:
        ValueError: If ``X`` is not two-dimensional.

    Time Complexity:
        O(1) for an ndarray, O(size) for a nested list.

    Space Complexity:
        O(size) when a copy is needed.

    Examples:
        >>> _as_matrix([[1, 2]], "A").shape
        (1, 2)
        >>> _as_matrix([1, 2], "A")
        Traceback (most recent call last):
        ...
        ValueError: A must be a 2-D matrix, got shape (2,)
    """
    M = np.asarray(X)
    if M.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {M.shape}")
    return M


def _check_conformable(A: np.ndarray, B: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``A @ B`` is defined.

    Args:
        A: An n x m matrix.
        B: An m' x p matrix.

    Raises:
        ValueError: If m != m'.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_conformable(np.zeros((2, 3)), np.zeros((3, 4)))
        >>> _check_conformable(np.zeros((2, 3)), np.zeros((2, 3)))
        Traceback (most recent call last):
        ...
        ValueError: shapes (2, 3) and (2, 3) are not conformable
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"shapes {A.shape} and {B.shape} are not conformable")


def _result_dtype(A: np.ndarray, B: np.ndarray) -> np.dtype:
    """Return float64, or complex128 when either operand is complex.

    Args:
        A: First operand.
        B: Second operand.

    Returns:
        The dtype of the product array.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _result_dtype(np.ones((1, 1), dtype=int), np.ones((1, 1)))
        dtype('float64')
    """
    return np.result_type(A.dtype, B.dtype, np.float64)


def _check_square(A: np.ndarray, name: str = "A") -> int:
    """Return n for an n x n matrix, raising ``ValueError`` otherwise.

    Args:
        A: A 2-D array.
        name: The argument name used in the error message.

    Returns:
        The order n.

    Raises:
        ValueError: If ``A`` is not square.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_square(np.zeros((3, 3)))
        3
        >>> _check_square(np.zeros((2, 3)))
        Traceback (most recent call last):
        ...
        ValueError: A must be square, got shape (2, 3)
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")
    return A.shape[0]


def _vec_norm2(v: np.ndarray) -> float:
    """Return the Euclidean norm of a vector, written by hand.

    Args:
        v: A 1-D real array.

    Returns:
        ``sqrt(sum(v_i^2))``.

    Time Complexity:
        O(len(v)).

    Space Complexity:
        O(len(v)) for the element-wise square.

    Examples:
        >>> _vec_norm2(np.array([3.0, 4.0]))
        5.0
    """
    return float(np.sqrt((v * v).sum()))


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two that is at least ``n`` (1 for n <= 1).

    Args:
        n: A non-negative integer.

    Returns:
        ``2**ceil(log2(n))``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> [_next_power_of_two(k) for k in (1, 2, 7, 64, 130)]
        [1, 2, 8, 64, 256]
    """
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


def matmul_classical(A, B) -> np.ndarray:
    """Multiply two matrices with the pure-Python i-k-j triple loop.

    The operands are converted to nested Python lists and every multiply-add
    is a Python operation, so this is the true Theta(n m p) algorithm with no
    numpy help. The loop order is i-k-j: for each row i of A and each k, the
    scalar ``A[i][k]`` is multiplied into row k of B and accumulated into row
    i of C, so the innermost loop walks two rows contiguously.

    Args:
        A: An n x m matrix (array-like).
        B: An m x p matrix (array-like).

    Returns:
        The n x p product as a float64 (or complex128) ``np.ndarray``.

    Raises:
        ValueError: If either operand is not 2-D or the inner dimensions
            differ.

    Time Complexity:
        Theta(n m p); Theta(n^3) for square inputs.

    Space Complexity:
        Theta(n m + m p + n p) for the list copies and the result.

    Examples:
        >>> matmul_classical([[1, 2], [3, 4]], [[5, 6], [7, 8]]).tolist()
        [[19.0, 22.0], [43.0, 50.0]]
        >>> matmul_classical([[1, 2, 3]], [[1], [1], [1]]).tolist()
        [[6.0]]
    """
    A_arr = _as_matrix(A, "A")
    B_arr = _as_matrix(B, "B")
    _check_conformable(A_arr, B_arr)
    dtype = _result_dtype(A_arr, B_arr)
    n, m = A_arr.shape
    p = B_arr.shape[1]
    a_rows: List[list] = A_arr.astype(dtype).tolist()
    b_rows: List[list] = B_arr.astype(dtype).tolist()
    zero = dtype.type(0).item()
    C: List[list] = []
    for i in range(n):
        a_row = a_rows[i]
        c_row = [zero] * p
        for k in range(m):
            a_ik = a_row[k]
            b_row = b_rows[k]
            for j in range(p):
                c_row[j] += a_ik * b_row[j]
        C.append(c_row)
    return np.array(C, dtype=dtype).reshape(n, p)


def matmul_blocked(A, B, block: int = 64) -> np.ndarray:
    """Multiply two matrices with tiled (blocked) loops and numpy row operations.

    The i, j and k loops step in tiles of ``block``. For one tile of C and
    one pair of tiles of A and B, the tile product is accumulated one k at a
    time as a rank-1 update: every row r of the C tile gains ``A[r, k]``
    times row k of the B tile. numpy applies that scaled-row addition to all
    rows of the tile in one vectorized operation; no ``@`` is used on a tile.
    The multiply-add count is the classical n m p, but the work happens in
    tile-sized chunks that stay in cache and avoid per-element interpreter
    overhead.

    Args:
        A: An n x m matrix (array-like).
        B: An m x p matrix (array-like).
        block: Tile edge length, at least 1. Edge tiles may be smaller.

    Returns:
        The n x p product as a float64 (or complex128) ``np.ndarray``.

    Raises:
        ValueError: If either operand is not 2-D, the inner dimensions
            differ, or ``block < 1``.

    Time Complexity:
        Theta(n m p) multiply-adds, issued as Theta(n m p / block^2)
        vectorized tile operations.

    Space Complexity:
        Theta(n p) for the result plus Theta(block^2) scratch.

    Examples:
        >>> matmul_blocked([[1, 2], [3, 4]], [[5, 6], [7, 8]], block=1).tolist()
        [[19.0, 22.0], [43.0, 50.0]]
        >>> rng = np.random.default_rng(42)
        >>> X, Y = rng.standard_normal((5, 3)), rng.standard_normal((3, 4))
        >>> bool(np.allclose(matmul_blocked(X, Y, block=2), X @ Y))
        True
    """
    A_arr = _as_matrix(A, "A")
    B_arr = _as_matrix(B, "B")
    _check_conformable(A_arr, B_arr)
    block = int(block)
    if block < 1:
        raise ValueError(f"block must be at least 1, got {block}")
    dtype = _result_dtype(A_arr, B_arr)
    A_arr = np.ascontiguousarray(A_arr, dtype=dtype)
    B_arr = np.ascontiguousarray(B_arr, dtype=dtype)
    n, m = A_arr.shape
    p = B_arr.shape[1]
    C = np.zeros((n, p), dtype=dtype)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(0, p, block):
            j1 = min(j0 + block, p)
            C_tile = C[i0:i1, j0:j1]  # a view: updates land in C
            for k0 in range(0, m, block):
                k1 = min(k0 + block, m)
                A_tile = A_arr[i0:i1, k0:k1]
                B_tile = B_arr[k0:k1, j0:j1]
                for t in range(k1 - k0):
                    # Row r of C_tile += A_tile[r, t] * B_tile[t], all r at once.
                    C_tile += A_tile[:, t : t + 1] * B_tile[t]
    return C


def _strassen_square(A: np.ndarray, B: np.ndarray, cutoff: int) -> np.ndarray:
    """Strassen recursion on two s x s matrices with s a power of two.

    Args:
        A: An s x s array.
        B: An s x s array.
        cutoff: Sizes ``s <= cutoff`` go to :func:`matmul_blocked`.

    Returns:
        The s x s product.

    Time Complexity:
        O(s^log2(7)) = O(s^2.807) above the cutoff.

    Space Complexity:
        O(s^2) live at each of the log2(s / cutoff) levels, O(s^2) in total.

    Examples:
        >>> A = np.arange(16.0).reshape(4, 4)
        >>> bool(np.array_equal(_strassen_square(A, A, 1), A @ A))
        True
    """
    s = A.shape[0]
    if s <= cutoff or s == 1:
        return matmul_blocked(A, B)
    h = s // 2
    A11, A12, A21, A22 = A[:h, :h], A[:h, h:], A[h:, :h], A[h:, h:]
    B11, B12, B21, B22 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]
    M1 = _strassen_square(A11 + A22, B11 + B22, cutoff)
    M2 = _strassen_square(A21 + A22, B11, cutoff)
    M3 = _strassen_square(A11, B12 - B22, cutoff)
    M4 = _strassen_square(A22, B21 - B11, cutoff)
    M5 = _strassen_square(A11 + A12, B22, cutoff)
    M6 = _strassen_square(A21 - A11, B11 + B12, cutoff)
    M7 = _strassen_square(A12 - A22, B21 + B22, cutoff)
    C = np.empty((s, s), dtype=M1.dtype)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    return C


def strassen(A, B, cutoff: int = 64) -> np.ndarray:
    """Multiply two matrices with Strassen's seven-product recursion.

    Both operands are zero-padded into one s x s size, s the smallest power
    of two covering every dimension, so rectangular inputs work and every
    split is exact. Each level splits the operands into 2 x 2 blocks and
    forms the four result blocks from seven block products M1..M7 and
    eighteen block additions. A block of size ``s <= cutoff`` is multiplied
    with :func:`matmul_blocked`. The padding is cropped off at the end.

    Args:
        A: An n x m matrix (array-like).
        B: An m x p matrix (array-like).
        cutoff: Recursion stops at block size ``<= cutoff``; at least 1.

    Returns:
        The n x p product as a float64 (or complex128) ``np.ndarray``.

    Raises:
        ValueError: If either operand is not 2-D, the inner dimensions
            differ, or ``cutoff < 1``.

    Time Complexity:
        O(s^log2(7)) = O(s^2.807) with s = next power of two of
        max(n, m, p); the padding can cost up to a constant factor.

    Space Complexity:
        O(s^2) for the padded operands and the temporaries of one recursion
        path.

    Examples:
        >>> strassen([[1, 2], [3, 4]], [[5, 6], [7, 8]], cutoff=1).tolist()
        [[19.0, 22.0], [43.0, 50.0]]
        >>> rng = np.random.default_rng(42)
        >>> X, Y = rng.standard_normal((7, 5)), rng.standard_normal((5, 3))
        >>> strassen(X, Y, cutoff=2).shape
        (7, 3)
        >>> bool(np.allclose(strassen(X, Y, cutoff=2), X @ Y))
        True
        >>> strassen(np.ones((2, 3)), np.ones((2, 3)))
        Traceback (most recent call last):
        ...
        ValueError: shapes (2, 3) and (2, 3) are not conformable
    """
    A_arr = _as_matrix(A, "A")
    B_arr = _as_matrix(B, "B")
    _check_conformable(A_arr, B_arr)
    cutoff = int(cutoff)
    if cutoff < 1:
        raise ValueError(f"cutoff must be at least 1, got {cutoff}")
    dtype = _result_dtype(A_arr, B_arr)
    n, m = A_arr.shape
    p = B_arr.shape[1]
    if n == 0 or m == 0 or p == 0:
        return np.zeros((n, p), dtype=dtype)
    s = _next_power_of_two(max(n, m, p))
    A_pad = np.zeros((s, s), dtype=dtype)
    B_pad = np.zeros((s, s), dtype=dtype)
    A_pad[:n, :m] = A_arr
    B_pad[:m, :p] = B_arr
    C = _strassen_square(A_pad, B_pad, cutoff)
    return np.ascontiguousarray(C[:n, :p])


# ---------------------------------------------------------------------------
# LU factorization and linear solves
# ---------------------------------------------------------------------------


def _lu_core(A, pivot: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Gaussian elimination returning the row order, L, U and the swap count.

    Args:
        A: A square matrix (array-like).
        pivot: Whether to use partial pivoting.

    Returns:
        ``(perm, L, U, swaps)`` with ``A[perm] == L @ U`` up to rounding.

    Raises:
        ValueError: If ``A`` is not square or a pivot is exactly zero.

    Time Complexity:
        About 2n^3/3 flops.

    Space Complexity:
        Theta(n^2).

    Examples:
        >>> perm, L, U, swaps = _lu_core([[0, 1], [1, 0]], pivot=True)
        >>> perm.tolist(), swaps
        ([1, 0], 1)
    """
    U = np.array(_as_matrix(A, "A"), dtype=np.float64)
    n = _check_square(U)
    L = np.eye(n)
    perm = np.arange(n)
    swaps = 0
    for k in range(n):
        if pivot:
            r = k + int(np.argmax(np.abs(U[k:, k])))
            if r != k:
                U[[k, r], k:] = U[[r, k], k:]
                L[[k, r], :k] = L[[r, k], :k]
                perm[[k, r]] = perm[[r, k]]
                swaps += 1
        pivot_value = U[k, k]
        if pivot_value == 0.0:
            raise ValueError(f"matrix is singular: zero pivot in column {k}")
        if k + 1 < n:
            multipliers = U[k + 1 :, k] / pivot_value
            L[k + 1 :, k] = multipliers
            # Subtract multiplier * pivot row from every row below, at once.
            U[k + 1 :, k + 1 :] -= multipliers[:, None] * U[k, k + 1 :]
            U[k + 1 :, k] = 0.0
    return perm, L, U, swaps


def lu_decompose(A, pivot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a square matrix as PA = LU by Gaussian elimination.

    Column k is eliminated by subtracting multiples of the pivot row from
    every row below it; the multipliers become column k of L. With
    ``pivot=True`` (partial pivoting) the row with the largest magnitude
    entry in column k, on or below the diagonal, is swapped up first, so
    every multiplier satisfies ``|l_ik| <= 1``. ``pivot=False`` is kept only
    to demonstrate instability, for example on ``[[1e-20, 1], [1, 1]]``.

    Args:
        A: An n x n matrix (array-like).
        pivot: Use partial pivoting (default True).

    Returns:
        ``(P, L, U)`` as float64 arrays: P an n x n permutation matrix, L unit
        lower triangular, U upper triangular, with ``P @ A == L @ U`` up to
        rounding.

    Raises:
        ValueError: If ``A`` is not square or a pivot is exactly zero (the
            matrix is singular, or with ``pivot=False`` needs a row swap).

    Time Complexity:
        About 2n^3/3 flops: Theta(n^3).

    Space Complexity:
        Theta(n^2) for P, L and U.

    Examples:
        >>> P, L, U = lu_decompose([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
        >>> P.tolist()
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        >>> L.tolist()
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.5, 1.0, 1.0]]
        >>> lu_decompose([[1, 2], [2, 4]])
        Traceback (most recent call last):
        ...
        ValueError: matrix is singular: zero pivot in column 1
    """
    perm, L, U, _ = _lu_core(A, pivot)
    P = np.eye(len(perm))[perm]
    return P, L, U


def lu_solve(P, L, U, b) -> np.ndarray:
    """Solve ``(P^T L U) x = b`` by forward then back substitution.

    The permutation is applied to b first (``P b`` reorders its rows), then
    ``L y = P b`` is solved top to bottom and ``U x = y`` bottom to top. Each
    step is one element-wise multiply and sum over the already-known
    entries. L does not have to be unit triangular (its diagonal is divided
    out), so the same routine also solves transposed systems such as
    ``U^T L^T x = b`` with ``lu_solve(I, U.T, L.T, b)``.

    Args:
        P: n x n permutation matrix from :func:`lu_decompose`.
        L: n x n lower triangular matrix.
        U: n x n upper triangular matrix.
        b: Right-hand side, shape (n,) or (n, k) for k systems at once.

    Returns:
        ``x`` with the same shape as ``b``, float64.

    Raises:
        ValueError: If the shapes disagree or a diagonal entry of L or U is
            exactly zero.

    Time Complexity:
        Theta(n^2) per right-hand side.

    Space Complexity:
        Theta(n k) for the intermediate y and the result.

    Examples:
        >>> P, L, U = lu_decompose([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
        >>> lu_solve(P, L, U, [5, -2, 9]).tolist()
        [1.0, 1.0, 2.0]
        >>> lu_solve(P, L, U, [[5, 1], [-2, 0], [9, 2]]).tolist()
        [[1.0, 0.0], [1.0, 0.0], [2.0, 1.0]]
    """
    P_arr = _as_matrix(P, "P")
    L_arr = np.asarray(_as_matrix(L, "L"), dtype=np.float64)
    U_arr = np.asarray(_as_matrix(U, "U"), dtype=np.float64)
    n = _check_square(L_arr, "L")
    if P_arr.shape != (n, n) or U_arr.shape != (n, n):
        raise ValueError("P, L and U must all be n x n with the same n")
    b_arr = np.asarray(b, dtype=np.float64)
    if b_arr.ndim not in (1, 2) or b_arr.shape[0] != n:
        raise ValueError(f"b must have {n} rows, got shape {b_arr.shape}")
    vector = b_arr.ndim == 1
    rhs = b_arr.reshape(n, -1)
    # P b: row i of P has its single 1 in column order[i].
    order = np.argmax(P_arr, axis=1)
    rhs = rhs[order]
    y = np.empty_like(rhs)
    for i in range(n):
        d = L_arr[i, i]
        if d == 0.0:
            raise ValueError(f"L has a zero on its diagonal at {i}")
        y[i] = (rhs[i] - (L_arr[i, :i, None] * y[:i]).sum(axis=0)) / d
    x = np.empty_like(y)
    for i in range(n - 1, -1, -1):
        d = U_arr[i, i]
        if d == 0.0:
            raise ValueError(f"U has a zero on its diagonal at {i}")
        x[i] = (y[i] - (U_arr[i, i + 1 :, None] * x[i + 1 :]).sum(axis=0)) / d
    return x[:, 0] if vector else x


def solve(A, b, pivot: bool = True) -> np.ndarray:
    """Solve ``A x = b`` with LU factorization and two triangular solves.

    The matrix is factored once as PA = LU by :func:`lu_decompose`, and
    :func:`lu_solve` then permutes b, runs forward substitution with L and
    back substitution with U. With ``pivot=False`` a tiny pivot produces a
    huge multiplier, and on ``[[1e-20, 1], [1, 1]]`` the answer comes out as
    ``[0, 1]`` instead of ``[1, 1]``; the doctests below show both.

    Args:
        A: An n x n matrix (array-like).
        b: Right-hand side, shape (n,) or (n, k).
        pivot: Use partial pivoting (default True). ``False`` is only for the
            stability demonstration.

    Returns:
        The solution ``x``, the same shape as ``b``.

    Raises:
        ValueError: If ``A`` is not square, a pivot is exactly zero, or ``b``
            has the wrong number of rows.

    Time Complexity:
        About 2n^3/3 flops for the factorization plus Theta(n^2) per
        right-hand side.

    Space Complexity:
        Theta(n^2).

    Examples:
        >>> solve([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], [5, -2, 9]).tolist()
        [1.0, 1.0, 2.0]
        >>> solve([[1e-20, 1], [1, 1]], [1, 2], pivot=False).tolist()
        [0.0, 1.0]
        >>> solve([[1e-20, 1], [1, 1]], [1, 2]).tolist()
        [1.0, 1.0]
    """
    P, L, U = lu_decompose(A, pivot=pivot)
    return lu_solve(P, L, U, b)


def determinant(A) -> float:
    """Return det(A) as the permutation sign times the product of U's diagonal.

    From PA = LU, det(P) det(A) = det(L) det(U) = prod(diag(U)), and
    det(P) = (-1)^(number of row swaps). A matrix whose elimination meets an
    exactly zero pivot is singular, so its determinant is 0.0.

    Args:
        A: An n x n matrix (array-like).

    Returns:
        The determinant as a Python float (1.0 for a 0 x 0 matrix).

    Raises:
        ValueError: If ``A`` is not a square 2-D matrix.

    Time Complexity:
        About 2n^3/3 flops.

    Space Complexity:
        Theta(n^2).

    Examples:
        >>> determinant([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
        -16.0
        >>> determinant([[1, 2], [2, 4]])
        0.0
    """
    _check_square(_as_matrix(A, "A"))
    try:
        _, _, U, swaps = _lu_core(A, pivot=True)
    except ValueError:
        return 0.0
    sign = -1.0 if swaps % 2 else 1.0
    return float(sign * np.prod(np.diag(U)))


# ---------------------------------------------------------------------------
# QR factorizations
# ---------------------------------------------------------------------------


def _check_tall(A) -> np.ndarray:
    """Return ``A`` as float64, requiring m >= n >= 1.

    Args:
        A: An m x n matrix (array-like).

    Returns:
        A float64 copy of ``A``.

    Raises:
        ValueError: If ``A`` is not 2-D, has no columns, or m < n.

    Time Complexity:
        O(m n).

    Space Complexity:
        O(m n).

    Examples:
        >>> _check_tall([[1, 2], [3, 4], [5, 6]]).shape
        (3, 2)
        >>> _check_tall([[1, 2, 3]])
        Traceback (most recent call last):
        ...
        ValueError: QR needs m >= n >= 1, got shape (1, 3)
    """
    M = np.array(_as_matrix(A, "A"), dtype=np.float64)
    m, n = M.shape
    if n < 1 or m < n:
        raise ValueError(f"QR needs m >= n >= 1, got shape {M.shape}")
    return M


def qr_householder(A) -> Tuple[np.ndarray, np.ndarray]:
    """Thin QR factorization by Householder reflections.

    For column k, the reflector ``H = I - 2 v v^T`` (v a unit vector) maps
    ``x = R[k:, k]`` onto ``alpha e_1`` with ``alpha = -sign(x_0) ||x||``;
    choosing that sign makes ``v = x - alpha e_1`` free of cancellation.
    Applying the n reflectors to A leaves R; applying them in reverse order
    to the first n columns of the identity builds Q. Every reflector is
    orthogonal to working precision, so ``||Q^T Q - I||`` stays near machine
    epsilon however ill-conditioned A is.

    Args:
        A: An m x n matrix with m >= n (array-like).

    Returns:
        ``(Q, R)``: Q is m x n with orthonormal columns, R is n x n upper
        triangular (entries below the diagonal exactly zero), and
        ``Q @ R == A`` up to rounding. Diagonal entries of R may be negative.

    Raises:
        ValueError: If ``A`` is not 2-D or m < n.

    Time Complexity:
        2mn^2 - 2n^3/3 flops for R, about as much again for the thin Q.

    Space Complexity:
        Theta(m n) for Q, R and the stored reflectors.

    Examples:
        >>> Q, R = qr_householder([[3, 0], [4, 5]])
        >>> np.round(R, 12).tolist()
        [[-5.0, -4.0], [0.0, -3.0]]
        >>> bool(np.allclose(Q @ R, [[3, 0], [4, 5]]))
        True
    """
    R = _check_tall(A)
    m, n = R.shape
    reflectors: List[np.ndarray] = []
    for k in range(n):
        x = R[k:, k]
        norm_x = _vec_norm2(x)
        v = x.copy()
        if norm_x == 0.0:
            reflectors.append(np.zeros_like(v))
            continue
        alpha = -norm_x if x[0] >= 0 else norm_x
        v[0] -= alpha
        v /= _vec_norm2(v)
        reflectors.append(v)
        # R[k:, k:] -= 2 v (v^T R[k:, k:]), with v^T R as multiply-and-sum.
        vTR = (v[:, None] * R[k:, k:]).sum(axis=0)
        R[k:, k:] -= 2.0 * v[:, None] * vTR
        R[k + 1 :, k] = 0.0
        R[k, k] = alpha
    Q = np.zeros((m, n))
    Q[:n, :n] = np.eye(n)
    for k in range(n - 1, -1, -1):
        v = reflectors[k]
        vTQ = (v[:, None] * Q[k:, :]).sum(axis=0)
        Q[k:, :] -= 2.0 * v[:, None] * vTQ
    return Q, np.triu(R[:n, :n])


def qr_gram_schmidt(A, modified: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Thin QR factorization by classical or modified Gram-Schmidt.

    Classical Gram-Schmidt (``modified=False``) computes all the
    coefficients ``r_ij = q_i^T a_j`` from the *original* column ``a_j`` and
    subtracts them at once. Modified Gram-Schmidt (``modified=True``)
    normalizes column k and immediately removes its component from every
    later column, so each later projection is taken against an already
    updated vector. In exact arithmetic they agree; in floating point CGS
    loses orthogonality like cond(A)^2 and MGS like cond(A), which on the
    Hilbert matrix is the difference between ``||Q^T Q - I||_F`` of order 1
    and order 1e-4 (n = 10).

    Args:
        A: An m x n matrix with m >= n (array-like).
        modified: Use MGS instead of CGS.

    Returns:
        ``(Q, R)``: Q m x n, R n x n upper triangular with a positive
        diagonal, ``Q @ R == A`` up to rounding.

    Raises:
        ValueError: If ``A`` is not 2-D, m < n, or a column becomes exactly
            zero (A is rank deficient).

    Time Complexity:
        2mn^2 flops.

    Space Complexity:
        Theta(m n).

    Examples:
        >>> Q, R = qr_gram_schmidt([[3, 0], [4, 5]])
        >>> np.round(R, 12).tolist()
        [[5.0, 4.0], [0.0, 3.0]]
        >>> Q2, R2 = qr_gram_schmidt([[3, 0], [4, 5]], modified=True)
        >>> bool(np.allclose(Q2 @ R2, [[3, 0], [4, 5]]))
        True
    """
    A_arr = _check_tall(A)
    m, n = A_arr.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    if modified:
        V = A_arr.copy()
        for k in range(n):
            r_kk = _vec_norm2(V[:, k])
            if r_kk == 0.0:
                raise ValueError(f"column {k} is linearly dependent on earlier columns")
            R[k, k] = r_kk
            Q[:, k] = V[:, k] / r_kk
            if k + 1 < n:
                # Remove q_k from every later column, using the updated columns.
                coeffs = (Q[:, k, None] * V[:, k + 1 :]).sum(axis=0)
                R[k, k + 1 :] = coeffs
                V[:, k + 1 :] -= Q[:, k, None] * coeffs
    else:
        for j in range(n):
            a_j = A_arr[:, j]
            # All coefficients from the original column, then one subtraction.
            coeffs = (Q[:, :j] * a_j[:, None]).sum(axis=0)
            R[:j, j] = coeffs
            v = a_j - (Q[:, :j] * coeffs).sum(axis=1)
            r_jj = _vec_norm2(v)
            if r_jj == 0.0:
                raise ValueError(f"column {j} is linearly dependent on earlier columns")
            R[j, j] = r_jj
            Q[:, j] = v / r_jj
    return Q, R


# ---------------------------------------------------------------------------
# Power method
# ---------------------------------------------------------------------------


def power_method(
    A, tol: float = 1e-10, max_iter: int = 10_000, seed: int = 42
) -> Tuple[float, np.ndarray, int]:
    """Estimate the dominant eigenvalue and eigenvector by power iteration.

    Starting from a seeded random unit vector v, each step forms
    ``w = A v`` (an element-wise multiply and row sum), takes the Rayleigh
    quotient ``lambda = v^T w`` and normalizes ``v = w / ||w||``. The
    component along the dominant eigenvector grows by ``|lambda_1|`` per
    step and every other one by at most ``|lambda_2|``, so the error falls
    like ``|lambda_2 / lambda_1|^k``. Iteration stops when the residual
    ``||A v - lambda v||_2`` is at most ``tol * max(1, |lambda|)``. If the
    two largest eigenvalues have equal magnitude the iteration need not
    converge; it then stops after ``max_iter`` steps and returns the last
    estimate (``iterations == max_iter`` signals this).

    Args:
        A: An n x n matrix (array-like).
        tol: Relative residual tolerance.
        max_iter: Maximum number of matrix-vector products, at least 1.
        seed: Seed for ``numpy.random.default_rng`` that draws the start
            vector.

    Returns:
        ``(eigenvalue, eigenvector, iterations)``: a float, a unit-norm
        float64 vector whose largest-magnitude entry is positive, and the
        number of matrix-vector products performed.

    Raises:
        ValueError: If ``A`` is not square, is empty, or ``max_iter < 1``.

    Time Complexity:
        Theta(n^2) per iteration, Theta(k n^2) for k iterations.

    Space Complexity:
        Theta(n) beyond the matrix.

    Examples:
        >>> lam, v, its = power_method([[2, 1], [1, 3]])
        >>> f"{lam:.10f}"
        '3.6180339887'
        >>> np.round(v, 6).tolist()
        [0.525731, 0.850651]
        >>> power_method([[4, 0], [0, 1]], seed=7)[0]
        4.0
    """
    M = np.asarray(_as_matrix(A, "A"), dtype=np.float64)
    n = _check_square(M)
    if n == 0:
        raise ValueError("A must not be empty")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= _vec_norm2(v)
    lam = 0.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        w = (M * v).sum(axis=1)
        lam = float((v * w).sum())
        residual = _vec_norm2(w - lam * v)
        norm_w = _vec_norm2(w)
        if norm_w == 0.0:
            # v is in the null space: eigenvalue 0 with eigenvector v.
            break
        v = w / norm_w
        if residual <= tol * max(1.0, abs(lam)):
            break
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    return lam, v, iterations
