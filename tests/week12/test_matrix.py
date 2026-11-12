"""Week 12 tests: matrix multiplication, LU, QR and the power method (spec 9.3).

Every expected value is from the spec 9.3 anchors or worked by hand, never
from the code under test. numpy and scipy appear only as the independent
cross-check.

Hand trace of LU with partial pivoting on ``A = [[2, 1, 1], [4, -6, 0],
[-2, 7, 2]]``:

* Column 0: the largest magnitude is 4 (row 1), so rows 0 and 1 swap. The
  multipliers are 2/4 = 0.5 and -2/4 = -0.5; row 1 becomes
  ``[2, 1, 1] - 0.5 [4, -6, 0] = [0, 4, 1]`` and row 2 becomes
  ``[-2, 7, 2] + 0.5 [4, -6, 0] = [0, 4, 2]``.
* Column 1: the candidates are 4 and 4, the first one wins (no swap), the
  multiplier is 1 and row 2 becomes ``[0, 0, 1]``.

So P swaps rows 0 and 1, L = [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]],
U = [[4, -6, 0], [0, 4, 1], [0, 0, 1]] and det(A) = (-1)(4)(4)(1) = -16.
Every number above is exact in binary floating point.

Without pivoting, ``[[1e-20, 1], [1, 1]] x = [1, 2]`` gives the multiplier
1e20, U[1, 1] = 1 - 1e20 which rounds to -1e20, y = [1, 2 - 1e20] = [1, -1e20],
x_1 = 1 and x_0 = (1 - 1) / 1e-20 = 0: the wrong answer [0, 1].
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg

from src.fft_analyzer.core.matrix import (
    determinant,
    lu_decompose,
    lu_solve,
    matmul_blocked,
    matmul_classical,
    power_method,
    qr_gram_schmidt,
    qr_householder,
    solve,
    strassen,
)
from src.utils.random_utils import derive_seed

SEED = 42

A_ANCHOR = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]


def _rng(*labels):
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(SEED, "test_matrix", *labels))


def _hilbert(n):
    """H[i, j] = 1 / (i + j + 1), built here so the test does not use numerical.py."""
    return 1.0 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0)


def _orth_loss(Q):
    """||Q^T Q - I||_F."""
    return float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), "fro"))


QR_METHODS = {
    "cgs": lambda A: qr_gram_schmidt(A, modified=False),
    "mgs": lambda A: qr_gram_schmidt(A, modified=True),
    "householder": qr_householder,
}

MULTIPLIES = {
    "classical": matmul_classical,
    "blocked": matmul_blocked,
    "strassen": strassen,
    "strassen_cutoff_2": lambda A, B: strassen(A, B, cutoff=2),
}


# ---------------------------------------------------------------------------
# LU, determinant and solve on the anchor matrix
# ---------------------------------------------------------------------------


def test_lu_anchor_factors():
    P, L, U = lu_decompose(A_ANCHOR)
    np.testing.assert_array_equal(P, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(L, [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]])
    np.testing.assert_array_equal(U, [[4, -6, 0], [0, 4, 1], [0, 0, 1]])
    np.testing.assert_array_equal(P @ np.array(A_ANCHOR, dtype=float), L @ U)


def test_lu_anchor_matches_scipy():
    # scipy returns A = p l u, so our P (with PA = LU) is p transposed.
    p, l, u = scipy.linalg.lu(np.array(A_ANCHOR, dtype=float))
    P, L, U = lu_decompose(A_ANCHOR)
    np.testing.assert_allclose(P, p.T)
    np.testing.assert_allclose(L, l)
    np.testing.assert_allclose(U, u)


def test_determinant_anchor():
    assert determinant(A_ANCHOR) == -16.0


def test_solve_anchor():
    x = solve(A_ANCHOR, [5, -2, 9])
    np.testing.assert_allclose(x, [1, 1, 2], rtol=0, atol=1e-12)


def test_lu_solve_anchor_reuses_factors():
    P, L, U = lu_decompose(A_ANCHOR)
    np.testing.assert_allclose(lu_solve(P, L, U, [5, -2, 9]), [1, 1, 2], rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [1, 5, 20])
def test_lu_random_matches_numpy(n):
    rng = _rng("lu", n)
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    P, L, U = lu_decompose(A)
    np.testing.assert_allclose(P @ A, L @ U, rtol=0, atol=1e-12)
    assert np.all(np.triu(L, 1) == 0) and np.all(np.diag(L) == 1)
    assert np.all(np.tril(U, -1) == 0)
    assert np.all(np.abs(L) <= 1.0)  # partial pivoting bounds every multiplier
    np.testing.assert_allclose(solve(A, b), np.linalg.solve(A, b), rtol=1e-9, atol=1e-12)
    assert determinant(A) == pytest.approx(np.linalg.det(A), rel=1e-9)


# ---------------------------------------------------------------------------
# The stability example: pivoting
# ---------------------------------------------------------------------------


def test_tiny_pivot_without_pivoting_is_wrong():
    x = solve([[1e-20, 1], [1, 1]], [1, 2], pivot=False)
    np.testing.assert_array_equal(x, [0.0, 1.0])


def test_tiny_pivot_with_pivoting_is_right():
    x = solve([[1e-20, 1], [1, 1]], [1, 2], pivot=True)
    np.testing.assert_allclose(x, [1.0, 1.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "singular",
    [
        [[1, 2], [2, 4]],
        [[1, 0, 2], [3, 0, 1], [5, 0, 7]],
        [[0, 0], [0, 0]],
    ],
)
def test_singular_lu_raises(singular):
    with pytest.raises(ValueError):
        lu_decompose(singular)


# ---------------------------------------------------------------------------
# QR: orthogonality loss on Hilbert matrices, reconstruction, triangularity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [10, 12])
def test_orthogonality_loss_ordering_on_hilbert(n):
    H = _hilbert(n)
    cgs = _orth_loss(qr_gram_schmidt(H, modified=False)[0])
    mgs = _orth_loss(qr_gram_schmidt(H, modified=True)[0])
    householder = _orth_loss(qr_householder(H)[0])
    assert cgs > mgs > householder
    assert householder < 1e-12
    assert cgs > 1e-2


def _qr_inputs():
    rng = _rng("qr")
    return {
        "hilbert10": _hilbert(10),
        "hilbert12": _hilbert(12),
        "random8x8": rng.standard_normal((8, 8)),
        "random20x5": rng.standard_normal((20, 5)),
        "anchor": np.array(A_ANCHOR, dtype=float),
        "single_column": rng.standard_normal((6, 1)),
    }


@pytest.mark.parametrize("method", sorted(QR_METHODS))
@pytest.mark.parametrize("name", sorted(_qr_inputs()))
def test_qr_reconstructs_and_r_is_upper_triangular(method, name):
    A = _qr_inputs()[name]
    Q, R = QR_METHODS[method](A)
    m, n = A.shape
    assert Q.shape == (m, n) and R.shape == (n, n)
    rel = np.linalg.norm(Q @ R - A) / np.linalg.norm(A)
    assert rel < 1e-10
    assert np.all(np.tril(R, -1) == 0)


@pytest.mark.parametrize("method", sorted(QR_METHODS))
def test_qr_matches_numpy_up_to_column_signs(method):
    A = _rng("qr-numpy").standard_normal((15, 6))
    Q, R = QR_METHODS[method](A)
    Q_ref, R_ref = np.linalg.qr(A)
    signs = np.sign(np.diag(R)) * np.sign(np.diag(R_ref))
    np.testing.assert_allclose(Q * signs, Q_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(R * signs[:, None], R_ref, rtol=0, atol=1e-10)
    assert _orth_loss(Q) < 1e-12


def test_qr_rejects_wide_matrix():
    for method in QR_METHODS.values():
        with pytest.raises(ValueError):
            method(np.ones((2, 3)))


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

MATMUL_SHAPES = [
    ((1, 1), (1, 1)),
    ((7, 7), (7, 7)),
    ((64, 64), (64, 64)),
    ((130, 70), (70, 90)),
]


@pytest.mark.parametrize("method", sorted(MULTIPLIES))
@pytest.mark.parametrize("shape_a, shape_b", MATMUL_SHAPES)
def test_multiplies_match_numpy(method, shape_a, shape_b):
    rng = _rng("matmul", shape_a, shape_b)
    A = rng.standard_normal(shape_a)
    B = rng.standard_normal(shape_b)
    C = MULTIPLIES[method](A, B)
    expected = A @ B
    assert isinstance(C, np.ndarray)
    assert C.shape == expected.shape
    np.testing.assert_allclose(C, expected, rtol=0, atol=1e-9)


def test_blocked_with_uneven_tiles():
    rng = _rng("blocked-uneven")
    A, B = rng.standard_normal((13, 11)), rng.standard_normal((11, 9))
    np.testing.assert_allclose(matmul_blocked(A, B, block=4), A @ B, rtol=0, atol=1e-9)


@pytest.mark.parametrize("method", [matmul_classical, matmul_blocked, strassen])
@pytest.mark.parametrize(
    "shape_a, shape_b", [((3, 4), (3, 4)), ((2, 5), (4, 2)), ((3,), (3, 3))]
)
def test_multiplies_reject_non_conformable(method, shape_a, shape_b):
    with pytest.raises(ValueError):
        method(np.ones(shape_a), np.ones(shape_b))


# ---------------------------------------------------------------------------
# Power method
# ---------------------------------------------------------------------------


def test_power_method_anchor():
    lam, v, iterations = power_method([[2, 1], [1, 3]])
    assert abs(lam - (5 + math.sqrt(5)) / 2) < 1e-8
    assert abs(lam - 3.618033988749895) < 1e-8
    assert 1 <= iterations < 10_000
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(A @ v, lam * v, rtol=0, atol=1e-8)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)


def test_power_method_matches_numpy_eig():
    rng = _rng("power")
    X = rng.standard_normal((6, 6))
    A = X @ X.T + np.diag(np.arange(6.0) * 3)  # symmetric positive definite
    lam, v, _ = power_method(A, tol=1e-12)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    top = int(np.argmax(np.abs(eigenvalues)))
    assert lam == pytest.approx(float(eigenvalues[top].real), rel=1e-8)
    ref = eigenvectors[:, top].real
    assert abs(abs(float(v @ ref)) - 1.0) < 1e-8


def test_power_method_is_reproducible():
    A = _rng("power-repeat").standard_normal((5, 5))
    A = A + A.T
    first = power_method(A, seed=7)
    second = power_method(A, seed=7)
    assert first[0] == second[0] and first[2] == second[2]
    np.testing.assert_array_equal(first[1], second[1])
