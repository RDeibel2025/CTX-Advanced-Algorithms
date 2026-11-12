"""Week 12 tests: errors, norms, condition numbers and cancellation (spec 9.2).

Every expected value is a spec 9.2 anchor or is worked by hand, never taken
from the code under test. numpy and scipy appear only as the independent
cross-check (``np.linalg.cond``, ``np.linalg.norm``, ``scipy.linalg.hilbert``,
``math.fsum``).

Hand derivations of the cancellation anchors:

* ``cos(1e-8) = 1 - 5e-17 + ...`` and half the gap below 1.0 is 5.55e-17, so
  it rounds to exactly 1.0 and ``(1 - cos x) / x^2`` is 0.0. The true value
  is ``1/2 - x^2/24 + ... = 0.5`` to double precision.
* ``x^2 - 1e8 x + 1``: ``sqrt(1e16 - 4) = 1e8 - 2e-8`` to within 1e-23, but floats
  near 1e8 are 2^-26 = 1.49e-8 apart, so it rounds to ``1e8 - 2^-26`` and the
  textbook small root is ``2^-27 = 7.450580596923828e-09``: relative error
  ``1 - 0.745058 = 0.2549``. Vieta gives ``c / x_big = 1 / 1e8 = 1e-8``.
* Above 2^53 the float spacing is 2, so ``1e16 + 1`` rounds to 1e16 and the
  naive difference is 0.0; ``1 / (1e8 + 1e8) = 5e-9``.
* A plain loop adding 0.1 one million times drifts by about 1.3e-6 (each of
  the 10^6 roundings is up to half an ulp of a total near 1e5, 7.3e-12).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg

from src.fft_analyzer.core.numerical import (
    abs_error,
    cancellation_examples,
    cond2_power,
    cond_estimate,
    hilbert,
    is_ill_conditioned,
    kahan_sum,
    machine_epsilon,
    norm,
    rel_error,
)
from src.utils.random_utils import derive_seed

MASTER_SEED = 42


def _rng(*labels) -> np.random.Generator:
    """Return a numpy generator seeded from the master seed and labels."""
    return np.random.default_rng(derive_seed(MASTER_SEED, "week12", "numerical", *labels))


@pytest.fixture(scope="module")
def cancellation():
    """Run the four cancellation cases once for the whole module."""
    return cancellation_examples()


# ---------------------------------------------------------------------------
# Machine epsilon
# ---------------------------------------------------------------------------


def test_machine_epsilon_equals_numpy_eps():
    eps = machine_epsilon()
    assert eps == np.finfo(float).eps
    assert eps == 2.220446049250313e-16
    assert eps == 2.0**-52


def test_machine_epsilon_is_the_gap_above_one():
    eps = machine_epsilon()
    assert 1.0 + eps != 1.0
    assert 1.0 + eps / 2.0 == 1.0


# ---------------------------------------------------------------------------
# 2-norm condition numbers of Hilbert matrices (spec 9.2 table)
# ---------------------------------------------------------------------------

HILBERT_COND2 = [
    # (n, spec reference cond2, required relative error)
    (4, 1.5514e4, 1e-3),
    (5, 4.7661e5, 1e-3),
    (8, 1.5258e10, 1e-2),
    (10, 1.6025e13, 0.05),
]


@pytest.mark.parametrize("n, reference, tol", HILBERT_COND2)
def test_cond2_power_hilbert_matches_numpy(n, reference, tol):
    H = hilbert(n)
    ours = cond2_power(H)
    numpy_cond = np.linalg.cond(H)
    assert abs(ours - numpy_cond) / numpy_cond < tol
    # The spec values are rounded to five significant figures (at most
    # 3.3e-5 relative), far inside every tolerance in the table.
    assert abs(ours - reference) / reference < tol


def test_cond2_power_diagonal_by_hand():
    # Singular values 3 and 0.5, so cond2 = 6; a permutation does not change it.
    assert cond2_power([[3.0, 0.0], [0.0, 0.5]]) == pytest.approx(6.0, rel=1e-12)
    assert cond2_power([[0.0, 3.0], [0.5, 0.0]]) == pytest.approx(6.0, rel=1e-12)
    assert cond2_power(np.eye(5)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("n", [1, 3, 6, 12])
def test_cond2_power_random_matches_numpy(n):
    A = _rng("cond2", n).standard_normal((n, n))
    assert cond2_power(A) == pytest.approx(np.linalg.cond(A), rel=1e-8)


def test_cond2_power_singular_is_inf():
    assert cond2_power([[1.0, 2.0], [2.0, 4.0]]) == math.inf
    assert cond2_power(np.zeros((3, 3))) == math.inf


@pytest.mark.parametrize("bad", [[[1.0, 2.0, 3.0]], [1.0, 2.0], np.zeros((0, 0))])
def test_cond2_power_rejects_non_square(bad):
    with pytest.raises(ValueError):
        cond2_power(bad)


# ---------------------------------------------------------------------------
# p-norm condition numbers through LU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_cond_estimate_hilbert_one_norm_matches_numpy(n):
    H = hilbert(n)
    assert cond_estimate(H, 1) == pytest.approx(np.linalg.cond(H, 1), rel=1e-6)


def test_cond_estimate_hilbert3_by_hand():
    # H(3)^-1 = [[9, -36, 30], [-36, 192, -180], [30, -180, 180]]: column sums
    # 75, 408, 390, so ||H^-1||_1 = 408; ||H||_1 = 1 + 1/2 + 1/3 = 11/6.
    assert cond_estimate(hilbert(3)) == pytest.approx(11 / 6 * 408, rel=1e-10)


@pytest.mark.parametrize("p", [np.inf, "fro"])
@pytest.mark.parametrize("n", [3, 5])
def test_cond_estimate_other_orders_match_numpy(n, p):
    H = hilbert(n)
    assert cond_estimate(H, p) == pytest.approx(np.linalg.cond(H, p), rel=1e-6)


def test_cond_estimate_random_matches_numpy():
    A = _rng("cond_estimate").standard_normal((8, 8))
    for p in (1, np.inf, "fro"):
        assert cond_estimate(A, p) == pytest.approx(np.linalg.cond(A, p), rel=1e-9)


def test_cond_estimate_singular_is_inf():
    assert cond_estimate([[1.0, 2.0], [2.0, 4.0]]) == math.inf


@pytest.mark.parametrize("p", [2, 3, "nuc"])
def test_cond_estimate_rejects_unsupported_order(p):
    with pytest.raises(ValueError):
        cond_estimate(hilbert(3), p)


def test_cond_estimate_rejects_non_square():
    with pytest.raises(ValueError):
        cond_estimate([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# ---------------------------------------------------------------------------
# is_ill_conditioned
# ---------------------------------------------------------------------------


def test_is_ill_conditioned_anchors():
    assert is_ill_conditioned(hilbert(10)) is True
    assert is_ill_conditioned(np.eye(5)) is False


def test_is_ill_conditioned_threshold():
    # cond2(H(4)) = 1.5514e4 sits between these two thresholds.
    assert is_ill_conditioned(hilbert(4), threshold=1e4) is True
    assert is_ill_conditioned(hilbert(4), threshold=1e5) is False
    assert is_ill_conditioned([[1.0, 1.0], [1.0, 1.0]]) is True


# ---------------------------------------------------------------------------
# Cancellation (spec 9.2 table)
# ---------------------------------------------------------------------------


def test_cancellation_keys(cancellation):
    assert set(cancellation) == {
        "one_minus_cos",
        "quadratic_small_root",
        "sqrt_difference",
        "sum_point_one",
    }
    for case in cancellation.values():
        assert set(case) == {"naive", "stable", "exact"}
        assert all(isinstance(v, float) for v in case.values())


def test_cancellation_one_minus_cos(cancellation):
    case = cancellation["one_minus_cos"]
    assert case["naive"] == 0.0
    assert case["stable"] == pytest.approx(0.5, rel=1e-12)
    assert case["exact"] == pytest.approx(0.5, rel=1e-15)


def test_cancellation_quadratic_small_root(cancellation):
    case = cancellation["quadratic_small_root"]
    naive_rel = abs(case["naive"] - 1e-8) / 1e-8
    assert naive_rel > 1e-3
    assert naive_rel == pytest.approx(0.2549, abs=1e-4)
    assert case["naive"] == 2.0**-27
    assert abs(case["stable"] - 1e-8) / 1e-8 < 1e-12
    assert case["exact"] == pytest.approx(1e-8, rel=1e-15)


def test_cancellation_sqrt_difference(cancellation):
    case = cancellation["sqrt_difference"]
    assert case["naive"] == 0.0
    assert case["stable"] == pytest.approx(5e-9, rel=1e-12)
    assert case["exact"] == pytest.approx(5e-9, rel=1e-12)


def test_cancellation_sum_point_one(cancellation):
    case = cancellation["sum_point_one"]
    assert abs(case["naive"] - 100_000.0) == pytest.approx(1.3e-6, rel=0.05)
    assert abs(case["stable"] - 100_000.0) < 1e-9
    assert case["exact"] == 100_000.0


def test_cancellation_stable_beats_naive(cancellation):
    for name, case in cancellation.items():
        naive_err = rel_error(case["naive"], case["exact"])
        stable_err = rel_error(case["stable"], case["exact"])
        assert stable_err < 1e-12, name
        assert naive_err > 1e3 * max(stable_err, 1e-16), name


# ---------------------------------------------------------------------------
# Kahan summation
# ---------------------------------------------------------------------------


def test_kahan_sum_recovers_tiny_terms():
    # A plain loop keeps 1.0 forever (1e-16 is below half an ulp of 1); the
    # exact sum is 1 + 1e-12.
    values = [1.0] + [1e-16] * 10_000
    plain = 0.0
    for v in values:
        plain += v
    assert plain == 1.0
    assert kahan_sum(values) == pytest.approx(1.0 + 1e-12, rel=1e-15)


def test_kahan_sum_matches_fsum_on_random_data():
    values = _rng("kahan").standard_normal(50_000) * 1e3
    assert kahan_sum(values) == pytest.approx(math.fsum(values), rel=1e-12, abs=1e-9)


def test_kahan_sum_empty():
    assert kahan_sum([]) == 0.0


# ---------------------------------------------------------------------------
# Norms against np.linalg.norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, np.inf])
@pytest.mark.parametrize("n", [1, 7, 100])
def test_vector_norm_matches_numpy(p, n):
    rng = _rng("vector_norm", n)
    real = rng.standard_normal(n)
    complex_ = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    for x in (real, complex_):
        assert norm(x, p) == pytest.approx(np.linalg.norm(x, p), rel=1e-12)


@pytest.mark.parametrize("p", [1, np.inf, "fro"])
@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (13, 4), (4, 13)])
def test_matrix_norm_matches_numpy(p, shape):
    rng = _rng("matrix_norm", *shape)
    real = rng.standard_normal(shape)
    complex_ = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    for A in (real, complex_):
        assert norm(A, p) == pytest.approx(np.linalg.norm(A, p), rel=1e-12)


def test_norm_by_hand():
    assert norm([3, -4], 1) == 7.0
    assert norm([3, -4], 2) == 5.0
    assert norm([3, -4], np.inf) == 4.0
    assert norm([[1, -2], [-3, 4]], 1) == 6.0
    assert norm([[1, -2], [-3, 4]], np.inf) == 7.0
    assert norm([[1, -2], [-3, 4]], "fro") == pytest.approx(math.sqrt(30), rel=1e-15)
    assert norm(np.zeros(4), 2) == 0.0


def test_norm_does_not_overflow():
    assert norm([3e200, 4e200], 2) == pytest.approx(5e200, rel=1e-15)
    assert norm([[3e200], [4e200]], "fro") == pytest.approx(5e200, rel=1e-15)


@pytest.mark.parametrize(
    "x, p",
    [
        ([1.0, 2.0], "fro"),
        ([1.0, 2.0], 3),
        ([[1.0, 2.0], [3.0, 4.0]], 2),
        ([[1.0, 2.0], [3.0, 4.0]], "nuc"),
        (np.zeros((2, 2, 2)), 1),
    ],
)
def test_norm_rejects_unsupported(x, p):
    with pytest.raises(ValueError):
        norm(x, p)


# ---------------------------------------------------------------------------
# abs_error and rel_error
# ---------------------------------------------------------------------------


def test_abs_error_basics():
    assert abs_error([1.0, 2.0, 3.0], [1.0, 2.5, 3.0]) == 0.5
    assert abs_error(2.0, 2.0) == 0.0
    assert abs_error(1.0, 3.0) == 2.0
    assert abs_error([[1, 2], [3, 4]], [[1, 2], [3, 1]]) == 3.0
    assert abs_error(1 + 1j, 1.0) == 1.0


def test_rel_error_basics():
    # max |x - ref| = 0.5, max |ref| = 2.5.
    assert rel_error([1.0, 2.0], [1.0, 2.5]) == pytest.approx(0.2, rel=1e-15)
    assert rel_error(1.01, 1.0) == pytest.approx(0.01, rel=1e-12)
    assert rel_error([-4.0, 1.0], [-4.0, 0.0]) == pytest.approx(0.25, rel=1e-15)
    assert rel_error([0.0], [0.0]) == 0.0
    assert rel_error([1.0], [0.0]) == math.inf


def test_errors_reject_shape_mismatch():
    with pytest.raises(ValueError):
        abs_error([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rel_error([[1.0, 2.0]], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Hilbert matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_hilbert_matches_scipy(n):
    np.testing.assert_array_equal(hilbert(n), scipy.linalg.hilbert(n))


def test_hilbert_rejects_bad_order():
    with pytest.raises(ValueError):
        hilbert(0)
    with pytest.raises(TypeError):
        hilbert(2.5)
