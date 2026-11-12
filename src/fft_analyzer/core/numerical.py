"""Numerical stability: errors, norms, condition numbers and cancellation.

Measuring error
---------------
:func:`abs_error` and :func:`rel_error` measure in the max norm: the largest
entry-wise distance ``max |x - x_ref|``, divided by ``max |x_ref|`` for the
relative error. One number per comparison, whatever the shape, so a test can
say "the FFT agrees with the reference to 1e-10" in one assertion.

Norms by hand
-------------
:func:`norm` writes out the norms the condition numbers need. For vectors:
the 1-norm ``sum |x_i|``, the 2-norm ``sqrt(sum |x_i|^2)`` and the inf-norm
``max |x_i|``. For matrices: the 1-norm (largest column sum of magnitudes),
the inf-norm (largest row sum) and the Frobenius norm (the 2-norm of all the
entries). The square roots are taken after dividing by the largest
magnitude, so squaring 1e200 cannot overflow.

Condition numbers
-----------------
The condition number ``kappa(A) = ||A|| ||A^-1||`` bounds how much a relative
change in b (or in A) can be amplified in the solution of ``A x = b``: about
``log10(kappa)`` decimal digits are lost. The Hilbert matrix
``H[i, j] = 1/(i + j + 1)`` is the classic bad case, with kappa growing like
``e^(3.5 n)``: 1.6e13 at n = 10, so of the 16 digits double precision
carries only about 3 survive.

* :func:`cond_estimate` builds ``A^-1`` explicitly, one column at a time, by
  solving ``A x_j = e_j`` with the LU factors of A (factored once), and
  multiplies the two p-norms (1, inf or Frobenius: cheap row and column
  sums).
* :func:`cond2_power` gives the 2-norm condition ``sigma_max / sigma_min``
  without an SVD. ``sigma_max^2`` is the largest eigenvalue of ``A^T A``,
  found by power iteration that applies ``A^T A`` as ``A^T (A v)``.
  ``1 / sigma_min^2`` is the largest eigenvalue of
  ``(A A^T)^-1 = A^-T A^-1``, found by power iteration that applies it as two
  LU solves, ``A y = v`` and then ``A^T u = y``. ``A^T A`` is never formed:
  forming it squares the condition number (2.6e26 for H(10), far past
  1/eps), so its smallest eigenvalue would be rounding noise.

Both follow the round trip that runs through the whole week: move the problem
where it is easy (A becomes triangular factors, or a symmetric positive
definite operator whose top eigenvalue power iteration finds), do the easy
thing (triangular solves, repeated multiplication), and move back (undo the
row permutation, take a square root).

Machine epsilon and cancellation
--------------------------------
:func:`machine_epsilon` halves a candidate until ``1 + eps/2`` rounds to 1,
which leaves ``2^-52``. Each floating-point operation has relative error at
most eps/2, but subtracting two nearly equal numbers keeps their absolute
error while cancelling their leading digits, so the relative error of the
difference can be enormous. :func:`cancellation_examples` runs four such
cases next to an algebraically equivalent rewrite with no harmful
subtraction; each rewrite is the same round trip on a formula (move it to an
equivalent form, evaluate that, and the result is the same quantity).
:func:`kahan_sum` carries the low-order bits that each addition drops in a
separate compensation term.

Rule for this package: numpy arrays and element-wise operations are used
freely, but no norm, solver, inverse, condition-number or eigenvalue routine
from numpy's or scipy's linear-algebra modules is called; the tests use those
only as the independent cross-check.

Examples:
    >>> machine_epsilon() == 2.0 ** -52
    True
    >>> norm([3, -4], 2), norm([3, -4], 1), norm([3, -4], float("inf"))
    (5.0, 7.0, 4.0)
    >>> f"{cond2_power(hilbert(4)):.4e}"
    '1.5514e+04'
    >>> cancellation_examples()["sqrt_difference"]["naive"]
    0.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import decimal
import math
import operator
from fractions import Fraction
from typing import Callable, Dict, Iterable, Tuple, Union

import numpy as np

from src.fft_analyzer.core.matrix import lu_decompose, lu_solve
from src.utils.random_utils import DEFAULT_SEED, derive_seed

__all__ = [
    "abs_error",
    "rel_error",
    "norm",
    "cond_estimate",
    "cond2_power",
    "is_ill_conditioned",
    "machine_epsilon",
    "cancellation_examples",
    "kahan_sum",
    "hilbert",
]

#: Power iteration in :func:`cond2_power` stops once the eigenvalue estimate
#: changes by at most this much, relative to itself, in one step.
_POWER_TOL: float = 1e-12

#: Upper bound on power-iteration steps in :func:`cond2_power`.
_POWER_MAX_ITER: int = 10_000

#: Seed for the start vector of the power iterations in :func:`cond2_power`.
_POWER_SEED: int = derive_seed(DEFAULT_SEED, "week12", "cond2_power")

Order = Union[int, float, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_numeric_array(x) -> np.ndarray:
    """Convert ``x`` to a float64 array, or keep it complex if it is complex.

    Args:
        x: A scalar or array-like of numbers.

    Returns:
        A complex array if ``x`` holds complex values, otherwise float64.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size) when a copy is needed.

    Examples:
        >>> _as_numeric_array([1, 2]).dtype
        dtype('float64')
        >>> _as_numeric_array([1j]).dtype
        dtype('complex128')
    """
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128)
    return arr.astype(np.float64)


def _parse_order(p: Order) -> str:
    """Normalize a norm order to one of ``"1"``, ``"2"``, ``"inf"``, ``"fro"``.

    Args:
        p: 1, 2, ``float("inf")`` (or ``np.inf``) or the string ``"fro"``.

    Returns:
        The canonical name of the order.

    Raises:
        ValueError: If ``p`` is not one of the supported orders.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _parse_order(1), _parse_order(2.0), _parse_order(np.inf), _parse_order("fro")
        ('1', '2', 'inf', 'fro')
        >>> _parse_order(3)
        Traceback (most recent call last):
        ...
        ValueError: unsupported norm order 3
    """
    if isinstance(p, str):
        if p == "fro":
            return "fro"
        raise ValueError(f"unsupported norm order {p!r}")
    try:
        value = float(p)
    except (TypeError, ValueError):
        raise ValueError(f"unsupported norm order {p!r}") from None
    if value == 1.0:
        return "1"
    if value == 2.0:
        return "2"
    if value == math.inf:
        return "inf"
    raise ValueError(f"unsupported norm order {p!r}")


def _scaled_root_sum_squares(magnitudes: np.ndarray) -> float:
    """Return ``sqrt(sum(m^2))`` for non-negative ``m`` without overflow.

    Dividing by the largest magnitude s first keeps every square at most 1,
    and the result is ``s * sqrt(sum((m / s)^2))``.

    Args:
        magnitudes: Array of non-negative values (any shape).

    Returns:
        The Euclidean length of the flattened array as a float.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size) for the scaled copy.

    Examples:
        >>> f"{_scaled_root_sum_squares(np.array([3e200, 4e200])):.6e}"
        '5.000000e+200'
        >>> _scaled_root_sum_squares(np.zeros(3))
        0.0
    """
    if magnitudes.size == 0:
        return 0.0
    s = float(magnitudes.max())
    if s == 0.0 or not math.isfinite(s):
        return s
    scaled = magnitudes / s
    return s * math.sqrt(float((scaled * scaled).sum()))


def _square_float_matrix(A) -> np.ndarray:
    """Return ``A`` as a non-empty square float64 matrix.

    Args:
        A: An n x n matrix (array-like), n >= 1.

    Returns:
        A float64 copy of ``A``.

    Raises:
        ValueError: If ``A`` is not 2-D, not square or empty.

    Time Complexity:
        O(n^2).

    Space Complexity:
        O(n^2).

    Examples:
        >>> _square_float_matrix([[1, 2], [3, 4]]).dtype
        dtype('float64')
        >>> _square_float_matrix([[1, 2, 3]])
        Traceback (most recent call last):
        ...
        ValueError: A must be a non-empty square matrix, got shape (1, 3)
    """
    M = np.array(A, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
        raise ValueError(f"A must be a non-empty square matrix, got shape {M.shape}")
    return M


def _largest_eigenvalue_psd(
    apply: Callable[[np.ndarray], Tuple[np.ndarray, float]], n: int
) -> float:
    """Power iteration for the largest eigenvalue of a symmetric PSD operator.

    ``apply(v)`` must return ``(M v, v^T M v)`` for a symmetric positive
    semidefinite M that is never formed. Starting from a seeded random unit
    vector, each step takes the Rayleigh quotient of the current unit vector
    and moves to ``M v / ||M v||``. For a symmetric M the Rayleigh quotient
    converges like ``(lambda_2 / lambda_1)^(2k)``, and because M is positive
    semidefinite a tie for the top eigenvalue does no harm: every vector in
    the top eigenspace has the same Rayleigh quotient.

    Args:
        apply: The operator, returning the product and the Rayleigh quotient.
        n: The dimension.

    Returns:
        The largest eigenvalue estimate (0.0 if M v is exactly zero).

    Time Complexity:
        O(k * cost(apply)) for k iterations, k at most 10,000.

    Space Complexity:
        O(n) beyond what ``apply`` uses.

    Examples:
        >>> D = np.array([4.0, 1.0])
        >>> f"{_largest_eigenvalue_psd(lambda v: (D * v, float((v * D * v).sum())), 2):.12f}"
        '4.000000000000'
    """
    rng = np.random.default_rng(_POWER_SEED)
    v = rng.standard_normal(n)
    v /= _scaled_root_sum_squares(np.abs(v))
    lam_old = 0.0
    lam = 0.0
    for _ in range(_POWER_MAX_ITER):
        w, lam = apply(v)
        w_norm = _scaled_root_sum_squares(np.abs(w))
        if w_norm == 0.0:
            return 0.0
        v = w / w_norm
        if abs(lam - lam_old) <= _POWER_TOL * lam:
            break
        lam_old = lam
    return lam


# ---------------------------------------------------------------------------
# Errors and norms
# ---------------------------------------------------------------------------


def abs_error(x, x_ref) -> float:
    """Return the absolute error ``max |x - x_ref|`` in the max norm.

    Scalars, vectors and matrices are all compared entry by entry, and
    complex values by the modulus of the difference.

    Args:
        x: The computed value (scalar or array-like).
        x_ref: The reference value, the same shape as ``x``.

    Returns:
        The largest entry-wise distance as a float (0.0 for empty inputs).

    Raises:
        ValueError: If the shapes differ.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size) for the difference.

    Examples:
        >>> abs_error([1.0, 2.0, 3.0], [1.0, 2.5, 3.0])
        0.5
        >>> abs_error(1 + 1j, 1)
        1.0
        >>> abs_error([1, 2], [1, 2, 3])
        Traceback (most recent call last):
        ...
        ValueError: shapes differ: (2,) and (3,)
    """
    a = _as_numeric_array(x)
    r = _as_numeric_array(x_ref)
    if a.shape != r.shape:
        raise ValueError(f"shapes differ: {a.shape} and {r.shape}")
    if a.size == 0:
        return 0.0
    return float(np.abs(a - r).max())


def rel_error(x, x_ref) -> float:
    """Return the relative error ``max |x - x_ref| / max |x_ref|``.

    Both maxima are over all entries (the max norm), so a single tiny
    reference entry cannot blow the ratio up the way an entry-wise relative
    error would. If the reference is entirely zero the ratio is undefined;
    the result is then 0.0 when ``x`` is also zero and ``inf`` otherwise.

    Args:
        x: The computed value (scalar or array-like).
        x_ref: The reference value, the same shape as ``x``.

    Returns:
        The relative error as a float.

    Raises:
        ValueError: If the shapes differ.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size) for the difference.

    Examples:
        >>> rel_error([1.0, 2.0], [1.0, 2.5])
        0.2
        >>> rel_error(7.450580596923828e-09, 1e-08) > 0.25
        True
        >>> rel_error([0.0], [0.0]), rel_error([1.0], [0.0])
        (0.0, inf)
    """
    err = abs_error(x, x_ref)
    r = _as_numeric_array(x_ref)
    scale = float(np.abs(r).max()) if r.size else 0.0
    if scale == 0.0:
        return 0.0 if err == 0.0 else math.inf
    return err / scale


def norm(x, p: Order) -> float:
    """Return the p-norm of a vector or a matrix, computed by hand.

    Vectors (1-D): ``p = 1`` is ``sum |x_i|``, ``p = 2`` is
    ``sqrt(sum |x_i|^2)`` and ``p = inf`` is ``max |x_i|``. Matrices (2-D):
    ``p = 1`` is the largest column sum ``max_j sum_i |a_ij|``, ``p = inf``
    the largest row sum ``max_i sum_j |a_ij|`` and ``p = "fro"`` the
    Frobenius norm ``sqrt(sum |a_ij|^2)``. The matrix 2-norm needs the
    largest singular value, which :func:`cond2_power` finds by power
    iteration, so it is not offered here. Square roots are taken after
    scaling by the largest magnitude, so the result does not overflow when
    the squares would.

    Args:
        x: A 1-D vector or 2-D matrix (array-like, real or complex).
        p: 1, 2 or ``inf`` for vectors; 1, ``inf`` or ``"fro"`` for matrices.

    Returns:
        The norm as a float (0.0 for an empty input).

    Raises:
        ValueError: If ``x`` is not 1-D or 2-D, or ``p`` is not supported for
            that dimension.

    Time Complexity:
        O(size).

    Space Complexity:
        O(size) for the magnitudes.

    Examples:
        >>> norm([3, -4], 1), norm([3, -4], 2), norm([3, -4], np.inf)
        (7.0, 5.0, 4.0)
        >>> A = [[1, -2], [-3, 4]]
        >>> norm(A, 1), norm(A, np.inf), f"{norm(A, 'fro'):.6f}"
        (6.0, 7.0, '5.477226')
        >>> norm(A, 2)
        Traceback (most recent call last):
        ...
        ValueError: matrix norm order must be 1, inf or 'fro', got 2
    """
    arr = _as_numeric_array(x)
    order = _parse_order(p)
    mags = np.abs(arr)
    if arr.ndim == 1:
        if order == "fro":
            raise ValueError("vector norm order must be 1, 2 or inf, got 'fro'")
        if mags.size == 0:
            return 0.0
        if order == "1":
            return float(mags.sum())
        if order == "inf":
            return float(mags.max())
        return _scaled_root_sum_squares(mags)
    if arr.ndim == 2:
        if order == "2":
            raise ValueError(f"matrix norm order must be 1, inf or 'fro', got {p!r}")
        if mags.size == 0:
            return 0.0
        if order == "1":
            return float(mags.sum(axis=0).max())
        if order == "inf":
            return float(mags.sum(axis=1).max())
        return _scaled_root_sum_squares(mags)
    raise ValueError(f"x must be a 1-D vector or a 2-D matrix, got shape {arr.shape}")


# ---------------------------------------------------------------------------
# Condition numbers
# ---------------------------------------------------------------------------


def cond_estimate(A, p: Order = 1) -> float:
    """Return ``kappa_p(A) = ||A||_p ||A^-1||_p`` with A^-1 built from LU.

    A is factored once as PA = LU (about 2n^3/3 flops). Column j of A^-1 is
    the solution of ``A x = e_j``, one forward and one back substitution
    (O(n^2) each), so the whole inverse costs another O(n^3). Both norms are
    then row or column sums of magnitudes (or the Frobenius norm). A matrix
    whose elimination meets an exactly zero pivot is singular and its
    condition number is ``inf``. For the 2-norm use :func:`cond2_power`.

    Args:
        A: An n x n matrix (array-like), n >= 1.
        p: 1 (default), ``inf`` or ``"fro"``.

    Returns:
        The condition number as a float, ``inf`` for an exactly singular A.

    Raises:
        ValueError: If ``A`` is not a non-empty square matrix or ``p`` is not
            1, ``inf`` or ``"fro"``.

    Time Complexity:
        Theta(n^3): the factorization plus n triangular-solve pairs.

    Space Complexity:
        Theta(n^2) for the factors and the inverse.

    Examples:
        >>> cond_estimate([[2, 0], [0, 0.5]])
        4.0
        >>> f"{cond_estimate(hilbert(3), 1):.1f}"
        '748.0'
        >>> cond_estimate([[1, 2], [2, 4]])
        inf
    """
    M = _square_float_matrix(A)
    order = _parse_order(p)
    if order == "2":
        raise ValueError("cond_estimate takes p = 1, inf or 'fro'; use cond2_power for p = 2")
    n = M.shape[0]
    try:
        P, L, U = lu_decompose(M, pivot=True)
    except ValueError:
        return math.inf
    inverse = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        inverse[:, j] = lu_solve(P, L, U, e_j)
    return norm(M, p) * norm(inverse, p)


def cond2_power(A) -> float:
    """Return the 2-norm condition ``sigma_max / sigma_min`` by power iteration.

    ``sigma_max^2`` is the largest eigenvalue of ``A^T A``. Power iteration
    finds it by applying ``A^T A`` to v as ``A^T (A v)``, two matrix-vector
    products, with Rayleigh quotient ``v^T A^T A v = ||A v||^2``.
    ``1 / sigma_min^2`` is the largest eigenvalue of
    ``(A A^T)^-1 = A^-T A^-1``. With PA = LU, applying it to v is two
    triangular-solve passes with the same factors: ``y = A^-1 v`` by
    ``lu_solve(P, L, U, v)``, then ``u = A^-T y`` by solving
    ``U^T L^T z = y`` (``lu_solve`` with the transposed factors) and undoing
    the permutation, ``u = P^T z``. Its Rayleigh quotient is ``||y||^2``.
    ``A^T A`` is never formed or inverted: forming it squares the condition
    number, which for H(10) is 2.6e26, and its smallest eigenvalue would be
    lost in rounding. Each power iteration starts from the same seeded
    random vector and stops when the estimate changes by at most 1e-12
    relative, or after 10,000 steps.

    Args:
        A: An n x n matrix (array-like), n >= 1.

    Returns:
        The 2-norm condition number as a float, ``inf`` if the LU
        factorization meets an exactly zero pivot.

    Raises:
        ValueError: If ``A`` is not a non-empty square matrix.

    Time Complexity:
        Theta(n^3) for the factorization plus Theta(n^2) per iteration;
        the iteration count depends on the gaps ``sigma_2 / sigma_1`` and
        ``sigma_min / sigma_(n-1)``.

    Space Complexity:
        Theta(n^2) for the factors.

    Examples:
        >>> cond2_power([[3, 0], [0, 0.5]])
        6.0
        >>> f"{cond2_power(hilbert(5)):.4e}"
        '4.7661e+05'
        >>> cond2_power([[1, 2], [2, 4]])
        inf
    """
    M = _square_float_matrix(A)
    n = M.shape[0]
    try:
        P, L, U = lu_decompose(M, pivot=True)
    except ValueError:
        return math.inf
    identity = np.eye(n)
    Lt = L.T.copy()
    Ut = U.T.copy()

    def apply_gram(v: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return ``(A^T (A v), ||A v||^2)``."""
        y = (M * v).sum(axis=1)
        return (M * y[:, None]).sum(axis=0), float((y * y).sum())

    def apply_inverse_gram(v: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return ``(A^-T (A^-1 v), ||A^-1 v||^2)`` with two LU solves."""
        y = lu_solve(P, L, U, v)
        z = lu_solve(identity, Ut, Lt, y)
        u = (P * z[:, None]).sum(axis=0)
        return u, float((y * y).sum())

    sigma_max_sq = _largest_eigenvalue_psd(apply_gram, n)
    inv_sigma_min_sq = _largest_eigenvalue_psd(apply_inverse_gram, n)
    return math.sqrt(sigma_max_sq) * math.sqrt(inv_sigma_min_sq)


def is_ill_conditioned(A, threshold: float = 1e10) -> bool:
    """Return whether the 2-norm condition number of A exceeds ``threshold``.

    The condition number comes from :func:`cond2_power`. With the default
    threshold 1e10, solving ``A x = b`` in double precision may lose more
    than ten of its roughly sixteen significant digits. An exactly singular
    matrix (condition ``inf``) counts as ill-conditioned.

    Args:
        A: An n x n matrix (array-like), n >= 1.
        threshold: The condition number above which A is ill-conditioned.

    Returns:
        ``True`` if ``cond2_power(A) > threshold``.

    Raises:
        ValueError: If ``A`` is not a non-empty square matrix.

    Time Complexity:
        That of :func:`cond2_power`, Theta(n^3) plus the iterations.

    Space Complexity:
        Theta(n^2).

    Examples:
        >>> is_ill_conditioned(hilbert(10)), is_ill_conditioned(np.eye(5))
        (True, False)
        >>> is_ill_conditioned(hilbert(4), threshold=1e4)
        True
    """
    return cond2_power(A) > threshold


# ---------------------------------------------------------------------------
# Machine epsilon, cancellation and compensated summation
# ---------------------------------------------------------------------------


def machine_epsilon() -> float:
    """Return the gap between 1.0 and the next float, found by halving.

    Starting from 1.0, the candidate is halved while ``1 + eps/2`` is still
    distinguishable from 1. In IEEE double precision the significand has 52
    stored bits, so the loop stops at ``2^-52``; ``1 + 2^-53`` lies exactly
    halfway and rounds to the even neighbour, 1.

    Args:
        None.

    Returns:
        ``2^-52`` = 2.220446049250313e-16, equal to ``np.finfo(float).eps``.

    Raises:
        Nothing.

    Time Complexity:
        O(t) for a t-bit significand: 52 halvings.

    Space Complexity:
        O(1).

    Examples:
        >>> machine_epsilon()
        2.220446049250313e-16
        >>> 1.0 + machine_epsilon() / 2 == 1.0
        True
    """
    eps = 1.0
    while 1.0 + eps / 2.0 != 1.0:
        eps /= 2.0
    return eps


def kahan_sum(values: Iterable[float]) -> float:
    """Sum floats with Kahan's compensated summation.

    A running compensation c holds the low-order part that the last addition
    rounded away. Each new value is corrected by c before it is added, and
    ``c = (t - s) - y`` recovers what the addition ``t = s + y`` lost (exact
    in IEEE arithmetic while ``|s| >= |y|``). The error bound is about
    ``2 eps sum |x_i|``, independent of n, against ``(n - 1) eps sum |x_i|``
    for a plain loop. (Python 3.12's built-in ``sum`` already compensates
    floats, which is why the naive baselines here are explicit loops.)

    Args:
        values: An iterable of real numbers.

    Returns:
        The compensated sum as a float (0.0 for an empty iterable).

    Raises:
        TypeError: If an item is not a real number (for example complex).
        ValueError: If an item is a string that does not parse as a float.

    Time Complexity:
        O(n): four floating-point operations per value.

    Space Complexity:
        O(1) beyond the iterable.

    Examples:
        >>> kahan_sum([0.1] * 10)
        1.0
        >>> kahan_sum([1.0] + [1e-16] * 10_000) > 1.0
        True
        >>> kahan_sum([])
        0.0
    """
    s = 0.0
    c = 0.0
    for value in values:
        y = float(value) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def _plain_loop_sum(values: Iterable[float]) -> float:
    """Sum floats left to right with one rounding per addition, no correction.

    Args:
        values: An iterable of real numbers.

    Returns:
        The rounded running total.

    Time Complexity:
        O(n).

    Space Complexity:
        O(1).

    Examples:
        >>> _plain_loop_sum([0.1] * 10)
        0.9999999999999999
    """
    total = 0.0
    for value in values:
        total += float(value)
    return total


def cancellation_examples() -> Dict[str, Dict[str, float]]:
    """Run four catastrophic-cancellation cases, each naive and stable.

    * ``"one_minus_cos"``: ``(1 - cos x) / x^2`` at x = 1e-8. ``cos(1e-8)``
      rounds to exactly 1, so the naive value is 0.0. The identity
      ``1 - cos x = 2 sin^2(x/2)`` removes the subtraction and gives 0.5.
      Exact: the series ``1/2 - x^2/24 + x^4/720`` in rational arithmetic.
    * ``"quadratic_small_root"``: the small root of ``x^2 - 1e8 x + 1 = 0``.
      The textbook ``(-b - sqrt(b^2 - 4ac)) / (2a)`` subtracts two numbers
      that agree to 16 digits and returns 7.45e-9 (relative error 0.2549).
      The stable route takes the large root
      ``x_big = -(b + sign(b) sqrt(b^2 - 4ac)) / (2a)``, which adds numbers
      of the same sign, and then ``c / (a x_big)`` = 1e-8 from Vieta's
      product of the roots. Exact: the textbook formula in 50-digit decimal
      arithmetic.
    * ``"sqrt_difference"``: ``sqrt(x + 1) - sqrt(x)`` at x = 1e16. Above
      2^53 the float ``x + 1`` is x again, so the naive value is 0.0; the
      rationalized form ``1 / (sqrt(x + 1) + sqrt(x))`` gives 5e-9. Exact:
      50-digit decimal arithmetic.
    * ``"sum_point_one"``: adding 0.1 one million times. A plain loop drifts
      to 100000.00000133288 (each addition rounds); :func:`kahan_sum` gives
      100000 within 1e-9. Exact: one million times the float 0.1 in rational
      arithmetic, 100000.0000000000055..., which rounds to 100000.0.

    Args:
        None.

    Returns:
        A dict with keys ``"one_minus_cos"``, ``"quadratic_small_root"``,
        ``"sqrt_difference"`` and ``"sum_point_one"``, each a dict with
        float values under ``"naive"``, ``"stable"`` and ``"exact"``.

    Raises:
        Nothing.

    Time Complexity:
        O(m) for the m = 10^6 additions of the summation case; the other
        three cases are O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> ex = cancellation_examples()
        >>> ex["one_minus_cos"]["naive"], ex["one_minus_cos"]["stable"]
        (0.0, 0.5)
        >>> ex["quadratic_small_root"]["naive"], ex["quadratic_small_root"]["stable"]
        (7.450580596923828e-09, 1e-08)
        >>> ex["sqrt_difference"]["stable"], ex["sum_point_one"]["naive"]
        (5e-09, 100000.00000133288)
    """
    # 1. (1 - cos x) / x^2 at x = 1e-8.
    x = 1e-8
    xf = Fraction(x)
    cos_case = {
        "naive": (1.0 - math.cos(x)) / (x * x),
        "stable": 2.0 * math.sin(x / 2.0) ** 2 / (x * x),
        "exact": float(Fraction(1, 2) - xf**2 / 24 + xf**4 / 720),
    }

    # 2. Small root of x^2 - 1e8 x + 1 = 0.
    a, b, c = 1.0, -1e8, 1.0
    root = math.sqrt(b * b - 4.0 * a * c)
    x_big = -(b + math.copysign(root, b)) / (2.0 * a)
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        da, db, dc = decimal.Decimal(a), decimal.Decimal(b), decimal.Decimal(c)
        exact_small = (-db - (db * db - 4 * da * dc).sqrt()) / (2 * da)
    quadratic_case = {
        "naive": (-b - root) / (2.0 * a),
        "stable": c / (a * x_big),
        "exact": float(exact_small),
    }

    # 3. sqrt(x + 1) - sqrt(x) at x = 1e16.
    y = 1e16
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        dy = decimal.Decimal(y)
        exact_diff = (dy + 1).sqrt() - dy.sqrt()
    sqrt_case = {
        "naive": math.sqrt(y + 1.0) - math.sqrt(y),
        "stable": 1.0 / (math.sqrt(y + 1.0) + math.sqrt(y)),
        "exact": float(exact_diff),
    }

    # 4. 0.1 added one million times.
    m = 1_000_000
    sum_case = {
        "naive": _plain_loop_sum(0.1 for _ in range(m)),
        "stable": kahan_sum(0.1 for _ in range(m)),
        "exact": float(Fraction(0.1) * m),
    }

    return {
        "one_minus_cos": cos_case,
        "quadratic_small_root": quadratic_case,
        "sqrt_difference": sqrt_case,
        "sum_point_one": sum_case,
    }


def hilbert(n: int) -> np.ndarray:
    """Return the n x n Hilbert matrix ``H[i, j] = 1 / (i + j + 1)``.

    H(n) is symmetric positive definite and notoriously ill-conditioned: its
    2-norm condition number grows like ``e^(3.5 n)``, from 1.6e4 at n = 4 to
    1.6e13 at n = 10, which makes it the standard test of whether an
    algorithm survives conditioning. The entries are built in one vectorized
    expression from an outer sum of indices.

    Args:
        n: The order, a positive integer.

    Returns:
        An n x n float64 array.

    Raises:
        TypeError: If ``n`` is not an integer.
        ValueError: If ``n < 1``.

    Time Complexity:
        Theta(n^2).

    Space Complexity:
        Theta(n^2).

    Examples:
        >>> hilbert(3)
        array([[1.        , 0.5       , 0.33333333],
               [0.5       , 0.33333333, 0.25      ],
               [0.33333333, 0.25      , 0.2       ]])
        >>> hilbert(0)
        Traceback (most recent call last):
        ...
        ValueError: n must be at least 1, got 0
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    i = np.arange(n, dtype=np.float64)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)
