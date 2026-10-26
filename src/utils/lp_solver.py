"""Thin, uniform wrappers around SciPy's HiGHS LP and MILP solvers.

Every Week 9 LP-based algorithm (vertex cover LP rounding, set cover
deterministic and randomized rounding, the exact facility-location MILP)
needs the same two things: solve a linear program, and solve the same
program with integer variables. This module gives both one calling
convention, ``(c, A_ub, b_ub, A_eq, b_eq, bounds)``, the convention of
:func:`scipy.optimize.linprog`, and one return shape, ``(objective, x)``
with ``x`` a plain ``list`` of ``float``.

The solver
----------
Both functions call HiGHS, the open-source solver SciPy ships:

* :func:`solve_lp` calls ``scipy.optimize.linprog(method="highs")``, which
  picks between the dual simplex method and an interior-point method. Both
  return a *basic* (vertex) optimum when the simplex path is taken, and an
  optimal point in every case.
* :func:`solve_milp` calls ``scipy.optimize.milp``, HiGHS's branch and
  bound (with cuts and presolve) over LP relaxations. ``A_ub``/``b_ub``
  become ``LinearConstraint(A_ub, -inf, b_ub)``, ``A_eq``/``b_eq`` become
  ``LinearConstraint(A_eq, b_eq, b_eq)`` and ``bounds`` is translated from
  the ``linprog`` convention to :class:`scipy.optimize.Bounds`. With
  ``integrality=None`` EVERY variable is integer, because the point of this
  wrapper is integer programs; pass an explicit 0/1 list for a mixed one.

Both raise :class:`RuntimeError` carrying the solver's own message whenever
the status is anything but "optimal" (infeasible, unbounded, iteration or
time limit), so a caller never silently uses a meaningless ``x``. Entries of
``x`` in ``(-1e-9, 0]`` are returned as ``0.0``: floating-point solvers
report values such as ``-1e-12`` for a variable whose true value is 0, and a
rounding rule like "pick S if x_S >= 1/f" or a probability "pick S with
probability x_S" must never see a negative number.

The guarantee these wrappers exist to expose: LP <= IP
-----------------------------------------------------
For a minimization integer program (IP), drop the integrality constraints
and the result is its *LP relaxation*. Then

    LP* <= IP* (= OPT),

and the proof is one line: every integer-feasible point satisfies all the
LP's constraints, so the LP minimizes over a *superset* of the IP's feasible
set, and a minimum over a larger set can only be smaller or equal. This is
the lower bound every LP-rounding analysis starts from: if a rounding turns
the fractional ``x`` into an integral solution of cost at most
``alpha * LP*``, then that cost is at most ``alpha * OPT``, an
``alpha``-approximation, without ever knowing OPT.

The ratio ``IP* / LP*`` on an instance is its *integrality gap*, and it
limits what any rounding of this LP can prove. The triangle K3 shows it for
vertex cover (minimize x_1 + x_2 + x_3 subject to x_u + x_v >= 1 for each of
the three edges, 0 <= x <= 1): setting every x_v = 1/2 satisfies every edge
with equality, so LP* <= 3/2, and adding the three constraints gives
2 (x_1 + x_2 + x_3) >= 3, so LP* >= 3/2; hence LP* = 1.5. Any integral cover
of a triangle needs two vertices (one vertex misses the opposite edge), so
IP* = 2. The gap 2 / 1.5 = 4/3 appears in the examples below.

======================= ============================= =====================
Function                Time                          Space
======================= ============================= =====================
solve_lp                polynomial in practice (HiGHS O(n m) to store the
                        simplex/IPM; m constraints,   dense constraint
                        n variables)                  matrices
solve_milp              exponential in the worst      O(n m) plus the
                        case (branch and bound; the   branch-and-bound
                        problem is NP-hard)           tree
======================= ============================= =====================

Examples:
    >>> value, x = solve_lp([1, 1], A_ub=[[-1, -2]], b_ub=[-2])
    >>> value, x
    (1.0, [0.0, 1.0])
    >>> triangle = [[-1, -1, 0], [0, -1, -1], [-1, 0, -1]]
    >>> solve_lp([1, 1, 1], A_ub=triangle, b_ub=[-1, -1, -1], bounds=(0, 1))[0]
    1.5
    >>> solve_milp([1, 1, 1], A_ub=triangle, b_ub=[-1, -1, -1], bounds=(0, 1))[0]
    2.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

__all__ = ["solve_lp", "solve_milp"]

#: Values of x in (-_NEG_TOL, 0] are reported as exactly 0.0.
_NEG_TOL = 1e-9


def _clean_x(x: Sequence[float]) -> List[float]:
    """Return ``x`` as a list of floats with tiny negatives replaced by 0.0.

    Args:
        x: The solver's solution vector.

    Returns:
        A new list; every entry in ``(-1e-9, 0]`` (including ``-0.0``)
        becomes ``0.0``, every other entry is ``float(entry)``.

    Time Complexity:
        O(n) for n variables.

    Space Complexity:
        O(n).

    Examples:
        >>> _clean_x([-1e-12, 0.5, -0.0, -2.0])
        [0.0, 0.5, 0.0, -2.0]
    """
    cleaned = []
    for value in x:
        value = float(value)
        if -_NEG_TOL < value <= 0.0:
            value = 0.0
        cleaned.append(value)
    return cleaned


def _is_empty(matrix: Any) -> bool:
    """Return True when a constraint matrix or vector is absent or has no rows.

    Args:
        matrix: ``None``, a list (of lists) or a NumPy array.

    Returns:
        ``True`` for ``None`` or a zero-length sequence, so a caller may pass
        ``A_ub=[]`` for "no inequality constraints" (for example a vertex
        cover LP on a graph with no edges).

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _is_empty(None), _is_empty([]), _is_empty([[1, 2]])
        (True, True, False)
    """
    return matrix is None or len(matrix) == 0


def _paired(A: Any, b: Any, name: str) -> bool:
    """Check that a matrix and its right-hand side are given together.

    Args:
        A: Constraint matrix or ``None``/empty.
        b: Right-hand side or ``None``/empty.
        name: ``"ub"`` or ``"eq"``, used in the error message.

    Returns:
        ``True`` when both are present, ``False`` when both are absent.

    Raises:
        ValueError: If exactly one of the two is present.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _paired([[1]], [2], "ub"), _paired(None, None, "eq")
        (True, False)
        >>> _paired([[1]], None, "ub")
        Traceback (most recent call last):
            ...
        ValueError: A_ub and b_ub must be given together
    """
    has_a, has_b = not _is_empty(A), not _is_empty(b)
    if has_a != has_b:
        raise ValueError(f"A_{name} and b_{name} must be given together")
    return has_a


def _is_bound_value(value: Any) -> bool:
    """Return True for a single bound entry: ``None`` or a real number.

    Args:
        value: A candidate lower or upper bound.

    Returns:
        ``True`` if ``value`` is ``None`` or a real (non-bool) number.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _is_bound_value(None), _is_bound_value(1.5), _is_bound_value((0, 1))
        (True, True, False)
    """
    if value is None:
        return True
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _to_scipy_bounds(bounds: Any, n: int) -> Bounds:
    """Translate ``linprog``-style bounds into :class:`scipy.optimize.Bounds`.

    Args:
        bounds: ``None`` (every variable in ``[0, inf)``), one ``(lb, ub)``
            pair applied to every variable, or a sequence of ``n`` pairs.
            ``None`` inside a pair means unbounded in that direction.
        n: Number of variables.

    Returns:
        A :class:`scipy.optimize.Bounds` with ``lb`` and ``ub`` arrays of
        length ``n``.

    Raises:
        ValueError: If ``bounds`` is neither one pair nor ``n`` pairs.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> b = _to_scipy_bounds([(0, 1), (None, 3)], 2)
        >>> b.lb.tolist(), b.ub.tolist()
        ([0.0, -inf], [1.0, 3.0])
        >>> _to_scipy_bounds(None, 2).ub.tolist()
        [inf, inf]
    """
    if bounds is None:
        pairs = [(0.0, None)] * n
    elif (
        len(bounds) == 2
        and _is_bound_value(bounds[0])
        and _is_bound_value(bounds[1])
    ):
        pairs = [tuple(bounds)] * n
    elif len(bounds) == n:
        pairs = [tuple(pair) for pair in bounds]
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("each bound must be a (lb, ub) pair")
    else:
        raise ValueError(
            f"bounds must be one (lb, ub) pair or {n} pairs, got {len(bounds)}"
        )
    lb = np.array([-math.inf if lo is None else float(lo) for lo, _ in pairs])
    ub = np.array([math.inf if hi is None else float(hi) for _, hi in pairs])
    return Bounds(lb, ub)


def solve_lp(
    c: Sequence[float],
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Any = None,
) -> Tuple[float, List[float]]:
    """Minimize ``c @ x`` subject to linear constraints with SciPy's HiGHS.

    Solves ``min c^T x`` subject to ``A_ub x <= b_ub``, ``A_eq x = b_eq``
    and the variable bounds, through
    ``scipy.optimize.linprog(method="highs")``. A covering constraint
    ``sum x >= 1`` is written ``-sum x <= -1`` to fit ``A_ub``. Any status
    other than optimal raises, so the returned ``x`` is always an optimum.

    Args:
        c: Objective coefficients, one per variable.
        A_ub: Inequality matrix (rows are constraints), or ``None``/``[]``.
        b_ub: Inequality right-hand sides, given exactly when ``A_ub`` is.
        A_eq: Equality matrix, or ``None``/``[]``.
        b_eq: Equality right-hand sides, given exactly when ``A_eq`` is.
        bounds: ``linprog`` convention: ``None`` means every ``x_i >= 0``;
            one ``(lb, ub)`` pair applies to every variable; a list of pairs
            gives one per variable; ``None`` inside a pair is unbounded.

    Returns:
        ``(value, x)``: the optimal objective as a ``float`` and the optimal
        point as a ``list`` of ``float``, with entries in ``(-1e-9, 0]``
        replaced by ``0.0``.

    Raises:
        RuntimeError: If HiGHS does not report an optimal solution
            (infeasible, unbounded, or a limit was hit); the message
            includes the solver's own message.
        ValueError: If a constraint matrix is given without its right-hand
            side or vice versa (SciPy may raise ``ValueError`` for
            inconsistent shapes too).

    Time Complexity:
        Polynomial in practice for m constraints and n variables (HiGHS dual
        simplex or interior point; interior-point methods are polynomial in
        the worst case, the simplex method is not but is fast in practice).

    Space Complexity:
        O(n m) for the dense constraint matrices plus the solver's working
        storage.

    Examples:
        Minimize x + y subject to x + 2y >= 2 (written -x - 2y <= -2); the
        optimum puts all weight on y:

        >>> solve_lp([1, 1], A_ub=[[-1, -2]], b_ub=[-2])
        (1.0, [0.0, 1.0])

        The vertex cover LP of a triangle is half-integral, value 1.5:

        >>> triangle = [[-1, -1, 0], [0, -1, -1], [-1, 0, -1]]
        >>> solve_lp([1, 1, 1], A_ub=triangle, b_ub=[-1, -1, -1], bounds=(0, 1))
        (1.5, [0.5, 0.5, 0.5])

        An infeasible LP (x <= 1 and x >= 2) raises:

        >>> solve_lp([1], A_ub=[[1], [-1]], b_ub=[1, -2])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        RuntimeError: linprog did not find an optimal solution (status 2): ...
    """
    has_ub = _paired(A_ub, b_ub, "ub")
    has_eq = _paired(A_eq, b_eq, "eq")
    result = linprog(
        c,
        A_ub=A_ub if has_ub else None,
        b_ub=b_ub if has_ub else None,
        A_eq=A_eq if has_eq else None,
        b_eq=b_eq if has_eq else None,
        bounds=(0, None) if bounds is None else bounds,
        method="highs",
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"linprog did not find an optimal solution (status {result.status}): "
            f"{result.message}"
        )
    return float(result.fun), _clean_x(result.x)


def solve_milp(
    c: Sequence[float],
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Any = None,
    integrality: Optional[Sequence[int]] = None,
) -> Tuple[float, List[float]]:
    """Minimize ``c @ x`` with integer variables using SciPy's HiGHS MILP.

    Same problem and argument convention as :func:`solve_lp`, solved by
    ``scipy.optimize.milp``. ``A_ub``/``b_ub`` become
    ``LinearConstraint(A_ub, -inf, b_ub)``, ``A_eq``/``b_eq`` become
    ``LinearConstraint(A_eq, b_eq, b_eq)``, and ``linprog``-style
    ``bounds`` are converted to :class:`scipy.optimize.Bounds`. With
    ``integrality=None`` every variable is integer.

    Args:
        c: Objective coefficients, one per variable.
        A_ub: Inequality matrix, or ``None``/``[]``.
        b_ub: Inequality right-hand sides, given exactly when ``A_ub`` is.
        A_eq: Equality matrix, or ``None``/``[]``.
        b_eq: Equality right-hand sides, given exactly when ``A_eq`` is.
        bounds: ``linprog`` convention; ``None`` means every variable lies
            in ``[0, inf)``.
        integrality: ``None`` (every variable integer) or one entry per
            variable in SciPy's convention (0 continuous, 1 integer).

    Returns:
        ``(value, x)``: the optimal objective as a ``float`` and the optimal
        point as a ``list`` of ``float``, with entries in ``(-1e-9, 0]``
        replaced by ``0.0``. Integer variables come back as integral floats
        up to HiGHS's feasibility tolerance.

    Raises:
        RuntimeError: If HiGHS does not report an optimal solution
            (infeasible, unbounded, or a limit was hit); the message
            includes the solver's own message.
        ValueError: If a constraint matrix is given without its right-hand
            side or vice versa, or ``bounds`` has the wrong length.

    Time Complexity:
        Exponential in the worst case: integer programming is NP-hard, and
        branch and bound may explore up to 2^n nodes for n binary variables,
        each solving an LP. Fine for the small exact baselines of Week 9.

    Space Complexity:
        O(n m) for the constraint matrices plus the branch-and-bound tree.

    Examples:
        The triangle vertex cover: LP value 1.5, integer optimum 2.0.

        >>> triangle = [[-1, -1, 0], [0, -1, -1], [-1, 0, -1]]
        >>> value, x = solve_milp([1, 1, 1], A_ub=triangle, b_ub=[-1, -1, -1],
        ...                       bounds=(0, 1))
        >>> value, sum(round(v) for v in x)
        (2.0, 2)

        Equality constraints and mixed integrality: x integer, y continuous,
        x + y = 2.5, minimize -x (so x is pushed up to the integer 2):

        >>> solve_milp([-1, 0], A_eq=[[1, 1]], b_eq=[2.5], integrality=[1, 0])
        (-2.0, [2.0, 0.5])

        An infeasible integer program (2x = 1 has no integer solution):

        >>> solve_milp([1], A_eq=[[2]], b_eq=[1])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        RuntimeError: milp did not find an optimal solution (status 2): ...
    """
    n = len(c)
    constraints = []
    if _paired(A_ub, b_ub, "ub"):
        constraints.append(
            LinearConstraint(
                np.asarray(A_ub, dtype=float),
                -np.inf,
                np.asarray(b_ub, dtype=float),
            )
        )
    if _paired(A_eq, b_eq, "eq"):
        b_eq_arr = np.asarray(b_eq, dtype=float)
        constraints.append(
            LinearConstraint(np.asarray(A_eq, dtype=float), b_eq_arr, b_eq_arr)
        )
    if integrality is None:
        integrality_arr = np.ones(n, dtype=int)
    else:
        integrality_arr = np.asarray(integrality, dtype=int)
    result = milp(
        np.asarray(c, dtype=float),
        integrality=integrality_arr,
        bounds=_to_scipy_bounds(bounds, n),
        constraints=constraints if constraints else None,
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"milp did not find an optimal solution (status {result.status}): "
            f"{result.message}"
        )
    return float(result.fun), _clean_x(result.x)
