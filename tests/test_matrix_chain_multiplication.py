"""Tests for the three matrix-chain multiplication solvers and their tables.

A matrix-chain answer is two things, a number and an order, and either can
be wrong in a way that looks right. A cost that is too small is not
achievable by any order; a cost that is too large is achievable but not
optimal; and an order string can be well formed, cheap and still describe a
different product, for instance one that skips or swaps a matrix. Checking
the module against a second copy of its own recurrence would agree with
every one of those mistakes, so the claims here rest on witnesses that do
not share its mechanism:

* **Hand-checked anchors.** Six chains from the Week 6 build spec, with
  their costs and orders worked out on paper, the CLRS 4e section 14.2
  instance among them. The arithmetic is written out in
  :class:`TestHandCheckedAnchors`, together with the full ``m`` and ``s``
  tables of CLRS figure 14.5. These are the only checks in the file that a
  consistently wrong implementation cannot fool.
* **Brute force over every parenthesization.**
  :func:`all_parenthesizations` builds every full parenthesization of the
  chain as a string, pricing each one as it is built, and
  :func:`brute_force_min` takes the minimum. There is no table, no memo
  and no choice of best split: it is the definition of the problem, and
  its own output is checked against the Catalan numbers before anything
  is compared with it. With at most 9 matrices there are at most 1430
  orders, so it is cheap.
* **numpy, actually multiplying.** :func:`multiply_in_written_order`
  evaluates the returned string with real integer matrices, counting the
  scalar multiplications each product performs, and the result is compared
  with the plain left-to-right product. Agreement proves the string is a
  valid reordering of the same product rather than merely a cheap string,
  and the counted cost is a third, independent check of the optimum.

Three further properties are checked because the assignment grades them:
the bottom-up tables are 1-indexed CLRS tables with ``m[i][i] == 0`` and
``s[i][j]`` in ``[i, j - 1]``; :func:`parenthesization_cost`, the pricing
helper the spec relies on, refuses malformed or out-of-order strings; and
``mcm_memoized`` keeps an explicit memo rather than ``functools.lru_cache``,
which is proven both from its parsed source and from its call counts.

Every random draw is seeded from a fixed constant, no chain exceeds 9
matrices, and the plain recursion is exponential, so the file stays fast.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import ast
import functools
import inspect
import random
import re
import textwrap
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
import pytest

import src.dp_advanced.matrix_chain_multiplication as mcm_module
from src.dp_advanced.matrix_chain_multiplication import (
    matrix_chain_order,
    mcm_bottom_up,
    mcm_memoized,
    mcm_recursive,
    optimal_parenthesization,
)
from src.utils.matrix_utils import parenthesization_cost

CostSolver = Callable[[Sequence[int]], int]


# ----------------------------------------------------------------------
# The independent oracles
# ----------------------------------------------------------------------
def all_parenthesizations(p: Sequence[int], i: int, j: int) -> List[Tuple[str, int]]:
    """Return every full parenthesization of ``Ai..Aj`` with its exact cost.

    A product of two or more matrices is always some left part times some
    right part, so every order of ``Ai..Aj`` is ``(L R)`` for one split
    ``k`` and one order ``L`` of ``Ai..Ak`` and one order ``R`` of
    ``A(k+1)..Aj``. Enumerating all three choices enumerates every order
    exactly once. Each order is priced as it is built: the left part is a
    ``p[i-1] x p[k]`` matrix, the right part a ``p[k] x p[j]`` matrix, and
    joining them costs ``p[i-1] * p[k] * p[j]``.

    Nothing is minimised here. The module keeps only the best order of
    each sub-chain; this keeps all of them, which is why it can serve as
    the oracle.

    Args:
        p: The dimension list; matrix ``Ai`` is ``p[i-1] x p[i]``.
        i: First matrix, 1-indexed.
        j: Last matrix, 1-indexed, with ``i <= j``.

    Returns:
        A list of ``(expression, cost)`` pairs, one per order, with the
        expression in the module's ``"((A1A2)A3)"`` style.
    """
    if i == j:
        return [(f"A{i}", 0)]

    orders: List[Tuple[str, int]] = []
    for k in range(i, j):
        lefts = all_parenthesizations(p, i, k)
        rights = all_parenthesizations(p, k + 1, j)
        join = p[i - 1] * p[k] * p[j]
        for left_expr, left_cost in lefts:
            for right_expr, right_cost in rights:
                orders.append(
                    (f"({left_expr}{right_expr})", left_cost + right_cost + join)
                )
    return orders


def brute_force_min(p: Sequence[int]) -> int:
    """Return the cheapest cost over every parenthesization of the chain."""
    return min(cost for _, cost in all_parenthesizations(p, 1, len(p) - 1))


def optimal_orders(p: Sequence[int]) -> Set[str]:
    """Return the set of every parenthesization that achieves the optimum."""
    orders = all_parenthesizations(p, 1, len(p) - 1)
    best = min(cost for _, cost in orders)
    return {expr for expr, cost in orders if cost == best}


_TOKEN = re.compile(r"\(|\)|A[1-9][0-9]*")


def multiply_in_written_order(
    expr: str, matrices: Sequence[np.ndarray]
) -> Tuple[np.ndarray, int, int]:
    """Evaluate a parenthesization with real matrices, in the order it says.

    The string is read as a stream of tokens. A leaf pushes its matrix, and
    each ``)`` pops the two most recent results and pushes their product,
    which is exactly the product that closing parenthesis writes. numpy
    raises if the two shapes do not chain, so a string that pairs the wrong
    neighbours cannot slip through. Each product's scalar multiplications
    are counted as ``rows * inner * cols`` of the two operands.

    This is deliberately a different parser from
    :func:`parenthesization_cost`, so the two cannot share a mistake.

    Args:
        expr: The parenthesization, for example ``"((A1A2)A3)"``.
        matrices: ``A1..An`` as numpy arrays, in chain order.

    Returns:
        ``(product, scalar_multiplications, products_performed)``.
    """
    tokens = _TOKEN.findall(expr)
    assert "".join(tokens) == expr, f"stray characters in {expr!r}"
    assert expr.count("(") == expr.count(")") == len(matrices) - 1

    stack: List[np.ndarray] = []
    next_leaf = 1
    scalar_multiplications = 0
    products = 0
    for token in tokens:
        if token == "(":
            continue
        if token == ")":
            right = stack.pop()
            left = stack.pop()
            scalar_multiplications += left.shape[0] * left.shape[1] * right.shape[1]
            stack.append(left @ right)
            products += 1
        else:
            index = int(token[1:])
            assert index == next_leaf, f"{token} out of order in {expr!r}"
            stack.append(matrices[index - 1])
            next_leaf += 1

    assert len(stack) == 1
    assert next_leaf == len(matrices) + 1
    return stack[0], scalar_multiplications, products


def chain_matrices(p: Sequence[int], seed: int) -> List[np.ndarray]:
    """Draw small integer matrices with the shapes ``p`` describes.

    Entries are integers in ``[-3, 3]`` stored as ``int64``, so every
    product is computed exactly and the comparison has no rounding to
    forgive. The largest anchor product stays far below ``2^63``.
    """
    rng = np.random.default_rng(seed)
    return [
        rng.integers(-3, 4, size=(p[index - 1], p[index]), dtype=np.int64)
        for index in range(1, len(p))
    ]


# ----------------------------------------------------------------------
# What is under test, and on what instances
# ----------------------------------------------------------------------
def bottom_up_cost(p: Sequence[int]) -> int:
    """Adapter reading the optimum ``m[1][n]`` out of the bottom-up table."""
    m, _ = mcm_bottom_up(p)
    n = len(m) - 1  # read n off the table, so p may be a one-shot iterator
    return m[1][n]


def chain_order_cost(p: Sequence[int]) -> int:
    """Adapter exposing :func:`matrix_chain_order`'s cost alone."""
    return matrix_chain_order(p)[0]


#: Every function that answers "how much". The spec names three
#: implementations; the convenience entry point joins them because it must
#: report the same number.
COST_SOLVERS: Dict[str, CostSolver] = {
    "recursive": mcm_recursive,
    "memoized": mcm_memoized,
    "bottom_up": bottom_up_cost,
    "matrix_chain_order": chain_order_cost,
}
SOLVER_IDS: List[str] = sorted(COST_SOLVERS)
SOLVER_PARAMS: List[CostSolver] = [COST_SOLVERS[name] for name in SOLVER_IDS]

#: Every public entry point that takes ``p``. The contract tests run over
#: all of them, so a check that drifts out of one of them is caught.
ENTRY_POINTS: Dict[str, Callable[[Sequence[int]], object]] = {
    "recursive": mcm_recursive,
    "memoized": mcm_memoized,
    "bottom_up": mcm_bottom_up,
    "matrix_chain_order": matrix_chain_order,
}
ENTRY_IDS: List[str] = sorted(ENTRY_POINTS)
ENTRY_PARAMS = [ENTRY_POINTS[name] for name in ENTRY_IDS]

#: The build spec's hand-checked table: ``(p, minimum cost, order)``. Each
#: order is checked below to be the *only* optimal one, which is what makes
#: comparing the exact string fair rather than tie-fragile.
ANCHORS: List[Tuple[List[int], int, str]] = [
    ([30, 35, 15, 5, 10, 20, 25], 15125, "((A1(A2A3))((A4A5)A6))"),
    ([10, 20, 30], 6000, "(A1A2)"),
    ([10, 20, 30, 40, 30], 30000, "(((A1A2)A3)A4)"),
    ([40, 20, 30, 10, 30], 26000, "((A1(A2A3))A4)"),
    ([5, 10, 3, 12, 5, 50, 6], 2010, "((A1A2)((A3A4)(A5A6)))"),
    ([2, 3], 0, "A1"),
]
ANCHOR_IDS: List[str] = ["clrs", "two", "left-deep", "inner-first", "balanced", "one"]
ANCHOR_PARAMS = [
    pytest.param(p, cost, expr, id=name)
    for (p, cost, expr), name in zip(ANCHORS, ANCHOR_IDS)
]

CLRS_P: List[int] = [30, 35, 15, 5, 10, 20, 25]

#: Two families of seeded chains. "wide" draws dimensions from 5 to 100,
#: so costs differ sharply between orders; "narrow" draws from 1 to 6, so
#: ties are common and dimensions of 1 (row and column vectors) appear.
FAMILY_SEEDS: Dict[str, int] = {"wide": 6000, "narrow": 7000}
FAMILY_RANGES: Dict[str, Tuple[int, int]] = {"wide": (5, 100), "narrow": (1, 6)}
CHAIN_LENGTHS: Tuple[int, ...] = tuple(range(1, 10))
SEEDS: Tuple[int, ...] = (0, 1, 2)


def make_chain(family: str, n: int, seed: int) -> List[int]:
    """Build one reproducible dimension list for a chain of ``n`` matrices.

    Every draw comes from a local :class:`random.Random` seeded from the
    family, the length and the seed together, so a failure can be rebuilt
    from its test id alone.
    """
    rng = random.Random(FAMILY_SEEDS[family] + 100 * n + seed)
    low, high = FAMILY_RANGES[family]
    return [rng.randint(low, high) for _ in range(n + 1)]


#: Every ``(family, n, seed)`` triple: 2 families x 9 lengths x 3 seeds.
BATTERY = [
    pytest.param(family, n, seed, id=f"{family}-n{n}-seed{seed}")
    for family in sorted(FAMILY_SEEDS)
    for n in CHAIN_LENGTHS
    for seed in SEEDS
]


# ----------------------------------------------------------------------
# The oracles are checked before anything is compared with them
# ----------------------------------------------------------------------
class TestOracles:
    """The brute force and the pricing helper, each checked on its own.

    An oracle that silently missed some orders would still return a
    minimum, just possibly the wrong one, so the enumeration is first
    checked to produce exactly the Catalan number ``C(n - 1)`` of distinct
    orders. :func:`parenthesization_cost` is the helper the spec uses to
    accept tied answers, so it is checked against hand arithmetic and then
    against every order the enumeration builds.
    """

    #: C(0) through C(8): the number of full parenthesizations of 1..9
    #: matrices. These are the standard Catalan numbers.
    CATALAN: Tuple[int, ...] = (1, 1, 2, 5, 14, 42, 132, 429, 1430)

    @pytest.mark.parametrize("n", CHAIN_LENGTHS)
    def test_the_enumeration_finds_every_order_exactly_once(self, n: int) -> None:
        orders = all_parenthesizations([2] * (n + 1), 1, n)
        expressions = [expr for expr, _ in orders]
        assert len(expressions) == self.CATALAN[n - 1]
        assert len(set(expressions)) == len(expressions)

    def test_the_brute_force_finds_the_clrs_optimum(self) -> None:
        assert brute_force_min(CLRS_P) == 15125

    def test_the_brute_force_prices_the_worst_clrs_orders_by_hand(self) -> None:
        """Left to right and right to left on the CLRS chain, worked on paper."""
        # Left to right: 30*35*15 + 30*15*5 + 30*5*10 + 30*10*20 + 30*20*25
        #              = 15750 + 2250 + 1500 + 6000 + 15000 = 40500.
        # Right to left: 10*20*25 + 5*10*25 + 15*5*25 + 35*15*25 + 30*35*25
        #              = 5000 + 1250 + 1875 + 13125 + 26250 = 47500.
        priced = dict(all_parenthesizations(CLRS_P, 1, 6))
        assert priced["(((((A1A2)A3)A4)A5)A6)"] == 40500
        assert priced["(A1(A2(A3(A4(A5A6)))))"] == 47500

    @pytest.mark.parametrize(
        "p, expr, cost",
        [
            # CLRS 4e section 14.2's opening example: 10 x 100, 100 x 5,
            # 5 x 50. (A1A2) first: 10*100*5 + 10*5*50 = 5000 + 2500.
            ([10, 100, 5, 50], "((A1A2)A3)", 7500),
            # (A2A3) first: 100*5*50 + 10*100*50 = 25000 + 50000.
            ([10, 100, 5, 50], "(A1(A2A3))", 75000),
            (CLRS_P, "((A1(A2A3))((A4A5)A6))", 15125),
            (CLRS_P, "(((((A1A2)A3)A4)A5)A6)", 40500),
            ([10, 20, 30], "(A1A2)", 6000),
            ([10, 20], "A1", 0),
        ],
        ids=[
            "clrs-intro-good",
            "clrs-intro-bad",
            "clrs-best",
            "clrs-ltr",
            "two",
            "one",
        ],
    )
    def test_parenthesization_cost_matches_hand_arithmetic(
        self, p: List[int], expr: str, cost: int
    ) -> None:
        assert parenthesization_cost(p, expr) == cost

    @pytest.mark.parametrize("family", sorted(FAMILY_SEEDS))
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_parenthesization_cost_agrees_on_every_enumerated_order(
        self, family: str, n: int
    ) -> None:
        p = make_chain(family, n, 0)
        for expr, cost in all_parenthesizations(p, 1, n):
            assert parenthesization_cost(p, expr) == cost, expr


# ----------------------------------------------------------------------
# The hand-checked cases
# ----------------------------------------------------------------------
class TestHandCheckedAnchors:
    """The spec's six chains, worked out on paper before any code ran.

    The arithmetic for each optimum, with ``Ai`` written as its shape:

    * ``[30, 35, 15, 5, 10, 20, 25]``, CLRS 4e section 14.2: ``(A2A3)``
      costs 35*15*5 = 2625, ``A1(A2A3)`` adds 30*35*5 = 5250, ``(A4A5)``
      costs 5*10*20 = 1000, ``(A4A5)A6`` adds 5*20*25 = 2500, and the final
      join is 30*5*25 = 3750. 2625 + 5250 + 1000 + 2500 + 3750 = **15125**.
    * ``[10, 20, 30]``: one product, 10*20*30 = **6000**.
    * ``[10, 20, 30, 40, 30]``: left to right is 10*20*30 + 10*30*40 +
      10*40*30 = 6000 + 12000 + 12000 = **30000**; the narrow first
      dimension makes every running product 10 rows tall.
    * ``[40, 20, 30, 10, 30]``: ``(A2A3)`` is 20*30*10 = 6000, then
      40*20*10 = 8000, then 40*10*30 = 12000, total **26000**.
    * ``[5, 10, 3, 12, 5, 50, 6]``: ``(A1A2)`` 5*10*3 = 150, ``(A3A4)``
      3*12*5 = 180, ``(A5A6)`` 5*50*6 = 1500, joining those two 3*5*6 = 90,
      and the final join 5*3*6 = 90. Total **2010**.
    * ``[2, 3]``: a single matrix needs no multiplication, cost **0**.

    The brute force confirms each order is the unique optimum, so the exact
    strings are asserted for every anchor, not only the CLRS one.
    """

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    @pytest.mark.parametrize("p, cost, expr", ANCHOR_PARAMS)
    def test_every_solver_returns_the_paper_cost(
        self, p: List[int], cost: int, expr: str, solver: CostSolver
    ) -> None:
        assert solver(p) == cost

    @pytest.mark.parametrize("p, cost, expr", ANCHOR_PARAMS)
    def test_the_returned_order_costs_the_optimum(
        self, p: List[int], cost: int, expr: str
    ) -> None:
        returned_cost, returned_expr = matrix_chain_order(p)
        assert returned_cost == cost
        assert parenthesization_cost(p, returned_expr) == cost

    @pytest.mark.parametrize("p, cost, expr", ANCHOR_PARAMS)
    def test_each_anchor_has_exactly_one_optimal_order(
        self, p: List[int], cost: int, expr: str
    ) -> None:
        """Makes the exact-string assertions below fair: there is no tie to break."""
        assert brute_force_min(p) == cost
        assert optimal_orders(p) == {expr}

    @pytest.mark.parametrize("p, cost, expr", ANCHOR_PARAMS)
    def test_the_returned_order_is_the_paper_order(
        self, p: List[int], cost: int, expr: str
    ) -> None:
        assert matrix_chain_order(p) == (cost, expr)

    def test_the_clrs_example_exact_string(self) -> None:
        """Required by the spec on its own: the textbook's printed answer."""
        assert matrix_chain_order(CLRS_P)[1] == "((A1(A2A3))((A4A5)A6))"

    def test_the_clrs_m_table_matches_figure_14_5(self) -> None:
        """Every cell of the textbook's m table, row by row from i = 1."""
        m, _ = mcm_bottom_up(CLRS_P)
        expected = [
            [0, 15750, 7875, 9375, 11875, 15125],
            [0, 0, 2625, 4375, 7125, 10500],
            [0, 0, 0, 750, 2500, 5375],
            [0, 0, 0, 0, 1000, 3500],
            [0, 0, 0, 0, 0, 5000],
            [0, 0, 0, 0, 0, 0],
        ]
        assert [row[1:] for row in m[1:]] == expected

    def test_the_clrs_s_table_matches_figure_14_5(self) -> None:
        """Every split of the textbook's s table, row by row from i = 1."""
        _, s = mcm_bottom_up(CLRS_P)
        expected = [
            [0, 1, 1, 3, 3, 3],
            [0, 0, 2, 3, 3, 3],
            [0, 0, 0, 3, 3, 3],
            [0, 0, 0, 0, 4, 5],
            [0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0],
        ]
        assert [row[1:] for row in s[1:]] == expected

    @pytest.mark.parametrize(
        "i, j, expected",
        [
            (1, 6, "((A1(A2A3))((A4A5)A6))"),
            # s[2][5] = 3, then s[2][3] = 2 and s[4][5] = 4.
            (2, 5, "((A2A3)(A4A5))"),
            # s[1][3] = 1, then s[2][3] = 2.
            (1, 3, "(A1(A2A3))"),
            # s[4][6] = 5, then s[4][5] = 4.
            (4, 6, "((A4A5)A6)"),
            (5, 6, "(A5A6)"),
            (1, 1, "A1"),
            (6, 6, "A6"),
        ],
    )
    def test_clrs_sub_chains_read_off_the_s_table(
        self, i: int, j: int, expected: str
    ) -> None:
        _, s = mcm_bottom_up(CLRS_P)
        assert optimal_parenthesization(s, i, j) == expected

    def test_the_clrs_optimum_beats_left_to_right_by_hand(self) -> None:
        """40500 for the naive order (worked in TestOracles) against 15125."""
        cost, expr = matrix_chain_order(CLRS_P)
        assert parenthesization_cost(CLRS_P, "(((((A1A2)A3)A4)A5)A6)") == 40500
        assert cost == 15125 < 40500

    @pytest.mark.parametrize(
        "p, cost, expr",
        [
            # CLRS 4e section 14.2's opening example, ten times cheaper.
            ([10, 100, 5, 50], 7500, "((A1A2)A3)"),
            # A row vector, a column vector, a row vector. (A1A2) is a
            # 1 x 1 scalar for 1*100*1 = 100, then 1*1*100 = 100: 200.
            # The other order builds a 100 x 100 outer product first,
            # 100*1*100 = 10000, then 1*100*100 = 10000: 20000.
            ([1, 100, 1, 100], 200, "((A1A2)A3)"),
            # Two matrices: a single product, 2*3*4.
            ([2, 3, 4], 24, "(A1A2)"),
            ([1, 1], 0, "A1"),
        ],
        ids=["clrs-intro", "vectors", "two-small", "one-by-one"],
    )
    def test_further_hand_checked_chains(
        self, p: List[int], cost: int, expr: str
    ) -> None:
        assert matrix_chain_order(p) == (cost, expr)
        for solver in COST_SOLVERS.values():
            assert solver(p) == cost

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_tuple_is_accepted(self, solver: CostSolver) -> None:
        assert solver(tuple(CLRS_P)) == 15125

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    def test_a_generator_is_accepted(self, solver: CostSolver) -> None:
        assert solver(dim for dim in CLRS_P) == 15125


# ----------------------------------------------------------------------
# Agreement with brute force
# ----------------------------------------------------------------------
class TestAgreementWithBruteForce:
    """Every solver against the enumeration of all orders, on 54 chains.

    The oracle is the definition of the problem, so where a solver
    disagrees with it the solver is wrong. The returned order is checked
    two ways: it must be one of the oracle's optimal orders, and the
    pricing helper must agree that it costs the optimum.
    """

    @pytest.mark.parametrize("solver", SOLVER_PARAMS, ids=SOLVER_IDS)
    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_solver_matches_the_oracle(
        self, family: str, n: int, seed: int, solver: CostSolver
    ) -> None:
        p = make_chain(family, n, seed)
        assert solver(p) == brute_force_min(p)

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_the_three_implementations_return_one_value(
        self, family: str, n: int, seed: int
    ) -> None:
        p = make_chain(family, n, seed)
        answers = {solver(p) for solver in COST_SOLVERS.values()}
        assert len(answers) == 1

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_the_returned_order_is_an_optimal_order(
        self, family: str, n: int, seed: int
    ) -> None:
        p = make_chain(family, n, seed)
        cost, expr = matrix_chain_order(p)
        assert expr in optimal_orders(p)
        assert parenthesization_cost(p, expr) == cost


# ----------------------------------------------------------------------
# The bottom-up tables
# ----------------------------------------------------------------------
class TestBottomUpTables:
    """``m`` and ``s`` are the CLRS tables: 1-indexed, triangular, consistent.

    Beyond the shape, every cell is checked for what it claims to be:
    ``m[i][j]`` is the optimum of the sub-chain ``Ai..Aj`` by brute force,
    and ``s[i][j]`` is a split that achieves it, the smallest such split,
    which is the tie rule :func:`matrix_chain_order` documents.
    """

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_both_tables_are_n_plus_one_square(
        self, family: str, n: int, seed: int
    ) -> None:
        m, s = mcm_bottom_up(make_chain(family, n, seed))
        for table in (m, s):
            assert len(table) == n + 1
            assert all(len(row) == n + 1 for row in table)

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_the_diagonal_of_m_is_zero(self, family: str, n: int, seed: int) -> None:
        m, _ = mcm_bottom_up(make_chain(family, n, seed))
        assert all(m[i][i] == 0 for i in range(1, n + 1))

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_row_and_column_zero_are_unused_padding(
        self, family: str, n: int, seed: int
    ) -> None:
        m, s = mcm_bottom_up(make_chain(family, n, seed))
        for table in (m, s):
            assert all(value == 0 for value in table[0])
            assert all(row[0] == 0 for row in table)

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_below_the_diagonal_is_left_at_zero(
        self, family: str, n: int, seed: int
    ) -> None:
        m, s = mcm_bottom_up(make_chain(family, n, seed))
        for i in range(1, n + 1):
            for j in range(1, i):
                assert m[i][j] == 0 and s[i][j] == 0
            assert s[i][i] == 0

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_every_split_lies_inside_its_interval(
        self, family: str, n: int, seed: int
    ) -> None:
        _, s = mcm_bottom_up(make_chain(family, n, seed))
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                assert i <= s[i][j] <= j - 1, (i, j, s[i][j])

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_every_cell_is_the_optimum_of_its_sub_chain(
        self, family: str, n: int, seed: int
    ) -> None:
        p = make_chain(family, n, seed)
        m, _ = mcm_bottom_up(p)
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                assert m[i][j] == brute_force_min(p[i - 1 : j + 1]), (i, j)

    @pytest.mark.parametrize("family, n, seed", BATTERY)
    def test_each_split_achieves_its_cell_and_is_the_first_that_does(
        self, family: str, n: int, seed: int
    ) -> None:
        p = make_chain(family, n, seed)
        m, s = mcm_bottom_up(p)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):

                def via(k: int) -> int:
                    return m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]

                k = s[i][j]
                assert via(k) == m[i][j]
                assert all(via(earlier) > m[i][j] for earlier in range(i, k))

    def test_equal_square_matrices_tie_and_the_smallest_split_wins(self) -> None:
        """Every order of four 4 x 4 matrices costs 3 * 4^3 = 192.

        With all five orders tied, the first minimum at each level is split
        1, so the answer nests to the right: A1 times (A2 times (A3A4)).
        """
        p = [4, 4, 4, 4, 4]
        assert {cost for _, cost in all_parenthesizations(p, 1, 4)} == {192}
        assert matrix_chain_order(p) == (192, "(A1(A2(A3A4)))")

    def test_a_single_matrix_gives_two_by_two_zero_tables(self) -> None:
        assert mcm_bottom_up([2, 3]) == ([[0, 0], [0, 0]], [[0, 0], [0, 0]])

    def test_each_call_builds_fresh_tables(self) -> None:
        first_m, first_s = mcm_bottom_up(CLRS_P)
        first_m[1][6] = -1
        first_s[1][6] = -1
        second_m, second_s = mcm_bottom_up(CLRS_P)
        assert second_m[1][6] == 15125 and second_s[1][6] == 3


# ----------------------------------------------------------------------
# Rendering the order from s
# ----------------------------------------------------------------------
class TestOptimalParenthesization:
    """PRINT-OPTIMAL-PARENS as a string, and its argument checks.

    The module walks an explicit stack rather than recursing, so that a
    maximally lopsided chain cannot hit the interpreter's recursion limit.
    That is checked on hand-built split tables 3000 matrices long, three
    times CPython's default limit of 1000. Only the cells the walk reads
    are filled, so each row is a dict rather than a 3001-wide list.
    """

    DEEP_N: int = 3000

    def test_a_left_deep_chain_far_past_the_recursion_limit(self) -> None:
        n = self.DEEP_N
        s: List[Dict[int, int]] = [{} for _ in range(n + 1)]
        for j in range(2, n + 1):
            s[1][j] = j - 1  # always peel the last matrix off: ((A1A2)A3)...
        expected = "(" * (n - 1) + "A1" + "".join(f"A{k})" for k in range(2, n + 1))
        assert optimal_parenthesization(s, 1, n) == expected

    def test_a_right_deep_chain_far_past_the_recursion_limit(self) -> None:
        n = self.DEEP_N
        s: List[Dict[int, int]] = [{} for _ in range(n + 1)]
        for i in range(1, n):
            s[i][n] = i  # always peel the first matrix off: (A1(A2(...)))
        expected = "".join(f"(A{k}" for k in range(1, n)) + f"A{n}" + ")" * (n - 1)
        assert optimal_parenthesization(s, 1, n) == expected

    def test_the_output_has_no_spaces(self) -> None:
        _, s = mcm_bottom_up(CLRS_P)
        assert " " not in optimal_parenthesization(s, 1, 6)

    @pytest.mark.parametrize(
        "i, j",
        [(0, 6), (1, 7), (3, 2), (0, 0), (7, 7), (-1, 6)],
        ids=["i-zero", "j-past-n", "i-after-j", "both-zero", "both-past-n", "negative"],
    )
    def test_an_interval_outside_the_table_is_a_value_error(
        self, i: int, j: int
    ) -> None:
        _, s = mcm_bottom_up(CLRS_P)
        with pytest.raises(ValueError, match="need 1 <= i <= j <= 6"):
            optimal_parenthesization(s, i, j)

    @pytest.mark.parametrize(
        "i, j, name",
        [
            (1.0, 6, "i"),
            (1, 6.0, "j"),
            (True, 6, "i"),
            (1, True, "j"),
            (None, 6, "i"),
            ("1", 6, "i"),
        ],
        ids=["float-i", "float-j", "bool-i", "bool-j", "none-i", "str-i"],
    )
    def test_a_non_int_index_is_a_type_error(
        self, i: object, j: object, name: str
    ) -> None:
        _, s = mcm_bottom_up(CLRS_P)
        with pytest.raises(TypeError, match=f"{name} must be an int"):
            optimal_parenthesization(s, i, j)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# The bonus check: multiply real matrices in the returned order
# ----------------------------------------------------------------------
class TestNumpyProductCheck:
    """The returned order computes the same product as left to right.

    Matrix multiplication is associative, so every *valid* order of
    ``A1..An`` gives the same product. Multiplying seeded integer matrices
    in the returned order and comparing with ``A1 @ A2 @ ... @ An`` taken
    strictly left to right therefore proves the string is a reordering of
    the same product: every matrix used once, in order, with neighbouring
    shapes that chain. The entries are small integers stored as ``int64``,
    so the comparison is exact as well as close. Along the way the
    evaluator counts the scalar multiplications numpy actually performed,
    which must equal the reported optimum.
    """

    @staticmethod
    def left_to_right(matrices: Sequence[np.ndarray]) -> np.ndarray:
        return functools.reduce(np.matmul, matrices)

    def check(self, p: Sequence[int], seed: int) -> None:
        cost, expr = matrix_chain_order(p)
        matrices = chain_matrices(p, seed)
        ordered, counted, products = multiply_in_written_order(expr, matrices)
        reference = self.left_to_right(matrices)

        assert ordered.shape == reference.shape == (p[0], p[-1])
        assert np.allclose(ordered, reference)
        assert np.array_equal(ordered, reference)
        assert counted == cost
        assert products == len(p) - 2

    @pytest.mark.parametrize("p, cost, expr", ANCHOR_PARAMS)
    def test_every_anchor_order_computes_the_same_product(
        self, p: List[int], cost: int, expr: str
    ) -> None:
        self.check(p, seed=11)

    @pytest.mark.parametrize("n", CHAIN_LENGTHS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_seeded_chains_compute_the_same_product(self, n: int, seed: int) -> None:
        # Small dimensions keep the matrices tiny; 1s give vectors and scalars.
        self.check(make_chain("narrow", n, seed), seed=seed)

    def test_the_clrs_products_counted_by_numpy_are_15125(self) -> None:
        matrices = chain_matrices(CLRS_P, seed=3)
        _, counted, _ = multiply_in_written_order("((A1(A2A3))((A4A5)A6))", matrices)
        _, naive, _ = multiply_in_written_order("(((((A1A2)A3)A4)A5)A6)", matrices)
        assert (counted, naive) == (15125, 40500)

    def test_the_comparison_has_teeth(self) -> None:
        """Control: a genuine reordering of the factors gives a different product.

        With square matrices every order of the factors chains, so only the
        values can tell ``A1 A2 A3`` from ``A1 A3 A2``. They differ for this
        seeded draw, which shows the equality above is not automatic.
        """
        a1, a2, a3 = chain_matrices([3, 3, 3, 3], seed=5)
        assert not np.array_equal(a1 @ a2 @ a3, a1 @ a3 @ a2)


# ----------------------------------------------------------------------
# The pricing helper refuses anything that is not A1..An in order
# ----------------------------------------------------------------------
class TestParenthesizationCostRejects:
    """Malformed or out-of-order strings raise, so the cost check cannot be fooled.

    The spec accepts a tied answer by checking its price. That only works
    if a string that skips, repeats or reorders a matrix, or that is not a
    valid expression at all, is refused rather than priced.
    """

    P3: List[int] = [10, 100, 5, 50]

    @pytest.mark.parametrize(
        "expr",
        [
            "",
            "()",
            "A1",
            "(A1A2)",
            "((A1A2)A3",
            "((A1A2)A3))",
            "(A1A2)A3",
            "A1A2A3",
            "(A1A2A3)",
            "((A1A3)A2)",
            "((A2A1)A3)",
            "(A3(A2A1))",
            "((A1A1)A2)",
            "((A1A2)A2)",
            "((A1A2)A4)",
            "((A0A1)A2)",
            "((A01A2)A3)",
            "((AA2)A3)",
            "((a1A2)A3)",
            "((B1A2)A3)",
            "((A1 A2)A3)",
            "((A1,A2)A3)",
            " ((A1A2)A3)",
            "((A1A2)A3) ",
            "((A1A2)A3]",
            "((A1A2)(A3))",
            "(((A1A2)A3))",
            "((A1A2)A3)((A1A2)A3)",
        ],
        ids=[
            "empty",
            "empty-parens",
            "too-few",
            "stops-early",
            "unclosed",
            "extra-close",
            "outer-missing",
            "no-parens",
            "three-in-one-product",
            "out-of-order",
            "swapped",
            "reversed",
            "repeated",
            "repeated-last",
            "out-of-range",
            "a-zero",
            "leading-zero",
            "no-digits",
            "lowercase",
            "wrong-letter",
            "space-inside",
            "comma",
            "leading-space",
            "trailing-space",
            "wrong-bracket",
            "redundant-leaf-parens",
            "redundant-outer-parens",
            "two-expressions",
        ],
    )
    def test_a_malformed_or_out_of_order_string_is_a_value_error(
        self, expr: str
    ) -> None:
        with pytest.raises(ValueError):
            parenthesization_cost(self.P3, expr)

    @pytest.mark.parametrize("expr", ["(A1)", "A2", "A1A1", "(A1A1)"])
    def test_a_single_matrix_only_accepts_a1(self, expr: str) -> None:
        assert parenthesization_cost([10, 20], "A1") == 0
        with pytest.raises(ValueError):
            parenthesization_cost([10, 20], expr)

    @pytest.mark.parametrize("expr", [None, 123, ["(", "A1", "A2", ")"]])
    def test_a_non_string_is_a_type_error(self, expr: object) -> None:
        with pytest.raises(TypeError, match="expr must be a str"):
            parenthesization_cost([10, 20, 30], expr)  # type: ignore[arg-type]

    @pytest.mark.parametrize("p", [[10], [10, 0, 5], [10, -2, 5]])
    def test_a_bad_dimension_list_is_a_value_error(self, p: List[int]) -> None:
        with pytest.raises(ValueError):
            parenthesization_cost(p, "(A1A2)")


# ----------------------------------------------------------------------
# Contracts
# ----------------------------------------------------------------------
class TestContracts:
    """Every entry point validates ``p`` the same way, before any work.

    Fewer than two dimensions is a ``ValueError`` (there is no matrix),
    a zero or negative dimension is a ``ValueError`` (a dimension of 0 is
    not a matrix), and anything that is not an ``int``, ``bool`` included,
    is a ``TypeError``.
    """

    @pytest.mark.parametrize("entry", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize("p", [[], [7]], ids=["empty", "one-entry"])
    def test_fewer_than_two_dimensions_is_a_value_error(
        self, entry: Callable, p: List[int]
    ) -> None:
        with pytest.raises(ValueError, match="at least 2 entries"):
            entry(p)

    @pytest.mark.parametrize("entry", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize(
        "p, index",
        [
            ([10, 0, 5], 1),
            ([10, -3, 5], 1),
            ([0, 5], 0),
            ([5, -1], 1),
            ([4, 4, 4, -100], 3),
        ],
        ids=[
            "zero-middle",
            "negative-middle",
            "zero-first",
            "negative-last",
            "negative-deep",
        ],
    )
    def test_a_non_positive_dimension_is_a_value_error(
        self, entry: Callable, p: List[int], index: int
    ) -> None:
        with pytest.raises(ValueError, match=rf"p\[{index}\] must be > 0"):
            entry(p)

    @pytest.mark.parametrize("entry", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize(
        "bad, type_name",
        [
            (2.5, "float"),
            (2.0, "float"),
            ("3", "str"),
            (None, "NoneType"),
            ([3], "list"),
        ],
        ids=["float", "integral-float", "str", "none", "list"],
    )
    def test_a_non_int_dimension_is_a_type_error(
        self, entry: Callable, bad: object, type_name: str
    ) -> None:
        with pytest.raises(TypeError, match=rf"p\[1\] must be an int, got {type_name}"):
            entry([10, bad, 5])

    @pytest.mark.parametrize("entry", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize("flag", [True, False])
    def test_a_bool_dimension_is_a_type_error(
        self, entry: Callable, flag: bool
    ) -> None:
        """``True == 1`` in Python, but a truth value is not a matrix dimension."""
        with pytest.raises(TypeError, match="got bool"):
            entry([10, flag, 5])
        with pytest.raises(TypeError, match="got bool"):
            entry([flag, 5])

    @pytest.mark.parametrize("entry", ENTRY_PARAMS, ids=ENTRY_IDS)
    @pytest.mark.parametrize("p", [7, None, 3.5], ids=["int", "none", "float"])
    def test_a_non_sequence_is_a_type_error(self, entry: Callable, p: object) -> None:
        with pytest.raises(TypeError, match="p must be a sequence of ints"):
            entry(p)

    @pytest.mark.parametrize("entry", ENTRY_PARAMS, ids=ENTRY_IDS)
    def test_the_caller_s_list_is_never_modified(self, entry: Callable) -> None:
        p = list(CLRS_P)
        entry(p)
        assert p == CLRS_P


# ----------------------------------------------------------------------
# The assignment requirement: an explicit memo, not functools.lru_cache
# ----------------------------------------------------------------------
class TestMemoIsExplicit:
    """``mcm_memoized`` keeps its own table; the memo is the graded idea.

    The module's docstrings mention ``functools.lru_cache`` in prose, where
    they explain why it is *not* used, so a substring search over the
    source would fail on the documentation. The source is parsed instead
    and only decorators, imports and names in code are inspected. The
    behavioural half counts calls: the plain recursion makes exactly
    ``3^(n-1)`` calls, the memoized one solves each of the ``n(n+1)/2``
    intervals exactly once, and a fresh memo is built on every call.
    """

    MODULE_TREE = ast.parse(inspect.getsource(mcm_module))

    @pytest.mark.parametrize(
        "function", [mcm_memoized, mcm_module._memoized], ids=["public", "worker"]
    )
    def test_the_memoized_functions_carry_no_decorator(
        self, function: Callable
    ) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        definition = tree.body[0]
        assert isinstance(definition, ast.FunctionDef)
        assert definition.decorator_list == []

    def test_no_function_in_the_module_is_decorated(self) -> None:
        for node in ast.walk(self.MODULE_TREE):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.decorator_list == [], f"{node.name} is decorated"

    def test_functools_is_not_imported(self) -> None:
        for node in ast.walk(self.MODULE_TREE):
            if isinstance(node, ast.Import):
                assert "functools" not in [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "functools"
        assert not hasattr(mcm_module, "functools")
        assert not hasattr(mcm_module, "lru_cache")

    def test_no_cache_name_appears_in_code(self) -> None:
        cache_names = {"lru_cache", "cache", "cached_property"}
        for node in ast.walk(self.MODULE_TREE):
            if isinstance(node, ast.Name):
                assert node.id not in cache_names, f"line {node.lineno}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in cache_names, f"line {node.lineno}"

    @pytest.mark.parametrize(
        "function", [mcm_memoized, mcm_module._memoized], ids=["public", "worker"]
    )
    def test_no_lru_cache_wrapper_is_attached(self, function: Callable) -> None:
        assert not hasattr(function, "cache_info")
        assert not hasattr(function, "cache_clear")
        assert not hasattr(function, "__wrapped__")

    def test_interleaved_chains_never_reuse_each_other_s_answers(self) -> None:
        """Two chains of four matrices with different optima, alternated.

        A cache that survived between calls and was keyed by ``(i, j)``
        alone would hand the second chain the first chain's 30000.
        """
        for _ in range(2):
            assert mcm_memoized([10, 20, 30, 40, 30]) == 30000
            assert mcm_memoized([40, 20, 30, 10, 30]) == 26000
            assert mcm_memoized(CLRS_P) == 15125
            assert mcm_memoized([10, 20, 30]) == 6000

    @pytest.mark.parametrize(
        "n, calls", [(1, 1), (2, 3), (3, 9), (4, 27), (6, 243), (9, 6561)]
    )
    def test_the_plain_recursion_makes_exactly_3_to_the_n_minus_1_calls(
        self, monkeypatch: pytest.MonkeyPatch, n: int, calls: int
    ) -> None:
        """T(1) = 1 and T(L) = 1 + 2 * (T(1) + ... + T(L-1)), so 1, 3, 9, 27, ..."""
        count = 0
        original = mcm_module._recursive

        def counting(p: List[int], i: int, j: int) -> int:
            nonlocal count
            count += 1
            return original(p, i, j)

        monkeypatch.setattr(mcm_module, "_recursive", counting)
        mcm_recursive([3] * (n + 1))
        assert count == calls

    @pytest.mark.parametrize("n", CHAIN_LENGTHS)
    def test_the_memo_solves_every_interval_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, n: int
    ) -> None:
        """Each of the n(n+1)/2 intervals is computed once, every later visit is a hit.

        An interval of length L >= 2 makes 2(L - 1) calls when it is
        solved, and there are n - L + 1 of them, so with the root call the
        total is 1 + 2 * C(n + 1, 3) = 1 + (n + 1) n (n - 1) / 3:
        1, 3, 9, 21, 41, ... rather than the recursion's 3^(n-1).
        """
        solved: List[Tuple[int, int]] = []
        count = 0
        original = mcm_module._memoized

        def counting(p: List[int], i: int, j: int, memo: List[List[object]]) -> int:
            nonlocal count
            count += 1
            if memo[i][j] is None:
                solved.append((i, j))
            return original(p, i, j, memo)

        monkeypatch.setattr(mcm_module, "_memoized", counting)
        p = make_chain("wide", n, 0)
        assert mcm_memoized(p) == brute_force_min(p)

        every_interval = {(i, j) for i in range(1, n + 1) for j in range(i, n + 1)}
        assert len(solved) == len(set(solved)) == n * (n + 1) // 2
        assert set(solved) == every_interval
        assert count == 1 + (n + 1) * n * (n - 1) // 3

    def test_each_call_starts_from_an_empty_memo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeating a call repeats all the work: nothing was kept from the first."""
        solved_per_call: List[int] = []
        original = mcm_module._memoized
        solved = 0

        def counting(p: List[int], i: int, j: int, memo: List[List[object]]) -> int:
            nonlocal solved
            if memo[i][j] is None:
                solved += 1
            return original(p, i, j, memo)

        monkeypatch.setattr(mcm_module, "_memoized", counting)
        for _ in range(3):
            solved = 0
            assert mcm_memoized(CLRS_P) == 15125
            solved_per_call.append(solved)
        assert solved_per_call == [21, 21, 21]
