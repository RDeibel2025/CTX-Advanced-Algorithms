"""Tests for the SAT solvers: plain backtracking, DPLL, WalkSAT and the random 3-SAT generator.

A SAT solver is easy to get almost right. A prune that fires one variable
too early discards real solutions, an undo that forgets one counter makes
the next branch see a clause as satisfied when it is not, and a "fill the
rest with False" step applied to an assignment that does not yet satisfy
every clause returns something that looks complete and is wrong. Each of
those still returns plausible output, so every claim here rests on a
witness that shares no mechanism with the module:

* **The spec anchors (WEEK8_BUILD_SPEC.md section 10.3), checked by hand.**
  The Sipser formula ``[[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]`` is
  ``(x1 or x2) and (not x1 or not x2) and (not x1 or x2)``. With x1 = True
  the last two clauses demand x2 = False and x2 = True, so x1 = False, and
  then the first clause demands x2 = True: the only satisfying assignment
  is **{1: False, 2: True}**. The eight clauses with every sign pattern over
  x1, x2, x3 forbid all eight assignments (each assignment falsifies the
  clause with the opposite signs), so that formula is **unsatisfiable**, as
  is ``[[1], [-1]]``.
* **Node counts traced by hand** on those anchors and a few more, branch by
  branch, in :class:`TestHandTracedNodeCounts`, using the node definition
  in the module docstring: the root plus every branching assignment.
* **Brute force over every assignment** with :mod:`itertools.product`,
  which knows nothing about search order, pruning, propagation or
  counters, on 100 seeded random 3-SAT formulas with n <= 10 at ratios
  2.0 to 6.0 (as section 10.3 requires) and on a battery of mixed-width
  formulas with repeated literals and tautologies.
* **Planted formulas.** Clauses are drawn at random and kept only if a
  hidden assignment satisfies them, so the formula is satisfiable by
  construction, at sizes where brute force would be too slow.
* **The contract itself.** :func:`random_3sat` is replayed step by step
  from ``make_rng(seed)`` as the contract describes it.

Satisfaction is always checked with the local evaluator :func:`satisfies`,
never only with the module's own verifier. Every random draw is seeded
through ``make_rng`` / ``derive_seed``, so nothing here can flake. No
assertion is about time.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import copy
import itertools
from typing import Callable, Dict, List, Tuple

import pytest

from src.complexity import reduction_demo
from src.complexity.satisfiability_solver import (
    is_satisfying,
    random_3sat,
    sat_backtracking,
    sat_dpll,
    walksat,
)
from src.utils.random_utils import derive_seed, make_rng

CNF = List[List[int]]

#: Master seed for every seeded battery in this file.
MASTER = 2026

#: The Sipser anchor and its unique satisfying assignment (worked by hand
#: in the module docstring above).
SIPSER: CNF = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]
SIPSER_SOLUTION: Dict[int, bool] = {1: False, 2: True}

#: All eight sign combinations over x1, x2, x3: unsatisfiable.
EIGHT: CNF = [
    [s1 * 1, s2 * 2, s3 * 3] for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)
]

#: The two contradictory unit clauses: unsatisfiable.
CONTRADICTION: CNF = [[1], [-1]]

#: Both complete solvers, for parametrising.
COMPLETE_SOLVERS = [
    pytest.param(sat_backtracking, id="backtracking"),
    pytest.param(sat_dpll, id="dpll"),
]


def satisfies(cnf: CNF, assignment: Dict[int, bool]) -> bool:
    """Return True if every clause has a literal made true by ``assignment``.

    Written independently of the module: a literal ``l`` is true when the
    variable ``abs(l)`` has the value ``l > 0``.
    """
    return all(
        any(assignment[abs(literal)] == (literal > 0) for literal in clause)
        for clause in cnf
    )


def brute_force_solutions(cnf: CNF, n_vars: int) -> List[Dict[int, bool]]:
    """Every satisfying assignment of ``cnf``, found by trying all ``2^n_vars``."""
    found = []
    for bits in itertools.product((False, True), repeat=n_vars):
        assignment = {v: bits[v - 1] for v in range(1, n_vars + 1)}
        if satisfies(cnf, assignment):
            found.append(assignment)
    return found


def brute_force_satisfiable(cnf: CNF, n_vars: int) -> bool:
    """True if some assignment satisfies ``cnf``; stops at the first one."""
    for bits in itertools.product((False, True), repeat=n_vars):
        assignment = {v: bits[v - 1] for v in range(1, n_vars + 1)}
        if satisfies(cnf, assignment):
            return True
    return False


def planted_formula(n_vars: int, n_clauses: int, seed: int) -> Tuple[CNF, Dict[int, bool]]:
    """A random 3-SAT formula satisfied by a hidden assignment, by construction.

    Clauses are drawn like random 3-SAT and kept only if the hidden
    assignment makes one of their literals true.
    """
    rng = make_rng(seed)
    hidden = {v: rng.random() < 0.5 for v in range(1, n_vars + 1)}
    formula: CNF = []
    while len(formula) < n_clauses:
        chosen = rng.sample(range(1, n_vars + 1), 3)
        clause = [v if rng.random() < 0.5 else -v for v in chosen]
        if any(hidden[abs(literal)] == (literal > 0) for literal in clause):
            formula.append(clause)
    return formula, hidden


def assert_complete_assignment(result: object, n_vars: int) -> None:
    """The result names every variable 1..n_vars, each with a real bool."""
    assert isinstance(result, dict)
    assert sorted(result) == list(range(1, n_vars + 1))
    assert all(type(value) is bool for value in result.values())


def random_battery() -> List[Tuple[int, CNF]]:
    """100 seeded random 3-SAT formulas, n in 3..10, ratio 2.0..6.0.

    Formula ``i`` has ``n = 3 + i % 8`` variables and ratio
    ``2.0 + 0.5 * (i % 9)``; 8 and 9 are coprime, so the 100 formulas sweep
    every (n, ratio) pair.
    """
    battery = []
    for i in range(100):
        n = 3 + i % 8
        ratio = 2.0 + 0.5 * (i % 9)
        m = round(ratio * n)
        battery.append((n, random_3sat(n, m, derive_seed(MASTER, "random-3sat", i))))
    return battery


def mixed_battery() -> List[Tuple[int, CNF]]:
    """150 seeded formulas with clauses of 1 to 4 literals over n in 1..8.

    Literals are drawn independently, so clauses can repeat a literal or
    hold both ``x`` and ``-x``; unit clauses exercise DPLL's propagation
    from the root, which random 3-SAT never does.
    """
    battery = []
    for i in range(150):
        rng = make_rng(derive_seed(MASTER, "mixed", i))
        n = rng.randint(1, 8)
        m = rng.randint(1, 4 * n)
        formula = []
        for _ in range(m):
            width = rng.randint(1, 4)
            formula.append(
                [rng.choice((1, -1)) * rng.randint(1, n) for _ in range(width)]
            )
        battery.append((n, formula))
    return battery


RANDOM_BATTERY = random_battery()
MIXED_BATTERY = mixed_battery()


# ----------------------------------------------------------------------
# The spec anchors (section 10.3)
# ----------------------------------------------------------------------
class TestSpecAnchors:
    """The three hand-checked formulas of section 10.3."""

    def test_sipser_anchor_has_exactly_one_solution(self) -> None:
        """Brute force confirms the hand argument: only x1 = False, x2 = True."""
        assert brute_force_solutions(SIPSER, 2) == [SIPSER_SOLUTION]

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_sipser_anchor(self, solver: Callable) -> None:
        assert solver(SIPSER, 2) == {1: False, 2: True}

    def test_eight_clause_anchor_is_unsatisfiable_by_brute_force(self) -> None:
        assert len(EIGHT) == 8
        assert brute_force_solutions(EIGHT, 3) == []

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_eight_clause_anchor(self, solver: Callable) -> None:
        assert solver(EIGHT, 3) is None

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_contradictory_units(self, solver: Callable) -> None:
        assert solver(CONTRADICTION, 1) is None

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
    def test_walksat_finds_the_sipser_solution(self, seed: int) -> None:
        """Any satisfying assignment WalkSAT returns must be the unique one."""
        assert walksat(SIPSER, 2, seed=seed) == SIPSER_SOLUTION

    @pytest.mark.parametrize("formula, n_vars", [(EIGHT, 3), (CONTRADICTION, 1)])
    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_walksat_cannot_return_on_unsatisfiable_anchors(
        self, formula: CNF, n_vars: int, seed: int
    ) -> None:
        """WalkSAT only returns satisfying assignments, and these have none."""
        assert walksat(formula, n_vars, max_flips=500, seed=seed) is None

    def test_is_satisfying_is_the_reduction_demo_verifier(self) -> None:
        """The re-export is the very same function, not a second copy."""
        assert is_satisfying is reduction_demo.is_satisfying
        assert is_satisfying(SIPSER, SIPSER_SOLUTION) is True
        assert is_satisfying(SIPSER, {1: True, 2: True}) is False


# ----------------------------------------------------------------------
# Node counts traced by hand
# ----------------------------------------------------------------------
class TestHandTracedNodeCounts:
    """Node counts worked out by hand from the documented definitions.

    Both solvers count the root plus every branching assignment (the True
    child and the False child each). For plain backtracking every
    assignment branches; DPLL does not count values forced by unit
    propagation or set by the pure-literal rule.
    """

    def test_sipser_backtracking(self) -> None:
        """root; x1=T; x2=T cuts (not x1 or not x2); x2=F cuts (not x1 or x2);
        x1=F; x2=T satisfies all three clauses. 1 + 5 = 6 nodes."""
        assert sat_backtracking(SIPSER, 2, count_nodes=True) == ({1: False, 2: True}, 6)

    def test_sipser_dpll(self) -> None:
        """root (no unit, no pure literal); x1=T makes (not x2) and (x2) unit:
        conflict; x1=F makes (x2) unit, propagation sets x2=T. 1 + 2 = 3."""
        assert sat_dpll(SIPSER, 2, count_nodes=True) == ({1: False, 2: True}, 3)

    def test_eight_clause_backtracking_visits_the_full_tree(self) -> None:
        """Every clause names x3, so nothing is cut above depth 3, and every
        full assignment falsifies one clause: the whole tree, 2^4 - 1 = 15."""
        assert sat_backtracking(EIGHT, 3, count_nodes=True) == (None, 15)

    def test_eight_clause_dpll(self) -> None:
        """root; x1=T leaves the four 2-clauses over x2, x3; x2=T makes (x3)
        and (not x3) unit: conflict; x2=F likewise; x1=F mirrors it.
        1 + 2 + 2 + 2 = 7."""
        assert sat_dpll(EIGHT, 3, count_nodes=True) == (None, 7)

    def test_dpll_never_exceeds_backtracking_on_the_eight_clause_anchor(self) -> None:
        """True by construction: 15 = 2^(3+1) - 1 is the most any search tree
        over three variables can have, and plain backtracking reaches it."""
        _, dpll_nodes = sat_dpll(EIGHT, 3, count_nodes=True)
        _, backtracking_nodes = sat_backtracking(EIGHT, 3, count_nodes=True)
        assert backtracking_nodes == 2 ** 4 - 1
        assert dpll_nodes <= backtracking_nodes

    def test_contradiction_backtracking(self) -> None:
        """root; x1=T cuts (not x1); x1=F cuts (x1). 3 nodes."""
        assert sat_backtracking(CONTRADICTION, 1, count_nodes=True) == (None, 3)

    def test_contradiction_dpll_is_refuted_at_the_root(self) -> None:
        """Both clauses are unit at the root; propagating one falsifies the
        other before any decision. 1 node."""
        assert sat_dpll(CONTRADICTION, 1, count_nodes=True) == (None, 1)

    def test_pure_literal_needs_no_search(self) -> None:
        """In (not x1 or x2) and (not x1 or not x2), x1 occurs only negated.
        DPLL sets x1=F by the pure-literal rule, which satisfies both
        clauses: root only, x2 filled with False. Backtracking tries x1=T,
        then x2=T and x2=F are each cut, then x1=F satisfies both: 5."""
        formula = [[-1, 2], [-1, -2]]
        assert sat_dpll(formula, 2, count_nodes=True) == ({1: False, 2: False}, 1)
        assert sat_backtracking(formula, 2, count_nodes=True) == ({1: False, 2: False}, 5)

    def test_unused_variables_are_filled_with_false(self) -> None:
        """(x1) with three variables. Backtracking: root, x1=T satisfies the
        only clause, stop: 2 nodes. DPLL: (x1) is unit at the root: 1 node.
        x2 and x3 are never decided and come back False."""
        expected = {1: True, 2: False, 3: False}
        assert sat_backtracking([[1]], 3, count_nodes=True) == (expected, 2)
        assert sat_dpll([[1]], 3, count_nodes=True) == (expected, 1)

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_empty_formula_is_satisfied_at_the_root(self, solver: Callable) -> None:
        assert solver([], 2, count_nodes=True) == ({1: False, 2: False}, 1)
        assert solver([], 0, count_nodes=True) == ({}, 1)

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_count_nodes_false_returns_the_bare_result(self, solver: Callable) -> None:
        assert solver(SIPSER, 2) == SIPSER_SOLUTION
        assert solver(EIGHT, 3) is None


# ----------------------------------------------------------------------
# Agreement with brute force
# ----------------------------------------------------------------------
class TestAgreementWithBruteForce:
    """Section 10.3's battery, plus a mixed-width one."""

    def test_random_battery_is_not_one_sided(self) -> None:
        """The battery holds both answers, so agreement is not vacuous."""
        answers = [brute_force_satisfiable(cnf, n) for n, cnf in RANDOM_BATTERY]
        assert len(answers) == 100
        assert any(answers) and not all(answers)

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_random_3sat_battery(self, solver: Callable) -> None:
        for n, cnf in RANDOM_BATTERY:
            expected = brute_force_satisfiable(cnf, n)
            result, nodes = solver(cnf, n, count_nodes=True)
            assert (result is not None) == expected, (n, cnf)
            if result is not None:
                assert_complete_assignment(result, n)
                assert satisfies(cnf, result), (n, cnf, result)
                assert is_satisfying(cnf, result)
            assert type(nodes) is int
            assert 1 <= nodes <= 2 ** (n + 1) - 1

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    def test_mixed_width_battery(self, solver: Callable) -> None:
        answers = []
        for n, cnf in MIXED_BATTERY:
            expected = brute_force_satisfiable(cnf, n)
            answers.append(expected)
            result, nodes = solver(cnf, n, count_nodes=True)
            assert (result is not None) == expected, (n, cnf)
            if result is not None:
                assert_complete_assignment(result, n)
                assert satisfies(cnf, result), (n, cnf, result)
            assert type(nodes) is int and nodes >= 1
        assert any(answers) and not all(answers)

    def test_walksat_on_the_random_battery(self) -> None:
        """When WalkSAT returns, the assignment satisfies; on an
        unsatisfiable formula it never returns."""
        for i, (n, cnf) in enumerate(RANDOM_BATTERY):
            result = walksat(cnf, n, max_flips=300, seed=derive_seed(MASTER, "walk", i))
            if result is not None:
                assert_complete_assignment(result, n)
                assert satisfies(cnf, result), (n, cnf, result)
            if not brute_force_satisfiable(cnf, n):
                assert result is None

    def test_walksat_on_the_mixed_battery(self) -> None:
        for i, (n, cnf) in enumerate(MIXED_BATTERY):
            result = walksat(cnf, n, max_flips=200, seed=derive_seed(MASTER, "walk-mixed", i))
            if result is not None:
                assert_complete_assignment(result, n)
                assert satisfies(cnf, result), (n, cnf, result)
            if not brute_force_satisfiable(cnf, n):
                assert result is None


# ----------------------------------------------------------------------
# Planted (satisfiable by construction) formulas beyond brute force
# ----------------------------------------------------------------------
class TestPlantedFormulas:
    """Formulas too large for brute force whose satisfiability is known."""

    @pytest.mark.parametrize("solver", COMPLETE_SOLVERS)
    @pytest.mark.parametrize("rep", range(5))
    def test_complete_solvers_find_a_solution(self, solver: Callable, rep: int) -> None:
        n = 22
        cnf, _ = planted_formula(n, round(4.0 * n), derive_seed(MASTER, "planted", rep))
        result, nodes = solver(cnf, n, count_nodes=True)
        assert result is not None
        assert_complete_assignment(result, n)
        assert satisfies(cnf, result)
        assert type(nodes) is int and nodes >= 1

    @pytest.mark.parametrize("rep", range(10))
    def test_walksat_solves_easy_satisfiable_formulas(self, rep: int) -> None:
        """At ratio 2.0 a planted formula has a great many solutions, and
        local search with the default 10,000 flips finds one."""
        n = 40
        cnf, _ = planted_formula(n, 2 * n, derive_seed(MASTER, "easy", rep))
        result = walksat(cnf, n, seed=derive_seed(MASTER, "easy-walk", rep))
        assert result is not None
        assert_complete_assignment(result, n)
        assert satisfies(cnf, result)

    def test_planted_formula_helper_is_satisfied_by_its_hidden_assignment(self) -> None:
        cnf, hidden = planted_formula(12, 50, derive_seed(MASTER, "helper"))
        assert len(cnf) == 50
        assert satisfies(cnf, hidden)


# ----------------------------------------------------------------------
# WalkSAT behaviour and parameters
# ----------------------------------------------------------------------
class TestWalkSAT:
    """Reproducibility, the extreme noise settings and parameter checks."""

    def test_seeded_runs_repeat(self) -> None:
        cnf, _ = planted_formula(30, 120, derive_seed(MASTER, "repeat"))
        first = walksat(cnf, 30, seed=11)
        assert first is not None
        assert walksat(cnf, 30, seed=11) == first

    @pytest.mark.parametrize("p", [0, 0.0, 0.25, 1, 1.0])
    def test_any_noise_level_only_returns_satisfying_assignments(self, p: float) -> None:
        for i, (n, cnf) in enumerate(RANDOM_BATTERY[:40]):
            result = walksat(cnf, n, max_flips=200, p=p, seed=i)
            if result is not None:
                assert satisfies(cnf, result)

    def test_greedy_move_flips_the_lowest_break_variable(self) -> None:
        """Traced by hand on F = (x1 or x2) and (x1 or not x2), with p = 0 so
        every move is greedy. F holds iff x1 is True. From x1=F, x2=F only
        clause 1 is unsatisfied: flipping x1 breaks nothing (no clause holds
        not x1), flipping x2 breaks clause 2, so x1 is flipped and F holds.
        From x1=F, x2=T only clause 2 is unsatisfied: x1 breaks nothing, x2
        breaks clause 1, so again x1 is flipped. So one flip solves F from
        every start, and the flip with the HIGHEST break count never would.
        """
        formula = [[1, 2], [1, -2]]
        seeds = range(20)
        for seed in seeds:
            result = walksat(formula, 2, max_flips=1, p=0, seed=seed)
            assert result is not None and result[1] is True
        # Some of these seeds start from an unsatisfying assignment, so the
        # greedy move really was exercised above.
        assert any(walksat(formula, 2, max_flips=0, seed=seed) is None for seed in seeds)

    def test_pure_random_walk_solves_the_sipser_anchor(self) -> None:
        assert walksat(SIPSER, 2, p=1.0, seed=3) == SIPSER_SOLUTION

    def test_zero_flips_only_checks_the_start(self) -> None:
        """With no flips the random start is returned only if it satisfies."""
        assert walksat([], 3, max_flips=0, seed=5) is not None
        assert_complete_assignment(walksat([], 3, max_flips=0, seed=5), 3)
        assert walksat(CONTRADICTION, 1, max_flips=0, seed=5) is None
        for seed in range(20):
            result = walksat(SIPSER, 2, max_flips=0, seed=seed)
            assert result is None or result == SIPSER_SOLUTION

    def test_empty_formula_with_no_variables(self) -> None:
        assert walksat([], 0, seed=1) == {}

    @pytest.mark.parametrize("max_flips", [-1, -100])
    def test_negative_max_flips(self, max_flips: int) -> None:
        with pytest.raises(ValueError):
            walksat(SIPSER, 2, max_flips=max_flips, seed=1)

    @pytest.mark.parametrize("max_flips", [10.0, "10", True, None])
    def test_non_int_max_flips(self, max_flips: object) -> None:
        with pytest.raises(TypeError):
            walksat(SIPSER, 2, max_flips=max_flips, seed=1)

    @pytest.mark.parametrize("p", [-0.1, 1.01, 2, float("nan")])
    def test_p_out_of_range(self, p: float) -> None:
        with pytest.raises(ValueError):
            walksat(SIPSER, 2, p=p, seed=1)

    @pytest.mark.parametrize("p", ["0.5", True, None, [0.5]])
    def test_p_not_a_number(self, p: object) -> None:
        with pytest.raises(TypeError):
            walksat(SIPSER, 2, p=p, seed=1)

    def test_bad_seed(self) -> None:
        with pytest.raises(TypeError):
            walksat(SIPSER, 2, seed="1")


# ----------------------------------------------------------------------
# random_3sat
# ----------------------------------------------------------------------
class TestRandom3Sat:
    """The generator, checked against its contract."""

    def test_matches_the_contract_replayed_by_hand(self) -> None:
        """Replay the documented recipe with an independent generator."""
        for n, m, seed in [(3, 10, 0), (5, 3, 42), (20, 85, 7), (50, 213, 2026)]:
            rng = make_rng(seed)
            expected = []
            for _ in range(m):
                chosen = rng.sample(range(1, n + 1), 3)
                expected.append([v if rng.random() < 0.5 else -v for v in chosen])
            assert random_3sat(n, m, seed) == expected

    def test_shape(self) -> None:
        n, m = 12, 300
        formula = random_3sat(n, m, seed=9)
        assert len(formula) == m
        for clause in formula:
            assert len(clause) == 3
            assert all(type(literal) is int and literal != 0 for literal in clause)
            variables = [abs(literal) for literal in clause]
            assert len(set(variables)) == 3
            assert all(1 <= v <= n for v in variables)
        signs = {literal > 0 for clause in formula for literal in clause}
        assert signs == {True, False}

    def test_three_variables_use_all_three_in_every_clause(self) -> None:
        for clause in random_3sat(3, 50, seed=4):
            assert sorted(abs(literal) for literal in clause) == [1, 2, 3]

    def test_reproducible_and_seed_dependent(self) -> None:
        assert random_3sat(10, 42, seed=1) == random_3sat(10, 42, seed=1)
        assert random_3sat(10, 42, seed=1) != random_3sat(10, 42, seed=2)

    def test_zero_clauses(self) -> None:
        assert random_3sat(4, 0, seed=1) == []

    @pytest.mark.parametrize("n_vars", [0, 1, 2, -5])
    def test_too_few_variables(self, n_vars: int) -> None:
        with pytest.raises(ValueError):
            random_3sat(n_vars, 5, seed=1)

    def test_negative_clause_count(self) -> None:
        with pytest.raises(ValueError):
            random_3sat(5, -1, seed=1)

    @pytest.mark.parametrize(
        "args",
        [(5.0, 3, 1), ("5", 3, 1), (True, 3, 1), (5, 3.0, 1), (5, False, 1), (5, 3, "1")],
    )
    def test_type_errors(self, args: tuple) -> None:
        with pytest.raises(TypeError):
            random_3sat(*args)


# ----------------------------------------------------------------------
# Input validation, shared by all three solvers
# ----------------------------------------------------------------------
ALL_SOLVERS = [
    pytest.param(sat_backtracking, id="backtracking"),
    pytest.param(sat_dpll, id="dpll"),
    pytest.param(lambda cnf, n: walksat(cnf, n, seed=0), id="walksat"),
]


class TestInputValidation:
    """Every solver rejects malformed formulas the same way."""

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("cnf", ["[[1]]", None, 5, {1: [1]}, {(1,)}])
    def test_cnf_not_a_list(self, solver: Callable, cnf: object) -> None:
        with pytest.raises(TypeError):
            solver(cnf, 1)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("cnf", [[1, 2], [[1], 2], [{1, 2}], ["12"]])
    def test_clause_not_a_list(self, solver: Callable, cnf: object) -> None:
        with pytest.raises(TypeError):
            solver(cnf, 2)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("cnf", [[[1.0]], [[True]], [["1"]], [[1, None]]])
    def test_literal_not_an_int(self, solver: Callable, cnf: object) -> None:
        with pytest.raises(TypeError):
            solver(cnf, 2)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("cnf", [[[0]], [[1, 0, 2]], [[1], [-2, 0]]])
    def test_literal_zero(self, solver: Callable, cnf: CNF) -> None:
        with pytest.raises(ValueError):
            solver(cnf, 2)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("cnf", [[[3]], [[-3]], [[1, 2, 99]]])
    def test_variable_out_of_range(self, solver: Callable, cnf: CNF) -> None:
        with pytest.raises(ValueError):
            solver(cnf, 2)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("cnf", [[[]], [[1], []], [[1, 2], [], [-1]]])
    def test_empty_clause(self, solver: Callable, cnf: CNF) -> None:
        with pytest.raises(ValueError):
            solver(cnf, 2)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize("n_vars", [2.0, "2", True, None])
    def test_n_vars_not_an_int(self, solver: Callable, n_vars: object) -> None:
        with pytest.raises(TypeError):
            solver(SIPSER, n_vars)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    def test_negative_n_vars(self, solver: Callable) -> None:
        with pytest.raises(ValueError):
            solver([], -1)

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    def test_input_is_not_modified(self, solver: Callable) -> None:
        formula = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2], [2, -2]]
        before = copy.deepcopy(formula)
        solver(formula, 2)
        assert formula == before

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    def test_tuples_and_tautologies_are_accepted(self, solver: Callable) -> None:
        """Tuples are read like lists, and a clause holding x2 and not x2
        constrains nothing, so the answer is still the Sipser solution."""
        formula = ((1, 1, 2), (-1, -2, -2), (-1, 2, 2), (2, -2, 1))
        assert solver(formula, 2) == SIPSER_SOLUTION
