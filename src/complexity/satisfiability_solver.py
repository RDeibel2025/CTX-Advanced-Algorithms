"""Satisfiability: plain backtracking, DPLL and WalkSAT on CNF formulas (Week 8, Part 2b).

The problem
-----------
A formula in conjunctive normal form (CNF) is an AND of clauses, each an OR
of literals. Here it is a ``list[list[int]]`` in DIMACS style: variables are
numbered from 1, the literal ``i`` means "variable i is true" and ``-i``
means "variable i is false". SAT asks whether some assignment of truth
values satisfies every clause. Cook (1971) and Levin (1973) proved SAT
NP-complete, and Karp (1972) showed that 3-SAT, where every clause has three
literals, stays NP-complete; it is the source problem of the Week 8
reductions in :mod:`src.complexity.reduction_demo`.

SAT is in NP because a "yes" has a short certificate, the assignment, and
:func:`is_satisfying` checks it in one O(L) pass over the L literals. No
known algorithm *finds* a satisfying assignment, or proves there is none, in
time polynomial in the formula size. Every complete solver below explores a
search tree with up to ``2^(n + 1) - 1`` nodes on n variables. That gap
between checking (linear) and finding (exponential) is what the Week 8
benchmark measures.

Three solvers, three trade-offs
-------------------------------
* :func:`sat_backtracking` is the brute-force baseline. It walks the binary
  tree of partial assignments depth first, deciding variables 1, 2, ..., n
  in that fixed order and trying True before False. It cuts a branch only
  when some clause has every literal assigned and false, and it stops as
  soon as every clause contains a true literal. It never *deduces* a value.
  Because the order is fixed, a clause can be falsified only at the moment
  its highest-numbered variable is assigned, so each node checks only the
  clauses whose last variable it just decided.
* :func:`sat_dpll` is the Davis-Putnam-Logemann-Loveland procedure (Davis
  and Putnam, 1960; Davis, Logemann and Loveland, 1962): the same
  backtracking plus two inference rules applied before every branch.

  - **Unit propagation.** A clause whose literals are all false except one
    unassigned literal forces that literal to be true. Forcing it may make
    more clauses unit, so the rule runs to a fixed point, and a clause that
    becomes all-false is a conflict that ends the branch at once.
  - **Pure-literal elimination.** If a variable occurs in the clauses that
    are not yet satisfied with only one sign, setting it that way satisfies
    those clauses and falsifies nothing that matters, so it is safe. A
    variable that occurs in no unsatisfied clause at all is pure in the
    vacuous sense (every value is safe) and is set to False by the same
    rule, so the search never branches on a variable the remaining formula
    does not mention.

  Branching is on the lowest-numbered unassigned variable, True first.
  Inference prunes enormous parts of the tree in practice, but the worst
  case is still exponential: no inference rule turns an NP-complete search
  into a polynomial one unless P = NP.
* :func:`walksat` is the randomized local search of Selman, Kautz and Cohen
  (1994), a **heuristic**. It starts from a random assignment and repairs
  it one flip at a time. It can be dramatically faster than either complete
  solver on satisfiable formulas, but it is *incomplete*: it can never prove
  a formula unsatisfiable (on an unsatisfiable formula it simply runs out
  of flips and returns ``None``), and ``None`` on a satisfiable formula
  just means it did not find a solution in time.

All three return a **complete** assignment: a dict holding every variable
1..n_vars. A complete solver can stop with some variables undecided, once
every clause already has a true literal; those variables are filled with
False before returning. Every returned assignment satisfies the formula.

Search nodes, the hardware-independent cost
-------------------------------------------
With ``count_nodes=True`` the complete solvers also return how many search
nodes they visited. A node is a point in the binary search tree:

* the **root** (the empty assignment) counts 1, and
* every **branching assignment** counts 1, the True child and the False
  child each, including a child that is cut the moment it is created.

In :func:`sat_backtracking` every assignment is a branching assignment. In
:func:`sat_dpll` only the decisions are; values forced by unit propagation
or set by the pure-literal rule are inference, not search, and are not
counted. A search tree over n variables has at most ``2^(n + 1) - 1``
nodes, so both counts lie in ``[1, 2^(n + 1) - 1]``.

Random 3-SAT and the 4.26 threshold
-----------------------------------
:func:`random_3sat` draws each clause as three distinct variables with
independent random signs. With ``alpha = m / n`` clauses per variable,
random formulas are almost always satisfiable for alpha well below about
4.26 and almost always unsatisfiable well above it, and the cost of
complete solvers peaks near that ratio (Mitchell, Selman and Levesque,
1992; Mezard, Parisi and Zecchina, 2002, computed 4.267 by the cavity
method). Below it, solutions are plentiful and found early; above it,
contradictions surface quickly. At the threshold neither happens, which is
why the benchmark measures exponential growth at alpha = 4.26 and contrasts
it with alpha = 2.0.

Input conventions
-----------------
Every function validates its formula: a list (or tuple) of non-empty
clauses, each a list (or tuple) of non-zero ``int`` literals naming
variables 1..n_vars. A repeated literal counts once (``[1, 1, 2]`` is the
clause "x1 or x2", which is how the Sipser anchor pads clauses to three
literals), and a clause holding both ``x`` and ``-x`` is always true and
is set aside. The solvers are iterative, with an explicit stack, so they
never come near Python's recursion limit.

References:
    Cook, S. A. (1971). The complexity of theorem-proving procedures.
    Proceedings of the 3rd ACM Symposium on Theory of Computing, 151-158.

    Levin, L. A. (1973). Universal sequential search problems. Problems
    of Information Transmission, 9(3), 115-116.

    Karp, R. M. (1972). Reducibility among combinatorial problems. In
    R. E. Miller and J. W. Thatcher (Eds.), Complexity of Computer
    Computations, 85-103. Plenum.

    Davis, M. and Putnam, H. (1960). A computing procedure for
    quantification theory. Journal of the ACM, 7(3), 201-215.

    Davis, M., Logemann, G. and Loveland, D. (1962). A machine program for
    theorem-proving. Communications of the ACM, 5(7), 394-397.

    Selman, B., Kautz, H. and Cohen, B. (1994). Noise strategies for
    improving local search. Proceedings of AAAI-94, 337-343.

    Mitchell, D., Selman, B. and Levesque, H. (1992). Hard and easy
    distributions of SAT problems. Proceedings of AAAI-92, 459-465.

    Mezard, M., Parisi, G. and Zecchina, R. (2002). Analytic and
    algorithmic solution of random satisfiability problems. Science,
    297(5582), 812-815.

    Sipser, M. (2013). Introduction to the Theory of Computation, 3rd
    edition. Cengage. Section 7.4 (the Cook-Levin theorem) and Theorem
    7.44, whose 3-SAT to Vertex Cover example formula is the anchor below.

Examples:
    The Sipser anchor ``(x1 or x2) and (not x1 or not x2) and (not x1 or x2)``
    has exactly one satisfying assignment, x1 = False, x2 = True:

    >>> sipser = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]
    >>> sat_backtracking(sipser, 2)
    {1: False, 2: True}
    >>> sat_dpll(sipser, 2)
    {1: False, 2: True}
    >>> walksat(sipser, 2, seed=42)
    {1: False, 2: True}
    >>> is_satisfying(sipser, {1: False, 2: True})
    True

    All eight sign patterns over three variables rule out every assignment.
    Plain backtracking visits the whole 15-node tree; unit propagation
    halves it:

    >>> eight = [[a * 1, b * 2, c * 3] for a in (1, -1) for b in (1, -1) for c in (1, -1)]
    >>> sat_backtracking(eight, 3, count_nodes=True)
    (None, 15)
    >>> sat_dpll(eight, 3, count_nodes=True)
    (None, 7)
    >>> walksat(eight, 3, max_flips=200, seed=1) is None
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.complexity.reduction_demo import is_satisfying  # re-exported: the SAT verifier
from src.utils.random_utils import make_rng

__all__ = [
    "is_satisfying",
    "random_3sat",
    "sat_backtracking",
    "sat_dpll",
    "walksat",
]

#: A clause after preprocessing: distinct, non-complementary literals.
Clause = Tuple[int, ...]

#: A complete assignment, variable number -> truth value.
Assignment = Dict[int, bool]

#: What a complete solver returns: an assignment or ``None``, optionally
#: paired with the node count.
SolverResult = Union[Optional[Assignment], Tuple[Optional[Assignment], int]]


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int`` (not a ``bool``), else raise ``TypeError``.

    ``bool`` is rejected even though it subclasses ``int``: ``True`` passed
    as a variable count or a literal is a bug in the caller, not a number.

    Args:
        value: The argument to check.
        name: Its parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``, or is a ``bool``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(3, "n_vars")
        3
        >>> _require_int(True, "n_vars")
        Traceback (most recent call last):
            ...
        TypeError: n_vars must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _prepare_cnf(cnf: Any, n_vars: Any) -> List[Clause]:
    """Validate a CNF formula and return its clauses ready for search.

    Every clause and literal is checked before anything is returned, so a
    malformed formula is always reported, wherever the fault is. The clauses
    are then normalised without changing which assignments satisfy the
    formula:

    * a repeated literal is kept once, in first-occurrence order, so
      ``[1, 1, 2]`` becomes ``(1, 2)``;
    * a clause containing both ``x`` and ``-x`` (a tautology) is true under
      every assignment and is dropped.

    After this every clause is non-empty and names each of its variables
    exactly once, which the solvers rely on.

    Args:
        cnf: A list (or tuple) of clauses, each a non-empty list (or tuple)
            of non-zero ``int`` literals.
        n_vars: The number of variables; every literal must satisfy
            ``1 <= abs(literal) <= n_vars``.

    Returns:
        The normalised clauses as tuples, in their original order, with
        tautologies removed.

    Raises:
        TypeError: If ``n_vars`` is not an ``int``, ``cnf`` or a clause is
            not a list or tuple, or a literal is not an ``int`` (``bool``
            rejected).
        ValueError: If ``n_vars`` is negative, a clause is empty, a literal
            is 0, or a literal names a variable above ``n_vars``.

    Time Complexity:
        O(L) where L is the total number of literals.

    Space Complexity:
        O(L) - the normalised copy.

    Examples:
        >>> _prepare_cnf([[1, 1, 2], [-1, 2, 1], [-2]], 2)
        [(1, 2), (-2,)]
        >>> _prepare_cnf([], 0)
        []
        >>> _prepare_cnf([[1, 4]], 3)
        Traceback (most recent call last):
            ...
        ValueError: literal 4 in clause 0 names variable 4, but n_vars is 3
        >>> _prepare_cnf([[1], []], 1)
        Traceback (most recent call last):
            ...
        ValueError: clause 1 is empty; every clause needs at least one literal
    """
    _require_int(n_vars, "n_vars")
    if n_vars < 0:
        raise ValueError(f"n_vars must be non-negative, got {n_vars}")
    if not isinstance(cnf, (list, tuple)):
        raise TypeError(f"cnf must be a list of clauses, got {type(cnf).__name__}")

    clauses: List[Clause] = []
    for index, clause in enumerate(cnf):
        if not isinstance(clause, (list, tuple)):
            raise TypeError(
                f"clause {index} must be a list of ints, got {type(clause).__name__}"
            )
        if not clause:
            raise ValueError(
                f"clause {index} is empty; every clause needs at least one literal"
            )
        for literal in clause:
            if isinstance(literal, bool) or not isinstance(literal, int):
                raise TypeError(
                    f"literals must be ints, got {type(literal).__name__} in clause {index}"
                )
            if literal == 0:
                raise ValueError(
                    f"literal 0 in clause {index} is not allowed "
                    "(variables are numbered from 1)"
                )
            if abs(literal) > n_vars:
                raise ValueError(
                    f"literal {literal} in clause {index} names variable "
                    f"{abs(literal)}, but n_vars is {n_vars}"
                )
        distinct = tuple(dict.fromkeys(clause))
        present = set(distinct)
        if any(-literal in present for literal in distinct):
            continue  # x or not x: always true, constrains nothing
        clauses.append(distinct)
    return clauses


def _occurrence_lists(clauses: Sequence[Clause], n_vars: int) -> List[List[int]]:
    """Index every clause under each literal it contains.

    The list has ``2 * n_vars + 1`` slots and is read as ``occ[literal]``
    for every literal in ``-n_vars .. n_vars``: positive literals use slots
    ``1 .. n_vars`` and Python's negative indexing sends ``-v`` to slot
    ``2 * n_vars + 1 - v``, which lies in ``n_vars + 1 .. 2 * n_vars``. The
    two ranges never overlap and slot 0 is unused, so one flat list serves
    both signs. The same trick is used for the per-literal counters in
    :func:`sat_dpll`.

    Args:
        clauses: Normalised clauses from :func:`_prepare_cnf`.
        n_vars: The number of variables.

    Returns:
        ``occ`` with ``occ[literal]`` the indices of the clauses containing
        ``literal``, in increasing order.

    Time Complexity:
        O(n_vars + L).

    Space Complexity:
        O(n_vars + L).

    Examples:
        >>> occ = _occurrence_lists([(1, 2), (-1, 2)], 2)
        >>> occ[1], occ[-1], occ[2], occ[-2]
        ([0], [1], [0, 1], [])
    """
    occ: List[List[int]] = [[] for _ in range(2 * n_vars + 1)]
    for index, clause in enumerate(clauses):
        for literal in clause:
            occ[literal].append(index)
    return occ


def _complete(values: Sequence[Optional[bool]], n_vars: int) -> Assignment:
    """Turn a 1-indexed value array into a complete assignment dict.

    Variables still undecided (``None``) are set to False, so the result
    always names every variable ``1 .. n_vars``.

    Args:
        values: ``values[v]`` is True, False or None for ``v`` in
            ``1 .. n_vars``; ``values[0]`` is ignored.
        n_vars: The number of variables.

    Returns:
        ``{v: bool}`` for every ``v`` in ``1 .. n_vars``.

    Time Complexity:
        O(n_vars).

    Space Complexity:
        O(n_vars).

    Examples:
        >>> _complete([None, True, None, False], 3)
        {1: True, 2: False, 3: False}
    """
    return {v: bool(values[v]) for v in range(1, n_vars + 1)}


def sat_backtracking(
    cnf: List[List[int]], n_vars: int, count_nodes: bool = False
) -> SolverResult:
    """Decide satisfiability by plain depth-first backtracking (the brute-force baseline).

    The search decides variables in the fixed order 1, 2, ..., n_vars,
    trying True before False, and does no inference at all. Two tests run
    at each node:

    1. **Prune** if a clause has every literal assigned and false. With a
       fixed variable order, a clause becomes fully assigned exactly when
       its highest-numbered variable is assigned, so only the clauses whose
       last variable was just decided need checking.
    2. **Succeed** as soon as every clause contains a true literal. A count
       of satisfied clauses, updated through per-literal occurrence lists,
       makes this O(1) per node. The variables not yet decided are filled
       with False.

    Nodes counted: 1 for the root (the empty assignment) plus 1 for every
    assignment of a value to a variable, the True child and the False child
    each, including a child that is pruned immediately. The full tree over
    n variables has ``2^(n + 1) - 1`` nodes, the most this can report.

    The walk is iterative (an explicit per-depth state array), so a formula
    with many variables never meets Python's recursion limit.

    Args:
        cnf: The formula, a list of non-empty clauses of non-zero ``int``
            literals (DIMACS style).
        n_vars: The number of variables; literals name ``1 .. n_vars``.
            Variables that appear in no clause are allowed.
        count_nodes: If true, also return the number of search nodes.

    Returns:
        A dict assigning every variable ``1 .. n_vars`` that satisfies
        ``cnf``, or ``None`` if the formula is unsatisfiable. With
        ``count_nodes``, the pair ``(result, nodes)``. The assignment found
        is the first satisfying one in True-before-False depth-first order,
        with undecided variables set to False.

    Raises:
        TypeError: If ``n_vars`` is not an ``int``, ``cnf`` or a clause is
            not a list, or a literal is not an ``int`` (``bool`` rejected).
        ValueError: If ``n_vars`` is negative, a clause is empty, a literal
            is 0, or a literal names a variable above ``n_vars``.

    Time Complexity:
        O(2^n * d) worst case for n variables and an average of d clause
        occurrences per variable: at most ``2^(n + 1) - 1`` nodes, each
        costing O(d) to assign, check and undo. O(L) setup for L literals.

    Space Complexity:
        O(n + L) - the value and state arrays, occurrence lists and clause
        counters.

    Examples:
        The Sipser anchor. x1 = True is refuted after trying both values of
        x2, then x1 = False, x2 = True satisfies every clause: six nodes.

        >>> sat_backtracking([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2, count_nodes=True)
        ({1: False, 2: True}, 6)

        The search stops once every clause is satisfied; the undecided
        variables 2 and 3 are filled with False:

        >>> sat_backtracking([[1]], 3, count_nodes=True)
        ({1: True, 2: False, 3: False}, 2)
        >>> sat_backtracking([[1], [-1]], 1, count_nodes=True)
        (None, 3)
        >>> sat_backtracking([], 2)
        {1: False, 2: False}
        >>> sat_backtracking([[1, 0]], 1)
        Traceback (most recent call last):
            ...
        ValueError: literal 0 in clause 0 is not allowed (variables are numbered from 1)
    """
    clauses = _prepare_cnf(cnf, n_vars)
    n = n_vars
    m = len(clauses)
    occ = _occurrence_lists(clauses, n)

    # last_of[v]: clauses whose highest-numbered variable is v. They become
    # fully assigned, and so can first be falsified, when v is decided.
    last_of: List[List[int]] = [[] for _ in range(n + 1)]
    for index, clause in enumerate(clauses):
        last_of[max(abs(literal) for literal in clause)].append(index)

    values: List[Optional[bool]] = [None] * (n + 1)
    true_count = [0] * m      # true literals in each clause
    satisfied = 0             # clauses with at least one true literal
    nodes = 1                 # the root

    if satisfied == m:        # no clauses: the empty assignment works
        result = _complete(values, n)
        return (result, nodes) if count_nodes else result

    # state[d]: 0 = try True next, 1 = try False next, 2 = both tried.
    state = [0] * (n + 1)
    depth = 1
    while depth > 0:
        current = values[depth]
        if current is not None:
            # Undo the previous attempt at this depth.
            for index in occ[depth if current else -depth]:
                true_count[index] -= 1
                if true_count[index] == 0:
                    satisfied -= 1
            values[depth] = None
        tried = state[depth]
        if tried == 2:
            state[depth] = 0
            depth -= 1
            continue
        value = tried == 0
        state[depth] = tried + 1
        values[depth] = value
        nodes += 1
        for index in occ[depth if value else -depth]:
            true_count[index] += 1
            if true_count[index] == 1:
                satisfied += 1
        # Every literal of a clause in last_of[depth] is now assigned, so a
        # zero true count means every literal is false.
        if any(true_count[index] == 0 for index in last_of[depth]):
            continue
        if satisfied == m:
            result = _complete(values, n)
            return (result, nodes) if count_nodes else result
        # Not pruned and not finished. At depth n every clause has been
        # checked and survived, so it is satisfied and we returned above:
        # depth never passes n.
        depth += 1

    return (None, nodes) if count_nodes else None


def sat_dpll(
    cnf: List[List[int]], n_vars: int, count_nodes: bool = False
) -> SolverResult:
    """Decide satisfiability with DPLL: backtracking plus unit propagation and pure literals.

    Before each branch the solver runs, to a fixed point:

    1. **Unit propagation.** A clause with no true literal and exactly one
       unassigned literal forces that literal true. A clause with no true
       and no unassigned literal is a conflict, and the branch is abandoned.
    2. **Pure-literal elimination.** An unassigned variable that occurs in
       the unsatisfied clauses with one sign only is set to make those
       occurrences true; one that occurs in no unsatisfied clause is set to
       False. Neither choice can falsify a clause, so neither loses a
       solution.

    If every clause is then satisfied the search stops (undecided variables
    are filled with False). Otherwise it branches on the lowest-numbered
    unassigned variable, True first. Backtracking undoes a whole decision
    level at once from an assignment trail.

    The bookkeeping is per clause and per literal, so no step re-scans the
    whole formula: each clause keeps its counts of true and of unassigned
    literals, each literal its count of unsatisfied clauses containing it,
    and assigning a variable touches only the clauses in the occurrence
    lists of its two literals.

    Nodes counted: 1 for the root plus 1 for every decision, the True child
    and the False child each. Values forced by unit propagation or set by
    the pure-literal rule are inference, not search, and are not counted.
    Each decision assigns a different variable on its path, so the count is
    at most ``2^(n + 1) - 1``.

    Args:
        cnf: The formula, a list of non-empty clauses of non-zero ``int``
            literals (DIMACS style).
        n_vars: The number of variables; literals name ``1 .. n_vars``.
        count_nodes: If true, also return the number of search nodes.

    Returns:
        A dict assigning every variable ``1 .. n_vars`` that satisfies
        ``cnf``, or ``None`` if the formula is unsatisfiable. With
        ``count_nodes``, the pair ``(result, nodes)``.

    Raises:
        TypeError: If ``n_vars`` is not an ``int``, ``cnf`` or a clause is
            not a list, or a literal is not an ``int`` (``bool`` rejected).
        ValueError: If ``n_vars`` is negative, a clause is empty, a literal
            is 0, or a literal names a variable above ``n_vars``.

    Time Complexity:
        O(2^n * (n + d)) worst case for n variables and an average of d
        clause occurrences per variable: at most ``2^(n + 1) - 1`` decision
        nodes, each paying O(d) per assignment for propagation and O(n) for
        the pure-literal scan and the choice of branching variable. Still
        exponential: inference prunes, it does not change the class.

    Space Complexity:
        O(n + L) - the trail, value array, occurrence lists and counters
        for L literals.

    Examples:
        The Sipser anchor. Deciding x1 = True makes two clauses unit with
        opposite literals of x2: a conflict. x1 = False forces x2 = True.
        Root plus two decisions: three nodes.

        >>> sat_dpll([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2, count_nodes=True)
        ({1: False, 2: True}, 3)

        Pure literals need no search at all: x1 occurs only negated.

        >>> sat_dpll([[-1, 2], [-1, -2]], 2, count_nodes=True)
        ({1: False, 2: False}, 1)

        Two contradictory unit clauses are refuted at the root:

        >>> sat_dpll([[1], [-1]], 1, count_nodes=True)
        (None, 1)
        >>> sat_dpll([[1]], 3)
        {1: True, 2: False, 3: False}
        >>> sat_dpll([[1, 2]], 1)
        Traceback (most recent call last):
            ...
        ValueError: literal 2 in clause 0 names variable 2, but n_vars is 1
    """
    clauses = _prepare_cnf(cnf, n_vars)
    n = n_vars
    m = len(clauses)
    occ = _occurrence_lists(clauses, n)

    values: List[Optional[bool]] = [None] * (n + 1)
    true_count = [0] * m                       # true literals per clause
    free_count = [len(clause) for clause in clauses]   # unassigned literals
    # active[lit]: unsatisfied clauses containing lit (same +/- indexing as occ).
    active = [0] * (2 * n + 1)
    for literal in range(1, n + 1):
        active[literal] = len(occ[literal])
        active[-literal] = len(occ[-literal])
    unsatisfied = m
    trail: List[int] = []                      # variables in assignment order
    units: List[int] = [index for index, clause in enumerate(clauses) if len(clause) == 1]

    def assign(literal: int) -> bool:
        """Make ``literal`` true; return True if some clause became all-false."""
        nonlocal unsatisfied
        variable = literal if literal > 0 else -literal
        values[variable] = literal > 0
        trail.append(variable)
        for index in occ[literal]:
            free_count[index] -= 1
            true_count[index] += 1
            if true_count[index] == 1:
                unsatisfied -= 1
                for other in clauses[index]:
                    active[other] -= 1
        conflict = False
        for index in occ[-literal]:
            free_count[index] -= 1
            if true_count[index] == 0:
                if free_count[index] == 0:
                    conflict = True
                elif free_count[index] == 1:
                    units.append(index)
        return conflict

    def undo_to(mark: int) -> None:
        """Unassign every variable assigned after the first ``mark`` trail entries."""
        nonlocal unsatisfied
        while len(trail) > mark:
            variable = trail.pop()
            literal = variable if values[variable] else -variable
            for index in occ[literal]:
                free_count[index] += 1
                true_count[index] -= 1
                if true_count[index] == 0:
                    unsatisfied += 1
                    for other in clauses[index]:
                        active[other] += 1
            for index in occ[-literal]:
                free_count[index] += 1
            values[variable] = None

    def propagate() -> bool:
        """Run unit propagation to a fixed point; return True on a conflict."""
        while units:
            index = units.pop()
            if true_count[index] or free_count[index] != 1:
                continue  # satisfied since it was queued
            for literal in clauses[index]:
                if values[literal if literal > 0 else -literal] is None:
                    break
            if assign(literal):
                return True
        return False

    def eliminate_pure() -> None:
        """Assign every pure (or unused) variable, repeating until none is left."""
        changed = True
        while changed:
            changed = False
            for variable in range(1, n + 1):
                if values[variable] is not None:
                    continue
                if active[-variable] == 0:
                    # Only positive occurrences (True) or none at all (False).
                    assign(variable if active[variable] else -variable)
                    changed = True
                elif active[variable] == 0:
                    assign(-variable)
                    changed = True

    nodes = 1                                  # the root
    conflict = propagate()
    decisions: List[Tuple[int, int, bool]] = []  # (trail mark, variable, on False branch)
    while True:
        if conflict:
            units.clear()
            while decisions:
                mark, variable, on_false_branch = decisions.pop()
                undo_to(mark)
                if not on_false_branch:
                    decisions.append((mark, variable, True))
                    nodes += 1
                    conflict = assign(-variable) or propagate()
                    break
            else:
                return (None, nodes) if count_nodes else None
            continue
        eliminate_pure()
        if unsatisfied == 0:
            result = _complete(values, n)
            return (result, nodes) if count_nodes else result
        # Some clause is unsatisfied with at least two unassigned literals
        # (none is unit and none is all-false), so a free variable exists.
        variable = next(v for v in range(1, n + 1) if values[v] is None)
        decisions.append((len(trail), variable, False))
        nodes += 1
        conflict = assign(variable) or propagate()


def walksat(
    cnf: List[List[int]],
    n_vars: int,
    max_flips: int = 10_000,
    p: float = 0.5,
    seed: Optional[int] = None,
) -> Optional[Assignment]:
    """Search for a satisfying assignment by WalkSAT randomized local search (a heuristic).

    The Selman-Kautz-Cohen WalkSAT loop:

    1. Start from a uniformly random assignment (each variable True with
       probability 1/2).
    2. If every clause is satisfied, return the assignment.
    3. Otherwise pick an unsatisfied clause uniformly at random. With
       probability ``p`` flip a variable of that clause chosen uniformly at
       random (the *noise* or random-walk move); otherwise flip the variable
       of that clause with the lowest *break count*, the number of clauses
       that are satisfied now and would become unsatisfied by the flip
       (the greedy move), breaking ties uniformly at random.
    4. Repeat from step 2, making at most ``max_flips`` flips in all; the
       assignment is checked once more after the last flip, then ``None`` is
       returned.

    Any flip of a variable in an unsatisfied clause makes that clause true,
    so every move repairs something; the break count measures the damage.
    Pure greed gets stuck in local minima, which the random-walk moves
    escape. (The 1994 paper also takes a zero-break "freebie" flip before
    tossing the coin; this version follows the course contract and tosses
    first.)

    WalkSAT is **incomplete**. It returns only assignments that satisfy
    every clause, so on an unsatisfiable formula it always returns ``None``,
    but ``None`` proves nothing: on a satisfiable formula it can also run
    out of flips. Its strength is speed on satisfiable formulas far beyond
    the reach of the complete solvers.

    Each clause's count of true literals and a list of the unsatisfied
    clauses (with O(1) insert and delete by position) are maintained, so a
    flip and a break count cost time proportional to the number of
    occurrences of the variable, never a pass over the whole formula.

    Args:
        cnf: The formula, a list of non-empty clauses of non-zero ``int``
            literals (DIMACS style).
        n_vars: The number of variables; literals name ``1 .. n_vars``.
        max_flips: The most flips to make. Must be a non-negative ``int``;
            0 only checks the random starting assignment.
        p: The noise probability, a real number in ``[0, 1]``.
        seed: Seed for :func:`~src.utils.random_utils.make_rng`. ``None``
            gives an unpredictable run; pass an ``int`` to reproduce one.

    Returns:
        A dict assigning every variable ``1 .. n_vars`` that satisfies
        ``cnf``, or ``None`` if no satisfying assignment was reached within
        ``max_flips`` flips.

    Raises:
        TypeError: If ``n_vars`` or ``max_flips`` is not an ``int``, ``p``
            is not a real number, ``seed`` is neither ``None`` nor an
            ``int``, ``cnf`` or a clause is not a list, or a literal is not
            an ``int`` (``bool`` rejected throughout).
        ValueError: If ``n_vars`` or ``max_flips`` is negative, ``p`` is
            outside ``[0, 1]``, a clause is empty, a literal is 0, or a
            literal names a variable above ``n_vars``.

    Time Complexity:
        O(L + n + max_flips * k * d) for L literals, n variables, clauses
        of at most k literals and at most d clause occurrences per variable:
        setup, then per flip k break counts and one update, each O(d).

    Space Complexity:
        O(n + L) - the assignment, occurrence lists and clause counters.

    Examples:
        >>> walksat([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2, seed=42)
        {1: False, 2: True}

        Seeded runs repeat exactly:

        >>> formula = random_3sat(20, 40, seed=3)
        >>> walksat(formula, 20, seed=7) == walksat(formula, 20, seed=7)
        True
        >>> is_satisfying(formula, walksat(formula, 20, seed=7))
        True

        On an unsatisfiable formula it can only give up:

        >>> walksat([[1], [-1]], 1, max_flips=50, seed=0) is None
        True
        >>> walksat([[1]], 1, p=1.5)
        Traceback (most recent call last):
            ...
        ValueError: p must be in [0, 1], got 1.5
    """
    clauses = _prepare_cnf(cnf, n_vars)
    _require_int(max_flips, "max_flips")
    if max_flips < 0:
        raise ValueError(f"max_flips must be non-negative, got {max_flips}")
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise TypeError(f"p must be a real number, got {type(p).__name__}")
    if math.isnan(p) or not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    rng = make_rng(seed)

    n = n_vars
    occ = _occurrence_lists(clauses, n)
    values: List[bool] = [False] + [rng.random() < 0.5 for _ in range(n)]

    true_count = [0] * len(clauses)
    unsat: List[int] = []                     # indices of unsatisfied clauses
    position = [-1] * len(clauses)            # where each one sits in unsat
    for index, clause in enumerate(clauses):
        count = sum(1 for literal in clause if values[abs(literal)] == (literal > 0))
        true_count[index] = count
        if count == 0:
            position[index] = len(unsat)
            unsat.append(index)

    def flip(variable: int) -> None:
        """Flip ``variable`` and update the true counts and the unsatisfied list."""
        was_true = variable if values[variable] else -variable
        values[variable] = not values[variable]
        for index in occ[was_true]:
            true_count[index] -= 1
            if true_count[index] == 0:
                position[index] = len(unsat)
                unsat.append(index)
        for index in occ[-was_true]:
            true_count[index] += 1
            if true_count[index] == 1:
                # Swap-remove the clause from the unsatisfied list.
                where = position[index]
                last = unsat.pop()
                if last != index:
                    unsat[where] = last
                    position[last] = where
                position[index] = -1

    for _ in range(max_flips):
        if not unsat:
            return {v: values[v] for v in range(1, n + 1)}
        clause = clauses[unsat[rng.randrange(len(unsat))]]
        if rng.random() < p:
            literal = clause[rng.randrange(len(clause))]
            chosen = literal if literal > 0 else -literal
        else:
            # Every literal of an unsatisfied clause is false, so the true
            # literal of its variable is -literal; flipping breaks each
            # clause in which that literal is the only true one.
            best_break = -1
            candidates: List[int] = []
            for literal in clause:
                breaks = 0
                for index in occ[-literal]:
                    if true_count[index] == 1:
                        breaks += 1
                variable = literal if literal > 0 else -literal
                if best_break < 0 or breaks < best_break:
                    best_break = breaks
                    candidates = [variable]
                elif breaks == best_break:
                    candidates.append(variable)
            chosen = (
                candidates[0]
                if len(candidates) == 1
                else candidates[rng.randrange(len(candidates))]
            )
        flip(chosen)

    if not unsat:
        return {v: values[v] for v in range(1, n + 1)}
    return None


def random_3sat(n_vars: int, n_clauses: int, seed: Optional[int]) -> List[List[int]]:
    """Generate a uniformly random 3-SAT formula.

    Built from one generator, ``rng = make_rng(seed)``. For each of the
    ``n_clauses`` clauses in turn: ``rng.sample(range(1, n_vars + 1), 3)``
    picks three distinct variables, then for each of them, in order,
    ``rng.random() < 0.5`` makes the literal positive and otherwise
    negative. Clauses may repeat; no clause repeats a variable, so none is a
    tautology. The clause-to-variable ratio ``n_clauses / n_vars`` controls
    hardness; near 4.26 random formulas are hardest (see the module
    docstring).

    Args:
        n_vars: The number of variables; at least 3, so that a clause can
            name three distinct ones.
        n_clauses: The number of clauses; non-negative.
        seed: Seed for :func:`~src.utils.random_utils.make_rng`; the same
            seed gives the same formula. ``None`` gives an unpredictable one.

    Returns:
        A list of ``n_clauses`` clauses, each a list of three non-zero
        ``int`` literals over three distinct variables in ``1 .. n_vars``.

    Raises:
        TypeError: If ``n_vars`` or ``n_clauses`` is not an ``int``
            (``bool`` rejected), or ``seed`` is neither ``None`` nor an
            ``int``.
        ValueError: If ``n_vars < 3`` or ``n_clauses < 0``.

    Time Complexity:
        O(n_clauses) - three draws and three coin flips per clause (the
        sample of 3 from a range costs O(1) expected).

    Space Complexity:
        O(n_clauses) - the formula.

    Examples:
        >>> random_3sat(5, 3, seed=42)
        [[1, 5, -3], [5, 1, 3], [5, -1, -3]]
        >>> random_3sat(5, 3, seed=42) == random_3sat(5, 3, seed=42)
        True
        >>> random_3sat(10, 0, seed=1)
        []
        >>> random_3sat(2, 1, seed=1)
        Traceback (most recent call last):
            ...
        ValueError: n_vars must be at least 3 for three distinct variables per clause, got 2
    """
    _require_int(n_vars, "n_vars")
    _require_int(n_clauses, "n_clauses")
    if n_vars < 3:
        raise ValueError(
            f"n_vars must be at least 3 for three distinct variables per clause, got {n_vars}"
        )
    if n_clauses < 0:
        raise ValueError(f"n_clauses must be non-negative, got {n_clauses}")
    rng = make_rng(seed)
    variables = range(1, n_vars + 1)
    formula: List[List[int]] = []
    for _ in range(n_clauses):
        chosen = rng.sample(variables, 3)
        formula.append([v if rng.random() < 0.5 else -v for v in chosen])
    return formula
