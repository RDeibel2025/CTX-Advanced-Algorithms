"""Polynomial-time reductions between NP-complete problems (Week 8, Part 1).

This module is a small framework for *showing* reductions: it builds the
target instance step by step, logs every construction step with its node
count, edge count and target number k, and carries certificates across the
reduction in both directions. Automated tests then confirm on hundreds of
formulas that the yes/no answer is preserved.

What a polynomial-time reduction is
-----------------------------------
A decision problem A *reduces in polynomial time* to a decision problem B
(written A <=p B) when there is a function f, computable in time
polynomial in the size of its input, such that for every instance x of A

    x is a yes-instance of A  iff  f(x) is a yes-instance of B.

Two consequences make reductions the working tool of NP-completeness
(Cook, 1971; Karp, 1972):

* if B has a polynomial-time algorithm, so does A: compute f(x), then
  solve B. A polynomial applied to the output of a polynomial is still a
  polynomial, which is why f itself must run in polynomial time (and so
  produce an instance of polynomial size);
* if A is NP-hard, so is B: every problem in NP reduces to A, and
  following that reduction with f reduces it to B.

So a reduction carries *hardness forward* (3-SAT is NP-complete, therefore
Vertex Cover is NP-hard) and *algorithms backward* (a fast Vertex Cover
solver would give a fast 3-SAT solver). The reductions here also carry
*certificates*: ``map_solution_forward`` turns a certificate for the
source instance into one for the target, and ``map_solution_back`` turns a
target certificate into a source certificate, each in polynomial time.

3-SAT to Vertex Cover (Garey and Johnson, 1979; Sipser, Theorem 7.44)
---------------------------------------------------------------------
For a 3-CNF formula with n variables and m clauses, the reduction builds a
graph G and a bound k from three gadgets. Node names are strings.

1. **Variable gadget.** For each variable x_i, two nodes ``"x{i}"`` and
   ``"¬x{i}"`` joined by an edge. Any vertex cover takes at least one of
   the two; the one it takes names the literal that is true.
2. **Clause gadget.** For each clause C_j, three nodes ``"c{j}_1"``,
   ``"c{j}_2"`` and ``"c{j}_3"``, one per literal position, joined in a
   triangle. Any vertex cover takes at least two of the three corners; the
   corner it leaves out points at a literal that satisfies the clause.
3. **Connecting edges.** Each clause node ``"c{j}_{p}"`` is joined to the
   variable-gadget node of the literal in position p.

The bound is k = n + 2m. A clause with fewer than three literals is padded
to three by repeating its last literal (``x OR y`` is the same clause as
``x OR y OR y``), and the step log says which clauses were padded. Padding
copies are distinct clause nodes, so even when two positions of a clause
carry the same literal their triangle edges and connecting edges are
distinct pairs, and the counts below hold for every formula.

**Why it is correct (proof sketch).** The n variable edges and the m
triangles share no nodes, so every vertex cover has at least
n + 2m = k nodes.

* (=>) From a satisfying assignment, put the true literal node of every
  variable into the cover (n nodes) and, in every triangle, the two
  corners other than one corner whose literal is true (2m nodes). Variable
  and triangle edges are covered. A connecting edge from a chosen corner
  is covered by that corner; the one from the left-out corner ends at a
  true literal node, which is in the cover. The cover has exactly k nodes.
* (<=) A cover with at most k nodes must take exactly one node of every
  variable edge and exactly two corners of every triangle. Set x_i = True
  iff ``"x{i}"`` is in the cover. In each triangle the left-out corner's
  connecting edge can only be covered from the variable side, so that
  literal's node is in the cover, so that literal is true and the clause is
  satisfied.

**Size.** G has 2n + 3m nodes and n + 3m + 3m = n + 6m edges (n variable
edges, 3m triangle edges, 3m connecting edges). The construction does O(1)
work per variable and per clause, so it runs in O(n + m) time: polynomial
in the size of the formula. That bound is the point of the exercise; a
construction of exponential size would prove nothing about hardness.

The rest of the chain
---------------------
Two shorter reductions keep the same vertex set:

* **Vertex Cover to Independent Set.** S is a vertex cover of G iff V - S
  is an independent set of G (an edge with both ends outside S is exactly
  an edge S fails to cover), so (G, k) maps to (G, |V| - k).
* **Independent Set to Clique.** S is independent in G iff S is a clique
  of the complement graph, so (G, k) maps to (complement of G, k).

:func:`reduce_chain` applies a list of reductions in order, so the whole
chain 3-SAT -> Vertex Cover -> Independent Set -> Clique is one call.

The framework
-------------
:class:`Reduction` has a ``name`` and four methods: ``transform`` builds
the target instance and returns a :class:`ReductionResult` (both
instances, the step log, and a short description of every target node);
``map_solution_forward`` and ``map_solution_back`` carry certificates
across and refuse input that is not a valid certificate; ``explain``
returns the step log as text and sends each line to :mod:`logging` at INFO
level.

Verifiers and the exact solver
------------------------------
The final section of the file holds the pieces every Week 8 module shares:

* the two polynomial-time **verifiers**, :func:`is_satisfying` for CNF
  formulas and ``is_vertex_cover`` (re-exported from
  :mod:`src.utils.graph_utils`). A verifier checks a proposed certificate
  in time linear in the input; that is what puts SAT and Vertex Cover in
  NP. Finding a certificate is the hard part.
* :func:`min_vertex_cover_exact`, an exact branch-and-bound solver used to
  confirm the reduction on small instances: a formula with n variables and
  m clauses is satisfiable iff its reduction graph has a vertex cover of
  size at most n + 2m.

============================== ====================== ====================
Operation                      Time                   Space
============================== ====================== ====================
3-SAT -> VC transform          O(n + m)               O(n + m)
3-SAT -> VC certificate maps   O(n + m)               O(n + m)
VC -> IS transform / maps      O(V + E)               O(V + E)
IS -> Clique transform / maps  O(V^2 + E)             O(V^2)
is_satisfying                  O(total literals)      O(1)
min_vertex_cover_exact         O(2^k (V + E))         O(V + E + k)
============================== ====================== ====================

References:
    Cook, S. A. (1971). The complexity of theorem-proving procedures.
    *Proceedings of the Third Annual ACM Symposium on Theory of
    Computing*, 151-158.

    Karp, R. M. (1972). Reducibility among combinatorial problems. In
    R. E. Miller & J. W. Thatcher (Eds.), *Complexity of Computer
    Computations* (pp. 85-103). Plenum.

    Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability:
    A Guide to the Theory of NP-Completeness*. W. H. Freeman.

    Sipser, M. (2013). *Introduction to the Theory of Computation* (3rd
    ed.), Theorem 7.44. Cengage.

Examples:
    >>> sipser = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]
    >>> reduction = ThreeSatToVertexCover()
    >>> result = reduction.transform((sipser, 2))
    >>> (vertices, edges), k = result.target_instance
    >>> len(vertices), len(edges), k
    (13, 20, 8)
    >>> cover = reduction.map_solution_forward((sipser, 2), {1: False, 2: True}, result)
    >>> len(cover), is_vertex_cover(edges, cover)
    (8, True)
    >>> reduction.map_solution_back((sipser, 2), cover, result)
    {1: False, 2: True}
    >>> is_satisfying(sipser, {1: False, 2: True})
    True
    >>> len(min_vertex_cover_exact(vertices, edges))
    8
    >>> sorted(min_vertex_cover_exact([0, 1, 2, 3, 4, 5], {(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)}))
    [0]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Sequence, Set, Tuple

from src.utils.graph_utils import (
    complement,
    is_clique,
    is_independent_set,
    is_vertex_cover,  # re-exported: the Vertex Cover verifier
    normalize_edges,
)

__all__ = [
    "IndependentSetToClique",
    "Reduction",
    "ReductionResult",
    "ThreeSatToVertexCover",
    "VertexCoverToIndependentSet",
    "is_satisfying",
    "is_vertex_cover",
    "min_vertex_cover_exact",
    "reduce_chain",
]

logger = logging.getLogger(__name__)

#: The logical-not sign used in negative-literal node names such as "¬x1".
_NOT_SIGN = "\u00ac"

#: A clause gadget has exactly this many literal positions.
_CLAUSE_WIDTH = 3


# --- The reduction framework ---


@dataclass
class ReductionResult:
    """The output of one reduction: both instances, the log and the node map.

    A reduction is only convincing when it can be inspected, so the result
    keeps everything a reader needs to follow it: the source instance as
    the reduction read it, the target instance it built, one human-readable
    line per construction step, and a short description of what every
    target node stands for in the source.

    Attributes:
        source_instance: A normalized copy of the input instance, for
            example ``(cnf, n_vars)`` or ``((vertices, edges), k)``. Later
            changes to the caller's objects do not affect it.
        target_instance: The instance the reduction built, for example
            ``((vertices, edges), k)``.
        steps: The construction log, one line per step, with node counts,
            edge counts and the target bound k. :meth:`Reduction.explain`
            prints and logs it.
        node_labels: Maps every target node to a short description of what
            it represents, for example ``"literal x1 in clause 2,
            position 1"``.

    Time Complexity:
        O(1) to construct; the reductions fill it in polynomial time.

    Space Complexity:
        O(size of both instances + length of the log).

    Examples:
        >>> result = ReductionResult(([[1]], 1), (([], set()), 0))
        >>> result.steps, result.node_labels
        ([], {})
    """

    source_instance: Any
    target_instance: Any
    steps: List[str] = field(default_factory=list)
    node_labels: Dict[Hashable, str] = field(default_factory=dict)


class Reduction:
    """Base class of the framework: one polynomial-time reduction A <=p B.

    A subclass sets :attr:`name` and implements three methods:

    * :meth:`transform` maps an instance of A to an instance of B and
      returns a :class:`ReductionResult` with the step-by-step log;
    * :meth:`map_solution_forward` maps a certificate for the A instance to
      a certificate for the B instance it was transformed into;
    * :meth:`map_solution_back` maps a certificate for the B instance back
      to a certificate for the A instance.

    :meth:`explain` is shared: it returns the log as text and sends each
    line to the module logger. Every mapping method takes the original
    ``instance`` *and* the ``result`` of transforming it, and refuses a
    result that was produced from a different instance; a subclass that
    uses that check also implements the private hook ``_comparable``
    (a normalized, comparable form of a source instance).

    Attributes:
        name: A short human-readable name such as
            ``"3-SAT -> Vertex Cover"``.

    Time Complexity:
        O(1) to construct (no state); each method's cost is in its own
        docstring and is polynomial in every subclass.

    Space Complexity:
        O(1) per reduction object.

    Examples:
        >>> class Identity(Reduction):
        ...     name = "identity"
        >>> Identity().explain(ReductionResult(1, 1, ["nothing to do"]))
        'nothing to do'
        >>> Identity().transform(1)
        Traceback (most recent call last):
            ...
        NotImplementedError: identity does not implement transform
    """

    name: str = "abstract reduction"

    def __repr__(self) -> str:
        """Return ``ClassName()``, the expression that rebuilds the reduction.

        Args:
            None.

        Returns:
            The class name followed by ``()``.

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> ThreeSatToVertexCover()
            ThreeSatToVertexCover()
        """
        return f"{type(self).__name__}()"

    def transform(self, instance: Any) -> ReductionResult:
        """Map a source instance to a target instance, logging every step.

        Args:
            instance: An instance of the source problem.

        Returns:
            A :class:`ReductionResult`.

        Raises:
            NotImplementedError: Always, in the base class.

        Time Complexity:
            Polynomial in the size of ``instance`` in every subclass.

        Space Complexity:
            Polynomial in the size of ``instance`` in every subclass.

        Examples:
            >>> Reduction().transform(([[1]], 1))
            Traceback (most recent call last):
                ...
            NotImplementedError: abstract reduction does not implement transform
        """
        raise NotImplementedError(f"{self.name} does not implement transform")

    def map_solution_forward(self, instance: Any, solution: Any, result: ReductionResult) -> Any:
        """Map a source certificate to a certificate for the target instance.

        Args:
            instance: The source instance that was transformed.
            solution: A certificate for ``instance``.
            result: The :class:`ReductionResult` of transforming ``instance``.

        Returns:
            A certificate for ``result.target_instance``.

        Raises:
            NotImplementedError: Always, in the base class.

        Time Complexity:
            Polynomial in the instance size in every subclass.

        Space Complexity:
            Polynomial in the instance size in every subclass.

        Examples:
            >>> Reduction().map_solution_forward(None, None, None)
            Traceback (most recent call last):
                ...
            NotImplementedError: abstract reduction does not implement map_solution_forward
        """
        raise NotImplementedError(f"{self.name} does not implement map_solution_forward")

    def map_solution_back(self, instance: Any, solution: Any, result: ReductionResult) -> Any:
        """Map a target certificate back to a certificate for the source.

        Args:
            instance: The source instance that was transformed.
            solution: A certificate for ``result.target_instance``.
            result: The :class:`ReductionResult` of transforming ``instance``.

        Returns:
            A certificate for ``instance``.

        Raises:
            NotImplementedError: Always, in the base class.

        Time Complexity:
            Polynomial in the instance size in every subclass.

        Space Complexity:
            Polynomial in the instance size in every subclass.

        Examples:
            >>> Reduction().map_solution_back(None, None, None)
            Traceback (most recent call last):
                ...
            NotImplementedError: abstract reduction does not implement map_solution_back
        """
        raise NotImplementedError(f"{self.name} does not implement map_solution_back")

    def explain(self, result: ReductionResult) -> str:
        """Return the step-by-step explanation and send each line to logging.

        The lines of ``result.steps`` are joined with newlines, so the text
        can be printed as is, and each line is also passed to
        ``logging.getLogger(__name__).info`` (the logger
        ``src.complexity.reduction_demo``), so a program that configures
        logging gets the same explanation in its log. Nothing is written
        to the console unless logging is configured to show INFO messages.

        Args:
            result: The :class:`ReductionResult` to explain.

        Returns:
            The steps joined by ``"\\n"`` (an empty string for no steps).

        Raises:
            TypeError: If ``result`` is not a :class:`ReductionResult`.

        Time Complexity:
            O(L) for a log of total length L.

        Space Complexity:
            O(L) for the joined text.

        Examples:
            >>> reduction = ThreeSatToVertexCover()
            >>> text = reduction.explain(reduction.transform(([[1, 2, 2]], 2)))
            >>> print(text.splitlines()[0])
            3-SAT -> Vertex Cover: n = 2 variables, m = 1 clause: (x1 OR x2 OR x2)
            >>> "k = 4" in text
            True
            >>> reduction.explain("steps")
            Traceback (most recent call last):
                ...
            TypeError: result must be a ReductionResult, got str
        """
        if not isinstance(result, ReductionResult):
            raise TypeError(
                f"result must be a ReductionResult, got {type(result).__name__}"
            )
        for line in result.steps:
            logger.info("%s", line)
        return "\n".join(result.steps)

    def _require_result_for(self, instance: Any, result: Any) -> None:
        """Raise unless ``result`` was produced by transforming ``instance``.

        Both are parsed with the subclass's ``_parse`` and compared, so an
        equal instance written differently (a tuple for a list, edges in
        another order) is accepted.

        Args:
            instance: The source instance.
            result: The claimed :class:`ReductionResult` of transforming it.

        Raises:
            TypeError: If ``result`` is not a :class:`ReductionResult`.
            ValueError: If ``result.source_instance`` is a different
                instance.

        Time Complexity:
            O(size of the instance) for parsing both.

        Space Complexity:
            O(size of the instance).

        Examples:
            >>> reduction = ThreeSatToVertexCover()
            >>> result = reduction.transform(([[1]], 1))
            >>> reduction._require_result_for(([(1,)], 1), result)
            >>> reduction._require_result_for(([[-1]], 1), result)
            Traceback (most recent call last):
                ...
            ValueError: result was not produced by transforming this instance
        """
        if not isinstance(result, ReductionResult):
            raise TypeError(
                f"result must be a ReductionResult, got {type(result).__name__}"
            )
        if self._comparable(instance) != self._comparable(result.source_instance):
            raise ValueError("result was not produced by transforming this instance")

    def _comparable(self, instance: Any) -> Any:
        """Return a normalized form of ``instance`` for equality checks.

        Args:
            instance: A source instance.

        Returns:
            A value that compares equal for equal instances.

        Raises:
            NotImplementedError: Always, in the base class.

        Time Complexity:
            O(size of the instance) in every subclass.

        Space Complexity:
            O(size of the instance) in every subclass.

        Examples:
            >>> Reduction()._comparable(1)
            Traceback (most recent call last):
                ...
            NotImplementedError: abstract reduction does not implement _comparable
        """
        raise NotImplementedError(f"{self.name} does not implement _comparable")


# --- Private helpers shared by the reductions ---


def _literal_node(literal: int) -> str:
    """Return the variable-gadget node name of a literal.

    Args:
        literal: A non-zero DIMACS literal.

    Returns:
        ``"x{i}"`` for the literal ``i`` and ``"¬x{i}"`` for ``-i``.

    Time Complexity:
        O(digits of the literal).

    Space Complexity:
        O(digits of the literal).

    Examples:
        >>> _literal_node(3), _literal_node(-3)
        ('x3', '¬x3')
    """
    return f"x{literal}" if literal > 0 else f"{_NOT_SIGN}x{-literal}"


def _literal_text(literal: int) -> str:
    """Return a plain-ASCII reading of a literal for the explanation text.

    Args:
        literal: A non-zero DIMACS literal.

    Returns:
        ``"x{i}"`` for ``i`` and ``"NOT x{i}"`` for ``-i``.

    Time Complexity:
        O(digits of the literal).

    Space Complexity:
        O(digits of the literal).

    Examples:
        >>> _literal_text(2), _literal_text(-2)
        ('x2', 'NOT x2')
    """
    return f"x{literal}" if literal > 0 else f"NOT x{-literal}"


def _clause_text(clause: Sequence[int]) -> str:
    """Return a clause as ``(lit OR lit OR lit)`` in plain ASCII.

    Args:
        clause: A non-empty sequence of non-zero DIMACS literals.

    Returns:
        The literals joined by ``" OR "`` inside parentheses.

    Time Complexity:
        O(length of the clause).

    Space Complexity:
        O(length of the clause).

    Examples:
        >>> _clause_text([1, -2, 3])
        '(x1 OR NOT x2 OR x3)'
    """
    return "(" + " OR ".join(_literal_text(literal) for literal in clause) + ")"


def _parse_cnf_instance(instance: Any) -> Tuple[List[List[int]], int]:
    """Validate a 3-SAT instance ``(cnf, n_vars)`` and return a clean copy.

    Args:
        instance: A pair ``(cnf, n_vars)``: ``cnf`` a list (or tuple) of
            clauses, each a list (or tuple) of one to three non-zero ``int``
            literals over the variables ``1 .. n_vars``.

    Returns:
        ``(clauses, n_vars)`` with ``clauses`` a new list of new lists (the
        clauses exactly as given, before padding).

    Raises:
        TypeError: If ``instance`` is not a list or tuple, ``n_vars`` is not
            an ``int``, ``cnf`` or a clause is not a list or tuple, or a
            literal is not an ``int`` (``bool`` is rejected throughout).
        ValueError: If ``instance`` does not have two parts, ``n_vars`` is
            negative, a clause is empty or has more than three literals, a
            literal is 0, or a literal names a variable outside
            ``1 .. n_vars``.

    Time Complexity:
        O(n + total literals).

    Space Complexity:
        O(total literals) for the copy.

    Examples:
        >>> _parse_cnf_instance(([(1, -2)], 2))
        ([[1, -2]], 2)
        >>> _parse_cnf_instance(([[1, 3]], 2))
        Traceback (most recent call last):
            ...
        ValueError: clause 1 has literal 3, but variables are numbered 1..2
        >>> _parse_cnf_instance(([[1], []], 2))
        Traceback (most recent call last):
            ...
        ValueError: clause 2 is empty; an empty clause can never be satisfied
    """
    if not isinstance(instance, (list, tuple)):
        raise TypeError(
            f"a 3-SAT instance must be a pair (cnf, n_vars), got {type(instance).__name__}"
        )
    if len(instance) != 2:
        raise ValueError(
            f"a 3-SAT instance must be a pair (cnf, n_vars), got {len(instance)} parts"
        )
    cnf, n_vars = instance
    if isinstance(n_vars, bool) or not isinstance(n_vars, int):
        raise TypeError(f"n_vars must be an int, got {type(n_vars).__name__}")
    if n_vars < 0:
        raise ValueError(f"n_vars must be non-negative, got {n_vars}")
    if not isinstance(cnf, (list, tuple)):
        raise TypeError(f"cnf must be a list of clauses, got {type(cnf).__name__}")
    clauses: List[List[int]] = []
    for j, clause in enumerate(cnf, start=1):
        if not isinstance(clause, (list, tuple)):
            raise TypeError(
                f"clause {j} must be a list of ints, got {type(clause).__name__}"
            )
        if not clause:
            raise ValueError(f"clause {j} is empty; an empty clause can never be satisfied")
        if len(clause) > _CLAUSE_WIDTH:
            raise ValueError(
                f"clause {j} has {len(clause)} literals; a 3-CNF clause has at most three"
            )
        for literal in clause:
            if isinstance(literal, bool) or not isinstance(literal, int):
                raise TypeError(
                    f"clause {j} has a literal of type {type(literal).__name__}; "
                    f"literals must be ints"
                )
            if literal == 0:
                raise ValueError(
                    f"clause {j} has literal 0, which is not allowed "
                    f"(variables are numbered from 1)"
                )
            if abs(literal) > n_vars:
                raise ValueError(
                    f"clause {j} has literal {literal}, but variables are numbered 1..{n_vars}"
                )
        clauses.append(list(clause))
    return clauses, n_vars


def _pad_clause(clause: Sequence[int]) -> List[int]:
    """Pad a clause to exactly three positions by repeating its last literal.

    ``l1 OR l2`` and ``l1 OR l2 OR l2`` are satisfied by exactly the same
    assignments, so padding never changes the formula. Every position still
    becomes its own clause node.

    Args:
        clause: A clause of one to three literals.

    Returns:
        A new list of exactly three literals.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _pad_clause([4]), _pad_clause([1, -2]), _pad_clause([1, 2, 3])
        ([4, 4, 4], [1, -2, -2], [1, 2, 3])
    """
    return list(clause) + [clause[-1]] * (_CLAUSE_WIDTH - len(clause))


def _distinct_vertices(vertices: Any) -> List[Hashable]:
    """Return the distinct vertices in a deterministic order.

    A list or other ordered iterable keeps its first-seen order. A ``set``
    or ``frozenset`` is sorted (with the module's process-stable fallback
    order for incomparable labels), because the iteration order of a set of
    ``str`` labels changes from process to process.

    Args:
        vertices: A non-string iterable of hashable labels.

    Returns:
        A new list of the distinct labels.

    Raises:
        TypeError: If ``vertices`` is a ``str``/``bytes``, is not iterable,
            or holds an unhashable label.

    Time Complexity:
        O(V), or O(V log V) for a set.

    Space Complexity:
        O(V).

    Examples:
        >>> _distinct_vertices([3, 1, 3, 2])
        [3, 1, 2]
        >>> _distinct_vertices({"b", "a"})
        ['a', 'b']
    """
    if isinstance(vertices, (set, frozenset)):
        return _canonical_vertices(vertices)
    if isinstance(vertices, (str, bytes)):
        raise TypeError(
            f"vertices must be an iterable collection, got {type(vertices).__name__}"
        )
    try:
        return list(dict.fromkeys(vertices))
    except TypeError:
        raise TypeError(
            f"vertices must be an iterable collection of hashable labels, "
            f"got {type(vertices).__name__}"
        ) from None


def _parse_graph_instance(instance: Any) -> Tuple[List[Hashable], Set[Tuple[Hashable, Hashable]], int]:
    """Validate a graph instance ``((vertices, edges), k)`` and normalize it.

    Args:
        instance: A pair ``((vertices, edges), k)`` with ``vertices`` a
            non-string iterable, ``edges`` an iterable of 2-element edges
            between those vertices, and ``k`` a non-negative ``int``.

    Returns:
        ``(vertex_list, edge_set, k)``: the distinct vertices (see
        :func:`_distinct_vertices`), the edges normalized by
        :func:`src.utils.graph_utils.normalize_edges`, and ``k``.

    Raises:
        TypeError: If ``instance`` or the graph is not a list or tuple,
            ``vertices`` or ``edges`` is not a non-string iterable, or ``k``
            is not an ``int`` (``bool`` rejected).
        ValueError: If ``instance`` or the graph does not have two parts,
            an edge is malformed or a self-loop, an edge endpoint is not a
            vertex, or ``k`` is negative.

    Time Complexity:
        O(V + E).

    Space Complexity:
        O(V + E).

    Examples:
        >>> vertex_list, edge_set, k = _parse_graph_instance((([1, 2, 3], [(2, 1), (3, 2)]), 1))
        >>> vertex_list, sorted(edge_set), k
        ([1, 2, 3], [(1, 2), (2, 3)], 1)
        >>> _parse_graph_instance((([1, 2], {(1, 2)}), -1))
        Traceback (most recent call last):
            ...
        ValueError: k must be non-negative, got -1
    """
    if not isinstance(instance, (list, tuple)):
        raise TypeError(
            f"a graph instance must be a pair ((vertices, edges), k), "
            f"got {type(instance).__name__}"
        )
    if len(instance) != 2:
        raise ValueError(
            f"a graph instance must be a pair ((vertices, edges), k), "
            f"got {len(instance)} parts"
        )
    graph, k = instance
    if not isinstance(graph, (list, tuple)):
        raise TypeError(
            f"the graph must be a pair (vertices, edges), got {type(graph).__name__}"
        )
    if len(graph) != 2:
        raise ValueError(
            f"the graph must be a pair (vertices, edges), got {len(graph)} parts"
        )
    vertices, edges = graph
    vertex_list = _distinct_vertices(vertices)
    edge_set = normalize_edges(edges)
    members = set(vertex_list)
    for u, v in edge_set:
        if u not in members or v not in members:
            raise ValueError(f"edge {(u, v)!r} has an endpoint that is not in vertices")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return vertex_list, edge_set, k


def _vertex_subset(solution: Any, name: str, vertex_list: Sequence[Hashable]) -> Set[Hashable]:
    """Return a certificate as a set of vertices, checking every member exists.

    Args:
        solution: A non-string iterable of vertices.
        name: What the certificate is (``"cover"``, ``"clique"``, ...), for
            error messages.
        vertex_list: The vertices of the graph it must come from.

    Returns:
        A new ``set`` of the vertices in ``solution``.

    Raises:
        TypeError: If ``solution`` is a ``str``/``bytes`` or not iterable.
        ValueError: If a member of ``solution`` is not a vertex.

    Time Complexity:
        O(V + |solution|).

    Space Complexity:
        O(V + |solution|).

    Examples:
        >>> sorted(_vertex_subset([2, 1, 2], "cover", [1, 2, 3]))
        [1, 2]
        >>> _vertex_subset({9}, "cover", [1, 2, 3])
        Traceback (most recent call last):
            ...
        ValueError: cover contains 9, which is not a vertex of the graph
    """
    if isinstance(solution, (str, bytes)):
        raise TypeError(
            f"{name} must be an iterable collection of vertices, got {type(solution).__name__}"
        )
    try:
        chosen = set(solution)
    except TypeError:
        raise TypeError(
            f"{name} must be an iterable collection of vertices, got {type(solution).__name__}"
        ) from None
    members = set(vertex_list)
    strangers = [vertex for vertex in chosen if vertex not in members]
    if strangers:
        # Sort only the offenders, so the message is the same in every
        # process and a valid certificate costs no sort.
        vertex = _canonical_vertices(strangers)[0]
        raise ValueError(f"{name} contains {vertex!r}, which is not a vertex of the graph")
    return chosen


# --- 3-SAT -> Vertex Cover ---


class ThreeSatToVertexCover(Reduction):
    """The Garey-Johnson reduction from 3-SAT to Vertex Cover (Sipser Thm 7.44).

    The source instance is ``(cnf, n_vars)``: a CNF formula in DIMACS style
    with clauses of one to three literals over the variables
    ``1 .. n_vars``. The target instance is ``((vertices, edges), k)`` with
    ``k = n + 2m``; the formula is satisfiable iff the graph has a vertex
    cover of size at most k. The gadgets, the proof and the size bounds
    (2n + 3m nodes, n + 6m edges) are in the module docstring.

    Node names: ``"x{i}"`` and ``"¬x{i}"`` for the variable gadgets,
    ``"c{j}_{p}"`` for clause ``j = 1..m`` and position ``p = 1..3``.

    Attributes:
        name: ``"3-SAT -> Vertex Cover"``.

    Time Complexity:
        O(n + m) for :meth:`transform` and for each certificate map.

    Space Complexity:
        O(n + m) for the 2n + 3m nodes, n + 6m edges and the step log.

    Examples:
        >>> reduction = ThreeSatToVertexCover()
        >>> result = reduction.transform(([[1, -2]], 2))
        >>> (vertices, edges), k = result.target_instance
        >>> vertices
        ['x1', '¬x1', 'x2', '¬x2', 'c1_1', 'c1_2', 'c1_3']
        >>> len(edges), k
        (8, 4)
    """

    name = "3-SAT -> Vertex Cover"

    def _comparable(self, instance: Any) -> Any:
        """Return ``(clauses as tuples, n_vars)`` for equality checks.

        Args:
            instance: A 3-SAT instance ``(cnf, n_vars)``.

        Returns:
            A hashable normalized form of the instance.

        Raises:
            TypeError: As :func:`_parse_cnf_instance`.
            ValueError: As :func:`_parse_cnf_instance`.

        Time Complexity:
            O(n + total literals).

        Space Complexity:
            O(total literals).

        Examples:
            >>> ThreeSatToVertexCover()._comparable(([[1, -2]], 2))
            (((1, -2),), 2)
        """
        clauses, n_vars = _parse_cnf_instance(instance)
        return tuple(tuple(clause) for clause in clauses), n_vars

    def transform(self, instance: Any) -> ReductionResult:
        """Build the Vertex Cover instance ``((vertices, edges), k)`` for a formula.

        The construction follows the three gadgets of the module docstring:
        a variable edge ``x{i} - ¬x{i}`` per variable, a triangle
        ``c{j}_1, c{j}_2, c{j}_3`` per clause, and a connecting edge from
        each clause node to the node of the literal in its position. Clauses
        with fewer than three literals are first padded by repeating their
        last literal, which the log reports. The log has one line per
        construction step (padding, variable gadgets, clause gadgets,
        connecting edges, target k, size summary), each with its node and
        edge counts, plus one indented line per clause listing its triangle
        and connecting edges.

        Args:
            instance: ``(cnf, n_vars)``, where ``cnf`` is a list of clauses,
                each a list of one to three non-zero ``int`` literals, and
                ``n_vars`` is the number of variables. Variables that occur
                in no clause still get a variable gadget.

        Returns:
            A :class:`ReductionResult` whose ``target_instance`` is
            ``((vertices, edges), k)``: ``vertices`` a list in the order
            ``x1, ¬x1, x2, ¬x2, ..., c1_1, c1_2, c1_3, c2_1, ...``,
            ``edges`` a set of pairs normalized by
            :func:`src.utils.graph_utils.normalize_edges`, and
            ``k = n_vars + 2 * len(cnf)``. ``source_instance`` is a copy of
            ``(cnf, n_vars)`` with the clauses as given (unpadded).

        Raises:
            TypeError: If the instance is not a pair, ``n_vars`` is not an
                ``int``, ``cnf`` or a clause is not a list or tuple, or a
                literal is not an ``int`` (``bool`` rejected).
            ValueError: If a clause is empty or has more than three
                literals, a literal is 0, a literal names a variable outside
                ``1 .. n_vars``, or ``n_vars`` is negative.

        Time Complexity:
            O(n + m): constant work per variable and per clause (2n + 3m
            nodes, n + 6m edges, one log line per clause).

        Space Complexity:
            O(n + m).

        Examples:
            >>> reduction = ThreeSatToVertexCover()
            >>> result = reduction.transform(([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2))
            >>> (vertices, edges), k = result.target_instance
            >>> len(vertices), len(edges), k
            (13, 20, 8)
            >>> ("c2_1", "¬x1") in edges
            True
            >>> result.node_labels["c2_1"]
            'literal NOT x1 in clause 2, position 1'
            >>> padded = reduction.transform(([[2]], 2))
            >>> padded.node_labels["c1_3"]
            'literal x2 in clause 1, position 3 (padding copy)'
            >>> reduction.transform(([[1, 0]], 2))
            Traceback (most recent call last):
                ...
            ValueError: clause 1 has literal 0, which is not allowed (variables are numbered from 1)
        """
        clauses, n_vars = _parse_cnf_instance(instance)
        m = len(clauses)
        padded = [_pad_clause(clause) for clause in clauses]

        vertices: List[str] = []
        labels: Dict[Hashable, str] = {}
        variable_edges: List[Tuple[str, str]] = []
        for i in range(1, n_vars + 1):
            positive, negative = _literal_node(i), _literal_node(-i)
            vertices.extend((positive, negative))
            labels[positive] = f"literal x{i} (variable gadget for x{i})"
            labels[negative] = f"literal NOT x{i} (variable gadget for x{i})"
            variable_edges.append((positive, negative))

        triangle_edges: List[Tuple[str, str]] = []
        connecting_edges: List[Tuple[str, str]] = []
        clause_lines: List[str] = []
        for j, (original, clause) in enumerate(zip(clauses, padded), start=1):
            corners = [f"c{j}_{p}" for p in range(1, _CLAUSE_WIDTH + 1)]
            vertices.extend(corners)
            triangle_edges.extend(
                [(corners[0], corners[1]), (corners[0], corners[2]), (corners[1], corners[2])]
            )
            links = []
            for p, (corner, literal) in enumerate(zip(corners, clause), start=1):
                note = " (padding copy)" if p > len(original) else ""
                labels[corner] = f"literal {_literal_text(literal)} in clause {j}, position {p}{note}"
                connecting_edges.append((corner, _literal_node(literal)))
                links.append(f"{corner} - {_literal_node(literal)}")
            clause_lines.append(
                f"    clause {j} {_clause_text(clause)}: triangle {', '.join(corners)}; "
                f"connecting edges {', '.join(links)}"
            )

        edges = normalize_edges(variable_edges + triangle_edges + connecting_edges)
        k = n_vars + 2 * m

        formula = " AND ".join(_clause_text(clause) for clause in clauses) or "(empty formula)"
        steps = [
            f"{self.name}: n = {n_vars} variable{'' if n_vars == 1 else 's'}, "
            f"m = {m} clause{'' if m == 1 else 's'}: {formula}"
        ]
        short = [(j, original) for j, original in enumerate(clauses, start=1)
                 if len(original) < _CLAUSE_WIDTH]
        if short:
            steps.append(
                f"Step 0 - padding: clauses with fewer than three literals are padded by "
                f"repeating their last literal, which does not change which assignments "
                f"satisfy them ({len(short)} of {m} padded)"
            )
            for j, original in short:
                steps.append(
                    f"    clause {j}: {_clause_text(original)} becomes "
                    f"{_clause_text(padded[j - 1])}"
                )
        else:
            steps.append(
                "Step 0 - padding: every clause already has three literal positions; "
                "no padding needed"
            )
        steps.append(
            f"Step 1 - variable gadgets: for each variable x_i, nodes x{{i}} and "
            f"{_NOT_SIGN}x{{i}} joined by one edge (any cover takes at least one of them): "
            f"2n = {2 * n_vars} nodes, n = {len(variable_edges)} edges"
        )
        steps.append(
            f"Step 2 - clause gadgets: for each clause C_j, nodes c{{j}}_1, c{{j}}_2, c{{j}}_3 "
            f"(one per literal position) joined in a triangle (any cover takes at least "
            f"two of them): 3m = {_CLAUSE_WIDTH * m} nodes, 3m = {len(triangle_edges)} edges"
        )
        steps.append(
            f"Step 3 - connecting edges: each clause node c{{j}}_{{p}} joined to the "
            f"variable-gadget node of the literal in position p: "
            f"3m = {len(connecting_edges)} edges"
        )
        steps.extend(clause_lines)
        steps.append(
            f"Step 4 - target: k = {k} (k = n + 2m = {n_vars} + 2 * {m}); the formula is "
            f"satisfiable iff the graph has a vertex cover of size at most k = {k}"
        )
        steps.append(
            f"Size summary: 2n + 3m = {len(vertices)} nodes and n + 6m = {len(edges)} edges "
            f"({len(variable_edges)} variable + {len(triangle_edges)} triangle + "
            f"{len(connecting_edges)} connecting), k = {k}; O(1) work per variable and per "
            f"clause, so the reduction runs in polynomial time, O(n + m)"
        )

        return ReductionResult(
            source_instance=(clauses, n_vars),
            target_instance=((vertices, edges), k),
            steps=steps,
            node_labels=labels,
        )

    def map_solution_forward(
        self, instance: Any, solution: Dict[int, bool], result: ReductionResult
    ) -> Set[str]:
        """Turn a satisfying assignment into a vertex cover of size exactly k.

        The cover takes, for each variable, the node of its true literal
        (``"x{i}"`` if ``x_i`` is True, else ``"¬x{i}"``), and in each
        clause triangle the two corners other than the *first* corner whose
        literal is true. That is n + 2m = k nodes. It covers every edge:
        each variable edge and triangle has a chosen node, a connecting edge
        from a chosen corner is covered by the corner, and the connecting
        edge from the left-out corner ends at a true literal node, which is
        in the cover.

        A variable missing from ``solution`` (or mapped to ``None``) is
        treated as False, as the SAT solvers do when they complete a
        partial assignment; the assignment must already satisfy every
        clause through its assigned variables.

        Args:
            instance: The 3-SAT instance ``(cnf, n_vars)`` that was
                transformed.
            solution: A ``dict`` assignment that satisfies the formula.
            result: The :class:`ReductionResult` of transforming
                ``instance``.

        Returns:
            A new ``set`` of node names, a vertex cover of the reduction
            graph with exactly ``k = n + 2m`` nodes.

        Raises:
            TypeError: If ``solution`` is not a ``dict`` or ``result`` is
                not a :class:`ReductionResult`.
            ValueError: If ``solution`` does not satisfy the formula (then
                no cover of size k exists to map it to), or ``result`` was
                not produced from ``instance``.

        Time Complexity:
            O(n + m).

        Space Complexity:
            O(n + m) for the cover.

        Examples:
            >>> sipser = ([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2)
            >>> reduction = ThreeSatToVertexCover()
            >>> result = reduction.transform(sipser)
            >>> sorted(reduction.map_solution_forward(sipser, {1: False, 2: True}, result))
            ['c1_1', 'c1_2', 'c2_2', 'c2_3', 'c3_2', 'c3_3', 'x2', '¬x1']
            >>> reduction.map_solution_forward(sipser, {1: True, 2: True}, result)
            Traceback (most recent call last):
                ...
            ValueError: the assignment does not satisfy the formula, so there is no cover of size k to map it to
        """
        self._require_result_for(instance, result)
        clauses, n_vars = _parse_cnf_instance(instance)
        if not isinstance(solution, dict):
            raise TypeError(f"solution must be a dict assignment, got {type(solution).__name__}")
        if not is_satisfying(clauses, solution):
            raise ValueError(
                "the assignment does not satisfy the formula, so there is no cover "
                "of size k to map it to"
            )
        value = {i: solution.get(i) is not None and bool(solution.get(i))
                 for i in range(1, n_vars + 1)}

        def is_true(literal: int) -> bool:
            return value[literal] if literal > 0 else not value[-literal]

        cover: Set[str] = {_literal_node(i if value[i] else -i) for i in range(1, n_vars + 1)}
        for j, clause in enumerate(clauses, start=1):
            padded = _pad_clause(clause)
            left_out = next(p for p, literal in enumerate(padded, start=1) if is_true(literal))
            cover.update(f"c{j}_{p}" for p in range(1, _CLAUSE_WIDTH + 1) if p != left_out)
        return cover

    def map_solution_back(
        self, instance: Any, solution: Iterable[str], result: ReductionResult
    ) -> Dict[int, bool]:
        """Read a satisfying assignment off a vertex cover of size at most k.

        Sets ``x_i = True`` iff node ``"x{i}"`` is in the cover.

        **Why the assignment satisfies the formula.** The n variable edges
        and the m clause triangles share no nodes. A vertex cover needs at
        least one endpoint of each variable edge and at least two corners of
        each triangle (one corner can cover only two of the three triangle
        edges), so every cover has at least n + 2m = k nodes. A cover with
        at most k nodes therefore takes *exactly* one node per variable edge
        and *exactly* two corners per triangle. So for every variable
        exactly one of ``"x{i}"`` and ``"¬x{i}"`` is in the cover, and the
        one in the cover is the node of a true literal under the assignment
        read off. In clause j, let ``c{j}_{p}`` be the corner left out. Its
        connecting edge goes to the node of the literal in position p, and
        since the corner is not in the cover, that literal's node must be.
        A literal whose node is in the cover is true, so clause j is
        satisfied, and this holds for every clause.

        Args:
            instance: The 3-SAT instance ``(cnf, n_vars)`` that was
                transformed.
            solution: A vertex cover of the reduction graph with at most
                ``k`` nodes, as an iterable of node names.
            result: The :class:`ReductionResult` of transforming
                ``instance``.

        Returns:
            A new ``dict`` mapping every variable ``1 .. n_vars`` to a
            ``bool``; it satisfies the formula.

        Raises:
            TypeError: If ``solution`` is a ``str`` or not iterable, or
                ``result`` is not a :class:`ReductionResult`.
            ValueError: If ``solution`` names a node that is not in the
                graph, is not a vertex cover, has more than ``k`` nodes (the
                proof needs at most k; a larger cover may read back as a
                non-satisfying assignment), or ``result`` was not produced
                from ``instance``.

        Time Complexity:
            O(n + m + |solution|): the cover check reads each of the n + 6m
            edges once.

        Space Complexity:
            O(n + m).

        Examples:
            >>> sipser = ([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2)
            >>> reduction = ThreeSatToVertexCover()
            >>> result = reduction.transform(sipser)
            >>> cover = {"¬x1", "x2", "c1_1", "c1_2", "c2_2", "c2_3", "c3_2", "c3_3"}
            >>> reduction.map_solution_back(sipser, cover, result)
            {1: False, 2: True}
            >>> reduction.map_solution_back(sipser, cover - {"x2"}, result)
            Traceback (most recent call last):
                ...
            ValueError: the solution is not a vertex cover of the reduction graph
        """
        self._require_result_for(instance, result)
        _, n_vars = _parse_cnf_instance(instance)
        (vertices, edges), k = result.target_instance
        chosen = _vertex_subset(solution, "cover", vertices)
        if not is_vertex_cover(edges, chosen):
            raise ValueError("the solution is not a vertex cover of the reduction graph")
        if len(chosen) > k:
            raise ValueError(
                f"the cover has {len(chosen)} nodes but k = {k}; only a cover of size at "
                f"most k is guaranteed to map back to a satisfying assignment"
            )
        return {i: _literal_node(i) in chosen for i in range(1, n_vars + 1)}


# --- Vertex Cover -> Independent Set ---


class VertexCoverToIndependentSet(Reduction):
    """Vertex Cover to Independent Set: the same graph, k becomes |V| - k.

    S is a vertex cover of G = (V, E) iff V - S is an independent set: an
    edge with both endpoints outside S is exactly an edge S fails to cover.
    So G has a vertex cover of size at most k iff it has an independent set
    of size at least |V| - k. The graph is unchanged, so the reduction is
    O(V + E), and certificates map by taking complements in both
    directions.

    Attributes:
        name: ``"Vertex Cover -> Independent Set"``.

    Time Complexity:
        O(V + E) for :meth:`transform` and for each certificate map.

    Space Complexity:
        O(V + E) for the copied graph.

    Examples:
        >>> reduction = VertexCoverToIndependentSet()
        >>> path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
        >>> result = reduction.transform(path)
        >>> result.target_instance[1]
        2
        >>> sorted(reduction.map_solution_forward(path, {2, 3}, result))
        [1, 4]
    """

    name = "Vertex Cover -> Independent Set"

    def _comparable(self, instance: Any) -> Any:
        """Return ``(vertex set, edge set, k)`` for equality checks.

        Args:
            instance: A graph instance ``((vertices, edges), k)``.

        Returns:
            A normalized form that ignores vertex and edge order.

        Raises:
            TypeError: As :func:`_parse_graph_instance`.
            ValueError: As :func:`_parse_graph_instance`.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V + E).

        Examples:
            >>> VertexCoverToIndependentSet()._comparable((([2, 1], [(2, 1)]), 1))
            (frozenset({1, 2}), frozenset({(1, 2)}), 1)
        """
        vertex_list, edges, k = _parse_graph_instance(instance)
        return frozenset(vertex_list), frozenset(edges), k

    def transform(self, instance: Any) -> ReductionResult:
        """Map ``((V, E), k)`` for Vertex Cover to ``((V, E), |V| - k)`` for Independent Set.

        Args:
            instance: ``((vertices, edges), k)``: a simple undirected graph
                and the cover bound, ``0 <= k <= |V|``.

        Returns:
            A :class:`ReductionResult` whose ``target_instance`` is
            ``((vertices, edges), len(vertices) - k)`` on the same
            (normalized) graph. The log has the source sizes, the graph
            step, the target bound and a size summary.

        Raises:
            TypeError: If the instance is malformed (see
                :func:`_parse_graph_instance`).
            ValueError: If an edge is malformed, a self-loop or has an
                endpoint outside ``vertices``, or ``k`` is negative or
                larger than ``|V|`` (a bound above |V| is always met by V
                itself; pass ``k = |V|`` instead).

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V + E).

        Examples:
            >>> reduction = VertexCoverToIndependentSet()
            >>> result = reduction.transform((([1, 2, 3], {(1, 2), (2, 3)}), 1))
            >>> (vertices, edges), k = result.target_instance
            >>> vertices, sorted(edges), k
            ([1, 2, 3], [(1, 2), (2, 3)], 2)
            >>> print(result.steps[2])
            Target: k' = |V| - k = 3 - 1 = 2; G has a vertex cover of size at most 1 iff G has an independent set of size at least 2
            >>> reduction.transform((([1, 2], {(1, 2)}), 3))
            Traceback (most recent call last):
                ...
            ValueError: k = 3 is larger than |V| = 2
        """
        vertex_list, edges, k = _parse_graph_instance(instance)
        n_vertices, n_edges = len(vertex_list), len(edges)
        if k > n_vertices:
            raise ValueError(f"k = {k} is larger than |V| = {n_vertices}")
        target_k = n_vertices - k
        steps = [
            f"{self.name}: G has |V| = {n_vertices} nodes and |E| = {n_edges} edges; "
            f"vertex cover bound k = {k}",
            f"Graph: unchanged, the same {n_vertices} nodes and {n_edges} edges; S is a "
            f"vertex cover of G iff V - S is an independent set of G (an edge with both "
            f"ends outside S is exactly an edge S fails to cover)",
            f"Target: k' = |V| - k = {n_vertices} - {k} = {target_k}; G has a vertex cover "
            f"of size at most {k} iff G has an independent set of size at least {target_k}",
            f"Size summary: {n_vertices} nodes, {n_edges} edges, k' = {target_k}; "
            f"O(V + E) time, polynomial",
        ]
        labels = {
            vertex: f"vertex {vertex} of G, unchanged (in the independent set iff not in the cover)"
            for vertex in vertex_list
        }
        return ReductionResult(
            source_instance=((list(vertex_list), set(edges)), k),
            target_instance=((list(vertex_list), set(edges)), target_k),
            steps=steps,
            node_labels=labels,
        )

    def map_solution_forward(
        self, instance: Any, solution: Iterable[Hashable], result: ReductionResult
    ) -> Set[Hashable]:
        """Map a vertex cover S to the independent set V - S.

        Every edge has an endpoint in S, so no edge has both endpoints in
        V - S. A cover of size at most k becomes an independent set of size
        at least |V| - k. The size is not enforced: the complement of any
        cover is independent, so a larger cover (for example a minimum
        cover of a no-instance) still maps to a valid, smaller independent
        set.

        Args:
            instance: The Vertex Cover instance ``((vertices, edges), k)``.
            solution: A vertex cover of the graph, an iterable of vertices.
            result: The :class:`ReductionResult` of transforming
                ``instance``.

        Returns:
            A new ``set``: the vertices not in ``solution``.

        Raises:
            TypeError: If ``solution`` is a ``str`` or not iterable, or
                ``result`` is not a :class:`ReductionResult`.
            ValueError: If ``solution`` names a non-vertex or is not a
                vertex cover, or ``result`` was not produced from
                ``instance``.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V).

        Examples:
            >>> reduction = VertexCoverToIndependentSet()
            >>> star = (([0, 1, 2, 3], {(0, 1), (0, 2), (0, 3)}), 1)
            >>> result = reduction.transform(star)
            >>> sorted(reduction.map_solution_forward(star, {0}, result))
            [1, 2, 3]
            >>> reduction.map_solution_forward(star, {1}, result)
            Traceback (most recent call last):
                ...
            ValueError: the solution is not a vertex cover of the source graph
        """
        self._require_result_for(instance, result)
        vertex_list, edges, _ = _parse_graph_instance(instance)
        chosen = _vertex_subset(solution, "cover", vertex_list)
        if not is_vertex_cover(edges, chosen):
            raise ValueError("the solution is not a vertex cover of the source graph")
        return set(vertex_list) - chosen

    def map_solution_back(
        self, instance: Any, solution: Iterable[Hashable], result: ReductionResult
    ) -> Set[Hashable]:
        """Map an independent set I back to the vertex cover V - I.

        No edge has both endpoints in I, so every edge has an endpoint in
        V - I. An independent set of size at least |V| - k becomes a cover
        of size at most k.

        Args:
            instance: The Vertex Cover instance ``((vertices, edges), k)``.
            solution: An independent set of the target graph.
            result: The :class:`ReductionResult` of transforming
                ``instance``.

        Returns:
            A new ``set``: the vertices not in ``solution``.

        Raises:
            TypeError: If ``solution`` is a ``str`` or not iterable, or
                ``result`` is not a :class:`ReductionResult`.
            ValueError: If ``solution`` names a non-vertex or is not
                independent in the target graph, or ``result`` was not
                produced from ``instance``.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V).

        Examples:
            >>> reduction = VertexCoverToIndependentSet()
            >>> path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
            >>> result = reduction.transform(path)
            >>> sorted(reduction.map_solution_back(path, {1, 4}, result))
            [2, 3]
        """
        self._require_result_for(instance, result)
        (vertex_list, edges), _ = result.target_instance
        chosen = _vertex_subset(solution, "independent set", vertex_list)
        if not is_independent_set(edges, chosen):
            raise ValueError("the solution is not an independent set of the target graph")
        return set(vertex_list) - chosen


# --- Independent Set -> Clique ---


class IndependentSetToClique(Reduction):
    """Independent Set to Clique: the complement graph, k unchanged.

    S is independent in G iff S is a clique in the complement graph, whose
    edges are exactly the pairs of distinct vertices that are *not*
    adjacent in G. So G has an independent set of size at least k iff its
    complement has a clique of size at least k. Building the complement
    examines every vertex pair once, O(V^2), which is polynomial; the
    certificate is the same vertex set in both directions.

    Attributes:
        name: ``"Independent Set -> Clique"``.

    Time Complexity:
        O(V^2 + E) for :meth:`transform` (every vertex pair is examined),
        O(V + E) to map forward and O(V^2) to map back.

    Space Complexity:
        O(V^2) for the complement graph.

    Examples:
        >>> reduction = IndependentSetToClique()
        >>> path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
        >>> (vertices, edges), k = reduction.transform(path).target_instance
        >>> sorted(edges), k
        ([(1, 3), (1, 4), (2, 4)], 2)
    """

    name = "Independent Set -> Clique"

    def _comparable(self, instance: Any) -> Any:
        """Return ``(vertex set, edge set, k)`` for equality checks.

        Args:
            instance: A graph instance ``((vertices, edges), k)``.

        Returns:
            A normalized form that ignores vertex and edge order.

        Raises:
            TypeError: As :func:`_parse_graph_instance`.
            ValueError: As :func:`_parse_graph_instance`.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V + E).

        Examples:
            >>> IndependentSetToClique()._comparable((([1], set()), 1))
            (frozenset({1}), frozenset(), 1)
        """
        vertex_list, edges, k = _parse_graph_instance(instance)
        return frozenset(vertex_list), frozenset(edges), k

    def transform(self, instance: Any) -> ReductionResult:
        """Map ``((V, E), k)`` for Independent Set to ``((V, complement(E)), k)`` for Clique.

        Args:
            instance: ``((vertices, edges), k)``: a simple undirected graph
                and the independent-set bound ``k >= 0``.

        Returns:
            A :class:`ReductionResult` whose ``target_instance`` is
            ``((vertices, complement(vertices, edges)), k)``. The
            complement has ``|V| (|V| - 1) / 2 - |E|`` edges.

        Raises:
            TypeError: If the instance is malformed (see
                :func:`_parse_graph_instance`).
            ValueError: If an edge is malformed, a self-loop or has an
                endpoint outside ``vertices``, or ``k`` is negative.

        Time Complexity:
            O(V^2 + E): every vertex pair is examined once.

        Space Complexity:
            O(V^2): the complement of a sparse graph is dense.

        Examples:
            >>> reduction = IndependentSetToClique()
            >>> result = reduction.transform((([0, 1, 2], {(0, 1)}), 2))
            >>> sorted(result.target_instance[0][1])
            [(0, 2), (1, 2)]
            >>> print(result.steps[1])
            Complement graph: the same 3 nodes, with an edge between every two distinct vertices that are not adjacent in G: 3 * 2 / 2 - 1 = 2 edges; S is independent in G iff S is a clique in the complement
        """
        vertex_list, edges, k = _parse_graph_instance(instance)
        n_vertices, n_edges = len(vertex_list), len(edges)
        target_edges = complement(vertex_list, edges)
        n_target = len(target_edges)
        steps = [
            f"{self.name}: G has |V| = {n_vertices} nodes and |E| = {n_edges} edges; "
            f"independent set bound k = {k}",
            f"Complement graph: the same {n_vertices} nodes, with an edge between every two "
            f"distinct vertices that are not adjacent in G: {n_vertices} * "
            f"{max(n_vertices - 1, 0)} / 2 - {n_edges} = {n_target} edges; S is independent "
            f"in G iff S is a clique in the complement",
            f"Target: k = {k}, unchanged; G has an independent set of size at least {k} iff "
            f"the complement has a clique of size at least {k}",
            f"Size summary: {n_vertices} nodes, {n_target} edges, k = {k}; O(V^2) time, "
            f"polynomial",
        ]
        labels = {
            vertex: f"vertex {vertex} of G, unchanged (adjacent here iff not adjacent in G)"
            for vertex in vertex_list
        }
        return ReductionResult(
            source_instance=((list(vertex_list), set(edges)), k),
            target_instance=((list(vertex_list), target_edges), k),
            steps=steps,
            node_labels=labels,
        )

    def map_solution_forward(
        self, instance: Any, solution: Iterable[Hashable], result: ReductionResult
    ) -> Set[Hashable]:
        """Map an independent set of G to the same set, a clique of the complement.

        Args:
            instance: The Independent Set instance ``((vertices, edges), k)``.
            solution: An independent set of G.
            result: The :class:`ReductionResult` of transforming
                ``instance``.

        Returns:
            A new ``set`` holding the same vertices.

        Raises:
            TypeError: If ``solution`` is a ``str`` or not iterable, or
                ``result`` is not a :class:`ReductionResult`.
            ValueError: If ``solution`` names a non-vertex or is not
                independent in G, or ``result`` was not produced from
                ``instance``.

        Time Complexity:
            O(V + E).

        Space Complexity:
            O(V).

        Examples:
            >>> reduction = IndependentSetToClique()
            >>> path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
            >>> result = reduction.transform(path)
            >>> sorted(reduction.map_solution_forward(path, {1, 4}, result))
            [1, 4]
        """
        self._require_result_for(instance, result)
        vertex_list, edges, _ = _parse_graph_instance(instance)
        chosen = _vertex_subset(solution, "independent set", vertex_list)
        if not is_independent_set(edges, chosen):
            raise ValueError("the solution is not an independent set of the source graph")
        return set(chosen)

    def map_solution_back(
        self, instance: Any, solution: Iterable[Hashable], result: ReductionResult
    ) -> Set[Hashable]:
        """Map a clique of the complement back to the same set, independent in G.

        Args:
            instance: The Independent Set instance ``((vertices, edges), k)``.
            solution: A clique of the complement graph.
            result: The :class:`ReductionResult` of transforming
                ``instance``.

        Returns:
            A new ``set`` holding the same vertices.

        Raises:
            TypeError: If ``solution`` is a ``str`` or not iterable, or
                ``result`` is not a :class:`ReductionResult`.
            ValueError: If ``solution`` names a non-vertex or is not a
                clique of the complement graph, or ``result`` was not
                produced from ``instance``.

        Time Complexity:
            O(V^2): the clique check looks up every pair of members.

        Space Complexity:
            O(V^2) for the normalized complement edge set.

        Examples:
            >>> reduction = IndependentSetToClique()
            >>> path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
            >>> result = reduction.transform(path)
            >>> sorted(reduction.map_solution_back(path, {1, 3}, result))
            [1, 3]
            >>> reduction.map_solution_back(path, {1, 2}, result)
            Traceback (most recent call last):
                ...
            ValueError: the solution is not a clique of the complement graph
        """
        self._require_result_for(instance, result)
        (vertex_list, edges), _ = result.target_instance
        chosen = _vertex_subset(solution, "clique", vertex_list)
        if not is_clique(edges, chosen):
            raise ValueError("the solution is not a clique of the complement graph")
        return set(chosen)


def reduce_chain(instance: Any, reductions: Iterable[Any]) -> List[ReductionResult]:
    """Apply several reductions in order, each to the previous target instance.

    ``reduce_chain((cnf, n), [ThreeSatToVertexCover(),
    VertexCoverToIndependentSet(), IndependentSetToClique()])`` performs
    3-SAT -> Vertex Cover -> Independent Set -> Clique: the first
    reduction transforms ``instance`` and every later one transforms the
    ``target_instance`` of the one before. A composition of
    polynomial-time reductions is a polynomial-time reduction, so the
    whole chain is one, and the per-stage results keep every step log and
    node map so each stage can be explained and its certificates mapped
    separately.

    Args:
        instance: The source instance of the first reduction.
        reductions: An iterable of :class:`Reduction` instances (a
            :class:`Reduction` subclass is also accepted and instantiated
            with no arguments).

    Returns:
        A new list with one :class:`ReductionResult` per reduction, in
        order; ``[]`` for no reductions.

    Raises:
        TypeError: If ``reductions`` is not iterable or holds something
            that is not a :class:`Reduction`, or a stage rejects its input
            type.
        ValueError: If a stage rejects its input (see each
            ``transform``).

    Time Complexity:
        The sum of the stages' costs; for 3-SAT -> VC -> IS -> Clique,
        O((n + m)^2), dominated by the complement graph on 2n + 3m nodes.

    Space Complexity:
        The total size of all the stages' results, O((n + m)^2) for that
        chain.

    Examples:
        >>> chain = [ThreeSatToVertexCover(), VertexCoverToIndependentSet(),
        ...          IndependentSetToClique()]
        >>> results = reduce_chain(([[1, 1, 2], [-1, -2, -2], [-1, 2, 2]], 2), chain)
        >>> [len(r.target_instance[0][1]) for r in results]
        [20, 20, 58]
        >>> [r.target_instance[1] for r in results]
        [8, 5, 5]
        >>> reduce_chain(([[1]], 1), [])
        []
        >>> reduce_chain(([[1]], 1), ["not a reduction"])
        Traceback (most recent call last):
            ...
        TypeError: reductions must be Reduction instances, got str
    """
    if isinstance(reductions, (str, bytes)):
        raise TypeError("reductions must be an iterable of Reduction instances, got str")
    try:
        stages = list(reductions)
    except TypeError:
        raise TypeError(
            f"reductions must be an iterable of Reduction instances, "
            f"got {type(reductions).__name__}"
        ) from None
    resolved: List[Reduction] = []
    for stage in stages:
        if isinstance(stage, type) and issubclass(stage, Reduction):
            stage = stage()
        if not isinstance(stage, Reduction):
            raise TypeError(
                f"reductions must be Reduction instances, got {type(stage).__name__}"
            )
        resolved.append(stage)
    results: List[ReductionResult] = []
    current = instance
    for stage in resolved:
        result = stage.transform(current)
        results.append(result)
        current = result.target_instance
    return results


# --- Verifiers and the exact vertex cover solver ---
#
# ``is_vertex_cover`` is the graph verifier from src.utils.graph_utils,
# imported above and re-exported through __all__ so callers can take both
# verifiers from this module.


def is_satisfying(cnf: Sequence[Sequence[int]], assignment: Dict[int, bool]) -> bool:
    """Check whether an assignment satisfies a CNF formula (the SAT verifier).

    A CNF formula is a list of clauses in DIMACS style: the literal ``i``
    means variable ``i`` is true, ``-i`` means it is false, and variables are
    numbered from 1. A clause is satisfied when at least one of its literals
    is made true, and the formula when every clause is. Checking that takes
    one pass over the literals, which is why SAT is in NP (Cook, 1971);
    finding a satisfying assignment is the NP-complete part.

    An unassigned variable (a missing key, or the value ``None``) makes
    *neither* of its literals true, so a partial assignment satisfies the
    formula only if the assigned variables already satisfy every clause.
    An empty clause is an empty disjunction and is never satisfied; an empty
    formula is satisfied by every assignment. Values are read with
    ``bool()``. Every literal is read once (no early exit), so malformed
    input is always reported and the cost is the same for a yes and a no.

    Args:
        cnf: A list (or tuple) of clauses, each a list (or tuple) of non-zero
            ``int`` literals.
        assignment: A ``dict`` mapping variable numbers to truth values.

    Returns:
        ``True`` if every clause contains a literal made true by
        ``assignment``, else ``False``.

    Raises:
        TypeError: If ``cnf`` or a clause is not a list or tuple, a literal
            is not an ``int`` (``bool`` is rejected), or ``assignment`` is not
            a ``dict``.
        ValueError: If a literal is 0 (0 is the DIMACS clause terminator,
            not a variable).

    Time Complexity:
        O(L) where L is the total number of literals in the formula.

    Space Complexity:
        O(1) beyond the input.

    Examples:
        >>> sipser = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]
        >>> is_satisfying(sipser, {1: False, 2: True})
        True
        >>> is_satisfying(sipser, {1: True, 2: True})
        False
        >>> is_satisfying([[1, 2]], {2: True})
        True
        >>> is_satisfying([[1, -2]], {})
        False
        >>> is_satisfying([], {})
        True
        >>> is_satisfying([[]], {1: True})
        False
        >>> is_satisfying([[1, 0]], {1: True})
        Traceback (most recent call last):
            ...
        ValueError: literal 0 is not allowed (variables are numbered from 1)
    """
    if not isinstance(cnf, (list, tuple)):
        raise TypeError(f"cnf must be a list of clauses, got {type(cnf).__name__}")
    if not isinstance(assignment, dict):
        raise TypeError(
            f"assignment must be a dict, got {type(assignment).__name__}"
        )
    all_satisfied = True
    for clause in cnf:
        if not isinstance(clause, (list, tuple)):
            raise TypeError(
                f"each clause must be a list of ints, got {type(clause).__name__}"
            )
        clause_satisfied = False
        for literal in clause:
            if isinstance(literal, bool) or not isinstance(literal, int):
                raise TypeError(
                    f"literals must be ints, got {type(literal).__name__}"
                )
            if literal == 0:
                raise ValueError(
                    "literal 0 is not allowed (variables are numbered from 1)"
                )
            value = assignment.get(literal if literal > 0 else -literal)
            if value is not None and bool(value) == (literal > 0):
                clause_satisfied = True
        if not clause_satisfied:
            all_satisfied = False
    return all_satisfied


def _canonical_vertices(vertices: Iterable[Hashable]) -> List[Hashable]:
    """Return the distinct vertices in a process-independent sorted order.

    The exact solver must be deterministic, and the Week 8 reduction graphs
    use ``str`` vertex names, whose ``set`` iteration order changes from
    process to process (``PYTHONHASHSEED``). Sorting fixes the order.

    Args:
        vertices: A non-string iterable of hashable labels.

    Returns:
        The distinct labels, sorted; labels that cannot be compared with
        ``<`` are sorted by ``(type name, repr)`` instead.

    Raises:
        TypeError: If ``vertices`` is a ``str``/``bytes`` or not iterable.

    Time Complexity:
        O(V log V).

    Space Complexity:
        O(V).

    Examples:
        >>> _canonical_vertices({"c1_1", "x1", "x2"})
        ['c1_1', 'x1', 'x2']
        >>> _canonical_vertices([2, "a", 1])
        [1, 2, 'a']
    """
    if isinstance(vertices, (str, bytes)):
        raise TypeError(
            f"vertices must be an iterable collection, got {type(vertices).__name__}"
        )
    try:
        distinct = list(dict.fromkeys(vertices))
    except TypeError:
        raise TypeError(
            f"vertices must be an iterable collection of hashable labels, "
            f"got {type(vertices).__name__}"
        ) from None
    try:
        return sorted(distinct)
    except TypeError:
        return sorted(distinct, key=lambda item: (type(item).__name__, repr(item)))


def _matching_bound_and_branch(
    adjacency: List[int], alive: int
) -> Tuple[int, int, int]:
    """Scan the uncovered subgraph once for a lower bound and a branch edge.

    The uncovered edges are exactly the edges with both endpoints in
    ``alive`` (the vertices not yet in the partial cover). One pass over the
    alive vertices, lowest index first, does two jobs:

    * builds a greedy maximal matching of the uncovered edges (match each
      still-free vertex to its lowest-index free uncovered neighbour). The
      matched edges share no endpoints, so any cover of the uncovered edges
      needs a distinct vertex for each one: the matching size is a lower
      bound on the vertices still to be added.
    * finds the alive vertex ``u`` with the most uncovered edges (lowest
      index on ties); the branch edge is ``(u, v)`` with ``v`` the
      lowest-index uncovered neighbour of ``u``. Branching on a high-degree
      endpoint first finds small covers early, which tightens the bound.

    Args:
        adjacency: ``adjacency[i]`` is the bitmask of the neighbours of
            vertex ``i``.
        alive: Bitmask of the vertices not in the partial cover.

    Returns:
        ``(matching_size, u, v)``, with ``u == v == -1`` when no uncovered
        edge remains (the partial cover is complete).

    Time Complexity:
        O(V) big-integer operations on V-bit masks.

    Space Complexity:
        O(1) beyond the masks.

    Examples:
        >>> path = [0b0010, 0b0101, 0b1010, 0b0100]   # 0-1-2-3
        >>> _matching_bound_and_branch(path, 0b1111)
        (2, 1, 0)
        >>> _matching_bound_and_branch(path, 0b1010)  # 0 and 2 covered
        (0, -1, -1)
    """
    free = alive
    matching = 0
    best_u = -1
    best_degree = 0
    rest = alive
    while rest:
        low = rest & -rest
        rest ^= low
        i = low.bit_length() - 1
        neighbours = adjacency[i] & alive
        if not neighbours:
            continue
        degree = neighbours.bit_count()
        if degree > best_degree:
            best_degree, best_u = degree, i
        if free & low:
            partners = neighbours & free
            if partners:
                free &= ~(low | (partners & -partners))
                matching += 1
    if best_u < 0:
        return 0, -1, -1
    uncovered = adjacency[best_u] & alive
    best_v = (uncovered & -uncovered).bit_length() - 1
    return matching, best_u, best_v


def min_vertex_cover_exact(vertices: Iterable[Hashable], edges: Iterable[Any]) -> Set[Hashable]:
    """Find a minimum vertex cover by branch and bound on uncovered edges.

    Every vertex cover must contain an endpoint of each edge, so for any
    uncovered edge ``(u, v)`` an optimal cover extending the current partial
    cover either includes ``u`` or includes ``v``. Branching on those two
    choices, depth first, and stopping when no edge is uncovered enumerates
    every minimal cover; a branch is cut as soon as

        ``len(partial) + lower_bound(uncovered edges) >= len(best)``,

    because it cannot then beat the best cover found so far. The lower bound
    is the size of a greedy maximal matching of the still-uncovered edges:
    matched edges share no endpoints, so any cover needs one vertex per
    matched edge. The initial ``best`` is the 2-approximation cover (both
    ends of a greedy maximal matching over the edges in sorted order), which
    is within a factor 2 of optimal from the start.

    Each branch adds one vertex and is cut once it reaches ``len(best)``, so
    the search tree has depth at most ``len(best) <= 2 * OPT`` and at most
    ``2^depth`` leaves: O(2^k * (V + E)) time for a cover of size k, the
    classic bounded-search-tree bound. That is exponential, as expected for
    an NP-hard problem, so the solver is meant for small instances (it
    proves the 30-vertex, 51-edge unsatisfiable 3-SAT anchor graph optimal
    well within a second). The tree is walked with an explicit
    stack rather than recursion, so depth never meets Python's recursion
    limit. Vertices are indexed in sorted order and stored as bitmasks, so
    the result is the same in every process (``str`` labels are not hashed
    into set order).

    Args:
        vertices: The vertices, a non-string iterable of hashable labels.
            Isolated vertices are allowed and never chosen.
        edges: The edges, in any form
            :func:`src.utils.graph_utils.normalize_edges` accepts.

    Returns:
        A new ``set`` of vertices forming a minimum vertex cover, ``set()``
        for an edgeless graph. When several minimum covers exist the same
        one is returned on every run.

    Raises:
        TypeError: If ``vertices`` or ``edges`` is not a non-string iterable.
        ValueError: If an edge is malformed, is a self-loop, or has an
            endpoint that is not in ``vertices``.

    Time Complexity:
        O(2^k * (V + E)) worst case for a minimum cover of size k;
        O(V log V + E) setup.

    Space Complexity:
        O(V + E) for the adjacency masks plus O(k) stack entries.

    Examples:
        >>> sorted(min_vertex_cover_exact([0, 1, 2, 3, 4, 5],
        ...                               {(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)}))
        [0]
        >>> len(min_vertex_cover_exact([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}))
        2
        >>> k33 = {(a, b) for a in range(3) for b in range(3, 6)}
        >>> len(min_vertex_cover_exact(range(6), k33))
        3
        >>> len(min_vertex_cover_exact([0, 1, 2], {(0, 1), (1, 2), (0, 2)}))
        2
        >>> min_vertex_cover_exact([1, 2, 3], set())
        set()
        >>> min_vertex_cover_exact([1, 2], {(1, 3)})
        Traceback (most recent call last):
            ...
        ValueError: edge (1, 3) has an endpoint that is not in vertices
    """
    order = _canonical_vertices(vertices)
    edge_set = normalize_edges(edges)
    index = {vertex: i for i, vertex in enumerate(order)}
    for u, v in edge_set:
        if u not in index or v not in index:
            raise ValueError(f"edge {(u, v)!r} has an endpoint that is not in vertices")
    if not edge_set:
        return set()

    pairs = sorted(
        (min(index[u], index[v]), max(index[u], index[v])) for u, v in edge_set
    )
    adjacency = [0] * len(order)
    for i, j in pairs:
        adjacency[i] |= 1 << j
        adjacency[j] |= 1 << i

    # Initial best: the 2-approximation, both ends of a greedy maximal
    # matching taken over the edges in sorted order.
    best_mask = 0
    for i, j in pairs:
        if not (best_mask >> i) & 1 and not (best_mask >> j) & 1:
            best_mask |= (1 << i) | (1 << j)
    best_size = best_mask.bit_count()

    everyone = (1 << len(order)) - 1
    stack: List[Tuple[int, int]] = [(0, 0)]   # (partial cover mask, its size)
    while stack:
        cover_mask, size = stack.pop()
        bound, u, v = _matching_bound_and_branch(adjacency, everyone & ~cover_mask)
        if u < 0:
            # No uncovered edge: the partial cover is a cover.
            if size < best_size:
                best_mask, best_size = cover_mask, size
            continue
        if size + bound >= best_size:
            continue
        # Push "include v" first so "include u" (the high-degree end) is
        # explored first.
        stack.append((cover_mask | (1 << v), size + 1))
        stack.append((cover_mask | (1 << u), size + 1))

    return {order[i] for i in range(len(order)) if (best_mask >> i) & 1}
