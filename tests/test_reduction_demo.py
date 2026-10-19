"""Tests for the Week 8 reduction framework: 3-SAT -> Vertex Cover -> Independent Set -> Clique.

A reduction is correct when it preserves the yes/no answer, runs in
polynomial time, and (for this framework) carries certificates both ways.
Every expected value below was worked out by hand from the construction
in the spec (section 4), never read back from the module:

* **The Sipser anchor** ``[[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]`` with
  n = 2, m = 3 gives 2n + 3m = 13 nodes, n + 6m = 20 edges and
  k = n + 2m = 8. The 13 node names and all 20 edges are listed by hand in
  :data:`SIPSER_EDGES`. Brute force over the four assignments (in this
  file) shows the only model is x1 = False, x2 = True. The exact minimum
  cover is 8. The forward cover is worked out by hand: the true literal
  nodes are ``¬x1`` and ``x2``; clause 1's first true position is 3 (x2),
  clauses 2 and 3 have their first true position at 1 (not x1), so the
  cover is ``{¬x1, x2, c1_1, c1_2, c2_2, c2_3, c3_2, c3_3}``.
* **The unsatisfiable anchor**, all eight sign patterns over x1, x2, x3:
  2*3 + 3*8 = 30 nodes, 3 + 6*8 = 51 edges, k = 3 + 2*8 = 19, and the exact
  minimum cover is 20, so no cover of size at most k exists.
* **Padding anchors.** ``[[1]]`` with n = 1: 5 nodes, 7 edges, k = 3, and
  a cover of size 3 ({x1, two triangle corners}). ``[[1], [-1]]`` with
  n = 1: 8 nodes, 13 edges, k = 5, but the minimum cover is 6 (whichever of
  x1 and ``¬x1`` the cover takes, the other clause's triangle needs all
  three corners). Padding with a repeated literal still gives three
  distinct clause nodes, so the triangle and connecting edges are distinct
  pairs and n + 6m holds.
* **The agreement battery.** 200 formulas from ``make_rng(derive_seed(2026,
  "reduction-battery", i))``, n in [2, 4], m in [1, 5], clauses of 1 to 3
  literals (so padding, repeated literals and tautological clauses all
  occur): satisfiable by brute force iff the exact minimum cover is at most
  k, the sizes are 2n + 3m and n + 6m for every formula, and certificates
  map both ways.
* **The chain.** VC -> IS -> Clique, checked with
  :mod:`src.utils.graph_utils`: the complement of a cover is independent,
  and an independent set is a clique of the complement graph. On the
  Sipser graph the complement has 13 * 12 / 2 - 20 = 58 edges and the
  bounds are k = 8, then 13 - 8 = 5, then 5.
* **Invalid input** (literal 0, variable out of range, empty clause, a
  clause of four literals, a non-int literal) is rejected.

No assertion is about time. Every random draw is seeded.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Sequence, Tuple

import pytest

from src.complexity.reduction_demo import (
    IndependentSetToClique,
    Reduction,
    ReductionResult,
    ThreeSatToVertexCover,
    VertexCoverToIndependentSet,
    min_vertex_cover_exact,
    reduce_chain,
)
from src.utils.graph_utils import (
    complement,
    is_clique,
    is_independent_set,
    is_vertex_cover,
    normalize_edges,
)
from src.utils.random_utils import derive_seed, make_rng

LOGGER_NAME = "src.complexity.reduction_demo"

#: Sipser's example formula (Theorem 7.44), n = 2, m = 3.
SIPSER: List[List[int]] = [[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]
SIPSER_N = 2

#: Every node of the Sipser reduction graph, by hand.
SIPSER_NODES = {
    "x1", "¬x1", "x2", "¬x2",
    "c1_1", "c1_2", "c1_3",
    "c2_1", "c2_2", "c2_3",
    "c3_1", "c3_2", "c3_3",
}

#: Every edge of the Sipser reduction graph, by hand: 2 variable edges,
#: 9 triangle edges, 9 connecting edges.
SIPSER_EDGES = {
    ("x1", "¬x1"), ("x2", "¬x2"),
    ("c1_1", "c1_2"), ("c1_1", "c1_3"), ("c1_2", "c1_3"),
    ("c2_1", "c2_2"), ("c2_1", "c2_3"), ("c2_2", "c2_3"),
    ("c3_1", "c3_2"), ("c3_1", "c3_3"), ("c3_2", "c3_3"),
    ("c1_1", "x1"), ("c1_2", "x1"), ("c1_3", "x2"),
    ("c2_1", "¬x1"), ("c2_2", "¬x2"), ("c2_3", "¬x2"),
    ("c3_1", "¬x1"), ("c3_2", "x2"), ("c3_3", "x2"),
}

#: The forward cover for x1 = False, x2 = True, by hand (see module docstring).
SIPSER_FORWARD_COVER = {
    "¬x1", "x2", "c1_1", "c1_2", "c2_2", "c2_3", "c3_2", "c3_3",
}

#: All eight sign patterns over x1, x2, x3: every assignment falsifies one clause.
UNSAT_3: List[List[int]] = [
    [s1 * 1, s2 * 2, s3 * 3] for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1)
]

BATTERY_SEED = 2026
BATTERY_SIZE = 200


# ----------------------------------------------------------------------
# Helpers that share no code with the module under test
# ----------------------------------------------------------------------
def satisfied_by(cnf: Sequence[Sequence[int]], values: Dict[int, bool]) -> bool:
    """Evaluate a CNF formula under a complete assignment, written out longhand."""
    for clause in cnf:
        if not any(values[abs(literal)] == (literal > 0) for literal in clause):
            return False
    return True


def brute_force_models(cnf: Sequence[Sequence[int]], n_vars: int) -> List[Dict[int, bool]]:
    """Every satisfying assignment, by trying all 2^n of them."""
    models = []
    for bits in itertools.product((False, True), repeat=n_vars):
        values = {i + 1: bits[i] for i in range(n_vars)}
        if satisfied_by(cnf, values):
            models.append(values)
    return models


def battery_formula(index: int) -> Tuple[List[List[int]], int]:
    """Formula ``index`` of the seeded agreement battery."""
    rng = make_rng(derive_seed(BATTERY_SEED, "reduction-battery", index))
    n_vars = rng.randint(2, 4)
    m = rng.randint(1, 5)
    cnf = []
    for _ in range(m):
        width = rng.randint(1, 3)
        cnf.append([rng.choice((1, -1)) * rng.randint(1, n_vars) for _ in range(width)])
    return cnf, n_vars


@pytest.fixture(scope="module")
def battery() -> List[dict]:
    """The 200 battery formulas with their reduction, exact cover and models."""
    reduction = ThreeSatToVertexCover()
    cases = []
    for index in range(BATTERY_SIZE):
        cnf, n_vars = battery_formula(index)
        result = reduction.transform((cnf, n_vars))
        (vertices, edges), k = result.target_instance
        cases.append({
            "index": index,
            "cnf": cnf,
            "n": n_vars,
            "m": len(cnf),
            "result": result,
            "vertices": vertices,
            "edges": edges,
            "k": k,
            "min_cover": min_vertex_cover_exact(vertices, edges),
            "models": brute_force_models(cnf, n_vars),
        })
    return cases


@pytest.fixture(scope="module")
def sipser_result() -> ReductionResult:
    """The 3-SAT -> Vertex Cover reduction of the Sipser formula."""
    return ThreeSatToVertexCover().transform((SIPSER, SIPSER_N))


# ----------------------------------------------------------------------
# The Sipser anchor
# ----------------------------------------------------------------------
class TestSipserAnchor:
    """``[[1, 1, 2], [-1, -2, -2], [-1, 2, 2]]``: 13 nodes, 20 edges, k = 8."""

    def test_graph_has_13_nodes_20_edges_and_k_8(self, sipser_result):
        (vertices, edges), k = sipser_result.target_instance
        assert len(vertices) == 13
        assert len(set(vertices)) == 13
        assert len(edges) == 20
        assert k == 8

    def test_nodes_and_edges_match_the_hand_construction(self, sipser_result):
        (vertices, edges), _ = sipser_result.target_instance
        assert set(vertices) == SIPSER_NODES
        assert normalize_edges(edges) == normalize_edges(SIPSER_EDGES)

    def test_only_satisfying_assignment_is_x1_false_x2_true(self):
        assert brute_force_models(SIPSER, SIPSER_N) == [{1: False, 2: True}]

    def test_exact_minimum_vertex_cover_is_8(self, sipser_result):
        (vertices, edges), _ = sipser_result.target_instance
        cover = min_vertex_cover_exact(vertices, edges)
        assert len(cover) == 8
        assert is_vertex_cover(edges, cover)

    def test_forward_cover_is_valid_with_size_8(self, sipser_result):
        reduction = ThreeSatToVertexCover()
        cover = reduction.map_solution_forward(
            (SIPSER, SIPSER_N), {1: False, 2: True}, sipser_result
        )
        (_, edges), _ = sipser_result.target_instance
        assert cover == SIPSER_FORWARD_COVER
        assert len(cover) == 8
        assert is_vertex_cover(edges, cover)

    def test_back_from_forward_cover_satisfies(self, sipser_result):
        reduction = ThreeSatToVertexCover()
        assignment = reduction.map_solution_back(
            (SIPSER, SIPSER_N), SIPSER_FORWARD_COVER, sipser_result
        )
        assert assignment == {1: False, 2: True}
        assert satisfied_by(SIPSER, assignment)

    def test_back_from_exact_minimum_cover_satisfies(self, sipser_result):
        reduction = ThreeSatToVertexCover()
        (vertices, edges), _ = sipser_result.target_instance
        cover = min_vertex_cover_exact(vertices, edges)
        assignment = reduction.map_solution_back((SIPSER, SIPSER_N), cover, sipser_result)
        assert set(assignment) == {1, 2}
        assert satisfied_by(SIPSER, assignment)
        # The model is unique, so any cover of size k must read back as it.
        assert assignment == {1: False, 2: True}

    def test_node_labels_describe_every_node(self, sipser_result):
        labels = sipser_result.node_labels
        assert set(labels) == SIPSER_NODES
        assert all(isinstance(text, str) and text for text in labels.values())
        assert "clause 2" in labels["c2_1"] and "position 1" in labels["c2_1"]
        assert "x1" in labels["c2_1"]
        assert "x2" in labels["¬x2"]

    def test_source_instance_is_the_formula(self, sipser_result):
        cnf, n_vars = sipser_result.source_instance
        assert [list(clause) for clause in cnf] == SIPSER
        assert n_vars == SIPSER_N


# ----------------------------------------------------------------------
# The unsatisfiable anchor and the small padding anchors
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def unsat():
    """The unsatisfiable anchor's reduction, graph, k and exact minimum cover."""
    result = ThreeSatToVertexCover().transform((UNSAT_3, 3))
    (vertices, edges), k = result.target_instance
    return result, vertices, edges, k, min_vertex_cover_exact(vertices, edges)


class TestUnsatisfiableAnchor:
    """All eight sign patterns over x1..x3: 30 nodes, 51 edges, k = 19, min cover 20."""

    def test_formula_has_eight_clauses_and_no_model(self):
        assert len(UNSAT_3) == 8
        assert brute_force_models(UNSAT_3, 3) == []

    def test_graph_has_30_nodes_51_edges_and_k_19(self, unsat):
        _, vertices, edges, k, _ = unsat
        assert len(vertices) == 30
        assert len(set(vertices)) == 30
        assert len(edges) == 51
        assert k == 19

    def test_exact_minimum_cover_is_20_so_none_of_size_k(self, unsat):
        _, _, edges, k, cover = unsat
        assert len(cover) == 20
        assert is_vertex_cover(edges, cover)
        assert len(cover) > k

    def test_back_refuses_a_cover_larger_than_k(self, unsat):
        result, _, _, _, cover = unsat
        with pytest.raises(ValueError):
            ThreeSatToVertexCover().map_solution_back((UNSAT_3, 3), cover, result)

    def test_forward_refuses_every_assignment(self, unsat):
        result = unsat[0]
        for bits in itertools.product((False, True), repeat=3):
            assignment = {1: bits[0], 2: bits[1], 3: bits[2]}
            with pytest.raises(ValueError):
                ThreeSatToVertexCover().map_solution_forward((UNSAT_3, 3), assignment, result)


class TestPaddingAnchors:
    """Short clauses are padded by repeating a literal; the counts still hold."""

    def test_single_literal_clause(self):
        result = ThreeSatToVertexCover().transform(([[1]], 1))
        (vertices, edges), k = result.target_instance
        assert set(vertices) == {"x1", "¬x1", "c1_1", "c1_2", "c1_3"}
        assert normalize_edges(edges) == normalize_edges({
            ("x1", "¬x1"),
            ("c1_1", "c1_2"), ("c1_1", "c1_3"), ("c1_2", "c1_3"),
            ("c1_1", "x1"), ("c1_2", "x1"), ("c1_3", "x1"),
        })
        assert (len(vertices), len(edges), k) == (5, 7, 3)
        assert len(min_vertex_cover_exact(vertices, edges)) == 3

    def test_contradictory_unit_clauses_need_a_cover_of_6(self):
        result = ThreeSatToVertexCover().transform(([[1], [-1]], 1))
        (vertices, edges), k = result.target_instance
        assert (len(vertices), len(edges), k) == (8, 13, 5)
        assert len(min_vertex_cover_exact(vertices, edges)) == 6

    def test_two_literal_clause_is_padded_with_its_last_literal(self):
        result = ThreeSatToVertexCover().transform(([[1, -2]], 2))
        (vertices, edges), k = result.target_instance
        assert (len(vertices), len(edges), k) == (7, 8, 4)
        assert ("c1_2", "¬x2") in normalize_edges(edges)
        assert ("c1_3", "¬x2") in normalize_edges(edges)
        assert "padding" in result.node_labels["c1_3"]
        assert "padding" not in result.node_labels["c1_2"]

    def test_padding_is_reported_in_the_step_log(self):
        reduction = ThreeSatToVertexCover()
        padded = reduction.explain(reduction.transform(([[1], [-1, 2]], 2))).lower()
        assert "padd" in padded
        assert "repeat" in padded

    def test_tautological_clause_keeps_the_counts(self):
        result = ThreeSatToVertexCover().transform(([[1, -1, 2]], 2))
        (vertices, edges), k = result.target_instance
        assert (len(vertices), len(edges), k) == (7, 8, 4)


# ----------------------------------------------------------------------
# The 200-formula agreement battery
# ----------------------------------------------------------------------
class TestAgreementBattery:
    """Satisfiable iff a cover of size at most k exists, on 200 seeded formulas."""

    def test_battery_is_not_degenerate(self, battery):
        # A check on the test data itself: the battery must exercise padding
        # and contain both satisfiable and unsatisfiable formulas.
        widths = [len(clause) for case in battery for clause in case["cnf"]]
        assert len(battery) == 200
        assert {1, 2, 3} <= set(widths)
        assert {case["n"] for case in battery} == {2, 3, 4}
        assert {case["m"] for case in battery} == {1, 2, 3, 4, 5}
        satisfiable = sum(1 for case in battery if case["models"])
        assert 0 < satisfiable < len(battery)

    def test_satisfiable_iff_cover_of_size_at_most_k(self, battery):
        failures = [
            case["index"] for case in battery
            if bool(case["models"]) != (len(case["min_cover"]) <= case["k"])
        ]
        assert failures == []

    def test_node_and_edge_counts_are_2n_plus_3m_and_n_plus_6m(self, battery):
        for case in battery:
            n, m = case["n"], case["m"]
            vertices, edges = case["vertices"], case["edges"]
            assert len(vertices) == 2 * n + 3 * m, case["index"]
            assert len(set(vertices)) == len(vertices), case["index"]
            assert len(edges) == n + 6 * m, case["index"]
            assert normalize_edges(edges) == set(edges), case["index"]
            assert all(u in set(vertices) and v in set(vertices) for u, v in edges)
            assert case["k"] == n + 2 * m, case["index"]

    def test_certificates_map_both_ways(self, battery):
        reduction = ThreeSatToVertexCover()
        for case in battery:
            instance = (case["cnf"], case["n"])
            if not case["models"]:
                with pytest.raises(ValueError):
                    reduction.map_solution_back(instance, case["min_cover"], case["result"])
                continue
            for model in case["models"]:
                cover = reduction.map_solution_forward(instance, model, case["result"])
                assert len(cover) == case["k"], case["index"]
                assert is_vertex_cover(case["edges"], cover), case["index"]
            assignment = reduction.map_solution_back(
                instance, case["min_cover"], case["result"]
            )
            assert set(assignment) == set(range(1, case["n"] + 1)), case["index"]
            assert satisfied_by(case["cnf"], assignment), case["index"]


# ----------------------------------------------------------------------
# explain() and logging
# ----------------------------------------------------------------------
class TestExplain:
    """The explanation names every gadget, the counts and k, and is logged."""

    def test_explain_mentions_every_gadget_and_k(self, sipser_result, caplog):
        reduction = ThreeSatToVertexCover()
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            text = reduction.explain(sipser_result)
        lower = text.lower()
        for phrase in ("variable gadget", "clause gadget", "connecting edge", "padding"):
            assert phrase in lower, phrase
        assert "triangle" in lower
        assert "k = 8" in text
        assert "13 nodes" in text and "20 edges" in text
        assert "2n + 3m" in text and "n + 6m" in text and "n + 2m" in text
        assert "polynomial" in lower

    def test_explain_joins_steps_and_logs_each_line(self, sipser_result, caplog):
        reduction = ThreeSatToVertexCover()
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            text = reduction.explain(sipser_result)
        assert text == "\n".join(sipser_result.steps)
        logged = [
            record.getMessage() for record in caplog.records
            if record.name == LOGGER_NAME and record.levelno == logging.INFO
        ]
        assert logged == sipser_result.steps

    def test_every_step_line_is_a_single_line(self, sipser_result):
        assert sipser_result.steps
        assert all("\n" not in line for line in sipser_result.steps)

    def test_chain_explanations_mention_k(self, sipser_result):
        is_result = VertexCoverToIndependentSet().transform(sipser_result.target_instance)
        clique_result = IndependentSetToClique().transform(is_result.target_instance)
        is_text = VertexCoverToIndependentSet().explain(is_result)
        clique_text = IndependentSetToClique().explain(clique_result)
        assert "k = 8" in is_text and "13 - 8 = 5" in is_text
        assert "k = 5" in clique_text and "58 edges" in clique_text

    def test_explain_rejects_non_results(self):
        with pytest.raises(TypeError):
            ThreeSatToVertexCover().explain(["a step"])


# ----------------------------------------------------------------------
# VC -> IS -> Clique
# ----------------------------------------------------------------------
class TestChain:
    """The chain preserves answers and certificates, checked with graph_utils."""

    def test_path_graph_by_hand(self):
        path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
        vc_to_is, is_to_clique = VertexCoverToIndependentSet(), IndependentSetToClique()
        is_result = vc_to_is.transform(path)
        (is_vertices, is_edges), is_k = is_result.target_instance
        assert sorted(is_vertices) == [1, 2, 3, 4]
        assert normalize_edges(is_edges) == {(1, 2), (2, 3), (3, 4)}
        assert is_k == 2
        clique_result = is_to_clique.transform(is_result.target_instance)
        (_, clique_edges), clique_k = clique_result.target_instance
        assert normalize_edges(clique_edges) == {(1, 3), (1, 4), (2, 4)}
        assert clique_k == 2

        independent = vc_to_is.map_solution_forward(path, {2, 3}, is_result)
        assert independent == {1, 4}
        clique = is_to_clique.map_solution_forward(is_result.target_instance, independent,
                                                   clique_result)
        assert clique == {1, 4}
        assert is_clique(clique_edges, clique)
        back = is_to_clique.map_solution_back(is_result.target_instance, clique, clique_result)
        assert vc_to_is.map_solution_back(path, back, is_result) == {2, 3}

    def test_reduce_chain_on_sipser(self):
        stages = [ThreeSatToVertexCover(), VertexCoverToIndependentSet(),
                  IndependentSetToClique()]
        results = reduce_chain((SIPSER, SIPSER_N), stages)
        assert len(results) == 3
        assert all(isinstance(result, ReductionResult) for result in results)

        (vc_vertices, vc_edges), vc_k = results[0].target_instance
        (is_vertices, is_edges), is_k = results[1].target_instance
        (cl_vertices, cl_edges), cl_k = results[2].target_instance
        assert (len(vc_vertices), len(vc_edges), vc_k) == (13, 20, 8)
        assert set(is_vertices) == SIPSER_NODES
        assert normalize_edges(is_edges) == normalize_edges(SIPSER_EDGES)
        assert is_k == 13 - 8
        assert set(cl_vertices) == SIPSER_NODES
        assert len(cl_edges) == 13 * 12 // 2 - 20 == 58
        assert normalize_edges(cl_edges) == complement(SIPSER_NODES, SIPSER_EDGES)
        assert cl_k == 5

        # Each stage read the previous stage's target instance.
        assert results[1].source_instance[1] == vc_k
        assert set(results[1].source_instance[0][0]) == set(vc_vertices)
        assert results[2].source_instance[1] == is_k

        # Certificates forward: assignment -> cover -> independent set -> clique.
        sat, vc, is_ = stages
        cover = sat.map_solution_forward((SIPSER, SIPSER_N), {1: False, 2: True}, results[0])
        independent = vc.map_solution_forward(results[0].target_instance, cover, results[1])
        clique = is_.map_solution_forward(results[1].target_instance, independent, results[2])
        assert len(cover) == 8 and is_vertex_cover(SIPSER_EDGES, cover)
        assert independent == SIPSER_NODES - cover
        assert len(independent) == 5 and is_independent_set(SIPSER_EDGES, independent)
        assert clique == independent
        assert is_clique(complement(SIPSER_NODES, SIPSER_EDGES), clique)

        # And back: clique -> independent set -> cover -> assignment.
        back_is = is_.map_solution_back(results[1].target_instance, clique, results[2])
        back_cover = vc.map_solution_back(results[0].target_instance, back_is, results[1])
        assignment = sat.map_solution_back((SIPSER, SIPSER_N), back_cover, results[0])
        assert back_is == independent
        assert back_cover == cover
        assert assignment == {1: False, 2: True}

    def test_chain_over_the_battery(self, battery):
        vc_to_is, is_to_clique = VertexCoverToIndependentSet(), IndependentSetToClique()
        for case in battery:
            vc_instance = (case["result"].target_instance[0], case["k"])
            results = reduce_chain(vc_instance, [vc_to_is, is_to_clique])
            vertices, edges, cover = case["vertices"], case["edges"], case["min_cover"]
            complement_edges = complement(vertices, edges)
            (_, clique_edges), clique_k = results[1].target_instance
            assert normalize_edges(clique_edges) == complement_edges, case["index"]
            assert clique_k == len(vertices) - case["k"], case["index"]

            independent = vc_to_is.map_solution_forward(vc_instance, cover, results[0])
            assert independent == set(vertices) - cover, case["index"]
            assert is_independent_set(edges, independent), case["index"]
            clique = is_to_clique.map_solution_forward(
                results[0].target_instance, independent, results[1]
            )
            assert is_clique(complement_edges, clique), case["index"]
            assert len(clique) == len(vertices) - len(cover), case["index"]
            back = vc_to_is.map_solution_back(
                vc_instance,
                is_to_clique.map_solution_back(results[0].target_instance, clique, results[1]),
                results[0],
            )
            assert back == cover, case["index"]

    def test_reduce_chain_accepts_classes_and_empty_lists(self):
        results = reduce_chain(([[1]], 1), [ThreeSatToVertexCover, VertexCoverToIndependentSet])
        assert [result.target_instance[1] for result in results] == [3, 2]
        assert reduce_chain(([[1]], 1), []) == []

    def test_reduce_chain_rejects_non_reductions(self):
        with pytest.raises(TypeError):
            reduce_chain(([[1]], 1), [ThreeSatToVertexCover(), "clique"])
        with pytest.raises(TypeError):
            reduce_chain(([[1]], 1), 5)

    def test_complement_of_a_non_cover_is_refused(self):
        path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
        result = VertexCoverToIndependentSet().transform(path)
        with pytest.raises(ValueError):
            VertexCoverToIndependentSet().map_solution_forward(path, {1, 4}, result)
        with pytest.raises(ValueError):
            VertexCoverToIndependentSet().map_solution_back(path, {2, 3}, result)

    def test_non_clique_is_refused(self):
        path = (([1, 2, 3, 4], {(1, 2), (2, 3), (3, 4)}), 2)
        result = IndependentSetToClique().transform(path)
        with pytest.raises(ValueError):
            IndependentSetToClique().map_solution_forward(path, {1, 2}, result)
        with pytest.raises(ValueError):
            IndependentSetToClique().map_solution_back(path, {2, 3}, result)


# ----------------------------------------------------------------------
# Invalid input
# ----------------------------------------------------------------------
class TestInvalidInput:
    """Malformed formulas, graphs and certificates are rejected."""

    @pytest.mark.parametrize(
        "cnf, n_vars",
        [
            ([[1, 0, 2]], 2),          # literal 0
            ([[1, 3]], 2),             # variable out of range
            ([[-3, 1]], 2),            # negative literal out of range
            ([[1], []], 2),            # empty clause
            ([[1, 2, -1, 2]], 2),      # four literals is not 3-CNF
            ([[1]], -1),               # negative variable count
            ([[1]], 0),                # no variables, so x1 is out of range
        ],
        ids=["literal-0", "out-of-range", "negative-out-of-range", "empty-clause",
             "four-literals", "negative-n", "zero-n"],
    )
    def test_bad_formula_raises_value_error(self, cnf, n_vars):
        with pytest.raises(ValueError):
            ThreeSatToVertexCover().transform((cnf, n_vars))

    @pytest.mark.parametrize(
        "instance",
        [
            ([["1"]], 1),              # string literal
            ([[1.0]], 1),              # float literal
            ([[True]], 1),             # bool literal
            ([[1]], "1"),              # n_vars not an int
            ("x1 or x2", 2),           # cnf not a list
            ([1, 2], 2),               # clause not a list
            "not an instance",
        ],
        ids=["str-literal", "float-literal", "bool-literal", "str-n", "str-cnf",
             "int-clause", "str-instance"],
    )
    def test_bad_types_raise_type_error(self, instance):
        with pytest.raises(TypeError):
            ThreeSatToVertexCover().transform(instance)

    def test_empty_formula_is_allowed(self):
        result = ThreeSatToVertexCover().transform(([], 2))
        (vertices, edges), k = result.target_instance
        assert (len(vertices), len(edges), k) == (4, 2, 2)

    @pytest.mark.parametrize(
        "reduction, instance",
        [
            (VertexCoverToIndependentSet(), (([1, 2], {(1, 2)}), 3)),     # k > |V|
            (VertexCoverToIndependentSet(), (([1, 2], {(1, 2)}), -1)),    # k < 0
            (IndependentSetToClique(), (([1, 2], {(1, 2)}), -1)),         # k < 0
            (VertexCoverToIndependentSet(), (([1, 2], {(1, 1)}), 1)),     # self-loop
            (IndependentSetToClique(), (([1, 2], {(1, 5)}), 1)),          # unknown endpoint
            (IndependentSetToClique(), (([1, 2], {(1, 2, 3)}), 1)),       # not a pair
        ],
        ids=["k-too-big", "vc-negative-k", "is-negative-k", "self-loop",
             "unknown-endpoint", "triple-edge"],
    )
    def test_bad_graph_instance_raises_value_error(self, reduction, instance):
        with pytest.raises(ValueError):
            reduction.transform(instance)

    def test_forward_rejects_a_non_satisfying_assignment(self, sipser_result):
        with pytest.raises(ValueError):
            ThreeSatToVertexCover().map_solution_forward(
                (SIPSER, SIPSER_N), {1: True, 2: True}, sipser_result
            )
        with pytest.raises(TypeError):
            ThreeSatToVertexCover().map_solution_forward(
                (SIPSER, SIPSER_N), [(1, False), (2, True)], sipser_result
            )

    def test_back_rejects_bad_covers(self, sipser_result):
        reduction = ThreeSatToVertexCover()
        instance = (SIPSER, SIPSER_N)
        with pytest.raises(ValueError):    # misses the edge c1_3 - x2
            reduction.map_solution_back(instance, SIPSER_FORWARD_COVER - {"x2"}, sipser_result)
        with pytest.raises(ValueError):    # a valid cover, but 13 > k = 8
            reduction.map_solution_back(instance, SIPSER_NODES, sipser_result)
        with pytest.raises(ValueError):    # names a node that does not exist
            reduction.map_solution_back(instance, SIPSER_FORWARD_COVER | {"x9"}, sipser_result)
        with pytest.raises(TypeError):
            reduction.map_solution_back(instance, "x2", sipser_result)

    def test_result_from_another_instance_is_refused(self, sipser_result):
        with pytest.raises(ValueError):
            ThreeSatToVertexCover().map_solution_forward(
                ([[1, 2]], 2), {1: True, 2: True}, sipser_result
            )
        with pytest.raises(TypeError):
            ThreeSatToVertexCover().map_solution_back((SIPSER, SIPSER_N), set(), None)

    def test_base_class_is_abstract(self):
        with pytest.raises(NotImplementedError):
            Reduction().transform((SIPSER, SIPSER_N))


# ----------------------------------------------------------------------
# Determinism and copies
# ----------------------------------------------------------------------
class TestDeterminismAndCopies:
    """The same formula always gives the same graph, and inputs are not shared."""

    def test_transform_is_deterministic(self):
        first = ThreeSatToVertexCover().transform((SIPSER, SIPSER_N))
        second = ThreeSatToVertexCover().transform((SIPSER, SIPSER_N))
        assert first.target_instance == second.target_instance
        assert first.steps == second.steps
        assert first.node_labels == second.node_labels

    def test_input_is_not_mutated_and_source_is_a_copy(self):
        cnf = [[1], [-1, 2]]
        result = ThreeSatToVertexCover().transform((cnf, 2))
        assert cnf == [[1], [-1, 2]]           # padding did not touch the input
        cnf[0].append(2)
        assert result.source_instance[0] == [[1], [-1, 2]]


# ----------------------------------------------------------------------
# Benchmark smoke test (spec section 10.6)
# ----------------------------------------------------------------------
import csv as _csv  # noqa: E402
import os as _os  # noqa: E402
import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_BENCHMARK = _os.path.join(_REPO_ROOT, "benchmarks", "week8_complexity_benchmark.py")
_EXPECTED_HEADER = [
    "problem", "method", "kind", "instance_family", "n", "param", "runs", "mean_time_s",
    "std_time_s", "verify_time_s", "nodes_mean", "solution_value", "optimal_value",
    "approx_ratio", "success_rate", "seed", "measurement",
]
_TIMING = {"mean_time_s", "std_time_s", "verify_time_s"}


def _run_smoke(out_dir: str, seed: int = 42) -> None:
    """Run the benchmark's smoke mode into ``out_dir`` and fail loudly on error."""
    completed = _subprocess.run(
        [_sys.executable, _BENCHMARK, "--smoke", "--out", out_dir, "--seed", str(seed)],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]


def _read_rows(path: str) -> tuple:
    with open(path, newline="") as handle:
        reader = _csv.reader(handle)
        header = next(reader)
        return header, [dict(zip(header, row)) for row in reader]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two smoke runs with the same master seed, in separate directories."""
    first = str(tmp_path_factory.mktemp("week8_smoke_a"))
    second = str(tmp_path_factory.mktemp("week8_smoke_b"))
    _run_smoke(first)
    _run_smoke(second)
    return first, second


class TestBenchmarkSmoke:
    """The benchmark runs end to end and writes what the report reads."""

    @pytest.mark.parametrize("name", ["runtime_growth.png", "approximation_quality.png",
                                      "comparison_table.csv"])
    def test_outputs_exist_and_are_non_empty(self, smoke_runs, name):
        path = _os.path.join(smoke_runs[0], name)
        assert _os.path.isfile(path)
        assert _os.path.getsize(path) > 0

    @pytest.mark.parametrize("name", ["runtime_growth.png", "approximation_quality.png"])
    def test_figures_are_png(self, smoke_runs, name):
        with open(_os.path.join(smoke_runs[0], name), "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_header_is_exactly_the_specified_columns(self, smoke_runs):
        header, rows = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        assert header == _EXPECTED_HEADER
        assert rows

    def test_kind_and_measurement_take_only_allowed_values(self, smoke_runs):
        _, rows = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        assert {r["kind"] for r in rows} <= {"exact", "approximation", "heuristic"}
        assert {r["measurement"] for r in rows} <= {"measured", "projected"}

    def test_every_problem_is_present(self, smoke_runs):
        _, rows = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        assert {r["problem"] for r in rows} == {"subset_sum", "sat", "vertex_cover", "tsp",
                                                "reduction"}

    def test_verifier_time_recorded_next_to_search_time(self, smoke_runs):
        _, rows = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        exact_with_certificates = [r for r in rows if r["kind"] == "exact"
                                   and r["problem"] in {"vertex_cover", "tsp"}]
        assert exact_with_certificates
        assert all(r["verify_time_s"] and r["mean_time_s"] for r in exact_with_certificates)

    def test_ratios_respect_the_proven_bound(self, smoke_runs):
        _, rows = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        for r in rows:
            if r["problem"] == "vertex_cover" and r["method"] == "two_approx" and r["approx_ratio"]:
                assert 1.0 <= float(r["approx_ratio"]) <= 2.0
            if r["approx_ratio"]:
                assert float(r["approx_ratio"]) >= 1.0 - 1e-9

    def test_anchor_ratios_are_exactly_two(self, smoke_runs):
        _, rows = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        anchors = {r["instance_family"]: float(r["approx_ratio"]) for r in rows
                   if r["problem"] == "vertex_cover" and r["method"] == "two_approx"
                   and r["instance_family"] in {"path", "star", "k33"}}
        assert anchors == {"path": 2.0, "star": 2.0, "k33": 2.0}

    def test_same_seed_gives_identical_non_timing_columns(self, smoke_runs):
        _, first = _read_rows(_os.path.join(smoke_runs[0], "comparison_table.csv"))
        _, second = _read_rows(_os.path.join(smoke_runs[1], "comparison_table.csv"))
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert {k: v for k, v in a.items() if k not in _TIMING} == \
                   {k: v for k, v in b.items() if k not in _TIMING}
