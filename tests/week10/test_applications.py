"""Week 10 tests: matching, segmentation and scheduling on the spec 9.3 anchors.

Each expected value is worked out by hand, so the tests do not repeat the code
under test.

Matching: the edges a-1, a-2, b-1, c-2, c-3, d-3 touch only the right nodes 1, 2
and 3, so size 3 is an upper bound, reached by a-1, c-2, d-3. With a-1, a-2, b-1
the only way to match both a and b is a-2 and b-1.

Scheduling: M1 has 2 slots and M2 has 1, so 3 of the 4 jobs is an upper bound,
reached by J1 and J3 on M1 and J4 on M2.

Segmentation: labelling the five bright pixels foreground costs 255 - I each
(55 + 45 + 65 + 50 + 60 = 275) and the seven dark pixels background costs I each
(40 + 30 + 35 + 20 + 60 + 25 + 15 = 225), which is 500. The four neighbour pairs
split by that boundary have differences 170, 170, 145 and 135, adding weights
50 exp(-d^2 / 1800) of about 0.0000, 0.0000, 0.0004 and 0.0020, so the real cut is
500.0024. Scaled by 1000 and rounded, those four weights become 0, 0, 0 and 2, so
the network's cut is 500002 / 1000.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.flow_network_solver.applications.image_seg import SCALE, save_segmentation, segment
from src.flow_network_solver.applications.matching import bipartite_matching, draw_matching
from src.flow_network_solver.applications.scheduling import schedule
from src.flow_network_solver.utils.generators import bipartite_graph, grid_image
from src.utils.random_utils import derive_seed

SOLVERS = ["ek", "pr"]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

LEFT = ["a", "b", "c", "d"]
RIGHT = [1, 2, 3, 4]
EDGES = [("a", 1), ("a", 2), ("b", 1), ("c", 2), ("c", 3), ("d", 3)]

IMAGE = [[200, 210, 40, 30], [190, 205, 35, 20], [195, 60, 25, 15]]
MASK = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]]


def assert_valid_matching(pairs, edges):
    """Check every pair is an input edge and no node is used twice."""
    assert set(pairs) <= set(edges)
    lefts = [l for l, _ in pairs]
    rights = [r for _, r in pairs]
    assert len(set(lefts)) == len(lefts)
    assert len(set(rights)) == len(rights)


def assert_png(path):
    """Check that a non-empty PNG exists at ``path``."""
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(PNG_SIGNATURE)


@pytest.mark.parametrize("solver", SOLVERS)
def test_matching_anchor(solver):
    pairs = bipartite_matching(LEFT, RIGHT, EDGES, solver=solver)
    assert len(pairs) == 3
    assert_valid_matching(pairs, EDGES)
    assert 4 not in {r for _, r in pairs}


@pytest.mark.parametrize("solver", SOLVERS)
def test_matching_reroutes_through_a_reverse_edge(solver):
    edges = [("a", 1), ("a", 2), ("b", 1)]
    assert sorted(bipartite_matching(["a", "b"], [1, 2], edges, solver=solver)) == [
        ("a", 2),
        ("b", 1),
    ]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    ("left", "right", "edges"),
    [([], [], []), (["a"], [], []), ([], [1], []), (["a", "b"], [1, 2], [])],
)
def test_matching_with_nothing_to_match(solver, left, right, edges):
    assert bipartite_matching(left, right, edges, solver=solver) == []


@pytest.mark.parametrize("solver", SOLVERS)
def test_matching_keeps_the_sides_apart_when_labels_overlap(solver):
    edges = [(1, 2), (2, 1)]
    assert sorted(bipartite_matching([1, 2], [1, 2], edges, solver=solver)) == [(1, 2), (2, 1)]


@pytest.mark.parametrize("solver", SOLVERS)
def test_matching_with_a_repeated_edge(solver):
    assert bipartite_matching(["a"], [1], [("a", 1), ("a", 1)], solver=solver) == [("a", 1)]


@pytest.mark.parametrize("solver", SOLVERS)
def test_matching_size_agrees_with_networkx_on_30_random_graphs(solver):
    for index in range(30):
        n_left, n_right = 3 + index % 8, 4 + (3 * index) % 7
        p = 0.1 + 0.02 * index
        seed = derive_seed(10, "matching", index)
        left, right, edges = bipartite_graph(n_left, n_right, p, seed)
        pairs = bipartite_matching(left, right, edges, solver=solver)
        graph = nx.Graph()
        graph.add_nodes_from(left)
        graph.add_nodes_from(right)
        graph.add_edges_from(edges)
        expected = len(nx.bipartite.maximum_matching(graph, top_nodes=left)) // 2
        assert len(pairs) == expected
        assert_valid_matching(pairs, edges)


@pytest.mark.parametrize(
    ("left", "right", "edges", "solver"),
    [
        (["a"], [1], [("a", 2)], "ek"),
        (["a"], [1], [("b", 1)], "ek"),
        (["a", "a"], [1], [], "ek"),
        (["a"], [1, 1], [], "ek"),
        (["a"], [1], [], "dinic"),
    ],
)
def test_matching_rejects_bad_input(left, right, edges, solver):
    with pytest.raises(ValueError):
        bipartite_matching(left, right, edges, solver=solver)


def test_draw_matching_writes_a_png(tmp_path):
    out = tmp_path / "new_folder" / "matching.png"
    pairs = bipartite_matching(LEFT, RIGHT, EDGES)
    assert draw_matching(LEFT, RIGHT, EDGES, pairs, str(out)) == str(out)
    assert_png(out)


def test_draw_matching_rejects_a_pair_that_is_not_an_edge(tmp_path):
    out = tmp_path / "matching.png"
    with pytest.raises(ValueError):
        draw_matching(LEFT, RIGHT, EDGES, [("d", 1)], str(out))
    assert not out.exists()


def test_schedule_anchor():
    jobs = ["J1", "J2", "J3", "J4"]
    capacities = {"M1": 2, "M2": 1}
    allowed = {"J1": ["M1"], "J2": ["M1", "M2"], "J3": ["M1"], "J4": ["M2"]}
    assignment, unassigned = schedule(jobs, ["M1", "M2"], allowed, capacities)
    assert len(assignment) == 3
    assert len(unassigned) == 1
    assert set(assignment) | set(unassigned) == set(jobs)
    for machine, capacity in capacities.items():
        assert list(assignment.values()).count(machine) <= capacity
    for job, machine in assignment.items():
        assert machine in allowed[job]


def test_schedule_staffing_example():
    people = ["Ana", "Ben", "Cho", "Dev"]
    head_count = {"Web": 2, "Data": 1}
    skills = {"Ana": ["Web"], "Ben": ["Web", "Data"], "Cho": ["Data"], "Dev": ["Data"]}
    staffed, free = schedule(people, ["Web", "Data"], skills, head_count)
    # Ana can only do Web. Three seats exist, and Data can seat just one of Ben, Cho, Dev.
    assert len(staffed) == 3 and len(free) == 1
    assert staffed["Ana"] == "Web"
    assert list(staffed.values()).count("Web") == 2
    assert list(staffed.values()).count("Data") == 1
    assert all(project in skills[person] for person, project in staffed.items())


def test_schedule_reroutes_through_a_reverse_edge():
    allowed = {"J1": ["M1", "M2"], "J2": ["M1"]}
    assignment, unassigned = schedule(["J1", "J2"], ["M1", "M2"], allowed, {"M1": 1, "M2": 1})
    assert assignment == {"J1": "M2", "J2": "M1"}
    assert unassigned == []


def test_schedule_with_nowhere_to_go():
    assert schedule([], [], {}, {}) == ({}, [])
    assert schedule(["J1", "J2"], ["M1"], {"J2": ["M1"]}, {"M1": 1}) == ({"J2": "M1"}, ["J1"])
    assert schedule(["J1"], ["M1"], {"J1": ["M1"]}, {"M1": 0}) == ({}, ["J1"])


def test_schedule_counts_a_repeated_machine_once():
    assert schedule(["J1"], ["M1"], {"J1": ["M1", "M1"]}, {"M1": 5}) == ({"J1": "M1"}, [])
    assignment, unassigned = schedule(
        ["J1"], ["M1"], {"J1": ["M1", "M1"]}, {"M1": 5}, demands={"J1": 2}
    )
    assert assignment == {"J1": ["M1"]}
    assert unassigned == ["J1"]


def test_schedule_demand_met_across_two_machines():
    allowed = {"J1": ["M1", "M2"], "J2": ["M1"]}
    assignment, unassigned = schedule(
        ["J1", "J2"], ["M1", "M2"], allowed, {"M1": 2, "M2": 1}, demands={"J1": 2}
    )
    assert assignment == {"J1": ["M1", "M2"], "J2": ["M1"]}
    assert unassigned == []


def test_schedule_part_of_a_demand_is_held_and_reported_short():
    allowed = {"J1": ["M1", "M2"]}
    assignment, unassigned = schedule(
        ["J1"], ["M1", "M2"], allowed, {"M1": 1, "M2": 1}, demands={"J1": 3}
    )
    assert assignment == {"J1": ["M1", "M2"]}
    assert unassigned == ["J1"]


def test_schedule_a_demand_takes_one_slot_per_machine():
    # A job-machine edge has capacity 1, so J1 cannot fill both M1 slots and J2 gets one.
    allowed = {"J1": ["M1"], "J2": ["M1"]}
    assignment, unassigned = schedule(["J1", "J2"], ["M1"], allowed, {"M1": 2}, demands={"J1": 2})
    assert assignment == {"J1": ["M1"], "J2": ["M1"]}
    assert unassigned == ["J1"]


@pytest.mark.parametrize(
    ("jobs", "machines", "allowed", "capacities", "demands"),
    [
        (["J1", "J1"], ["M1"], {}, {"M1": 1}, None),
        (["J1"], ["M1", "M1"], {}, {"M1": 1}, None),
        (["J1"], ["M1"], {"J2": ["M1"]}, {"M1": 1}, None),
        (["J1"], ["M1"], {"J1": ["M2"]}, {"M1": 1}, None),
        (["J1"], ["M1"], {"J1": ["M1"]}, {}, None),
        (["J1"], ["M1"], {"J1": ["M1"]}, {"M1": -1}, None),
        (["J1"], ["M1"], {"J1": ["M1"]}, {"M1": 1}, {"J1": 0}),
        (["J1"], ["M1"], {"J1": ["M1"]}, {"M1": 1}, {"J2": 1}),
    ],
)
def test_schedule_rejects_bad_input(jobs, machines, allowed, capacities, demands):
    with pytest.raises(ValueError):
        schedule(jobs, machines, allowed, capacities, demands)


def test_schedule_rejects_a_non_integer_capacity():
    with pytest.raises(TypeError):
        schedule(["J1"], ["M1"], {"J1": ["M1"]}, {"M1": 1.5})


@pytest.mark.parametrize("solver", SOLVERS)
def test_segment_anchor(solver):
    mask, cut = segment(IMAGE, lam=50, sigma=30, solver=solver)
    assert mask == MASK
    assert abs(cut - 500.0024) <= 1.0
    assert cut == pytest.approx(500002 / SCALE)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float64])
def test_segment_accepts_numpy_input(solver, dtype):
    mask, cut = segment(np.array(IMAGE, dtype=dtype), lam=50, sigma=30, solver=solver)
    assert mask == MASK
    assert abs(cut - 500.0024) <= 1.0


@pytest.mark.parametrize("solver", SOLVERS)
def test_segment_single_pixel_takes_the_cheaper_label(solver):
    assert segment([[200]], solver=solver) == ([[1]], 55.0)
    assert segment([[50]], solver=solver) == ([[0]], 50.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_segment_uniform_bright_image_is_all_foreground(solver):
    # Foreground costs 255 - 200 = 55 per pixel and splits no neighbours.
    assert segment([[200, 200], [200, 200]], solver=solver) == ([[1, 1], [1, 1]], 220.0)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("image", [[[100, 200]], [[200, 100]], [[100], [200]], [[200], [100]]])
def test_segment_links_neighbours_both_ways_along_rows_and_columns(solver, image):
    # At lam 200 and sigma 1000 the pair weight is 200 exp(-0.005), about 199. Both foreground
    # costs 155 + 55 = 210, both background 100 + 200 = 300, and the split 100 + 55 + 199 = 354.
    # Without the edge from the bright pixel to the dark one the split would cost only 155.
    mask, cut = segment(image, lam=200, sigma=1000, solver=solver)
    assert mask == [[1] * len(image[0]) for _ in image]
    assert cut == 210.0


def test_segment_mostly_recovers_the_disc():
    rows, cols = 32, 32
    mask, _ = segment(grid_image(rows, cols, 1))
    centre = (rows - 1) / 2
    radius_squared = (0.3 * rows) ** 2
    truth = [
        [int((r - centre) ** 2 + (c - centre) ** 2 <= radius_squared) for c in range(cols)]
        for r in range(rows)
    ]
    found = np.array(mask, dtype=bool)
    disc = np.array(truth, dtype=bool)
    assert (found == disc).mean() >= 0.95
    assert (found & disc).sum() / (found | disc).sum() >= 0.9


@pytest.mark.parametrize(
    ("image", "kwargs"),
    [
        ([1, 2, 3], {}),
        ([[1, 2], [3]], {}),
        ([[]], {}),
        ([[256]], {}),
        ([[-1]], {}),
        ([[float("nan")]], {}),
        ([[1]], {"lam": -1}),
        ([[1]], {"sigma": 0}),
        ([[1]], {"solver": "dinic"}),
    ],
)
def test_segment_rejects_bad_input(image, kwargs):
    with pytest.raises(ValueError):
        segment(image, **kwargs)


def test_save_segmentation_writes_a_png(tmp_path):
    out = tmp_path / "new_folder" / "segmentation.png"
    assert save_segmentation(IMAGE, MASK, str(out)) == str(out)
    assert_png(out)


def test_save_segmentation_accepts_numpy_input(tmp_path):
    out = tmp_path / "segmentation.png"
    save_segmentation(np.array(IMAGE, dtype=np.uint8), np.array(MASK), str(out))
    assert_png(out)


def test_save_segmentation_rejects_a_mismatched_mask(tmp_path):
    out = tmp_path / "segmentation.png"
    with pytest.raises(ValueError):
        save_segmentation(IMAGE, [[1, 0]], str(out))
    assert not out.exists()
