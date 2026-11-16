"""Week 13 tests: range-query engine, mini version control and DNA index (spec 9.7 anchors).

The rulers are independent of the code under test: plain Python lists with
one saved snapshot per version (queried with the built-in ``sum``, ``min`` and
``max`` on slices), plain ``dict`` snapshots and set operations for the
version-control diffs, ``hashlib.sha1`` for the commit-id formula,
``sorted(range(n), key=lambda i: text[i:])`` for suffix arrays, sorted
rotations for the BWT, ``str.count`` and ``sorted`` on windows, and a naive
overlapping scan for pattern counts. Those library calls appear only here,
as the cross-check (spec 1.1).

Hand check of the 9.7 anchors
-----------------------------
* ``A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]`` with INCLUSIVE ranges:
  ``sum(2, 6) = 8 + 1 + 9 + 3 + 7 = 28``. ``range_add(3, 7, 10)`` adds 50 in
  total, so ``sum(0, 9) = 45 + 50 = 95``; positions 2..5 become
  ``8, 11, 19, 13``, so ``max(2, 5) = 19``; version 0 still sums to 45.
* ``DNAIndex("ACGTACGTTGCA")``: positions 2..9 are ``GTACGTTG`` (one A);
  positions 3..8 are ``TACGTT``, sorted ``ACGTTT``, so k = 1, 3, 6 give A, G,
  T. ACG occurs at 0 and 4, CG at 1 and 5, GT at 2 and 6, T at 3, 7 and 8,
  TTG at 7, GCA at 9, AAA nowhere.

No timings are asserted.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from src.advds.applications import dna_index, range_queries, version_control
from src.advds.applications.dna_index import (
    DEFAULT_FASTA,
    DNAIndex,
    build_suffix_array,
    bwt_from_suffix_array,
    make_synthetic_fasta,
    read_fasta,
    synthetic_dna,
)
from src.advds.applications.range_queries import RangeQueryEngine, inclusive_to_half_open
from src.advds.applications.version_control import MiniVCS, diff_maps
from src.advds.trees.persistent_tree import PersistentMap
from src.utils.random_utils import derive_seed, make_rng

SEED = 42
REPO = Path(__file__).resolve().parents[2]

#: The spec 9.1 array.
A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]


def _rng(*labels):
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(SEED, "test_applications", *labels))


# ===========================================================================
# RangeQueryEngine
# ===========================================================================


def test_engine_spec_anchors():
    eng = RangeQueryEngine(A)
    assert eng.sum(2, 6) == 28
    assert eng.range_add(3, 7, 10) == 1
    assert eng.sum(0, 9) == 95
    assert eng.max(2, 5) == 19
    assert eng.sum_at(0, 0, 9) == 45


def test_engine_first_queries_and_versions():
    eng = RangeQueryEngine(A)
    assert (eng.min(1, 4), eng.max(0, 9), eng.min(5, 9)) == (1, 9, 0)
    assert eng.current_version() == 0
    assert eng.set(0, 50) == 1
    assert eng.add(9, -5) == 2
    assert eng.range_add(0, 9, 1) == 3
    assert eng.current_version() == 3
    assert eng.sum_at(0, 0, 9) == 45
    assert eng.sum_at(1, 0, 9) == 90
    assert eng.sum_at(2, 0, 9) == 85
    assert eng.sum(0, 9) == 95
    assert (eng.min_at(2, 0, 9), eng.max_at(2, 0, 9)) == (-5, 50)


def test_inclusive_conversion_and_errors():
    assert inclusive_to_half_open(0, 0, 1) == (0, 1)
    assert inclusive_to_half_open(np.int64(2), 6, 10) == (2, 7)
    with pytest.raises(IndexError):
        inclusive_to_half_open(0, 10, 10)
    with pytest.raises(IndexError):
        inclusive_to_half_open(-1, 3, 10)
    with pytest.raises(ValueError):
        inclusive_to_half_open(4, 3, 10)
    with pytest.raises(TypeError):
        inclusive_to_half_open(1.0, 3, 10)
    with pytest.raises(TypeError):
        inclusive_to_half_open(True, 3, 10)
    with pytest.raises(IndexError):
        inclusive_to_half_open(0, 0, 0)


def test_engine_rejected_updates_change_nothing():
    eng = RangeQueryEngine(A)
    bad = [
        (lambda: eng.set(10, 1), IndexError),
        (lambda: eng.add(-1, 1), IndexError),
        (lambda: eng.range_add(5, 3, 1), ValueError),
        (lambda: eng.range_add(0, 10, 1), IndexError),
        (lambda: eng.add(0, float("nan")), ValueError),
        (lambda: eng.set(0, float("inf")), ValueError),
        (lambda: eng.add(0, "1"), TypeError),
        (lambda: eng.set(1.0, 3), TypeError),
    ]
    for call, error in bad:
        with pytest.raises(error):
            call()
    assert eng.current_version() == 0
    assert [eng.sum(i, i) for i in range(10)] == A
    with pytest.raises(IndexError):
        eng.sum_at(1, 0, 9)
    with pytest.raises(ValueError):
        eng.min(3, 2)
    with pytest.raises(ValueError):
        RangeQueryEngine([])


def test_engine_floats():
    eng = RangeQueryEngine([0.5, 1.25, -2.0])
    eng.range_add(0, 1, 0.25)  # [0.75, 1.5, -2.0]
    assert eng.sum(0, 2) == pytest.approx(0.25)
    assert (eng.min(0, 2), eng.max(0, 2)) == (-2.0, 1.5)
    assert eng.sum_at(0, 0, 2) == pytest.approx(-0.25)


def test_engine_set_rejects_overflowing_delta_without_changing_anything():
    # Regression (review): value - current overflows to inf for floats near
    # the largest double. The Fenwick tree rejects that delta; before the fix
    # the segment tree had already taken the new value, so min/max and sum
    # disagreed afterwards.
    eng = RangeQueryEngine([-1.5e308, 0.0])
    with pytest.raises(ValueError):
        eng.set(0, 1.5e308)
    assert eng.current_version() == 0
    assert (eng.sum(0, 0), eng.min(0, 0), eng.max(0, 0)) == (-1.5e308,) * 3
    assert (eng.min(0, 1), eng.max(0, 1)) == (-1.5e308, 0.0)
    # A delta that stays finite is still accepted, by all three structures.
    assert eng.set(0, -1.0e308) == 1
    assert eng.sum(0, 1) == pytest.approx(-1.0e308, rel=1e-12)
    assert (eng.min(0, 1), eng.max(0, 1)) == (-1.0e308, 0.0)
    assert eng.sum_at(1, 0, 1) == pytest.approx(-1.0e308, rel=1e-12)
    assert eng.sum_at(0, 0, 1) == -1.5e308


@pytest.mark.parametrize("trial", range(100))
def test_engine_random_sequences_match_list_and_snapshots(trial):
    rng = _rng("engine", trial)
    n = int(rng.integers(1, 31))
    plain = [int(x) for x in rng.integers(-50, 51, n)]
    eng = RangeQueryEngine(plain)
    snapshots: List[List[int]] = [list(plain)]
    for _ in range(40):
        op = int(rng.integers(0, 7))
        l = int(rng.integers(0, n))
        r = int(rng.integers(l, n))
        value = int(rng.integers(-60, 61))
        if op == 0:
            eng.set(l, value)
            plain[l] = value
        elif op == 1:
            eng.add(l, value)
            plain[l] += value
        elif op == 2:
            eng.range_add(l, r, value)
            for i in range(l, r + 1):
                plain[i] += value
        elif op == 3:
            assert eng.sum(l, r) == sum(plain[l : r + 1])
        elif op == 4:
            assert eng.min(l, r) == min(plain[l : r + 1])
        elif op == 5:
            assert eng.max(l, r) == max(plain[l : r + 1])
        else:
            v = int(rng.integers(0, len(snapshots)))
            snap = snapshots[v]
            assert eng.sum_at(v, l, r) == sum(snap[l : r + 1])
            assert eng.min_at(v, l, r) == min(snap[l : r + 1])
            assert eng.max_at(v, l, r) == max(snap[l : r + 1])
        if op <= 2:
            snapshots.append(list(plain))
        assert eng.current_version() == len(snapshots) - 1
    # Every saved snapshot is still answered exactly, on every element.
    for v, snap in enumerate(snapshots):
        assert [eng.sum_at(v, i, i) for i in range(n)] == snap
        assert eng.sum_at(v, 0, n - 1) == sum(snap)
        assert eng.min_at(v, 0, n - 1) == min(snap)
        assert eng.max_at(v, 0, n - 1) == max(snap)
    assert [eng.sum(i, i) for i in range(n)] == plain


# ===========================================================================
# MiniVCS
# ===========================================================================


#: Commit ids of :func:`_three_commits`, pinned from the first run.
PINNED_IDS = ("23e634d407c9", "c9490cb80d02", "339c61c9db13")


def _three_commits():
    """The spec scenario: add a, b; modify b, add c; remove a."""
    vcs = MiniVCS()
    vcs.add("a", "alpha\n")
    vcs.add("b", "beta\n")
    c1 = vcs.commit("add a and b")
    vcs.add("b", "beta, revised\n")
    vcs.add("c", "gamma\n")
    c2 = vcs.commit("modify b, add c")
    vcs.rm("a")
    c3 = vcs.commit("remove a")
    return vcs, (c1, c2, c3)


def _dict_diff(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, List[str]]:
    """The diff of two plain snapshots, by set operations and sorting."""
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "modified": sorted(p for p in set(a) & set(b) if a[p] != b[p]),
    }


def test_vcs_three_commit_diff_and_checkout():
    vcs, (c1, c2, c3) = _three_commits()
    d = vcs.diff(c1, c3)
    assert d["added"] == ["c"]
    assert d["removed"] == ["a"]
    assert d["modified"] == ["b"]
    assert isinstance(d["nodes_visited"], int) and d["nodes_visited"] > 0
    assert vcs.checkout(c1) == {"a": "alpha\n", "b": "beta\n"}
    assert vcs.checkout(c2) == {"a": "alpha\n", "b": "beta, revised\n", "c": "gamma\n"}
    assert vcs.checkout(c3) == {"b": "beta, revised\n", "c": "gamma\n"}
    assert vcs.log() == [(c3, "remove a"), (c2, "modify b, add c"), (c1, "add a and b")]
    back = vcs.diff(c3, c1)
    assert (back["added"], back["removed"], back["modified"]) == (["a"], ["c"], ["b"])


def test_vcs_ids_follow_the_formula():
    vcs, ids = _three_commits()
    parent = ""
    for cid, (_, message) in zip(ids, reversed(vcs.log())):
        root_hash = vcs.tree(cid).root_hash
        want = hashlib.sha1((root_hash + parent + message).encode("utf-8")).hexdigest()[:12]
        assert cid == want
        parent = cid
    # The tree maps each path to sha1(content).
    assert vcs.tree(ids[0]).items() == [
        ("a", hashlib.sha1(b"alpha\n").hexdigest()),
        ("b", hashlib.sha1(b"beta\n").hexdigest()),
    ]


def test_vcs_ids_are_deterministic_across_runs():
    _, ids = _three_commits()
    _, again = _three_commits()
    assert ids == again
    # Pinned from the first run: any change to the hashing scheme shows here.
    assert ids == PINNED_IDS
    # The same steps (this file's own source of _three_commits) in a fresh
    # interpreter with a different string-hash salt each time.
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from src.advds.applications.version_control import MiniVCS\n"
        + inspect.getsource(_three_commits)
        + "print(','.join(_three_commits()[1]))\n"
    )
    for hash_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert tuple(out.stdout.strip().split(",")) == ids


def test_vcs_one_modified_file_among_1000_visits_fewer_than_100_nodes():
    vcs = MiniVCS()
    for i in range(1000):
        vcs.add(f"src/module_{i:04d}.py", f"synthetic content {i}\n")
    c1 = vcs.commit("1,000 synthetic files")
    target = f"src/module_{make_rng(derive_seed(SEED, 'vcs_1000')).randrange(1000):04d}.py"
    vcs.add(target, "modified\n")
    c2 = vcs.commit("modify one file")
    d = vcs.diff(c1, c2)
    assert (d["added"], d["removed"], d["modified"]) == ([], [], [target])
    assert d["nodes_visited"] < 100
    # A value change copies exactly the root-to-node path: depth + 1 new nodes.
    depth, node = 0, vcs.tree(c1).root
    while node.key != target:
        node = node.left if target < node.key else node.right
        depth += 1
    assert vcs.history()[1]["new_nodes"] == depth + 1
    assert vcs.history()[1]["files_changed"] == 1


def test_every_single_modification_among_1000_files_stays_below_100():
    base = PersistentMap([(f"src/module_{i:04d}.py", f"h{i}") for i in range(1000)])
    worst = 0
    for i in range(1000):
        path = f"src/module_{i:04d}.py"
        d = diff_maps(base, base.set(path, "changed"))
        assert (d["added"], d["removed"], d["modified"]) == ([], [], [path])
        worst = max(worst, d["nodes_visited"])
    assert worst < 100


@pytest.mark.parametrize("trial", range(10))
def test_vcs_random_histories_match_dict_snapshots(trial):
    rng = make_rng(derive_seed(SEED, "vcs_random", trial))
    vcs = MiniVCS()
    working: Dict[str, str] = {}
    snaps: Dict[str, Dict[str, str]] = {}
    ids: List[str] = []
    for c in range(12):
        for _ in range(rng.randint(0, 6)):
            op = rng.random()
            if op < 0.45 or not working:
                path = f"dir{rng.randint(0, 3)}/file_{rng.randint(0, 60):02d}.txt"
                content = f"synthetic {rng.randint(0, 5)}\n"
                working[path] = content
                vcs.add(path, content)
            elif op < 0.75:
                path = rng.choice(sorted(working))
                working[path] = f"synthetic {rng.randint(0, 5)}\n"
                vcs.add(path, working[path])
            else:
                path = rng.choice(sorted(working))
                del working[path]
                vcs.rm(path)
        cid = vcs.commit(f"commit {c}")
        ids.append(cid)
        snaps[cid] = dict(working)
    for cid in ids:
        assert vcs.checkout(cid) == snaps[cid]
    for a in ids:
        for b in ids:
            d = vcs.diff(a, b)
            assert {k: d[k] for k in ("added", "removed", "modified")} == _dict_diff(
                snaps[a], snaps[b]
            )
            assert 0 <= d["nodes_visited"] <= len(snaps[a]) + len(snaps[b])
    history = vcs.history()
    for k, h in enumerate(history):
        before = snaps[ids[k - 1]] if k else {}
        want = _dict_diff(before, snaps[ids[k]])
        assert h["files_changed"] == sum(len(v) for v in want.values())
        assert (h["number"], h["id"], h["files"]) == (k + 1, ids[k], len(snaps[ids[k]]))


def test_vcs_stats_against_an_independent_node_count():
    rng = make_rng(derive_seed(SEED, "vcs_stats"))
    vcs = MiniVCS()
    for i in range(50):
        vcs.add(f"f{i:02d}.txt", f"v0 {i}\n")
    vcs.commit("50 files")
    for c in range(1, 8):
        for path in rng.sample([f"f{i:02d}.txt" for i in range(50)], 3):
            vcs.add(path, f"v{c} {path}\n")
        vcs.commit(f"three changes {c}")
    stats = vcs.stats()
    seen = set()
    full = 0
    for cid, _ in vcs.log():
        full += len(vcs.tree(cid))
        stack = [vcs.tree(cid).root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            seen.add(id(node))
            stack.extend((node.left, node.right))
    assert stats["commits"] == 8
    assert stats["files"] == 50
    assert stats["full_copy_nodes"] == full == 400
    assert stats["unique_nodes"] == len(seen)
    assert stats["unique_nodes"] == sum(h["new_nodes"] for h in vcs.history())
    assert stats["saved_nodes"] == full - len(seen)
    assert stats["stored_fraction"] == pytest.approx(len(seen) / full)
    assert stats["unique_nodes"] < 0.5 * stats["full_copy_nodes"]


def test_vcs_edge_cases_and_errors():
    vcs = MiniVCS()
    assert vcs.head is None and vcs.log() == [] and vcs.stats()["full_copy_nodes"] == 0
    empty = vcs.commit("empty")
    assert vcs.checkout(empty) == {}
    vcs.add("x", "1")
    c2 = vcs.commit("x")
    vcs.checkout(empty)  # read-only: the next commit still follows c2
    vcs.add("y", "2")
    c3 = vcs.commit("y")
    assert vcs.history()[2]["parent"] == c2 and vcs.head == c3
    same = vcs.commit("no change")
    assert vcs.diff(c3, same)["nodes_visited"] == 2
    assert vcs.history()[3]["new_nodes"] == 0
    with pytest.raises(KeyError):
        vcs.rm("missing")
    with pytest.raises(KeyError):
        vcs.checkout("000000000000")
    with pytest.raises(KeyError):
        vcs.diff(c2, "nope")
    with pytest.raises(TypeError):
        vcs.add("z", b"bytes")
    with pytest.raises(TypeError):
        vcs.commit(None)
    with pytest.raises(ValueError):
        vcs.add("", "x")


# ===========================================================================
# DNA index
# ===========================================================================


def test_dna_small_anchors():
    idx = DNAIndex("ACGTACGTTGCA")
    assert idx.count("A", 2, 9) == 1
    assert [idx.kth_smallest(3, 8, k) for k in (1, 3, 6)] == ["A", "G", "T"]
    expected = {"ACG": 2, "CG": 2, "GT": 2, "T": 3, "TTG": 1, "GCA": 1, "AAA": 0}
    assert {p: idx.pattern_count(p) for p in expected} == expected
    assert idx.bwt() == "AC$TGAATCCGTG"
    text = "ACGTACGTTGCA$"
    assert idx.suffix_array().tolist() == sorted(range(len(text)), key=lambda i: text[i:])


@pytest.fixture(scope="module")
def big():
    """``synthetic_dna(100000, 42)`` and its index, built once."""
    seq = synthetic_dna(100000, 42)
    return seq, DNAIndex(seq)


def _naive_count(seq: str, pattern: str) -> int:
    """Overlapping occurrences by repeated ``str.find`` from one past each hit."""
    count, start = 0, seq.find(pattern)
    while start != -1:
        count += 1
        start = seq.find(pattern, start + 1)
    return count


def test_synthetic_dna_anchors(big):
    seq, idx = big
    assert seq.startswith("ATGCCTAGAAGTGTGTGATC")
    assert idx.frequencies() == {"A": 25021, "C": 24890, "G": 25154, "T": 24935}
    assert idx.count("A", 1000, 1999) == 254 == seq[1000:2000].count("A")
    assert [idx.kth_smallest(5000, 5099, k) for k in (1, 50, 100)] == ["A", "G", "T"]
    window = sorted(seq[5000:5100])
    assert [window[k - 1] for k in (1, 50, 100)] == ["A", "G", "T"]
    expected = {"GATTACA": 8, "ACGT": 399, "AAAA": 387, "CG": 6383}
    for pattern, want in expected.items():
        assert idx.pattern_count(pattern) == want == _naive_count(seq, pattern)


def test_synthetic_dna_is_exactly_the_spec_expression():
    for n, seed in ((0, 1), (1, 0), (37, 7), (1000, 42)):
        want = "".join(np.array(list("ACGT"))[np.random.default_rng(seed).integers(0, 4, n)])
        assert synthetic_dna(n, seed) == want
    with pytest.raises(ValueError):
        synthetic_dna(-1, 42)
    with pytest.raises(TypeError):
        synthetic_dna(10.0, 42)


def test_suffix_array_and_bwt_match_brute_force():
    rng = _rng("suffix_array")
    texts = ["", "a", "aaaa", "abab", "banana", "banana$", "mississippi$"]
    for _ in range(60):
        alphabet = "ab" if rng.random() < 0.5 else "ACGT"
        n = int(rng.integers(1, 200))
        body = "".join(alphabet[int(x)] for x in rng.integers(0, len(alphabet), n))
        texts.append(body)
        texts.append(body + "$")
    for text in texts:
        sa = build_suffix_array(text)
        assert sa.dtype == np.int64
        assert sa.tolist() == sorted(range(len(text)), key=lambda i: text[i:])
        if text.endswith("$"):
            rotations = sorted(text[i:] + text[:i] for i in range(len(text)))
            assert bwt_from_suffix_array(text, sa) == "".join(r[-1] for r in rotations)
    with pytest.raises(ValueError):
        bwt_from_suffix_array("abc", [0, 1])
    with pytest.raises(TypeError):
        build_suffix_array(["a", "b"])


def test_suffix_array_and_bwt_accept_any_str():
    # Regression (review): the UTF-32 encoding raised UnicodeEncodeError on a
    # lone surrogate, although the functions take any str; also code points
    # past the BMP and NUL.
    for text in ("b\ud800a\ud800", "\U0001F600a\x00\U0001F600$", "\udfff$"):
        sa = build_suffix_array(text)
        assert sa.tolist() == sorted(range(len(text)), key=lambda i: text[i:])
        assert bwt_from_suffix_array(text, sa) == "".join(text[i - 1] for i in sa.tolist())


@pytest.mark.parametrize("trial", range(6))
def test_dna_random_queries_match_brute_force(trial):
    rng = _rng("dna_queries", trial)
    n = int(rng.integers(1, 400))
    seq = "".join("ACGT"[int(x)] for x in rng.integers(0, 4 if trial else 2, n))
    idx = DNAIndex(seq)
    assert idx.frequencies() == {b: seq.count(b) for b in "ACGT"}
    for _ in range(200):
        l = int(rng.integers(0, n))
        r = int(rng.integers(l, n))
        base = "ACGT"[int(rng.integers(0, 4))]
        assert idx.count(base, l, r) == seq[l : r + 1].count(base)
        k = int(rng.integers(1, r - l + 2))
        assert idx.kth_smallest(l, r, k) == sorted(seq[l : r + 1])[k - 1]
    for _ in range(100):
        m = int(rng.integers(1, 7))
        if rng.random() < 0.5 and n >= m:
            start = int(rng.integers(0, n - m + 1))
            pattern = seq[start : start + m]
        else:
            pattern = "".join("ACGT"[int(x)] for x in rng.integers(0, 4, m))
        assert idx.pattern_count(pattern) == _naive_count(seq, pattern)
    assert idx.pattern_count(seq) == 1
    assert idx.pattern_count(seq + "A") == 0


def test_dna_edge_cases_and_errors():
    idx = DNAIndex("AAAA")
    assert idx.pattern_count("AA") == 3
    assert idx.kth_smallest(0, 3, 4) == "A"
    empty = DNAIndex("")
    assert len(empty) == 0 and empty.bwt() == "$" and empty.pattern_count("A") == 0
    assert empty.frequencies() == {"A": 0, "C": 0, "G": 0, "T": 0}
    with pytest.raises(IndexError):
        empty.count("A", 0, 0)
    idx = DNAIndex("ACGTACGTTGCA")
    assert idx.pattern_count("acg") == 0 and idx.pattern_count("A$") == 0
    with pytest.raises(ValueError):
        idx.count("N", 0, 3)
    with pytest.raises(ValueError):
        idx.count("A", 5, 4)
    with pytest.raises(IndexError):
        idx.count("A", 0, 12)
    with pytest.raises(IndexError):
        idx.kth_smallest(0, 3, 5)
    with pytest.raises(ValueError):
        idx.pattern_count("")
    with pytest.raises(TypeError):
        idx.pattern_count(None)
    with pytest.raises(ValueError):
        DNAIndex("ACG$")
    with pytest.raises(TypeError):
        DNAIndex(list("ACGT"))


def test_committed_synthetic_fasta():
    path = Path(DEFAULT_FASTA)
    assert path == REPO / "src" / "advds" / "examples" / "synthetic_fragment.fa"
    assert path.exists() and path.stat().st_size < 100 * 1024
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0].startswith(">") and "synthetic" in lines[0] and "not a real genome" in lines[0]
    assert all(len(line) == 60 for line in lines[1:-1]) and 0 < len(lines[-1]) <= 60
    assert read_fasta(path) == synthetic_dna(20000, 42)


def test_make_synthetic_fasta_round_trip(tmp_path):
    target = tmp_path / "nested" / "fragment.fa"
    returned = make_synthetic_fasta(target, n=1234, seed=7)
    assert returned == str(target)
    assert read_fasta(target) == synthetic_dna(1234, 7)
    lines = target.read_text(encoding="ascii").splitlines()
    assert [len(x) for x in lines[1:]] == [60] * 20 + [34]


# ===========================================================================
# Demos, packaging and the spec 1.1 rule
# ===========================================================================


def test_package_exports_resolve():
    for package in ("src.advds", "src.advds.trees", "src.advds.succinct",
                    "src.advds.cache_oblivious", "src.advds.applications"):
        module = importlib.import_module(package)
        assert module.__doc__
        for name in module.__all__:
            assert hasattr(module, name), f"{package} does not define {name}"


@pytest.mark.parametrize(
    "module",
    [
        "src.advds",
        "src.advds.applications.dna_index",
        "src.advds.applications.version_control",
        "src.advds.trees.persistent_tree",
        "src.advds.succinct.wavelet_tree",
        "src.advds.cache_oblivious.matrix_ops",
    ],
)
def test_fresh_import_has_no_cycle(module):
    out = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr


#: Calls that must not stand in for a structure's query in src/advds (spec 1.1).
_BANNED_ATTRS = {"bit_count", "cumsum", "searchsorted", "count_nonzero", "unpackbits", "sum"}


@pytest.mark.parametrize("module", [range_queries, version_control, dna_index])
def test_applications_follow_rule_1_1(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            assert "bisect" not in names
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                assert func.id not in {"sorted", "sum"}, f"{func.id}() in {module.__name__}"
            if isinstance(func, ast.Attribute):
                assert func.attr not in _BANNED_ATTRS, f".{func.attr}() in {module.__name__}"
    # np.lexsort appears only inside the suffix-array builder.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "build_suffix_array":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute):
                    assert inner.attr != "lexsort", f"lexsort in {node.name}"
