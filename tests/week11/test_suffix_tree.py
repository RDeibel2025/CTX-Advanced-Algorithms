"""Week 11 tests: the suffix tree built from SA + LCP over ``text + "\\x00"``.

The anchor rows are the spec 9.4 table, checked by hand. ``banana$`` has 7
suffixes, so 7 leaves; its repeated right-branching substrings are ``a``
(followed by ``$`` or ``n``), ``na`` (``$`` or ``n``) and ``ana`` (``$`` or
``n``), so 3 internal nodes. ``mississippi$`` has 12 leaves and internal nodes
``i, p, s, si, ssi, issi``; ``abracadabra$`` has 12 leaves and ``a, ra, bra,
abra``. The distinct-substring counts (15, 53, 54) and the longest repeats
(``ana``, ``issi``, ``abra``) are the same as the suffix array table in 9.3.
In ``banana`` the prefix ``ana`` of ``anaz`` occurs and ``anaz`` does not, so
the longest prefix match is 3.

Every other expected value comes from an independent check: a window scan for
:meth:`SuffixTree.find` and :meth:`SuffixTree.count`, and the O(n^2)
insert-every-suffix builder :meth:`SuffixTree.from_naive` for the structure.
"""

from __future__ import annotations

import re
from typing import List

import pytest

from src.string_search_engine.core.suffix_array import SuffixArray
from src.string_search_engine.core.suffix_tree import SuffixTree
from src.utils.random_utils import derive_seed, make_rng

MASTER_SEED = 1109

ANCHORS = [
    ("banana", 7, 3, ["a", "na", "ana"], 15, "ana"),
    ("mississippi", 12, 6, ["i", "p", "s", "si", "ssi", "issi"], 53, "issi"),
    ("abracadabra", 12, 4, ["a", "ra", "bra", "abra"], 54, "abra"),
]

ALPHABETS = ["ab", "ACGT", "abcdefghijklmnopqrstuvwxyz", "aé\U0001d11e"]


def brute_find(text: str, pattern: str) -> List[int]:
    m = len(pattern)
    return [i for i in range(len(text) - m + 1) if text[i : i + m] == pattern]


def random_string(rng, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


# --------------------------------------------------------------------- anchors


@pytest.mark.parametrize("text, leaves, internal, labels, distinct, lrs", ANCHORS)
def test_anchor_row(text, leaves, internal, labels, distinct, lrs):
    tree = SuffixTree(text)
    total = leaves + internal
    assert tree.node_counts() == {"leaves": leaves, "internal": internal, "total": total}
    assert tree.internal_path_labels() == sorted(labels)
    assert tree.count_distinct_substrings() == distinct
    assert tree.longest_repeated_substring() == lrs


@pytest.mark.parametrize("text, leaves, internal, labels, distinct, lrs", ANCHORS)
def test_anchor_row_naive_builder(text, leaves, internal, labels, distinct, lrs):
    tree = SuffixTree.from_naive(text)
    total = leaves + internal
    assert tree.node_counts() == {"leaves": leaves, "internal": internal, "total": total}
    assert tree.internal_path_labels() == sorted(labels)
    assert tree.count_distinct_substrings() == distinct
    assert tree.longest_repeated_substring() == lrs


def test_longest_prefix_match_banana():
    tree = SuffixTree("banana")
    assert tree.longest_prefix_match("anaz") == 3
    assert tree.longest_prefix_match("z") == 0
    assert tree.longest_prefix_match("banana") == 6


def test_anchor_matches():
    assert SuffixTree("abracadabra").find("abra") == [0, 7]
    assert SuffixTree("aaaaa").find("aa") == [0, 1, 2, 3]
    assert SuffixTree("abababab").find("abab") == [0, 2, 4]
    assert SuffixTree("mississippi").find("issi") == [1, 4]
    assert SuffixTree("ab").find("abc") == []


# ----------------------------------------------------------------- conventions


def test_empty_pattern_raises():
    tree = SuffixTree("banana")
    for method in (tree.find, tree.count, tree.contains, tree.longest_prefix_match):
        with pytest.raises(ValueError):
            method("")


def test_text_with_terminator_raises():
    with pytest.raises(ValueError):
        SuffixTree("ab\x00c")
    with pytest.raises(ValueError):
        SuffixTree.from_naive("\x00")


def test_empty_text():
    tree = SuffixTree("")
    assert tree.find("a") == []
    assert tree.count("a") == 0
    assert tree.node_counts() == {"leaves": 1, "internal": 0, "total": 1}
    assert tree.count_distinct_substrings() == 0
    assert tree.longest_repeated_substring() == ""


def test_terminator_in_pattern_never_matches():
    tree = SuffixTree("banana")
    assert tree.find("a\x00") == []
    assert not tree.contains("\x00")
    assert tree.longest_prefix_match("na\x00") == 2


def test_unicode():
    assert SuffixTree("naïve café naïve").find("naïve") == [0, 11]


def test_given_sa_and_lcp_build_same_tree():
    text = "mississippi"
    index = SuffixArray(text + "\x00")
    given = SuffixTree(text, sa=index.sa, lcp=index.lcp)
    assert given.internal_path_labels() == SuffixTree(text).internal_path_labels()
    assert given.find("ssi") == [2, 5]
    with pytest.raises(ValueError):
        SuffixTree(text, sa=SuffixArray(text).sa, lcp=SuffixArray(text).lcp)


# ------------------------------------------------------------------- to_dot


def test_to_dot_is_valid_dot_text():
    tree = SuffixTree('ab"a\\b')
    dot = tree.to_dot()
    assert "$" in dot
    assert "\x00" not in dot
    lines = dot.split("\n")
    assert lines[0] == "digraph SuffixTree {"
    assert lines[-1] == "}"
    quoted = r'"(?:[^"\\]|\\.)*"'
    node_line = re.compile(rf"^  n\d+ \[(?:shape=box, )?label={quoted}\];$")
    edge_line = re.compile(rf"^  n\d+ -> n\d+ \[label={quoted}\];$")
    edges = 0
    for line in lines[2:-1]:
        if edge_line.match(line):
            edges += 1
        else:
            assert node_line.match(line), line
    assert edges == tree.node_counts()["total"]


# ------------------------------------------------------ agreement batteries


def test_find_and_count_match_brute_force():
    for trial in range(60):
        rng = make_rng(derive_seed(MASTER_SEED, "tree-find", trial))
        alphabet = ALPHABETS[trial % len(ALPHABETS)]
        text = random_string(rng, alphabet, rng.randint(0, 120))
        tree = SuffixTree(text)
        for _ in range(10):
            m = rng.randint(1, 8)
            if text and rng.random() < 0.5:
                start = rng.randint(0, max(0, len(text) - m))
                pattern = text[start : start + m]
            else:
                pattern = random_string(rng, alphabet, m)
            expected = brute_find(text, pattern)
            assert tree.find(pattern) == expected
            assert tree.count(pattern) == len(expected)
            assert tree.contains(pattern) == bool(expected)


def test_structure_matches_naive_builder():
    for trial in range(30):
        rng = make_rng(derive_seed(MASTER_SEED, "tree-naive", trial))
        alphabet = ALPHABETS[trial % len(ALPHABETS)]
        text = random_string(rng, alphabet, rng.randint(0, 25))
        fast = SuffixTree(text)
        naive = SuffixTree.from_naive(text)
        assert fast.node_counts()["leaves"] == naive.node_counts()["leaves"] == len(text) + 1
        assert fast.internal_path_labels() == naive.internal_path_labels()
        assert fast.node_counts() == naive.node_counts()
