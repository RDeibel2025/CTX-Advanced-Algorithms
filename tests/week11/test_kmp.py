"""Week 11 tests: KMP and the naive matcher on the spec 9.1 anchors.

Every expected value is from the spec 9.1 tables or traced by hand, never from
the code under test.

Hand trace of ``ababaca`` (LPS ``[0, 0, 1, 2, 3, 0, 1]``) in
``bacbababaabcbab``. q is the number of pattern characters matched; each line
lists the comparisons made at that text index.

===== ==== ============================================== ======
Index Char Comparisons and fallbacks                       Count
===== ==== ============================================== ======
0     b    p[0]=a no                                      1
1     a    p[0]=a yes, q=1                                1
2     c    p[1]=b no, q 1->0; p[0]=a no                   2
3     b    p[0]=a no                                      1
4-8   ababa p[q] yes each time, q rises 1..5              5
9     a    p[5]=c no, q 5->3; p[3]=b no, q 3->1;          4
           p[1]=b no, q 1->0; p[0]=a yes, q=1
10    b    p[1]=b yes, q=2                                1
11    c    p[2]=a no, q 2->0; p[0]=a no                   2
12    b    p[0]=a no                                      1
13-14 ab   p[q] yes, q rises to 2                         2
===== ==== ============================================== ======

So the fallbacks are (2, 1, 0), (9, 5, 3), (9, 3, 1), (9, 1, 0), (11, 2, 0),
there are 20 scan comparisons and no match. Building the LPS array takes 8
comparisons: one each at i = 1, 2, 3, 4, 6 and three at i = 5 (c against
p[3], p[1] and p[0]).

Smaller traces used for the counts:

* ``ab`` in ``aab``: LPS [0, 0] takes 1 comparison. Index 0: 1; index 1:
  p[1]=b no, q 1->0, p[0]=a yes (2); index 2: p[1]=b yes, match at 1 (1).
  Total 4, one fallback (1, 1, 0). Naive: shift 0 makes 2 (a yes, b no) and
  shift 1 makes 2, total 4.
* ``ab`` in ``aaab``: indices 1 and 2 each fall back 1->0 (2 comparisons
  each), indices 0 and 3 take 1 each: 6. Naive: 2 per shift over 3 shifts, 6.
* ``aa`` in ``aaaaa``: after the first match q stays at 1, so every text
  character takes exactly one comparison: 5, with LPS [0, 1] taking 1.
* ``abab`` in ``abababab``: LPS [0, 0, 1, 2] takes 3 comparisons and every
  text character matches on its first comparison: 8, no fallbacks.
* ``"a" * 5`` in ``"a" * 50``: naive makes (50 - 5 + 1) * 5 = 230 comparisons
  and finds the 46 shifts 0..45.
"""

from __future__ import annotations

import pytest

from src.string_search_engine.core.kmp import (
    MatchStats,
    compute_lps,
    kmp_search,
    naive_search,
)

LPS_ANCHORS = [
    ("ababaca", [0, 0, 1, 2, 3, 0, 1]),
    ("aabaaab", [0, 1, 0, 1, 2, 2, 3]),
    ("abcabcabc", [0, 0, 0, 1, 2, 3, 4, 5, 6]),
    ("aaaa", [0, 1, 2, 3]),
    ("abacabab", [0, 0, 1, 0, 1, 2, 3, 2]),
]

MATCH_ANCHORS = [
    ("abracadabra", "abra", [0, 7]),
    ("aaaaa", "aa", [0, 1, 2, 3]),
    ("abababab", "abab", [0, 2, 4]),
    ("mississippi", "issi", [1, 4]),
    ("ab", "abc", []),
]

CLRS_TEXT = "bacbababaabcbab"
CLRS_PATTERN = "ababaca"


@pytest.mark.parametrize("pattern, expected", LPS_ANCHORS)
def test_lps_anchors(pattern, expected):
    assert compute_lps(pattern) == expected


@pytest.mark.parametrize("text, pattern, expected", MATCH_ANCHORS)
def test_kmp_match_anchors(text, pattern, expected):
    assert kmp_search(text, pattern) == expected


@pytest.mark.parametrize("text, pattern, expected", MATCH_ANCHORS)
def test_naive_match_anchors(text, pattern, expected):
    assert naive_search(text, pattern) == expected


@pytest.mark.parametrize("text, pattern, expected", MATCH_ANCHORS)
def test_flags_do_not_change_positions(text, pattern, expected):
    positions, events = kmp_search(text, pattern, trace=True)
    assert positions == expected
    positions, st = kmp_search(text, pattern, stats=True)
    assert positions == expected and isinstance(st, MatchStats)
    positions, events, st = kmp_search(text, pattern, trace=True, stats=True)
    assert positions == expected
    positions, st = naive_search(text, pattern, stats=True)
    assert positions == expected


@pytest.mark.parametrize("search", [kmp_search, naive_search])
def test_empty_pattern_raises(search):
    with pytest.raises(ValueError):
        search("abc", "")


def test_empty_text_returns_empty():
    assert kmp_search("", "a") == []
    assert naive_search("", "a") == []
    assert kmp_search("", "a", trace=True, stats=True) == ([], [], MatchStats(0, 0))


def test_fallback_trace_clrs_example():
    positions, events = kmp_search(CLRS_TEXT, CLRS_PATTERN, trace=True)
    assert positions == []
    assert events
    assert all(q_after < q_before for _, q_before, q_after in events)
    assert events == [(2, 1, 0), (9, 5, 3), (9, 3, 1), (9, 1, 0), (11, 2, 0)]


def test_comparisons_clrs_example():
    _, _, st = kmp_search(CLRS_TEXT, CLRS_PATTERN, trace=True, stats=True)
    assert st == MatchStats(comparisons=20, lps_comparisons=8)


@pytest.mark.parametrize(
    "text, pattern, positions, events, comparisons, lps_comparisons",
    [
        ("aab", "ab", [1], [(1, 1, 0)], 4, 1),
        ("aaab", "ab", [2], [(1, 1, 0), (2, 1, 0)], 6, 1),
        ("aaaaa", "aa", [0, 1, 2, 3], [], 5, 1),
        ("abababab", "abab", [0, 2, 4], [], 8, 3),
    ],
)
def test_kmp_hand_traced_counts(text, pattern, positions, events, comparisons, lps_comparisons):
    got = kmp_search(text, pattern, trace=True, stats=True)
    assert got == (positions, events, MatchStats(comparisons, lps_comparisons))


@pytest.mark.parametrize(
    "text, pattern, positions, comparisons",
    [("aab", "ab", [1], 4), ("aaab", "ab", [2], 6)],
)
def test_naive_hand_traced_counts(text, pattern, positions, comparisons):
    assert naive_search(text, pattern, stats=True) == (positions, MatchStats(comparisons, 0))


def test_naive_worst_case_comparisons():
    positions, st = naive_search("a" * 50, "a" * 5, stats=True)
    assert positions == list(range(46))
    assert st.comparisons == (50 - 5 + 1) * 5 == 230


def test_unicode():
    assert kmp_search("naïve café naïve", "naïve") == [0, 11]
    assert naive_search("naïve café naïve", "naïve") == [0, 11]
