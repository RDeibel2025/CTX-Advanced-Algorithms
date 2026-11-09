"""Week 11 tests: the suffix array, Kasai's LCP array and the queries built on them.

The anchor rows are the spec 9.3 table, checked by hand. For ``banana`` the
sorted suffixes are ``a, ana, anana, banana, na, nana`` (starts 5, 3, 1, 0, 4,
2); neighbours share ``-, a, ana, -, -, na``, so the LCP array is
``[0, 1, 3, 0, 0, 2]``, there are 6 * 7 / 2 - 6 = 15 distinct substrings and
the longest repeated substring is ``ana``. The other two rows work the same
way: ``mississippi`` gives 66 - 13 = 53 and ``issi``, ``abracadabra`` gives
66 - 12 = 54 and ``abra``.

Every other expected value comes from an independent brute force: Python's
own string order for the suffix array, pairwise prefix comparison for the LCP
array and the longest repeat, a set of all substrings for the distinct count
and a window scan for :meth:`SuffixArray.find`.
"""

from __future__ import annotations

from typing import List

import pytest

from src.string_search_engine.core.suffix_array import (
    SuffixArray,
    build_lcp,
    build_suffix_array,
)
from src.utils.random_utils import derive_seed, make_rng

MASTER_SEED = 1109

ANCHORS = [
    ("banana", [5, 3, 1, 0, 4, 2], [0, 1, 3, 0, 0, 2], 15, "ana"),
    (
        "mississippi",
        [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2],
        [0, 1, 1, 4, 0, 0, 1, 0, 2, 1, 3],
        53,
        "issi",
    ),
    (
        "abracadabra",
        [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2],
        [0, 1, 4, 1, 1, 0, 3, 0, 0, 0, 2],
        54,
        "abra",
    ),
]

ALPHABETS = ["ab", "ACGT", "abcdefghijklmnopqrstuvwxyz", "aéß\U0001d11e"]


# ---------------------------------------------------------------- brute force


def brute_sa(s: str) -> List[int]:
    return sorted(range(len(s)), key=lambda i: s[i:])


def common_prefix(a: str, b: str) -> int:
    h = 0
    while h < len(a) and h < len(b) and a[h] == b[h]:
        h += 1
    return h


def brute_lcp(s: str, sa: List[int]) -> List[int]:
    return [0] + [common_prefix(s[sa[r - 1] :], s[sa[r] :]) for r in range(1, len(sa))]


def brute_lrs_length(s: str) -> int:
    n = len(s)
    return max(
        (common_prefix(s[i:], s[j:]) for i in range(n) for j in range(i + 1, n)),
        default=0,
    )


def brute_distinct(s: str) -> int:
    return len({s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)})


def brute_find(s: str, p: str) -> List[int]:
    m = len(p)
    return [i for i in range(len(s) - m + 1) if s[i : i + m] == p]


def random_string(rng, max_len: int) -> str:
    alphabet = rng.choice(ALPHABETS)
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))


# -------------------------------------------------------------- spec 9.3 rows


@pytest.mark.parametrize("text, sa, lcp, distinct, lrs", ANCHORS)
def test_anchor_functions(text, sa, lcp, distinct, lrs):
    assert build_suffix_array(text) == sa
    assert build_lcp(text, sa) == lcp


@pytest.mark.parametrize("text, sa, lcp, distinct, lrs", ANCHORS)
def test_anchor_index(text, sa, lcp, distinct, lrs):
    index = SuffixArray(text)
    assert index.text == text
    assert index.sa == sa
    assert index.lcp == lcp
    assert index.count_distinct_substrings() == distinct
    assert index.longest_repeated_substring() == lrs
    stats = index.stats()
    assert stats == {
        "n": len(text),
        "mean_lcp": sum(lcp) / len(text),
        "max_lcp": len(lrs),
        "distinct_substrings": distinct,
        "lrs_length": len(lrs),
    }
    assert type(stats["distinct_substrings"]) is int
    assert set(index.build_time_s) == {"sa", "lcp"}
    assert all(isinstance(t, float) and t >= 0 for t in index.build_time_s.values())


@pytest.mark.parametrize(
    "text, pattern, positions",
    [
        ("abracadabra", "abra", [0, 7]),
        ("aaaaa", "aa", [0, 1, 2, 3]),
        ("abababab", "abab", [0, 2, 4]),
        ("mississippi", "issi", [1, 4]),
        ("ab", "abc", []),
        ("a" * 200, "a" * 5, list(range(196))),
    ],
)
def test_find_spec_matches(text, pattern, positions):
    index = SuffixArray(text)
    assert index.find(pattern) == positions
    assert index.count(pattern) == len(positions)


# ------------------------------------------------------------ random batteries


def test_random_sa_and_lcp_match_brute_force():
    for trial in range(100):
        s = random_string(make_rng(derive_seed(MASTER_SEED, "sa-lcp", trial)), 200)
        expected_sa = brute_sa(s)
        assert build_suffix_array(s) == expected_sa, s
        assert build_lcp(s, expected_sa) == brute_lcp(s, expected_sa), s
        index = SuffixArray(s)
        assert index.sa == expected_sa
        assert index.lcp == brute_lcp(s, expected_sa)


def test_random_distinct_and_lrs_match_brute_force():
    for trial in range(50):
        s = random_string(make_rng(derive_seed(MASTER_SEED, "distinct-lrs", trial)), 40)
        index = SuffixArray(s)
        assert index.count_distinct_substrings() == brute_distinct(s), s
        lrs = index.longest_repeated_substring()
        assert len(lrs) == brute_lrs_length(s), s
        assert index.stats()["lrs_length"] == len(lrs)
        if lrs:
            assert len(brute_find(s, lrs)) >= 2, s


def test_random_find_matches_scan():
    for trial in range(100):
        rng = make_rng(derive_seed(MASTER_SEED, "find", trial))
        s = random_string(rng, 200)
        index = SuffixArray(s)
        alphabet = sorted(set(s)) or ["a"]
        for q in range(6):
            m = rng.randint(1, 12)
            if q % 2 == 0 and len(s) >= m:
                start = rng.randrange(len(s) - m + 1)
                p = s[start : start + m]
            else:
                p = "".join(rng.choice(alphabet) for _ in range(m))
            expected = brute_find(s, p)
            assert index.find(p) == expected, (s, p)
            assert index.count(p) == len(expected), (s, p)


# ------------------------------------------------------------- Unicode, edges


def test_unicode_text():
    text = "naïve café naïve"
    index = SuffixArray(text)
    assert index.find("naïve") == [0, 11]
    assert index.find("é") == [9]
    assert index.sa == brute_sa(text)
    assert index.lcp == brute_lcp(text, brute_sa(text))
    astral = "\U0001d11ea\U0001d11eab\U0001d11e"
    astral_index = SuffixArray(astral)
    assert astral_index.sa == brute_sa(astral)
    assert astral_index.find("\U0001d11ea") == [0, 2]


def test_single_character_text():
    index = SuffixArray("a")
    assert index.sa == [0]
    assert index.lcp == [0]
    assert index.count_distinct_substrings() == 1
    assert index.longest_repeated_substring() == ""
    assert index.find("a") == [0]
    assert index.find("b") == []
    assert index.find("aa") == []
    assert index.stats() == {
        "n": 1,
        "mean_lcp": 0.0,
        "max_lcp": 0,
        "distinct_substrings": 1,
        "lrs_length": 0,
    }


def test_empty_text():
    assert build_suffix_array("") == []
    assert build_lcp("", []) == []
    index = SuffixArray("")
    assert index.sa == []
    assert index.lcp == []
    assert index.count_distinct_substrings() == 0
    assert index.longest_repeated_substring() == ""
    assert index.find("a") == []
    assert index.count("a") == 0
    assert index.stats() == {
        "n": 0,
        "mean_lcp": 0.0,
        "max_lcp": 0,
        "distinct_substrings": 0,
        "lrs_length": 0,
    }


def test_empty_pattern_raises():
    index = SuffixArray("banana")
    with pytest.raises(ValueError):
        index.find("")
    with pytest.raises(ValueError):
        index.count("")
    with pytest.raises(ValueError):
        SuffixArray("").find("")


def test_bad_suffix_array_rejected():
    with pytest.raises(ValueError):
        build_lcp("abc", [0, 1])
    with pytest.raises(ValueError):
        build_lcp("abc", [0, 1, 1])
    with pytest.raises(ValueError):
        build_lcp("abc", [0, 1, 3])
