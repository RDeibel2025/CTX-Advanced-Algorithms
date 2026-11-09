"""Week 11 tests: the unified ``StringSearchEngine`` (spec 3.5) and the 9.5 batteries.

Hand-checked anchors used here:

* ``abracadabra`` / ``abra`` occurs at ``[0, 7]`` and ``abababab`` / ``abab`` at
  ``[0, 2, 4]`` (spec 9.1).
* ``"naïve café naïve"``: ``naïve`` is characters 0-4, a space at 5, ``café``
  at 6-9, a space at 10, and ``naïve`` again at 11, so ``[0, 11]`` (spec 9.5).
* ``"a" * 200`` with ``"a" * 5``: every start 0..195 matches, 196 positions.

Timing-based behaviour (``stats`` and ``recommend``) is tested with a scripted
clock: ``search_engine.time`` is replaced so that ``perf_counter`` returns
chosen values, two calls around an index build and two around each query. The
expected break-even counts are then worked by hand, e.g. KMP 0.25 s a query,
a 1.0 s suffix array build and 0.0625 s suffix array queries give
ceil(1.0 / (0.25 - 0.0625)) = ceil(5.33) = 6 queries. No assertion depends on
real elapsed time.

The agreement battery checks every method against ``naive_search``, the
independent window-by-window baseline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import pytest

from src.string_search_engine.core import search_engine
from src.string_search_engine.core.kmp import naive_search
from src.string_search_engine.core.rabin_karp import rabin_karp_search
from src.string_search_engine.core.search_engine import (
    METHODS,
    SearchResult,
    StringSearchEngine,
)
from src.utils.random_utils import derive_seed, make_rng

MASTER_SEED = 1109
ALPHABETS = ["ab", "ACGT", "abcdefghijklmnopqrstuvwxyz"]


def scripted_clock(monkeypatch, times: Iterable[float]) -> None:
    """Make the engine's ``time.perf_counter`` return ``times`` in order."""
    ticks = iter(times)
    monkeypatch.setattr(search_engine, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))


# ------------------------------------------------------------ lazy indexing


def test_indexes_are_built_lazily_and_once(monkeypatch):
    built = {"sa": [], "tree": []}
    real_sa, real_tree = search_engine.SuffixArray, search_engine.SuffixTree

    def counting_sa(text):
        built["sa"].append(text)
        return real_sa(text)

    def counting_tree(text, sa=None, lcp=None):
        built["tree"].append((text, sa is not None, lcp is not None))
        return real_tree(text, sa=sa, lcp=lcp)

    monkeypatch.setattr(search_engine, "SuffixArray", counting_sa)
    monkeypatch.setattr(search_engine, "SuffixTree", counting_tree)

    engine = StringSearchEngine("abracadabra")
    engine.find("abra", method="kmp")
    engine.find("abra", method="rk")
    assert built == {"sa": [], "tree": []}
    assert engine.preprocessing_time_s == {"kmp": 0.0, "rk": 0.0}

    first = engine.find("abra", method="sa")
    second = engine.find("cad", method="sa")
    assert built["sa"] == ["abracadabra"]
    assert built["tree"] == []
    recorded = engine.preprocessing_time_s["sa"]
    assert isinstance(recorded, float) and recorded >= 0.0
    assert first.preprocessing_time_s == recorded == second.preprocessing_time_s

    engine.find("abra", method="tree")
    engine.find("bra", method="tree")
    # The tree reuses the suffix array and LCP of text + "\x00".
    assert built["sa"] == ["abracadabra", "abracadabra\x00"]
    assert built["tree"] == [("abracadabra", True, True)]
    assert set(engine.preprocessing_time_s) == set(METHODS)
    assert engine.preprocessing_time_s["sa"] == recorded


def test_preprocessing_recorded_once_with_scripted_clock(monkeypatch):
    # sa build 1.0 -> 3.0 (2.0 s), then two queries; no second build.
    scripted_clock(monkeypatch, [1.0, 3.0, 4.0, 4.5, 5.0, 5.25])
    engine = StringSearchEngine("abracadabra")
    first = engine.find("abra", method="sa")
    second = engine.find("a", method="sa")
    assert engine.preprocessing_time_s == {"sa": 2.0}
    assert first.preprocessing_time_s == second.preprocessing_time_s == 2.0
    assert (first.search_time_s, second.search_time_s) == (0.5, 0.25)


# ------------------------------------------------------------ SearchResult


@pytest.mark.parametrize("method", METHODS)
def test_search_result_fields(method):
    engine = StringSearchEngine("abracadabra")
    result = engine.find("abra", method=method)
    assert isinstance(result, SearchResult)
    assert result.method == method
    assert result.pattern == "abra"
    assert result.positions == [0, 7]
    assert result.count == 2
    assert isinstance(result.search_time_s, float) and result.search_time_s >= 0.0
    assert result.preprocessing_time_s == engine.preprocessing_time_s[method]
    if method in ("kmp", "rk"):
        assert result.preprocessing_time_s == 0.0


def test_default_method_is_kmp():
    assert StringSearchEngine("abracadabra").find("abra").method == "kmp"


@pytest.mark.parametrize("method", METHODS)
def test_shared_conventions(method):
    engine = StringSearchEngine("abracadabra")
    with pytest.raises(ValueError):
        engine.find("", method=method)
    with pytest.raises(TypeError):
        engine.find(5, method=method)
    assert engine.find("abracadabras", method=method).positions == []
    assert StringSearchEngine("").find("a", method=method).positions == []
    assert StringSearchEngine("aaaaa").find("aa", method=method).positions == [0, 1, 2, 3]


def test_bad_inputs():
    with pytest.raises(TypeError):
        StringSearchEngine(b"abc")
    with pytest.raises(ValueError):
        StringSearchEngine("abc").find("a", method="boyer-moore")
    engine = StringSearchEngine("a\x00b")
    assert engine.find("b", method="sa").positions == [2]
    with pytest.raises(ValueError):
        engine.find("b", method="tree")


# ------------------------------------------------------------ find_all


def test_find_all_agrees():
    results = StringSearchEngine("abababab").find_all("abab")
    assert list(results) == list(METHODS)
    for method, result in results.items():
        assert result.method == method
        assert result.positions == [0, 2, 4]
        assert result.count == 3


def test_find_all_raises_on_disagreement(monkeypatch):
    monkeypatch.setattr(search_engine, "rabin_karp_search", lambda text, pattern: [])
    with pytest.raises(AssertionError):
        StringSearchEngine("abracadabra").find_all("abra")


# ------------------------------------------------------------ stats


def test_stats_before_any_query():
    report = StringSearchEngine("banana").stats()
    assert set(report) == set(METHODS)
    for method in METHODS:
        assert set(report[method]) == {"queries", "mean_search_s", "preprocessing_s", "index_memory_kb"}
        assert report[method]["queries"] == 0
        assert report[method]["mean_search_s"] is None
    assert report["kmp"]["preprocessing_s"] == report["rk"]["preprocessing_s"] == 0.0
    assert report["sa"]["preprocessing_s"] is None and report["tree"]["preprocessing_s"] is None
    assert report["kmp"]["index_memory_kb"] == report["rk"]["index_memory_kb"] == 0.0
    assert report["sa"]["index_memory_kb"] is None and report["tree"]["index_memory_kb"] is None


def test_stats_from_scripted_times(monkeypatch):
    scripted_clock(
        monkeypatch,
        [
            0.0, 0.25,            # kmp query: 0.25 s
            1.0, 1.75,            # kmp query: 0.75 s
            2.0, 3.0, 4.0, 4.5,   # sa build 1.0 s, query 0.5 s
        ],
    )
    engine = StringSearchEngine("banana")
    engine.find("ana", method="kmp")
    engine.find("na", method="kmp")
    engine.find("ana", method="sa")
    report = engine.stats()
    assert report["kmp"]["queries"] == 2
    assert report["kmp"]["mean_search_s"] == 0.5
    assert report["sa"]["queries"] == 1
    assert report["sa"]["mean_search_s"] == 0.5
    assert report["sa"]["preprocessing_s"] == 1.0
    assert report["sa"]["index_memory_kb"] > 0.0
    assert report["tree"]["queries"] == 0


def test_tree_memory_exceeds_suffix_array_memory():
    engine = StringSearchEngine("mississippi" * 20)
    engine.find_all("issi")
    report = engine.stats()
    assert report["tree"]["index_memory_kb"] > report["sa"]["index_memory_kb"] > 0.0


# ------------------------------------------------------------ recommend


def test_recommend_without_measurements():
    engine = StringSearchEngine("banana")
    sentence = engine.recommend(10)
    assert isinstance(sentence, str) and sentence.endswith(".")
    assert "not enough" in sentence.lower()
    engine.find("ana", method="kmp")
    assert "not enough" in engine.recommend(10).lower()


def test_recommend_suffix_array_wins_after_break_even(monkeypatch):
    # KMP 0.25 s a query; SA build 1.0 s and 0.0625 s a query.
    # Break-even ceil(1.0 / 0.1875) = 6. Ten queries: KMP 2.5 s, SA 1.625 s.
    scripted_clock(monkeypatch, [0.0, 0.25, 1.0, 2.0, 3.0, 3.0625])
    engine = StringSearchEngine("abracadabra")
    engine.find("abra", method="kmp")
    engine.find("abra", method="sa")
    sentence = engine.recommend(10)
    assert sentence.startswith("For 10 queries on this text, the suffix array is fastest overall")
    assert "repaid after 6 queries" in sentence
    assert "1.00 s build" in sentence
    assert "62.50 ms" in sentence and "250.00 ms" in sentence
    assert sentence.endswith(".")


def test_recommend_kmp_wins_below_break_even(monkeypatch):
    # Same measurements; three queries: KMP 0.75 s, SA 1.1875 s.
    scripted_clock(monkeypatch, [0.0, 0.25, 1.0, 2.0, 3.0, 3.0625])
    engine = StringSearchEngine("abracadabra")
    engine.find("abra", method="kmp")
    engine.find("abra", method="sa")
    sentence = engine.recommend(3)
    assert sentence.startswith("For 3 queries on this text, KMP is fastest overall")
    assert "repaid after 6 queries" in sentence


def test_recommend_break_even_follows_the_measurements(monkeypatch):
    # KMP 0.5 s, SA build 3.0 s, SA query 0.25 s: ceil(3.0 / 0.25) = 12.
    scripted_clock(monkeypatch, [0.0, 0.5, 1.0, 4.0, 5.0, 5.25])
    engine = StringSearchEngine("abracadabra")
    engine.find("abra", method="kmp")
    engine.find("abra", method="sa")
    assert "repaid after 12 queries" in engine.recommend(100)


def test_recommend_never_repaid_when_sa_queries_are_not_faster(monkeypatch):
    # KMP 0.25 s a query, SA query 0.5 s: the denominator is negative.
    scripted_clock(monkeypatch, [0.0, 0.25, 1.0, 2.0, 3.0, 3.5])
    engine = StringSearchEngine("abracadabra")
    engine.find("abra", method="kmp")
    engine.find("abra", method="sa")
    sentence = engine.recommend(1000)
    assert sentence.startswith("For 1000 queries on this text, KMP is fastest overall")
    assert "never repaid" in sentence


def test_recommend_uses_pattern_length(monkeypatch):
    # KMP: 0.25 s on a 4-character pattern, 0.75 s on a 2-character one.
    # SA: build 1.0 s, 0.0625 s on the 4-character pattern.
    # Length 4: ceil(1.0 / 0.1875) = 6. All lengths: ceil(1.0 / 0.4375) = 3.
    scripted_clock(monkeypatch, [0.0, 0.25, 1.0, 1.75, 2.0, 3.0, 4.0, 4.0625])
    engine = StringSearchEngine("abracadabra")
    engine.find("abra", method="kmp")
    engine.find("ab", method="kmp")
    engine.find("abra", method="sa")
    by_length = engine.recommend(10, pattern_length=4)
    assert by_length.startswith("For 10 queries on this text with 4-character patterns, ")
    assert "repaid after 6 queries" in by_length
    assert "repaid after 3 queries" in engine.recommend(10)


def test_recommend_validates_arguments():
    engine = StringSearchEngine("banana")
    with pytest.raises(ValueError):
        engine.recommend(0)
    with pytest.raises(TypeError):
        engine.recommend(True)
    with pytest.raises(ValueError):
        engine.recommend(5, pattern_length=0)


# ------------------------------------------------------------ 9.5 batteries


def test_agreement_battery_300_pairs():
    pairs = 0
    for alphabet in ALPHABETS:
        for trial in range(100):
            rng = make_rng(derive_seed(MASTER_SEED, "engine-battery", alphabet, trial))
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300)))
            m = rng.randint(1, 12)
            if trial % 2 == 0 and len(text) >= m:
                start = rng.randint(0, len(text) - m)
                pattern = text[start : start + m]
            else:
                pattern = "".join(rng.choice(alphabet) for _ in range(m))
            expected = naive_search(text, pattern)
            engine = StringSearchEngine(text)
            for method in METHODS:
                assert engine.find(pattern, method=method).positions == expected, (method, text, pattern)
            assert rabin_karp_search(text, pattern, modulus=7) == expected, (text, pattern)
            pairs += 1
    assert pairs == 300


def test_unicode_check():
    text = "naïve café naïve"
    assert [ord(c) for c in text[:5]] == [0x6E, 0x61, 0xEF, 0x76, 0x65]
    assert ord(text[9]) == 0xE9
    engine = StringSearchEngine(text)
    for method in METHODS:
        assert engine.find("naïve", method=method).positions == [0, 11]


def test_worst_case_all_a():
    engine = StringSearchEngine("a" * 200)
    for method in METHODS:
        result = engine.find("a" * 5, method=method)
        assert result.count == 196
        assert result.positions == list(range(196))
    assert rabin_karp_search("a" * 200, "a" * 5, modulus=7) == list(range(196))
