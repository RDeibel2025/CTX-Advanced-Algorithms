"""One text, four matchers: a unified engine that times them and recommends one.

The engine
----------
:class:`StringSearchEngine` holds one text and answers ``find(pattern,
method)`` with any of the four Week 11 matchers: ``"kmp"``
(Knuth-Morris-Pratt), ``"rk"`` (Rabin-Karp), ``"sa"`` (suffix array) and
``"tree"`` (suffix tree). Every query is timed with ``time.perf_counter`` and
returned as a :class:`SearchResult`, so the same text can be searched by every
method and the costs compared directly. :meth:`StringSearchEngine.find_all`
runs all four and checks that they agree, :meth:`StringSearchEngine.stats`
summarises the measurements and :meth:`StringSearchEngine.recommend` turns
them into a sentence about which method to use for a given number of queries.

What it stores before it searches
---------------------------------
The engine itself stores nothing about the *text* until an index is needed:
KMP stores only the LPS array of each *pattern* (rebuilt per query, so its
time is part of the query) and Rabin-Karp only the *pattern*'s hash. On the
first ``"sa"`` query it builds and keeps the suffix array and LCP array of the
*text*; on the first ``"tree"`` query it builds the suffix array and LCP array
of ``text + "\\x00"`` and keeps the suffix tree built from them. Each build is
timed once, in ``preprocessing_time_s``, and reused by every later query.

The trade-off the recommendation measures
-----------------------------------------
KMP pays O(n + m) on every query and nothing up front. The suffix array pays
its build once and then O(m log n) per query. For q queries the totals are
``q * kmp`` and ``build_sa + q * sa``, so the suffix array wins from
``ceil(build_sa / (kmp - sa))`` queries on, when ``kmp > sa``. The engine
computes that break-even from its own measured means, never from fixed
numbers.

Conventions (shared with every Week 11 matcher): positions are 0-based, every
occurrence is reported including overlapping ones, the result is a sorted
``list[int]``, an empty pattern raises ``ValueError``, and a pattern longer
than the text (or an empty text) gives ``[]``.

Examples:
    >>> engine = StringSearchEngine("abracadabra")
    >>> engine.find("abra").positions
    [0, 7]
    >>> engine.find("abra", method="sa").count
    2
    >>> sorted(engine.find_all("bra"))
    ['kmp', 'rk', 'sa', 'tree']

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.string_search_engine.core.kmp import kmp_search
from src.string_search_engine.core.rabin_karp import rabin_karp_search
from src.string_search_engine.core.suffix_array import SuffixArray
from src.string_search_engine.core.suffix_tree import TERMINATOR, SuffixTree

__all__ = ["METHODS", "SearchResult", "StringSearchEngine"]

#: The method names :meth:`StringSearchEngine.find` accepts, in report order.
METHODS: Tuple[str, ...] = ("kmp", "rk", "sa", "tree")

#: How each method is named in the recommendation sentence.
_NAMES: Dict[str, str] = {
    "kmp": "KMP",
    "rk": "Rabin-Karp",
    "sa": "the suffix array",
    "tree": "the suffix tree",
}

#: Methods that keep no index over the text.
_NO_INDEX: Tuple[str, ...] = ("kmp", "rk")


@dataclass
class SearchResult:
    """The answer to one :meth:`StringSearchEngine.find` query, with its timing.

    Attributes:
        method: The method that answered, one of :data:`METHODS`.
        pattern: The pattern searched for.
        positions: Sorted 0-based start positions of every occurrence.
        count: ``len(positions)``.
        search_time_s: Seconds the query itself took (``time.perf_counter``).
            For ``"kmp"`` this includes building the pattern's LPS array.
        preprocessing_time_s: The method's one-time preprocessing cost in
            seconds, the same value on every query: the index build time for
            ``"sa"`` and ``"tree"``, ``0.0`` for ``"kmp"`` and ``"rk"``.

    Examples:
        >>> result = SearchResult("kmp", "ab", [0, 2], 2, 0.001, 0.0)
        >>> result.count, result.positions
        (2, [0, 2])
        >>> StringSearchEngine("abab").find("ab", method="rk").method
        'rk'
    """

    method: str
    pattern: str
    positions: List[int]
    count: int
    search_time_s: float
    preprocessing_time_s: float


def _format_seconds(seconds: float) -> str:
    """Format a duration for a sentence: seconds, milliseconds or microseconds.

    Args:
        seconds: A non-negative duration.

    Returns:
        The duration with two decimals and the largest unit that keeps it at
        least 1 (``"s"``, ``"ms"`` or ``"us"``).

    Examples:
        >>> _format_seconds(1.5), _format_seconds(0.25), _format_seconds(0.00004)
        ('1.50 s', '250.00 ms', '40.00 us')
    """
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.2f} us"


def _int_bytes(value: int) -> int:
    """Bytes of an int object CPython does not share (it caches -5 to 256)."""
    return sys.getsizeof(value) if (value > 256 or value < -5) else 0


def _int_list_bytes(values: List[int]) -> int:
    """Bytes of a list of ints: the list itself plus its uncached int objects."""
    return sys.getsizeof(values) + sum(_int_bytes(v) for v in values)


class StringSearchEngine:
    """Search one text with KMP, Rabin-Karp, a suffix array or a suffix tree.

    The engine keeps the text and builds an index only when a method first
    needs it: the suffix array (:class:`SuffixArray` of the text) on the first
    ``"sa"`` query and the suffix tree on the first ``"tree"`` query, the tree
    built from ``SuffixArray(text + "\\x00")``'s ``sa`` and ``lcp``. Each
    build is timed once into :attr:`preprocessing_time_s`; every query is
    timed separately and remembered (pattern length and seconds) for
    :meth:`stats` and :meth:`recommend`.

    Attributes:
        text: The text being searched.
        preprocessing_time_s: ``{method: seconds}``, recorded once per method
            on its first use: the build time for ``"sa"`` and ``"tree"``,
            ``0.0`` for ``"kmp"`` and ``"rk"``.

    Args:
        text: Any Python ``str``, Unicode included. It may be empty. The
            ``"tree"`` method needs a text without ``"\\x00"``.

    Raises:
        TypeError: If ``text`` is not a ``str``.

    Time Complexity:
        O(1) to construct. A ``"kmp"`` query is O(n + m), ``"rk"`` O(n + m)
        on average, ``"sa"`` O(m log n + occ log occ) after an O(n log n)
        build, ``"tree"`` O(m + occ log occ) after the same suffix array build
        plus an O(n) tree construction.

    Space Complexity:
        O(n) for the text and for each index once built; the suffix tree is
        several times larger than the suffix array.

    Examples:
        >>> engine = StringSearchEngine("mississippi")
        >>> "sa" in engine.preprocessing_time_s
        False
        >>> engine.find("issi", method="sa").positions
        [1, 4]
        >>> "sa" in engine.preprocessing_time_s
        True
        >>> engine.find("ss", method="tree").count
        2
    """

    def __init__(self, text: str) -> None:
        """Store the text; every index is built later, on first use.

        Args:
            text: Any Python ``str``. It may be empty.

        Returns:
            None.

        Raises:
            TypeError: If ``text`` is not a ``str``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1) beyond the text, which is not copied.

        Examples:
            >>> StringSearchEngine("banana").text
            'banana'
            >>> StringSearchEngine(b"banana")
            Traceback (most recent call last):
                ...
            TypeError: text must be a str, got bytes
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        self.text: str = text
        self.preprocessing_time_s: Dict[str, float] = {}
        self._indexes: Dict[str, Any] = {}
        self._timings: Dict[str, List[Tuple[int, float]]] = {method: [] for method in METHODS}
        self._memory_kb: Dict[str, float] = {}

    # ------------------------------------------------------------------ queries

    def _prepare(self, method: str) -> float:
        """Build the method's index on first use; return its preprocessing time.

        ``"kmp"`` and ``"rk"`` have nothing to build and record ``0.0``. For
        ``"tree"`` the text is checked for ``"\\x00"`` before anything is
        built, so a rejected text costs no suffix array.
        """
        if method in self.preprocessing_time_s:
            return self.preprocessing_time_s[method]
        if method in _NO_INDEX:
            self.preprocessing_time_s[method] = 0.0
            return 0.0
        if method == "tree" and TERMINATOR in self.text:
            raise ValueError("the 'tree' method needs a text without the terminator '\\x00'")
        start = time.perf_counter()
        if method == "sa":
            index: Any = SuffixArray(self.text)
        else:
            terminated = SuffixArray(self.text + TERMINATOR)
            index = SuffixTree(self.text, sa=terminated.sa, lcp=terminated.lcp)
        elapsed = time.perf_counter() - start
        self._indexes[method] = index
        self.preprocessing_time_s[method] = elapsed
        return elapsed

    def find(self, pattern: str, method: str = "kmp") -> SearchResult:
        """Find every occurrence of ``pattern`` with one method and time it.

        The pattern and method are checked first, then the method's index is
        built if this is its first use (timed once, not counted in the query),
        and then the query alone is timed.

        Args:
            pattern: The non-empty pattern to find.
            method: ``"kmp"`` (default), ``"rk"``, ``"sa"`` or ``"tree"``.

        Returns:
            A :class:`SearchResult` with the sorted positions, their count,
            the query time and the method's one-time preprocessing time.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty, ``method`` is unknown, or
                ``method="tree"`` and the text contains ``"\\x00"``.

        Time Complexity:
            As the chosen method (see the class docstring), plus the index
            build on the first ``"sa"`` or ``"tree"`` query.

        Space Complexity:
            O(occ) for the result, plus the index on its first build.

        Examples:
            >>> engine = StringSearchEngine("aaaaa")
            >>> engine.find("aa").positions
            [0, 1, 2, 3]
            >>> result = engine.find("aa", method="tree")
            >>> result.method, result.count
            ('tree', 4)
            >>> engine.find("aaaaaa", method="sa").positions
            []
            >>> engine.find("", method="rk")
            Traceback (most recent call last):
                ...
            ValueError: pattern must be non-empty
            >>> engine.find("a", method="bm")
            Traceback (most recent call last):
                ...
            ValueError: method must be one of ('kmp', 'rk', 'sa', 'tree'), got 'bm'
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a str, got {type(pattern).__name__}")
        if not pattern:
            raise ValueError("pattern must be non-empty")
        preprocessing = self._prepare(method)
        start = time.perf_counter()
        if method == "kmp":
            positions = kmp_search(self.text, pattern)
        elif method == "rk":
            positions = rabin_karp_search(self.text, pattern)
        else:
            positions = self._indexes[method].find(pattern)
        elapsed = time.perf_counter() - start
        self._timings[method].append((len(pattern), elapsed))
        return SearchResult(method, pattern, positions, len(positions), elapsed, preprocessing)

    def find_all(self, pattern: str) -> Dict[str, SearchResult]:
        """Run every method on ``pattern`` and check that they agree.

        Each method is run through :meth:`find`, so every query is timed and
        counted in :meth:`stats`, and indexes are built on first use.

        Args:
            pattern: The non-empty pattern to find.

        Returns:
            ``{method: SearchResult}`` for ``"kmp"``, ``"rk"``, ``"sa"`` and
            ``"tree"``, in that order.

        Raises:
            AssertionError: If any method's positions differ from KMP's.
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty or the text contains
                ``"\\x00"`` (the tree cannot index it).

        Time Complexity:
            The sum of the four methods' query times, plus any first builds.

        Space Complexity:
            O(occ) per method for the results, plus any first builds.

        Examples:
            >>> results = StringSearchEngine("abababab").find_all("abab")
            >>> {method: r.positions for method, r in results.items()}
            {'kmp': [0, 2, 4], 'rk': [0, 2, 4], 'sa': [0, 2, 4], 'tree': [0, 2, 4]}
        """
        results = {method: self.find(pattern, method) for method in METHODS}
        reference = results["kmp"].positions
        for method, result in results.items():
            if result.positions != reference:
                raise AssertionError(
                    f"methods disagree on {pattern!r}: kmp gives {reference}, "
                    f"{method} gives {result.positions}"
                )
        return results

    # ------------------------------------------------------------- measurements

    def _index_memory_kb(self, method: str) -> Optional[float]:
        """Estimate the method's index size in KB with ``sys.getsizeof``.

        ``0.0`` for ``"kmp"`` and ``"rk"`` (no index over the text), ``None``
        before the index is built. The suffix array counts its ``sa`` and
        ``lcp`` lists and their uncached int objects; the suffix tree counts
        its terminated copy of the text, every node object, every child dict
        and the nodes' uncached int fields (a leaf's ``end`` is one shared
        int, so it is not counted per leaf). The estimate is computed once and
        cached; the text itself is shared by every method and not counted.
        """
        if method in _NO_INDEX:
            return 0.0
        if method not in self._indexes:
            return None
        if method not in self._memory_kb:
            index = self._indexes[method]
            if method == "sa":
                total = _int_list_bytes(index.sa) + _int_list_bytes(index.lcp)
            else:
                total = sys.getsizeof(index.terminated)
                stack = [index.root]
                while stack:
                    node = stack.pop()
                    total += sys.getsizeof(node) + _int_bytes(node.start) + _int_bytes(node.depth)
                    children = node.children
                    if children is not None:
                        total += sys.getsizeof(children) + _int_bytes(node.end)
                        stack.extend(children.values())
            self._memory_kb[method] = total / 1024.0
        return self._memory_kb[method]

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Summarise every method's measurements so far.

        Args:
            None.

        Returns:
            ``{method: {"queries", "mean_search_s", "preprocessing_s",
            "index_memory_kb"}}`` for all four methods. ``queries`` is the
            number of queries run; ``mean_search_s`` their mean time, or
            ``None`` before the first; ``preprocessing_s`` the one-time build
            time (``0.0`` for ``"kmp"`` and ``"rk"``, ``None`` for an index not
            built yet); ``index_memory_kb`` a ``sys.getsizeof`` estimate of the
            index (``0.0`` for ``"kmp"`` and ``"rk"``, which keep no index over
            the text, ``None`` before a build).

        Raises:
            Nothing.

        Time Complexity:
            O(Q) over the Q recorded queries, plus one O(n) walk of each built
            index the first time its memory is estimated.

        Space Complexity:
            O(1) beyond the result (O(height) stack for the tree walk).

        Examples:
            >>> engine = StringSearchEngine("banana")
            >>> _ = engine.find("ana", method="sa")
            >>> _ = engine.find("na", method="sa")
            >>> report = engine.stats()
            >>> report["sa"]["queries"], report["kmp"]["queries"]
            (2, 0)
            >>> report["kmp"]["mean_search_s"], report["tree"]["preprocessing_s"]
            (None, None)
            >>> report["sa"]["index_memory_kb"] > 0, report["rk"]["index_memory_kb"]
            (True, 0.0)
        """
        report: Dict[str, Dict[str, Any]] = {}
        for method in METHODS:
            times = [seconds for _, seconds in self._timings[method]]
            if method in _NO_INDEX:
                preprocessing: Optional[float] = 0.0
            else:
                preprocessing = self.preprocessing_time_s.get(method)
            report[method] = {
                "queries": len(times),
                "mean_search_s": sum(times) / len(times) if times else None,
                "preprocessing_s": preprocessing,
                "index_memory_kb": self._index_memory_kb(method),
            }
        return report

    def _mean_query_s(self, method: str, pattern_length: Optional[int]) -> Optional[float]:
        """Mean measured query time, over queries of ``pattern_length`` if any.

        When ``pattern_length`` is given and the method has queries of that
        length, only those are averaged; otherwise every query of the method
        is. ``None`` when the method has no queries.
        """
        records = self._timings[method]
        if pattern_length is not None:
            same = [seconds for length, seconds in records if length == pattern_length]
            if same:
                return sum(same) / len(same)
        if not records:
            return None
        return sum(seconds for _, seconds in records) / len(records)

    def recommend(self, expected_queries: int, pattern_length: Optional[int] = None) -> str:
        """Say in one sentence which method is cheapest for this many queries.

        For every method with at least one measured query, the total cost of
        ``expected_queries`` queries is its preprocessing time plus
        ``expected_queries`` times its mean query time; the cheapest total
        wins. The sentence also gives the suffix array's break-even against
        KMP, ``ceil(preprocess_sa / (mean_kmp_query - mean_sa_query))``
        queries, when that denominator is positive, and says the build is
        never repaid when it is not. Every number comes from this engine's
        own measurements. With ``pattern_length``, each method's mean is taken
        over its queries of that length when it has some, and over all its
        queries otherwise.

        Args:
            expected_queries: How many queries will be run on this text (>= 1).
            pattern_length: Optional pattern length the queries will have.

        Returns:
            A plain sentence. Before at least one ``"kmp"`` and one ``"sa"``
            query have been measured it says so instead of guessing.

        Raises:
            TypeError: If ``expected_queries`` or ``pattern_length`` is not an
                ``int`` (``bool`` rejected).
            ValueError: If either is smaller than 1.

        Time Complexity:
            O(Q) over the Q recorded queries.

        Space Complexity:
            O(Q) for the filtered timings.

        Examples:
            >>> engine = StringSearchEngine("acgtacgtgacg" * 50)
            >>> engine.recommend(10).startswith("Not enough measurements yet")
            True
            >>> _ = engine.find_all("gacg")
            >>> engine.recommend(10).startswith("For 10 queries on this text, ")
            True
            >>> engine.recommend(10, pattern_length=4).startswith(
            ...     "For 10 queries on this text with 4-character patterns, ")
            True
        """
        for name, value in (("expected_queries", expected_queries), ("pattern_length", pattern_length)):
            if value is None and name == "pattern_length":
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        means = {method: self._mean_query_s(method, pattern_length) for method in METHODS}
        if means["kmp"] is None or means["sa"] is None:
            return (
                "Not enough measurements yet to recommend a method for this text: run at least "
                "one query with method='kmp' and one with method='sa' (find_all runs every "
                "method), then ask again."
            )

        q = expected_queries
        totals = {
            method: self.preprocessing_time_s.get(method, 0.0) + q * mean
            for method, mean in means.items()
            if mean is not None
        }
        best = min(totals, key=lambda method: (totals[method], METHODS.index(method)))
        kmp_mean, sa_mean = means["kmp"], means["sa"]
        build_sa = self.preprocessing_time_s["sa"]

        noun = "query" if q == 1 else "queries"
        length_part = f" with {pattern_length}-character patterns" if pattern_length is not None else ""
        head = (
            f"For {q} {noun} on this text{length_part}, {_NAMES[best]} is fastest overall "
            f"({_format_seconds(totals[best])} in total"
        )
        if best != "kmp":
            head += f" against {_format_seconds(totals['kmp'])} for KMP"
        head += ")"

        denominator = kmp_mean - sa_mean
        if denominator > 0:
            break_even = max(1, math.ceil(build_sa / denominator))
            owner = "its" if best == "sa" else "the suffix array's"
            tail = (
                f"{owner} {_format_seconds(build_sa)} build is repaid after {break_even} "
                f"{'query' if break_even == 1 else 'queries'}, since a suffix array query takes "
                f"{_format_seconds(sa_mean)} against {_format_seconds(kmp_mean)} for a KMP scan"
            )
        else:
            tail = (
                f"a suffix array query ({_format_seconds(sa_mean)}) is no faster than a KMP scan "
                f"({_format_seconds(kmp_mean)}), so its {_format_seconds(build_sa)} build is "
                f"never repaid"
            )
        return f"{head}: {tail}."
