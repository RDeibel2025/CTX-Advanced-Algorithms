"""Knuth-Morris-Pratt string matching, with a naive baseline for comparison.

The algorithm
-------------
The naive matcher tries every shift s of the pattern against the text and,
after a mismatch, slides the pattern one place and starts again from the
pattern's first character. On ``"a" * n`` with ``"a" * m`` that rereads the
same text characters over and over: (n - m + 1) * m comparisons.

Knuth-Morris-Pratt never moves backwards in the text. It keeps q, the number of
pattern characters matched so far, and on a mismatch at text index i it asks:
of the q characters just matched, what is the longest proper prefix of the
pattern that is also a suffix of them? That length, ``lps[q - 1]``, is how many
characters are still matched after sliding the pattern as far as is safe, so q
falls back to it and the same text character is compared again. Each fallback
lowers q and q rises by at most one per text character, so the scan makes at
most 2n comparisons and the whole search is O(n + m).

What it stores before it searches
---------------------------------
Before scanning, KMP stores the LPS array (the failure function) of the
*pattern*: one integer per pattern position, O(m) space, built in O(m) time by
running the same fallback logic on the pattern against itself. It stores
nothing about the text, so every query rescans the whole text.

Measurement
-----------
Both searches can return a :class:`MatchStats`. ``comparisons`` counts
text-versus-pattern character comparisons during the scan and
``lps_comparisons`` counts pattern-versus-pattern comparisons while building
the LPS array, so the report can quote comparisons per text character.
``kmp_search(..., trace=True)`` also returns every fallback as a
``(text_index, q_before, q_after)`` tuple, which is how the demo shows the
"mismatch, then fall back" behaviour.

Conventions (shared with every Week 11 matcher): positions are 0-based, every
occurrence is reported including overlapping ones, the result is a sorted
``list[int]``, an empty pattern raises ``ValueError``, and a pattern longer
than the text (or an empty text) gives ``[]``.

Examples:
    >>> compute_lps("ababaca")
    [0, 0, 1, 2, 3, 0, 1]
    >>> kmp_search("abracadabra", "abra")
    [0, 7]
    >>> kmp_search("aaaaa", "aa") == naive_search("aaaaa", "aa")
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

__all__ = ["MatchStats", "compute_lps", "kmp_search", "naive_search"]


@dataclass
class MatchStats:
    """Character-comparison counts from one search.

    Attributes:
        comparisons: Text-versus-pattern character comparisons made while
            scanning the text.
        lps_comparisons: Pattern-versus-pattern character comparisons made
            while building the LPS array (always 0 for the naive search).

    Examples:
        >>> MatchStats(comparisons=5)
        MatchStats(comparisons=5, lps_comparisons=0)
        >>> MatchStats(20, 8).comparisons
        20
    """

    comparisons: int
    lps_comparisons: int = 0


def _lps_with_count(pattern: str) -> Tuple[List[int], int]:
    """Build the LPS array and count the character comparisons it took.

    Args:
        pattern: The pattern string.

    Returns:
        ``(lps, comparisons)``.

    Time Complexity:
        O(m), at most 2m comparisons.

    Space Complexity:
        O(m) for the array.

    Examples:
        >>> _lps_with_count("ababaca")
        ([0, 0, 1, 2, 3, 0, 1], 8)
    """
    m = len(pattern)
    lps = [0] * m
    k = 0
    comparisons = 0
    for i in range(1, m):
        c = pattern[i]
        while True:
            comparisons += 1
            if pattern[k] == c:
                k += 1
                break
            if k == 0:
                break
            k = lps[k - 1]
        lps[i] = k
    return lps, comparisons


def compute_lps(pattern: str) -> List[int]:
    """Return the longest-proper-prefix-that-is-also-a-suffix length per index.

    ``lps[i]`` is the length of the longest proper prefix of ``pattern[: i + 1]``
    that is also a suffix of it. It is built left to right: k is the current
    border length, and when ``pattern[i]`` does not extend it, k falls back to
    ``lps[k - 1]``, the next shorter border, exactly as the search falls back
    on a text mismatch.

    Args:
        pattern: The pattern string. An empty pattern gives ``[]``.

    Returns:
        A list of ``len(pattern)`` integers with ``lps[0] == 0``.

    Raises:
        Nothing for string input.

    Time Complexity:
        O(m): k rises by at most one per character and every fallback lowers
        it, so there are at most 2m comparisons.

    Space Complexity:
        O(m) for the returned array.

    Examples:
        >>> compute_lps("aabaaab")
        [0, 1, 0, 1, 2, 2, 3]
        >>> compute_lps("aaaa")
        [0, 1, 2, 3]
        >>> compute_lps("")
        []
    """
    return _lps_with_count(pattern)[0]


def _check_pattern(pattern: str) -> None:
    """Raise ``ValueError`` for an empty pattern (the shared convention).

    Args:
        pattern: The pattern to check.

    Raises:
        ValueError: If ``pattern`` is empty.

    Examples:
        >>> _check_pattern("a")
        >>> _check_pattern("")
        Traceback (most recent call last):
        ...
        ValueError: pattern must be non-empty
    """
    if len(pattern) == 0:
        raise ValueError("pattern must be non-empty")


def _kmp_fast(text: str, pattern: str, lps: List[int]) -> List[int]:
    """Run the KMP scan with no counting or tracing.

    Args:
        text: The text to scan.
        pattern: A non-empty pattern no longer than ``text``.
        lps: ``compute_lps(pattern)``.

    Returns:
        The sorted start positions of every occurrence.

    Time Complexity:
        O(n).

    Space Complexity:
        O(k) for k matches.

    Examples:
        >>> _kmp_fast("abababab", "abab", [0, 0, 1, 2])
        [0, 2, 4]
    """
    m = len(pattern)
    last = m - 1
    after_match = lps[last]
    positions: List[int] = []
    q = 0
    for i, c in enumerate(text):
        while q and pattern[q] != c:
            q = lps[q - 1]
        if pattern[q] == c:
            if q == last:
                positions.append(i - last)
                q = after_match
            else:
                q += 1
    return positions


def kmp_search(text: str, pattern: str, trace: bool = False, stats: bool = False):
    """Find every occurrence of ``pattern`` in ``text`` with Knuth-Morris-Pratt.

    The text is read once, left to right. q counts the pattern characters
    matched so far. At each text character the pattern character ``pattern[q]``
    is compared with it: on a match q grows, and on a mismatch with q > 0 q
    falls back to ``lps[q - 1]`` and the same text character is compared again.
    When q reaches m an occurrence is recorded and q drops to ``lps[m - 1]`` so
    that overlapping occurrences are found too. That post-match shift is not a
    mismatch, so it is not recorded as a fallback event.

    Args:
        text: The text to search (any Python ``str``).
        pattern: The non-empty pattern to find.
        trace: If ``True``, also return the fallback events, one
            ``(text_index, q_before, q_after)`` tuple per fallback step taken
            after a mismatch, in order. Every event has ``q_after < q_before``.
        stats: If ``True``, also return a :class:`MatchStats` with the scan's
            and the LPS build's character comparisons.

    Returns:
        The sorted list of 0-based start positions when both flags are
        ``False``; ``(positions, events)`` with ``trace=True``;
        ``(positions, MatchStats)`` with ``stats=True``; and
        ``(positions, events, MatchStats)`` with both. A pattern longer than
        the text (or an empty text) gives ``[]``, no events and zero counts.

    Raises:
        ValueError: If ``pattern`` is empty.

    Time Complexity:
        O(n + m): O(m) to build the LPS array and at most 2n comparisons in
        the scan.

    Space Complexity:
        O(m + k) for the LPS array and k matches, plus O(n) for the events
        when ``trace=True`` (there are fewer fallbacks than text characters).

    Examples:
        >>> kmp_search("mississippi", "issi")
        [1, 4]
        >>> kmp_search("aaaaa", "aa")
        [0, 1, 2, 3]
        >>> kmp_search("ab", "abc")
        []
        >>> kmp_search("aab", "ab", trace=True)
        ([1], [(1, 1, 0)])
        >>> kmp_search("aab", "ab", stats=True)
        ([1], MatchStats(comparisons=4, lps_comparisons=1))
        >>> positions, events, st = kmp_search("bacbababaabcbab", "ababaca",
        ...                                    trace=True, stats=True)
        >>> positions, events
        ([], [(2, 1, 0), (9, 5, 3), (9, 3, 1), (9, 1, 0), (11, 2, 0)])
        >>> st
        MatchStats(comparisons=20, lps_comparisons=8)
        >>> kmp_search("abc", "")
        Traceback (most recent call last):
        ...
        ValueError: pattern must be non-empty
    """
    _check_pattern(pattern)
    n = len(text)
    m = len(pattern)
    if m > n:
        return _shape([], [], MatchStats(0, 0), trace, stats)

    lps, lps_comparisons = _lps_with_count(pattern)
    if not trace and not stats:
        return _kmp_fast(text, pattern, lps)

    last = m - 1
    positions: List[int] = []
    events: List[Tuple[int, int, int]] = []
    comparisons = 0
    q = 0
    for i, c in enumerate(text):
        while True:
            comparisons += 1
            if pattern[q] == c:
                q += 1
                break
            if q == 0:
                break
            q_after = lps[q - 1]
            if trace:
                events.append((i, q, q_after))
            q = q_after
        if q == m:
            positions.append(i - last)
            q = lps[last]
    return _shape(positions, events, MatchStats(comparisons, lps_comparisons), trace, stats)


def _shape(positions: List[int], events: list, match_stats: MatchStats, trace: bool, stats: bool):
    """Pack the result in the shape the flags ask for.

    Args:
        positions: The match positions.
        events: The fallback events.
        match_stats: The comparison counts.
        trace: Whether the events are wanted.
        stats: Whether the counts are wanted.

    Returns:
        ``positions``, ``(positions, events)``, ``(positions, match_stats)`` or
        ``(positions, events, match_stats)``.

    Examples:
        >>> _shape([0], [], MatchStats(1), True, False)
        ([0], [])
    """
    if trace and stats:
        return positions, events, match_stats
    if trace:
        return positions, events
    if stats:
        return positions, match_stats
    return positions


def naive_search(text: str, pattern: str, stats: bool = False):
    """Find every occurrence of ``pattern`` by trying every shift.

    For each shift s from 0 to n - m the pattern is compared with
    ``text[s: s + m]`` one character at a time, stopping at the first
    mismatch. This is the baseline the tests check every method against and
    the slow side of the worst-case benchmark; it stores nothing before it
    searches.

    Args:
        text: The text to search (any Python ``str``).
        pattern: The non-empty pattern to find.
        stats: If ``True``, also return a :class:`MatchStats` whose
            ``comparisons`` counts the character comparisons
            (``lps_comparisons`` is 0).

    Returns:
        The sorted list of 0-based start positions, or
        ``(positions, MatchStats)`` with ``stats=True``. A pattern longer than
        the text (or an empty text) gives ``[]``.

    Raises:
        ValueError: If ``pattern`` is empty.

    Time Complexity:
        O((n - m + 1) * m) in the worst case, reached on ``"a" * n`` with
        ``"a" * m``, which makes exactly (n - m + 1) * m comparisons.

    Space Complexity:
        O(k) for k matches.

    Examples:
        >>> naive_search("abracadabra", "abra")
        [0, 7]
        >>> naive_search("aab", "ab", stats=True)
        ([1], MatchStats(comparisons=4, lps_comparisons=0))
        >>> naive_search("a" * 10, "a" * 3, stats=True)[1].comparisons
        24
    """
    _check_pattern(pattern)
    n = len(text)
    m = len(pattern)
    positions: List[int] = []
    comparisons = 0
    for s in range(n - m + 1):
        j = 0
        while j < m:
            comparisons += 1
            if text[s + j] != pattern[j]:
                break
            j += 1
        if j == m:
            positions.append(s)
    if stats:
        return positions, MatchStats(comparisons, 0)
    return positions
