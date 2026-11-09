"""Rabin-Karp string matching: compare fingerprints first, characters only on a hit.

The algorithm
-------------
Rabin-Karp hashes the pattern once, then slides a window of the same length
along the text, updating the window's hash in O(1) per step with the rolling
hash of :mod:`src.string_search_engine.utils.rolling_hash`::

    h(s) = sum(ord(s[i]) * base^(m-1-i)) mod q

Unequal hashes prove the window is not the pattern, so almost every window is
rejected by one integer comparison. Equal hashes (a *hash hit*) only suggest a
match: two different strings can share a hash modulo q. Every hit is therefore
verified character by character, stopping at the first mismatch, and a hit
that fails verification is a *spurious hit*. Because of that check the
positions returned are always exact, whatever the modulus; a bad modulus can
only cost time, never correctness.

What it stores before it searches
---------------------------------
It preprocesses the pattern, not the text: before the scan it stores only the
pattern's hash and ``base^(m-1) mod q``, O(1) numbers, and keeps nothing about
the text between queries.

Running time
------------
Average O(n + m): O(m) to hash the pattern and the first window, O(1) per roll
over the n - m + 1 windows, plus O(m) for each hit. With q = 2^61 - 1 spurious
hits are vanishingly rare, so the verification cost is O(m) per real match.
Worst case O(nm): when most windows hit, for example ``'a' * n`` with
``'a' * m``, every one of the n - m + 1 windows is a true match and each is
verified in full, (n - m + 1) * m character comparisons. A tiny modulus such
as q = 7 causes the same blow-up through spurious hits.

Many patterns at once
---------------------
:func:`rabin_karp_multi` groups the patterns by length. For each distinct
length it makes one rolling pass over the text and looks each window's hash
up in a ``dict`` from hash to the patterns sharing it, so k patterns of one
length cost one pass, not k.

Conventions
-----------
Positions are 0-based, all occurrences are reported (overlapping ones
included) as a sorted ``list[int]``. An empty pattern raises ``ValueError``;
a pattern longer than the text, or an empty text, gives ``[]``. Any Python
``str`` works, Unicode included, since each character contributes its code
point.

Examples:
    >>> rabin_karp_search("abracadabra", "abra")
    [0, 7]
    >>> positions, st = rabin_karp_search("abracadabra", "abra", modulus=7, stats=True)
    >>> positions, st.hash_hits, st.spurious_hits
    ([0, 7], 5, 3)
    >>> rabin_karp_multi("abracadabra", ["abra", "cad"])
    {'abra': [0, 7], 'cad': [4]}

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple, Union

from src.string_search_engine.utils.rolling_hash import (
    DEFAULT_BASE,
    DEFAULT_MODULUS,
    RollingHash,
)

__all__ = ["RKStats", "rabin_karp_multi", "rabin_karp_search"]


@dataclass
class RKStats:
    """Counters from one Rabin-Karp run, returned when ``stats=True``.

    Attributes:
        hash_hits: Windows whose hash equalled a pattern's hash. In
            :func:`rabin_karp_multi` a window counts once per pattern sharing
            its hash.
        spurious_hits: Hash hits whose characters differed from the pattern
            (a collision modulo q).
        verifications: Character-by-character checks run. Every hash hit is
            verified, so this equals ``hash_hits``.
        comparisons: Characters compared during verification, each check
            stopping at its first mismatch.

    Examples:
        >>> RKStats(hash_hits=2, spurious_hits=0, verifications=2, comparisons=8)
        RKStats(hash_hits=2, spurious_hits=0, verifications=2, comparisons=8)
    """

    hash_hits: int
    spurious_hits: int
    verifications: int
    comparisons: int


def _require_str(value: Any, name: str) -> str:
    """Return ``value`` if it is a ``str``, else raise ``TypeError``.

    Args:
        value: The value to check.
        name: The parameter name used in the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not a ``str``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_str("abc", "text")
        'abc'
        >>> _require_str(b"abc", "text")
        Traceback (most recent call last):
            ...
        TypeError: text must be a str, got bytes
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _verify(text: str, start: int, pattern: str) -> Tuple[bool, int]:
    """Compare ``pattern`` with ``text[start:]`` one character at a time.

    The check stops at the first mismatch, so it reports how many characters
    it actually compared, which is what ``RKStats.comparisons`` adds up.

    Args:
        text: The text.
        start: Where the window starts; ``start + len(pattern) <= len(text)``.
        pattern: The pattern.

    Returns:
        ``(matched, comparisons)``.

    Raises:
        IndexError: If the window runs past the end of ``text``.

    Time Complexity:
        O(m) worst case, O(1) when the first character differs.

    Space Complexity:
        O(1).

    Examples:
        >>> _verify("abracadabra", 7, "abra")
        (True, 4)
        >>> _verify("abracadabra", 3, "abra")
        (False, 2)
    """
    compared = 0
    for j, c in enumerate(pattern):
        compared += 1
        if text[start + j] != c:
            return False, compared
    return True, compared


def rabin_karp_search(
    text: str,
    pattern: str,
    base: int = DEFAULT_BASE,
    modulus: int = DEFAULT_MODULUS,
    stats: bool = False,
) -> Union[List[int], Tuple[List[int], RKStats]]:
    """Find every occurrence of ``pattern`` in ``text`` by rolling hash.

    The pattern and the first window are hashed with :class:`RollingHash`,
    then the window slides one character at a time with the O(1) roll
    ``((h - ord(out) * base^(m-1)) * base + ord(in)) mod q``, done inline for
    speed. Each window whose hash equals the pattern's is verified character
    by character, stopping at the first mismatch, so the result is exact for
    every modulus.

    With ``stats=False`` the verification is ``str.startswith``, which makes
    the same left-to-right comparison in C; with ``stats=True`` it runs a
    Python loop so the compared characters can be counted.

    Args:
        text: The text to search. Any ``str``, Unicode included.
        pattern: The pattern. Must not be empty.
        base: The hash radix, at least 1. ``base = 1`` hashes a window to the
            sum of its code points, so every anagram collides.
        modulus: The hash modulus q, at least 2. Small values cause spurious
            hits; the default is the prime 2^61 - 1.
        stats: If ``True``, also return an :class:`RKStats`.

    Returns:
        The sorted 0-based start positions of all (overlapping) occurrences,
        or ``(positions, RKStats)`` when ``stats=True``.

    Raises:
        TypeError: If ``text`` or ``pattern`` is not a ``str``, or ``base`` or
            ``modulus`` is not an ``int``.
        ValueError: If ``pattern`` is empty, ``base < 1`` or ``modulus < 2``.

    Time Complexity:
        Average O(n + m): one O(1) roll per window plus O(m) per hash hit,
        and with a large prime modulus hits are almost all real matches.
        Worst case O(nm) when most windows hit, e.g. ``'a' * n`` with
        ``'a' * m`` ((n - m + 1) * m comparisons), or a tiny modulus.

    Space Complexity:
        O(1) beyond the output: two hashes and ``base^(m-1) mod q``.

    Examples:
        >>> rabin_karp_search("abracadabra", "abra")
        [0, 7]
        >>> rabin_karp_search("aaaaa", "aa")
        [0, 1, 2, 3]
        >>> rabin_karp_search("mississippi", "issi", modulus=7, stats=True)
        ([1, 4], RKStats(hash_hits=3, spurious_hits=1, verifications=3, comparisons=9))

        Every window of ``'a' * 10`` hits and is verified in full.

        >>> positions, st = rabin_karp_search("a" * 10, "a" * 3, stats=True)
        >>> len(positions), st.hash_hits, st.comparisons
        (8, 8, 24)

        Unicode, a pattern longer than the text, and an empty text.

        >>> rabin_karp_search("naïve café naïve", "naïve")
        [0, 11]
        >>> rabin_karp_search("ab", "abc"), rabin_karp_search("", "a")
        ([], [])
        >>> rabin_karp_search("abc", "")
        Traceback (most recent call last):
            ...
        ValueError: pattern must not be empty
    """
    _require_str(text, "text")
    _require_str(pattern, "pattern")
    if not pattern:
        raise ValueError("pattern must not be empty")
    rh = RollingHash(base, modulus)
    n, m = len(text), len(pattern)
    positions: List[int] = []
    if m > n:
        return (positions, RKStats(0, 0, 0, 0)) if stats else positions

    q = rh.modulus
    b = rh.base
    pw = rh.power(m)
    target = rh.hash(pattern)
    h = rh.hash(text[:m])

    if not stats:
        startswith = text.startswith
        if h == target and startswith(pattern, 0):
            positions.append(0)
        i = 1
        for out_c, in_c in zip(map(ord, text), map(ord, islice(text, m, None))):
            h = ((h - out_c * pw) * b + in_c) % q
            if h == target and startswith(pattern, i):
                positions.append(i)
            i += 1
        return positions

    hits = spurious = comparisons = 0
    for i in range(n - m + 1):
        if i:
            h = ((h - ord(text[i - 1]) * pw) * b + ord(text[i + m - 1])) % q
        if h == target:
            hits += 1
            matched, compared = _verify(text, i, pattern)
            comparisons += compared
            if matched:
                positions.append(i)
            else:
                spurious += 1
    return positions, RKStats(hits, spurious, hits, comparisons)


def rabin_karp_multi(
    text: str,
    patterns: Iterable[str],
    base: int = DEFAULT_BASE,
    modulus: int = DEFAULT_MODULUS,
    stats: bool = False,
) -> Union[Dict[str, List[int]], Tuple[Dict[str, List[int]], RKStats]]:
    """Find every occurrence of several patterns with one pass per pattern length.

    The distinct patterns are grouped by length. For each length L that fits
    in the text, the patterns' hashes go into a ``dict`` from hash to the
    list of patterns sharing it, then one rolling pass over the text looks
    each window's hash up. Every pattern under a matching hash is verified
    character by character (stopping at the first mismatch), so two patterns
    that collide with each other are still told apart.

    Args:
        text: The text to search.
        patterns: The patterns. Duplicates are searched once; none may be
            empty.
        base: The hash radix, at least 1.
        modulus: The hash modulus q, at least 2.
        stats: If ``True``, also return an :class:`RKStats` totalled over all
            passes, where a window counts one hash hit per pattern sharing
            its hash.

    Returns:
        A ``dict`` from each distinct pattern, in first-seen order, to its
        sorted list of 0-based positions (``[]`` when absent or longer than
        the text), or ``(dict, RKStats)`` when ``stats=True``.

    Raises:
        TypeError: If ``text`` or any pattern is not a ``str``, or ``base`` or
            ``modulus`` is not an ``int``.
        ValueError: If any pattern is empty, ``base < 1`` or ``modulus < 2``.

    Time Complexity:
        Average O(d n + M), where d is the number of distinct pattern lengths
        and M the total pattern length: one O(n) pass per length plus O(L)
        per hash hit. Worst case O(n M) when every window hits every pattern
        of its length.

    Space Complexity:
        O(k) for the hash tables of the k patterns, plus the output.

    Examples:
        >>> rabin_karp_multi("abracadabra", ["abra", "cad", "bra", "zzz"])
        {'abra': [0, 7], 'cad': [4], 'bra': [1, 8], 'zzz': []}
        >>> found, st = rabin_karp_multi("abracadabra", ["abra", "bra"], stats=True)
        >>> found, st.hash_hits, st.spurious_hits, st.comparisons
        ({'abra': [0, 7], 'bra': [1, 8]}, 4, 0, 14)
        >>> rabin_karp_multi("abc", [])
        {}
        >>> rabin_karp_multi("abc", ["a", ""])
        Traceback (most recent call last):
            ...
        ValueError: patterns must not be empty strings
    """
    _require_str(text, "text")
    rh = RollingHash(base, modulus)
    results: Dict[str, List[int]] = {}
    by_length: Dict[int, List[str]] = {}
    for p in patterns:
        _require_str(p, "pattern")
        if not p:
            raise ValueError("patterns must not be empty strings")
        if p not in results:
            results[p] = []
            by_length.setdefault(len(p), []).append(p)

    n = len(text)
    q, b = rh.modulus, rh.base
    hits = spurious = comparisons = 0
    for m, group in by_length.items():
        if m > n:
            continue
        table: Dict[int, List[str]] = {}
        for p in group:
            table.setdefault(rh.hash(p), []).append(p)
        pw = rh.power(m)
        h = rh.hash(text[:m])
        for i in range(n - m + 1):
            if i:
                h = ((h - ord(text[i - 1]) * pw) * b + ord(text[i + m - 1])) % q
            candidates = table.get(h)
            if candidates is None:
                continue
            for p in candidates:
                hits += 1
                if stats:
                    matched, compared = _verify(text, i, p)
                    comparisons += compared
                else:
                    matched = text.startswith(p, i)
                if matched:
                    results[p].append(i)
                else:
                    spurious += 1
    if stats:
        return results, RKStats(hits, spurious, hits, comparisons)
    return results
