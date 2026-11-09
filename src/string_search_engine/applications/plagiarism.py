"""Plagiarism detection: word shingles, Rabin-Karp fingerprints and a suffix array.

Shingles and the score
----------------------
Each document is split into lowercase words, keeping every word's character
span in the original text. A word k-shingle is a run of k consecutive words,
written as one string with single spaces (``"the cat sat"`` for k = 3). Two
documents that share a passage share every shingle inside it, so the score is
the Jaccard similarity of the two shingle *sets*::

    score = |A & B| / |A | B|

1.0 means the same set of shingles, 0.0 means none in common. Word order only
matters inside a shingle, so a small k misses reordering: swapping "cat" and
"dog" in "the cat sat on the mat and the dog sat on the log" leaves every
2-shingle in place (score 1.0) but changes two 3-shingles (score 9/11).

Fingerprints, verified
----------------------
Comparing every shingle of one document with every shingle of the other
costs O(|A| |B|) string comparisons. Instead each shingle is fingerprinted
with the Rabin-Karp polynomial hash
(:class:`~src.string_search_engine.utils.rolling_hash.RollingHash`) over its
joined text, and only shingles with equal fingerprints are compared. Equal
fingerprints do not prove equal text when the modulus is small, so every
fingerprint match is verified by string comparison, exactly as Rabin-Karp
verifies every hash hit; a match that fails verification is counted in
``spurious_fingerprint_matches`` and never affects the score.

Matched regions and the longest common substring
------------------------------------------------
The character spans of the shared shingles are merged into
``matched_regions_a`` and ``matched_regions_b``, which :func:`highlight`
wraps in ``[[`` and ``]]``. Independently of words, the longest common
substring of the raw texts comes from one
:class:`~src.string_search_engine.core.suffix_array.SuffixArray` over
``doc_a + "\\x01" + doc_b``: it is the largest LCP between two neighbouring
suffixes that start in different documents. The separator occurs once, so no
common prefix can run across it.

What it stores before it searches
---------------------------------
Before scanning ``doc_b`` it stores a dict from fingerprint to the distinct
shingles of ``doc_a`` (the role of the text index), and each shingle of
``doc_b`` then plays the pattern. For the longest common substring it stores
the suffix array and LCP array of the joined text of both documents.

================== ======================== =================
Function           Time                     Space
================== ======================== =================
compare_documents  O(k (Na + Nb) + C log C) O(k (Na + Nb) + C)
highlight          O(len(doc) + R)          O(len(doc))
================== ======================== =================

Here Na and Nb are the word counts, C = len(doc_a) + len(doc_b) + 1 is the
length of the joined text (its suffix array costs O(C log C) per doubling
round), and R is the number of matched regions. Spurious fingerprint matches
add one comparison each; with the default 61-bit modulus they essentially
never happen.

Examples:
    >>> d1 = "the cat sat on the mat and the dog sat on the log"
    >>> d2 = "the dog sat on the mat and the cat sat on the log"
    >>> report = compare_documents(d1, d2, k=3)
    >>> report.shared_shingles, report.total_shingles, round(report.score, 4)
    (9, 11, 0.8182)
    >>> compare_documents(d1, d2, k=2).score
    1.0
    >>> compare_documents("xabxac", "abcabxabcd").longest_common_substring
    'abxa'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.string_search_engine.core.suffix_array import SuffixArray
from src.string_search_engine.utils.rolling_hash import (
    DEFAULT_BASE,
    DEFAULT_MODULUS,
    RollingHash,
)

__all__ = ["PlagiarismReport", "SEPARATOR", "compare_documents", "highlight"]

#: Placed between the two documents for the longest-common-substring suffix array.
SEPARATOR: str = "\x01"

#: A word: letters, digits or underscores, with inner apostrophes kept ("vince's").
_WORD = re.compile(r"\w+(?:'\w+)*")

#: A character span ``(start, end)``, end exclusive.
Span = Tuple[int, int]


@dataclass
class PlagiarismReport:
    """The result of :func:`compare_documents` for one pair of documents.

    The score and shingle counts are over the two shingle *sets* (a shingle
    repeated in one document counts once). The regions are merged character
    spans, end exclusive, in increasing order.

    Attributes:
        score: Jaccard similarity ``shared_shingles / total_shingles`` of the
            shingle sets, in [0, 1]; 0.0 when neither document has a shingle.
        shared_shingles: Number of distinct shingles in both documents.
        total_shingles: Number of distinct shingles in either document (the
            size of the union).
        spurious_fingerprint_matches: Pairs of distinct shingles, one from
            each document, with equal fingerprints but different text.
        matched_regions_a: Merged spans of ``doc_a`` covered by shared
            shingles.
        matched_regions_b: The same for ``doc_b``.
        longest_common_substring: The longest string occurring in both raw
            documents (case and punctuation included), ``""`` if none.
        k: The shingle length in words.

    Args:
        As the attributes, in the same order.

    Returns:
        A new report.

    Raises:
        Nothing.

    Time Complexity:
        O(1) to construct.

    Space Complexity:
        O(R) for R regions.

    Examples:
        >>> PlagiarismReport(0.5, 1, 2, 0, [(0, 5)], [(4, 9)], "hello", 1).score
        0.5
    """

    score: float
    shared_shingles: int
    total_shingles: int
    spurious_fingerprint_matches: int
    matched_regions_a: List[Span] = field(default_factory=list)
    matched_regions_b: List[Span] = field(default_factory=list)
    longest_common_substring: str = ""
    k: int = 3


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
        >>> _require_str(3, "doc_a")
        Traceback (most recent call last):
            ...
        TypeError: doc_a must be a str, got int
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _shingles(doc: str, k: int) -> List[Tuple[str, Span]]:
    """Return every word k-shingle of ``doc`` with its character span.

    Words are the matches of ``\\w+(?:'\\w+)*``, lowercased; the span of a
    shingle runs from the start of its first word to the end of its last, in
    the original text. A document with fewer than k words has no shingles.

    Args:
        doc: The document.
        k: Words per shingle, at least 1.

    Returns:
        ``[(shingle_text, (start, end)), ...]`` in document order, repeats
        included.

    Raises:
        Nothing.

    Time Complexity:
        O(len(doc) + k * N) for N words.

    Space Complexity:
        O(k * N).

    Examples:
        >>> _shingles("The cat, the hat.", 2)
        [('the cat', (0, 7)), ('cat the', (4, 12)), ('the hat', (9, 16))]
        >>> _shingles("one two", 3)
        []
    """
    matches = list(_WORD.finditer(doc))
    words = [match.group().lower() for match in matches]
    return [
        (" ".join(words[i:i + k]), (matches[i].start(), matches[i + k - 1].end()))
        for i in range(len(words) - k + 1)
    ]


def _merge(spans: List[Span]) -> List[Span]:
    """Merge overlapping or touching spans into sorted disjoint ones.

    Args:
        spans: Spans ``(start, end)`` in any order.

    Returns:
        Sorted, disjoint spans covering the same characters; spans that
        overlap or touch (``next_start <= end``) become one.

    Raises:
        Nothing.

    Time Complexity:
        O(s log s) for s spans.

    Space Complexity:
        O(s).

    Examples:
        >>> _merge([(8, 12), (0, 5), (3, 7), (12, 14)])
        [(0, 7), (8, 14)]
        >>> _merge([])
        []
    """
    merged: List[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _longest_common_substring(doc_a: str, doc_b: str) -> str:
    """Return the longest substring of both documents, from one suffix array.

    Builds a :class:`SuffixArray` over ``doc_a + SEPARATOR + doc_b``. A suffix
    starting before the separator comes from ``doc_a``, one starting after it
    from ``doc_b``. The longest common substring is the largest ``lcp`` entry
    between sorted neighbours from different documents; the separator occurs
    once, so such a common prefix never includes it. Ties go to the first
    pair in suffix order (the lexicographically smallest string).

    Args:
        doc_a: First document, without ``SEPARATOR``.
        doc_b: Second document, without ``SEPARATOR``.

    Returns:
        The substring, or ``""`` when no character is shared.

    Raises:
        ValueError: If either document contains ``SEPARATOR``.

    Time Complexity:
        O(C log C) per prefix-doubling round plus O(C) for the LCP array and
        the scan, for C = len(doc_a) + len(doc_b) + 1.

    Space Complexity:
        O(C).

    Examples:
        >>> _longest_common_substring("xabxac", "abcabxabcd")
        'abxa'
        >>> _longest_common_substring("abc", "xyz")
        ''
    """
    if SEPARATOR in doc_a or SEPARATOR in doc_b:
        raise ValueError("documents must not contain the separator '\\x01'")
    joined = doc_a + SEPARATOR + doc_b
    boundary = len(doc_a)
    index = SuffixArray(joined)
    sa, lcp = index.sa, index.lcp
    best, best_start = 0, 0
    for r in range(1, len(sa)):
        i, j = sa[r - 1], sa[r]
        different = (i < boundary < j) or (j < boundary < i)
        if different and lcp[r] > best:
            best, best_start = lcp[r], j
    return joined[best_start:best_start + best]


def compare_documents(
    doc_a: str,
    doc_b: str,
    k: int = 3,
    base: int = DEFAULT_BASE,
    modulus: int = DEFAULT_MODULUS,
) -> PlagiarismReport:
    """Compare two documents by word k-shingles and report what they share.

    Fingerprints every distinct shingle of ``doc_a`` with
    ``RollingHash(base, modulus).hash`` into a dict from fingerprint to
    shingles, then fingerprints each distinct shingle of ``doc_b`` and
    compares it, character by character, with every ``doc_a`` shingle in its
    bucket. Equal text is a shared shingle; equal fingerprint with different
    text is a spurious fingerprint match, counted and otherwise ignored, so
    the score is exact for every modulus. The longest common substring of the
    raw texts comes from a suffix array over ``doc_a + "\\x01" + doc_b``.

    Args:
        doc_a: First document, any ``str`` without ``"\\x01"``.
        doc_b: Second document, any ``str`` without ``"\\x01"``.
        k: Words per shingle, at least 1.
        base: Base of the fingerprint hash, at least 1.
        modulus: Modulus of the fingerprint hash, at least 2. A tiny modulus
            such as 2 forces spurious fingerprint matches.

    Returns:
        A :class:`PlagiarismReport`.

    Raises:
        TypeError: If a document is not a ``str``, or ``k``, ``base`` or
            ``modulus`` is not an ``int`` (``bool`` rejected).
        ValueError: If ``k < 1``, ``base < 1``, ``modulus < 2``, or a document
            contains ``"\\x01"``.

    Time Complexity:
        O(k (Na + Nb)) to build and fingerprint the shingles, O(k) per
        fingerprint match to verify it, O(Na log Na) to merge the regions, and
        the suffix array build over the C = len(doc_a) + len(doc_b) + 1
        characters of the joined text.

    Space Complexity:
        O(k (Na + Nb) + C).

    Examples:
        >>> d1 = "the cat sat on the mat and the dog sat on the log"
        >>> d2 = "the dog sat on the mat and the cat sat on the log"
        >>> report = compare_documents(d1, d2, k=3)
        >>> report.shared_shingles, report.total_shingles
        (9, 11)
        >>> report.matched_regions_a == [(0, len(d1))]
        True

        With an even base, a hash modulo 2 is the parity of the last code
        point, so many different shingles collide; every collision is caught
        by verification and the score does not change.

        >>> noisy = compare_documents(d1, d2, k=3, modulus=2)
        >>> noisy.score == report.score, noisy.spurious_fingerprint_matches > 0
        (True, True)
        >>> compare_documents("one two", "one two", k=0)
        Traceback (most recent call last):
            ...
        ValueError: k must be at least 1, got 0
    """
    _require_str(doc_a, "doc_a")
    _require_str(doc_b, "doc_b")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    hasher = RollingHash(base, modulus)

    shingles_a = _shingles(doc_a, k)
    shingles_b = _shingles(doc_b, k)
    distinct_a = dict.fromkeys(text for text, _ in shingles_a)
    distinct_b = dict.fromkeys(text for text, _ in shingles_b)

    buckets: Dict[int, List[str]] = {}
    for text in distinct_a:
        buckets.setdefault(hasher.hash(text), []).append(text)

    shared = set()
    spurious = 0
    for text in distinct_b:
        for candidate in buckets.get(hasher.hash(text), ()):
            if candidate == text:
                shared.add(text)
            else:
                spurious += 1

    union = len(distinct_a) + len(distinct_b) - len(shared)
    score = len(shared) / union if union else 0.0
    return PlagiarismReport(
        score=score,
        shared_shingles=len(shared),
        total_shingles=union,
        spurious_fingerprint_matches=spurious,
        matched_regions_a=_merge([span for text, span in shingles_a if text in shared]),
        matched_regions_b=_merge([span for text, span in shingles_b if text in shared]),
        longest_common_substring=_longest_common_substring(doc_a, doc_b),
        k=k,
    )


def highlight(report: PlagiarismReport, doc: str, which: str = "a") -> str:
    """Return ``doc`` with each matched region wrapped in ``[[`` and ``]]``.

    The report stores spans, not text, so the document is passed again;
    ``which`` says whether it is the report's ``doc_a`` or ``doc_b``.

    Args:
        report: A report from :func:`compare_documents`.
        doc: The same document that was passed as ``doc_a`` (``which="a"``)
            or ``doc_b`` (``which="b"``).
        which: ``"a"`` or ``"b"``.

    Returns:
        The document with markers inserted; unchanged when nothing matched.

    Raises:
        TypeError: If ``doc`` is not a ``str``.
        ValueError: If ``which`` is not ``"a"`` or ``"b"``, or a region runs
            past the end of ``doc`` (the wrong document was passed).

    Time Complexity:
        O(len(doc) + R) for R regions.

    Space Complexity:
        O(len(doc)).

    Examples:
        >>> a, b = "one two three four", "zero one two three five"
        >>> report = compare_documents(a, b, k=2)
        >>> highlight(report, a)
        '[[one two three]] four'
        >>> highlight(report, b, which="b")
        'zero [[one two three]] five'
        >>> highlight(report, a, which="c")
        Traceback (most recent call last):
            ...
        ValueError: which must be 'a' or 'b', got 'c'
    """
    _require_str(doc, "doc")
    if which not in ("a", "b"):
        raise ValueError(f"which must be 'a' or 'b', got {which!r}")
    regions = report.matched_regions_a if which == "a" else report.matched_regions_b
    if regions and regions[-1][1] > len(doc):
        raise ValueError(f"a matched region ends past the end of doc (document {which!r}?)")
    parts: List[str] = []
    previous = 0
    for start, end in regions:
        parts.extend((doc[previous:start], "[[", doc[start:end], "]]"))
        previous = end
    parts.append(doc[previous:])
    return "".join(parts)
