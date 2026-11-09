"""Suffix array and LCP array: sort every suffix of the text once, then binary-search.

What it stores
--------------
A suffix array preprocesses the *text*, not the pattern. Before any query it
stores ``sa``, the starting positions of all n suffixes of the text in sorted
order, and ``lcp``, the length of the longest common prefix of each pair of
neighbours in that order; a query then only binary-searches ``sa``.

Building the suffix array: prefix doubling
------------------------------------------
Sorting the suffixes directly with ``sorted(range(n), key=lambda i: text[i:])``
copies O(n) characters per key and costs O(n^2 log n), which never reaches a
megabyte. Prefix doubling sorts them by ever longer prefixes instead. Each
character is first mapped to its rank in ``sorted(set(text))``, so any Python
``str`` works, Unicode included, and ``rank[i]`` orders the suffixes by their
first character. If ``rank[i]`` orders them by their first k characters, then
the pair ``(rank[i], rank[i + k])`` orders them by their first 2k characters,
with ``-1`` standing in past the end of the text because a suffix that ends
sorts before any suffix that continues. One ``numpy.lexsort`` on those pairs
gives the new order, and neighbours with different pairs get new consecutive
ranks. k doubles until every rank is distinct, which happens once k exceeds
the longest repeated substring, so random text needs only a handful of rounds.
No sentinel character is appended.

The LCP array: Kasai's algorithm
--------------------------------
``lcp[0] = 0`` and ``lcp[r]`` is the longest common prefix of the suffixes
``sa[r - 1]`` and ``sa[r]``. Kasai walks the suffixes in text order, not
sorted order: if suffix i shares h characters with the suffix just before it in
sorted order, suffix i + 1 shares at least h - 1 with its own predecessor. So
the running match length h drops by at most one per step, rises at most n times
in all, and the whole array costs O(n).

Searching
---------
Every occurrence of a pattern P of length m is a suffix that starts with P, and
those suffixes sit in one contiguous block of ``sa``. Two binary searches,
comparing ``text[sa[mid]: sa[mid] + m]`` with P, find the block's two ends in
O(m log n), and the positions in the block are the matches. From the same two
arrays, ``n(n+1)/2 - sum(lcp)`` counts the distinct substrings (each suffix
adds its length minus what it shares with its predecessor) and the largest LCP
entry gives the longest repeated substring.

Examples:
    >>> index = SuffixArray("banana")
    >>> index.sa
    [5, 3, 1, 0, 4, 2]
    >>> index.lcp
    [0, 1, 3, 0, 0, 2]
    >>> index.find("ana")
    [1, 3]
    >>> index.count_distinct_substrings(), index.longest_repeated_substring()
    (15, 'ana')

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["SuffixArray", "build_lcp", "build_suffix_array"]


def _check_text(text: Any) -> None:
    """Raise ``TypeError`` unless ``text`` is a ``str``."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")


def _initial_ranks(text: str) -> np.ndarray:
    """Map each character of ``text`` to its rank in ``sorted(set(text))``.

    The code points come from one UTF-32 encode, read as a numpy array, and
    ``numpy.searchsorted`` against the sorted alphabet turns them into ranks
    without a Python loop over the text. ``surrogatepass`` lets a lone
    surrogate through, since a Python ``str`` may hold one. Comparing single
    characters compares code points, so the code point order is exactly the
    order of ``sorted(set(text))``.
    """
    alphabet = sorted(set(text))
    alphabet_codes = np.array([ord(c) for c in alphabet], dtype=np.uint32)
    codes = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype="<u4")
    dtype = np.int32 if len(text) < 2**31 - 1 else np.int64
    return np.searchsorted(alphabet_codes, codes).astype(dtype)


def build_suffix_array(text: str) -> List[int]:
    """Return the starting positions of the suffixes of ``text`` in sorted order.

    Prefix doubling with ``numpy.lexsort``. The initial ranks are the
    characters' ranks in ``sorted(set(text))``. Each round sorts the suffixes
    by the pair ``(rank[i], rank[i + k])``, with ``-1`` when ``i + k`` is past
    the end, then gives neighbours with different pairs consecutive new ranks.
    After the round with step k the ranks order the suffixes by their first 2k
    characters; k doubles until all n ranks are distinct. Order is plain
    Python string order (code point by code point, a proper prefix first), and
    no sentinel is added.

    Args:
        text: Any Python ``str``, Unicode included.

    Returns:
        A ``list[int]`` permutation of ``range(len(text))``: ``sa[r]`` is the
        start of the r-th smallest suffix. Empty for an empty text.

    Raises:
        TypeError: If ``text`` is not a ``str``.

    Time Complexity:
        O(n log n) per round for the comparison sort inside ``lexsort``, over
        O(log L) rounds where L is the length of the longest repeated
        substring: O(n log n log log n) on random text, O(n log^2 n) in the
        worst case (``"a" * n``).

    Space Complexity:
        O(n): a few numpy integer arrays of length n plus the returned list.

    Examples:
        >>> build_suffix_array("banana")
        [5, 3, 1, 0, 4, 2]
        >>> build_suffix_array("aaaa")
        [3, 2, 1, 0]

        Characters sort by code point, so ``"é"`` comes after ``"f"``.

        >>> build_suffix_array("café")
        [1, 0, 2, 3]
        >>> build_suffix_array("")
        []
        >>> build_suffix_array(b"abc")
        Traceback (most recent call last):
            ...
        TypeError: text must be a str, got bytes
    """
    _check_text(text)
    n = len(text)
    if n == 0:
        return []

    rank = _initial_ranks(text)
    sa = np.argsort(rank, kind="stable")
    if int(rank.max()) == n - 1:
        # Every character is different, so one character already decides.
        return sa.tolist()

    second = np.empty(n, dtype=rank.dtype)
    k = 1
    while True:
        # Second key: the rank k positions later, -1 past the end of the text.
        second[: n - k] = rank[k:]
        second[n - k :] = -1
        sa = np.lexsort((second, rank))  # last key is the primary key
        first_sorted = rank[sa]
        second_sorted = second[sa]
        changed = (first_sorted[1:] != first_sorted[:-1]) | (
            second_sorted[1:] != second_sorted[:-1]
        )
        new_sorted = np.empty(n, dtype=rank.dtype)
        new_sorted[0] = 0
        np.cumsum(changed, out=new_sorted[1:])
        rank = np.empty(n, dtype=rank.dtype)
        rank[sa] = new_sorted
        if int(new_sorted[-1]) == n - 1:
            break
        k *= 2
    return sa.tolist()


def build_lcp(text: str, sa: Sequence[int]) -> List[int]:
    """Return the LCP array of ``text`` for its suffix array ``sa`` (Kasai).

    ``lcp[0] = 0`` and ``lcp[r]`` is the length of the longest common prefix
    of the suffixes starting at ``sa[r - 1]`` and ``sa[r]``. Kasai's algorithm
    visits the suffixes in text order and carries the match length h from
    suffix i to suffix i + 1, losing at most one character each step, so the
    total number of character comparisons is O(n).

    Args:
        text: The text ``sa`` was built for.
        sa: Its suffix array, a permutation of ``range(len(text))`` (a list or
            a numpy integer array).

    Returns:
        A ``list[int]`` of length ``len(text)``.

    Raises:
        TypeError: If ``text`` is not a ``str``.
        ValueError: If ``sa`` has the wrong length or is not a permutation of
            ``range(len(text))``.

    Time Complexity:
        O(n): h rises at most 2n times in total and falls by one per step.

    Space Complexity:
        O(n): the inverse permutation and the output.

    Examples:
        >>> build_lcp("banana", [5, 3, 1, 0, 4, 2])
        [0, 1, 3, 0, 0, 2]
        >>> build_lcp("aaa", [2, 1, 0])
        [0, 1, 2]
        >>> build_lcp("", [])
        []
        >>> build_lcp("ab", [0])
        Traceback (most recent call last):
            ...
        ValueError: sa has 1 entries but text has 2 characters
        >>> build_lcp("ab", [0, 0])
        Traceback (most recent call last):
            ...
        ValueError: sa must be a permutation of range(2)
    """
    _check_text(text)
    n = len(text)
    if len(sa) != n:
        raise ValueError(f"sa has {len(sa)} entries but text has {n} characters")
    if n == 0:
        return []

    sa_arr = np.asarray(sa, dtype=np.int64)
    if sa_arr.ndim != 1 or int(sa_arr.min()) < 0 or int(sa_arr.max()) >= n:
        raise ValueError(f"sa must be a permutation of range({n})")
    inverse = np.full(n, -1, dtype=np.int64)
    inverse[sa_arr] = np.arange(n, dtype=np.int64)
    if int(inverse.min()) < 0:
        raise ValueError(f"sa must be a permutation of range({n})")

    sa_list = sa_arr.tolist()
    rank = inverse.tolist()
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue
        j = sa_list[r - 1]
        limit = n - (i if i > j else j)
        while h < limit and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return lcp


class SuffixArray:
    """A suffix array index over one text, with its LCP array and build times.

    The constructor sorts the suffixes once (:func:`build_suffix_array`) and
    computes the LCP array (:func:`build_lcp`), timing each with
    ``time.perf_counter``. Every query afterwards reads only those two arrays
    and the text: :meth:`find` and :meth:`count` binary-search ``sa``, while
    :meth:`count_distinct_substrings` and :meth:`longest_repeated_substring`
    read ``lcp``.

    Attributes:
        text: The indexed text.
        sa: The suffix array, a ``list[int]``.
        lcp: The LCP array, a ``list[int]`` with ``lcp[0] = 0``.
        build_time_s: ``{"sa": seconds, "lcp": seconds}``.

    Args:
        text: Any Python ``str``, Unicode included. It may be empty.

    Raises:
        TypeError: If ``text`` is not a ``str``.

    Time Complexity:
        Construction as :func:`build_suffix_array` plus O(n) for the LCP array.
        :meth:`find` is O(m log n + occ log occ), :meth:`count` O(m log n).

    Space Complexity:
        O(n): the text, ``sa`` and ``lcp``.

    Examples:
        >>> index = SuffixArray("mississippi")
        >>> index.find("issi"), index.count("ss")
        ([1, 4], 2)
        >>> index.longest_repeated_substring(), index.count_distinct_substrings()
        ('issi', 53)
        >>> sorted(index.build_time_s)
        ['lcp', 'sa']
        >>> SuffixArray("").find("a")
        []
    """

    def __init__(self, text: str) -> None:
        """Build the suffix array and LCP array of ``text`` and time each step.

        Calls :func:`build_suffix_array` and then :func:`build_lcp`, recording
        each one's wall-clock time with ``time.perf_counter`` in
        ``build_time_s``.

        Args:
            text: Any Python ``str``, Unicode included. It may be empty.

        Returns:
            None.

        Raises:
            TypeError: If ``text`` is not a ``str``.

        Time Complexity:
            As :func:`build_suffix_array` plus O(n) for the LCP array.

        Space Complexity:
            O(n): ``sa`` and ``lcp``.

        Examples:
            >>> index = SuffixArray("banana")
            >>> index.sa, index.lcp
            ([5, 3, 1, 0, 4, 2], [0, 1, 3, 0, 0, 2])
        """
        _check_text(text)
        self.text: str = text
        start = time.perf_counter()
        self.sa: List[int] = build_suffix_array(text)
        middle = time.perf_counter()
        self.lcp: List[int] = build_lcp(text, self.sa)
        end = time.perf_counter()
        self.build_time_s: Dict[str, float] = {"sa": middle - start, "lcp": end - middle}

    def _range(self, pattern: str) -> Tuple[int, int]:
        """Return the half-open block ``[lo, hi)`` of ``sa`` whose suffixes start with ``pattern``.

        The m-character prefixes of the sorted suffixes are themselves in
        sorted order, so a lower-bound and an upper-bound binary search on
        ``text[sa[mid]: sa[mid] + m]`` find the block of prefixes equal to the
        pattern. A suffix shorter than m can never equal it.
        """
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a str, got {type(pattern).__name__}")
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        text, sa, m, n = self.text, self.sa, len(pattern), len(self.sa)
        if m > n:
            return 0, 0
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            start = sa[mid]
            if text[start : start + m] < pattern:
                lo = mid + 1
            else:
                hi = mid
        left = lo
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            start = sa[mid]
            if text[start : start + m] <= pattern:
                lo = mid + 1
            else:
                hi = mid
        return left, lo

    def find(self, pattern: str) -> List[int]:
        """Return every position where ``pattern`` occurs in the text.

        Two binary searches find the block of ``sa`` whose suffixes start with
        the pattern; those suffixes' starts are exactly the occurrences,
        overlapping ones included, and are returned in increasing order.

        Args:
            pattern: A non-empty ``str``.

        Returns:
            A sorted ``list[int]`` of 0-based positions. ``[]`` when the
            pattern is longer than the text or the text is empty.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m log n) for the two searches plus O(occ log occ) to sort the
            occ matches.

        Space Complexity:
            O(occ) for the result.

        Examples:
            >>> index = SuffixArray("abracadabra")
            >>> index.find("abra")
            [0, 7]
            >>> SuffixArray("aaaaa").find("aa")
            [0, 1, 2, 3]
            >>> index.find("abracadabras")
            []
            >>> index.find("")
            Traceback (most recent call last):
                ...
            ValueError: pattern must be a non-empty string
        """
        lo, hi = self._range(pattern)
        return sorted(self.sa[lo:hi])

    def count(self, pattern: str) -> int:
        """Return the number of occurrences of ``pattern`` in the text.

        The size of the ``sa`` block found by :meth:`find`, without listing or
        sorting the positions.

        Args:
            pattern: A non-empty ``str``.

        Returns:
            The number of (possibly overlapping) occurrences.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m log n).

        Space Complexity:
            O(m) for the compared slices.

        Examples:
            >>> index = SuffixArray("banana")
            >>> index.count("a"), index.count("ana"), index.count("nab")
            (3, 2, 0)
        """
        lo, hi = self._range(pattern)
        return hi - lo

    def longest_repeated_substring(self) -> str:
        """Return the longest substring that occurs at least twice.

        Two occurrences of a repeated substring are two suffixes sharing that
        prefix, and the longest shared prefix of any two suffixes is shared by
        some pair of sorted neighbours, so the answer is the prefix of length
        ``max(lcp)`` at the first position where that maximum is reached
        (the lexicographically smallest one on a tie). Occurrences may overlap.

        Args:
            None.

        Returns:
            The substring, or ``""`` when no character repeats.

        Raises:
            Nothing.

        Time Complexity:
            O(n) to scan ``lcp``.

        Space Complexity:
            O(L) for the returned string of length L.

        Examples:
            >>> SuffixArray("abracadabra").longest_repeated_substring()
            'abra'
            >>> SuffixArray("aaaa").longest_repeated_substring()
            'aaa'
            >>> SuffixArray("abc").longest_repeated_substring()
            ''
        """
        if not self.lcp:
            return ""
        best = max(self.lcp)
        if best == 0:
            return ""
        start = self.sa[self.lcp.index(best)]
        return self.text[start : start + best]

    def count_distinct_substrings(self) -> int:
        """Return the number of distinct non-empty substrings of the text.

        Every substring is a prefix of some suffix. The suffix ``sa[r]``
        contributes its ``n - sa[r]`` prefixes, of which the first ``lcp[r]``
        were already counted with its sorted predecessor, so the total is
        ``n(n+1)/2 - sum(lcp)``.

        Args:
            None.

        Returns:
            A Python ``int``.

        Raises:
            Nothing.

        Time Complexity:
            O(n) to sum ``lcp``.

        Space Complexity:
            O(1).

        Examples:
            >>> SuffixArray("banana").count_distinct_substrings()
            15
            >>> SuffixArray("aaaa").count_distinct_substrings()
            4
            >>> SuffixArray("").count_distinct_substrings()
            0
        """
        n = len(self.text)
        return n * (n + 1) // 2 - sum(self.lcp)

    def stats(self) -> Dict[str, Any]:
        """Return summary numbers about the text read from ``sa`` and ``lcp``.

        ``mean_lcp`` is the mean of all n LCP entries (``lcp[0] = 0``
        included) and ``distinct_substrings`` is a Python ``int``, exact at
        any size.

        Args:
            None.

        Returns:
            ``{"n": text length, "mean_lcp": float (0.0 when empty),
            "max_lcp": int, "distinct_substrings": int, "lrs_length": int}``,
            where ``lrs_length`` is the length of the longest repeated
            substring and equals ``max_lcp``.

        Raises:
            Nothing.

        Time Complexity:
            O(n).

        Space Complexity:
            O(1) beyond the returned dict.

        Examples:
            >>> SuffixArray("banana").stats()
            {'n': 6, 'mean_lcp': 1.0, 'max_lcp': 3, 'distinct_substrings': 15, 'lrs_length': 3}
            >>> SuffixArray("").stats()
            {'n': 0, 'mean_lcp': 0.0, 'max_lcp': 0, 'distinct_substrings': 0, 'lrs_length': 0}
        """
        n = len(self.text)
        total = sum(self.lcp)
        max_lcp = max(self.lcp) if self.lcp else 0
        return {
            "n": n,
            "mean_lcp": total / n if n else 0.0,
            "max_lcp": max_lcp,
            "distinct_substrings": n * (n + 1) // 2 - total,
            "lrs_length": max_lcp,
        }
