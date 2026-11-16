"""DNA index: a wavelet tree for window queries and an FM-index for pattern counts.

The idea
--------
:class:`DNAIndex` answers two kinds of question about a DNA sequence over the
alphabet ``A < C < G < T``:

* **Window questions** ("how many A's in positions [l, r]?", "what is the
  k-th smallest base in the window?", "how often does each base occur?")
  go to a balanced :class:`~src.advds.succinct.wavelet_tree.WaveletTree`
  over the sequence with alphabet ``"ACGT"``. It halves the alphabet at
  every level, so DNA needs exactly 2 levels and 2 bits per base, and each
  query is a few ``RankSelect`` calls per level.
* **Pattern counts** ("how many times does GATTACA occur?") go to an
  FM-index over ``seq + "$"``: the Burrows-Wheeler transform (BWT) of that
  text, a :class:`~src.advds.succinct.wavelet_tree.WaveletTree` over the BWT
  with alphabet ``"$ACGT"`` for ``rank``, and the C array, where ``C[c]`` is
  the number of symbols of the text smaller than c. The terminator ``$``
  occurs once and sorts before every base.

Suffix array and BWT
--------------------
:func:`build_suffix_array` sorts the suffixes by prefix doubling: round k
gives every suffix the rank of its first 2^k symbols by sorting the pairs
(rank of the first half, rank of the second half) with ``np.lexsort``, the
one place numpy's sort is allowed (spec 1.1); the rounds stop when every
rank is distinct, after at most ``ceil(log2 n) + 1`` rounds. Ranks inside a
round are assigned with ``np.flatnonzero`` and ``np.repeat`` over the group
boundaries. :func:`bwt_from_suffix_array` reads ``BWT[i] = text[SA[i] - 1]``
(the symbol before each sorted suffix, wrapping to ``$`` for ``SA[i] = 0``).

Backward search
---------------
:meth:`DNAIndex.pattern_count` keeps the half-open range ``[lo, hi)`` of
suffix-array rows whose suffixes start with the part of the pattern read so
far, reading the pattern from its last symbol to its first. Starting from
all ``n + 1`` rows, each symbol c maps the range to
``[C[c] + rank_c(lo), C[c] + rank_c(hi))``, two ``WaveletTree.rank`` calls
on the BWT (LF-mapping). The final range size is the number of occurrences,
overlapping ones included, in O(m log sigma) for a pattern of length m,
whatever n is.

Conventions (spec 2.3)
----------------------
* Positions are 0-based. **The window methods take INCLUSIVE ranges**
  ``[l, r]`` with ``0 <= l <= r <= n - 1``, converted once to the half-open
  ``[l, r + 1)`` of the wavelet tree by
  :func:`~src.advds.applications.range_queries.inclusive_to_half_open`.
  An index outside ``[0, n - 1]`` raises ``IndexError``, ``l > r`` raises
  ``ValueError``, a non-integer index raises ``TypeError``.
* ``k`` in :meth:`DNAIndex.kth_smallest` is 1-based, in alphabet order
  ``A < C < G < T``.
* Sequences are uppercase ``str`` over ``ACGT`` only (:func:`read_fasta`
  uppercases soft-masked lowercase bases); any other symbol, ``$`` included,
  raises ``ValueError``. Patterns are case-sensitive; a pattern holding a
  symbol other than ``A``, ``C``, ``G``, ``T`` occurs 0 times.

Synthetic data only
-------------------
:func:`synthetic_dna` and :func:`make_synthetic_fasta` produce **synthetic**
sequences: independent uniform draws of A, C, G and T. They are not a real
genome and carry none of a real genome's structure. The instructions'
real-dataset option is not used.

Examples:
    >>> idx = DNAIndex("ACGTACGTTGCA")
    >>> idx.count("A", 2, 9), [idx.kth_smallest(3, 8, k) for k in (1, 3, 6)]
    (1, ['A', 'G', 'T'])
    >>> idx.pattern_count("ACG"), idx.pattern_count("T"), idx.pattern_count("AAA")
    (2, 3, 0)
    >>> idx.bwt()
    'AC$TGAATCCGTG'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import numpy as np

from src.advds.applications.range_queries import inclusive_to_half_open
from src.advds.succinct.wavelet_tree import WaveletTree

__all__ = [
    "DEFAULT_FASTA",
    "DNAIndex",
    "DNA_ALPHABET",
    "FASTA_LINE_WIDTH",
    "FM_ALPHABET",
    "TERMINATOR",
    "build_suffix_array",
    "bwt_from_suffix_array",
    "make_synthetic_fasta",
    "read_fasta",
    "synthetic_dna",
]

#: The four bases in alphabet order, A < C < G < T.
DNA_ALPHABET: str = "ACGT"

#: The end-of-text symbol of the FM-index; it sorts before every base.
TERMINATOR: str = "$"

#: The BWT alphabet: the terminator, then the bases.
FM_ALPHABET: str = TERMINATOR + DNA_ALPHABET

#: Bases per line in the FASTA files this module writes.
FASTA_LINE_WIDTH: int = 60

#: The committed synthetic example, ``src/advds/examples/synthetic_fragment.fa``.
DEFAULT_FASTA: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "synthetic_fragment.fa",
)

_BASES = frozenset(DNA_ALPHABET)

PathLike = Union[str, "os.PathLike[str]"]


# ---------------------------------------------------------------------------
# Synthetic data and FASTA files
# ---------------------------------------------------------------------------


def _check_count(value: Any, name: str) -> int:
    """Return ``value`` as an ``int`` if it is a non-negative integer, else raise.

    Args:
        value: The proposed length or seed.
        name: The parameter name, for the error message.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an integer (``bool`` rejected).
        ValueError: If ``value`` is negative.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_count(5, "n")
        5
        >>> _check_count(-1, "n")
        Traceback (most recent call last):
            ...
        ValueError: n must be non-negative, got -1
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def synthetic_dna(n: int, seed: int) -> str:
    """Return n synthetic bases: ``np.random.default_rng(seed).integers(0, 4, n)`` mapped to "ACGT".

    The expression is exactly
    ``"".join(np.array(list("ACGT"))[np.random.default_rng(seed).integers(0, 4, n)])``.
    Apart from the benchmark module, which seeds numpy generators from
    ``derive_seed``, this is the one place in the package where numpy's
    generator is used directly instead of
    :func:`src.utils.random_utils.make_rng`, because the spec 9.7 anchors
    were computed with this exact expression. The result is
    synthetic (independent uniform bases), never a real genome.

    Args:
        n: The number of bases, ``n >= 0``.
        seed: The seed of ``np.random.default_rng``, ``seed >= 0``.

    Returns:
        A ``str`` of length n over ``ACGT``.

    Raises:
        TypeError: If ``n`` or ``seed`` is not an integer.
        ValueError: If ``n`` or ``seed`` is negative.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> synthetic_dna(20, 42)
        'ATGCCTAGAAGTGTGTGATC'
        >>> synthetic_dna(0, 42)
        ''
    """
    n = _check_count(n, "n")
    seed = _check_count(seed, "seed")
    return "".join(np.array(list("ACGT"))[np.random.default_rng(seed).integers(0, 4, n)])


def read_fasta(path: PathLike) -> str:
    """Read the sequence of a FASTA file as one uppercase string.

    Header lines (starting with ``>``) and comment lines (starting with
    ``;``) are skipped; every other line is stripped of whitespace,
    uppercased (lowercase marks soft-masked bases) and appended. A file with
    several records gives their sequences joined end to end.

    Args:
        path: The FASTA file.

    Returns:
        The concatenated sequence (``""`` if the file has none).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If it cannot be read.

    Time Complexity:
        O(file size).

    Space Complexity:
        O(file size).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = os.path.join(d, "x.fa")
        ...     with open(p, "w") as f:
        ...         _ = f.write(">demo\\nacgt\\nTTGA\\n")
        ...     read_fasta(p)
        'ACGTTTGA'
    """
    parts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">") or line.startswith(";"):
                continue
            stripped = line.strip()
            if stripped:
                parts.append(stripped.upper())
    return "".join(parts)


def make_synthetic_fasta(path: PathLike, n: int = 20000, seed: int = 42) -> str:
    """Write ``synthetic_dna(n, seed)`` as a FASTA file with 60-base lines.

    The header line states that the sequence is synthetic, so the file can
    never be mistaken for real data. Parent directories are created. With
    the defaults the file is about 20 KB.

    Args:
        path: Where to write the file (overwritten if it exists).
        n: The number of bases.
        seed: The seed passed to :func:`synthetic_dna`.

    Returns:
        ``path`` as a ``str``.

    Raises:
        TypeError: If ``n`` or ``seed`` is not an integer.
        ValueError: If ``n`` or ``seed`` is negative.
        OSError: If the file cannot be written.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = make_synthetic_fasta(os.path.join(d, "s.fa"), n=130, seed=42)
        ...     lines = open(p).read().splitlines()
        ...     same = read_fasta(p) == synthetic_dna(130, 42)
        >>> "synthetic" in lines[0], [len(x) for x in lines[1:]], same
        (True, [60, 60, 10], True)
    """
    n = _check_count(n, "n")
    seed = _check_count(seed, "seed")
    seq = synthetic_dna(n, seed)
    lines = [
        f">synthetic_fragment n={n} seed={seed} | synthetic DNA, uniform random "
        "A/C/G/T from numpy default_rng(seed), not a real genome"
    ]
    for start in range(0, n, FASTA_LINE_WIDTH):
        lines.append(seq[start : start + FASTA_LINE_WIDTH])
    target = os.fspath(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return target


# ---------------------------------------------------------------------------
# Suffix array and Burrows-Wheeler transform
# ---------------------------------------------------------------------------


def _code_points(text: str) -> np.ndarray:
    """Return the Unicode code points of ``text`` as an int64 array.

    Args:
        text: Any string.

    Returns:
        ``np.array([ord(ch) for ch in text])`` computed in one pass over the
        UTF-32 encoding (``surrogatepass``, so a lone surrogate such as
        ``"\\ud800"`` keeps its code point instead of raising).

    Raises:
        Nothing.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _code_points("$AC").tolist()
        [36, 65, 67]
        >>> _code_points("a\\ud800").tolist()
        [97, 55296]
    """
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype="<u4").astype(np.int64)


def build_suffix_array(text: str) -> np.ndarray:
    """Return the suffix array of ``text`` by prefix doubling with ``np.lexsort``.

    ``SA[i]`` is the start of the i-th smallest suffix. Symbols compare by
    code point (``$`` < ``A`` < ``C`` < ``G`` < ``T``), and a suffix that is
    a prefix of another sorts first. Round k sorts the pairs (rank of the
    first 2^k symbols, rank of the next 2^k symbols, -1 past the end) with
    ``np.lexsort``, then gives equal pairs equal new ranks (group starts by
    ``np.flatnonzero``, group ids spread with ``np.repeat``). It stops once
    all n ranks are distinct.

    Args:
        text: Any string (the FM-index passes ``seq + "$"``).

    Returns:
        An ``int64`` numpy array, a permutation of ``range(len(text))``.

    Raises:
        TypeError: If ``text`` is not a ``str``.

    Time Complexity:
        O(n log^2 n) at worst: at most ``ceil(log2 n) + 1`` rounds of an
        O(n log n) sort; random DNA needs only about ``log2(log4 n) + 2``
        rounds.

    Space Complexity:
        O(n) words.

    Examples:
        >>> build_suffix_array("banana$").tolist()
        [6, 5, 3, 1, 0, 4, 2]
        >>> build_suffix_array("ACGTACGTTGCA$").tolist()
        [12, 11, 0, 4, 10, 1, 5, 9, 2, 6, 3, 8, 7]
        >>> build_suffix_array("").tolist()
        []
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    n = len(text)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rank = _code_points(text)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        if k < n:
            second[: n - k] = rank[k:]
        # lexsort sorts by the LAST key first: rank, then second as tie-break.
        sa = np.lexsort((second, rank)).astype(np.int64)
        first_sorted = rank[sa]
        second_sorted = second[sa]
        boundary = np.empty(n, dtype=bool)
        boundary[0] = True
        boundary[1:] = (first_sorted[1:] != first_sorted[:-1]) | (
            second_sorted[1:] != second_sorted[:-1]
        )
        starts = np.flatnonzero(boundary)
        groups = len(starts)
        lengths = np.diff(np.append(starts, n))
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[sa] = np.repeat(np.arange(groups, dtype=np.int64), lengths)
        rank = new_rank
        if groups == n:
            return sa
        k *= 2


def bwt_from_suffix_array(text: str, sa: Any) -> str:
    """Return the Burrows-Wheeler transform ``BWT[i] = text[SA[i] - 1]``.

    For ``SA[i] = 0`` the index wraps to the last symbol, which is the
    terminator when ``text`` ends with ``$``.

    Args:
        text: The text the suffix array was built for.
        sa: Its suffix array (a numpy array or a list of ints).

    Returns:
        The BWT, a ``str`` of the same length.

    Raises:
        TypeError: If ``text`` is not a ``str``.
        ValueError: If ``sa`` does not have the length of ``text``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> t = "ACGTACGTTGCA$"
        >>> bwt_from_suffix_array(t, build_suffix_array(t))
        'AC$TGAATCCGTG'
        >>> bwt_from_suffix_array("banana$", build_suffix_array("banana$"))
        'annb$aa'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    order = np.asarray(sa, dtype=np.int64)
    n = len(text)
    if order.shape != (n,):
        raise ValueError(f"suffix array has shape {order.shape}, text has length {n}")
    if n == 0:
        return ""
    codes = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype="<u4")
    before = codes[(order - 1) % n]
    return before.astype("<u4").tobytes().decode("utf-32-le", "surrogatepass")


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


class DNAIndex:
    """Wavelet-tree window queries and FM-index pattern counts over a DNA sequence.

    Window methods take INCLUSIVE ranges ``[l, r]``. See the module
    docstring for the structures and the conventions.

    Args:
        seq: An uppercase ``str`` over ``ACGT`` (may be empty).

    Raises:
        TypeError: If ``seq`` is not a ``str``.
        ValueError: If ``seq`` holds a symbol other than ``A``, ``C``, ``G``
            or ``T``.

    Time Complexity:
        Build O(n log^2 n) at worst for the suffix array (a few lexsort
        rounds on random DNA) plus O(n log sigma) for the two wavelet trees.
        :meth:`count` and :meth:`kth_smallest` O(log sigma) = 2 levels;
        :meth:`pattern_count` O(m log sigma) for a pattern of length m.

    Space Complexity:
        Wavelet payloads of 2 bits per base for the sequence and between 2
        and 3 per symbol for the BWT (its 5-symbol alphabet puts ``$`` and
        ``A`` on level 3 and ``C``, ``G``, ``T`` on level 2, about 2.25 bits
        on uniform DNA), plus their rank directories, the BWT string and the
        suffix array (kept for :meth:`suffix_array`).

    Examples:
        >>> idx = DNAIndex("GATTACA")
        >>> len(idx), idx.frequencies()
        (7, {'A': 3, 'C': 1, 'G': 1, 'T': 2})
        >>> idx.pattern_count("TA"), idx.pattern_count("GATTACA"), idx.pattern_count("N")
        (1, 1, 0)
        >>> DNAIndex("ACGN")
        Traceback (most recent call last):
            ...
        ValueError: symbol 'N' at position 3 is not in the alphabet
    """

    __slots__ = ("_n", "_wt", "_sa", "_bwt", "_fm", "_c")

    def __init__(self, seq: str) -> None:
        """Build the ACGT wavelet tree over ``seq`` and the FM-index over ``seq + "$"``.

        Args:
            seq: An uppercase ``str`` over ``ACGT`` (may be empty).

        Returns:
            None.

        Raises:
            TypeError: If ``seq`` is not a ``str``.
            ValueError: If ``seq`` holds a symbol other than ``A``, ``C``,
                ``G`` or ``T`` (``$`` and lowercase included).

        Time Complexity:
            O(n log^2 n) at worst for the suffix array, plus O(n log sigma)
            for the two wavelet trees and O(sigma) for the C array.

        Space Complexity:
            O(n) words for the suffix array, O(n log sigma) bits for the
            wavelet trees.

        Examples:
            >>> DNAIndex("ACGTACGTTGCA").bwt()
            'AC$TGAATCCGTG'
            >>> DNAIndex("ACG$")
            Traceback (most recent call last):
                ...
            ValueError: symbol '$' at position 3 is not in the alphabet
        """
        if not isinstance(seq, str):
            raise TypeError(f"seq must be a str, got {type(seq).__name__}")
        # The ACGT wavelet tree is built first, so a bad symbol is reported
        # before any suffix sorting is done.
        self._wt: WaveletTree = WaveletTree(seq, alphabet=DNA_ALPHABET)
        self._n: int = len(seq)
        text = seq + TERMINATOR
        self._sa: np.ndarray = build_suffix_array(text)
        self._bwt: str = bwt_from_suffix_array(text, self._sa)
        self._fm: WaveletTree = WaveletTree(self._bwt, alphabet=FM_ALPHABET)
        counts = self._fm.frequencies()
        c_array: Dict[str, int] = {}
        smaller = 0
        for symbol in FM_ALPHABET:
            c_array[symbol] = smaller
            smaller += counts[symbol]
        self._c: Dict[str, int] = c_array

    def __len__(self) -> int:
        """Return n, the number of bases (the terminator is not counted).

        Returns:
            The sequence length.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(DNAIndex("ACGT")), len(DNAIndex(""))
            (4, 0)
        """
        return self._n

    def count(self, base: str, l: int, r: int) -> int:
        """Return how many times ``base`` occurs in the inclusive range ``[l, r]``.

        Two wavelet-tree ranks on the half-open ``[l, r + 1)``.

        Args:
            base: One of ``"A"``, ``"C"``, ``"G"``, ``"T"``.
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.

        Returns:
            The number of positions p with ``l <= p <= r`` holding ``base``.

        Raises:
            ValueError: If ``base`` is not one of the four bases, or
                ``l > r``.
            TypeError: If ``l`` or ``r`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]``.

        Time Complexity:
            O(log sigma): 2 levels, two ``rank1`` calls each.

        Space Complexity:
            O(1).

        Examples:
            >>> idx = DNAIndex("ACGTACGTTGCA")
            >>> idx.count("A", 2, 9), idx.count("T", 0, 11), idx.count("G", 2, 2)
            (1, 3, 1)
            >>> idx.count("N", 0, 3)
            Traceback (most recent call last):
                ...
            ValueError: base must be one of 'A', 'C', 'G', 'T', got 'N'
        """
        if not isinstance(base, str) or base not in _BASES:
            raise ValueError(f"base must be one of 'A', 'C', 'G', 'T', got {base!r}")
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._wt.range_count(base, lo, hi)

    def kth_smallest(self, l: int, r: int, k: int) -> str:
        """Return the k-th smallest base (1-based, A < C < G < T) of the inclusive range ``[l, r]``.

        The wavelet tree's range quantile on ``[l, r + 1)``: at each of the
        2 levels, the number of 0 bits in the range says whether the k-th
        smallest is in the A/C half or the G/T half.

        Args:
            l: First position, inclusive.
            r: Last position, inclusive, ``l <= r <= n - 1``.
            k: The rank, ``1 <= k <= r - l + 1``; 1 is the minimum.

        Returns:
            One of ``"A"``, ``"C"``, ``"G"``, ``"T"``.

        Raises:
            TypeError: If ``l``, ``r`` or ``k`` is not an integer.
            IndexError: If ``l`` or ``r`` is outside ``[0, n - 1]``, or ``k``
                is outside ``[1, r - l + 1]``.
            ValueError: If ``l > r``.

        Time Complexity:
            O(log sigma): 2 levels.

        Space Complexity:
            O(1).

        Examples:
            >>> idx = DNAIndex("ACGTACGTTGCA")
            >>> [idx.kth_smallest(3, 8, k) for k in range(1, 7)]
            ['A', 'C', 'G', 'T', 'T', 'T']
            >>> idx.kth_smallest(3, 8, 7)
            Traceback (most recent call last):
                ...
            IndexError: quantile rank 7 out of range [1, 6]
        """
        lo, hi = inclusive_to_half_open(l, r, self._n)
        return self._wt.range_quantile(lo, hi, k)

    def frequencies(self) -> Dict[str, int]:
        """Return the number of occurrences of each base, in the order A, C, G, T.

        Returns:
            ``{"A": ..., "C": ..., "G": ..., "T": ...}`` read from the
            wavelet tree's rank directories (0 for an absent base).

        Raises:
            Nothing.

        Time Complexity:
            O(sigma log sigma).

        Space Complexity:
            O(sigma).

        Examples:
            >>> DNAIndex("ACGTACGTTGCA").frequencies()
            {'A': 3, 'C': 3, 'G': 3, 'T': 3}
        """
        return self._wt.frequencies()

    def pattern_count(self, pattern: str) -> int:
        """Return the number of occurrences of ``pattern``, overlaps included, by backward search.

        Reads the pattern from right to left, keeping the half-open range
        ``[lo, hi)`` of suffix-array rows that start with the part read so
        far: ``lo = C[c] + rank_c(lo)``, ``hi = C[c] + rank_c(hi)``, with
        ``rank_c`` the wavelet tree over the BWT.

        Args:
            pattern: A non-empty ``str``; case-sensitive.

        Returns:
            The number of start positions p with ``seq[p:p + m] == pattern``
            (overlaps included, so ``"AA"`` occurs 3 times in ``"AAAA"``); 0
            if the pattern holds a symbol other than ``A``, ``C``, ``G``,
            ``T`` or is longer than the sequence.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m log sigma) for a pattern of length m: two BWT ranks per
            symbol (2 levels for C, G and T, 3 for A), independent of n.

        Space Complexity:
            O(1).

        Examples:
            >>> idx = DNAIndex("ACGTACGTTGCA")
            >>> [idx.pattern_count(p) for p in ("ACG", "CG", "GT", "T", "TTG", "GCA", "AAA")]
            [2, 2, 2, 3, 1, 1, 0]
            >>> DNAIndex("AAAA").pattern_count("AA")
            3
            >>> idx.pattern_count("")
            Traceback (most recent call last):
                ...
            ValueError: pattern must not be empty
        """
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a str, got {type(pattern).__name__}")
        if not pattern:
            raise ValueError("pattern must not be empty")
        fm = self._fm
        c_array = self._c
        lo = 0
        hi = self._n + 1
        for j in range(len(pattern) - 1, -1, -1):
            symbol = pattern[j]
            if symbol not in _BASES:
                return 0
            base = c_array[symbol]
            lo = base + fm.rank(symbol, lo)
            hi = base + fm.rank(symbol, hi)
            if lo >= hi:
                return 0
        return hi - lo

    def bwt(self) -> str:
        """Return the Burrows-Wheeler transform of ``seq + "$"``.

        Returns:
            A ``str`` of length n + 1 over ``$ACGT``.

        Raises:
            Nothing.

        Time Complexity:
            O(1) (the stored string).

        Space Complexity:
            O(1).

        Examples:
            >>> DNAIndex("ACGTACGTTGCA").bwt()
            'AC$TGAATCCGTG'
        """
        return self._bwt

    def suffix_array(self) -> np.ndarray:
        """Return a copy of the suffix array of ``seq + "$"``.

        Returns:
            An ``int64`` numpy array of length n + 1; entry 0 is n, the
            position of the terminator, which is the smallest suffix.

        Raises:
            Nothing.

        Time Complexity:
            O(n) for the copy.

        Space Complexity:
            O(n).

        Examples:
            >>> DNAIndex("GATTACA").suffix_array().tolist()
            [7, 6, 4, 1, 5, 0, 3, 2]
        """
        return self._sa.copy()

    def __repr__(self) -> str:
        """Return the length of the indexed sequence.

        Returns:
            For example ``DNAIndex(n=12)``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> DNAIndex("ACGT")
            DNAIndex(n=4)
        """
        return f"DNAIndex(n={self._n})"
