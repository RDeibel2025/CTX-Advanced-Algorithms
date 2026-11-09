"""Seeded texts and patterns for the Week 11 tests, applications and benchmark.

Every random draw comes from one :func:`src.utils.random_utils.make_rng`
generator per call, so a seed fixes the whole text or pattern set and no large
text ever needs to be committed: the benchmark regenerates its 5 MB inputs
from seeds at run time.

The generators cover the inputs that make the four search methods behave
differently:

* :func:`random_text` and :func:`dna_text` draw every character uniformly and
  independently, so repeats are short (the longest repeated substring of n
  random DNA bases is about ``2 log_4 n``) and LCP values stay small.
* :func:`english_like_text` draws whole words from a fixed 2,000-word
  synthetic vocabulary with Zipf weights ``1 / r**1.1`` for the word of rank
  ``r``. A few words dominate, so phrases repeat the way they do in prose and
  the LCP values are larger than in random text. The vocabulary is made once,
  deterministically, from consonant-vowel syllables with its own fixed seed.
* :func:`periodic_text` and :func:`worst_case_rk` are the adversarial inputs:
  every window of ``'a' * n`` matches ``'a' * m``, which forces Rabin-Karp to
  verify every window (its O(nm) case) while KMP stays O(n + m).

Nothing here searches, so nothing is stored before a search: these functions
only produce the text that the indexes are built over and the patterns that
query it. :func:`write_fasta` and :func:`generate_sample_genome` made the
committed ``examples/sample_genome.fasta``, three synthetic sequences that are
not real genomic data.

====================== =============== ===============
Function               Time            Space
====================== =============== ===============
random_text            O(n)            O(n)
dna_text               O(n)            O(n)
english_like_text      O(n log V)      O(n)
periodic_text          O(n)            O(n)
worst_case_rk          O(n + m)        O(n + m)
sample_patterns        O(n + count m)  O(n + count m)
write_fasta            O(total length) O(total length)
generate_sample_genome O(n_seqs len)   O(n_seqs len)
====================== =============== ===============

Here V = 2,000 is the vocabulary size (one binary search per word drawn).

Examples:
    >>> dna_text(12, 1) == dna_text(12, 1)
    True
    >>> len(english_like_text(1000, 3))
    1000
    >>> periodic_text(10, 3)
    'abcabcabca'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import string
from functools import lru_cache
from itertools import accumulate
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from src.utils.random_utils import derive_seed, make_rng

__all__ = [
    "DNA_ALPHABET",
    "VOCABULARY_SIZE",
    "ZIPF_EXPONENT",
    "dna_text",
    "english_like_text",
    "generate_sample_genome",
    "periodic_text",
    "random_text",
    "sample_patterns",
    "worst_case_rk",
    "write_fasta",
]

#: The four DNA bases, in the order the sample genome draws them.
DNA_ALPHABET: str = "ACGT"

#: Number of distinct words in the synthetic vocabulary.
VOCABULARY_SIZE: int = 2_000

#: Zipf exponent s: the word of rank r has weight 1 / r**s.
ZIPF_EXPONENT: float = 1.1

#: Fixed seed for building the vocabulary, so it is the same for every text seed.
_VOCAB_SEED: int = 1_111

#: Syllable parts for the synthetic words.
_ONSETS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"
_CODAS = "lmnrst"

#: Probabilities of an onset consonant and of a closing consonant per syllable.
_P_ONSET = 0.85
_P_CODA = 0.3

#: Syllables per word, drawn uniformly from this tuple (2 and 3 are favoured).
_SYLLABLE_COUNTS = (1, 2, 2, 3, 3, 4)

#: Extra words drawn on top of the expected need, so one batch nearly always suffices.
_DRAW_MARGIN = 1.02

#: Path-like values accepted by :func:`write_fasta`.
PathLike = Union[str, "os.PathLike[str]"]


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` if it is an ``int`` of at least ``minimum``.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.
        minimum: The smallest value allowed.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int`` (``bool`` rejected).
        ValueError: If ``value`` is below ``minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(3, "n", 0)
        3
        >>> _require_int(-1, "n", 0)
        Traceback (most recent call last):
            ...
        ValueError: n must be at least 0, got -1
        >>> _require_int(2.0, "n", 0)
        Traceback (most recent call last):
            ...
        TypeError: n must be an int, got float
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _require_alphabet(alphabet: Sequence[str]) -> Sequence[str]:
    """Return ``alphabet`` if it is a non-empty sequence of single characters.

    Args:
        alphabet: A ``str`` such as ``"ACGT"`` or a sequence of 1-character
            strings such as ``sorted(set(text))``.

    Returns:
        ``alphabet`` unchanged.

    Raises:
        ValueError: If ``alphabet`` is empty or holds anything other than
            1-character strings.

    Time Complexity:
        O(k) for an alphabet of k symbols.

    Space Complexity:
        O(1).

    Examples:
        >>> _require_alphabet("ACGT")
        'ACGT'
        >>> _require_alphabet(["a", "b"])
        ['a', 'b']
        >>> _require_alphabet("")
        Traceback (most recent call last):
            ...
        ValueError: alphabet must be a non-empty sequence of single characters
        >>> _require_alphabet(["ab"])
        Traceback (most recent call last):
            ...
        ValueError: alphabet must be a non-empty sequence of single characters
    """
    if len(alphabet) == 0 or not all(isinstance(c, str) and len(c) == 1 for c in alphabet):
        raise ValueError("alphabet must be a non-empty sequence of single characters")
    return alphabet


def random_text(n: int, alphabet: Sequence[str], seed: Optional[int]) -> str:
    """Return ``n`` characters drawn uniformly and independently from ``alphabet``.

    Each position is an independent uniform draw (:meth:`random.Random.choices`
    without weights), so for a ``k``-letter alphabet a fixed window of length
    ``m`` occurs at a given position with probability ``k**-m``. That makes
    random text the easy case for every method: matches are rare and repeats
    are short.

    Args:
        n: Text length. Must be non-negative.
        alphabet: A non-empty ``str`` or sequence of 1-character strings.
            Repeating a symbol makes it proportionally more likely.
        seed: Passed to :func:`make_rng`; an ``int`` makes the text
            reproducible.

    Returns:
        A ``str`` of exactly ``n`` characters from ``alphabet``.

    Raises:
        TypeError: If ``n`` is not an ``int`` or ``seed`` is not an ``int``
            or ``None``.
        ValueError: If ``n`` is negative or ``alphabet`` is invalid.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the draws and the joined string.

    Examples:
        >>> random_text(12, "ab", 7)
        'aababaabaaaa'
        >>> random_text(5, ["x", "y", "z"], 7) == random_text(5, "xyz", 7)
        True
        >>> random_text(0, "ab", 7)
        ''
        >>> random_text(3, "", 7)
        Traceback (most recent call last):
            ...
        ValueError: alphabet must be a non-empty sequence of single characters
    """
    _require_int(n, "n", 0)
    symbols = _require_alphabet(alphabet)
    rng = make_rng(seed)
    return "".join(rng.choices(symbols, k=n))


def dna_text(n: int, seed: Optional[int]) -> str:
    """Return ``n`` uniformly random DNA bases over ``"ACGT"``.

    This is :func:`random_text` with the alphabet fixed to
    :data:`DNA_ALPHABET`, the benchmark's ``dna`` dataset. With only four
    letters, short patterns match often, but the longest repeat in ``n``
    random bases is only about ``2 log_4 n`` characters.

    Args:
        n: Number of bases. Must be non-negative.
        seed: Passed to :func:`make_rng`.

    Returns:
        A ``str`` of exactly ``n`` characters from ``"ACGT"``.

    Raises:
        TypeError: If ``n`` is not an ``int`` or ``seed`` is not an ``int``
            or ``None``.
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> dna_text(16, 1)
        'ATTCCCGTAATCTACG'
        >>> set(dna_text(1000, 2)) == set("ACGT")
        True
        >>> dna_text(-1, 1)
        Traceback (most recent call last):
            ...
        ValueError: n must be at least 0, got -1
    """
    return random_text(n, DNA_ALPHABET, seed)


def _make_word(rng: Any) -> str:
    """Build one synthetic word from 1 to 4 consonant-vowel syllables.

    Each syllable is an optional onset consonant (probability 0.85), one
    vowel, and an optional closing consonant (probability 0.3).

    Args:
        rng: The :class:`random.Random` generator to draw from.

    Returns:
        A non-empty lowercase ``str``.

    Raises:
        Nothing.

    Time Complexity:
        O(1) - at most four syllables of at most three letters.

    Space Complexity:
        O(1).

    Examples:
        >>> _make_word(make_rng(5))
        'purkave'
        >>> _make_word(make_rng(5)) == _make_word(make_rng(5))
        True
    """
    parts: List[str] = []
    for _ in range(rng.choice(_SYLLABLE_COUNTS)):
        if rng.random() < _P_ONSET:
            parts.append(rng.choice(_ONSETS))
        parts.append(rng.choice(_VOWELS))
        if rng.random() < _P_CODA:
            parts.append(rng.choice(_CODAS))
    return "".join(parts)


@lru_cache(maxsize=1)
def _vocabulary() -> Tuple[Tuple[str, ...], Tuple[float, ...], float]:
    """Build the fixed synthetic vocabulary and its cumulative Zipf weights.

    Draws words with :func:`_make_word` from ``make_rng(_VOCAB_SEED)`` until
    :data:`VOCABULARY_SIZE` distinct words exist; drawing order is rank
    order. The word at rank ``r`` (1-based) gets weight
    ``1 / r**ZIPF_EXPONENT``. The mean drawn word is about 5 letters, close
    to English running text. The result is cached, so every call to
    :func:`english_like_text` shares one vocabulary.

    Args:
        None.

    Returns:
        ``(words, cum_weights, mean_cost)``: the words in rank order, their
        running weight totals for :meth:`random.Random.choices`, and the
        expected characters per drawn word including its trailing space.

    Raises:
        Nothing.

    Time Complexity:
        O(V) for V = 2,000 words, once per process.

    Space Complexity:
        O(V).

    Examples:
        >>> words, cum, cost = _vocabulary()
        >>> len(words), len(set(words)), len(cum)
        (2000, 2000, 2000)
        >>> words[:5]
        ('giha', 'fosla', 'metti', 'ga', 'tehu')
        >>> 2 < cost < 8
        True
    """
    rng = make_rng(_VOCAB_SEED)
    seen: Dict[str, None] = {}
    while len(seen) < VOCABULARY_SIZE:
        seen.setdefault(_make_word(rng), None)
    words = tuple(seen)
    weights = [1.0 / (rank ** ZIPF_EXPONENT) for rank in range(1, VOCABULARY_SIZE + 1)]
    total = sum(weights)
    mean_cost = sum(w * (len(word) + 1) for w, word in zip(weights, words)) / total
    return words, tuple(accumulate(weights)), mean_cost


def english_like_text(n_chars: int, seed: Optional[int]) -> str:
    """Return exactly ``n_chars`` characters of Zipf-distributed synthetic prose.

    Words are drawn independently from the fixed 2,000-word vocabulary with
    Zipf weights ``1 / r**1.1`` and joined by single spaces. With s = 1.1 the
    most frequent word is about 17% of all words and the top ten about 45%,
    so short phrases recur the way they do in English and the suffix array
    sees long LCP runs. The generator draws the expected number of words
    for ``n_chars`` in one batch (plus 2%), draws more only if that falls
    short, and cuts the joined text at ``n_chars``, so the last word may be
    cut off.

    Args:
        n_chars: Exact length of the result. Must be non-negative.
        seed: Passed to :func:`make_rng`. The vocabulary itself does not
            depend on it.

    Returns:
        A ``str`` of exactly ``n_chars`` characters: lowercase letters and
        single spaces.

    Raises:
        TypeError: If ``n_chars`` is not an ``int`` or ``seed`` is not an
            ``int`` or ``None``.
        ValueError: If ``n_chars`` is negative.

    Time Complexity:
        O(n log V) - one binary search over the cumulative weights per word.

    Space Complexity:
        O(n).

    Examples:
        >>> english_like_text(40, 1)
        'giha polue vuteson metti rifibete jipi v'
        >>> text = english_like_text(5000, 2)
        >>> len(text), set(text) <= set(" abcdefghijklmnopqrstuvwxyz")
        (5000, True)
        >>> english_like_text(5000, 2) == text
        True
        >>> english_like_text(0, 2)
        ''
    """
    _require_int(n_chars, "n_chars", 0)
    if n_chars == 0:
        return ""
    words, cum_weights, mean_cost = _vocabulary()
    rng = make_rng(seed)
    pieces: List[str] = []
    length = 0
    while length < n_chars:
        k = max(16, math.ceil((n_chars - length) / mean_cost * _DRAW_MARGIN))
        batch = " ".join(rng.choices(words, cum_weights=cum_weights, k=k)) + " "
        pieces.append(batch)
        length += len(batch)
    return "".join(pieces)[:n_chars]


def periodic_text(n: int, period: int) -> str:
    """Return the first ``period`` lowercase letters repeated and cut to ``n``.

    The text is fully deterministic (no seed): ``periodic_text(10, 3)`` is
    ``"abcabcabca"``. Every suffix shares a prefix of length ``n - i - period``
    with the suffix ``period`` places later, so the LCP values grow linearly,
    the opposite extreme to random text.

    Args:
        n: Text length. Must be non-negative.
        period: Length of the repeated block, from 1 to 26.

    Returns:
        A ``str`` of exactly ``n`` characters.

    Raises:
        TypeError: If ``n`` or ``period`` is not an ``int``.
        ValueError: If ``n`` is negative or ``period`` is outside 1 to 26.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> periodic_text(10, 3)
        'abcabcabca'
        >>> periodic_text(4, 1)
        'aaaa'
        >>> periodic_text(5, 27)
        Traceback (most recent call last):
            ...
        ValueError: period must be at most 26, got 27
    """
    _require_int(n, "n", 0)
    _require_int(period, "period", 1)
    if period > len(string.ascii_lowercase):
        raise ValueError(f"period must be at most 26, got {period}")
    block = string.ascii_lowercase[:period]
    return (block * (n // period + 1))[:n]


def worst_case_rk(n: int, m: int) -> Tuple[str, str]:
    """Return ``('a' * n, 'a' * m)``, the input that maximises Rabin-Karp's work.

    Every one of the ``n - m + 1`` windows has the pattern's hash because it
    is the pattern, so Rabin-Karp verifies every window character by
    character: ``(n - m + 1) * m`` comparisons, its O(nm) worst case. KMP on
    the same input still makes at most ``2n`` comparisons.

    Args:
        n: Text length. Must be non-negative.
        m: Pattern length. Must be non-negative.

    Returns:
        The pair ``(text, pattern)``.

    Raises:
        TypeError: If ``n`` or ``m`` is not an ``int``.
        ValueError: If ``n`` or ``m`` is negative.

    Time Complexity:
        O(n + m).

    Space Complexity:
        O(n + m).

    Examples:
        >>> worst_case_rk(6, 3)
        ('aaaaaa', 'aaa')
        >>> worst_case_rk(2, -1)
        Traceback (most recent call last):
            ...
        ValueError: m must be at least 0, got -1
    """
    _require_int(n, "n", 0)
    _require_int(m, "m", 0)
    return "a" * n, "a" * m


def sample_patterns(
    text: str,
    count: int,
    m: int,
    seed: Optional[int],
    present_fraction: float = 0.5,
) -> List[str]:
    """Return ``count`` query patterns of length ``m``, some taken from ``text``.

    The first ``round(count * present_fraction)`` patterns are substrings of
    ``text`` starting at uniformly random positions, so each has at least one
    match. The rest are uniform random strings over the text's own alphabet
    ``sorted(set(text))``; for realistic ``m`` they almost never occur, which
    exercises the "no match" path of every method. ``round`` is Python's
    round-half-to-even, so 5 patterns at 0.5 gives 2 present ones.

    Args:
        text: The text the patterns will query. Must be non-empty.
        count: Number of patterns. Must be non-negative.
        m: Length of every pattern, at least 1. Must be at most
            ``len(text)`` when any present pattern is requested.
        seed: Passed to :func:`make_rng`.
        present_fraction: Fraction of patterns cut from ``text``, in [0, 1].

    Returns:
        A list of ``count`` strings of length ``m``: present ones first, then
        random ones.

    Raises:
        TypeError: If ``count`` or ``m`` is not an ``int`` or ``seed`` is not
            an ``int`` or ``None``.
        ValueError: If ``text`` is empty, ``count`` is negative, ``m < 1``,
            ``present_fraction`` is outside [0, 1], or ``m > len(text)``
            while present patterns are requested.

    Time Complexity:
        O(n + count * m) - building the alphabet reads the whole text once.

    Space Complexity:
        O(k + count * m) for an alphabet of k symbols.

    Examples:
        >>> text = dna_text(200, 3)
        >>> pats = sample_patterns(text, 4, 6, 9)
        >>> [len(p) for p in pats], all(p in text for p in pats[:2])
        ([6, 6, 6, 6], True)
        >>> sample_patterns(text, 4, 6, 9) == pats
        True
        >>> len(sample_patterns(text, 5, 6, 9, present_fraction=0.5))
        5
        >>> sample_patterns("abc", 2, 4, 9)
        Traceback (most recent call last):
            ...
        ValueError: m = 4 is longer than the text (3 characters)
    """
    if not text:
        raise ValueError("text must be non-empty")
    _require_int(count, "count", 0)
    _require_int(m, "m", 1)
    if not 0.0 <= present_fraction <= 1.0:
        raise ValueError(f"present_fraction must be in [0, 1], got {present_fraction}")
    n_present = round(count * present_fraction)
    if n_present > 0 and m > len(text):
        raise ValueError(f"m = {m} is longer than the text ({len(text)} characters)")
    rng = make_rng(seed)
    last_start = len(text) - m
    patterns = []
    for _ in range(n_present):
        start = rng.randint(0, last_start)
        patterns.append(text[start:start + m])
    alphabet = sorted(set(text))
    for _ in range(count - n_present):
        patterns.append("".join(rng.choices(alphabet, k=m)))
    return patterns


def write_fasta(
    records: Union[Mapping[str, str], Iterable[Tuple[str, str]]],
    path_or_none: Optional[PathLike],
    width: int = 60,
) -> str:
    """Format sequences as FASTA text and optionally write it to a file.

    Each record becomes a header line ``">" + header`` followed by the
    sequence wrapped at ``width`` characters per line. The header is written
    exactly as given, so ``"seq1 some description"`` keeps its description
    after the first space; FASTA readers take the identifier up to that
    space. Lines end in ``"\\n"`` on every platform.

    Args:
        records: A mapping from header (without ``">"``) to sequence, or an
            iterable of ``(header, sequence)`` pairs, written in order.
        path_or_none: A file path to write, or ``None`` to only return the
            text. An existing file is overwritten.
        width: Sequence characters per line, at least 1.

    Returns:
        The FASTA text (the same string that was written, if a path was
        given). No records give ``""``.

    Raises:
        TypeError: If ``width`` is not an ``int``.
        ValueError: If ``width < 1`` or a header is empty or contains a
            newline.
        OSError: If the file cannot be written.

    Time Complexity:
        O(L) for L total characters.

    Space Complexity:
        O(L) - the FASTA text.

    Examples:
        >>> print(write_fasta({"s1 demo": "ACGTACGTAC", "s2": "GG"}, None, width=4), end="")
        >s1 demo
        ACGT
        ACGT
        AC
        >s2
        GG
        >>> import os, tempfile
        >>> fd, path = tempfile.mkstemp(suffix=".fasta")
        >>> os.close(fd)
        >>> text = write_fasta([("x", "ACG")], path)
        >>> open(path, encoding="utf-8").read() == text == ">x\\nACG\\n"
        True
        >>> os.remove(path)
        >>> write_fasta({"bad\\nheader": "A"}, None)
        Traceback (most recent call last):
            ...
        ValueError: FASTA headers must be non-empty single lines, got 'bad\\nheader'
    """
    _require_int(width, "width", 1)
    pairs = records.items() if isinstance(records, Mapping) else records
    lines: List[str] = []
    for header, sequence in pairs:
        if not header or "\n" in header or "\r" in header:
            raise ValueError(f"FASTA headers must be non-empty single lines, got {header!r}")
        lines.append(">" + header)
        lines.extend(sequence[i:i + width] for i in range(0, len(sequence), width))
    text = "".join(line + "\n" for line in lines)
    if path_or_none is not None:
        with open(path_or_none, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return text


def generate_sample_genome(seed: int = 11, n_seqs: int = 3, length: int = 2000) -> Dict[str, str]:
    """Return the synthetic sequences of ``examples/sample_genome.fasta``.

    Sequence ``i`` (1-based) is ``dna_text(length, derive_seed(seed,
    "sample_genome", i))``, so each sequence has its own independent seed.
    Its header is ``"synthetic_seq<i> synthetic sequence, seed <seed>, not
    real data"``: the data are uniform random bases, not a real genome.
    ``write_fasta(generate_sample_genome(), path)`` reproduces the committed
    file exactly, which the last example checks.

    Args:
        seed: Master seed for :func:`derive_seed`.
        n_seqs: Number of sequences. Must be non-negative.
        length: Bases per sequence. Must be non-negative.

    Returns:
        A dict from full header line (without ``">"``) to sequence, in order
        ``synthetic_seq1``, ``synthetic_seq2``, ...

    Raises:
        TypeError: If an argument is not an ``int``.
        ValueError: If ``n_seqs`` or ``length`` is negative.

    Time Complexity:
        O(n_seqs * length).

    Space Complexity:
        O(n_seqs * length).

    Examples:
        >>> genome = generate_sample_genome()
        >>> list(genome)[0]
        'synthetic_seq1 synthetic sequence, seed 11, not real data'
        >>> [len(s) for s in genome.values()]
        [2000, 2000, 2000]
        >>> from pathlib import Path
        >>> sample = Path(__file__).resolve().parents[1] / "examples" / "sample_genome.fasta"
        >>> sample.read_text(encoding="utf-8") == write_fasta(genome, None)
        True
    """
    _require_int(n_seqs, "n_seqs", 0)
    _require_int(length, "length", 0)
    return {
        f"synthetic_seq{i} synthetic sequence, seed {seed}, not real data": dna_text(
            length, derive_seed(seed, "sample_genome", i)
        )
        for i in range(1, n_seqs + 1)
    }
