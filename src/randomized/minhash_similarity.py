"""MinHash: estimating Jaccard similarity from short random signatures.

The Jaccard similarity of two sets is ``J(A, B) = |A & B| / |A | B|``, the
fraction of all their distinct tokens that the two sets share. Computing it
exactly needs both sets in full. MinHash (Broder, 1997) replaces each set
with a fixed-length *signature* of ``k`` integers, and two signatures alone
are enough to estimate ``J``.

Why the minimum works
---------------------
Apply one uniformly random permutation ``pi`` to the token universe and
look at the smallest permuted value in each set. Every token of ``A | B`` is
equally likely to be the overall minimum, and the two sets have the same
minimum exactly when that token lies in ``A & B``. So::

    P[min pi(A) == min pi(B)] = |A & B| / |A | B| = J(A, B)

Each signature position is therefore a coin that lands "equal" with
probability ``J``. With ``k`` independent permutations the fraction of equal
positions is the mean of ``k`` Bernoulli(J) trials: an unbiased estimate of
``J`` with standard error::

    SE = sqrt(J (1 - J) / k)

Quadrupling ``k`` halves the error. That ``k^(-1/2)`` rate is what the
Week 7 benchmark measures, from ``k = 16`` to ``k = 512``.

Why a hash instead of a permutation
-----------------------------------
A random permutation of a 64-bit token universe cannot be stored. In its
place each of the ``k`` positions uses a universal hash::

    h(x) = (a x + b) mod p,    p = 2^61 - 1  (MERSENNE_61)

with ``a`` uniform in ``[1, p - 1]`` and ``b`` uniform in ``[0, p - 1]``.
Because ``p`` is prime and ``a != 0``, ``h`` is a bijection on
``{0, ..., p - 1}``: it is itself a permutation of the reduced token values,
so two different tokens can share a minimum only if their integers already
collided. It is not a *uniformly random* permutation (only ``p (p - 1)`` of
them are reachable), so the min-wise property above holds approximately
rather than exactly. At the set sizes used here that bias is far below the
sampling error.

Tokens to integers: never ``hash()``
------------------------------------
A token becomes an integer through
:func:`~src.utils.random_utils.stable_hash64` (64-bit BLAKE2b of
``repr(token)``, salt 0), reduced modulo ``p``. Python's built-in ``hash()``
would be shorter and wrong: ``hash`` of a ``str`` is salted per interpreter
process (``PYTHONHASHSEED``), so every signature and every estimate would
change from one run to the next even with the hash parameters seeded. The
reduction modulo ``p`` changes no hash value, since ``(a x + b) mod p``
depends only on ``x mod p``; it only keeps the products smaller.

The arithmetic stays in plain Python integers on purpose. ``a x`` is the
product of two 61-bit numbers, up to 122 bits, and a numpy ``int64`` array
would wrap around silently and return wrong minima with no error. Each
token's integer is computed once per :meth:`MinHash.signature` call, and
the inner loop is a single ``min`` over ``(a * x + b) % p``.

Monte Carlo, with two-sided error
---------------------------------
MinHash is a MONTE CARLO algorithm: its work is fixed at O(k |S|) per
signature and O(k) per comparison, and the number it returns is random. The
error is TWO-SIDED: the estimate can land above or below the true ``J``.
That is unlike a Bloom filter, whose only possible mistake is a false
"present". Two cases are exact. Identical sets have identical minima under
every hash, so they always estimate exactly 1.0. Disjoint sets can share a
minimum only if two of their tokens' integers collide modulo ``p``, so in
practice they estimate 0.0.

The hash parameters are drawn as ``(a, b)`` pairs, one pair per position,
from ``make_rng(seed)``. The first 16 positions of a ``num_perm=512``
signature are therefore exactly the ``num_perm=16`` signature with the same
seed, so a sweep over ``k`` cuts one sequence of hashes at different
lengths. Two signatures are comparable only when they come from the same
seed and have the same length; :meth:`MinHash.estimate` can check the
length but not the seed, because a signature is just a list of integers.

The functions
-------------
========================== ================ ================================
Function                   Time             Space
========================== ================ ================================
MinHash(num_perm, seed)    O(k)             O(k) - the k (a, b) pairs
MinHash.signature(S)       O(k |S|)         O(k + |S|) - signature, ints
MinHash.estimate(s1, s2)   O(k)             O(1)
jaccard(A, B)              O(|A| + |B|)     O(|A| + |B|) - the two sets
shingles(text, n)          O(n L)           O(n L) - words and shingles
estimate_similarity(A, B)  O(k (|A| + |B|)) O(k + |A| + |B|)
========================== ================ ================================

Here ``k`` is ``num_perm`` and ``L`` is the length of the text.

Reference:
    Broder, A. Z. (1997). On the resemblance and containment of documents.
    Proceedings of Compression and Complexity of Sequences 1997, 21-29.

Examples:
    >>> a, b = set(range(0, 100)), set(range(50, 150))
    >>> jaccard(a, b) == 1 / 3
    True
    >>> estimate = estimate_similarity(a, b, num_perm=512, seed=7)
    >>> abs(estimate - 1 / 3) < 0.1
    True
    >>> estimate_similarity(a, a, num_perm=16, seed=7)
    1.0

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from src.utils.random_utils import make_rng, stable_hash64

__all__ = [
    "MERSENNE_61",
    "MinHash",
    "estimate_similarity",
    "jaccard",
    "shingles",
]

#: The prime modulus of every MinHash hash function: 2^61 - 1.
MERSENNE_61: int = 2**61 - 1

#: One run of letters. ``[^\W\d_]`` is a word character that is neither a
#: digit nor an underscore, which is exactly a letter (Unicode included).
_WORD_RE = re.compile(r"[^\W\d_]+")


def _check_positive_int(value: object, name: str) -> int:
    """Return ``value`` if it is an int of at least 1, otherwise raise.

    Shared by :class:`MinHash` (``num_perm``) and :func:`shingles` (``n``),
    both of which count something and are meaningless below 1.

    Args:
        value: The value to check.
        name: The parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an int. ``bool`` is rejected, because
            ``True`` is an accident, not a count.
        ValueError: If ``value`` is less than 1.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_positive_int(16, "num_perm")
        16
        >>> _check_positive_int(True, "num_perm")
        Traceback (most recent call last):
            ...
        TypeError: num_perm must be an int, got bool
        >>> _check_positive_int(0, "n")
        Traceback (most recent call last):
            ...
        ValueError: n must be at least 1, got 0
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _token_ints(tokens: Iterable[object]) -> list[int]:
    """Map every distinct token to its integer in ``[0, MERSENNE_61)``, once.

    The integer is ``stable_hash64(token, salt=0) % MERSENNE_61``, which is
    the same in every process, unlike ``hash()``. Duplicates collapse here,
    so a token that appears twice is hashed into the signature once, as a
    set requires. Doing this once per signature keeps the BLAKE2b work out
    of the ``k |S|`` inner loop.

    Args:
        tokens: Any iterable of tokens. Tokens need not be hashable; only
            their ``repr`` is used.

    Returns:
        A list of the distinct reduced token integers, in no fixed order.

    Raises:
        TypeError: If ``tokens`` is not iterable.

    Time Complexity:
        O(|S|) hashes, one per token.

    Space Complexity:
        O(|S|).

    Examples:
        >>> len(_token_ints(["a", "b", "a"]))
        2
        >>> all(0 <= x < MERSENNE_61 for x in _token_ints(range(50)))
        True
        >>> _token_ints([])
        []
    """
    return list({stable_hash64(token, salt=0) % MERSENNE_61 for token in tokens})


class MinHash:
    """A seeded family of ``num_perm`` hash functions that signs sets.

    Construction draws ``num_perm`` pairs ``(a, b)`` from ``make_rng(seed)``,
    with ``a`` uniform in ``[1, p - 1]`` and ``b`` uniform in ``[0, p - 1]``
    for ``p = MERSENNE_61``. Position ``i`` of every signature is the
    minimum of ``(a_i x + b_i) mod p`` over the set's token integers ``x``.
    Build one instance and sign every set with it: signatures from different
    seeds share no hash functions and cannot be compared.

    The randomness is all in the construction. Once the pairs are drawn,
    :meth:`signature` is a deterministic function of the set, so the same
    seed always gives the same signature. The estimate built from two
    signatures is MONTE CARLO with TWO-SIDED error: fixed O(k |S|) work, and
    an estimate that may land above or below the true Jaccard similarity,
    with standard error ``sqrt(J (1 - J) / k)``.

    Args:
        num_perm: The signature length ``k``, the number of hash functions.
            Defaults to 128.
        seed: Seed for the hash parameters. Defaults to 0, so signatures are
            reproducible. ``None`` draws an unpredictable family from the
            operating system, which is still self-consistent for one
            instance but not reproducible across runs.

    Raises:
        TypeError: If ``num_perm`` is not an int (``bool`` rejected), or if
            ``seed`` is neither an int nor None.
        ValueError: If ``num_perm`` is less than 1.

    Time Complexity:
        O(k) to draw the parameters.

    Space Complexity:
        O(k) - one ``(a, b)`` pair per position.

    Examples:
        >>> minhash = MinHash(num_perm=64, seed=1)
        >>> minhash.num_perm, minhash.seed
        (64, 1)
        >>> signature = minhash.signature({"apple", "banana", "cherry"})
        >>> len(signature)
        64
        >>> same_set = ["cherry", "apple", "cherry", "banana"]
        >>> MinHash(num_perm=64, seed=1).signature(same_set) == signature
        True
        >>> MinHash(num_perm=0)
        Traceback (most recent call last):
            ...
        ValueError: num_perm must be at least 1, got 0
    """

    def __init__(self, num_perm: int = 128, seed: int | None = 0) -> None:
        """Draw the ``num_perm`` hash parameter pairs from ``make_rng(seed)``."""
        self._num_perm = _check_positive_int(num_perm, "num_perm")
        rng = make_rng(seed)
        p = MERSENNE_61
        self._params: list[tuple[int, int]] = [
            (rng.randint(1, p - 1), rng.randint(0, p - 1)) for _ in range(num_perm)
        ]
        self._seed = seed

    @property
    def num_perm(self) -> int:
        """The signature length ``k``: how many hash functions are used."""
        return self._num_perm

    @property
    def seed(self) -> int | None:
        """The seed the hash parameters were drawn from."""
        return self._seed

    def signature(self, tokens: Iterable[object]) -> list[int]:
        """Return the MinHash signature of a set of tokens.

        Each token is mapped to its integer once, through
        :func:`~src.utils.random_utils.stable_hash64` (never ``hash()``),
        and position ``i`` of the result is ``min((a_i x + b_i) % p)`` over
        those integers. Duplicates and order do not matter: the signature
        depends only on the set of tokens. The products are plain Python
        integers, so the 122-bit ``a x`` is exact.

        Args:
            tokens: An iterable of tokens, treated as a set. Shingles from
                :func:`shingles`, words, or integers all work.

        Returns:
            A list of ``num_perm`` integers, each in ``[0, MERSENNE_61)``.

        Raises:
            ValueError: If ``tokens`` is empty. An empty set has no minimum
                under any hash, so it has no signature.
            TypeError: If ``tokens`` is not iterable.

        Time Complexity:
            O(k |S|) - one hash evaluation per position per distinct token,
            plus O(|S|) BLAKE2b hashes.

        Space Complexity:
            O(k + |S|) - the signature and the token integers.

        Examples:
            >>> tokens = shingles("the cat sat on the mat")
            >>> long_sig = MinHash(num_perm=512, seed=5).signature(tokens)
            >>> short_sig = MinHash(num_perm=16, seed=5).signature(tokens)
            >>> long_sig[:16] == short_sig
            True
            >>> all(0 <= value < MERSENNE_61 for value in long_sig)
            True
            >>> MinHash(num_perm=8).signature(set())
            Traceback (most recent call last):
                ...
            ValueError: cannot sign an empty set: it has no minimum under any hash
        """
        xs = _token_ints(tokens)
        if not xs:
            raise ValueError(
                "cannot sign an empty set: it has no minimum under any hash"
            )
        p = MERSENNE_61
        return [min([(a * x + b) % p for x in xs]) for a, b in self._params]

    @staticmethod
    def estimate(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        """Estimate Jaccard similarity as the fraction of equal positions.

        Position ``i`` agrees with probability about ``J`` (see the module
        docstring), so the fraction of agreeing positions is an unbiased
        estimate of ``J``: MONTE CARLO with TWO-SIDED error, standard error
        ``sqrt(J (1 - J) / k)``. Both signatures must come from the same
        :class:`MinHash` seed; only their lengths can be checked here.

        Args:
            sig_a: A signature from :meth:`signature`.
            sig_b: A signature of the same length, from the same seed.

        Returns:
            A float in ``[0, 1]``.

        Raises:
            ValueError: If the signatures have different lengths, or are
                empty.

        Time Complexity:
            O(k).

        Space Complexity:
            O(1).

        Examples:
            >>> MinHash.estimate([1, 2, 3, 4], [1, 9, 3, 9])
            0.5
            >>> MinHash.estimate([1, 2, 3], [1, 2])
            Traceback (most recent call last):
                ...
            ValueError: signatures must have the same length, got 3 and 2
        """
        if len(sig_a) != len(sig_b):
            raise ValueError(
                "signatures must have the same length, got "
                f"{len(sig_a)} and {len(sig_b)}"
            )
        if not sig_a:
            raise ValueError("signatures must not be empty")
        matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return matches / len(sig_a)


def jaccard(a: Iterable[object], b: Iterable[object]) -> float:
    """Return the exact Jaccard similarity ``|A & B| / |A | B|``.

    The ground truth that :class:`MinHash` estimates. Both inputs are
    turned into sets first, so duplicates do not count twice. Two empty
    sets are defined to have similarity 1.0: they are identical, and the
    ratio would otherwise be 0 / 0.

    Args:
        a: An iterable of hashable tokens.
        b: Another iterable of hashable tokens.

    Returns:
        A float in ``[0, 1]``.

    Raises:
        TypeError: If either input is not iterable or holds an unhashable
            token.

    Time Complexity:
        O(|A| + |B|) expected, for building and intersecting the sets.

    Space Complexity:
        O(|A| + |B|) - the two sets.

    Examples:
        >>> jaccard(range(0, 100), range(50, 150)) == 1 / 3
        True
        >>> jaccard({"x", "y"}, {"x", "y"})
        1.0
        >>> jaccard({1, 2}, {3, 4})
        0.0
        >>> jaccard([], set())
        1.0
    """
    set_a, set_b = set(a), set(b)
    shared = len(set_a & set_b)
    union = len(set_a) + len(set_b) - shared
    if union == 0:
        return 1.0
    return shared / union


def shingles(text: str, n: int = 3) -> set[str]:
    """Return the set of word ``n``-grams (shingles) of ``text``.

    The text is lowercased and split into words on every non-letter, so
    punctuation, digits and apostrophes all separate words ("don't" becomes
    "don" and "t"). Each run of ``n`` consecutive words, joined by single
    spaces, is one shingle. Text with fewer than ``n`` words gives a single
    shingle of all its words, so a short document still has a signature;
    text with no words at all gives the empty set.

    Args:
        text: The document.
        n: Words per shingle. Defaults to 3.

    Returns:
        A set of strings.

    Raises:
        TypeError: If ``text`` is not a str, or ``n`` is not an int
            (``bool`` rejected).
        ValueError: If ``n`` is less than 1.

    Time Complexity:
        O(n L) for a text of length ``L``: each word is copied into at most
        ``n`` shingles.

    Space Complexity:
        O(n L) - the words and the shingles.

    Examples:
        >>> expected = {"the cat sat", "cat sat on", "sat on the", "on the mat"}
        >>> shingles("the cat sat on the mat", 3) == expected
        True
        >>> shingles("Hello, WORLD!")
        {'hello world'}
        >>> sorted(shingles("Don't panic", n=2))
        ['don t', 't panic']
        >>> shingles("  42 ... ")
        set()
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    _check_positive_int(n, "n")
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def estimate_similarity(
    a: Iterable[object],
    b: Iterable[object],
    num_perm: int = 128,
    seed: int | None = 0,
) -> float:
    """Estimate the Jaccard similarity of two token sets with MinHash.

    Builds one :class:`MinHash` of ``num_perm`` hashes from ``seed``, signs
    both sets with it, and compares the signatures. MONTE CARLO with
    TWO-SIDED error: the work is fixed at O(k (|A| + |B|)), and the result
    may be above or below :func:`jaccard` with standard error
    ``sqrt(J (1 - J) / k)``. Identical sets always give exactly 1.0.

    Args:
        a: An iterable of tokens, treated as a set. Must not be empty.
        b: Another iterable of tokens. Must not be empty.
        num_perm: The signature length ``k``. Defaults to 128.
        seed: Seed for the hash parameters. Defaults to 0.

    Returns:
        A float in ``[0, 1]``, a multiple of ``1 / num_perm``.

    Raises:
        TypeError: If ``num_perm`` is not an int or ``seed`` is neither an
            int nor None.
        ValueError: If ``num_perm`` is less than 1, or either set is empty.

    Time Complexity:
        O(k (|A| + |B|)).

    Space Complexity:
        O(k + |A| + |B|).

    Examples:
        >>> a, b = set(range(0, 100)), set(range(50, 150))
        >>> abs(estimate_similarity(a, b, num_perm=512, seed=3) - 1 / 3) < 0.1
        True
        >>> estimate_similarity(a, a, num_perm=16, seed=3)
        1.0
        >>> estimate_similarity(range(0, 100), range(100, 200), num_perm=64, seed=3)
        0.0
        >>> estimate_similarity(a, [], seed=3)
        Traceback (most recent call last):
            ...
        ValueError: cannot sign an empty set: it has no minimum under any hash
    """
    minhash = MinHash(num_perm=num_perm, seed=seed)
    return MinHash.estimate(minhash.signature(a), minhash.signature(b))
