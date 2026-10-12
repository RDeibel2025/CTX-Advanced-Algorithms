"""Seeded randomness and stable hashing for the Week 7 randomized algorithms.

Every randomized algorithm in ``src.randomized`` draws its random numbers
from a :class:`random.Random` *instance*, never from the module-level
functions in :mod:`random`. The module-level functions share one hidden
global state, so a call anywhere in the program (a library, a plot, another
test) shifts every later draw and a "seeded" experiment silently stops being
reproducible. An instance owns its state: the same seed gives the same
sequence no matter what else runs. :func:`make_rng` is the one place those
instances are built.

Two seeding policies, on purpose
--------------------------------
* ``make_rng(seed)`` with an ``int`` gives a reproducible generator. Every
  experiment, benchmark and doctest passes one, usually :data:`DEFAULT_SEED`
  or a value from :func:`derive_seed`.
* ``make_rng(None)`` gives a generator seeded from the operating system,
  which is deliberately *unpredictable*. That is the right default for a
  randomized algorithm used for real: if randomized quicksort fell back to a
  fixed seed, an adversary who read the code could replay its pivot choices
  and build an input that forces the O(n^2) case. Randomization only defeats
  a worst-case input when the input cannot depend on the coin flips.

Stable hashing, not ``hash()``
------------------------------
The Bloom filter and MinHash need hash values that are the same on every
run. The built-in :func:`hash` is not: ``str`` and ``bytes`` hashes are
salted per process (``PYTHONHASHSEED``), so a Bloom filter built with it
would answer differently each time the program starts, even with every RNG
seeded. :func:`stable_hash64` and :func:`derive_seed` use
:func:`hashlib.blake2b` over ``repr(...)`` instead, which depends only on
the bytes it is given.

==================== ======================================================
Function             Purpose
==================== ======================================================
make_rng             The one constructor for ``random.Random`` instances
derive_seed          Independent child seeds from a master seed and labels
seeded_shuffle       A shuffled copy, leaving the input untouched
random_int_list      ``n`` uniform integers in ``[low, high]``
random_distinct_list A seeded permutation of ``range(n)``
stable_hash64        A 64-bit hash that is identical across processes
==================== ======================================================

Examples:
    >>> rng = make_rng(DEFAULT_SEED)
    >>> rng.random()
    0.6394267984578837
    >>> random_distinct_list(5, 42) == random_distinct_list(5, 42)
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Hashable, Iterable, List, Optional, TypeVar

__all__ = [
    "DEFAULT_SEED",
    "derive_seed",
    "make_rng",
    "random_distinct_list",
    "random_int_list",
    "seeded_shuffle",
    "stable_hash64",
]

#: The seed every experiment and test uses unless it has a reason not to.
DEFAULT_SEED: int = 42

#: blake2b accepts a salt of at most 16 bytes, so salts live in [0, 2**128).
_SALT_LIMIT: int = 2**128

_T = TypeVar("_T")


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int`` (not a ``bool``), else raise.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``. ``bool`` is rejected,
            because ``True`` as a seed or size is an accident, not a choice.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(7, "seed")
        7
        >>> _require_int(True, "seed")
        Traceback (most recent call last):
            ...
        TypeError: seed must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_size(n: Any) -> int:
    """Return ``n`` if it is a non-negative ``int``, else raise.

    Args:
        n: A requested list length.

    Returns:
        ``n`` unchanged.

    Raises:
        TypeError: If ``n`` is not an ``int`` (``bool`` rejected).
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_size(0)
        0
        >>> _require_size(-1)
        Traceback (most recent call last):
            ...
        ValueError: n must be non-negative, got -1
    """
    _require_int(n, "n")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return n


def make_rng(seed: Optional[int] = None) -> random.Random:
    """Build a private :class:`random.Random` generator.

    With an ``int`` seed the generator is reproducible: two generators made
    from the same seed produce the same sequence, whatever else the program
    does with randomness. With ``seed=None`` the generator is seeded from
    the operating system and is deliberately unpredictable. That default
    protects a randomized algorithm in real use: with a fixed default seed an
    adversary who reads the code could predict every pivot and build a bad
    input for it. Experiments and tests therefore always pass an explicit
    seed, and only real use relies on the unpredictable default.

    Args:
        seed: An ``int`` for a reproducible generator, or ``None`` for one
            seeded from the operating system.

    Returns:
        A new :class:`random.Random` instance that shares no state with the
        module-level :mod:`random` functions.

    Raises:
        TypeError: If ``seed`` is neither ``None`` nor an ``int``. ``bool`` is
            rejected.

    Time Complexity:
        O(1) - one generator initialisation.

    Space Complexity:
        O(1) - the Mersenne Twister state is a fixed 624 words.

    Examples:
        >>> a, b = make_rng(42), make_rng(42)
        >>> [a.randint(1, 6) for _ in range(5)] == [b.randint(1, 6) for _ in range(5)]
        True
        >>> make_rng(42).random()
        0.6394267984578837
        >>> isinstance(make_rng(), random.Random)
        True
        >>> make_rng("42")
        Traceback (most recent call last):
            ...
        TypeError: seed must be an int, got str
        >>> make_rng(True)
        Traceback (most recent call last):
            ...
        TypeError: seed must be an int, got bool
    """
    if seed is None:
        return random.Random()
    return random.Random(_require_int(seed, "seed"))


def derive_seed(master_seed: int, *labels: Hashable) -> int:
    """Derive a child seed from a master seed and a sequence of labels.

    An experiment with many trials needs a different seed per trial that is
    still fully determined by one master seed. Seeding trial ``i`` with
    ``master + i`` makes neighbouring experiments share almost all of their
    seeds; hashing ``(master_seed, *labels)`` instead gives unrelated seeds
    for different labels and the same seed for the same labels, in every
    process. The hash is ``blake2b`` over
    ``repr((master_seed,) + labels).encode("utf-8")`` with an 8-byte digest
    read little-endian, never the salted built-in :func:`hash`.

    Args:
        master_seed: The experiment's master seed.
        *labels: Anything with a stable ``repr`` that names the child, for
            example ``"quicksort", n, trial``.

    Returns:
        An ``int`` in ``[0, 2**64)``, suitable for :func:`make_rng`.

    Raises:
        TypeError: If ``master_seed`` is not an ``int`` (``bool`` rejected).

    Time Complexity:
        O(L) where L is the length of the ``repr`` being hashed.

    Space Complexity:
        O(L) - the encoded ``repr``.

    Examples:
        >>> derive_seed(42, "trial", 0)
        10018139530472409056
        >>> derive_seed(42, "trial", 0) == derive_seed(42, "trial", 0)
        True
        >>> derive_seed(42, "trial", 0) == derive_seed(42, "trial", 1)
        False
        >>> 0 <= derive_seed(42) < 2**64
        True
    """
    _require_int(master_seed, "master_seed")
    payload = repr((master_seed,) + labels).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def seeded_shuffle(seq: Iterable[_T], seed: Optional[int]) -> List[_T]:
    """Return a shuffled copy of ``seq``, leaving ``seq`` itself untouched.

    The copy is shuffled in place with Fisher-Yates
    (:meth:`random.Random.shuffle`) on a generator from :func:`make_rng`, so
    every permutation is equally likely and the same seed gives the same
    permutation.

    Args:
        seq: Any finite iterable. It is read once and never modified.
        seed: Passed to :func:`make_rng`. ``None`` means an unpredictable
            shuffle.

    Returns:
        A new list holding the items of ``seq`` in shuffled order.

    Raises:
        TypeError: If ``seq`` is not iterable, or ``seed`` is not an ``int``
            or ``None``.

    Time Complexity:
        O(n) - one copy and one Fisher-Yates pass.

    Space Complexity:
        O(n) - the copy.

    Examples:
        >>> items = ["a", "b", "c", "d", "e"]
        >>> seeded_shuffle(items, 7)
        ['e', 'a', 'd', 'b', 'c']
        >>> items
        ['a', 'b', 'c', 'd', 'e']
        >>> seeded_shuffle(items, 7) == seeded_shuffle(items, 7)
        True
        >>> seeded_shuffle([], 7)
        []
    """
    rng = make_rng(seed)
    items = list(seq)
    rng.shuffle(items)
    return items


def random_int_list(
    n: int, seed: Optional[int], low: int = 0, high: int = 1_000_000
) -> List[int]:
    """Return ``n`` integers drawn uniformly and independently from ``[low, high]``.

    Both ends are inclusive, as with :meth:`random.Random.randint`. Values
    may repeat; use :func:`random_distinct_list` when they must not.

    Args:
        n: How many integers to draw. Must be non-negative.
        seed: Passed to :func:`make_rng`.
        low: The smallest value that can be drawn.
        high: The largest value that can be drawn.

    Returns:
        A new list of ``n`` ints.

    Raises:
        TypeError: If ``n``, ``low`` or ``high`` is not an ``int`` (``bool``
            rejected), or ``seed`` is not an ``int`` or ``None``.
        ValueError: If ``n`` is negative or ``low > high``.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the result.

    Examples:
        >>> random_int_list(5, 42)
        [670487, 116739, 26225, 777572, 288389]
        >>> random_int_list(8, 1, low=1, high=6)
        [2, 5, 1, 3, 1, 4, 4, 4]
        >>> random_int_list(0, 42)
        []
        >>> random_int_list(3, 42, low=5, high=1)
        Traceback (most recent call last):
            ...
        ValueError: low must be <= high, got low=5, high=1
    """
    _require_size(n)
    _require_int(low, "low")
    _require_int(high, "high")
    if low > high:
        raise ValueError(f"low must be <= high, got low={low}, high={high}")
    rng = make_rng(seed)
    return [rng.randint(low, high) for _ in range(n)]


def random_distinct_list(n: int, seed: Optional[int]) -> List[int]:
    """Return a uniformly random permutation of ``range(n)``.

    The standard input for sorting and selection experiments: ``n`` distinct
    keys in random order, so every comparison has a strict answer and the
    textbook expected-case analyses apply exactly.

    Args:
        n: The length of the permutation. Must be non-negative.
        seed: Passed to :func:`make_rng`.

    Returns:
        A new list containing each of ``0 .. n - 1`` exactly once.

    Raises:
        TypeError: If ``n`` is not an ``int`` (``bool`` rejected), or ``seed``
            is not an ``int`` or ``None``.
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the result.

    Examples:
        >>> perm = random_distinct_list(10, 42)
        >>> perm
        [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
        >>> sorted(perm) == list(range(10))
        True
        >>> random_distinct_list(0, 42)
        []
    """
    return seeded_shuffle(range(_require_size(n)), seed)


def stable_hash64(item: Any, salt: int = 0) -> int:
    """Hash ``item`` to a 64-bit integer that is the same in every process.

    Computes ``blake2b(repr(item).encode("utf-8"), digest_size=8,
    salt=salt.to_bytes(16, "little"))`` and reads the digest little-endian.
    Unlike the built-in :func:`hash`, whose ``str`` results change with
    ``PYTHONHASHSEED`` on every run, this depends only on ``repr(item)`` and
    ``salt``. Different salts act as independent hash functions over the same
    items. Items are identified by their ``repr``, so ``1`` and ``"1"`` hash
    differently, as they should.

    Args:
        item: Any value with a deterministic ``repr`` (str, int, tuple, ...).
        salt: Selects the hash function; must satisfy
            ``0 <= salt < 2**128``, because blake2b takes a 16-byte salt.

    Returns:
        An ``int`` in ``[0, 2**64)``.

    Raises:
        TypeError: If ``salt`` is not an ``int`` (``bool`` rejected).
        ValueError: If ``salt`` is negative or at least ``2**128``.

    Time Complexity:
        O(L) where L is the length of ``repr(item)``.

    Space Complexity:
        O(L) - the encoded ``repr``.

    Examples:
        >>> stable_hash64("apple")
        18272638738543383483
        >>> stable_hash64("apple") == stable_hash64("apple")
        True
        >>> stable_hash64("apple", salt=1) == stable_hash64("apple")
        False
        >>> 0 <= stable_hash64(("tuple", 3)) < 2**64
        True
        >>> stable_hash64("apple", salt=-1)
        Traceback (most recent call last):
            ...
        ValueError: salt must satisfy 0 <= salt < 2**128, got -1
    """
    _require_int(salt, "salt")
    if not 0 <= salt < _SALT_LIMIT:
        raise ValueError(f"salt must satisfy 0 <= salt < 2**128, got {salt}")
    digest = hashlib.blake2b(
        repr(item).encode("utf-8"),
        digest_size=8,
        salt=salt.to_bytes(16, "little"),
    ).digest()
    return int.from_bytes(digest, "little")
