"""Monte Carlo primality testing: Fermat, Miller-Rabin, and why the bases are random.

Deciding whether a large ``n`` is prime by trial division costs about
``sqrt(n)`` divisions, which is exponential in the number of digits. The two
randomized tests here cost ``O(k log^3 n)`` bit operations instead: each of
``k`` rounds picks a base ``a`` and runs one modular exponentiation with the
built-in ``pow(a, e, n)``. They buy that speed by accepting a small,
quantifiable chance of a wrong answer, which makes them MONTE CARLO
algorithms: the work is bounded, and the answer is correct with high
probability.

One-sided error
---------------
Both tests can only be wrong in one direction.

* **"Composite" (False) is always correct.** It is returned only after a
  base ``a`` was found that a prime could never produce: ``a^(n-1) != 1
  (mod n)`` breaks Fermat's little theorem, and a square root of 1 other
  than ``+1`` or ``-1`` cannot exist modulo a prime. Such a base is a
  *witness*, and it is a proof that ``n`` is composite.
* **"Prime" (True) can be wrong.** It means only that ``k`` bases were tried
  and none was a witness. A composite ``n`` with many non-witnesses (called
  *liars*) can survive every round.

A prime ``n`` has no witnesses at all, so a prime is never reported as
composite. The only possible mistake is a composite reported as prime, and
its probability is at most ``(liars / bases)^k`` because the ``k`` bases are
drawn independently and uniformly from ``[2, n - 2]``, with replacement.

Miller-Rabin: at most 4^-k
--------------------------
Write ``n - 1 = 2^s * d`` with ``d`` odd. The Miller-Rabin round computes
``x = a^d mod n`` and squares it up to ``s - 1`` times. For a prime, the
sequence either starts at 1 or reaches ``n - 1`` before it reaches 1. If it
does neither, ``a`` is a witness. Rabin (1980), and independently Monier
(1980), proved that for every odd composite ``n > 9`` at most a quarter of
the bases in ``[1, n - 1]`` are strong liars. So one round is fooled with
probability at most 1/4, and ``k`` independent rounds with probability at
most ``4^-k``: about ``10^-12`` at ``k = 20``. The bound holds for *every*
composite, including the Carmichael numbers below; on 561 only 8 of the 558
bases in ``[2, 559]`` lie, about 1.4 percent.

Fermat: no bound at all
-----------------------
The Fermat test checks only ``a^(n-1) = 1 (mod n)``. For a composite that is
not a Carmichael number the liars form a proper subgroup of the units modulo
``n``, so at most half the bases lie. A *Carmichael number* breaks that
argument: it satisfies ``a^(n-1) = 1 (mod n)`` for **every** ``a`` coprime
to ``n`` (Korselt's criterion: ``n`` is square-free and ``p - 1`` divides
``n - 1`` for every prime ``p`` dividing ``n``). The only Fermat witnesses
are then the bases that share a factor with ``n``, and when ``n``'s prime
factors are large those are a vanishing fraction of all bases. There are
infinitely many Carmichael numbers (Alford, Granville and Pomerance, 1994),
so the per-round error is not bounded by any constant below 1: no ``k``
fixed in advance makes the Fermat test reliable. On 561 = 3 x 11 x 17, 318
of the 558 bases lie, so each round is fooled 57 percent of the time.

Random bases against an adversary
---------------------------------
Miller (1976) gave the same witness test with a *fixed* list of bases, which
is deterministic and correct if the extended Riemann hypothesis holds. Rabin
(1980) made one change: choose the bases at random, at run time. That turns
an unproven hypothesis into a probability bound that is proved, and it
changes who controls the error.

The 4^-k bound is a probability over the algorithm's own coin flips, not
over the inputs, so it holds for each ``n`` separately, including an ``n``
chosen by someone trying to break the test. With a fixed base list there
are no coin flips: the test is a deterministic function, and whoever knows
the list can search for a composite that passes it. The smallest composite
that is a strong pseudoprime to all of 2, 3, 5 and 7 is 3,215,031,751 =
151 x 751 x 28,351 (OEIS A014233), and
:func:`miller_rabin_fixed_bases` with ``[2, 3, 5, 7]`` calls it prime. With
random bases the adversary must commit to ``n`` before the bases exist, and
then each round catches any composite with probability at least 3/4. That
is also why :func:`~src.utils.random_utils.make_rng` seeds from the
operating system when no seed is given: a fixed default seed would turn the
"random" bases back into fixed ones that can be read from the source code.
Experiments and doctests pass an explicit seed so their results repeat.

The functions
-------------
======================== ======================= ==================================
Function                 Time                    Error
======================== ======================= ==================================
fermat_test              O(k log^3 n)            one-sided, unbounded on Carmichael
miller_rabin             O(k log^3 n)            one-sided, at most 4^-k
miller_rabin_fixed_bases O(len(bases) log^3 n)   one-sided, beatable by a chosen n
is_prime_deterministic   O(sqrt(n)) divisions    none
count_liars              O(n log^3 n)            none (exhaustive, small n only)
======================== ======================= ==================================

All three tests share one front end: ``n < 2`` is not prime, 2 and 3 are
prime, and an even ``n > 2`` is composite. Every ``n`` below 5 is therefore
settled before a base is chosen, so the range ``[2, n - 2]`` is never empty.

References:
    Rabin, M. O. (1980). Probabilistic algorithm for testing primality.
    Journal of Number Theory, 12(1), 128-138.

    Miller, G. L. (1976). Riemann's hypothesis and tests for primality.
    Journal of Computer and System Sciences, 13(3), 300-317.

    Cormen, Leiserson, Rivest and Stein, Introduction to Algorithms, 4th
    edition, section 31.8.

Examples:
    The small truth table, identical for all three tests:

    >>> table = (2, 3, 5, 7919, 1, 9, 341)
    >>> [fermat_test(n, k=20, seed=7) for n in table]
    [True, True, True, True, False, False, False]
    >>> [miller_rabin(n, k=20, seed=7) for n in table]
    [True, True, True, True, False, False, False]
    >>> [is_prime_deterministic(n) for n in table]
    [True, True, True, True, False, False, False]

    The Carmichael number 561 fools Fermat far more often than Miller-Rabin:

    >>> count_liars(561, "fermat"), count_liars(561, "strong")
    (318, 8)

    Fixed bases can be attacked; random ones cannot:

    >>> miller_rabin_fixed_bases(3215031751, [2, 3, 5, 7])
    True
    >>> 151 * 751 * 28351
    3215031751
    >>> miller_rabin(3215031751, k=20, seed=1)
    False

    A 61-bit Mersenne prime, far beyond trial division in a doctest:

    >>> miller_rabin(2**61 - 1, k=20, seed=1)
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, List, Optional, Tuple

from src.utils.random_utils import make_rng

__all__ = [
    "count_liars",
    "fermat_test",
    "is_fermat_liar",
    "is_prime_deterministic",
    "is_strong_liar",
    "miller_rabin",
    "miller_rabin_fixed_bases",
]

#: The two kinds of liar :func:`count_liars` can count.
_LIAR_KINDS = ("fermat", "strong")


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int``, otherwise raise ``TypeError``.

    ``bool`` is rejected even though it subclasses ``int``: ``True`` passed
    as ``n`` or ``k`` is a bug in the caller, not a number to test.

    Args:
        value: The argument to check.
        name: Its parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``, or is a ``bool``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(7, "n")
        7
        >>> _require_int(True, "n")
        Traceback (most recent call last):
            ...
        TypeError: n must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _resolve_rng(
    k: Any, seed: Optional[int], rng: Optional[random.Random]
) -> random.Random:
    """Validate the round count and return the generator to draw bases from.

    ``rng`` takes precedence over ``seed``, so a caller running many trials
    can pass one generator and let each call continue its stream. Otherwise
    a fresh generator comes from :func:`~src.utils.random_utils.make_rng`,
    which seeds from the operating system when ``seed`` is None.

    Args:
        k: The number of rounds requested.
        seed: Seed for ``make_rng`` when ``rng`` is None.
        rng: An existing generator, or None.

    Returns:
        The ``random.Random`` instance to use.

    Raises:
        TypeError: If ``k`` is not an int, ``rng`` is not a
            ``random.Random``, or ``seed`` is not an int or None.
        ValueError: If ``k < 1``; zero rounds would answer "prime" with no
            evidence at all.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _resolve_rng(5, 3, None).random() == make_rng(3).random()
        True
        >>> _resolve_rng(0, 3, None)
        Traceback (most recent call last):
            ...
        ValueError: k must be at least 1, got 0
    """
    _require_int(k, "k")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if rng is not None:
        if not isinstance(rng, random.Random):
            raise TypeError(
                f"rng must be a random.Random, got {type(rng).__name__}"
            )
        return rng
    return make_rng(seed)


def _settle_small(n: int) -> Optional[bool]:
    """Answer the cases the shared front end decides without any base.

    ``n < 2`` is not prime, 2 and 3 are prime, and an even ``n > 2`` is
    composite. Anything left is odd and at least 5, so ``[2, n - 2]`` holds
    at least two bases.

    Args:
        n: The integer being tested.

    Returns:
        True or False when the front end decides ``n``, None when a real
        test is needed.

    Raises:
        Nothing; ``n`` is already known to be an int.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> [_settle_small(n) for n in (-7, 0, 1, 2, 3, 4, 5, 100)]
        [False, False, False, True, True, False, None, False]
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    return None


def _split_power_of_two(n: int) -> Tuple[int, int]:
    """Write ``n - 1 = 2^s * d`` with ``d`` odd and return ``(s, d)``.

    Args:
        n: An odd integer, at least 3.

    Returns:
        The pair ``(s, d)``.

    Raises:
        Nothing; callers pass only odd ``n >= 3``.

    Time Complexity:
        O(log n) - one halving per factor of 2.

    Space Complexity:
        O(1).

    Examples:
        >>> _split_power_of_two(561)
        (4, 35)
        >>> _split_power_of_two(13)
        (2, 3)
    """
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return s, d


def _is_witness(a: int, n: int, s: int, d: int) -> bool:
    """Return True when ``a`` proves the odd number ``n`` composite.

    The standard Miller-Rabin round. ``x = a^d mod n`` is squared up to
    ``s - 1`` times. A prime either starts the sequence at 1 or reaches
    ``n - 1`` (that is, -1) before 1, because the only square roots of 1
    modulo a prime are 1 and -1. A sequence that does neither has found a
    non-trivial square root of 1, or ended with ``a^(n-1) != 1``; either is
    impossible for a prime.

    Args:
        a: The base, already reduced into ``[1, n - 1]``.
        n: An odd integer, at least 3.
        s: The power of 2 in ``n - 1``.
        d: The odd part of ``n - 1``.

    Returns:
        True if ``a`` is a witness (``n`` is certainly composite), False if
        ``a`` is consistent with ``n`` being prime.

    Raises:
        Nothing.

    Time Complexity:
        O(log n) modular multiplications, so O(log^3 n) bit operations.

    Space Complexity:
        O(1) numbers of O(log n) bits.

    Examples:
        >>> _is_witness(2, 561, 4, 35)
        True
        >>> _is_witness(2, 13, 2, 3)
        False
    """
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return False
        if x == 1:
            # A square root of 1 other than +1 or -1: n cannot be prime.
            return True
    return True


def _require_base(n: int, a: Any) -> int:
    """Check that ``a`` is an int base in ``[1, n - 1]`` and return it.

    Args:
        n: The modulus.
        a: The proposed base.

    Returns:
        ``a`` unchanged.

    Raises:
        TypeError: If ``a`` is not an int (``bool`` rejected).
        ValueError: If ``a`` lies outside ``[1, n - 1]``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_base(561, 2)
        2
        >>> _require_base(561, 561)
        Traceback (most recent call last):
            ...
        ValueError: a must satisfy 1 <= a <= n - 1 = 560, got 561
    """
    _require_int(a, "a")
    if not 1 <= a <= n - 1:
        raise ValueError(f"a must satisfy 1 <= a <= n - 1 = {n - 1}, got {a}")
    return a


def _require_odd_modulus(n: Any) -> int:
    """Check that ``n`` is an odd int of at least 3 and return it.

    The Miller-Rabin round splits ``n - 1`` into ``2^s * d``, which needs
    ``n`` odd; for even ``n`` the question is settled by dividing by 2.

    Args:
        n: The proposed modulus.

    Returns:
        ``n`` unchanged.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected).
        ValueError: If ``n`` is even or below 3.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_odd_modulus(561)
        561
        >>> _require_odd_modulus(10)
        Traceback (most recent call last):
            ...
        ValueError: n must be odd and at least 3, got 10
    """
    _require_int(n, "n")
    if n < 3 or n % 2 == 0:
        raise ValueError(f"n must be odd and at least 3, got {n}")
    return n


def fermat_test(
    n: int,
    k: int = 10,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> bool:
    """Test ``n`` for primality with ``k`` rounds of the Fermat test.

    After the shared front end, each round draws a base ``a`` uniformly from
    ``[2, n - 2]`` (with replacement) and checks Fermat's little theorem,
    ``a^(n-1) = 1 (mod n)``. A failure proves ``n`` composite at once.

    MONTE CARLO with ONE-SIDED error. False ("composite") is always
    correct. True ("prime") can be wrong, with probability ``(L / (n - 3))^k``
    where ``L`` is the number of Fermat liars in ``[2, n - 2]``. For a
    Carmichael number every base coprime to ``n`` is a liar, so that
    fraction can be arbitrarily close to 1 and there is no bound like
    Miller-Rabin's ``4^-k``. See the module docstring.

    Args:
        n: The integer to test. Any int; values below 2 are not prime.
        k: The number of rounds, at least 1.
        seed: Seed for :func:`~src.utils.random_utils.make_rng` when ``rng``
            is None. None (the default) seeds from the operating system, so
            the bases cannot be predicted; experiments pass an int.
        rng: A ``random.Random`` to draw the bases from. It takes precedence
            over ``seed``.

    Returns:
        False if ``n`` is certainly composite, True if ``n`` is probably
        prime.

    Raises:
        TypeError: If ``n`` or ``k`` is not an int (``bool`` rejected),
            ``seed`` is not an int or None, or ``rng`` is not a
            ``random.Random``.
        ValueError: If ``k < 1``.

    Time Complexity:
        O(k log^3 n) - ``k`` modular exponentiations of O(log n)
        multiplications on O(log n)-bit numbers. O(1) when the front end
        decides ``n``.

    Space Complexity:
        O(1) numbers of O(log n) bits.

    Examples:
        >>> [fermat_test(n, k=20, seed=7) for n in (2, 3, 5, 7919, 1, 9, 341)]
        [True, True, True, True, False, False, False]

        The Carmichael number 118,901,521 = 271 x 541 x 811 has only about
        0.7 percent Fermat witnesses, so twenty rounds usually miss them
        all; Miller-Rabin is not fooled:

        >>> fermat_test(118901521, k=20, seed=1)
        True
        >>> miller_rabin(118901521, k=20, seed=1)
        False

        The same seed gives the same answer:

        >>> fermat_test(561, k=1, seed=3) == fermat_test(561, k=1, seed=3)
        True
    """
    _require_int(n, "n")
    generator = _resolve_rng(k, seed, rng)
    settled = _settle_small(n)
    if settled is not None:
        return settled
    exponent = n - 1
    for _ in range(k):
        a = generator.randint(2, n - 2)
        if pow(a, exponent, n) != 1:
            return False
    return True


def miller_rabin(
    n: int,
    k: int = 10,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> bool:
    """Test ``n`` for primality with ``k`` rounds of Miller-Rabin.

    After the shared front end, ``n - 1`` is written as ``2^s * d`` with
    ``d`` odd, and each round draws a base ``a`` uniformly from
    ``[2, n - 2]`` (with replacement) and runs the standard witness loop:
    ``a^d mod n`` and up to ``s - 1`` squarings must either start at 1 or
    reach ``n - 1``. A base that does neither proves ``n`` composite.

    MONTE CARLO with ONE-SIDED error. False ("composite") is always
    correct. True ("prime") can be wrong only for a composite ``n``, and at
    most a quarter of the bases are strong liars for any odd composite
    (Rabin 1980), so ``k`` rounds are fooled with probability at most
    ``4^-k``. The bound is over the random bases, so it holds for every
    ``n``, including one chosen by an adversary.

    Args:
        n: The integer to test. Any int; values below 2 are not prime.
        k: The number of rounds, at least 1.
        seed: Seed for :func:`~src.utils.random_utils.make_rng` when ``rng``
            is None. None (the default) seeds from the operating system, so
            the bases cannot be predicted; experiments pass an int.
        rng: A ``random.Random`` to draw the bases from. It takes precedence
            over ``seed``.

    Returns:
        False if ``n`` is certainly composite, True if ``n`` is probably
        prime (wrong with probability at most ``4^-k``).

    Raises:
        TypeError: If ``n`` or ``k`` is not an int (``bool`` rejected),
            ``seed`` is not an int or None, or ``rng`` is not a
            ``random.Random``.
        ValueError: If ``k < 1``.

    Time Complexity:
        O(k log^3 n) - each round is one modular exponentiation plus at
        most ``s - 1 < log2 n`` squarings. O(1) when the front end decides
        ``n``.

    Space Complexity:
        O(1) numbers of O(log n) bits.

    Examples:
        >>> [miller_rabin(n, k=20, seed=7) for n in (2, 3, 5, 7919, 1, 9, 341)]
        [True, True, True, True, False, False, False]
        >>> miller_rabin(2**61 - 1, k=20, seed=1)
        True

        The composite that defeats the fixed bases 2, 3, 5 and 7 is caught
        by random ones:

        >>> miller_rabin(3215031751, k=20, seed=1)
        False

        A shared generator continues its stream across calls:

        >>> shared = make_rng(5)
        >>> [miller_rabin(n, k=5, rng=shared) for n in (561, 1105, 1729)]
        [False, False, False]
    """
    _require_int(n, "n")
    generator = _resolve_rng(k, seed, rng)
    settled = _settle_small(n)
    if settled is not None:
        return settled
    s, d = _split_power_of_two(n)
    for _ in range(k):
        a = generator.randint(2, n - 2)
        if _is_witness(a, n, s, d):
            return False
    return True


def miller_rabin_fixed_bases(n: int, bases: Iterable[int]) -> bool:
    """Run the Miller-Rabin witness test with caller-chosen bases.

    The same front end and witness loop as :func:`miller_rabin`, but the
    bases come from the caller instead of a generator. It exists for the
    adversarial demonstration: a fixed list is a deterministic test, and
    whoever knows the list can pick a composite that passes it. Each base
    is reduced modulo ``n`` first, and a base divisible by ``n`` is skipped
    because it says nothing (so ``[2, 3, 5, 7]`` still works for ``n = 7``).

    No randomness is involved, but the error is still ONE-SIDED: False is
    always correct (a witness was found), while True can be wrong for any
    composite that is a strong pseudoprime to every base in the list.
    Such lists are exact below a known limit; ``[2, 3, 5, 7]`` is correct
    for every ``n < 3,215,031,751`` and wrong at that number.

    Args:
        n: The integer to test. Any int; values below 2 are not prime.
        bases: A non-empty iterable of int bases. Any int is accepted and
            reduced modulo ``n``.

    Returns:
        False if some base proves ``n`` composite, True otherwise.

    Raises:
        TypeError: If ``n`` or any base is not an int (``bool`` rejected),
            or ``bases`` is not iterable.
        ValueError: If ``bases`` is empty; no bases would mean answering
            "prime" with no evidence.

    Time Complexity:
        O(b log^3 n) for ``b`` bases.

    Space Complexity:
        O(b) - the bases are copied into a list once.

    Examples:
        >>> table = (2, 3, 5, 7919, 1, 9, 341)
        >>> [miller_rabin_fixed_bases(n, [2, 3, 5, 7]) for n in table]
        [True, True, True, True, False, False, False]

        Two composites built to pass fixed bases:

        >>> miller_rabin_fixed_bases(3215031751, [2, 3, 5, 7])
        True
        >>> 151 * 751 * 28351
        3215031751
        >>> miller_rabin_fixed_bases(2047, [2]), 23 * 89
        (True, 2047)
        >>> miller_rabin_fixed_bases(2047, [3])
        False
        >>> miller_rabin_fixed_bases(97, [])
        Traceback (most recent call last):
            ...
        ValueError: bases must not be empty
    """
    _require_int(n, "n")
    try:
        base_list: List[Any] = list(bases)
    except TypeError:
        raise TypeError(
            f"bases must be an iterable of ints, got {type(bases).__name__}"
        ) from None
    if not base_list:
        raise ValueError("bases must not be empty")
    for base in base_list:
        _require_int(base, "every base")
    settled = _settle_small(n)
    if settled is not None:
        return settled
    s, d = _split_power_of_two(n)
    for base in base_list:
        a = base % n
        if a == 0:
            continue
        if _is_witness(a, n, s, d):
            return False
    return True


def is_prime_deterministic(n: int) -> bool:
    """Decide primality exactly by trial division.

    Divide by 2, then by every odd ``f`` while ``f * f <= n``. A composite
    ``n`` always has a factor no larger than ``sqrt(n)``, so finding none
    proves ``n`` prime. This is the always-correct baseline the randomized
    tests are measured against; it is exact but exponential in the number of
    digits of ``n``.

    Args:
        n: The integer to test. Any int; values below 2 are not prime.

    Returns:
        True if ``n`` is prime, False otherwise. Never wrong.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected).

    Time Complexity:
        Best:    O(1)        - ``n`` even or below 2
        Average: O(sqrt(n))  - most composites have a small factor
        Worst:   O(sqrt(n))  - ``n`` prime, about ``sqrt(n) / 2`` divisions

    Space Complexity:
        O(1).

    Examples:
        >>> [is_prime_deterministic(n) for n in (2, 3, 5, 7919, 1, 9, 341)]
        [True, True, True, True, False, False, False]
        >>> is_prime_deterministic(3215031751)
        False
        >>> is_prime_deterministic(1_000_003)
        True
    """
    _require_int(n, "n")
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def is_fermat_liar(n: int, a: int) -> bool:
    """Return True when ``a^(n-1) = 1 (mod n)``, so ``a`` passes the Fermat test.

    For a composite ``n`` such a base is a Fermat liar: it lets the Fermat
    test answer "prime". For a prime ``n`` every base passes, so the word
    only means something when ``n`` is composite.

    Args:
        n: The modulus, an int of at least 2.
        a: The base, an int in ``[1, n - 1]``.

    Returns:
        ``pow(a, n - 1, n) == 1``.

    Raises:
        TypeError: If ``n`` or ``a`` is not an int (``bool`` rejected).
        ValueError: If ``n < 2`` or ``a`` lies outside ``[1, n - 1]``.

    Time Complexity:
        O(log^3 n) - one modular exponentiation.

    Space Complexity:
        O(1) numbers of O(log n) bits.

    Examples:
        >>> is_fermat_liar(341, 2), is_fermat_liar(341, 3)
        (True, False)
        >>> is_fermat_liar(561, 2), is_fermat_liar(561, 3)
        (True, False)
    """
    _require_int(n, "n")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    _require_base(n, a)
    return pow(a, n - 1, n) == 1


def is_strong_liar(n: int, a: int) -> bool:
    """Return True when ``a`` is not a Miller-Rabin witness for the odd ``n``.

    For a composite ``n`` such a base is a strong liar: one Miller-Rabin
    round with this base answers "prime". At most a quarter of the bases are
    strong liars for any odd composite, which is where the ``4^-k`` bound
    comes from. Every strong liar is also a Fermat liar, never the reverse.

    Args:
        n: The modulus, an odd int of at least 3.
        a: The base, an int in ``[1, n - 1]``.

    Returns:
        True if the witness loop with base ``a`` finds nothing, False if
        ``a`` proves ``n`` composite.

    Raises:
        TypeError: If ``n`` or ``a`` is not an int (``bool`` rejected).
        ValueError: If ``n`` is even or below 3, or ``a`` lies outside
            ``[1, n - 1]``.

    Time Complexity:
        O(log^3 n) - one exponentiation and at most ``s - 1`` squarings.

    Space Complexity:
        O(1) numbers of O(log n) bits.

    Examples:
        >>> is_strong_liar(2047, 2), is_strong_liar(2047, 3)
        (True, False)
        >>> is_fermat_liar(561, 2), is_strong_liar(561, 2)
        (True, False)
    """
    _require_odd_modulus(n)
    _require_base(n, a)
    s, d = _split_power_of_two(n)
    return not _is_witness(a, n, s, d)


def count_liars(n: int, kind: str = "fermat") -> int:
    """Count the liars among the bases ``2 .. n - 2`` by trying every one.

    These are exactly the bases :func:`fermat_test` and
    :func:`miller_rabin` draw from, so ``count_liars(n, kind) / (n - 3)`` is
    the exact probability that one round is fooled on the composite ``n``,
    and its ``k``-th power is the exact false-positive rate of ``k`` rounds.
    For a prime ``n`` every base passes and the count is ``n - 3``.

    Exhaustive, so meant for small ``n`` only (thousands, not millions).

    Args:
        n: The modulus: an int of at least 2 for ``"fermat"``, an odd int
            of at least 3 for ``"strong"``.
        kind: ``"fermat"`` to count Fermat liars, ``"strong"`` to count
            strong (Miller-Rabin) liars.

    Returns:
        The number of liars in ``[2, n - 2]``; 0 when that range is empty.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected).
        ValueError: If ``kind`` is not ``"fermat"`` or ``"strong"``, or
            ``n`` is out of range for that kind.

    Time Complexity:
        O(n log^3 n) - one O(log^3 n) test for each of ``n - 3`` bases.

    Space Complexity:
        O(1) numbers of O(log n) bits.

    Examples:
        >>> count_liars(561, "fermat"), count_liars(561, "strong")
        (318, 8)
        >>> count_liars(15, "fermat"), count_liars(15, "strong")
        (2, 0)
        >>> count_liars(7, "strong")
        4
        >>> count_liars(561, "lucas")
        Traceback (most recent call last):
            ...
        ValueError: kind must be 'fermat' or 'strong', got 'lucas'
    """
    if kind not in _LIAR_KINDS:
        raise ValueError(f"kind must be 'fermat' or 'strong', got {kind!r}")
    if kind == "strong":
        _require_odd_modulus(n)
        s, d = _split_power_of_two(n)
        return sum(1 for a in range(2, n - 1) if not _is_witness(a, n, s, d))
    _require_int(n, "n")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    exponent = n - 1
    return sum(1 for a in range(2, n - 1) if pow(a, exponent, n) == 1)
