"""Tests for the Week 7 Monte Carlo primality tests: Fermat, Miller-Rabin, fixed bases.

A primality test is hard to test because almost every wrong implementation
is right almost all of the time. Random odd numbers are mostly divisible by
3, 5 or 7, and any test at all catches those. The bugs that matter live on
a handful of special composites, so this file is built around them and
around four claims the module makes:

* **Exact liar counts.** For the Carmichael numbers 561 .. 8911, the strong
  pseudoprime 2047 and the small composites 15, 91 and 341, the number of
  Fermat liars and strong liars among the bases ``2 .. n - 2`` was counted
  exhaustively and by hand-checked arithmetic, outside the module. Those
  counts are asserted literally, and :func:`count_liars` is cross-checked
  against :func:`brute_strong_liar`, a second Miller-Rabin written here
  from the textbook definition (the full sequence ``a^(d * 2^r)``, not the
  module's early-exit squaring loop), for every ``n <= 341``.
* **The 1/4 bound against the Carmichael numbers.** Every strong-liar
  fraction is at most 1/4 (Rabin 1980), while Fermat on 561 is fooled 57
  percent of the time. The sampled rates are checked against the exact
  counts with tolerances of at least five standard errors, and every
  sampling loop runs from a fixed seed, so the suite cannot flake.
* **One-sided error.** A "composite" (False) answer is a proof and is never
  wrong; a prime is never reported composite. Checked against an
  independent Sieve of Eratosthenes, :func:`sieve`, over every ``n`` below
  3000 and many seeds.
* **Fixed bases can be attacked.** The OEIS A014233 numbers (2047,
  1,373,653, 25,326,001 and 3,215,031,751) pass the first one, two, three
  and four prime bases, and random bases still catch the largest of them.

Hand checks used below (each one done on paper, not by running the module):

* ``2^10 = 1024 = 3 * 341 + 1``, so 2 is a Fermat liar for 341.
* ``2^11 = 2048 = 2047 + 1``, and ``2046 = 2 * 1023 = 2 * 11 * 93``, so
  ``2^1023 = 1 (mod 2047)`` and 2 is a strong liar for 2047.
* For 561 = 3 x 11 x 17, ``560 = 2^4 * 35`` and ``2^35`` is (-1, -1, 8)
  modulo (3, 11, 17). Squaring gives (1, 1, 13), (1, 1, -1), (1, 1, 1):
  the sequence reaches 1 without ever being -1 in all three, so 2 is a
  Fermat liar but a strong WITNESS for 561.
* For 2047 = 23 x 89, ``3^11 = 1 (mod 23)`` and ``3^55 = 52 (mod 89)``
  (since ``3^11 = 37`` and ``3^44 = -1`` by quadratic reciprocity), so
  ``3^1023`` is not +-1 and 3 is a witness.
* For 15, ``4^2 = 16 = 1`` and ``11^2 = 121 = 1 (mod 15)``, so 4 and 11 are
  the Fermat liars; ``14 = 2 * 7`` and ``4^7 = 4``, ``11^7 = 11``, so
  neither is a strong liar.
* For 25, ``7^2 = 49 = -1 (mod 25)``; ``24 = 2^3 * 3`` and ``7^3 = -7 = 18``,
  then ``18^2 = 324 = 24 = -1``, so 7 is a strong liar for 25.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import pytest

from src.randomized.montecarlo_primality import (
    count_liars,
    fermat_test,
    is_fermat_liar,
    is_prime_deterministic,
    is_strong_liar,
    miller_rabin,
    miller_rabin_fixed_bases,
)
from src.utils.random_utils import derive_seed, make_rng

PrimalityTest = Callable[..., bool]


# ----------------------------------------------------------------------
# Hand-checked anchors
# ----------------------------------------------------------------------
#: ``n -> (Fermat liars, strong liars)`` over the bases ``2 .. n - 2``.
#: Counted exhaustively outside the module; never regenerated from it.
LIAR_TABLE: Dict[int, Tuple[int, int]] = {
    561: (318, 8),
    1105: (766, 28),
    1729: (1294, 160),
    2465: (1790, 68),
    2821: (2158, 268),
    6601: (5278, 328),
    8911: (7126, 1780),
    2047: (482, 240),
    15: (2, 0),
    91: (34, 16),
    341: (98, 48),
}

#: The first seven Carmichael numbers (OEIS A002997).
CARMICHAEL: List[int] = [561, 1105, 1729, 2465, 2821, 6601, 8911]

#: The truth table every test must reproduce.
SMALL_PRIMES: List[int] = [2, 3, 5, 7, 11, 7919, 104729]
SMALL_NON_PRIMES: List[int] = [0, 1, 4, 9, 15, 91, 341]

#: The smallest prime above 10^6, 10^7, ... 10^12 (and 10^9 + 7).
LARGE_PRIMES: List[int] = [
    1000003,
    10000019,
    100000007,
    1000000007,
    10000000019,
    100000000003,
    1000000000039,
]

#: OEIS A014233: the smallest odd composite that is a strong pseudoprime to
#: every one of the first ``len(bases)`` primes, with its factorisation.
#: The next prime base in the list catches each one.
STRONG_PSEUDOPRIMES: List[Tuple[int, List[int], Tuple[int, ...], int]] = [
    (2047, [2], (23, 89), 3),
    (1373653, [2, 3], (829, 1657), 5),
    (25326001, [2, 3, 5], (2251, 11251), 7),
    (3215031751, [2, 3, 5, 7], (151, 751, 28351), 11),
]

#: OEIS A001262: the strong pseudoprimes to base 2 below 100,000.
SPSP_BASE_2: List[int] = [
    2047, 3277, 4033, 4681, 8321, 15841, 29341, 42799,
    49141, 52633, 65281, 74665, 80581, 85489, 88357, 90751,
]

#: The two randomized tests, which share a signature.
RANDOMIZED: List[Any] = [
    pytest.param(fermat_test, id="fermat"),
    pytest.param(miller_rabin, id="miller_rabin"),
]


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def sieve(limit: int) -> List[bool]:
    """Return ``is_prime[0 .. limit - 1]`` by the Sieve of Eratosthenes.

    Crosses out multiples; never divides, never exponentiates, so it shares
    nothing with any function under test.

    Args:
        limit: One past the largest number classified.

    Returns:
        A list of ``limit`` booleans.
    """
    flags = [True] * limit
    flags[0:2] = [False, False][: min(2, limit)]
    for p in range(2, math.isqrt(max(limit - 1, 0)) + 1):
        if flags[p]:
            flags[p * p :: p] = [False] * len(range(p * p, limit, p))
    return flags


SIEVE_LIMIT = 20000
IS_PRIME: List[bool] = sieve(SIEVE_LIMIT)


def brute_fermat_liar(n: int, a: int) -> bool:
    """Fermat's condition, straight from the definition: ``a^(n-1) = 1 (mod n)``."""
    return pow(a, n - 1, n) == 1


def brute_strong_liar(n: int, a: int) -> bool:
    """Return True if ``a`` is a strong liar for the odd ``n``, from the definition.

    Writes ``n - 1 = 2^s * d`` and builds the WHOLE sequence
    ``a^d, a^(2d), ..., a^(2^(s-1) d)`` with separate exponentiations, then
    asks the textbook question: does it start at 1, or contain ``n - 1``?
    No early exit, no shared squaring loop, so a mistake in the module's
    loop bounds cannot be copied here.

    Args:
        n: An odd integer, at least 3.
        a: A base in ``[1, n - 1]``.

    Returns:
        True when ``a`` is not a Miller-Rabin witness.
    """
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    seq = [pow(a, d * (2**r), n) for r in range(s)]
    return seq[0] == 1 or (n - 1) in seq


def binomial_tolerance(p: float, trials: int, sigmas: float = 5.0) -> float:
    """Return ``sigmas`` standard errors of a sample proportion."""
    return sigmas * math.sqrt(p * (1.0 - p) / trials)


class ScriptedRandom(random.Random):
    """A ``random.Random`` that hands out a fixed list of bases, and records requests.

    The module draws each base with ``randint(2, n - 2)``. Scripting the
    answers lets a test pick the exact base a round uses, so one round's
    verdict can be compared with a hand-checked liar or witness, and the
    recorded ranges show which interval the module asked for.
    """

    def __init__(self, bases: Iterable[int]) -> None:
        super().__init__(0)
        self.bases: List[int] = list(bases)
        self.calls: List[Tuple[int, int]] = []

    def randint(self, a: int, b: int) -> int:  # type: ignore[override]
        self.calls.append((a, b))
        return self.bases.pop(0)


# ----------------------------------------------------------------------
# The truth table, for all three tests
# ----------------------------------------------------------------------
class TestTruthTable:
    """Primes 2 .. 104729 are prime; 0, 1, 4, 9, 15, 91, 341 are not."""

    @pytest.mark.parametrize("seed", [0, 7, 2026])
    @pytest.mark.parametrize("n", SMALL_PRIMES)
    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_randomized_primes(self, test: PrimalityTest, n: int, seed: int) -> None:
        assert test(n, k=20, seed=seed) is True

    @pytest.mark.parametrize("seed", [0, 7, 2026])
    @pytest.mark.parametrize("n", SMALL_NON_PRIMES)
    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_randomized_non_primes(
        self, test: PrimalityTest, n: int, seed: int
    ) -> None:
        assert test(n, k=20, seed=seed) is False

    @pytest.mark.parametrize("n", SMALL_PRIMES)
    def test_deterministic_primes(self, n: int) -> None:
        assert is_prime_deterministic(n) is True

    @pytest.mark.parametrize("n", SMALL_NON_PRIMES)
    def test_deterministic_non_primes(self, n: int) -> None:
        assert is_prime_deterministic(n) is False

    @pytest.mark.parametrize("n", SMALL_PRIMES + SMALL_NON_PRIMES)
    def test_fixed_bases_agree_on_the_table(self, n: int) -> None:
        assert miller_rabin_fixed_bases(n, [2, 3, 5, 7]) is (n in SMALL_PRIMES)

    @pytest.mark.parametrize("n", [-1, -2, -7, -561, -(2**61 - 1)])
    @pytest.mark.parametrize(
        "test",
        [
            lambda n: fermat_test(n, k=3, seed=1),
            lambda n: miller_rabin(n, k=3, seed=1),
            is_prime_deterministic,
            lambda n: miller_rabin_fixed_bases(n, [2]),
        ],
        ids=["fermat", "miller_rabin", "deterministic", "fixed_bases"],
    )
    def test_negatives_are_not_prime(self, test: Callable[[int], bool], n: int) -> None:
        assert test(n) is False

    def test_table_agrees_with_the_sieve(self) -> None:
        # The anchors themselves, confirmed by an oracle that never divides.
        assert all(IS_PRIME[n] for n in SMALL_PRIMES if n < SIEVE_LIMIT)
        assert not any(IS_PRIME[n] for n in SMALL_NON_PRIMES)

    def test_deterministic_matches_the_sieve_everywhere_below_20000(self) -> None:
        wrong = [
            n for n in range(SIEVE_LIMIT) if is_prime_deterministic(n) is not IS_PRIME[n]
        ]
        assert wrong == []


# ----------------------------------------------------------------------
# Exact liar counts
# ----------------------------------------------------------------------
class TestLiarCounts:
    """Exhaustive liar counts over ``2 .. n - 2``, asserted from the table.

    The strong-liar fraction for 8911 is 1780 / 8908, about 0.1998, the
    largest in the table and still below Rabin's 1/4. The Fermat fraction
    for 561 is 318 / 558, about 0.5699: there is no constant bound at all.
    """

    @pytest.mark.parametrize("n, counts", LIAR_TABLE.items(), ids=str)
    def test_fermat_count(self, n: int, counts: Tuple[int, int]) -> None:
        assert count_liars(n, "fermat") == counts[0]

    @pytest.mark.parametrize("n, counts", LIAR_TABLE.items(), ids=str)
    def test_strong_count(self, n: int, counts: Tuple[int, int]) -> None:
        assert count_liars(n, "strong") == counts[1]

    def test_default_kind_is_fermat(self) -> None:
        assert count_liars(561) == 318

    @pytest.mark.parametrize("n", LIAR_TABLE, ids=str)
    def test_number_of_bases_is_n_minus_3(self, n: int) -> None:
        # The table's last column: bases 2 .. n - 2 inclusive.
        assert len(range(2, n - 1)) == n - 3

    @pytest.mark.parametrize("n, counts", LIAR_TABLE.items(), ids=str)
    def test_strong_fraction_respects_rabins_quarter(
        self, n: int, counts: Tuple[int, int]
    ) -> None:
        assert counts[1] / (n - 3) <= 0.25
        assert count_liars(n, "strong") / (n - 3) <= 0.25

    def test_8911_is_the_worst_case_in_the_table(self) -> None:
        fractions = {n: s / (n - 3) for n, (_, s) in LIAR_TABLE.items()}
        assert max(fractions, key=fractions.__getitem__) == 8911
        assert fractions[8911] == pytest.approx(0.1998, abs=1e-4)

    @pytest.mark.parametrize("n", CARMICHAEL, ids=str)
    def test_carmichael_fermat_fraction_far_exceeds_a_quarter(self, n: int) -> None:
        # Every base coprime to n lies, so Fermat has no 1/4 (or 1/2) bound.
        assert LIAR_TABLE[n][0] / (n - 3) > 0.5

    @pytest.mark.parametrize("n", CARMICHAEL, ids=str)
    def test_carmichael_fermat_liars_are_exactly_the_units(self, n: int) -> None:
        # Korselt: every a with gcd(a, n) = 1 is a Fermat liar, and no other.
        units = sum(1 for a in range(2, n - 1) if math.gcd(a, n) == 1)
        assert units == LIAR_TABLE[n][0]

    def test_strong_liars_never_outnumber_fermat_liars(self) -> None:
        assert all(s <= f for f, s in LIAR_TABLE.values())

    @pytest.mark.parametrize("p", [5, 7, 13, 101, 7919])
    @pytest.mark.parametrize("kind", ["fermat", "strong"])
    def test_prime_has_every_base_a_liar(self, p: int, kind: str) -> None:
        assert count_liars(p, kind) == p - 3

    def test_docstring_small_cases(self) -> None:
        assert count_liars(7, "strong") == 4
        assert (count_liars(2, "fermat"), count_liars(3, "fermat")) == (0, 0)
        assert count_liars(3, "strong") == 0
        assert count_liars(4, "fermat") == 0
        assert count_liars(9, "fermat") == 0 == count_liars(9, "strong")


class TestLiarCountsAgainstBruteForce:
    """``count_liars`` against the definitions in this file, for every ``n <= 341``."""

    def test_fermat_counts_match_for_every_n_up_to_341(self) -> None:
        for n in range(2, 342):
            expected = sum(1 for a in range(2, n - 1) if brute_fermat_liar(n, a))
            assert count_liars(n, "fermat") == expected, n

    def test_strong_counts_match_for_every_odd_n_up_to_341(self) -> None:
        for n in range(3, 342, 2):
            expected = sum(1 for a in range(2, n - 1) if brute_strong_liar(n, a))
            assert count_liars(n, "strong") == expected, n

    @pytest.mark.parametrize("n", [15, 91, 341])
    def test_brute_force_reproduces_the_table(self, n: int) -> None:
        bases = range(2, n - 1)
        fermat = sum(1 for a in bases if brute_fermat_liar(n, a))
        strong = sum(1 for a in bases if brute_strong_liar(n, a))
        assert (fermat, strong) == LIAR_TABLE[n]

    def test_rabin_bound_holds_for_every_odd_composite_up_to_341(self) -> None:
        for n in range(9, 342, 2):
            if not IS_PRIME[n]:
                assert count_liars(n, "strong") / (n - 3) <= 0.25, n


# ----------------------------------------------------------------------
# Single-base predicates on hand-checkable cases
# ----------------------------------------------------------------------
class TestLiarPredicates:
    """``is_fermat_liar`` and ``is_strong_liar`` on cases checked by hand."""

    HAND_CASES: List[Any] = [
        # (n, a, fermat liar?, strong liar?)
        pytest.param(341, 2, True, False, id="341-base-2"),
        pytest.param(341, 3, False, False, id="341-base-3"),
        pytest.param(561, 2, True, False, id="561-base-2"),
        pytest.param(561, 3, False, False, id="561-base-3-shares-factor"),
        pytest.param(2047, 2, True, True, id="2047-base-2"),
        pytest.param(2047, 3, False, False, id="2047-base-3"),
        pytest.param(15, 4, True, False, id="15-base-4"),
        pytest.param(15, 11, True, False, id="15-base-11"),
        pytest.param(15, 2, False, False, id="15-base-2"),
        pytest.param(25, 7, True, True, id="25-base-7"),
        pytest.param(9, 2, False, False, id="9-base-2"),
    ]

    @pytest.mark.parametrize("n, a, fermat, strong", HAND_CASES)
    def test_hand_checked(self, n: int, a: int, fermat: bool, strong: bool) -> None:
        assert is_fermat_liar(n, a) is fermat
        assert is_strong_liar(n, a) is strong

    def test_hand_arithmetic(self) -> None:
        # The paper arithmetic behind the cases above.
        assert 2**10 == 3 * 341 + 1
        assert 2**11 == 2047 + 1 and 2046 == 2 * 11 * 93
        assert pow(2, 35, 3) == 2 and pow(2, 35, 11) == 10 and pow(2, 35, 17) == 8
        assert 4**2 % 15 == 1 and 11**2 % 15 == 1
        assert 7**2 % 25 == 24 and 18**2 % 25 == 24

    def test_fermat_liars_of_15_are_4_and_11(self) -> None:
        assert [a for a in range(2, 14) if is_fermat_liar(15, a)] == [4, 11]

    @pytest.mark.parametrize("n", [9, 15, 25, 91, 341, 561, 2047])
    def test_one_and_minus_one_always_lie(self, n: int) -> None:
        for a in (1, n - 1):
            assert is_fermat_liar(n, a) is True
            assert is_strong_liar(n, a) is True

    @pytest.mark.parametrize("p", [3, 5, 13, 97])
    def test_prime_has_no_witnesses(self, p: int) -> None:
        assert all(is_fermat_liar(p, a) for a in range(1, p))
        assert all(is_strong_liar(p, a) for a in range(1, p))

    @pytest.mark.parametrize("n", [341, 561, 1105, 2047])
    def test_every_strong_liar_is_a_fermat_liar(self, n: int) -> None:
        for a in range(1, n):
            if is_strong_liar(n, a):
                assert is_fermat_liar(n, a), a

    @pytest.mark.parametrize("n", [91, 561, 2047])
    def test_predicates_match_brute_force(self, n: int) -> None:
        for a in range(1, n):
            assert is_fermat_liar(n, a) is brute_fermat_liar(n, a)
            assert is_strong_liar(n, a) is brute_strong_liar(n, a)


# ----------------------------------------------------------------------
# Scripted bases: one round at a time
# ----------------------------------------------------------------------
class TestScriptedRounds:
    """Forcing the base each round uses, via :class:`ScriptedRandom`."""

    def test_fermat_fooled_by_the_liar_2_on_561(self) -> None:
        rng = ScriptedRandom([2])
        assert fermat_test(561, k=1, rng=rng) is True
        assert rng.calls == [(2, 559)]

    def test_fermat_catches_561_with_the_factor_3(self) -> None:
        assert fermat_test(561, k=1, rng=ScriptedRandom([3])) is False

    def test_miller_rabin_not_fooled_by_2_on_561(self) -> None:
        # 2 is a Fermat liar for 561 but a strong witness (hand-checked).
        assert miller_rabin(561, k=1, rng=ScriptedRandom([2])) is False

    def test_miller_rabin_fooled_by_2_on_2047_and_caught_by_3(self) -> None:
        assert miller_rabin(2047, k=1, rng=ScriptedRandom([2])) is True
        assert miller_rabin(2047, k=2, rng=ScriptedRandom([2, 3])) is False

    def test_liars_do_not_stop_the_test_but_a_witness_does(self) -> None:
        rng = ScriptedRandom([4, 11, 2, 7, 8])
        assert fermat_test(15, k=5, rng=rng) is False
        assert rng.calls == [(2, 13)] * 3  # stopped at the witness 2

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_prime_uses_all_k_rounds_from_2_to_n_minus_2(
        self, test: PrimalityTest
    ) -> None:
        rng = ScriptedRandom([2, 6, 11])
        assert test(13, k=3, rng=rng) is True
        assert rng.calls == [(2, 11)] * 3

    @pytest.mark.parametrize("n", [-3, 0, 1, 2, 3, 4, 100])
    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_front_end_draws_no_base(self, test: PrimalityTest, n: int) -> None:
        rng = ScriptedRandom([])
        assert test(n, k=5, rng=rng) is (n in (2, 3))
        assert rng.calls == []

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_five_draws_from_2_to_3(self, test: PrimalityTest) -> None:
        # n = 5 is the smallest odd n that needs a base; the range is [2, 3].
        rng = ScriptedRandom([3])
        assert test(5, k=1, rng=rng) is True
        assert rng.calls == [(2, 3)]


# ----------------------------------------------------------------------
# Fermat on 561: the sampled rate and the k-th power
# ----------------------------------------------------------------------
class TestFermatOn561:
    """One Fermat round on 561 lies with probability exactly 318 / 558."""

    P_ONE_ROUND = 318 / 558

    def test_exact_one_round_probability(self) -> None:
        assert count_liars(561, "fermat") / (561 - 3) == self.P_ONE_ROUND
        assert self.P_ONE_ROUND == pytest.approx(0.5699, abs=1e-4)

    def test_predicted_rates_for_5_and_10_rounds(self) -> None:
        assert self.P_ONE_ROUND**5 == pytest.approx(0.0601, abs=1e-4)
        assert self.P_ONE_ROUND**10 == pytest.approx(0.00361, abs=1e-5)

    def test_one_round_rate_over_5000_seeded_trials(self) -> None:
        trials = 5000
        fooled = sum(fermat_test(561, k=1, seed=s) for s in range(trials))
        # 0.05 is about seven standard errors (SE ~ 0.0070).
        assert 0.05 >= binomial_tolerance(self.P_ONE_ROUND, trials, 5.0)
        assert abs(fooled / trials - self.P_ONE_ROUND) <= 0.05

    def test_one_round_rate_with_a_shared_generator(self) -> None:
        trials = 5000
        rng = make_rng(561)
        fooled = sum(fermat_test(561, k=1, rng=rng) for _ in range(trials))
        assert abs(fooled / trials - self.P_ONE_ROUND) <= 0.05

    def test_five_round_rate_is_the_fifth_power(self) -> None:
        trials = 5000
        p5 = self.P_ONE_ROUND**5
        rng = make_rng(derive_seed(7, "fermat-561-k5"))
        fooled = sum(fermat_test(561, k=5, rng=rng) for _ in range(trials))
        # 5 SE is about 0.0168; 0.02 is wider still.
        assert 0.02 >= binomial_tolerance(p5, trials, 5.0)
        assert abs(fooled / trials - p5) <= 0.02

    def test_miller_rabin_one_round_rate_on_561(self) -> None:
        trials = 5000
        p = 8 / 558
        rng = make_rng(derive_seed(7, "mr-561-k1"))
        fooled = sum(miller_rabin(561, k=1, rng=rng) for _ in range(trials))
        assert abs(fooled / trials - p) <= binomial_tolerance(p, trials, 6.0)

    def test_miller_rabin_one_round_rate_on_8911_is_below_a_quarter(self) -> None:
        trials = 4000
        p = 1780 / 8908
        rng = make_rng(derive_seed(7, "mr-8911-k1"))
        fooled = sum(miller_rabin(8911, k=1, rng=rng) for _ in range(trials))
        tol = binomial_tolerance(p, trials, 5.0)
        assert abs(fooled / trials - p) <= tol
        assert fooled / trials <= 0.25 + tol


# ----------------------------------------------------------------------
# Fixed bases can be attacked; random bases cannot
# ----------------------------------------------------------------------
class TestFixedBaseAttack:
    """OEIS A014233 composites built to pass the first few prime bases."""

    def test_3215031751_fools_bases_2_3_5_7(self) -> None:
        assert 151 * 751 * 28351 == 3215031751
        assert miller_rabin_fixed_bases(3215031751, [2, 3, 5, 7]) is True
        assert is_prime_deterministic(3215031751) is False

    def test_2047_fools_base_2(self) -> None:
        assert 23 * 89 == 2047
        assert miller_rabin_fixed_bases(2047, [2]) is True
        assert miller_rabin_fixed_bases(2047, [3]) is False

    @pytest.mark.parametrize("seed", range(50))
    def test_random_bases_catch_3215031751(self, seed: int) -> None:
        assert miller_rabin(3215031751, k=5, seed=seed) is False

    @pytest.mark.parametrize(
        "n, bases, factors, next_base",
        STRONG_PSEUDOPRIMES,
        ids=[str(row[0]) for row in STRONG_PSEUDOPRIMES],
    )
    def test_each_a014233_entry_passes_its_bases_and_fails_the_next(
        self, n: int, bases: List[int], factors: Tuple[int, ...], next_base: int
    ) -> None:
        assert math.prod(factors) == n
        assert miller_rabin_fixed_bases(n, bases) is True
        assert miller_rabin_fixed_bases(n, bases + [next_base]) is False
        assert all(is_strong_liar(n, a) for a in bases)
        assert all(brute_strong_liar(n, a) for a in bases)

    @pytest.mark.parametrize("n", SPSP_BASE_2)
    def test_base_2_pseudoprimes_are_caught_by_2_3_5_7(self, n: int) -> None:
        assert miller_rabin_fixed_bases(n, [2]) is True
        assert miller_rabin_fixed_bases(n, [2, 3, 5, 7]) is False
        assert is_prime_deterministic(n) is False

    @pytest.mark.parametrize(
        "bases, limit",
        [
            pytest.param([2], 2047, id="2-exact-below-2047"),
            pytest.param([2, 3], SIEVE_LIMIT, id="2-3"),
            pytest.param([2, 3, 5, 7], SIEVE_LIMIT, id="2-3-5-7"),
        ],
    )
    def test_fixed_bases_are_exact_below_their_limit(
        self, bases: List[int], limit: int
    ) -> None:
        # 2047 is the SMALLEST strong pseudoprime to base 2, so [2] is
        # exact on every n below it; the longer lists are exact far beyond
        # the sieve, and are checked against it on its whole range.
        wrong = [
            n
            for n in range(limit)
            if miller_rabin_fixed_bases(n, bases) is not IS_PRIME[n]
        ]
        assert wrong == []

    def test_base_2_first_fails_at_2047(self) -> None:
        assert miller_rabin_fixed_bases(2047, [2]) is not IS_PRIME[2047]

    def test_base_equal_to_n_is_skipped_for_a_prime(self) -> None:
        assert miller_rabin_fixed_bases(7, [2, 3, 5, 7]) is True
        assert miller_rabin_fixed_bases(5, [5, 10]) is True

    def test_bases_are_reduced_modulo_n(self) -> None:
        # 2049 = 2 (mod 2047) and -2045 = 2 (mod 2047), so both lie like 2.
        assert miller_rabin_fixed_bases(2047, [2049]) is True
        assert miller_rabin_fixed_bases(2047, [-2045]) is True
        assert miller_rabin_fixed_bases(2047, [2047 + 3]) is False

    def test_bases_may_be_any_iterable(self) -> None:
        assert miller_rabin_fixed_bases(2047, iter([2, 3])) is False
        assert miller_rabin_fixed_bases(2047, (b for b in [2])) is True

    def test_every_base_a_multiple_of_a_composite_n_answers_prime(self) -> None:
        # FLAGGED (reported in bugs_found): when every base is a multiple of
        # n, every base is skipped and the function answers "prime" with no
        # evidence at all, the very case the empty-list ValueError exists to
        # refuse. 9 and 15 are composite. Harmless for prime-base lists
        # such as [2, 3, 5, 7], where n | p forces n == p. This test
        # documents the CURRENT behaviour; it is not an endorsement.
        assert miller_rabin_fixed_bases(9, [9]) is True
        assert miller_rabin_fixed_bases(15, [15, 30]) is True


# ----------------------------------------------------------------------
# Large numbers
# ----------------------------------------------------------------------
class TestLargeNumbers:
    """Primes far beyond the sieve, where trial division is the slow oracle."""

    @pytest.mark.parametrize("p", LARGE_PRIMES, ids=str)
    def test_deterministic_and_randomized_agree_on_large_primes(self, p: int) -> None:
        assert is_prime_deterministic(p) is True
        assert miller_rabin(p, k=20, seed=p % 1000) is True
        assert fermat_test(p, k=20, seed=p % 1000) is True

    @pytest.mark.parametrize("seed", range(10))
    def test_mersenne_61_prime_by_miller_rabin(self, seed: int) -> None:
        # 2**61 - 1 is never trial-divided: about 7.6e8 divisions.
        assert miller_rabin(2**61 - 1, k=20, seed=seed) is True

    @pytest.mark.parametrize("seed", range(10))
    def test_mersenne_31_prime_by_miller_rabin(self, seed: int) -> None:
        assert miller_rabin(2**31 - 1, k=20, seed=seed) is True

    def test_mersenne_61_passes_fermat_too(self) -> None:
        # A prime has no Fermat witness either.
        assert fermat_test(2**61 - 1, k=20, seed=1) is True

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_1000001_is_composite_for_all_three(self, seed: int) -> None:
        assert 101 * 9901 == 1000001
        assert is_prime_deterministic(1000001) is False
        assert miller_rabin(1000001, k=20, seed=seed) is False
        assert fermat_test(1000001, k=20, seed=seed) is False

    def test_large_composite_with_large_factors(self) -> None:
        # 1000003 * 1000033: no small factor for trial division to stumble on.
        n = 1000003 * 1000033
        assert miller_rabin(n, k=20, seed=3) is False
        assert fermat_test(n, k=20, seed=3) is False

    def test_mersenne_composite(self) -> None:
        # 2**67 - 1 = 193707721 * 761838257287 (Cole, 1903).
        assert 193707721 * 761838257287 == 2**67 - 1
        assert miller_rabin(2**67 - 1, k=20, seed=1) is False


# ----------------------------------------------------------------------
# One-sided error
# ----------------------------------------------------------------------
class TestOneSidedError:
    """False ("composite") is never wrong; a prime is never called composite."""

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_false_answers_are_always_composite_below_3000(
        self, test: PrimalityTest
    ) -> None:
        # k = 1 makes the test as weak as it gets, so it is fooled often and
        # both answers occur; still, no False may land on a prime.
        fooled = 0
        for seed in range(20):
            rng = make_rng(derive_seed(5300, test.__name__, seed))
            for n in range(-3, 3000):
                answer = test(n, k=1, rng=rng)
                if answer is False:
                    assert not (n >= 0 and IS_PRIME[n]), (n, seed)
                elif n >= 2 and not IS_PRIME[n]:
                    fooled += 1
                if n >= 2 and IS_PRIME[n]:
                    assert answer is True, (n, seed)
        # Not vacuous: some composites really did slip through.
        assert fooled > 0

    @pytest.mark.parametrize("p", LARGE_PRIMES + [2**31 - 1, 2**61 - 1], ids=str)
    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_primes_never_reported_composite(self, test: PrimalityTest, p: int) -> None:
        rng = make_rng(derive_seed(5300, "primes", p))
        assert all(test(p, k=1, rng=rng) for _ in range(200))

    def test_only_the_8_strong_liars_of_561_fool_miller_rabin(self) -> None:
        # Every Fermat liar of 561 that is NOT a strong liar is caught by
        # Miller-Rabin; the 8 strong liars are the only bases that fool it.
        fermat_liars = [a for a in range(2, 560) if brute_fermat_liar(561, a)]
        assert len(fermat_liars) == 318
        fooled_mr = [
            a for a in fermat_liars if miller_rabin(561, k=1, rng=ScriptedRandom([a]))
        ]
        assert len(fooled_mr) == 8
        assert all(fermat_test(561, k=1, rng=ScriptedRandom([a])) for a in fermat_liars)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
class TestReproducibility:
    """The same seed gives the same bases, and so the same answer."""

    @pytest.mark.parametrize("n", CARMICHAEL + [2047, 341])
    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_same_seed_same_answer(self, test: PrimalityTest, n: int) -> None:
        for seed in range(25):
            assert test(n, k=1, seed=seed) == test(n, k=1, seed=seed)

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_same_seed_same_answer_sequence(self, test: PrimalityTest) -> None:
        first = [test(561, k=1, seed=s) for s in range(200)]
        second = [test(561, k=1, seed=s) for s in range(200)]
        assert first == second
        assert True in first and False in first  # the seed really matters

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_rng_takes_precedence_over_seed(self, test: PrimalityTest) -> None:
        for s in range(50):
            assert test(561, k=1, seed=999, rng=make_rng(s)) == test(561, k=1, seed=s)

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_shared_generator_continues_its_stream(self, test: PrimalityTest) -> None:
        a, b = make_rng(42), make_rng(42)
        run_a = [test(561, k=1, rng=a) for _ in range(100)]
        run_b = [test(561, k=1, rng=b) for _ in range(100)]
        assert run_a == run_b

    def test_seed_equals_a_plain_random_with_that_seed(self) -> None:
        for s in range(50):
            assert fermat_test(561, k=1, seed=s) == fermat_test(
                561, k=1, rng=random.Random(s)
            )

    def test_docstring_shared_generator_example(self) -> None:
        shared = make_rng(5)
        assert [miller_rabin(n, k=5, rng=shared) for n in (561, 1105, 1729)] == [
            False,
            False,
            False,
        ]


# ----------------------------------------------------------------------
# Contracts
# ----------------------------------------------------------------------
class TestContracts:
    """Bad input is refused with the documented exception type."""

    RANDOMIZED_BAD: List[Any] = [
        pytest.param((7,), {"k": 0}, ValueError, id="k-zero"),
        pytest.param((7,), {"k": -3}, ValueError, id="k-negative"),
        pytest.param((7,), {"k": True}, TypeError, id="k-bool"),
        pytest.param((7,), {"k": 2.0}, TypeError, id="k-float"),
        pytest.param((7.0,), {}, TypeError, id="n-float"),
        pytest.param((True,), {}, TypeError, id="n-bool"),
        pytest.param(("7",), {}, TypeError, id="n-str"),
        pytest.param((7,), {"seed": "x"}, TypeError, id="seed-str"),
        pytest.param((7,), {"seed": True}, TypeError, id="seed-bool"),
        pytest.param((7,), {"rng": 5}, TypeError, id="rng-int"),
        pytest.param((2,), {"k": 0}, ValueError, id="k-checked-before-front-end"),
    ]

    @pytest.mark.parametrize("args, kwargs, error", RANDOMIZED_BAD)
    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_randomized_bad_input(
        self,
        test: PrimalityTest,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        error: type,
    ) -> None:
        with pytest.raises(error):
            test(*args, **kwargs)

    @pytest.mark.parametrize(
        "bases, error",
        [
            pytest.param([], ValueError, id="empty"),
            pytest.param(5, TypeError, id="not-iterable"),
            pytest.param([2.0], TypeError, id="float-base"),
            pytest.param([True], TypeError, id="bool-base"),
            pytest.param(["2"], TypeError, id="str-base"),
            pytest.param([2, None], TypeError, id="none-base"),
        ],
    )
    def test_fixed_bases_bad_input(self, bases: Any, error: type) -> None:
        with pytest.raises(error):
            miller_rabin_fixed_bases(97, bases)

    def test_fixed_bases_bad_n(self) -> None:
        with pytest.raises(TypeError):
            miller_rabin_fixed_bases(97.0, [2])  # type: ignore[arg-type]

    @pytest.mark.parametrize("n", [7.0, True, "7", None])
    def test_deterministic_rejects_non_int(self, n: Any) -> None:
        with pytest.raises(TypeError):
            is_prime_deterministic(n)

    @pytest.mark.parametrize(
        "n, kind, error",
        [
            pytest.param(561, "lucas", ValueError, id="bad-kind"),
            pytest.param(561, "Fermat", ValueError, id="kind-case"),
            pytest.param(560, "strong", ValueError, id="strong-even"),
            pytest.param(1, "strong", ValueError, id="strong-one"),
            pytest.param(1, "fermat", ValueError, id="fermat-one"),
            pytest.param(-5, "fermat", ValueError, id="fermat-negative"),
            pytest.param(True, "fermat", TypeError, id="bool-n"),
            pytest.param(5.0, "strong", TypeError, id="float-n"),
        ],
    )
    def test_count_liars_bad_input(self, n: Any, kind: str, error: type) -> None:
        with pytest.raises(error):
            count_liars(n, kind)

    @pytest.mark.parametrize(
        "n, a, error",
        [
            pytest.param(561, 0, ValueError, id="a-zero"),
            pytest.param(561, 561, ValueError, id="a-equals-n"),
            pytest.param(561, -1, ValueError, id="a-negative"),
            pytest.param(1, 1, ValueError, id="n-one"),
            pytest.param(561, True, TypeError, id="a-bool"),
            pytest.param(561, 2.0, TypeError, id="a-float"),
            pytest.param(561.0, 2, TypeError, id="n-float"),
        ],
    )
    def test_is_fermat_liar_bad_input(self, n: Any, a: Any, error: type) -> None:
        with pytest.raises(error):
            is_fermat_liar(n, a)

    @pytest.mark.parametrize(
        "n, a, error",
        [
            pytest.param(560, 2, ValueError, id="n-even"),
            pytest.param(1, 1, ValueError, id="n-one"),
            pytest.param(561, 0, ValueError, id="a-zero"),
            pytest.param(561, 561, ValueError, id="a-equals-n"),
            pytest.param(561, True, TypeError, id="a-bool"),
            pytest.param(True, 1, TypeError, id="n-bool"),
        ],
    )
    def test_is_strong_liar_bad_input(self, n: Any, a: Any, error: type) -> None:
        with pytest.raises(error):
            is_strong_liar(n, a)

    def test_smallest_valid_moduli(self) -> None:
        assert is_fermat_liar(2, 1) is True
        assert is_strong_liar(3, 1) is True and is_strong_liar(3, 2) is True

    @pytest.mark.parametrize("test", RANDOMIZED)
    def test_unseeded_call_still_answers_correctly(self, test: PrimalityTest) -> None:
        # seed=None draws from the OS; on a prime and on an even number the
        # answer cannot depend on the draw, so this cannot flake.
        assert test(7919) is True
        assert test(7920) is False
