"""Tests for MinHash signatures, exact Jaccard similarity, and word shingles.

MinHash is a Monte Carlo estimator with two-sided error: the number it
returns is random, and it can land above or below the truth. A test of a
random number can go wrong in two ways. It can assert whatever number the
code happened to print, which proves nothing, or it can use a tolerance so
tight that one unlucky draw fails the build. This file avoids both:

* **Exact anchors, counted by hand.** :func:`jaccard` is the ground truth
  that MinHash estimates, so it is checked on sets whose overlap can be
  counted on paper: ``{0..99}`` and ``{50..149}`` share the 50 tokens
  ``50..99`` out of 150 distinct ones, so ``J = 50 / 150 = 1/3`` exactly.
  The expected outputs of :func:`shingles` are written out word by word.
* **Cases MinHash must get exactly right.** Identical sets have identical
  minima under every hash, so they estimate exactly 1.0 for every ``k`` and
  every seed. The minimum over ``A | B`` is the smaller of the two minima,
  so the signature of a union is the element-wise minimum of the two
  signatures. Neither claim involves any chance.
* **An independent reference.** :func:`reference_signature` rebuilds a
  signature from the Week 7 specification alone (``random.Random(seed)``
  drawing ``(a, b)`` pairs, BLAKE2b of ``repr(token)`` reduced modulo
  ``2^61 - 1``, then ``min((a x + b) mod p)``), calling ``hashlib``
  directly instead of the module's helpers. Agreement means the module
  computes the hash family the report describes, and
  :func:`is_prime_by_bases` confirms the modulus really is prime, which is
  what makes each hash a bijection.
* **Statistics with room to spare.** One estimate from ``k`` positions has
  standard error ``sqrt(J (1 - J) / k)``, which is 0.0208 for ``J = 1/3``
  at ``k = 512``. Every tolerance on a single estimate is at least five
  standard errors, and every draw comes from a fixed seed, so the suite
  cannot flake. The error shrinking with ``k`` is checked as an ordering of
  two means, never as a value.
* **Reproducibility across processes.** A signature computed in a child
  interpreter under a different ``PYTHONHASHSEED`` must equal the one
  computed here. That is the test that would fail if the built-in hash,
  which is salted per process for strings, had been used to turn tokens
  into integers.

No assertion depends on a timing, and the file runs in a few seconds next
to the seven thousand other tests.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Tuple

import pytest

from src.randomized.minhash_similarity import (
    MERSENNE_61,
    MinHash,
    estimate_similarity,
    jaccard,
    shingles,
)
from src.utils.random_utils import make_rng

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The modulus, written out here rather than read from the module.
P: int = 2**61 - 1

#: The assignment's anchor pair: 50 shared tokens out of 150, so J = 1/3.
A_THIRD = frozenset(range(0, 100))
B_THIRD = frozenset(range(50, 150))
J_THIRD = 1 / 3

#: A pair with nothing in common: 0..99 and 100..199.
DISJOINT_A = frozenset(range(0, 100))
DISJOINT_B = frozenset(range(100, 200))

#: Pairs drawn from the universe 0..99 with controlled overlap. In each one
#: A = range(0, x) and B = range(x - c, 100), so the union is all 100 tokens
#: and the intersection is the c tokens x - c .. x - 1. Hence J = c / 100.
CONTROLLED_PAIRS = [
    # 45..54 shared: 10 tokens.
    pytest.param(range(0, 55), range(45, 100), 10, id="J=0.10"),
    # 37..61 shared: 25 tokens.
    pytest.param(range(0, 62), range(37, 100), 25, id="J=0.25"),
    # 25..74 shared: 50 tokens.
    pytest.param(range(0, 75), range(25, 100), 50, id="J=0.50"),
    # 12..86 shared: 75 tokens.
    pytest.param(range(0, 87), range(12, 100), 75, id="J=0.75"),
    # 5..94 shared: 90 tokens.
    pytest.param(range(0, 95), range(5, 100), 90, id="J=0.90"),
]

#: Two original ten-word sentences that differ in one word (index 5,
#: "speed" -> "time"). All ten words are distinct, so each sentence has
#: 10 - 3 + 1 = 8 three-word shingles. The changed word sits inside the
#: shingles that start at indices 3, 4 and 5, so 3 of the 8 change and 5
#: survive. Union 8 + 3 = 11, intersection 5, so J = 5/11.
DOC_SPEED = "randomized algorithms trade certainty for speed in a controlled way"
DOC_TIME = "randomized algorithms trade certainty for time in a controlled way"

#: An original sentence for the cross-process test. Its 3-shingles are a
#: set of str, whose iteration order follows the per-process string hash.
CHILD_TEXT = (
    "a bloom filter can only answer maybe or no while minhash says how much "
    "two documents overlap using short signatures built from random minima"
)


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def standard_error(j: float, k: int) -> float:
    """Standard error of a k-position MinHash estimate: ``sqrt(J (1 - J) / k)``."""
    return math.sqrt(j * (1 - j) / k)


def reference_signature(tokens: Iterable[object], num_perm: int, seed: int) -> List[int]:
    """Rebuild a MinHash signature from the specification, without the module.

    Parameters are ``num_perm`` pairs ``(a, b)`` drawn in that order from
    ``random.Random(seed)``, with ``a`` in ``[1, p - 1]`` and ``b`` in
    ``[0, p - 1]``. A token becomes the 64-bit little-endian BLAKE2b digest
    of ``repr(token)`` with a 16-byte zero salt, reduced modulo ``p``.
    Position ``i`` is the minimum of ``(a_i x + b_i) mod p`` over the set.

    Args:
        tokens: The set to sign, as any iterable. Duplicates collapse.
        num_perm: Signature length ``k``.
        seed: Seed for the parameter draw.

    Returns:
        A list of ``num_perm`` integers in ``[0, p)``.
    """
    rng = random.Random(seed)
    params = [(rng.randint(1, P - 1), rng.randint(0, P - 1)) for _ in range(num_perm)]
    xs = {
        int.from_bytes(
            hashlib.blake2b(
                repr(token).encode("utf-8"), digest_size=8, salt=bytes(16)
            ).digest(),
            "little",
        )
        % P
        for token in tokens
    }
    return [min((a * x + b) % P for x in xs) for a, b in params]


def is_prime_by_bases(n: int) -> bool:
    """Deterministic strong-probable-prime test with the first twelve prime bases.

    Bases 2, 3, 5, ..., 37 together decide primality exactly for every
    ``n < 3.18 * 10^23`` (Sorenson and Webster, 2015), which covers
    ``2^61 - 1``. Written here so the primality of the modulus does not rest
    on any Week 7 module.
    """
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in bases:
        return True
    if any(n % b == 0 for b in bases):
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for base in bases:
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@lru_cache(maxsize=None)
def third_estimates(num_perm: int, seeds: Tuple[int, ...]) -> Tuple[float, ...]:
    """Estimates of the J = 1/3 anchor pair at ``num_perm``, one per seed.

    Cached because two statistical tests read the same 100-seed sweep.
    """
    estimates = []
    for seed in seeds:
        minhash = MinHash(num_perm=num_perm, seed=seed)
        estimates.append(
            MinHash.estimate(minhash.signature(A_THIRD), minhash.signature(B_THIRD))
        )
    return tuple(estimates)


# ----------------------------------------------------------------------
# Exact Jaccard similarity: the ground truth
# ----------------------------------------------------------------------
class TestJaccard:
    """``jaccard`` against overlaps counted by hand."""

    def test_assignment_anchor_is_exactly_one_third(self) -> None:
        # 50..99 shared (50 tokens); union 0..149 (150 tokens). 50/150 = 1/3.
        assert jaccard(set(range(0, 100)), set(range(50, 150))) == 1 / 3

    def test_ranges_and_generators_are_accepted(self) -> None:
        assert jaccard(range(0, 100), range(50, 150)) == 1 / 3
        assert jaccard((x for x in range(0, 100)), iter(range(50, 150))) == 1 / 3

    def test_identical_sets_are_one(self) -> None:
        assert jaccard({"x", "y", "z"}, {"z", "y", "x"}) == 1.0
        assert jaccard(range(10), list(range(10))) == 1.0

    def test_disjoint_sets_are_zero(self) -> None:
        assert jaccard({1, 2}, {3, 4}) == 0.0
        assert jaccard(DISJOINT_A, DISJOINT_B) == 0.0

    @pytest.mark.parametrize(
        "a, b",
        [
            pytest.param(set(), set(), id="set-set"),
            pytest.param([], set(), id="list-set"),
            pytest.param("", (), id="str-tuple"),
        ],
    )
    def test_two_empty_sets_are_one_by_convention(self, a: Any, b: Any) -> None:
        assert jaccard(a, b) == 1.0

    def test_one_empty_set_is_zero(self) -> None:
        # Union has 3 tokens, intersection none: 0/3, not the 0/0 convention.
        assert jaccard(set(), {1, 2, 3}) == 0.0
        assert jaccard({1, 2, 3}, []) == 0.0

    def test_duplicates_count_once(self) -> None:
        # As sets: {1, 2} and {2, 3}. Shared {2}, union {1, 2, 3}: 1/3.
        assert jaccard([1, 1, 2, 2, 2], [2, 3, 3]) == 1 / 3

    def test_subset(self) -> None:
        # {0..4} inside {0..9}: 5 shared out of 10.
        assert jaccard(range(10), range(5)) == 0.5

    def test_symmetric(self) -> None:
        assert jaccard(range(0, 7), range(3, 20)) == jaccard(range(3, 20), range(0, 7))
        # 3..6 shared (4 tokens), union 0..19 (20 tokens).
        assert jaccard(range(0, 7), range(3, 20)) == 4 / 20

    def test_returns_float(self) -> None:
        assert isinstance(jaccard({1}, {1}), float)
        assert isinstance(jaccard({1}, {2}), float)
        assert isinstance(jaccard(set(), set()), float)

    def test_one_and_string_one_are_different_tokens(self) -> None:
        assert jaccard({1}, {"1"}) == 0.0

    @pytest.mark.parametrize("a, b, shared", CONTROLLED_PAIRS)
    def test_controlled_pairs_hit_their_designed_value(
        self, a: range, b: range, shared: int
    ) -> None:
        assert len(set(a) | set(b)) == 100
        assert len(set(a) & set(b)) == shared
        assert jaccard(a, b) == shared / 100

    def test_unhashable_token_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            jaccard([[1, 2]], [[1, 2]])

    def test_non_iterable_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            jaccard(5, {1})


# ----------------------------------------------------------------------
# Shingles
# ----------------------------------------------------------------------
class TestShingles:
    """Word n-grams: lowercase, split on every non-letter, join with one space."""

    def test_assignment_anchor(self) -> None:
        assert shingles("the cat sat on the mat", 3) == {
            "the cat sat",
            "cat sat on",
            "sat on the",
            "on the mat",
        }

    def test_default_n_is_three(self) -> None:
        assert shingles("the cat sat on the mat") == shingles("the cat sat on the mat", 3)

    def test_lowercases(self) -> None:
        assert shingles("The CAT sAt", 3) == {"the cat sat"}
        assert shingles("THE CAT SAT ON THE MAT", 3) == shingles("the cat sat on the mat", 3)

    @pytest.mark.parametrize(
        "text, n, expected",
        [
            pytest.param("Hello, WORLD!", 3, {"hello world"}, id="comma-bang"),
            pytest.param(
                "cat,sat;on.the-mat",
                2,
                {"cat sat", "sat on", "on the", "the mat"},
                id="punctuation-only-separators",
            ),
            pytest.param("Don't panic", 2, {"don t", "t panic"}, id="apostrophe"),
            pytest.param("abc123def", 2, {"abc def"}, id="digits"),
            pytest.param("snake_case_name", 3, {"snake case name"}, id="underscores"),
            pytest.param(
                "the\tcat\n\nsat   down",
                3,
                {"the cat sat", "cat sat down"},
                id="whitespace",
            ),
            pytest.param("...the...cat...", 2, {"the cat"}, id="leading-trailing"),
        ],
    )
    def test_splits_on_every_non_letter(
        self, text: str, n: int, expected: set
    ) -> None:
        assert shingles(text, n) == expected

    def test_unicode_letters_are_letters(self) -> None:
        # "Cafe Creme" with accents, written as escapes to keep the file ASCII.
        text = "Caf\u00e9 CR\u00c8ME br\u00fbl\u00e9e"
        assert shingles(text, 3) == {"caf\u00e9 cr\u00e8me br\u00fbl\u00e9e"}

    def test_n_equal_one_gives_distinct_words(self) -> None:
        assert shingles("a b a c b", 1) == {"a", "b", "c"}

    def test_repeated_shingles_collapse(self) -> None:
        # Five 2-grams: "a b", "b a", "a b", "b a", "a b" -> two distinct.
        assert shingles("a b a b a b", 2) == {"a b", "b a"}

    def test_exactly_n_words_is_one_shingle(self) -> None:
        assert shingles("one two three", 3) == {"one two three"}

    def test_fewer_than_n_words_is_one_shingle_of_all_words(self) -> None:
        assert shingles("only two", 3) == {"only two"}
        assert shingles("solo", 5) == {"solo"}

    @pytest.mark.parametrize("text", ["", "   ", "  42 ... ", "123-456_789", "!?"])
    def test_no_words_gives_empty_set(self, text: str) -> None:
        assert shingles(text, 3) == set()

    def test_distinct_words_give_length_minus_n_plus_one(self) -> None:
        # The ten words of DOC_SPEED are all distinct: 10 - 3 + 1 = 8.
        assert len(shingles(DOC_SPEED, 3)) == 8
        assert len(shingles(DOC_SPEED, 1)) == 10
        assert len(shingles(DOC_SPEED, 10)) == 1

    def test_one_word_edit_gives_hand_counted_jaccard(self) -> None:
        a, b = shingles(DOC_SPEED, 3), shingles(DOC_TIME, 3)
        assert a - b == {"certainty for speed", "for speed in", "speed in a"}
        assert b - a == {"certainty for time", "for time in", "time in a"}
        assert a & b == {
            "randomized algorithms trade",
            "algorithms trade certainty",
            "trade certainty for",
            "in a controlled",
            "a controlled way",
        }
        assert jaccard(a, b) == 5 / 11

    @pytest.mark.parametrize("text", [b"bytes", None, ["a", "b"], 42])
    def test_text_must_be_str(self, text: Any) -> None:
        with pytest.raises(TypeError):
            shingles(text, 3)

    @pytest.mark.parametrize(
        "n, error",
        [
            pytest.param(0, ValueError, id="zero"),
            pytest.param(-2, ValueError, id="negative"),
            pytest.param(True, TypeError, id="bool"),
            pytest.param(2.0, TypeError, id="float"),
            pytest.param("3", TypeError, id="str"),
        ],
    )
    def test_bad_n_raises(self, n: Any, error: type) -> None:
        with pytest.raises(error):
            shingles("the cat sat on the mat", n)


# ----------------------------------------------------------------------
# The MinHash family and its signatures
# ----------------------------------------------------------------------
class TestModulus:
    """``MERSENNE_61`` is the prime 2^61 - 1, so every ``a x + b`` is a bijection."""

    def test_value(self) -> None:
        assert MERSENNE_61 == P == 2_305_843_009_213_693_951

    def test_is_prime(self) -> None:
        assert is_prime_by_bases(MERSENNE_61)

    def test_primality_helper_is_not_vacuous(self) -> None:
        # 2^61 + 1 is divisible by 3 (2 is -1 mod 3, and 61 is odd), and
        # 2^59 - 1 = 179951 * 3203431780337 is a composite Mersenne number.
        assert not is_prime_by_bases(2**61 + 1)
        assert not is_prime_by_bases(2**59 - 1)
        assert 179951 * 3203431780337 == 2**59 - 1
        # 3215031751 fools bases 2, 3, 5 and 7; base 11 exposes it.
        assert not is_prime_by_bases(3215031751)
        assert is_prime_by_bases(2**31 - 1)


class TestSignature:
    """Shape, determinism, and agreement with :func:`reference_signature`."""

    @pytest.mark.parametrize("num_perm", [1, 2, 16, 100, 128])
    def test_length_is_num_perm_and_values_are_in_range(self, num_perm: int) -> None:
        signature = MinHash(num_perm=num_perm, seed=3).signature(A_THIRD)
        assert len(signature) == num_perm
        assert all(isinstance(v, int) and 0 <= v < P for v in signature)

    @pytest.mark.parametrize(
        "tokens, num_perm, seed",
        [
            pytest.param(["apple"], 8, 0, id="single-token"),
            pytest.param({"apple", "banana", "cherry"}, 16, 1, id="fruit"),
            pytest.param(range(0, 40), 32, 7, id="ints"),
            pytest.param(shingles(DOC_SPEED, 3), 24, 2026, id="shingles"),
            pytest.param([(1, "a"), (2, "b")], 8, 5, id="tuples"),
        ],
    )
    def test_matches_reference_built_from_the_specification(
        self, tokens: Iterable[object], num_perm: int, seed: int
    ) -> None:
        tokens = list(tokens)
        assert MinHash(num_perm=num_perm, seed=seed).signature(tokens) == (
            reference_signature(tokens, num_perm, seed)
        )

    def test_unhashable_tokens_work_because_only_repr_is_used(self) -> None:
        tokens = [[1, 2], [3], {"k": "v"}]
        minhash = MinHash(num_perm=8, seed=4)
        assert minhash.signature(tokens) == reference_signature(tokens, 8, 4)
        assert minhash.signature([[3], {"k": "v"}, [1, 2], [3]]) == minhash.signature(tokens)

    def test_order_duplicates_and_container_do_not_matter(self) -> None:
        minhash = MinHash(num_perm=64, seed=11)
        base = minhash.signature(set(range(30)))
        assert minhash.signature(list(range(29, -1, -1))) == base
        assert minhash.signature(list(range(30)) * 3) == base
        assert minhash.signature(frozenset(range(30))) == base
        assert minhash.signature(tuple(range(30))) == base
        assert minhash.signature(x for x in range(30)) == base

    def test_same_instance_signs_the_same_set_the_same_way(self) -> None:
        minhash = MinHash(num_perm=64, seed=12)
        assert minhash.signature(A_THIRD) == minhash.signature(A_THIRD)

    @pytest.mark.parametrize("seed", [0, 1, 42, 2026])
    def test_same_seed_gives_identical_signature(self, seed: int) -> None:
        first = MinHash(num_perm=128, seed=seed).signature(A_THIRD)
        second = MinHash(num_perm=128, seed=seed).signature(A_THIRD)
        assert first == second

    def test_different_seeds_give_different_signatures(self) -> None:
        signatures = [
            tuple(MinHash(num_perm=32, seed=seed).signature(A_THIRD)) for seed in range(6)
        ]
        assert len(set(signatures)) == 6
        # Not just different somewhere: two unrelated hash families agree
        # on a position only by coincidence.
        assert MinHash.estimate(signatures[0], signatures[1]) <= 0.1

    @pytest.mark.parametrize("seed", [0, 5])
    def test_short_signature_is_prefix_of_long_one(self, seed: int) -> None:
        # The (a, b) pairs are drawn one position at a time from one stream,
        # so k = 16 is the first 16 positions of k = 512 with the same seed.
        tokens = shingles(DOC_SPEED, 3)
        long_sig = MinHash(num_perm=512, seed=seed).signature(tokens)
        assert long_sig[:16] == MinHash(num_perm=16, seed=seed).signature(tokens)
        assert long_sig[:100] == MinHash(num_perm=100, seed=seed).signature(tokens)

    def test_union_signature_is_elementwise_minimum(self) -> None:
        # min over A | B = min(min over A, min over B), position by position.
        minhash = MinHash(num_perm=64, seed=21)
        sig_a, sig_b = minhash.signature(A_THIRD), minhash.signature(B_THIRD)
        sig_union = minhash.signature(A_THIRD | B_THIRD)
        assert sig_union == [min(x, y) for x, y in zip(sig_a, sig_b)]

    def test_superset_minimum_never_exceeds_subset_minimum(self) -> None:
        minhash = MinHash(num_perm=64, seed=22)
        sig_big = minhash.signature(range(0, 100))
        sig_small = minhash.signature(range(20, 40))
        assert all(big <= small for big, small in zip(sig_big, sig_small))

    @pytest.mark.parametrize(
        "tokens",
        [
            pytest.param(set(), id="set"),
            pytest.param([], id="list"),
            pytest.param(iter(()), id="exhausted-iterator"),
            pytest.param(shingles(" 42 ... ", 3), id="wordless-text"),
        ],
    )
    def test_empty_set_has_no_signature(self, tokens: Iterable[object]) -> None:
        with pytest.raises(ValueError):
            MinHash(num_perm=8, seed=0).signature(tokens)

    def test_non_iterable_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            MinHash(num_perm=8, seed=0).signature(5)


class TestConstruction:
    """Constructor contract: defaults, properties, and validation."""

    def test_defaults(self) -> None:
        minhash = MinHash()
        assert minhash.num_perm == 128
        assert minhash.seed == 0
        assert len(minhash.signature({"x"})) == 128

    def test_properties_report_arguments(self) -> None:
        minhash = MinHash(num_perm=48, seed=9)
        assert (minhash.num_perm, minhash.seed) == (48, 9)

    def test_seed_none_is_still_self_consistent(self) -> None:
        # The parameters come from the operating system, so nothing here
        # depends on what they are: one instance must still sign the same
        # set the same way, and identical sets must still estimate 1.0.
        minhash = MinHash(num_perm=32, seed=None)
        assert minhash.seed is None
        sig = minhash.signature(A_THIRD)
        assert sig == minhash.signature(sorted(A_THIRD, reverse=True))
        assert MinHash.estimate(sig, minhash.signature(A_THIRD)) == 1.0

    @pytest.mark.parametrize(
        "num_perm, error",
        [
            pytest.param(0, ValueError, id="zero"),
            pytest.param(-5, ValueError, id="negative"),
            pytest.param(True, TypeError, id="bool"),
            pytest.param(16.0, TypeError, id="float"),
            pytest.param("16", TypeError, id="str"),
            pytest.param(None, TypeError, id="none"),
        ],
    )
    def test_bad_num_perm_raises(self, num_perm: Any, error: type) -> None:
        with pytest.raises(error):
            MinHash(num_perm=num_perm, seed=0)

    @pytest.mark.parametrize("seed", ["1", 1.5, True, [1]])
    def test_bad_seed_raises_type_error(self, seed: Any) -> None:
        with pytest.raises(TypeError):
            MinHash(num_perm=8, seed=seed)


# ----------------------------------------------------------------------
# Comparing signatures
# ----------------------------------------------------------------------
class TestEstimate:
    """``MinHash.estimate`` is the fraction of equal positions."""

    def test_hand_example(self) -> None:
        # Positions 0 and 2 agree, 1 and 3 do not: 2 of 4.
        assert MinHash.estimate([1, 2, 3, 4], [1, 9, 3, 9]) == 0.5

    def test_all_equal_and_none_equal(self) -> None:
        assert MinHash.estimate([5, 6, 7], [5, 6, 7]) == 1.0
        assert MinHash.estimate([5, 6, 7], [8, 9, 10]) == 0.0

    def test_one_of_three(self) -> None:
        assert MinHash.estimate((1, 2, 3), (0, 2, 0)) == 1 / 3

    def test_callable_on_class_and_instance(self) -> None:
        assert MinHash(num_perm=4, seed=0).estimate([1, 2], [1, 3]) == 0.5

    @pytest.mark.parametrize(
        "sig_a, sig_b",
        [
            pytest.param([1, 2, 3], [1, 2], id="3-vs-2"),
            pytest.param([1], [], id="1-vs-0"),
            pytest.param([], [1, 2], id="0-vs-2"),
        ],
    )
    def test_different_lengths_raise(self, sig_a: List[int], sig_b: List[int]) -> None:
        with pytest.raises(ValueError, match="same length"):
            MinHash.estimate(sig_a, sig_b)

    def test_signatures_from_different_k_are_rejected(self) -> None:
        tokens = shingles(DOC_SPEED, 3)
        with pytest.raises(ValueError):
            MinHash.estimate(
                MinHash(num_perm=16, seed=0).signature(tokens),
                MinHash(num_perm=32, seed=0).signature(tokens),
            )

    def test_two_empty_signatures_raise(self) -> None:
        with pytest.raises(ValueError):
            MinHash.estimate([], [])

    @pytest.mark.parametrize("num_perm", [16, 64, 100])
    def test_estimate_is_a_count_over_k(self, num_perm: int) -> None:
        estimate = estimate_similarity(A_THIRD, B_THIRD, num_perm=num_perm, seed=4)
        assert 0.0 <= estimate <= 1.0
        assert estimate in {m / num_perm for m in range(num_perm + 1)}


class TestEstimateSimilarity:
    """The one-call helper is the class path with the same parameters."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_equals_class_path(self, seed: int) -> None:
        minhash = MinHash(num_perm=64, seed=seed)
        expected = MinHash.estimate(minhash.signature(A_THIRD), minhash.signature(B_THIRD))
        assert estimate_similarity(A_THIRD, B_THIRD, num_perm=64, seed=seed) == expected

    def test_defaults_are_k128_seed0(self) -> None:
        minhash = MinHash(num_perm=128, seed=0)
        expected = MinHash.estimate(minhash.signature(A_THIRD), minhash.signature(B_THIRD))
        assert estimate_similarity(A_THIRD, B_THIRD) == expected

    @pytest.mark.parametrize(
        "a, b",
        [
            pytest.param(A_THIRD, [], id="second-empty"),
            pytest.param(set(), B_THIRD, id="first-empty"),
            pytest.param(set(), set(), id="both-empty"),
        ],
    )
    def test_empty_input_raises(self, a: Iterable[object], b: Iterable[object]) -> None:
        # jaccard calls two empty sets 1.0, but MinHash cannot sign them.
        with pytest.raises(ValueError):
            estimate_similarity(a, b, num_perm=8, seed=0)

    def test_bad_num_perm_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_similarity(A_THIRD, B_THIRD, num_perm=0)
        with pytest.raises(TypeError):
            estimate_similarity(A_THIRD, B_THIRD, num_perm=True)


# ----------------------------------------------------------------------
# The cases MinHash gets exactly right
# ----------------------------------------------------------------------
class TestExactCases:
    """No tolerance needed: identical sets and disjoint sets."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 2026])
    @pytest.mark.parametrize("num_perm", [1, 16, 64, 512])
    def test_identical_sets_estimate_exactly_one(self, num_perm: int, seed: int) -> None:
        # Same tokens, presented differently: the minima cannot differ.
        same_again = sorted(A_THIRD, reverse=True) + [0, 1, 2]
        assert estimate_similarity(A_THIRD, same_again, num_perm=num_perm, seed=seed) == 1.0

    @pytest.mark.parametrize("seed", [0, 3])
    def test_identical_documents_estimate_exactly_one(self, seed: int) -> None:
        a = shingles(DOC_SPEED, 3)
        b = shingles(DOC_SPEED.upper() + "!", 3)
        assert a == b
        assert estimate_similarity(a, b, num_perm=128, seed=seed) == 1.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_disjoint_sets_estimate_at_most_two_percent_at_k512(self, seed: int) -> None:
        estimate = estimate_similarity(DISJOINT_A, DISJOINT_B, num_perm=512, seed=seed)
        assert estimate <= 0.02

    def test_disjoint_strings_estimate_at_most_two_percent_at_k512(self) -> None:
        estimate = estimate_similarity(
            {"alpha", "beta", "gamma"}, {"delta", "epsilon"}, num_perm=512, seed=8
        )
        assert estimate <= 0.02

    def test_one_and_string_one_are_different_tokens(self) -> None:
        # repr(1) is "1" and repr("1") is "'1'", so these are disjoint sets.
        assert estimate_similarity({1}, {"1"}, num_perm=512, seed=0) <= 0.02


# ----------------------------------------------------------------------
# Accuracy: fixed seeds, tolerances of at least five standard errors
# ----------------------------------------------------------------------
class TestAccuracy:
    """Estimates against the hand-counted J, with the error budget in the open.

    One estimate at ``k`` positions has standard error ``sqrt(J (1 - J) / k)``.
    For the anchor pair at ``k = 512`` that is ``sqrt((2/9) / 512) = 0.0208``,
    and the assignment's 0.10 tolerance is 4.8 of those. Every seed is fixed.
    """

    def test_standard_error_anchor(self) -> None:
        assert round(standard_error(J_THIRD, 512), 4) == 0.0208
        assert 0.10 / standard_error(J_THIRD, 512) > 4.7

    @pytest.mark.parametrize("seed", range(10))
    def test_anchor_pair_within_point_one_at_k512(self, seed: int) -> None:
        minhash = MinHash(num_perm=512, seed=seed)
        estimate = MinHash.estimate(minhash.signature(A_THIRD), minhash.signature(B_THIRD))
        assert abs(estimate - J_THIRD) <= 0.10

    @pytest.mark.parametrize("seed", [0, 1])
    @pytest.mark.parametrize("a, b, shared", CONTROLLED_PAIRS)
    def test_controlled_pairs_within_five_standard_errors_at_k512(
        self, a: range, b: range, shared: int, seed: int
    ) -> None:
        j = shared / 100
        estimate = estimate_similarity(a, b, num_perm=512, seed=seed)
        assert abs(estimate - j) <= 5 * standard_error(j, 512)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_one_word_edit_within_five_standard_errors_at_k512(self, seed: int) -> None:
        # J = 5/11 by hand (see DOC_SPEED); SE = sqrt((30/121) / 512) = 0.0220.
        j = 5 / 11
        estimate = estimate_similarity(
            shingles(DOC_SPEED, 3), shingles(DOC_TIME, 3), num_perm=512, seed=seed
        )
        assert abs(estimate - j) <= 5 * standard_error(j, 512)

    def test_mean_absolute_error_shrinks_from_k16_to_k512(self) -> None:
        # Monotonicity, not a value. Theory puts the two MAEs near
        # 0.8 * 0.118 = 0.094 and 0.8 * 0.0208 = 0.017, far apart.
        seeds = tuple(range(30))
        mae = {
            k: fmean(abs(e - J_THIRD) for e in third_estimates(k, seeds))
            for k in (16, 512)
        }
        assert mae[512] < mae[16]

    def test_mean_estimate_over_100_seeds_is_unbiased(self) -> None:
        # One k = 64 estimate has SE 0.0589; the mean of 100 independent
        # seeds has SE 0.00589. The 0.02 window is 3.4 of those, and the
        # seeds are fixed, so the outcome is decided once and cannot flake.
        estimates = third_estimates(64, tuple(range(100)))
        assert abs(fmean(estimates) - J_THIRD) <= 0.02

    def test_error_is_two_sided(self) -> None:
        # Monte Carlo with two-sided error: over the same 100 seeds some
        # estimates land above 1/3 and some below. (No k = 64 estimate can
        # equal 1/3, which is not a multiple of 1/64.)
        estimates = third_estimates(64, tuple(range(100)))
        assert min(estimates) < J_THIRD < max(estimates)

    def test_seeded_rng_draws_feed_reproducible_estimates(self) -> None:
        # Seeds for a sweep can themselves come from a seeded generator,
        # and the sweep is then reproducible end to end.
        def sweep() -> List[float]:
            rng = make_rng(77)
            return [
                estimate_similarity(A_THIRD, B_THIRD, num_perm=32, seed=rng.randrange(2**32))
                for _ in range(5)
            ]

        first = sweep()
        assert first == sweep()
        assert all(abs(e - J_THIRD) <= 5 * standard_error(J_THIRD, 32) for e in first)


# ----------------------------------------------------------------------
# Reproducibility across interpreter processes
# ----------------------------------------------------------------------
CHILD_CODE = """
import json
import sys
from src.randomized.minhash_similarity import MinHash, shingles
tokens = shingles(sys.argv[1], 3)
print(json.dumps({
    "order": list(tokens),
    "signature": MinHash(num_perm=64, seed=7).signature(tokens),
}))
"""

#: Two different string-hash salts for the child interpreters.
CHILD_HASH_SEEDS: Tuple[str, ...] = ("1", "2")


@pytest.fixture(scope="module")
def child_runs() -> Dict[str, Dict[str, Any]]:
    """Run :data:`CHILD_CODE` once per ``PYTHONHASHSEED`` and parse its JSON."""
    runs: Dict[str, Dict[str, Any]] = {}
    for hash_seed in CHILD_HASH_SEEDS:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD_CODE, CHILD_TEXT],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        runs[hash_seed] = json.loads(completed.stdout)
    return runs


class TestCrossProcessReproducibility:
    """A signature must not depend on the interpreter's string-hash salt."""

    @pytest.mark.parametrize("hash_seed", CHILD_HASH_SEEDS)
    def test_child_signature_equals_local_signature(
        self, child_runs: Dict[str, Dict[str, Any]], hash_seed: str
    ) -> None:
        local = MinHash(num_perm=64, seed=7).signature(shingles(CHILD_TEXT, 3))
        assert child_runs[hash_seed]["signature"] == local

    def test_children_really_hashed_strings_differently(
        self, child_runs: Dict[str, Dict[str, Any]]
    ) -> None:
        # Precondition that keeps the test above honest: the two children
        # iterate the same set of shingles in different orders, which can
        # only happen if their string hashing differed. A signature built
        # on the built-in hash would have differed with it.
        orders = [child_runs[s]["order"] for s in CHILD_HASH_SEEDS]
        assert sorted(orders[0]) == sorted(orders[1]) == sorted(shingles(CHILD_TEXT, 3))
        assert orders[0] != orders[1]
