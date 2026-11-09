"""Week 11 tests: Rabin-Karp and the rolling hash on the spec 9.2 anchors.

Each expected value is worked out by hand, so the tests do not repeat the code
under test.

Spurious hits with q = 7, base 256: 256 = 4 (mod 7), so the digit weights
base^3, base^2, base, 1 are 1, 2, 4, 1 and a 4-character window w hashes to
w0 + 2 w1 + 4 w2 + w3 (mod 7).

- abracadabra / abra: a, b, c, d, r are 6, 0, 1, 2, 2 mod 7. "abra" hashes to
  6 + 0 + 8 + 6 = 20 = 6. The windows abra, brac, raca, acad, cada, adab, dabr,
  abra hash to 6, 1, 3, 6, 6, 6, 2, 6: five hits, two real, so 3 spurious.
- mississippi / issi: m, i, s, p are 4, 0, 3, 0 mod 7. "issi" hashes to
  0 + 6 + 12 + 0 = 18 = 4. The windows miss, issi, ssis, siss, issi, ssip,
  sipp, ippi hash to 5, 4, 5, 4, 4, 2, 3, 0: "siss" is the 1 spurious hit.

Base 1 makes the hash the sum of the code points, so in "abccba" the window
"cba" (sum 294) collides with "abc" while "bcc" and "ccb" (sum 296) do not:
2 hits, 1 spurious, found at [0]. The spurious check fails at its first
character, so the comparisons are 3 + 1 = 4.

On "a" * 100 with "a" * 5 all 96 windows equal the pattern, so every window is
a hash hit and is verified in full: 96 * 5 = 480 comparisons.
"""

from __future__ import annotations

import pytest

from src.string_search_engine.core.rabin_karp import (
    RKStats,
    rabin_karp_multi,
    rabin_karp_search,
)
from src.string_search_engine.utils.rolling_hash import RollingHash
from src.utils.random_utils import derive_seed, make_rng

MERSENNE_61 = (1 << 61) - 1


# --------------------------------------------------------------- RollingHash


def test_rolling_hash_defaults_and_definition():
    rh = RollingHash()
    assert rh.base == 256
    assert rh.modulus == MERSENNE_61
    # "ab" = 97 * 256 + 98 = 24930, far below the modulus.
    assert rh.hash("ab") == 24930
    assert rh.hash("") == 0
    # 256 = 4, 256^2 = 2, 256^3 = 1 (mod 7).
    small = RollingHash(256, 7)
    assert [small.power(m) for m in (1, 2, 3, 4)] == [1, 4, 2, 1]
    assert small.hash("abra") == 6
    assert small.hash("issi") == 4


@pytest.mark.parametrize("base, modulus", [(0, 101), (-3, 101), (256, 1), (256, 0)])
def test_rolling_hash_rejects_bad_parameters(base, modulus):
    with pytest.raises(ValueError):
        RollingHash(base, modulus)


@pytest.mark.parametrize("modulus", [7, 13, 101, MERSENNE_61])
def test_roll_agrees_with_hash_on_every_window(modulus):
    rng = make_rng(derive_seed(11, "rk", "roll", modulus))
    text = "".join(rng.choice("abcdefghé中") for _ in range(300))
    for m in (1, 2, 5, 17):
        rh = RollingHash(256, modulus)
        h = rh.hash(text[:m])
        for i in range(1, len(text) - m + 1):
            h = rh.roll(h, text[i - 1], text[i + m - 1], m)
            assert h == rh.hash(text[i:i + m])


def test_base_one_makes_anagram_windows_collide():
    rh = RollingHash(base=1, modulus=MERSENNE_61)
    assert rh.hash("abc") == rh.hash("cab") == rh.hash("bca") == 294
    positions, st = rabin_karp_search("abccba", "abc", base=1, stats=True)
    assert positions == [0]
    assert st == RKStats(hash_hits=2, spurious_hits=1, verifications=2, comparisons=4)


# --------------------------------------------------------- rabin_karp_search


@pytest.mark.parametrize("modulus", [7, 13, 101, MERSENNE_61])
def test_abracadabra_positions_for_every_modulus(modulus):
    assert rabin_karp_search("abracadabra", "abra", modulus=modulus) == [0, 7]
    positions, _ = rabin_karp_search("abracadabra", "abra", modulus=modulus, stats=True)
    assert positions == [0, 7]


@pytest.mark.parametrize("modulus, spurious", [(7, 3), (13, 0), (101, 0), (MERSENNE_61, 0)])
def test_abracadabra_spurious_hits(modulus, spurious):
    _, st = rabin_karp_search("abracadabra", "abra", modulus=modulus, stats=True)
    assert st.spurious_hits == spurious
    assert st.hash_hits == 2 + spurious
    assert st.verifications == st.hash_hits


@pytest.mark.parametrize("modulus, spurious", [(7, 1), (13, 0)])
def test_mississippi_spurious_hits(modulus, spurious):
    positions, st = rabin_karp_search("mississippi", "issi", modulus=modulus, stats=True)
    assert positions == [1, 4]
    assert st.spurious_hits == spurious
    assert st.hash_hits == 2 + spurious


def test_mississippi_default_modulus():
    assert rabin_karp_search("mississippi", "issi") == [1, 4]


def test_worst_case_every_window_hits():
    positions, st = rabin_karp_search("a" * 100, "a" * 5, stats=True)
    assert positions == list(range(96))
    assert st.hash_hits == 96
    assert st.verifications == 96
    assert st.spurious_hits == 0
    assert st.comparisons == 96 * 5


def test_overlapping_and_unicode():
    assert rabin_karp_search("aaaaa", "aa") == [0, 1, 2, 3]
    assert rabin_karp_search("abababab", "abab") == [0, 2, 4]
    assert rabin_karp_search("naïve café naïve", "naïve") == [0, 11]


def test_shared_conventions():
    with pytest.raises(ValueError):
        rabin_karp_search("abc", "")
    with pytest.raises(ValueError):
        rabin_karp_search("", "")
    assert rabin_karp_search("ab", "abc") == []
    assert rabin_karp_search("", "a") == []
    assert rabin_karp_search("ab", "abc", stats=True) == ([], RKStats(0, 0, 0, 0))


# ---------------------------------------------------------- rabin_karp_multi


def test_multi_anchor():
    found = rabin_karp_multi("abracadabra", ["abra", "cad", "bra", "zzz"])
    assert found == {"abra": [0, 7], "cad": [4], "bra": [1, 8], "zzz": []}


def test_multi_anchor_with_small_modulus_and_stats():
    found, st = rabin_karp_multi(
        "abracadabra", ["abra", "cad", "bra", "zzz"], modulus=7, stats=True
    )
    assert found == {"abra": [0, 7], "cad": [4], "bra": [1, 8], "zzz": []}
    assert isinstance(st, RKStats)
    assert st.verifications == st.hash_hits
    # Five real matches are among the hits; the rest are spurious.
    assert st.hash_hits - st.spurious_hits == 5


def test_multi_conventions():
    assert rabin_karp_multi("abc", []) == {}
    assert rabin_karp_multi("ab", ["abc", "b"]) == {"abc": [], "b": [1]}
    assert rabin_karp_multi("", ["a"]) == {"a": []}
    with pytest.raises(ValueError):
        rabin_karp_multi("abc", ["a", ""])
