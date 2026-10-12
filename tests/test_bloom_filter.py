"""Tests for the Week 7 Bloom filter: one-sided error, sizing, and reproducible bits.

A Bloom filter makes exactly one promise: an item that was added always
checks True. Everything else it says comes with a probability attached.
So this file proves the promise outright, and checks every probabilistic
claim against numbers worked out by hand rather than numbers read back
from the module:

* **No false negatives.** Every added item checks True: 1,000 items of six
  kinds (ints, strings, floats, tuples, unhashable lists and dicts, and a
  mix) under five seeds, plus tiny, wrapping and badly overfilled
  geometries. This is the one guarantee, so it gets the widest battery.
* **The sizing anchor.** ``from_capacity(1000, 0.01)`` must give
  ``m = 9586`` and ``k = 7``, from ``m = ceil(-n ln p / (ln 2)^2)`` and
  ``k = round((m / n) ln 2)``, worked by hand in :class:`TestSizing`. At
  ``n = 1000`` the textbook rate ``(1 - e^(-k n / m))^k`` is about
  ``0.01003`` (``0.010035`` to six places), also worked by hand.
* **The measured rate.** Over 100,000 strings that were never added, the
  observed false-positive rate at that configuration lies in
  ``[0.005, 0.02]``. One binomial standard error at ``p = 0.01`` and
  ``N = 100,000`` is about ``0.0003``, so that band is more than fifteen
  standard errors wide on each side.
* **The documented hashing.** The module docstring specifies the exact
  probe sequence (one salted blake2b digest split into ``h1`` and an odd
  ``h2``, then ``(h1 + i h2) mod m``) and the bit layout (bit ``j`` in byte
  ``j >> 3`` at position ``j & 7``). :func:`documented_probes` re-derives
  both from that text with :mod:`hashlib`, sharing no code with the module,
  and the set bits must match exactly.
* **Reproducibility across processes.** The same filter is built in child
  interpreters started with ``PYTHONHASHSEED`` set to 1 and to 2, and their
  bit arrays must equal the one built here. Python salts the built-in hash
  function of a ``str`` per process, so a filter that used it would fail
  this test. A companion test proves the two children really did salt
  their ``str`` hashes differently, so the agreement is not an accident.

Every draw is seeded. No assertion is about time. Probabilistic assertions
use fixed seeds and tolerances of at least five standard errors, so the
suite cannot flake.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

from src.randomized.bloom_filter import BloomFilter
from src.utils.random_utils import make_rng, random_distinct_list

#: The repository root, so child interpreters can import ``src``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The seeds every determinism and no-false-negative battery runs under.
#: ``2**128 - 1`` is the largest seed the 16-byte blake2b salt can hold.
SEEDS: List[int] = [0, 1, 42, 2026, 2**128 - 1]

#: The kinds of item the filter is fed. "unhashable" holds lists and dicts,
#: which a filter built on the built-in hash function could not accept.
ITEM_KINDS: List[str] = ["ints", "strings", "floats", "tuples", "unhashable", "mixed"]

#: Non-members probed when measuring the false-positive rate.
FP_PROBES = 100_000


# ----------------------------------------------------------------------
# Independent helpers
# ----------------------------------------------------------------------
def set_bits(bits: bytes) -> Set[int]:
    """Return the indexes of the 1 bits, read with the documented layout.

    Bit ``j`` lives in byte ``j >> 3`` at position ``j & 7``, as the module
    docstring states. Written here from that sentence alone.

    Args:
        bits: A bit array as returned by :attr:`BloomFilter.bits`.

    Returns:
        Every ``j`` whose bit is 1.
    """
    return {
        8 * byte_index + position
        for byte_index, byte in enumerate(bits)
        for position in range(8)
        if (byte >> position) & 1
    }


def documented_probes(item: Any, m: int, k: int, seed: int) -> List[int]:
    """Re-derive the ``k`` probe indexes from the module docstring's formula.

    ``digest = blake2b(repr(item).encode("utf-8"), digest_size=16,
    salt=seed.to_bytes(16, "little"))``; ``h1`` is the first 8 bytes and
    ``h2`` the last 8 with the low bit forced to 1, both little-endian; the
    probes are ``(h1 + i * h2) mod m`` for ``i = 0 .. k - 1``. This shares
    no code with the module, so agreement means the filter does what its
    documentation says.

    Args:
        item: The item to probe.
        m: Number of bits.
        k: Number of probes.
        seed: The filter seed (the blake2b salt).

    Returns:
        The ``k`` probe indexes, repeats included.
    """
    digest = hashlib.blake2b(
        repr(item).encode("utf-8"), digest_size=16, salt=seed.to_bytes(16, "little")
    ).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little") | 1
    return [(h1 + i * h2) % m for i in range(k)]


def make_items(kind: str, n: int, seed: int) -> List[Any]:
    """Build ``n`` items of one kind, in an order fixed by ``seed``.

    Args:
        kind: One of :data:`ITEM_KINDS`.
        n: How many items.
        seed: Seeds the shuffle through :func:`make_rng` (ints use
            :func:`random_distinct_list`).

    Returns:
        A new list of ``n`` items.
    """
    rng = make_rng(seed)
    if kind == "ints":
        return [value * 7919 - 3_000_000 for value in random_distinct_list(n, seed)]
    if kind == "strings":
        items: List[Any] = [f"word-{i}-{rng.randint(0, 10**9)}" for i in range(n)]
    elif kind == "floats":
        items = [i / 7 - 50.0 for i in range(n)]
    elif kind == "tuples":
        items = [(i, f"t{i}", -i) for i in range(n)]
    elif kind == "unhashable":
        items = [[i, i + 1] if i % 2 else {"id": i, "tag": "x"} for i in range(n)]
    elif kind == "mixed":
        makers = [
            lambda i: i,
            lambda i: f"s-{i}",
            lambda i: (i, "pair"),
            lambda i: i + 0.5,
            lambda i: f"b{i}".encode("ascii"),
            lambda i: [i, None],
        ]
        items = [makers[i % len(makers)](i) for i in range(n)]
    else:  # pragma: no cover - a typo in a parametrize list
        raise ValueError(f"unknown kind {kind!r}")
    rng.shuffle(items)
    return items


def filled(m: int, k: int, seed: int, items: List[Any]) -> BloomFilter:
    """Return a ``BloomFilter(m, k, seed)`` with every item in ``items`` added."""
    bloom = BloomFilter(m, k, seed=seed)
    for item in items:
        bloom.add(item)
    return bloom


@functools.lru_cache(maxsize=None)
def measured_rate(
    expected_items: int, fp_rate: float, seed: int, probes: int
) -> Tuple[int, float, int, int]:
    """Fill a filter to capacity and count false positives among non-members.

    Members are ``"member-0" .. "member-{n-1}"``; probes are
    ``"probe-0" .. "probe-{probes-1}"``. The prefixes differ, so no probe
    was ever added and every True is a false positive. Cached because
    several tests read the same 100,000-probe measurement; an immutable
    tuple is returned so no test can disturb another's filter.

    Returns:
        ``(false_positives, fill_ratio, m, k)``.
    """
    bloom = BloomFilter.from_capacity(expected_items, fp_rate, seed=seed)
    for i in range(expected_items):
        bloom.add(f"member-{i}")
    false_positives = sum(bloom.check(f"probe-{i}") for i in range(probes))
    return false_positives, bloom.fill_ratio, bloom.m, bloom.k


def binomial_se(p: float, trials: int) -> float:
    """One standard error of an observed proportion: ``sqrt(p (1 - p) / N)``."""
    return math.sqrt(p * (1.0 - p) / trials)


def fp_standard_error(m: int, k: int, n: int, probes: int) -> float:
    """One standard error of a measured false-positive rate around the formula.

    Two sources of noise: the binomial noise of ``probes`` checks, and the
    filter-to-filter spread of the fill ratio ``f``. The count of set bits
    has variance at most ``m / 4``, so ``f`` has standard error at most
    ``0.5 / sqrt(m)``, and since the rate is about ``f^k`` that moves it by
    ``k f^(k - 1)`` times as much. The two are added in quadrature.

    Args:
        m: Number of bits.
        k: Number of probes.
        n: Number of items added.
        probes: Number of non-members checked.

    Returns:
        A conservative standard error for the measured rate.
    """
    fill = 1.0 - math.exp(-k * n / m)
    rate = fill**k
    fill_part = k * fill ** (k - 1) * 0.5 / math.sqrt(m)
    return math.sqrt(binomial_se(rate, probes) ** 2 + fill_part**2)


# ----------------------------------------------------------------------
# The one guarantee: no false negatives
# ----------------------------------------------------------------------
class TestNoFalseNegatives:
    """Every item that was added checks True, always.

    Bits are only ever set, and an added item set all ``k`` of its own, so
    it cannot find a 0 among them later. No tolerance is needed: this is a
    certainty, and a single False anywhere is a bug.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("kind", ITEM_KINDS)
    def test_one_thousand_items_all_check_true(self, kind: str, seed: int) -> None:
        items = make_items(kind, 1000, seed)
        bloom = BloomFilter.from_capacity(1000, 0.01, seed=seed)
        for item in items:
            bloom.add(item)
        missing = [item for item in items if not bloom.check(item)]
        assert missing == []
        assert all(item in bloom for item in items)
        assert len(bloom) == 1000

    @pytest.mark.parametrize(
        "m, k",
        [
            pytest.param(1, 1, id="m1-k1"),
            pytest.param(1, 5, id="m1-k5"),
            pytest.param(7, 3, id="m7-k3"),
            pytest.param(13, 20, id="k-exceeds-m"),
            pytest.param(64, 8, id="m64-k8"),
            pytest.param(1024, 8, id="power-of-two"),
            pytest.param(9586, 7, id="anchor-geometry"),
        ],
    )
    def test_tiny_wrapping_and_power_of_two_geometries(self, m: int, k: int) -> None:
        # k > m forces the probe index to wrap round the array many times.
        items = make_items("mixed", 300, seed=m * 100 + k)
        bloom = filled(m, k, 5, items)
        assert all(bloom.check(item) for item in items)

    def test_overfilled_filter_still_has_none(self) -> None:
        # Fifty times the capacity: the false-positive rate is awful, but an
        # added item is still never reported missing.
        items = make_items("strings", 5000, seed=11)
        bloom = BloomFilter.from_capacity(100, 0.01, seed=3)
        for item in items:
            bloom.add(item)
        assert all(bloom.check(item) for item in items)

    def test_items_added_early_survive_later_adds(self) -> None:
        bloom = BloomFilter.from_capacity(500, 0.01, seed=8)
        added: List[str] = []
        for i in range(500):
            item = f"step-{i}"
            bloom.add(item)
            added.append(item)
            if i % 50 == 49:
                assert all(bloom.check(old) for old in added)

    def test_every_sample_array_has_no_false_negatives(
        self, sample_arrays: Dict[str, List[Any]]
    ) -> None:
        for name, values in sample_arrays.items():
            bloom = BloomFilter(2048, 5, seed=13)
            for value in values:
                bloom.add(value)
            assert all(value in bloom for value in values), name
            assert len(bloom) == len(values), name
            if not values:
                assert set_bits(bloom.bits) == set(), name

    def test_unhashable_items_are_accepted(self) -> None:
        # Only repr(item) is hashed, so lists and dicts work; a filter built
        # on the built-in hash function would raise TypeError here.
        bloom = BloomFilter(4096, 4, seed=0)
        bloom.add([1, 2, 3])
        bloom.add({"user": "ann", "id": 7})
        bloom.add([[0], {"nested": [1]}])
        assert [1, 2, 3] in bloom
        assert {"user": "ann", "id": 7} in bloom
        assert [[0], {"nested": [1]}] in bloom

    def test_items_with_the_same_repr_are_the_same_item(self) -> None:
        class Label:
            """Two Labels never compare equal, but share a repr."""

            def __init__(self, text: str) -> None:
                self.text = text

            def __repr__(self) -> str:
                return f"Label({self.text!r})"

        bloom = BloomFilter(4096, 4, seed=0)
        bloom.add(Label("alpha"))
        assert Label("alpha") in bloom  # a different object, the same repr

    def test_equal_items_with_different_reprs_are_different_items(self) -> None:
        # CAVEAT (documented contract, not a code defect): the filter keys on
        # repr(item), so equal values whose reprs differ are DIFFERENT items.
        # {1, 9} and {9, 1} are equal sets, but 1 and 9 collide in an 8-slot
        # set table, so insertion order decides the repr: "{1, 9}" vs
        # "{9, 1}". Likewise 0.0 == -0.0 but repr differs. One item in 10,000
        # bits with k = 5 makes a false positive essentially impossible, so
        # these checks are False. Callers should canonicalise such items
        # (for example sorted tuples) before adding them.
        assert {1, 9} == {9, 1} and repr({1, 9}) != repr({9, 1})
        bloom = BloomFilter(10_000, 5, seed=0)
        bloom.add({1, 9})
        bloom.add(-0.0)
        assert bloom.check({1, 9}) is True
        assert bloom.check({9, 1}) is False
        assert bloom.check(-0.0) is True
        assert bloom.check(0.0) is False


# ----------------------------------------------------------------------
# Sizing: from_capacity and the hand-checked anchor
# ----------------------------------------------------------------------
class TestSizing:
    """``from_capacity(n, p)`` uses ``m = ceil(-n ln p / (ln 2)^2)``, ``k = round((m / n) ln 2)``.

    The anchor, by hand: ``ln 0.01 = -4.605170``, ``(ln 2)^2 = 0.480453``,
    so ``m = ceil(4605.170 / 0.480453) = ceil(9585.06) = 9586``; then
    ``(9586 / 1000) * 0.693147 = 6.6445``, which rounds to ``k = 7``.

    The other rows, the same way:

    * ``(1000, 0.001)``: ``6907.755 / 0.480453 = 14377.6`` so ``m = 14378``;
      ``14.378 * 0.693147 = 9.966`` so ``k = 10``.
    * ``(1000, 0.1)``: ``2302.585 / 0.480453 = 4792.5`` so ``m = 4793``;
      ``4.793 * 0.693147 = 3.322`` so ``k = 3``.
    * ``(100, 0.01)``: ``958.506`` so ``m = 959``; ``9.59 * 0.693147 = 6.647``
      so ``k = 7``.
    * ``(10, 0.9)``: ``1.053605 / 0.480453 = 2.193`` so ``m = 3``;
      ``0.3 * 0.693147 = 0.208`` rounds to 0, and the floor of 1 gives ``k = 1``.
    * ``(1, 0.5)``: ``0.693147 / 0.480453 = 1.4427`` so ``m = 2``;
      ``2 * 0.693147 = 1.386`` so ``k = 1``.
    """

    def test_anchor_1000_items_at_one_percent(self) -> None:
        bloom = BloomFilter.from_capacity(1000, 0.01)
        assert (bloom.m, bloom.k) == (9586, 7)

    @pytest.mark.parametrize(
        "n, p, m, k",
        [
            pytest.param(1000, 0.01, 9586, 7, id="anchor"),
            pytest.param(1000, 0.001, 14378, 10, id="tenth-percent"),
            pytest.param(1000, 0.1, 4793, 3, id="ten-percent"),
            pytest.param(100, 0.01, 959, 7, id="hundred-items"),
            pytest.param(10, 0.9, 3, 1, id="loose-target-floor-k"),
            pytest.param(1, 0.5, 2, 1, id="one-item"),
        ],
    )
    def test_hand_computed_sizes(self, n: int, p: float, m: int, k: int) -> None:
        bloom = BloomFilter.from_capacity(n, p)
        assert (bloom.m, bloom.k) == (m, k)
        assert type(bloom.m) is int and type(bloom.k) is int

    def test_sizing_formula_rederived_here(self) -> None:
        # The same formulas typed out independently of the module.
        ln2 = math.log(2)
        for n, p in [(1000, 0.01), (1000, 0.001), (5000, 0.02), (37, 0.25)]:
            m = math.ceil(-n * math.log(p) / (ln2 * ln2))
            k = max(1, round(m / n * ln2))
            bloom = BloomFilter.from_capacity(n, p)
            assert (bloom.m, bloom.k) == (m, k)

    def test_from_capacity_returns_an_empty_filter_with_the_seed(self) -> None:
        bloom = BloomFilter.from_capacity(1000, 0.01, seed=77)
        assert isinstance(bloom, BloomFilter)
        assert bloom.seed == 77
        assert len(bloom) == 0
        assert bloom.fill_ratio == 0.0
        assert bloom.bits == bytes(1199)

    def test_anchor_memory_and_bits_per_item_at_capacity(self) -> None:
        # ceil(9586 / 8) = ceil(1198.25) = 1199 bytes, and 9586 / 1000 =
        # 9.586 bits per item: the "about 9.6 bits for 1%" rule of thumb.
        bloom = BloomFilter.from_capacity(1000, 0.01, seed=1)
        for i in range(1000):
            bloom.add(i)
        assert bloom.memory_bytes() == 1199
        assert bloom.bits_per_item() == pytest.approx(9.586, abs=1e-12)
        assert bloom.load_factor() == pytest.approx(1000 / 9586, abs=1e-15)

    def test_tighter_target_costs_more_bits(self) -> None:
        sizes = [BloomFilter.from_capacity(1000, p).m for p in (0.1, 0.01, 0.001)]
        assert sizes == [4793, 9586, 14378]
        # Each factor of ten costs about 4.79 bits per item: ln 10 / (ln 2)^2.
        per_decade = math.log(10) / math.log(2) ** 2
        assert per_decade == pytest.approx(4.7925, abs=1e-4)
        assert (sizes[1] - sizes[0]) / 1000 == pytest.approx(per_decade, abs=0.01)
        assert (sizes[2] - sizes[1]) / 1000 == pytest.approx(per_decade, abs=0.01)


# ----------------------------------------------------------------------
# The textbook false-positive formula
# ----------------------------------------------------------------------
class TestTheoreticalRate:
    """``theoretical_fp_rate`` is ``(1 - e^(-k n / m))^k``.

    At the anchor, by hand: ``k n / m = 7000 / 9586 = 0.730232``,
    ``e^(-0.730232) = 0.481797``, ``1 - 0.481797 = 0.518203``, and
    ``0.518203^7 = 0.010035``. At ``n = 2000``: ``k n / m = 1.460463``,
    ``e^(-1.460463) = 0.232129``, ``0.767871^7 = 0.157``.
    """

    def test_empty_filter_is_exactly_zero(self) -> None:
        rate = BloomFilter.from_capacity(1000, 0.01).theoretical_fp_rate()
        assert rate == 0.0
        assert math.copysign(1.0, rate) == 1.0  # not -0.0

    def test_anchor_at_capacity_is_about_0_01003(self) -> None:
        bloom = BloomFilter.from_capacity(1000, 0.01)
        for i in range(1000):
            bloom.add(f"member-{i}")
        rate = bloom.theoretical_fp_rate()
        assert round(rate, 5) == 0.01003
        assert rate == pytest.approx(0.010035, abs=1e-6)

    def test_overfilled_to_double_capacity_is_about_0_157(self) -> None:
        bloom = BloomFilter(9586, 7)
        for i in range(2000):
            bloom.add(i)
        assert bloom.theoretical_fp_rate() == pytest.approx(0.157, abs=5e-4)

    @pytest.mark.parametrize("n", [1, 10, 250, 1000, 3000])
    def test_matches_the_formula_typed_out_here(self, n: int) -> None:
        bloom = BloomFilter(9586, 7, seed=4)
        for i in range(n):
            bloom.add(i)
        expected = (1.0 - math.exp(-7 * n / 9586)) ** 7
        assert bloom.theoretical_fp_rate() == pytest.approx(expected, rel=1e-9)

    def test_rises_with_every_batch_of_adds(self) -> None:
        bloom = BloomFilter(2000, 4, seed=2)
        rates = [bloom.theoretical_fp_rate()]
        for batch in range(8):
            for i in range(100):
                bloom.add((batch, i))
            rates.append(bloom.theoretical_fp_rate())
        assert all(a < b for a, b in zip(rates, rates[1:]))
        assert 0.0 <= rates[0] and rates[-1] <= 1.0


# ----------------------------------------------------------------------
# The measured false-positive rate
# ----------------------------------------------------------------------
class TestMeasuredFalsePositiveRate:
    """Count Trues among strings that were never added.

    Fixed seeds make each measurement deterministic; the bands are still
    chosen wide enough (at least five standard errors) that any correct
    filter passes for any seed.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2026])
    def test_anchor_rate_over_100000_non_members(self, seed: int) -> None:
        false_positives, _, m, k = measured_rate(1000, 0.01, seed, FP_PROBES)
        assert (m, k) == (9586, 7)
        observed = false_positives / FP_PROBES
        assert 0.005 <= observed <= 0.02
        # The band is wider than five standard errors on each side, even
        # counting the spread of the fill from one filter to the next.
        se = fp_standard_error(9586, 7, 1000, FP_PROBES)
        assert 0.010035 - 0.005 > 5 * se and 0.02 - 0.010035 > 5 * se

    @pytest.mark.parametrize(
        "p, m, k",
        [
            pytest.param(0.1, 4793, 3, id="ten-percent"),
            pytest.param(0.001, 14378, 10, id="tenth-percent"),
        ],
    )
    def test_other_targets_land_within_a_factor_of_three(
        self, p: float, m: int, k: int
    ) -> None:
        false_positives, _, got_m, got_k = measured_rate(1000, p, 0, FP_PROBES)
        assert (got_m, got_k) == (m, k)
        low, high = p / 3, p * 3
        assert low <= false_positives / FP_PROBES <= high
        # The band clears five standard errors around the formula's rate.
        theory = (1.0 - math.exp(-k * 1000 / m)) ** k
        se = fp_standard_error(m, k, 1000, FP_PROBES)
        assert theory - low > 5 * se and high - theory > 5 * se

    @pytest.mark.parametrize("seed", [0, 1, 2026])
    def test_rate_tracks_fill_ratio_to_the_k(self, seed: int) -> None:
        # Given the filter that was actually built, a fresh item is a false
        # positive when all k probes hit set bits: about fill_ratio ** k.
        # That removes the filter-to-filter variation in the fill, leaving
        # only binomial noise, so five binomial standard errors suffice.
        false_positives, fill, _, k = measured_rate(1000, 0.01, seed, FP_PROBES)
        predicted = fill**k
        observed = false_positives / FP_PROBES
        assert abs(observed - predicted) <= 5 * binomial_se(predicted, FP_PROBES)

    def test_overfilled_rate_matches_the_formula(self) -> None:
        # Twice the capacity: the formula says 0.157 (worked by hand above).
        bloom = BloomFilter(9586, 7, seed=0)
        for i in range(2000):
            bloom.add(f"member-{i}")
        probes = 20_000
        observed = sum(bloom.check(f"probe-{i}") for i in range(probes)) / probes
        theory = (1.0 - math.exp(-7 * 2000 / 9586)) ** 7
        assert theory == pytest.approx(0.157, abs=5e-4)
        assert abs(observed - theory) <= 5 * fp_standard_error(9586, 7, 2000, probes)
        # Given this filter's fill, only binomial noise is left.
        predicted = bloom.fill_ratio**7
        assert abs(observed - predicted) <= 5 * binomial_se(predicted, probes)
        assert observed > 0.02  # far above the 1% the filter was sized for

    def test_check_never_changes_the_filter(self) -> None:
        bloom = filled(4096, 5, 6, make_items("strings", 300, seed=6))
        before = (bloom.bits, len(bloom), bloom.fill_ratio)
        for i in range(5000):
            bloom.check(f"probe-{i}")
            _ = f"other-{i}" in bloom
        assert (bloom.bits, len(bloom), bloom.fill_ratio) == before

    def test_a_saturated_filter_says_yes_to_everything(self) -> None:
        # m = 8, k = 8 and an odd stride visits all 8 residues, so ONE add
        # sets every bit. The false-positive rate is then exactly 1.
        bloom = BloomFilter(8, 8, seed=0)
        bloom.add("only")
        assert bloom.bits == b"\xff"
        assert bloom.fill_ratio == 1.0
        assert all(bloom.check(f"never-added-{i}") for i in range(1000))


# ----------------------------------------------------------------------
# Seeds and determinism
# ----------------------------------------------------------------------
class TestDeterminismAndSeeds:
    """Same seed, same bits; different seed, a different hash family."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_same_seed_gives_identical_bits_and_answers(self, seed: int) -> None:
        items = make_items("mixed", 1000, seed=99)
        first = BloomFilter.from_capacity(1000, 0.01, seed=seed)
        second = BloomFilter.from_capacity(1000, 0.01, seed=seed)
        for item in items:
            first.add(item)
            second.add(item)
        assert first.bits == second.bits
        probes = [f"probe-{i}" for i in range(2000)]
        assert [first.check(p) for p in probes] == [second.check(p) for p in probes]

    @pytest.mark.parametrize(
        "seed_a, seed_b",
        [(0, 1), (1, 2), (42, 43), (0, 2**128 - 1), (2026, 2027)],
    )
    def test_different_seeds_give_different_bits(self, seed_a: int, seed_b: int) -> None:
        items = make_items("strings", 1000, seed=5)
        a = BloomFilter.from_capacity(1000, 0.01, seed=seed_a)
        b = BloomFilter.from_capacity(1000, 0.01, seed=seed_b)
        for item in items:
            a.add(item)
            b.add(item)
        assert a.bits != b.bits
        # Different bits, but both still hold every item.
        assert all(item in a and item in b for item in items)

    def test_one_item_under_two_seeds_lands_on_different_probes(self) -> None:
        a = filled(9586, 7, 0, ["apple"])
        b = filled(9586, 7, 1, ["apple"])
        assert set_bits(a.bits) != set_bits(b.bits)

    @pytest.mark.parametrize("shuffle_seed", [1, 2, 3])
    def test_insertion_order_does_not_matter(self, shuffle_seed: int) -> None:
        items = make_items("tuples", 800, seed=0)
        reordered = list(items)
        make_rng(shuffle_seed).shuffle(reordered)
        assert reordered != items
        assert filled(6000, 6, 9, items).bits == filled(6000, 6, 9, reordered).bits

    def test_repeat_adds_leave_bits_unchanged(self) -> None:
        once = filled(512, 4, 3, ["x"])
        many = filled(512, 4, 3, ["x"] * 25)
        assert once.bits == many.bits
        assert (len(once), len(many)) == (1, 25)


# ----------------------------------------------------------------------
# The documented hashing and bit layout
# ----------------------------------------------------------------------
GEOMETRIES = [
    pytest.param(97, 4, 2, id="prime-m"),
    pytest.param(1024, 8, 0, id="power-of-two"),
    pytest.param(9586, 7, 7, id="anchor"),
    pytest.param(13, 20, 3, id="k-exceeds-m"),
    pytest.param(1, 3, 2**128 - 1, id="single-bit-max-seed"),
]


class TestDocumentedHashing:
    """The set bits must be exactly those :func:`documented_probes` predicts."""

    @pytest.mark.parametrize("m, k, seed", GEOMETRIES)
    @pytest.mark.parametrize(
        "item",
        ["alice", 42, ("a", 1), 3.25, [1, 2], None, "", "unicode-\u00e9\u4e2d"],
        ids=["str", "int", "tuple", "float", "list", "none", "empty-str", "unicode"],
    )
    def test_single_add_sets_exactly_the_documented_bits(
        self, m: int, k: int, seed: int, item: Any
    ) -> None:
        bloom = filled(m, k, seed, [item])
        assert set_bits(bloom.bits) == set(documented_probes(item, m, k, seed))

    @pytest.mark.parametrize("m, k, seed", GEOMETRIES)
    def test_many_adds_set_the_union_of_documented_bits(
        self, m: int, k: int, seed: int
    ) -> None:
        items = make_items("mixed", 200, seed=seed % 1000)
        bloom = filled(m, k, seed, items)
        expected: Set[int] = set()
        for item in items:
            expected.update(documented_probes(item, m, k, seed))
        assert set_bits(bloom.bits) == expected

    @pytest.mark.parametrize("m, k", [(8, 4), (16, 16), (1024, 8), (4096, 12)])
    def test_power_of_two_m_gets_k_distinct_probes(self, m: int, k: int) -> None:
        # The stride h2 is forced odd, and an odd number is invertible mod a
        # power of two, so i * h2 mod m is distinct for i < m: one add on a
        # fresh filter sets exactly k bits. An even stride would set fewer.
        for i in range(40):
            bloom = filled(m, k, i, [f"item-{i}"])
            assert len(set_bits(bloom.bits)) == k

    @pytest.mark.parametrize(
        "m, k, expected",
        [
            pytest.param(1, 1, b"\x01", id="m1-k1"),
            pytest.param(1, 4, b"\x01", id="m1-k4"),
            pytest.param(2, 2, b"\x03", id="m2-k2"),
            pytest.param(8, 8, b"\xff", id="m8-k8"),
            pytest.param(16, 16, b"\xff\xff", id="m16-k16"),
        ],
    )
    def test_hand_checked_bit_arrays(self, m: int, k: int, expected: bytes) -> None:
        # With m a power of two and k = m, the odd stride visits every bit.
        for item in ["a", 7, ("z", 0)]:
            bloom = BloomFilter(m, k, seed=5)
            assert bloom.bits == bytes(len(expected))
            bloom.add(item)
            assert bloom.bits == expected

    def test_trailing_bits_past_m_stay_zero(self) -> None:
        # 9586 = 8 * 1198 + 2, so the last byte holds bits 9584 and 9585 at
        # positions 0 and 1; positions 2 to 7 lie past m and must stay 0
        # even in a heavily overfilled filter.
        bloom = filled(9586, 7, 0, list(range(5000)))
        assert len(bloom.bits) == 1199
        assert bloom.bits[-1] & 0b1111_1100 == 0
        assert max(set_bits(bloom.bits)) < 9586

    def test_prime_m_with_k_over_m_saturates_to_exactly_m_bits(self) -> None:
        # 13 is prime, so any non-zero stride visits all 13 residues within
        # 13 probes; k = 20 >= 13, so one add with a non-zero stride sets all
        # 13 bits. A zero stride has chance 1/13 per item, so after 50 items
        # the filter is full: bits 0-7 in byte 0, bits 8-12 in byte 1.
        bloom = filled(13, 20, 3, [f"w{i}" for i in range(50)])
        assert bloom.bits == b"\xff\x1f"
        assert bloom.fill_ratio == 1.0

    def test_bits_is_an_immutable_snapshot(self) -> None:
        bloom = BloomFilter(256, 3, seed=1)
        snapshot = bloom.bits
        assert type(snapshot) is bytes
        for i in range(50):
            bloom.add(i)
        assert snapshot == bytes(32)
        assert bloom.bits != snapshot


# ----------------------------------------------------------------------
# The empty filter
# ----------------------------------------------------------------------
class TestEmptyFilter:
    """Nothing added means every bit is 0, so every check is False."""

    PROBES: List[Any] = [
        0, 1, -1, "", "a", None, (), [], {}, b"", 0.0, float("inf"), True, ("x", 1),
    ] + [f"probe-{i}" for i in range(1000)]

    @pytest.mark.parametrize(
        "m, k, seed",
        [(1, 1, 0), (8, 3, 1), (9586, 7, 0), (100_000, 10, 2026), (13, 20, 2**128 - 1)],
    )
    def test_checks_false_for_everything(self, m: int, k: int, seed: int) -> None:
        bloom = BloomFilter(m, k, seed=seed)
        assert not any(bloom.check(item) for item in self.PROBES)
        assert not any(item in bloom for item in self.PROBES)

    def test_empty_statistics(self) -> None:
        bloom = BloomFilter.from_capacity(1000, 0.01)
        assert len(bloom) == 0
        assert bloom.fill_ratio == 0.0
        assert bloom.load_factor() == 0.0
        assert bloom.bits_per_item() == math.inf
        assert bloom.theoretical_fp_rate() == 0.0
        assert bloom.bits == bytes(1199)


# ----------------------------------------------------------------------
# Inspection: len, fill_ratio, load_factor, bits_per_item, memory_bytes
# ----------------------------------------------------------------------
class TestInspection:
    """The bookkeeping methods, against hand values."""

    def test_len_counts_add_calls_including_repeats(self) -> None:
        bloom = BloomFilter(256, 3)
        counts = []
        for item in ["a", "b", "a", "a", 1, "1", [1]]:
            bloom.add(item)
            counts.append(len(bloom))
        assert counts == [1, 2, 3, 4, 5, 6, 7]

    def test_fill_ratio_stays_in_unit_interval_and_grows(self) -> None:
        bloom = BloomFilter(2048, 5, seed=4)
        ratios = [bloom.fill_ratio]
        for i in range(1500):
            bloom.add(f"grow-{i}")
            ratios.append(bloom.fill_ratio)
        assert ratios[0] == 0.0
        assert all(0.0 <= r <= 1.0 for r in ratios)
        assert all(a <= b for a, b in zip(ratios, ratios[1:]))
        assert ratios[1] > 0.0
        assert ratios[-1] > ratios[len(ratios) // 2] > ratios[1]

    def test_fill_ratio_is_set_bits_over_m(self) -> None:
        for m, k, n in [(9586, 7, 1000), (13, 2, 4), (777, 3, 150)]:
            bloom = filled(m, k, 1, list(range(n)))
            assert bloom.fill_ratio == len(set_bits(bloom.bits)) / m

    def test_fill_ratio_after_one_add_with_one_probe_is_one_over_m(self) -> None:
        # k = 1 sets exactly one bit: 1 / 64 = 0.015625.
        bloom = BloomFilter(64, 1, seed=0)
        bloom.add("only")
        assert bloom.fill_ratio == 0.015625

    @pytest.mark.parametrize("seed", [0, 1, 2026])
    def test_fill_ratio_at_capacity_is_near_one_minus_e_to_minus_kn_over_m(
        self, seed: int
    ) -> None:
        # Expected fill 1 - e^(-7000 / 9586) = 1 - 0.481797 = 0.518203. The
        # count of set bits has variance at most m / 4, so one standard
        # error of the fraction is at most 0.5 / sqrt(m) = 0.0051; allow 5.
        _, fill, m, k = measured_rate(1000, 0.01, seed, FP_PROBES)
        assert fill == pytest.approx(0.518203, abs=5 * 0.5 / math.sqrt(m))
        assert 1.0 - math.exp(-k * 1000 / m) == pytest.approx(0.518203, abs=1e-6)

    @pytest.mark.parametrize(
        "m, expected",
        [
            (1, 1), (7, 1), (8, 1), (9, 2), (15, 2), (16, 2), (17, 3),
            (100, 13), (1024, 128), (9586, 1199), (14378, 1798),
        ],
    )
    def test_memory_bytes_is_ceil_m_over_8(self, m: int, expected: int) -> None:
        bloom = BloomFilter(m, 2)
        assert bloom.memory_bytes() == expected == math.ceil(m / 8)
        assert len(bloom.bits) == expected

    def test_memory_does_not_grow_with_items(self) -> None:
        bloom = BloomFilter.from_capacity(1000, 0.01)
        sizes = {bloom.memory_bytes()}
        for i in range(3000):
            bloom.add(("big item", i, "x" * 50))
        sizes.add(bloom.memory_bytes())
        assert sizes == {1199}

    def test_load_factor_and_bits_per_item_hand_values(self) -> None:
        # 25 items in 100 bits: n / m = 0.25 and m / n = 4.0.
        bloom = filled(100, 2, 0, list(range(25)))
        assert bloom.load_factor() == 0.25
        assert bloom.bits_per_item() == 4.0

    @pytest.mark.parametrize("m, n", [(9586, 1000), (100, 3), (64, 64), (10, 40)])
    def test_load_factor_is_n_over_m_and_bits_per_item_is_m_over_n(
        self, m: int, n: int
    ) -> None:
        bloom = filled(m, 3, 0, [f"i{j}" for j in range(n)])
        assert bloom.load_factor() == n / m
        assert bloom.bits_per_item() == m / n
        assert bloom.load_factor() * bloom.bits_per_item() == pytest.approx(1.0)

    def test_repeats_count_toward_load(self) -> None:
        # n is the number of add calls, so ten adds of one item is n = 10.
        bloom = filled(50, 2, 0, ["same"] * 10)
        assert bloom.load_factor() == 10 / 50
        assert bloom.bits_per_item() == 5.0

    def test_properties_report_construction_arguments(self) -> None:
        bloom = BloomFilter(321, 6, seed=99)
        assert (bloom.m, bloom.k, bloom.seed) == (321, 6, 99)

    @pytest.mark.parametrize("name", ["m", "k", "seed", "bits", "fill_ratio"])
    def test_properties_are_read_only(self, name: str) -> None:
        bloom = BloomFilter(321, 6, seed=99)
        with pytest.raises(AttributeError):
            setattr(bloom, name, 5)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
class TestValidation:
    """Bad arguments are refused with the documented exception type."""

    @pytest.mark.parametrize(
        "m_bits, k_hashes, seed, error, field",
        [
            pytest.param(0, 3, 0, ValueError, "m_bits", id="m-zero"),
            pytest.param(-8, 3, 0, ValueError, "m_bits", id="m-negative"),
            pytest.param(10, 0, 0, ValueError, "k_hashes", id="k-zero"),
            pytest.param(10, -1, 0, ValueError, "k_hashes", id="k-negative"),
            pytest.param(True, 3, 0, TypeError, "m_bits", id="m-bool"),
            pytest.param(8.0, 3, 0, TypeError, "m_bits", id="m-float"),
            pytest.param("8", 3, 0, TypeError, "m_bits", id="m-str"),
            pytest.param(None, 3, 0, TypeError, "m_bits", id="m-none"),
            pytest.param(10, True, 0, TypeError, "k_hashes", id="k-bool"),
            pytest.param(10, 2.5, 0, TypeError, "k_hashes", id="k-float"),
            pytest.param(10, None, 0, TypeError, "k_hashes", id="k-none"),
            pytest.param(10, 3, -1, ValueError, "seed", id="seed-negative"),
            pytest.param(10, 3, 2**128, ValueError, "seed", id="seed-too-big"),
            pytest.param(10, 3, 1.5, TypeError, "seed", id="seed-float"),
            pytest.param(10, 3, True, TypeError, "seed", id="seed-bool"),
            pytest.param(10, 3, None, TypeError, "seed", id="seed-none"),
            pytest.param(10, 3, "0", TypeError, "seed", id="seed-str"),
        ],
    )
    def test_constructor_rejects(
        self, m_bits: Any, k_hashes: Any, seed: Any, error: type, field: str
    ) -> None:
        with pytest.raises(error, match=field):
            BloomFilter(m_bits, k_hashes, seed=seed)

    @pytest.mark.parametrize(
        "m_bits, k_hashes, seed",
        [(1, 1, 0), (1, 1, 2**128 - 1), (10**6, 50, 12345)],
    )
    def test_constructor_accepts_boundary_values(
        self, m_bits: int, k_hashes: int, seed: int
    ) -> None:
        bloom = BloomFilter(m_bits, k_hashes, seed=seed)
        assert (bloom.m, bloom.k, bloom.seed) == (m_bits, k_hashes, seed)

    @pytest.mark.parametrize(
        "fp_rate",
        [
            pytest.param(0, id="int-zero"),
            pytest.param(0.0, id="zero"),
            pytest.param(1, id="int-one"),
            pytest.param(1.0, id="one"),
            pytest.param(-0.01, id="negative"),
            pytest.param(-1, id="minus-one"),
            pytest.param(1.5, id="above-one"),
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("nan"), id="nan"),
        ],
    )
    def test_from_capacity_rejects_rates_outside_open_unit_interval(
        self, fp_rate: float
    ) -> None:
        with pytest.raises(ValueError, match="fp_rate"):
            BloomFilter.from_capacity(1000, fp_rate)

    @pytest.mark.parametrize(
        "fp_rate", [True, False, "0.01", None, [0.01]], ids=["true", "false", "str", "none", "list"]
    )
    def test_from_capacity_rejects_non_real_rates(self, fp_rate: Any) -> None:
        with pytest.raises(TypeError, match="fp_rate"):
            BloomFilter.from_capacity(1000, fp_rate)

    @pytest.mark.parametrize("expected_items", [0, -1, -1000])
    def test_from_capacity_rejects_fewer_than_one_item(self, expected_items: int) -> None:
        with pytest.raises(ValueError, match="expected_items"):
            BloomFilter.from_capacity(expected_items, 0.01)

    @pytest.mark.parametrize(
        "expected_items", [1000.0, True, "1000", None], ids=["float", "bool", "str", "none"]
    )
    def test_from_capacity_rejects_non_int_item_counts(self, expected_items: Any) -> None:
        with pytest.raises(TypeError, match="expected_items"):
            BloomFilter.from_capacity(expected_items, 0.01)

    @pytest.mark.parametrize("seed", [-1, 2**128])
    def test_from_capacity_rejects_bad_seed(self, seed: int) -> None:
        with pytest.raises(ValueError, match="seed"):
            BloomFilter.from_capacity(1000, 0.01, seed=seed)

    def test_from_capacity_accepts_rates_just_inside_the_interval(self) -> None:
        for p in (1e-9, 0.5, 0.999):
            bloom = BloomFilter.from_capacity(10, p)
            assert bloom.m >= 1 and bloom.k >= 1


# ----------------------------------------------------------------------
# Reproducibility across interpreter processes
# ----------------------------------------------------------------------
#: Seed and items of the reference filter. The child script below builds
#: the same filter; keep the two in step.
REFERENCE_SEED = 2026


def build_reference_filter() -> BloomFilter:
    """Build the reference filter in THIS process.

    500 strings and 500 tuples holding strings. The built-in hash function
    of both is salted per process, so a filter that used it would set
    different bits here and in each child.
    """
    bloom = BloomFilter.from_capacity(1000, 0.01, seed=REFERENCE_SEED)
    for i in range(500):
        bloom.add(f"user-{i}@example.org")
        bloom.add(("order", i, f"sku-{i}"))
    return bloom


#: Run with ``python -c`` from the repository root. Prints one JSON line.
CHILD_SCRIPT = textwrap.dedent(
    """
    import json
    from src.randomized.bloom_filter import BloomFilter

    bloom = BloomFilter.from_capacity(1000, 0.01, seed=2026)
    for i in range(500):
        bloom.add(f"user-{i}@example.org")
        bloom.add(("order", i, f"sku-{i}"))

    letters = frozenset("abcdefghijklmnopqrstuvwxyz")
    caveat = BloomFilter(1024, 4, seed=2026)
    caveat.add(letters)

    print(json.dumps({
        "bits": bloom.bits.hex(),
        "m": bloom.m,
        "k": bloom.k,
        "n": len(bloom),
        "letters_repr": repr(letters),
        "caveat_bits": caveat.bits.hex(),
    }))
    """
)


@functools.lru_cache(maxsize=None)
def run_child(hash_seed: str) -> Dict[str, Any]:
    """Build the reference filter in a fresh interpreter with ``PYTHONHASHSEED``.

    Cached so the two tests that read each child share one subprocess.

    Args:
        hash_seed: The value for ``PYTHONHASHSEED`` in the child.

    Returns:
        The child's JSON report as a dict.
    """
    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestReproducibleAcrossProcesses:
    """Same seed, same bits, in every interpreter, whatever its hash salt."""

    @pytest.mark.parametrize("hash_seed", ["1", "2"])
    def test_child_bit_array_matches_this_process(self, hash_seed: str) -> None:
        local = build_reference_filter()
        child = run_child(hash_seed)
        assert (child["m"], child["k"], child["n"]) == (9586, 7, 1000)
        assert (local.m, local.k, len(local)) == (9586, 7, 1000)
        assert child["bits"] == local.bits.hex()

    def test_children_really_salt_str_hashes_differently(self) -> None:
        # The teeth of the test above: a frozenset of strings iterates in
        # hash order, so its repr shows each child's str hash salt. The
        # reprs differ, so the salts differ, yet the filters agree bit for
        # bit. A filter on the built-in hash function could not.
        one, two = run_child("1"), run_child("2")
        assert one["letters_repr"] != two["letters_repr"]
        assert one["bits"] == two["bits"]

    def test_salt_dependent_reprs_are_not_reproducible(self) -> None:
        # CAVEAT (documented contract, not a code defect): reproducibility
        # holds for items whose repr is stable. A set or frozenset of strings
        # has a repr that changes with PYTHONHASHSEED, so the same item sets
        # different bits in different processes. Canonicalise such items
        # (for example sorted tuples) before adding them.
        one, two = run_child("1"), run_child("2")
        assert one["letters_repr"] != two["letters_repr"]
        assert one["caveat_bits"] != two["caveat_bits"]
        # The same letters as a sorted tuple have one repr everywhere.
        canonical = tuple(sorted("abcdefghijklmnopqrstuvwxyz"))
        here = filled(1024, 4, REFERENCE_SEED, [canonical])
        assert set_bits(here.bits) == set(
            documented_probes(canonical, 1024, 4, REFERENCE_SEED)
        )
