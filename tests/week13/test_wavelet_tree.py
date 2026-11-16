"""Week 13 tests: balanced and Huffman-shaped wavelet trees (spec 4.3 and the 9.5 anchors).

The rulers are independent of the code under test (spec 1.1 allows them in
tests only): ``str`` slicing and ``str.count`` for rank and range counts, a
list of positions for select, ``sorted`` for quantiles, ``collections.Counter``
for frequencies, ``heapq`` for the optimal Huffman cost, and numpy for H0.

Anchors (9.5), computed with a brute-force reference on 11/16:

* On ``"ACGTACGTTGCA"``: ``range_count("A", 2, 10) = 1``,
  ``rank("C", 12) = 3``, ``range_quantile(3, 9, k)`` for k = 1, 3, 6 is
  ``"A"``, ``"G"``, ``"T"`` (the window is ``"TACGTT"``), and ``decode()``
  returns the input.
* ``access``, ``rank``, ``select``, ``range_count``, ``range_count_values``
  and ``range_quantile`` match brute force on 200 random queries over random
  strings with alphabet sizes 2, 4 and 26, for both shapes where the method
  applies (``range_quantile`` is balanced only).
* Huffman shape on the repository ``README.md``: payload bits per symbol is
  within 1 bit of H0 and below the balanced shape's ``ceil(log2 sigma)``.
* Synthetic DNA (``np.random.default_rng(42)`` over ACGT, n = 10^5): the
  balanced payload is exactly 2 bits per symbol and the compression ratio is
  below 0.30.

The same synthetic DNA is also checked against the 9.7 numbers the DNA index
will rely on. Seeds come from ``derive_seed(42, ...)``. No timings are
asserted.
"""

from __future__ import annotations

import ast
import heapq
import inspect
import math
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pytest

import src.advds.succinct.wavelet_tree as wavelet_module
from src.advds.succinct.wavelet_tree import SHAPES, WaveletTree, zero_order_entropy
from src.utils.random_utils import DEFAULT_SEED, derive_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"

ANCHOR_TEXT = "ACGTACGTTGCA"
ALPHABETS: Dict[int, str] = {2: "ab", 4: "ACGT", 26: string.ascii_lowercase}
DISTRIBUTIONS = ("uniform", "zipf")
N_QUERIES = 200
RANDOM_LENGTH = 3000  # crosses the 2,048-bit superblock boundary at the root
DNA_LENGTH = 100_000


# ---------------------------------------------------------------- rulers


def _synthetic_dna(n: int, seed: int) -> str:
    """The spec 6.3 generator, written out so the test does not depend on the app."""
    return "".join(np.array(list("ACGT"))[np.random.default_rng(seed).integers(0, 4, n)])


def _random_text(sigma: int, dist: str, n: int = RANDOM_LENGTH) -> str:
    """A seeded random string over the first ``sigma`` symbols of ALPHABETS."""
    symbols = ALPHABETS[sigma]
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "wavelet-text", sigma, dist))
    if dist == "uniform":
        probs = np.full(sigma, 1.0 / sigma)
    else:  # skewed, so the Huffman code lengths differ
        weights = 1.0 / np.arange(1, sigma + 1) ** 1.2
        probs = weights / weights.sum()
    picks = rng.choice(sigma, size=n, p=probs)
    return "".join(symbols[j] for j in picks.tolist())


def _query_rng(*labels: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(DEFAULT_SEED, "wavelet-queries", *labels))


def _random_range(rng: np.random.Generator, n: int) -> Tuple[int, int]:
    a, b = (int(x) for x in rng.integers(0, n + 1, size=2))
    return (a, b) if a <= b else (b, a)


def _order(alphabet: str) -> Dict[str, int]:
    return {c: j for j, c in enumerate(alphabet)}


def _optimal_prefix_cost(counts: Iterable[int]) -> int:
    """Cost of an optimal prefix code: the sum of all Huffman merge weights."""
    heap = [c for c in counts if c > 0]
    heapq.heapify(heap)
    cost = 0
    while len(heap) > 1:
        merged = heapq.heappop(heap) + heapq.heappop(heap)
        cost += merged
        heapq.heappush(heap, merged)
    return cost


def _numpy_entropy(text: str) -> float:
    counts = np.array(list(Counter(text).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _rank_directory_bits(m: int) -> int:
    """Spec 4.2 rank directory size for one node of length m."""
    return 32 * math.ceil(m / 2048) + 16 * math.ceil(m / 256)


RANDOM_CASES = [
    (sigma, shape, dist) for sigma in ALPHABETS for shape in SHAPES for dist in DISTRIBUTIONS
]
BALANCED_CASES = [case for case in RANDOM_CASES if case[1] == "balanced"]


def _case_id(case: Tuple[int, str, str]) -> str:
    return f"sigma{case[0]}-{case[1]}-{case[2]}"


# ---------------------------------------------------- 9.5 anchors (exact)


def test_anchor_acgt_balanced() -> None:
    wt = WaveletTree(ANCHOR_TEXT)
    assert wt.range_count("A", 2, 10) == 1
    assert wt.rank("C", 12) == 3
    assert ANCHOR_TEXT[3:9] == "TACGTT"
    assert [wt.range_quantile(3, 9, k) for k in (1, 3, 6)] == ["A", "G", "T"]
    assert wt.decode() == ANCHOR_TEXT


@pytest.mark.parametrize("shape", SHAPES)
def test_anchor_acgt_counts_and_decode_on_both_shapes(shape: str) -> None:
    wt = WaveletTree(ANCHOR_TEXT, shape=shape)
    assert wt.range_count("A", 2, 10) == 1
    assert wt.rank("C", 12) == 3
    assert wt.decode() == ANCHOR_TEXT


def test_anchor_acgt_quantile_is_balanced_only() -> None:
    with pytest.raises(ValueError):
        WaveletTree(ANCHOR_TEXT, shape="huffman").range_quantile(3, 9, 1)


def test_random_cases_cover_the_spec_alphabet_sizes_and_shapes() -> None:
    assert sorted({case[0] for case in RANDOM_CASES}) == [2, 4, 26]
    assert {case[1] for case in RANDOM_CASES} == {"balanced", "huffman"}
    for sigma in ALPHABETS:
        for dist in DISTRIBUTIONS:
            assert set(_random_text(sigma, dist)) == set(ALPHABETS[sigma])


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_anchor_access_matches_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    rng = _query_rng("access", *case)
    for i in rng.integers(0, len(text), size=N_QUERIES).tolist():
        assert wt.access(i) == text[i]


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_anchor_rank_matches_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    rng = _query_rng("rank", *case)
    candidates = ALPHABETS[sigma] + "#"  # "#" is not in the alphabet: rank 0
    for _ in range(N_QUERIES):
        c = candidates[int(rng.integers(0, len(candidates)))]
        i = int(rng.integers(0, len(text) + 1))
        assert wt.rank(c, i) == text[:i].count(c)


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_anchor_select_matches_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    positions = {c: [p for p, ch in enumerate(text) if ch == c] for c in ALPHABETS[sigma]}
    rng = _query_rng("select", *case)
    for _ in range(N_QUERIES):
        c = ALPHABETS[sigma][int(rng.integers(0, sigma))]
        k = int(rng.integers(1, len(positions[c]) + 1))
        p = wt.select(c, k)
        assert p == positions[c][k - 1]
        assert wt.rank(c, p) == k - 1 and wt.rank(c, p + 1) == k


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_anchor_range_count_matches_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    rng = _query_rng("range_count", *case)
    for _ in range(N_QUERIES):
        c = ALPHABETS[sigma][int(rng.integers(0, sigma))]
        l, r = _random_range(rng, len(text))
        assert wt.range_count(c, l, r) == text[l:r].count(c)


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_anchor_range_count_values_matches_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    order = _order(wt.alphabet)
    rng = _query_rng("range_count_values", *case)
    for _ in range(N_QUERIES):
        lo = ALPHABETS[sigma][int(rng.integers(0, sigma))]
        hi = ALPHABETS[sigma][int(rng.integers(0, sigma))]  # lo after hi gives 0
        l, r = _random_range(rng, len(text))
        expected = 0
        for ch in text[l:r]:
            if order[lo] <= order[ch] <= order[hi]:
                expected += 1
        assert wt.range_count_values(l, r, lo, hi) == expected


@pytest.mark.parametrize("case", BALANCED_CASES, ids=_case_id)
def test_anchor_range_quantile_matches_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    order = _order(wt.alphabet)
    rng = _query_rng("range_quantile", *case)
    for _ in range(N_QUERIES):
        l, r = _random_range(rng, len(text))
        if l == r:
            r = l + 1 if l < len(text) else r
            l = r - 1
        k = int(rng.integers(1, r - l + 1))
        assert wt.range_quantile(l, r, k) == sorted(text[l:r], key=order.__getitem__)[k - 1]


@pytest.mark.parametrize("sigma", sorted(ALPHABETS))
def test_range_quantile_raises_on_huffman_for_every_case(sigma: int) -> None:
    wt = WaveletTree(_random_text(sigma, "zipf"), shape="huffman")
    with pytest.raises(ValueError):
        wt.range_quantile(0, 10, 1)


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_decode_and_frequencies_match_brute_force(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    assert wt.decode() == text
    assert wt.frequencies() == dict(Counter(text))
    assert list(wt.frequencies()) == list(wt.alphabet)
    short = text[:300]
    small = WaveletTree(short, shape=shape)
    assert "".join(small.access(i) for i in range(len(short))) == small.decode() == short


# ------------------------------------------------- README (Huffman vs H0)


def test_anchor_huffman_on_readme_is_within_one_bit_of_h0() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    n = len(text)
    huffman = WaveletTree(text, shape="huffman")
    balanced = WaveletTree(text)
    h0 = zero_order_entropy(text)
    assert h0 == pytest.approx(_numpy_entropy(text), abs=1e-12)
    bits_per_symbol = huffman.size_in_bits() / n
    assert abs(bits_per_symbol - h0) <= 1.0  # the anchor
    assert h0 <= bits_per_symbol < h0 + 1.0  # Huffman's bound, which implies it
    ceiling = math.ceil(math.log2(balanced.sigma))
    assert bits_per_symbol < ceiling
    assert balanced.size_in_bits() <= n * ceiling
    assert huffman.bits_per_symbol() == bits_per_symbol
    assert huffman.entropy() == pytest.approx(h0, abs=1e-12)
    assert huffman.decode() == text and balanced.decode() == text


def test_huffman_payload_on_readme_is_an_optimal_prefix_code() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    huffman = WaveletTree(text, shape="huffman")
    assert huffman.size_in_bits() == _optimal_prefix_cost(Counter(text).values())


# -------------------------------------------------------- synthetic DNA


def test_anchor_synthetic_dna_two_bits_per_symbol_and_ratio() -> None:
    dna = _synthetic_dna(DNA_LENGTH, 42)
    wt = WaveletTree(dna)
    assert wt.size_in_bits() == 2 * DNA_LENGTH
    assert wt.bits_per_symbol() == 2.0
    assert wt.compression_ratio() < 0.30
    assert wt.decode() == dna


def test_synthetic_dna_space_accounting_is_exact() -> None:
    dna = _synthetic_dna(DNA_LENGTH, 42)
    wt = WaveletTree(dna)
    counts = Counter(dna)
    left = counts["A"] + counts["C"]  # the balanced root sends A, C left and G, T right
    right = counts["G"] + counts["T"]
    expected_overhead = (
        _rank_directory_bits(DNA_LENGTH) + _rank_directory_bits(left) + _rank_directory_bits(right)
    )
    assert wt.overhead_bits() == expected_overhead
    assert wt.compression_ratio() == (2 * DNA_LENGTH + expected_overhead) / (8 * DNA_LENGTH)
    assert 0 < wt.select_sample_bits() < wt.overhead_bits()


def test_synthetic_dna_agrees_with_the_dna_index_anchors() -> None:
    # Spec 9.7 numbers for synthetic_dna(100000, 42), read through the wavelet tree.
    dna = _synthetic_dna(DNA_LENGTH, 42)
    wt = WaveletTree(dna, alphabet="ACGT")
    assert dna.startswith("ATGCCTAGAAGTGTGTGATC")
    assert wt.frequencies() == {"A": 25_021, "C": 24_890, "G": 25_154, "T": 24_935}
    assert wt.range_count("A", 1000, 2000) == 254  # inclusive [1000, 1999]
    assert [wt.range_quantile(5000, 5100, k) for k in (1, 50, 100)] == ["A", "G", "T"]


def test_synthetic_dna_huffman_shape_stays_at_two_bits() -> None:
    dna = _synthetic_dna(DNA_LENGTH, 42)
    wt = WaveletTree(dna, shape="huffman")
    assert wt.size_in_bits() == 2 * DNA_LENGTH  # four near-equal counts: all codes have length 2
    assert sorted(len(code) for code in wt.codes().values()) == [2, 2, 2, 2]


def test_million_symbol_build_and_queries() -> None:
    n = 1_000_000
    dna = _synthetic_dna(n, 7)
    wt = WaveletTree(dna, alphabet="ACGT")
    assert wt.size_in_bits() == 2 * n
    assert wt.compression_ratio() < 0.30
    rng = _query_rng("million")
    for _ in range(20):
        l, r = _random_range(rng, n)
        c = "ACGT"[int(rng.integers(0, 4))]
        assert wt.range_count(c, l, r) == dna.count(c, l, r)
        i = int(rng.integers(0, n))
        assert wt.access(i) == dna[i]
    assert wt.decode() == dna


# --------------------------------------------------- shapes and payload


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_payload_equals_count_times_code_length(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape=shape)
    codes = wt.codes()
    expected = 0
    for c, count in Counter(text).items():
        expected += count * len(codes[c])
    assert wt.size_in_bits() == expected


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_codes_are_prefix_free(case: Tuple[int, str, str]) -> None:
    sigma, shape, dist = case
    words = list(WaveletTree(_random_text(sigma, dist), shape=shape).codes().values())
    for a in words:
        for b in words:
            if a is not b:
                assert not b.startswith(a)


@pytest.mark.parametrize("sigma", [1, 2, 3, 4, 5, 7, 8, 26, 99])
def test_balanced_depths_are_floor_or_ceil_of_log2_sigma(sigma: int) -> None:
    alphabet = "".join(chr(33 + j) for j in range(sigma))
    wt = WaveletTree(alphabet * 3)
    lengths = [len(code) for code in wt.codes().values()]
    low = math.floor(math.log2(sigma))
    high = math.ceil(math.log2(sigma))
    assert set(lengths) <= {low, high}
    assert wt.size_in_bits() <= len(alphabet * 3) * high
    # the left half always gets ceil(m / 2), so the first leaf is a deepest one
    assert lengths[0] == high


@pytest.mark.parametrize("sigma", [2, 4, 8, 16])
def test_balanced_payload_is_exact_for_powers_of_two(sigma: int) -> None:
    alphabet = string.ascii_letters[:sigma]
    rng = _query_rng("power-of-two", sigma)
    text = "".join(alphabet[j] for j in rng.integers(0, sigma, size=5000).tolist())
    wt = WaveletTree(text, alphabet=alphabet)
    assert wt.size_in_bits() == len(text) * int(math.log2(sigma))


def test_balanced_split_gives_the_left_half_the_larger_part() -> None:
    assert WaveletTree("$ACGT").codes() == {
        "$": "000",
        "A": "001",
        "C": "01",
        "G": "10",
        "T": "11",
    }


@pytest.mark.parametrize("case", RANDOM_CASES, ids=_case_id)
def test_huffman_payload_is_optimal_and_within_one_bit_of_h0(case: Tuple[int, str, str]) -> None:
    sigma, _, dist = case
    text = _random_text(sigma, dist)
    wt = WaveletTree(text, shape="huffman")
    assert wt.size_in_bits() == _optimal_prefix_cost(Counter(text).values())
    h0 = zero_order_entropy(text)
    assert h0 <= wt.bits_per_symbol() < h0 + 1.0


def test_huffman_code_is_deterministic_with_ties_broken_by_symbol_order() -> None:
    assert WaveletTree("abracadabra", shape="huffman").codes() == {
        "a": "0",
        "b": "110",
        "c": "1110",
        "d": "1111",
        "r": "10",
    }
    assert WaveletTree("dcba", shape="huffman").codes() == {
        "a": "00",
        "b": "01",
        "c": "10",
        "d": "11",
    }
    text = _random_text(26, "uniform")
    assert WaveletTree(text, shape="huffman").codes() == WaveletTree(text, shape="huffman").codes()
    assert WaveletTree("abracadabra", shape="huffman").size_in_bits() == 23


# ------------------------------------------ alphabet, symbols, edge cases


@pytest.mark.parametrize("shape", SHAPES)
def test_explicit_alphabet_with_absent_symbols(shape: str) -> None:
    wt = WaveletTree("ACCA", alphabet="$ACGT", shape=shape)
    assert wt.alphabet == "$ACGT" and wt.sigma == 5
    assert wt.frequencies() == {"$": 0, "A": 2, "C": 2, "G": 0, "T": 0}
    assert wt.rank("G", 4) == 0 and wt.rank("$", 4) == 0 and wt.rank("N", 4) == 0
    assert wt.range_count("T", 0, 4) == 0
    assert wt.range_count_values(0, 4, "$", "A") == 2
    assert wt.range_count_values(0, 4, "G", "T") == 0
    assert wt.range_count_values(1, 3, "A", "T") == 2
    with pytest.raises(IndexError):
        wt.select("G", 1)
    with pytest.raises(IndexError):
        wt.select("N", 1)
    assert [wt.select("C", k) for k in (1, 2)] == [1, 2]
    assert wt.decode() == "ACCA"
    if shape == "balanced":
        assert wt.range_quantile(0, 4, 1) == "A" and wt.range_quantile(0, 4, 4) == "C"
        assert wt.size_in_bits() <= 4 * 3
    else:
        assert set(wt.codes()) == {"A", "C"}  # absent symbols get no code word
        assert wt.size_in_bits() == 4


@pytest.mark.parametrize("shape", SHAPES)
def test_bwt_alphabet_with_terminator(shape: str) -> None:
    bwt = "AC$TGAATCCGTG"  # spec 9.7: the BWT of "ACGTACGTTGCA$"
    wt = WaveletTree(bwt, alphabet="$ACGT", shape=shape)
    assert wt.decode() == bwt
    for c in "$ACGT":
        for i in range(len(bwt) + 1):
            assert wt.rank(c, i) == bwt[:i].count(c)


def test_explicit_alphabet_order_is_the_order_given() -> None:
    wt = WaveletTree("ACGTTA", alphabet="TGCA")
    assert wt.range_quantile(0, 6, 1) == "T"
    assert wt.range_quantile(0, 6, 6) == "A"
    assert wt.range_count_values(0, 6, "G", "C") == 2
    assert wt.range_count_values(0, 6, "C", "G") == 0  # C comes after G here
    assert list(wt.frequencies()) == ["T", "G", "C", "A"]


@pytest.mark.parametrize("shape", SHAPES)
def test_sequence_inputs_agree_with_the_string(shape: str) -> None:
    text = "GATTACA"
    expected = WaveletTree(text, shape=shape)
    for seq in (list(text), tuple(text), np.array(list(text))):
        wt = WaveletTree(seq, shape=shape)
        assert wt.decode() == text
        assert wt.codes() == expected.codes()
        assert [wt.rank("A", i) for i in range(8)] == [expected.rank("A", i) for i in range(8)]


@pytest.mark.parametrize("shape", SHAPES)
def test_non_ascii_symbols_round_trip(shape: str) -> None:
    text = "h\u00e9llo w\u00f6rld \u03bb\u2192\U0001f600 h\u00e9llo"
    wt = WaveletTree(text, shape=shape)
    assert wt.decode() == text
    assert wt.alphabet == "".join(sorted(set(text)))
    assert wt.rank("\u00e9", len(text)) == 2
    assert wt.select("\U0001f600", 1) == text.index("\U0001f600")
    assert [wt.access(i) for i in range(len(text))] == list(text)


@pytest.mark.parametrize("shape", SHAPES)
def test_single_symbol_sequence(shape: str) -> None:
    wt = WaveletTree("zzzz", shape=shape)
    assert wt.size_in_bits() == 0 and wt.overhead_bits() == 0
    assert wt.codes() == {"z": ""}
    assert [wt.access(i) for i in range(4)] == ["z"] * 4
    assert [wt.rank("z", i) for i in range(5)] == [0, 1, 2, 3, 4]
    assert [wt.select("z", k) for k in range(1, 5)] == [0, 1, 2, 3]
    assert wt.range_count_values(1, 3, "z", "z") == 2
    assert wt.decode() == "zzzz" and wt.entropy() == 0.0
    if shape == "balanced":
        assert wt.range_quantile(0, 4, 2) == "z"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("alphabet", [None, "ACGT"])
def test_empty_sequence(shape: str, alphabet) -> None:
    wt = WaveletTree("", alphabet=alphabet, shape=shape)
    assert len(wt) == 0
    assert wt.decode() == ""
    assert wt.size_in_bits() == 0 and wt.overhead_bits() == 0
    assert wt.rank("A", 0) == 0
    assert wt.entropy() == 0.0
    assert wt.frequencies() == {c: 0 for c in (alphabet or "")}
    with pytest.raises(IndexError):
        wt.access(0)
    with pytest.raises(IndexError):
        wt.select("A", 1)
    with pytest.raises(ValueError):
        wt.compression_ratio()
    with pytest.raises(ValueError):
        wt.bits_per_symbol()
    if alphabet:
        assert wt.range_count_values(0, 0, "A", "T") == 0
    if shape == "balanced":
        with pytest.raises(IndexError):
            wt.range_quantile(0, 0, 1)


def test_size_counts_lengths_not_padded_words() -> None:
    wt = WaveletTree("ab")
    assert wt.size_in_bits() == 2  # one node of 2 bits, stored in a 64-bit word
    wt = WaveletTree("ACGT" * 100)
    assert wt.size_in_bits() == 800  # 400 + 200 + 200, not 3 padded words of 64


# --------------------------------------------------------------- errors


def test_constructor_errors() -> None:
    with pytest.raises(ValueError):
        WaveletTree("ACGT", shape="avl")
    with pytest.raises(ValueError):
        WaveletTree("ACGT", shape=None)
    with pytest.raises(ValueError):
        WaveletTree("ACGN", alphabet="ACGT")
    with pytest.raises(ValueError):
        WaveletTree("ACGT", alphabet="ACGTA")
    with pytest.raises(ValueError):
        WaveletTree(["A", "CG"])
    with pytest.raises(TypeError):
        WaveletTree([1, 2, 3])
    with pytest.raises(TypeError):
        WaveletTree(b"ACGT")
    with pytest.raises(TypeError):
        WaveletTree(42)
    with pytest.raises(TypeError):
        WaveletTree("ACGT", alphabet=[1, 2])


def test_query_errors() -> None:
    wt = WaveletTree(ANCHOR_TEXT)
    n = len(ANCHOR_TEXT)
    for bad in (-1, n):
        with pytest.raises(IndexError):
            wt.access(bad)
    for bad in (-1, n + 1):
        with pytest.raises(IndexError):
            wt.rank("A", bad)
    with pytest.raises(TypeError):
        wt.rank("A", 1.5)
    with pytest.raises(TypeError):
        wt.access(True)
    for k in (0, 4):
        with pytest.raises(IndexError):
            wt.select("A", k)
    with pytest.raises(IndexError):
        wt.range_count("A", 0, n + 1)
    with pytest.raises(ValueError):
        wt.range_count("A", 5, 3)
    with pytest.raises(ValueError):
        wt.range_count_values(0, n, "A", "N")
    with pytest.raises(ValueError):
        wt.range_count_values(0, n, 0, "T")
    with pytest.raises(ValueError):
        wt.range_quantile(5, 3, 1)
    for k in (0, 7):
        with pytest.raises(IndexError):
            wt.range_quantile(3, 9, k)
    assert wt.range_count("N", 0, n) == 0
    assert wt.rank(3, 5) == 0  # not a symbol of the alphabet


def test_entropy_function() -> None:
    assert zero_order_entropy("aabb") == 1.0
    assert zero_order_entropy("ACGT" * 7) == 2.0
    assert zero_order_entropy("") == 0.0
    assert zero_order_entropy("aaaa") == 0.0
    assert zero_order_entropy(list("abracadabra")) == pytest.approx(_numpy_entropy("abracadabra"))
    text = _random_text(26, "zipf")
    assert WaveletTree(text).entropy() == pytest.approx(zero_order_entropy(text), abs=1e-12)


def test_repr_and_properties() -> None:
    wt = WaveletTree("banana", shape="huffman")
    assert repr(wt) == "<WaveletTree n=6, sigma=3, shape='huffman'>"
    assert (len(wt), wt.alphabet, wt.sigma, wt.shape) == (6, "abn", 3, "huffman")


# ----------------------------------------------- spec 1.1: built by hand


_FORBIDDEN_ATTRIBUTES = {
    "bit_count",
    "count",
    "count_nonzero",
    "cumsum",
    "index",
    "lexsort",
    "searchsorted",
    "sort",
    "argsort",
    "sum",
    "unpackbits",
}
_FORBIDDEN_CALLS = {"bin", "sorted", "sum"}


def _forbidden_uses(module) -> List[str]:
    """Names of library shortcuts used in a module's code (docstrings ignored)."""
    found = []
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            found.append(f"line {node.lineno}: .{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            found.append(f"line {node.lineno}: {node.func.id}()")
        elif isinstance(node, ast.Import) and any(a.name == "bisect" for a in node.names):
            found.append(f"line {node.lineno}: import bisect")
        elif isinstance(node, ast.ImportFrom) and node.module == "bisect":
            found.append(f"line {node.lineno}: from bisect import")
    return found


def test_no_library_counting_or_searching_in_the_wavelet_tree() -> None:
    assert _forbidden_uses(wavelet_module) == []


def test_every_node_uses_a_rank_select_over_a_bit_vector() -> None:
    from src.advds.succinct.bit_vector import BitVector
    from src.advds.succinct.rank_select import RankSelect

    for shape in SHAPES:
        wt = WaveletTree(_random_text(26, "zipf"), shape=shape)
        nodes = wt._internal
        assert len(nodes) == len(wt.codes()) - 1  # a full binary tree
        for node in nodes:
            assert isinstance(node.rs, RankSelect)
            assert isinstance(node.rs.bitvector, BitVector) and node.rs.bitvector.frozen


@pytest.mark.parametrize("shape", SHAPES)
def test_build_shares_one_side_table_and_leaves_it_clear(shape: str) -> None:
    # Regression: every internal node used to allocate its own sigma-entry
    # lookup table, an O(sigma^2) build against the documented
    # O(n h + sigma log sigma). The build now shares one table, and each node
    # must clear the entries it set, or a later node would misroute symbols.
    sigma = 300
    alphabet = "".join(chr(0x100 + j) for j in range(sigma))
    rng = _query_rng("side-table", shape)
    weights = 1.0 / np.arange(1, sigma + 1)
    picks = rng.choice(sigma, size=6000, p=weights / weights.sum())
    text = "".join(alphabet[j] for j in picks.tolist())
    if shape == "balanced":
        tree_shape = wavelet_module._balanced_shape(0, sigma)
    else:
        tree_shape = wavelet_module._huffman_shape(np.bincount(picks, minlength=sigma).tolist())
    side = np.zeros(sigma, dtype=bool)
    root = wavelet_module._build_node(tree_shape, picks.astype(np.int64), side)
    assert not side.any()
    assert wavelet_module._rebuild(root, len(text)).tolist() == picks.tolist()
    wt = WaveletTree(text, alphabet=alphabet, shape=shape)
    assert wt.decode() == text
    assert wt.frequencies() == {c: text.count(c) for c in alphabet}
