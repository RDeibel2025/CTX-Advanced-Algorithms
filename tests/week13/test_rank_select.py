"""Week 13 tests: packed bit vector and rank/select (spec 4.1, 4.2 and the 9.4 anchors).

The rulers are independent of the code under test (spec 1.1 allows them in
tests only): ``int.bit_count`` on a Python integer built from the bit string,
``np.flatnonzero`` for the positions of the 1s and 0s, and ``np.cumsum`` for
exhaustive rank tables on short vectors.

Anchors (9.4), computed with a brute-force reference on 11/16:

* ``"1011001110001011"``: rank1(0..16) and select1(1..9) as listed below.
* ``np.random.default_rng(42).random(1000) < 0.5`` has 503 ones;
  rank1 at 0, 1, 500, 1000 is 0, 0, 249, 503 and select1 at 1, 100, 250, 503
  is 1, 192, 506, 999.
* 20 random vectors of lengths 1 .. 70,000 at densities 0.01, 0.5 and 0.99
  agree with the rulers at every tested position, for rank1, rank0, select1
  and select0, and satisfy ``rank1(select1(k)) == k - 1`` and
  ``rank1(select1(k) + 1) == k`` (and the same for 0s).
* For n = 2^20, ``overhead_bits() / n`` is 0.078125 within 1%.

Seeds come from ``derive_seed(42, ...)``. No timings are asserted.
"""

from __future__ import annotations

import ast
import inspect
from typing import Iterable, List

import numpy as np
import pytest

import src.advds.succinct.bit_vector as bit_vector_module
import src.advds.succinct.rank_select as rank_select_module
from src.advds.succinct.bit_vector import BitVector
from src.advds.succinct.rank_select import (
    BLOCK_BITS,
    POPCOUNT8,
    SELECT8,
    SELECT_SAMPLE_RATE,
    SUPERBLOCK_BITS,
    RankSelect,
)
from src.utils.random_utils import DEFAULT_SEED, derive_seed

ANCHOR_STRING = "1011001110001011"
ANCHOR_RANK1 = [0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 6, 7, 7, 8, 9]
ANCHOR_SELECT1 = [0, 2, 3, 6, 7, 8, 12, 14, 15]

DENSITIES = (0.01, 0.5, 0.99)
N_RANDOM_VECTORS = 20
MAX_RANDOM_LENGTH = 70_000

# Exhaustive below this size; sampled (all boundaries plus random) above it.
EXHAUSTIVE_LIMIT = 4096
N_RANDOM_PROBES = 1500


# ---------------------------------------------------------------- rulers


def _as_python_int(bits: np.ndarray) -> int:
    """Return the integer whose bit i is ``bits[i]``, built from a string."""
    text = "".join("1" if b else "0" for b in bits[::-1].tolist())
    return int(text, 2) if text else 0


def _ref_rank1(big: int, i: int) -> int:
    """Number of 1s in positions [0, i) via ``int.bit_count``."""
    return (big & ((1 << i) - 1)).bit_count()


def _tested_positions(n: int, rng: np.random.Generator) -> List[int]:
    """Every position for short vectors; every word, block and superblock
    boundary (and its neighbours) plus random positions for long ones."""
    if n <= EXHAUSTIVE_LIMIT:
        return list(range(n + 1))
    chosen = {0, 1, n - 1, n}
    for step in (64, BLOCK_BITS, SUPERBLOCK_BITS):
        for m in range(0, n + 1, step):
            for d in (-1, 0, 1):
                if 0 <= m + d <= n:
                    chosen.add(m + d)
    chosen.update(rng.integers(0, n + 1, N_RANDOM_PROBES).tolist())
    return sorted(chosen)


def _tested_ranks(m: int, rng: np.random.Generator) -> List[int]:
    """Every k in 1..m for small m; otherwise every sample boundary
    (k = 512 t + 1 and its neighbours), the ends and random k."""
    if m <= EXHAUSTIVE_LIMIT:
        return list(range(1, m + 1))
    chosen = {1, 2, m - 1, m}
    for base in range(0, m + 1, SELECT_SAMPLE_RATE):
        for d in (-1, 0, 1, 2):
            if 1 <= base + d <= m:
                chosen.add(base + d)
    chosen.update(rng.integers(1, m + 1, N_RANDOM_PROBES).tolist())
    return sorted(chosen)


def _check_against_rulers(bits: np.ndarray, rng: np.random.Generator) -> RankSelect:
    """Assert rank1/rank0/select1/select0 and the identities on ``bits``."""
    n = int(bits.shape[0])
    rs = RankSelect(BitVector.from_bits(bits))
    big = _as_python_int(bits)
    ones = np.flatnonzero(bits)
    zeros = np.flatnonzero(~bits)
    assert len(rs) == n
    assert rs.num_ones == len(ones) == big.bit_count()
    assert rs.num_zeros == len(zeros)

    for i in _tested_positions(n, rng):
        expected = _ref_rank1(big, i)
        assert rs.rank1(i) == expected, f"rank1({i}), n={n}"
        assert rs.rank0(i) == i - expected, f"rank0({i}), n={n}"

    for k in _tested_ranks(len(ones), rng):
        p = rs.select1(k)
        assert p == int(ones[k - 1]), f"select1({k}), n={n}"
        assert rs.rank1(p) == k - 1
        assert rs.rank1(p + 1) == k

    for k in _tested_ranks(len(zeros), rng):
        p = rs.select0(k)
        assert p == int(zeros[k - 1]), f"select0({k}), n={n}"
        assert rs.rank0(p) == k - 1
        assert rs.rank0(p + 1) == k
    return rs


# ------------------------------------------------------------ the tables


def test_popcount8_matches_bit_count() -> None:
    assert len(POPCOUNT8) == 256
    assert POPCOUNT8 == [x.bit_count() for x in range(256)]


def test_select8_matches_positions_of_ones() -> None:
    assert len(SELECT8) == 256
    for x in range(256):
        positions = [p for p in range(8) if (x >> p) & 1]
        assert SELECT8[x] == positions + [-1] * (8 - len(positions))


# ------------------------------------------------------------ BitVector


def test_bit_vector_layout_bit_i_is_word_i_div_64_bit_i_mod_64() -> None:
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-bitvector-layout"))
    bits = rng.random(1000) < 0.5
    bv = BitVector.from_bits(bits)
    words = bv.words.tolist()
    assert len(words) == (1000 + 63) // 64
    for i in range(1000):
        assert (words[i // 64] >> (i % 64)) & 1 == int(bits[i])
        assert bv[i] == int(bits[i])
    # padding bits of the last word are 0
    assert words[-1] >> (1000 % 64) == 0


@pytest.mark.parametrize("n", [0, 1, 7, 63, 64, 65, 200, 4097])
def test_bit_vector_inputs_agree(n: int) -> None:
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-bitvector-inputs", n))
    bits = rng.random(n) < 0.5
    text = "".join("1" if b else "0" for b in bits.tolist())
    variants = [
        BitVector.from_bits(bits),
        BitVector.from_bits(bits.astype(np.int64)),
        BitVector.from_bits(bits.astype(np.uint8)),
        BitVector.from_bits(bits.tolist()),
        BitVector.from_bits([int(b) for b in bits.tolist()]),
        BitVector.from_bits(iter(bits.tolist())),
        BitVector.from_string(text),
    ]
    for bv in variants:
        assert len(bv) == n
        assert bv.to_string() == text
        assert list(bv) == [int(b) for b in bits.tolist()]
        assert bv.size_in_bits() == 64 * ((n + 63) // 64)
        assert bv.words.tolist() == variants[0].words.tolist()


def test_bit_vector_set_get_and_padding() -> None:
    bv = BitVector(130)
    assert bv.to_string() == "0" * 130
    for i in (0, 63, 64, 127, 129):
        bv.set(i, 1)
    assert [i for i in range(130) if bv[i]] == [0, 63, 64, 127, 129]
    assert bv.words.tolist() == [1 | (1 << 63), 1 | (1 << 63), 2]
    bv.set(63, False)
    bv.set(64, np.bool_(False))
    assert [i for i in range(130) if bv[i]] == [0, 127, 129]
    assert bv.size_in_bits() == 192


def test_bit_vector_words_are_read_only() -> None:
    bv = BitVector.from_string("101")
    with pytest.raises(ValueError):
        bv.words[0] = 0
    assert bv.to_string() == "101"


def test_bit_vector_freezes_when_rank_select_is_built() -> None:
    bv = BitVector.from_string("0101")
    bv.set(0, 1)
    assert not bv.frozen
    rs = RankSelect(bv)
    assert bv.frozen
    assert rs.bitvector is bv
    with pytest.raises(RuntimeError):
        bv.set(1, 0)
    assert bv.to_string() == "1101"
    assert rs.rank1(4) == 3


def test_bit_vector_errors() -> None:
    bv = BitVector(4)
    with pytest.raises(IndexError):
        bv[4]
    with pytest.raises(IndexError):
        bv[-1]
    with pytest.raises(IndexError):
        bv.set(4, 1)
    with pytest.raises(TypeError):
        bv[1.0]
    with pytest.raises(TypeError):
        bv[True]
    with pytest.raises(ValueError):
        bv.set(0, 2)
    with pytest.raises(TypeError):
        bv.set(0, "1")
    with pytest.raises(ValueError):
        BitVector(-3)
    with pytest.raises(ValueError):
        BitVector.from_bits([0, 1, 2])
    with pytest.raises(ValueError):
        BitVector.from_bits(np.array([0, 1, -1]))
    with pytest.raises(ValueError):
        BitVector.from_bits(np.zeros((2, 2), dtype=bool))
    with pytest.raises(TypeError):
        BitVector.from_bits([0.0, 1.0])
    with pytest.raises(TypeError):
        BitVector.from_bits("0101")
    with pytest.raises(ValueError):
        BitVector.from_string("01x1")
    with pytest.raises(TypeError):
        BitVector.from_string(101)


def test_bit_vector_empty() -> None:
    for bv in (BitVector(), BitVector(0), BitVector.from_bits([]), BitVector.from_string("")):
        assert len(bv) == 0
        assert bv.size_in_bits() == 0
        assert bv.words.shape == (0,)
        assert list(bv) == []
        with pytest.raises(IndexError):
            bv[0]


# ------------------------------------------------------- 9.4 anchors


def test_anchor_sixteen_bit_string() -> None:
    rs = RankSelect(BitVector.from_string(ANCHOR_STRING))
    assert [rs.rank1(i) for i in range(17)] == ANCHOR_RANK1
    assert [rs.select1(k) for k in range(1, 10)] == ANCHOR_SELECT1
    assert [rs.rank0(i) for i in range(17)] == [i - r for i, r in enumerate(ANCHOR_RANK1)]
    assert [rs.select0(k) for k in range(1, 8)] == [
        i for i, ch in enumerate(ANCHOR_STRING) if ch == "0"
    ]


def test_anchor_seeded_thousand_bits() -> None:
    bits = np.random.default_rng(42).random(1000) < 0.5
    rs = RankSelect(BitVector.from_bits(bits))
    assert rs.num_ones == 503
    assert [rs.rank1(i) for i in (0, 1, 500, 1000)] == [0, 0, 249, 503]
    assert [rs.select1(k) for k in (1, 100, 250, 503)] == [1, 192, 506, 999]


def _random_vector_cases() -> List[tuple]:
    """20 (index, length, density) cases: lengths 1 and 70,000 plus 18 random
    lengths in between, densities cycling through 0.01, 0.5 and 0.99."""
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-rank-select-lengths"))
    lengths = [1, MAX_RANDOM_LENGTH] + rng.integers(
        2, MAX_RANDOM_LENGTH, N_RANDOM_VECTORS - 2
    ).tolist()
    return [(idx, n, DENSITIES[idx % 3]) for idx, n in enumerate(lengths)]


@pytest.mark.parametrize("idx,n,density", _random_vector_cases())
def test_anchor_random_vectors_match_rulers(idx: int, n: int, density: float) -> None:
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-rank-select-vector", idx))
    bits = rng.random(n) < density
    _check_against_rulers(bits, rng)


def test_random_vector_cases_cover_the_spec_ranges() -> None:
    cases = _random_vector_cases()
    assert len(cases) == N_RANDOM_VECTORS
    assert min(n for _, n, _ in cases) == 1
    assert max(n for _, n, _ in cases) == MAX_RANDOM_LENGTH
    assert {d for _, _, d in cases} == set(DENSITIES)


def test_anchor_overhead_ratio_at_two_to_the_twenty() -> None:
    n = 1 << 20
    bits = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-overhead")).random(n) < 0.5
    rs = RankSelect(BitVector.from_bits(bits))
    ratio = rs.overhead_bits() / n
    assert ratio == pytest.approx(0.078125, rel=0.01)
    # the stated layout: 32 bits per 2,048-bit superblock + 16 bits per 256-bit block
    assert rs.overhead_bits() == 32 * (n // SUPERBLOCK_BITS) + 16 * (n // BLOCK_BITS)


# ------------------------------------------------------------ edge cases


def test_length_zero() -> None:
    rs = RankSelect(BitVector(0))
    assert len(rs) == 0
    assert rs.rank1(0) == 0
    assert rs.rank0(0) == 0
    assert rs.num_ones == rs.num_zeros == 0
    assert rs.overhead_bits() == 0
    assert rs.select_sample_bits() == 0
    for call in (rs.select1, rs.select0):
        with pytest.raises(IndexError):
            call(1)
    with pytest.raises(IndexError):
        rs.rank1(1)


@pytest.mark.parametrize("text", ["0", "1"])
def test_length_one(text: str) -> None:
    rs = RankSelect(BitVector.from_string(text))
    one = int(text)
    assert rs.rank1(0) == 0 and rs.rank1(1) == one
    assert rs.rank0(0) == 0 and rs.rank0(1) == 1 - one
    if one:
        assert rs.select1(1) == 0
        with pytest.raises(IndexError):
            rs.select0(1)
    else:
        assert rs.select0(1) == 0
        with pytest.raises(IndexError):
            rs.select1(1)
    assert rs.overhead_bits() == 32 + 16


EDGE_LENGTHS = [2, 63, 64, 65, 255, 256, 257, 511, 513, 2047, 2048, 2049, 4095, 4096, 4097, 6001]


@pytest.mark.parametrize("n", EDGE_LENGTHS)
def test_all_zeros(n: int) -> None:
    rs = RankSelect(BitVector(n))
    assert all(rs.rank1(i) == 0 for i in range(n + 1))
    assert [rs.select0(k) for k in range(1, n + 1)] == list(range(n))
    with pytest.raises(IndexError):
        rs.select1(1)
    with pytest.raises(IndexError):
        rs.select0(n + 1)


@pytest.mark.parametrize("n", EDGE_LENGTHS)
def test_all_ones(n: int) -> None:
    rs = RankSelect(BitVector.from_bits(np.ones(n, dtype=bool)))
    assert [rs.rank1(i) for i in range(n + 1)] == list(range(n + 1))
    assert [rs.select1(k) for k in range(1, n + 1)] == list(range(n))
    with pytest.raises(IndexError):
        rs.select0(1)
    with pytest.raises(IndexError):
        rs.select1(n + 1)


@pytest.mark.parametrize("n", EDGE_LENGTHS)
@pytest.mark.parametrize("density", DENSITIES)
def test_lengths_off_word_block_and_superblock_boundaries(n: int, density: float) -> None:
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-rank-select-edge", n, density))
    bits = rng.random(n) < density
    rs = RankSelect(BitVector.from_bits(bits))
    # exhaustive, against an independent prefix-sum table as well
    prefix = np.concatenate([[0], np.cumsum(bits)]).tolist()
    assert [rs.rank1(i) for i in range(n + 1)] == prefix
    assert [rs.select1(k) for k in range(1, rs.num_ones + 1)] == np.flatnonzero(bits).tolist()
    assert [rs.select0(k) for k in range(1, rs.num_zeros + 1)] == np.flatnonzero(~bits).tolist()


def test_select0_never_returns_padding() -> None:
    # 70 bits: the last word has 58 padding bits that must not count as 0s.
    bits = np.ones(70, dtype=bool)
    bits[[3, 69]] = False
    rs = RankSelect(BitVector.from_bits(bits))
    assert rs.num_zeros == 2
    assert [rs.select0(1), rs.select0(2)] == [3, 69]
    with pytest.raises(IndexError):
        rs.select0(3)


def test_select_across_wide_gaps_between_samples() -> None:
    # Clustered 1s: samples far apart in superblocks, so the binary search
    # between two neighbouring samples has a long range to cover.
    n = 1 << 18
    bits = np.zeros(n, dtype=bool)
    bits[5] = True
    bits[3000:3000 + 1100] = True
    bits[[150_000, 150_001, n - 1]] = True
    rs = _check_against_rulers(bits, np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-gaps")))
    ones = np.flatnonzero(bits)
    assert [rs.select1(k) for k in range(1, len(ones) + 1)] == ones.tolist()


def test_out_of_range_and_type_errors() -> None:
    rs = RankSelect(BitVector.from_string(ANCHOR_STRING))
    for bad in (-1, 17):
        with pytest.raises(IndexError):
            rs.rank1(bad)
        with pytest.raises(IndexError):
            rs.rank0(bad)
    for bad in (0, 10, -1):
        with pytest.raises(IndexError):
            rs.select1(bad)
    for bad in (0, 8):
        with pytest.raises(IndexError):
            rs.select0(bad)
    with pytest.raises(TypeError):
        rs.rank1(1.0)
    with pytest.raises(TypeError):
        rs.select1(True)
    with pytest.raises(TypeError):
        RankSelect([1, 0, 1])
    # numpy integers are accepted as indices
    assert rs.rank1(np.int64(16)) == 9
    assert rs.select1(np.int32(9)) == 15


def test_select_sample_bits_counts_both_kinds() -> None:
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-samples"))
    n = 100_000
    bits = rng.random(n) < 0.3
    rs = RankSelect(BitVector.from_bits(bits))
    ones = int(np.count_nonzero(bits))
    zeros = n - ones
    # one uint32 per 512 bits of each kind, plus one closing sample per kind
    expected = 32 * (
        -(-ones // SELECT_SAMPLE_RATE) + 1 + -(-zeros // SELECT_SAMPLE_RATE) + 1
    )
    assert rs.select_sample_bits() == expected
    # a kind that does not occur stores no samples at all
    assert RankSelect(BitVector.from_bits(np.ones(1000, dtype=bool))).select_sample_bits() == (
        32 * (-(-1000 // SELECT_SAMPLE_RATE) + 1)
    )


class _CountingReads:
    """Wrap a directory and count the reads made through it (a step count, not a timing)."""

    def __init__(self, data) -> None:
        self.data = data
        self.reads = 0

    def __getitem__(self, index: int) -> int:
        self.reads += 1
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


@pytest.mark.parametrize("kind", [1, 0])
def test_select_binary_search_stops_at_the_last_bit_of_its_kind(kind: int) -> None:
    # Regression: queries past the last regular sample used to binary-search
    # on to the end of the vector (about 480 superblocks here, up to 9
    # superblock reads per query). The closing sample (superblock of the last 1 / 0)
    # keeps the search between two neighbouring samples: with every
    # superblock of the dense head holding far more than 512 bits of the
    # kind, the range is at most 2 superblocks, so at most 2 reads.
    n = 1 << 20
    rng = np.random.default_rng(derive_seed(DEFAULT_SEED, "w13-closing-sample"))
    head = rng.random(n // 16) < 0.5
    bits = np.zeros(n, dtype=bool) if kind else np.ones(n, dtype=bool)
    bits[: n // 16] = head
    rs = RankSelect(BitVector.from_bits(bits))
    positions = np.flatnonzero(bits if kind else ~bits).tolist()
    select = rs.select1 if kind else rs.select0
    counter = _CountingReads(rs._super)
    rs._super = counter
    worst = 0
    for k in range(1, len(positions) + 1):
        counter.reads = 0
        assert select(k) == positions[k - 1]
        worst = max(worst, counter.reads)
    assert worst <= 2


def test_from_bits_errors_hold_for_values_numpy_cannot_type() -> None:
    # Regression: these used to raise TypeError ("got dtype object/float64")
    # although they hold integers, and valid 0/1 object arrays were rejected.
    for bad in ([0, 2**64], [0, 2**70], [np.uint64(1), -1]):
        with pytest.raises(ValueError):
            BitVector.from_bits(bad)
    for bad in ([0, 0.5], [None, 1], np.array([0, None], dtype=object)):
        with pytest.raises(TypeError):
            BitVector.from_bits(bad)
    assert BitVector.from_bits(np.array([0, 1, 1], dtype=object)).to_string() == "011"
    assert BitVector.from_bits([np.uint64(1), 0, np.int8(1)]).to_string() == "101"


# ----------------------------------------------- spec 1.1: built by hand


_FORBIDDEN_ATTRIBUTES = {
    "bit_count",
    "cumsum",
    "searchsorted",
    "count_nonzero",
    "unpackbits",
    "sum",
    "lexsort",
}
_FORBIDDEN_CALLS = {"bin", "sorted", "sum"}


def _forbidden_uses(module) -> Iterable[str]:
    """Names of library shortcuts used in a module's code (docstrings ignored)."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            yield f"line {node.lineno}: .{node.attr}"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            yield f"line {node.lineno}: {node.func.id}()"
        elif isinstance(node, ast.Import) and any(a.name == "bisect" for a in node.names):
            yield f"line {node.lineno}: import bisect"
        elif isinstance(node, ast.ImportFrom) and node.module == "bisect":
            yield f"line {node.lineno}: from bisect import"


@pytest.mark.parametrize("module", [bit_vector_module, rank_select_module])
def test_no_library_counting_or_searching_in_the_structures(module) -> None:
    assert list(_forbidden_uses(module)) == []
