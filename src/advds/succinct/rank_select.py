"""Constant-time rank and sampled select over a packed :class:`BitVector`.

The idea
--------
``rank1(i)`` counts the 1 bits before position i. Counting them one by one
costs O(n); storing every answer costs n words. Jacobson's two-level
directory sits in between: it cuts the bit vector into fixed levels and
stores one count per piece, so a query adds two stored counts and finishes
with a bounded amount of table lookups.

The fixed layout (spec 4.2)
---------------------------
* **Superblock = 2,048 bits (32 words, 8 blocks).** Stores the absolute
  number of 1s before it as a ``uint32``: 32 / 2048 = 1.5625% of n.
* **Block = 256 bits (4 words, 32 bytes).** Stores the number of 1s between
  the start of its superblock and the start of the block as a ``uint16``
  (at most 7 * 256 = 1,792): 16 / 256 = 6.25% of n.
* **Byte = 8 bits.** ``POPCOUNT8[x]`` is the number of 1s in byte x and
  ``SELECT8[x][j]`` the position of the (j+1)-th 1 inside byte x, or -1.
  Both 256-entry tables are computed by hand below (256 bytes and 2 KB as
  ``uint8`` data) and are not counted in :meth:`RankSelect.overhead_bits`.

So the rank directory costs ``32 * ceil(n / 2048) + 16 * ceil(n / 256)``
bits, which is exactly 7.8125% of n when n is a multiple of 2,048.

Rank: O(1)
----------
``rank1(i) = superblock[i // 2048] + block[i // 256] + (1s from the start of
block i // 256 up to i)``. The last term is at most 31 whole-byte lookups in
``POPCOUNT8`` plus one masked byte, so at most 32 lookups: a fixed number of
steps whatever n is. ``rank0(i) = i - rank1(i)``.

Select: O(1) except a bounded binary search
-------------------------------------------
``select1(k)`` returns the position of the k-th 1 bit, k 1-based. The
structure stores, for every 512th 1 bit (the 1st, the 513th, the 1025th, ...),
the index of the superblock that holds it, as a ``uint32``, and one closing
sample: the superblock that holds the last 1 bit. For a query k the two
neighbouring samples bound the superblock of the answer (the closing sample
bounds queries past the last regular sample, so the search never runs on to
the end of the vector), and a binary search runs only over the superblocks
between them. Inside that superblock the code scans at most 8 block counts,
then at most 4 words (each counted with 8 ``POPCOUNT8`` lookups) and at most
8 bytes, and finishes with one ``SELECT8`` lookup. Everything after the
binary search is a constant number of steps. The binary search is
O(log g) for a gap of g superblocks between two samples: g is at most 1
when every superblock holds at least 512 1s, and grows only where the 1s
are sparse, so select is O(1) except for that bounded binary search.
``select0`` works the same way with its own samples of every 512th 0 bit
and of the last 0 bit. 0 bits are counted only over the n real positions:
the zero padding of the last word is never counted or returned.

Conventions (spec 2.3)
----------------------
* Positions are 0-based. ``rank1(i)`` counts 1s in ``[0, i)`` for
  ``0 <= i <= n``; ``select1(k)`` takes ``1 <= k <= number of 1s``. Hence
  ``rank1(select1(k)) == k - 1`` and ``rank1(select1(k) + 1) == k``, and the
  same for ``rank0`` / ``select0``.
* Out-of-range arguments raise ``IndexError``; non-integers raise
  ``TypeError``.
* Building a ``RankSelect`` freezes its :class:`BitVector`, so the directory
  can never go stale.

What numpy does here (spec 1.1)
-------------------------------
numpy arrays store the directories (``uint32`` superblocks and samples,
``uint16`` blocks) and the bit vector's words. The build looks up
``POPCOUNT8`` for every byte at once with numpy fancy indexing, which is a
table lookup and not a counting routine, and adds the 32 byte counts of each
block column by column. The superblock and block ranks and the samples are
then accumulated with explicit Python loops (no ``np.cumsum``). Queries read
the arrays through zero-copy ``memoryview`` objects and use only the two
hand-built tables, never ``int.bit_count``, ``bisect`` or a numpy counting
or searching call.

Examples:
    >>> rs = RankSelect(BitVector.from_string("1011001110001011"))
    >>> [rs.rank1(i) for i in (0, 1, 4, 16)]
    [0, 1, 3, 9]
    >>> [rs.select1(k) for k in (1, 2, 9)]
    [0, 2, 15]
    >>> rs.rank0(16), rs.select0(1)
    (7, 1)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import sys
from typing import List

import numpy as np

from src.advds.succinct.bit_vector import BitVector, _as_int

__all__ = [
    "BLOCK_BITS",
    "POPCOUNT8",
    "RankSelect",
    "SELECT8",
    "SELECT_SAMPLE_RATE",
    "SUPERBLOCK_BITS",
]

#: Bits per superblock (32 words); its absolute rank is stored as uint32.
SUPERBLOCK_BITS: int = 2048

#: Bits per block (4 words, 32 bytes); its rank relative to the superblock is stored as uint16.
BLOCK_BITS: int = 256

#: One select sample is stored for every this many 1 bits (and, separately, 0 bits).
SELECT_SAMPLE_RATE: int = 512

_BLOCKS_PER_SUPERBLOCK: int = SUPERBLOCK_BITS // BLOCK_BITS  # 8
_WORDS_PER_BLOCK: int = BLOCK_BITS // 64  # 4
_BYTES_PER_BLOCK: int = BLOCK_BITS // 8  # 32
_BYTES_PER_WORD: int = 8

#: The largest n whose ranks fit the uint32 superblock entries.
_MAX_BITS: int = (1 << 32) - 1


def _build_popcount8() -> List[int]:
    """Return the 256-entry table of 1-bit counts, built by recurrence.

    ``count(x) = count(x >> 1) + (x & 1)``: dropping the lowest bit leaves a
    smaller byte already in the table.

    Returns:
        A list ``t`` with ``t[x]`` = number of 1 bits in byte x.

    Time Complexity:
        O(256).

    Space Complexity:
        O(256).

    Examples:
        >>> t = _build_popcount8()
        >>> len(t), t[0], t[1], t[0b10110], t[255]
        (256, 0, 1, 3, 8)
    """
    table = [0] * 256
    for x in range(1, 256):
        table[x] = table[x >> 1] + (x & 1)
    return table


def _build_select8() -> List[List[int]]:
    """Return the 256 x 8 table of in-byte 1-bit positions.

    Row x lists the positions (0 = least significant bit) of the 1 bits of
    byte x in increasing order, padded with -1 to length 8.

    Returns:
        A list of 256 lists of 8 ints.

    Time Complexity:
        O(256 * 8).

    Space Complexity:
        O(256 * 8).

    Examples:
        >>> t = _build_select8()
        >>> t[0b10110]
        [1, 2, 4, -1, -1, -1, -1, -1]
        >>> t[0] == [-1] * 8, t[255] == list(range(8))
        (True, True)
    """
    table = []
    for x in range(256):
        row = []
        for pos in range(8):
            if (x >> pos) & 1:
                row.append(pos)
        row.extend([-1] * (8 - len(row)))
        table.append(row)
    return table


#: POPCOUNT8[x] = number of 1 bits in the byte x (0 <= x < 256).
POPCOUNT8: List[int] = _build_popcount8()

#: SELECT8[x][j] = position (0..7, least significant first) of the (j+1)-th 1 bit of byte x, or -1.
SELECT8: List[List[int]] = _build_select8()


class RankSelect:
    """Rank and select over a frozen :class:`BitVector` (spec 4.2 layout).

    The constructor freezes ``bitvector``, builds the superblock (uint32)
    and block (uint16) rank directories and the select samples of every
    512th 1 bit and every 512th 0 bit (each list closed by the superblock
    of the last bit of its kind).

    Args:
        bitvector: The bits to index. It is frozen and shared, not copied.

    Raises:
        TypeError: If ``bitvector`` is not a :class:`BitVector`.
        ValueError: If ``len(bitvector) >= 2**32``, which the uint32
            superblock ranks cannot represent.

    Time Complexity:
        O(n) to build: one vectorised ``POPCOUNT8`` lookup per byte, then an
        explicit loop over the n / 256 blocks and the n / 2048 superblocks.

    Space Complexity:
        ``overhead_bits()`` = 0.078125 n bits for the rank directory (exact
        when n is a multiple of 2,048) plus ``select_sample_bits()`` = 32
        bits per 512 ones and per 512 zeros (and one closing sample per
        kind); the payload is the bit vector's, not copied.

    Examples:
        >>> bv = BitVector.from_string("0110")
        >>> rs = RankSelect(bv)
        >>> rs.rank1(3), rs.select1(2), rs.select0(2)
        (2, 2, 3)
        >>> bv.frozen
        True
        >>> RankSelect("0110")
        Traceback (most recent call last):
            ...
        TypeError: RankSelect needs a BitVector, got str
    """

    __slots__ = (
        "_bv",
        "_n",
        "_ones",
        "_n_super",
        "_n_blocks",
        "_super_arr",
        "_block_arr",
        "_samples1_arr",
        "_samples0_arr",
        "_bytes",
        "_super",
        "_block",
        "_samples1",
        "_samples0",
    )

    def __init__(self, bitvector: BitVector) -> None:
        if not isinstance(bitvector, BitVector):
            raise TypeError(f"RankSelect needs a BitVector, got {type(bitvector).__name__}")
        n = len(bitvector)
        if n > _MAX_BITS:
            raise ValueError(f"RankSelect supports at most 2**32 - 1 bits, got {n}")
        bitvector.freeze()
        self._bv = bitvector
        self._n = n
        self._n_super = (n + SUPERBLOCK_BITS - 1) // SUPERBLOCK_BITS
        self._n_blocks = (n + BLOCK_BITS - 1) // BLOCK_BITS

        # Byte view of the words: bit i is bit i % 8 of byte i // 8 when the
        # words are read little-endian. On a little-endian machine this is a
        # zero-copy view of the bit vector's storage.
        words = bitvector.words
        if sys.byteorder != "little":  # pragma: no cover - big-endian hosts only
            words = words.astype("<u8")
        byte_view = words.view(np.uint8)

        block_counts = self._count_blocks(byte_view)
        super_ranks, block_ranks, ones = self._accumulate(block_counts)
        self._ones = ones
        self._super_arr = np.array(super_ranks, dtype=np.uint32)
        self._block_arr = np.array(block_ranks, dtype=np.uint16)
        samples1, samples0 = self._sample(super_ranks)
        self._samples1_arr = np.array(samples1, dtype=np.uint32)
        self._samples0_arr = np.array(samples0, dtype=np.uint32)

        # Zero-copy memoryviews: indexing them yields Python ints quickly.
        self._bytes = memoryview(byte_view)
        self._super = memoryview(self._super_arr)
        self._block = memoryview(self._block_arr)
        self._samples1 = memoryview(self._samples1_arr)
        self._samples0 = memoryview(self._samples0_arr)

    # ------------------------------------------------------------- build

    def _count_blocks(self, byte_view: np.ndarray) -> List[int]:
        """Return the number of 1 bits in each 256-bit block.

        ``POPCOUNT8`` is looked up for every byte at once with numpy fancy
        indexing (a table lookup), the byte counts are padded to whole
        blocks of 32 bytes, and the 32 columns are added one at a time.

        Args:
            byte_view: The bit vector's words viewed as little-endian bytes.

        Returns:
            A list of ``ceil(n / 256)`` block counts, each in ``[0, 256]``.

        Time Complexity:
            O(n / 8) vectorised byte lookups and additions.

        Space Complexity:
            O(n / 8) temporary bytes.

        Examples:
            >>> rs = RankSelect(BitVector.from_string("1" * 300))
            >>> rs._count_blocks(rs.bitvector.words.view(np.uint8))
            [256, 44]
        """
        table = np.array(POPCOUNT8, dtype=np.uint16)
        padded = np.zeros(self._n_blocks * _BYTES_PER_BLOCK, dtype=np.uint16)
        padded[: byte_view.shape[0]] = table[byte_view]
        grid = padded.reshape(self._n_blocks, _BYTES_PER_BLOCK)
        counts = grid[:, 0].copy()
        for column in range(1, _BYTES_PER_BLOCK):
            counts += grid[:, column]
        return counts.tolist()

    def _accumulate(self, block_counts: List[int]) -> tuple:
        """Turn block counts into superblock and block ranks with one loop.

        Args:
            block_counts: The number of 1s in each block, in order.

        Returns:
            ``(super_ranks, block_ranks, ones)``: the absolute rank before
            each superblock, the rank of each block relative to its
            superblock, and the total number of 1 bits.

        Time Complexity:
            O(n / 256).

        Space Complexity:
            O(n / 256).

        Examples:
            >>> rs = RankSelect(BitVector(0))
            >>> rs._accumulate([3] * 9)
            ([0, 24], [0, 3, 6, 9, 12, 15, 18, 21, 0], 27)
        """
        super_ranks: List[int] = []
        block_ranks: List[int] = []
        total = 0
        base = 0
        for b, count in enumerate(block_counts):
            if b % _BLOCKS_PER_SUPERBLOCK == 0:
                base = total
                super_ranks.append(total)
            block_ranks.append(total - base)
            total += count
        return super_ranks, block_ranks, total

    def _sample(self, super_ranks: List[int]) -> tuple:
        """Record the superblock of every 512th 1 bit and every 512th 0 bit.

        Sample t of the 1s is the superblock holding the (512 t + 1)-th 1;
        likewise for the 0s, which are counted only over the n real
        positions (the last superblock ends at n, not at its padded end).
        Each list then gets one closing sample, the superblock holding the
        last 1 (or the last 0), so every query has a right-hand neighbour
        sample and the binary search never runs past the last bit of its
        kind.

        Args:
            super_ranks: The absolute rank before each superblock.

        Returns:
            ``(samples1, samples0)``, lists of superblock indices with
            ``ceil(ones / 512) + 1`` and ``ceil(zeros / 512) + 1`` entries
            (no entries at all for a kind that does not occur).

        Time Complexity:
            O(n / 2048 + n / 512).

        Space Complexity:
            O(n / 512).

        Examples:
            >>> rs = RankSelect(BitVector.from_string("1" * 3000))
            >>> rs._sample([0, 2048])
            ([0, 0, 0, 0, 1, 1, 1], [])
            >>> RankSelect(BitVector.from_string("1" + "0" * 5000))._samples1.tolist()
            [0, 0]
        """
        n = self._n
        samples1: List[int] = []
        samples0: List[int] = []
        next1 = 1
        next0 = 1
        last1 = -1
        last0 = -1
        for s in range(self._n_super):
            ones_end = super_ranks[s + 1] if s + 1 < self._n_super else self._ones
            end = (s + 1) * SUPERBLOCK_BITS
            if end > n:
                end = n
            zeros_end = end - ones_end
            if ones_end > super_ranks[s]:
                last1 = s
            if zeros_end > (s << 11) - super_ranks[s]:
                last0 = s
            while next1 <= ones_end:
                samples1.append(s)
                next1 += SELECT_SAMPLE_RATE
            while next0 <= zeros_end:
                samples0.append(s)
                next0 += SELECT_SAMPLE_RATE
        if last1 >= 0:
            samples1.append(last1)
        if last0 >= 0:
            samples0.append(last0)
        return samples1, samples0

    # ------------------------------------------------------------- basics

    def __len__(self) -> int:
        """Return n, the number of bits indexed.

        Returns:
            ``len(bitvector)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(RankSelect(BitVector(10)))
            10
        """
        return self._n

    @property
    def bitvector(self) -> BitVector:
        """The (frozen) :class:`BitVector` this structure indexes.

        Returns:
            The bit vector passed to the constructor.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector.from_string("10")).bitvector.to_string()
            '10'
        """
        return self._bv

    @property
    def num_ones(self) -> int:
        """The total number of 1 bits, ``rank1(n)``.

        Returns:
            The number of 1 bits in the vector.

        Time Complexity:
            O(1) - stored at build time.

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector.from_string("10110")).num_ones
            3
        """
        return self._ones

    @property
    def num_zeros(self) -> int:
        """The total number of 0 bits among the n real positions, ``rank0(n)``.

        Returns:
            ``n - num_ones``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector.from_string("10110")).num_zeros
            2
        """
        return self._n - self._ones

    def access(self, i: int) -> int:
        """Return bit i (the same as ``bitvector[i]``).

        Args:
            i: A 0-based index with ``0 <= i < n``.

        Returns:
            0 or 1.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector.from_string("01")).access(1)
            1
        """
        return self._bv[i]

    # --------------------------------------------------------------- rank

    def rank1(self, i: int) -> int:
        """Return the number of 1 bits in positions ``[0, i)``.

        Adds the superblock rank, the block rank and at most 32 ``POPCOUNT8``
        lookups (31 whole bytes and one masked byte) from the start of the
        block to position i.

        Args:
            i: A position with ``0 <= i <= n``.

        Returns:
            The count of 1s strictly before position i.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n]``.

        Time Complexity:
            O(1) - two directory reads and at most 32 table lookups,
            whatever n is.

        Space Complexity:
            O(1).

        Examples:
            >>> rs = RankSelect(BitVector.from_string("1011001110001011"))
            >>> [rs.rank1(i) for i in range(17)]
            [0, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6, 6, 7, 7, 8, 9]
            >>> rs.rank1(17)
            Traceback (most recent call last):
                ...
            IndexError: rank position 17 out of range [0, 16]
        """
        i = _as_int(i, "i")
        if i < 0 or i > self._n:
            raise IndexError(f"rank position {i} out of range [0, {self._n}]")
        if i == self._n:
            return self._ones
        count = self._super[i >> 11] + self._block[i >> 8]
        table = POPCOUNT8
        data = self._bytes
        j = (i >> 8) << 5  # first byte of the block holding i
        stop = i >> 3  # the byte holding i
        while j < stop:  # at most 31 whole bytes
            count += table[data[j]]
            j += 1
        rest = i & 7
        if rest:  # the low bits of the byte holding i
            count += table[data[stop] & ((1 << rest) - 1)]
        return count

    def rank0(self, i: int) -> int:
        """Return the number of 0 bits in positions ``[0, i)``.

        Args:
            i: A position with ``0 <= i <= n``.

        Returns:
            ``i - rank1(i)``.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n]``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> rs = RankSelect(BitVector.from_string("1011001110001011"))
            >>> rs.rank0(0), rs.rank0(6), rs.rank0(16)
            (0, 3, 7)
        """
        i = _as_int(i, "i")
        return i - self.rank1(i)

    # ------------------------------------------------------------- select

    def select1(self, k: int) -> int:
        """Return the position of the k-th 1 bit (k is 1-based).

        The samples of every 512th 1 bit (plus the closing sample, the
        superblock of the last 1) bound the superblock of the answer; a
        binary search over only those superblocks finds it, then at most
        8 blocks, 4 words and 8 bytes are scanned and ``SELECT8`` gives the
        position inside the final byte.

        Args:
            k: A rank with ``1 <= k <= num_ones``.

        Returns:
            The position p with bit p = 1 and ``rank1(p) == k - 1``.

        Raises:
            TypeError: If ``k`` is not an integer.
            IndexError: If ``k`` is outside ``[1, num_ones]``.

        Time Complexity:
            O(1) except the binary search between two neighbouring samples,
            which is O(log g) for a gap of g superblocks between them
            (g <= 1 when every superblock holds at least 512 1s).

        Space Complexity:
            O(1).

        Examples:
            >>> rs = RankSelect(BitVector.from_string("1011001110001011"))
            >>> [rs.select1(k) for k in range(1, 10)]
            [0, 2, 3, 6, 7, 8, 12, 14, 15]
            >>> rs.select1(0)
            Traceback (most recent call last):
                ...
            IndexError: select1 rank 0 out of range [1, 9]
        """
        k = _as_int(k, "k")
        if k < 1 or k > self._ones:
            raise IndexError(f"select1 rank {k} out of range [1, {self._ones}]")
        sup = self._super
        samples = self._samples1
        t = (k - 1) >> 9  # sample t holds the (512 t + 1)-th 1, at or before the answer
        lo = samples[t]
        hi = samples[t + 1]  # the next sample, or the closing one (last 1), at or after it
        while lo < hi:  # last superblock s in [lo, hi] with sup[s] < k
            mid = (lo + hi + 1) >> 1
            if sup[mid] < k:
                lo = mid
            else:
                hi = mid - 1
        r = k - sup[lo]

        blk = self._block
        b = lo << 3
        last = b + _BLOCKS_PER_SUPERBLOCK - 1
        if last > self._n_blocks - 1:
            last = self._n_blocks - 1
        while b < last and blk[b + 1] < r:  # at most 8 blocks
            b += 1
        r -= blk[b]

        table = POPCOUNT8
        data = self._bytes
        o = b << 5  # first byte of the block
        for _ in range(_WORDS_PER_BLOCK):  # at most 4 words
            c = (
                table[data[o]] + table[data[o + 1]] + table[data[o + 2]] + table[data[o + 3]]
                + table[data[o + 4]] + table[data[o + 5]] + table[data[o + 6]]
                + table[data[o + 7]]
            )
            if r <= c:
                break
            r -= c
            o += _BYTES_PER_WORD
        for _ in range(_BYTES_PER_WORD):  # at most 8 bytes
            c = table[data[o]]
            if r <= c:
                break
            r -= c
            o += 1
        return (o << 3) + SELECT8[data[o]][r - 1]

    def select0(self, k: int) -> int:
        """Return the position of the k-th 0 bit (k is 1-based).

        The same search as :meth:`select1`, on its own samples of every
        512th 0 bit (plus the closing sample, the superblock of the last
        0), with zero counts derived from the stored ranks (a
        superblock s starts after ``2048 s - rank`` zeros). Only the n real
        positions count, so the zero padding of the last word is never
        returned.

        Args:
            k: A rank with ``1 <= k <= num_zeros``.

        Returns:
            The position p with bit p = 0 and ``rank0(p) == k - 1``.

        Raises:
            TypeError: If ``k`` is not an integer.
            IndexError: If ``k`` is outside ``[1, num_zeros]``.

        Time Complexity:
            O(1) except the binary search between two neighbouring samples,
            which is O(log g) for a gap of g superblocks between them.

        Space Complexity:
            O(1).

        Examples:
            >>> rs = RankSelect(BitVector.from_string("1011001110001011"))
            >>> [rs.select0(k) for k in range(1, 8)]
            [1, 4, 5, 9, 10, 11, 13]
            >>> RankSelect(BitVector.from_string("111")).select0(1)
            Traceback (most recent call last):
                ...
            IndexError: select0 rank 1 out of range [1, 0]
        """
        k = _as_int(k, "k")
        zeros = self._n - self._ones
        if k < 1 or k > zeros:
            raise IndexError(f"select0 rank {k} out of range [1, {zeros}]")
        sup = self._super
        samples = self._samples0
        t = (k - 1) >> 9
        lo = samples[t]
        hi = samples[t + 1]  # the next sample, or the closing one (last 0)
        while lo < hi:  # last superblock s in [lo, hi] with zeros-before(s) < k
            mid = (lo + hi + 1) >> 1
            if (mid << 11) - sup[mid] < k:
                lo = mid
            else:
                hi = mid - 1
        r = k - ((lo << 11) - sup[lo])

        blk = self._block
        first = lo << 3
        b = first
        last = first + _BLOCKS_PER_SUPERBLOCK - 1
        if last > self._n_blocks - 1:
            last = self._n_blocks - 1
        while b < last and ((b + 1 - first) << 8) - blk[b + 1] < r:  # at most 8 blocks
            b += 1
        r -= ((b - first) << 8) - blk[b]

        table = POPCOUNT8
        data = self._bytes
        o = b << 5
        for _ in range(_WORDS_PER_BLOCK):  # at most 4 words
            c = 64 - (
                table[data[o]] + table[data[o + 1]] + table[data[o + 2]] + table[data[o + 3]]
                + table[data[o + 4]] + table[data[o + 5]] + table[data[o + 6]]
                + table[data[o + 7]]
            )
            if r <= c:
                break
            r -= c
            o += _BYTES_PER_WORD
        for _ in range(_BYTES_PER_WORD):  # at most 8 bytes
            c = 8 - table[data[o]]
            if r <= c:
                break
            r -= c
            o += 1
        return (o << 3) + SELECT8[data[o] ^ 0xFF][r - 1]

    # --------------------------------------------------------------- space

    def overhead_bits(self) -> int:
        """Return the size of the rank directory in bits.

        ``32`` bits per superblock plus ``16`` bits per block, that is
        ``32 * ceil(n / 2048) + 16 * ceil(n / 256)``: 7.8125% of n when n is
        a multiple of 2,048. The fixed ``POPCOUNT8`` / ``SELECT8`` tables and
        the select samples are not included.

        Returns:
            The rank directory size in bits.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector(2048)).overhead_bits()
            160
            >>> RankSelect(BitVector(1 << 20)).overhead_bits() / (1 << 20)
            0.078125
            >>> RankSelect(BitVector(0)).overhead_bits()
            0
        """
        return 32 * self._n_super + 16 * self._n_blocks

    def select_sample_bits(self) -> int:
        """Return the size of the select samples in bits, both kinds.

        One ``uint32`` per 512 ones and one per 512 zeros, plus one closing
        ``uint32`` for each kind that occurs (the superblock of the last 1
        and of the last 0): ``32 * (ceil(ones / 512) + 1 + ceil(zeros / 512)
        + 1)`` when both kinds occur. Each kind costs about 32 / 512 = 6.25%
        of its own count, so about 3.1% of n each and 6.25% of n for both
        at density 0.5. Reported separately from :meth:`overhead_bits`.

        Returns:
            The select sample size in bits.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector.from_string("10" * 1024)).select_sample_bits()
            192
            >>> RankSelect(BitVector(0)).select_sample_bits()
            0
        """
        return 32 * (len(self._samples1) + len(self._samples0))

    def __repr__(self) -> str:
        """Return a one-line summary.

        Returns:
            ``<RankSelect n=..., ones=...>``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RankSelect(BitVector.from_string("0110"))
            <RankSelect n=4, ones=2>
        """
        return f"<RankSelect n={self._n}, ones={self._ones}>"
