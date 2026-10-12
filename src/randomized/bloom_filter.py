"""A Bloom filter: set membership in a few bits per item, with one-sided error.

A Bloom filter (Bloom, 1970) answers "has this item been added?" using an
array of ``m`` bits and ``k`` hash functions, and nothing else. Adding an
item sets the ``k`` bits its hashes point to; checking an item looks at the
same ``k`` bits and answers True only if all of them are set. The items
themselves are never stored, which is where the space saving comes from: a
filter sized for a 1% false-positive rate needs about 9.6 bits per item no
matter how large the items are, where a Python ``set`` keeps a pointer, a
hash and the object itself for every member.

Monte Carlo, one-sided error
----------------------------
The filter does a fixed amount of work per query (exactly ``k`` hashes and
at most ``k`` bit reads), so it is a MONTE CARLO structure: the running time
is bounded and it is the answer that may be wrong. The error is ONE-SIDED:

* ``check`` returning False is always correct. Every bit an added item
  touched was set when it was added and bits are never cleared, so an item
  that was added cannot find one of its bits at 0. There are no false
  negatives.
* ``check`` returning True may be a FALSE POSITIVE: the item was never
  added, but other items happened to set all ``k`` of its bits. With ``n``
  items in ``m`` bits and independent uniform hashes, the probability is
  about ``(1 - e^(-k n / m))^k``, see :meth:`BloomFilter.theoretical_fp_rate`.
  For a fixed ratio ``m / n`` it is smallest at ``k = (m / n) ln 2``, which
  is what :meth:`BloomFilter.from_capacity` chooses.

The randomness here is in the hash functions, not in the queries: for a
fixed seed the filter is fully deterministic, and the false-positive rate is
a statement about how the hashes scatter items that were never inserted.

Why there is no delete
----------------------
A standard Bloom filter cannot support deletion. Bits are shared: a bit set
by the item being removed may also be one of the ``k`` bits of some other
item that is still present. Clearing it would make that other item check
False, a false negative, which breaks the one guarantee the structure
gives. Telling whether a bit is shared would mean remembering which items
set it, which is exactly the information the filter throws away to be
small. (Counting Bloom filters replace each bit with a small counter to
allow deletion, at several times the memory; that is a different
structure.)

Double hashing (Kirsch and Mitzenmacher, 2006)
----------------------------------------------
Computing ``k`` independent hashes per item would cost ``k`` digests.
Kirsch and Mitzenmacher (2006) showed that two are enough: with base hashes
``h1`` and ``h2``, the ``k`` indexes ``g_i(x) = (h1(x) + i * h2(x)) mod m``
for ``i = 0, ..., k - 1`` give the same asymptotic false-positive rate as
``k`` truly independent hash functions. This module takes both base hashes
from a single digest::

    digest = blake2b(repr(item).encode("utf-8"), digest_size=16,
                     salt=seed.to_bytes(16, "little"))
    h1 = first 8 bytes,  h2 = last 8 bytes with its low bit forced to 1

so ``add`` and ``check`` each compute one digest and then derive the ``k``
indexes with integer arithmetic. ``h2`` is forced odd so that the stride is
never 0 when ``m`` is a power of two (an even stride would only visit half
the residues, and a zero stride would make all ``k`` indexes equal).

Why not ``hash()``
------------------
Python's built-in ``hash()`` of a ``str`` is salted per interpreter process
(``PYTHONHASHSEED``), so a filter built with it would set different bits in
every run and every measured false-positive count would change from run to
run even with every random generator seeded. ``hashlib.blake2b`` over
``repr(item)`` gives the same digest in every process, and its ``salt``
parameter turns the integer ``seed`` into an independent family of hash
functions: different seeds give different, but individually reproducible,
filters.

The bit array is a ``bytearray`` of ``ceil(m / 8)`` bytes manipulated with
bit operations (bit ``j`` lives in byte ``j >> 3`` at position ``j & 7``),
never a list of bools, so :meth:`BloomFilter.memory_bytes` reports the real
cost of the structure.

====================== ================= ===================================
Operation              Time              Space
====================== ================= ===================================
add                    O(k + |repr|)     O(1) extra
check                  O(k + |repr|)     O(1) extra
fill_ratio             O(m)              O(m / 8) temporary
the filter itself                        ceil(m / 8) bytes
====================== ================= ===================================

References:
    Bloom, B. H. (1970). Space/time trade-offs in hash coding with allowable
    errors. *Communications of the ACM, 13*(7), 422-426.

    Kirsch, A., & Mitzenmacher, M. (2006). Less hashing, same performance:
    Building a better Bloom filter. *ESA 2006*, LNCS 4168, 456-467.

Examples:
    >>> bloom = BloomFilter.from_capacity(1000, 0.01, seed=7)
    >>> bloom.m, bloom.k
    (9586, 7)
    >>> for word in ["apple", "pear", "plum"]:
    ...     bloom.add(word)
    >>> "pear" in bloom, len(bloom)
    (True, 3)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from typing import Any, Tuple

from src.utils.probability_analysis import (
    bloom_fp_theoretical,
    bloom_optimal_parameters,
)

__all__ = ["BloomFilter"]

#: Exclusive upper bound on a seed: blake2b accepts a salt of at most 16 bytes.
_SEED_LIMIT = 2**128


def _require_positive_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int`` of at least 1, else raise.

    Args:
        value: The candidate value.
        name: The parameter name, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``. ``bool`` is rejected,
            because ``True`` is an accident, not a size.
        ValueError: If ``value`` is less than 1.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_positive_int(8, "m_bits")
        8
        >>> _require_positive_int(0, "k_hashes")
        Traceback (most recent call last):
            ...
        ValueError: k_hashes must be at least 1, got 0
        >>> _require_positive_int(True, "m_bits")
        Traceback (most recent call last):
            ...
        TypeError: m_bits must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _require_seed(seed: Any) -> int:
    """Return ``seed`` if it can be used as a blake2b salt, else raise.

    Args:
        seed: The candidate seed.

    Returns:
        ``seed`` unchanged.

    Raises:
        TypeError: If ``seed`` is not an ``int`` (``bool`` rejected).
        ValueError: Unless ``0 <= seed < 2**128``, the range that fits in
            the 16-byte salt.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_seed(0)
        0
        >>> _require_seed(-1)
        Traceback (most recent call last):
            ...
        ValueError: seed must satisfy 0 <= seed < 2**128, got -1
        >>> _require_seed(1.5)
        Traceback (most recent call last):
            ...
        TypeError: seed must be an int, got float
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**128, got {seed}")
    return seed


class BloomFilter:
    """A Bloom filter over an ``m``-bit array with ``k`` double-hashed probes.

    MONTE CARLO with ONE-SIDED error: :meth:`check` returning False is always
    correct, while True may be a false positive. Items are hashed with one
    salted blake2b digest each, split into the two base hashes of
    Kirsch-Mitzenmacher double hashing (see the module docstring). Items of
    any type can be added as long as ``repr(item)`` identifies them; two
    items with the same ``repr`` are the same item to the filter.

    Args:
        m_bits: Number of bits in the array. At least 1.
        k_hashes: Number of probes per item. At least 1.
        seed: Selects the hash family through the blake2b salt. Same seed,
            same bits; different seeds, independent filters. Must satisfy
            ``0 <= seed < 2**128``.

    Raises:
        TypeError: If any argument is not an ``int`` (``bool`` rejected).
        ValueError: If ``m_bits`` or ``k_hashes`` is less than 1, or
            ``seed`` is outside ``[0, 2**128)``.

    Time Complexity:
        O(m / 8) to allocate the zeroed byte array.

    Space Complexity:
        ``ceil(m / 8)`` bytes for the bit array, independent of how many
        items are added or how large they are.

    Examples:
        >>> bloom = BloomFilter(1024, 5, seed=3)
        >>> bloom.add("alice")
        >>> bloom.check("alice"), "alice" in bloom
        (True, True)
        >>> BloomFilter(1024, 5).check("anything")
        False
        >>> bloom.memory_bytes() == math.ceil(bloom.m / 8)
        True
        >>> BloomFilter(0, 3)
        Traceback (most recent call last):
            ...
        ValueError: m_bits must be at least 1, got 0
    """

    __slots__ = ("_m", "_k", "_seed", "_salt", "_bits", "_count")

    def __init__(self, m_bits: int, k_hashes: int, seed: int = 0) -> None:
        self._m = _require_positive_int(m_bits, "m_bits")
        self._k = _require_positive_int(k_hashes, "k_hashes")
        self._seed = _require_seed(seed)
        self._salt = seed.to_bytes(16, "little")
        self._bits = bytearray((m_bits + 7) // 8)
        self._count = 0

    @classmethod
    def from_capacity(
        cls, expected_items: int, fp_rate: float, seed: int = 0
    ) -> BloomFilter:
        """Build a filter sized for ``expected_items`` at a target FP rate.

        Uses the textbook optimum from
        :func:`~src.utils.probability_analysis.bloom_optimal_parameters`:
        ``m = ceil(-n ln p / (ln 2)^2)`` bits and ``k = round(m / n * ln 2)``
        probes. At that ``k`` each bit is set with probability about 1/2 once
        ``n`` items are in, which is what minimises ``(1 - e^(-k n / m))^k``
        for a given ``m / n``. Adding more than ``expected_items`` items is
        allowed; the false-positive rate simply rises above ``fp_rate``.

        Args:
            expected_items: The number of items the filter is sized for. At
                least 1.
            fp_rate: The target false-positive rate, strictly between 0 and 1.
            seed: Passed through to the constructor.

        Returns:
            A new, empty :class:`BloomFilter`.

        Raises:
            TypeError: If ``expected_items`` is not an ``int`` or
                ``fp_rate`` is not a real number (``bool`` rejected for both).
            ValueError: If ``expected_items < 1``, ``fp_rate`` is not in
                ``(0, 1)``, or ``seed`` is out of range.

        Time Complexity:
            O(m / 8) to allocate the array.

        Space Complexity:
            ``ceil(m / 8)`` bytes, about ``1.44 n log2(1 / p) / 8``.

        Examples:
            >>> bloom = BloomFilter.from_capacity(1000, 0.01)
            >>> bloom.m, bloom.k
            (9586, 7)
            >>> bloom.memory_bytes() == math.ceil(9586 / 8)
            True
            >>> BloomFilter.from_capacity(1000, 1.0)
            Traceback (most recent call last):
                ...
            ValueError: fp_rate must satisfy 0 < fp_rate < 1, got 1.0
        """
        _require_positive_int(expected_items, "expected_items")
        if isinstance(fp_rate, bool) or not isinstance(fp_rate, numbers.Real):
            raise TypeError(
                f"fp_rate must be a real number, got {type(fp_rate).__name__}"
            )
        if not 0 < fp_rate < 1:
            raise ValueError(f"fp_rate must satisfy 0 < fp_rate < 1, got {fp_rate}")
        m_bits, k_hashes = bloom_optimal_parameters(expected_items, fp_rate)
        return cls(int(m_bits), int(k_hashes), seed=seed)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------
    def _hash_pair(self, item: Any) -> Tuple[int, int]:
        """Return the double-hashing base pair ``(h1 mod m, h2 mod m)``.

        One salted 16-byte blake2b digest of ``repr(item)``: ``h1`` is its
        first 8 bytes and ``h2`` its last 8 bytes with the low bit forced to
        1, both read little-endian. Both are reduced mod ``m`` here so that
        the probe loop works with small integers; since
        ``(h1 + i h2) mod m == (h1 mod m + i (h2 mod m)) mod m``, the probe
        sequence is exactly ``(h1 + i * h2) % m``.

        Args:
            item: Any object; only ``repr(item)`` is hashed.

        Returns:
            ``(start, step)``, the first index and the stride, both in
            ``[0, m)``.

        Raises:
            None.

        Time Complexity:
            O(|repr(item)|) for the digest.

        Space Complexity:
            O(|repr(item)|) for the encoded bytes.

        Examples:
            >>> bloom = BloomFilter(1000, 3, seed=1)
            >>> bloom._hash_pair("x") == bloom._hash_pair("x")
            True
            >>> all(0 <= part < 1000 for part in bloom._hash_pair("x"))
            True
        """
        digest = hashlib.blake2b(
            repr(item).encode("utf-8"), digest_size=16, salt=self._salt
        ).digest()
        m = self._m
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1
        return h1 % m, h2 % m

    def _indexes(self, item: Any) -> list[int]:
        """Return the ``k`` bit indexes ``(h1 + i * h2) % m`` for ``item``.

        Not used by :meth:`add` or :meth:`check`, which inline the same
        loop for speed; kept so the probe sequence can be inspected.

        Args:
            item: Any object; only ``repr(item)`` is hashed.

        Returns:
            A list of ``k`` indexes in ``[0, m)``. Repeats are possible.

        Raises:
            None.

        Time Complexity:
            O(k + |repr(item)|).

        Space Complexity:
            O(k).

        Examples:
            >>> bloom = BloomFilter(97, 4, seed=2)
            >>> idx = bloom._indexes(("a", 1))
            >>> len(idx), all(0 <= i < 97 for i in idx)
            (4, True)
            >>> step = (idx[1] - idx[0]) % 97
            >>> all((idx[0] + i * step) % 97 == idx[i] for i in range(4))
            True
        """
        start, step = self._hash_pair(item)
        m = self._m
        return [(start + i * step) % m for i in range(self._k)]

    # ------------------------------------------------------------------
    # The two operations the assignment asks for
    # ------------------------------------------------------------------
    def add(self, item: Any) -> None:
        """Insert ``item`` by setting its ``k`` bits.

        Adding the same item twice sets the same bits twice, so the bit
        array does not change, but :func:`len` counts both calls: the filter
        cannot tell a repeat from a new item without storing items.

        Args:
            item: Any object whose ``repr`` identifies it.

        Returns:
            None.

        Raises:
            None.

        Time Complexity:
            O(k + |repr(item)|): one digest, then ``k`` bit writes.

        Space Complexity:
            O(1) extra beyond the digest.

        Examples:
            >>> bloom = BloomFilter(256, 3)
            >>> bloom.add(42)
            >>> bloom.check(42), len(bloom)
            (True, 1)
            >>> bloom.add(42)
            >>> len(bloom)
            2
        """
        index, step = self._hash_pair(item)
        m = self._m
        bits = self._bits
        for _ in range(self._k):
            bits[index >> 3] |= 1 << (index & 7)
            index += step
            if index >= m:
                index -= m
        self._count += 1

    def check(self, item: Any) -> bool:
        """Return True if ``item`` may have been added, False if it was not.

        MONTE CARLO, ONE-SIDED: False is always correct (no false
        negatives); True is wrong with probability about
        :meth:`theoretical_fp_rate` for an item that was never added. The
        loop stops at the first 0 bit, so most non-members cost fewer than
        ``k`` reads.

        Args:
            item: Any object whose ``repr`` identifies it.

        Returns:
            True if all ``k`` bits of ``item`` are set, otherwise False.

        Raises:
            None.

        Time Complexity:
            O(k + |repr(item)|): one digest, then at most ``k`` bit reads.

        Space Complexity:
            O(1) extra beyond the digest.

        Examples:
            >>> bloom = BloomFilter(512, 4, seed=5)
            >>> bloom.check("missing")
            False
            >>> bloom.add("present")
            >>> bloom.check("present")
            True
        """
        index, step = self._hash_pair(item)
        m = self._m
        bits = self._bits
        for _ in range(self._k):
            if not bits[index >> 3] & (1 << (index & 7)):
                return False
            index += step
            if index >= m:
                index -= m
        return True

    def __contains__(self, item: Any) -> bool:
        """Alias for :meth:`check`, so ``item in bloom`` works.

        Args:
            item: Any object whose ``repr`` identifies it.

        Returns:
            ``self.check(item)``.

        Raises:
            None.

        Time Complexity:
            O(k + |repr(item)|).

        Space Complexity:
            O(1) extra.

        Examples:
            >>> bloom = BloomFilter(128, 2)
            >>> bloom.add("x")
            >>> "x" in bloom, "y" in BloomFilter(128, 2)
            (True, False)
        """
        return self.check(item)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of :meth:`add` calls, repeats included.

        Args:
            None.

        Returns:
            The count ``n`` used by :meth:`load_factor` and the FP formula.

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(BloomFilter(64, 2))
            0
        """
        return self._count

    @property
    def m(self) -> int:
        """Number of bits in the array.

        Examples:
            >>> BloomFilter(100, 3).m
            100
        """
        return self._m

    @property
    def k(self) -> int:
        """Number of probes per item.

        Examples:
            >>> BloomFilter(100, 3).k
            3
        """
        return self._k

    @property
    def seed(self) -> int:
        """The seed that selects the hash family (the blake2b salt).

        Examples:
            >>> BloomFilter(100, 3, seed=9).seed
            9
        """
        return self._seed

    @property
    def bits(self) -> bytes:
        """An immutable copy of the bit array, for comparing filters.

        Bit ``j`` is ``(bits[j >> 3] >> (j & 7)) & 1``. The trailing bits of
        the last byte beyond ``m`` are always 0.

        Examples:
            Same seed, same bits; a different seed scatters the same items
            differently:

            >>> words = ["ant", "bee", "cat", "dog", "eel"]
            >>> first, second = BloomFilter(256, 4, 11), BloomFilter(256, 4, 11)
            >>> other = BloomFilter(256, 4, 12)
            >>> for word in words:
            ...     first.add(word)
            ...     second.add(word)
            ...     other.add(word)
            >>> first.bits == second.bits, first.bits == other.bits
            (True, False)
            >>> len(first.bits)
            32
        """
        return bytes(self._bits)

    @property
    def fill_ratio(self) -> float:
        """Fraction of the ``m`` bits that are set, in ``[0, 1]``.

        With ``n`` items the expected value is ``1 - e^(-k n / m)``, and the
        false-positive rate is close to ``fill_ratio ** k``, so this is the
        quantity to watch as a filter fills. Counted with one population
        count over the whole array.

        Examples:
            >>> bloom = BloomFilter(64, 1)
            >>> bloom.fill_ratio
            0.0
            >>> bloom.add("only")
            >>> bloom.fill_ratio
            0.015625
        """
        return int.from_bytes(self._bits, "little").bit_count() / self._m

    def load_factor(self) -> float:
        """Return items per bit, ``n / m``.

        Args:
            None.

        Returns:
            ``len(self) / self.m``. At the optimum for a 1% target this is
            about 0.104 (9.6 bits per item).

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> bloom = BloomFilter(100, 2)
            >>> for number in range(25):
            ...     bloom.add(number)
            >>> bloom.load_factor()
            0.25
        """
        return self._count / self._m

    def bits_per_item(self) -> float:
        """Return bits per item, ``m / n``, or infinity for an empty filter.

        The reciprocal of :meth:`load_factor`, and the axis of the
        memory-versus-accuracy trade-off: each extra bit per item (at the
        optimal ``k``) multiplies the false-positive rate by about
        ``0.6185``.

        Args:
            None.

        Returns:
            ``self.m / len(self)``, or ``math.inf`` when nothing has been
            added.

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> bloom = BloomFilter(100, 2)
            >>> bloom.bits_per_item()
            inf
            >>> for number in range(25):
            ...     bloom.add(number)
            >>> bloom.bits_per_item()
            4.0
        """
        if self._count == 0:
            return math.inf
        return self._m / self._count

    def theoretical_fp_rate(self) -> float:
        """Return the textbook false-positive estimate ``(1 - e^(-k n / m))^k``.

        This is Bloom's approximation, assuming independent uniform hashes;
        with double hashing the true rate converges to it as ``m`` grows
        (Kirsch and Mitzenmacher, 2006). It is the curve the measured rates
        are plotted against. It is 0 for an empty filter.

        Args:
            None.

        Returns:
            The predicted probability that :meth:`check` returns True for an
            item that was never added.

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> bloom = BloomFilter.from_capacity(1000, 0.01)
            >>> bloom.theoretical_fp_rate()
            0.0
            >>> for number in range(1000):
            ...     bloom.add(number)
            >>> round(bloom.theoretical_fp_rate(), 5)
            0.01003
        """
        if self._count == 0:
            return 0.0  # exact, and avoids a -0.0 from the expm1 form
        return float(bloom_fp_theoretical(self._m, self._count, self._k))

    def memory_bytes(self) -> int:
        """Return the size of the bit array in bytes, ``ceil(m / 8)``.

        Only the bit array is counted, not the Python object overhead,
        because that is the part that scales with the filter and the part a
        C implementation would also pay.

        Args:
            None.

        Returns:
            ``ceil(m / 8)``.

        Raises:
            None.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> BloomFilter(9586, 7).memory_bytes()
            1199
            >>> BloomFilter(8, 1).memory_bytes(), BloomFilter(9, 1).memory_bytes()
            (1, 2)
        """
        return len(self._bits)
