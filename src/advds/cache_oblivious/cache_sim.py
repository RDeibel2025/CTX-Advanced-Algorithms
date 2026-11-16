"""A fully associative LRU cache simulator for counting block transfers.

Why simulate
------------
The cache-oblivious algorithms in :mod:`src.advds.cache_oblivious.matrix_ops`
and the tree layouts in :mod:`src.advds.cache_oblivious.layouts` are about
how many blocks move between a small fast memory and a large slow one. In
Python every array access goes through the interpreter, and that overhead
hides the real hardware cache almost completely, so wall-clock time cannot
show the effect. Instead the algorithms replay their memory accesses as
integer addresses into this simulator, which counts the misses exactly. This
is the ideal-cache model of Frigo, Leiserson, Prokop and Ramachandran (1999)
with LRU replacement.

The model
---------
* Memory is a flat array of words with integer addresses ``0, 1, 2, ...``.
* The cache holds ``M`` words split into blocks (cache lines) of ``B``
  words. Address ``a`` lives in block ``a // B``, so the cache holds
  ``M // B`` blocks.
* The cache is fully associative: any block may sit in any slot.
* An access to a resident block is a hit and makes that block the most
  recently used. An access to any other block is a miss: the block is
  loaded and, when the cache is full, the least recently used block is
  evicted first.
* Reads and writes are not distinguished. A write to a block that is not
  resident loads it (write-allocate), which is what the traces in
  ``matrix_ops`` assume.

Implementation
--------------
The recency order is a hand-written circular doubly linked list over a
fixed array of ``M // B`` slots (``_prev`` and ``_next`` hold slot indices,
with one sentinel slot at index ``M // B``), and a dict maps each resident
block number to its slot. A hit unlinks the slot and relinks it at the
front; a miss reuses the slot at the back. Both are O(1). No library LRU
container stands in for the recency list. A hit on the block that is
already the most recent one (the common case for sequential access) skips
the relinking entirely.

Examples:
    >>> cache = LRUCache(M=4, B=2)          # 2 blocks of 2 words
    >>> [cache.access(a) for a in (0, 1, 2, 3, 4, 0)]
    [False, True, False, True, False, False]
    >>> cache.misses, cache.accesses
    (4, 6)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numbers
from typing import Dict, List

__all__ = ["LRUCache"]


def _require_positive_int(value, name: str) -> int:
    """Return ``value`` as an ``int`` if it is a positive integer, else raise.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an integer (``bool`` rejected).
        ValueError: If ``value`` is not positive.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_positive_int(16, "B")
        16
        >>> _require_positive_int(0, "B")
        Traceback (most recent call last):
        ...
        ValueError: B must be positive, got 0
        >>> _require_positive_int(True, "B")
        Traceback (most recent call last):
        ...
        TypeError: B must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


class LRUCache:
    """A fully associative cache of ``M`` words in blocks of ``B`` words, LRU replacement.

    Attributes:
        M: The cache size in words.
        B: The block (cache line) size in words.
        capacity: The number of blocks the cache holds, ``M // B``.

    Examples:
        >>> cache = LRUCache(1024, 16)
        >>> cache.capacity
        64
        >>> cache.access(5), cache.access(15), cache.access(16)
        (False, True, False)
        >>> cache.hits
        1
    """

    __slots__ = (
        "M",
        "B",
        "capacity",
        "_sentinel",
        "_slot_of",
        "_block",
        "_prev",
        "_next",
        "_size",
        "_misses",
        "_accesses",
    )

    def __init__(self, M: int, B: int) -> None:
        """Create an empty cache of ``M`` words split into blocks of ``B`` words.

        Args:
            M: The cache size in words, a positive int.
            B: The block size in words, a positive int with ``B <= M``.

        Returns:
            None.

        Raises:
            TypeError: If ``M`` or ``B`` is not an int (``bool`` rejected).
            ValueError: If ``M`` or ``B`` is not positive, or ``M < B`` (the
                cache would hold no block at all).

        Time Complexity:
            O(M / B) to allocate the slot arrays.

        Space Complexity:
            O(M / B).

        Examples:
            >>> LRUCache(256, 8).capacity
            32
            >>> LRUCache(4, 8)
            Traceback (most recent call last):
            ...
            ValueError: M must be at least B so the cache holds a block, got M=4, B=8
        """
        self.M = _require_positive_int(M, "M")
        self.B = _require_positive_int(B, "B")
        if self.M < self.B:
            raise ValueError(
                f"M must be at least B so the cache holds a block, got M={self.M}, B={self.B}"
            )
        self.capacity = self.M // self.B
        self._sentinel = self.capacity
        self._slot_of: Dict[int, int] = {}
        self._block: List[int] = []
        self._prev: List[int] = []
        self._next: List[int] = []
        self._size = 0
        self._misses = 0
        self._accesses = 0
        self.reset()

    def reset(self) -> None:
        """Empty the cache and set the miss and access counters back to zero.

        Returns:
            None.

        Raises:
            Nothing.

        Time Complexity:
            O(M / B).

        Space Complexity:
            O(M / B) - the slot arrays are rebuilt.

        Examples:
            >>> cache = LRUCache(4, 2)
            >>> cache.access(0)
            False
            >>> cache.reset()
            >>> cache.misses, cache.accesses, cache.access(0)
            (0, 0, False)
        """
        slots = self.capacity + 1  # real slots 0 .. capacity-1, then the sentinel
        self._slot_of = {}
        self._block = [-1] * slots
        self._prev = [self._sentinel] * slots
        self._next = [self._sentinel] * slots
        self._size = 0
        self._misses = 0
        self._accesses = 0

    @property
    def misses(self) -> int:
        """The number of accesses so far that missed (loaded a block).

        Returns:
            The miss count since construction or the last :meth:`reset`.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> cache = LRUCache(4, 2)
            >>> _ = [cache.access(a) for a in (0, 2, 4, 0)]
            >>> cache.misses
            4
        """
        return self._misses

    @property
    def accesses(self) -> int:
        """The number of accesses so far, hits and misses together.

        Returns:
            The access count since construction or the last :meth:`reset`.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> cache = LRUCache(4, 2)
            >>> _ = [cache.access(a) for a in range(10)]
            >>> cache.accesses
            10
        """
        return self._accesses

    @property
    def hits(self) -> int:
        """The number of accesses so far that hit, ``accesses - misses``.

        Returns:
            The hit count since construction or the last :meth:`reset`.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> cache = LRUCache(4, 2)
            >>> _ = [cache.access(a) for a in range(10)]
            >>> cache.hits, cache.misses
            (5, 5)
        """
        return self._accesses - self._misses

    def access(self, address: int) -> bool:
        """Touch one word: report a hit, or load its block (evicting the LRU block if full).

        Args:
            address: The word address, a non-negative int. It lives in block
                ``address // B``.

        Returns:
            ``True`` if the block was resident (a hit), ``False`` on a miss.

        Raises:
            ValueError: If ``address`` is negative.

        Time Complexity:
            O(1) expected: one dict lookup and a constant number of list
            updates to relink the slot.

        Space Complexity:
            O(1); the cache never holds more than ``M // B`` blocks.

        Examples:
            >>> cache = LRUCache(M=4, B=2)        # holds 2 blocks
            >>> cache.access(0), cache.access(2)  # blocks 0 and 1 load
            (False, False)
            >>> cache.access(1)                   # block 0 is resident, now most recent
            True
            >>> cache.access(4)                   # block 2 evicts block 1, the LRU one
            False
            >>> cache.access(0), cache.access(3)
            (True, False)
            >>> cache.access(-1)
            Traceback (most recent call last):
            ...
            ValueError: address must be non-negative, got -1
        """
        if address < 0:
            raise ValueError(f"address must be non-negative, got {address}")
        self._accesses += 1
        block = address // self.B
        nxt = self._next
        prv = self._prev
        s = self._sentinel
        slot = self._slot_of.get(block)
        if slot is not None:
            if nxt[s] != slot:
                # Unlink the slot ...
                p = prv[slot]
                q = nxt[slot]
                nxt[p] = q
                prv[q] = p
                # ... and relink it at the front (most recently used).
                head = nxt[s]
                nxt[slot] = head
                prv[slot] = s
                prv[head] = slot
                nxt[s] = slot
            return True
        self._misses += 1
        if self._size < self.capacity:
            slot = self._size
            self._size += 1
        else:
            # Evict the least recently used block, the one at the back.
            slot = prv[s]
            del self._slot_of[self._block[slot]]
            p = prv[slot]
            nxt[p] = s
            prv[s] = p
        self._block[slot] = block
        self._slot_of[block] = slot
        head = nxt[s]
        nxt[slot] = head
        prv[slot] = s
        prv[head] = slot
        nxt[s] = slot
        return False

    def __repr__(self) -> str:
        """Return a short description with the geometry and the counters.

        Returns:
            For example ``LRUCache(M=1024, B=16, misses=0, accesses=0)``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> LRUCache(1024, 16)
            LRUCache(M=1024, B=16, misses=0, accesses=0)
        """
        return (
            f"LRUCache(M={self.M}, B={self.B}, misses={self._misses}, "
            f"accesses={self._accesses})"
        )
