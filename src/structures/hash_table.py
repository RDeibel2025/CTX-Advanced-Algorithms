"""Hash table with two selectable collision strategies.

A hash table turns a key into an array index with ``hash(key)``, so a lookup
can jump straight to where the key must be instead of searching for it.
That is the whole source of its O(1) average cost - and also of its O(n)
worst case, because two keys can hash to the same index. What a table does
about those *collisions* is the difference between its two classic forms,
and both are implemented here, selectable per instance:

* **Separate chaining** (``strategy="chaining"``) - each slot holds a small
  list of every entry that hashed there. A collision just makes one list
  longer. Degrades gently as the table fills.
* **Open addressing with linear probing** (``strategy="linear_probing"``) -
  every entry lives directly in the array. A collision moves on to the next
  slot, then the next. No per-entry allocation and good cache behaviour,
  but collisions *cluster*: runs of occupied slots merge and grow, and
  lookups slow down sharply as the table fills.

That difference is why the two strategies get different resize thresholds.
Knuth's analysis of linear probing puts the expected cost of an
unsuccessful search at about ``(1 + 1 / (1 - a)^2) / 2`` probes at load
factor ``a``: 2.5 probes at a = 0.5, but 8.5 at a = 0.75. Chaining's
equivalent cost is about ``a``, under one entry examined even at a = 0.75.
So chaining resizes at 0.75 and probing at 0.5, roughly equalising the cost
of the slow path. The Week 3 benchmark measures these curves.

========================= ============= ================================
Operation                 Average       Worst case
========================= ============= ================================
insert                    O(1) amortised O(n) - every key in one slot
get / ``in``              O(1)          O(n)
delete                    O(1)          O(n)
========================= ============= ================================

"Amortised" matters for insert. When the load factor passes the threshold
the table doubles its capacity and reinserts every live entry - an O(n)
step. Because the capacity doubles each time, those rebuilds move fewer
than 2n entries in total across n inserts, so the average cost per insert
stays constant even though individual inserts occasionally cost O(n).
:attr:`HashTable.rehash_moves` counts the entries moved, so this can be
checked rather than taken on trust.

**Deletion under linear probing uses tombstones.** Emptying a deleted slot
would cut every probe sequence that passed through it, and keys stored
further along the run would become unfindable - lookups silently return a
miss for keys that are present. A tombstone marks the slot as "deleted,
keep probing": lookups step over it, inserts may reuse it, and the next
rehash drops it.

**The index is ``hash(key) & (capacity - 1)``**, with capacity always a
power of two. There is deliberately no extra mixing of the hash. Python
hashes small integers to themselves, so the index is just a key's low bits:
ordinary keys spread evenly, but keys that are all multiples of the capacity
land in slot 0 together. That makes the worst case easy to construct and
measure. Production tables defend against it - CPython's ``dict`` perturbs
the hash - and the benchmark shows the difference.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

__all__ = ["HashTable", "CHAINING", "LINEAR_PROBING", "DEFAULT_MAX_LOAD_FACTOR"]

CHAINING = "chaining"
LINEAR_PROBING = "linear_probing"

#: Resize thresholds per strategy. See the module docstring for why they
#: differ: linear probing's cost rises much faster with load than chaining's.
DEFAULT_MAX_LOAD_FACTOR: Dict[str, float] = {
    CHAINING: 0.75,
    LINEAR_PROBING: 0.5,
}

#: Smallest capacity a table will allocate. Capacities are powers of two.
MIN_CAPACITY = 8


class _Sentinel:
    """A named marker object, compared by identity only."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


#: Marks a never-used slot. A lookup that reaches one stops: the key is absent.
_EMPTY = _Sentinel("<empty>")
#: Marks a deleted slot. A lookup steps over it; an insert may reuse it.
_TOMBSTONE = _Sentinel("<tombstone>")
#: Default for ``__getitem__`` and ``__contains__``, distinct from any value.
_MISSING = _Sentinel("<missing>")


def _capacity_for(requested: int) -> int:
    """Smallest power of two that is at least ``requested`` and MIN_CAPACITY."""
    capacity = MIN_CAPACITY
    while capacity < requested:
        capacity <<= 1
    return capacity


class HashTable:
    """Key-value hash table using separate chaining or linear probing.

    Args:
        capacity: Initial number of slots. Rounded up to a power of two, and
            to at least :data:`MIN_CAPACITY`.
        strategy: ``"chaining"`` (the default) or ``"linear_probing"``.
        max_load_factor: Load factor above which the table resizes. Defaults
            to 0.75 for chaining and 0.5 for linear probing. Must be positive,
            and below 1 for linear probing - an open-addressed table must
            always keep an empty slot, or an unsuccessful lookup never ends.

    Raises:
        ValueError: For an unknown strategy, a non-positive capacity, or an
            out-of-range load factor.

    Attributes:
        rehash_count: Number of times the table has been rebuilt.
        rehash_moves: Total live entries reinserted across all rebuilds - the
            quantity amortised analysis bounds by 2n.

    Examples:
        >>> table = HashTable()
        >>> table.insert("apple", 3)
        >>> table.insert("pear", 5)
        >>> table.get("apple"), table.get("plum"), table.get("plum", 0)
        (3, None, 0)
        >>> table.delete("apple")
        >>> "apple" in table, len(table)
        (False, 1)

        Both strategies behave identically from the outside:

        >>> probing = HashTable(strategy="linear_probing")
        >>> for number in range(100):
        ...     probing.insert(number, number * number)
        >>> probing.get(12), len(probing), probing.capacity
        (144, 100, 256)
    """

    __slots__ = (
        "_strategy",
        "_chaining",
        "_max_load",
        "_capacity",
        "_mask",
        "_size",
        "_tombstones",
        "_buckets",
        "_keys",
        "_values",
        "rehash_count",
        "rehash_moves",
    )

    def __init__(
        self,
        capacity: int = MIN_CAPACITY,
        strategy: str = CHAINING,
        max_load_factor: Optional[float] = None,
    ) -> None:
        if strategy not in DEFAULT_MAX_LOAD_FACTOR:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected {CHAINING!r} or "
                f"{LINEAR_PROBING!r}"
            )
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if max_load_factor is None:
            max_load_factor = DEFAULT_MAX_LOAD_FACTOR[strategy]
        if not max_load_factor > 0:
            raise ValueError(f"max_load_factor must be positive, got {max_load_factor}")
        if strategy == LINEAR_PROBING and not max_load_factor < 1:
            raise ValueError(
                "max_load_factor must be below 1 for linear probing, got "
                f"{max_load_factor}"
            )

        self._strategy = strategy
        self._chaining = strategy == CHAINING
        self._max_load = float(max_load_factor)
        self._size = 0
        self._tombstones = 0
        self.rehash_count = 0
        self.rehash_moves = 0
        self._allocate(_capacity_for(capacity))

    def _allocate(self, capacity: int) -> None:
        """Replace the storage with an empty array of ``capacity`` slots."""
        self._capacity = capacity
        self._mask = capacity - 1
        if self._chaining:
            # Buckets are created lazily: an empty slot costs one pointer
            # rather than one empty list, which matters at a million slots.
            self._buckets: Optional[List[Optional[List[Tuple[Any, Any]]]]] = (
                [None] * capacity
            )
            self._keys: Optional[List[Any]] = None
            self._values: Optional[List[Any]] = None
        else:
            self._buckets = None
            self._keys = [_EMPTY] * capacity
            self._values = [None] * capacity

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of live entries. O(1)."""
        return self._size

    def __repr__(self) -> str:
        return (
            f"HashTable(strategy={self._strategy!r}, size={self._size}, "
            f"capacity={self._capacity}, load_factor={self.load_factor:.3f})"
        )

    @property
    def strategy(self) -> str:
        """``"chaining"`` or ``"linear_probing"``."""
        return self._strategy

    @property
    def capacity(self) -> int:
        """Current number of slots. Always a power of two."""
        return self._capacity

    @property
    def max_load_factor(self) -> float:
        """Load factor above which the table resizes."""
        return self._max_load

    @property
    def load_factor(self) -> float:
        """Live entries divided by capacity.

        Tombstones are not counted here, because they hold no entry. They do
        count towards the resize trigger under linear probing, since they
        still lengthen probe sequences - see :attr:`tombstones`.

        Examples:
            >>> table = HashTable(capacity=16)
            >>> for key in range(4):
            ...     table.insert(key)
            >>> table.load_factor
            0.25
        """
        return self._size / self._capacity

    @property
    def tombstones(self) -> int:
        """Deleted slots awaiting cleanup. Always 0 under chaining."""
        return self._tombstones

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert(self, key: Any, value: Any = None) -> None:
        """Store ``value`` under ``key``, replacing any existing value.

        Args:
            key: Any hashable key.
            value: Payload to store. Defaults to None, for set-like use.

        Raises:
            TypeError: If ``key`` is not hashable.

        Time Complexity:
            O(1) amortised, O(n) worst case. Occasionally this insert
            triggers a rebuild that reinserts every entry, but the doubling
            schedule keeps the total rebuild work below 2n over n inserts.

        Examples:
            >>> table = HashTable()
            >>> table.insert("k", 1)
            >>> table.insert("k", 2)        # existing key: value replaced
            >>> table.get("k"), len(table)
            (2, 1)
        """
        if self._chaining:
            self._insert_chaining(key, value)
        else:
            self._insert_probing(key, value)

    def _insert_chaining(self, key: Any, value: Any) -> None:
        buckets = self._buckets
        assert buckets is not None
        index = hash(key) & self._mask
        bucket = buckets[index]
        if bucket is None:
            buckets[index] = [(key, value)]
        else:
            for position, (existing, _old) in enumerate(bucket):
                if existing is key or existing == key:
                    bucket[position] = (key, value)
                    return
            bucket.append((key, value))
        self._size += 1
        if self._size > self._max_load * self._capacity:
            self._rehash(self._capacity * 2)

    def _insert_probing(self, key: Any, value: Any) -> None:
        keys, mask = self._keys, self._mask
        assert keys is not None and self._values is not None
        index = hash(key) & mask
        reusable = -1
        # Walk the whole run, not just to the first tombstone: the key may
        # already be stored further along, and inserting it again at the
        # tombstone would create a duplicate.
        while True:
            existing = keys[index]
            if existing is _EMPTY:
                break
            if existing is _TOMBSTONE:
                if reusable < 0:
                    reusable = index
            elif existing is key or existing == key:
                self._values[index] = value
                return
            index = (index + 1) & mask
        if reusable >= 0:
            index = reusable
            self._tombstones -= 1
        keys[index] = key
        self._values[index] = value
        self._size += 1
        if self._size + self._tombstones > self._max_load * self._capacity:
            # Grow if live entries are what filled the table; if tombstones
            # did, rebuild at the same size, which simply clears them.
            if self._size > self._max_load * self._capacity / 2:
                self._rehash(self._capacity * 2)
            else:
                self._rehash(self._capacity)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value stored under ``key``, or ``default`` if absent.

        Mirrors ``dict.get``: a miss is not an error. Use :meth:`__getitem__`
        (``table[key]``) for a lookup that raises ``KeyError`` instead.

        Args:
            key: Key to look up.
            default: Returned when ``key`` is not present.

        Returns:
            The stored value, or ``default``.

        Time Complexity:
            O(1) average, O(n) worst case.

        Examples:
            >>> table = HashTable(strategy="linear_probing")
            >>> table.insert(7, "seven")
            >>> table.get(7), table.get(8), table.get(8, "none")
            ('seven', None, 'none')
        """
        if self._chaining:
            buckets = self._buckets
            assert buckets is not None
            bucket = buckets[hash(key) & self._mask]
            if bucket is not None:
                for existing, value in bucket:
                    if existing is key or existing == key:
                        return value
            return default

        keys, mask = self._keys, self._mask
        assert keys is not None and self._values is not None
        index = hash(key) & mask
        while True:
            existing = keys[index]
            if existing is _EMPTY:
                return default
            if existing is not _TOMBSTONE and (existing is key or existing == key):
                return self._values[index]
            index = (index + 1) & mask

    def __getitem__(self, key: Any) -> Any:
        """``table[key]``: the stored value, or ``KeyError`` if absent."""
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        """``key in table``. O(1) average."""
        return self.get(key, _MISSING) is not _MISSING

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete(self, key: Any) -> None:
        """Remove ``key`` and its value.

        Under linear probing the slot becomes a tombstone rather than empty,
        so that keys stored further along the same probe run stay findable.

        Args:
            key: Key to remove.

        Raises:
            KeyError: If ``key`` is not present. The table is unchanged.

        Time Complexity:
            O(1) average, O(n) worst case.

        Examples:
            >>> table = HashTable(strategy="linear_probing")
            >>> table.insert("a", 1)
            >>> table.delete("a")
            >>> table.get("a"), table.tombstones
            (None, 1)
            >>> table.delete("a")
            Traceback (most recent call last):
                ...
            KeyError: 'a'
        """
        if self._chaining:
            buckets = self._buckets
            assert buckets is not None
            index = hash(key) & self._mask
            bucket = buckets[index]
            if bucket is not None:
                for position, (existing, _value) in enumerate(bucket):
                    if existing is key or existing == key:
                        del bucket[position]
                        if not bucket:
                            buckets[index] = None
                        self._size -= 1
                        return
            raise KeyError(key)

        keys, mask = self._keys, self._mask
        assert keys is not None and self._values is not None
        index = hash(key) & mask
        while True:
            existing = keys[index]
            if existing is _EMPTY:
                raise KeyError(key)
            if existing is not _TOMBSTONE and (existing is key or existing == key):
                keys[index] = _TOMBSTONE
                self._values[index] = None
                self._size -= 1
                self._tombstones += 1
                return
            index = (index + 1) & mask

    def __setitem__(self, key: Any, value: Any) -> None:
        """``table[key] = value``. Same as :meth:`insert`."""
        self.insert(key, value)

    def __delitem__(self, key: Any) -> None:
        """``del table[key]``. Same as :meth:`delete`."""
        self.delete(key)

    # ------------------------------------------------------------------
    # Rehashing
    # ------------------------------------------------------------------
    def _rehash(self, new_capacity: int) -> None:
        """Rebuild into ``new_capacity`` slots, keeping every live entry.

        Tombstones are dropped, since the new array has none. Keys are known
        to be unique, so each is placed directly without a duplicate check.
        """
        entries = list(self.items())
        self._allocate(new_capacity)
        mask = self._mask
        if self._chaining:
            buckets = self._buckets
            assert buckets is not None
            for key, value in entries:
                index = hash(key) & mask
                bucket = buckets[index]
                if bucket is None:
                    buckets[index] = [(key, value)]
                else:
                    bucket.append((key, value))
        else:
            keys, values = self._keys, self._values
            assert keys is not None and values is not None
            for key, value in entries:
                index = hash(key) & mask
                while keys[index] is not _EMPTY:
                    index = (index + 1) & mask
                keys[index] = key
                values[index] = value
        self._size = len(entries)
        self._tombstones = 0
        self.rehash_count += 1
        self.rehash_moves += len(entries)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Yield every live ``(key, value)`` pair, in slot order. O(capacity)."""
        if self._chaining:
            assert self._buckets is not None
            for bucket in self._buckets:
                if bucket is not None:
                    yield from bucket
        else:
            assert self._keys is not None and self._values is not None
            for key, value in zip(self._keys, self._values):
                if key is not _EMPTY and key is not _TOMBSTONE:
                    yield key, value

    def keys(self) -> Iterator[Any]:
        """Yield every live key, in slot order."""
        for key, _value in self.items():
            yield key

    def values(self) -> Iterator[Any]:
        """Yield every live value, in slot order."""
        for _key, value in self.items():
            yield value

    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys, like a ``dict``."""
        return self.keys()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def probe_count(self, key: Any) -> int:
        """Entries a lookup of ``key`` examines. For analysis, not hot paths.

        The hardware-independent cost of a lookup, used by the benchmark to
        compare measured probe counts against Knuth's formulas:

        * Chaining - keys compared within the bucket: the key's 1-based
          position if present, the whole bucket's length if absent (0 for an
          empty bucket).
        * Linear probing - slots inspected, counting the slot where the key
          is found or, for a miss, the empty slot that ends the search.

        Args:
            key: Key to look up.

        Returns:
            The number of entries examined.

        Examples:
            >>> table = HashTable(capacity=8, strategy="linear_probing",
            ...                   max_load_factor=0.9)
            >>> for key in (0, 8, 16):       # all hash to slot 0
            ...     table.insert(key)
            >>> [table.probe_count(k) for k in (0, 8, 16, 24)]
            [1, 2, 3, 4]
        """
        if self._chaining:
            assert self._buckets is not None
            bucket = self._buckets[hash(key) & self._mask]
            if bucket is None:
                return 0
            for position, (existing, _value) in enumerate(bucket, start=1):
                if existing is key or existing == key:
                    return position
            return len(bucket)

        assert self._keys is not None
        keys, mask = self._keys, self._mask
        index = hash(key) & mask
        probes = 0
        while True:
            probes += 1
            existing = keys[index]
            if existing is _EMPTY:
                return probes
            if existing is not _TOMBSTONE and (existing is key or existing == key):
                return probes
            index = (index + 1) & mask
