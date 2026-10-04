"""Bitmask helpers: a set of small non-negative integers stored in one int.

Held-Karp, the bitmask dynamic program for the travelling salesman problem
in :mod:`src.dp_advanced.bitmask_traveling_salesman`, indexes its table by
*sets of cities*. Storing each set as an ``int`` whose bit ``i`` is 1
exactly when city ``i`` is in the set makes a set hashable for free, lets
it index a plain list directly (``dp[mask][i]``), and turns every set
operation into one or two integer instructions. This module names those
instructions, so the TSP code reads as set algebra rather than as shifts
and masks.

Each function, read as an operation on a set ``S`` of cities numbered
``0..n-1``:

======================== ===================================================
Function                 Meaning for a set of cities
======================== ===================================================
has_bit(S, i)            membership: is city ``i`` in ``S``?
set_bit(S, i)            insertion: ``S`` with city ``i`` added
clear_bit(S, i)          removal: ``S`` with city ``i`` taken out
full_mask(n)             the full set: all ``n`` cities
popcount(S)              the size of the set, ``|S|``
iter_bits(S)             the cities in ``S``, in increasing order
mask_to_list(S)          the cities in ``S`` as a sorted list
subsets_of_size(n, k)    every ``k``-city subset of the ``n`` cities
======================== ===================================================

:func:`subsets_of_size` yields its masks in increasing numeric order. That
order matters to Held-Karp: removing a city from a set clears a bit, which
always makes the mask numerically smaller, so any loop that visits masks
in increasing order has already finished every subset a mask depends on.
Grouping by size as well gives the textbook layering, all sets of size
``k`` before any set of size ``k + 1``.

The one-``int`` representation is also why the TSP module caps ``n``:
there are ``2^n`` masks, and the table holds ``n`` entries for each one.

Contract shared by every public function:

* Every mask, bit index, ``n`` and ``k`` must be a non-negative ``int``.
  Anything else raises ``TypeError``, and ``bool`` is rejected as a type
  error, because ``True`` is an accident, not a set of cities. A negative
  value raises ``ValueError``: Python's negative ints have infinitely many
  1 bits in two's complement and do not describe a finite set.
* No function mutates anything. ``int`` is immutable, so "insertion" and
  "removal" return a new mask and leave the argument unchanged.

Examples:
    >>> tour_so_far = set_bit(set_bit(0, 0), 2)
    >>> tour_so_far
    5
    >>> mask_to_list(tour_so_far), popcount(tour_so_far)
    ([0, 2], 2)
    >>> has_bit(tour_so_far, 1)
    False
    >>> mask_to_list(clear_bit(full_mask(4), 0))
    [1, 2, 3]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Iterator, List

__all__ = [
    "clear_bit",
    "full_mask",
    "has_bit",
    "iter_bits",
    "mask_to_list",
    "popcount",
    "set_bit",
    "subsets_of_size",
]


def _require_non_negative_int(value: int, name: str) -> None:
    """Raise unless ``value`` is a non-negative ``int`` (``bool`` rejected).

    Every public function checks its arguments through this one helper, so
    the error messages are identical across the module.

    Args:
        value: The argument to check.
        name: The parameter name, used in the error message.

    Returns:
        None. This helper is called for its exception, not its value.

    Raises:
        TypeError: If ``value`` is not an ``int``, or is a ``bool``.
        ValueError: If ``value`` is negative.

    Time Complexity:
        O(1) - one isinstance check and one comparison.

    Space Complexity:
        O(1).

    Examples:
        >>> _require_non_negative_int(0, "mask") is None
        True
        >>> _require_non_negative_int(2.0, "mask")
        Traceback (most recent call last):
            ...
        TypeError: mask must be an int, got float
        >>> _require_non_negative_int(True, "i")
        Traceback (most recent call last):
            ...
        TypeError: i must be an int, got bool
        >>> _require_non_negative_int(-1, "n")
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 0, got -1
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def has_bit(mask: int, i: int) -> bool:
    """Return whether bit ``i`` of ``mask`` is set: is city ``i`` in the set?

    Shifts the mask right by ``i`` and tests the lowest bit. An index past
    the highest set bit is simply absent, so it returns ``False`` rather
    than raising.

    Args:
        mask: The set, as a non-negative int.
        i: The bit index (city number) to test.

    Returns:
        ``True`` if bit ``i`` is 1, otherwise ``False``.

    Raises:
        TypeError: If ``mask`` or ``i`` is not an ``int``, or is a ``bool``.
        ValueError: If ``mask`` or ``i`` is negative.

    Time Complexity:
        O(1) for masks that fit in a machine word, which covers every mask
        the TSP module builds; O(b) in general for a b-bit Python int.

    Space Complexity:
        O(1) for word-sized masks.

    Examples:
        >>> has_bit(0b101, 0)
        True
        >>> has_bit(0b101, 1)
        False
        >>> has_bit(0b101, 2)
        True
        >>> has_bit(0, 0)
        False
        >>> has_bit(0b101, 50)
        False
        >>> has_bit(-1, 0)
        Traceback (most recent call last):
            ...
        ValueError: mask must be >= 0, got -1
    """
    _require_non_negative_int(mask, "mask")
    _require_non_negative_int(i, "i")
    return (mask >> i) & 1 == 1


def set_bit(mask: int, i: int) -> int:
    """Return ``mask`` with bit ``i`` set: the set with city ``i`` inserted.

    A bitwise OR with ``1 << i``. Inserting a city that is already present
    returns the same mask, exactly as adding an existing element to a set
    leaves it unchanged.

    Args:
        mask: The set, as a non-negative int. Not modified.
        i: The bit index (city number) to insert.

    Returns:
        The new mask, ``mask | (1 << i)``.

    Raises:
        TypeError: If ``mask`` or ``i`` is not an ``int``, or is a ``bool``.
        ValueError: If ``mask`` or ``i`` is negative.

    Time Complexity:
        O(1) for word-sized masks; O(b) in general for a b-bit Python int.

    Space Complexity:
        O(1) for word-sized masks.

    Examples:
        >>> set_bit(0, 0)
        1
        >>> set_bit(0b100, 1)
        6
        >>> set_bit(0b110, 1)
        6
        >>> set_bit(0, 3.0)
        Traceback (most recent call last):
            ...
        TypeError: i must be an int, got float
    """
    _require_non_negative_int(mask, "mask")
    _require_non_negative_int(i, "i")
    return mask | (1 << i)


def clear_bit(mask: int, i: int) -> int:
    """Return ``mask`` with bit ``i`` cleared: the set with city ``i`` removed.

    A bitwise AND with the complement of ``1 << i``. Removing a city that is
    not present returns the same mask. The result is always non-negative,
    because clearing a bit of a non-negative int cannot set the sign.

    Args:
        mask: The set, as a non-negative int. Not modified.
        i: The bit index (city number) to remove.

    Returns:
        The new mask, ``mask & ~(1 << i)``.

    Raises:
        TypeError: If ``mask`` or ``i`` is not an ``int``, or is a ``bool``.
        ValueError: If ``mask`` or ``i`` is negative.

    Time Complexity:
        O(1) for word-sized masks; O(b) in general for a b-bit Python int.

    Space Complexity:
        O(1) for word-sized masks.

    Examples:
        >>> clear_bit(0b111, 0)
        6
        >>> clear_bit(0b111, 1)
        5
        >>> clear_bit(0b100, 0)
        4
        >>> clear_bit(0, 0)
        0
        >>> clear_bit(True, 0)
        Traceback (most recent call last):
            ...
        TypeError: mask must be an int, got bool
    """
    _require_non_negative_int(mask, "mask")
    _require_non_negative_int(i, "i")
    return mask & ~(1 << i)


def full_mask(n: int) -> int:
    """Return the mask with the low ``n`` bits set: the set of all n cities.

    ``(1 << n) - 1`` is ``n`` ones in binary. Held-Karp reads its answer
    from the row of this mask, the row where every city has been visited.
    ``full_mask(0)`` is the empty set, 0.

    Args:
        n: The number of cities.

    Returns:
        ``(1 << n) - 1``.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(1) for word-sized results; O(n) in general, the bits written.

    Space Complexity:
        O(1) for word-sized results; O(n) bits in general.

    Examples:
        >>> full_mask(0)
        0
        >>> full_mask(1)
        1
        >>> full_mask(4)
        15
        >>> bin(full_mask(5))
        '0b11111'
        >>> full_mask(-2)
        Traceback (most recent call last):
            ...
        ValueError: n must be >= 0, got -2
    """
    _require_non_negative_int(n, "n")
    return (1 << n) - 1


def popcount(mask: int) -> int:
    """Return the number of set bits in ``mask``: the size of the set.

    Delegates to :meth:`int.bit_count` (Python 3.10+), which counts bits in
    C rather than looping over them in Python.

    Args:
        mask: The set, as a non-negative int.

    Returns:
        The number of 1 bits, ``|S|``.

    Raises:
        TypeError: If ``mask`` is not an ``int``, or is a ``bool``.
        ValueError: If ``mask`` is negative.

    Time Complexity:
        O(1) for word-sized masks; O(b) in general for a b-bit Python int.

    Space Complexity:
        O(1).

    Examples:
        >>> popcount(0)
        0
        >>> popcount(1)
        1
        >>> popcount(0b1011)
        3
        >>> popcount(full_mask(20))
        20
        >>> popcount("7")
        Traceback (most recent call last):
            ...
        TypeError: mask must be an int, got str
    """
    _require_non_negative_int(mask, "mask")
    return mask.bit_count()


def _iter_set_bits(mask: int) -> Iterator[int]:
    """Yield the indices of the set bits of a validated ``mask``, ascending.

    Kept separate from :func:`iter_bits` so that the public function can
    validate eagerly. A generator body does not run until it is first
    advanced, so a check placed inside it would let a bad mask through
    until iteration started.

    Each step isolates the lowest set bit with ``mask & -mask``, reports
    its position, and clears it, so the loop runs once per member of the
    set rather than once per bit position.

    Args:
        mask: A non-negative int, already validated.

    Returns:
        An iterator over the set bit indices in increasing order.

    Time Complexity:
        O(popcount(mask)) steps, each O(1) for word-sized masks.

    Space Complexity:
        O(1) - one int of state.

    Examples:
        >>> list(_iter_set_bits(0b10110))
        [1, 2, 4]
        >>> list(_iter_set_bits(0))
        []
    """
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def iter_bits(mask: int) -> Iterator[int]:
    """Iterate over the indices of the set bits: the cities in the set.

    Indices come out in increasing order. Validation happens when this
    function is called, not when the iterator is first advanced.

    Args:
        mask: The set, as a non-negative int.

    Returns:
        An iterator over the bit indices that are 1, ascending.

    Raises:
        TypeError: If ``mask`` is not an ``int``, or is a ``bool``.
        ValueError: If ``mask`` is negative.

    Time Complexity:
        O(popcount(mask)) to exhaust, O(1) per element for word-sized masks.

    Space Complexity:
        O(1) - the iterator holds one int.

    Examples:
        >>> list(iter_bits(0b1101))
        [0, 2, 3]
        >>> list(iter_bits(1))
        [0]
        >>> list(iter_bits(0))
        []
        >>> iter_bits(-4)
        Traceback (most recent call last):
            ...
        ValueError: mask must be >= 0, got -4
    """
    _require_non_negative_int(mask, "mask")
    return _iter_set_bits(mask)


def mask_to_list(mask: int) -> List[int]:
    """Return the indices of the set bits as a sorted list: the set's cities.

    The list form of :func:`iter_bits`, for printing a set or comparing it
    in a test.

    Args:
        mask: The set, as a non-negative int.

    Returns:
        A new list of the bit indices that are 1, ascending.

    Raises:
        TypeError: If ``mask`` is not an ``int``, or is a ``bool``.
        ValueError: If ``mask`` is negative.

    Time Complexity:
        O(popcount(mask)) for word-sized masks.

    Space Complexity:
        O(popcount(mask)) - the returned list.

    Examples:
        >>> mask_to_list(0b101001)
        [0, 3, 5]
        >>> mask_to_list(full_mask(3))
        [0, 1, 2]
        >>> mask_to_list(0)
        []
        >>> mask_to_list([1, 2])
        Traceback (most recent call last):
            ...
        TypeError: mask must be an int, got list
    """
    return list(iter_bits(mask))


def _gosper_subsets(n: int, k: int) -> Iterator[int]:
    """Yield every mask of ``k`` bits among the low ``n``, in increasing order.

    The private generator behind :func:`subsets_of_size`, with arguments
    already validated. It uses Gosper's hack, which maps a mask to the next
    larger int with the same popcount in a constant number of word
    operations:

    * ``c = x & -x`` isolates the lowest set bit;
    * ``r = x + c`` carries through the lowest run of ones, moving its top
      bit one place left;
    * ``((r ^ x) >> 2) // c`` is the rest of that run, shifted back down to
      the bottom, and OR-ing it into ``r`` restores the popcount.

    Starting from ``(1 << k) - 1``, the smallest mask with ``k`` bits, it
    walks every such mask in increasing order and stops at the first one
    that needs bit ``n`` or higher. ``k == 0`` is handled separately,
    because Gosper's step divides by the lowest set bit and 0 has none.

    Args:
        n: Number of low bits available, already validated.
        k: Bits to set, already validated, with ``k <= n``.

    Returns:
        An iterator over the masks, ascending.

    Time Complexity:
        O(C(n, k)) to exhaust - O(1) per mask for word-sized masks.

    Space Complexity:
        O(1) - two ints of state.

    Examples:
        >>> list(_gosper_subsets(3, 1))
        [1, 2, 4]
        >>> list(_gosper_subsets(3, 0))
        [0]
    """
    if k == 0:
        yield 0
        return

    limit = 1 << n
    x = (1 << k) - 1
    while x < limit:
        yield x
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r


def subsets_of_size(n: int, k: int) -> Iterator[int]:
    """Iterate over every k-city subset of n cities, in increasing numeric order.

    Yields each mask with exactly ``k`` of its low ``n`` bits set, and no
    higher bits, exactly once: ``C(n, k)`` masks in all. ``k == 0`` yields
    the empty set once, ``k == n`` yields :func:`full_mask` ``(n)`` once,
    and ``k > n`` yields nothing, since no such subset exists.

    Increasing numeric order means Held-Karp can process the masks of each
    size in the order they arrive. Validation happens when this function is
    called, not when the iterator is first advanced.

    Args:
        n: The number of cities, so the number of low bits available.
        k: The number of cities in each subset.

    Returns:
        An iterator over the masks, ascending.

    Raises:
        TypeError: If ``n`` or ``k`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` or ``k`` is negative.

    Time Complexity:
        O(C(n, k)) to exhaust - O(1) per mask via Gosper's hack, so no work
        is spent on masks of the wrong size.

    Space Complexity:
        O(1) - the iterator holds two ints.

    Examples:
        >>> list(subsets_of_size(4, 2))
        [3, 5, 6, 9, 10, 12]
        >>> [mask_to_list(m) for m in subsets_of_size(3, 2)]
        [[0, 1], [0, 2], [1, 2]]
        >>> list(subsets_of_size(4, 0))
        [0]
        >>> list(subsets_of_size(4, 4))
        [15]
        >>> list(subsets_of_size(0, 0))
        [0]
        >>> list(subsets_of_size(3, 5))
        []
        >>> sum(1 for _ in subsets_of_size(10, 5))
        252
        >>> subsets_of_size(4, -1)
        Traceback (most recent call last):
            ...
        ValueError: k must be >= 0, got -1
    """
    _require_non_negative_int(n, "n")
    _require_non_negative_int(k, "k")
    if k > n:
        return iter(())
    return _gosper_subsets(n, k)
