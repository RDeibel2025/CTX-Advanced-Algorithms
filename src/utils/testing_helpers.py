"""Predicates and fixtures-in-waiting shared by the test suite.

These live in ``src`` rather than in ``tests/conftest.py`` for two
reasons: they are ordinary library code that benefits from being
doctested alongside everything else, and keeping one implementation means
``conftest.py`` and any future test module cannot drift apart on what
"sorted" or "same elements" means.

The two predicates that matter most:

* :func:`is_sorted` — is the output in non-decreasing order?
* :func:`has_same_elements` — is the output a *permutation* of the input?

Both are needed. Either alone is easy to pass with a broken sort:
``lambda a: []`` is trivially sorted, and ``lambda a: a`` trivially has the
same elements. Only the conjunction says the algorithm actually sorted.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable, Iterable, List, Sequence

__all__ = [
    "is_sorted",
    "has_same_elements",
    "is_valid_sort",
    "assert_sorts_correctly",
    "StabilityRecord",
    "make_stability_records",
    "extract_tags",
    "random_list",
    "broken_sort_drops_element",
    "broken_sort_returns_zeros",
    "broken_sort_unsorted",
]


def is_sorted(arr: Sequence[Any]) -> bool:
    """Return True if ``arr`` is in non-decreasing order.

    Non-decreasing rather than strictly increasing, because duplicates are
    legal in a sorted list.

    Args:
        arr: Any indexable sequence of comparable elements.

    Returns:
        True if every element is less than or equal to its successor.
        Empty and single-element sequences are sorted by definition.

    Examples:
        >>> is_sorted([1, 2, 2, 3])
        True
        >>> is_sorted([1, 3, 2])
        False
        >>> is_sorted([])
        True
        >>> is_sorted([7])
        True
        >>> is_sorted(["apple", "fig", "pear"])
        True
    """
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def has_same_elements(arr1: Iterable[Any], arr2: Iterable[Any]) -> bool:
    """Return True if both iterables contain the same elements the same number of times.

    This is a *multiset* comparison, using :class:`collections.Counter`.
    Comparing as sets would let a sort that duplicated one element and
    dropped another slip through; comparing lengths alone would let a sort
    that swapped in the wrong value slip through.

    Args:
        arr1: First iterable.
        arr2: Second iterable.

    Returns:
        True if the two are permutations of each other.

    Examples:
        >>> has_same_elements([1, 2, 3], [3, 2, 1])
        True
        >>> has_same_elements([1, 1, 2], [1, 2, 2])
        False
        >>> has_same_elements([1, 2, 3], [1, 2])
        False
        >>> has_same_elements([], [])
        True

        A set comparison would wrongly call this pair equal; a multiset
        comparison does not:

        >>> set([1, 1, 2]) == set([1, 2, 2])
        True
        >>> has_same_elements([1, 1, 2], [1, 2, 2])
        False
    """
    try:
        return Counter(arr1) == Counter(arr2)
    except TypeError:
        # Unhashable elements (lists, dicts): fall back to an
        # order-insensitive comparison that still respects duplicates.
        return sorted(arr1) == sorted(arr2)


def is_valid_sort(original: Sequence[Any], result: Sequence[Any]) -> bool:
    """Return True if ``result`` is a correctly sorted permutation of ``original``.

    Args:
        original: The input that was handed to the algorithm.
        result: What the algorithm returned.

    Returns:
        True only if ``result`` is both ordered and a permutation.

    Examples:
        >>> is_valid_sort([3, 1, 2], [1, 2, 3])
        True
        >>> is_valid_sort([3, 1, 2], [1, 2])          # dropped an element
        False
        >>> is_valid_sort([3, 1, 2], [3, 2, 1])       # not ordered
        False
        >>> is_valid_sort([3, 1, 2], [0, 0, 0])       # ordered, wrong values
        False
    """
    return is_sorted(result) and has_same_elements(original, result)


def assert_sorts_correctly(sort_function: Callable, data: List[Any]) -> List[Any]:
    """Run ``sort_function`` on ``data`` and assert the full contract holds.

    Checks all three obligations at once: the output matches Python's own
    :func:`sorted`, the input list was not mutated, and the output is a
    fresh list rather than the caller's list handed back.

    Args:
        sort_function: The sort under test.
        data: Input list. Not modified.

    Returns:
        The sorted output, so a caller can make further assertions on it.

    Raises:
        AssertionError: If any part of the contract is violated.

    Examples:
        >>> from src.sorting import bubble_sort
        >>> assert_sorts_correctly(bubble_sort, [3, 1, 2])
        [1, 2, 3]
        >>> assert_sorts_correctly(bubble_sort, [])
        []
        >>> assert_sorts_correctly(lambda a: sorted(a) + [99], [3, 1, 2])
        Traceback (most recent call last):
            ...
        AssertionError: <lambda> did not return a permutation of its input
    """
    name = getattr(sort_function, "__name__", repr(sort_function))
    snapshot = list(data)

    result = sort_function(data)

    assert isinstance(result, list), f"{name} returned {type(result).__name__}, not a list"
    assert has_same_elements(snapshot, result), (
        f"{name} did not return a permutation of its input"
    )
    assert is_sorted(result), f"{name} returned a list that is not in order"
    assert result == sorted(snapshot), f"{name} disagrees with Python's sorted()"
    assert data == snapshot, f"{name} mutated its argument"
    assert result is not data, f"{name} returned the caller's list instead of a copy"
    return result


class StabilityRecord:
    """A ``(key, tag)`` pair whose *ordering* depends on the key alone.

    This class exists because the obvious way to test stability does not
    work. Tagging elements as plain ``(key, tag)`` tuples fails, because
    Python compares tuples element by element: when two keys tie, the
    comparison falls through to the tag and silently *imposes* ascending
    tag order. Every sort then looks stable, including one that is not.

    Ordering here reads ``key`` only, so a tie is a genuine tie and the
    algorithm's own tie-handling is what decides the output order. That
    makes stability observable.

    Equality, by contrast, compares the whole record. Ordering and
    equality deliberately disagree: ordering must be blind to the tag for
    the test to mean anything, while equality and hashing must see it so
    that multiset checks such as :func:`has_same_elements` still
    distinguish two records that share a key.

    Attributes:
        key: The value the sort compares.
        tag: The record's original position, used as a tracer.

    Examples:
        >>> a, b = StabilityRecord(1, 0), StabilityRecord(1, 1)
        >>> a < b, b < a          # equal keys: neither precedes the other
        (False, False)
        >>> a > b, b > a          # and neither follows the other
        (False, False)
        >>> a == b                # but they are still distinct records
        False
        >>> StabilityRecord(1, 5) < StabilityRecord(2, 0)
        True
        >>> StabilityRecord(3, 7)
        StabilityRecord(key=3, tag=7)
    """

    __slots__ = ("key", "tag")

    def __init__(self, key: Any, tag: int) -> None:
        self.key = key
        self.tag = tag

    # All four ordering operators are written out rather than derived with
    # functools.total_ordering. total_ordering builds __gt__ as
    # "not (self < other or self == other)", which would consult __eq__ —
    # and __eq__ here sees the tag. Two records with equal keys and
    # different tags would then report BOTH a > b and b > a, and a stable
    # sort would look unstable. Defining each operator directly keeps
    # ordering strictly key-based.
    def __lt__(self, other: "StabilityRecord") -> bool:
        """True if this record's key sorts before the other's."""
        if not isinstance(other, StabilityRecord):
            return NotImplemented
        return self.key < other.key

    def __le__(self, other: "StabilityRecord") -> bool:
        """True if this record's key sorts at or before the other's."""
        if not isinstance(other, StabilityRecord):
            return NotImplemented
        return self.key <= other.key

    def __gt__(self, other: "StabilityRecord") -> bool:
        """True if this record's key sorts after the other's."""
        if not isinstance(other, StabilityRecord):
            return NotImplemented
        return self.key > other.key

    def __ge__(self, other: "StabilityRecord") -> bool:
        """True if this record's key sorts at or after the other's."""
        if not isinstance(other, StabilityRecord):
            return NotImplemented
        return self.key >= other.key

    def __eq__(self, other: Any) -> bool:
        """Compare the whole record, so distinct tags remain distinguishable."""
        if not isinstance(other, StabilityRecord):
            return NotImplemented
        return self.key == other.key and self.tag == other.tag

    def __hash__(self) -> int:
        return hash((self.key, self.tag))

    def __repr__(self) -> str:
        return f"StabilityRecord(key={self.key!r}, tag={self.tag!r})"


def make_stability_records(keys: Sequence[Any]) -> List[StabilityRecord]:
    """Build key-bearing records for testing stability.

    Each element carries a unique, ascending tag. A *stable* sort leaves
    the tags within every equal-key group in ascending order; an unstable
    one may reorder them. See :class:`StabilityRecord` for why a plain
    tuple will not do here.

    Args:
        keys: The sort keys, in their original order.

    Returns:
        A list of :class:`StabilityRecord`, tagged ``0, 1, 2, ...``.

    Examples:
        >>> make_stability_records([2, 1, 2])
        [StabilityRecord(key=2, tag=0), StabilityRecord(key=1, tag=1), StabilityRecord(key=2, tag=2)]

        Insertion sort is stable, so the two key-2 records keep their
        original 0-then-2 order:

        >>> from src.sorting import insertion_sort
        >>> extract_tags(insertion_sort(make_stability_records([2, 1, 2])))
        [1, 0, 2]

        Selection sort is not, and its long-range swap shows up here:

        >>> from src.sorting import selection_sort
        >>> extract_tags(selection_sort(make_stability_records([1, 1, 0])))
        [2, 1, 0]
    """
    return [StabilityRecord(key, index) for index, key in enumerate(keys)]


def extract_tags(records: Sequence[StabilityRecord]) -> List[int]:
    """Pull the tracer tags out of records made by :func:`make_stability_records`.

    Examples:
        >>> extract_tags(make_stability_records(["b", "a"]))
        [0, 1]
        >>> extract_tags([StabilityRecord(1, 4), StabilityRecord(2, 3)])
        [4, 3]
    """
    return [record.tag for record in records]


def random_list(size: int, seed: int = 42, low: int = -1000, high: int = 1000) -> List[int]:
    """Build a reproducible list of random integers.

    Uses a dedicated :class:`random.Random` instance so the global random
    state — which other tests may depend on — is never disturbed.

    Args:
        size: Number of elements.
        seed: Seed for this list.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        A list of ``size`` integers in ``[low, high]``.

    Examples:
        >>> random_list(5, seed=1) == random_list(5, seed=1)
        True
        >>> len(random_list(1000))
        1000
        >>> all(-1000 <= v <= 1000 for v in random_list(200))
        True
    """
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(size)]


# ----------------------------------------------------------------------
# Deliberately incorrect sorts. These exist so the test suite can prove
# that the verification machinery actually fails when it should — a
# checker that never rejects anything is not a checker.
# ----------------------------------------------------------------------
def broken_sort_drops_element(arr: List[Any]) -> List[Any]:
    """Sort correctly but silently drop the last element.

    The output is perfectly ordered, so an ordering check alone accepts it.
    Only the multiset check catches the loss.

    Examples:
        >>> broken_sort_drops_element([3, 1, 2])
        [1, 2]
        >>> is_sorted(broken_sort_drops_element([3, 1, 2]))
        True
        >>> has_same_elements([3, 1, 2], broken_sort_drops_element([3, 1, 2]))
        False
    """
    return sorted(arr)[:-1]


def broken_sort_returns_zeros(arr: List[Any]) -> List[Any]:
    """Return a list of zeros of the right length.

    Ordered and the right length, but the values are wrong.

    Examples:
        >>> broken_sort_returns_zeros([3, 1, 2])
        [0, 0, 0]
        >>> is_sorted(broken_sort_returns_zeros([3, 1, 2]))
        True
        >>> has_same_elements([3, 1, 2], broken_sort_returns_zeros([3, 1, 2]))
        False
    """
    return [0] * len(arr)


def broken_sort_unsorted(arr: List[Any]) -> List[Any]:
    """Return the input unchanged.

    A permutation of the input (trivially), but not ordered.

    Examples:
        >>> broken_sort_unsorted([3, 1, 2])
        [3, 1, 2]
        >>> has_same_elements([3, 1, 2], broken_sort_unsorted([3, 1, 2]))
        True
        >>> is_sorted(broken_sort_unsorted([3, 1, 2]))
        False
    """
    return list(arr)
