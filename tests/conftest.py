"""Shared pytest fixtures and predicates for the Week 1 test suite.

``sample_arrays`` is the centre of the sorting tests: one dictionary
holding every edge case the assignment calls for, so a new algorithm can
be put through all of them by adding a single entry to
``SORTING_ALGORITHMS``.

The two predicates :func:`is_sorted` and :func:`has_same_elements` are the
pair that together define "sorted correctly". Their implementations live
in :mod:`src.utils.testing_helpers` so that library code and test code
cannot drift apart on what those words mean; the functions here are the
conftest-level entry points onto that single implementation.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Sequence

import pytest

from src.sorting import bubble_sort, insertion_sort, selection_sort
from src.utils import testing_helpers

#: Every algorithm under test, by display name. Parametrising on this is
#: what makes "every case, against all three algorithms" one line of code
#: per test rather than three.
SORTING_ALGORITHMS: Dict[str, Callable] = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
}

#: The subset that is stable. Bubble and insertion sort only ever move an
#: element past a strictly greater one; selection sort's long-range swap
#: can jump an element over an equal one, so it is excluded.
STABLE_ALGORITHMS: Dict[str, Callable] = {
    "bubble_sort": bubble_sort,
    "insertion_sort": insertion_sort,
}


def is_sorted(arr: List) -> bool:
    """Return True if ``arr`` is in non-decreasing order.

    Thin entry point onto :func:`src.utils.testing_helpers.is_sorted`, kept
    here so tests can use it without importing from ``src``.

    Args:
        arr: Any indexable sequence of comparable elements.

    Returns:
        True if every element is less than or equal to its successor.
    """
    return testing_helpers.is_sorted(arr)


def has_same_elements(arr1: List, arr2: List) -> bool:
    """Return True if the two lists are permutations of each other.

    Compares as multisets via :class:`collections.Counter`, so a sort that
    returned an ordered list with an element dropped or duplicated is
    caught. Thin entry point onto
    :func:`src.utils.testing_helpers.has_same_elements`.

    Args:
        arr1: First list.
        arr2: Second list.

    Returns:
        True if both hold the same elements with the same multiplicities.
    """
    return testing_helpers.has_same_elements(arr1, arr2)


@pytest.fixture
def sample_arrays() -> Dict[str, List]:
    """Every edge case the assignment requires, in one dictionary.

    Returns:
        A mapping of case name to input list, covering: empty input, a
        single element, two elements in both orders, already-sorted and
        reverse-sorted input, duplicates, all-identical elements, negative
        numbers, mixed signs, floats, strings, and a large 1,000-element
        array.
    """
    rng = random.Random(2026)
    return {
        "empty": [],
        "single": [42],
        "two_sorted": [1, 2],
        "two_reversed": [2, 1],
        "already_sorted": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "reverse_sorted": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "random_small": [64, 34, 25, 12, 22, 11, 90],
        "with_duplicates": [5, 2, 8, 2, 9, 1, 5, 5, 8, 1],
        "all_identical": [7, 7, 7, 7, 7, 7, 7],
        "negatives": [-5, -1, -9, -3, -7],
        "mixed_signs": [3, -1, 0, -7, 12, -12, 0, 5],
        "floats": [3.5, -0.5, 2.25, 100.0, -0.5, 0.0],
        "strings": ["pear", "apple", "fig", "banana", "apple"],
        "already_sorted_large": list(range(500)),
        "reverse_sorted_large": list(range(500, 0, -1)),
        "large": [rng.randint(-10_000, 10_000) for _ in range(1000)],
    }


@pytest.fixture
def large_random_array() -> List[int]:
    """A reproducible 1,000-element random array.

    Seeded with 42 through a dedicated :class:`random.Random` instance
    rather than the module-level :func:`random.seed`, so the fixture is
    reproducible without perturbing global random state that another test
    might rely on.

    Returns:
        1,000 integers in the range [-10000, 10000].
    """
    rng = random.Random(42)
    return [rng.randint(-10_000, 10_000) for _ in range(1000)]


@pytest.fixture(params=sorted(SORTING_ALGORITHMS), ids=sorted(SORTING_ALGORITHMS))
def sorting_algorithm(request) -> Callable:
    """Parametrised fixture yielding each sorting algorithm in turn."""
    return SORTING_ALGORITHMS[request.param]


@pytest.fixture(params=sorted(STABLE_ALGORITHMS), ids=sorted(STABLE_ALGORITHMS))
def stable_algorithm(request) -> Callable:
    """Parametrised fixture yielding each *stable* sorting algorithm in turn."""
    return STABLE_ALGORITHMS[request.param]
