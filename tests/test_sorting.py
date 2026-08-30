"""Tests for the three basic sorting algorithms.

The suite is organised around the two halves of the Week 1 contract:

* **What the algorithm must produce** — a new list that is in order and is
  a permutation of the input, for every input shape the assignment names.
  Both properties are always checked together, because either alone is
  trivially satisfiable by a broken sort.
* **How the algorithm must behave** — reject a non-list argument with
  ``TypeError``, never mutate the caller's list, and never hand the
  caller's own list back.

Nearly every test is parametrised across all three algorithms, so a case
written once is a case run three times.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random

import pytest

from src.sorting import bubble_sort, insertion_sort, selection_sort
from src.sorting.basic_sorts import _require_list
from src.utils.testing_helpers import (
    assert_sorts_correctly,
    extract_tags,
    has_same_elements,
    is_sorted,
    make_stability_records,
)
from tests.conftest import SORTING_ALGORITHMS, STABLE_ALGORITHMS

ALGORITHMS = [SORTING_ALGORITHMS[name] for name in sorted(SORTING_ALGORITHMS)]
ALGORITHM_IDS = sorted(SORTING_ALGORITHMS)
STABLE = [STABLE_ALGORITHMS[name] for name in sorted(STABLE_ALGORITHMS)]
STABLE_IDS = sorted(STABLE_ALGORITHMS)


# ----------------------------------------------------------------------
# 1-9. The required input shapes
# ----------------------------------------------------------------------
class TestRequiredEdgeCases:
    """Every input shape the assignment names, against all three algorithms."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_empty_list(self, algorithm):
        """An empty list sorts to an empty list without special-casing."""
        assert algorithm([]) == []

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_single_element(self, algorithm):
        """A one-element list is returned unchanged in value."""
        assert algorithm([42]) == [42]

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_two_elements_both_orders(self, algorithm):
        """The smallest input where order actually matters."""
        assert algorithm([1, 2]) == [1, 2]
        assert algorithm([2, 1]) == [1, 2]

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_already_sorted(self, algorithm):
        """Sorted input comes back unchanged — the best case for bubble/insertion."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert algorithm(data) == data

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_reverse_sorted(self, algorithm):
        """Reverse-sorted input — the worst case for all three."""
        assert algorithm([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == list(range(1, 11))

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_duplicates(self, algorithm):
        """Repeated values are all kept, in the right multiplicity."""
        data = [5, 2, 8, 2, 9, 1, 5, 5, 8, 1]
        result = algorithm(data)
        assert result == sorted(data)
        assert result.count(5) == 3
        assert has_same_elements(data, result)

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_all_identical(self, algorithm):
        """Every element equal: nothing should move and nothing should be lost."""
        data = [7] * 25
        assert algorithm(data) == data

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_negative_numbers(self, algorithm):
        """Negative values order correctly — no accidental reliance on sign."""
        assert algorithm([-5, -1, -9, -3, -7]) == [-9, -7, -5, -3, -1]

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_mixed_positive_and_negative(self, algorithm):
        """Mixed signs, including duplicate zeros straddling the boundary."""
        data = [3, -1, 0, -7, 12, -12, 0, 5]
        assert algorithm(data) == [-12, -7, -1, 0, 0, 3, 5, 12]

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_large_array(self, algorithm, large_random_array):
        """A 1,000-element random array sorts correctly."""
        result = algorithm(large_random_array)
        assert len(result) == 1000
        assert is_sorted(result)
        assert has_same_elements(large_random_array, result)
        assert result == sorted(large_random_array)

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_every_sample_array(self, algorithm, sample_arrays):
        """Sweep the whole fixture: every named edge case, one algorithm."""
        for case_name, data in sample_arrays.items():
            snapshot = list(data)
            result = algorithm(data)
            assert result == sorted(snapshot), f"wrong output for case {case_name!r}"
            assert data == snapshot, f"input mutated for case {case_name!r}"


# ----------------------------------------------------------------------
# 10. Correctness: ordered AND a permutation
# ----------------------------------------------------------------------
class TestCorrectness:
    """Output must be both in order and a permutation of the input."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_sorted_and_same_elements(self, algorithm, sample_arrays):
        """Both halves of correctness, checked together on every case.

        Either check alone is weak: ``lambda a: []`` passes the ordering
        check and ``lambda a: a`` passes the permutation check.
        """
        for case_name, data in sample_arrays.items():
            result = algorithm(data)
            assert is_sorted(result), f"not ordered for case {case_name!r}"
            assert has_same_elements(data, result), (
                f"not a permutation for case {case_name!r}"
            )

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_agrees_with_python_sorted(self, algorithm):
        """Cross-check against the reference implementation on random input.

        Two hundred randomly shaped and randomly valued arrays, compared
        element for element with Python's own ``sorted``. This is the test
        most likely to catch an off-by-one in a loop bound that the fixed
        cases happen to miss.
        """
        rng = random.Random(20260830)
        for _ in range(200):
            size = rng.randint(0, 60)
            data = [rng.randint(-50, 50) for _ in range(size)]
            assert algorithm(data) == sorted(data)

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_full_contract_helper(self, algorithm, sample_arrays):
        """The bundled contract assertion passes for every case.

        ``assert_sorts_correctly`` checks ordering, permutation, agreement
        with ``sorted``, non-mutation and that a fresh list was returned,
        all in one call.
        """
        for data in sample_arrays.values():
            assert_sorts_correctly(algorithm, data)

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_idempotent(self, algorithm):
        """Sorting an already-sorted result changes nothing."""
        data = [5, 3, 9, 1, 3]
        once = algorithm(data)
        assert algorithm(once) == once

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_non_integer_comparable_types(self, algorithm):
        """Any comparable element type works, not just int."""
        assert algorithm([3.5, -0.5, 2.25, 100.0, -0.5]) == [-0.5, -0.5, 2.25, 3.5, 100.0]
        assert algorithm(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]
        assert algorithm([(2, "b"), (1, "z"), (2, "a")]) == [(1, "z"), (2, "a"), (2, "b")]
        assert algorithm([True, False, True]) == [False, True, True]


# ----------------------------------------------------------------------
# 11. Stability
# ----------------------------------------------------------------------
class TestStability:
    """Equal keys must keep their relative order — for the sorts that promise it."""

    @pytest.mark.parametrize("algorithm", STABLE, ids=STABLE_IDS)
    def test_stable_algorithms_preserve_order_of_equal_keys(self, algorithm):
        """Bubble and insertion sort keep equal-key records in input order.

        The records compare on their key alone (see
        :class:`~src.utils.testing_helpers.StabilityRecord`), so a tie is a
        real tie and the algorithm's own tie-handling decides the output.
        """
        records = make_stability_records([3, 1, 3, 1, 2, 3])
        result = algorithm(records)

        assert [r.key for r in result] == [1, 1, 2, 3, 3, 3]
        # Within each equal-key group the tracer tags stay ascending.
        assert extract_tags(result) == [1, 3, 4, 0, 2, 5]

    @pytest.mark.parametrize("algorithm", STABLE, ids=STABLE_IDS)
    def test_stable_algorithms_on_all_identical_keys(self, algorithm):
        """With every key equal, a stable sort must not move anything at all."""
        records = make_stability_records([9] * 8)
        assert extract_tags(algorithm(records)) == list(range(8))

    @pytest.mark.parametrize("algorithm", STABLE, ids=STABLE_IDS)
    def test_stable_algorithms_over_many_random_inputs(self, algorithm):
        """Stability holds across random inputs with a small key alphabet.

        Only four distinct keys over twenty positions guarantees many ties,
        which is where an unstable implementation gives itself away.
        """
        rng = random.Random(7)
        for _ in range(50):
            keys = [rng.randint(0, 3) for _ in range(20)]
            result = algorithm(make_stability_records(keys))
            # Python's sorted() is stable, so it defines the expected answer.
            expected = [tag for _key, tag in sorted(zip(keys, range(20)))]
            assert extract_tags(result) == expected

    def test_selection_sort_is_documented_as_unstable(self):
        """Selection sort is *not* stable, and this pins the known behaviour.

        This is a characterisation test, not a requirement: it records that
        the long-range swap moves the first key-1 record behind the second
        one, which is exactly why selection sort is excluded from
        ``STABLE_ALGORITHMS``.
        """
        records = make_stability_records([1, 1, 0])
        result = selection_sort(records)

        assert [r.key for r in result] == [0, 1, 1]
        assert extract_tags(result) == [2, 1, 0]
        # Its output is still a correct sort by key — just not a stable one.
        assert extract_tags(result) != [2, 0, 1]


# ----------------------------------------------------------------------
# Contract: error handling and non-mutation (spec section 5.1)
# ----------------------------------------------------------------------
class TestErrorHandling:
    """A non-list argument must raise TypeError, with a useful message."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    @pytest.mark.parametrize(
        "bad_input",
        [
            "not a list",
            42,
            3.14,
            None,
            (3, 1, 2),
            {3, 1, 2},
            {"a": 1},
            iter([3, 1, 2]),
            b"bytes",
        ],
        ids=[
            "str",
            "int",
            "float",
            "None",
            "tuple",
            "set",
            "dict",
            "iterator",
            "bytes",
        ],
    )
    def test_type_error_on_non_list(self, algorithm, bad_input):
        """Every non-list type is rejected before any work is done."""
        with pytest.raises(TypeError):
            algorithm(bad_input)

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_type_error_message_names_the_function_and_the_type(self, algorithm):
        """The message says which function was called and what it received."""
        with pytest.raises(TypeError) as caught:
            algorithm("abc")
        message = str(caught.value)
        assert algorithm.__name__ in message
        assert "str" in message
        assert "list" in message

    def test_require_list_accepts_a_list(self):
        """The shared validator passes a genuine list through silently."""
        assert _require_list([1, 2, 3], "demo") is None

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_incomparable_elements_raise_type_error(self, algorithm):
        """Comparing incomparable elements propagates TypeError from Python."""
        with pytest.raises(TypeError):
            algorithm([1, "two", 3])


class TestNonDestructive:
    """The caller's list must survive the call untouched."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_input_is_not_mutated(self, algorithm, sample_arrays):
        """Every fixture case is byte-identical after sorting."""
        for case_name, data in sample_arrays.items():
            snapshot = list(data)
            algorithm(data)
            assert data == snapshot, f"{algorithm.__name__} mutated case {case_name!r}"

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_returns_a_new_list_object(self, algorithm):
        """The returned list is a distinct object, even when nothing moved."""
        for data in ([], [1], [1, 2, 3]):
            result = algorithm(data)
            assert result is not data
            assert isinstance(result, list)

    @pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHM_IDS)
    def test_mutating_the_result_does_not_affect_the_input(self, algorithm):
        """The copy is genuine, not a view: writing to it leaves the input alone."""
        data = [3, 1, 2]
        result = algorithm(data)
        result[0] = 999
        assert data == [3, 1, 2]


# ----------------------------------------------------------------------
# The optimization the rubric asks for by name
# ----------------------------------------------------------------------
class TestBubbleSortOptimization:
    """Bubble sort must carry the early-exit flag, not just be correct."""

    def test_early_exit_avoids_comparisons_on_sorted_input(self):
        """On sorted input the optimized version does one pass, not n-1.

        Counting comparisons directly is the only way to distinguish an
        optimized bubble sort from a naive one, since both produce the same
        output. A list of n sorted elements should cost exactly n-1
        comparisons: one clean pass, then the early exit. The naive version
        would cost n(n-1)/2.
        """
        comparisons = {"count": 0}

        class Counted(int):
            """An int that tallies every ``>`` comparison it takes part in."""

            def __gt__(self, other):
                comparisons["count"] += 1
                return int(self) > int(other)

        size = 50
        data = [Counted(i) for i in range(size)]

        bubble_sort(data)

        naive_cost = size * (size - 1) // 2
        assert comparisons["count"] == size - 1, (
            "optimized bubble sort should make exactly n-1 comparisons on "
            "sorted input"
        )
        assert comparisons["count"] < naive_cost

    def test_early_exit_still_sorts_nearly_sorted_input(self):
        """The early exit must not stop before the list is actually sorted."""
        data = list(range(100))
        data[0], data[99] = data[99], data[0]
        assert bubble_sort(data) == list(range(100))

    def test_worst_case_still_does_the_full_work(self):
        """On reverse-sorted input no pass is swap-free, so no pass is skipped."""
        comparisons = {"count": 0}

        class Counted(int):
            def __gt__(self, other):
                comparisons["count"] += 1
                return int(self) > int(other)

        size = 30
        data = [Counted(i) for i in range(size - 1, -1, -1)]

        assert bubble_sort(data) == list(range(size))
        assert comparisons["count"] == size * (size - 1) // 2


class TestSelectionSortComparisonCount:
    """Selection sort's comparison count is fixed by n alone."""

    @pytest.mark.parametrize(
        "arrangement",
        ["sorted", "reverse", "random"],
        ids=["sorted", "reverse", "random"],
    )
    def test_comparison_count_is_independent_of_input_order(self, arrangement):
        """n(n-1)/2 comparisons regardless of arrangement.

        This is the mechanism behind the benchmark finding that selection
        sort's runtime is nearly flat across data types, established here
        by counting rather than by timing.
        """
        comparisons = {"count": 0}

        class Counted(int):
            def __lt__(self, other):
                comparisons["count"] += 1
                return int(self) < int(other)

        size = 40
        values = list(range(size))
        if arrangement == "reverse":
            values.reverse()
        elif arrangement == "random":
            random.Random(3).shuffle(values)

        selection_sort([Counted(v) for v in values])

        assert comparisons["count"] == size * (size - 1) // 2


class TestInsertionSortBestCase:
    """Insertion sort's best case is linear, and that is observable."""

    def test_sorted_input_costs_n_minus_one_comparisons(self):
        """Each element stops immediately against its already-sorted neighbour."""
        comparisons = {"count": 0}

        class Counted(int):
            def __gt__(self, other):
                comparisons["count"] += 1
                return int(self) > int(other)

        size = 50
        insertion_sort([Counted(i) for i in range(size)])
        assert comparisons["count"] == size - 1

    def test_reverse_input_costs_the_quadratic_maximum(self):
        """Reverse-sorted input makes every element travel the full distance."""
        comparisons = {"count": 0}

        class Counted(int):
            def __gt__(self, other):
                comparisons["count"] += 1
                return int(self) > int(other)

        size = 40
        insertion_sort([Counted(i) for i in range(size - 1, -1, -1)])
        assert comparisons["count"] == size * (size - 1) // 2
