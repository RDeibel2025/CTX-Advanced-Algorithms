"""Tests for randomized quicksort and its partition schemes.

Beyond the usual correctness coverage, three properties get their own
tests because they are the ones that fail silently or catastrophically:

* **Bounded recursion depth.** Recursing into the smaller partition is
  what keeps quicksort inside Python's 1000-frame limit at n=50,000.
  :class:`TestRecursionDepth` measures the depth actually reached and
  sorts inputs large enough that a two-sided recursion would raise
  ``RecursionError``.
* **Partition invariants.** :func:`partition` and
  :func:`partition_three_way` are separately testable functions, and a
  partition that returns ``high`` sends the driver into an infinite loop
  rather than a wrong answer.
* **Scheme equivalence.** ``three_way=True`` and ``three_way=False`` are
  different algorithms that must agree on every output.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

import pytest

import src.sorting.quick_sort as quick_sort_module
from src.sorting.quick_sort import (
    INSERTION_SORT_CUTOFF,
    partition,
    partition_lomuto,
    partition_three_way,
    quick_sort,
    quick_sort_lomuto,
)
from src.utils.testing_helpers import has_same_elements, is_sorted

SCHEMES = [False, True]
SCHEME_IDS = ["two_way_hoare", "three_way_dnf"]


class TestEdgeCases:
    """The input shapes the assignment names explicitly, both schemes."""

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_empty_list(self, three_way):
        assert quick_sort([], three_way=three_way) == []

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_single_element(self, three_way):
        assert quick_sort([42], three_way=three_way) == [42]

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_two_elements(self, three_way):
        assert quick_sort([1, 2], three_way=three_way) == [1, 2]
        assert quick_sort([2, 1], three_way=three_way) == [1, 2]

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_already_sorted(self, three_way):
        data = list(range(200))
        assert quick_sort(data, seed=1, three_way=three_way) == data

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_reverse_sorted(self, three_way):
        data = list(range(200, 0, -1))
        assert quick_sort(data, seed=1, three_way=three_way) == list(range(1, 201))

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_all_elements_identical(self, three_way):
        assert quick_sort([7] * 300, seed=1, three_way=three_way) == [7] * 300

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_many_duplicates(self, three_way):
        rng = random.Random(4)
        data = [rng.randrange(10) for _ in range(400)]
        result = quick_sort(data, seed=1, three_way=three_way)
        assert result == sorted(data)
        assert Counter(result) == Counter(data)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_few_unique_values(self, three_way):
        """Three distinct values — the case that breaks Lomuto partitioning."""
        rng = random.Random(5)
        data = [rng.randrange(3) for _ in range(500)]
        assert quick_sort(data, seed=1, three_way=three_way) == sorted(data)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_negative_numbers(self, three_way):
        assert quick_sort([-5, -1, -9, -3, -7], three_way=three_way) == [
            -9, -7, -5, -3, -1
        ]

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_mixed_positive_and_negative(self, three_way):
        data = [3, -1, 0, -7, 12, -12, 0, 5]
        assert quick_sort(data, three_way=three_way) == [-12, -7, -1, 0, 0, 3, 5, 12]

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_floats(self, three_way):
        data = [3.5, -0.5, 2.25, 100.0, -0.5, 0.0]
        assert quick_sort(data, three_way=three_way) == sorted(data)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_strings(self, three_way):
        data = ["pear", "apple", "fig", "banana", "apple"]
        assert quick_sort(data, three_way=three_way) == sorted(data)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_every_length_up_to_beyond_the_cutoff(self, three_way):
        """Sizes straddling INSERTION_SORT_CUTOFF exercise both code paths."""
        for size in range(0, 3 * INSERTION_SORT_CUTOFF + 5):
            data = list(range(size, 0, -1))
            assert quick_sort(data, seed=size, three_way=three_way) == sorted(data), (
                f"failed at size {size}"
            )


class TestContract:
    """The contract shared with every other sort in this package."""

    @pytest.mark.parametrize(
        "bad_input",
        ["not a list", 42, 3.14, None, (3, 1, 2), {3, 1, 2}, {"a": 1}, b"bytes"],
        ids=["str", "int", "float", "None", "tuple", "set", "dict", "bytes"],
    )
    def test_type_error_on_non_list(self, bad_input):
        with pytest.raises(TypeError):
            quick_sort(bad_input)

    def test_type_error_message_names_function_and_type(self):
        with pytest.raises(TypeError) as caught:
            quick_sort("abc")
        message = str(caught.value)
        assert "quick_sort" in message and "str" in message and "list" in message

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_input_is_not_mutated(self, three_way):
        rng = random.Random(6)
        data = [rng.randint(-100, 100) for _ in range(300)]
        snapshot = list(data)
        quick_sort(data, seed=1, three_way=three_way)
        assert data == snapshot

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_returns_a_new_list_object(self, three_way):
        for data in ([], [1], [3, 1, 2]):
            result = quick_sort(data, three_way=three_way)
            assert result is not data
            assert isinstance(result, list)

    def test_mutating_the_result_leaves_the_input_alone(self):
        data = [3, 1, 2]
        result = quick_sort(data)
        result[0] = 999
        assert data == [3, 1, 2]


class TestSchemeEquivalenceAndDeterminism:
    """Both partition schemes must agree, and a seeded run must repeat."""

    def test_both_schemes_produce_identical_output(self):
        rng = random.Random(8)
        for _ in range(150):
            size = rng.randint(0, 120)
            alphabet = rng.choice([2, 3, 10, 1000])
            data = [rng.randrange(alphabet) for _ in range(size)]
            two_way = quick_sort(data, seed=42, three_way=False)
            three_way = quick_sort(data, seed=42, three_way=True)
            assert two_way == three_way == sorted(data)

    def test_lomuto_agrees_with_the_production_scheme(self):
        rng = random.Random(9)
        for _ in range(100):
            data = [rng.randrange(rng.choice([3, 50])) for _ in range(rng.randint(0, 90))]
            assert quick_sort_lomuto(data, seed=7) == quick_sort(data, seed=7)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_same_seed_gives_the_same_result(self, three_way):
        rng = random.Random(10)
        data = [rng.randint(-500, 500) for _ in range(400)]
        first = quick_sort(data, seed=2026, three_way=three_way)
        second = quick_sort(data, seed=2026, three_way=three_way)
        assert first == second

    def test_different_seeds_still_sort_correctly(self):
        """Pivot choice changes the work done, never the answer."""
        rng = random.Random(11)
        data = [rng.randint(-50, 50) for _ in range(200)]
        expected = sorted(data)
        for seed in range(25):
            assert quick_sort(data, seed=seed) == expected

    def test_unseeded_runs_still_sort_correctly(self):
        rng = random.Random(12)
        data = [rng.randint(-50, 50) for _ in range(200)]
        for _ in range(10):
            assert quick_sort(data) == sorted(data)


class TestProperties:
    """Randomised property testing against Python's own sorted()."""

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_output_equals_sorted_and_is_a_permutation(self, three_way):
        rng = random.Random(20260906)
        for trial in range(300):
            size = rng.randint(0, 90)
            data = [rng.randint(-100, 100) for _ in range(size)]
            result = quick_sort(data, seed=trial, three_way=three_way)
            assert result == sorted(data)
            assert Counter(result) == Counter(data)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_holds_across_small_alphabets(self, three_way):
        """Tiny alphabets are where partition schemes go wrong."""
        rng = random.Random(13)
        for alphabet in (1, 2, 3, 5):
            for _ in range(40):
                size = rng.randint(0, 150)
                data = [rng.randrange(alphabet) for _ in range(size)]
                result = quick_sort(data, seed=alphabet, three_way=three_way)
                assert is_sorted(result)
                assert has_same_elements(data, result)

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_large_input(self, three_way):
        rng = random.Random(14)
        data = [rng.randint(-10_000, 10_000) for _ in range(5000)]
        assert quick_sort(data, seed=1, three_way=three_way) == sorted(data)


class TestPartitionHelpers:
    """The partition functions are named and separately testable."""

    def test_hoare_partition_invariant(self):
        rng = random.Random(15)
        for _ in range(200):
            size = rng.randint(2, 60)
            data = [rng.randint(-30, 30) for _ in range(size)]
            working = list(data)
            split = partition(working, 0, size - 1, random.Random(3))

            # The split must land inside the range. Returning `high` would
            # leave the driver recursing on an unshrinking range forever.
            assert 0 <= split < size - 1, (
                f"split {split} outside [0, {size - 2}]"
            )
            assert max(working[: split + 1]) <= min(working[split + 1 :])
            assert Counter(working) == Counter(data)

    def test_hoare_partition_never_returns_high(self):
        """Returning `high` is the infinite-loop bug this guards against."""
        rng = random.Random(16)
        for _ in range(300):
            size = rng.randint(2, 40)
            alphabet = rng.choice([1, 2, 3])
            data = [rng.randrange(alphabet) for _ in range(size)]
            assert partition(data, 0, size - 1, random.Random(rng.random())) < size - 1

    def test_hoare_splits_all_equal_ranges_near_the_middle(self):
        """Hoare's advantage over Lomuto, shown directly."""
        data = [5] * 100
        split = partition(data, 0, 99, random.Random(0))
        assert 20 < split < 80, "an all-equal range should not split lopsidedly"

    def test_three_way_partition_invariant(self):
        rng = random.Random(17)
        for _ in range(200):
            size = rng.randint(2, 60)
            data = [rng.randrange(rng.choice([2, 3, 40])) for _ in range(size)]
            working = list(data)
            lt, gt = partition_three_way(working, 0, size - 1, random.Random(4))

            pivot = working[lt]
            assert all(v < pivot for v in working[:lt])
            assert all(v == pivot for v in working[lt : gt + 1])
            assert all(v > pivot for v in working[gt + 1 :])
            assert Counter(working) == Counter(data)

    def test_three_way_finishes_an_all_equal_range_in_one_pass(self):
        data = [4] * 50
        assert partition_three_way(data, 0, 49, random.Random(0)) == (0, 49)

    def test_lomuto_partition_invariant(self):
        rng = random.Random(18)
        for _ in range(200):
            size = rng.randint(2, 60)
            data = [rng.randint(-30, 30) for _ in range(size)]
            working = list(data)
            pivot_position = partition_lomuto(working, 0, size - 1, random.Random(5))
            pivot = working[pivot_position]
            assert all(v <= pivot for v in working[:pivot_position])
            assert all(v >= pivot for v in working[pivot_position + 1 :])
            assert Counter(working) == Counter(data)

    def test_lomuto_degenerates_on_equal_elements(self):
        """Characterisation test: this is why Lomuto is not the default.

        Every element compares `<= pivot`, so all of them land on the left
        and the split is maximally lopsided. Hoare, on the same input,
        splits down the middle.
        """
        data = [4] * 50
        assert partition_lomuto(data, 0, 49, random.Random(0)) == 49

        hoare_data = [4] * 50
        hoare_split = partition(hoare_data, 0, 49, random.Random(0))
        assert hoare_split < 49


class TestOptimizations:
    """The three required optimizations are present and doing something."""

    def test_insertion_sort_cutoff_constant_is_exposed(self):
        """The report cites the exact threshold, so it must be a constant."""
        assert isinstance(INSERTION_SORT_CUTOFF, int)
        assert 2 <= INSERTION_SORT_CUTOFF <= 64

    def test_small_ranges_are_handed_to_insertion_sort(self, monkeypatch):
        """Ranges at or below the cutoff must not be partitioned further."""
        calls = {"insertion": 0, "partition": 0}

        real_insertion = quick_sort_module._insertion_sort_range
        real_partition = quick_sort_module.partition

        def counted_insertion(arr, low, high):
            calls["insertion"] += 1
            assert high - low + 1 <= INSERTION_SORT_CUTOFF
            return real_insertion(arr, low, high)

        def counted_partition(arr, low, high, rng):
            calls["partition"] += 1
            return real_partition(arr, low, high, rng)

        monkeypatch.setattr(quick_sort_module, "_insertion_sort_range", counted_insertion)
        monkeypatch.setattr(quick_sort_module, "partition", counted_partition)

        rng = random.Random(19)
        data = [rng.randint(0, 1000) for _ in range(500)]
        assert quick_sort(data, seed=1) == sorted(data)

        assert calls["insertion"] > 0, "the cutoff never fired"
        assert calls["partition"] > 0, "nothing was partitioned"

    def test_a_range_at_the_cutoff_is_not_partitioned(self, monkeypatch):
        partitioned = {"count": 0}
        real_partition = quick_sort_module.partition

        def counted(arr, low, high, rng):
            partitioned["count"] += 1
            return real_partition(arr, low, high, rng)

        monkeypatch.setattr(quick_sort_module, "partition", counted)
        data = list(range(INSERTION_SORT_CUTOFF, 0, -1))
        assert quick_sort(data, seed=1) == sorted(data)
        assert partitioned["count"] == 0


class TestRecursionDepth:
    """Stack depth must stay O(log n) on every input, including the bad ones."""

    @staticmethod
    def _peak_depth(data, three_way, seed=1):
        """Sort `data` while recording the deepest recursion reached."""
        state = {"depth": 0, "peak": 0}
        real_driver = quick_sort_module._quick_sort_in_place

        def traced(arr, low, high, rng, tw):
            state["depth"] += 1
            state["peak"] = max(state["peak"], state["depth"])
            try:
                return real_driver(arr, low, high, rng, tw)
            finally:
                state["depth"] -= 1

        quick_sort_module._quick_sort_in_place = traced
        try:
            result = quick_sort(data, seed=seed, three_way=three_way)
        finally:
            quick_sort_module._quick_sort_in_place = real_driver
        return result, state["peak"]

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    @pytest.mark.parametrize(
        "shape",
        ["random", "sorted", "reverse", "all_equal", "three_unique"],
    )
    def test_depth_stays_logarithmic(self, three_way, shape):
        """Depth must scale like log2(n), not like n."""
        size = 20_000
        rng = random.Random(20)
        data = {
            "random": [rng.randint(0, 10**6) for _ in range(size)],
            "sorted": list(range(size)),
            "reverse": list(range(size, 0, -1)),
            "all_equal": [7] * size,
            "three_unique": [i % 3 for i in range(size)],
        }[shape]

        result, peak = self._peak_depth(data, three_way)

        assert result == sorted(data)
        # log2(20000) is about 14.3. A generous multiple of that still sits
        # far below the n-deep recursion a naive quicksort would produce.
        assert peak <= 60, f"{shape}: recursion reached depth {peak}"

    @pytest.mark.parametrize("three_way", SCHEMES, ids=SCHEME_IDS)
    def test_sorts_input_far_larger_than_the_recursion_limit(self, three_way):
        """A two-sided recursion would raise RecursionError well before this.

        Sorted input of 50,000 elements is the exact case that kills a
        textbook quicksort: the recursion limit is 1000, and a naive
        implementation recurses once per element.
        """
        size = 50_000
        assert size > sys.getrecursionlimit() * 10

        data = list(range(size))
        assert quick_sort(data, seed=1, three_way=three_way) == data

        reverse = list(range(size, 0, -1))
        assert quick_sort(reverse, seed=1, three_way=three_way) == sorted(reverse)

        few_unique = [i % 3 for i in range(size)]
        assert quick_sort(few_unique, seed=1, three_way=three_way) == sorted(few_unique)
