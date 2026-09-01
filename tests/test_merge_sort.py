"""Tests for merge sort and its linear merge helper.

Two things are tested that a plain "does it sort?" suite would miss:

* **The merge is linear.** A merge built on ``pop(0)`` still produces the
  right answer, so no correctness test can catch it — but it makes merge
  sort O(n^2 log n). :class:`TestMergeIsLinear` counts operations and
  checks the wall-clock scaling instead.
* **The sort is stable.** Merge sort's stability rests entirely on the
  ``<=`` in the merge comparison; flipping it to ``<`` breaks stability
  while leaving every ordering test passing.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
import time
from collections import Counter

import pytest

from src.sorting.merge_sort import merge, merge_sort
from src.utils.testing_helpers import (
    extract_tags,
    has_same_elements,
    is_sorted,
    make_stability_records,
)


class TestEdgeCases:
    """The input shapes the assignment names explicitly."""

    def test_empty_list(self):
        assert merge_sort([]) == []

    def test_single_element(self):
        assert merge_sort([42]) == [42]

    def test_two_elements(self):
        assert merge_sort([1, 2]) == [1, 2]
        assert merge_sort([2, 1]) == [1, 2]

    def test_already_sorted(self):
        data = list(range(20))
        assert merge_sort(data) == data

    def test_reverse_sorted(self):
        assert merge_sort(list(range(20, 0, -1))) == list(range(1, 21))

    def test_all_elements_identical(self):
        assert merge_sort([7] * 30) == [7] * 30

    def test_many_duplicates(self):
        data = [5, 2, 8, 2, 9, 1, 5, 5, 8, 1, 2, 9, 9]
        result = merge_sort(data)
        assert result == sorted(data)
        assert Counter(result) == Counter(data)

    def test_negative_numbers(self):
        assert merge_sort([-5, -1, -9, -3, -7]) == [-9, -7, -5, -3, -1]

    def test_mixed_positive_and_negative(self):
        data = [3, -1, 0, -7, 12, -12, 0, 5]
        assert merge_sort(data) == [-12, -7, -1, 0, 0, 3, 5, 12]

    def test_floats(self):
        data = [3.5, -0.5, 2.25, 100.0, -0.5, 0.0]
        assert merge_sort(data) == sorted(data)

    def test_strings(self):
        data = ["pear", "apple", "fig", "banana", "apple"]
        assert merge_sort(data) == sorted(data)

    def test_tuples(self):
        data = [(2, "b"), (1, "z"), (2, "a")]
        assert merge_sort(data) == [(1, "z"), (2, "a"), (2, "b")]

    def test_odd_and_even_lengths_both_work(self):
        """The midpoint split leaves unequal halves on odd lengths."""
        for size in range(0, 34):
            data = list(range(size, 0, -1))
            assert merge_sort(data) == sorted(data), f"failed at size {size}"


class TestContract:
    """The contract shared with every other sort in this package."""

    @pytest.mark.parametrize(
        "bad_input",
        ["not a list", 42, 3.14, None, (3, 1, 2), {3, 1, 2}, {"a": 1}, b"bytes"],
        ids=["str", "int", "float", "None", "tuple", "set", "dict", "bytes"],
    )
    def test_type_error_on_non_list(self, bad_input):
        with pytest.raises(TypeError):
            merge_sort(bad_input)

    def test_type_error_message_names_function_and_type(self):
        with pytest.raises(TypeError) as caught:
            merge_sort("abc")
        message = str(caught.value)
        assert "merge_sort" in message and "str" in message and "list" in message

    def test_input_is_not_mutated(self):
        data = [5, 3, 9, 1, 3, 8, 2]
        snapshot = list(data)
        merge_sort(data)
        assert data == snapshot

    def test_returns_a_new_list_object(self):
        for data in ([], [1], [3, 1, 2]):
            result = merge_sort(data)
            assert result is not data
            assert isinstance(result, list)

    def test_mutating_the_result_leaves_the_input_alone(self):
        data = [3, 1, 2]
        result = merge_sort(data)
        result[0] = 999
        assert data == [3, 1, 2]


class TestStability:
    """Equal keys must keep their input order."""

    def test_equal_keys_keep_their_order(self):
        records = make_stability_records([3, 1, 3, 1, 2, 3])
        result = merge_sort(records)
        assert [r.key for r in result] == [1, 1, 2, 3, 3, 3]
        assert extract_tags(result) == [1, 3, 4, 0, 2, 5]

    def test_all_identical_keys_do_not_move(self):
        records = make_stability_records([9] * 12)
        assert extract_tags(merge_sort(records)) == list(range(12))

    def test_stability_over_many_random_inputs(self):
        """A small key alphabet guarantees many ties, which is where it shows."""
        rng = random.Random(11)
        for _ in range(60):
            keys = [rng.randint(0, 3) for _ in range(24)]
            result = merge_sort(make_stability_records(keys))
            expected = [tag for _key, tag in sorted(zip(keys, range(24)))]
            assert extract_tags(result) == expected


class TestProperties:
    """Randomised property testing against Python's own sorted()."""

    def test_output_equals_sorted_and_is_a_permutation(self):
        rng = random.Random(20260906)
        for _ in range(300):
            size = rng.randint(0, 80)
            data = [rng.randint(-100, 100) for _ in range(size)]
            result = merge_sort(data)
            assert result == sorted(data)
            assert Counter(result) == Counter(data)

    def test_holds_for_every_data_shape(self):
        rng = random.Random(5)
        shapes = {
            "sorted": list(range(200)),
            "reverse": list(range(200, 0, -1)),
            "identical": [1] * 200,
            "few_unique": [rng.randint(0, 2) for _ in range(200)],
            "random": [rng.randint(-500, 500) for _ in range(200)],
        }
        for name, data in shapes.items():
            result = merge_sort(data)
            assert is_sorted(result), f"not ordered: {name}"
            assert has_same_elements(data, result), f"not a permutation: {name}"

    def test_large_input(self):
        rng = random.Random(3)
        data = [rng.randint(-10_000, 10_000) for _ in range(5000)]
        assert merge_sort(data) == sorted(data)


class TestMergeHelper:
    """`merge` is a named, separately testable function, as required."""

    def test_merges_two_sorted_runs(self):
        assert merge([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]

    def test_handles_empty_runs(self):
        assert merge([], []) == []
        assert merge([], [1, 2]) == [1, 2]
        assert merge([1, 2], []) == [1, 2]

    def test_handles_disjoint_runs(self):
        assert merge([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]
        assert merge([4, 5, 6], [1, 2, 3]) == [1, 2, 3, 4, 5, 6]

    def test_handles_unequal_lengths(self):
        assert merge([1], [0, 2, 3, 4, 5]) == [0, 1, 2, 3, 4, 5]

    def test_keeps_duplicates_from_both_sides(self):
        assert merge([1, 1, 2], [1, 2, 2]) == [1, 1, 1, 2, 2, 2]

    def test_takes_from_the_left_on_ties(self):
        """This tie-break is the source of merge sort's stability."""
        left = make_stability_records([1, 1])
        right = [r for r in make_stability_records([1])]
        right[0].tag = 99
        assert extract_tags(merge(left, right)) == [0, 1, 99]

    def test_does_not_mutate_its_arguments(self):
        left, right = [1, 3, 5], [2, 4, 6]
        left_snapshot, right_snapshot = list(left), list(right)
        merge(left, right)
        assert left == left_snapshot and right == right_snapshot

    @pytest.mark.parametrize("bad_input", ["ab", (1, 2), None, 5])
    def test_type_error_on_non_list(self, bad_input):
        with pytest.raises(TypeError):
            merge(bad_input, [1, 2])
        with pytest.raises(TypeError):
            merge([1, 2], bad_input)

    def test_random_runs_merge_correctly(self):
        rng = random.Random(17)
        for _ in range(200):
            left = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
            right = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
            assert merge(left, right) == sorted(left + right)


class TestMergeIsLinear:
    """The merge must be O(n+m), not accidentally quadratic.

    A merge written with ``left.pop(0)`` returns exactly the right answer,
    so no correctness assertion can detect it. These two tests can.
    """

    def test_element_access_count_is_linear(self):
        """Counting reads shows the work is proportional to n+m, not n*m."""
        reads = {"count": 0}

        class Counted(int):
            def __le__(self, other):
                reads["count"] += 1
                return int(self) <= int(other)

            def __gt__(self, other):  # pragma: no cover - completeness
                reads["count"] += 1
                return int(self) > int(other)

        size = 500
        left = [Counted(2 * i) for i in range(size)]
        right = [Counted(2 * i + 1) for i in range(size)]

        merge(left, right)

        # A correct merge makes at most n+m-1 comparisons; a quadratic one
        # would make orders of magnitude more.
        assert reads["count"] <= 2 * size
        assert reads["count"] < size * size / 10

    def test_wall_clock_scales_linearly(self):
        """Ten times the input should cost roughly ten times the time.

        A quadratic merge would cost roughly a hundred times as much. The
        threshold is deliberately loose - this test is separating O(n) from
        O(n^2), not measuring a constant factor.
        """

        def timed(size: int) -> float:
            left = list(range(0, 2 * size, 2))
            right = list(range(1, 2 * size, 2))
            best = float("inf")
            for _ in range(3):
                start = time.perf_counter()
                merge(left, right)
                best = min(best, time.perf_counter() - start)
            return best

        small = timed(2_000)
        large = timed(20_000)

        # Linear predicts a ratio near 10, quadratic near 100.
        assert large / max(small, 1e-9) < 30, (
            "merge appears to scale worse than linearly; check for pop(0) "
            "or a nested scan in the merge loop"
        )
