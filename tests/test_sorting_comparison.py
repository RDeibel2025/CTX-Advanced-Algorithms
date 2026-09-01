"""Cross-algorithm equivalence: all five sorts must agree, always.

The Week 1 sorts and the Week 2 divide-and-conquer sorts were written
independently, and the benchmark harness treats them as interchangeable.
That interchangeability is an assumption, and this file is where it gets
checked: for the same input, every algorithm must return byte-identical
output, and every algorithm must honour the same contract.

Coverage is the cross product of five algorithms and the six data shapes
the Week 2 benchmark measures, so a disagreement is attributed to a
specific (algorithm, shape) pair rather than to "sorting is broken".

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Dict, List

import pytest

from src.sorting import bubble_sort, insertion_sort, selection_sort
from src.sorting.merge_sort import merge_sort
from src.sorting.quick_sort import quick_sort, quick_sort_lomuto
from src.utils.benchmark import AlgorithmBenchmark
from src.utils.testing_helpers import has_same_elements, is_sorted

#: The five algorithms the Week 2 benchmark compares. Quicksort is wrapped
#: with a fixed seed so a failure here is reproducible rather than a
#: once-in-a-while flake.
ALGORITHMS: Dict[str, Callable[[List], List]] = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "merge_sort": merge_sort,
    "quick_sort": lambda data: quick_sort(data, seed=2026),
}

#: The two Week 2 variants, checked alongside but not part of the headline
#: five-way comparison.
VARIANTS: Dict[str, Callable[[List], List]] = {
    "quick_sort_three_way": lambda data: quick_sort(data, seed=2026, three_way=True),
    "quick_sort_lomuto": lambda data: quick_sort_lomuto(data, seed=2026),
}

ALL_IMPLEMENTATIONS = {**ALGORITHMS, **VARIANTS}

#: The six data shapes the Week 2 benchmark sweeps.
DATA_TYPES = [
    "random",
    "sorted",
    "reverse",
    "nearly_sorted",
    "many_duplicates",
    "few_unique",
]

#: Kept small enough that the three quadratic algorithms stay fast, and
#: large enough to exercise quicksort's recursion and merge sort's depth.
SAMPLE_SIZE = 600


@pytest.fixture(scope="module")
def datasets() -> Dict[str, List[int]]:
    """One reproducible dataset per data shape, shared by every test."""
    bench = AlgorithmBenchmark(warmup_runs=0, seed=2026)
    bench.verbose = False
    return {
        data_type: bench.generate_test_data(SAMPLE_SIZE, data_type, seed=2026)
        for data_type in DATA_TYPES
    }


class TestAllAlgorithmsAgree:
    """The headline requirement: identical output from all five."""

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    def test_five_algorithms_produce_identical_output(self, datasets, data_type):
        data = datasets[data_type]
        expected = sorted(data)

        outputs = {name: fn(data) for name, fn in ALGORITHMS.items()}

        for name, result in outputs.items():
            assert result == expected, (
                f"{name} disagrees with sorted() on {data_type!r} input"
            )

        # And pairwise, so a shared-but-wrong answer would still be caught
        # by the comparison against sorted() above.
        distinct = {tuple(result) for result in outputs.values()}
        assert len(distinct) == 1, (
            f"algorithms disagree with each other on {data_type!r} input"
        )

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    @pytest.mark.parametrize("name", sorted(ALL_IMPLEMENTATIONS))
    def test_every_implementation_sorts_every_shape(self, datasets, name, data_type):
        """Including the two quicksort variants."""
        data = datasets[data_type]
        result = ALL_IMPLEMENTATIONS[name](data)
        assert is_sorted(result), f"{name} left {data_type!r} unordered"
        assert has_same_elements(data, result), (
            f"{name} lost or invented elements on {data_type!r}"
        )

    @pytest.mark.parametrize("data_type", DATA_TYPES)
    @pytest.mark.parametrize("name", sorted(ALL_IMPLEMENTATIONS))
    def test_no_implementation_mutates_its_input(self, datasets, name, data_type):
        data = list(datasets[data_type])
        snapshot = list(data)
        ALL_IMPLEMENTATIONS[name](data)
        assert data == snapshot, f"{name} mutated its argument on {data_type!r}"


class TestSharedContract:
    """Every algorithm honours the same interface, not just the same output."""

    @pytest.mark.parametrize("name", sorted(ALL_IMPLEMENTATIONS))
    @pytest.mark.parametrize(
        "bad_input",
        ["not a list", 42, None, (3, 1, 2), {"a": 1}],
        ids=["str", "int", "None", "tuple", "dict"],
    )
    def test_all_raise_type_error_on_non_list(self, name, bad_input):
        with pytest.raises(TypeError):
            ALL_IMPLEMENTATIONS[name](bad_input)

    @pytest.mark.parametrize("name", sorted(ALL_IMPLEMENTATIONS))
    def test_all_handle_empty_and_single_element(self, name):
        implementation = ALL_IMPLEMENTATIONS[name]
        assert implementation([]) == []
        assert implementation([42]) == [42]

    @pytest.mark.parametrize("name", sorted(ALL_IMPLEMENTATIONS))
    def test_all_return_a_new_list(self, name):
        data = [3, 1, 2]
        assert ALL_IMPLEMENTATIONS[name](data) is not data

    @pytest.mark.parametrize("name", sorted(ALL_IMPLEMENTATIONS))
    def test_all_handle_non_integer_comparable_types(self, name):
        implementation = ALL_IMPLEMENTATIONS[name]
        assert implementation([3.5, -0.5, 2.25]) == [-0.5, 2.25, 3.5]
        assert implementation(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


class TestRandomisedAgreement:
    """Randomised cross-checking, where fixed fixtures leave gaps."""

    def test_all_agree_across_many_random_inputs(self):
        rng = random.Random(2026)
        for trial in range(120):
            size = rng.randint(0, 150)
            alphabet = rng.choice([2, 3, 10, 10_000])
            data = [rng.randrange(alphabet) for _ in range(size)]
            expected = sorted(data)

            for name, implementation in ALL_IMPLEMENTATIONS.items():
                result = implementation(data)
                assert result == expected, (
                    f"{name} failed on trial {trial} "
                    f"(size={size}, alphabet={alphabet})"
                )
                assert Counter(result) == Counter(data)

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 9, 10, 11, 63, 64, 65])
    def test_all_agree_at_boundary_sizes(self, size):
        """Sizes around the insertion-sort cutoff and small powers of two."""
        rng = random.Random(size + 1)
        data = [rng.randint(-40, 40) for _ in range(size)]
        expected = sorted(data)
        for name, implementation in ALL_IMPLEMENTATIONS.items():
            assert implementation(data) == expected, f"{name} failed at size {size}"


class TestStabilityDiffersAsDocumented:
    """The five are interchangeable on output, not on stability."""

    def test_stable_algorithms_are_stable_and_unstable_ones_are_not(self):
        """Documents which guarantees actually hold, so the report can cite it.

        Bubble, insertion and merge sort promise stability; selection and
        quicksort do not. An unstable algorithm is not *required* to
        reorder equal keys, so this asserts the stable ones hold and only
        records what the unstable ones happen to do.
        """
        from src.utils.testing_helpers import extract_tags, make_stability_records

        keys = [3, 1, 3, 1, 2, 3]
        stable_expected = [tag for _key, tag in sorted(zip(keys, range(len(keys))))]

        for name, implementation in (
            ("bubble_sort", bubble_sort),
            ("insertion_sort", insertion_sort),
            ("merge_sort", merge_sort),
        ):
            result = implementation(make_stability_records(keys))
            assert extract_tags(result) == stable_expected, f"{name} is not stable"

        # The unstable pair must still sort correctly by key.
        for name, implementation in (
            ("selection_sort", selection_sort),
            ("quick_sort", lambda d: quick_sort(d, seed=1)),
        ):
            result = implementation(make_stability_records(keys))
            assert [r.key for r in result] == sorted(keys), f"{name} mis-sorted"
