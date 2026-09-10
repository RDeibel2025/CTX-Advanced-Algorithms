"""Tests for the binary heaps and the priority queue.

The heap property - no child comes before its parent - is checked here by
an independent function rather than by trusting ``heap.is_valid()``, so a
bug in the heap and a matching bug in its self-check cannot cancel out.

Three claims made in the docstrings are tested by *counting comparisons*
rather than by timing, because counts do not depend on the machine:

* ``heapify`` is O(n), not the O(n log n) of repeated insertion;
* ``insert`` is O(log n) in the worst case but O(1) expected on random
  input;
* ``extract`` is Theta(log n).

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import heapq
import math
import operator
import random

import pytest

import src.structures.heap as heap_module
from src.structures.heap import MaxHeap, MinHeap, PriorityQueue
from tests.conftest import HEAP_CLASSES

#: For each heap: the predicate "a belongs above b", the name of its extract
#: method, and whether draining it yields descending order.
ORDERING = {
    "min_heap": (operator.lt, "extract_min", False),
    "max_heap": (operator.gt, "extract_max", True),
}
HEAP_NAMES = sorted(HEAP_CLASSES)


@pytest.fixture(params=HEAP_NAMES, ids=HEAP_NAMES)
def heap_name(request) -> str:
    """Parametrised over both heaps, by name, so the ordering can be looked up."""
    return request.param


def assert_heap_property(heap, before) -> None:
    """Fail if any item in the heap's array comes before its parent."""
    data = heap.to_list()
    for child in range(1, len(data)):
        parent = (child - 1) // 2
        assert not before(data[child], data[parent]), (
            f"item {data[child]!r} at index {child} belongs above its parent "
            f"{data[parent]!r} at index {parent}"
        )


def drain(heap, name: str) -> list:
    """Extract every item from ``heap`` using the right method for its ordering."""
    extract = getattr(heap, ORDERING[name][1])
    return [extract() for _ in range(len(heap))]


def expected_order(values, name: str) -> list:
    return sorted(values, reverse=ORDERING[name][2])


class Counted:
    """An orderable wrapper that counts every comparison made on it."""

    comparisons = 0
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    def __lt__(self, other: "Counted") -> bool:
        Counted.comparisons += 1
        return self.value < other.value

    def __gt__(self, other: "Counted") -> bool:
        Counted.comparisons += 1
        return self.value > other.value


# ----------------------------------------------------------------------
# Empty, single, duplicates
# ----------------------------------------------------------------------
class TestEdgeCases:
    """Empty heaps, one element, and repeated values."""

    def test_new_heap_is_empty(self, heap_name):
        heap = HEAP_CLASSES[heap_name]()
        assert heap.is_empty()
        assert len(heap) == 0
        assert heap.to_list() == []

    def test_peek_on_empty_heap_raises(self, heap_name):
        with pytest.raises(IndexError):
            HEAP_CLASSES[heap_name]().peek()

    def test_extract_on_empty_heap_raises(self, heap_name):
        heap = HEAP_CLASSES[heap_name]()
        with pytest.raises(IndexError):
            getattr(heap, ORDERING[heap_name][1])()

    def test_single_element(self, heap_name):
        heap = HEAP_CLASSES[heap_name]()
        heap.insert(7)
        assert heap.peek() == 7 and len(heap) == 1 and not heap.is_empty()
        assert getattr(heap, ORDERING[heap_name][1])() == 7
        assert heap.is_empty()

    def test_duplicates_are_all_kept(self, heap_name):
        values = [5, 3, 5, 1, 3, 5, 9, 1]
        heap = HEAP_CLASSES[heap_name](values)
        assert drain(heap, heap_name) == expected_order(values, heap_name)

    def test_all_identical_values(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([4] * 50)
        assert drain(heap, heap_name) == [4] * 50

    def test_extracting_down_to_empty_and_reusing(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([3, 1, 2])
        drain(heap, heap_name)
        assert heap.is_empty()
        heap.insert(10)
        assert heap.peek() == 10


# ----------------------------------------------------------------------
# Ordering and the heap property
# ----------------------------------------------------------------------
class TestOrdering:
    """Draining returns sorted order, and the invariant holds throughout."""

    def test_inserted_keys_drain_in_order(self, heap_name, structure_key_sets):
        for case, keys in structure_key_sets.items():
            heap = HEAP_CLASSES[heap_name]()
            for key in keys:
                heap.insert(key)
            assert drain(heap, heap_name) == expected_order(keys, heap_name), case

    def test_heapified_keys_drain_in_order(self, heap_name, structure_key_sets):
        for case, keys in structure_key_sets.items():
            heap = HEAP_CLASSES[heap_name](keys)
            assert drain(heap, heap_name) == expected_order(keys, heap_name), case

    def test_heap_property_after_every_insert_and_every_extract(self, heap_name):
        """2,000 random operations, with the invariant checked after each one."""
        before, extract_name, _descending = ORDERING[heap_name]
        rng = random.Random(7)
        heap = HEAP_CLASSES[heap_name]()
        for _ in range(2000):
            if len(heap) and rng.random() < 0.4:
                getattr(heap, extract_name)()
            else:
                heap.insert(rng.randint(-100, 100))
            assert_heap_property(heap, before)

    def test_min_heap_agrees_with_heapq_on_random_operations(self):
        rng = random.Random(8)
        mine, reference = MinHeap(), []
        for _ in range(5000):
            if reference and rng.random() < 0.45:
                assert mine.extract_min() == heapq.heappop(reference)
            else:
                value = rng.randint(-1000, 1000)
                mine.insert(value)
                heapq.heappush(reference, value)
            assert len(mine) == len(reference)
            if reference:
                assert mine.peek() == reference[0]

    def test_peek_does_not_remove(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([3, 1, 2])
        top = heap.peek()
        assert heap.peek() == top and len(heap) == 3

    @pytest.mark.parametrize(
        "values",
        [
            ["pear", "apple", "fig", "banana"],
            [3.5, -0.5, 2.25, 100.0, 0.0],
            [(2, "b"), (1, "z"), (2, "a")],
        ],
        ids=["strings", "floats", "tuples"],
    )
    def test_any_orderable_type(self, heap_name, values):
        heap = HEAP_CLASSES[heap_name](values)
        assert drain(heap, heap_name) == expected_order(values, heap_name)

    def test_type_defining_only_less_than_works_in_both_heaps(self, heap_name):
        """The max-heap's > falls back to the reflected __lt__."""

        class OnlyLt:
            __slots__ = ("value",)

            def __init__(self, value):
                self.value = value

            def __lt__(self, other):
                return self.value < other.value

        heap = HEAP_CLASSES[heap_name]([OnlyLt(v) for v in [3, 9, 1, 7]])
        drained = [item.value for item in drain(heap, heap_name)]
        assert drained == expected_order([3, 9, 1, 7], heap_name)


# ----------------------------------------------------------------------
# heapify
# ----------------------------------------------------------------------
class TestHeapify:
    """heapify() builds a valid heap bottom-up, in linear time."""

    def test_heapify_on_arbitrary_arrays_produces_valid_heaps(self, heap_name):
        before = ORDERING[heap_name][0]
        rng = random.Random(9)
        for _ in range(300):
            values = [rng.randint(-50, 50) for _ in range(rng.randint(0, 120))]
            heap = HEAP_CLASSES[heap_name]()
            heap.heapify(values)
            assert_heap_property(heap, before)
            assert sorted(heap.to_list()) == sorted(values)

    def test_heapify_replaces_existing_contents(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([100, 200])
        heap.heapify([3, 1, 2])
        assert sorted(heap.to_list()) == [1, 2, 3]

    def test_heapify_without_arguments_repairs_the_current_array(self, heap_name):
        before = ORDERING[heap_name][0]
        heap = HEAP_CLASSES[heap_name](range(64))
        heap._data.reverse()                     # deliberately break the property
        assert not heap.is_valid()
        heap.heapify()
        assert_heap_property(heap, before)

    def test_heapify_does_not_mutate_its_argument(self, heap_name):
        values = [5, 3, 8, 1]
        HEAP_CLASSES[heap_name](values)
        assert values == [5, 3, 8, 1]

    @pytest.mark.parametrize("size", [1_000, 10_000, 50_000])
    def test_heapify_is_linear_while_repeated_insert_is_not(self, size):
        """Bottom-up build: at most 2n comparisons, at every size.

        Inserting the same reverse-ordered input one item at a time makes
        every item sift to the root, so its cost per item grows with log n.
        """
        values = list(range(size, 0, -1))

        Counted.comparisons = 0
        MinHeap([Counted(v) for v in values])
        build = Counted.comparisons

        Counted.comparisons = 0
        heap = MinHeap()
        for value in values:
            heap.insert(Counted(value))
        repeated = Counted.comparisons

        assert build <= 2 * size
        assert repeated > build * 3


# ----------------------------------------------------------------------
# Complexity claims, by counting
# ----------------------------------------------------------------------
class TestComplexityClaims:
    """The docstrings' complexity claims, checked with comparison counts."""

    def test_insert_is_constant_expected_on_random_input(self):
        """A random new item usually stops a level or two above the leaf."""
        size = 2 ** 14
        rng = random.Random(10)
        heap = MinHeap()
        Counted.comparisons = 0
        for _ in range(size):
            heap.insert(Counted(rng.random()))
        assert Counted.comparisons / size < 3

    def test_insert_is_logarithmic_in_the_worst_case(self):
        """Descending input into a min-heap sends every item to the root."""
        size = 2 ** 14
        heap = MinHeap()
        Counted.comparisons = 0
        for value in range(size, 0, -1):
            heap.insert(Counted(value))
        per_insert = Counted.comparisons / size
        assert math.log2(size) - 3 < per_insert <= math.log2(size)

    def test_extract_is_logarithmic(self):
        size = 2 ** 14
        rng = random.Random(11)
        heap = MinHeap([Counted(rng.random()) for _ in range(size)])
        Counted.comparisons = 0
        for _ in range(size):
            heap.extract_min()
        per_extract = Counted.comparisons / size
        assert math.log2(size) < per_extract <= 2 * math.log2(size)


# ----------------------------------------------------------------------
# Inspection helpers
# ----------------------------------------------------------------------
class TestInspection:
    """Membership, iteration, copying, repr and the self-check."""

    def test_contains_is_a_membership_scan(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([5, 3, 8])
        assert 3 in heap and 4 not in heap

    def test_iteration_yields_every_item_in_storage_order(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([5, 3, 8, 1])
        assert list(heap) == heap.to_list()
        assert sorted(heap) == [1, 3, 5, 8]

    def test_to_list_is_a_copy(self, heap_name):
        heap = HEAP_CLASSES[heap_name]([2, 1])
        snapshot = heap.to_list()
        snapshot.append(99)
        assert len(heap) == 2

    def test_repr_names_the_class(self, heap_name):
        assert type(HEAP_CLASSES[heap_name]()).__name__ in repr(HEAP_CLASSES[heap_name]())

    def test_is_valid_detects_a_broken_heap(self):
        heap = MinHeap([1, 2, 3])
        assert heap.is_valid()
        heap._data[0] = 99
        assert not heap.is_valid()


# ----------------------------------------------------------------------
# PriorityQueue
# ----------------------------------------------------------------------
class TestPriorityQueue:
    """A stable priority queue that delegates to the heap."""

    def test_items_come_out_in_priority_order(self):
        queue = PriorityQueue()
        for item, priority in [("c", 3), ("a", 1), ("d", 4), ("b", 2)]:
            queue.push(item, priority)
        assert [queue.pop() for _ in range(4)] == ["a", "b", "c", "d"]

    def test_equal_priorities_come_out_first_in_first_out(self):
        queue = PriorityQueue()
        for index in range(50):
            queue.push(index, priority=index % 3)
        drained = [queue.pop() for _ in range(50)]
        expected = [i for p in range(3) for i in range(50) if i % 3 == p]
        assert drained == expected

    def test_unorderable_items_never_raise(self):
        """Without the counter tiebreak, a tie would compare the dicts."""
        queue = PriorityQueue()
        for index in range(20):
            queue.push({"job": index}, priority=1)
        assert [queue.pop()["job"] for _ in range(20)] == list(range(20))

    def test_highest_first_serves_largest_priority_and_stays_stable(self):
        queue = PriorityQueue(highest_first=True)
        for index, priority in enumerate([2, 1, 2, 1, 2]):
            queue.push({"id": index}, priority)
        assert [queue.pop()["id"] for _ in range(5)] == [0, 2, 4, 1, 3]

    def test_matches_a_stable_sort_on_random_input(self):
        rng = random.Random(12)
        queue = PriorityQueue()
        entries = [(rng.randint(0, 9), index) for index in range(1000)]
        for priority, index in entries:
            queue.push(index, priority)
        expected = [index for _p, index in sorted(entries, key=lambda e: e[0])]
        assert [queue.pop() for _ in range(1000)] == expected

    def test_pop_with_priority_and_peeks(self):
        queue = PriorityQueue()
        queue.push("x", 5)
        queue.push("y", 2)
        assert queue.peek() == "y" and queue.peek_priority() == 2
        assert queue.pop_with_priority() == (2, "y")
        assert len(queue) == 1

    def test_empty_queue_raises_on_pop_and_peek(self):
        queue = PriorityQueue()
        assert queue.is_empty()
        for method in (queue.pop, queue.peek, queue.peek_priority, queue.pop_with_priority):
            with pytest.raises(IndexError):
                method()

    def test_delegates_to_min_heap_or_max_heap(self):
        assert isinstance(PriorityQueue()._heap, MinHeap)
        assert isinstance(PriorityQueue(highest_first=True)._heap, MaxHeap)

    def test_every_push_goes_through_the_heap_insert(self, monkeypatch):
        """Behavioural proof of delegation: patch the heap and watch it called."""
        calls = {"insert": 0}
        real_insert = heap_module._BinaryHeap.insert

        def counting_insert(self, value):
            calls["insert"] += 1
            return real_insert(self, value)

        monkeypatch.setattr(heap_module._BinaryHeap, "insert", counting_insert)
        queue = PriorityQueue()
        for index in range(10):
            queue.push(index, index)
        assert calls["insert"] == 10
