"""Array-backed binary heaps, and a priority queue built on them.

A binary heap keeps a complete binary tree in a flat Python list. The
children of index ``i`` live at ``2i + 1`` and ``2i + 2`` and its parent at
``(i - 1) // 2``, so there are no node objects and no pointers - only the
list and one ordering rule: every parent comes before both of its children.
That rule is weaker than sorting, which is exactly why a heap is cheaper to
maintain than a sorted list while still handing back the next item in
constant time.

The ordering is supplied as a comparator, so the min-heap and the max-heap
are one implementation rather than two copies:

* :class:`MinHeap` - the smallest item is on top, removed by
  :meth:`~MinHeap.extract_min`.
* :class:`MaxHeap` - the largest item is on top, removed by
  :meth:`~MaxHeap.extract_max`.
* :class:`PriorityQueue` - a stable priority queue that delegates every
  operation to one of the two heaps above.

=============== ============ ================================================
Operation       Worst case   Notes
=============== ============ ================================================
insert          O(log n)     O(1) *expected* on random input: a new item
                             usually stops within a level or two of the leaf
extract         O(log n)     the replacement item sinks almost to a leaf, so
                             this is Theta(log n) in practice, not just worst
peek            O(1)
heapify         O(n)         bottom-up build, not n successive inserts
=============== ============ ================================================

The one genuinely surprising line there is ``heapify``. Building a heap by
inserting n items costs O(n log n), but sifting down from the last internal
node to the root costs O(n): most nodes sit near the bottom and can only
sink a level or two, and the sum of those short distances is linear.

Both heaps compare elements with a single operator (``<`` for the min-heap,
``>`` for the max-heap). Python falls back to the reflected ``__lt__`` when a
type defines only that, so any type orderable by ``<`` works in either heap.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import operator
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

T = TypeVar("T")

__all__ = ["MinHeap", "MaxHeap", "PriorityQueue"]


class _BinaryHeap(Generic[T]):
    """Array-backed binary heap ordered by a comparator.

    Not used directly: :class:`MinHeap` and :class:`MaxHeap` are thin
    subclasses that supply the comparator and name the extract method.

    The comparator ``before(a, b)`` answers "does ``a`` belong above ``b``?".
    Every sift operation is written against that one question, so the heap
    property - no child comes before its parent - is the only invariant the
    code has to maintain.

    Args:
        before: Strict ordering predicate. ``operator.lt`` gives a min-heap,
            ``operator.gt`` a max-heap.
        items: Optional initial contents, built in O(n) by :meth:`heapify`.
    """

    __slots__ = ("_data", "_before")

    def __init__(
        self,
        before: Callable[[Any, Any], bool],
        items: Optional[Iterable[T]] = None,
    ) -> None:
        self._before = before
        self._data: List[T] = []
        if items is not None:
            self.heapify(items)

    # ------------------------------------------------------------------
    # Size and inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of items in the heap. O(1)."""
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        """Iterate in storage order, which is **not** sorted order.

        Useful for inspection; use repeated extraction for sorted output.
        """
        return iter(self._data)

    def __contains__(self, value: object) -> bool:
        """Membership test by linear scan.

        A heap keeps no search structure - only the parent-before-child
        rule - so membership is O(n). It is provided for completeness and
        for cross-checking against other structures, not for hot paths.
        """
        return value in self._data

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def is_empty(self) -> bool:
        """Return True if the heap holds no items.

        Time Complexity:
            O(1).

        Examples:
            >>> MinHeap().is_empty()
            True
            >>> MinHeap([3]).is_empty()
            False
        """
        return not self._data

    def peek(self) -> T:
        """Return the top item without removing it.

        Returns:
            The smallest item of a :class:`MinHeap`, or the largest of a
            :class:`MaxHeap`.

        Raises:
            IndexError: If the heap is empty.

        Time Complexity:
            O(1) - the top item is always at index 0.

        Examples:
            >>> MinHeap([5, 1, 3]).peek()
            1
            >>> MaxHeap([5, 1, 3]).peek()
            5
            >>> MinHeap().peek()
            Traceback (most recent call last):
                ...
            IndexError: peek from an empty heap
        """
        if not self._data:
            raise IndexError("peek from an empty heap")
        return self._data[0]

    def to_list(self) -> List[T]:
        """Return a copy of the underlying array, in storage order.

        Examples:
            >>> MinHeap([3, 1, 2]).to_list()
            [1, 3, 2]
        """
        return list(self._data)

    def is_valid(self) -> bool:
        """Check the heap property across the whole array.

        Intended for debugging and tests: returns True only if no item
        comes before its own parent.

        Time Complexity:
            O(n).

        Examples:
            >>> MinHeap([9, 4, 7, 1]).is_valid()
            True
        """
        data, before = self._data, self._before
        return all(
            not before(data[child], data[(child - 1) >> 1])
            for child in range(1, len(data))
        )

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------
    def insert(self, value: T) -> None:
        """Add an item, restoring the heap property by sifting it up.

        The item is appended as the last leaf, then swapped upward while it
        belongs above its parent.

        Args:
            value: Item to add. Must be comparable with the items already
                in the heap.

        Raises:
            TypeError: If ``value`` cannot be compared with existing items.

        Time Complexity:
            O(log n) worst case - one comparison per level of a tree of
            height floor(log2 n). O(1) expected on random input, because a
            random new item is usually larger than its parent in a min-heap
            and stops after a level or two.

        Examples:
            >>> heap = MinHeap()
            >>> for value in [5, 2, 8, 1]:
            ...     heap.insert(value)
            >>> heap.peek(), len(heap)
            (1, 4)
        """
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def heapify(self, items: Optional[Iterable[T]] = None) -> None:
        """Build a valid heap in place in O(n), bottom-up.

        With ``items``, the heap's contents are replaced by them; without,
        the current array is re-heapified (useful after bulk edits).

        Every internal node, from the last one back to the root, is sifted
        down. A node at height h can sink at most h levels, and only about
        n / 2^(h+1) nodes have height h, so the total work is
        sum over h of h * n / 2^(h+1), which is at most n. That is why this
        is O(n) while n successive inserts are O(n log n).

        Args:
            items: Optional iterable to build the heap from.

        Time Complexity:
            O(n).

        Examples:
            >>> heap = MinHeap()
            >>> heap.heapify([9, 4, 7, 1, 8, 2])
            >>> heap.is_valid(), heap.peek()
            (True, 1)
        """
        if items is not None:
            self._data = list(items)
        for index in range(len(self._data) // 2 - 1, -1, -1):
            self._sift_down(index)

    def _extract(self) -> T:
        """Remove and return the top item. Shared by both public extracts.

        The last leaf replaces the root and sifts down. Popping from the end
        of the list first keeps every step O(1) apart from the sift.
        """
        data = self._data
        if not data:
            raise IndexError("extract from an empty heap")
        last = data.pop()
        if not data:
            return last
        top = data[0]
        data[0] = last
        self._sift_down(0)
        return top

    # ------------------------------------------------------------------
    # Sifting
    # ------------------------------------------------------------------
    # Both sifts move a "hole" rather than swapping pairs: the moving item is
    # held in a local, each displaced item is shifted one step, and the held
    # item is written once at its final position. That halves the writes of
    # the swap-based textbook version without changing the algorithm.

    def _sift_up(self, index: int) -> None:
        """Move the item at ``index`` up until its parent comes before it."""
        data, before = self._data, self._before
        item = data[index]
        while index > 0:
            parent = (index - 1) >> 1
            if before(item, data[parent]):
                data[index] = data[parent]
                index = parent
            else:
                break
        data[index] = item

    def _sift_down(self, index: int) -> None:
        """Move the item at ``index`` down until it comes before its children."""
        data, before = self._data, self._before
        size = len(data)
        item = data[index]
        child = 2 * index + 1
        while child < size:
            right = child + 1
            if right < size and before(data[right], data[child]):
                child = right
            if before(data[child], item):
                data[index] = data[child]
                index = child
                child = 2 * index + 1
            else:
                break
        data[index] = item


class MinHeap(_BinaryHeap[T]):
    """Binary min-heap: the smallest item is always on top.

    Args:
        items: Optional initial contents, built in O(n).

    Examples:
        >>> heap = MinHeap([7, 2, 9, 4])
        >>> [heap.extract_min() for _ in range(len(heap))]
        [2, 4, 7, 9]
    """

    __slots__ = ()

    def __init__(self, items: Optional[Iterable[T]] = None) -> None:
        super().__init__(operator.lt, items)

    def extract_min(self) -> T:
        """Remove and return the smallest item.

        Returns:
            The smallest item in the heap.

        Raises:
            IndexError: If the heap is empty.

        Time Complexity:
            O(log n) - and Theta(log n) in practice, because the last leaf
            that replaces the root is typically large and sinks nearly to
            the bottom.

        Examples:
            >>> heap = MinHeap([3, 1, 2])
            >>> heap.extract_min(), heap.extract_min(), heap.extract_min()
            (1, 2, 3)
            >>> heap.extract_min()
            Traceback (most recent call last):
                ...
            IndexError: extract from an empty heap
        """
        return self._extract()


class MaxHeap(_BinaryHeap[T]):
    """Binary max-heap: the largest item is always on top.

    Args:
        items: Optional initial contents, built in O(n).

    Examples:
        >>> heap = MaxHeap([7, 2, 9, 4])
        >>> [heap.extract_max() for _ in range(len(heap))]
        [9, 7, 4, 2]
    """

    __slots__ = ()

    def __init__(self, items: Optional[Iterable[T]] = None) -> None:
        super().__init__(operator.gt, items)

    def extract_max(self) -> T:
        """Remove and return the largest item.

        Returns:
            The largest item in the heap.

        Raises:
            IndexError: If the heap is empty.

        Time Complexity:
            O(log n).

        Examples:
            >>> heap = MaxHeap([3, 1, 2])
            >>> heap.extract_max(), heap.extract_max(), heap.extract_max()
            (3, 2, 1)
        """
        return self._extract()


class PriorityQueue(Generic[T]):
    """A stable priority queue that delegates to :class:`MinHeap` or :class:`MaxHeap`.

    Every entry is stored in the heap as a ``(priority, order, item)`` tuple,
    where ``order`` comes from a monotonically increasing counter. Tuples
    compare field by field, so two entries of equal priority are separated
    by ``order`` and **the items themselves are never compared**. That does
    two jobs:

    * It makes the queue *stable*: equal priorities come out in the order
      they went in.
    * It lets items be anything. Without the counter, a tie on priority
      would fall through to comparing the items, and raise ``TypeError`` for
      any type that is not orderable - dicts, most custom objects.

    Args:
        highest_first: If False (the default), the lowest priority value is
            served first, as with ``heapq``. If True, the highest is.

    Examples:
        >>> queue = PriorityQueue()
        >>> queue.push("write report", priority=2)
        >>> queue.push("fix bug", priority=1)
        >>> queue.push("run benchmark", priority=2)
        >>> [queue.pop() for _ in range(len(queue))]
        ['fix bug', 'write report', 'run benchmark']

        Items need not be comparable - only priorities are:

        >>> queue = PriorityQueue()
        >>> queue.push({"job": "a"}, priority=1)
        >>> queue.push({"job": "b"}, priority=1)
        >>> queue.pop()
        {'job': 'a'}
    """

    __slots__ = ("_heap", "_extract", "_counter", "_highest_first")

    def __init__(self, highest_first: bool = False) -> None:
        self._highest_first = highest_first
        if highest_first:
            heap: _BinaryHeap = MaxHeap()
            self._extract: Callable[[], Tuple[Any, int, T]] = heap.extract_max
        else:
            heap = MinHeap()
            self._extract = heap.extract_min
        self._heap = heap
        self._counter = itertools.count()

    def __len__(self) -> int:
        """Number of queued items. O(1)."""
        return len(self._heap)

    def __repr__(self) -> str:
        order = "highest_first" if self._highest_first else "lowest_first"
        return f"PriorityQueue({order}, size={len(self)})"

    def is_empty(self) -> bool:
        """Return True if nothing is queued.

        Examples:
            >>> PriorityQueue().is_empty()
            True
        """
        return self._heap.is_empty()

    def push(self, item: T, priority: Any) -> None:
        """Queue ``item`` at ``priority``.

        Args:
            item: Anything. Never compared.
            priority: Must be orderable against the other priorities.

        Time Complexity:
            O(log n).

        Examples:
            >>> queue = PriorityQueue(highest_first=True)
            >>> queue.push("low", 1)
            >>> queue.push("high", 9)
            >>> queue.pop()
            'high'
        """
        order = next(self._counter)
        # Under highest_first the heap serves the *largest* tuple, so the
        # tiebreaker is negated to keep earlier entries ahead of later ones.
        tiebreak = -order if self._highest_first else order
        self._heap.insert((priority, tiebreak, item))

    def pop(self) -> T:
        """Remove and return the item with the most urgent priority.

        Ties are broken first-in, first-out.

        Raises:
            IndexError: If the queue is empty.

        Time Complexity:
            O(log n).

        Examples:
            >>> PriorityQueue().pop()
            Traceback (most recent call last):
                ...
            IndexError: pop from an empty priority queue
        """
        return self.pop_with_priority()[1]

    def pop_with_priority(self) -> Tuple[Any, T]:
        """Remove and return ``(priority, item)`` for the most urgent entry.

        Raises:
            IndexError: If the queue is empty.

        Time Complexity:
            O(log n).

        Examples:
            >>> queue = PriorityQueue()
            >>> queue.push("task", 3)
            >>> queue.pop_with_priority()
            (3, 'task')
        """
        if self._heap.is_empty():
            raise IndexError("pop from an empty priority queue")
        priority, _order, item = self._extract()
        return priority, item

    def peek(self) -> T:
        """Return the most urgent item without removing it.

        Raises:
            IndexError: If the queue is empty.

        Time Complexity:
            O(1).

        Examples:
            >>> queue = PriorityQueue()
            >>> queue.push("only", 5)
            >>> queue.peek(), len(queue)
            ('only', 1)
        """
        if self._heap.is_empty():
            raise IndexError("peek from an empty priority queue")
        return self._heap.peek()[2]

    def peek_priority(self) -> Any:
        """Return the priority of the most urgent item without removing it.

        Raises:
            IndexError: If the queue is empty.

        Examples:
            >>> queue = PriorityQueue()
            >>> queue.push("x", 4)
            >>> queue.peek_priority()
            4
        """
        if self._heap.is_empty():
            raise IndexError("peek from an empty priority queue")
        return self._heap.peek()[0]
