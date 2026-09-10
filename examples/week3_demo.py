#!/usr/bin/env python3
"""Runnable demonstration of the Week 3 data structures.

    python examples/week3_demo.py

Walks through the three structures on inputs small enough to read, and
shows the one property that makes each of them worth having:

1. **Heaps** - the smallest (or largest) item is always one step away, and
   a heap can be built from a list in linear time.
2. **Priority queue** - a stable queue built on the heap, which never needs
   to compare the items it carries.
3. **AVL tree** - sorted input, the worst case for a plain binary search
   tree, still produces a perfectly balanced tree.
4. **Hash table** - both collision strategies, a live look at a linear
   probing run, and why deletion there needs tombstones.
5. **Agreement** - the same keys through all three structures.

Every claim printed is also asserted, so the script fails loudly rather
than printing something untrue. It exits 0 when everything holds, and
takes well under a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Allow `python examples/week3_demo.py` from the repository root without the
# package being installed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.structures.avl_tree import AVLTree, avl_max_height, perfect_height  # noqa: E402
from src.structures.hash_table import (  # noqa: E402
    CHAINING,
    LINEAR_PROBING,
    HashTable,
    _EMPTY,
    _TOMBSTONE,
)
from src.structures.heap import MaxHeap, MinHeap, PriorityQueue  # noqa: E402

WIDTH = 74


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<26} {value}")


class Counted:
    """An integer wrapper that counts the comparisons made on it."""

    total = 0
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    def __lt__(self, other: "Counted") -> bool:
        Counted.total += 1
        return self.value < other.value


# ----------------------------------------------------------------------
def section_heaps() -> None:
    banner(1, "Binary heaps: the top item is always one step away")

    values = [42, 7, 19, 3, 25, 11, 3]
    min_heap = MinHeap()
    for value in values:
        min_heap.insert(value)
    show("inserted", values)
    show("min-heap array", min_heap.to_list())
    show("peek (smallest)", min_heap.peek())
    drained = [min_heap.extract_min() for _ in range(len(min_heap))]
    show("extract_min, repeatedly", drained)
    assert drained == sorted(values)

    max_heap = MaxHeap(values)                     # built with heapify
    show("max-heap via heapify", max_heap.to_list())
    drained = [max_heap.extract_max() for _ in range(len(max_heap))]
    show("extract_max, repeatedly", drained)
    assert drained == sorted(values, reverse=True)

    print("\n  Building a heap from a list: heapify versus inserting one by one.")
    print("  Counting comparisons, on reverse-ordered input:")
    for size in (1_000, 8_000):
        values = list(range(size, 0, -1))
        Counted.total = 0
        MinHeap([Counted(v) for v in values])
        build = Counted.total
        Counted.total = 0
        heap = MinHeap()
        for value in values:
            heap.insert(Counted(value))
        repeated = Counted.total
        print(f"    n = {size:>5,}   heapify {build:>7,} ({build / size:.2f} per item)"
              f"   inserts {repeated:>8,} ({repeated / size:.2f} per item)")
        assert build <= 2 * size < repeated
    print("  heapify stays at ~2 comparisons per item - O(n). Inserting grows")
    print("  with log n - O(n log n).")


def section_priority_queue() -> None:
    banner(2, "Priority queue: stable, and it never compares the items")

    queue = PriorityQueue()
    jobs = [
        ({"job": "write report"}, 2),
        ({"job": "fix failing test"}, 1),
        ({"job": "run benchmark"}, 2),
        ({"job": "push to GitHub"}, 3),
        ({"job": "update README"}, 2),
    ]
    for payload, priority in jobs:
        queue.push(payload, priority)
    print("  Jobs are dicts - Python cannot order dicts - pushed with ties:")
    for payload, priority in jobs:
        print(f"    priority {priority}  {payload['job']}")
    served = [queue.pop_with_priority() for _ in range(len(queue))]
    print("\n  Served in order:")
    for priority, payload in served:
        print(f"    priority {priority}  {payload['job']}")
    assert [p for p, _ in served] == [1, 2, 2, 2, 3]
    ties = [payload["job"] for p, payload in served if p == 2]
    assert ties == ["write report", "run benchmark", "update README"]
    print("\n  The three priority-2 jobs kept their arrival order, and no dict")
    print("  was ever compared: a counter breaks every tie first.")


def section_avl() -> None:
    banner(3, "AVL tree: sorted input stays balanced")

    tree = AVLTree()
    for key in range(1, 16):
        tree.insert(key, f"value-{key}")
    show("inserted, in order", "1, 2, 3, ... 15")
    show("tree height", tree.height())
    show("plain BST would be", "15 (a linked list)")
    show("perfect height for 15", perfect_height(15))
    show("rotations performed", tree.rotation_count)
    show("root key", tree.root.key)
    assert tree.height() == perfect_height(15) == 4
    tree.validate()

    show("search(11)", tree.search(11))
    show("search(99)", tree.search(99))
    tree.delete(8)                                 # the root: two children
    show("after delete(8)", tree.in_order())
    show("new root", tree.root.key)
    assert 8 not in tree and tree.in_order() == [k for k in range(1, 16) if k != 8]
    tree.validate()

    big = AVLTree(range(100_000))
    print(f"\n  100,000 sorted keys: height {big.height()} "
          f"(perfect {perfect_height(100_000)}, AVL bound {avl_max_height(100_000)})")
    assert big.height() == perfect_height(100_000)


def section_hash_table() -> None:
    banner(4, "Hash table: two collision strategies, and tombstones")

    words = ["heap", "tree", "hash", "queue", "graph", "array", "stack", "trie"]
    for strategy in (CHAINING, LINEAR_PROBING):
        table = HashTable(strategy=strategy)
        for word in words * 3:                     # repeats update, not duplicate
            table.insert(word, len(word))
        show(f"{strategy}", f"{len(table)} keys, capacity {table.capacity}, "
             f"load factor {table.load_factor:.2f}, rehashes {table.rehash_count}")
        assert len(table) == len(words)
        assert all(table.get(w) == len(w) for w in words)

    print("\n  Growth: inserting 1,000 keys into a chaining table.")
    table = HashTable()
    for key in range(1000):
        table.insert(key)
    show("capacity", table.capacity)
    show("rebuilds", table.rehash_count)
    show("entries moved in total", f"{table.rehash_moves:,}  "
         f"({table.rehash_moves / 1000:.2f} per insert - amortised O(1))")
    assert table.rehash_moves < 2 * 1000

    print("\n  Linear probing, up close. Keys 0, 16, 32 and 48 all hash to slot")
    print("  0 of a 16-slot table, so they fill slots 0 to 3 in a run:")
    table = HashTable(capacity=16, strategy=LINEAR_PROBING, max_load_factor=0.9)
    for key in (0, 16, 32, 48):
        table.insert(key, f"v{key}")
    show("slots 0-4", table._keys[:5])
    show("probes to find 48", table.probe_count(48))

    table.delete(16)
    print("\n  Delete 16, from the middle of the run:")
    show("slots 0-4", table._keys[:5])
    show("get(32)", table.get(32))
    show("get(48)", table.get(48))
    assert table._keys[1] is _TOMBSTONE
    assert table.get(32) == "v32" and table.get(48) == "v48"

    naive = HashTable(capacity=16, strategy=LINEAR_PROBING, max_load_factor=0.9)
    for key in (0, 16, 32, 48):
        naive.insert(key, f"v{key}")
    naive._keys[1] = _EMPTY                        # what a naive delete would do
    print("\n  Had the slot simply been emptied instead:")
    show("get(32)", naive.get(32, "MISSING"))
    show("get(48)", naive.get(48, "MISSING"))
    assert naive.get(32, "MISSING") == "MISSING"
    print("  Both keys are still in the table, and neither can be found. The")
    print("  tombstone tells a lookup to keep going.")


def section_agreement() -> None:
    banner(5, "Same keys, all three structures")

    keys = [50, 20, 80, 10, 30, 70, 90, 20, 60]
    heap = MinHeap(keys)
    tree = AVLTree(keys)
    tables = [HashTable(strategy=s) for s in (CHAINING, LINEAR_PROBING)]
    for table in tables:
        for key in keys:
            table.insert(key)

    distinct = sorted(set(keys))
    from_heap = [heap.extract_min() for _ in range(len(heap))]
    show("keys (20 twice)", keys)
    show("heap drain", from_heap)
    show("AVL in-order", tree.in_order())
    show("chaining, sorted", sorted(tables[0].keys()))
    show("probing, sorted", sorted(tables[1].keys()))
    assert from_heap == sorted(keys)
    assert tree.in_order() == distinct
    assert all(sorted(t.keys()) == distinct for t in tables)
    for absent in (0, 55, 100):
        assert absent not in tree and all(absent not in t for t in tables)
    print("\n  The heap keeps the duplicate 20; the tree and tables keep one")
    print("  of each. All agree on membership and on order.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 3 demonstration")
    print("Heaps, AVL trees and hash tables - Robert Deibel")
    print("=" * WIDTH)
    print("  Every line below is also asserted; this script exits non-zero if")
    print("  any demonstrated claim fails to hold.")

    section_heaps()
    section_priority_queue()
    section_avl()
    section_hash_table()
    section_agreement()

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
