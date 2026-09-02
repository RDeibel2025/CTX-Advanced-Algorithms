#!/usr/bin/env python3
"""Runnable demonstration of the Week 2 divide-and-conquer sorts.

    python examples/week2_demo.py

Walks through what merge sort and quicksort do and what they promise, on
inputs small enough to read. Six sections:

1. Both algorithms on a small list, with the result shown next to Python's
   own ``sorted`` so the output can be checked by eye.
2. The edge cases - empty, single element, two elements, all equal - which
   the caller should never have to special-case.
3. The non-destructive contract: the caller's list is unchanged and the
   returned list is a genuinely separate object.
4. Any comparable element type, not just integers.
5. Three-way partitioning against two-way on duplicate-heavy input, timed,
   which is where the two differ.
6. Stability: merge sort preserves the order of equal keys, quicksort does
   not promise to.

Every claim printed is also asserted, so the script fails loudly rather
than printing something untrue. It exits 0 when everything holds.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter
from typing import Any, Callable, List

# Allow `python examples/week2_demo.py` from the repository root without the
# package being installed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.sorting import bubble_sort, insertion_sort, selection_sort  # noqa: E402
from src.sorting.merge_sort import merge, merge_sort  # noqa: E402
from src.sorting.quick_sort import (  # noqa: E402
    INSERTION_SORT_CUTOFF,
    quick_sort,
)
from src.utils.testing_helpers import (  # noqa: E402
    extract_tags,
    make_stability_records,
)

#: Fixed so the demonstration prints the same thing every time it is run.
SEED = 2026

WIDTH = 74


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    """Print one labelled line, truncating anything too wide to read."""
    text = repr(value)
    if len(text) > WIDTH - 24:
        text = text[: WIDTH - 27] + "..."
    print(f"  {label:<20} {text}")


# ----------------------------------------------------------------------
def section_basics() -> None:
    banner(1, "Sorting a small list")

    data = [38, 27, 43, 3, 9, 82, 10]
    show("input", data)
    print()

    for name, algorithm in (
        ("merge_sort", merge_sort),
        ("quick_sort", lambda values: quick_sort(values, seed=SEED)),
    ):
        result = algorithm(data)
        assert result == sorted(data), f"{name} disagrees with sorted()"
        show(name, result)

    show("sorted() reference", sorted(data))
    print("\n  Both agree with Python's own sorted().")

    # The merge helper is a separate, separately usable function.
    print("\n  merge() combines two already-sorted runs in linear time:")
    left, right = [3, 9, 27, 38], [10, 43, 82]
    show("left", left)
    show("right", right)
    merged = merge(left, right)
    assert merged == sorted(left + right)
    show("merge(left, right)", merged)


def section_edge_cases() -> None:
    banner(2, "Edge cases the caller should not have to handle")

    cases = {
        "empty": [],
        "single element": [42],
        "two, in order": [1, 2],
        "two, reversed": [2, 1],
        "all equal": [7, 7, 7, 7, 7],
        "already sorted": [1, 2, 3, 4, 5],
        "reverse sorted": [5, 4, 3, 2, 1],
    }
    algorithms = {
        "merge_sort": merge_sort,
        "quick_sort": lambda values: quick_sort(values, seed=SEED),
        "quick_sort 3-way": lambda values: quick_sort(
            values, seed=SEED, three_way=True
        ),
    }

    header = f"  {'case':<16}{'input':<20}" + "".join(
        f"{name:<20}" for name in algorithms
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for case, data in cases.items():
        cells = ""
        for algorithm in algorithms.values():
            result = algorithm(data)
            assert result == sorted(data), f"{case} failed"
            cells += f"{repr(result):<20}"
        print(f"  {case:<16}{repr(data):<20}{cells}")

    print("\n  No special-casing by the caller; no crashes on empty input.")


def section_non_destructive() -> None:
    banner(3, "The caller's list is never modified")

    original = [5, 3, 8, 1, 9, 2]
    snapshot = list(original)
    show("before sorting", original)

    result = merge_sort(original)
    show("merge_sort returns", result)
    show("original afterwards", original)

    assert original == snapshot, "merge_sort mutated its argument"
    assert result is not original, "merge_sort returned the caller's own list"
    print("\n  The input is byte-for-byte unchanged, and the result is a")
    print("  separate object - writing to it cannot affect the original:")

    result[0] = 999
    show("result[0] = 999", result)
    show("original still", original)
    assert original == snapshot

    # Quicksort partitions in place, but only ever inside its own copy.
    quick_original = [5, 3, 8, 1, 9, 2]
    quick_snapshot = list(quick_original)
    quick_result = quick_sort(quick_original, seed=SEED)
    assert quick_original == quick_snapshot, "quick_sort mutated its argument"
    assert quick_result is not quick_original
    print("\n  quick_sort holds to the same contract, even though its")
    print("  partitioning genuinely happens in place - on a copy it owns.")


def section_element_types() -> None:
    banner(4, "Any comparable element type, not just integers")

    cases = [
        ("integers", [3, 1, 2]),
        ("negatives", [-5, -1, -9, -3]),
        ("floats", [3.5, -0.5, 2.25, 0.0]),
        ("strings", ["pear", "apple", "fig", "banana"]),
        ("tuples", [(2, "b"), (1, "z"), (2, "a")]),
    ]
    for label, data in cases:
        result = merge_sort(data)
        assert result == sorted(data)
        assert quick_sort(data, seed=SEED) == sorted(data)
        show(label, result)

    print("\n  The algorithms only ever compare elements with < and >, so")
    print("  anything orderable works.")


def section_three_way() -> None:
    banner(5, "Three-way partitioning on duplicate-heavy input")

    print(f"  quick_sort takes a three_way flag. It splits into < = > rather")
    print(f"  than < >=, so every element equal to the pivot is finished in")
    print(f"  one pass and never looked at again.\n")

    small = [5, 1, 5, 3, 5, 2, 5, 4, 5, 0, 5, 3, 1]
    show("input", small)
    two_way = quick_sort(small, seed=SEED, three_way=False)
    three_way = quick_sort(small, seed=SEED, three_way=True)
    show("two-way", two_way)
    show("three-way", three_way)
    assert two_way == three_way == sorted(small), "the two schemes disagree"
    print("\n  Identical output. The difference is how much work each does.")

    # Timed on input with only three distinct values, which is where the
    # equal-element handling actually matters.
    size = 20_000
    rng = random.Random(SEED)
    duplicate_heavy = [rng.randrange(3) for _ in range(size)]

    def timed(three_way_flag: bool) -> float:
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            result = quick_sort(duplicate_heavy, seed=SEED, three_way=three_way_flag)
            best = min(best, time.perf_counter() - start)
            assert result == sorted(duplicate_heavy)
        return best

    two_way_time = timed(False)
    three_way_time = timed(True)

    print(f"\n  n = {size:,}, only 3 distinct values, best of 3 runs:")
    print(f"    two-way   {two_way_time:.5f} s")
    print(f"    three-way {three_way_time:.5f} s")
    print(f"    three-way is {two_way_time / three_way_time:.1f}x faster here")
    assert three_way_time < two_way_time, "three-way should win on this input"

    print("\n  It is a trade, not a free improvement: on data where values")
    print("  are mostly distinct, three-way partitioning is the slower of")
    print("  the two. See analysis/week2_report.md section 5.")


def section_stability() -> None:
    banner(6, "Stability: merge sort preserves the order of equal keys")

    print("  Records carry a key and a tag. They are compared by key only,")
    print("  so the tags show what happened to elements that tied.\n")

    # Eleven elements, not fewer. Quicksort hands any range of
    # INSERTION_SORT_CUTOFF (10) or fewer elements straight to insertion
    # sort without partitioning at all, and insertion sort is stable - so
    # below this size quicksort cannot show its instability even in
    # principle. This is the smallest input on which it does.
    keys = [1, 0, 2, 1, 1, 2, 0, 1, 2, 2, 2]
    assert len(keys) == INSERTION_SORT_CUTOFF + 1
    records = make_stability_records(keys)

    show("keys", keys)
    show("tags in", list(range(len(keys))))
    print()

    stable_tags = [tag for _key, tag in sorted(zip(keys, range(len(keys))))]

    merged = merge_sort(records)
    merged_tags = extract_tags(merged)
    assert [r.key for r in merged] == sorted(keys)
    assert merged_tags == stable_tags, "merge sort is not stable"
    show("merge_sort keys", [r.key for r in merged])
    show("merge_sort tags", merged_tags)
    print("    -> equal keys kept their input order: merge sort is stable.")

    quick = quick_sort(records, seed=SEED)
    quick_tags = extract_tags(quick)
    # Sorting by key correctly is required of every algorithm.
    assert [r.key for r in quick] == sorted(keys), "quick_sort mis-sorted"
    print()
    show("quick_sort keys", [r.key for r in quick])
    show("quick_sort tags", quick_tags)

    if quick_tags == stable_tags:
        print("    -> on this input quicksort happened to preserve the ties,")
        print("       but it makes no such promise.")
    else:
        moved = [
            (stable, actual)
            for stable, actual in zip(stable_tags, quick_tags)
            if stable != actual
        ]
        actual = [a for _s, a in moved]
        wanted = [s for s, _a in moved]
        print(f"    -> {len(moved)} positions differ from the stable order:")
        print(f"       quicksort put {actual}, a stable sort would put {wanted}.")
        print("       Both are correctly sorted by key. Quicksort simply makes")
        print("       no promise about ties, because partitioning swaps")
        print("       non-adjacent elements.")

    print()
    print(f"  Note the size: {len(keys)} elements. Quicksort finishes any range of")
    print(f"  {INSERTION_SORT_CUTOFF} or fewer with insertion sort, which is stable, so below")
    print(f"  {INSERTION_SORT_CUTOFF + 1} elements it never partitions and cannot reorder a tie.")


def section_cross_check() -> None:
    banner(7, "All five algorithms agree")

    rng = random.Random(SEED)
    data = [rng.randint(-50, 50) for _ in range(200)]
    algorithms = {
        "bubble_sort": bubble_sort,
        "selection_sort": selection_sort,
        "insertion_sort": insertion_sort,
        "merge_sort": merge_sort,
        "quick_sort": lambda values: quick_sort(values, seed=SEED),
    }

    expected = sorted(data)
    outputs = {}
    for name, algorithm in algorithms.items():
        result = algorithm(data)
        assert result == expected, f"{name} disagrees"
        assert Counter(result) == Counter(data), f"{name} lost an element"
        outputs[name] = result

    print(f"  200 random integers, sorted by each of the five algorithms.")
    print(f"  Every output is identical and matches sorted():\n")
    for name in algorithms:
        print(f"    {name:<16} first 8: {outputs[name][:8]}")

    assert len({tuple(v) for v in outputs.values()}) == 1
    print("\n  The Week 1 and Week 2 implementations are interchangeable.")
    print("  tests/test_sorting_comparison.py checks this exhaustively.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 2 demonstration")
    print("Merge sort and quicksort - Robert Deibel")
    print("=" * WIDTH)
    print(f"  random seed              {SEED}")
    print(f"  INSERTION_SORT_CUTOFF    {INSERTION_SORT_CUTOFF}")
    print("  Every line printed below is also asserted; this script exits")
    print("  non-zero if any demonstrated claim fails to hold.")

    section_basics()
    section_edge_cases()
    section_non_destructive()
    section_element_types()
    section_three_way()
    section_stability()
    section_cross_check()

    print()
    print("=" * WIDTH)
    print("All demonstrations held. See also:")
    print("  analysis/week2_report.md        the performance study")
    print("  analysis/week2_recurrences.md   Master Theorem solutions")
    print("  benchmarks/week2_performance.py the full benchmark")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
