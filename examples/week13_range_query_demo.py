#!/usr/bin/env python3
"""Runnable demonstration of the Week 13 range-query engine, with time travel.

    python examples/week13_range_query_demo.py

The engine answers range sums with a range Fenwick tree, range minima and
maxima with a lazy segment tree, and questions about any earlier version
with a persistent segment tree. Ranges here are INCLUSIVE ``[l, r]``, as the
instructions phrase them.

Three steps on the spec 9.1 array ``[5, 2, 8, 1, 9, 3, 7, 4, 6, 0]``:

1. Queries on version 0, with the spec 9.7 anchor ``sum(2, 6) = 28``.
2. A few updates, each making a new version: ``range_add(3, 7, 10)`` (the
   anchors ``sum(0, 9) = 95`` and ``max(2, 5) = 19``), then ``set(0, 50)``
   and ``add(9, -5)``.
3. Time travel: the same queries on every version, including the anchor
   ``sum_at(0, 0, 9) = 45`` after all the updates.

Every printed value is checked against a plain Python list kept beside the
engine (one saved snapshot per version, queried with the built-in ``sum``,
``min`` and ``max`` on slices), and every anchor is asserted, so the script
fails rather than printing something untrue. The array is a small synthetic
example. It finishes in well under a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.advds.applications.range_queries import RangeQueryEngine  # noqa: E402

WIDTH = 74

#: The spec 9.1 array.
A = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]


def banner(number: int, title: str) -> None:
    """Print a numbered section heading."""
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    """Print one labelled result line."""
    print(f"  {label:<40} {value}")


def check_against(eng: RangeQueryEngine, snapshots: List[List[int]]) -> None:
    """Assert every inclusive range of every version against the saved snapshots."""
    n = len(snapshots[0])
    current = snapshots[-1]
    for l in range(n):
        for r in range(l, n):
            assert eng.sum(l, r) == sum(current[l : r + 1])
            assert eng.min(l, r) == min(current[l : r + 1])
            assert eng.max(l, r) == max(current[l : r + 1])
            for v, snap in enumerate(snapshots):
                assert eng.sum_at(v, l, r) == sum(snap[l : r + 1])
                assert eng.min_at(v, l, r) == min(snap[l : r + 1])
                assert eng.max_at(v, l, r) == max(snap[l : r + 1])


def main() -> int:
    """Run the three steps, asserting every anchor; return 0."""
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 13 range-query demonstration")
    print("Sum by range Fenwick, min/max by segment tree, time travel by persistence")
    print("=" * WIDTH)
    print("  All ranges are INCLUSIVE [l, r]. Every anchor below is asserted.")

    eng = RangeQueryEngine(A)
    plain = list(A)
    snapshots = [list(plain)]

    banner(1, "Version 0: the spec 9.1 array")
    show("values", plain)
    s = eng.sum(2, 6)
    assert s == 28 == sum(plain[2:7])
    show("sum(2, 6)  [anchor 28]", s)
    show("min(1, 4), max(0, 9)", (eng.min(1, 4), eng.max(0, 9)))
    assert (eng.min(1, 4), eng.max(0, 9)) == (1, 9)

    banner(2, "Updates, one new version each")
    v = eng.range_add(3, 7, 10)
    for i in range(3, 8):
        plain[i] += 10
    snapshots.append(list(plain))
    show("range_add(3, 7, 10) -> version", v)
    show("values", plain)
    s, mx = eng.sum(0, 9), eng.max(2, 5)
    assert (s, mx) == (95, 19) == (sum(plain), max(plain[2:6]))
    show("sum(0, 9)  [anchor 95]", s)
    show("max(2, 5)  [anchor 19]", mx)

    v = eng.set(0, 50)
    plain[0] = 50
    snapshots.append(list(plain))
    show("set(0, 50) -> version", v)

    v = eng.add(9, -5)
    plain[9] -= 5
    snapshots.append(list(plain))
    show("add(9, -5) -> version", v)
    show("values", plain)
    show("sum(0, 9), min(0, 9), max(0, 9)", (eng.sum(0, 9), eng.min(0, 9), eng.max(0, 9)))
    assert (eng.sum(0, 9), eng.min(0, 9), eng.max(0, 9)) == (sum(plain), min(plain), max(plain))
    assert eng.current_version() == 3

    banner(3, "Time travel through every version")
    print(f"  {'version':<9}{'sum_at(v, 0, 9)':>17}{'min_at(v, 0, 9)':>17}{'max_at(v, 2, 5)':>17}")
    for version, snap in enumerate(snapshots):
        row = (eng.sum_at(version, 0, 9), eng.min_at(version, 0, 9), eng.max_at(version, 2, 5))
        assert row == (sum(snap), min(snap), max(snap[2:6]))
        print(f"  {version:<9}{row[0]:>17}{row[1]:>17}{row[2]:>17}")
    old = eng.sum_at(0, 0, 9)
    assert old == 45
    show("sum_at(0, 0, 9)  [anchor 45]", old)

    check_against(eng, snapshots)
    show("all inclusive ranges, all versions", "match the saved snapshots")

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
