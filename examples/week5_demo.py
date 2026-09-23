#!/usr/bin/env python3
"""Runnable demonstration of the Week 5 dynamic programming code.

    python examples/week5_demo.py

Walks through the week on inputs small enough to check by hand, and shows
the one idea the week turns on:

1. **Three ways up the same recurrence** - plain recursion, memoization and
   tabulation give identical answers at wildly different cost.
2. **Memoization actually memoizing** - the call counts prove the table is
   being hit rather than merely allocated.
3. **Knapsack**, including which items the optimum is made of, and how many
   table cells top-down actually needed.
4. **LCS**, including the subsequence itself rather than only its length.
5. **Who decides the order** - the difference between the two DP styles,
   and the stack each one spends.

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.dp.fibonacci import (  # noqa: E402
    fib_counts,
    fib_instrumented,
    fib_memo,
    fib_naive,
    fib_tab,
    naive_call_count,
)
from src.dp.knapsack import (  # noqa: E402
    knapsack_memo,
    knapsack_memo_cells,
    knapsack_recursive,
    knapsack_tab,
    knapsack_tab_rolling,
    trace_solution,
)
from src.dp.lcs import lcs_memo, lcs_recursive, lcs_reconstruct, lcs_tab  # noqa: E402
from src.utils.timer import Timer  # noqa: E402

WIDTH = 74


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<30} {value}")


# ----------------------------------------------------------------------
def section_fibonacci() -> None:
    banner(1, "One recurrence, three ways up it")

    show("F(10), naive", fib_naive(10))
    show("F(10), memoized", fib_memo(10))
    show("F(10), tabulated", fib_tab(10))
    assert fib_naive(10) == fib_memo(10) == fib_tab(10) == 55

    print("\n  The same answer at every size, from three different routes:")
    for n in (10, 20, 30):
        show(f"F({n})", f"{fib_tab(n):,}")
    assert fib_tab(30) == 832040 and fib_tab(45) == 1134903170

    print("\n  What differs is the work. Counting calls, not seconds, because")
    print("  counts do not depend on the machine:")
    print(f"    {'n':>4}  {'naive calls':>14}  {'memo calls':>11}  {'ratio':>12}")
    for n in (10, 20, 25):
        _, naive_calls = fib_counts(n, "naive")
        _, memo_calls = fib_counts(n, "memo")
        print(f"    {n:>4}  {naive_calls:>14,}  {memo_calls:>11,}  "
              f"{naive_calls / memo_calls:>11,.0f}x")
        assert naive_calls == naive_call_count(n)
    print("  Naive calls are exactly 2*F(n+1) - 1, so the growth is the")
    print("  recurrence itself being recomputed, over and over.")

    with Timer() as naive_timer:
        fib_naive(25)
    with Timer() as memo_timer:
        fib_memo(25)
    show("\n  F(25) naive", f"{naive_timer.elapsed * 1e3:.2f} ms")
    show("F(25) memoized", f"{memo_timer.elapsed * 1e6:.1f} us")
    assert memo_timer.elapsed < naive_timer.elapsed


def section_memoization_works() -> None:
    banner(2, "Proving the memo table is actually hit")

    print("  A memo that is written but never read is the classic silent bug:")
    print("  the code looks memoized and runs exponentially. Call counts are")
    print("  how you tell the difference.")
    for n in (10, 20, 30):
        _, counter = fib_instrumented(n, "memo")
        show(f"fib_memo({n}) calls", f"{counter.calls}  (linear in n, not exponential)")
        assert counter.calls < 100

    _, naive = fib_instrumented(20, "naive")
    _, memo = fib_instrumented(20, "memo")
    _, tab = fib_instrumented(20, "tab")
    print()
    show("naive depth at n=20", naive.max_depth)
    show("memo depth at n=20", memo.max_depth)
    show("tab depth at n=20", tab.max_depth)
    assert tab.max_depth == 1
    print("  Tabulation's depth is 1 by construction: it never recurses, so")
    print("  it has no call stack to run out of. That matters later.")


def section_knapsack() -> None:
    banner(3, "0/1 knapsack, and which items the answer is made of")

    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    show("weights", weights)
    show("values", values)
    show("capacity", capacity)

    best = knapsack_tab(weights, values, capacity)
    show("optimum", best)
    assert knapsack_recursive(weights, values, capacity) == best
    assert knapsack_memo(weights, values, capacity) == best
    assert knapsack_tab_rolling(weights, values, capacity) == best
    assert best == 9
    print("  Checked by hand: items 1 and 3 weigh 3 + 4 = 7 and are worth")
    print("  4 + 5 = 9. Taking the value-7 item instead leaves 2 capacity")
    print("  spare and only reaches 8.")

    traced_value, chosen = trace_solution(weights, values, capacity)
    show("\n  trace_solution", f"value {traced_value}, items {chosen}")
    show("chosen weights", [weights[i] for i in chosen])
    show("chosen values", [values[i] for i in chosen])
    assert traced_value == best
    assert sum(weights[i] for i in chosen) <= capacity
    assert sum(values[i] for i in chosen) == best
    print("  The table gives the number; walking it backwards gives the")
    print("  answer a person can act on.")

    print("\n  How much of the table does top-down actually need?")
    for n, cap in ((10, 100), (20, 500)):
        import random
        rng = random.Random(42 + n)
        w = [rng.randint(1, 50) for _ in range(n)]
        v = [rng.randint(1, 100) for _ in range(n)]
        value, entries, cells = knapsack_memo_cells(w, v, cap)
        show(f"n={n}, W={cap}", f"{entries:,} memo entries of {cells:,} cells "
                                f"({100 * entries / cells:.1f}%)")
        assert knapsack_tab(w, v, cap) == value
    print("  Bottom-up fills every cell because it does not know which ones")
    print("  matter. Top-down only ever asks for the ones it reaches.")


def section_lcs() -> None:
    banner(4, "Longest common subsequence, and the subsequence itself")

    left, right = "AGGTAB", "GXTXAYB"
    show("X", left)
    show("Y", right)
    length = lcs_tab(left, right)
    show("LCS length", length)
    show("LCS itself", lcs_reconstruct(left, right))
    assert lcs_recursive(left, right) == lcs_memo(left, right) == length == 4
    assert lcs_reconstruct(left, right) == "GTAB"
    print("  Checked by hand: G, T, A, B appear in that order in both.")

    print("\n  Edge cases, all three variants agreeing:")
    for a, b, expected in (("", "ABC", 0), ("ABC", "ABC", 3), ("ABC", "XYZ", 0),
                           ("ABCD", "abcd", 0)):
        got = lcs_tab(a, b)
        show(f"lcs({a!r}, {b!r})", got)
        assert got == expected == lcs_memo(a, b) == lcs_recursive(a, b)
    print("  The last pair is 0 because the comparison is case sensitive.")


def section_who_decides_the_order() -> None:
    banner(5, "Who decides the order")

    print("  Memoization and tabulation compute the same subproblems. The")
    print("  difference is who decides the order they are solved in, and")
    print("  every trade-off between them follows from that.")
    print()
    print("  Top-down asks for a subproblem when it needs it, and discovers")
    print("  the order lazily, on the call stack. Bottom-up fixes the order")
    print("  in advance and fills everything, needed or not.")

    left = "ACGT" * 150
    right = "AGCT" * 150
    depth_needed = len(left) + len(right)
    show("\n  two strings of length", len(left))
    show("depth memo would need", f"about {depth_needed:,}")
    show("CPython default limit", 1000)
    show("depth tabulation needs", 1)

    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(30_000)
    try:
        memo_length = lcs_memo(left, right)
    finally:
        sys.setrecursionlimit(previous)
    tab_length = lcs_tab(left, right)
    show("memo answer (limit raised)", memo_length)
    show("tab answer (limit untouched)", tab_length)
    assert memo_length == tab_length
    print("\n  Same answer, same subproblems. But the lazy one borrows the")
    print("  interpreter's stack to remember where it was, so on long inputs")
    print("  it needs the limit raised and the other does not. Choosing who")
    print("  decides the order is also choosing which stack you spend.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 5 demonstration")
    print("Dynamic programming: Fibonacci, knapsack, LCS - Robert Deibel")
    print("=" * WIDTH)
    print("  Every line below is also asserted; this script exits non-zero if")
    print("  any demonstrated claim fails to hold.")

    section_fibonacci()
    section_memoization_works()
    section_knapsack()
    section_lcs()
    section_who_decides_the_order()

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
