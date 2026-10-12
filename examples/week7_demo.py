#!/usr/bin/env python3
"""Runnable demonstration of the Week 7 randomized algorithms.

    python examples/week7_demo.py

Five short sections, every random draw seeded so the output is the same on
every run:

1. **QuickSort** - randomized and deterministic comparison counts next to
   the expected 2(n+1)H_n - 4n, and what sorted input does to each.
2. **Selection** - the minimum, median and maximum from randomized
   QuickSelect and from median of medians.
3. **Primality** - Fermat, Miller-Rabin and trial division on a Carmichael
   number, a large prime, and 3,215,031,751, which fools Miller-Rabin with
   the fixed bases 2, 3, 5 and 7 but not with random ones.
4. **Bloom filter** - measured against theoretical false-positive rate.
5. **MinHash** - estimated Jaccard similarity at k = 16 and k = 512 next to
   the true value.

Every claim printed is also asserted, so the script exits non-zero rather
than printing something untrue. It finishes in a few seconds.

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

from src.randomized.bloom_filter import BloomFilter  # noqa: E402
from src.randomized.minhash_similarity import estimate_similarity, jaccard  # noqa: E402
from src.randomized.montecarlo_primality import (  # noqa: E402
    fermat_test,
    is_prime_deterministic,
    miller_rabin,
    miller_rabin_fixed_bases,
)
from src.randomized.randomized_quicksort import (  # noqa: E402
    deterministic_quicksort,
    expected_comparisons,
    randomized_quicksort,
)
from src.randomized.randomized_selection import (  # noqa: E402
    deterministic_select,
    randomized_select,
)
from src.utils.random_utils import random_distinct_list  # noqa: E402

WIDTH = 74


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<34} {value}")


def section_quicksort() -> None:
    banner(1, "QuickSort: random pivots against a fixed rule")
    n = 1_000
    data = random_distinct_list(n, 42)
    rand_sorted, rand_count = randomized_quicksort(data, seed=7, count_comparisons=True)
    det_sorted, det_count = deterministic_quicksort(data, pivot="last", count_comparisons=True)
    assert rand_sorted == det_sorted == sorted(data)
    show("random input, n = 1,000", "")
    show("randomized comparisons", f"{rand_count:,}")
    show("deterministic (last) comparisons", f"{det_count:,}")
    show("expected 2(n+1)H_n - 4n", f"{expected_comparisons(n):,.1f}")

    ordered = list(range(n))
    _, det_sorted_count = deterministic_quicksort(ordered, pivot="last", count_comparisons=True)
    _, rand_sorted_count = randomized_quicksort(ordered, seed=7, count_comparisons=True)
    show("\n  sorted input, deterministic (last)", f"{det_sorted_count:,}  = n(n-1)/2")
    show("sorted input, randomized", f"{rand_sorted_count:,}")
    assert det_sorted_count == n * (n - 1) // 2
    assert rand_sorted_count < det_sorted_count / 10
    print("  Sorted input is the worst case for a fixed last-element pivot. With a")
    print("  random pivot no input is the worst case, because the input cannot")
    print("  anticipate the pivots.")


def section_selection() -> None:
    banner(2, "Selection: randomized QuickSelect against median of medians")
    data = random_distinct_list(10_001, 3)
    ordered = sorted(data)
    for label, k in (("minimum", 1), ("median", 5_001), ("maximum", 10_001)):
        fast, stats = randomized_select(data, k, seed=11, return_stats=True)
        slow = deterministic_select(data, k)
        assert fast == slow == ordered[k - 1]
        show(f"{label} (k = {k:,})",
             f"{fast:,} both ways; QuickSelect took {stats.depth} rounds, "
             f"{stats.comparisons / len(data):.2f}n comparisons")
    print("  QuickSelect is Las Vegas: the answer is always right, and only the")
    print("  number of partition rounds changes from seed to seed.")


def section_primality() -> None:
    banner(3, "Primality: Fermat, Miller-Rabin and trial division")
    for n, label in ((561, "561 = 3 x 11 x 17 (Carmichael)"),
                     (1_000_000_007, "1,000,000,007")):
        f = fermat_test(n, k=10, seed=1)
        m = miller_rabin(n, k=10, seed=1)
        t = is_prime_deterministic(n)
        show(label, f"Fermat {f}, Miller-Rabin {m}, trial division {t}")
        assert m == t
    assert not is_prime_deterministic(561) and is_prime_deterministic(1_000_000_007)

    n = 3_215_031_751
    fixed = miller_rabin_fixed_bases(n, [2, 3, 5, 7])
    random_bases = miller_rabin(n, k=5, seed=1)
    show("\n  3,215,031,751 = 151 x 751 x 28,351", "")
    show("Miller-Rabin, bases 2, 3, 5, 7", f"{fixed}  (fooled)")
    show("Miller-Rabin, 5 random bases", f"{random_bases}  (caught)")
    show("trial division", is_prime_deterministic(n))
    assert fixed and not random_bases and not is_prime_deterministic(n)
    print("  A fixed set of bases can be targeted by whoever chooses the input.")
    print("  Random bases cannot, because the adversary does not know them.")


def section_bloom() -> None:
    banner(4, "Bloom filter: measured against theoretical false positives")
    bloom = BloomFilter.from_capacity(1_000, 0.01, seed=5)
    for i in range(1_000):
        bloom.add(f"member-{i}")
    assert all(f"member-{i}" in bloom for i in range(1_000))
    queries = 20_000
    false_positives = sum(bloom.check(f"query-{q}") for q in range(queries))
    show("bits / hash functions", f"{bloom.m:,} / {bloom.k}")
    show("memory", f"{bloom.memory_bytes():,} bytes for 1,000 items")
    show("measured false-positive rate", f"{false_positives / queries:.4f} "
                                         f"({false_positives} of {queries:,} non-members)")
    show("theoretical false-positive rate", f"{bloom.theoretical_fp_rate():.4f}")
    show("false negatives", "0 - every inserted item checks True")
    assert 0.003 < false_positives / queries < 0.03


def section_minhash() -> None:
    banner(5, "MinHash: estimated against true Jaccard similarity")
    a, b = set(range(0, 100)), set(range(50, 150))
    true_j = jaccard(a, b)
    show("A = {0..99}, B = {50..149}", f"true Jaccard {true_j:.4f}")
    for k in (16, 512):
        estimate = estimate_similarity(a, b, num_perm=k, seed=3)
        show(f"MinHash estimate, k = {k}", f"{estimate:.4f}  (error {abs(estimate - true_j):.4f})")
    assert abs(estimate_similarity(a, b, num_perm=512, seed=3) - true_j) < 0.1
    print("  The standard error is sqrt(J(1 - J) / k): 32 times more hash functions")
    print("  give an error about sqrt(32), or 5.7 times, smaller on average.")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 7 demonstration")
    print("Randomized algorithms - Robert Deibel")
    print("=" * WIDTH)
    print("  Every random draw is seeded and every line below is asserted.")
    section_quicksort()
    section_selection()
    section_primality()
    section_bloom()
    section_minhash()
    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
