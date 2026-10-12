"""Randomized algorithms (Week 7).

Five algorithms that use random choices, in two kinds:

* **Las Vegas** - always correct, with a random running time:
  :mod:`randomized_quicksort` and :mod:`randomized_selection`.
* **Monte Carlo** - bounded work, with an answer that is wrong with a small,
  bounded probability: :mod:`montecarlo_primality` (one-sided error),
  :mod:`bloom_filter` (one-sided: false positives only) and
  :mod:`minhash_similarity` (two-sided estimation error).

Every random draw goes through a seeded ``random.Random`` from
:mod:`src.utils.random_utils`, and item hashing uses blake2b rather than the
per-process-salted built-in ``hash()``, so results are reproducible.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.randomized.bloom_filter import BloomFilter
from src.randomized.minhash_similarity import (
    MERSENNE_61,
    MinHash,
    estimate_similarity,
    jaccard,
    shingles,
)
from src.randomized.montecarlo_primality import (
    count_liars,
    fermat_test,
    is_fermat_liar,
    is_prime_deterministic,
    is_strong_liar,
    miller_rabin,
    miller_rabin_fixed_bases,
)
from src.randomized.randomized_quicksort import (
    comparison_std_asymptotic,
    deterministic_quicksort,
    expected_comparisons,
    randomized_quicksort,
)
from src.randomized.randomized_selection import (
    SelectionStats,
    deterministic_select,
    median,
    randomized_select,
)

__all__ = [
    "randomized_quicksort", "deterministic_quicksort", "expected_comparisons",
    "comparison_std_asymptotic", "randomized_select", "deterministic_select", "median",
    "SelectionStats", "fermat_test", "miller_rabin", "miller_rabin_fixed_bases",
    "is_prime_deterministic", "is_fermat_liar", "is_strong_liar", "count_liars",
    "BloomFilter", "MinHash", "MERSENNE_61", "jaccard", "shingles", "estimate_similarity",
]
