"""Core: four exact matchers and the engine that compares them (Week 11).

Five modules. Each matcher differs in what it stores before it searches and
about which string:

* :mod:`kmp` - Knuth-Morris-Pratt, :func:`kmp_search`, stores the LPS array of
  the *pattern* (:func:`compute_lps`) and scans the text once, O(n + m).
  :func:`naive_search` is the baseline, and :class:`MatchStats` counts
  character comparisons.
* :mod:`rabin_karp` - Rabin-Karp, :func:`rabin_karp_search`, stores the hash of
  the *pattern* and rolls a window hash over the text, verifying every hit.
  :func:`rabin_karp_multi` makes one pass per pattern length, and
  :class:`RKStats` counts hash hits and spurious hits.
* :mod:`suffix_array` - :func:`build_suffix_array` (numpy prefix doubling) and
  :func:`build_lcp` (Kasai) store an index of the *text*, and
  :class:`SuffixArray` answers a query with two binary searches.
* :mod:`suffix_tree` - :class:`SuffixTree`, Option A: the compressed suffix
  trie of the *text* plus :data:`TERMINATOR`, built in O(n) from the suffix
  array and LCP array, with :class:`SuffixTreeNode` nodes.
* :mod:`search_engine` - :class:`StringSearchEngine` runs any of the four
  (:data:`METHODS`) on one text, times each query into a
  :class:`SearchResult`, builds the indexes lazily and recommends a method
  from its own measurements.

Every matcher follows the same conventions: 0-based positions, every
occurrence including overlapping ones, a sorted ``list[int]``, ``ValueError``
for an empty pattern and ``[]`` when the pattern is longer than the text.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.string_search_engine.core.kmp import MatchStats, compute_lps, kmp_search, naive_search
from src.string_search_engine.core.rabin_karp import RKStats, rabin_karp_multi, rabin_karp_search
from src.string_search_engine.core.search_engine import METHODS, SearchResult, StringSearchEngine
from src.string_search_engine.core.suffix_array import SuffixArray, build_lcp, build_suffix_array
from src.string_search_engine.core.suffix_tree import TERMINATOR, SuffixTree, SuffixTreeNode

__all__ = [
    "MatchStats", "compute_lps", "kmp_search", "naive_search", "RKStats", "rabin_karp_search",
    "rabin_karp_multi", "SuffixArray", "build_suffix_array", "build_lcp", "SuffixTree",
    "SuffixTreeNode", "TERMINATOR", "METHODS", "SearchResult", "StringSearchEngine",
]
