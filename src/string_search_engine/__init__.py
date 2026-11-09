"""Week 11: fast pattern matching, indexing and text processing.

This package is the instructions' ``StringSearchEngine/`` project, it sits
under ``src/`` because the repository holds every week's project, and its
suffix tree is Option A, the compressed suffix trie built in linear time from
the suffix array and LCP array.

Four sub-packages, one question asked four ways. Every matcher here finds the
same occurrences and differs only in what it stores before it searches, and
about which string:

* :mod:`~src.string_search_engine.core` - KMP stores the LPS array of the
  pattern, Rabin-Karp the hash of the pattern, the suffix array and the suffix
  tree an index of the text. :class:`StringSearchEngine` runs all four on one
  text, times them and recommends one.
* :mod:`~src.string_search_engine.applications` - plagiarism detection with
  rolling-hash shingle fingerprints, DNA search over FASTA sequences with
  suffix arrays and suffix trees, and autocomplete over a radix trie.
* :mod:`~src.string_search_engine.utils` - the rolling hash, seeded text
  generators and a chunked text loader.
* :mod:`~src.string_search_engine.visualization` - the suffix array and suffix
  tree figures.

The public names below load from their sub-package on first use, so importing
this package imports none of the sub-packages until one of their names is used.

Examples:
    >>> from src.string_search_engine import StringSearchEngine
    >>> StringSearchEngine("abracadabra").find("abra", method="sa").positions
    [0, 7]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: The public names of each sub-package.
_EXPORTS = {
    "core": (
        "METHODS", "MatchStats", "RKStats", "SearchResult", "StringSearchEngine", "SuffixArray",
        "SuffixTree", "SuffixTreeNode", "TERMINATOR", "build_lcp", "build_suffix_array",
        "compute_lps", "kmp_search", "naive_search", "rabin_karp_multi", "rabin_karp_search",
    ),
    "applications": (
        "Autocomplete", "DEFAULT_TREE_CAP", "GenomeIndex", "PlagiarismReport", "RadixNode",
        "SEPARATOR", "compare_documents", "highlight", "load_fasta",
    ),
    "utils": (
        "DEFAULT_BASE", "DEFAULT_CHUNK_BYTES", "DEFAULT_MODULUS", "DNA_ALPHABET", "RollingHash",
        "VOCABULARY_SIZE", "ZIPF_EXPONENT", "dna_text", "english_like_text",
        "generate_sample_genome", "iter_chunks", "load_text", "periodic_text", "random_text",
        "sample_patterns", "worst_case_rk", "write_fasta",
    ),
    "visualization": (
        "MAX_DRAW_LEAVES", "MAX_PLOT_CHARS", "draw_suffix_tree", "plot_suffix_array", "write_dot",
    ),
}

#: Each public name mapped to the sub-package that provides it.
_HOME = {name: package for package, names in _EXPORTS.items() for name in names}

__all__ = list(_HOME)


def __getattr__(name: str) -> Any:
    """Load a public name from its sub-package on first use.

    Python calls this only for a name the module does not already hold.

    Args:
        name: The attribute being looked up.

    Returns:
        The object of that name from its sub-package.

    Raises:
        AttributeError: If ``name`` is not a public name of this package.

    Time Complexity:
        O(1), plus the sub-package import the first time.

    Space Complexity:
        O(1).

    Examples:
        >>> import src.string_search_engine as package
        >>> package.kmp_search.__name__
        'kmp_search'
        >>> package.boyer_moore
        Traceback (most recent call last):
            ...
        AttributeError: module 'src.string_search_engine' has no attribute 'boyer_moore'
    """
    if name in _HOME:
        return getattr(import_module(f"{__name__}.{_HOME[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
