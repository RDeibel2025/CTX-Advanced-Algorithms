"""Applications built on the core matchers (Week 11).

Three modules, each reading a different structure:

* :mod:`plagiarism` - :func:`compare_documents` fingerprints word k-shingles
  with the Rabin-Karp :class:`~src.string_search_engine.utils.RollingHash`,
  verifies every fingerprint match and scores the pair by Jaccard similarity
  in a :class:`PlagiarismReport`. The longest common substring comes from a
  suffix array over both documents joined by :data:`SEPARATOR`, and
  :func:`highlight` marks the matched regions.
* :mod:`dna_search` - :func:`load_fasta` reads FASTA, and :class:`GenomeIndex`
  keeps a suffix array per sequence and a suffix tree for sequences up to
  :data:`DEFAULT_TREE_CAP` bases. ``find_gene`` is an exact search and
  ``find_mutations`` a Hamming-distance scan.
* :mod:`autocomplete` - :class:`Autocomplete` ranks completions of a prefix by
  weight over a compressed (radix) trie of :class:`RadixNode` nodes, and
  recorded queries raise a word's weight.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.string_search_engine.applications.autocomplete import Autocomplete, RadixNode
from src.string_search_engine.applications.dna_search import (
    DEFAULT_TREE_CAP,
    GenomeIndex,
    load_fasta,
)
from src.string_search_engine.applications.plagiarism import (
    SEPARATOR,
    PlagiarismReport,
    compare_documents,
    highlight,
)

__all__ = [
    "compare_documents", "highlight", "PlagiarismReport", "SEPARATOR", "load_fasta",
    "GenomeIndex", "DEFAULT_TREE_CAP", "Autocomplete", "RadixNode",
]
