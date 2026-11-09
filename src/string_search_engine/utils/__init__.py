"""Text loading, the rolling hash and seeded text generators (Week 11).

Three modules:

* :mod:`rolling_hash` - :class:`RollingHash`, the polynomial hash
  ``sum(ord(s[i]) * base^(m-1-i)) mod q`` with an O(1) roll to the next
  window, shared by Rabin-Karp and the plagiarism fingerprints.
  :data:`DEFAULT_BASE` is 256 and :data:`DEFAULT_MODULUS` the prime 2^61 - 1.
* :mod:`generators` - :func:`random_text`, :func:`dna_text`,
  :func:`english_like_text` (a fixed synthetic vocabulary of
  :data:`VOCABULARY_SIZE` words with Zipf weights, exponent
  :data:`ZIPF_EXPONENT`), :func:`periodic_text`, :func:`worst_case_rk`,
  :func:`sample_patterns`, :func:`write_fasta` and
  :func:`generate_sample_genome`. Every random draw goes through a seeded
  ``random.Random`` from :mod:`src.utils.random_utils`, so every text is
  reproducible and none has to be stored in the repository.
* :mod:`text_loader` - :func:`load_text` and :func:`iter_chunks` read any
  user file in chunks of :data:`DEFAULT_CHUNK_BYTES`.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from src.string_search_engine.utils.generators import (
    DNA_ALPHABET,
    VOCABULARY_SIZE,
    ZIPF_EXPONENT,
    dna_text,
    english_like_text,
    generate_sample_genome,
    periodic_text,
    random_text,
    sample_patterns,
    worst_case_rk,
    write_fasta,
)
from src.string_search_engine.utils.rolling_hash import DEFAULT_BASE, DEFAULT_MODULUS, RollingHash
from src.string_search_engine.utils.text_loader import DEFAULT_CHUNK_BYTES, iter_chunks, load_text

__all__ = [
    "RollingHash", "DEFAULT_BASE", "DEFAULT_MODULUS", "random_text", "dna_text",
    "english_like_text", "periodic_text", "worst_case_rk", "sample_patterns", "write_fasta",
    "generate_sample_genome", "DNA_ALPHABET", "VOCABULARY_SIZE", "ZIPF_EXPONENT", "load_text",
    "iter_chunks", "DEFAULT_CHUNK_BYTES",
]
