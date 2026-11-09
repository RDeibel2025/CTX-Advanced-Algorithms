"""DNA search: FASTA loading, exact gene search by index, and a mismatch scan.

FASTA
-----
A FASTA file is a list of records. Each starts with a header line ``>id
description`` and continues with sequence lines until the next header.
:func:`load_fasta` keeps the identifier (the header up to the first space),
joins the sequence lines, uppercases the bases and ignores blank lines.

Exact search with the two text indexes
--------------------------------------
:class:`GenomeIndex` builds, for every sequence, a
:class:`~src.string_search_engine.core.suffix_array.SuffixArray` and (for
sequences no longer than ``tree_cap``) a
:class:`~src.string_search_engine.core.suffix_tree.SuffixTree`. A gene or
motif query then costs O(m log n) by binary search in the suffix array or
O(m) by walking down the suffix tree, plus the cost of listing the matches,
however many queries are run. The tree uses far more memory per base than the
suffix array, which is why it is capped.

Approximate search: a Hamming-distance scan
-------------------------------------------
:meth:`GenomeIndex.find_mutations` reports every window of the sequence that
differs from the pattern in at most ``max_mismatches`` positions
(substitutions only, the Hamming distance). It uses no index: for each of the
m pattern positions one numpy comparison marks the mismatching windows, and
the per-window counts are summed, O(nm) work done in m vectorized passes.

What it stores before it searches
---------------------------------
Before any query it stores, for each sequence (the text), the suffix array
and LCP array and, below the cap, the suffix tree of ``sequence + "\\x00"``;
the pattern is not preprocessed. The mismatch scan stores nothing beyond the
sequences themselves.

======================= ============================ ===========
Operation               Time                         Space
======================= ============================ ===========
load_fasta              O(L)                         O(L)
GenomeIndex             suffix array builds, O(N)    O(N)
find_gene, "sa"         O(s m log n + occ log occ)   O(occ)
find_gene, "tree"       O(s m + occ log occ)         O(occ)
find_mutations          O(N m)                       O(n)
======================= ============================ ===========

Here L is the file length, s the number of sequences, n the length of one
sequence, N the total length of all of them, m the pattern length and occ the
number of matches.

Examples:
    >>> genome = load_fasta(">seq1 sample\\nACGTACGTGACG\\n>seq2\\nTTGACGA\\nGACG\\n")
    >>> genome
    {'seq1': 'ACGTACGTGACG', 'seq2': 'TTGACGAGACG'}
    >>> index = GenomeIndex(genome)
    >>> index.find_gene("GACG")
    {'seq1': [8], 'seq2': [2, 7]}
    >>> index.find_mutations("ACGA", 1)["seq1"]
    [0, 4]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Union

import numpy as np

from src.string_search_engine.core.suffix_array import SuffixArray
from src.string_search_engine.core.suffix_tree import SuffixTree
from src.string_search_engine.utils.text_loader import load_text

__all__ = ["DEFAULT_TREE_CAP", "GenomeIndex", "load_fasta"]

#: Longest sequence that also gets a suffix tree (the tree is memory-hungry).
DEFAULT_TREE_CAP: int = 200_000

#: Path-like values accepted by :func:`load_fasta`.
PathLike = Union[str, "os.PathLike[str]"]


def _check_pattern(pattern: Any) -> str:
    """Return ``pattern`` if it is a non-empty ``str``, else raise.

    Args:
        pattern: The value to check.

    Returns:
        ``pattern`` unchanged.

    Raises:
        TypeError: If ``pattern`` is not a ``str``.
        ValueError: If ``pattern`` is empty.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_pattern("GA")
        'GA'
        >>> _check_pattern("")
        Traceback (most recent call last):
            ...
        ValueError: pattern must be a non-empty string
    """
    if not isinstance(pattern, str):
        raise TypeError(f"pattern must be a str, got {type(pattern).__name__}")
    if not pattern:
        raise ValueError("pattern must be a non-empty string")
    return pattern


def load_fasta(path_or_text: PathLike) -> Dict[str, str]:
    """Parse FASTA text, or read and parse a FASTA file, into ``{id: sequence}``.

    A ``str`` whose first non-blank character is ``">"`` or that contains a
    newline is parsed as FASTA text; any other ``str`` or path object is read
    as a file with :func:`~src.string_search_engine.utils.text_loader.load_text`.
    Each record's identifier is its header up to the first whitespace. Its
    sequence lines are joined with all whitespace removed and uppercased.
    Blank lines are ignored, and a header with no sequence lines gives ``""``.

    Args:
        path_or_text: FASTA text, or a path to a FASTA file.

    Returns:
        A dict from identifier to sequence, in file order. Blank text gives
        ``{}``.

    Raises:
        TypeError: If ``path_or_text`` is neither a ``str`` nor a path.
        ValueError: If sequence data comes before the first header, a header
            has no identifier, or an identifier repeats.
        FileNotFoundError: If a path is given and does not exist.

    Time Complexity:
        O(L) for L characters of FASTA text.

    Space Complexity:
        O(L).

    Examples:
        >>> load_fasta(">a first\\nacg\\n\\nTT\\n>b\\n")
        {'a': 'ACGTT', 'b': ''}
        >>> load_fasta("ACGT\\n>a\\n")
        Traceback (most recent call last):
            ...
        ValueError: FASTA sequence data before the first '>' header
        >>> load_fasta(">a\\nA\\n>a\\nC\\n")
        Traceback (most recent call last):
            ...
        ValueError: repeated FASTA identifier 'a'
    """
    if isinstance(path_or_text, os.PathLike):
        text = load_text(path_or_text)
    elif isinstance(path_or_text, str):
        stripped = path_or_text.lstrip()
        if not stripped or stripped.startswith(">") or "\n" in path_or_text:
            text = path_or_text
        else:
            text = load_text(path_or_text)
    else:
        raise TypeError(
            f"path_or_text must be a str or a path, got {type(path_or_text).__name__}"
        )

    records: Dict[str, str] = {}
    current = None
    chunks: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records[current] = "".join(chunks)
            fields = line[1:].split()
            if not fields:
                raise ValueError("FASTA header without an identifier")
            current = fields[0]
            if current in records:
                raise ValueError(f"repeated FASTA identifier {current!r}")
            records[current] = ""
            chunks = []
        else:
            if current is None:
                raise ValueError("FASTA sequence data before the first '>' header")
            chunks.append("".join(line.split()).upper())
    if current is not None:
        records[current] = "".join(chunks)
    return records


class GenomeIndex:
    """Suffix array and suffix tree indexes over a set of named sequences.

    Each sequence gets a :class:`SuffixArray` and, when it has at most
    ``tree_cap`` characters, a :class:`SuffixTree` (built from the suffix
    array and LCP array of the sequence plus its terminator). Exact queries
    use one of the two indexes; the mismatch scan reads the sequences
    directly. Matching is exact and case-sensitive: :func:`load_fasta`
    uppercases sequences, so query with uppercase patterns.

    Attributes:
        sequences: ``{seq_id: sequence}``, a copy of the input.
        tree_cap: The longest sequence that gets a suffix tree.
        suffix_arrays: ``{seq_id: SuffixArray}`` for every sequence.
        suffix_trees: ``{seq_id: SuffixTree}`` for sequences within the cap.

    Args:
        sequences: A mapping from identifier (``str``) to sequence (``str``),
            for example the result of :func:`load_fasta`.
        tree_cap: Longest sequence to build a suffix tree for, at least 0.

    Raises:
        TypeError: If ``sequences`` is not a mapping of ``str`` to ``str``,
            or ``tree_cap`` is not an ``int``.
        ValueError: If ``tree_cap`` is negative, or a sequence within the cap
            contains ``"\\x00"``.

    Time Complexity:
        One suffix array build per sequence, plus a second one of the
        terminated sequence and the O(n) stack construction for each tree.

    Space Complexity:
        O(N) integers for the suffix arrays; the trees hold up to 2n + 1
        Python objects each, far more memory per base.

    Examples:
        >>> index = GenomeIndex({"s1": "GATTACA", "s2": "ACAGATTACA"}, tree_cap=8)
        >>> sorted(index.suffix_arrays), sorted(index.suffix_trees)
        (['s1', 's2'], ['s1'])
        >>> index.find_gene("ACA", method="tree")
        {'s1': [4], 's2': [0, 7]}
    """

    def __init__(self, sequences: Mapping[str, str], tree_cap: int = DEFAULT_TREE_CAP) -> None:
        """Build a suffix array for every sequence and a tree for the short ones.

        Args:
            sequences: A mapping from identifier to sequence.
            tree_cap: Longest sequence to build a suffix tree for, at least 0.

        Returns:
            None.

        Raises:
            TypeError: If ``sequences`` is not a mapping of ``str`` to
                ``str``, or ``tree_cap`` is not an ``int`` (``bool``
                rejected).
            ValueError: If ``tree_cap`` is negative, or a sequence within the
                cap contains ``"\\x00"``.

        Time Complexity:
            As the class.

        Space Complexity:
            As the class.

        Examples:
            >>> GenomeIndex({"x": "ACGT"}, tree_cap=0).suffix_trees
            {}
            >>> GenomeIndex({"x": "ACGT"}, tree_cap=-1)
            Traceback (most recent call last):
                ...
            ValueError: tree_cap must be at least 0, got -1
        """
        if not isinstance(sequences, Mapping):
            raise TypeError(f"sequences must be a mapping, got {type(sequences).__name__}")
        if isinstance(tree_cap, bool) or not isinstance(tree_cap, int):
            raise TypeError(f"tree_cap must be an int, got {type(tree_cap).__name__}")
        if tree_cap < 0:
            raise ValueError(f"tree_cap must be at least 0, got {tree_cap}")
        for seq_id, sequence in sequences.items():
            if not isinstance(seq_id, str) or not isinstance(sequence, str):
                raise TypeError("sequences must map str identifiers to str sequences")
        self.sequences: Dict[str, str] = dict(sequences)
        self.tree_cap: int = tree_cap
        self.suffix_arrays: Dict[str, SuffixArray] = {
            seq_id: SuffixArray(sequence) for seq_id, sequence in self.sequences.items()
        }
        self.suffix_trees: Dict[str, SuffixTree] = {
            seq_id: SuffixTree(sequence)
            for seq_id, sequence in self.sequences.items()
            if len(sequence) <= tree_cap
        }

    def find_gene(self, pattern: str, method: str = "sa") -> Dict[str, List[int]]:
        """Return every exact occurrence of ``pattern`` in every sequence.

        ``method="sa"`` binary-searches each sequence's suffix array;
        ``method="tree"`` walks each sequence's suffix tree. A sequence longer
        than ``tree_cap`` has no tree, so a ``"tree"`` query uses its suffix
        array instead; both give the same positions.

        Args:
            pattern: A non-empty ``str``.
            method: ``"sa"`` or ``"tree"``.

        Returns:
            ``{seq_id: sorted 0-based positions}`` for every sequence, in
            input order, with ``[]`` where the pattern does not occur.
            Overlapping occurrences are all reported.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty or ``method`` is unknown.

        Time Complexity:
            O(m log n + occ log occ) per sequence with ``"sa"``, O(m + occ log
            occ) with ``"tree"``.

        Space Complexity:
            O(occ) for the results.

        Examples:
            >>> index = GenomeIndex({"s": "AAAA"})
            >>> index.find_gene("AA"), index.find_gene("AA", method="tree")
            ({'s': [0, 1, 2]}, {'s': [0, 1, 2]})
            >>> index.find_gene("AA", method="kmp")
            Traceback (most recent call last):
                ...
            ValueError: method must be 'sa' or 'tree', got 'kmp'
        """
        _check_pattern(pattern)
        if method not in ("sa", "tree"):
            raise ValueError(f"method must be 'sa' or 'tree', got {method!r}")
        result: Dict[str, List[int]] = {}
        for seq_id in self.sequences:
            tree = self.suffix_trees.get(seq_id) if method == "tree" else None
            if tree is not None:
                result[seq_id] = tree.find(pattern)
            else:
                result[seq_id] = self.suffix_arrays[seq_id].find(pattern)
        return result

    def find_mutations(self, pattern: str, max_mismatches: int = 1) -> Dict[str, List[int]]:
        """Return every window within Hamming distance ``max_mismatches`` of ``pattern``.

        For a sequence of length n there are n - m + 1 windows of length m.
        The sequence's code points are read into a numpy array once, and for
        each pattern position j a vectorized comparison of the shifted array
        with ``pattern[j]`` adds 1 to every window that mismatches there. The
        windows whose count is at most ``max_mismatches`` are the answers,
        exact matches included. Only substitutions are counted, not
        insertions or deletions.

        Args:
            pattern: A non-empty ``str``.
            max_mismatches: Largest number of differing positions allowed, at
                least 0. 0 gives the exact matches.

        Returns:
            ``{seq_id: sorted 0-based window starts}`` for every sequence, in
            input order; ``[]`` when the pattern is longer than the sequence.

        Raises:
            TypeError: If ``pattern`` is not a ``str`` or ``max_mismatches``
                is not an ``int`` (``bool`` rejected).
            ValueError: If ``pattern`` is empty or ``max_mismatches < 0``.

        Time Complexity:
            O(n m) per sequence, done as m numpy passes over n - m + 1 windows.

        Space Complexity:
            O(n) for the code-point array and the per-window counts.

        Examples:
            >>> index = GenomeIndex({"s": "ACGTTCGA"})
            >>> index.find_mutations("ACGA", 1), index.find_mutations("ACGA", 0)
            ({'s': [0, 4]}, {'s': []})
            >>> index.find_mutations("TCGA", 0), index.find_mutations("TCGA", 2)
            ({'s': [4]}, {'s': [0, 4]})
        """
        _check_pattern(pattern)
        if isinstance(max_mismatches, bool) or not isinstance(max_mismatches, int):
            raise TypeError(
                f"max_mismatches must be an int, got {type(max_mismatches).__name__}"
            )
        if max_mismatches < 0:
            raise ValueError(f"max_mismatches must be at least 0, got {max_mismatches}")
        m = len(pattern)
        result: Dict[str, List[int]] = {}
        for seq_id, sequence in self.sequences.items():
            n = len(sequence)
            if m > n:
                result[seq_id] = []
                continue
            windows = n - m + 1
            codes = np.frombuffer(sequence.encode("utf-32-le", "surrogatepass"), dtype="<u4")
            mismatches = np.zeros(windows, dtype=np.int64)
            for j, char in enumerate(pattern):
                mismatches += codes[j:j + windows] != ord(char)
            result[seq_id] = np.flatnonzero(mismatches <= max_mismatches).tolist()
        return result
