"""Autocomplete: top-k completions of a prefix from a compressed (radix) trie.

The radix trie
--------------
A trie stores a set of words as paths from the root, one character per edge,
so every word that starts with a prefix lies in the subtree where the prefix's
path ends. A compressed (radix) trie merges each chain of single-child nodes
into one edge labelled with a whole substring, so a trie of W words has at
most 2W nodes however long the words are. Inserting a word walks down the
edges it shares and splits an edge where the word leaves it: inserting
``"cat"`` into a trie holding ``"car"`` splits the edge ``"car"`` into
``"ca"`` with the children ``"r"`` and ``"t"``.

Completion and weights
----------------------
:meth:`Autocomplete.complete` walks the prefix down from the root (the prefix
may end inside an edge), collects every word in the subtree below that point,
and keeps the ``k`` with the largest weight, ties broken alphabetically. A
word's weight is its corpus frequency plus the number of times it was
recorded as a query with :meth:`Autocomplete.record_query`, so popular
searches rise over time.

What it stores before it searches
---------------------------------
It preprocesses the vocabulary, not the query: before any completion it
stores every word of the corpus (the text) with its weight in the radix trie,
and each prefix typed then plays the pattern, matched in O(length of the
prefix) node steps.

================== =========================== ===========
Operation          Time                        Space
================== =========================== ===========
from_text          O(L)                        O(L)
from_frequencies   O(total word length)        O(total)
complete           O(p + S log k)              O(S)
record_query       O(len(word))                O(1)
================== =========================== ===========

Here L is the text length, p the prefix length, and S the number of words
(and nodes) in the subtree below the prefix.

Examples:
    >>> ac = Autocomplete.from_frequencies(
    ...     {"car": 5, "card": 3, "care": 7, "careful": 2, "cat": 4, "dog": 6})
    >>> ac.complete("car", 3)
    [('care', 7.0), ('car', 5.0), ('card', 3.0)]
    >>> for _ in range(6):
    ...     _ = ac.record_query("careful")
    >>> ac.complete("car", 3)
    [('careful', 8.0), ('care', 7.0), ('car', 5.0)]

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = ["Autocomplete", "RadixNode"]

#: A word for :meth:`Autocomplete.from_text`, as in the plagiarism tokenizer.
_WORD = re.compile(r"\w+(?:'\w+)*")


class RadixNode:
    """One node of the radix trie, entered by an edge labelled with a substring.

    ``label`` is the substring on the edge from the parent (``""`` for the
    root). ``children`` maps the first character of each child's label to the
    child, so no two children start with the same character. ``weight`` is
    ``None`` when the path to this node is not a word, else the word's weight.

    Args:
        label: The incoming edge label.

    Returns:
        A new node with no children and no weight.

    Raises:
        Nothing.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1) plus the label and the child dict.

    Examples:
        >>> node = RadixNode("ca")
        >>> node.label, node.children, node.weight
        ('ca', {}, None)
    """

    __slots__ = ("label", "children", "weight")

    def __init__(self, label: str) -> None:
        self.label: str = label
        self.children: Dict[str, "RadixNode"] = {}
        self.weight: Optional[float] = None


def _require_word(word: Any, name: str = "word") -> str:
    """Return ``word`` if it is a non-empty ``str``, else raise.

    Args:
        word: The value to check.
        name: The parameter name used in the error message.

    Returns:
        ``word`` unchanged.

    Raises:
        TypeError: If ``word`` is not a ``str``.
        ValueError: If ``word`` is empty.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_word("car")
        'car'
        >>> _require_word("")
        Traceback (most recent call last):
            ...
        ValueError: word must be a non-empty string
    """
    if not isinstance(word, str):
        raise TypeError(f"{name} must be a str, got {type(word).__name__}")
    if not word:
        raise ValueError(f"{name} must be a non-empty string")
    return word


class Autocomplete:
    """Weighted prefix completion over a compressed (radix) trie of words.

    Build it with :meth:`from_text` (weights are word counts) or
    :meth:`from_frequencies` (weights given). :meth:`complete` returns the
    top-k words starting with a prefix by weight, ties broken alphabetically,
    and :meth:`record_query` adds 1 to a word's weight. ``Autocomplete()``
    starts empty.

    Attributes:
        root: The root :class:`RadixNode` (label ``""``).

    Args:
        None.

    Raises:
        Nothing.

    Time Complexity:
        O(1) to create the empty trie; see the methods.

    Space Complexity:
        O(total length of the words) for the edge labels and at most 2W nodes
        for W words.

    Examples:
        >>> ac = Autocomplete.from_frequencies({"car": 5, "cat": 4, "dog": 6})
        >>> sorted(ac.root.children), ac.root.children["c"].label
        (['c', 'd'], 'ca')
        >>> ac.complete("")
        [('dog', 6.0), ('car', 5.0), ('cat', 4.0)]
    """

    def __init__(self) -> None:
        """Create an empty autocomplete with a bare root.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> Autocomplete().complete("a")
            []
        """
        self.root: RadixNode = RadixNode("")

    @classmethod
    def from_text(cls, text: str) -> "Autocomplete":
        """Build from a corpus, weighting each lowercase word by its count.

        Words are the matches of ``\\w+(?:'\\w+)*`` (letters, digits and
        underscores, with inner apostrophes kept), lowercased.

        Args:
            text: Any ``str``.

        Returns:
            A new :class:`Autocomplete` holding every word of ``text``.

        Raises:
            TypeError: If ``text`` is not a ``str``.

        Time Complexity:
            O(L) to tokenize and count, plus O(total length of the distinct
            words) to insert them.

        Space Complexity:
            O(L).

        Examples:
            >>> ac = Autocomplete.from_text("The cat and the car. The end.")
            >>> ac.complete("th"), ac.complete("ca")
            ([('the', 3.0)], [('car', 1.0), ('cat', 1.0)])
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        counts = Counter(match.group().lower() for match in _WORD.finditer(text))
        return cls.from_frequencies(counts)

    @classmethod
    def from_frequencies(cls, frequencies: Mapping[str, float]) -> "Autocomplete":
        """Build from a ``{word: weight}`` mapping, keeping the words as given.

        Args:
            frequencies: Non-empty ``str`` words mapped to finite, non-negative
                ``int`` or ``float`` weights.

        Returns:
            A new :class:`Autocomplete`.

        Raises:
            TypeError: If ``frequencies`` is not a mapping, a word is not a
                ``str``, or a weight is not an ``int`` or ``float`` (``bool``
                rejected).
            ValueError: If a word is empty or a weight is negative or not
                finite.

        Time Complexity:
            O(total length of the words), one walk and at most one edge split
            per insertion.

        Space Complexity:
            O(total length of the words).

        Examples:
            >>> ac = Autocomplete.from_frequencies({"car": 5, "care": 7})
            >>> ac.complete("ca")
            [('care', 7.0), ('car', 5.0)]
            >>> Autocomplete.from_frequencies({"car": -1})
            Traceback (most recent call last):
                ...
            ValueError: weight of 'car' must be finite and non-negative, got -1
        """
        if not isinstance(frequencies, Mapping):
            raise TypeError(f"frequencies must be a mapping, got {type(frequencies).__name__}")
        auto = cls()
        for word, weight in frequencies.items():
            _require_word(word)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"weight of {word!r} must be a number, got {type(weight).__name__}")
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"weight of {word!r} must be finite and non-negative, got {weight}"
                )
            auto._add(word, float(weight))
        return auto

    def _add(self, word: str, amount: float) -> float:
        """Add ``amount`` to ``word``'s weight, inserting the word if needed.

        Walks down the edges the word shares with the trie. Where the word
        leaves an edge part-way, the edge is split by a new node at that
        point. A new word becomes either a new leaf or a split node.

        Args:
            word: A non-empty ``str``.
            amount: The weight to add.

        Returns:
            The word's new weight.

        Raises:
            Nothing.

        Time Complexity:
            O(len(word)).

        Space Complexity:
            O(1) new nodes (at most two) plus the new labels.

        Examples:
            >>> ac = Autocomplete()
            >>> ac._add("car", 1.0), ac._add("cat", 2.0), ac._add("car", 1.0)
            (1.0, 2.0, 2.0)
            >>> ac.root.children["c"].label, sorted(ac.root.children["c"].children)
            ('ca', ['r', 't'])
        """
        node = self.root
        i, n = 0, len(word)
        while i < n:
            child = node.children.get(word[i])
            if child is None:
                leaf = RadixNode(word[i:])
                node.children[word[i]] = leaf
                node = leaf
                break
            label = child.label
            limit = min(len(label), n - i)
            j = 0
            while j < limit and label[j] == word[i + j]:
                j += 1
            if j < len(label):
                # The word leaves the edge after j characters: split it there.
                middle = RadixNode(label[:j])
                child.label = label[j:]
                middle.children[child.label[0]] = child
                node.children[word[i]] = middle
                child = middle
            node = child
            i += j
        node.weight = (node.weight or 0.0) + amount
        return node.weight

    def complete(self, prefix: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return the ``k`` heaviest words starting with ``prefix``.

        Walks the prefix down from the root. If it leaves the trie the answer
        is empty; if it ends inside an edge, the candidates are the words
        below that edge. Every word in the subtree is collected (iteratively)
        and the top ``k`` by weight are kept with a heap, ties broken by the
        word in alphabetical order. The empty prefix matches every word.

        Args:
            prefix: Any ``str``, possibly empty.
            k: How many completions to return, at least 0.

        Returns:
            Up to ``k`` pairs ``(word, weight)``, heaviest first.

        Raises:
            TypeError: If ``prefix`` is not a ``str`` or ``k`` is not an
                ``int`` (``bool`` rejected).
            ValueError: If ``k < 0``.

        Time Complexity:
            O(p) for the walk plus O(S log k) for the S words below it.

        Space Complexity:
            O(S) for the collected words.

        Examples:
            >>> ac = Autocomplete.from_frequencies(
            ...     {"car": 5, "card": 3, "care": 7, "careful": 2, "cat": 4, "dog": 6})
            >>> ac.complete("ca", 2)
            [('care', 7.0), ('car', 5.0)]
            >>> ac.complete("", 3)
            [('care', 7.0), ('dog', 6.0), ('car', 5.0)]
            >>> ac.complete("caref"), ac.complete("z")
            ([('careful', 2.0)], [])
        """
        if not isinstance(prefix, str):
            raise TypeError(f"prefix must be a str, got {type(prefix).__name__}")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__}")
        if k < 0:
            raise ValueError(f"k must be at least 0, got {k}")

        node, spelled, i = self.root, "", 0
        while i < len(prefix):
            child = node.children.get(prefix[i])
            if child is None:
                return []
            label, rest = child.label, prefix[i:]
            if rest.startswith(label):
                i += len(label)
            elif label.startswith(rest):
                i = len(prefix)
            else:
                return []
            spelled += label
            node = child

        candidates: List[Tuple[float, str]] = []
        stack = [(node, spelled)]
        while stack:
            current, word = stack.pop()
            if current.weight is not None:
                candidates.append((-current.weight, word))
            for child in current.children.values():
                stack.append((child, word + child.label))
        return [(word, -negative) for negative, word in heapq.nsmallest(k, candidates)]

    def record_query(self, word: str) -> float:
        """Add 1 to ``word``'s weight, inserting it with weight 1 if new.

        This is the query-frequency weighting: each recorded search makes the
        word rank higher in later completions.

        Args:
            word: A non-empty ``str``, stored as given (:meth:`from_text`
                lowercases its words).

        Returns:
            The word's new weight.

        Raises:
            TypeError: If ``word`` is not a ``str``.
            ValueError: If ``word`` is empty.

        Time Complexity:
            O(len(word)).

        Space Complexity:
            O(1) new nodes.

        Examples:
            >>> ac = Autocomplete.from_frequencies({"car": 5})
            >>> ac.record_query("car"), ac.record_query("cab")
            (6.0, 1.0)
            >>> ac.complete("ca")
            [('car', 6.0), ('cab', 1.0)]
        """
        _require_word(word)
        return self._add(word, 1.0)
