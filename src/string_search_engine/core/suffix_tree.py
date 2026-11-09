"""Suffix tree: the compressed trie of every suffix, built in O(n) from SA + LCP.

What it stores
--------------
A suffix tree preprocesses the *text*, not the pattern. Before any query it
stores one tree over ``text + "\\x00"``: every suffix of that terminated text
is a root-to-leaf path, each edge carries a ``(start, end)`` slice of the text
rather than a copied substring, and a query then only walks down from the root.

Option A: build from the suffix array and LCP array
---------------------------------------------------
Ukkonen's algorithm builds the tree online in O(n) but is long and delicate.
This module takes the other standard route (Option A): the sorted suffixes
are exactly the leaves of the tree read left to right, and ``lcp[r]`` is the
string depth of the deepest common ancestor of leaves ``r - 1`` and ``r``. So
one left-to-right pass over the suffix array, keeping the rightmost
root-to-leaf path on a stack, builds the tree: for suffix ``sa[r]``, pop every
node deeper than ``lcp[r]``, split an edge with a new internal node of depth
``lcp[r]`` if no node on the path has that depth, and hang the new leaf below.
Each node is pushed and popped once, so the pass is O(n) on top of the suffix
array, which :func:`~src.string_search_engine.core.suffix_array.build_suffix_array`
builds with numpy prefix doubling.

The terminator ``"\\x00"`` sorts before every other character and occurs only
at the end, so no suffix is a prefix of another, every suffix ends at its own
leaf (n + 1 leaves for a text of length n), and every internal node has at
least two children. A text that already contains ``"\\x00"`` is rejected.

Layout
------
Nodes use ``__slots__``. An edge label is the pair ``(start, end)`` into the
terminated text, so the tree holds O(n) integers however long the labels are.
Children sit in a ``dict`` keyed by the first character of their edge, in
sorted order. A node's string depth is stored, so its path label from the root
is ``terminated[end - depth: end]`` and a leaf's suffix start is
``len(terminated) - depth``. Every traversal is iterative, so a megabyte of
text does not reach Python's recursion limit.

Examples:
    >>> tree = SuffixTree("banana")
    >>> tree.find("ana")
    [1, 3]
    >>> tree.node_counts()
    {'leaves': 7, 'internal': 3, 'total': 10}
    >>> tree.internal_path_labels()
    ['a', 'ana', 'na']
    >>> tree.count_distinct_substrings(), tree.longest_repeated_substring()
    (15, 'ana')

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.string_search_engine.core.suffix_array import build_lcp, build_suffix_array

__all__ = ["SuffixTree", "SuffixTreeNode", "TERMINATOR"]

#: The end marker appended to the text; it sorts before every other character.
TERMINATOR: str = "\x00"


class SuffixTreeNode:
    """One node of a :class:`SuffixTree`, with its incoming edge as a slice.

    The edge from the parent spells ``terminated[start:end]`` of the tree's
    terminated text. ``depth`` is the string depth, the length of the path
    label from the root to this node. A leaf has ``children = None`` and
    represents the suffix starting at ``len(terminated) - depth``; an internal
    node (and the root) has a ``dict`` from the first character of each
    child's edge to the child.

    Args:
        start: Start of the incoming edge label in the terminated text.
        end: End (exclusive) of the incoming edge label.
        depth: String depth of the node.
        children: ``None`` for a leaf, else the child dict.

    Raises:
        Nothing.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1) per node plus its child dict.

    Examples:
        >>> leaf = SuffixTreeNode(5, 7, 2, None)
        >>> leaf.children is None, leaf.end - leaf.start
        (True, 2)
    """

    __slots__ = ("start", "end", "depth", "children")

    def __init__(
        self, start: int, end: int, depth: int, children: Optional[Dict[str, "SuffixTreeNode"]]
    ) -> None:
        self.start = start
        self.end = end
        self.depth = depth
        self.children = children


def _check_str(value: Any, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a ``str``."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")


class SuffixTree:
    """A compressed suffix trie of ``text + "\\x00"`` built from its SA and LCP.

    The constructor builds the suffix array and LCP array of the terminated
    text (or takes them, so an engine that already has them does not sort
    twice) and runs the O(n) stack construction described in the module
    docstring. Queries walk down from :attr:`root` comparing the pattern with
    edge slices; the matches are the leaves below the point where the walk
    ends.

    Attributes:
        text: The indexed text, without the terminator.
        terminated: ``text + "\\x00"``, the string the edges index into.
        root: The root :class:`SuffixTreeNode` (depth 0).
        build_time_s: Seconds the constructor took (``time.perf_counter``),
            including the suffix array and LCP array when it built them.

    Args:
        text: Any Python ``str`` without ``"\\x00"``. It may be empty.
        sa: Optional suffix array of ``text + "\\x00"`` (list or numpy array).
        lcp: Optional LCP array of ``text + "\\x00"``, ``lcp[0] = 0``. Give
            both or neither.

    Raises:
        TypeError: If ``text`` is not a ``str``.
        ValueError: If ``text`` contains ``"\\x00"``, if only one of ``sa`` and
            ``lcp`` is given, or if they do not have length ``len(text) + 1``
            with the terminator suffix first and ``lcp[0] = 0``.

    Time Complexity:
        O(n) for the stack construction plus the suffix array build when it is
        not given. :meth:`find` is O(m + occ log occ), :meth:`contains` O(m).

    Space Complexity:
        O(n) nodes, at most 2n + 1, each a few machine words plus a dict for
        internal nodes; in Python this is far larger than the suffix array.

    Examples:
        >>> tree = SuffixTree("mississippi")
        >>> tree.find("issi"), tree.count("s"), tree.contains("sip")
        ([1, 4], 4, True)
        >>> tree.longest_repeated_substring()
        'issi'
        >>> SuffixTree("a\\x00b")
        Traceback (most recent call last):
            ...
        ValueError: text must not contain the terminator '\\x00'
    """

    def __init__(
        self,
        text: str,
        sa: Optional[Sequence[int]] = None,
        lcp: Optional[Sequence[int]] = None,
    ) -> None:
        begin = time.perf_counter()
        _check_str(text, "text")
        if TERMINATOR in text:
            raise ValueError("text must not contain the terminator '\\x00'")
        terminated = text + TERMINATOR
        n = len(terminated)
        if (sa is None) != (lcp is None):
            raise ValueError("give both sa and lcp or neither")
        if sa is None:
            sa = build_suffix_array(terminated)
            lcp = build_lcp(terminated, sa)
        else:
            sa = sa.tolist() if hasattr(sa, "tolist") else sa
            lcp = lcp.tolist() if hasattr(lcp, "tolist") else lcp
            if len(sa) != n or len(lcp) != n or sa[0] != n - 1 or lcp[0] != 0:
                raise ValueError("sa and lcp must be the arrays of text + '\\x00'")
        self.text: str = text
        self.terminated: str = terminated
        self.root: SuffixTreeNode = self._build(terminated, sa, lcp)
        self.build_time_s: float = time.perf_counter() - begin

    @staticmethod
    def _build(t: str, sa: Sequence[int], lcp: Sequence[int]) -> SuffixTreeNode:
        """Run the stack construction over the sorted suffixes; return the root.

        The stack holds the rightmost root-to-leaf path. A node gets its edge
        start and its slot in the parent's dict only when it is popped, since
        a new internal node may still be inserted above it until then.
        """
        n = len(t)
        root = SuffixTreeNode(0, 0, 0, {})
        stack = [root]
        pop, push = stack.pop, stack.append
        for r in range(n):
            s = sa[r]
            lo = lcp[r]
            last = None
            top = stack[-1]
            while top.depth > lo:
                last = pop()
                top = stack[-1]
                if top.depth >= lo:
                    last.start = last.end - last.depth + top.depth
                    top.children[t[last.start]] = last
            if top.depth < lo:
                # Split: a new internal node of depth lo between top and last.
                p = last.end - last.depth
                node = SuffixTreeNode(0, p + lo, lo, {})
                last.start = p + lo
                node.children[t[last.start]] = last
                push(node)
            push(SuffixTreeNode(0, n, n - s, None))
        while len(stack) > 1:
            last = pop()
            top = stack[-1]
            last.start = last.end - last.depth + top.depth
            top.children[t[last.start]] = last
        return root

    @classmethod
    def from_naive(cls, text: str) -> "SuffixTree":
        """Build the same tree by inserting every suffix one at a time.

        Each suffix of ``text + "\\x00"`` walks down from the root; where it
        leaves the tree, inside an edge or at a node, the edge is split if
        needed and a leaf is added. This is O(n^2) and exists only so tests
        can cross-check the SA + LCP construction on short strings. Child
        dicts are re-sorted at the end to match the fast builder's order.

        Args:
            text: Any Python ``str`` without ``"\\x00"``.

        Returns:
            A :class:`SuffixTree` with the same structure as ``SuffixTree(text)``.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            ValueError: If ``text`` contains ``"\\x00"``.

        Time Complexity:
            O(n^2) character comparisons.

        Space Complexity:
            O(n) nodes.

        Examples:
            >>> naive = SuffixTree.from_naive("banana")
            >>> naive.internal_path_labels() == SuffixTree("banana").internal_path_labels()
            True
            >>> naive.node_counts()["leaves"]
            7
        """
        begin = time.perf_counter()
        _check_str(text, "text")
        if TERMINATOR in text:
            raise ValueError("text must not contain the terminator '\\x00'")
        t = text + TERMINATOR
        n = len(t)
        root = SuffixTreeNode(0, 0, 0, {})
        for i in range(n):
            node, d = root, 0
            while True:
                child = node.children.get(t[i + d])
                if child is None:
                    node.children[t[i + d]] = SuffixTreeNode(i + d, n, n - i, None)
                    break
                k = 0
                length = child.end - child.start
                while k < length and t[child.start + k] == t[i + d + k]:
                    k += 1
                if k == length:
                    node, d = child, d + k
                    continue
                middle = SuffixTreeNode(child.start, child.start + k, d + k, {})
                child.start += k
                middle.children[t[child.start]] = child
                middle.children[t[i + d + k]] = SuffixTreeNode(i + d + k, n, n - i, None)
                node.children[t[middle.start]] = middle
                break
        stack = [root]
        while stack:
            node = stack.pop()
            node.children = {key: node.children[key] for key in sorted(node.children)}
            stack.extend(c for c in node.children.values() if c.children is not None)
        tree = cls.__new__(cls)
        tree.text = text
        tree.terminated = t
        tree.root = root
        tree.build_time_s = time.perf_counter() - begin
        return tree

    # ------------------------------------------------------------------ walking

    def _walk(self, pattern: str) -> Tuple[SuffixTreeNode, int]:
        """Walk ``pattern`` down from the root; return ``(node, matched)``.

        ``matched`` is the length of the longest prefix of the pattern that
        spells a path from the root. ``node`` is the node at or just below
        where that path ends, so when ``matched == len(pattern)`` its leaves
        are exactly the occurrences. The walk stops at a ``"\\x00"`` in the
        pattern, since the terminator is not part of the text.
        """
        _check_str(pattern, "pattern")
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        t = self.terminated
        cut = pattern.find(TERMINATOR)
        m = len(pattern) if cut < 0 else cut
        node, i = self.root, 0
        while i < m:
            child = node.children.get(pattern[i]) if node.children else None
            if child is None:
                return node, i
            start, end = child.start, child.end
            k = min(end - start, m - i)
            if t[start : start + k] != pattern[i : i + k]:
                j = 0
                while t[start + j] == pattern[i + j]:
                    j += 1
                return child, i + j
            i += k
            node = child
        return node, i

    def _leaf_starts(self, node: SuffixTreeNode) -> List[int]:
        """Return the suffix starts of every leaf below ``node`` (unsorted)."""
        n = len(self.terminated)
        out: List[int] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current.children is None:
                out.append(n - current.depth)
            else:
                stack.extend(current.children.values())
        return out

    def contains(self, pattern: str) -> bool:
        """Return whether ``pattern`` occurs in the text.

        The pattern occurs exactly when it spells a path from the root, ending
        at a node or inside an edge.

        Args:
            pattern: A non-empty ``str``.

        Returns:
            ``True`` if the pattern is a substring of the text.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m), independent of the text length.

        Space Complexity:
            O(m) for the compared slices.

        Examples:
            >>> tree = SuffixTree("banana")
            >>> tree.contains("nan"), tree.contains("nab"), tree.contains("bananas")
            (True, False, False)
        """
        return self._walk(pattern)[1] == len(pattern)

    def find(self, pattern: str) -> List[int]:
        """Return every position where ``pattern`` occurs in the text.

        Walk the pattern down from the root; each leaf below the point where
        it ends is one suffix that starts with the pattern, so the leaves'
        suffix starts are the occurrences, overlapping ones included.

        Args:
            pattern: A non-empty ``str``.

        Returns:
            A sorted ``list[int]`` of 0-based positions; ``[]`` when the
            pattern is absent, longer than the text, or the text is empty.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m) for the walk plus O(occ log occ) to collect and sort the
            occ matches (the subtree has fewer than 2 * occ nodes).

        Space Complexity:
            O(occ).

        Examples:
            >>> SuffixTree("abracadabra").find("abra")
            [0, 7]
            >>> SuffixTree("aaaaa").find("aa")
            [0, 1, 2, 3]
            >>> SuffixTree("").find("a")
            []
            >>> SuffixTree("ab").find("")
            Traceback (most recent call last):
                ...
            ValueError: pattern must be a non-empty string
        """
        node, matched = self._walk(pattern)
        if matched != len(pattern):
            return []
        return sorted(self._leaf_starts(node))

    def count(self, pattern: str) -> int:
        """Return the number of (possibly overlapping) occurrences of ``pattern``.

        The number of leaves below the end of the pattern's walk.

        Args:
            pattern: A non-empty ``str``.

        Returns:
            The occurrence count.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m + occ).

        Space Complexity:
            O(occ) for the traversal stack in the worst case.

        Examples:
            >>> tree = SuffixTree("banana")
            >>> tree.count("a"), tree.count("ana"), tree.count("x")
            (3, 2, 0)
        """
        node, matched = self._walk(pattern)
        if matched != len(pattern):
            return 0
        return len(self._leaf_starts(node))

    def longest_prefix_match(self, pattern: str) -> int:
        """Return the length of the longest prefix of ``pattern`` that occurs in the text.

        The walk from the root goes as far as the text allows; the number of
        pattern characters it matched is the answer. A ``"\\x00"`` in the
        pattern stops the walk, since the terminator is not text.

        Args:
            pattern: A non-empty ``str``.

        Returns:
            An ``int`` between 0 and ``len(pattern)``.

        Raises:
            TypeError: If ``pattern`` is not a ``str``.
            ValueError: If ``pattern`` is empty.

        Time Complexity:
            O(m).

        Space Complexity:
            O(m) for the compared slices.

        Examples:
            >>> tree = SuffixTree("banana")
            >>> tree.longest_prefix_match("anaz"), tree.longest_prefix_match("zzz")
            (3, 0)
            >>> tree.longest_prefix_match("banana!")
            6
        """
        return self._walk(pattern)[1]

    # --------------------------------------------------------------- structure

    def _internal_nodes(self) -> List[SuffixTreeNode]:
        """Return the internal nodes (root excluded) in lexicographic preorder."""
        out: List[SuffixTreeNode] = []
        stack = [c for c in reversed(list(self.root.children.values())) if c.children is not None]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(
                c for c in reversed(list(node.children.values())) if c.children is not None
            )
        return out

    def longest_repeated_substring(self) -> str:
        """Return the longest substring that occurs at least twice.

        A substring repeats exactly when its path ends at or above an internal
        node (two leaves below), so the answer is the path label of the
        deepest internal node by string depth. On a tie the lexicographically
        smallest label wins, the same choice the suffix array makes.
        Occurrences may overlap.

        Args:
            None.

        Returns:
            The substring, or ``""`` when no character repeats.

        Raises:
            Nothing.

        Time Complexity:
            O(n) to visit the internal nodes.

        Space Complexity:
            O(n) for the list of internal nodes.

        Examples:
            >>> SuffixTree("abracadabra").longest_repeated_substring()
            'abra'
            >>> SuffixTree("aaaa").longest_repeated_substring()
            'aaa'
            >>> SuffixTree("abc").longest_repeated_substring()
            ''
        """
        best: Optional[SuffixTreeNode] = None
        for node in self._internal_nodes():
            if best is None or node.depth > best.depth:
                best = node
        if best is None:
            return ""
        return self.terminated[best.end - best.depth : best.end]

    def count_distinct_substrings(self) -> int:
        """Return the number of distinct non-empty substrings of the text.

        Every distinct substring of the terminated text is one point on one
        edge, so the sum of all edge lengths counts them. The substrings that
        include the terminator are the last character of each leaf edge, one
        per leaf, so subtracting the leaf count leaves the text's own.

        Args:
            None.

        Returns:
            A Python ``int``.

        Raises:
            Nothing.

        Time Complexity:
            O(n) to visit every node.

        Space Complexity:
            O(n) for the traversal stack in the worst case.

        Examples:
            >>> SuffixTree("banana").count_distinct_substrings()
            15
            >>> SuffixTree("aaaa").count_distinct_substrings()
            4
            >>> SuffixTree("").count_distinct_substrings()
            0
        """
        total = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            total += node.end - node.start
            if node.children is None:
                total -= 1
            else:
                stack.extend(node.children.values())
        return total

    def node_counts(self) -> Dict[str, int]:
        """Return how many leaves and internal nodes the tree has.

        Args:
            None.

        Returns:
            ``{"leaves": ..., "internal": ..., "total": ...}`` where
            ``internal`` excludes the root and ``total = leaves + internal``.
            There are always ``len(text) + 1`` leaves.

        Raises:
            Nothing.

        Time Complexity:
            O(n).

        Space Complexity:
            O(n) for the traversal stack in the worst case.

        Examples:
            >>> SuffixTree("mississippi").node_counts()
            {'leaves': 12, 'internal': 6, 'total': 18}
            >>> SuffixTree("").node_counts()
            {'leaves': 1, 'internal': 0, 'total': 1}
        """
        leaves = internal = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if node.children is None:
                leaves += 1
            else:
                internal += 1
                stack.extend(node.children.values())
        return {"leaves": leaves, "internal": internal, "total": leaves + internal}

    def internal_path_labels(self) -> List[str]:
        """Return the sorted path labels of the internal nodes, root excluded.

        The path label of a node is the string spelled from the root to it.
        Internal nodes are exactly the right-branching repeated substrings, so
        this list is the tree's shape in a form two builders can compare.

        Args:
            None.

        Returns:
            A sorted ``list[str]``; never contains the terminator.

        Raises:
            Nothing.

        Time Complexity:
            O(n + total label length).

        Space Complexity:
            O(total label length).

        Examples:
            >>> SuffixTree("abracadabra").internal_path_labels()
            ['a', 'abra', 'bra', 'ra']
            >>> SuffixTree("abc").internal_path_labels()
            []
        """
        t = self.terminated
        return sorted(t[node.end - node.depth : node.end] for node in self._internal_nodes())

    def to_dot(self) -> str:
        """Return the tree as Graphviz DOT text.

        Nodes are named ``n0`` (the root), ``n1``, ... in preorder. Internal
        nodes are small circles, leaves are boxes labelled with their suffix
        start, and each edge is labelled with its substring, showing the
        terminator as ``"$"``. Quotes, backslashes and line breaks in labels
        are escaped, so any text gives valid DOT.

        Args:
            None.

        Returns:
            A ``str`` starting with ``digraph SuffixTree {``.

        Raises:
            Nothing.

        Time Complexity:
            O(n + total edge label length), which is O(n^2) for long leaf
            edges, so draw only small trees.

        Space Complexity:
            The same as the output size.

        Examples:
            >>> print(SuffixTree("aa").to_dot())
            digraph SuffixTree {
              node [shape=circle, label="", width=0.2];
              n0 [label="root"];
              n0 -> n1 [label="$"];
              n1 [shape=box, label="2"];
              n0 -> n2 [label="a"];
              n2 -> n3 [label="$"];
              n3 [shape=box, label="1"];
              n2 -> n4 [label="a$"];
              n4 [shape=box, label="0"];
            }
        """
        t = self.terminated
        n = len(t)

        def escape(s: str) -> str:
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            s = s.replace("\n", "\\n").replace("\r", "\\r")
            return s.replace(TERMINATOR, "$")

        lines = [
            "digraph SuffixTree {",
            '  node [shape=circle, label="", width=0.2];',
            '  n0 [label="root"];',
        ]
        next_id = 1
        stack = [(child, 0) for child in reversed(list(self.root.children.values()))]
        while stack:
            node, parent_id = stack.pop()
            node_id = next_id
            next_id += 1
            label = escape(t[node.start : node.end])
            lines.append(f'  n{parent_id} -> n{node_id} [label="{label}"];')
            if node.children is None:
                lines.append(f'  n{node_id} [shape=box, label="{n - node.depth}"];')
            else:
                stack.extend((c, node_id) for c in reversed(list(node.children.values())))
        lines.append("}")
        return "\n".join(lines)
