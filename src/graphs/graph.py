"""One graph class covering directed/undirected and weighted/unweighted.

A graph is a set of nodes plus a set of connections between them, and almost
every design question about a graph library is really a question about how
those connections are stored. This module answers it once: the primary store
is an **adjacency list**, a dict of dicts,

    self._adjacency: Dict[node, Dict[neighbour, weight]]

so "who is next to v?" is a single dict lookup and "is there an edge u-v?"
is two. Nodes may be any hashable object, and node order is always Python
dict **insertion order** - never sorted - so a traversal written against this
class returns the same answer on every run, which is what makes the Week 4
benchmarks and doctests reproducible.

The four kinds of graph are two boolean flags rather than four classes:

* ``directed`` decides whether ``add_edge(u, v)`` also stores ``v -> u``.
* ``weighted`` decides whether edges carry a number and what
  :meth:`Graph.get_neighbors` hands back.

Writing it once matters because BFS, DFS and Dijkstra all read the same
structure. Traversal code uses :meth:`Graph.neighbors`, which always yields
bare nodes; Dijkstra uses :meth:`Graph.neighbor_items`, which always yields
``(neighbour, weight)`` pairs with weight 1.0 on an unweighted graph. Neither
algorithm has to ask which kind of graph it was handed.

The second representation, the adjacency matrix, is produced on demand by
:meth:`Graph.to_adjacency_matrix`. It is backed by numpy, not by nested
Python lists: at the Week 4 benchmark size of V = 10,000 the matrix has 10^8
cells, which is 100 MB as ``uint8`` and roughly 3 GB as a nested list of
Python ints. Every docstring below therefore quotes complexity for **both**
representations, because choosing between them is the point of Part 1:

=========================== ================== ====================
Operation                   Adjacency list     Adjacency matrix
=========================== ================== ====================
Add node                    O(1)               O(V^2) reallocation
Add / remove one edge       O(1)               O(1)
Edge existence u-v          O(1)               O(1)
Neighbours of v             O(deg(v))          O(V) row scan
Iterate every edge          O(V + E)           O(V^2)
Space                       O(V + E)           O(V^2)
=========================== ================== ====================

The list wins on sparse graphs, which is nearly all real ones; the matrix
wins only when the graph is dense enough that O(V^2) cells are mostly in use,
or when constant-time random access to an arbitrary cell is the hot path.

Two documented limitations, both deliberate:

* Self-loops are allowed and are stored once, so a self-loop adds 1 to
  :meth:`Graph.degree`, not the 2 some textbooks use.
* An explicit edge of weight 0.0 is indistinguishable from "no edge" in the
  matrix, because the matrix fills empty cells with 0. The adjacency list
  keeps the distinction; use :meth:`Graph.has_edge` when it matters.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np

__all__ = ["Graph", "AdjacencyMatrix"]


class AdjacencyMatrix(NamedTuple):
    """A dense matrix view of a graph, together with its row labels.

    A bare numpy array is not enough on its own: the array knows that row 2
    connects to row 5, but not which nodes those rows are. This tuple carries
    the labels alongside the numbers so the matrix can actually be read.

    Because it is a :class:`typing.NamedTuple` it unpacks positionally
    (``matrix, nodes, index = view``) while still supporting attribute
    access, and it costs no more memory than a plain tuple.

    Attributes:
        matrix: Square numpy array. Cell ``[i, j]`` describes the edge from
            ``nodes[i]`` to ``nodes[j]``. ``uint8`` (1 = edge, 0 = none) for
            an unweighted graph, ``float32`` (0.0 = no edge) for a weighted
            one. Measure the footprint of a representation with
            ``matrix.nbytes``.
        nodes: Row labels in graph insertion order, so ``nodes[i]`` is the
            node owning row ``i``.
        index: The inverse of ``nodes``: node to row number.

    Examples:
        >>> g = Graph()
        >>> g.add_edge("a", "b")
        >>> view = g.to_adjacency_matrix()
        >>> view.nodes
        ['a', 'b']
        >>> view.index
        {'a': 0, 'b': 1}
        >>> view.matrix.tolist()
        [[0, 1], [1, 0]]
        >>> view.matrix.nbytes
        4

        It unpacks like any tuple:

        >>> matrix, nodes, index = view
        >>> int(matrix[index["a"], index["b"]])
        1
    """

    matrix: np.ndarray
    nodes: List[Any]
    index: Dict[Any, int]


class Graph:
    """A graph stored as an adjacency list, with an adjacency matrix on demand.

    One class covers all four combinations of the two constructor flags.
    Nodes are any hashable object; edges carry a float weight, which is
    fixed at 1.0 unless the graph was created with ``weighted=True``.

    **Node order is insertion order, everywhere.** ``nodes()``, ``edges()``,
    iteration, neighbour iteration and matrix row order all follow the order
    things were added, never a sort. Traversals built on this class are
    therefore deterministic without needing sortable node labels.

    Args:
        directed: If True, ``add_edge(u, v)`` records only ``u -> v``. If
            False (the default), the edge is stored from both endpoints.
        weighted: If True, edges carry caller-supplied weights and
            :meth:`get_neighbors` returns ``(neighbour, weight)`` pairs. If
            False (the default), passing any weight other than 1.0 to
            :meth:`add_edge` is an error.

    Time Complexity:
        Construction is O(1). Per-operation costs are given on each method
        for both the adjacency list and the adjacency matrix.

    Space Complexity:
        O(V + E) for the adjacency list held by the instance. A matrix
        produced by :meth:`to_adjacency_matrix` is a separate O(V^2)
        allocation and is not cached.

    Examples:
        An undirected, unweighted graph:

        >>> g = Graph()
        >>> g.add_edge("a", "b")
        >>> g.add_edge("b", "c")
        >>> g.nodes()
        ['a', 'b', 'c']
        >>> g.get_neighbors("b")
        ['a', 'c']
        >>> print(g)
        Graph(undirected, unweighted): 3 nodes, 2 edges
        >>> g.to_adjacency_matrix().matrix.tolist()
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

        The same edge in a directed graph goes one way only:

        >>> d = Graph(directed=True)
        >>> d.add_edge("a", "b")
        >>> d.get_neighbors("a"), d.get_neighbors("b")
        (['b'], [])
        >>> d.has_edge("a", "b"), d.has_edge("b", "a")
        (True, False)

        A weighted graph reports weights alongside the neighbours:

        >>> w = Graph(weighted=True)
        >>> w.add_edge("a", "b", 2.5)
        >>> w.add_edge("b", "c", 0.5)
        >>> w.get_neighbors("b")
        [('a', 2.5), ('c', 0.5)]
        >>> w.to_adjacency_matrix().matrix.tolist()
        [[0.0, 2.5, 0.0], [2.5, 0.0, 0.5], [0.0, 0.5, 0.0]]

        Membership, size and iteration work as expected:

        >>> len(g), "a" in g, "z" in g
        (3, True, False)
        >>> list(g)
        ['a', 'b', 'c']
    """

    __slots__ = ("_adjacency", "_directed", "_weighted", "_edge_count")

    def __init__(self, directed: bool = False, weighted: bool = False) -> None:
        self._directed = bool(directed)
        self._weighted = bool(weighted)
        self._adjacency: Dict[Any, Dict[Any, float]] = {}
        self._edge_count = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _require_node(self, v: Any) -> Dict[Any, float]:
        """Return the neighbour row of ``v``, or raise a clear KeyError.

        Every method that must be given an existing node routes through
        this, so the message is identical everywhere and no method has to
        repeat the check. The row is returned rather than discarded because
        the caller almost always wants it.

        Args:
            v: The node to look up.

        Returns:
            The live ``{neighbour: weight}`` dict belonging to ``v``. It is
            the internal mapping, not a copy, so callers must not mutate it.

        Raises:
            KeyError: If ``v`` is not in the graph.
            TypeError: If ``v`` is not hashable.

        Time Complexity:
            O(1) list / O(1) matrix (a row index lookup).

        Space Complexity:
            O(1) - nothing is copied.

        Examples:
            >>> g = Graph()
            >>> g.add_edge(1, 2)
            >>> g._require_node(1)
            {2: 1.0}
            >>> g._require_node(9)
            Traceback (most recent call last):
                ...
            KeyError: 'node 9 is not in the graph'
        """
        try:
            return self._adjacency[v]
        except KeyError:
            raise KeyError(f"node {v!r} is not in the graph") from None

    def _missing_edge(self, u: Any, v: Any) -> KeyError:
        """Build the KeyError raised when an edge is asked for but absent.

        Returns the exception instead of raising it so the caller keeps the
        ``raise`` statement, which reads better at the call site. The arrow
        reflects the graph kind: ``->`` when directed, ``--`` when not.

        Args:
            u: The edge's first endpoint.
            v: The edge's second endpoint.

        Returns:
            An unraised :class:`KeyError` carrying the message.

        Time Complexity:
            O(1) for both representations.

        Space Complexity:
            O(1).

        Examples:
            >>> raise Graph()._missing_edge("a", "b")
            Traceback (most recent call last):
                ...
            KeyError: "no edge 'a' -- 'b'"
            >>> raise Graph(directed=True)._missing_edge("a", "b")
            Traceback (most recent call last):
                ...
            KeyError: "no edge 'a' -> 'b'"
        """
        arrow = "->" if self._directed else "--"
        return KeyError(f"no edge {u!r} {arrow} {v!r}")

    def _index(self) -> Dict[Any, int]:
        """Map every node to its row number in the adjacency matrix.

        Insertion order is the row order, so this is simply an enumeration
        of the adjacency dict. It is rebuilt on each call rather than cached
        because adding a node would invalidate a cache anyway.

        Returns:
            A dict from node to zero-based row index.

        Time Complexity:
            O(V) for both representations.

        Space Complexity:
            O(V).

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_node("c")
            >>> g._index()
            {'a': 0, 'b': 1, 'c': 2}
        """
        return {node: i for i, node in enumerate(self._adjacency)}

    # ------------------------------------------------------------------
    # Flags and size
    # ------------------------------------------------------------------
    @property
    def directed(self) -> bool:
        """True if edges go one way only. Read-only, fixed at construction.

        Time Complexity:
            O(1) for both representations.

        Examples:
            >>> Graph().directed, Graph(directed=True).directed
            (False, True)
        """
        return self._directed

    @property
    def weighted(self) -> bool:
        """True if edges carry caller-supplied weights. Read-only.

        Time Complexity:
            O(1) for both representations.

        Examples:
            >>> Graph().weighted, Graph(weighted=True).weighted
            (False, True)
        """
        return self._weighted

    @property
    def node_count(self) -> int:
        """Number of nodes, V.

        Time Complexity:
            O(1) list (the dict knows its length) / O(1) matrix (a side
            length), though building the matrix itself is O(V^2).

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.node_count
            2
        """
        return len(self._adjacency)

    @property
    def edge_count(self) -> int:
        """Number of edges, E, counting each undirected edge exactly once.

        Maintained incrementally as edges are added and removed, so reading
        it never walks the graph. A self-loop counts as one edge.

        Time Complexity:
            O(1) list (a stored counter) / O(V^2) matrix (there is no
            counter in a matrix; every cell has to be inspected).

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "a")
            >>> g.edge_count
            1
            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.add_edge("b", "a")
            >>> d.edge_count
            2
        """
        return self._edge_count

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def add_node(self, v: Any) -> None:
        """Add a node with no edges. Adding an existing node does nothing.

        Idempotence is deliberate: callers that build a graph from a stream
        of edges should not have to track which endpoints they have already
        seen, and re-adding a node must never discard its edges.

        Args:
            v: Any hashable object to use as the node's identity.

        Returns:
            None.

        Raises:
            TypeError: If ``v`` is not hashable.

        Time Complexity:
            O(1) list (one dict insertion) / O(V^2) matrix, where the whole
            array has to be reallocated one row and one column larger.

        Space Complexity:
            O(1) amortised list / O(V^2) matrix.

        Examples:
            >>> g = Graph()
            >>> g.add_node("a")
            >>> g.add_edge("a", "b")
            >>> g.add_node("a")
            >>> g.get_neighbors("a")
            ['b']
            >>> g.nodes()
            ['a', 'b']
            >>> g.add_node(["unhashable"])
            Traceback (most recent call last):
                ...
            TypeError: unhashable type: 'list'
        """
        if v not in self._adjacency:
            self._adjacency[v] = {}

    def remove_node(self, v: Any) -> None:
        """Remove a node and every edge that touches it.

        On a directed graph this means the incoming edges too, not only the
        row belonging to ``v``: leaving ``u -> v`` behind after ``v`` is
        gone would corrupt the structure. The scan over the other rows is
        taken from a snapshot of the keys, so no dict is resized while it is
        being iterated.

        Args:
            v: The node to remove.

        Returns:
            None.

        Raises:
            KeyError: If ``v`` is not in the graph.

        Time Complexity:
            O(V + deg(v)) list, dominated by the scan for incoming edges; it
            is O(deg(v)) on an undirected graph in the sense that only real
            neighbours are touched, but the key snapshot is still O(V).
            O(V^2) matrix, which must be rebuilt without the row and column.

        Space Complexity:
            O(V) list for the snapshot of node keys / O(V^2) matrix.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> g.remove_node("b")
            >>> g.nodes(), g.edges(), g.edge_count
            (['a', 'c'], [], 0)

            Incoming directed edges go as well:

            >>> d = Graph(directed=True)
            >>> d.add_edge(1, 2)
            >>> d.add_edge(3, 2)
            >>> d.remove_node(2)
            >>> d.nodes(), d.edges(), d.edge_count
            ([1, 3], [], 0)
            >>> d.remove_node(2)
            Traceback (most recent call last):
                ...
            KeyError: 'node 2 is not in the graph'
        """
        row = self._require_node(v)

        # Out-edges, plus a self-loop if there is one: both live in v's row.
        removed = len(row)

        # Iterate a snapshot of the keys, because the loop deletes entries
        # from the rows it visits and, at the end, a row from the graph.
        for other in list(self._adjacency):
            if other == v:
                continue
            if v in self._adjacency[other]:
                del self._adjacency[other][v]
                # Undirected: that entry is the mirror of one already
                # counted in v's own row, so it is not a separate edge.
                if self._directed:
                    removed += 1

        del self._adjacency[v]
        self._edge_count -= removed

    def nodes(self) -> List[Any]:
        """Return every node as a list, in insertion order.

        Returns:
            A new list; mutating it does not affect the graph.

        Time Complexity:
            O(V) for both representations.

        Space Complexity:
            O(V).

        Examples:
            >>> g = Graph()
            >>> g.add_edge("b", "a")
            >>> g.add_node("c")
            >>> g.nodes()
            ['b', 'a', 'c']
        """
        return list(self._adjacency)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """Connect ``u`` to ``v``, creating either node if it is not present.

        **Documented choice:** an edge whose endpoints do not yet exist
        creates them rather than raising. The assignment allows either
        behaviour provided the choice is documented and tested, and creating
        them is what makes ``for u, v in pairs: g.add_edge(u, v)`` work
        without a separate node pass. Use :meth:`add_node` when a node with
        no edges is wanted.

        On an undirected graph the edge is stored from both endpoints, so
        both neighbour lookups are O(1). A self-loop (``u == v``) is legal
        and is stored once, never mirrored. Re-adding an existing edge
        updates its weight and does not change :attr:`edge_count`.

        Args:
            u: Source endpoint.
            v: Target endpoint. On an undirected graph the roles are
                interchangeable, except that the first-inserted endpoint is
                the one :meth:`edges` reports first.
            weight: Edge weight, converted to ``float``. Must be left at the
                1.0 default unless the graph is weighted. Negative weights
                are accepted here; it is Dijkstra, not the graph, that
                rejects them.

        Returns:
            None.

        Raises:
            ValueError: If a weight other than 1.0 is given for an
                unweighted graph, or if ``weight`` is a string that is not a
                number.
            TypeError: If ``weight`` is not convertible to ``float``, or if
                either endpoint is not hashable.

        Time Complexity:
            O(1) list (one or two dict writes) / O(1) matrix once the matrix
            exists, but O(V^2) if adding the edge introduces a new node and
            forces the matrix to be reallocated.

        Space Complexity:
            O(1) amortised list / O(V^2) matrix.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.has_edge("a", "b"), g.has_edge("b", "a")
            (True, True)
            >>> g.nodes()
            ['a', 'b']

            Re-adding updates the weight without double counting:

            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 4)
            >>> w.get_edge_weight("a", "b")
            4.0
            >>> w.add_edge("a", "b", 1.5)
            >>> w.get_edge_weight("b", "a"), w.edge_count
            (1.5, 1)

            A self-loop is stored once:

            >>> s = Graph()
            >>> s.add_edge("a", "a")
            >>> s.get_neighbors("a"), s.edge_count, s.degree("a")
            (['a'], 1, 1)

            An unweighted graph refuses weights:

            >>> Graph().add_edge("a", "b", 3.0)
            Traceback (most recent call last):
                ...
            ValueError: weight 3.0 on an unweighted graph; use weighted=True
        """
        weight = float(weight)
        if not self._weighted and weight != 1.0:
            raise ValueError(
                f"weight {weight} on an unweighted graph; use weighted=True"
            )

        self.add_node(u)
        self.add_node(v)

        # Check before writing: the write itself cannot tell a new edge from
        # an update, and only a new edge changes the edge count.
        is_new = v not in self._adjacency[u]

        self._adjacency[u][v] = weight
        if not self._directed and u != v:
            self._adjacency[v][u] = weight

        if is_new:
            self._edge_count += 1

    def remove_edge(self, u: Any, v: Any) -> None:
        """Remove the edge between ``u`` and ``v``, leaving both nodes.

        On an undirected graph both stored directions go. Removing an edge
        that is not there is an error rather than a silent no-op, because a
        caller that believes it is deleting something should hear about it.

        Args:
            u: Source endpoint.
            v: Target endpoint. On a directed graph only ``u -> v`` is
                removed; ``v -> u`` is a different edge.

        Returns:
            None.

        Raises:
            KeyError: If either node is missing or the edge does not exist.

        Time Complexity:
            O(1) list (one or two dict deletions) / O(1) matrix (write a 0
            into one cell, or two for an undirected graph).

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> g.remove_edge("b", "a")
            >>> g.has_edge("a", "b"), g.nodes(), g.edge_count
            (False, ['a', 'b', 'c'], 1)
            >>> g.remove_edge("a", "b")
            Traceback (most recent call last):
                ...
            KeyError: "no edge 'a' -- 'b'"

            One direction of a directed pair survives the other:

            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.add_edge("b", "a")
            >>> d.remove_edge("a", "b")
            >>> d.has_edge("a", "b"), d.has_edge("b", "a")
            (False, True)
        """
        row = self._adjacency.get(u)
        if row is None or v not in row:
            raise self._missing_edge(u, v)

        del row[v]
        if not self._directed and u != v:
            del self._adjacency[v][u]

        self._edge_count -= 1

    def has_edge(self, u: Any, v: Any) -> bool:
        """Report whether an edge runs from ``u`` to ``v``, without raising.

        A missing node is simply a missing edge here. This is the query
        form: use :meth:`get_edge_weight` when the absence of an edge should
        be an error instead of a ``False``.

        Args:
            u: Source endpoint.
            v: Target endpoint.

        Returns:
            True if the edge exists. On an undirected graph the answer is
            the same in either argument order. Unknown or unhashable
            arguments give False rather than an exception.

        Time Complexity:
            O(1) list (two dict lookups) / O(1) matrix (one cell read). This
            is the one operation where the matrix has no advantage.

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.has_edge("a", "b"), g.has_edge("b", "a")
            (True, True)
            >>> g.has_edge("a", "zz"), g.has_edge("zz", "a")
            (False, False)
            >>> g.has_edge(["unhashable"], "a")
            False
        """
        try:
            row = self._adjacency.get(u)
            return row is not None and v in row
        except TypeError:
            # An unhashable argument cannot be a node, so it has no edges.
            return False

    def get_edge_weight(self, u: Any, v: Any) -> float:
        """Return the weight of the edge from ``u`` to ``v``.

        On an unweighted graph every edge weighs 1.0, so this still answers
        and the answer is always 1.0. That keeps weight-agnostic code, such
        as a shared shortest-path routine, from special-casing the flag.

        Args:
            u: Source endpoint.
            v: Target endpoint.

        Returns:
            The stored weight as a ``float``.

        Raises:
            KeyError: If the edge does not exist, including when a node
                does not exist.

        Time Complexity:
            O(1) list (two dict lookups) / O(1) matrix (one cell read).

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 2.5)
            >>> w.get_edge_weight("a", "b"), w.get_edge_weight("b", "a")
            (2.5, 2.5)

            Unweighted graphs answer 1.0 rather than refusing:

            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.get_edge_weight("a", "b")
            1.0
            >>> g.get_edge_weight("a", "zz")
            Traceback (most recent call last):
                ...
            KeyError: "no edge 'a' -- 'zz'"
        """
        row = self._adjacency.get(u)
        if row is None or v not in row:
            raise self._missing_edge(u, v)
        return row[v]

    def edges(self) -> List[Tuple[Any, ...]]:
        """Return every edge once, as tuples, in insertion order.

        An undirected edge is stored twice internally but reported once
        here, in a canonical orientation: the endpoint that entered the
        graph first comes first. Without that rule the same graph could list
        ``("a", "b")`` or ``("b", "a")`` depending on which row was reached,
        and edge lists would not compare equal across runs.

        Returns:
            ``(u, v, weight)`` triples on a weighted graph, ``(u, v)`` pairs
            otherwise. A self-loop appears once, as ``(v, v)``.

        Time Complexity:
            O(V + E) list, which is optimal since every edge is reported /
            O(V^2) matrix, where every cell must be examined even though
            most of them are empty.

        Space Complexity:
            O(V + E) for the returned list, plus O(V) for the index used to
            orient undirected edges.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("c", "a")
            >>> g.edges()
            [('a', 'b'), ('a', 'c')]

            Weighted graphs include the weight:

            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 2.5)
            >>> w.edges()
            [('a', 'b', 2.5)]

            Both directions of a directed pair are separate edges:

            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.add_edge("b", "a")
            >>> d.edges()
            [('a', 'b'), ('b', 'a')]
        """
        index = self._index()
        result: List[Tuple[Any, ...]] = []
        for u, row in self._adjacency.items():
            rank_u = index[u]
            for v, weight in row.items():
                # Undirected: keep only the copy stored under the endpoint
                # that was inserted first. `<=` lets a self-loop through.
                if not self._directed and rank_u > index[v]:
                    continue
                result.append((u, v, weight) if self._weighted else (u, v))
        return result

    # ------------------------------------------------------------------
    # Neighbour access
    # ------------------------------------------------------------------
    def get_neighbors(self, v: Any) -> Union[List[Any], List[Tuple[Any, float]]]:
        """Return the neighbours of ``v``, with weights when the graph has them.

        This is the method the assignment names, and its return shape
        follows the graph's ``weighted`` flag. Because that shape is not
        fixed, algorithm code should prefer :meth:`neighbors` (always bare
        nodes) or :meth:`neighbor_items` (always pairs), which do not change
        shape underneath it.

        Args:
            v: The node whose neighbours are wanted.

        Returns:
            A new list of neighbour nodes on an unweighted graph, or of
            ``(neighbour, weight)`` tuples on a weighted one, in the order
            the edges were added. On a directed graph these are the
            successors of ``v``, its out-edges.

        Raises:
            KeyError: If ``v`` is not in the graph.

        Time Complexity:
            O(deg(v)) list, touching only real neighbours / O(V) matrix,
            which must scan the whole row and skip the empty cells. This
            gap is the main reason the adjacency list is the primary store.

        Space Complexity:
            O(deg(v)) for the returned list.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("a", "c")
            >>> g.get_neighbors("a")
            ['b', 'c']
            >>> g.get_neighbors("b")
            ['a']

            Weighted graphs return pairs:

            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 2.5)
            >>> w.add_edge("a", "c", 7)
            >>> w.get_neighbors("a")
            [('b', 2.5), ('c', 7.0)]

            A node with no edges has an empty neighbour list, and an unknown
            node is an error:

            >>> g.add_node("lonely")
            >>> g.get_neighbors("lonely")
            []
            >>> g.get_neighbors("zz")
            Traceback (most recent call last):
                ...
            KeyError: "node 'zz' is not in the graph"
        """
        row = self._require_node(v)
        if self._weighted:
            return list(row.items())
        return list(row)

    def neighbors(self, v: Any) -> Iterator[Any]:
        """Iterate the neighbours of ``v`` as bare nodes, whatever the weighting.

        BFS and DFS use this. They care only about which nodes are reachable
        in one step, so giving them a shape that never varies removes a
        branch from the middle of their inner loops. Nothing is copied: the
        returned iterator is a view over the live adjacency row.

        Args:
            v: The node whose neighbours are wanted.

        Returns:
            An iterator over neighbour nodes, in edge insertion order.
            Successors only, on a directed graph.

        Raises:
            KeyError: If ``v`` is not in the graph. The error is raised by
                this call, not deferred to the first step of iteration.
            RuntimeError: If the graph gains or loses an edge at ``v``
                while the iterator is still being consumed.

        Time Complexity:
            O(1) to obtain, O(deg(v)) to consume fully, for the list / O(V)
            to consume for the matrix, which cannot skip empty cells.

        Space Complexity:
            O(1) - the iterator holds no copy of the row.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("a", "c")
            >>> list(g.neighbors("a"))
            ['b', 'c']

            The shape does not change on a weighted graph:

            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 2.5)
            >>> list(w.neighbors("a"))
            ['b']
            >>> list(w.neighbors("zz"))
            Traceback (most recent call last):
                ...
            KeyError: "node 'zz' is not in the graph"
        """
        return iter(self._require_node(v))

    def neighbor_items(self, v: Any) -> Iterator[Tuple[Any, float]]:
        """Iterate ``(neighbour, weight)`` pairs for ``v``, whatever the weighting.

        Dijkstra uses this. An unweighted graph stores 1.0 for every edge,
        so the pairs are still well formed and a shortest-path search run
        over an unweighted graph simply counts hops.

        Args:
            v: The node whose neighbours are wanted.

        Returns:
            An iterator of ``(neighbour, weight)`` tuples in edge insertion
            order, with ``weight`` always a ``float``.

        Raises:
            KeyError: If ``v`` is not in the graph, raised by this call
                rather than on first iteration.
            RuntimeError: If the graph gains or loses an edge at ``v``
                while the iterator is still being consumed.

        Time Complexity:
            O(1) to obtain, O(deg(v)) to consume fully, for the list / O(V)
            to consume for the matrix.

        Space Complexity:
            O(1) - a view over the live row, not a copy.

        Examples:
            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 2.5)
            >>> w.add_edge("a", "c", 0.5)
            >>> list(w.neighbor_items("a"))
            [('b', 2.5), ('c', 0.5)]

            Unweighted graphs report 1.0 rather than nothing:

            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> list(g.neighbor_items("a"))
            [('b', 1.0)]
        """
        return iter(self._require_node(v).items())

    # ------------------------------------------------------------------
    # Degrees and density
    # ------------------------------------------------------------------
    def degree(self, v: Any) -> int:
        """Return the degree of ``v``: its out-degree when the graph is directed.

        A self-loop contributes 1, not the 2 used by the convention that
        counts both ends. That follows from storing the loop once, and it
        keeps ``sum(degree(v) for v in g) == 2 * edge_count`` false but
        ``degree(v) == len(get_neighbors(v))`` true, which is the identity
        the traversal code actually relies on.

        Args:
            v: The node to measure.

        Returns:
            The number of distinct neighbours reachable in one step.

        Raises:
            KeyError: If ``v`` is not in the graph.

        Time Complexity:
            O(1) list (the row's length is already known) / O(V) matrix,
            which has to count the non-zero cells in the row.

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("a", "c")
            >>> g.add_edge("a", "a")
            >>> g.degree("a"), g.degree("b")
            (3, 1)

            Directed degree is out-degree:

            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.add_edge("c", "b")
            >>> d.degree("b")
            0
            >>> d.degree("zz")
            Traceback (most recent call last):
                ...
            KeyError: "node 'zz' is not in the graph"
        """
        return len(self._require_node(v))

    def out_degree(self, v: Any) -> int:
        """Return the number of edges leaving ``v``.

        On an undirected graph this equals :meth:`degree` and
        :meth:`in_degree`, because every edge is stored in both directions.

        Args:
            v: The node to measure.

        Returns:
            The size of ``v``'s adjacency row.

        Raises:
            KeyError: If ``v`` is not in the graph.

        Time Complexity:
            O(1) list / O(V) matrix (count the non-zero cells of the row).

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.add_edge("a", "c")
            >>> d.out_degree("a"), d.out_degree("b")
            (2, 0)
        """
        return len(self._require_node(v))

    def in_degree(self, v: Any) -> int:
        """Return the number of edges arriving at ``v``.

        This is where the adjacency list is at its weakest. Out-edges sit in
        one row and are counted for free; in-edges are scattered across
        every other row, so answering costs a full sweep. A matrix answers
        it with a column scan, and keeping a reverse adjacency list would
        make it O(1) at the cost of doubling the writes on every edit.

        Args:
            v: The node to measure.

        Returns:
            The number of nodes with an edge pointing at ``v``. A self-loop
            counts as one incoming edge. On an undirected graph the answer
            equals :meth:`degree`.

        Raises:
            KeyError: If ``v`` is not in the graph.

        Time Complexity:
            O(V + E) list, scanning every row / O(V) matrix, one column.

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.add_edge("c", "b")
            >>> d.in_degree("b"), d.in_degree("a")
            (2, 0)

            Undirected in-degree and out-degree agree:

            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.in_degree("a"), g.out_degree("a")
            (1, 1)
        """
        row = self._require_node(v)
        if not self._directed:
            return len(row)
        return sum(1 for other in self._adjacency.values() if v in other)

    def density(self) -> float:
        """Return the measured density: edges present over edges possible.

        Density is what separates the sparse case, where the adjacency list
        is the clear winner, from the dense case, where the matrix starts to
        pay for its O(V^2) cells. The denominator is ``V(V-1)`` for a
        directed graph and half that for an undirected one, matching the
        convention that a graph has no self-loops.

        Returns:
            A float, normally in [0.0, 1.0], and 0.0 for a graph with fewer
            than two nodes, where no edge between distinct nodes is
            possible. Self-loops are counted in the numerator but not the
            denominator, so a graph full of them can exceed 1.0.

        Time Complexity:
            O(1) list, from the stored node and edge counts / O(V^2)
            matrix, which must count its non-zero cells first.

        Space Complexity:
            O(1) for both representations.

        Examples:
            >>> g = Graph()
            >>> for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            ...     g.add_edge(u, v)
            >>> g.density()
            0.5
            >>> Graph().density(), Graph(weighted=True).density()
            (0.0, 0.0)

            The same three edges are half as dense when directed, because
            twice as many are possible:

            >>> d = Graph(directed=True)
            >>> for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
            ...     d.add_edge(u, v)
            >>> d.density()
            0.25
        """
        n = len(self._adjacency)
        if n < 2:
            return 0.0
        possible = n * (n - 1)
        if not self._directed:
            possible //= 2
        return self._edge_count / possible

    # ------------------------------------------------------------------
    # The second representation
    # ------------------------------------------------------------------
    def to_adjacency_matrix(self, dtype: Optional[Any] = None) -> AdjacencyMatrix:
        """Build the adjacency matrix view of this graph, backed by numpy.

        The matrix is computed fresh on every call and never cached, so it
        is always a snapshot: later edits to the graph do not reach it. Row
        ``i`` and column ``i`` belong to ``nodes()[i]``, and an undirected
        graph produces a symmetric matrix.

        The array is allocated with ``np.zeros`` and then filled from the
        adjacency list, which is the only affordable way to do this at the
        benchmark sizes. A nested Python list of 10,000^2 cells would hold
        10^8 pointers to boxed integers, several gigabytes; the ``uint8``
        array holding the same information is 100 MB, and the allocation
        itself is one call rather than 10^8.

        Args:
            dtype: Numpy dtype overriding the default, which is ``uint8``
                for an unweighted graph (1 = edge, 0 = none) and ``float32``
                for a weighted one (0.0 = no edge). Pass ``np.float64`` when
                weights need more precision than float32 gives, or a wider
                integer type when unweighted counts will be accumulated into
                the matrix afterwards.

        Returns:
            An :class:`AdjacencyMatrix` holding the array, the row labels in
            insertion order, and the node-to-row index.

        Raises:
            TypeError: If ``dtype`` is not a valid numpy dtype.

        Time Complexity:
            O(V^2 + E): the ``np.zeros`` allocation is proportional to the
            cell count and the fill visits each stored direction once. The
            V^2 term dominates on any sparse graph, which is exactly the
            cost being measured in the Part 1 benchmark.

        Space Complexity:
            O(V^2) for the array, plus O(V) for the labels and the index.
            The array's exact footprint is ``matrix.nbytes``.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> view = g.to_adjacency_matrix()
            >>> view.nodes
            ['a', 'b', 'c']
            >>> view.matrix.dtype
            dtype('uint8')
            >>> view.matrix.tolist()
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            >>> view.matrix.nbytes
            9

            A directed graph is not symmetric:

            >>> d = Graph(directed=True)
            >>> d.add_edge("a", "b")
            >>> d.to_adjacency_matrix().matrix.tolist()
            [[0, 1], [0, 0]]

            A weighted graph stores the weights themselves:

            >>> w = Graph(weighted=True)
            >>> w.add_edge("a", "b", 2.5)
            >>> weighted_view = w.to_adjacency_matrix()
            >>> weighted_view.matrix.dtype
            dtype('float32')
            >>> weighted_view.matrix.tolist()
            [[0.0, 2.5], [2.5, 0.0]]

            The dtype can be overridden, and an empty graph is legal:

            >>> g.to_adjacency_matrix(dtype=np.int16).matrix.dtype
            dtype('int16')
            >>> Graph().to_adjacency_matrix().matrix.tolist()
            []
        """
        nodes = list(self._adjacency)
        index = {node: i for i, node in enumerate(nodes)}

        if dtype is None:
            dtype = np.float32 if self._weighted else np.uint8

        size = len(nodes)
        matrix = np.zeros((size, size), dtype=dtype)

        for u, row in self._adjacency.items():
            i = index[u]
            for v, weight in row.items():
                j = index[v]
                value = weight if self._weighted else 1
                matrix[i, j] = value
                # Undirected rows already mirror each other, so this write
                # is redundant; it is kept so the symmetry is guaranteed by
                # this method rather than by an invariant set elsewhere.
                if not self._directed:
                    matrix[j, i] = value

        return AdjacencyMatrix(matrix=matrix, nodes=nodes, index=index)

    # ------------------------------------------------------------------
    # Dunder protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of nodes, so ``len(graph)`` is V.

        Time Complexity:
            O(1) for both representations.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> len(g)
            2
            >>> len(Graph())
            0
        """
        return len(self._adjacency)

    def __contains__(self, v: object) -> bool:
        """Report node membership, so ``v in graph`` asks about nodes.

        Unhashable values answer False instead of raising, so a membership
        test can be written without first checking the type.

        Time Complexity:
            O(1) list / O(1) matrix (an index lookup).

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> "a" in g, "zz" in g, ["unhashable"] in g
            (True, False, False)
        """
        try:
            return v in self._adjacency
        except TypeError:
            return False

    def __iter__(self) -> Iterator[Any]:
        """Iterate the nodes in insertion order.

        Time Complexity:
            O(1) to obtain, O(V) to consume.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("b", "a")
            >>> list(g)
            ['b', 'a']
        """
        return iter(self._adjacency)

    def __str__(self) -> str:
        """Return a one-line human summary: kind, node count and edge count.

        Time Complexity:
            O(1) - both counts are already maintained.

        Examples:
            >>> g = Graph()
            >>> g.add_edge("a", "b")
            >>> print(g)
            Graph(undirected, unweighted): 2 nodes, 1 edges
            >>> print(Graph(directed=True, weighted=True))
            Graph(directed, weighted): 0 nodes, 0 edges
        """
        direction = "directed" if self._directed else "undirected"
        weighting = "weighted" if self._weighted else "unweighted"
        return (
            f"Graph({direction}, {weighting}): "
            f"{len(self._adjacency)} nodes, {self._edge_count} edges"
        )

    def __repr__(self) -> str:
        """Return an unambiguous developer summary naming both flags.

        Time Complexity:
            O(1).

        Examples:
            >>> Graph(weighted=True)
            Graph(directed=False, weighted=True, nodes=0, edges=0)
        """
        return (
            f"Graph(directed={self._directed}, weighted={self._weighted}, "
            f"nodes={len(self._adjacency)}, edges={self._edge_count})"
        )
