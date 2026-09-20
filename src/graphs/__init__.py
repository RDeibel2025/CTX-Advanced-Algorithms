"""Graph representations and traversal algorithms (Week 4).

The package holds one :class:`~src.graphs.graph.Graph` class covering all
four combinations of directed/undirected and weighted/unweighted, and the
three algorithms that walk it:

* :mod:`src.graphs.bfs` - breadth-first traversal, an explicit FIFO queue.
* :mod:`src.graphs.dfs` - depth-first traversal, offered both iteratively
  with an explicit LIFO stack and recursively on the call stack.
* :mod:`src.graphs.dijkstra` - single-source shortest paths, built on the
  Week 3 priority queue in :mod:`src.structures.heap`.

The three share one skeleton: take a node from a pending collection, mark
it visited, push its unvisited neighbours. The collection is what differs,
and what the collection is decides which algorithm you have.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""


from src.graphs.graph import AdjacencyMatrix, Graph

__all__ = ["Graph", "AdjacencyMatrix"]
