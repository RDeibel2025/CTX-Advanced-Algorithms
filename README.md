# CSC 5300 Advanced Algorithms - Course Project

**Robert Deibel**  
Concordia University Texas · CSC 5300 Advanced Algorithms · Fall 2026

Repository: <https://github.com/RDeibel2025/CTX-Advanced-Algorithms>

| Week | Assignment | Deliverable |
|---|---|---|
| 1 | Algorithm Laboratory Setup | [`docs/performance_analysis.md`](docs/performance_analysis.md) |
| 2 | Divide and Conquer | [`analysis/week2_report.md`](analysis/week2_report.md) · [`analysis/week2_recurrences.md`](analysis/week2_recurrences.md) |
| 3 | Data Structures | [`analysis/week3_report.md`](analysis/week3_report.md) |
| 4 | Graph Algorithms | [`analysis/week4_report.md`](analysis/week4_report.md) |

---

## What this project is

A working laboratory for measuring how algorithms and data structures
actually behave, rather than only reasoning about how they should behave.

It contains five things:

1. **Five sorting algorithms** - an optimized bubble sort, selection sort
   and insertion sort (Week 1), plus merge sort and a randomized quicksort
   (Week 2) - all written to a single explicit contract: reject non-list
   input, never mutate the caller's list, handle empty and single-element
   input, and work on any comparable element type. Because they share that
   contract they are interchangeable in the benchmark harness, which
   [`tests/test_sorting_comparison.py`](tests/test_sorting_comparison.py)
   verifies.
2. **Three data structures** (Week 3) - a binary min-heap and max-heap with
   a priority queue built on them, an AVL tree that rebalances on deletion
   as well as insertion, and a hash table offering both separate chaining
   and linear probing. Each is benchmarked against the Python built-in
   that does the same job (`heapq`, `dict`), with a plain `list` as the
   linear baseline.
3. **Three graph algorithms** (Week 4) - BFS, DFS in both iterative and
   recursive form, and Dijkstra's shortest paths built on the Week 3
   priority queue, over one `Graph` class that holds both an adjacency
   list and a numpy-backed adjacency matrix. The three are deliberately
   one loop with three different containers.
4. **A benchmarking framework** that generates eight different shapes of
   input, times an algorithm - or, since Week 3, any operation on a
   structure - over repeated runs with `time.perf_counter`, reports mean,
   standard deviation, minimum and maximum, fits the measurements against
   O(n), O(n log n) and O(n²) reference models, plots the comparison, and
   stores the results as CSV for later retrieval.
5. **A performance study each week** built from a real run of that
   framework - charts, a results table, and a report on what the
   measurements actually show.

The remaining directories (`src/searching/`,
`src/dynamic_programming/`) are the semester's
scaffolding. They are real Python packages with documented placeholders,
ready for later weeks.

---

## Project structure

```
Advanced Algorithms/
├── README.md                       This file
├── SUBMISSION.md                   Cover document for the Week 1 submission
├── requirements.txt                Pinned dependency versions (pip freeze)
├── setup.py                        Packaging metadata; `pip install -e .`
├── check_environment.py            Environment verification script
├── .gitignore
├── .github/workflows/tests.yml     CI: runs the test suite on every push
├── src/
│   ├── sorting/
│   │   ├── basic_sorts.py          Bubble (optimized), selection, insertion
│   │   ├── merge_sort.py           Merge sort + linear merge helper
│   │   ├── quick_sort.py           Randomized quicksort, 2-way and 3-way
│   │   └── advanced_sorts.py       Reserved for a later week
│   ├── structures/
│   │   ├── heap.py                 MinHeap, MaxHeap, PriorityQueue
│   │   ├── avl_tree.py             AVL tree with deletion rebalancing
│   │   └── hash_table.py           Separate chaining and linear probing
│   ├── searching/                  Reserved for a later week
│   ├── graphs/
│   │   ├── graph.py                Graph: adjacency list and numpy matrix
│   │   ├── bfs.py                  Breadth-first traversal, explicit queue
│   │   ├── dfs.py                  Depth-first, iterative and recursive
│   │   └── dijkstra.py             Shortest paths on the Week 3 heap
│   ├── dynamic_programming/        Reserved for a later week
│   └── utils/
│       ├── benchmark.py            BenchmarkResult, AlgorithmBenchmark
│       ├── visualization.py        Supplementary charts, traversal figures
│       ├── graph_generator.py      Seeded sparse/dense/random graphs
│       └── testing_helpers.py      Shared predicates and test fixtures
├── tests/
│   ├── conftest.py                 Shared fixtures and parametrisation
│   ├── test_sorting.py             Week 1 sorting algorithm tests
│   ├── test_merge_sort.py          Merge sort tests
│   ├── test_quick_sort.py          QuickSort tests
│   ├── test_sorting_comparison.py  All five algorithms must agree
│   ├── test_heap.py                Heap and priority queue tests
│   ├── test_avl_tree.py            AVL invariants after every operation
│   ├── test_hash_table.py          Both strategies, tombstones, rehashing
│   ├── test_data_structure_comparison.py   All three structures must agree
│   ├── test_graph_representation.py  List and matrix must agree
│   ├── test_bfs.py                 BFS order against an independent oracle
│   ├── test_dfs.py                 Iterative and recursive must match
│   ├── test_dijkstra.py            Hand-computed paths, negative weights
│   ├── test_graph_benchmark.py     The harness runs and writes its outputs
│   ├── test_searching.py           Reserved for a later week
│   └── test_utils.py               Benchmarking framework tests
├── benchmarks/
│   ├── sorting_benchmarks.py       The Week 1 end-to-end benchmark driver
│   ├── week2_performance.py        The Week 2 divide-and-conquer benchmark
│   ├── week3_structures_benchmark.py   The Week 3 data structures benchmark
│   ├── week4_graph_benchmark.py    The Week 4 graph benchmark
│   ├── complexity_validation.py    Reserved for a later week
│   └── results/                    Week 2 and Week 3 charts and measurements
├── analysis/
│   ├── week2_report.md             Week 2 technical report
│   ├── week2_recurrences.md        Master Theorem solutions
│   ├── week3_report.md             Week 3 technical report
│   └── week4_report.md             Week 4 technical report
├── docs/
│   ├── performance_analysis.md     The Week 1 report (generated)
│   ├── AI_USE.md                   AI use disclosure, every week
│   └── figures/                    Week 1 charts (PNG)
├── tools/
│   ├── build_report.py             Renders the Week 1 report from the result CSVs
│   ├── week2_facts.py              Prints every figure the Week 2 report quotes
│   ├── week2_sync_report.py        Keeps the Week 2 report's tables in step with the CSVs
│   ├── build_week2_pdf.py          Builds the Week 2 submission PDF
│   ├── week3_facts.py              Prints every figure the Week 3 report quotes
│   ├── build_week3_pdf.py          Builds the Week 3 submission PDF
│   ├── week4_facts.py              Prints every figure the Week 4 report quotes
│   ├── build_week4_pdf.py          Builds the Week 4 submission PDF
│   ├── md_to_pdf.py                Markdown to PDF export
│   └── package_submission.sh       Builds the Week 1 submission zip and PDF
├── submissions/                    What was handed in, one folder per week
├── notebooks/                      Reserved for exploratory work
└── examples/
    ├── week2_demo.py               Runnable Week 2 demonstration
    ├── week3_demo.py               Runnable Week 3 demonstration
```

---

## Setup

Requires Python 3.9 or later. Developed and measured on Python 3.12.4.

```bash
git clone https://github.com/RDeibel2025/CTX-Advanced-Algorithms.git
cd CTX-Advanced-Algorithms

python3 -m venv algorithms_course
source algorithms_course/bin/activate        # macOS / Linux
# .\algorithms_course\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

The nine required packages are numpy, matplotlib, pandas, jupyter, pytest,
scipy, scikit-learn, plotly and seaborn. Week 4 adds `networkx`, which
draws the traversal figures. `requirements.txt` also pins
`markdown`, which only `tools/md_to_pdf.py` uses when exporting
`SUBMISSION.md` to PDF; nothing in `src/`, `tests/` or `benchmarks/`
depends on it.

---

## Verifying the environment

```bash
python check_environment.py
```

Checks the Python version, imports all nine packages and prints their
versions, confirms every required directory and file exists, and confirms
Git is available and this directory is a working tree. It exits `0` when
everything passes and `1` when anything fails, naming each failure.

---

## Running the tests

```bash
pytest tests/ -v
```

Covering: every required edge case against all three algorithms
(empty, single element, sorted, reverse sorted, duplicates, all identical,
negatives, mixed signs, 1,000-element arrays), output correctness as
*both* ordered and a permutation of the input, stability, `TypeError` on
non-list input, non-mutation of the caller's list, direct comparison
counting to prove bubble sort's early exit is present, and the
benchmarking framework itself - its eight generators, its seeding, its
statistics, its rejection of deliberately broken sorts, and its CSV
round trip.

Week 3 adds four suites: the heap property after every insert and
extract, `heapify` on arbitrary input, and priority-queue order and
stability; the AVL balance and stored-height invariants checked after
every operation of randomised sequences of 1,000 and more, with height
held to the theoretical bound; both hash-table strategies through insert,
get, delete, rehash and load factor, including the linear-probing
tombstone case; and a cross-check that all three structures agree on
membership for the same keys.

Week 4 adds five more: the adjacency list and the adjacency matrix must
agree on every ordered pair of nodes; BFS visit order is checked against
an independent brute-force level expansion; iterative and recursive DFS
must return identical orders, and the recursion limit must be restored
afterwards; Dijkstra is checked against a hand-computed graph, against
BFS hop levels on unweighted input, and against its own O(V^2) baseline;
and the benchmark harness is run end to end to confirm it writes every
output. The suite is 2,630 tests.

The docstring examples are executable too:

```bash
pytest --doctest-modules src/
```

---

## Week 4: graph algorithms

Week 4 adds one `networkx` dependency, already pinned in
`requirements.txt`. Everything else is the standard library and numpy.

### The four modules

| Module | Contents |
|---|---|
| [`src/graphs/graph.py`](src/graphs/graph.py) | `Graph`, covering directed/undirected and weighted/unweighted in one class. Adjacency list is the primary store; `to_adjacency_matrix()` returns the second representation, backed by numpy. |
| [`src/graphs/bfs.py`](src/graphs/bfs.py) | `bfs`, `bfs_component`, `bfs_levels`, `bfs_tree`. An explicit `collections.deque`. |
| [`src/graphs/dfs.py`](src/graphs/dfs.py) | `dfs`, `dfs_iterative`, `dfs_recursive`, `dfs_component`. Both forms return identical orders. |
| [`src/graphs/dijkstra.py`](src/graphs/dijkstra.py) | `dijkstra`, `shortest_path`, `reconstruct_path`, and `dijkstra_linear_scan` as the O(V^2) baseline. Built on the Week 3 `PriorityQueue`, not `heapq`. |

Graphs for benchmarks and demos come from
[`src/utils/graph_generator.py`](src/utils/graph_generator.py), seeded at 42.

```python
from src.graphs.bfs import bfs, bfs_levels
from src.graphs.dfs import dfs_iterative, dfs_recursive
from src.graphs.dijkstra import dijkstra, shortest_path
from src.graphs.graph import Graph

graph = Graph()                      # undirected, unweighted
for u, v in [(1, 2), (1, 3), (2, 4), (3, 4)]:
    graph.add_edge(u, v)             # add_edge creates missing nodes
graph.get_neighbors(1)               # [2, 3]
bfs(graph, 1)                        # [1, 2, 3, 4] - every component
bfs_levels(graph, 1)                 # {1: 0, 2: 1, 3: 1, 4: 2}
dfs_iterative(graph, 1) == dfs_recursive(graph, 1)      # True

dense = graph.to_adjacency_matrix()  # numpy uint8, rows in insertion order
dense.matrix.nbytes                  # 16

roads = Graph(weighted=True)
for u, v, w in [("A", "B", 4), ("A", "C", 2), ("C", "B", 1)]:
    roads.add_edge(u, v, w)
distances, predecessors = dijkstra(roads, "A")          # {'A': 0.0, 'B': 3.0, 'C': 2.0}
shortest_path(roads, "A", "B")                          # (['A', 'C', 'B'], 3.0)
```

### Running the demonstration

```bash
python examples/week4_demo.py
```

Five sections on graphs small enough to read: the four kinds of graph, both
representations agreeing on the same graph, BFS and DFS differing by nothing
but their container, Dijkstra on a hand-checkable graph including the
unreachable and negative-weight cases, and all three algorithms on one
graph. Every claim it prints is asserted. It runs in about 0.6 seconds.

### Running the benchmark

```bash
python benchmarks/week4_graph_benchmark.py
```

Four studies: adjacency list against matrix for memory and edge-lookup cost
at V = 100, 1,000 and 10,000; BFS against both DFS forms on sparse and dense
graphs; Dijkstra on the Week 3 heap against an O(V^2) linear scan; and the
drawn traversal figures. The full run takes about 9 seconds on an M2 Max and
writes `bfs_vs_dfs_sparse.png`, `bfs_vs_dfs_dense.png`,
`dijkstra_performance.png` and `comparison_table.csv`, plus the supporting
CSVs and figures, to [`benchmarks/results/`](benchmarks/results/).

Both runtime limits are handled by reducing and recording rather than
dropping a point: a matrix past a 512 MB cap is skipped with the reason
written into the CSV, and the linear-scan Dijkstra is capped at V = 2,000
while the heap runs to 8,000.

For a smoke run, send it somewhere else so it does not overwrite the
committed results:

```bash
python benchmarks/week4_graph_benchmark.py --quick --out /tmp/week4_quick
```

Every figure quoted in the Week 4 report is recomputed from the CSVs by:

```bash
python tools/week4_facts.py
```

---

## Week 3: data structures

No new dependencies. The three structures use only the standard library;
the benchmark uses numpy, matplotlib and pandas from `requirements.txt`.

### The three modules

| Module | Classes | Operations |
|---|---|---|
| [`src/structures/heap.py`](src/structures/heap.py) | `MinHeap`, `MaxHeap`, `PriorityQueue` | `insert`, `extract_min` / `extract_max`, `heapify` (bottom-up, O(n)), `peek`, `is_empty`; `push` and `pop` on the queue |
| [`src/structures/avl_tree.py`](src/structures/avl_tree.py) | `AVLTree` | `insert`, `delete` (rebalances every ancestor), `search`, `in_order`, `height`, `validate` |
| [`src/structures/hash_table.py`](src/structures/hash_table.py) | `HashTable` | `insert`, `get`, `delete`, `load_factor`, automatic rehash; `strategy="chaining"` (default) or `"linear_probing"` |

All three are importable from the package:

```python
from src.structures import AVLTree, HashTable, MinHeap, PriorityQueue

heap = MinHeap([5, 3, 8, 1])        # built with heapify, O(n)
heap.insert(2)
heap.extract_min()                  # 1

queue = PriorityQueue()             # stable for equal priorities
queue.push("write report", 2)
queue.push("fix failing test", 1)
queue.pop()                         # 'fix failing test'

tree = AVLTree()
for key in range(1, 16):
    tree.insert(key, f"value-{key}")
tree.height()                       # 4: sorted input stays balanced
tree.delete(8)
tree.search(11)                     # 'value-11'

table = HashTable(strategy="linear_probing")
table.insert("heap", 4)
table.get("heap")                   # 4
table.delete("heap")                # leaves a tombstone, so probe runs stay intact
table.load_factor                   # 0.0
```

### Running the demonstration

```bash
python examples/week3_demo.py
```

Five short sections on inputs small enough to read: heaps and why
`heapify` is O(n), a priority queue that stays stable and never compares
its payloads, an AVL tree staying perfectly balanced on sorted input, both
hash-table strategies with a look inside a linear-probing run and why
deletion there needs tombstones, and all three structures agreeing on the
same keys. Every claim it prints is also asserted. It runs in well under a
second.

### Running the benchmark

```bash
python benchmarks/week3_structures_benchmark.py
```

Times insert, search and delete at n = 10³, 10⁴, 10⁵ and 10⁶ for the heap
(against `heapq`), the AVL tree (against `dict`, with a `list` baseline)
and both hash-table strategies, with 5, 5, 3 and 1 measured runs per size
after discarded warm-ups. It then runs four supporting studies: probes and
lookup time against load factor, an engineered all-colliding key set,
the per-insert cost of rehashing, and AVL height against its bounds. The
full run takes about two minutes on an M2 Max, and writes the three
required charts (`heap_performance.png`, `tree_performance.png`,
`hash_performance.png`), `structures_comparison_table.csv` (asymptotic against
empirical, with run counts) and the supporting CSVs and charts to
[`benchmarks/results/`](benchmarks/results/).

For a smoke run of about ten seconds, use `--quick`, and send it somewhere
else so it does not overwrite the committed full results:

```bash
python benchmarks/week3_structures_benchmark.py --quick --out /tmp/week3_quick
```

Every figure quoted in the Week 3 report can be recomputed from the result
CSVs:

```bash
python tools/week3_facts.py
```

---

## Running the Week 2 demonstration

```bash
python examples/week2_demo.py
```

Walks through merge sort and quicksort on inputs small enough to read: the
edge cases, the non-destructive contract, non-integer element types,
three-way partitioning on duplicate-heavy data, stability, and a
cross-check that all five algorithms agree. Every claim it prints is also
asserted, so it exits non-zero if any of them stops holding.

## Running the Week 1 benchmarks

```bash
python benchmarks/sorting_benchmarks.py
```

Sweeps all three algorithms across sizes 100 / 500 / 1,000 / 5,000 /
10,000 and five data types (random, sorted, reverse, nearly sorted,
duplicates), five measured runs each after two discarded warm-up runs.
It writes the charts to `docs/figures/` and the full results table to
`benchmarks/results/`.

**This takes a while.** Bubble and selection sort at n = 10,000 are
quadratic in pure Python; the full sweep runs for tens of minutes. That
cost is itself one of the report's findings. Use `--quick` for a fast
smoke run over the smaller sizes:

```bash
python benchmarks/sorting_benchmarks.py --quick
```

---

## Where the analysis lives

**[`analysis/week3_report.md`](analysis/week3_report.md)** - the Week 3
report: heaps against `heapq`, the AVL tree against `dict`, chaining
against linear probing, and the amortised cost of rehashing.

**[`analysis/week2_report.md`](analysis/week2_report.md)** - the Week 2
report on merge sort and quicksort, with the Master Theorem solutions in
[`analysis/week2_recurrences.md`](analysis/week2_recurrences.md).

**[`docs/performance_analysis.md`](docs/performance_analysis.md)** - the
Week 1 report: methodology, the measured results with their standard
deviations, the empirical complexity fits, and the conclusions drawn from
them. Charts are in [`docs/figures/`](docs/figures/) and the raw
measurements in [`benchmarks/results/`](benchmarks/results/).

Every number in the Week 1 report - every table cell and every figure
quoted in the prose - is computed from the result CSVs by
[`tools/build_report.py`](tools/build_report.py), so the write-up cannot
drift out of step with the data after a re-run:

```bash
python benchmarks/sorting_benchmarks.py     # measure
python tools/build_report.py                # write the report from the measurements
```

**[`SUBMISSION.md`](SUBMISSION.md)** - the Week 1 cover document: what was
built, where each piece lives, the actual environment and test output, and
the headline benchmark findings.

---

## Use of AI on this project

I used Anthropic's Claude (Claude Code) as an assistant on this project,
working from a detailed written specification of the requirements that I
prepared from the assignment instructions and the course reading. The
model drafted the source files, the test suite and the first draft of the
written documents, Week 4's graph algorithms included; I specified the
requirements, directed the work,
reviewed the output, ran the benchmarks, and am responsible for what is
submitted here. No text was copied from an AI site and presented as a
quotation.

A full, file-by-file description is in
**[`docs/AI_USE.md`](docs/AI_USE.md)**, and the same disclosure appears at
the end of `SUBMISSION.md`, as the course policy requires.
