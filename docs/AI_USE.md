# Use of AI on this project

**Robert Deibel** · CSC 5300 Advanced Algorithms · Concordia University Texas

Covers every assignment in this repository, in order: Week 1 through
Week 6.

Concordia University Texas's policy requires that any use of AI be
acknowledged, that text copied directly from an AI tool be treated and
cited as a direct quote, and that other uses be clearly described. This
document is that description. It is written to be specific enough that a
reader can tell exactly which parts of the project the tool produced.

---

# Week 1 - Algorithm Laboratory Setup

## Tool used

**Anthropic Claude, via the Claude Code command-line agent**, run locally
on my machine during the development of this assignment. No other AI tool
was used.

## How it was used

I wrote a detailed specification of the assignment's requirements -
derived from the Blackboard assignment instructions, the Detailed Grading
Criteria, and Chapter 1 §§1.6-1.10 of Amakobe, *Advanced Computational
Algorithms* (2nd ed., 2026) - and directed the tool to implement the
project against it. I reviewed the output, ran the tests and benchmarks
myself, and made the decisions about scope, structure and what the report
should claim.

The model did the drafting. I did the specifying, the directing, the
reviewing, and the deciding. I am responsible for everything submitted
here.

## What was AI-generated

Essentially all of the code and the first draft of the prose was written
by the model against my specification:

| File or directory | Status |
|---|---|
| `src/sorting/basic_sorts.py` | AI-drafted, reviewed by me |
| `src/utils/benchmark.py` | AI-drafted, reviewed by me |
| `src/utils/visualization.py` | AI-drafted, reviewed by me |
| `src/utils/testing_helpers.py` | AI-drafted, reviewed by me |
| `tests/` (all files) | AI-drafted, reviewed by me |
| `benchmarks/sorting_benchmarks.py` | AI-drafted, reviewed by me |
| `tools/build_report.py` | AI-drafted, reviewed by me |
| `check_environment.py`, `setup.py`, `.github/workflows/tests.yml` | AI-drafted, reviewed by me |
| All `__init__.py` files and placeholder modules | AI-drafted, reviewed by me |
| `README.md`, `docs/performance_analysis.md`, `SUBMISSION.md`, this file | AI-drafted, edited by me |
| Git commit messages | AI-drafted |

## What was *not* AI-generated

**The measurements.** Every timing, standard deviation, complexity fit,
comparison count and chart in `benchmarks/results/` and `docs/figures/`
was produced by actually running `benchmarks/sorting_benchmarks.py` on my
machine - an Apple M2 Max under macOS 14.5 and Python 3.12.4. The full
study took 187 seconds. No result was estimated, adjusted, or invented,
and the report is written from the output of that run.

To make that verifiable rather than merely asserted, every number in
`docs/performance_analysis.md` is computed from the result CSVs by
`tools/build_report.py` at render time. None of the figures in the report
were typed in by hand, by me or by the model, so none of them can have
been fabricated - re-running the two commands regenerates the document
from the data.

**The judgement calls.** Which findings the report is entitled to claim,
and how strongly, is mine. One example is worth naming: the report's §3.4
concludes that insertion sort does **not** become linear on this
project's `nearly_sorted` data - it wins a large constant factor and stays
quadratic. That is contrary to the usual textbook shorthand, and it is
what the measured exponent and doubling ratio actually show for this
particular definition of "nearly sorted". Reporting the measurement rather
than the expectation was a deliberate decision.

## Direct quotation

**None.** No text produced by the AI tool is presented in this submission
as a quotation from a source, and no text from any source is reproduced
verbatim without attribution. The prose in the README and the reports was
drafted by the model as original writing for this assignment and edited by
me, which the table above discloses; it is not quoted material.

## Sources other than AI

* Blackboard: the CSC 5300 Week 1 assignment instructions, submission
  checklist and Detailed Grading Criteria.
* Amakobe, *Advanced Computational Algorithms*, 2nd ed. (2026), Chapter 1
  §§1.6-1.10 - the source of the required project structure, the
  `AlgorithmBenchmark` interface, and the package list.
* Standard library and package documentation for Python 3.12, pytest,
  matplotlib, scipy and pandas.

## Why I am comfortable submitting this

The learning objectives of this assignment are the empirical method - set
up a laboratory, measure real algorithms, and draw conclusions that the
data supports. I directed that process, ran it, checked it, and made the
calls about what the results mean. The specific finding in §3.4 came out
of examining what the numbers said and asking why they disagreed with the
expectation, which is the part of the work that was worth doing.

---

# Week 2 - Divide and Conquer Implementation Project

## Tool used

The same: **Anthropic Claude, via the Claude Code command-line agent**, run
locally. No other AI tool was used.

## How it was used

The same working method as Week 1, and the same division of labour. I
wrote a specification of the requirements from the Blackboard instructions,
Dr. Amakobe's Week 2 announcement, the Week 2 Plan and the assigned reading
(CLRS Ch. 2.3-2.4 and Ch. 4, Amakobe Ch. 2), and directed the tool to
implement against it. The model drafted the code, the tests and the first
draft of the written documents; I set the requirements, made the design
decisions, ran the benchmarks and decided what the results support.

## What was AI-generated

| File | Status |
|---|---|
| `src/sorting/merge_sort.py` | AI-drafted, reviewed by me |
| `src/sorting/quick_sort.py` | AI-drafted, reviewed by me |
| `tests/test_merge_sort.py`, `tests/test_quick_sort.py`, `tests/test_sorting_comparison.py` | AI-drafted, reviewed by me |
| `benchmarks/week2_performance.py` | AI-drafted, reviewed by me |
| `tools/week2_facts.py` | AI-drafted, reviewed by me |
| The two new data generators in `src/utils/benchmark.py` | AI-drafted, reviewed by me |
| `analysis/week2_report.md`, `analysis/week2_recurrences.md` | AI-drafted, edited by me |
| Git commit messages | AI-drafted |

## What was *not* AI-generated

**The measurements.** Every timing, doubling ratio, growth rate and chart
in `benchmarks/results/` came from actually running
`benchmarks/week2_performance.py` on my machine. Nothing was estimated or
adjusted. As in Week 1, the figures quoted in the report are computed from
the result CSVs by `tools/week2_facts.py` rather than transcribed, so they
can be re-checked against a fresh run in one command.

**The engineering judgment.** Three decisions in this week's work were
mine, and each changed the outcome:

*Hoare's partition scheme rather than Lomuto's.* The textbook default is
Lomuto, and it would have been the obvious thing to write. It is also
quadratic on duplicate-heavy input, and two of this assignment's six
required data types have only 10 and 3 distinct values. I had the Lomuto
version implemented as well and benchmarked both, which is how the report
is able to show the difference - a measured 245× gap at n=16,000 - instead
of merely claiming one.

*Recursing into the smaller partition and looping on the larger.* Python's
recursion limit is 1000 and this project sorts 50,000 elements, so a
textbook two-sided recursion would have raised `RecursionError` on sorted
input. Raising the limit would have hidden the problem rather than fixed
it.

*Measuring the capped cells instead of only projecting them.* The
assignment's guidance was to cap the O(n²) algorithms and extrapolate. My
own projection put the omitted cells at roughly an hour of compute, not the
much larger figure I had assumed, so I ran them and checked the projection
against reality. Reporting the extrapolation *and* its verification is
worth more than either alone.

## Direct quotation

**None.** No text produced by the AI tool is presented as a quotation from
a source, and no text from any source is reproduced verbatim without
attribution.

## Sources other than AI

* The Blackboard Week 2 assignment instructions, the Week 2 Plan, and
  Dr. Amakobe's Week 2 announcement of 31 August 2026.
* Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, 4th
  ed. - Ch. 2.3-2.4 (merge sort), Ch. 4 (the Master Theorem), Ch. 7
  (quicksort, and the Hoare partition problem).
* Amakobe, *Advanced Computational Algorithms*, 2nd ed., Ch. 2.
* Python, pytest, matplotlib and pandas documentation.

---

# Week 3 - Data Structures

## Tool used

The same: **Anthropic Claude, via the Claude Code command-line agent**, run
locally. No other AI tool was used.

## How it was used

The same working method and division of labour as Weeks 1 and 2. I wrote a
specification from the Blackboard Week 3 instructions and their Submission
Checklist, and directed the tool to implement against it. The specification
fixed the design points that decide correctness - bottom-up `heapify`, a
counter as the priority queue's tie-breaker, rebalancing every ancestor on
AVL deletion, tombstones for linear-probing deletion, separate resize
thresholds for the two hash strategies - and the argument the report is
built around. The model drafted the code, the tests and the first draft of
the report; I reviewed them and decided what the results support.

## What was AI-generated

| File | Status |
|---|---|
| `src/structures/heap.py`, `avl_tree.py`, `hash_table.py` | AI-drafted, reviewed by me |
| `tests/test_heap.py`, `test_avl_tree.py`, `test_hash_table.py`, `test_data_structure_comparison.py`, and the `conftest.py` additions | AI-drafted, reviewed by me |
| `time_operation` in `src/utils/benchmark.py`, and its tests | AI-drafted, reviewed by me |
| `benchmarks/week3_structures_benchmark.py` | AI-drafted, reviewed by me |
| `examples/week3_demo.py`, `tools/week3_facts.py`, `tools/build_week3_pdf.py` | AI-drafted, reviewed by me |
| `analysis/week3_report.md`, the README's Week 3 section | AI-drafted, edited by me |
| Git commit messages | AI-drafted |

## What was *not* AI-generated

**The measurements.** Every timing, probe count and chart this week in
`benchmarks/results/` came from running
`benchmarks/week3_structures_benchmark.py` on my machine; the full study
took 121 seconds. Nothing was estimated or adjusted. The report's figures
are printed from the result CSVs by `tools/week3_facts.py`, so they can be
re-checked against a fresh run in one command.

**What the report claims.** Two findings ran against the argument the
report was planned around, and both are reported as measured rather than
bent to fit. The AVL tree's search gap against `dict` turned out to be
almost exactly log₂ n - an asymptotic difference, not the constant factor
I expected. And seven O(1) hash-table series classify as O(log n) because
of a step at 10⁶; the report keeps the classifier's verdict and shows,
with operation counts, that the step is memory rather than work.

**A bug the tests caught.** The first `AVLTree` constructor treated any
2-tuple in its input as a (key, value) pair, so tuple keys were silently
misread. The test suite caught it, and the fix decides by the input's type:
a mapping supplies pairs, anything else supplies keys.

## Direct quotation

**None.** No text produced by the AI tool is presented as a quotation from
a source, and no text from any source is reproduced verbatim without
attribution.

## Sources other than AI

* The Blackboard Week 3 assignment instructions and Submission Checklist.
* Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, 4th
  ed. - Ch. 6 (heaps), Ch. 11 (hash tables), Ch. 12-13 (search trees; AVL
  trees are Problem 13-3), §16.4 (dynamic tables).
* Amakobe, *Advanced Computational Algorithms*, 2nd ed., Ch. 3.
* Knuth, *The Art of Computer Programming*, Vol. 3, §6.4, for the expected
  probe counts under linear probing.
* Python, pytest, matplotlib and pandas documentation.

---

# Week 4 - Graph Algorithms

## Tool used

The same: **Anthropic Claude, via the Claude Code command-line agent**, run
locally. No other AI tool was used.

## How it was used

The same method and division of labour as Weeks 1 through 3. I wrote a
specification from the Blackboard Week 4 instructions and their Submission
Checklist and directed the tool to implement against it. The specification
fixed the decisions that decide correctness: one Graph class covering all
four directed/weighted combinations, a numpy-backed adjacency matrix rather
than nested lists, DFS in both iterative and recursive form, Dijkstra built
on the Week 3 priority queue rather than heapq, and the argument the report
is built around. The model drafted the code, the tests and the first draft
of the report; I reviewed them and decided what the results support.

## What was AI-generated

| File | Status |
|---|---|
| `src/graphs/graph.py`, `bfs.py`, `dfs.py`, `dijkstra.py` | AI-drafted, reviewed by me |
| `src/utils/graph_generator.py`, and the traversal figures added to `src/utils/visualization.py` | AI-drafted, reviewed by me |
| `tests/test_graph_representation.py`, `test_bfs.py`, `test_dfs.py`, `test_dijkstra.py`, `test_graph_benchmark.py` | AI-drafted, reviewed by me |
| `benchmarks/week4_graph_benchmark.py` | AI-drafted, reviewed by me |
| `examples/week4_demo.py`, `tools/week4_facts.py`, `tools/build_week4_pdf.py` | AI-drafted, reviewed by me |
| `analysis/week4_report.md`, the README's Week 4 section | AI-drafted, edited by me |
| Git commit messages | AI-drafted |

## What was *not* AI-generated

**The measurements.** Every timing, memory figure and chart this week came
from running `benchmarks/week4_graph_benchmark.py` on my machine; the full
study takes 9.0 seconds. Nothing was estimated or adjusted, and the
report's figures are printed from the result CSVs by
`tools/week4_facts.py`, so they can be re-checked against a fresh run in
one command.

**The engineering judgment.** Three decisions this week were mine and each
changed the result. Backing the adjacency matrix with numpy rather than
nested Python lists is what made V = 10,000 measurable at all: 10^8 cells
as `uint8` is 100 MB, and as nested lists it is gigabytes. Capping the
O(V^2) linear-scan Dijkstra at V = 2,000 while letting the heap version run
to 8,000, and saying plainly in the report that the two series cover
different ranges, follows the rule set in Week 1: reduce and document,
never silently drop a required data point. And reporting the two
classification disagreements as measured, rather than quietly relabelling
them, is the same call I made in Week 3.

**What the report claims.** The report's thesis is that the container
decides which traversal you have, and it names the two places that breaks
rather than hiding them. The second one is a real limit: the Week 3
priority queue has no decrease-key, so Dijkstra uses lazy deletion and the
heap holds O(E) entries rather than O(V). That is stated in the docstring
and in the report because it changes the analysis.

## Direct quotation

**None.** No text produced by the AI tool is presented as a quotation from
a source, and no text from any source is reproduced verbatim without
attribution.

## Sources other than AI

* The Blackboard Week 4 assignment instructions and Submission Checklist.
* Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, 4th
  ed. - Ch. 22, Elementary Graph Algorithms, pp. 589-619, and Ch. 24.3,
  pp. 658-666.
* Dijkstra, E. W. (1959). "A Note on Two Problems in Connexion with
  Graphs." *Numerische Mathematik* 1(1), 269-271.
* Amakobe, *Advanced Computational Algorithms*, 2nd ed.
* networkx, numpy, matplotlib, pytest and Python documentation.

---

# Week 5 - Dynamic Programming

## Tool used

The same: **Anthropic Claude, via the Claude Code command-line agent**, run
locally. No other AI tool was used.

## How it was used

The same method and division of labour as Weeks 1 through 4. I wrote a
specification from the Blackboard Week 5 instructions and their Submission
Checklist and directed the tool to implement against it. The specification
fixed the decisions that decide correctness: three problems each solved
three ways, a hand-written dict memo rather than an lru_cache decorator,
call counting through a counter object rather than a module global,
trace_solution and lcs_reconstruct written even where the instructions call
reconstruction optional, and the argument the report is built around. The
model drafted the code, the tests and the first draft of the report; I
reviewed them and decided what the results support.

## What was AI-generated

| File | Status |
|---|---|
| `src/dp/fibonacci.py`, `knapsack.py`, `lcs.py` | AI-drafted, reviewed by me |
| `src/utils/timer.py`, and the three Week 5 figures appended to `src/utils/visualization.py` | AI-drafted, reviewed by me |
| `tests/test_fibonacci.py`, `test_knapsack.py`, `test_lcs.py`, `test_dp_benchmark.py` | AI-drafted, reviewed by me |
| `benchmarks/week5_dp_benchmark.py` | AI-drafted, reviewed by me |
| `examples/week5_demo.py`, `tools/week5_facts.py`, `tools/build_week5_pdf.py` | AI-drafted, reviewed by me |
| `analysis/week5_report.md`, the README's Week 5 section | AI-drafted, edited by me |
| Git commit messages | AI-drafted |

## What was *not* AI-generated

**The measurements.** Every timing, call count, memory figure and chart
this week came from running `benchmarks/week5_dp_benchmark.py` on my
machine; the full study takes 60 seconds. Nothing was estimated except the
two Fibonacci points explicitly labelled as projections, and those are
labelled as such in the CSV, drawn differently in the figure, and explained
in the report. The report's figures are printed from the result CSVs by
`tools/week5_facts.py`, so they can be re-checked against a fresh run in
one command.

**The engineering judgment.** Three decisions this week were mine and each
changed what the study could show. Measuring naive Fibonacci to n = 35 and
projecting 40 and 45 from the measured per-call cost times the exact
closed-form call count, rather than either running it for most of an hour
or quietly dropping the two largest points, follows the rule set in Week 1:
reduce and document. Leaving `speedup_vs_recursive` blank wherever the
baseline was projected keeps a ratio of a measurement to an estimate out of
a column that otherwise reports measurements. And probing where memoized
LCS fails at CPython's default recursion limit, before raising that limit
for the main run, turned the report's central claim about which stack each
approach spends into a measured threshold of 600 characters rather than an
assertion.

**What the report claims.** The report argues that memoization and
tabulation differ only in who decides the order of the subproblems, and it
names the place that argument breaks rather than hiding it. Knapsack is the
honest counterexample: top-down was faster while the subproblem space was
sparse, at 5.0% table occupancy and 10 items, and lost once it densified.
Both outcomes are reported as measured.

## Direct quotation

**None.** No text produced by the AI tool is presented as a quotation from
a source, and no text from any source is reproduced verbatim without
attribution.

## Sources other than AI

* The Blackboard Week 5 assignment instructions and Submission Checklist.
* Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, 4th
  ed., Ch. 14: §14.1 Rod Cutting, pp. 360-369; §14.3 Elements of Dynamic
  Programming, pp. 378-386; §14.4 Longest Common Subsequence, pp. 386-392.
* Bellman, R. (1966). "Dynamic Programming." *Science*, 153(3731), 34-37.
* Needleman, S. B., and Wunsch, C. D. (1970). *Journal of Molecular
  Biology*, 48(3), 443-453.
* Amakobe, M. (2025). *Advanced Computational Algorithms*, 2nd ed., Ch. 6.
* Python, pytest, matplotlib and pandas documentation.

---

# Week 6 - Dynamic Programming II

## Tool used

The same: **Anthropic Claude, via the Claude Code command-line agent**, run
locally. No other AI tool was used.

## How it was used

The same method and division of labour as Weeks 1 through 5. I wrote a
specification from the Blackboard Week 6 instructions, the Week 6 Plan and
the readings, including every hand-checked anchor value the tests assert
against, and directed the tool to implement against it. The specification
fixed the decisions that decide correctness: Week 6 code in its own
package, Week 5's knapsack reused as the baseline rather than rewritten, a
deliberately wrong upward-scan knapsack to demonstrate why the scan runs
downward, a hand-written memo for matrix chains, Week 4's Dijkstra reused
unchanged for the comparison, and the two-pairs organization of the report.
The model drafted the code, the tests and the first draft of the report; I
reviewed them and decided what the results support.

## What was AI-generated

| File | Status |
|---|---|
| `src/dp_advanced/` (all four modules and the package init) | AI-drafted, reviewed by me |
| `src/utils/matrix_utils.py`, `src/utils/bitmask_utils.py`, and the Week 6 figures appended to `src/utils/visualization.py` | AI-drafted, reviewed by me |
| `tests/test_space_optimized_knapsack.py`, `test_matrix_chain_multiplication.py`, `test_floyd_warshall.py`, `test_bitmask_tsp.py`, `test_benchmark_comparison.py` | AI-drafted, reviewed by me |
| `benchmarks/week6_dp_advanced_benchmark.py` | AI-drafted, reviewed by me |
| `examples/week6_demo.py`, `tools/week6_facts.py`, `tools/build_week6_pdf.py` | AI-drafted, reviewed by me |
| `analysis/week6_report.md`, the README's Week 6 section | AI-drafted, edited by me |
| Git commit messages | AI-drafted |

## What was *not* AI-generated

**The measurements.** Every timing, memory figure and chart this week came
from running `benchmarks/week6_dp_advanced_benchmark.py` on my machine; the
full study takes about three minutes. Every required size was measured,
none projected. The report's figures are printed from the result CSVs by
`tools/week6_facts.py`, so they can be re-checked against a fresh run in one
command.

**The anchor values.** The expected answers the tests assert - the six
matrix-chain costs, the CLRS all-pairs distance matrix, the four-city tour
cost and the 7-against-9 knapsack proof case - were worked out and checked
before any code was written, so the tests check the code against known
answers rather than against its own output.

**What the report claims.** The report organizes the four algorithms as two
pairs and states where the measurements depart from the textbook bounds,
including one place the measurement corrected my own benchmark: plain
matrix-chain recursion grew by a factor of three per added matrix, which
matches its exact 3^(n-1) call count, so the benchmark's theory label was
changed from the Catalan count of parenthesizations to O(3^n) before the
final run.

## Direct quotation

**None.** No text produced by the AI tool is presented as a quotation from
a source, and no text from any source is reproduced verbatim without
attribution.

## Sources other than AI

* The Blackboard Week 6 assignment instructions, Week 6 Plan and Readings.
* Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, 4th
  ed., §14.2 Matrix-Chain Multiplication and §23.2 The Floyd-Warshall
  algorithm.
* Amakobe, M. (2025). *Advanced Computational Algorithms*, Ch. 6, §6.5,
  §6.7 and §6.8.
* Held, M., and Karp, R. M. (1962). *Journal of the Society for Industrial
  and Applied Mathematics*, 10(1), 196-210.
* Hirschberg, D. S. (1975). *Communications of the ACM*, 18(6), 341-343.
* Python, pytest, numpy and matplotlib documentation.
