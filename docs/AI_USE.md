# Use of AI on this project

**Robert Deibel** · CSC 5300 Advanced Algorithms · Concordia University Texas

Covers every assignment in this repository. Week 1 is documented first,
Week 2 below it.

Concordia University Texas's policy requires that any use of AI be
acknowledged, that text copied directly from an AI tool be treated and
cited as a direct quote, and that other uses be clearly described. This
document is that description. It is written to be specific enough that a
reader can tell exactly which parts of the project the tool produced.

---

# Week 1 — Algorithm Laboratory Setup

## Tool used

**Anthropic Claude, via the Claude Code command-line agent**, run locally
on my machine during the development of this assignment. No other AI tool
was used.

## How it was used

I wrote a detailed specification of the assignment's requirements —
derived from the Blackboard assignment instructions, the Detailed Grading
Criteria, and Chapter 1 §§1.6–1.10 of Amakobe, *Advanced Computational
Algorithms* (2nd ed., 2026) — and directed the tool to implement the
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
machine — an Apple M2 Max under macOS 14.5 and Python 3.12.4. The full
study took 187 seconds. No result was estimated, adjusted, or invented,
and the report is written from the output of that run.

To make that verifiable rather than merely asserted, every number in
`docs/performance_analysis.md` is computed from the result CSVs by
`tools/build_report.py` at render time. None of the figures in the report
were typed in by hand, by me or by the model, so none of them can have
been fabricated — re-running the two commands regenerates the document
from the data.

**The judgement calls.** Which findings the report is entitled to claim,
and how strongly, is mine. One example is worth naming: the report's §3.4
concludes that insertion sort does **not** become linear on this
project's `nearly_sorted` data — it wins a large constant factor and stays
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
  §§1.6–1.10 — the source of the required project structure, the
  `AlgorithmBenchmark` interface, and the package list.
* Standard library and package documentation for Python 3.12, pytest,
  matplotlib, scipy and pandas.

## Why I am comfortable submitting this

The learning objectives of this assignment are the empirical method — set
up a laboratory, measure real algorithms, and draw conclusions that the
data supports. I directed that process, ran it, checked it, and made the
calls about what the results mean. The specific finding in §3.4 came out
of examining what the numbers said and asking why they disagreed with the
expectation, which is the part of the work that was worth doing.

---

# Week 2 — Divide and Conquer Implementation Project

## Tool used

The same: **Anthropic Claude, via the Claude Code command-line agent**, run
locally. No other AI tool was used.

## How it was used

The same working method as Week 1, and the same division of labour. I
wrote a specification of the requirements from the Blackboard instructions,
Dr. Amakobe's Week 2 announcement, the Week 2 Plan and the assigned reading
(CLRS Ch. 2.3–2.4 and Ch. 4, Amakobe Ch. 2), and directed the tool to
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
is able to show the difference — a measured 245× gap at n=16,000 — instead
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
  ed. — Ch. 2.3–2.4 (merge sort), Ch. 4 (the Master Theorem), Ch. 7
  (quicksort, and the Hoare partition problem).
* Amakobe, *Advanced Computational Algorithms*, 2nd ed., Ch. 2.
* Python, pytest, matplotlib and pandas documentation.
