# Use of AI on this project

**Robert Deibel** · CSC 5300 Advanced Algorithms · Week 1 Project — Algorithm Laboratory Setup

Concordia University Texas's policy requires that any use of AI be
acknowledged, that text copied directly from an AI tool be treated and
cited as a direct quote, and that other uses be clearly described. This
document is that description. It is written to be specific enough that a
reader can tell exactly which parts of the project the tool produced.

---

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
