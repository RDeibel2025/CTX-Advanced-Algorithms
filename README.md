# CSC 5300 Advanced Algorithms - Course Project

**Robert Deibel**  
Concordia University Texas · CSC 5300 Advanced Algorithms · Fall 2026

Repository: <https://github.com/RDeibel2025/CTX-Advanced-Algorithms>

| Week | Assignment | Deliverable |
|---|---|---|
| 1 | Algorithm Laboratory Setup | [`docs/performance_analysis.md`](docs/performance_analysis.md) |
| 2 | Divide and Conquer | [`analysis/week2_report.md`](analysis/week2_report.md) · [`analysis/week2_recurrences.md`](analysis/week2_recurrences.md) |

---

## What this project is

A working laboratory for measuring how sorting algorithms actually behave,
rather than only reasoning about how they should behave.

It contains three things:

1. **Five sorting algorithms** - an optimized bubble sort, selection sort
   and insertion sort (Week 1), plus merge sort and a randomized quicksort
   (Week 2) - all written to a single explicit contract: reject non-list
   input, never mutate the caller's list, handle empty and single-element
   input, and work on any comparable element type. Because they share that
   contract they are interchangeable in the benchmark harness, which
   [`tests/test_sorting_comparison.py`](tests/test_sorting_comparison.py)
   verifies.
2. **A benchmarking framework** that generates eight different shapes of
   input, times an algorithm over repeated runs with `time.perf_counter`,
   reports mean, standard deviation, minimum and maximum, fits the
   measurements against O(n), O(n log n) and O(n²) reference models, plots
   the comparison, and stores the results as CSV for later retrieval.
3. **A performance study** built from a real run of that framework -
   charts, a results table, and an analysis of what the measurements
   actually show about each algorithm's growth rate and its sensitivity to
   the order of its input.

The remaining directories (`src/searching/`, `src/graph/`,
`src/dynamic_programming/`, `src/data_structures/`) are the semester's
scaffolding. They are real Python packages with documented placeholders,
ready for later weeks.

---

## Project structure

```
Advanced Algorithms/
├── README.md                       This file
├── SUBMISSION.md                   Cover document for the Blackboard submission
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
│   ├── searching/                  Reserved for a later week
│   ├── graph/                      Reserved for a later week
│   ├── dynamic_programming/        Reserved for a later week
│   ├── data_structures/            Reserved for a later week
│   └── utils/
│       ├── benchmark.py            BenchmarkResult, AlgorithmBenchmark
│       ├── visualization.py        Supplementary charts
│       └── testing_helpers.py      Shared predicates and test fixtures
├── tests/
│   ├── conftest.py                 Fixtures: sample_arrays, large_random_array
│   ├── test_sorting.py             Week 1 sorting algorithm tests
│   ├── test_merge_sort.py          Merge sort tests
│   ├── test_quick_sort.py          QuickSort tests
│   ├── test_sorting_comparison.py  All five algorithms must agree
│   ├── test_searching.py           Reserved for a later week
│   └── test_utils.py               Benchmarking framework tests
├── benchmarks/
│   ├── sorting_benchmarks.py       The Week 1 end-to-end benchmark driver
│   ├── week2_performance.py        The Week 2 divide-and-conquer benchmark
│   ├── complexity_validation.py    Reserved for a later week
│   └── results/                    Week 2 charts and measurements
├── analysis/
│   ├── week2_report.md             Week 2 technical report
│   └── week2_recurrences.md        Master Theorem solutions
├── docs/
│   ├── performance_analysis.md     The performance report (generated)
│   ├── AI_USE.md                   AI use disclosure
│   └── figures/                    Generated charts (PNG)
├── tools/
│   ├── build_report.py             Renders the report from the result CSVs
│   ├── md_to_pdf.py                Exports SUBMISSION.md to PDF
│   └── package_submission.sh       Builds the submission zip and PDF
├── submissions/
│   └── week-01-algorithm-lab/      What was handed in for Week 1
├── notebooks/                      Reserved for exploratory work
└── examples/                       Reserved for worked examples
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
scipy, scikit-learn, plotly and seaborn. `requirements.txt` also pins
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

The docstring examples are executable too:

```bash
pytest --doctest-modules src/
```

---

## Running the benchmarks

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

**[`docs/performance_analysis.md`](docs/performance_analysis.md)** - the
full report: methodology, the measured results with their standard
deviations, the empirical complexity fits, and the conclusions drawn from
them. Charts are in [`docs/figures/`](docs/figures/) and the raw
measurements in [`benchmarks/results/`](benchmarks/results/).

Every number in that report - every table cell and every figure quoted in
the prose - is computed from the result CSVs by
[`tools/build_report.py`](tools/build_report.py), so the write-up cannot
drift out of step with the data after a re-run:

```bash
python benchmarks/sorting_benchmarks.py     # measure
python tools/build_report.py                # write the report from the measurements
```

**[`SUBMISSION.md`](SUBMISSION.md)** - the cover document: what was built,
where each piece lives, the actual environment and test output, and the
headline benchmark findings.

---

## Use of AI on this project

I used Anthropic's Claude (Claude Code) as an assistant on this project,
working from a detailed written specification of the requirements that I
prepared from the assignment instructions and the course reading. The
model drafted the source files, the test suite and the first draft of the
written documents; I specified the requirements, directed the work,
reviewed the output, ran the benchmarks, and am responsible for what is
submitted here. No text was copied from an AI site and presented as a
quotation.

A full, file-by-file description is in
**[`docs/AI_USE.md`](docs/AI_USE.md)**, and the same disclosure appears at
the end of `SUBMISSION.md`, as the course policy requires.
