# Week 13 - Advanced Data Structures

The Week 13 submission for Blackboard.

| File | What it is |
|---|---|
| `Deibel_CSC5300_Week13_Report.pdf` | The technical report, with the repository URL as literal text on page 1 and its figures embedded inline. |

No ZIP. Dr. Amakobe's Code Submission Requirements announcement of
6 October 2026 asks for a GitHub repository link with the report, and
Weeks 2 through 12 were submitted the same way. The repository is public at
<https://github.com/RDeibel2025/CTX-Advanced-Algorithms>, and the same
report renders there as
[`analysis/week13_report.md`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/analysis/week13_report.md).

The report body is inside the 1,200-1,500 range the instructions specify by
`wc -w`; the references and the AI-use statement sit outside it. The
instructions name the report `docs/chapter12_report.md`; it is
`analysis/week13_report.md` here, because this is Chapter 13 in the course
text and every week's report lives in `analysis/`.

`src/advds/` is the instructions' `AdvancedDataStructures/advds/`, and
`benchmarks/week13_benchmark.py` writes every figure and `benchmark.csv`.
The DNA sequences, the FASTA fragment and the version history are synthetic
and generated from seeds.

Regenerate with:

```bash
python tools/build_week13_pdf.py
```
