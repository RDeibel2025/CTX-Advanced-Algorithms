# Week 8 - Computational Complexity

The Week 8 submission for Blackboard.

| File | What it is |
|---|---|
| `Deibel_CSC5300_Week8_Report.pdf` | The technical report, with the repository URL as literal text on page 1 and both figures embedded inline. |

No ZIP. Dr. Amakobe's Code Submission Requirements announcement of
6 October 2026 asks for a GitHub repository link with the report, and
Weeks 2 through 7 were submitted the same way. The repository is public at
<https://github.com/RDeibel2025/CTX-Advanced-Algorithms>, and the same
report renders there as
[`analysis/week8_report.md`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/analysis/week8_report.md).

The report body is 1,477 words by `wc -w`, inside the 1,200-1,500 range the
instructions specify; the references and the AI-use statement sit outside it.

The instructions name `benchmarks/results/comparison_table.csv`, which is
Week 6's table, so Week 8's figures and table are in
[`benchmarks/results/week8/`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/tree/main/benchmarks/results/week8)
under the names the instructions specify.

Regenerate with:

```bash
python tools/build_week8_pdf.py
```
