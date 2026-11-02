# Week 10 - Network Flows, Matching, and Large-Scale Graph Optimization

The Week 10 submission for Blackboard.

| File | What it is |
|---|---|
| `Deibel_CSC5300_Week10_Report.pdf` | The technical report, with the repository URL as literal text on page 1 and all eight figures embedded inline. |

No ZIP. Dr. Amakobe's Code Submission Requirements announcement of
6 October 2026 asks for a GitHub repository link with the report, and
Weeks 2 through 9 were submitted the same way. The repository is public at
<https://github.com/RDeibel2025/CTX-Advanced-Algorithms>, and the same
report renders there as
[`analysis/week10_report.md`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/analysis/week10_report.md).
The two animations are linked from the report and live in
[`benchmarks/results/week10/`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/tree/main/benchmarks/results/week10).

The report body is inside the 1,200-1,500 range the instructions specify by
`wc -w`; the references and the AI-use statement sit outside it.

`src/flow_network_solver/` is the instructions' `FlowNetworkSolver/`, and
`analysis/week10_report.md` is their `ch9_report.md`.

Regenerate with:

```bash
python tools/build_week10_pdf.py
```
