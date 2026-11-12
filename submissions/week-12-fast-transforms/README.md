# Week 12 - Fast Transforms, Numerical Stability, and Signal and Image Processing

The Week 12 submission for Blackboard.

| File | What it is |
|---|---|
| `Deibel_CSC5300_Week12_Report.pdf` | The technical report, with the repository URL as literal text on page 1 and its figures embedded inline. |

No ZIP. Dr. Amakobe's Code Submission Requirements announcement of
6 October 2026 asks for a GitHub repository link with the report, and
Weeks 2 through 11 were submitted the same way. The repository is public at
<https://github.com/RDeibel2025/CTX-Advanced-Algorithms>, and the same
report renders there as
[`analysis/week12_report.md`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/analysis/week12_report.md).

The report body is inside the 1,200-1,500 range the instructions specify by
`wc -w`; the references and the AI-use statement sit outside it.

`src/fft_analyzer/` is the instructions' `FFTAnalyzer/fft_analyzer/`, and
`benchmarks/week12_benchmark.py` writes every figure, both animations, the
filtered WAV and `benchmark.csv`. The example audio, image and time series
are synthetic and generated from seeds.

Regenerate with:

```bash
python tools/build_week12_pdf.py
```
