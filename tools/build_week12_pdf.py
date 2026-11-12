#!/usr/bin/env python3
"""Render the Week 12 submission PDF.

    python tools/build_week12_pdf.py

Produces
``submissions/week-12-fast-transforms/Deibel_CSC5300_Week12_Report.pdf``
from ``analysis/week12_report.md``. The report is the whole submission -
everything else is graded by following its GitHub link - so the PDF adds
one thing the Markdown cannot do for itself: the repository URL as
literal, selectable text in a boxed banner under the title, readable on
paper as well as clickable.

Figures are embedded rather than linked. The report addresses them by
absolute GitHub URL so the Markdown renders anywhere; ``md_to_pdf`` maps
those back to the working copy and inlines the bytes.

Two gates run first, so a PDF is never built from a broken state: the
report must be inside the assignment's 1200-1500 words, and
the three ``examples/week12_*_demo.py`` scripts must run clean.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT = os.path.join(REPO_ROOT, "analysis", "week12_report.md")
DEMOS = [os.path.join(REPO_ROOT, "examples", name)
         for name in ("week12_signal_demo.py", "week12_audio_demo.py", "week12_image_demo.py")]
OUT_DIR = os.path.join(REPO_ROOT, "submissions", "week-12-fast-transforms")
OUT_PDF = os.path.join(OUT_DIR, "Deibel_CSC5300_Week12_Report.pdf")

REPO_URL = "https://github.com/RDeibel2025/CTX-Advanced-Algorithms"
TITLE = "Deibel - CSC 5300 Week 12 - Fast Transforms"

#: The instructions' limits, counted as ``wc -w`` counts: whitespace-separated
#: tokens across the whole file.
WORD_RANGE = (1200, 1500)

#: Inserted directly after the report's H1, so the URL is the first thing on
#: page 1 after the title, as literal monospaced text. The pointers are kept
#: to two short lines: the banner style breaks anywhere, so one long line
#: would split mid-word.
URL_BANNER = (
    '<div class="url-banner"><strong>Project repository (complete source, '
    "tests, benchmarks and results):</strong><br>"
    f"{REPO_URL}<br>"
    "<span style='font-size:8.5pt'>Fast transforms: src/fft_analyzer/ &nbsp;·&nbsp; "
    "demonstration: examples/week12_signal_demo.py<br>"
    "Benchmark: benchmarks/week12_benchmark.py</span></div>"
)


def body_text(text: str) -> str:
    """The report body: everything before the References heading.

    The Week 12 instructions put the references and the AI-use statement
    outside the 1,200-1,500 word range, so the gate counts the body only.
    """
    return text.split("\n## References", 1)[0]


def check_word_count(text: str) -> int:
    """Return the report body's word count, or stop if it is out of range."""
    words = len(body_text(text).split())
    low, high = WORD_RANGE
    if not low <= words <= high:
        raise SystemExit(f"report is {words} words, outside {low}-{high}; not building")
    return words


def run_demo() -> None:
    """Run the three Week 12 demos as a build gate.

    Each demo asserts every claim it prints and exits non-zero if any of
    them stops holding, so a failure here fails the build.
    """
    for demo in DEMOS:
        print(f"=== running examples/{os.path.basename(demo)} ===")
        result = subprocess.run(
            [sys.executable, demo],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(result.stdout[-3000:], file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            raise SystemExit("a demonstration failed; not building from a broken tree")
        print("    demo ran clean")


#: Week 12's line charts are 10 x 6.5 in figures; a 4.6 in height cap lets them
#: fill the text width (about 0.7 of their size), so legends and ticks stay
#: legible on paper. This override applies to the Week 12 PDF only;
#: md_to_pdf itself is unchanged.
FIGURE_STYLE = "<style>img { max-height: 4.6in; }</style>"

#: Spans that should not break across lines in the PDF: "n = 10", "N = 2^20",
#: and a number followed by its unit. Only the staged copy is changed; the
#: report itself keeps ordinary spaces.
_NBSP = "\u00a0"
_KEEP_TOGETHER = [
    (re.compile(r"\b([nN]) = (?=\d)"), "\\1" + _NBSP + "=" + _NBSP),
    (re.compile(r"(\d) (?=(?:Hz|kHz|dB|ms|s|GB)\b)"), "\\1" + _NBSP),
]


def keep_together(body: str) -> str:
    """Return ``body`` with non-breaking spaces in the spans above.

    Examples:
        >>> keep_together("n = 10 and 109.4 Hz") == "n\u00a0=\u00a010 and 109.4\u00a0Hz"
        True
    """
    for pattern, replacement in _KEEP_TOGETHER:
        body = pattern.sub(replacement, body)
    return body


def add_banner(body: str) -> str:
    """Return the report with the URL banner placed after its title line."""
    lines = body.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("# "):
            lines.insert(index + 1, "\n" + URL_BANNER + "\n\n" + FIGURE_STYLE + "\n")
            return "\n".join(lines)
    raise SystemExit("report has no H1 to anchor the URL banner to")


def main() -> int:
    with open(REPORT, encoding="utf-8") as handle:
        body = handle.read()
    words = check_word_count(body)
    print(f"report body: {words} words (inside {WORD_RANGE[0]}-{WORD_RANGE[1]})")
    run_demo()

    os.makedirs(OUT_DIR, exist_ok=True)
    staged = os.path.join(OUT_DIR, ".week12_staged.md")
    with open(staged, "w", encoding="utf-8") as handle:
        handle.write(add_banner(keep_together(body)))
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "tools", "md_to_pdf.py"),
             staged, OUT_PDF, "--title", TITLE],
            cwd=REPO_ROOT,
        )
    finally:
        os.remove(staged)
    if result.returncode != 0 or not os.path.exists(OUT_PDF):
        raise SystemExit("PDF build failed")

    print(f"\nwrote {OUT_PDF} ({os.path.getsize(OUT_PDF) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
