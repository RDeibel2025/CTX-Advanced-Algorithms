#!/usr/bin/env python3
"""Render the Week 3 submission PDF.

    python tools/build_week3_pdf.py

Produces
``submissions/week-03-data-structures/Deibel_CSC5300_Week3_Report.pdf``
from ``analysis/week3_report.md``. The report is the whole submission -
everything else is graded by following its GitHub link - so the PDF adds
one thing the Markdown cannot do for itself: the repository URL as
literal, selectable text in a boxed banner under the title, readable on
paper as well as clickable.

Figures are embedded rather than linked. The report addresses them by
absolute GitHub URL so the Markdown renders anywhere; ``md_to_pdf`` maps
those back to the working copy and inlines the bytes.

Two gates run first, so a PDF is never built from a broken state: the
report must be inside the assignment's 1000-1500 words, and
``examples/week3_demo.py`` must run clean.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPORT = os.path.join(REPO_ROOT, "analysis", "week3_report.md")
DEMO = os.path.join(REPO_ROOT, "examples", "week3_demo.py")
OUT_DIR = os.path.join(REPO_ROOT, "submissions", "week-03-data-structures")
OUT_PDF = os.path.join(OUT_DIR, "Deibel_CSC5300_Week3_Report.pdf")

REPO_URL = "https://github.com/RDeibel2025/CTX-Advanced-Algorithms"
TITLE = "Deibel - CSC 5300 Week 3 - Data Structures"

#: The instructions' limits, counted as ``wc -w`` counts: whitespace-separated
#: tokens across the whole file.
WORD_RANGE = (1000, 1500)

#: Inserted directly after the report's H1, so the URL is the first thing on
#: page 1 after the title, as literal monospaced text. The pointers are kept
#: to two short lines: the banner style breaks anywhere, so one long line
#: would split mid-word.
URL_BANNER = (
    '<div class="url-banner"><strong>Project repository (complete source, '
    "tests, benchmarks and results):</strong><br>"
    f"{REPO_URL}<br>"
    "<span style='font-size:8.5pt'>Structures: src/structures/ &nbsp;·&nbsp; "
    "demonstration: examples/week3_demo.py<br>"
    "Benchmark: benchmarks/week3_structures_benchmark.py</span></div>"
)


def check_word_count(text: str) -> int:
    """Return the report's word count, or stop if it is out of range."""
    words = len(text.split())
    low, high = WORD_RANGE
    if not low <= words <= high:
        raise SystemExit(f"report is {words} words, outside {low}-{high}; not building")
    return words


def run_demo() -> None:
    """Run examples/week3_demo.py as a build gate.

    The demo asserts every claim it prints and exits non-zero if any of
    them stops holding, so a failure here fails the build.
    """
    print("=== running examples/week3_demo.py ===")
    result = subprocess.run(
        [sys.executable, DEMO],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit("the demonstration failed; not building from a broken tree")
    print("    demo ran clean")


def add_banner(body: str) -> str:
    """Return the report with the URL banner placed after its title line."""
    lines = body.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("# "):
            lines.insert(index + 1, "\n" + URL_BANNER + "\n")
            return "\n".join(lines)
    raise SystemExit("report has no H1 to anchor the URL banner to")


def main() -> int:
    with open(REPORT, encoding="utf-8") as handle:
        body = handle.read()
    words = check_word_count(body)
    print(f"report: {words} words (inside {WORD_RANGE[0]}-{WORD_RANGE[1]})")
    run_demo()

    os.makedirs(OUT_DIR, exist_ok=True)
    staged = os.path.join(OUT_DIR, ".week3_staged.md")
    with open(staged, "w", encoding="utf-8") as handle:
        handle.write(add_banner(body))
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
