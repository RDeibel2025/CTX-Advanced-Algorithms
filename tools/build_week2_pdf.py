#!/usr/bin/env python3
"""Assemble and render the Week 2 submission PDF.

    python tools/build_week2_pdf.py

Produces
``submissions/week-02-divide-and-conquer/Deibel_CSC5300_Week2_Report.pdf``
from ``analysis/week2_report.md``, with the Master Theorem solutions
appended as an appendix so the PDF stands on its own: the assignment's
checklist asks for ten or more recurrences, and a grader reading only the
report should not have to follow a link to find them.

Two things the report's Markdown cannot do for itself are added here:

* **The repository URL as literal, selectable text**, in a boxed banner at
  the top of page 1. The Markdown already carries an autolink, but a
  grader printing the PDF needs the URL readable on paper, not just
  clickable.
* **A page break** before the appendix, so the body and the appendix do
  not run together.

Figures are embedded rather than linked. The report addresses them by
absolute GitHub URL so the Markdown renders anywhere; ``md_to_pdf`` maps
those back to the working copy and inlines the bytes.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

REPORT = os.path.join(REPO_ROOT, "analysis", "week2_report.md")
RECURRENCES = os.path.join(REPO_ROOT, "analysis", "week2_recurrences.md")
OUT_DIR = os.path.join(REPO_ROOT, "submissions", "week-02-divide-and-conquer")
OUT_PDF = os.path.join(OUT_DIR, "Deibel_CSC5300_Week2_Report.pdf")

REPO_URL = "https://github.com/RDeibel2025/CTX-Advanced-Algorithms"

TITLE = "Deibel - CSC 5300 Week 2 - Divide and Conquer"

#: Inserted immediately after the report's H1 so the URL is the first thing
#: on page 1 after the title, as literal monospaced text.
URL_BANNER = (
    '<div class="url-banner"><strong>Project repository (complete source, '
    "tests, benchmarks and results):</strong><br>"
    f"{REPO_URL}</div>"
)


def build_markdown() -> str:
    """Report body, URL banner, then the recurrence appendix."""
    with open(REPORT, encoding="utf-8") as handle:
        body = handle.read()
    with open(RECURRENCES, encoding="utf-8") as handle:
        appendix = handle.read()

    lines = body.split("\n")
    # Place the banner directly after the title line.
    for index, line in enumerate(lines):
        if line.startswith("# "):
            lines.insert(index + 1, "\n" + URL_BANNER + "\n")
            break
    else:  # pragma: no cover - the report always has an H1
        raise SystemExit("report has no H1 to anchor the URL banner to")
    body = "\n".join(lines)

    # The appendix keeps its own H1; demote nothing, just separate the two.
    appendix = appendix.replace(
        "# Master Theorem - Recurrence Relations",
        "# Appendix: Master Theorem - Recurrence Relations",
        1,
    )

    return (
        body.rstrip("\n")
        + '\n\n<div class="page-break"></div>\n\n'
        + appendix.lstrip("\n")
    )


def main() -> int:
    # Sync first, always. week2_sync_report rewrites the report's tables from
    # the result CSVs, and running it after a build silently leaves the PDF a
    # revision behind - which has happened twice. Doing it here makes the
    # ordering impossible to get wrong.
    print("=== syncing report tables to benchmarks/results/ ===")
    synced = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "tools", "week2_sync_report.py")],
        cwd=REPO_ROOT,
    )
    if synced.returncode != 0:
        raise SystemExit("report sync failed; not building a PDF from stale data")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    combined_path = os.path.join(OUT_DIR, ".week2_combined.md")
    combined = build_markdown()
    with open(combined_path, "w", encoding="utf-8") as handle:
        handle.write(combined)

    body_words = len(open(REPORT, encoding="utf-8").read().split())
    print(f"report body: {body_words} words "
          f"({'in range' if 800 <= body_words <= 1200 else 'OUT OF RANGE'})")
    print(f"combined document: {len(combined.split())} words (body + appendix)")

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "tools", "md_to_pdf.py"),
         combined_path, OUT_PDF, "--title", TITLE],
        cwd=REPO_ROOT,
    )
    os.remove(combined_path)
    if result.returncode != 0 or not os.path.exists(OUT_PDF):
        raise SystemExit("PDF build failed")

    print(f"\nwrote {OUT_PDF} ({os.path.getsize(OUT_PDF) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
