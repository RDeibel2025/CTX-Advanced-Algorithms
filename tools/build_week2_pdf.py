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
DEMO = os.path.join(REPO_ROOT, "examples", "week2_demo.py")
OUT_DIR = os.path.join(REPO_ROOT, "submissions", "week-02-divide-and-conquer")
OUT_PDF = os.path.join(OUT_DIR, "Deibel_CSC5300_Week2_Report.pdf")

REPO_URL = "https://github.com/RDeibel2025/CTX-Advanced-Algorithms"

TITLE = "Deibel - CSC 5300 Week 2 - Divide and Conquer"

#: Inserted immediately after the report's H1 so the URL is the first thing
#: on page 1 after the title, as literal monospaced text.
URL_BANNER = (
    '<div class="url-banner"><strong>Project repository (complete source, '
    "tests, benchmarks and results):</strong><br>"
    f"{REPO_URL}<br>"
    "<span style='font-size:8.5pt'>Full recurrence derivations: "
    "analysis/week2_recurrences.md &nbsp;·&nbsp; "
    "runnable demonstration: examples/week2_demo.py</span></div>"
)

#: Recurrences whose reasoning needs more than one table row. The other ten
#: are routine applications and are fully specified by the summary table,
#: which carries a, b, f(n), log_b(a), the case, the justification and the
#: resulting bound for every one of the fourteen.
APPENDIX_SECTIONS = ("9.", "11.", "13.", "14.")


def abridge_recurrences(text: str) -> str:
    """Condense the recurrence document for inclusion in the PDF.

    The full document devotes a section to each of the fourteen
    recurrences and runs to ten PDF pages. The assignment asks that each
    be shown with its a, b, f(n), log_b(a), the case applied and why, and
    the resulting bound - all of which the summary table carries for every
    one of them. So the PDF keeps the theorem statement, that table, and
    full discussion only for the four whose reasoning a table row cannot
    hold: the three the theorem does not settle and the one needing the
    extended case 2. The other ten are routine, and the complete
    derivations stay a click away in the repository.

    Sections are matched on their headings, so adding a recurrence to the
    source document needs no change here.
    """
    parts = text.split("\n## ")
    head, sections = parts[0], parts[1:]
    by_title = {section.split("\n", 1)[0].strip(): section for section in sections}

    def take(*prefixes: str) -> list:
        found = []
        for title, section in by_title.items():
            if any(title.startswith(prefix) for prefix in prefixes):
                found.append("## " + section.rstrip("\n"))
        return found

    theorem = take("The theorem")
    summary = take("Summary")
    special = take(*APPENDIX_SECTIONS)
    references = take("References")
    if not (theorem and summary and references) or len(special) != len(APPENDIX_SECTIONS):
        # `references` is checked but deliberately not emitted; see below.
        raise SystemExit(
            "recurrence appendix structure changed; abridgement would drop "
            f"content (theorem={len(theorem)} summary={len(summary)} "
            f"special={len(special)} references={len(references)})"
        )

    head = head.replace(
        "# Master Theorem - Recurrence Relations",
        "# Appendix A: Master Theorem - Recurrence Relations",
        1,
    ).rstrip("\n")
    head += (
        "\n\n> **Abridged for this PDF.** The table below solves all fourteen "
        "recurrences in full - a, b, f(n), log_b(a), the case applied, why, "
        "and the resulting bound. The four whose reasoning needs more than a "
        "table row are then discussed individually. Worked derivations for "
        "the other ten are in "
        f"[`analysis/week2_recurrences.md`]({REPO_URL}/blob/main/analysis/"
        "week2_recurrences.md)."
    )

    intro_to_special = (
        "## The four the table cannot hold\n\n"
        "Three of these fall outside the Master Theorem entirely and one "
        "needs the extended case 2. The remaining ten are routine "
        "applications, fully specified by the table above."
    )
    # The four discussions run consecutively without horizontal rules
    # between them - their headings already separate them, and three rules
    # plus their margins cost most of a page.
    # The appendix's own reference list is dropped here: it cites CLRS Ch. 4
    # and Amakobe Ch. 2, both already in the report's section 8, and on its
    # own it orphans a whole page for four lines. It stays in the full
    # document in the repository.
    special_block = "\n\n".join([intro_to_special] + special)

    major = [head] + theorem + summary + [special_block]
    return "\n\n---\n\n".join(major) + "\n"


def run_demo() -> str:
    """Run examples/week2_demo.py and return its output.

    Executed at build time rather than pasted from a previous run, so the
    transcript in the PDF is always what the committed code actually
    prints. The demo asserts every claim it makes and exits non-zero if
    any of them stops holding, so a failure here fails the build.
    """
    print("=== running examples/week2_demo.py ===")
    result = subprocess.run(
        [sys.executable, DEMO],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit("the demonstration failed; not embedding a broken run")
    lines = result.stdout.rstrip("\n").split("\n")
    print(f"    captured {len(lines)} lines, "
          f"widest {max(len(line) for line in lines)} columns")
    return result.stdout.rstrip("\n")


def build_markdown() -> str:
    """Report body, URL banner, then the two appendices."""
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

    appendix = abridge_recurrences(appendix)

    # The demonstration is run as a build gate but not embedded: its
    # transcript ran to two pages, and the script is one click away from
    # the banner on page 1. A failing demo still fails the build.
    run_demo()

    return (
        body.rstrip("\n")
        + '\n\n<div class="page-break"></div>\n\n'
        + appendix.rstrip("\n")
        + "\n"
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
