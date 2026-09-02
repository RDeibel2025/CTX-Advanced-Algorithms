#!/usr/bin/env python3
"""Render a Markdown file to a self-contained PDF with embedded images.

Images are inlined as base64 ``data:`` URIs before rendering, so the PDF
carries its figures rather than referencing files that will not exist on
the reader's machine. Rendering is done by headless Google Chrome, which
is present on macOS installs without needing a LaTeX toolchain.

Usage::

    python tools/md_to_pdf.py SOURCE.md OUTPUT.pdf [--title "Title"]

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import markdown

#: Raw-content prefix for this project's own repository. Image sources in the
#: reports are absolute GitHub URLs so the Markdown renders correctly wherever
#: it is read, but a PDF must carry its figures rather than fetch them, so
#: sources under this prefix are mapped back to the working copy and embedded.
REPO_RAW_PREFIX = (
    "https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/"
)

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

CSS = """
@page { size: Letter; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 20pt; margin: 0 0 .4em; line-height: 1.25;
     border-bottom: 2px solid #333; padding-bottom: .25em; }
h2 { font-size: 15pt; margin: 1.6em 0 .5em; padding-bottom: .2em;
     border-bottom: 1px solid #ccc; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 1.3em 0 .4em; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 1.1em 0 .3em; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: #14507d; text-decoration: none; word-break: break-word; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.8pt;
       background: #f4f4f4; padding: .1em .3em; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px;
      padding: .7em .9em; overflow-x: auto; page-break-inside: avoid;
      font-size: 8pt; line-height: 1.42; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: .9em 0;
        font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: .32em .5em; text-align: left;
         vertical-align: top; }
th { background: #eef1f4; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: .7em auto;
      page-break-inside: avoid; border: 1px solid #e4e4e4; border-radius: 4px; }
/* Benchmark charts are exported at 9x6.2in; unconstrained they scale to the
   full text width and take a page each, leaving half of every page blank.
   Capping the height lets two sit comfortably on one page while staying
   large enough to read the axis labels. */
img { max-height: 4.1in; width: auto; }
blockquote { border-left: 3px solid #c8d3dc; margin: 1em 0; padding: .2em 1em;
             color: #444; background: #f8fafb; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.8em 0; }
em { color: inherit; }
figcaption { font-size: 8.5pt; color: #555; text-align: center;
             margin-top: -.5em; margin-bottom: 1.2em; }
.page-break { page-break-before: always; }
/* Captured console output. The demo's widest line is 98 columns, which at
   the 8pt used for ordinary code blocks is fractionally wider than the
   text block and would be clipped in print rather than scrolled. */
pre.demo-output { font-size: 6.6pt; line-height: 1.35;
                  /* Ordinary code blocks avoid breaking across pages, but a
                     135-line console transcript cannot fit on one - keeping
                     the rule would push the whole block to a fresh page and
                     leave the one before it blank. */
                  page-break-inside: auto; }
.url-banner { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 10pt;
              background: #eef4fa; border: 1px solid #b9d3ea; border-radius: 5px;
              padding: .6em .8em; margin: .8em 0; word-break: break-all; }
"""


def inline_images(html_body: str, base_dir: str) -> str:
    """Replace every local ``<img src=...>`` with an embedded data URI."""

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def replace(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith("data:"):
            return match.group(0)

        if src.startswith(REPO_RAW_PREFIX):
            # One of our own figures, addressed absolutely. Resolve it against
            # the working copy so the PDF embeds the bytes instead of leaving
            # the grader's viewer to fetch them over the network.
            path = os.path.normpath(
                os.path.join(repo_root, src[len(REPO_RAW_PREFIX):])
            )
        elif src.startswith(("http://", "https://")):
            print(f"  WARNING: external image left as a link: {src}",
                  file=sys.stderr)
            return match.group(0)
        else:
            path = os.path.normpath(os.path.join(base_dir, src))

        if not os.path.isfile(path):
            print(f"  WARNING: image not found, left as a link: {src}",
                  file=sys.stderr)
            return match.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("ascii")
        print(f"  embedded {src} ({os.path.getsize(path) / 1024:.0f} KiB)")
        return match.group(0).replace(src, f"data:{mime};base64,{payload}")

    return re.sub(r'<img[^>]+src="([^"]+)"', replace, html_body)


def find_chrome() -> str:
    """Return the path to a Chromium-family browser, or raise."""
    for candidate in CHROME_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("chromium") or shutil.which("google-chrome")
    if found:
        return found
    raise FileNotFoundError(
        "no Chromium-family browser found; tried:\n  "
        + "\n  ".join(CHROME_CANDIDATES)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    base_dir = os.path.dirname(source)

    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "codehilite", "sane_lists", "attr_list"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    print(f"rendering {os.path.basename(source)} -> {os.path.basename(output)}")
    body = inline_images(body, base_dir)

    title = args.title or os.path.splitext(os.path.basename(source))[0]
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )

    with tempfile.TemporaryDirectory() as workdir:
        html_path = os.path.join(workdir, "document.html")
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(document)

        chrome = find_chrome()
        print(f"  browser: {chrome}")
        staged = os.path.join(workdir, "out.pdf")
        log_path = os.path.join(workdir, "chrome.log")

        # Two details matter here, both learned the hard way:
        #
        # 1. `--headless=new`. Chrome removed the original headless mode in
        #    version 132; passing the bare `--headless` to a current build
        #    launches the full browser and never returns.
        # 2. Output redirected to a FILE, not to a pipe. Chrome spawns a
        #    detached auto-updater that inherits the pipe and holds it open
        #    after Chrome itself has exited, so capture_output=True blocks
        #    until the timeout even on a completely successful render.
        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-component-update",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={staged}",
            f"--user-data-dir={os.path.join(workdir, 'profile')}",
            f"file://{html_path}",
        ]

        # Chrome writes the PDF and then does not always exit promptly, so
        # poll for the file and stop waiting once its size has settled
        # rather than blocking on the process for the full timeout.
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + 180
            previous = -1
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                if os.path.exists(staged):
                    current = os.path.getsize(staged)
                    if current > 0 and current == previous:
                        process.terminate()
                        break
                    previous = current
                time.sleep(0.5)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()

        if not os.path.exists(staged):
            with open(log_path, encoding="utf-8", errors="replace") as log:
                print(log.read()[-3000:], file=sys.stderr)
            raise RuntimeError(f"Chrome did not produce a PDF for {source}")

        os.makedirs(os.path.dirname(output), exist_ok=True)
        shutil.copyfile(staged, output)

    print(f"  wrote {output} ({os.path.getsize(output) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
