#!/usr/bin/env bash
# Build the Week 1 submission artifacts into submissions/week-01-algorithm-lab/.
#
#   ./tools/package_submission.sh
#
# Produces:
#   Deibel_CSC5300_Week1_AlgorithmLab.zip   the repository, minus the things
#                                           that should not travel with it
#   Deibel_CSC5300_Week1_Submission.pdf     SUBMISSION.md with its figures
#                                           embedded rather than linked
#
# The archive deliberately excludes submissions/ — without that, each rebuild
# would pack the previous zip and PDF inside the new zip.
#
# Author: Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
PARENT_DIR="$(dirname "$REPO_ROOT")"
OUT_DIR="$REPO_ROOT/submissions/week-01-algorithm-lab"

ZIP_NAME="Deibel_CSC5300_Week1_AlgorithmLab.zip"
PDF_NAME="Deibel_CSC5300_Week1_Submission.pdf"
PDF_TITLE="Deibel — CSC 5300 Week 1 — Algorithm Laboratory Setup"

PYTHON="$REPO_ROOT/algorithms_course/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

mkdir -p "$OUT_DIR"

echo "==> Exporting $PDF_NAME"
"$PYTHON" "$REPO_ROOT/tools/md_to_pdf.py" \
    "$REPO_ROOT/SUBMISSION.md" "$OUT_DIR/$PDF_NAME" --title "$PDF_TITLE"

echo "==> Building $ZIP_NAME"
rm -f "$OUT_DIR/$ZIP_NAME"
# zip is run from the parent so the archive contains a single top-level
# folder rather than a spray of loose files.
( cd "$PARENT_DIR" && zip -r -q "$OUT_DIR/$ZIP_NAME" "$REPO_NAME" \
    -x "*/algorithms_course/*" \
       "*/__pycache__/*" \
       "*/.pytest_cache/*" \
       "*/.git/*" \
       "*/submissions/*" \
       "*.pyc" \
       "*/.DS_Store" \
       "*/BUILD_SPEC.md" )

echo "==> Verifying exclusions"
# -Z1 lists entry names only. Plain `unzip -l` prints a header containing the
# archive's own path, which lives under submissions/ and matches the pattern
# below every time - a false positive that fails a perfectly good archive.
if unzip -Z1 "$OUT_DIR/$ZIP_NAME" \
     | grep -E "algorithms_course|__pycache__|\.git/|\.pyc|\.pytest_cache|submissions/|BUILD_SPEC"; then
    echo "FAIL: the archive contains something it should not" >&2
    exit 1
fi
echo "    clean ($(unzip -Z1 "$OUT_DIR/$ZIP_NAME" | wc -l | tr -d " ") entries)"

echo
echo "Wrote:"
echo "  $OUT_DIR/$ZIP_NAME"
echo "  $OUT_DIR/$PDF_NAME"
