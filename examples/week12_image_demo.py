#!/usr/bin/env python3
"""Runnable demonstration of the Week 12 image pipeline: FFT filters and DCT compression.

    python examples/week12_image_demo.py

Two short sections. Apart from the time, the output is the same on every
run, because the test scene is made from a fixed seed:

1. **The pipeline** - ``run_pipeline`` on
   ``src/fft_analyzer/examples/test_scene.png`` (created with
   ``make_test_scene(seed=42)`` if it is missing), a synthetic 256 x 256 scene
   with a gradient, a bright disc, a checkerboard square and noise. It blurs,
   finds Sobel edges, denoises and compresses at quality 10, 50 and 90, and
   writes every output to ``benchmarks/results/week12/image_pipeline/``.
2. **The PSNR table** - for each quality, the PSNR against the input and the
   fraction of quantized DCT coefficients that are not zero, which stands in
   for the file size (this is the transform and quantization stages of JPEG
   only, with no entropy coder).

Every operation is the round trip this week is built on: move the problem
where it is easy (the 2-D FFT, or the DCT of each 8x8 block), do the easy
thing (multiply bin by bin, or quantize), and move back (the inverse
transform). The anchors come from section 9.5 of the Week 12 build spec, and
every anchor printed is also asserted, so the script exits non-zero rather
than printing something untrue. It finishes in about a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.fft_analyzer.applications.image_processor import (  # noqa: E402
    BLUR_SIGMA,
    DENOISE_RADIUS,
    QUALITIES,
    load_image,
    make_test_scene,
    run_pipeline,
)
from src.utils.timer import Timer  # noqa: E402

WIDTH = 74
SEED = 42
SCENE = os.path.join(REPO_ROOT, "src", "fft_analyzer", "examples", "test_scene.png")
OUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results", "week12", "image_pipeline")


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<36} {value}")


def relpath(path: str) -> str:
    """``path`` relative to the repository root, for printing."""
    return os.path.relpath(path, REPO_ROOT)


def section_pipeline() -> dict:
    banner(1, "The pipeline: blur, Sobel, denoise and compress")
    if os.path.exists(SCENE):
        show("found", relpath(SCENE))
    else:
        make_test_scene(SCENE, seed=SEED)
        show("created with make_test_scene(seed=42)", relpath(SCENE))
    img = load_image(SCENE)
    assert img.shape == (256, 256) and 0.0 <= img.min() and img.max() <= 255.0
    show("image", f"{img.shape[0]} x {img.shape[1]} grayscale, values "
         f"{img.min():.0f}-{img.max():.0f} (synthetic)")
    with Timer() as clock:
        result = run_pipeline(SCENE, OUT_DIR)
    outputs = [result["blur"], result["sobel"], result["denoise"]]
    outputs += [result["compression"][q]["path"] for q in QUALITIES]
    for path in outputs:
        assert os.path.dirname(path) == OUT_DIR and os.path.getsize(path) > 0
    show("run_pipeline", f"{clock.elapsed * 1e3:.0f} ms, {len(outputs)} PNGs in "
         f"{relpath(OUT_DIR)}/")
    show("  Gaussian blur, through the 2-D FFT", f"sigma {BLUR_SIGMA:g} px: "
         f"{os.path.basename(result['blur'])}")
    show("  Sobel gradient magnitude", f"scaled to 0-255: {os.path.basename(result['sobel'])}")
    show("  FFT denoise, circular low-pass", f"radius {DENOISE_RADIUS:g} bins: "
         f"{os.path.basename(result['denoise'])}")
    show("  DCT compression", ", ".join(os.path.basename(result["compression"][q]["path"])
                                         for q in QUALITIES))
    return result


def section_table(result: dict) -> None:
    banner(2, "The PSNR table: what quantization keeps and what it costs")
    rows = result["compression"]
    assert set(rows) == {10, 50, 90}
    print(f"    {'quality':>7}  {'PSNR (dB)':>9}  {'nonzero ratio':>13}  output")
    for quality in QUALITIES:
        row = rows[quality]
        print(f"    {quality:>7}  {row['psnr']:>9.2f}  {row['nonzero_ratio']:>13.4f}  "
              f"{os.path.basename(row['path'])}")
    psnrs = [rows[q]["psnr"] for q in (10, 50, 90)]
    ratios = [rows[q]["nonzero_ratio"] for q in (10, 50, 90)]
    # Spec 9.5: a finer quantization table keeps more coefficients and loses
    # less, so PSNR(q90) > PSNR(q50) > PSNR(q10) and the nonzero ratio rises
    # with quality.
    assert psnrs[2] > psnrs[1] > psnrs[0]
    assert ratios[0] < ratios[1] < ratios[2]
    print()
    show("PSNR(q90) > PSNR(q50) > PSNR(q10)", "holds")
    show("nonzero ratio rises with quality", "holds")
    show("quality 10 keeps", f"{ratios[0]:.1%} of the coefficients for {psnrs[0]:.2f} dB")
    show("quality 90 keeps", f"{ratios[2]:.1%} of the coefficients for {psnrs[2]:.2f} dB")


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 12 image demonstration")
    print("FFT image filters and block-DCT compression - Robert Deibel")
    print("=" * WIDTH)
    print("  Move the problem where it is easy (the 2-D FFT or the 8x8 DCT), do the")
    print("  easy thing there, and move back. Every anchor below is asserted.")
    with Timer() as total:
        result = section_pipeline()
        section_table(result)
    print()
    print("=" * WIDTH)
    print(f"All demonstrations held ({total.elapsed:.2f} s).")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
