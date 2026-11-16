#!/usr/bin/env python3
"""Week 13 benchmark: what each halving costs, and what it returns.

A thin wrapper so the command matches earlier weeks. The benchmark itself
lives in :mod:`src.advds.benchmarks.performance` (the instructions'
``advds/benchmarks/performance.py``); this file only puts the repository
root on ``sys.path`` and calls its ``main()``.

Run from the repository root with the project virtualenv active::

    python benchmarks/week13_benchmark.py                 # the full study (under 5 minutes)
    python benchmarks/week13_benchmark.py --smoke         # seconds, tiny sizes
    python benchmarks/week13_benchmark.py --redraw        # figures from the CSV, nothing measured
    python benchmarks/week13_benchmark.py --seed 7 --out /tmp/w13

Results go to ``benchmarks/results/week13/`` by default.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.advds.benchmarks.performance import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
