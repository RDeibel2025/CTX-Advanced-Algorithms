"""Week 13 benchmark: what each halving costs, and what it returns.

Every Week 13 structure halves a different axis of its problem. The segment
tree halves the index range and the Fenwick tree splits it by low bits
(positions); the persistent tree copies one halved path per update (time,
that is versions); rank/select cuts a bit vector into fixed levels (bits);
the wavelet tree halves the alphabet (symbols); the recursive matrix
operations and the van Emde Boas layout halve until a piece fits a block
they are never told the size of (memory). This module measures, on each
axis, what the halving costs and what it returns.

Run it from the repository root with the project virtualenv active::

    python benchmarks/week13_benchmark.py              # the full study (under 5 minutes)
    python benchmarks/week13_benchmark.py --smoke      # tiny sizes, seconds
    python benchmarks/week13_benchmark.py --redraw     # figures from OUT/benchmark.csv only
    python benchmarks/week13_benchmark.py --seed 7 --out /tmp/w13

The experiments (spec section 8)
--------------------------------
1. **Segment tree vs Fenwick tree** (``segment_vs_fenwick``), n = 2^10 to
   2^20: build time, 10,000 random point updates (assignments) and 10,000
   random range sums. A naive reference (Python ``sum`` over a list slice)
   is timed for n <= 2^16. The first 200 range-sum answers of every method
   are checked against ``sum`` over the slice at every size, and a mismatch
   raises ``RuntimeError``.
2. **Persistent overhead** (``persistent_overhead``), n = 2^10 to 2^18
   (2^19 and 2^20 are configured caps, spec 2.4): build time, 1,000 random
   point sets chained from version 0, the nodes they create, the
   ``tracemalloc`` peak after the build and after the updates (measured in
   a separate untimed run), 10,000 range sums on the newest version against
   the plain array segment tree holding the same values, and the nodes a
   full copy per version would need, ``versions * (2n - 1)``.
3. **Rank/select** (``rank_select``), n = 2^16 to 2^24 bits at densities
   0.5 and 0.1: 10,000 random ``rank1`` and 10,000 random ``select1``
   queries, the rank-directory overhead and the select samples. The naive
   reference counts the bits of the prefix with ``int.bit_count`` on one
   big Python int (n <= 2^20).
4. **Wavelet tree** (``wavelet_dna`` on synthetic DNA of 10^4, 10^5 and 10^6
   symbols from :func:`~src.advds.applications.dna_index.synthetic_dna`,
   ``wavelet_text`` on the repository's own ``README.md`` as an English
   text sample): payload bits per symbol and compression ratio for the
   balanced and Huffman shapes, H0 for comparison, and the time of 10,000
   ``range_count`` queries (both shapes), 10,000 ``range_quantile``
   queries (balanced) and, on DNA, 1,000 FM-index ``pattern_count`` calls
   (patterns of 4 to 12 bases cut from the sequence). Naive references for
   n <= 10^5: ``str.count`` over the window, and an overlapping scan with
   ``str.find``.
5. **Cache-oblivious matrix operations** (``cache_misses`` and
   ``matrix_time``): LRU-simulated misses (``base=1`` traces, section 5.1
   layout) for transpose n = 32 to 256 and multiply n = 16 to 64 at
   ``(M, B) = (256, 8), (1024, 16), (4096, 64)``, naive against recursive;
   the simulation is capped at transpose n = 256 and multiply n = 64
   (spec 2.4), so n = 512 and n = 128 get capped rows. Wall-clock time of
   the real functions (transpose up to n = 512, multiply up to 128, the
   recursive ones with ``base=16``) on random integer matrices, with
   numpy's ``A.T.copy()`` and ``A @ B`` as library references (a bare
   ``.T`` is a view and does no work, so the copy is what is timed).
6. **Layouts** (``layout_blocks``): mean distinct blocks per random
   successful search for h = 10 to 20 at B = 16 and 64, BFS against vEB,
   10,000 searches each, the same seeded searches for both layouts.
7. **Version-control history** (``vcs_history``): a seeded synthetic
   repository of 1,000 files and 40 commits (each commit modifies 1 to 64
   files; every fifth also adds two files and removes one), with per-commit
   files changed, new nodes stored, and the nodes the Merkle-pruned diff
   against the parent visits.

Measurement protocol
--------------------
Every timed method gets **one untimed warm-up call per size** and then
``runs`` timed calls through :func:`src.utils.timer.time_call` (garbage
collection paused inside each call); at least five for the numpy library
references, whose times are tiny. Every call, warm-up included, runs under
the **20-second rule**: :func:`capped` arms ``SIGALRM`` and stops a call
that passes ``cap_s`` seconds (the alarm is armed before the clock starts
and disarmed after it stops, so its cost is never part of a timing). That
size is then recorded as capped and
every larger size of the same series is capped without being run. Sizes
outside a configured limit are never run and are recorded as capped too.
The reason for every cap is printed. Memory is measured with
:mod:`tracemalloc` in separate untimed runs, never while timing.

The CSV (``benchmark.csv``)
---------------------------
The columns are fixed (:data:`COLUMNS`). Every row has every column; a
cell that does not apply is blank, and no number is written that the code
did not produce. Conventions shared by every experiment:

* ``mean_time_s`` / ``std_time_s``: mean and sample standard deviation of
  one timed call over ``runs`` calls. ``ops`` is the number of operations
  inside one call, so the time per operation is ``mean_time_s / ops``.
* ``capped``: ``True`` on a row for a size that was not run (configured
  limit or the 20 s rule; its measurements are blank), ``False`` on every
  other row.
* ``seed``: the ``derive_seed(master, label, n, ...)`` value the row's
  random input came from; blank where nothing random was used.

Per experiment:

* ``segment_vs_fenwick``: structures ``segment_tree``, ``fenwick_tree``,
  ``naive_reference``; methods ``build`` (ops 1), ``point_update`` and
  ``range_sum`` (ops 10,000).
* ``persistent_overhead``: ``persistent_segment_tree`` ``build``
  (``nodes_created`` = 2n - 1, ``memory_kib`` = tracemalloc peak of the
  build), ``point_set`` (``nodes_created`` = nodes the 1,000 updates made,
  ``memory_kib`` = peak after build and updates, ``overhead_ratio`` = all
  nodes of all versions / (2n - 1)) and ``query_sum``; ``segment_tree``
  ``build`` (``memory_kib`` only) and ``query_sum``; ``full_copy_model``
  ``copy_per_version`` (``nodes_created`` = versions * (2n - 1),
  ``overhead_ratio`` = versions).
* ``rank_select``: ``rank_select`` ``rank1`` (``overhead_ratio`` = rank
  directory bits / n) and ``select1`` (``overhead_ratio`` = all select
  sample bits / n, the select1 and the select0 samples together, as
  ``RankSelect.select_sample_bits()`` counts them: about 6.25% of n at
  any density, of which the select1 samples alone are 6.25% of the ones);
  ``naive_reference`` ``rank1_bit_count``; ``density`` set.
* ``wavelet_dna`` / ``wavelet_text``: ``wavelet_balanced`` and
  ``wavelet_huffman`` ``space`` rows (``bits_per_symbol`` = payload bits /
  n, ``compression_ratio`` = (payload + directory) / 8n,
  ``overhead_ratio`` = directory / payload), ``zero_order_entropy`` ``h0``
  (``bits_per_symbol`` = H0); timed ``range_count``, ``range_quantile``,
  ``fm_index`` ``pattern_count``, ``naive_reference``
  ``range_count_scan`` and ``pattern_count_scan``.
* ``cache_misses``: structure ``transpose`` or ``matmul``, method
  ``naive`` or ``recursive``, ``cache_M``, ``cache_B``, ``misses`` and
  ``ops`` = simulated accesses. ``matrix_time``: methods ``naive``,
  ``recursive_base16`` and ``numpy_reference``.
* ``layout_blocks``: structures ``bfs_layout`` and ``veb_layout``, method
  ``random_search``, ``n`` = 2^h - 1 keys, ``cache_B`` = B, ``ops`` =
  searches, ``blocks_per_search``.
* ``vcs_history`` (structure ``mini_vcs``): ``commit`` rows with ``n`` =
  commit number, ``ops`` = files changed against the parent and
  ``nodes_created`` = new nodes the commit stored; ``tree_nodes`` rows with
  ``nodes_created`` = nodes in that commit's tree (one per file, what a
  full copy of the commit would store); ``diff_vs_parent`` rows with
  ``ops`` = nodes the diff visited and ``overhead_ratio`` = visited /
  (nodes in both trees); one ``all_commits_stored`` row (``n`` = commits,
  ``nodes_created`` = distinct nodes stored, ``overhead_ratio`` = stored /
  full copy) and one ``all_commits_full_copy`` row (``nodes_created`` =
  the sum of the tree sizes).

References and checks (spec 1.1)
--------------------------------
Every structure and query measured here is the package's own hand-written
code. Library calls appear only as clearly labelled naive or library
reference curves and as the independent answer checks: ``sum`` over list
slices, ``int.bit_count``, ``np.flatnonzero``, ``str.count``,
``str.find`` and numpy's ``.T.copy()`` and ``@``. None of them answers a
query of a structure under test.

Outputs
-------
``benchmark.csv`` and seven figures (:data:`FIGURES`), every PNG saved with
:func:`src.utils.visualization.save_png_under_limit` (under 100 KB). The
figures are always drawn from the CSV as read back from disk, so
``--redraw`` produces the same figures and leaves the CSV byte for byte.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import platform
import signal as os_signal
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, replace  # replace: the doctests shrink a Config with it
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
)

from src.advds.applications.dna_index import DNAIndex, synthetic_dna  # noqa: E402
from src.advds.applications.version_control import MiniVCS  # noqa: E402
from src.advds.cache_oblivious.cache_sim import LRUCache  # noqa: E402
from src.advds.cache_oblivious.layouts import blocks_per_search  # noqa: E402
from src.advds.cache_oblivious.matrix_ops import (  # noqa: E402
    co_matmul,
    co_transpose,
    naive_matmul,
    naive_transpose,
    trace_matmul,
    trace_transpose,
)
from src.advds.succinct.bit_vector import BitVector  # noqa: E402
from src.advds.succinct.rank_select import RankSelect  # noqa: E402
from src.advds.succinct.wavelet_tree import WaveletTree, zero_order_entropy  # noqa: E402
from src.advds.trees.fenwick_tree import FenwickTree  # noqa: E402
from src.advds.trees.persistent_tree import PersistentSegmentTree  # noqa: E402
from src.advds.trees.segment_tree import SegmentTree  # noqa: E402
from src.utils.random_utils import DEFAULT_SEED, derive_seed, make_rng  # noqa: E402
from src.utils.timer import peak_memory_kib, time_call  # noqa: E402
from src.utils.visualization import apply_house_style, save_png_under_limit  # noqa: E402

__all__ = [
    "COLUMNS",
    "Config",
    "FIGURES",
    "FULL",
    "SMOKE",
    "RunCapExceeded",
    "STUDIES",
    "Study",
    "cache_study",
    "capped",
    "figures",
    "fmt",
    "layout_study",
    "main",
    "measure",
    "persistent_study",
    "plot_cache_misses",
    "plot_layout_blocks",
    "plot_persistent_overhead",
    "plot_rank_select",
    "plot_segment_vs_fenwick",
    "plot_vcs_history",
    "plot_wavelet_compression",
    "rank_select_study",
    "read_csv",
    "redraw",
    "row",
    "segment_fenwick_study",
    "vcs_study",
    "wavelet_study",
    "write_csv",
]

#: The repository root (four levels above this file).
REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: Where the full run writes its CSV and figures by default.
RESULTS_DIR: str = os.path.join(REPO_ROOT, "benchmarks", "results", "week13")

#: The English text sample of experiment 4: the repository's own README.
README_PATH: str = os.path.join(REPO_ROOT, "README.md")

#: The CSV's columns, fixed by spec section 8.
COLUMNS: List[str] = [
    "experiment", "structure", "method", "n", "density", "runs", "mean_time_s", "std_time_s", "ops",
    "nodes_created", "memory_kib", "overhead_ratio", "bits_per_symbol", "compression_ratio", "cache_M",
    "cache_B", "misses", "blocks_per_search", "capped", "seed",
]

#: The columns that hold wall-clock measurements (they differ from run to run).
TIMING_COLUMNS: Tuple[str, ...] = ("mean_time_s", "std_time_s")

#: Columns that name a series rather than measure it; a capped row keeps them.
_KEY_FIELDS: Tuple[str, ...] = ("density", "cache_M", "cache_B")

_INT_COLUMNS = frozenset({"n", "runs", "ops", "nodes_created", "cache_M", "cache_B", "misses", "seed"})
_TEXT_COLUMNS = frozenset({"experiment", "structure", "method"})

#: The figures the benchmark writes, in the order it writes them.
FIGURES: Tuple[str, ...] = (
    "segment_vs_fenwick.png",
    "persistent_overhead.png",
    "rank_select.png",
    "wavelet_compression.png",
    "cache_misses.png",
    "layout_blocks.png",
    "vcs_history.png",
)


@dataclass(frozen=True)
class Config:
    """Every size, count and limit of one benchmark mode.

    Attributes:
        runs: Timed calls per method and size, after one untimed warm-up.
        reference_runs: The minimum timed calls of a numpy library reference.
        cap_s: The 20-second rule: a single call longer than this is stopped.
        check_queries: How many answers of each method are checked.
        tree_exponents: Experiment 1 sizes, n = 2^e.
        tree_ops: Point updates and range sums per timed call.
        naive_max_exponent: Largest e for which the naive range sum is timed.
        persistent_exponents: Experiment 2 sizes (larger ones are capped).
        persistent_max_exponent: Largest e the persistent tree is run at.
        persistent_updates: Point sets per timed call (and versions - 1).
        persistent_queries: Range sums per timed call.
        rs_exponents: Experiment 3 bit-vector lengths, n = 2^e.
        rs_densities: Fractions of 1 bits.
        rs_queries: ``rank1`` and ``select1`` queries per timed call.
        rs_naive_max_exponent: Largest e for which the naive rank is timed.
        dna_sizes: Experiment 4 synthetic DNA lengths.
        include_text: Whether to add the README text sample.
        wavelet_naive_max: Largest n for which the naive scans are timed.
        wavelet_queries: ``range_count`` and ``range_quantile`` queries per call.
        pattern_queries: FM ``pattern_count`` calls per timed call.
        pattern_lengths: Smallest and largest pattern length.
        transpose_sizes: Experiment 5 transpose sizes for the wall clock.
        transpose_sim_max: Largest transpose n that is simulated.
        matmul_sizes: Experiment 5 multiply sizes for the wall clock.
        matmul_sim_max: Largest multiply n that is simulated.
        cache_settings: The ``(M, B)`` pairs of the LRU simulation.
        co_base: The base-case size of the timed recursive functions.
        layout_heights: Experiment 6 tree heights.
        layout_blocks: Experiment 6 block sizes.
        layout_trials: Searches per height, block size and layout.
        vcs_files: Files in the synthetic repository of experiment 7.
        vcs_commits: Commits in it.

    Examples:
        >>> FULL.tree_exponents[0], FULL.tree_exponents[-1], FULL.naive_max_exponent
        (10, 20, 16)
        >>> SMOKE.tree_exponents[-1] <= 12 and max(SMOKE.dna_sizes) <= 10**4
        True
    """

    runs: int
    reference_runs: int
    cap_s: float
    check_queries: int
    tree_exponents: Tuple[int, ...]
    tree_ops: int
    naive_max_exponent: int
    persistent_exponents: Tuple[int, ...]
    persistent_max_exponent: int
    persistent_updates: int
    persistent_queries: int
    rs_exponents: Tuple[int, ...]
    rs_densities: Tuple[float, ...]
    rs_queries: int
    rs_naive_max_exponent: int
    dna_sizes: Tuple[int, ...]
    include_text: bool
    wavelet_naive_max: int
    wavelet_queries: int
    pattern_queries: int
    pattern_lengths: Tuple[int, int]
    transpose_sizes: Tuple[int, ...]
    transpose_sim_max: int
    matmul_sizes: Tuple[int, ...]
    matmul_sim_max: int
    cache_settings: Tuple[Tuple[int, int], ...]
    co_base: int
    layout_heights: Tuple[int, ...]
    layout_blocks: Tuple[int, ...]
    layout_trials: int
    vcs_files: int
    vcs_commits: int


#: The full study (spec section 8).
FULL = Config(
    runs=5, reference_runs=5, cap_s=20.0, check_queries=200,
    tree_exponents=tuple(range(10, 21)), tree_ops=10_000, naive_max_exponent=16,
    persistent_exponents=tuple(range(10, 21)), persistent_max_exponent=18,
    persistent_updates=1_000, persistent_queries=10_000,
    rs_exponents=tuple(range(16, 25)), rs_densities=(0.5, 0.1), rs_queries=10_000, rs_naive_max_exponent=20,
    dna_sizes=(10**4, 10**5, 10**6), include_text=True, wavelet_naive_max=10**5,
    wavelet_queries=10_000, pattern_queries=1_000, pattern_lengths=(4, 12),
    transpose_sizes=(32, 64, 128, 256, 512), transpose_sim_max=256,
    matmul_sizes=(16, 32, 64, 128), matmul_sim_max=64,
    cache_settings=((256, 8), (1024, 16), (4096, 64)), co_base=16,
    layout_heights=tuple(range(10, 21)), layout_blocks=(16, 64), layout_trials=10_000,
    vcs_files=1_000, vcs_commits=40,
)

#: Seconds-long smoke run: n <= 2^12, DNA <= 10^4, transpose n <= 32, multiply n <= 16, h <= 10.
SMOKE = Config(
    runs=2, reference_runs=5, cap_s=20.0, check_queries=50,
    tree_exponents=(10, 11, 12), tree_ops=1_000, naive_max_exponent=11,
    persistent_exponents=(10, 11, 12), persistent_max_exponent=11,
    persistent_updates=100, persistent_queries=1_000,
    rs_exponents=(10, 11, 12), rs_densities=(0.5, 0.1), rs_queries=1_000, rs_naive_max_exponent=11,
    dna_sizes=(10**3, 10**4), include_text=True, wavelet_naive_max=10**3,
    wavelet_queries=1_000, pattern_queries=100, pattern_lengths=(4, 8),
    transpose_sizes=(8, 16, 32), transpose_sim_max=16,
    matmul_sizes=(4, 8, 16), matmul_sim_max=8,
    cache_settings=((256, 8), (1024, 16), (4096, 64)), co_base=16,
    layout_heights=(8, 9, 10), layout_blocks=(16, 64), layout_trials=1_000,
    vcs_files=64, vcs_commits=6,
)


# ---------------------------------------------------------------------------
# The 20-second rule and the timing protocol
# ---------------------------------------------------------------------------


class RunCapExceeded(Exception):
    """Raised by :func:`capped` when one call passes the cap.

    Examples:
        >>> issubclass(RunCapExceeded, Exception)
        True
    """


def capped(function: Callable[[], Any], cap_s: float) -> Any:
    """Call ``function`` and stop it with :class:`RunCapExceeded` if it passes ``cap_s`` seconds.

    ``SIGALRM`` is armed with :func:`signal.setitimer` before the call and
    disarmed (and the previous handler restored) afterwards, whatever
    happens. Where no alarm can be armed (no ``SIGALRM``, or not the main
    thread) the function simply runs.

    Args:
        function: A callable with no arguments.
        cap_s: The cap in seconds, greater than 0.

    Returns:
        Whatever ``function`` returns.

    Raises:
        RunCapExceeded: If the call passes ``cap_s`` seconds.
        ValueError: If ``cap_s`` is not positive.

    Time Complexity:
        The cost of one call, at most ``cap_s`` seconds.

    Space Complexity:
        O(1) beyond the call.

    Examples:
        >>> capped(lambda: 6 * 7, 20.0)
        42
        >>> def slow():
        ...     while True:
        ...         pass
        >>> stopped = False
        >>> try:
        ...     capped(slow, 0.05)
        ... except RunCapExceeded:
        ...     stopped = True
        >>> stopped
        True
    """
    if not cap_s > 0:
        raise ValueError(f"cap_s must be positive, got {cap_s}")

    def on_alarm(signum: int, frame: Any) -> None:
        raise RunCapExceeded()

    try:
        previous = os_signal.signal(os_signal.SIGALRM, on_alarm)
    except (AttributeError, ValueError):  # pragma: no cover - no SIGALRM or not the main thread
        return function()
    os_signal.setitimer(os_signal.ITIMER_REAL, cap_s)
    try:
        return function()
    finally:
        os_signal.setitimer(os_signal.ITIMER_REAL, 0)
        os_signal.signal(os_signal.SIGALRM, previous)


def measure(function: Callable[[], Any], runs: int, cap_s: float) -> Tuple[Any, Dict[str, float]]:
    """One untimed warm-up call, then ``runs`` timed calls, every call under the cap.

    The warm-up call's result is returned so the caller can check the
    answers without paying for another call. Each timed call is one
    :func:`src.utils.timer.time_call` run (a collection is forced before the
    call and garbage collection is paused during it) made inside
    :func:`capped`, so the alarm is armed before the clock starts and
    disarmed after it stops: the signal set-up (about 10 microseconds) is
    never charged to the call, which matters for the numpy references
    whose calls take only a few microseconds.

    Args:
        function: A callable with no arguments.
        runs: Timed calls, at least 1.
        cap_s: The per-call cap in seconds.

    Returns:
        ``(result, stats)`` where ``stats`` has ``mean``, ``std`` (sample
        standard deviation, 0.0 for one run), ``min``, ``max`` (seconds)
        and ``runs``.

    Raises:
        RunCapExceeded: If any call, warm-up included, passes the cap.
        ValueError: If ``runs`` is below 1.

    Time Complexity:
        ``(runs + 1)`` calls of ``function``.

    Space Complexity:
        O(runs) for the timings, plus the result.

    Examples:
        >>> result, stats = measure(lambda: sum(range(100)), 3, 20.0)
        >>> result, stats["runs"], stats["min"] <= stats["mean"] <= stats["max"]
        (4950, 3, True)
        >>> measure(lambda: 1, 0, 20.0)
        Traceback (most recent call last):
            ...
        ValueError: runs must be at least 1, got 0
    """
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    result = capped(function, cap_s)
    # The cap wraps the stopwatch, not the other way round: arming SIGALRM
    # inside the timed region would add its own cost to every timing.
    timings = [capped(lambda: time_call(function, repeat=1, warmup=0), cap_s)["mean"] for _ in range(runs)]
    stats = {
        "mean": statistics.fmean(timings),
        "std": statistics.stdev(timings) if runs > 1 else 0.0,
        "min": min(timings),
        "max": max(timings),
        "runs": runs,
    }
    return result, stats


def fmt(value: Any, digits: int = 8) -> str:
    """Format one CSV cell; ``None`` becomes an empty cell.

    Args:
        value: ``None``, a bool, an int, a float (numpy scalars included) or text.
        digits: Significant digits for floats.

    Returns:
        The cell text.

    Raises:
        Nothing.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> [fmt(None), fmt(True), fmt(np.int64(7)), fmt(0.078125), fmt(1 / 3), fmt("x")]
        ['', 'True', '7', '0.078125', '0.33333333', 'x']
    """
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def row(experiment: str, structure: str, method: str, n: Any, seed: Any = None, **fields: Any) -> Dict[str, Any]:
    """Build one CSV row with every column present (``None`` where a field does not apply).

    Args:
        experiment: The experiment name.
        structure: The structure (or reference) measured.
        method: The operation measured.
        n: The size.
        seed: The seed of the row's random input, or ``None``.
        **fields: Any other columns.

    Returns:
        A dict with exactly the keys of :data:`COLUMNS`.

    Raises:
        KeyError: If a field is not one of :data:`COLUMNS`.

    Time Complexity:
        O(number of columns).

    Space Complexity:
        O(number of columns).

    Examples:
        >>> r = row("layout_blocks", "veb_layout", "random_search", 1023, seed=5, cache_B=16)
        >>> list(r) == COLUMNS, r["cache_B"], r["misses"]
        (True, 16, None)
    """
    entry: Dict[str, Any] = {c: None for c in COLUMNS}
    for key in fields:
        if key not in entry:
            raise KeyError(f"unknown column {key!r}")
    entry.update(experiment=experiment, structure=structure, method=method, n=n, seed=seed)
    entry.update(fields)
    return entry


def _timing_fields(stats: Dict[str, float]) -> Dict[str, Any]:
    """Turn :func:`measure` statistics into the ``runs``, ``mean_time_s`` and ``std_time_s`` cells.

    The standard deviation is left blank for a single run, where it is
    undefined.

    Args:
        stats: The dict :func:`measure` returns.

    Returns:
        The three cells.

    Examples:
        >>> _timing_fields({"runs": 1, "mean": 0.5, "std": 0.0})
        {'runs': 1, 'mean_time_s': 0.5, 'std_time_s': None}
    """
    runs = int(stats["runs"])
    return {"runs": runs, "mean_time_s": stats["mean"], "std_time_s": stats["std"] if runs > 1 else None}


def _quiet(text: str) -> None:
    """Discard a progress line (the default, so library calls stay silent).

    Examples:
        >>> _quiet("nothing is written") is None
        True
    """
    return None


def _console(text: str) -> None:
    """Write one progress line to standard output (the command-line channel of :func:`main`).

    Examples:
        >>> _console("hello")
        hello
    """
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


@dataclass
class Study:
    """The state one benchmark run shares across its experiments.

    Attributes:
        cfg: The sizes and counts (:data:`FULL` or :data:`SMOKE`).
        master: The master seed.
        rows: The CSV rows collected so far, in order.
        log: Where progress lines and cap reasons go (silent by default).
        dead: Series stopped by the 20 s rule, mapped to the reason their
            larger sizes are capped.

    Examples:
        >>> study = Study(SMOKE, 42)
        >>> study.add("demo", "s", "m", 8, ops=3)["capped"]
        False
        >>> len(study.rows)
        1
    """

    cfg: Config
    master: int
    rows: List[Dict[str, Any]] = field(default_factory=list)
    log: Callable[[str], None] = _quiet
    dead: Dict[Tuple[Any, ...], str] = field(default_factory=dict)

    def add(self, experiment: str, structure: str, method: str, n: Any, seed: Any = None,
            **fields: Any) -> Dict[str, Any]:
        """Append a measured (not capped) row and return it.

        Args:
            experiment: The experiment name.
            structure: The structure measured.
            method: The operation measured.
            n: The size.
            seed: The seed of the input, or ``None``.
            **fields: The measured cells.

        Returns:
            The appended row (``capped`` is ``False``).

        Raises:
            KeyError: If a field is not a CSV column.

        Time Complexity:
            O(number of columns).

        Space Complexity:
            O(number of columns).

        Examples:
            >>> Study(SMOKE, 42).add("e", "s", "m", 4, misses=9)["misses"]
            9
        """
        entry = row(experiment, structure, method, n, seed, capped=False, **fields)
        self.rows.append(entry)
        return entry

    def cap(self, experiment: str, structure: str, method: str, n: Any, reason: str, seed: Any = None,
            **fields: Any) -> Dict[str, Any]:
        """Append a capped row (measurements blank) and log the reason.

        Only the series-naming fields (``density``, ``cache_M``,
        ``cache_B``) are kept from ``fields``; everything measured stays
        blank.

        Args:
            experiment: The experiment name.
            structure: The structure that was not run.
            method: The operation that was not run.
            n: The size that was not run.
            reason: Why, printed with the row's identity.
            seed: The seed the size would use, or ``None``.
            **fields: Any cells; only the series-naming ones are kept.

        Returns:
            The appended row (``capped`` is ``True``).

        Raises:
            KeyError: If a field is not a CSV column.

        Time Complexity:
            O(number of columns).

        Space Complexity:
            O(number of columns).

        Examples:
            >>> lines = []
            >>> study = Study(SMOKE, 42, log=lines.append)
            >>> r = study.cap("e", "s", "m", 64, "configured limit", density=0.5, ops=10)
            >>> r["capped"], r["density"], r["ops"]
            (True, 0.5, None)
            >>> lines
            ['    [cap] e / s / m n = 64 (density 0.5): configured limit']
        """
        keys = {k: v for k, v in fields.items() if k in _KEY_FIELDS}
        entry = row(experiment, structure, method, n, seed, capped=True, **keys)
        self.rows.append(entry)
        extra = ", ".join(f"{k.replace('cache_', '')} {v}" if k.startswith("cache_") else f"{k} {v}"
                          for k, v in keys.items() if v is not None)
        self.log(f"    [cap] {experiment} / {structure} / {method} n = {n}"
                 + (f" ({extra})" if extra else "") + f": {reason}")
        return entry

    def _key(self, experiment: str, structure: str, method: str, fields: Dict[str, Any]) -> Tuple[Any, ...]:
        """The identity of a series: names plus its series-naming fields.

        Examples:
            >>> Study(SMOKE, 42)._key("e", "s", "m", {"density": 0.1, "ops": 3})
            ('e', 's', 'm', 0.1, None, None)
        """
        return (experiment, structure, method) + tuple(fields.get(k) for k in _KEY_FIELDS)

    def attempt(self, experiment: str, structure: str, method: str, n: Any, function: Callable[[], Any],
                seed: Any = None, limit: Optional[str] = None, **fields: Any) -> Tuple[bool, Any]:
        """Run ``function`` once under the cap, or record a capped row instead.

        Used for untimed measurements (simulated misses, blocks per search).
        A configured ``limit``, an earlier cap of the same series, or a call
        that passes the cap each produce a capped row and ``(False, None)``.
        On success nothing is appended; the caller writes the row.

        Args:
            experiment: The experiment name.
            structure: The structure.
            method: The operation.
            n: The size.
            function: A callable with no arguments.
            seed: The input seed, or ``None``.
            limit: A configured-limit reason, or ``None`` to run.
            **fields: Series-naming fields (kept on a capped row).

        Returns:
            ``(True, result)`` or ``(False, None)``.

        Raises:
            Whatever ``function`` raises, other than :class:`RunCapExceeded`.

        Time Complexity:
            One call of ``function``.

        Space Complexity:
            Whatever ``function`` uses.

        Examples:
            >>> study = Study(SMOKE, 42)
            >>> study.attempt("e", "s", "m", 1, lambda: 5)
            (True, 5)
            >>> study.attempt("e", "s", "m", 2, lambda: 5, limit="configured limit")
            (False, None)
            >>> study.rows[-1]["capped"]
            True
        """
        key = self._key(experiment, structure, method, fields)
        reason = limit or self.dead.get(key)
        if reason is None:
            try:
                return True, capped(function, self.cfg.cap_s)
            except RunCapExceeded:
                reason = f"a single run passed the {self.cfg.cap_s:.0f} s cap"
                self.dead[key] = f"a smaller size passed the {self.cfg.cap_s:.0f} s cap"
        self.cap(experiment, structure, method, n, reason, seed, **fields)
        return False, None

    def timed(self, experiment: str, structure: str, method: str, n: Any, function: Callable[[], Any],
              seed: Any = None, limit: Optional[str] = None, runs: Optional[int] = None,
              **fields: Any) -> Tuple[bool, Any]:
        """Warm up, time and record ``function``, or record a capped row instead.

        A configured ``limit``, an earlier cap of the same series, or any
        call that passes the cap each produce a capped row and
        ``(False, None)``. Otherwise one measured row is appended with the
        timing cells and ``fields``, and the warm-up call's result returned.

        Args:
            experiment: The experiment name.
            structure: The structure.
            method: The operation.
            n: The size.
            function: A callable with no arguments (one timed call).
            seed: The input seed, or ``None``.
            limit: A configured-limit reason, or ``None`` to run.
            runs: Timed calls (default ``cfg.runs``).
            **fields: Other cells of the row (``ops``, ``density``, ...).

        Returns:
            ``(True, result)`` or ``(False, None)``.

        Raises:
            Whatever ``function`` raises, other than :class:`RunCapExceeded`.

        Time Complexity:
            ``runs + 1`` calls of ``function``.

        Space Complexity:
            Whatever ``function`` uses.

        Examples:
            >>> study = Study(SMOKE, 42)
            >>> study.timed("e", "s", "m", 10, lambda: sum(range(10)), ops=10)
            (True, 45)
            >>> r = study.rows[-1]
            >>> r["runs"], r["ops"], r["capped"], r["mean_time_s"] > 0
            (2, 10, False, True)
        """
        key = self._key(experiment, structure, method, fields)
        reason = limit or self.dead.get(key)
        if reason is None:
            try:
                result, stats = measure(function, runs or self.cfg.runs, self.cfg.cap_s)
            except RunCapExceeded:
                reason = f"a single run passed the {self.cfg.cap_s:.0f} s cap"
                self.dead[key] = f"a smaller size passed the {self.cfg.cap_s:.0f} s cap"
        if reason is not None:
            self.cap(experiment, structure, method, n, reason, seed, **fields)
            return False, None
        self.add(experiment, structure, method, n, seed, **_timing_fields(stats), **fields)
        return True, result


def _check(name: str, got: Sequence[Any], expected: Sequence[Any]) -> None:
    """Raise ``RuntimeError`` if the first answers differ from the reference.

    Args:
        name: What is being checked, for the message.
        got: The answers under test (at least as many as ``expected``).
        expected: The reference answers.

    Raises:
        RuntimeError: On the first difference.

    Examples:
        >>> _check("demo", [1, 2, 3], [1, 2])
        >>> _check("demo", [1, 5], [1, 2])
        Traceback (most recent call last):
            ...
        RuntimeError: demo: answer 1 is 5, the reference says 2
    """
    for i, want in enumerate(expected):
        if got[i] != want:
            raise RuntimeError(f"{name}: answer {i} is {got[i]!r}, the reference says {want!r}")


def _ranges(rng: np.random.Generator, n: int, count: int) -> List[Tuple[int, int]]:
    """``count`` random non-empty half-open ranges ``[l, r)`` inside ``[0, n)``.

    Both ends are drawn uniformly; ``l`` is the smaller and ``r`` the larger
    plus one.

    Examples:
        >>> rs = _ranges(np.random.default_rng(1), 10, 50)
        >>> all(0 <= l < r <= 10 for l, r in rs)
        True
    """
    a = rng.integers(0, n, count)
    b = rng.integers(0, n, count)
    return list(zip(np.minimum(a, b).tolist(), (np.maximum(a, b) + 1).tolist()))


def _pow2(e: int) -> str:
    """Plain-text ``2^e`` for log lines and cap reasons.

    Examples:
        >>> _pow2(16)
        '2^16'
    """
    return f"2^{e}"


# ---------------------------------------------------------------------------
# Experiment 1: segment tree vs Fenwick tree
# ---------------------------------------------------------------------------


def segment_fenwick_study(study: Study) -> None:
    """Experiment 1: build, 10,000 point updates and 10,000 range sums, n = 2^10 to 2^20.

    Per size one seed gives the array (values 0 to 999), the update list
    (index, new value) and the query ranges. Updates are assignments, so
    repeating the same list (warm-up and timed calls) leaves the same state.
    The naive reference (``sum`` over a list slice) is timed only up to
    ``2^naive_max_exponent``; above it the row is a configured cap. The
    first ``check_queries`` range-sum answers of every method are checked
    against ``sum`` over the slice of the updated list at every size.

    Args:
        study: The run's shared state; rows are appended to ``study.rows``.

    Returns:
        None.

    Raises:
        RuntimeError: If an answer differs from the naive one.

    Time Complexity:
        O(sum over sizes of (runs + 1) * (n + ops log n)), plus the naive
        reference's O(ops * n) per call up to its limit.

    Space Complexity:
        O(n) for the array and both trees.

    Examples:
        >>> study = Study(replace(SMOKE, tree_exponents=(6, 7), naive_max_exponent=6, tree_ops=200), 42)
        >>> segment_fenwick_study(study)
        >>> sorted({(r["structure"], r["method"]) for r in study.rows})[:3]
        [('fenwick_tree', 'build'), ('fenwick_tree', 'point_update'), ('fenwick_tree', 'range_sum')]
        >>> [(r["n"], r["capped"]) for r in study.rows if r["structure"] == "naive_reference"]
        [(64, False), (128, True)]
    """
    cfg, E = study.cfg, "segment_vs_fenwick"
    study.log("\n=== EXPERIMENT 1 - segment tree vs Fenwick tree ===")
    for e in cfg.tree_exponents:
        n = 1 << e
        seed = derive_seed(study.master, E, n)
        rng = np.random.default_rng(seed)
        values = rng.integers(0, 1000, n).tolist()
        updates = list(zip(rng.integers(0, n, cfg.tree_ops).tolist(), rng.integers(0, 1000, cfg.tree_ops).tolist()))
        queries = _ranges(rng, n, cfg.tree_ops)
        naive = list(values)
        for i, v in updates:
            naive[i] = v
        expected = [sum(naive[l:r]) for l, r in queries[: cfg.check_queries]]
        parts = []
        for structure, cls, update_name, query_name in (("segment_tree", SegmentTree, "update", "query_sum"),
                                                        ("fenwick_tree", FenwickTree, "set", "range_sum")):
            ok, tree = study.timed(E, structure, "build", n, lambda: cls(values), seed=seed, ops=1)
            if not ok:
                for method in ("point_update", "range_sum"):
                    study.cap(E, structure, method, n, "its build was capped", seed)
                continue
            update = getattr(tree, update_name)
            query = getattr(tree, query_name)

            def run_updates() -> None:
                for i, v in updates:
                    update(i, v)

            ok, _ = study.timed(E, structure, "point_update", n, run_updates, seed=seed, ops=cfg.tree_ops)
            if not ok:
                study.cap(E, structure, "range_sum", n, "its updates were capped", seed)
                continue
            ok, answers = study.timed(E, structure, "range_sum", n, lambda: [query(l, r) for l, r in queries],
                                      seed=seed, ops=cfg.tree_ops)
            if ok:
                _check(f"{structure} range_sum n = {n}", answers, expected)
                parts.append(f"{structure} {study.rows[-1]['mean_time_s'] / cfg.tree_ops * 1e6:6.2f} us/sum")
        limit = None if e <= cfg.naive_max_exponent else (
            f"configured limit: the naive reference runs only for n <= {_pow2(cfg.naive_max_exponent)}")
        ok, answers = study.timed(E, "naive_reference", "range_sum", n,
                                  lambda: [sum(naive[l:r]) for l, r in queries], seed=seed, limit=limit,
                                  ops=cfg.tree_ops)
        if ok:
            _check(f"naive range_sum n = {n}", answers, expected)
            parts.append(f"naive {study.rows[-1]['mean_time_s'] / cfg.tree_ops * 1e6:8.2f} us/sum")
        study.log(f"    n = {_pow2(e):<5} " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Experiment 2: persistent overhead
# ---------------------------------------------------------------------------

#: The (structure, method) series of experiment 2, for the capped sizes.
_PERSISTENT_SERIES: Tuple[Tuple[str, str], ...] = (
    ("persistent_segment_tree", "build"),
    ("persistent_segment_tree", "point_set"),
    ("full_copy_model", "copy_per_version"),
    ("segment_tree", "build"),
    ("persistent_segment_tree", "query_sum"),
    ("segment_tree", "query_sum"),
)


def _persistent_memory(values: List[int], sets: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """Build a persistent tree and chain the point sets under :mod:`tracemalloc` (untimed).

    The traced peak is read once after the build and once after the
    updates; the second is the peak of the whole run, the memory the
    structure needed to hold every version. If tracing was already on, the
    peak is reset instead and the tracer left running.

    Args:
        values: The initial array.
        sets: ``(index, value)`` point sets, chained from version 0.

    Returns:
        ``build_kib``, ``all_kib``, ``build_nodes``, ``set_nodes`` and
        ``versions``.

    Examples:
        >>> m = _persistent_memory([0] * 16, [(3, 1), (5, 2)])
        >>> m["build_nodes"], m["set_nodes"], m["versions"], m["all_kib"] >= m["build_kib"] > 0
        (31, 10, 3, True)
    """
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    try:
        tree = PersistentSegmentTree(values)
        build_peak = tracemalloc.get_traced_memory()[1]
        build_nodes = tree.nodes_created()
        version = 0
        for i, v in sets:
            version = tree.point_set(version, i, v)
        all_peak = tracemalloc.get_traced_memory()[1]
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return {"build_kib": build_peak / 1024.0, "all_kib": all_peak / 1024.0, "build_nodes": build_nodes,
            "set_nodes": tree.nodes_created() - build_nodes, "versions": tree.versions()}


def persistent_study(study: Study) -> None:
    """Experiment 2: what versions cost a path-copying segment tree, n = 2^10 to 2^18.

    Sizes above ``2^persistent_max_exponent`` are configured caps (spec
    2.4), recorded for every series. For each size: the traced memory run
    (:func:`_persistent_memory`), the plain segment tree's traced build
    memory, the timed build, the timed chain of point sets from version 0
    (each timed call makes its own branch of the same versions), the
    full-copy model, and 10,000 range sums on the newest version against
    the plain segment tree holding the same values. The first answers of
    both are checked against ``sum`` over the slice.

    Args:
        study: The run's shared state.

    Returns:
        None.

    Raises:
        RuntimeError: If an answer differs from the naive one.

    Time Complexity:
        O((runs + 1) * (n + (updates + queries) log n)) per size, plus one
        traced build.

    Space Complexity:
        O(n + (runs + 2) * updates * log n) nodes per size.

    Examples:
        >>> cfg = replace(SMOKE, persistent_exponents=(4, 5), persistent_max_exponent=4,
        ...               persistent_updates=10, persistent_queries=20, check_queries=5)
        >>> study = Study(cfg, 42)
        >>> persistent_study(study)
        >>> r = [r for r in study.rows if r["method"] == "point_set" and not r["capped"]][0]
        >>> r["n"], r["ops"], r["nodes_created"], round(r["overhead_ratio"], 4)
        (16, 10, 50, 2.6129)
        >>> sum(r["capped"] for r in study.rows)
        6
    """
    cfg, E = study.cfg, "persistent_overhead"
    study.log("\n=== EXPERIMENT 2 - persistent overhead ===")
    U, Q = cfg.persistent_updates, cfg.persistent_queries
    for e in cfg.persistent_exponents:
        n = 1 << e
        seed = derive_seed(study.master, E, n)
        if e > cfg.persistent_max_exponent:
            reason = (f"configured limit: the node-based persistent tree runs only for "
                      f"n <= {_pow2(cfg.persistent_max_exponent)} (spec 2.4)")
            for structure, method in _PERSISTENT_SERIES:
                study.cap(E, structure, method, n, reason, seed)
            continue
        rng = np.random.default_rng(seed)
        values = rng.integers(0, 1000, n).tolist()
        sets = list(zip(rng.integers(0, n, U).tolist(), rng.integers(0, 1000, U).tolist()))
        queries = _ranges(rng, n, Q)
        final = list(values)
        for i, v in sets:
            final[i] = v
        expected = [sum(final[l:r]) for l, r in queries[: cfg.check_queries]]

        memory = _persistent_memory(values, sets)
        plain_tree, plain_kib = peak_memory_kib(SegmentTree, final)
        ok, tree = study.timed(E, "persistent_segment_tree", "build", n, lambda: PersistentSegmentTree(values),
                               seed=seed, ops=1, nodes_created=memory["build_nodes"], memory_kib=memory["build_kib"])
        latest = None
        if ok:
            point_set = tree.point_set

            def chain() -> int:
                version = 0
                for i, v in sets:
                    version = point_set(version, i, v)
                return version

            ok, latest = study.timed(
                E, "persistent_segment_tree", "point_set", n, chain, seed=seed, ops=U,
                nodes_created=memory["set_nodes"], memory_kib=memory["all_kib"],
                overhead_ratio=(memory["build_nodes"] + memory["set_nodes"]) / (2 * n - 1))
            if not ok:
                latest = None
        else:
            study.cap(E, "persistent_segment_tree", "point_set", n, "its build was capped", seed)
        study.add(E, "full_copy_model", "copy_per_version", n, seed, ops=U,
                  nodes_created=memory["versions"] * (2 * n - 1), overhead_ratio=float(memory["versions"]))
        study.add(E, "segment_tree", "build", n, seed, memory_kib=plain_kib)
        parts = [f"{memory['set_nodes'] / U:5.1f} nodes/update", f"{memory['all_kib']:9.0f} KiB"]
        if latest is not None:
            query_sum = tree.query_sum
            ok, answers = study.timed(E, "persistent_segment_tree", "query_sum", n,
                                      lambda: [query_sum(latest, l, r) for l, r in queries], seed=seed, ops=Q)
            if ok:
                _check(f"persistent query_sum n = {n}", answers, expected)
                parts.append(f"persistent {study.rows[-1]['mean_time_s'] / Q * 1e6:5.2f} us/sum")
        else:
            study.cap(E, "persistent_segment_tree", "query_sum", n, "its updates were capped", seed)
        plain_query = plain_tree.query_sum
        ok, answers = study.timed(E, "segment_tree", "query_sum", n,
                                  lambda: [plain_query(l, r) for l, r in queries], seed=seed, ops=Q)
        if ok:
            _check(f"segment_tree query_sum n = {n}", answers, expected)
            parts.append(f"plain {study.rows[-1]['mean_time_s'] / Q * 1e6:5.2f} us/sum")
        study.log(f"    n = {_pow2(e):<5} " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Experiment 3: rank/select
# ---------------------------------------------------------------------------


def rank_select_study(study: Study) -> None:
    """Experiment 3: ``rank1`` and ``select1`` on n = 2^16 to 2^24 bits, densities 0.5 and 0.1.

    The bits are ``rng.random(n) < density``. Rank positions are uniform in
    ``[0, n]`` and select ranks uniform in ``[1, ones]``. The naive
    reference keeps the whole vector as one Python int and computes
    ``(big & ((1 << i) - 1)).bit_count()``; it is timed up to
    ``2^rs_naive_max_exponent``. The first answers are checked at every
    size: rank against that same expression, select against
    ``np.flatnonzero``.

    Args:
        study: The run's shared state.

    Returns:
        None.

    Raises:
        RuntimeError: If an answer differs from the reference.

    Time Complexity:
        O(n) to build per size and density, O(queries) per timed call, and
        O(queries * n / 64) per naive call.

    Space Complexity:
        O(n) bits plus a temporary O(n) array.

    Examples:
        >>> cfg = replace(SMOKE, rs_exponents=(11, 12), rs_densities=(0.5,), rs_queries=100,
        ...               rs_naive_max_exponent=11, check_queries=20)
        >>> study = Study(cfg, 42)
        >>> rank_select_study(study)
        >>> [(r["method"], r["n"], r["capped"]) for r in study.rows if r["structure"] == "naive_reference"]
        [('rank1_bit_count', 2048, False), ('rank1_bit_count', 4096, True)]
        >>> [r["overhead_ratio"] for r in study.rows if r["method"] == "rank1"]
        [0.078125, 0.078125]
    """
    cfg, E = study.cfg, "rank_select"
    study.log("\n=== EXPERIMENT 3 - rank/select ===")
    Q, C = cfg.rs_queries, cfg.check_queries
    for e in cfg.rs_exponents:
        n = 1 << e
        parts = []
        for d in cfg.rs_densities:
            seed = derive_seed(study.master, E, n, d)
            rng = np.random.default_rng(seed)
            bits = rng.random(n) < d
            rs = RankSelect(BitVector.from_bits(bits))
            ones = rs.num_ones
            rank_at = rng.integers(0, n + 1, Q).tolist()
            select_k = rng.integers(1, ones + 1, Q).tolist() if ones else []
            big = int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")
            positions = np.flatnonzero(bits)
            rank1, select1 = rs.rank1, rs.select1
            ok, answers = study.timed(E, "rank_select", "rank1", n, lambda: [rank1(i) for i in rank_at],
                                      seed=seed, density=d, ops=Q, overhead_ratio=rs.overhead_bits() / n)
            if ok:
                _check(f"rank1 n = {n} d = {d}", answers, [(big & ((1 << i) - 1)).bit_count() for i in rank_at[:C]])
                parts.append(f"d {d}: rank {study.rows[-1]['mean_time_s'] / Q * 1e6:5.2f} us")
            ok, answers = study.timed(E, "rank_select", "select1", n, lambda: [select1(k) for k in select_k],
                                      seed=seed, density=d, ops=len(select_k),
                                      overhead_ratio=rs.select_sample_bits() / n)
            if ok:
                _check(f"select1 n = {n} d = {d}", answers, [int(positions[k - 1]) for k in select_k[:C]])
                parts.append(f"select {study.rows[-1]['mean_time_s'] / max(len(select_k), 1) * 1e6:5.2f} us")
            limit = None if e <= cfg.rs_naive_max_exponent else (
                f"configured limit: the naive reference runs only for n <= {_pow2(cfg.rs_naive_max_exponent)}")
            ok, _ = study.timed(E, "naive_reference", "rank1_bit_count", n,
                                lambda: [(big & ((1 << i) - 1)).bit_count() for i in rank_at],
                                seed=seed, limit=limit, density=d, ops=Q)
            if ok:
                parts.append(f"naive {study.rows[-1]['mean_time_s'] / Q * 1e6:6.2f} us")
        study.log(f"    n = {_pow2(e):<5} " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Experiment 4: wavelet tree
# ---------------------------------------------------------------------------


def _naive_pattern_count(text: str, pattern: str) -> int:
    """Count overlapping occurrences by scanning the whole text with ``str.find`` (the naive reference).

    Examples:
        >>> _naive_pattern_count("AAAA", "AA"), _naive_pattern_count("ACGT", "GA")
        (3, 0)
    """
    count = 0
    start = text.find(pattern)
    while start != -1:
        count += 1
        start = text.find(pattern, start + 1)
    return count


def _quantile_by_counting(text: str, alphabet: str, l: int, r: int, k: int) -> str:
    """The k-th smallest symbol of ``text[l:r]`` by counting each symbol with ``str.count`` (a check).

    Examples:
        >>> _quantile_by_counting("TACGTT", "ACGT", 0, 6, 3)
        'G'
    """
    for symbol in alphabet:
        m = text.count(symbol, l, r)
        if k <= m:
            return symbol
        k -= m
    raise ValueError("k is larger than the window")


def _wavelet_dataset(study: Study, E: str, seq: str, seq_seed: Optional[int], query_seed: int,
                     with_fm: bool) -> str:
    """Space rows, H0 and timed queries for one sequence; return a one-line summary.

    Examples:
        >>> study = Study(replace(SMOKE, wavelet_queries=50, pattern_queries=10, check_queries=10), 42)
        >>> line = _wavelet_dataset(study, "wavelet_dna", synthetic_dna(500, 1), 1, 2, True)
        >>> sorted({(r["structure"], r["method"]) for r in study.rows})[:2]
        [('fm_index', 'pattern_count'), ('naive_reference', 'pattern_count_scan')]
    """
    cfg = study.cfg
    n = len(seq)
    trees = {"wavelet_balanced": WaveletTree(seq, shape="balanced"),
             "wavelet_huffman": WaveletTree(seq, shape="huffman")}
    for name, wt in trees.items():
        study.add(E, name, "space", n, seq_seed, bits_per_symbol=wt.bits_per_symbol(),
                  compression_ratio=wt.compression_ratio(), overhead_ratio=wt.overhead_bits() / wt.size_in_bits())
    h0 = zero_order_entropy(seq)
    study.add(E, "zero_order_entropy", "h0", n, seq_seed, bits_per_symbol=h0)
    alphabet = trees["wavelet_balanced"].alphabet
    rng = np.random.default_rng(query_seed)
    Q, C = cfg.wavelet_queries, cfg.check_queries
    windows = _ranges(rng, n, Q)
    symbols = [alphabet[j] for j in rng.integers(0, len(alphabet), Q).tolist()]
    count_q = [(c, l, r) for c, (l, r) in zip(symbols, windows)]
    ks = rng.integers(1, np.array([r - l for l, r in windows]) + 1).tolist()
    quant_q = [(l, r, k) for (l, r), k in zip(windows, ks)]
    want_counts = [seq.count(c, l, r) for c, l, r in count_q[:C]]
    want_quant = [_quantile_by_counting(seq, alphabet, l, r, k) for l, r, k in quant_q[:C]]
    limit = None if n <= cfg.wavelet_naive_max else (
        f"configured limit: the naive scans run only for n <= {cfg.wavelet_naive_max:,}")
    parts = []
    for name, wt in trees.items():
        range_count = wt.range_count
        ok, answers = study.timed(E, name, "range_count", n, lambda: [range_count(c, l, r) for c, l, r in count_q],
                                  seed=query_seed, ops=Q)
        if ok:
            _check(f"{name} range_count n = {n}", answers, want_counts)
            parts.append(f"{name.split('_')[1]} count {study.rows[-1]['mean_time_s'] / Q * 1e6:5.2f} us")
    range_quantile = trees["wavelet_balanced"].range_quantile
    ok, answers = study.timed(E, "wavelet_balanced", "range_quantile", n,
                              lambda: [range_quantile(l, r, k) for l, r, k in quant_q], seed=query_seed, ops=Q)
    if ok:
        _check(f"range_quantile n = {n}", answers, want_quant)
        parts.append(f"quantile {study.rows[-1]['mean_time_s'] / Q * 1e6:5.2f} us")
    ok, _ = study.timed(E, "naive_reference", "range_count_scan", n, lambda: [seq.count(c, l, r) for c, l, r in count_q],
                        seed=query_seed, limit=limit, ops=Q)
    if with_fm:
        P = cfg.pattern_queries
        lo, hi = cfg.pattern_lengths
        lengths = rng.integers(lo, hi + 1, P).tolist()
        starts = rng.integers(0, n - np.array(lengths) + 1).tolist()
        patterns = [seq[s:s + m] for s, m in zip(starts, lengths)]
        index = DNAIndex(seq)
        pattern_count = index.pattern_count
        ok, answers = study.timed(E, "fm_index", "pattern_count", n, lambda: [pattern_count(p) for p in patterns],
                                  seed=query_seed, ops=P)
        if ok:
            _check(f"pattern_count n = {n}", answers, [_naive_pattern_count(seq, p) for p in patterns[:C]])
            parts.append(f"FM {study.rows[-1]['mean_time_s'] / P * 1e6:6.2f} us/pattern")
        study.timed(E, "naive_reference", "pattern_count_scan", n,
                    lambda: [_naive_pattern_count(seq, p) for p in patterns], seed=query_seed, limit=limit, ops=P)
    bal, huf = trees["wavelet_balanced"], trees["wavelet_huffman"]
    return (f"bits/symbol balanced {bal.bits_per_symbol():.3f} huffman {huf.bits_per_symbol():.3f} "
            f"H0 {h0:.3f}  " + "  ".join(parts))


def wavelet_study(study: Study, text_path: str = README_PATH) -> None:
    """Experiment 4: wavelet trees on synthetic DNA and on the README text.

    Each DNA length n gets the sequence ``synthetic_dna(n, seed)`` with
    ``seed = derive_seed(master, "wavelet_dna", n)`` and its queries from
    ``derive_seed(master, "wavelet_dna", n, "queries")``. The README is read
    as UTF-8; its queries come from ``derive_seed(master, "wavelet_text",
    n, "queries")``, and its space rows have no seed (nothing random). The
    FM-index (``DNAIndex``) is DNA only, so the text has no pattern rows.

    Args:
        study: The run's shared state.
        text_path: The English text sample (the repository README).

    Returns:
        None.

    Raises:
        RuntimeError: If an answer differs from the reference.
        OSError: If ``text_path`` cannot be read.

    Time Complexity:
        O(n log sigma) per build and O(queries log sigma) per timed call,
        O(n log^2 n) for the suffix array, O(queries * n) per naive call.

    Space Complexity:
        O(n) per dataset.

    Examples:
        >>> cfg = replace(SMOKE, dna_sizes=(300,), include_text=False, wavelet_queries=30,
        ...               pattern_queries=10, check_queries=5)
        >>> study = Study(cfg, 42)
        >>> wavelet_study(study)
        >>> [r["bits_per_symbol"] for r in study.rows if r["method"] == "space"]
        [2.0, 2.0]
    """
    cfg = study.cfg
    study.log("\n=== EXPERIMENT 4 - wavelet tree ===")
    for n in cfg.dna_sizes:
        seed = derive_seed(study.master, "wavelet_dna", n)
        line = _wavelet_dataset(study, "wavelet_dna", synthetic_dna(n, seed), seed,
                                derive_seed(study.master, "wavelet_dna", n, "queries"), with_fm=True)
        study.log(f"    DNA n = {n:<8,} {line}")
    if cfg.include_text:
        with open(text_path, encoding="utf-8") as handle:
            text = handle.read()
        line = _wavelet_dataset(study, "wavelet_text", text, None,
                                derive_seed(study.master, "wavelet_text", len(text), "queries"), with_fm=False)
        study.log(f"    README n = {len(text):<8,} {line}")


# ---------------------------------------------------------------------------
# Experiment 5: cache-oblivious matrix operations
# ---------------------------------------------------------------------------


def cache_study(study: Study) -> None:
    """Experiment 5: simulated LRU misses and wall-clock time of transpose and multiply.

    Simulation: for every size up to the configured maximum, every
    ``(M, B)`` and both methods, a fresh :class:`LRUCache` replays the
    ``base=1`` trace; larger sizes are configured caps (spec 2.4). Wall
    clock: random integer matrices in ``[-9, 9]`` (seed
    ``derive_seed(master, "matrix_time", op, n)``), the naive and the
    recursive (``base = co_base``) functions on lists of lists, and numpy's
    ``A.T.copy()`` / ``A @ B`` on arrays as the library reference (at least
    ``reference_runs`` timed calls). Our results must equal numpy's exactly.

    Args:
        study: The run's shared state.

    Returns:
        None.

    Raises:
        RuntimeError: If a result differs from numpy's.

    Time Complexity:
        Theta(n^2) accesses per transpose trace and Theta(n^3) per multiply
        trace and per timed multiply.

    Space Complexity:
        O(n^2) per matrix and O(M / B) per cache.

    Examples:
        >>> cfg = replace(SMOKE, transpose_sizes=(8, 16), transpose_sim_max=8, matmul_sizes=(4,),
        ...               matmul_sim_max=4, cache_settings=((256, 8),))
        >>> study = Study(cfg, 42)
        >>> cache_study(study)
        >>> [(r["structure"], r["method"], r["n"], r["misses"]) for r in study.rows
        ...  if r["experiment"] == "cache_misses" and r["structure"] == "transpose"]
        [('transpose', 'naive', 8, 16), ('transpose', 'recursive', 8, 16), ('transpose', 'naive', 16, None), ('transpose', 'recursive', 16, None)]
    """
    cfg = study.cfg
    study.log("\n=== EXPERIMENT 5 - cache-oblivious matrix operations ===")
    for op, sizes, sim_max, tracer in (("transpose", cfg.transpose_sizes, cfg.transpose_sim_max, trace_transpose),
                                       ("matmul", cfg.matmul_sizes, cfg.matmul_sim_max, trace_matmul)):
        for n in sizes:
            limit = None if n <= sim_max else (
                f"configured limit: the {op} simulation runs only for n <= {sim_max} (spec 2.4)")
            parts = []
            for M, B in cfg.cache_settings:
                for method in ("naive", "recursive"):
                    cache = LRUCache(M, B)
                    ok, misses = study.attempt("cache_misses", op, method, n,
                                               lambda: tracer(n, method, cache), limit=limit, cache_M=M, cache_B=B)
                    if ok:
                        study.add("cache_misses", op, method, n, cache_M=M, cache_B=B, misses=misses,
                                  ops=cache.accesses)
                        parts.append(f"{method[:3]} {misses}")
            if parts:
                study.log(f"    {op:<9} n = {n:<4} simulated misses (per (M, B), naive/recursive): "
                          + " ".join(parts))
        for n in sizes:
            seed = derive_seed(study.master, "matrix_time", op, n)
            rng = np.random.default_rng(seed)
            A_np = rng.integers(-9, 10, (n, n))
            A = A_np.tolist()
            if op == "transpose":
                expected = A_np.T.tolist()
                methods = (("naive", lambda: naive_transpose(A)),
                           (f"recursive_base{cfg.co_base}", lambda: co_transpose(A, base=cfg.co_base)),
                           ("numpy_reference", lambda: A_np.T.copy()))
            else:
                B_np = rng.integers(-9, 10, (n, n))
                B = B_np.tolist()
                expected = (A_np @ B_np).tolist()
                methods = (("naive", lambda: naive_matmul(A, B)),
                           (f"recursive_base{cfg.co_base}", lambda: co_matmul(A, B, base=cfg.co_base)),
                           ("numpy_reference", lambda: A_np @ B_np))
            parts = []
            for method, function in methods:
                reference = method == "numpy_reference"
                runs = max(cfg.runs, cfg.reference_runs) if reference else cfg.runs
                ok, result = study.timed("matrix_time", op, method, n, function, seed=seed, runs=runs, ops=1)
                if ok:
                    if not reference and result != expected:
                        raise RuntimeError(f"{op} {method} n = {n} differs from numpy")
                    parts.append(f"{method} {study.rows[-1]['mean_time_s'] * 1e3:8.3f} ms")
            study.log(f"    {op:<9} n = {n:<4} wall clock: " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Experiment 6: layouts
# ---------------------------------------------------------------------------


def layout_study(study: Study) -> None:
    """Experiment 6: distinct blocks per random successful search, BFS against vEB.

    For every height h and block size B both layouts get the same seed,
    ``derive_seed(master, "layout_blocks", 2^h - 1, B)``, so they replay
    the same searches. ``n`` in the CSV is the number of keys, ``2^h - 1``.

    Args:
        study: The run's shared state.

    Returns:
        None.

    Raises:
        Nothing beyond what :func:`blocks_per_search` raises.

    Time Complexity:
        O(trials * h) per height, block size and layout.

    Space Complexity:
        O(h).

    Examples:
        >>> study = Study(replace(SMOKE, layout_heights=(10,), layout_blocks=(16,), layout_trials=1000), 42)
        >>> layout_study(study)
        >>> [(r["structure"], r["n"], r["cache_B"]) for r in study.rows]
        [('bfs_layout', 1023, 16), ('veb_layout', 1023, 16)]
        >>> study.rows[1]["blocks_per_search"] < 0.6 * study.rows[0]["blocks_per_search"]
        True
    """
    cfg, E = study.cfg, "layout_blocks"
    study.log("\n=== EXPERIMENT 6 - BFS vs van Emde Boas layout ===")
    for h in cfg.layout_heights:
        n = (1 << h) - 1
        parts = []
        for B in cfg.layout_blocks:
            seed = derive_seed(study.master, E, n, B)
            for layout in ("bfs", "veb"):
                ok, value = study.attempt(E, f"{layout}_layout", "random_search", n,
                                          lambda: blocks_per_search(h, B, layout, cfg.layout_trials, seed),
                                          seed=seed, cache_B=B)
                if ok:
                    study.add(E, f"{layout}_layout", "random_search", n, seed, cache_B=B, ops=cfg.layout_trials,
                              blocks_per_search=value)
                    parts.append(f"B {B} {layout} {value:6.3f}")
        study.log(f"    h = {h:<3} " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Experiment 7: version-control history
# ---------------------------------------------------------------------------


def vcs_study(study: Study) -> None:
    """Experiment 7: per-commit sharing and diff cost in a seeded synthetic repository.

    ``vcs_files`` files named ``src/module_XXXX.py`` are committed first.
    Each later commit modifies k files, k drawn from 1, 2, 4, 8, 16, 32 and
    64 (at most the number of files), and every fifth commit also adds two
    new files and removes one. Contents are short synthetic strings with a
    random token. Everything comes from ``make_rng(derive_seed(master,
    "vcs_history", vcs_files))``. Then every commit's history entry and the
    diff against its parent are recorded (see the module docstring for
    which columns hold what).

    Args:
        study: The run's shared state.

    Returns:
        None.

    Raises:
        RuntimeError: If a diff's file count disagrees with the history.

    Time Complexity:
        O(commits * (k log n + diff cost)) for k changed files of n.

    Space Complexity:
        O(n + commits * k log n) nodes.

    Examples:
        >>> study = Study(replace(SMOKE, vcs_files=32, vcs_commits=5), 42)
        >>> vcs_study(study)
        >>> [r["method"] for r in study.rows][:3]
        ['commit', 'tree_nodes', 'commit']
        >>> [r["nodes_created"] for r in study.rows if r["method"] == "tree_nodes"][:2]
        [32, 32]
    """
    cfg, E = study.cfg, "vcs_history"
    study.log("\n=== EXPERIMENT 7 - version-control history ===")
    seed = derive_seed(study.master, E, cfg.vcs_files)
    rng = make_rng(seed)
    vcs = MiniVCS()
    live = [f"src/module_{i:04d}.py" for i in range(cfg.vcs_files)]
    for path in live:
        vcs.add(path, f"# {path}\n# revision 0\ntoken = {rng.getrandbits(32):08x}\n")
    vcs.commit(f"import {cfg.vcs_files} synthetic files")
    next_file = cfg.vcs_files
    for number in range(2, cfg.vcs_commits + 1):
        k = min(rng.choice((1, 2, 4, 8, 16, 32, 64)), len(live))
        for path in rng.sample(live, k):
            vcs.add(path, f"# {path}\n# revision {number}\ntoken = {rng.getrandbits(32):08x}\n")
        message = f"commit {number}: modify {k} files"
        if number % 5 == 0:
            for _ in range(2):
                path = f"src/module_{next_file:04d}.py"
                next_file += 1
                vcs.add(path, f"# {path}\n# new in commit {number}\ntoken = {rng.getrandbits(32):08x}\n")
                live.append(path)
            gone = live.pop(rng.randrange(len(live)))
            vcs.rm(gone)
            message += ", add 2, remove 1"
        vcs.commit(message)
    history = vcs.history()
    previous = None
    for entry in history:
        number = entry["number"]
        study.add(E, "mini_vcs", "commit", number, seed, ops=entry["files_changed"], nodes_created=entry["new_nodes"])
        study.add(E, "mini_vcs", "tree_nodes", number, seed, nodes_created=entry["files"])
        if previous is not None:
            d = vcs.diff(previous["id"], entry["id"])
            changed = len(d["added"]) + len(d["removed"]) + len(d["modified"])
            if changed != entry["files_changed"]:
                raise RuntimeError(f"diff of commit {number} reports {changed} files, history {entry['files_changed']}")
            both = previous["files"] + entry["files"]
            study.add(E, "mini_vcs", "diff_vs_parent", number, seed, ops=d["nodes_visited"],
                      overhead_ratio=d["nodes_visited"] / both)
            study.log(f"    commit {number:<3} files changed {entry['files_changed']:>3}, "
                      f"new nodes {entry['new_nodes']:>4}, diff visited {d['nodes_visited']:>4} of {both} nodes")
        previous = entry
    stats = vcs.stats()
    study.add(E, "mini_vcs", "all_commits_stored", stats["commits"], seed, nodes_created=stats["unique_nodes"],
              overhead_ratio=stats["stored_fraction"])
    study.add(E, "mini_vcs", "all_commits_full_copy", stats["commits"], seed, nodes_created=stats["full_copy_nodes"])
    study.log(f"    {stats['commits']} commits store {stats['unique_nodes']:,} nodes against "
              f"{stats['full_copy_nodes']:,} for a full copy per commit ({stats['stored_fraction']:.1%})")


#: The experiments in the order they run (and appear in the CSV).
STUDIES: Tuple[Callable[[Study], None], ...] = (
    segment_fenwick_study, persistent_study, rank_select_study, wavelet_study, cache_study, layout_study,
    vcs_study,
)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` with exactly the :data:`COLUMNS` header.

    Args:
        path: The CSV path (its directory must exist).
        rows: Row dicts; missing columns are written blank.

    Returns:
        None.

    Raises:
        OSError: If the file cannot be written.

    Time Complexity:
        O(rows * columns).

    Space Complexity:
        O(columns).

    Examples:
        >>> import tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "b.csv")
        >>> write_csv(path, [row("e", "s", "m", 4, seed=1, capped=False)])
        >>> read_csv(path)[0]["n"], read_csv(path)[0]["capped"], read_csv(path)[0]["misses"]
        (4, False, None)
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for entry in rows:
            writer.writerow({c: fmt(entry.get(c)) for c in COLUMNS})


def read_csv(path: str) -> List[Dict[str, Any]]:
    """Read a ``benchmark.csv`` back into typed rows (blank cells become ``None``).

    Args:
        path: The CSV path.

    Returns:
        One dict per row with every column: text for the names, ``bool``
        for ``capped``, ``int`` for counts and seeds, ``float`` otherwise.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a cell does not parse.

    Time Complexity:
        O(rows * columns).

    Space Complexity:
        O(rows * columns).

    Examples:
        >>> import tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "b.csv")
        >>> write_csv(path, [row("e", "s", "m", 8, density=0.5, mean_time_s=1e-3, capped=True)])
        >>> r = read_csv(path)[0]
        >>> r["density"], r["mean_time_s"], r["capped"], r["seed"]
        (0.5, 0.001, True, None)
    """
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            entry: Dict[str, Any] = {}
            for column in COLUMNS:
                text = raw.get(column, "") or ""
                if text == "":
                    entry[column] = None
                elif column in _TEXT_COLUMNS:
                    entry[column] = text
                elif column == "capped":
                    entry[column] = text == "True"
                elif column in _INT_COLUMNS:
                    entry[column] = int(text)
                else:
                    entry[column] = float(text)
            rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

_REF_COLOR = "#7f8c8d"
_FONT = {"title": 14.0, "label": 13.0, "tick": 13.0, "legend": 11.5}
_FONT_MULTI = {"title": 15.5, "label": 14.5, "tick": 14.0, "legend": 12.5}


def _pick(rows: Sequence[Dict[str, Any]], experiment: str, structure: Optional[str] = None,
          method: Optional[str] = None, capped_rows: bool = False, **match: Any) -> List[Dict[str, Any]]:
    """Rows of one series, sorted by n; measured rows by default, capped ones on request.

    Floats in ``match`` are compared with a tolerance.

    Examples:
        >>> rows = [row("e", "s", "m", 8, capped=False, density=0.1), row("e", "s", "m", 4, capped=False, density=0.1)]
        >>> [r["n"] for r in _pick(rows, "e", "s", "m", density=0.1)]
        [4, 8]
        >>> _pick(rows, "e", "s", "m", capped_rows=True)
        []
    """
    out = []
    for r in rows:
        if r["experiment"] != experiment or bool(r.get("capped")) != capped_rows:
            continue
        if structure is not None and r["structure"] != structure:
            continue
        if method is not None and r["method"] != method:
            continue
        if all(r.get(k) is not None and abs(float(r[k]) - float(v)) < 1e-12 for k, v in match.items()):
            out.append(r)
    return sorted(out, key=lambda r: r["n"])


def _per_op_us(r: Dict[str, Any]) -> Optional[float]:
    """Microseconds per operation, ``mean_time_s / ops * 1e6``, or ``None`` if not measured.

    Examples:
        >>> _per_op_us({"mean_time_s": 0.01, "ops": 10_000}), _per_op_us({"mean_time_s": None, "ops": 1})
        (1.0, None)
    """
    if r.get("mean_time_s") is None or not r.get("ops"):
        return None
    return r["mean_time_s"] / r["ops"] * 1e6


def _line(ax: Axes, xs: Sequence[float], ys: Sequence[Optional[float]], style: Dict[str, Any], label: str) -> bool:
    """Plot one series (points with ``None`` dropped); return whether anything was drawn.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _line(ax, [1, 2], [None, 3.0], {"marker": "o", "linestyle": "-", "color": "k"}, "x")
        True
        >>> _line(ax, [], [], {"marker": "o", "linestyle": "-", "color": "k"}, "y")
        False
        >>> plt.close(fig)
    """
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and y > 0]
    if not pts:
        return False
    hollow = style.get("hollow", False)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=style["marker"], linestyle=style["linestyle"],
            color=style["color"], markerfacecolor="white" if hollow else style["color"],
            markeredgewidth=1.8 if hollow else 1.0, markersize=style.get("size", 8), linewidth=2.0, label=label)
    return True


def _n_label(n: float, powers: bool) -> str:
    """Tick text for a size: ``$2^{k}$`` for a power of two when ``powers``, else the number.

    Examples:
        >>> _n_label(1024, True), _n_label(96, True), _n_label(64, False)
        ('$2^{10}$', '96', '64')
    """
    n = int(n)
    if powers and n > 0 and n & (n - 1) == 0:
        return f"$2^{{{n.bit_length() - 1}}}$"
    return f"{n:,}"


def _size_axis(ax: Axes, ns: Sequence[int], powers: bool = True, max_ticks: int = 6) -> None:
    """Base-2 log x axis with ticks at the sizes (every other one when there are many).

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _size_axis(ax, [2 ** e for e in range(10, 21)])
        >>> len(ax.get_xticks())
        6
        >>> plt.close(fig)
    """
    ax.set_xscale("log", base=2)
    ticks = sorted(set(int(n) for n in ns))
    if not ticks:
        return
    step = 1
    while len(ticks[::step]) > max_ticks:
        step += 1
    shown = ticks[::-1][::step][::-1]
    ax.xaxis.set_major_locator(FixedLocator(shown))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _n_label(v, powers)))
    ax.xaxis.set_minor_locator(NullLocator())
    pad = 2 ** 0.35
    ax.set_xlim(ticks[0] / pad, ticks[-1] * pad)


def _value_label(v: float) -> str:
    """Tick text for a value on a log axis.

    Examples:
        >>> [_value_label(v) for v in (0.5, 2.0, 1000.0, 1e4, 2e4, 1e6, 2e-5)]
        ['0.5', '2', '1,000', '$10^{4}$', '20,000', '$10^{6}$', '2e-05']
    """
    if v <= 0:
        return ""
    k = math.log10(v)
    if v >= 1e4 and abs(k - round(k)) < 1e-9:
        return f"$10^{{{int(round(k))}}}$"
    if 1e-3 <= v < 1e5:
        return f"{v:,.6g}"
    return f"{v:.0e}"


def _log_y(ax: Axes, reserve: float = 0.0) -> None:
    """Log y axis with readable ticks; optionally keep the bottom ``reserve`` of the axes free of data.

    Ticks sit at every integer multiple of a power of ten when the data
    span about one decade, at 1, 2 and 5 times powers of ten up to about
    three decades, and at powers of ten (of a hundred past seven decades)
    beyond (labelled ``10^k``), so short ranges still carry labels and long
    ones are labelled one way. Under about one decade the bottom limit is
    lowered to the nearest whole multiple of a power of ten, so the lowest
    values sit above a labelled tick. ``reserve`` lowers the
    bottom limit so a cap note can sit under every data point.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([1, 2], [3.0, 40.0])
        >>> _log_y(ax, reserve=0.2)
        >>> ax.get_yscale(), bool(ax.get_ylim()[0] < 3.0)
        ('log', True)
        >>> plt.close(fig)
    """
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    if reserve > 0 and lo > 0 and hi > lo:
        span = math.log10(hi) - math.log10(lo)
        lo = 10 ** (math.log10(lo) - span * reserve / (1 - reserve))
        ax.set_ylim(lo, hi)
    decades = math.log10(hi / lo) if lo > 0 and hi > lo else 1.0
    if decades < 1.2 and lo > 0 and hi > lo:
        # Lower the bottom to the nearest whole multiple of a power of ten,
        # so a labelled tick always sits at or below the lowest value.
        step = 10.0 ** math.floor(math.log10(lo))
        lo = step * math.floor(lo / step + 1e-9) * 0.97
        ax.set_ylim(lo, hi)
        decades = math.log10(hi / lo)
    if decades > 7:
        ax.yaxis.set_major_locator(LogLocator(base=100.0, subs=(1.0,)))
    elif decades < 1.2:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=tuple(float(s) for s in range(1, 10))))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0) if decades <= 3.2 else (1.0,)))
    if decades > 3.2:
        # Ticks only at powers of ten: label every one the same way, 10^k.
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(round(math.log10(v)))}}}$"
                                                   if v > 0 else ""))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _value_label(v)))
    minor = 1.2 <= decades <= 7
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(float(s) for s in range(2, 10))) if minor
                               else NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _reserve_linear(ax: Axes, reserve: float) -> None:
    """Lower a linear y axis so the bottom ``reserve`` of the axes holds no data.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([1, 2], [10.0, 20.0])
        >>> _reserve_linear(ax, 0.25)
        >>> bool(ax.get_ylim()[0] < 10.0)
        True
        >>> plt.close(fig)
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * reserve / (1 - reserve), hi)


def _fonts(ax: Axes, sizes: Dict[str, float]) -> None:
    """Apply tick, axis-label and title sizes to one axes.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _fonts(ax, _FONT)
        >>> ax.title.get_size()
        14.0
        >>> plt.close(fig)
    """
    ax.tick_params(labelsize=sizes["tick"])
    ax.xaxis.label.set_size(sizes["label"])
    ax.yaxis.label.set_size(sizes["label"])
    ax.title.set_size(sizes["title"])


def _cap_note(ax: Axes, lines: Sequence[str], sizes: Dict[str, float]) -> None:
    """Write cap notes as text in the bottom-left corner (kept free of data by the caller).

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _cap_note(ax, ["naive: not run for n >= $2^{17}$ (cap)"], _FONT)
        >>> len(ax.texts)
        1
        >>> plt.close(fig)
    """
    if lines:
        ax.text(0.02, 0.03, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=sizes["legend"] - 0.5, style="italic", color="#212121",
                bbox={"facecolor": "white", "edgecolor": "#b3b3b3", "pad": 3.0})


def _cap_text(capped_rows: Sequence[Dict[str, Any]], what: str, powers: bool = True,
              unit: str = "n") -> Optional[str]:
    """One cap note for a series, from its capped rows, or ``None`` if it has none.

    Examples:
        >>> rows = [{"n": 2 ** 19}, {"n": 2 ** 20}, {"n": 2 ** 19}]
        >>> _cap_text(rows, "persistent tree")
        'persistent tree: not run for n = $2^{19}$ to $2^{20}$ (cap)'
        >>> _cap_text([{"n": 512}], "simulation", powers=False)
        'simulation: not run for n = 512 (cap)'
        >>> _cap_text([], "x") is None
        True
    """
    ns = sorted({int(r["n"]) for r in capped_rows})
    if not ns:
        return None
    first, last = _n_label(ns[0], powers), _n_label(ns[-1], powers)
    span = first if len(ns) == 1 else f"{first} to {last}"
    return f"{what}: not run for {unit} = {span} (cap)"


#: The ``gid`` that marks a legend :func:`_finish` keeps just under its axes' x label.
_BELOW = "legend-below-axes"


def _legend_below(ax: Axes, sizes: Dict[str, float], ncol: int = 1, offset: float = -0.2) -> None:
    """Put the axes' legend under it, where it cannot cover data (nothing if no entries).

    ``offset`` is only a first guess; :func:`_finish` moves the legend to
    just below the measured tick labels and axis label once the layout is
    known.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([1, 2], [1, 2], label="a")
        >>> _legend_below(ax, _FONT)
        >>> ax.get_legend() is not None
        True
        >>> plt.close(fig)
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol,
                           fontsize=sizes["legend"], frameon=True)
        legend.set_gid(_BELOW)


def _place_legends_below(fig: Figure) -> None:
    """Move every below-axes legend to just under its axes' tick labels and x label.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([1, 2], [1, 2], label="a")
        >>> _ = ax.set_xlabel("x")
        >>> _legend_below(ax, _FONT)
        >>> fig.canvas.draw()
        >>> _place_legends_below(fig)
        >>> bool(ax.get_legend().get_bbox_to_anchor().y0 < ax.get_window_extent().y0)
        True
        >>> plt.close(fig)
    """
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None or legend.get_gid() != _BELOW:
            continue
        box = ax.xaxis.get_tightbbox(renderer)
        if box is None:
            continue
        gap = 4.0 * fig.dpi / 72.0  # 4 points under the axis label
        y = ax.transAxes.inverted().transform((0.0, box.y0 - gap))[1]
        legend.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)


def _figure_legend(fig: Figure, axes: Sequence[Axes], sizes: Dict[str, float], ncol: int) -> None:
    """One legend for several panels, centred under the figure (duplicates merged).

    Examples:
        >>> fig, (a, b) = plt.subplots(1, 2)
        >>> _ = a.plot([1], [1], label="x"); _ = b.plot([1], [1], label="x")
        >>> _figure_legend(fig, (a, b), _FONT, 2)
        >>> len(fig.legends[0].get_texts())
        1
        >>> plt.close(fig)
    """
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
                   fontsize=sizes["legend"], frameon=True)


def _finish(fig: Figure, save_path: Optional[str]) -> Figure:
    """Lay out ``fig``, save it under 100 KB when a path is given, and return it.

    Layout and legend placement alternate a few times: the layout makes room
    for each below-axes legend, and each legend is then moved to just under
    its axes' labels, which the next layout pass accounts for.

    Examples:
        >>> fig = _finish(plt.figure(), None)
        >>> plt.close(fig)
    """
    for _ in range(3):
        fig.tight_layout()
        fig.canvas.draw()
        _place_legends_below(fig)
    save_png_under_limit(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_SEG = {"marker": "o", "linestyle": "-", "color": "#2471a3"}
_FEN = {"marker": "s", "linestyle": "-.", "color": "#d68910"}
_NAIVE = {"marker": "x", "linestyle": "--", "color": _REF_COLOR}


def plot_segment_vs_fenwick(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``segment_vs_fenwick.png``: build, update and range-sum cost against n.

    Panel (a) is the build time; (b) and (c) the time per point update and
    per range sum, each the mean of 10,000 operations. The naive reference
    (``sum`` over a list slice) is grey and dashed; the sizes it was not
    run at are named in a text note.

    Args:
        rows: Rows read from ``benchmark.csv`` (other experiments are ignored).
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data; an empty panel stays empty.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_segment_vs_fenwick([])
        >>> len(fig.axes)
        3
        >>> plt.close(fig)
    """
    E = "segment_vs_fenwick"
    sizes = _FONT_MULTI
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    panels = (("build", "(a) Build", "Build time (ms, log scale)", False),
              ("point_update", "(b) Point updates", "Time per update (microseconds, log scale)", True),
              ("range_sum", "(c) Range sums", "Time per range sum (microseconds, log scale)", True))
    all_n: List[int] = []
    for ax, (method, title, ylabel, per_op) in zip(axes, panels):
        series = [("segment_tree", _SEG, "segment tree (array, lazy tags)"),
                  ("fenwick_tree", _FEN, "Fenwick tree")]
        if method == "range_sum":
            series.append(("naive_reference", _NAIVE, "naive reference: sum over a list slice"))
        for structure, style, label in series:
            mine = _pick(rows, E, structure, method)
            all_n += [r["n"] for r in mine]
            ys = [_per_op_us(r) if per_op else r["mean_time_s"] * 1e3 for r in mine]
            _line(ax, [r["n"] for r in mine], ys, style, label)
        notes = [t for t in (_cap_text(_pick(rows, E, s, method, capped_rows=True), label.split(":")[0])
                             for s, _, label in series) if t]
        _log_y(ax, reserve=0.22 if notes else 0.0)
        _cap_note(ax, notes, sizes)
        ax.set_title(title)
        ax.set_xlabel("Array size n (log scale)")
        ax.set_ylabel(ylabel)
    for ax in axes:
        _size_axis(ax, all_n)
        _fonts(ax, sizes)
    _figure_legend(fig, axes, sizes, ncol=3)
    return _finish(fig, save_path)


def plot_persistent_overhead(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``persistent_overhead.png``: what versions cost, in nodes, memory and query time.

    (a) New nodes per point set against the path length ``log2 n + 1``
    (grey reference). (b) Nodes held for all versions: the persistent tree
    (build plus every update's path), one version (``2n - 1``) and a full
    copy per version (a grey dashed model). (c) ``tracemalloc`` peaks:
    persistent build, persistent after the point sets, plain array segment
    tree. (d) Time per range sum on the newest version against the plain
    segment tree. Capped sizes are named in text notes.

    Args:
        rows: Rows read from ``benchmark.csv``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_persistent_overhead([])
        >>> len(fig.axes)
        4
        >>> plt.close(fig)
    """
    E = "persistent_overhead"
    sizes = _FONT_MULTI
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.8))
    (a, b), (c, d) = axes
    sets = _pick(rows, E, "persistent_segment_tree", "point_set")
    builds = {r["n"]: r for r in _pick(rows, E, "persistent_segment_tree", "build")}
    copies = _pick(rows, E, "full_copy_model", "copy_per_version")
    updates = sets[0]["ops"] if sets else None
    versions = int(copies[0]["overhead_ratio"]) if copies else None
    cap_rows = _pick(rows, E, "persistent_segment_tree", None, capped_rows=True)
    note = _cap_text(cap_rows, "persistent tree")
    notes = [note] if note else []
    all_n = sorted({r["n"] for r in _pick(rows, E)})

    ns = [r["n"] for r in sets]
    if ns:
        # Drawn first and wide: the measured points fall exactly on it.
        a.plot(ns, [math.log2(n) + 1 for n in ns], color=_REF_COLOR, linestyle="--", linewidth=5.0, alpha=0.6,
               label="log2 n + 1, the root-to-leaf path (reference)")
    _line(a, ns, [r["nodes_created"] / r["ops"] for r in sets], _SEG, "persistent tree: new nodes per point set")
    a.set_title("(a) Nodes copied per update")
    a.set_ylabel("New nodes per point set")
    if ns:
        # From 0, so the path lengths (at least 11 nodes) sit in the upper half
        # and the cap note in the lower-left corner covers nothing.
        a.set_ylim(0, (math.log2(max(ns)) + 1) * 1.12)

    stored = [(r["n"], builds[r["n"]]["nodes_created"] + r["nodes_created"]) for r in sets if r["n"] in builds]
    upd_text = f"{updates:,} point sets" if updates else "the point sets"
    ver_text = f"{versions:,} versions" if versions else "all versions"
    _line(b, [x for x, _ in stored], [y for _, y in stored], _SEG, f"persistent tree, {ver_text} (measured)")
    _line(b, list(builds), [r["nodes_created"] for r in builds.values()], _FEN, "one version, 2n - 1 nodes")
    _line(b, [r["n"] for r in copies], [r["nodes_created"] for r in copies],
          {"marker": "x", "linestyle": "--", "color": _REF_COLOR},
          f"full copy per version, {ver_text} x (2n - 1) (model)")
    b.set_title(f"(b) Nodes held for {ver_text}")
    b.set_ylabel("Nodes (log scale)")

    mem_build = list(builds.values())
    _line(c, [r["n"] for r in sets], [r["memory_kib"] for r in sets], _SEG,
          f"persistent tree after {upd_text}")
    _line(c, [r["n"] for r in mem_build], [r["memory_kib"] for r in mem_build], _FEN, "persistent tree, build only")
    plain = _pick(rows, E, "segment_tree", "build")
    _line(c, [r["n"] for r in plain], [r["memory_kib"] for r in plain],
          {"marker": "^", "linestyle": ":", "color": "#7d3c98"}, "plain array segment tree, build")
    c.set_title("(c) Peak traced memory (tracemalloc)")
    c.set_ylabel("Peak memory (KiB, log scale)")

    pq = _pick(rows, E, "persistent_segment_tree", "query_sum")
    sq = _pick(rows, E, "segment_tree", "query_sum")
    _line(d, [r["n"] for r in pq], [_per_op_us(r) for r in pq], _SEG, "persistent tree, newest version")
    _line(d, [r["n"] for r in sq], [_per_op_us(r) for r in sq],
          {"marker": "^", "linestyle": ":", "color": "#7d3c98"}, "plain array segment tree")
    d.set_title("(d) Range-sum query time")
    d.set_ylabel("Time per range sum (microseconds, log scale)")
    for ax in (b, c, d):
        _log_y(ax, reserve=0.22 if notes else 0.0)
    for ax in (a, b, c, d):
        _cap_note(ax, notes, sizes)
        ax.set_xlabel("Array size n (log scale)")
        _size_axis(ax, all_n)
        _fonts(ax, sizes)
        _legend_below(ax, sizes, offset=-0.17)
    return _finish(fig, save_path)


_RS_STYLES = {
    ("rank1", 0.5): {"marker": "o", "linestyle": "-", "color": "#2471a3"},
    ("rank1", 0.1): {"marker": "s", "linestyle": "-.", "color": "#5dade2", "hollow": True, "size": 11},
    ("select1", 0.5): {"marker": "^", "linestyle": "-", "color": "#d68910"},
    ("select1", 0.1): {"marker": "v", "linestyle": "-.", "color": "#a04000", "hollow": True, "size": 11},
    ("rank1_bit_count", 0.5): {"marker": "x", "linestyle": "--", "color": _REF_COLOR},
    ("rank1_bit_count", 0.1): {"marker": "+", "linestyle": (0, (2, 2)), "color": "#5d6d7e", "size": 11},
}


def plot_rank_select(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``rank_select.png``: query time and space overhead against the bit-vector length.

    (a) Time per ``rank1`` and per ``select1`` at both densities, with the
    naive ``int.bit_count`` prefix count as grey dashed references. (b) The
    rank directory and the select samples (the select1 and the select0
    samples together) as a percentage of n, against the
    7.8125% the spec 4.2 layout predicts (grey dashed, drawn wide underneath
    because the measured directory falls on it). The 0.1-density series are
    drawn first with large hollow markers, so a coinciding 0.5 point shows
    inside its ring.

    Args:
        rows: Rows read from ``benchmark.csv``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_rank_select([])
        >>> len(fig.axes)
        2
        >>> plt.close(fig)
    """
    E = "rank_select"
    sizes = _FONT_MULTI
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    # The sparser density is drawn first with large hollow markers, so where the
    # two densities coincide both stay visible.
    densities = sorted({r["density"] for r in rows if r["experiment"] == E and r["density"] is not None})
    notes = []
    for method, structure, what in (("rank1", "rank_select", "rank1"), ("select1", "rank_select", "select1"),
                                    ("rank1_bit_count", "naive_reference", "naive rank (int.bit_count, reference)")):
        for d in densities:
            mine = _pick(rows, E, structure, method, density=d)
            style = _RS_STYLES.get((method, round(d, 3)), {"marker": "D", "linestyle": ":", "color": "#212121"})
            _line(a, [r["n"] for r in mine], [_per_op_us(r) for r in mine], style, f"{what}, density {d:g}")
        if method == "rank1_bit_count":
            note = _cap_text(_pick(rows, E, structure, method, capped_rows=True), "naive rank")
            if note:
                notes.append(note)
    _log_y(a, reserve=0.22 if notes else 0.0)
    _cap_note(a, notes, sizes)
    a.set_title("(a) Time per query (10,000 random queries)")
    a.set_ylabel("Time per query\n(microseconds, log scale)")
    measured = _pick(rows, E, "rank_select", "rank1")
    if measured:
        ns = sorted({r["n"] for r in measured})
        b.plot(ns, [7.8125] * len(ns), color=_REF_COLOR, linestyle="--", linewidth=5.0, alpha=0.6,
               label="7.8125%: 32 bits per 2,048 + 16 per 256 (reference)")
    for method, what in (("rank1", "rank directory"), ("select1", "select samples (1s and 0s)")):
        for d in densities:
            mine = _pick(rows, E, "rank_select", method, density=d)
            style = _RS_STYLES.get((method, round(d, 3)), {"marker": "D", "linestyle": ":", "color": "#212121"})
            _line(b, [r["n"] for r in mine], [r["overhead_ratio"] * 100 if r["overhead_ratio"] is not None else None
                                              for r in mine], style, f"{what}, density {d:g}")
    if measured:
        top = max([r["overhead_ratio"] * 100 for r in _pick(rows, E, "rank_select") if r["overhead_ratio"]]
                  + [7.8125])
        b.set_ylim(0, top * 1.25)
    b.set_title("(b) Space overhead")
    b.set_ylabel("Extra bits as % of n")
    all_n = [r["n"] for r in rows if r["experiment"] == E]
    for ax in (a, b):
        ax.set_xlabel("Bit-vector length n (log scale)")
        _size_axis(ax, all_n)
        _fonts(ax, sizes)
        _legend_below(ax, sizes, offset=-0.17)
    return _finish(fig, save_path)


def _pow10_label(n: float) -> str:
    """Tick text ``$10^{k}$`` for a power of ten, else the number with separators.

    Examples:
        >>> _pow10_label(100000), _pow10_label(76754)
        ('$10^{5}$', '76,754')
    """
    n = int(n)
    k = math.log10(n) if n > 0 else 0.5
    return f"$10^{{{int(round(k))}}}$" if n > 0 and abs(k - round(k)) < 1e-9 else f"{n:,}"


def _datasets(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    """The wavelet datasets present: ``(label, experiment, n)``, DNA by size then the text.

    Examples:
        >>> rows = [row("wavelet_text", "wavelet_balanced", "space", 70000),
        ...         row("wavelet_dna", "wavelet_balanced", "space", 10000)]
        >>> [d[0] for d in _datasets(rows)]
        ['DNA\\n$10^{4}$', 'README\\ntext']
    """
    out = []
    for E, name in (("wavelet_dna", "DNA"), ("wavelet_text", "README")):
        for n in sorted({r["n"] for r in rows if r["experiment"] == E and r["method"] == "space"}):
            if E == "wavelet_dna":
                out.append((f"{name}\n{_pow10_label(n)}", E, n))
            else:
                out.append(("README\ntext", E, n))
    return out


def plot_wavelet_compression(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``wavelet_compression.png``: space per symbol and query time of the wavelet trees.

    (a) Payload bits per symbol of the balanced and Huffman shapes beside
    the zero-order entropy H0 (grey, hatched, a reference) for each
    dataset. (b) Compression ratio, payload plus rank directories against
    one byte per symbol (grey dashed line at 1). (c) On synthetic DNA, time
    per ``range_count``, ``range_quantile`` and FM ``pattern_count``, with
    the naive scans as grey dashed references; capped sizes in a note.

    Args:
        rows: Rows read from ``benchmark.csv``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_wavelet_compression([])
        >>> len(fig.axes)
        3
        >>> plt.close(fig)
    """
    sizes = _FONT_MULTI
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.0, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    data = _datasets(rows)
    xs = np.arange(len(data))
    bars = (("wavelet_balanced", "space", "bits_per_symbol", "balanced shape", "#2471a3", "//"),
            ("wavelet_huffman", "space", "bits_per_symbol", "Huffman shape", "#d68910", ".."),
            ("zero_order_entropy", "h0", "bits_per_symbol", "zero-order entropy H0 (reference)", _REF_COLOR, "xx"))
    width = 0.27
    top = 0.0
    for j, (structure, method, column, label, colour, hatch) in enumerate(bars):
        values = []
        for _, E, n in data:
            hit = [r for r in _pick(rows, E, structure, method) if r["n"] == n]
            values.append(hit[0][column] if hit else None)
        present = [(x, v) for x, v in zip(xs, values) if v is not None]
        if present:
            pos = [x + (j - 1) * width for x, _ in present]
            a.bar(pos, [v for _, v in present], width, color=colour, hatch=hatch, edgecolor="#212121",
                  linewidth=0.8, label=label, alpha=0.9)
            for p, (_, v) in zip(pos, present):
                a.text(p, v + 0.08, f"{v:.2f}", ha="center", va="bottom", rotation=90, fontsize=10.5)
            top = max(top, max(v for _, v in present))
    a.set_ylim(0, top * 1.35 if top else 1)
    a.set_title("(a) Payload bits per symbol")
    a.set_ylabel("Bits per symbol")
    top = 0.0
    for j, (structure, label, colour, hatch) in enumerate((("wavelet_balanced", "balanced shape", "#2471a3", "//"),
                                                         ("wavelet_huffman", "Huffman shape", "#d68910", ".."))):
        values = []
        for _, E, n in data:
            hit = [r for r in _pick(rows, E, structure, "space") if r["n"] == n]
            values.append(hit[0]["compression_ratio"] if hit else None)
        present = [(x, v) for x, v in zip(xs, values) if v is not None]
        if present:
            pos = [x + (j - 0.5) * width for x, _ in present]
            b.bar(pos, [v for _, v in present], width, color=colour, hatch=hatch, edgecolor="#212121",
                  linewidth=0.8, label=label, alpha=0.9)
            for p, (_, v) in zip(pos, present):
                b.text(p, v + 0.02, f"{v:.2f}", ha="center", va="bottom", rotation=90, fontsize=10.5)
            top = max(top, max(v for _, v in present))
    if data:
        b.axhline(1.0, color=_REF_COLOR, linestyle="--", linewidth=1.8, label="one byte per symbol (reference)")
    b.set_ylim(0, max(top, 1.0) * 1.3)
    b.set_title("(b) Compression ratio")
    b.set_ylabel("(payload + directories) / 8n")
    for ax in (a, b):
        ax.set_xticks(xs)
        ax.set_xticklabels([d[0] for d in data])
        ax.set_xlabel("Dataset")
        ax.grid(axis="x", visible=False)
    E = "wavelet_dna"
    series = (("wavelet_balanced", "range_count", "range_count, balanced", _SEG),
              ("wavelet_huffman", "range_count", "range_count, Huffman", _FEN),
              ("wavelet_balanced", "range_quantile", "range_quantile, balanced",
               {"marker": "^", "linestyle": ":", "color": "#7d3c98"}),
              ("fm_index", "pattern_count", "FM pattern_count", {"marker": "D", "linestyle": "-", "color": "#1e8449"}),
              ("naive_reference", "range_count_scan", "naive count (str.count, reference)", _NAIVE),
              ("naive_reference", "pattern_count_scan", "naive pattern scan (str.find, reference)",
               {"marker": "+", "linestyle": (0, (2, 2)), "color": "#5d6d7e"}))
    for structure, method, label, style in series:
        mine = _pick(rows, E, structure, method)
        _line(c, [r["n"] for r in mine], [_per_op_us(r) for r in mine], style, label)
    notes = [t for t in (_cap_text(_pick(rows, E, "naive_reference", m, capped_rows=True), what, powers=False)
                         for m, what in (("range_count_scan", "naive count"), ("pattern_count_scan", "naive scan")))
             if t]
    c.set_xscale("log")
    dna_n = sorted({r["n"] for r in rows if r["experiment"] == E})
    if dna_n:
        c.xaxis.set_major_locator(FixedLocator(dna_n))
        c.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _pow10_label(v)))
        c.xaxis.set_minor_locator(NullLocator())
        c.set_xlim(dna_n[0] / 2, dna_n[-1] * 2)
    _log_y(c, reserve=0.22 if notes else 0.0)
    _cap_note(c, notes, sizes)
    c.set_title("(c) Query time on synthetic DNA")
    c.set_xlabel("Sequence length n (log scale)")
    c.set_ylabel("Time per query\n(microseconds, log scale)")
    for ax, offset in ((a, -0.32), (b, -0.32), (c, -0.26)):
        _fonts(ax, sizes)
        _legend_below(ax, sizes, offset=offset)
    return _finish(fig, save_path)


_CACHE_MARKERS = ("s", "o", "D", "^", "P", "v")
_CACHE_COLOURS = ("#2471a3", "#d68910", "#7d3c98")


def plot_cache_misses(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``cache_misses.png``: simulated LRU misses, transpose and multiply panels.

    One line per cache setting ``(M, B)`` and method: recursive solid with
    small filled markers, naive dotted with large hollow markers (drawn
    first, so a coinciding recursive point shows inside it), each setting
    its own colour and marker pair. The sizes the simulation was capped at are
    named in a text note.

    Args:
        rows: Rows read from ``benchmark.csv``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_cache_misses([])
        >>> len(fig.axes)
        2
        >>> plt.close(fig)
    """
    E = "cache_misses"
    sizes = _FONT_MULTI
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    settings = sorted({(r["cache_M"], r["cache_B"]) for r in rows if r["experiment"] == E
                       and r["cache_M"] is not None})
    for ax, op, title in ((axes[0], "transpose", "(a) Transpose, n x n"), (axes[1], "matmul", "(b) Multiply, n x n")):
        for s, (M, B) in enumerate(settings):
            for m, method in enumerate(("naive", "recursive")):
                # Naive first with large hollow markers, recursive on top with small
                # filled ones, so both stay visible where the counts coincide.
                mine = _pick(rows, E, op, method, cache_M=M, cache_B=B)
                style = {"marker": _CACHE_MARKERS[(2 * s + m) % len(_CACHE_MARKERS)],
                         "linestyle": "-" if method == "recursive" else ":",
                         "color": _CACHE_COLOURS[s % len(_CACHE_COLOURS)], "hollow": method == "naive",
                         "size": 11 if method == "naive" else 6.5}
                _line(ax, [r["n"] for r in mine], [r["misses"] for r in mine], style,
                      f"{method}, M = {M:,}, B = {B}")
        note = _cap_text(_pick(rows, E, op, None, capped_rows=True), "simulation", powers=False)
        notes = [note] if note else []
        _log_y(ax, reserve=0.2 if notes else 0.0)
        _cap_note(ax, notes, sizes)
        ax.set_title(title)
        ax.set_xlabel("Matrix size n (log scale)")
        ax.set_ylabel("Simulated cache misses (log scale)")
        _size_axis(ax, [r["n"] for r in rows if r["experiment"] == E and r["structure"] == op], powers=False)
        _fonts(ax, sizes)
    _figure_legend(fig, axes, sizes, ncol=3)
    return _finish(fig, save_path)


def plot_layout_blocks(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``layout_blocks.png``: distinct blocks per search against tree height, BFS and vEB.

    Args:
        rows: Rows read from ``benchmark.csv``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_layout_blocks([])
        >>> len(fig.axes)
        1
        >>> plt.close(fig)
    """
    E = "layout_blocks"
    sizes = _FONT
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    blocks = sorted({r["cache_B"] for r in rows if r["experiment"] == E and r["cache_B"] is not None})
    styles = {("bfs", 0): {"marker": "^", "linestyle": ":", "color": "#c0392b"},
              ("bfs", 1): {"marker": "v", "linestyle": "-.", "color": "#d68910"},
              ("veb", 0): {"marker": "o", "linestyle": "-", "color": "#2471a3"},
              ("veb", 1): {"marker": "s", "linestyle": (0, (5, 1, 1, 1, 1, 1)), "color": "#1e8449"}}
    for layout, name in (("bfs", "BFS (heap) order"), ("veb", "van Emde Boas order")):
        for j, B in enumerate(blocks):
            mine = _pick(rows, E, f"{layout}_layout", "random_search", cache_B=B)
            style = styles.get((layout, j), {"marker": "D", "linestyle": "-", "color": "#212121"})
            _line(ax, [round(math.log2(r["n"] + 1)) for r in mine], [r["blocks_per_search"] for r in mine],
                  style, f"{name}, B = {B}")
    notes = [t for t in (_cap_text(_pick(rows, E, None, None, capped_rows=True), "layouts", powers=False,
                                   unit="keys"),) if t]
    heights = sorted({round(math.log2(r["n"] + 1)) for r in rows if r["experiment"] == E})
    if heights:
        ax.set_xticks(heights)
        ax.set_ylim(bottom=0)
        if notes:
            _reserve_linear(ax, 0.2)
    _cap_note(ax, notes, sizes)
    ax.set_title("Blocks touched per random successful search")
    ax.set_xlabel("Tree height h (levels; 2^h - 1 keys)")
    ax.set_ylabel("Distinct blocks per search")
    _fonts(ax, sizes)
    _legend_below(ax, sizes, ncol=2, offset=-0.14)
    return _finish(fig, save_path)


def plot_vcs_history(rows: Sequence[Dict[str, Any]], save_path: Optional[str] = None) -> Figure:
    """Draw ``vcs_history.png``: per-commit sharing and Merkle-pruned diff cost.

    (a) For each commit, the files changed against its parent and the new
    tree nodes it stored, beside the nodes of its whole tree (what a full
    copy of the commit would store, grey dashed reference). (b) The nodes
    the diff against the parent visited, beside the nodes of both trees
    (what a full comparison reads, grey dashed reference).

    Args:
        rows: Rows read from ``benchmark.csv``.
        save_path: PNG path, or ``None`` to skip saving.

    Returns:
        The figure.

    Raises:
        Nothing for missing data.

    Time Complexity:
        O(rows).

    Space Complexity:
        O(rows).

    Examples:
        >>> fig = plot_vcs_history([])
        >>> len(fig.axes)
        2
        >>> plt.close(fig)
    """
    E = "vcs_history"
    sizes = _FONT_MULTI
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.8))
    commits = _pick(rows, E, "mini_vcs", "commit")
    trees = {r["n"]: r["nodes_created"] for r in _pick(rows, E, "mini_vcs", "tree_nodes")}
    diffs = _pick(rows, E, "mini_vcs", "diff_vs_parent")
    _line(a, [r["n"] for r in commits], [r["ops"] for r in commits],
          {"marker": "^", "linestyle": ":", "color": "#7d3c98"}, "files changed against the parent")
    _line(a, [r["n"] for r in commits], [r["nodes_created"] for r in commits], _SEG, "new tree nodes stored")
    _line(a, list(trees), list(trees.values()), {"marker": "x", "linestyle": "--", "color": _REF_COLOR},
          "nodes in the commit's tree, a full copy (reference)")
    _line(b, [r["n"] for r in diffs], [r["ops"] for r in diffs], _SEG, "nodes the diff visited")
    both = [(r["n"], trees.get(r["n"] - 1, 0) + trees.get(r["n"], 0)) for r in diffs
            if r["n"] - 1 in trees and r["n"] in trees]
    _line(b, [x for x, _ in both], [y for _, y in both], {"marker": "x", "linestyle": "--", "color": _REF_COLOR},
          "nodes in both trees, a full comparison (reference)")
    a.set_title("(a) What each commit stores")
    a.set_ylabel("Files or tree nodes (log scale)")
    b.set_title("(b) Diff against the parent commit")
    b.set_ylabel("Tree nodes (log scale)")
    for ax in (a, b):
        _log_y(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Commit number")
        _fonts(ax, sizes)
        _legend_below(ax, sizes, offset=-0.17)
    return _finish(fig, save_path)


#: Figure file name -> drawing function, in the order :data:`FIGURES` lists them.
_PLOTS: Tuple[Tuple[str, Callable[..., Figure]], ...] = (
    ("segment_vs_fenwick.png", plot_segment_vs_fenwick),
    ("persistent_overhead.png", plot_persistent_overhead),
    ("rank_select.png", plot_rank_select),
    ("wavelet_compression.png", plot_wavelet_compression),
    ("cache_misses.png", plot_cache_misses),
    ("layout_blocks.png", plot_layout_blocks),
    ("vcs_history.png", plot_vcs_history),
)


def figures(out: str, rows: Sequence[Dict[str, Any]], log: Callable[[str], None] = _quiet) -> List[str]:
    """Draw every figure from ``rows`` into ``out`` and return the paths written.

    Args:
        out: The output directory (created if missing).
        rows: Rows as :func:`read_csv` returns them.
        log: Where one line per figure goes.

    Returns:
        The PNG paths, in :data:`FIGURES` order.

    Raises:
        OSError: If a file cannot be written.

    Time Complexity:
        O(rows) per figure plus the rendering.

    Space Complexity:
        O(rows) plus one rendered figure at a time.

    Examples:
        >>> import tempfile
        >>> paths = figures(tempfile.mkdtemp(), [])
        >>> [os.path.basename(p) for p in paths] == list(FIGURES)
        True
    """
    os.makedirs(out, exist_ok=True)
    paths = []
    for name, plot in _PLOTS:
        path = os.path.join(out, name)
        fig = plot(rows, save_path=path)
        plt.close(fig)
        paths.append(path)
        log(f"  wrote {name} ({os.path.getsize(path) / 1024:.0f} KiB)")
    return paths


def redraw(out: str, log: Callable[[str], None] = _quiet) -> List[str]:
    """Redraw every figure from ``out/benchmark.csv`` without measuring anything.

    The CSV is only read, so it stays byte for byte, and the figures are
    the ones a measured run draws (it draws from the CSV as read back too).

    Args:
        out: The directory holding ``benchmark.csv``.
        log: Where progress lines go.

    Returns:
        The PNG paths written.

    Raises:
        OSError: If the CSV cannot be read.

    Time Complexity:
        O(rows) plus the rendering.

    Space Complexity:
        O(rows).

    Examples:
        >>> import tempfile
        >>> out = tempfile.mkdtemp()
        >>> write_csv(os.path.join(out, "benchmark.csv"), [])
        >>> len(redraw(out))
        7
    """
    rows = read_csv(os.path.join(out, "benchmark.csv"))
    log(f"\n=== redrawing {len(FIGURES)} figures from {len(rows)} CSV rows (nothing is measured) ===")
    return figures(out, rows, log)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the Week 13 benchmark from the command line.

    Flags: ``--smoke`` (tiny sizes), ``--out DIR`` (default
    ``benchmarks/results/week13``), ``--seed`` (default 42) and
    ``--redraw`` (figures from ``OUT/benchmark.csv`` only).

    Args:
        argv: The arguments (``None`` for ``sys.argv[1:]``).

    Returns:
        0 on success.

    Raises:
        RuntimeError: If any checked answer differs from its reference.
        SystemExit: On ``--help`` or a bad flag (from argparse).

    Time Complexity:
        Under five minutes for the full study; seconds with ``--smoke``.

    Space Complexity:
        Dominated by the largest structures: 2^24 bits and 2^20-element trees.

    Examples:
        >>> import tempfile
        >>> out = tempfile.mkdtemp()
        >>> write_csv(os.path.join(out, "benchmark.csv"), [])
        >>> main(["--redraw", "--out", out])  # doctest: +ELLIPSIS
        <BLANKLINE>
        === redrawing 7 figures from 0 CSV rows (nothing is measured) ===
          wrote segment_vs_fenwick.png ...
        0
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, finishes in seconds")
    parser.add_argument("--out", default=RESULTS_DIR, help="directory for benchmark.csv and the figures")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed (default 42)")
    parser.add_argument("--redraw", action="store_true",
                        help="rebuild every figure from OUT/benchmark.csv without measuring anything")
    args = parser.parse_args(argv)
    out = os.path.abspath(args.out)
    apply_house_style()
    if args.redraw:
        redraw(out, _console)
        return 0

    cfg = SMOKE if args.smoke else FULL
    os.makedirs(out, exist_ok=True)
    _console("=" * 78)
    _console("CSC 5300 Week 13 - persistent, succinct and cache-oblivious structures - Robert Deibel")
    _console("=" * 78)
    _console(f"  python      : {platform.python_version()} ({platform.python_implementation()})")
    _console(f"  platform    : {platform.platform()}")
    _console(f"  mode        : {'SMOKE RUN' if args.smoke else 'full study'}")
    _console(f"  master seed : {args.seed}  (inputs: derive_seed(master, experiment, n, ...))")
    _console(f"  protocol    : 1 untimed warm-up + {cfg.runs} timed runs per method and size "
             f"(numpy references >= {cfg.reference_runs}); {cfg.cap_s:.0f} s cap per run")
    _console(f"  output      : {out}")
    started = time.perf_counter()
    study = Study(cfg, args.seed, log=_console)
    for experiment in STUDIES:
        experiment(study)
    path = os.path.join(out, "benchmark.csv")
    write_csv(path, study.rows)
    _console("\n=== results ===")
    caps = [r for r in study.rows if r["capped"]]
    _console(f"  {len(study.rows)} rows, {len(caps)} capped (reasons printed above as [cap])")
    figures(out, read_csv(path), _console)
    elapsed = time.perf_counter() - started
    _console("\n" + "=" * 78)
    _console(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    _console("=" * 78)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper benchmarks/week13_benchmark.py is the entry point
    sys.exit(main())
