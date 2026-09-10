#!/usr/bin/env python3
"""Week 3 benchmark: a binary heap, an AVL tree and hash tables, timed and counted.

Run from the repository root with the project virtualenv active::

    python benchmarks/week3_structures_benchmark.py            # the full study
    python benchmarks/week3_structures_benchmark.py --quick    # smoke run
    python benchmarks/week3_structures_benchmark.py --quick --out /tmp/w3

Five studies, all driven through :meth:`AlgorithmBenchmark.time_operation`,
the Week 1 timing framework extended for data structures:

1. **Main matrix** - insert, search and delete at n = 10^3, 10^4, 10^5 and
   10^6 for the Week 3 heap, AVL tree and both hash-table strategies, with
   Python's ``heapq``, ``dict`` and ``list`` as baselines. Reported *per
   operation*, because per-operation cost is what separates O(1), O(log n)
   and O(n).
2. **Load factor** - both hash strategies filled to fixed load factors from
   0.1 to 0.95 at a fixed capacity, measuring probes per lookup against
   Knuth's formulas and time per lookup.
3. **Worst case** - keys engineered to share one slot at every capacity,
   which turns O(1) lookups into a measured O(n) - and ``dict``, which
   perturbs its hash, as the control.
4. **Amortised rehashing** - a table grown one insert at a time, with the
   cost of every insert and every rebuild recorded.
5. **AVL balance** - measured tree height against the perfect and AVL
   bounds, for random and for sorted insertion.

Repetitions fall with size: 5, 5, 3 and 1 measured runs at 10^3, 10^4, 10^5
and 10^6, after 2, 2, 1 and 0 discarded warm-ups. A single unrepeated run at
10^6 is a deliberate trade - several configurations take seconds there -
and the count is written into every CSV row, so it never has to be guessed.

Every structure is checked for correctness at every size, off the clock,
before any of its timings are taken.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import argparse
import csv
import gc
import heapq
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.structures.avl_tree import AVLTree, avl_max_height, perfect_height  # noqa: E402
from src.structures.hash_table import CHAINING, LINEAR_PROBING, HashTable  # noqa: E402
from src.structures.heap import MinHeap  # noqa: E402
from src.utils.benchmark import AlgorithmBenchmark, BenchmarkResult  # noqa: E402
from src.utils.visualization import apply_house_style  # noqa: E402

#: Base seed. Every workload derives a deterministic seed from it.
SEED = 2026

#: Keys are drawn from this range, so they are distinct and their low bits -
#: which is all a power-of-two hash table looks at - are uniform.
KEY_SPACE = 10 ** 12

RESULTS_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")

CLASSES = ("O(1)", "O(log n)", "O(n)")


@dataclass
class Config:
    """Every size and repetition count the study uses, in one place."""

    sizes: List[int]
    runs: Dict[int, int]
    warmups: Dict[int, int]
    #: A list lookup is O(n), so n of them would be O(n^2): time a fixed
    #: sample instead and report per operation.
    list_ops: Dict[int, int]
    load_capacity: int = 2 ** 16
    load_factors: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )
    load_seeds: int = 3
    load_lookups: int = 10_000
    worst_sizes: List[int] = field(default_factory=lambda: [250, 500, 1000, 2000, 4000])
    amortized_n: int = 2 ** 17
    balance_sizes: Optional[List[int]] = None


FULL = Config(
    sizes=[10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6],
    runs={10 ** 3: 5, 10 ** 4: 5, 10 ** 5: 3, 10 ** 6: 1},
    warmups={10 ** 3: 2, 10 ** 4: 2, 10 ** 5: 1, 10 ** 6: 0},
    list_ops={10 ** 3: 1000, 10 ** 4: 1000, 10 ** 5: 300, 10 ** 6: 100},
)

QUICK = Config(
    sizes=[10 ** 3, 10 ** 4],
    runs={10 ** 3: 2, 10 ** 4: 2},
    warmups={10 ** 3: 1, 10 ** 4: 1},
    list_ops={10 ** 3: 500, 10 ** 4: 500},
    load_factors=[0.25, 0.5, 0.75],
    load_seeds=1,
    load_lookups=2000,
    worst_sizes=[250, 500, 1000],
    amortized_n=2 ** 12,
)


# ----------------------------------------------------------------------
# Workloads and builders
# ----------------------------------------------------------------------
class Workload:
    """The keys for one size, and the orders they are used in."""

    def __init__(self, size: int, list_ops: int, rng: random.Random) -> None:
        self.n = size
        self.keys = rng.sample(range(KEY_SPACE), size)
        self.lookups = self.keys[:]
        rng.shuffle(self.lookups)
        self.deletes = self.keys[:]
        rng.shuffle(self.deletes)
        self.descending = sorted(self.keys, reverse=True)
        self.list_probe = self.lookups[:list_ops]


def build_avl(keys: List[int]) -> AVLTree:
    tree = AVLTree()
    insert = tree.insert
    for key in keys:
        insert(key)
    return tree


def build_table(keys: List[int], strategy: str) -> HashTable:
    table = HashTable(strategy=strategy)
    insert = table.insert
    for key in keys:
        insert(key)
    return table


def build_heapq(keys: List[int]) -> List[int]:
    heap = list(keys)
    heapq.heapify(heap)
    return heap


def _get_all(table: Any, keys: List[int]) -> None:
    get = table.get
    for key in keys:
        get(key)


def verify(workload: Workload) -> None:
    """Check every structure is correct at this size, before any timing."""
    keys, n = workload.keys, workload.n
    expected = sorted(keys)
    sample = workload.lookups[: min(n, 20_000)]

    heap = MinHeap(keys)
    assert heap.is_valid()
    assert [heap.extract_min() for _ in range(n)] == expected, "MinHeap order"
    reference = build_heapq(keys)
    assert [heapq.heappop(reference) for _ in range(n)] == expected

    tree = build_avl(keys)
    tree.validate()
    assert tree.height() <= avl_max_height(n)
    assert all(key in tree for key in sample), "AVL search"
    for key in workload.deletes:
        tree.delete(key)
    assert tree.is_empty(), "AVL delete"

    for strategy in (CHAINING, LINEAR_PROBING):
        table = build_table(keys, strategy)
        assert len(table) == n and all(key in table for key in sample), strategy
        for key in workload.deletes:
            table.delete(key)
        assert len(table) == 0, strategy


# ----------------------------------------------------------------------
# Study 1: the main matrix
# ----------------------------------------------------------------------
SeriesKey = Tuple[str, str]


class Matrix:
    """Collects main-matrix results, keyed by (structure, operation) then n."""

    def __init__(self, bench: AlgorithmBenchmark, config: Config) -> None:
        self.bench = bench
        self.config = config
        self.results: Dict[SeriesKey, Dict[int, BenchmarkResult]] = {}
        self.meta: Dict[SeriesKey, Tuple[str, str]] = {}

    def measure(
        self,
        structure: str,
        operation: str,
        asymptotic: str,
        expected: str,
        size: int,
        setup: Optional[Callable[[], Any]],
        run: Callable[[Any], Any],
        ops: int,
    ) -> BenchmarkResult:
        """Time one (structure, operation, size) cell and record it.

        Args:
            asymptotic: The textbook bound, as stated in the table.
            expected: The per-operation class this measurement should
                show - O(1), O(log n) or O(n). Usually the same as the
                bound; different where the bound is a worst case that
                random input does not reach (heap insert) or a total that
                spreads across n elements (heapify).
        """
        started = time.perf_counter()
        result = self.bench.time_operation(
            run,
            setup,
            runs=self.config.runs[size],
            input_size=size,
            ops_per_run=ops,
            name=f"{structure} | {operation}",
            warmup_runs=self.config.warmups[size],
        )
        result.metadata.update(
            {
                "structure": structure,
                "operation": operation,
                "data_type": "descending" if "descending" in operation else "random",
            }
        )
        self.results.setdefault((structure, operation), {})[size] = result
        self.meta[(structure, operation)] = (asymptotic, expected)
        print(
            f"    {structure:<22} {operation:<20} n={size:<9,} "
            f"{per_op_ns(result):>12,.1f} ns/op  runs={self.config.runs[size]}  "
            f"(wall {time.perf_counter() - started:5.1f}s)",
            flush=True,
        )
        return result


def per_op_ns(result: BenchmarkResult) -> float:
    return result.metadata["per_op_time"] * 1e9


def run_matrix(matrix: Matrix, config: Config) -> None:
    rng = random.Random(SEED)
    for size in config.sizes:
        print(f"\n  --- n = {size:,} ---", flush=True)
        work = Workload(size, config.list_ops[size], rng)
        started = time.perf_counter()
        verify(work)
        print(f"    correctness verified off the clock "
              f"({time.perf_counter() - started:.1f}s)", flush=True)
        heap_group(matrix, work)
        tree_group(matrix, work)
        hash_group(matrix, work)
        del work
        gc.collect()


def heap_group(matrix: Matrix, work: Workload) -> None:
    keys, descending, n = work.keys, work.descending, work.n

    def mine_insert(heap: MinHeap, keys: List[int] = keys) -> None:
        insert = heap.insert
        for key in keys:
            insert(key)

    def mine_insert_desc(heap: MinHeap, keys: List[int] = descending) -> None:
        insert = heap.insert
        for key in keys:
            insert(key)

    def mine_extract(heap: MinHeap, n: int = n) -> None:
        extract = heap.extract_min
        for _ in range(n):
            extract()

    def mine_heapify(_state: Any, keys: List[int] = keys) -> None:
        MinHeap(keys)

    def q_insert(heap: List[int], keys: List[int] = keys) -> None:
        push = heapq.heappush
        for key in keys:
            push(heap, key)

    def q_insert_desc(heap: List[int], keys: List[int] = descending) -> None:
        push = heapq.heappush
        for key in keys:
            push(heap, key)

    def q_extract(heap: List[int], n: int = n) -> None:
        pop = heapq.heappop
        for _ in range(n):
            pop(heap)

    def q_heapify(_state: Any, keys: List[int] = keys) -> None:
        heapq.heapify(list(keys))

    m = matrix.measure
    m("Heap (mine)", "insert", "O(log n) worst", "O(1)", n, MinHeap, mine_insert, n)
    m("Heap (mine)", "insert (descending)", "O(log n)", "O(log n)", n, MinHeap,
      mine_insert_desc, n)
    m("Heap (mine)", "extract_min", "O(log n)", "O(log n)", n,
      lambda: MinHeap(keys), mine_extract, n)
    m("Heap (mine)", "heapify", "O(n) total", "O(1)", n, None, mine_heapify, n)
    m("heapq", "insert", "O(log n) worst", "O(1)", n, list, q_insert, n)
    m("heapq", "insert (descending)", "O(log n)", "O(log n)", n, list, q_insert_desc, n)
    m("heapq", "extract_min", "O(log n)", "O(log n)", n,
      lambda: build_heapq(keys), q_extract, n)
    m("heapq", "heapify", "O(n) total", "O(1)", n, None, q_heapify, n)


def tree_group(matrix: Matrix, work: Workload) -> None:
    keys, lookups, deletes, n = work.keys, work.lookups, work.deletes, work.n
    probe = work.list_probe
    m = matrix.measure

    def avl_insert(tree: AVLTree, keys: List[int] = keys) -> None:
        insert = tree.insert
        for key in keys:
            insert(key)

    def avl_search(tree: AVLTree, lookups: List[int] = lookups) -> None:
        search = tree.search
        for key in lookups:
            search(key)

    def avl_delete(tree: AVLTree, deletes: List[int] = deletes) -> None:
        delete = tree.delete
        for key in deletes:
            delete(key)

    m("AVL tree", "insert", "O(log n)", "O(log n)", n, AVLTree, avl_insert, n)
    built = build_avl(keys)
    m("AVL tree", "search", "O(log n)", "O(log n)", n, lambda: built, avl_search, n)
    del built
    m("AVL tree", "delete", "O(log n)", "O(log n)", n,
      lambda: build_avl(keys), avl_delete, n)

    def dict_insert(table: dict, keys: List[int] = keys) -> None:
        for key in keys:
            table[key] = None

    def dict_search(table: dict, lookups: List[int] = lookups) -> None:
        get = table.get
        for key in lookups:
            get(key)

    def dict_delete(table: dict, deletes: List[int] = deletes) -> None:
        for key in deletes:
            del table[key]

    m("dict", "insert", "O(1) amortised", "O(1)", n, dict, dict_insert, n)
    lookup_table = dict.fromkeys(keys)
    m("dict", "search", "O(1) average", "O(1)", n, lambda: lookup_table, dict_search, n)
    del lookup_table
    m("dict", "delete", "O(1) average", "O(1)", n,
      lambda: dict.fromkeys(keys), dict_delete, n)

    def list_append(values: list, keys: List[int] = keys) -> None:
        append = values.append
        for key in keys:
            append(key)

    def list_search(values: list, probe: List[int] = probe) -> None:
        found = 0
        for key in probe:
            if key in values:
                found += 1
        assert found == len(probe)

    def list_delete(values: list, probe: List[int] = probe) -> None:
        remove = values.remove
        for key in probe:
            remove(key)

    m("list", "insert (append)", "O(1) amortised", "O(1)", n, list, list_append, n)
    haystack = list(keys)
    m("list", "search", "O(n)", "O(n)", n, lambda: haystack, list_search, len(probe))
    del haystack
    m("list", "delete", "O(n)", "O(n)", n, lambda: list(keys), list_delete, len(probe))


def hash_group(matrix: Matrix, work: Workload) -> None:
    keys, lookups, deletes, n = work.keys, work.lookups, work.deletes, work.n
    m = matrix.measure
    for strategy, label in ((CHAINING, "Hash (chaining)"),
                            (LINEAR_PROBING, "Hash (linear probing)")):

        def table_insert(table: HashTable, keys: List[int] = keys) -> None:
            insert = table.insert
            for key in keys:
                insert(key)

        def table_search(table: HashTable, lookups: List[int] = lookups) -> None:
            get = table.get
            for key in lookups:
                get(key)

        def table_delete(table: HashTable, deletes: List[int] = deletes) -> None:
            delete = table.delete
            for key in deletes:
                delete(key)

        m(label, "insert", "O(1) amortised", "O(1)", n,
          lambda s=strategy: HashTable(strategy=s), table_insert, n)
        built = build_table(keys, strategy)
        m(label, "search", "O(1) average", "O(1)", n,
          lambda b=built: b, table_search, n)
        del built
        m(label, "delete", "O(1) average", "O(1)", n,
          lambda s=strategy: build_table(keys, s), table_delete, n)


# ----------------------------------------------------------------------
# Studies 2-5
# ----------------------------------------------------------------------
def knuth(strategy: str, alpha: float) -> Tuple[float, float]:
    """Expected probes for a (successful, unsuccessful) lookup at load ``alpha``.

    Linear probing: Knuth, TAOCP vol. 3, section 6.4. Chaining: 1 + a/2
    comparisons to find a present key, and the chain length a for a miss.
    """
    if strategy == LINEAR_PROBING:
        return 0.5 * (1 + 1 / (1 - alpha)), 0.5 * (1 + 1 / (1 - alpha) ** 2)
    return 1 + alpha / 2, alpha


def load_factor_study(config: Config) -> List[dict]:
    """Probes and time per lookup at fixed load factors, no resizing."""
    print("\n=== STUDY 2 - load factor vs lookup cost ===", flush=True)
    bench = AlgorithmBenchmark(warmup_runs=1, precision=9, seed=SEED)
    bench.verbose = False
    capacity = config.load_capacity
    rows = []
    for alpha in config.load_factors:
        count = int(alpha * capacity)
        for strategy in (CHAINING, LINEAR_PROBING):
            hit_probes, miss_probes, hit_ns, miss_ns = [], [], [], []
            for seed in range(config.load_seeds):
                rng = random.Random(SEED * 7919 + seed * 101 + int(alpha * 1000))
                present = rng.sample(range(KEY_SPACE), count)
                absent = rng.sample(range(KEY_SPACE, 2 * KEY_SPACE), config.load_lookups)
                # Uniform over *every* present key. Sampling the earliest
                # inserts instead biases towards keys that went in while the
                # table was nearly empty and still sit in their home slot.
                sample = rng.sample(present, min(config.load_lookups, count))
                table = HashTable(capacity=capacity, strategy=strategy,
                                  max_load_factor=0.97)
                insert = table.insert
                for key in present:
                    insert(key)
                assert table.capacity == capacity, "table resized during load study"

                hit_probes.append(statistics.fmean(table.probe_count(k) for k in sample))
                miss_probes.append(statistics.fmean(table.probe_count(k) for k in absent))
                for keys, bucket in ((sample, hit_ns), (absent, miss_ns)):
                    result = bench.time_operation(
                        lambda t, keys=keys: _get_all(t, keys),
                        setup=lambda t=table: t,
                        runs=5,
                        ops_per_run=len(keys),
                        name=f"load {strategy} {alpha}",
                    )
                    bucket.append(per_op_ns(result))

            hit_theory, miss_theory = knuth(strategy, alpha)
            rows.append({
                "load_factor": alpha,
                "strategy": strategy,
                "entries": count,
                "capacity": capacity,
                "tables_averaged": config.load_seeds,
                "hit_probes": round(statistics.fmean(hit_probes), 4),
                "hit_probes_theory": round(hit_theory, 4),
                "miss_probes": round(statistics.fmean(miss_probes), 4),
                "miss_probes_theory": round(miss_theory, 4),
                "hit_ns": round(statistics.fmean(hit_ns), 2),
                "miss_ns": round(statistics.fmean(miss_ns), 2),
            })
            row = rows[-1]
            print(f"    a={alpha:<5} {strategy:<15} hit {row['hit_probes']:6.2f} probes "
                  f"(theory {hit_theory:6.2f}) {row['hit_ns']:7.1f} ns | miss "
                  f"{row['miss_probes']:6.2f} (theory {miss_theory:6.2f}) "
                  f"{row['miss_ns']:7.1f} ns", flush=True)
    return rows


def worst_case_study(config: Config) -> List[dict]:
    """Keys that collide at every capacity, against random keys, plus dict."""
    print("\n=== STUDY 3 - engineered worst case ===", flush=True)
    bench = AlgorithmBenchmark(warmup_runs=1, precision=9, seed=SEED)
    bench.verbose = False
    rows = []
    for size in config.worst_sizes:
        rng = random.Random(SEED + size)
        keysets = {
            "colliding": [k * 2 ** 40 for k in range(size)],
            "random": rng.sample(range(KEY_SPACE), size),
        }
        for keyset_name, keyset in keysets.items():
            order = keyset[:]
            rng.shuffle(order)
            for label, strategy in (("Hash (chaining)", CHAINING),
                                    ("Hash (linear probing)", LINEAR_PROBING),
                                    ("dict", None)):
                if strategy is None:
                    built: Any = dict.fromkeys(keyset)
                    probes: Any = ""
                else:
                    built = build_table(keyset, strategy)
                    probes = round(statistics.fmean(built.probe_count(k) for k in keyset), 2)
                result = bench.time_operation(
                    lambda t, order=order: _get_all(t, order),
                    setup=lambda b=built: b,
                    runs=3,
                    ops_per_run=size,
                    name=f"worst {label} {keyset_name}",
                )
                rows.append({
                    "n": size,
                    "keys": keyset_name,
                    "structure": label,
                    "ns_per_lookup": round(per_op_ns(result), 2),
                    "mean_probes": probes,
                    "runs": 3,
                })
                print(f"    n={size:<5} {keyset_name:<9} {label:<22} "
                      f"{per_op_ns(result):>10,.1f} ns/lookup  probes {probes}", flush=True)
    return rows


def amortized_study(config: Config) -> Tuple[List[dict], Dict[str, dict]]:
    """Grow each table one insert at a time; record every insert's cost."""
    print("\n=== STUDY 4 - amortised rehashing ===", flush=True)
    rows: List[dict] = []
    traces: Dict[str, dict] = {}
    size = config.amortized_n
    for strategy in (CHAINING, LINEAR_PROBING):
        keys = random.Random(SEED + 17).sample(range(KEY_SPACE), size)
        table = HashTable(strategy=strategy)
        seconds = [0.0] * size
        moved = [0] * size
        clock = time.perf_counter
        insert = table.insert
        gc.collect()
        gc.disable()
        try:
            for index, key in enumerate(keys):
                before = table.rehash_moves
                start = clock()
                insert(key)
                seconds[index] = clock() - start
                moved[index] = table.rehash_moves - before
        finally:
            gc.enable()

        ordinary = sorted(s for s, m in zip(seconds, moved) if m == 0)
        rebuilds = [(i + 1, s, m) for i, (s, m) in enumerate(zip(seconds, moved)) if m]
        total = sum(seconds)
        summary = {
            "strategy": strategy,
            "inserts": size,
            "rebuilds": len(rebuilds),
            "entries_moved": sum(moved),
            "moves_per_insert": round(sum(moved) / size, 4),
            "total_seconds": round(total, 6),
            "mean_ns_per_insert": round(total / size * 1e9, 2),
            "median_ordinary_ns": round(ordinary[len(ordinary) // 2] * 1e9, 2),
            "p99_ordinary_ns": round(ordinary[int(len(ordinary) * 0.99)] * 1e9, 2),
            "slowest_rebuild_ns": round(max(s for _, s, _ in rebuilds) * 1e9, 2),
            "rebuild_share_of_time": round(sum(s for _, s, _ in rebuilds) / total, 4),
        }
        traces[strategy] = {"seconds": seconds, "moved": moved, "summary": summary}
        for insert_number, spent, entries in rebuilds:
            rows.append({
                "strategy": strategy,
                "row": "rebuild",
                "insert_number": insert_number,
                "ns": round(spent * 1e9, 1),
                "entries_moved": entries,
            })
        rows.append({"strategy": strategy, "row": "summary", **summary})
        print(f"    {strategy:<15} {len(rebuilds)} rebuilds, "
              f"{summary['moves_per_insert']} moves/insert, median insert "
              f"{summary['median_ordinary_ns']:.0f} ns, slowest rebuild "
              f"{summary['slowest_rebuild_ns'] / 1e6:.2f} ms, mean "
              f"{summary['mean_ns_per_insert']:.0f} ns/insert", flush=True)
    return rows, traces


def avl_balance_study(sizes: List[int]) -> List[dict]:
    """Measured AVL heights and rotation rates against the theoretical bounds."""
    print("\n=== STUDY 5 - AVL balance ===", flush=True)
    rows = []
    for size in sizes:
        keys = random.Random(SEED + 99 + size).sample(range(KEY_SPACE), size)
        tree = build_avl(keys)
        random_height = tree.height()
        rotations_per_insert = tree.rotation_count / size
        rebalances_before = tree.rebalance_count
        worst_delete = 0
        half = keys[: size // 2]
        for key in half:
            before = tree.rebalance_count
            tree.delete(key)
            worst_delete = max(worst_delete, tree.rebalance_count - before)
        rebalances_per_delete = (tree.rebalance_count - rebalances_before) / len(half)
        del tree
        ordered = build_avl(range(size))
        sorted_height = ordered.height()
        del ordered
        rows.append({
            "n": size,
            "height_random": random_height,
            "height_sorted": sorted_height,
            "perfect_height": perfect_height(size),
            "avl_max_height": avl_max_height(size),
            "rotations_per_insert": round(rotations_per_insert, 4),
            "rebalances_per_delete": round(rebalances_per_delete, 4),
            "most_rebalances_one_delete": worst_delete,
        })
        print(f"    n={size:<9,} height random {random_height:>2}, sorted {sorted_height:>2} "
              f"(perfect {perfect_height(size)}, AVL max {avl_max_height(size)})  "
              f"rotations/insert {rotations_per_insert:.3f}  worst delete "
              f"{worst_delete} rebalances", flush=True)
        gc.collect()
    return rows


# ----------------------------------------------------------------------
# The comparison table: asymptotic against empirical
# ----------------------------------------------------------------------
def predicted_growth(klass: str, low: int, high: int) -> float:
    """Per-operation growth a class predicts between sizes ``low`` and ``high``."""
    if klass == "O(1)":
        return 1.0
    if klass == "O(log n)":
        return math.log2(high) / math.log2(low)
    return high / low


def classify(growth: float, low: int, high: int) -> str:
    """Nearest class to a measured growth ratio, measured in log space."""
    return min(
        CLASSES,
        key=lambda k: abs(math.log(growth) - math.log(predicted_growth(k, low, high))),
    )


def comparison_rows(matrix: Matrix, sizes: List[int]) -> List[dict]:
    rows = []
    for (structure, operation), by_size in matrix.results.items():
        asymptotic, expected = matrix.meta[(structure, operation)]
        measured = [n for n in sizes if n in by_size]
        ns = np.array([per_op_ns(by_size[n]) for n in measured])
        slope = float(np.polyfit(np.log(measured), np.log(ns), 1)[0])
        low, high = measured[0], measured[-1]
        growth = float(ns[-1] / ns[0])
        empirical = classify(growth, low, high)
        row: Dict[str, Any] = {
            "structure": structure,
            "operation": operation,
            "asymptotic": asymptotic,
            "expected_per_op": expected,
        }
        for n in sizes:
            row[f"ns_per_op_n{n}"] = round(per_op_ns(by_size[n]), 2) if n in by_size else ""
        for n in sizes:
            row[f"runs_n{n}"] = by_size[n].metadata["runs"] if n in by_size else ""
        row.update({
            "loglog_slope": round(slope, 3),
            "growth_measured": round(growth, 3),
            "growth_predicted": round(predicted_growth(expected, low, high), 3),
            "empirical_class": empirical,
            "agrees": empirical == expected,
        })
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(path: str, rows: List[dict]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(rows)} rows)")


COLORS = plt.get_cmap("tab10").colors
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def series(matrix: Matrix, key: SeriesKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_size = matrix.results.get(key, {})
    sizes = sorted(by_size)
    x = np.array(sizes, dtype=float)
    y = np.array([per_op_ns(by_size[n]) for n in sizes])
    err = np.array([
        by_size[n].std_deviation / by_size[n].metadata["ops_per_run"] * 1e9 for n in sizes
    ])
    return x, y, err


def draw(ax, matrix: Matrix, entries: List[Tuple[SeriesKey, str]]) -> None:
    for index, (key, label) in enumerate(entries):
        x, y, err = series(matrix, key)
        if x.size:
            ax.errorbar(x, y, yerr=err, label=label, marker=MARKERS[index % 8],
                        color=COLORS[index % 10], linewidth=1.8, markersize=6,
                        capsize=3, elinewidth=1.0)


def finish(ax, title: str, xlabel: str = "Structure size n (elements)",
           ylabel: str = "Time per operation (ns)", loglog: bool = True) -> None:
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=8)


def reference(ax, x0: float, y0: float, xs: List[float], kind: str) -> None:
    xs_arr = np.array(xs, dtype=float)
    if kind == "O(1)":
        ys = np.full_like(xs_arr, y0)
    elif kind == "O(log n)":
        ys = y0 * np.log2(xs_arr) / math.log2(x0)
    else:
        ys = y0 * xs_arr / x0
    style = {"O(1)": ":", "O(log n)": "--", "O(n)": "-."}[kind]
    ax.plot(xs_arr, ys, style, color="0.45", linewidth=1.2, label=f"{kind} reference")


def save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} "
          f"({os.path.getsize(path) / 1024:.0f} KiB)")


def plot_heap(matrix: Matrix, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    (a, b), (c, d) = axes
    draw(a, matrix, [
        (("Heap (mine)", "insert"), "mine - random input"),
        (("Heap (mine)", "insert (descending)"), "mine - descending (worst case)"),
        (("heapq", "insert"), "heapq - random input"),
        (("heapq", "insert (descending)"), "heapq - descending (worst case)"),
    ])
    finish(a, "Insert: O(1) expected on random input, O(log n) worst case")
    draw(b, matrix, [
        (("Heap (mine)", "extract_min"), "mine"),
        (("heapq", "extract_min"), "heapq"),
    ])
    finish(b, "Extract-min: Theta(log n)")
    draw(c, matrix, [
        (("Heap (mine)", "heapify"), "mine"),
        (("heapq", "heapify"), "heapq"),
    ])
    finish(c, "Heapify, per element: flat means O(n) in total",
           ylabel="Time per element (ns)")

    for index, (operation, label) in enumerate([
        ("insert", "insert - random"),
        ("insert (descending)", "insert - descending"),
        ("extract_min", "extract-min"),
        ("heapify", "heapify"),
    ]):
        xm, ym, _ = series(matrix, ("Heap (mine)", operation))
        xq, yq, _ = series(matrix, ("heapq", operation))
        if xm.size and xq.size:
            d.plot(xm, ym / yq, marker=MARKERS[index], color=COLORS[index],
                   linewidth=1.8, markersize=6, label=label)
    d.set_xscale("log")
    d.set_ylim(bottom=0)
    finish(d, "Same algorithm, same complexity: the constant factor",
           ylabel="Mine / heapq (times slower)", loglog=False)
    d.set_xscale("log")
    fig.suptitle("Binary heap: a pure-Python implementation against heapq (C)",
                 fontsize=13)
    save(fig, path)


def plot_tree(matrix: Matrix, balance: List[dict], path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    (a, b), (c, d) = axes
    draw(a, matrix, [
        (("list", "search"), "list - linear scan"),
        (("AVL tree", "search"), "AVL tree"),
        (("dict", "search"), "dict"),
    ])
    for key, kind in ((("list", "search"), "O(n)"),
                      (("AVL tree", "search"), "O(log n)"),
                      (("dict", "search"), "O(1)")):
        x, y, _ = series(matrix, key)
        if x.size:
            reference(a, x[0], y[0], list(x), kind)
    finish(a, "Search: linear, logarithmic, constant")

    draw(b, matrix, [
        (("AVL tree", "insert"), "AVL insert"),
        (("AVL tree", "delete"), "AVL delete"),
        (("dict", "insert"), "dict insert"),
        (("dict", "delete"), "dict delete"),
        (("list", "delete"), "list delete (remove)"),
    ])
    finish(b, "Insert and delete")

    if balance:
        ns = np.array([row["n"] for row in balance], dtype=float)
        smooth = np.unique(np.round(np.logspace(math.log10(ns.min()),
                                                math.log10(ns.max()), 60)).astype(int))
        c.plot(smooth, [avl_max_height(int(n)) for n in smooth], "-", color="0.35",
               label="AVL maximum (~1.44 log2 n)")
        c.plot(smooth, [perfect_height(int(n)) for n in smooth], "--", color="0.6",
               label="perfectly balanced (log2 n)")
        c.plot(ns, [row["height_random"] for row in balance], "o", color=COLORS[0],
               markersize=8, label="measured, random insertion")
        c.plot(ns, [row["height_sorted"] for row in balance], "s", color=COLORS[1],
               markersize=6, label="measured, sorted insertion")
        c.set_xscale("log")
    finish(c, "AVL height stays between the bounds", ylabel="Tree height (levels)",
           loglog=False)
    c.set_xscale("log")

    for index, operation in enumerate(("search", "insert", "delete")):
        xa, ya, _ = series(matrix, ("AVL tree", operation))
        xd, yd, _ = series(matrix, ("dict", operation))
        if xa.size and xd.size:
            d.plot(xa, ya / yd, marker=MARKERS[index], color=COLORS[index],
                   linewidth=1.8, markersize=6, label=f"AVL / dict - {operation}")
    xa, _, _ = series(matrix, ("AVL tree", "search"))
    if xa.size:
        d.plot(xa, np.log2(xa), ":", color="0.4", linewidth=1.4,
               label="log2(n), for scale")
    d.set_xscale("log")
    finish(d, "How far apart: AVL tree against dict", ylabel="Ratio (times slower)",
           loglog=False)
    d.set_xscale("log")
    fig.suptitle("AVL tree against Python's dict and list", fontsize=13)
    save(fig, path)


def plot_hash(matrix: Matrix, load: List[dict], worst: List[dict], path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    (a, b), (c, d) = axes
    draw(a, matrix, [
        (("Hash (chaining)", "insert"), "chaining - insert"),
        (("Hash (chaining)", "search"), "chaining - search"),
        (("Hash (chaining)", "delete"), "chaining - delete"),
        (("Hash (linear probing)", "insert"), "probing - insert"),
        (("Hash (linear probing)", "search"), "probing - search"),
        (("Hash (linear probing)", "delete"), "probing - delete"),
        (("dict", "search"), "dict - search"),
    ])
    finish(a, "Per operation, at each strategy's own resize threshold")

    alphas = np.linspace(0.05, 0.95, 120)
    for index, (strategy, name) in enumerate(((LINEAR_PROBING, "probing"),
                                              (CHAINING, "chaining"))):
        rows = [r for r in load if r["strategy"] == strategy]
        color_hit, color_miss = COLORS[2 * index], COLORS[2 * index + 1]
        b.plot(alphas, [knuth(strategy, x)[0] for x in alphas], "-", color=color_hit,
               linewidth=1.2, alpha=0.8, label=f"{name} hit - theory")
        b.plot(alphas, [knuth(strategy, x)[1] for x in alphas], "--", color=color_miss,
               linewidth=1.2, alpha=0.8, label=f"{name} miss - theory")
        b.plot([r["load_factor"] for r in rows], [r["hit_probes"] for r in rows], "o",
               color=color_hit, label=f"{name} hit - measured")
        b.plot([r["load_factor"] for r in rows], [max(r["miss_probes"], 0.05) for r in rows],
               "s", color=color_miss, label=f"{name} miss - measured")
        c.plot([r["load_factor"] for r in rows], [r["hit_ns"] for r in rows], "o-",
               color=color_hit, label=f"{name} - hit")
        c.plot([r["load_factor"] for r in rows], [r["miss_ns"] for r in rows], "s--",
               color=color_miss, label=f"{name} - miss")
    for axis in (b, c):
        axis.axvline(0.5, color="0.3", linestyle=":", linewidth=1)
        axis.axvline(0.75, color="0.3", linestyle=":", linewidth=1)
    b.set_yscale("log")
    finish(b, "Probes per lookup vs load factor (Knuth's formulas as lines)",
           xlabel="Load factor (entries / capacity)",
           ylabel="Entries examined per lookup", loglog=False)
    b.set_yscale("log")
    b.text(0.505, b.get_ylim()[1] * 0.6, "probing\nresizes", fontsize=8, color="0.3")
    b.text(0.755, b.get_ylim()[1] * 0.6, "chaining\nresizes", fontsize=8, color="0.3")
    finish(c, "Time per lookup vs load factor", xlabel="Load factor (entries / capacity)",
           ylabel="Time per lookup (ns)", loglog=False)

    for index, (structure, keys, label) in enumerate([
        ("Hash (chaining)", "colliding", "chaining - colliding keys"),
        ("Hash (linear probing)", "colliding", "probing - colliding keys"),
        ("dict", "colliding", "dict - same colliding keys"),
        ("Hash (chaining)", "random", "chaining - random keys"),
        ("Hash (linear probing)", "random", "probing - random keys"),
    ]):
        rows = sorted((r for r in worst if r["structure"] == structure and r["keys"] == keys),
                      key=lambda r: r["n"])
        if rows:
            d.plot([r["n"] for r in rows], [r["ns_per_lookup"] for r in rows],
                   marker=MARKERS[index], color=COLORS[index], linewidth=1.8,
                   markersize=6, linestyle="-" if keys == "colliding" else "--",
                   label=label)
    colliding = sorted((r for r in worst if r["structure"] == "Hash (chaining)"
                        and r["keys"] == "colliding"), key=lambda r: r["n"])
    if colliding:
        reference(d, colliding[0]["n"], colliding[0]["ns_per_lookup"],
                  [r["n"] for r in colliding], "O(n)")
    finish(d, "Worst case: every key in one slot", xlabel="Keys in the table (n)",
           ylabel="Time per lookup (ns)")
    fig.suptitle("Hash tables: separate chaining against linear probing", fontsize=13)
    save(fig, path)


def plot_amortized(traces: Dict[str, dict], path: str) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))
    trace = traces[CHAINING]
    seconds = np.array(trace["seconds"]) * 1e6
    moved = np.array(trace["moved"])
    index = np.arange(1, seconds.size + 1)
    ordinary = moved == 0
    a.scatter(index[ordinary], seconds[ordinary], s=2, color=COLORS[0], alpha=0.25,
              label="ordinary insert")
    a.scatter(index[~ordinary], seconds[~ordinary], s=40, color=COLORS[3], zorder=5,
              label="insert that triggered a rebuild")
    a.plot(index, np.cumsum(seconds) / index, color="black", linewidth=1.6,
           label="running mean")
    a.set_yscale("log")
    finish(a, "Chaining: the cost of every single insert", xlabel="Insert number",
           ylabel="Time for that insert (microseconds)", loglog=False)
    a.set_yscale("log")

    for offset, (strategy, name) in enumerate(((CHAINING, "chaining (resizes at 0.75)"),
                                               (LINEAR_PROBING, "probing (resizes at 0.5)"))):
        moved = np.array(traces[strategy]["moved"])
        counts = np.arange(1, moved.size + 1)
        b.plot(counts, np.cumsum(moved) / counts, color=COLORS[offset], linewidth=1.6,
               label=name)
    b.axhline(2, color="0.3", linestyle="--", linewidth=1, label="bound: 2 moves per insert")
    b.axhline(1, color="0.6", linestyle=":", linewidth=1)
    b.set_xscale("log")
    finish(b, "Entries moved by rebuilds, per insert so far", xlabel="Inserts so far (n)",
           ylabel="Rebuild moves / n", loglog=False)
    b.set_xscale("log")
    b.set_ylim(0, 2.4)
    fig.suptitle("Amortised cost of rehashing", fontsize=13)
    save(fig, path)


def plot_overview(matrix: Matrix, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.6))
    entries = [
        (("list", "search"), "list search - linear"),
        (("AVL tree", "search"), "AVL search - logarithmic"),
        (("Heap (mine)", "extract_min"), "heap extract-min - logarithmic"),
        (("Hash (chaining)", "search"), "hash (chaining) get - constant"),
        (("Hash (linear probing)", "search"), "hash (probing) get - constant"),
        (("dict", "search"), "dict get - constant"),
    ]
    draw(ax, matrix, entries)
    for key, kind in ((("list", "search"), "O(n)"), (("AVL tree", "search"), "O(log n)"),
                      (("dict", "search"), "O(1)")):
        x, y, _ = series(matrix, key)
        if x.size:
            reference(ax, x[0], y[0], list(x), kind)
    finish(ax, "Logarithmic, constant and linear - measured\n"
               "(log-log axes; error bars are 1 SD over repeated runs)")
    save(fig, path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small, fast smoke run")
    parser.add_argument("--out", default=RESULTS_DIR,
                        help="directory for CSVs and PNGs (default: benchmarks/results)")
    args = parser.parse_args(argv)
    config = QUICK if args.quick else FULL
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    started = time.perf_counter()
    print("=" * 78)
    print("CSC 5300 Week 3 - data structures benchmark - Robert Deibel")
    print("=" * 78)
    print(f"  python   : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()} / {os.cpu_count()} logical CPUs")
    print(f"  sizes    : {config.sizes}")
    print(f"  runs     : {config.runs}   warm-ups: {config.warmups}")
    print(f"  mode     : {'QUICK SMOKE RUN' if args.quick else 'full study'}")
    print(f"  output   : {out}")

    bench = AlgorithmBenchmark(warmup_runs=2, precision=9, seed=SEED)
    bench.verbose = False
    matrix = Matrix(bench, config)
    print("\n=== STUDY 1 - main matrix ===", flush=True)
    run_matrix(matrix, config)

    load_rows = load_factor_study(config)
    worst_rows = worst_case_study(config)
    amortized_rows, traces = amortized_study(config)
    balance_rows = avl_balance_study(config.balance_sizes or config.sizes)

    print("\n=== writing results ===")
    bench.export_results(os.path.join(out, "week3_raw_results.csv"))
    write_csv(os.path.join(out, "comparison_table.csv"), comparison_rows(matrix, config.sizes))
    write_csv(os.path.join(out, "week3_load_factor.csv"), load_rows)
    write_csv(os.path.join(out, "week3_worst_case.csv"), worst_rows)
    write_csv(os.path.join(out, "week3_amortized.csv"), amortized_rows)
    write_csv(os.path.join(out, "week3_avl_balance.csv"), balance_rows)

    print("\n=== charts ===")
    apply_house_style()
    plot_heap(matrix, os.path.join(out, "heap_performance.png"))
    plot_tree(matrix, balance_rows, os.path.join(out, "tree_performance.png"))
    plot_hash(matrix, load_rows, worst_rows, os.path.join(out, "hash_performance.png"))
    plot_amortized(traces, os.path.join(out, "hash_amortized.png"))
    plot_overview(matrix, os.path.join(out, "structures_overview.png"))

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 78)
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
