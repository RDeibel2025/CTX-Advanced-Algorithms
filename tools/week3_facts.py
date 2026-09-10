#!/usr/bin/env python3
"""Print every figure analysis/week3_report.md quotes, computed from the CSVs.

The Week 3 report is prose under a word limit, so it cannot be generated
wholesale. This script is the compromise used in Week 2: it computes each
number the report cites from ``benchmarks/results/``, so figures are read off
one verified source rather than transcribed from scrollback, and the report
can be re-checked against a fresh run in one command.

    python benchmarks/week3_structures_benchmark.py
    python tools/week3_facts.py

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results")
SIZES = [1000, 10000, 100000, 1000000]


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    table = pd.read_csv(os.path.join(RESULTS, "comparison_table.csv"))
    load = pd.read_csv(os.path.join(RESULTS, "week3_load_factor.csv"))
    worst = pd.read_csv(os.path.join(RESULTS, "week3_worst_case.csv"))
    amortized = pd.read_csv(os.path.join(RESULTS, "week3_amortized.csv"))
    balance = pd.read_csv(os.path.join(RESULTS, "week3_avl_balance.csv"))

    def ns(structure: str, operation: str, n: int) -> float:
        row = table[(table.structure == structure) & (table.operation == operation)]
        return float(row[f"ns_per_op_n{n}"].iloc[0])

    heading("1. Nanoseconds per operation (runs per size in brackets)")
    runs = "/".join(str(int(table[f"runs_n{n}"].iloc[0])) for n in SIZES)
    print(f"  runs at 10^3/10^4/10^5/10^6: {runs}")
    print(f"  {'structure':<22}{'operation':<21}" + "".join(f"{n:>12,}" for n in SIZES))
    for row in table.itertuples():
        cells = "".join(f"{getattr(row, f'ns_per_op_n{n}'):>12,.1f}" for n in SIZES)
        print(f"  {row.structure:<22}{row.operation:<21}{cells}")

    heading("2. Heap: mine / heapq, same algorithm, same complexity")
    for operation in ("insert", "insert (descending)", "extract_min", "heapify"):
        ratios = [ns("Heap (mine)", operation, n) / ns("heapq", operation, n) for n in SIZES]
        print(f"  {operation:<22}" + "".join(f"{r:>9.2f}x" for r in ratios)
              + f"   range {min(ratios):.2f}-{max(ratios):.2f}")
    all_ratios = [ns("Heap (mine)", op, n) / ns("heapq", op, n)
                  for op in ("insert", "insert (descending)", "extract_min", "heapify")
                  for n in SIZES]
    print(f"  overall range: {min(all_ratios):.2f}x to {max(all_ratios):.2f}x")
    for n in SIZES:
        mine_ext = ns("Heap (mine)", "extract_min", n)
        mine_ins = ns("Heap (mine)", "insert", n)
        print(f"  n={n:>9,}: my extract / my insert = {mine_ext / mine_ins:5.2f}x "
              f"(insert {mine_ins:.1f} ns, extract {mine_ext:.1f} ns)")

    heading("3. AVL tree / dict, against log2(n)")
    print(f"  {'operation':<10}" + "".join(f"{n:>12,}" for n in SIZES))
    for operation in ("search", "insert", "delete"):
        ratios = [ns("AVL tree", operation, n) / ns("dict", operation, n) for n in SIZES]
        print(f"  {operation:<10}" + "".join(f"{r:>11.1f}x" for r in ratios))
    print(f"  {'log2(n)':<10}" + "".join(f"{math.log2(n):>12.1f}" for n in SIZES))
    for n in SIZES:
        per_level = ns("AVL tree", "search", n) / math.log2(n)
        print(f"  n={n:>9,}: AVL search {ns('AVL tree', 'search', n):7.1f} ns = "
              f"{per_level:5.1f} ns per log2(n) level; dict get {ns('dict', 'search', n):5.1f} ns")

    heading("4. Hash strategies at their own thresholds, and dict")
    for operation in ("insert", "search", "delete"):
        chain = [ns("Hash (chaining)", operation, n) for n in SIZES]
        probe = [ns("Hash (linear probing)", operation, n) for n in SIZES]
        print(f"  {operation:<8} chaining " + " ".join(f"{v:7.1f}" for v in chain)
              + " | probing " + " ".join(f"{v:7.1f}" for v in probe))
    for n in SIZES:
        mine = ns("Hash (chaining)", "search", n)
        print(f"  n={n:>9,}: chaining get / dict get = {mine / ns('dict', 'search', n):5.2f}x")

    heading("5. Load factor: probes against Knuth, and time")
    for row in load.itertuples():
        print(f"  a={row.load_factor:<5} {row.strategy:<15} hit {row.hit_probes:6.2f} "
              f"(theory {row.hit_probes_theory:6.2f})  miss {row.miss_probes:7.2f} "
              f"(theory {row.miss_probes_theory:7.2f})  hit {row.hit_ns:6.1f} ns  "
              f"miss {row.miss_ns:7.1f} ns")
    for strategy in ("chaining", "linear_probing"):
        rows = load[load.strategy == strategy].set_index("load_factor")
        print(f"  {strategy}: miss time 0.1 -> 0.95 = {rows.miss_ns.iloc[0]:.1f} -> "
              f"{rows.miss_ns.iloc[-1]:.1f} ns ({rows.miss_ns.iloc[-1] / rows.miss_ns.iloc[0]:.1f}x);"
              f" hit {rows.hit_ns.iloc[0]:.1f} -> {rows.hit_ns.iloc[-1]:.1f} ns")
    for alpha in (0.5, 0.7, 0.8):
        at = load[(load.load_factor - alpha).abs() < 1e-9]
        c = at[at.strategy == "chaining"].iloc[0]
        p = at[at.strategy == "linear_probing"].iloc[0]
        print(f"  at a={alpha}: miss probes probing {p.miss_probes:.2f} vs chaining "
              f"{c.miss_probes:.2f}; miss ns probing {p.miss_ns:.1f} vs chaining {c.miss_ns:.1f}")
    rel = []
    for row in load.itertuples():
        for measured, theory in ((row.hit_probes, row.hit_probes_theory),
                                 (row.miss_probes, row.miss_probes_theory)):
            if theory > 0.2:
                rel.append((abs(measured - theory) / theory, row.strategy, row.load_factor))
    rel.sort()
    print(f"  probe counts vs theory: median relative error "
          f"{100 * rel[len(rel) // 2][0]:.1f}%, worst {100 * rel[-1][0]:.1f}% "
          f"({rel[-1][1]} at a={rel[-1][2]})")

    heading("6. Engineered worst case")
    # "keys" is also a DataFrame method, so the column needs subscripting.
    for structure in ("Hash (chaining)", "Hash (linear probing)", "dict"):
        for keys in ("colliding", "random"):
            rows = worst[(worst.structure == structure) & (worst["keys"] == keys)].sort_values("n")
            values = rows.ns_per_lookup.tolist()
            doublings = [b / a for a, b in zip(values, values[1:])]
            print(f"  {structure:<22} {keys:<9} " + " ".join(f"{v:9.1f}" for v in values)
                  + "  doubling " + " ".join(f"{d:.2f}" for d in doublings))
    big = worst[worst.n == worst.n.max()]
    for structure in ("Hash (chaining)", "Hash (linear probing)"):
        col = big[(big.structure == structure) & (big["keys"] == "colliding")].iloc[0]
        ran = big[(big.structure == structure) & (big["keys"] == "random")].iloc[0]
        print(f"  n={int(col.n)}: {structure} colliding / random = "
              f"{col.ns_per_lookup / ran.ns_per_lookup:.0f}x, mean probes {col.mean_probes}")

    heading("7. Amortised rehashing")
    summaries = amortized[amortized.row == "summary"]
    for row in summaries.itertuples():
        print(f"  {row.strategy:<15} {int(row.inserts):,} inserts, {int(row.rebuilds)} rebuilds, "
              f"{row.moves_per_insert} moves/insert, median {row.median_ordinary_ns:.0f} ns, "
              f"p99 {row.p99_ordinary_ns:.0f} ns, slowest rebuild "
              f"{row.slowest_rebuild_ns / 1e6:.2f} ms "
              f"({row.slowest_rebuild_ns / row.median_ordinary_ns:,.0f}x median), mean "
              f"{row.mean_ns_per_insert:.0f} ns/insert, rebuilds = "
              f"{100 * row.rebuild_share_of_time:.1f}% of total time")

    heading("8. AVL balance")
    for row in balance.itertuples():
        print(f"  n={int(row.n):>9,}: height random {row.height_random}, sorted "
              f"{row.height_sorted}, perfect {row.perfect_height}, AVL max "
              f"{row.avl_max_height}; rotations/insert {row.rotations_per_insert}; "
              f"rebalances/delete {row.rebalances_per_delete}; worst delete "
              f"{row.most_rebalances_one_delete}")

    heading("9. Asymptotic against empirical")
    for row in table.itertuples():
        flag = "" if row.agrees else "   <-- disagrees"
        print(f"  {row.structure:<22}{row.operation:<21}{row.asymptotic:<16}"
              f"expected {row.expected_per_op:<9} slope {row.loglog_slope:>6.3f} "
              f"growth {row.growth_measured:>8.2f} (pred {row.growth_predicted:>7.2f}) "
              f"-> {row.empirical_class}{flag}")
    agree = int(table.agrees.sum())
    print(f"\n  {agree} of {len(table)} series agree with their expected class")

    # Counts are hardware-independent. Rebuilding the exact structures the
    # benchmark timed (same seed, same key order) and counting what a
    # successful lookup examines separates the algorithm from the machine:
    # where time rises and the count does not, the rise is the memory system.
    # Imported here because only this section needs the benchmark's
    # workloads, and the benchmark module pulls in matplotlib.
    import random
    import statistics

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from benchmarks.week3_structures_benchmark import (
        FULL,
        SEED,
        Workload,
        build_avl,
        build_table,
    )
    from src.structures.hash_table import CHAINING, LINEAR_PROBING

    heading("10. Operation counts at each size: what a successful lookup examines")
    rng = random.Random(SEED)              # the same sequence run_matrix draws
    for n in SIZES:
        work = Workload(n, FULL.list_ops[n], rng)
        sample = work.lookups[:20_000]
        for strategy, label in ((CHAINING, "Hash (chaining)"),
                                (LINEAR_PROBING, "Hash (linear probing)")):
            built = build_table(work.keys, strategy)
            probes = statistics.fmean(built.probe_count(k) for k in sample)
            took = ns(label, "search", n)
            print(f"  n={n:>9,}  {label:<22} capacity {built.capacity:>9,}  load "
                  f"{built.load_factor:.3f}  probes/hit {probes:.3f}  {took:6.1f} ns "
                  f"= {took / probes:6.1f} ns per probe")
            del built
        tree = build_avl(work.keys)
        total = count = 0
        stack = [(tree.root, 1)]
        while stack:
            node, depth = stack.pop()
            if node is not None:
                total += depth
                count += 1
                stack.append((node.left, depth + 1))
                stack.append((node.right, depth + 1))
        visited = total / count
        took = ns("AVL tree", "search", n)
        print(f"  n={n:>9,}  AVL tree: {visited:.2f} nodes per successful search, "
              f"{took:7.1f} ns = {took / visited:5.1f} ns per node "
              f"(dict get {ns('dict', 'search', n):5.1f} ns)")
        del tree, work

    heading("11. The step at 10^6: cost above what the 10^5 measurement predicts")
    step = math.log2(10 ** 6) / math.log2(10 ** 5)
    for structure, operation, logarithmic in (
        ("Heap (mine)", "extract_min", True),
        ("heapq", "extract_min", True),
        ("AVL tree", "search", True),
        ("dict", "search", False),
        ("Hash (chaining)", "search", False),
        ("Hash (linear probing)", "search", False),
    ):
        at5, at6 = ns(structure, operation, 10 ** 5), ns(structure, operation, 10 ** 6)
        predicted = at5 * (step if logarithmic else 1.0)
        print(f"  {structure:<22}{operation:<12} 10^5 {at5:7.1f}  predicted "
              f"{predicted:7.1f}  measured {at6:7.1f}  excess {at6 - predicted:6.1f} ns "
              f"({at6 / predicted:.2f}x)")

    heading("12. Other ratios the report quotes")
    top = 10 ** 6
    print(f"  at 10^6: list search / AVL search = "
          f"{ns('list', 'search', top) / ns('AVL tree', 'search', top):,.0f}x, "
          f"list search / dict get = "
          f"{ns('list', 'search', top) / ns('dict', 'search', top):,.0f}x")
    for n in SIZES:
        level = math.log2(n)
        print(f"  n={n:>9,}: AVL/dict per log2(n) level - insert "
              f"{ns('AVL tree', 'insert', n) / ns('dict', 'insert', n) / level:4.1f}, "
              f"delete {ns('AVL tree', 'delete', n) / ns('dict', 'delete', n) / level:4.1f}, "
              f"search {ns('AVL tree', 'search', n) / ns('dict', 'search', n) / level:4.2f}"
              f" | chaining/probing - insert "
              f"{ns('Hash (chaining)', 'insert', n) / ns('Hash (linear probing)', 'insert', n):4.2f}"
              f", search "
              f"{ns('Hash (chaining)', 'search', n) / ns('Hash (linear probing)', 'search', n):4.2f}"
              f", delete "
              f"{ns('Hash (chaining)', 'delete', n) / ns('Hash (linear probing)', 'delete', n):4.2f}")
    at = load[(load.load_factor - 0.95).abs() < 1e-9].set_index("strategy")
    print(f"  at a=0.95: probing miss / chaining miss = "
          f"{at.miss_ns['linear_probing'] / at.miss_ns['chaining']:.1f}x")

    heading("13. Run-to-run spread in the main matrix")
    raw = pd.read_csv(os.path.join(RESULTS, "week3_raw_results.csv"))
    repeated = raw[raw.runs > 1]
    spread = repeated.std_deviation / repeated.average_time
    widest = raw.loc[spread.idxmax()]
    print(f"  {len(repeated)} cells with repeated runs: median relative SD "
          f"{100 * spread.median():.1f}%, 90th percentile "
          f"{100 * spread.quantile(0.9):.1f}%, max {100 * spread.max():.1f}% "
          f"({widest.algorithm_name} at n={int(widest.input_size):,})")
    print(f"  cells with a single run (n=10^6): {int((raw.runs == 1).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
