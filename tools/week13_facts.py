#!/usr/bin/env python3
"""Print every number the Week 13 report quotes, straight from the benchmark CSV.

Run from the repository root after ``benchmarks/week13_benchmark.py``::

    python tools/week13_facts.py

Reads ``benchmarks/results/week13/benchmark.csv``. If a benchmark number in
``analysis/week13_report.md`` is not printed here, it should not be in the
report. Values are printed to 3 significant figures, and every range names
the size at which each end occurs.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmarks", "results", "week13")
README = os.path.join(REPO_ROOT, "README.md")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sig(x: float, digits: int = 3) -> str:
    """Format ``x`` to ``digits`` significant figures, with thousands separators for large values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = max(digits - 1 - magnitude, 0)
    rounded = round(x, decimals)
    if decimals == 0:
        return f"{int(round(rounded)):,}"
    return f"{rounded:,.{decimals}f}"


def pow2(n: float) -> str:
    e = math.log2(n)
    return f"2^{int(e)}" if abs(e - round(e)) < 1e-9 else f"{int(n):,}"


def span(values, ns, fmt=sig, label=pow2) -> str:
    """'lo (at n) to hi (at n)' for paired values and sizes."""
    values = list(values)
    ns = list(ns)
    i_lo = int(np.argmin(values))
    i_hi = int(np.argmax(values))
    return f"{fmt(values[i_lo])} (n = {label(ns[i_lo])}) to {fmt(values[i_hi])} (n = {label(ns[i_hi])})"


def slope(ns, values) -> float:
    return float(np.polyfit(np.log2(np.asarray(ns, dtype=float)), np.log2(np.asarray(values, dtype=float)), 1)[0])


def pick(t: pd.DataFrame, experiment: str, structure: str, method: str) -> pd.DataFrame:
    g = t[(t.experiment == experiment) & (t.structure == structure) & (t.method == method) & (t.capped != True)]  # noqa: E712
    return g.sort_values("n").reset_index(drop=True)


def per_op_us(g: pd.DataFrame) -> np.ndarray:
    return (g.mean_time_s / g.ops * 1e6).to_numpy()


def main() -> int:
    path = os.path.join(RESULTS, "benchmark.csv")
    t = pd.read_csv(path)
    header("0. The run")
    print(f"  rows {len(t)}, capped rows {int((t.capped == True).sum())}, "  # noqa: E712
          f"timed rows have runs = {sorted(set(int(r) for r in t.runs.dropna()))}")

    # ------------------------------------------------------------------
    header("1. Segment tree vs Fenwick tree (segment_vs_fenwick)")
    E = "segment_vs_fenwick"
    sb, fb = pick(t, E, "segment_tree", "build"), pick(t, E, "fenwick_tree", "build")
    su, fu = pick(t, E, "segment_tree", "point_update"), pick(t, E, "fenwick_tree", "point_update")
    ss, fs = pick(t, E, "segment_tree", "range_sum"), pick(t, E, "fenwick_tree", "range_sum")
    ns_ = pick(t, E, "naive_reference", "range_sum")
    for i, n in enumerate(sb.n):
        line = (f"  n = {pow2(n):<5} build seg {sig(sb.mean_time_s[i] * 1e3)} ms fen {sig(fb.mean_time_s[i] * 1e3)} ms | "
                f"update seg {sig(per_op_us(su)[i])} us fen {sig(per_op_us(fu)[i])} us | "
                f"sum seg {sig(per_op_us(ss)[i])} us fen {sig(per_op_us(fs)[i])} us")
        m = ns_[ns_.n == n]
        if len(m):
            line += f" naive {sig(per_op_us(m)[0])} us"
        print(line)
    print(f"  segment/Fenwick, range sum: {span(per_op_us(ss) / per_op_us(fs), ss.n)}")
    print(f"  segment/Fenwick, point update: {span(per_op_us(su) / per_op_us(fu), su.n)}")
    print(f"  segment/Fenwick, build: {span(sb.mean_time_s / fb.mean_time_s, sb.n)}")
    for name, g in (("segment sum", ss), ("Fenwick sum", fs), ("segment update", su), ("Fenwick update", fu)):
        v = per_op_us(g)
        print(f"  {name}: {sig(v[0])} us at {pow2(g.n[0])} -> {sig(v[-1])} us at {pow2(g.n[len(g) - 1])}, "
              f"x{sig(v[-1] / v[0])} for x{int(g.n.iloc[-1] / g.n.iloc[0]):,} in n")
    print(f"  build log-log slope 2^14..2^20: segment {slope(sb.n[4:], sb.mean_time_s[4:]):.2f}, "
          f"Fenwick {slope(fb.n[4:], fb.mean_time_s[4:]):.2f}")
    vn = per_op_us(ns_)
    print(f"  naive sum per query {span(vn, ns_.n)}; log-log slope 2^12..2^16 {slope(ns_.n[2:], vn[2:]):.2f}")
    for i, n in enumerate(ns_.n):
        s = per_op_us(ss)[list(ss.n).index(n)]
        f = per_op_us(fs)[list(fs.n).index(n)]
        print(f"    n = {pow2(n):<5} naive/segment {sig(vn[i] / s)}  naive/Fenwick {sig(vn[i] / f)}")
    print(f"  Fenwick range sum per query at 2^20 = {sig(per_op_us(fs)[-1])} us; segment {sig(per_op_us(ss)[-1])} us")

    # ------------------------------------------------------------------
    header("2. Persistent overhead (persistent_overhead)")
    E = "persistent_overhead"
    pb, pp = pick(t, E, "persistent_segment_tree", "build"), pick(t, E, "persistent_segment_tree", "point_set")
    pq, sq = pick(t, E, "persistent_segment_tree", "query_sum"), pick(t, E, "segment_tree", "query_sum")
    smem, fc = pick(t, E, "segment_tree", "build"), pick(t, E, "full_copy_model", "copy_per_version")
    exact = True
    for i, n in enumerate(pb.n):
        per_update = pp.nodes_created[i] / pp.ops[i]
        exact &= per_update == math.log2(n) + 1 and pb.nodes_created[i] == 2 * n - 1
        stored = pb.nodes_created[i] + pp.nodes_created[i]
        print(f"  n = {pow2(n):<5} build {sig(pb.mean_time_s[i] * 1e3)} ms, {int(pb.nodes_created[i]):,} nodes | "
              f"{per_update:.1f} nodes/point set | 1,001 versions hold {int(stored):,} nodes "
              f"({sig(pp.overhead_ratio[i])} x one version); full copy {int(fc.nodes_created[i]):,} "
              f"(stored/full {sig(stored / fc.nodes_created[i])}) | mem build {sig(pb.memory_kib[i])} KiB, "
              f"after {sig(pp.memory_kib[i])} KiB, plain {sig(smem.memory_kib[i])} KiB | "
              f"query pers {sig(per_op_us(pq)[i])} us plain {sig(per_op_us(sq)[i])} us")
    print(f"  every size: nodes per point set == log2 n + 1 and build == 2n - 1: {exact}")
    print(f"  versions in the full-copy model: {sorted(set(int(v) for v in fc.overhead_ratio))}")
    print(f"  point_set time per update: {span(per_op_us(pp), pp.n)}")
    stored = pb.nodes_created + pp.nodes_created
    print(f"  stored / full copy: {span(stored / fc.nodes_created, pb.n, fmt=lambda x: f'{x:.3g}')}")
    print(f"  all versions / one version (overhead_ratio): {span(pp.overhead_ratio, pp.n)}")
    print(f"  persistent build memory / plain build memory: {span(pb.memory_kib / smem.memory_kib, pb.n)}")
    print(f"  extra memory of 1,000 point sets (after - build): "
          f"{span(pp.memory_kib - pb.memory_kib, pp.n)} KiB")
    print(f"  bytes per persistent node at build: {span(pb.memory_kib * 1024 / pb.nodes_created, pb.n)}")
    print(f"  persistent/plain query time: {span(per_op_us(pq) / per_op_us(sq), pq.n)}")
    print(f"  persistent build log-log slope 2^14..2^18: {slope(pb.n[4:], pb.mean_time_s[4:]):.2f}; "
          f"at 2^18 {sig(pb.mean_time_s.iloc[-1])} s")

    # ------------------------------------------------------------------
    header("3. Rank/select (rank_select)")
    E = "rank_select"
    for d in (0.5, 0.1):
        r1 = pick(t, E, "rank_select", "rank1")
        r1 = r1[r1.density == d].reset_index(drop=True)
        s1 = pick(t, E, "rank_select", "select1")
        s1 = s1[s1.density == d].reset_index(drop=True)
        nv = pick(t, E, "naive_reference", "rank1_bit_count")
        nv = nv[nv.density == d].reset_index(drop=True)
        print(f"  density {d}: rank1 {span(per_op_us(r1), r1.n)} us; x{sig(per_op_us(r1)[-1] / per_op_us(r1)[0])} "
              f"from {pow2(r1.n[0])} to {pow2(r1.n.iloc[-1])} (n x{int(r1.n.iloc[-1] / r1.n[0])})")
        print(f"               select1 {span(per_op_us(s1), s1.n)} us; x{sig(per_op_us(s1)[-1] / per_op_us(s1)[0])}")
        print(f"               select1/rank1 {span(per_op_us(s1) / per_op_us(r1), s1.n)}")
        print(f"               naive {span(per_op_us(nv), nv.n)} us; slope {slope(nv.n, per_op_us(nv)):.2f}; "
              f"naive/rank at {pow2(nv.n.iloc[-1])} = {sig(per_op_us(nv)[-1] / per_op_us(r1)[list(r1.n).index(nv.n.iloc[-1])])}")
        print(f"               rank directory overhead values {sorted(set(r1.overhead_ratio))}; "
              f"select samples (1s and 0s) {span(s1.overhead_ratio * 100, s1.n)} % of n")

    # ------------------------------------------------------------------
    header("4. Wavelet trees (wavelet_dna, wavelet_text)")
    for E in ("wavelet_dna", "wavelet_text"):
        bal, huf = pick(t, E, "wavelet_balanced", "space"), pick(t, E, "wavelet_huffman", "space")
        h0 = pick(t, E, "zero_order_entropy", "h0")
        cb, ch = pick(t, E, "wavelet_balanced", "range_count"), pick(t, E, "wavelet_huffman", "range_count")
        qb = pick(t, E, "wavelet_balanced", "range_quantile")
        fm = pick(t, E, "fm_index", "pattern_count")
        nc, npat = pick(t, E, "naive_reference", "range_count_scan"), pick(t, E, "naive_reference", "pattern_count_scan")
        for i, n in enumerate(bal.n):
            line = (f"  {E} n = {int(n):,}: bits/symbol balanced {bal.bits_per_symbol[i]:.3f} Huffman "
                    f"{huf.bits_per_symbol[i]:.3f} H0 {h0.bits_per_symbol[i]:.3f} | compression balanced "
                    f"{bal.compression_ratio[i]:.3f} Huffman {huf.compression_ratio[i]:.3f} | directory/payload "
                    f"balanced {bal.overhead_ratio[i]:.4f} Huffman {huf.overhead_ratio[i]:.4f}")
            print(line)
            parts = [f"count bal {sig(per_op_us(cb)[i])} us", f"count huf {sig(per_op_us(ch)[i])} us",
                     f"quantile {sig(per_op_us(qb)[i])} us"]
            if len(fm):
                parts.append(f"FM {sig(per_op_us(fm)[i])} us/pattern")
            m = nc[nc.n == n]
            if len(m):
                parts.append(f"naive count {sig(per_op_us(m)[0])} us")
            m = npat[npat.n == n] if len(npat) else npat
            if len(m):
                parts.append(f"naive pattern scan {sig(per_op_us(m)[0])} us")
            print("      " + ", ".join(parts))
        if E == "wavelet_dna":
            print(f"  FM per pattern {span(per_op_us(fm), fm.n, label=lambda n: f'{int(n):,}')}; "
                  f"count balanced {span(per_op_us(cb), cb.n, label=lambda n: f'{int(n):,}')}")
            for i, n in enumerate(nc.n):
                c = per_op_us(cb)[list(cb.n).index(n)]
                f = per_op_us(fm)[list(fm.n).index(n)]
                print(f"    n = {int(n):,}: naive count / wavelet count {sig(per_op_us(nc)[i] / c)}; "
                      f"naive pattern scan / FM {sig(per_op_us(npat)[i] / f)}")
        else:
            n = int(bal.n[0])
            text = open(README, encoding="utf-8").read()
            sigma = len(set(text))
            same = len(text) == n
            print(f"  README.md now has {len(text):,} characters; CSV n = {n:,}; same file as the run: {same}")
            if same:
                print(f"  sigma = {sigma}, ceil(log2 sigma) = {math.ceil(math.log2(sigma))}, "
                      f"log2 sigma = {math.log2(sigma):.3f}")
            print(f"  Huffman - H0 = {huf.bits_per_symbol[0] - h0.bits_per_symbol[0]:.3f} bits; "
                  f"balanced - Huffman = {bal.bits_per_symbol[0] - huf.bits_per_symbol[0]:.3f} bits; "
                  f"Huffman/balanced payload {sig(huf.bits_per_symbol[0] / bal.bits_per_symbol[0])}")
            print(f"  naive count / wavelet balanced count {sig(per_op_us(nc)[0] / per_op_us(cb)[0])}; "
                  f"Huffman/balanced count time {sig(per_op_us(ch)[0] / per_op_us(cb)[0])}")

    # ------------------------------------------------------------------
    header("5. Cache-oblivious matrix operations (cache_misses, matrix_time)")
    E = "cache_misses"
    for op, words in (("transpose", 2), ("matmul", 3)):
        nv, rc = pick(t, E, op, "naive"), pick(t, E, op, "recursive")
        for (M, B), g in nv.groupby(["cache_M", "cache_B"]):
            r = rc[(rc.cache_M == M) & (rc.cache_B == B)].sort_values("n")
            g = g.sort_values("n")
            for (_, a), (_, b) in zip(g.iterrows(), r.iterrows()):
                n = int(a.n)
                compulsory = words * n * n / B
                print(f"  {op:<9} n = {n:<3} (M {int(M):>4}, B {int(B):>2}): naive {int(a.misses):>7,} "
                      f"recursive {int(b.misses):>6,} | naive/recursive {sig(a.misses / b.misses)} | "
                      f"{words}n^2/B = {sig(compulsory)}: recursive/that {sig(b.misses / compulsory)}, "
                      f"naive/that {sig(a.misses / compulsory)} | accesses {int(a.ops):,}")
    E = "matrix_time"
    for op in ("transpose", "matmul"):
        nv, rc, npy = pick(t, E, op, "naive"), pick(t, E, op, "recursive_base16"), pick(t, E, op, "numpy_reference")
        for i, n in enumerate(nv.n):
            print(f"  wall {op:<9} n = {int(n):<3}: naive {sig(nv.mean_time_s[i] * 1e3)} ms, recursive base16 "
                  f"{sig(rc.mean_time_s[i] * 1e3)} ms, numpy {sig(npy.mean_time_s[i] * 1e3)} ms | recursive/naive "
                  f"{sig(rc.mean_time_s[i] / nv.mean_time_s[i])} | naive/numpy {sig(nv.mean_time_s[i] / npy.mean_time_s[i])}"
                  f" | numpy runs {int(npy.runs[i])}")
        print(f"  {op}: recursive/naive wall clock {span(rc.mean_time_s / nv.mean_time_s, nv.n, label=lambda n: str(int(n)))}")

    # ------------------------------------------------------------------
    header("6. Layouts (layout_blocks)")
    E = "layout_blocks"
    bfs, veb = pick(t, E, "bfs_layout", "random_search"), pick(t, E, "veb_layout", "random_search")
    for B in (16, 64):
        b = bfs[bfs.cache_B == B].sort_values("n").reset_index(drop=True)
        v = veb[veb.cache_B == B].sort_values("n").reset_index(drop=True)
        h = [int(round(math.log2(n + 1))) for n in b.n]
        for i in range(len(b)):
            print(f"  B = {B:<2} h = {h[i]:<2} (n = {int(b.n[i]):>9,}): BFS {b.blocks_per_search[i]:.3f} "
                  f"vEB {v.blocks_per_search[i]:.3f} ratio {v.blocks_per_search[i] / b.blocks_per_search[i]:.3f} "
                  f"| BFS - (h - log2 B) = {b.blocks_per_search[i] - (h[i] - math.log2(B)):.3f} "
                  f"| searches {int(b.ops[i]):,}")
        ratio = (v.blocks_per_search / b.blocks_per_search).to_numpy()
        print(f"  B = {B}: vEB/BFS {span(ratio, h, fmt=lambda x: f'{x:.3f}', label=lambda x: f'h {x}')}")
        print(f"  B = {B}: BFS growth per level {(b.blocks_per_search.iloc[-1] - b.blocks_per_search[0]) / (h[-1] - h[0]):.3f}; "
              f"vEB growth per level {(v.blocks_per_search.iloc[-1] - v.blocks_per_search[0]) / (h[-1] - h[0]):.3f}")

    # ------------------------------------------------------------------
    header("7. Version-control history (vcs_history)")
    E = "vcs_history"
    c, tn = pick(t, E, "mini_vcs", "commit"), pick(t, E, "mini_vcs", "tree_nodes")
    d = pick(t, E, "mini_vcs", "diff_vs_parent")
    st, fc = pick(t, E, "mini_vcs", "all_commits_stored"), pick(t, E, "mini_vcs", "all_commits_full_copy")
    print(f"  commits {int(st.n[0])}; stored {int(st.nodes_created[0]):,} nodes; full copy {int(fc.nodes_created[0]):,}; "
          f"stored fraction {st.overhead_ratio[0]:.4f}")
    print(f"  files per commit: {int(tn.nodes_created.min()):,} to {int(tn.nodes_created.max()):,}")
    later = c[c.n > 1]
    print(f"  files changed per later commit: {int(later.ops.min())} to {int(later.ops.max())}; "
          f"new nodes {int(later.nodes_created.min())} to {int(later.nodes_created.max())}; "
          f"new nodes per changed file {sig((later.nodes_created / later.ops).min())} to "
          f"{sig((later.nodes_created / later.ops).max())}, median {sig(float(np.median(later.nodes_created / later.ops)))}")
    print(f"  new nodes of later commits / commit tree size: {sig((later.nodes_created.to_numpy() / tn[tn.n > 1].nodes_created.to_numpy()).min())} "
          f"to {sig((later.nodes_created.to_numpy() / tn[tn.n > 1].nodes_created.to_numpy()).max())}")
    print(f"  diff visited / both trees: {sig(d.overhead_ratio.min())} (commit {int(d.n[d.overhead_ratio.idxmin()])}) "
          f"to {sig(d.overhead_ratio.max())} (commit {int(d.n[d.overhead_ratio.idxmax()])}); "
          f"median {sig(float(d.overhead_ratio.median()))}")
    sizes = dict(zip(tn.n.astype(int), tn.nodes_created.astype(int)))
    for label, i in (("smallest", d.overhead_ratio.idxmin()), ("largest", d.overhead_ratio.idxmax())):
        n = int(d.n[i])
        both = sizes[n - 1] + sizes[n]
        print(f"  {label} diff: commit {n} visited {int(d.ops[i]):,} of {both:,} nodes in both trees "
              f"({d.ops[i] / both:.1%})")
    merged = d.merge(c[["n", "ops"]], on="n", suffixes=("_visited", "_changed"))
    for k in sorted(set(merged.ops_changed)):
        g = merged[merged.ops_changed == k]
        print(f"    {int(k):>2} files changed: commits {list(g.n.astype(int))}, visited {list(g.ops_visited.astype(int))}")
    fifth = merged[merged.n % 5 == 0]
    print(f"  commits that add 2 and remove 1 (every fifth): visited {list(fifth.ops_visited.astype(int))} "
          f"with files changed {list(fifth.ops_changed.astype(int))}")

    # ------------------------------------------------------------------
    header("8. Caps (configured limits; every capped row)")
    capped = t[t.capped == True]  # noqa: E712
    for (e, s, m), g in capped.groupby(["experiment", "structure", "method"]):
        extra = ""
        if g.cache_B.notna().any():
            extra = f" at {len(g)} rows over (M, B) settings"
        elif g.density.notna().any():
            extra = f" at densities {sorted(set(g.density))}"
        print(f"  {e} / {s} / {m}: n = {', '.join(pow2(n) if e not in ('cache_misses', 'wavelet_dna') else f'{int(n):,}' for n in sorted(set(g.n)))}{extra}")
    timed = t[(t.capped != True) & t.mean_time_s.notna()]  # noqa: E712
    longest = timed.loc[timed.mean_time_s.idxmax()]
    print(f"  slowest timed method: {sig(longest.mean_time_s)} s per call (mean) ({longest.experiment} / {longest.structure} / "
          f"{longest.method}, n = {int(longest.n):,}); the 20 s rule never fired")

    # ------------------------------------------------------------------
    header("9. Replay of the seeded version history (deterministic, nothing timed)")
    replay_vcs(c, d)
    return 0


def replay_vcs(commits: pd.DataFrame, diffs: pd.DataFrame) -> None:
    """Rebuild experiment 7's seeded history and show where each add/remove commit's keys sit.

    The history is deterministic (master seed 42), so this recomputes the
    same trees the benchmark measured; it first checks that every diff's
    nodes_visited equals the CSV value.
    """
    sys.path.insert(0, REPO_ROOT)
    import src.advds.benchmarks.performance as perf  # noqa: E402

    captured = {}
    original = perf.MiniVCS

    class Capture(original):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            captured["vcs"] = self

    perf.MiniVCS = Capture
    try:
        study = perf.Study(perf.FULL, 42)
        study.log = lambda *args, **kwargs: None
        perf.vcs_study(study)
    finally:
        perf.MiniVCS = original
    vcs = captured["vcs"]
    history = vcs.history()

    def locate(root, key):
        depth, x = 0, root
        while x is not None:
            if key == x.key:
                stack, size = [x], 0
                while stack:
                    y = stack.pop()
                    if y is not None:
                        size += 1
                        stack += [y.left, y.right]
                return depth, size
            x = x.left if key < x.key else x.right
            depth += 1
        return None

    same = True
    for number in range(5, len(history) + 1, 5):
        a, b = history[number - 2], history[number - 1]
        result = vcs.diff(a["id"], b["id"])
        csv_value = int(diffs[diffs.n == number].ops.iloc[0])
        same &= result["nodes_visited"] == csv_value
        parts = [f"removed {k} at depth {locate(vcs.tree(a['id']).root, k)[0]} "
                 f"(subtree {locate(vcs.tree(a['id']).root, k)[1]})" for k in result["removed"]]
        parts += [f"added {k} at depth {locate(vcs.tree(b['id']).root, k)[0]} "
                  f"(subtree {locate(vcs.tree(b['id']).root, k)[1]})" for k in result["added"]]
        print(f"  commit {number:<2} visited {result['nodes_visited']:>5,} (CSV {csv_value:,}); " + "; ".join(parts))
    print(f"  every replayed nodes_visited equals the CSV: {same}")


if __name__ == "__main__":
    sys.exit(main())
