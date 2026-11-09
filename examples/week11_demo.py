#!/usr/bin/env python3
"""Runnable demonstration of the Week 11 string search engine.

    python examples/week11_demo.py

Six short sections. Apart from the times, the output is the same on every
run, because the one large text is generated from a fixed seed:

1. **KMP** - the LPS arrays, the matches and the fallback trace of
   ``ababaca`` in ``bacbababaabcbab``.
2. **Rabin-Karp** - the matches at several moduli and the spurious-hit
   demonstration of ``abra`` in ``abracadabra`` at q = 7.
3. **Suffix array and suffix tree** - both indexes on the three anchor texts.
4. **Agreement** - every match anchor, the Unicode check and the worst case
   with all methods, and the comparisons each one makes.
5. **The engine** - ``find_all`` and ``recommend`` on 1 MB of seeded DNA.
6. **Applications** - plagiarism with highlighted regions, a FASTA gene
   search on the shipped synthetic genome, and autocomplete before and after
   recorded queries.

Each section reads its results through one idea: every matcher finds the same
occurrences and differs only in what it stores before it searches, and about
which string. The anchor values come from section 9 of the Week 11 build
spec, and every anchor printed is also asserted, so the script exits non-zero
rather than printing something untrue. The seeded agreement batteries are in
``tests/week11/``. It finishes in about ten seconds, most of it building the
two indexes over the 1 MB text.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.string_search_engine import StringSearchEngine  # noqa: E402
from src.string_search_engine.applications.autocomplete import Autocomplete  # noqa: E402
from src.string_search_engine.applications.dna_search import GenomeIndex, load_fasta  # noqa: E402
from src.string_search_engine.applications.plagiarism import (  # noqa: E402
    compare_documents,
    highlight,
)
from src.string_search_engine.core.kmp import compute_lps, kmp_search, naive_search  # noqa: E402
from src.string_search_engine.core.rabin_karp import (  # noqa: E402
    rabin_karp_multi,
    rabin_karp_search,
)
from src.string_search_engine.core.suffix_array import SuffixArray  # noqa: E402
from src.string_search_engine.core.suffix_tree import SuffixTree  # noqa: E402
from src.string_search_engine.utils.generators import dna_text, sample_patterns  # noqa: E402
from src.string_search_engine.utils.rolling_hash import RollingHash  # noqa: E402

WIDTH = 74
EXAMPLES = os.path.join(REPO_ROOT, "src", "string_search_engine", "examples")
MERSENNE_61 = (1 << 61) - 1


def banner(number: int, title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    print(f"  {label:<34} {value}")


def fmt_seconds(seconds: float) -> str:
    """Seconds as a short string in s, ms or us."""
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} us"


def wrap(text: str, indent: str = "    ") -> None:
    """Print ``text`` wrapped to the page width, keeping its own line breaks."""
    for paragraph in text.splitlines():
        line = indent
        for word in paragraph.split():
            if len(line) + len(word) + 1 > WIDTH and line.strip():
                print(line.rstrip())
                line = indent
            line += word + " "
        print(line.rstrip())


def section_kmp() -> None:
    banner(1, "Knuth-Morris-Pratt: the LPS array of the pattern")
    # Spec 9.1 anchors. lps[i] is the longest proper prefix of pattern[:i + 1]
    # that is also its suffix: for ababaca, "aba" at i = 4 and "a" at i = 6.
    lps_anchors = {
        "ababaca": [0, 0, 1, 2, 3, 0, 1],
        "aabaaab": [0, 1, 0, 1, 2, 2, 3],
        "abcabcabc": [0, 0, 0, 1, 2, 3, 4, 5, 6],
        "aaaa": [0, 1, 2, 3],
        "abacabab": [0, 0, 1, 0, 1, 2, 3, 2],
    }
    for pattern, expected in lps_anchors.items():
        assert compute_lps(pattern) == expected
        show(f"lps({pattern})", expected)

    print()
    for text, pattern, expected in MATCH_ANCHORS:
        assert kmp_search(text, pattern) == expected
        show(f"kmp_search({text!r}, {pattern!r})", expected)
    try:
        kmp_search("abc", "")
    except ValueError as error:
        show("empty pattern", f"ValueError: {error}")
    else:
        raise AssertionError("an empty pattern must raise ValueError")

    print()
    print("  What KMP does on a mismatch: q falls back to lps[q - 1] and the same")
    print("  text character is compared again; the text index never moves back.")
    text, pattern = "bacbababaabcbab", "ababaca"
    positions, events, stats = kmp_search(text, pattern, trace=True, stats=True)
    # Hand-checked scan. At i = 2 'c' breaks "a" (q 1 -> 0). The run "ababa"
    # from i = 4 breaks at i = 9 ('a' where 'c' is needed) and falls back
    # 5 -> lps[4] = 3, 3 -> lps[2] = 1, 1 -> lps[0] = 0 before 'a' matches
    # again. At i = 11 'c' breaks "ab" (q 2 -> 0). No occurrence. The scan
    # compares 1 + 1 + 2 + 1 + 5 + 4 + 1 + 2 + 1 + 1 + 1 = 20 characters, and
    # building the LPS array compares 8.
    assert positions == []
    assert events == [(2, 1, 0), (9, 5, 3), (9, 3, 1), (9, 1, 0), (11, 2, 0)]
    assert (stats.comparisons, stats.lps_comparisons) == (20, 8)
    show("text, pattern", f"{text}, {pattern}")
    print(f"    {'i':>3}  {'text[i]':>7}  {'q before':>8}  {'q after':>7}  matched before -> after")
    for i, q_before, q_after in events:
        print(f"    {i:>3}  {text[i]:>7}  {q_before:>8}  {q_after:>7}  "
              f"{pattern[:q_before]!r} -> {pattern[:q_after]!r}")
    show("positions", positions)
    show("comparisons (scan, LPS build)", f"{stats.comparisons}, {stats.lps_comparisons} "
         f"for {len(text)} text characters")


def section_rabin_karp() -> None:
    banner(2, "Rabin-Karp: the hash of the pattern, verified on every hit")
    for q in (7, 13, 101, MERSENNE_61):
        positions, stats = rabin_karp_search("abracadabra", "abra", modulus=q, stats=True)
        # Spec 9.2 anchors: 3 spurious hits at q = 7, none at the larger moduli.
        assert positions == [0, 7]
        assert stats.spurious_hits == (3 if q == 7 else 0)
        assert stats.hash_hits == stats.verifications == 2 + stats.spurious_hits
        name = "2^61 - 1" if q == MERSENNE_61 else str(q)
        show(f"abracadabra / abra, q = {name}",
             f"{positions}, {stats.hash_hits} hash hits, {stats.spurious_hits} spurious")
    for q, spurious in ((7, 1), (13, 0)):
        positions, stats = rabin_karp_search("mississippi", "issi", modulus=q, stats=True)
        assert positions == [1, 4] and stats.spurious_hits == spurious
        show(f"mississippi / issi, q = {q}", f"{positions}, {stats.spurious_hits} spurious")
    multi = rabin_karp_multi("abracadabra", ["abra", "cad", "bra", "zzz"])
    assert multi == {"abra": [0, 7], "cad": [4], "bra": [1, 8], "zzz": []}
    show("multi-pattern, abracadabra", multi)

    print()
    print("  The spurious hits at q = 7: windows whose hash equals h(abra) but whose")
    print("  characters differ. Verification throws them out, so the result is exact.")
    text, pattern = "abracadabra", "abra"
    hasher = RollingHash(base=256, modulus=7)
    target = hasher.hash(pattern)
    # Hand-checked: 256 = 4 (mod 7), so the weights are 4^3 = 1, 4^2 = 2, 4, 1,
    # and a = 6, b = 0, c = 1, d = 2, r = 2 (mod 7). h(abra) = 6 + 0 + 8 + 6
    # = 20 = 6. acad = 6 + 2 + 24 + 2 = 34 = 6, cada = 1 + 12 + 8 + 6 = 27 = 6,
    # adab = 6 + 4 + 24 + 0 = 34 = 6: three spurious hits. brac, raca and dabr
    # hash to 1, 3 and 2.
    assert target == 6
    expected_hashes = [6, 1, 3, 6, 6, 6, 2, 6]
    m = len(pattern)
    h = hasher.hash(text[:m])
    spurious = []
    print(f"    {'window':>6}  {'text':>4}  {'hash mod 7':>10}  verdict")
    for s in range(len(text) - m + 1):
        if s > 0:
            h = hasher.roll(h, text[s - 1], text[s + m - 1], m)
        window = text[s:s + m]
        assert h == hasher.hash(window) == expected_hashes[s]
        if h != target:
            verdict = ""
        elif window == pattern:
            verdict = "hit, verified: match"
        else:
            verdict = "hit, verified: SPURIOUS"
            spurious.append(s)
        print(f"    {s:>6}  {window:>4}  {h:>10}  {verdict}".rstrip())
    assert spurious == [3, 4, 5]
    show("h(abra) mod 7", target)


def section_indexes() -> None:
    banner(3, "Suffix array and suffix tree: an index of the text")
    # Spec 9.3 and 9.4 anchors. Distinct substrings are n(n + 1)/2 - sum(lcp):
    # banana 21 - 6 = 15, mississippi 66 - 13 = 53, abracadabra 66 - 12 = 54.
    anchors: Dict[str, Dict[str, Any]] = {
        "banana": {
            "sa": [5, 3, 1, 0, 4, 2], "lcp": [0, 1, 3, 0, 0, 2], "distinct": 15, "lrs": "ana",
            "leaves": 7, "internal": 3, "labels": ["a", "na", "ana"],
        },
        "mississippi": {
            "sa": [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2], "lcp": [0, 1, 1, 4, 0, 0, 1, 0, 2, 1, 3],
            "distinct": 53, "lrs": "issi", "leaves": 12, "internal": 6,
            "labels": ["i", "p", "s", "si", "ssi", "issi"],
        },
        "abracadabra": {
            "sa": [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2], "lcp": [0, 1, 4, 1, 1, 0, 3, 0, 0, 0, 2],
            "distinct": 54, "lrs": "abra", "leaves": 12, "internal": 4,
            "labels": ["a", "ra", "bra", "abra"],
        },
    }
    for text, a in anchors.items():
        sa = SuffixArray(text)
        assert sa.sa == a["sa"] and sa.lcp == a["lcp"]
        assert sa.count_distinct_substrings() == a["distinct"]
        assert sa.longest_repeated_substring() == a["lrs"]
        tree = SuffixTree(text)
        counts = tree.node_counts()
        assert counts == {"leaves": a["leaves"], "internal": a["internal"],
                          "total": a["leaves"] + a["internal"]}
        assert tree.internal_path_labels() == sorted(a["labels"])
        assert tree.count_distinct_substrings() == a["distinct"]
        assert tree.longest_repeated_substring() == a["lrs"]
        # The naive O(n^2) builder inserts every suffix one by one; the stack
        # construction from SA + LCP must give the same structure.
        naive = SuffixTree.from_naive(text)
        assert naive.node_counts() == counts
        assert naive.internal_path_labels() == tree.internal_path_labels()

        print(f"  {text}")
        show("  suffix array", sa.sa)
        show("  LCP", sa.lcp)
        show("  distinct substrings (SA, tree)",
             f"{sa.count_distinct_substrings()}, {tree.count_distinct_substrings()}")
        show("  longest repeated (SA, tree)",
             f"{sa.longest_repeated_substring()}, {tree.longest_repeated_substring()}")
        show("  tree leaves, internal nodes", f"{counts['leaves']}, {counts['internal']} "
             "(one leaf per suffix of text + $)")
        show("  internal path labels", ", ".join(a["labels"]))
        show("  same as the naive builder", "yes")

    # Spec 9.4: "ana" is the longest prefix of "anaz" that occurs in banana.
    assert SuffixTree("banana").longest_prefix_match("anaz") == 3
    show("banana longest_prefix_match(anaz)", 3)


#: Spec 9.1 match anchors, checked with every method in sections 1 and 4.
MATCH_ANCHORS = [
    ("abracadabra", "abra", [0, 7]),
    ("aaaaa", "aa", [0, 1, 2, 3]),
    ("abababab", "abab", [0, 2, 4]),
    ("mississippi", "issi", [1, 4]),
    ("ab", "abc", []),
]


def all_methods() -> Dict[str, Callable[[str, str], List[int]]]:
    """Every matcher as a ``(text, pattern) -> positions`` function."""
    return {
        "naive": naive_search,
        "KMP": kmp_search,
        "RK": rabin_karp_search,
        "RK q=7": lambda text, pattern: rabin_karp_search(text, pattern, modulus=7),
        "SA": lambda text, pattern: SuffixArray(text).find(pattern),
        "tree": lambda text, pattern: SuffixTree(text).find(pattern),
    }


def section_agreement() -> None:
    banner(4, "Agreement: every method finds the same occurrences")
    methods = all_methods()
    print(f"  {'text / pattern':<26} {'expected':<14} methods agreeing")
    cases = MATCH_ANCHORS + [("naïve café naïve", "naïve", [0, 11])]
    for text, pattern, expected in cases:
        for name, search in methods.items():
            assert search(text, pattern) == expected, (name, text, pattern)
        print(f"  {text + ' / ' + pattern:<26} {str(expected):<14} all {len(methods)}")

    print()
    # Spec 9.5 worst case. Hand-checked counts: there are 200 - 5 + 1 = 196
    # windows and every one matches. Naive compares all 5 characters of each,
    # 980. KMP compares each text character once (after a match q drops to
    # lps[4] = 4 and the next 'a' matches), 200, plus 4 to build lps("aaaaa").
    # Every Rabin-Karp window is a true hash hit, verified in full: 980 again,
    # which is the O(nm) worst case.
    text, pattern = "a" * 200, "a" * 5
    for name, search in methods.items():
        assert search(text, pattern) == list(range(196)), name
    show("'a' * 200 / 'a' * 5", f"196 positions from all {len(methods)} methods")
    naive_positions, naive_stats = naive_search(text, pattern, stats=True)
    kmp_positions, kmp_stats = kmp_search(text, pattern, stats=True)
    rk_positions, rk_stats = rabin_karp_search(text, pattern, stats=True)
    assert naive_stats.comparisons == 980
    assert (kmp_stats.comparisons, kmp_stats.lps_comparisons) == (200, 4)
    assert (rk_stats.hash_hits, rk_stats.spurious_hits, rk_stats.comparisons) == (196, 0, 980)
    show("naive comparisons", f"{naive_stats.comparisons} = 196 windows x 5")
    show("KMP comparisons (scan + LPS)", f"{kmp_stats.comparisons} + {kmp_stats.lps_comparisons}")
    show("Rabin-Karp hits, comparisons", f"{rk_stats.hash_hits} true hits, "
         f"{rk_stats.comparisons} verified characters")


def section_engine() -> None:
    banner(5, "The engine: four methods on 1 MB of seeded DNA")
    n = 1_000_000
    text = dna_text(n, seed=11)
    patterns = sample_patterns(text, count=6, m=12, seed=11, present_fraction=0.5)
    show("text", f"dna_text({n:,}, seed=11), {len(text):,} bases")
    show("patterns", "6 of length 12; the first 3 cut from the text")
    engine = StringSearchEngine(text)
    print()
    print(f"  {'pattern':<14} {'count':>5}  {'KMP':>9}  {'Rabin-Karp':>10}  {'SA':>9}  {'tree':>9}")
    for index, pattern in enumerate(patterns):
        results = engine.find_all(pattern)  # raises AssertionError on any disagreement
        count = results["kmp"].count
        if index < 3:
            assert count >= 1  # cut from the text, so it occurs at least once
        times = "  ".join(f"{fmt_seconds(results[m].search_time_s):>{w}}"
                          for m, w in (("kmp", 9), ("rk", 10), ("sa", 9), ("tree", 9)))
        print(f"  {pattern:<14} {count:>5}  {times}")
    print("  (query times only; every row is the same positions from all four)")

    print()
    stats = engine.stats()
    for method, name in (("kmp", "KMP"), ("rk", "Rabin-Karp"), ("sa", "suffix array"),
                         ("tree", "suffix tree")):
        row = stats[method]
        assert row["queries"] == len(patterns)
        if method in ("kmp", "rk"):
            assert row["preprocessing_s"] == 0.0 and row["index_memory_kb"] == 0.0
            show(f"{name}: build, index size", "none, none (nothing kept about the text)")
        else:
            show(f"{name}: build, index size", f"{fmt_seconds(row['preprocessing_s'])}, "
                 f"{row['index_memory_kb'] / 1024:.1f} MB")
    print()
    print("  recommend(expected_queries):")
    for queries in (1, 10, 100, 1000):
        wrap(engine.recommend(queries))


def section_applications() -> None:
    banner(6, "Applications: plagiarism, DNA search and autocomplete")
    print("  Plagiarism: word 3-shingles fingerprinted with the rolling hash.")
    d1 = "the cat sat on the mat and the dog sat on the log"
    d2 = "the dog sat on the mat and the cat sat on the log"
    # Spec 9.6 anchors, hand-checked. Each sentence has 13 words, 11 shingles
    # and 10 distinct ones ("sat on the" twice). They share 9: only "and the
    # dog" and "and the cat" differ, so the union is 11 and 9 / 11 = 0.8182.
    # With k = 2 every bigram of one appears in the other, so the score is 1.
    report = compare_documents(d1, d2, k=3)
    assert (report.shared_shingles, report.total_shingles) == (9, 11)
    assert abs(report.score - 0.8182) <= 1e-4
    report2 = compare_documents(d1, d2, k=2)
    assert report2.score == 1.0
    lcs = compare_documents("xabxac", "abcabxabcd").longest_common_substring
    assert lcs == "abxa"
    show("d1", d1)
    show("d2", d2)
    show("k = 3: shared / total, score", f"{report.shared_shingles} / {report.total_shingles}, "
         f"{report.score:.4f}")
    show("k = 2: score", f"{report2.score:.4f} (bigrams miss the reordering)")
    show("LCS(xabxac, abcabxabcd)", lcs)

    print()
    with open(os.path.join(EXAMPLES, "doc_a.txt"), encoding="utf-8") as handle:
        doc_a = handle.read()
    with open(os.path.join(EXAMPLES, "doc_b.txt"), encoding="utf-8") as handle:
        doc_b = handle.read()
    shared = [
        "Every new arrival is graded for sleeve and disc condition before it gets a price sticker.",
        "Damaged sleeves are set aside in a clearance crate near the register and marked down by "
        "half.",
    ]
    for sentence in shared:
        assert sentence in doc_a and sentence in doc_b
    docs = compare_documents(doc_a, doc_b, k=3)
    marked_a, marked_b = highlight(docs, doc_a, "a"), highlight(docs, doc_b, "b")
    # The two shared sentences must come out highlighted in both documents.
    for sentence in shared:
        body = sentence.rstrip(".")
        assert any(body in marked_a[s:e] for s, e in _marked_spans(marked_a))
        assert any(body in marked_b[s:e] for s, e in _marked_spans(marked_b))
    assert 0 < docs.score < 1 and docs.spurious_fingerprint_matches == 0
    show("doc_a.txt, doc_b.txt", f"{len(doc_a.split())} and {len(doc_b.split())} words")
    show("k = 3: shared / total, score", f"{docs.shared_shingles} / {docs.total_shingles}, "
         f"{docs.score:.4f}")
    show("spurious fingerprint matches", docs.spurious_fingerprint_matches)
    show("longest common substring", f"{len(docs.longest_common_substring)} characters:")
    wrap(repr(docs.longest_common_substring))
    print("  doc_a with matched regions in [[ ]]:")
    wrap(marked_a)
    print("  doc_b with matched regions in [[ ]]:")
    wrap(marked_b)

    print()
    print("  DNA search: one suffix array and one suffix tree per sequence.")
    fasta = ">seq1 sample\nACGTACGTGACG\n>seq2\nTTGACGA\nGACG\n"
    # Spec 9.6 anchors. seq2 is its two lines joined: TTGACGA + GACG. GACG
    # starts at 8 in seq1 and at 2 and 7 in seq2. ACGA is within one mismatch
    # of ACGT at 0 and ACGT at 4 in seq1 (last base differs).
    sequences = load_fasta(fasta)
    assert sequences == {"seq1": "ACGTACGTGACG", "seq2": "TTGACGAGACG"}
    index = GenomeIndex(sequences)
    for method in ("sa", "tree"):
        assert index.find_gene("GACG", method=method) == {"seq1": [8], "seq2": [2, 7]}
    assert index.find_mutations("ACGA", 1)["seq1"] == [0, 4]
    show("load_fasta", sequences)
    show("find_gene(GACG), sa and tree", index.find_gene("GACG"))
    show("find_mutations(ACGA, 1), seq1", index.find_mutations("ACGA", 1)["seq1"])

    print()
    genome = load_fasta(os.path.join(EXAMPLES, "sample_genome.fasta"))
    assert list(genome) == ["synthetic_seq1", "synthetic_seq2", "synthetic_seq3"]
    assert all(len(seq) == 2000 for seq in genome.values())
    gene = genome["synthetic_seq2"][1000:1008]
    index = GenomeIndex(genome)
    assert sorted(index.suffix_trees) == sorted(genome)
    by_sa, by_tree = index.find_gene(gene, method="sa"), index.find_gene(gene, method="tree")
    assert by_sa == by_tree == {name: kmp_search(seq, gene) for name, seq in genome.items()}
    assert 1000 in by_sa["synthetic_seq2"]
    near = index.find_mutations(gene, 1)
    for name, seq in genome.items():
        brute = [s for s in range(len(seq) - len(gene) + 1)
                 if sum(a != b for a, b in zip(seq[s:s + len(gene)], gene)) <= 1]
        assert near[name] == brute
    show("sample_genome.fasta (synthetic)", f"{len(genome)} sequences of 2,000 bases")
    show("gene = synthetic_seq2[1000:1008]", gene)
    print("  find_gene(gene), sa and tree; then find_mutations(gene, 1):")
    for name in genome:
        print(f"    {name:<16} exact {by_sa[name]}; within 1 mismatch {near[name]}")

    print()
    print("  Autocomplete: top-k completions by weight over a radix trie.")
    ac = Autocomplete.from_frequencies({"car": 5, "card": 3, "care": 7, "careful": 2, "cat": 4,
                                        "dog": 6})
    # Spec 9.6 anchors: ties would break alphabetically, weights decide here.
    before = {
        ("car", 3): [("care", 7), ("car", 5), ("card", 3)],
        ("ca", 2): [("care", 7), ("car", 5)],
        ("", 3): [("care", 7), ("dog", 6), ("car", 5)],
        ("z", 5): [],
    }
    for (prefix, k), expected in before.items():
        assert ac.complete(prefix, k) == expected
        show(f"complete({prefix!r}, {k})", _pairs(ac.complete(prefix, k)))
    for _ in range(6):
        ac.record_query("careful")
    after = ac.complete("car", 3)
    # careful rises from 2 to 2 + 6 = 8, above care's 7.
    assert after == [("careful", 8), ("care", 7), ("car", 5)]
    show("record_query('careful') x 6", "careful: 2 -> 8")
    show("complete('car', 3) after", _pairs(after))


def _marked_spans(marked: str) -> List[tuple]:
    """The ``(start, end)`` spans inside ``[[ ]]`` markers of a highlighted text."""
    spans, start = [], 0
    while True:
        open_at = marked.find("[[", start)
        if open_at < 0:
            return spans
        close_at = marked.find("]]", open_at)
        spans.append((open_at + 2, close_at))
        start = close_at + 2


def _pairs(pairs: List[tuple]) -> str:
    """Completion pairs with integral weights shown as integers."""
    return "[" + ", ".join(f"({word!r}, {weight:g})" for word, weight in pairs) + "]"


def main() -> int:
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 11 demonstration")
    print("Fast pattern matching, indexing and text processing - Robert Deibel")
    print("=" * WIDTH)
    print("  The same occurrences, four ways: what each method stores before it")
    print("  searches, and about which string. Every anchor below is asserted.")
    section_kmp()
    section_rabin_karp()
    section_indexes()
    section_agreement()
    section_engine()
    section_applications()
    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
