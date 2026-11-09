"""Week 11 tests: plagiarism, DNA search, autocomplete, example files, generators.

Every expected value is a spec 9.6 anchor or is worked out by hand below, so
the tests never repeat the output of the code under test.

Plagiarism: d1 has the 3-shingles "the cat sat", "cat sat on", "sat on the"
(twice), "on the mat", "the mat and", "mat and the", "and the dog", "the dog
sat", "dog sat on", "on the log", 10 distinct. d2 swaps "cat" and "dog", so
its only new shingle is "and the cat" and d1's only lost one is "and the dog":
9 shared of 11 in the union, 9 / 11 = 0.8182. Every 2-shingle survives the
swap, so k = 2 scores 1.0. In d1 the shared shingles start at words 0-5 and
7-10, covering all 13 words with overlaps, so the matched region is the
whole of d1 (49 characters).

DNA: in seq1 = ACGTACGTGACG the only "GACG" starts at 8; in seq2 =
TTGACGAGACG it starts at 2 and 7. The 4-base windows of seq1 within one
substitution of "ACGA" are "ACGT" at 0 and 4; in seq2 "ACGA" itself is at 3.

Autocomplete: the rows are the spec 9.6 table; "careful" goes from 2 to 8
after six recorded queries.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.string_search_engine.applications.autocomplete import Autocomplete
from src.string_search_engine.applications.dna_search import GenomeIndex, load_fasta
from src.string_search_engine.applications.plagiarism import (
    PlagiarismReport,
    compare_documents,
    highlight,
)
from src.string_search_engine.utils.generators import (
    dna_text,
    english_like_text,
    generate_sample_genome,
    random_text,
    sample_patterns,
    write_fasta,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "src" / "string_search_engine" / "examples"

D1 = "the cat sat on the mat and the dog sat on the log"
D2 = "the dog sat on the mat and the cat sat on the log"

FASTA = ">seq1 sample\nACGTACGTGACG\n>seq2\nTTGACGA\nGACG\n"

FREQUENCIES = {"car": 5, "card": 3, "care": 7, "careful": 2, "cat": 4, "dog": 6}


# ---------------------------------------------------------------- plagiarism


def test_plagiarism_k3_anchor():
    report = compare_documents(D1, D2, k=3)
    assert isinstance(report, PlagiarismReport)
    assert report.shared_shingles == 9
    assert report.total_shingles == 11
    assert report.score == pytest.approx(0.8182, abs=1e-4)
    assert report.k == 3
    assert report.spurious_fingerprint_matches == 0


def test_plagiarism_k2_misses_reordering():
    assert compare_documents(D1, D2, k=2).score == pytest.approx(1.0)


def test_plagiarism_matched_region_covers_whole_document():
    report = compare_documents(D1, D2, k=3)
    assert len(D1) == 49
    assert report.matched_regions_a == [(0, 49)]
    assert report.matched_regions_b == [(0, 49)]
    assert highlight(report, D1) == "[[" + D1 + "]]"


def test_plagiarism_partial_highlight():
    # Shared 2-shingles: "one two" and "two three", merged into one region.
    a, b = "one two three four", "zero one two three five"
    report = compare_documents(a, b, k=2)
    assert report.shared_shingles == 2
    assert report.total_shingles == 5
    assert highlight(report, a, which="a") == "[[one two three]] four"
    assert highlight(report, b, which="b") == "zero [[one two three]] five"


def test_plagiarism_spurious_fingerprints_are_verified():
    # Base 256 is even, so a hash modulo 2 is the parity of the last code
    # point: "and the cat" (ends in 't') collides with "the cat sat" but the
    # texts differ. Verification keeps the score at 9 / 11.
    report = compare_documents(D1, D2, k=3, modulus=2)
    assert report.spurious_fingerprint_matches > 0
    assert report.shared_shingles == 9
    assert report.score == pytest.approx(9 / 11, abs=1e-4)


def test_longest_common_substring_anchor():
    report = compare_documents("xabxac", "abcabxabcd")
    assert report.longest_common_substring == "abxa"
    assert len(report.longest_common_substring) == 4


def test_plagiarism_errors():
    with pytest.raises(ValueError):
        compare_documents(D1, D2, k=0)
    with pytest.raises(ValueError):
        compare_documents("a\x01b", "ab")
    with pytest.raises(ValueError):
        highlight(compare_documents(D1, D2), D1, which="c")


def test_example_documents_share_two_sentences():
    doc_a = (EXAMPLES / "doc_a.txt").read_text(encoding="utf-8")
    doc_b = (EXAMPLES / "doc_b.txt").read_text(encoding="utf-8")

    def sentences(doc):
        return set(re.split(r"(?<=\.)\s+", doc.strip()))

    shared = sentences(doc_a) & sentences(doc_b)
    assert len(shared) == 2

    report = compare_documents(doc_a, doc_b, k=3)
    assert 0.0 < report.score < 1.0
    # Every 3-shingle inside a shared sentence is shared, so each sentence
    # (up to its last word) lies inside one matched region of both documents.
    for sentence in shared:
        body = sentence.rstrip(".")
        for doc, regions in ((doc_a, report.matched_regions_a), (doc_b, report.matched_regions_b)):
            start = doc.index(body)
            end = start + len(body)
            assert any(lo <= start and end <= hi for lo, hi in regions)
        assert len(report.longest_common_substring) >= len(sentence)
    assert "[[" in highlight(report, doc_a) and "]]" in highlight(report, doc_b, which="b")


# ---------------------------------------------------------------------- DNA


def test_load_fasta_anchor():
    assert load_fasta(FASTA) == {"seq1": "ACGTACGTGACG", "seq2": "TTGACGAGACG"}


def test_load_fasta_from_path(tmp_path):
    path = tmp_path / "tiny.fasta"
    path.write_text(FASTA, encoding="utf-8")
    assert load_fasta(path) == {"seq1": "ACGTACGTGACG", "seq2": "TTGACGAGACG"}
    assert load_fasta(str(path)) == {"seq1": "ACGTACGTGACG", "seq2": "TTGACGAGACG"}


@pytest.mark.parametrize("method", ["sa", "tree"])
def test_find_gene_anchor(method):
    index = GenomeIndex(load_fasta(FASTA))
    assert index.find_gene("GACG", method=method) == {"seq1": [8], "seq2": [2, 7]}


def test_find_gene_tree_above_cap_uses_suffix_array():
    index = GenomeIndex(load_fasta(FASTA), tree_cap=5)
    assert index.suffix_trees == {}
    assert index.find_gene("GACG", method="tree") == {"seq1": [8], "seq2": [2, 7]}


def test_find_mutations_anchor():
    result = GenomeIndex(load_fasta(FASTA)).find_mutations("ACGA", 1)
    assert result["seq1"] == [0, 4]
    assert result["seq2"] == [3]


def test_dna_errors():
    index = GenomeIndex(load_fasta(FASTA))
    with pytest.raises(ValueError):
        index.find_gene("", method="sa")
    with pytest.raises(ValueError):
        index.find_gene("GACG", method="kmp")
    with pytest.raises(ValueError):
        index.find_mutations("ACGA", -1)
    with pytest.raises(ValueError):
        load_fasta("ACGT\n>late header\n")


def test_sample_genome_file():
    path = EXAMPLES / "sample_genome.fasta"
    headers = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">")]
    assert len(headers) == 3
    assert all("synthetic" in header for header in headers)
    genome = load_fasta(path)
    assert len(genome) == 3
    assert all("synthetic" in seq_id for seq_id in genome)
    assert [len(sequence) for sequence in genome.values()] == [2000, 2000, 2000]
    assert all(set(sequence) <= set("ACGT") for sequence in genome.values())


# ------------------------------------------------------------- autocomplete


def test_autocomplete_rows():
    ac = Autocomplete.from_frequencies(FREQUENCIES)
    assert ac.complete("car", 3) == [("care", 7), ("car", 5), ("card", 3)]
    assert ac.complete("ca", 2) == [("care", 7), ("car", 5)]
    assert ac.complete("", 3) == [("care", 7), ("dog", 6), ("car", 5)]
    assert ac.complete("z") == []


def test_autocomplete_record_query():
    ac = Autocomplete.from_frequencies(FREQUENCIES)
    for _ in range(6):
        ac.record_query("careful")
    assert ac.complete("car", 3) == [("careful", 8), ("care", 7), ("car", 5)]


def test_autocomplete_trie_is_compressed():
    # "car", "card", "care", "careful" and "cat" all start with "ca", so the
    # root has one edge "ca" and one edge "dog".
    ac = Autocomplete.from_frequencies(FREQUENCIES)
    labels = sorted(child.label for child in ac.root.children.values())
    assert labels == ["ca", "dog"]


def test_autocomplete_from_text_counts_words():
    # "the" occurs 3 times; "car" and "cat" once each, so they tie and sort
    # alphabetically.
    ac = Autocomplete.from_text("The cat and the car. The end.")
    assert ac.complete("th") == [("the", 3)]
    assert ac.complete("ca") == [("car", 1), ("cat", 1)]


# --------------------------------------------------------------- generators


def test_generators_are_reproducible_with_a_seed():
    assert random_text(500, "abc", 5) == random_text(500, "abc", 5)
    assert dna_text(1000, 7) == dna_text(1000, 7)
    assert dna_text(1000, 7) != dna_text(1000, 8)
    assert english_like_text(2000, 3) == english_like_text(2000, 3)
    assert len(english_like_text(2000, 3)) == 2000
    text = dna_text(500, 1)
    patterns = sample_patterns(text, 6, 8, 9)
    assert patterns == sample_patterns(text, 6, 8, 9)
    assert all(pattern in text for pattern in patterns[:3])


def test_sample_genome_regenerates_from_seed():
    path = EXAMPLES / "sample_genome.fasta"
    assert write_fasta(generate_sample_genome(), None) == path.read_text(encoding="utf-8")
