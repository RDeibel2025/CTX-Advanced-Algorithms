"""Tests for the three longest common subsequence implementations.

An LCS bug does not announce itself. It returns a plausible integer, or a
string that looks like an answer, and only a second opinion can tell the
difference. Re-deriving the same recurrence inside the test file would
reproduce any mistake in it and then call the echo a confirmation, so the
claims here are checked against witnesses that share no mechanism with
``src/dp/lcs.py``:

* **A pair worked out by hand.** ``"AGGTAB"`` against ``"GXTXAYB"`` has
  LCS ``"GTAB"``, length four. The reasoning is written out in the
  comment above the test, so that one expected value in the file came
  from paper rather than from code.
* **Brute force over subsequences.** For short pairs every subsequence of
  the shorter string is enumerated, longest first, and the first one that
  also survives inside the longer string is the answer by definition. It
  knows nothing about optimal substructure, prefixes or tables.
* **An independent subsequence checker.** :func:`is_subsequence` walks a
  candidate against a text with a single iterator, which is the
  definition of "appears in order" and not a re-run of the backward walk
  that :func:`lcs_reconstruct` uses to produce it.
* **A closed form for the recursion tree.** On two strings with no
  character in common every call forks, and the node count of that tree
  is exactly ``2 * C(m + n, m) - 1``. That number comes from the
  binomial coefficient, not from the counter being tested.

Two rules shape what is asserted. First, when several longest common
subsequences are equally valid, the tests assert the *length* and the
*subsequence property*, never one particular string: the module documents
its tie-break as an implementation choice, and a test that pinned the
exact output would freeze a decision the module is free to revisit.
Second, the recursive variant is never asked for more than a handful of
characters, because the whole point of its presence is that it is
exponential.

The instrumentation tests exist because the report's argument rests on
them: ``"tab"`` reports a maximum recursion depth of 1 at every size,
while ``"memo"`` and ``"recursive"`` both descend ``len(X) + len(Y)``
frames. Memoization removes repeated calls, not the depth of the deepest
chain, and that asymmetry is a property of the runtime rather than of the
algorithm.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Tuple

import pytest

from src.dp.lcs import (
    VARIANTS,
    lcs_instrumented,
    lcs_memo,
    lcs_reconstruct,
    lcs_recursive,
    lcs_tab,
)

#: The three length-only entry points, in a fixed order, so that every
#: property true of the algorithm rather than of one implementation can be
#: parametrised instead of written out three times.
VARIANT_NAMES: List[str] = ["recursive", "memo", "tab"]

#: The same three functions, looked up through the module's own registry so
#: that a test failure tells us whether VARIANTS drifted from the functions.
VARIANT_FUNCS = [VARIANTS[name] for name in VARIANT_NAMES]

#: Longest input the exponential variant is ever handed. Two disjoint
#: 12-character strings already cost 2 * C(24, 12) - 1 calls, about 5.4
#: million, so the battery keeps a small alphabet and the instrumented
#: tests stay far below this.
MAX_RECURSIVE_LENGTH = 12


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def is_subsequence(candidate: str, text: str) -> bool:
    """Return True if ``candidate`` appears inside ``text`` in order.

    One iterator is walked across ``text`` once. Searching it for the next
    wanted character consumes everything up to and including the match, so
    each character of ``candidate`` can only be matched to a position
    strictly after the previous one. That is the definition of a
    subsequence, written without any reference to prefixes, tables or the
    backward walk that produced the candidate.

    Args:
        candidate: The string that should appear in order.
        text: The string it should appear inside.

    Returns:
        True if every character of ``candidate`` can be matched, in order,
        to a distinct position of ``text``.
    """
    remaining = iter(text)
    return all(character in remaining for character in candidate)


def brute_force_lcs_length(X: str, Y: str) -> int:
    """Return the LCS length by enumerating subsequences, longest first.

    The definition turned straight into code: every subsequence of the
    shorter string is tried in decreasing length, and the first one that
    also sits inside the longer string is a longest common subsequence.
    There is no recurrence and no memory here, which is exactly why it is
    trustworthy as a second opinion; it is also why it is only ever used
    on short inputs.

    Args:
        X: The first string.
        Y: The second string.

    Returns:
        The length of the longest common subsequence.
    """
    shorter, longer = (X, Y) if len(X) <= len(Y) else (Y, X)
    for size in range(len(shorter), 0, -1):
        for combination in itertools.combinations(shorter, size):
            if is_subsequence("".join(combination), longer):
                return size
    return 0


def brute_force_optimal_strings(X: str, Y: str) -> set:
    """Return every distinct longest common subsequence of two strings.

    Used only to *prove that a tie exists* before a test asserts that the
    implementation may return any of the winners. Exponential, so it is
    handed nothing longer than a few characters.

    Args:
        X: The first string.
        Y: The second string.

    Returns:
        A set of strings, all of the same maximal length, each a
        subsequence of both inputs.
    """
    shorter, longer = (X, Y) if len(X) <= len(Y) else (Y, X)
    for size in range(len(shorter), 0, -1):
        winners = {
            "".join(combination)
            for combination in itertools.combinations(shorter, size)
            if is_subsequence("".join(combination), longer)
        }
        if winners:
            return winners
    return {""}


def build_battery(seed: int = 5300, count: int = 60) -> List[Tuple[str, str]]:
    """Return a reproducible list of short string pairs.

    The alphabets are deliberately tiny. Two random strings over 26 letters
    almost never share anything, so a battery drawn that way would test the
    zero case sixty times over; over two to four symbols the pairs have
    real, non-trivial common subsequences and the three variants have
    something to disagree about.

    Lengths run from 0 to ``MAX_RECURSIVE_LENGTH`` inclusive, empty strings
    included, so the exponential variant can be run on the whole battery
    without the suite noticing.

    Args:
        seed: Passed to a dedicated :class:`random.Random` so the battery is
            identical on every run without disturbing global random state.
        count: How many pairs to generate.

    Returns:
        A list of ``(X, Y)`` pairs.
    """
    rng = random.Random(seed)
    alphabets = ["AB", "ABC", "ABCD"]
    pairs: List[Tuple[str, str]] = []
    for _ in range(count):
        alphabet = rng.choice(alphabets)
        lengths = (
            rng.randint(0, MAX_RECURSIVE_LENGTH),
            rng.randint(0, MAX_RECURSIVE_LENGTH),
        )
        pairs.append(
            tuple(
                "".join(rng.choice(alphabet) for _ in range(length))
                for length in lengths
            )
        )
    return pairs


#: The shared battery. Built once at import so that every test in the file
#: is looking at the same pairs and a failure in one can be compared with a
#: failure in another.
BATTERY: List[Tuple[str, str]] = build_battery()

#: Readable ids for the parametrised battery tests.
BATTERY_IDS = [f"{index:02d}-{len(x)}x{len(y)}" for index, (x, y) in enumerate(BATTERY)]


# ----------------------------------------------------------------------
# The hand-computed anchor
# ----------------------------------------------------------------------
class TestHandComputedAnchor:
    """The one pair in this file whose answer was found on paper."""

    def test_anchor_length_is_four(self):
        """X = "AGGTAB", Y = "GXTXAYB" have LCS "GTAB", so length 4.

        Worked out by hand, not by running the code. Line the two up:

            X = A G G T A B
            Y = G X T X A Y B

        Take the second G of X and the G of Y; the T of X and the T of Y,
        which lies after that G; the A of X and the A of Y, which lies
        after that T; finally the B that ends both strings. G, T, A, B in
        that order sit inside both strings, so the LCS is at least 4.

        It cannot be 5. A common subsequence of length 5 would have to use
        five of the six characters of X, so it would drop at most one of
        A, G, G, T, A, B. Y contains exactly one G, one T, one A and one B,
        and they occur in Y in the order G, T, A, B, so at most one of X's
        two Gs and at most one of X's two As can ever be matched. That
        already forces two characters of X to be dropped, which caps any
        common subsequence at 4.
        """
        assert lcs_tab("AGGTAB", "GXTXAYB") == 4

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_every_variant_reaches_the_anchor(self, variant: str):
        assert VARIANTS[variant]("AGGTAB", "GXTXAYB") == 4

    def test_anchor_reconstructs_to_gtab(self):
        """"GTAB" is the only length-4 common subsequence of this pair.

        There is no tie to worry about here, which is why the exact string
        is safe to assert on this one pair and nowhere else. The set of
        optimal strings is computed by brute force below so the claim of
        uniqueness is checked rather than assumed.
        """
        assert brute_force_optimal_strings("AGGTAB", "GXTXAYB") == {"GTAB"}
        assert lcs_reconstruct("AGGTAB", "GXTXAYB") == "GTAB"

    def test_anchor_agrees_with_brute_force(self):
        assert brute_force_lcs_length("AGGTAB", "GXTXAYB") == 4

    def test_clrs_example(self):
        """CLRS section 14.4: "ABCBDAB" and "BDCABA" have LCS length 4."""
        assert lcs_tab("ABCBDAB", "BDCABA") == 4
        assert brute_force_lcs_length("ABCBDAB", "BDCABA") == 4


# ----------------------------------------------------------------------
# Agreement across the battery
# ----------------------------------------------------------------------
class TestAgreement:
    """All three variants compute the same function, and it is the right one."""

    @pytest.mark.parametrize("pair", BATTERY, ids=BATTERY_IDS)
    def test_all_three_variants_agree(self, pair: Tuple[str, str]):
        X, Y = pair
        results = {name: VARIANTS[name](X, Y) for name in VARIANT_NAMES}
        assert len(set(results.values())) == 1, (
            f"variants disagreed on {X!r} and {Y!r}: {results}"
        )

    def test_the_agreed_answer_is_the_brute_force_answer(self):
        """Agreement is worthless if all three share a bug, so price it.

        The brute force oracle enumerates subsequences and knows nothing
        about the recurrence, so this is the test that says the common
        answer is correct rather than merely consistent.
        """
        for X, Y in BATTERY:
            expected = brute_force_lcs_length(X, Y)
            assert lcs_tab(X, Y) == expected, f"{X!r} vs {Y!r}"

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_lcs_is_symmetric(self, func):
        """A common subsequence does not care which string was named first."""
        for X, Y in BATTERY:
            assert func(X, Y) == func(Y, X), f"{X!r} vs {Y!r}"

    def test_the_battery_actually_exercises_non_trivial_answers(self):
        """Guard against a battery that quietly became all zeros.

        A seeded battery over a large alphabet would pass every agreement
        test above while testing almost nothing, so the shape of the
        battery is itself asserted.
        """
        lengths = [lcs_tab(X, Y) for X, Y in BATTERY]
        assert max(lengths) >= 5
        assert sum(1 for length in lengths if length > 0) > len(BATTERY) // 2

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_result_never_exceeds_the_shorter_input(self, func):
        """A common subsequence lives inside both strings, so it cannot be longer."""
        for X, Y in BATTERY:
            assert 0 <= func(X, Y) <= min(len(X), len(Y))

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_inputs_are_not_modified(self, func):
        """The docstrings promise the arguments are not touched."""
        X, Y = "AGGTAB", "GXTXAYB"
        func(X, Y)
        assert X == "AGGTAB" and Y == "GXTXAYB"


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
class TestReconstruct:
    """The string, not just its length, checked against the definition."""

    @pytest.mark.parametrize("pair", BATTERY, ids=BATTERY_IDS)
    def test_reconstruction_is_a_common_subsequence_of_the_right_length(
        self, pair: Tuple[str, str]
    ):
        """The two properties that define a correct answer, over the battery.

        Nothing here mentions a particular string. The candidate has to sit
        inside both inputs in order, and it has to be as long as
        :func:`lcs_tab` says the optimum is. Any string satisfying both is
        a correct answer, and no string failing either is.
        """
        X, Y = pair
        answer = lcs_reconstruct(X, Y)
        assert is_subsequence(answer, X), f"{answer!r} is not inside {X!r}"
        assert is_subsequence(answer, Y), f"{answer!r} is not inside {Y!r}"
        assert len(answer) == lcs_tab(X, Y)

    def test_the_subsequence_checker_rejects_non_subsequences(self):
        """The oracle is only useful if it can say no."""
        assert is_subsequence("", "ANYTHING")
        assert is_subsequence("GTAB", "AGGTAB")
        assert not is_subsequence("BA", "AB")          # order matters
        assert not is_subsequence("AA", "A")           # multiplicity matters
        assert not is_subsequence("AX", "AGGTAB")      # missing character
        assert not is_subsequence("A", "")

    def test_reconstruction_is_deterministic(self):
        """Same input, same output: the tie-break is fixed, even if arbitrary."""
        for X, Y in BATTERY[:10]:
            assert lcs_reconstruct(X, Y) == lcs_reconstruct(X, Y)

    def test_a_pair_with_several_distinct_optima(self):
        """"ABCBDAB" and "BDCABA" admit more than one length-4 answer.

        "BCBA", "BCAB" and "BDAB" are all common subsequences of length 4.
        Asserting on any one of them would encode this implementation's
        tie-break, which its own docstring says callers must not rely on,
        so the assertions are on the length and on the subsequence
        property. The tie is proved to exist first, otherwise the test
        would silently weaken into an ordinary single-answer check.
        """
        X, Y = "ABCBDAB", "BDCABA"
        optima = brute_force_optimal_strings(X, Y)
        assert len(optima) > 1, "this pair was supposed to have several optima"
        assert all(len(candidate) == 4 for candidate in optima)

        answer = lcs_reconstruct(X, Y)
        assert len(answer) == 4
        assert is_subsequence(answer, X) and is_subsequence(answer, Y)
        assert answer in optima

    @pytest.mark.parametrize(
        "X, Y",
        [("ABC", "ACB"), ("AGCAT", "GAC"), ("AB", "BA"), ("XMJYAUZ", "MZJAWXU")],
        ids=["abc-acb", "agcat-gac", "ab-ba", "xmjyauz-mzjawxu"],
    )
    def test_ties_are_answered_by_property_not_by_identity(self, X: str, Y: str):
        optima = brute_force_optimal_strings(X, Y)
        answer = lcs_reconstruct(X, Y)
        assert len(answer) == brute_force_lcs_length(X, Y)
        assert is_subsequence(answer, X) and is_subsequence(answer, Y)
        assert answer in optima

    def test_nothing_in_common_gives_the_empty_string(self):
        """Empty string, not None: the return type does not change."""
        assert lcs_reconstruct("abc", "xyz") == ""
        assert lcs_reconstruct("", "ABC") == ""
        assert lcs_reconstruct("", "") == ""

    def test_identical_strings_reconstruct_to_themselves(self):
        assert lcs_reconstruct("SAME", "SAME") == "SAME"

    def test_reconstruction_does_not_modify_its_inputs(self):
        X, Y = "AGGTAB", "GXTXAYB"
        lcs_reconstruct(X, Y)
        assert X == "AGGTAB" and Y == "GXTXAYB"


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    """Empty, identical, disjoint, single characters, prefixes, repeats."""

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    @pytest.mark.parametrize(
        "X, Y, expected",
        [
            ("", "", 0),
            ("", "ANYTHING", 0),
            ("ANYTHING", "", 0),
            ("SAME", "SAME", 4),
            ("A", "A", 1),
            ("A", "B", 0),
            ("abc", "xyz", 0),
            ("ABCDE", "ABC", 3),
            ("ABC", "ABCDE", 3),
            ("AAAA", "AA", 2),
            ("AA", "AAAA", 2),
            ("AAAA", "AAAA", 4),
            ("ABCDEFG", "AXCXEXG", 4),
            ("AGGTAB", "GXTXAYB", 4),
        ],
        ids=[
            "both-empty",
            "first-empty",
            "second-empty",
            "identical",
            "single-match",
            "single-mismatch",
            "disjoint-alphabets",
            "prefix-of-first",
            "prefix-of-second",
            "repeats-longer-first",
            "repeats-shorter-first",
            "all-repeats-equal",
            "non-contiguous",
            "anchor",
        ],
    )
    def test_edge_case_lengths(self, func, X: str, Y: str, expected: int):
        assert func(X, Y) == expected

    @pytest.mark.parametrize(
        "X, Y",
        [
            ("", ""),
            ("", "ANYTHING"),
            ("ANYTHING", ""),
            ("SAME", "SAME"),
            ("A", "A"),
            ("A", "B"),
            ("abc", "xyz"),
            ("ABCDE", "ABC"),
            ("AAAA", "AA"),
        ],
        ids=[
            "both-empty",
            "first-empty",
            "second-empty",
            "identical",
            "single-match",
            "single-mismatch",
            "disjoint-alphabets",
            "prefix",
            "repeats",
        ],
    )
    def test_edge_case_reconstructions(self, X: str, Y: str):
        answer = lcs_reconstruct(X, Y)
        assert is_subsequence(answer, X) and is_subsequence(answer, Y)
        assert len(answer) == lcs_tab(X, Y)

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    @pytest.mark.parametrize("size", [1, 5, 40])
    def test_a_string_against_itself_is_its_own_length(self, func, size: int):
        """Identical strings are the recursion's best case: one diagonal chain.

        40 characters is safe even for the exponential variant, because a
        match never forks; it is disjoint alphabets that explode.
        """
        text = "".join("ABCDEFGHIJ"[index % 10] for index in range(size))
        assert func(text, text) == size

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_a_prefix_is_entirely_contained(self, func):
        text = "DYNAMIC"
        for cut in range(len(text) + 1):
            assert func(text, text[:cut]) == cut

    def test_tabulation_handles_inputs_the_recursive_forms_cannot(self):
        """Depth 1 means no interpreter limit applies, at any size.

        600 characters a side would need a recursion limit of 1,200 in the
        top-down forms, against CPython's default of 1,000. The module
        deliberately does not raise that limit.
        """
        assert lcs_tab("A" * 600, "A" * 600) == 600


# ----------------------------------------------------------------------
# Case sensitivity
# ----------------------------------------------------------------------
class TestCaseSensitivity:
    """Characters are compared with ==, so case is part of the character."""

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_upper_and_lower_case_never_match(self, func):
        assert func("ABCD", "abcd") == 0

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_only_the_matching_case_counts(self, func):
        """"aBcD" against "ABCD" shares only the two upper case letters."""
        assert func("aBcD", "ABCD") == 2

    def test_reconstruction_preserves_case(self):
        assert lcs_reconstruct("aBcD", "ABCD") == "BD"

    def test_case_folding_is_the_callers_job(self):
        """The documented workaround, stated as a test so it stays true."""
        X, Y = "Dynamic Programming", "DYNAMIC programming"
        assert lcs_tab(X, Y) < len(X)
        assert lcs_tab(X.casefold(), Y.casefold()) == len(X)


# ----------------------------------------------------------------------
# Instrumentation: the asymmetry the report is built on
# ----------------------------------------------------------------------
class TestInstrumentation:
    """Call counts and recursion depth, the numbers the write-up quotes."""

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 6])
    def test_tabulation_has_depth_one_while_the_others_grow(self, size: int):
        """The headline asymmetry, at six sizes.

        The two strings share no character, so no call takes the diagonal
        branch and the deepest chain consumes one character per frame:
        depth is exactly ``len(X) + len(Y)`` for both top-down forms.
        Memoization removes repeated calls, not the length of that chain,
        which is why "memo" is no shallower than "recursive". Bottom-up
        never nests at all.
        """
        X = "ABCDEF"[:size]
        Y = "uvwxyz"[:size]

        for variant in ("recursive", "memo"):
            _length, counter = lcs_instrumented(X, Y, variant)
            assert counter.max_depth == len(X) + len(Y), variant

        _length, counter = lcs_instrumented(X, Y, "tab")
        assert counter.max_depth == 1

    def test_depth_is_strictly_increasing_for_the_top_down_forms(self):
        """Said as a trend rather than a formula, which is how the report says it."""
        for variant in ("recursive", "memo"):
            depths = [
                lcs_instrumented("ABCDEF"[:size], "uvwxyz"[:size], variant)[1].max_depth
                for size in range(1, 7)
            ]
            assert depths == sorted(depths) and len(set(depths)) == len(depths), variant

    def test_tabulation_depth_stays_one_however_long_the_input(self):
        for size in (1, 10, 100, 400):
            _length, counter = lcs_instrumented("A" * size, "B" * size, "tab")
            assert counter.max_depth == 1

    @pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 5])
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_plain_recursion_visits_exactly_the_binomial_tree(self, m: int, n: int):
        """2 * C(m + n, m) - 1 nodes, from the binomial coefficient.

        With no character in common every call forks, so the tree size
        satisfies T(i, j) = 1 + T(i-1, j) + T(i, j-1) with T(0, j) =
        T(i, 0) = 1. That solves to 2 * C(i + j, i) - 1, and
        :func:`math.comb` supplies the number without consulting the code
        under test. When either string is empty the whole run is the single
        base-case frame.
        """
        X = "ABCDE"[:m]
        Y = "vwxyz"[:n]
        expected = 2 * math.comb(m + n, m) - 1 if m and n else 1

        _length, counter = lcs_instrumented(X, Y, "recursive")
        assert counter.calls == expected

    @pytest.mark.parametrize("m", [0, 1, 2, 3, 5, 7])
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 7])
    def test_tabulation_evaluates_every_cell_of_the_table(self, m: int, n: int):
        """Bottom-up counts subproblems, and the table has (m+1)(n+1) of them."""
        X = "ABCDEFG"[:m]
        Y = "tuvwxyz"[:n]
        _length, counter = lcs_instrumented(X, Y, "tab")
        assert counter.calls == (m + 1) * (n + 1)

    @pytest.mark.parametrize("size", [3, 4, 5, 6])
    def test_memoization_collapses_the_exponential_tree(self, size: int):
        """Memo calls stay inside a polynomial bound while the tree explodes.

        The bound 2 * (m + 1) * (n + 1) is loose on purpose: it says
        "at most a constant times the number of subproblems", which is the
        claim, rather than pinning an exact figure that would just be the
        implementation read back.
        """
        X = "ABCDEF"[:size]
        Y = "uvwxyz"[:size]

        _length, memo_counter = lcs_instrumented(X, Y, "memo")
        _length, recursive_counter = lcs_instrumented(X, Y, "recursive")

        assert memo_counter.calls <= 2 * (size + 1) * (size + 1)
        assert memo_counter.calls < recursive_counter.calls

    def test_the_gap_between_memo_and_recursion_widens_with_size(self):
        """Not a constant-factor saving: the ratio itself grows."""
        ratios = []
        for size in (3, 4, 5, 6):
            X = "ABCDEF"[:size]
            Y = "uvwxyz"[:size]
            memo = lcs_instrumented(X, Y, "memo")[1].calls
            recursive = lcs_instrumented(X, Y, "recursive")[1].calls
            ratios.append(recursive / memo)
        assert ratios == sorted(ratios)
        assert ratios[-1] > 4 * ratios[0]

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_instrumented_length_matches_the_plain_function(self, variant: str):
        """Instrumentation must not change the answer it is measuring."""
        for X, Y in BATTERY[:20]:
            length, _counter = lcs_instrumented(X, Y, variant)
            assert length == VARIANTS[variant](X, Y), f"{variant}: {X!r} vs {Y!r}"

    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_each_run_gets_a_fresh_counter(self, variant: str):
        """Counters are created per call, so two runs cannot pool their totals."""
        first = lcs_instrumented("ABC", "xyz", variant)[1]
        second = lcs_instrumented("ABC", "xyz", variant)[1]
        assert first is not second
        assert first.calls == second.calls
        assert first.depth == 0 and second.depth == 0

    def test_empty_inputs_still_report_a_frame(self):
        """Documented behaviour: even nothing costs one visit.

        The top-down forms enter once and hit the base case; the table has
        a single corner cell. None of the three reports zero.
        """
        for variant in VARIANT_NAMES:
            length, counter = lcs_instrumented("", "", variant)
            assert length == 0
            assert counter.calls == 1
            assert counter.max_depth == 1


# ----------------------------------------------------------------------
# Contracts
# ----------------------------------------------------------------------
class TestContracts:
    """Type checking, the variant registry, and the unknown-variant error."""

    #: The non-string arguments the module is required to reject, with the
    #: type name its message is required to quote.
    NON_STRINGS: List[Tuple[Any, str]] = [
        (5, "int"),
        (None, "NoneType"),
        (["A", "B"], "list"),
        (3.5, "float"),
        (b"AB", "bytes"),
        (("A", "B"), "tuple"),
    ]

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    @pytest.mark.parametrize(
        "bad, type_name",
        NON_STRINGS,
        ids=[name for _value, name in NON_STRINGS],
    )
    def test_non_string_first_argument_raises_type_error(
        self, func, bad: Any, type_name: str
    ):
        with pytest.raises(TypeError) as error:
            func(bad, "ABC")
        assert type_name in str(error.value) and "X" in str(error.value)

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    @pytest.mark.parametrize(
        "bad, type_name",
        NON_STRINGS,
        ids=[name for _value, name in NON_STRINGS],
    )
    def test_non_string_second_argument_raises_type_error(
        self, func, bad: Any, type_name: str
    ):
        with pytest.raises(TypeError) as error:
            func("ABC", bad)
        assert type_name in str(error.value) and "Y" in str(error.value)

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_the_first_bad_argument_is_the_one_reported(self, func):
        """X is checked before Y, so a doubly wrong call names X."""
        with pytest.raises(TypeError, match="for X"):
            func(1, 2)

    @pytest.mark.parametrize("func", VARIANT_FUNCS, ids=VARIANT_NAMES)
    def test_the_failing_function_names_itself(self, func):
        with pytest.raises(TypeError, match=func.__name__):
            func(None, "ABC")

    @pytest.mark.parametrize(
        "bad, type_name",
        NON_STRINGS,
        ids=[name for _value, name in NON_STRINGS],
    )
    def test_reconstruct_and_instrumented_check_types_too(
        self, bad: Any, type_name: str
    ):
        with pytest.raises(TypeError, match=type_name):
            lcs_reconstruct(bad, "ABC")
        with pytest.raises(TypeError, match=type_name):
            lcs_reconstruct("ABC", bad)
        with pytest.raises(TypeError, match=type_name):
            lcs_instrumented(bad, "ABC", "tab")
        with pytest.raises(TypeError, match=type_name):
            lcs_instrumented("ABC", bad, "tab")

    @pytest.mark.parametrize(
        "variant",
        ["quadratic", "", "TAB", "Memo", "recursion", "rolling", None, 7],
        ids=[
            "unknown-word",
            "empty",
            "wrong-case-tab",
            "wrong-case-memo",
            "near-miss",
            "other-module",
            "none",
            "integer",
        ],
    )
    def test_unknown_variant_raises_value_error(self, variant: Any):
        """Variant names are matched exactly, including their case."""
        with pytest.raises(ValueError, match="unknown variant"):
            lcs_instrumented("AB", "BA", variant)

    def test_the_error_message_lists_the_legal_variants(self):
        with pytest.raises(ValueError) as error:
            lcs_instrumented("AB", "BA", "quadratic")
        message = str(error.value)
        for name in VARIANT_NAMES:
            assert repr(name) in message or name in message

    def test_bad_strings_are_reported_before_a_bad_variant(self):
        """Both arguments are wrong; the type check runs first.

        Documented in the order of the function body rather than chosen
        here, and worth pinning because a caller debugging a typo in the
        variant name would otherwise be told about it only on the second
        attempt.
        """
        with pytest.raises(TypeError):
            lcs_instrumented(7, "BA", "quadratic")

    def test_variants_registry_matches_the_public_functions(self):
        assert VARIANTS == {
            "recursive": lcs_recursive,
            "memo": lcs_memo,
            "tab": lcs_tab,
        }

    def test_string_subclasses_are_accepted(self):
        """isinstance, not type equality, so a str subclass is still a str."""

        class Sequence(str):
            pass

        assert lcs_tab(Sequence("AGGTAB"), "GXTXAYB") == 4
        assert lcs_reconstruct("AGGTAB", Sequence("GXTXAYB")) == "GTAB"

    def test_return_types(self):
        """Lengths are ints and reconstruction is a plain str, not a list."""
        assert isinstance(lcs_tab("AB", "AB"), int)
        assert isinstance(lcs_memo("AB", "AB"), int)
        assert isinstance(lcs_recursive("AB", "AB"), int)
        answer = lcs_reconstruct("AB", "AB")
        assert isinstance(answer, str) and type(answer) is str
        length, counter = lcs_instrumented("AB", "AB", "tab")
        assert isinstance(length, int)
        assert isinstance(counter.calls, int) and isinstance(counter.max_depth, int)
