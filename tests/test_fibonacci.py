"""Tests for the four Fibonacci variants and their instrumentation.

Fibonacci is easy to get right and easy to test badly. Every variant in
:mod:`src.dp.fibonacci` computes the same sequence, so checking them
against each other proves only that they agree, not that they agree on
the right numbers - three copies of one off-by-one error would pass such
a suite unanimously. Two kinds of independent witness are used instead:

* **Two oracles written here in the test file.** ``series_fibonacci``
  accumulates the whole sequence in a list, and ``doubling_fibonacci``
  jumps from F(k) to F(2k) by the fast-doubling identities. Neither is a
  rolling pair and neither is a recurrence on a memo table, so neither
  shares a mechanism with anything under test. They are checked against
  each other first, so the rest of the file rests on something.
* **The hand-known values** at n = 0, 1, 10, 20, 30, 35, 40 and 45,
  which were not produced by any code in this repository.

The call counts get the same treatment. :func:`naive_call_count` returns
2*F(n+1) - 1 from a closed form, and the closed form is exactly the sort
of algebra that can be quietly wrong, so it is checked against
``calls_by_recurrence`` - C(0) = C(1) = 1, C(n) = 1 + C(n-1) + C(n-2)
summed iteratively here in the test file - and then against the
instrumented recursion actually counting its own calls at sizes small
enough to run.

Three claims in the module docstring are behavioural rather than
numerical, and each gets a test that could fail:

* memoization actually reads its table. A memo that is written but never
  read still returns correct answers; only the call count exposes it, so
  ``fib_counts(30, "memo")`` being 59 rather than 2,692,537 is the proof.
* the memo default is ``None`` and not ``{}``. A shared mutable default
  is invisible in the return values, so the table handed to the recursion
  is captured directly and two successive calls are shown to receive two
  different empty dicts.
* tabulation never nests. ``max_depth`` of 1 is the space bound seen from
  the stack, and it is checked at every size rather than at one.

Nothing here calls ``fib_naive`` above n = 25: this file runs on every
commit, and the whole point of the module is that the naive variant
cannot be run at the sizes the others reach.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Tuple

import pytest

import src.dp.fibonacci as fibonacci_module
from src.dp.fibonacci import (
    VARIANTS,
    fib_counts,
    fib_instrumented,
    fib_lru,
    fib_memo,
    fib_naive,
    fib_tab,
    naive_call_count,
)
from src.utils.timer import CallCounter

#: The largest index at which the exponential variant may be run. 25 costs
#: 242,785 calls; 30 would cost 2.7 million and 35 would cost 30 million.
SAFE_NAIVE_LIMIT = 25

#: Indices whose Fibonacci numbers are stated in the assignment and in the
#: module docstring. None of these came from running this repository.
KNOWN_VALUES: List[Tuple[int, int]] = [
    (0, 0),
    (1, 1),
    (10, 55),
    (20, 6765),
    (30, 832040),
    (35, 9227465),
    (40, 102334155),
    (45, 1134903170),
]

#: The naive call totals the Week 5 report quotes. The last two are the
#: reason naive_call_count exists: 331 million and 3.67 billion calls.
KNOWN_NAIVE_CALLS: List[Tuple[int, int]] = [
    (10, 177),
    (20, 21891),
    (30, 2692537),
    (35, 29860703),
    (40, 331160281),
    (45, 3672623805),
]

#: The three graded variants, in a stable order, for parametrising the
#: shared contract over all of them at once.
VARIANT_ITEMS: List[Tuple[str, Callable[[int], int]]] = sorted(VARIANTS.items())
VARIANT_NAMES: List[str] = [name for name, _func in VARIANT_ITEMS]
VARIANT_FUNCTIONS: List[Callable[[int], int]] = [func for _name, func in VARIANT_ITEMS]

#: A seeded battery of sizes the naive variant can afford. Every index up
#: to 20 plus three drawn from 21..25, so the battery is reproducible but
#: not hand-picked to be convenient.
AGREEMENT_SIZES: List[int] = sorted(
    set(range(21)) | set(random.Random(5300).sample(range(21, SAFE_NAIVE_LIMIT + 1), 3))
)


# ----------------------------------------------------------------------
# Independent oracles
# ----------------------------------------------------------------------
def series_fibonacci(n: int) -> int:
    """F(n) by accumulating the whole sequence in a list.

    The definition written out with nothing thrown away: keep appending
    the sum of the last two until the list is long enough, then index it.
    It is wasteful on purpose. ``fib_tab`` gets its O(1) space from
    knowing that only the last two values matter, and an oracle that made
    the same assumption would be the same code twice.

    Args:
        n: A non-negative index.

    Returns:
        The nth Fibonacci number, with F(0) = 0 and F(1) = 1.
    """
    series = [0, 1]
    while len(series) <= n:
        series.append(series[-1] + series[-2])
    return series[n]


def doubling_fibonacci(n: int) -> int:
    """F(n) by fast doubling, which never walks the sequence at all.

    Uses F(2k) = F(k) * (2*F(k+1) - F(k)) and F(2k+1) = F(k)^2 + F(k+1)^2,
    reading the bits of n from the top down. It touches O(log n) indices
    and none of the ones in between, so it shares no loop, no table and no
    recurrence step with any variant under test. Where this and
    ``series_fibonacci`` agree, the number is right rather than merely
    consistent.

    Args:
        n: A non-negative index.

    Returns:
        The nth Fibonacci number.
    """
    current, following = 0, 1  # F(0), F(1)
    for bit in bin(n)[2:]:
        doubled = current * (2 * following - current)
        doubled_plus_one = current * current + following * following
        if bit == "0":
            current, following = doubled, doubled_plus_one
        else:
            current, following = doubled_plus_one, doubled + doubled_plus_one
    return current


def calls_by_recurrence(n: int) -> int:
    """How many calls the naive recursion makes, counted without algebra.

    The call tree has one node per invocation, so its size obeys
    C(0) = C(1) = 1 and C(n) = 1 + C(n-1) + C(n-2). Summing that
    iteratively is an independent check on the closed form
    :func:`naive_call_count` uses: if 2*F(n+1) - 1 were off by one or
    shifted by an index, this would disagree at every n above 1.

    Args:
        n: A non-negative index.

    Returns:
        The exact number of ``fib_naive`` invocations F(n) would require.
    """
    previous, current = 1, 1  # C(0), C(1)
    if n < 2:
        return 1
    for _ in range(2, n + 1):
        previous, current = current, 1 + current + previous
    return current


# ----------------------------------------------------------------------
# The oracles themselves
# ----------------------------------------------------------------------
class TestOraclesAgreeWithEachOther:
    """The witnesses are cross-examined before anything is tried on them."""

    def test_both_oracles_agree_on_a_long_run(self):
        assert [series_fibonacci(i) for i in range(200)] == [
            doubling_fibonacci(i) for i in range(200)
        ]

    @pytest.mark.parametrize("n, expected", KNOWN_VALUES, ids=str)
    def test_oracles_reproduce_the_hand_known_values(self, n: int, expected: int):
        assert series_fibonacci(n) == expected
        assert doubling_fibonacci(n) == expected

    def test_oracles_satisfy_the_recurrence_they_are_meant_to_encode(self):
        for n in range(2, 60):
            assert series_fibonacci(n) == series_fibonacci(n - 1) + series_fibonacci(
                n - 2
            )

    @pytest.mark.parametrize("n, expected", KNOWN_NAIVE_CALLS, ids=str)
    def test_call_count_oracle_reproduces_the_quoted_totals(self, n: int, expected: int):
        assert calls_by_recurrence(n) == expected


# ----------------------------------------------------------------------
# Agreement
# ----------------------------------------------------------------------
class TestAllVariantsAgree:
    """One sequence, four implementations, two oracles that share nothing."""

    @pytest.mark.parametrize("n", AGREEMENT_SIZES, ids=str)
    def test_three_variants_agree_with_each_other(self, n: int):
        assert fib_naive(n) == fib_memo(n) == fib_tab(n)

    @pytest.mark.parametrize("n", AGREEMENT_SIZES, ids=str)
    def test_three_variants_agree_with_both_oracles(self, n: int):
        expected = series_fibonacci(n)
        assert doubling_fibonacci(n) == expected, "oracles disagree, not the module"
        assert fib_naive(n) == expected
        assert fib_memo(n) == expected
        assert fib_tab(n) == expected

    @pytest.mark.parametrize("n", AGREEMENT_SIZES, ids=str)
    def test_the_extra_lru_variant_agrees_too(self, n: int):
        assert fib_lru(n) == series_fibonacci(n)

    def test_the_opening_stretch_is_the_sequence_everyone_recognises(self):
        assert [fib_tab(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert [fib_memo(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert [fib_naive(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_every_variant_satisfies_the_recurrence_it_implements(self):
        for n in range(2, SAFE_NAIVE_LIMIT + 1):
            assert fib_naive(n) == fib_naive(n - 1) + fib_naive(n - 2)
            assert fib_memo(n) == fib_memo(n - 1) + fib_memo(n - 2)
            assert fib_tab(n) == fib_tab(n - 1) + fib_tab(n - 2)

    def test_variants_registry_holds_the_three_graded_functions(self):
        """fib_lru is the extra, so the benchmark must not find it here."""
        assert VARIANTS == {"naive": fib_naive, "memo": fib_memo, "tab": fib_tab}
        assert fib_lru not in VARIANTS.values()

    @pytest.mark.parametrize("n", AGREEMENT_SIZES, ids=str)
    def test_the_registry_entries_compute_what_their_names_claim(self, n: int):
        expected = series_fibonacci(n)
        for name, function in VARIANT_ITEMS:
            assert function(n) == expected, name


# ----------------------------------------------------------------------
# The stated values, including the sizes only two variants can reach
# ----------------------------------------------------------------------
class TestKnownValues:
    """The eight indices whose answers were known before any code ran.

    ``fib_naive`` is absent above n = 25 by design. F(45) naively is
    3.67 billion calls, and a test suite that waited for it would not be
    run.
    """

    @pytest.mark.parametrize("n, expected", KNOWN_VALUES, ids=str)
    def test_fib_memo(self, n: int, expected: int):
        assert fib_memo(n) == expected

    @pytest.mark.parametrize("n, expected", KNOWN_VALUES, ids=str)
    def test_fib_tab(self, n: int, expected: int):
        assert fib_tab(n) == expected

    @pytest.mark.parametrize("n, expected", KNOWN_VALUES, ids=str)
    def test_fib_lru(self, n: int, expected: int):
        assert fib_lru(n) == expected

    @pytest.mark.parametrize(
        "n, expected",
        [(n, value) for n, value in KNOWN_VALUES if n <= SAFE_NAIVE_LIMIT],
        ids=str,
    )
    def test_fib_naive_where_it_can_still_be_afforded(self, n: int, expected: int):
        assert fib_naive(n) == expected

    def test_base_cases_are_zero_and_one_not_one_and_one(self):
        """The commonest indexing error in the whole subject."""
        for variant in VARIANT_FUNCTIONS + [fib_lru]:
            assert variant(0) == 0
            assert variant(1) == 1
            assert variant(2) == 1

    def test_tabulation_reaches_sizes_that_would_exhaust_the_stack(self):
        """No recursion means no recursion limit, which is the O(1) claim."""
        assert fib_tab(200) == series_fibonacci(200)
        assert len(str(fib_tab(10_000))) == 2090

    def test_memoization_reaches_sizes_the_naive_variant_never_could(self):
        assert fib_memo(90) == 2880067194370816120
        assert fib_memo(90) == doubling_fibonacci(90)


# ----------------------------------------------------------------------
# naive_call_count: the closed form
# ----------------------------------------------------------------------
class TestNaiveCallCount:
    """2*F(n+1) - 1, checked three ways and timed once."""

    @pytest.mark.parametrize("n, expected", KNOWN_NAIVE_CALLS, ids=str)
    def test_quoted_totals(self, n: int, expected: int):
        assert naive_call_count(n) == expected

    def test_base_cases_cost_one_call_each(self):
        assert naive_call_count(0) == 1
        assert naive_call_count(1) == 1

    def test_closed_form_matches_the_recurrence_oracle(self):
        """If the algebra were shifted by an index this fails from n = 2."""
        for n in range(80):
            assert naive_call_count(n) == calls_by_recurrence(n), n

    def test_closed_form_is_literally_two_f_of_n_plus_one_minus_one(self):
        for n in range(60):
            assert naive_call_count(n) == 2 * series_fibonacci(n + 1) - 1, n

    @pytest.mark.parametrize("n", list(range(SAFE_NAIVE_LIMIT + 1)), ids=str)
    def test_the_instrumented_recursion_counts_exactly_what_was_predicted(self, n: int):
        """The counter and the closed form, agreeing call for call."""
        value, calls = fib_counts(n, "naive")
        assert value == series_fibonacci(n)
        assert calls == naive_call_count(n)

    def test_it_is_instant_at_forty_five_which_is_why_it_exists(self):
        """Every count up to 45 in well under a second, including 3.67 billion.

        Running the recursion for the last of these would take hours, so
        a bound this loose is still decisive: it can only pass if nothing
        recursed.
        """
        start = time.perf_counter()
        counts = [naive_call_count(n) for n in range(46)]
        elapsed = time.perf_counter() - start
        assert counts[45] == 3672623805
        assert elapsed < 0.5, f"naive_call_count took {elapsed:.3f}s for 46 sizes"

    def test_it_never_calls_the_naive_variant_at_all(self, monkeypatch):
        """The timing claim without the clock: fib_naive is simply unused."""

        def explode(n: int) -> int:
            raise AssertionError("naive_call_count() ran the recursion")

        monkeypatch.setattr(fibonacci_module, "fib_naive", explode)
        assert naive_call_count(45) == 3672623805

    def test_it_grows_by_the_golden_ratio_per_step(self):
        """Theta(phi^n): the step ratio settles on 1.618, not on 2."""
        for n in range(20, 45):
            ratio = naive_call_count(n + 1) / naive_call_count(n)
            assert 1.61 < ratio < 1.62, n


# ----------------------------------------------------------------------
# Memoization has to actually read its table
# ----------------------------------------------------------------------
class TestMemoizationActuallyMemoizes:
    """A memo written but never read returns right answers at wrong cost.

    Nothing in the return value of ``fib_memo`` can distinguish a working
    memo table from a decorative one; both give 832040 for F(30). Only the
    call count separates them, and it separates them by five orders of
    magnitude.
    """

    def test_thirty_costs_fewer_than_a_hundred_calls(self):
        value, calls = fib_counts(30, "memo")
        assert value == 832040
        assert calls < 100, "memo table is being written but not read"
        assert calls == 59

    def test_the_same_size_costs_2_692_537_calls_without_a_table(self):
        """The contrast that makes the previous number mean something."""
        assert naive_call_count(30) == 2692537
        assert fib_counts(30, "memo")[1] * 40_000 < naive_call_count(30)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20, 30, 50, 100, 200], ids=str)
    def test_the_count_is_two_n_minus_one_at_every_size(self, n: int):
        """One call down the spine and one cache hit per level, exactly."""
        value, calls = fib_counts(n, "memo")
        assert value == series_fibonacci(n)
        assert calls == 2 * n - 1

    def test_the_count_grows_linearly_and_not_exponentially(self):
        """Doubling n doubles the work. Doubling it naively squares it."""
        sizes = [25, 50, 100, 200, 400]
        counts = [fib_counts(n, "memo")[1] for n in sizes]
        for smaller, larger in zip(counts, counts[1:]):
            assert 1.9 < larger / smaller < 2.1
        # An exponential curve would have left this behind long ago.
        assert counts[-1] < 10 * sizes[-1]

    def test_zero_is_the_one_size_where_the_formula_bends(self):
        """F(0) still costs one call, not minus one."""
        assert fib_counts(0, "memo") == (0, 1)

    def test_tabulation_evaluates_each_subproblem_exactly_once(self):
        """For the loop, "calls" reads as "subproblems", and there are n + 1."""
        for n in range(12):
            value, calls = fib_counts(n, "tab")
            assert value == series_fibonacci(n)
            assert calls == n + 1

    def test_the_three_variants_priced_side_by_side(self):
        """The headline table of the whole module, at the size it is quoted."""
        assert fib_counts(10, "naive") == (55, 177)
        assert fib_counts(10, "memo") == (55, 19)
        assert fib_counts(10, "tab") == (55, 11)


# ----------------------------------------------------------------------
# The caller-supplied memo
# ----------------------------------------------------------------------
class TestSuppliedMemo:
    """The table is an argument, is honoured, is filled, and is never shared."""

    def test_the_default_is_none_and_not_an_empty_dict(self):
        assert fib_memo.__defaults__ == (None,)

    def test_a_supplied_table_is_filled_in_place(self):
        table: Dict[int, int] = {}
        assert fib_memo(10, table) == 55
        assert sorted(table) == list(range(11))
        assert all(table[i] == series_fibonacci(i) for i in table)

    def test_a_warm_table_is_extended_rather_than_rebuilt(self):
        table: Dict[int, int] = {}
        fib_memo(10, table)
        assert fib_memo(12, table) == 144
        assert (table[11], table[12]) == (89, 144)
        assert sorted(table) == list(range(13))

    def test_entries_already_present_are_trusted_and_not_recomputed(self):
        """Poisoning one cell proves the table is read, not just written."""
        assert fib_memo(10, {10: -1}) == -1
        assert fib_memo(8, {6: 0, 7: 0}) == 0

    def test_the_table_may_be_passed_by_keyword(self):
        table: Dict[int, int] = {}
        assert fib_memo(9, memo=table) == 34
        assert table[9] == 34

    def test_a_poisoned_table_does_not_leak_into_the_next_default_call(self):
        """Call it twice: the second call must not see the first one's table."""
        assert fib_memo(10, {10: -1}) == -1
        assert fib_memo(10) == 55

    def test_two_default_calls_receive_two_different_empty_tables(self, monkeypatch):
        """The mutable-default bug is invisible in the answers, so look at the dict.

        The recursion is wrapped so the table it is handed can be captured
        at the moment of the call. A shared ``{}`` default would show up
        here as the same object twice, with the second call starting warm.
        The wrapper recurses onto itself, so only the outermost frame of
        each run is recorded.
        """
        real_recursion = fibonacci_module._fib_memo
        handed: List[Tuple[Dict[int, int], int]] = []
        nesting = {"level": 0}

        def capturing(n: int, memo: Dict[int, int]) -> int:
            if nesting["level"] == 0:
                handed.append((memo, len(memo)))  # the tuple keeps the dict alive
            nesting["level"] += 1
            try:
                return real_recursion(n, memo)
            finally:
                nesting["level"] -= 1

        monkeypatch.setattr(fibonacci_module, "_fib_memo", capturing)
        assert fib_memo(20) == 6765
        assert fib_memo(20) == 6765

        assert len(handed) == 2
        first_table, first_size = handed[0]
        second_table, second_size = handed[1]
        assert first_size == 0 and second_size == 0, "a table arrived warm"
        assert first_table is not second_table, "the same dict was reused"

    def test_repeated_default_calls_keep_giving_the_same_answer(self):
        assert [fib_memo(30) for _ in range(5)] == [832040] * 5

    @pytest.mark.parametrize(
        "bad", [[], "table", 5, set(), (1, 2), 0.0], ids=lambda b: type(b).__name__
    )
    def test_a_memo_that_is_not_a_dict_is_refused(self, bad):
        with pytest.raises(TypeError, match="expects a dict memo or None"):
            fib_memo(10, bad)

    def test_the_counted_run_always_starts_from_a_cold_table(self):
        """The reported figure has to be a property of n, not of history."""
        fib_memo(40)
        assert fib_counts(40, "memo")[1] == 2 * 40 - 1
        assert fib_counts(40, "memo")[1] == fib_counts(40, "memo")[1]


# ----------------------------------------------------------------------
# Depth, which is the space bound seen from the stack
# ----------------------------------------------------------------------
class TestInstrumentedDepth:
    """Both recursive variants nest to n. Tabulation nests to 1, always."""

    @pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 25, 100, 500], ids=str)
    def test_tabulation_never_nests(self, n: int):
        value, counter = fib_instrumented(n, "tab")
        assert value == series_fibonacci(n)
        assert counter.max_depth == 1

    @pytest.mark.parametrize("variant", ["naive", "memo"], ids=str)
    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20], ids=str)
    def test_the_recursive_variants_nest_to_exactly_n(self, variant: str, n: int):
        _value, counter = fib_instrumented(n, variant)
        assert counter.max_depth == n

    @pytest.mark.parametrize("variant", ["naive", "memo"], ids=str)
    def test_depth_grows_strictly_with_n(self, variant: str):
        depths = [
            fib_instrumented(n, variant)[1].max_depth
            for n in range(1, SAFE_NAIVE_LIMIT + 1)
        ]
        assert depths == sorted(depths)
        assert all(later > earlier for earlier, later in zip(depths, depths[1:]))

    @pytest.mark.parametrize("variant", VARIANT_NAMES, ids=str)
    def test_zero_still_costs_one_frame(self, variant: str):
        value, counter = fib_instrumented(0, variant)
        assert (value, counter.calls, counter.max_depth) == (0, 1, 1)

    @pytest.mark.parametrize("variant", VARIANT_NAMES, ids=str)
    def test_every_frame_entered_is_left_again(self, variant: str):
        """depth back to 0 means the enter/leave pairs balanced."""
        _value, counter = fib_instrumented(15, variant)
        assert counter.depth == 0

    def test_the_three_depths_at_the_size_the_report_quotes(self):
        assert fib_instrumented(10, "naive")[1].max_depth == 10
        assert fib_instrumented(10, "memo")[1].max_depth == 10
        assert fib_instrumented(10, "tab")[1].max_depth == 1

    def test_the_counter_returned_is_a_real_call_counter(self):
        _value, counter = fib_instrumented(10, "memo")
        assert isinstance(counter, CallCounter)
        assert (counter.calls, counter.max_depth, counter.depth) == (19, 10, 0)

    def test_each_run_gets_its_own_counter(self):
        """Module-level tallies would let two benchmarks corrupt each other."""
        _first_value, first = fib_instrumented(10, "naive")
        _second_value, second = fib_instrumented(10, "memo")
        assert first is not second
        assert (first.calls, second.calls) == (177, 19)

    def test_fib_counts_is_fib_instrumented_with_the_depth_dropped(self):
        for n in (0, 1, 7, 15):
            for variant in VARIANT_NAMES:
                value, counter = fib_instrumented(n, variant)
                assert fib_counts(n, variant) == (value, counter.calls)


# ----------------------------------------------------------------------
# The shared contract
# ----------------------------------------------------------------------
class TestContracts:
    """One validation rule, enforced identically by every entry point."""

    @pytest.mark.parametrize("variant", VARIANT_FUNCTIONS, ids=VARIANT_NAMES)
    @pytest.mark.parametrize(
        "bad", ["13", "", 3.0, 0.5, True, False, None, [10], complex(1, 0)],
        ids=lambda b: f"{type(b).__name__}_{b!r}",
    )
    def test_type_error_on_anything_that_is_not_an_int(self, variant, bad):
        with pytest.raises(TypeError, match="expects an int"):
            variant(bad)

    @pytest.mark.parametrize("variant", VARIANT_FUNCTIONS, ids=VARIANT_NAMES)
    @pytest.mark.parametrize("bad", [-1, -2, -10, -1000], ids=str)
    def test_value_error_on_a_negative_index(self, variant, bad):
        with pytest.raises(ValueError, match="expects n >= 0"):
            variant(bad)

    @pytest.mark.parametrize("variant", VARIANT_FUNCTIONS, ids=VARIANT_NAMES)
    def test_the_message_names_the_function_and_the_offending_type(self, variant):
        with pytest.raises(TypeError) as caught:
            variant(3.0)
        assert variant.__name__ in str(caught.value)
        assert "float" in str(caught.value)

    @pytest.mark.parametrize("variant", VARIANT_FUNCTIONS, ids=VARIANT_NAMES)
    def test_bool_is_refused_although_python_would_accept_it_as_an_int(self, variant):
        """``fib(True)`` would silently answer F(1); an index arriving as a
        flag is a caller bug worth hearing about."""
        with pytest.raises(TypeError, match="got bool"):
            variant(True)
        assert variant(1) == 1, "the int 1 is of course still fine"

    @pytest.mark.parametrize(
        "entry_point",
        [fib_lru, naive_call_count],
        ids=["fib_lru", "naive_call_count"],
    )
    def test_the_extras_keep_the_same_contract(self, entry_point):
        for bad in ("13", 3.0, True):
            with pytest.raises(TypeError, match="expects an int"):
                entry_point(bad)
        with pytest.raises(ValueError, match="expects n >= 0"):
            entry_point(-1)

    @pytest.mark.parametrize(
        "entry_point", [fib_counts, fib_instrumented], ids=["counts", "instrumented"]
    )
    def test_the_counting_entry_points_validate_n_as_well(self, entry_point):
        with pytest.raises(TypeError, match="expects an int"):
            entry_point(3.0, "tab")
        with pytest.raises(ValueError, match="expects n >= 0"):
            entry_point(-1, "tab")

    def test_zero_is_valid_and_is_not_confused_with_a_falsy_argument(self):
        for variant in VARIANT_FUNCTIONS + [fib_lru]:
            assert variant(0) == 0
        assert naive_call_count(0) == 1


# ----------------------------------------------------------------------
# Variant dispatch
# ----------------------------------------------------------------------
class TestVariantNames:
    """A name that is not one of the three is a caller mistake, not a guess."""

    @pytest.mark.parametrize(
        "entry_point", [fib_counts, fib_instrumented], ids=["counts", "instrumented"]
    )
    @pytest.mark.parametrize(
        "bad", ["lru", "iterative", "Tab", "", "naive "], ids=repr
    )
    def test_an_unknown_name_raises_value_error(self, entry_point, bad: str):
        with pytest.raises(ValueError, match="unknown variant"):
            entry_point(10, bad)

    @pytest.mark.parametrize(
        "entry_point", [fib_counts, fib_instrumented], ids=["counts", "instrumented"]
    )
    @pytest.mark.parametrize(
        "bad", [None, 5, ["tab"], b"tab", {"tab": 1}], ids=lambda b: type(b).__name__
    )
    def test_a_non_string_is_refused_the_same_way(self, entry_point, bad):
        """ValueError, not an unrelated TypeError out of a dict lookup.

        ``["tab"]`` is unhashable, so an unguarded lookup would raise
        TypeError and the caller would be told the wrong thing.
        """
        with pytest.raises(ValueError, match="unknown variant"):
            entry_point(10, bad)

    @pytest.mark.parametrize(
        "entry_point, name",
        [(fib_counts, "fib_counts"), (fib_instrumented, "fib_instrumented")],
        ids=["counts", "instrumented"],
    )
    def test_the_message_names_the_caller_and_lists_the_alternatives(
        self, entry_point, name: str
    ):
        with pytest.raises(ValueError) as caught:
            entry_point(10, "quick")
        message = str(caught.value)
        assert message.startswith(f"{name}() got unknown variant 'quick'")
        assert "'memo', 'naive', 'tab'" in message

    @pytest.mark.parametrize("variant", VARIANT_NAMES, ids=str)
    def test_every_name_in_variants_is_accepted_by_both_entry_points(self, variant: str):
        assert fib_counts(12, variant)[0] == 144
        assert fib_instrumented(12, variant)[0] == 144


# ----------------------------------------------------------------------
# The extra: functools.lru_cache
# ----------------------------------------------------------------------
class TestLruExtra:
    """The decorator version, and the cache state a benchmark has to clear."""

    def test_it_agrees_with_the_hand_written_memo_throughout(self):
        assert all(fib_lru(i) == fib_memo(i) for i in range(60))

    def test_the_cache_survives_between_calls(self):
        fib_lru.cache_clear()
        assert fib_lru.cache_info().currsize == 0
        fib_lru(45)
        assert fib_lru.cache_info().currsize == 46
        fib_lru.cache_clear()
        assert fib_lru.cache_info().currsize == 0

    def test_a_second_call_is_served_from_the_cache(self):
        fib_lru.cache_clear()
        fib_lru(30)
        misses_after_cold_run = fib_lru.cache_info().misses
        fib_lru(30)
        info = fib_lru.cache_info()
        assert info.misses == misses_after_cold_run, "a warm call still missed"
        assert info.hits >= 1
        fib_lru.cache_clear()

    def test_bool_is_rejected_before_the_cache_is_consulted(self):
        """hash(True) == hash(1), so a cache in front of the check would
        answer fib_lru(True) out of the entry stored for fib_lru(1)."""
        fib_lru.cache_clear()
        assert fib_lru(1) == 1
        with pytest.raises(TypeError, match="got bool"):
            fib_lru(True)
        fib_lru.cache_clear()
