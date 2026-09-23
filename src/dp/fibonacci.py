"""Fibonacci four ways, and the instrumentation that prices the difference.

Fibonacci is the smallest problem that exhibits both conditions dynamic
programming needs. It has *optimal substructure* - F(n) is built from
F(n-1) and F(n-2) and nothing else - and it has *overlapping
subproblems*, because the two branches of that recurrence ask for most of
the same values. Plain recursion recomputes them; memoization and
tabulation each remember them, and differ only in who decides the order
the subproblems are solved in.

That one difference is the whole module. Top-down memoization asks for a
subproblem when it needs it and discovers the order lazily at run time,
paying a call frame and a dict hash per subproblem. Bottom-up tabulation
fixes the order in advance, which costs nothing here because every
subproblem is needed anyway - and knowing the order is exactly what lets
it throw away everything except the last two values.

=============== ============ =========== =========================
Function        Time         Space       Notes
=============== ============ =========== =========================
fib_naive       O(2^n)       O(n) stack  2*F(n+1) - 1 calls
fib_memo        O(n)         O(n)        hand-written dict, O(n) stack
fib_tab         O(n)         O(1)        rolling pair, depth 1
fib_lru         O(n)         O(n)        extra, for comparison only
=============== ============ =========== =========================

Separation of concerns
----------------------
The four plain functions above carry **no instrumentation at all**: no
counter argument, no module-level tally, no bookkeeping in the hot loop.
They are what the Week 5 benchmark times, and a counter inside the
recursion would corrupt the measurement it is supposed to support.

Counting happens only in :func:`fib_counts` and :func:`fib_instrumented`,
which run separate private helpers that thread a
:class:`~src.utils.timer.CallCounter` through the recursion explicitly.
The counter is created per call and handed back to the caller, so two
benchmarks running in the same process cannot corrupt each other's
counts - which a module-level global would allow.

Counting the naive variant is only affordable up to about n = 30. Beyond
that, :func:`naive_call_count` gives the exact figure from the closed form
2*F(n+1) - 1 without making a single recursive call, which is how the
benchmark reports n = 40 and n = 45 instead of waiting out 3.67 billion
calls.

Recursion depth
---------------
Both recursive variants nest to depth n, so n beyond roughly 990 hits
CPython's default recursion limit of 1000. This module does **not** raise
that limit; the caller owns that decision, and the benchmark makes it
explicitly.

Examples:
    All variants agree, and the call counts show why only two of them
    scale:

    >>> fib_naive(10) == fib_memo(10) == fib_tab(10) == 55
    True
    >>> [fib_counts(10, name)[1] for name in ("naive", "memo", "tab")]
    [177, 19, 11]

    The naive count is available for sizes nobody can afford to run:

    >>> naive_call_count(45)
    3672623805

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from src.utils.timer import CallCounter

__all__ = [
    "fib_naive",
    "fib_memo",
    "fib_tab",
    "fib_lru",
    "fib_counts",
    "fib_instrumented",
    "naive_call_count",
    "VARIANTS",
]


def _require_natural_int(n: Any, func_name: str) -> None:
    """Raise unless ``n`` is a non-negative ``int``.

    Every public function in this module shares one contract, so the check
    lives in one place: the message is then identical everywhere and each
    algorithm body stays free of validation.

    ``bool`` is rejected even though it is a subclass of ``int``. Python
    would happily evaluate ``fib_naive(True)`` as ``fib_naive(1)``, and a
    Fibonacci index that arrived as a flag is a caller bug worth hearing
    about rather than a value worth guessing at.

    Args:
        n: The object to validate as a Fibonacci index.
        func_name: Name of the calling function, used in the message.

    Returns:
        None. This helper is called for its exception, not its value.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative. Fibonacci is defined here on the
            non-negative integers only.

    Time Complexity:
        O(1) - two type checks and one comparison.

    Space Complexity:
        O(1).

    Examples:
        >>> _require_natural_int(10, "demo") is None
        True
        >>> _require_natural_int(0, "demo") is None
        True
        >>> _require_natural_int(3.0, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects an int, got float
        >>> _require_natural_int(True, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects an int, got bool
        >>> _require_natural_int(-1, "demo")
        Traceback (most recent call last):
            ...
        ValueError: demo() expects n >= 0, got -1
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{func_name}() expects an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"{func_name}() expects n >= 0, got {n}")


def _require_variant(
    variant: Any,
    table: Dict[str, Any],
    func_name: str,
) -> Any:
    """Look up ``variant`` in ``table``, or raise :class:`ValueError`.

    The two counting entry points dispatch on a string so that the
    benchmark can loop over variant names read from a list rather than
    hard-coding three call sites. A name that is not in the table is a
    caller mistake, and the message names the alternatives so the caller
    does not have to read the source to find them.

    Anything that is not a known name raises, including objects that are
    not strings at all. Checking the type first also keeps an unhashable
    argument from raising an unrelated :class:`TypeError` out of the
    dictionary lookup.

    Args:
        variant: The name to look up.
        table: Mapping of variant name to the callable implementing it.
        func_name: Name of the calling function, used in the message.

    Returns:
        The callable ``table`` stores under ``variant``.

    Raises:
        ValueError: If ``variant`` is not a key of ``table``.

    Time Complexity:
        O(1) average for the lookup; O(k log k) for the k variant names in
        the message, which is only built on the failure path.

    Space Complexity:
        O(1) on success.

    Examples:
        >>> _require_variant("tab", VARIANTS, "demo") is fib_tab
        True
        >>> _require_variant("quick", VARIANTS, "demo")
        ... # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: demo() got unknown variant 'quick'; expected one of
        'memo', 'naive', 'tab'

        A non-string is refused the same way, rather than raising an
        unrelated TypeError out of the lookup:

        >>> _require_variant(["tab"], VARIANTS, "demo")
        ... # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: demo() got unknown variant ['tab']; expected one of
        'memo', 'naive', 'tab'
    """
    if isinstance(variant, str) and variant in table:
        return table[variant]

    known = ", ".join(repr(name) for name in sorted(table))
    raise ValueError(
        f"{func_name}() got unknown variant {variant!r}; expected one of {known}"
    )


def fib_naive(n: int) -> int:
    """Return the nth Fibonacci number by plain double recursion.

    The recurrence written out literally: F(0) = 0, F(1) = 1, and
    F(n) = F(n-1) + F(n-2) otherwise. Nothing is remembered between the
    two branches, so F(n-2) is computed once inside the F(n-1) subtree and
    again beside it, and the same duplication repeats all the way down.
    The result is a call tree whose size is itself Fibonacci: exactly
    2*F(n+1) - 1 calls, which is Theta(phi^n) and grows by roughly 1.618x
    per step.

    This is the control in the Week 5 benchmark. It is correct, it is the
    clearest statement of the recurrence, and it is unusable past about
    n = 40 - which is the point of measuring it.

    Args:
        n: The index to compute. Must be a non-negative ``int``.

    Returns:
        The nth Fibonacci number, with F(0) = 0 and F(1) = 1.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative.
        RecursionError: For ``n`` beyond CPython's recursion limit, since
            the deepest chain of calls is n frames long. Long before that
            limit matters the running time has already made the call
            impractical.

    Time Complexity:
        Best:    O(1)     - n of 0 or 1 returns from the base case
        Average: O(2^n)   - tight bound Theta(phi^n), phi = 1.618...
        Worst:   O(2^n)
        There is no data-dependent variation: the cost is fixed by n
        alone, so average and worst case are the same curve.

    Space Complexity:
        O(n) - no table is kept, but the call stack reaches depth n. The
        tree is explored depth first, so only one root-to-leaf path is
        live at a time.

    Examples:
        >>> fib_naive(0)
        0
        >>> fib_naive(1)
        1
        >>> fib_naive(10)
        55
        >>> [fib_naive(i) for i in range(10)]
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        >>> fib_naive(20)
        6765

        The cost of that last answer, exactly:

        >>> naive_call_count(20)
        21891

        The contract is the same for all four variants:

        >>> fib_naive(-1)
        Traceback (most recent call last):
            ...
        ValueError: fib_naive() expects n >= 0, got -1
        >>> fib_naive(3.0)
        Traceback (most recent call last):
            ...
        TypeError: fib_naive() expects an int, got float
        >>> fib_naive(True)
        Traceback (most recent call last):
            ...
        TypeError: fib_naive() expects an int, got bool
    """
    _require_natural_int(n, "fib_naive")
    return _fib_naive(n)


def _fib_naive(n: int) -> int:
    """Recurse on the Fibonacci recurrence with no validation and no cache.

    Split out from :func:`fib_naive` so that the argument check runs once,
    at the top, instead of on every one of the 2*F(n+1) - 1 calls. That
    matters here more than anywhere else in the module: this function is
    the benchmark's baseline, and validation inside the recursion would
    inflate the very number the report quotes.

    Args:
        n: A non-negative index, already validated by the caller.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(2^n), tightly Theta(phi^n).

    Space Complexity:
        O(n) of call stack.

    Examples:
        >>> _fib_naive(12)
        144
        >>> _fib_naive(0), _fib_naive(1)
        (0, 1)
    """
    if n < 2:
        return n
    return _fib_naive(n - 1) + _fib_naive(n - 2)


def fib_memo(n: int, memo: Optional[Dict[int, int]] = None) -> int:
    """Return the nth Fibonacci number top down, with an explicit dict memo.

    The same recurrence as :func:`fib_naive`, with one addition: before
    computing F(k) the function looks for it in a dictionary, and stores
    the answer there on the way back up. Every subproblem is therefore
    solved at most once, and the call tree collapses from exponential to a
    single spine of n calls with one cache hit hanging off each level -
    2n - 1 calls in total for n >= 1.

    The memo is a hand-written ``dict`` rather than
    :func:`functools.lru_cache` because writing the memoization is the
    exercise; the cached-decorator version lives in :func:`fib_lru` as a
    point of comparison.

    The default is ``None``, not ``{}``. A mutable default argument is
    evaluated once at function definition time, so a literal empty dict
    would be shared by every caller for the life of the process - one
    caller's table would silently answer another's questions, and the
    benchmark would time a dict lookup instead of an algorithm. Passing a
    memo in explicitly is fully supported, and it is honoured: entries
    already present are trusted and new ones are added, which is how a
    caller can amortise a series of related queries.

    Args:
        n: The index to compute. Must be a non-negative ``int``.
        memo: An optional existing table mapping index to Fibonacci
            number. Filled in place as a side effect. ``None``, the
            default, creates a fresh table for this call only.

    Returns:
        The nth Fibonacci number.

    Raises:
        TypeError: If ``n`` is not an ``int``, is a ``bool``, or if
            ``memo`` is neither ``None`` nor a ``dict``.
        ValueError: If ``n`` is negative.
        RecursionError: For ``n`` beyond CPython's recursion limit. The
            deepest chain is n frames, so this bites at roughly n = 990 on
            a default interpreter - the one size where memoization is
            worse off than tabulation.

    Time Complexity:
        Best:    O(1)     - n already present in a supplied memo
        Average: O(n)     - each of the n subproblems solved once
        Worst:   O(n)
        Each solved subproblem costs one call frame and two dict
        operations, which is where tabulation's constant-factor advantage
        comes from.

    Space Complexity:
        O(n) - n entries in the memo, plus n frames of call stack. Both
        terms are O(n), and neither can be reduced: top-down cannot know
        which entries it will still be asked for.

    Examples:
        >>> fib_memo(0), fib_memo(1)
        (0, 1)
        >>> fib_memo(10)
        55
        >>> fib_memo(45)
        1134903170

        Sizes the naive version cannot reach are routine here:

        >>> fib_memo(90)
        2880067194370816120

        Only 19 calls for what cost 177 naively:

        >>> fib_counts(10, "memo")
        (55, 19)

        A supplied memo is filled in place and reused on the next call:

        >>> table = {}
        >>> fib_memo(10, table)
        55
        >>> sorted(table)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> fib_memo(12, table)
        144
        >>> table[11], table[12]
        (89, 144)

        The default really is ``None``, so no table is shared between
        callers that did not ask to share one:

        >>> fib_memo.__defaults__
        (None,)

        >>> fib_memo(-3)
        Traceback (most recent call last):
            ...
        ValueError: fib_memo() expects n >= 0, got -3
        >>> fib_memo(10, memo=[])
        Traceback (most recent call last):
            ...
        TypeError: fib_memo() expects a dict memo or None, got list
    """
    _require_natural_int(n, "fib_memo")
    if memo is None:
        memo = {}
    elif not isinstance(memo, dict):
        raise TypeError(
            f"fib_memo() expects a dict memo or None, got {type(memo).__name__}"
        )
    return _fib_memo(n, memo)


def _fib_memo(n: int, memo: Dict[int, int]) -> int:
    """Recurse with a memo table, without revalidating the arguments.

    The recursion is kept out of :func:`fib_memo` for the same reason
    :func:`_fib_naive` is kept out of :func:`fib_naive`: the checks belong
    at the boundary, not in the loop being timed. The base cases are
    written into the table too, so that every index the recursion touches
    leaves exactly one entry behind and the memo's size is a faithful
    count of the subproblems solved.

    Args:
        n: A non-negative index, already validated by the caller.
        memo: The table to consult and extend. Modified in place.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(n) on a cold table, O(1) on a hit.

    Space Complexity:
        O(n) of table plus O(n) of call stack.

    Examples:
        >>> table = {}
        >>> _fib_memo(6, table)
        8
        >>> sorted(table.items())
        [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 5), (6, 8)]

        A pre-seeded entry is trusted rather than recomputed, which is
        what makes the table worth passing in:

        >>> _fib_memo(4, {4: 999})
        999
    """
    cached = memo.get(n)
    if cached is not None:
        return cached
    if n < 2:
        memo[n] = n
        return n
    value = _fib_memo(n - 1, memo) + _fib_memo(n - 2, memo)
    memo[n] = value
    return value


def fib_tab(n: int) -> int:
    """Return the nth Fibonacci number bottom up, in constant space.

    Tabulation fixes the order of the subproblems in advance: F(0), then
    F(1), then F(2), and so on up to F(n). Nothing is asked for before it
    exists, so there is no recursion, no call stack and no lookup - just
    n - 1 additions in a loop.

    Fixing the order in advance is also what makes the space collapse.
    Because F(k) is known to need only the two values immediately below
    it, and because those are known to be the two computed most recently,
    the table never has to exist: a rolling pair of variables is enough.
    No list is allocated here. Memoization cannot do this, because it does
    not know in advance which entries it will still be asked for.

    Args:
        n: The index to compute. Must be a non-negative ``int``.

    Returns:
        The nth Fibonacci number.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative.

    Time Complexity:
        Best:    O(1)     - n of 0 or 1 skips the loop
        Average: O(n)     - one addition per index
        Worst:   O(n)
        Counting bit operations rather than additions it is O(n^2), since
        F(n) has Theta(n) digits and the additions widen as the loop runs.
        The same caveat applies to every variant in this module.

    Space Complexity:
        O(1) - two integers, whatever n is. This is the only variant of
        the four with no term that grows with n, and no recursion depth to
        run out of.

    Examples:
        >>> fib_tab(0), fib_tab(1)
        (0, 1)
        >>> fib_tab(10)
        55
        >>> [fib_tab(i) for i in range(10)]
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        >>> fib_tab(45)
        1134903170

        No stack to exhaust, so sizes that would raise
        :exc:`RecursionError` in both recursive variants are
        unremarkable here:

        >>> fib_tab(200)
        280571172992510140037611932413038677189525
        >>> len(str(fib_tab(10000)))
        2090

        All three graded variants agree wherever the naive one can be run:

        >>> all(fib_tab(i) == fib_memo(i) == fib_naive(i) for i in range(20))
        True

        >>> fib_tab(-1)
        Traceback (most recent call last):
            ...
        ValueError: fib_tab() expects n >= 0, got -1
    """
    _require_natural_int(n, "fib_tab")

    if n < 2:
        return n

    # The rolling pair holds F(k-2) and F(k-1) as the loop enters step k.
    previous, current = 0, 1
    for _ in range(2, n + 1):
        previous, current = current, previous + current
    return current


@lru_cache(maxsize=None)
def _fib_lru(n: int) -> int:
    """Recurse on the Fibonacci recurrence behind an unbounded LRU cache.

    The cache is attached here rather than to :func:`fib_lru` so that the
    argument validation stays outside it. That is not a stylistic
    preference: ``hash(True) == hash(1)``, so a cache wrapped around the
    public function would answer ``fib_lru(True)`` out of the entry stored
    for ``fib_lru(1)`` and the ``TypeError`` this module promises would
    never be raised.

    The cache is module-level state that outlives any single call, which
    is exactly why this variant is an extra and not the graded one. See
    :func:`fib_lru` for what a benchmark has to do about that.

    Args:
        n: A non-negative index, already validated by the caller.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(n) on a cold cache, O(1) on a hit.

    Space Complexity:
        O(n) of cache entries plus O(n) of call stack on a cold cache.

    Examples:
        >>> _fib_lru.cache_clear()
        >>> _fib_lru(10)
        55
        >>> _fib_lru.cache_info().currsize
        11
    """
    if n < 2:
        return n
    return _fib_lru(n - 1) + _fib_lru(n - 2)


def fib_lru(n: int) -> int:
    """Return the nth Fibonacci number using :func:`functools.lru_cache`.

    The extra, not one of the three variants the assignment grades. It is
    here because it is the comparison every reader asks for: the decorator
    memoizes the same subproblems :func:`fib_memo` memoizes, in the same
    top-down order, so the gap between the two is the cost of a
    hand-written dict against a C-level cache and nothing else.

    It is an extra rather than the answer because the decorator hides the
    thing being studied. Nothing about the memo table is visible in the
    source: not when an entry is written, not how many entries exist, not
    what happens at the base case.

    One practical difference matters to the benchmark. The cache is
    module-level and survives between calls, so a second timed run of the
    same n measures a dictionary lookup rather than an algorithm. Clear it
    between runs with ``fib_lru.cache_clear()``; ``fib_lru.cache_info()``
    reports the current hit, miss and size figures.

    Args:
        n: The index to compute. Must be a non-negative ``int``.

    Returns:
        The nth Fibonacci number.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``. The check
            runs before the cache is consulted, so ``True`` is rejected
            rather than being served the entry cached for ``1``.
        ValueError: If ``n`` is negative.
        RecursionError: For ``n`` beyond CPython's recursion limit on a
            cold cache, at depth n as with :func:`fib_memo`.

    Time Complexity:
        Best:    O(1)     - n already cached from an earlier call
        Average: O(n)     - cold cache, one miss per subproblem
        Worst:   O(n)

    Space Complexity:
        O(n) of cache entries, retained after the call returns, plus O(n)
        of call stack while it runs.

    Examples:
        >>> fib_lru.cache_clear()
        >>> fib_lru(10)
        55
        >>> fib_lru(45)
        1134903170

        The cache persists across calls, which is the property a
        benchmark has to neutralise:

        >>> fib_lru.cache_info().currsize
        46
        >>> fib_lru.cache_clear()
        >>> fib_lru.cache_info().currsize
        0

        It agrees with the hand-written memo:

        >>> all(fib_lru(i) == fib_memo(i) for i in range(30))
        True

        And it keeps the shared contract, ``bool`` included:

        >>> fib_lru(True)
        Traceback (most recent call last):
            ...
        TypeError: fib_lru() expects an int, got bool
    """
    _require_natural_int(n, "fib_lru")
    return _fib_lru(n)


# Surface the cache controls of the private worker on the public wrapper,
# so callers never have to reach for a private name to clear state that
# would otherwise leak from one timed run into the next.
fib_lru.cache_clear = _fib_lru.cache_clear  # type: ignore[attr-defined]
fib_lru.cache_info = _fib_lru.cache_info  # type: ignore[attr-defined]


def naive_call_count(n: int) -> int:
    """Return how many calls :func:`fib_naive` would make, without making them.

    The naive call tree has one node per call, and its shape is the
    recurrence itself: C(0) = C(1) = 1 and C(n) = 1 + C(n-1) + C(n-2).
    Solving that gives the closed form C(n) = 2*F(n+1) - 1, so the count
    is available from a single Fibonacci evaluation - and that evaluation
    is :func:`fib_tab`, which is O(n) and iterative.

    This function exists because the Week 5 benchmark has to report call
    counts at n = 40 and n = 45, where actually running the recursion
    would mean 331 million and 3.67 billion calls, minutes to hours of
    wall clock for a number that is known exactly in advance. It never
    recurses and it never calls :func:`fib_naive`; it is instant at any n.

    Args:
        n: The index whose naive call count is wanted. Must be a
            non-negative ``int``.

    Returns:
        The exact number of :func:`fib_naive` invocations required to
        evaluate F(n), counting the outermost call.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(n) integer additions, from the single :func:`fib_tab` call. No
        recursion at any n.

    Space Complexity:
        O(1) - the rolling pair inside :func:`fib_tab`.

    Examples:
        >>> naive_call_count(0), naive_call_count(1)
        (1, 1)
        >>> naive_call_count(10)
        177
        >>> naive_call_count(20)
        21891
        >>> naive_call_count(30)
        2692537
        >>> naive_call_count(35)
        29860703

        The two sizes the benchmark projects rather than measures:

        >>> naive_call_count(40)
        331160281
        >>> naive_call_count(45)
        3672623805

        The closed form is not an estimate. Counted against the
        instrumented recursion, it agrees exactly:

        >>> all(naive_call_count(i) == fib_counts(i, "naive")[1]
        ...     for i in range(16))
        True

        It is also 2*F(n+1) - 1 by construction:

        >>> naive_call_count(30) == 2 * fib_tab(31) - 1
        True
    """
    _require_natural_int(n, "naive_call_count")
    return 2 * fib_tab(n + 1) - 1


def _fib_naive_counted(n: int, counter: CallCounter) -> int:
    """Recurse naively, recording each call and the depth it reached.

    A separate copy of :func:`_fib_naive` rather than a flag on it. The
    plain version is the benchmark's baseline, and threading a counter
    through it would add a context manager to every one of exponentially
    many calls - the measurement would then be of the instrumentation.

    The counter arrives as an argument and is never read from module
    scope, so two runs in the same process keep separate tallies.

    Args:
        n: A non-negative index, already validated by the caller.
        counter: The tally to update. One frame is entered per call.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(2^n), with a heavier constant than :func:`_fib_naive` because
        of the per-call bookkeeping.

    Space Complexity:
        O(n) of call stack.

    Examples:
        >>> counter = CallCounter()
        >>> _fib_naive_counted(10, counter)
        55
        >>> counter.calls, counter.max_depth
        (177, 10)
    """
    with counter.frame():
        if n < 2:
            return n
        return _fib_naive_counted(n - 1, counter) + _fib_naive_counted(
            n - 2, counter
        )


def _fib_memo_counted(n: int, counter: CallCounter) -> int:
    """Run the memoized recursion on a fresh table, counting calls.

    The table is created here rather than accepted as an argument so that
    the counted run always starts cold. A warm table would make the count
    depend on what some earlier call happened to leave behind, and the
    figure the report quotes has to be a property of n alone.

    Args:
        n: A non-negative index, already validated by the caller.
        counter: The tally to update.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) of table plus O(n) of call stack.

    Examples:
        >>> counter = CallCounter()
        >>> _fib_memo_counted(10, counter)
        55
        >>> counter.calls, counter.max_depth
        (19, 10)
    """
    return _fib_memo_recurse_counted(n, {}, counter)


def _fib_memo_recurse_counted(
    n: int,
    memo: Dict[int, int],
    counter: CallCounter,
) -> int:
    """Recurse with a memo table, recording each call and the depth reached.

    The counted twin of :func:`_fib_memo`. The call total it produces is
    the evidence that memoization is doing its job: 2n - 1 calls for
    n >= 1, one per subproblem down the spine plus one cache hit at each
    level, against 2*F(n+1) - 1 for the same recurrence without a table.

    Args:
        n: A non-negative index, already validated by the caller.
        memo: The table to consult and extend. Modified in place.
        counter: The tally to update. One frame is entered per call,
            cache hits included - a hit is work the naive version would
            have done, so hiding it would understate the saving.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) of table plus O(n) of call stack.

    Examples:
        >>> counter = CallCounter()
        >>> _fib_memo_recurse_counted(6, {}, counter)
        8
        >>> counter.calls, counter.max_depth
        (11, 6)
    """
    with counter.frame():
        cached = memo.get(n)
        if cached is not None:
            return cached
        if n < 2:
            memo[n] = n
            return n
        value = _fib_memo_recurse_counted(
            n - 1, memo, counter
        ) + _fib_memo_recurse_counted(n - 2, memo, counter)
        memo[n] = value
        return value


def _fib_tab_counted(n: int, counter: CallCounter) -> int:
    """Iterate bottom up, recording one frame per subproblem evaluated.

    Tabulation makes no calls at all, so "calls" has to be read as
    "subproblems evaluated" for this variant if the three are to be
    compared on one axis. That is n + 1: the two base values plus one
    addition for each index from 2 to n.

    The depth it reports is 1, and that is the finding rather than a
    formality. Every subproblem is entered and left at the same level,
    because the loop never asks for a value that does not already exist.

    Args:
        n: A non-negative index, already validated by the caller.
        counter: The tally to update.

    Returns:
        The nth Fibonacci number.

    Time Complexity:
        O(n).

    Space Complexity:
        O(1) beyond the counter - the rolling pair, as in
        :func:`fib_tab`.

    Examples:
        >>> counter = CallCounter()
        >>> _fib_tab_counted(10, counter)
        55
        >>> counter.calls, counter.max_depth
        (11, 1)
        >>> zero = CallCounter()
        >>> _fib_tab_counted(0, zero)
        0
        >>> zero.calls, zero.max_depth
        (1, 1)
    """
    # F(0) and F(1) are subproblems too. They are read off the base case
    # rather than computed, but the table has a cell for each of them and
    # the count would be off by two if they were skipped.
    with counter.frame():
        previous = 0
    if n == 0:
        return previous

    with counter.frame():
        current = 1
    if n == 1:
        return current

    for _ in range(2, n + 1):
        with counter.frame():
            previous, current = current, previous + current
    return current


def fib_instrumented(n: int, variant: str) -> Tuple[int, CallCounter]:
    """Compute F(n) with one variant and return the counter that watched it.

    The full instrumented result: the value, and a
    :class:`~src.utils.timer.CallCounter` carrying both the number of
    calls made and the deepest nesting reached. :func:`fib_counts` is the
    same thing with the counter unpacked, for callers that only want the
    total.

    The counter is created inside this function and returned to the
    caller. Nothing is stored at module scope, so two benchmark runs in
    the same process cannot read or corrupt each other's figures.

    What the depth figure shows is worth stating plainly: the two
    recursive variants both nest to depth n, and tabulation reports 1.
    That is the same difference as the space bound, seen from the stack.

    Args:
        n: The index to compute. Must be a non-negative ``int``.
        variant: One of ``"naive"``, ``"memo"`` or ``"tab"``.

    Returns:
        A ``(value, counter)`` pair. ``counter.calls`` is the number of
        instrumented calls, or subproblems evaluated for ``"tab"``, and
        ``counter.max_depth`` is the deepest nesting reached.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative, or ``variant`` is not one of the
            three names.
        RecursionError: For large ``n`` under ``"naive"`` or ``"memo"``,
            at depth n. ``"tab"`` never recurses.

    Time Complexity:
        O(2^n) for ``"naive"``, O(n) for ``"memo"`` and ``"tab"``. The
        bookkeeping adds a constant factor per call and does not change
        any of the three exponents.

    Space Complexity:
        O(n) for ``"naive"`` (stack) and ``"memo"`` (stack and table);
        O(1) for ``"tab"``.

    Examples:
        The same value three ways, at wildly different cost:

        >>> value, counter = fib_instrumented(10, "naive")
        >>> value, counter.calls, counter.max_depth
        (55, 177, 10)
        >>> value, counter = fib_instrumented(10, "memo")
        >>> value, counter.calls, counter.max_depth
        (55, 19, 10)
        >>> value, counter = fib_instrumented(10, "tab")
        >>> value, counter.calls, counter.max_depth
        (55, 11, 1)

        Memoization stays linear where the naive tree explodes:

        >>> _, counter = fib_instrumented(30, "memo")
        >>> counter.calls
        59
        >>> naive_call_count(30)
        2692537

        Each run gets its own counter, so neither disturbs the other:

        >>> _, first = fib_instrumented(10, "naive")
        >>> _, second = fib_instrumented(10, "memo")
        >>> first.calls, second.calls
        (177, 19)

        >>> fib_instrumented(10, "lru")
        ... # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: fib_instrumented() got unknown variant 'lru';
        expected one of 'memo', 'naive', 'tab'
    """
    _require_natural_int(n, "fib_instrumented")
    counted = _require_variant(variant, _COUNTED_VARIANTS, "fib_instrumented")

    counter = CallCounter()
    value = counted(n, counter)
    return value, counter


def fib_counts(n: int, variant: str) -> Tuple[int, int]:
    """Compute F(n) with one variant and report how many calls it took.

    The short form of :func:`fib_instrumented`, for the common case where
    the call total is the whole question. The depth figure is dropped; ask
    for the counter itself when it is wanted.

    For ``"naive"`` this is only affordable to about n = 30. Above that
    use :func:`naive_call_count`, which gives the identical figure from
    the closed form without running anything.

    Args:
        n: The index to compute. Must be a non-negative ``int``.
        variant: One of ``"naive"``, ``"memo"`` or ``"tab"``.

    Returns:
        A ``(value, calls)`` pair. ``calls`` counts instrumented calls,
        read as subproblems evaluated for ``"tab"``.

    Raises:
        TypeError: If ``n`` is not an ``int``, or is a ``bool``.
        ValueError: If ``n`` is negative, or ``variant`` is not one of the
            three names.

    Time Complexity:
        O(2^n) for ``"naive"``, O(n) for ``"memo"`` and ``"tab"``.

    Space Complexity:
        O(n) for ``"naive"`` and ``"memo"``, O(1) for ``"tab"``.

    Examples:
        The headline comparison of the whole module:

        >>> fib_counts(10, "naive")
        (55, 177)
        >>> fib_counts(10, "memo")
        (55, 19)
        >>> fib_counts(10, "tab")
        (55, 11)

        Memoized counts are 2n - 1, linear in n rather than exponential:

        >>> [fib_counts(i, "memo")[1] for i in range(1, 8)]
        [1, 3, 5, 7, 9, 11, 13]
        >>> fib_counts(25, "memo")[1]
        49

        Tabulation evaluates each subproblem exactly once, n + 1 of them:

        >>> [fib_counts(i, "tab")[1] for i in range(6)]
        [1, 2, 3, 4, 5, 6]

        The base cases still cost one call each:

        >>> fib_counts(0, "naive"), fib_counts(1, "naive")
        ((0, 1), (1, 1))

        >>> fib_counts(10, "iterative")
        ... # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: fib_counts() got unknown variant 'iterative';
        expected one of 'memo', 'naive', 'tab'
    """
    _require_natural_int(n, "fib_counts")
    counted = _require_variant(variant, _COUNTED_VARIANTS, "fib_counts")

    counter = CallCounter()
    value = counted(n, counter)
    return value, counter.calls


# The three graded implementations, keyed by the names the benchmark and
# the tests loop over. fib_lru is deliberately absent: it is the extra,
# not one of the three the assignment compares.
VARIANTS: Dict[str, Callable[[int], int]] = {
    "naive": fib_naive,
    "memo": fib_memo,
    "tab": fib_tab,
}

# The counted twin of VARIANTS, private because the counter argument is an
# implementation detail of fib_counts and fib_instrumented. Keeping it
# separate is what keeps every function in VARIANTS free of bookkeeping.
_COUNTED_VARIANTS: Dict[str, Callable[[int, CallCounter], int]] = {
    "naive": _fib_naive_counted,
    "memo": _fib_memo_counted,
    "tab": _fib_tab_counted,
}
