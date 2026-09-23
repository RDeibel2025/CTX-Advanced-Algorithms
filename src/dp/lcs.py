"""Longest common subsequence, written three ways so it can be measured.

The longest common subsequence (LCS) of two strings is the longest string
that can be obtained from both by deleting characters without reordering
the ones that remain. ``"GTAB"`` survives inside ``"AGGTAB"`` and inside
``"GXTXAYB"``, and nothing longer does, so the LCS length of that pair is
four. Subsequences are not substrings: the characters need to appear in
order, not next to each other.

One recurrence drives all three implementations. Writing ``X[:i]`` and
``Y[:j]`` for the prefixes of length ``i`` and ``j``::

    L(0, j) = L(i, 0) = 0
    L(i, j) = L(i - 1, j - 1) + 1                 if X[i-1] == Y[j-1]
    L(i, j) = max(L(i - 1, j), L(i, j - 1))       otherwise

The last character either matches, in which case it is worth one and the
problem shrinks on both sides at once, or it does not, in which case one
of the two strings must give up its last character and the better of the
two choices wins. That is optimal substructure. The reason memoization
and tabulation help at all is the second condition: the branches overlap
heavily, because ``L(i - 1, j)`` and ``L(i, j - 1)`` both go on to ask for
``L(i - 1, j - 1)``.

The three functions below solve exactly the same subproblems. They differ
only in who decides the order:

====================== ================ ==================== ==========
Function               Time             Space                Depth
====================== ================ ==================== ==========
lcs_recursive          O(2^(m+n))       O(m + n) stack       m + n
lcs_memo               O(m*n)           O(m*n) + stack       m + n
lcs_tab                O(m*n)           O(m*n) table         1
====================== ================ ==================== ==========

Two consequences of that difference are worth stating up front, because
they are the measurements this module exists to support.

**The memo is keyed on the index pair, never on string slices.** Keying a
memo on ``(X[:i], Y[:j])`` is the natural-looking mistake, and it is a
real trap: building each key copies O(m + n) characters and hashing it
costs another O(m + n) pass, so the table lookup that is supposed to be
O(1) silently becomes O(m + n). That turns an O(m*n) algorithm into an
O(m*n*(m+n)) one while every answer it returns stays correct, which is
what makes the bug so hard to see. The keys here are ``(i, j)`` pairs of
small integers: cheap to build, cheap to hash, and they never copy the
input.

**Laziness borrows the interpreter's call stack.** Both recursive forms
descend one frame per character consumed, so the worst case reaches a
depth of ``len(X) + len(Y)``: a pair of 1,000-character strings needs a
recursion limit near 2,000 against CPython's default of 1,000, and
without one it raises :exc:`RecursionError` long before it runs out of
time or memory. Tabulation has depth 1 by construction and does not care.
This module deliberately does **not** call :func:`sys.setrecursionlimit`.
Raising a process-wide limit is the caller's decision, not a library's,
and the benchmark that owns that decision has to state it; silently
raising it here would hide the very asymmetry being measured.

Case sensitivity: characters are compared with ``==``, so ``"A"`` and
``"a"`` are different characters and ``lcs_tab("ABC", "abc")`` is 0.
Callers who want a case-insensitive match should casefold both inputs
before calling.

Examples:
    >>> lcs_tab("AGGTAB", "GXTXAYB")
    4
    >>> lcs_reconstruct("AGGTAB", "GXTXAYB")
    'GTAB'

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from src.utils.timer import CallCounter

__all__ = [
    "VARIANTS",
    "lcs_instrumented",
    "lcs_memo",
    "lcs_recursive",
    "lcs_reconstruct",
    "lcs_tab",
]


def _require_strings(X: Any, Y: Any, func_name: str) -> None:
    """Raise :class:`TypeError` unless both arguments are ``str``.

    Centralising the check keeps the message identical across the public
    functions and keeps each algorithm body about the algorithm. The
    argument that failed is named, because the two are easy to swap and a
    message that only says "expects str" makes the caller guess which one
    went wrong.

    Args:
        X: The object intended as the first string.
        Y: The object intended as the second string.
        func_name: Name of the calling function, used in the message.

    Returns:
        None. This helper is called for its exception, not its value.

    Raises:
        TypeError: If either argument is not a ``str``. ``X`` is reported
            first when both are wrong.

    Time Complexity:
        O(1) - two isinstance checks, no scan of the strings themselves.

    Space Complexity:
        O(1).

    Examples:
        >>> _require_strings("AB", "BA", "demo") is None
        True
        >>> _require_strings("", "", "demo") is None
        True
        >>> _require_strings(["A"], "BA", "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects str arguments, got list for X
        >>> _require_strings("AB", 5, "demo")
        Traceback (most recent call last):
            ...
        TypeError: demo() expects str arguments, got int for Y
    """
    for name, value in (("X", X), ("Y", Y)):
        if not isinstance(value, str):
            raise TypeError(
                f"{func_name}() expects str arguments, "
                f"got {type(value).__name__} for {name}"
            )


def _lcs_recursive(X: str, Y: str, i: int, j: int) -> int:
    """Return the LCS length of ``X[:i]`` and ``Y[:j]`` by plain recursion.

    The recursion carries prefix *lengths* rather than the prefixes
    themselves. Passing ``X[:i]`` down would copy the string at every
    level and add an O(m + n) factor to a function that is already
    exponential, which would confuse the measurement this module exists
    for. Indices are two integers and cost nothing to pass.

    Nothing is remembered between calls, so the overlapping subproblems
    are recomputed every time they are reached. That is the point: this
    is the baseline the memoized and tabulated versions are priced
    against.

    Args:
        X: The first string.
        Y: The second string.
        i: How many leading characters of ``X`` are in play, 0 to len(X).
        j: How many leading characters of ``Y`` are in play, 0 to len(Y).

    Returns:
        The length of the longest common subsequence of the two prefixes.

    Raises:
        RecursionError: If ``i + j`` exceeds the interpreter's recursion
            limit. The deepest chain consumes one character per frame.

    Time Complexity:
        Best:    O(m + n) - every character matches, so each call takes
                 the diagonal branch and the recursion is a single chain.
        Average: exponential in ``m + n``.
        Worst:   O(2^(m+n)) - no character matches, so every call forks
                 in two. The exact node count of that tree satisfies
                 T(i, j) = 1 + T(i-1, j) + T(i, j-1), which solves to
                 2 * C(i + j, i) - 1; the 2^(m+n) bound is the loose
                 form of that binomial.

    Space Complexity:
        O(m + n) - call stack only, one frame per character consumed.

    Examples:
        >>> _lcs_recursive("AGGTAB", "GXTXAYB", 6, 7)
        4
        >>> _lcs_recursive("AGGTAB", "GXTXAYB", 0, 7)
        0
        >>> _lcs_recursive("ABCBDAB", "BDCABA", 7, 6)
        4
    """
    if i == 0 or j == 0:
        return 0

    if X[i - 1] == Y[j - 1]:
        return _lcs_recursive(X, Y, i - 1, j - 1) + 1

    return max(
        _lcs_recursive(X, Y, i - 1, j),
        _lcs_recursive(X, Y, i, j - 1),
    )


def _lcs_memo(
    X: str,
    Y: str,
    i: int,
    j: int,
    memo: Dict[Tuple[int, int], int],
) -> int:
    """Return the LCS length of ``X[:i]`` and ``Y[:j]``, remembering answers.

    The body is :func:`_lcs_recursive` with two lines added: a lookup
    before the work and a store after it. Every ``(i, j)`` is computed at
    most once, so the exponential tree collapses to at most ``m * n``
    evaluated nodes and the cost becomes O(m*n).

    The dictionary is keyed on the ``(i, j)`` integer pair. See the module
    docstring for why keying it on string slices would be a correct
    program with the wrong complexity.

    Base cases are not stored. They return 0 without touching the memo,
    which is already O(1), so storing them would add ``m + n + 1`` entries
    that buy nothing and would inflate the memo-size figure the benchmark
    reports.

    Args:
        X: The first string.
        Y: The second string.
        i: How many leading characters of ``X`` are in play.
        j: How many leading characters of ``Y`` are in play.
        memo: Maps ``(i, j)`` to the LCS length of those prefixes. Mutated
            in place, and reused across the whole descent.

    Returns:
        The length of the longest common subsequence of the two prefixes.

    Raises:
        RecursionError: If ``i + j`` exceeds the interpreter's recursion
            limit. Memoization cuts the *number* of calls, not the depth
            of the deepest chain, so this limit binds exactly as it does
            for the plain recursion.

    Time Complexity:
        Best:    O(m + n) - identical strings walk one diagonal, and no
                 subproblem is ever asked for twice.
        Average: O(m*n)
        Worst:   O(m*n) - each of the m*n index pairs is evaluated once
                 at O(1) cost, plus O(1) lookups for the repeats.

    Space Complexity:
        O(m*n) for the memo in the worst case, plus O(m + n) of call
        stack. The stack term is the one that fails first in CPython.

    Examples:
        >>> _lcs_memo("AGGTAB", "GXTXAYB", 6, 7, {})
        4
        >>> memo = {}
        >>> _lcs_memo("ABCBDAB", "BDCABA", 7, 6, memo)
        4
        >>> memo[(7, 6)]
        4
        >>> (0, 3) in memo
        False
    """
    if i == 0 or j == 0:
        return 0

    key = (i, j)
    remembered = memo.get(key)
    if remembered is not None:
        return remembered

    if X[i - 1] == Y[j - 1]:
        result = _lcs_memo(X, Y, i - 1, j - 1, memo) + 1
    else:
        result = max(
            _lcs_memo(X, Y, i - 1, j, memo),
            _lcs_memo(X, Y, i, j - 1, memo),
        )

    memo[key] = result
    return result


def _build_table(X: str, Y: str) -> List[List[int]]:
    """Fill the full ``(m+1) x (n+1)`` LCS table bottom-up.

    Row ``i`` and column ``j`` hold the LCS length of ``X[:i]`` and
    ``Y[:j]``, so the answer for the whole pair ends up in the bottom
    right corner. Row 0 and column 0 are zero because a prefix of length
    zero shares nothing with anything.

    Filling the rows in increasing ``i`` and the columns in increasing
    ``j`` guarantees that the three cells each step reads - above, left
    and diagonally up-left - are already written. That ordering is the
    whole of bottom-up dynamic programming: no call stack, no lookups
    that might miss, and no question of whether a value is ready yet.

    :func:`lcs_tab` and :func:`lcs_reconstruct` share this helper because
    they need the same table. Reconstruction is why the full table is kept
    rather than the rolling pair of rows that the length alone would
    allow: the backward walk reads cells from every row.

    Args:
        X: The first string.
        Y: The second string.

    Returns:
        A list of ``len(X) + 1`` rows, each of ``len(Y) + 1`` ints.

    Time Complexity:
        O(m*n) in every case - the two loops run to completion whatever
        the strings contain. There is no early exit and no best case,
        which is the price of deciding the order in advance.

    Space Complexity:
        O(m*n) - the table itself.

    Examples:
        >>> for row in _build_table("AB", "ACB"):
        ...     print(row)
        [0, 0, 0, 0]
        [0, 1, 1, 1]
        [0, 1, 1, 2]
        >>> _build_table("", "ACB")
        [[0, 0, 0, 0]]
    """
    m = len(X)
    n = len(Y)
    table: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        x_char = X[i - 1]
        row = table[i]
        previous = table[i - 1]
        for j in range(1, n + 1):
            if x_char == Y[j - 1]:
                row[j] = previous[j - 1] + 1
            else:
                row[j] = max(previous[j], row[j - 1])

    return table


def _lcs_recursive_counted(
    X: str,
    Y: str,
    i: int,
    j: int,
    counter: CallCounter,
) -> int:
    """Run the plain recursion while recording calls and depth.

    A deliberate duplicate of :func:`_lcs_recursive`, one frame wrapper
    heavier. The plain version stays free of instrumentation because it is
    what the benchmark times, and a counter update inside an exponential
    hot loop would price the bookkeeping rather than the algorithm. This
    version is never timed; it is read for its counts.

    The counter is a parameter rather than a module-level global, so two
    benchmark runs in the same process cannot corrupt each other's totals.

    Args:
        X: The first string.
        Y: The second string.
        i: How many leading characters of ``X`` are in play.
        j: How many leading characters of ``Y`` are in play.
        counter: Receives one :meth:`CallCounter.enter` per call, base
            cases included. Mutated in place.

    Returns:
        The length of the longest common subsequence of the two prefixes.

    Raises:
        RecursionError: If ``i + j`` exceeds the interpreter's recursion
            limit.

    Time Complexity:
        As :func:`_lcs_recursive`, O(2^(m+n)) in the worst case, with a
        heavier constant for the counter frame.

    Space Complexity:
        O(m + n) - call stack; the counter itself is O(1).

    Examples:
        Two strings with no character in common produce the full
        recursion tree, and its size is exactly 2 * C(i + j, i) - 1:
        C(4, 2) is 6, so a 2-by-2 pair costs 11 calls and reaches a depth
        of i + j.

        >>> counter = CallCounter()
        >>> _lcs_recursive_counted("AB", "CD", 2, 2, counter)
        0
        >>> counter.calls, counter.max_depth
        (11, 4)

        Identical strings never fork, so the tree is one chain:

        >>> counter = CallCounter()
        >>> _lcs_recursive_counted("ABC", "ABC", 3, 3, counter)
        3
        >>> counter.calls, counter.max_depth
        (4, 4)
    """
    with counter.frame():
        if i == 0 or j == 0:
            return 0

        if X[i - 1] == Y[j - 1]:
            return _lcs_recursive_counted(X, Y, i - 1, j - 1, counter) + 1

        return max(
            _lcs_recursive_counted(X, Y, i - 1, j, counter),
            _lcs_recursive_counted(X, Y, i, j - 1, counter),
        )


def _lcs_memo_counted(
    X: str,
    Y: str,
    i: int,
    j: int,
    memo: Dict[Tuple[int, int], int],
    counter: CallCounter,
) -> int:
    """Run the memoized recursion while recording calls and depth.

    The instrumented twin of :func:`_lcs_memo`, kept separate for the same
    reason: the timed function carries no bookkeeping. Cache hits are
    counted as calls, because they are calls - a frame is pushed, the
    dictionary is consulted and the frame is popped - and the gap between
    that number and the exponential count of the plain recursion is the
    measurement the report reads.

    Args:
        X: The first string.
        Y: The second string.
        i: How many leading characters of ``X`` are in play.
        j: How many leading characters of ``Y`` are in play.
        memo: Maps ``(i, j)`` to an already computed length. Mutated in
            place; ``len(memo)`` afterwards is the number of distinct
            non-trivial subproblems that were evaluated.
        counter: Receives one :meth:`CallCounter.enter` per call, base
            cases and cache hits included. Mutated in place.

    Returns:
        The length of the longest common subsequence of the two prefixes.

    Raises:
        RecursionError: If ``i + j`` exceeds the interpreter's recursion
            limit.

    Time Complexity:
        O(m*n) worst case, as :func:`_lcs_memo`.

    Space Complexity:
        O(m*n) for the memo plus O(m + n) of call stack.

    Examples:
        The same 2-by-2 disjoint pair that costs 11 calls unmemoized
        costs 9, and the gap widens fast with length:

        >>> counter = CallCounter()
        >>> memo = {}
        >>> _lcs_memo_counted("AB", "CD", 2, 2, memo, counter)
        0
        >>> counter.calls, counter.max_depth
        (9, 4)
        >>> len(memo)
        4
    """
    with counter.frame():
        if i == 0 or j == 0:
            return 0

        key = (i, j)
        remembered = memo.get(key)
        if remembered is not None:
            return remembered

        if X[i - 1] == Y[j - 1]:
            result = _lcs_memo_counted(X, Y, i - 1, j - 1, memo, counter) + 1
        else:
            result = max(
                _lcs_memo_counted(X, Y, i - 1, j, memo, counter),
                _lcs_memo_counted(X, Y, i, j - 1, memo, counter),
            )

        memo[key] = result
        return result


def _lcs_tab_counted(X: str, Y: str, counter: CallCounter) -> int:
    """Fill the LCS table while counting the cells it evaluates.

    Tabulation makes no calls, so "calls" is reported as the number of
    subproblems evaluated, which for the table is every one of its
    ``(m + 1) * (n + 1)`` cells including the zero row and column. That is
    the honest figure to set beside the memoized call count: it is the
    work bottom-up does whether or not the answer needs it.

    Maximum depth is 1 by construction. There is no recursion to nest, so
    every cell is entered from the top level and left again before the
    next one starts. Reporting that 1 is the point, not an artefact.

    Args:
        X: The first string.
        Y: The second string.
        counter: Receives one :meth:`CallCounter.enter` per table cell.
            Mutated in place.

    Returns:
        The length of the longest common subsequence of ``X`` and ``Y``.

    Time Complexity:
        O(m*n) in every case, with a heavier constant than
        :func:`_build_table` because of the per-cell counter frame.

    Space Complexity:
        O(m*n) - the table.

    Examples:
        >>> counter = CallCounter()
        >>> _lcs_tab_counted("AB", "ACB", counter)
        2
        >>> counter.calls, counter.max_depth
        (12, 1)

        Even an empty pair evaluates the single corner cell:

        >>> counter = CallCounter()
        >>> _lcs_tab_counted("", "", counter)
        0
        >>> counter.calls, counter.max_depth
        (1, 1)
    """
    m = len(X)
    n = len(Y)
    table: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            with counter.frame():
                if i == 0 or j == 0:
                    table[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])

    return table[m][n]


def lcs_recursive(X: str, Y: str) -> int:
    """Return the LCS length of two strings by plain double recursion.

    The recurrence, straight from its definition and with nothing
    remembered between calls. Each call either consumes a matching pair of
    final characters or forks into two smaller problems, so the same
    prefix pair is solved again every time a different path reaches it.
    This is the baseline: it is here to be slow in a way the memoized and
    tabulated versions can be measured against.

    Use it on short strings only. At 14 characters a side the recursion
    tree already runs to millions of nodes, which is why the Week 5
    benchmark caps this variant at that length and says so in the report
    rather than quietly dropping the point.

    Comparison is case sensitive, as it is throughout this module.

    Args:
        X: The first string. Not modified.
        Y: The second string. Not modified.

    Returns:
        The length of the longest common subsequence, 0 when either
        string is empty or the two share no character.

    Raises:
        TypeError: If either argument is not a ``str``.
        RecursionError: If ``len(X) + len(Y)`` exceeds the interpreter's
            recursion limit, which defaults to 1,000 in CPython. This
            module does not raise that limit; see the module docstring.

    Time Complexity:
        Best:    O(m + n) - identical strings match at every step, so the
                 recursion never forks and is a single diagonal chain.
        Average: exponential in ``m + n``.
        Worst:   O(2^(m+n)) - disjoint alphabets, where every call forks.
                 The exact tree size is 2 * C(m + n, m) - 1.

    Space Complexity:
        O(m + n) - the call stack, one frame per character consumed. No
        table is built, which is the one thing this version has going
        for it.

    Examples:
        The hand-checked anchor. "GTAB" appears in order in both strings,
        and nothing longer does:

        >>> lcs_recursive("AGGTAB", "GXTXAYB")
        4

        The CLRS 4th edition example from section 14.4, whose LCS is
        "BCBA":

        >>> lcs_recursive("ABCBDAB", "BDCABA")
        4

        Empty, identical and disjoint inputs:

        >>> lcs_recursive("", "ANYTHING")
        0
        >>> lcs_recursive("SAME", "SAME")
        4
        >>> lcs_recursive("abc", "xyz")
        0

        Case matters, so an upper case A does not match a lower case one:

        >>> lcs_recursive("ABC", "abc")
        0

        A subsequence need not be contiguous:

        >>> lcs_recursive("ABCDEFG", "AXCXEXG")
        4

        >>> lcs_recursive("AB", 5)
        Traceback (most recent call last):
            ...
        TypeError: lcs_recursive() expects str arguments, got int for Y
    """
    _require_strings(X, Y, "lcs_recursive")
    return _lcs_recursive(X, Y, len(X), len(Y))


def lcs_memo(X: str, Y: str) -> int:
    """Return the LCS length of two strings, top-down with an explicit dict.

    The same recursion as :func:`lcs_recursive`, asked to remember. A
    fresh dictionary is created per call and threaded down the descent, so
    nothing survives between calls and two runs in the same process cannot
    interfere. Each distinct prefix pair is evaluated once and read back
    on every later visit, which drops the cost from exponential to
    O(m*n).

    The memo is keyed on the ``(i, j)`` index pair. Keying it on the
    slices ``(X[:i], Y[:j])`` would still return the right answers while
    adding an O(m + n) copy and hash to every lookup, turning O(m*n) into
    O(m*n*(m+n)). The module docstring says more about why that trap is
    worth naming.

    Top-down only evaluates the subproblems it actually reaches. For LCS
    that saving is small, because almost every cell is reachable, but the
    property is real and the Week 5 knapsack module measures it where it
    pays.

    Recursion depth is the catch. Memoization removes repeated calls, not
    the length of the deepest chain, so this still descends
    ``len(X) + len(Y)`` frames in the worst case and still meets CPython's
    1,000-frame default on inputs a few hundred characters long. The
    tabulated version has no such limit, and that asymmetry is a property
    of the runtime rather than of the algorithm.

    Args:
        X: The first string. Not modified.
        Y: The second string. Not modified.

    Returns:
        The length of the longest common subsequence.

    Raises:
        TypeError: If either argument is not a ``str``.
        RecursionError: If ``len(X) + len(Y)`` exceeds the interpreter's
            recursion limit. Raising that limit is the caller's decision
            and this module does not make it.

    Time Complexity:
        Best:    O(m + n) - identical strings, one diagonal, no repeats.
        Average: O(m*n)
        Worst:   O(m*n) - every index pair evaluated once at O(1).

    Space Complexity:
        O(m*n) for the memo, plus O(m + n) of call stack. Base cases are
        not stored, so the dictionary holds at most m*n entries.

    Examples:
        >>> lcs_memo("AGGTAB", "GXTXAYB")
        4
        >>> lcs_memo("ABCBDAB", "BDCABA")
        4

        It agrees with the plain recursion everywhere, and unlike the
        plain recursion it can be asked for longer strings:

        >>> lcs_memo("ABCDEFG", "AXCXEXG") == lcs_recursive("ABCDEFG", "AXCXEXG")
        True
        >>> lcs_memo("dynamic programming", "programming dynamically")
        11

        Edge cases behave as they do everywhere in this module:

        >>> lcs_memo("", "")
        0
        >>> lcs_memo("SAME", "SAME")
        4
        >>> lcs_memo("abc", "xyz")
        0
        >>> lcs_memo(None, "abc")
        Traceback (most recent call last):
            ...
        TypeError: lcs_memo() expects str arguments, got NoneType for X
    """
    _require_strings(X, Y, "lcs_memo")
    memo: Dict[Tuple[int, int], int] = {}
    return _lcs_memo(X, Y, len(X), len(Y), memo)


def lcs_tab(X: str, Y: str) -> int:
    """Return the LCS length of two strings, bottom-up with a full table.

    Every cell of the ``(m+1) x (n+1)`` table is filled in increasing row
    and column order, so each cell's three inputs are already written when
    it is reached and no cell ever has to ask whether a value is ready.
    There are no call frames and no dictionary hashing, which is why this
    beats :func:`lcs_memo` on constants whenever the whole table is needed
    anyway - and for two strings with no structure, nearly all of it is.

    The cost of deciding the order in advance is that the table is filled
    whether or not every cell is needed, and there is no early exit: the
    best case and the worst case are both O(m*n).

    Space is O(m*n) here on purpose. The length alone needs only the
    previous row, an O(min(m, n)) rolling window if the shorter string is
    laid along the columns, but reconstruction reads cells from every row,
    so :func:`lcs_reconstruct` needs the full table and shares this one
    through the same private builder.

    Args:
        X: The first string. Not modified.
        Y: The second string. Not modified.

    Returns:
        The length of the longest common subsequence.

    Raises:
        TypeError: If either argument is not a ``str``.

    Time Complexity:
        Best:    O(m*n)
        Average: O(m*n)
        Worst:   O(m*n) - the loops run to completion in every case.

    Space Complexity:
        O(m*n) for the table. A rolling row would make it O(min(m, n))
        when only the length is wanted. Recursion depth is 1, so no
        interpreter limit applies at any input size.

    Examples:
        >>> lcs_tab("AGGTAB", "GXTXAYB")
        4
        >>> lcs_tab("ABCBDAB", "BDCABA")
        4

        All three variants agree, which is the property the test suite
        checks over a random battery:

        >>> lcs_tab("ABCDEFG", "AXCXEXG") == lcs_memo("ABCDEFG", "AXCXEXG")
        True

        An empty string shares nothing, identical strings share
        everything, and disjoint alphabets share nothing:

        >>> lcs_tab("", "ABC")
        0
        >>> lcs_tab("ABC", "")
        0
        >>> lcs_tab("SAME", "SAME")
        4
        >>> lcs_tab("abc", "xyz")
        0
        >>> lcs_tab("ABC", "abc")
        0

        Unlike the recursive forms, length is no obstacle:

        >>> lcs_tab("A" * 600, "A" * 600)
        600
        >>> lcs_tab(3.5, "ABC")
        Traceback (most recent call last):
            ...
        TypeError: lcs_tab() expects str arguments, got float for X
    """
    _require_strings(X, Y, "lcs_tab")
    return _build_table(X, Y)[len(X)][len(Y)]


def lcs_reconstruct(X: str, Y: str) -> str:
    """Return an actual longest common subsequence, not just its length.

    The table says how long the answer is; recovering the answer itself
    means walking back through the decisions that produced it. Starting at
    the bottom right corner and reading the same recurrence backwards: if
    the two current characters match, that character belongs to the
    subsequence and both indices step back together; otherwise the walk
    follows whichever neighbour the forward pass took its value from, up
    when the cell above is at least as large and left otherwise. Each step
    consumes at least one character, so the walk visits at most ``m + n``
    cells.

    Characters are collected from the back and reversed at the end, which
    is cheaper than prepending to a string ``m + n`` times.

    Ties are broken towards the cell above. When both neighbours hold the
    same value there are several longest common subsequences and this walk
    returns one of them; the choice is deterministic, so the same input
    always gives the same output, but a different tie-break would return a
    different string of the same length. Callers should test the property
    - a common subsequence of the right length - not the exact string.

    This is why :func:`lcs_tab` keeps the whole table rather than a
    rolling row: the walk reads cells from every row it passes through.

    Args:
        X: The first string. Not modified.
        Y: The second string. Not modified.

    Returns:
        A string that is a subsequence of both inputs and whose length
        equals ``lcs_tab(X, Y)``. Empty when there is nothing in common.

    Raises:
        TypeError: If either argument is not a ``str``.

    Time Complexity:
        O(m*n) - dominated by building the table. The backward walk
        itself is O(m + n).

    Space Complexity:
        O(m*n) for the table, plus O(min(m, n)) for the characters
        collected, since no common subsequence can be longer than the
        shorter input.

    Examples:
        The hand-checked anchor: "GTAB" is the subsequence behind the
        length of 4:

        >>> lcs_reconstruct("AGGTAB", "GXTXAYB")
        'GTAB'

        The CLRS section 14.4 pair:

        >>> lcs_reconstruct("ABCBDAB", "BDCABA")
        'BCBA'

        The two properties that matter, stated as the tests state them:

        >>> answer = lcs_reconstruct("ABCDEFG", "AXCXEXG")
        >>> len(answer) == lcs_tab("ABCDEFG", "AXCXEXG")
        True
        >>> answer
        'ACEG'

        Nothing in common gives the empty string, not ``None``:

        >>> lcs_reconstruct("abc", "xyz")
        ''
        >>> lcs_reconstruct("", "ABC")
        ''
        >>> lcs_reconstruct("SAME", "SAME")
        'SAME'
        >>> lcs_reconstruct("ABC", ["a"])
        Traceback (most recent call last):
            ...
        TypeError: lcs_reconstruct() expects str arguments, got list for Y
    """
    _require_strings(X, Y, "lcs_reconstruct")

    table = _build_table(X, Y)
    i = len(X)
    j = len(Y)
    collected: List[str] = []

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            collected.append(X[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1

    collected.reverse()
    return "".join(collected)


def lcs_instrumented(
    X: str,
    Y: str,
    variant: str,
) -> Tuple[int, CallCounter]:
    """Run one LCS variant and report its call count and recursion depth.

    The three public functions above carry no instrumentation, because
    they are what the benchmark times and a counter update inside the hot
    loop would price the bookkeeping instead of the algorithm. This
    function is the other half of that split: it runs private twins that
    do the counting, is never timed, and is read for its counts.

    The counter is created here and passed down, never stored at module
    level, so two benchmark runs in the same process cannot corrupt each
    other's totals.

    What gets counted depends on what the variant does. ``"recursive"``
    and ``"memo"`` count one entry per call, base cases and cache hits
    included, and report the deepest nesting reached. ``"tab"`` makes no
    calls, so it counts subproblems evaluated - every one of the
    ``(m + 1) * (n + 1)`` table cells - and reports a maximum depth of 1,
    which is the whole point of bottom-up.

    Args:
        X: The first string. Not modified.
        Y: The second string. Not modified.
        variant: One of ``"recursive"``, ``"memo"`` or ``"tab"``.

    Returns:
        A ``(length, counter)`` pair. ``length`` is the same number the
        matching public function returns, and ``counter`` is a fresh
        :class:`CallCounter` whose ``calls`` and ``max_depth`` describe
        that run.

    Raises:
        TypeError: If either string argument is not a ``str``.
        ValueError: If ``variant`` is not one of the three names.
        RecursionError: From the ``"recursive"`` and ``"memo"`` variants
            if ``len(X) + len(Y)`` exceeds the interpreter's limit.

    Time Complexity:
        That of the chosen variant, with a constant factor added for the
        counter: O(2^(m+n)) for ``"recursive"``, O(m*n) for ``"memo"``
        and ``"tab"``.

    Space Complexity:
        That of the chosen variant. The counter itself is O(1).

    Examples:
        The comparison the report is built on. Two disjoint 5-character
        strings force the full recursion tree, 2 * C(10, 5) - 1 calls of
        it, while memoization visits each index pair once:

        >>> length, counter = lcs_instrumented("ABCDE", "VWXYZ", "recursive")
        >>> length, counter.calls, counter.max_depth
        (0, 503, 10)
        >>> length, counter = lcs_instrumented("ABCDE", "VWXYZ", "memo")
        >>> length, counter.calls, counter.max_depth
        (0, 51, 10)

        Both recursive forms reach the same depth, m + n; only tabulation
        escapes it, at the cost of evaluating every cell:

        >>> length, counter = lcs_instrumented("ABCDE", "VWXYZ", "tab")
        >>> length, counter.calls, counter.max_depth
        (0, 36, 1)

        >>> lcs_instrumented("AB", "BA", "quadratic")
        Traceback (most recent call last):
            ...
        ValueError: unknown variant 'quadratic'; expected 'recursive', 'memo' or 'tab'
        >>> lcs_instrumented(7, "BA", "memo")
        Traceback (most recent call last):
            ...
        TypeError: lcs_instrumented() expects str arguments, got int for X
    """
    _require_strings(X, Y, "lcs_instrumented")

    counter = CallCounter()

    if variant == "recursive":
        length = _lcs_recursive_counted(X, Y, len(X), len(Y), counter)
    elif variant == "memo":
        memo: Dict[Tuple[int, int], int] = {}
        length = _lcs_memo_counted(X, Y, len(X), len(Y), memo, counter)
    elif variant == "tab":
        length = _lcs_tab_counted(X, Y, counter)
    else:
        raise ValueError(
            f"unknown variant {variant!r}; "
            "expected 'recursive', 'memo' or 'tab'"
        )

    return length, counter


#: The three length-only implementations, keyed by the variant names the
#: benchmark and the instrumented runner use. Keeping them in one mapping
#: is what lets the benchmark loop over variants instead of naming each
#: function, and what keeps those names spelled the same everywhere.
VARIANTS: Dict[str, Callable[[str, str], int]] = {
    "recursive": lcs_recursive,
    "memo": lcs_memo,
    "tab": lcs_tab,
}
