"""Matrix chain multiplication, written three ways over one interval recurrence.

Given a chain of matrices ``A1 A2 ... An`` where matrix ``Ai`` has shape
``p[i-1] x p[i]``, choose the order of the ``n - 1`` multiplications that
minimises the number of scalar multiplications. Matrix multiplication is
associative, so every parenthesization yields the same product, but the
costs differ enormously: multiplying a ``p x q`` matrix by a ``q x r``
matrix costs ``p * q * r`` scalar multiplications, and a bad order can be
orders of magnitude more expensive than a good one.

The notation follows CLRS 4e section 14.2 throughout, including its
1-indexing. Writing ``m[i][j]`` for the cheapest way to compute the
product ``Ai..Aj``::

    m[i][j] = 0                                                  if i == j
    m[i][j] = min over i <= k < j of
                  m[i][k] + m[k+1][j] + p[i-1] * p[k] * p[j]     if i < j

The split point ``k`` says the last multiplication performed is
``(Ai..Ak)(Ak+1..Aj)``. The recurrence is an *interval* recurrence: every
subproblem is a contiguous subchain ``i..j``, so there are only
``n * (n + 1) / 2`` of them, while the number of parenthesizations is the
Catalan number ``C(n-1)``, which grows like ``4^n / n^1.5``.

This module is shaped as the Week 6 contrast to the space-optimized
knapsack and Floyd-Warshall. Those two tables have a dimension that can be
dropped because each layer is read only by the next. Here nothing can be
dropped: ``m[i][j]`` reads every shorter interval inside ``i..j``, from
length 1 up to length ``j - i``, so the whole triangle must stay alive
until the end. The table is filled by *increasing chain length*, which is
the order that guarantees every interval a cell reads is already written.

====================== ================ ==================================
Function               Time             Space
====================== ================ ==================================
mcm_recursive          Theta(3^n)       O(n) call stack
mcm_memoized           O(n^3)           O(n^2) memo + O(n) call stack
mcm_bottom_up          O(n^3)           O(n^2) for the m and s tables
====================== ================ ==================================

:func:`optimal_parenthesization` is CLRS PRINT-OPTIMAL-PARENS, returning a
string instead of printing, and :func:`matrix_chain_order` ties the table
and the string together.

Contract shared by every public function that takes ``p``:

* ``p`` is a sequence of at least two entries (``ValueError`` otherwise),
  describing ``n = len(p) - 1`` matrices.
* Every entry is a positive ``int``. Non-``int`` entries, ``bool``
  included, raise ``TypeError``; zero or negative entries raise
  ``ValueError``. A dimension of 0 is not a matrix.
* A single matrix (``len(p) == 2``) costs 0 and is written ``"A1"``.
* All three cost variants agree on every valid input.

Examples:
    >>> p = [30, 35, 15, 5, 10, 20, 25]
    >>> mcm_recursive(p), mcm_memoized(p), mcm_bottom_up(p)[0][1][6]
    (15125, 15125, 15125)
    >>> matrix_chain_order(p)
    (15125, '((A1(A2A3))((A4A5)A6))')

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

__all__ = [
    "matrix_chain_order",
    "mcm_bottom_up",
    "mcm_memoized",
    "mcm_recursive",
    "optimal_parenthesization",
]


def _validate_dimensions(p: Sequence[int]) -> List[int]:
    """Copy the dimension list ``p`` into a list after checking every entry.

    Validation runs once at the public boundary so the private recursive
    workers below can trust their arguments; they are called once per
    subproblem, or once per *path* in the plain recursion, and re-checking
    there would be measured along with the algorithm.

    Args:
        p: The dimension sequence, where matrix ``Ai`` is ``p[i-1] x p[i]``.

    Returns:
        A new list holding the same entries.

    Raises:
        TypeError: If ``p`` is not iterable, or any entry is not an ``int``.
            ``bool`` is treated as a wrong type.
        ValueError: If ``p`` has fewer than two entries, or any entry is
            zero or negative.

    Time Complexity:
        O(n) - one pass over the entries.

    Space Complexity:
        O(n) - the returned copy.

    Examples:
        >>> _validate_dimensions((10, 20, 30))
        [10, 20, 30]
        >>> _validate_dimensions([10])
        Traceback (most recent call last):
            ...
        ValueError: p must have at least 2 entries, got 1
        >>> _validate_dimensions([10, 2.5])
        Traceback (most recent call last):
            ...
        TypeError: p[1] must be an int, got float
        >>> _validate_dimensions([10, True])
        Traceback (most recent call last):
            ...
        TypeError: p[1] must be an int, got bool
        >>> _validate_dimensions([10, 0, 5])
        Traceback (most recent call last):
            ...
        ValueError: p[1] must be > 0, got 0
        >>> _validate_dimensions(7)
        Traceback (most recent call last):
            ...
        TypeError: p must be a sequence of ints, got int
    """
    try:
        result = list(p)
    except TypeError:
        raise TypeError(
            f"p must be a sequence of ints, got {type(p).__name__}"
        ) from None

    if len(result) < 2:
        raise ValueError(f"p must have at least 2 entries, got {len(result)}")

    for index, entry in enumerate(result):
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(
                f"p[{index}] must be an int, got {type(entry).__name__}"
            )
        if entry <= 0:
            raise ValueError(f"p[{index}] must be > 0, got {entry}")

    return result


def _recursive(p: List[int], i: int, j: int) -> int:
    """Solve ``m[i][j]`` by trying every split, remembering nothing.

    The private worker behind :func:`mcm_recursive`. It is the recurrence
    transcribed literally: for each split ``k`` it solves both halves from
    scratch, so a subchain shared by many splits is re-solved every time
    it appears.

    Args:
        p: Validated dimension list.
        i: First matrix of the subchain, 1-indexed.
        j: Last matrix of the subchain, 1-indexed, with ``i <= j``.

    Returns:
        The minimum scalar multiplication count for ``Ai..Aj``.

    Time Complexity:
        Theta(3^(j-i)) - the call count ``T(L)`` for a chain of ``L``
        matrices satisfies ``T(L) = 1 + 2 * (T(1) + ... + T(L-1))``, which
        solves to exactly ``3^(L-1)``.

    Space Complexity:
        O(j - i) - the call stack; each level shortens the interval.

    Examples:
        >>> _recursive([10, 20, 30, 40], 1, 3)
        18000
        >>> _recursive([10, 20, 30, 40], 2, 2)
        0
    """
    if i == j:
        return 0

    best: Optional[int] = None
    for k in range(i, j):
        cost = (
            _recursive(p, i, k)
            + _recursive(p, k + 1, j)
            + p[i - 1] * p[k] * p[j]
        )
        if best is None or cost < best:
            best = cost

    assert best is not None  # i < j, so the loop ran at least once
    return best


def _memoized(
    p: List[int],
    i: int,
    j: int,
    memo: List[List[Optional[int]]],
) -> int:
    """Solve ``m[i][j]`` top-down, caching every interval in ``memo``.

    The private worker behind :func:`mcm_memoized`. The cache is an
    explicit ``(n + 1) x (n + 1)`` list of lists, 1-indexed exactly like
    the CLRS tables, with ``None`` meaning "not solved yet". It is passed
    down the recursion rather than kept in a decorator or a module-level
    global, so each public call owns its own cache.

    Args:
        p: Validated dimension list.
        i: First matrix of the subchain, 1-indexed.
        j: Last matrix of the subchain, 1-indexed, with ``i <= j``.
        memo: The cache, indexed ``memo[i][j]``. Mutated in place.

    Returns:
        The minimum scalar multiplication count for ``Ai..Aj``.

    Time Complexity:
        O(n^3) - each of the O(n^2) intervals is solved once, with an O(n)
        loop over its split points.

    Space Complexity:
        O(n^2) for the cache, plus O(n) of call stack.

    Examples:
        >>> table = [[None] * 4 for _ in range(4)]
        >>> _memoized([10, 20, 30, 40], 1, 3, table)
        18000
        >>> table[1][2], table[2][3]
        (6000, 24000)
    """
    cached = memo[i][j]
    if cached is not None:
        return cached

    if i == j:
        memo[i][j] = 0
        return 0

    best: Optional[int] = None
    for k in range(i, j):
        cost = (
            _memoized(p, i, k, memo)
            + _memoized(p, k + 1, j, memo)
            + p[i - 1] * p[k] * p[j]
        )
        if best is None or cost < best:
            best = cost

    assert best is not None  # i < j, so the loop ran at least once
    memo[i][j] = best
    return best


def mcm_recursive(p: Sequence[int]) -> int:
    """Return the minimum chain cost by plain recursion over every split.

    No memo of any kind: this is CLRS RECURSIVE-MATRIX-CHAIN, the baseline
    that shows why dynamic programming is needed. The search space it
    stands in for is the ``C(n-1)`` Catalan-number many parenthesizations,
    about ``4^n / n^1.5``. The recursion does not enumerate those one by
    one, because it optimises the two sides of each split independently,
    but it re-solves every shared subchain from scratch and so still makes
    exactly ``3^(n-1)`` calls. That is unusable past roughly 15 matrices.

    Args:
        p: Dimension list; matrix ``Ai`` is ``p[i-1] x p[i]``. Not modified.

    Returns:
        The minimum number of scalar multiplications for ``A1..An``.

    Raises:
        TypeError: If ``p`` is not a sequence of ``int``.
        ValueError: If ``p`` has fewer than two entries or any entry is not
            positive.

    Time Complexity:
        Theta(3^n) in every case - exponential. CLRS 4e section 14.3 proves
        the weaker Omega(2^n); the exact call count is ``3^(n-1)``. The
        loop bounds never depend on the dimension values, so best, average
        and worst coincide.

    Space Complexity:
        O(n) - the call stack only.

    Examples:
        >>> mcm_recursive([30, 35, 15, 5, 10, 20, 25])
        15125
        >>> mcm_recursive([10, 20, 30])
        6000
        >>> mcm_recursive([2, 3])
        0
        >>> mcm_recursive([5])
        Traceback (most recent call last):
            ...
        ValueError: p must have at least 2 entries, got 1
    """
    dims = _validate_dimensions(p)
    return _recursive(dims, 1, len(dims) - 1)


def mcm_memoized(p: Sequence[int]) -> int:
    """Return the minimum chain cost top-down with an explicit 2D memo.

    Structurally identical to :func:`mcm_recursive`, with one table in
    front of it: CLRS MEMOIZED-MATRIX-CHAIN. Before solving an interval the
    recursion looks it up; after solving it, it stores it. The
    ``3^(n-1)`` calls collapse onto the ``n * (n + 1) / 2`` distinct
    intervals. The memo is a 1-indexed list of lists created fresh on each
    call, not ``functools.lru_cache``, so nothing survives between calls.

    The recursion depth is at most ``n``, because each level shortens the
    interval by at least one. That is far below CPython's default limit of
    1000 for any chain this course benchmarks.

    Args:
        p: Dimension list; matrix ``Ai`` is ``p[i-1] x p[i]``. Not modified.

    Returns:
        The minimum number of scalar multiplications for ``A1..An``.

    Raises:
        TypeError: If ``p`` is not a sequence of ``int``.
        ValueError: If ``p`` has fewer than two entries or any entry is not
            positive.

    Time Complexity:
        O(n^3) - O(n^2) intervals, each with an O(n) loop over splits.
        Unlike knapsack, every interval is reachable from ``(1, n)``, so
        laziness saves nothing over the bottom-up fill.

    Space Complexity:
        O(n^2) for the memo, plus O(n) of call stack.

    Examples:
        >>> mcm_memoized([30, 35, 15, 5, 10, 20, 25])
        15125
        >>> mcm_memoized([10, 20, 30])
        6000
        >>> mcm_memoized([2, 3])
        0

        It agrees with the exhaustive recursion:

        >>> p = [5, 10, 3, 12, 5, 50, 6]
        >>> mcm_memoized(p) == mcm_recursive(p)
        True
    """
    dims = _validate_dimensions(p)
    n = len(dims) - 1
    memo: List[List[Optional[int]]] = [[None] * (n + 1) for _ in range(n + 1)]
    return _memoized(dims, 1, n, memo)


def mcm_bottom_up(p: Sequence[int]) -> Tuple[List[List[int]], List[List[int]]]:
    """Fill the CLRS ``m`` and ``s`` tables bottom-up by increasing length.

    This is CLRS 4e MATRIX-CHAIN-ORDER (section 14.2). Both tables are
    ``(n + 1) x (n + 1)`` and **1-indexed** as in the text: row 0 and
    column 0 are unused padding, ``m[i][j]`` for ``1 <= i <= j <= n`` is
    the minimum cost of ``Ai..Aj``, and ``s[i][j]`` for ``i < j`` is the
    split ``k`` that achieves it. Cells outside that triangle, and the
    diagonal of ``s``, are left at 0.

    The order is the whole point. Three nested loops run over increasing
    chain length, then start index, then split point. Every interval read
    by ``m[i][j]``, namely ``m[i][k]`` and ``m[k+1][j]``, is strictly
    shorter than ``i..j``, so filling by length guarantees each one is
    already final when it is read.

    Args:
        p: Dimension list; matrix ``Ai`` is ``p[i-1] x p[i]``. Not modified.

    Returns:
        The pair ``(m, s)``. The optimal cost of the whole chain is
        ``m[1][n]``, and :func:`optimal_parenthesization` turns ``s`` into
        the order that achieves it.

    Raises:
        TypeError: If ``p`` is not a sequence of ``int``.
        ValueError: If ``p`` has fewer than two entries or any entry is not
            positive.

    Time Complexity:
        O(n^3) in every case - about ``n^3 / 6`` split evaluations, each
        O(1). The loop bounds depend only on ``n``.

    Space Complexity:
        O(n^2) - the two tables. Neither can be collapsed, because the
        longest interval reads intervals of every shorter length.

    Examples:
        The CLRS figure 14.5 instance:

        >>> m, s = mcm_bottom_up([30, 35, 15, 5, 10, 20, 25])
        >>> m[1][6], m[2][5], s[1][6], s[2][5]
        (15125, 7125, 3, 3)
        >>> for row in m[1:]:
        ...     print(row[1:])
        [0, 15750, 7875, 9375, 11875, 15125]
        [0, 0, 2625, 4375, 7125, 10500]
        [0, 0, 0, 750, 2500, 5375]
        [0, 0, 0, 0, 1000, 3500]
        [0, 0, 0, 0, 0, 5000]
        [0, 0, 0, 0, 0, 0]

        A single matrix gives 2 x 2 tables of zeros:

        >>> mcm_bottom_up([2, 3])
        ([[0, 0], [0, 0]], [[0, 0], [0, 0]])
    """
    dims = _validate_dimensions(p)
    n = len(dims) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    # Loop 1 - chain length L, from 2 up to n. Length-1 chains cost 0 and
    # are already in place on the diagonal. Every interval a length-L cell
    # reads has length < L, so it was finished by an earlier iteration.
    for length in range(2, n + 1):
        # Loop 2 - start index i; the chain is Ai..Aj with j = i + L - 1.
        for i in range(1, n - length + 2):
            j = i + length - 1
            best: Optional[int] = None
            best_split = i
            # Loop 3 - split point k: the last product is (Ai..Ak)(Ak+1..Aj).
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i - 1] * dims[k] * dims[j]
                if best is None or cost < best:
                    best = cost
                    best_split = k
            assert best is not None  # length >= 2, so k took a value
            m[i][j] = best
            s[i][j] = best_split

    return m, s


def optimal_parenthesization(s: Sequence[Sequence[int]], i: int, j: int) -> str:
    """Render the optimal order of ``Ai..Aj`` from the split table ``s``.

    CLRS PRINT-OPTIMAL-PARENS, returning the string instead of printing
    it. A single matrix is written ``"Ai"``; otherwise the result is
    ``"(" + left + right + ")"`` around the split ``k = s[i][j]``. There
    are no spaces, so ``"((A1(A2A3))((A4A5)A6))"`` is the shape.

    The CLRS procedure is recursive. This version walks an explicit stack
    instead, so a maximally lopsided chain cannot hit the interpreter's
    recursion limit; the output is identical.

    Args:
        s: A 1-indexed split table as returned by :func:`mcm_bottom_up`.
        i: First matrix of the subchain, 1-indexed.
        j: Last matrix of the subchain, 1-indexed.

    Returns:
        The fully parenthesized product, for example ``"(A1A2)"``.

    Raises:
        TypeError: If ``i`` or ``j`` is not an ``int`` (``bool`` included).
        ValueError: Unless ``1 <= i <= j <= len(s) - 1``.

    Time Complexity:
        O(j - i) - one visit per matrix and one per split, each O(1).

    Space Complexity:
        O(j - i) - the stack and the output pieces.

    Examples:
        >>> _, s = mcm_bottom_up([30, 35, 15, 5, 10, 20, 25])
        >>> optimal_parenthesization(s, 1, 6)
        '((A1(A2A3))((A4A5)A6))'
        >>> optimal_parenthesization(s, 2, 5)
        '((A2A3)(A4A5))'
        >>> optimal_parenthesization(s, 4, 4)
        'A4'
        >>> optimal_parenthesization(s, 3, 2)
        Traceback (most recent call last):
            ...
        ValueError: need 1 <= i <= j <= 6, got i=3, j=2
    """
    for name, value in (("i", i), ("j", j)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    n = len(s) - 1
    if not 1 <= i <= j <= n:
        raise ValueError(f"need 1 <= i <= j <= {n}, got i={i}, j={j}")

    pieces: List[str] = []
    # Each stack entry is either a literal ")" still to be emitted or an
    # interval still to be expanded. Pushing in reverse order makes the
    # pops come out left to right.
    stack: List[Union[str, Tuple[int, int]]] = [(i, j)]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            pieces.append(item)
            continue
        low, high = item
        if low == high:
            pieces.append(f"A{low}")
            continue
        k = s[low][high]
        pieces.append("(")
        stack.append(")")
        stack.append((k + 1, high))
        stack.append((low, k))

    return "".join(pieces)


def matrix_chain_order(p: Sequence[int]) -> Tuple[int, str]:
    """Return the optimal cost and its parenthesization for the chain ``p``.

    The convenience entry point: runs :func:`mcm_bottom_up` once and reads
    both answers out of the same pair of tables. When several orders tie
    for the minimum, the one returned is the one with the smallest split
    at each level, because the split loop keeps the first minimum it
    finds.

    Args:
        p: Dimension list; matrix ``Ai`` is ``p[i-1] x p[i]``. Not modified.

    Returns:
        ``(cost, parenthesization)``, where ``cost == m[1][n]`` and the
        string is in the ``"((A1A2)A3)"`` style with no spaces.

    Raises:
        TypeError: If ``p`` is not a sequence of ``int``.
        ValueError: If ``p`` has fewer than two entries or any entry is not
            positive.

    Time Complexity:
        O(n^3) - dominated by the table fill; the string is O(n).

    Space Complexity:
        O(n^2) - the two tables.

    Examples:
        >>> matrix_chain_order([30, 35, 15, 5, 10, 20, 25])
        (15125, '((A1(A2A3))((A4A5)A6))')
        >>> matrix_chain_order([10, 20, 30])
        (6000, '(A1A2)')
        >>> matrix_chain_order([2, 3])
        (0, 'A1')

        Order matters: for 10 x 100, 100 x 5 and 5 x 50, multiplying the
        first pair first costs 7500, the other order costs 75000.

        >>> matrix_chain_order([10, 100, 5, 50])
        (7500, '((A1A2)A3)')
        >>> matrix_chain_order([10, -1])
        Traceback (most recent call last):
            ...
        ValueError: p[1] must be > 0, got -1
    """
    dims = _validate_dimensions(p)
    n = len(dims) - 1
    m, s = mcm_bottom_up(dims)
    return m[1][n], optimal_parenthesization(s, 1, n)
