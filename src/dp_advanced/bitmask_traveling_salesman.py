"""The traveling salesman problem, by Held-Karp bitmask DP and by brute force.

Given n cities and a matrix ``dist`` in which ``dist[i][j]`` is the cost of
travelling from city ``i`` to city ``j``, find the cheapest tour that leaves
a fixed start city, visits every other city exactly once and returns to the
start. The matrix may be asymmetric, so a tour is directed: the same cycle
driven the other way round can cost a different amount, and both functions
here treat it that way.

Brute force fixes the start and tries every ordering of the other n - 1
cities, which is (n - 1)! tours. Held and Karp (1962) observed that almost
all of that work is repeated. The cheapest way to *finish* a tour depends
only on which cities have been visited and where the salesman stands now,
not on the order in which the visited cities were reached. So the state is
a pair (set of visited cities, current city), and there are only n * 2^n of
those. Writing ``dp[S][j]`` for the cheapest path that starts at ``start``,
visits exactly the cities in ``S`` and ends at ``j``::

    dp[{start}][start] = 0
    dp[S][j] = min over i in S - {j} of  dp[S - {j}][i] + dist[i][j]
    best     = min over j != start   of  dp[ALL][j] + dist[j][start]

The set ``S`` is stored as an integer bitmask: bit ``j`` is set when city
``j`` has been visited. A set then becomes a list index, membership becomes
a shift and a mask (``mask >> j & 1``), adding a city becomes an OR
(``mask | (1 << j)``), and the set of all cities is ``(1 << n) - 1``. The
helpers in :mod:`src.utils.bitmask_utils` name those operations and are used
for the setup and the reconstruction; the hot inner loop writes the same
shifts out inline, because a function call per transition would cost more
than the transition itself.

The shape of that state space decides how the module is written:

* **No layer can be discarded.** A mask with k cities is built from masks
  with k - 1 cities, and the answer reads the full mask, so every subset
  has to stay available until the end. Unlike the knapsack row, this table
  cannot lose a dimension; it can only be filled in an order in which every
  predecessor is ready first. Increasing numeric order of the mask is such
  an order, because adding a city to a set always makes the integer larger.
* **Only masks that contain the start city are reachable.** Every path
  begins at ``start``, so a mask without the start bit never holds a finite
  value. The DP iterates only the masks that contain it - for start 0,
  exactly the odd masks - which halves the work outright.

=================== ================ ======================================
Function            Time             Space
=================== ================ ======================================
tsp_brute_force     O(n!)            O(n) - one permutation at a time
tsp_bitmask         O(n^2 * 2^n)     O(n * 2^n) - the dp and parent tables
=================== ================ ======================================

Both are exponential, but the gap is enormous. At n = 12, n^2 * 2^n is
about 590 thousand while 11! is 39.9 million; at n = 18 the DP's bound of
85 million transitions compares with 17!, about 3.6 * 10^14 orderings. The
price of the DP is memory, which grows as n * 2^n, so :func:`tsp_bitmask`
refuses inputs above ``max_n`` cities unless the caller raises the limit.

Contract shared by both public functions:

* ``dist`` is a square n x n matrix of non-negative real numbers, n >= 1,
  given as a list of lists or anything that iterates as rows. ``bool`` is
  rejected as a type error. ``math.inf`` marks a missing edge. The diagonal
  is never read, because a tour never moves from a city to itself.
* ``start`` is an ``int`` in ``range(n)``.
* The result is ``(cost, tour)``. ``tour`` starts and ends at ``start`` and
  visits every other city exactly once in between, and the sum of its edges
  is ``cost``. ``n == 1`` gives ``(0, [start])`` and ``n == 2`` gives the
  single round trip ``(d[s][o] + d[o][s], [s, o, s])``.
* Where several tours tie, any of them is a correct answer. If no tour has
  a finite cost, every tour ties at ``inf``, and both functions return
  ``inf`` with the tour that visits the other cities in index order.

Examples:
    >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25],
    ...         [15, 35, 0, 30], [20, 25, 30, 0]]
    >>> tsp_bitmask(dist)[0], tsp_brute_force(dist)[0]
    (80, 80)

References:
    Held, M., & Karp, R. M. (1962). A dynamic programming approach to
    sequencing problems. *Journal of the Society for Industrial and Applied
    Mathematics, 10*(1), 196-210.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
from itertools import permutations
from typing import Iterator, List, Optional, Sequence, Tuple

from src.utils.bitmask_utils import clear_bit, full_mask, set_bit

__all__ = [
    "tsp_bitmask",
    "tsp_brute_force",
]

#: Cost of an unreached DP state, and of a missing edge in ``dist``.
INF = math.inf


def _validate(dist: Sequence[Sequence[float]], start: int) -> List[List[float]]:
    """Check ``dist`` and ``start`` and copy the matrix into plain lists.

    Both public functions share these checks, so the error messages are
    identical across the module and each algorithm body stays focused on
    its own search. The copy is a fresh list of lists of Python ``int`` and
    ``float``, which means the caller's matrix is never mutated and the hot
    loops index plain lists rather than, say, numpy scalars.

    Args:
        dist: The candidate distance matrix.
        start: The candidate start city.

    Returns:
        The validated matrix as a new list of lists. Integral entries become
        ``int`` and other real entries become ``float``.

    Raises:
        TypeError: If ``dist`` does not iterate as rows, if any entry is not
            a real number (``bool`` included), or if ``start`` is not an
            ``int``.
        ValueError: If ``dist`` is empty or not square, if any entry is
            negative or NaN, or if ``start`` is outside ``range(n)``.

    Time Complexity:
        O(n^2) - one check per entry.

    Space Complexity:
        O(n^2) - the copy.

    Examples:
        >>> _validate(((0, 2.5), (1, 0)), 1)
        [[0, 2.5], [1, 0]]
        >>> _validate([], 0)
        Traceback (most recent call last):
            ...
        ValueError: dist must have at least one city
        >>> _validate([[0, 1], [1]], 0)
        Traceback (most recent call last):
            ...
        ValueError: dist must be square: row 1 has 1 entries, expected 2
        >>> _validate([[0, -1], [1, 0]], 0)
        Traceback (most recent call last):
            ...
        ValueError: dist[0][1] must be >= 0, got -1
        >>> _validate([[0, True], [1, 0]], 0)
        Traceback (most recent call last):
            ...
        TypeError: dist[0][1] must be a real number, got bool
        >>> _validate([1, 2], 0)
        Traceback (most recent call last):
            ...
        TypeError: dist must be a square matrix of numbers, got a row of type int
        >>> _validate([[0, 1], [1, 0]], 2)
        Traceback (most recent call last):
            ...
        ValueError: start must be in range(2), got 2
        >>> _validate([[0, 1], [1, 0]], 1.0)
        Traceback (most recent call last):
            ...
        TypeError: start must be an int, got float
    """
    try:
        raw_rows = list(dist)
    except TypeError:
        raise TypeError(
            f"dist must be a square matrix of numbers, got {type(dist).__name__}"
        ) from None

    rows: List[List[float]] = []
    for row in raw_rows:
        try:
            rows.append(list(row))
        except TypeError:
            raise TypeError(
                "dist must be a square matrix of numbers, got a row of "
                f"type {type(row).__name__}"
            ) from None

    n = len(rows)
    if n == 0:
        raise ValueError("dist must have at least one city")

    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(
                f"dist must be square: row {i} has {len(row)} entries, "
                f"expected {n}"
            )
        for j, weight in enumerate(row):
            if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                raise TypeError(
                    f"dist[{i}][{j}] must be a real number, "
                    f"got {type(weight).__name__}"
                )
            # ``not >=`` rather than ``<`` so that NaN is refused as well.
            if not weight >= 0:
                raise ValueError(f"dist[{i}][{j}] must be >= 0, got {weight}")
            if not isinstance(weight, int):
                row[j] = (
                    int(weight)
                    if isinstance(weight, numbers.Integral)
                    else float(weight)
                )

    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError(f"start must be an int, got {type(start).__name__}")
    if not 0 <= start < n:
        raise ValueError(f"start must be in range({n}), got {start}")

    return rows


def _trivial_tour(
    rows: List[List[float]], start: int
) -> Optional[Tuple[float, List[int]]]:
    """Answer the one- and two-city instances directly, or return ``None``.

    With one city there is nothing to visit, so the tour is the start alone
    and costs 0. With two there is exactly one tour, out and back. Neither
    needs a search, and handling them here keeps both algorithms free of
    special cases.

    Args:
        rows: A validated distance matrix.
        start: A validated start city.

    Returns:
        ``(cost, tour)`` when ``n <= 2``, otherwise ``None``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _trivial_tour([[7]], 0)
        (0, [0])
        >>> _trivial_tour([[0, 4], [9, 0]], 1)
        (13, [1, 0, 1])
        >>> _trivial_tour([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0) is None
        True
    """
    n = len(rows)
    if n == 1:
        return 0, [start]
    if n == 2:
        other = 1 - start
        return rows[start][other] + rows[other][start], [start, other, start]
    return None


def _index_order_tour(n: int, start: int) -> List[int]:
    """Return the tour that visits the non-start cities in index order.

    Used only when no tour has a finite cost. Every tour then ties at
    ``inf``, so this one is as optimal as any other, and returning a real
    tour keeps the ``(cost, tour)`` contract intact.

    Args:
        n: Number of cities.
        start: The start city.

    Returns:
        ``[start, c1, c2, ..., start]`` with the other cities ascending.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n).

    Examples:
        >>> _index_order_tour(4, 2)
        [2, 0, 1, 3, 2]
    """
    return [start, *(city for city in range(n) if city != start), start]


def _masks_containing(n: int, start: int) -> Iterator[int]:
    """Yield every n-bit mask that contains ``start``, in increasing order.

    Each of the 2^(n-1) masks over the *other* cities is widened by
    inserting the start bit at position ``start``: the bits below it stay
    put and the bits at or above it move up by one. That map preserves
    numeric order, so the masks come out ascending, which is the order the
    DP needs. For start 0 the result is exactly the odd numbers below 2^n.

    Args:
        n: Number of cities, at least 1.
        start: The bit every yielded mask must contain.

    Yields:
        The masks, ascending.

    Time Complexity:
        O(2^n) in total, O(1) per mask.

    Space Complexity:
        O(1) - a generator.

    Examples:
        >>> list(_masks_containing(3, 0))
        [1, 3, 5, 7]
        >>> list(_masks_containing(3, 1))
        [2, 3, 6, 7]
        >>> list(_masks_containing(3, 2))
        [4, 5, 6, 7]
    """
    start_bit = set_bit(0, start)
    below = start_bit - 1
    for rest in range(1 << (n - 1)):
        yield ((rest & ~below) << 1) | start_bit | (rest & below)


def _reconstruct(parent: List[int], n: int, start: int, last: int) -> List[int]:
    """Walk the parent table back from ``last`` and return the closed tour.

    ``parent[mask * n + j]`` is the city visited just before ``j`` on the
    cheapest path that covers ``mask`` and ends at ``j``. Starting from the
    full mask, the walk repeatedly records the current city, steps to its
    parent and removes the current city from the mask, until it reaches the
    start. That produces the path backwards; reversing it and appending the
    start closes the tour.

    Args:
        parent: The flat parent table filled by :func:`tsp_bitmask`.
        n: Number of cities.
        start: The start city.
        last: The final city before returning to ``start``.

    Returns:
        The tour, starting and ending at ``start``.

    Time Complexity:
        O(n) - one step per city.

    Space Complexity:
        O(n) - the tour.

    Examples:
        Three cities, start 0, with the path 0 -> 2 -> 1 recorded:

        >>> parent = [-1] * (8 * 3)
        >>> parent[0b111 * 3 + 1] = 2
        >>> parent[0b101 * 3 + 2] = 0
        >>> _reconstruct(parent, 3, 0, 1)
        [0, 2, 1, 0]
    """
    path: List[int] = []
    mask = full_mask(n)
    city = last
    while city != start:
        path.append(city)
        previous = parent[mask * n + city]
        mask = clear_bit(mask, city)
        city = previous
    path.append(start)
    path.reverse()
    path.append(start)
    return path


def tsp_bitmask(
    dist: Sequence[Sequence[float]],
    start: int = 0,
    max_n: int = 20,
) -> Tuple[float, List[int]]:
    """Solve TSP exactly with the Held-Karp bitmask dynamic program.

    The table ``dp`` holds, for every pair (mask, city), the cheapest path
    that starts at ``start``, visits exactly the cities in the mask and ends
    at the city; ``parent`` holds the city before it on that path. Both are
    flat lists of length ``n * 2^n`` indexed ``mask * n + city``, with
    ``math.inf`` marking a state not yet reached. Masks without the start
    bit are allocated but never touched, which keeps the indexing a single
    multiply.

    The table is filled forwards. Masks are visited in increasing order, and
    only those that contain the start city, so half of them are skipped
    outright. For each reachable ``dp[mask][i]`` the loop tries every city
    ``j`` not yet in the mask and relaxes ``dp[mask | (1 << j)][j]``. By the
    time a mask is visited every smaller mask has been, so its values are
    final. Three constant-factor measures keep the n^2 * 2^n inner loop
    tight: the unvisited cities and their target cells are listed once per
    mask, unreachable states are skipped before the inner loop starts, and
    each city's row of ``dist`` is fetched once per state.

    The answer closes the tour, taking the cheapest ``dp[ALL][j] +
    dist[j][start]``, and the parent table is walked back from that ``j``.

    Args:
        dist: Square matrix of non-negative edge costs; ``dist[i][j]`` is
            the cost of going from ``i`` to ``j``. May be asymmetric.
            ``math.inf`` marks a missing edge. Not modified.
        start: The city the tour begins and ends at.
        max_n: The largest number of cities accepted. The tables hold
            ``n * 2^n`` cells each, so the default of 20 already means about
            21 million cells per table; raise it deliberately or not at all.

    Returns:
        A ``(cost, tour)`` pair. ``tour`` starts and ends at ``start``,
        visits every other city once, and its edges sum to ``cost``.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers, or if
            ``start`` or ``max_n`` is not an ``int``.
        ValueError: If ``dist`` is empty, not square, or holds a negative
            or NaN entry, if ``start`` is out of range, or if ``n`` exceeds
            ``max_n``.

    Time Complexity:
        O(n^2 * 2^n) - n * 2^n states, each relaxed against up to n
        successors. Only the 2^(n-1) masks containing the start are
        visited, which halves the constant but not the order.

    Space Complexity:
        O(n * 2^n) - the ``dp`` and ``parent`` tables. No layer can be
        dropped, because the final step reads the full mask and every mask
        is built from all of its one-smaller subsets.

    Examples:
        The four-city anchor. It is symmetric, so a tour and its reverse
        both cost 80; whichever is returned must be a valid tour of that
        cost:

        >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25],
        ...         [15, 35, 0, 30], [20, 25, 30, 0]]
        >>> cost, tour = tsp_bitmask(dist)
        >>> cost
        80
        >>> tour[0] == tour[-1] == 0, sorted(tour[:-1]) == [0, 1, 2, 3]
        (True, True)
        >>> sum(dist[a][b] for a, b in zip(tour, tour[1:]))
        80

        A different start city gives the same cost:

        >>> cost, tour = tsp_bitmask(dist, start=2)
        >>> cost, tour[0], tour[-1]
        (80, 2, 2)

        One and two cities need no search:

        >>> tsp_bitmask([[0]])
        (0, [0])
        >>> tsp_bitmask([[0, 3], [5, 0]])
        (8, [0, 1, 0])

        An asymmetric instance, where direction matters. Going round
        0 -> 1 -> 2 costs 3 and the other way costs 30:

        >>> one_way = [[0, 1, 10], [10, 0, 1], [1, 10, 0]]
        >>> tsp_bitmask(one_way)
        (3, [0, 1, 2, 0])
        >>> tsp_bitmask(one_way) == tsp_brute_force(one_way)
        True

        Without a finite tour, every tour ties at infinity:

        >>> inf = float("inf")
        >>> tsp_bitmask([[0, 1, inf], [inf, 0, inf], [inf, 1, 0]])
        (inf, [0, 1, 2, 0])

        The memory guard, and its override:

        >>> tsp_bitmask([[0] * 21 for _ in range(21)])
        Traceback (most recent call last):
            ...
        ValueError: n = 21 exceeds max_n = 20: Held-Karp needs O(n * 2^n) \
memory (21 * 2^21 = 44040192 cells per table); pass a larger max_n to override
        >>> tsp_bitmask([[0, 1, 1], [1, 0, 1], [1, 1, 0]], max_n=2)[0]
        Traceback (most recent call last):
            ...
        ValueError: n = 3 exceeds max_n = 2: Held-Karp needs O(n * 2^n) \
memory (3 * 2^3 = 24 cells per table); pass a larger max_n to override
    """
    rows = _validate(dist, start)
    n = len(rows)
    if isinstance(max_n, bool) or not isinstance(max_n, int):
        raise TypeError(f"max_n must be an int, got {type(max_n).__name__}")
    if n > max_n:
        raise ValueError(
            f"n = {n} exceeds max_n = {max_n}: Held-Karp needs O(n * 2^n) "
            f"memory ({n} * 2^{n} = {n << n} cells per table); pass a larger "
            "max_n to override"
        )

    trivial = _trivial_tour(rows, start)
    if trivial is not None:
        return trivial

    full = full_mask(n)
    cells = (full + 1) * n
    dp: List[float] = [INF] * cells
    parent: List[int] = [-1] * cells
    dp[set_bit(0, start) * n + start] = 0
    cities = range(n)

    for mask in _masks_containing(n, start):
        if mask == full:
            break  # The largest mask, and nothing left to add to it.

        # The inline shifts below are the operations has_bit and set_bit in
        # src.utils.bitmask_utils document; they are written out here
        # because a call per transition would dominate the n^2 * 2^n loop.
        # Each entry pairs an unvisited city j with the flat index of
        # dp[mask | (1 << j)][j], computed once per mask, not once per state.
        unvisited = [
            (j, (mask | (1 << j)) * n + j) for j in cities if not mask >> j & 1
        ]

        base = mask * n
        for i, cost_i in enumerate(dp[base : base + n]):
            if cost_i == INF:
                continue  # Unreached state, or i not in mask: nothing to extend.
            row = rows[i]
            for j, cell in unvisited:
                candidate = cost_i + row[j]
                if candidate < dp[cell]:
                    dp[cell] = candidate
                    parent[cell] = i

    best: float = INF
    last = -1
    base = full * n
    for j in cities:
        if j != start:
            candidate = dp[base + j] + rows[j][start]
            if candidate < best:
                best, last = candidate, j

    if last < 0:
        return best, _index_order_tour(n, start)
    return best, _reconstruct(parent, n, start, last)


def tsp_brute_force(
    dist: Sequence[Sequence[float]],
    start: int = 0,
) -> Tuple[float, List[int]]:
    """Solve TSP exactly by trying every ordering of the non-start cities.

    The start is fixed, so the search covers the (n - 1)! orderings of the
    other cities that :func:`itertools.permutations` yields, in
    lexicographic order. Each tour is priced and the cheapest kept; on a
    tie the first one found stays.

    This is the baseline the DP is measured against, so the per-tour work
    is kept small: each tour is priced by one tight loop over the
    permutation tuple, carrying the current city's row of ``dist``, with no
    list built per tour. On CPython 3.12 that loop measured faster than a
    ``sum(map(...))`` over chained iterators, whose per-tour setup costs
    more than it saves at these lengths. The point is that the measured gap
    between the two functions is a property of the algorithms rather than
    of a careless loop.

    There is no size guard. Memory is O(n) at any size; time is the
    problem, and n = 12 already means 39.9 million tours.

    Args:
        dist: Square matrix of non-negative edge costs; ``dist[i][j]`` is
            the cost of going from ``i`` to ``j``. May be asymmetric.
            ``math.inf`` marks a missing edge. Not modified.
        start: The city the tour begins and ends at.

    Returns:
        A ``(cost, tour)`` pair with the same meaning as in
        :func:`tsp_bitmask`. The costs always agree; the tours may differ
        when several tie.

    Raises:
        TypeError: If ``dist`` is not a matrix of real numbers, or if
            ``start`` is not an ``int``.
        ValueError: If ``dist`` is empty, not square, or holds a negative
            or NaN entry, or if ``start`` is out of range.

    Time Complexity:
        O(n!) - (n - 1)! tours, each priced in O(n). There is no best case:
        every ordering is examined whatever the weights are.

    Space Complexity:
        O(n) - one permutation and the best tour so far. Nothing grows with
        the number of tours, which is the one thing brute force has going
        for it.

    Examples:
        >>> dist = [[0, 10, 15, 20], [10, 0, 35, 25],
        ...         [15, 35, 0, 30], [20, 25, 30, 0]]
        >>> tsp_brute_force(dist)
        (80, [0, 1, 3, 2, 0])
        >>> tsp_brute_force(dist)[0] == tsp_bitmask(dist)[0]
        True
        >>> tsp_brute_force([[0]], 0)
        (0, [0])
        >>> tsp_brute_force([[0, 3], [5, 0]], 1)
        (8, [1, 0, 1])

        Agreement with the DP on an asymmetric five-city instance:

        >>> uneven = [[0, 3, 9, 4, 7], [8, 0, 2, 6, 3], [5, 7, 0, 1, 9],
        ...           [2, 8, 6, 0, 4], [6, 1, 8, 5, 0]]
        >>> tsp_brute_force(uneven)[0] == tsp_bitmask(uneven)[0]
        True

        Validation is shared with :func:`tsp_bitmask`:

        >>> tsp_brute_force([[0, 1, 2], [1, 0, 3]])
        Traceback (most recent call last):
            ...
        ValueError: dist must be square: row 0 has 3 entries, expected 2
    """
    rows = _validate(dist, start)

    trivial = _trivial_tour(rows, start)
    if trivial is not None:
        return trivial

    n = len(rows)
    others = [city for city in range(n) if city != start]
    start_row = rows[start]

    best: float = INF
    best_order: Tuple[int, ...] = tuple(others)
    for order in permutations(others):
        # Carry the current city's row instead of its index, so each step
        # is one lookup into it and one fetch of the next row.
        cost = 0
        row = start_row
        for city in order:
            cost += row[city]
            row = rows[city]
        cost += row[start]
        if cost < best:
            best, best_order = cost, order

    return best, [start, *best_order, start]
