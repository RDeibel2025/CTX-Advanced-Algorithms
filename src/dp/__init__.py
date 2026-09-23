"""Dynamic programming: Fibonacci, 0/1 knapsack and longest common subsequence.

Three classical problems, each written three ways - plain recursion,
top-down memoization and bottom-up tabulation - so the same recurrence can
be measured under all three regimes.

The three memoized and three tabulated versions compute exactly the same
subproblems. They differ only in who decides the order those subproblems
are solved in: top-down discovers the order lazily as it recurses,
bottom-up fixes it in advance and fills the table. Every trade-off between
them, from constant factors to peak memory to which stack runs out first,
follows from that one difference.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""
