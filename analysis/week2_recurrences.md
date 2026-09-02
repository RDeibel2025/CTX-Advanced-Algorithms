# Master Theorem - Recurrence Relations

**Robert Deibel**  
CSC 5300 Advanced Algorithms · Concordia University Texas · Fall 2026  
Week 2 - Divide and Conquer Implementation Project

Repository: <https://github.com/RDeibel2025/CTX-Advanced-Algorithms>

Fourteen recurrences solved below. Ten are straightforward applications of
the Master Theorem; three are included precisely because the theorem does
**not** settle them (§9, §13 and §14), since knowing where a tool stops
working is part of knowing how to use it. The remaining one, §11, falls in
the gap between cases 2 and 3 and needs the *extended* case 2 rather than
the three-case statement. That is 10 + 3 + 1 = 14.

---

## The theorem, as used here

For a recurrence of the form

> **T(n) = a·T(n/b) + f(n)**, with a ≥ 1, b > 1

compare f(n) against the *watershed function* n^(log_b a). That exponent is
what the recursion costs on its own - it counts the leaves of the recursion
tree - and the three cases ask whether the work done splitting and
combining is smaller, equal, or larger than that.

| Case | Condition | Result |
|---|---|---|
| 1 | f(n) = O(n^(log_b a − ε)) for some ε > 0 | Θ(n^(log_b a)) - leaves dominate |
| 2 | f(n) = Θ(n^(log_b a) · log^k n), k ≥ 0 | Θ(n^(log_b a) · log^(k+1) n) - balanced |
| 3 | f(n) = Ω(n^(log_b a + ε)) for some ε > 0, **and** a·f(n/b) ≤ c·f(n) for some c < 1 and large n | Θ(f(n)) - the root dominates |

Two requirements are easy to skip past, and they are what §11 and §13 fall
foul of. Cases 1 and 3 need f(n) to differ from the watershed by a
*polynomial* factor - an n^ε gap, not merely a logarithmic one. And case 3
additionally needs the **regularity condition**, which is what rules out
pathologically oscillating f.

---

## 1. Merge sort - T(n) = 2T(n/2) + Θ(n)

The recurrence implemented in [`src/sorting/merge_sort.py`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/src/sorting/merge_sort.py):
two recursive calls on half the input, plus a linear merge.

| | |
|---|---|
| a | 2 |
| b | 2 |
| f(n) | Θ(n) |
| log_b a | log₂ 2 = **1** |
| watershed | n¹ = n |

f(n) = Θ(n) = Θ(n^(log_b a) · log⁰ n), so **case 2 with k = 0**.

> **T(n) = Θ(n log n)**

Every level of the recursion does Θ(n) work merging, and there are log₂ n
levels - the cost is spread evenly rather than concentrated at the root or
the leaves. Because the split is positional it is always balanced, so this
recurrence describes merge sort's best, average *and* worst case. The
measured doubling ratios of 2.03-2.20 tabulated in
[`week2_report.md`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/analysis/week2_report.md) §4 - one per data shape,
each the mean of the two doubling steps - are this result observed rather
than derived.

---

## 2. Binary search - T(n) = T(n/2) + Θ(1)

| | |
|---|---|
| a | 1 |
| b | 2 |
| f(n) | Θ(1) |
| log_b a | log₂ 1 = **0** |
| watershed | n⁰ = 1 |

f(n) = Θ(1) = Θ(n⁰ · log⁰ n), so **case 2 with k = 0**.

> **T(n) = Θ(log n)**

Only one subproblem survives each step, so the recursion tree is a path
rather than a tree. The Θ(1) comparison at each node is exactly the
watershed, and the log factor comes from the depth.

---

## 3. Tree traversal - T(n) = 2T(n/2) + Θ(1)

| | |
|---|---|
| a | 2 |
| b | 2 |
| f(n) | Θ(1) |
| log_b a | **1** |
| watershed | n |

Θ(1) is polynomially *smaller* than n - take ε = 1, giving
f(n) = O(n^(1−1)) = O(1). **Case 1.**

> **T(n) = Θ(n)**

The leaves dominate. This is the useful contrast with §1: the same 2T(n/2)
split costs Θ(n) when the per-node work is constant and Θ(n log n) when it
is linear. All the extra cost in merge sort comes from the merge, not from
the recursion.

---

## 4. Karatsuba multiplication - T(n) = 3T(n/2) + Θ(n)

| | |
|---|---|
| a | 3 |
| b | 2 |
| f(n) | Θ(n) |
| log_b a | log₂ 3 ≈ **1.585** |
| watershed | n^1.585 |

f(n) = Θ(n) = O(n^(1.585 − ε)) with ε = 0.585. **Case 1.**

> **T(n) = Θ(n^log₂3) ≈ Θ(n^1.585)**

Karatsuba's insight is arithmetic, not structural: the naive algorithm
needs four half-size multiplications, and it computes the same result with
three, using extra additions. Those additions are the Θ(n) term and they
are dominated. Reducing a from 4 to 3 is what beats Θ(n²).

---

## 5. The naive version - T(n) = 4T(n/2) + Θ(n)

| | |
|---|---|
| a | 4 |
| b | 2 |
| f(n) | Θ(n) |
| log_b a | log₂ 4 = **2** |
| watershed | n² |

f(n) = Θ(n) = O(n^(2 − ε)) with ε = 1. **Case 1.**

> **T(n) = Θ(n²)**

Worth solving next to §4 because the two differ only in a. Four recursive
calls buy nothing over schoolbook long multiplication; three beat it. The
recursion structure is identical, so the entire improvement is attributable
to that one parameter.

---

## 6. Strassen's matrix multiplication - T(n) = 7T(n/2) + Θ(n²)

| | |
|---|---|
| a | 7 |
| b | 2 |
| f(n) | Θ(n²) |
| log_b a | log₂ 7 ≈ **2.807** |
| watershed | n^2.807 |

f(n) = Θ(n²) = O(n^(2.807 − ε)) with ε ≈ 0.807. **Case 1.**

> **T(n) = Θ(n^log₂7) ≈ Θ(n^2.807)**

Same trade as Karatsuba, one dimension up. Eight submatrix products
(a = 8, log₂ 8 = 3) reproduce the Θ(n³) of the standard algorithm; Strassen
finds seven, and the extra matrix additions cost Θ(n²), which is
polynomially below the watershed and therefore free.

---

## 7. Standard matrix multiplication - T(n) = 8T(n/2) + Θ(n²)

| | |
|---|---|
| a | 8 |
| b | 2 |
| f(n) | Θ(n²) |
| log_b a | log₂ 8 = **3** |
| watershed | n³ |

f(n) = Θ(n²) = O(n^(3 − ε)) with ε = 1. **Case 1.**

> **T(n) = Θ(n³)**

The baseline §6 improves on, confirming that block recursion by itself buys
nothing.

---

## 8. Quicksort's balanced case - T(n) = 2T(n/2) + Θ(n)

Identical in form to §1, and worth stating separately because in quicksort
this recurrence is a *hope* rather than a guarantee. The partition is
Θ(n) and the split is even only when the pivot lands near the median.

| | |
|---|---|
| a | 2, b | 2, f(n) | Θ(n), log_b a = 1 |

**Case 2**, so:

> **T(n) = Θ(n log n)**

Randomized pivot selection does not make this recurrence hold; it makes it
hold *in expectation*, where §1 delivers it deterministically. That
distinction did not show up in the measured spread across input shapes -
merge sort varied by 1.51× at n = 10,000 and quicksort by 1.54×, which is
randomization delivering in practice what merge sort guarantees in
principle. Where it does show up is the tail: change the partition scheme
and quicksort reaches §9 instead, as the Lomuto study did.

---

## 9. Quicksort's worst case - T(n) = T(n−1) + Θ(n)

| | |
|---|---|
| a | 1 |
| b | - |
| f(n) | Θ(n) |

**The Master Theorem does not apply.** The theorem requires subproblems of
size n/b for a constant b > 1 - a *fixed fraction* of the input. Here the
subproblem shrinks by a constant *amount*, so there is no b to speak of.

Solve by direct expansion instead:

T(n) = n + (n−1) + (n−2) + … + 1 = n(n+1)/2

> **T(n) = Θ(n²)**

This is the recurrence that arises when every partition puts all remaining
elements on one side. Section 4 of the report records that it was never
observed for the production quicksort across any of the six data shapes -
but the Lomuto partition study reached it exactly, with a measured growth
ratio of 3.98 per doubling against the predicted 4.

---

## 10. Binary search variant - T(n) = 2T(n/2) + Θ(log n)

| | |
|---|---|
| a | 2 |
| b | 2 |
| f(n) | Θ(log n) |
| log_b a | **1** |
| watershed | n |

Θ(log n) is polynomially smaller than n: for ε = 0.5, log n = O(n^0.5).
**Case 1.**

> **T(n) = Θ(n)**

A useful check on intuition - a logarithmic combine step is asymptotically
invisible against a linear watershed, so this costs no more than §3 with
its Θ(1) combine.

---

## 11. Case 2 extended - T(n) = 2T(n/2) + Θ(n log n)

| | |
|---|---|
| a | 2 |
| b | 2 |
| f(n) | Θ(n log n) |
| log_b a | **1** |
| watershed | n |

This one needs care. f(n) = Θ(n log n) is **larger** than the watershed n,
which looks like case 3 - but case 3 requires f(n) = Ω(n^(1+ε)) for some
ε > 0, and n log n is *not* polynomially larger than n. It grows faster by
a logarithmic factor only, and log n = o(n^ε) for every ε > 0. So the
**three-case Master Theorem as usually stated does not cover this
recurrence**; it falls in the gap between cases 2 and 3.

The *extended* case 2 does cover it. In the form
f(n) = Θ(n^(log_b a) · log^k n) with k = 1:

> **T(n) = Θ(n^(log_b a) · log^(k+1) n) = Θ(n log² n)**

Confirmed by the recursion tree: log₂ n levels, and level i does
2^i · (n/2^i) · log(n/2^i) = n · log(n/2^i) work. Summing over
i = 0 … log n − 1 gives n · Σ(log n − i) = n · Θ(log² n).

The lesson is that "f is bigger than the watershed" is not enough for
case 3. The gap has to be polynomial.

---

## 12. Case 3, with the regularity condition checked - T(n) = 2T(n/2) + Θ(n²)

| | |
|---|---|
| a | 2 |
| b | 2 |
| f(n) | Θ(n²) |
| log_b a | **1** |
| watershed | n |

f(n) = Θ(n²) = Ω(n^(1+ε)) with ε = 1 - a genuine polynomial gap, unlike
§11. Case 3 also requires regularity, which most treatments assert and skip;
here it is checked:

> a·f(n/b) = 2·(n/2)² = 2·n²/4 = n²/2 = **0.5·f(n)** ≤ c·f(n) with c = 0.5 < 1 ✓

Both conditions hold. **Case 3.**

> **T(n) = Θ(n²)**

The root dominates so completely that the entire recursion below it is a
constant-factor detail. The total is a geometric series in which each level
costs half the one above, summing to at most 2·f(n).

---

## 13. Where the Master Theorem fails - T(n) = 2T(n/2) + n/log n

| | |
|---|---|
| a | 2 |
| b | 2 |
| f(n) | n / log n |
| log_b a | **1** |
| watershed | n |

f(n) = n/log n is **smaller** than n, which suggests case 1. But case 1
requires f(n) = O(n^(1−ε)) for some fixed ε > 0, and that fails: for any
ε > 0,

> lim (n→∞) [ n/log n ] / n^(1−ε) = lim n^ε / log n = **∞**

so n/log n is not O(n^(1−ε)) for any ε. It is smaller than the watershed by
a logarithmic factor only, never a polynomial one - the mirror image of the
§11 problem. Case 2 does not apply either, since n/log n = n·log^(−1) n
requires k = −1 and the extended case 2 requires k ≥ 0.

**The recurrence falls between cases 1 and 2, and the Master Theorem
returns no answer at all.**

It is still solvable, just not with this tool. By the recursion tree: there
are log₂ n levels, and level i does 2^i · (n/2^i)/log(n/2^i) =
n / (log n − i) work. Summing over i = 0 … log n − 1:

T(n) = n · Σ(i=0 to log n−1) 1/(log n − i) = n · (1 + 1/2 + … + 1/log n)
= n · H(log n)

where H is the harmonic series, H(m) = Θ(log m). Therefore:

> **T(n) = Θ(n log log n)**

A bound strictly between the Θ(n) of case 1 and the Θ(n log n) of case 2 -
which is exactly why no case could produce it. The Akra-Bazzi method
handles this class of recurrence in general.

---

## 14. Fibonacci by naive recursion - T(n) = T(n−1) + T(n−2) + Θ(1)

| | |
|---|---|
| a | 2 subproblems, of *different* sizes |
| b | - |

**The Master Theorem does not apply**, for two independent reasons: the
subproblems shrink by a constant amount rather than a constant factor (as
in §9), and they are not the same size as each other, which the theorem's
a·T(n/b) form cannot express.

The recurrence is its own solution. T(n) exceeds the Fibonacci numbers
themselves, which grow as φⁿ/√5 with φ = (1+√5)/2:

> **T(n) = Θ(φⁿ) ≈ Θ(1.618ⁿ)**

Included as the reminder that "divide and conquer" is not automatically
efficient. Splitting a problem in two only helps when the pieces are
*fractions* of the original; splitting into pieces of size n−1 and n−2
duplicates almost all the work, which is what memoisation eliminates to
give Θ(n).

---

## Summary

| # | Recurrence | a | b | f(n) | log_b a | Case | Why | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | 2T(n/2) + Θ(n) | 2 | 2 | Θ(n) | 1 | 2 | f = Θ(n^log_b a) exactly | Θ(n log n) |
| 2 | T(n/2) + Θ(1) | 1 | 2 | Θ(1) | 0 | 2 | f = Θ(n⁰) = watershed | Θ(log n) |
| 3 | 2T(n/2) + Θ(1) | 2 | 2 | Θ(1) | 1 | 1 | Θ(1) = O(n^(1−1)) | Θ(n) |
| 4 | 3T(n/2) + Θ(n) | 3 | 2 | Θ(n) | 1.585 | 1 | n = O(n^(1.585−0.585)) | Θ(n^1.585) |
| 5 | 4T(n/2) + Θ(n) | 4 | 2 | Θ(n) | 2 | 1 | n = O(n^(2−1)) | Θ(n²) |
| 6 | 7T(n/2) + Θ(n²) | 7 | 2 | Θ(n²) | 2.807 | 1 | n² = O(n^(2.807−0.807)) | Θ(n^2.807) |
| 7 | 8T(n/2) + Θ(n²) | 8 | 2 | Θ(n²) | 3 | 1 | n² = O(n^(3−1)) | Θ(n³) |
| 8 | 2T(n/2) + Θ(n) | 2 | 2 | Θ(n) | 1 | 2 | f = Θ(n) = watershed | Θ(n log n) |
| 9 | T(n−1) + Θ(n) | - | - | Θ(n) | - | **N/A** | shrinks by a constant, no b | Θ(n²) |
| 10 | 2T(n/2) + Θ(log n) | 2 | 2 | Θ(log n) | 1 | 1 | log n = O(n^0.5) | Θ(n) |
| 11 | 2T(n/2) + Θ(n log n) | 2 | 2 | Θ(n log n) | 1 | 2 ext. | f = Θ(n·log¹n), so k = 1 | Θ(n log² n) |
| 12 | 2T(n/2) + Θ(n²) | 2 | 2 | Θ(n²) | 1 | 3 | n² = Ω(n²); 2·f(n/2) = 0.5·f(n) | Θ(n²) |
| 13 | 2T(n/2) + n/log n | 2 | 2 | n/log n | 1 | **N/A** | gap is logarithmic, not polynomial | Θ(n log log n) |
| 14 | T(n−1) + T(n−2) + Θ(1) | - | - | Θ(1) | - | **N/A** | subproblems unequal, not a·T(n/b) | Θ(φⁿ) |

Three of the fourteen (§9, §13, §14) are outside the theorem's reach, for
three different reasons: subproblems that shrink by a constant amount, a
gap that is logarithmic rather than polynomial, and subproblems of unequal
size.

---

## References

- Cormen, Leiserson, Rivest and Stein, *Introduction to Algorithms*, 4th ed., Ch. 4 (the Master Theorem, its proof, and the Akra-Bazzi generalisation).
- Amakobe, *Advanced Computational Algorithms*, 2nd ed., Ch. 2.
