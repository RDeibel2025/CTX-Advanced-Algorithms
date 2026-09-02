# Week 2 Technical Report - Divide and Conquer

**Robert Deibel** · CSC 5300 Advanced Algorithms · Concordia University Texas · Fall 2026

## 📎 Complete project repository

### **<https://github.com/RDeibel2025/CTX-Advanced-Algorithms>**

[Fourteen Master Theorem solutions](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/analysis/week2_recurrences.md).

---

## 1. Executive Summary

Merge sort and quicksort beat the Week 1 quadratics by a compounding factor -
parity at n = 100, 154× at n = 10,000 - and both matched O(n log n), with doubling
ratios of 2.01-2.25 and 2.10-2.40 against a predicted ~2.1. Quicksort's O(n²)
worst case never appeared. The most interesting result was negative:
three-way partitioning rescues duplicate-heavy input only from a weakness Hoare's
scheme lacks.

## 2. Methodology

Apple M2 Max, 12 cores, 64 GB RAM; macOS 14.5 (arm64); CPython 3.12.4; timing by
`time.perf_counter()`. Six algorithms (three Week 1 quadratics, merge, quicksort,
quicksort three-way) over sizes 100 … 50,000 and six shapes: random, sorted, reverse, nearly sorted (5% transposed),
many duplicates (10 distinct values), few unique (3). Two warm-up runs are
discarded per cell, five measured below n = 10,000 and three above.
Generation is seeded; every algorithm gets the identical list per cell, and
every run is verified a sorted permutation. Median relative SD: 0.7%.

**The O(n²) size cap.** The quadratics were measured only to n = 10,000; the
eighteen cells at n = 50,000 appear in
[`comparison_table.csv`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/benchmarks/results/comparison_table.csv) as `omitted`
with a reason. Fitting t = c·n² projects 63 s, 29 s and 27 s
per run on random data - about an hour. I measured them anyway and checked the
projection (§4).

## 3. Results

![random](https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/benchmarks/results/random_data.png)
![sorted](https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/benchmarks/results/sorted_data.png)
![reverse](https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/benchmarks/results/reverse_data.png)
![nearly](https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/benchmarks/results/nearly_sorted.png)
![duplicates](https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/benchmarks/results/many_duplicates.png)
![few unique](https://raw.githubusercontent.com/RDeibel2025/CTX-Advanced-Algorithms/main/benchmarks/results/few_unique.png)

Mean seconds at n = 10,000:

| Algorithm | random | sorted | reverse | nearly | many dup | few uniq |
|---|---|---|---|---|---|---|
| Bubble | 2.5498 | 0.0003 | 3.2360 | 1.5665 | 2.3059 | 1.9940 |
| Selection | 1.1182 | 1.1031 | 1.1556 | 1.0929 | 1.0782 | 1.1047 |
| Insertion | 1.1167 | 0.0005 | 2.0667 | 0.1313 | 0.9004 | 0.6855 |
| Merge | 0.0137 | 0.0091 | 0.0091 | 0.0116 | 0.0121 | 0.0116 |
| Quick | 0.0073 | 0.0049 | 0.0052 | 0.0052 | 0.0047 | 0.0047 |
| Quick 3-way | 0.0108 | 0.0099 | 0.0099 | 0.0097 | 0.0016 | 0.0008 |

Speedup over the best quadratic:

| n | 100 | 500 | 1,000 | 5,000 | 10,000 |
|---|---|---|---|---|---|
| Merge | 1.0× | 4.3× | 9.9× | 46.7× | 81.5× |
| Quick | 1.9× | 8.7× | 18.2× | 93.2× | 153.6× |

The gap compounds; at n = 100 insertion and merge sort are indistinguishable
(0.000080 s vs 0.000079 s, inside the run-to-run variation). Asymptotic
superiority is a claim about large n.

**Merge sort** was among the most consistent - 1.5× variation across the six
shapes at n = 10,000, against 10,823× for bubble and 3,856× for insertion. Its
positional split gives the same recursion tree whatever the input order: no best
or worst case to find. The price is O(n) auxiliary space.

**QuickSort** was fastest on five of six shapes, spread 1.5×: best on few-unique
(0.0047 s), worst on random (0.0073 s) - nothing like the O(n log n)-to-O(n²)
range theory permits.

**Surprising finding.** Three-way partitioning is *slower* than two-way wherever
values are mostly distinct - roughly half speed - paying off only on
duplicate-heavy input: 2.9× at ten distinct values, 5.6× at three.

## 4. Complexity Validation

Observed t(2n)/t(n), 500→1,000 and 5,000→10,000:

| Algorithm | random | sorted | reverse | few uniq | Predicted |
|---|---|---|---|---|---|
| Bubble | 4.40 | 1.90 | 4.32 | 4.23 | 4 / 2 adaptive |
| Selection | 4.11 | 4.11 | 4.13 | 4.19 | 4 |
| Insertion | 4.38 | 2.01 | 4.22 | 4.14 | 4 / 2 adaptive |
| Merge | 2.20 | 2.07 | 2.03 | 2.08 | ~2.1 |
| Quick | 2.39 | 2.20 | 2.18 | 2.15 | ~2.1 |

Merge sort stays within 0.15 of prediction everywhere. Bubble and insertion drop
to ~2.1 on sorted input, their early exit appearing as a change of complexity
class.

**Did I observe quicksort's O(n²) worst case? No** - every shape gave a ratio near
2.2, including the three meant to be adversarial. Three mechanisms prevent it:
randomized pivots leave no fixed input adversarial, so the worst case needs an
unlucky *sequence of draws*; Hoare's partition splits a run of equal elements near
its middle rather than onto one side; and recursing into the smaller partition
caps stack depth at O(log n) - 10-14 frames at n = 100,000, against Python's
1,000-frame limit.

Unreachable *by this implementation* is not unreachable. Lomuto's partition, the
scheme most often taught, reaches it at once:

| Scheme (3 values) | growth/doubling | t(16,000) |
|---|---|---|
| Lomuto (2-way) | **3.98** | 1.9246 s |
| Hoare (2-way) | 2.15 | 0.0079 s |
| Dutch flag (3-way) | 2.02 | 0.0016 s |

3.99 against a predicted 4 is the worst case, measured - 244× slower than Hoare
on identical input.

**The capped cells.** Measuring all eighteen put the fitted t = c·n² projection
within **2.7%** on the sixteen genuinely quadratic cells (mean 1.4%), and off by
**+436%** on the two sorted-input cells, where these algorithms are linear and n²
is the wrong model. The extrapolation holds where its assumption does.

## 5. Optimization Impact

**Insertion-sort cutoff**, measured by varying the threshold. On random input at
n = 50,000, disabling it costs 0.0563 s; a threshold of 10 gives 0.0389 s
(**1.44×**) and the optimum of 20 gives 0.0353 s (1.59×). Past 20 the curve turns
back up as insertion sort's O(k²) reasserts itself; on nearly-sorted and
few-unique input it was still improving at 80, so one constant is a compromise.

**Three-way partitioning:** 5.6× at three distinct values, 2.9× at ten, ~2× slower
otherwise. **Randomized pivot** cannot be isolated by timing; its evidence is
§4's negative result.

## 6. Practical Recommendations

- **n below ~100:** insertion sort - it matches merge sort there.
- **General purpose:** quicksort - randomized pivot, Hoare partition, insertion cutoff.
- **Stability or a guarantee:** merge sort, 1.9× slower on random.
- **Duplicate-heavy data:** three-way partitioning.
- **Never use Lomuto without a duplicate guard.**

## 7. Conclusion

Measurement matched theory closely enough that the interesting results were where
it did not: asymptotic advantage is real but only past a measurable crossover,
quicksort's worst case belongs to the partition scheme rather than to quicksort,
and three-way partitioning is a trade, not a free win.

## 8. References

- Cormen, Leiserson, Rivest, Stein. *Introduction to Algorithms*, 4th ed., Ch. 2.3-2.4, 4, 7
- Amakobe. *Advanced Computational Algorithms*, 2nd ed., Ch. 2
- Sedgewick, Wayne. *Algorithms*, 4th ed., §2.3

## AI use acknowledgement

I used Anthropic's Claude (Claude Code) to draft the code, tests and first draft of
this report, from a specification I wrote off the instructions and assigned
reading. The measurements are my own, every figure above
computed from the result CSVs rather than transcribed. The design decisions
were mine - Hoare's partition over Lomuto's, recursing into the smaller partition,
and measuring the capped cells rather than projecting them. Full disclosure:
[`docs/AI_USE.md`](https://github.com/RDeibel2025/CTX-Advanced-Algorithms/blob/main/docs/AI_USE.md).
