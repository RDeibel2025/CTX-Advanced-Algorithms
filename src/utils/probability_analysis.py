"""Measuring randomized algorithms: summaries, confidence bounds and the theory.

A deterministic algorithm run twice on the same input does the same work
twice, so one measurement describes it. A randomized algorithm does not: its
comparison count, its running time and, for a Monte Carlo algorithm, its
answer are random variables. The Week 7 experiments therefore report
*distributions*, and every number they print comes from one of the small,
deterministic functions in this module. None of them draws a random number;
they only describe samples that the seeded algorithms produced.

The module is shaped around the two kinds of randomized algorithm:

* **Las Vegas** algorithms (randomized quicksort, randomized selection) are
  always correct, so the question is how their *cost* is spread.
  :func:`summarize` gives mean, sample variance, spread and the coefficient
  of variation ``cv = std / mean``, which is what makes runs of different
  sizes comparable: quicksort's comparison count has a standard deviation
  that grows like ``0.6483 n`` while its mean grows like ``2 n ln n``, so
  its ``cv`` shrinks as ``n`` grows. :func:`tail_fraction` answers the
  question the mean hides: how often a run is expensive, for example how
  often quickselect needs more than ``4n`` comparisons.
* **Monte Carlo** algorithms (Fermat, Miller-Rabin, the Bloom filter,
  MinHash) bound their work and may be wrong, so the question is how often.
  :func:`empirical_rate` is the observed error rate and
  :func:`wilson_interval` puts a 95% confidence interval around it. The
  Wilson score interval is used instead of the textbook ``p +- z sqrt(p (1 -
  p) / n)`` (Wald) interval because the Wald interval collapses to the
  single point ``[0, 0]`` when no errors are observed, which is exactly what
  happens when Miller-Rabin is run on thousands of composites. Wilson stays
  honest there: zero errors in 100 trials still allows a true rate of about
  3.7%. :func:`rule_of_three_upper` is the quick version of the same idea,
  ``3 / n`` as a 95% upper bound after ``n`` clean trials.

The last three functions are the theory the measurements are checked
against, kept here so the data structures and the experiments use one copy
of each formula:

====================== =====================================================
Function               Formula
====================== =====================================================
bloom_fp_theoretical   ``(1 - e^(-k n / m))^k``, the false positive rate of a
                       Bloom filter with m bits, k hashes and n items
bloom_optimal_param... ``m = ceil(-n ln p / (ln 2)^2)``,
                       ``k = max(1, round(m / n * ln 2))``
minhash_standard_error ``sqrt(J (1 - J) / k)``, the standard error of a
                       k-permutation MinHash estimate of Jaccard J
====================== =====================================================

Everything is pure standard library. Counts must be integers (``bool`` is
rejected, since ``True`` successes is an accident rather than a count) and
samples must be real numbers.

Examples:
    A Monte Carlo test that made no mistakes in 100 trials has an observed
    error rate of zero, but the data cannot rule out a rate of a few
    percent:

    >>> empirical_rate(0, 100)
    0.0
    >>> low, high = wilson_interval(0, 100)
    >>> low, round(high, 4)
    (0.0, 0.037)
    >>> rule_of_three_upper(100)
    0.03

    The Bloom filter sizing used by ``BloomFilter.from_capacity``:

    >>> bloom_optimal_parameters(1000, 0.01)
    (9586, 7)
    >>> round(bloom_fp_theoretical(9586, 1000, 7), 5)
    0.01003

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import math
import numbers
import statistics
from typing import Any, Dict, Iterable, List, Tuple

__all__ = [
    "bloom_fp_theoretical",
    "bloom_optimal_parameters",
    "empirical_rate",
    "minhash_standard_error",
    "rule_of_three_upper",
    "summarize",
    "tail_fraction",
    "wilson_interval",
]


def _check_int(value: Any, name: str) -> int:
    """Validate that ``value`` is an integer count and return it as an ``int``.

    Any :class:`numbers.Integral` is accepted, so a numpy integer from an
    experiment's array works as well as a plain ``int``. ``bool`` is
    rejected even though it is an ``int`` subclass.

    Args:
        value: The value to check.
        name: The parameter name, for the error message.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an integer, or is a ``bool``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_int(7, "trials")
        7
        >>> _check_int(True, "trials")
        Traceback (most recent call last):
            ...
        TypeError: trials must be an int, got bool
        >>> _check_int(2.0, "k")
        Traceback (most recent call last):
            ...
        TypeError: k must be an int, got float
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _check_real(value: Any, name: str) -> float:
    """Validate that ``value`` is a real number and return it as a ``float``.

    Args:
        value: The value to check.
        name: The parameter name, for the error message.

    Returns:
        ``float(value)``.

    Raises:
        TypeError: If ``value`` is not a real number, or is a ``bool``.
        ValueError: If ``value`` is NaN, which would silently poison every
            sum and comparison it took part in.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_real(3, "z")
        3.0
        >>> _check_real("3", "z")
        Traceback (most recent call last):
            ...
        TypeError: z must be a real number, got str
        >>> _check_real(float("nan"), "z")
        Traceback (most recent call last):
            ...
        ValueError: z must not be NaN
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must not be NaN")
    return result


def _as_floats(samples: Iterable[float]) -> List[float]:
    """Validate a sample and return it as a new non-empty list of floats.

    Args:
        samples: Any iterable of real numbers. Not modified; a generator is
            consumed.

    Returns:
        A new list holding ``float(x)`` for every ``x`` in ``samples``.

    Raises:
        TypeError: If ``samples`` is not iterable or holds a non-real value.
        ValueError: If ``samples`` is empty or holds NaN.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the copy.

    Examples:
        >>> _as_floats((1, 2.5))
        [1.0, 2.5]
        >>> _as_floats([])
        Traceback (most recent call last):
            ...
        ValueError: samples must not be empty
        >>> _as_floats(5)
        Traceback (most recent call last):
            ...
        TypeError: samples must be an iterable of numbers, got int
    """
    try:
        items = list(samples)
    except TypeError:
        raise TypeError(
            "samples must be an iterable of numbers, "
            f"got {type(samples).__name__}"
        ) from None
    if not items:
        raise ValueError("samples must not be empty")
    return [_check_real(item, f"samples[{index}]") for index, item in enumerate(items)]


def _check_counts(successes: Any, trials: Any) -> Tuple[int, int]:
    """Validate a ``(successes, trials)`` pair of counts.

    Args:
        successes: The number of trials with the outcome of interest.
        trials: The number of trials run.

    Returns:
        ``(successes, trials)`` as plain ints.

    Raises:
        TypeError: If either is not an int (``bool`` rejected).
        ValueError: Unless ``trials >= 1`` and ``0 <= successes <= trials``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _check_counts(3, 10)
        (3, 10)
        >>> _check_counts(0, 0)
        Traceback (most recent call last):
            ...
        ValueError: trials must be at least 1, got 0
        >>> _check_counts(11, 10)
        Traceback (most recent call last):
            ...
        ValueError: successes must be in [0, trials] = [0, 10], got 11
    """
    successes = _check_int(successes, "successes")
    trials = _check_int(trials, "trials")
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(
            f"successes must be in [0, trials] = [0, {trials}], got {successes}"
        )
    return successes, trials


def summarize(samples: Iterable[float]) -> Dict[str, float]:
    """Describe a sample of measurements: centre, spread and extremes.

    The one-call description of a Las Vegas algorithm's cost over many
    seeded runs. The variance is the *sample* variance, dividing by
    ``n - 1``, because the runs are a sample from the algorithm's cost
    distribution rather than the whole of it. ``cv = std / mean`` is the
    coefficient of variation, the spread measured in units of the mean,
    which is what lets a run on ``n = 100`` be compared with one on
    ``n = 100,000``.

    Mean and variance are computed in two passes with :func:`math.fsum`, so
    they are accurate and stay O(n) even for long samples. A constant sample
    is detected first and reported with variance exactly 0.0, so a
    deterministic algorithm's identical counts do not show a spurious
    rounding-error spread.

    Args:
        samples: A non-empty iterable of real numbers. Not modified.

    Returns:
        A new dict with keys, in this order: ``mean``, ``variance`` (sample,
        ``n - 1``; 0.0 when ``n < 2``), ``std``, ``min``, ``max``,
        ``median``, ``cv`` (``std / mean``; 0.0 when the mean is 0, and
        negative when the mean is) and ``n``. Every value is a float except
        ``n``, which is an int.

    Raises:
        TypeError: If ``samples`` is not iterable or holds a non-real value
            (``bool`` rejected).
        ValueError: If ``samples`` is empty or holds NaN.

    Time Complexity:
        O(n log n) - the median sorts a copy; everything else is O(n).

    Space Complexity:
        O(n) - the float copy and the sorted copy for the median.

    Examples:
        >>> stats = summarize([1, 2, 3, 4, 5])
        >>> for key, value in stats.items():
        ...     print(key, round(value, 4))
        mean 3.0
        variance 2.5
        std 1.5811
        min 1.0
        max 5.0
        median 3.0
        cv 0.527
        n 5

        One sample has no spread, and a zero mean has no meaningful ``cv``:

        >>> summarize([7])["variance"], summarize([7])["std"]
        (0.0, 0.0)
        >>> summarize([-1, 1])["cv"]
        0.0

        A constant sample has variance exactly zero, not rounding noise:

        >>> summarize([0.1] * 3)["variance"]
        0.0

        An even-length sample's median is the mean of the middle two:

        >>> summarize([4, 1, 3, 2])["median"]
        2.5
        >>> summarize([])
        Traceback (most recent call last):
            ...
        ValueError: samples must not be empty
    """
    values = _as_floats(samples)
    count = len(values)
    lowest = min(values)
    highest = max(values)

    if lowest == highest:
        mean = lowest
        variance = 0.0
    else:
        mean = math.fsum(values) / count
        variance = math.fsum((x - mean) ** 2 for x in values) / (count - 1)

    std = math.sqrt(variance)
    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "min": lowest,
        "max": highest,
        "median": float(statistics.median(values)),
        "cv": std / mean if mean != 0 else 0.0,
        "n": count,
    }


def empirical_rate(successes: int, trials: int) -> float:
    """Return the observed proportion ``successes / trials``.

    The point estimate of a probability, such as a Monte Carlo test's error
    rate or a Bloom filter's false positive rate. On its own it says nothing
    about how far the true rate may be from it; pair it with
    :func:`wilson_interval`.

    Args:
        successes: How many trials had the outcome being counted.
        trials: How many trials were run, at least 1.

    Returns:
        A float in ``[0.0, 1.0]``.

    Raises:
        TypeError: If either count is not an int (``bool`` rejected).
        ValueError: Unless ``trials >= 1`` and ``0 <= successes <= trials``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> empirical_rate(3, 12)
        0.25
        >>> empirical_rate(0, 100), empirical_rate(100, 100)
        (0.0, 1.0)
        >>> empirical_rate(1, 0)
        Traceback (most recent call last):
            ...
        ValueError: trials must be at least 1, got 0
    """
    successes, trials = _check_counts(successes, trials)
    return successes / trials


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Return the Wilson score confidence interval for a binomial proportion.

    With ``p = successes / trials``, ``n = trials`` and ``z`` the normal
    quantile (1.96 for 95%)::

        centre = (p + z^2 / (2n)) / (1 + z^2 / n)
        half   = z / (1 + z^2 / n) * sqrt(p (1 - p) / n + z^2 / (4 n^2))
        (low, high) = (centre - half, centre + half)

    Unlike the Wald interval ``p +- z sqrt(p (1 - p) / n)``, it does not
    collapse to a point at ``p = 0`` or ``p = 1``, which are exactly the
    observations a good Monte Carlo algorithm produces. At 0 successes the
    formula's lower end is exactly 0 and at ``trials`` successes its upper
    end is exactly 1; both are set exactly rather than left to floating
    point, and both ends are clamped to ``[0, 1]``.

    Args:
        successes: How many trials had the outcome being counted.
        trials: How many trials were run, at least 1.
        z: The standard normal quantile for the confidence level: 1.96 for
            95% (the default), 2.576 for 99%. Must be positive.

    Returns:
        A ``(low, high)`` pair of floats with
        ``0.0 <= low <= p <= high <= 1.0``.

    Raises:
        TypeError: If a count is not an int (``bool`` rejected) or ``z`` is
            not a real number.
        ValueError: Unless ``trials >= 1``, ``0 <= successes <= trials`` and
            ``z`` is positive and finite.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        Zero errors in 100 trials: the lower end is exactly 0, but the upper
        end is not, so the data still allow a rate of about 3.7%:

        >>> low, high = wilson_interval(0, 100)
        >>> low
        0.0
        >>> round(high, 6)
        0.036995
        >>> high > 0
        True

        Half of 100 trials:

        >>> low, high = wilson_interval(50, 100)
        >>> round(low, 4), round(high, 4)
        (0.4038, 0.5962)

        Every trial a success is the mirror image of none:

        >>> low, high = wilson_interval(100, 100)
        >>> round(low, 6), high
        (0.963005, 1.0)

        A higher confidence level widens the interval:

        >>> wilson_interval(50, 100, z=2.576)[1] > wilson_interval(50, 100)[1]
        True
        >>> wilson_interval(5, 100, z=0)
        Traceback (most recent call last):
            ...
        ValueError: z must be positive and finite, got 0.0
    """
    successes, trials = _check_counts(successes, trials)
    z = _check_real(z, "z")
    if not 0.0 < z < math.inf:
        raise ValueError(f"z must be positive and finite, got {z!r}")

    p = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (p + z_squared / (2 * trials)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p * (1.0 - p) / trials + z_squared / (4 * trials * trials)
    )

    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == trials else min(1.0, centre + half_width)
    return low, high


def rule_of_three_upper(trials: int) -> float:
    """Return ``3 / trials``, the rule-of-three 95% upper bound after no events.

    If an event of probability ``q`` never happens in ``n`` independent
    trials, then ``(1 - q)^n >= 0.05`` at 95% confidence, and since
    ``(1 - q)^n <= e^(-q n)`` this gives ``q <= -ln(0.05) / n``, which is
    about ``2.996 / n``. So "zero false 'prime' answers in 1,000 runs" means
    the true rate is below 0.3% with 95% confidence, not that it is zero.

    The bound is only informative once ``trials`` is large; for
    ``trials < 3`` it exceeds 1 and is returned as computed rather than
    clamped, so callers always get exactly ``3 / trials``.

    Args:
        trials: How many trials were run with no event, at least 1.

    Returns:
        ``3 / trials`` as a float.

    Raises:
        TypeError: If ``trials`` is not an int (``bool`` rejected).
        ValueError: If ``trials < 1``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> rule_of_three_upper(1000) == 0.003
        True
        >>> rule_of_three_upper(30)
        0.1

        It agrees closely with the Wilson upper end at zero successes:

        >>> round(wilson_interval(0, 1000)[1], 4)
        0.0038
        >>> rule_of_three_upper(0)
        Traceback (most recent call last):
            ...
        ValueError: trials must be at least 1, got 0
    """
    trials = _check_int(trials, "trials")
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    return 3 / trials


def bloom_fp_theoretical(m: int, n: int, k: int) -> float:
    """Return the standard false positive estimate ``(1 - e^(-k n / m))^k``.

    After ``n`` insertions with ``k`` hashes each into ``m`` bits, a given
    bit is still 0 with probability ``(1 - 1/m)^(k n)``, which is about
    ``e^(-k n / m)``. A query for an absent item is a false positive when
    all ``k`` of its bits are 1, so the rate is about
    ``(1 - e^(-k n / m))^k``. It treats the ``k`` probes as independent,
    which is the usual textbook approximation and is very accurate for the
    sizes used here. ``1 - e^(-x)`` is computed as ``-expm1(-x)`` so it
    stays accurate when ``k n / m`` is tiny.

    Args:
        m: The number of bits, at least 1.
        n: The number of items inserted, at least 0.
        k: The number of hash functions, at least 1.

    Returns:
        A float in ``[0.0, 1.0]``; 0.0 when ``n`` is 0.

    Raises:
        TypeError: If any argument is not an int (``bool`` rejected).
        ValueError: Unless ``m >= 1``, ``n >= 0`` and ``k >= 1``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> round(bloom_fp_theoretical(9586, 1000, 7), 5)
        0.01003
        >>> bloom_fp_theoretical(9586, 0, 7)
        0.0

        Overfilling the same filter raises the rate sharply:

        >>> round(bloom_fp_theoretical(9586, 2000, 7), 3)
        0.157
        >>> bloom_fp_theoretical(0, 10, 3)
        Traceback (most recent call last):
            ...
        ValueError: m must be at least 1, got 0
    """
    m = _check_int(m, "m")
    n = _check_int(n, "n")
    k = _check_int(k, "k")
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m}")
    if n < 0:
        raise ValueError(f"n must be at least 0, got {n}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if n == 0:
        # -expm1(-0.0) is -0.0; an empty filter has no false positives.
        return 0.0
    return (-math.expm1(-k * n / m)) ** k


def bloom_optimal_parameters(n: int, p: float) -> Tuple[int, int]:
    """Return the bit count ``m`` and hash count ``k`` for ``n`` items at rate ``p``.

    For fixed ``m`` and ``n``, the rate ``(1 - e^(-k n / m))^k`` is
    smallest at ``k = (m / n) ln 2``, where half the bits end up set and the
    rate is ``2^(-k)``. Solving that for ``m`` at a target rate ``p`` gives
    the two formulas used here::

        m = ceil(-n ln p / (ln 2)^2)
        k = max(1, round(m / n * ln 2))

    ``m`` is rounded up so the filter is never smaller than the target
    needs, and ``k`` is rounded to the nearest integer (at least 1) because
    a fraction of a hash function is not available. This is what
    ``BloomFilter.from_capacity`` calls.

    Args:
        n: The number of items the filter is expected to hold, at least 1.
        p: The target false positive rate, strictly between 0 and 1.

    Returns:
        An ``(m, k)`` pair of positive ints.

    Raises:
        TypeError: If ``n`` is not an int (``bool`` rejected) or ``p`` is
            not a real number.
        ValueError: Unless ``n >= 1`` and ``0 < p < 1``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> bloom_optimal_parameters(1000, 0.01)
        (9586, 7)

        About 9.6 bits per item buys a 1% rate, and every further factor of
        ten costs about 4.8 bits more per item:

        >>> bloom_optimal_parameters(1000, 0.001)
        (14378, 10)
        >>> m, k = bloom_optimal_parameters(1000, 0.01)
        >>> bloom_fp_theoretical(m, 1000, k) < 0.0101
        True

        A very loose target still gets one hash:

        >>> bloom_optimal_parameters(10, 0.9)[1]
        1
        >>> bloom_optimal_parameters(1000, 1.0)
        Traceback (most recent call last):
            ...
        ValueError: p must be strictly between 0 and 1, got 1.0
    """
    n = _check_int(n, "n")
    p = _check_real(p, "p")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be strictly between 0 and 1, got {p!r}")

    ln2 = math.log(2)
    m = math.ceil(-n * math.log(p) / (ln2 * ln2))
    k = max(1, round(m / n * ln2))
    return m, k


def minhash_standard_error(j: float, k: int) -> float:
    """Return ``sqrt(J (1 - J) / k)``, the standard error of a MinHash estimate.

    Each of the ``k`` signature positions agrees with probability exactly
    ``J``, the true Jaccard similarity, so the estimate (the fraction of
    agreeing positions) is a binomial proportion with this standard error.
    Its consequence is the cost of accuracy: halving the error takes four
    times as many permutations. The error is largest at ``J = 0.5`` and
    zero at ``J = 0`` and ``J = 1``.

    Args:
        j: The true Jaccard similarity, in ``[0, 1]``.
        k: The number of hash functions (permutations), at least 1.

    Returns:
        A non-negative float.

    Raises:
        TypeError: If ``j`` is not a real number or ``k`` is not an int
            (``bool`` rejected for both).
        ValueError: Unless ``0 <= j <= 1`` and ``k >= 1``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> round(minhash_standard_error(1 / 3, 512), 4)
        0.0208

        Quadrupling ``k`` halves the error:

        >>> math.isclose(minhash_standard_error(0.5, 400) * 2,
        ...              minhash_standard_error(0.5, 100))
        True
        >>> minhash_standard_error(1.0, 128)
        0.0
        >>> minhash_standard_error(1.5, 128)
        Traceback (most recent call last):
            ...
        ValueError: j must be in [0, 1], got 1.5
    """
    j = _check_real(j, "j")
    k = _check_int(k, "k")
    if not 0.0 <= j <= 1.0:
        raise ValueError(f"j must be in [0, 1], got {j!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return math.sqrt(j * (1.0 - j) / k)


def tail_fraction(samples: Iterable[float], threshold: float) -> float:
    """Return the fraction of ``samples`` strictly greater than ``threshold``.

    The empirical tail probability ``P(X > threshold)``. For a Las Vegas
    algorithm, whose answer is always right, this is the informative
    probability: for example the fraction of quickselect runs on ``n``
    items that needed more than ``4n`` comparisons. The comparison is
    strict, so a sample equal to the threshold is not in the tail.

    Args:
        samples: A non-empty iterable of real numbers. Not modified.
        threshold: The cut-off, a real number.

    Returns:
        A float in ``[0.0, 1.0]``.

    Raises:
        TypeError: If ``samples`` is not iterable, holds a non-real value,
            or ``threshold`` is not a real number (``bool`` rejected).
        ValueError: If ``samples`` is empty or anything is NaN.

    Time Complexity:
        O(n).

    Space Complexity:
        O(n) - the validated float copy.

    Examples:
        >>> tail_fraction([1, 2, 3, 4, 5], 3)
        0.4
        >>> tail_fraction([1, 2, 3], 3)
        0.0
        >>> tail_fraction([10, 20], -math.inf)
        1.0
        >>> tail_fraction([], 3)
        Traceback (most recent call last):
            ...
        ValueError: samples must not be empty
    """
    values = _as_floats(samples)
    cutoff = _check_real(threshold, "threshold")
    above = sum(1 for value in values if value > cutoff)
    return above / len(values)
