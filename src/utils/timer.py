"""Small-scale instrumentation for the Week 5 dynamic-programming study.

:mod:`src.utils.benchmark` already knows how to sweep an algorithm across
many input sizes and many data shapes. The dynamic-programming comparison
needs something smaller and sharper: time *one* call, count the recursive
calls *one* run makes, and measure the peak allocation of *one* execution.
This module is that companion. It does not re-implement
:class:`~src.utils.benchmark.AlgorithmBenchmark`; it borrows its
conventions so that a Week 5 number and a Week 1 number mean the same
thing:

* :func:`time.perf_counter` as the clock, the highest-resolution one
  Python exposes.
* Warm-up runs executed and discarded before anything is recorded.
* Mean, **sample** standard deviation, minimum and maximum reported over
  the measured runs, never a single stopwatch reading.
* Garbage collection paused inside each timed region and restored
  afterwards, as :mod:`timeit` does by default.

Why counting and timing are separate
------------------------------------
The plain implementations - ``fib_naive``, ``knapsack_tab``, ``lcs_memo``
and the rest - carry **no instrumentation at all**: no counter argument,
no bookkeeping in the hot loop. Those are the functions the benchmark
times, and an increment inside a recursion that runs 3.7 billion times
would not merely add noise, it would become the measurement. Counting
therefore happens in separate instrumented twins that thread a
:class:`CallCounter` through their own private helper, and those twins are
never the ones on the clock.

For the same reason a :class:`CallCounter` is an ordinary object rather
than a module-level global. Two benchmarks running in the same process
each hold their own counter, so neither can corrupt the other's totals,
and a counter can be inspected long after the run that filled it.

Timing and memory are also measured in separate runs.
:func:`peak_memory_kib` traces allocations with :mod:`tracemalloc`, which
costs roughly an order of magnitude in speed; a figure taken under
tracing is a memory measurement and nothing else.

Typical use::

    from src.utils.timer import CallCounter, Timer, peak_memory_kib, time_call

    stats = time_call(fib_tab, 30, repeat=5, warmup=2)
    print(stats["mean"], stats["std"])

    value, kib = peak_memory_kib(fib_tab, 30)

    counter = CallCounter()
    with counter.frame():
        ...
    print(counter.calls, counter.max_depth)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

__all__ = ["CallCounter", "Timer", "peak_memory_kib", "time_call"]

#: Bytes per kibibyte. Memory is reported in KiB rather than kB because
#: :mod:`tracemalloc` counts bytes and 1024 is the divisor that keeps the
#: reported figure a clean power of two.
_BYTES_PER_KIB = 1024.0


class CallCounter:
    """Counts calls and tracks maximum recursion depth for a single run.

    One counter belongs to one run of one instrumented function. It is a
    plain object, deliberately not a module-level global: two benchmarks
    in the same process hold two counters and cannot corrupt each other's
    totals, and a counter survives the run so its numbers can be read
    afterwards.

    The counted functions are deliberately separate from the timed ones.
    ``fib_naive`` makes 2*F(n+1) - 1 calls; an increment inside it would
    be executed as often as the recursion itself and would dominate the
    very runtime the benchmark is trying to measure. So the plain
    implementations stay bare, and their instrumented twins thread a
    counter through a private helper instead.

    The counter is its own context manager, which is why :meth:`frame`
    returns ``self``: entering a frame costs one attribute read and two
    integer updates, with no object allocated per call. Reentrancy is not
    a problem because the nesting is tracked by an integer rather than by
    the manager object.

    Attributes:
        calls: Total instrumented calls made since construction or the
            last :meth:`reset`.
        max_depth: Deepest nesting reached. Tabulation reaches 1 by
            construction, which is the point the Week 5 report makes.
        depth: Current nesting. Back to 0 once a balanced run finishes.

    Examples:
        >>> counter = CallCounter()
        >>> def countdown(n: int, c: CallCounter) -> int:
        ...     with c.frame():
        ...         return 0 if n == 0 else countdown(n - 1, c)
        >>> countdown(3, counter)
        0
        >>> counter.calls, counter.max_depth, counter.depth
        (4, 4, 0)
        >>> counter
        CallCounter(calls=4, max_depth=4, depth=0)

        A second run accumulates unless the counter is reset:

        >>> counter.reset()
        >>> counter.calls, counter.max_depth, counter.depth
        (0, 0, 0)

        Two counters are independent, which is what keeps concurrent
        measurements honest:

        >>> other = CallCounter()
        >>> _ = countdown(1, other)
        >>> other.calls, counter.calls
        (2, 0)
    """

    # __slots__ keeps a frame cheap: attribute access goes through a slot
    # descriptor rather than an instance dictionary, and no per-instance
    # dict is allocated at all.
    __slots__ = ("calls", "max_depth", "depth")

    def __init__(self) -> None:
        self.calls: int = 0
        self.max_depth: int = 0
        self.depth: int = 0

    def enter(self) -> None:
        """Record the start of one instrumented call.

        Increments the call total, descends one level, and raises the
        high-water mark if this is the deepest nesting seen so far.

        Returns:
            None. Called for its effect on the counter.

        Time Complexity:
            O(1) - three integer operations.

        Space Complexity:
            O(1).

        Examples:
            >>> counter = CallCounter()
            >>> counter.enter()
            >>> counter.calls, counter.depth, counter.max_depth
            (1, 1, 1)
        """
        self.calls += 1
        self.depth += 1
        if self.depth > self.max_depth:
            self.max_depth = self.depth

    def leave(self) -> None:
        """Record the end of one instrumented call.

        Ascends one level. ``max_depth`` is untouched, because it is a
        high-water mark rather than a current reading. Calls to
        :meth:`enter` and :meth:`leave` must balance; :meth:`frame`
        guarantees that even when the instrumented function raises.

        Returns:
            None. Called for its effect on the counter.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> counter = CallCounter()
            >>> counter.enter()
            >>> counter.leave()
            >>> counter.depth, counter.calls, counter.max_depth
            (0, 1, 1)
        """
        self.depth -= 1

    def reset(self) -> None:
        """Clear all three figures so the counter can be reused.

        Returns:
            None. Called for its effect on the counter.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> counter = CallCounter()
            >>> counter.enter()
            >>> counter.reset()
            >>> counter.calls, counter.max_depth, counter.depth
            (0, 0, 0)
        """
        self.calls = 0
        self.max_depth = 0
        self.depth = 0

    def frame(self) -> "CallCounter":
        """Return a context manager covering one instrumented call.

        The counter is its own context manager, so this returns ``self``
        rather than building a helper object: entering and leaving a frame
        allocate nothing, which matters when the block is entered millions
        of times inside a recursion. The ``with`` form is preferred over
        bare :meth:`enter` and :meth:`leave` because the depth still
        unwinds correctly when the instrumented function raises.

        Returns:
            This counter, ready to be used as a ``with`` target.

        Time Complexity:
            O(1) per frame.

        Space Complexity:
            O(1) - no object is created per frame.

        Examples:
            >>> counter = CallCounter()
            >>> with counter.frame():
            ...     with counter.frame():
            ...         pass
            >>> counter.calls, counter.max_depth, counter.depth
            (2, 2, 0)

            The depth unwinds even when the body raises:

            >>> try:
            ...     with counter.frame():
            ...         raise ValueError("boom")
            ... except ValueError:
            ...     pass
            >>> counter.depth
            0
        """
        return self

    def __enter__(self) -> "CallCounter":
        """Enter one frame. See :meth:`frame`."""
        self.enter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        """Leave one frame, whether or not the body raised.

        Returns:
            ``False`` always, so an exception raised inside the frame
            continues to propagate.
        """
        self.leave()
        return False

    def __repr__(self) -> str:
        """Return an unambiguous representation of the three figures.

        Examples:
            >>> CallCounter()
            CallCounter(calls=0, max_depth=0, depth=0)
        """
        return (
            f"CallCounter(calls={self.calls}, max_depth={self.max_depth}, "
            f"depth={self.depth})"
        )


class Timer:
    """Context manager timing one block with :func:`time.perf_counter`.

    The smallest useful instrument in the module: wrap a block, read
    :attr:`elapsed` afterwards. It measures exactly what is inside the
    ``with`` statement and does nothing else - in particular it does not
    pause garbage collection and does not repeat the block. For a figure
    worth putting in a report, use :func:`time_call`, which adds warm-ups,
    repeats and a standard deviation; use :class:`Timer` for a single
    wall-clock reading such as the total runtime of a benchmark script.

    :func:`time.perf_counter` is used rather than :func:`time.time`
    because it is monotonic and has the highest resolution the platform
    offers, and because it is the clock
    :class:`~src.utils.benchmark.AlgorithmBenchmark` uses. A single
    timing style across the project keeps Week 1 and Week 5 numbers
    comparable.

    A timer is reusable: entering it again starts a fresh measurement and
    discards the previous one.

    Args:
        label: Optional name for this measurement, carried only so that
            demo scripts can print a timer without tracking names
            separately.

    Attributes:
        label: The name given at construction.
        elapsed: Seconds measured. ``0.0`` before the block is entered,
            the running total while it is open, and the final figure
            after it closes.

    Examples:
        >>> with Timer() as clock:
        ...     total = sum(range(10_000))
        >>> clock.elapsed > 0
        True
        >>> total
        49995000

        The value is a duration in seconds, not a point in time:

        >>> clock.elapsed < 1.0
        True

        A fresh timer has not measured anything yet:

        >>> idle = Timer("naive fibonacci")
        >>> idle.elapsed
        0.0
        >>> idle.label
        'naive fibonacci'

        Re-entering starts a new measurement:

        >>> with idle:
        ...     _ = sum(range(1000))
        >>> idle.elapsed > 0
        True
    """

    __slots__ = ("label", "_start", "_end")

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    @property
    def elapsed(self) -> float:
        """Seconds measured by this timer.

        Returns:
            ``0.0`` before the first ``with`` block, the time since the
            block opened while it is still open, and the duration of the
            block once it has closed.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> Timer().elapsed
            0.0
        """
        if self._start is None:
            return 0.0
        if self._end is None:
            return time.perf_counter() - self._start
        return self._end - self._start

    def __enter__(self) -> "Timer":
        """Start the clock and return this timer."""
        self._end = None
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        """Stop the clock, then let any exception propagate.

        The clock is read first so that the cost of unwinding an
        exception is not charged to the block being measured.

        Returns:
            ``False`` always, so an exception raised inside the block is
            never swallowed by the timer.
        """
        self._end = time.perf_counter()
        return False

    def __repr__(self) -> str:
        """Return the label and the elapsed reading.

        Examples:
            >>> Timer("demo")
            Timer('demo', elapsed=0.000000s)
        """
        return f"Timer({self.label!r}, elapsed={self.elapsed:.6f}s)"


def time_call(
    func: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
    warmup: int = 2,
    **kwargs: Any,
) -> Dict[str, float]:
    """Time one call repeatedly and return its summary statistics.

    The protocol is the one
    :meth:`~src.utils.benchmark.AlgorithmBenchmark.time_algorithm` uses,
    applied to a single call rather than to a sweep:

    1. ``warmup`` executions run un-measured and are thrown away. They
       absorb first-call costs - import side effects, a cold instruction
       cache, CPU frequency ramp-up - that would otherwise land entirely
       on the first recorded run.
    2. ``repeat`` executions are timed individually with
       :func:`time.perf_counter`.
    3. Garbage collection is paused for the duration of each timed run,
       with a collection forced beforehand, and its previous state is
       restored in a ``finally`` block. A collection that happens to land
       inside one run and not another is noise, not signal; restoring the
       previous state matters because the caller may already have
       disabled collection for reasons of its own.
    4. Mean, sample standard deviation, minimum and maximum are computed
       over the ``repeat`` timings.

    The function's return value is discarded. This is a stopwatch, not a
    caller; check correctness outside the timed region.

    Note that ``repeat`` and ``warmup`` are keyword-only and are consumed
    here, so a function of its own with a parameter by either of those
    names must be wrapped - ``functools.partial`` or a ``lambda`` - before
    it can be timed.

    Args:
        func: The callable to time.
        *args: Positional arguments passed to ``func`` on every run.
        repeat: Number of measured runs. Must be at least 1; a standard
            deviation needs at least 2 to mean anything.
        warmup: Number of un-measured runs performed first. Must be
            non-negative. Lower it when a single run is expensive enough
            that discarding two is not worth paying for.
        **kwargs: Keyword arguments passed to ``func`` on every run.

    Returns:
        A dictionary with five keys: ``"mean"``, ``"std"``, ``"min"`` and
        ``"max"`` in seconds, and ``"runs"``, the integer number of
        measured runs. ``"std"`` is the sample standard deviation
        (:func:`statistics.stdev`) and is ``0.0`` when ``repeat`` is 1.
        Times are returned unrounded; rounding for display is the
        caller's decision.

    Raises:
        TypeError: If ``func`` is not callable, or ``repeat`` or
            ``warmup`` is not an integer.
        ValueError: If ``repeat`` is less than 1 or ``warmup`` is
            negative.

    Time Complexity:
        O((warmup + repeat) * C), where C is the cost of one call. The
        harness itself adds O(repeat) work to summarise the timings.

    Space Complexity:
        O(repeat) - one float per measured run is held to compute the
        statistics.

    Examples:
        >>> stats = time_call(sum, range(1000), repeat=3, warmup=1)
        >>> sorted(stats)
        ['max', 'mean', 'min', 'runs', 'std']
        >>> stats["runs"]
        3
        >>> stats["min"] <= stats["mean"] <= stats["max"]
        True
        >>> stats["std"] >= 0.0
        True

        A single run reports a standard deviation of zero rather than
        raising, because one measurement has no spread:

        >>> single = time_call(sum, range(100), repeat=1, warmup=0)
        >>> single["std"], single["runs"]
        (0.0, 1)
        >>> single["min"] == single["max"] == single["mean"]
        True

        Keyword arguments reach the callable:

        >>> stats = time_call(sorted, [3, 1, 2], reverse=True, repeat=2, warmup=0)
        >>> stats["runs"]
        2

        The contract is checked before anything is timed:

        >>> time_call(sum, [1], repeat=0)
        Traceback (most recent call last):
            ...
        ValueError: repeat must be >= 1, got 0
        >>> time_call(42)
        Traceback (most recent call last):
            ...
        TypeError: func must be callable, got int
    """
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func).__name__}")
    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise TypeError(f"repeat must be an int, got {type(repeat).__name__}")
    if isinstance(warmup, bool) or not isinstance(warmup, int):
        raise TypeError(f"warmup must be an int, got {type(warmup).__name__}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    def one_run() -> float:
        """Execute ``func`` once under a paused collector and return seconds."""
        was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            func(*args, **kwargs)
            return time.perf_counter() - start
        finally:
            # Restore rather than simply enable: the caller may have had
            # collection disabled before it ever reached this function.
            if was_enabled:
                gc.enable()

    for _ in range(warmup):
        one_run()

    timings: List[float] = [one_run() for _ in range(repeat)]

    return {
        "mean": statistics.fmean(timings),
        "std": statistics.stdev(timings) if repeat > 1 else 0.0,
        "min": min(timings),
        "max": max(timings),
        "runs": repeat,
    }


def peak_memory_kib(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, float]:
    """Run a callable once under :mod:`tracemalloc` and report its peak.

    Exactly one call is made, wrapped in
    :func:`tracemalloc.start` / :func:`tracemalloc.get_traced_memory` /
    :func:`tracemalloc.stop`. The peak is the high-water mark of memory
    allocated by Python during that call, which is the figure the Week 5
    comparison needs: an O(n*W) knapsack table and an O(W) rolling row
    differ in peak allocation, not in the memory they happen to be
    holding when they return.

    **Measure memory and time in separate runs.** Tracing every
    allocation slows execution by roughly an order of magnitude, so a
    timing taken while :mod:`tracemalloc` is active is a measurement of
    the tracer. Call :func:`time_call` for the clock and this function for
    the memory, and never nest one inside the other.

    Tracing is stopped in a ``finally`` block, so a callable that raises
    does not leave the tracer running and slow down everything measured
    afterwards. If tracing was already active when this function was
    called, the peak is reset instead of the tracer being restarted and
    the tracer is left running, so an enclosing profile is not disturbed.

    Args:
        func: The callable to measure.
        *args: Positional arguments passed to ``func``.
        **kwargs: Keyword arguments passed to ``func``.

    Returns:
        A ``(result, peak_kib)`` pair: whatever ``func`` returned, and the
        peak traced allocation during the call in kibibytes as a float.
        The result is returned as well as the figure so that a caller can
        check the answer without paying for a second run.

    Raises:
        TypeError: If ``func`` is not callable.
        Exception: Anything ``func`` raises propagates unchanged, after
            tracing has been stopped.

    Time Complexity:
        O(C), where C is the cost of one call, multiplied by the
        tracemalloc overhead - roughly an order of magnitude.

    Space Complexity:
        O(1) beyond whatever ``func`` allocates, plus the tracer's own
        bookkeeping.

    Examples:
        >>> value, kib = peak_memory_kib(lambda n: [0] * n, 50_000)
        >>> len(value)
        50000
        >>> kib > 0
        True

        The return value is passed through untouched:

        >>> peak_memory_kib(sorted, [3, 1, 2])[0]
        [1, 2, 3]

        A failing call still leaves the tracer as it found it:

        >>> import tracemalloc
        >>> before = tracemalloc.is_tracing()
        >>> def explode() -> None:
        ...     raise ValueError("no result")
        >>> try:
        ...     peak_memory_kib(explode)
        ... except ValueError as exc:
        ...     print(exc)
        no result
        >>> tracemalloc.is_tracing() == before
        True
    """
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func).__name__}")

    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        # Someone else owns the tracer. Zero the high-water mark so this
        # call's peak is not inherited from theirs, and leave their
        # tracing running when we are done.
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()

    try:
        result = func(*args, **kwargs)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    return result, peak / _BYTES_PER_KIB
