"""Shared utilities: benchmarking, visualization and testing helpers.

    >>> from src.utils import AlgorithmBenchmark
    >>> bench = AlgorithmBenchmark(warmup_runs=0)
    >>> bench.generate_test_data(5, "sorted", seed=42)
    [0, 1, 2, 3, 4]
"""

from src.utils.benchmark import AlgorithmBenchmark, BenchmarkResult

__all__ = ["AlgorithmBenchmark", "BenchmarkResult"]
