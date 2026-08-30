"""Packaging metadata for the CSC 5300 algorithm laboratory.

Installing the project in editable mode makes ``src`` importable from
anywhere, which is what lets ``pytest`` and the benchmark scripts use
absolute imports such as ``from src.sorting import bubble_sort`` without
any ``sys.path`` manipulation::

    pip install -e .

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="csc5300-algorithm-lab",
    version="1.0.0",
    description=(
        "Week 1 algorithm laboratory: basic sorting algorithms with an "
        "empirical benchmarking and complexity-analysis framework."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Robert Deibel",
    url="https://github.com/RDeibel2025/CTX-Advanced-Algorithms",
    license="MIT",
    packages=find_packages(include=["src", "src.*", "benchmarks", "benchmarks.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "jupyter",
        "pytest",
        "scipy",
        "scikit-learn",
        "plotly",
        "seaborn",
    ],
    extras_require={"dev": ["pytest>=7.0"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
    ],
    keywords="algorithms sorting benchmarking complexity-analysis education",
)
