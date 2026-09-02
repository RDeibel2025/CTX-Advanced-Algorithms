#!/usr/bin/env python3
"""Verify that this machine can build and run the CSC 5300 algorithm laboratory.

Run it from the repository root, with the project virtual environment active::

    source algorithms_course/bin/activate
    python check_environment.py

Four things are checked, and every failure is reported by name rather than
as a bare non-zero exit:

1. **Python version** — 3.9 or later is required. Anything older is a hard
   failure, because the project uses the built-in generic type syntax and
   :func:`statistics.fmean`.
2. **Packages** — all nine required third-party packages import cleanly,
   with their installed versions printed. Anything missing is listed with
   the exact ``pip install`` command that would fix it.
3. **Project structure** — every directory and file the assignment
   specifies is present. Anything absent is named.
4. **Version control** — ``git`` is on PATH and the working directory is
   inside a Git working tree.

Exit status is ``0`` when everything passes and ``1`` when anything fails,
so the script is usable as a gate in a shell script or in CI.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from typing import List, Tuple

#: Minimum interpreter this project supports.
MINIMUM_PYTHON = (3, 9)

#: (distribution name on PyPI, module name to import). The two differ for
#: scikit-learn, which is the usual reason a naive check reports a false
#: failure.
REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("jupyter", "jupyter"),
    ("pytest", "pytest"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("plotly", "plotly"),
    ("seaborn", "seaborn"),
]

#: Directories the assignment's project structure requires.
REQUIRED_DIRECTORIES: List[str] = [
    "src",
    "src/sorting",
    "src/searching",
    "src/graph",
    "src/dynamic_programming",
    "src/data_structures",
    "src/utils",
    "tests",
    "benchmarks",
    "notebooks",
    "docs",
    "docs/figures",
    "examples",
    ".github/workflows",
    # Week 2
    "analysis",
    "benchmarks/results",
    "tools",
]

#: Files the assignment's project structure requires.
REQUIRED_FILES: List[str] = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "check_environment.py",
    ".gitignore",
    ".github/workflows/tests.yml",
    "src/__init__.py",
    "src/sorting/__init__.py",
    "src/sorting/basic_sorts.py",
    "src/sorting/advanced_sorts.py",
    "src/searching/__init__.py",
    "src/graph/__init__.py",
    "src/dynamic_programming/__init__.py",
    "src/data_structures/__init__.py",
    "src/utils/__init__.py",
    "src/utils/benchmark.py",
    "src/utils/visualization.py",
    "src/utils/testing_helpers.py",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_sorting.py",
    "tests/test_searching.py",
    "tests/test_utils.py",
    "benchmarks/__init__.py",
    "benchmarks/sorting_benchmarks.py",
    "benchmarks/complexity_validation.py",
    "docs/performance_analysis.md",
    # Week 2 - divide and conquer
    "src/sorting/merge_sort.py",
    "src/sorting/quick_sort.py",
    "tests/test_merge_sort.py",
    "tests/test_quick_sort.py",
    "tests/test_sorting_comparison.py",
    "benchmarks/week2_performance.py",
    "analysis/week2_report.md",
    "analysis/week2_recurrences.md",
    "examples/week2_demo.py",
]

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

#: Repository root, resolved from this file's own location so the script
#: gives the same answer no matter which directory it is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def heading(text: str) -> None:
    """Print a section heading."""
    print()
    print(text)
    print("-" * len(text))


def check_python_version() -> bool:
    """Report the interpreter version and fail if it is below 3.9.

    Returns:
        True if the running interpreter is at least :data:`MINIMUM_PYTHON`.
    """
    heading("1. Python interpreter")

    version = sys.version_info
    printed = f"{version.major}.{version.minor}.{version.micro}"
    required = ".".join(str(part) for part in MINIMUM_PYTHON)

    print(f"       executable : {sys.executable}")
    print(f"       platform   : {platform.platform()}")
    print(f"       machine    : {platform.machine()}")

    if version[:2] < MINIMUM_PYTHON:
        print(f"{FAIL} Python {printed} is too old — this project requires {required}+.")
        print("       Install a newer Python and rebuild the virtual environment:")
        print("           python3 -m venv algorithms_course")
        return False

    print(f"{OK} Python {printed} (requires {required}+)")

    # Not a pass/fail condition, but running against the system interpreter
    # instead of the project venv is the single most common reason this
    # script passes on a machine where the project does not actually work.
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        print(f"{OK} Running inside a virtual environment: {sys.prefix}")
    else:
        print(f"{WARN} Not running inside a virtual environment.")
        print("       Expected:  source algorithms_course/bin/activate")
    return True


def check_packages() -> bool:
    """Import every required package and report its installed version.

    Returns:
        True if all nine imported successfully.
    """
    heading("2. Required packages")

    missing: List[str] = []
    for distribution, module_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            print(f"{FAIL} {distribution:<14} not importable — {exc}")
            missing.append(distribution)
            continue

        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown version")

        note = "" if distribution == module_name else f"  (imports as '{module_name}')"
        print(f"{OK} {distribution:<14} {version}{note}")

    if missing:
        print()
        print(f"{FAIL} {len(missing)} package(s) missing. Install them with:")
        print(f"           pip install {' '.join(missing)}")
        return False

    print(f"{OK} All {len(REQUIRED_PACKAGES)} required packages are installed.")
    return True


def check_structure() -> bool:
    """Confirm every required directory and file exists, naming what is absent.

    Returns:
        True if the project structure is complete.
    """
    heading("3. Project structure")

    missing_dirs = [
        path
        for path in REQUIRED_DIRECTORIES
        if not os.path.isdir(os.path.join(PROJECT_ROOT, path))
    ]
    missing_files = [
        path
        for path in REQUIRED_FILES
        if not os.path.isfile(os.path.join(PROJECT_ROOT, path))
    ]

    empty_files = [
        path
        for path in REQUIRED_FILES
        if os.path.isfile(os.path.join(PROJECT_ROOT, path))
        and os.path.getsize(os.path.join(PROJECT_ROOT, path)) == 0
    ]

    print(f"       root       : {PROJECT_ROOT}")

    if not missing_dirs:
        print(f"{OK} All {len(REQUIRED_DIRECTORIES)} required directories present.")
    else:
        for path in missing_dirs:
            print(f"{FAIL} missing directory: {path}/")

    if not missing_files:
        print(f"{OK} All {len(REQUIRED_FILES)} required files present.")
    else:
        for path in missing_files:
            print(f"{FAIL} missing file: {path}")

    if empty_files:
        for path in empty_files:
            print(f"{FAIL} file is empty: {path}")

    return not (missing_dirs or missing_files or empty_files)


def check_git() -> bool:
    """Confirm git is installed and the project is inside a working tree.

    Returns:
        True if git is available and this directory is version controlled.
    """
    heading("4. Version control")

    git_path = shutil.which("git")
    if git_path is None:
        print(f"{FAIL} git is not on PATH. Install it with: xcode-select --install")
        return False

    try:
        version = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        print(f"{OK} {version}  ({git_path})")
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"{FAIL} could not run git: {exc}")
        return False

    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"{FAIL} could not query the Git working tree: {exc}")
        return False

    if inside.returncode != 0 or inside.stdout.strip() != "true":
        print(f"{FAIL} {PROJECT_ROOT} is not inside a Git working tree.")
        print("       Initialise it with: git init -b main")
        return False

    print(f"{OK} Project directory is inside a Git working tree.")

    # Informational only: a repo with no remote still satisfies the
    # requirement, so this never changes the exit status.
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=15,
    )
    if remote.returncode == 0 and remote.stdout.strip():
        print(f"{OK} origin remote: {remote.stdout.strip()}")
    else:
        print(f"{WARN} No 'origin' remote configured (not required).")

    return True


def main() -> int:
    """Run every check and return the process exit status.

    Returns:
        ``0`` if all checks passed, ``1`` otherwise.
    """
    print("=" * 72)
    print("CSC 5300 Advanced Algorithms — Environment Check")
    print("Robert Deibel · Week 1 Project: Algorithm Laboratory Setup")
    print("=" * 72)

    checks = [
        ("Python interpreter", check_python_version),
        ("Required packages", check_packages),
        ("Project structure", check_structure),
        ("Version control", check_git),
    ]

    outcomes = [(label, runner()) for label, runner in checks]

    heading("Summary")
    for label, passed in outcomes:
        print(f"{OK if passed else FAIL} {label}")

    failures = [label for label, passed in outcomes if not passed]
    print()
    if failures:
        print(f"{FAIL} {len(failures)} of {len(outcomes)} checks failed: "
              + ", ".join(failures))
        print("Environment is NOT ready. Fix the items above and re-run.")
        return 1

    print(f"{OK} All {len(outcomes)} checks passed. Environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
