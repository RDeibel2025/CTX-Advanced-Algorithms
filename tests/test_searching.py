"""Tests for the searching package — reserved for a later week.

This module is part of the required CSC 5300 project structure. Week 1
implements no search algorithms, so there is nothing yet to test for
correctness. What *is* worth asserting now is that the package exists and
imports cleanly: a placeholder package that has quietly broken is a
problem worth catching before code is added to it.

Linear, binary and interpolation search tests belong here in a later week.

Author:
    Robert Deibel — CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

import importlib


def test_searching_package_imports():
    """The reserved package is importable, so later weeks can build on it."""
    module = importlib.import_module("src.searching")
    assert module.__doc__, "the placeholder package should document its status"


def test_searching_package_is_empty_for_now():
    """Nothing is exported yet — Week 1 defines no search algorithms."""
    module = importlib.import_module("src.searching")
    public_names = [name for name in dir(module) if not name.startswith("_")]
    assert public_names == [], f"unexpected public names: {public_names}"
