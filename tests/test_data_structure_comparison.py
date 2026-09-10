"""Cross-structure agreement: the same keys, through every structure.

The heap, the AVL tree and both hash-table strategies were written
independently, and each is tested on its own terms elsewhere. This file
checks the thing none of those can: that given the **same key set**, they
agree - on which keys are members, on which are not, and on the order the
keys come back in when order is defined. Python's own ``set`` and ``dict``
are the ground truth.

A heap cannot delete an arbitrary key, so the deletion tests use the heap's
own removal operation - extracting the smallest keys - and check that the
tree and tables, after deleting those same keys, agree with it.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

from src.structures.avl_tree import AVLTree
from src.structures.hash_table import CHAINING, LINEAR_PROBING, HashTable
from src.structures.heap import MaxHeap, MinHeap
from tests.conftest import MAPPING_STRUCTURES, lookup

KEY_SET_NAMES = [
    "empty", "single", "duplicates", "ascending", "descending", "negatives", "random",
]


def build_all(keys: List[Any]) -> Dict[str, Any]:
    """Load ``keys`` into every structure, plus Python's set as ground truth."""
    structures: Dict[str, Any] = {
        "min_heap": MinHeap(),
        "max_heap": MaxHeap(),
        "avl_tree": AVLTree(),
        "hash_chaining": HashTable(strategy=CHAINING),
        "hash_linear_probing": HashTable(strategy=LINEAR_PROBING),
    }
    for key in keys:
        for name, structure in structures.items():
            if name.endswith("heap"):
                structure.insert(key)
            else:
                structure.insert(key, key)
    structures["reference"] = set(keys)
    return structures


def absent_keys(keys: List[int], count: int = 50) -> List[int]:
    """Keys guaranteed not to be in ``keys``."""
    top = max(keys, default=0)
    return [top + 1 + step for step in range(count)] + [
        min(keys, default=0) - 1 - step for step in range(count)
    ]


class TestMembershipAgreement:
    """Every structure agrees on what is in the set and what is not."""

    @pytest.mark.parametrize("case", KEY_SET_NAMES)
    def test_all_structures_agree_on_present_keys(self, case, structure_key_sets):
        keys = structure_key_sets[case]
        structures = build_all(keys)
        reference = structures.pop("reference")
        for key in reference:
            for name, structure in structures.items():
                assert key in structure, f"{name} lost {key!r} ({case})"

    @pytest.mark.parametrize("case", KEY_SET_NAMES)
    def test_all_structures_agree_on_absent_keys(self, case, structure_key_sets):
        keys = structure_key_sets[case]
        structures = build_all(keys)
        structures.pop("reference")
        for key in absent_keys(keys):
            for name, structure in structures.items():
                assert key not in structure, f"{name} invented {key!r} ({case})"

    @pytest.mark.parametrize("case", KEY_SET_NAMES)
    def test_sizes_agree(self, case, structure_key_sets):
        """Heaps keep duplicates; the tree and tables keep one of each."""
        keys = structure_key_sets[case]
        structures = build_all(keys)
        assert len(structures["min_heap"]) == len(keys)
        assert len(structures["max_heap"]) == len(keys)
        for name in ("avl_tree", "hash_chaining", "hash_linear_probing"):
            assert len(structures[name]) == len(set(keys)), name


class TestOrderAgreement:
    """Where an order is defined, every structure produces the same one."""

    @pytest.mark.parametrize("case", KEY_SET_NAMES)
    def test_sorted_output_agrees(self, case, structure_key_sets):
        keys = structure_key_sets[case]
        structures = build_all(keys)
        distinct_sorted = sorted(set(keys))

        min_heap = structures["min_heap"]
        from_min_heap = [min_heap.extract_min() for _ in range(len(min_heap))]
        max_heap = structures["max_heap"]
        from_max_heap = [max_heap.extract_max() for _ in range(len(max_heap))]

        assert from_min_heap == sorted(keys)
        assert from_max_heap == sorted(keys, reverse=True)
        assert sorted(set(from_min_heap)) == distinct_sorted
        assert structures["avl_tree"].in_order() == distinct_sorted
        assert sorted(structures["hash_chaining"].keys()) == distinct_sorted
        assert sorted(structures["hash_linear_probing"].keys()) == distinct_sorted


class TestAgreementAfterDeletion:
    """Removing the same keys from every structure leaves them in agreement."""

    @pytest.mark.parametrize("removed", [1, 10, 100, 250])
    def test_removing_the_smallest_keys(self, removed):
        rng = random.Random(22)
        keys = rng.sample(range(-5000, 5000), 500)
        structures = build_all(keys)
        reference = structures.pop("reference")

        heap = structures.pop("min_heap")
        structures.pop("max_heap")
        gone = [heap.extract_min() for _ in range(removed)]
        assert gone == sorted(keys)[:removed]

        for key in gone:
            reference.discard(key)
            for structure in structures.values():
                structure.delete(key)

        remaining_from_heap = sorted(heap.to_list())
        assert remaining_from_heap == sorted(reference)
        for name, structure in structures.items():
            assert len(structure) == len(reference), name
            assert all(key in structure for key in reference), name
            assert all(key not in structure for key in gone), name


class TestMappingStructuresAgree:
    """The three key-value structures agree on values, not just membership."""

    def test_random_workload_against_dict(self, mapping_factory):
        rng = random.Random(23)
        structure, model = mapping_factory(), {}
        for _ in range(3000):
            key = rng.randint(0, 400)
            if key in model and rng.random() < 0.4:
                structure.delete(key)
                del model[key]
            else:
                value = rng.random()
                structure.insert(key, value)
                model[key] = value
        assert len(structure) == len(model)
        for key in range(401):
            assert lookup(structure, key) == model.get(key)

    def test_all_three_agree_with_each_other(self):
        rng = random.Random(24)
        operations = [(rng.randint(0, 300), rng.random()) for _ in range(4000)]
        built = {name: factory() for name, factory in MAPPING_STRUCTURES.items()}
        for name, structure in built.items():
            present = set()
            for key, roll in operations:
                if key in present and roll < 0.4:
                    structure.delete(key)
                    present.discard(key)
                else:
                    structure.insert(key, key * 3)
                    present.add(key)
        snapshots = {
            name: {key: lookup(structure, key) for key in range(301) if key in structure}
            for name, structure in built.items()
        }
        first, *rest = snapshots.values()
        assert all(snapshot == first for snapshot in rest)
