"""Tests for the hash table, under both collision strategies.

Nearly every test is parametrised over ``hash_strategy`` from conftest, so
chaining and linear probing are held to the same contract.

The most important tests here are in :class:`TestTombstones`. A linear-
probing delete that simply empties its slot breaks every probe sequence
that ran through it, and lookups for keys further along start silently
returning a miss. Nothing crashes, so only a test built for exactly that
case catches it: insert colliding keys, delete one from the middle of the
run, and check the others are still found.

Documented miss behaviour, tested throughout:

* ``get(key)`` returns ``None`` (or the supplied ``default``) for an absent
  key, like ``dict.get``;
* ``table[key]`` raises ``KeyError`` for an absent key, like ``dict``;
* ``delete(key)`` raises ``KeyError`` for an absent key.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import random

import pytest

from src.structures.hash_table import (
    CHAINING,
    DEFAULT_MAX_LOAD_FACTOR,
    LINEAR_PROBING,
    MIN_CAPACITY,
    HashTable,
    _EMPTY,
    _TOMBSTONE,
)


def colliding_keys(count: int) -> list:
    """Keys that all hash to slot 0 at every capacity up to 2**40."""
    return [k * 2 ** 40 for k in range(count)]


# ----------------------------------------------------------------------
# Basic contract
# ----------------------------------------------------------------------
class TestBasics:
    """Empty tables, single keys, updates and the documented miss behaviour."""

    def test_empty_table(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        assert len(table) == 0 and table.load_factor == 0.0
        assert table.get("x") is None and "x" not in table
        assert list(table.items()) == []
        with pytest.raises(KeyError):
            table.delete("x")
        with pytest.raises(KeyError):
            table["x"]

    def test_single_key(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        table.insert("k", 1)
        assert table.get("k") == 1 and table["k"] == 1 and "k" in table
        assert len(table) == 1

    def test_get_after_insert(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        for key in range(100):
            table.insert(key, key * 10)
        assert all(table.get(key) == key * 10 for key in range(100))

    def test_get_after_delete_returns_the_documented_miss(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        table.insert("gone", 1)
        table.delete("gone")
        assert table.get("gone") is None
        assert table.get("gone", "default") == "default"
        assert "gone" not in table
        with pytest.raises(KeyError):
            table["gone"]
        with pytest.raises(KeyError):
            table.delete("gone")

    def test_inserting_an_existing_key_replaces_its_value(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        table.insert("k", 1)
        table.insert("k", 2)
        assert table.get("k") == 2 and len(table) == 1

    def test_stored_none_is_distinguishable_from_absent(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        table.insert("k")
        assert table.get("k", "MISSING") is None
        assert "k" in table and table["k"] is None

    def test_dict_style_operators(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        table["a"] = 1
        assert table["a"] == 1
        del table["a"]
        assert "a" not in table

    def test_key_sets(self, hash_strategy, structure_key_sets):
        for case, keys in structure_key_sets.items():
            table = HashTable(strategy=hash_strategy)
            for key in keys:
                table.insert(key, -key)
            assert len(table) == len(set(keys)), case
            assert all(table.get(key) == -key for key in keys), case
            assert sorted(table.keys()) == sorted(set(keys)), case

    @pytest.mark.parametrize(
        "keys",
        [
            ["apple", "pear", "fig"],
            [3.5, -0.5, 2.25],
            [(1, "a"), (2, "b"), ("x", 3)],
            [-1, -2, -3, 0],
            [None, True, "mixed", 7],
        ],
        ids=["strings", "floats", "tuples", "negatives", "mixed"],
    )
    def test_any_hashable_key(self, hash_strategy, keys):
        table = HashTable(strategy=hash_strategy)
        for index, key in enumerate(keys):
            table.insert(key, index)
        assert [table.get(key) for key in keys] == list(range(len(keys)))

    def test_unhashable_keys_raise_type_error(self, hash_strategy):
        with pytest.raises(TypeError):
            HashTable(strategy=hash_strategy).insert([1, 2], "list")


class TestConstruction:
    """Arguments are validated, and capacity is a power of two."""

    def test_defaults(self):
        assert HashTable().strategy == CHAINING
        assert HashTable().max_load_factor == DEFAULT_MAX_LOAD_FACTOR[CHAINING]
        probing = HashTable(strategy=LINEAR_PROBING)
        assert probing.max_load_factor == DEFAULT_MAX_LOAD_FACTOR[LINEAR_PROBING]

    def test_chaining_and_probing_resize_at_different_loads(self):
        assert DEFAULT_MAX_LOAD_FACTOR[CHAINING] == 0.75
        assert DEFAULT_MAX_LOAD_FACTOR[LINEAR_PROBING] == 0.5

    @pytest.mark.parametrize("requested, actual", [(1, 8), (8, 8), (9, 16), (100, 128)])
    def test_capacity_rounds_up_to_a_power_of_two(self, requested, actual):
        assert HashTable(capacity=requested).capacity == actual
        assert MIN_CAPACITY == 8

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"strategy": "quadratic"},
            {"capacity": 0},
            {"capacity": -4},
            {"max_load_factor": 0},
            {"strategy": LINEAR_PROBING, "max_load_factor": 1.0},
            {"strategy": LINEAR_PROBING, "max_load_factor": 1.5},
        ],
        ids=["strategy", "zero-capacity", "negative-capacity", "zero-load",
             "probing-load-1", "probing-load-over-1"],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            HashTable(**kwargs)

    def test_chaining_may_exceed_load_one(self):
        """Chains can hold any number of entries, so no upper limit applies."""
        assert HashTable(strategy=CHAINING, max_load_factor=2.0).max_load_factor == 2.0


# ----------------------------------------------------------------------
# Tombstones (linear probing)
# ----------------------------------------------------------------------
class TestTombstones:
    """Deleting from the middle of a probe run must not strand later keys."""

    def build_run(self):
        """Four keys sharing slot 0 of a 16-slot table: slots 0, 1, 2 and 3."""
        table = HashTable(capacity=16, strategy=LINEAR_PROBING, max_load_factor=0.9)
        for key in (0, 16, 32, 48):
            table.insert(key, f"v{key}")
        assert table._keys[:5] == [0, 16, 32, 48, _EMPTY]
        return table

    def test_keys_after_a_deleted_slot_are_still_found(self):
        table = self.build_run()
        table.delete(16)
        assert table._keys[1] is _TOMBSTONE
        assert table.get(32) == "v32"
        assert table.get(48) == "v48"
        assert table.get(0) == "v0"
        assert table.get(16) is None
        assert table.tombstones == 1 and len(table) == 3

    @pytest.mark.parametrize("victim", [0, 16, 32, 48], ids=["first", "second", "third", "last"])
    def test_deleting_any_position_in_the_run(self, victim):
        table = self.build_run()
        table.delete(victim)
        for key in (0, 16, 32, 48):
            if key == victim:
                assert key not in table
            else:
                assert table.get(key) == f"v{key}"

    def test_emptying_the_slot_instead_would_lose_keys(self):
        """The bug tombstones prevent, reproduced deliberately."""
        table = self.build_run()
        table._keys[1] = _EMPTY          # a naive delete
        table._size -= 1
        assert table.get(32, "MISSING") == "MISSING"
        assert table.get(48, "MISSING") == "MISSING"

    def test_an_insert_reuses_the_tombstone(self):
        table = self.build_run()
        table.delete(16)
        table.insert(64, "v64")
        assert table._keys[1] == 64
        assert table.tombstones == 0
        assert table.get(64) == "v64" and table.get(48) == "v48"

    def test_reinserting_a_key_past_a_tombstone_does_not_duplicate_it(self):
        """The insert must scan the whole run, not stop at the first tombstone."""
        table = self.build_run()
        table.delete(16)
        table.insert(48, "updated")
        assert table.get(48) == "updated"
        assert len(table) == 3
        assert list(table.keys()).count(48) == 1

    def test_delete_insert_churn_never_strands_keys(self):
        rng = random.Random(17)
        table = HashTable(strategy=LINEAR_PROBING)
        model = {}
        for _ in range(20_000):
            key = rng.randint(0, 500)
            if key in model and rng.random() < 0.5:
                table.delete(key)
                del model[key]
            else:
                table.insert(key, key)
                model[key] = key
        assert dict(table.items()) == model
        assert all(table.get(k) == v for k, v in model.items())

    def test_tombstones_trigger_a_cleanup_rebuild(self):
        table = HashTable(capacity=64, strategy=LINEAR_PROBING)
        for key in range(30):
            table.insert(key)
        for key in range(30):
            table.delete(key)
        assert table.tombstones == 30
        table.insert(1000)
        table.insert(1001)
        table.insert(1002)
        assert table.tombstones < 30
        assert table.capacity == 64            # rebuilt in place, not grown

    def test_chaining_never_has_tombstones(self):
        table = HashTable(strategy=CHAINING)
        for key in range(50):
            table.insert(key)
        for key in range(50):
            table.delete(key)
        assert table.tombstones == 0


# ----------------------------------------------------------------------
# Load factor and rehashing
# ----------------------------------------------------------------------
class TestLoadFactorAndRehashing:
    """Load factor is exact, and rehashing keeps every live entry."""

    def test_load_factor_is_live_entries_over_capacity(self, hash_strategy):
        table = HashTable(capacity=64, strategy=hash_strategy)
        for key in range(16):
            table.insert(key)
        assert table.load_factor == 16 / 64
        for key in range(4):
            table.delete(key)
        assert table.load_factor == 12 / 64

    def test_load_factor_never_exceeds_the_threshold(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        for key in range(5000):
            table.insert(key)
            assert table.load_factor <= table.max_load_factor

    def test_rehash_preserves_every_live_entry(self, hash_strategy):
        rng = random.Random(18)
        table = HashTable(strategy=hash_strategy)
        keys = rng.sample(range(10 ** 9), 10_000)
        for key in keys:
            table.insert(key, -key)
        for key in keys[:3000]:
            table.delete(key)
        for key in rng.sample(range(10 ** 9, 2 * 10 ** 9), 3000):
            table.insert(key, 0)
        assert table.rehash_count > 5
        assert all(table.get(key) == -key for key in keys[3000:])
        assert all(key not in table for key in keys[:3000])

    def test_capacity_doubles_and_stays_a_power_of_two(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        capacities = set()
        for key in range(3000):
            table.insert(key)
            capacities.add(table.capacity)
        assert all(c & (c - 1) == 0 for c in capacities)
        ordered = sorted(capacities)
        assert all(b == 2 * a for a, b in zip(ordered, ordered[1:]))

    def test_rehash_drops_tombstones(self):
        table = HashTable(strategy=LINEAR_PROBING)
        for key in range(100):
            table.insert(key)
        for key in range(50):
            table.delete(key)
        assert table.tombstones == 50
        for key in range(1000, 1200):
            table.insert(key)
        assert table.tombstones < 50

    def test_amortised_rehash_work_is_below_two_moves_per_insert(self, hash_strategy):
        """Doubling means the rebuilds move fewer than 2n entries in total."""
        for size in (1000, 5000, 50_000):
            table = HashTable(strategy=hash_strategy)
            for key in range(size):
                table.insert(key)
            assert table.rehash_moves < 2 * size


# ----------------------------------------------------------------------
# Worst case and probe counting
# ----------------------------------------------------------------------
class TestWorstCaseAndProbes:
    """Engineered collisions degrade to O(n) - and stay correct."""

    def test_colliding_keys_all_share_slot_zero(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        keys = colliding_keys(300)
        for key in keys:
            table.insert(key, key)
        assert all(key & (table.capacity - 1) == 0 for key in keys)
        assert all(table.get(key) == key for key in keys)

    def test_probe_count_grows_linearly_with_collisions(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        keys = colliding_keys(200)
        for key in keys:
            table.insert(key)
        assert [table.probe_count(key) for key in keys] == list(range(1, 201))

    def test_probe_count_for_a_miss(self):
        chaining = HashTable(capacity=8)
        assert chaining.probe_count(5) == 0             # empty bucket
        for key in (0, 8, 16):
            chaining.insert(key)
        assert chaining.probe_count(24) == 3            # whole chain scanned

        probing = HashTable(capacity=8, strategy=LINEAR_PROBING, max_load_factor=0.9)
        assert probing.probe_count(5) == 1              # one empty slot
        for key in (0, 8, 16):
            probing.insert(key)
        assert probing.probe_count(24) == 4             # three full + one empty

    def test_average_case_lookups_are_short(self, hash_strategy):
        rng = random.Random(19)
        table = HashTable(strategy=hash_strategy)
        keys = rng.sample(range(10 ** 12), 20_000)
        for key in keys:
            table.insert(key)
        mean = sum(table.probe_count(k) for k in keys) / len(keys)
        assert mean < 2.0


# ----------------------------------------------------------------------
# Iteration and cross-checking
# ----------------------------------------------------------------------
class TestIterationAndAgreement:
    """keys, values and items agree with each other and with dict."""

    def test_iteration_views(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        for key in range(50):
            table.insert(key, key * 2)
        assert sorted(table) == list(range(50))
        assert sorted(table.keys()) == list(range(50))
        assert sorted(table.values()) == [k * 2 for k in range(50)]
        assert dict(table.items()) == {k: k * 2 for k in range(50)}

    def test_random_operations_against_dict(self, hash_strategy):
        rng = random.Random(20)
        table, model = HashTable(capacity=8, strategy=hash_strategy), {}
        for _ in range(4000):
            key = rng.randint(-300, 300)
            roll = rng.random()
            if roll < 0.5:
                table.insert(key, roll)
                model[key] = roll
            elif roll < 0.8 and model:
                victim = rng.choice(list(model))
                table.delete(victim)
                del model[victim]
            else:
                assert table.get(key) == model.get(key)
            assert len(table) == len(model)
        assert dict(table.items()) == model

    def test_both_strategies_end_in_the_same_state(self):
        rng = random.Random(21)
        operations = [(rng.randint(0, 200), rng.random() < 0.7) for _ in range(3000)]
        tables = [HashTable(strategy=s) for s in (CHAINING, LINEAR_PROBING)]
        for table in tables:
            for key, is_insert in operations:
                if is_insert:
                    table.insert(key, key)
                elif key in table:
                    table.delete(key)
        assert dict(tables[0].items()) == dict(tables[1].items())

    def test_repr_mentions_strategy_and_size(self, hash_strategy):
        table = HashTable(strategy=hash_strategy)
        table.insert(1)
        text = repr(table)
        assert hash_strategy in text and "size=1" in text
