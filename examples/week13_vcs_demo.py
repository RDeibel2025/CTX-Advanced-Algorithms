#!/usr/bin/env python3
"""Runnable demonstration of the Week 13 mini version control system.

    python examples/week13_vcs_demo.py

Twenty synthetic files (seeded text, not real source code) go through ten
commits: the first commit adds all twenty, and each later one modifies a few
files and sometimes adds or removes one. Every commit is a persistent treap
(path -> SHA-1 of the content) that shares every unchanged node with its
parent. The demo prints:

1. The log, newest first, with the files changed and new nodes per commit.
2. One diff (first commit against the last) with ``nodes_visited``, and a
   one-file diff that shows the Merkle pruning skipping most of the tree.
3. The sharing statistics: distinct nodes stored against the nodes a full
   copy per commit would need.

A plain ``dict`` snapshot is kept beside every commit. Every ``checkout`` is
asserted equal to its snapshot, every diff is asserted equal to the one
computed from the two dicts, and the ids are asserted to follow the spec's
formula ``sha1(root_hash + parent_id + message)[:12]``. Randomness comes from
``make_rng(derive_seed(42, ...))``, so the output is the same on every run.
It finishes in well under a second.

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.advds.applications.version_control import MiniVCS, sha1_hex  # noqa: E402
from src.utils.random_utils import derive_seed, make_rng  # noqa: E402

WIDTH = 74
SEED = 42
N_FILES = 20
N_COMMITS = 10


def banner(number: int, title: str) -> None:
    """Print a numbered section heading."""
    print()
    print("=" * WIDTH)
    print(f"{number}. {title}")
    print("=" * WIDTH)


def show(label: str, value: Any) -> None:
    """Print one labelled result line."""
    print(f"  {label:<40} {value}")


def synthetic_content(rng: random.Random, path: str, revision: int) -> str:
    """A few lines of seeded synthetic text for one file revision."""
    words = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma", "theta"]
    lines = [f"# synthetic file {path}, revision {revision}"]
    for _ in range(rng.randint(3, 8)):
        lines.append(" ".join(rng.choice(words) for _ in range(6)))
    return "\n".join(lines) + "\n"


def expected_diff(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, List[str]]:
    """The diff of two plain snapshots, by set operations and sorting."""
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "modified": sorted(p for p in set(a) & set(b) if a[p] != b[p]),
    }


def main() -> int:
    """Build the history, print the log, a diff and the statistics; return 0."""
    print("=" * WIDTH)
    print("CSC 5300 Advanced Algorithms - Week 13 mini version control demonstration")
    print("Commits as persistent treaps, diffs pruned by Merkle hashes")
    print("=" * WIDTH)
    print(f"  {N_FILES} synthetic files, {N_COMMITS} commits, seed {SEED}. "
          "Every check is asserted.")

    rng = make_rng(derive_seed(SEED, "week13_vcs_demo"))
    vcs = MiniVCS()
    working: Dict[str, str] = {}
    snapshots: Dict[str, Dict[str, str]] = {}
    ids: List[str] = []
    next_file = N_FILES

    for number in range(1, N_COMMITS + 1):
        if number == 1:
            for i in range(N_FILES):
                path = f"src/module_{i:02d}.py"
                working[path] = synthetic_content(rng, path, 1)
                vcs.add(path, working[path])
            message = f"add {N_FILES} synthetic files"
        else:
            paths = sorted(working)
            changed = rng.sample(paths, 1 if number == 2 else rng.randint(1, 3))
            for path in changed:
                working[path] = synthetic_content(rng, path, number)
                vcs.add(path, working[path])
            parts = [f"modify {len(changed)}"]
            if number % 3 == 0:
                path = f"src/module_{next_file:02d}.py"
                next_file += 1
                working[path] = synthetic_content(rng, path, number)
                vcs.add(path, working[path])
                parts.append(f"add {path}")
            if number % 4 == 0:
                path = rng.choice(sorted(p for p in working if p not in changed))
                del working[path]
                vcs.rm(path)
                parts.append(f"remove {path}")
            message = ", ".join(parts)
        parent = ids[-1] if ids else ""
        cid = vcs.commit(message)
        assert cid == sha1_hex(vcs.tree(cid).root_hash + parent + message)[:12]
        ids.append(cid)
        snapshots[cid] = dict(working)
        assert vcs.checkout(cid) == snapshots[cid]

    banner(1, "The log, newest first")
    history = {h["id"]: h for h in vcs.history()}
    print(f"  {'id':<14}{'files':>6}{'changed':>9}{'new nodes':>11}  message")
    for cid, message in vcs.log():
        h = history[cid]
        print(f"  {cid:<14}{h['files']:>6}{h['files_changed']:>9}{h['new_nodes']:>11}  {message}")
    assert [cid for cid, _ in vcs.log()] == ids[::-1]
    for k, cid in enumerate(ids):
        before = snapshots[ids[k - 1]] if k else {}
        d = expected_diff(before, snapshots[cid])
        assert history[cid]["files_changed"] == sum(len(v) for v in d.values())

    banner(2, "Diffs with Merkle pruning")
    first, last = ids[0], ids[-1]
    d = vcs.diff(first, last)
    assert {k: d[k] for k in ("added", "removed", "modified")} == expected_diff(
        snapshots[first], snapshots[last]
    )
    print(f"  diff({first}, {last}), first commit against the last:")
    show("  added", d["added"])
    show("  removed", d["removed"])
    show("  modified", f"{len(d['modified'])} files")
    show("  nodes_visited", f"{d['nodes_visited']} (two trees of {len(snapshots[first])} and "
         f"{len(snapshots[last])} nodes)")
    for a, b in zip(ids, ids[1:]):
        step = vcs.diff(a, b)
        assert {k: step[k] for k in ("added", "removed", "modified")} == expected_diff(
            snapshots[a], snapshots[b]
        )
    # Commit 2 modifies exactly one file, so its diff shows the pruning alone.
    a, b = ids[0], ids[1]
    step = vcs.diff(a, b)
    assert len(step["modified"]) == 1 and not step["added"] and not step["removed"]
    show(f"diff({a}, {b})", f"modified {step['modified']}")
    show("  nodes_visited", f"{step['nodes_visited']} of the {2 * len(snapshots[b])} "
         "nodes in the two trees")
    assert step["nodes_visited"] < 2 * len(snapshots[b])

    banner(3, "Node sharing across all commits")
    stats = vcs.stats()
    for key in ("commits", "files", "objects", "unique_nodes", "full_copy_nodes", "saved_nodes"):
        show(key, stats[key])
    show("stored_fraction", f"{stats['stored_fraction']:.3f}")
    assert stats["commits"] == N_COMMITS
    assert stats["full_copy_nodes"] == sum(len(s) for s in snapshots.values())
    assert stats["unique_nodes"] == sum(h["new_nodes"] for h in history.values())
    assert stats["unique_nodes"] < stats["full_copy_nodes"]

    print()
    print("=" * WIDTH)
    print("All demonstrations held.")
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
