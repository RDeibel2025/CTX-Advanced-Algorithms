"""Mini version control: commits as persistent maps, a Merkle-pruned diff, and node sharing.

The idea
--------
A commit is a snapshot of every file. Copying the whole file table at every
commit costs one table entry per file per commit. :class:`MiniVCS` keeps the
table in a :class:`~src.advds.trees.persistent_tree.PersistentMap` instead, a
treap with path copying, so a commit that changes c of n files creates only
O(c log n) expected new nodes and shares every other node with its parent.
Every old commit stays readable, because no node is ever changed after it
is built.

Objects, trees and commits
--------------------------
* :meth:`MiniVCS.add` stores a file's content in an object store keyed by
  ``sha1(content.encode("utf-8")).hexdigest()`` (a "blob", as in git) and
  stages ``path -> that 40-hex hash`` on the working map. :meth:`MiniVCS.rm`
  stages the removal of a path. Staging only replaces the working map by a
  new version; nothing that is already committed changes.
* :meth:`MiniVCS.commit` freezes the working map as the commit's tree. Its
  id is ``sha1((root_hash + parent_id + message).encode("utf-8"))
  .hexdigest()[:12]``: plain string concatenation, with no separator, of the
  tree's 128-hex Merkle :attr:`~src.advds.trees.persistent_tree.PersistentMap.root_hash`,
  the parent commit's 12-hex id (``""`` for the first commit) and the
  message, encoded as UTF-8; the first 12 hex digits of the SHA-1 are kept.
* History is linear: every commit's parent is the previous commit.
  :meth:`MiniVCS.checkout` reads a commit's files back and does not move
  anything, so the next commit still follows the newest one.

Ids are deterministic across runs and machines: the treap shape depends only
on the set of paths (priorities ``stable_hash64(path)``), every node hash is
``blake2b`` over a fixed encoding, and the commit id is ``sha1`` over fixed
strings, so nothing depends on ``PYTHONHASHSEED`` or on memory addresses.

The Merkle-pruned diff
----------------------
Every treap node stores a Merkle hash of its key, its value and its
children's hashes, so two subtrees with equal hashes hold the same
``(path, blob hash)`` pairs. Because the shape depends only on the key set,
two commits that differ in a few files have the same shape almost
everywhere. :func:`diff_maps` walks both trees together from the roots:

1. A pair of aligned subtrees whose Merkle hashes are equal (or that are the
   same shared object, which implies equal hashes) is skipped without
   looking inside. Two empty subtrees are skipped too.
2. If the two aligned nodes have the same key, their values are compared
   (a different blob hash means the file was modified) and the walk recurses
   on left/left and right/right. This is sound because the left subtrees
   hold exactly the keys below that key on both sides, and the right
   subtrees the keys above it.
3. If the keys differ, or one side is empty, a file was added or removed
   there and the shapes no longer line up. The walk falls back to an
   in-order merge of just those two subtrees (they cover the same key
   interval), which reports every added, removed and modified path inside.

The walk handles the left pair, then the node, then the right pair, so the
``added``, ``removed`` and ``modified`` lists come out in increasing path
order without a sort.

``nodes_visited`` counts the nodes the diff examines: both nodes of every
aligned pair of non-empty subtrees the walk reaches (their hashes are
compared, so the roots of the pairs it then skips count too, and a shared
object counts the same as an equal copy, so the figure depends only on the
contents of the two maps), plus every node the in-order fallback reads. One
modified file at depth d (the root is depth 0) costs at most ``4d + 6``
nodes (the pair on the path and its sibling pair on every level, plus the
modified node's two child pairs), O(log n) expected, against the 2n nodes a
full comparison reads;
two identical maps cost 2 (the roots). :meth:`MiniVCS.diff` reports it so
the benchmark can compare it with the tree size.

Sharing statistics
------------------
:meth:`MiniVCS.stats` counts the distinct node objects across all commit
trees (what is really stored, shared nodes once) against the sum of the tree
sizes over all commits (what a full copy of the table per commit would
need). :meth:`MiniVCS.history` gives per-commit figures for a history plot:
the number of files changed against the parent and the number of new nodes
the commit's tree added to the store.

Examples:
    >>> vcs = MiniVCS()
    >>> vcs.add("a.txt", "alpha"); vcs.add("b.txt", "beta")
    >>> c1 = vcs.commit("add a and b")
    >>> vcs.add("b.txt", "beta v2"); vcs.add("c.txt", "gamma")
    >>> c2 = vcs.commit("modify b, add c")
    >>> vcs.rm("a.txt")
    >>> c3 = vcs.commit("remove a")
    >>> d = vcs.diff(c1, c3)
    >>> d["added"], d["removed"], d["modified"]
    (['c.txt'], ['a.txt'], ['b.txt'])
    >>> vcs.checkout(c1)
    {'a.txt': 'alpha', 'b.txt': 'beta'}
    >>> [message for _, message in vcs.log()]
    ['remove a', 'modify b, add c', 'add a and b']

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from src.advds.trees.persistent_tree import MapNode, PersistentMap

__all__ = ["COMMIT_ID_LENGTH", "MiniVCS", "diff_maps", "sha1_hex"]

#: Number of hex digits of the SHA-1 kept as a commit id.
COMMIT_ID_LENGTH: int = 12


def sha1_hex(text: str) -> str:
    """Return the SHA-1 hexdigest of ``text`` encoded as UTF-8.

    Used for blob ids (``sha1(content)``) and, truncated, for commit ids.

    Args:
        text: Any string.

    Returns:
        40 lowercase hex characters.

    Raises:
        TypeError: If ``text`` is not a ``str``.

    Time Complexity:
        O(len(text)).

    Space Complexity:
        O(len(text)) for the encoded bytes.

    Examples:
        >>> sha1_hex("")
        'da39a3ee5e6b4b0d3255bfef95601890afd80709'
        >>> sha1_hex("abc")
        'a9993e364706816aba3e25717850c26c9cd0d89d'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _inorder(node: Optional[MapNode]) -> List[MapNode]:
    """Return the nodes of a subtree in increasing key order (iterative in-order walk).

    Args:
        node: The subtree root, or ``None``.

    Returns:
        A new list of the subtree's nodes, smallest key first.

    Raises:
        Nothing.

    Time Complexity:
        O(s) for a subtree of s nodes.

    Space Complexity:
        O(s) for the result, O(depth) for the stack.

    Examples:
        >>> m = PersistentMap({"b": "2", "a": "1", "c": "3"})
        >>> [n.key for n in _inorder(m.root)], _inorder(None)
        (['a', 'b', 'c'], [])
    """
    out: List[MapNode] = []
    stack: List[MapNode] = []
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node)
        node = node.right
    return out


def diff_maps(old: PersistentMap, new: PersistentMap) -> Dict[str, Any]:
    """Diff two persistent maps by walking both treaps together, skipping equal Merkle hashes.

    See the module docstring for the walk. Aligned subtrees with equal
    hashes are skipped; aligned nodes with the same key are compared by
    value and the walk recurses left/left and right/right; where the keys
    differ (or one side is empty) the two subtrees are merged in order.

    Args:
        old: The earlier map.
        new: The later map.

    Returns:
        A dict with ``"added"`` (keys in ``new`` only), ``"removed"`` (keys
        in ``old`` only), ``"modified"`` (keys in both with different
        values), each a list in increasing key order, and
        ``"nodes_visited"``, the number of nodes examined: both nodes of
        every aligned non-empty pair whose hashes were compared, plus every
        node read by an in-order fallback.

    Raises:
        TypeError: If either argument is not a :class:`PersistentMap`.

    Time Complexity:
        O(d + s) where d is the number of aligned pairs reached and s the
        total size of the subtrees merged in order; O(log n) expected for a
        single modified key, O(n) at worst.

    Space Complexity:
        O(depth + s) for the recursion and the merged subtrees, plus the
        output lists.

    Examples:
        >>> a = PersistentMap({"x": "1", "y": "2", "z": "3"})
        >>> b = a.set("y", "9").delete("z").set("w", "0")
        >>> d = diff_maps(a, b)
        >>> d["added"], d["removed"], d["modified"]
        (['w'], ['z'], ['y'])
        >>> diff_maps(a, a)
        {'added': [], 'removed': [], 'modified': [], 'nodes_visited': 2}
        >>> diff_maps(PersistentMap(), PersistentMap())["nodes_visited"]
        0
    """
    if not isinstance(old, PersistentMap) or not isinstance(new, PersistentMap):
        raise TypeError("diff_maps needs two PersistentMap objects")
    added: List[Any] = []
    removed: List[Any] = []
    modified: List[Any] = []
    visited = 0

    def merge(x: Optional[MapNode], y: Optional[MapNode]) -> None:
        """In-order merge of two subtrees covering the same key interval."""
        nonlocal visited
        xs = _inorder(x)
        ys = _inorder(y)
        visited += len(xs) + len(ys)
        i = j = 0
        while i < len(xs) and j < len(ys):
            a, b = xs[i], ys[j]
            if a.key == b.key:
                if a.value != b.value:
                    modified.append(a.key)
                i += 1
                j += 1
            elif a.key < b.key:
                removed.append(a.key)
                i += 1
            else:
                added.append(b.key)
                j += 1
        while i < len(xs):
            removed.append(xs[i].key)
            i += 1
        while j < len(ys):
            added.append(ys[j].key)
            j += 1

    def walk(x: Optional[MapNode], y: Optional[MapNode]) -> None:
        """Compare two aligned subtrees, left part first, so output stays in key order."""
        nonlocal visited
        if x is None and y is None:
            return
        if x is not None and y is not None:
            # Both roots are examined whether or not they are the same shared
            # object, so the count depends only on the contents of the maps.
            visited += 2
            if x is y or x.hash == y.hash:
                return
            if x.key == y.key:
                walk(x.left, y.left)
                if x.value != y.value:
                    modified.append(x.key)
                walk(x.right, y.right)
                return
            visited -= 2  # the merge below counts these two nodes itself
        merge(x, y)

    walk(old.root, new.root)
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "nodes_visited": visited,
    }


class _Commit(NamedTuple):
    """One commit record (internal).

    Attributes:
        id: The 12-hex commit id.
        parent: The parent's id, ``""`` for the first commit.
        message: The commit message.
        tree: The committed :class:`PersistentMap` (path -> blob hash).
        number: 1-based position in the history.
        files_changed: Paths added, removed or modified against the parent.
        new_nodes: Tree nodes not present in any earlier commit.
    """

    id: str
    parent: str
    message: str
    tree: PersistentMap
    number: int
    files_changed: int
    new_nodes: int


class MiniVCS:
    """A tiny linear version-control system on top of a persistent treap.

    Files are staged with :meth:`add` and :meth:`rm` on a working
    :class:`PersistentMap` (path -> ``sha1(content)``); :meth:`commit`
    freezes it. Contents live once each in an object store keyed by their
    SHA-1. See the module docstring for the id formula, the diff and the
    statistics.

    Raises:
        Nothing on construction.

    Time Complexity:
        :meth:`add` and :meth:`rm` O(len(content) + log n) expected;
        :meth:`commit` O(log n + changed nodes) expected for a few changes;
        :meth:`checkout` O(n); :meth:`diff` O(log n) expected per changed
        file.

    Space Complexity:
        Distinct contents once each, plus the distinct treap nodes of all
        commits: n nodes for the first commit and O(log n) expected per
        changed file after that.

    Examples:
        >>> vcs = MiniVCS()
        >>> vcs.add("readme.md", "hello")
        >>> c1 = vcs.commit("first")
        >>> len(c1), vcs.head == c1, vcs.checkout(c1)
        (12, True, {'readme.md': 'hello'})
    """

    def __init__(self) -> None:
        """Start with no commits, an empty working map and an empty object store.

        Returns:
            None.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> vcs = MiniVCS()
            >>> len(vcs), vcs.head, vcs.stats()["objects"]
            (0, None, 0)
        """
        self._working: PersistentMap = PersistentMap()
        self._objects: Dict[str, str] = {}
        self._commits: Dict[str, _Commit] = {}
        self._order: List[str] = []
        # Every node of every committed tree, by identity (MapNode keeps the
        # default identity hash). Holding them also keeps them alive.
        self._stored: Set[MapNode] = set()

    # ------------------------------------------------------------ helpers

    def _commit_of(self, commit_id: Any) -> _Commit:
        """Return the record of ``commit_id``.

        Args:
            commit_id: A commit id returned by :meth:`commit`.

        Returns:
            The internal commit record.

        Raises:
            KeyError: If there is no such commit.

        Time Complexity:
            O(1) expected.

        Space Complexity:
            O(1).

        Examples:
            >>> MiniVCS()._commit_of("abc")
            Traceback (most recent call last):
                ...
            KeyError: "unknown commit 'abc'"
        """
        record = self._commits.get(commit_id) if isinstance(commit_id, str) else None
        if record is None:
            raise KeyError(f"unknown commit {commit_id!r}")
        return record

    def _absorb(self, root: Optional[MapNode]) -> int:
        """Add the nodes of a tree to the store and return how many were new.

        A node already stored was absorbed together with its whole subtree
        (nodes are immutable), so the walk stops there.

        Args:
            root: The root of a committed tree.

        Returns:
            The number of nodes not stored before.

        Raises:
            Nothing.

        Time Complexity:
            O(new nodes + their children).

        Space Complexity:
            O(new nodes).

        Examples:
            >>> vcs = MiniVCS()
            >>> m = PersistentMap({"a": "1", "b": "2"})
            >>> vcs._absorb(m.root), vcs._absorb(m.root), vcs._absorb(m.set("a", "3").root) > 0
            (2, 0, True)
        """
        new = 0
        stack: List[MapNode] = [] if root is None else [root]
        stored = self._stored
        while stack:
            node = stack.pop()
            if node in stored:
                continue
            stored.add(node)
            new += 1
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        return new

    # ------------------------------------------------------------ staging

    def add(self, path: str, content: str) -> None:
        """Stage a file: store ``content`` under its SHA-1 and map ``path`` to that hash.

        Adding a path that is already tracked replaces its content (a
        modification). Adding the same content again changes nothing.

        Args:
            path: A non-empty file path.
            content: The file's text.

        Returns:
            None.

        Raises:
            TypeError: If ``path`` or ``content`` is not a ``str``.
            ValueError: If ``path`` is empty.

        Time Complexity:
            O(len(content)) for the hash plus O(log n) expected for the map.

        Space Complexity:
            O(len(content)) once per distinct content, O(log n) expected
            new working-map nodes.

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "x")
            >>> vcs.add("", "x")
            Traceback (most recent call last):
                ...
            ValueError: path must not be empty
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be a str, got {type(path).__name__}")
        if not isinstance(content, str):
            raise TypeError(f"content must be a str, got {type(content).__name__}")
        if not path:
            raise ValueError("path must not be empty")
        blob = sha1_hex(content)
        self._objects[blob] = content
        self._working = self._working.set(path, blob)

    def rm(self, path: str) -> None:
        """Stage the removal of a tracked path.

        The content stays in the object store, because earlier commits still
        refer to it.

        Args:
            path: A path in the working map.

        Returns:
            None.

        Raises:
            TypeError: If ``path`` is not a ``str``.
            KeyError: If ``path`` is not tracked.

        Time Complexity:
            O(log n) expected.

        Space Complexity:
            O(log n) expected new working-map nodes.

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "x")
            >>> vcs.rm("a.txt")
            >>> vcs.rm("a.txt")
            Traceback (most recent call last):
                ...
            KeyError: "path not tracked: 'a.txt'"
        """
        if not isinstance(path, str):
            raise TypeError(f"path must be a str, got {type(path).__name__}")
        if path not in self._working:
            raise KeyError(f"path not tracked: {path!r}")
        self._working = self._working.delete(path)

    # ------------------------------------------------------------ commits

    def commit(self, message: str) -> str:
        """Freeze the working map as a new commit and return its id.

        ``id = sha1((root_hash + parent_id + message).encode("utf-8"))
        .hexdigest()[:12]`` with ``parent_id = ""`` for the first commit
        (plain concatenation, no separator). A commit with no staged change
        is allowed; it shares its parent's whole tree.

        Args:
            message: The commit message.

        Returns:
            The 12-hex commit id.

        Raises:
            TypeError: If ``message`` is not a ``str``.
            ValueError: If the id is already taken (cannot happen in a linear
                history, where every parent is new; checked anyway).

        Time Complexity:
            O(changed paths x log n) expected for the diff against the parent
            plus O(new nodes) to record them.

        Space Complexity:
            O(new nodes).

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "x")
            >>> cid = vcs.commit("first")
            >>> cid == sha1_hex(vcs.tree(cid).root_hash + "" + "first")[:12]
            True
        """
        if not isinstance(message, str):
            raise TypeError(f"message must be a str, got {type(message).__name__}")
        parent = self._order[-1] if self._order else ""
        tree = self._working
        cid = sha1_hex(tree.root_hash + parent + message)[:COMMIT_ID_LENGTH]
        if cid in self._commits:
            raise ValueError(f"commit id {cid!r} already exists")
        parent_tree = self._commits[parent].tree if parent else PersistentMap()
        delta = diff_maps(parent_tree, tree)
        changed = len(delta["added"]) + len(delta["removed"]) + len(delta["modified"])
        new_nodes = self._absorb(tree.root)
        self._commits[cid] = _Commit(
            cid, parent, message, tree, len(self._order) + 1, changed, new_nodes
        )
        self._order.append(cid)
        return cid

    @property
    def head(self) -> Optional[str]:
        """The id of the newest commit, or ``None`` before the first commit.

        Returns:
            A 12-hex id or ``None``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> MiniVCS().head is None
            True
        """
        return self._order[-1] if self._order else None

    def tree(self, commit_id: str) -> PersistentMap:
        """Return the committed :class:`PersistentMap` (path -> blob hash) of a commit.

        Args:
            commit_id: A commit id.

        Returns:
            The immutable map; it is shared, not copied.

        Raises:
            KeyError: If there is no such commit.

        Time Complexity:
            O(1) expected.

        Space Complexity:
            O(1).

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "x")
            >>> vcs.tree(vcs.commit("first")).items() == [("a.txt", sha1_hex("x"))]
            True
        """
        return self._commit_of(commit_id).tree

    def checkout(self, commit_id: str) -> Dict[str, str]:
        """Return the files of a commit as ``{path: content}``, in path order.

        Reads only; the working map and the history are not changed.

        Args:
            commit_id: A commit id.

        Returns:
            A new dict from path to file content.

        Raises:
            KeyError: If there is no such commit.

        Time Complexity:
            O(n) for n files.

        Space Complexity:
            O(n) for the result.

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("b.txt", "2"); vcs.add("a.txt", "1")
            >>> c1 = vcs.commit("two files")
            >>> vcs.rm("a.txt")
            >>> _ = vcs.commit("one file")
            >>> vcs.checkout(c1)
            {'a.txt': '1', 'b.txt': '2'}
        """
        objects = self._objects
        return {path: objects[blob] for path, blob in self._commit_of(commit_id).tree.items()}

    def log(self) -> List[Tuple[str, str]]:
        """Return the history as ``(id, message)`` pairs, newest first.

        Returns:
            A new list, one pair per commit.

        Raises:
            Nothing.

        Time Complexity:
            O(number of commits).

        Space Complexity:
            O(number of commits).

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "1")
            >>> c1 = vcs.commit("one")
            >>> vcs.add("a.txt", "2")
            >>> c2 = vcs.commit("two")
            >>> vcs.log() == [(c2, "two"), (c1, "one")]
            True
        """
        out: List[Tuple[str, str]] = []
        for k in range(len(self._order) - 1, -1, -1):
            record = self._commits[self._order[k]]
            out.append((record.id, record.message))
        return out

    def diff(self, a_id: str, b_id: str) -> Dict[str, Any]:
        """Return what changed from commit ``a_id`` to commit ``b_id`` (Merkle-pruned).

        Args:
            a_id: The earlier commit (any commit may be given).
            b_id: The later commit.

        Returns:
            ``{"added": [...], "removed": [...], "modified": [...],
            "nodes_visited": int}``: paths in ``b`` only, in ``a`` only, and
            in both with different content, each sorted, and the number of
            tree nodes examined (see :func:`diff_maps`).

        Raises:
            KeyError: If either commit does not exist.

        Time Complexity:
            O(log n) expected per changed file; 2 nodes (the two roots) for
            equal non-empty trees, 0 for two empty trees.

        Space Complexity:
            O(depth) plus the output.

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "1"); vcs.add("b.txt", "2")
            >>> c1 = vcs.commit("one")
            >>> vcs.add("b.txt", "3")
            >>> c2 = vcs.commit("two")
            >>> d = vcs.diff(c1, c2)
            >>> d["added"], d["removed"], d["modified"], d["nodes_visited"] > 0
            ([], [], ['b.txt'], True)
            >>> vcs.diff(c2, c2)["nodes_visited"]
            2
        """
        return diff_maps(self._commit_of(a_id).tree, self._commit_of(b_id).tree)

    # ------------------------------------------------------------ statistics

    def stats(self) -> Dict[str, Any]:
        """Return node-sharing statistics over all commits.

        Keys:

        * ``"commits"``: number of commits.
        * ``"files"``: number of files in the newest commit (0 before any).
        * ``"objects"``: distinct contents in the object store.
        * ``"unique_nodes"``: distinct treap node objects across all commit
          trees, each shared node counted once (what is stored).
        * ``"full_copy_nodes"``: sum of the tree sizes over all commits, the
          nodes needed if every commit copied the whole table.
        * ``"saved_nodes"``: ``full_copy_nodes - unique_nodes``.
        * ``"stored_fraction"``: ``unique_nodes / full_copy_nodes`` (0.0 when
          there is nothing to store).

        Returns:
            A dict with the keys above.

        Raises:
            Nothing.

        Time Complexity:
            O(number of commits).

        Space Complexity:
            O(1).

        Examples:
            >>> vcs = MiniVCS()
            >>> for i in range(8):
            ...     vcs.add(f"f{i}.txt", str(i))
            >>> _ = vcs.commit("eight files")
            >>> vcs.add("f3.txt", "changed")
            >>> _ = vcs.commit("one change")
            >>> s = vcs.stats()
            >>> s["commits"], s["files"], s["objects"], s["full_copy_nodes"]
            (2, 8, 9, 16)
            >>> s["unique_nodes"] < s["full_copy_nodes"], s["saved_nodes"] > 0
            (True, True)
        """
        full = 0
        for cid in self._order:
            full += len(self._commits[cid].tree)
        unique = len(self._stored)
        return {
            "commits": len(self._order),
            "files": len(self._commits[self._order[-1]].tree) if self._order else 0,
            "objects": len(self._objects),
            "unique_nodes": unique,
            "full_copy_nodes": full,
            "saved_nodes": full - unique,
            "stored_fraction": unique / full if full else 0.0,
        }

    def history(self) -> List[Dict[str, Any]]:
        """Return per-commit figures for a history plot, oldest first.

        Each entry has ``"number"`` (1-based commit number), ``"id"``,
        ``"parent"`` (``""`` for the first), ``"message"``, ``"files"``
        (files in that commit), ``"files_changed"`` (paths added, removed or
        modified against the parent; for the first commit, against an empty
        tree) and ``"new_nodes"`` (tree nodes that no earlier commit had,
        the nodes this commit added to the store).

        Returns:
            A new list of dicts, one per commit.

        Raises:
            Nothing.

        Time Complexity:
            O(number of commits).

        Space Complexity:
            O(number of commits).

        Examples:
            >>> vcs = MiniVCS()
            >>> vcs.add("a.txt", "1"); vcs.add("b.txt", "2")
            >>> _ = vcs.commit("one")
            >>> vcs.add("b.txt", "3")
            >>> _ = vcs.commit("two")
            >>> [(h["number"], h["files"], h["files_changed"]) for h in vcs.history()]
            [(1, 2, 2), (2, 2, 1)]
            >>> vcs.history()[0]["new_nodes"]
            2
        """
        out: List[Dict[str, Any]] = []
        for cid in self._order:
            record = self._commits[cid]
            out.append(
                {
                    "number": record.number,
                    "id": record.id,
                    "parent": record.parent,
                    "message": record.message,
                    "files": len(record.tree),
                    "files_changed": record.files_changed,
                    "new_nodes": record.new_nodes,
                }
            )
        return out

    def __len__(self) -> int:
        """Return the number of commits.

        Returns:
            The commit count.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(MiniVCS())
            0
        """
        return len(self._order)

    def __repr__(self) -> str:
        """Return the number of commits and the head id.

        Returns:
            For example ``MiniVCS(commits=3, head='1a2b3c4d5e6f')``.

        Raises:
            Nothing.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> MiniVCS()
            MiniVCS(commits=0, head=None)
        """
        return f"MiniVCS(commits={len(self._order)}, head={self.head!r})"
