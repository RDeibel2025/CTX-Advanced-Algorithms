"""Stream a text file from disk in fixed-size byte chunks.

The search methods take a Python ``str``, so a user's file has to be read and
decoded first. Reading a large file with one ``read()`` call is fine for the
sizes in this project, but streaming keeps peak memory at one chunk plus the
result and lets a caller stop early with ``max_bytes``. The file is opened in
binary mode and read ``size`` bytes at a time; an incremental decoder
(:func:`codecs.getincrementaldecoder`) turns the bytes into text and holds back
the first bytes of a multi-byte character that a chunk boundary splits, so a
chunk never contains half a UTF-8 character. Binary mode also means line
endings are kept exactly as stored (``"\\r\\n"`` stays ``"\\r\\n"``), so
character positions found by a search correspond to the file as written.

Nothing here searches, so nothing is stored before a search: the loader only
produces the text that the indexes are later built over. The benchmark does not
depend on downloaded files; it generates its texts from seeds.

============ ================== ==================
Function     Time               Space
============ ================== ==================
iter_chunks  O(file size)       O(size) per chunk
load_text    O(min(file, max))  O(min(file, max))
============ ================== ==================

Examples:
    >>> import os, tempfile
    >>> fd, path = tempfile.mkstemp(suffix=".txt")
    >>> with os.fdopen(fd, "w", encoding="utf-8") as handle:
    ...     _ = handle.write("banana")
    >>> load_text(path)
    'banana'
    >>> list(iter_chunks(path, 4))
    ['bana', 'na']
    >>> os.remove(path)

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import codecs
import os
from typing import Any, Iterator, Optional, Union

__all__ = ["DEFAULT_CHUNK_BYTES", "iter_chunks", "load_text"]

#: Bytes read per chunk by :func:`load_text` (1 MiB).
DEFAULT_CHUNK_BYTES: int = 1 << 20

#: Path-like values accepted by the loaders.
PathLike = Union[str, "os.PathLike[str]"]


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` if it is an ``int`` of at least ``minimum``.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.
        minimum: The smallest value allowed.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int`` (``bool`` rejected).
        ValueError: If ``value`` is below ``minimum``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(4, "size", 1)
        4
        >>> _require_int(0, "size", 1)
        Traceback (most recent call last):
            ...
        ValueError: size must be at least 1, got 0
        >>> _require_int("4", "size", 1)
        Traceback (most recent call last):
            ...
        TypeError: size must be an int, got str
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _decoded_chunks(
    path: PathLike, size: int, encoding: str, max_bytes: Optional[int]
) -> Iterator[str]:
    """Yield the decoded text of ``path``, read ``size`` bytes at a time.

    Reads at most ``max_bytes`` bytes when it is given. At the true end of the
    file the decoder is flushed with ``final=True``, so a file that really
    ends inside a character raises :class:`UnicodeDecodeError`. When the
    ``max_bytes`` budget runs out first, the bytes of a character split by
    that cut are dropped instead, because the rest of the character is
    simply not read.

    Args:
        path: The file to read.
        size: Bytes per read, at least 1.
        encoding: Any codec name known to :mod:`codecs`.
        max_bytes: Byte budget, or ``None`` to read the whole file.

    Returns:
        An iterator of non-empty ``str`` chunks.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        LookupError: If ``encoding`` is unknown.
        UnicodeDecodeError: If the bytes are not valid in ``encoding``.

    Time Complexity:
        O(B) for B bytes read.

    Space Complexity:
        O(size) per chunk.

    Examples:
        >>> import os, tempfile
        >>> fd, path = tempfile.mkstemp()
        >>> with os.fdopen(fd, "wb") as handle:
        ...     _ = handle.write("xé".encode("utf-8"))
        >>> list(_decoded_chunks(path, 2, "utf-8", None))
        ['x', 'é']
        >>> list(_decoded_chunks(path, 2, "utf-8", 2))
        ['x']
        >>> os.remove(path)
    """
    decoder = codecs.getincrementaldecoder(encoding)()
    remaining = max_bytes
    with open(path, "rb") as handle:
        while remaining is None or remaining > 0:
            raw = handle.read(size if remaining is None else min(size, remaining))
            if not raw:
                break
            if remaining is not None:
                remaining -= len(raw)
            piece = decoder.decode(raw)
            if piece:
                yield piece
        else:
            return  # budget used up: drop any character split by the cut
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def iter_chunks(path: PathLike, size: int, encoding: str = "utf-8") -> Iterator[str]:
    """Yield the text of a file in chunks of at most ``size`` bytes each.

    Each chunk is the decoded text of one ``size``-byte read, so it holds at
    most ``size`` characters and fewer when characters take several bytes.
    A character split between two reads is completed and returned with the
    next chunk, never cut in half. Empty chunks are skipped, so joining the
    chunks gives exactly the decoded file.

    Args:
        path: The file to read (``str`` or :class:`os.PathLike`).
        size: Bytes per read, at least 1.
        encoding: Any codec name known to :mod:`codecs`.

    Returns:
        A generator of non-empty ``str`` chunks. The file is opened on the
        first ``next()`` and closed when the generator finishes or is closed.

    Raises:
        TypeError: If ``size`` is not an ``int``.
        ValueError: If ``size < 1``.
        FileNotFoundError: If ``path`` does not exist (on the first
            ``next()``).
        LookupError: If ``encoding`` is unknown.
        UnicodeDecodeError: If the bytes are not valid in ``encoding``.

    Time Complexity:
        O(file size) over the whole iteration.

    Space Complexity:
        O(size) per chunk.

    Examples:
        >>> import os, tempfile
        >>> fd, path = tempfile.mkstemp(suffix=".txt")
        >>> with os.fdopen(fd, "w", encoding="utf-8") as handle:
        ...     _ = handle.write("naïve café")
        >>> list(iter_chunks(path, 3))
        ['na', 'ïve', ' ca', 'fé']
        >>> "".join(iter_chunks(path, 5)) == "naïve café"
        True
        >>> os.remove(path)
        >>> iter_chunks(path, 0)
        Traceback (most recent call last):
            ...
        ValueError: size must be at least 1, got 0
    """
    _require_int(size, "size", 1)
    codecs.lookup(encoding)
    return _decoded_chunks(path, size, encoding, None)


def load_text(
    path: PathLike, max_bytes: Optional[int] = None, encoding: str = "utf-8"
) -> str:
    """Read a text file into one ``str``, streaming it in 1 MiB chunks.

    The file is read in :data:`DEFAULT_CHUNK_BYTES` pieces and decoded
    incrementally (see the module docstring), then the pieces are joined
    once. With ``max_bytes`` only the first ``max_bytes`` bytes are read,
    which is how a caller takes a prefix of a large file without loading
    all of it; a character that the cut splits is dropped rather than
    raising.

    Args:
        path: The file to read (``str`` or :class:`os.PathLike`).
        max_bytes: Read at most this many bytes, or ``None`` for the whole
            file. Must be non-negative.
        encoding: Any codec name known to :mod:`codecs`.

    Returns:
        The decoded text, with line endings exactly as stored.

    Raises:
        TypeError: If ``max_bytes`` is neither ``None`` nor an ``int``.
        ValueError: If ``max_bytes`` is negative.
        FileNotFoundError: If ``path`` does not exist.
        LookupError: If ``encoding`` is unknown.
        UnicodeDecodeError: If the bytes are not valid in ``encoding``.

    Time Complexity:
        O(B) for B = min(file size, max_bytes) bytes read.

    Space Complexity:
        O(B) - the result plus one chunk.

    Examples:
        >>> import os, tempfile
        >>> fd, path = tempfile.mkstemp(suffix=".txt")
        >>> with os.fdopen(fd, "w", encoding="utf-8") as handle:
        ...     _ = handle.write("naïve café")
        >>> load_text(path)
        'naïve café'
        >>> load_text(path, max_bytes=3)
        'na'
        >>> load_text(path, max_bytes=4)
        'naï'
        >>> load_text(path, max_bytes=0)
        ''
        >>> os.remove(path)
        >>> load_text(path, max_bytes=-1)
        Traceback (most recent call last):
            ...
        ValueError: max_bytes must be at least 0, got -1
    """
    if max_bytes is not None:
        _require_int(max_bytes, "max_bytes", 0)
    codecs.lookup(encoding)
    return "".join(_decoded_chunks(path, DEFAULT_CHUNK_BYTES, encoding, max_bytes))
