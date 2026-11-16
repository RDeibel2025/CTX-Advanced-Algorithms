"""A fixed-length bit vector packed into numpy ``uint64`` words.

This is the payload that :mod:`src.advds.succinct.rank_select` indexes and
that every node of the wavelet tree stores. It holds n bits in
``ceil(n / 64)`` words and nothing else, so its size is n bits rounded up to
a whole word.

Layout
------
Bit i lives in word ``i // 64`` at bit position ``i % 64``, least significant
bit first: bits 0 .. 63 are word 0 read from its low end, bits 64 .. 127 are
word 1, and so on. The unused high bits of the last word (the padding) are
always 0. Read as bytes in little-endian order, the same storage puts bit i
at bit ``i % 8`` of byte ``i // 8``, which is the view the rank/select byte
tables use.

Conventions
-----------
* Indices are 0-based and must satisfy ``0 <= i < n``. Anything else raises
  ``IndexError``; negative indices are not wrapped around, because a rank or
  select caller that computes ``-1`` has a bug that should surface.
* A bit is the int 0 or 1. Inputs may be ``bool``, ``numpy.bool_`` or the
  integers 0 and 1.
* The length is fixed when the vector is made. :meth:`BitVector.set` may
  change a bit only until a ``RankSelect`` is built on the vector: the
  ``RankSelect`` constructor calls :meth:`BitVector.freeze`, after which
  ``set`` raises ``RuntimeError``, because a changed bit would silently
  invalidate the rank directory built over it.
* :meth:`BitVector.size_in_bits` is the payload only, 64 bits per stored
  word with the padding included. Directory bits are reported by
  ``RankSelect``.

What numpy does here (spec 1.1)
-------------------------------
numpy is used for storage (the ``uint64`` words) and for packing input bits:
``np.packbits(..., bitorder="little")`` packs eight bits per byte and a
little-endian ``uint64`` view groups eight bytes per word. No counting
routine is used in this module; counting 1 bits is the job of the hand-built
``POPCOUNT8`` and ``SELECT8`` tables in ``rank_select.py``.

Examples:
    >>> bv = BitVector.from_string("1011001110001011")
    >>> len(bv), bv[0], bv[1], bv[2]
    (16, 1, 0, 1)
    >>> int(bv.words[0]) == 0b1101000111001101
    True
    >>> bv.size_in_bits()
    64

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

import numbers
from typing import Any, Iterable, Iterator

import numpy as np

__all__ = ["BitVector", "WORD_BITS"]

#: Bits per storage word.
WORD_BITS: int = 64

#: All 64 bits of a word set, for clearing a bit with Python ints.
_WORD_MASK: int = (1 << WORD_BITS) - 1


def _as_int(value: Any, name: str) -> int:
    """Return ``value`` as a Python ``int`` if it is an integer, else raise.

    ``bool`` is rejected (``True`` as an index is an accident, not a choice);
    numpy integer scalars are accepted.

    Args:
        value: The value to check.
        name: Its parameter name, for the error message.

    Returns:
        ``int(value)``.

    Raises:
        TypeError: If ``value`` is not an integer, or is a ``bool``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _as_int(7, "i")
        7
        >>> _as_int(np.int64(3), "i")
        3
        >>> _as_int(True, "i")
        Traceback (most recent call last):
            ...
        TypeError: i must be an int, got bool
        >>> _as_int(1.5, "i")
        Traceback (most recent call last):
            ...
        TypeError: i must be an int, got float
    """
    if type(value) is int:
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_bit(value: Any) -> int:
    """Return a bit value as the int 0 or 1.

    Args:
        value: ``bool``, ``numpy.bool_``, or an integer equal to 0 or 1.

    Returns:
        0 or 1.

    Raises:
        TypeError: If ``value`` is neither a boolean nor an integer.
        ValueError: If ``value`` is an integer other than 0 or 1.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _as_bit(True), _as_bit(0), _as_bit(np.bool_(True))
        (1, 0, 1)
        >>> _as_bit(2)
        Traceback (most recent call last):
            ...
        ValueError: a bit must be 0 or 1, got 2
        >>> _as_bit("1")
        Traceback (most recent call last):
            ...
        TypeError: a bit must be a bool or the int 0 or 1, got str
    """
    if isinstance(value, (bool, np.bool_)):
        return 1 if value else 0
    if not isinstance(value, numbers.Integral):
        raise TypeError(
            f"a bit must be a bool or the int 0 or 1, got {type(value).__name__}"
        )
    if value != 0 and value != 1:
        raise ValueError(f"a bit must be 0 or 1, got {value}")
    return int(value)


def _pack_bool_array(bits: np.ndarray) -> np.ndarray:
    """Pack a 1-D boolean array into little-endian-first ``uint64`` words.

    ``np.packbits(bits, bitorder="little")`` puts bit i at bit ``i % 8`` of
    byte ``i // 8``. The bytes are zero-padded to a multiple of 8 and viewed
    as little-endian ``uint64``, which puts bit i at bit ``i % 64`` of word
    ``i // 64``; the result is converted to the native byte order. The
    padding bits of the last word are 0.

    Args:
        bits: A 1-D ``numpy`` array of dtype ``bool``.

    Returns:
        A new, writable ``uint64`` array of ``ceil(len(bits) / 64)`` words.

    Time Complexity:
        O(n) - a single vectorised packing pass.

    Space Complexity:
        O(n / 8) bytes for the packed copy.

    Examples:
        >>> _pack_bool_array(np.array([1, 0, 1, 1], dtype=bool)).tolist()
        [13]
        >>> _pack_bool_array(np.zeros(65, dtype=bool)).shape
        (2,)
        >>> _pack_bool_array(np.zeros(0, dtype=bool)).shape
        (0,)
    """
    n = int(bits.shape[0])
    n_words = (n + WORD_BITS - 1) // WORD_BITS
    packed = np.packbits(bits, bitorder="little")
    buffer = np.zeros(n_words * 8, dtype=np.uint8)
    buffer[: packed.shape[0]] = packed
    return buffer.view("<u8").astype(np.uint64)


class BitVector:
    """A fixed-length sequence of bits packed into numpy ``uint64`` words.

    Bit i is bit ``i % 64`` of word ``i // 64`` (least significant first);
    the padding bits of the last word are 0. The vector can be changed with
    :meth:`set` until :meth:`freeze` is called, which ``RankSelect`` does
    when it builds its directories.

    Args:
        n: The number of bits. Every bit starts at 0.

    Raises:
        TypeError: If ``n`` is not an integer (``bool`` rejected).
        ValueError: If ``n`` is negative.

    Time Complexity:
        O(n / 64) to allocate the zeroed words.

    Space Complexity:
        64 * ceil(n / 64) bits.

    Examples:
        >>> bv = BitVector(5)
        >>> bv.to_string()
        '00000'
        >>> bv.set(3, 1)
        >>> bv.to_string()
        '00010'
        >>> BitVector(-1)
        Traceback (most recent call last):
            ...
        ValueError: n must be non-negative, got -1
    """

    __slots__ = ("_n", "_words", "_frozen")

    def __init__(self, n: int = 0) -> None:
        n = _as_int(n, "n")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._n: int = n
        self._words: np.ndarray = np.zeros((n + WORD_BITS - 1) // WORD_BITS, dtype=np.uint64)
        self._frozen: bool = False

    # ------------------------------------------------------------ builders

    @classmethod
    def _from_words(cls, words: np.ndarray, n: int) -> "BitVector":
        """Wrap an already packed word array of the right length.

        Args:
            words: A ``uint64`` array of ``ceil(n / 64)`` words whose padding
                bits are 0. It is taken over, not copied.
            n: The number of bits.

        Returns:
            A new, unfrozen :class:`BitVector`.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1) beyond ``words``.

        Examples:
            >>> BitVector._from_words(np.array([5], dtype=np.uint64), 3).to_string()
            '101'
        """
        bv = cls.__new__(cls)
        bv._n = n
        bv._words = words
        bv._frozen = False
        return bv

    @classmethod
    def from_bits(cls, bits: Iterable[Any]) -> "BitVector":
        """Build a bit vector from an iterable or a 1-D numpy array of bits.

        A numpy ``bool`` array is packed directly with ``np.packbits``. A
        numpy integer array must hold only 0 and 1. Any other iterable (a
        list, a generator, another :class:`BitVector`) is read once into a
        numpy array and packed the same way, so large inputs are packed by
        numpy rather than bit by bit in Python. When numpy cannot give the
        values one integer dtype (an ``object`` array, a huge Python int, or
        numpy unsigned ints mixed with negative ints), each value is checked
        as a bit on its own, so the errors below hold for every input.

        Args:
            bits: Booleans or the integers 0 and 1. A ``str`` is rejected;
                use :meth:`from_string` for text such as ``"1011"``.

        Returns:
            A new, unfrozen :class:`BitVector` of length ``len(bits)``.

        Raises:
            TypeError: If ``bits`` is a string, or holds values that are not
                booleans or integers.
            ValueError: If an integer other than 0 or 1 appears, or the
                input is not one-dimensional.

        Time Complexity:
            O(n), vectorised for numpy inputs.

        Space Complexity:
            O(n / 8) bytes for the words, plus O(n) for a temporary array
            when the input is not already a numpy ``bool`` array.

        Examples:
            >>> BitVector.from_bits([1, 0, 1, 1]).to_string()
            '1011'
            >>> BitVector.from_bits(np.array([True, False, True])).to_string()
            '101'
            >>> BitVector.from_bits(np.array([0, 1, 1], dtype=np.int64)).to_string()
            '011'
            >>> len(BitVector.from_bits([]))
            0
            >>> BitVector.from_bits([0, 2])
            Traceback (most recent call last):
                ...
            ValueError: bits must be 0 or 1
            >>> BitVector.from_bits([0, 2**70])
            Traceback (most recent call last):
                ...
            ValueError: a bit must be 0 or 1, got 1180591620717411303424
            >>> BitVector.from_bits(np.array([1, 0], dtype=object)).to_string()
            '10'
            >>> BitVector.from_bits("101")
            Traceback (most recent call last):
                ...
            TypeError: from_bits does not take a str; use BitVector.from_string
        """
        if isinstance(bits, (str, bytes)):
            raise TypeError("from_bits does not take a str; use BitVector.from_string")
        items = None
        if isinstance(bits, np.ndarray):
            arr = bits
        else:
            items = list(bits)
            if not items:
                return cls(0)
            arr = np.asarray(items)
        if arr.ndim != 1:
            raise ValueError(f"bits must be one-dimensional, got shape {arr.shape}")
        if arr.shape[0] == 0:
            return cls(0)
        if arr.dtype == np.bool_:
            flags = arr
        elif np.issubdtype(arr.dtype, np.integer):
            if not bool(np.all((arr == 0) | (arr == 1))):
                raise ValueError("bits must be 0 or 1")
            flags = arr != 0
        elif arr.dtype == np.object_ or items is not None:
            # numpy found no single integer dtype for the values: check each
            # original value as a bit (TypeError / ValueError as documented).
            source = arr.tolist() if items is None else items
            flags = np.array([_as_bit(value) for value in source], dtype=bool)
        else:
            raise TypeError(
                f"bits must be booleans or the integers 0 and 1, got dtype {arr.dtype}"
            )
        return cls._from_words(_pack_bool_array(np.ascontiguousarray(flags)), int(arr.shape[0]))

    @classmethod
    def from_string(cls, text: str) -> "BitVector":
        """Build a bit vector from a string of ``'0'`` and ``'1'`` characters.

        Character i of ``text`` becomes bit i, so ``"1011"`` has bit 0 = 1
        and bit 1 = 0 (the string reads left to right in index order, not as
        a binary number).

        Args:
            text: A string made only of ``'0'`` and ``'1'``; may be empty.

        Returns:
            A new, unfrozen :class:`BitVector` of length ``len(text)``.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            ValueError: If ``text`` contains any other character.

        Time Complexity:
            O(n).

        Space Complexity:
            O(n) for a temporary byte array, then n bits of words.

        Examples:
            >>> bv = BitVector.from_string("0110")
            >>> [bv[i] for i in range(4)]
            [0, 1, 1, 0]
            >>> len(BitVector.from_string(""))
            0
            >>> BitVector.from_string("10a1")
            Traceback (most recent call last):
                ...
            ValueError: text must contain only '0' and '1'
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not set(text) <= {"0", "1"}:
            raise ValueError("text must contain only '0' and '1'")
        if not text:
            return cls(0)
        codes = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        return cls._from_words(_pack_bool_array(codes == ord("1")), len(text))

    # ------------------------------------------------------------- access

    def __len__(self) -> int:
        """Return the number of bits n.

        Returns:
            n.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> len(BitVector.from_string("101"))
            3
        """
        return self._n

    def _check_index(self, i: Any) -> int:
        """Validate a bit index and return it as an int.

        Args:
            i: The index to check.

        Returns:
            ``i`` as an int with ``0 <= i < n``.

        Raises:
            TypeError: If ``i`` is not an integer.
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> BitVector(3)._check_index(2)
            2
            >>> BitVector(3)._check_index(3)
            Traceback (most recent call last):
                ...
            IndexError: bit index 3 out of range for length 3
        """
        i = _as_int(i, "index")
        if i < 0 or i >= self._n:
            raise IndexError(f"bit index {i} out of range for length {self._n}")
        return i

    def __getitem__(self, i: int) -> int:
        """Return bit i as the int 0 or 1.

        Args:
            i: A 0-based index with ``0 <= i < n``.

        Returns:
            Bit ``i % 64`` of word ``i // 64``.

        Raises:
            TypeError: If ``i`` is not an integer (slices are not supported).
            IndexError: If ``i`` is outside ``[0, n)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> bv = BitVector.from_string("0010")
            >>> bv[2], bv[3]
            (1, 0)
            >>> bv[-1]
            Traceback (most recent call last):
                ...
            IndexError: bit index -1 out of range for length 4
        """
        i = self._check_index(i)
        return (int(self._words[i >> 6]) >> (i & 63)) & 1

    def __iter__(self) -> Iterator[int]:
        """Yield the bits in index order as ints 0 and 1.

        Yields:
            Bit 0, bit 1, ..., bit n - 1.

        Time Complexity:
            O(n).

        Space Complexity:
            O(n / 64) for the list of Python word values.

        Examples:
            >>> list(BitVector.from_string("1101"))
            [1, 1, 0, 1]
        """
        n = self._n
        for w, word in enumerate(self._words.tolist()):
            base = w << 6
            stop = n - base if n - base < WORD_BITS else WORD_BITS
            for r in range(stop):
                yield (word >> r) & 1

    def set(self, i: int, b: Any) -> None:
        """Set bit i to b. Allowed only before a ``RankSelect`` is built.

        Args:
            i: A 0-based index with ``0 <= i < n``.
            b: The new bit: ``bool`` or the int 0 or 1.

        Raises:
            RuntimeError: If the vector is frozen (a ``RankSelect`` was built
                on it).
            TypeError: If ``i`` is not an integer or ``b`` is not a bit.
            IndexError: If ``i`` is outside ``[0, n)``.
            ValueError: If ``b`` is an integer other than 0 or 1.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> bv = BitVector(70)
            >>> bv.set(65, True)
            >>> bv[65], int(bv.words[1])
            (1, 2)
            >>> bv.set(65, 0)
            >>> bv[65]
            0
            >>> bv.freeze()
            >>> bv.set(0, 1)
            Traceback (most recent call last):
                ...
            RuntimeError: BitVector is frozen: a RankSelect was built on it
        """
        if self._frozen:
            raise RuntimeError("BitVector is frozen: a RankSelect was built on it")
        i = self._check_index(i)
        bit = _as_bit(b)
        w, r = i >> 6, i & 63
        word = int(self._words[w])
        if bit:
            word |= 1 << r
        else:
            word &= _WORD_MASK ^ (1 << r)
        self._words[w] = word

    # ------------------------------------------------------- frozen state

    def freeze(self) -> None:
        """Forbid further :meth:`set` calls. ``RankSelect`` calls this.

        Freezing is permanent: the rank directory is a summary of the bits,
        so once one exists the bits must not change.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> bv = BitVector(2)
            >>> bv.frozen
            False
            >>> bv.freeze()
            >>> bv.frozen
            True
        """
        self._frozen = True

    @property
    def frozen(self) -> bool:
        """Whether :meth:`set` is forbidden because a ``RankSelect`` exists.

        Returns:
            ``True`` after :meth:`freeze`, else ``False``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> BitVector(1).frozen
            False
        """
        return self._frozen

    # ------------------------------------------------------------ storage

    @property
    def words(self) -> np.ndarray:
        """A read-only view of the ``uint64`` storage words.

        Word w holds bits ``64 w .. 64 w + 63``, least significant first; the
        padding bits of the last word are 0. The view shares memory with the
        vector (no copy) and cannot be written through.

        Returns:
            A non-writable ``uint64`` array of ``ceil(n / 64)`` words.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1) - a view.

        Examples:
            >>> w = BitVector.from_string("11").words
            >>> w.dtype.name, w.tolist(), w.flags.writeable
            ('uint64', [3], False)
        """
        view = self._words.view()
        view.flags.writeable = False
        return view

    def size_in_bits(self) -> int:
        """Return the payload size: 64 bits for every stored word.

        The padding of the last word is included, because it is stored.
        Directory bits are not; ``RankSelect`` reports those.

        Returns:
            ``64 * ceil(n / 64)``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> BitVector(0).size_in_bits(), BitVector(1).size_in_bits(), BitVector(65).size_in_bits()
            (0, 64, 128)
        """
        return WORD_BITS * int(self._words.shape[0])

    def to_string(self) -> str:
        """Return the bits as a ``'0'``/``'1'`` string in index order.

        The inverse of :meth:`from_string`.

        Returns:
            A string of length n whose character i is bit i.

        Time Complexity:
            O(n).

        Space Complexity:
            O(n).

        Examples:
            >>> BitVector.from_string("100101").to_string()
            '100101'
        """
        return "".join("1" if bit else "0" for bit in self)

    def __repr__(self) -> str:
        """Return a short description; the bits themselves when n <= 64.

        Returns:
            ``BitVector.from_string('...')`` for short vectors, otherwise
            ``<BitVector n=..., words=...>``.

        Time Complexity:
            O(min(n, 64)).

        Space Complexity:
            O(min(n, 64)).

        Examples:
            >>> BitVector.from_string("101")
            BitVector.from_string('101')
            >>> BitVector(100)
            <BitVector n=100, words=2>
        """
        if self._n <= WORD_BITS:
            return f"BitVector.from_string({self.to_string()!r})"
        return f"<BitVector n={self._n}, words={int(self._words.shape[0])}>"
