"""A polynomial rolling hash, the fingerprint Rabin-Karp slides along the text.

The hash
--------
A string ``s`` of length m is read as an m-digit number in base ``base``,
one digit per character (its code point), and reduced modulo ``modulus``::

    h(s) = sum(ord(s[i]) * base^(m-1-i) for i in range(m)) mod q

The first character is the most significant digit. :meth:`RollingHash.hash`
computes it left to right by Horner's rule, ``h = (h * base + ord(c)) mod q``,
which gives exactly that sum. The Week 11 test anchors (spurious hits for
q = 7 and q = 13) depend on this definition.

Rolling
-------
The point of the polynomial form is that the next window's hash costs O(1)
instead of O(m). Moving from ``text[i:i+m]`` to ``text[i+1:i+m+1]`` removes the
leading digit, shifts the rest up one place and appends the new digit::

    h' = ((h - ord(out) * base^(m-1)) * base + ord(in)) mod q

Python's ``%`` always returns a value in ``[0, q)``, so the subtraction never
leaves a negative hash.

What it stores before it searches
---------------------------------
For a window length m it stores one number, ``base^(m-1) mod q``, computed
once and cached per m; it depends only on the pattern length, never on the
text.

Examples:
    >>> rh = RollingHash(base=256, modulus=101)
    >>> rh.hash("ab") == (97 * 256 + 98) % 101
    True
    >>> h = rh.hash("abc")
    >>> rh.roll(h, "a", "d", 3) == rh.hash("bcd")
    True

Author:
    Robert Deibel - CSC 5300 Advanced Algorithms, Concordia University Texas.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["DEFAULT_BASE", "DEFAULT_MODULUS", "RollingHash"]

#: One digit per byte-sized character; any code point still works.
DEFAULT_BASE: int = 256

#: The Mersenne prime 2^61 - 1: collisions on real text are vanishingly rare.
DEFAULT_MODULUS: int = (1 << 61) - 1


def _require_int(value: Any, name: str) -> int:
    """Return ``value`` if it is an ``int`` (``bool`` rejected), else raise.

    Args:
        value: The value to check.
        name: The parameter name used in the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: If ``value`` is not an ``int``, or is a ``bool``.

    Time Complexity:
        O(1).

    Space Complexity:
        O(1).

    Examples:
        >>> _require_int(5, "base")
        5
        >>> _require_int(True, "base")
        Traceback (most recent call last):
            ...
        TypeError: base must be an int, got bool
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


class RollingHash:
    """The polynomial hash ``sum(ord(s[i]) * base^(m-1-i)) mod q`` with O(1) rolls.

    One instance fixes the base and the modulus. :meth:`hash` fingerprints a
    whole string, :meth:`roll` slides a window one character to the right,
    and :meth:`power` returns the cached ``base^(m-1) mod q`` that the roll
    needs to remove the leading character.

    A small modulus makes collisions (equal hashes, different strings)
    common, which is what the q = 7 test anchors use to show spurious hits.
    ``base = 1`` makes the hash the plain sum of code points, so every
    anagram of a window collides with it.

    Attributes:
        base: The radix, at least 1.
        modulus: The modulus q, at least 2.

    Examples:
        >>> rh = RollingHash()
        >>> rh.base, rh.modulus == (1 << 61) - 1
        (256, True)
        >>> RollingHash(base=1, modulus=97).hash("abc") == RollingHash(1, 97).hash("cab")
        True
    """

    __slots__ = ("base", "modulus", "_powers")

    def __init__(self, base: int = DEFAULT_BASE, modulus: int = DEFAULT_MODULUS) -> None:
        """Fix the base and modulus of the hash.

        The power cache starts empty and fills one entry per distinct window
        length passed to :meth:`power` or :meth:`roll`.

        Args:
            base: The radix of the polynomial. Must be at least 1.
            modulus: The modulus q. Must be at least 2.

        Returns:
            None.

        Raises:
            TypeError: If ``base`` or ``modulus`` is not an ``int`` (``bool``
                rejected).
            ValueError: If ``base < 1`` or ``modulus < 2``.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).

        Examples:
            >>> RollingHash(31, 1009).base
            31
            >>> RollingHash(base=0)
            Traceback (most recent call last):
                ...
            ValueError: base must be at least 1, got 0
            >>> RollingHash(modulus=1)
            Traceback (most recent call last):
                ...
            ValueError: modulus must be at least 2, got 1
        """
        base = _require_int(base, "base")
        modulus = _require_int(modulus, "modulus")
        if base < 1:
            raise ValueError(f"base must be at least 1, got {base}")
        if modulus < 2:
            raise ValueError(f"modulus must be at least 2, got {modulus}")
        self.base: int = base
        self.modulus: int = modulus
        self._powers: Dict[int, int] = {}

    def __repr__(self) -> str:
        """Show the base and modulus.

        Args:
            None.

        Returns:
            ``"RollingHash(base=..., modulus=...)"``.

        Raises:
            Nothing.

        Time Complexity:
            O(log q) to format the modulus.

        Space Complexity:
            O(log q).

        Examples:
            >>> RollingHash(256, 101)
            RollingHash(base=256, modulus=101)
        """
        return f"RollingHash(base={self.base}, modulus={self.modulus})"

    def hash(self, s: str) -> int:
        """Return ``sum(ord(s[i]) * base^(m-1-i)) mod q`` for ``s`` of length m.

        Computed left to right by Horner's rule, ``h = (h * base + ord(c)) mod
        q``, so the first character is the most significant digit. The empty
        string hashes to 0.

        Args:
            s: Any Python string, Unicode included.

        Returns:
            The hash, an ``int`` in ``[0, modulus)``.

        Raises:
            TypeError: If ``s`` is not a ``str``.

        Time Complexity:
            O(m) arithmetic operations on numbers below ``base * q``.

        Space Complexity:
            O(1).

        Examples:
            >>> rh = RollingHash(256, 7)
            >>> rh.hash("abra"), rh.hash("acad"), rh.hash("brac")
            (6, 6, 1)
            >>> RollingHash(10, 10**9).hash("\\x01\\x02\\x03")
            123
            >>> RollingHash().hash("")
            0
        """
        if not isinstance(s, str):
            raise TypeError(f"s must be a str, got {type(s).__name__}")
        base, q = self.base, self.modulus
        h = 0
        for c in s:
            h = (h * base + ord(c)) % q
        return h

    def power(self, m: int) -> int:
        """Return ``base^(m-1) mod q``, the weight of the leading character.

        This is the only value a roll needs besides the current hash. It is
        computed once per window length with three-argument :func:`pow` and
        cached on the instance.

        Args:
            m: The window length. Must be at least 1.

        Returns:
            ``pow(base, m - 1, modulus)``.

        Raises:
            TypeError: If ``m`` is not an ``int`` (``bool`` rejected).
            ValueError: If ``m < 1``.

        Time Complexity:
            O(log m) the first time for each m, O(1) after that.

        Space Complexity:
            O(1) per distinct m cached.

        Examples:
            >>> rh = RollingHash(256, 7)
            >>> [rh.power(m) for m in (1, 2, 3, 4)]
            [1, 4, 2, 1]
            >>> rh.power(0)
            Traceback (most recent call last):
                ...
            ValueError: m must be at least 1, got 0
        """
        m = _require_int(m, "m")
        if m < 1:
            raise ValueError(f"m must be at least 1, got {m}")
        cached = self._powers.get(m)
        if cached is None:
            cached = pow(self.base, m - 1, self.modulus)
            self._powers[m] = cached
        return cached

    def roll(self, h: int, out_char: str, in_char: str, m: int) -> int:
        """Slide a length-m window one character right in O(1).

        Given ``h = hash(text[i:i+m])`` with ``out_char = text[i]`` and
        ``in_char = text[i+m]``, return ``hash(text[i+1:i+m+1])`` as
        ``((h - ord(out_char) * base^(m-1)) * base + ord(in_char)) mod q``.

        Args:
            h: The hash of the current window.
            out_char: The character leaving on the left.
            in_char: The character entering on the right.
            m: The window length. Must be at least 1.

        Returns:
            The hash of the next window, in ``[0, modulus)``.

        Raises:
            TypeError: If ``m`` is not an ``int``, or a character argument is
                not a single-character ``str`` (from :func:`ord`).
            ValueError: If ``m < 1``.

        Time Complexity:
            O(1) once ``power(m)`` is cached.

        Space Complexity:
            O(1).

        Examples:
            >>> rh = RollingHash(256, 101)
            >>> text, m = "abracadabra", 4
            >>> h = rh.hash(text[:m])
            >>> ok = []
            >>> for i in range(len(text) - m):
            ...     h = rh.roll(h, text[i], text[i + m], m)
            ...     ok.append(h == rh.hash(text[i + 1:i + 1 + m]))
            >>> all(ok), len(ok)
            (True, 7)
        """
        return ((h - ord(out_char) * self.power(m)) * self.base + ord(in_char)) % self.modulus
