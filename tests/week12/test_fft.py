"""Week 12 tests: the hand-written DFT and FFTs on the spec 9.1 anchors.

The small anchors are worked out by hand, never from the code under test; the
random cases use ``np.fft`` (allowed in tests only, spec 1.1) as the
independent cross-check.

Hand values:

- ``fft([1, 2, 3, 4])``: with w = exp(-2 pi i / 4) = -i, X[0] = 1 + 2 + 3 + 4
  = 10, X[1] = 1 - 2i - 3 + 4i = -2 + 2i, X[2] = 1 - 2 + 3 - 4 = -2,
  X[3] = 1 + 2i - 3 - 4i = -2 - 2i.
- The impulse ``[1, 0, ..., 0]`` has only the n = 0 term, exp(0) = 1, in
  every bin.
- ``cos(2 pi 3 n / 16) = (exp(2 pi i 3n/16) + exp(-2 pi i 3n/16)) / 2``: each
  exponential sums to N = 16 in exactly one bin (3 and 16 - 3 = 13), so the
  magnitude is 8 there and 0 elsewhere.
- ``circular_convolve([1, 2, 3], [0, 1, 0.5])``: c[k] = sum_j a[j] b[(k-j) mod 3]
  gives c[0] = 1*0 + 2*0.5 + 3*1 = 4, c[1] = 1*1 + 2*0 + 3*0.5 = 2.5,
  c[2] = 1*0.5 + 2*1 + 3*0 = 2.5.
- ``linear_convolve([1, 2, 3], [0, 1, 0.5])``: c[0] = 0, c[1] = 1, c[2] = 0.5 + 2
  = 2.5, c[3] = 1 + 3 = 4, c[4] = 1.5.

Relative error is ``max|ours - ref| / max|ref|`` throughout. No timings are
asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.fft_analyzer.core.fft import (
    circular_convolve,
    dft,
    fft,
    fft2,
    fft_iterative,
    fft_iterative_stages,
    fft_recursive,
    ifft,
    ifft2,
    linear_convolve,
    rfft_freqs,
)
from src.utils.random_utils import derive_seed

MASTER_SEED = 42

FFTS = {
    "iterative": fft_iterative,
    "recursive": fft_recursive,
}


def _rng(*labels) -> np.random.Generator:
    """A numpy generator seeded from the master seed and the given labels."""
    return np.random.default_rng(derive_seed(MASTER_SEED, "test_fft", *labels))


def _random_complex(n: int, *labels) -> np.ndarray:
    rng = _rng(*labels)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def _rel_error(ours, ref) -> float:
    ours = np.asarray(ours)
    ref = np.asarray(ref)
    return float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))


# ------------------------------------------------------------ hand anchors


@pytest.mark.parametrize(
    "transform",
    [fft_iterative, fft_recursive, dft, fft, lambda x: fft(x, method="recursive")],
    ids=["iterative", "recursive", "dft", "fft", "fft-recursive"],
)
def test_fft_of_1234(transform):
    X = transform([1, 2, 3, 4])
    assert X.dtype == np.complex128
    np.testing.assert_allclose(X, [10, -2 + 2j, -2, -2 - 2j], atol=1e-12)


@pytest.mark.parametrize("method", list(FFTS))
def test_impulse_gives_all_ones(method):
    x = [1, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_allclose(FFTS[method](x), np.ones(8), atol=1e-12)
    np.testing.assert_allclose(dft(x), np.ones(8), atol=1e-12)


@pytest.mark.parametrize("method", list(FFTS))
def test_cosine_lands_in_bins_3_and_13(method):
    n = np.arange(16)
    x = np.cos(2 * np.pi * 3 * n / 16)
    mag = np.abs(FFTS[method](x))
    assert mag[3] == pytest.approx(8.0, abs=1e-9)
    assert mag[13] == pytest.approx(8.0, abs=1e-9)
    others = np.delete(mag, [3, 13])
    assert np.all(others < 1e-9)


# ------------------------------------------------- random cross-checks


@pytest.mark.parametrize("method", list(FFTS))
@pytest.mark.parametrize("exponent", range(1, 13))
def test_matches_numpy_on_50_random_complex_vectors(method, exponent):
    n = 2**exponent
    transform = FFTS[method]
    for trial in range(50):
        x = _random_complex(n, "random", exponent, trial)
        assert _rel_error(transform(x), np.fft.fft(x)) < 1e-10


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 12, 31, 100, 257])
def test_dft_matches_numpy_for_any_length(n):
    x = _random_complex(n, "dft", n)
    assert _rel_error(dft(x), np.fft.fft(x)) < 1e-10


@pytest.mark.parametrize("cutoff", [1, 2, 8, 32, 64, 4096])
def test_recursive_cutoff_does_not_change_result(cutoff):
    x = _random_complex(256, "cutoff")
    assert _rel_error(fft_recursive(x, cutoff=cutoff), np.fft.fft(x)) < 1e-10


@pytest.mark.parametrize("method", list(FFTS))
def test_ifft_matches_numpy(method):
    X = _random_complex(512, "ifft")
    assert _rel_error(ifft(X, method=method), np.fft.ifft(X)) < 1e-10


@pytest.fixture(scope="module")
def big_case():
    x = _random_complex(2**20, "big")
    return x, np.fft.fft(x)


@pytest.mark.parametrize("method", list(FFTS))
def test_both_ffts_at_2_to_the_20(big_case, method):
    x, ref = big_case
    X = fft(x, method=method)
    assert X.shape == (2**20,)
    assert _rel_error(X, ref) < 1e-9
    back = ifft(X, method=method)
    assert np.max(np.abs(back - x)) < 1e-12


# ------------------------------------------------------------ convolution


def test_circular_convolve_anchor():
    c = circular_convolve([1, 2, 3], [0, 1, 0.5])
    assert c.dtype == np.float64
    np.testing.assert_allclose(c, [4, 2.5, 2.5], atol=1e-12)


@pytest.mark.parametrize("n", [1, 4, 6, 16, 33, 64])
def test_circular_convolve_matches_definition_and_numpy(n):
    rng = _rng("circular", n)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    direct = np.array(
        [sum(a[j] * b[(k - j) % n] for j in range(n)) for k in range(n)]
    )
    via_numpy = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
    c = circular_convolve(a, b)
    assert c.dtype == np.float64
    np.testing.assert_allclose(c, direct, atol=1e-9)
    np.testing.assert_allclose(c, via_numpy, atol=1e-9)


def test_circular_convolve_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        circular_convolve([1, 2, 3], [1, 2])


def test_linear_convolve_anchor():
    c = linear_convolve([1, 2, 3], [0, 1, 0.5])
    assert c.dtype == np.float64
    np.testing.assert_allclose(c, [0, 1, 2.5, 4, 1.5], atol=1e-12)


def test_linear_convolve_matches_np_convolve_on_30_random_pairs():
    rng = _rng("linear", "lengths")
    for trial in range(30):
        m, k = (int(v) for v in rng.integers(1, 200, size=2))
        pair_rng = _rng("linear", trial)
        a = pair_rng.standard_normal(m)
        b = pair_rng.standard_normal(k)
        full = linear_convolve(a, b)
        assert full.shape == (m + k - 1,)
        np.testing.assert_allclose(full, np.convolve(a, b), atol=1e-9)
        same = linear_convolve(a, b, mode="same")
        np.testing.assert_allclose(same, np.convolve(a, b, mode="same"), atol=1e-9)


@pytest.mark.parametrize("m, k", [(3, 2), (7, 4), (8, 5), (6, 6), (1, 1), (2, 5)])
def test_linear_convolve_same_mode_edges(m, k):
    rng = _rng("same", m, k)
    a = rng.standard_normal(m)
    b = rng.standard_normal(k)
    same = linear_convolve(a, b, mode="same")
    np.testing.assert_allclose(same, np.convolve(a, b, mode="same"), atol=1e-9)


def test_linear_convolve_rejects_unknown_mode():
    with pytest.raises(ValueError):
        linear_convolve([1, 2], [1], mode="valid")


# -------------------------------------------------------------------- 2-D


def test_fft2_matches_numpy_on_64x64():
    img = _rng("fft2").standard_normal((64, 64))
    assert _rel_error(fft2(img), np.fft.fft2(img)) < 1e-10


@pytest.mark.parametrize("shape", [(64, 64), (32, 128), (48, 60)])
def test_ifft2_round_trip(shape):
    img = _rng("ifft2", *shape).standard_normal(shape) * 255
    back = ifft2(fft2(img))
    np.testing.assert_allclose(back.real, img, atol=1e-9)
    assert np.max(np.abs(back.imag)) < 1e-9
    assert _rel_error(ifft2(np.fft.fft2(img)), np.fft.ifft2(np.fft.fft2(img))) < 1e-10


@pytest.mark.parametrize("shape", [(48, 60), (5, 8), (16, 7), (1, 3)])
def test_fft2_non_power_of_two_shape_matches_numpy(shape):
    img = _rng("fft2-npot", *shape).standard_normal(shape)
    assert _rel_error(fft2(img), np.fft.fft2(img)) < 1e-10


def test_fft2_rejects_non_2d():
    with pytest.raises(ValueError):
        fft2(np.ones(8))


# ---------------------------------------------------------------- lengths


@pytest.mark.parametrize("n", [3, 6, 12, 100, 1000])
def test_non_power_of_two_length_raises(n):
    x = np.ones(n)
    with pytest.raises(ValueError):
        fft(x)
    with pytest.raises(ValueError):
        fft(x, method="recursive")
    with pytest.raises(ValueError):
        fft_iterative(x)
    with pytest.raises(ValueError):
        fft_recursive(x)
    with pytest.raises(ValueError):
        ifft(x)
    with pytest.raises(ValueError):
        fft_iterative_stages(x)


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        fft([])
    with pytest.raises(ValueError):
        dft([])
    with pytest.raises(ValueError):
        fft([1, 2], method="bluestein")
    with pytest.raises(ValueError):
        fft([1, 2], n=0)
    with pytest.raises(ValueError):
        fft_recursive([1, 2], cutoff=0)


@pytest.mark.parametrize("method", list(FFTS))
@pytest.mark.parametrize("length, n", [(3, 4), (5, 8), (100, 128), (1000, 1024), (8, 8), (10, 8)])
def test_fft_with_n_pads_like_numpy(method, length, n):
    x = _random_complex(length, "pad", length, n)
    assert _rel_error(fft(x, n=n, method=method), np.fft.fft(x, n)) < 1e-10


def test_fft_with_n_not_power_of_two_raises():
    with pytest.raises(ValueError):
        fft(np.ones(5), n=6)


# ------------------------------------------------------------------ stages


@pytest.mark.parametrize("n", [1, 2, 16, 64])
def test_fft_iterative_stages_end_with_the_fft(n):
    x = _random_complex(n, "stages", n)
    stages = fft_iterative_stages(x)
    assert len(stages) == int(np.log2(n)) + 1
    assert all(s.shape == (n,) for s in stages)
    np.testing.assert_allclose(stages[-1], np.fft.fft(x), atol=1e-10)
    np.testing.assert_allclose(stages[-1], fft_iterative(x), atol=0)


def test_fft_iterative_stages_start_with_bit_reversal():
    stages = fft_iterative_stages(np.arange(8))
    np.testing.assert_array_equal(stages[0].real, [0, 4, 2, 6, 1, 5, 3, 7])
    # Each stage is an independent copy.
    stages[0][0] = 99
    assert stages[1][0] != 99


# ------------------------------------------------------------ frequencies


@pytest.mark.parametrize("n, fs", [(8, 1000), (1000, 1000), (7, 44100), (32768, 16000), (1, 10)])
def test_rfft_freqs_matches_numpy(n, fs):
    freqs = rfft_freqs(n, fs)
    assert freqs.shape == (n // 2 + 1,)
    np.testing.assert_allclose(freqs, np.fft.rfftfreq(n, 1 / fs), rtol=1e-12, atol=0)
