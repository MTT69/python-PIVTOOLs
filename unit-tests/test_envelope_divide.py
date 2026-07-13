"""
Tests for the pair-count envelope divide (single_pass_accumulator._linear_pair_envelope).

The envelope E(Δ) = (W_A ⋆ W_B)(Δ) / (W_A ⋆ W_B)(0) is divided out of the accumulated
ensemble correlation planes before the k-space fit (Westerweel loss-of-correlation).
Analytic forms validated empirically against the production C correlator on 2026-07-06
(manual_tools/kspace/envelope_probe_empirical.py, <1%):

  std square window N      -> separable triangle product, E = prod_axis(1 - |d|/N)
  single autos (bsingle)   -> same triangle over the sum window
  single AB (singlepix/bsingle 16-in-32) -> trapezoid, plateau E = 1 for |d| <= 8
"""

import numpy as np
import pytest
from scipy.signal import correlate2d

from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_backend.single_pass_accumulator import _linear_pair_envelope


def bruteforce_envelope(wa, wb, corr_size):
    """Reference: direct linear cross-correlation, centre-cropped and centre-normalized."""
    full = correlate2d(wa.astype(np.float64), wb.astype(np.float64), mode="full")
    h, w = wa.shape
    ch, cw = corr_size
    y0 = (h - 1) - ch // 2
    x0 = (w - 1) - cw // 2
    env = full[y0 : y0 + ch, x0 : x0 + cw]
    return env / env[ch // 2, cw // 2]


def test_square_std_is_triangle_product():
    n = 32
    W = np.ones((n, n), dtype=np.float32)
    E = _linear_pair_envelope(W, W, (n, n))
    d = np.arange(n) - n // 2
    tri = 1.0 - np.abs(d) / n
    expected = np.outer(tri, tri)
    np.testing.assert_allclose(E, expected, atol=1e-10)
    assert E[n // 2, n // 2] == pytest.approx(1.0)
    assert E[0, 0] == pytest.approx(0.25)  # (1 - 1/2)^2 at the stored-plane corner


@pytest.mark.parametrize("wtype", ["square", "blackman", "gaussian"])
def test_matches_bruteforce_std(wtype):
    win = (32, 32)
    W = CrossCorrelator._window_weight_fun(win, wtype)
    E = _linear_pair_envelope(W, W, win)
    np.testing.assert_allclose(E, bruteforce_envelope(W, W, win), atol=1e-9)


def test_matches_bruteforce_single_mode():
    win, sumwin = (16, 16), (32, 32)
    WA = CrossCorrelator._window_weight_fun(win, "singlepix", sumwin)
    WB = CrossCorrelator._window_weight_fun(sumwin, "bsingle", sumwin)
    assert WA.shape == sumwin and WB.shape == sumwin
    E_ab = _linear_pair_envelope(WA, WB, sumwin)
    E_auto = _linear_pair_envelope(WB, WB, sumwin)
    np.testing.assert_allclose(E_ab, bruteforce_envelope(WA, WB, sumwin), atol=1e-9)
    np.testing.assert_allclose(E_auto, bruteforce_envelope(WB, WB, sumwin), atol=1e-9)


def test_single_ab_plateau_is_exactly_one():
    """The sum-window margin puts the AB peak on a trapezoid plateau E = 1 for
    |d| <= (sumwin - win)/2 = 8. The accumulator divides AB by this true envelope
    (2026-07-13) — a no-op on the plateau, a real correction beyond it (e.g. when
    a sum-window axis has zero margin and the envelope is a triangle)."""
    win, sumwin = (16, 16), (32, 32)
    WA = CrossCorrelator._window_weight_fun(win, "singlepix", sumwin)
    WB = CrossCorrelator._window_weight_fun(sumwin, "bsingle", sumwin)
    E_ab = _linear_pair_envelope(WA, WB, sumwin)
    c = sumwin[0] // 2
    plateau = E_ab[c - 8 : c + 9, c - 8 : c + 9]
    np.testing.assert_allclose(plateau, 1.0, atol=1e-9)
    # and it genuinely falls off beyond the plateau (corner of the stored plane)
    assert E_ab[0, 0] < 0.6


def test_single_auto_is_sum_window_triangle():
    sumwin = (32, 32)
    WB = CrossCorrelator._window_weight_fun(sumwin, "bsingle", sumwin)
    E_auto = _linear_pair_envelope(WB, WB, sumwin)
    d = np.arange(32) - 16
    tri = 1.0 - np.abs(d) / 32
    np.testing.assert_allclose(E_auto, np.outer(tri, tri), atol=1e-10)


def test_fit_window_crop_matches_full():
    """When ensemble_sum_fitting_window trims the stored plane, the envelope must be
    the central crop of the full-support envelope (same zero-lag pixel)."""
    sumwin = (32, 32)
    WB = CrossCorrelator._window_weight_fun(sumwin, "bsingle", sumwin)
    E_full = _linear_pair_envelope(WB, WB, (32, 32))
    E_crop = _linear_pair_envelope(WB, WB, (24, 24))
    np.testing.assert_allclose(
        E_crop, E_full[16 - 12 : 16 + 12, 16 - 12 : 16 + 12], atol=1e-12
    )
    assert E_crop[12, 12] == pytest.approx(1.0)


def test_rectangular_window():
    win = (16, 32)
    W = CrossCorrelator._window_weight_fun(win, "square")
    E = _linear_pair_envelope(W, W, win)
    dy = np.arange(16) - 8
    dx = np.arange(32) - 16
    expected = np.outer(1.0 - np.abs(dy) / 16, 1.0 - np.abs(dx) / 32)
    np.testing.assert_allclose(E, expected, atol=1e-10)


def test_corr_size_exceeding_support_raises():
    W = np.ones((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds envelope support"):
        _linear_pair_envelope(W, W, (32, 32))


def test_mismatched_weight_shapes_raise():
    with pytest.raises(ValueError, match="share a shape"):
        _linear_pair_envelope(np.ones((16, 16)), np.ones((32, 32)), (16, 16))
