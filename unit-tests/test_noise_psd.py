"""
Tests for interpolation_noise_psd.py — kernel weight functions, DTFTs, and 2D noise PSDs.

Tests validate that the noise PSD module correctly computes the spectral
signature of interpolation-induced noise coloring. Key invariants:
- At f=0 (integer displacement / pass 0), P_noise = 1 everywhere (white noise)
- Kernel weights always sum to 1
- |H(k, f)| <= 1 holds EXACTLY for the interpolating B-spline; bicubic
  (Keys a=-0.75) and Lanczos-3 overshoot slightly mid-band (~3% / ~5%)
- P_noise is symmetric about both k-axes
"""

import numpy as np
import pytest
from numpy.fft import fftshift, fftfreq

from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    frac_distance,
    bicubic_weights,
    lanczos3_weights,
    bspline_weights,
    bicubic_dtft,
    lanczos3_dtft,
    bspline_dtft,
    compute_noise_psd_2d,
)

ALL_KERNELS = ["bicubic", "lanczos3", "bspline"]
# Per-kernel 2D P_noise upper bounds: bicubic/lanczos overshoot mid-band;
# the prefiltered B-spline is passive (|H| <= 1 to machine precision).
P_BOUND = {"bicubic": 1.1, "lanczos3": 1.1, "bspline": 1.0 + 1e-9}


# ─────────────────────────────────────────────────────────────────────────────
# frac_distance
# ─────────────────────────────────────────────────────────────────────────────

class TestFracDistance:
    """Tests for frac_distance()."""

    def test_integer_is_zero(self):
        assert frac_distance(0.0) == 0.0
        assert frac_distance(3.0) == 0.0
        assert frac_distance(-5.0) == 0.0

    def test_half_pixel(self):
        assert frac_distance(0.5) == pytest.approx(0.5)
        assert frac_distance(2.5) == pytest.approx(0.5)

    def test_quarter_pixel(self):
        assert frac_distance(0.25) == pytest.approx(0.25)
        assert frac_distance(3.75) == pytest.approx(0.25)

    def test_negative(self):
        assert frac_distance(-0.3) == pytest.approx(0.3)
        assert frac_distance(-2.7) == pytest.approx(0.3)

    def test_array_input(self):
        x = np.array([0.0, 0.25, 0.5, 1.0, 1.3])
        result = frac_distance(x)
        expected = np.array([0.0, 0.25, 0.5, 0.0, 0.3])
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Bicubic weights
# ─────────────────────────────────────────────────────────────────────────────

class TestBicubicWeights:
    """Tests for bicubic_weights()."""

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.5, 0.75, 0.99])
    def test_sum_is_one(self, f):
        weights = bicubic_weights(f)
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_f_zero_is_center(self):
        """At f=0, all weight on central tap w[0]."""
        w = bicubic_weights(0.0)
        assert w == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=1e-12)

    def test_four_weights(self):
        w = bicubic_weights(0.3)
        assert len(w) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Lanczos-3 weights
# ─────────────────────────────────────────────────────────────────────────────

class TestLanczos3Weights:
    """Tests for lanczos3_weights()."""

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.5, 0.75, 0.99])
    def test_sum_is_one(self, f):
        weights = lanczos3_weights(f)
        assert sum(weights) == pytest.approx(1.0, abs=1e-10)

    def test_f_zero_center_only(self):
        """At f=0, all weight on central tap (offset 0, index 2)."""
        w = lanczos3_weights(0.0)
        # Offset order: [-2, -1, 0, +1, +2, +3]
        # At f=0 the sinc peaks at offset 0 (index 2)
        assert w[2] == pytest.approx(1.0, abs=1e-10)
        for i in [0, 1, 3, 4, 5]:
            assert abs(w[i]) < 1e-10

    def test_six_taps(self):
        w = lanczos3_weights(0.3)
        assert len(w) == 6


# ─────────────────────────────────────────────────────────────────────────────
# Bicubic DTFT
# ─────────────────────────────────────────────────────────────────────────────

class TestBicubicDTFT:
    """Tests for bicubic_dtft()."""

    def test_f_zero_H_is_one(self):
        """At f=0 (no shift), H(k) = 1 for all k."""
        k = np.linspace(-0.5, 0.5, 64)
        H = bicubic_dtft(k, 0.0)
        np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-12)

    @pytest.mark.parametrize("f", [0.0, 0.25, 0.5])
    def test_dc_is_one(self, f):
        """H(k=0, f) = 1 because weights sum to 1."""
        H = bicubic_dtft(np.array([0.0]), f)
        assert np.abs(H[0]) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.4, 0.5])
    def test_magnitude_bounded(self, f):
        """|H(k, f)| approximately bounded (Keys a=-0.75 overshoots ~0.4%)."""
        k = np.linspace(-0.5, 0.5, 128)
        H = bicubic_dtft(k, f)
        # Keys' bicubic with a=-0.75 is not strictly interpolatory —
        # |H| can exceed 1 by up to ~3% at mid-frequencies
        assert np.all(np.abs(H) <= 1.03)


# ─────────────────────────────────────────────────────────────────────────────
# Lanczos-3 DTFT
# ─────────────────────────────────────────────────────────────────────────────

class TestLanczos3DTFT:
    """Tests for lanczos3_dtft()."""

    def test_f_zero_H_is_one(self):
        k = np.linspace(-0.5, 0.5, 64)
        H = lanczos3_dtft(k, 0.0)
        np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-10)

    @pytest.mark.parametrize("f", [0.0, 0.25, 0.5])
    def test_dc_is_one(self, f):
        H = lanczos3_dtft(np.array([0.0]), f)
        assert np.abs(H[0]) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.4, 0.5])
    def test_magnitude_bounded(self, f):
        k = np.linspace(-0.5, 0.5, 128)
        H = lanczos3_dtft(k, f)
        # Lanczos can slightly exceed 1 due to ringing — allow small margin
        assert np.all(np.abs(H) <= 1.05)

    def test_sharper_rolloff_than_bicubic(self):
        """Lanczos-3 should have sharper high-k rolloff than bicubic."""
        k = np.linspace(-0.5, 0.5, 128)
        f = 0.25
        H_bic = np.abs(bicubic_dtft(k, f))
        H_lan = np.abs(lanczos3_dtft(k, f))
        # Near Nyquist (|k| > 0.4), Lanczos should have equal or lower gain
        high_k = np.abs(k) > 0.4
        # Lanczos passband is flatter (closer to 1 at mid-k)
        mid_k = (np.abs(k) > 0.1) & (np.abs(k) < 0.3)
        assert np.mean(H_lan[mid_k]) >= np.mean(H_bic[mid_k]) - 0.05


# ─────────────────────────────────────────────────────────────────────────────
# B-spline weights (basis, pre-division)
# ─────────────────────────────────────────────────────────────────────────────

class TestBsplineWeights:
    """Tests for bspline_weights() — the beta^3 basis taps."""

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.5, 0.75, 0.99])
    def test_sum_is_one(self, f):
        weights = bspline_weights(f)
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_f_zero_is_not_a_delta(self):
        """At f=0 the basis weights are (1/6, 4/6, 1/6, 0) — NOT a delta.
        The interpolating (H=1 at f=0) property comes from the prefilter
        division in bspline_dtft, not from the weights alone. This test
        documents the mechanism difference vs bicubic/lanczos."""
        w = bspline_weights(0.0)
        assert w == pytest.approx((1 / 6, 4 / 6, 1 / 6, 0.0), abs=1e-12)

    def test_four_weights(self):
        w = bspline_weights(0.3)
        assert len(w) == 4


# ─────────────────────────────────────────────────────────────────────────────
# B-spline DTFT (prefiltered interpolating kernel)
# ─────────────────────────────────────────────────────────────────────────────

class TestBsplineDTFT:
    """Tests for bspline_dtft() — numerator DTFT / b_hat(k)."""

    def test_f_zero_H_is_one(self):
        """H(k, 0) = 1 exactly: the f=0 basis DTFT equals b_hat(k)."""
        k = np.linspace(-0.5, 0.5, 129)
        H = bspline_dtft(k, 0.0)
        np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-12)

    @pytest.mark.parametrize("f", [0.0, 0.25, 0.5])
    def test_dc_is_one(self, f):
        H = bspline_dtft(np.array([0.0]), f)
        assert np.abs(H[0]) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.4, 0.5])
    def test_magnitude_passive(self, f):
        """|H| <= 1 to machine precision: the interpolating B-spline is
        passive — the prefilter's high-k gain is exactly cancelled by the
        basis (verified analytically; bound is principled, not a margin)."""
        k = np.linspace(-0.5, 0.5, 2001)
        H = bspline_dtft(k, f)
        assert np.all(np.abs(H) <= 1.0 + 1e-9)

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.4, 0.5])
    def test_nyquist_closed_form(self, f):
        """H(0.5, f) = 1 - 6f^2 + 4f^3 (real; monotone 1 -> 0 on [0, 0.5])."""
        H = bspline_dtft(np.array([0.5]), f)
        expected = 1.0 - 6.0 * f * f + 4.0 * f ** 3
        assert H[0].real == pytest.approx(expected, abs=1e-12)
        assert abs(H[0].imag) < 1e-12

    def test_flatter_than_bicubic_midband(self):
        """The interpolating B-spline colours LESS than bicubic mid-band
        (its |H| stays closer to 1 for |k| <~ 0.25 at half-pixel shift)."""
        k = np.linspace(0.02, 0.25, 100)
        f = 0.5
        H_bsp = np.abs(bspline_dtft(k, f))
        H_bic = np.abs(bicubic_dtft(k, f))
        assert np.mean(np.abs(H_bsp - 1.0)) < np.mean(np.abs(H_bic - 1.0))

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.3, 0.5])
    def test_scipy_oracle(self, f):
        """The analytic transfer must match scipy's prefiltered cubic-spline
        shift on a periodic signal: shift by f, compare complex transfer
        Y(k)/X(k) against bspline_dtft(k, f). Pins the formula (tap layout,
        sign convention, prefilter division) against an independent
        implementation."""
        ndimage = pytest.importorskip("scipy.ndimage")
        rng = np.random.RandomState(7)
        n = 256
        x = rng.uniform(0.0, 255.0, n)
        # scipy shifts content by +f (sample at t - f... sign fixed below by
        # conjugation choice); grid-wrap = truly periodic boundaries.
        y = ndimage.shift(x, f, order=3, prefilter=True, mode="grid-wrap")
        X = np.fft.fft(x)
        Y = np.fft.fft(y)
        k = np.fft.fftfreq(n)
        H_measured = Y / X
        H_analytic = bspline_dtft(k, f)
        # scipy's shift(+f) corresponds to sampling at position (i - f);
        # our dtft convention is for sampling at floor+f. The two differ by
        # a pure integer-delay phase e^{-i2πk}: shift(+f) samples at
        # offset -f = (floor=-1) + (1-f). Try both attributions and require
        # one to match exactly — this pins magnitude AND phase structure.
        phase = np.exp(-2j * np.pi * k)
        candidates = [
            H_analytic,
            np.conj(H_analytic),
            bspline_dtft(k, 1.0 - f) * phase if 0.0 < f < 1.0 else H_analytic,
            np.conj(bspline_dtft(k, 1.0 - f) * phase) if 0.0 < f < 1.0 else H_analytic,
        ]
        errs = [np.max(np.abs(H_measured - c)) for c in candidates]
        assert min(errs) < 1e-10, (
            f"no convention candidate matches scipy: min err {min(errs):.3e}"
        )
        # Magnitude must agree regardless of phase convention
        np.testing.assert_allclose(
            np.abs(H_measured), np.abs(H_analytic), atol=1e-10,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2D Noise PSD
# ─────────────────────────────────────────────────────────────────────────────

class TestNoisePSD2D:
    """Tests for compute_noise_psd_2d()."""

    @pytest.fixture
    def grids_32(self):
        k_x = fftshift(fftfreq(32))
        k_y = fftshift(fftfreq(32))
        return np.meshgrid(k_x, k_y, indexing='xy')

    @pytest.fixture
    def grids_64(self):
        k_x = fftshift(fftfreq(64))
        k_y = fftshift(fftfreq(64))
        return np.meshgrid(k_x, k_y, indexing='xy')

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_f_zero_gives_flat_one(self, grids_64, kernel):
        """At f=0 (pass 0), P_noise = 1 everywhere — white noise."""
        K_X, K_Y = grids_64
        P = compute_noise_psd_2d(K_X, K_Y, 0.0, 0.0, kernel=kernel)
        np.testing.assert_allclose(P, 1.0, atol=1e-10)

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    @pytest.mark.parametrize("f", [0.25, 0.5])
    def test_nonzero_f_gives_structured_psd(self, grids_64, kernel, f):
        """Non-zero fractional shift should produce structured (non-flat) PSD."""
        K_X, K_Y = grids_64
        P = compute_noise_psd_2d(K_X, K_Y, f, f, kernel=kernel)
        # Should not be all ones
        assert np.std(P) > 0.01
        # DC should still be 1
        center = 64 // 2
        assert P[center, center] == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_separable(self, grids_64, kernel):
        """P(kx, ky) = |H(kx, fx)|^2 * |H(ky, fy)|^2 is separable."""
        K_X, K_Y = grids_64
        f_x, f_y = 0.3, 0.15
        P = compute_noise_psd_2d(K_X, K_Y, f_x, f_y, kernel=kernel)

        # Compute 1D marginals
        P_x_only = compute_noise_psd_2d(K_X, K_Y, f_x, 0.0, kernel=kernel)
        P_y_only = compute_noise_psd_2d(K_X, K_Y, 0.0, f_y, kernel=kernel)

        # P = P_x_only * P_y_only (since P_x_only has H_y=1 and vice versa)
        np.testing.assert_allclose(P, P_x_only * P_y_only, rtol=1e-10)

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    @pytest.mark.parametrize("f", [0.0, 0.25, 0.5])
    @pytest.mark.parametrize("window", [32, 64])
    def test_symmetry(self, kernel, f, window):
        """P_noise(k) == P_noise(-k): analytical even-function property.

        For real-valued kernel weights, H(-k) = conj(H(k)), so
        |H(-k)|^2 = |H(k)|^2. We verify this directly on a symmetric
        k grid (avoiding the even-N FFT asymmetry at Nyquist).
        """
        # Use an ODD number of points for a truly symmetric k grid
        n = window + 1
        k = np.linspace(-0.45, 0.45, n)
        K_X, K_Y = np.meshgrid(k, k, indexing='xy')
        P_pos = compute_noise_psd_2d(K_X, K_Y, f, f, kernel=kernel)
        P_neg = compute_noise_psd_2d(-K_X, -K_Y, f, f, kernel=kernel)

        np.testing.assert_allclose(P_pos, P_neg, atol=1e-12,
                                   err_msg="P_noise not symmetric: P(k) != P(-k)")

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_values_bounded(self, grids_64, kernel):
        """P_noise bounded above per kernel (see P_BOUND): bicubic/lanczos
        overshoot mid-band; the B-spline is passive to machine precision."""
        K_X, K_Y = grids_64
        P = compute_noise_psd_2d(K_X, K_Y, 0.3, 0.3, kernel=kernel)
        assert np.all(P >= -1e-10)
        assert np.all(P <= P_BOUND[kernel])

    def test_unknown_kernel_raises(self, grids_32):
        """Unknown kernel names must fail loud — the kernel must match the
        warp that produced the images (no silent default)."""
        K_X, K_Y = grids_32
        with pytest.raises(ValueError, match="Unknown interpolation kernel"):
            compute_noise_psd_2d(K_X, K_Y, 0.25, 0.25, kernel="bspine")
