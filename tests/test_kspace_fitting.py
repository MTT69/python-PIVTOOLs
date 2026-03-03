"""Tests for k-space fitting pipeline.

Tests production code in:
  - pivtools_cli/piv/piv_backend/interpolation_noise_psd.py
  - pivtools_cli/piv/piv_backend/kspace_fitting.py

All tests use synthetic data (no .mat files needed).
"""
import numpy as np
import pytest

from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    bicubic_weights, bicubic_dtft,
    lanczos3_weights, lanczos3_dtft,
    compute_noise_psd_2d, frac_distance,
)
from pivtools_cli.piv.piv_backend.kspace_fitting import (
    _fit_fref_joint,
    _fit_single_window_kspace,
)


class TestInterpolationNoisePSD:
    """Test the analytical interpolation kernel noise PSD module."""

    def test_bicubic_weights_at_zero(self):
        """At f=0, all weight on central tap -> H(k)=1 -> P_noise=1."""
        w = bicubic_weights(0.0)
        assert w == pytest.approx((0.0, 1.0, 0.0, 0.0))

    def test_bicubic_weights_sum_to_one(self):
        """Bicubic weights must partition unity for any f."""
        for f in [0.0, 0.1, 0.25, 0.5]:
            assert sum(bicubic_weights(f)) == pytest.approx(1.0, abs=1e-14)

    def test_bicubic_weights_at_half_pixel(self):
        """At f=0.5 with a=-0.75 (OpenCV), known analytical weights."""
        w = bicubic_weights(0.5)
        expected = (-0.09375, 0.59375, 0.59375, -0.09375)
        assert w == pytest.approx(expected, abs=1e-10)

    def test_bicubic_dtft_unity_at_zero_frac(self):
        """At f=0, H(k)=1 for all k (delta function)."""
        k = np.linspace(-0.5, 0.5, 65)
        H = bicubic_dtft(k, 0.0)
        np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-14)

    def test_bicubic_dtft_dc_always_one(self):
        """H(k=0) = sum of weights = 1 for any fractional shift."""
        for f in [0.0, 0.1, 0.25, 0.5]:
            H_dc = bicubic_dtft(np.array([0.0]), f)
            assert abs(H_dc[0]) == pytest.approx(1.0, abs=1e-14)

    def test_bicubic_pnoise_flat_at_zero(self):
        """P_noise=1 everywhere when f_x=f_y=0 (no warping)."""
        k = np.linspace(-0.5, 0.5, 33)
        K_X, K_Y = np.meshgrid(k, k)
        P = compute_noise_psd_2d(K_X, K_Y, 0.0, 0.0, kernel='bicubic')
        np.testing.assert_allclose(P, 1.0, atol=1e-14)

    def test_lanczos3_weights_at_zero(self):
        """At f=0, all weight on central tap."""
        w = lanczos3_weights(0.0)
        assert w == pytest.approx((0, 0, 1, 0, 0, 0), abs=1e-12)

    def test_lanczos3_weights_sum_to_one(self):
        """Lanczos-3 weights must partition unity for any f."""
        for f in [0.0, 0.1, 0.25, 0.5]:
            assert sum(lanczos3_weights(f)) == pytest.approx(1.0, abs=1e-12)

    def test_lanczos3_dtft_dc_always_one(self):
        """Lanczos-3 H(k=0) = 1 for any fractional shift."""
        for f in [0.0, 0.1, 0.25, 0.5]:
            H_dc = lanczos3_dtft(np.array([0.0]), f)
            assert abs(H_dc[0]) == pytest.approx(1.0, abs=1e-12)

    def test_lanczos3_pnoise_flat_at_zero(self):
        """Lanczos-3 P_noise=1 everywhere when f=0."""
        k = np.linspace(-0.5, 0.5, 33)
        K_X, K_Y = np.meshgrid(k, k)
        P = compute_noise_psd_2d(K_X, K_Y, 0.0, 0.0, kernel='lanczos3')
        np.testing.assert_allclose(P, 1.0, atol=1e-12)

    def test_pnoise_attenuates_at_half_pixel(self):
        """At frac=0.5, high-k noise PSD should be significantly < 1."""
        k = np.linspace(-0.5, 0.5, 33)
        K_X, K_Y = np.meshgrid(k, k)
        P = compute_noise_psd_2d(K_X, K_Y, 0.5, 0.5, kernel='bicubic')
        corners = (np.abs(K_X) > 0.35) & (np.abs(K_Y) > 0.35)
        assert np.median(P[corners]) < 0.3  # Significant attenuation

    def test_pnoise_symmetric_in_k(self):
        """P_noise should be symmetric: P(kx, ky) = P(-kx, -ky)."""
        k = np.linspace(-0.5, 0.5, 33)
        K_X, K_Y = np.meshgrid(k, k)
        P = compute_noise_psd_2d(K_X, K_Y, 0.3, 0.2, kernel='bicubic')
        np.testing.assert_allclose(P, P[::-1, ::-1], atol=1e-12)

    def test_frac_distance(self):
        """Fractional distance to nearest integer."""
        assert frac_distance(0.0) == pytest.approx(0.0)
        assert frac_distance(0.5) == pytest.approx(0.5)
        assert frac_distance(1.0) == pytest.approx(0.0)
        assert frac_distance(1.3) == pytest.approx(0.3)
        assert frac_distance(-0.7) == pytest.approx(0.3, abs=1e-14)

    def test_frac_distance_vectorized(self):
        """frac_distance works on arrays."""
        x = np.array([0.0, 0.5, 1.0, 2.3, -0.7])
        expected = np.array([0.0, 0.5, 0.0, 0.3, 0.3])
        np.testing.assert_allclose(frac_distance(x), expected, atol=1e-14)


class TestFitFrefJoint:
    """Test the production joint noise+signal fitter."""

    def _make_synthetic_fref(self, corr_n=32, sigma=2.0, N0=0.05,
                              f_x=0.0, f_y=0.0):
        """Build a synthetic F_ref with known parameters.

        Uses the approximate model (production): F_ref = A*Gauss + N0*P_noise
        """
        k1d = np.fft.fftshift(np.fft.fftfreq(corr_n))
        K_X, K_Y = np.meshgrid(k1d, k1d)
        gaussian = np.exp(-2 * np.pi**2 * sigma**2 * (K_X**2 + K_Y**2))
        P_noise = compute_noise_psd_2d(K_X, K_Y, f_x, f_y)
        # Approximate model (matches production _fit_fref_joint line 314):
        F_ref = gaussian + N0 * P_noise
        return F_ref, K_X, K_Y, P_noise

    def test_joint_fit_recovers_parameters_pass0(self):
        """At f=0 (pass 0), P_noise=1, joint fit should recover sigma."""
        F_ref, K_X, K_Y, P_noise = self._make_synthetic_fref(
            sigma=2.0, N0=0.05, f_x=0.0, f_y=0.0)
        result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)
        assert result['success']
        assert result['sigma_x'] == pytest.approx(2.0, rel=0.05)
        assert result['sigma_y'] == pytest.approx(2.0, rel=0.05)

    def test_joint_fit_recovers_parameters_warped(self):
        """At f=0.3, joint fit should still recover parameters."""
        F_ref, K_X, K_Y, P_noise = self._make_synthetic_fref(
            sigma=2.0, N0=0.05, f_x=0.3, f_y=0.2)
        result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)
        assert result['success']
        assert result['sigma_x'] == pytest.approx(2.0, rel=0.1)
        assert result['sigma_y'] == pytest.approx(2.0, rel=0.1)

    def test_joint_fit_clean_fref_nonnegative(self):
        """F_ref_clean should be non-negative everywhere."""
        F_ref, K_X, K_Y, P_noise = self._make_synthetic_fref(
            sigma=2.0, N0=0.1, f_x=0.5, f_y=0.5)
        result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)
        assert result['success']
        assert np.all(result['F_ref_clean'] >= 0)

    def test_joint_fit_n0_recovery_pass0(self):
        """At f=0, P_noise=1, so N0 should match the input noise level."""
        F_ref, K_X, K_Y, P_noise = self._make_synthetic_fref(
            sigma=2.0, N0=0.05, f_x=0.0, f_y=0.0)
        result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)
        assert result['success']
        # F_dc ≈ A + N0 ≈ 1.05 (Gaussian peak is 1.0 at k=0)
        # N0_abs = N0_relative * F_dc ≈ 0.05 * 1.05
        assert result['N0'] == pytest.approx(0.05 * 1.05, rel=0.15)

    def test_joint_fit_fails_on_zero_input(self):
        """Joint fit should handle zero F_ref gracefully."""
        corr_n = 32
        k1d = np.fft.fftshift(np.fft.fftfreq(corr_n))
        K_X, K_Y = np.meshgrid(k1d, k1d)
        F_ref = np.zeros((corr_n, corr_n))
        P_noise = np.ones((corr_n, corr_n))
        result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)
        assert not result['success']


class TestFitSingleWindowKspace:
    """Test the full single-window k-space fitting pipeline."""

    def _make_synthetic_window(self, corr_n=32, sigma_A=2.0, delta_x=0.3,
                                delta_y=-0.1, Sigma_xx=0.5, Sigma_yy=0.3,
                                N0=0.02):
        """Build synthetic correlation planes for a single window.

        Generates R_AA, R_BB, R_AB consistent with the k-space model.
        """
        k1d = np.fft.fftshift(np.fft.fftfreq(corr_n))
        K_X, K_Y = np.meshgrid(k1d, k1d)

        # Particle image envelope (sigma_A cancels in T, but shapes F_ref)
        gauss_A = np.exp(-2 * np.pi**2 * sigma_A**2 * (K_X**2 + K_Y**2))

        # Transfer function: T(k) = exp(-2pi^2(Sigma_xx*kx^2 + Sigma_yy*ky^2))
        #                          * exp(-i*2pi*(delta_x*kx + delta_y*ky))
        T_envelope = np.exp(-2 * np.pi**2 * (Sigma_xx * K_X**2 + Sigma_yy * K_Y**2))
        T_phase = np.exp(-1j * 2 * np.pi * (delta_x * K_X + delta_y * K_Y))
        T = T_envelope * T_phase

        # F_AA, F_BB ~ gauss_A (no cross-pair broadening)
        F_AA = gauss_A + N0
        F_BB = gauss_A + N0
        F_ref = np.sqrt(F_AA * F_BB)

        # F_AB = T * F_ref
        F_AB = T * F_ref

        # Convert to spatial domain correlation planes
        from numpy.fft import ifft2, ifftshift, fftshift
        R_AA = np.real(fftshift(ifft2(ifftshift(F_AA))))
        R_BB = np.real(fftshift(ifft2(ifftshift(F_BB))))
        R_AB = np.real(fftshift(ifft2(ifftshift(F_AB))))

        center_x = corr_n / 2.0
        center_y = corr_n / 2.0

        return R_AA, R_BB, R_AB, K_X, K_Y, k1d, k1d, center_x, center_y

    def test_kspace_fit_recovers_displacement(self):
        """K-space fitter should recover the sub-pixel displacement."""
        corr_n = 32
        R_AA, R_BB, R_AB, K_X, K_Y, k_x, k_y, cx, cy = \
            self._make_synthetic_window(
                corr_n=corr_n, delta_x=0.3, delta_y=-0.1,
                Sigma_xx=0.5, Sigma_yy=0.3)

        result = _fit_single_window_kspace(
            R_AA, R_BB, R_AB, K_X, K_Y, k_x, k_y,
            corr_size=(corr_n, corr_n), snr_threshold=1.0,
            center_x=cx, center_y=cy,
            use_soft_weighting=True,
        )

        assert result['status'] == 0
        params = result['params']
        mu_x = params[14] - params[12]
        mu_y = params[15] - params[13]
        assert mu_x == pytest.approx(0.3, abs=0.15)
        assert mu_y == pytest.approx(-0.1, abs=0.15)

    def test_kspace_fit_recovers_stresses(self):
        """K-space fitter should recover displacement variance (stresses)."""
        corr_n = 32
        R_AA, R_BB, R_AB, K_X, K_Y, k_x, k_y, cx, cy = \
            self._make_synthetic_window(
                corr_n=corr_n, delta_x=0.0, delta_y=0.0,
                Sigma_xx=0.8, Sigma_yy=0.4)

        result = _fit_single_window_kspace(
            R_AA, R_BB, R_AB, K_X, K_Y, k_x, k_y,
            corr_size=(corr_n, corr_n), snr_threshold=1.0,
            center_x=cx, center_y=cy,
            use_soft_weighting=True,
        )

        assert result['status'] == 0
        params = result['params']
        # sigma_AB_x - sigma_A_x = Sigma_xx
        Sigma_xx_recovered = params[9]
        Sigma_yy_recovered = params[10]
        assert Sigma_xx_recovered == pytest.approx(0.8, rel=0.3)
        assert Sigma_yy_recovered == pytest.approx(0.4, abs=0.25)
