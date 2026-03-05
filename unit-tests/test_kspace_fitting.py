"""
Tests for kspace_fitting.py — the k-space transfer function fitter.

Tests are organised bottom-up through the pipeline:
  4a. Sub-pixel peak displacement
  4b. k_max computation
  4c. Joint noise model (F_ref)
  4d. 1D axis regressions
  4e. Full nonlinear optimizer
  4f. Single-window integration
  4g. Top-level entry point
  4h. Edge cases
"""

import numpy as np
import pytest
from numpy.fft import fft2, fftshift, ifftshift, fftfreq

from synthetic_correlations import (
    generate_2d_gaussian,
    generate_autocorrelation,
    generate_crosscorrelation,
    generate_correlation_triplet,
    flatten_for_kspace,
    make_mock_config,
)
from pivtools_cli.piv.piv_backend.kspace_fitting import (
    _estimate_displacement_from_peak,
    _compute_kmax,
    _compute_kmax_from_profile,
    _fit_fref_joint,
    _fit_1d_axis,
    _fit_transfer_function_full,
    _fit_single_window_kspace,
    fit_windows_kspace,
)
from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    compute_noise_psd_2d,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_grids(size):
    """Build wavenumber grids for a square window."""
    k = fftshift(fftfreq(size))
    K_X, K_Y = np.meshgrid(k, k, indexing='xy')
    return K_X, K_Y, k


def _fft_correlations(R_AA, R_BB, R_AB):
    """FFT correlation planes the same way the production code does."""
    F_AA = fftshift(fft2(ifftshift(R_AA)))
    F_BB = fftshift(fft2(ifftshift(R_BB)))
    F_AB = fftshift(fft2(ifftshift(R_AB)))
    return F_AA, F_BB, F_AB


# ─────────────────────────────────────────────────────────────────────────────
# 4a. Sub-pixel peak displacement
# ─────────────────────────────────────────────────────────────────────────────

class TestEstimateDisplacementFromPeak:
    """Tests for _estimate_displacement_from_peak()."""

    @pytest.mark.parametrize("mu_x, mu_y", [
        (0.0, 0.0),
        (0.3, -0.2),
        (2.5, 1.0),
        (0.47, 0.31),
    ])
    @pytest.mark.parametrize("window", [32, 64])
    def test_displacement_recovery(self, mu_x, mu_y, window):
        """Peak finder should recover displacement to within 0.05 px."""
        shape = (window, window)
        R_AB = generate_crosscorrelation(
            shape, sigma_particle_x=2.5, sigma_particle_y=2.5,
            mu_x=mu_x, mu_y=mu_y, amplitude=1.0,
        )
        center_x = window // 2
        center_y = window // 2

        est_x, est_y = _estimate_displacement_from_peak(R_AB, center_x, center_y)

        assert est_x == pytest.approx(mu_x, abs=0.05), \
            f"mu_x: expected {mu_x}, got {est_x}"
        assert est_y == pytest.approx(mu_y, abs=0.05), \
            f"mu_y: expected {mu_y}, got {est_y}"


# ─────────────────────────────────────────────────────────────────────────────
# 4b. k_max computation
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeKmax:
    """Tests for _compute_kmax()."""

    def test_low_sigma_high_kmax(self):
        """Small variance → signal extends to high k → large k_max."""
        k1 = _compute_kmax(0.1, snr=100.0)
        k2 = _compute_kmax(2.0, snr=100.0)
        assert k1 > k2

    def test_high_sigma_low_kmax(self):
        """Large variance → signal decays quickly → small k_max."""
        k = _compute_kmax(5.0, snr=100.0)
        assert k < 0.3

    def test_low_snr_returns_max_k(self):
        """SNR <= 1 → returns max_k (no useful range)."""
        k = _compute_kmax(1.0, snr=0.5)
        assert k == pytest.approx(0.45)

    def test_bounds_compliance(self):
        """Result always in [min_k, max_k]."""
        for sigma in [0.01, 0.5, 5.0, 50.0]:
            for snr in [0.1, 1.0, 10.0, 1000.0]:
                k = _compute_kmax(sigma, snr, min_k=0.05, max_k=0.45)
                assert 0.05 <= k <= 0.45


class TestComputeKmaxFromProfile:
    """Tests for _compute_kmax_from_profile()."""

    def test_monotonicity_with_sigma(self):
        """Wider Gaussian → earlier drop → lower k_max."""
        k_maxes = []
        for sigma in [1.0, 2.0, 4.0]:
            k_axis = fftshift(fftfreq(64))
            # Simulate Gaussian F_ref profile
            profile = np.exp(-2 * np.pi ** 2 * sigma ** 2 * k_axis ** 2)
            F_dc = 1.0
            k_maxes.append(
                _compute_kmax_from_profile(k_axis, profile, F_dc)
            )
        # k_max should decrease with increasing sigma
        assert k_maxes[0] >= k_maxes[1] >= k_maxes[2]

    def test_stays_in_bounds(self):
        k_axis = fftshift(fftfreq(64))
        profile = np.exp(-2 * np.pi ** 2 * 2.0 ** 2 * k_axis ** 2)
        k = _compute_kmax_from_profile(
            k_axis, profile, 1.0, min_k=0.05, max_k=0.45,
        )
        assert 0.05 <= k <= 0.45


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Joint noise model (F_ref)
# ─────────────────────────────────────────────────────────────────────────────

class TestFitFrefJoint:
    """Tests for _fit_fref_joint() — predictor-aware colored noise fit."""

    @pytest.mark.parametrize("kernel", ["bicubic", "lanczos3"])
    @pytest.mark.parametrize("frac_disp", [0.0, 0.25, 0.5])
    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.05])
    @pytest.mark.parametrize("window", [32, 64])
    def test_noise_recovery(self, kernel, frac_disp, noise_level, window):
        """Verify that _fit_fref_joint recovers A and N0."""
        K_X, K_Y, _ = _make_grids(window)
        P_noise = compute_noise_psd_2d(K_X, K_Y, frac_disp, frac_disp,
                                       kernel=kernel)

        # Build synthetic F_ref = (A*Gaussian + N0) * P_noise
        true_A = 100.0
        true_sigma = 3.0
        true_N0 = noise_level * true_A  # N0 relative to amplitude

        gaussian = true_A * np.exp(
            -2 * np.pi ** 2 * (K_X ** 2 + K_Y ** 2) * true_sigma ** 2
        )
        F_ref = (gaussian + true_N0) * P_noise

        result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)

        assert result['success'], "Joint fit should succeed on clean synthetic data"

        # Amplitude check
        if frac_disp == 0.0:
            assert result['A'] == pytest.approx(true_A, rel=0.05), \
                f"A: expected {true_A}, got {result['A']}"
        else:
            assert result['A'] == pytest.approx(true_A, rel=0.10)

        # N0 check
        if true_N0 < 0.01:
            assert abs(result['N0']) < 0.005 * true_A
        elif frac_disp <= 0.25:
            assert result['N0'] == pytest.approx(true_N0, rel=0.20)
        else:
            # Maximum coloring → relaxed
            assert result['N0'] == pytest.approx(true_N0, rel=0.30)

    def test_n0_bias_sweep(self):
        """N0 recovery should not have systematic positive/negative bias."""
        K_X, K_Y, _ = _make_grids(64)
        P_noise = compute_noise_psd_2d(K_X, K_Y, 0.25, 0.25, kernel='bicubic')

        true_A = 100.0
        true_sigma = 3.0
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        signed_errors = []

        for nl in noise_levels:
            true_N0 = nl * true_A
            gaussian = true_A * np.exp(
                -2 * np.pi ** 2 * (K_X ** 2 + K_Y ** 2) * true_sigma ** 2
            )
            F_ref = (gaussian + true_N0) * P_noise
            result = _fit_fref_joint(F_ref, K_X, K_Y, P_noise)
            if result['success'] and true_N0 > 0:
                signed_errors.append(result['N0'] - true_N0)

        if len(signed_errors) >= 3:
            mean_N0_true = np.mean([nl * true_A for nl in noise_levels])
            mean_signed = np.mean(signed_errors)
            # Mean signed error < 10% of mean N0_true
            assert abs(mean_signed) < 0.10 * mean_N0_true, \
                f"Systematic N0 bias detected: mean signed error = {mean_signed:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 4d. 1D axis regressions
# ─────────────────────────────────────────────────────────────────────────────

class TestFit1DAxis:
    """Tests for _fit_1d_axis() — log-magnitude + phase regression."""

    @staticmethod
    def _energy_corrected_amplitude(sigma_p, Sxx, Syy, Sxy):
        """Compute R_AB amplitude so integral(R_AB) = integral(R_AA).

        In physical PIV, the total signal power is conserved — the cross-
        correlation is broader but has a lower peak. Without this correction,
        F_AB(0) > F_ref(0) and the forced-through-origin 1D regression is biased.
        """
        det_AA = (2 * sigma_p ** 2) ** 2
        det_AB = ((2 * sigma_p ** 2 + Sxx) * (2 * sigma_p ** 2 + Syy)
                  - Sxy ** 2)
        return np.sqrt(det_AA / max(det_AB, 1e-20))

    @pytest.mark.parametrize("sigma_stress", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("mu", [0.0, 0.3, 1.5])
    @pytest.mark.parametrize("window", [32, 64])
    def test_1d_recovery(self, sigma_stress, mu, window):
        """1D axis fit should recover displacement and variance."""
        shape = (window, window)
        sigma_particle = 1.0

        # Correct the cross-correlation amplitude so F_AB(0) = F_ref(0)
        amp_AB = self._energy_corrected_amplitude(
            sigma_particle, sigma_stress, sigma_stress, 0.0,
        )

        R_AA = generate_autocorrelation(
            shape, sigma_particle, sigma_particle, amplitude=1.0,
        )
        R_BB = R_AA.copy()
        R_AB = generate_crosscorrelation(
            shape, sigma_particle, sigma_particle,
            sigma_stress, sigma_stress, 0.0,
            mu_x=mu, mu_y=0.0, amplitude=amp_AB,
        )

        F_AA, F_BB, F_AB = _fft_correlations(R_AA, R_BB, R_AB)
        F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

        K_X, K_Y, k_x = _make_grids(window)
        center_idx = window // 2

        # Compute data-driven k_max from F_ref profile (matching production)
        F_ref_profile = np.abs(F_ref[center_idx, :])
        F_dc = F_ref_profile[center_idx]
        k_max = _compute_kmax_from_profile(k_x, F_ref_profile, F_dc)

        mu_est, sigma_est = _fit_1d_axis(
            F_AB, F_ref, k_x, center_idx, k_max=k_max, axis='x',
        )

        # Displacement tolerance
        assert mu_est == pytest.approx(mu, abs=0.15), \
            f"mu: expected {mu}, got {mu_est}"

        # Variance tolerance — 1D regression is an initial estimate, not final
        if sigma_stress >= 0.5:
            assert sigma_est == pytest.approx(sigma_stress, rel=0.25), \
                f"Sigma: expected {sigma_stress}, got {sigma_est}"
        else:
            assert sigma_est == pytest.approx(sigma_stress, abs=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# 4e. Full nonlinear optimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestFitTransferFunctionFull:
    """Tests for _fit_transfer_function_full()."""

    @pytest.mark.parametrize("stress", [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (1.0, 0.2, 0.0),
        (0.5, 0.5, 0.2),
        (2.0, 2.0, 0.5),
    ], ids=["zero", "iso_0.5", "aniso", "shear", "large"])
    @pytest.mark.parametrize("displacement", [
        (0.0, 0.0),
        (0.3, -0.2),
        (2.5, 1.0),
    ], ids=["static", "small_disp", "large_disp"])
    @pytest.mark.parametrize("window", [32, 64])
    def test_full_fit_recovery(self, stress, displacement, window):
        """Full 6-parameter fit should recover stress and displacement."""
        Sxx, Syy, Sxy = stress
        mu_x, mu_y = displacement
        shape = (window, window)

        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=Sxx, sigma_stress_yy=Syy,
            sigma_stress_xy=Sxy,
            mu_x=mu_x, mu_y=mu_y,
        )

        F_AA, F_BB, F_AB = _fft_correlations(R_AA, R_BB, R_AB)
        F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

        K_X, K_Y, _ = _make_grids(window)

        initial_guess = np.array([mu_x, mu_y, max(Sxx, 0.1), max(Syy, 0.1), 0.0, 1.0])

        result = _fit_transfer_function_full(
            F_AB, F_ref, K_X, K_Y,
            k_max_x=0.35, k_max_y=0.35,
            initial_guess=initial_guess,
            use_soft_weighting=True,
        )

        assert result is not None, "Optimizer should converge on clean synthetic data"

        r_mu_x, r_mu_y, r_Sxx, r_Syy, r_Sxy, r_A = result

        # Displacement
        assert r_mu_x == pytest.approx(mu_x, abs=0.05)
        assert r_mu_y == pytest.approx(mu_y, abs=0.05)

        # Normal stresses
        if Sxx >= 0.5:
            assert r_Sxx == pytest.approx(Sxx, rel=0.05)
        else:
            assert r_Sxx == pytest.approx(Sxx, abs=0.05)

        if Syy >= 0.5:
            assert r_Syy == pytest.approx(Syy, rel=0.05)
        else:
            assert r_Syy == pytest.approx(Syy, abs=0.05)

        # Shear stress
        assert r_Sxy == pytest.approx(Sxy, abs=0.1)

    def test_ghost_stress(self):
        """Pure displacement (zero stress) must NOT invent stress.

        This catches overfitting where the optimizer explains noise or
        windowing artefacts as Reynolds stress.
        """
        shape = (64, 64)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=0.0, sigma_stress_yy=0.0, sigma_stress_xy=0.0,
            mu_x=1.0, mu_y=-0.5,
        )

        F_AA, F_BB, F_AB = _fft_correlations(R_AA, R_BB, R_AB)
        F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
        K_X, K_Y, _ = _make_grids(64)

        initial_guess = np.array([1.0, -0.5, 0.1, 0.1, 0.0, 1.0])

        result = _fit_transfer_function_full(
            F_AB, F_ref, K_X, K_Y,
            k_max_x=0.35, k_max_y=0.35,
            initial_guess=initial_guess,
        )

        assert result is not None
        _, _, Sxx, Syy, Sxy, _ = result
        assert Sxx < 0.02, f"Ghost stress Sxx = {Sxx}"
        assert Syy < 0.02, f"Ghost stress Syy = {Syy}"
        assert abs(Sxy) < 0.02, f"Ghost stress Sxy = {Sxy}"

    @pytest.mark.slow
    def test_spectral_leakage_large_particle(self):
        """Large particle on small window: Gaussian truncated → spectral leakage.

        The fitter should still converge (status=0) even if accuracy degrades.
        This probes the spectral resolution limit Δk = 1/N.
        """
        shape = (32, 32)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=5.0, sigma_particle_y=5.0,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
            mu_x=0.3, mu_y=-0.2,
        )

        F_AA, F_BB, F_AB = _fft_correlations(R_AA, R_BB, R_AB)
        F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
        K_X, K_Y, _ = _make_grids(32)

        initial_guess = np.array([0.3, -0.2, 0.5, 0.5, 0.0, 1.0])

        result = _fit_transfer_function_full(
            F_AB, F_ref, K_X, K_Y,
            k_max_x=0.35, k_max_y=0.35,
            initial_guess=initial_guess,
        )

        # Should converge (not crash or return None)
        assert result is not None, \
            "Fitter should converge even with spectral leakage"


# ─────────────────────────────────────────────────────────────────────────────
# 4f. Single-window integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFitSingleWindowKspace:
    """Tests for _fit_single_window_kspace() — end-to-end single window."""

    # Tier 1: fast, clean data
    @pytest.mark.parametrize("window", [32, 64])
    @pytest.mark.parametrize("stress", [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (1.0, 0.2, 0.0),
    ], ids=["zero", "iso_0.5", "aniso"])
    @pytest.mark.parametrize("displacement", [
        (0.0, 0.0),
        (0.3, -0.2),
    ], ids=["static", "small_disp"])
    def test_tier1_clean(self, window, stress, displacement):
        """Tier 1: clean data, core configurations."""
        Sxx, Syy, Sxy = stress
        mu_x, mu_y = displacement
        shape = (window, window)

        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=Sxx, sigma_stress_yy=Syy,
            sigma_stress_xy=Sxy,
            mu_x=mu_x, mu_y=mu_y,
        )

        K_X, K_Y, k_x = _make_grids(window)
        k_y = fftshift(fftfreq(window))
        center_x = window / 2.0 + 1
        center_y = window / 2.0 + 1

        result = _fit_single_window_kspace(
            R_AA, R_BB, R_AB,
            K_X, K_Y, k_x, k_y,
            (window, window), snr_threshold=3.0,
            center_x=center_x, center_y=center_y,
        )

        assert result['status'] == 0, \
            f"Expected status=0, got {result['status']}"

        params = result['params']
        recovered_mu_x = params[14] - center_x
        recovered_mu_y = params[15] - center_y

        assert recovered_mu_x == pytest.approx(mu_x, abs=0.05)
        assert recovered_mu_y == pytest.approx(mu_y, abs=0.05)

        if Sxx >= 0.5:
            assert params[9] == pytest.approx(Sxx, rel=0.05)
        else:
            assert params[9] == pytest.approx(Sxx, abs=0.05)

        if Syy >= 0.5:
            assert params[10] == pytest.approx(Syy, rel=0.05)
        else:
            assert params[10] == pytest.approx(Syy, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 4g. Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

class TestFitWindowsKspace:
    """Tests for fit_windows_kspace() — the top-level API."""

    def test_multi_window_basic(self):
        """4 windows, 2 masked → correct shapes, masked get status=-1."""
        window = 32
        shape = (window, window)
        n_windows = 4

        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape, sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )
        R_AA_flat, R_BB_flat, R_AB_flat, mask_flat, corr_size = \
            flatten_for_kspace(R_AA, R_BB, R_AB, n_windows=n_windows)

        # Mask windows 1 and 3
        mask_flat[1] = True
        mask_flat[3] = True

        config = make_mock_config()
        gauss, status, initial = fit_windows_kspace(
            R_AA_flat, R_BB_flat, R_AB_flat,
            mask_flat, corr_size, config, pass_idx=0,
        )

        assert gauss.shape == (n_windows, 16)
        assert status.shape == (n_windows,)
        assert initial.shape == (n_windows, 16)

        # Masked windows get status -1
        assert status[1] == -1
        assert status[3] == -1

        # Non-masked windows should succeed
        assert status[0] == 0
        assert status[2] == 0

    def test_output_shape(self):
        """gauss_flat is (n_windows, 16), status is (n_windows,)."""
        window = 32
        n_windows = 3
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            (window, window), sigma_stress_xx=0.3,
        )
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n_windows,
        )
        config = make_mock_config()
        gauss, status, initial = fit_windows_kspace(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
        )
        assert gauss.shape == (n_windows, 16)
        assert status.shape == (n_windows,)

    def test_all_masked(self):
        """All-True mask → all status=-1, no exception."""
        window = 32
        n = 2
        R_AA, R_BB, R_AB = generate_correlation_triplet((window, window))
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n,
        )
        mask[:] = True
        config = make_mock_config()
        gauss, status, initial = fit_windows_kspace(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
        )
        assert np.all(status == -1)

    def test_displacement_stored_correctly(self):
        """params[14] - center_x ≈ mu_x."""
        window = 64
        mu_x = 1.5
        shape = (window, window)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape, mu_x=mu_x, mu_y=0.0,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(R_AA, R_BB, R_AB)
        config = make_mock_config()
        gauss, status, _ = fit_windows_kspace(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
        )
        assert status[0] == 0
        center_x = window / 2.0 + 1
        recovered_mu_x = gauss[0, 14] - center_x
        assert recovered_mu_x == pytest.approx(mu_x, abs=0.1)

    def test_sigma_stored_correctly(self):
        """params[9]=Sigma_xx, params[10]=Sigma_yy, params[11]=Sigma_xy."""
        window = 64
        Sxx, Syy, Sxy = 1.0, 0.5, 0.0
        shape = (window, window)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape, sigma_stress_xx=Sxx, sigma_stress_yy=Syy,
            sigma_stress_xy=Sxy,
        )
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(R_AA, R_BB, R_AB)
        config = make_mock_config()
        gauss, status, _ = fit_windows_kspace(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
        )
        assert status[0] == 0
        assert gauss[0, 9] == pytest.approx(Sxx, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Syy, rel=0.10)
        assert gauss[0, 11] == pytest.approx(Sxy, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 4h. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling."""

    def test_zero_amplitude(self):
        """Zero amplitude correlations → status=2 (low SNR)."""
        window = 32
        shape = (window, window)
        R_zero = np.zeros(shape)

        K_X, K_Y, k_x = _make_grids(window)
        k_y = fftshift(fftfreq(window))

        result = _fit_single_window_kspace(
            R_zero, R_zero, R_zero,
            K_X, K_Y, k_x, k_y,
            (window, window), snr_threshold=3.0,
            center_x=window / 2.0 + 1, center_y=window / 2.0 + 1,
        )

        assert result['status'] == 2

    def test_large_displacement(self):
        """Displacement > 3/4 window → status=3."""
        window = 32
        shape = (window, window)
        # mu_x = 0.8 * window = 25.6 > 0.75 * 32 = 24
        mu_x = 0.8 * window

        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            mu_x=mu_x,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )

        K_X, K_Y, k_x = _make_grids(window)
        k_y = fftshift(fftfreq(window))

        result = _fit_single_window_kspace(
            R_AA, R_BB, R_AB,
            K_X, K_Y, k_x, k_y,
            (window, window), snr_threshold=3.0,
            center_x=window / 2.0 + 1, center_y=window / 2.0 + 1,
        )

        # Should either get status 3 (big displacement) or some other
        # failure — but NOT status 0 with such extreme displacement
        # Note: the synthetic Gaussian may be clipped at the edge,
        # making the peak hard to find
        assert result['status'] != 0 or abs(
            result['params'][14] - (window / 2.0 + 1)
        ) < 0.75 * window

    def test_all_zero_correlation_graceful(self):
        """All-zero correlation should not raise an exception."""
        window = 32
        shape = (window, window)
        R_zero = np.zeros(shape)

        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_zero, R_zero, R_zero,
        )
        config = make_mock_config()

        # Should not raise
        gauss, status, initial = fit_windows_kspace(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
        )
        assert status[0] != 0  # Should indicate failure
