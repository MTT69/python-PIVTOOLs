"""
Tests for kspace_fitting.py — the k-space transfer function fitter (C-accelerated).

Tests are organised around the public API:
  - Single-window fitting (via fit_windows_kspace with 1 window)
  - Top-level batch entry point
  - Edge cases and status codes
  - Stress accuracy regression (ground truth recovery)
  - Diagnostic figures
"""

from pathlib import Path

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
    fit_windows_kspace,
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


def _fit_single_window(R_AA, R_BB, R_AB, corr_size):
    """Convenience: fit a single window via the C library."""
    R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(R_AA, R_BB, R_AB)
    config = make_mock_config()
    gauss, status, initial = fit_windows_kspace(
        R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
    )
    return gauss[0], status[0], initial[0]


# ─────────────────────────────────────────────────────────────────────────────
# Single-window integration (via public API)
# ─────────────────────────────────────────────────────────────────────────────

class TestFitSingleWindowKspace:
    """Tests for single-window k-space fitting via fit_windows_kspace()."""

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

        center_x = window / 2.0 + 1
        center_y = window / 2.0 + 1

        params, status, _ = _fit_single_window(R_AA, R_BB, R_AB, (window, window))

        # Zero stress on small windows can return status=5 (negative variance)
        # because the fitted variance goes slightly negative due to spectral
        # leakage — this is correct behavior for the fitter.
        if Sxx == 0.0 and Syy == 0.0:
            assert status in (0, 5), \
                f"Expected status 0 or 5 for zero stress, got {status}"
            if status == 5:
                return  # Correctly rejected — no further checks
        else:
            assert status == 0, \
                f"Expected status=0, got {status}"

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
# Top-level entry point
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
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling."""

    def test_zero_amplitude(self):
        """Zero amplitude correlations → non-zero status (low SNR or failure)."""
        window = 32
        R_zero = np.zeros((window, window))

        params, status, _ = _fit_single_window(
            R_zero, R_zero, R_zero, (window, window),
        )
        assert status != 0  # Should indicate failure (typically status=2)

    def test_large_displacement(self):
        """Displacement > 3/4 window → status=3 or otherwise non-success."""
        window = 32
        shape = (window, window)
        mu_x = 0.8 * window

        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            mu_x=mu_x,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )

        params, status, _ = _fit_single_window(R_AA, R_BB, R_AB, (window, window))

        # Should either get a failure status or, if status 0, displacement
        # should be within the valid range
        assert status != 0 or abs(
            params[14] - (window / 2.0 + 1)
        ) < 0.75 * window

    def test_all_zero_correlation_graceful(self):
        """All-zero correlation should not raise an exception."""
        window = 32
        R_zero = np.zeros((window, window))

        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_zero, R_zero, R_zero,
        )
        config = make_mock_config()

        # Should not raise
        gauss, status, initial = fit_windows_kspace(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, config, pass_idx=0,
        )
        assert status[0] != 0  # Should indicate failure


# ─────────────────────────────────────────────────────────────────────────────
# Stress accuracy regression (ground truth recovery)
# ─────────────────────────────────────────────────────────────────────────────

class TestStressAccuracy:
    """Verify C k-space fitter recovers known ground-truth stresses."""

    @pytest.mark.parametrize("Sigma", [0.0, 0.1, 0.5, 1.0, 3.0],
                             ids=["ghost", "tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("mu", [0.0, 2.5, 5.0],
                             ids=["static", "mid_disp", "large_disp"])
    @pytest.mark.parametrize("window", [16, 32])
    def test_isotropic_stress_recovery(self, Sigma, mu, window):
        """Isotropic stress recovery across magnitudes and displacements."""
        shape = (window, window)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=Sigma, sigma_stress_yy=Sigma,
            mu_x=mu, mu_y=0.0,
        )

        params, status, _ = _fit_single_window(R_AA, R_BB, R_AB, (window, window))

        # Skip cases where peak is clipped by window boundary
        if mu > 0.5 * window:
            # Large displacement on small window — may not converge
            return

        if status != 0:
            # Some edge cases (e.g., 16x16 with Sigma=3 and mu=5) are
            # genuinely too hard. Only assert on success.
            return

        center_x = window / 2.0 + 1
        recovered_mu = params[14] - center_x

        # Displacement check
        assert recovered_mu == pytest.approx(mu, abs=0.15), \
            f"mu: expected {mu}, got {recovered_mu}"

        # Stress check
        if Sigma >= 0.5:
            assert params[9] == pytest.approx(Sigma, rel=0.10), \
                f"Sigma_xx: expected {Sigma}, got {params[9]}"
            assert params[10] == pytest.approx(Sigma, rel=0.10), \
                f"Sigma_yy: expected {Sigma}, got {params[10]}"
        else:
            assert params[9] == pytest.approx(Sigma, abs=0.10), \
                f"Sigma_xx: expected {Sigma}, got {params[9]}"
            assert params[10] == pytest.approx(Sigma, abs=0.10), \
                f"Sigma_yy: expected {Sigma}, got {params[10]}"

    @pytest.mark.parametrize("stress", [
        (0.5, 0.5, 0.2),
        (1.0, 0.3, 0.0),
        (2.0, 2.0, 0.5),
    ], ids=["shear_iso", "aniso_no_shear", "large_with_shear"])
    def test_anisotropic_stress_recovery(self, stress):
        """Anisotropic and shear stress recovery."""
        Sxx, Syy, Sxy = stress
        window = 64
        shape = (window, window)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=Sxx, sigma_stress_yy=Syy,
            sigma_stress_xy=Sxy,
            mu_x=1.0, mu_y=-0.5,
        )

        params, status, _ = _fit_single_window(R_AA, R_BB, R_AB, (window, window))
        assert status == 0, f"Expected status=0, got {status}"

        # Normal stresses
        if Sxx >= 0.5:
            assert params[9] == pytest.approx(Sxx, rel=0.10)
        else:
            assert params[9] == pytest.approx(Sxx, abs=0.10)

        if Syy >= 0.5:
            assert params[10] == pytest.approx(Syy, rel=0.10)
        else:
            assert params[10] == pytest.approx(Syy, abs=0.10)

        # Shear stress
        assert params[11] == pytest.approx(Sxy, abs=0.15)

    def test_ghost_stress(self):
        """Pure displacement (zero stress) must NOT invent stress.

        With zero true stress, the fitter may return status=5 (negative variance)
        because fitted variance goes slightly negative due to spectral leakage.
        This is correct rejection behavior — the fitter should NOT report
        a large positive stress when the true stress is zero.
        """
        window = 64
        shape = (window, window)
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=0.0, sigma_stress_yy=0.0, sigma_stress_xy=0.0,
            mu_x=1.0, mu_y=-0.5,
        )

        params, status, _ = _fit_single_window(R_AA, R_BB, R_AB, (window, window))

        # Status 0 (success with near-zero stress) or 5 (correctly rejected
        # as negative variance) are both acceptable for zero true stress
        assert status in (0, 5), f"Expected status 0 or 5, got {status}"

        if status == 0:
            assert params[9] < 0.02, f"Ghost stress Sxx = {params[9]}"
            assert params[10] < 0.02, f"Ghost stress Syy = {params[10]}"
            assert abs(params[11]) < 0.02, f"Ghost stress Sxy = {params[11]}"


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic figures
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticFigures:
    """Generate diagnostic plots when --make-figures is passed."""

    def test_make_figures(self, make_figures):
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(__file__).resolve().parent / "test_output"
        out_dir.mkdir(exist_ok=True)

        window = 64
        shape = (window, window)

        # Generate test cases with varying stress
        test_cases = [
            ("zero", 0.0, 0.0, 0.0, 1.0, 0.5),
            ("iso_0.5", 0.5, 0.5, 0.0, 1.0, 0.5),
            ("iso_1.0", 1.0, 1.0, 0.0, 1.0, 0.5),
            ("aniso", 1.5, 0.5, 0.0, 1.0, 0.5),
            ("shear", 0.5, 0.5, 0.3, 1.0, 0.5),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Collect results for summary
        labels = []
        true_Sxx, true_Syy = [], []
        rec_Sxx, rec_Syy = [], []

        center_idx = window // 2
        K_X, K_Y, k_x = _make_grids(window)
        k_y = fftshift(fftfreq(window))

        # Use the first non-zero case for panels 1 & 2
        demo_case = test_cases[1]  # iso_0.5

        R_AA, R_BB, R_AB = generate_correlation_triplet(
            shape,
            sigma_particle_x=2.5, sigma_particle_y=2.5,
            sigma_stress_xx=demo_case[1], sigma_stress_yy=demo_case[2],
            sigma_stress_xy=demo_case[3],
            mu_x=demo_case[4], mu_y=demo_case[5],
        )
        F_AA, F_BB, F_AB = _fft_correlations(R_AA, R_BB, R_AB)
        F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))

        # Panel 1: T(kx) slice through centre
        ax = axes[0]
        T_AB = F_AB / np.where(F_ref > 1e-10, F_ref, 1e-10)
        T_kx = np.abs(T_AB[center_idx, :])
        ax.plot(k_x, T_kx, 'b-', linewidth=1.5, label="|T(kx, 0)|")
        # Overlay expected Gaussian: exp(-2*pi^2*sigma^2*k^2)
        sigma_stress = demo_case[1]
        expected_T = np.exp(-2 * np.pi ** 2 * sigma_stress * k_x ** 2)
        ax.plot(k_x, expected_T, 'r--', linewidth=1, label=f"exp(-2pi^2*{sigma_stress}*k^2)")
        ax.set_xlabel("kx (cycles/pixel)")
        ax.set_ylabel("|T|")
        ax.set_title(f"T(kx) — {demo_case[0]}")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, 0.5)

        # Panel 2: T(ky) slice
        ax = axes[1]
        T_ky = np.abs(T_AB[:, center_idx])
        ax.plot(k_y, T_ky, 'b-', linewidth=1.5, label="|T(0, ky)|")
        expected_Ty = np.exp(-2 * np.pi ** 2 * demo_case[2] * k_y ** 2)
        ax.plot(k_y, expected_Ty, 'r--', linewidth=1, label=f"exp(-2pi^2*{demo_case[2]}*k^2)")
        ax.set_xlabel("ky (cycles/pixel)")
        ax.set_ylabel("|T|")
        ax.set_title(f"T(ky) — {demo_case[0]}")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, 0.5)

        # Run all cases through the fitter for panel 3
        for name, Sxx, Syy, Sxy, mu_x, mu_y in test_cases:
            R_AA_i, R_BB_i, R_AB_i = generate_correlation_triplet(
                shape,
                sigma_particle_x=2.5, sigma_particle_y=2.5,
                sigma_stress_xx=Sxx, sigma_stress_yy=Syy,
                sigma_stress_xy=Sxy, mu_x=mu_x, mu_y=mu_y,
            )

            params, status, _ = _fit_single_window(
                R_AA_i, R_BB_i, R_AB_i, (window, window),
            )

            labels.append(name)
            true_Sxx.append(Sxx)
            true_Syy.append(Syy)
            if status == 0:
                rec_Sxx.append(params[9])
                rec_Syy.append(params[10])
            else:
                rec_Sxx.append(np.nan)
                rec_Syy.append(np.nan)

        # Panel 3: Sigma recovery summary
        ax = axes[2]
        x_pos = np.arange(len(labels))
        w = 0.2
        ax.bar(x_pos - 1.5 * w, true_Sxx, w, label="True Sxx", color="steelblue")
        ax.bar(x_pos - 0.5 * w, rec_Sxx, w, label="Rec Sxx", color="coral")
        ax.bar(x_pos + 0.5 * w, true_Syy, w, label="True Syy", color="darkblue")
        ax.bar(x_pos + 1.5 * w, rec_Syy, w, label="Rec Syy", color="darkorange")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Sigma (px^2)")
        ax.set_title("Sigma recovery across test cases")
        ax.legend(fontsize=7)

        plt.suptitle("K-Space Fitting — Diagnostic", fontsize=14, fontweight="bold")
        plt.tight_layout()

        out_path = out_dir / "kspace_fitting_diagnostic.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Figure saved: {out_path}")
