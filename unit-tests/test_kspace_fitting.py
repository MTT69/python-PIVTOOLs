"""
Tests for kspace_linear_fitting.py -- the closed-form, GSL-free k-space fitter
(``fit_windows_kspace_linear``) that replaced the C/GSL two-stage fitter.

Driven with the PRODUCTION recipe (floor_mode="joint", weight_mode="refc") and
parametrized over both shape models exposed by ensemble_piv.kspace_kurtosis:
  - "gauss"               (kspace_kurtosis: false)
  - "kurtosis_decoupled"  (kspace_kurtosis: true)

The synthetic correlations are pure-Gaussian PDFs, so the decoupled-kurtosis
k^4 columns should fit ~0 and recover the same Sigma as the Gaussian model.

Tests are organised around the public API:
  - Output contract + masking
  - Displacement recovery (phase slope)
  - Stress recovery (isotropic / anisotropic / shear)
  - Ghost-stress rejection + edge cases
"""

import numpy as np
import pytest

from synthetic_correlations import (
    generate_correlation_triplet,
    flatten_for_kspace,
)
from pivtools_cli.piv.piv_backend.kspace_linear_fitting import (
    fit_windows_kspace_linear,
)


# Both shape models the production toggle selects between.
SHAPE_MODES = ["gauss", "kurtosis_decoupled"]

# A small, realistic noise floor so the joint floor has something to estimate
# (real ensemble planes always carry a white pedestal under the signal).
_OFFSET = 0.01


def _fit(R_AA, R_BB, R_AB, n_windows=1, *, shape_mode="gauss"):
    """Run the production-configured linear fitter on tiled synthetic planes."""
    R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
        R_AA, R_BB, R_AB, n_windows=n_windows
    )
    gauss, status, initial = fit_windows_kspace_linear(
        R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
        floor_mode="joint", weight_mode="refc", shape_mode=shape_mode,
    )
    return gauss, status, initial


def _triplet(shape, **kw):
    """Synthetic triplet with a small white noise floor baked in."""
    kw.setdefault("offset_A", _OFFSET)
    kw.setdefault("offset_B", _OFFSET)
    kw.setdefault("offset_AB", _OFFSET)
    return generate_correlation_triplet(shape, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Output contract + masking
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:
    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    def test_output_shapes(self, shape_mode):
        """gauss_flat (n,16), status (n,), initial (n,16)."""
        n = 3
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        gauss, status, initial = _fit(R_AA, R_BB, R_AB, n, shape_mode=shape_mode)
        assert gauss.shape == (n, 16)
        assert status.shape == (n,)
        assert initial.shape == (n, 16)

    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    def test_masked_windows(self, shape_mode):
        """Masked windows keep status=-1; unmasked ones succeed."""
        n = 4
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n_windows=n
        )
        mask[1] = True
        mask[3] = True
        gauss, status, _ = fit_windows_kspace_linear(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            floor_mode="joint", weight_mode="refc", shape_mode=shape_mode,
        )
        assert status[1] == -1 and status[3] == -1
        assert status[0] == 0 and status[2] == 0

    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    def test_all_masked(self, shape_mode):
        """All-masked → all status=-1, no exception."""
        n = 2
        R_AA, R_BB, R_AB = _triplet((32, 32))
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n_windows=n
        )
        mask[:] = True
        _, status, _ = fit_windows_kspace_linear(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            floor_mode="joint", weight_mode="refc", shape_mode=shape_mode,
        )
        assert np.all(status == -1)


# ─────────────────────────────────────────────────────────────────────────────
# Displacement recovery (phase slope of T)
# ─────────────────────────────────────────────────────────────────────────────

class TestDisplacement:
    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    @pytest.mark.parametrize("mu", [(0.0, 0.0), (1.5, 0.0), (0.7, -1.2)],
                             ids=["static", "x_only", "diag"])
    def test_displacement(self, shape_mode, mu):
        mu_x, mu_y = mu
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window), mu_x=mu_x, mu_y=mu_y,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB, shape_mode=shape_mode)
        assert status[0] == 0
        center = window / 2.0 + 1
        assert gauss[0, 14] - center == pytest.approx(mu_x, abs=0.1)
        assert gauss[0, 15] - center == pytest.approx(mu_y, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Stress recovery (Sigma = coeffs[1:4], both shape models)
# ─────────────────────────────────────────────────────────────────────────────

class TestStressRecovery:
    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    @pytest.mark.parametrize("Sigma", [0.5, 1.0, 3.0],
                             ids=["small", "medium", "large"])
    @pytest.mark.parametrize("window", [32, 64])
    def test_isotropic(self, shape_mode, Sigma, window):
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=Sigma, sigma_stress_yy=Sigma,
            mu_x=1.0, mu_y=0.0,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB, shape_mode=shape_mode)
        assert status[0] == 0, f"status={status[0]}"
        assert gauss[0, 9] == pytest.approx(Sigma, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Sigma, rel=0.10)

    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    @pytest.mark.parametrize("stress", [
        (1.0, 0.3, 0.0),
        (2.0, 2.0, 0.5),
        (0.8, 0.8, 0.3),
    ], ids=["aniso_no_shear", "large_with_shear", "iso_shear"])
    def test_anisotropic(self, shape_mode, stress):
        Sxx, Syy, Sxy = stress
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=Sxx, sigma_stress_yy=Syy, sigma_stress_xy=Sxy,
            mu_x=1.0, mu_y=-0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB, shape_mode=shape_mode)
        assert status[0] == 0, f"status={status[0]}"
        assert gauss[0, 9] == pytest.approx(Sxx, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Syy, rel=0.10)
        assert gauss[0, 11] == pytest.approx(Sxy, abs=0.15)

    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    def test_ghost_stress(self, shape_mode):
        """Pure displacement (zero stress) must not invent stress.

        Zero true stress may return status=0 with ~zero Sigma or status=5
        (variance fitted slightly negative) -- both are correct rejection.
        """
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=0.0, sigma_stress_yy=0.0, sigma_stress_xy=0.0,
            mu_x=1.0, mu_y=-0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB, shape_mode=shape_mode)
        assert status[0] in (0, 5), f"status={status[0]}"
        if status[0] == 0:
            assert gauss[0, 9] < 0.05
            assert gauss[0, 10] < 0.05
            assert abs(gauss[0, 11]) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    def test_zero_correlation_graceful(self, shape_mode):
        """All-zero correlation must not raise and must not report success."""
        window = 32
        R0 = np.zeros((window, window))
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(R0, R0, R0)
        gauss, status, _ = fit_windows_kspace_linear(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            floor_mode="joint", weight_mode="refc", shape_mode=shape_mode,
        )
        assert status[0] != 0

    @pytest.mark.parametrize("shape_mode", SHAPE_MODES)
    def test_large_displacement(self, shape_mode):
        """Displacement > 3/4 window → non-success or flagged in-range."""
        window = 32
        R_AA, R_BB, R_AB = _triplet(
            (window, window), mu_x=0.8 * window,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB, shape_mode=shape_mode)
        assert status[0] != 0 or abs(gauss[0, 14] - (window / 2.0 + 1)) < 0.75 * window
