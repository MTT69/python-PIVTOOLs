"""
Tests for kspace_lm_fitting.py -- the batched-LM k-space fitter (GSL replica minus
beta) that is the production ensemble fitter since 2026-07-08.

Driven exactly as the production call site does (``single_pass_accumulator``):
positional args, use_soft_weighting=True, k_max_cap=None. Gaussian displacement PDF
only -- there is no shape option (kurtosis was tested and rejected).

Mirrors test_kspace_fitting.py (which still guards the dormant closed-form module):
  - Output contract + masking
  - Displacement recovery (complex phase)
  - Stress recovery (isotropic / anisotropic / shear)
  - Ghost-stress rejection + edge cases
plus an LM-specific convergence-health check via the diagnostics return.
"""

import numpy as np
import pytest

from synthetic_correlations import (
    generate_correlation_triplet,
    flatten_for_kspace,
)
from pivtools_cli.piv.piv_backend.kspace_lm_fitting import (
    fit_windows_kspace_lm,
)


# Relative height of the white-noise pedestal injected into the auto planes.
_PEDESTAL = 0.05


def _fit(R_AA, R_BB, R_AB, n_windows=1, **kw):
    """Run the LM fitter with the production call-site argument pattern."""
    R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
        R_AA, R_BB, R_AB, n_windows=n_windows
    )
    return fit_windows_kspace_lm(
        R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
        True,          # use_soft_weighting (production value)
        False,         # debug
        None,          # predictor_displacements
        "bicubic",     # interp_kernel
        None,          # k_max_cap
        **kw,
    )


def _triplet(shape, **kw):
    """Synthetic triplet with a PHYSICALLY-correct white noise floor.

    White camera noise adds a +2*sigma_n^2 delta at the AUTOcorrelation centre
    pixel (flat pedestal in k-space — what the stage-1 floor models) and cancels
    in the cross plane (A/B noise independence). The generator's ``offset``
    kwarg is NOT that: a plane-wide constant is a DC-only delta in k-space, and
    fitting a white-floor model to it over-subtracts at every k>0 (the linear
    fitter's refc margin happens to fence that mismatch off; the LM fitter's
    soft weighting does not).
    """
    R_AA, R_BB, R_AB = generate_correlation_triplet(shape, **kw)
    cy, cx = shape[0] // 2, shape[1] // 2
    spike = _PEDESTAL * R_AA.max()
    R_AA[cy, cx] += spike
    R_BB[cy, cx] += spike
    return R_AA, R_BB, R_AB


# ─────────────────────────────────────────────────────────────────────────────
# Output contract + masking
# ─────────────────────────────────────────────────────────────────────────────

class TestContract:
    def test_output_shapes(self):
        """gauss_flat (n,16), status (n,), initial (n,16)."""
        n = 3
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        gauss, status, initial = _fit(R_AA, R_BB, R_AB, n)
        assert gauss.shape == (n, 16)
        assert status.shape == (n,)
        assert initial.shape == (n, 16)

    def test_masked_windows(self):
        """Masked windows keep status=-1; unmasked ones succeed."""
        n = 4
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n_windows=n
        )
        mask[1] = True
        mask[3] = True
        gauss, status, _ = fit_windows_kspace_lm(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            True, False, None, "bicubic", None,
        )
        assert status[1] == -1 and status[3] == -1
        assert status[0] == 0 and status[2] == 0

    def test_all_masked(self):
        """All-masked → all status=-1, no exception."""
        n = 2
        R_AA, R_BB, R_AB = _triplet((32, 32))
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n_windows=n
        )
        mask[:] = True
        _, status, _ = fit_windows_kspace_lm(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            True, False, None, "bicubic", None,
        )
        assert np.all(status == -1)

    def test_convergence_health(self):
        """On clean synthetic planes the batched LM must fully converge."""
        n = 8
        R_AA, R_BB, R_AB = _triplet(
            (32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5, mu_x=1.0
        )
        gauss, status, _, diag = _fit(R_AA, R_BB, R_AB, n, return_diagnostics=True)
        assert np.all(status == 0)
        assert diag["s1_conv"].sum() >= 0.99 * n
        assert diag["s2_conv"].sum() >= 0.99 * n
        # nowhere near the iteration caps on clean data
        assert np.median(diag["s2_iter"]) < 50

    def test_particle_size_slots_nan(self):
        """Slots 6:9 are NaN by contract (k-space cancels particle shape)."""
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert np.all(np.isnan(gauss[0, 6:9]))


# ─────────────────────────────────────────────────────────────────────────────
# Displacement recovery (complex phase of T)
# ─────────────────────────────────────────────────────────────────────────────

class TestDisplacement:
    @pytest.mark.parametrize("mu", [(0.0, 0.0), (1.5, 0.0), (0.7, -1.2)],
                             ids=["static", "x_only", "diag"])
    def test_displacement(self, mu):
        mu_x, mu_y = mu
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window), mu_x=mu_x, mu_y=mu_y,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] == 0
        center = window / 2.0 + 1
        assert gauss[0, 14] - center == pytest.approx(mu_x, abs=0.1)
        assert gauss[0, 15] - center == pytest.approx(mu_y, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Stress recovery (Sigma at slots 9/10/11)
# ─────────────────────────────────────────────────────────────────────────────

class TestStressRecovery:
    @pytest.mark.parametrize("Sigma", [0.5, 1.0, 3.0],
                             ids=["small", "medium", "large"])
    @pytest.mark.parametrize("window", [32, 64])
    def test_isotropic(self, Sigma, window):
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=Sigma, sigma_stress_yy=Sigma,
            mu_x=1.0, mu_y=0.0,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] == 0, f"status={status[0]}"
        assert gauss[0, 9] == pytest.approx(Sigma, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Sigma, rel=0.10)

    @pytest.mark.parametrize("stress", [
        (1.0, 0.3, 0.0),
        (2.0, 2.0, 0.5),
        (0.8, 0.8, 0.3),
    ], ids=["aniso_no_shear", "large_with_shear", "iso_shear"])
    def test_anisotropic(self, stress):
        Sxx, Syy, Sxy = stress
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=Sxx, sigma_stress_yy=Syy, sigma_stress_xy=Sxy,
            mu_x=1.0, mu_y=-0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] == 0, f"status={status[0]}"
        assert gauss[0, 9] == pytest.approx(Sxx, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Syy, rel=0.10)
        assert gauss[0, 11] == pytest.approx(Sxy, abs=0.15)

    def test_ghost_stress(self):
        """Pure displacement (zero stress) must not invent stress.

        Zero true stress may return status=0 with ~zero Sigma or status=5 --
        both are correct rejection (the LM projects Sigma onto >= 0).
        """
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=0.0, sigma_stress_yy=0.0, sigma_stress_xy=0.0,
            mu_x=1.0, mu_y=-0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] in (0, 5), f"status={status[0]}"
        if status[0] == 0:
            assert gauss[0, 9] < 0.05
            assert gauss[0, 10] < 0.05
            assert abs(gauss[0, 11]) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_correlation_graceful(self):
        """All-zero correlation must not raise and must not report success."""
        window = 32
        R0 = np.zeros((window, window))
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(R0, R0, R0)
        gauss, status, _ = fit_windows_kspace_lm(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            True, False, None, "bicubic", None,
        )
        assert status[0] != 0

    def test_large_displacement(self):
        """Displacement > 3/4 window → non-success or flagged in-range."""
        window = 32
        R_AA, R_BB, R_AB = _triplet(
            (window, window), mu_x=0.8 * window,
            sigma_stress_xx=0.5, sigma_stress_yy=0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] != 0 or abs(gauss[0, 14] - (window / 2.0 + 1)) < 0.75 * window

    def test_nan_predictor_handled(self):
        """NaN predictor displacements (invalid windows) must not crash P_noise."""
        n = 2
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
            R_AA, R_BB, R_AB, n_windows=n
        )
        pred = np.array([[0.3, 1.2], [np.nan, np.nan]])
        gauss, status, _ = fit_windows_kspace_lm(
            R_AA_f, R_BB_f, R_AB_f, mask, cs, None, 0,
            True, False, pred, "bicubic", None,
        )
        assert status.shape == (n,)
        assert np.all(np.isfinite(gauss[status == 0][:, 9:12]))
