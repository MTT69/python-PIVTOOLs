"""Unit tests for the k-space AC CoC fitter.

The fitter recovers Σ_diff — the spatial-variance gap between the CoC plane
and the AC F_ref reference. Per the wiki algebra (sessions/2026-04-30):

    σ²_AB,frame = 2σ²_p + Σ_within
    σ²_AC      = 2·σ²_AB = 4σ²_p + 2Σ_within
    σ²_coc     = σ²_AC + Σ_disp = 4σ²_p + 2Σ_within + Σ_disp

So ``Σ_diff_recovered = σ²_coc − σ²_AC = Σ_disp``. Σ_disp = Σ_11_true +
Σ_22_true − 2·Σ_12_true is the dispersion (frame-to-frame) variance — it
doesn't include the within-frame variance Σ_within, which lives in σ²_AC and
cancels through the F_ref subtraction.

This file tests that contract directly: given AC and CoC planes built from
prescribed σ²_AC and Σ_disp, the fitter recovers Σ_disp via the curvature
LSQ. Reference impl: ``manual_tools/coc_kspace_vs_gaussian.py:267-301``
(``fit_kspace_quadratic``) and ``manual_tools/coc_vs_ab_autocorr_inspector.py``
(``_kspace_sigma_diff_one_window``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pivtools_cli.piv.piv_backend.stereo_ensemble_accumulator import (
    StereoEnsembleAccumulator,
)


# ---------------------------------------------------------------------------
# Synthetic plane construction
# ---------------------------------------------------------------------------

def _gaussian_plane(shape, cov, amplitude=1.0, dtype=np.float64):
    """Build a 2-D Gaussian plane with the peak at the geometric centre.

    ``cov`` is the variance tensor of the underlying Gaussian. The plane
    is fftshifted by construction (peak at (h//2, w//2)).
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    y = np.arange(h, dtype=np.float64) - cy
    x = np.arange(w, dtype=np.float64) - cx
    X, Y = np.meshgrid(x, y)

    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    inv_xx = cov[1, 1] / det
    inv_yy = cov[0, 0] / det
    inv_xy = -cov[0, 1] / det
    quad = inv_xx * X * X + inv_yy * Y * Y + 2.0 * inv_xy * X * Y
    return (amplitude * np.exp(-0.5 * quad)).astype(dtype)


def _ac_and_coc(shape, sigma_p, Sigma_within, Sigma_disp):
    """Build matched-camera AC reference and CoC plane.

    Per the wiki algebra:
        σ²_AC  = 4σ²_p I + 2·Σ_within
        σ²_coc = σ²_AC + Σ_disp
    """
    sp2 = sigma_p * sigma_p
    cov_AC = 4.0 * sp2 * np.eye(2) + 2.0 * Sigma_within
    cov_CoC = cov_AC + Sigma_disp
    coc = _gaussian_plane(shape, cov_CoC)
    ac1 = _gaussian_plane(shape, cov_AC)
    ac2 = _gaussian_plane(shape, cov_AC)  # matched cameras
    return coc, ac1, ac2


def _fit_one(coc, ac1, ac2, k_max=0.35):
    """Call _fit_coc_kspace_ac on a single window."""
    h, w = coc.shape
    coc_flat = coc.ravel()[None, :].astype(np.float64)
    ac1_flat = ac1.ravel()[None, :].astype(np.float64)
    ac2_flat = ac2.ravel()[None, :].astype(np.float64)
    mask = np.zeros(1, dtype=np.int32)
    return StereoEnsembleAccumulator._fit_coc_kspace_ac(
        # `self` is unused by the method body — pass any object.
        object(),
        coc_flat, ac1_flat, ac2_flat,
        h, w, 1, mask, k_max=k_max,
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Sigma_disp_xx, Sigma_disp_yy, Sigma_disp_xy",
    [
        (1.0, 1.0, 0.0),
        (2.0, 0.5, 0.0),
        (1.5, 1.5, 0.4),  # off-diagonal
    ],
    ids=["iso", "aniso_axes", "aniso_xy"],
)
def test_recovers_sigma_disp(Sigma_disp_xx, Sigma_disp_yy, Sigma_disp_xy):
    """Σ_diff_recovered = σ²_coc − σ²_AC = Σ_disp for any Σ_within.

    Σ_within drops out structurally — it's in both σ²_AC and σ²_coc with
    equal weight, so the F_ref subtraction cancels it.
    """
    window = 64
    sigma_p = 1.4

    Sigma_within = np.diag([0.7, 0.5])  # arbitrary; should not affect recovery
    Sigma_disp = np.array([
        [Sigma_disp_xx, Sigma_disp_xy],
        [Sigma_disp_xy, Sigma_disp_yy],
    ])

    coc, ac1, ac2 = _ac_and_coc((window, window), sigma_p, Sigma_within, Sigma_disp)
    fit = _fit_one(coc, ac1, ac2)

    assert int(fit["status"][0]) == 0, "fit should succeed on synthetic data"

    np.testing.assert_allclose(fit["sigma_diff_xx"][0], Sigma_disp_xx, rtol=0.03, atol=0.02)
    np.testing.assert_allclose(fit["sigma_diff_yy"][0], Sigma_disp_yy, rtol=0.03, atol=0.02)
    np.testing.assert_allclose(fit["sigma_diff_xy"][0], Sigma_disp_xy, atol=0.03)


def test_sigma_within_cancels():
    """Two runs with different Σ_within but identical Σ_disp must give the same fit."""
    window = 64
    sigma_p = 1.4
    Sigma_disp = np.diag([1.0, 1.0])

    Sigma_within_a = np.diag([0.0, 0.0])
    Sigma_within_b = np.diag([2.0, 2.0])

    coc_a, ac1_a, ac2_a = _ac_and_coc((window, window), sigma_p, Sigma_within_a, Sigma_disp)
    coc_b, ac1_b, ac2_b = _ac_and_coc((window, window), sigma_p, Sigma_within_b, Sigma_disp)

    fit_a = _fit_one(coc_a, ac1_a, ac2_a)
    fit_b = _fit_one(coc_b, ac1_b, ac2_b)

    np.testing.assert_allclose(fit_a["sigma_diff_xx"][0], fit_b["sigma_diff_xx"][0], rtol=0.02)
    np.testing.assert_allclose(fit_a["sigma_diff_yy"][0], fit_b["sigma_diff_yy"][0], rtol=0.02)


def test_sigma_12_round_trip():
    """Σ_12 = (Σ_11_fit + Σ_22_fit − Σ_diff)/2 round-trips to Σ_12_true + Σ_within.

    This is the production extraction. The wiki algebra says
    Σ_12_recovered = Σ_12_true + Σ_within, and that the +Σ_within bias
    cancels in R_zz = (A − Σ_12)/(2 sin²θ) because A also has +Σ_within.
    """
    window = 64
    sigma_p = 1.4

    # Truth values (the production fitter would never see these directly).
    Sigma_11_true = 1.5
    Sigma_22_true = 1.5
    Sigma_12_true = 0.6
    Sigma_within_scalar = 0.7

    # Σ_disp = Σ_11_true + Σ_22_true − 2·Σ_12_true (matched-camera limit).
    Sigma_disp_xx = Sigma_11_true + Sigma_22_true - 2.0 * Sigma_12_true

    Sigma_within = np.diag([Sigma_within_scalar, Sigma_within_scalar])
    Sigma_disp = np.diag([Sigma_disp_xx, Sigma_disp_xx])

    coc, ac1, ac2 = _ac_and_coc((window, window), sigma_p, Sigma_within, Sigma_disp)
    fit = _fit_one(coc, ac1, ac2)
    assert int(fit["status"][0]) == 0

    # Production-side: per-camera fit picks up Σ_within in addition to truth.
    Sigma_11_fit = Sigma_11_true + Sigma_within_scalar
    Sigma_22_fit = Sigma_22_true + Sigma_within_scalar

    Sigma_12_recovered = (
        Sigma_11_fit + Sigma_22_fit - fit["sigma_diff_xx"][0]
    ) / 2.0
    expected = Sigma_12_true + Sigma_within_scalar  # the documented +Σ_within shift
    np.testing.assert_allclose(Sigma_12_recovered, expected, rtol=0.03, atol=0.02)


def test_masked_window_returns_status_one():
    """A window with mask==1 must produce status=1, Σ_diff=NaN."""
    window = 64
    sigma_p = 1.4
    coc, ac1, ac2 = _ac_and_coc(
        (window, window), sigma_p,
        np.diag([0.5, 0.5]), np.diag([1.0, 1.0]),
    )
    coc_flat = coc.ravel()[None, :].astype(np.float64)
    ac1_flat = ac1.ravel()[None, :].astype(np.float64)
    ac2_flat = ac2.ravel()[None, :].astype(np.float64)
    mask = np.array([1], dtype=np.int32)

    fit = StereoEnsembleAccumulator._fit_coc_kspace_ac(
        object(), coc_flat, ac1_flat, ac2_flat,
        window, window, 1, mask, k_max=0.35,
    )
    assert int(fit["status"][0]) == 1
    assert np.isnan(fit["sigma_diff_xx"][0])


def test_zero_plane_fails_gracefully():
    """All-zero CoC + AC plane must produce status=1, no crash, no exception."""
    window = 32
    coc_flat = np.zeros((1, window * window), dtype=np.float32)
    ac1_flat = np.zeros((1, window * window), dtype=np.float32)
    ac2_flat = np.zeros((1, window * window), dtype=np.float32)
    mask = np.zeros(1, dtype=np.int32)

    fit = StereoEnsembleAccumulator._fit_coc_kspace_ac(
        object(),
        coc_flat, ac1_flat, ac2_flat,
        window, window, 1, mask, k_max=0.35,
    )
    assert int(fit["status"][0]) == 1


def test_batched_consistency():
    """Fitting many windows in one call matches per-window calls."""
    window = 64
    sigma_p = 1.4
    Sigma_within = np.diag([0.5, 0.5])
    n_windows = 5
    rng = np.random.default_rng(0)

    coc_planes = np.empty((n_windows, window * window), dtype=np.float64)
    ac1_planes = np.empty_like(coc_planes)
    ac2_planes = np.empty_like(coc_planes)
    truths = []
    for wi in range(n_windows):
        s = float(rng.uniform(0.5, 2.0))
        truths.append(s)
        Sigma_disp = np.diag([s, s])
        coc, ac1, ac2 = _ac_and_coc((window, window), sigma_p, Sigma_within, Sigma_disp)
        coc_planes[wi] = coc.ravel()
        ac1_planes[wi] = ac1.ravel()
        ac2_planes[wi] = ac2.ravel()

    mask = np.zeros(n_windows, dtype=np.int32)
    fit_batch = StereoEnsembleAccumulator._fit_coc_kspace_ac(
        object(), coc_planes, ac1_planes, ac2_planes,
        window, window, n_windows, mask, k_max=0.35,
    )
    for wi, truth in enumerate(truths):
        assert int(fit_batch["status"][wi]) == 0, f"window {wi} failed"
        np.testing.assert_allclose(
            fit_batch["sigma_diff_xx"][wi], truth, rtol=0.03, atol=0.03
        )
