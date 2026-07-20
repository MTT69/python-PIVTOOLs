"""
Tests for kspace_lm_fitting.py -- the batched-LM k-space fitter. Since 2026-07-14
this is the one-stage joint fit (mu, Sigma, gain g, in-model noise floor N0) of
the raw transfer ratio; the two-stage GSL-replica design it replaces lives in
git history. Since 2026-07-17 ``ensemble_piv.kspace_shape`` optionally appends
free-signed quartic exponent terms (kx4 / ky4 / kx4+ky4) that absorb
displacement-PDF kurtosis; the default ``gaussian`` path is unchanged.

Driven exactly as the production call site does (``single_pass_accumulator``):
positional args, with a mock config supplying ``ensemble_kspace_shape``.

Mirrors test_kspace_fitting.py (which still guards the dormant closed-form module):
  - Output contract + masking
  - Displacement recovery (complex phase)
  - Stress recovery (isotropic / anisotropic / shear)
  - Ghost-stress rejection + edge cases
plus an LM-specific convergence-health check via the diagnostics return and the
quartic shape-mode suite (null self-extinction, biased-Gaussian recovery, and a
sentinel guarding that the gaussian shape still runs the untouched v6 residual).
"""

import numpy as np
import pytest
from synthetic_correlations import (
    flatten_for_kspace,
    generate_correlation_triplet,
    generate_quartic_triplet,
    make_mock_config,
)

from pivtools_cli.piv.piv_backend.kspace_lm_fitting import (
    fit_windows_kspace_lm,
)

# Relative height of the white-noise pedestal injected into the auto planes.
_PEDESTAL = 0.05

ALL_SHAPES = ("gaussian", "kx4", "ky4", "kx4+ky4")


def _cfg(shape="gaussian"):
    return make_mock_config(ensemble_kspace_shape=shape)


def _fit(R_AA, R_BB, R_AB, n_windows=1, shape="gaussian", **kw):
    """Run the LM fitter with the production call-site argument pattern."""
    R_AA_f, R_BB_f, R_AB_f, mask, cs = flatten_for_kspace(
        R_AA, R_BB, R_AB, n_windows=n_windows
    )
    return fit_windows_kspace_lm(
        R_AA_f,
        R_BB_f,
        R_AB_f,
        mask,
        cs,
        _cfg(shape),
        0,
        False,  # debug
        **kw,
    )


def _triplet(shape, **kw):
    """Synthetic triplet with a PHYSICALLY-correct white noise floor.

    White camera noise adds a +2*sigma_n^2 delta at the AUTOcorrelation centre
    pixel (flat pedestal in k-space — what the joint fit's in-model floor N0
    models) and cancels in the cross plane (A/B noise independence). The
    generator's ``offset`` kwarg is NOT that: a plane-wide constant is a DC-only
    delta in k-space, unrelated to a white floor.
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
            R_AA_f,
            R_BB_f,
            R_AB_f,
            mask,
            cs,
            _cfg(),
            0,
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
            R_AA_f,
            R_BB_f,
            R_AB_f,
            mask,
            cs,
            _cfg(),
            0,
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
        assert diag["conv"].sum() >= 0.99 * n
        # nowhere near the iteration cap on clean data
        assert np.median(diag["iter"]) < 50

    def test_particle_size_slots_nan(self):
        """Slots 6:9 are NaN by contract (k-space cancels particle shape)."""
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert np.all(np.isnan(gauss[0, 6:9]))


# ─────────────────────────────────────────────────────────────────────────────
# Displacement recovery (complex phase of T)
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplacement:
    @pytest.mark.parametrize(
        "mu", [(0.0, 0.0), (1.5, 0.0), (0.7, -1.2)], ids=["static", "x_only", "diag"]
    )
    def test_displacement(self, mu):
        mu_x, mu_y = mu
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            mu_x=mu_x,
            mu_y=mu_y,
            sigma_stress_xx=0.5,
            sigma_stress_yy=0.5,
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
    @pytest.mark.parametrize("Sigma", [0.5, 1.0, 3.0], ids=["small", "medium", "large"])
    @pytest.mark.parametrize("window", [32, 64])
    def test_isotropic(self, Sigma, window):
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=Sigma,
            sigma_stress_yy=Sigma,
            mu_x=1.0,
            mu_y=0.0,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] == 0, f"status={status[0]}"
        assert gauss[0, 9] == pytest.approx(Sigma, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Sigma, rel=0.10)

    @pytest.mark.parametrize(
        "stress",
        [
            (1.0, 0.3, 0.0),
            (2.0, 2.0, 0.5),
            (0.8, 0.8, 0.3),
        ],
        ids=["aniso_no_shear", "large_with_shear", "iso_shear"],
    )
    def test_anisotropic(self, stress):
        Sxx, Syy, Sxy = stress
        window = 64
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            sigma_stress_xx=Sxx,
            sigma_stress_yy=Syy,
            sigma_stress_xy=Sxy,
            mu_x=1.0,
            mu_y=-0.5,
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
            sigma_stress_xx=0.0,
            sigma_stress_yy=0.0,
            sigma_stress_xy=0.0,
            mu_x=1.0,
            mu_y=-0.5,
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
            R_AA_f,
            R_BB_f,
            R_AB_f,
            mask,
            cs,
            _cfg(),
            0,
        )
        assert status[0] != 0

    def test_large_displacement(self):
        """Displacement > 3/4 window → non-success or flagged in-range."""
        window = 32
        R_AA, R_BB, R_AB = _triplet(
            (window, window),
            mu_x=0.8 * window,
            sigma_stress_xx=0.5,
            sigma_stress_yy=0.5,
        )
        gauss, status, _ = _fit(R_AA, R_BB, R_AB)
        assert status[0] != 0 or abs(gauss[0, 14] - (window / 2.0 + 1)) < 0.75 * window


# ─────────────────────────────────────────────────────────────────────────────
# Quartic shape modes (ensemble_piv.kspace_shape, 2026-07-17)
# ─────────────────────────────────────────────────────────────────────────────

# Quartic ground truth for the recovery tests: sigma_particle=1.0 keeps the
# F_ref weights alive out to |k| ~ 0.35 cycles/px so the k^4 curvature is
# actually in-band. b4=30 <=> kappa_4 = -24*30/(2pi)^4 = -0.462 px^4, i.e.
# excess kurtosis gamma_2 ~ -0.46 at Sigma=1 px^2 — the near-wall channel level.
_B4_TRUE = 30.0


class TestShapeModes:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_contract_and_diag_keys(self, shape):
        """Every shape keeps the 16-col contract; b4 diag keys appear iff enabled."""
        n = 3
        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        gauss, status, initial, diag = _fit(
            R_AA, R_BB, R_AB, n, shape=shape, return_diagnostics=True
        )
        assert gauss.shape == (n, 16)
        assert status.shape == (n,)
        assert initial.shape == (n, 16)
        assert np.all(status == 0)
        assert ("b4x" in diag) == ("kx4" in shape)
        assert ("b4y" in diag) == ("ky4" in shape)

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_null_on_gaussian_data(self, shape):
        """On Gaussian planes every shape recovers Sigma and b4 -> ~0.

        The self-extinction property: the quartic term must not invent
        curvature where the data has none, so all four modes agree. A small
        white plane noise keeps the LM cost off the float floor — perfectly
        noiseless data trips the solver's no-progress exit at cost ~1e-34
        (see generate_quartic_triplet docstring); real planes always carry a
        pair-count/sensor floor.
        """
        Sxx, Syy = 1.0, 0.5
        R_AA, R_BB, R_AB = _triplet(
            (64, 64),
            sigma_stress_xx=Sxx,
            sigma_stress_yy=Syy,
            mu_x=1.0,
            noise_std=1e-4,
        )
        gauss, status, _, diag = _fit(
            R_AA, R_BB, R_AB, shape=shape, return_diagnostics=True
        )
        assert status[0] == 0
        assert gauss[0, 9] == pytest.approx(Sxx, rel=0.10)
        assert gauss[0, 10] == pytest.approx(Syy, rel=0.10)
        # The term absorbs a sliver of the injected plane noise, so "zero"
        # means an order below the recovery-test signal (_B4_TRUE = 30):
        # |b4| = 5 is gamma_2 ~ 0.08 at Sigma = 1 px^2 — negligible curvature,
        # and the Sigma asserts above are the real null criterion.
        if "b4x" in diag:
            assert abs(diag["b4x"][0]) < 5.0
        if "b4y" in diag:
            assert abs(diag["b4y"][0]) < 5.0

    def test_kx4_recovers_biased_gaussian(self):
        """Sub-Gaussian x-displacement PDF: gaussian mode over-reads Sxx, kx4 fixes it."""
        Sxx, Syy = 1.0, 0.5
        R_AA, R_BB, R_AB = generate_quartic_triplet(
            (64, 64),
            sigma_particle_x=1.0,
            sigma_particle_y=1.0,
            sigma_stress_xx=Sxx,
            sigma_stress_yy=Syy,
            b4x=_B4_TRUE,
            mu_x=1.0,
            noise_std=1e-4,
        )
        g_gauss, st_g, _ = _fit(R_AA, R_BB, R_AB, shape="gaussian")
        g_k4, st_k, _, diag = _fit(
            R_AA, R_BB, R_AB, shape="kx4", return_diagnostics=True
        )
        assert st_g[0] == 0 and st_k[0] == 0
        # the pure Gaussian must show the kurtosis bias (that's the disease)...
        assert g_gauss[0, 9] > Sxx * 1.03
        # ...and the quartic term must remove it and read back the true b4
        assert g_k4[0, 9] == pytest.approx(Sxx, rel=0.05)
        assert g_k4[0, 10] == pytest.approx(Syy, rel=0.10)
        assert diag["b4x"][0] == pytest.approx(_B4_TRUE, rel=0.15)

    def test_ky4_recovers_biased_gaussian(self):
        """Symmetric case on the y axis."""
        Sxx, Syy = 0.5, 1.0
        R_AA, R_BB, R_AB = generate_quartic_triplet(
            (64, 64),
            sigma_particle_x=1.0,
            sigma_particle_y=1.0,
            sigma_stress_xx=Sxx,
            sigma_stress_yy=Syy,
            b4y=_B4_TRUE,
            mu_y=-0.5,
            noise_std=1e-4,
        )
        g_gauss, st_g, _ = _fit(R_AA, R_BB, R_AB, shape="gaussian")
        g_k4, st_k, _, diag = _fit(
            R_AA, R_BB, R_AB, shape="ky4", return_diagnostics=True
        )
        assert st_g[0] == 0 and st_k[0] == 0
        assert g_gauss[0, 10] > Syy * 1.03
        assert g_k4[0, 10] == pytest.approx(Syy, rel=0.05)
        assert g_k4[0, 9] == pytest.approx(Sxx, rel=0.10)
        assert diag["b4y"][0] == pytest.approx(_B4_TRUE, rel=0.15)

    def test_kx4_ky4_recovers_both_axes(self):
        """9-parameter mode: independent curvature on both axes at once."""
        Sxx, Syy = 1.0, 0.8
        b4x, b4y = _B4_TRUE, 0.5 * _B4_TRUE
        R_AA, R_BB, R_AB = generate_quartic_triplet(
            (64, 64),
            sigma_particle_x=1.0,
            sigma_particle_y=1.0,
            sigma_stress_xx=Sxx,
            sigma_stress_yy=Syy,
            b4x=b4x,
            b4y=b4y,
            mu_x=1.0,
            mu_y=-0.5,
            noise_std=1e-4,
        )
        gauss, status, _, diag = _fit(
            R_AA, R_BB, R_AB, shape="kx4+ky4", return_diagnostics=True
        )
        assert status[0] == 0
        assert gauss[0, 9] == pytest.approx(Sxx, rel=0.05)
        assert gauss[0, 10] == pytest.approx(Syy, rel=0.05)
        assert diag["b4x"][0] == pytest.approx(b4x, rel=0.15)
        assert diag["b4y"][0] == pytest.approx(b4y, rel=0.15)

    def test_gaussian_shape_uses_v6_residual(self, monkeypatch):
        """The default shape must run the untouched v6 residual (bit-exact path)."""
        import pivtools_cli.piv.piv_backend.kspace_lm_fitting as klm

        calls = {"v6": 0, "quartic": 0}
        real_v6 = klm._resid_jac_v6
        real_q = klm._resid_jac_quartic

        def spy_v6(*a, **kw):
            calls["v6"] += 1
            return real_v6(*a, **kw)

        def spy_q(*a, **kw):
            calls["quartic"] += 1
            return real_q(*a, **kw)

        monkeypatch.setattr(klm, "_resid_jac_v6", spy_v6)
        monkeypatch.setattr(klm, "_resid_jac_quartic", spy_q)

        R_AA, R_BB, R_AB = _triplet((32, 32), sigma_stress_xx=0.5, sigma_stress_yy=0.5)
        _fit(R_AA, R_BB, R_AB, shape="gaussian")
        assert calls["v6"] > 0 and calls["quartic"] == 0

        calls["v6"] = 0
        _fit(R_AA, R_BB, R_AB, shape="kx4+ky4")
        assert calls["quartic"] > 0 and calls["v6"] == 0
