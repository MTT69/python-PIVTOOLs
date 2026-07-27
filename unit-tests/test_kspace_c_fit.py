"""Parity gate: the production C k-space fitter vs the retained NumPy oracle.

``fit_windows_kspace_lm`` dispatches to libkspacefit; ``fit_windows_kspace_lm_numpy``
is the reference implementation it replaced. These tests hold the two to the same
fits across every shape x floor combination, and pin the output contract that the
Dask caller and the .mat writers depend on.

Tolerances are deliberate, not bit-identity. Two documented sources of last-ulp
divergence exist:
  * libm exp/cos/sin differ from NumPy's float64 routines in the last ulp, so LM
    accept/reject can flip on a marginal step and iteration paths can differ;
  * the flat floor's attenuation is a true division in Python against a
    reciprocal-multiply in C (the C unifies both floors behind one Dv buffer).
The real end-to-end gate is the offline harness on saved production planes; this
file guards the wiring, the contract and the shape/floor switches.
"""

import numpy as np
import pytest
from synthetic_correlations import generate_correlation_triplet, make_mock_config

from pivtools_cli.piv.piv_backend.kspace_lm_fitting import (
    STATUS_MASKED,
    STATUS_SUCCESS,
    fit_windows_kspace_lm,
    fit_windows_kspace_lm_numpy,
)

from pivtools_cli.piv.piv_backend.kspace_lm_fitting import _load_kspace_lib

try:
    _load_kspace_lib()
    _LIB_OK = True
except FileNotFoundError:
    # ONLY the genuine "not built" case, which the loader raises explicitly.
    # OSError (unresolvable libomp, arch mismatch, corrupt DLL) and RuntimeError
    # (stale build, missing export) must propagate: skipping on those would
    # silently disable the only automated check on the production maths, under a
    # reason string that says "not built" when the truth is "built and broken".
    _LIB_OK = False

pytestmark = pytest.mark.skipif(
    not _LIB_OK, reason="libkspacefit not built — run `python setup.py build`"
)

# 24x24 is a 2^k*3 size: it needs the mixed-radix codelets, and is the plane size
# on the experimental datasets. A power-of-two-only test would not exercise them.
CORR = (24, 24)
N_WINDOWS = 40

SHAPES = ("gaussian", "kx4", "ky4", "kx4+ky4")
FLOORS = ("flat", "coloured")

SIGMA_TOL = 1e-8  # |dSigma| on jointly-converged windows
MU_TOL = 1e-8


def _batch(seed=7):
    """A batch of genuinely different windows, with masked ones interleaved."""
    rng = np.random.default_rng(seed)
    aa, bb, ab = [], [], []
    for i in range(N_WINDOWS):
        R_AA, R_BB, R_AB = generate_correlation_triplet(
            CORR,
            sigma_particle_x=1.8 + 0.4 * rng.random(),
            sigma_particle_y=1.8 + 0.4 * rng.random(),
            sigma_stress_xx=0.3 * rng.random(),
            sigma_stress_yy=0.3 * rng.random(),
            mu_x=rng.uniform(-1.5, 1.5),
            mu_y=rng.uniform(-1.5, 1.5),
            noise_std=0.002,
            seed=100 + i,
        )
        aa.append(R_AA.ravel())
        bb.append(R_BB.ravel())
        ab.append(R_AB.ravel())
    mask = np.zeros(N_WINDOWS, dtype=bool)
    mask[::7] = True
    return (
        np.concatenate(aa),
        np.concatenate(bb),
        np.concatenate(ab),
        mask,
    )


def _p_win():
    """A smooth positive P in the CENTRED layout.

    Any positive array is a valid parity probe — both implementations receive the
    identical array, and the centred->natural mapping is the C's own business.
    """
    ky, kx = np.meshgrid(
        np.linspace(-1, 1, CORR[0]), np.linspace(-1, 1, CORR[1]), indexing="ij"
    )
    return np.tile((1.0 + 0.6 * np.exp(-(kx**2 + ky**2) / 0.25)).ravel(), N_WINDOWS)


def _run_both(shape, floor):
    R_AA, R_BB, R_AB, mask = _batch()
    cfg = make_mock_config(
        ensemble_kspace_shape=shape, ensemble_kspace_floor=floor
    )
    kw = {"P_win": _p_win()} if floor == "coloured" else {}
    c = fit_windows_kspace_lm(
        R_AA, R_BB, R_AB, mask, CORR, cfg, 0, return_diagnostics=True, **kw
    )
    p = fit_windows_kspace_lm_numpy(
        R_AA, R_BB, R_AB, mask, CORR, cfg, 0, return_diagnostics=True, **kw
    )
    return c, p, mask


@pytest.mark.parametrize("floor", FLOORS)
@pytest.mark.parametrize("shape", SHAPES)
class TestCParity:
    def test_status_agreement(self, shape, floor):
        (_, sc, _, _), (_, sp, _, _), _ = _run_both(shape, floor)
        assert np.array_equal(sc, sp), (
            f"status disagreement {shape}/{floor}: "
            f"{int(np.sum(sc != sp))}/{sc.size} windows differ"
        )

    def test_sigma_and_mu_agreement(self, shape, floor):
        (gc, sc, _, _), (gp, sp, _, _), _ = _run_both(shape, floor)
        ok = (sc == STATUS_SUCCESS) & (sp == STATUS_SUCCESS)
        assert ok.any(), (
            "no jointly-converged windows - fixture is not exercising the fit"
        )
        assert np.abs(gc[ok, 9:12] - gp[ok, 9:12]).max() < SIGMA_TOL
        assert np.abs(gc[ok, 14:16] - gp[ok, 14:16]).max() < MU_TOL

    def test_initial_guess_exact(self, shape, floor):
        """Seeds are computed before the LM runs, so they must match exactly.

        Isolates any Sigma difference to the LM trajectory, ruling out the FFT,
        the k-grids, the valid-bin mask and the seeding as sources.

        Columns 14:15 carry mu, which reaches the seed through three `log` calls
        (`subpix_3pt` vs `np.log`). NumPy's SIMD float64 log is not guaranteed to
        agree with libm in the last ulp, so those two get a tight tolerance
        rather than bit-equality — demanding exactness there would contradict
        this file's own stated tolerance philosophy and could flake on another
        toolchain. Everything else is bit-exact and asserted as such.
        """
        (_, _, ic, _), (_, _, ip, _), _ = _run_both(shape, floor)
        cols = [c for c in range(16) if c not in (14, 15)]
        assert np.array_equal(ic[:, cols], ip[:, cols], equal_nan=True)
        np.testing.assert_allclose(
            ic[:, 14:16], ip[:, 14:16], rtol=0, atol=1e-12
        )

    def test_diag_keys_match(self, shape, floor):
        (_, _, _, dc), (_, _, _, dp), _ = _run_both(shape, floor)
        assert set(dc) == set(dp)
        assert ("b4x" in dc) == ("kx4" in shape)
        assert ("b4y" in dc) == ("ky4" in shape)

    def test_diag_values_agreement(self, shape, floor):
        """Diagnostics must agree in VALUE, not just in key.

        These arrays are concatenated straight into fit_diagnostics_pass_N.mat
        (single_pass_accumulator.py), so a wrong one is a wrong published number.
        Nothing else in this file would notice: extracting the quartic
        coefficients in the wrong order, for instance, leaves every fitted
        parameter untouched and only swaps b4x/b4y in the sidecar.
        """
        (_, sc, _, dc), (_, sp, _, dp), _ = _run_both(shape, floor)

        # n_valid is decided before the LM runs, from the FFT and the valid-bin
        # mask alone, so it is deterministic and must match exactly on EVERY
        # window — the sharpest available probe of the transform and the mask.
        assert np.array_equal(dc["n_valid"], dp["n_valid"])

        ok = (sc == STATUS_SUCCESS) & (sp == STATUS_SUCCESS)
        assert ok.any()
        for key in ("gain", "N0", "cost_per_pt", "b4x", "b4y"):
            if key not in dc:
                continue
            np.testing.assert_allclose(
                dc[key][ok], dp[key][ok], rtol=1e-8, atol=1e-10,
                err_msg=f"diag['{key}'] disagrees for {shape}/{floor}",
            )


class TestOutputContract:
    """Pins what single_pass_accumulator and the .mat writers rely on."""

    def test_masked_rows(self):
        (gc, sc, ic, dc), _, mask = _run_both("gaussian", "coloured")
        assert np.all(sc[mask] == STATUS_MASKED)
        # initial_guess masked rows stay all-zero; gauss_flat masked rows carry
        # the NaN particle-size slots and the window centres.
        assert np.all(ic[mask] == 0.0)
        assert np.all(np.isnan(gc[mask][:, 6:9]))
        assert np.all(gc[mask][:, 12] == CORR[1] / 2.0 + 1.0)
        assert np.all(gc[mask][:, 13] == CORR[0] / 2.0 + 1.0)
        assert np.all(np.isnan(dc["gain"][mask]))

    def test_all_masked_batch(self):
        R_AA, R_BB, R_AB, _ = _batch()
        mask = np.ones(N_WINDOWS, dtype=bool)
        cfg = make_mock_config()
        gc, sc, ic, _ = fit_windows_kspace_lm(
            R_AA, R_BB, R_AB, mask, CORR, cfg, 0, return_diagnostics=True
        )
        gp, sp, ip, _ = fit_windows_kspace_lm_numpy(
            R_AA, R_BB, R_AB, mask, CORR, cfg, 0, return_diagnostics=True
        )
        # production mirrors gauss_flat into initial_guess in this branch only
        assert np.array_equal(gc, gp, equal_nan=True)
        assert np.array_equal(ic, ip, equal_nan=True)
        assert np.array_equal(sc, sp)

    def test_dtypes_and_shapes(self):
        (gc, sc, ic, dc), _, _ = _run_both("kx4+ky4", "coloured")
        assert gc.shape == (N_WINDOWS, 16) and gc.dtype == np.float64
        assert ic.shape == (N_WINDOWS, 16) and ic.dtype == np.float64
        assert sc.shape == (N_WINDOWS,) and sc.dtype == np.int32
        assert dc["conv"].dtype == np.bool_
        assert dc["n_valid"].dtype == np.int32 and dc["iter"].dtype == np.int32

    def test_thread_count_does_not_change_results(self):
        """Windows are independent and no reduction crosses them, so the fit
        must be invariant to the OpenMP width. A difference here would mean a
        shared-state bug in the parallel region."""
        R_AA, R_BB, R_AB, mask = _batch()
        out = []
        for threads in ("1", "4"):
            cfg = make_mock_config(omp_threads=threads)
            out.append(fit_windows_kspace_lm(R_AA, R_BB, R_AB, mask, CORR, cfg, 0))
        assert np.array_equal(out[0][0], out[1][0], equal_nan=True)
        assert np.array_equal(out[0][1], out[1][1])


class TestLoaderContract:
    def test_wiring_guards_still_raise(self):
        """The validation is shared with the oracle and must fire before C."""
        R_AA, R_BB, R_AB, mask = _batch()
        with pytest.raises(ValueError, match="requires P_win"):
            fit_windows_kspace_lm(
                R_AA, R_BB, R_AB, mask, CORR,
                make_mock_config(ensemble_kspace_floor="coloured"), 0,
            )
        with pytest.raises(ValueError, match="wiring bug"):
            fit_windows_kspace_lm(
                R_AA, R_BB, R_AB, mask, CORR, make_mock_config(), 0,
                P_win=_p_win(),
            )

    def test_unsupported_corr_size_raises(self):
        """An axis length with no generated codelet must fail loudly, not fall
        back to anything."""
        bad = (20, 20)  # not in BUILT_FFT_SIZES
        n = 2
        planes = np.zeros(n * bad[0] * bad[1])
        mask = np.zeros(n, dtype=bool)
        with pytest.raises(RuntimeError, match="BUILT_FFT_SIZES"):
            fit_windows_kspace_lm(
                planes, planes, planes, mask, bad, make_mock_config(), 0
            )
