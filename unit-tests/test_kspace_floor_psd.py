"""Tests for kspace_floor_psd.py — the analytic coloured noise-floor P(k; fx, fy).

Three layers of evidence, mirroring the offline validation chain:
  1. Exact algebra: tap-autocorrelation properties, the E_S(0)=0 hole, and a
     DENSE covariance-propagation cross-check of the rank-2 mean-removal terms
     at toy size (the one genuinely new derivation of the promotion).
  2. Monte-Carlo referee: the analytic P against a seeded mini port of the
     offline ``simulate_floor`` oracle (std + single chains, both kernels,
     hole on/off) at unit-test size. The tight 1024-pair gates (0.15-0.18 %
     worst per-cell median) ran offline; here the bar is 3 % at small n_pairs.
  3. Grid plumbing: build_P_grid normalization/validation and bilinear
     interp_P node-exactness.
"""

import numpy as np
import pytest
from floor_sim_helper import simulate_floor

from pivtools_cli.piv.piv_backend.base import CrossCorrelator
from pivtools_cli.piv.piv_backend.kspace_floor_psd import (
    KERNEL_TAPS,
    analytic_floor_single,
    build_P_grid,
    interp_P,
    tap_autocorr,
)
from pivtools_cli.piv.piv_backend.single_pass_accumulator import (
    _linear_pair_envelope,
)

KERNELS = ("lanczos", "cubic")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry fixtures (unit-test sized)
# ─────────────────────────────────────────────────────────────────────────────


def _std_geometry(h=16, w=16):
    """Std chain: square windows, flat weight, envelope from itself."""
    w_B = np.ones((h, w))
    env = _linear_pair_envelope(w_B, w_B, (h, w))
    return env, w_B


def _single_geometry(win=8, sum_win=24, crop=16):
    """Single chain: singlepix A in a bsingle sum window, central crop."""
    w_B = np.asarray(
        CrossCorrelator._window_weight_fun(
            (sum_win, sum_win), "bsingle", (sum_win, sum_win)
        ),
        dtype=np.float64,
    )
    w_A = np.asarray(
        CrossCorrelator._window_weight_fun(
            (win, win), "singlepix", (sum_win, sum_win)
        ),
        dtype=np.float64,
    )
    env = _linear_pair_envelope(w_B, w_B, (crop, crop))
    return env, w_A, w_B


# ─────────────────────────────────────────────────────────────────────────────
# 1a. Tap autocorrelation
# ─────────────────────────────────────────────────────────────────────────────


class TestTapAutocorr:
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_symmetric(self, kernel):
        """r(tau) = r(-tau): autocorrelation of a real sequence."""
        _, r = tap_autocorr(0.3, kernel)
        assert np.allclose(r, r[::-1])

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_zero_lag_is_energy(self, kernel):
        """r(0) = sum of squared tap weights."""
        offsets, weights_fun = KERNEL_TAPS[kernel]
        c = np.asarray(weights_fun(0.3))
        taus, r = tap_autocorr(0.3, kernel)
        assert np.isclose(r[np.flatnonzero(taus == 0)[0]], np.sum(c**2))

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_f0_is_delta(self, kernel):
        """At f=0 all weight sits on one tap: r = delta (white, no colouring)."""
        taus, r = tap_autocorr(0.0, kernel)
        expect = np.zeros_like(r)
        expect[np.flatnonzero(taus == 0)[0]] = 1.0
        assert np.allclose(r, expect, atol=1e-12)

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="unknown kernel"):
            tap_autocorr(0.2, "sinc7")


# ─────────────────────────────────────────────────────────────────────────────
# 1b. Exact algebra: DC hole + dense operator cross-check
# ─────────────────────────────────────────────────────────────────────────────


def _dense_expected_spectrum(fx, fy, w_B, kernel, hole):
    """E[|DFT2(A u)|^2] by dense covariance propagation (the referee for the
    rank-2 algebra). A = D_w (I - 1 w^T / sum w) when hole, else D_w; the
    warped-noise covariance is the separable C = C_y (x) C_x built from the
    tap autocorrelation. Returns the (sh, sw) spectrum in unshifted bin order.
    """
    sh, sw = w_B.shape

    def toeplitz_1d(f, n):
        taus, r = tap_autocorr(f, kernel)
        C = np.zeros((n, n))
        for tau, rv in zip(taus, r):
            idx = np.arange(max(0, -tau), min(n, n - tau))
            C[idx, idx + tau] = rv
        return C

    C = np.kron(toeplitz_1d(fy, sh), toeplitz_1d(fx, sw))  # row-major (y, x)
    wf = w_B.ravel()
    A = np.diag(wf)
    if hole:
        A = A @ (np.eye(wf.size) - np.outer(np.ones(wf.size), wf) / wf.sum())
    M = A @ C @ A.T

    ky, kx = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    yy, xx = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    E = np.empty((sh, sw))
    for iy in range(sh):
        for ix in range(sw):
            f_k = np.exp(
                -2j * np.pi * (iy * yy / sh + ix * xx / sw)
            ).ravel()
            E[iy, ix] = np.real(np.conj(f_k) @ M @ f_k)
    return E


class TestExactAlgebra:
    @pytest.mark.parametrize("kernel", KERNELS)
    @pytest.mark.parametrize("hole", (True, False))
    def test_dense_operator_crosscheck(self, kernel, hole):
        """analytic_floor_single == dense A C A^H propagation at toy size.

        With env = 1 and no crop the post-chain is a pure re-centring, so the
        returned P is |fftshift(E_S)| — comparable bin-by-bin against the
        dense quadratic form.
        """
        sh, sw = 10, 12
        rng = np.random.default_rng(3)
        w_B = 0.5 + rng.random((sh, sw))  # generic positive weight
        env = np.ones((sh, sw))
        fx, fy = 0.3, 0.15

        P = analytic_floor_single(fx, fy, env, w_B, kernel, hole)
        E_dense = _dense_expected_spectrum(fx, fy, w_B, kernel, hole)
        expect = np.abs(np.fft.fftshift(E_dense))
        assert np.allclose(P, expect, rtol=1e-9, atol=1e-9 * expect.max())

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_dc_hole_exact(self, kernel):
        """hole=True zeroes E_S(0) exactly (weighted-mean removal): with no
        crop and env = 1 the returned P is |fftshift(E_S)|, so the centre bin
        IS the DC of the expected spectrum. hole=False leaves it positive.
        (After a central crop the hole smears into neighbouring bins — the
        mean-hole width story — so DC exactness is only testable un-cropped.)
        """
        _, _, w_B = _single_geometry()
        sh, sw = w_B.shape
        env_full = np.ones((sh, sw))
        cy, cx = sh // 2, sw // 2
        P_hole = analytic_floor_single(0.2, 0.1, env_full, w_B, kernel, True)
        P_nohole = analytic_floor_single(0.2, 0.1, env_full, w_B, kernel, False)
        assert P_hole[cy, cx] < 1e-9 * P_hole.max()
        assert P_nohole[cy, cx] > 0.1 * P_nohole.max()

    def test_broken_weight_raises(self):
        """The internal E_S(0,0)=0 identity guard trips on inconsistent
        inputs rather than returning silent nonsense (here: a weight with
        negative entries breaks no identity — use direct API misuse via
        build_P_grid validation instead)."""
        env, w_B = _std_geometry()
        with pytest.raises(ValueError, match="2-D"):
            build_P_grid(env, "lanczos", True, w_B.ravel())
        with pytest.raises(ValueError, match="smaller than"):
            build_P_grid(np.ones((32, 32)), "lanczos", True, np.ones((16, 16)))

    def test_float32_weight_upcast(self):
        """Production weights arrive float32 (_window_weight_fun); float32 tap
        sums broke the exact E_S(0)=0 identity on a 64x64 window (caught in
        the first e2e run). The module must upcast internally: float32 input
        == float64 input bit-for-bit, no guard trip."""
        h = w = 64
        w_B64 = np.ones((h, w))
        env = _linear_pair_envelope(w_B64, w_B64, (h, w))
        P64 = analytic_floor_single(0.5, 0.0, env, w_B64, "lanczos", True)
        # only the WEIGHT is float32 in production (env is float64 from
        # _linear_pair_envelope); ones are exactly representable so the
        # upcast must reproduce the float64 result bit-for-bit
        P32 = analytic_floor_single(
            0.5, 0.0, env, w_B64.astype(np.float32), "lanczos", True
        )
        assert np.array_equal(P32, P64)

    def test_fy_produces_ky_structure(self):
        """fy != 0 colours the ky axis (the 2-D generalisation): the kx=0
        column varies; at fy = 0 with hole off the same column is flat."""
        env, w_B = _std_geometry()
        w = env.shape[1]
        cx = w // 2
        P_fy = analytic_floor_single(0.0, 0.35, env, w_B, "lanczos", False)
        P_f0 = analytic_floor_single(0.0, 0.0, env, w_B, "lanczos", False)
        cut_fy = P_fy[:, cx]
        cut_f0 = P_f0[:, cx]
        assert np.ptp(cut_f0) < 1e-9 * cut_f0.max()  # flat at fy=0, no hole
        assert np.ptp(cut_fy) > 0.1 * cut_fy.max()  # coloured at fy=0.35


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monte-Carlo referee (mini simulate_floor port)
# ─────────────────────────────────────────────────────────────────────────────

_MC_TOL = 0.03  # median relative deviation at unit-test n_pairs
_N_PAIRS = 48
_IMG = 192


def _median_rel_dev(P_ana, P_sim):
    """Median bin-wise relative deviation after matching the normalization."""
    P_a = P_ana / P_ana.mean()
    P_s = P_sim / P_sim.mean()
    return np.median(np.abs(P_a - P_s) / np.maximum(P_s, 1e-12))


class TestVsMonteCarlo:
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_std_chain(self, kernel):
        env, w_B = _std_geometry()
        h, w = env.shape
        rng = np.random.default_rng(11)
        fx = 0.25
        P_sim = simulate_floor(fx, env, _N_PAIRS, _IMG, h, w, rng, kernel=kernel)
        P_ana = analytic_floor_single(fx, 0.0, env, w_B, kernel, True)
        assert _median_rel_dev(P_ana, P_sim) < _MC_TOL

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_single_chain(self, kernel):
        env, w_A, w_B = _single_geometry()
        h, w = env.shape
        rng = np.random.default_rng(12)
        fx = 0.25
        P_sim = simulate_floor(
            fx, env, _N_PAIRS, _IMG, h, w, rng, weights=(w_A, w_B), kernel=kernel
        )
        P_ana = analytic_floor_single(fx, 0.0, env, w_B, kernel, True)
        assert _median_rel_dev(P_ana, P_sim) < _MC_TOL

    def test_fy_axis(self):
        """The 2-D generalisation against the sim's fy path."""
        env, w_B = _std_geometry()
        h, w = env.shape
        rng = np.random.default_rng(13)
        P_sim = simulate_floor(
            0.0, env, _N_PAIRS, _IMG, h, w, rng, kernel="lanczos", fy=0.3
        )
        P_ana = analytic_floor_single(0.0, 0.3, env, w_B, "lanczos", True)
        assert _median_rel_dev(P_ana, P_sim) < _MC_TOL

    def test_no_window_mean(self):
        """hole=False against the sim without per-pair mean removal."""
        env, w_B = _std_geometry()
        h, w = env.shape
        rng = np.random.default_rng(14)
        P_sim = simulate_floor(
            0.25, env, _N_PAIRS, _IMG, h, w, rng, kernel="lanczos",
            window_mean=False,
        )
        P_ana = analytic_floor_single(0.25, 0.0, env, w_B, "lanczos", False)
        assert _median_rel_dev(P_ana, P_sim) < _MC_TOL


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grid + interpolation plumbing
# ─────────────────────────────────────────────────────────────────────────────


class TestGridPlumbing:
    def test_build_normalization(self):
        env, w_B = _std_geometry(8, 8)
        fs, P_grid = build_P_grid(env, "lanczos", True, w_B, n_fracs=3)
        assert fs.shape == (3,)
        assert P_grid.shape == (3, 3, 8, 8)
        assert np.isclose(P_grid[0, 0].mean(), 1.0)

    def test_interp_node_exact(self):
        env, w_B = _std_geometry(8, 8)
        fs, P_grid = build_P_grid(env, "lanczos", True, w_B, n_fracs=3)
        # (fx, fy) exactly on grid nodes reproduces the stored planes
        for iy, fy in enumerate(fs):
            for ix, fx in enumerate(fs):
                out = interp_P(fs, P_grid, np.array([fx]), np.array([fy]))
                assert np.allclose(out[0], P_grid[iy, ix].ravel())

    def test_interp_bilinear_midpoint(self):
        env, w_B = _std_geometry(8, 8)
        fs, P_grid = build_P_grid(env, "lanczos", True, w_B, n_fracs=3)
        mid = 0.5 * (fs[0] + fs[1])
        out = interp_P(fs, P_grid, np.array([mid]), np.array([fs[0]]))
        expect = 0.5 * (P_grid[0, 0] + P_grid[0, 1])
        assert np.allclose(out[0], expect.ravel())

    def test_interp_out_of_range_raises(self):
        env, w_B = _std_geometry(8, 8)
        fs, P_grid = build_P_grid(env, "lanczos", True, w_B, n_fracs=3)
        with pytest.raises(ValueError, match="outside the P grid"):
            interp_P(fs, P_grid, np.array([0.7]), np.array([0.0]))

    def test_interp_length_mismatch_raises(self):
        env, w_B = _std_geometry(8, 8)
        fs, P_grid = build_P_grid(env, "lanczos", True, w_B, n_fracs=3)
        with pytest.raises(ValueError, match="length mismatch"):
            interp_P(fs, P_grid, np.zeros(3), np.zeros(4))
