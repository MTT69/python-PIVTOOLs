"""
Batched Levenberg-Marquardt k-space fitter: GSL-replica minus the (1+beta*q^2) term.

PRODUCTION ensemble fitter since 2026-07-08 (promoted from manual_tools/kspace after
beating the closed-form linear fitter on the noisy planar validation set: med|relerr|
vs DNS uu 3.3/vv 2.2/uv 5.4% vs 8.1/9.8/8.4%, outer band 2.2% vs -11% — see
wiki/sessions/2026-07-07-lm-gsl-replica-fitter.md). The closed-form predecessor
``kspace_linear_fitting`` is dormant in-tree for possible revert.

A faithful, GSL-free port of the deleted two-stage nonlinear fitter
``pivtools_cli/lib/kspace_fitting.c`` (recover with ``git show f1b9ec9^:...``), with
exactly ONE modelling change: the stage-1 kurtosis-like envelope term ``(1+beta*q^2)``
is removed (beta was degenerate with N0 and sigma). No kurtosis term anywhere: a
per-axis Edgeworth quartic in stage 2 was tested and REJECTED 2026-07-08 (vv degraded,
stage-2 convergence collapsed to 55% — ill-conditioned on real planes; see the wiki
session page).

Pipeline per window (C step numbers kept):
  1. FFT R_AA/R_BB/R_AB -> F_AA/F_BB/F_AB (centred, fftshift(fft2(ifftshift(.)))).
  2. Reference spectrum ref = sqrt(|F_AA|*|F_BB|); analytic warp-noise MTF P_noise.
  3. Stage-1 joint noise fit over the WHOLE normalised auto-spectrum, flat weights:
         ref/ref_dc ~= (A*exp(-q) + N0) * P_noise,   q = 2*pi^2 (kx^2 sx^2 + ky^2 sy^2)
     4 params (A, sx, sy, N0), p0 = (1, 2, 2, 0.01), boxes A[0.01,10] s[0.1,20] N0[0,1];
     then subtract the coloured floor: ref <- max(ref - N0_abs*P_noise, ref_dc*1e-8).
  5. Per-axis k_max from the 1%-of-DC profile threshold, floored 0.05, capped 0.35
     (or ``k_max_cap``); elliptical trust mask.
  6. Seeds: 3-point log-Gaussian sub-pixel peak (mu); through-origin 1D log-magnitude
     regression per axis (Sigma, k_min = 1.5/N, Sigma default 1.0 / floor 0.01).
  7. Stage-2 main fit of the DC-self-normalised complex transfer function
         T_norm(k) = exp(-2*pi^2 (Sxx kx^2 + 2 Sxy kx ky + Syy ky^2))
                     * exp(-2*pi*i (kx mu_x + ky mu_y))
     5 params, residuals = weights * (real, imag) stacked, analytic Jacobian; soft
     weighting w_snr * w_soft (production default) or flat weights on the ellipse.
  8. Accept if converged-by-tol OR (max_iter reached AND cost/n_valid < 1.0); clamp
     Sigma to >= 0; reject |mu| > 0.75*window (status 3).

Honest deviations from the C (complete list):
  * beta removed from the stage-1 model+Jacobian (the requested change).
  * LM implementation: classic Marquardt lambda-damping with diagonal scaling, batched
    over windows (per-window lambda, accept/reject, convergence), instead of GSL's
    trust-region LM + geodesic acceleration. Same weighted SSE, same Jacobian; the
    manual harness's scipy oracle (manual_tools/kspace/lm_fit_vs_gt.py) cross-validates
    the optimum. Lambda blow-out (no progress possible) maps to GSL_ENOPROG ->
    rejected, matching the C's acceptance of only GSL_SUCCESS/GSL_EMAXITER.
  * float64 numpy.fft instead of float32 FFTW (FFT convention itself identical).
  * Box constraints by projecting the step onto the box, instead of the C's
    clamp-inside-residual + Jacobian-zeroing (same feasible set). Stage 2 bounds
    Sxx, Syy >= 0 (the C's fmax) via the same projection.
  * The C computed a 1D phase-slope mu inside fit_1d_variance and then discarded it
    (dead output; the seed used the peak-based mu). Not reproduced.

Parallelism: pure NumPy, vectorised over windows, chunked by identical fractional
shift; native threadpools pinned to 1 (the accumulator submits one window block per
Dask worker, so total threads == workers). Same positional signature and 16-column
``gauss_flat`` return contract as the dormant ``fit_windows_kspace_linear``; unlike it,
``use_soft_weighting`` and ``k_max_cap`` are honoured (they were real GSL knobs).
"""

import logging
from typing import Optional

import numpy as np

from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    compute_noise_psd_2d, frac_distance,
)
from pivtools_cli.piv.piv_backend.kspace_common import (
    _fft_planes, _kgrids, _get_threadpool_controller,
)

logger = logging.getLogger(__name__)

# --- constants mirroring the C #defines / inline values ---------------------------------
JOINT_MAX_ITER = 200         # stage-1 LM iteration cap
MAIN_MAX_ITER = 250          # stage-2 LM iteration cap
LM_XTOL = 1e-8               # step-size convergence (GSL XTOL)
LM_FTOL = 1e-8               # cost-reduction convergence (GSL FTOL)
K_MAX_DEFAULT = 0.35         # default k_max cap (avoids the old noise-estimation corner)
PROFILE_THRESHOLD = 0.01     # k_max profile scan: 1% of clean DC
PROFILE_KMIN = 0.05          # k_max floor
SEED_KMIN_FACTOR = 1.5       # 1D seed regression k_min = 1.5/N
MIN_VALID_PTS = 10           # minimum k-points inside the ellipse
MAX_DISP_FRAC = 0.75         # |mu| gate as fraction of window size
COST_PER_PT_ACCEPT = 1.0     # EMAXITER acceptance: cost/n_valid < this

JOINT_P0 = np.array([1.0, 2.0, 2.0, 0.01])
JOINT_LO = np.array([0.01, 0.1, 0.1, 0.0])
JOINT_HI = np.array([10.0, 20.0, 20.0, 1.0])
MAIN_LO = np.array([-np.inf, -np.inf, 0.0, 0.0, -np.inf])   # Sxx, Syy >= 0 (C's fmax)
MAIN_HI = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])

STATUS_MASKED = -1
STATUS_SUCCESS = 0
STATUS_NO_CONVERGE = 1
STATUS_LOW_SNR = 2
STATUS_BIG_DISP = 3
STATUS_NEG_VAR = 5           # kept for contract parity (the C defined but never set it)

_TWO_PI = 2.0 * np.pi
_TWO_PI2 = 2.0 * np.pi ** 2

# Stage-2 leaves Sxy unbounded (MAIN_LO/HI), so an LM trial step can drive Sxy^2 > Sxx*Syy,
# making the covariance Sigma indefinite and quad = k^T Sigma k negative at high k. Then
# decay = exp(-2pi^2 * quad) = exp(+big) overflows to inf, which poisons the residual and
# Jacobian (NaN) and spuriously fails flat/low-SNR windows. Clip the exponent: for feasible
# steps (quad >= 0 -> arg <= 0) this never triggers, so converged fits are bit-identical; on
# a bad step the residual/gradient stay finite and large, so LM rejects and backtracks cleanly.
_EXP_ARG_MAX = 300.0  # exp(300) ~ 2e130: huge (forces rejection) yet far below float64 overflow


# ==========================================================================================
# Batched Levenberg-Marquardt
# ==========================================================================================
def _batched_lm(fn, x0, lo, hi, max_iter, xtol=LM_XTOL, ftol=LM_FTOL):
    """Marquardt-damped Gauss-Newton over a batch of independent small problems.

    fn(x, idx, jac) -> residuals (m, M) [and Jacobian (m, M, K) when jac=True] for the
    subset ``idx`` of windows at parameters ``x`` (m, K). Residuals include the weights
    (r = w*(y - model)), so cost = sum(r^2) is the weighted SSE and the normal equations
    are J^T J delta = -J^T r.

    Per-window lambda (init 1e-3, /3 on accept, *10 on reject) scales the diagonal of
    J^T J (Marquardt scaling). Converged (accepted step AND (xtol step or ftol drop))
    windows freeze. Lambda > 1e12 with a rejected step = no progress possible
    (GSL_ENOPROG analogue): the window deactivates UNconverged with niter < max_iter,
    which the caller must treat as a failed fit.

    Returns (x, converged, cost, niter).
    """
    x = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    N, K = x.shape
    r0 = fn(x, np.arange(N), jac=False)
    cost = np.einsum("nm,nm->n", r0, r0)
    lam = np.full(N, 1e-3)
    conv = np.zeros(N, dtype=bool)
    niter = np.zeros(N, dtype=np.int32)
    active = np.isfinite(cost)
    kk = np.arange(K)

    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        xi = x[idx]
        r, J = fn(xi, idx, jac=True)
        A = np.einsum("nmi,nmj->nij", J, J)
        g = np.einsum("nmi,nm->ni", J, r)
        diag = np.einsum("nii->ni", A)
        Ad = A.copy()
        Ad[:, kk, kk] += lam[idx][:, None] * np.maximum(diag, 1e-30)
        try:
            delta = np.linalg.solve(Ad, -g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            delta = np.full_like(g, np.nan)
            for j in range(Ad.shape[0]):
                try:
                    delta[j] = np.linalg.solve(Ad[j], -g[j])
                except np.linalg.LinAlgError:
                    pass
        xn = np.clip(xi + delta, lo, hi)
        rn = fn(xn, idx, jac=False)
        costn = np.einsum("nm,nm->n", rn, rn)
        acc = np.isfinite(costn) & (costn <= cost[idx]) & np.all(np.isfinite(xn), axis=1)

        small_step = np.all(np.abs(xn - xi) <= xtol * (xtol + np.abs(xn)), axis=1)
        small_drop = (cost[idx] - costn) <= ftol * np.maximum(cost[idx], 1e-300)
        newly = acc & (small_step | small_drop)

        upd = idx[acc]
        x[upd] = xn[acc]
        cost[upd] = costn[acc]
        lam[upd] = np.maximum(lam[upd] / 3.0, 1e-12)
        rej = idx[~acc]
        lam[rej] *= 10.0
        niter[idx] += 1
        conv[idx[newly]] = True
        active[idx[newly]] = False
        active[rej[lam[rej] > 1e12]] = False       # ENOPROG: stuck, give up unconverged

    return x, conv, cost, niter


# ==========================================================================================
# Residuals + analytic Jacobians (ports of joint_residual_f/df, main_residual_f/df)
# ==========================================================================================
def _stage1_resid_jac(x, Y, Pn, KX2, KY2, jac=False):
    """Stage-1 joint floor, beta ==0: r = Y - (A*exp(-q) + N0)*P.  x (m,4), Y (m,P)."""
    A = x[:, 0:1]
    sx = x[:, 1:2]
    sy = x[:, 2:3]
    N0 = x[:, 3:4]
    q = _TWO_PI2 * (KX2[None] * sx ** 2 + KY2[None] * sy ** 2)     # (m, P)
    E = np.exp(-q)
    r = Y - (A * E + N0) * Pn[None]
    if not jac:
        return r
    m, P = r.shape
    J = np.empty((m, P, 4))
    J[:, :, 0] = -E * Pn[None]
    # dmodel/dsx = dsignal/dq * dq/dsx * P, dsignal/dq = -A*E (beta=0), dq/dsx = 2*2pi^2*kx^2*sx
    J[:, :, 1] = A * E * (2.0 * _TWO_PI2 * KX2[None] * sx) * Pn[None]
    J[:, :, 2] = A * E * (2.0 * _TWO_PI2 * KY2[None] * sy) * Pn[None]
    J[:, :, 3] = -Pn[None] * np.ones_like(E)
    return r, J


def _stage2_resid_jac(x, Tre, Tim, W, KXf, KYf, jac=False):
    """Stage-2 complex transfer-function fit. x (m,5) = (mu_x, mu_y, Sxx, Syy, Sxy).

    Residuals (m, 2P): [W*(Tre - decay*cos), W*(Tim - decay*sin)]. Sxx/Syy arrive >= 0
    (projected box); no residual-side clamp needed.
    """
    mux = x[:, 0:1]
    muy = x[:, 1:2]
    Sxx = x[:, 2:3]
    Syy = x[:, 3:4]
    Sxy = x[:, 4:5]
    KX2 = KXf * KXf
    KY2 = KYf * KYf
    KXKY = KXf * KYf
    quad = Sxx * KX2[None] + 2.0 * Sxy * KXKY[None] + Syy * KY2[None]   # (m, P)
    # clip guards against non-PSD trial steps (quad<0); no-op for feasible steps (see _EXP_ARG_MAX)
    decay = np.exp(np.minimum(-_TWO_PI2 * quad, _EXP_ARG_MAX))
    phase = -_TWO_PI * (KXf[None] * mux + KYf[None] * muy)
    cosp = np.cos(phase)
    sinp = np.sin(phase)
    r = np.concatenate([W * (Tre - decay * cosp), W * (Tim - decay * sinp)], axis=1)
    if not jac:
        return r
    m, P = decay.shape
    J = np.empty((m, 2 * P, 5))
    dpx = -_TWO_PI * KXf[None]                       # dphase/dmu_x
    dpy = -_TWO_PI * KYf[None]
    J[:, :P, 0] = -W * decay * (-sinp) * dpx
    J[:, P:, 0] = -W * decay * cosp * dpx
    J[:, :P, 1] = -W * decay * (-sinp) * dpy
    J[:, P:, 1] = -W * decay * cosp * dpy
    ddxx = decay * (-_TWO_PI2 * KX2[None])           # ddecay/dSxx
    ddyy = decay * (-_TWO_PI2 * KY2[None])
    ddxy = decay * (-_TWO_PI2 * 2.0 * KXKY[None])
    J[:, :P, 2] = -W * ddxx * cosp
    J[:, P:, 2] = -W * ddxx * sinp
    J[:, :P, 3] = -W * ddyy * cosp
    J[:, P:, 3] = -W * ddyy * sinp
    J[:, :P, 4] = -W * ddxy * cosp
    J[:, P:, 4] = -W * ddxy * sinp
    return r, J


# ==========================================================================================
# Seeds + k_max (ports of estimate_displacement_from_peak, fit_1d_variance,
# compute_kmax_from_profile)
# ==========================================================================================
def _peak_mu(R_AB, cy, cx):
    """3-point log-Gaussian sub-pixel peak of each correlation plane. R_AB (n, h, w)."""
    n, h, w = R_AB.shape
    flat = R_AB.reshape(n, -1)
    pidx = flat.argmax(axis=1)
    py, px = np.unravel_index(pidx, (h, w))
    rows = np.arange(n)

    def _subpix(lo, ce, hi_, interior):
        sub = np.zeros(n)
        m = interior & (lo > 0) & (ce > 0) & (hi_ > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_l = np.log(np.where(m, lo, 1.0))
            ln_c = np.log(np.where(m, ce, 1.0))
            ln_r = np.log(np.where(m, hi_, 1.0))
        denom = 2.0 * (ln_l - 2.0 * ln_c + ln_r)
        m &= np.abs(denom) > 1e-12
        sub[m] = ((ln_l - ln_r) / np.where(m, denom, 1.0))[m]
        return sub

    ix = np.clip(px, 1, w - 2)                    # safe gather; interior mask gates use
    sub_x = _subpix(R_AB[rows, py, ix - 1], R_AB[rows, py, ix], R_AB[rows, py, ix + 1],
                    (px > 0) & (px < w - 1))
    iy = np.clip(py, 1, h - 2)
    sub_y = _subpix(R_AB[rows, iy - 1, px], R_AB[rows, iy, px], R_AB[rows, iy + 1, px],
                    (py > 0) & (py < h - 1))
    return (px + sub_x) - cx, (py + sub_y) - cy


def _seed_sigma_1d(fab_prof, fref_prof, k_axis, k_max):
    """Weighted through-origin regression of ln|F_AB|-ln|F_ref| on k^2 (per window).

    fab_prof/fref_prof (n, N) magnitude profiles along one axis; k_max (n,).
    Returns Sigma_init (n,): max(-slope/2pi^2, 0.01), default 1.0 when < 3 points.
    """
    N = k_axis.size
    ak = np.abs(k_axis)[None]                                   # (1, N)
    k_min = SEED_KMIN_FACTOR / N
    in_rng = (ak > k_min) & (ak < k_max[:, None])               # (n, N)
    max_fab = np.max(np.where(in_rng, fab_prof, 0.0), axis=1)
    max_fab = np.where(max_fab < 1e-12, 1.0, max_fab)
    valid = in_rng & (fab_prof > 1e-12) & (fref_prof > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logT = np.where(valid, np.log(np.where(valid, fab_prof, 1.0))
                        - np.log(np.where(valid, fref_prof, 1.0)), 0.0)
    wgt = np.where(valid, fab_prof / max_fab[:, None], 0.0)
    k2 = (k_axis * k_axis)[None]
    sum_wyk2 = np.sum(wgt * logT * k2, axis=1)
    sum_wk4 = np.sum(wgt * k2 * k2, axis=1)
    count = valid.sum(axis=1)
    ok = (count >= 3) & (np.abs(sum_wk4) > 1e-20)
    slope = np.where(ok, sum_wyk2 / np.where(ok, sum_wk4, 1.0), 0.0)
    return np.where(ok, np.maximum(-slope / _TWO_PI2, 0.01), 1.0)


def _profile_kmax(prof, k_axis, F_dc_clean, k_max_limit):
    """First profile point below 1% of clean DC, scanning outward from the centre.

    prof (n, N) = |F_ref| along one axis (post floor subtraction); returns k_max (n,)
    clamped to [PROFILE_KMIN, k_max_limit].
    """
    n, N = prof.shape
    center = N // 2
    thresh = (F_dc_clean * PROFILE_THRESHOLD)[:, None]
    below = prof[:, center:] < thresh                            # scan i = center .. N-1
    has = below.any(axis=1)
    j = below.argmax(axis=1)                                     # first True offset
    k_prev = k_axis[np.clip(center + j - 1, 0, N - 1)]           # k_axis[i-1]
    k_max = np.where(has, np.where(j > 0, k_prev, PROFILE_KMIN), k_max_limit)
    return np.clip(k_max, PROFILE_KMIN, k_max_limit)


# ==========================================================================================
# Chunk pipeline (steps 1-7): preparation shared by the batched fit and the oracle
# ==========================================================================================
def _prepare_chunk(R_AA, R_BB, R_AB, KX, KY, cy, cx, f_xy, kernel,
                   use_soft_weighting, k_max_limit):
    """Run C steps 1-6 for one constant-fractional-shift chunk. Returns a prep dict.

    Everything the stage-2 LM needs (T_norm, weights, seeds), plus per-window status
    for windows that fail a gate on the way, plus stage-1 diagnostics.
    """
    n = R_AA.shape[0]
    corr_h, corr_w = KX.shape
    P = corr_h * corr_w
    kx_axis = KX[0, :]
    ky_axis = KY[:, 0]
    KX2 = (KX * KX).ravel()
    KY2 = (KY * KY).ravel()
    KXf = KX.ravel()
    KYf = KY.ravel()

    status = np.full(n, STATUS_NO_CONVERGE, dtype=np.int32)

    # ---- amplitudes + low-SNR gate (C step 0) ----
    amp_A = R_AA[:, cy, cx].astype(np.float64)
    amp_B = R_BB[:, cy, cx].astype(np.float64)
    amp_AB = R_AB.reshape(n, -1).max(axis=1).astype(np.float64)
    viable = (amp_A >= 1e-12) & (amp_B >= 1e-12)
    status[~viable] = STATUS_LOW_SNR

    # ---- step 1: FFTs ----
    F_AA = _fft_planes(R_AA, corr_h, corr_w)
    F_BB = _fft_planes(R_BB, corr_h, corr_w)
    F_AB = _fft_planes(R_AB, corr_h, corr_w)

    # ---- step 2: reference spectrum + noise PSD ----
    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))                 # (n, h, w)
    Pn2d = compute_noise_psd_2d(KX, KY, f_xy[0], f_xy[1], kernel=kernel)
    Pn = Pn2d.ravel()

    F_dc = F_ref[:, cy, cx]
    dc_ok = F_dc >= 1e-10
    status[viable & ~dc_ok] = STATUS_LOW_SNR
    viable &= dc_ok

    # ---- step 3: stage-1 joint floor fit (beta-free), flat weights, all pixels ----
    F_ref_norm = F_ref.reshape(n, -1) / np.maximum(F_dc, 1e-30)[:, None]
    s1_conv = np.zeros(n, dtype=bool)
    s1_iter = np.zeros(n, dtype=np.int32)
    N0_abs = np.zeros(n)
    v_idx = np.flatnonzero(viable)
    if v_idx.size:
        Yv = F_ref_norm[v_idx]

        def s1_fn(x, idx, jac=False):
            return _stage1_resid_jac(x, Yv[idx], Pn, KX2, KY2, jac=jac)

        x0 = np.broadcast_to(JOINT_P0, (v_idx.size, 4)).copy()
        xs1, conv1, cost1, it1 = _batched_lm(s1_fn, x0, JOINT_LO, JOINT_HI,
                                             JOINT_MAX_ITER)
        s1_conv[v_idx] = conv1
        s1_iter[v_idx] = it1
        # C acceptance: SUCCESS or EMAXITER; ENOPROG (stuck early, unconverged) rejected
        s1_ok = conv1 | (it1 >= JOINT_MAX_ITER)
        status[v_idx[~s1_ok]] = STATUS_NO_CONVERGE
        N0_abs[v_idx] = np.where(s1_ok, np.maximum(xs1[:, 3], 0.0) * F_dc[v_idx], 0.0)
        viable[v_idx[~s1_ok]] = False

    # subtract the coloured floor (only meaningful for viable windows; harmless else)
    eps_sub = (F_dc * 1e-8)[:, None, None]
    F_ref = np.maximum(F_ref - N0_abs[:, None, None] * Pn2d[None], eps_sub)

    # ---- step 4: SNR diagnostics (informational, no gate — as in the C) ----
    F_dc_clean = F_ref[:, cy, cx]
    noise_power = N0_abs ** 2 + 1e-12
    snr = F_dc_clean ** 2 / noise_power

    # ---- step 5: per-axis k_max from the profile threshold ----
    prof_ref_x = F_ref[:, cy, :]                                 # |.| implicit: F_ref >= 0
    prof_ref_y = F_ref[:, :, cx]
    k_max_x = _profile_kmax(prof_ref_x, kx_axis, F_dc_clean, k_max_limit)
    k_max_y = _profile_kmax(prof_ref_y, ky_axis, F_dc_clean, k_max_limit)

    # ---- step 6: seeds ----
    mu_x0, mu_y0 = _peak_mu(R_AB.astype(np.float64, copy=False), cy, cx)
    fab_x = np.abs(F_AB[:, cy, :])
    fab_y = np.abs(F_AB[:, :, cx])
    Sxx0 = _seed_sigma_1d(fab_x, prof_ref_x, kx_axis, k_max_x)
    Syy0 = _seed_sigma_1d(fab_y, prof_ref_y, ky_axis, k_max_y)
    seed = np.stack([mu_x0, mu_y0, Sxx0, Syy0, np.zeros(n)], axis=1)

    # ---- step 7 prep: T_norm + elliptical mask + weights ----
    eps = np.maximum(F_dc_clean, 1.0) * 1e-8
    T = F_AB / (F_ref + eps[:, None, None])
    T0 = T[:, cy, cx]
    t0_ok = np.abs(T0) >= 1e-6
    status[viable & ~t0_ok] = STATUS_NO_CONVERGE
    viable &= t0_ok
    T0_safe = np.where(t0_ok, T0, 1.0)
    Tn = (T * (np.conj(T0_safe) / (np.abs(T0_safe) ** 2))[:, None, None]).reshape(n, -1)

    ell = (KX2[None] / (k_max_x ** 2)[:, None]
           + KY2[None] / (k_max_y ** 2)[:, None]) <= 1.0         # (n, P)
    n_valid = ell.sum(axis=1)
    nv_ok = n_valid >= MIN_VALID_PTS
    status[viable & ~nv_ok] = STATUS_NO_CONVERGE
    viable &= nv_ok

    if use_soft_weighting:
        w_snr = np.where(ell, F_ref.reshape(n, -1), 0.0) / (np.sqrt(noise_power)
                                                            + 1e-12)[:, None]
        w_max = w_snr.max(axis=1)
        w_snr /= np.where(w_max > 1e-12, w_max, 1.0)[:, None]
        k0x2 = 1.0 / (_TWO_PI2 * np.maximum(Sxx0, 0.01) + 1e-12)  # (n,)
        k0y2 = 1.0 / (_TWO_PI2 * np.maximum(Syy0, 0.01) + 1e-12)
        w_soft = np.exp(-KX2[None] / k0x2[:, None] - KY2[None] / k0y2[:, None])
        W = w_snr * w_soft
    else:
        W = ell.astype(np.float64)
    W = np.where(ell, W, 0.0)

    return dict(
        n=n, corr_h=corr_h, corr_w=corr_w, KXf=KXf, KYf=KYf,
        amps=np.stack([amp_A, amp_B, amp_AB], axis=1),
        status=status, viable=viable, seed=seed,
        Tre=np.real(Tn), Tim=np.imag(Tn), W=W, n_valid=n_valid,
        snr=snr, N0_abs=N0_abs, k_max_x=k_max_x, k_max_y=k_max_y,
        s1_conv=s1_conv, s1_iter=s1_iter,
    )


def _run_stage2(prep):
    """Batched stage-2 LM on the viable windows of a prepared chunk (C steps 7-8)."""
    n = prep["n"]
    corr_h, corr_w = prep["corr_h"], prep["corr_w"]
    status = prep["status"]
    mu = np.zeros((n, 2))
    Sigma = np.zeros((n, 3))
    s2_conv = np.zeros(n, dtype=bool)
    s2_iter = np.zeros(n, dtype=np.int32)
    s2_cost = np.full(n, np.nan)

    v_idx = np.flatnonzero(prep["viable"])
    if v_idx.size:
        Tre, Tim, W = prep["Tre"][v_idx], prep["Tim"][v_idx], prep["W"][v_idx]
        KXf, KYf = prep["KXf"], prep["KYf"]

        def s2_fn(x, idx, jac=False):
            return _stage2_resid_jac(x, Tre[idx], Tim[idx], W[idx], KXf, KYf, jac=jac)

        xs, conv, cost, it = _batched_lm(s2_fn, prep["seed"][v_idx], MAIN_LO, MAIN_HI,
                                         MAIN_MAX_ITER)
        s2_conv[v_idx] = conv
        s2_iter[v_idx] = it
        s2_cost[v_idx] = cost / prep["n_valid"][v_idx]
        # C acceptance: SUCCESS, or EMAXITER with cost/n_valid < 1; ENOPROG rejected
        fit_ok = conv | ((it >= MAIN_MAX_ITER)
                         & (cost / prep["n_valid"][v_idx] < COST_PER_PT_ACCEPT))

        mu_f = xs[:, 0:2]
        Sxx_f = np.maximum(xs[:, 2], 0.0)          # projection keeps >= 0; belt+braces
        Syy_f = np.maximum(xs[:, 3], 0.0)
        big = ((np.abs(mu_f[:, 0]) > MAX_DISP_FRAC * corr_w)
               | (np.abs(mu_f[:, 1]) > MAX_DISP_FRAC * corr_h))

        st = np.full(v_idx.size, STATUS_NO_CONVERGE, dtype=np.int32)
        st[fit_ok] = STATUS_SUCCESS
        st[fit_ok & big] = STATUS_BIG_DISP
        status[v_idx] = st
        good = fit_ok & ~big
        mu[v_idx[good]] = mu_f[good]
        Sigma[v_idx[good]] = np.stack([Sxx_f, Syy_f, xs[:, 4]], axis=1)[good]

    return mu, Sigma, status, s2_conv, s2_iter, s2_cost


# ==========================================================================================
# Public entry point — same contract as the dormant fit_windows_kspace_linear
# ==========================================================================================
def fit_windows_kspace_lm(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    corr_size: tuple,
    config,
    pass_idx: int,
    use_soft_weighting: bool = True,
    debug: bool = False,
    predictor_displacements: Optional[np.ndarray] = None,
    interp_kernel: str = "bicubic",
    k_max_cap: Optional[float] = None,
    return_diagnostics: bool = False,
):
    """Batched-LM GSL-replica fit (beta-free). Same contract as fit_windows_kspace_linear.

    Returns ``(gauss_flat[n,16], status_flat[n], initial_guess_flat[n,16])`` and, when
    ``return_diagnostics=True``, a fourth element: a dict of per-window arrays
    (snr, N0_abs, k_max_x, k_max_y, s1/s2 convergence + iterations, s2 cost/point).
    """
    corr_h, corr_w = corr_size
    n_windows = len(mask_flat)
    n_per = corr_h * corr_w
    if R_AA.size != n_windows * n_per:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {n_windows * n_per} "
            f"(n_windows={n_windows}, corr_size={corr_size})"
        )

    R_AA = np.ascontiguousarray(R_AA, dtype=np.float64).reshape(n_windows, corr_h, corr_w)
    R_BB = np.ascontiguousarray(R_BB, dtype=np.float64).reshape(n_windows, corr_h, corr_w)
    R_AB = np.ascontiguousarray(R_AB, dtype=np.float64).reshape(n_windows, corr_h, corr_w)
    mask = np.asarray(mask_flat, dtype=bool)

    KX, KY, _ = _kgrids(corr_h, corr_w)
    cy, cx = corr_h // 2, corr_w // 2
    k_max_limit = float(k_max_cap) if (k_max_cap is not None and k_max_cap > 0) \
        else K_MAX_DEFAULT

    if predictor_displacements is not None:
        pred = np.asarray(predictor_displacements, dtype=np.float64).reshape(-1, 2)
        f_y = np.nan_to_num(frac_distance(pred[:, 0] / 2.0), nan=0.0)
        f_x = np.nan_to_num(frac_distance(pred[:, 1] / 2.0), nan=0.0)
    else:
        f_y = np.zeros(n_windows)
        f_x = np.zeros(n_windows)

    gauss_flat = np.zeros((n_windows, 16), dtype=np.float64)
    status_flat = np.full(n_windows, STATUS_MASKED, dtype=np.int32)
    initial_guess_flat = np.zeros((n_windows, 16), dtype=np.float64)
    diag = {k: np.full(n_windows, np.nan) for k in
            ("snr", "N0_abs", "k_max_x", "k_max_y", "s2_cost_per_pt")}
    diag.update({k: np.zeros(n_windows, dtype=np.int32) for k in ("s1_iter", "s2_iter")})
    diag.update({k: np.zeros(n_windows, dtype=bool) for k in ("s1_conv", "s2_conv")})
    diag["n_valid"] = np.zeros(n_windows, dtype=np.int32)

    center_x = corr_w / 2.0 + 1.0                  # 1-based centres (C convention)
    center_y = corr_h / 2.0 + 1.0
    gauss_flat[:, 6:9] = np.nan                    # particle-size slots: NaN by contract
    gauss_flat[:, 12] = center_x
    gauss_flat[:, 13] = center_y
    gauss_flat[:, 14] = center_x
    gauss_flat[:, 15] = center_y

    proc = np.where(~mask)[0]
    if proc.size == 0:
        logger.info(f"Pass {pass_idx + 1}: k-space-LM, all {n_windows} windows masked")
        initial_guess_flat[:] = gauss_flat
        if return_diagnostics:
            return gauss_flat, status_flat, initial_guess_flat, diag
        return gauss_flat, status_flat, initial_guess_flat

    CHUNK = 4096
    fkey = np.round(np.stack([f_x[proc], f_y[proc]], axis=1), 4)
    order = np.lexsort((fkey[:, 1], fkey[:, 0]))
    proc_sorted = proc[order]
    fkey_sorted = fkey[order]

    start = 0
    Ns = proc_sorted.size
    with _get_threadpool_controller().limit(limits=1):
        while start < Ns:
            end = min(start + CHUNK, Ns)
            same = np.all(fkey_sorted[start:end] == fkey_sorted[start], axis=1)
            if not np.all(same):
                end = start + int(np.argmin(same))
            idx = proc_sorted[start:end]
            f_xy = (float(fkey_sorted[start, 0]), float(fkey_sorted[start, 1]))

            prep = _prepare_chunk(R_AA[idx], R_BB[idx], R_AB[idx], KX, KY, cy, cx,
                                  f_xy, interp_kernel, use_soft_weighting, k_max_limit)
            mu, Sigma, status, s2_conv, s2_iter, s2_cost = _run_stage2(prep)

            gauss_flat[idx, 0:3] = prep["amps"]
            gauss_flat[idx, 9] = Sigma[:, 0]
            gauss_flat[idx, 10] = Sigma[:, 1]
            gauss_flat[idx, 11] = Sigma[:, 2]
            gauss_flat[idx, 14] = center_x + mu[:, 0]
            gauss_flat[idx, 15] = center_y + mu[:, 1]
            status_flat[idx] = status

            initial_guess_flat[idx, 0:3] = prep["amps"]
            initial_guess_flat[idx, 6:9] = np.nan
            initial_guess_flat[idx, 9] = prep["seed"][:, 2]
            initial_guess_flat[idx, 10] = prep["seed"][:, 3]
            initial_guess_flat[idx, 12] = center_x
            initial_guess_flat[idx, 13] = center_y
            initial_guess_flat[idx, 14] = center_x + prep["seed"][:, 0]
            initial_guess_flat[idx, 15] = center_y + prep["seed"][:, 1]

            for k in ("snr", "N0_abs", "k_max_x", "k_max_y"):
                diag[k][idx] = prep[k]
            diag["n_valid"][idx] = prep["n_valid"]
            diag["s1_conv"][idx] = prep["s1_conv"]
            diag["s1_iter"][idx] = prep["s1_iter"]
            diag["s2_conv"][idx] = s2_conv
            diag["s2_iter"][idx] = s2_iter
            diag["s2_cost_per_pt"][idx] = s2_cost
            start = end

    n_ok = int(np.sum(status_flat == STATUS_SUCCESS))
    logger.info(
        f"Pass {pass_idx + 1}: k-space-LM fit {n_ok}/{proc.size} ok "
        f"(soft_weighting={use_soft_weighting}, k_max_cap={k_max_limit:.2f})"
    )
    if debug and n_ok:
        ok = status_flat == STATUS_SUCCESS
        logger.info(
            f"  Sigma_xx median={np.nanmedian(gauss_flat[ok, 9]):.4f} "
            f"Sigma_yy median={np.nanmedian(gauss_flat[ok, 10]):.4f}"
        )

    if return_diagnostics:
        return gauss_flat, status_flat, initial_guess_flat, diag
    return gauss_flat, status_flat, initial_guess_flat
