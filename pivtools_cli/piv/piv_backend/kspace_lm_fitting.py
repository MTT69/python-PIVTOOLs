"""
Batched Levenberg-Marquardt k-space fitter: one-stage 7-parameter joint fit.

PRODUCTION ensemble fitter. Promoted 2026-07-14, replacing the two-stage GSL-replica
design (stage-1 LM noise-floor fit + stage-2 transfer-function fit) this module
carried since 2026-07-08: the stage-1 flat-weight floor over-fit N0 by ~2.4x on
experimental planes; over-subtraction flattened the ln|T| slope and collapsed the
k_max trust ellipse, under-reading Reynolds stresses (uu -20%, vv -50%, uv ~0 on the
Ashley Cam4 validation run vs hot wire). Root cause + the 8-variant offline lab that
selected this model: wiki/sessions/2026-07-14-noise-floor-overfit-v6-one-stage.md.
The two-stage predecessor's own lineage (GSL C port minus beta, DC-free anchoring +
gain g): wiki/sessions/2026-07-07-lm-gsl-replica-fitter.md and
2026-07-13-window-mean-dc-zero-lm-divergence.md; recover the code from git history.

Model, fitted per window to the RAW complex ratio T(k) = F_AB / sqrt(|F_AA| |F_BB|)
on every bin except DC — no noise-floor subtraction, no pre-fit stage, no trust
ellipse:

    T_hat(k) = g * exp(-2 pi^2 k^T Sigma k) * exp(-2 pi i k.mu) * (1 - N0/F_ref(k))

7 parameters (mu_x, mu_y, Sxx, Syy, Sxy, g, N0):
  * mu    — mean displacement (phase ramp).
  * Sigma — displacement covariance = Reynolds stresses in px^2 (Gaussian decay).
  * g     — real peak gain (loss-of-correlation): the low-k amplitude plateau of
            experimental planes that a forced T(0)=1 books as fake Sigma (4-10x
            inflation measured 2026-07-13).
  * N0    — additive auto-spectrum noise floor, in-model as the coherence
            attenuation (1 - N0/F_ref): if the autos carry an additive floor that
            the complex-averaged cross does not, then T = physics * S/(S+N0)
            = physics * (1 - N0/F_ref) with the MEASURED F_ref as regressor — the
            textbook coherence noise-bias correction. The floor level is identified
            by the high-k bins where the attenuation dominates. Fitting it jointly
            removes the subtractive design's failure mode outright: no denominator
            is ever deflated, so floor-driven blow-ups are structurally impossible.

Weighting: ONE measured inverse-uncertainty weight, w(k) proportional to F_ref(k)
(sigma_T ~ 1/F_ref), max-normalised per window. The DC bin is the only hard
exclusion (zero weight): ``window_mean`` background subtraction zeroes it exactly,
and it is a brightness-covariance value otherwise. There is no soft-weight taper,
no k_max ellipse, no |T|/refc fence — each of the old fences guarded a pathology of
the subtractive two-stage design that this model does not have.

Pipeline per window:
  0. Viability: R_AA/R_BB centre amplitude >= 1e-12, else LOW_SNR.
  1. FFT R_AA/R_BB/R_AB (centred, fftshift(fft2(ifftshift(.)))); F_ref =
     sqrt(|F_AA|*|F_BB|); T = F_AB/F_ref.
  2. valid bins = (F_ref > 1e-12) minus the DC bin; require >= MIN_VALID_PTS.
  3. Seeds: 3-point log-Gaussian sub-pixel peak (mu); Sigma (1.0, 0.2, 0); g = ring-1
     median of |T| clipped [1e-2, 10]; N0 = 0 (the joint fit identifies the floor
     from the tail bins regardless of its seed — validated on the full grid).
  4. Batched LM (Marquardt damping, box by step projection). Accept converged-by-tol
     OR (max_iter reached AND cost/n_valid < 1). Clamp Sxx, Syy >= 0; reject
     |mu| > 0.75*window (status 3).

g and N0 are reported only in the diagnostics dict ('gain', 'N0'), never in the
16-column ``gauss_flat`` output — that contract is unchanged (Sigma cols 9-11, mu as
centre offsets in 14-15, particle-size slots 6:9 NaN), same as the dormant
``fit_windows_kspace_linear``.

Validation trail (full numbers in the 2026-07-14 wiki session page): full validation
pass-4 grid 5457/5457 data-bearing windows converged, median 5-8 LM iterations;
calibrated vs hot wire uu+ -22% -> -8%, vv+ 0.50 -> 0.80 (instantaneous upper bound
1.15), uv unchanged and smooth (the two-stage Sxy=0 spikes vanish); mu shift vs the
two-stage fit 0.010 px median.

Parallelism: pure NumPy, vectorised over windows in fixed-size chunks; native
threadpools pinned to 1 (the accumulator submits one window block per Dask worker).
The warp-noise MTF (interpolation_noise_psd) left with stage 1 — the floor is
in-model and flat (the lab's anisotropic-floor variant showed the floor's shape is
irrelevant, only its level matters) — so windows no longer chunk by fractional
predictor shift and the predictor/interp-kernel/soft-weighting/k_max arguments are
gone from the signature.
"""

import logging

import numpy as np

from pivtools_cli.piv.piv_backend.kspace_common import (
    _fft_planes,
    _get_threadpool_controller,
    _kgrids,
)

logger = logging.getLogger(__name__)

# --- constants ---------------------------------------------------------------------------
MAIN_MAX_ITER = 250  # LM iteration cap
LM_XTOL = 1e-8  # step-size convergence (GSL XTOL)
LM_FTOL = 1e-8  # cost-reduction convergence (GSL FTOL)
MIN_VALID_PTS = 10  # minimum usable k-bins per window
MAX_DISP_FRAC = 0.75  # |mu| gate as fraction of window size
COST_PER_PT_ACCEPT = 1.0  # EMAXITER acceptance: cost/n_valid < this

# (mu_x, mu_y, Sxx, Syy, Sxy, g, N0): Sxx, Syy >= 0; gain g in [1e-3, 1e3]; N0 >= 0
MAIN_LO = np.array([-np.inf, -np.inf, 0.0, 0.0, -np.inf, 1e-3, 0.0])
MAIN_HI = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1e3, np.inf])

# Sigma seed (Sxx0, Syy0) in px^2; Sxy seeds 0. Seeds steer convergence only (the
# optimum is set by the residuals); these values converged in 5-8 iterations across
# the full validation grid.
SIGMA_SEED = (1.0, 0.2)

STATUS_MASKED = -1
STATUS_SUCCESS = 0
STATUS_NO_CONVERGE = 1
STATUS_LOW_SNR = 2
STATUS_BIG_DISP = 3
STATUS_NEG_VAR = 5  # kept for contract parity (defined but never set, as in the C)

_TWO_PI = 2.0 * np.pi
_TWO_PI2 = 2.0 * np.pi**2

# Sxy is unbounded (MAIN_LO/HI), so an LM trial step can drive Sxy^2 > Sxx*Syy, making
# the covariance Sigma indefinite and quad = k^T Sigma k negative at high k. Then
# decay = exp(-2pi^2 * quad) = exp(+big) overflows to inf, which poisons the residual and
# Jacobian (NaN) and spuriously fails flat/low-SNR windows. Clip the exponent: for feasible
# steps (quad >= 0 -> arg <= 0) this never triggers, so converged fits are bit-identical; on
# a bad step the residual/gradient stay finite and large, so LM rejects and backtracks cleanly.
_EXP_ARG_MAX = (
    300.0  # exp(300) ~ 2e130: huge (forces rejection) yet far below float64 overflow
)


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
        acc = (
            np.isfinite(costn) & (costn <= cost[idx]) & np.all(np.isfinite(xn), axis=1)
        )

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
        active[rej[lam[rej] > 1e12]] = False  # ENOPROG: stuck, give up unconverged

    return x, conv, cost, niter


# ==========================================================================================
# Residuals + analytic Jacobian (the one-stage joint model)
# ==========================================================================================
def _resid_jac_v6(x, Tre, Tim, W, Fr, KXf, KYf, jac=False):
    """One-stage joint fit of the raw transfer ratio. x (m,7) = (mu_x, mu_y, Sxx,
    Syy, Sxy, g, N0).

    Model = g * decay * att * exp(i*phase) with att = 1 - N0/F_ref (coherence
    noise-bias attenuation on the MEASURED reference spectrum ``Fr``).
    Residuals (m, 2P): [W*(Tre - model_re), W*(Tim - model_im)]. Sxx/Syy arrive
    >= 0, g > 0 and N0 >= 0 (projected box); no residual-side clamp needed.
    """
    mux = x[:, 0:1]
    muy = x[:, 1:2]
    Sxx = x[:, 2:3]
    Syy = x[:, 3:4]
    Sxy = x[:, 4:5]
    g = x[:, 5:6]
    N0 = x[:, 6:7]
    KX2 = KXf * KXf
    KY2 = KYf * KYf
    KXKY = KXf * KYf
    quad = Sxx * KX2[None] + 2.0 * Sxy * KXKY[None] + Syy * KY2[None]  # (m, P)
    # clip guards against non-PSD trial steps (quad<0); no-op for feasible steps (see _EXP_ARG_MAX)
    decay = np.exp(np.minimum(-_TWO_PI2 * quad, _EXP_ARG_MAX))
    att = 1.0 - N0 / np.maximum(Fr, 1e-30)
    phase = -_TWO_PI * (KXf[None] * mux + KYf[None] * muy)
    cosp = np.cos(phase)
    sinp = np.sin(phase)
    gd = g * decay * att
    r = np.concatenate([W * (Tre - gd * cosp), W * (Tim - gd * sinp)], axis=1)
    if not jac:
        return r
    m, P = decay.shape
    J = np.empty((m, 2 * P, 7))
    dpx = -_TWO_PI * KXf[None]  # dphase/dmu_x
    dpy = -_TWO_PI * KYf[None]
    J[:, :P, 0] = -W * gd * (-sinp) * dpx
    J[:, P:, 0] = -W * gd * cosp * dpx
    J[:, :P, 1] = -W * gd * (-sinp) * dpy
    J[:, P:, 1] = -W * gd * cosp * dpy
    for c, KK in ((2, KX2), (3, KY2), (4, 2.0 * KXKY)):
        dd = gd * (-_TWO_PI2 * KK[None])  # d(model amp)/dSigma_c
        J[:, :P, c] = -W * dd * cosp
        J[:, P:, c] = -W * dd * sinp
    J[:, :P, 5] = -W * decay * att * cosp  # dmodel/dg
    J[:, P:, 5] = -W * decay * att * sinp
    dN = g * decay * (-1.0 / np.maximum(Fr, 1e-30))  # dmodel/dN0
    J[:, :P, 6] = -W * dN * cosp
    J[:, P:, 6] = -W * dN * sinp
    return r, J


# ==========================================================================================
# Displacement seed (port of estimate_displacement_from_peak)
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

    ix = np.clip(px, 1, w - 2)  # safe gather; interior mask gates use
    sub_x = _subpix(
        R_AB[rows, py, ix - 1],
        R_AB[rows, py, ix],
        R_AB[rows, py, ix + 1],
        (px > 0) & (px < w - 1),
    )
    iy = np.clip(py, 1, h - 2)
    sub_y = _subpix(
        R_AB[rows, iy - 1, px],
        R_AB[rows, iy, px],
        R_AB[rows, iy + 1, px],
        (py > 0) & (py < h - 1),
    )
    return (px + sub_x) - cx, (py + sub_y) - cy


# ==========================================================================================
# Chunk pipeline: preparation shared by the batched fit and the oracle
# ==========================================================================================
def _prepare_chunk(R_AA, R_BB, R_AB, KX, KY, cy, cx):
    """Prepare one chunk for the joint fit: raw T, measured weights, seeds, gates.

    Everything the LM needs (T, weights, F_ref, seeds), plus per-window status for
    windows that fail a gate on the way.
    """
    n = R_AA.shape[0]
    corr_h, corr_w = KX.shape
    KXf = KX.ravel()
    KYf = KY.ravel()
    dc_flat = cy * corr_w + cx  # DC-free: this bin never enters a residual

    status = np.full(n, STATUS_NO_CONVERGE, dtype=np.int32)

    # ---- amplitudes + low-SNR gate ----
    amp_A = R_AA[:, cy, cx].astype(np.float64)
    amp_B = R_BB[:, cy, cx].astype(np.float64)
    amp_AB = R_AB.reshape(n, -1).max(axis=1).astype(np.float64)
    viable = (amp_A >= 1e-12) & (amp_B >= 1e-12)
    status[~viable] = STATUS_LOW_SNR

    # ---- FFTs + raw transfer ratio ----
    F_AA = _fft_planes(R_AA, corr_h, corr_w)
    F_BB = _fft_planes(R_BB, corr_h, corr_w)
    F_AB = _fft_planes(R_AB, corr_h, corr_w)
    Fr = np.sqrt(np.abs(F_AA) * np.abs(F_BB)).reshape(n, -1)  # (n, P)
    T = F_AB.reshape(n, -1) / np.maximum(Fr, 1e-30)

    # ---- valid bins: measurable reference, DC excluded ----
    valid = Fr > 1e-12
    valid[:, dc_flat] = False
    n_valid = valid.sum(axis=1)
    nv_ok = n_valid >= MIN_VALID_PTS
    viable &= nv_ok

    # ---- one measured weight: w ~ F_ref (sigma_T ~ 1/F_ref), max-normalised ----
    W = np.where(valid, Fr / np.maximum(Fr.max(axis=1), 1e-30)[:, None], 0.0)

    # ---- seeds ----
    mu_x0, mu_y0 = _peak_mu(R_AB.astype(np.float64, copy=False), cy, cx)
    ring = np.abs(T.reshape(n, corr_h, corr_w)[:, cy - 1 : cy + 2, cx - 1 : cx + 2])
    ring = np.delete(ring.reshape(n, 9), 4, axis=1)
    g0 = np.clip(np.median(ring, axis=1), 1e-2, 10.0)
    seed = np.stack(
        [
            mu_x0,
            mu_y0,
            np.full(n, SIGMA_SEED[0]),
            np.full(n, SIGMA_SEED[1]),
            np.zeros(n),
            g0,
            np.zeros(n),  # N0 seed 0: the tail bins pin the floor regardless
        ],
        axis=1,
    )

    return dict(
        n=n,
        corr_h=corr_h,
        corr_w=corr_w,
        KXf=KXf,
        KYf=KYf,
        amps=np.stack([amp_A, amp_B, amp_AB], axis=1),
        status=status,
        viable=viable,
        seed=seed,
        Tre=np.real(T),
        Tim=np.imag(T),
        W=W,
        Fr=Fr,
        n_valid=n_valid,
    )


def _run_fit(prep):
    """Batched joint LM on the viable windows of a prepared chunk."""
    n = prep["n"]
    corr_h, corr_w = prep["corr_h"], prep["corr_w"]
    status = prep["status"]
    mu = np.zeros((n, 2))
    Sigma = np.zeros((n, 3))
    gain = np.full(n, np.nan)
    N0 = np.full(n, np.nan)
    conv_out = np.zeros(n, dtype=bool)
    iter_out = np.zeros(n, dtype=np.int32)
    cost_out = np.full(n, np.nan)

    v_idx = np.flatnonzero(prep["viable"])
    if v_idx.size:
        Tre, Tim = prep["Tre"][v_idx], prep["Tim"][v_idx]
        W, Fr = prep["W"][v_idx], prep["Fr"][v_idx]
        KXf, KYf = prep["KXf"], prep["KYf"]

        def fn(x, idx, jac=False):
            return _resid_jac_v6(
                x, Tre[idx], Tim[idx], W[idx], Fr[idx], KXf, KYf, jac=jac
            )

        xs, conv, cost, it = _batched_lm(
            fn, prep["seed"][v_idx], MAIN_LO, MAIN_HI, MAIN_MAX_ITER
        )
        conv_out[v_idx] = conv
        iter_out[v_idx] = it
        cpp = cost / prep["n_valid"][v_idx]
        cost_out[v_idx] = cpp
        # accept: converged, or EMAXITER with cost/n_valid < 1; ENOPROG rejected
        fit_ok = conv | ((it >= MAIN_MAX_ITER) & (cpp < COST_PER_PT_ACCEPT))

        mu_f = xs[:, 0:2]
        Sxx_f = np.maximum(xs[:, 2], 0.0)  # projection keeps >= 0; belt+braces
        Syy_f = np.maximum(xs[:, 3], 0.0)
        big = (np.abs(mu_f[:, 0]) > MAX_DISP_FRAC * corr_w) | (
            np.abs(mu_f[:, 1]) > MAX_DISP_FRAC * corr_h
        )

        st = np.full(v_idx.size, STATUS_NO_CONVERGE, dtype=np.int32)
        st[fit_ok] = STATUS_SUCCESS
        st[fit_ok & big] = STATUS_BIG_DISP
        status[v_idx] = st
        good = fit_ok & ~big
        mu[v_idx[good]] = mu_f[good]
        Sigma[v_idx[good]] = np.stack([Sxx_f, Syy_f, xs[:, 4]], axis=1)[good]
        gain[v_idx[good]] = xs[good, 5]
        N0[v_idx[good]] = xs[good, 6]

    return mu, Sigma, gain, N0, status, conv_out, iter_out, cost_out


# ==========================================================================================
# Public entry point — same 16-column contract as the dormant fit_windows_kspace_linear
# ==========================================================================================
def fit_windows_kspace_lm(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    corr_size: tuple,
    config,
    pass_idx: int,
    debug: bool = False,
    return_diagnostics: bool = False,
):
    """One-stage 7-parameter joint LM fit (see module docstring for the model).

    Returns ``(gauss_flat[n,16], status_flat[n], initial_guess_flat[n,16])`` and, when
    ``return_diagnostics=True``, a fourth element: a dict of per-window arrays
    (gain = fitted peak gain g, N0 = fitted noise floor in F_ref units, cost_per_pt,
    n_valid, conv, iter).
    """
    corr_h, corr_w = corr_size
    n_windows = len(mask_flat)
    n_per = corr_h * corr_w
    if R_AA.size != n_windows * n_per:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {n_windows * n_per} "
            f"(n_windows={n_windows}, corr_size={corr_size})"
        )

    R_AA = np.ascontiguousarray(R_AA, dtype=np.float64).reshape(
        n_windows, corr_h, corr_w
    )
    R_BB = np.ascontiguousarray(R_BB, dtype=np.float64).reshape(
        n_windows, corr_h, corr_w
    )
    R_AB = np.ascontiguousarray(R_AB, dtype=np.float64).reshape(
        n_windows, corr_h, corr_w
    )
    mask = np.asarray(mask_flat, dtype=bool)

    KX, KY, _ = _kgrids(corr_h, corr_w)
    cy, cx = corr_h // 2, corr_w // 2

    gauss_flat = np.zeros((n_windows, 16), dtype=np.float64)
    status_flat = np.full(n_windows, STATUS_MASKED, dtype=np.int32)
    initial_guess_flat = np.zeros((n_windows, 16), dtype=np.float64)
    diag = {
        k: np.full(n_windows, np.nan)
        for k in (
            "gain",  # fitted peak gain g (loss-of-correlation factor)
            "N0",  # fitted noise floor (F_ref units)
            "cost_per_pt",
        )
    }
    diag["n_valid"] = np.zeros(n_windows, dtype=np.int32)
    diag["iter"] = np.zeros(n_windows, dtype=np.int32)
    diag["conv"] = np.zeros(n_windows, dtype=bool)

    center_x = corr_w / 2.0 + 1.0  # 1-based centres (C convention)
    center_y = corr_h / 2.0 + 1.0
    gauss_flat[:, 6:9] = np.nan  # particle-size slots: NaN by contract
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
    with _get_threadpool_controller().limit(limits=1):
        for start in range(0, proc.size, CHUNK):
            idx = proc[start : start + CHUNK]

            prep = _prepare_chunk(R_AA[idx], R_BB[idx], R_AB[idx], KX, KY, cy, cx)
            mu, Sigma, gain, N0, status, conv, niter, cpp = _run_fit(prep)

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

            diag["gain"][idx] = gain
            diag["N0"][idx] = N0
            diag["cost_per_pt"][idx] = cpp
            diag["n_valid"][idx] = prep["n_valid"]
            diag["conv"][idx] = conv
            diag["iter"][idx] = niter

    n_ok = int(np.sum(status_flat == STATUS_SUCCESS))
    logger.info(f"Pass {pass_idx + 1}: k-space-LM joint fit {n_ok}/{proc.size} ok")
    if debug and n_ok:
        ok = status_flat == STATUS_SUCCESS
        logger.info(
            f"  Sigma_xx median={np.nanmedian(gauss_flat[ok, 9]):.4f} "
            f"Sigma_yy median={np.nanmedian(gauss_flat[ok, 10]):.4f} "
            f"gain median={np.nanmedian(diag['gain'][ok]):.3f} "
            f"N0 median={np.nanmedian(diag['N0'][ok]):.3f}"
        )

    if return_diagnostics:
        return gauss_flat, status_flat, initial_guess_flat, diag
    return gauss_flat, status_flat, initial_guess_flat
