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

Coloured noise floor (``ensemble_piv.kspace_floor``, added 2026-07-24, default
``coloured``): the attenuation generalises to (1 - N0 * P(k;fx,fy)/F_ref) with P
the ANALYTIC pipeline colour of a white sensor floor (warp-kernel tap
autocorrelation x window-mean DC hole x envelope divide — ``kspace_floor_psd``),
evaluated per window at (fx, fy) = frac(pred/2). The flat-level assumption was
the last wrong one in the chain: the floor of F_ref is jagged in ky at ALL f
(ring-1 +30-65 %), which flat N0 books as fake vv (noisy vv+ 1.37x of DNS ->
1.02x with the coloured floor; centre uu 1.77x -> 0.99x). P carries SHAPE only
(normalized to unit mean at f=0); the per-window fitted N0 remains the level.
``flat`` preserves the pre-promotion behaviour on a byte-identical code path.
Derivation + validation: wiki/pipelines/ensemble-piv-fitting.md and
wiki/sessions/2026-07-22-noise-floor-colouring-psim.md.

Optional quartic shape terms (``ensemble_piv.kspace_shape``, added 2026-07-17): the
Gaussian exponent may be extended with free-signed per-axis quartic terms,

    exponent = -2 pi^2 k^T Sigma k - b4x*kx^4 - b4y*ky^4

selected by ``kspace_shape`` = ``gaussian`` (default, exactly the 7-parameter model
below) | ``kx4`` | ``ky4`` | ``kx4+ky4`` (8/8/9 parameters). Each b4 is the next
cumulant of the displacement PDF on that axis (excess kurtosis
gamma_2 = kappa_4/Sigma^2 with kappa_4 = -24*b4/(2 pi)^4): a non-Gaussian
displacement PDF curves ln|T|, and forcing a pure Gaussian through that curvature
biases Sigma by approximately gamma_2 * band_depth / 6 (+15-20 % streamwise below
y+ 400 on the channel benchmarks). The quartic term absorbs the curvature and
self-extinguishes (b4 -> 0) where the data has none, so no gating is applied. No
cross kx^2*ky^2 term (tested and rejected: streamwise curvature leaks into the
near-Gaussian transverse axis). Selection evidence — 10-fitter bake-off + full
profiles on clean/noisy synthetic and Cam4:
wiki/sessions/2026-07-17-fitter-bakeoff-e-debug.md.

7 base parameters (mu_x, mu_y, Sxx, Syy, Sxy, g, N0):
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
Windows do not chunk by fractional predictor shift: the coloured floor arrives as
the precomputed per-window ``P_win`` regressor (built by the accumulator from the
analytic grid + the pass predictor), so the fitter itself stays predictor-free.
The 2026-07-14 note that "the floor's shape is irrelevant, only its level
matters" was measured on the old subtractive design's k-band; the 2026-07-22
white-noise null through the exact pipeline showed the ky ring-1 colour is real
and vv-critical — hence ``kspace_floor``.
"""

import ctypes
import logging
import os

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

# Coloured floor (kspace_floor='coloured'): the offline free-arm configuration
# (refit_coloured_floor.py --arm free, validated 2026-07-22/23). N0 is bounded
# [0, 10] in normalized F_ref units and seeded from the tail median of Fr/P
# over |k| >= 0.35, where the floor dominates. The flat branch keeps seed 0 /
# [0, inf) untouched (bit-identity).
COLOURED_N0_HI = 10.0
COLOURED_SEED_KR_MIN = 0.35
COLOURED_SEED_CLIP = (1e-3, 10.0)

# kspace_shape -> (use_kx4, use_ky4). Enabled quartic coefficients append to the
# parameter vector after N0, b4x before b4y; both are free-signed (sub-Gaussian
# displacement PDF => b4 > 0) and seed at 0, mirroring N0.
_KSPACE_SHAPES = {
    "gaussian": (False, False),
    "kx4": (True, False),
    "ky4": (False, True),
    "kx4+ky4": (True, True),
}

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

# --- libkspacefit binding ------------------------------------------------------------------
# Loaded once per process (each Dask worker is a process, so each pays it once).
# There is no config knob and no NumPy fallback: a missing or stale library is a
# build error, not a reason to silently run a different implementation.
_KSPACE_LIB = None

_KSPACE_ERRORS = {
    -1: (
        "correlation-plane axis length outside BUILT_FFT_SIZES — no FFT codelet "
        "was generated for it"
    ),
    -2: "scratch allocation failure in the C fitter",
    -3: (
        "coloured floor requested but no |k| >= COLOURED_SEED_KR_MIN bin exists "
        "to seed N0 from"
    ),
}


def _load_kspace_lib():
    """Bind libkspacefit. Raises rather than degrading — see module note above."""
    global _KSPACE_LIB
    if _KSPACE_LIB is not None:
        return _KSPACE_LIB

    lib_ext = ".dll" if os.name == "nt" else ".so"  # macOS emits .so too
    lib_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "lib", f"libkspacefit{lib_ext}"
        )
    )
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(
            f"libkspacefit{lib_ext} not found at {lib_path}. The ensemble k-space "
            "fitter is a C extension — build it with `python setup.py build` "
            "before running ensemble PIV. (`pip install -e .` will NOT build it: "
            "PEP 660 editable installs skip the custom build command.)"
        )

    # Staleness guard. The sources are tracked; the library is gitignored. Pull a
    # commit that changes the model, skip the rebuild, and the old binary still
    # loads, still exports the symbol and still has the right signature — it just
    # computes the previous version of the maths. That is the silent-wrong-number
    # failure this module exists to make impossible, so check it explicitly.
    # Sources are absent in an installed wheel, where the check is meaningless.
    lib_dir = os.path.dirname(lib_path)
    lib_mtime = os.path.getmtime(lib_path)
    for src in ("kspace_lm_fit.c", "kspace_lm_fit.h"):
        src_path = os.path.join(lib_dir, src)
        if os.path.isfile(src_path) and os.path.getmtime(src_path) > lib_mtime:
            raise RuntimeError(
                f"{src} is newer than {os.path.basename(lib_path)} — the built "
                "library does not match the source and would silently compute the "
                "previous model. Rebuild with `python setup.py build`."
            )

    lib = ctypes.CDLL(lib_path)
    if not hasattr(lib, "kspace_lm_fit_batch"):
        raise RuntimeError(
            f"{lib_path} loaded but kspace_lm_fit_batch is missing from its export "
            "table — stale build. Rebuild with `python setup.py build`."
        )

    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    lib.kspace_lm_fit_batch.restype = ctypes.c_int
    lib.kspace_lm_fit_batch.argtypes = [
        f64, f64, f64,  # R_AA, R_BB, R_AB
        u8,  # mask_flat
        ctypes.c_void_p,  # P_win — NULL selects the flat floor
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_windows, corr_h, corr_w
        ctypes.c_int, ctypes.c_int,  # use_kx4, use_ky4
        ctypes.c_int,  # n_threads
        f64, i32, f64,  # gauss_flat, status_flat, initial_guess_flat
        f64, f64, f64, f64,  # diag: gain, N0, b4x, b4y
        f64, i32, i32, u8,  # diag: cost_per_pt, n_valid, iter, conv
    ]
    lib.kspace_lm_fit_max_threads.restype = ctypes.c_int

    # Actually call it. ctypes.CDLL binds lazily on Linux/macOS, so a library
    # built against an unresolvable libomp loads fine here and would instead die
    # on the first kspace_lm_fit_batch — inside a Dask worker, mid-run, after
    # hours of accumulation. One call forces the resolution now.
    max_threads = lib.kspace_lm_fit_max_threads()
    logger.debug(
        "libkspacefit loaded from %s (OpenMP max threads %d)", lib_path, max_threads
    )

    _KSPACE_LIB = lib
    return lib


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
def _resid_jac_v6(x, Tre, Tim, W, Fr, KXf, KYf, jac=False, D=None):
    """One-stage joint fit of the raw transfer ratio. x (m,7) = (mu_x, mu_y, Sxx,
    Syy, Sxy, g, N0).

    Model = g * decay * att * exp(i*phase) with att = 1 - N0/F_ref (coherence
    noise-bias attenuation on the MEASURED reference spectrum ``Fr``).
    Residuals (m, 2P): [W*(Tre - model_re), W*(Tim - model_im)]. Sxx/Syy arrive
    >= 0, g > 0 and N0 >= 0 (projected box); no residual-side clamp needed.

    ``D`` (m, P) switches the floor to the coloured model att = 1 - N0*D with
    D = P(k;fx,fy)/F_ref precomputed per chunk (kspace_floor='coloured');
    D=None keeps the flat floor on a byte-identical code path.
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
    att = 1.0 - N0 / np.maximum(Fr, 1e-30) if D is None else 1.0 - N0 * D
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
    # dmodel/dN0 (flat: -1/F_ref; coloured: -D)
    dN = g * decay * (-1.0 / np.maximum(Fr, 1e-30)) if D is None else g * decay * (-D)
    J[:, :P, 6] = -W * dN * cosp
    J[:, P:, 6] = -W * dN * sinp
    return r, J


def _resid_jac_quartic(
    x, Tre, Tim, W, Fr, KXf, KYf, use_kx4, use_ky4, jac=False, D=None
):
    """_resid_jac_v6 model plus free-signed quartic exponent terms (kspace_shape).

    x (m, 7+n4) = (mu_x, mu_y, Sxx, Syy, Sxy, g, N0, [b4x], [b4y]) with the enabled
    quartic coefficients appended in order (b4x before b4y). The exponent becomes
    -2 pi^2 quad - b4x*kx^4 - b4y*ky^4; everything else is identical to the
    Gaussian model, including the flat/coloured floor switch via ``D`` (see
    _resid_jac_v6). Only called when at least one term is enabled — the gaussian
    shape keeps the untouched _resid_jac_v6 path (bit-exact default by construction).
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
    quart = []  # (param column, k^4 grid) for the enabled terms
    col = 7
    if use_kx4:
        quart.append((col, KX2 * KX2))
        col += 1
    if use_ky4:
        quart.append((col, KY2 * KY2))
        col += 1
    quad = Sxx * KX2[None] + 2.0 * Sxy * KXKY[None] + Syy * KY2[None]  # (m, P)
    arg = -_TWO_PI2 * quad
    for c, K4 in quart:
        arg = arg - x[:, c : c + 1] * K4[None]
    # clip guards non-PSD trial steps and b4 < 0 blow-ups at high k (see _EXP_ARG_MAX)
    decay = np.exp(np.minimum(arg, _EXP_ARG_MAX))
    att = 1.0 - N0 / np.maximum(Fr, 1e-30) if D is None else 1.0 - N0 * D
    phase = -_TWO_PI * (KXf[None] * mux + KYf[None] * muy)
    cosp = np.cos(phase)
    sinp = np.sin(phase)
    gd = g * decay * att
    r = np.concatenate([W * (Tre - gd * cosp), W * (Tim - gd * sinp)], axis=1)
    if not jac:
        return r
    m, P = decay.shape
    J = np.empty((m, 2 * P, col))
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
    # dmodel/dN0 (flat: -1/F_ref; coloured: -D)
    dN = g * decay * (-1.0 / np.maximum(Fr, 1e-30)) if D is None else g * decay * (-D)
    J[:, :P, 6] = -W * dN * cosp
    J[:, P:, 6] = -W * dN * sinp
    for c, K4 in quart:
        db = gd * (-K4[None])  # dmodel/db4
        J[:, :P, c] = -W * db * cosp
        J[:, P:, c] = -W * db * sinp
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
def _prepare_chunk(R_AA, R_BB, R_AB, KX, KY, cy, cx, P=None):
    """Prepare one chunk for the joint fit: raw T, measured weights, seeds, gates.

    Everything the LM needs (T, weights, F_ref, seeds), plus per-window status for
    windows that fail a gate on the way.

    ``P`` (n, P_bins) is the per-window coloured-floor shape (kspace_floor=
    'coloured'): the prep gains ``D = P/max(F_ref, 1e-30)`` for the residuals
    and the N0 seed switches from 0 to the tail median of F_ref/P over
    |k| >= COLOURED_SEED_KR_MIN (the floor-dominated band). P=None (flat) is
    byte-identical to before.
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

    # ---- coloured floor: D regressor + tail N0 seed ----
    D = None
    if P is not None:
        D = P / np.maximum(Fr, 1e-30)
        tail = (KXf * KXf + KYf * KYf) >= COLOURED_SEED_KR_MIN**2
        if not tail.any():
            raise ValueError(
                f"no |k| >= {COLOURED_SEED_KR_MIN} bins in a "
                f"{corr_h}x{corr_w} window — cannot seed the coloured N0"
            )
        n0_seed = np.clip(
            np.median(Fr[:, tail] / np.maximum(P[:, tail], 1e-30), axis=1),
            *COLOURED_SEED_CLIP,
        )

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
            # N0 seed: 0 for flat (the tail bins pin the floor regardless);
            # tail median of Fr/P for coloured (offline free-arm recipe)
            np.zeros(n) if P is None else n0_seed,
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
        D=D,
        n_valid=n_valid,
    )


def _run_fit(prep, use_kx4=False, use_ky4=False):
    """Batched joint LM on the viable windows of a prepared chunk."""
    n = prep["n"]
    corr_h, corr_w = prep["corr_h"], prep["corr_w"]
    status = prep["status"]
    mu = np.zeros((n, 2))
    Sigma = np.zeros((n, 3))
    gain = np.full(n, np.nan)
    N0 = np.full(n, np.nan)
    b4x = np.full(n, np.nan)
    b4y = np.full(n, np.nan)
    conv_out = np.zeros(n, dtype=bool)
    iter_out = np.zeros(n, dtype=np.int32)
    cost_out = np.full(n, np.nan)

    v_idx = np.flatnonzero(prep["viable"])
    if v_idx.size:
        Tre, Tim = prep["Tre"][v_idx], prep["Tim"][v_idx]
        W, Fr = prep["W"][v_idx], prep["Fr"][v_idx]
        KXf, KYf = prep["KXf"], prep["KYf"]
        D = prep["D"]
        Dv = D[v_idx] if D is not None else None
        if Dv is None:
            # flat floor: MAIN_LO/HI objects untouched (bit-identity)
            lo7, hi7 = MAIN_LO, MAIN_HI
        else:
            # coloured floor: N0 bounded [0, COLOURED_N0_HI] (offline free arm)
            lo7 = MAIN_LO
            hi7 = MAIN_HI.copy()
            hi7[6] = COLOURED_N0_HI

        n4 = int(use_kx4) + int(use_ky4)
        if n4:
            # quartic modes: append free-signed b4 columns (seed 0, unbounded)
            def fn(x, idx, jac=False):
                return _resid_jac_quartic(
                    x, Tre[idx], Tim[idx], W[idx], Fr[idx], KXf, KYf,
                    use_kx4, use_ky4, jac=jac,
                    D=Dv[idx] if Dv is not None else None,
                )

            lo = np.concatenate([lo7, np.full(n4, -np.inf)])
            hi = np.concatenate([hi7, np.full(n4, np.inf)])
            seed = np.concatenate(
                [prep["seed"][v_idx], np.zeros((v_idx.size, n4))], axis=1
            )
        else:
            # gaussian: the untouched 7-parameter path, bit-identical to before
            def fn(x, idx, jac=False):
                return _resid_jac_v6(
                    x, Tre[idx], Tim[idx], W[idx], Fr[idx], KXf, KYf, jac=jac,
                    D=Dv[idx] if Dv is not None else None,
                )

            lo, hi = lo7, hi7
            seed = prep["seed"][v_idx]

        xs, conv, cost, it = _batched_lm(fn, seed, lo, hi, MAIN_MAX_ITER)
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
        col = 7
        if use_kx4:
            b4x[v_idx[good]] = xs[good, col]
            col += 1
        if use_ky4:
            b4y[v_idx[good]] = xs[good, col]

    return mu, Sigma, gain, N0, b4x, b4y, status, conv_out, iter_out, cost_out


# ==========================================================================================
# Public entry point — same 16-column contract as the dormant fit_windows_kspace_linear
# ==========================================================================================
def _validate_inputs(R_AA, R_BB, R_AB, mask_flat, corr_size, config, P_win):
    """Validate and normalise the fitter inputs.

    Shared by the production C entry point and the NumPy oracle so that a parity
    gate compares the two *fits*, not two input-handling paths.

    Returns ``(shape, use_kx4, use_ky4, floor, corr_h, corr_w, n_windows,
    R_AA, R_BB, R_AB, mask, P_win)`` with the planes reshaped to
    ``(n_windows, corr_h, corr_w)`` float64 C-contiguous and ``P_win`` to
    ``(n_windows, corr_h*corr_w)`` or ``None``.
    """
    shape = config.ensemble_kspace_shape
    use_kx4, use_ky4 = _KSPACE_SHAPES[shape]
    floor = config.ensemble_kspace_floor
    corr_h, corr_w = corr_size
    n_windows = len(mask_flat)
    n_per = corr_h * corr_w
    if R_AA.size != n_windows * n_per:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {n_windows * n_per} "
            f"(n_windows={n_windows}, corr_size={corr_size})"
        )
    if floor == "coloured":
        if P_win is None:
            raise ValueError(
                "kspace_floor='coloured' requires P_win (the per-window "
                "analytic floor shape) — the caller must build it via "
                "kspace_floor_psd.build_P_grid/interp_P"
            )
        if P_win.size != n_windows * n_per:
            raise ValueError(
                f"P_win size {P_win.size} != expected {n_windows * n_per} "
                f"(n_windows={n_windows}, corr_size={corr_size})"
            )
        P_win = np.ascontiguousarray(P_win, dtype=np.float64).reshape(
            n_windows, n_per
        )
    elif P_win is not None:
        raise ValueError(
            f"P_win passed but kspace_floor='{floor}' — wiring bug (the flat "
            "floor must not receive a P grid)"
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
    return (
        shape, use_kx4, use_ky4, floor, corr_h, corr_w, n_windows,
        R_AA, R_BB, R_AB, mask, P_win,
    )


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
    *,
    P_win: np.ndarray | None = None,
):
    """One-stage joint LM fit (see module docstring for the model).

    The fit itself runs in C (``libkspacefit``, one LM per window, OpenMP over
    windows). This function owns validation, logging and output assembly; the
    C owns the maths. ``fit_windows_kspace_lm_numpy`` below is the reference
    implementation it is gated against and is not on any production path.

    7 parameters for the default ``gaussian`` shape; ``config.ensemble_kspace_shape``
    (``kx4`` | ``ky4`` | ``kx4+ky4``) appends free-signed quartic exponent
    coefficients (8/8/9 parameters) that absorb displacement-PDF kurtosis.

    ``config.ensemble_kspace_floor`` selects the floor model: ``flat`` is the
    pre-2026-07 behaviour (scalar N0 on 1/F_ref); ``coloured`` switches to
    att = 1 - N0*P(k;fx,fy)/F_ref and REQUIRES ``P_win`` — the per-window
    analytic floor shape (n_windows, h*w) from ``kspace_floor_psd`` (built and
    interpolated by the caller, which holds the envelope/weights/predictor).
    Passing P_win with the flat floor raises (wiring-bug guard).

    ``P_win`` is handed to C in the CENTRED k-layout it arrives in; the C
    applies the ifftshift mapping internally.

    OpenMP width is ``config.omp_threads``, passed explicitly as a C argument and
    applied through a ``num_threads()`` clause. That clause takes precedence over
    the ``nthreads-var`` ICV, so this fitter is immune to a surrounding
    ``threadpool_limits`` context — unlike ``libbulkxcorr2d`` and
    ``libfusedwarp``, which carry no clause and would be clamped to one thread by
    one. It is also why the width is not left to ``OMP_NUM_THREADS``.

    Returns ``(gauss_flat[n,16], status_flat[n], initial_guess_flat[n,16])`` and, when
    ``return_diagnostics=True``, a fourth element: a dict of per-window arrays
    (gain = fitted peak gain g, N0 = fitted noise floor in F_ref units, cost_per_pt,
    n_valid, conv, iter, plus b4x/b4y when the shape enables them).
    """
    (
        shape, use_kx4, use_ky4, floor, corr_h, corr_w, n_windows,
        R_AA, R_BB, R_AB, mask, P_win,
    ) = _validate_inputs(
        R_AA, R_BB, R_AB, mask_flat, corr_size, config, P_win
    )

    lib = _load_kspace_lib()
    n_threads = max(1, int(config.omp_threads))

    # P_win crosses as a raw address (c_void_p, because NULL selects the flat
    # floor), so it is the one argument ndpointer does not police. Check what
    # ndpointer would have checked.
    if P_win is not None and not (
        P_win.flags["C_CONTIGUOUS"] and P_win.dtype == np.float64
    ):
        raise ValueError(
            f"P_win must be C-contiguous float64, got dtype={P_win.dtype} "
            f"contiguous={P_win.flags['C_CONTIGUOUS']}"
        )

    mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
    gauss_flat = np.zeros((n_windows, 16), dtype=np.float64)
    status_flat = np.full(n_windows, STATUS_MASKED, dtype=np.int32)
    initial_guess_flat = np.zeros((n_windows, 16), dtype=np.float64)
    # b4x/b4y are always allocated: the C fills them unconditionally and will
    # not accept NULL, even when the shape leaves them unused.
    d_gain = np.full(n_windows, np.nan)
    d_N0 = np.full(n_windows, np.nan)
    d_b4x = np.full(n_windows, np.nan)
    d_b4y = np.full(n_windows, np.nan)
    d_cpp = np.full(n_windows, np.nan)
    d_nvalid = np.zeros(n_windows, dtype=np.int32)
    d_iter = np.zeros(n_windows, dtype=np.int32)
    d_conv = np.zeros(n_windows, dtype=np.uint8)

    ret = lib.kspace_lm_fit_batch(
        R_AA, R_BB, R_AB,
        mask_u8,
        P_win.ctypes.data if P_win is not None else None,
        # int() on the sizes: corr_size arrives as numpy integers when it comes
        # from a .mat, and ctypes' c_int conversion of those is incidental
        # rather than guaranteed.
        int(n_windows), int(corr_h), int(corr_w),
        int(use_kx4), int(use_ky4),
        int(n_threads),
        gauss_flat, status_flat, initial_guess_flat,
        d_gain, d_N0, d_b4x, d_b4y,
        d_cpp, d_nvalid, d_iter, d_conv,
    )
    if ret != 0:
        raise RuntimeError(
            f"libkspacefit kspace_lm_fit_batch failed (ret={ret}): "
            f"{_KSPACE_ERRORS.get(ret, 'unknown error code')} "
            f"[corr_size={corr_size}, shape={shape}, floor={floor}]"
        )

    diag = {
        "gain": d_gain,
        "N0": d_N0,
        "cost_per_pt": d_cpp,
        "n_valid": d_nvalid,
        "iter": d_iter,
        "conv": d_conv.astype(bool),
    }
    if use_kx4:
        diag["b4x"] = d_b4x
    if use_ky4:
        diag["b4y"] = d_b4y

    n_proc = n_windows - int(mask.sum())
    if n_proc == 0:
        logger.info(f"Pass {pass_idx + 1}: k-space-LM, all {n_windows} windows masked")
    else:
        n_ok = int(np.sum(status_flat == STATUS_SUCCESS))
        logger.info(
            f"Pass {pass_idx + 1}: k-space-LM joint fit ({shape}, {floor} floor) "
            f"{n_ok}/{n_proc} ok [C, {n_threads} threads]"
        )
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


def fit_windows_kspace_lm_numpy(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    corr_size: tuple,
    config,
    pass_idx: int,
    debug: bool = False,
    return_diagnostics: bool = False,
    *,
    P_win: np.ndarray | None = None,
):
    """Reference (oracle) implementation of :func:`fit_windows_kspace_lm`.

    NOT on any production path — ``fit_windows_kspace_lm`` dispatches to C. This
    is retained because it is the oracle every C parity gate scores against, and
    because it is the readable statement of the algorithm. Same precedent as
    ``kspace_linear_fitting`` (retired from production selection 2026-07-21,
    module and tests kept).

    Identical signature and return contract. Differences from the C are limited
    to floating-point last-ulp effects: libm exp/cos/sin differ from NumPy's in
    the last ulp, so per-window LM paths can diverge on marginal accept/reject
    steps, and the flat floor's attenuation is a true division here against a
    reciprocal-multiply in C (<=1 ulp).
    """
    (
        shape, use_kx4, use_ky4, floor, corr_h, corr_w, n_windows,
        R_AA, R_BB, R_AB, mask, P_win,
    ) = _validate_inputs(
        R_AA, R_BB, R_AB, mask_flat, corr_size, config, P_win
    )

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
    # quartic exponent coefficients, present only when the shape enables them
    if use_kx4:
        diag["b4x"] = np.full(n_windows, np.nan)
    if use_ky4:
        diag["b4y"] = np.full(n_windows, np.nan)

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

            prep = _prepare_chunk(
                R_AA[idx],
                R_BB[idx],
                R_AB[idx],
                KX,
                KY,
                cy,
                cx,
                P=P_win[idx] if P_win is not None else None,
            )
            mu, Sigma, gain, N0, b4x, b4y, status, conv, niter, cpp = _run_fit(
                prep, use_kx4, use_ky4
            )

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
            if use_kx4:
                diag["b4x"][idx] = b4x
            if use_ky4:
                diag["b4y"][idx] = b4y
            diag["cost_per_pt"][idx] = cpp
            diag["n_valid"][idx] = prep["n_valid"]
            diag["conv"][idx] = conv
            diag["iter"][idx] = niter

    n_ok = int(np.sum(status_flat == STATUS_SUCCESS))
    logger.info(
        f"Pass {pass_idx + 1}: k-space-LM joint fit ({shape}, {floor} floor) "
        f"{n_ok}/{proc.size} ok [numpy reference]"
    )
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
