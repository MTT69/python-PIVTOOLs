"""
Closed-form (GSL-free) k-space transfer-function fitting for ensemble PIV.

DORMANT since 2026-07-08 — NOT imported by production. Replaced as the ensemble fitter
by ``kspace_lm_fitting.fit_windows_kspace_lm`` (batched-LM GSL replica minus beta),
which beat this fitter on the noisy planar validation set (med|relerr| vs DNS
uu 3.3/vv 2.2/uv 5.4% vs 8.1/9.8/8.4%; outer band 2.2% vs -11%) — see
wiki/sessions/2026-07-07-lm-gsl-replica-fitter.md. Kept intact for reference and
possible revert: restoring it = swapping the import and submit kwargs back in
``single_pass_accumulator`` (the exact pre-2026-07-08 call is in git history).
manual_tools/kspace research scripts still import this module.

This is the *de-janked* fitter: single stage, no beta, no nonlinear solver. It is a
drop-in replacement for ``fit_windows_kspace`` (same positional signature and the same
16-element ``gauss_flat`` output contract), but every step is closed-form linear algebra:

    1. Reference / transfer function (particle shape cancels algebraically):
           ref(k) = sqrt(|F_AA| * |F_BB|)            (auto-spectrum geometric mean)
           T(k)   = F_AB / ref                       (transfer function; Gaussian PDF model)

    2. Noise floor (scalar level x coloured shape) -- NO shape fit, NO beta:
           floor(k) = N0 * P_noise(k)
           N0       = median( ref/P_noise  over the high-k ring kr in [0.40, 0.50] )
       P_noise is the analytic interpolation-kernel MTF (warp colouring). At f=0 (pass 0 /
       integer shift) P_noise == 1 everywhere, so the coloured floor degenerates to a flat
       scalar floor with no special-casing. A pure flat floor is available as a cross-check
       (``use_pnoise=False``) but is NOT the default -- P_noise is validated physics.

    3. Covariance Sigma from a single weighted linear least-squares (4 unknowns):
           ln|T| = lnA - 2*pi^2 ( Sigma_xx kx^2 + Sigma_yy ky^2 + 2 Sigma_xy kx ky )
       linear in (lnA, Sigma_xx, Sigma_yy, Sigma_xy); weight = refc = ref - floor.

    4. Displacement mu from the linear phase slope of T (2 unknowns):
           phase(T) = -2*pi ( kx mu_x + ky mu_y )
       de-ramped by the integer AB-peak displacement first, so the residual phase never
       wraps; mu = d_int + d_sub.

The covariance maps directly to in-plane Reynolds stress (UU=Sigma_xx, VV=Sigma_yy,
UV=Sigma_xy) downstream. See ``single_pass_accumulator.finalize_pass`` for consumption and
``kspace_fitting.fit_windows_kspace`` for the GSL/FFTW two-stage predecessor this replaces.

Vectorised over all windows in NumPy, chunked to bound memory. No C extension, no GSL, no
FFTW (uses ``numpy.fft``, BSD).
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

# --- fixed fit hyper-parameters (the de-jank recipe) -------------------------------------
# High-k ring for the noise floor (cycles/px). _RING_LO is NOT arbitrary: it is the radius
# where the particle signal S(k)=exp(-2*pi^2*c*k^2) has fallen to ~5-8% of DC for the nominal
# particle-image size (autocorr c~0.8-0.9 px^2 here), i.e. "sample k-space where only noise
# survives". Deriving it per-run from the measured c was investigated (manual_tools/kspace/
# derive_ring.py) and REJECTED: it just moves the magic to the threshold X, which is less
# robust across passes than this fixed k. _RING_HI is the Nyquist edge (not a tuned value).
_RING_LO = 0.40          # ~6% particle-signal threshold (see note above)
_RING_HI = 0.50          # high-k ring outer radius (= Nyquist)
_TRUST_KMAX = 0.35       # magnitude-fit trust radius (model valid near the peak)
_TRUST_FLOOR_MARGIN = 1.5  # require refc > margin * floor (signal clear of the floor)
_TRUST_TMIN = 1e-3       # |T| sanity bounds
_TRUST_TMAX = 1.3
_PHASE_KMAX = 0.25       # phase-slope fit trust radius (avoid high-k wrap)
_PNOISE_GUARD = 1e-3     # clip P_noise away from zero (half-integer-shift dead-end guard)
_MIN_TRUST_PTS = 6       # minimum trusted points for a determined 4-param fit
_AUTOCROSS_KMAX = 0.15   # autocross floor: low-k band for the k->0 intercept fit
# autofit floor: marginalised particle-envelope width grid (px^2) for the auto-only S+N fit
_AUTOFIT_CGRID = np.linspace(0.15, 4.0, 28)
_AUTOFIT_KMAX = 0.49     # fit refn = A*exp(-2pi^2 c k^2) + N over k in (0, this)
# joint floor: particle-autocorr sigma grid (px, per axis) for the beta-free joint fit
_JOINT_SIG_GRID = np.linspace(0.45, 1.7, 12)
_KURT_KMAX = 0.0         # kurtosis-mode fit band cap (cycles/px); 0 = OFF (the validated default).
#                          TESTED + REJECTED: the trust band is already floor-limited to ~0.2, so a
#                          cap >=0.22 is non-binding and a cap <=0.15 lowers Sigma GLOBALLY (hurts the
#                          log region too) -- it cannot selectively fix the centerline over-fit, which
#                          is a curvature-IDENTIFIABILITY problem (low Sigma -> straight ln|T|), not a
#                          high-k-noise problem. Kept as an off-by-default knob to document the test.
_KURT_RIDGE = 0.10       # shape_mode='kurtosis_reg': Gaussian-prior ridge on the k^4 coeffs.
#                          Dimensionless (scaled to the k^4 columns' reference-band norm). The
#                          k^4 term is free where the trusted band supports it (log region) and
#                          shrinks to 0 where it does not (centerline) -> SNR-gated, y+-free.


def _batched_wls(feat, W, y, n_unknowns, ridge_diag=None):
    """Weighted linear least squares for many windows sharing the same design matrix.

    feat : (P, K) design matrix (SAME for every window -- the k-grid is fixed)
    W    : (N, P) per-window, per-point weights (0 outside the trust region)
    y    : (N, P) per-window observations
    ridge_diag : (K,) optional Tikhonov diagonal added to the normal matrix (a Gaussian prior
                 toward coeff=0). Used by shape_mode='kurtosis_reg' to SNR-gate the k^4 terms.
    Returns coeffs (N, K) and an OK mask (N,) (False where the system was singular).

    Builds the K x K normal equations via the identity
        M[n,i,j] = sum_p W[n,p] feat[p,i] feat[p,j] = W @ (feat[:,i]*feat[:,j])
        b[n,i]   = sum_p (W[n,p] y[n,p]) feat[p,i]  = (W*y) @ feat[:,i]
    so no (N, P, K) tensor is ever materialised.
    """
    N, P = W.shape
    K = n_unknowns
    M = np.empty((N, K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            g_ij = feat[:, i] * feat[:, j]          # (P,)
            M[:, i, j] = W @ g_ij                   # (N,)
            if j != i:
                M[:, j, i] = M[:, i, j]
    Z = W * y                                       # (N, P)
    b = Z @ feat                                    # (N, K)

    if ridge_diag is not None:                      # Gaussian prior toward coeff=0 (SNR-gating)
        idx = np.arange(K)
        M[:, idx, idx] += ridge_diag[None, :]

    coeffs = np.full((N, K), np.nan, dtype=np.float64)
    # Regularise + solve only where the system is well-posed (enough trusted points).
    npts = np.count_nonzero(W, axis=1)
    ok = npts >= max(K + 2, _MIN_TRUST_PTS)
    if np.any(ok):
        Mok = M[ok]
        # tiny ridge for conditioning (negligible vs. data scale, avoids LinAlgError)
        diag = np.einsum("nii->ni", Mok)
        ridge = 1e-12 * np.maximum(diag.max(axis=1, keepdims=True), 1e-30)
        Mok = Mok + ridge[:, :, None] * np.eye(K)[None]
        bok = b[ok][..., None]                  # (n_ok, K, 1) stacked RHS vectors (numpy>=2)
        try:
            sol = np.linalg.solve(Mok, bok)[..., 0]
        except np.linalg.LinAlgError:
            sol = np.full((Mok.shape[0], K), np.nan)
            for idx in range(Mok.shape[0]):
                try:
                    sol[idx] = np.linalg.solve(Mok[idx], bok[idx, :, 0])
                except np.linalg.LinAlgError:
                    pass
        coeffs[ok] = sol
    return coeffs, ok


def _autocross_intercept(diff, KR):
    """White noise floor as the k->0 intercept of (ref - |F_AB|)/ref_dc, per window.

    ``diff`` is (n, h, w), DC-normalised ((ref - |F_AB|)/ref_dc). Camera/warp noise is
    independent between frames A and B, so it survives in the auto-spectra (ref) but cancels
    in the cross (|F_AB|) -- their difference is the noise pedestal. The difference also
    carries the signal's own turbulence-decay curvature S(k)(1-|CF|) ~ b k^2 + c k^4, which
    vanishes only at DC; we fit ``a + b k^2 + c k^4`` over a low-k band and keep the
    intercept ``a``. Equal-per-radius weighting (w = 1/|k|) cancels the 2D mode-density bias
    that would otherwise pull the intercept toward the (contaminated) high-k end. The design
    and weights are shared across windows, so the 3x3 normal matrix is inverted once.
    """
    n = diff.shape[0]
    band = (KR > 0) & (KR <= _AUTOCROSS_KMAX)
    kr_b = KR[band]                                   # (P,)
    if kr_b.size < 6:
        return np.full(n, np.nan)
    k2 = kr_b * kr_b
    X = np.stack([np.ones_like(k2), k2, k2 * k2], axis=1)   # (P, 3)
    w = 1.0 / kr_b                                          # equal-per-radius weighting
    M = X.T @ (X * w[:, None])                              # (3, 3) shared across windows
    M = M + 1e-12 * np.trace(M) * np.eye(3)
    try:
        Minv0 = np.linalg.inv(M)[0]                         # first row -> intercept solver
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)
    b = (diff[:, band] * w[None]) @ X                       # (n, 3) per-window RHS
    return b @ Minv0                                        # (n,) intercept a


def _autofit_floor(refn, KR):
    """Ring-free white floor from an auto-only S(k)+N model fit, per window.

    The high-k "ring" is NOT signal-free on real PIV: the auto-spectrum envelope S(k) still
    has a shoulder there (worse at the centerline, where S is broadest), so a ring median
    over-reads the floor. Instead model the whole auto-spectrum as
        refn(k) = A * exp(-2 pi^2 c k^2) + N
    and return N -- the additive camera/warp pedestal -- with the particle envelope S
    explicitly fitted out rather than assumed absent. c is marginalised over a grid (the model
    is linear in A, N for each c, with the design shared across windows), so this is one 2x2
    solve per grid point, no per-window loop.

    NOTE this replaces the auto-MINUS-cross 'autocross' estimator, which is invalid on real
    data: the cross-spectrum carries a multiplicative decorrelation D (out-of-plane loss) that
    is absent from the additive auto floor, so ref - |F_AB| ~ S(1 - D|T|) + N is dominated by
    the decorrelation term, not the floor. The floor lives in the auto-spectra alone.
    """
    two_pi2 = 2.0 * np.pi ** 2
    n = refn.shape[0]
    krf = KR.ravel()
    m = (krf > 0) & (krf < _AUTOFIT_KMAX)
    k2 = krf[m] ** 2
    Y = refn.reshape(n, -1)[:, m]                       # (n, P)
    best_N = np.full(n, np.nan)
    best_ss = np.full(n, np.inf)
    for c in _AUTOFIT_CGRID:
        g = np.exp(-two_pi2 * c * k2)                   # (P,)
        X = np.stack([g, np.ones_like(g)], axis=1)      # (P, 2) shared across windows
        M = X.T @ X
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        coef = (Y @ X) @ Minv                           # (n, 2): [A, N]
        resid = Y - coef @ X.T                          # (n, P)
        ss = np.einsum("np,np->n", resid, resid)
        better = ss < best_ss
        best_ss = np.where(better, ss, best_ss)
        best_N = np.where(better, coef[:, 1], best_N)
    return best_N


def _joint_floor(refn, P, KX, KY):
    """beta-free joint floor: fit refn = (A*exp(-2pi^2(sx^2 kx^2 + sy^2 ky^2)) + N0)*P per window.

    This is the old GSL stage-1 joint fit (kspace_fitting.c) with the kurtosis term beta
    DROPPED -- the one piece that was actually janky (beta<->N0<->sigma degeneracy). The floor
    N0 is estimated JOINTLY with the particle envelope from the whole auto-spectrum, not read
    off a high-k ring. GSL-free: for a fixed particle width (sx,sy) the model is LINEAR in
    (A, N0) with known basis columns [G*P, P] (G = the Gaussian envelope), so we grid over
    (sx,sy) and do a shared-design 2x2 solve -- no nonlinear solver, no division by small P.
    Returns N0 (n,); the caller forms the coloured floor N0*P.
    """
    n = refn.shape[0]
    two_pi2 = 2.0 * np.pi ** 2
    KX2 = (KX * KX).ravel()
    KY2 = (KY * KY).ravel()
    Pf = (P[0] if P.ndim == 3 else P).ravel()
    band = (KX2 + KY2) > 0                              # exclude DC (normalisation pins it)
    Pb = Pf[band]
    Y = refn.reshape(n, -1)[:, band]                   # (n, Pb)
    best_N = np.full(n, np.nan)
    best_ss = np.full(n, np.inf)
    for sx in _JOINT_SIG_GRID:
        for sy in _JOINT_SIG_GRID:
            G = np.exp(-two_pi2 * (sx * sx * KX2[band] + sy * sy * KY2[band]))
            c1 = G * Pb                                # A column
            c2 = Pb                                    # N0 column
            S11 = c1 @ c1; S12 = c1 @ c2; S22 = c2 @ c2
            det = S11 * S22 - S12 * S12
            if abs(det) < 1e-30:
                continue
            b1 = Y @ c1; b2 = Y @ c2                    # (n,)
            A = (S22 * b1 - S12 * b2) / det
            N0 = (S11 * b2 - S12 * b1) / det
            pred = A[:, None] * c1[None] + N0[:, None] * c2[None]
            resid = Y - pred
            ss = np.einsum("np,np->n", resid, resid)
            better = ss < best_ss
            best_ss = np.where(better, ss, best_ss)
            best_N = np.where(better, N0, best_N)
    return best_N


def _fit_chunk(R_AA, R_BB, R_AB, KX, KY, KR, cy, cx, f_xy, kernel, floor_mode,
               weight_mode="refc", shape_mode="gauss"):
    """Fit one chunk of windows. Returns Sigma (n,3), mu (n,2), amps (n,3), status (n)."""
    n = R_AA.shape[0]
    corr_h, corr_w = KX.shape

    F_AA = _fft_planes(R_AA, corr_h, corr_w)
    F_BB = _fft_planes(R_BB, corr_h, corr_w)
    F_AB = _fft_planes(R_AB, corr_h, corr_w)

    amp_A = R_AA.reshape(n, corr_h, corr_w)[:, cy, cx]
    amp_B = R_BB.reshape(n, corr_h, corr_w)[:, cy, cx]
    amp_AB = R_AB.reshape(n, corr_h, corr_w)[:, cy, cx]

    ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))             # (n, h, w)
    ref_dc = ref[:, cy, cx][:, None, None]
    refn = ref / np.maximum(ref_dc, 1e-30)                 # DC-normalised reference

    fab_mag = np.abs(F_AB)
    fab_dc = fab_mag[:, cy, cx][:, None, None]
    fabn = fab_mag / np.maximum(fab_dc, 1e-30)             # DC-normalised |T| numerator

    # --- noise floor --------------------------------------------------------------------
    # The autocorrelation floor has two physical parts: a WHITE +2*sigma^2 pedestal (camera
    # noise, flat in k) and a COLOURED warp-interpolation term (proportional to P_noise, the
    # kernel MTF). floor_mode selects how much structure to model:
    #   'flat'      floor = median(refn) over a high-k ring           (captures the white
    #               pedestal; the recommended default -- matches DNS best at 4000 images).
    #   'autocross' floor = k->0 intercept of (ref - |F_AB|)/ref_dc   (ring-free; reads the
    #               white pedestal at low k via A-B noise independence -- see PRD H1).
    #   'coloured'  floor = level * P_noise, level = median(refn/P).  KNOWN DEAD END: forces
    #               the white pedestal to be coloured -> over-subtracts near half-integer
    #               shift. Kept only as a cross-check.
    #   'coloured2' floor = a + b*P_noise, (a,b) from a 2-param linear LS over the high-k
    #               band -- separates the white pedestal (a) from the coloured warp (b).
    #               Reduces to flat when f=0 (P==1). The principled analytic attempt.
    band = (KR >= _RING_LO - 0.10) & (KR <= _RING_HI)      # wider band so P varies for a/b
    ring = (KR >= _RING_LO) & (KR <= _RING_HI)
    if floor_mode == "none":
        floor = np.zeros((refn.shape[0], 1, 1))            # no floor (e.g. pre-removed at source)
    elif floor_mode == "flat":
        floor = np.median(refn[:, ring], axis=1)[:, None, None]
    elif floor_mode == "autofit":
        # ring-free auto-only S+N model fit; N (the white pedestal) with the envelope fitted
        # out. The corrected H1 estimator (see _autofit_floor) for real PIV.
        a = _autofit_floor(refn, KR)
        floor = np.maximum(a, 0.0)[:, None, None]
    elif floor_mode == "autocross":
        # k->0 intercept of (ref - |F_AB|)/ref_dc. VALID ONLY at low decorrelation (synthetic);
        # on real PIV the cross decorrelation D<1 swamps the floor -- use 'autofit' instead.
        diff = (ref - fab_mag) / np.maximum(ref_dc, 1e-30)
        a = _autocross_intercept(diff, KR)
        floor = np.maximum(a, 0.0)[:, None, None]          # guard against a slightly-neg fit
    else:
        P = compute_noise_psd_2d(KX, KY, f_xy[0], f_xy[1], kernel=kernel)
        P = np.maximum(P, _PNOISE_GUARD)[None]
        if floor_mode == "joint":
            # beta-free joint floor: (A*Gauss + N0)*P fit over the whole auto-spectrum
            # (old GSL stage-1 minus kurtosis). Coloured floor = N0 * P.
            N0 = _joint_floor(refn, P, KX, KY)
            floor = np.maximum(N0, 0.0)[:, None, None] * P
        elif floor_mode == "coloured2":
            # per-window 2x2 normal equations for refn = a + b*P over the band
            Pb = np.broadcast_to(P, refn.shape)[:, band]   # (n, nb)
            rb = refn[:, band]                             # (n, nb)
            one = np.ones_like(Pb)
            S11 = one.sum(1); S1P = Pb.sum(1); SPP = (Pb * Pb).sum(1)
            T1 = rb.sum(1); TP = (rb * Pb).sum(1)
            det = S11 * SPP - S1P * S1P
            det = np.where(np.abs(det) < 1e-30, np.nan, det)
            a = (T1 * SPP - TP * S1P) / det
            b = (S11 * TP - S1P * T1) / det
            floor = a[:, None, None] + b[:, None, None] * P
        else:  # 'coloured'
            level = np.median((refn / P)[:, ring], axis=1)[:, None, None]
            floor = level * P
    refc = refn - floor

    # --- magnitude fit: Sigma (4 unknowns) -------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        T_mag = fabn / refc
        lnT = np.log(T_mag)

    two_pi2 = 2.0 * np.pi ** 2
    # common sanity mask (no hard k cap); the weighting decides what high-k modes contribute
    sane = (refc > 0) & (T_mag > _TRUST_TMIN) & (T_mag < _TRUST_TMAX) & np.isfinite(lnT)
    lnT_flat = np.where(sane, lnT, 0.0).reshape(n, -1)
    sane_f = sane.reshape(n, -1)
    # design matrix (shared). Gaussian PDF -> ln|T| linear in [1, kx^2, ky^2, 2 kx ky].
    # shape_mode='kurtosis' adds the 4th-order even terms [kx^4, ky^4, 2 kx^2 ky^2] so a
    # non-Gaussian (kurtotic) PDF's k^4 curvature is absorbed by nuisance coeffs instead of
    # biasing the k^2 (Sigma) slope. Still one linear LS; Sigma is always coeffs[1..3].
    KX2f = (KX * KX).ravel(); KY2f = (KY * KY).ravel()
    cols = [np.ones(corr_h * corr_w), KX2f, KY2f, (2.0 * KX * KY).ravel()]
    if shape_mode in ("kurtosis", "kurtosis_reg"):
        cols += [KX2f * KX2f, KY2f * KY2f, 2.0 * KX2f * KY2f]
    elif shape_mode == "kurtosis_decoupled":
        # separable-kurtosis model: per-axis k^4 ONLY (kx^4 -> Sigma_xx, ky^4 -> Sigma_yy), and
        # DROP the cross term 2 kx^2 ky^2. For a separable PDF p(dx)p(dy) the cross-kurtosis is
        # identically zero, so that column is a spurious DOF that couples the two directions and
        # lets the streamwise (kx^4) curvature drag ky^4 into fitting vv's near-Gaussian noise
        # (the 7-col 'kurtosis' degraded vv 6.9->13.4%). Removing it decouples the axes.
        cols += [KX2f * KX2f, KY2f * KY2f]
    feat_mag = np.stack(cols, axis=1)  # (P, 4), (P, 6) or (P, 7)
    K = feat_mag.shape[1]

    # 'kurtosis_reg': a Gaussian-prior ridge on the k^4 coeffs, scaled to those columns' norm
    # over a fixed reference band so a single global lambda self-adapts (free where the trusted
    # band supports curvature, shrunk to the Gaussian fit where it does not). No y+/Sigma gate.
    ridge = None
    if shape_mode == "kurtosis_reg" and K >= 7:
        refband = KR.ravel() < _TRUST_KMAX
        S0 = (feat_mag[refband] ** 2).sum(axis=0)   # (K,) unit-weight column norms
        ridge = np.zeros(K)
        ridge[4:7] = _KURT_RIDGE * S0[4:7]

    if weight_mode == "soft":
        # GSL stage-2 weighting (GSL-free): w = w_snr * w_soft, with the soft decay scale
        # DERIVED from Sigma itself -- w_soft = exp(-2 pi^2 (Sigma_xx kx^2 + Sigma_yy ky^2)),
        # anisotropic, no hard k cap; w_snr = refc/floor (SNR). Sigma is needed to build the
        # weight, so seed with a refc fit then refine twice. This is the self-adapting,
        # cap-free weighting the old GSL fitter used (and what H2's invvar fumbled).
        refc_f = refc.reshape(n, -1)
        floor_b = np.broadcast_to(floor, refc.shape).reshape(n, -1)
        w_snr = np.where(sane_f, refc_f / np.maximum(floor_b, 1e-6), 0.0)
        W = np.where(sane_f, np.maximum(refc_f, 0.0), 0.0)        # seed = refc weighting
        coeffs, ok_mag = _batched_wls(feat_mag, W, lnT_flat, K, ridge_diag=ridge)
        for _ in range(2):
            Sxx = np.maximum(-coeffs[:, 1] / two_pi2, 0.0)
            Syy = np.maximum(-coeffs[:, 2] / two_pi2, 0.0)
            w_soft = np.exp(-two_pi2 * (Sxx[:, None] * KX2f[None] + Syy[:, None] * KY2f[None]))
            W = w_snr * w_soft
            coeffs, ok_mag = _batched_wls(feat_mag, W, lnT_flat, K, ridge_diag=ridge)
    elif weight_mode == "invvar":
        # H2: continuous cross-power inverse-variance weighting (see PRD H2; biased low on real
        # data -- the cap-free soft weighting above is preferred).
        W = np.where(sane_f, fabn.reshape(n, -1) ** 2, 0.0)
        coeffs, ok_mag = _batched_wls(feat_mag, W, lnT_flat, K, ridge_diag=ridge)
    else:  # 'refc' -- original hard-cap trust region + refc weighting
        # kurtosis modes use a tighter band cap: the k^4 term is only identifiable below ~0.25,
        # so fitting it (and Sigma) past that just feeds high-k noise into the shape coeffs.
        kmax_trust = (_KURT_KMAX if (shape_mode in ("kurtosis", "kurtosis_reg",
                                                    "kurtosis_decoupled") and _KURT_KMAX > 0)
                      else _TRUST_KMAX)
        trust = sane & (KR[None] < kmax_trust) & (refc > _TRUST_FLOOR_MARGIN * floor)
        W = np.maximum(np.where(trust, refc, 0.0).reshape(n, -1), 0.0)
        coeffs, ok_mag = _batched_wls(feat_mag, W, lnT_flat, K, ridge_diag=ridge)

    Sigma_xx = -coeffs[:, 1] / two_pi2
    Sigma_yy = -coeffs[:, 2] / two_pi2
    Sigma_xy = -coeffs[:, 3] / two_pi2

    # --- phase fit: mu (2 unknowns), de-ramped by the integer AB peak ----------------------
    mu = _fit_phase(F_AB, refc, KX, KY, KR, cy, cx)

    # --- status ---------------------------------------------------------------------------
    status = np.zeros(n, dtype=np.int32)
    neg_var = (Sigma_xx < 0) | (Sigma_yy < 0)
    status[~ok_mag] = 1                         # underdetermined / singular magnitude fit
    status[neg_var & ok_mag] = 5                # negative variance
    # clamp negative variances to zero (matches downstream max(.,0) for UU/VV)
    Sigma_xx = np.maximum(Sigma_xx, 0.0)
    Sigma_yy = np.maximum(Sigma_yy, 0.0)
    big = (np.abs(mu[:, 0]) > 0.75 * corr_w) | (np.abs(mu[:, 1]) > 0.75 * corr_h)
    status[big] = 3

    Sigma = np.stack([Sigma_xx, Sigma_yy, Sigma_xy], axis=1)
    amps = np.stack([amp_A, amp_B, amp_AB], axis=1)
    return Sigma, mu, amps, status


def _fit_phase(F_AB, refc, KX, KY, KR, cy, cx):
    """Displacement from the phase slope of T, de-ramped by the integer AB-peak shift."""
    n, corr_h, corr_w = F_AB.shape
    # integer displacement from the cross-correlation peak (per window)
    R_AB = np.abs(np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(F_AB, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)))
    flat_idx = R_AB.reshape(n, -1).argmax(axis=1)
    pj, pi = np.unravel_index(flat_idx, (corr_h, corr_w))
    d_int_x = (pi - cx).astype(np.float64)     # integer dx (px)
    d_int_y = (pj - cy).astype(np.float64)     # integer dy (px)

    # de-ramp: remove the linear phase of the integer shift so the residual never wraps
    ramp = np.exp(2j * np.pi * (KX[None] * d_int_x[:, None, None]
                                + KY[None] * d_int_y[:, None, None]))
    F_deramped = F_AB * ramp
    phase = np.angle(F_deramped)               # small residual phase (n, h, w)

    trust = (KR[None] < _PHASE_KMAX) & (KR[None] > 0) & (refc > 0)
    W = np.where(trust, np.maximum(refc, 0.0), 0.0).reshape(n, -1)
    y = np.where(trust, phase, 0.0).reshape(n, -1)

    # model: phase = -2*pi (kx mu_sx + ky mu_sy)  ->  design [-2pi kx, -2pi ky]
    feat = np.stack([(-2.0 * np.pi * KX).ravel(),
                     (-2.0 * np.pi * KY).ravel()], axis=1)  # (P, 2)
    coeffs, ok = _batched_wls(feat, W, y, 2)
    d_sub = np.where(np.isfinite(coeffs), coeffs, 0.0)
    mu_x = d_int_x + d_sub[:, 0]
    mu_y = d_int_y + d_sub[:, 1]
    return np.stack([mu_x, mu_y], axis=1)


def fit_windows_kspace_linear(
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
    use_pnoise: bool = False,
    floor_mode: Optional[str] = None,
    weight_mode: str = "refc",
    shape_mode: str = "gauss",
):
    """Closed-form k-space fit. Drop-in for ``fit_windows_kspace`` (same arg order/contract).

    ``use_soft_weighting`` and ``k_max_cap`` are accepted for signature compatibility but
    not used (the trust region + refc weighting replace them). ``use_pnoise`` selects the
    coloured noise floor (default True); set False for the flat-floor cross-check.

    ``weight_mode`` selects the magnitude-fit weighting: ``'refc'`` (default, the original
    hard-cap trust region weighted by refc) or ``'invvar'`` (continuous cross-power
    inverse-variance weighting ``w=|F_AB|^2`` with no hard k cap -- PRD H2).

    Returns ``(gauss_flat[n,16] float64, status_flat[n] int32, initial_guess_flat[n,16])``.
    """
    # floor_mode resolves the floor model; use_pnoise kept for back-compat (True->'coloured')
    if floor_mode is None:
        floor_mode = "coloured" if use_pnoise else "flat"
    if floor_mode not in ("flat", "autofit", "autocross", "joint",
                          "coloured", "coloured2", "none"):
        raise ValueError(
            f"floor_mode must be flat|autofit|autocross|joint|coloured|coloured2|none, "
            f"got {floor_mode!r}")
    if weight_mode not in ("refc", "invvar", "soft"):
        raise ValueError(f"weight_mode must be refc|invvar|soft, got {weight_mode!r}")
    if shape_mode not in ("gauss", "kurtosis", "kurtosis_reg", "kurtosis_decoupled"):
        raise ValueError(
            f"shape_mode must be gauss|kurtosis|kurtosis_reg|kurtosis_decoupled, got {shape_mode!r}")

    corr_h, corr_w = corr_size
    n_windows = len(mask_flat)
    n_per = corr_h * corr_w
    expected = n_windows * n_per
    if R_AA.size != expected:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {expected} "
            f"(n_windows={n_windows}, corr_size={corr_size})"
        )

    R_AA = np.ascontiguousarray(R_AA, dtype=np.float64).reshape(n_windows, corr_h, corr_w)
    R_BB = np.ascontiguousarray(R_BB, dtype=np.float64).reshape(n_windows, corr_h, corr_w)
    R_AB = np.ascontiguousarray(R_AB, dtype=np.float64).reshape(n_windows, corr_h, corr_w)
    mask = np.asarray(mask_flat, dtype=bool)

    KX, KY, KR = _kgrids(corr_h, corr_w)
    cy, cx = corr_h // 2, corr_w // 2          # DC / peak index after fftshift

    # per-window fractional displacement for P_noise (warp splits the shift A/B -> pred/2)
    if predictor_displacements is not None and floor_mode not in ("flat", "autofit", "autocross"):
        pred = np.asarray(predictor_displacements, dtype=np.float64).reshape(-1, 2)
        # A real multipass predictor carries NaN at invalid/edge windows. A NaN
        # fractional shift is meaningless for the noise PSD AND would break the
        # constant-f chunk grouping below (NaN != NaN -> empty chunk -> crash),
        # so default NaN windows to zero shift (the un-warped, flat-P_noise case).
        f_y = np.nan_to_num(frac_distance(pred[:, 0] / 2.0), nan=0.0)
        f_x = np.nan_to_num(frac_distance(pred[:, 1] / 2.0), nan=0.0)
    else:
        f_y = np.zeros(n_windows)
        f_x = np.zeros(n_windows)

    gauss_flat = np.zeros((n_windows, 16), dtype=np.float64)
    status_flat = np.full(n_windows, -1, dtype=np.int32)
    initial_guess_flat = np.zeros((n_windows, 16), dtype=np.float64)

    center_x = corr_w / 2.0 + 1.0              # 1-based centre (matches C + velocity extract)
    center_y = corr_h / 2.0 + 1.0

    proc = np.where(~mask)[0]
    if proc.size == 0:
        logger.info(f"Pass {pass_idx + 1}: k-space-linear, all {n_windows} windows masked")
        if return_diagnostics:
            return gauss_flat, status_flat, initial_guess_flat, None
        return gauss_flat, status_flat, initial_guess_flat

    # P_noise depends on (f_x, f_y); group windows by identical fractional shift so each
    # chunk shares one P_noise field. When P_noise is off (or f==0) this is a single group.
    CHUNK = 4096
    fkey = np.round(np.stack([f_x[proc], f_y[proc]], axis=1), 4)
    order = np.lexsort((fkey[:, 1], fkey[:, 0]))
    proc_sorted = proc[order]
    fkey_sorted = fkey[order]

    start = 0
    Ns = proc_sorted.size
    # Cap native threadpools to 1 for the batched FFT / linear solves. Under
    # Dask each worker process already runs one task at a time
    # (threads_per_worker=1), so N worker processes each opening their own
    # BLAS/FFT pool would oversubscribe the cores; pin to 1 so total threads
    # == workers. Numerics are thread-count-invariant, so this is free.
    with _get_threadpool_controller().limit(limits=1):
        while start < Ns:
            # extend the chunk while the fractional shift is constant (bounded by CHUNK)
            end = min(start + CHUNK, Ns)
            same = np.all(fkey_sorted[start:end] == fkey_sorted[start], axis=1)
            if not np.all(same):
                end = start + int(np.argmin(same))
            idx = proc_sorted[start:end]
            f_xy = (float(fkey_sorted[start, 0]), float(fkey_sorted[start, 1]))

            Sigma, mu, amps, status = _fit_chunk(
                R_AA[idx], R_BB[idx], R_AB[idx], KX, KY, KR, cy, cx,
                f_xy, interp_kernel, floor_mode, weight_mode, shape_mode,
            )

            gauss_flat[idx, 0:3] = amps
            gauss_flat[idx, 3:6] = 0.0
            gauss_flat[idx, 6:9] = np.nan      # particle-size slots -> NaN (UU=Sigma_xx)
            gauss_flat[idx, 9] = Sigma[:, 0]   # Sigma_xx
            gauss_flat[idx, 10] = Sigma[:, 1]  # Sigma_yy
            gauss_flat[idx, 11] = Sigma[:, 2]  # Sigma_xy
            gauss_flat[idx, 12] = center_x
            gauss_flat[idx, 13] = center_y
            gauss_flat[idx, 14] = center_x + mu[:, 0]
            gauss_flat[idx, 15] = center_y + mu[:, 1]
            status_flat[idx] = status
            start = end

    initial_guess_flat[:] = gauss_flat         # closed-form: no separate initial guess

    n_valid = int(proc.size)
    n_ok = int(np.sum(status_flat == 0))
    logger.info(
        f"Pass {pass_idx + 1}: k-space-linear fit {n_ok}/{n_valid} ok "
        f"(floor_mode={floor_mode}, kernel={interp_kernel})"
    )
    if debug and n_ok > 0:
        ok = status_flat == 0
        logger.info(
            f"  Sigma_xx median={np.nanmedian(gauss_flat[ok, 9]):.4f} "
            f"Sigma_yy median={np.nanmedian(gauss_flat[ok, 10]):.4f} "
            f"mu_x median={np.nanmedian(gauss_flat[ok, 14] - center_x):.4f} "
            f"mu_y median={np.nanmedian(gauss_flat[ok, 15] - center_y):.4f}"
        )

    if return_diagnostics:
        return gauss_flat, status_flat, initial_guess_flat, None
    return gauss_flat, status_flat, initial_guess_flat
