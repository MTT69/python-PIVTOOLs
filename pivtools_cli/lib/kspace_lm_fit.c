/******************************************************************************
 * kspace_lm_fit.c — the production k-space LM ensemble fitter.
 *
 * Built into libkspacefit by setup.py and driven through ctypes from
 * pivtools_cli/piv/piv_backend/kspace_lm_fitting.py, which owns the input
 * validation, logging and output assembly. This file does the maths only.
 *
 * The NumPy implementation in that module (fit_windows_kspace_lm_numpy) is
 * retained as the test-only oracle this file is gated against; it is not on
 * any production path. Constants below are MIRRORED from it — change one and
 * you must change the other. unit-tests/test_kspace_c_constants.py fails the
 * build if they drift.
 *
 * Scope: the WHOLE per-window pipeline of fit_windows_kspace_lm —
 *   viability gates -> centred FFTs -> transfer ratio T + weights + seeds ->
 *   per-window Levenberg-Marquardt (7..9 params: mu_x, mu_y, Sxx, Syy, Sxy,
 *   g, N0 [, b4x][, b4y]) -> status/output contract (16-col gauss_flat).
 *
 * Design choices vs the NumPy original (documented in the 2026-07-21 session
 * note; all validated by the gate tests):
 *   - Everything double precision (Sigma IS the science output).
 *   - Double-precision mixed-radix complex FFT from generated fixed-size
 *     codelets (codelets_double_gen.h, emitted by codelet_gen/gen_codelet.py
 *     with --isa scalar_d --transforms cfft). Covers every
 *     BUILT_FFT_SIZES axis length including the 2^k*3 sizes. NumPy's
 *     convention F = fftshift(fft2(ifftshift(R))) is reproduced WITHOUT any
 *     data shuffles: the input is loaded with ifftshift index mapping and the
 *     spectrum is consumed in natural order against natural-order k-grids
 *     (fftfreq without fftshift). All reductions over bins are order-
 *     independent, so this is exact, not approximate.
 *   - Full-plane residuals (both Hermitian halves), matching Python. The
 *     half-spectrum trick (2x) is a documented follow-up, not done here.
 *   - Per-window LM instead of chunk-batched LM: a straggler window costs
 *     only its own iterations (the NumPy chunk pays ~250 iterations of
 *     overhead when ONE window in 4096 straggles).
 *   - Flat and coloured noise floors share ONE code path. The production
 *     model is att = 1 - N0*D with D = P(k;fx,fy)/F_ref (kspace_floor=
 *     'coloured', default since 2026-07-24); the flat floor is the same
 *     expression with D = 1/F_ref. Both are materialised once per window
 *     into scratch.Dv, so the residual/Jacobian inner loops carry no branch
 *     and one model change cannot touch one floor and miss the other.
 *     KNOWN DEVIATION (reciprocal-multiply vs true division). This file forms
 *     x*(1/max(d,1e-30)) where the NumPy oracle writes x/max(d,1e-30) — a
 *     <=1-ULP difference on some bins. It applies in TWO places:
 *       (a) the flat att, N0*(1/F_ref) vs N0/F_ref;
 *       (b) the transfer ratio T = F_AB*(1/F_ref) vs F_AB/F_ref — the
 *           OBSERVATION vector, so this one is present under BOTH floors.
 *     Only the dModel/dN0 Jacobian columns are exact, because Python
 *     multiplies by the reciprocal there too. Neither branch is byte-identical
 *     to the oracle; the gates are tolerance-based by design and the measured
 *     |dSigma| median is 1.1e-16.
 *   - P_win arrives in the CENTRED (fftshift'd) layout that Python's Fr uses,
 *     because kspace_floor_psd builds it through the same _fft_planes helper.
 *     It is loaded into natural order with the identical ifftshift index
 *     mapping fft2_centered applies to the planes. A mis-ordered P yields a
 *     plausible-but-wrong fit rather than an obvious failure, so the harness
 *     gates it directly via kspace_debug_prep.
 *   - JtJ/grad are cached across REJECTED LM steps (parameters unchanged =>
 *     bit-identical recomputation in the original; we just skip it).
 *   - OpenMP dynamic schedule over windows.
 *
 * Numerical-parity note — three documented sources of divergence from the
 * NumPy oracle, none of which permits bit equality:
 *   1. libm exp/cos/sin/log differ from NumPy's float64 routines in the last
 *      ulp, so accept/reject can flip on a marginal step and per-window
 *      iteration paths may differ.
 *   2. The reciprocal-multiply above.
 *   3. The normal equations are solved by Cholesky here (chol_solve below) but
 *      by LU in the oracle (np.linalg.solve). For a rank-deficient J^T J — a
 *      parameter with no local sensitivity — the damped A + lam*max(diag,1e-30)
 *      can be numerically non-PD, and this returns -1 (rejecting a step NumPy
 *      would take). Self-correcting via the lam*10 backtrack, but it is a real
 *      difference in trajectory, not just in arithmetic.
 * The gate tests therefore check (a) status agreement, (b) converged-window
 * parameter agreement at tolerance, (c) grid-statistic agreement.
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "codelets_double_gen.h"
#include "kspace_lm_fit.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── constants: MUST track kspace_lm_fitting.py ─────────────────────────── */
#define MAIN_MAX_ITER 250
#define LM_XTOL 1e-8
#define LM_FTOL 1e-8
#define MIN_VALID_PTS 10
#define MAX_DISP_FRAC 0.75
#define COST_PER_PT_ACCEPT 1.0
#define EXP_ARG_MAX 300.0 /* clip guard for non-PSD trial steps */
#define LAM_INIT 1e-3
#define LAM_MIN 1e-12
#define LAM_ENOPROG 1e12
#define GAIN_LO 1e-3
#define GAIN_HI 1e3
#define SIGMA_SEED_XX 1.0
#define SIGMA_SEED_YY 0.2

/* Coloured floor (kspace_lm_fitting.py:137-139). N0 is bounded [0, 10] in
 * normalized F_ref units and seeded from the tail median of F_ref/P over
 * |k| >= 0.35, where the floor dominates. The flat branch keeps seed 0 and an
 * unbounded N0. */
#define COLOURED_N0_HI 10.0
#define COLOURED_SEED_KR_MIN 0.35
#define COLOURED_SEED_CLIP_LO 1e-3
#define COLOURED_SEED_CLIP_HI 10.0

#define STATUS_MASKED (-1)
#define STATUS_SUCCESS 0
#define STATUS_NO_CONVERGE 1
#define STATUS_LOW_SNR 2
#define STATUS_BIG_DISP 3

#define KMAX 9 /* 7 base + b4x + b4y */

static const double TWO_PI = 6.283185307179586476925286766559;
static const double TWO_PI2 = 2.0 * 9.869604401089358618834490999876;

/* =========================================================================
 * Complex FFT (double), natural order out, forward, numpy sign convention
 * e^{-2*pi*i*k*n/N}, unnormalised.
 *
 * Supplied by codelets_double.h: fully-unrolled fixed-size codelets with the
 * twiddles baked in as double constants, generated by gen_codelet_double.py
 * (the offline 'scalar_d' fork of the production codelet generator). This
 * replaces the hand-written radix-2 loop that used to live here, which could
 * only do power-of-two N and therefore could not transform over half of
 * production's BUILT_FFT_SIZES — 12, 24, 48 and 96 are 2^k*3. The generator's
 * mixed-radix decomposition already handled those and is machine-checked
 * against numpy for every emitted size before any C is rendered.
 * ========================================================================= */
typedef void (*cfft_fn)(const double *Xr, const double *Xi, double *Yr,
                        double *Yi);

/* NULL for any axis length without a built codelet. The set matches
 * pivtools_core/fft_sizes.py BUILT_FFT_SIZES, which is what
 * validate_window_sizes already restricts production configs to. */
static cfft_fn cfft_for(int n) {
    switch (n) {
    case 8:   return cfft8_scalar_d;
    case 12:  return cfft12_scalar_d;
    case 16:  return cfft16_scalar_d;
    case 24:  return cfft24_scalar_d;
    case 32:  return cfft32_scalar_d;
    case 48:  return cfft48_scalar_d;
    case 64:  return cfft64_scalar_d;
    case 96:  return cfft96_scalar_d;
    case 128: return cfft128_scalar_d;
    default:  return NULL;
    }
}

/* =========================================================================
 * Per-thread scratch: FFT buffers + spectra + fit workspace, sized H*W once.
 * ========================================================================= */
typedef struct {
    int H, W, P;
    cfft_fn row_fft, col_fft; /* codelets for the W- and H-length axes */
    double *fre[3], *fim[3]; /* spectra of R_AA, R_BB, R_AB, natural order */
    double *tmp_re, *tmp_im; /* column FFT gather buffers, length H */
    /* the codelets are out-of-place, so each axis needs a result buffer */
    double *row_or, *row_oi; /* length W */
    double *col_or, *col_oi; /* length H */
    double *Fr;              /* sqrt(|F_AA|*|F_BB|) */
    double *Tre, *Tim;       /* transfer ratio */
    double *Wgt;             /* fit weights */
    double *kx, *ky;         /* natural-order fftfreq grids, length W and H */
    double *Dv;              /* floor regressor: P/F_ref (coloured) or 1/F_ref */
    double *Pn;              /* P in natural order (coloured only; N0 seed) */
    double *tailbuf;         /* gather buffer for the N0 seed median */
    int *tail_idx;           /* natural-order bins with |k| >= COLOURED_SEED_KR_MIN */
    int n_tail;
} scratch;

static int scratch_init(scratch *s, int H, int W) {
    memset(s, 0, sizeof(*s));
    s->H = H;
    s->W = W;
    s->P = H * W;
    s->row_fft = cfft_for(W);
    s->col_fft = cfft_for(H);
    if (!s->row_fft || !s->col_fft) return -1;
    for (int i = 0; i < 3; ++i) {
        s->fre[i] = (double *)malloc(sizeof(double) * (size_t)s->P);
        s->fim[i] = (double *)malloc(sizeof(double) * (size_t)s->P);
        if (!s->fre[i] || !s->fim[i]) return -1;
    }
    s->tmp_re = (double *)malloc(sizeof(double) * (size_t)H);
    s->tmp_im = (double *)malloc(sizeof(double) * (size_t)H);
    s->row_or = (double *)malloc(sizeof(double) * (size_t)W);
    s->row_oi = (double *)malloc(sizeof(double) * (size_t)W);
    s->col_or = (double *)malloc(sizeof(double) * (size_t)H);
    s->col_oi = (double *)malloc(sizeof(double) * (size_t)H);
    s->Fr = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->Tre = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->Tim = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->Wgt = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->kx = (double *)malloc(sizeof(double) * (size_t)W);
    s->ky = (double *)malloc(sizeof(double) * (size_t)H);
    s->Dv = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->Pn = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->tailbuf = (double *)malloc(sizeof(double) * (size_t)s->P);
    s->tail_idx = (int *)malloc(sizeof(int) * (size_t)s->P);
    if (!s->tmp_re || !s->tmp_im || !s->Fr || !s->Tre || !s->Tim || !s->Wgt ||
        !s->kx || !s->ky || !s->Dv || !s->Pn || !s->tailbuf || !s->tail_idx ||
        !s->row_or || !s->row_oi || !s->col_or || !s->col_oi)
        return -1;
    /* np.fft.fftfreq, natural order, cycles/pixel */
    for (int k = 0; k < W; ++k)
        s->kx[k] = (k < (W + 1) / 2 ? (double)k : (double)(k - W)) / (double)W;
    for (int k = 0; k < H; ++k)
        s->ky[k] = (k < (H + 1) / 2 ? (double)k : (double)(k - H)) / (double)H;
    /* Tail bins for the coloured N0 seed. The mask depends only on the k-grid,
     * never on the window, so it is built once per thread. */
    const double kr2_min = COLOURED_SEED_KR_MIN * COLOURED_SEED_KR_MIN;
    s->n_tail = 0;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            if (s->ky[r] * s->ky[r] + s->kx[c] * s->kx[c] >= kr2_min)
                s->tail_idx[s->n_tail++] = r * W + c;
    return 0;
}

static void scratch_free(scratch *s) {
    for (int i = 0; i < 3; ++i) {
        free(s->fre[i]);
        free(s->fim[i]);
    }
    free(s->tmp_re);
    free(s->tmp_im);
    free(s->Fr);
    free(s->Tre);
    free(s->Tim);
    free(s->Wgt);
    free(s->kx);
    free(s->ky);
    free(s->Dv);
    free(s->Pn);
    free(s->tailbuf);
    free(s->tail_idx);
    free(s->row_or);
    free(s->row_oi);
    free(s->col_or);
    free(s->col_oi);
}

/* 2D FFT of one real plane with ifftshift load mapping:
 * buf[r][c] <- plane[(r+H/2)%H][(c+W/2)%W], then rows FFT then cols FFT.
 * Result in natural k order == fftshift-free equivalent of the Python
 * fftshift(fft2(ifftshift(R))) (the centred spectrum is a permutation). */
static void fft2_centered(scratch *s, const double *plane, int slot) {
    const int H = s->H, W = s->W;
    const int hh = H / 2, hw = W / 2;
    double *re = s->fre[slot], *im = s->fim[slot];
    for (int r = 0; r < H; ++r) {
        const double *src = plane + (size_t)((r + hh) % H) * W;
        double *rr = re + (size_t)r * W;
        double *ri = im + (size_t)r * W;
        for (int c = 0; c < W; ++c) {
            rr[c] = src[(c + hw) % W];
            ri[c] = 0.0;
        }
        /* out-of-place codelet, then copy back so the spectrum stays in the
         * row-major (H, W) buffers the column pass and Fr/T loops expect */
        s->row_fft(rr, ri, s->row_or, s->row_oi);
        memcpy(rr, s->row_or, sizeof(double) * (size_t)W);
        memcpy(ri, s->row_oi, sizeof(double) * (size_t)W);
    }
    for (int c = 0; c < W; ++c) {
        for (int r = 0; r < H; ++r) {
            s->tmp_re[r] = re[(size_t)r * W + c];
            s->tmp_im[r] = im[(size_t)r * W + c];
        }
        s->col_fft(s->tmp_re, s->tmp_im, s->col_or, s->col_oi);
        for (int r = 0; r < H; ++r) {
            re[(size_t)r * W + c] = s->col_or[r];
            im[(size_t)r * W + c] = s->col_oi[r];
        }
    }
}

/* =========================================================================
 * Seeds (ports of _peak_mu and the ring-1 gain median)
 * ========================================================================= */
static double subpix_3pt(double lo, double ce, double hi, int interior) {
    if (!interior || lo <= 0.0 || ce <= 0.0 || hi <= 0.0) return 0.0;
    double ln_l = log(lo), ln_c = log(ce), ln_r = log(hi);
    double denom = 2.0 * (ln_l - 2.0 * ln_c + ln_r);
    if (fabs(denom) <= 1e-12) return 0.0;
    return (ln_l - ln_r) / denom;
}

/* 3-point log-Gaussian sub-pixel peak of the RAW (unshifted) R_AB plane.
 * cy/cx are the centre pixel indices (H/2, W/2); returns offsets from them. */
static void peak_mu_seed(const double *R_AB, int H, int W, int cy, int cx,
                         double *mu_x0, double *mu_y0) {
    int pi = 0;
    double best = R_AB[0];
    const int P = H * W;
    for (int i = 1; i < P; ++i)
        if (R_AB[i] > best) { /* strict '>' == np.argmax first occurrence */
            best = R_AB[i];
            pi = i;
        }
    int py = pi / W, px = pi % W;
    double sub_x = 0.0, sub_y = 0.0;
    if (px > 0 && px < W - 1)
        sub_x = subpix_3pt(R_AB[(size_t)py * W + px - 1], R_AB[(size_t)py * W + px],
                           R_AB[(size_t)py * W + px + 1], 1);
    if (py > 0 && py < H - 1)
        sub_y = subpix_3pt(R_AB[(size_t)(py - 1) * W + px], R_AB[(size_t)py * W + px],
                           R_AB[(size_t)(py + 1) * W + px], 1);
    *mu_x0 = ((double)px + sub_x) - (double)cx;
    *mu_y0 = ((double)py + sub_y) - (double)cy;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* median of |T| on the 8 bins ring-1 around DC (natural order: indices
 * +-1 mod N in each dim), np.median convention (mean of 4th/5th of 8). */
static double gain_seed(const scratch *s) {
    const int H = s->H, W = s->W;
    double vals[8];
    int nv = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) continue;
            int r = (dy + H) % H, c = (dx + W) % W;
            size_t i = (size_t)r * W + c;
            vals[nv++] = hypot(s->Tre[i], s->Tim[i]);
        }
    qsort(vals, 8, sizeof(double), cmp_double);
    double med = 0.5 * (vals[3] + vals[4]);
    if (med < 1e-2) med = 1e-2;
    if (med > 10.0) med = 10.0;
    return med;
}

/* =========================================================================
 * Noise-floor regressor (port of the coloured-floor block in _prepare_chunk)
 * ========================================================================= */

/* Materialise the per-window floor regressor into s->Dv (natural order).
 *
 * P_centred is this window's (H*W) analytic floor shape in Python's CENTRED
 * layout, or NULL for the flat floor. Python:
 *     coloured:  D = P / max(F_ref, 1e-30)
 *     flat:      the model divides by F_ref directly (D == 1/F_ref)
 * Both collapse to att = 1 - N0*Dv, so the inner loops never branch.
 *
 * The centred->natural conversion is the same index map fft2_centered uses on
 * the planes. That map is ifftshift (roll by -(n/2)), which is the correct
 * inverse of the centring for ANY n, odd or even — no power-of-two or even-n
 * assumption is involved, and this file accepts 12/24/48/96 which are not
 * powers of two. s->Pn keeps P in natural order for the seed, because the
 * seed clamps P where D clamps F_ref — the two are not reciprocals. */
static void fill_regressor(scratch *s, const double *P_centred) {
    const int H = s->H, W = s->W;
    const int hh = H / 2, hw = W / 2;
    if (!P_centred) {
        for (int i = 0; i < s->P; ++i) {
            const double fr = s->Fr[i];
            s->Dv[i] = 1.0 / (fr > 1e-30 ? fr : 1e-30);
        }
        return;
    }
    for (int r = 0; r < H; ++r) {
        const double *src = P_centred + (size_t)((r + hh) % H) * W;
        for (int c = 0; c < W; ++c) {
            const size_t i = (size_t)r * W + c;
            const double p = src[(c + hw) % W];
            const double fr = s->Fr[i];
            s->Pn[i] = p;
            s->Dv[i] = p / (fr > 1e-30 ? fr : 1e-30);
        }
    }
}

/* Coloured N0 seed: clip(median(F_ref/max(P,1e-30)) over |k| >= 0.35).
 * np.median convention — mean of the two central order statistics for an even
 * count, the central one for odd. Call only after a coloured fill_regressor. */
static double n0_seed_coloured(scratch *s) {
    const int n = s->n_tail;
    for (int t = 0; t < n; ++t) {
        const int i = s->tail_idx[t];
        const double p = s->Pn[i];
        s->tailbuf[t] = s->Fr[i] / (p > 1e-30 ? p : 1e-30);
    }
    qsort(s->tailbuf, (size_t)n, sizeof(double), cmp_double);
    double med = (n & 1) ? s->tailbuf[n / 2]
                         : 0.5 * (s->tailbuf[n / 2 - 1] + s->tailbuf[n / 2]);
    if (med < COLOURED_SEED_CLIP_LO) med = COLOURED_SEED_CLIP_LO;
    if (med > COLOURED_SEED_CLIP_HI) med = COLOURED_SEED_CLIP_HI;
    return med;
}

/* =========================================================================
 * Residual + JtJ/grad accumulation (port of _resid_jac_v6/_resid_jac_quartic)
 * ========================================================================= */

/* cost only (trial-step evaluation) */
static double eval_cost(const scratch *s, const double *x, int K,
                        int use_kx4, int use_ky4) {
    const int H = s->H, W = s->W;
    const double mux = x[0], muy = x[1], Sxx = x[2], Syy = x[3], Sxy = x[4];
    const double g = x[5], N0 = x[6];
    double b4x = 0.0, b4y = 0.0;
    int col = 7;
    if (use_kx4) b4x = x[col++];
    if (use_ky4) b4y = x[col++];
    (void)K;
    double cost = 0.0;
    for (int r = 0; r < H; ++r) {
        const double kyv = s->ky[r], ky2 = kyv * kyv;
        const size_t row0 = (size_t)r * W;
        for (int c = 0; c < W; ++c) {
            const double w = s->Wgt[row0 + c];
            if (w == 0.0) continue; /* exact-zero residual terms in Python */
            const double kxv = s->kx[c], kx2 = kxv * kxv;
            double arg = -TWO_PI2 *
                         (Sxx * kx2 + 2.0 * Sxy * kxv * kyv + Syy * ky2);
            if (use_kx4) arg -= b4x * kx2 * kx2;
            if (use_ky4) arg -= b4y * ky2 * ky2;
            if (arg > EXP_ARG_MAX) arg = EXP_ARG_MAX;
            const double decay = exp(arg);
            const double att = 1.0 - N0 * s->Dv[row0 + c];
            const double phase = -TWO_PI * (kxv * mux + kyv * muy);
            const double gd = g * decay * att;
            const double rre = w * (s->Tre[row0 + c] - gd * cos(phase));
            const double rim = w * (s->Tim[row0 + c] - gd * sin(phase));
            cost += rre * rre + rim * rim;
        }
    }
    return cost;
}

/* cost + JtJ (upper triangle, row-major KxK) + gradient Jt*r */
static double eval_cost_jtj(const scratch *s, const double *x, int K,
                            int use_kx4, int use_ky4, double *A, double *g_vec) {
    const int H = s->H, W = s->W;
    const double mux = x[0], muy = x[1], Sxx = x[2], Syy = x[3], Sxy = x[4];
    const double g = x[5], N0 = x[6];
    double b4x = 0.0, b4y = 0.0;
    int col = 7;
    if (use_kx4) b4x = x[col++];
    if (use_ky4) b4y = x[col++];

    memset(A, 0, sizeof(double) * (size_t)(K * K));
    memset(g_vec, 0, sizeof(double) * (size_t)K);
    double cost = 0.0;
    double Jre[KMAX], Jim[KMAX];

    for (int r = 0; r < H; ++r) {
        const double kyv = s->ky[r], ky2 = kyv * kyv;
        const size_t row0 = (size_t)r * W;
        for (int c = 0; c < W; ++c) {
            const double w = s->Wgt[row0 + c];
            if (w == 0.0) continue;
            const double kxv = s->kx[c], kx2 = kxv * kxv;
            const double kxky = kxv * kyv;
            double arg = -TWO_PI2 * (Sxx * kx2 + 2.0 * Sxy * kxky + Syy * ky2);
            if (use_kx4) arg -= b4x * kx2 * kx2;
            if (use_ky4) arg -= b4y * ky2 * ky2;
            if (arg > EXP_ARG_MAX) arg = EXP_ARG_MAX;
            const double decay = exp(arg);
            const double dv = s->Dv[row0 + c];
            const double att = 1.0 - N0 * dv;
            const double phase = -TWO_PI * (kxv * mux + kyv * muy);
            const double cosp = cos(phase), sinp = sin(phase);
            const double gd = g * decay * att;
            const double rre = w * (s->Tre[row0 + c] - gd * cosp);
            const double rim = w * (s->Tim[row0 + c] - gd * sinp);
            cost += rre * rre + rim * rim;

            /* residual Jacobian (J = d r / d x), mirrors _resid_jac_v6:  */
            const double dpx = -TWO_PI * kxv; /* dphase/dmu_x */
            const double dpy = -TWO_PI * kyv;
            Jre[0] = -w * gd * (-sinp) * dpx;
            Jim[0] = -w * gd * cosp * dpx;
            Jre[1] = -w * gd * (-sinp) * dpy;
            Jim[1] = -w * gd * cosp * dpy;
            const double ddx = gd * (-TWO_PI2 * kx2); /* dmodel/dSxx */
            Jre[2] = -w * ddx * cosp;
            Jim[2] = -w * ddx * sinp;
            const double ddy = gd * (-TWO_PI2 * ky2);
            Jre[3] = -w * ddy * cosp;
            Jim[3] = -w * ddy * sinp;
            const double ddc = gd * (-TWO_PI2 * 2.0 * kxky);
            Jre[4] = -w * ddc * cosp;
            Jim[4] = -w * ddc * sinp;
            Jre[5] = -w * decay * att * cosp; /* dmodel/dg */
            Jim[5] = -w * decay * att * sinp;
            const double dN = g * decay * (-dv); /* dmodel/dN0 */
            Jre[6] = -w * dN * cosp;
            Jim[6] = -w * dN * sinp;
            int cc = 7;
            if (use_kx4) {
                const double db = gd * (-(kx2 * kx2));
                Jre[cc] = -w * db * cosp;
                Jim[cc] = -w * db * sinp;
                ++cc;
            }
            if (use_ky4) {
                const double db = gd * (-(ky2 * ky2));
                Jre[cc] = -w * db * cosp;
                Jim[cc] = -w * db * sinp;
            }

            for (int i = 0; i < K; ++i) {
                g_vec[i] += Jre[i] * rre + Jim[i] * rim;
                for (int j = i; j < K; ++j)
                    A[(size_t)i * K + j] += Jre[i] * Jre[j] + Jim[i] * Jim[j];
            }
        }
    }
    /* mirror to lower triangle */
    for (int i = 1; i < K; ++i)
        for (int j = 0; j < i; ++j)
            A[(size_t)i * K + j] = A[(size_t)j * K + i];
    return cost;
}

/* Cholesky solve of (KxK) M d = b, in-place on copies. Returns 0 on success,
 * -1 if not positive definite (step is then rejected, mirroring the NumPy
 * LinAlgError -> NaN-delta -> rejected path). */
static int chol_solve(int K, const double *M, const double *b, double *d) {
    double L[KMAX * KMAX];
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = M[(size_t)i * K + j];
            for (int k = 0; k < j; ++k)
                sum -= L[(size_t)i * K + k] * L[(size_t)j * K + k];
            if (i == j) {
                if (sum <= 0.0 || !isfinite(sum)) return -1;
                L[(size_t)i * K + i] = sqrt(sum);
            } else {
                L[(size_t)i * K + j] = sum / L[(size_t)j * K + j];
            }
        }
    }
    double y[KMAX];
    for (int i = 0; i < K; ++i) {
        double sum = b[i];
        for (int k = 0; k < i; ++k) sum -= L[(size_t)i * K + k] * y[k];
        y[i] = sum / L[(size_t)i * K + i];
    }
    for (int i = K - 1; i >= 0; --i) {
        double sum = y[i];
        for (int k = i + 1; k < K; ++k) sum -= L[(size_t)k * K + i] * d[k];
        d[i] = sum / L[(size_t)i * K + i];
    }
    return 0;
}

/* =========================================================================
 * Per-window fit
 * ========================================================================= */
typedef struct {
    double x[KMAX];
    double cost;
    int niter;
    int converged;
} lm_result;

static void lm_fit_window(const scratch *s, int K, int use_kx4, int use_ky4,
                          const double *seed, const double *lo, const double *hi,
                          lm_result *out) {
    double x[KMAX];
    for (int i = 0; i < K; ++i) {
        x[i] = seed[i];
        if (x[i] < lo[i]) x[i] = lo[i];
        if (x[i] > hi[i]) x[i] = hi[i];
    }
    double cost = eval_cost(s, x, K, use_kx4, use_ky4);
    out->niter = 0;
    out->converged = 0;
    if (!isfinite(cost)) { /* active = isfinite(cost) in the original */
        memcpy(out->x, x, sizeof(double) * (size_t)K);
        out->cost = cost;
        return;
    }

    double lam = LAM_INIT;
    double A[KMAX * KMAX], g_vec[KMAX], Ad[KMAX * KMAX];
    int have_jac = 0;

    for (int it = 0; it < MAIN_MAX_ITER; ++it) {
        if (!have_jac) {
            /* Discard the returned cost. It is already correct at this x — from
             * the seed evaluation or from the accepted costn — and _batched_lm
             * likewise never refreshes cost from its Jacobian pass. Assigning it
             * here would make the accept test (costn <= cost) compare an
             * eval_cost value against an eval_cost_jtj value, so a future model
             * edit that touched one function and not the other by 1 ULP would
             * silently change which LM steps are accepted. */
            (void)eval_cost_jtj(s, x, K, use_kx4, use_ky4, A, g_vec);
            have_jac = 1;
        }
        memcpy(Ad, A, sizeof(double) * (size_t)(K * K));
        for (int i = 0; i < K; ++i) {
            double diag = A[(size_t)i * K + i];
            if (diag < 1e-30) diag = 1e-30;
            Ad[(size_t)i * K + i] += lam * diag;
        }
        double delta[KMAX], neg_g[KMAX], xn[KMAX];
        for (int i = 0; i < K; ++i) neg_g[i] = -g_vec[i];
        int solved = chol_solve(K, Ad, neg_g, delta) == 0;

        int acc = 0;
        double costn = INFINITY;
        if (solved) {
            int finite_xn = 1;
            for (int i = 0; i < K; ++i) {
                xn[i] = x[i] + delta[i];
                if (xn[i] < lo[i]) xn[i] = lo[i];
                if (xn[i] > hi[i]) xn[i] = hi[i];
                if (!isfinite(xn[i])) finite_xn = 0;
            }
            if (finite_xn) {
                costn = eval_cost(s, xn, K, use_kx4, use_ky4);
                acc = isfinite(costn) && costn <= cost;
            }
        }

        ++out->niter;
        if (acc) {
            int small_step = 1;
            for (int i = 0; i < K; ++i)
                if (fabs(xn[i] - x[i]) > LM_XTOL * (LM_XTOL + fabs(xn[i]))) {
                    small_step = 0;
                    break;
                }
            double ref = cost > 1e-300 ? cost : 1e-300;
            int small_drop = (cost - costn) <= LM_FTOL * ref;
            memcpy(x, xn, sizeof(double) * (size_t)K);
            cost = costn;
            lam /= 3.0;
            if (lam < LAM_MIN) lam = LAM_MIN;
            have_jac = 0; /* parameters moved: JtJ must be rebuilt */
            if (small_step || small_drop) {
                out->converged = 1;
                break;
            }
        } else {
            lam *= 10.0;
            /* ENOPROG: stuck, deactivate unconverged. Production cannot
             * distinguish this from EMAXITER either (both leave conv=False),
             * so it is not recorded — the acceptance rule downstream uses
             * niter >= MAIN_MAX_ITER, which an ENOPROG break fails. */
            if (lam > LAM_ENOPROG) break;
        }
    }
    memcpy(out->x, x, sizeof(double) * (size_t)K);
    out->cost = cost;
}

/* =========================================================================
 * Public entry point
 * ========================================================================= */
KSPACE_EXPORT int kspace_lm_fit_batch(
    const double *R_AA, /* (n_windows, H, W) C-contiguous */
    const double *R_BB, const double *R_AB,
    const unsigned char *mask_flat, /* 1 = masked (skip) */
    /* (n_windows, H*W) coloured-floor shape in Python's CENTRED layout, or
     * NULL for the flat floor — mirrors the Python D=None switch. */
    const double *P_win,
    int n_windows, int corr_h, int corr_w,
    int use_kx4, int use_ky4,
    int n_threads, /* <=0: OpenMP default */
    /* outputs — caller allocates; layouts mirror fit_windows_kspace_lm */
    double *gauss_flat,         /* (n_windows, 16) */
    int *status_flat,           /* (n_windows,) */
    double *initial_guess_flat, /* (n_windows, 16) */
    double *diag_gain, double *diag_N0, double *diag_b4x, double *diag_b4y,
    double *diag_cost_per_pt, int *diag_n_valid, int *diag_iter,
    unsigned char *diag_conv) {
    if (!cfft_for(corr_h) || !cfft_for(corr_w))
        return -1; /* axis length outside BUILT_FFT_SIZES */

    const int P = corr_h * corr_w;
    const int cy = corr_h / 2, cx = corr_w / 2;
    const double center_x = corr_w / 2.0 + 1.0; /* 1-based centres (C conv.) */
    const double center_y = corr_h / 2.0 + 1.0;
    const int K = 7 + (use_kx4 ? 1 : 0) + (use_ky4 ? 1 : 0);

    double lo[KMAX], hi[KMAX];
    lo[0] = -INFINITY; lo[1] = -INFINITY; lo[2] = 0.0; lo[3] = 0.0;
    lo[4] = -INFINITY; lo[5] = GAIN_LO;   lo[6] = 0.0;
    hi[0] = INFINITY;  hi[1] = INFINITY;  hi[2] = INFINITY; hi[3] = INFINITY;
    hi[4] = INFINITY;  hi[5] = GAIN_HI;
    /* coloured: N0 bounded [0, 10] in normalized F_ref units (offline free arm) */
    hi[6] = P_win ? COLOURED_N0_HI : INFINITY;
    for (int i = 7; i < K; ++i) { lo[i] = -INFINITY; hi[i] = INFINITY; }

    /* Coloured seeding needs at least one |k| >= COLOURED_SEED_KR_MIN bin.
     * Mirrors the ValueError at kspace_lm_fitting.py:470-474 — fail loudly
     * rather than seeding N0 from an empty band.
     *
     * This cannot fire for any BUILT_FFT_SIZES axis length: the smallest is 8,
     * whose fftfreq grid reaches +-0.375 > 0.35. It is kept as a guard against
     * a future smaller size or a lowered COLOURED_SEED_KR_MIN, and is checked
     * here (before the parallel region) so the failure is a clean return code
     * rather than an error raised inside an OpenMP worksharing construct.
     * The per-thread tail index itself is built once in scratch_init. */
    if (P_win) {
        scratch probe;
        if (scratch_init(&probe, corr_h, corr_w) != 0) {
            scratch_free(&probe);
            return -2;
        }
        const int n_tail = probe.n_tail;
        scratch_free(&probe);
        if (n_tail == 0) return -3;
    }

    /* gauss_flat defaults apply to EVERY window, but initial_guess_flat's do
     * not: production writes it per-chunk over the unmasked index set only
     * (kspace_lm_fitting.py:754-761), leaving masked rows all-zero, and mirrors
     * gauss_flat into it solely in the all-masked early return (:722). Writing
     * the defaults unconditionally would emit NaN + centres where production
     * emits 0.0 on every masked window. */
    int n_proc = 0;
    for (int w = 0; w < n_windows; ++w) n_proc += !mask_flat[w];
    const int all_masked = (n_proc == 0);

    /* static parts of the 16-column contract */
    for (int w = 0; w < n_windows; ++w) {
        double *gf = gauss_flat + (size_t)w * 16;
        double *ig = initial_guess_flat + (size_t)w * 16;
        for (int c = 0; c < 16; ++c) { gf[c] = 0.0; ig[c] = 0.0; }
        gf[6] = gf[7] = gf[8] = NAN; /* particle-size slots by contract */
        gf[12] = center_x; gf[13] = center_y; gf[14] = center_x; gf[15] = center_y;
        if (all_masked || !mask_flat[w]) {
            ig[6] = ig[7] = ig[8] = NAN;
            ig[12] = center_x; ig[13] = center_y;
            ig[14] = center_x; ig[15] = center_y;
        }
        status_flat[w] = STATUS_MASKED;
        diag_gain[w] = NAN; diag_N0[w] = NAN;
        diag_b4x[w] = NAN;  diag_b4y[w] = NAN;
        diag_cost_per_pt[w] = NAN;
        diag_n_valid[w] = 0; diag_iter[w] = 0; diag_conv[w] = 0;
    }

    int had_alloc_error = 0;

    /* num_threads() is scoped to this region; omp_set_num_threads() would
     * mutate the process-wide ICV and leave a later call (or
     * kspace_lm_fit_max_threads) reporting whatever the last caller asked for. */
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads > 0 ? n_threads : omp_get_max_threads())
#endif
    {
        scratch s;
        int s_ok = scratch_init(&s, corr_h, corr_w) == 0;
        if (!s_ok) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
            had_alloc_error = 1;
        }

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
        for (int w = 0; w < n_windows; ++w) {
            if (!s_ok || mask_flat[w]) continue;
            double *gf = gauss_flat + (size_t)w * 16;
            double *ig = initial_guess_flat + (size_t)w * 16;
            const double *pAA = R_AA + (size_t)w * P;
            const double *pBB = R_BB + (size_t)w * P;
            const double *pAB = R_AB + (size_t)w * P;

            /* ---- amplitudes + low-SNR gate (centre of the RAW plane) ---- */
            const double amp_A = pAA[(size_t)cy * corr_w + cx];
            const double amp_B = pBB[(size_t)cy * corr_w + cx];
            double amp_AB = pAB[0];
            for (int i = 1; i < P; ++i)
                if (pAB[i] > amp_AB) amp_AB = pAB[i];
            gf[0] = amp_A; gf[1] = amp_B; gf[2] = amp_AB;
            ig[0] = amp_A; ig[1] = amp_B; ig[2] = amp_AB;
            const int amp_ok = (amp_A >= 1e-12) && (amp_B >= 1e-12);

            /* ---- FFTs + transfer ratio + weights (natural k order) ---- */
            fft2_centered(&s, pAA, 0);
            fft2_centered(&s, pBB, 1);
            fft2_centered(&s, pAB, 2);
            double fr_max = 0.0;
            for (int i = 0; i < P; ++i) {
                const double fa = hypot(s.fre[0][i], s.fim[0][i]);
                const double fb = hypot(s.fre[1][i], s.fim[1][i]);
                const double fr = sqrt(fa * fb);
                s.Fr[i] = fr;
                const double inv = 1.0 / (fr > 1e-30 ? fr : 1e-30);
                s.Tre[i] = s.fre[2][i] * inv;
                s.Tim[i] = s.fim[2][i] * inv;
                if (fr > fr_max) fr_max = fr;
            }
            if (fr_max < 1e-30) fr_max = 1e-30;
            int n_valid = 0;
            for (int i = 0; i < P; ++i) {
                const int valid = (s.Fr[i] > 1e-12) && (i != 0); /* DC at 0 */
                s.Wgt[i] = valid ? s.Fr[i] / fr_max : 0.0;
                n_valid += valid;
            }
            diag_n_valid[w] = n_valid;

            /* ---- seeds: computed for EVERY processed window (the Python
             * _prepare_chunk seeds the whole chunk before viability filters),
             * so initial_guess columns match even for gated windows ---- */
            double seed[KMAX] = {0};
            peak_mu_seed(pAB, corr_h, corr_w, cy, cx, &seed[0], &seed[1]);
            seed[2] = SIGMA_SEED_XX;
            seed[3] = SIGMA_SEED_YY;
            seed[4] = 0.0;
            seed[6] = 0.0;
            /* b4 seeds are 0 (zero-initialised) */
            ig[9] = seed[2]; ig[10] = seed[3];
            ig[14] = center_x + seed[0];
            ig[15] = center_y + seed[1];

            /* status quirks mirror _prepare_chunk: amp fail -> LOW_SNR,
             * n_valid-only fail -> NO_CONVERGE */
            if (!amp_ok) { status_flat[w] = STATUS_LOW_SNR; continue; }
            if (n_valid < MIN_VALID_PTS) {
                status_flat[w] = STATUS_NO_CONVERGE;
                continue;
            }
            /* gain seed only steers the LM — not recorded anywhere — so it is
             * computed after the gates (no observable difference) */
            seed[5] = gain_seed(&s);

            /* ---- floor regressor + coloured N0 seed ----
             * Same reasoning as the gain seed: neither Dv nor seed[6] reaches
             * initial_guess_flat, so building them after the gates is
             * unobservable and skips the work on gated windows. */
            fill_regressor(&s, P_win ? P_win + (size_t)w * P : NULL);
            if (P_win) seed[6] = n0_seed_coloured(&s);

            /* ---- LM ---- */
            lm_result res;
            lm_fit_window(&s, K, use_kx4, use_ky4, seed, lo, hi, &res);
            diag_iter[w] = res.niter;
            diag_conv[w] = (unsigned char)res.converged;
            const double cpp = res.cost / (double)n_valid;
            diag_cost_per_pt[w] = cpp;

            const int fit_ok =
                res.converged ||
                (res.niter >= MAIN_MAX_ITER && cpp < COST_PER_PT_ACCEPT);
            if (!fit_ok) { status_flat[w] = STATUS_NO_CONVERGE; continue; }

            const double mux = res.x[0], muy = res.x[1];
            const int big = fabs(mux) > MAX_DISP_FRAC * corr_w ||
                            fabs(muy) > MAX_DISP_FRAC * corr_h;
            if (big) { status_flat[w] = STATUS_BIG_DISP; continue; }

            status_flat[w] = STATUS_SUCCESS;
            double Sxx = res.x[2] > 0.0 ? res.x[2] : 0.0; /* belt+braces */
            double Syy = res.x[3] > 0.0 ? res.x[3] : 0.0;
            gf[9] = Sxx; gf[10] = Syy; gf[11] = res.x[4];
            gf[14] = center_x + mux;
            gf[15] = center_y + muy;
            diag_gain[w] = res.x[5];
            diag_N0[w] = res.x[6];
            int cc = 7;
            if (use_kx4) diag_b4x[w] = res.x[cc++];
            if (use_ky4) diag_b4y[w] = res.x[cc];
        }

        scratch_free(&s);
    }

    return had_alloc_error ? -2 : 0;
}

/* Introspection helpers for the harness */
KSPACE_EXPORT int kspace_lm_fit_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Single-window prep introspection so the harness can gate the floor
 * regressor in isolation. Returns Dv in NATURAL order (compare against
 * np.fft.ifftshift of Python's centred D) and the coloured N0 seed; with
 * P_centred == NULL it returns the flat 1/F_ref regressor and a seed of 0.
 *
 * This exists because a centred/natural mix-up in P does not crash — it
 * produces a plausible but wrong fit. Gating Dv directly turns that into a
 * pinpointed failure instead of a diffuse parameter mismatch. */
KSPACE_EXPORT int kspace_debug_prep(const double *R_AA, const double *R_BB,
                                    const double *R_AB, const double *P_centred,
                                    int H, int W, double *out_Dv,
                                    double *out_n0_seed) {
    if (!cfft_for(H) || !cfft_for(W)) return -1;
    scratch s;
    if (scratch_init(&s, H, W) != 0) {
        scratch_free(&s);
        return -2;
    }
    if (P_centred && s.n_tail == 0) {
        scratch_free(&s);
        return -3;
    }
    fft2_centered(&s, R_AA, 0);
    fft2_centered(&s, R_BB, 1);
    fft2_centered(&s, R_AB, 2);
    for (int i = 0; i < s.P; ++i) {
        const double fa = hypot(s.fre[0][i], s.fim[0][i]);
        const double fb = hypot(s.fre[1][i], s.fim[1][i]);
        s.Fr[i] = sqrt(fa * fb);
    }
    fill_regressor(&s, P_centred);
    *out_n0_seed = P_centred ? n0_seed_coloured(&s) : 0.0;
    memcpy(out_Dv, s.Dv, sizeof(double) * (size_t)s.P);
    scratch_free(&s);
    return 0;
}

/* FFT-only entry so the harness can gate the transform against np.fft:
 * out_re/out_im are (H, W) natural-order spectra of the centred transform. */
KSPACE_EXPORT int kspace_fft2_centered(const double *plane, int H, int W,
                                       double *out_re, double *out_im) {
    if (!cfft_for(H) || !cfft_for(W)) return -1;
    scratch s;
    if (scratch_init(&s, H, W) != 0) {
        scratch_free(&s);
        return -2;
    }
    fft2_centered(&s, plane, 0);
    memcpy(out_re, s.fre[0], sizeof(double) * (size_t)H * W);
    memcpy(out_im, s.fim[0], sizeof(double) * (size_t)H * W);
    scratch_free(&s);
    return 0;
}
