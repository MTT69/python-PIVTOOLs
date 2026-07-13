#include "peak_locate_lm_batch.h"
#include "common.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

/******************************************************************************
 * Batched one-window-per-lane LM Gaussian peak fitter — implementation.
 *
 * STRUCTURE (deliberately mirrors peak_locate_lm.c 1:1 so every line has a
 * scalar oracle counterpart; the accepted code duplication matches the
 * scalar file's own 4/5/6 duplication):
 *
 *   Phase 1 (scalar, per lane): peak search, validity gate (NaN sentinel),
 *     5x5 subwindow extraction, 3-point seed + clamps — verbatim scalar
 *     logic, so NaN search masks agree with the oracle BY CONSTRUCTION.
 *   Phase 2 (vector lockstep): the LM iterations only. Per-lane lambda,
 *     accept/stagnation/convergence masks, masked double Cholesky, Lever-2
 *     pred cache with accept-masked promotion.
 *   Phase 3 (scalar, per lane): trust-rule writeback or NaN sentinel.
 *
 * THE LEVER-2 BLEND CONTRACT (the #1 foreseeable bug — read before editing):
 *   resid_pass writes pred_trial[] UNCONDITIONALLY for all lanes (same as
 *   the scalar pred_buf contract). Only ACCEPTED lanes may promote
 *   pred_trial -> pred_cur. jac_pass reads pred_cur — never pred_trial —
 *   so on non-accepted lanes the full-width Jacobian recompute reproduces
 *   the exact same doubles those lanes already held (their params AND
 *   pred_cur are unchanged). That value-idempotence is why no JtJ blend is
 *   needed, and it holds ONLY while pred_cur feeds the Jacobian.
 *
 * NUMERICAL PARITY WITH THE SCALAR ORACLE:
 *   - JtJ/Jtr accumulate in double vectors; Cholesky solves in double;
 *     params/residual/pred/exp stay float (peak_locate_lm.c contract).
 *   - The trust rule, clamps, Marquardt schedule and tolerances are copied
 *     verbatim; the constants below MUST track peak_locate_lm.c.
 *   - exp is pk_vexpf (Cephes degree-5, max rel err 8.4e-8 measured); build
 *     with -DPIV_BATCH_LIBM_EXP for the per-lane-libm reference flavor used
 *     by the tight gate thresholds.
 ******************************************************************************/

#include "peak_simd.h"

#if !PK_BATCH_AVAILABLE

/* Plain MSVC cl stub: the batch fitter is unavailable. Loud, never silent —
 * the runtime selector refuses impl=batch when available()==0; this entry
 * only exists so the DLL exports resolve. */
int peakfit_batch_available(void) { return 0; }
int peakfit_batch_lanes(void)     { return 0; }

void lsqpeaklocate_lm_batch(const float *planes, int L_real,
                            const int *N, int iFitType,
                            float *peak_loc, float *std_dev,
                            const float *fPlaneWeight)
{
	(void)planes; (void)N; (void)iFitType; (void)fPlaneWeight;
	fprintf(stderr, "lsqpeaklocate_lm_batch: batch fitter not compiled in "
	                "(plain MSVC cl build) - this call is a selector bug\n");
	for (int l = 0; l < L_real; ++l) {
		peak_loc[0 * 1 + l] = NAN;  /* lanes unknown in stub: width 1 layout */
		peak_loc[1 * 1 + l] = NAN;
	}
}

#else /* PK_BATCH_AVAILABLE */

int peakfit_batch_available(void) { return 1; }
int peakfit_batch_lanes(void)     { return PK_LANES; }

/* ── constants: MUST track peak_locate_lm.c ─────────────────────────────── */
#define LM_ACCEPT_RESID_FRAC 1e-3f
#define LM_MAX_ITER 20
#define LM_TOL 1e-6f

#define NSUB (PKSIZE_X * PKSIZE_Y)   /* 25 */

/* packed lower-triangle index for the symmetric JtJ (j <= i) */
#define PIDX(i, j) ((i) * ((i) + 1) / 2 + (j))
#define NPACK6 21
#define NPARAM_MAX 6

/* ── scalar helpers copied verbatim from peak_locate_lm.c ───────────────── */

/* 3-point parabolic estimator (verbatim; operates on one lane's subwindow) */
static void threept_estimate_b(const float *xcorr, const int *N, float *peak_loc,
                               float *A_out, float *sx_out, float *sy_out)
{
	float x_fit[3], y_fit[3];
	int i;

	for (i = 0; i < 3; ++i) {
		x_fit[i] = xcorr[(i - 1 + (N[0]-1)/2) * N[1] + (N[1]-1)/2];
		y_fit[i] = xcorr[(N[0]-1)/2 * N[1] + (i - 1 + (N[1]-1)/2)];
		x_fit[i] = (float)log((x_fit[i] < FLT_EPSILON) ? FLT_EPSILON : x_fit[i]);
		y_fit[i] = (float)log((y_fit[i] < FLT_EPSILON) ? FLT_EPSILON : y_fit[i]);
	}

	float denom_x = 2*x_fit[0] - 4*x_fit[1] + 2*x_fit[2];
	float denom_y = 2*y_fit[0] - 4*y_fit[1] + 2*y_fit[2];

	if (fabs(denom_x) > FLT_EPSILON)
		peak_loc[0] = (x_fit[0] - x_fit[2]) / denom_x;
	else
		peak_loc[0] = 0.0f;

	if (fabs(denom_y) > FLT_EPSILON)
		peak_loc[1] = (y_fit[0] - y_fit[2]) / denom_y;
	else
		peak_loc[1] = 0.0f;

	*A_out = xcorr[(N[0]-1)/2 * N[1] + (N[1]-1)/2];
	*sx_out = (float)sqrt(-4.0f / (denom_x + FLT_EPSILON));
	*sy_out = (float)sqrt(-4.0f / (denom_y + FLT_EPSILON));
}

/* ── masked vector Cholesky solve: (JtJ + lambda*diag) delta = -Jtr ──────── */
/* Packed lower-triangle double vectors. No per-lane early-out: lanes whose
 * pivot goes non-positive are flagged in the returned mask and their pivot
 * is forced to 1.0 (sticky) so all arithmetic stays finite; the caller
 * marks them done and the trust rule judges their frozen params — mirroring
 * the scalar `if(solve!=0) break`. */
static pk_vi solve_lm_step_vec(const pk_vd *JtJ, const pk_vd *Jtr,
                               pk_vf lambda, pk_vf *delta, int n)
{
	pk_vd A[NPACK6], L[NPACK6], y[NPARAM_MAX], d[NPARAM_MAX];
	int i, j, k;

	memcpy(A, JtJ, (size_t)PIDX(n - 1, n - 1) * sizeof(pk_vd) + sizeof(pk_vd));

	pk_vd lam1 = pk_vd_set1(1.0) + pk_cvt_f2d(lambda);
	for (i = 0; i < n; ++i)
		A[PIDX(i, i)] = A[PIDX(i, i)] * lam1;

	pk_vl bad = pk_castd2l(pk_vd_zero());   /* all-zero mask */
	pk_vd one_d = pk_vd_set1(1.0);
	pk_vd zero_d = pk_vd_zero();

	for (i = 0; i < n; ++i) {
		for (j = 0; j <= i; ++j) {
			pk_vd sum = A[PIDX(i, j)];
			for (k = 0; k < j; ++k)
				sum = sum - L[PIDX(i, k)] * L[PIDX(j, k)];
			if (i == j) {
				bad = bad | (sum <= zero_d);          /* sticky non-PD flag  */
				sum = pk_seld(bad, one_d, sum);       /* sanitize bad lanes  */
				L[PIDX(i, i)] = pk_sqrtd(sum);
			} else {
				L[PIDX(i, j)] = sum / L[PIDX(j, j)];
			}
		}
	}

	/* forward substitution: L y = -Jtr */
	for (i = 0; i < n; ++i) {
		pk_vd sum = zero_d - Jtr[i];
		for (j = 0; j < i; ++j)
			sum = sum - L[PIDX(i, j)] * y[j];
		y[i] = sum / L[PIDX(i, i)];
	}

	/* back substitution: L^T d = y */
	for (i = n - 1; i >= 0; --i) {
		pk_vd sum = y[i];
		for (j = i + 1; j < n; ++j)
			sum = sum - L[PIDX(j, i)] * d[j];   /* L[j][i], j > i */
		d[i] = sum / L[PIDX(i, i)];
	}

	for (i = 0; i < n; ++i)
		delta[i] = pk_cvt_d2f(d[i]);

	return pk_mask_l2i(bad);
}

/* ── pixel coordinate table for the 5x5 subwindow (i, j in [-2, 2]) ──────── */
static inline float pix_i(int p) { return (float)(p / PKSIZE_Y - (PKSIZE_X - 1) / 2); }
static inline float pix_j(int p) { return (float)(p % PKSIZE_Y - (PKSIZE_Y - 1) / 2); }

/******************************************************************************
 * Residual / Jacobian passes per model. resid_* is the HOT path (every trial,
 * including rejects): float only, one pk_vexpf per pixel, writes pred_trial
 * unconditionally. jac_* is the COLD path (once per accepted step): reads the
 * pred cache (zero exp calls — Lever 2), accumulates double vectors.
 * Pixel order matches the scalar loops (row-major) so per-lane float
 * accumulation order is identical to the oracle's.
 ******************************************************************************/

/* -- 4-DOF circular: A * exp(-((i-i0)^2 + (j-j0)^2) / s^2) ----------------- */
static pk_vf resid_pass_4(const pk_vf *sub, pk_vf A, pk_vf i0, pk_vf j0, pk_vf s,
                          pk_vf *pred_out)
{
	pk_vf rsum = pk_vf_zero();
	for (int p = 0; p < NSUB; ++p) {
		pk_vf di = (pk_vf_set1(pix_i(p)) - i0) / s;
		pk_vf dj = (pk_vf_set1(pix_j(p)) - j0) / s;
		pk_vf pred = A * pk_vexpf(pk_vf_zero() - (di * di + dj * dj));
		pred_out[p] = pred;
		pk_vf r = pred - sub[p];
		rsum = rsum + r * r;
	}
	return rsum;
}

static void jac_pass_4(const pk_vf *sub, const pk_vf *pred_cur,
                       pk_vf A, pk_vf i0, pk_vf j0, pk_vf s,
                       pk_vd *JtJ, pk_vd *Jtr)
{
	const int n = 4;
	for (int t = 0; t < PIDX(n - 1, n - 1) + 1; ++t) JtJ[t] = pk_vd_zero();
	for (int t = 0; t < n; ++t) Jtr[t] = pk_vd_zero();

	for (int p = 0; p < NSUB; ++p) {
		pk_vf pred = pred_cur[p];
		pk_vf r = pred - sub[p];
		pk_vf di = (pk_vf_set1(pix_i(p)) - i0) / s;
		pk_vf dj = (pk_vf_set1(pix_j(p)) - j0) / s;
		pk_vf r2 = di * di + dj * dj;

		pk_vf J[4];
		J[0] = pred / A;                              /* dF/dA  */
		J[1] = pk_vf_set1(2.0f) * pred * di / s;      /* dF/di0 */
		J[2] = pk_vf_set1(2.0f) * pred * dj / s;      /* dF/dj0 */
		J[3] = pk_vf_set1(2.0f) * pred * r2 / s;      /* dF/ds  */

		pk_vd rd = pk_cvt_f2d(r);
		for (int p1 = 0; p1 < n; ++p1) {
			pk_vd J1 = pk_cvt_f2d(J[p1]);
			Jtr[p1] = Jtr[p1] + J1 * rd;
			for (int p2 = 0; p2 <= p1; ++p2)
				JtJ[PIDX(p1, p2)] = JtJ[PIDX(p1, p2)] + J1 * pk_cvt_f2d(J[p2]);
		}
	}
}

/* -- 5-DOF elliptical: A * exp(-((i-i0)^2/sx^2 + (j-j0)^2/sy^2)) ----------- */
static pk_vf resid_pass_5(const pk_vf *sub, pk_vf A, pk_vf i0, pk_vf j0,
                          pk_vf sx, pk_vf sy, pk_vf *pred_out)
{
	pk_vf rsum = pk_vf_zero();
	for (int p = 0; p < NSUB; ++p) {
		pk_vf di = (pk_vf_set1(pix_i(p)) - i0) / sx;
		pk_vf dj = (pk_vf_set1(pix_j(p)) - j0) / sy;
		pk_vf pred = A * pk_vexpf(pk_vf_zero() - (di * di + dj * dj));
		pred_out[p] = pred;
		pk_vf r = pred - sub[p];
		rsum = rsum + r * r;
	}
	return rsum;
}

static void jac_pass_5(const pk_vf *sub, const pk_vf *pred_cur,
                       pk_vf A, pk_vf i0, pk_vf j0, pk_vf sx, pk_vf sy,
                       pk_vd *JtJ, pk_vd *Jtr)
{
	const int n = 5;
	for (int t = 0; t < PIDX(n - 1, n - 1) + 1; ++t) JtJ[t] = pk_vd_zero();
	for (int t = 0; t < n; ++t) Jtr[t] = pk_vd_zero();

	for (int p = 0; p < NSUB; ++p) {
		pk_vf pred = pred_cur[p];
		pk_vf r = pred - sub[p];
		pk_vf di = (pk_vf_set1(pix_i(p)) - i0) / sx;
		pk_vf dj = (pk_vf_set1(pix_j(p)) - j0) / sy;

		pk_vf J[5];
		J[0] = pred / A;                               /* dF/dA  */
		J[1] = pk_vf_set1(2.0f) * pred * di / sx;      /* dF/di0 */
		J[2] = pk_vf_set1(2.0f) * pred * dj / sy;      /* dF/dj0 */
		J[3] = pk_vf_set1(2.0f) * pred * di * di / sx; /* dF/dsx */
		J[4] = pk_vf_set1(2.0f) * pred * dj * dj / sy; /* dF/dsy */

		pk_vd rd = pk_cvt_f2d(r);
		for (int p1 = 0; p1 < n; ++p1) {
			pk_vd J1 = pk_cvt_f2d(J[p1]);
			Jtr[p1] = Jtr[p1] + J1 * rd;
			for (int p2 = 0; p2 <= p1; ++p2)
				JtJ[PIDX(p1, p2)] = JtJ[PIDX(p1, p2)] + J1 * pk_cvt_f2d(J[p2]);
		}
	}
}

/* -- 6-DOF rotated (inverse covariance; sx, sy behave like variances):
 *    A * exp(-0.5 * (di^2/sx + dj^2/sy + 2*di*dj*sxy)) --------------------- */
static pk_vf resid_pass_6(const pk_vf *sub, pk_vf A, pk_vf i0, pk_vf j0,
                          pk_vf sx, pk_vf sy, pk_vf sxy, pk_vf *pred_out)
{
	pk_vf rsum = pk_vf_zero();
	pk_vf half = pk_vf_set1(0.5f);
	pk_vf two = pk_vf_set1(2.0f);
	for (int p = 0; p < NSUB; ++p) {
		pk_vf di = pk_vf_set1(pix_i(p)) - i0;
		pk_vf dj = pk_vf_set1(pix_j(p)) - j0;
		pk_vf q = di * di / sx + dj * dj / sy + two * di * dj * sxy;
		pk_vf pred = A * pk_vexpf(pk_vf_zero() - half * q);
		pred_out[p] = pred;
		pk_vf r = pred - sub[p];
		rsum = rsum + r * r;
	}
	return rsum;
}

static void jac_pass_6(const pk_vf *sub, const pk_vf *pred_cur,
                       pk_vf A, pk_vf i0, pk_vf j0, pk_vf sx, pk_vf sy, pk_vf sxy,
                       pk_vd *JtJ, pk_vd *Jtr)
{
	const int n = 6;
	for (int t = 0; t < PIDX(n - 1, n - 1) + 1; ++t) JtJ[t] = pk_vd_zero();
	for (int t = 0; t < n; ++t) Jtr[t] = pk_vd_zero();

	pk_vf half = pk_vf_set1(0.5f);
	for (int p = 0; p < NSUB; ++p) {
		pk_vf pred = pred_cur[p];
		pk_vf r = pred - sub[p];
		pk_vf di = pk_vf_set1(pix_i(p)) - i0;
		pk_vf dj = pk_vf_set1(pix_j(p)) - j0;

		pk_vf J[6];
		J[0] = pred / A;                                   /* dF/dA   */
		J[1] = pred * (di / sx + dj * sxy);                /* dF/di0  */
		J[2] = pred * (dj / sy + di * sxy);                /* dF/dj0  */
		J[3] = half * pred * di * di / (sx * sx);          /* dF/dsx  */
		J[4] = half * pred * dj * dj / (sy * sy);          /* dF/dsy  */
		J[5] = pk_vf_zero() - pred * di * dj;              /* dF/dsxy */

		pk_vd rd = pk_cvt_f2d(r);
		for (int p1 = 0; p1 < n; ++p1) {
			pk_vd J1 = pk_cvt_f2d(J[p1]);
			Jtr[p1] = Jtr[p1] + J1 * rd;
			for (int p2 = 0; p2 <= p1; ++p2)
				JtJ[PIDX(p1, p2)] = JtJ[PIDX(p1, p2)] + J1 * pk_cvt_f2d(J[p2]);
		}
	}
}

/******************************************************************************
 * Lockstep LM fitters. Each takes the SoA subwindows + per-lane seeds
 * (already clamped, phase 1) + the active mask, and returns fit params plus
 * the per-lane fit_ok mask. The loop skeleton is identical across the three
 * — kept separate to mirror the scalar file 1:1 (same accepted duplication).
 ******************************************************************************/

/* per-iteration mask bookkeeping shared by all three fitters */
#define LM_STEP_MASKS(new_resid_expr)                                          \
	pk_vf new_resid = (new_resid_expr);                                        \
	pk_vi accept = ~done & (new_resid < residual);                             \
	pk_vf improve = (residual - new_resid) / (residual + pk_vf_set1(FLT_EPSILON));

static pk_vi lm_gauss4_fit_batch(const pk_vf *sub, pk_vi active,
                                 pk_vf seedA, pk_vf seedi0, pk_vf seedj0, pk_vf seeds,
                                 pk_vf *out_i0, pk_vf *out_j0, pk_vf *out_sig0,
                                 pk_vf *out_sig1, pk_vf *out_sig2)
{
	pk_vf A = seedA, i0 = seedi0, j0 = seedj0, s = seeds;
	pk_vf pred_cur[NSUB], pred_trial[NSUB];
	pk_vd JtJ[PIDX(3, 3) + 1], Jtr[4];
	pk_vf delta[4];
	pk_vf lambda = pk_vf_set1(0.01f);
	pk_vi done = ~active;
	pk_vi fit_ok = pk_vi_zero();
	pk_vf residual;
	int iter;

	residual = resid_pass_4(sub, A, i0, j0, s, pred_trial);
	memcpy(pred_cur, pred_trial, sizeof(pred_cur));       /* initial point is "accepted" */
	jac_pass_4(sub, pred_cur, A, i0, j0, s, JtJ, Jtr);

	for (iter = 0; iter < LM_MAX_ITER; ++iter) {
		if (!pk_any(~done)) break;

		pk_vi chol_fail = solve_lm_step_vec(JtJ, Jtr, lambda, delta, 4);
		done = done | chol_fail;                           /* scalar: break -> judged by trust rule */

		pk_vf A_new  = A + delta[0];
		pk_vf i0_new = i0 + delta[1];
		pk_vf j0_new = j0 + delta[2];
		pk_vf s_new  = s + delta[3];

		A_new  = pk_maxf(A_new, A * pk_vf_set1(0.5f));
		i0_new = pk_minf(pk_maxf(i0_new, pk_vf_set1(-2.5f)), pk_vf_set1(2.5f));
		j0_new = pk_minf(pk_maxf(j0_new, pk_vf_set1(-2.5f)), pk_vf_set1(2.5f));
		s_new  = pk_minf(pk_maxf(s_new, pk_vf_set1(0.25f)), pk_vf_set1(4.0f));

		LM_STEP_MASKS(resid_pass_4(sub, A_new, i0_new, j0_new, s_new, pred_trial))

		A  = pk_self(accept, A_new, A);
		i0 = pk_self(accept, i0_new, i0);
		j0 = pk_self(accept, j0_new, j0);
		s  = pk_self(accept, s_new, s);
		residual = pk_self(accept, new_resid, residual);
		lambda = pk_self(accept, lambda * pk_vf_set1(0.5f),
		         pk_self(~done, lambda * pk_vf_set1(2.0f), lambda));

		pk_vi stagn = ~done & ~accept & (lambda > pk_vf_set1(1e6f));
		done = done | stagn;                               /* judged by trust rule below */

		for (int p = 0; p < NSUB; ++p)                     /* THE Lever-2 blend */
			pred_cur[p] = pk_self(accept, pred_trial[p], pred_cur[p]);

		if (pk_any(accept))
			jac_pass_4(sub, pred_cur, A, i0, j0, s, JtJ, Jtr);

		pk_vi conv = accept & (improve < pk_vf_set1(LM_TOL));
		fit_ok = fit_ok | conv;
		done = done | conv;
	}

	/* trust rule (verbatim scalar semantics): converged OR residual small.
	 * NaN residuals fail the compare -> correctly rejected. */
	fit_ok = fit_ok | (residual <= pk_vf_set1(LM_ACCEPT_RESID_FRAC * (float)NSUB) * A * A);
	fit_ok = fit_ok & active;

	*out_i0 = i0; *out_j0 = j0;
	*out_sig0 = s; *out_sig1 = s; *out_sig2 = pk_vf_zero();
	return fit_ok;
}

static pk_vi lm_gauss5_fit_batch(const pk_vf *sub, pk_vi active,
                                 pk_vf seedA, pk_vf seedi0, pk_vf seedj0,
                                 pk_vf seedsx, pk_vf seedsy,
                                 pk_vf *out_i0, pk_vf *out_j0, pk_vf *out_sig0,
                                 pk_vf *out_sig1, pk_vf *out_sig2)
{
	pk_vf A = seedA, i0 = seedi0, j0 = seedj0, sx = seedsx, sy = seedsy;
	pk_vf pred_cur[NSUB], pred_trial[NSUB];
	pk_vd JtJ[PIDX(4, 4) + 1], Jtr[5];
	pk_vf delta[5];
	pk_vf lambda = pk_vf_set1(0.01f);
	pk_vi done = ~active;
	pk_vi fit_ok = pk_vi_zero();
	pk_vf residual;
	int iter;

	residual = resid_pass_5(sub, A, i0, j0, sx, sy, pred_trial);
	memcpy(pred_cur, pred_trial, sizeof(pred_cur));
	jac_pass_5(sub, pred_cur, A, i0, j0, sx, sy, JtJ, Jtr);

	for (iter = 0; iter < LM_MAX_ITER; ++iter) {
		if (!pk_any(~done)) break;

		pk_vi chol_fail = solve_lm_step_vec(JtJ, Jtr, lambda, delta, 5);
		done = done | chol_fail;

		pk_vf A_new  = A + delta[0];
		pk_vf i0_new = i0 + delta[1];
		pk_vf j0_new = j0 + delta[2];
		pk_vf sx_new = sx + delta[3];
		pk_vf sy_new = sy + delta[4];

		A_new  = pk_maxf(A_new, A * pk_vf_set1(0.5f));
		i0_new = pk_minf(pk_maxf(i0_new, pk_vf_set1(-2.5f)), pk_vf_set1(2.5f));
		j0_new = pk_minf(pk_maxf(j0_new, pk_vf_set1(-2.5f)), pk_vf_set1(2.5f));
		sx_new = pk_minf(pk_maxf(sx_new, pk_vf_set1(0.25f)), pk_vf_set1(4.0f));
		sy_new = pk_minf(pk_maxf(sy_new, pk_vf_set1(0.25f)), pk_vf_set1(4.0f));

		LM_STEP_MASKS(resid_pass_5(sub, A_new, i0_new, j0_new, sx_new, sy_new, pred_trial))

		A  = pk_self(accept, A_new, A);
		i0 = pk_self(accept, i0_new, i0);
		j0 = pk_self(accept, j0_new, j0);
		sx = pk_self(accept, sx_new, sx);
		sy = pk_self(accept, sy_new, sy);
		residual = pk_self(accept, new_resid, residual);
		lambda = pk_self(accept, lambda * pk_vf_set1(0.5f),
		         pk_self(~done, lambda * pk_vf_set1(2.0f), lambda));

		pk_vi stagn = ~done & ~accept & (lambda > pk_vf_set1(1e6f));
		done = done | stagn;

		for (int p = 0; p < NSUB; ++p)
			pred_cur[p] = pk_self(accept, pred_trial[p], pred_cur[p]);

		if (pk_any(accept))
			jac_pass_5(sub, pred_cur, A, i0, j0, sx, sy, JtJ, Jtr);

		pk_vi conv = accept & (improve < pk_vf_set1(LM_TOL));
		fit_ok = fit_ok | conv;
		done = done | conv;
	}

	fit_ok = fit_ok | (residual <= pk_vf_set1(LM_ACCEPT_RESID_FRAC * (float)NSUB) * A * A);
	fit_ok = fit_ok & active;

	*out_i0 = i0; *out_j0 = j0;
	*out_sig0 = sx; *out_sig1 = sy; *out_sig2 = pk_vf_zero();
	return fit_ok;
}

static pk_vi lm_gauss6_fit_batch(const pk_vf *sub, pk_vi active,
                                 pk_vf seedA, pk_vf seedi0, pk_vf seedj0,
                                 pk_vf seedsx, pk_vf seedsy,
                                 pk_vf *out_i0, pk_vf *out_j0, pk_vf *out_sig0,
                                 pk_vf *out_sig1, pk_vf *out_sig2)
{
	pk_vf A = seedA, i0 = seedi0, j0 = seedj0, sx = seedsx, sy = seedsy;
	pk_vf sxy = pk_vf_zero();
	pk_vf pred_cur[NSUB], pred_trial[NSUB];
	pk_vd JtJ[NPACK6], Jtr[6];
	pk_vf delta[6];
	pk_vf lambda = pk_vf_set1(0.01f);
	pk_vi done = ~active;
	pk_vi fit_ok = pk_vi_zero();
	pk_vf residual;
	int iter;

	residual = resid_pass_6(sub, A, i0, j0, sx, sy, sxy, pred_trial);
	memcpy(pred_cur, pred_trial, sizeof(pred_cur));
	jac_pass_6(sub, pred_cur, A, i0, j0, sx, sy, sxy, JtJ, Jtr);

	for (iter = 0; iter < LM_MAX_ITER; ++iter) {
		if (!pk_any(~done)) break;

		pk_vi chol_fail = solve_lm_step_vec(JtJ, Jtr, lambda, delta, 6);
		done = done | chol_fail;

		pk_vf A_new   = A + delta[0];
		pk_vf i0_new  = i0 + delta[1];
		pk_vf j0_new  = j0 + delta[2];
		pk_vf sx_new  = sx + delta[3];
		pk_vf sy_new  = sy + delta[4];
		pk_vf sxy_new = sxy + delta[5];

		A_new  = pk_maxf(A_new, A * pk_vf_set1(0.5f));
		i0_new = pk_minf(pk_maxf(i0_new, pk_vf_set1(-2.5f)), pk_vf_set1(2.5f));
		j0_new = pk_minf(pk_maxf(j0_new, pk_vf_set1(-2.5f)), pk_vf_set1(2.5f));
		sx_new = pk_minf(pk_maxf(sx_new, pk_vf_set1(0.1f)), pk_vf_set1(16.0f));
		sy_new = pk_minf(pk_maxf(sy_new, pk_vf_set1(0.1f)), pk_vf_set1(16.0f));
		pk_vf sxy_max = pk_vf_set1(0.95f) / pk_sqrtf(sx_new * sy_new);
		sxy_new = pk_minf(pk_maxf(sxy_new, pk_vf_zero() - sxy_max), sxy_max);

		LM_STEP_MASKS(resid_pass_6(sub, A_new, i0_new, j0_new, sx_new, sy_new, sxy_new, pred_trial))

		A   = pk_self(accept, A_new, A);
		i0  = pk_self(accept, i0_new, i0);
		j0  = pk_self(accept, j0_new, j0);
		sx  = pk_self(accept, sx_new, sx);
		sy  = pk_self(accept, sy_new, sy);
		sxy = pk_self(accept, sxy_new, sxy);
		residual = pk_self(accept, new_resid, residual);
		lambda = pk_self(accept, lambda * pk_vf_set1(0.5f),
		         pk_self(~done, lambda * pk_vf_set1(2.0f), lambda));

		pk_vi stagn = ~done & ~accept & (lambda > pk_vf_set1(1e6f));
		done = done | stagn;

		for (int p = 0; p < NSUB; ++p)
			pred_cur[p] = pk_self(accept, pred_trial[p], pred_cur[p]);

		if (pk_any(accept))
			jac_pass_6(sub, pred_cur, A, i0, j0, sx, sy, sxy, JtJ, Jtr);

		pk_vi conv = accept & (improve < pk_vf_set1(LM_TOL));
		fit_ok = fit_ok | conv;
		done = done | conv;
	}

	fit_ok = fit_ok | (residual <= pk_vf_set1(LM_ACCEPT_RESID_FRAC * (float)NSUB) * A * A);
	fit_ok = fit_ok & active;

	*out_i0 = i0; *out_j0 = j0;
	/* 6-DOF output convention (variances; matches scalar): sig0=sx, sig1=sy, sig2=sxy */
	*out_sig0 = sx; *out_sig1 = sy; *out_sig2 = sxy;
	return fit_ok;
}

/******************************************************************************
 * Batch entry point.
 ******************************************************************************/
void lsqpeaklocate_lm_batch(const float *planes, int L_real,
                            const int *N, int iFitType,
                            float *peak_loc, float *std_dev,
                            const float *fPlaneWeight)
{
	const int numel = N[0] * N[1];
	const int Nsub[2] = { PKSIZE_X, PKSIZE_Y };
	const int W = PK_LANES;

	pk_vf sub[NSUB];
	pk_vi active = pk_vi_zero();
	pk_vf seedA = pk_vf_set1(1.0f), seedi0 = pk_vf_zero(), seedj0 = pk_vf_zero();
	pk_vf seedsx = pk_vf_set1(1.0f), seedsy = pk_vf_set1(1.0f);
	int pk_i0[PK_LANES], pk_j0[PK_LANES];
	float pk_height[PK_LANES];
	int l, i, j, p;

	if (iFitType < 4 || iFitType > 6) {
		fprintf(stderr, "lsqpeaklocate_lm_batch: unsupported iFitType %d "
		                "(caller must route 3-point fits to the scalar path)\n", iFitType);
		for (l = 0; l < W; ++l) {
			peak_loc[0 * W + l] = NAN; peak_loc[1 * W + l] = NAN; peak_loc[2 * W + l] = 0;
			std_dev[0 * W + l] = 0; std_dev[1 * W + l] = 0; std_dev[2 * W + l] = 0;
		}
		return;
	}

	/* dead-lane defaults: zero subwindows, safe finite params */
	for (p = 0; p < NSUB; ++p) sub[p] = pk_vf_zero();
	for (l = 0; l < W; ++l) { pk_i0[l] = 0; pk_j0[l] = 0; pk_height[l] = 0.0f; }

	/* ── phase 1: scalar per-lane search, gate, extract, seed (verbatim
	 *    scalar logic — NaN search masks agree with the oracle by
	 *    construction) ───────────────────────────────────────────────────── */
	for (l = 0; l < L_real && l < W; ++l) {
		const float *src = planes + (size_t)l * numel;
		int i0 = 0, j0 = 0;
		float fPeakHeight = 0;

		/* Westerweel-style search (verbatim scalar logic — see
		 * peak_locate_lm.c): largest 3x3 local max in the quarter-rule
		 * region |disp| <= N/4, first-found wins ties. */
		for (i = N[0]/4; i < N[0]*3/4; ++i) {
			for (j = N[1]/4; j < N[1]*3/4; ++j) {
				float v = src[SUB2IND_2D(i, j, N[1])];
				if (v <= fPeakHeight) continue;
				int is_max = 1;
				for (int di = -1; di <= 1 && is_max; ++di)
					for (int dj = -1; dj <= 1; ++dj)
						if (src[SUB2IND_2D(i + di, j + dj, N[1])] > v) { is_max = 0; break; }
				if (is_max) { fPeakHeight = v; i0 = i; j0 = j; }
			}
		}

		if (fPeakHeight <= 0 ||
		    i0 < (PKSIZE_X-1)/2 || i0 >= N[0]-(PKSIZE_X-1)/2 ||
		    j0 < (PKSIZE_Y-1)/2 || j0 >= N[1]-(PKSIZE_Y-1)/2) {
			continue;   /* lane stays dead -> NaN sentinel in phase 3 */
		}

		/* extract subwindow into SoA lane l, applying the patch-local
		 * loss-of-correlation compensation (verbatim scalar logic) */
		float tmp[NSUB];
		for (i = 0; i < PKSIZE_X; ++i)
			for (j = 0; j < PKSIZE_Y; ++j) {
				int idx = SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[1]);
				tmp[i * PKSIZE_Y + j] = fPlaneWeight ? src[idx] * fPlaneWeight[idx] : src[idx];
			}
		for (p = 0; p < NSUB; ++p) sub[p][l] = tmp[p];

		/* 3-point seed + the fit type's seed clamps (verbatim scalar) */
		float peak[2], A, sx, sy;
		threept_estimate_b(tmp, Nsub, peak, &A, &sx, &sy);
		float si0 = fminf(fmaxf(peak[0], -2.0f), 2.0f);
		float sj0 = fminf(fmaxf(peak[1], -2.0f), 2.0f);
		if (iFitType == 4) {
			float s = sqrtf(sx * sx + sy * sy);
			s = fminf(fmaxf(s, 0.5f), 3.0f);
			seedsx[l] = s;            /* gauss4 carries s in the sx slot */
			seedsy[l] = s;
		} else if (iFitType == 5) {
			seedsx[l] = fminf(fmaxf(sx, 0.5f), 3.0f);
			seedsy[l] = fminf(fmaxf(sy, 0.5f), 3.0f);
		} else { /* 6: inverse-covariance variances */
			seedsx[l] = fminf(fmaxf(sx * sx, 0.25f), 9.0f);
			seedsy[l] = fminf(fmaxf(sy * sy, 0.25f), 9.0f);
		}
		seedA[l] = A;
		seedi0[l] = si0;
		seedj0[l] = sj0;
		active[l] = -1;
		pk_i0[l] = i0;
		pk_j0[l] = j0;
		pk_height[l] = fPeakHeight;
	}

	/* ── phase 2: lockstep LM ────────────────────────────────────────────── */
	pk_vf fi0, fj0, sig0, sig1, sig2;
	pk_vi fit_ok = pk_vi_zero();

	if (pk_any(active)) {
		switch (iFitType) {
		case 6:
			fit_ok = lm_gauss6_fit_batch(sub, active, seedA, seedi0, seedj0,
			                             seedsx, seedsy, &fi0, &fj0, &sig0, &sig1, &sig2);
			break;
		case 5:
			fit_ok = lm_gauss5_fit_batch(sub, active, seedA, seedi0, seedj0,
			                             seedsx, seedsy, &fi0, &fj0, &sig0, &sig1, &sig2);
			break;
		default: /* 4 */
			fit_ok = lm_gauss4_fit_batch(sub, active, seedA, seedi0, seedj0,
			                             seedsx, &fi0, &fj0, &sig0, &sig1, &sig2);
			break;
		}
	} else {
		fi0 = pk_vf_zero(); fj0 = pk_vf_zero();
		sig0 = pk_vf_zero(); sig1 = pk_vf_zero(); sig2 = pk_vf_zero();
	}

	/* ── phase 3: scalar writeback — trusted fit or NaN sentinel (exact
	 *    scalar convention: NaN row/col, height 0, std 0) ─────────────────── */
	for (l = 0; l < W; ++l) {
		if (fit_ok[l]) {
			peak_loc[0 * W + l] = fi0[l] + (float)pk_i0[l];
			peak_loc[1 * W + l] = fj0[l] + (float)pk_j0[l];
			peak_loc[2 * W + l] = pk_height[l];
			std_dev[0 * W + l] = sig0[l];
			std_dev[1 * W + l] = sig1[l];
			std_dev[2 * W + l] = sig2[l];
		} else {
			peak_loc[0 * W + l] = NAN;
			peak_loc[1 * W + l] = NAN;
			peak_loc[2 * W + l] = 0;
			std_dev[0 * W + l] = 0;
			std_dev[1 * W + l] = 0;
			std_dev[2 * W + l] = 0;
		}
	}
}

#endif /* PK_BATCH_AVAILABLE */
