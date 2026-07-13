#include "peak_locate_lm.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/******************************************************************************
 * Lightweight Levenberg-Marquardt implementation for Gaussian peak fitting
 *
 * Supports 3-point, 4-DOF, 5-DOF, and 6-DOF Gaussian fits for PIV analysis
 *
 * FAILURE SEMANTICS (2026-07-06): a FAILED LM fit (Cholesky breakdown,
 * max_iter exhausted without converging, or stagnation with a large residual)
 * writes the same NaN sentinel as a failed peak search: peak_loc row/col =
 * NaN, height = 0, std_dev = 0. Previously all LM exits wrote finite
 * last-best params, indistinguishable from convergence. Normal-equation
 * accumulation, the Cholesky solve, AND the residual sum / improvement
 * statistic (2026-07-13) are double internally (params/data stay float —
 * the residual must be double because tol=1e-6 on a difference of two
 * nearly-equal float sums is below float32's resolution).
 *
 * KNOWN TECHNICAL DEBT:
 * - Code duplication: LM iteration logic is repeated in lm_gauss4_fit,
 *   lm_gauss5_fit, and lm_gauss6_fit. This should be refactored into a
 *   common helper function that accepts function pointers for model
 *   evaluation and Jacobian computation.
 *
 * - Non-standard 6-DOF parameterization: The 6-DOF model uses inverse
 *   covariance matrix elements instead of standard deviations and rotation,
 *   making it confusing and error-prone. See lm_gauss6_fit() for details.
 ******************************************************************************/

/* Fast 3-point parabolic estimator - used as initial guess and fallback */
static void threept_estimate(const float *xcorr, const int *N, float *peak_loc, float *A_out, float *sx_out, float *sy_out)
{
	float x_fit[3], y_fit[3];
	int i;

	/* Extract 3 points along each axis */
	for(i = 0; i < 3; ++i)
	{
		x_fit[i] = xcorr[(i - 1 + (N[0]-1)/2) * N[1] + (N[1]-1)/2];
		y_fit[i] = xcorr[(N[0]-1)/2 * N[1] + (i - 1 + (N[1]-1)/2)];
		x_fit[i] = (float)log((x_fit[i] < FLT_EPSILON) ? FLT_EPSILON : x_fit[i]);
		y_fit[i] = (float)log((y_fit[i] < FLT_EPSILON) ? FLT_EPSILON : y_fit[i]);
	}

	/* Parabolic fit: peak location is at i0 = numer/denom */
	float denom_x = 2*x_fit[0] - 4*x_fit[1] + 2*x_fit[2];
	float denom_y = 2*y_fit[0] - 4*y_fit[1] + 2*y_fit[2];

	if(fabs(denom_x) > FLT_EPSILON)
		peak_loc[0] = (x_fit[0] - x_fit[2]) / denom_x;
	else
		peak_loc[0] = 0.0f;

	if(fabs(denom_y) > FLT_EPSILON)
		peak_loc[1] = (y_fit[0] - y_fit[2]) / denom_y;
	else
		peak_loc[1] = 0.0f;

	*A_out = xcorr[(N[0]-1)/2 * N[1] + (N[1]-1)/2];
	*sx_out = (float)sqrt(-4.0f / (denom_x + FLT_EPSILON));
	*sy_out = (float)sqrt(-4.0f / (denom_y + FLT_EPSILON));
}

/* Evaluate 4-DOF Gaussian: A * exp(-((i-i0)^2 + (j-j0)^2)/s^2) - circular Gaussian */
static inline float eval_gauss4(float i, float j, float A, float i0, float j0, float s)
{
	float di = (i - i0) / s;
	float dj = (j - j0) / s;
	return A * expf(-(di*di + dj*dj));
}

/* Evaluate 5-DOF Gaussian: A * exp(-((i-i0)^2/sx^2 + (j-j0)^2/sy^2)) - elliptical */
static inline float eval_gauss5(float i, float j, float A, float i0, float j0, float sx, float sy)
{
	float di = (i - i0) / sx;
	float dj = (j - j0) / sy;
	return A * expf(-(di*di + dj*dj));
}

/* Evaluate 6-DOF Gaussian with correlation term - rotated elliptical
 * NOTE: sx, sy, sxy are elements of the INVERSE covariance matrix, not standard deviations
 * Model: A * exp(-0.5 * (di^2/sx + dj^2/sy + 2*di*dj*sxy))
 * where sx = sigma_x^2, sy = sigma_y^2 in the inverse covariance representation
 */
static inline float eval_gauss6(float i, float j, float A, float i0, float j0, float sx, float sy, float sxy)
{
	float di = i - i0;
	float dj = j - j0;
	return A * expf(-0.5f * (di*di/sx + dj*dj/sy + 2.0f*di*dj*sxy));
}

/* Compute residual and Jacobian for 4-DOF Gaussian fit.
 *
 * pred_buf (length N[0]*N[1]) carries the model values between the two passes
 * (Lever 2): the residual-only pass WRITES every pred; the Jacobian pass READS
 * them instead of re-calling exp. In the LM loop a trial residual is always
 * evaluated immediately before an accepted Jacobian rebuild at the SAME point,
 * so the cache is exact (no new approximation) and the Jacobian path makes zero
 * exp calls — removing the redundant half of the exp workload (Lever 2). This is
 * the actual win, and it is compiler-independent (it deletes work, not just
 * reschedules it): ~15-25% per fit on MSVC, clang, and ARM alike.
 * CONTRACT: pred_buf must be non-NULL on EVERY call — the residual pass writes it
 * unconditionally (the store is kept branch-free). */
static double compute_residual_jacobian_4dof(
	const float *xcorr, const int *N,
	float A, float i0, float j0, float s,
	double *JtJ, double *Jtr, int compute_jacobian, float *pred_buf)
{
	int ii, jj, idx;
	double residual_sum = 0.0;
	const int n_params = 4;

	/* Hot path (every LM trial, including rejects): residual only. The branch
	   that used to gate the Jacobian is hoisted OUT of the inner loop so the body
	   is branch-free, and pred_buf is written unconditionally so the accepted-step
	   Jacobian pass can read the model values back (Lever 2). exp is libm expf: an
	   inline polynomial approximation was A/B-benchmarked (MSVC/clang/ARM) and was
	   slower AND did not auto-vectorize (the 5x5 reduction loop won't vectorize on
	   MSVC or clang), so it was dropped — libm is faster, simpler, and exact. */
	if(!compute_jacobian) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			int row = ii * N[1];
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				float pred = eval_gauss4(i, j, A, i0, j0, s);
				pred_buf[row + jj] = pred;
				float r = pred - xcorr[row + jj];
				residual_sum += (double)r * (double)r;
			}
		}
		return residual_sum;
	}

	/* Jacobian path: runs once per ACCEPTED step (rare); kept scalar. Reads the
	   cached pred (filled by the immediately-preceding residual pass) -> no exp. */
	memset(JtJ, 0, n_params * n_params * sizeof(double));
	memset(Jtr, 0, n_params * sizeof(double));

	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);

		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;

			float pred = pred_buf[idx];
			float r = pred - xcorr[idx];
			residual_sum += (double)r * (double)r;

			float di = (i - i0) / s;
			float dj = (j - j0) / s;
			float r2 = di*di + dj*dj;

			float J[4];
			J[0] = pred / A;                           /* dF/dA */
			J[1] = 2.0f * pred * di / s;              /* dF/di0 */
			J[2] = 2.0f * pred * dj / s;              /* dF/dj0 */
			J[3] = 2.0f * pred * r2 / s;              /* dF/ds */

			/* Accumulate in double: 25 float products per entry lose precision
			   in float sums, and the 6-DOF JtJ is ill-conditioned for
			   near-circular peaks. J itself stays float (model-limited). */
			for(int p1 = 0; p1 < n_params; ++p1) {
				Jtr[p1] += (double)J[p1] * (double)r;
				for(int p2 = 0; p2 <= p1; ++p2) {
					JtJ[p1 * n_params + p2] += (double)J[p1] * (double)J[p2];
				}
			}
		}
	}

	for(int p1 = 0; p1 < n_params; ++p1) {
		for(int p2 = p1 + 1; p2 < n_params; ++p2) {
			JtJ[p1 * n_params + p2] = JtJ[p2 * n_params + p1];
		}
	}

	return residual_sum;
}

/* Compute residual and Jacobian for 5-DOF Gaussian fit.
 * pred_buf carries model values between passes (Lever 2; see the 4-DOF note). */
static double compute_residual_jacobian_5dof(
	const float *xcorr, const int *N,
	float A, float i0, float j0, float sx, float sy,
	double *JtJ, double *Jtr, int compute_jacobian, float *pred_buf)
{
	int ii, jj, idx;
	double residual_sum = 0.0;
	const int n_params = 5;

	/* Hot path: residual only, branch-free; fills the Lever 2 pred cache
	   (see compute_residual_jacobian_4dof for the rationale). */
	if(!compute_jacobian) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			int row = ii * N[1];
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				float pred = eval_gauss5(i, j, A, i0, j0, sx, sy);
				pred_buf[row + jj] = pred;
				float r = pred - xcorr[row + jj];
				residual_sum += (double)r * (double)r;
			}
		}
		return residual_sum;
	}

	/* Jacobian path: once per accepted step (rare); kept scalar. Reads cached pred. */
	memset(JtJ, 0, n_params * n_params * sizeof(double));
	memset(Jtr, 0, n_params * sizeof(double));

	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);

		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;

			float pred = pred_buf[idx];
			float r = pred - xcorr[idx];
			residual_sum += (double)r * (double)r;

			float di = (i - i0) / sx;
			float dj = (j - j0) / sy;

			float J[5];
			J[0] = pred / A;                    /* dF/dA */
			J[1] = 2.0f * pred * di / sx;      /* dF/di0 */
			J[2] = 2.0f * pred * dj / sy;      /* dF/dj0 */
			J[3] = 2.0f * pred * di * di / sx; /* dF/dsx */
			J[4] = 2.0f * pred * dj * dj / sy; /* dF/dsy */

			/* Accumulate in double: 25 float products per entry lose precision
			   in float sums, and the 6-DOF JtJ is ill-conditioned for
			   near-circular peaks. J itself stays float (model-limited). */
			for(int p1 = 0; p1 < n_params; ++p1) {
				Jtr[p1] += (double)J[p1] * (double)r;
				for(int p2 = 0; p2 <= p1; ++p2) {
					JtJ[p1 * n_params + p2] += (double)J[p1] * (double)J[p2];
				}
			}
		}
	}

	for(int p1 = 0; p1 < n_params; ++p1) {
		for(int p2 = p1 + 1; p2 < n_params; ++p2) {
			JtJ[p1 * n_params + p2] = JtJ[p2 * n_params + p1];
		}
	}

	return residual_sum;
}

/* Compute residual and Jacobian for 6-DOF Gaussian fit.
 * pred_buf carries model values between passes (Lever 2; see the 4-DOF note). */
static double compute_residual_jacobian_6dof(
	const float *xcorr, const int *N,
	float A, float i0, float j0, float sx, float sy, float sxy,
	double *JtJ, double *Jtr, int compute_jacobian, float *pred_buf)
{
	int ii, jj, idx;
	double residual_sum = 0.0;
	const int n_params = 6;

	/* Hot path: residual only, branch-free; fills the Lever 2 pred cache
	   (see compute_residual_jacobian_4dof for the rationale). */
	if(!compute_jacobian) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			int row = ii * N[1];
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				float pred = eval_gauss6(i, j, A, i0, j0, sx, sy, sxy);
				pred_buf[row + jj] = pred;
				float r = pred - xcorr[row + jj];
				residual_sum += (double)r * (double)r;
			}
		}
		return residual_sum;
	}

	/* Jacobian path: once per accepted step (rare); kept scalar. Reads cached pred. */
	memset(JtJ, 0, n_params * n_params * sizeof(double));
	memset(Jtr, 0, n_params * sizeof(double));

	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);

		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;

			float pred = pred_buf[idx];
			float r = pred - xcorr[idx];
			residual_sum += (double)r * (double)r;

			float di = i - i0;
			float dj = j - j0;

			float J[6];
			J[0] = pred / A;                                    /* dF/dA */
			J[1] = pred * (di/sx + dj*sxy);                    /* dF/di0 */
			J[2] = pred * (dj/sy + di*sxy);                    /* dF/dj0 */
			J[3] = 0.5f * pred * di * di / (sx * sx);          /* dF/dsx - FIXED: removed incorrect negative sign */
			J[4] = 0.5f * pred * dj * dj / (sy * sy);          /* dF/dsy - FIXED: removed incorrect negative sign */
			J[5] = -pred * di * dj;                            /* dF/dsxy */

			/* Accumulate in double: 25 float products per entry lose precision
			   in float sums, and the 6-DOF JtJ is ill-conditioned for
			   near-circular peaks. J itself stays float (model-limited). */
			for(int p1 = 0; p1 < n_params; ++p1) {
				Jtr[p1] += (double)J[p1] * (double)r;
				for(int p2 = 0; p2 <= p1; ++p2) {
					JtJ[p1 * n_params + p2] += (double)J[p1] * (double)J[p2];
				}
			}
		}
	}

	for(int p1 = 0; p1 < n_params; ++p1) {
		for(int p2 = p1 + 1; p2 < n_params; ++p2) {
			JtJ[p1 * n_params + p2] = JtJ[p2 * n_params + p1];
		}
	}

	return residual_sum;
}

/* Solve (JtJ + lambda*diag(JtJ)) * delta = -Jtr using Cholesky decomposition.
 * All internal arithmetic is double: the normal equations condense 25 float
 * products per entry and the 6-DOF system is near-singular for near-circular
 * peaks — float Cholesky tips over into spurious "not positive-definite"
 * failures that double solves fine. Inputs accumulate in double upstream;
 * only the returned step is narrowed back to float. Cost is negligible
 * (<=6x6, once per LM iteration). */
static int solve_lm_step(const double *JtJ, const double *Jtr, float lambda, float *delta, int n)
{
	double A[36]; /* Max 6x6 matrix */
	double L[36];
	double y[6];
	double d[6];
	int i, j, k;

	memcpy(A, JtJ, n * n * sizeof(double));
	for(i = 0; i < n; ++i) {
		A[i * n + i] *= (1.0 + (double)lambda);
	}

	/* Cholesky decomposition: A = L * L^T */
	memset(L, 0, n * n * sizeof(double));
	for(i = 0; i < n; ++i) {
		for(j = 0; j <= i; ++j) {
			double sum = A[i * n + j];
			for(k = 0; k < j; ++k) {
				sum -= L[i * n + k] * L[j * n + k];
			}
			if(i == j) {
				if(sum <= 0.0) return -1;
				L[i * n + j] = sqrt(sum);
			} else {
				L[i * n + j] = sum / L[j * n + j];
			}
		}
	}

	/* Forward substitution: L * y = -Jtr */
	for(i = 0; i < n; ++i) {
		double sum = -Jtr[i];
		for(j = 0; j < i; ++j) {
			sum -= L[i * n + j] * y[j];
		}
		y[i] = sum / L[i * n + i];
	}

	/* Back substitution: L^T * d = y */
	for(i = n - 1; i >= 0; --i) {
		double sum = y[i];
		for(j = i + 1; j < n; ++j) {
			sum -= L[j * n + i] * d[j];
		}
		d[i] = sum / L[i * n + i];
	}

	for(i = 0; i < n; ++i) delta[i] = (float)d[i];

	return 0;
}

/* Residual threshold for trusting a fit that did NOT converge by tolerance.
 * The exit path alone cannot classify failure: on perfect data the residual
 * decays geometrically so the relative-improvement test never fires (loop
 * runs to max_iter with a superb fit), and lambda blow-up fires when the
 * 3-point seed is already exactly optimal. So the rule is: trust the fit iff
 * it converged by tolerance OR the summed squared residual is small relative
 * to the peak energy (A^2 * npix) — roughly RMS residual <= ~3% of peak
 * amplitude. A good fit sits orders of magnitude below this threshold, a
 * plateau/noise/garbage fit orders above. NaN residuals (NaN-poisoned
 * window) fail the comparison and are correctly rejected. */
#define LM_ACCEPT_RESID_FRAC 1e-3f

/* Fast Levenberg-Marquardt for 4-DOF Gaussian fitting (circular).
 * Returns 0 if the fit is trustworthy, -1 if it FAILED (Cholesky breakdown,
 * max_iter exhausted without converging, or stagnated with a large residual).
 * peak_loc/sig/fitval are written either way (fitval feeds the multi-peak
 * subtraction); the caller decides what a failure means for its output. */
static int lm_gauss4_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, s;
	double JtJ[16], Jtr[4];
	float delta[4];
	int fit_ok = 0;   /* 1 = trustworthy; stays 0 on max_iter fall-through */
	float pred_cache[PKSIZE_X * PKSIZE_Y];  /* Lever 2: model values shared trial->Jacobian */
	float lambda = 0.01f;
	/* residual/improvement in DOUBLE (2026-07-13): summed float squares and
	 * the difference of two nearly-equal sums cannot resolve tol=1e-6 in
	 * float32 (~8x FLT_EPSILON) — convergence became a coin flip on last-ulp
	 * dust and NaN'd ~4% of good windows (see wiki
	 * sessions/2026-07-13-code1-lm-convergence-race.md). */
	double residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 50;   /* was 20: converged fits exit early, only
	                            * marginal windows use the headroom */
	const double tol = 1e-6;

	/* Get initial guess */
	float sx, sy;
	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];
	s = sqrtf(sx * sx + sy * sy); /* Combined width */

	/* Clamp bounds */
	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	s = fminf(fmaxf(s, 0.5f), 3.0f);

	/* Fill the pred cache (residual pass), then build the Jacobian from it. */
	residual = compute_residual_jacobian_4dof(xcorr, N, A, i0, j0, s, NULL, NULL, 0, pred_cache);
	compute_residual_jacobian_4dof(xcorr, N, A, i0, j0, s, JtJ, Jtr, 1, pred_cache);

	for(iter = 0; iter < max_iter; ++iter) {
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 4) != 0) break;  /* FAILED: degenerate normal equations */

		float A_new = A + delta[0];
		float i0_new = i0 + delta[1];
		float j0_new = j0 + delta[2];
		float s_new = s + delta[3];

		A_new = fmaxf(A_new, A * 0.5f);
		i0_new = fminf(fmaxf(i0_new, -2.5f), 2.5f);
		j0_new = fminf(fmaxf(j0_new, -2.5f), 2.5f);
		s_new = fminf(fmaxf(s_new, 0.25f), 4.0f);

		new_residual = compute_residual_jacobian_4dof(xcorr, N, A_new, i0_new, j0_new, s_new, NULL, NULL, 0, pred_cache);

		if(new_residual < residual) {
			A = A_new; i0 = i0_new; j0 = j0_new; s = s_new;
			double improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f;
			compute_residual_jacobian_4dof(xcorr, N, A, i0, j0, s, JtJ, Jtr, 1, pred_cache);
			if(improvement < tol) { fit_ok = 1; break; }  /* converged */
		} else {
			lambda *= 2.0f;
			if(lambda > 1e6f) break;  /* stagnated: judged by residual below */
		}
	}
	/* Trust the fit iff it converged by tolerance OR the residual is small
	   (see LM_ACCEPT_RESID_FRAC — exit path alone cannot classify failure). */
	if(!fit_ok)
		fit_ok = (residual <= LM_ACCEPT_RESID_FRAC * A * A * (float)(N[0] * N[1]));
	/* Degenerate-solution guard (the ONLY post-exit guard, 2026-07-13): a
	   width sitting ON a step-clamp bound means the optimizer converged into
	   the constraint wall (delta spike / ever-widening blob), not onto a
	   peak. Anything else that converges becomes a vector — rejection of
	   dubious-but-well-formed fits is owned by the downstream validation
	   stack (peak_mag threshold, normalized median), Westerweel-style. */
	if(fit_ok)
		fit_ok = (s > 0.25f && s < 4.0f);

	peak_loc[0] = i0;
	peak_loc[1] = j0;
	sig[0] = s;
	sig[1] = s;
	sig[2] = 0.0f;

	if(fitval) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				idx = ii * N[1] + jj;
				fitval[idx] = eval_gauss4(i, j, A, i0, j0, s);
			}
		}
	}

	return fit_ok ? 0 : -1;
}

/* Fast Levenberg-Marquardt for 5-DOF Gaussian fitting.
 * Returns 0 = trustworthy, -1 = FAILED (see lm_gauss4_fit). */
static int lm_gauss5_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, sx, sy;
	double JtJ[25], Jtr[5];
	float delta[5];
	int fit_ok = 0;   /* 1 = trustworthy; stays 0 on max_iter fall-through */
	float pred_cache[PKSIZE_X * PKSIZE_Y];  /* Lever 2: model values shared trial->Jacobian */
	float lambda = 0.01f;
	/* residual/improvement in DOUBLE (2026-07-13): summed float squares and
	 * the difference of two nearly-equal sums cannot resolve tol=1e-6 in
	 * float32 (~8x FLT_EPSILON) — convergence became a coin flip on last-ulp
	 * dust and NaN'd ~4% of good windows (see wiki
	 * sessions/2026-07-13-code1-lm-convergence-race.md). */
	double residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 50;   /* was 20: converged fits exit early, only
	                            * marginal windows use the headroom */
	const double tol = 1e-6;

	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];

	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	sx = fminf(fmaxf(sx, 0.5f), 3.0f);
	sy = fminf(fmaxf(sy, 0.5f), 3.0f);

	/* Fill the pred cache (residual pass), then build the Jacobian from it. */
	residual = compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, NULL, NULL, 0, pred_cache);
	compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, JtJ, Jtr, 1, pred_cache);

	for(iter = 0; iter < max_iter; ++iter) {
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 5) != 0) break;  /* FAILED: degenerate normal equations */

		float A_new = A + delta[0];
		float i0_new = i0 + delta[1];
		float j0_new = j0 + delta[2];
		float sx_new = sx + delta[3];
		float sy_new = sy + delta[4];

		A_new = fmaxf(A_new, A * 0.5f);
		i0_new = fminf(fmaxf(i0_new, -2.5f), 2.5f);
		j0_new = fminf(fmaxf(j0_new, -2.5f), 2.5f);
		sx_new = fminf(fmaxf(sx_new, 0.25f), 4.0f);
		sy_new = fminf(fmaxf(sy_new, 0.25f), 4.0f);

		new_residual = compute_residual_jacobian_5dof(xcorr, N, A_new, i0_new, j0_new, sx_new, sy_new, NULL, NULL, 0, pred_cache);

		if(new_residual < residual) {
			A = A_new; i0 = i0_new; j0 = j0_new; sx = sx_new; sy = sy_new;
			double improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f;
			compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, JtJ, Jtr, 1, pred_cache);
			if(improvement < tol) { fit_ok = 1; break; }  /* converged */
		} else {
			lambda *= 2.0f;
			if(lambda > 1e6f) break;  /* stagnated: judged by residual below */
		}
	}
	/* Trust the fit iff it converged by tolerance OR the residual is small. */
	if(!fit_ok)
		fit_ok = (residual <= LM_ACCEPT_RESID_FRAC * A * A * (float)(N[0] * N[1]));
	/* Degenerate-solution guard: clamp-pinned widths are not a peak (see the
	   gauss4 comment — the only post-exit guard). */
	if(fit_ok)
		fit_ok = (sx > 0.25f && sx < 4.0f && sy > 0.25f && sy < 4.0f);

	peak_loc[0] = i0;
	peak_loc[1] = j0;
	sig[0] = sx;
	sig[1] = sy;
	sig[2] = 0.0f;

	if(fitval) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				idx = ii * N[1] + jj;
				fitval[idx] = eval_gauss5(i, j, A, i0, j0, sx, sy);
			}
		}
	}

	return fit_ok ? 0 : -1;
}

/* Fast Levenberg-Marquardt for 6-DOF Gaussian fitting
 *
 * WARNING: This function uses a non-standard parameterization!
 * - Parameters sx, sy, sxy represent elements of the INVERSE covariance matrix
 * - sx and sy behave like variances (sigma^2), NOT standard deviations
 * - Output parameters sig[0] and sig[1] are SWAPPED (sig[0]=sy, sig[1]=sx)
 *
 * KNOWN ISSUES:
 * - Confusing parameterization makes the code hard to understand and verify
 * - Output parameter swapping is error-prone and undocumented
 *
 * RECOMMENDATION: Refactor to use standard Gaussian parameterization with
 * amplitude, center (i0, j0), standard deviations (sigma_x, sigma_y), and
 * rotation angle theta. This would make derivatives easier to verify and
 * output easier to interpret.
 */
static int lm_gauss6_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, sx, sy, sxy;
	double JtJ[36], Jtr[6];
	float delta[6];
	int fit_ok = 0;   /* 1 = trustworthy; stays 0 on max_iter fall-through */
	float pred_cache[PKSIZE_X * PKSIZE_Y];  /* Lever 2: model values shared trial->Jacobian */
	float lambda = 0.01f;
	/* residual/improvement in DOUBLE (2026-07-13): summed float squares and
	 * the difference of two nearly-equal sums cannot resolve tol=1e-6 in
	 * float32 (~8x FLT_EPSILON) — convergence became a coin flip on last-ulp
	 * dust and NaN'd ~4% of good windows (see wiki
	 * sessions/2026-07-13-code1-lm-convergence-race.md). */
	double residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 50;   /* was 20: converged fits exit early, only
	                            * marginal windows use the headroom */
	const double tol = 1e-6;

	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];
	sxy = 0.0f;

	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	sx = fminf(fmaxf(sx * sx, 0.25f), 9.0f);
	sy = fminf(fmaxf(sy * sy, 0.25f), 9.0f);

	/* Fill the pred cache (residual pass), then build the Jacobian from it. */
	residual = compute_residual_jacobian_6dof(xcorr, N, A, i0, j0, sx, sy, sxy, NULL, NULL, 0, pred_cache);
	compute_residual_jacobian_6dof(xcorr, N, A, i0, j0, sx, sy, sxy, JtJ, Jtr, 1, pred_cache);

	for(iter = 0; iter < max_iter; ++iter) {
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 6) != 0) break;  /* FAILED: degenerate normal equations */

		float A_new = A + delta[0];
		float i0_new = i0 + delta[1];
		float j0_new = j0 + delta[2];
		float sx_new = sx + delta[3];
		float sy_new = sy + delta[4];
		float sxy_new = sxy + delta[5];

		A_new = fmaxf(A_new, A * 0.5f);
		i0_new = fminf(fmaxf(i0_new, -2.5f), 2.5f);
		j0_new = fminf(fmaxf(j0_new, -2.5f), 2.5f);
		sx_new = fminf(fmaxf(sx_new, 0.1f), 16.0f);
		sy_new = fminf(fmaxf(sy_new, 0.1f), 16.0f);
		float sxy_max = 0.95f / sqrtf(sx_new * sy_new);
		sxy_new = fminf(fmaxf(sxy_new, -sxy_max), sxy_max);

		new_residual = compute_residual_jacobian_6dof(xcorr, N, A_new, i0_new, j0_new, sx_new, sy_new, sxy_new, NULL, NULL, 0, pred_cache);

		if(new_residual < residual) {
			A = A_new; i0 = i0_new; j0 = j0_new;
			sx = sx_new; sy = sy_new; sxy = sxy_new;
			double improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f;
			compute_residual_jacobian_6dof(xcorr, N, A, i0, j0, sx, sy, sxy, JtJ, Jtr, 1, pred_cache);
			if(improvement < tol) { fit_ok = 1; break; }  /* converged */
		} else {
			lambda *= 2.0f;
			if(lambda > 1e6f) break;  /* stagnated: judged by residual below */
		}
	}
	/* Trust the fit iff it converged by tolerance OR the residual is small. */
	if(!fit_ok)
		fit_ok = (residual <= LM_ACCEPT_RESID_FRAC * A * A * (float)(N[0] * N[1]));
	/* Degenerate-solution guard: clamp-pinned (inverse-covariance) widths are
	   not a peak (see the gauss4 comment — the only post-exit guard). Step
	   clamps here are [0.1, 16] (variances). */
	if(fit_ok)
		fit_ok = (sx > 0.1f && sx < 16.0f && sy > 0.1f && sy < 16.0f);

	peak_loc[0] = i0;
	peak_loc[1] = j0;
	/* Output convention (consistent with 4-DOF and 5-DOF):
	 * sig[0] = variance in row (i) direction
	 * sig[1] = variance in col (j) direction
	 * sig[2] = covariance term (sxy)
	 * Note: 6-DOF uses inverse covariance parameterization (variances, not sigmas) */
	sig[0] = sx;  /* Row direction variance */
	sig[1] = sy;  /* Col direction variance */
	sig[2] = sxy; /* Covariance term */

	if(fitval) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				idx = ii * N[1] + jj;
				fitval[idx] = eval_gauss6(i, j, A, i0, j0, sx, sy, sxy);
			}
		}
	}

	return fit_ok ? 0 : -1;
}

/******************************************************************************
 * Peak detectability (PIVware pwInterrogateDisplacement.m SNR type 2):
 * tallest / second-tallest 3x3 local maximum, over the SAME candidate set as
 * the detection search below (quarter region N/4..3N/4, positive, non-strict
 * local max). Peak 2 excludes only peak 1's own pixel — the local-maxima map
 * cannot contain peak 1's shoulders by construction, exactly like PIVware's
 * `Peaks(jpeak,ipeak) = 0; Pmax2 = max(Peaks)`. Conventions (PIVware-
 * faithful): no candidate -> 0, no second positive candidate -> 0
 * (pwInterrogateDisplacement.m:46). Two equal tallest peaks -> ratio 1.
 * Diagnostic only: stored per window alongside pk_height, NEVER used as a
 * gate in C — thresholding (if any) belongs to the downstream validation
 * stack.
 *****************************************************************************/
PEAK_EXPORT float peak_detectability(const float *xcorr, const int *N)
{
	float p1 = 0.0f, p2 = 0.0f;
	for(int i = N[0]/4; i < N[0]*3/4; ++i) {
		for(int j = N[1]/4; j < N[1]*3/4; ++j) {
			float v = xcorr[SUB2IND_2D(i, j, N[1])];
			if(v <= p2 || v <= 0.0f) continue;   /* cannot enter the top two */
			int is_max = 1;
			for(int di = -1; di <= 1 && is_max; ++di)
				for(int dj = -1; dj <= 1; ++dj)
					if(xcorr[SUB2IND_2D(i + di, j + dj, N[1])] > v) { is_max = 0; break; }
			if(!is_max) continue;
			if(v > p1) { p2 = p1; p1 = v; }
			else       { p2 = v; }               /* v > p2 guaranteed above */
		}
	}
	return (p2 > 0.0f) ? p1 / p2 : 0.0f;
}

/******************************************************************************
 * Main peak localization function
 *
 * fPlaneWeight (nullable): per-pixel loss-of-correlation compensation
 * (nPx/(W conv W)) applied ONLY to the extracted fit patch, at the patch's own
 * plane coordinates — Westerweel/PIVware semantics. Detection (the peak
 * search) runs on the RAW plane: the envelope attenuates signal and noise at
 * a given displacement equally, so pre-multiplying the plane cannot aid
 * detection — it only lets boosted far-field noise win the argmax. NULL
 * disables compensation entirely (tests, callers with unweighted planes).
 *****************************************************************************/
PEAK_EXPORT void lsqpeaklocate_lm(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev, const float *fPlaneWeight)
{
	int i, j, iPeak, idx;
	int i0, j0;
	float *xcorr_copy = NULL;
	const float *src;
	float fPeakHeight;
	float subxcorr[PKSIZE_X * PKSIZE_Y];
	float fitval[PKSIZE_X * PKSIZE_Y];
	int Nsub[2];
	float peak[2];
	float sig[3];

	/* Lever 1: the mutable full-plane copy exists only so multi-peak can
	 * subtract a fitted peak and re-search. For the common nPeaks==1 case it is
	 * pure waste — read the const input directly and skip the malloc/memcpy and
	 * the subtract loop. Bit-identical: same data read, same fit functions. */
	if(nPeaks > 1) {
		xcorr_copy = (float*)malloc(sizeof(float) * N[0] * N[1]);
		memcpy(xcorr_copy, xcorr, N[0] * N[1] * sizeof(float));
		src = xcorr_copy;
	} else {
		src = xcorr;
	}
	Nsub[0] = PKSIZE_X;
	Nsub[1] = PKSIZE_Y;

	for(iPeak = 0; iPeak < nPeaks; ++iPeak)
	{
		/* Westerweel-style search (PIVware pwDetectPeaks/pwGetPeaks): a
		 * candidate is a pixel equal to the max of its 3x3 neighbourhood
		 * (non-strict); the LARGEST candidate wins, first-found on ties.
		 * Plateaus/ridges cannot hijack the selection or kill the window —
		 * the brightest well-formed peak is fitted instead. The region is
		 * the quarter rule |disp| <= N/4: anything outside it would fail
		 * the caller's displacement gate anyway. */
		i0 = j0 = 0;
		fPeakHeight = 0;
		for(i = N[0]/4; i < N[0]*3/4; ++i) {
			for(j = N[1]/4; j < N[1]*3/4; ++j) {
				float v = src[SUB2IND_2D(i, j, N[1])];
				if(v <= fPeakHeight) continue;
				int is_max = 1;
				for(int di = -1; di <= 1 && is_max; ++di)
					for(int dj = -1; dj <= 1; ++dj)
						if(src[SUB2IND_2D(i + di, j + dj, N[1])] > v) { is_max = 0; break; }
				if(is_max) { fPeakHeight = v; i0 = i; j0 = j; }
			}
		}

		if(fPeakHeight <= 0 ||
		   i0 < (PKSIZE_X-1)/2 || i0 >= N[0]-(PKSIZE_X-1)/2  ||
		   j0 < (PKSIZE_Y-1)/2 || j0 >= N[1]-(PKSIZE_Y-1)/2)
		{
			peak_loc[SUB2IND_2D(0, iPeak, nPeaks)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, nPeaks)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(0, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(1, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(2, iPeak, nPeaks)] = 0;
			continue;
		}

		/* Extract subwindow, applying the loss-of-correlation compensation
		 * at each patch pixel's own plane coordinate (PIVware FI semantics:
		 * the fit must see envelope-corrected values or the envelope's local
		 * slope biases the sub-pixel position toward zero displacement). */
		for(i = 0; i < PKSIZE_X; ++i) {
			for(j = 0; j < PKSIZE_Y; ++j) {
				idx = SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[1]);
				subxcorr[i * PKSIZE_Y + j] = fPlaneWeight ? src[idx] * fPlaneWeight[idx] : src[idx];
			}
		}

		/* Perform fit based on type - only use higher order fits for first peak */
		int fit_status = 0;  /* LM fitters report failure; 3-point cannot fail past the gate */
		if(iPeak == 0) {
			switch(iFitType) {
				case 6:
					fit_status = lm_gauss6_fit(subxcorr, Nsub, peak, fitval, sig);
					break;
				case 5:
					fit_status = lm_gauss5_fit(subxcorr, Nsub, peak, fitval, sig);
					break;
				case 4:
					fit_status = lm_gauss4_fit(subxcorr, Nsub, peak, fitval, sig);
					break;
				case 3:
				default:
					/* 3-point estimator */
					{
						float A, sx, sy;
						threept_estimate(subxcorr, Nsub, peak, &A, &sx, &sy);
						sig[0] = sx;
						sig[1] = sy;
						sig[2] = 0.0f;
						for(i = 0; i < PKSIZE_X; ++i) {
							float fi = (float)(i - (PKSIZE_X-1)/2);
							for(j = 0; j < PKSIZE_Y; ++j) {
								float fj = (float)(j - (PKSIZE_Y-1)/2);
								fitval[i * PKSIZE_Y + j] = eval_gauss5(fi, fj, A, peak[0], peak[1], sx, sy);
							}
						}
					}
					break;
			}
		} else {
			/* Subsequent peaks: use fast 3-point for speed */
			float A, sx, sy;
			threept_estimate(subxcorr, Nsub, peak, &A, &sx, &sy);
			sig[0] = sx;
			sig[1] = sy;
			sig[2] = 0.0f;
			for(i = 0; i < PKSIZE_X; ++i) {
				float fi = (float)(i - (PKSIZE_X-1)/2);
				for(j = 0; j < PKSIZE_Y; ++j) {
					float fj = (float)(j - (PKSIZE_Y-1)/2);
					fitval[i * PKSIZE_Y + j] = eval_gauss5(fi, fj, A, peak[0], peak[1], sx, sy);
				}
			}
		}

		/* Save results. A FAILED LM fit gets the same NaN sentinel as a failed
		 * peak search above — the caller must not mistake a non-converged /
		 * degenerate fit for a converged one (no silent fallbacks). Downstream
		 * already treats NaN as the invalid-vector marker. */
		if(fit_status != 0) {
			peak_loc[SUB2IND_2D(0, iPeak, nPeaks)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, nPeaks)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(0, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(1, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(2, iPeak, nPeaks)] = 0;
		} else {
			peak_loc[SUB2IND_2D(0, iPeak, nPeaks)] = peak[0] + i0;
			peak_loc[SUB2IND_2D(1, iPeak, nPeaks)] = peak[1] + j0;
			peak_loc[SUB2IND_2D(2, iPeak, nPeaks)] = fPeakHeight;
			std_dev[SUB2IND_2D(0, iPeak, nPeaks)] = sig[0];
			std_dev[SUB2IND_2D(1, iPeak, nPeaks)] = sig[1];
			std_dev[SUB2IND_2D(2, iPeak, nPeaks)] = sig[2];
		}

		/* Subtract fit from correlation plane (only needed to find the next
		 * peak; for nPeaks==1 there is no copy to write and no next search).
		 * Runs with the last-best fitval even on a failed fit, so peak-2+
		 * search behaviour is unchanged. */
		if(nPeaks > 1) {
			for(i = 0; i < PKSIZE_X; ++i) {
				for(j = 0; j < PKSIZE_Y; ++j) {
					idx = SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[1]);
					/* fitval fits the COMPENSATED patch; the copy holds the
					 * raw plane — un-compensate before subtracting. */
					float f = fitval[i * PKSIZE_Y + j];
					if(fPlaneWeight) f /= fPlaneWeight[idx];
					xcorr_copy[idx] = MAX(0, xcorr_copy[idx] - f);
				}
			}
		}
	}

	if(xcorr_copy) free(xcorr_copy);
}

/* CPU capability check for the wheel's ISA floor. The x86 wheels are built
 * with AVX2+FMA (Haswell 2013+ / Excavator+); arm64 NEON is architectural
 * baseline. Deliberately NOT __builtin_cpu_supports: that builtin drags in
 * the libgcc/compiler-rt __cpu_model runtime (undefined symbol under
 * clang-cl, and a shared-library init dependency on Linux). Explicit
 * CPUID + XGETBV instructions have no runtime dependence at all. */
#if defined(_M_X64)
#include <intrin.h>              /* MSVC cl and clang-cl */
#endif

PEAK_EXPORT int pivtools_cpu_supported(void)
{
#if defined(_M_X64) || defined(__x86_64__)
	int eax1, ebx1, ecx1, edx1;      /* CPUID leaf 1 */
	int ebx7;                        /* CPUID leaf 7.0 EBX */
	unsigned int xlo, xhi;           /* XGETBV(0) = XCR0 */
#if defined(_M_X64)
	int info[4];
	unsigned long long xcr0;
	__cpuid(info, 1);
	ecx1 = info[2];
	(void)eax1; (void)ebx1; (void)edx1;
	if (!(ecx1 & (1 << 12)))         /* FMA */
		return 0;
	if (!(ecx1 & (1 << 27)))         /* OSXSAVE: XGETBV usable */
		return 0;
	xcr0 = _xgetbv(0);
	if ((xcr0 & 0x6) != 0x6)         /* OS saves XMM+YMM state */
		return 0;
	__cpuidex(info, 7, 0);
	ebx7 = info[1];
#else
	__asm__ volatile("cpuid"
	                 : "=a"(eax1), "=b"(ebx1), "=c"(ecx1), "=d"(edx1)
	                 : "a"(1), "c"(0));
	if (!(ecx1 & (1 << 12)))         /* FMA */
		return 0;
	if (!(ecx1 & (1 << 27)))         /* OSXSAVE: XGETBV usable */
		return 0;
	__asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
	if ((xlo & 0x6) != 0x6)          /* OS saves XMM+YMM state */
		return 0;
	__asm__ volatile("cpuid"
	                 : "=a"(eax1), "=b"(ebx7), "=c"(ecx1), "=d"(edx1)
	                 : "a"(7), "c"(0));
#endif
	return (ebx7 & (1 << 5)) != 0;   /* AVX2 */
#else
	return 1;                        /* arm64: NEON is baseline */
#endif
}
