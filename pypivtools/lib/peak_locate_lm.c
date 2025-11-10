#include "peak_locate_lm.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/******************************************************************************
 * Lightweight Levenberg-Marquardt Implementation for Gaussian Peak Fitting
 *
 * OVERVIEW:
 * This module provides a fast, numerically stable implementation of Gaussian
 * peak fitting for Particle Image Velocimetry (PIV) correlation analysis.
 * It uses the Levenberg-Marquardt algorithm to fit 2D Gaussian models to
 * correlation peaks, enabling sub-pixel displacement accuracy.
 *
 * SUPPORTED GAUSSIAN MODELS:
 * - 3-point parabolic estimator (fast fallback/initial guess)
 * - 4-DOF: Circular Gaussian (A, i0, j0, s)
 * - 5-DOF: Elliptical Gaussian (A, i0, j0, sx, sy)
 * - 6-DOF: Rotated elliptical Gaussian with correlation (A, i0, j0, sx, sy, sxy)
 *
 * ALGORITHM OVERVIEW:
 * 1. Initial peak detection using brute-force search in central region
 * 2. Sub-window extraction around detected peak
 * 3. Initial parameter estimation using 3-point parabolic fit
 * 4. Levenberg-Marquardt nonlinear least squares optimization
 * 5. Peak subtraction for multi-peak detection
 *
 * KEY FEATURES:
 * - Fast Cholesky-based linear system solving
 * - Robust parameter bounds and damping
 * - Automatic model selection based on fit type
 * - Multi-peak capability with sequential fitting
 *
 * TECHNICAL NOTES:
 * - Uses inverse covariance parameterization for 6-DOF model
 * - Implements trust-region Levenberg-Marquardt with adaptive damping
 * - Memory efficient with fixed-size matrices (max 6x6)
 * - Thread-safe (no global state)
 *
 ******************************************************************************/

/* Fast 3-point parabolic estimator - used as initial guess and fallback */
static void threept_estimate(const float *xcorr, const int *N, float *peak_loc, float *A_out, float *sx_out, float *sy_out)
{
	float x_fit[3], y_fit[3];
	int i;
	
	/* Extract 3 points along each axis centered on the correlation peak */
	for(i = 0; i < 3; ++i)
	{
		x_fit[i] = xcorr[(i - 1 + (N[0]-1)/2) * N[1] + (N[1]-1)/2];
		y_fit[i] = xcorr[(N[0]-1)/2 * N[1] + (i - 1 + (N[1]-1)/2)];
		/* Convert to log space for parabolic fitting (handles exponential decay) */
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

/* Compute residual and Jacobian for 4-DOF Gaussian fit */
static float compute_residual_jacobian_4dof(
	const float *xcorr, const int *N,
	float A, float i0, float j0, float s,
	float *JtJ, float *Jtr, int compute_jacobian)
{
	int ii, jj, idx;
	float residual_sum = 0.0f;
	const int n_params = 4;
	
	if(compute_jacobian) {
		memset(JtJ, 0, n_params * n_params * sizeof(float));
		memset(Jtr, 0, n_params * sizeof(float));
	}
	
	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);
		
		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;
			
			float pred = eval_gauss4(i, j, A, i0, j0, s);
			float r = pred - xcorr[idx];
			residual_sum += r * r;
			
			if(compute_jacobian) {
				float di = (i - i0) / s;
				float dj = (j - j0) / s;
				float r2 = di*di + dj*dj;
				
				/* Compute Jacobian matrix elements (partial derivatives) */
				float J[4];
				J[0] = pred / A;                           /* dF/dA */
				J[1] = 2.0f * pred * di / s;              /* dF/di0 */
				J[2] = 2.0f * pred * dj / s;              /* dF/dj0 */
				J[3] = 2.0f * pred * r2 / s;              /* dF/ds */
				
				/* Accumulate J^T * J and J^T * r for normal equations */
				for(int p1 = 0; p1 < n_params; ++p1) {
					Jtr[p1] += J[p1] * r;
					for(int p2 = 0; p2 <= p1; ++p2) {
						JtJ[p1 * n_params + p2] += J[p1] * J[p2];
					}
				}
			}
		}
	}
	
	if(compute_jacobian) {
		for(int p1 = 0; p1 < n_params; ++p1) {
			for(int p2 = p1 + 1; p2 < n_params; ++p2) {
				JtJ[p1 * n_params + p2] = JtJ[p2 * n_params + p1];
			}
		}
	}
	
	return residual_sum;
}

/* Compute residual and Jacobian for 5-DOF Gaussian fit */
static float compute_residual_jacobian_5dof(
	const float *xcorr, const int *N,
	float A, float i0, float j0, float sx, float sy,
	float *JtJ, float *Jtr, int compute_jacobian)
{
	int ii, jj, idx;
	float residual_sum = 0.0f;
	const int n_params = 5;
	
	if(compute_jacobian) {
		memset(JtJ, 0, n_params * n_params * sizeof(float));
		memset(Jtr, 0, n_params * sizeof(float));
	}
	
	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);
		
		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;
			
			float pred = eval_gauss5(i, j, A, i0, j0, sx, sy);
			float r = pred - xcorr[idx];
			residual_sum += r * r;
			
			if(compute_jacobian) {
				float di = (i - i0) / sx;
				float dj = (j - j0) / sy;
				
				/* Compute Jacobian matrix elements (partial derivatives) */
				float J[5];
				J[0] = pred / A;                    /* dF/dA */
				J[1] = 2.0f * pred * di / sx;      /* dF/di0 */
				J[2] = 2.0f * pred * dj / sy;      /* dF/dj0 */
				J[3] = 2.0f * pred * di * di / sx; /* dF/dsx */
				J[4] = 2.0f * pred * dj * dj / sy; /* dF/dsy */
				
				/* Accumulate J^T * J and J^T * r for normal equations */
				for(int p1 = 0; p1 < n_params; ++p1) {
					Jtr[p1] += J[p1] * r;
					for(int p2 = 0; p2 <= p1; ++p2) {
						JtJ[p1 * n_params + p2] += J[p1] * J[p2];
					}
				}
			}
		}
	}
	
	if(compute_jacobian) {
		for(int p1 = 0; p1 < n_params; ++p1) {
			for(int p2 = p1 + 1; p2 < n_params; ++p2) {
				JtJ[p1 * n_params + p2] = JtJ[p2 * n_params + p1];
			}
		}
	}
	
	return residual_sum;
}

/* Compute residual and Jacobian for 6-DOF Gaussian fit */
static float compute_residual_jacobian_6dof(
	const float *xcorr, const int *N,
	float A, float i0, float j0, float sx, float sy, float sxy,
	float *JtJ, float *Jtr, int compute_jacobian)
{
	int ii, jj, idx;
	float residual_sum = 0.0f;
	const int n_params = 6;
	
	if(compute_jacobian) {
		memset(JtJ, 0, n_params * n_params * sizeof(float));
		memset(Jtr, 0, n_params * sizeof(float));
	}
	
	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);
		
		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;
			
			float pred = eval_gauss6(i, j, A, i0, j0, sx, sy, sxy);
			float r = pred - xcorr[idx];
			residual_sum += r * r;
			
			if(compute_jacobian) {
				float di = i - i0;
				float dj = j - j0;
				
				/* Compute Jacobian matrix elements (partial derivatives) */
				float J[6];
				J[0] = pred / A;                                    /* dF/dA */
				J[1] = pred * (di/sx + dj*sxy);                    /* dF/di0 */
				J[2] = pred * (dj/sy + di*sxy);                    /* dF/dj0 */
				J[3] = 0.5f * pred * di * di / (sx * sx);          /* dF/dsx -  */
				J[4] = 0.5f * pred * dj * dj / (sy * sy);          /* dF/dsy -  */
				J[5] = -pred * di * dj;                            /* dF/dsxy */
				
				/* Accumulate J^T * J and J^T * r for normal equations */
				for(int p1 = 0; p1 < n_params; ++p1) {
					Jtr[p1] += J[p1] * r;
					for(int p2 = 0; p2 <= p1; ++p2) {
						JtJ[p1 * n_params + p2] += J[p1] * J[p2];
					}
				}
			}
		}
	}
	
	if(compute_jacobian) {
		for(int p1 = 0; p1 < n_params; ++p1) {
			for(int p2 = p1 + 1; p2 < n_params; ++p2) {
				JtJ[p1 * n_params + p2] = JtJ[p2 * n_params + p1];
			}
		}
	}
	
	return residual_sum;
}

/* Solve (JtJ + lambda*diag(JtJ)) * delta = -Jtr using Cholesky decomposition */
static int solve_lm_step(const float *JtJ, const float *Jtr, float lambda, float *delta, int n)
{
	float A[36]; /* Max 6x6 matrix */
	float L[36];
	float y[6];
	int i, j, k;
	
	/* Form damped normal matrix: A = JtJ + lambda * diag(JtJ) */
	memcpy(A, JtJ, n * n * sizeof(float));
	for(i = 0; i < n; ++i) {
		A[i * n + i] *= (1.0f + lambda);
	}
	
	/* Cholesky decomposition: A = L * L^T */
	memset(L, 0, n * n * sizeof(float));
	for(i = 0; i < n; ++i) {
		for(j = 0; j <= i; ++j) {
			float sum = A[i * n + j];
			for(k = 0; k < j; ++k) {
				sum -= L[i * n + k] * L[j * n + k];
			}
			if(i == j) {
				if(sum <= 0.0f) return -1; /* Matrix not positive definite */
				L[i * n + j] = sqrtf(sum);
			} else {
				L[i * n + j] = sum / L[j * n + j];
			}
		}
	}
	
	/* Forward substitution: L * y = -Jtr */
	for(i = 0; i < n; ++i) {
		float sum = -Jtr[i];
		for(j = 0; j < i; ++j) {
			sum -= L[i * n + j] * y[j];
		}
		y[i] = sum / L[i * n + i];
	}
	
	/* Back substitution: L^T * delta = y */
	for(i = n - 1; i >= 0; --i) {
		float sum = y[i];
		for(j = i + 1; j < n; ++j) {
			sum -= L[j * n + i] * delta[j];
		}
		delta[i] = sum / L[i * n + i];
	}
	
	return 0;
}

/* Fast Levenberg-Marquardt for 4-DOF Gaussian fitting (circular) */
static void lm_gauss4_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, s;
	float JtJ[16], Jtr[4], delta[4];
	float lambda = 0.01f;
	float residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 20;
	const float tol = 1e-6f;
	
	/* Get initial guess using 3-point parabolic estimator */
	float sx, sy;
	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];
	s = sqrtf(sx * sx + sy * sy); /* Combined width */
	
	/* Clamp parameters to reasonable bounds */
	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	s = fminf(fmaxf(s, 0.5f), 3.0f);
	
	/* Compute initial residual and Jacobian */
	residual = compute_residual_jacobian_4dof(xcorr, N, A, i0, j0, s, JtJ, Jtr, 1);
	
	/* Levenberg-Marquardt optimization loop */
	for(iter = 0; iter < max_iter; ++iter) {
		/* Solve for parameter update */
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 4) != 0) break;
		
		/* Apply parameter update with bounds checking */
		float A_new = A + delta[0];
		float i0_new = i0 + delta[1];
		float j0_new = j0 + delta[2];
		float s_new = s + delta[3];
		
		A_new = fmaxf(A_new, A * 0.5f);
		i0_new = fminf(fmaxf(i0_new, -2.5f), 2.5f);
		j0_new = fminf(fmaxf(j0_new, -2.5f), 2.5f);
		s_new = fminf(fmaxf(s_new, 0.25f), 4.0f);
		
		/* Evaluate new residual */
		new_residual = compute_residual_jacobian_4dof(xcorr, N, A_new, i0_new, j0_new, s_new, NULL, NULL, 0);
		
		if(new_residual < residual) {
			/* Accept step: update parameters */
			A = A_new; i0 = i0_new; j0 = j0_new; s = s_new;
			float improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f; /* Decrease damping */
			compute_residual_jacobian_4dof(xcorr, N, A, i0, j0, s, JtJ, Jtr, 1);
			if(improvement < tol) break; /* Converged */
		} else {
			lambda *= 2.0f; /* Increase damping */
			if(lambda > 1e6f) break; /* Diverged */
		}
	}
	
	/* Store final results */
	peak_loc[0] = i0;
	peak_loc[1] = j0;
	sig[0] = s;
	sig[1] = s;
	sig[2] = 0.0f;
	
	/* Generate fitted values if requested */
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
}

/* Fast Levenberg-Marquardt for 5-DOF Gaussian fitting */
static void lm_gauss5_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, sx, sy;
	float JtJ[25], Jtr[5], delta[5];
	float lambda = 0.01f;
	float residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 20;
	const float tol = 1e-6f;
	
	/* Get initial guess using 3-point parabolic estimator */
	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];
	
	/* Clamp parameters to reasonable bounds */
	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	sx = fminf(fmaxf(sx, 0.5f), 3.0f);
	sy = fminf(fmaxf(sy, 0.5f), 3.0f);
	
	/* Compute initial residual and Jacobian */
	residual = compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, JtJ, Jtr, 1);
	
	/* Levenberg-Marquardt optimization loop */
	for(iter = 0; iter < max_iter; ++iter) {
		/* Solve for parameter update */
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 5) != 0) break;
		
		/* Apply parameter update with bounds checking */
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
		
		/* Evaluate new residual */
		new_residual = compute_residual_jacobian_5dof(xcorr, N, A_new, i0_new, j0_new, sx_new, sy_new, NULL, NULL, 0);
		
		if(new_residual < residual) {
			/* Accept step: update parameters */
			A = A_new; i0 = i0_new; j0 = j0_new; sx = sx_new; sy = sy_new;
			float improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f; /* Decrease damping */
			compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, JtJ, Jtr, 1);
			if(improvement < tol) break; /* Converged */
		} else {
			lambda *= 2.0f; /* Increase damping */
			if(lambda > 1e6f) break; /* Diverged */
		}
	}
	
	/* Store final results */
	peak_loc[0] = i0;
	peak_loc[1] = j0;
	sig[0] = sx;
	sig[1] = sy;
	sig[2] = 0.0f;
	
	/* Generate fitted values if requested */
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
}

/* Fast Levenberg-Marquardt for 6-DOF Gaussian fitting 
 */
static void lm_gauss6_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, sx, sy, sxy;
	float JtJ[36], Jtr[6], delta[6];
	float lambda = 0.01f;
	float residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 20;
	const float tol = 1e-6f;
	
	/* Get initial guess using 3-point parabolic estimator */
	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];
	sxy = 0.0f; /* Initial correlation coefficient */
	
	/* Clamp parameters to reasonable bounds */
	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	sx = fminf(fmaxf(sx * sx, 0.25f), 9.0f); /* Convert to variance */
	sy = fminf(fmaxf(sy * sy, 0.25f), 9.0f);
	
	/* Compute initial residual and Jacobian */
	residual = compute_residual_jacobian_6dof(xcorr, N, A, i0, j0, sx, sy, sxy, JtJ, Jtr, 1);
	
	/* Levenberg-Marquardt optimization loop */
	for(iter = 0; iter < max_iter; ++iter) {
		/* Solve for parameter update */
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 6) != 0) break;
		
		/* Apply parameter update with bounds checking */
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
		float sxy_max = 0.95f / sqrtf(sx_new * sy_new); /* Prevent singular covariance */
		sxy_new = fminf(fmaxf(sxy_new, -sxy_max), sxy_max);
		
		/* Evaluate new residual */
		new_residual = compute_residual_jacobian_6dof(xcorr, N, A_new, i0_new, j0_new, sx_new, sy_new, sxy_new, NULL, NULL, 0);
		
		if(new_residual < residual) {
			/* Accept step: update parameters */
			A = A_new; i0 = i0_new; j0 = j0_new; 
			sx = sx_new; sy = sy_new; sxy = sxy_new;
			float improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f; /* Decrease damping */
			compute_residual_jacobian_6dof(xcorr, N, A, i0, j0, sx, sy, sxy, JtJ, Jtr, 1);
			if(improvement < tol) break; /* Converged */
		} else {
			lambda *= 2.0f; /* Increase damping */
			if(lambda > 1e6f) break; /* Diverged */
		}
	}
	
	/* Store final results */
	peak_loc[0] = i0;
	peak_loc[1] = j0;
	/* WARNING: Output parameters are swapped! sig[0]=sy, sig[1]=sx
	 * This is confusing and should be refactored. The 6-DOF model uses
	 * inverse covariance parameterization which differs from standard
	 * Gaussian parameterization used in 4-DOF and 5-DOF models. */
	sig[0] = sy;
	sig[1] = sx;
	sig[2] = -(sxy * (sx * sy - sxy * sxy)); /* Covariance determinant */
	
	/* Generate fitted values if requested */
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
}

/******************************************************************************
 * Main peak localization function
 * 
 * Performs multi-peak detection and fitting on correlation data.
 * 
 * Parameters:
 *   xcorr: Input correlation array (N[0] x N[1])
 *   N: Dimensions of correlation array [rows, cols]
 *   peak_loc: Output peak locations (3 x nPeaks array)
 *             peak_loc[0,i] = x-coordinate, peak_loc[1,i] = y-coordinate, peak_loc[2,i] = peak height
 *   nPeaks: Maximum number of peaks to find
 *   iFitType: Fitting model (3=3pt, 4=circular, 5=elliptical, 6=rotated elliptical)
 *   std_dev: Output standard deviations (3 x nPeaks array)
 *            std_dev[0,i] = sigma_x, std_dev[1,i] = sigma_y, std_dev[2,i] = correlation/covariance
 *****************************************************************************/
void lsqpeaklocate_lm(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev)
{
	int i, j, iPeak, idx;
	int i0, j0;
	float *xcorr_copy;
	float fPeakHeight;
	float subxcorr[PKSIZE_X * PKSIZE_Y];
	float fitval[PKSIZE_X * PKSIZE_Y];
	int Nsub[2];
	float peak[2];
	float sig[3];
	
	/* Create working copy of correlation data for peak subtraction */
	xcorr_copy = (float*)malloc(sizeof(float) * N[0] * N[1]);
	memcpy(xcorr_copy, xcorr, N[0] * N[1] * sizeof(float));
	Nsub[0] = PKSIZE_X;
	Nsub[1] = PKSIZE_Y;
	
	/* Process each peak sequentially */
	for(iPeak = 0; iPeak < nPeaks; ++iPeak)
	{
		/* Find highest remaining peak in central region 
		 * N[0] = number of rows, N[1] = number of columns
		 * i indexes rows, j indexes columns
		 * Row-major indexing: SUB2IND_2D(row, col, num_cols)
		 */
		i0 = j0 = 0;
		fPeakHeight = 0;
		for(i = N[0]/8; i < N[0]*7/8; ++i) {
			for(j = N[1]/8; j < N[1]*7/8; ++j) {
				if(xcorr_copy[SUB2IND_2D(i, j, N[1])] > fPeakHeight) {
					fPeakHeight = xcorr_copy[SUB2IND_2D(i, j, N[1])];
					i0 = i;
					j0 = j;
				}
			}
		}
		
		/* Validate peak: check height and local maximum condition 
		 * Row-major indexing: SUB2IND_2D(row, col, num_cols)
		 */
		if(fPeakHeight <= 0 ||
		   i0 < (PKSIZE_X-1)/2 || i0 >= N[0]-(PKSIZE_X-1)/2  ||
		   j0 < (PKSIZE_Y-1)/2 || j0 >= N[1]-(PKSIZE_Y-1)/2 ||
		   fPeakHeight <= xcorr_copy[SUB2IND_2D(i0-1, j0, N[1])] ||
		   fPeakHeight <= xcorr_copy[SUB2IND_2D(i0+1, j0, N[1])] ||
		   fPeakHeight <= xcorr_copy[SUB2IND_2D(i0, j0-1, N[1])] ||
		   fPeakHeight <= xcorr_copy[SUB2IND_2D(i0, j0+1, N[1])])
		{
			/* Invalid peak: mark as NaN 
			 * Row-major: array[3, nPeaks] -> stride is nPeaks
			 */
			peak_loc[SUB2IND_2D(0, iPeak, nPeaks)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, nPeaks)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(0, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(1, iPeak, nPeaks)] = 0;
			std_dev[SUB2IND_2D(2, iPeak, nPeaks)] = 0;
			continue;
		}
		
		/* Extract sub-window around peak for fitting 
		 * Row-major indexing for both subxcorr and xcorr_copy
		 */
		for(i = 0; i < PKSIZE_X; ++i) {
			for(j = 0; j < PKSIZE_Y; ++j) {
				subxcorr[i * PKSIZE_Y + j] = xcorr_copy[SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[1])];
			}
		}
		
		/* Perform fit based on type - only use higher order fits for first peak */
		if(iPeak == 0) {
			switch(iFitType) {
				case 6:
					lm_gauss6_fit(subxcorr, Nsub, peak, fitval, sig);
					break;
				case 5:
					lm_gauss5_fit(subxcorr, Nsub, peak, fitval, sig);
					break;
				case 4:
					lm_gauss4_fit(subxcorr, Nsub, peak, fitval, sig);
					break;
				case 3:
				default:
					/* 3-point estimator fallback */
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
		
		/* Save results: convert sub-window coordinates back to full image 
		 * peak[0] = row offset, peak[1] = col offset (both relative to sub-window center)
		 * i0 = row index of peak in full correlation plane
		 * j0 = col index of peak in full correlation plane
		 * Output: peak_loc is [3, nPeaks] array in row-major
		 *   peak_loc[0, iPeak] = row (Y) position
		 *   peak_loc[1, iPeak] = col (X) position
		 *   peak_loc[2, iPeak] = peak height
		 */
		peak_loc[SUB2IND_2D(0, iPeak, nPeaks)] = peak[0] + i0;  // Row (Y)
		peak_loc[SUB2IND_2D(1, iPeak, nPeaks)] = peak[1] + j0;  // Col (X)
		peak_loc[SUB2IND_2D(2, iPeak, nPeaks)] = fPeakHeight;
		std_dev[SUB2IND_2D(0, iPeak, nPeaks)] = sig[0];
		std_dev[SUB2IND_2D(1, iPeak, nPeaks)] = sig[1];
		std_dev[SUB2IND_2D(2, iPeak, nPeaks)] = sig[2];
		
		/* Subtract fitted peak from correlation plane to find remaining peaks 
		 * Row-major indexing
		 */
		for(i = 0; i < PKSIZE_X; ++i) {
			for(j = 0; j < PKSIZE_Y; ++j) {
				idx = SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[1]);
				xcorr_copy[idx] = MAX(0, xcorr_copy[idx] - fitval[i * PKSIZE_Y + j]);
			}
		}
	}
	
	free(xcorr_copy);
}
			