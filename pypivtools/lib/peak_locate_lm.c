#include "peak_locate_lm.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/******************************************************************************
 * Lightweight Levenberg-Marquardt implementation for Gaussian peak fitting
 * 
 * This replaces the heavy GSL multifit solver with a fast, specialized
 * implementation optimized for PIV peak localization.
 * 
 * Key optimizations:
 * - Direct Jacobian computation without matrix library overhead
 * - Small fixed-size problems (5x5 windows) allow stack allocation
 * - Specialized for Gaussian fitting (no general-purpose solver overhead)
 * - Fewer iterations needed for PIV correlation peaks
 *****************************************************************************/

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

/* Evaluate 2D Gaussian: A * exp(-((i-i0)^2/sx^2 + (j-j0)^2/sy^2)) */
static inline float eval_gauss(float i, float j, float A, float i0, float j0, float sx, float sy)
{
	float di = (i - i0) / sx;
	float dj = (j - j0) / sy;
	return A * expf(-(di*di + dj*dj));
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
	
	/* Initialize outputs */
	if(compute_jacobian) {
		memset(JtJ, 0, n_params * n_params * sizeof(float));
		memset(Jtr, 0, n_params * sizeof(float));
	}
	
	/* Loop over all pixels in window */
	for(ii = 0; ii < N[0]; ++ii) {
		float i = (float)(ii - (N[0]-1)/2);
		
		for(jj = 0; jj < N[1]; ++jj) {
			float j = (float)(jj - (N[1]-1)/2);
			idx = ii * N[1] + jj;
			
			/* Compute prediction and residual */
			float pred = eval_gauss(i, j, A, i0, j0, sx, sy);
			float r = pred - xcorr[idx];
			residual_sum += r * r;
			
			if(compute_jacobian) {
				/* Compute Jacobian: J = [dF/dA, dF/di0, dF/dj0, dF/dsx, dF/dsy] */
				float di = (i - i0) / sx;
				float dj = (j - j0) / sy;
				
				float J[5];
				J[0] = pred / A;                    /* dF/dA */
				J[1] = 2.0f * pred * di / sx;      /* dF/di0 */
				J[2] = 2.0f * pred * dj / sy;      /* dF/dj0 */
				J[3] = 2.0f * pred * di * di / sx; /* dF/dsx */
				J[4] = 2.0f * pred * dj * dj / sy; /* dF/dsy */
				
				/* Accumulate J^T * J and J^T * r */
				for(int p1 = 0; p1 < n_params; ++p1) {
					Jtr[p1] += J[p1] * r;
					for(int p2 = 0; p2 <= p1; ++p2) {
						JtJ[p1 * n_params + p2] += J[p1] * J[p2];
					}
				}
			}
		}
	}
	
	/* Fill upper triangle of JtJ (symmetric matrix) */
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
	float A[25]; /* Max 5x5 matrix */
	float L[25];
	int i, j, k;
	
	/* A = JtJ + lambda * diag(JtJ) */
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
				if(sum <= 0.0f) return -1; /* Not positive definite */
				L[i * n + j] = sqrtf(sum);
			} else {
				L[i * n + j] = sum / L[j * n + j];
			}
		}
	}
	
	/* Forward substitution: L * y = -Jtr */
	float y[5];
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

/* Fast Levenberg-Marquardt for 5-DOF Gaussian fitting */
void lm_gauss5_fit(const float *xcorr, const int *N, float *peak_loc, float *fitval, float *sig)
{
	float A, i0, j0, sx, sy;
	float JtJ[25], Jtr[5], delta[5];
	float lambda = 0.01f;
	float residual, new_residual;
	int iter, ii, jj, idx;
	const int max_iter = 20; /* Reduced from 100 - PIV peaks converge fast */
	const float tol = 1e-6f;
	
	/* Get initial guess from 3-point estimator */
	threept_estimate(xcorr, N, peak_loc, &A, &sx, &sy);
	i0 = peak_loc[0];
	j0 = peak_loc[1];
	
	/* Clamp initial guess to reasonable bounds */
	i0 = fminf(fmaxf(i0, -2.0f), 2.0f);
	j0 = fminf(fmaxf(j0, -2.0f), 2.0f);
	sx = fminf(fmaxf(sx, 0.5f), 3.0f);
	sy = fminf(fmaxf(sy, 0.5f), 3.0f);
	
	/* Initial residual */
	residual = compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, JtJ, Jtr, 1);
	
	/* Levenberg-Marquardt iterations */
	for(iter = 0; iter < max_iter; ++iter) {
		/* Solve for step */
		if(solve_lm_step(JtJ, Jtr, lambda, delta, 5) != 0) {
			break; /* Singular matrix, stop */
		}
		
		/* Try step */
		float A_new = A + delta[0];
		float i0_new = i0 + delta[1];
		float j0_new = j0 + delta[2];
		float sx_new = sx + delta[3];
		float sy_new = sy + delta[4];
		
		/* Enforce bounds */
		A_new = fmaxf(A_new, A * 0.5f);
		i0_new = fminf(fmaxf(i0_new, -2.5f), 2.5f);
		j0_new = fminf(fmaxf(j0_new, -2.5f), 2.5f);
		sx_new = fminf(fmaxf(sx_new, 0.25f), 4.0f);
		sy_new = fminf(fmaxf(sy_new, 0.25f), 4.0f);
		
		/* Evaluate new residual */
		new_residual = compute_residual_jacobian_5dof(xcorr, N, A_new, i0_new, j0_new, sx_new, sy_new, NULL, NULL, 0);
		
		/* Accept or reject step */
		if(new_residual < residual) {
			/* Good step: accept and decrease damping */
			A = A_new;
			i0 = i0_new;
			j0 = j0_new;
			sx = sx_new;
			sy = sy_new;
			
			float improvement = (residual - new_residual) / (residual + FLT_EPSILON);
			residual = new_residual;
			lambda *= 0.5f;
			
			/* Recompute Jacobian at new point */
			compute_residual_jacobian_5dof(xcorr, N, A, i0, j0, sx, sy, JtJ, Jtr, 1);
			
			/* Check convergence */
			if(improvement < tol) {
				break;
			}
		} else {
			/* Bad step: reject and increase damping */
			lambda *= 2.0f;
			if(lambda > 1e6f) {
				break; /* Damping too large, stop */
			}
		}
	}
	
	/* Store results */
	peak_loc[0] = i0;
	peak_loc[1] = j0;
	sig[0] = sx;
	sig[1] = sy;
	sig[2] = 0.0f; /* No correlation term in 5-DOF fit */
	
	/* Evaluate fit function if requested */
	if(fitval) {
		for(ii = 0; ii < N[0]; ++ii) {
			float i = (float)(ii - (N[0]-1)/2);
			for(jj = 0; jj < N[1]; ++jj) {
				float j = (float)(jj - (N[1]-1)/2);
				idx = ii * N[1] + jj;
				fitval[idx] = eval_gauss(i, j, A, i0, j0, sx, sy);
			}
		}
	}
}

/******************************************************************************
 * Main peak localization function - drop-in replacement for lsqpeaklocate
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
	
	/* Make a copy of xcorr that we can manipulate */
	xcorr_copy = (float*)malloc(sizeof(float) * N[0] * N[1]);
	memcpy(xcorr_copy, xcorr, N[0] * N[1] * sizeof(float));
	Nsub[0] = PKSIZE_X;
	Nsub[1] = PKSIZE_Y;
	
	/* Iterate over peaks */
	for(iPeak = 0; iPeak < nPeaks; ++iPeak)
	{
		/* Find maximum value in array - search in central 3/4 of domain */
		i0 = j0 = 0;
		fPeakHeight = 0;
		for(i = N[0]/8; i < N[0]*7/8; ++i)
		{
			for(j = N[1]/8; j < N[1]*7/8; ++j)
			{
				if(xcorr_copy[SUB2IND_2D(i, j, N[0])] > fPeakHeight)
				{
					fPeakHeight = xcorr_copy[SUB2IND_2D(i, j, N[0])];
					i0 = i;
					j0 = j;
				}
			}
		}
		
		/* Error out if peak height is not positive */
		if(fPeakHeight <= 0)
		{
			peak_loc[SUB2IND_2D(0, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, 3)] = 0;
			continue;
		}
		
		/* Error out if too close to edges or not a local maximum */
		if(i0 < (PKSIZE_X-1)/2 || i0 >= N[0]-(PKSIZE_X-1)/2  
			|| j0 < (PKSIZE_Y-1)/2 || j0 >= N[1]-(PKSIZE_Y-1)/2 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0-1, j0, N[0])] 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0+1, j0, N[0])] 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0, j0-1, N[0])] 
			|| fPeakHeight <= xcorr_copy[SUB2IND_2D(i0, j0+1, N[0])])
		{
			peak_loc[SUB2IND_2D(0, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(1, iPeak, 3)] = NAN;
			peak_loc[SUB2IND_2D(2, iPeak, 3)] = 0;
			continue;
		}
		
		/* Extract subwindow around peak */
		for(i = 0; i < PKSIZE_X; ++i)
		{
			for(j = 0; j < PKSIZE_Y; ++j)
			{
				subxcorr[i * PKSIZE_Y + j] = xcorr_copy[SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[0])];
			}
		}
		
		/* Perform least-squares fit
		 * For first peak, use requested fit type; for subsequent peaks use 3-point
		 */
		if(iPeak == 0 && iFitType >= 4) {
			/* Use fast LM solver for 5-DOF fit (iFitType 4, 5, 6 all use 5-DOF) */
			lm_gauss5_fit(subxcorr, Nsub, peak, fitval, sig);
		} else {
			/* Use 3-point estimator for speed */
			float A, sx, sy;
			threept_estimate(subxcorr, Nsub, peak, &A, &sx, &sy);
			sig[0] = sx;
			sig[1] = sy;
			sig[2] = 0.0f;
			
			/* Evaluate fit */
			for(i = 0; i < PKSIZE_X; ++i) {
				float fi = (float)(i - (PKSIZE_X-1)/2);
				for(j = 0; j < PKSIZE_Y; ++j) {
					float fj = (float)(j - (PKSIZE_Y-1)/2);
					fitval[i * PKSIZE_Y + j] = eval_gauss(fi, fj, A, peak[0], peak[1], sx, sy);
				}
			}
		}
		
		/* Save peak location and subtract fit from correlation plane */
		peak_loc[SUB2IND_2D(0, iPeak, 3)] = peak[0] + i0;
		peak_loc[SUB2IND_2D(1, iPeak, 3)] = peak[1] + j0;
		peak_loc[SUB2IND_2D(2, iPeak, 3)] = fPeakHeight;
		std_dev[SUB2IND_2D(0, iPeak, 3)] = sig[0];
		std_dev[SUB2IND_2D(1, iPeak, 3)] = sig[1];
		std_dev[SUB2IND_2D(2, iPeak, 3)] = sig[2];
		
		for(i = 0; i < PKSIZE_X; ++i)
		{
			for(j = 0; j < PKSIZE_Y; ++j)
			{
				idx = SUB2IND_2D(i0 + i - (PKSIZE_X-1)/2, j0 + j - (PKSIZE_Y-1)/2, N[0]);
				xcorr_copy[idx] = MAX(0, xcorr_copy[idx] - fitval[i * PKSIZE_Y + j]);
			}
		}
	}
	
	/* Clean up and exit */
	free(xcorr_copy);
	return;
}
