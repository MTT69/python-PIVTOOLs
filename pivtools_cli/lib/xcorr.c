/****************************************************************************
 * xcorr.c -- cross-correlation primitives, FFTW-free.
 *
 * Thin adapter over the codelet FFT engine (codelet_fft.c). The transform
 * math (r2c, conjugate-multiply, c2r, 1/numel normalize + fftshift) is
 * identical to the previous FFTW implementation -- only the FFT engine
 * changed, so the correlation surfaces are the same to float32 tolerance
 * (see test_codelet_gate.c).
 ****************************************************************************/
#include "xcorr.h"
#include "common.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/****************************************************
 * direct_xcorr_shifted(A, B, out, H, W)
 *
 * out = fftshift( IFFT( FFT(A) .* conj(FFT(B)) ) ) / (H*W)
 * computed directly (no FFT) as a circular cross-correlation. This is the
 * exact operation the old FFTW `xcorr` performed; it is numerically equal to
 * the codelet engine (gate-verified to ~3e-7). Used only by `convolve`, which
 * runs once per pass on zero-padded weight windows -- not the hot path. The
 * zero-skip keeps it O((H*W)*nnz) rather than O((H*W)^2).
 */
static void direct_xcorr_shifted(const float *A, const float *B,
                                 float *out, int H, int W)
{
	const double mul = 1.0 / ((double)H * (double)W);
	int m;  /* MSVC OpenMP needs the counter declared outside the for-init */
#ifdef _OPENMP
	#pragma omp parallel for schedule(static)
#endif
	for (m = 0; m < H; ++m)
	{
		for (int n = 0; n < W; ++n)
		{
			double acc = 0.0;
			for (int i = 0; i < H; ++i)
			{
				int ii = ((i - m) % H + H) % H;
				const float *Arow = &A[i * W];
				const float *Brow = &B[ii * W];
				for (int j = 0; j < W; ++j)
				{
					float a = Arow[j];
					if (a == 0.0f) continue;          /* skip zero-padding */
					int jj = ((j - n) % W + W) % W;
					acc += (double)a * (double)Brow[jj];
				}
			}
			int row_swap = (m + H / 2) % H;
			int col_swap = (n + W / 2) % W;
			out[row_swap * W + col_swap] = (float)(acc * mul);
		}
	}
}

/****************************************************
 * unsigned convolve(a, b, c, N)
 * cross-correlation of a with b via zero-padded correlation, centre extracted.
 * Used for the correlation-plane weighting (autocorrelation of the window
 * weight). FFTW-free; behaviour identical to the previous implementation.
 */
unsigned convolve(const float *fA, const float *fB, float *fC, const int *N)
{
	int Npad[2];
	int Nvox;
	int i, j;
	unsigned uError;
	float *fApad, *fBpad, *fCpad;

	/* allocate memory for padded versions of a, b and c */
	Npad[0] = N[0]*2;
	Npad[1] = N[1]*2;
	Nvox	  = Npad[0] * Npad[1];

	fApad		= (float*)malloc(Nvox * sizeof(float));
	fBpad		= (float*)malloc(Nvox * sizeof(float));
	fCpad		= (float*)malloc(Nvox * sizeof(float));
	if(!fApad || !fBpad || !fCpad)
	{
		if(fApad) free(fApad);
		if(fBpad) free(fBpad);
		if(fCpad) free(fCpad);
		return ERROR_NOMEM;
	}

	/* Copy input to zero-padded arrays (row-major) */
	memset(fApad, 0, Nvox * sizeof(float));
	memset(fBpad, 0, Nvox * sizeof(float));
	for(i = 0; i < N[0]; ++i)  /* rows */
	{
		for(j = 0; j < N[1]; ++j)  /* columns */
		{
			fApad[SUB2IND_2D(i+N[0]/2, j+N[1]/2, Npad[1])] =
				fA[SUB2IND_2D(i, j, N[1])];
			fBpad[SUB2IND_2D(i+N[0]/2, j+N[1]/2, Npad[1])] =
				fB[SUB2IND_2D(i, j, N[1])];
		}
	}

	/* Cross-correlate via the codelet FFT at the padded size 2N (this is exactly
	 * what the old FFTW path did). The 2N codelets -- including 192/256 for the
	 * 96/128 windows -- make this O(N^2 log N) instead of the previous O(N^4)
	 * direct sum, which dominated the pipeline at large windows. */
	uError = xcorr(fApad, fBpad, fCpad, Npad);
	if (uError != ERROR_NONE)
	{
		free(fApad);
		free(fBpad);
		free(fCpad);
		return uError;
	}

	/* copy centre of fCpad into fC */
	for(i = 0; i < N[0]; ++i)
	{
		for(j = 0; j < N[1]; ++j)
		{
			fC[SUB2IND_2D(i, j, N[1])] =
				fCpad[SUB2IND_2D(i+N[0]/2, j+N[1]/2, Npad[1])];
		}
	}

	/* free memory */
	free(fApad);
	free(fBpad);
	free(fCpad);

	return ERROR_NONE;
}


/****************************************************
 * unsigned xcorr_create_plan(N, planstruct)
 *
 * Create a codelet-backed plan. Unlike the old FFTW planner this is
 * thread-safe (malloc only), but callers may still wrap it in omp critical;
 * that is harmless.
 */
unsigned xcorr_create_plan(const int *N, sPlan *pPlanStruct)
{
	if (!pPlanStruct) return ERROR_NOMEM;

	if (N[0] <= 0 || N[1] <= 0) {
		fprintf(stderr, "xcorr_create_plan: invalid N: N[0]=%d N[1]=%d\n", N[0], N[1]);
		return ERROR_NOPLAN_BWD;
	}
	if (!codelet_size_ok(N[0]) || !codelet_size_ok(N[1])) {
		fprintf(stderr, "xcorr_create_plan: window %dx%d is not a built codelet size "
		                "(supported: 8 12 16 24 32 48 64 96 128)\n", N[0], N[1]);
		return ERROR_NOPLAN_BWD;
	}

	pPlanStruct->cp = codelet_plan_create(N[0], N[1]);
	if (!pPlanStruct->cp) {
		fprintf(stderr, "xcorr_create_plan: codelet_plan_create failed for %dx%d\n", N[0], N[1]);
		return ERROR_NOMEM;
	}
	pPlanStruct->N[0] = N[0];
	pPlanStruct->N[1] = N[1];
	return ERROR_NONE;
}


/****************************************************
 * unsigned xcorr_destroy_plan(planstruct)
 */
unsigned xcorr_destroy_plan(sPlan *pPlanStruct)
{
	if(!pPlanStruct)
		return ERROR_NOMEM;
	codelet_plan_destroy(pPlanStruct->cp);
	pPlanStruct->cp = NULL;
	return ERROR_NONE;
}

/****************************************************
 * unsigned xcorr_preplanned(a, b, c, sPlan)
 *
 * c = fftshift( IFFT(FFT(a) .* conj(FFT(b))) ) / numel
 * Thread-safe given a per-thread plan.
 */
unsigned xcorr_preplanned(const float *a, const float *b, float *c, sPlan *pPlanStruct)
{
	if(!pPlanStruct || !pPlanStruct->cp)
		return ERROR_NOMEM;

	codelet_forward(pPlanStruct->cp, a, 0);   /* slot 0 plays role of `a` */
	codelet_forward(pPlanStruct->cp, b, 1);   /* slot 1 plays role of `b` */
	codelet_emit_xcorr(pPlanStruct->cp, c);   /* spec0 .* conj(spec1) -> ifft -> shift */

	return ERROR_NONE;
}

/****************************************************
 * unsigned xcorr(a, b, c, N)
 * Single-shot correlation (create plan -> run -> destroy). Built sizes only.
 */
unsigned xcorr(const float *a, const float *b, float *c, const int *N)
{
	sPlan spPlan;
	unsigned uError;

	memset(&spPlan, 0, sizeof(spPlan));
	uError = xcorr_create_plan(N, &spPlan);
	if(uError != ERROR_NONE)
		return uError;

	uError = xcorr_preplanned(a, b, c, &spPlan);

	xcorr_destroy_plan(&spPlan);

	return uError;
}
