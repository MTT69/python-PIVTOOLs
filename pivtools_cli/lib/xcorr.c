#include "xcorr.h"
#include "common.h"
#include <fftw3.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/****************************************************
 * unsigned convolve(a, b, c, N)
 * convolve a with b and store the result in c
 * accomplished using zero padded cross-correlation of a and b
 */
unsigned convolve(const float *fA, const float *fB, float *fC, const int *N)
{
	int Npad[2];
	int Nvox;
	int i, j;
	float *fApad, *fBpad, *fCpad;
	unsigned uError;

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

	/* cross-correlate */
	uError = xcorr(fApad, fBpad, fCpad, Npad);

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

	return uError;
}


/****************************************************
 * unsigned xcorr_create_plan(N, planstruct)
 *
 * create plans for cross-correlation
 * xcorr_plan is NOT thread safe 
 * so run it inside an OMP critical directive
 *
 */

/* diagnostic / robust xcorr_create_plan */
unsigned xcorr_create_plan(const int *N, sPlan *pPlanStruct)
{
    fftwf_plan plan_AB_fft = NULL;
    fftwf_plan plan_C_ifft = NULL;
    fftwf_real *ab_copy = NULL;
    fftwf_real *c_copy = NULL;
    fftwf_complex *AB_copy = NULL;
    fftwf_complex *C = NULL;
    unsigned long long numel_ull = 0ull;
    unsigned long long numel_fft_ull = 0ull;
    unsigned numel = 0u;
    unsigned numel_fft = 0u;
    int Nbackwards[2];
    unsigned uError = ERROR_NONE;

    if (!pPlanStruct) return ERROR_NOMEM;

    /* Basic validation */
    if (N[0] <= 0 || N[1] <= 0) {
        fprintf(stderr, "xcorr_create_plan: invalid N: N[0]=%d N[1]=%d\n", N[0], N[1]);
        return ERROR_NOPLAN_BWD; /* use an error to indicate invalid sizes */
    }

    /* Compute sizes using 64-bit temporaries to detect overflow */
    numel_ull = (unsigned long long)N[0] * (unsigned long long)N[1];
    /* For r2c transform: output shape is (N[0], N[1]/2+1) */
    numel_fft_ull = (unsigned long long)N[0] * (unsigned long long)(N[1]/2 + 1);

    /* Sane upper bound check (prevent crazy values). Adjust as appropriate. */
    const unsigned long long MAX_PIXELS = 1ULL << 30; /* ~1 billion elements */
    if (numel_ull == 0 || numel_fft_ull == 0 || numel_ull > MAX_PIXELS || numel_fft_ull > MAX_PIXELS) {
        fprintf(stderr, "xcorr_create_plan: requested size too large or zero: N=(%d,%d), numel=%llu, numel_fft=%llu\n",
                N[0], N[1], numel_ull, numel_fft_ull);
        return ERROR_NOPLAN_BWD;
    }

    /* safe cast to unsigned (FFTW functions expect int/size_t) */
    numel = (unsigned)numel_ull;
    numel_fft = (unsigned)numel_fft_ull;

    /* Debug print to see exactly what sizes are requested */

    /* FFTW uses row-major order: first dim = rows (height), second dim = cols (width) */
    Nbackwards[0] = N[0];
    Nbackwards[1] = N[1];

    /* allocate aligned memory for plan scratch */
    ab_copy = (fftwf_real *)fftwf_alloc_real((size_t)numel * 2);
    AB_copy = (fftwf_complex *)fftwf_alloc_complex((size_t)numel_fft * 2);
    C       = (fftwf_complex *)fftwf_alloc_complex((size_t)numel_fft);
    c_copy  = (fftwf_real *)fftwf_alloc_real((size_t)numel);

    if (!ab_copy || !AB_copy || !C || !c_copy) {
        fprintf(stderr, "xcorr_create_plan: fftwf_alloc_* failed (memory). ab=%p AB=%p C=%p c=%p\n",
                (void*)ab_copy, (void*)AB_copy, (void*)C, (void*)c_copy);
        if (ab_copy) fftwf_free(ab_copy);
        if (AB_copy) fftwf_free(AB_copy);
        if (C) fftwf_free(C);
        if (c_copy) fftwf_free(c_copy);
        return ERROR_NOMEM;
    }

    /* Try MEASURE first, then ESTIMATE if MEASURE fails */
    int flags_measure = FFTW_MEASURE | FFTW_DESTROY_INPUT;
    int flags_estimate = FFTW_ESTIMATE | FFTW_DESTROY_INPUT;

    plan_AB_fft = fftwf_plan_many_dft_r2c(2, Nbackwards, 2,
                                         ab_copy, NULL,
                                         1, (int)numel,
                                         AB_copy, NULL,
                                         1, (int)numel_fft,
                                         flags_measure);

    plan_C_ifft = fftwf_plan_dft_c2r_2d(N[0], N[1], C, c_copy, flags_measure);

    if (!plan_AB_fft || !plan_C_ifft) {
        fprintf(stderr, "xcorr_create_plan: MEASURE plan failed for N=(%d,%d). Falling back to ESTIMATE\n", N[0], N[1]);
        if (plan_AB_fft) { fftwf_destroy_plan(plan_AB_fft); plan_AB_fft = NULL; }
        if (plan_C_ifft) { fftwf_destroy_plan(plan_C_ifft); plan_C_ifft = NULL; }

        plan_AB_fft = fftwf_plan_many_dft_r2c(2, Nbackwards, 2,
                                             ab_copy, NULL,
                                             1, (int)numel,
                                             AB_copy, NULL,
                                             1, (int)numel_fft,
                                             flags_estimate);

        plan_C_ifft = fftwf_plan_dft_c2r_2d(N[0], N[1], C, c_copy, flags_estimate);

        if (!plan_AB_fft || !plan_C_ifft) {
            fprintf(stderr, "xcorr_create_plan: ESTIMATE plan also failed for N=(%d,%d). plan_AB=%p plan_C=%p\n",
                    N[0], N[1], (void*)plan_AB_fft, (void*)plan_C_ifft);
            if (plan_AB_fft) { fftwf_destroy_plan(plan_AB_fft); plan_AB_fft = NULL; }
            if (plan_C_ifft) { fftwf_destroy_plan(plan_C_ifft); plan_C_ifft = NULL; }
            fftwf_free(AB_copy);
            fftwf_free(c_copy);
            fftwf_free(ab_copy);
            fftwf_free(C);
            if (!plan_C_ifft) {
                return ERROR_NOPLAN_BWD;
            }
            return ERROR_NOPLAN_FWD;
        }
    }

    /* success: populate plan struct */
    pPlanStruct->plan_AB_fft = plan_AB_fft;
    pPlanStruct->plan_C_ifft = plan_C_ifft;
    pPlanStruct->ab_copy = ab_copy;
    pPlanStruct->AB_copy = AB_copy;
    pPlanStruct->C = C;
    pPlanStruct->c_copy = c_copy;
    pPlanStruct->N[0] = N[0];
    pPlanStruct->N[1] = N[1];

    return ERROR_NONE;
}


/****************************************************
 * unsigned xcorr_destroy_plan(planstruct)
 *
 * destroy plans created by xcorr_create_plan
 * xcorr_destroy_plan is NOT thread safe
 * so call it from an OMP critical section
 */

unsigned xcorr_destroy_plan(sPlan *pPlanStruct)
{
	if(!pPlanStruct)
		return ERROR_NOMEM;

	/* destroy plans first */
	if(pPlanStruct->plan_AB_fft)	fftwf_destroy_plan(pPlanStruct->plan_AB_fft);
	if(pPlanStruct->plan_C_ifft)	fftwf_destroy_plan(pPlanStruct->plan_C_ifft);

	/* deallocate memory*/
	if(pPlanStruct->ab_copy) fftwf_free(pPlanStruct->ab_copy);
	if(pPlanStruct->AB_copy) fftwf_free(pPlanStruct->AB_copy);
	if(pPlanStruct->C)		 fftwf_free(pPlanStruct->C);
	if(pPlanStruct->c_copy)	 fftwf_free(pPlanStruct->c_copy);

	return ERROR_NONE;
}

/****************************************************
 * unsigned xcorr_preplanned(a, b, c, sPlan)
 * same as xcorr(a,b,c,sPlan->N), but with preplanned FFTs
 *
 * calculates c = fftshift( IFFT(FFT(a) .* FFT(b)') )
 * xcorr_preplanned is thread safe
 * HOWEVER, you cannot use plans generated by other threads
 *
 */

unsigned xcorr_preplanned(const float *a, const float *b, float *c, sPlan *pPlanStruct)
{
	fftwf_plan plan_AB_fft;
	fftwf_plan plan_C_ifft;
	fftwf_real *ab_copy;
	fftwf_real *c_copy;
	fftwf_complex *AB_copy;
	fftwf_complex *C;
	unsigned numel, numel_fft;
	int N[2];
	int jswap, j;
	float mul;

	if(!pPlanStruct)
		return ERROR_NOMEM;

	/* load plan from sPlanStruct */
	N[0]			= pPlanStruct->N[0];
	N[1]			= pPlanStruct->N[1];
	plan_AB_fft = pPlanStruct->plan_AB_fft;
	plan_C_ifft = pPlanStruct->plan_C_ifft;
	ab_copy		= pPlanStruct->ab_copy;
	AB_copy		= pPlanStruct->AB_copy;
	C				= pPlanStruct->C;
	c_copy		= pPlanStruct->c_copy;

	/****
	 * initialise variables to be used for later
	 */
	numel			= N[0] * N[1];
	numel_fft	= N[0] * (N[1]/2+1);  /* r2c output shape: (N[0], N[1]/2+1) */

	/****
	 * copy in a and b, should be done after planning stage as fftwf_plan can overwrite 
	 * input array
	 */
	memcpy(&ab_copy[0    ], a, numel*sizeof(float));
	memcpy(&ab_copy[numel], b, numel*sizeof(float));

	/**** 
	 * execute forward transform 
	 */
	fftwf_execute(plan_AB_fft);

	/****
	 * do C = A .* conj(B)
	 */
	multiply_conjugate((const fftwf_complex*)&AB_copy[0], (const fftwf_complex*)&AB_copy[numel_fft], C, numel_fft);

	/*** 
	 * execute backward transform and renormalise
	 */
	fftwf_execute(plan_C_ifft);

	/***
	 * Copy ifft result into c, after shifting data with fftshift
	 * For row-major arrays with shape [N[0], N[1]]:
	 * - N[0] = number of rows (height)
	 * - N[1] = number of columns (width)
	 * fftshift swaps quadrants: moves zero-frequency to center
	 */
	// Renormalise first
	mul = 1.0f / (float)numel;
	for(j=0; j<(int)numel; ++j)	{	c_copy[j] = c_copy[j] * mul;	}

	/* Perform fftshift for row-major data
	 * For each row, swap left/right halves
	 * Also swap top/bottom halves
	 */
	for(int row = 0; row < N[0]; ++row)
	{
		int row_swap = (row + N[0]/2) % N[0];
		/* Copy left half of swapped row to right half of output */
		memcpy(&c[SUB2IND_2D(row, N[1]/2, N[1])], 
		       &c_copy[SUB2IND_2D(row_swap, 0, N[1])], 
		       N[1]/2 * sizeof(float));
		/* Copy right half of swapped row to left half of output */
		memcpy(&c[SUB2IND_2D(row, 0, N[1])], 
		       &c_copy[SUB2IND_2D(row_swap, N[1]/2, N[1])], 
		       N[1]/2 * sizeof(float));
	}

	return ERROR_NONE;
}

/****************************************************
 * unsigned xcorr(a, b, c, N)
 *
 * calculates c = fftshift( IFFT(FFT(a) .* FFT(b)') )
 *
 * a, b, and c are 2D arrays of size N[0]xN[1] supplied in row-major (C-contiguous) format
 * where N[0] = number of rows (height), N[1] = number of columns (width)
 *
 * return value is the error code. return value of 0 means success.
 * NOTE: everything apart from fftwf_execute in fftwf_ library is NOT thread safe
 * so must encapsulate it in an omp critical section
 *
 */

unsigned xcorr(const float *a, const float *b, float *c, const int *N)
{
	sPlan spPlan;
	unsigned uError;

	/* create plan */
	#pragma omp critical
	{
		uError	= xcorr_create_plan(N, &spPlan);
	}
	if(uError != ERROR_NONE)
		return uError;

	/* cross-correlate with xcorr_preplanned */
	uError = xcorr_preplanned(a, b, c, &spPlan);

	/* destroy plan */	
	#pragma omp critical	
	{
		xcorr_destroy_plan(&spPlan);
	}

	return uError;
}


/****************************************************
 * void multiply_conjugate(const fftwf_complex *A, const fftwf_complex *B, fftwf_complex *C, int N)
 *
 * performs C = A .* conj(B)
 *
 * A, B and C are treated as 1xN vectors of fftwf_complex
 * Optimized with restrict pointers for better compiler optimization
 */

void multiply_conjugate(const fftwf_complex * restrict A, const fftwf_complex * restrict B, fftwf_complex * restrict C, int N)
{
	int i;
	
	/* Use restrict to help compiler optimize and enable SIMD vectorization */
	#pragma omp simd
	for(i=0; i<N; ++i)
	{
		const float Ar = A[i][0];
		const float Ai = A[i][1];
		const float Br = B[i][0];
		const float Bi = B[i][1];
		
		C[i][0] = Ar*Br + Ai*Bi;
		C[i][1] = Ai*Br - Ar*Bi;
	}

	return;
}
