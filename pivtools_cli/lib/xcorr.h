#ifndef XCORR_H
#define XCORR_H

#include <fftw3.h>

/**** defines ****/
#define fftwf_real						float

/**** data structures ****/
typedef struct _sPlan
{
	fftwf_plan plan_AB_fft;
	fftwf_plan plan_C_ifft;
	fftwf_real *ab_copy;
	fftwf_real *c_copy;
	fftwf_complex *AB_copy;
	fftwf_complex *C;
	int N[2];
} sPlan;

/**** functions ****/
unsigned convolve(const float *w1, const float *w2, float *conv, const int *N);
unsigned xcorr(const float *w1, const float *w2, float *corr, const int *N);

unsigned xcorr_create_plan(const int *N, sPlan *pPlanStruct);
unsigned xcorr_destroy_plan(sPlan *pPlanStruct);
unsigned xcorr_preplanned(const float *w1, const float *w2, float *corr, sPlan *pPlanStruct);

void multiply_conjugate(const fftwf_complex * restrict A, const fftwf_complex * restrict B, fftwf_complex * restrict C, int N);

#endif