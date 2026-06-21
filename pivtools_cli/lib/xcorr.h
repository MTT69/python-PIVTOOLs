#ifndef XCORR_H
#define XCORR_H

#include "codelet_fft.h"

/**** data structures ****/
/* Cross-correlation plan: a thin wrapper over a codelet_plan (the permissive,
 * FFTW-free transform engine). Internal to the library -- it never crosses the
 * Python/ctypes boundary. NOT thread-safe to share; create one per OMP thread,
 * exactly as the old FFTW-backed sPlan was used. */
typedef struct _sPlan
{
	codelet_plan *cp;
	int N[2];          /* [rows (H), cols (W)] */
} sPlan;

/**** functions ****/
unsigned convolve(const float *w1, const float *w2, float *conv, const int *N);
unsigned xcorr(const float *w1, const float *w2, float *corr, const int *N);

unsigned xcorr_create_plan(const int *N, sPlan *pPlanStruct);
unsigned xcorr_destroy_plan(sPlan *pPlanStruct);
unsigned xcorr_preplanned(const float *w1, const float *w2, float *corr, sPlan *pPlanStruct);

#endif
