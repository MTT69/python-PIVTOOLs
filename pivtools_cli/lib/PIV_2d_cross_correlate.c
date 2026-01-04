#include "PIV_2d_cross_correlate.h"
#include "common.h"
#include "xcorr.h"
#include "xcorr_cache.h"      /* FFTW wisdom caching */
#include "peak_locate_lm.h"   /* Fast LM solver instead of GSL */
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

unsigned char bulkxcorr2d(
const float *fImageA_stack, const float *fImageB_stack, const float *fMask,
const int *nImageSize, int N_images,
const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows,
float *fWindowWeightA, bool bEnsemble,
const float *fWindowWeightB, const int *nWindowSize, int nPeaks, int iPeakFinder,
float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy,
float *fCorrelPlane_Out)
{

float *fCorrelPlane;
float *fWindowA;
float *fWindowB;
float *fStd;
float *fPeakLoc;
//xcorr_plan sCCPlan;
sPlan sCCPlan;
float fMeanA, fMeanB, fEnergyA, fEnergyB, fEnergyNorm;
int idx, n, ii, jj, i, j, x, y;

//int i, j, ii, jj, n, idx, x, y;
int nWindowsTotal = nWindows[0] * nWindows[1];
int nPxPerWindow = nWindowSize[0] * nWindowSize[1];
unsigned uError = ERROR_NONE;

float *fCorrelWeight = (float*)malloc(nPxPerWindow * sizeof(float));
if (!fCorrelWeight) return ERROR_NOMEM;

/* Precompute correlation plane weighting */
uError = convolve(fWindowWeightB, fWindowWeightB, fCorrelWeight, nWindowSize);
if (uError) { free(fCorrelWeight); return uError; }
for (i = 0; i < nPxPerWindow; ++i) fCorrelWeight[i] = nPxPerWindow / fCorrelWeight[i];

/* Initialize FFTW threads (thread-safe, only runs once) */
fftw_library_init();

/* Load FFTW wisdom */
char wisdom_path[512];
xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
xcorr_cache_init(wisdom_path);

int total_windows = N_images * nWindowsTotal;

/* Flattened parallel loop over all windows in all images */
#pragma omp parallel \
    default(none) \
    shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
           fWinCtrsX, fWinCtrsY, nWindows, bEnsemble, fCorrelWeight, fWindowWeightA, fWindowWeightB, nWindowSize, \
           nPeaks, iPeakFinder, fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out, nPxPerWindow, nWindowsTotal,total_windows) \
    private(idx, n, ii, jj, i, j, x, y, fCorrelPlane, fWindowA, fWindowB, fStd, fPeakLoc, sCCPlan, \
            fMeanA, fMeanB, fEnergyA, fEnergyB, fEnergyNorm) \
    reduction(|:uError)
{
    fCorrelPlane = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
    fWindowA = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
    fWindowB = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
    fStd = (float*)malloc(3 * nPeaks * sizeof(float));
    fPeakLoc = (float*)malloc(3 * nPeaks * sizeof(float));

    if(!fWindowA || !fWindowB || !fCorrelPlane || !fPeakLoc || !fStd)
    { uError = ERROR_NOMEM; goto thread_cleanup; }

    memset(&sCCPlan, 0, sizeof(sCCPlan));
    #pragma omp critical
    uError = xcorr_create_plan(nWindowSize, &sCCPlan);
    if(uError) goto thread_cleanup;

    #pragma omp for schedule(static)
    for(idx = 0; idx < total_windows; ++idx)
    {
        n = idx / nWindowsTotal;        // image index
        int iWindowIdx = idx % nWindowsTotal;  // window index
        ii = iWindowIdx % nWindows[1];
        jj = iWindowIdx / nWindows[1];

        const float *fImageA = &fImageA_stack[n * nImageSize[0] * nImageSize[1]];
        const float *fImageB = &fImageB_stack[n * nImageSize[0] * nImageSize[1]];

        int mask_idx = jj * nWindows[1] + ii;
        if (mask_idx < 0 || mask_idx >= nWindowsTotal) continue;
        if (fMask[mask_idx] == 1) continue;

        int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0)/2.0 + 0.5);
        int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0)/2.0 + 0.5);
        if(row_min < 0 || col_min < 0 || row_min + nWindowSize[0] > nImageSize[0] || col_min + nWindowSize[1] > nImageSize[1]) continue;

        for(i = 0; i < nWindowSize[0]; ++i)
            for(j = 0; j < nWindowSize[1]; ++j)
            {
                int img_idx = (row_min+i)*nImageSize[1] + (col_min+j);
                int win_idx = i*nWindowSize[1] + j;
                fWindowA[win_idx] = fImageA[img_idx];
                fWindowB[win_idx] = fImageB[img_idx];
            }

        fMeanA = fMeanB = 0.0f;
        #pragma omp simd reduction(+:fMeanA,fMeanB)
        for(i = 0; i < nPxPerWindow; ++i)
        {
            fWindowA[i] *= fWindowWeightA[i];
            fWindowB[i] *= fWindowWeightB[i];
            fMeanA += fWindowA[i];
            fMeanB += fWindowB[i];
        }
        fMeanA /= nPxPerWindow;
        fMeanB /= nPxPerWindow;

        fEnergyA = fEnergyB = 0.0f;
        if(!bEnsemble)
        {
            #pragma omp simd reduction(+:fEnergyA,fEnergyB)
            for(i = 0; i < nPxPerWindow; ++i)
            {
                fWindowA[i] -= fMeanA;
                fWindowB[i] -= fMeanB;
                fEnergyA += fWindowA[i]*fWindowA[i];
                fEnergyB += fWindowB[i]*fWindowB[i];
            }
        }
        else
        {
            #pragma omp simd reduction(+:fEnergyA,fEnergyB)
            for(i = 0; i < nPxPerWindow; ++i)
            {
                fEnergyA += fWindowA[i]*fWindowA[i];
                fEnergyB += fWindowB[i]*fWindowB[i];
            }
        }
        fEnergyNorm = 1.0f / sqrtf(fEnergyA * fEnergyB);

        xcorr_preplanned(fWindowB, fWindowA, fCorrelPlane, &sCCPlan);

        if(!bEnsemble)
        {
            #pragma omp simd
            for(i = 0; i < nPxPerWindow; ++i)
                fCorrelPlane[i] *= fCorrelWeight[i];
        }

        /* Copy correlation plane to output - only needed for ensemble mode */
        if (bEnsemble && fCorrelPlane_Out != NULL) {
            memcpy(&fCorrelPlane_Out[idx * nPxPerWindow], fCorrelPlane, nPxPerWindow * sizeof(float));
        }

        /* Peak finder */
        if(!bEnsemble)
        {
            lsqpeaklocate_lm(fCorrelPlane, nWindowSize, fPeakLoc, nPeaks, iPeakFinder, fStd);
            for(i = 0; i < nPeaks; ++i)
            {
                int out_idx = n * nPeaks * nWindowsTotal + i * nWindowsTotal + iWindowIdx;
                float peak_row = fPeakLoc[0*nPeaks + i];
                float peak_col = fPeakLoc[1*nPeaks + i];
                float peak_mag = fPeakLoc[2*nPeaks + i];

                fPkLocX[out_idx] = peak_col - nWindowSize[1]/2.0f;
                fPkLocY[out_idx] = peak_row - nWindowSize[0]/2.0f;
                fSx[out_idx] = fStd[0*nPeaks + i];
                fSy[out_idx] = fStd[1*nPeaks + i];
                fSxy[out_idx] = fStd[2*nPeaks + i];

                int pk_row = fmin(fmax(0, (int)peak_row), nWindowSize[0]-1);
                int pk_col = fmin(fmax(0, (int)peak_col), nWindowSize[1]-1);
                fPkHeight[out_idx] = peak_mag * fEnergyNorm / fCorrelWeight[pk_row*nWindowSize[1] + pk_col];
            }
        }
    }

thread_cleanup:
if(fWindowA) fftwf_free(fWindowA);
if(fWindowB) fftwf_free(fWindowB);
if(fCorrelPlane) fftwf_free(fCorrelPlane);
if(fStd) free(fStd);
if(fPeakLoc) free(fPeakLoc);
#pragma omp critical
xcorr_destroy_plan(&sCCPlan);
}

xcorr_cache_save_wisdom(wisdom_path);
free(fCorrelWeight);
return uError;

}

/**
 * Ensemble-optimized cross-correlation with internal accumulation.
 * Option C: Parallel over windows, sequential over images.
 */
unsigned char bulkxcorr2d_accumulate(
    const float *fImageA_stack, const float *fImageB_stack, const float *fMask,
    const int *nImageSize, int N_images,
    const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows,
    const float *fWindowWeightA, const float *fWindowWeightB,
    const int *nWindowSize,
    float *fCorrelPlane_Sum)
{
    int nWindowsTotal = nWindows[0] * nWindows[1];
    int nPxPerWindow = nWindowSize[0] * nWindowSize[1];
    int nImagePixels = nImageSize[0] * nImageSize[1];
    unsigned uError = ERROR_NONE;
    int i, j, n, iWindowIdx, ii, jj;

    /* Initialize FFTW threads */
    fftw_library_init();

    /* Load FFTW wisdom */
    char wisdom_path[512];
    xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
    xcorr_cache_init(wisdom_path);

    /* Initialize output to zero */
    memset(fCorrelPlane_Sum, 0, nWindowsTotal * nPxPerWindow * sizeof(float));

    /* OPTION C: Parallel over windows, sequential over images */
    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, fWindowWeightA, fWindowWeightB, \
               nWindowSize, fCorrelPlane_Sum, nPxPerWindow, nWindowsTotal, nImagePixels) \
        private(i, j, n, iWindowIdx, ii, jj) \
        reduction(|:uError)
    {
        /* Thread-local workspace - small! Only one correlation plane */
        float *fCorrelPlane = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        float *fWindowA = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        float *fWindowB = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        sPlan sCCPlan;

        if (!fCorrelPlane || !fWindowA || !fWindowB) {
            uError = ERROR_NOMEM;
            goto thread_cleanup;
        }

        memset(&sCCPlan, 0, sizeof(sCCPlan));
        #pragma omp critical
        uError = xcorr_create_plan(nWindowSize, &sCCPlan);
        if (uError) goto thread_cleanup;

        /* Outer loop: parallel over windows */
        #pragma omp for schedule(static)
        for (iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
        {
            ii = iWindowIdx % nWindows[1];  /* column */
            jj = iWindowIdx / nWindows[1];  /* row */

            /* Skip masked windows */
            if (fMask[iWindowIdx] == 1) continue;

            /* Compute window bounds */
            int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0f)/2.0f + 0.5f);
            int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0f)/2.0f + 0.5f);

            /* Bounds check */
            if (row_min < 0 || col_min < 0 ||
                row_min + nWindowSize[0] > nImageSize[0] ||
                col_min + nWindowSize[1] > nImageSize[1]) continue;

            /* Pointer to this window's output (this thread owns it!) */
            float *out_ptr = &fCorrelPlane_Sum[iWindowIdx * nPxPerWindow];

            /* Inner loop: sequential over images, accumulating */
            for (n = 0; n < N_images; ++n)
            {
                const float *fImageA = &fImageA_stack[n * nImagePixels];
                const float *fImageB = &fImageB_stack[n * nImagePixels];

                /* Extract windows and apply weights */
                float fMeanA = 0.0f, fMeanB = 0.0f;
                for (i = 0; i < nWindowSize[0]; ++i) {
                    for (j = 0; j < nWindowSize[1]; ++j) {
                        int img_idx = (row_min + i) * nImageSize[1] + (col_min + j);
                        int win_idx = i * nWindowSize[1] + j;
                        fWindowA[win_idx] = fImageA[img_idx] * fWindowWeightA[win_idx];
                        fWindowB[win_idx] = fImageB[img_idx] * fWindowWeightB[win_idx];
                        fMeanA += fWindowA[win_idx];
                        fMeanB += fWindowB[win_idx];
                    }
                }
                fMeanA /= nPxPerWindow;
                fMeanB /= nPxPerWindow;

                /* For ensemble: compute energy but DON'T subtract mean */
                /* (mean subtraction happens via background correlation in Python) */
                float fEnergyA = 0.0f, fEnergyB = 0.0f;
                for (i = 0; i < nPxPerWindow; ++i) {
                    fEnergyA += fWindowA[i] * fWindowA[i];
                    fEnergyB += fWindowB[i] * fWindowB[i];
                }

                /* Cross-correlation via FFT */
                xcorr_preplanned(fWindowB, fWindowA, fCorrelPlane, &sCCPlan);

                /* Accumulate to output (no atomics - this thread owns this window!) */
                for (i = 0; i < nPxPerWindow; ++i) {
                    out_ptr[i] += fCorrelPlane[i];
                }
            }
        }

    thread_cleanup:
        if (fWindowA) fftwf_free(fWindowA);
        if (fWindowB) fftwf_free(fWindowB);
        if (fCorrelPlane) fftwf_free(fCorrelPlane);
        #pragma omp critical
        xcorr_destroy_plan(&sCCPlan);
    }

    xcorr_cache_save_wisdom(wisdom_path);
    return uError;
}

/* fminvec, find minimum element in vector */
float fminvec(const float *fVec, int n)
{
	int i;
	float ret;

	ret = fVec[0];
	for(i = 1; i < n; ++i)
		ret = MIN(ret, fVec[i]);

	return ret;
}

/* fmaxvec, find maximum element in vector */
float fmaxvec(const float *fVec, int n)
{
	int i;
	float ret;

	ret = fVec[0];
	for(i = 1; i < n; ++i)
		ret = MAX(ret, fVec[i]);

	return ret;
}
