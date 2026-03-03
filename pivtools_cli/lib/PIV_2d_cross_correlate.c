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
 *
 * If nFitWindowSize is not NULL, extracts only the central region of each
 * correlation plane for accumulation. This reduces memory usage and fitting time.
 */
unsigned char bulkxcorr2d_accumulate(
    const float *fImageA_stack, const float *fImageB_stack, const float *fMask,
    const int *nImageSize, int N_images,
    const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows,
    const float *fWindowWeightA, const float *fWindowWeightB,
    const int *nWindowSize,
    const int *nFitWindowSize,  /* NEW: output size, NULL = use full nWindowSize */
    float *fCorrelPlane_Sum)
{
    int nWindowsTotal = nWindows[0] * nWindows[1];
    int nPxPerWindow = nWindowSize[0] * nWindowSize[1];  /* FFT computation size */
    int nImagePixels = nImageSize[0] * nImageSize[1];
    unsigned uError = ERROR_NONE;
    int i, j, n, iWindowIdx, ii, jj;

    /* Determine output dimensions */
    int out_h = nFitWindowSize ? nFitWindowSize[0] : nWindowSize[0];
    int out_w = nFitWindowSize ? nFitWindowSize[1] : nWindowSize[1];
    int nPxPerOutput = out_h * out_w;

    /* Extraction offsets (centered) */
    int start_y = (nWindowSize[0] - out_h) / 2;
    int start_x = (nWindowSize[1] - out_w) / 2;

    /* Initialize FFTW threads */
    fftw_library_init();

    /* Load FFTW wisdom */
    char wisdom_path[512];
    xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
    xcorr_cache_init(wisdom_path);

    /* Initialize output to zero - NOTE: uses output size, not computation size! */
    memset(fCorrelPlane_Sum, 0, nWindowsTotal * nPxPerOutput * sizeof(float));

    /* OPTION C: Parallel over windows, sequential over images */
    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, fWindowWeightA, fWindowWeightB, \
               nWindowSize, fCorrelPlane_Sum, nPxPerWindow, nWindowsTotal, nImagePixels, \
               out_h, out_w, nPxPerOutput, start_y, start_x) \
        private(i, j, n, iWindowIdx, ii, jj) \
        reduction(|:uError)
    {
        /* Thread-local workspace - uses computation size for FFT */
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
            /* NOTE: uses nPxPerOutput for offset, not nPxPerWindow */
            float *out_ptr = &fCorrelPlane_Sum[iWindowIdx * nPxPerOutput];

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

                /* Accumulate only the central region to output */
                for (i = 0; i < out_h; ++i) {
                    for (j = 0; j < out_w; ++j) {
                        int src_idx = (start_y + i) * nWindowSize[1] + (start_x + j);
                        int dst_idx = i * out_w + j;
                        out_ptr[dst_idx] += fCorrelPlane[src_idx];
                    }
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

/**
 * Fused triple cross-correlation with internal accumulation.
 *
 * Computes AB (cross-correlation), AA and BB (auto-correlations) in a
 * single pass per window per image.  FFT(A) and FFT(B) are computed
 * ONCE and reused for all three products, eliminating 4 of the 6
 * forward FFTs that the 3× bulkxcorr2d_accumulate path requires.
 *
 * For AB:   IFFT( FFT(weightA·A) · conj(FFT(weightB·B)) )
 * For AA:   IFFT( |FFT(autoW·A)|² )
 * For BB:   IFFT( |FFT(autoW·B)|² )
 *
 * The auto-correlation uses autoWeightA/autoWeightB (typically the
 * same full Hanning window for both) so the particle sigma matches.
 *
 * Output layout per plane: nWindowsTotal × nPxPerOutput  (row-major).
 * Three separate output buffers: fCorrAB_Sum, fCorrAA_Sum, fCorrBB_Sum.
 */
unsigned char bulkxcorr2d_accumulate_triple(
    const float *fImageA_stack,
    const float *fImageB_stack,
    const float *fMask,
    const int   *nImageSize,
    int          N_images,
    const float *fWinCtrsX,
    const float *fWinCtrsY,
    const int   *nWindows,
    /* AB weights (may be asymmetric for single mode) */
    const float *fWindowWeightA_AB,
    const float *fWindowWeightB_AB,
    /* AA/BB weights (always full/symmetric) */
    const float *fAutoWeightA,
    const float *fAutoWeightB,
    const int   *nWindowSize,
    const int   *nFitWindowSize,
    float       *fCorrAB_Sum,
    float       *fCorrAA_Sum,
    float       *fCorrBB_Sum)
{
    int nWindowsTotal = nWindows[0] * nWindows[1];
    int nPxPerWindow  = nWindowSize[0] * nWindowSize[1];
    int nImagePixels  = nImageSize[0] * nImageSize[1];
    unsigned uError   = ERROR_NONE;
    int i, j, n, iWindowIdx, ii, jj;

    /* Output dimensions (central extraction if nFitWindowSize != NULL) */
    int out_h = nFitWindowSize ? nFitWindowSize[0] : nWindowSize[0];
    int out_w = nFitWindowSize ? nFitWindowSize[1] : nWindowSize[1];
    int nPxPerOutput = out_h * out_w;
    int start_y = (nWindowSize[0] - out_h) / 2;
    int start_x = (nWindowSize[1] - out_w) / 2;

    /* Frequency-domain size for r2c transform */
    int numel     = nWindowSize[0] * nWindowSize[1];
    int numel_fft = nWindowSize[0] * (nWindowSize[1] / 2 + 1);

    fftw_library_init();

    char wisdom_path[512];
    xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
    xcorr_cache_init(wisdom_path);

    /* Zero all three output buffers */
    memset(fCorrAB_Sum, 0, (size_t)nWindowsTotal * nPxPerOutput * sizeof(float));
    memset(fCorrAA_Sum, 0, (size_t)nWindowsTotal * nPxPerOutput * sizeof(float));
    memset(fCorrBB_Sum, 0, (size_t)nWindowsTotal * nPxPerOutput * sizeof(float));

    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, \
               fWindowWeightA_AB, fWindowWeightB_AB, fAutoWeightA, fAutoWeightB, \
               nWindowSize, fCorrAB_Sum, fCorrAA_Sum, fCorrBB_Sum, \
               nPxPerWindow, nWindowsTotal, nImagePixels, numel, numel_fft, \
               out_h, out_w, nPxPerOutput, start_y, start_x) \
        private(i, j, n, iWindowIdx, ii, jj) \
        reduction(|:uError)
    {
        /* ---- thread-local workspace ------------------------------------ */
        /* Raw pixel windows (before weighting) */
        float *fRawA = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        float *fRawB = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));

        /* Cross-correlation plan (xcorr_preplanned pattern):
         * ab_copy holds [A|B] interleaved for batched r2c,
         * AB_copy holds [FFT(A)|FFT(B)], C is scratch for IFFT. */
        sPlan sCCPlan;

        /* Extra IFFT scratch: we need two more IFFTs (AA, BB)
         * but sCCPlan only has one c_copy/C pair.  Allocate extras. */
        fftwf_complex *C_extra1  = NULL;
        fftwf_complex *C_extra2  = NULL;
        float         *c_extra1  = NULL;
        float         *c_extra2  = NULL;
        float         *fCorrelAB = NULL;
        float         *fCorrelAA = NULL;
        float         *fCorrelBB = NULL;

        if (!fRawA || !fRawB) { uError = ERROR_NOMEM; goto triple_cleanup; }

        memset(&sCCPlan, 0, sizeof(sCCPlan));
        #pragma omp critical
        uError = xcorr_create_plan(nWindowSize, &sCCPlan);
        if (uError) goto triple_cleanup;

        /* Allocate extra IFFT workspace (same size as plan's C / c_copy) */
        C_extra1  = (fftwf_complex*)fftwf_alloc_complex(numel_fft);
        C_extra2  = (fftwf_complex*)fftwf_alloc_complex(numel_fft);
        c_extra1  = (float*)fftwf_alloc_real(numel);
        c_extra2  = (float*)fftwf_alloc_real(numel);
        fCorrelAB = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrelAA = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrelBB = (float*)fftwf_malloc(nPxPerWindow * sizeof(float));

        if (!C_extra1 || !C_extra2 || !c_extra1 || !c_extra2 ||
            !fCorrelAB || !fCorrelAA || !fCorrelBB) {
            uError = ERROR_NOMEM; goto triple_cleanup;
        }

        /* ---- parallel over windows ------------------------------------ */
        #pragma omp for schedule(static)
        for (iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
        {
            ii = iWindowIdx % nWindows[1];
            jj = iWindowIdx / nWindows[1];
            if (fMask[iWindowIdx] == 1) continue;

            int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0f)/2.0f + 0.5f);
            int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0f)/2.0f + 0.5f);
            if (row_min < 0 || col_min < 0 ||
                row_min + nWindowSize[0] > nImageSize[0] ||
                col_min + nWindowSize[1] > nImageSize[1]) continue;

            float *outAB = &fCorrAB_Sum[iWindowIdx * nPxPerOutput];
            float *outAA = &fCorrAA_Sum[iWindowIdx * nPxPerOutput];
            float *outBB = &fCorrBB_Sum[iWindowIdx * nPxPerOutput];

            /* ---- sequential over images ------------------------------- */
            for (n = 0; n < N_images; ++n)
            {
                const float *fImageA = &fImageA_stack[n * nImagePixels];
                const float *fImageB = &fImageB_stack[n * nImagePixels];

                /* 1. Extract raw pixels ONCE (no weighting yet) */
                for (i = 0; i < nWindowSize[0]; ++i) {
                    for (j = 0; j < nWindowSize[1]; ++j) {
                        int img_idx = (row_min + i) * nImageSize[1] + (col_min + j);
                        int win_idx = i * nWindowSize[1] + j;
                        fRawA[win_idx] = fImageA[img_idx];
                        fRawB[win_idx] = fImageB[img_idx];
                    }
                }

                /* ============================================================
                 * 2. CROSS-CORRELATION AB  (using AB weights)
                 *    Apply AB weights → FFT both → conjugate multiply → IFFT
                 *    This reuses the plan's batched r2c + IFFT path.
                 * ============================================================ */
                {
                    /* Apply AB weights into plan's ab_copy buffer */
                    float *ab = sCCPlan.ab_copy;
                    for (i = 0; i < numel; ++i) {
                        ab[i]         = fRawB[i] * fWindowWeightB_AB[i]; /* slot 0 = B_AB */
                        ab[numel + i] = fRawA[i] * fWindowWeightA_AB[i]; /* slot 1 = A_AB */
                    }

                    /* Batched forward FFT: FFT(B_AB) and FFT(A_AB) */
                    fftwf_execute(sCCPlan.plan_AB_fft);

                    /* C_AB = FFT(B_AB) .* conj(FFT(A_AB)) */
                    multiply_conjugate(
                        (const fftwf_complex*)&sCCPlan.AB_copy[0],
                        (const fftwf_complex*)&sCCPlan.AB_copy[numel_fft],
                        sCCPlan.C,
                        numel_fft);

                    /* Inverse FFT → fftshift → fCorrelAB */
                    fftwf_execute(sCCPlan.plan_C_ifft);
                    {
                        float mul = 1.0f / (float)numel;
                        float *cc = sCCPlan.c_copy;
                        for (i = 0; i < numel; ++i) cc[i] *= mul;
                        for (int row = 0; row < nWindowSize[0]; ++row) {
                            int row_swap = (row + nWindowSize[0]/2) % nWindowSize[0];
                            memcpy(&fCorrelAB[row * nWindowSize[1] + nWindowSize[1]/2],
                                   &cc[row_swap * nWindowSize[1]],
                                   (nWindowSize[1]/2) * sizeof(float));
                            memcpy(&fCorrelAB[row * nWindowSize[1]],
                                   &cc[row_swap * nWindowSize[1] + nWindowSize[1]/2],
                                   (nWindowSize[1]/2) * sizeof(float));
                        }
                    }
                }

                /* ============================================================
                 * 3. AUTO-CORRELATION AA  (using auto weights)
                 *    Apply autoWeightA → FFT → |FFT|² → IFFT
                 * ============================================================ */
                {
                    float *ab = sCCPlan.ab_copy;
                    /* We only need one transform for AA, but plan_AB_fft is
                     * batched (howmany=2). Put A in BOTH slots to keep it valid.
                     * Alternatively: put A in slot 0, slot 1 is ignored for the
                     * autocorrelation since we use |FFT(A)|².
                     * Using both slots: FFT(A_auto) ends up in AB_copy[0..numel_fft-1]. */
                    for (i = 0; i < numel; ++i) {
                        float wa = fRawA[i] * fAutoWeightA[i];
                        ab[i]         = wa;  /* slot 0 */
                        ab[numel + i] = wa;  /* slot 1 (same data, auto-corr) */
                    }
                    fftwf_execute(sCCPlan.plan_AB_fft);

                    /* |FFT(A)|²: C_AA[k] = re² + im² (purely real spectrum) */
                    const fftwf_complex *FA = (const fftwf_complex*)&sCCPlan.AB_copy[0];
                    for (i = 0; i < numel_fft; ++i) {
                        C_extra1[i][0] = FA[i][0]*FA[i][0] + FA[i][1]*FA[i][1];
                        C_extra1[i][1] = 0.0f;
                    }

                    /* IFFT(|FFT(A)|²) using plan's IFFT (c2r).
                     * We can't use sCCPlan.plan_C_ifft directly with C_extra1 because
                     * FFTW c2r plans are tied to their (C, c_copy) buffers.
                     * Instead: copy into plan's C buffer, execute, read from c_copy. */
                    memcpy(sCCPlan.C, C_extra1, numel_fft * sizeof(fftwf_complex));
                    fftwf_execute(sCCPlan.plan_C_ifft);
                    {
                        float mul = 1.0f / (float)numel;
                        float *cc = sCCPlan.c_copy;
                        for (i = 0; i < numel; ++i) cc[i] *= mul;
                        for (int row = 0; row < nWindowSize[0]; ++row) {
                            int row_swap = (row + nWindowSize[0]/2) % nWindowSize[0];
                            memcpy(&fCorrelAA[row * nWindowSize[1] + nWindowSize[1]/2],
                                   &cc[row_swap * nWindowSize[1]],
                                   (nWindowSize[1]/2) * sizeof(float));
                            memcpy(&fCorrelAA[row * nWindowSize[1]],
                                   &cc[row_swap * nWindowSize[1] + nWindowSize[1]/2],
                                   (nWindowSize[1]/2) * sizeof(float));
                        }
                    }
                }

                /* ============================================================
                 * 4. AUTO-CORRELATION BB  (using auto weights)
                 * ============================================================ */
                {
                    float *ab = sCCPlan.ab_copy;
                    for (i = 0; i < numel; ++i) {
                        float wb = fRawB[i] * fAutoWeightB[i];
                        ab[i]         = wb;
                        ab[numel + i] = wb;
                    }
                    fftwf_execute(sCCPlan.plan_AB_fft);

                    const fftwf_complex *FB = (const fftwf_complex*)&sCCPlan.AB_copy[0];
                    for (i = 0; i < numel_fft; ++i) {
                        C_extra2[i][0] = FB[i][0]*FB[i][0] + FB[i][1]*FB[i][1];
                        C_extra2[i][1] = 0.0f;
                    }

                    memcpy(sCCPlan.C, C_extra2, numel_fft * sizeof(fftwf_complex));
                    fftwf_execute(sCCPlan.plan_C_ifft);
                    {
                        float mul = 1.0f / (float)numel;
                        float *cc = sCCPlan.c_copy;
                        for (i = 0; i < numel; ++i) cc[i] *= mul;
                        for (int row = 0; row < nWindowSize[0]; ++row) {
                            int row_swap = (row + nWindowSize[0]/2) % nWindowSize[0];
                            memcpy(&fCorrelBB[row * nWindowSize[1] + nWindowSize[1]/2],
                                   &cc[row_swap * nWindowSize[1]],
                                   (nWindowSize[1]/2) * sizeof(float));
                            memcpy(&fCorrelBB[row * nWindowSize[1]],
                                   &cc[row_swap * nWindowSize[1] + nWindowSize[1]/2],
                                   (nWindowSize[1]/2) * sizeof(float));
                        }
                    }
                }

                /* ============================================================
                 * 5. Accumulate all three to output (central extraction)
                 * ============================================================ */
                for (i = 0; i < out_h; ++i) {
                    for (j = 0; j < out_w; ++j) {
                        int src_idx = (start_y + i) * nWindowSize[1] + (start_x + j);
                        int dst_idx = i * out_w + j;
                        outAB[dst_idx] += fCorrelAB[src_idx];
                        outAA[dst_idx] += fCorrelAA[src_idx];
                        outBB[dst_idx] += fCorrelBB[src_idx];
                    }
                }
            } /* end image loop */
        } /* end window loop */

    triple_cleanup:
        if (fRawA)     fftwf_free(fRawA);
        if (fRawB)     fftwf_free(fRawB);
        if (fCorrelAB) fftwf_free(fCorrelAB);
        if (fCorrelAA) fftwf_free(fCorrelAA);
        if (fCorrelBB) fftwf_free(fCorrelBB);
        if (C_extra1)  fftwf_free(C_extra1);
        if (C_extra2)  fftwf_free(C_extra2);
        if (c_extra1)  fftwf_free(c_extra1);
        if (c_extra2)  fftwf_free(c_extra2);
        #pragma omp critical
        xcorr_destroy_plan(&sCCPlan);
    } /* end omp parallel */

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
