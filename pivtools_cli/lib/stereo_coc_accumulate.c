/*
 * stereo_coc_accumulate.c — Fused dual-camera cross-correlation + CoC accumulation.
 *
 * Extends the bulkxcorr2d_accumulate_triple pattern to handle two cameras
 * simultaneously, plus per-frame Correlation-of-Correlations (CoC).
 *
 * The CoC uses the SAME circular FFT correlation as per-camera (no zero-padding).
 * AB planes are mean-subtracted before CoC to remove the per-frame DC
 * pedestal (~10^21) that would cause float32 catastrophic cancellation.
 *
 * Build (macOS):
 *   gcc-15 -O3 -fPIC -fopenmp -DFFTW_THREADS -shared \
 *     stereo_coc_accumulate.c xcorr.c xcorr_cache.c \
 *     -o libstereo_coc.so -lfftw3f -lm
 */

#include "stereo_coc_accumulate.h"
#include "common.h"
#include "xcorr.h"
#include "xcorr_cache.h"

#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>


/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper: in-place fftshift/ifftshift (equivalent for even-sized arrays)     */
/* Swaps quadrants: top-left ↔ bottom-right, top-right ↔ bottom-left        */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void fftshift_inplace(float *data, int h, int w) {
    int hh = h / 2;
    int hw = w / 2;
    for (int i = 0; i < hh; ++i) {
        for (int j = 0; j < hw; ++j) {
            /* Swap top-left ↔ bottom-right */
            int tl = i * w + j;
            int br = (i + hh) * w + (j + hw);
            float tmp = data[tl]; data[tl] = data[br]; data[br] = tmp;
            /* Swap top-right ↔ bottom-left */
            int tr = i * w + (j + hw);
            int bl = (i + hh) * w + j;
            tmp = data[tr]; data[tr] = data[bl]; data[bl] = tmp;
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper: extract sub-image                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void extract_raw_subimage(
    const float *image,
    int row_min, int col_min,
    const int *nWindowSize, const int *nImageSize,
    float *raw_out)
{
    for (int i = 0; i < nWindowSize[0]; ++i) {
        int src_row = (row_min + i) * nImageSize[1];
        int dst_row = i * nWindowSize[1];
        for (int j = 0; j < nWindowSize[1]; ++j) {
            raw_out[dst_row + j] = image[src_row + col_min + j];
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper: compute AB cross-correlation using a pre-planned FFT                */
/*   Result written to fCorrelOut (full window size, fftshifted)               */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void compute_AB_xcorr(
    const float *fRawA, const float *fRawB,
    const float *weightA, const float *weightB,
    int numel, int numel_fft,
    const int *nWindowSize,
    sPlan *plan,
    float *fCorrelOut)
{
    float *ab = plan->ab_copy;

    for (int i = 0; i < numel; ++i) {
        ab[i]         = fRawB[i] * weightB[i];
        ab[numel + i] = fRawA[i] * weightA[i];
    }

    fftwf_execute(plan->plan_AB_fft);

    multiply_conjugate(
        (const fftwf_complex *)&plan->AB_copy[0],
        (const fftwf_complex *)&plan->AB_copy[numel_fft],
        plan->C,
        numel_fft);

    fftwf_execute(plan->plan_C_ifft);
    {
        float mul = 1.0f / (float)numel;
        float *cc = plan->c_copy;
        for (int i = 0; i < numel; ++i) cc[i] *= mul;

        int half_h = nWindowSize[0] / 2;
        int half_w = nWindowSize[1] / 2;
        for (int row = 0; row < nWindowSize[0]; ++row) {
            int row_swap = (row + half_h) % nWindowSize[0];
            memcpy(&fCorrelOut[row * nWindowSize[1] + half_w],
                   &cc[row_swap * nWindowSize[1]],
                   half_w * sizeof(float));
            memcpy(&fCorrelOut[row * nWindowSize[1]],
                   &cc[row_swap * nWindowSize[1] + half_w],
                   half_w * sizeof(float));
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper: compute auto-correlation                                            */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void compute_auto_xcorr(
    const float *fRaw, const float *weight,
    int numel, int numel_fft,
    const int *nWindowSize,
    sPlan *plan,
    fftwf_complex *C_extra,
    float *fCorrelOut)
{
    float *ab = plan->ab_copy;

    for (int i = 0; i < numel; ++i) {
        float w = fRaw[i] * weight[i];
        ab[i]         = w;
        ab[numel + i] = w;
    }

    fftwf_execute(plan->plan_AB_fft);

    const fftwf_complex *FA = (const fftwf_complex *)&plan->AB_copy[0];
    for (int i = 0; i < numel_fft; ++i) {
        C_extra[i][0] = FA[i][0] * FA[i][0] + FA[i][1] * FA[i][1];
        C_extra[i][1] = 0.0f;
    }

    memcpy(plan->C, C_extra, numel_fft * sizeof(fftwf_complex));
    fftwf_execute(plan->plan_C_ifft);
    {
        float mul = 1.0f / (float)numel;
        float *cc = plan->c_copy;
        for (int i = 0; i < numel; ++i) cc[i] *= mul;

        int half_h = nWindowSize[0] / 2;
        int half_w = nWindowSize[1] / 2;
        for (int row = 0; row < nWindowSize[0]; ++row) {
            int row_swap = (row + half_h) % nWindowSize[0];
            memcpy(&fCorrelOut[row * nWindowSize[1] + half_w],
                   &cc[row_swap * nWindowSize[1]],
                   half_w * sizeof(float));
            memcpy(&fCorrelOut[row * nWindowSize[1]],
                   &cc[row_swap * nWindowSize[1] + half_w],
                   half_w * sizeof(float));
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper: extract central region from a correlation plane                      */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void extract_central(
    const float *src, int src_w,
    int start_y, int start_x,
    int out_h, int out_w,
    float *dst)
{
    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            dst[i * out_w + j] = src[(start_y + i) * src_w + (start_x + j)];
        }
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper: cross-correlate two planes using an existing plan (reuse)            */
/*   Subtracts mean from both inputs before correlation.                        */
/*   Result is fftshifted.                                                      */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void xcorr_meansub(
    const float *f, const float *g,
    int numel, int numel_fft,
    const int *size,
    sPlan *plan,
    float *result)
{
    float *ab = plan->ab_copy;

    /* Compute means */
    float mean_f = 0.0f, mean_g = 0.0f;
    for (int i = 0; i < numel; ++i) {
        mean_f += f[i];
        mean_g += g[i];
    }
    mean_f /= numel;
    mean_g /= numel;

    /* Copy mean-subtracted data into plan's batched input */
    for (int i = 0; i < numel; ++i) {
        ab[i]         = f[i] - mean_f;   /* slot 0 */
        ab[numel + i] = g[i] - mean_g;   /* slot 1 */
    }

    /* Batched FFT */
    fftwf_execute(plan->plan_AB_fft);

    /* FFT(f) * conj(FFT(g)) */
    multiply_conjugate(
        (const fftwf_complex *)&plan->AB_copy[0],
        (const fftwf_complex *)&plan->AB_copy[numel_fft],
        plan->C,
        numel_fft);

    /* IFFT → normalize → fftshift */
    fftwf_execute(plan->plan_C_ifft);
    {
        float mul = 1.0f / (float)numel;
        float *cc = plan->c_copy;
        for (int i = 0; i < numel; ++i) cc[i] *= mul;

        int half_h = size[0] / 2;
        int half_w = size[1] / 2;
        for (int row = 0; row < size[0]; ++row) {
            int row_swap = (row + half_h) % size[0];
            memcpy(&result[row * size[1] + half_w],
                   &cc[row_swap * size[1]],
                   half_w * sizeof(float));
            memcpy(&result[row * size[1]],
                   &cc[row_swap * size[1] + half_w],
                   half_w * sizeof(float));
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Main exported function                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

unsigned char bulkxcorr2d_stereo_coc_accumulate(
    const float *fImage1A_stack, const float *fImage1B_stack,
    const float *fImage2A_stack, const float *fImage2B_stack,
    const float *fMask,
    const int   *nImageSize,
    int          N_images,
    const float *fWinCtrsX,
    const float *fWinCtrsY,
    const int   *nWindows,
    const float *fWindowWeightA_AB,
    const float *fWindowWeightB_AB,
    const float *fAutoWeightA,
    const float *fAutoWeightB,
    const int   *nWindowSize,
    const int   *nFitWindowSize,
    float       *fCorr1AB_Sum,
    float       *fCorr1AA_Sum,
    float       *fCorr1BB_Sum,
    float       *fCorr2AB_Sum,
    float       *fCorr2AA_Sum,
    float       *fCorr2BB_Sum,
    float       *fCoC_Sum,
    int          diag_window_idx,
    float       *fDiag_AB1,
    float       *fDiag_AB2,
    float       *fDiag_CoC)
{
    int nWindowsTotal = nWindows[0] * nWindows[1];
    int nPxPerWindow  = nWindowSize[0] * nWindowSize[1];
    int nImagePixels  = nImageSize[0] * nImageSize[1];
    unsigned uError   = ERROR_NONE;
    int i, j, n, iWindowIdx, ii, jj;

    /* Per-camera correlation output dimensions (central extraction) */
    int out_h = nFitWindowSize ? nFitWindowSize[0] : nWindowSize[0];
    int out_w = nFitWindowSize ? nFitWindowSize[1] : nWindowSize[1];
    int nPxPerOutput = out_h * out_w;
    int start_y = (nWindowSize[0] - out_h) / 2;
    int start_x = (nWindowSize[1] - out_w) / 2;

    /* CoC output is SAME size as per-camera output (no zero-padding) */
    int nPxPerCoC = nPxPerOutput;

    /* Frequency-domain sizes */
    int numel     = nWindowSize[0] * nWindowSize[1];
    int numel_fft = nWindowSize[0] * (nWindowSize[1] / 2 + 1);

    /* For CoC when nFitWindowSize != nWindowSize, we need a separate plan.
     * When they're equal (no sum_fitting_window), reuse the main plan. */
    int coc_numel     = out_h * out_w;
    int coc_numel_fft = out_h * (out_w / 2 + 1);
    int coc_needs_own_plan = (out_h != nWindowSize[0] || out_w != nWindowSize[1]);
    int coc_plan_size[2] = {out_h, out_w};

    fftw_library_init();

    char wisdom_path[512];
    xcorr_cache_get_default_wisdom_path(wisdom_path, sizeof(wisdom_path));
    xcorr_cache_init(wisdom_path);

    #pragma omp parallel \
        default(none) \
        shared(fImage1A_stack, fImage1B_stack, fImage2A_stack, fImage2B_stack, \
               fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, \
               fWindowWeightA_AB, fWindowWeightB_AB, fAutoWeightA, fAutoWeightB, \
               nWindowSize, nFitWindowSize, \
               fCorr1AB_Sum, fCorr1AA_Sum, fCorr1BB_Sum, \
               fCorr2AB_Sum, fCorr2AA_Sum, fCorr2BB_Sum, \
               fCoC_Sum, \
               nPxPerWindow, nWindowsTotal, nImagePixels, numel, numel_fft, \
               out_h, out_w, nPxPerOutput, nPxPerCoC, start_y, start_x, \
               coc_numel, coc_numel_fft, coc_needs_own_plan, coc_plan_size, \
               diag_window_idx, fDiag_AB1, fDiag_AB2, fDiag_CoC) \
        private(i, j, n, iWindowIdx, ii, jj) \
        reduction(|:uError)
    {
        /* ── Thread-local workspace ─────────────────────────────────────── */

        float *fRaw1A = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        float *fRaw1B = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        float *fRaw2A = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        float *fRaw2B = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));

        /* Image-size FFTW plan (for per-camera AB/AA/BB correlations) */
        sPlan sCCPlan;
        memset(&sCCPlan, 0, sizeof(sCCPlan));

        /* CoC plan: same as main plan when no fit_window extraction,
         * or a separate smaller plan when fit_window != window_size */
        sPlan sCoCPlan;
        memset(&sCoCPlan, 0, sizeof(sCoCPlan));

        /* Extra IFFT scratch for auto-correlations */
        fftwf_complex *C_auto1 = NULL, *C_auto2 = NULL;
        fftwf_complex *C_auto3 = NULL, *C_auto4 = NULL;

        /* Per-frame correlation planes */
        float *fCorrel1AB = NULL, *fCorrel1AA = NULL, *fCorrel1BB = NULL;
        float *fCorrel2AB = NULL, *fCorrel2AA = NULL, *fCorrel2BB = NULL;

        /* CoC workspace */
        float *fAB1_central = NULL, *fAB2_central = NULL;
        float *fCoC_result  = NULL;

        if (!fRaw1A || !fRaw1B || !fRaw2A || !fRaw2B) {
            uError = ERROR_NOMEM; goto stereo_cleanup;
        }

        #pragma omp critical
        uError = xcorr_create_plan(nWindowSize, &sCCPlan);
        if (uError) goto stereo_cleanup;

        /* CoC plan: create separate plan only if sizes differ */
        if (coc_needs_own_plan) {
            #pragma omp critical
            uError = xcorr_create_plan(coc_plan_size, &sCoCPlan);
            if (uError) goto stereo_cleanup;
        }

        C_auto1 = (fftwf_complex *)fftwf_alloc_complex(numel_fft);
        C_auto2 = (fftwf_complex *)fftwf_alloc_complex(numel_fft);
        C_auto3 = (fftwf_complex *)fftwf_alloc_complex(numel_fft);
        C_auto4 = (fftwf_complex *)fftwf_alloc_complex(numel_fft);

        fCorrel1AB = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrel1AA = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrel1BB = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrel2AB = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrel2AA = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));
        fCorrel2BB = (float *)fftwf_malloc(nPxPerWindow * sizeof(float));

        fAB1_central = (float *)fftwf_malloc(nPxPerOutput * sizeof(float));
        fAB2_central = (float *)fftwf_malloc(nPxPerOutput * sizeof(float));
        fCoC_result  = (float *)fftwf_malloc(nPxPerCoC * sizeof(float));

        if (!C_auto1 || !C_auto2 || !C_auto3 || !C_auto4 ||
            !fCorrel1AB || !fCorrel1AA || !fCorrel1BB ||
            !fCorrel2AB || !fCorrel2AA || !fCorrel2BB ||
            !fAB1_central || !fAB2_central || !fCoC_result) {
            uError = ERROR_NOMEM; goto stereo_cleanup;
        }

        /* ── Parallel over windows ──────────────────────────────────────── */
        #pragma omp for schedule(static)
        for (iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
        {
            ii = iWindowIdx % nWindows[1];
            jj = iWindowIdx / nWindows[1];
            if (fMask[iWindowIdx] == 1) continue;

            int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0] - 1.0f) / 2.0f + 0.5f);
            int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1] - 1.0f) / 2.0f + 0.5f);
            if (row_min < 0 || col_min < 0 ||
                row_min + nWindowSize[0] > nImageSize[0] ||
                col_min + nWindowSize[1] > nImageSize[1]) continue;

            float *out1AB = &fCorr1AB_Sum[iWindowIdx * nPxPerOutput];
            float *out1AA = &fCorr1AA_Sum[iWindowIdx * nPxPerOutput];
            float *out1BB = &fCorr1BB_Sum[iWindowIdx * nPxPerOutput];
            float *out2AB = &fCorr2AB_Sum[iWindowIdx * nPxPerOutput];
            float *out2AA = &fCorr2AA_Sum[iWindowIdx * nPxPerOutput];
            float *out2BB = &fCorr2BB_Sum[iWindowIdx * nPxPerOutput];
            float *outCoC = &fCoC_Sum[iWindowIdx * nPxPerCoC];

            for (n = 0; n < N_images; ++n)
            {
                const float *fIm1A = &fImage1A_stack[n * nImagePixels];
                const float *fIm1B = &fImage1B_stack[n * nImagePixels];
                const float *fIm2A = &fImage2A_stack[n * nImagePixels];
                const float *fIm2B = &fImage2B_stack[n * nImagePixels];

                /* ─── 1. Extract raw sub-images ─────────────────────────── */
                extract_raw_subimage(fIm1A, row_min, col_min, nWindowSize, nImageSize, fRaw1A);
                extract_raw_subimage(fIm1B, row_min, col_min, nWindowSize, nImageSize, fRaw1B);
                extract_raw_subimage(fIm2A, row_min, col_min, nWindowSize, nImageSize, fRaw2A);
                extract_raw_subimage(fIm2B, row_min, col_min, nWindowSize, nImageSize, fRaw2B);

                /* ─── 2. AB cross-correlations ──────────────────────────── */
                compute_AB_xcorr(fRaw1A, fRaw1B,
                                 fWindowWeightA_AB, fWindowWeightB_AB,
                                 numel, numel_fft, nWindowSize,
                                 &sCCPlan, fCorrel1AB);

                compute_AB_xcorr(fRaw2A, fRaw2B,
                                 fWindowWeightA_AB, fWindowWeightB_AB,
                                 numel, numel_fft, nWindowSize,
                                 &sCCPlan, fCorrel2AB);

                /* ─── 3. Auto-correlations ──────────────────────────────── */
                compute_auto_xcorr(fRaw1A, fAutoWeightA,
                                   numel, numel_fft, nWindowSize,
                                   &sCCPlan, C_auto1, fCorrel1AA);
                compute_auto_xcorr(fRaw1B, fAutoWeightB,
                                   numel, numel_fft, nWindowSize,
                                   &sCCPlan, C_auto2, fCorrel1BB);
                compute_auto_xcorr(fRaw2A, fAutoWeightA,
                                   numel, numel_fft, nWindowSize,
                                   &sCCPlan, C_auto3, fCorrel2AA);
                compute_auto_xcorr(fRaw2B, fAutoWeightB,
                                   numel, numel_fft, nWindowSize,
                                   &sCCPlan, C_auto4, fCorrel2BB);

                /* ─── 4. Accumulate per-camera correlation planes ───────── */
                for (i = 0; i < out_h; ++i) {
                    for (j = 0; j < out_w; ++j) {
                        int src_idx = (start_y + i) * nWindowSize[1] + (start_x + j);
                        int dst_idx = i * out_w + j;
                        out1AB[dst_idx] += fCorrel1AB[src_idx];
                        out1AA[dst_idx] += fCorrel1AA[src_idx];
                        out1BB[dst_idx] += fCorrel1BB[src_idx];
                        out2AB[dst_idx] += fCorrel2AB[src_idx];
                        out2AA[dst_idx] += fCorrel2AA[src_idx];
                        out2BB[dst_idx] += fCorrel2BB[src_idx];
                    }
                }

                /* ─── 5. CoC: cross-correlate AB1 × AB2 ────────────────── */
                /* Extract central regions from AB planes */
                extract_central(fCorrel1AB, nWindowSize[1],
                                start_y, start_x, out_h, out_w,
                                fAB1_central);
                extract_central(fCorrel2AB, nWindowSize[1],
                                start_y, start_x, out_h, out_w,
                                fAB2_central);

                /* Cross-correlate AB1 × AB2 WITHOUT mean subtraction.
                 * The AB planes already had image DC removed during per-camera
                 * correlation. The remaining spatial mean is small (~0.4% of peak).
                 * Mean subtraction creates a negative floor that produces banding
                 * artifacts in the CoC via peak×floor cross-terms. */
                {
                    sPlan *coc_plan = coc_needs_own_plan ? &sCoCPlan : &sCCPlan;

                    xcorr_preplanned(
                        fAB1_central, fAB2_central,
                        fCoC_result, coc_plan);
                }

                /* ─── 6. Accumulate CoC ─────────────────────────────────── */
                for (i = 0; i < nPxPerCoC; ++i) {
                    outCoC[i] += fCoC_result[i];
                }

                /* ─── 7. Diagnostic: store per-frame planes ────────────── */
                if (iWindowIdx == diag_window_idx && fDiag_AB1 && fDiag_AB2 && fDiag_CoC) {
                    int frame_offset = n * nPxPerOutput;
                    for (i = 0; i < nPxPerOutput; ++i) {
                        fDiag_AB1[frame_offset + i] = fAB1_central[i];
                        fDiag_AB2[frame_offset + i] = fAB2_central[i];
                    }
                    for (i = 0; i < nPxPerCoC; ++i) {
                        fDiag_CoC[frame_offset + i] = fCoC_result[i];
                    }
                }

            } /* end frame loop */
        } /* end window loop */

    stereo_cleanup:
        if (fRaw1A)       fftwf_free(fRaw1A);
        if (fRaw1B)       fftwf_free(fRaw1B);
        if (fRaw2A)       fftwf_free(fRaw2A);
        if (fRaw2B)       fftwf_free(fRaw2B);
        if (fCorrel1AB)   fftwf_free(fCorrel1AB);
        if (fCorrel1AA)   fftwf_free(fCorrel1AA);
        if (fCorrel1BB)   fftwf_free(fCorrel1BB);
        if (fCorrel2AB)   fftwf_free(fCorrel2AB);
        if (fCorrel2AA)   fftwf_free(fCorrel2AA);
        if (fCorrel2BB)   fftwf_free(fCorrel2BB);
        if (C_auto1)      fftwf_free(C_auto1);
        if (C_auto2)      fftwf_free(C_auto2);
        if (C_auto3)      fftwf_free(C_auto3);
        if (C_auto4)      fftwf_free(C_auto4);
        if (fAB1_central) fftwf_free(fAB1_central);
        if (fAB2_central) fftwf_free(fAB2_central);
        if (fCoC_result)  fftwf_free(fCoC_result);

        #pragma omp critical
        xcorr_destroy_plan(&sCCPlan);
        if (coc_needs_own_plan) {
            #pragma omp critical
            xcorr_destroy_plan(&sCoCPlan);
        }

    } /* end omp parallel */

    xcorr_cache_save_wisdom(wisdom_path);
    return uError;
}
