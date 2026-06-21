#include "PIV_2d_cross_correlate.h"
#include "common.h"
#include "xcorr.h"
#include "codelet_fft.h"      /* permissive FFTW-free transform engine */
#include "peak_locate_lm.h"   /* Fast LM solver instead of GSL */
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* Post-process one finished correlation lane (instantaneous path): correlation-
 * plane weighting, optional raw-plane copy-out, and peak fit + output writes.
 * This is the EXACT per-window post-processing of the original scalar loop,
 * factored out so the only thing the batching changed is where the FFT runs.
 * `plane` is one window's correlation surface (length nPxPerWindow). */
static void instant_postprocess_lane(
    float *plane, int idx, int n, int iWindowIdx, float fEnergyNorm,
    bool bEnsemble, const float *fCorrelWeight, const int *nWindowSize,
    int nPeaks, int iPeakFinder, int nWindowsTotal, int nPxPerWindow,
    float *fPeakLoc, float *fStd,
    float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy,
    float *fCorrelPlane_Out)
{
    if (!bEnsemble) {
        #pragma omp simd
        for (int i = 0; i < nPxPerWindow; ++i) plane[i] *= fCorrelWeight[i];
    }

    /* Copy correlation plane to output - only needed for ensemble mode */
    if (bEnsemble && fCorrelPlane_Out != NULL) {
        memcpy(&fCorrelPlane_Out[(size_t)idx * nPxPerWindow], plane, nPxPerWindow * sizeof(float));
    }

    /* Peak finder */
    if (!bEnsemble) {
        lsqpeaklocate_lm(plane, nWindowSize, fPeakLoc, nPeaks, iPeakFinder, fStd);
        for (int i = 0; i < nPeaks; ++i) {
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

/* Run the batched cross-correlation for L_real prepared lanes (packed
 * [LANES][numel]) and post-process each. slot0 = B, slot1 = A, matching the
 * original xcorr_preplanned(fWindowB, fWindowA, ...). The packed input must be
 * zero-padded to LANES on the final partial batch; only the L_real real lanes
 * are post-processed. */
static void instant_flush_batch(
    codelet_plan_b *pb, int L_real, int numel,
    const float *packB, const float *packA, float *packC,
    const int *lane_idx, const int *lane_n, const int *lane_win, const float *lane_enorm,
    bool bEnsemble, const float *fCorrelWeight, const int *nWindowSize,
    int nPeaks, int iPeakFinder, int nWindowsTotal, int nPxPerWindow,
    float *fPeakLoc, float *fStd,
    float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy,
    float *fCorrelPlane_Out)
{
    codelet_forward_batch(pb, packB, 0);     /* slot 0 = B */
    codelet_forward_batch(pb, packA, 1);     /* slot 1 = A */
    codelet_emit_xcorr_batch(pb, packC);

    for (int l = 0; l < L_real; ++l) {
        instant_postprocess_lane(
            &packC[(size_t)l * numel], lane_idx[l], lane_n[l], lane_win[l], lane_enorm[l],
            bEnsemble, fCorrelWeight, nWindowSize, nPeaks, iPeakFinder,
            nWindowsTotal, nPxPerWindow, fPeakLoc, fStd,
            fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out);
    }
}

unsigned char bulkxcorr2d(
const float *fImageA_stack, const float *fImageB_stack, const float *fMask,
const int *nImageSize, int N_images,
const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows,
float *fWindowWeightA, bool bEnsemble,
const float *fWindowWeightB, const int *nWindowSize, int nPeaks, int iPeakFinder,
float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy,
float *fCorrelPlane_Out)
{
    int nWindowsTotal = nWindows[0] * nWindows[1];
    int nPxPerWindow = nWindowSize[0] * nWindowSize[1];
    unsigned uError = ERROR_NONE;

    float *fCorrelWeight = (float*)malloc(nPxPerWindow * sizeof(float));
    if (!fCorrelWeight) return ERROR_NOMEM;

    /* Precompute correlation plane weighting */
    uError = convolve(fWindowWeightB, fWindowWeightB, fCorrelWeight, nWindowSize);
    if (uError) { free(fCorrelWeight); return uError; }
    for (int i = 0; i < nPxPerWindow; ++i) fCorrelWeight[i] = nPxPerWindow / fCorrelWeight[i];

    int total_windows = N_images * nWindowsTotal;

    /* Flattened parallel loop over all windows in all images. Stage B: each
     * thread accumulates valid windows into a SIMD batch (one window per lane);
     * a full batch of LANES is cross-correlated in one call, then each lane is
     * post-processed. The final partial batch is zero-padded so every window
     * goes through the identical batched path (no scalar-tail numerical seam). */
    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, bEnsemble, fCorrelWeight, fWindowWeightA, fWindowWeightB, nWindowSize, \
               nPeaks, iPeakFinder, fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out, nPxPerWindow, nWindowsTotal, total_windows) \
        reduction(|:uError)
    {
        const int LANES = codelet_lanes();
        const int numel = nPxPerWindow;

        codelet_plan_b *pb = codelet_plan_create_batched(nWindowSize[0], nWindowSize[1]);
        float *packA = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packB = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packC = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *fStd     = (float*)malloc(3 * nPeaks * sizeof(float));
        float *fPeakLoc = (float*)malloc(3 * nPeaks * sizeof(float));
        int   *lane_idx = (int*)malloc(LANES * sizeof(int));
        int   *lane_n   = (int*)malloc(LANES * sizeof(int));
        int   *lane_win = (int*)malloc(LANES * sizeof(int));
        float *lane_enorm = (float*)malloc(LANES * sizeof(float));

        if (!pb || !packA || !packB || !packC || !fStd || !fPeakLoc ||
            !lane_idx || !lane_n || !lane_win || !lane_enorm)
        { uError = ERROR_NOMEM; goto thread_cleanup; }

        int batch = 0;

        #pragma omp for schedule(static)
        for (int idx = 0; idx < total_windows; ++idx)
        {
            int n = idx / nWindowsTotal;            // image index
            int iWindowIdx = idx % nWindowsTotal;   // window index
            int ii = iWindowIdx % nWindows[1];
            int jj = iWindowIdx / nWindows[1];

            const float *fImageA = &fImageA_stack[(size_t)n * nImageSize[0] * nImageSize[1]];
            const float *fImageB = &fImageB_stack[(size_t)n * nImageSize[0] * nImageSize[1]];

            int mask_idx = jj * nWindows[1] + ii;
            if (mask_idx < 0 || mask_idx >= nWindowsTotal) continue;
            if (fMask[mask_idx] == 1) continue;

            int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0)/2.0 + 0.5);
            int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0)/2.0 + 0.5);
            if(row_min < 0 || col_min < 0 || row_min + nWindowSize[0] > nImageSize[0] || col_min + nWindowSize[1] > nImageSize[1]) continue;

            /* Extract this valid window into the current batch lane. */
            float *pA = &packA[(size_t)batch * numel];
            float *pB = &packB[(size_t)batch * numel];
            for(int i = 0; i < nWindowSize[0]; ++i)
                for(int j = 0; j < nWindowSize[1]; ++j)
                {
                    int img_idx = (row_min+i)*nImageSize[1] + (col_min+j);
                    int win_idx = i*nWindowSize[1] + j;
                    pA[win_idx] = fImageA[img_idx];
                    pB[win_idx] = fImageB[img_idx];
                }

            float fMeanA = 0.0f, fMeanB = 0.0f;
            #pragma omp simd reduction(+:fMeanA,fMeanB)
            for(int i = 0; i < nPxPerWindow; ++i)
            {
                pA[i] *= fWindowWeightA[i];
                pB[i] *= fWindowWeightB[i];
                fMeanA += pA[i];
                fMeanB += pB[i];
            }
            fMeanA /= nPxPerWindow;
            fMeanB /= nPxPerWindow;

            float fEnergyA = 0.0f, fEnergyB = 0.0f;
            if(!bEnsemble)
            {
                #pragma omp simd reduction(+:fEnergyA,fEnergyB)
                for(int i = 0; i < nPxPerWindow; ++i)
                {
                    pA[i] -= fMeanA;
                    pB[i] -= fMeanB;
                    fEnergyA += pA[i]*pA[i];
                    fEnergyB += pB[i]*pB[i];
                }
            }
            else
            {
                #pragma omp simd reduction(+:fEnergyA,fEnergyB)
                for(int i = 0; i < nPxPerWindow; ++i)
                {
                    fEnergyA += pA[i]*pA[i];
                    fEnergyB += pB[i]*pB[i];
                }
            }

            lane_idx[batch]   = idx;
            lane_n[batch]     = n;
            lane_win[batch]   = iWindowIdx;
            lane_enorm[batch] = 1.0f / sqrtf(fEnergyA * fEnergyB);
            batch++;

            if (batch == LANES) {
                instant_flush_batch(pb, LANES, numel, packB, packA, packC,
                    lane_idx, lane_n, lane_win, lane_enorm,
                    bEnsemble, fCorrelWeight, nWindowSize, nPeaks, iPeakFinder,
                    nWindowsTotal, nPxPerWindow, fPeakLoc, fStd,
                    fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out);
                batch = 0;
            }
        }

        /* Tail: zero-pad the partial batch's unused lanes, process real lanes. */
        if (batch > 0 && !uError) {
            memset(&packA[(size_t)batch * numel], 0, (size_t)(LANES - batch) * numel * sizeof(float));
            memset(&packB[(size_t)batch * numel], 0, (size_t)(LANES - batch) * numel * sizeof(float));
            instant_flush_batch(pb, batch, numel, packB, packA, packC,
                lane_idx, lane_n, lane_win, lane_enorm,
                bEnsemble, fCorrelWeight, nWindowSize, nPeaks, iPeakFinder,
                nWindowsTotal, nPxPerWindow, fPeakLoc, fStd,
                fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out);
        }

    thread_cleanup:
        if (pb) codelet_plan_destroy_batched(pb);
        free(packA); free(packB); free(packC);
        free(fStd); free(fPeakLoc);
        free(lane_idx); free(lane_n); free(lane_win); free(lane_enorm);
    }

    free(fCorrelWeight);
    return uError;
}

/* Batched cross-correlation accumulation: run the batched xcorr for L_real
 * prepared image lanes (packed [LANES][numel], slot0=B, slot1=A) and add each
 * lane's central region into the per-window accumulator out_ptr. Final partial
 * batch must be zero-padded to LANES; only L_real real lanes are accumulated. */
static void accum_flush_batch(
    codelet_plan_b *pb, int L_real, int numel,
    const float *packB, const float *packA, float *packC,
    float *out_ptr, int out_h, int out_w, int start_y, int start_x, int W)
{
    codelet_forward_batch(pb, packB, 0);
    codelet_forward_batch(pb, packA, 1);
    codelet_emit_xcorr_batch(pb, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                out_ptr[i * out_w + j] += plane[(start_y + i) * W + (start_x + j)];
    }
}

/* Batched triple accumulation: AB (cross), AA, BB (autos) for L_real image
 * lanes, preserving the three distinct weightings. Inputs are pre-weighted,
 * packed [LANES][numel]: AB uses (B_AB slot0, A_AB slot1); AA uses A_auto;
 * BB uses B_auto. Each result's central region is added to its accumulator. */
static void triple_flush_batch(
    codelet_plan_b *pb, int L_real, int numel,
    const float *packAB_B, const float *packAB_A,
    const float *packAA_A, const float *packBB_B, float *packC,
    float *outAB, float *outAA, float *outBB,
    int out_h, int out_w, int start_y, int start_x, int W)
{
    /* AB: cross-correlation of the AB-weighted windows. */
    codelet_forward_batch(pb, packAB_B, 0);
    codelet_forward_batch(pb, packAB_A, 1);
    codelet_emit_xcorr_batch(pb, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                outAB[i * out_w + j] += plane[(start_y + i) * W + (start_x + j)];
    }
    /* AA: auto-correlation of the auto-weighted A windows. */
    codelet_forward_batch(pb, packAA_A, 0);
    codelet_emit_power_batch(pb, 0, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                outAA[i * out_w + j] += plane[(start_y + i) * W + (start_x + j)];
    }
    /* BB: auto-correlation of the auto-weighted B windows. */
    codelet_forward_batch(pb, packBB_B, 0);
    codelet_emit_power_batch(pb, 0, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                outBB[i * out_w + j] += plane[(start_y + i) * W + (start_x + j)];
    }
}

/**
 * Ensemble-optimized cross-correlation with internal accumulation.
 * Parallel over windows, batched over images (one image per SIMD lane).
 *
 * If nFitWindowSize is not NULL, extracts only the central region of each
 * correlation plane for accumulation. This reduces memory usage and fitting time.
 *
 * Stage B note: the original computed a per-image mean and energy that were
 * never used by the accumulation (ensemble does not mean-subtract here -- that
 * happens via background correlation in Python). Those provably-dead sums are
 * dropped; the windowing (image * weight) that feeds the FFT is unchanged, so
 * the accumulated output is identical.
 *
 * NOT yet validated end-to-end (instantaneous was the validated path); built on
 * the gate-proven batched primitives and to be validated with the user.
 */
unsigned char bulkxcorr2d_accumulate(
    const float *fImageA_stack, const float *fImageB_stack, const float *fMask,
    const int *nImageSize, int N_images,
    const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows,
    const float *fWindowWeightA, const float *fWindowWeightB,
    const int *nWindowSize,
    const int *nFitWindowSize,  /* output size, NULL = use full nWindowSize */
    float *fCorrelPlane_Sum)
{
    int nWindowsTotal = nWindows[0] * nWindows[1];
    int nPxPerWindow = nWindowSize[0] * nWindowSize[1];  /* FFT computation size */
    int nImagePixels = nImageSize[0] * nImageSize[1];
    unsigned uError = ERROR_NONE;

    /* Determine output dimensions */
    int out_h = nFitWindowSize ? nFitWindowSize[0] : nWindowSize[0];
    int out_w = nFitWindowSize ? nFitWindowSize[1] : nWindowSize[1];
    int nPxPerOutput = out_h * out_w;

    /* Extraction offsets (centered) */
    int start_y = (nWindowSize[0] - out_h) / 2;
    int start_x = (nWindowSize[1] - out_w) / 2;

    /* NOTE: Output buffer is NOT zeroed here — the caller is responsible for
     * clearing buffers before the first call.  This allows multiple calls to
     * += accumulate into the same buffer without losing previous data. */

    /* Parallel over windows; batched over images (one image per SIMD lane). */
    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, fWindowWeightA, fWindowWeightB, \
               nWindowSize, fCorrelPlane_Sum, nPxPerWindow, nWindowsTotal, nImagePixels, \
               out_h, out_w, nPxPerOutput, start_y, start_x) \
        reduction(|:uError)
    {
        const int LANES = codelet_lanes();
        const int numel = nPxPerWindow;
        const int W = nWindowSize[1];

        codelet_plan_b *pb = codelet_plan_create_batched(nWindowSize[0], nWindowSize[1]);
        float *packA = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packB = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packC = (float*)malloc((size_t)LANES * numel * sizeof(float));

        if (!pb || !packA || !packB || !packC) { uError = ERROR_NOMEM; goto thread_cleanup; }

        /* Outer loop: parallel over windows */
        #pragma omp for schedule(static)
        for (int iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
        {
            int ii = iWindowIdx % nWindows[1];  /* column */
            int jj = iWindowIdx / nWindows[1];  /* row */

            if (fMask[iWindowIdx] == 1) continue;

            int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0f)/2.0f + 0.5f);
            int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0f)/2.0f + 0.5f);
            if (row_min < 0 || col_min < 0 ||
                row_min + nWindowSize[0] > nImageSize[0] ||
                col_min + nWindowSize[1] > nImageSize[1]) continue;

            /* This thread owns this window's output (offset uses nPxPerOutput). */
            float *out_ptr = &fCorrelPlane_Sum[(size_t)iWindowIdx * nPxPerOutput];

            int batch = 0;
            for (int n = 0; n < N_images; ++n)
            {
                const float *fImageA = &fImageA_stack[(size_t)n * nImagePixels];
                const float *fImageB = &fImageB_stack[(size_t)n * nImagePixels];

                float *pA = &packA[(size_t)batch * numel];
                float *pB = &packB[(size_t)batch * numel];
                for (int i = 0; i < nWindowSize[0]; ++i) {
                    for (int j = 0; j < nWindowSize[1]; ++j) {
                        int img_idx = (row_min + i) * nImageSize[1] + (col_min + j);
                        int win_idx = i * nWindowSize[1] + j;
                        pA[win_idx] = fImageA[img_idx] * fWindowWeightA[win_idx];
                        pB[win_idx] = fImageB[img_idx] * fWindowWeightB[win_idx];
                    }
                }
                batch++;

                if (batch == LANES) {
                    accum_flush_batch(pb, LANES, numel, packB, packA, packC,
                                      out_ptr, out_h, out_w, start_y, start_x, W);
                    batch = 0;
                }
            }
            /* Tail: zero-pad unused lanes, accumulate only the real ones. */
            if (batch > 0) {
                memset(&packA[(size_t)batch * numel], 0, (size_t)(LANES - batch) * numel * sizeof(float));
                memset(&packB[(size_t)batch * numel], 0, (size_t)(LANES - batch) * numel * sizeof(float));
                accum_flush_batch(pb, batch, numel, packB, packA, packC,
                                  out_ptr, out_h, out_w, start_y, start_x, W);
            }
        }

    thread_cleanup:
        if (pb) codelet_plan_destroy_batched(pb);
        free(packA); free(packB); free(packC);
    }

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

    /* Output dimensions (central extraction if nFitWindowSize != NULL) */
    int out_h = nFitWindowSize ? nFitWindowSize[0] : nWindowSize[0];
    int out_w = nFitWindowSize ? nFitWindowSize[1] : nWindowSize[1];
    int nPxPerOutput = out_h * out_w;
    int start_y = (nWindowSize[0] - out_h) / 2;
    int start_x = (nWindowSize[1] - out_w) / 2;

    /* Window element count. */
    int numel = nWindowSize[0] * nWindowSize[1];

    /* NOTE: Output buffers are NOT zeroed here — the caller is responsible for
     * clearing buffers before the first call.  This allows multiple calls to
     * += accumulate into the same buffer without losing previous data. */

    /* Parallel over windows; batched over images (one image per SIMD lane). The
     * three distinct weightings (AB asymmetric, AA/BB auto) are preserved: each
     * image fills four pre-weighted lane buffers, and a full batch runs one
     * batched AB xcorr + AA power + BB power. */
    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, \
               fWindowWeightA_AB, fWindowWeightB_AB, fAutoWeightA, fAutoWeightB, \
               nWindowSize, fCorrAB_Sum, fCorrAA_Sum, fCorrBB_Sum, \
               nPxPerWindow, nWindowsTotal, nImagePixels, numel, \
               out_h, out_w, nPxPerOutput, start_y, start_x) \
        reduction(|:uError)
    {
        const int LANES = codelet_lanes();
        const int W = nWindowSize[1];

        codelet_plan_b *pb = codelet_plan_create_batched(nWindowSize[0], nWindowSize[1]);
        /* Per-image raw extraction (one image at a time before weighting). */
        float *fRawA = (float*)malloc((size_t)numel * sizeof(float));
        float *fRawB = (float*)malloc((size_t)numel * sizeof(float));
        /* Four pre-weighted lane buffers, packed [LANES][numel]. */
        float *packAB_B = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packAB_A = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packAA_A = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packBB_B = (float*)malloc((size_t)LANES * numel * sizeof(float));
        float *packC    = (float*)malloc((size_t)LANES * numel * sizeof(float));

        if (!pb || !fRawA || !fRawB || !packAB_B || !packAB_A ||
            !packAA_A || !packBB_B || !packC) {
            uError = ERROR_NOMEM; goto triple_cleanup;
        }

        /* ---- parallel over windows ------------------------------------ */
        #pragma omp for schedule(static)
        for (int iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
        {
            int ii = iWindowIdx % nWindows[1];
            int jj = iWindowIdx / nWindows[1];
            if (fMask[iWindowIdx] == 1) continue;

            int row_min = (int)floor(fWinCtrsY[jj] - ((float)nWindowSize[0]-1.0f)/2.0f + 0.5f);
            int col_min = (int)floor(fWinCtrsX[ii] - ((float)nWindowSize[1]-1.0f)/2.0f + 0.5f);
            if (row_min < 0 || col_min < 0 ||
                row_min + nWindowSize[0] > nImageSize[0] ||
                col_min + nWindowSize[1] > nImageSize[1]) continue;

            float *outAB = &fCorrAB_Sum[(size_t)iWindowIdx * nPxPerOutput];
            float *outAA = &fCorrAA_Sum[(size_t)iWindowIdx * nPxPerOutput];
            float *outBB = &fCorrBB_Sum[(size_t)iWindowIdx * nPxPerOutput];

            /* ---- batched over images ---------------------------------- */
            int batch = 0;
            for (int n = 0; n < N_images; ++n)
            {
                const float *fImageA = &fImageA_stack[(size_t)n * nImagePixels];
                const float *fImageB = &fImageB_stack[(size_t)n * nImagePixels];

                /* Extract raw pixels for this image. */
                for (int i = 0; i < nWindowSize[0]; ++i) {
                    for (int j = 0; j < nWindowSize[1]; ++j) {
                        int img_idx = (row_min + i) * nImageSize[1] + (col_min + j);
                        int win_idx = i * nWindowSize[1] + j;
                        fRawA[win_idx] = fImageA[img_idx];
                        fRawB[win_idx] = fImageB[img_idx];
                    }
                }

                /* Fill the four pre-weighted lane buffers (slot/weight layout
                 * matches the original: AB slot0=B_AB slot1=A_AB; AA=auto·A;
                 * BB=auto·B). */
                float *pAB_B = &packAB_B[(size_t)batch * numel];
                float *pAB_A = &packAB_A[(size_t)batch * numel];
                float *pAA_A = &packAA_A[(size_t)batch * numel];
                float *pBB_B = &packBB_B[(size_t)batch * numel];
                for (int i = 0; i < numel; ++i) {
                    pAB_B[i] = fRawB[i] * fWindowWeightB_AB[i];
                    pAB_A[i] = fRawA[i] * fWindowWeightA_AB[i];
                    pAA_A[i] = fRawA[i] * fAutoWeightA[i];
                    pBB_B[i] = fRawB[i] * fAutoWeightB[i];
                }
                batch++;

                if (batch == LANES) {
                    triple_flush_batch(pb, LANES, numel,
                        packAB_B, packAB_A, packAA_A, packBB_B, packC,
                        outAB, outAA, outBB, out_h, out_w, start_y, start_x, W);
                    batch = 0;
                }
            } /* end image loop */

            /* Tail: zero-pad unused lanes, accumulate only the real ones. */
            if (batch > 0) {
                size_t padoff = (size_t)batch * numel;
                size_t padlen = (size_t)(LANES - batch) * numel * sizeof(float);
                memset(&packAB_B[padoff], 0, padlen);
                memset(&packAB_A[padoff], 0, padlen);
                memset(&packAA_A[padoff], 0, padlen);
                memset(&packBB_B[padoff], 0, padlen);
                triple_flush_batch(pb, batch, numel,
                    packAB_B, packAB_A, packAA_A, packBB_B, packC,
                    outAB, outAA, outBB, out_h, out_w, start_y, start_x, W);
            }
        } /* end window loop */

    triple_cleanup:
        if (pb) codelet_plan_destroy_batched(pb);
        free(fRawA); free(fRawB);
        free(packAB_B); free(packAB_A); free(packAA_A); free(packBB_B); free(packC);
    } /* end omp parallel */

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
