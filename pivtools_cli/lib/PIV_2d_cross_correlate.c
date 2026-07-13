#include "PIV_2d_cross_correlate.h"
#include "common.h"
#include "xcorr.h"
#include "codelet_fft.h"           /* permissive FFTW-free transform engine */
#include "peak_locate_lm.h"        /* Fast LM solver instead of GSL */
#include "peak_locate_lm_batch.h"  /* lockstep one-window-per-lane LM fitter */
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- benchmark sub-kernel timing (flag-gated, additive) ----------------------
 * Splits the batched FFT/xcorr (codelet_*_batch) from the LM peak-fit
 * (lsqpeaklocate_lm) inside bulkxcorr2d so the codelet-vs-FFTW A/B can isolate
 * the FFT speedup from the constant peak-fit cost. The two globals accumulate
 * total thread-seconds (sum across OpenMP threads); they reconcile to wall-clock
 * only at OMP_NUM_THREADS=1. Zero cost when g_kernel_timing == 0 (production
 * default). Bound from the benchmark harness via ctypes; production never enables
 * it. */
static volatile int g_kernel_timing = 0;
static double g_t_fft = 0.0;   /* total thread-seconds in the batched FFT/xcorr */
static double g_t_fit = 0.0;   /* total thread-seconds in lsqpeaklocate_lm       */

void bulkxcorr2d_set_timing_enabled(int on) { g_kernel_timing = on ? 1 : 0; }
void bulkxcorr2d_reset_timing(void) { g_t_fft = 0.0; g_t_fit = 0.0; }
void bulkxcorr2d_get_timing(double *fft_s, double *fit_s)
{
    if (fft_s) *fft_s = g_t_fft;
    if (fit_s) *fit_s = g_t_fit;
}

/* --- peak-fit implementation selector ----------------------------------------
 * 0 = scalar per-lane lsqpeaklocate_lm (default), 1 = lockstep batch fitter
 * (lsqpeaklocate_lm_batch, one window per SIMD lane). Read ONCE at bulkxcorr2d
 * entry (fused_warp discipline) so a mid-run flip cannot tear a batch. The
 * setter REFUSES batch when the fitter is not compiled in (plain MSVC cl
 * build) — explicit failure, never a silent fallback. The batch path covers
 * the instantaneous nPeaks==1, iPeakFinder>=4 case; everything else keeps the
 * scalar per-lane loop regardless of the selector. */
static volatile int g_peakfit_impl = 0;

int bulkxcorr2d_set_peakfit_impl(int impl)
{
    if (impl && !peakfit_batch_available()) return -1;
    g_peakfit_impl = impl ? 1 : 0;
    return 0;
}
int bulkxcorr2d_get_peakfit_impl(void) { return g_peakfit_impl; }
int bulkxcorr2d_peakfit_batch_available(void) { return peakfit_batch_available(); }

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
    float *fCorrelPlane_Out, double *t_fit)
{
    if (!bEnsemble) {
        int i;  /* MSVC OpenMP needs the counter declared outside the for-init */
        #pragma omp simd
        for (i = 0; i < nPxPerWindow; ++i) plane[i] *= fCorrelWeight[i];
    }

    /* Copy correlation plane to output (ensemble accumulation, or the
     * instantaneous debug dump). In the instantaneous path this runs AFTER
     * the fCorrelWeight weighting above, so the dumped plane is the exact
     * surface the peak fitter sees. NULL (the instantaneous default) skips
     * the copy entirely. */
    if (fCorrelPlane_Out != NULL) {
        memcpy(&fCorrelPlane_Out[(size_t)idx * nPxPerWindow], plane, nPxPerWindow * sizeof(float));
    }

    /* Peak finder */
    if (!bEnsemble) {
        if (g_kernel_timing) {
            double t_mark = omp_get_wtime();
            lsqpeaklocate_lm(plane, nWindowSize, fPeakLoc, nPeaks, iPeakFinder, fStd);
            *t_fit += omp_get_wtime() - t_mark;
        } else {
            lsqpeaklocate_lm(plane, nWindowSize, fPeakLoc, nPeaks, iPeakFinder, fStd);
        }
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
    float *fCorrelPlane_Out, int peakfit_impl, double *t_fft, double *t_fit)
{
    if (g_kernel_timing) {
        double t_mark = omp_get_wtime();
        codelet_forward_batch(pb, packB, 0);     /* slot 0 = B */
        codelet_forward_batch(pb, packA, 1);     /* slot 1 = A */
        codelet_emit_xcorr_batch(pb, packC);
        *t_fft += omp_get_wtime() - t_mark;
    } else {
        codelet_forward_batch(pb, packB, 0);     /* slot 0 = B */
        codelet_forward_batch(pb, packA, 1);     /* slot 1 = A */
        codelet_emit_xcorr_batch(pb, packC);
    }

    /* Lockstep batch peak fit: all L_real lanes in one call, one window per
     * SIMD lane. Scope: instantaneous single-peak LM fits only — everything
     * else takes the unchanged per-lane scalar loop below. The caller
     * verified peakfit_batch_lanes() == codelet_lanes() <= PEAKFIT_MAX_LANES
     * at entry. Output writes replicate instant_postprocess_lane exactly
     * (nPeaks == 1 collapses its peak loop to out_idx = n*total + win). */
    if (peakfit_impl && !bEnsemble && nPeaks == 1 && iPeakFinder >= 4) {
        const int W = peakfit_batch_lanes();
        float ploc[3 * PEAKFIT_MAX_LANES], pstd[3 * PEAKFIT_MAX_LANES];

        for (int l = 0; l < L_real; ++l) {   /* hoisted plane weighting */
            float *plane = &packC[(size_t)l * numel];
            int i;
            #pragma omp simd
            for (i = 0; i < nPxPerWindow; ++i) plane[i] *= fCorrelWeight[i];
        }

        /* Debug plane dump (instantaneous): mirrors the copy-out in
         * instant_postprocess_lane — post-weighting, same idx layout. */
        if (fCorrelPlane_Out != NULL) {
            for (int l = 0; l < L_real; ++l)
                memcpy(&fCorrelPlane_Out[(size_t)lane_idx[l] * nPxPerWindow],
                       &packC[(size_t)l * numel], nPxPerWindow * sizeof(float));
        }

        if (g_kernel_timing) {
            double t_mark = omp_get_wtime();
            lsqpeaklocate_lm_batch(packC, L_real, nWindowSize, iPeakFinder, ploc, pstd);
            *t_fit += omp_get_wtime() - t_mark;
        } else {
            lsqpeaklocate_lm_batch(packC, L_real, nWindowSize, iPeakFinder, ploc, pstd);
        }

        for (int l = 0; l < L_real; ++l) {
            int out_idx = lane_n[l] * nWindowsTotal + lane_win[l];
            float peak_row = ploc[0 * W + l];
            float peak_col = ploc[1 * W + l];
            float peak_mag = ploc[2 * W + l];

            fPkLocX[out_idx] = peak_col - nWindowSize[1]/2.0f;
            fPkLocY[out_idx] = peak_row - nWindowSize[0]/2.0f;
            fSx[out_idx] = pstd[0 * W + l];
            fSy[out_idx] = pstd[1 * W + l];
            fSxy[out_idx] = pstd[2 * W + l];

            int pk_row = fmin(fmax(0, (int)peak_row), nWindowSize[0]-1);
            int pk_col = fmin(fmax(0, (int)peak_col), nWindowSize[1]-1);
            fPkHeight[out_idx] = peak_mag * lane_enorm[l] / fCorrelWeight[pk_row*nWindowSize[1] + pk_col];
        }
        return;
    }

    for (int l = 0; l < L_real; ++l) {
        instant_postprocess_lane(
            &packC[(size_t)l * numel], lane_idx[l], lane_n[l], lane_win[l], lane_enorm[l],
            bEnsemble, fCorrelWeight, nWindowSize, nPeaks, iPeakFinder,
            nWindowsTotal, nPxPerWindow, fPeakLoc, fStd,
            fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out, t_fit);
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

    /* Peak-fit implementation: read the selector ONCE for this whole call so a
     * mid-run flip cannot tear a batch. When batch is selected, its lane count
     * MUST match the FFT engine's (the fitter consumes packC[LANES][numel])
     * and fit within the stack output arrays — hard error, never silent. */
    const int peakfit_impl = g_peakfit_impl;
    if (peakfit_impl) {
        if (!peakfit_batch_available() ||
            peakfit_batch_lanes() != codelet_lanes() ||
            peakfit_batch_lanes() > PEAKFIT_MAX_LANES) {
            fprintf(stderr, "bulkxcorr2d: peakfit impl=batch unusable "
                            "(available=%d, batch lanes=%d, fft lanes=%d)\n",
                    peakfit_batch_available(), peakfit_batch_lanes(), codelet_lanes());
            free(fCorrelWeight);
            return ERROR_NOPLAN;
        }
    }

    /* Sub-kernel timers: stack locals reduced (+) across threads, then folded into
     * the globals after the region. Kept out of the region's lexical body so the
     * file-scope globals are never referenced under default(none) (clang-cl
     * requires explicit data-sharing for any global named inside the construct).
     * The flush helpers add to the per-thread reduction copies only when timing
     * is enabled, so these stay 0 in production (zero cost). */
    double t_fft_red = 0.0, t_fit_red = 0.0;

    /* Flattened parallel loop over all windows in all images. Stage B: each
     * thread accumulates valid windows into a SIMD batch (one window per lane);
     * a full batch of LANES is cross-correlated in one call, then each lane is
     * post-processed. The final partial batch is zero-padded so every window
     * goes through the identical batched path (no scalar-tail numerical seam). */
    #pragma omp parallel \
        default(none) \
        shared(fImageA_stack, fImageB_stack, fMask, nImageSize, N_images, \
               fWinCtrsX, fWinCtrsY, nWindows, bEnsemble, fCorrelWeight, fWindowWeightA, fWindowWeightB, nWindowSize, \
               nPeaks, iPeakFinder, fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out, nPxPerWindow, nWindowsTotal, total_windows, peakfit_impl) \
        reduction(|:uError) reduction(+:t_fft_red,t_fit_red)
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
        int idx;  /* MSVC OpenMP needs the counter declared outside the for-init */

        #pragma omp for schedule(static)
        for (idx = 0; idx < total_windows; ++idx)
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
            int i;  /* MSVC OpenMP: counter declared outside the for-init (reused below) */
            #pragma omp simd reduction(+:fMeanA,fMeanB)
            for(i = 0; i < nPxPerWindow; ++i)
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
                for(i = 0; i < nPxPerWindow; ++i)
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
                for(i = 0; i < nPxPerWindow; ++i)
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
                    fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out,
                    peakfit_impl, &t_fft_red, &t_fit_red);
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
                fPkLocX, fPkLocY, fPkHeight, fSx, fSy, fSxy, fCorrelPlane_Out,
                peakfit_impl, &t_fft_red, &t_fit_red);
        }

    thread_cleanup:
        if (pb) codelet_plan_destroy_batched(pb);
        free(packA); free(packB); free(packC);
        free(fStd); free(fPeakLoc);
        free(lane_idx); free(lane_n); free(lane_win); free(lane_enorm);
    }

    free(fCorrelWeight);

    /* Fold this pass's thread-summed sub-kernel times into the globals (additive,
     * matching the original semantics). Outside the parallel region, so touching
     * the globals here is unrestricted. No-op work when timing is off. */
    if (g_kernel_timing) {
        g_t_fft += t_fft_red;
        g_t_fit += t_fit_red;
    }

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
 * BB uses B_auto. Each result's central region is added to its accumulator.
 *
 * Per-pair normalization (bPerPairNorm): energyAA/energyBB hold the per-lane
 * weighted window energies e = sum((W*X)^2), which equal the zero-lag values
 * of the emitted AA/BB planes (unnormalized-FFT pair cancels the 1/numel in
 * the emit). Each lane's contribution is scaled by 1/eAA, 1/eBB and
 * 1/sqrt(eAA*eBB) so every pair enters the ensemble with unit auto peaks —
 * the geometric-mean AB scaling keeps T = F_AB/sqrt(F_AA*F_BB) invariant.
 * A lane with a non-positive energy carries no signal and contributes nothing
 * to any of the three planes (consistent deflation cancels in T). */
static void triple_flush_batch(
    codelet_plan_b *pb, int L_real, int numel,
    const float *packAB_B, const float *packAB_A,
    const float *packAA_A, const float *packBB_B, float *packC,
    float *outAB, float *outAA, float *outBB,
    int out_h, int out_w, int start_y, int start_x, int W,
    int bPerPairNorm, const float *energyAA, const float *energyBB)
{
    /* AB: cross-correlation of the AB-weighted windows. */
    codelet_forward_batch(pb, packAB_B, 0);
    codelet_forward_batch(pb, packAB_A, 1);
    codelet_emit_xcorr_batch(pb, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        float s = 1.0f;
        if (bPerPairNorm) {
            float prod = energyAA[l] * energyBB[l];
            if (energyAA[l] <= 0.0f || energyBB[l] <= 0.0f) continue;
            s = 1.0f / sqrtf(prod);
        }
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                outAB[i * out_w + j] += s * plane[(start_y + i) * W + (start_x + j)];
    }
    /* AA: auto-correlation of the auto-weighted A windows. */
    codelet_forward_batch(pb, packAA_A, 0);
    codelet_emit_power_batch(pb, 0, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        float s = 1.0f;
        if (bPerPairNorm) {
            if (energyAA[l] <= 0.0f || energyBB[l] <= 0.0f) continue;
            s = 1.0f / energyAA[l];
        }
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                outAA[i * out_w + j] += s * plane[(start_y + i) * W + (start_x + j)];
    }
    /* BB: auto-correlation of the auto-weighted B windows. */
    codelet_forward_batch(pb, packBB_B, 0);
    codelet_emit_power_batch(pb, 0, packC);
    for (int l = 0; l < L_real; ++l) {
        const float *plane = &packC[(size_t)l * numel];
        float s = 1.0f;
        if (bPerPairNorm) {
            if (energyAA[l] <= 0.0f || energyBB[l] <= 0.0f) continue;
            s = 1.0f / energyBB[l];
        }
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
                outBB[i * out_w + j] += s * plane[(start_y + i) * W + (start_x + j)];
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
        int iWindowIdx;  /* MSVC OpenMP needs the counter declared outside the for-init */
        #pragma omp for schedule(static)
        for (iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
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
    /* bMeanSubtract: subtract each window's weighted mean (per buffer support,
     * mean_W = sum(W*X)/sum(W)) before weighting — per-pair pedestal removal
     * ('window_mean' background method). bPerPairNorm: scale each pair's
     * contribution by its zero-lag auto energies (see triple_flush_batch). */
    int          bMeanSubtract,
    int          bPerPairNorm,
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

    /* Weight sums for the weighted means (constant across windows/images). */
    float fSumW_AB_A = 0.0f, fSumW_AB_B = 0.0f, fSumW_auto_A = 0.0f, fSumW_auto_B = 0.0f;
    if (bMeanSubtract) {
        for (int i = 0; i < numel; ++i) {
            fSumW_AB_A  += fWindowWeightA_AB[i];
            fSumW_AB_B  += fWindowWeightB_AB[i];
            fSumW_auto_A += fAutoWeightA[i];
            fSumW_auto_B += fAutoWeightB[i];
        }
    }

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
               out_h, out_w, nPxPerOutput, start_y, start_x, \
               bMeanSubtract, bPerPairNorm, \
               fSumW_AB_A, fSumW_AB_B, fSumW_auto_A, fSumW_auto_B) \
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
        /* Per-lane weighted window energies (zero-lag auto values). */
        float *energyAA = (float*)malloc((size_t)LANES * sizeof(float));
        float *energyBB = (float*)malloc((size_t)LANES * sizeof(float));

        if (!pb || !fRawA || !fRawB || !packAB_B || !packAB_A ||
            !packAA_A || !packBB_B || !packC || !energyAA || !energyBB) {
            uError = ERROR_NOMEM; goto triple_cleanup;
        }

        /* ---- parallel over windows ------------------------------------ */
        int iWindowIdx;  /* MSVC OpenMP needs the counter declared outside the for-init */
        #pragma omp for schedule(static)
        for (iWindowIdx = 0; iWindowIdx < nWindowsTotal; ++iWindowIdx)
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

                /* Weighted means per buffer support (window_mean pedestal
                 * removal): mean_W = sum(W*X)/sum(W). For flat square weights
                 * this is the plain window mean (Westerweel's mean2); for the
                 * asymmetric single-mode weights each weighted window becomes
                 * zero-mean under its own support. */
                float mAB_A = 0.0f, mAB_B = 0.0f, mAuto_A = 0.0f, mAuto_B = 0.0f;
                if (bMeanSubtract) {
                    float sAB_A = 0.0f, sAB_B = 0.0f, sAuto_A = 0.0f, sAuto_B = 0.0f;
                    for (int i = 0; i < numel; ++i) {
                        sAB_A  += fRawA[i] * fWindowWeightA_AB[i];
                        sAB_B  += fRawB[i] * fWindowWeightB_AB[i];
                        sAuto_A += fRawA[i] * fAutoWeightA[i];
                        sAuto_B += fRawB[i] * fAutoWeightB[i];
                    }
                    mAB_A  = (fSumW_AB_A  > 0.0f) ? sAB_A  / fSumW_AB_A  : 0.0f;
                    mAB_B  = (fSumW_AB_B  > 0.0f) ? sAB_B  / fSumW_AB_B  : 0.0f;
                    mAuto_A = (fSumW_auto_A > 0.0f) ? sAuto_A / fSumW_auto_A : 0.0f;
                    mAuto_B = (fSumW_auto_B > 0.0f) ? sAuto_B / fSumW_auto_B : 0.0f;
                }

                /* Fill the four pre-weighted lane buffers (slot/weight layout
                 * matches the original: AB slot0=B_AB slot1=A_AB; AA=auto·A;
                 * BB=auto·B). */
                float *pAB_B = &packAB_B[(size_t)batch * numel];
                float *pAB_A = &packAB_A[(size_t)batch * numel];
                float *pAA_A = &packAA_A[(size_t)batch * numel];
                float *pBB_B = &packBB_B[(size_t)batch * numel];
                float eAA = 0.0f, eBB = 0.0f;
                for (int i = 0; i < numel; ++i) {
                    pAB_B[i] = (fRawB[i] - mAB_B) * fWindowWeightB_AB[i];
                    pAB_A[i] = (fRawA[i] - mAB_A) * fWindowWeightA_AB[i];
                    pAA_A[i] = (fRawA[i] - mAuto_A) * fAutoWeightA[i];
                    pBB_B[i] = (fRawB[i] - mAuto_B) * fAutoWeightB[i];
                    eAA += pAA_A[i] * pAA_A[i];
                    eBB += pBB_B[i] * pBB_B[i];
                }
                energyAA[batch] = eAA;
                energyBB[batch] = eBB;
                batch++;

                if (batch == LANES) {
                    triple_flush_batch(pb, LANES, numel,
                        packAB_B, packAB_A, packAA_A, packBB_B, packC,
                        outAB, outAA, outBB, out_h, out_w, start_y, start_x, W,
                        bPerPairNorm, energyAA, energyBB);
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
                    outAB, outAA, outBB, out_h, out_w, start_y, start_x, W,
                    bPerPairNorm, energyAA, energyBB);
            }
        } /* end window loop */

    triple_cleanup:
        if (pb) codelet_plan_destroy_batched(pb);
        free(fRawA); free(fRawB);
        free(packAB_B); free(packAB_A); free(packAA_A); free(packBB_B); free(packC);
        free(energyAA); free(energyBB);
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
