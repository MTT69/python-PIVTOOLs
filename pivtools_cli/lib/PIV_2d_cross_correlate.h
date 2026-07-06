#ifndef PIV_2D_XCORR_H
#define PIV_2D_XCORR_H
#include <stdbool.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/**** function declarations ****/

EXPORT unsigned char bulkxcorr2d(const float *fImageA, const float *fImageB,const float *fMask, const int *nImageSize, int N_images,
                           const float *fWinCtrsX, const float *fWinCtrsY, const int *nWindows, float * fWindowWeightA, bool bEnsemble,
                           const float *fWindowWeightB, const int *nWindowSize, int nPeaks, int iPeakFinder,
                           float *fPkLocX, float *fPkLocY, float *fPkHeight, float *fSx, float *fSy, float *fSxy, float *fCorrelPlane_Out);

/**
 * Ensemble-optimized cross-correlation with internal accumulation.
 *
 * Unlike bulkxcorr2d which outputs N separate correlation planes,
 * this function accumulates across all N images internally and
 * outputs only the SUM (one plane per window).
 *
 * Output size: nWindows[0] * nWindows[1] * nFitWindowSize[0] * nFitWindowSize[1]
 * (or nWindowSize if nFitWindowSize is NULL)
 * (NOT multiplied by N_images)
 *
 * Loop structure: Parallel over windows, sequential over images.
 * Each thread owns its output region - no atomics needed.
 */
EXPORT unsigned char bulkxcorr2d_accumulate(
    const float *fImageA_stack,      /* Input: (N, H, W) flattened */
    const float *fImageB_stack,      /* Input: (N, H, W) flattened */
    const float *fMask,              /* Window mask (nWindows total) */
    const int *nImageSize,           /* [H, W] */
    int N_images,                    /* Number of image pairs */
    const float *fWinCtrsX,          /* Window center X coords */
    const float *fWinCtrsY,          /* Window center Y coords */
    const int *nWindows,             /* [n_win_y, n_win_x] */
    const float *fWindowWeightA,     /* Taper weights for image A */
    const float *fWindowWeightB,     /* Taper weights for image B */
    const int *nWindowSize,          /* [corr_h, corr_w] - FFT computation size */
    const int *nFitWindowSize,       /* [out_h, out_w] - output size, NULL = use nWindowSize */
    float *fCorrelPlane_Sum          /* Output: accumulated correlation planes */
);

/**
 * Fused triple cross-correlation with internal accumulation.
 *
 * Computes AB (cross), AA and BB (auto) correlations in one pass per
 * window per image.  FFT(A) and FFT(B) are computed once and reused
 * for all three products — 3 forward FFTs instead of 6.
 *
 * AB uses asymmetric weights (fWindowWeightA_AB, fWindowWeightB_AB).
 * AA/BB use symmetric weights (fAutoWeightA, fAutoWeightB) so that
 * the particle image sigma is correctly estimated.
 *
 * Three separate output buffers: fCorrAB_Sum, fCorrAA_Sum, fCorrBB_Sum.
 */
EXPORT unsigned char bulkxcorr2d_accumulate_triple(
    const float *fImageA_stack,
    const float *fImageB_stack,
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
    float       *fCorrAB_Sum,
    float       *fCorrAA_Sum,
    float       *fCorrBB_Sum
);

EXPORT float fminvec(const float *fVec, int n);
EXPORT float fmaxvec(const float *fVec, int n);

/* Benchmark sub-kernel timing (flag-gated, additive). Splits the per-window FFT
 * cross-correlation from the LM peak-fit inside bulkxcorr2d so the codelet-vs-FFTW
 * A/B can isolate the FFT speedup. Off by default (zero cost). Bound from the
 * benchmark harness via ctypes; production never enables it. Times are total
 * thread-seconds (sum across OpenMP threads), so they reconcile to wall-clock
 * only at OMP_NUM_THREADS=1. */
EXPORT void bulkxcorr2d_set_timing_enabled(int on);
EXPORT void bulkxcorr2d_reset_timing(void);
EXPORT void bulkxcorr2d_get_timing(double *fft_s, double *fit_s);

/* Peak-fit implementation selector: 0 = scalar (default), 1 = lockstep batch
 * (one window per SIMD lane, see peak_locate_lm_batch.h). set returns 0 on
 * success, -1 if batch was requested but is not compiled in (plain MSVC cl
 * build) — the caller must treat -1 as a hard error, never fall back
 * silently. Scope: the batch path serves instantaneous nPeaks==1 LM fits
 * (iPeakFinder >= 4); other calls use the scalar path regardless. */
EXPORT int bulkxcorr2d_set_peakfit_impl(int impl);
EXPORT int bulkxcorr2d_get_peakfit_impl(void);
EXPORT int bulkxcorr2d_peakfit_batch_available(void);

#endif