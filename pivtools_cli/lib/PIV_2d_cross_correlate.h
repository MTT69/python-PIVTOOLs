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
 * Output size: nWindows[0] * nWindows[1] * nWindowSize[0] * nWindowSize[1]
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
    const int *nWindowSize,          /* [corr_h, corr_w] */
    float *fCorrelPlane_Sum          /* Output: accumulated correlation planes */
);

EXPORT float fminvec(const float *fVec, int n);
EXPORT float fmaxvec(const float *fVec, int n);

#endif