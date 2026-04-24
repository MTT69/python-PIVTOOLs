/*
 * stereo_coc_accumulate.h — Fused dual-camera cross-correlation + CoC accumulation.
 *
 * Computes per-camera ensemble cross-correlation (AB, AA, BB) for TWO cameras
 * plus per-frame Correlation-of-Correlations (CoC) between them, all in a
 * single fused pass per window per frame.
 *
 * The CoC cross-correlates per-frame AB correlation planes from camera 1 and
 * camera 2, accumulating the result.  This provides the cross-camera covariance
 * Sigma_12 needed to decouple R_xx and R_zz in stereo PIV Reynolds stress
 * reconstruction.
 *
 * Architecture:
 *   - OpenMP parallel over windows, sequential over frames
 *   - Single FFTW plan set per thread (same size for per-camera and CoC)
 *   - Sub-images extracted ONCE per camera per frame
 *   - AB planes mean-subtracted before CoC cross-correlation
 *   - CoC uses same circular FFT correlation as per-camera (no zero-padding)
 *   - Triple optimization: symmetric auto weights for AA/BB
 */

#ifndef STEREO_COC_ACCUMULATE_H
#define STEREO_COC_ACCUMULATE_H

#if defined(_WIN32) || defined(__WIN32__)
  #define PIV_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
  #define PIV_EXPORT __attribute__((visibility("default")))
#else
  #define PIV_EXPORT
#endif

/**
 * Fused stereo ensemble accumulation with Correlation-of-Correlations.
 *
 * For each non-masked window (OpenMP parallel):
 *   For each frame (sequential):
 *     1. Extract sub-images: A1, B1, A2, B2 from window region
 *     2. AB cross-correlation per camera (asymmetric weights)
 *     3. AA/BB auto-correlation per camera (symmetric weights)
 *     4. Extract central fit_window_size from AB1, AB2
 *     5. Mean-subtract extracted AB planes (removes DC pedestal)
 *     6. CoC: circular FFT cross-correlate mean-subtracted AB planes
 *     7. Accumulate all 7 planes (6 per-camera + 1 CoC)
 *
 * The CoC output has the SAME dimensions as the per-camera correlation
 * output (out_h × out_w), using circular correlation at the same size.
 * No zero-padding is needed because the CoC peak is at ~zero displacement.
 *
 * NOTE: Output buffers are NOT zeroed internally — caller must zero them
 * before the first call.
 *
 * @param fImage1A_stack  Camera 1 image A stack (N_images, H, W) float32
 * @param fImage1B_stack  Camera 1 image B stack
 * @param fImage2A_stack  Camera 2 image A stack
 * @param fImage2B_stack  Camera 2 image B stack
 * @param fMask           Per-window mask: 1=skip, 0=process
 * @param nImageSize      [H, W] dewarped image dimensions
 * @param N_images        Number of frame pairs in batch
 * @param fWinCtrsX       Window center X coordinates (n_win_x)
 * @param fWinCtrsY       Window center Y coordinates (n_win_y)
 * @param nWindows        [n_win_y, n_win_x]
 * @param fWindowWeightA_AB  AB taper weight for image A
 * @param fWindowWeightB_AB  AB taper weight for image B
 * @param fAutoWeightA    Auto-correlation taper for image A (symmetric)
 * @param fAutoWeightB    Auto-correlation taper for image B (symmetric)
 * @param nWindowSize     [corr_h, corr_w] FFT computation size
 * @param nFitWindowSize  [fit_h, fit_w] central extraction, or NULL for nWindowSize
 * @param fCorr1AB_Sum    Output: cam1 AB accumulated (nWindowsTotal * out_h * out_w)
 * @param fCorr1AA_Sum    Output: cam1 AA accumulated
 * @param fCorr1BB_Sum    Output: cam1 BB accumulated
 * @param fCorr2AB_Sum    Output: cam2 AB accumulated
 * @param fCorr2AA_Sum    Output: cam2 AA accumulated
 * @param fCorr2BB_Sum    Output: cam2 BB accumulated
 * @param fCoC_Sum        Output: CoC accumulated (nWindowsTotal * out_h * out_w)
 * @return                Error code (0 = success)
 */
PIV_EXPORT unsigned char bulkxcorr2d_stereo_coc_accumulate(
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
    float       *fDiag_CoC
);

#endif /* STEREO_COC_ACCUMULATE_H */
