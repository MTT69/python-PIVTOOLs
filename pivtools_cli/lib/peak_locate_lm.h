#ifndef PEAK_LOCATE_LM_H
#define PEAK_LOCATE_LM_H

#ifdef _WIN32
#define PEAK_EXPORT __declspec(dllexport)
#else
#define PEAK_EXPORT
#endif

/**** defines ****/
/* Peak localization window size (odd numbers only: 3, 5, 7, 9, ...)
 * 5×5 is optimal for most PIV applications
 * 7×7 provides more robustness for noisy data but is ~2x slower
 * 3×3 is faster but less accurate
 */
#define PKSIZE_X    5
#define PKSIZE_Y    5

/******************************************************************************
 * Fast Levenberg-Marquardt peak localization
 *
 * Drop-in replacement for GSL-based lsqpeaklocate
 * Optimized for PIV correlation peak fitting with:
 * - No external dependencies (GSL-free)
 * - Direct Jacobian computation
 * - Fast convergence for PIV peaks
 * - Reduced iteration count
 *****************************************************************************/

/* Main peak localization function - compatible with existing interface.
 * Invalid results are NaN: peak_loc row/col = NaN (height 0, std_dev 0) when
 * the peak search fails (flat/border/non-max) OR when the LM fit fails to
 * converge (since 2026-07-06). A finite result is a trustworthy fit. */
PEAK_EXPORT void lsqpeaklocate_lm(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev);

#endif
