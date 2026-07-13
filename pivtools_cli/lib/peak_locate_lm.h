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

/* Main peak localization function.
 * Search (since 2026-07-13, Westerweel/PIVware style): all 3x3 local maxima in
 * the quarter-rule region |disp| <= N/4 are candidates; the largest is fitted.
 * fPlaneWeight (nullable): loss-of-correlation compensation nPx/(W conv W),
 * applied ONLY to the extracted PKSIZE fit patch at its own plane coordinates —
 * detection always runs on the raw plane. NULL = no compensation.
 * Invalid results are NaN: peak_loc row/col = NaN (height 0, std_dev 0) when
 * the peak search fails (flat/border/no-local-max) OR when the LM fit fails to
 * converge (since 2026-07-06). A finite result is a trustworthy fit. */
PEAK_EXPORT void lsqpeaklocate_lm(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev, const float *fPlaneWeight);

/* Peak detectability (PIVware pwInterrogateDisplacement SNR type 2): ratio of
 * the tallest to the second-tallest 3x3 local maximum, over the SAME candidate
 * set as the lsqpeaklocate_lm search (quarter region, positive, non-strict
 * local max). No candidate or no second positive candidate -> 0 (PIVware
 * convention). Diagnostic only — never used as a gate in C. */
PEAK_EXPORT float peak_detectability(const float *xcorr, const int *N);

/* Returns 1 if this CPU can run the SIMD kernels this binary was built with.
 * Lives in the scalar TU (compiled with no arch flags on every platform) so
 * calling it is always safe, even on a CPU below the wheel's ISA floor. */
PEAK_EXPORT int pivtools_cpu_supported(void);

#endif
