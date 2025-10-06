#ifndef PEAK_LOCATE_LM_H
#define PEAK_LOCATE_LM_H

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

/* Main peak localization function - compatible with existing interface */
void lsqpeaklocate_lm(const float *xcorr, const int *N, float *peak_loc, int nPeaks, int iFitType, float *std_dev);

#endif
