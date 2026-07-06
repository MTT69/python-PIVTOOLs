#ifndef PEAK_LOCATE_LM_BATCH_H
#define PEAK_LOCATE_LM_BATCH_H

#include "peak_locate_lm.h"   /* PEAK_EXPORT, PKSIZE_X/Y */

/******************************************************************************
 * Batched (one-window-per-SIMD-lane) LM Gaussian peak fitter.
 *
 * Lockstep counterpart of lsqpeaklocate_lm: fits up to peakfit_batch_lanes()
 * correlation planes at once, one plane per SIMD lane, from a single
 * lane-width-generic source (peak_simd.h; 4-lane NEON / 8-lane AVX2 via
 * clang-cl / 16-lane AVX-512). Semantics mirror the scalar fitter exactly:
 * same peak search + validity gate (NaN sentinel), same 3-point seed and
 * clamps, same Marquardt schedule, same trust rule (converged-by-tolerance
 * OR residual <= LM_ACCEPT_RESID_FRAC * A^2 * npix — see peak_locate_lm.c),
 * same NaN sentinel on failure. The scalar fitter remains the oracle; gate
 * tests assert agreement (test_peakfit_gate.c, unit-tests/test_peakfit_batch.py).
 *
 * Scope (enforced by the caller in PIV_2d_cross_correlate.c): nPeaks == 1,
 * iFitType in {4, 5, 6}, instantaneous (non-ensemble) path only.
 *
 * This header is deliberately free of peak_simd.h so any TU can include it;
 * lane width is exposed via peakfit_batch_lanes().
 ******************************************************************************/

/* Upper bound on peakfit_batch_lanes() across all ISA routes (AVX-512 = 16).
 * Callers may size stack output arrays with this. */
#define PEAKFIT_MAX_LANES 16

/* Fit L_real planes (window-major, contiguous: planes + l*N[0]*N[1] is lane
 * l's plane — the packC layout of instant_flush_batch). Outputs are dense
 * component-major arrays of width peakfit_batch_lanes():
 *   peak_loc[c*W + l], c = 0 row / 1 col / 2 height
 *   std_dev [c*W + l], c = 0 sx  / 1 sy  / 2 sxy (6-DOF: variances)
 * Lanes l >= L_real are written with the NaN sentinel. Planes must already
 * carry the correlation-plane weighting (the caller hoists it). */
PEAK_EXPORT void lsqpeaklocate_lm_batch(const float *planes, int L_real,
                                        const int *N, int iFitType,
                                        float *peak_loc, float *std_dev);

/* 1 when the batch fitter is compiled in (clang/gcc vecext); 0 under plain
 * MSVC cl (PIVTOOLS_WIN_COMPILER=cl escape hatch) — selecting the batch
 * implementation then fails loudly, it never silently falls back. */
PEAK_EXPORT int peakfit_batch_available(void);

/* SIMD lane count the batch fitter was compiled for (0 when unavailable).
 * Must equal codelet_lanes() — the caller checks once at plan creation. */
PEAK_EXPORT int peakfit_batch_lanes(void);

#endif /* PEAK_LOCATE_LM_BATCH_H */
