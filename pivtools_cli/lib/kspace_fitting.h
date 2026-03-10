// kspace_fitting.h
// K-Space Transfer Function Fitting for Ensemble PIV
//
// Fits k-space transfer function T(k) = exp(-2*pi^2 * k^T*Sigma*k) * exp(-2*pi*i * k.mu)
// to correlation planes in Fourier domain. Particle shape cancels algebraically via
// F_ref = sqrt(|F_AA| * |F_BB|), reducing 16 params to 5 (mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy).
//
// Uses float32 FFTs (fftwf) + double-precision GSL fitting, matching the existing
// libbulkxcorr2d (float32 FFT) and libmarquadt (double GSL) architecture.

#ifndef KSPACE_FITTING_H
#define KSPACE_FITTING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__WIN32__)
  #define PIV_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
  #define PIV_EXPORT __attribute__((visibility("default")))
#else
  #define PIV_EXPORT
#endif

/**
 * Batch k-space transfer function fitting with OpenMP parallelization.
 *
 * For each non-masked window:
 *   1. FFT correlation planes (float32 fftwf)
 *   2. Compute F_ref = sqrt(|F_AA| * |F_BB|)
 *   3. Joint noise fit: F_ref = (A*Gauss + N0) * P_noise  (4 params, GSL double)
 *   4. Initial guesses: sub-pixel peak (displacement), 1D log-regression (variance)
 *   5. Full 5-param fit of T_norm = T/T(0)  (GSL double)
 *   6. Validation and output in 16-element gauss_flat format
 *
 * Parameters per window (16 total, same as Gaussian fitter):
 *   [0]  amp_A        - Peak height in R_AA at center
 *   [1]  amp_B        - Peak height in R_BB at center
 *   [2]  amp_AB       - Peak height in R_AB (max)
 *   [3-5] c_A,c_B,c_AB - 0 (not used in k-space)
 *   [6-8] sig_A_x/y/xy - NaN (particle shape cancelled in k-space)
 *   [9]  Sigma_xx     - Reynolds stress (UU)
 *   [10] Sigma_yy     - Reynolds stress (VV)
 *   [11] Sigma_xy     - Reynolds stress (UV)
 *   [12] x0_A         - Window center x (1-based)
 *   [13] y0_A         - Window center y (1-based)
 *   [14] x0_AB        - center_x + mu_x (displacement)
 *   [15] y0_AB        - center_y + mu_y (displacement)
 *
 * Status codes:
 *   -1  Masked/skipped
 *    0  Success (negative variance clamped to zero)
 *    1  Optimizer did not converge / joint fit failed
 *    3  Displacement > 3/4 window
 *
 * @param num_windows      Number of windows
 * @param corr_h           Correlation plane height
 * @param corr_w           Correlation plane width
 * @param R_AA             Auto-correlation A planes (float32, flat: num_windows * corr_h * corr_w)
 * @param R_BB             Auto-correlation B planes (float32, flat)
 * @param R_AB             Cross-correlation AB planes (float32, flat)
 * @param mask             Per-window mask: 1=skip, 0=process (int32)
 * @param pred_disp        Per-window predictor displacement (num_windows*2: dy,dx) or NULL
 * @param interp_kernel    0=bicubic, 1=lanczos3
 * @param use_soft_weighting  1=anisotropic soft decay, 0=uniform
 * @param k_max_cap        Hard cap on k_max (<=0 = default 0.35)
 * @param out_params       Output: num_windows * 16 (double, gauss_flat format)
 * @param out_status       Output: num_windows status codes (int32)
 * @param out_initial_guess Output: num_windows * 16 (double)
 * @param out_diagnostics  Output: num_windows * 4 (double: snr, N0, k_max_x, k_max_y) or NULL
 * @return                 Number of successfully fitted windows
 */
PIV_EXPORT int fit_kspace_batch(
    size_t num_windows,
    size_t corr_h,
    size_t corr_w,
    const float  *R_AA,
    const float  *R_BB,
    const float  *R_AB,
    const int    *mask,
    const double *pred_disp,
    int    interp_kernel,
    int    use_soft_weighting,
    double k_max_cap,
    double *out_params,
    int    *out_status,
    double *out_initial_guess,
    double *out_diagnostics
);

#ifdef __cplusplus
}
#endif

#endif /* KSPACE_FITTING_H */
