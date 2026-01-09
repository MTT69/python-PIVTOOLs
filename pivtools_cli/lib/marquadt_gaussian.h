// marquadt_gaussian.h
// Header for optimized stacked Gaussian fitting (matrix-free Levenberg-Marquardt)
// For ensemble PIV correlation plane fitting

#ifndef MARQUADT_GAUSSIAN_H
#define MARQUADT_GAUSSIAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Batch fitting of multiple windows with OpenMP parallelization.
 *
 * Uses a matrix-free Levenberg-Marquardt solver that accumulates J^T*J and J^T*r
 * directly without storing the full Jacobian matrix. This keeps data in L1 cache
 * and achieves ~10-20x speedup over the GSL multifit approach.
 *
 * Supports rectangular windows (win_height != win_width).
 *
 * Model: f(x,y) = amplitude * exp(-0.5 * quadratic_form) + offset
 *
 * Parameters per window (16 total):
 *   [0]  amplitude_A    - Peak height in auto-correlation A
 *   [1]  amplitude_B    - Peak height in auto-correlation B
 *   [2]  amplitude_AB   - Peak height in cross-correlation AB
 *   [3]  c_A            - Background offset for A plane
 *   [4]  c_B            - Background offset for B plane
 *   [5]  c_AB           - Background offset for AB plane
 *   [6]  sigma_x_A      - Variance (x-direction) for A
 *   [7]  sigma_y_A      - Variance (y-direction) for A
 *   [8]  sigma_xy_A     - Covariance for A
 *   [9]  sigma_x_AB     - Variance (x-direction) for predictor displacement
 *   [10] sigma_y_AB     - Variance (y-direction) for predictor displacement
 *   [11] sigma_xy_AB    - Covariance for predictor displacement
 *   [12] x0_A           - X-center of A auto-correlation
 *   [13] y0_A           - Y-center of A auto-correlation
 *   [14] x0_AB          - X-displacement (cross-correlation peak)
 *   [15] y0_AB          - Y-displacement (cross-correlation peak)
 *
 * @param num_windows     Number of windows to fit
 * @param n_per_window    Number of grid points per plane (win_height * win_width)
 * @param win_height      Window height in pixels (rows)
 * @param win_width       Window width in pixels (columns)
 * @param X1              Y-grid coordinates (length n_per_window, shared)
 * @param X2              X-grid coordinates (length n_per_window, shared)
 * @param y_all           Stacked data for all windows (length num_windows * 3 * n_per_window)
 * @param initial_guesses Initial parameter guesses (length num_windows * 16)
 * @param out_params      Output fitted parameters (length num_windows * 16)
 * @param out_statuses    Output status codes (length num_windows), 1=success, 0=fail
 * @return                Number of successfully fitted windows
 */
int fit_stacked_gaussian_batch_export(
    size_t num_windows,
    size_t n_per_window,
    size_t win_height,
    size_t win_width,
    const double *X1,
    const double *X2,
    const double *y_all,
    const double *initial_guesses,
    double *out_params,
    int *out_statuses
);

#ifdef __cplusplus
}
#endif

#endif /* MARQUADT_GAUSSIAN_H */
