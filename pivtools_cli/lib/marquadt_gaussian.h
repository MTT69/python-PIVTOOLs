// marquadt_gaussian.h
// Header for stacked Gaussian fitting using Levenberg-Marquardt algorithm
// For ensemble PIV correlation plane fitting
//
// BATCH VERSION with Analytical Jacobians for maximum performance

#ifndef MARQUADT_GAUSSIAN_H
#define MARQUADT_GAUSSIAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fit a stacked 2D Gaussian model to multiple correlation plane sets (batch mode).
 *
 * Uses GSL's Levenberg-Marquardt nonlinear least-squares solver with analytical
 * Jacobians to fit a 13-parameter model to ensemble-averaged correlation data.
 * Processes multiple windows in a single call for maximum efficiency.
 *
 * Parameters (13 total per window):
 *   [0]  amplitude_A    - Peak height in auto-correlation A
 *   [1]  amplitude_B    - Peak height in auto-correlation B
 *   [2]  amplitude_AB   - Peak height in cross-correlation AB
 *   [3]  sigma_x_A      - Variance (x-direction) for A
 *   [4]  sigma_y_A      - Variance (y-direction) for A
 *   [5]  sigma_xy_A     - Covariance for A
 *   [6]  sigma_x_AB     - Variance (x-direction) for predictor displacement
 *   [7]  sigma_y_AB     - Variance (y-direction) for predictor displacement
 *   [8]  sigma_xy_AB    - Covariance for predictor displacement
 *   [9]  x0_A           - X-center of A auto-correlation
 *   [10] y0_A           - Y-center of A auto-correlation
 *   [11] x0_AB          - X-displacement (cross-correlation peak)
 *   [12] y0_AB          - Y-displacement (cross-correlation peak)
 *
 * Data layout for y_batch:
 *   [win0_AA, win0_BB, win0_AB, win1_AA, win1_BB, win1_AB, ...]
 *   Each plane has n = win_h * win_w points
 *   Total length: num_windows * 3 * n
 *
 * @param num_windows          Number of windows to fit
 * @param n                    Number of grid points per plane (win_h * win_w)
 * @param X1                   Y-grid coordinates (length n, shared across windows)
 * @param X2                   X-grid coordinates (length n, shared across windows)
 * @param y_batch              Stacked correlation data for all windows
 * @param initial_guess_batch  Initial parameter guesses (length num_windows * 13)
 * @param out_params_batch     Output fitted parameters (length num_windows * 13)
 * @param out_status_batch     Output status codes (length num_windows)
 * @return                     1 on success, 0 on allocation failure
 */
int fit_stacked_gaussian_batch(
    size_t num_windows,
    size_t n,
    const double *X1,
    const double *X2,
    const double *y_batch,
    const double *initial_guess_batch,
    double *out_params_batch,
    int *out_status_batch
);

#ifdef __cplusplus
}
#endif

#endif /* MARQUADT_GAUSSIAN_H */
