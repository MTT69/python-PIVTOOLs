// marquadt_gaussian.h
// Header for stacked Gaussian fitting using Levenberg-Marquardt algorithm
// For ensemble PIV correlation plane fitting

#ifndef MARQUADT_GAUSSIAN_H
#define MARQUADT_GAUSSIAN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fit a stacked 2D Gaussian model to three correlation planes (AA, BB, AB).
 *
 * Uses GSL's Levenberg-Marquardt nonlinear least-squares solver to fit
 * a 13-parameter model to ensemble-averaged correlation data.
 *
 * Parameters (13 total):
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
 * @param n              Number of grid points per plane (win_h * win_w)
 * @param X1             Y-grid coordinates (length n)
 * @param X2             X-grid coordinates (length n)
 * @param y              Stacked data [AA; BB; AB] (length 3*n)
 * @param initial_guess  Initial parameter guess (length 13)
 * @param out_params     Output fitted parameters (length 13)
 * @param out_status     Output status code (GSL status)
 * @return               1 on success, 0 on failure
 */
int fit_stacked_gaussian_export(
    size_t n,
    const double *X1,
    const double *X2,
    const double *y,
    const double *initial_guess,
    double *out_params,
    int *out_status
);

#ifdef __cplusplus
}
#endif

#endif /* MARQUADT_GAUSSIAN_H */
