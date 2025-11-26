// marquadt_gaussian.c
// Optimized Stacked Gaussian fitting using Levenberg-Marquardt algorithm via GSL
// For ensemble PIV correlation plane fitting
//
// BATCH VERSION with Analytical Jacobians for maximum performance

#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>

/* number of model parameters */
static const size_t P_PARAMS = 13;

struct fit_data {
    size_t n;        /* points per plane */
    const double *X1;
    const double *X2;
    const double *y; /* length 3*n: [AA; BB; AB] */
};

/* Helper: Unpack parameters with stability clamping */
static void unpack_params(const gsl_vector *x,
    double *amp_A, double *amp_B, double *amp_AB,
    double *sx_A, double *sy_A, double *sxy_A,
    double *sx_AB, double *sy_AB, double *sxy_AB,
    double *x0_A, double *y0_A, double *x0_AB, double *y0_AB)
{
    *amp_A = gsl_vector_get(x, 0);   if (*amp_A < 0.0) *amp_A = 1e-5;
    *amp_B = gsl_vector_get(x, 1);   if (*amp_B < 0.0) *amp_B = 1e-5;
    *amp_AB = gsl_vector_get(x, 2);  if (*amp_AB < 0.0) *amp_AB = 1e-5;

    *sx_A = gsl_vector_get(x, 3);    if (*sx_A < 1e-9) *sx_A = 0.01;
    *sy_A = gsl_vector_get(x, 4);    if (*sy_A < 1e-9) *sy_A = 0.01;
    *sxy_A = gsl_vector_get(x, 5);

    *sx_AB = gsl_vector_get(x, 6);   if (*sx_AB < 1e-6) *sx_AB = 0.01;
    *sy_AB = gsl_vector_get(x, 7);   if (*sy_AB < 1e-6) *sy_AB = 0.01;
    *sxy_AB = gsl_vector_get(x, 8);

    *x0_A = gsl_vector_get(x, 9);
    *y0_A = gsl_vector_get(x, 10);
    *x0_AB = gsl_vector_get(x, 11);
    *y0_AB = gsl_vector_get(x, 12);
}

/* Helper: Compute Cholesky decomposition L and its inverse for covariance matrix
 * Covariance matrix Σ = [sx, sxy; sxy, sy]
 * Cholesky: Σ = L * L^T where L is lower triangular
 */
static void compute_cholesky_inv(double sx, double sy, double sxy,
    double *L00, double *L10, double *L11,
    double *invL00, double *invL10, double *invL11)
{
    double sqrt_sx = sqrt(sx);
    double term = sxy / sqrt_sx;
    double rad = sy - term * term;

    *L00 = sqrt_sx;
    *L10 = term;
    *L11 = (rad > 0) ? sqrt(rad) : 1e-9;

    *invL00 = 1.0 / (*L00);
    *invL11 = 1.0 / (*L11);
    *invL10 = -(*L10) * (*invL00) * (*invL11);
}

/* The Objective Function (Optimized)
 * Calculates the residuals for all 3 planes simultaneously.
 */
static int gauss2d_stacked_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct fit_data *d = (struct fit_data *)data;
    size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;
    const double *y = d->y;

    // Unpack parameters
    double amp_A, amp_B, amp_AB;
    double sx_A, sy_A, sxy_A;
    double sx_AB, sy_AB, sxy_AB;
    double x0_A, y0_A, x0_AB, y0_AB;

    unpack_params(x, &amp_A, &amp_B, &amp_AB,
                  &sx_A, &sy_A, &sxy_A,
                  &sx_AB, &sy_AB, &sxy_AB,
                  &x0_A, &y0_A, &x0_AB, &y0_AB);

    // Cholesky for Matrix A (Plane A and B autocorrelation)
    double LA_00, LA_10, LA_11, inv_LA_00, inv_LA_10, inv_LA_11;
    compute_cholesky_inv(sx_A, sy_A, sxy_A,
                         &LA_00, &LA_10, &LA_11,
                         &inv_LA_00, &inv_LA_10, &inv_LA_11);

    // Cholesky for Matrix AB (Cross Correlation: Σ_A + Σ_AB)
    double sum_sx = sx_A + sx_AB;
    double sum_sy = sy_A + sy_AB;
    double sum_sxy = sxy_A + sxy_AB;

    double LAB_00, LAB_10, LAB_11, inv_LAB_00, inv_LAB_10, inv_LAB_11;
    compute_cholesky_inv(sum_sx, sum_sy, sum_sxy,
                         &LAB_00, &LAB_10, &LAB_11,
                         &inv_LAB_00, &inv_LAB_10, &inv_LAB_11);

    // Loop over data points
    for (size_t i = 0; i < n; i++) {
        // Shape A (Shared by Plane A and B)
        double dx_A = X1[i] - x0_A;
        double dy_A = X2[i] - y0_A;
        double tA_0 = inv_LA_00 * dx_A;
        double tA_1 = inv_LA_10 * dx_A + inv_LA_11 * dy_A;
        double quad_A = tA_0 * tA_0 + tA_1 * tA_1;
        double exp_A = exp(-0.5 * quad_A);

        // Shape AB (Cross Correlation)
        double dx_AB = X1[i] - x0_AB;
        double dy_AB = X2[i] - y0_AB;
        double tAB_0 = inv_LAB_00 * dx_AB;
        double tAB_1 = inv_LAB_10 * dx_AB + inv_LAB_11 * dy_AB;
        double quad_AB = tAB_0 * tAB_0 + tAB_1 * tAB_1;
        double exp_AB = exp(-0.5 * quad_AB);

        // Residuals: model - data
        gsl_vector_set(f, i, amp_A * exp_A - y[i]);
        gsl_vector_set(f, i + n, amp_B * exp_A - y[i + n]);
        gsl_vector_set(f, i + 2 * n, amp_AB * exp_AB - y[i + 2 * n]);
    }

    return GSL_SUCCESS;
}

/* Analytical Jacobian for the stacked Gaussian model
 *
 * Parameters (13):
 *   [0] amp_A, [1] amp_B, [2] amp_AB
 *   [3] sx_A, [4] sy_A, [5] sxy_A
 *   [6] sx_AB, [7] sy_AB, [8] sxy_AB
 *   [9] x0_A, [10] y0_A, [11] x0_AB, [12] y0_AB
 *
 * Jacobian matrix J is (3n x 13) where:
 *   Rows 0 to n-1: Plane A (autocorrelation 1)
 *   Rows n to 2n-1: Plane B (autocorrelation 2)
 *   Rows 2n to 3n-1: Plane AB (cross-correlation)
 */
static int gauss2d_stacked_jac(const gsl_vector *x, void *data, gsl_matrix *J) {
    struct fit_data *d = (struct fit_data *)data;
    size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;

    // Unpack parameters
    double amp_A, amp_B, amp_AB;
    double sx_A, sy_A, sxy_A;
    double sx_AB, sy_AB, sxy_AB;
    double x0_A, y0_A, x0_AB, y0_AB;

    unpack_params(x, &amp_A, &amp_B, &amp_AB,
                  &sx_A, &sy_A, &sxy_A,
                  &sx_AB, &sy_AB, &sxy_AB,
                  &x0_A, &y0_A, &x0_AB, &y0_AB);

    // Cholesky for Matrix A
    double LA_00, LA_10, LA_11, inv_LA_00, inv_LA_10, inv_LA_11;
    compute_cholesky_inv(sx_A, sy_A, sxy_A,
                         &LA_00, &LA_10, &LA_11,
                         &inv_LA_00, &inv_LA_10, &inv_LA_11);

    // Cholesky for Matrix AB (Σ_A + Σ_AB)
    double sum_sx = sx_A + sx_AB;
    double sum_sy = sy_A + sy_AB;
    double sum_sxy = sxy_A + sxy_AB;

    double LAB_00, LAB_10, LAB_11, inv_LAB_00, inv_LAB_10, inv_LAB_11;
    compute_cholesky_inv(sum_sx, sum_sy, sum_sxy,
                         &LAB_00, &LAB_10, &LAB_11,
                         &inv_LAB_00, &inv_LAB_10, &inv_LAB_11);

    // Pre-compute derivatives of Cholesky factors w.r.t. covariance parameters
    // For A: L_A depends on sx_A, sy_A, sxy_A
    // dL00/dsx = 0.5/sqrt(sx)
    // dL10/dsx = -0.5*sxy/sx^(3/2)
    // dL10/dsxy = 1/sqrt(sx)
    // dL11/dsx, dL11/dsy, dL11/dsxy via chain rule

    double dLA00_dsx = 0.5 / LA_00;  // 0.5/sqrt(sx_A)
    double dLA10_dsx = -0.5 * sxy_A / (sx_A * LA_00);  // -0.5*sxy_A/sx_A^(3/2)
    double dLA10_dsxy = 1.0 / LA_00;  // 1/sqrt(sx_A)

    // L11 = sqrt(sy - (sxy/sqrt(sx))^2) = sqrt(sy - sxy^2/sx)
    // dL11/dsx = 0.5 * sxy^2 / (sx^2 * L11)
    // dL11/dsy = 0.5 / L11
    // dL11/dsxy = -sxy / (sx * L11)
    double dLA11_dsx = 0.5 * sxy_A * sxy_A / (sx_A * sx_A * LA_11);
    double dLA11_dsy = 0.5 / LA_11;
    double dLA11_dsxy = -sxy_A / (sx_A * LA_11);

    // Same for AB matrix (sum covariances)
    double dLAB00_dsx = 0.5 / LAB_00;
    double dLAB10_dsx = -0.5 * sum_sxy / (sum_sx * LAB_00);
    double dLAB10_dsxy = 1.0 / LAB_00;
    double dLAB11_dsx = 0.5 * sum_sxy * sum_sxy / (sum_sx * sum_sx * LAB_11);
    double dLAB11_dsy = 0.5 / LAB_11;
    double dLAB11_dsxy = -sum_sxy / (sum_sx * LAB_11);

    // Derivatives of inverse Cholesky
    // invL00 = 1/L00 => d(invL00)/dsx = -dL00/dsx / L00^2
    // invL11 = 1/L11 => d(invL11)/dsy etc.
    // invL10 = -L10 * invL00 * invL11

    double dinvLA00_dsx = -dLA00_dsx / (LA_00 * LA_00);
    double dinvLA11_dsx = -dLA11_dsx / (LA_11 * LA_11);
    double dinvLA11_dsy = -dLA11_dsy / (LA_11 * LA_11);
    double dinvLA11_dsxy = -dLA11_dsxy / (LA_11 * LA_11);

    // invL10 = -L10 / (L00 * L11)
    // d(invL10)/dsx = -[dL10/dsx * L00*L11 - L10*(dL00/dsx*L11 + L00*dL11/dsx)] / (L00*L11)^2
    double L00L11_A = LA_00 * LA_11;
    double dinvLA10_dsx = -(dLA10_dsx * L00L11_A - LA_10 * (dLA00_dsx * LA_11 + LA_00 * dLA11_dsx)) / (L00L11_A * L00L11_A);
    double dinvLA10_dsy = LA_10 * LA_00 * dLA11_dsy / (L00L11_A * L00L11_A);
    double dinvLA10_dsxy = -(dLA10_dsxy * L00L11_A - LA_10 * LA_00 * dLA11_dsxy) / (L00L11_A * L00L11_A);

    // Same for AB
    double dinvLAB00_dsx = -dLAB00_dsx / (LAB_00 * LAB_00);
    double dinvLAB11_dsx = -dLAB11_dsx / (LAB_11 * LAB_11);
    double dinvLAB11_dsy = -dLAB11_dsy / (LAB_11 * LAB_11);
    double dinvLAB11_dsxy = -dLAB11_dsxy / (LAB_11 * LAB_11);

    double L00L11_AB = LAB_00 * LAB_11;
    double dinvLAB10_dsx = -(dLAB10_dsx * L00L11_AB - LAB_10 * (dLAB00_dsx * LAB_11 + LAB_00 * dLAB11_dsx)) / (L00L11_AB * L00L11_AB);
    double dinvLAB10_dsy = LAB_10 * LAB_00 * dLAB11_dsy / (L00L11_AB * L00L11_AB);
    double dinvLAB10_dsxy = -(dLAB10_dsxy * L00L11_AB - LAB_10 * LAB_00 * dLAB11_dsxy) / (L00L11_AB * L00L11_AB);

    // Loop over data points
    for (size_t i = 0; i < n; i++) {
        // ===== PLANE A & B (use same Gaussian shape, different amplitudes) =====
        double dx_A = X1[i] - x0_A;
        double dy_A = X2[i] - y0_A;

        double tA_0 = inv_LA_00 * dx_A;
        double tA_1 = inv_LA_10 * dx_A + inv_LA_11 * dy_A;
        double quad_A = tA_0 * tA_0 + tA_1 * tA_1;
        double exp_A = exp(-0.5 * quad_A);

        double f_A = amp_A * exp_A;
        double f_B = amp_B * exp_A;

        // Derivatives of quad_A w.r.t. t's
        // dquad/dt0 = 2*t0, dquad/dt1 = 2*t1

        // Derivatives of t w.r.t. position (x0_A, y0_A)
        // t0 = invL00 * dx => dt0/dx0 = -invL00
        // t1 = invL10 * dx + invL11 * dy => dt1/dx0 = -invL10, dt1/dy0 = -invL11
        double dt0_dx0_A = -inv_LA_00;
        double dt1_dx0_A = -inv_LA_10;
        double dt1_dy0_A = -inv_LA_11;

        // dquad/dx0 = 2*t0*dt0/dx0 + 2*t1*dt1/dx0
        double dquad_dx0_A = 2 * (tA_0 * dt0_dx0_A + tA_1 * dt1_dx0_A);
        double dquad_dy0_A = 2 * tA_1 * dt1_dy0_A;

        // df/dx0 = A * exp * (-0.5) * dquad/dx0 = -0.5 * f * dquad/dx0
        double df_dx0_A = -0.5 * f_A * dquad_dx0_A;
        double df_dy0_A = -0.5 * f_A * dquad_dy0_A;
        double df_dx0_B = -0.5 * f_B * dquad_dx0_A;
        double df_dy0_B = -0.5 * f_B * dquad_dy0_A;

        // Derivatives w.r.t. covariance parameters (sx_A, sy_A, sxy_A)
        // t0 = invL00 * dx => dt0/dsx = d(invL00)/dsx * dx
        // t1 = invL10 * dx + invL11 * dy => dt1/dsx = d(invL10)/dsx * dx + d(invL11)/dsx * dy
        double dt0_dsx_A = dinvLA00_dsx * dx_A;
        double dt1_dsx_A = dinvLA10_dsx * dx_A + dinvLA11_dsx * dy_A;
        double dt1_dsy_A = dinvLA10_dsy * dx_A + dinvLA11_dsy * dy_A;
        double dt1_dsxy_A = dinvLA10_dsxy * dx_A + dinvLA11_dsxy * dy_A;

        double dquad_dsx_A = 2 * (tA_0 * dt0_dsx_A + tA_1 * dt1_dsx_A);
        double dquad_dsy_A = 2 * tA_1 * dt1_dsy_A;
        double dquad_dsxy_A = 2 * tA_1 * dt1_dsxy_A;

        double df_dsx_A = -0.5 * f_A * dquad_dsx_A;
        double df_dsy_A = -0.5 * f_A * dquad_dsy_A;
        double df_dsxy_A = -0.5 * f_A * dquad_dsxy_A;
        double df_dsx_B = -0.5 * f_B * dquad_dsx_A;
        double df_dsy_B = -0.5 * f_B * dquad_dsy_A;
        double df_dsxy_B = -0.5 * f_B * dquad_dsxy_A;

        // Jacobian for Plane A (row i)
        gsl_matrix_set(J, i, 0, exp_A);      // df_A/d(amp_A)
        gsl_matrix_set(J, i, 1, 0.0);        // df_A/d(amp_B)
        gsl_matrix_set(J, i, 2, 0.0);        // df_A/d(amp_AB)
        gsl_matrix_set(J, i, 3, df_dsx_A);   // df_A/d(sx_A)
        gsl_matrix_set(J, i, 4, df_dsy_A);   // df_A/d(sy_A)
        gsl_matrix_set(J, i, 5, df_dsxy_A);  // df_A/d(sxy_A)
        gsl_matrix_set(J, i, 6, 0.0);        // df_A/d(sx_AB)
        gsl_matrix_set(J, i, 7, 0.0);        // df_A/d(sy_AB)
        gsl_matrix_set(J, i, 8, 0.0);        // df_A/d(sxy_AB)
        gsl_matrix_set(J, i, 9, df_dx0_A);   // df_A/d(x0_A)
        gsl_matrix_set(J, i, 10, df_dy0_A);  // df_A/d(y0_A)
        gsl_matrix_set(J, i, 11, 0.0);       // df_A/d(x0_AB)
        gsl_matrix_set(J, i, 12, 0.0);       // df_A/d(y0_AB)

        // Jacobian for Plane B (row i+n) - same shape, different amplitude
        gsl_matrix_set(J, i + n, 0, 0.0);        // df_B/d(amp_A)
        gsl_matrix_set(J, i + n, 1, exp_A);      // df_B/d(amp_B)
        gsl_matrix_set(J, i + n, 2, 0.0);        // df_B/d(amp_AB)
        gsl_matrix_set(J, i + n, 3, df_dsx_B);   // df_B/d(sx_A)
        gsl_matrix_set(J, i + n, 4, df_dsy_B);   // df_B/d(sy_A)
        gsl_matrix_set(J, i + n, 5, df_dsxy_B);  // df_B/d(sxy_A)
        gsl_matrix_set(J, i + n, 6, 0.0);        // df_B/d(sx_AB)
        gsl_matrix_set(J, i + n, 7, 0.0);        // df_B/d(sy_AB)
        gsl_matrix_set(J, i + n, 8, 0.0);        // df_B/d(sxy_AB)
        gsl_matrix_set(J, i + n, 9, df_dx0_B);   // df_B/d(x0_A)
        gsl_matrix_set(J, i + n, 10, df_dy0_B);  // df_B/d(y0_A)
        gsl_matrix_set(J, i + n, 11, 0.0);       // df_B/d(x0_AB)
        gsl_matrix_set(J, i + n, 12, 0.0);       // df_B/d(y0_AB)

        // ===== PLANE AB (Cross Correlation) =====
        double dx_AB = X1[i] - x0_AB;
        double dy_AB = X2[i] - y0_AB;

        double tAB_0 = inv_LAB_00 * dx_AB;
        double tAB_1 = inv_LAB_10 * dx_AB + inv_LAB_11 * dy_AB;
        double quad_AB = tAB_0 * tAB_0 + tAB_1 * tAB_1;
        double exp_AB = exp(-0.5 * quad_AB);
        double f_AB = amp_AB * exp_AB;

        // Position derivatives (x0_AB, y0_AB)
        double dt0_dx0_AB = -inv_LAB_00;
        double dt1_dx0_AB = -inv_LAB_10;
        double dt1_dy0_AB = -inv_LAB_11;

        double dquad_dx0_AB = 2 * (tAB_0 * dt0_dx0_AB + tAB_1 * dt1_dx0_AB);
        double dquad_dy0_AB = 2 * tAB_1 * dt1_dy0_AB;

        double dfAB_dx0 = -0.5 * f_AB * dquad_dx0_AB;
        double dfAB_dy0 = -0.5 * f_AB * dquad_dy0_AB;

        // Covariance derivatives for AB
        // The AB Gaussian uses sum covariance (Σ_A + Σ_AB)
        // So derivatives w.r.t. sx_A affect AB through sum_sx = sx_A + sx_AB
        // Similarly for sx_AB

        // Derivatives w.r.t. sx_A (affects AB through sum_sx)
        double dt0_dsx_A_AB = dinvLAB00_dsx * dx_AB;
        double dt1_dsx_A_AB = dinvLAB10_dsx * dx_AB + dinvLAB11_dsx * dy_AB;
        double dquad_dsx_A_AB = 2 * (tAB_0 * dt0_dsx_A_AB + tAB_1 * dt1_dsx_A_AB);
        double dfAB_dsx_A = -0.5 * f_AB * dquad_dsx_A_AB;

        double dt1_dsy_A_AB = dinvLAB10_dsy * dx_AB + dinvLAB11_dsy * dy_AB;
        double dquad_dsy_A_AB = 2 * tAB_1 * dt1_dsy_A_AB;
        double dfAB_dsy_A = -0.5 * f_AB * dquad_dsy_A_AB;

        double dt1_dsxy_A_AB = dinvLAB10_dsxy * dx_AB + dinvLAB11_dsxy * dy_AB;
        double dquad_dsxy_A_AB = 2 * tAB_1 * dt1_dsxy_A_AB;
        double dfAB_dsxy_A = -0.5 * f_AB * dquad_dsxy_A_AB;

        // Derivatives w.r.t. sx_AB, sy_AB, sxy_AB (same effect as sx_A etc on sum)
        // The derivatives of L_AB w.r.t sum_sx are the same as w.r.t sx_A
        double dfAB_dsx_AB = dfAB_dsx_A;   // same sensitivity since sum_sx = sx_A + sx_AB
        double dfAB_dsy_AB = dfAB_dsy_A;
        double dfAB_dsxy_AB = dfAB_dsxy_A;

        // Jacobian for Plane AB (row i+2n)
        gsl_matrix_set(J, i + 2*n, 0, 0.0);           // df_AB/d(amp_A)
        gsl_matrix_set(J, i + 2*n, 1, 0.0);           // df_AB/d(amp_B)
        gsl_matrix_set(J, i + 2*n, 2, exp_AB);        // df_AB/d(amp_AB)
        gsl_matrix_set(J, i + 2*n, 3, dfAB_dsx_A);    // df_AB/d(sx_A)
        gsl_matrix_set(J, i + 2*n, 4, dfAB_dsy_A);    // df_AB/d(sy_A)
        gsl_matrix_set(J, i + 2*n, 5, dfAB_dsxy_A);   // df_AB/d(sxy_A)
        gsl_matrix_set(J, i + 2*n, 6, dfAB_dsx_AB);   // df_AB/d(sx_AB)
        gsl_matrix_set(J, i + 2*n, 7, dfAB_dsy_AB);   // df_AB/d(sy_AB)
        gsl_matrix_set(J, i + 2*n, 8, dfAB_dsxy_AB);  // df_AB/d(sxy_AB)
        gsl_matrix_set(J, i + 2*n, 9, 0.0);           // df_AB/d(x0_A)
        gsl_matrix_set(J, i + 2*n, 10, 0.0);          // df_AB/d(y0_A)
        gsl_matrix_set(J, i + 2*n, 11, dfAB_dx0);     // df_AB/d(x0_AB)
        gsl_matrix_set(J, i + 2*n, 12, dfAB_dy0);     // df_AB/d(y0_AB)
    }

    return GSL_SUCCESS;
}

/* Batch fitting function
 * Processes multiple windows in a single call, reusing GSL workspace.
 *
 * Parameters:
 *   num_windows      - Number of windows to fit
 *   n                - Points per plane (win_h * win_w)
 *   X1, X2           - Grid coordinates (shared, length n)
 *   y_batch          - Stacked correlation data for all windows
 *                      Layout: [win0_AA, win0_BB, win0_AB, win1_AA, win1_BB, win1_AB, ...]
 *                      Total length: num_windows * 3 * n
 *   initial_guess_batch - Initial parameters for all windows (num_windows * 13)
 *   out_params_batch    - Output parameters (num_windows * 13)
 *   out_status_batch    - Output status codes (num_windows)
 *
 * Returns: 1 on success, 0 on allocation failure
 */
static int fit_stacked_gaussian_batch_internal(
    size_t num_windows,
    size_t n,
    const double *X1,
    const double *X2,
    const double *y_batch,
    const double *initial_guess_batch,
    double *out_params_batch,
    int *out_status_batch
) {
    if (num_windows == 0) return 1;
    if (!X1 || !X2 || !y_batch || !initial_guess_batch || !out_params_batch || !out_status_batch) {
        return 0;
    }

    size_t m = 3 * n;  // Total data points per window
    size_t p = P_PARAMS;  // 13 parameters

    if (m < p) {
        // Not enough data points - mark all as failed
        for (size_t w = 0; w < num_windows; w++) {
            out_status_batch[w] = -1;
        }
        return 0;
    }

    // Solver configuration
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();

    // Accuracy settings
    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 0.0;

    // Allocate workspace ONCE for all windows
    gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(T, &fdf_params, m, p);
    if (!work) return 0;

    gsl_vector *wts = gsl_vector_alloc(m);
    if (!wts) {
        gsl_multifit_nlinear_free(work);
        return 0;
    }
    gsl_vector_set_all(wts, 1.0);

    // Fit data structure (will be updated per window)
    struct fit_data d;
    d.n = n;
    d.X1 = X1;
    d.X2 = X2;

    // Function definition with analytical Jacobian
    gsl_multifit_nlinear_fdf fdf;
    fdf.f = gauss2d_stacked_f;
    fdf.df = gauss2d_stacked_jac;  // Analytical Jacobian
    fdf.fvv = NULL;
    fdf.n = m;
    fdf.p = p;
    fdf.params = &d;

    // Process each window
    for (size_t w = 0; w < num_windows; w++) {
        // Point to this window's data (no copy)
        const double *y_window = y_batch + w * m;
        const double *guess = initial_guess_batch + w * p;
        double *out = out_params_batch + w * p;
        int *status = out_status_batch + w;

        // Update fit_data to point to this window's correlation data
        d.y = y_window;

        // Initialize solver with this window's guess
        gsl_vector_view xv = gsl_vector_view_array((double *)guess, p);
        gsl_multifit_nlinear_winit(&xv.vector, wts, &fdf, work);

        // Run solver
        int info = 0;
        int gsl_status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, NULL, NULL, &info, work);

        // Extract results
        gsl_vector *x_out = gsl_multifit_nlinear_position(work);
        for (size_t i = 0; i < p; ++i) {
            out[i] = gsl_vector_get(x_out, i);
        }

        *status = gsl_status;
    }

    // Cleanup (ONCE for all windows)
    gsl_vector_free(wts);
    gsl_multifit_nlinear_free(work);

    return 1;
}

/* Export wrapper for Python (ctypes), MATLAB, etc.
 * Ensures correct symbol visibility.
 */
#ifdef __GNUC__
__attribute__((visibility("default")))
#endif
int fit_stacked_gaussian_batch(
    size_t num_windows,
    size_t n,
    const double *X1,
    const double *X2,
    const double *y_batch,
    const double *initial_guess_batch,
    double *out_params_batch,
    int *out_status_batch
) {
    return fit_stacked_gaussian_batch_internal(
        num_windows, n, X1, X2, y_batch,
        initial_guess_batch, out_params_batch, out_status_batch
    );
}
