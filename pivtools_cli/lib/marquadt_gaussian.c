// marquadt_gaussian.c
// Optimized Stacked Gaussian fitting using Levenberg-Marquardt algorithm via GSL
// For ensemble PIV correlation plane fitting

#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>

/* number of model parameters: 3 amplitudes + 3 offsets + 6 sigmas + 4 positions */
static const size_t P_PARAMS = 16;

struct fit_data {
    size_t n;        /* points per plane */
    const double *X1;
    const double *X2;
    const double *y; /* length 3*n: [AA; BB; AB] */
};

/* * Helper: Computes derivatives of the Cholesky inverse elements w.r.t Sigmas
 * This avoids re-computing the chain rule constants N times inside the loop.
 */
struct cholesky_derivs {
    double dL00_dsx, dL10_dsx, dL11_dsx;
    double dL00_dsy, dL10_dsy, dL11_dsy;
    double dL00_dsxy, dL10_dsxy, dL11_dsxy;
};

static void calc_sigma_derivs(double sx, double sy, double sxy,
                              double L00, double L10, double L11,
                              struct cholesky_derivs *d) {
    // Basic terms
    double inv_sx = 1.0 / sx;
    double term = sxy * inv_sx; // sxy / sx

    // 1. Derivatives w.r.t SX
    d->dL00_dsx = -0.5 * L00 * inv_sx;
    // For L11: it depends on (sy - sxy^2/sx). Let R = sy - sxy^2/sx.
    // L11 = R^-0.5. dL11/dR = -0.5 * L11^3.
    // dR/dsx = sxy^2 / sx^2.
    // dL11/dsx = (-0.5 * L11*L11*L11) * (term * term);
    double dL11_dR = -0.5 * L11 * L11 * L11;
    d->dL11_dsx = dL11_dR * (term * term);

    // For L10: L10 = -term * L11. Product rule.
    // dL10/dsx = -( d(term)/dsx * L11 + term * dL11/dsx )
    // d(term)/dsx = -sxy / sx^2 = -term / sx
    d->dL10_dsx = - ( (-term * inv_sx) * L11 + term * d->dL11_dsx );

    // 2. Derivatives w.r.t SY
    d->dL00_dsy = 0.0;
    // dR/dsy = 1.
    d->dL11_dsy = dL11_dR; // * 1
    d->dL10_dsy = -term * d->dL11_dsy;

    // 3. Derivatives w.r.t SXY
    d->dL00_dsxy = 0.0;
    // dR/dsxy = -2 * sxy / sx = -2 * term
    d->dL11_dsxy = dL11_dR * (-2.0 * term);
    // d(term)/dsxy = 1/sx
    d->dL10_dsxy = - ( inv_sx * L11 + term * d->dL11_dsxy );
}

/* * The Objective Function (Optimized)
 * Calculates the residuals for all 3 planes simultaneously.
 */
static int gauss2d_stacked_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct fit_data *d = (struct fit_data *)data;
    size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;
    const double *y = d->y;

    // --- 1. Unpack Parameters with Stability Clamping ---
    // Amplitudes [0-2]
    double amp_A = gsl_vector_get(x, 0);   if (amp_A < 0.0) amp_A = 1e-5;
    double amp_B = gsl_vector_get(x, 1);   if (amp_B < 0.0) amp_B = 1e-5;
    double amp_AB = gsl_vector_get(x, 2);  if (amp_AB < 0.0) amp_AB = 1e-5;

    // Offsets [3-5] (unconstrained, only numerical stability clamping)
    double c_A = gsl_vector_get(x, 3);
    double c_B = gsl_vector_get(x, 4);
    double c_AB = gsl_vector_get(x, 5);

    // Sigma A (autocorrelation) [6-8]
    double sx_A = gsl_vector_get(x, 6);    if (sx_A < 1e-9) sx_A = 0.01;
    double sy_A = gsl_vector_get(x, 7);    if (sy_A < 1e-9) sy_A = 0.01;
    double sxy_A = gsl_vector_get(x, 8);

    // Sigma AB (cross-correlation) [9-11]
    double sx_AB = gsl_vector_get(x, 9);   if (sx_AB < 1e-6) sx_AB = 0.01;
    double sy_AB = gsl_vector_get(x, 10);  if (sy_AB < 1e-6) sy_AB = 0.01;
    double sxy_AB = gsl_vector_get(x, 11); if (sxy_AB < 1e-12) sxy_AB = 0.01;

    // Positions [12-15]
    double x0_A = gsl_vector_get(x, 12);
    double y0_A = gsl_vector_get(x, 13);
    double x0_AB = gsl_vector_get(x, 14);
    double y0_AB = gsl_vector_get(x, 15);

    // --- 2. Pre-calculate Matrix Inverses (Outside the loop) ---
    
    // Matrix A (For Plane A and Plane B)
    double sqrt_sx_A = sqrt(sx_A);
    double term_A = sxy_A / sqrt_sx_A;
    double LA_00 = sqrt_sx_A;
    double LA_10 = term_A;
    double rad_A = sy_A - term_A * term_A;
    double LA_11 = (rad_A > 0) ? sqrt(rad_A) : 1e-9; // Safety check

    // Inverse of Lower Triangular Matrix A
    double inv_LA_00 = 1.0 / LA_00;
    double inv_LA_11 = 1.0 / LA_11;
    double inv_LA_10 = -LA_10 * (inv_LA_00 * inv_LA_11); 

    // Matrix AB (For Plane AB - Cross Correlation)
    double sum_sx = sx_A + sx_AB;
    double sum_sxy = sxy_A + sxy_AB;
    double sum_sy = sy_A + sy_AB;
    double sqrt_sum_sx = sqrt(sum_sx);
    double term_AB = sum_sxy / sqrt_sum_sx;
    
    double LAB_00 = sqrt_sum_sx;
    double LAB_10 = term_AB;
    double rad_AB = sum_sy - term_AB * term_AB;
    double LAB_11 = (rad_AB > 0) ? sqrt(rad_AB) : 1e-9; // Safety check

    // Inverse of Lower Triangular Matrix AB
    double inv_LAB_00 = 1.0 / LAB_00;
    double inv_LAB_11 = 1.0 / LAB_11;
    double inv_LAB_10 = -LAB_10 * (inv_LAB_00 * inv_LAB_11);

    // --- 3. Loop over data points ---
    size_t i;
    for (i = 0; i < n; i++) {
        // --- Compute Shape A (Shared by Plane A and B) ---
        double dx_A = X1[i] - x0_A;
        double dy_A = X2[i] - y0_A;
        
        // Matrix multiplication: L_inv * dX
        double tA_0 = inv_LA_00 * dx_A;
        double tA_1 = inv_LA_10 * dx_A + inv_LA_11 * dy_A;
        
        // Quadratic form: x^T * Sigma^-1 * x
        double quad_A = tA_0 * tA_0 + tA_1 * tA_1;
        double exp_A = exp(-0.5 * quad_A);

        // --- Compute Shape AB (For Plane AB) ---
        double dx_AB = X1[i] - x0_AB;
        double dy_AB = X2[i] - y0_AB;

        double tAB_0 = inv_LAB_00 * dx_AB;
        double tAB_1 = inv_LAB_10 * dx_AB + inv_LAB_11 * dy_AB;
        
        double quad_AB = tAB_0 * tAB_0 + tAB_1 * tAB_1;
        double exp_AB = exp(-0.5 * quad_AB);

        // --- Set Residuals (model: amp * exp(...) + c) ---
        // Plane A (Auto-Corr 1)
        gsl_vector_set(f, i, (amp_A * exp_A + c_A) - y[i]);

        // Plane B (Auto-Corr 2) - Uses same shape A, different Amplitude and Offset
        gsl_vector_set(f, i + n, (amp_B * exp_A + c_B) - y[i + n]);

        // Plane AB (Cross-Corr) - Uses shape AB
        gsl_vector_set(f, i + 2 * n, (amp_AB * exp_AB + c_AB) - y[i + 2 * n]);
    }

    return GSL_SUCCESS;
}

/* * The Analytical Jacobian Function
 * Calculates df/d_params for all 3 planes simultaneously.
 */
static int gauss2d_stacked_df(const gsl_vector *x, void *data, gsl_matrix *J) {
    struct fit_data *d = (struct fit_data *)data;
    size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;

    // --- 1. Unpack Parameters (Same as function f) ---
    double amp_A = gsl_vector_get(x, 0);
    double amp_B = gsl_vector_get(x, 1);
    double amp_AB = gsl_vector_get(x, 2);

    double sx_A = gsl_vector_get(x, 6);    if (sx_A < 1e-9) sx_A = 0.01;
    double sy_A = gsl_vector_get(x, 7);    if (sy_A < 1e-9) sy_A = 0.01;
    double sxy_A = gsl_vector_get(x, 8);

    double sx_AB = gsl_vector_get(x, 9);   if (sx_AB < 1e-6) sx_AB = 0.01;
    double sy_AB = gsl_vector_get(x, 10);  if (sy_AB < 1e-6) sy_AB = 0.01;
    double sxy_AB = gsl_vector_get(x, 11); if (sxy_AB < 1e-12) sxy_AB = 0.01;

    double x0_A = gsl_vector_get(x, 12);
    double y0_A = gsl_vector_get(x, 13);
    double x0_AB = gsl_vector_get(x, 14);
    double y0_AB = gsl_vector_get(x, 15);

    // --- 2. Pre-calculate Matrices and Derivative Factors ---

    // Plane A/B Inverses
    double sqrt_sx_A = sqrt(sx_A);
    double term_A = sxy_A / sqrt_sx_A;
    double LA_00 = sqrt_sx_A;
    double LA_10 = term_A;
    double rad_A = sy_A - term_A * term_A;
    double LA_11 = (rad_A > 0) ? sqrt(rad_A) : 1e-9;

    double inv_LA_00 = 1.0 / LA_00;
    double inv_LA_11 = 1.0 / LA_11;
    double inv_LA_10 = -LA_10 * (inv_LA_00 * inv_LA_11);

    // Plane A/B Derivatives w.r.t Sigmas
    struct cholesky_derivs dA;
    calc_sigma_derivs(sx_A, sy_A, sxy_A, inv_LA_00, inv_LA_10, inv_LA_11, &dA);

    // Plane AB Inverses
    double sum_sx = sx_A + sx_AB;
    double sum_sxy = sxy_A + sxy_AB;
    double sum_sy = sy_A + sy_AB;
    double sqrt_sum_sx = sqrt(sum_sx);
    double term_AB = sum_sxy / sqrt_sum_sx;

    double LAB_00 = sqrt_sum_sx;
    double LAB_10 = term_AB;
    double rad_AB = sum_sy - term_AB * term_AB;
    double LAB_11 = (rad_AB > 0) ? sqrt(rad_AB) : 1e-9;

    double inv_LAB_00 = 1.0 / LAB_00;
    double inv_LAB_11 = 1.0 / LAB_11;
    double inv_LAB_10 = -LAB_10 * (inv_LAB_00 * inv_LAB_11);

    // Plane AB Derivatives w.r.t Sigmas
    // Note: Plane AB parameters are SX_AB, but the math uses (SX_A + SX_AB).
    // The derivative d(sum)/d(param) is 1.0, so the logic holds directly.
    struct cholesky_derivs dAB;
    calc_sigma_derivs(sum_sx, sum_sy, sum_sxy, inv_LAB_00, inv_LAB_10, inv_LAB_11, &dAB);

    // --- 3. Main Loop ---
    // Initialize Jacobian to zero
    gsl_matrix_set_zero(J);

    size_t i;
    for (i = 0; i < n; i++) {
        /* * --- PART 1: PLANE A/B (Shared Shape) ---
         */
        double dx = X1[i] - x0_A;
        double dy = X2[i] - y0_A;

        double t0 = inv_LA_00 * dx;
        double t1 = inv_LA_10 * dx + inv_LA_11 * dy;
        double exp_A_val = exp(-0.5 * (t0 * t0 + t1 * t1));

        // 1.1 Amplitudes (Linear)
        // d(AmpA)/dA = exp_val
        gsl_matrix_set(J, i, 0, exp_A_val);     // Plane A, Param 0 (Amp A)
        gsl_matrix_set(J, i + n, 1, exp_A_val); // Plane B, Param 1 (Amp B)

        // 1.2 Offsets (Linear)
        gsl_matrix_set(J, i, 3, 1.0);     // Plane A, Param 3 (Offset A)
        gsl_matrix_set(J, i + n, 4, 1.0); // Plane B, Param 4 (Offset B)

        // Common Factor for Shape Derivatives: -0.5 * Amp * Exp * dQ/dTheta
        // Note: dQ/dTheta = 2*t0*dt0 + 2*t1*dt1.
        // The 2 cancels the 0.5.
        // Factor = -Amp * Exp * (t0*dt0 + t1*dt1)
        double fact_A = -amp_A * exp_A_val;
        double fact_B = -amp_B * exp_A_val;

        // 1.3 Positions (x0_A, y0_A) - Indices 12, 13
        // dx/dx0 = -1 -> dt0/dx0 = -inv_LA_00
        double dt0_dx0 = -inv_LA_00;
        double dt1_dx0 = -inv_LA_10;
        double dQ_dx0 = t0 * dt0_dx0 + t1 * dt1_dx0; // (Half derivative)

        // dy/dy0 = -1
        double dt1_dy0 = -inv_LA_11;
        double dQ_dy0 = t1 * dt1_dy0; // t0 term is 0

        gsl_matrix_set(J, i, 12, fact_A * dQ_dx0);
        gsl_matrix_set(J, i, 13, fact_A * dQ_dy0);
        gsl_matrix_set(J, i + n, 12, fact_B * dQ_dx0);
        gsl_matrix_set(J, i + n, 13, fact_B * dQ_dy0);

        // 1.4 Sigmas (sx_A, sy_A, sxy_A) - Indices 6, 7, 8
        // Using pre-calculated chain rule: dt/dSigma = dx * dL/dSigma

        // d/dSx
        double dt0_dsx = dx * dA.dL00_dsx;
        double dt1_dsx = dx * dA.dL10_dsx + dy * dA.dL11_dsx;
        double val_dsx = t0 * dt0_dsx + t1 * dt1_dsx;

        gsl_matrix_set(J, i, 6, fact_A * val_dsx);
        gsl_matrix_set(J, i + n, 6, fact_B * val_dsx);

        // d/dSy
        double dt1_dsy = dx * dA.dL10_dsy + dy * dA.dL11_dsy;
        double val_dsy = t1 * dt1_dsy; // dt0 is 0

        gsl_matrix_set(J, i, 7, fact_A * val_dsy);
        gsl_matrix_set(J, i + n, 7, fact_B * val_dsy);

        // d/dSxy
        double dt1_dsxy = dx * dA.dL10_dsxy + dy * dA.dL11_dsxy;
        double val_dsxy = t1 * dt1_dsxy;

        gsl_matrix_set(J, i, 8, fact_A * val_dsxy);
        gsl_matrix_set(J, i + n, 8, fact_B * val_dsxy);


        /* * --- PART 2: PLANE AB (Cross Correlation) ---
         */
        size_t row_ab = i + 2 * n;
        double dx_ab = X1[i] - x0_AB;
        double dy_ab = X2[i] - y0_AB;

        double t0_ab = inv_LAB_00 * dx_ab;
        double t1_ab = inv_LAB_10 * dx_ab + inv_LAB_11 * dy_ab;
        double exp_AB_val = exp(-0.5 * (t0_ab * t0_ab + t1_ab * t1_ab));

        // 2.1 Amplitude & Offset
        gsl_matrix_set(J, row_ab, 2, exp_AB_val); // Param 2 (Amp AB)
        gsl_matrix_set(J, row_ab, 5, 1.0);        // Param 5 (Offset AB)

        double fact_AB = -amp_AB * exp_AB_val;

        // 2.2 Positions (x0_AB, y0_AB) - Indices 14, 15
        double dt0_dx0_ab = -inv_LAB_00;
        double dt1_dx0_ab = -inv_LAB_10;
        double dQ_dx0_ab = t0_ab * dt0_dx0_ab + t1_ab * dt1_dx0_ab;

        double dt1_dy0_ab = -inv_LAB_11;
        double dQ_dy0_ab = t1_ab * dt1_dy0_ab;

        gsl_matrix_set(J, row_ab, 14, fact_AB * dQ_dx0_ab);
        gsl_matrix_set(J, row_ab, 15, fact_AB * dQ_dy0_ab);

        // 2.3 Sigmas (sx_AB, sy_AB, sxy_AB) - Indices 9, 10, 11
        // Note: The math uses dAB struct which is derived from (SigmaA + SigmaAB)
        // d(SigmaTotal)/d(SigmaAB) = 1, so the logic is identical.

        // d/dSx
        double dt0_dsx_ab = dx_ab * dAB.dL00_dsx;
        double dt1_dsx_ab = dx_ab * dAB.dL10_dsx + dy_ab * dAB.dL11_dsx;
        gsl_matrix_set(J, row_ab, 9, fact_AB * (t0_ab * dt0_dsx_ab + t1_ab * dt1_dsx_ab));

        // d/dSy
        double dt1_dsy_ab = dx_ab * dAB.dL10_dsy + dy_ab * dAB.dL11_dsy;
        gsl_matrix_set(J, row_ab, 10, fact_AB * (t1_ab * dt1_dsy_ab));

        // d/dSxy
        double dt1_dsxy_ab = dx_ab * dAB.dL10_dsxy + dy_ab * dAB.dL11_dsxy;
        gsl_matrix_set(J, row_ab, 11, fact_AB * (t1_ab * dt1_dsxy_ab));

        // 2.4 CROSS COUPLING:
        // Plane AB shape ALSO depends on Plane A sigmas (Indices 6, 7, 8)
        // because SigmaTotal = SigmaA + SigmaAB.
        // Therefore, we must fill columns 6,7,8 for Row AB as well!
        gsl_matrix_set(J, row_ab, 6, fact_AB * (t0_ab * dt0_dsx_ab + t1_ab * dt1_dsx_ab));
        gsl_matrix_set(J, row_ab, 7, fact_AB * (t1_ab * dt1_dsy_ab));
        gsl_matrix_set(J, row_ab, 8, fact_AB * (t1_ab * dt1_dsxy_ab));
    }

    return GSL_SUCCESS;
}

/* Optional callback */
static void fit_callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w) {
    (void) params; (void) w; (void) iter;
}

int fit_stacked_gaussian(
    size_t n,
    const double *X1,
    const double *X2,
    const double *y,
    const double *initial_guess,
    double *out_params,
    int *out_status
) {
    if (!X1 || !X2 || !y || !initial_guess || !out_params || !out_status) {
        return 0;
    }

    int ret = 0;
    int info = 0;
    int status = 0;

    struct fit_data d;
    d.n = n;
    d.X1 = X1;
    d.X2 = X2;
    d.y = y;

    size_t m = 3 * n; // Total data points
    size_t p = P_PARAMS; // 16 parameters

    if (m < p) {
        if (out_status) *out_status = -1;
        return 0;
    }

    // Solver Configuration
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    
    // Accuracy settings (Trade-off: speed vs precision)
    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 0.0; 

    gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(T, &fdf_params, m, p);
    if (!work) return 0;

    gsl_vector *wts = gsl_vector_alloc(m);
    if (!wts) {
        gsl_multifit_nlinear_free(work);
        return 0;
    }
    gsl_vector_set_all(wts, 1.0);

    // Initialize parameters
    gsl_vector_view xv = gsl_vector_view_array((double *)initial_guess, p);

    // Set up function structure
    gsl_multifit_nlinear_fdf fdf;
    fdf.f = gauss2d_stacked_f;
    fdf.df = gauss2d_stacked_df;  // Analytical Jacobian for faster convergence
    fdf.fvv = NULL;
    fdf.n = m;
    fdf.p = p;
    fdf.params = &d;

    // Initialize solver
    gsl_multifit_nlinear_winit(&xv.vector, wts, &fdf, work);

    // Iterate
    status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, fit_callback, NULL, &info, work);

    // Extract results
    gsl_vector *x_out = gsl_multifit_nlinear_position(work);
    for (size_t i = 0; i < p; ++i) {
        out_params[i] = gsl_vector_get(x_out, i);
    }

    if (out_status) *out_status = status;
    ret = 1; // Success

    // Cleanup
    gsl_vector_free(wts);
    gsl_multifit_nlinear_free(work);

    return ret;
}

/* * Export wrapper for Python (ctypes), MATLAB, etc.
 * Ensures correct symbol visibility.
 */
#ifdef __GNUC__
__attribute__((visibility("default")))
#endif
int fit_stacked_gaussian_export(size_t n, const double *X1, const double *X2, const double *y, const double *initial_guess, double *out_params, int *out_status) {
    return fit_stacked_gaussian(n, X1, X2, y, initial_guess, out_params, out_status);
}

/**
 * Batch fitting of multiple windows with OpenMP parallelization.
 *
 * This function processes multiple correlation plane windows in parallel,
 * significantly improving throughput compared to sequential per-window calls.
 *
 * Parameters:
 *   num_windows   : Number of windows to process
 *   n_per_window  : Number of data points per window (h * w)
 *   X1            : X1 grid coordinates (shared across all windows), length n_per_window
 *   X2            : X2 grid coordinates (shared across all windows), length n_per_window
 *   y_all         : Concatenated correlation data [AA|BB|AB] for all windows
 *                   Shape: (num_windows * 3 * n_per_window,)
 *                   Layout: window0_AA, window0_BB, window0_AB, window1_AA, ...
 *   initial_guesses: Initial parameter guesses for all windows
 *                   Shape: (num_windows * P_PARAMS,)
 *   out_params    : Output fitted parameters for all windows
 *                   Shape: (num_windows * P_PARAMS,)
 *   out_statuses  : Output status codes for all windows
 *                   Shape: (num_windows,)
 *
 * Returns:
 *   Number of successfully fitted windows
 */
#ifdef __GNUC__
__attribute__((visibility("default")))
#endif
int fit_stacked_gaussian_batch_export(
    size_t num_windows,
    size_t n_per_window,
    const double *X1,
    const double *X2,
    const double *y_all,
    const double *initial_guesses,
    double *out_params,
    int *out_statuses
) {
    int success_count = 0;
    size_t corr_size = 3 * n_per_window;  // AA + BB + AB per window

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:success_count) schedule(dynamic, 16)
    #endif
    for (size_t i = 0; i < num_windows; i++) {
        // Pointers into the flattened arrays for this window
        const double *y_window = y_all + i * corr_size;
        const double *guess_window = initial_guesses + i * P_PARAMS;
        double *params_window = out_params + i * P_PARAMS;
        int *status_window = out_statuses + i;

        int ret = fit_stacked_gaussian(
            n_per_window,
            X1,  // Shared X1 grid
            X2,  // Shared X2 grid
            y_window,
            guess_window,
            params_window,
            status_window
        );

        if (ret && *status_window == GSL_SUCCESS) {
            success_count++;
        }
    }

    return success_count;
}