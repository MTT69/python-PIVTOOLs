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

/* number of model parameters */
static const size_t P_PARAMS = 13;

struct fit_data {
    size_t n;        /* points per plane */
    const double *X1;
    const double *X2;
    const double *y; /* length 3*n: [AA; BB; AB] */
};

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
    double amp_A = gsl_vector_get(x, 0);   if (amp_A < 0.0) amp_A = 1e-5;
    double amp_B = gsl_vector_get(x, 1);   if (amp_B < 0.0) amp_B = 1e-5;
    double amp_AB = gsl_vector_get(x, 2);  if (amp_AB < 0.0) amp_AB = 1e-5;
    
    double sx_A = gsl_vector_get(x, 3);    if (sx_A < 1e-9) sx_A = 0.01;
    double sy_A = gsl_vector_get(x, 4);    if (sy_A < 1e-9) sy_A = 0.01;
    double sxy_A = gsl_vector_get(x, 5);

    double sx_AB = gsl_vector_get(x, 6);   if (sx_AB < 1e-6) sx_AB = 0.01;
    double sy_AB = gsl_vector_get(x, 7);   if (sy_AB < 1e-6) sy_AB = 0.01;
    double sxy_AB = gsl_vector_get(x, 8);  if (sxy_AB < 1e-12) sxy_AB = 0.01;

    double x0_A = gsl_vector_get(x, 9);
    double y0_A = gsl_vector_get(x, 10);
    double x0_AB = gsl_vector_get(x, 11);
    double y0_AB = gsl_vector_get(x, 12);

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

        // --- Set Residuals ---
        // Plane A (Auto-Corr 1)
        gsl_vector_set(f, i, (amp_A * exp_A) - y[i]);
        
        // Plane B (Auto-Corr 2) - Uses same shape A, different Amplitude
        gsl_vector_set(f, i + n, (amp_B * exp_A) - y[i + n]);
        
        // Plane AB (Cross-Corr) - Uses shape AB
        gsl_vector_set(f, i + 2 * n, (amp_AB * exp_AB) - y[i + 2 * n]);
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
    size_t p = P_PARAMS; // 13 parameters

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
    fdf.df = NULL;  // Using numerical derivatives (simpler, plug-and-play)
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