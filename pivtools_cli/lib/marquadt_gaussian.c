// marquadt_gaussian.c
// Optimized Stacked Gaussian fitting using Levenberg-Marquardt via GSL

#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(__WIN32__)
  #define PIV_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
  #define PIV_EXPORT __attribute__((visibility("default")))
#else
  #define PIV_EXPORT
#endif

#define P_PARAMS 16
#define EXTRACT_SIZE 32         // Extract 32x32 region around peak (full resolution)
#define SIGMA_MIN 1e-6          // Minimum variance to prevent singular matrices
#define FIT_TOL 1e-6            // Convergence tolerance (tightened for high accuracy)
#define MAX_ITER 50             // Max LM iterations (increased for tighter tolerance)

// Global flag to disable offset (+C) fitting for testing
// 0 = fit offsets normally, 1 = fix offsets to zero
static int g_disable_offset = 0;

// Global flag to mask center pixel in AA/BB autocorrelation
// 1 = mask center pixel (remove noise spike), 0 = include all pixels
static int g_mask_center = 1;

// Export function to toggle offset fitting mode
PIV_EXPORT void set_disable_offset(int disable) {
    g_disable_offset = disable;
}

// Export function to toggle center pixel masking
PIV_EXPORT void set_mask_center(int enable) {
    g_mask_center = enable;
}

struct fit_data {
    size_t n;
    const double *X1;
    const double *X2;
    const double *y;
    double weight_auto;  // Weight for AA/BB residuals: sqrt(sigma_AB / sigma_A)
    int center_idx;      // Index of center pixel to mask for AA/BB, or -1 to disable
};

struct cholesky_derivs {
    double dL00_dsx, dL10_dsx, dL11_dsx;
    double dL00_dsy, dL10_dsy, dL11_dsy;
    double dL00_dsxy, dL10_dsxy, dL11_dsxy;
};

static void calc_sigma_derivs(double sx, double sy, double sxy,
                              double L00, double L10, double L11,
                              struct cholesky_derivs *d) {
    double inv_sx = 1.0 / sx;
    double term = sxy * inv_sx;
    double dL11_dR = -0.5 * L11 * L11 * L11;

    d->dL00_dsx = -0.5 * L00 * inv_sx;
    d->dL11_dsx = dL11_dR * (term * term);
    d->dL10_dsx = -((-term * inv_sx) * L11 + term * d->dL11_dsx);

    d->dL00_dsy = 0.0;
    d->dL11_dsy = dL11_dR;
    d->dL10_dsy = -term * d->dL11_dsy;

    d->dL00_dsxy = 0.0;
    d->dL11_dsxy = dL11_dR * (-2.0 * term);
    d->dL10_dsxy = -(inv_sx * L11 + term * d->dL11_dsxy);
}

static inline double clamp_sigma(double s) {
    return (s < SIGMA_MIN) ? SIGMA_MIN : s;
}

static inline double safe_sqrt_rad(double rad) {
    return (rad > 0.0) ? sqrt(rad) : SIGMA_MIN;
}

static int gauss2d_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct fit_data *d = (struct fit_data *)data;
    const size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;
    const double *y = d->y;

    double amp_A  = fmax(gsl_vector_get(x, 0), 0.0);
    double amp_B  = fmax(gsl_vector_get(x, 1), 0.0);
    double amp_AB = fmax(gsl_vector_get(x, 2), 0.0);
    // When g_disable_offset is set, fix offsets to zero (don't read from params)
    double c_A    = g_disable_offset ? 0.0 : gsl_vector_get(x, 3);
    double c_B    = g_disable_offset ? 0.0 : gsl_vector_get(x, 4);
    double c_AB   = g_disable_offset ? 0.0 : gsl_vector_get(x, 5);
    double sx_A   = clamp_sigma(gsl_vector_get(x, 6));
    double sy_A   = clamp_sigma(gsl_vector_get(x, 7));
    double sxy_A  = gsl_vector_get(x, 8);
    double sx_AB  = clamp_sigma(gsl_vector_get(x, 9));
    double sy_AB  = clamp_sigma(gsl_vector_get(x, 10));
    double sxy_AB = gsl_vector_get(x, 11);
    double x0_A   = gsl_vector_get(x, 12);
    double y0_A   = gsl_vector_get(x, 13);
    double x0_AB  = gsl_vector_get(x, 14);
    double y0_AB  = gsl_vector_get(x, 15);

    // Cholesky decomposition for sigma_A (particle size - from autocorrelation)
    double sqrt_sx_A = sqrt(sx_A);
    double term_A = sxy_A / sqrt_sx_A;
    double LA_11 = safe_sqrt_rad(sy_A - term_A * term_A);
    double inv_LA_00 = 1.0 / sqrt_sx_A;
    double inv_LA_11 = 1.0 / LA_11;
    double inv_LA_10 = -term_A * inv_LA_00 * inv_LA_11;

    // RUNTIME CONSTRAINT: sigma_AB >= sigma_A (before Cholesky decomposition)
    // This prevents the optimizer from shrinking sigma_AB below sigma_A
    double sx_AB_clamped = fmax(sx_AB, sx_A);
    double sy_AB_clamped = fmax(sy_AB, sy_A);

    // Cholesky decomposition for sigma_AB (total cross-correlation width)
    // Uses sigma_AB DIRECTLY (no longer added to sigma_A)
    double sqrt_sx_AB = sqrt(sx_AB_clamped);
    double term_AB = sxy_AB / sqrt_sx_AB;
    double LAB_11 = safe_sqrt_rad(sy_AB_clamped - term_AB * term_AB);
    double inv_LAB_00 = 1.0 / sqrt_sx_AB;
    double inv_LAB_11 = 1.0 / LAB_11;
    double inv_LAB_10 = -term_AB * inv_LAB_00 * inv_LAB_11;

    for (size_t i = 0; i < n; i++) {
        double dx_A = X1[i] - x0_A;
        double dy_A = X2[i] - y0_A;
        double tA_0 = inv_LA_00 * dx_A;
        double tA_1 = inv_LA_10 * dx_A + inv_LA_11 * dy_A;
        double exp_A = exp(-0.5 * (tA_0 * tA_0 + tA_1 * tA_1));

        double dx_AB = X1[i] - x0_AB;
        double dy_AB = X2[i] - y0_AB;
        double tAB_0 = inv_LAB_00 * dx_AB;
        double tAB_1 = inv_LAB_10 * dx_AB + inv_LAB_11 * dy_AB;
        double exp_AB = exp(-0.5 * (tAB_0 * tAB_0 + tAB_1 * tAB_1));

        if ((int)i == d->center_idx) {
            // MASK: Zero residual for AA/BB center pixel (noise spike from self-correlation)
            gsl_vector_set(f, i,       0.0);
            gsl_vector_set(f, i + n,   0.0);
        } else {
            // WEIGHTED AA/BB residuals (variance normalization)
            gsl_vector_set(f, i,       d->weight_auto * (amp_A  * exp_A  + c_A  - y[i]));
            gsl_vector_set(f, i + n,   d->weight_auto * (amp_B  * exp_A  + c_B  - y[i + n]));
        }
        // AB residual ALWAYS computed (no self-noise in cross-correlation)
        gsl_vector_set(f, i + 2*n, amp_AB * exp_AB + c_AB - y[i + 2*n]);
    }
    return GSL_SUCCESS;
}

static int gauss2d_df(const gsl_vector *x, void *data, gsl_matrix *J) {
    struct fit_data *d = (struct fit_data *)data;
    const size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;

    double amp_A  = gsl_vector_get(x, 0);
    double amp_B  = gsl_vector_get(x, 1);
    double amp_AB = gsl_vector_get(x, 2);
    double sx_A   = clamp_sigma(gsl_vector_get(x, 6));
    double sy_A   = clamp_sigma(gsl_vector_get(x, 7));
    double sxy_A  = gsl_vector_get(x, 8);
    double sx_AB  = clamp_sigma(gsl_vector_get(x, 9));
    double sy_AB  = clamp_sigma(gsl_vector_get(x, 10));
    double sxy_AB = gsl_vector_get(x, 11);
    double x0_A   = gsl_vector_get(x, 12);
    double y0_A   = gsl_vector_get(x, 13);
    double x0_AB  = gsl_vector_get(x, 14);
    double y0_AB  = gsl_vector_get(x, 15);

    // Cholesky decomposition for sigma_A (particle size)
    double sqrt_sx_A = sqrt(sx_A);
    double term_A = sxy_A / sqrt_sx_A;
    double LA_11 = safe_sqrt_rad(sy_A - term_A * term_A);
    double inv_LA_00 = 1.0 / sqrt_sx_A;
    double inv_LA_11 = 1.0 / LA_11;
    double inv_LA_10 = -term_A * inv_LA_00 * inv_LA_11;

    struct cholesky_derivs dA;
    calc_sigma_derivs(sx_A, sy_A, sxy_A, inv_LA_00, inv_LA_10, inv_LA_11, &dA);

    // RUNTIME CONSTRAINT: sigma_AB >= sigma_A (same constraint as in gauss2d_f)
    double sx_AB_clamped = fmax(sx_AB, sx_A);
    double sy_AB_clamped = fmax(sy_AB, sy_A);

    // Cholesky decomposition for sigma_AB (total width, used DIRECTLY)
    double sqrt_sx_AB = sqrt(sx_AB_clamped);
    double term_AB = sxy_AB / sqrt_sx_AB;
    double LAB_11 = safe_sqrt_rad(sy_AB_clamped - term_AB * term_AB);
    double inv_LAB_00 = 1.0 / sqrt_sx_AB;
    double inv_LAB_11 = 1.0 / LAB_11;
    double inv_LAB_10 = -term_AB * inv_LAB_00 * inv_LAB_11;

    // Compute derivatives for sigma_AB (not sum - decoupled from sigma_A)
    struct cholesky_derivs dAB;
    calc_sigma_derivs(sx_AB_clamped, sy_AB_clamped, sxy_AB, inv_LAB_00, inv_LAB_10, inv_LAB_11, &dAB);

    gsl_matrix_set_zero(J);

    // Weight factor for AA/BB Jacobian entries (variance normalization)
    double w = d->weight_auto;

    for (size_t i = 0; i < n; i++) {
        double dx = X1[i] - x0_A;
        double dy = X2[i] - y0_A;
        double t0 = inv_LA_00 * dx;
        double t1 = inv_LA_10 * dx + inv_LA_11 * dy;
        double exp_A = exp(-0.5 * (t0*t0 + t1*t1));

        // MASK: Skip AA/BB Jacobian entries for center pixel (rows i and i+n stay zero)
        if ((int)i != d->center_idx) {
            // WEIGHTED AA/BB amplitude derivatives
            gsl_matrix_set(J, i, 0, w * exp_A);
            gsl_matrix_set(J, i + n, 1, w * exp_A);
            // Skip offset Jacobians if disabled (offsets fixed to zero)
            if (!g_disable_offset) {
                gsl_matrix_set(J, i, 3, w * 1.0);      // d/d(c_A) - WEIGHTED
                gsl_matrix_set(J, i + n, 4, w * 1.0);  // d/d(c_B) - WEIGHTED
            }

            // WEIGHTED prefactors for AA/BB
            double fact_A = w * (-amp_A * exp_A);
            double fact_B = w * (-amp_B * exp_A);

            double dQ_dx0 = t0 * (-inv_LA_00) + t1 * (-inv_LA_10);
            double dQ_dy0 = t1 * (-inv_LA_11);
            gsl_matrix_set(J, i, 12, fact_A * dQ_dx0);
            gsl_matrix_set(J, i, 13, fact_A * dQ_dy0);
            gsl_matrix_set(J, i + n, 12, fact_B * dQ_dx0);
            gsl_matrix_set(J, i + n, 13, fact_B * dQ_dy0);

            double val_dsx = t0 * (dx * dA.dL00_dsx) + t1 * (dx * dA.dL10_dsx + dy * dA.dL11_dsx);
            double val_dsy = t1 * (dx * dA.dL10_dsy + dy * dA.dL11_dsy);
            double val_dsxy = t1 * (dx * dA.dL10_dsxy + dy * dA.dL11_dsxy);
            gsl_matrix_set(J, i, 6, fact_A * val_dsx);
            gsl_matrix_set(J, i + n, 6, fact_B * val_dsx);
            gsl_matrix_set(J, i, 7, fact_A * val_dsy);
            gsl_matrix_set(J, i + n, 7, fact_B * val_dsy);
            gsl_matrix_set(J, i, 8, fact_A * val_dsxy);
            gsl_matrix_set(J, i + n, 8, fact_B * val_dsxy);
        }
        // If i == center_idx, rows i and i+n remain zero from gsl_matrix_set_zero

        // AB row ALWAYS computed (row i + 2*n) - no self-noise in cross-correlation
        size_t row_ab = i + 2 * n;
        double dx_ab = X1[i] - x0_AB;
        double dy_ab = X2[i] - y0_AB;
        double t0_ab = inv_LAB_00 * dx_ab;
        double t1_ab = inv_LAB_10 * dx_ab + inv_LAB_11 * dy_ab;
        double exp_AB = exp(-0.5 * (t0_ab*t0_ab + t1_ab*t1_ab));

        gsl_matrix_set(J, row_ab, 2, exp_AB);
        // Skip offset Jacobian for AB if disabled
        if (!g_disable_offset) {
            gsl_matrix_set(J, row_ab, 5, 1.0);  // d/d(c_AB)
        }

        double fact_AB = -amp_AB * exp_AB;

        double dQ_dx0_ab = t0_ab * (-inv_LAB_00) + t1_ab * (-inv_LAB_10);
        double dQ_dy0_ab = t1_ab * (-inv_LAB_11);
        gsl_matrix_set(J, row_ab, 14, fact_AB * dQ_dx0_ab);
        gsl_matrix_set(J, row_ab, 15, fact_AB * dQ_dy0_ab);

        double val_dsx_ab = t0_ab * (dx_ab * dAB.dL00_dsx) + t1_ab * (dx_ab * dAB.dL10_dsx + dy_ab * dAB.dL11_dsx);
        double val_dsy_ab = t1_ab * (dx_ab * dAB.dL10_dsy + dy_ab * dAB.dL11_dsy);
        double val_dsxy_ab = t1_ab * (dx_ab * dAB.dL10_dsxy + dy_ab * dAB.dL11_dsxy);

        // AB row only affects params [9-11] (sigma_AB) - NO coupling to sigma_A
        // DECOUPLING: Lines setting J[row_ab, 6/7/8] have been REMOVED
        // This ensures sigma_A is constrained only by AA/BB, sigma_AB only by AB
        gsl_matrix_set(J, row_ab, 9, fact_AB * val_dsx_ab);
        gsl_matrix_set(J, row_ab, 10, fact_AB * val_dsy_ab);
        gsl_matrix_set(J, row_ab, 11, fact_AB * val_dsxy_ab);
        // REMOVED: gsl_matrix_set(J, row_ab, 6, ...) - no coupling to sigma_A
        // REMOVED: gsl_matrix_set(J, row_ab, 7, ...) - no coupling to sigma_A
        // REMOVED: gsl_matrix_set(J, row_ab, 8, ...) - no coupling to sigma_A
    }
    return GSL_SUCCESS;
}

// --- Single fit using pre-allocated workspace (NO allocation per call) ---
static int fit_one_reuse(
    gsl_multifit_nlinear_workspace *work,
    size_t n,
    const double *X1,
    const double *X2,
    const double *y,
    double weight_auto,  // Weight for AA/BB residuals: sqrt(sigma_AB / sigma_A)
    int center_idx,      // Index of center pixel to mask for AA/BB, or -1 to disable
    const double *guess,
    double *result
) {
    struct fit_data d;
    d.n  = n;
    d.X1 = X1;
    d.X2 = X2;
    d.y  = y;
    d.weight_auto = weight_auto;
    d.center_idx = center_idx;  // Pass center index for AA/BB masking

    gsl_multifit_nlinear_fdf fdf;
    fdf.f      = gauss2d_f;
    fdf.df     = gauss2d_df;
    fdf.fvv    = NULL;
    fdf.n      = 3 * n;
    fdf.p      = P_PARAMS;
    fdf.params = &d;

    gsl_vector_view xv = gsl_vector_view_array((double *)guess, P_PARAMS);
    gsl_multifit_nlinear_init(&xv.vector, &fdf, work);

    int info;
    int status = gsl_multifit_nlinear_driver(MAX_ITER, FIT_TOL, FIT_TOL, FIT_TOL, NULL, NULL, &info, work);

    gsl_vector *x_out = gsl_multifit_nlinear_position(work);
    for (size_t i = 0; i < P_PARAMS; i++) {
        result[i] = gsl_vector_get(x_out, i);
    }

    // Clamp sigma values to ensure positive (unconstrained optimizer can go negative)
    // Indices: 6=sx_A, 7=sy_A, 9=sx_AB, 10=sy_AB
    result[6] = fmax(result[6], SIGMA_MIN);
    result[7] = fmax(result[7], SIGMA_MIN);
    result[9] = fmax(result[9], SIGMA_MIN);
    result[10] = fmax(result[10], SIGMA_MIN);

    // FINAL CONSTRAINT: sigma_AB >= sigma_A
    // The runtime constraint in gauss2d_f/df keeps the optimizer happy,
    // but the final output must also satisfy this constraint
    const double EPS = 0.01;  // Small margin to avoid numerical issues
    if (result[9] < result[6] + EPS) result[9] = result[6] + EPS;   // sx_AB >= sx_A
    if (result[10] < result[7] + EPS) result[10] = result[7] + EPS; // sy_AB >= sy_A
    // No constraint on sxy (cross-terms can be negative)

    return (status == GSL_SUCCESS || status == GSL_EMAXITER);
}

// --- Batch Export ---
// Note: win_height and win_width support rectangular windows (height != width)
PIV_EXPORT int fit_stacked_gaussian_batch_export(
    size_t num_windows,
    size_t n_per_window,
    size_t win_height,
    size_t win_width,
    const double *X1,
    const double *X2,
    const double *y_all,
    const double *initial_guesses,
    const double *weights_auto,  // Per-window weights: sqrt(sigma_AB / sigma_A)
    double *out_params,
    int *out_statuses
) {
    if (!X1 || !X2 || !y_all || !initial_guesses || !weights_auto || !out_params || !out_statuses) {
        fprintf(stderr, "[fit] NULL pointer argument\n");
        return 0;
    }

    int success_count = 0;
    int win_h = (int)win_height;
    int win_w = (int)win_width;

    // Determine extraction region size (min of EXTRACT_SIZE and window size for each dimension)
    int extract_h = (win_h < EXTRACT_SIZE) ? win_h : EXTRACT_SIZE;
    int extract_w = (win_w < EXTRACT_SIZE) ? win_w : EXTRACT_SIZE;
    size_t n_extract = (size_t)(extract_h * extract_w);

    fprintf(stderr, "[fit] %zu windows, %dx%d -> extracting %dx%d (%zu pts) around peak\n",
            num_windows, win_h, win_w, extract_h, extract_w, n_extract);

    // Log weight statistics for verification
    double w_min = weights_auto[0], w_max = weights_auto[0], w_sum = 0.0;
    for (size_t i = 0; i < num_windows; i++) {
        if (weights_auto[i] < w_min) w_min = weights_auto[i];
        if (weights_auto[i] > w_max) w_max = weights_auto[i];
        w_sum += weights_auto[i];
    }
    fprintf(stderr, "[fit] AA/BB weights: min=%.3f, max=%.3f, mean=%.3f\n",
            w_min, w_max, w_sum / num_windows);

    // Log center pixel masking status
    int center_row = extract_h / 2;
    int center_col = extract_w / 2;
    int center_idx_log = g_mask_center ? (center_row * extract_w + center_col) : -1;
    fprintf(stderr, "[fit] Center pixel masking: %s (idx=%d in %dx%d extraction)\n",
            g_mask_center ? "ENABLED" : "DISABLED", center_idx_log, extract_h, extract_w);

    gsl_set_error_handler_off();

    // GSL workspace params (shared, read-only)
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    // Use more robust solver settings for production
    fdf_params.solver = gsl_multifit_nlinear_solver_cholesky;
    fdf_params.scale = gsl_multifit_nlinear_scale_more;
    fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;  // Geodesic acceleration for better convergence

    #ifdef _OPENMP
    #pragma omp parallel reduction(+:success_count)
    {
        #pragma omp single
        {
            fprintf(stderr, "[fit] OpenMP threads: %d\n", omp_get_num_threads());
        }
    #else
        fprintf(stderr, "[fit] OpenMP threads: 1 (no OpenMP)\n");
    #endif
        // Thread-local buffers for extracted region
        double *ext_X1 = malloc(n_extract * sizeof(double));
        double *ext_X2 = malloc(n_extract * sizeof(double));
        double *ext_y = malloc(3 * n_extract * sizeof(double));

        // Thread-local GSL workspace
        gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(T, &fdf_params, 3 * n_extract, P_PARAMS);

        if (!ext_X1 || !ext_X2 || !ext_y || !work) {
            fprintf(stderr, "[fit] thread allocation failed\n");
        }

        int i;  /* Declared outside for MSVC OpenMP 2.0 compatibility */
        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 16)
        #endif
        for (i = 0; i < (int)num_windows; i++) {
            if (!ext_X1 || !ext_X2 || !ext_y || !work) {
                out_statuses[i] = 0;
                continue;
            }

            const double *y_win = y_all + (size_t)i * 3 * n_per_window;
            const double *guess = initial_guesses + (size_t)i * P_PARAMS;
            double *params = out_params + (size_t)i * P_PARAMS;

            // Get peak location from initial guess (x0_A, y0_A)
            double peak_x = guess[12];
            double peak_y = guess[13];

            // Convert to grid indices (coordinates are typically 1..win_h/win_w in 1-based)
            int peak_col = (int)(peak_x + 0.5);
            int peak_row = (int)(peak_y + 0.5);

            // Compute extraction region bounds (centered on peak)
            // Use separate half sizes for rectangular extraction regions
            int half_h = extract_h / 2;
            int half_w = extract_w / 2;
            int r_start = peak_row - half_h;
            int c_start = peak_col - half_w;

            // Clamp to window bounds (use win_h for rows, win_w for columns)
            if (r_start < 0) r_start = 0;
            if (c_start < 0) c_start = 0;
            if (r_start + extract_h > win_h) r_start = win_h - extract_h;
            if (c_start + extract_w > win_w) c_start = win_w - extract_w;

            // Extract region at full resolution (rectangular)
            size_t k = 0;
            for (int r = r_start; r < r_start + extract_h; r++) {
                for (int c = c_start; c < c_start + extract_w; c++) {
                    // Row-major indexing: row * width + col
                    size_t src = (size_t)(r * win_w + c);
                    ext_X1[k] = X1[src];
                    ext_X2[k] = X2[src];
                    ext_y[k]              = y_win[src];
                    ext_y[k + n_extract]   = y_win[src + n_per_window];
                    ext_y[k + 2*n_extract] = y_win[src + 2*n_per_window];
                    k++;
                }
            }

            // Get per-window weight for AA/BB residuals
            double w_auto = weights_auto[i];

            // Compute center pixel index for the extracted region (row-major order)
            // The extraction region is centered on the peak, so the center of the
            // extracted region is at (extract_h/2, extract_w/2) in local coordinates
            int center_row = extract_h / 2;
            int center_col = extract_w / 2;
            int center_idx = g_mask_center ? (center_row * extract_w + center_col) : -1;

            // Fit using extracted region (with weight and center masking)
            int ok = fit_one_reuse(work, n_extract, ext_X1, ext_X2, ext_y, w_auto, center_idx, guess, params);
            out_statuses[i] = ok;

            if (ok) {
                success_count++;
            } else {
                memcpy(params, guess, P_PARAMS * sizeof(double));
            }
        }

        // Free thread-local resources
        if (work) gsl_multifit_nlinear_free(work);
        free(ext_X1);
        free(ext_X2);
        free(ext_y);

    #ifdef _OPENMP
    }
    #endif

    fprintf(stderr, "[fit] completed: %d/%zu succeeded\n", success_count, num_windows);

    return success_count;
}

// Single-window wrapper (for backwards compatibility / testing)
// For square windows, pass win_height = win_width = sqrt(n)
PIV_EXPORT int fit_stacked_gaussian_export(
    size_t n,
    size_t win_height,
    size_t win_width,
    const double *X1,
    const double *X2,
    const double *y,
    const double *initial_guess,
    double weight_auto,  // Weight for AA/BB residuals: sqrt(sigma_AB / sigma_A)
    double *out_params,
    int *out_status
) {
    int status;
    int ret = fit_stacked_gaussian_batch_export(1, n, win_height, win_width, X1, X2, y, initial_guess, &weight_auto, out_params, &status);
    if (out_status) *out_status = status ? 0 : -1;
    return ret;
}