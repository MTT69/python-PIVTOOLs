// marquadt_gaussian.c
// Stacked Gaussian fitting using Levenberg-Marquardt algorithm via GSL
// For ensemble PIV correlation plane fitting

#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>

/* number of model parameters */
static const size_t P_PARAMS = 13;

struct fit_data {
    size_t n;       /* points per plane */
    const double *X1;
    const double *X2;
    const double *y; /* length 3*n: [AA; BB; AB] */
};

static int gauss2d_stacked_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct fit_data *d = (struct fit_data *)data;
    size_t n = d->n;
    const double *X1 = d->X1;
    const double *X2 = d->X2;
    const double *y = d->y;

    double amplitude_A = gsl_vector_get(x, 0);
    if (amplitude_A < 0.0) amplitude_A = 0.01;
    double amplitude_B = gsl_vector_get(x, 1);
    if (amplitude_B < 0.0) amplitude_B = 0.01;
    double amplitude_AB = gsl_vector_get(x, 2);
    if (amplitude_AB < 0.0) amplitude_AB = 0.01;

    double sigma_x_A = gsl_vector_get(x, 3);
    if (sigma_x_A < 1e-12) sigma_x_A = 0.01;
    double sigma_y_A = gsl_vector_get(x, 4);
    if (sigma_y_A < 1e-12) sigma_y_A = 0.01;
    double sigma_xy_A = gsl_vector_get(x, 5);

    double sigma_x_AB = gsl_vector_get(x, 6);
    if (sigma_x_AB < 1e-6) sigma_x_AB = 0.01;
    double sigma_y_AB = gsl_vector_get(x, 7);
    if (sigma_y_AB < 1e-6) sigma_y_AB = 0.01;

    double sigma_xy_AB = gsl_vector_get(x, 8);
    if (sigma_xy_AB < 1e-12) sigma_xy_AB = 0.01;

    double x0_A = gsl_vector_get(x, 9);
    double y0_A = gsl_vector_get(x, 10);
    double x0_AB = gsl_vector_get(x, 11);
    double y0_AB = gsl_vector_get(x, 12);

    size_t i;
    double *quad_form_A = (double *)malloc(n * sizeof(double));
    double *transformed_term_A = (double *)malloc(n * sizeof(double));
    double *quad_form_AB = (double *)malloc(n * sizeof(double));
    double *transformed_term_AB = (double *)malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        double X_A = X1[i] - x0_A;
        double Y_A = X2[i] - y0_A;
        double X_AB = X1[i] - x0_AB;
        double Y_AB = X2[i] - y0_AB;

        double LA[2][2] = {
            {sqrt(sigma_x_A), 0},
            {sigma_xy_A / sqrt(sigma_x_A), sqrt(sigma_y_A - (sigma_xy_A / sqrt(sigma_x_A)) * (sigma_xy_A / sqrt(sigma_x_A)))}
        };
        double LA_inv[2][2] = {
            {1 / LA[0][0], 0},
            {-LA[1][0] / (LA[0][0] * LA[1][1]), 1 / LA[1][1]}
        };

        double LAB[2][2] = {
            {sqrt(sigma_x_A + sigma_x_AB), 0},
            {(sigma_xy_A + sigma_xy_AB) / sqrt(sigma_x_A + sigma_x_AB), sqrt(sigma_y_A + sigma_y_AB - ((sigma_xy_A + sigma_xy_AB) / sqrt(sigma_x_A + sigma_x_AB)) * (sigma_xy_A + sigma_xy_AB) / sqrt(sigma_x_A + sigma_x_AB))}
        };
        double LAB_inv[2][2] = {
            {1 / LAB[0][0], 0},
            {-LAB[1][0] / (LAB[0][0] * LAB[1][1]), 1 / LAB[1][1]}
        };

        double term_A[2] = {X_A, Y_A};
        double term_AB[2] = {X_AB, Y_AB};

        transformed_term_A[i] = LA_inv[0][0] * term_A[0] + LA_inv[0][1] * term_A[1];
        quad_form_A[i] = transformed_term_A[i] * transformed_term_A[i] +
                        (LA_inv[1][0] * term_A[0] + LA_inv[1][1] * term_A[1]) *
                        (LA_inv[1][0] * term_A[0] + LA_inv[1][1] * term_A[1]);

        transformed_term_AB[i] = LAB_inv[0][0] * term_AB[0] + LAB_inv[0][1] * term_AB[1];
        quad_form_AB[i] = transformed_term_AB[i] * transformed_term_AB[i] +
                        (LAB_inv[1][0] * term_AB[0] + LAB_inv[1][1] * term_AB[1]) *
                        (LAB_inv[1][0] * term_AB[0] + LAB_inv[1][1] * term_AB[1]);

        double gauss_A = amplitude_A * exp(-0.5 * quad_form_A[i]);
        double gauss_B = amplitude_B * exp(-0.5 * quad_form_A[i]);
        double gauss_AB = amplitude_AB * exp(-0.5 * quad_form_AB[i]);

        gsl_vector_set(f, i, gauss_A - y[i]);
        gsl_vector_set(f, i + n, gauss_B - y[i + n]);
        gsl_vector_set(f, i + 2 * n, gauss_AB - y[i + 2 * n]);
    }

    free(quad_form_A);
    free(transformed_term_A);
    free(quad_form_AB);
    free(transformed_term_AB);
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

    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;

    size_t m = 3 * n;
    size_t p = P_PARAMS;
    if (m < p) {
        /* not enough data points to fit this many parameters */
        if (out_status) *out_status = -1;
        return 0;
    }
    gsl_matrix *covar = gsl_matrix_alloc(p, p);

    gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(T, &fdf_params, m, p);
    if (!work) goto cleanup;

    gsl_vector *x = gsl_vector_alloc(p);
    gsl_vector_view xv = gsl_vector_view_array((double *)initial_guess, p);
    gsl_vector_memcpy(x, &xv.vector);

    gsl_vector *wts = gsl_vector_alloc(m);
    if (!wts) goto cleanup;
    gsl_vector_set_all(wts, 1.0);

    fdf.f = gauss2d_stacked_f;
    fdf.df = NULL;
    fdf.fvv = NULL;
    fdf.n = m;
    fdf.p = p;
    fdf.params = &d;

    gsl_multifit_nlinear_winit(&xv.vector, wts, &fdf, work);

    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 0.0;
    status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, fit_callback, NULL, &info, work);

    /* compute Jacobian and covariance (optional) */
    gsl_matrix *J = gsl_multifit_nlinear_jac(work);
    if (J) {
        gsl_matrix *covar = gsl_matrix_alloc(p, p);
        if (covar != NULL) {
            int cov_status = gsl_multifit_nlinear_covar(J, 0.0, covar);
            (void)cov_status;
            gsl_matrix_free(covar);
        }
    }

    /* copy fitted params out */
    gsl_vector *x_out = gsl_multifit_nlinear_position(work);
    for (size_t i = 0; i < p; ++i) {
        out_params[i] = gsl_vector_get(x_out, i);
    }
    fflush(stderr);
    *out_status = status;

    ret = 1; /* success */

cleanup:
    if (wts) gsl_vector_free(wts);
    if (work) gsl_multifit_nlinear_free(work);
    if (x) gsl_vector_free(x);
    return ret;
}

#ifdef __GNUC__
__attribute__((visibility("default")))
#endif
int fit_stacked_gaussian_export(size_t n, const double *X1, const double *X2, const double *y, const double *initial_guess, double *out_params, int *out_status) {
    return fit_stacked_gaussian(n, X1, X2, y, initial_guess, out_params, out_status);
}
