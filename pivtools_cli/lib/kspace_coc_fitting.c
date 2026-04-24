/*
 * kspace_coc_fitting.c — Implicit model k-space fitting for CoC planes.
 *
 * Fits the CoC magnitude spectrum using an implicit model:
 *   |F(CoC)|(k) = F_ref(k) × A × exp(-2π² k^T Σ k)
 *
 * where F_ref(k) = √(|F(AA₁)|·|F(BB₁)|·|F(AA₂)|·|F(BB₂)|) is the
 * particle image envelope from the 4 auto-correlation planes.
 *
 * F_ref is multiplied INTO the model, not divided out of the data.
 * This handles DC=0 naturally (both |F(CoC)| and F_ref are ~0 at DC,
 * so the residual is 0−0=0) and cancels the particle image |P(k)|⁴
 * without explicit division.
 *
 * The fitted Σ is particle-free — directly gives var(d₁−d₂).
 *
 * Build (macOS):
 *   gcc-15 -O3 -fPIC -fopenmp -DFFTW_THREADS -shared \
 *     kspace_coc_fitting.c -o libkspace_coc.so \
 *     -lfftw3f -lgsl -lgslcblas -lm
 */

#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#include <fftw3.h>

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define COC_NPARAMS  4   /* A, Sxx, Syy, Sxy */
#define COC_MAX_ITER 200
#define XTOL 1e-8
#define GTOL 1e-8
#define FTOL 1e-8

/* Status codes */
#define STATUS_MASKED      -1
#define STATUS_SUCCESS      0
#define STATUS_NO_CONVERGE  1


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Implicit model with noise floor:                                          */
/*   |F_CoC|(k) = F_ref(k) × A × Gaussian(k) + N0                          */
/*                                                                             */
/* Params: [0]=A, [1]=Sxx, [2]=Syy, [3]=Sxy, [4]=N0                         */
/*                                                                             */
/* F_ref = √(|F(AA₁)|·|F(BB₁)|·|F(AA₂)|·|F(BB₂)|) cancels particle.       */
/* N0 captures the flat noise floor in |F(CoC)| from finite ensemble         */
/* averaging. At high k, F_ref×Gaussian→0 and |F(CoC)|→N0.                  */
/* No P_noise needed — the kernel coloring is already in F_ref.               */
/* ═══════════════════════════════════════════════════════════════════════════ */

#undef COC_NPARAMS
#define COC_NPARAMS  5   /* A, Sxx, Syy, Sxy, N0 */

struct coc_fit_data {
    size_t n;
    const double *K_X;
    const double *K_Y;
    const double *F_coc;     /* |F(CoC)|(k) — magnitude spectrum */
    const double *F_ref;     /* √(|F_AA1|·|F_BB1|·|F_AA2|·|F_BB2|)(k) */
    const double *weights;
};

static int coc_residual_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct coc_fit_data *d = (struct coc_fit_data *)data;
    double A   = gsl_vector_get(x, 0);
    double Sxx = fmax(gsl_vector_get(x, 1), 0.0);
    double Syy = fmax(gsl_vector_get(x, 2), 0.0);
    double Sxy = gsl_vector_get(x, 3);
    double N0  = fmax(gsl_vector_get(x, 4), 0.0);
    double two_pi_sq = 2.0 * M_PI * M_PI;

    for (size_t i = 0; i < d->n; i++) {
        double kx = d->K_X[i], ky = d->K_Y[i];
        double quad = Sxx * kx * kx + 2.0 * Sxy * kx * ky + Syy * ky * ky;
        double model = d->F_ref[i] * A * exp(-two_pi_sq * quad) + N0;
        gsl_vector_set(f, i, d->weights[i] * (d->F_coc[i] - model));
    }
    return GSL_SUCCESS;
}

static int coc_residual_df(const gsl_vector *x, void *data, gsl_matrix *J) {
    struct coc_fit_data *d = (struct coc_fit_data *)data;
    double A   = gsl_vector_get(x, 0);
    double Sxx = fmax(gsl_vector_get(x, 1), 0.0);
    double Syy = fmax(gsl_vector_get(x, 2), 0.0);
    double Sxy = gsl_vector_get(x, 3);
    double two_pi_sq = 2.0 * M_PI * M_PI;

    int Sxx_active = (gsl_vector_get(x, 1) >= 0.0);
    int Syy_active = (gsl_vector_get(x, 2) >= 0.0);
    int N0_active  = (gsl_vector_get(x, 4) >= 0.0);

    gsl_matrix_set_zero(J);

    for (size_t i = 0; i < d->n; i++) {
        double kx = d->K_X[i], ky = d->K_Y[i];
        double quad = Sxx * kx * kx + 2.0 * Sxy * kx * ky + Syy * ky * ky;
        double gauss = exp(-two_pi_sq * quad);
        double w = d->weights[i];
        double fr = d->F_ref[i];

        /* dmodel/dA = F_ref × gauss */
        gsl_matrix_set(J, i, 0, -w * fr * gauss);

        /* dmodel/dSxx = F_ref × A × (-2π²kx²) × gauss */
        if (Sxx_active)
            gsl_matrix_set(J, i, 1, w * fr * A * two_pi_sq * kx * kx * gauss);

        /* dmodel/dSyy = F_ref × A × (-2π²ky²) × gauss */
        if (Syy_active)
            gsl_matrix_set(J, i, 2, w * fr * A * two_pi_sq * ky * ky * gauss);

        /* dmodel/dSxy = F_ref × A × (-2π²·2kxky) × gauss */
        gsl_matrix_set(J, i, 3, w * fr * A * two_pi_sq * 2.0 * kx * ky * gauss);

        /* dmodel/dN0 = 1 */
        if (N0_active)
            gsl_matrix_set(J, i, 4, -w);
    }
    return GSL_SUCCESS;
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Helper: profile-based k_max estimation                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */

static double compute_kmax_profile(
    const double *k_vals,
    const double *profile,
    size_t n,
    double dc_val,
    double threshold,
    double fallback,
    double cap)
{
    double thresh = fmax(dc_val * threshold, 1e-12);
    double kmax = fallback;
    size_t center = n / 2;
    for (size_t i = center + 1; i < n; i++) {
        if (profile[i] < thresh) {
            kmax = fabs(k_vals[i]);
            break;
        }
    }
    return (kmax < cap) ? kmax : cap;
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Helper: 3-point Gaussian sub-pixel peak in spatial domain                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void subpixel_peak(
    const float *plane, size_t h, size_t w,
    double *out_x, double *out_y)
{
    size_t peak_idx = 0;
    float peak_val = plane[0];
    for (size_t i = 1; i < h * w; i++) {
        if (plane[i] > peak_val) {
            peak_val = plane[i];
            peak_idx = i;
        }
    }
    size_t py = peak_idx / w;
    size_t px = peak_idx % w;
    size_t cx = w / 2, cy = h / 2;

    double sub_x = 0.0;
    if (px > 0 && px < w - 1) {
        float l = plane[py * w + px - 1];
        float c = plane[py * w + px];
        float r = plane[py * w + px + 1];
        if (l > 0 && c > 0 && r > 0) {
            double lnl = log((double)l), lnc = log((double)c), lnr = log((double)r);
            double denom = 2.0 * (lnl - 2.0 * lnc + lnr);
            if (fabs(denom) > 1e-12) sub_x = (lnl - lnr) / denom;
        }
    }
    double sub_y = 0.0;
    if (py > 0 && py < h - 1) {
        float t = plane[(py - 1) * w + px];
        float c = plane[py * w + px];
        float b = plane[(py + 1) * w + px];
        if (t > 0 && c > 0 && b > 0) {
            double lnt = log((double)t), lnc = log((double)c), lnb = log((double)b);
            double denom = 2.0 * (lnt - 2.0 * lnc + lnb);
            if (fabs(denom) > 1e-12) sub_y = (lnt - lnb) / denom;
        }
    }

    *out_x = (double)px + sub_x - (double)cx;
    *out_y = (double)py + sub_y - (double)cy;
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Helper: compute r2c FFT magnitude spectrum in fftshift order              */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void compute_fft_magnitude(
    const float *plane,
    float *fft_buf,
    fftwf_complex *F_out,
    fftwf_plan plan,
    double *F_mag,
    size_t h, size_t w)
{
    size_t n_pixels = h * w;

    /* ifftshift: center → corner */
    for (size_t r = 0; r < h; r++) {
        for (size_t c = 0; c < w; c++) {
            size_t rs = (r + h / 2) % h;
            size_t cs = (c + w / 2) % w;
            fft_buf[rs * w + cs] = plane[r * w + c];
        }
    }

    fftwf_execute_dft_r2c(plan, fft_buf, F_out);

    /* Reconstruct full fftshift-ordered magnitude from r2c half-plane */
    for (size_t r = 0; r < h; r++) {
        int k_r = (int)r - (int)(h / 2);
        for (size_t c = 0; c < w; c++) {
            int k_c = (int)c - (int)(w / 2);

            size_t r_nat = (size_t)(((k_r % (int)h) + (int)h) % (int)h);
            size_t c_nat;

            if (k_c >= 0) {
                c_nat = (size_t)k_c;
            } else {
                c_nat = (size_t)(-k_c);
                r_nat = (r_nat == 0) ? 0 : h - r_nat;
            }

            size_t idx = r_nat * (w / 2 + 1) + c_nat;
            double re = (double)F_out[idx][0];
            double im = (double)F_out[idx][1];
            F_mag[r * w + c] = sqrt(re * re + im * im);
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Main exported function                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

PIV_EXPORT int fit_kspace_coc_batch(
    size_t num_windows,
    size_t coc_h,
    size_t coc_w,
    const float  *R_CoC,
    const float  *R_AA1,
    const float  *R_BB1,
    const float  *R_AA2,
    const float  *R_BB2,
    const int    *mask,
    int    use_soft_weighting,
    double k_max_cap,
    /* Diagnostic output (nullable — pass NULL to skip) */
    double *diag_F_coc,          /* per-window |F(CoC)| magnitude (num_windows * coc_h * coc_w) */
    double *diag_F_ref,          /* per-window F_ref (num_windows * coc_h * coc_w) */
    /* Output arrays */
    double *out_spread_xx,
    double *out_spread_yy,
    double *out_spread_xy,
    double *out_center_x,
    double *out_center_y,
    int    *out_status)
{
    if (k_max_cap <= 0.0) k_max_cap = 0.35;

    size_t n_pixels = coc_h * coc_w;
    size_t n_fft = coc_h * (coc_w / 2 + 1);
    size_t center_y = coc_h / 2;
    size_t center_x = coc_w / 2;

    int n_success = 0;

    /* GSL solver configuration */
    const gsl_multifit_nlinear_type *T_gsl = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters fdf_params =
        gsl_multifit_nlinear_default_parameters();
    fdf_params.solver = gsl_multifit_nlinear_solver_cholesky;
    fdf_params.scale  = gsl_multifit_nlinear_scale_more;

    /* Pre-compute normalised k-frequency grids */
    double *k_x = (double *)malloc(coc_w * sizeof(double));
    double *k_y = (double *)malloc(coc_h * sizeof(double));
    double *K_X = (double *)malloc(n_pixels * sizeof(double));
    double *K_Y = (double *)malloc(n_pixels * sizeof(double));

    if (!k_x || !k_y || !K_X || !K_Y) {
        free(k_x); free(k_y); free(K_X); free(K_Y);
        return 0;
    }

    for (size_t c = 0; c < coc_w; c++) {
        int ic = (int)c - (int)(coc_w / 2);
        k_x[c] = (double)ic / (double)coc_w;
    }
    for (size_t r = 0; r < coc_h; r++) {
        int ir = (int)r - (int)(coc_h / 2);
        k_y[r] = (double)ir / (double)coc_h;
    }
    for (size_t r = 0; r < coc_h; r++) {
        for (size_t c = 0; c < coc_w; c++) {
            K_X[r * coc_w + c] = k_x[c];
            K_Y[r * coc_w + c] = k_y[r];
        }
    }

    /* Initialise masked outputs */
    for (size_t i = 0; i < num_windows; i++) {
        out_spread_xx[i] = 0.0;
        out_spread_yy[i] = 0.0;
        out_spread_xy[i] = 0.0;
        out_center_x[i] = 0.0;
        out_center_y[i] = 0.0;
        out_status[i] = STATUS_MASKED;
    }

    /* FFTW thread init */
    #ifdef _OPENMP
    fftwf_init_threads();
    #endif

    #pragma omp parallel reduction(+:n_success)
    {
        /* Thread-local buffers */
        float *fft_buf = (float *)fftwf_malloc(n_pixels * sizeof(float));
        fftwf_complex *F_out = (fftwf_complex *)fftwf_alloc_complex(n_fft);
        double *F_coc_mag = (double *)malloc(n_pixels * sizeof(double));
        double *F_ref     = (double *)malloc(n_pixels * sizeof(double));
        double *F_auto    = (double *)malloc(n_pixels * sizeof(double));
        double *fit_kx  = (double *)malloc(n_pixels * sizeof(double));
        double *fit_ky  = (double *)malloc(n_pixels * sizeof(double));
        double *fit_fcoc = (double *)malloc(n_pixels * sizeof(double));
        double *fit_fref = (double *)malloc(n_pixels * sizeof(double));
        double *fit_wts = (double *)malloc(n_pixels * sizeof(double));
        double *prof_x  = (double *)malloc(coc_w * sizeof(double));
        double *prof_y  = (double *)malloc(coc_h * sizeof(double));

        fftwf_plan plan_fwd = NULL;
        #pragma omp critical
        plan_fwd = fftwf_plan_dft_r2c_2d(
            (int)coc_h, (int)coc_w, fft_buf, F_out,
            FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        gsl_multifit_nlinear_workspace *work = NULL;

        if (!fft_buf || !F_out || !F_coc_mag || !F_ref || !F_auto ||
            !fit_kx || !fit_ky || !fit_fcoc || !fit_fref || !fit_wts ||
            !prof_x || !prof_y || !plan_fwd) {
            goto coc_fit_cleanup;
        }

        #pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < num_windows; i++) {
            if (mask[i] == 1) continue;

            const float *coc_plane = &R_CoC[i * n_pixels];

            /* ── 1. Sub-pixel peak displacement (diagnostic) ─────────── */
            subpixel_peak(coc_plane, coc_h, coc_w,
                          &out_center_x[i], &out_center_y[i]);

            /* ── 2. FFT the CoC plane → |F_CoC|(k) ──────────────────── */
            compute_fft_magnitude(coc_plane, fft_buf, F_out, plan_fwd,
                                  F_coc_mag, coc_h, coc_w);

            /* ── 3. Compute F_ref = √(|AA1|·|BB1|·|AA2|·|BB2|) ──────── */
            /* Use log-space accumulation to avoid float64 overflow:
             * F_ref = exp(0.5 * (log|AA1| + log|BB1| + log|AA2| + log|BB2|)) */
            for (size_t p = 0; p < n_pixels; p++) F_ref[p] = 0.0;

            const float *auto_planes[4] = {
                &R_AA1[i * n_pixels], &R_BB1[i * n_pixels],
                &R_AA2[i * n_pixels], &R_BB2[i * n_pixels]
            };

            for (int a = 0; a < 4; a++) {
                compute_fft_magnitude(auto_planes[a], fft_buf, F_out,
                                      plan_fwd, F_auto, coc_h, coc_w);
                for (size_t p = 0; p < n_pixels; p++)
                    F_ref[p] += log(fmax(F_auto[p], 1e-30));
            }

            for (size_t p = 0; p < n_pixels; p++)
                F_ref[p] = exp(0.5 * F_ref[p]);

            /* ── 3b. Copy diagnostics if requested ───────────────────── */
            if (diag_F_coc)
                memcpy(&diag_F_coc[i * n_pixels], F_coc_mag, n_pixels * sizeof(double));
            if (diag_F_ref)
                memcpy(&diag_F_ref[i * n_pixels], F_ref, n_pixels * sizeof(double));

            /* ── 4. Profile-based k_max (from F_ref envelope) ────────── */
            for (size_t c = 0; c < coc_w; c++)
                prof_x[c] = F_ref[center_y * coc_w + c];
            for (size_t r = 0; r < coc_h; r++)
                prof_y[r] = F_ref[r * coc_w + center_x];

            double F_ref_dc = F_ref[center_y * coc_w + center_x];
            if (F_ref_dc < 1e-12) {
                for (size_t p = 0; p < n_pixels; p++)
                    if (F_ref[p] > F_ref_dc) F_ref_dc = F_ref[p];
            }
            if (F_ref_dc < 1e-12) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            double k_max_x = compute_kmax_profile(k_x, prof_x, coc_w,
                                                    F_ref_dc, 0.01, 0.05, k_max_cap);
            double k_max_y = compute_kmax_profile(k_y, prof_y, coc_h,
                                                    F_ref_dc, 0.01, 0.05, k_max_cap);

            /* ── 5. Initial guess via 1D regression on log(F_coc/F_ref) ── */
            double k_min_cut = 1.5 / fmax((double)coc_h, (double)coc_w);
            double sum_k4_x = 0, sum_k2_lnR_x = 0;
            double sum_k4_y = 0, sum_k2_lnR_y = 0;
            int cnt_x = 0, cnt_y = 0;

            for (size_t c = 0; c < coc_w; c++) {
                double k = fabs(k_x[c]);
                if (k < k_min_cut || k > k_max_x) continue;
                size_t p = center_y * coc_w + c;
                if (F_ref[p] < 1e-12) continue;
                double ratio = F_coc_mag[p] / F_ref[p];
                if (ratio < 1e-12) continue;
                double ln_r = log(ratio);
                double k2 = k_x[c] * k_x[c];
                sum_k4_x += k2 * k2;
                sum_k2_lnR_x += k2 * ln_r;
                cnt_x++;
            }

            for (size_t r = 0; r < coc_h; r++) {
                double k = fabs(k_y[r]);
                if (k < k_min_cut || k > k_max_y) continue;
                size_t p = r * coc_w + center_x;
                if (F_ref[p] < 1e-12) continue;
                double ratio = F_coc_mag[p] / F_ref[p];
                if (ratio < 1e-12) continue;
                double ln_r = log(ratio);
                double k2 = k_y[r] * k_y[r];
                sum_k4_y += k2 * k2;
                sum_k2_lnR_y += k2 * ln_r;
                cnt_y++;
            }

            double two_pi_sq = 2.0 * M_PI * M_PI;
            double Sxx_init = (cnt_x >= 3 && sum_k4_x > 1e-20)
                ? fmax(-sum_k2_lnR_x / (two_pi_sq * sum_k4_x), 0.01) : 1.0;
            double Syy_init = (cnt_y >= 3 && sum_k4_y > 1e-20)
                ? fmax(-sum_k2_lnR_y / (two_pi_sq * sum_k4_y), 0.01) : 1.0;
            if (Sxx_init < 0.001) Sxx_init = 1.0;
            if (Syy_init < 0.001) Syy_init = 1.0;

            /* Initial A: ratio at lowest valid k */
            double A_init = 1.0;
            {
                double best_k2 = 1e30;
                for (size_t p = 0; p < n_pixels; p++) {
                    double kx = K_X[p], ky = K_Y[p];
                    double k2 = kx * kx + ky * ky;
                    if (k2 < k_min_cut * k_min_cut) continue;
                    if (F_ref[p] < 1e-6) continue;
                    if (k2 < best_k2) {
                        best_k2 = k2;
                        A_init = F_coc_mag[p] / F_ref[p];
                    }
                }
            }

            /* ── 6. Collect valid k-points ───────────────────────────── */
            double k_max_x_sq = k_max_x * k_max_x;
            double k_max_y_sq = k_max_y * k_max_y;
            double k_min_sq = k_min_cut * k_min_cut;
            size_t n_valid = 0;

            for (size_t p = 0; p < n_pixels; p++) {
                double kx = K_X[p], ky = K_Y[p];
                if (kx * kx + ky * ky < k_min_sq)
                    continue;
                if (kx * kx / k_max_x_sq + ky * ky / k_max_y_sq > 1.0)
                    continue;

                fit_kx[n_valid]   = kx;
                fit_ky[n_valid]   = ky;
                fit_fcoc[n_valid] = F_coc_mag[p];
                fit_fref[n_valid] = F_ref[p];
                n_valid++;
            }

            if (n_valid < 10) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            /* ── 7. Compute weights ──────────────────────────────────── */
            if (use_soft_weighting) {
                double k0_x_sq = 1.0 / (two_pi_sq * fmax(Sxx_init, 0.01) + 1e-12);
                double k0_y_sq = 1.0 / (two_pi_sq * fmax(Syy_init, 0.01) + 1e-12);
                for (size_t j = 0; j < n_valid; j++) {
                    double kx = fit_kx[j], ky = fit_ky[j];
                    fit_wts[j] = exp(-kx * kx / k0_x_sq - ky * ky / k0_y_sq);
                }
            } else {
                for (size_t j = 0; j < n_valid; j++)
                    fit_wts[j] = 1.0;
            }

            /* ── 8. GSL implicit model + N0 fit ──────────────────────── */
            struct coc_fit_data cdata;
            cdata.n = n_valid;
            cdata.K_X = fit_kx;
            cdata.K_Y = fit_ky;
            cdata.F_coc = fit_fcoc;
            cdata.F_ref = fit_fref;
            cdata.weights = fit_wts;

            gsl_multifit_nlinear_fdf fdf;
            fdf.f      = coc_residual_f;
            fdf.df     = coc_residual_df;
            fdf.fvv    = NULL;
            fdf.n      = n_valid;
            fdf.p      = COC_NPARAMS;
            fdf.params = &cdata;

            if (work) {
                gsl_multifit_nlinear_free(work);
                work = NULL;
            }

            work = gsl_multifit_nlinear_alloc(T_gsl, &fdf_params,
                                               n_valid, COC_NPARAMS);
            if (!work) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            /* Initial N0: median of high-k |F(CoC)| values (noise floor) */
            double N0_init = 0.0;
            {
                /* Use the collected fit points — the highest-k ones are noise */
                size_t n_top = n_valid / 4;  /* top 25% of k values */
                if (n_top > 5) {
                    double sum = 0.0;
                    for (size_t j = n_valid - n_top; j < n_valid; j++)
                        sum += fit_fcoc[j];
                    N0_init = sum / (double)n_top;
                }
            }

            double p0[COC_NPARAMS] = { A_init, Sxx_init, Syy_init, 0.0, N0_init };
            gsl_vector_view xv = gsl_vector_view_array(p0, COC_NPARAMS);

            int init_status = gsl_multifit_nlinear_init(&xv.vector, &fdf, work);
            if (init_status != GSL_SUCCESS) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            int info;
            int fit_status = gsl_multifit_nlinear_driver(
                COC_MAX_ITER, XTOL, GTOL, FTOL,
                NULL, NULL, &info, work);

            if (fit_status != GSL_SUCCESS && fit_status != GSL_EMAXITER) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            /* Extract fitted parameters */
            gsl_vector *x_result = gsl_multifit_nlinear_position(work);
            double Sxx = fmax(gsl_vector_get(x_result, 1), 0.0);
            double Syy = fmax(gsl_vector_get(x_result, 2), 0.0);
            double Sxy = gsl_vector_get(x_result, 3);

            out_spread_xx[i] = Sxx;
            out_spread_yy[i] = Syy;
            out_spread_xy[i] = Sxy;
            out_status[i] = STATUS_SUCCESS;
            n_success++;

        } /* end window loop */

    coc_fit_cleanup:
        if (fft_buf)   fftwf_free(fft_buf);
        if (F_out)     fftwf_free(F_out);
        if (F_coc_mag) free(F_coc_mag);
        if (F_ref)     free(F_ref);
        if (F_auto)    free(F_auto);
        if (fit_kx)    free(fit_kx);
        if (fit_ky)    free(fit_ky);
        if (fit_fcoc)  free(fit_fcoc);
        if (fit_fref)  free(fit_fref);
        if (fit_wts)   free(fit_wts);
        if (prof_x)    free(prof_x);
        if (prof_y)    free(prof_y);
        if (work)      gsl_multifit_nlinear_free(work);
        if (plan_fwd) {
            #pragma omp critical
            fftwf_destroy_plan(plan_fwd);
        }

    } /* end omp parallel */

    free(k_x); free(k_y); free(K_X); free(K_Y);
    return n_success;
}
