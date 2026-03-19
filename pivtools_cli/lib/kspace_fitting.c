// kspace_fitting.c
// K-Space Transfer Function Fitting for Ensemble PIV
//
// Port of kspace_fitting.py to C with FFTW (float32) + GSL (double) + OpenMP.
// Follows the pattern of marquadt_gaussian.c.

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

#define KSPACE_PARAMS 16
#define JOINT_NPARAMS 5
#define MAIN_NPARAMS 5
#define JOINT_MAX_ITER 200
#define MAIN_MAX_ITER 250
#define XTOL 1e-8
#define GTOL 1e-8
#define FTOL 1e-8

// Status codes
#define STATUS_MASKED  -1
#define STATUS_SUCCESS  0
#define STATUS_NO_CONVERGE 1
#define STATUS_LOW_SNR  2
#define STATUS_BIG_DISP 3
#define STATUS_NEG_VAR  5

// ============================================================================
// Interpolation kernel noise PSD (ported from interpolation_noise_psd.py)
// ============================================================================

static double frac_distance(double x) {
    double rem = fabs(fmod(x, 1.0));
    return (rem <= 0.5) ? rem : 1.0 - rem;
}

// Bicubic kernel (Keys a=-0.75, 4-tap)
static void bicubic_weights(double f, double w[4]) {
    double a = -0.75;
    double f2 = f * f;
    double f3 = f2 * f;
    w[0] = a * f3 - 2.0 * a * f2 + a * f;           // w[-1]
    w[1] = (a + 2.0) * f3 - (a + 3.0) * f2 + 1.0;   // w[0]
    w[2] = -(a + 2.0) * f3 + (2.0 * a + 3.0) * f2 - a * f; // w[1]
    w[3] = -a * f3 + a * f2;                          // w[2]
}

// |H(k, f)|^2 for bicubic kernel
static double bicubic_mag_sq(double k, double f) {
    double w[4];
    bicubic_weights(f, w);
    double twopik = 2.0 * M_PI * k;
    // H = w[0]*e^{+i*twopik} + w[1] + w[2]*e^{-i*twopik} + w[3]*e^{-2i*twopik}
    // Offsets: -1, 0, +1, +2
    double Hr = w[0] * cos(twopik) + w[1] + w[2] * cos(twopik) + w[3] * cos(2.0 * twopik);
    double Hi = w[0] * sin(twopik) - w[2] * sin(twopik) - w[3] * sin(2.0 * twopik);
    return Hr * Hr + Hi * Hi;
}

// Lanczos-3 single weight
static double lanczos3_single_weight(double t) {
    if (fabs(t) < 1e-12) return 1.0;
    if (fabs(t) >= 3.0) return 0.0;
    double pit = M_PI * t;
    return (sin(pit) / pit) * (sin(pit / 3.0) / (pit / 3.0));
}

// |H(k, f)|^2 for Lanczos-3 kernel
static double lanczos3_mag_sq(double k, double f) {
    // 6 taps at offsets [-2, -1, 0, +1, +2, +3]
    int offsets[6] = {-2, -1, 0, 1, 2, 3};
    double raw[6], w[6];
    double sum = 0.0;
    for (int i = 0; i < 6; i++) {
        raw[i] = lanczos3_single_weight(f - offsets[i]);
        sum += raw[i];
    }
    if (sum < 1e-12) {
        // Degenerate: all weight on center
        return 1.0;
    }
    for (int i = 0; i < 6; i++) w[i] = raw[i] / sum;

    double twopik = 2.0 * M_PI * k;
    double Hr = 0.0, Hi = 0.0;
    for (int i = 0; i < 6; i++) {
        double angle = -offsets[i] * twopik;
        Hr += w[i] * cos(angle);
        Hi += w[i] * sin(angle);
    }
    return Hr * Hr + Hi * Hi;
}

// ============================================================================
// FFT helpers: ifftshift / fftshift for 2D
// ============================================================================

// ifftshift: swap quadrants before FFT (move center to corner)
static void ifftshift_2d_float(const float *in, float *out, size_t h, size_t w) {
    size_t hh = h / 2, hw = w / 2;
    for (size_t r = 0; r < h; r++) {
        for (size_t c = 0; c < w; c++) {
            size_t sr = (r + hh) % h;
            size_t sc = (c + hw) % w;
            out[r * w + c] = in[sr * w + sc];
        }
    }
}

// fftshift on complex output: swap quadrants after FFT
static void fftshift_2d_complex(fftwf_complex *data, size_t h, size_t w, fftwf_complex *tmp) {
    size_t hh = h / 2, hw = w / 2;
    memcpy(tmp, data, h * w * sizeof(fftwf_complex));
    for (size_t r = 0; r < h; r++) {
        for (size_t c = 0; c < w; c++) {
            size_t sr = (r + hh) % h;
            size_t sc = (c + hw) % w;
            data[sr * w + sc][0] = tmp[r * w + c][0];
            data[sr * w + sc][1] = tmp[r * w + c][1];
        }
    }
}

// ============================================================================
// Joint noise fit data structure
// ============================================================================

struct joint_fit_data {
    size_t n;
    const double *K_X;
    const double *K_Y;
    const double *F_ref_norm;
    const double *P_noise;
    const double *weights;
};

static int joint_residual_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct joint_fit_data *d = (struct joint_fit_data *)data;
    double A    = fmin(fmax(gsl_vector_get(x, 0), 0.01), 10.0);
    double sx   = fmin(fmax(gsl_vector_get(x, 1), 0.1),  20.0);
    double sy   = fmin(fmax(gsl_vector_get(x, 2), 0.1),  20.0);
    double N0   = fmin(fmax(gsl_vector_get(x, 3), 0.0),   1.0);
    double beta = fmin(fmax(gsl_vector_get(x, 4), -1.0), 10.0);
    double two_pi_sq = 2.0 * M_PI * M_PI;

    for (size_t i = 0; i < d->n; i++) {
        double kx = d->K_X[i], ky = d->K_Y[i];
        double q = two_pi_sq * (kx * kx * sx * sx + ky * ky * sy * sy);
        double signal = A * exp(-q) * (1.0 + beta * q * q);
        double model = (signal + N0) * d->P_noise[i];
        gsl_vector_set(f, i, d->weights[i] * (d->F_ref_norm[i] - model));
    }
    return GSL_SUCCESS;
}

static int joint_residual_df(const gsl_vector *x, void *data, gsl_matrix *J) {
    struct joint_fit_data *d = (struct joint_fit_data *)data;
    double A    = fmin(fmax(gsl_vector_get(x, 0), 0.01), 10.0);
    double sx   = fmin(fmax(gsl_vector_get(x, 1), 0.1),  20.0);
    double sy   = fmin(fmax(gsl_vector_get(x, 2), 0.1),  20.0);
    double beta = fmin(fmax(gsl_vector_get(x, 4), -1.0), 10.0);
    double two_pi_sq = 2.0 * M_PI * M_PI;

    // Constraint activity flags: zero Jacobian when at lower OR upper bound
    double raw_A    = gsl_vector_get(x, 0);
    double raw_sx   = gsl_vector_get(x, 1);
    double raw_sy   = gsl_vector_get(x, 2);
    double raw_N0   = gsl_vector_get(x, 3);
    double raw_beta = gsl_vector_get(x, 4);
    int A_active    = (raw_A    >= 0.01 && raw_A    <= 10.0);
    int sx_active   = (raw_sx   >= 0.1  && raw_sx   <= 20.0);
    int sy_active   = (raw_sy   >= 0.1  && raw_sy   <= 20.0);
    int N0_active   = (raw_N0   >= 0.0  && raw_N0   <= 1.0);
    int beta_active = (raw_beta >= -1.0 && raw_beta <= 10.0);

    for (size_t i = 0; i < d->n; i++) {
        double kx = d->K_X[i], ky = d->K_Y[i];
        double q = two_pi_sq * (kx * kx * sx * sx + ky * ky * sy * sy);
        double exp_neg_q = exp(-q);
        double kurt_factor = 1.0 + beta * q * q;
        double w = d->weights[i];
        double P = d->P_noise[i];

        // dsignal/dq = A*exp(-q)*[-(1+beta*q^2) + 2*beta*q]
        //            = A*exp(-q)*[-1 - beta*q^2 + 2*beta*q]
        double dsignal_dq = A * exp_neg_q * (-1.0 - beta * q * q + 2.0 * beta * q);

        // d/dA: signal/A * P = exp(-q)*(1+beta*q^2)*P
        gsl_matrix_set(J, i, 0, A_active ? -w * exp_neg_q * kurt_factor * P : 0.0);
        // d/dsx: dsignal/dq * dq/dsx * P, where dq/dsx = 2*two_pi_sq*kx^2*sx
        double dq_dsx = 2.0 * two_pi_sq * kx * kx * sx;
        gsl_matrix_set(J, i, 1, sx_active ? -w * dsignal_dq * dq_dsx * P : 0.0);
        // d/dsy: dsignal/dq * dq/dsy * P, where dq/dsy = 2*two_pi_sq*ky^2*sy
        double dq_dsy = 2.0 * two_pi_sq * ky * ky * sy;
        gsl_matrix_set(J, i, 2, sy_active ? -w * dsignal_dq * dq_dsy * P : 0.0);
        // d/dN0: unchanged
        gsl_matrix_set(J, i, 3, N0_active ? -w * P : 0.0);
        // d/dbeta: A*exp(-q)*q^2*P
        gsl_matrix_set(J, i, 4, beta_active ? -w * A * exp_neg_q * q * q * P : 0.0);
    }
    return GSL_SUCCESS;
}

// ============================================================================
// Main transfer function fit data structure
// ============================================================================

struct main_fit_data {
    size_t n;           // Number of valid k-points
    const double *K_X;
    const double *K_Y;
    const double *T_norm_real;
    const double *T_norm_imag;
    const double *weights;
};

static int main_residual_f(const gsl_vector *x, void *data, gsl_vector *f) {
    struct main_fit_data *d = (struct main_fit_data *)data;
    double mu_x    = gsl_vector_get(x, 0);
    double mu_y    = gsl_vector_get(x, 1);
    double Sxx     = fmax(gsl_vector_get(x, 2), 0.0);
    double Syy     = fmax(gsl_vector_get(x, 3), 0.0);
    double Sxy     = gsl_vector_get(x, 4);
    double two_pi  = 2.0 * M_PI;
    double two_pi_sq = two_pi * M_PI;

    for (size_t i = 0; i < d->n; i++) {
        double kx = d->K_X[i], ky = d->K_Y[i];
        double quad = Sxx * kx * kx + 2.0 * Sxy * kx * ky + Syy * ky * ky;
        double decay = exp(-two_pi_sq * quad);
        double phase = -two_pi * (kx * mu_x + ky * mu_y);
        double model_r = decay * cos(phase);
        double model_i = decay * sin(phase);
        double w = d->weights[i];
        gsl_vector_set(f, i,         w * (d->T_norm_real[i] - model_r));
        gsl_vector_set(f, i + d->n,  w * (d->T_norm_imag[i] - model_i));
    }
    return GSL_SUCCESS;
}

static int main_residual_df(const gsl_vector *x, void *data, gsl_matrix *J) {
    struct main_fit_data *d = (struct main_fit_data *)data;
    double mu_x    = gsl_vector_get(x, 0);
    double mu_y    = gsl_vector_get(x, 1);
    double Sxx     = fmax(gsl_vector_get(x, 2), 0.0);
    double Syy     = fmax(gsl_vector_get(x, 3), 0.0);
    double Sxy     = gsl_vector_get(x, 4);
    double two_pi  = 2.0 * M_PI;
    double two_pi_sq = two_pi * M_PI;

    int Sxx_active = (gsl_vector_get(x, 2) >= 0.0);
    int Syy_active = (gsl_vector_get(x, 3) >= 0.0);

    gsl_matrix_set_zero(J);

    for (size_t i = 0; i < d->n; i++) {
        double kx = d->K_X[i], ky = d->K_Y[i];
        double quad = Sxx * kx * kx + 2.0 * Sxy * kx * ky + Syy * ky * ky;
        double decay = exp(-two_pi_sq * quad);
        double phase = -two_pi * (kx * mu_x + ky * mu_y);
        double cos_p = cos(phase), sin_p = sin(phase);
        double w = d->weights[i];

        double dphase_dmux = -two_pi * kx;
        double dphase_dmuy = -two_pi * ky;
        double dTr_dmux = decay * (-sin_p * dphase_dmux);
        double dTi_dmux = decay * ( cos_p * dphase_dmux);
        double dTr_dmuy = decay * (-sin_p * dphase_dmuy);
        double dTi_dmuy = decay * ( cos_p * dphase_dmuy);

        gsl_matrix_set(J, i,         0, -w * dTr_dmux);
        gsl_matrix_set(J, i + d->n,  0, -w * dTi_dmux);
        gsl_matrix_set(J, i,         1, -w * dTr_dmuy);
        gsl_matrix_set(J, i + d->n,  1, -w * dTi_dmuy);

        double ddecay_dSxx = decay * (-two_pi_sq * kx * kx);
        double ddecay_dSyy = decay * (-two_pi_sq * ky * ky);
        double ddecay_dSxy = decay * (-two_pi_sq * 2.0 * kx * ky);

        if (Sxx_active) {
            gsl_matrix_set(J, i,         2, -w * ddecay_dSxx * cos_p);
            gsl_matrix_set(J, i + d->n,  2, -w * ddecay_dSxx * sin_p);
        }
        if (Syy_active) {
            gsl_matrix_set(J, i,         3, -w * ddecay_dSyy * cos_p);
            gsl_matrix_set(J, i + d->n,  3, -w * ddecay_dSyy * sin_p);
        }
        gsl_matrix_set(J, i,         4, -w * ddecay_dSxy * cos_p);
        gsl_matrix_set(J, i + d->n,  4, -w * ddecay_dSxy * sin_p);
    }
    return GSL_SUCCESS;
}

// ============================================================================
// Helper: 3-point Gaussian sub-pixel peak finder
// ============================================================================

static void estimate_displacement_from_peak(
    const float *R_AB, size_t h, size_t w,
    size_t center_y, size_t center_x,
    double *mu_x, double *mu_y)
{
    // Find integer peak
    size_t peak_idx = 0;
    float peak_val = R_AB[0];
    for (size_t i = 1; i < h * w; i++) {
        if (R_AB[i] > peak_val) {
            peak_val = R_AB[i];
            peak_idx = i;
        }
    }
    size_t py = peak_idx / w;
    size_t px = peak_idx % w;

    // Sub-pixel x
    double sub_x = 0.0;
    if (px > 0 && px < w - 1) {
        float left   = R_AB[py * w + px - 1];
        float center = R_AB[py * w + px];
        float right  = R_AB[py * w + px + 1];
        if (left > 0 && center > 0 && right > 0) {
            double ln_l = log((double)left);
            double ln_c = log((double)center);
            double ln_r = log((double)right);
            double denom = 2.0 * (ln_l - 2.0 * ln_c + ln_r);
            if (fabs(denom) > 1e-12) {
                sub_x = (ln_l - ln_r) / denom;
            }
        }
    }

    // Sub-pixel y
    double sub_y = 0.0;
    if (py > 0 && py < h - 1) {
        float top    = R_AB[(py - 1) * w + px];
        float center = R_AB[py * w + px];
        float bottom = R_AB[(py + 1) * w + px];
        if (top > 0 && center > 0 && bottom > 0) {
            double ln_t = log((double)top);
            double ln_c = log((double)center);
            double ln_b = log((double)bottom);
            double denom = 2.0 * (ln_t - 2.0 * ln_c + ln_b);
            if (fabs(denom) > 1e-12) {
                sub_y = (ln_t - ln_b) / denom;
            }
        }
    }

    *mu_x = ((double)px + sub_x) - (double)center_x;
    *mu_y = ((double)py + sub_y) - (double)center_y;
}

// ============================================================================
// Helper: 1D log-magnitude regression for variance estimation
// ============================================================================

static double fit_1d_variance(
    const double *F_AB_mag, const double *F_AB_phase,
    const double *F_ref_profile,
    const double *k_axis, size_t N,
    double k_max, double *out_mu)
{
    double k_min = 1.5 / (double)N;
    double Sigma = 1.0;
    *out_mu = 0.0;

    // Magnitude fit: ln|T| = ln|F_AB| - ln|F_ref| = -2*pi^2 * Sigma * k^2
    // Weighted least squares through origin: slope = sum(w*y*k^2) / sum(w*k^4)
    double sum_wyk2 = 0.0, sum_wk4 = 0.0;
    double max_fab = 0.0;
    for (size_t i = 0; i < N; i++) {
        double ak = fabs(k_axis[i]);
        if (ak > k_min && ak < k_max && F_AB_mag[i] > max_fab)
            max_fab = F_AB_mag[i];
    }
    if (max_fab < 1e-12) max_fab = 1.0;

    int count_mag = 0;
    for (size_t i = 0; i < N; i++) {
        double ak = fabs(k_axis[i]);
        if (ak <= k_min || ak >= k_max) continue;
        double fab = F_AB_mag[i];
        double fref = F_ref_profile[i];
        if (fab < 1e-12 || fref < 1e-12) continue;

        double log_T = log(fab) - log(fref);
        double k2 = k_axis[i] * k_axis[i];
        double w = fab / max_fab;
        sum_wyk2 += w * log_T * k2;
        sum_wk4  += w * k2 * k2;
        count_mag++;
    }

    if (count_mag >= 3 && fabs(sum_wk4) > 1e-20) {
        double slope = sum_wyk2 / sum_wk4;
        double two_pi_sq = 2.0 * M_PI * M_PI;
        Sigma = fmax(-slope / two_pi_sq, 0.01);
    }

    // Phase fit: phase(F_AB) = -2*pi * k * mu (through origin)
    double phase_k_max = (k_max < 0.25) ? k_max : 0.25;
    double sum_wpk = 0.0, sum_wk2p = 0.0;
    double max_fab_phase = 0.0;
    for (size_t i = 0; i < N; i++) {
        double ak = fabs(k_axis[i]);
        if (ak > k_min && ak < phase_k_max && F_AB_mag[i] > max_fab_phase)
            max_fab_phase = F_AB_mag[i];
    }
    if (max_fab_phase < 1e-12) max_fab_phase = 1.0;

    int count_phase = 0;
    for (size_t i = 0; i < N; i++) {
        double ak = fabs(k_axis[i]);
        if (ak <= k_min || ak >= phase_k_max) continue;
        double w = F_AB_mag[i] / max_fab_phase;
        double k = k_axis[i];
        sum_wpk  += w * F_AB_phase[i] * k;
        sum_wk2p += w * k * k;
        count_phase++;
    }

    if (count_phase >= 3 && fabs(sum_wk2p) > 1e-20) {
        double slope_phase = sum_wpk / sum_wk2p;
        *out_mu = -slope_phase / (2.0 * M_PI);
    }

    return Sigma;
}

// ============================================================================
// Helper: compute k_max from profile threshold
// ============================================================================

static double compute_kmax_from_profile(
    const double *k_axis, const double *F_profile,
    size_t N, double F_dc, double threshold_frac,
    double min_k, double max_k)
{
    double threshold = F_dc * threshold_frac;
    size_t center = N / 2;

    // Scan positive k (right half after fftshift)
    double k_max = max_k;
    for (size_t i = center; i < N; i++) {
        if (F_profile[i] < threshold) {
            if (i > center)
                k_max = k_axis[i - 1];
            else
                k_max = min_k;
            break;
        }
    }

    if (k_max < min_k) k_max = min_k;
    if (k_max > max_k) k_max = max_k;
    return k_max;
}

// ============================================================================
// Helper: compute k_max from variance and SNR
// ============================================================================

static double compute_kmax_variance(double sigma_sq, double snr,
                                    double min_k, double max_k) {
    if (sigma_sq <= 0.0 || snr <= 1.0) return max_k;
    double two_pi_sq = 2.0 * M_PI * M_PI;
    double k_max = sqrt(log(snr) / (two_pi_sq * sigma_sq + 1e-12));
    if (k_max < min_k) k_max = min_k;
    if (k_max > max_k) k_max = max_k;
    return k_max;
}

// ============================================================================
// Helper: build 16-element output params
// ============================================================================

static void build_default_params(double *params,
                                 double amp_A, double amp_B, double amp_AB,
                                 double center_x, double center_y) {
    memset(params, 0, KSPACE_PARAMS * sizeof(double));
    params[0] = amp_A;
    params[1] = amp_B;
    params[2] = amp_AB;
    // params[3..5] = 0
    // NaN for sig_A (k-space cancels particle shape, so sig_A is not estimated)
    params[6] = NAN;
    params[7] = NAN;
    params[8] = NAN;
    // params[9..11] = 0 (Sigma)
    params[12] = center_x;
    params[13] = center_y;
    params[14] = center_x;
    params[15] = center_y;
}

static void build_params_from_fit(double *params,
                                  double mu_x, double mu_y,
                                  double Sxx, double Syy, double Sxy,
                                  double amp_A, double amp_B, double amp_AB,
                                  double center_x, double center_y) {
    memset(params, 0, KSPACE_PARAMS * sizeof(double));
    params[0] = amp_A;
    params[1] = amp_B;
    params[2] = amp_AB;
    params[6] = NAN;
    params[7] = NAN;
    params[8] = NAN;
    params[9]  = Sxx;
    params[10] = Syy;
    params[11] = Sxy;
    params[12] = center_x;
    params[13] = center_y;
    params[14] = center_x + mu_x;
    params[15] = center_y + mu_y;
}

// ============================================================================
// Main batch function
// ============================================================================

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
    double *out_diagnostics)
{
    if (!R_AA || !R_BB || !R_AB || !mask || !out_params || !out_status || !out_initial_guess) {
        fprintf(stderr, "[kspace] NULL pointer argument\n");
        return 0;
    }

    size_t n_pixels = corr_h * corr_w;
    size_t center_idx_y = corr_h / 2;
    size_t center_idx_x = corr_w / 2;
    double center_x = (double)corr_w / 2.0 + 1.0;  // 1-based
    double center_y = (double)corr_h / 2.0 + 1.0;

    // Default k_max_cap: 0.35 avoids corner region used for noise estimation
    double k_max_limit;
    if (k_max_cap > 0.0)
        k_max_limit = k_max_cap;
    else
        k_max_limit = 0.35;

    // Build wavenumber grids (shared, read-only)
    double *k_x = (double *)malloc(corr_w * sizeof(double));
    double *k_y = (double *)malloc(corr_h * sizeof(double));
    double *K_X = (double *)malloc(n_pixels * sizeof(double));
    double *K_Y = (double *)malloc(n_pixels * sizeof(double));

    if (!k_x || !k_y || !K_X || !K_Y) {
        fprintf(stderr, "[kspace] Failed to allocate wavenumber grids\n");
        free(k_x); free(k_y); free(K_X); free(K_Y);
        return 0;
    }

    // fftfreq + fftshift for k_x
    for (size_t i = 0; i < corr_w; i++) {
        double freq;
        if (i <= corr_w / 2)
            freq = (double)i / (double)corr_w;
        else
            freq = ((double)i - (double)corr_w) / (double)corr_w;
        k_x[i] = freq;
    }
    // fftshift: rotate so that negative freqs come first
    {
        double *tmp = (double *)malloc(corr_w * sizeof(double));
        size_t half = corr_w / 2;
        for (size_t i = 0; i < corr_w; i++)
            tmp[i] = k_x[(i + half) % corr_w];
        memcpy(k_x, tmp, corr_w * sizeof(double));
        free(tmp);
    }

    // fftfreq + fftshift for k_y
    for (size_t i = 0; i < corr_h; i++) {
        double freq;
        if (i <= corr_h / 2)
            freq = (double)i / (double)corr_h;
        else
            freq = ((double)i - (double)corr_h) / (double)corr_h;
        k_y[i] = freq;
    }
    {
        double *tmp = (double *)malloc(corr_h * sizeof(double));
        size_t half = corr_h / 2;
        for (size_t i = 0; i < corr_h; i++)
            tmp[i] = k_y[(i + half) % corr_h];
        memcpy(k_y, tmp, corr_h * sizeof(double));
        free(tmp);
    }

    // Meshgrid (xy indexing: K_X varies along columns, K_Y along rows)
    for (size_t r = 0; r < corr_h; r++) {
        for (size_t c = 0; c < corr_w; c++) {
            K_X[r * corr_w + c] = k_x[c];
            K_Y[r * corr_w + c] = k_y[r];
        }
    }

    fprintf(stderr, "[kspace] %zu windows, %zux%zu, soft_weighting=%d, k_max_cap=%.2f\n",
            num_windows, corr_h, corr_w, use_soft_weighting, k_max_limit);

    int success_count = 0;

    gsl_set_error_handler_off();

    // GSL workspace params (shared, read-only)
    const gsl_multifit_nlinear_type *T_gsl = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters fdf_params_joint = gsl_multifit_nlinear_default_parameters();
    fdf_params_joint.solver = gsl_multifit_nlinear_solver_cholesky;
    fdf_params_joint.scale  = gsl_multifit_nlinear_scale_more;

    gsl_multifit_nlinear_parameters fdf_params_main = gsl_multifit_nlinear_default_parameters();
    fdf_params_main.solver = gsl_multifit_nlinear_solver_cholesky;
    fdf_params_main.scale  = gsl_multifit_nlinear_scale_more;
    fdf_params_main.trs    = gsl_multifit_nlinear_trs_lmaccel;  // Geodesic acceleration

    int diag_count = 0;  // Limit diagnostic prints (shared across threads)

    #ifdef _OPENMP
    #pragma omp parallel reduction(+:success_count)
    {
        #pragma omp single
        {
            fprintf(stderr, "[kspace] OpenMP threads: %d\n", omp_get_num_threads());
        }
    #else
        fprintf(stderr, "[kspace] OpenMP threads: 1 (no OpenMP)\n");
    #endif

        // Thread-local FFTW buffers (float32)
        float *fft_in     = (float *)fftwf_malloc(n_pixels * sizeof(float));
        fftwf_complex *fft_out_AA = (fftwf_complex *)fftwf_malloc(n_pixels * sizeof(fftwf_complex));
        fftwf_complex *fft_out_BB = (fftwf_complex *)fftwf_malloc(n_pixels * sizeof(fftwf_complex));
        fftwf_complex *fft_out_AB = (fftwf_complex *)fftwf_malloc(n_pixels * sizeof(fftwf_complex));
        fftwf_complex *fft_tmp    = (fftwf_complex *)fftwf_malloc(n_pixels * sizeof(fftwf_complex));
        fftwf_complex *fft_in_c   = (fftwf_complex *)fftwf_malloc(n_pixels * sizeof(fftwf_complex));

        // Thread-local double arrays for fitting
        double *F_ref       = (double *)malloc(n_pixels * sizeof(double));
        double *P_noise     = (double *)malloc(n_pixels * sizeof(double));
        double *F_ref_norm  = (double *)malloc(n_pixels * sizeof(double));
        double *joint_wts   = (double *)malloc(n_pixels * sizeof(double));
        // For main fit
        double *T_norm_r    = (double *)malloc(n_pixels * sizeof(double));
        double *T_norm_i    = (double *)malloc(n_pixels * sizeof(double));
        double *main_K_X    = (double *)malloc(n_pixels * sizeof(double));
        double *main_K_Y    = (double *)malloc(n_pixels * sizeof(double));
        double *main_wts    = (double *)malloc(n_pixels * sizeof(double));
        // 1D profiles
        double *prof_mag_x  = (double *)malloc(corr_w * sizeof(double));
        double *prof_phase_x= (double *)malloc(corr_w * sizeof(double));
        double *prof_ref_x  = (double *)malloc(corr_w * sizeof(double));
        double *prof_mag_y  = (double *)malloc(corr_h * sizeof(double));
        double *prof_phase_y= (double *)malloc(corr_h * sizeof(double));
        double *prof_ref_y  = (double *)malloc(corr_h * sizeof(double));
        double *F_ref_abs   = (double *)malloc(n_pixels * sizeof(double));
        // F_AB complex (double)
        double *F_AB_real   = (double *)malloc(n_pixels * sizeof(double));
        double *F_AB_imag   = (double *)malloc(n_pixels * sizeof(double));

        // FFTW plan (created in critical section)
        fftwf_plan plan = NULL;
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            // Complex-to-complex forward FFT
            plan = fftwf_plan_dft_2d((int)corr_h, (int)corr_w,
                                     fft_in_c, fft_out_AA,
                                     FFTW_FORWARD, FFTW_ESTIMATE);
        }

        // Thread-local GSL workspaces
        gsl_multifit_nlinear_workspace *joint_work = gsl_multifit_nlinear_alloc(
            T_gsl, &fdf_params_joint, n_pixels, JOINT_NPARAMS);
        gsl_multifit_nlinear_workspace *main_work = NULL;
        // main_work allocated per window because n_valid varies

        int alloc_ok = (fft_in && fft_out_AA && fft_out_BB && fft_out_AB && fft_tmp &&
                        fft_in_c && F_ref && P_noise && F_ref_norm && joint_wts &&
                        T_norm_r && T_norm_i && main_K_X && main_K_Y && main_wts &&
                        prof_mag_x && prof_phase_x && prof_ref_x &&
                        prof_mag_y && prof_phase_y && prof_ref_y &&
                        F_ref_abs && F_AB_real && F_AB_imag &&
                        plan && joint_work);

        if (!alloc_ok) {
            fprintf(stderr, "[kspace] thread allocation failed\n");
        }

        int i;  // MSVC OpenMP 2.0 compatibility
        #ifdef _OPENMP
        #pragma omp for schedule(dynamic, 16)
        #endif
        for (i = 0; i < (int)num_windows; i++) {
            // Default: masked
            out_status[i] = STATUS_MASKED;

            if (!alloc_ok) continue;

            // Skip masked windows
            if (mask[i]) {
                build_default_params(out_params + i * KSPACE_PARAMS,
                                     0.0, 0.0, 0.0, center_x, center_y);
                memcpy(out_initial_guess + i * KSPACE_PARAMS,
                       out_params + i * KSPACE_PARAMS,
                       KSPACE_PARAMS * sizeof(double));
                continue;
            }

            const float *raa = R_AA + (size_t)i * n_pixels;
            const float *rbb = R_BB + (size_t)i * n_pixels;
            const float *rab = R_AB + (size_t)i * n_pixels;

            // Amplitudes
            double amp_A  = (double)raa[center_idx_y * corr_w + center_idx_x];
            double amp_B  = (double)rbb[center_idx_y * corr_w + center_idx_x];
            double amp_AB = (double)rab[0];
            for (size_t p = 1; p < n_pixels; p++) {
                if ((double)rab[p] > amp_AB) amp_AB = (double)rab[p];
            }

            // Default params
            build_default_params(out_params + i * KSPACE_PARAMS,
                                 amp_A, amp_B, amp_AB, center_x, center_y);
            memcpy(out_initial_guess + i * KSPACE_PARAMS,
                   out_params + i * KSPACE_PARAMS,
                   KSPACE_PARAMS * sizeof(double));

            if (amp_A < 1e-12 || amp_B < 1e-12) {
                if (diag_count < 3)
                    fprintf(stderr, "[kspace] win %d: LOW_SNR(amp) amp_A=%.4e amp_B=%.4e\n",
                            i, amp_A, amp_B);
                diag_count++;
                out_status[i] = STATUS_LOW_SNR;
                continue;
            }

            // ---- Step 1: FFT all three planes ----
            // FFT R_AA
            ifftshift_2d_float(raa, fft_in, corr_h, corr_w);
            for (size_t p = 0; p < n_pixels; p++) {
                fft_in_c[p][0] = fft_in[p];
                fft_in_c[p][1] = 0.0f;
            }
            fftwf_execute_dft(plan, fft_in_c, fft_out_AA);
            fftshift_2d_complex(fft_out_AA, corr_h, corr_w, fft_tmp);

            // FFT R_BB
            ifftshift_2d_float(rbb, fft_in, corr_h, corr_w);
            for (size_t p = 0; p < n_pixels; p++) {
                fft_in_c[p][0] = fft_in[p];
                fft_in_c[p][1] = 0.0f;
            }
            fftwf_execute_dft(plan, fft_in_c, fft_out_BB);
            fftshift_2d_complex(fft_out_BB, corr_h, corr_w, fft_tmp);

            // FFT R_AB
            ifftshift_2d_float(rab, fft_in, corr_h, corr_w);
            for (size_t p = 0; p < n_pixels; p++) {
                fft_in_c[p][0] = fft_in[p];
                fft_in_c[p][1] = 0.0f;
            }
            fftwf_execute_dft(plan, fft_in_c, fft_out_AB);
            fftshift_2d_complex(fft_out_AB, corr_h, corr_w, fft_tmp);

            // ---- Step 2: Reference spectrum ----
            for (size_t p = 0; p < n_pixels; p++) {
                double mag_AA = sqrt(fft_out_AA[p][0] * (double)fft_out_AA[p][0] +
                                     fft_out_AA[p][1] * (double)fft_out_AA[p][1]);
                double mag_BB = sqrt(fft_out_BB[p][0] * (double)fft_out_BB[p][0] +
                                     fft_out_BB[p][1] * (double)fft_out_BB[p][1]);
                F_ref[p] = sqrt(mag_AA * mag_BB);
                F_AB_real[p] = (double)fft_out_AB[p][0];
                F_AB_imag[p] = (double)fft_out_AB[p][1];
            }

            // ---- Step 2b: Noise PSD ----
            double pred_dx = 0.0, pred_dy = 0.0;
            if (pred_disp) {
                pred_dy = pred_disp[(size_t)i * 2];
                pred_dx = pred_disp[(size_t)i * 2 + 1];
            }
            double f_x = frac_distance(pred_dx / 2.0);
            double f_y = frac_distance(pred_dy / 2.0);

            for (size_t p = 0; p < n_pixels; p++) {
                double kx = K_X[p], ky = K_Y[p];
                double hx_sq, hy_sq;
                if (interp_kernel == 1) {
                    hx_sq = lanczos3_mag_sq(kx, f_x);
                    hy_sq = lanczos3_mag_sq(ky, f_y);
                } else {
                    hx_sq = bicubic_mag_sq(kx, f_x);
                    hy_sq = bicubic_mag_sq(ky, f_y);
                }
                P_noise[p] = hx_sq * hy_sq;
            }

            // ---- Step 3: Joint noise fit ----
            double F_dc = F_ref[center_idx_y * corr_w + center_idx_x];
            if (F_dc < 1e-10) {
                if (diag_count < 3)
                    fprintf(stderr, "[kspace] win %d: LOW_SNR(F_dc) F_dc=%.4e\n", i, F_dc);
                diag_count++;
                out_status[i] = STATUS_LOW_SNR;
                continue;
            }

            // Normalise
            for (size_t p = 0; p < n_pixels; p++)
                F_ref_norm[p] = F_ref[p] / F_dc;

            // Flat (uniform) weights — joint model (A*Gauss + N0)*P_noise
            // explicitly separates signal from noise at all wavenumbers
            for (size_t p = 0; p < n_pixels; p++) {
                joint_wts[p] = 1.0;
            }

            // Set up joint fit
            struct joint_fit_data jdata;
            jdata.n = n_pixels;
            jdata.K_X = K_X;
            jdata.K_Y = K_Y;
            jdata.F_ref_norm = F_ref_norm;
            jdata.P_noise = P_noise;
            jdata.weights = joint_wts;

            gsl_multifit_nlinear_fdf joint_fdf;
            joint_fdf.f      = joint_residual_f;
            joint_fdf.df     = joint_residual_df;
            joint_fdf.fvv    = NULL;
            joint_fdf.n      = n_pixels;
            joint_fdf.p      = JOINT_NPARAMS;
            joint_fdf.params = &jdata;

            double joint_p0[5] = {1.0, 2.0, 2.0, 0.01, 0.0};
            gsl_vector_view joint_xv = gsl_vector_view_array(joint_p0, JOINT_NPARAMS);
            int joint_init_status = gsl_multifit_nlinear_init(&joint_xv.vector, &joint_fdf, joint_work);

            int joint_ok = 0;
            double N0_abs = 0.0;

            if (joint_init_status == GSL_SUCCESS) {
                int info;
                int joint_status = gsl_multifit_nlinear_driver(
                    JOINT_MAX_ITER, XTOL, GTOL, FTOL, NULL, NULL, &info, joint_work);

                if (joint_status == GSL_SUCCESS || joint_status == GSL_EMAXITER) {
                    gsl_vector *x_result = gsl_multifit_nlinear_position(joint_work);
                    double N0_norm = fmax(gsl_vector_get(x_result, 3), 0.0);
                    N0_abs = N0_norm * F_dc;

                    // Subtract colored noise floor
                    double epsilon = F_dc * 1e-8;
                    for (size_t p = 0; p < n_pixels; p++) {
                        double noise_floor = N0_abs * P_noise[p];
                        F_ref[p] = fmax(F_ref[p] - noise_floor, epsilon);
                    }
                    joint_ok = 1;
                }
            }

            if (!joint_ok) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            // ---- Step 4: Compute SNR for diagnostics (no gate) ----
            double F_dc_clean = F_ref[center_idx_y * corr_w + center_idx_x];
            double dc_power = F_dc_clean * F_dc_clean;
            double noise_power = N0_abs * N0_abs + 1e-12;
            double snr = dc_power / noise_power;

            // Store diagnostics (SNR, N0) — SNR is informational only, not used for gating
            if (out_diagnostics) {
                out_diagnostics[i * 4 + 0] = snr;
                out_diagnostics[i * 4 + 1] = N0_abs;
                out_diagnostics[i * 4 + 2] = 0.0;  // k_max_x (filled later)
                out_diagnostics[i * 4 + 3] = 0.0;  // k_max_y
            }

            // ---- Step 5: k_max from profile ----
            for (size_t p = 0; p < n_pixels; p++)
                F_ref_abs[p] = fabs(F_ref[p]);

            // Profile along x (at k_y=0)
            for (size_t c = 0; c < corr_w; c++)
                prof_ref_x[c] = F_ref_abs[center_idx_y * corr_w + c];

            double k_max_x_prof = compute_kmax_from_profile(k_x, prof_ref_x, corr_w,
                                                             F_dc_clean, 0.01, 0.05, k_max_limit);
            // Profile along y (at k_x=0)
            for (size_t r = 0; r < corr_h; r++)
                prof_ref_y[r] = F_ref_abs[r * corr_w + center_idx_x];

            double k_max_y_prof = compute_kmax_from_profile(k_y, prof_ref_y, corr_h,
                                                             F_dc_clean, 0.01, 0.05, k_max_limit);

            double k_max_init_x = (k_max_x_prof < k_max_limit) ? k_max_x_prof : k_max_limit;
            double k_max_init_y = (k_max_y_prof < k_max_limit) ? k_max_y_prof : k_max_limit;

            // ---- Step 6: Initial guesses ----
            // Displacement from sub-pixel peak
            double mu_x_init, mu_y_init;
            estimate_displacement_from_peak(rab, corr_h, corr_w,
                                            center_idx_y, center_idx_x,
                                            &mu_x_init, &mu_y_init);

            // Variance from 1D regression
            // Build 1D profiles for F_AB along x (k_y=0)
            for (size_t c = 0; c < corr_w; c++) {
                size_t idx = center_idx_y * corr_w + c;
                double re = F_AB_real[idx], im = F_AB_imag[idx];
                prof_mag_x[c] = sqrt(re * re + im * im);
                prof_phase_x[c] = atan2(im, re);
            }
            double mu_x_1d;
            double Sxx_init = fit_1d_variance(prof_mag_x, prof_phase_x, prof_ref_x,
                                               k_x, corr_w, k_max_init_x, &mu_x_1d);

            // Build 1D profiles for F_AB along y (k_x=0)
            for (size_t r = 0; r < corr_h; r++) {
                size_t idx = r * corr_w + center_idx_x;
                double re = F_AB_real[idx], im = F_AB_imag[idx];
                prof_mag_y[r] = sqrt(re * re + im * im);
                prof_phase_y[r] = atan2(im, re);
            }
            double mu_y_1d;
            double Syy_init = fit_1d_variance(prof_mag_y, prof_phase_y, prof_ref_y,
                                               k_y, corr_h, k_max_init_y, &mu_y_1d);

            // Validate
            if (Sxx_init < 0.0) Sxx_init = 0.1;
            if (Syy_init < 0.0) Syy_init = 0.1;

            // Use profile-based k_max directly (capped by k_max_limit)
            double k_max_x = k_max_x_prof;
            double k_max_y = k_max_y_prof;

            // Update diagnostics with k_max values
            if (out_diagnostics) {
                out_diagnostics[i * 4 + 2] = k_max_x;
                out_diagnostics[i * 4 + 3] = k_max_y;
            }

            // Save initial guess
            build_params_from_fit(out_initial_guess + i * KSPACE_PARAMS,
                                  mu_x_init, mu_y_init, Sxx_init, Syy_init, 0.0,
                                  amp_A, amp_B, amp_AB, center_x, center_y);

            // ---- Step 7: Full 5-param fit ----
            // Compute T_norm = (F_AB / F_ref) / T(0) and collect valid k-points
            double epsilon = fmax(F_dc_clean, 1.0) * 1e-8;
            size_t n_valid = 0;

            // T(0) at DC
            size_t dc_idx = center_idx_y * corr_w + center_idx_x;
            double T0_re = F_AB_real[dc_idx] / (F_ref[dc_idx] + epsilon);
            double T0_im = F_AB_imag[dc_idx] / (F_ref[dc_idx] + epsilon);
            double T0_mag = sqrt(T0_re * T0_re + T0_im * T0_im);

            if (T0_mag < 1e-6) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            // T0_inv = conj(T0) / |T0|^2 for division
            double T0_inv_re =  T0_re / (T0_mag * T0_mag);
            double T0_inv_im = -T0_im / (T0_mag * T0_mag);

            // Collect valid k-points within elliptical mask
            double k_max_x_sq = k_max_x * k_max_x;
            double k_max_y_sq = k_max_y * k_max_y;

            for (size_t p = 0; p < n_pixels; p++) {
                double kx = K_X[p], ky = K_Y[p];
                if (kx * kx / k_max_x_sq + ky * ky / k_max_y_sq > 1.0)
                    continue;

                // T_measured = F_AB / F_ref
                double Tm_re = F_AB_real[p] / (F_ref[p] + epsilon);
                double Tm_im = F_AB_imag[p] / (F_ref[p] + epsilon);

                // T_norm = T_measured * T0_inv (complex multiply)
                double Tn_re = Tm_re * T0_inv_re - Tm_im * T0_inv_im;
                double Tn_im = Tm_re * T0_inv_im + Tm_im * T0_inv_re;

                T_norm_r[n_valid] = Tn_re;
                T_norm_i[n_valid] = Tn_im;
                main_K_X[n_valid] = kx;
                main_K_Y[n_valid] = ky;
                n_valid++;
            }

            if (n_valid < 10) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            // Compute weights
            if (use_soft_weighting) {
                // SNR weighting
                double max_fref = 0.0;
                for (size_t j = 0; j < n_valid; j++) {
                    // Find F_ref at this k-point's original index
                    // We need to reconstruct... simpler: store F_ref for valid points
                }
                // Recompute: iterate valid k-points again to get F_ref values
                size_t vi = 0;
                double *valid_fref = main_wts;  // Reuse temporarily
                for (size_t p = 0; p < n_pixels; p++) {
                    double kx = K_X[p], ky = K_Y[p];
                    if (kx * kx / k_max_x_sq + ky * ky / k_max_y_sq > 1.0)
                        continue;
                    valid_fref[vi] = fabs(F_ref[p]);
                    vi++;
                }

                double max_w_snr = 0.0;
                double sqrt_noise = sqrt(noise_power) + 1e-12;
                for (size_t j = 0; j < n_valid; j++) {
                    double w_snr = valid_fref[j] / sqrt_noise;
                    if (w_snr > max_w_snr) max_w_snr = w_snr;
                    main_wts[j] = w_snr;
                }
                if (max_w_snr > 1e-12) {
                    for (size_t j = 0; j < n_valid; j++)
                        main_wts[j] /= max_w_snr;
                }

                // Anisotropic soft decay
                double k0_x_sq = 1.0 / (2.0 * M_PI * M_PI * fmax(Sxx_init, 0.01) + 1e-12);
                double k0_y_sq = 1.0 / (2.0 * M_PI * M_PI * fmax(Syy_init, 0.01) + 1e-12);
                for (size_t j = 0; j < n_valid; j++) {
                    double kx = main_K_X[j], ky = main_K_Y[j];
                    double w_soft = exp(-kx * kx / k0_x_sq - ky * ky / k0_y_sq);
                    main_wts[j] *= w_soft;
                }
            } else {
                for (size_t j = 0; j < n_valid; j++)
                    main_wts[j] = 1.0;
            }

            // Set up main fit
            struct main_fit_data mdata;
            mdata.n = n_valid;
            mdata.K_X = main_K_X;
            mdata.K_Y = main_K_Y;
            mdata.T_norm_real = T_norm_r;
            mdata.T_norm_imag = T_norm_i;
            mdata.weights = main_wts;

            gsl_multifit_nlinear_fdf main_fdf;
            main_fdf.f      = main_residual_f;
            main_fdf.df     = main_residual_df;
            main_fdf.fvv    = NULL;
            main_fdf.n      = 2 * n_valid;  // Real + imag stacked
            main_fdf.p      = MAIN_NPARAMS;
            main_fdf.params = &mdata;

            // Allocate main workspace for this n_valid
            main_work = gsl_multifit_nlinear_alloc(T_gsl, &fdf_params_main,
                                                    2 * n_valid, MAIN_NPARAMS);
            if (!main_work) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            double main_p0[5] = {mu_x_init, mu_y_init, Sxx_init, Syy_init, 0.0};
            gsl_vector_view main_xv = gsl_vector_view_array(main_p0, MAIN_NPARAMS);
            int main_init_status = gsl_multifit_nlinear_init(&main_xv.vector, &main_fdf, main_work);

            int fit_ok = 0;
            double mu_x_fit = mu_x_init, mu_y_fit = mu_y_init;
            double Sxx_fit = Sxx_init, Syy_fit = Syy_init, Sxy_fit = 0.0;

            if (main_init_status == GSL_SUCCESS) {
                int info;
                int main_status = gsl_multifit_nlinear_driver(
                    MAIN_MAX_ITER, XTOL, GTOL, FTOL, NULL, NULL, &info, main_work);

                if (main_status == GSL_SUCCESS || main_status == GSL_EMAXITER) {
                    gsl_vector *x_result = gsl_multifit_nlinear_position(main_work);
                    mu_x_fit = gsl_vector_get(x_result, 0);
                    mu_y_fit = gsl_vector_get(x_result, 1);
                    Sxx_fit  = gsl_vector_get(x_result, 2);
                    Syy_fit  = gsl_vector_get(x_result, 3);
                    Sxy_fit  = gsl_vector_get(x_result, 4);

                    // Also accept if cost/n_points is small
                    double cost = 0.0;
                    gsl_vector *f_vec = gsl_multifit_nlinear_residual(main_work);
                    gsl_blas_ddot(f_vec, f_vec, &cost);
                    if (main_status == GSL_SUCCESS || cost / (double)n_valid < 1.0) {
                        fit_ok = 1;
                    }
                }
            }

            gsl_multifit_nlinear_free(main_work);
            main_work = NULL;

            if (!fit_ok) {
                out_status[i] = STATUS_NO_CONVERGE;
                continue;
            }

            // ---- Step 8: Validation ----
            // Clamp variance to zero (optimizer already clamps internally via
            // fmax in residual/Jacobian, but final value can still be slightly
            // negative due to floating-point). No rejection — displacement is
            // valid and stress near zero is physically plausible (e.g., near walls).
            if (Sxx_fit < 0.0) Sxx_fit = 0.0;
            if (Syy_fit < 0.0) Syy_fit = 0.0;

            double max_disp_x = 0.75 * (double)corr_w;
            double max_disp_y = 0.75 * (double)corr_h;
            if (fabs(mu_x_fit) > max_disp_x || fabs(mu_y_fit) > max_disp_y) {
                out_status[i] = STATUS_BIG_DISP;
                continue;
            }

            // ---- Success ----
            build_params_from_fit(out_params + i * KSPACE_PARAMS,
                                  mu_x_fit, mu_y_fit, Sxx_fit, Syy_fit, Sxy_fit,
                                  amp_A, amp_B, amp_AB, center_x, center_y);
            out_status[i] = STATUS_SUCCESS;
            success_count++;
        }

        // Free thread-local resources
        if (plan) {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                fftwf_destroy_plan(plan);
            }
        }
        fftwf_free(fft_in);
        fftwf_free(fft_out_AA);
        fftwf_free(fft_out_BB);
        fftwf_free(fft_out_AB);
        fftwf_free(fft_tmp);
        fftwf_free(fft_in_c);
        free(F_ref);
        free(P_noise);
        free(F_ref_norm);
        free(joint_wts);
        free(T_norm_r);
        free(T_norm_i);
        free(main_K_X);
        free(main_K_Y);
        free(main_wts);
        free(prof_mag_x);
        free(prof_phase_x);
        free(prof_ref_x);
        free(prof_mag_y);
        free(prof_phase_y);
        free(prof_ref_y);
        free(F_ref_abs);
        free(F_AB_real);
        free(F_AB_imag);
        if (joint_work) gsl_multifit_nlinear_free(joint_work);

    #ifdef _OPENMP
    }
    #endif

    free(k_x);
    free(k_y);
    free(K_X);
    free(K_Y);

    fprintf(stderr, "[kspace] completed: %d/%zu succeeded\n", success_count, num_windows);

    // Diagnostic: status breakdown when 0% success
    if (success_count == 0 && num_windows > 0) {
        int n_masked = 0, n_low_snr = 0, n_no_converge = 0;
        int n_big_disp = 0, n_neg_var = 0;
        for (size_t i = 0; i < num_windows; i++) {
            switch (out_status[i]) {
                case STATUS_MASKED:     n_masked++; break;
                case STATUS_LOW_SNR:    n_low_snr++; break;
                case STATUS_NO_CONVERGE: n_no_converge++; break;
                case STATUS_BIG_DISP:   n_big_disp++; break;
                case STATUS_NEG_VAR:    n_neg_var++; break;
            }
        }
        fprintf(stderr, "[kspace] DIAGNOSTIC (0%% success): masked=%d, low_snr=%d, "
                "no_converge=%d, big_disp=%d, neg_var=%d\n",
                n_masked, n_low_snr, n_no_converge, n_big_disp, n_neg_var);
    }

    return success_count;
}
