#ifndef INTERP_NOISE_PSD_H
#define INTERP_NOISE_PSD_H

/*
 * interp_noise_psd.h — Interpolation kernel DTFT functions for noise PSD.
 *
 * Computes |H(k, f)|^2 for bicubic (Keys a=-0.75) and Lanczos-3 kernels,
 * where k is wavenumber (cycles/pixel) and f is fractional pixel displacement.
 *
 * The noise PSD is: P_noise(kx, ky) = |H(kx, fx)|^2 * |H(ky, fy)|^2
 *
 * Shared between kspace_fitting.c (per-camera) and kspace_coc_fitting.c (CoC).
 * All functions are static inline to avoid linker conflicts.
 *
 * Ported from interpolation_noise_psd.py.
 */

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Distance to nearest integer: 0 at integers, 0.5 at half-pixels. */
static inline double noise_frac_distance(double x) {
    double rem = fabs(fmod(x, 1.0));
    return (rem <= 0.5) ? rem : 1.0 - rem;
}

/* Bicubic kernel weights (Keys a=-0.75, 4-tap).
 * Offsets: [-1, 0, +1, +2] relative to floor(position). */
static inline void noise_bicubic_weights(double f, double w[4]) {
    double a = -0.75;
    double f2 = f * f;
    double f3 = f2 * f;
    w[0] = a * f3 - 2.0 * a * f2 + a * f;
    w[1] = (a + 2.0) * f3 - (a + 3.0) * f2 + 1.0;
    w[2] = -(a + 2.0) * f3 + (2.0 * a + 3.0) * f2 - a * f;
    w[3] = -a * f3 + a * f2;
}

/* |H(k, f)|^2 for bicubic kernel (Keys a=-0.75). */
static inline double noise_bicubic_mag_sq(double k, double f) {
    double w[4];
    noise_bicubic_weights(f, w);
    double twopik = 2.0 * M_PI * k;
    double Hr = w[0] * cos(twopik) + w[1] + w[2] * cos(twopik) + w[3] * cos(2.0 * twopik);
    double Hi = w[0] * sin(twopik) - w[2] * sin(twopik) - w[3] * sin(2.0 * twopik);
    return Hr * Hr + Hi * Hi;
}

/* Lanczos-3 single weight: sinc(t) * sinc(t/3), |t| < 3. */
static inline double noise_lanczos3_single_weight(double t) {
    if (fabs(t) < 1e-12) return 1.0;
    if (fabs(t) >= 3.0) return 0.0;
    double pit = M_PI * t;
    return (sin(pit) / pit) * (sin(pit / 3.0) / (pit / 3.0));
}

/* |H(k, f)|^2 for Lanczos-3 kernel (6-tap windowed sinc). */
static inline double noise_lanczos3_mag_sq(double k, double f) {
    int offsets[6] = {-2, -1, 0, 1, 2, 3};
    double raw[6], w[6];
    double sum = 0.0;
    int i;
    for (i = 0; i < 6; i++) {
        raw[i] = noise_lanczos3_single_weight(f - offsets[i]);
        sum += raw[i];
    }
    if (sum < 1e-12) return 1.0;
    for (i = 0; i < 6; i++) w[i] = raw[i] / sum;

    double twopik = 2.0 * M_PI * k;
    double Hr = 0.0, Hi = 0.0;
    for (i = 0; i < 6; i++) {
        double angle = -offsets[i] * twopik;
        Hr += w[i] * cos(angle);
        Hi += w[i] * sin(angle);
    }
    return Hr * Hr + Hi * Hi;
}

/* Compute P_noise value at one (kx, ky) point.
 * interp_kernel: 0=bicubic, 1=lanczos3. */
static inline double noise_psd_value(double kx, double ky,
                                      double f_x, double f_y,
                                      int interp_kernel) {
    double hx_sq, hy_sq;
    if (interp_kernel == 1) {
        hx_sq = noise_lanczos3_mag_sq(kx, f_x);
        hy_sq = noise_lanczos3_mag_sq(ky, f_y);
    } else {
        hx_sq = noise_bicubic_mag_sq(kx, f_x);
        hy_sq = noise_bicubic_mag_sq(ky, f_y);
    }
    return hx_sq * hy_sq;
}

#endif /* INTERP_NOISE_PSD_H */
