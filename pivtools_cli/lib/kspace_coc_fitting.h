#ifndef KSPACE_COC_FITTING_H
#define KSPACE_COC_FITTING_H

/*
 * kspace_coc_fitting.h — Implicit model + noise floor for CoC k-space fitting.
 *
 * Model: |F(CoC)|(k) = F_ref(k) × A × exp(-2π² k^T Σ k) + N0
 *
 * F_ref = √(|F(AA₁)|·|F(BB₁)|·|F(AA₂)|·|F(BB₂)|) cancels particle.
 * N0 captures the flat noise floor in |F(CoC)| from finite ensemble
 * averaging. At high k, F_ref×Gaussian→0 and the model → N0.
 * No P_noise needed — kernel coloring is already in F_ref.
 *
 * The fitted Σ is particle-free (cancelled by F_ref algebraically).
 *
 * Uses: FFTW3f (float32 FFTs) + GSL (double-precision nonlinear LS) + OpenMP.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__WIN32__)
  #define PIV_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
  #define PIV_EXPORT __attribute__((visibility("default")))
#else
  #define PIV_EXPORT
#endif

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
    double *diag_F_coc,       /* nullable */
    double *diag_F_ref,       /* nullable */
    double *out_spread_xx,
    double *out_spread_yy,
    double *out_spread_xy,
    double *out_center_x,
    double *out_center_y,
    int    *out_status
);

#ifdef __cplusplus
}
#endif

#endif /* KSPACE_COC_FITTING_H */
