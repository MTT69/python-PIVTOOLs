/******************************************************************************
 * kspace_lm_fit.h — public surface of libkspacefit.
 *
 * The k-space transfer-function LM ensemble fitter. Implementation and the
 * full design rationale live in kspace_lm_fit.c; the Python caller is
 * pivtools_cli/piv/piv_backend/kspace_lm_fitting.py.
 *
 * NOTE: every symbol callable from ctypes must carry KSPACE_EXPORT. Functions
 * declared without it (as in codelet_fft.h) are fine on Linux/macOS via default
 * visibility but are absent from the Windows DLL export table.
 ******************************************************************************/
#pragma once

#ifdef _WIN32
#define KSPACE_EXPORT __declspec(dllexport)
#else
#define KSPACE_EXPORT __attribute__((visibility("default")))
#endif

/* Fit every unmasked window. Array layouts mirror fit_windows_kspace_lm.
 *
 * Inputs:
 *   R_AA/R_BB/R_AB  (n_windows, corr_h, corr_w) C-contiguous correlation planes
 *   mask_flat       (n_windows,) 1 = masked, skip
 *   P_win           (n_windows, corr_h*corr_w) coloured-floor shape in Python's
 *                   CENTRED k-layout, or NULL for the flat floor (mirrors the
 *                   Python D=None switch)
 *   use_kx4/use_ky4 quartic shape terms; K = 7 + use_kx4 + use_ky4
 *   n_threads       OpenMP threads for this call; <=0 uses the OpenMP default.
 *                   Applied via a num_threads() clause, so it does NOT mutate
 *                   the process-global ICV.
 *
 * Outputs (caller-allocated, all length n_windows unless noted). Every output
 * is fully initialised here — the caller need not pre-zero, and NONE may be
 * NULL, including diag_b4x/diag_b4y when the quartic terms are disabled (they
 * are NaN-filled unconditionally).
 *   gauss_flat          (n_windows, 16)
 *   status_flat         int32; -1 masked, 0 ok, 1 no-converge, 2 low-SNR,
 *                       3 big-displacement
 *   initial_guess_flat  (n_windows, 16). Masked rows stay all-zero rather than
 *                       taking the NaN + centre defaults, matching production,
 *                       which fills this only over the unmasked index set
 *                       (kspace_lm_fitting.py:754-761). The sole exception is
 *                       an all-masked batch, where production mirrors
 *                       gauss_flat into it.
 *   diag_*              per-window diagnostics
 *
 * Returns 0 on success, or:
 *   -1  corr_h or corr_w outside BUILT_FFT_SIZES (no codelet exists)
 *   -2  scratch allocation failure
 *   -3  coloured floor requested but no |k| >= COLOURED_SEED_KR_MIN bin exists
 *       to seed N0 from (unreachable for any BUILT_FFT_SIZES length; guards a
 *       future smaller size). */
KSPACE_EXPORT int kspace_lm_fit_batch(
    const double *R_AA, const double *R_BB, const double *R_AB,
    const unsigned char *mask_flat, const double *P_win, int n_windows,
    int corr_h, int corr_w, int use_kx4, int use_ky4, int n_threads,
    double *gauss_flat, int *status_flat, double *initial_guess_flat,
    double *diag_gain, double *diag_N0, double *diag_b4x, double *diag_b4y,
    double *diag_cost_per_pt, int *diag_n_valid, int *diag_iter,
    unsigned char *diag_conv);

/* omp_get_max_threads(), or 1 when built without OpenMP. Doubles as a liveness
 * probe: the Python loader calls it to prove the DLL resolved and links libomp. */
KSPACE_EXPORT int kspace_lm_fit_max_threads(void);

/* --- Gate entry points -----------------------------------------------------
 * The two below serve the OUT-OF-TREE harness manual_tools/kspace_c_port/
 * test_kspace_c_port.py, which gates this library against saved production
 * planes. Nothing under unit-tests/ calls them and _load_kspace_lib declares no
 * argtypes for them — an in-repo caller MUST set argtypes first, or 64-bit
 * pointers are truncated to c_int. */

/* Single-window regressor + N0 seed, for the ordering gate only. A centred vs
 * natural mix-up in P does not crash — it produces a plausible but wrong fit —
 * so out_Dv (natural order, length H*W) is gated directly against Python's
 * _prepare_chunk rather than inferred from a parameter mismatch.
 * Pass P_centred = NULL for the flat regressor. Return codes as above. */
KSPACE_EXPORT int kspace_debug_prep(const double *R_AA, const double *R_BB,
                                    const double *R_AB, const double *P_centred,
                                    int H, int W, double *out_Dv,
                                    double *out_n0_seed);

/* FFT-only entry for gating the transform against np.fft: out_re/out_im are the
 * (H, W) natural-order spectra of the centred transform. */
KSPACE_EXPORT int kspace_fft2_centered(const double *plane, int H, int W,
                                       double *out_re, double *out_im);
