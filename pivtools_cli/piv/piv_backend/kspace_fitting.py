"""
K-Space Transfer Function Fitting for Ensemble PIV

This module implements k-space (Fourier domain) fitting for Reynolds stress
extraction in ensemble PIV. The key advantage over physical-space Gaussian
fitting is that the particle image contribution cancels algebraically:

    F(R_AB) = sqrt(F(R_AA) * F(R_BB)) * T(k)

where T(k) is the transfer function encoding only velocity PDF parameters:

    T(k) = A * exp(-2*pi^2 * k^T * Sigma * k) * exp(-2*pi*i * k . mu)

This assumes a Gaussian velocity PDF (normal distribution) and reduces
the problem from 16 parameters (physical-space) to 5:
    - mu_x, mu_y: Mean displacement
    - Sigma_xx, Sigma_yy, Sigma_xy: Reynolds stress tensor components

The fitting is implemented in C (libkspace) using FFTW (float32) for FFTs
and GSL (double) for nonlinear least-squares, with OpenMP parallelization.
This provides ~50-100x speedup over the previous Python/SciPy implementation.

**Anisotropic soft decay weighting**:
    The fitting uses combined SNR and soft decay weighting to optimally
    balance signal quality vs model validity:

    w(k) = w_snr(k) * w_soft(k)

    where:
    - w_snr = |F_ref| / noise_floor  (emphasizes high-signal regions)
    - w_soft = exp(-k_x^2/k0_x^2 - k_y^2/k0_y^2)  (anisotropic decay)
    - k0_x = 1 / sqrt(2*pi^2 * Sigma_xx)  (x decay wavenumber)
    - k0_y = 1 / sqrt(2*pi^2 * Sigma_yy)  (y decay wavenumber)

    The anisotropic weighting matches the elliptical decay of the transfer
    function when Sigma_xx != Sigma_yy, properly handling flows with
    different turbulence intensities in x and y directions.

    This avoids hard k_max caps by naturally down-weighting regions
    where the model becomes unreliable.
"""

import ctypes
import logging
import os
from typing import Optional

import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, fftfreq

from pivtools_cli.piv.piv_backend.interpolation_noise_psd import (
    compute_noise_psd_2d, frac_distance
)

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# Module-level cache for the kspace library
_kspace_lib = None


def _load_kspace_lib():
    """Load the kspace C library for k-space fitting."""
    global _kspace_lib
    if _kspace_lib is not None:
        return _kspace_lib

    lib_extension = ".dll" if os.name == "nt" else ".so"

    possible_paths = [
        # Installed package location (relative to this file)
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib", f"libkspace{lib_extension}")),
        # Development: from current working directory
        os.path.abspath(os.path.join("pivtools_cli", "lib", f"libkspace{lib_extension}")),
    ]

    for path in possible_paths:
        libpath = os.path.abspath(path)
        if os.path.isfile(libpath):
            break
    else:
        raise FileNotFoundError(
            f"Kspace library not found. Tried paths: {possible_paths}"
        )

    logging.debug(f"Loading kspace library from: {libpath}")
    lib = ctypes.CDLL(libpath)

    # Set up ctypes bindings
    lib.fit_kspace_batch.argtypes = [
        ctypes.c_size_t,     # num_windows
        ctypes.c_size_t,     # corr_h
        ctypes.c_size_t,     # corr_w
        np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # R_AA
        np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # R_BB
        np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # R_AB
        np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'),    # mask
        ctypes.c_void_p,     # pred_disp (nullable)
        ctypes.c_int,        # interp_kernel
        ctypes.c_int,        # use_soft_weighting
        ctypes.c_double,     # k_max_cap
        np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS'),  # out_params
        np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'),    # out_status
        np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS'),  # out_initial_guess
        ctypes.c_void_p,     # out_diagnostics (nullable)
    ]
    lib.fit_kspace_batch.restype = ctypes.c_int

    _kspace_lib = lib
    return lib


def fit_windows_kspace(
    R_AA: np.ndarray,
    R_BB: np.ndarray,
    R_AB: np.ndarray,
    mask_flat: np.ndarray,
    corr_size: tuple,
    config,
    pass_idx: int,
    use_soft_weighting: bool = True,
    debug: bool = False,
    predictor_displacements: Optional[np.ndarray] = None,
    interp_kernel: str = 'bicubic',
    k_max_cap: Optional[float] = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit k-space transfer function to correlation planes.

    This function provides a drop-in replacement for fit_windows_openmp()
    using k-space fitting instead of Levenberg-Marquardt Gaussian fitting.

    Parameters
    ----------
    R_AA : np.ndarray
        Flattened auto-correlation A planes, shape (num_windows * corr_h * corr_w,)
    R_BB : np.ndarray
        Flattened auto-correlation B planes
    R_AB : np.ndarray
        Flattened cross-correlation AB planes
    mask_flat : np.ndarray
        Boolean mask, shape (num_windows,). True = skip this window.
    corr_size : tuple
        (height, width) of correlation window
    config : Config
        Configuration object
    pass_idx : int
        Current pass index
    use_soft_weighting : bool
        If True (default), use combined SNR × anisotropic soft decay weighting.
        If False, use uniform weighting within k_max bounds.
    predictor_displacements : np.ndarray, optional
        Per-window predictor displacement in pixels, shape (n_windows, 2)
        where [:, 0] = dy, [:, 1] = dx. None on pass 0 (no warping).
    interp_kernel : str
        Interpolation kernel used for warping: 'bicubic' or 'lanczos3'.
    k_max_cap : float, optional
        Hard cap on k_max in cycles/pixel (0.1 to 0.5). If None, uses 0.35
        (avoids corner region used for noise estimation).

    Returns
    -------
    gauss_flat : np.ndarray
        Fitted parameters in 16-element format for compatibility.
        Shape: (num_windows, 16)

        Index mapping:
        [0-2]: amp_A, amp_B, amp_AB (peak values from correlation planes)
        [3-5]: c_A, c_B, c_AB = 0 (not used in k-space)
        [6-8]: sig_A_x, sig_A_y, sig_A_xy = NaN (not estimated in k-space)
        [9-11]: sig_AB_x, sig_AB_y, sig_AB_xy = Sigma_xx, Sigma_yy, Sigma_xy
        [12-13]: x0_A, y0_A = window center
        [14-15]: x0_AB, y0_AB = center + mu_x, center + mu_y (displacement)

    status_flat : np.ndarray
        Status codes, shape (num_windows,)
        -1 = masked/skipped, 0 = success, >0 = error code

    initial_guess_flat : np.ndarray
        Initial guesses used, shape (num_windows, 16)
    """
    lib = _load_kspace_lib()

    corr_h, corr_w = corr_size
    num_windows = len(mask_flat)

    # Validate input shapes
    n_per_window = corr_h * corr_w
    expected_size = num_windows * n_per_window
    if R_AA.size != expected_size:
        raise ValueError(
            f"R_AA size {R_AA.size} != expected {expected_size} "
            f"(num_windows={num_windows}, corr_size={corr_size})"
        )

    # Prepare inputs (ensure contiguous float32 — matches C correlator output)
    R_AA_c = np.ascontiguousarray(R_AA.ravel(), dtype=np.float32)
    R_BB_c = np.ascontiguousarray(R_BB.ravel(), dtype=np.float32)
    R_AB_c = np.ascontiguousarray(R_AB.ravel(), dtype=np.float32)

    # C expects mask: 1=skip, 0=process (inverse of Python boolean mask)
    mask_c = np.ascontiguousarray(mask_flat.astype(np.int32))

    # Predictor displacement (nullable)
    pred_ptr = None
    if predictor_displacements is not None:
        pred_c = np.ascontiguousarray(predictor_displacements, dtype=np.float64)
        pred_ptr = pred_c.ctypes.data

    # Allocate outputs
    gauss_flat = np.zeros((num_windows, 16), dtype=np.float64)
    status_flat = np.full(num_windows, -1, dtype=np.int32)
    initial_guess_flat = np.zeros((num_windows, 16), dtype=np.float64)
    diagnostics_flat = None
    diag_ptr = None
    if return_diagnostics:
        diagnostics_flat = np.full((num_windows, 4), np.nan, dtype=np.float64)
        diag_ptr = diagnostics_flat.ctypes.data

    kernel_code = 1 if interp_kernel == 'lanczos3' else 0
    cap_val = k_max_cap if k_max_cap is not None else -1.0

    logger.info(
        f"Pass {pass_idx + 1}: K-space fitting {np.sum(~mask_flat)}/{num_windows} windows "
        f"(soft_weighting={use_soft_weighting}, k_max_cap={cap_val:.2f})"
    )

    success_count = lib.fit_kspace_batch(
        num_windows, corr_h, corr_w,
        R_AA_c, R_BB_c, R_AB_c, mask_c,
        pred_ptr, kernel_code,
        int(use_soft_weighting),
        cap_val,
        gauss_flat, status_flat, initial_guess_flat,
        diag_ptr,
    )

    # Log summary
    n_valid = int(np.sum(~mask_flat))
    if n_valid > 0:
        success_rate = success_count / n_valid
        logger.info(
            f"Pass {pass_idx + 1}: K-space fitting success rate: {success_rate:.1%} "
            f"({success_count}/{n_valid} non-masked windows)"
        )

        if debug:
            status_counts = {}
            for s in status_flat:
                status_counts[int(s)] = status_counts.get(int(s), 0) + 1
            status_names = {-1: 'masked', 0: 'success', 1: 'no_converge',
                            2: 'low_snr', 3: 'big_disp', 5: 'neg_var'}
            status_str = ', '.join(
                [f"{status_names.get(k, f'status_{k}')}={v}"
                 for k, v in sorted(status_counts.items()) if v > 0]
            )
            logger.info(f"  Status breakdown: {status_str}")

            # Log parameter statistics for successful fits
            success_mask = status_flat == 0
            if np.any(success_mask):
                center_x = corr_w / 2.0 + 1
                center_y = corr_h / 2.0 + 1
                mu_x = gauss_flat[success_mask, 14] - center_x
                mu_y = gauss_flat[success_mask, 15] - center_y
                logger.info(f"  Sigma_xx (UU): min={np.nanmin(gauss_flat[success_mask, 9]):.4f}, "
                            f"median={np.nanmedian(gauss_flat[success_mask, 9]):.4f}, "
                            f"max={np.nanmax(gauss_flat[success_mask, 9]):.4f}")
                logger.info(f"  Sigma_yy (VV): min={np.nanmin(gauss_flat[success_mask, 10]):.4f}, "
                            f"median={np.nanmedian(gauss_flat[success_mask, 10]):.4f}, "
                            f"max={np.nanmax(gauss_flat[success_mask, 10]):.4f}")
                logger.info(f"  mu_x: min={np.nanmin(mu_x):.4f}, median={np.nanmedian(mu_x):.4f}, "
                            f"max={np.nanmax(mu_x):.4f}")
                logger.info(f"  mu_y: min={np.nanmin(mu_y):.4f}, median={np.nanmedian(mu_y):.4f}, "
                            f"max={np.nanmax(mu_y):.4f}")

    if return_diagnostics:
        return gauss_flat, status_flat, initial_guess_flat, diagnostics_flat
    return gauss_flat, status_flat, initial_guess_flat


def plot_kspace_diagnostic(
    R_AA_2d: np.ndarray,
    R_BB_2d: np.ndarray,
    R_AB_2d: np.ndarray,
    true_params: Optional[dict] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[dict]:
    """
    Create diagnostic k-space plot for debugging and visualization.

    This function stays in Python as it's a visualization tool that runs
    on single windows interactively, not in the hot path.

    Parameters
    ----------
    R_AA_2d : np.ndarray
        Auto-correlation A plane, shape (h, w)
    R_BB_2d : np.ndarray
        Auto-correlation B plane, shape (h, w)
    R_AB_2d : np.ndarray
        Cross-correlation AB plane, shape (h, w)
    true_params : dict, optional
        Ground truth parameters {'mu_x', 'mu_y', 'Sigma_xx', 'Sigma_yy', 'Sigma_xy'}
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    dict
        Fitted parameters and diagnostics, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available for k-space diagnostic plot")
        return None

    corr_h, corr_w = R_AA_2d.shape
    center_idx_x = corr_w // 2
    center_idx_y = corr_h // 2

    # Run the C fitter on a single window to get results
    mask = np.array([False], dtype=bool)
    R_AA_flat = R_AA_2d.ravel().astype(np.float32)
    R_BB_flat = R_BB_2d.ravel().astype(np.float32)
    R_AB_flat = R_AB_2d.ravel().astype(np.float32)

    gauss_flat, status_flat, initial_guess_flat = fit_windows_kspace(
        R_AA_flat, R_BB_flat, R_AB_flat,
        mask, (corr_h, corr_w), None, 0,
    )

    center_x = corr_w / 2.0 + 1
    center_y = corr_h / 2.0 + 1

    mu_x_init = initial_guess_flat[0, 14] - center_x
    mu_y_init = initial_guess_flat[0, 15] - center_y
    Sigma_xx_init = initial_guess_flat[0, 9]
    Sigma_yy_init = initial_guess_flat[0, 10]

    # Build wavenumber grids for plotting
    k_x = fftshift(fftfreq(corr_w))
    k_y = fftshift(fftfreq(corr_h))
    K_X, K_Y = np.meshgrid(k_x, k_y, indexing='xy')

    # Compute FFTs for visualization
    F_AA = fftshift(fft2(ifftshift(R_AA_2d)))
    F_BB = fftshift(fft2(ifftshift(R_BB_2d)))
    F_AB = fftshift(fft2(ifftshift(R_AB_2d)))

    F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
    epsilon = np.max(np.abs(F_ref)) * 1e-8
    T_measured = F_AB / (F_ref + epsilon)
    T_mag = np.abs(T_measured)
    T_phase = np.angle(T_measured)

    F_dc = F_ref[center_idx_y, center_idx_x]

    # Estimate k_max from profile
    def _kmax_from_prof(k_axis, profile, F_dc, threshold_frac=0.01, min_k=0.05, max_k=0.45):
        threshold = F_dc * threshold_frac
        center = len(k_axis) // 2
        k_pos = k_axis[center:]
        F_pos = profile[center:]
        below = F_pos < threshold
        if np.any(below):
            idx = np.argmax(below)
            km = k_pos[max(0, idx - 1)]
        else:
            km = max_k
        return np.clip(km, min_k, max_k)

    F_ref_profile_x = np.abs(F_ref[center_idx_y, :])
    k_max_x = _kmax_from_prof(k_x, F_ref_profile_x, F_dc)

    F_ref_profile_y = np.abs(F_ref[:, center_idx_x])
    k_max_y = _kmax_from_prof(k_y, F_ref_profile_y, F_dc)

    k_max_init_x = min(k_max_x, 0.45)
    k_max_init_y = min(k_max_y, 0.45)

    # SNR estimate
    snr = F_dc ** 2 / (1e-12 + 1e-12)

    T_profile_x = T_measured[center_idx_y, :]
    T_profile_y = T_measured[:, center_idx_x]

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Row 1: Physical space correlation planes
    ax1 = fig.add_subplot(3, 4, 1)
    im1 = ax1.imshow(R_AA_2d, cmap='RdBu_r', origin='lower')
    ax1.set_title('R_AA (auto-corr A)')
    ax1.set_xlabel('x [px]')
    ax1.set_ylabel('y [px]')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(3, 4, 2)
    im2 = ax2.imshow(R_BB_2d, cmap='RdBu_r', origin='lower')
    ax2.set_title('R_BB (auto-corr B)')
    ax2.set_xlabel('x [px]')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(3, 4, 3)
    im3 = ax3.imshow(R_AB_2d, cmap='RdBu_r', origin='lower')
    ax3.set_title('R_AB (cross-corr)')
    ax3.set_xlabel('x [px]')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Row 1, col 4: log magnitude of F_ref with k-bounds
    ax4 = fig.add_subplot(3, 4, 4)
    log_F_ref = np.log10(np.abs(F_ref) + 1e-12)
    im4 = ax4.imshow(
        log_F_ref, cmap='viridis', origin='lower',
        extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]]
    )
    ellipse = Ellipse(
        (0, 0), 2*k_max_x, 2*k_max_y,
        fill=False, edgecolor='red', linewidth=2, linestyle='--',
        label=f'k_max ({k_max_x:.2f}, {k_max_y:.2f})'
    )
    ax4.add_patch(ellipse)
    ax4.set_title('log10|F_ref| + k-bounds')
    ax4.set_xlabel('k_x [cycles/px]')
    ax4.set_ylabel('k_y [cycles/px]')
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylim(-0.5, 0.5)
    ax4.legend(loc='upper right', fontsize=8)
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    # Row 2: Transfer function magnitude and phase
    ax5 = fig.add_subplot(3, 4, 5)
    log_T_mag = np.log10(T_mag + 1e-12)
    im5 = ax5.imshow(
        log_T_mag, cmap='viridis', origin='lower',
        extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]]
    )
    ellipse2 = Ellipse(
        (0, 0), 2*k_max_x, 2*k_max_y,
        fill=False, edgecolor='red', linewidth=2, linestyle='--'
    )
    ax5.add_patch(ellipse2)
    ax5.set_title('log10|T| (transfer function)')
    ax5.set_xlabel('k_x [cycles/px]')
    ax5.set_ylabel('k_y [cycles/px]')
    ax5.set_xlim(-0.5, 0.5)
    ax5.set_ylim(-0.5, 0.5)
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    ax6 = fig.add_subplot(3, 4, 6)
    im6 = ax6.imshow(
        T_phase, cmap='twilight', origin='lower',
        extent=[k_x[0], k_x[-1], k_y[0], k_y[-1]],
        vmin=-np.pi, vmax=np.pi
    )
    ellipse3 = Ellipse(
        (0, 0), 2*k_max_x, 2*k_max_y,
        fill=False, edgecolor='white', linewidth=2, linestyle='--'
    )
    ax6.add_patch(ellipse3)
    ax6.set_title('phase(T) [rad]')
    ax6.set_xlabel('k_x [cycles/px]')
    ax6.set_ylabel('k_y [cycles/px]')
    ax6.set_xlim(-0.5, 0.5)
    ax6.set_ylim(-0.5, 0.5)
    plt.colorbar(im6, ax=ax6, shrink=0.8)

    # Row 2, cols 3-4: 1D magnitude profiles with fits
    ax7 = fig.add_subplot(3, 4, 7)
    valid_k_x = np.abs(k_x) < k_max_x
    k_x_valid = k_x[valid_k_x]
    mag_x = np.abs(T_profile_x[valid_k_x])
    ax7.semilogy(k_x_valid, mag_x, 'b.-', label='|T(k_x, 0)|', markersize=4)

    if Sigma_xx_init > 0:
        k_fit = np.linspace(-k_max_x, k_max_x, 100)
        mag_fit = np.exp(-2 * np.pi**2 * Sigma_xx_init * k_fit**2)
        ax7.semilogy(k_fit, mag_fit, 'r-', lw=2,
                     label=f'fit: Sigma_xx={Sigma_xx_init:.4f}')
    if true_params and 'Sigma_xx' in true_params:
        k_fit = np.linspace(-k_max_x, k_max_x, 100)
        mag_true = np.exp(-2 * np.pi**2 * true_params['Sigma_xx'] * k_fit**2)
        ax7.semilogy(k_fit, mag_true, 'g--', lw=2,
                     label=f'true: Sigma_xx={true_params["Sigma_xx"]:.4f}')
    ax7.axvline(-k_max_x, color='gray', linestyle=':', alpha=0.7)
    ax7.axvline(k_max_x, color='gray', linestyle=':', alpha=0.7)
    ax7.set_xlabel('k_x [cycles/px]')
    ax7.set_ylabel('|T|')
    ax7.set_title('|T| profile along k_y=0')
    ax7.legend(fontsize=8)
    ax7.set_ylim(1e-3, 2)
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(3, 4, 8)
    valid_k_y = np.abs(k_y) < k_max_y
    k_y_valid = k_y[valid_k_y]
    mag_y = np.abs(T_profile_y[valid_k_y])
    ax8.semilogy(k_y_valid, mag_y, 'b.-', label='|T(0, k_y)|', markersize=4)

    if Sigma_yy_init > 0:
        k_fit_y = np.linspace(-k_max_y, k_max_y, 100)
        mag_fit_y = np.exp(-2 * np.pi**2 * Sigma_yy_init * k_fit_y**2)
        ax8.semilogy(k_fit_y, mag_fit_y, 'r-', lw=2,
                     label=f'fit: Sigma_yy={Sigma_yy_init:.4f}')
    if true_params and 'Sigma_yy' in true_params:
        k_fit_y = np.linspace(-k_max_y, k_max_y, 100)
        mag_true_y = np.exp(-2 * np.pi**2 * true_params['Sigma_yy'] * k_fit_y**2)
        ax8.semilogy(k_fit_y, mag_true_y, 'g--', lw=2,
                     label=f'true: Sigma_yy={true_params["Sigma_yy"]:.4f}')
    ax8.axvline(-k_max_y, color='gray', linestyle=':', alpha=0.7)
    ax8.axvline(k_max_y, color='gray', linestyle=':', alpha=0.7)
    ax8.set_xlabel('k_y [cycles/px]')
    ax8.set_ylabel('|T|')
    ax8.set_title('|T| profile along k_x=0')
    ax8.legend(fontsize=8)
    ax8.set_ylim(1e-3, 2)
    ax8.grid(True, alpha=0.3)

    # Row 3: Phase profiles and summary
    ax9 = fig.add_subplot(3, 4, 9)
    phase_k_max = min(k_max_x, 0.25)
    valid_k_phase = (np.abs(k_x) > 1.5 / corr_w) & (np.abs(k_x) < phase_k_max)
    k_x_phase = k_x[valid_k_phase]
    phase_x = np.angle(T_profile_x[valid_k_phase])
    ax9.plot(k_x_phase, phase_x, 'b.-', label='phase(T(k_x, 0))', markersize=4)

    k_fit_phase = np.linspace(-phase_k_max, phase_k_max, 100)
    phase_fit = -2 * np.pi * mu_x_init * k_fit_phase
    ax9.plot(k_fit_phase, phase_fit, 'r-', lw=2,
             label=f'fit: mu_x={mu_x_init:.4f}')
    if true_params and 'mu_x' in true_params:
        phase_true = -2 * np.pi * true_params['mu_x'] * k_fit_phase
        ax9.plot(k_fit_phase, phase_true, 'g--', lw=2,
                 label=f'true: mu_x={true_params["mu_x"]:.4f}')
    ax9.axhline(-np.pi, color='gray', linestyle=':', alpha=0.5)
    ax9.axhline(np.pi, color='gray', linestyle=':', alpha=0.5)
    ax9.set_xlabel('k_x [cycles/px]')
    ax9.set_ylabel('phase [rad]')
    ax9.set_title('phase(T) profile along k_y=0')
    ax9.legend(fontsize=8)
    ax9.set_ylim(-np.pi*1.2, np.pi*1.2)
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(3, 4, 10)
    phase_k_max_y = min(k_max_y, 0.25)
    valid_k_phase_y = (np.abs(k_y) > 1.5 / corr_h) & (np.abs(k_y) < phase_k_max_y)
    k_y_phase = k_y[valid_k_phase_y]
    phase_y = np.angle(T_profile_y[valid_k_phase_y])
    ax10.plot(k_y_phase, phase_y, 'b.-', label='phase(T(0, k_y))', markersize=4)

    k_fit_phase_y = np.linspace(-phase_k_max_y, phase_k_max_y, 100)
    phase_fit_y = -2 * np.pi * mu_y_init * k_fit_phase_y
    ax10.plot(k_fit_phase_y, phase_fit_y, 'r-', lw=2,
              label=f'fit: mu_y={mu_y_init:.4f}')
    if true_params and 'mu_y' in true_params:
        phase_true_y = -2 * np.pi * true_params['mu_y'] * k_fit_phase_y
        ax10.plot(k_fit_phase_y, phase_true_y, 'g--', lw=2,
                  label=f'true: mu_y={true_params["mu_y"]:.4f}')
    ax10.axhline(-np.pi, color='gray', linestyle=':', alpha=0.5)
    ax10.axhline(np.pi, color='gray', linestyle=':', alpha=0.5)
    ax10.set_xlabel('k_y [cycles/px]')
    ax10.set_ylabel('phase [rad]')
    ax10.set_title('phase(T) profile along k_x=0')
    ax10.legend(fontsize=8)
    ax10.set_ylim(-np.pi*1.2, np.pi*1.2)
    ax10.grid(True, alpha=0.3)

    # Summary panel
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')
    summary_text = f"""K-Space Diagnostic Summary
{'='*35}
Initial k_max: ({k_max_init_x:.3f}, {k_max_init_y:.3f})

Adaptive k-bounds:
  k_max_x: {k_max_x:.3f}
  k_max_y: {k_max_y:.3f}

Initial Estimates:
  mu_x: {mu_x_init:.4f} px
  mu_y: {mu_y_init:.4f} px
  Sigma_xx: {Sigma_xx_init:.4f} px^2
  Sigma_yy: {Sigma_yy_init:.4f} px^2"""

    if true_params:
        summary_text += f"""

Ground Truth:
  mu_x: {true_params.get('mu_x', 'N/A'):.4f} px
  mu_y: {true_params.get('mu_y', 'N/A'):.4f} px
  Sigma_xx: {true_params.get('Sigma_xx', 'N/A'):.4f} px^2
  Sigma_yy: {true_params.get('Sigma_yy', 'N/A'):.4f} px^2"""

    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
              fontsize=10, family='monospace', verticalalignment='top')

    # log|T| vs k^2 plot (linearized)
    ax12 = fig.add_subplot(3, 4, 12)
    valid_k_sq = (np.abs(k_x) > 1.5 / corr_w) & (np.abs(k_x) < k_max_x)
    k_x_sq = k_x[valid_k_sq] ** 2
    log_mag_x = np.log(np.maximum(np.abs(T_profile_x[valid_k_sq]), 1e-12))
    ax12.plot(k_x_sq, log_mag_x, 'b.-', label='ln|T| vs k_x^2', markersize=4)

    k_sq_fit = np.linspace(0, k_max_x**2, 100)
    log_fit = -2 * np.pi**2 * Sigma_xx_init * k_sq_fit
    ax12.plot(k_sq_fit, log_fit, 'r-', lw=2,
              label=f'slope = -2pi^2 * {Sigma_xx_init:.4f}')
    if true_params and 'Sigma_xx' in true_params:
        log_true = -2 * np.pi**2 * true_params['Sigma_xx'] * k_sq_fit
        ax12.plot(k_sq_fit, log_true, 'g--', lw=2,
                  label=f'true slope')
    ax12.set_xlabel('k_x^2 [cycles^2/px^2]')
    ax12.set_ylabel('ln|T|')
    ax12.set_title('Linearized magnitude fit')
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"K-space diagnostic saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'mu_x_1d': mu_x_init,
        'mu_y_1d': mu_y_init,
        'Sigma_xx_1d': Sigma_xx_init,
        'Sigma_yy_1d': Sigma_yy_init,
        'k_max_x': k_max_x,
        'k_max_y': k_max_y,
        'status': int(status_flat[0]),
    }
