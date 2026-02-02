"""
Gradient correction for ensemble PIV Reynolds stresses.

Applies correction for velocity gradient bias in stress estimates based on the
derivation from autocorrelation/cross-correlation peak fitting analysis.

The complete correction formulas account for two effects:
1. Window averaging effect: L²/12 × gradient² (uniform distribution variance)
2. Particle extent effect: σ_A × gradient² (particle image variance)

Correction formulas:
    Corr_uu = (∂u/∂x)² × (L_x²/12 + σ_A_x) + (∂u/∂y)² × (L_y²/12 + σ_A_y)
    Corr_vv = (∂v/∂x)² × (L_x²/12 + σ_A_x) + (∂v/∂y)² × (L_y²/12 + σ_A_y)
    Corr_uv = (∂u/∂x)(∂v/∂x) × (L_x²/12 + σ_A_x) + (∂u/∂y)(∂v/∂y) × (L_y²/12 + σ_A_y)

Where:
- L_x, L_y = particle window dimensions (from config.ensemble_window_sizes)
- σ_A_x, σ_A_y = particle image VARIANCE from autocorrelation fit (already σ², not σ)

Sign convention notes:
- This module operates on data in physical coordinates (as saved to .mat files)
- Y decreases with row index (image convention), so dy is negative
- numpy.gradient correctly handles negative spacing
"""
import logging
from typing import Optional, Tuple

import numpy as np


def compute_gradient_corrections(
    U: np.ndarray,
    V: np.ndarray,
    sig_A_x: np.ndarray,
    sig_A_y: np.ndarray,
    UU_stress: np.ndarray,
    VV_stress: np.ndarray,
    UV_stress: np.ndarray,
    dx: float,
    dy: float,
    window_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradient-corrected Reynolds stresses with separate window and particle corrections.

    Parameters
    ----------
    U : np.ndarray
        Mean x-velocity field (physical coords)
    V : np.ndarray
        Mean y-velocity field (physical coords, already sign-corrected)
    sig_A_x : np.ndarray
        Particle image VARIANCE in x (σ_A_x²) from autocorrelation fit.
        This is already a variance value - use directly without squaring.
    sig_A_y : np.ndarray
        Particle image VARIANCE in y (σ_A_y²) from autocorrelation fit.
        This is already a variance value - use directly without squaring.
    UU_stress : np.ndarray
        Raw UU Reynolds stress
    VV_stress : np.ndarray
        Raw VV Reynolds stress
    UV_stress : np.ndarray
        Raw UV Reynolds stress (physical coords, already sign-corrected)
    dx : float
        Grid spacing in x (positive)
    dy : float
        Grid spacing in y (SIGNED - negative if Y decreases with row index)
    window_size : Tuple[int, int]
        Particle window dimensions (L_y, L_x) in pixels

    Returns
    -------
    tuple of 9 arrays
        (UU_corrected, VV_corrected, UV_corrected,
         UU_window_corr, VV_window_corr, UV_window_corr,
         UU_particle_corr, VV_particle_corr, UV_particle_corr)

        - First 3: Corrected stresses (stress - total_correction)
        - Next 3: Window averaging corrections (L²/12 effect)
        - Last 3: Particle extent corrections (σ_A effect)
    """
    L_y, L_x = window_size

    # Compute ALL velocity gradients using SIGNED spacing
    # axis=0 is rows (y direction), axis=1 is columns (x direction)
    dU_dx = np.gradient(U, dx, axis=1)
    dU_dy = np.gradient(U, dy, axis=0)
    dV_dx = np.gradient(V, dx, axis=1)
    dV_dy = np.gradient(V, dy, axis=0)

    # Window averaging effect: L²/12 (variance of uniform distribution over window)
    L_x_sq_12 = (L_x ** 2) / 12.0
    L_y_sq_12 = (L_y ** 2) / 12.0

    # =========================================================================
    # UU corrections: Corr_uu = (∂u/∂x)² × (L_x²/12 + σ_A_x) + (∂u/∂y)² × (L_y²/12 + σ_A_y)
    # =========================================================================
    dU_dx_sq = dU_dx ** 2
    dU_dy_sq = dU_dy ** 2

    # Window correction for UU
    UU_window_corr = L_x_sq_12 * dU_dx_sq + L_y_sq_12 * dU_dy_sq

    # Particle correction for UU (sig_A is already variance, use directly)
    UU_particle_corr = sig_A_x * dU_dx_sq + sig_A_y * dU_dy_sq

    # =========================================================================
    # VV corrections: Corr_vv = (∂v/∂x)² × (L_x²/12 + σ_A_x) + (∂v/∂y)² × (L_y²/12 + σ_A_y)
    # =========================================================================
    dV_dx_sq = dV_dx ** 2
    dV_dy_sq = dV_dy ** 2

    # Window correction for VV
    VV_window_corr = L_x_sq_12 * dV_dx_sq + L_y_sq_12 * dV_dy_sq

    # Particle correction for VV (sig_A is already variance, use directly)
    VV_particle_corr = sig_A_x * dV_dx_sq + sig_A_y * dV_dy_sq

    # =========================================================================
    # UV corrections: Corr_uv = (∂u/∂x)(∂v/∂x) × (L_x²/12 + σ_A_x) + (∂u/∂y)(∂v/∂y) × (L_y²/12 + σ_A_y)
    # Note: This is a PRODUCT of gradients, not a sum
    # =========================================================================
    dU_dV_dx = dU_dx * dV_dx  # Product of x-gradients
    dU_dV_dy = dU_dy * dV_dy  # Product of y-gradients

    # Window correction for UV
    UV_window_corr = L_x_sq_12 * dU_dV_dx + L_y_sq_12 * dU_dV_dy

    # Particle correction for UV (sig_A is already variance, use directly)
    UV_particle_corr = sig_A_x * dU_dV_dx + sig_A_y * dU_dV_dy

    # =========================================================================
    # Total corrections and corrected stresses
    # =========================================================================
    UU_total_corr = UU_window_corr + UU_particle_corr
    VV_total_corr = VV_window_corr + VV_particle_corr
    UV_total_corr = UV_window_corr + UV_particle_corr

    UU_corrected = UU_stress - UU_total_corr
    VV_corrected = VV_stress - VV_total_corr
    UV_corrected = UV_stress - UV_total_corr

    return (UU_corrected, VV_corrected, UV_corrected,
            UU_window_corr, VV_window_corr, UV_window_corr,
            UU_particle_corr, VV_particle_corr, UV_particle_corr)


def apply_gradient_correction_to_pass(
    ux: np.ndarray,
    uy: np.ndarray,
    UU_stress: np.ndarray,
    VV_stress: np.ndarray,
    UV_stress: np.ndarray,
    sig_A_x: Optional[np.ndarray],
    sig_A_y: Optional[np.ndarray],
    win_ctrs_x: np.ndarray,
    win_ctrs_y: np.ndarray,
    image_height: int,
    window_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply gradient correction to a single pass result.

    This function operates on data that has ALREADY been converted to physical
    coordinates (uy negated, UV_stress negated).

    Parameters
    ----------
    ux : np.ndarray
        X-velocity field (physical coords)
    uy : np.ndarray
        Y-velocity field (physical coords, already negated)
    UU_stress : np.ndarray
        Raw UU Reynolds stress
    VV_stress : np.ndarray
        Raw VV Reynolds stress
    UV_stress : np.ndarray
        Raw UV Reynolds stress (physical coords, already negated)
    sig_A_x : np.ndarray or None
        Particle image VARIANCE in x (σ_A_x²) from autocorrelation fit.
        This is already a variance value - use directly without squaring.
    sig_A_y : np.ndarray or None
        Particle image VARIANCE in y (σ_A_y²) from autocorrelation fit.
        This is already a variance value - use directly without squaring.
    win_ctrs_x : np.ndarray
        Window center x coordinates (1D, pixel coords)
    win_ctrs_y : np.ndarray
        Window center y coordinates (1D, pixel coords)
    image_height : int
        Image height for coordinate conversion
    window_size : tuple of (int, int), optional
        Particle window dimensions (L_y, L_x) in pixels. Required for full correction.
        If not provided, only particle extent correction is applied (window correction = 0).

    Returns
    -------
    tuple of 9 values
        (UU_corrected, VV_corrected, UV_corrected,
         UU_window_corr, VV_window_corr, UV_window_corr,
         UU_particle_corr, VV_particle_corr, UV_particle_corr)

        - First 3: Corrected stresses (stress - total_correction)
        - Next 3: Window averaging corrections (L²/12 effect), or None if not available
        - Last 3: Particle extent corrections (σ_A effect), or None if not available

        If correction not possible, returns original stresses with None for all correction terms.
    """
    # Check if we have required parameters
    if sig_A_x is None or sig_A_y is None:
        logging.warning("sig_A_x or sig_A_y not available, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress, None, None, None, None, None, None

    if ux is None or uy is None:
        logging.warning("Velocity fields not available, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress, None, None, None, None, None, None

    if UU_stress is None or VV_stress is None or UV_stress is None:
        logging.warning("Stress fields not available, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress, None, None, None, None, None, None

    if window_size is None:
        logging.warning("window_size not provided, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress, None, None, None, None, None, None

    # Compute grid spacing from window centers
    # X spacing (always positive)
    if len(win_ctrs_x) > 1:
        dx = float(win_ctrs_x[1] - win_ctrs_x[0])
    else:
        dx = 1.0

    # Y spacing in physical coordinates
    # Physical Y = (H-1) - pixel_y, so if pixel_y increases, physical_y decreases
    # For consecutive rows: dy_physical = -dy_pixel (assuming pixel Y increases with row)
    if len(win_ctrs_y) > 1:
        dy_pixel = float(win_ctrs_y[1] - win_ctrs_y[0])
        # Physical Y decreases as pixel Y increases, so negate
        dy = -dy_pixel
    else:
        dy = -1.0

    logging.debug(f"Gradient correction: dx={dx:.2f}, dy={dy:.2f}, window_size={window_size} (physical coords)")

    # Apply correction
    (UU_corrected, VV_corrected, UV_corrected,
     UU_window_corr, VV_window_corr, UV_window_corr,
     UU_particle_corr, VV_particle_corr, UV_particle_corr) = compute_gradient_corrections(
        U=ux,
        V=uy,
        sig_A_x=sig_A_x,
        sig_A_y=sig_A_y,
        UU_stress=UU_stress,
        VV_stress=VV_stress,
        UV_stress=UV_stress,
        dx=dx,
        dy=dy,
        window_size=window_size,
    )

    return (UU_corrected, VV_corrected, UV_corrected,
            UU_window_corr, VV_window_corr, UV_window_corr,
            UU_particle_corr, VV_particle_corr, UV_particle_corr)
