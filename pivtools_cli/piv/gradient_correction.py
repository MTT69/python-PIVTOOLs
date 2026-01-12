"""
Gradient correction for ensemble PIV Reynolds stresses.

Applies correction for velocity gradient bias in stress estimates:
    UU_corrected = UU_stress - 0.5 * sig_A_x * (dU/dy)²
    VV_corrected = VV_stress - 0.5 * sig_A_y * (dV/dx)²
    UV_corrected = UV_stress - 0.5 * sig_A_x * (dU/dy + dV/dx)

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradient-corrected Reynolds stresses.

    Parameters
    ----------
    U : np.ndarray
        Mean x-velocity field (physical coords)
    V : np.ndarray
        Mean y-velocity field (physical coords, already sign-corrected)
    sig_A_x : np.ndarray
        Gaussian width parameter in x
    sig_A_y : np.ndarray
        Gaussian width parameter in y
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

    Returns
    -------
    tuple
        (UU_corrected, VV_corrected, UV_corrected)
    """
    # Compute velocity gradients using SIGNED spacing
    # axis=0 is rows (y direction), axis=1 is columns (x direction)
    dU_dy = np.gradient(U, dy, axis=0)
    dV_dx = np.gradient(V, dx, axis=1)

    # Compute corrections
    # UU correction: 0.5 * sig_A_x * (dU/dy)²
    UU_correction = 0.5 * sig_A_x * (dU_dy ** 2)

    # VV correction: 0.5 * sig_A_y * (dV/dx)²
    VV_correction = 0.5 * sig_A_y * (dV_dx ** 2)

    # UV correction: 0.5 * sig_A_x * (dU/dy + dV/dx)
    UV_correction = 0.5 * sig_A_x * (dU_dy + dV_dx)

    # Apply corrections
    UU_corrected = UU_stress - UU_correction
    VV_corrected = VV_stress - VV_correction
    UV_corrected = UV_stress - UV_correction

    return UU_corrected, VV_corrected, UV_corrected


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Gaussian width parameter in x (required for correction)
    sig_A_y : np.ndarray or None
        Gaussian width parameter in y (required for correction)
    win_ctrs_x : np.ndarray
        Window center x coordinates (1D, pixel coords)
    win_ctrs_y : np.ndarray
        Window center y coordinates (1D, pixel coords)
    image_height : int
        Image height for coordinate conversion

    Returns
    -------
    tuple
        (UU_corrected, VV_corrected, UV_corrected) or original stresses if correction not possible
    """
    # Check if we have required parameters
    if sig_A_x is None or sig_A_y is None:
        logging.warning("sig_A_x or sig_A_y not available, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress

    if ux is None or uy is None:
        logging.warning("Velocity fields not available, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress

    if UU_stress is None or VV_stress is None or UV_stress is None:
        logging.warning("Stress fields not available, skipping gradient correction")
        return UU_stress, VV_stress, UV_stress

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

    logging.debug(f"Gradient correction: dx={dx:.2f}, dy={dy:.2f} (physical coords)")

    # Apply correction
    UU_corrected, VV_corrected, UV_corrected = compute_gradient_corrections(
        U=ux,
        V=uy,
        sig_A_x=sig_A_x,
        sig_A_y=sig_A_y,
        UU_stress=UU_stress,
        VV_stress=VV_stress,
        UV_stress=UV_stress,
        dx=dx,
        dy=dy,
    )

    return UU_corrected, VV_corrected, UV_corrected
