"""
Stereo PIV Self-Calibration (Wieneke 2005).

Detects and corrects laser-sheet misalignment in stereo PIV by cross-correlating
dewarped images from Camera 1 vs Camera 2 (same time instant), measuring the
residual disparity, fitting it to a plane (Z-offset + tilts), and iterating.

Usage
-----
    from pivtools_gui.stereo_reconstruction.self_calibration import (
        PinholeCamera, run_self_calibration,
    )

    result = run_self_calibration(
        cam1, cam2,
        images_cam1, images_cam2,
        world_bounds=(-40, 40, -40, 40),
    )
"""

import ctypes
import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from pivtools_cli.piv.piv_backend.infilling import infill_nearest
from pivtools_cli.piv.piv_backend.outlier_detection import median_outlier_detection
from pivtools_core.window_utils import compute_window_centers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PinholeCamera:
    """Pinhole camera model compatible with OpenCV conventions."""

    K: np.ndarray  # (3,3) intrinsic matrix
    dist: np.ndarray  # (5,) distortion coefficients
    R: np.ndarray  # (3,3) rotation world→camera
    t: np.ndarray  # (3,1) translation world→camera
    image_size: Tuple[int, int]  # (width, height)

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project world points (N,3) to image coordinates (N,2)."""
        rvec, _ = cv2.Rodrigues(self.R)
        pts2d, _ = cv2.projectPoints(
            points_world.astype(np.float64),
            rvec,
            self.t.astype(np.float64),
            self.K.astype(np.float64),
            self.dist.astype(np.float64),
        )
        return pts2d.reshape(-1, 2)


@dataclass
class IterationRecord:
    """Record for one self-calibration iteration."""

    iteration: int
    rms_disparity: float
    delta_z: float
    delta_tilt_x: float
    delta_tilt_y: float
    cumulative_z: float
    cumulative_tilt_x: float
    cumulative_tilt_y: float


@dataclass
class SelfCalibrationResult:
    """Full result of self-calibration."""

    converged: bool
    n_iterations: int
    z_offset: float
    tilt_x: float
    tilt_y: float
    final_rms_disparity: float
    history: List[IterationRecord] = field(default_factory=list)
    dx_before: Optional[np.ndarray] = None
    dy_before: Optional[np.ndarray] = None
    dx_after: Optional[np.ndarray] = None
    dy_after: Optional[np.ndarray] = None
    grid_x_mm: Optional[np.ndarray] = None
    grid_y_mm: Optional[np.ndarray] = None
    peak_quality: Optional[np.ndarray] = None
    # Accumulated cross-correlation planes from the first and final
    # iterations, shape (n_win_y, n_win_x, ws, ws). Saved by the production
    # pipeline as `correlation_planes.mat` for diagnostic inspection. None
    # if the run aborted before iteration 1.
    corr_first_iter: Optional[np.ndarray] = None
    corr_last_iter: Optional[np.ndarray] = None
    win_ctrs_x: Optional[np.ndarray] = None
    win_ctrs_y: Optional[np.ndarray] = None
    n_win_x: Optional[int] = None
    n_win_y: Optional[int] = None
    window_size_used: Optional[int] = None
    mm_per_pixel: Optional[float] = None
    # Geometric disparity sensitivity from estimate_disparity_sensitivity.
    # Used by diagnostics to predict the expected disparity field from the
    # converged (z, tilt_x, tilt_y) for forward-model validation.
    disp_px_per_mm: Optional[float] = None
    disp_direction: Optional[np.ndarray] = None  # unit vector (dx, dy) in dewarped px


# ---------------------------------------------------------------------------
# Dewarping primitives
# ---------------------------------------------------------------------------


def estimate_pixel_scale(
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    world_bounds: Tuple[float, float, float, float],
    z: float = 0.0,
) -> float:
    """Estimate the native pixel scale (mm/pixel) on the world plane.

    Projects small steps in world X and Y through each camera to measure
    how many raw pixels correspond to 1 mm.  Returns the coarsest scale
    (largest mm/px) across both cameras and both directions, so that
    neither camera is upsampled in any direction.

    Parameters
    ----------
    cam1, cam2 : PinholeCamera instances
    world_bounds : (x_min, x_max, y_min, y_max) in mm
    z : Z-coordinate of the world plane (mm)

    Returns
    -------
    mm_per_pixel : float
    """
    x_min, x_max, y_min, y_max = world_bounds
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    step = 0.5  # mm

    center = np.array([[cx, cy, z]])
    dx_pt = np.array([[cx + step, cy, z]])
    dy_pt = np.array([[cx, cy + step, z]])

    mm_per_px_values = []
    for cam in [cam1, cam2]:
        p0 = cam.project(center)[0]
        px = cam.project(dx_pt)[0]
        py = cam.project(dy_pt)[0]

        px_per_mm_x = np.linalg.norm(px - p0) / step
        px_per_mm_y = np.linalg.norm(py - p0) / step

        # Coarsest direction for this camera
        coarsest_px_per_mm = min(px_per_mm_x, px_per_mm_y)
        mm_per_px_values.append(1.0 / coarsest_px_per_mm)

    # Use the coarsest camera (largest mm/px)
    return max(mm_per_px_values)


def estimate_disparity_sensitivity(
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    world_bounds: Tuple[float, float, float, float],
    mm_per_pixel: float,
    z: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Numerically estimate disparity sensitivity and direction.

    A particle at Z=dZ projects to different raw pixels in each camera.
    When dewarped to Z=0, each camera places the particle at a different
    apparent world position.  The disparity vector (in dewarped pixels)
    encodes both the magnitude and direction of the stereo mismatch.

    Works for any stereo geometry including transmission (~180°).

    Parameters
    ----------
    cam1, cam2 : PinholeCamera instances
    world_bounds : (x_min, x_max, y_min, y_max) in mm
    mm_per_pixel : dewarped output scale
    z : current Z-plane

    Returns
    -------
    disp_px_per_mm : float
        Disparity magnitude (dewarped pixels) per mm of Z-offset.
    disp_direction : ndarray, shape (2,)
        Unit vector [dx, dy] in dewarped pixel space giving the direction
        of the disparity.  For horizontal stereo this is ~[1, 0]; for
        transmission it may be ~[0, -1] or any direction.
    """
    from .global_coordinate_alignment import (
        _pixels_to_world_mm,
    )

    x_min, x_max, y_min, y_max = world_bounds
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    dz = 0.1  # mm perturbation

    pt_dz = np.array([[cx, cy, z + dz]])
    raw_px_cam1 = cam1.project(pt_dz)
    raw_px_cam2 = cam2.project(pt_dz)

    rvec1, _ = cv2.Rodrigues(cam1.R)
    rvec2, _ = cv2.Rodrigues(cam2.R)

    q1 = _pixels_to_world_mm(
        raw_px_cam1.astype(np.float32),
        cam1.K,
        cam1.dist,
        rvec1.flatten(),
        cam1.t.flatten(),
    )[0]

    q2 = _pixels_to_world_mm(
        raw_px_cam2.astype(np.float32),
        cam2.K,
        cam2.dist,
        rvec2.flatten(),
        cam2.t.flatten(),
    )[0]

    disp_mm = q2 - q1
    disp_px = disp_mm / mm_per_pixel
    disp_magnitude_px = float(np.linalg.norm(disp_px))
    disp_px_per_mm = disp_magnitude_px / dz

    # Unit direction vector
    if disp_magnitude_px > 1e-10:
        disp_direction = (disp_px / disp_magnitude_px).astype(np.float64)
    else:
        disp_direction = np.array([1.0, 0.0])  # default to horizontal

    angle_deg = float(np.degrees(np.arctan2(disp_direction[1], disp_direction[0])))
    logger.info(
        f"Disparity sensitivity: {disp_px_per_mm:.2f} dewarped-px/mm, "
        f"direction: [{disp_direction[0]:.3f}, {disp_direction[1]:.3f}] "
        f"({angle_deg:.1f} deg from horizontal)"
    )
    return disp_px_per_mm, disp_direction


def compute_dewarp_maps(
    camera: PinholeCamera,
    world_bounds: Tuple[float, float, float, float],
    mm_per_pixel: float,
    z_offset: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build remap tables that dewarp a camera image onto a world-XY plane.

    Parameters
    ----------
    camera : PinholeCamera
    world_bounds : (x_min, x_max, y_min, y_max) in mm
    mm_per_pixel : spatial scale of the dewarped output grid
    z_offset, tilt_x, tilt_y : laser-sheet correction parameters

    Returns
    -------
    map_x, map_y : float32 arrays of shape (out_h, out_w)
    """
    x_min, x_max, y_min, y_max = world_bounds
    out_w = max(1, int(round((x_max - x_min) / mm_per_pixel)))
    out_h = max(1, int(round((y_max - y_min) / mm_per_pixel)))

    # Build world-coordinate meshgrid
    world_x = np.linspace(x_min, x_max, out_w, dtype=np.float64)
    world_y = np.linspace(y_min, y_max, out_h, dtype=np.float64)
    wx, wy = np.meshgrid(world_x, world_y)

    # Z at each point from plane equation
    wz = z_offset + wx * np.tan(tilt_y) + wy * np.tan(tilt_x)

    # Stack as (N,3) world points
    world_pts = np.column_stack([wx.ravel(), wy.ravel(), wz.ravel()])

    # Project to image coordinates
    img_pts = camera.project(world_pts)

    map_x = img_pts[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = img_pts[:, 1].reshape(out_h, out_w).astype(np.float32)
    return map_x, map_y


def dewarp_image(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Remap an image using precomputed dewarp maps."""
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# C library correlation
# ---------------------------------------------------------------------------

_xcorr_lib = None


def _load_xcorr_library():
    """Load libbulkxcorr2d, caching at module level."""
    global _xcorr_lib
    if _xcorr_lib is not None:
        return _xcorr_lib

    lib_ext = ".dll" if os.name == "nt" else ".so"
    lib_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "pivtools_cli",
        "lib",
        f"libbulkxcorr2d{lib_ext}",
    )
    lib_path = os.path.abspath(lib_path)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Cross-correlation library not found: {lib_path}")

    lib = ctypes.CDLL(lib_path)
    # hasattr guard: locally built libs may predate the CPU-check symbol
    if hasattr(lib, "pivtools_cpu_supported") and not lib.pivtools_cpu_supported():
        raise RuntimeError(
            "pivtools binary wheels require AVX2+FMA (Intel Haswell 2013+ / "
            "AMD Excavator+) and this CPU lacks them. Install from source "
            "instead: pip install --no-binary pivtools pivtools"
        )

    lib.bulkxcorr2d_accumulate.restype = ctypes.c_ubyte
    lib.bulkxcorr2d_accumulate.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA_stack
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB_stack
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nImageSize
        ctypes.c_int,  # N_images
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nWindows
        np.ctypeslib.ndpointer(
            dtype=np.float32, flags="C_CONTIGUOUS"
        ),  # fWindowWeightA
        np.ctypeslib.ndpointer(
            dtype=np.float32, flags="C_CONTIGUOUS"
        ),  # fWindowWeightB
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nWindowSize
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),  # nFitWindowSize
        np.ctypeslib.ndpointer(
            dtype=np.float32, flags="C_CONTIGUOUS"
        ),  # fCorrelPlane_Sum (output)
    ]

    _xcorr_lib = lib
    return lib


def accumulate_ensemble_correlation(
    lib,
    dewarped_cam1: np.ndarray,
    dewarped_cam2: np.ndarray,
    win_ctrs_x: np.ndarray,
    win_ctrs_y: np.ndarray,
    n_win_x: int,
    n_win_y: int,
    window_size: int,
) -> np.ndarray:
    """Cross-correlate dewarped cam1 vs cam2 using C ensemble accumulation.

    Parameters
    ----------
    lib : ctypes.CDLL
    dewarped_cam1 : (N, H, W) float32
    dewarped_cam2 : (N, H, W) float32
    win_ctrs_x, win_ctrs_y : 1-D float32 window center arrays
    n_win_x, n_win_y : grid dimensions
    window_size : int

    Returns
    -------
    corr_sum : (n_win_y, n_win_x, ws, ws) float32 accumulated correlation
    """
    N, H, W = dewarped_cam1.shape
    ws = window_size
    total_windows = n_win_y * n_win_x

    # Prepare C-contiguous stacks
    stack_a = np.ascontiguousarray(dewarped_cam1, dtype=np.float32)
    stack_b = np.ascontiguousarray(dewarped_cam2, dtype=np.float32)

    # No masking — process all windows
    mask = np.zeros(total_windows, dtype=np.float32)

    image_size = np.array([H, W], dtype=np.int32)
    n_windows = np.array([n_win_y, n_win_x], dtype=np.int32)

    # Match main PIV ensemble pipeline: square window (all ones), no taper.
    weight = np.ones(ws * ws, dtype=np.float32)

    win_size_arr = np.array([ws, ws], dtype=np.int32)

    # Output buffer
    corr_sum = np.zeros(total_windows * ws * ws, dtype=np.float32)

    error_code = lib.bulkxcorr2d_accumulate(
        stack_a,
        stack_b,
        mask,
        image_size,
        N,
        np.ascontiguousarray(win_ctrs_x, dtype=np.float32),
        np.ascontiguousarray(win_ctrs_y, dtype=np.float32),
        n_windows,
        weight,
        weight,
        win_size_arr,
        win_size_arr,  # nFitWindowSize == nWindowSize (no central extraction)
        corr_sum,
    )

    if error_code != 0:
        raise RuntimeError(f"bulkxcorr2d_accumulate returned error code {error_code}")

    return corr_sum.reshape(n_win_y, n_win_x, ws, ws)


# ---------------------------------------------------------------------------
# 6-DOF Gaussian peak fitting (Python re-implementation of peak_locate_lm.c)
# ---------------------------------------------------------------------------


def _gaussian_6dof_model(params, i_coords, j_coords):
    """Rotated elliptical Gaussian model (type=6 from peak_locate_lm.c).

    G(i,j) = A * exp(-0.5 * (di^2*sx + dj^2*sy + 2*di*dj*sxy))
    """
    A, i0, j0, sx, sy, sxy = params
    di = i_coords - i0
    dj = j_coords - j0
    exponent = -0.5 * (di * di * sx + dj * dj * sy + 2.0 * di * dj * sxy)
    return A * np.exp(np.clip(exponent, -50, 50))


def _gaussian_6dof_residuals(params, i_coords, j_coords, data):
    return _gaussian_6dof_model(params, i_coords, j_coords) - data


def fit_gaussian_6dof_peak(corr_plane: np.ndarray):
    """Fit a 6-DOF rotated elliptical Gaussian to a correlation plane.

    Parameters
    ----------
    corr_plane : (ws, ws) float array

    Returns
    -------
    (dy, dx, peak_value) : sub-pixel displacement from centre, or (NaN, NaN, NaN)
    """
    ws = corr_plane.shape[0]
    center = ws // 2

    # Search central 75% for coarse peak
    margin = max(ws // 8, 2)
    search_region = corr_plane[margin : ws - margin, margin : ws - margin]
    coarse_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    iy = coarse_idx[0] + margin
    ix = coarse_idx[1] + margin

    # Reject if peak is on the absolute border
    if iy < 2 or iy >= ws - 2 or ix < 2 or ix >= ws - 2:
        return np.nan, np.nan, np.nan

    # Extract 5x5 region
    r = 2
    region = corr_plane[iy - r : iy + r + 1, ix - r : ix + r + 1].copy()
    region_min = region.min()
    region = region - region_min  # shift to zero baseline

    # Build coordinate grids for the 5x5 region
    ii, jj = np.mgrid[-r : r + 1, -r : r + 1]
    ii_flat = ii.ravel().astype(np.float64)
    jj_flat = jj.ravel().astype(np.float64)
    data_flat = region.ravel().astype(np.float64)

    # Initial guess via 3-point parabolic estimator
    cy = region[:, r]  # column through center x
    cx = region[r, :]  # row through center y

    eps = 1e-10
    log_vals_y = np.log(np.maximum(cy, eps))
    log_vals_x = np.log(np.maximum(cx, eps))

    # Sub-pixel offset from parabolic fit
    denom_y = log_vals_y[r - 1] - 2 * log_vals_y[r] + log_vals_y[r + 1]
    denom_x = log_vals_x[r - 1] - 2 * log_vals_x[r] + log_vals_x[r + 1]

    di0 = 0.0
    if abs(denom_y) > eps:
        di0 = 0.5 * (log_vals_y[r - 1] - log_vals_y[r + 1]) / denom_y

    dj0 = 0.0
    if abs(denom_x) > eps:
        dj0 = 0.5 * (log_vals_x[r - 1] - log_vals_x[r + 1]) / denom_x

    # Estimate sigma from parabolic curvature
    sx_init = max(0.25, min(9.0, -denom_y if abs(denom_y) > eps else 1.0))
    sy_init = max(0.25, min(9.0, -denom_x if abs(denom_x) > eps else 1.0))
    A_init = float(region.max())

    p0 = [A_init, di0, dj0, sx_init, sy_init, 0.0]

    try:
        result = least_squares(
            _gaussian_6dof_residuals,
            p0,
            args=(ii_flat, jj_flat, data_flat),
            method="lm",
            max_nfev=20,
        )
        A_fit, i0_fit, j0_fit = result.x[0], result.x[1], result.x[2]
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        logger.debug(f"Gaussian 6-DOF peak fit failed: {e}")
        return np.nan, np.nan, np.nan

    # Reject bad fits
    if abs(i0_fit) > r or abs(j0_fit) > r or A_fit <= 0:
        return np.nan, np.nan, np.nan

    # Displacement from center of the full plane
    dy = (iy + i0_fit) - center
    dx = (ix + j0_fit) - center
    peak_value = A_fit + region_min
    return dy, dx, peak_value


def extract_disparity_field(
    ensemble_corr: np.ndarray,
    n_images: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sub-pixel disparity from accumulated correlation planes.

    Parameters
    ----------
    ensemble_corr : (n_win_y, n_win_x, ws, ws) accumulated sum
    n_images : number of images that were accumulated

    Returns
    -------
    dx, dy, peak_quality : (n_win_y, n_win_x) arrays
    """
    n_win_y, n_win_x = ensemble_corr.shape[:2]
    dx = np.full((n_win_y, n_win_x), np.nan, dtype=np.float64)
    dy = np.full((n_win_y, n_win_x), np.nan, dtype=np.float64)
    peak_quality = np.full((n_win_y, n_win_x), np.nan, dtype=np.float64)

    for iy in range(n_win_y):
        for ix in range(n_win_x):
            plane = ensemble_corr[iy, ix].astype(np.float64)
            # Average and subtract noise floor
            plane = plane / n_images
            plane = plane - plane.min()

            d_y, d_x, pval = fit_gaussian_6dof_peak(plane)
            dy[iy, ix] = d_y
            dx[iy, ix] = d_x
            peak_quality[iy, ix] = pval

    return dx, dy, peak_quality


# ---------------------------------------------------------------------------
# Outlier detection + plane fitting
# ---------------------------------------------------------------------------


def clean_disparity_field(
    dx: np.ndarray,
    dy: np.ndarray,
    peak_quality: np.ndarray,
    quality_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers and infill the disparity field."""
    # Mask low-quality points
    low_q = peak_quality < quality_threshold
    dx_c = dx.copy()
    dy_c = dy.copy()
    dx_c[low_q] = np.nan
    dy_c[low_q] = np.nan

    # Median outlier detection on valid points
    dx_for_det = np.where(np.isnan(dx_c), 0.0, dx_c).astype(np.float32)
    dy_for_det = np.where(np.isnan(dy_c), 0.0, dy_c).astype(np.float32)
    outlier_mask = median_outlier_detection(dx_for_det, dy_for_det)
    combined_mask = outlier_mask | np.isnan(dx_c)

    # Set outliers to NaN before infilling
    dx_c[combined_mask] = np.nan
    dy_c[combined_mask] = np.nan

    # Infill
    dx_filled, dy_filled = infill_nearest(
        dx_c.astype(np.float32),
        dy_c.astype(np.float32),
        combined_mask,
    )
    return dx_filled.astype(np.float64), dy_filled.astype(np.float64)


def fit_disparity_plane(
    dx: np.ndarray,
    dy: np.ndarray,
    grid_x_mm: np.ndarray,
    grid_y_mm: np.ndarray,
    disp_px_per_mm: float,
    mm_per_pixel: float,
    disp_direction: Optional[np.ndarray] = None,
) -> dict:
    """Fit a plane to the disparity field projected onto the stereo direction.

    Projects the measured (dx, dy) disparity onto the known disparity
    direction (from ``estimate_disparity_sensitivity``), then fits a plane
    to recover Z-offset and tilts.  This works for any camera arrangement:
    horizontal stereo (disparity ~pure dx), vertical, or transmission
    (~pure dy).

    Parameters
    ----------
    dx, dy : (n_win_y, n_win_x) disparity in dewarped pixels
    grid_x_mm, grid_y_mm : world-coordinate grids in mm
    disp_px_per_mm : disparity magnitude per mm of Z-offset
    mm_per_pixel : spatial scale
    disp_direction : (2,) unit vector giving expected disparity direction.
        If None, defaults to [1, 0] (horizontal — backward compatible).

    Returns
    -------
    dict with z_offset, tilt_x, tilt_y, rms_residual
    """
    if disp_direction is None:
        disp_direction = np.array([1.0, 0.0])

    # Project (dx, dy) onto the disparity direction to get a scalar field
    valid = np.isfinite(dx) & np.isfinite(dy)
    X = grid_x_mm[valid].ravel()
    Y = grid_y_mm[valid].ravel()
    D = dx[valid].ravel() * disp_direction[0] + dy[valid].ravel() * disp_direction[1]

    # Least-squares: D_projected = a + b*X + c*Y
    A_mat = np.column_stack([np.ones_like(X), X, Y])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, D, rcond=None)
    a, b, c = coeffs

    residual = D - A_mat @ coeffs
    rms = float(np.sqrt(np.mean(residual**2)))

    # Convert projected disparity to physical corrections.
    # D_projected (px) = disp_px_per_mm * dZ(mm)
    # => dZ = D_projected / disp_px_per_mm
    if disp_px_per_mm < 1e-6:
        logger.warning(
            f"Disparity sensitivity too low ({disp_px_per_mm:.4f} px/mm) — "
            f"Z-offset estimation unreliable"
        )
        conversion = 0.0
    else:
        conversion = 1.0 / disp_px_per_mm
    z_offset = a * conversion
    tilt_y = np.arctan(b * conversion)
    tilt_x = np.arctan(c * conversion)

    return {
        "z_offset": float(z_offset),
        "tilt_x": float(tilt_x),
        "tilt_y": float(tilt_y),
        "rms_residual": rms,
        "coeffs": (a, b, c),
    }


# ---------------------------------------------------------------------------
# Main iterative loop
# ---------------------------------------------------------------------------


def run_self_calibration(
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    images_cam1: List[np.ndarray],
    images_cam2: List[np.ndarray],
    world_bounds: Tuple[float, float, float, float],
    window_size: int = 64,
    overlap: float = 50.0,
    max_iterations: int = 10,
    convergence_threshold: float = 0.1,
    quality_threshold: float = 0.3,
) -> SelfCalibrationResult:
    """Run iterative stereo PIV self-calibration.

    Parameters
    ----------
    cam1, cam2 : PinholeCamera instances
    images_cam1, images_cam2 : lists of uint8 images (same length)
    world_bounds : (x_min, x_max, y_min, y_max) in mm
    window_size : correlation window size in pixels
    overlap : window overlap percentage
    max_iterations : maximum correction iterations
    convergence_threshold : RMS disparity (px) below which to stop
    quality_threshold : minimum peak quality to accept a disparity vector

    Returns
    -------
    SelfCalibrationResult
    """
    n_images = len(images_cam1)
    x_min, x_max, y_min, y_max = world_bounds

    # Compute pixel scale from native camera resolution
    mm_per_pixel = estimate_pixel_scale(cam1, cam2, world_bounds)
    out_w = max(1, int(round((x_max - x_min) / mm_per_pixel)))
    out_h = max(1, int(round((y_max - y_min) / mm_per_pixel)))
    logger.info(
        f"Native pixel scale: {mm_per_pixel:.4f} mm/px -> "
        f"dewarped size {out_w}x{out_h}"
    )

    # Stereo angle (informational)
    R_rel = cam2.R @ cam1.R.T
    trace_val = np.trace(R_rel)
    full_angle = math.acos(max(-1.0, min(1.0, (trace_val - 1.0) / 2.0)))
    logger.info(f"Stereo full angle: {math.degrees(full_angle):.1f} deg")

    # Numerically compute disparity sensitivity and direction — works for any
    # stereo geometry including transmission (~180°) where the analytic
    # tan(θ) formula diverges.
    disp_px_per_mm, disp_direction = estimate_disparity_sensitivity(
        cam1,
        cam2,
        world_bounds,
        mm_per_pixel,
    )

    # Use the user-supplied window size verbatim. If the search range
    # (±window_size/2) is too small for the real disparity, the user
    # can bump the window themselves.
    logger.info(
        f"Self-cal window size: {window_size} px "
        f"(disparity sensitivity: {disp_px_per_mm:.1f} px/mm)"
    )

    # Window grid
    wc = compute_window_centers(
        (out_h, out_w),
        (window_size, window_size),
        overlap,
    )
    win_ctrs_x = wc.win_ctrs_x
    win_ctrs_y = wc.win_ctrs_y
    n_win_x = wc.n_win_x
    n_win_y = wc.n_win_y

    # World-coordinate grids for each window center
    world_x_1d = np.linspace(x_min, x_max, out_w)
    world_y_1d = np.linspace(y_min, y_max, out_h)
    grid_x_px = np.round(win_ctrs_x).astype(int)
    grid_y_px = np.round(win_ctrs_y).astype(int)
    grid_x_mm_1d = world_x_1d[np.clip(grid_x_px, 0, out_w - 1)]
    grid_y_mm_1d = world_y_1d[np.clip(grid_y_px, 0, out_h - 1)]
    grid_x_mm, grid_y_mm = np.meshgrid(grid_x_mm_1d, grid_y_mm_1d)

    # Load C library
    lib = _load_xcorr_library()

    cumulative_z = 0.0
    cumulative_tilt_x = 0.0
    cumulative_tilt_y = 0.0
    history = []
    dx_before = None
    dy_before = None
    dx_after = None
    dy_after = None
    peak_q = None
    corr_first_iter = None  # iteration 1 accumulated correlation planes
    corr_last_iter = None  # most recent accumulated correlation planes

    for iteration in range(max_iterations):
        logger.info(f"Self-calibration iteration {iteration + 1}/{max_iterations}")

        # Build dewarp maps with cumulative corrections
        maps_cam1 = compute_dewarp_maps(
            cam1,
            world_bounds,
            mm_per_pixel,
            cumulative_z,
            cumulative_tilt_x,
            cumulative_tilt_y,
        )
        maps_cam2 = compute_dewarp_maps(
            cam2,
            world_bounds,
            mm_per_pixel,
            cumulative_z,
            cumulative_tilt_x,
            cumulative_tilt_y,
        )

        # Dewarp all images
        dw1 = np.stack(
            [
                dewarp_image(img, maps_cam1[0], maps_cam1[1]).astype(np.float32)
                for img in images_cam1
            ]
        )
        dw2 = np.stack(
            [
                dewarp_image(img, maps_cam2[0], maps_cam2[1]).astype(np.float32)
                for img in images_cam2
            ]
        )

        # Cross-correlate cam1 vs cam2
        corr_sum = accumulate_ensemble_correlation(
            lib,
            dw1,
            dw2,
            win_ctrs_x,
            win_ctrs_y,
            n_win_x,
            n_win_y,
            window_size,
        )

        # Capture correlation planes — first iteration is kept verbatim,
        # last iteration is overwritten each loop until we exit
        if iteration == 0:
            corr_first_iter = corr_sum.copy()
        corr_last_iter = corr_sum.copy()

        # Extract disparity field
        dx_raw, dy_raw, peak_q = extract_disparity_field(corr_sum, n_images)

        # Store "before" on first iteration
        if iteration == 0:
            dx_before = dx_raw.copy()
            dy_before = dy_raw.copy()

        # Clean
        dx_clean, dy_clean = clean_disparity_field(
            dx_raw,
            dy_raw,
            peak_q,
            quality_threshold,
        )

        # RMS disparity
        valid = np.isfinite(dx_clean) & np.isfinite(dy_clean)
        if not valid.any():
            logger.warning("No valid disparity points — aborting")
            break

        rms = float(np.sqrt(np.mean(dx_clean[valid] ** 2 + dy_clean[valid] ** 2)))
        logger.info(f"  RMS disparity: {rms:.4f} px")

        # Fit plane to get corrections
        fit = fit_disparity_plane(
            dx_clean,
            dy_clean,
            grid_x_mm,
            grid_y_mm,
            disp_px_per_mm,
            mm_per_pixel,
            disp_direction=disp_direction,
        )
        delta_z = fit["z_offset"]
        delta_tx = fit["tilt_x"]
        delta_ty = fit["tilt_y"]

        cumulative_z += delta_z
        cumulative_tilt_x += delta_tx
        cumulative_tilt_y += delta_ty

        history.append(
            IterationRecord(
                iteration=iteration + 1,
                rms_disparity=rms,
                delta_z=delta_z,
                delta_tilt_x=delta_tx,
                delta_tilt_y=delta_ty,
                cumulative_z=cumulative_z,
                cumulative_tilt_x=cumulative_tilt_x,
                cumulative_tilt_y=cumulative_tilt_y,
            )
        )

        logger.info(
            f"  Corrections: dZ={delta_z:.4f} mm, "
            f"tilt_x={math.degrees(delta_tx):.4f} deg, "
            f"tilt_y={math.degrees(delta_ty):.4f} deg"
        )
        logger.info(
            f"  Cumulative: Z={cumulative_z:.4f} mm, "
            f"tilt_x={math.degrees(cumulative_tilt_x):.4f} deg, "
            f"tilt_y={math.degrees(cumulative_tilt_y):.4f} deg"
        )

        # Convergence: corrections have stabilised (parameters stopped changing).
        # Check that all three corrections are below threshold, meaning the
        # algorithm has found the misalignment and further iterations won't help.
        corrections_stable = (
            abs(delta_z) < convergence_threshold * mm_per_pixel
            and abs(delta_tx) < math.radians(0.01)
            and abs(delta_ty) < math.radians(0.01)
        )
        if corrections_stable and iteration >= 1:
            logger.info(
                f"Converged at iteration {iteration + 1} — corrections "
                f"stabilised (dZ={delta_z:.4f} mm, RMS={rms:.4f} px)"
            )
            dx_after = dx_raw.copy()
            dy_after = dy_raw.copy()
            return SelfCalibrationResult(
                converged=True,
                n_iterations=iteration + 1,
                z_offset=cumulative_z,
                tilt_x=cumulative_tilt_x,
                tilt_y=cumulative_tilt_y,
                final_rms_disparity=rms,
                history=history,
                dx_before=dx_before,
                dy_before=dy_before,
                dx_after=dx_after,
                dy_after=dy_after,
                grid_x_mm=grid_x_mm,
                grid_y_mm=grid_y_mm,
                peak_quality=peak_q,
                corr_first_iter=corr_first_iter,
                corr_last_iter=corr_last_iter,
                win_ctrs_x=win_ctrs_x,
                win_ctrs_y=win_ctrs_y,
                n_win_x=n_win_x,
                n_win_y=n_win_y,
                window_size_used=window_size,
                mm_per_pixel=mm_per_pixel,
                disp_px_per_mm=float(disp_px_per_mm),
                disp_direction=np.asarray(disp_direction, dtype=np.float64),
            )

    # Did not converge — do one final pass to get "after" disparity
    logger.info("Generating final disparity field...")
    maps_cam1 = compute_dewarp_maps(
        cam1,
        world_bounds,
        mm_per_pixel,
        cumulative_z,
        cumulative_tilt_x,
        cumulative_tilt_y,
    )
    maps_cam2 = compute_dewarp_maps(
        cam2,
        world_bounds,
        mm_per_pixel,
        cumulative_z,
        cumulative_tilt_x,
        cumulative_tilt_y,
    )
    dw1 = np.stack(
        [
            dewarp_image(img, maps_cam1[0], maps_cam1[1]).astype(np.float32)
            for img in images_cam1
        ]
    )
    dw2 = np.stack(
        [
            dewarp_image(img, maps_cam2[0], maps_cam2[1]).astype(np.float32)
            for img in images_cam2
        ]
    )
    corr_sum = accumulate_ensemble_correlation(
        lib,
        dw1,
        dw2,
        win_ctrs_x,
        win_ctrs_y,
        n_win_x,
        n_win_y,
        window_size,
    )
    # The post-loop pass IS the most recent correlation we have at the
    # final cumulative correction — overwrite corr_last_iter with it.
    corr_last_iter = corr_sum.copy()
    dx_after, dy_after, _ = extract_disparity_field(corr_sum, n_images)

    final_rms = history[-1].rms_disparity if history else float("inf")

    # Check if corrections were small in the last iterations even though
    # we exhausted max_iterations (still converged, just slowly)
    last = history[-1] if history else None
    final_converged = last is not None and (
        abs(last.delta_z) < convergence_threshold * mm_per_pixel
        and abs(last.delta_tilt_x) < math.radians(0.01)
        and abs(last.delta_tilt_y) < math.radians(0.01)
    )

    # RMS improvement is reported for diagnostics but does NOT count as
    # convergence: "halved the RMS while corrections still moving" means the
    # plane fit has not stabilised, and calling that converged hides it.
    initial_rms = history[0].rms_disparity if history else float("inf")
    rms_improved = final_rms < initial_rms * 0.5  # at least 2x improvement

    if not final_converged and not rms_improved:
        logger.warning(
            f"Self-calibration did not converge (RMS={final_rms:.2f} px "
            f"after {len(history)} iterations, corrections still changing)"
        )
    elif final_converged:
        logger.info(
            f"Self-calibration converged (corrections stabilised after "
            f"{len(history)} iterations, final RMS={final_rms:.2f} px)"
        )
    else:
        logger.info(
            f"Self-calibration improved RMS {initial_rms:.1f} -> {final_rms:.1f} px "
            f"({initial_rms/final_rms:.1f}x) but corrections still changing"
        )

    return SelfCalibrationResult(
        converged=final_converged,
        n_iterations=len(history),
        z_offset=cumulative_z,
        tilt_x=cumulative_tilt_x,
        tilt_y=cumulative_tilt_y,
        final_rms_disparity=final_rms,
        history=history,
        dx_before=dx_before,
        dy_before=dy_before,
        dx_after=dx_after,
        dy_after=dy_after,
        grid_x_mm=grid_x_mm,
        grid_y_mm=grid_y_mm,
        peak_quality=peak_q,
        corr_first_iter=corr_first_iter,
        corr_last_iter=corr_last_iter,
        win_ctrs_x=win_ctrs_x,
        win_ctrs_y=win_ctrs_y,
        n_win_x=n_win_x,
        n_win_y=n_win_y,
        window_size_used=window_size,
        mm_per_pixel=mm_per_pixel,
        disp_px_per_mm=float(disp_px_per_mm),
        disp_direction=np.asarray(disp_direction, dtype=np.float64),
    )
