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
        output_size=(512, 512),
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

from pivtools_core.window_utils import compute_window_centers
from pivtools_cli.piv.piv_backend.outlier_detection import median_outlier_detection
from pivtools_cli.piv.piv_backend.infilling import infill_local_median

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PinholeCamera:
    """Pinhole camera model compatible with OpenCV conventions."""

    K: np.ndarray          # (3,3) intrinsic matrix
    dist: np.ndarray       # (5,) distortion coefficients
    R: np.ndarray          # (3,3) rotation world→camera
    t: np.ndarray          # (3,1) translation world→camera
    image_size: Tuple[int, int]  # (width, height)

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project world points (N,3) to image coordinates (N,2)."""
        rvec, _ = cv2.Rodrigues(self.R)
        pts2d, _ = cv2.projectPoints(
            points_world.astype(np.float64),
            rvec, self.t.astype(np.float64),
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


# ---------------------------------------------------------------------------
# Dewarping primitives
# ---------------------------------------------------------------------------

def compute_dewarp_maps(
    camera: PinholeCamera,
    output_size: Tuple[int, int],
    world_bounds: Tuple[float, float, float, float],
    z_offset: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build remap tables that dewarp a camera image onto a world-XY plane.

    Parameters
    ----------
    camera : PinholeCamera
    output_size : (out_h, out_w)
    world_bounds : (x_min, x_max, y_min, y_max) in mm
    z_offset, tilt_x, tilt_y : laser-sheet correction parameters

    Returns
    -------
    map_x, map_y : float32 arrays of shape (out_h, out_w)
    """
    out_h, out_w = output_size
    x_min, x_max, y_min, y_max = world_bounds

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
        image, map_x, map_y,
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
        os.path.dirname(__file__), "..", "..",
        "pivtools_cli", "lib", f"libbulkxcorr2d{lib_ext}",
    )
    lib_path = os.path.abspath(lib_path)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(
            f"Cross-correlation library not found: {lib_path}"
        )

    lib = ctypes.CDLL(lib_path)

    lib.bulkxcorr2d_accumulate.restype = ctypes.c_ubyte
    lib.bulkxcorr2d_accumulate.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageA_stack
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fImageB_stack
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fMask
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nImageSize
        ctypes.c_int,                                                      # N_images
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsX
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWinCtrsY
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindows
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightA
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fWindowWeightB
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nWindowSize
        np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),    # nFitWindowSize
        ctypes.c_int,                                                      # bNormalize
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),  # fCorrelPlane_Sum (output)
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

    # Hanning window weight
    hann = np.outer(np.hanning(ws), np.hanning(ws)).astype(np.float32)
    weight = np.ascontiguousarray(hann.ravel())

    win_size_arr = np.array([ws, ws], dtype=np.int32)

    # Output buffer
    corr_sum = np.zeros(total_windows * ws * ws, dtype=np.float32)

    error_code = lib.bulkxcorr2d_accumulate(
        stack_a, stack_b,
        mask, image_size, N,
        np.ascontiguousarray(win_ctrs_x, dtype=np.float32),
        np.ascontiguousarray(win_ctrs_y, dtype=np.float32),
        n_windows,
        weight, weight,
        win_size_arr,
        win_size_arr,  # nFitWindowSize == nWindowSize (no central extraction)
        0,             # bNormalize = 0 (raw)
        corr_sum,
    )

    if error_code != 0:
        raise RuntimeError(
            f"bulkxcorr2d_accumulate returned error code {error_code}"
        )

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
    search_region = corr_plane[margin:ws - margin, margin:ws - margin]
    coarse_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    iy = coarse_idx[0] + margin
    ix = coarse_idx[1] + margin

    # Reject if peak is on the absolute border
    if iy < 2 or iy >= ws - 2 or ix < 2 or ix >= ws - 2:
        return np.nan, np.nan, np.nan

    # Extract 5x5 region
    r = 2
    region = corr_plane[iy - r:iy + r + 1, ix - r:ix + r + 1].copy()
    region_min = region.min()
    region = region - region_min  # shift to zero baseline

    # Build coordinate grids for the 5x5 region
    ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
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
            _gaussian_6dof_residuals, p0,
            args=(ii_flat, jj_flat, data_flat),
            method='lm', max_nfev=20,
        )
        A_fit, i0_fit, j0_fit = result.x[0], result.x[1], result.x[2]
    except Exception:
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
    dx_filled, dy_filled = infill_local_median(
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
    stereo_angle_rad: float,
    mm_per_pixel: float,
) -> dict:
    """Fit a plane to the dx disparity field → physical corrections.

    The dx disparity (horizontal) encodes the stereo mismatch caused by
    Z-offset and tilts.  dy is expected to be small and is not used for
    the plane fit.

    Parameters
    ----------
    dx : (n_win_y, n_win_x) disparity in pixels
    dy : (n_win_y, n_win_x) disparity in pixels (informational)
    grid_x_mm, grid_y_mm : world-coordinate grids in mm
    stereo_angle_rad : half-angle between cameras
    mm_per_pixel : spatial scale

    Returns
    -------
    dict with z_offset, tilt_x, tilt_y, rms_residual
    """
    valid = np.isfinite(dx)
    X = grid_x_mm[valid].ravel()
    Y = grid_y_mm[valid].ravel()
    D = dx[valid].ravel()

    # Least-squares: dx = a + b*X + c*Y
    A_mat = np.column_stack([np.ones_like(X), X, Y])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, D, rcond=None)
    a, b, c = coeffs

    residual = D - A_mat @ coeffs
    rms = float(np.sqrt(np.mean(residual ** 2)))

    # Convert pixel disparity to physical corrections.
    # Positive dx disparity means cam2 is shifted right relative to cam1,
    # which results from a positive Z offset.  The correction applied to
    # the dewarp maps must therefore be in the SAME direction as the fitted
    # offset:  dZ = +a * conversion.
    # disparity (px) = 2 * tan(theta) * dZ(mm) / mm_per_pixel
    # => dZ = disparity * mm_per_pixel / (2 * tan(theta))
    conversion = mm_per_pixel / (2.0 * np.tan(stereo_angle_rad))
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
    output_size: Tuple[int, int],
    world_bounds: Tuple[float, float, float, float],
    window_size: int = 64,
    overlap: float = 50.0,
    max_iterations: int = 10,
    convergence_threshold: float = 0.1,
    quality_threshold: float = 0.3,
    use_c_library: bool = True,
) -> SelfCalibrationResult:
    """Run iterative stereo PIV self-calibration.

    Parameters
    ----------
    cam1, cam2 : PinholeCamera instances
    images_cam1, images_cam2 : lists of uint8 images (same length)
    output_size : (out_h, out_w) dewarped image size
    world_bounds : (x_min, x_max, y_min, y_max) in mm
    window_size : correlation window size in pixels
    overlap : window overlap percentage
    max_iterations : maximum correction iterations
    convergence_threshold : RMS disparity (px) below which to stop
    quality_threshold : minimum peak quality to accept a disparity vector
    use_c_library : if True, use C library for correlation; else pure Python

    Returns
    -------
    SelfCalibrationResult
    """
    n_images = len(images_cam1)
    out_h, out_w = output_size
    x_min, x_max, y_min, y_max = world_bounds

    mm_per_pixel_x = (x_max - x_min) / out_w
    mm_per_pixel_y = (y_max - y_min) / out_h
    mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y) / 2.0

    # Stereo half-angle from relative rotation
    R_rel = cam2.R @ cam1.R.T
    trace_val = np.trace(R_rel)
    full_angle = math.acos(max(-1.0, min(1.0, (trace_val - 1.0) / 2.0)))
    stereo_half_angle = full_angle / 2.0
    logger.info(
        f"Stereo full angle: {math.degrees(full_angle):.1f} deg, "
        f"half-angle: {math.degrees(stereo_half_angle):.1f} deg"
    )

    # Window grid
    wc = compute_window_centers(
        (out_h, out_w), (window_size, window_size), overlap,
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

    # Load C library if requested
    lib = None
    if use_c_library:
        try:
            lib = _load_xcorr_library()
        except FileNotFoundError:
            logger.warning("C library not found, falling back to Python correlation")
            use_c_library = False

    cumulative_z = 0.0
    cumulative_tilt_x = 0.0
    cumulative_tilt_y = 0.0
    history = []
    dx_before = None
    dy_before = None
    dx_after = None
    dy_after = None
    peak_q = None

    for iteration in range(max_iterations):
        logger.info(f"Self-calibration iteration {iteration + 1}/{max_iterations}")

        # Build dewarp maps with cumulative corrections
        maps_cam1 = compute_dewarp_maps(
            cam1, output_size, world_bounds,
            cumulative_z, cumulative_tilt_x, cumulative_tilt_y,
        )
        maps_cam2 = compute_dewarp_maps(
            cam2, output_size, world_bounds,
            cumulative_z, cumulative_tilt_x, cumulative_tilt_y,
        )

        # Dewarp all images
        dw1 = np.stack([
            dewarp_image(img, maps_cam1[0], maps_cam1[1]).astype(np.float32)
            for img in images_cam1
        ])
        dw2 = np.stack([
            dewarp_image(img, maps_cam2[0], maps_cam2[1]).astype(np.float32)
            for img in images_cam2
        ])

        # Cross-correlate cam1 vs cam2
        if use_c_library and lib is not None:
            corr_sum = accumulate_ensemble_correlation(
                lib, dw1, dw2,
                win_ctrs_x, win_ctrs_y,
                n_win_x, n_win_y, window_size,
            )
        else:
            corr_sum = _python_ensemble_correlation(
                dw1, dw2, win_ctrs_x, win_ctrs_y,
                n_win_x, n_win_y, window_size,
            )

        # Extract disparity field
        dx_raw, dy_raw, peak_q = extract_disparity_field(corr_sum, n_images)

        # Store "before" on first iteration
        if iteration == 0:
            dx_before = dx_raw.copy()
            dy_before = dy_raw.copy()

        # Clean
        dx_clean, dy_clean = clean_disparity_field(
            dx_raw, dy_raw, peak_q, quality_threshold,
        )

        # RMS disparity
        valid = np.isfinite(dx_clean) & np.isfinite(dy_clean)
        if not valid.any():
            logger.warning("No valid disparity points — aborting")
            break

        rms = float(np.sqrt(
            np.mean(dx_clean[valid] ** 2 + dy_clean[valid] ** 2)
        ))
        logger.info(f"  RMS disparity: {rms:.4f} px")

        # Fit plane to get corrections
        fit = fit_disparity_plane(
            dx_clean, dy_clean,
            grid_x_mm, grid_y_mm,
            stereo_half_angle, mm_per_pixel,
        )
        delta_z = fit["z_offset"]
        delta_tx = fit["tilt_x"]
        delta_ty = fit["tilt_y"]

        cumulative_z += delta_z
        cumulative_tilt_x += delta_tx
        cumulative_tilt_y += delta_ty

        history.append(IterationRecord(
            iteration=iteration + 1,
            rms_disparity=rms,
            delta_z=delta_z,
            delta_tilt_x=delta_tx,
            delta_tilt_y=delta_ty,
            cumulative_z=cumulative_z,
            cumulative_tilt_x=cumulative_tilt_x,
            cumulative_tilt_y=cumulative_tilt_y,
        ))

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

        if rms < convergence_threshold:
            logger.info(f"Converged at iteration {iteration + 1} (RMS={rms:.4f} px)")
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
            )

    # Did not converge — do one final pass to get "after" disparity
    logger.info("Generating final disparity field...")
    maps_cam1 = compute_dewarp_maps(
        cam1, output_size, world_bounds,
        cumulative_z, cumulative_tilt_x, cumulative_tilt_y,
    )
    maps_cam2 = compute_dewarp_maps(
        cam2, output_size, world_bounds,
        cumulative_z, cumulative_tilt_x, cumulative_tilt_y,
    )
    dw1 = np.stack([
        dewarp_image(img, maps_cam1[0], maps_cam1[1]).astype(np.float32)
        for img in images_cam1
    ])
    dw2 = np.stack([
        dewarp_image(img, maps_cam2[0], maps_cam2[1]).astype(np.float32)
        for img in images_cam2
    ])
    if use_c_library and lib is not None:
        corr_sum = accumulate_ensemble_correlation(
            lib, dw1, dw2,
            win_ctrs_x, win_ctrs_y,
            n_win_x, n_win_y, window_size,
        )
    else:
        corr_sum = _python_ensemble_correlation(
            dw1, dw2, win_ctrs_x, win_ctrs_y,
            n_win_x, n_win_y, window_size,
        )
    dx_after, dy_after, _ = extract_disparity_field(corr_sum, n_images)

    final_rms = history[-1].rms_disparity if history else float("inf")
    if final_rms > 1.0 and len(history) >= 5:
        logger.warning(
            f"Self-calibration did not converge well (RMS={final_rms:.2f} px "
            f"after {len(history)} iterations)"
        )

    return SelfCalibrationResult(
        converged=final_rms < convergence_threshold,
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
    )


# ---------------------------------------------------------------------------
# Pure-Python fallback correlation (no C library needed)
# ---------------------------------------------------------------------------

def _python_ensemble_correlation(
    images_a: np.ndarray,
    images_b: np.ndarray,
    win_ctrs_x: np.ndarray,
    win_ctrs_y: np.ndarray,
    n_win_x: int,
    n_win_y: int,
    window_size: int,
) -> np.ndarray:
    """Pure-Python ensemble cross-correlation (FFT-based, NumPy only).

    Slower than C but works without compiled libraries.
    """
    N, H, W = images_a.shape
    ws = window_size
    half = ws // 2
    hann = np.outer(np.hanning(ws), np.hanning(ws)).astype(np.float64)

    corr_sum = np.zeros((n_win_y, n_win_x, ws, ws), dtype=np.float64)

    for n in range(N):
        img_a = images_a[n].astype(np.float64)
        img_b = images_b[n].astype(np.float64)

        for iy in range(n_win_y):
            cy = int(round(win_ctrs_y[iy]))
            y0 = cy - half
            y1 = y0 + ws
            if y0 < 0 or y1 > H:
                continue

            for ix in range(n_win_x):
                cx = int(round(win_ctrs_x[ix]))
                x0 = cx - half
                x1 = x0 + ws
                if x0 < 0 or x1 > W:
                    continue

                win_a = img_a[y0:y1, x0:x1] * hann
                win_b = img_b[y0:y1, x0:x1] * hann

                # Subtract mean
                win_a = win_a - win_a.mean()
                win_b = win_b - win_b.mean()

                # FFT cross-correlation
                fa = np.fft.rfft2(win_a, s=(ws, ws))
                fb = np.fft.rfft2(win_b, s=(ws, ws))
                cc = np.fft.irfft2(fa * np.conj(fb), s=(ws, ws))
                cc = np.fft.fftshift(cc)

                corr_sum[iy, ix] += cc

    return corr_sum.astype(np.float32)
