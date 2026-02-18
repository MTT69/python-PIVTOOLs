"""
Test harness: predictor field interpolation methods.

Replicates the _get_im_mesh() pipeline from cpu_ensemble.py using pass 1
output, generates the dense predictor field for pass 2 using multiple methods,
and compares vertical/horizontal profiles and 2D maps.

Does NOT modify any pipeline code — purely a diagnostic/comparison script.

Methods tested:
  A. CURRENT:   edge pad → gaussian smooth → cv2.remap CUBIC, BORDER_CONSTANT=0
  B. REPLICATE: edge pad → gaussian smooth → cv2.remap CUBIC, BORDER_REPLICATE
  C. LINEAR_EXTRAP: linear extrapolation pad → gaussian smooth → cv2.remap CUBIC, BORDER_REPLICATE
  D. PCHIP:     no pad → scipy PchipInterpolator (separable 2D) → no cv2
  E. PCHIP_EXTRAP: linear extrap pad → PCHIP interpolation
  F. NO_SMOOTH:  edge pad → NO gaussian smooth → cv2.remap CUBIC, BORDER_REPLICATE
  G. AKIMA:     no pad → scipy Akima1DInterpolator (separable 2D) → smoother than PCHIP
  H. BICUBIC_SPLINE: scipy RectBivariateSpline (true 2D bicubic) → standard in DaVis/PIVlab
  I. BILINEAR:  edge pad → cv2.remap LINEAR, BORDER_REPLICATE → simplest baseline
"""

import time
import numpy as np
import scipy.io
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from scipy.interpolate import (
    PchipInterpolator, RegularGridInterpolator, RectBivariateSpline,
    Akima1DInterpolator,
)
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
MAT_PATH = Path(
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
    r"\validation_4000_kspace_predictorupdate\uncalibrated_piv\4000\Cam1\ensemble"
    r"\ensemble_result.mat"
)
OUT_DIR = MAT_PATH.parent

# Pass configuration (from config.yaml)
PASS1_WS = (96, 96)     # window_size (y, x)
PASS1_OVERLAP = 0.75
PASS2_WS = (16, 16)
PASS2_OVERLAP = 0.50


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_pass_data(path):
    """Load ensemble_result.mat, return per-pass data."""
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = mat["ensemble_result"]
    passes = list(data.flat) if isinstance(data, np.ndarray) and data.dtype == object else [data]
    results = []
    for p in passes:
        d = {}
        for field in ["ux", "uy", "pred_x", "pred_y", "window_size", "win_ctrs_x", "win_ctrs_y"]:
            val = getattr(p, field, None)
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    d[field] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
                    continue
            d[field] = None
        results.append(d)
    return results


def compute_window_centers_std(H, win_size, overlap):
    """Standard mode window centers (matching base.py)."""
    spacing = int(win_size * (1 - overlap))
    first = win_size // 2
    centers = np.arange(first, H - win_size // 2 + 1, spacing, dtype=np.float32)
    return centers, spacing


def compute_window_centers_single(H, win_size, overlap):
    """Single mode window centers — pads image to fill integer windows."""
    spacing = int(win_size * (1 - overlap))
    n_windows = int(np.ceil((H - win_size) / spacing)) + 1
    total_extent = win_size + (n_windows - 1) * spacing
    pad_total = total_extent - H
    pad_top = pad_total // 2
    first = win_size // 2 - pad_top
    centers = np.arange(first, first + n_windows * spacing, spacing, dtype=np.float32)
    return centers, spacing, (pad_top, pad_total - pad_top)


def compute_padded_centers(centers, spacing, H):
    """Compute pre/post padding centers (replicating base.py:860-918)."""
    # Pre-padding: extend before first center
    pre = np.arange(1, centers[0] - spacing / 2, spacing)
    if pre.size == 0:
        pre = np.array([1.0])
    pre = pre - 1
    while len(pre) < 2:
        extra = pre[0] - spacing
        pre = np.concatenate([[max(0, extra)], pre])

    # Post-padding: extend after last center
    post = np.arange(H, centers[-1] + spacing / 2, -spacing)
    if post.size == 0:
        post = np.array([float(H)])
    post = post - 1
    while len(post) < 2:
        extra = post[-1] + spacing
        post = np.concatenate([post, [min(H - 1, extra)]])

    centers_all = np.concatenate([pre, centers, post[::-1]]).astype(np.float32)
    n_pre = len(pre)
    n_post = len(post)
    return centers_all, n_pre, n_post


def compute_smoothing_params(prev_n_windows, prev_spacing):
    """Compute ksize_filt and sd (replicating base.py:928-946)."""
    k_filt = np.round(np.array(prev_n_windows) / np.array(prev_spacing)).astype(int) + 1
    k_filt = tuple(int(k) + (int(k) % 2 == 0) for k in k_filt)  # ensure odd
    sd = np.sqrt(np.prod(k_filt)) / 3 * 0.65
    return k_filt, sd


def linear_extrapolate_pad(field, n_pre_y, n_post_y, n_pre_x, n_post_x):
    """
    Pad a 2D field using linear extrapolation from the last 2 points.
    Much better than mode="edge" for fields with strong boundary gradients.
    """
    ny, nx = field.shape

    # Pad Y (top and bottom)
    result = field.copy()

    # Top: extrapolate using rows 0 and 1
    if n_pre_y > 0:
        grad_top = field[0, :] - field[1, :]  # gradient per row step (going upward)
        top_rows = np.array([field[0, :] + grad_top * (k + 1) for k in range(n_pre_y)])
        result = np.vstack([top_rows[::-1], result])

    # Bottom: extrapolate using rows -2 and -1
    if n_post_y > 0:
        grad_bot = field[-1, :] - field[-2, :]  # gradient per row step (going downward)
        bot_rows = np.array([field[-1, :] + grad_bot * (k + 1) for k in range(n_post_y)])
        result = np.vstack([result, bot_rows])

    # Pad X (left and right) on the already y-padded result
    ny2 = result.shape[0]

    if n_pre_x > 0:
        grad_left = result[:, 0] - result[:, 1]
        left_cols = np.array([result[:, 0] + grad_left * (k + 1) for k in range(n_pre_x)]).T
        result = np.hstack([left_cols[:, ::-1], result])

    if n_post_x > 0:
        grad_right = result[:, -1] - result[:, -2]
        right_cols = np.array([result[:, -1] + grad_right * (k + 1) for k in range(n_post_x)]).T
        result = np.hstack([result, right_cols])

    return result


def pchip_interp_2d(src_y, src_x, field_2d, dst_y, dst_x):
    """
    Separable 2D PCHIP interpolation (monotone-preserving, no overshoot).

    Step 1: PCHIP along Y for each source column → intermediate grid (dst_ny, src_nx)
    Step 2: PCHIP along X for each intermediate row → final grid (dst_ny, dst_nx)

    Extrapolation: uses extrapolate=True so boundary values extend naturally.
    """
    src_ny, src_nx = field_2d.shape
    dst_ny = len(dst_y)
    dst_nx = len(dst_x)

    # Step 1: interpolate along Y (columns)
    intermediate = np.empty((dst_ny, src_nx), dtype=np.float32)
    for j in range(src_nx):
        pchip = PchipInterpolator(src_y, field_2d[:, j], extrapolate=True)
        intermediate[:, j] = pchip(dst_y)

    # Step 2: interpolate along X (rows)
    result = np.empty((dst_ny, dst_nx), dtype=np.float32)
    for i in range(dst_ny):
        pchip = PchipInterpolator(src_x, intermediate[i, :], extrapolate=True)
        result[i, :] = pchip(dst_x)

    return result


def akima_interp_2d(src_y, src_x, field_2d, dst_y, dst_x):
    """
    Separable 2D Akima interpolation.
    Akima uses a weighted average of neighbouring slopes — smoother than PCHIP
    and allows small overshoots, but much less than cubic spline.
    """
    src_ny, src_nx = field_2d.shape
    dst_ny = len(dst_y)
    dst_nx = len(dst_x)

    # Step 1: interpolate along Y (columns)
    intermediate = np.empty((dst_ny, src_nx), dtype=np.float32)
    for j in range(src_nx):
        akima = Akima1DInterpolator(src_y, field_2d[:, j])
        vals = akima(dst_y, extrapolate=True)
        intermediate[:, j] = vals

    # Step 2: interpolate along X (rows)
    result = np.empty((dst_ny, dst_nx), dtype=np.float32)
    for i in range(dst_ny):
        akima = Akima1DInterpolator(src_x, intermediate[i, :])
        vals = akima(dst_x, extrapolate=True)
        result[i, :] = vals

    return result



# ─── Method implementations ──────────────────────────────────────────────────

def method_current(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                   src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                   dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """A. CURRENT pipeline: edge pad → gaussian → cv2.remap CUBIC, BORDER_CONSTANT=0"""
    # Stack as [uy, ux] matching pipeline convention
    pred = np.stack([predictor_uy, predictor_ux], axis=-1)

    # Pad with edge
    pred_padded = np.pad(pred, ((n_pre_y, n_post_y), (n_pre_x, n_post_x), (0, 0)), mode="edge")

    # Gaussian smooth
    smoothed = np.zeros_like(pred_padded)
    for d in range(2):
        smoothed[..., d] = gaussian_filter(pred_padded[..., d], sigma=sd,
                                           truncate=(ksize_filt[0] - 1) / (2 * sd) if sd > 0 else 0,
                                           mode="nearest")

    # Build dense maps (src padded grid → every pixel)
    map_x_1d = np.interp(np.arange(W, dtype=np.float32), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    map_y_1d = np.interp(np.arange(H, dtype=np.float32), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_y_2d, map_x_2d = np.meshgrid(map_y_1d.astype(np.float32), map_x_1d.astype(np.float32), indexing="ij")

    dense = np.zeros((H, W, 2), dtype=np.float32)
    for d in range(2):
        dense[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                  map_x_2d, map_y_2d,
                                  cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Also compute at pass 2 window centers
    win_y, win_x = np.meshgrid(dst_ctrs_y, dst_ctrs_x, indexing="ij")
    ix = np.interp(win_x.ravel(), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    iy = np.interp(win_y.ravel(), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_x_p = ix.reshape(win_x.shape).astype(np.float32)
    map_y_p = iy.reshape(win_y.shape).astype(np.float32)

    pred_at_win = np.zeros((len(dst_ctrs_y), len(dst_ctrs_x), 2), dtype=np.float32)
    for d in range(2):
        pred_at_win[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                        map_x_p, map_y_p,
                                        cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return dense, pred_at_win, "A. Current (edge+gauss+BORDER_CONSTANT=0)"


def method_border_replicate(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                            src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                            dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """B. Same as current but BORDER_REPLICATE instead of BORDER_CONSTANT=0"""
    pred = np.stack([predictor_uy, predictor_ux], axis=-1)
    pred_padded = np.pad(pred, ((n_pre_y, n_post_y), (n_pre_x, n_post_x), (0, 0)), mode="edge")

    smoothed = np.zeros_like(pred_padded)
    for d in range(2):
        smoothed[..., d] = gaussian_filter(pred_padded[..., d], sigma=sd,
                                           truncate=(ksize_filt[0] - 1) / (2 * sd) if sd > 0 else 0,
                                           mode="nearest")

    map_x_1d = np.interp(np.arange(W, dtype=np.float32), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    map_y_1d = np.interp(np.arange(H, dtype=np.float32), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_y_2d, map_x_2d = np.meshgrid(map_y_1d.astype(np.float32), map_x_1d.astype(np.float32), indexing="ij")

    dense = np.zeros((H, W, 2), dtype=np.float32)
    for d in range(2):
        dense[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                  map_x_2d, map_y_2d,
                                  cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    win_y, win_x = np.meshgrid(dst_ctrs_y, dst_ctrs_x, indexing="ij")
    ix = np.interp(win_x.ravel(), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    iy = np.interp(win_y.ravel(), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_x_p = ix.reshape(win_x.shape).astype(np.float32)
    map_y_p = iy.reshape(win_y.shape).astype(np.float32)

    pred_at_win = np.zeros((len(dst_ctrs_y), len(dst_ctrs_x), 2), dtype=np.float32)
    for d in range(2):
        pred_at_win[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                        map_x_p, map_y_p,
                                        cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

    return dense, pred_at_win, "B. BORDER_REPLICATE"


def method_linear_extrap(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                         src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                         dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """C. Linear extrapolation padding → gaussian → cv2.remap CUBIC, BORDER_REPLICATE"""
    # Pad each component with linear extrapolation
    uy_padded = linear_extrapolate_pad(predictor_uy, n_pre_y, n_post_y, n_pre_x, n_post_x)
    ux_padded = linear_extrapolate_pad(predictor_ux, n_pre_y, n_post_y, n_pre_x, n_post_x)
    pred_padded = np.stack([uy_padded, ux_padded], axis=-1)

    smoothed = np.zeros_like(pred_padded)
    for d in range(2):
        smoothed[..., d] = gaussian_filter(pred_padded[..., d], sigma=sd,
                                           truncate=(ksize_filt[0] - 1) / (2 * sd) if sd > 0 else 0,
                                           mode="nearest")

    map_x_1d = np.interp(np.arange(W, dtype=np.float32), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    map_y_1d = np.interp(np.arange(H, dtype=np.float32), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_y_2d, map_x_2d = np.meshgrid(map_y_1d.astype(np.float32), map_x_1d.astype(np.float32), indexing="ij")

    dense = np.zeros((H, W, 2), dtype=np.float32)
    for d in range(2):
        dense[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                  map_x_2d, map_y_2d,
                                  cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    win_y, win_x = np.meshgrid(dst_ctrs_y, dst_ctrs_x, indexing="ij")
    ix = np.interp(win_x.ravel(), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    iy = np.interp(win_y.ravel(), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_x_p = ix.reshape(win_x.shape).astype(np.float32)
    map_y_p = iy.reshape(win_y.shape).astype(np.float32)

    pred_at_win = np.zeros((len(dst_ctrs_y), len(dst_ctrs_x), 2), dtype=np.float32)
    for d in range(2):
        pred_at_win[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                        map_x_p, map_y_p,
                                        cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

    return dense, pred_at_win, "C. Linear extrap pad + REPLICATE"


def method_pchip(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                 src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                 dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """D. PCHIP interpolation — no padding, no cv2.remap, scipy only.
    PCHIP naturally extrapolates and preserves monotonicity."""
    pixel_y = np.arange(H, dtype=np.float32)
    pixel_x = np.arange(W, dtype=np.float32)

    # Interpolate ux and uy from source grid to every pixel
    dense_uy = pchip_interp_2d(src_ctrs_y, src_ctrs_x, predictor_uy, pixel_y, pixel_x)
    dense_ux = pchip_interp_2d(src_ctrs_y, src_ctrs_x, predictor_ux, pixel_y, pixel_x)
    dense = np.stack([dense_uy, dense_ux], axis=-1)

    # Interpolate to pass 2 window centers
    win_uy = pchip_interp_2d(src_ctrs_y, src_ctrs_x, predictor_uy, dst_ctrs_y, dst_ctrs_x)
    win_ux = pchip_interp_2d(src_ctrs_y, src_ctrs_x, predictor_ux, dst_ctrs_y, dst_ctrs_x)
    pred_at_win = np.stack([win_uy, win_ux], axis=-1)

    return dense, pred_at_win, "D. PCHIP (no pad, no smooth)"


def method_pchip_with_extrap_pad(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                                 src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                                 dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """E. Linear extrap pad → PCHIP interpolation (uses padded grid as source)."""
    uy_padded = linear_extrapolate_pad(predictor_uy, n_pre_y, n_post_y, n_pre_x, n_post_x)
    ux_padded = linear_extrapolate_pad(predictor_ux, n_pre_y, n_post_y, n_pre_x, n_post_x)

    pixel_y = np.arange(H, dtype=np.float32)
    pixel_x = np.arange(W, dtype=np.float32)

    dense_uy = pchip_interp_2d(src_ctrs_y_all, src_ctrs_x_all, uy_padded, pixel_y, pixel_x)
    dense_ux = pchip_interp_2d(src_ctrs_y_all, src_ctrs_x_all, ux_padded, pixel_y, pixel_x)
    dense = np.stack([dense_uy, dense_ux], axis=-1)

    win_uy = pchip_interp_2d(src_ctrs_y_all, src_ctrs_x_all, uy_padded, dst_ctrs_y, dst_ctrs_x)
    win_ux = pchip_interp_2d(src_ctrs_y_all, src_ctrs_x_all, ux_padded, dst_ctrs_y, dst_ctrs_x)
    pred_at_win = np.stack([win_uy, win_ux], axis=-1)

    return dense, pred_at_win, "E. Linear extrap + PCHIP"


def method_no_smooth(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                     src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                     dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """F. Edge pad, NO gaussian smooth, cv2.remap CUBIC, BORDER_REPLICATE.
    Tests whether smoothing is helping or hurting."""
    pred = np.stack([predictor_uy, predictor_ux], axis=-1)
    pred_padded = np.pad(pred, ((n_pre_y, n_post_y), (n_pre_x, n_post_x), (0, 0)), mode="edge")

    # Skip smoothing — use padded field directly
    smoothed = pred_padded

    map_x_1d = np.interp(np.arange(W, dtype=np.float32), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    map_y_1d = np.interp(np.arange(H, dtype=np.float32), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_y_2d, map_x_2d = np.meshgrid(map_y_1d.astype(np.float32), map_x_1d.astype(np.float32), indexing="ij")

    dense = np.zeros((H, W, 2), dtype=np.float32)
    for d in range(2):
        dense[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                  map_x_2d, map_y_2d,
                                  cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    win_y, win_x = np.meshgrid(dst_ctrs_y, dst_ctrs_x, indexing="ij")
    ix = np.interp(win_x.ravel(), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    iy = np.interp(win_y.ravel(), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_x_p = ix.reshape(win_x.shape).astype(np.float32)
    map_y_p = iy.reshape(win_y.shape).astype(np.float32)

    pred_at_win = np.zeros((len(dst_ctrs_y), len(dst_ctrs_x), 2), dtype=np.float32)
    for d in range(2):
        pred_at_win[..., d] = cv2.remap(smoothed[..., d].astype(np.float32),
                                        map_x_p, map_y_p,
                                        cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

    return dense, pred_at_win, "F. No smooth + REPLICATE"


def method_akima(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                 src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                 dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """G. Akima interpolation — separable 2D, no padding, no smooth.
    Akima uses weighted slope averaging: smoother than PCHIP, allows small
    overshoots but much less than cubic spline. Common in geophysics/PIV."""
    pixel_y = np.arange(H, dtype=np.float32)
    pixel_x = np.arange(W, dtype=np.float32)

    dense_uy = akima_interp_2d(src_ctrs_y, src_ctrs_x, predictor_uy, pixel_y, pixel_x)
    dense_ux = akima_interp_2d(src_ctrs_y, src_ctrs_x, predictor_ux, pixel_y, pixel_x)
    dense = np.stack([dense_uy, dense_ux], axis=-1)

    win_uy = akima_interp_2d(src_ctrs_y, src_ctrs_x, predictor_uy, dst_ctrs_y, dst_ctrs_x)
    win_ux = akima_interp_2d(src_ctrs_y, src_ctrs_x, predictor_ux, dst_ctrs_y, dst_ctrs_x)
    pred_at_win = np.stack([win_uy, win_ux], axis=-1)

    return dense, pred_at_win, "G. Akima (no pad, no smooth)"


def method_bicubic_spline(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                           src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                           dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """H. RectBivariateSpline (bicubic) — standard in DaVis/PIVlab.
    True 2D spline (not separable). Extrapolation outside source grid
    follows the spline polynomial (can overshoot significantly)."""
    pixel_y = np.arange(H, dtype=np.float64)
    pixel_x = np.arange(W, dtype=np.float64)

    # RectBivariateSpline needs float64
    spl_ux = RectBivariateSpline(src_ctrs_y.astype(np.float64),
                                  src_ctrs_x.astype(np.float64),
                                  predictor_ux.astype(np.float64), kx=3, ky=3)
    spl_uy = RectBivariateSpline(src_ctrs_y.astype(np.float64),
                                  src_ctrs_x.astype(np.float64),
                                  predictor_uy.astype(np.float64), kx=3, ky=3)

    dense_ux = spl_ux(pixel_y, pixel_x).astype(np.float32)
    dense_uy = spl_uy(pixel_y, pixel_x).astype(np.float32)
    dense = np.stack([dense_uy, dense_ux], axis=-1)

    win_ux = spl_ux(dst_ctrs_y.astype(np.float64),
                     dst_ctrs_x.astype(np.float64)).astype(np.float32)
    win_uy = spl_uy(dst_ctrs_y.astype(np.float64),
                     dst_ctrs_x.astype(np.float64)).astype(np.float32)
    pred_at_win = np.stack([win_uy, win_ux], axis=-1)

    return dense, pred_at_win, "H. Bicubic spline (RectBivariateSpline)"


def method_bilinear(predictor_ux, predictor_uy, src_ctrs_y, src_ctrs_x,
                     src_ctrs_y_all, src_ctrs_x_all, n_pre_y, n_post_y, n_pre_x, n_post_x,
                     dst_ctrs_y, dst_ctrs_x, H, W, ksize_filt, sd):
    """I. Bilinear interpolation (edge pad + cv2.remap LINEAR + REPLICATE).
    Simplest standard approach — many PIV codes use this as baseline."""
    pred = np.stack([predictor_uy, predictor_ux], axis=-1)
    pred_padded = np.pad(pred, ((n_pre_y, n_post_y), (n_pre_x, n_post_x), (0, 0)), mode="edge")

    map_x_1d = np.interp(np.arange(W, dtype=np.float32), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    map_y_1d = np.interp(np.arange(H, dtype=np.float32), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_y_2d, map_x_2d = np.meshgrid(map_y_1d.astype(np.float32), map_x_1d.astype(np.float32), indexing="ij")

    dense = np.zeros((H, W, 2), dtype=np.float32)
    for d in range(2):
        dense[..., d] = cv2.remap(pred_padded[..., d].astype(np.float32),
                                  map_x_2d, map_y_2d,
                                  cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

    win_y, win_x = np.meshgrid(dst_ctrs_y, dst_ctrs_x, indexing="ij")
    ix = np.interp(win_x.ravel(), src_ctrs_x_all, np.arange(len(src_ctrs_x_all)))
    iy = np.interp(win_y.ravel(), src_ctrs_y_all, np.arange(len(src_ctrs_y_all)))
    map_x_p = ix.reshape(win_x.shape).astype(np.float32)
    map_y_p = iy.reshape(win_y.shape).astype(np.float32)

    pred_at_win = np.zeros((len(dst_ctrs_y), len(dst_ctrs_x), 2), dtype=np.float32)
    for d in range(2):
        pred_at_win[..., d] = cv2.remap(pred_padded[..., d].astype(np.float32),
                                        map_x_p, map_y_p,
                                        cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)

    return dense, pred_at_win, "I. Bilinear + REPLICATE"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PREDICTOR FIELD METHOD COMPARISON")
    print("=" * 80)

    # ── Load data ──
    passes = load_pass_data(MAT_PATH)
    p1 = passes[0]
    has_p2 = len(passes) > 1

    ux1 = p1["ux"]    # (82, 82) — pass 1 output (saved as physical, uy negated)
    uy1 = p1["uy"]    # (82, 82) — uy is NEGATED in file (physical convention)
    ux2 = passes[1]["ux"] if has_p2 else None  # pass 2 ground truth (if available)

    # Un-negate uy to get back to image coords (internal pipeline convention)
    # In file: uy_saved = -uy_image. So uy_image = -uy_saved
    uy1_img = -uy1

    print(f"Pass 1 ux: shape={ux1.shape}, range=[{ux1.min():.2f}, {ux1.max():.2f}]")
    print(f"Pass 1 uy: shape={uy1.shape}, range=[{uy1.min():.4f}, {uy1.max():.4f}] (file/physical)")
    if has_p2:
        print(f"Pass 2 ux: shape={ux2.shape}, range=[{ux2.min():.2f}, {ux2.max():.2f}] (ground truth)")
    else:
        print("Pass 2: NOT AVAILABLE (single-pass dataset) — will show upscaled fields only")

    # ── Compute grids ──
    # Determine image size from pass 1 grid
    p1_spacing_y = int(PASS1_WS[0] * (1 - PASS1_OVERLAP))  # 24
    p1_spacing_x = int(PASS1_WS[1] * (1 - PASS1_OVERLAP))  # 24
    ny1, nx1 = ux1.shape

    # Reconstruct window centers for pass 1
    first_y = PASS1_WS[0] // 2  # 48
    first_x = PASS1_WS[1] // 2  # 48
    ctrs_y1 = np.arange(first_y, first_y + ny1 * p1_spacing_y, p1_spacing_y, dtype=np.float32)
    ctrs_x1 = np.arange(first_x, first_x + nx1 * p1_spacing_x, p1_spacing_x, dtype=np.float32)

    # Image dimensions
    H = int(ctrs_y1[-1] + first_y)  # last center + half window
    W = int(ctrs_x1[-1] + first_x)

    print(f"\nImage size: H={H}, W={W}")
    print(f"Pass 1 centers: y=[{ctrs_y1[0]:.0f}..{ctrs_y1[-1]:.0f}] ({ny1}), "
          f"x=[{ctrs_x1[0]:.0f}..{ctrs_x1[-1]:.0f}] ({nx1}), spacing={p1_spacing_y}")

    # Padded centers for pass 1 (used as interpolation source for pass 2)
    ctrs_y1_all, n_pre_y, n_post_y = compute_padded_centers(ctrs_y1, p1_spacing_y, H)
    ctrs_x1_all, n_pre_x, n_post_x = compute_padded_centers(ctrs_x1, p1_spacing_x, W)

    print(f"Padded pass 1: y_all={len(ctrs_y1_all)} (pre={n_pre_y}, post={n_post_y}), "
          f"x_all={len(ctrs_x1_all)} (pre={n_pre_x}, post={n_post_x})")

    # Window centers for pass 2
    p2_spacing_y = int(PASS2_WS[0] * (1 - PASS2_OVERLAP))  # 8
    p2_spacing_x = int(PASS2_WS[1] * (1 - PASS2_OVERLAP))  # 8

    if has_p2:
        # Use actual pass 2 shape to anchor computed grid
        ny2 = ux2.shape[0]
        nx2 = ux2.shape[1]
    else:
        # No pass 2 — compute expected grid size from config
        ny2 = None
        nx2 = None

    ctrs_y2, _, _ = compute_window_centers_single(H, PASS2_WS[0], PASS2_OVERLAP)
    ctrs_x2, _, _ = compute_window_centers_single(W, PASS2_WS[1], PASS2_OVERLAP)

    if ny2 is not None:
        # If our computed count doesn't match actual, pad/trim
        if len(ctrs_y2) < ny2:
            extra = ctrs_y2[-1] + p2_spacing_y
            ctrs_y2 = np.append(ctrs_y2, extra)
        if len(ctrs_x2) < nx2:
            extra = ctrs_x2[-1] + p2_spacing_x
            ctrs_x2 = np.append(ctrs_x2, extra)
        ctrs_y2 = ctrs_y2[:ny2].astype(np.float32)
        ctrs_x2 = ctrs_x2[:nx2].astype(np.float32)
    else:
        ny2 = len(ctrs_y2)
        nx2 = len(ctrs_x2)
        ctrs_y2 = ctrs_y2.astype(np.float32)
        ctrs_x2 = ctrs_x2.astype(np.float32)

    print(f"Pass 2 centers: y=[{ctrs_y2[0]:.0f}..{ctrs_y2[-1]:.0f}] ({len(ctrs_y2)}), "
          f"x=[{ctrs_x2[0]:.0f}..{ctrs_x2[-1]:.0f}] ({len(ctrs_x2)}), spacing={p2_spacing_y}")

    # Smoothing params (from pass 1 → pass 2)
    ksize_filt, sd = compute_smoothing_params(
        (ny1, nx1), (p1_spacing_y, p1_spacing_x)
    )
    print(f"Smoothing: ksize={ksize_filt}, sigma={sd:.4f}")

    # ── Run all methods ──
    common_args = dict(
        predictor_ux=ux1, predictor_uy=uy1_img,
        src_ctrs_y=ctrs_y1, src_ctrs_x=ctrs_x1,
        src_ctrs_y_all=ctrs_y1_all, src_ctrs_x_all=ctrs_x1_all,
        n_pre_y=n_pre_y, n_post_y=n_post_y,
        n_pre_x=n_pre_x, n_post_x=n_post_x,
        dst_ctrs_y=ctrs_y2, dst_ctrs_x=ctrs_x2,
        H=H, W=W, ksize_filt=ksize_filt, sd=sd,
    )

    methods = [
        method_current,
        method_border_replicate,
        method_linear_extrap,
        method_pchip,
        method_pchip_with_extrap_pad,
        method_no_smooth,
        method_akima,
        method_bicubic_spline,
        method_bilinear,
    ]

    results = []
    timings = []
    for mfn in methods:
        print(f"\nRunning: {mfn.__name__}...")
        t0 = time.perf_counter()
        dense, pred_win, label = mfn(**common_args)
        elapsed = time.perf_counter() - t0
        results.append((dense, pred_win, label))
        timings.append(elapsed)
        # pred_win[..., 1] is ux component
        ux_pred = pred_win[..., 1]
        print(f"  ux predictor at pass2 grid: shape={ux_pred.shape}, "
              f"range=[{ux_pred.min():.4f}, {ux_pred.max():.4f}], mean={np.nanmean(ux_pred):.4f}"
              f"  [{elapsed*1000:.1f} ms]")

    # ── Plotting ──
    n_methods = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    # ── Fig 1: Vertical profiles at mid-column (the money plot) ──
    print("\n" + "=" * 80)
    print("FIGURE 1: Vertical ux profiles at mid-column (wall region focus)")
    print("=" * 80)

    mid_col_dense = W // 2
    mid_col_win = len(ctrs_x2) // 2

    fig1, axes1 = plt.subplots(1, 3, figsize=(21, 7))

    # Full profile
    ax = axes1[0]
    ax.plot(np.arange(H), np.zeros(H), 'k-', lw=0.3, alpha=0.3)
    for idx, (dense, pred_win, label) in enumerate(results):
        ax.plot(dense[:, mid_col_dense, 1], label=label, color=colors[idx], lw=1.2,
                alpha=0.8 if idx > 0 else 1.0, ls="-" if idx == 0 else "--")
    # Overlay pass 1 source points
    ax.plot(ctrs_y1, ux1[:, nx1 // 2], "ko", ms=4, zorder=10, label="Pass 1 output (source)")
    ax.set_xlabel("pixel y")
    ax.set_ylabel("ux predictor [px]")
    ax.set_title("Full vertical profile (dense grid)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # wall at right (high y = bottom of image)

    # Bottom zoom (wall region) — last 15% of image
    ax = axes1[1]
    y_wall_start = int(H * 0.85)
    for idx, (dense, pred_win, label) in enumerate(results):
        ax.plot(np.arange(y_wall_start, H), dense[y_wall_start:, mid_col_dense, 1],
                label=label, color=colors[idx], lw=1.5,
                alpha=0.8 if idx > 0 else 1.0, ls="-" if idx == 0 else "--")
    wall_mask = ctrs_y1 >= y_wall_start
    if wall_mask.any():
        ax.plot(ctrs_y1[wall_mask], ux1[wall_mask, nx1 // 2], "ko", ms=5, zorder=10, label="Pass 1 source")
    ax.set_xlabel("pixel y")
    ax.set_ylabel("ux predictor [px]")
    ax.set_title(f"BOTTOM / WALL zoom (y>{y_wall_start})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Top zoom (centerline) — first 10% of image
    ax = axes1[2]
    y_top_end = int(H * 0.10)
    for idx, (dense, pred_win, label) in enumerate(results):
        ax.plot(np.arange(0, y_top_end), dense[:y_top_end, mid_col_dense, 1],
                label=label, color=colors[idx], lw=1.5,
                alpha=0.8 if idx > 0 else 1.0, ls="-" if idx == 0 else "--")
    top_mask = ctrs_y1 <= y_top_end
    if top_mask.any():
        ax.plot(ctrs_y1[top_mask], ux1[top_mask, nx1 // 2], "ko", ms=5, zorder=10, label="Pass 1 source")
    ax.set_xlabel("pixel y")
    ax.set_ylabel("ux predictor [px]")
    ax.set_title(f"TOP / CENTERLINE zoom (y<{y_top_end})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig1.suptitle("Vertical ux profiles: predictor methods comparison", fontsize=14)
    fig1.tight_layout()
    fig1.savefig(str(OUT_DIR / "test_methods_fig1_profiles.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'test_methods_fig1_profiles.png'}")

    # ── Fig 2: Horizontal profiles at multiple y-positions ──
    print("\nFIGURE 2: Horizontal ux profiles (edge focus)")

    y_positions_frac = [0.02, 0.50, 0.95, 0.99]  # top edge, mid, near wall, very near wall
    fig2, axes2 = plt.subplots(1, len(y_positions_frac), figsize=(7 * len(y_positions_frac), 6))

    for pidx, frac in enumerate(y_positions_frac):
        y_pos = int(frac * (H - 1))
        ax = axes2[pidx]
        for idx, (dense, pred_win, label) in enumerate(results):
            ax.plot(dense[y_pos, :, 1], label=label, color=colors[idx], lw=1.2,
                    alpha=0.8 if idx > 0 else 1.0, ls="-" if idx == 0 else "--")
        ax.set_xlabel("pixel x")
        ax.set_ylabel("ux predictor [px]")
        ax.set_title(f"Horizontal at y={y_pos} ({frac:.0%} from top)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Horizontal ux profiles: edge artefact comparison", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(str(OUT_DIR / "test_methods_fig2_horiz.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'test_methods_fig2_horiz.png'}")

    # ── Fig 2b: Dense 2D maps (full image resolution) with edge zooms ──
    print("\nFIGURE 2b: Dense 2D ux predictor maps (full image, per method)")

    # Full-field maps — one per method + the current method's bottom/right zoom
    edge_px = 120  # pixels to show in edge zoom panels

    for idx, (dense, pred_win, label) in enumerate(results):
        fig2b, axes2b = plt.subplots(2, 2, figsize=(16, 14),
                                     gridspec_kw={"height_ratios": [3, 1], "width_ratios": [3, 1]})

        ux_dense = dense[..., 1]  # ux component

        # Top-left: full 2D map
        ax = axes2b[0, 0]
        vmin_d = np.nanpercentile(ux_dense, 0.5)
        vmax_d = np.nanpercentile(ux_dense, 99.5)
        im = ax.imshow(ux_dense, cmap="viridis", vmin=vmin_d, vmax=vmax_d, aspect="equal")
        ax.set_title(f"{label}\nFull dense field ({H}×{W})", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        # Mark zoom regions
        ax.axhline(H - edge_px, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(W - edge_px, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(edge_px, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(edge_px, color="cyan", ls="--", lw=0.8, alpha=0.6)

        # Top-right: right edge zoom (full height, last edge_px columns)
        ax = axes2b[0, 1]
        right_strip = ux_dense[:, -edge_px:]
        im = ax.imshow(right_strip, cmap="viridis", vmin=vmin_d, vmax=vmax_d, aspect="auto",
                       extent=[W - edge_px, W, H, 0])
        ax.set_title(f"Right edge\n(last {edge_px} cols)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02)

        # Bottom-left: bottom edge zoom (last edge_px rows, full width)
        ax = axes2b[1, 0]
        bot_strip = ux_dense[-edge_px:, :]
        im = ax.imshow(bot_strip, cmap="viridis", vmin=vmin_d, vmax=vmax_d, aspect="auto",
                       extent=[0, W, H, H - edge_px])
        ax.set_title(f"Bottom edge (last {edge_px} rows)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # Bottom-right: corner zoom (bottom-right edge_px × edge_px)
        ax = axes2b[1, 1]
        corner = ux_dense[-edge_px:, -edge_px:]
        im = ax.imshow(corner, cmap="viridis", vmin=vmin_d, vmax=vmax_d, aspect="equal",
                       extent=[W - edge_px, W, H, H - edge_px])
        ax.set_title(f"Bottom-right corner", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02)

        fig2b.tight_layout()
        tag = label.split(".")[0].strip()  # "A", "B", etc.
        fname = f"test_methods_fig2b_{tag}_dense_2d.png"
        fig2b.savefig(str(OUT_DIR / fname), dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUT_DIR / fname}")
        plt.close(fig2b)

    # ── Fig 3: 2D maps of ux predictor at pass 2 window centers ──
    print("\nFIGURE 3: 2D ux predictor fields at pass 2 window grid")

    # Use consistent colorbar from first method's predictor (or pass 2 actual if available)
    ref_for_cbar = ux2 if has_p2 else results[0][1][..., 1]
    vmin_global = np.nanpercentile(ref_for_cbar, 0.5)
    vmax_global = np.nanpercentile(ref_for_cbar, 99.5)

    ncols = 3
    extra_panels = 2 if has_p2 else 0
    nrows = (n_methods + extra_panels + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes3_flat = axes3.flatten()

    for idx, (dense, pred_win, label) in enumerate(results):
        ax = axes3_flat[idx]
        im = ax.imshow(pred_win[..., 1], cmap="viridis", vmin=vmin_global, vmax=vmax_global, aspect="auto")
        ax.set_title(label, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    if has_p2:
        # Actual pass 2 output
        ax = axes3_flat[n_methods]
        im = ax.imshow(ux2, cmap="viridis", vmin=vmin_global, vmax=vmax_global, aspect="auto")
        ax.set_title("ACTUAL pass 2 ux output", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Difference: current method vs actual
        if ux2.shape == results[0][1][..., 1].shape:
            ax = axes3_flat[n_methods + 1]
            diff = ux2 - results[0][1][..., 1]
            dmax = max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)))
            im = ax.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto")
            ax.set_title("Pass 2 actual - current predictor", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for j in range(n_methods + extra_panels, len(axes3_flat)):
        axes3_flat[j].axis("off")

    fig3.suptitle("2D ux predictor fields at pass 2 window centers", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(str(OUT_DIR / "test_methods_fig3_2d.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_DIR / 'test_methods_fig3_2d.png'}")

    if has_p2:
        # ── Fig 4: Residual (pass 2 actual - predictor) for each method ──
        print("\nFIGURE 4: Residuals vs actual pass 2 output")

        fig4, axes4 = plt.subplots(2, ncols, figsize=(7 * ncols, 10))
        axes4_flat = axes4.flatten()

        for idx, (dense, pred_win, label) in enumerate(results):
            if ux2.shape != pred_win[..., 1].shape:
                print(f"  Skipping {label}: shape mismatch {pred_win[...,1].shape} vs {ux2.shape}")
                continue
            ax = axes4_flat[idx]
            residual = ux2 - pred_win[..., 1]
            rmax = max(abs(np.nanpercentile(residual, 1)), abs(np.nanpercentile(residual, 99)))
            if rmax < 0.01:
                rmax = 1.0
            im = ax.imshow(residual, cmap="RdBu_r", vmin=-rmax, vmax=rmax, aspect="auto")
            ax.set_title(f"Residual: actual - {label}\n"
                         f"mean={np.nanmean(residual):.3f} std={np.nanstd(residual):.3f}",
                         fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)

        for j in range(n_methods, len(axes4_flat)):
            axes4_flat[j].axis("off")

        fig4.suptitle("Residual: pass 2 actual output - predictor (ideally small + uniform)", fontsize=13)
        fig4.tight_layout()
        fig4.savefig(str(OUT_DIR / "test_methods_fig4_residuals.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUT_DIR / 'test_methods_fig4_residuals.png'}")
    else:
        print("\nFIGURE 4: SKIPPED (no pass 2 ground truth for residuals)")

    # ── Fig 5: Quantitative comparison table ──
    print("\n" + "=" * 80)
    if has_p2:
        print("QUANTITATIVE COMPARISON (predictor vs actual pass 2 ux)")
    else:
        print("QUANTITATIVE SUMMARY (predictor field statistics)")
    print("=" * 80)

    if has_p2:
        print(f"{'Method':<50s} {'Mean res':>10s} {'Std res':>10s} {'Max |res|':>10s} {'Bot 5':>10s} {'Interior':>10s} {'Time ms':>10s}")
        print("-" * 112)
        for i, (dense, pred_win, label) in enumerate(results):
            ux_pred = pred_win[..., 1]
            if ux2.shape != ux_pred.shape:
                print(f"{label:<50s}  SHAPE MISMATCH")
                continue
            residual = ux2 - ux_pred
            edge = 5
            bot_mean = np.nanmean(ux_pred[-edge:, :])
            int_mean = np.nanmean(ux_pred[edge:-edge, edge:-edge])
            print(f"{label:<50s} {np.nanmean(residual):>10.4f} {np.nanstd(residual):>10.4f} "
                  f"{np.nanmax(np.abs(residual)):>10.4f} {bot_mean:>10.4f} {int_mean:>10.4f} {timings[i]*1000:>10.1f}")
        edge = 5
        print(f"{'ACTUAL pass 2 ux':<50s} {'---':>10s} {'---':>10s} {'---':>10s} "
              f"{np.nanmean(ux2[-edge:, :]):>10.4f} {np.nanmean(ux2[edge:-edge, edge:-edge]):>10.4f}")
    else:
        print(f"{'Method':<50s} {'Min ux':>10s} {'Max ux':>10s} {'Mean ux':>10s} {'Bot 5':>10s} {'Interior':>10s} {'Time ms':>10s}")
        print("-" * 112)
        for i, (dense, pred_win, label) in enumerate(results):
            ux_pred = pred_win[..., 1]
            edge = 5
            bot_mean = np.nanmean(ux_pred[-edge:, :])
            int_mean = np.nanmean(ux_pred[edge:-edge, edge:-edge])
            print(f"{label:<50s} {np.nanmin(ux_pred):>10.4f} {np.nanmax(ux_pred):>10.4f} "
                  f"{np.nanmean(ux_pred):>10.4f} {bot_mean:>10.4f} {int_mean:>10.4f} {timings[i]*1000:>10.1f}")

    plt.close("all")
    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
