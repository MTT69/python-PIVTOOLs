#!/usr/bin/env python3
"""
stereo_dewarp_diagnostic.py

Standalone diagnostic for stereo dewarp overlays — works with any stereo
geometry including transmission (~180° stereo angle).

Loads a stereo model .mat, reads raw images from a .set (or any supported
format), dewarps both cameras to a common world Z-plane, and displays a
red-cyan overlay with full diagnostics.

Interactive Z slider lets you sweep Z to visualise disparity.

Usage (edit the CONFIGURATION block, then run):
    python stereo_dewarp_diagnostic.py
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Ensure pivtools is importable
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    dewarp_image,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
# Path to the stereo model .mat saved by stepped_board / dotboard / charuco
STEREO_MODEL_PATH = r"D:\Andre\NEW\Project_FlowMaster_260129_103855_x0_2\Recording_Date=260129_Time=114615_AoA0\MT_test\calibration\stereo_cam1_cam2\model\stereo_model.mat"

# Source images — either individual files or a .set container
# For .set: set SOURCE_TYPE="set" and give the .set path
# For files: set SOURCE_TYPE="file" and give per-camera paths
SOURCE_TYPE = "set"  # "set" or "file"
SET_PATH = r"D:\Andre\NEW\Project_FlowMaster_260129_103855_x0_2\Recording_Date=260129_Time=114615_AoA0.set"
FRAME_INDEX = 1          # 1-based
CAM1_NUM = 1             # Camera numbers (1-based, as configured)
CAM2_NUM = 2

# For SOURCE_TYPE="file":
CAM1_IMAGE_PATH = None   # e.g. r"...\cam1_frame007.tif"
CAM2_IMAGE_PATH = None

# Background subtraction: time-minimum over this many pairs (both A+B frames)
N_BG_FRAMES = 10

# Dewarped output resolution
MM_PER_PIXEL = 0.1       # mm per output pixel (coarser = faster)

# Z sweep range (mm) for interactive slider
Z_MIN = -10.0
Z_MAX = 10.0
Z_INIT = 0.0

# Use per-camera Z? (like calibration figure — set True for calibration images)
# When False, both cameras dewarp to the same Z (correct for particle images)
PER_CAMERA_Z = False
CAM1_Z = 0.0             # Only used if PER_CAMERA_Z is True
CAM2_Z = 0.0
# =========================================================


def load_stereo_model(path: str):
    """Load stereo model and return two PinholeCameras + raw dicts.

    Returns (cam1, cam2, pr1, pr2, metadata).
    pr1/pr2 are dicts with {K, dist, rvec, tvec} for direct cv2 use.
    """
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)

    image_size_arr = np.array(mat["image_size"]).flatten()
    w, h = int(image_size_arr[0]), int(image_size_arr[1])

    K1 = np.array(mat["camera_matrix_1"]).astype(np.float64)
    K2 = np.array(mat["camera_matrix_2"]).astype(np.float64)
    dist1 = np.array(mat["dist_coeffs_1"]).flatten().astype(np.float64)
    dist2 = np.array(mat["dist_coeffs_2"]).flatten().astype(np.float64)

    # --- Cam1 extrinsics (directly from model) ---
    rvecs1 = np.array(mat["rvecs_1"]).astype(np.float64)
    tvecs1 = np.array(mat["tvecs_1"]).astype(np.float64)

    datum_frame = int(mat["datum_frame"]) if "datum_frame" in mat else 0
    if rvecs1.ndim == 1:
        rvec1 = rvecs1
        tvec1 = tvecs1
    else:
        idx = max(0, datum_frame - 1) if datum_frame > 0 else 0
        rvec1 = rvecs1[idx].flatten()
        tvec1 = tvecs1[idx].flatten()

    R1, _ = cv2.Rodrigues(rvec1)
    t1 = tvec1.reshape(3, 1)

    # --- Cam2 extrinsics ---
    # Try saved rvecs_2/tvecs_2 first (stepped board saves these)
    if "rvecs_2" in mat and "tvecs_2" in mat:
        rvecs2 = np.array(mat["rvecs_2"]).astype(np.float64)
        tvecs2 = np.array(mat["tvecs_2"]).astype(np.float64)
        if rvecs2.ndim == 1:
            rvec2 = rvecs2
            tvec2 = tvecs2
        else:
            idx2 = max(0, datum_frame - 1) if datum_frame > 0 else 0
            rvec2 = rvecs2[idx2].flatten()
            tvec2 = tvecs2[idx2].flatten()
        R2, _ = cv2.Rodrigues(rvec2)
        t2 = tvec2.reshape(3, 1)
        logger.info("Loaded cam2 extrinsics directly from model (rvecs_2/tvecs_2)")
    else:
        # Derive from stereo R/T (standard dotboard/charuco path)
        R_stereo = np.array(mat["rotation_matrix"]).astype(np.float64)
        T_stereo = np.array(mat["translation_vector"]).astype(np.float64).reshape(3, 1)
        R2 = R_stereo @ R1
        t2 = R_stereo @ t1 + T_stereo
        rvec2, _ = cv2.Rodrigues(R2)
        rvec2 = rvec2.flatten()
        tvec2 = t2.flatten()
        logger.info("Derived cam2 extrinsics from stereo R/T")

    # Also derive from stereo R/T to compare
    R_stereo = np.array(mat["rotation_matrix"]).astype(np.float64)
    T_stereo = np.array(mat["translation_vector"]).astype(np.float64).reshape(3, 1)
    R2_derived = R_stereo @ R1
    t2_derived = R_stereo @ t1 + T_stereo

    # Check consistency
    R_diff = np.linalg.norm(R2 - R2_derived)
    t_diff = np.linalg.norm(t2 - t2_derived)
    if R_diff > 1e-6 or t_diff > 1e-6:
        logger.warning(
            f"Cam2 extrinsic mismatch! R_diff={R_diff:.2e}, t_diff={t_diff:.2e}"
        )
    else:
        logger.info(f"Cam2 extrinsics consistent (R_diff={R_diff:.2e}, t_diff={t_diff:.2e})")

    cam1 = PinholeCamera(K=K1, dist=dist1, R=R1, t=t1, image_size=(w, h))
    cam2 = PinholeCamera(K=K2, dist=dist2, R=R2, t=t2, image_size=(w, h))

    pr1 = {"K": K1, "dist": dist1, "rvec": rvec1, "tvec": tvec1.flatten()}
    pr2 = {"K": K2, "dist": dist2, "rvec": rvec2, "tvec": tvec2.flatten()}

    # --- Diagnostics ---
    R_rel = R2 @ R1.T
    import math
    trace_val = np.trace(R_rel)
    full_angle = math.acos(max(-1.0, min(1.0, (trace_val - 1.0) / 2.0)))
    baseline = float(np.linalg.norm(t2 - R_rel @ t1))

    # Camera positions in world coords
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    logger.info(f"Image size: {w}x{h}")
    logger.info(f"Stereo angle: {math.degrees(full_angle):.1f}°")
    logger.info(f"Baseline: {baseline:.1f} mm")
    logger.info(f"Cam1 world position: [{C1[0,0]:.1f}, {C1[1,0]:.1f}, {C1[2,0]:.1f}]")
    logger.info(f"Cam2 world position: [{C2[0,0]:.1f}, {C2[1,0]:.1f}, {C2[2,0]:.1f}]")

    metadata = {
        "image_size": (w, h),
        "full_angle_deg": math.degrees(full_angle),
        "baseline_mm": baseline,
        "cam1_world": C1.flatten(),
        "cam2_world": C2.flatten(),
    }

    if "relative_angle_deg" in mat:
        logger.info(f"Model relative_angle_deg: {float(mat['relative_angle_deg']):.1f}°")

    return cam1, cam2, pr1, pr2, metadata


def _compute_time_minimum(set_path, cam_num, n_bg_frames=10):
    """Compute per-pixel minimum across n_bg_frames pairs (both A and B frames).

    This gives a robust background estimate — anything that doesn't move
    (sensor noise floor, reflections, static features) stays; particles vanish.
    """
    from pivtools_core.image_handling.readers.set_reader import read_set_pair
    bg = None
    for idx in range(1, n_bg_frames + 1):
        try:
            pair = read_set_pair(set_path, cam_num, idx)
            for frame in (pair[0], pair[1]) if pair.ndim == 3 else (pair,):
                f = frame.astype(np.float64)
                bg = f if bg is None else np.minimum(bg, f)
        except Exception:
            break
    if bg is None:
        raise ValueError(f"Could not load any frames for cam {cam_num}")
    logger.info(f"Cam{cam_num} time-minimum from {idx - 1} pairs: "
                f"range=[{bg.min():.0f}, {bg.max():.0f}]")
    return bg


def load_images(config_locals):
    """Load source images for both cameras. Returns (img1, img2) as float64.

    Computes a per-pixel time-minimum background from N_BG_FRAMES pairs
    and subtracts it, leaving only particle signal.
    """
    if config_locals["source_type"] == "set":
        from pivtools_core.image_handling.readers.set_reader import read_set_pair
        set_path = config_locals["set_path"]
        frame_idx = config_locals["frame_index"]
        cam1_num = config_locals["cam1_num"]
        cam2_num = config_locals["cam2_num"]
        n_bg = config_locals.get("n_bg_frames", 10)

        logger.info(f"Computing time-minimum background from {n_bg} pairs...")
        bg1 = _compute_time_minimum(set_path, cam1_num, n_bg)
        bg2 = _compute_time_minimum(set_path, cam2_num, n_bg)

        pair1 = read_set_pair(set_path, cam1_num, frame_idx)
        pair2 = read_set_pair(set_path, cam2_num, frame_idx)
        img1 = (pair1[0] if pair1.ndim == 3 else pair1).astype(np.float64) - bg1
        img2 = (pair2[0] if pair2.ndim == 3 else pair2).astype(np.float64) - bg2
    elif config_locals["source_type"] == "file":
        img1 = cv2.imread(config_locals["cam1_image_path"], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(config_locals["cam2_image_path"], cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError(f"Unknown SOURCE_TYPE: {config_locals['source_type']}")

    if img1 is None:
        raise ValueError("Failed to load cam1 image")
    if img2 is None:
        raise ValueError("Failed to load cam2 image")

    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Clip negatives from background subtraction
    img1 = np.clip(img1, 0, None)
    img2 = np.clip(img2, 0, None)

    logger.info(f"Cam1 image: shape={img1.shape}, dtype={img1.dtype}, "
                f"range=[{img1.min()}, {img1.max()}]")
    logger.info(f"Cam2 image: shape={img2.shape}, dtype={img2.dtype}, "
                f"range=[{img2.min()}, {img2.max()}]")

    return img1.astype(np.float64), img2.astype(np.float64)


def compute_world_bounds(pr1, pr2, w, h):
    """Project image edges to Z=0 for both cameras, return intersection.

    Also returns per-camera bounds for diagnostics.
    """
    from pivtools_gui.calibration.global_coordinate_alignment import _pixels_to_world_mm

    def camera_bounds(pr, w, h, label):
        n = 20
        top = np.column_stack([np.linspace(0, w - 1, n), np.zeros(n)])
        bottom = np.column_stack([np.linspace(0, w - 1, n), np.full(n, h - 1)])
        left = np.column_stack([np.zeros(n), np.linspace(0, h - 1, n)])
        right = np.column_stack([np.full(n, w - 1), np.linspace(0, h - 1, n)])
        pts = np.vstack([top, bottom, left, right]).astype(np.float32)

        world = _pixels_to_world_mm(pts, pr["K"], pr["dist"], pr["rvec"], pr["tvec"])
        valid = ~np.isnan(world).any(axis=1)
        n_valid = valid.sum()
        world = world[valid]

        if world.size == 0:
            raise ValueError(f"{label}: All edge projections returned NaN")

        bounds = (
            float(world[:, 0].min()), float(world[:, 0].max()),
            float(world[:, 1].min()), float(world[:, 1].max()),
        )
        logger.info(
            f"{label} FOV on Z=0: x=[{bounds[0]:.1f}, {bounds[1]:.1f}], "
            f"y=[{bounds[2]:.1f}, {bounds[3]:.1f}] mm  "
            f"({n_valid}/{len(pts)} valid projections)"
        )
        return bounds

    b1 = camera_bounds(pr1, w, h, "Cam1")
    b2 = camera_bounds(pr2, w, h, "Cam2")

    xmin = max(b1[0], b2[0])
    xmax = min(b1[1], b2[1])
    ymin = max(b1[2], b2[2])
    ymax = min(b1[3], b2[3])

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            f"No overlapping FOV! Cam1: {b1}, Cam2: {b2}"
        )

    logger.info(
        f"Intersection: x=[{xmin:.1f}, {xmax:.1f}], "
        f"y=[{ymin:.1f}, {ymax:.1f}] mm"
    )
    return (xmin, xmax, ymin, ymax), b1, b2


def build_dewarp_maps(pr, world_bounds, mm_per_pixel, z_mm):
    """Build dewarp remap tables using cv2.projectPoints directly (like calibration figure)."""
    x_min, x_max, y_min, y_max = world_bounds
    nx = max(1, int(round((x_max - x_min) / mm_per_pixel)))
    ny = max(1, int(round((y_max - y_min) / mm_per_pixel)))

    x_1d = np.linspace(x_min, x_max, nx)
    y_1d = np.linspace(y_min, y_max, ny)
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d)
    Z_grid = np.full_like(X_grid, z_mm)

    world_pts = np.column_stack([
        X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()
    ]).astype(np.float64)

    projected, _ = cv2.projectPoints(
        world_pts, pr["rvec"], pr["tvec"], pr["K"], pr["dist"]
    )
    projected = projected.reshape(-1, 2)

    map_x = projected[:, 0].reshape(ny, nx).astype(np.float32)
    map_y = projected[:, 1].reshape(ny, nx).astype(np.float32)
    return map_x, map_y, (nx, ny)


def diagnose_maps(map_x, map_y, img_w, img_h, label):
    """Print diagnostic info about dewarp maps."""
    total = map_x.size
    valid = (map_x >= 0) & (map_x < img_w) & (map_y >= 0) & (map_y < img_h)
    n_valid = valid.sum()
    pct = 100.0 * n_valid / total

    logger.info(
        f"{label} maps: "
        f"map_x=[{map_x.min():.1f}, {map_x.max():.1f}], "
        f"map_y=[{map_y.min():.1f}, {map_y.max():.1f}], "
        f"valid={n_valid}/{total} ({pct:.1f}%), "
        f"image={img_w}x{img_h}"
    )
    return valid


def dewarp_and_split(img1, img2, pr1, pr2, world_bounds, mm_per_pixel, z1, z2):
    """Dewarp both cameras, return raw float arrays + valid masks + extent."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    map1_x, map1_y, (nx, ny) = build_dewarp_maps(pr1, world_bounds, mm_per_pixel, z1)
    map2_x, map2_y, _ = build_dewarp_maps(pr2, world_bounds, mm_per_pixel, z2)

    valid1 = diagnose_maps(map1_x, map1_y, w1, h1, "Cam1")
    valid2 = diagnose_maps(map2_x, map2_y, w2, h2, "Cam2")

    dw1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dw2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    dw1[~valid1] = 0
    dw2[~valid2] = 0

    logger.info(f"Dewarped cam1: range=[{dw1.min():.1f}, {dw1.max():.1f}]")
    logger.info(f"Dewarped cam2: range=[{dw2.min():.1f}, {dw2.max():.1f}]")

    x_min, x_max, y_min, y_max = world_bounds
    extent = [x_min, x_max, y_min, y_max]
    return dw1, dw2, valid1, valid2, extent


def build_overlay(dw1, dw2, valid1, valid2, vmax_pct):
    """Build red-cyan overlay from raw dewarped float images.

    vmax_pct controls brightness: percentile of positive pixels used as white point.
    Lower = brighter (more clipping). Range ~90-100.
    """
    def normalize_u8(im, valid_mask):
        pos = im[valid_mask & (im > 0)]
        if pos.size == 0:
            return np.zeros(im.shape, dtype=np.uint8)
        lo = float(np.percentile(pos, 1))
        hi = float(np.percentile(pos, vmax_pct))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        out = ((im - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        out[~valid_mask] = 0
        return out

    r_ch = normalize_u8(dw1, valid1)
    c_ch = normalize_u8(dw2, valid2)
    return np.stack([r_ch, c_ch, c_ch], axis=-1)


def main():
    logger.info("=" * 60)
    logger.info("Stereo Dewarp Diagnostic Tool")
    logger.info("=" * 60)

    # Load model
    cam1, cam2, pr1, pr2, meta = load_stereo_model(STEREO_MODEL_PATH)
    w, h = meta["image_size"]

    # Load images
    cfg = {
        "source_type": SOURCE_TYPE,
        "set_path": SET_PATH,
        "frame_index": FRAME_INDEX,
        "cam1_num": CAM1_NUM,
        "cam2_num": CAM2_NUM,
        "cam1_image_path": CAM1_IMAGE_PATH,
        "cam2_image_path": CAM2_IMAGE_PATH,
        "n_bg_frames": N_BG_FRAMES,
    }
    img1, img2 = load_images(cfg)

    # Compute world bounds
    world_bounds, b1, b2 = compute_world_bounds(pr1, pr2, w, h)

    # Dewarp once (expensive), then rescale interactively (cheap)
    if PER_CAMERA_Z:
        z1, z2 = CAM1_Z, CAM2_Z
    else:
        z1 = z2 = Z_INIT

    logger.info("Computing dewarp maps (this may take a moment)...")
    dw1, dw2, valid1, valid2, extent = dewarp_and_split(
        img1, img2, pr1, pr2, world_bounds, MM_PER_PIXEL, z1, z2
    )

    # --- Normalize dewarped images for display ---
    def normalize_float(im, valid_mask, pct):
        pos = im[valid_mask & (im > 0)]
        if pos.size == 0:
            return np.zeros(im.shape, dtype=np.float32)
        lo = float(np.percentile(pos, 1))
        hi = float(np.percentile(pos, pct))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        out = ((im - lo) / (hi - lo)).clip(0, 1).astype(np.float32)
        out[~valid_mask] = 0
        return out

    init_pct = 99.5

    # --- 3-panel figure: Cam1 | Overlay | Cam2 ---
    fig, (ax1, ax_ov, ax2) = plt.subplots(1, 3, figsize=(20, 8),
                                           sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.12, wspace=0.05)

    n1 = normalize_float(dw1, valid1, init_pct)
    n2 = normalize_float(dw2, valid2, init_pct)
    overlay = np.stack([n1, n2, n2], axis=-1)

    im1_h = ax1.imshow(n1, extent=extent, origin="lower", aspect="equal",
                        cmap="gray", vmin=0, vmax=1)
    im_ov = ax_ov.imshow(overlay, extent=extent, origin="lower", aspect="equal")
    im2_h = ax2.imshow(n2, extent=extent, origin="lower", aspect="equal",
                        cmap="gray", vmin=0, vmax=1)

    ax1.set_title(f"Cam{CAM1_NUM}", fontsize=11)
    ax_ov.set_title(
        f"Overlay  |  Z={z1:.1f}mm  |  {meta['full_angle_deg']:.1f}°  |  "
        f"Frame {FRAME_INDEX}",
        fontsize=11,
    )
    ax2.set_title(f"Cam{CAM2_NUM}", fontsize=11)
    ax1.set_ylabel("Y world (mm)")
    for a in (ax1, ax_ov, ax2):
        a.set_xlabel("X world (mm)")

    # Start zoomed into center ~20mm window so particles are visible
    cx = (extent[0] + extent[1]) / 2
    cy = (extent[2] + extent[3]) / 2
    half_w = 10.0  # mm
    ax1.set_xlim(cx - half_w, cx + half_w)
    ax1.set_ylim(cy - half_w, cy + half_w)

    # --- Brightness slider ---
    ax_slider = plt.axes([0.15, 0.03, 0.7, 0.025])
    scale_slider = Slider(
        ax_slider, "Brightness (percentile)",
        90.0, 100.0, valinit=init_pct, valstep=0.1,
    )

    def update_scale(val):
        pct = scale_slider.val
        n1_new = normalize_float(dw1, valid1, pct)
        n2_new = normalize_float(dw2, valid2, pct)
        ov_new = np.stack([n1_new, n2_new, n2_new], axis=-1)
        im1_h.set_data(n1_new)
        im_ov.set_data(ov_new)
        im2_h.set_data(n2_new)
        fig.canvas.draw_idle()

    scale_slider.on_changed(update_scale)

    # --- Synced scroll-to-zoom across all 3 panels ---
    _zoom_state = {"home_xlim": None, "home_ylim": None}
    all_axes = [ax1, ax_ov, ax2]

    def on_scroll(event):
        if event.inaxes not in all_axes:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xdata, ydata = event.xdata, event.ydata
        xl, xr = event.inaxes.get_xlim()
        yb, yt = event.inaxes.get_ylim()
        new_xl = xdata - (xdata - xl) * factor
        new_xr = xdata - (xdata - xr) * factor
        new_yb = ydata - (ydata - yb) * factor
        new_yt = ydata - (ydata - yt) * factor
        for a in all_axes:
            a.set_xlim(new_xl, new_xr)
            a.set_ylim(new_yb, new_yt)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "r":
            if _zoom_state["home_xlim"] is not None:
                for a in all_axes:
                    a.set_xlim(_zoom_state["home_xlim"])
                    a.set_ylim(_zoom_state["home_ylim"])
                fig.canvas.draw_idle()

    def on_draw(event):
        if _zoom_state["home_xlim"] is None:
            _zoom_state["home_xlim"] = ax1.get_xlim()
            _zoom_state["home_ylim"] = ax1.get_ylim()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("draw_event", on_draw)

    logger.info("Controls: scroll=zoom (synced), 'r'=reset zoom, slider=brightness")
    logger.info("Initial view: 20mm x 20mm centered on FOV")
    plt.show()


if __name__ == "__main__":
    main()
