#!/usr/bin/env python3
"""
image_dewarp_overlay.py

Dewarps raw PIV images from multiple cameras into physical coordinates
and overlays them in a single interactive matplotlib figure.

Interaction modes (toggle with 'm'):
  Navigate: click to show (x_mm, y_mm) with a crosshair
  Measure:  click two points to show distance, angle, and y-gap

Scroll wheel to zoom in/out (centred on cursor).
Press 'r' to reset zoom to full extent.
Press 'c' to clear all markers and annotations.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from scipy.io import loadmat

sys.path.append(str(Path(__file__).parent.parent))
from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    compute_dewarp_maps,
    dewarp_image,
)
from pivtools_gui.calibration.global_coordinate_alignment import _pixels_to_world_mm
from pivtools_gui.calibration.camera_model_utils import (
    load_pinhole_camera,
    compute_camera_world_bounds,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================
# Paths (required — no config.yaml fallback)
BASE_DIR = r"C:\Users\mtt1e23\Downloads\cropped_calibrationimgs\cropped_calibrationimgs\calibrationwarping"
SOURCE_DIR = r"C:\Users\mtt1e23\Downloads\cropped_calibrationimgs\cropped_calibrationimgs"

# Override or None to load from config.yaml
CAMERA_NUMS = [1, 2, 3, 4, 5]
MODEL_TYPE = "dotboard"
IMAGE_FORMAT = "B%05d.tif"   # Calibration image naming
DATUM_CAMERA = None          # None = config datum_camera
DATUM_PIXEL = None           # None = config datum_pixel, or e.g. [100.5, 2000.3]
DATUM_PHYSICAL = None        # None = config datum_physical, or e.g. [0.0, 0.0]
OVERLAP_PAIRS = None         # None = config overlap_pairs
INVERT_UX = None             # None = config invert_ux

# Source image camera subfolder pattern: "Cam{N}" or "camera{N}"
SOURCE_CAMERA_FOLDER = "camera{cam}"  # Use {cam} as placeholder for camera number

# Script-specific (always set here)
FRAME_INDEX = 1              # Which frame to dewarp (1-based)
MM_PER_PIXEL = 0.1           # Dewarped output resolution (mm per output pixel)
ALPHA = 0.5                  # Overlay transparency
CAMERA_CMAPS = None          # None = all gray, or ["Reds", "Greens", ...]
SHOW_CAMERA_BOUNDARIES = True
SAVE_DIAGNOSTICS = True          # Save per-camera diagnostic PNGs (raw, dewarped, overlay)

# Per-camera manual adjustments: rotation (deg) about image centre, then x/y offset (mm)
# Applied ON TOP of datum/overlap chain shifts.
CAMERA_ADJUSTMENTS = {
    # cam_num: (rotation_deg, x_offset_mm, y_offset_mm)
    # 1: (0.0, 0.0, 0.0),
    # 2: (0.5, 1.2, -0.3),
}
# =========================================================


def resolve_config() -> dict:
    """Merge script-level overrides with config.yaml fallbacks.

    Script-level values always win when non-None.
    None values fall back to config.yaml via get_config().
    """
    cfg_data = {}
    try:
        cfg = get_config()
        gc = cfg.global_coordinates_config
        cfg_data = {
            "camera_nums": cfg.camera_numbers,
            "model_type": cfg.active_calibration_method,
            "image_format": cfg.image_format[0],
            "datum_camera": gc.get("datum_camera", 1),
            "datum_pixel": gc.get("datum_pixel"),
            "datum_physical": gc.get("datum_physical", [0.0, 0.0]),
            "overlap_pairs": cfg.global_coordinates_overlap_pairs,
            "invert_ux": gc.get("invert_ux", False),
        }
        logger.info("Loaded fallback values from config.yaml")
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}")

    resolved = {
        "base_dir": BASE_DIR,
        "source_dir": SOURCE_DIR,
        "camera_nums": CAMERA_NUMS if CAMERA_NUMS is not None else cfg_data.get("camera_nums"),
        "model_type": MODEL_TYPE if MODEL_TYPE is not None else cfg_data.get("model_type"),
        "image_format": IMAGE_FORMAT if IMAGE_FORMAT is not None else cfg_data.get("image_format"),
        "datum_camera": DATUM_CAMERA if DATUM_CAMERA is not None else cfg_data.get("datum_camera"),
        "datum_pixel": DATUM_PIXEL if DATUM_PIXEL is not None else cfg_data.get("datum_pixel"),
        "datum_physical": DATUM_PHYSICAL if DATUM_PHYSICAL is not None else cfg_data.get("datum_physical"),
        "overlap_pairs": OVERLAP_PAIRS if OVERLAP_PAIRS is not None else cfg_data.get("overlap_pairs", []),
        "invert_ux": INVERT_UX if INVERT_UX is not None else cfg_data.get("invert_ux", False),
        "frame_index": FRAME_INDEX,
        "mm_per_pixel": MM_PER_PIXEL,
        "alpha": ALPHA,
        "camera_cmaps": CAMERA_CMAPS,
        "show_boundaries": SHOW_CAMERA_BOUNDARIES,
        "camera_adjustments": CAMERA_ADJUSTMENTS,
        "source_camera_folder": SOURCE_CAMERA_FOLDER,
        "save_diagnostics": SAVE_DIAGNOSTICS,
    }

    for key in ("camera_nums", "model_type", "image_format"):
        if resolved[key] is None:
            raise ValueError(
                f"'{key}' is None and config.yaml did not provide a fallback. "
                f"Set the corresponding script variable or ensure config.yaml is accessible."
            )

    return resolved


# ---------------------------------------------------------------------------
# Step 1–2: load_pinhole_camera and compute_camera_world_bounds are now
#           imported from pivtools_gui.calibration.camera_model_utils
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 3: Compute global alignment shifts
# ---------------------------------------------------------------------------

def compute_camera_shifts(
    cameras: Dict[int, PinholeCamera],
    model_data: Dict[int, dict],
    config: dict,
) -> Dict[int, Tuple[float, float]]:
    """Compute global alignment shifts for each camera.

    1. Convert datum pixel -> world coords on datum camera
    2. Compute datum shift = desired_physical - calibrated_physical
    3. Chain through overlap pairs to propagate shifts
    """
    datum_camera = config.get("datum_camera", 1)
    datum_pixel = config.get("datum_pixel")
    datum_physical = config.get("datum_physical", [0.0, 0.0])
    overlap_pairs = config.get("overlap_pairs", [])

    if datum_pixel is None:
        logger.info("No datum_pixel configured — all cameras use zero shift")
        return {cam: (0.0, 0.0) for cam in cameras}

    # Datum camera: raw pixel -> world -> shift
    md = model_data[datum_camera]
    datum_px = np.array([datum_pixel], dtype=np.float32)
    datum_world = _pixels_to_world_mm(
        datum_px, md["camera_matrix"], md["dist_coeffs"], md["rvec"], md["tvec"]
    )
    datum_calibrated = (float(datum_world[0, 0]), float(datum_world[0, 1]))

    datum_shift_x = datum_physical[0] - datum_calibrated[0]
    datum_shift_y = datum_physical[1] - datum_calibrated[1]
    shifts = {datum_camera: (datum_shift_x, datum_shift_y)}
    logger.info(
        f"Datum cam {datum_camera}: pixel {datum_pixel} -> "
        f"calibrated ({datum_calibrated[0]:.3f}, {datum_calibrated[1]:.3f}) mm, "
        f"shift ({datum_shift_x:.3f}, {datum_shift_y:.3f}) mm"
    )

    # Chain through overlap pairs (sorted for deterministic order)
    for pair in sorted(overlap_pairs, key=lambda p: (p["camera_a"], p["camera_b"])):
        cam_a = pair["camera_a"]
        cam_b = pair["camera_b"]
        pixel_a = pair.get("pixel_on_a")
        pixel_b = pair.get("pixel_on_b")

        if pixel_a is None or pixel_b is None:
            logger.warning(f"Skipping pair ({cam_a}, {cam_b}): incomplete pixels")
            continue
        if cam_a not in shifts:
            logger.warning(
                f"Skipping pair ({cam_a}, {cam_b}): cam {cam_a} has no shift yet "
                f"(chain broken)"
            )
            continue

        md_a = model_data[cam_a]
        phys_a = _pixels_to_world_mm(
            np.array([pixel_a], dtype=np.float32),
            md_a["camera_matrix"], md_a["dist_coeffs"],
            md_a["rvec"], md_a["tvec"],
        )
        shift_a = shifts[cam_a]
        phys_a_shifted = (
            float(phys_a[0, 0]) + shift_a[0],
            float(phys_a[0, 1]) + shift_a[1],
        )

        md_b = model_data[cam_b]
        phys_b = _pixels_to_world_mm(
            np.array([pixel_b], dtype=np.float32),
            md_b["camera_matrix"], md_b["dist_coeffs"],
            md_b["rvec"], md_b["tvec"],
        )

        shift_b_x = phys_a_shifted[0] - float(phys_b[0, 0])
        shift_b_y = phys_a_shifted[1] - float(phys_b[0, 1])
        shifts[cam_b] = (shift_b_x, shift_b_y)
        logger.info(f"Cam {cam_b} shift: ({shift_b_x:.3f}, {shift_b_y:.3f}) mm")

    # Any cameras not reached by the chain get zero shift
    for cam in cameras:
        if cam not in shifts:
            logger.warning(f"Cam {cam} not in shift chain — using zero shift")
            shifts[cam] = (0.0, 0.0)

    return shifts


# ---------------------------------------------------------------------------
# Step 4: Load raw image and dewarp
# ---------------------------------------------------------------------------

def load_and_dewarp_camera(
    camera: PinholeCamera,
    cam_num: int,
    bounds: Tuple[float, float, float, float],
    mm_per_pixel: float,
    source_dir: str,
    image_format: str,
    frame_index: int,
    camera_folder_pattern: str = "Cam{cam}",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a raw image and dewarp it into world coordinates.

    Returns (raw_image, dewarped_image) or None if the image cannot be loaded.
    """
    cam_folder = camera_folder_pattern.format(cam=cam_num)
    image_path = Path(source_dir) / cam_folder / (image_format % frame_index)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return None

    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        logger.error(f"Failed to read image: {image_path}")
        return None

    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    x_min, x_max, y_min, y_max = bounds
    out_w = max(1, int(round((x_max - x_min) / mm_per_pixel)))
    out_h = max(1, int(round((y_max - y_min) / mm_per_pixel)))

    if out_w > 10000 or out_h > 10000:
        logger.warning(
            f"Cam {cam_num}: output size {out_w}x{out_h} px exceeds 10000. "
            f"Consider increasing MM_PER_PIXEL to reduce memory usage."
        )

    map_x, map_y = compute_dewarp_maps(camera, bounds, mm_per_pixel)
    dewarped = dewarp_image(raw, map_x, map_y)

    logger.info(
        f"Cam {cam_num}: dewarped {raw.shape} -> {dewarped.shape} "
        f"({out_w}x{out_h} px at {mm_per_pixel} mm/px)"
    )
    return raw, dewarped


# ---------------------------------------------------------------------------
# Step 5: Build the matplotlib overlay figure
# ---------------------------------------------------------------------------

def _draw_overlay_on_axes(
    ax: plt.Axes,
    dewarped_images: Dict[int, np.ndarray],
    bounds: Dict[int, Tuple[float, float, float, float]],
    shifts: Dict[int, Tuple[float, float]],
    config: dict,
) -> None:
    """Draw all dewarped cameras onto an existing axes.

    Core rendering logic shared by the interactive overlay and diagnostic PNGs.
    """
    camera_cmaps = config.get("camera_cmaps")
    alpha = config.get("alpha", 0.5)
    show_boundaries = config.get("show_boundaries", True)
    invert_ux = config.get("invert_ux", False)
    adjustments = config.get("camera_adjustments", {})
    cam_nums = sorted(dewarped_images.keys())

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")

    if camera_cmaps and len(camera_cmaps) >= len(cam_nums):
        cmaps = camera_cmaps
    else:
        cmaps = ["gray"] * len(cam_nums)

    boundary_colors = plt.cm.tab10.colors
    all_corners = []

    for i, cam in enumerate(cam_nums):
        img = np.flipud(dewarped_images[cam])  # flip to match negated y-axis
        x_min, x_max, y_min, y_max = bounds[cam]
        sx, sy = shifts.get(cam, (0.0, 0.0))

        # Shifted (pre-rotation) extent — negate y so boundary layer reads
        # 0 → +175 mm (physical convention: y increases away from wall)
        ext_x0 = x_min + sx
        ext_x1 = x_max + sx
        ext_y0 = -(y_max + sy)   # negate and swap min/max
        ext_y1 = -(y_min + sy)
        extent = [ext_x0, ext_x1, ext_y0, ext_y1]

        # Per-camera adjustment: (rotation_deg, x_offset_mm, y_offset_mm)
        angle_deg, dx, dy = adjustments.get(cam, (0.0, 0.0, 0.0))

        # Build affine transform: rotate about image centre, then translate
        cx = (ext_x0 + ext_x1) / 2
        cy = (ext_y0 + ext_y1) / 2
        rot_tf = Affine2D().rotate_deg_around(cx, cy, angle_deg).translate(dx, dy)

        im = ax.imshow(
            img,
            extent=extent,
            origin="lower",
            cmap=cmaps[i],
            alpha=alpha,
            interpolation="bilinear",
        )
        im.set_transform(rot_tf + ax.transData)

        if show_boundaries:
            color = boundary_colors[i % len(boundary_colors)]
            rect = Rectangle(
                (ext_x0, ext_y0),
                ext_x1 - ext_x0,
                ext_y1 - ext_y0,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                label=f"Cam {cam}",
            )
            ax.add_patch(rect)
            rect.set_transform(rot_tf + ax.transData)

        # Collect rotated corners for manual axis limits
        corners = np.array([
            [ext_x0, ext_y0], [ext_x1, ext_y0],
            [ext_x1, ext_y1], [ext_x0, ext_y1],
        ])
        all_corners.append(rot_tf.transform(corners))

    if show_boundaries:
        ax.legend(loc="upper right", fontsize=8)

    if invert_ux:
        ax.invert_xaxis()

    # Manual axis limits (autoscale_view ignores Affine2D transforms on artists)
    all_pts = np.vstack(all_corners)
    pad = 5.0  # mm padding
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)


def build_overlay(
    dewarped_images: Dict[int, np.ndarray],
    bounds: Dict[int, Tuple[float, float, float, float]],
    shifts: Dict[int, Tuple[float, float]],
    config: dict,
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay all dewarped cameras on a single interactive figure."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    _draw_overlay_on_axes(ax, dewarped_images, bounds, shifts, config)
    return fig, ax


# ---------------------------------------------------------------------------
# Step 6: Per-camera diagnostic figures
# ---------------------------------------------------------------------------

def save_diagnostic_figures(
    raw_images: Dict[int, np.ndarray],
    dewarped_images: Dict[int, np.ndarray],
    bounds: Dict[int, Tuple[float, float, float, float]],
    shifts: Dict[int, Tuple[float, float]],
    config: dict,
) -> None:
    """Save a diagnostic PNG per camera: (a) raw, (b) dewarped, (c) raw vs dewarped overlay.

    Subplot (c) is a red-cyan composite in pixel space: raw in red, dewarped
    (resized to raw dims) in cyan. Grey = identical; colour fringing = distortion.

    Output: {base_dir}/dewarp_diagnostics/cam{N}_diagnostic.png
    """
    base_dir = Path(config["base_dir"])
    out_dir = base_dir / "dewarp_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_nums = sorted(dewarped_images.keys())

    for cam in cam_nums:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        raw = raw_images[cam]
        dw = dewarped_images[cam]

        # --- (a) Raw image ---
        ax_raw = axes[0]
        ax_raw.imshow(raw, cmap="gray", origin="upper")
        ax_raw.set_title(f"(a) Raw — Camera {cam}", fontsize=12)
        ax_raw.set_xlabel("px")
        ax_raw.set_ylabel("px")

        # --- (b) Dewarped image in world coords ---
        ax_dw = axes[1]
        x_min, x_max, y_min, y_max = bounds[cam]
        sx, sy = shifts.get(cam, (0.0, 0.0))
        ext_x0 = x_min + sx
        ext_x1 = x_max + sx
        ext_y0 = -(y_max + sy)
        ext_y1 = -(y_min + sy)
        ax_dw.imshow(
            np.flipud(dw),
            cmap="gray",
            extent=[ext_x0, ext_x1, ext_y0, ext_y1],
            origin="lower",
            interpolation="bilinear",
        )
        ax_dw.set_title(f"(b) Dewarped — Camera {cam}", fontsize=12)
        ax_dw.set_xlabel("x (mm)")
        ax_dw.set_ylabel("y (mm)")
        ax_dw.set_aspect("equal")

        # --- (c) Raw vs dewarped overlay (red-cyan composite in pixel space) ---
        ax_ov = axes[2]
        # Resize dewarped back to raw dimensions for direct comparison
        dw_resized = cv2.resize(dw, (raw.shape[1], raw.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # Normalise both to 0-255 uint8
        def _norm_u8(img):
            lo, hi = float(img.min()), float(img.max())
            if hi - lo < 1e-6:
                return np.zeros(img.shape, dtype=np.uint8)
            return ((img - lo) / (hi - lo) * 255).astype(np.uint8)

        raw_u8 = _norm_u8(raw)
        dw_u8 = _norm_u8(dw_resized)

        # Red-cyan: raw → red channel, dewarped → green+blue channels
        composite = np.stack([raw_u8, dw_u8, dw_u8], axis=-1)

        ax_ov.imshow(composite, origin="upper")
        ax_ov.set_title(
            f"(c) Distortion Overlay — Camera {cam}\n"
            f"Red = raw, Cyan = dewarped",
            fontsize=11,
        )
        ax_ov.set_xlabel("px")
        ax_ov.set_ylabel("px")

        fig.suptitle(f"Camera {cam} — Dewarp Diagnostic", fontsize=14, y=1.0)
        fig.tight_layout()

        out_path = out_dir / f"cam{cam}_diagnostic.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved diagnostic: {out_path}")

    logger.info(f"All diagnostics saved to {out_dir}")


# ---------------------------------------------------------------------------
# Step 7: Interactive coordinate readout and measurement
# ---------------------------------------------------------------------------

class InteractiveOverlay:
    """Matplotlib event handlers for coordinate readout, measurement, and zoom.

    Modes (toggle with 'm'):
      navigate: click -> print & mark (x_mm, y_mm)
      measure:  click two points -> distance, y-gap, angle
    Scroll wheel: zoom in/out centred on cursor.
    Press 'r' to reset zoom to full extent.
    Press 'c' to clear all markers and annotations.
    """

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.mode = "navigate"
        self.measure_points = []
        self.artists = []
        self._initial_xlim = ax.get_xlim()
        self._initial_ylim = ax.get_ylim()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._update_title()

    def _update_title(self):
        mode_str = "NAVIGATE" if self.mode == "navigate" else "MEASURE"
        self.ax.set_title(
            f"Mode: {mode_str}  |  'm' toggle  |  'c' clear  |  scroll zoom  |  'r' reset",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "m":
            self.mode = "measure" if self.mode == "navigate" else "navigate"
            self.measure_points.clear()
            self._update_title()
        elif event.key == "c":
            self._clear()
        elif event.key == "r":
            self._reset_zoom()

    def _clear(self):
        for a in self.artists:
            a.remove()
        self.artists.clear()
        self.measure_points.clear()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        """Scroll-wheel zoom centred on cursor position."""
        if event.inaxes != self.ax or event.xdata is None:
            return

        zoom_factor = 0.8 if event.button == "up" else 1.25
        cx, cy = event.xdata, event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_w = (xlim[1] - xlim[0]) * zoom_factor
        new_h = (ylim[1] - ylim[0]) * zoom_factor

        # Keep cursor position at the same relative location in the view
        rel_x = (cx - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (cy - ylim[0]) / (ylim[1] - ylim[0])

        self.ax.set_xlim(cx - rel_x * new_w, cx + (1 - rel_x) * new_w)
        self.ax.set_ylim(cy - rel_y * new_h, cy + (1 - rel_y) * new_h)
        self.fig.canvas.draw_idle()

    def _reset_zoom(self):
        """Reset to full extent (stored at init)."""
        self.ax.set_xlim(self._initial_xlim)
        self.ax.set_ylim(self._initial_ylim)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        x, y = event.xdata, event.ydata

        if self.mode == "navigate":
            self._handle_navigate(x, y)
        elif self.mode == "measure":
            self._handle_measure(x, y)

        self.fig.canvas.draw_idle()

    def _handle_navigate(self, x, y):
        (marker,) = self.ax.plot(x, y, "r+", markersize=12, markeredgewidth=2)
        txt = self.ax.annotate(
            f"({x:.2f}, {y:.2f}) mm",
            (x, y),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
            color="red",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        self.artists.extend([marker, txt])
        print(f"  ({x:.3f}, {y:.3f}) mm")

    def _handle_measure(self, x, y):
        (marker,) = self.ax.plot(x, y, "bo", markersize=6)
        self.artists.append(marker)
        self.measure_points.append((x, y))

        if len(self.measure_points) == 2:
            (x1, y1), (x2, y2) = self.measure_points
            dx = x2 - x1
            dy = y2 - y1
            dist = np.hypot(dx, dy)
            angle = np.degrees(np.arctan2(dy, dx))

            (line,) = self.ax.plot([x1, x2], [y1, y2], "b-", linewidth=1.5)
            txt = self.ax.annotate(
                f"d={dist:.2f} mm\n"
                f"|dy|={abs(dy):.2f} mm\n"
                f"angle={angle:.1f}\u00b0",
                ((x1 + x2) / 2, (y1 + y2) / 2),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=8,
                color="blue",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )
            self.artists.extend([line, txt])
            print(
                f"  Measure: d={dist:.3f} mm, |dy|={abs(dy):.3f} mm, "
                f"angle={angle:.1f} deg"
            )
            self.measure_points.clear()


# ---------------------------------------------------------------------------
# Step 7: Main orchestration
# ---------------------------------------------------------------------------

def main():
    config = resolve_config()
    base_dir = config["base_dir"]
    source_dir = config["source_dir"]
    cam_nums = config["camera_nums"]
    method = config["model_type"]
    mm_per_pixel = config["mm_per_pixel"]
    frame_index = config["frame_index"]
    image_format = config["image_format"]
    cam_folder_pattern = config.get("source_camera_folder", "Cam{cam}")

    logger.info(f"Cameras: {cam_nums}, method: {method}, frame: {frame_index}")

    # --- Load pinhole models ---
    cameras = {}
    model_data = {}
    for cam in cam_nums:
        camera, md = load_pinhole_camera(base_dir, cam, method)
        cameras[cam] = camera
        model_data[cam] = md
        logger.info(
            f"Cam {cam}: loaded model ({md['image_width']}x{md['image_height']})"
        )

    # --- Compute world bounds per camera ---
    all_bounds = {}
    for cam in cam_nums:
        md = model_data[cam]
        b = compute_camera_world_bounds(
            md["camera_matrix"], md["dist_coeffs"],
            md["rvec"], md["tvec"],
            md["image_width"], md["image_height"],
        )
        all_bounds[cam] = b
        logger.info(
            f"Cam {cam} bounds: x=[{b[0]:.1f}, {b[1]:.1f}], "
            f"y=[{b[2]:.1f}, {b[3]:.1f}] mm"
        )

    # --- Compute global alignment shifts ---
    shifts = compute_camera_shifts(cameras, model_data, config)

    # --- Dewarp each camera ---
    raw_images = {}
    dewarped = {}
    for cam in cam_nums:
        result = load_and_dewarp_camera(
            cameras[cam], cam, all_bounds[cam], mm_per_pixel,
            source_dir, image_format, frame_index, cam_folder_pattern,
        )
        if result is not None:
            raw_images[cam] = result[0]
            dewarped[cam] = result[1]

    if not dewarped:
        logger.error("No images were dewarped successfully — exiting")
        return

    # --- Save per-camera diagnostic PNGs ---
    if config.get("save_diagnostics", False):
        save_diagnostic_figures(raw_images, dewarped, all_bounds, shifts, config)

    # --- Build overlay and attach interactive handlers ---
    fig, ax = build_overlay(dewarped, all_bounds, shifts, config)
    _overlay = InteractiveOverlay(fig, ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
