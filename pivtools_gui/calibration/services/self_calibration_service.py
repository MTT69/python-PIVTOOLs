"""
self_calibration_service.py

Service layer for stereo self-calibration. Used by both GUI routes and CLI.
Handles camera loading, image loading, dewarp overlay generation, and
orchestrating the self-calibration pipeline.
"""

import logging
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from pivtools_gui.calibration.camera_model_utils import (
    compute_camera_world_bounds,
    load_cameras_from_stereo_model,
)
from pivtools_gui.stereo_reconstruction.self_calibration import (
    PinholeCamera,
    SelfCalibrationResult,
    compute_dewarp_maps,
    dewarp_image,
    estimate_pixel_scale,
    run_self_calibration,
)

logger = logging.getLogger(__name__)


def load_stereo_cameras(
    base_dir: str, cam1_num: int, cam2_num: int, method: str
) -> Tuple[PinholeCamera, PinholeCamera, dict, dict]:
    """Load PinholeCamera objects from the stereo model.

    Returns (cam1, cam2, model_data1, model_data2).
    """
    return load_cameras_from_stereo_model(base_dir, cam1_num, cam2_num)


def compute_stereo_world_bounds(
    md1: dict, md2: dict
) -> Tuple[float, float, float, float]:
    """Intersection of both cameras' FOV.

    Returns (xmin, xmax, ymin, ymax) in mm.
    """
    b1 = compute_camera_world_bounds(
        md1["camera_matrix"], md1["dist_coeffs"],
        md1["rvec"], md1["tvec"],
        md1["image_width"], md1["image_height"],
    )
    b2 = compute_camera_world_bounds(
        md2["camera_matrix"], md2["dist_coeffs"],
        md2["rvec"], md2["tvec"],
        md2["image_width"], md2["image_height"],
    )
    # Intersection
    xmin = max(b1[0], b2[0])
    xmax = min(b1[1], b2[1])
    ymin = max(b1[2], b2[2])
    ymax = min(b1[3], b2[3])

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            f"Cameras have no overlapping FOV. "
            f"Cam1 bounds: x=[{b1[0]:.1f},{b1[1]:.1f}], y=[{b1[2]:.1f},{b1[3]:.1f}]. "
            f"Cam2 bounds: x=[{b2[0]:.1f},{b2[1]:.1f}], y=[{b2[2]:.1f},{b2[3]:.1f}]."
        )
    return xmin, xmax, ymin, ymax


def load_source_images(
    config, source_path_idx: int, cam1_num: int, cam2_num: int, n_images: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load N source images (frame A only) for self-cal.

    Evenly samples across available frames if n_images < total.
    Uses read_pair() from pivtools_core.image_handling.load_images.
    """
    from pivtools_core.image_handling.load_images import read_pair
    from pivtools_core.image_handling.path_utils import build_piv_camera_path

    total = config.num_frame_pairs
    if n_images >= total:
        indices = list(range(1, total + 1))
    else:
        # Evenly sample
        step = total / n_images
        indices = [int(round(1 + i * step)) for i in range(n_images)]
        indices = [min(i, total) for i in indices]

    cam1_path = build_piv_camera_path(config, source_path_idx, cam1_num)
    cam2_path = build_piv_camera_path(config, source_path_idx, cam2_num)
    images_cam1 = []
    images_cam2 = []

    for idx in indices:
        try:
            # Load both cameras before appending to keep lists in sync
            pair1 = read_pair(idx, cam1_path, cam1_num, config)
            pair2 = read_pair(idx, cam2_path, cam2_num, config)

            img1 = pair1[0]  # Frame A only
            if img1.ndim == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            img2 = pair2[0]
            if img2.ndim == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            images_cam1.append(img1)
            images_cam2.append(img2)
        except Exception as e:
            logger.warning(f"Failed to load frame {idx}: {e}")
            continue

    if not images_cam1:
        raise ValueError("No source images could be loaded")

    logger.info(f"Loaded {len(images_cam1)} source image pairs from {cam1_path} / {cam2_path}")
    return images_cam1, images_cam2


def generate_dewarp_overlay(
    cam1: PinholeCamera,
    cam2: PinholeCamera,
    img1: np.ndarray,
    img2: np.ndarray,
    world_bounds: Tuple[float, float, float, float],
    mm_per_pixel: float,
    z_offset: float = 0.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dewarp and overlay: cam1=red, cam2=cyan.

    Returns (overlay_rgb, cam1_normalized_u8, cam2_normalized_u8).
    """
    map1_x, map1_y = compute_dewarp_maps(
        cam1, world_bounds, mm_per_pixel,
        z_offset=z_offset, tilt_x=tilt_x, tilt_y=tilt_y,
    )
    map2_x, map2_y = compute_dewarp_maps(
        cam2, world_bounds, mm_per_pixel,
        z_offset=z_offset, tilt_x=tilt_x, tilt_y=tilt_y,
    )

    dw1 = dewarp_image(img1, map1_x, map1_y)
    dw2 = dewarp_image(img2, map2_x, map2_y)

    # Normalize to 0-255
    def _norm_u8(img):
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-6:
            return np.zeros(img.shape, dtype=np.uint8)
        return ((img - lo) / (hi - lo) * 255).astype(np.uint8)

    r = _norm_u8(dw1)
    c = _norm_u8(dw2)

    # Red-cyan: cam1 → red, cam2 → green+blue
    overlay = np.stack([r, c, c], axis=-1)
    return overlay, r, c


def run_self_cal_job(
    config,
    base_dir: str,
    source_path_idx: int,
    cam1_num: int,
    cam2_num: int,
    method: str,
    n_images: int = 20,
    window_size: int = 64,
    overlap: float = 50.0,
    convergence_threshold: float = 0.1,
    quality_threshold: float = 0.3,
    progress_callback: Optional[Callable] = None,
) -> SelfCalibrationResult:
    """Full pipeline: load cameras, load images, run self-calibration, return result."""
    if progress_callback:
        progress_callback({"status": "loading_cameras", "progress": 5})

    cam1, cam2, md1, md2 = load_stereo_cameras(base_dir, cam1_num, cam2_num, method)

    if progress_callback:
        progress_callback({"status": "computing_bounds", "progress": 10})

    world_bounds = compute_stereo_world_bounds(md1, md2)
    logger.info(
        f"Stereo FOV intersection: x=[{world_bounds[0]:.1f},{world_bounds[1]:.1f}], "
        f"y=[{world_bounds[2]:.1f},{world_bounds[3]:.1f}] mm"
    )

    if progress_callback:
        progress_callback({"status": "loading_images", "progress": 15})

    images_cam1, images_cam2 = load_source_images(
        config, source_path_idx, cam1_num, cam2_num, n_images
    )

    if progress_callback:
        progress_callback({"status": "running_self_calibration", "progress": 25})

    result = run_self_calibration(
        cam1, cam2,
        images_cam1, images_cam2,
        world_bounds=world_bounds,
        window_size=window_size,
        overlap=overlap,
        convergence_threshold=convergence_threshold,
        quality_threshold=quality_threshold,
    )

    if progress_callback:
        progress_callback({"status": "complete", "progress": 100})

    return result


def save_self_cal_to_config(config, result: SelfCalibrationResult, **params):
    """Save self-calibration results to config.yaml."""
    sc_data = {
        "z_offset": float(result.z_offset),
        "tilt_x": float(result.tilt_x),
        "tilt_y": float(result.tilt_y),
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "final_rms_disparity": float(result.final_rms_disparity),
    }
    # Merge with any user-set parameters
    sc_data.update(params)

    if "calibration" not in config.data:
        config.data["calibration"] = {}
    config.data["calibration"]["self_calibration"] = sc_data
    config.save()
    logger.info(
        f"Saved self-cal to config: z={result.z_offset:.4f} mm, "
        f"tilt_x={math.degrees(result.tilt_x):.4f}°, "
        f"tilt_y={math.degrees(result.tilt_y):.4f}°"
    )


def result_to_dict(result: SelfCalibrationResult) -> dict:
    """Convert SelfCalibrationResult to JSON-serializable dict."""
    # Prepend iteration 0: baseline RMS before any corrections were applied
    initial_rms = result.history[0].rms_disparity if result.history else 0.0
    history = [
        {
            "iteration": 0,
            "rms_disparity": initial_rms,
            "delta_z": 0.0,
            "delta_tilt_x": 0.0,
            "delta_tilt_y": 0.0,
            "cumulative_z": 0.0,
            "cumulative_tilt_x": 0.0,
            "cumulative_tilt_y": 0.0,
        },
    ] + [
        {
            "iteration": rec.iteration,
            "rms_disparity": rec.rms_disparity,
            "delta_z": rec.delta_z,
            "delta_tilt_x": rec.delta_tilt_x,
            "delta_tilt_y": rec.delta_tilt_y,
            "cumulative_z": rec.cumulative_z,
            "cumulative_tilt_x": rec.cumulative_tilt_x,
            "cumulative_tilt_y": rec.cumulative_tilt_y,
        }
        for rec in result.history
    ]
    return {
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "z_offset": float(result.z_offset),
        "tilt_x": float(result.tilt_x),
        "tilt_y": float(result.tilt_y),
        "tilt_x_deg": float(math.degrees(result.tilt_x)),
        "tilt_y_deg": float(math.degrees(result.tilt_y)),
        "final_rms_disparity": float(result.final_rms_disparity),
        "history": history,
    }
