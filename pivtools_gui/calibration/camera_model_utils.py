"""
camera_model_utils.py

Shared utilities for loading pinhole camera models and computing world-coordinate
bounding boxes. Extracted from image_dewarp_overlay.py for reuse by the
self-calibration service and other modules.
"""

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.io import loadmat

from pivtools_core.paths import get_data_paths
from pivtools_gui.stereo_reconstruction.self_calibration import PinholeCamera
from pivtools_gui.calibration.global_coordinate_alignment import _pixels_to_world_mm

logger = logging.getLogger(__name__)


def load_cameras_from_stereo_model(
    base_dir: str, cam1_num: int, cam2_num: int
) -> Tuple[PinholeCamera, PinholeCamera, dict, dict]:
    """Load both PinholeCameras from a stereo model .mat file.

    Cam1 uses the first calibration view's extrinsics as the world frame.
    Cam2 is derived via the stereo rotation/translation so both cameras
    share a consistent coordinate system.

    Path: {base_dir}/calibration/stereo_cam{A}_cam{B}/model/stereo_model.mat

    Returns (cam1, cam2, model_dict1, model_dict2).
    """
    base = Path(base_dir)
    model_path = (
        base / "calibration" / f"stereo_cam{cam1_num}_cam{cam2_num}"
        / "model" / "stereo_model.mat"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Stereo model not found: {model_path}. Run stereo calibration first."
        )

    mat = loadmat(str(model_path), squeeze_me=True, struct_as_record=False)

    image_size_arr = np.array(mat["image_size"]).flatten()
    w, h = int(image_size_arr[0]), int(image_size_arr[1])

    K1 = np.array(mat["camera_matrix_1"]).astype(np.float64)
    K2 = np.array(mat["camera_matrix_2"]).astype(np.float64)
    dist1 = np.array(mat["dist_coeffs_1"]).flatten().astype(np.float64)
    dist2 = np.array(mat["dist_coeffs_2"]).flatten().astype(np.float64)

    # Cam1: use datum calibration view's extrinsics as the world frame
    rvecs1 = np.array(mat["rvecs_1"]).astype(np.float64)
    tvecs1 = np.array(mat["tvecs_1"]).astype(np.float64)

    datum_frame = int(mat["datum_frame"]) if "datum_frame" in mat else 0
    if rvecs1.ndim == 1:
        rvec1 = rvecs1.astype(np.float64)
        tvec1 = tvecs1.astype(np.float64)
    else:
        idx = max(0, datum_frame - 1) if datum_frame > 0 else 0
        rvec1 = rvecs1[idx].flatten()
        tvec1 = tvecs1[idx].flatten()

    R1, _ = cv2.Rodrigues(rvec1)
    t1 = tvec1.reshape(3, 1)

    # Cam2: derive from stereo R/T so both cameras share a consistent
    # coordinate system.  The stereo R/T from cv2.stereoCalibrate relates
    # cam1 → cam2:  X_cam2 = R_stereo @ X_cam1 + T_stereo.
    # Combined with cam1 world-to-camera:  X_cam1 = R1 @ X_world + t1
    # gives:  X_cam2 = (R_stereo @ R1) @ X_world + (R_stereo @ t1 + T_stereo)
    R_stereo = np.array(mat["rotation_matrix"]).astype(np.float64)
    T_stereo = np.array(mat["translation_vector"]).astype(np.float64).reshape(3, 1)
    R2 = R_stereo @ R1
    t2 = R_stereo @ t1 + T_stereo
    rvec2, _ = cv2.Rodrigues(R2)
    rvec2 = rvec2.flatten()
    tvec2 = t2.flatten()

    cam1 = PinholeCamera(K=K1, dist=dist1, R=R1, t=t1, image_size=(w, h))
    cam2 = PinholeCamera(K=K2, dist=dist2, R=R2, t=t2, image_size=(w, h))

    md1 = {
        "camera_matrix": K1, "dist_coeffs": dist1,
        "rvec": rvec1, "tvec": tvec1.flatten(),
        "image_width": w, "image_height": h,
    }
    md2 = {
        "camera_matrix": K2, "dist_coeffs": dist2,
        "rvec": rvec2, "tvec": tvec2,
        "image_width": w, "image_height": h,
    }
    return cam1, cam2, md1, md2


def load_pinhole_camera(
    base_dir: str, cam_num: int, method: str
) -> Tuple[PinholeCamera, dict]:
    """Load a .mat pinhole model and return (PinholeCamera, raw_model_dict).

    Model path: {base_dir}/calibration/Cam{N}/{method}_planar/model/{file}.mat
    Uses datum_frame stored in the model (if present) for rvec/tvec selection.
    """
    calib_paths = get_data_paths(
        base_dir, num_frame_pairs=1, cam=cam_num, type_name="", calibration=True
    )
    calib_dir = calib_paths["calib_dir"]

    if method == "charuco":
        model_path = calib_dir / "charuco_planar" / "model" / "camera_model.mat"
    elif method == "stepped_board":
        model_path = calib_dir / "stepped_board" / "model" / "camera_model.mat"
    else:
        model_path = calib_dir / "dotboard_planar" / "model" / "dotboard_model.mat"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Calibration model not found: {model_path}. Run 'Generate Model' first."
        )

    mat = loadmat(str(model_path), squeeze_me=True, struct_as_record=False)

    K = np.array(mat["camera_matrix"]).astype(np.float64)
    dist = np.array(mat["dist_coeffs"]).flatten().astype(np.float64)
    rvecs = mat["rvecs"]
    tvecs = mat["tvecs"]

    # Select rvec/tvec using datum_frame from model (1-based), fallback to index 0
    datum_frame = int(mat["datum_frame"]) if "datum_frame" in mat else 0
    if rvecs.ndim == 1:
        rvec = rvecs.astype(np.float64)
        tvec = tvecs.astype(np.float64)
    else:
        idx = max(0, datum_frame - 1) if datum_frame > 0 else 0
        rvec = rvecs[idx].flatten().astype(np.float64)
        tvec = tvecs[idx].flatten().astype(np.float64)

    R, _ = cv2.Rodrigues(rvec)
    w = int(mat["image_width"])
    h = int(mat["image_height"])

    camera = PinholeCamera(
        K=K, dist=dist, R=R, t=tvec.reshape(3, 1), image_size=(w, h)
    )

    model_dict = {
        "camera_matrix": K,
        "dist_coeffs": dist,
        "rvec": rvec,
        "tvec": tvec,
        "image_width": w,
        "image_height": h,
    }
    return camera, model_dict


def compute_camera_world_bounds(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    """Sample ~80 points along all 4 image edges, project to Z=0 world plane.

    Returns (x_min, x_max, y_min, y_max) in mm.
    """
    n = 20  # points per edge
    top = np.column_stack([np.linspace(0, w - 1, n), np.zeros(n)])
    bottom = np.column_stack([np.linspace(0, w - 1, n), np.full(n, h - 1)])
    left = np.column_stack([np.zeros(n), np.linspace(0, h - 1, n)])
    right = np.column_stack([np.full(n, w - 1), np.linspace(0, h - 1, n)])
    pts_px = np.vstack([top, bottom, left, right]).astype(np.float32)

    world_pts = _pixels_to_world_mm(pts_px, camera_matrix, dist_coeffs, rvec, tvec)
    valid = ~np.isnan(world_pts).any(axis=1)
    world_pts = world_pts[valid]

    if world_pts.size == 0:
        raise ValueError(
            "All edge projections returned NaN — check camera model parameters"
        )

    return (
        float(world_pts[:, 0].min()),
        float(world_pts[:, 0].max()),
        float(world_pts[:, 1].min()),
        float(world_pts[:, 1].max()),
    )
