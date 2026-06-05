"""
Shared utilities for synthetic calibration board generation.

Used by generate_synthetic_charuco.py and generate_synthetic_dotboard.py.
"""

from pathlib import Path

import cv2
import numpy as np


def make_camera_matrix(W: int, H: int) -> np.ndarray:
    """Camera with fx=fy=W (~60 deg FOV), principal point at image centre."""
    fx = fy = float(W)
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def make_poses(
    n_views: int,
    board_centre_3d: np.ndarray,
    fx: float,
    image_w: int,
    board_width_3d: float,
    fill_fraction_range: tuple = (0.75, 0.75),
    seed: int = 42,
) -> list:
    """
    Deterministic poses.  View 0 is frontal; views 1-9 have progressive tilts.

    The board is centred in the image by offsetting tvec so that board_centre_3d
    projects to the image centre.  tz is chosen so the board fills a fraction
    of the image width controlled by *fill_fraction_range*.

    fill_fraction_range = (high, low):
      - View 0 (frontal) uses *high* -- board nearly fills the frame.
      - Tilted views interpolate from *high* down to *low* based on tilt magnitude.
      This ensures edge coverage in near-frontal views while preventing dot merging
      in heavily tilted views.

    Returns list of (rvec, tvec) tuples.
    """
    rng = np.random.default_rng(seed)

    fill_high, fill_low = fill_fraction_range

    tx_centre = -board_centre_3d[0]
    ty_centre = -board_centre_3d[1]

    max_tilt = 0.25

    poses = []
    for i in range(n_views):
        if i == 0:
            rvec = np.zeros(3, dtype=np.float64)
            fill = fill_high
        else:
            rx = rng.uniform(-max_tilt, max_tilt)
            ry = rng.uniform(-max_tilt, max_tilt)
            rz = rng.uniform(-0.10, 0.10)
            rvec = np.array([rx, ry, rz], dtype=np.float64)
            tilt_mag = np.sqrt(rx**2 + ry**2)
            t = min(tilt_mag / max_tilt, 1.0)
            fill = fill_high + t * (fill_low - fill_high)

        tz = fx * board_width_3d / (fill * image_w) + rng.uniform(-0.02, 0.02)
        tx = tx_centre + rng.uniform(-0.005, 0.005)
        ty = ty_centre + rng.uniform(-0.005, 0.005)
        tvec = np.array([tx, ty, tz], dtype=np.float64)
        poses.append((rvec, tvec))
    return poses


def warp_board(
    frontal_img: np.ndarray,
    src_corners_px: np.ndarray,
    dst_corners_px: np.ndarray,
    out_size: tuple,
) -> np.ndarray:
    """
    Warp *frontal_img* so that *src_corners_px* map to *dst_corners_px*.

    Uses a least-squares homography from ALL corners (not just 4).
    """
    H, _ = cv2.findHomography(
        src_corners_px.astype(np.float32),
        dst_corners_px.astype(np.float32),
        method=0,
    )
    if H is None:
        raise RuntimeError("findHomography returned None")
    return cv2.warpPerspective(
        frontal_img, H, out_size,
        flags=cv2.INTER_LINEAR,
        borderValue=200,
    )


def save_ground_truth(out_path: Path, camera_matrix: np.ndarray, poses: list, **extra_arrays):
    """Save ground-truth camera parameters and poses to a .npz file."""
    rvecs = np.array([p[0] for p in poses])
    tvecs = np.array([p[1] for p in poses])
    np.savez(
        str(out_path),
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvecs=rvecs,
        tvecs=tvecs,
        **extra_arrays,
    )
