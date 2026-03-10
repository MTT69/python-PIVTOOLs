#!/usr/bin/env python3
"""
Generate synthetic stereo calibration images with known ground-truth
camera and inter-camera parameters for stereo calibration recovery tests.

Usage:
    python pivtools_cli/generate_synthetic_stereo.py
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pivtools_cli.synthetic_calibration_common import (
    make_camera_matrix,
    make_poses,
    warp_board,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_BASE = Path(__file__).resolve().parent.parent / "unit-tests" / "synthetic_calibration"
N_VIEWS = 10
MEGAPIXELS = 1.0

# Dotboard params (match generate_synthetic_dotboard.py)
DOT_COLS = 15
DOT_ROWS = 12
DOT_SPACING_MM = 15.0
DOT_RADIUS_RATIO = 0.22

# ChArUco params (match generate_synthetic_charuco.py)
SQUARES_H = 10
SQUARES_V = 7
SQUARE_SIZE = 0.030       # metres
MARKER_RATIO = 0.5
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_1000
OVERSAMPLE = 4

# Stereo baseline
STEREO_ROTATION_DEG = 5.0   # rotation around Y axis
STEREO_BASELINE_MM = 50.0   # horizontal translation


# ---------------------------------------------------------------------------
# Stereo geometry
# ---------------------------------------------------------------------------

def make_stereo_transform():
    """Create known inter-camera rotation and translation."""
    angle_rad = np.radians(STEREO_ROTATION_DEG)
    rvec_stereo = np.array([0, angle_rad, 0], dtype=np.float64)
    R_stereo, _ = cv2.Rodrigues(rvec_stereo)
    T_stereo = np.array(
        [[STEREO_BASELINE_MM / 1000.0], [0.0], [0.0]], dtype=np.float64,
    )
    return R_stereo, T_stereo


def compose_stereo_poses(poses_cam1, R_stereo, T_stereo):
    """Compute camera 2 poses: R2 = R_stereo @ R1, t2 = R_stereo @ t1 + T_stereo."""
    poses_cam2 = []
    for rvec1, tvec1 in poses_cam1:
        R1, _ = cv2.Rodrigues(rvec1)
        R2 = R_stereo @ R1
        t2 = R_stereo @ tvec1.reshape(3, 1) + T_stereo
        rvec2, _ = cv2.Rodrigues(R2)
        poses_cam2.append((rvec2.flatten(), t2.flatten()))
    return poses_cam2


# ---------------------------------------------------------------------------
# Rendering (reuses same logic as planar generators)
# ---------------------------------------------------------------------------

def generate_dotboard_images(out_dir, camera_matrix, poses, W, H):
    """Render dotboard views with adaptive radius."""
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_pts = np.zeros((DOT_ROWS * DOT_COLS, 3), dtype=np.float64)
    for r in range(DOT_ROWS):
        for c in range(DOT_COLS):
            idx = r * DOT_COLS + c
            obj_pts[idx, 0] = c * DOT_SPACING_MM / 1000.0
            obj_pts[idx, 1] = r * DOT_SPACING_MM / 1000.0

    dist_coeffs = np.zeros(5)
    for i, (rvec, tvec) in enumerate(poses):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        centres_px = proj.reshape(-1, 2)

        tree = cKDTree(centres_px)
        dists, _ = tree.query(centres_px, k=2)
        local_radii = np.maximum(
            np.round(DOT_RADIUS_RATIO * dists[:, 1]).astype(int), 2,
        )

        img = np.zeros((H, W), dtype=np.uint8)
        for idx_dot, (cx, cy) in enumerate(centres_px):
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < W and 0 <= iy < H:
                cv2.circle(img, (ix, iy), int(local_radii[idx_dot]), 255, -1)

        cv2.imwrite(str(out_dir / f"calib{i + 1:05d}.png"), img)
    print(f"  Wrote {len(poses)} dotboard images to {out_dir}")


def generate_charuco_images(out_dir, camera_matrix, poses, W, H):
    """Render charuco views with oversample + perspective warp."""
    out_dir.mkdir(parents=True, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_H, SQUARES_V), SQUARE_SIZE, SQUARE_SIZE * MARKER_RATIO, aruco_dict,
    )

    hi_w, hi_h = W * OVERSAMPLE, H * OVERSAMPLE
    frontal = board.generateImage((hi_w, hi_h))

    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters(),
    )
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(frontal)
    if charuco_corners is None or len(charuco_corners) == 0:
        raise RuntimeError("Failed to detect corners in hi-res frontal image")

    id_to_frontal_px = {}
    for corner, cid in zip(charuco_corners, charuco_ids.flatten()):
        id_to_frontal_px[int(cid)] = corner.flatten()

    obj_pts_all = board.getChessboardCorners()
    dist_coeffs = np.zeros(5)

    for i, (rvec, tvec) in enumerate(poses):
        proj_all, _ = cv2.projectPoints(
            obj_pts_all, rvec, tvec, camera_matrix, dist_coeffs,
        )
        proj_all = proj_all.reshape(-1, 2)

        src_list, dst_list = [], []
        for cid, frontal_px in id_to_frontal_px.items():
            src_list.append(frontal_px)
            dst_list.append(proj_all[cid])

        src_px = np.array(src_list, dtype=np.float32)
        dst_px = np.array(dst_list, dtype=np.float32)

        warped_hi = warp_board(frontal, src_px, dst_px * OVERSAMPLE, (hi_w, hi_h))
        warped = cv2.resize(warped_hi, (W, H), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(out_dir / f"calib{i + 1:05d}.png"), warped)
    print(f"  Wrote {len(poses)} charuco images to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_px = int(MEGAPIXELS * 1e6)
    W = H = int(total_px ** 0.5)
    print(f"Stereo target image: {W}x{H}  ({W * H / 1e6:.2f} MP)")

    cam_mtx = make_camera_matrix(W, H)
    fx = cam_mtx[0, 0]
    R_stereo, T_stereo = make_stereo_transform()

    # --- Dotboard ---
    board_w = (DOT_COLS - 1) * DOT_SPACING_MM / 1000.0
    board_centre_dot = np.array([board_w / 2, (DOT_ROWS - 1) * DOT_SPACING_MM / 2000.0, 0])

    poses_cam1_dot = make_poses(
        N_VIEWS, board_centre_dot, fx, W, board_w,
        fill_fraction_range=(0.80, 0.55),
    )
    poses_cam2_dot = compose_stereo_poses(poses_cam1_dot, R_stereo, T_stereo)

    generate_dotboard_images(OUTPUT_BASE / "stereo_dotboard" / "cam1", cam_mtx, poses_cam1_dot, W, H)
    generate_dotboard_images(OUTPUT_BASE / "stereo_dotboard" / "cam2", cam_mtx, poses_cam2_dot, W, H)

    # --- ChArUco ---
    charuco_board_w = SQUARES_H * SQUARE_SIZE
    charuco_centre = np.array([charuco_board_w / 2, SQUARES_V * SQUARE_SIZE / 2, 0])

    poses_cam1_char = make_poses(
        N_VIEWS, charuco_centre, fx, W, charuco_board_w,
        fill_fraction_range=(0.85, 0.65),
    )
    poses_cam2_char = compose_stereo_poses(poses_cam1_char, R_stereo, T_stereo)

    generate_charuco_images(OUTPUT_BASE / "stereo_charuco" / "cam1", cam_mtx, poses_cam1_char, W, H)
    generate_charuco_images(OUTPUT_BASE / "stereo_charuco" / "cam2", cam_mtx, poses_cam2_char, W, H)

    # --- Save ground truth ---
    gt_path = OUTPUT_BASE / "stereo_ground_truth.npz"
    np.savez(
        str(gt_path),
        camera_matrix=cam_mtx,
        R_stereo=R_stereo,
        T_stereo=T_stereo,
        rvecs_cam1=np.array([p[0] for p in poses_cam1_dot]),
        tvecs_cam1=np.array([p[1] for p in poses_cam1_dot]),
        rvecs_cam2=np.array([p[0] for p in poses_cam2_dot]),
        tvecs_cam2=np.array([p[1] for p in poses_cam2_dot]),
        image_width=W,
        image_height=H,
    )

    print(f"\nStereo ground truth saved to {gt_path}")
    print(f"  R_stereo angle: {STEREO_ROTATION_DEG:.1f} deg")
    T_flat = T_stereo.flatten()
    print(f"  T_stereo: [{T_flat[0]*1000:.1f}, {T_flat[1]:.1f}, {T_flat[2]:.1f}] mm")
    print(f"  ||T||: {np.linalg.norm(T_stereo)*1000:.1f} mm")


if __name__ == "__main__":
    main()
