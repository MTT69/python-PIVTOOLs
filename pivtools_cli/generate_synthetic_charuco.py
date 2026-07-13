#!/usr/bin/env python3
"""
Generate synthetic ChArUco calibration board images with known ground-truth
camera parameters for calibration recovery tests.

Usage:
    python pivtools_cli/generate_synthetic_charuco.py
"""

from pathlib import Path

import cv2
import numpy as np

from pivtools_cli.synthetic_calibration_common import (
    make_camera_matrix,
    make_poses,
    save_ground_truth,
    warp_board,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "unit-tests"
    / "synthetic_calibration"
    / "charuco"
)
N_VIEWS = 10
MEGAPIXELS = 1.0
SQUARES_H = 10
SQUARES_V = 7
SQUARE_SIZE = 0.030  # metres
MARKER_RATIO = 0.5
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_1000
OVERSAMPLE = 4


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_charuco_images(
    out_dir: Path,
    camera_matrix: np.ndarray,
    poses: list,
    target_w: int,
    target_h: int,
    *,
    sq_h: int = SQUARES_H,
    sq_v: int = SQUARES_V,
    sq_size: float = SQUARE_SIZE,
    marker_ratio: float = MARKER_RATIO,
    dict_id: int = ARUCO_DICT_ID,
    oversample: int = OVERSAMPLE,
):
    """Generate warped ChArUco board images for each pose."""
    out_dir.mkdir(parents=True, exist_ok=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (sq_h, sq_v),
        sq_size,
        sq_size * marker_ratio,
        aruco_dict,
    )

    hi_w, hi_h = target_w * oversample, target_h * oversample
    frontal = board.generateImage((hi_w, hi_h))

    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        cv2.aruco.DetectorParameters(),
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
            obj_pts_all,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        proj_all = proj_all.reshape(-1, 2)

        src_list, dst_list = [], []
        for cid, frontal_px in id_to_frontal_px.items():
            src_list.append(frontal_px)
            dst_list.append(proj_all[cid])

        src_px = np.array(src_list, dtype=np.float32)
        dst_px = np.array(dst_list, dtype=np.float32)

        warped_hi = warp_board(frontal, src_px, dst_px * oversample, (hi_w, hi_h))
        warped = cv2.resize(
            warped_hi, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

        fname = out_dir / f"calib{i + 1:05d}.png"
        cv2.imwrite(str(fname), warped)
    print(f"  Wrote {len(poses)} charuco images to {out_dir}")


def generate_charuco_dataset(
    output_dir: Path = OUTPUT_DIR,
    n_views: int = N_VIEWS,
    megapixels: float = MEGAPIXELS,
    *,
    sq_h: int = SQUARES_H,
    sq_v: int = SQUARES_V,
    sq_size: float = SQUARE_SIZE,
    marker_ratio: float = MARKER_RATIO,
    dict_id: int = ARUCO_DICT_ID,
    oversample: int = OVERSAMPLE,
):
    """Orchestrator: create camera + poses, generate images, save ground truth."""
    total_px = int(megapixels * 1e6)
    W = H = int(total_px**0.5)
    print(f"ChArUco target image: {W}x{H}  ({W * H / 1e6:.2f} MP)")

    cam_mtx = make_camera_matrix(W, H)
    fx = cam_mtx[0, 0]

    board_w = sq_h * sq_size
    board_h = sq_v * sq_size
    board_centre = np.array([board_w / 2, board_h / 2, 0])

    poses = make_poses(
        n_views,
        board_centre,
        fx,
        W,
        board_w,
        fill_fraction_range=(0.85, 0.65),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_charuco_images(
        output_dir,
        cam_mtx,
        poses,
        W,
        H,
        sq_h=sq_h,
        sq_v=sq_v,
        sq_size=sq_size,
        marker_ratio=marker_ratio,
        dict_id=dict_id,
        oversample=oversample,
    )

    gt_path = output_dir.parent / "ground_truth.npz"
    save_ground_truth(
        gt_path,
        cam_mtx,
        poses,
        image_width=W,
        image_height=H,
    )
    print(f"ChArUco ground truth saved to {gt_path}")
    print(
        f"  fx={cam_mtx[0,0]:.1f}  fy={cam_mtx[1,1]:.1f}  "
        f"cx={cam_mtx[0,2]:.1f}  cy={cam_mtx[1,2]:.1f}"
    )


if __name__ == "__main__":
    generate_charuco_dataset()
