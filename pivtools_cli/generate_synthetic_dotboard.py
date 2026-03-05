#!/usr/bin/env python3
"""
Generate synthetic dotboard calibration images with known ground-truth
camera parameters for calibration recovery tests.

Usage:
    python pivtools_cli/generate_synthetic_dotboard.py
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pivtools_cli.synthetic_calibration_common import (
    make_camera_matrix,
    make_poses,
    save_ground_truth,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "unit-tests" / "synthetic_calibration" / "dotboard"
N_VIEWS = 10
MEGAPIXELS = 1.0
DOT_COLS = 15
DOT_ROWS = 12
DOT_SPACING_MM = 15.0
DOT_RADIUS_RATIO = 0.22      # radius / spacing


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_dotboard_images(
    out_dir: Path,
    camera_matrix: np.ndarray,
    poses: list,
    target_w: int,
    target_h: int,
    *,
    cols: int = DOT_COLS,
    rows: int = DOT_ROWS,
    spacing_mm: float = DOT_SPACING_MM,
    radius_ratio: float = DOT_RADIUS_RATIO,
):
    """Render each dotboard view by drawing filled circles at projected positions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_pts = np.zeros((rows * cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            obj_pts[idx, 0] = c * spacing_mm / 1000.0
            obj_pts[idx, 1] = r * spacing_mm / 1000.0

    dist_coeffs = np.zeros(5)

    for i, (rvec, tvec) in enumerate(poses):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        centres_px = proj.reshape(-1, 2)

        tree = cKDTree(centres_px)
        dists, _ = tree.query(centres_px, k=2)
        spacing_px = np.median(dists[:, 1])
        radius_px = max(int(round(radius_ratio * spacing_px)), 2)

        img = np.zeros((target_h, target_w), dtype=np.uint8)
        for cx, cy in centres_px:
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < target_w and 0 <= iy < target_h:
                cv2.circle(img, (ix, iy), radius_px, 255, -1)

        fname = out_dir / f"calib{i + 1:05d}.png"
        cv2.imwrite(str(fname), img)
    print(f"  Wrote {len(poses)} dotboard images to {out_dir}")


def generate_dotboard_dataset(
    output_dir: Path = OUTPUT_DIR,
    n_views: int = N_VIEWS,
    megapixels: float = MEGAPIXELS,
    *,
    cols: int = DOT_COLS,
    rows: int = DOT_ROWS,
    spacing_mm: float = DOT_SPACING_MM,
    radius_ratio: float = DOT_RADIUS_RATIO,
):
    """Orchestrator: create camera + poses, generate images, save ground truth."""
    total_px = int(megapixels * 1e6)
    W = H = int(total_px ** 0.5)
    print(f"Dotboard target image: {W}x{H}  ({W * H / 1e6:.2f} MP)")

    cam_mtx = make_camera_matrix(W, H)
    fx = cam_mtx[0, 0]

    board_w = (cols - 1) * spacing_mm / 1000.0
    board_h = (rows - 1) * spacing_mm / 1000.0
    board_centre = np.array([board_w / 2, board_h / 2, 0])

    poses = make_poses(
        n_views, board_centre, fx, W, board_w,
        fill_fraction_range=(0.80, 0.55),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_dotboard_images(
        output_dir, cam_mtx, poses, W, H,
        cols=cols, rows=rows, spacing_mm=spacing_mm, radius_ratio=radius_ratio,
    )

    gt_path = output_dir.parent / "ground_truth.npz"
    save_ground_truth(
        gt_path, cam_mtx, poses,
        image_width=W, image_height=H,
    )
    print(f"Dotboard ground truth saved to {gt_path}")
    print(f"  fx={cam_mtx[0,0]:.1f}  fy={cam_mtx[1,1]:.1f}  "
          f"cx={cam_mtx[0,2]:.1f}  cy={cam_mtx[1,2]:.1f}")


if __name__ == "__main__":
    generate_dotboard_dataset()
