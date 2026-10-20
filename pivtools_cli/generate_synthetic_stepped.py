#!/usr/bin/env python3
"""
Generate synthetic two-level stepped-plate calibration images (LaVision 309-15-3
layout) with exact ground truth, for mono and stereo stepped-calibration tests.

Geometry (``synthetic_calibration_common.stepped_board_points_mm``)
    - peak level: 21 x 21 dots on a 15 mm grid at z = 0 (peak (0, 0) = origin)
    - trough level: 20 x 20 dots on the half-pitch-offset grid at z = -3 mm
    - dot diameter 3 mm, projected through a pinhole (no distortion)

Dots are rendered anti-aliased at sub-pixel centres (4x oversample, ``cv2.circle``
fixed-point ``shift``, ``INTER_AREA`` reduction).  The old in-test renderers
rounded every centre to an integer pixel; that quantisation limits the detector
under test and is the reason the original dotboard generator was removed.

Stereo (``--cameras 2``) builds camera 2 by rotating camera 1 about the board
centroid by ``R_stereo`` (``T_stereo = (I - R) @ [0, 0, Z]``).
    same_side     R_stereo = Ry(angle)         both cameras see the same face
    transmission  R_stereo = Ry(180 deg + angle) camera 2 sees the back face, so
                  the level nearest camera 2 is the farthest from camera 1 and
                  its peak/trough labels invert (recorded in ``levels``).

Output (default ``unit-tests/synthetic_calibration/stepped``)
    cam1/calib00001.png ... (and cam2/ when --cameras 2), 8-bit, white dots on black
    stepped_ground_truth.npz
        camera_matrix (3, 3)      dist_coeffs (5,)         image_width, image_height
        rvecs_cam{c}, tvecs_cam{c} (n_poses, 3)  board -> camera c
        R_stereo (3, 3), T_stereo (3, 1)  (--cameras 2 only; T in mm)
        step_mm, spacing_mm, dot_diameter_mm
        fiducials_px_cam{c} (3, 2)  origin, +X, +Y dots of the datum (pose 0) view
        levels (n_poses, n_cameras) int8  0 = the z = 0 level is nearer that camera,
                                          1 = the z = -step level is nearer
        points_mm (N, 3)  peak points first;  points_px_cam{c} (n_poses, N, 2)
        visible_cam{c} (n_poses, N) bool -- disc wholly inside the frame

Units: object points, tvecs and T_stereo are in millimetres (the unit the stepped
calibrator works in).

Usage:
    python pivtools_cli/generate_synthetic_stepped.py --out /tmp/stepped \\
        --n-poses 5 --megapixels 1.3 --cameras 2 --stereo-config transmission
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pivtools_cli.synthetic_calibration_common import (
    STEPPED_DOT_DIAMETER_MM,
    STEPPED_PEAK_SHAPE,
    STEPPED_SPACING_MM,
    STEPPED_STEP_MM,
    STEPPED_TROUGH_SHAPE,
    compose_second_camera,
    fully_inside,
    make_camera_matrix,
    make_centred_poses,
    render_stepped_view,
    rig_from_centroid_rotation,
    save_overview_figure,
    stepped_board_points_mm,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "unit-tests"
    / "synthetic_calibration"
    / "stepped"
)
DEFAULT_ASPECT = 1.25  # 1280 x 1024 at 1.3 MP, as the real sensor
# Frontal board extent as a fraction of the short image side.  The blob detector
# drops blobs under 50 px^2, and 3 mm dots fall under that (silently, 70 blobs
# found) at 1.3 MP when the fill is 0.85; 0.90 clears it.
BOARD_FILL = 0.90
DEFAULT_MAX_TILT_RAD = 0.15  # gentle: detector yield drops past ~25 deg obliquity
DEFAULT_STEREO_ANGLE_DEG = 14.3  # cam 2 rotation about the board centroid
STEREO_CONFIGS = ("same_side", "transmission")


def image_size(megapixels: float, aspect: float) -> tuple:
    """(W, H) with W / H = aspect and W * H ~= megapixels * 1e6."""
    W = int(round(np.sqrt(megapixels * 1e6 * aspect)))
    return W, int(round(W / aspect))


def stereo_rotation(config: str, angle_deg: float) -> np.ndarray:
    """R_stereo for ``same_side`` (Ry(angle)) or ``transmission`` (Ry(180 + angle))."""
    if config not in STEREO_CONFIGS:
        raise ValueError(f"stereo config {config!r} not in {STEREO_CONFIGS}")
    total = np.radians(angle_deg + (180.0 if config == "transmission" else 0.0))
    return cv2.Rodrigues(np.array([0.0, total, 0.0]))[0]


def nearer_level(
    peak_mm: np.ndarray,
    trough_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> int:
    """0 if the z = 0 level has the smaller mean camera depth, else 1."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    t = np.asarray(tvec, np.float64).reshape(3)
    z_peak = np.einsum("j,nj->n", R[2], peak_mm).mean() + t[2]
    z_trough = np.einsum("j,nj->n", R[2], trough_mm).mean() + t[2]
    return 0 if z_peak < z_trough else 1


def generate(
    out: Path,
    n_poses: int = 5,
    megapixels: float = 1.3,
    cameras: int = 1,
    stereo_config: str = "same_side",
    seed: int = 42,
    aspect: float = DEFAULT_ASPECT,
    max_tilt: float = DEFAULT_MAX_TILT_RAD,
    stereo_angle_deg: float = DEFAULT_STEREO_ANGLE_DEG,
    overview_figure: Path | None = None,
    fill: float = BOARD_FILL,
    dot_diameter_mm: float = STEPPED_DOT_DIAMETER_MM,
) -> dict:
    """Render the set into ``out`` and return the ground-truth dict.

    See the module docstring for the on-disk layout and ground-truth keys.
    """
    if cameras not in (1, 2):
        raise ValueError(f"cameras must be 1 or 2, got {cameras}")
    if n_poses < 1:
        raise ValueError("n_poses must be >= 1")
    out = Path(out)
    W, H = image_size(megapixels, aspect)
    K = make_camera_matrix(W, H)
    fx = float(K[0, 0])

    peak, trough = stepped_board_points_mm(
        STEPPED_PEAK_SHAPE, STEPPED_TROUGH_SHAPE, STEPPED_SPACING_MM, STEPPED_STEP_MM
    )
    allpts = np.vstack([peak, trough])
    board_w = (STEPPED_PEAK_SHAPE[0] - 1) * STEPPED_SPACING_MM
    poses1 = make_centred_poses(
        n_poses,
        allpts.mean(axis=0),
        fx,
        float(min(W, H)),
        board_w,
        fill=fill,
        max_tilt=max_tilt,
        seed=seed,
    )
    poses = {1: poses1}
    gt: dict = {
        "camera_matrix": K,
        "dist_coeffs": np.zeros(5),
        "image_width": W,
        "image_height": H,
        "step_mm": STEPPED_STEP_MM,
        "spacing_mm": STEPPED_SPACING_MM,
        "dot_diameter_mm": dot_diameter_mm,
        "points_mm": allpts,
    }
    if cameras == 2:
        R_stereo = stereo_rotation(stereo_config, stereo_angle_deg)
        # Pose 0 is frontal, so the centroid depth is t_z + centroid_z.
        depth = float(poses1[0][1][2] + allpts.mean(axis=0)[2])
        T_stereo = rig_from_centroid_rotation(R_stereo, depth)
        poses[2] = compose_second_camera(poses1, R_stereo, T_stereo)
        gt["R_stereo"] = R_stereo
        gt["T_stereo"] = T_stereo.reshape(3, 1)

    levels = np.zeros((n_poses, cameras), np.int8)
    datum_img, datum_px = None, None
    for cam, cam_poses in poses.items():
        cam_dir = out / f"cam{cam}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        pts_px, vis = [], []
        for i, (rvec, tvec) in enumerate(cam_poses):
            img, px, radii = render_stepped_view(
                peak, trough, rvec, tvec, K, (W, H), dot_diameter_mm
            )
            if not cv2.imwrite(str(cam_dir / f"calib{i + 1:05d}.png"), img):
                raise OSError(f"could not write {cam_dir / f'calib{i + 1:05d}.png'}")
            pts_px.append(px)
            vis.append(fully_inside(px, radii, (W, H)))
            levels[i, cam - 1] = nearer_level(peak, trough, rvec, tvec)
            if cam == 1 and i == 0:
                datum_img, datum_px = img, px
        pts_px = np.array(pts_px)
        # Datum fiducials: peak (col 0, row 0) = origin, (col 1, row 0) = +X,
        # (col 0, row 1) = +Y.
        cols = STEPPED_PEAK_SHAPE[0]
        gt[f"rvecs_cam{cam}"] = np.array([p[0] for p in cam_poses])
        gt[f"tvecs_cam{cam}"] = np.array([p[1] for p in cam_poses])
        gt[f"points_px_cam{cam}"] = pts_px
        gt[f"visible_cam{cam}"] = np.array(vis)
        gt[f"fiducials_px_cam{cam}"] = pts_px[0][[0, 1, cols]]
        print(f"  cam{cam}: wrote {n_poses} images ({W}x{H}) to {cam_dir}")
    gt["levels"] = levels
    np.savez(str(out / "stepped_ground_truth.npz"), **gt)
    print(f"  ground truth: {out / 'stepped_ground_truth.npz'}")
    if overview_figure is not None:
        save_overview_figure(
            datum_img,
            datum_px,
            overview_figure,
            f"Synthetic stepped plate, datum view ({W}x{H}, {n_poses} poses, "
            f"{cameras} camera(s))",
        )
    return gt


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--n-poses", type=int, default=5)
    p.add_argument("--megapixels", type=float, default=1.3)
    p.add_argument("--cameras", type=int, choices=(1, 2), default=1)
    p.add_argument("--stereo-config", choices=STEREO_CONFIGS, default="same_side")
    p.add_argument("--stereo-angle-deg", type=float, default=DEFAULT_STEREO_ANGLE_DEG)
    p.add_argument(
        "--max-tilt",
        type=float,
        default=DEFAULT_MAX_TILT_RAD,
        help="max per-axis board tilt, rad",
    )
    p.add_argument("--aspect", type=float, default=DEFAULT_ASPECT, help="W / H")
    p.add_argument(
        "--fill",
        type=float,
        default=BOARD_FILL,
        help="frontal board extent / short image side",
    )
    p.add_argument("--dot-diameter-mm", type=float, default=STEPPED_DOT_DIAMETER_MM)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overview-figure",
        type=Path,
        default=None,
        help="save a datum-view overview PNG with ground-truth dots",
    )
    return p


def main(argv: list | None = None) -> None:
    """CLI entry point."""
    a = build_parser().parse_args(argv)
    generate(
        a.out,
        a.n_poses,
        a.megapixels,
        a.cameras,
        a.stereo_config,
        a.seed,
        a.aspect,
        a.max_tilt,
        a.stereo_angle_deg,
        a.overview_figure,
        a.fill,
        a.dot_diameter_mm,
    )


if __name__ == "__main__":
    main()
