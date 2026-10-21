#!/usr/bin/env python3
"""
Generate synthetic two-level stepped-plate calibration images (LaVision 309-15-3
layout) with exact ground truth, for mono and stereo stepped-calibration tests.

Geometry (``synthetic_calibration_common.stepped_board_points_mm``)
    - peak level: 21 x 21 dots on a 15 mm grid at z = 0 (peak (0, 0) = origin)
    - trough level: 20 x 20 dots on the half-pitch-offset grid at z = -3 mm
    - dot diameter 3 mm, projected through a pinhole (no distortion)
    - plate thickness T = --thickness-mm (default STEPPED_THICKNESS_MM = 14.8,
      the calibrator default), front peak plane to back peak plane

Dots are rendered anti-aliased at sub-pixel centres (4x oversample, ``cv2.circle``
fixed-point ``shift``, ``INTER_AREA`` reduction).  The old in-test renderers
rounded every centre to an integer pixel; that quantisation limits the detector
under test and is the reason the original dotboard generator was removed.

Stereo (``--cameras 2``) is a symmetric rig: both cameras turn about the board
centroid by half the included angle ``theta = --stereo-angle-deg``
(``synthetic_calibration_common.symmetric_stereo_rig``).
    same_side     cam 1 Ry(-theta/2), cam 2 Ry(+theta/2); R_stereo = Ry(theta)
                  both cameras see the same face
    transmission  cam 1 Ry(-theta/2), cam 2 Ry(180 deg + theta/2);
                  R_stereo = Ry(180 deg + theta).  Camera 2 renders the BACK
                  face (``back_face_points_mm``): the front set translated by
                  -(T - step) along the board normal, the model of the stepped
                  calibrator (``stepped_calibrate.compute_z_and_offsets``):
                      front  peak lattice   z = 0          (cam 1's datum level)
                             trough lattice z = -step
                      back   peak lattice   z = -(T - step) (cam 2's datum level)
                             trough lattice z = -T
                  Same (x, y) lattices on both faces, so cam 2's datum dot is
                  the one straight through the plate from cam 1's.  No explicit
                  mirror: viewing from Ry(180 deg) reverses the image handedness
                  of the (+X, +Y) dots, as the calibrator's classifier expects.
                  T = step reproduces the old zero-thickness render bitwise.
                  The generator's clicks (+X = +col, +Y = +row, image-down)
                  give a frame with +z AWAY from cam 1, so the back-face set
                  lies on cam 1's side of the front set: each camera images
                  the face farther from it.  This is the calibrator's model
                  in this frame, not an opaque-plate rendering.
``R_stereo`` / ``T_stereo`` map camera 1 to camera 2 for every pose
(``R2 = R_stereo R1``, ``t2 = R_stereo t1 + T_stereo``).

Output (default ``unit-tests/synthetic_calibration/stepped``)
    cam1/calib00001.png ... (and cam2/ when --cameras 2), 8-bit, white dots on black
    stepped_ground_truth.npz
        camera_matrix (3, 3)      dist_coeffs (5,)         image_width, image_height
        rvecs_cam{c}, tvecs_cam{c} (n_poses, 3)  board -> camera c
        R_stereo (3, 3), T_stereo (3, 1)  (--cameras 2 only; T in mm)
        step_mm, spacing_mm, dot_diameter_mm
        fiducial_grid_index (2,)  peak (col, row) of the datum origin dot: the first
                                  peak node, row-major from (0, 0), whose triple
                                  (c, r), (c + 1, r), (c, r + 1) is wholly inside
                                  the pose-0 frame of every camera
        fiducials_px_cam{c} (3, 2)  that origin, +X, +Y triple in the pose-0 view
        thickness_mm              plate thickness T (all configs; only transmission
                                  cam 2 renders it)
        levels (n_poses, n_cameras) int8  for the face that camera renders:
                                          0 = its peak lattice is nearer the camera,
                                          1 = its trough lattice is nearer
        points_mm (N, 3)  front face, peak points first (cam 1, same_side cam 2)
        points_mm_cam2 (N, 3)  (--cameras 2) the 3-D points cam 2 renders: the
                               back face for transmission, == points_mm for
                               same_side.  Project cam c with its own set.
        points_px_cam{c} (n_poses, N, 2)
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
    STEPPED_THICKNESS_MM,
    STEPPED_TROUGH_SHAPE,
    centred_pose_depth,
    fully_inside,
    make_camera_matrix,
    make_centred_poses,
    render_stepped_view,
    save_overview_figure,
    stepped_board_points_mm,
    symmetric_stereo_rig,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "unit-tests"
    / "synthetic_calibration"
    / "stepped"
)
DEFAULT_ASPECT = 1.25  # 1280 x 1024 at 1.3 MP, as the real sensor
# Frontal board extent as a fraction of the short image side; 0.90 keeps the
# edge dots clear of the frame.
BOARD_FILL = 0.90
DEFAULT_MAX_TILT_RAD = 0.15  # gentle: detector yield drops past ~25 deg obliquity
DEFAULT_STEREO_ANGLE_DEG = 14.3  # included angle between the two cameras
STEREO_CONFIGS = ("same_side", "transmission")


def image_size(megapixels: float, aspect: float) -> tuple:
    """(W, H) with W / H = aspect and W * H ~= megapixels * 1e6."""
    W = int(round(np.sqrt(megapixels * 1e6 * aspect)))
    return W, int(round(W / aspect))


def half_angle_rotation(angle_deg: float) -> np.ndarray:
    """Ry(angle / 2): each camera's share of the included stereo angle."""
    return cv2.Rodrigues(np.array([0.0, np.radians(angle_deg) / 2.0, 0.0]))[0]


def fiducial_triple(visible_pose0: list, cols: int, rows: int, label: str) -> tuple:
    """First peak node (c, r), row-major from (0, 0), clickable in every camera.

    ``visible_pose0`` holds one (N,) bool array per camera (peak dots first,
    row-major).  The triple (c, r), (c + 1, r), (c, r + 1) must be wholly
    inside the pose-0 frame of every camera so a click on each snaps to it.

    Returns ``((c, r), [origin, +X, +Y] flat peak indices)``.
    """
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx = [r * cols + c, r * cols + c + 1, (r + 1) * cols + c]
            if all(v[idx].all() for v in visible_pose0):
                return (c, r), idx
    raise ValueError(f"no fully visible fiducial triple in pose 0 for {label}")


def back_face_points_mm(
    peak_mm: np.ndarray, trough_mm: np.ndarray, thickness_mm: float, step_mm: float
) -> tuple:
    """Back face of a plate ``thickness_mm`` thick, as the stepped calibrator models it.

    ``compute_z_and_offsets`` (transmission, cam 1 datum level at z = 0) puts
    camera 2's datum (peak) lattice at z = -(T - step) and its other lattice
    at z = -T, on the same (x, y) lattices as the front.  That is the front
    set translated by -(T - step) along z (a constant wall T - step).

    Returns (peak (Np, 3), trough (Nt, 3)) of the back face.
    """
    if thickness_mm < step_mm:
        raise ValueError(
            f"thickness_mm ({thickness_mm}) must be >= step_mm ({step_mm}): the "
            "back face would cut through the front"
        )
    shift = np.array([0.0, 0.0, -(thickness_mm - step_mm)])
    return peak_mm + shift, trough_mm + shift


def nearer_level(
    peak_mm: np.ndarray,
    trough_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> int:
    """0 if the peak lattice has the smaller mean camera depth, else 1."""
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
    thickness_mm: float = STEPPED_THICKNESS_MM,
) -> dict:
    """Render the set into ``out`` and return the ground-truth dict.

    ``thickness_mm`` only changes transmission camera 2, which renders the
    back face (``back_face_points_mm``).  See the module docstring for the
    on-disk layout and ground-truth keys.
    """
    if cameras not in (1, 2):
        raise ValueError(f"cameras must be 1 or 2, got {cameras}")
    if n_poses < 1:
        raise ValueError("n_poses must be >= 1")
    if cameras == 2 and stereo_config not in STEREO_CONFIGS:
        raise ValueError(f"stereo config {stereo_config!r} not in {STEREO_CONFIGS}")
    out = Path(out)
    W, H = image_size(megapixels, aspect)
    K = make_camera_matrix(W, H)
    fx = float(K[0, 0])

    peak, trough = stepped_board_points_mm(
        STEPPED_PEAK_SHAPE, STEPPED_TROUGH_SHAPE, STEPPED_SPACING_MM, STEPPED_STEP_MM
    )
    back = back_face_points_mm(peak, trough, thickness_mm, STEPPED_STEP_MM)
    allpts = np.vstack([peak, trough])
    board_w = (STEPPED_PEAK_SHAPE[0] - 1) * STEPPED_SPACING_MM
    fit_px = float(min(W, H))
    poses_rig = make_centred_poses(
        n_poses,
        allpts.mean(axis=0),
        fx,
        fit_px,
        board_w,
        fill=fill,
        max_tilt=max_tilt,
        seed=seed,
    )
    poses = {1: poses_rig}
    gt: dict = {
        "camera_matrix": K,
        "dist_coeffs": np.zeros(5),
        "image_width": W,
        "image_height": H,
        "step_mm": STEPPED_STEP_MM,
        "spacing_mm": STEPPED_SPACING_MM,
        "dot_diameter_mm": dot_diameter_mm,
        "thickness_mm": float(thickness_mm),
        "points_mm": allpts,
    }
    # The (peak, trough) set each camera renders.
    faces = {1: (peak, trough)}
    if cameras == 2:
        poses[1], poses[2], R_stereo, T_stereo = symmetric_stereo_rig(
            poses_rig,
            half_angle_rotation(stereo_angle_deg),
            centred_pose_depth(fx, fit_px, board_w, fill),
            transmission=stereo_config == "transmission",
        )
        gt["R_stereo"] = R_stereo
        gt["T_stereo"] = T_stereo.reshape(3, 1)
        faces[2] = back if stereo_config == "transmission" else (peak, trough)
        gt["points_mm_cam2"] = np.vstack(faces[2])

    levels = np.zeros((n_poses, cameras), np.int8)
    datum_img, datum_px = None, None
    pts_px_by_cam, vis_by_cam = {}, {}
    for cam, cam_poses in poses.items():
        cam_dir = out / f"cam{cam}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        pts_px, vis = [], []
        face_peak, face_trough = faces[cam]
        for i, (rvec, tvec) in enumerate(cam_poses):
            img, px, radii = render_stepped_view(
                face_peak, face_trough, rvec, tvec, K, (W, H), dot_diameter_mm
            )
            if not cv2.imwrite(str(cam_dir / f"calib{i + 1:05d}.png"), img):
                raise OSError(f"could not write {cam_dir / f'calib{i + 1:05d}.png'}")
            pts_px.append(px)
            vis.append(fully_inside(px, radii, (W, H)))
            levels[i, cam - 1] = nearer_level(face_peak, face_trough, rvec, tvec)
            if cam == 1 and i == 0:
                datum_img, datum_px = img, px
        pts_px_by_cam[cam] = np.array(pts_px)
        vis_by_cam[cam] = np.array(vis)
        gt[f"rvecs_cam{cam}"] = np.array([p[0] for p in cam_poses])
        gt[f"tvecs_cam{cam}"] = np.array([p[1] for p in cam_poses])
        gt[f"points_px_cam{cam}"] = pts_px_by_cam[cam]
        gt[f"visible_cam{cam}"] = vis_by_cam[cam]
        print(f"  cam{cam}: wrote {n_poses} images ({W}x{H}) to {cam_dir}")
    # Datum fiducials: origin, +X, +Y peak dots, clickable in every camera.
    (c0, r0), fid_idx = fiducial_triple(
        [v[0] for v in vis_by_cam.values()],
        STEPPED_PEAK_SHAPE[0],
        STEPPED_PEAK_SHAPE[1],
        stereo_config if cameras == 2 else "mono",
    )
    gt["fiducial_grid_index"] = np.array([c0, r0])
    for cam, pts_px in pts_px_by_cam.items():
        gt[f"fiducials_px_cam{cam}"] = pts_px[0][fid_idx]
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
    p.add_argument(
        "--stereo-angle-deg",
        type=float,
        default=DEFAULT_STEREO_ANGLE_DEG,
        help="included angle between the two cameras, deg (each turns by half "
        "about the board centroid; transmission adds 180 deg to camera 2)",
    )
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
    p.add_argument(
        "--thickness-mm",
        type=float,
        default=STEPPED_THICKNESS_MM,
        help="plate thickness, front peak plane to back peak plane, mm (transmission "
        "camera 2 renders the back face; must be >= the 3 mm step)",
    )
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
        a.thickness_mm,
    )


if __name__ == "__main__":
    main()
