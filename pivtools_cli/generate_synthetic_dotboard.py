#!/usr/bin/env python3
"""
Generate synthetic planar dot-grid calibration images (mono or stereo) with exact
ground truth, an optional illumination field, and an optional one-camera clip.

This replaces the dotboard renderer removed on 2026-08-28, which rounded every dot
centre to an integer pixel and drew aliased discs.  Dots here are rendered at
sub-pixel centres (4x oversample, ``cv2.circle`` fixed-point ``shift``,
``INTER_AREA`` reduction) with per-dot radius proportional to the local
nearest-neighbour spacing so perspective never merges neighbours.

Board: ``--cols`` x ``--rows`` white dots on black, ``--spacing-mm`` pitch, dot
(col 0, row 0) at the origin, +x along columns, +y along rows, z = 0.

Illumination (``--illumination``) multiplies the image by a smooth field in
[0.4, 1.0].  ``ramp`` rises linearly left to right, ``vignette`` falls with the
square of the distance from the centre, ``none`` leaves the render untouched.

Stereo (``--cameras 2``) is a symmetric same-side rig: camera 1 and camera 2 turn
about the board centroid by -theta/2 and +theta/2 about Y, theta =
``--stereo-angle-deg`` the included angle between them
(``synthetic_calibration_common.symmetric_stereo_rig``; ``R_stereo = Ry(theta)``
maps camera 1 to camera 2).  ``--clip-cam2 PX`` then removes every
dot of camera 2 whose disc reaches left of column PX (a partially visible board
in camera 2 only, so its grid indices re-zero at a different physical dot).

Output
    mono    <out>/calib00001.png ...            <out>/ground_truth.npz
    stereo  <out>/cam1/calib*.png, <out>/cam2/  <out>/stereo_ground_truth.npz
    8-bit greyscale PNG.  Ground truth (metres, board -> camera):
        camera_matrix, dist_coeffs, image_width, image_height,
        rvecs / tvecs (mono)  or  rvecs_cam{1,2} / tvecs_cam{1,2}, R_stereo, T_stereo
        points_m (N, 3), points_px[_cam{c}] (n_views, N, 2),
        visible[_cam{c}] (n_views, N) bool -- disc fully inside the frame and not
        clipped, i.e. a dot the detector is entitled to find,
        cols, rows, spacing_mm, illumination, clip_cam2_px (-1 when unused)

Usage:
    python pivtools_cli/generate_synthetic_dotboard.py --out /tmp/dots \\
        --n-views 8 --megapixels 4.5 --aspect 1.5 --illumination ramp
    python pivtools_cli/generate_synthetic_dotboard.py --cameras 2 --clip-cam2 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pivtools_cli.synthetic_calibration_common import (
    adaptive_dot_radii,
    centred_pose_depth,
    fully_inside,
    make_camera_matrix,
    make_centred_poses,
    project_points,
    render_dots,
    save_overview_figure,
    symmetric_stereo_rig,
)

DEFAULT_OUT_ROOT = (
    Path(__file__).resolve().parent.parent / "unit-tests" / "synthetic_calibration"
)
ILLUMINATION_MODES = ("ramp", "vignette", "none")
ILLUMINATION_MIN = 0.4  # darkest point of the field
ILLUMINATION_MAX = 1.0
BOARD_FILL = 0.70  # frontal board width as a fraction of image width
DEFAULT_MAX_TILT_RAD = 0.25  # ~14 deg
DEFAULT_STEREO_ANGLE_DEG = 8.0  # included angle between the two cameras
DOT_RADIUS_FRACTION = 0.22  # of the nearest-neighbour pixel distance
MAX_MEGAPIXELS = 25.0
ROW_CHUNK = 256  # rows per illumination multiply (bounds float memory)
DOT_BORDER_MARGIN_PX = 1.0  # a disc closer than this to the frame edge is not visible


def image_size(megapixels: float, aspect: float) -> tuple:
    """(W, H) with W / H = aspect and W * H ~= megapixels * 1e6."""
    if not 0 < megapixels <= MAX_MEGAPIXELS:
        raise ValueError(
            f"megapixels must be in (0, {MAX_MEGAPIXELS}], got {megapixels}"
        )
    W = int(round(np.sqrt(megapixels * 1e6 * aspect)))
    return W, int(round(W / aspect))


def illumination_field(mode: str, rows: slice, size: tuple) -> np.ndarray:
    """Multiplicative field (rows, W) in [ILLUMINATION_MIN, ILLUMINATION_MAX]."""
    W, H = size
    y = np.arange(rows.start, rows.stop, dtype=np.float64)[:, None]
    x = np.arange(W, dtype=np.float64)[None, :]
    span = ILLUMINATION_MAX - ILLUMINATION_MIN
    if mode == "ramp":
        s = np.broadcast_to(x / max(W - 1, 1), (y.shape[0], W))
        return ILLUMINATION_MIN + span * s
    if mode == "vignette":
        r2 = ((x - (W - 1) / 2) ** 2 + (y - (H - 1) / 2) ** 2) / (
            ((W - 1) / 2) ** 2 + ((H - 1) / 2) ** 2
        )
        return ILLUMINATION_MAX - span * r2
    raise ValueError(f"illumination {mode!r} not in {ILLUMINATION_MODES}")


def apply_illumination(img: np.ndarray, mode: str) -> np.ndarray:
    """Multiply ``img`` (uint8) by the illumination field, in place, row-chunked."""
    if mode == "none":
        return img
    H, W = img.shape
    for y0 in range(0, H, ROW_CHUNK):
        sl = slice(y0, min(y0 + ROW_CHUNK, H))
        field = illumination_field(mode, sl, (W, H))
        img[sl] = np.round(img[sl] * field).astype(np.uint8)
    return img


def board_points_m(cols: int, rows: int, spacing_mm: float) -> np.ndarray:
    """(cols * rows, 3) dot centres in metres, row-major, origin at dot (0, 0)."""
    sp = spacing_mm / 1000.0
    return np.array(
        [[c * sp, r * sp, 0.0] for r in range(rows) for c in range(cols)], np.float64
    )


def generate(
    out: Path,
    n_views: int = 10,
    megapixels: float = 1.0,
    aspect: float = 1.0,
    cols: int = 15,
    rows: int = 12,
    spacing_mm: float = 15.0,
    illumination: str = "none",
    cameras: int = 1,
    clip_cam2: float | None = None,
    stereo_angle_deg: float = DEFAULT_STEREO_ANGLE_DEG,
    max_tilt: float = DEFAULT_MAX_TILT_RAD,
    seed: int = 42,
    overview_figure: Path | None = None,
) -> dict:
    """Render the set into ``out`` and return the ground-truth dict."""
    if cameras not in (1, 2):
        raise ValueError(f"cameras must be 1 or 2, got {cameras}")
    if illumination not in ILLUMINATION_MODES:
        raise ValueError(f"illumination {illumination!r} not in {ILLUMINATION_MODES}")
    if clip_cam2 is not None and cameras != 2:
        raise ValueError("--clip-cam2 needs --cameras 2")
    out = Path(out)
    W, H = image_size(megapixels, aspect)
    K = make_camera_matrix(W, H)
    pts = board_points_m(cols, rows, spacing_mm)
    board_w = (cols - 1) * spacing_mm / 1000.0
    fx = float(K[0, 0])
    poses_rig = make_centred_poses(
        n_views,
        pts.mean(axis=0),
        fx,
        float(W),
        board_w,
        fill=BOARD_FILL,
        max_tilt=max_tilt,
        seed=seed,
    )
    gt: dict = {
        "camera_matrix": K,
        "dist_coeffs": np.zeros(5),
        "image_width": W,
        "image_height": H,
        "points_m": pts,
        "cols": cols,
        "rows": rows,
        "spacing_mm": spacing_mm,
        "illumination": illumination,
        "clip_cam2_px": -1.0 if clip_cam2 is None else float(clip_cam2),
    }
    poses_by_cam = {1: poses_rig}
    if cameras == 2:
        half = cv2.Rodrigues(np.array([0.0, np.radians(stereo_angle_deg) / 2.0, 0.0]))[
            0
        ]
        poses_by_cam[1], poses_by_cam[2], R_stereo, T_stereo = symmetric_stereo_rig(
            poses_rig, half, centred_pose_depth(fx, float(W), board_w, BOARD_FILL)
        )
        gt["R_stereo"] = R_stereo
        gt["T_stereo"] = T_stereo.reshape(3, 1)

    datum_img, datum_px, datum_vis = None, None, None
    for cam, cam_poses in poses_by_cam.items():
        img_dir = out if cameras == 1 else out / f"cam{cam}"
        img_dir.mkdir(parents=True, exist_ok=True)
        all_px, all_vis = [], []
        for i, (rvec, tvec) in enumerate(cam_poses):
            px = project_points(pts, rvec, tvec, K)
            radii = adaptive_dot_radii(px, DOT_RADIUS_FRACTION)
            inside = fully_inside(px, radii, (W, H), DOT_BORDER_MARGIN_PX)
            clipped = np.zeros(len(px), bool)
            if cam == 2 and clip_cam2 is not None:
                clipped = px[:, 0] - radii < clip_cam2
            # Dots straddling the frame edge are drawn (the detector must reject
            # them) but are not counted as visible.
            on_frame = (
                (px[:, 0] + radii > 0)
                & (px[:, 0] - radii < W)
                & (px[:, 1] + radii > 0)
                & (px[:, 1] - radii < H)
            )
            keep = on_frame & ~clipped
            drawn = inside & ~clipped
            img = render_dots(px[keep], radii[keep], (W, H))
            apply_illumination(img, illumination)
            path = img_dir / f"calib{i + 1:05d}.png"
            if not cv2.imwrite(str(path), img):
                raise OSError(f"could not write {path}")
            all_px.append(px)
            all_vis.append(drawn)
            if cam == 1 and i == 0:
                datum_img, datum_px, datum_vis = img, px, drawn
        suffix = "" if cameras == 1 else f"_cam{cam}"
        gt[f"rvecs{suffix}"] = np.array([p[0] for p in cam_poses])
        gt[f"tvecs{suffix}"] = np.array([p[1] for p in cam_poses])
        gt[f"points_px{suffix}"] = np.array(all_px)
        gt[f"visible{suffix}"] = np.array(all_vis)
        print(f"  cam{cam}: wrote {n_views} images ({W}x{H}) to {img_dir}")
    gt_path = out / ("ground_truth.npz" if cameras == 1 else "stereo_ground_truth.npz")
    np.savez(str(gt_path), **gt)
    print(f"  ground truth: {gt_path}")
    if overview_figure is not None:
        save_overview_figure(
            datum_img,
            datum_px[datum_vis],
            overview_figure,
            f"Synthetic dotboard, datum view ({W}x{H}, {cols}x{rows} dots, "
            f"illumination={illumination})",
        )
    return gt


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default unit-tests/synthetic_calibration/"
        "dotboard, or stereo_dotboard with --cameras 2)",
    )
    p.add_argument("--n-views", type=int, default=10)
    p.add_argument(
        "--megapixels", type=float, default=1.0, help=f"up to {MAX_MEGAPIXELS}"
    )
    p.add_argument("--aspect", type=float, default=1.0, help="W / H")
    p.add_argument("--cols", type=int, default=15)
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--spacing-mm", type=float, default=15.0)
    p.add_argument("--illumination", choices=ILLUMINATION_MODES, default="none")
    p.add_argument("--cameras", type=int, choices=(1, 2), default=1)
    p.add_argument(
        "--clip-cam2",
        type=float,
        default=None,
        metavar="PX",
        help="camera 2 only: drop dots whose disc reaches left of column PX",
    )
    p.add_argument(
        "--stereo-angle-deg",
        type=float,
        default=DEFAULT_STEREO_ANGLE_DEG,
        help="included angle between the two cameras, deg (each turns by half "
        "about the board centroid)",
    )
    p.add_argument(
        "--max-tilt",
        type=float,
        default=DEFAULT_MAX_TILT_RAD,
        help="max per-axis board tilt, rad",
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
    out = a.out or DEFAULT_OUT_ROOT / (
        "dotboard" if a.cameras == 1 else "stereo_dotboard"
    )
    generate(
        out,
        a.n_views,
        a.megapixels,
        a.aspect,
        a.cols,
        a.rows,
        a.spacing_mm,
        a.illumination,
        a.cameras,
        a.clip_cam2,
        a.stereo_angle_deg,
        a.max_tilt,
        a.seed,
        a.overview_figure,
    )


if __name__ == "__main__":
    main()
