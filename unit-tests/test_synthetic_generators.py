"""Synthetic stepped / dotboard generators: ground truth, anti-aliasing, recovery.

Each generator runs into ``tmp_path`` (the images are gitignored, the generators
are the tracked artefact).  Recovery is the fraction of ground-truth dots that
are fully inside the frame (``visible``) for which the production detector
returns a centre within ``MATCH_TOL_PX``.  Thresholds are the plan's: >= 99 %
on the clean set and >= 95 % under the 0.4-1.0 illumination ramp at 2600 and
6000 px long side (the latter pins the ``min(70, ...)`` flat-field kernel).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from pivtools_cli import generate_synthetic_dotboard as gdot
from pivtools_cli import generate_synthetic_stepped as gstep
from pivtools_cli.synthetic_calibration_common import adaptive_dot_radii, render_dots
from pivtools_gui.calibration.detection.dotboard import DotboardDetector, DotboardParams
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams

MATCH_TOL_PX = 1.0  # detected centre must lie this close to the true centre
CLEAN_RECOVERY = 0.99
STEPPED_DOT_CASES = [3.0, 4.0]
RAMP_RECOVERY = 0.95
LONG_SIDE_PX = (2600, 6000)
ASPECT_LONG = 1.5
SUBPIXEL_TOL_PX = 0.05  # centroid of one rendered disc vs its exact centre


def _megapixels_for_long_side(long_side: int, aspect: float) -> float:
    return long_side * round(long_side / aspect) / 1e6


def recovery(detected: np.ndarray, truth: np.ndarray) -> tuple:
    """(fraction of truth dots matched within MATCH_TOL_PX, max matched error px)."""
    if len(detected) == 0:
        return 0.0, float("nan")
    d, _ = cKDTree(detected).query(truth)
    ok = d <= MATCH_TOL_PX
    return float(ok.mean()), float(d[ok].max()) if ok.any() else float("nan")


def _read(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert img is not None, path
    return img


def _dot_detect(img):
    return DotboardDetector(DotboardParams(dot_spacing_mm=15.0)).detect(img)


def _save_overlay(output_dir, name, img, truth, detected, title):
    from pivtools_cli.synthetic_calibration_common import save_overview_figure

    save_overview_figure(
        img, truth, output_dir / "synthetic_generators" / f"{name}.png", title, detected
    )


# ---------------------------------------------------------------------------
# Shared renderer
# ---------------------------------------------------------------------------


def test_render_dots_centroid_is_subpixel_exact():
    """A disc at a fractional centre has its intensity centroid on that centre."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(20, 44, size=(2, 2)) + np.array([[0, 0], [90, 0]])
    radii = np.array([6.3, 9.7])
    img = render_dots(centres, radii, (140, 64)).astype(np.float64)
    yy, xx = np.mgrid[:64, :140]
    for c, r in zip(centres, radii):
        m = (np.hypot(xx - c[0], yy - c[1]) < r + 3) * img
        cx, cy = (m * xx).sum() / m.sum(), (m * yy).sum() / m.sum()
        assert abs(cx - c[0]) < SUBPIXEL_TOL_PX and abs(cy - c[1]) < SUBPIXEL_TOL_PX
    # anti-aliased: edge pixels take intermediate grey levels
    assert ((img > 0) & (img < 255)).sum() > 40


def test_render_dots_strip_seam_matches_single_pass():
    """Discs straddling a strip boundary are identical to a one-strip render."""
    from pivtools_cli import synthetic_calibration_common as common

    pts = np.array([[30.2, common.STRIP_HEIGHT_PX - 1.3], [70.6, 40.4]])
    rad = np.array([8.0, 5.5])
    stripped = render_dots(pts, rad, (100, common.STRIP_HEIGHT_PX + 60))
    old = common.STRIP_HEIGHT_PX
    try:
        common.STRIP_HEIGHT_PX = 10_000
        whole = render_dots(pts, rad, (100, old + 60))
    finally:
        common.STRIP_HEIGHT_PX = old
    assert np.array_equal(stripped, whole)


# ---------------------------------------------------------------------------
# Stepped generator
# ---------------------------------------------------------------------------

STEPPED_KEYS = {
    "camera_matrix",
    "rvecs_cam1",
    "tvecs_cam1",
    "step_mm",
    "spacing_mm",
    "fiducials_px_cam1",
    "fiducial_grid_index",
    "levels",
    "image_width",
    "image_height",
}


@pytest.mark.parametrize("dot_mm", STEPPED_DOT_CASES)
def test_stepped_single_camera_ground_truth_and_recovery(
    tmp_path, make_figures, output_dir, dot_mm
):
    gstep.main(
        [
            "--out",
            str(tmp_path),
            "--n-poses",
            "3",
            "--cameras",
            "1",
            "--dot-diameter-mm",
            str(dot_mm),
        ]
    )
    gt = np.load(tmp_path / "stepped_ground_truth.npz")
    assert STEPPED_KEYS <= set(gt.files)
    assert "R_stereo" not in gt.files and "rvecs_cam2" not in gt.files
    assert gt["levels"].shape == (3, 1)
    assert gt["fiducials_px_cam1"].shape == (3, 2)
    assert len(list((tmp_path / "cam1").glob("calib*.png"))) == 3

    det = SteppedDetector(SteppedParams(dot_spacing_mm=15.0, step_height_mm=3.0))
    W, H = int(gt["image_width"]), int(gt["image_height"])
    for i in range(3):
        img = _read(tmp_path / "cam1" / f"calib{i + 1:05d}.png")
        assert img.shape == (H, W) and img.dtype == np.uint8
        res = det.detect(img)
        assert res.success, res.diagnostics.get("error")
        truth = gt["points_px_cam1"][i][gt["visible_cam1"][i]]
        frac, _ = recovery(res.image_points, truth)
        print(
            f"stepped {dot_mm} mm dots pose {i}: recovery {frac:.4f} "
            f"({len(res.image_points)} found)"
        )
        assert frac >= CLEAN_RECOVERY, f"pose {i}: {frac:.3f}"
        if make_figures:
            _save_overlay(
                output_dir,
                f"stepped_{dot_mm:g}mm_pose{i}",
                img,
                truth,
                res.image_points,
                f"stepped recovery {frac:.3f} (pose {i})",
            )


@pytest.mark.parametrize("config", gstep.STEREO_CONFIGS)
def test_stepped_stereo_ground_truth_is_self_consistent(tmp_path, config):
    gstep.main(
        [
            "--out",
            str(tmp_path),
            "--n-poses",
            "3",
            "--cameras",
            "2",
            "--stereo-config",
            config,
        ]
    )
    gt = np.load(tmp_path / "stepped_ground_truth.npz")
    assert STEPPED_KEYS | {
        "rvecs_cam2",
        "tvecs_cam2",
        "R_stereo",
        "T_stereo",
        "fiducials_px_cam2",
    } <= set(gt.files)
    K = gt["camera_matrix"]
    pts = gt["points_mm"]
    # cam2 pose == R_stereo . cam1 pose + T_stereo, and projections match the images' truth
    for i in range(3):
        R1 = cv2.Rodrigues(gt["rvecs_cam1"][i])[0]
        R2 = cv2.Rodrigues(gt["rvecs_cam2"][i])[0]
        assert np.allclose(R2, gt["R_stereo"] @ R1, atol=1e-12)
        assert np.allclose(
            gt["tvecs_cam2"][i],
            gt["R_stereo"] @ gt["tvecs_cam1"][i] + gt["T_stereo"].ravel(),
            atol=1e-9,
        )
        px2 = cv2.projectPoints(
            pts, gt["rvecs_cam2"][i], gt["tvecs_cam2"][i], K, np.zeros(5)
        )[0].reshape(-1, 2)
        assert np.allclose(px2, gt["points_px_cam2"][i], atol=1e-9)
    # Symmetric rig: each camera takes half the included angle theta about Y.
    half = np.radians(gstep.DEFAULT_STEREO_ANGLE_DEG) / 2.0
    R1 = cv2.Rodrigues(gt["rvecs_cam1"][0])[0]
    R2 = cv2.Rodrigues(gt["rvecs_cam2"][0])[0]
    assert np.allclose(gt["rvecs_cam1"][0], [0.0, -half, 0.0], atol=1e-12)
    if config == "same_side":
        assert np.allclose(gt["rvecs_cam2"][0], [0.0, half, 0.0], atol=1e-12)
    else:
        Ry_back = cv2.Rodrigues(np.array([0.0, np.pi + half, 0.0]))[0]
        assert np.allclose(R2, Ry_back, atol=1e-12)
    # Optical axis of camera k in the board frame is row 3 of R_k.  Pose 0 has
    # R_k = Ry(phi_k), so the axes differ by phi_2 - phi_1: theta (same_side) or
    # 180 deg + theta, i.e. an unsigned angle of 180 deg - theta (transmission).
    axis_deg = np.degrees(np.arccos(np.clip(R1[2] @ R2[2], -1.0, 1.0)))
    expected_axis_deg = (
        gstep.DEFAULT_STEREO_ANGLE_DEG
        if config == "same_side"
        else 180.0 - gstep.DEFAULT_STEREO_ANGLE_DEG
    )
    assert np.isclose(axis_deg, expected_axis_deg, atol=1e-9)
    # Which level faces each camera.  Every camera of the symmetric rig is at most
    # theta/2 + max_tilt (~16 deg) off the board normal, so the trough level
    # (z = -step, the side the cameras view from) is nearer cam 1 and same_side
    # cam 2 in every pose; transmission cam 2 views the back face, so the peak
    # level is nearer.  Verified with gstep.nearer_level on these poses.
    expected_cam2 = 1 if config == "same_side" else 0
    assert (gt["levels"][:, 0] == 1).all()
    assert (gt["levels"][:, 1] == expected_cam2).all()
    for cam in (1, 2):
        assert len(list((tmp_path / f"cam{cam}").glob("calib*.png"))) == 3


@pytest.mark.parametrize("config", gstep.STEREO_CONFIGS)
def test_stepped_fiducials_visible_and_snappable(tmp_path, config):
    """Each GT fiducial is wholly in frame and a click on it snaps to itself.

    The GUI ``snap_fiducial`` route snaps a click to the nearest detected point
    (``world_frame._snap``), so a detected centre within MATCH_TOL_PX of every
    fiducial means the datum triple is clickable in both cameras.
    """
    gstep.main(
        [
            "--out",
            str(tmp_path),
            "--n-poses",
            "1",
            "--cameras",
            "2",
            "--stereo-config",
            config,
        ]
    )
    gt = np.load(tmp_path / "stepped_ground_truth.npz")
    c, r = (int(v) for v in gt["fiducial_grid_index"])
    cols = gstep.STEPPED_PEAK_SHAPE[0]
    idx = [r * cols + c, r * cols + c + 1, (r + 1) * cols + c]
    det = SteppedDetector(SteppedParams())
    for cam in (1, 2):
        fid = gt[f"fiducials_px_cam{cam}"]
        assert fid.shape == (3, 2)
        assert np.array_equal(fid, gt[f"points_px_cam{cam}"][0][idx])
        assert gt[f"visible_cam{cam}"][0][idx].all()
        res = det.detect(_read(tmp_path / f"cam{cam}" / "calib00001.png"))
        assert res.success, res.diagnostics.get("error")
        d, _ = cKDTree(res.image_points).query(fid)
        print(f"stepped {config} cam{cam} fiducial {c, r}: snap distances {d}")
        assert (d <= MATCH_TOL_PX).all(), f"cam{cam}: {d}"


def test_stepped_transmission_recovery_on_both_cameras(tmp_path):
    gstep.main(
        [
            "--out",
            str(tmp_path),
            "--n-poses",
            "2",
            "--cameras",
            "2",
            "--stereo-config",
            "transmission",
            "--dot-diameter-mm",
            "4",
        ]
    )
    gt = np.load(tmp_path / "stepped_ground_truth.npz")
    det = SteppedDetector(SteppedParams(dot_spacing_mm=15.0, step_height_mm=3.0))
    for cam in (1, 2):
        for i in range(2):
            res = det.detect(_read(tmp_path / f"cam{cam}" / f"calib{i + 1:05d}.png"))
            assert res.success, res.diagnostics.get("error")
            frac, _ = recovery(
                res.image_points,
                gt[f"points_px_cam{cam}"][i][gt[f"visible_cam{cam}"][i]],
            )
            print(f"stepped transmission cam{cam} pose {i}: recovery {frac:.4f}")
            assert frac >= CLEAN_RECOVERY, f"cam{cam} pose {i}: {frac:.3f}"


# ---------------------------------------------------------------------------
# Dotboard generator
# ---------------------------------------------------------------------------

DOTBOARD_KEYS = {
    "camera_matrix",
    "dist_coeffs",
    "rvecs",
    "tvecs",
    "image_width",
    "image_height",
    "points_px",
    "visible",
}


def test_dotboard_clean_ground_truth_and_recovery(tmp_path, make_figures, output_dir):
    gdot.main(["--out", str(tmp_path), "--n-views", "4"])
    gt = np.load(tmp_path / "ground_truth.npz")
    assert DOTBOARD_KEYS <= set(gt.files)
    for i in range(4):
        img = _read(tmp_path / f"calib{i + 1:05d}.png")
        res = _dot_detect(img)
        assert res.success
        truth = gt["points_px"][i][gt["visible"][i]]
        frac, err = recovery(res.image_points, truth)
        print(f"dotboard clean view {i}: recovery {frac:.4f}, max err {err:.3f} px")
        assert frac >= CLEAN_RECOVERY, f"view {i}: {frac:.3f}"
        if make_figures:
            _save_overlay(
                output_dir,
                f"dotboard_clean_view{i}",
                img,
                truth,
                res.image_points,
                f"dotboard clean, recovery {frac:.3f}",
            )


@pytest.mark.parametrize("illumination", ["ramp", "vignette"])
@pytest.mark.parametrize("long_side", LONG_SIDE_PX)
def test_dotboard_illumination_recovery(
    tmp_path, make_figures, output_dir, long_side, illumination
):
    mp = _megapixels_for_long_side(long_side, ASPECT_LONG)
    gdot.main(
        [
            "--out",
            str(tmp_path),
            "--n-views",
            "2",
            "--megapixels",
            f"{mp:.6f}",
            "--aspect",
            str(ASPECT_LONG),
            "--illumination",
            illumination,
        ]
    )
    gt = np.load(tmp_path / "ground_truth.npz")
    assert int(gt["image_width"]) == long_side
    for i in range(2):
        img = _read(tmp_path / f"calib{i + 1:05d}.png")
        res = _dot_detect(img)
        assert res.success
        truth = gt["points_px"][i][gt["visible"][i]]
        frac, err = recovery(res.image_points, truth)
        print(
            f"dotboard {illumination} {long_side}px view {i}: recovery {frac:.4f}, "
            f"max err {err:.3f} px"
        )
        assert (
            frac >= RAMP_RECOVERY
        ), f"{illumination} {long_side}px view {i}: {frac:.3f}"
        if make_figures and i == 0:
            _save_overlay(
                output_dir,
                f"dotboard_{illumination}_{long_side}",
                img,
                truth,
                res.image_points,
                f"dotboard {illumination} {long_side}px, recovery {frac:.3f}",
            )


def test_illumination_field_range():
    for mode in ("ramp", "vignette"):
        f = gdot.illumination_field(mode, slice(0, 50), (80, 50))
        assert f.min() >= gdot.ILLUMINATION_MIN - 1e-9
        assert f.max() <= gdot.ILLUMINATION_MAX + 1e-9
        assert f.max() - f.min() > 0.5


def test_dotboard_stereo_clip_cam2(tmp_path, make_figures, output_dir):
    clip = 450
    gdot.main(
        [
            "--out",
            str(tmp_path),
            "--cameras",
            "2",
            "--n-views",
            "3",
            "--clip-cam2",
            str(clip),
        ]
    )
    gt = np.load(tmp_path / "stereo_ground_truth.npz")
    assert {"R_stereo", "T_stereo", "rvecs_cam1", "tvecs_cam2", "visible_cam2"} <= set(
        gt.files
    )
    for i in range(3):
        v1, v2 = gt["visible_cam1"][i], gt["visible_cam2"][i]
        assert v2.sum() < v1.sum(), "clip removed no dots from camera 2"
        # Clipped dots are absent from the image, not merely unlabelled.
        px2 = gt["points_px_cam2"][i]
        res2 = _dot_detect(_read(tmp_path / "cam2" / f"calib{i + 1:05d}.png"))
        assert res2.success
        assert res2.image_points[:, 0].min() >= clip - 1.0
        frac2, _ = recovery(res2.image_points, px2[v2])
        res1 = _dot_detect(_read(tmp_path / "cam1" / f"calib{i + 1:05d}.png"))
        frac1, _ = recovery(res1.image_points, gt["points_px_cam1"][i][v1])
        print(
            f"dotboard clipped cam2 view {i}: cam1 {frac1:.4f} ({v1.sum()} dots), "
            f"cam2 {frac2:.4f} ({v2.sum()} of {len(v1)} dots)"
        )
        assert frac1 >= CLEAN_RECOVERY and frac2 >= CLEAN_RECOVERY
        if make_figures and i == 0:
            _save_overlay(
                output_dir,
                "dotboard_clip_cam2",
                _read(tmp_path / "cam2" / "calib00001.png"),
                px2[v2],
                res2.image_points,
                f"cam2 clipped at x={clip}px, recovery {frac2:.3f}",
            )


def test_dotboard_rejects_clip_without_stereo(tmp_path):
    with pytest.raises(ValueError, match="cameras 2"):
        gdot.generate(tmp_path, clip_cam2=100.0)


def test_adaptive_radii_follow_spacing():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 40.0]])
    r = adaptive_dot_radii(pts, 0.22, min_radius_px=0.0)
    assert np.allclose(r[:2], 2.2) and np.isclose(r[2], 0.22 * 40.0)
