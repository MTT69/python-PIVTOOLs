"""S2 — stepped (dual-level) pinhole MONO calibrator recovery tests.

Renders multiple poses of a synthetic stepped board (peak grid at z=0, interleaved
trough grid at z=-step, +half-spacing in x,y) through a KNOWN pinhole K, then runs
``calibrate_stepped_mono`` and checks it recovers that K, fits sub-pixel, anchors
the world frame to the datum fiducial clicks, and round-trips through save/load.

The DaVis numeric bar (per-cam RMS ~0.42/0.51 px on the real plate) is an on-data
check handed to the user (cannot run python near ~/Downloads); here we validate the
fit machinery on noise-free synthetic ground truth.

Run with --make-figures to drop a diagnostic PNG in PyPIVTools/figures/debug/.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import pytest

from pivtools_cli.synthetic_calibration_common import make_camera_matrix
from pivtools_gui.calibration2.camera_model import CameraModel
from pivtools_gui.calibration2.detection.stepped import SteppedDetector, SteppedParams
from pivtools_gui.calibration2.detection.stepped_levels import SteppedBoardSpec
from pivtools_gui.calibration2.record import load_mono, save_mono
from pivtools_gui.calibration2.stepped_calibrate import calibrate_stepped_mono

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "debug"

SPACING_MM = 15.0
STEP_MM = 3.0
OFFSET_MM = SPACING_MM / 2.0
PEAK_COLS, PEAK_ROWS = 9, 9        # 81 peak dots (the larger level)
TROUGH_COLS, TROUGH_ROWS = 8, 8    # 64 trough dots

# Distinct board orientations for good intrinsic conditioning; index 0 = datum.
POSE_RVECS = [
    (0.05, 0.03, 0.00),
    (0.20, -0.10, 0.05),
    (-0.16, 0.18, -0.04),
    (0.10, 0.22, 0.02),
    (-0.22, -0.12, 0.03),
]
W, H = 1280, 1024


def _board_points_mm():
    """Peak + trough 3D points in mm; peak (col=0,row=0) is at the origin."""
    peak, trough = [], []
    for r in range(PEAK_ROWS):
        for c in range(PEAK_COLS):
            peak.append([c * SPACING_MM, r * SPACING_MM, 0.0])
    for r in range(TROUGH_ROWS):
        for c in range(TROUGH_COLS):
            trough.append([c * SPACING_MM + OFFSET_MM, r * SPACING_MM + OFFSET_MM, -STEP_MM])
    return np.array(peak, np.float64), np.array(trough, np.float64)


def _render_pose(rvec, K, fill=0.62):
    """Project both levels through one pinhole pose (mm object pts) and draw dots."""
    fx = float(K[0, 0])
    peak, trough = _board_points_mm()
    allpts = np.vstack([peak, trough])
    centre = allpts.mean(axis=0)

    rvec = np.asarray(rvec, np.float64)
    R, _ = cv2.Rodrigues(rvec)
    board_w = (PEAK_COLS - 1) * SPACING_MM
    Z = fx * board_w / (fill * W)
    tvec = np.array([0.0, 0.0, Z]) - R @ centre

    dist = np.zeros(5)
    proj = cv2.projectPoints(allpts, rvec, tvec, K, dist)[0].reshape(-1, 2)

    from scipy.spatial import cKDTree
    tree = cKDTree(proj)
    nn = tree.query(proj, k=2)[0][:, 1]
    radii = np.maximum(np.round(0.22 * nn).astype(int), 2)

    img = np.zeros((H, W), np.uint8)
    for (cx, cy), rad in zip(proj, radii):
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(img, (ix, iy), int(rad), 255, -1)
    return img, proj


def _figpath(slug: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.date(2026, 6, 6).isoformat()
    return FIG_DIR / f"{today}-calib2-{slug}.png"


def _level_a_label(detection, proj, gt_z):
    """Physical level ('peak'|'trough') of detected level A, from ground truth.

    The parity-based A/B split is per-pose arbitrary — which physical level lands on
    the topmost detected row flips with board tilt. In production the GUI click-to-
    label step supplies this per-pose label; here the synthetic ground truth stands
    in for that human/GUI knowledge.
    """
    from scipy.spatial import cKDTree

    a_full = detection.diagnostics["_level_a_full"]
    centers = np.asarray(a_full["centers"], dtype=np.float64)
    idx = cKDTree(proj).query(centers)[1]
    return "peak" if np.median(gt_z[idx]) > -STEP_MM / 2.0 else "trough"


@pytest.fixture(scope="module")
def stepped_mono_scene():
    K_true = make_camera_matrix(W, H)
    images, projs = [], []
    for rv in POSE_RVECS:
        img, proj = _render_pose(rv, K_true)
        images.append(img)
        projs.append(proj)
    # Datum (pose 0) fiducials: peak (0,0)=origin, (col1,row0)=+X, (col0,row1)=+Y.
    datum_proj = projs[0]
    fiducials = {
        "origin": datum_proj[0].tolist(),
        "x_axis": datum_proj[1].tolist(),
        "y_axis": datum_proj[PEAK_COLS].tolist(),
    }
    det = SteppedDetector(SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM))
    detections = [det.detect(im) for im in images]

    # Ground-truth z per projected point (peak block z=0, trough block z=-step),
    # used to label each pose's level A — the synthetic analogue of GUI click-to-label.
    n_peak = PEAK_COLS * PEAK_ROWS
    gt_z = np.concatenate([np.zeros(n_peak), np.full(TROUGH_COLS * TROUGH_ROWS, -STEP_MM)])
    pose_levels = [_level_a_label(d, p, gt_z) for d, p in zip(detections, projs)]
    return K_true, images, detections, fiducials, pose_levels


def test_stepped_mono_recovers_intrinsics_and_frame(stepped_mono_scene, request):
    K_true, images, detections, fiducials, pose_levels = stepped_mono_scene
    assert all(d.success for d in detections), "a synthetic pose failed detection"

    board = SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    record = calibrate_stepped_mono(
        detections=detections,
        fiducials=fiducials,
        clicked_level="peak",
        pose_levels=pose_levels,  # per-pose level-A label (parity flips with tilt)
        board=board,
        image_size=(W, H),
        camera=1,
        datum_index=0,
    )

    cam = record.camera_model
    assert isinstance(cam, CameraModel)

    # Intrinsics recovered (noise-free synthetic -> tight).
    fx_true = float(K_true[0, 0])
    fx_fit = float(cam.K[0, 0])
    assert abs(fx_fit - fx_true) / fx_true < 0.03, f"fx {fx_fit:.1f} vs true {fx_true:.1f}"
    assert abs(cam.K[0, 2] - W / 2.0) < 0.06 * W, f"cx off: {cam.K[0, 2]:.1f}"
    assert abs(cam.K[1, 2] - H / 2.0) < 0.06 * H, f"cy off: {cam.K[1, 2]:.1f}"

    # Sub-pixel fit on noise-free renders.
    assert cam.rms < 1.0, f"overall RMS too high: {cam.rms:.3f}px"
    assert max(record.per_view_rms) < 1.0, f"per-view RMS too high: {record.per_view_rms}"

    # World frame anchored to the datum clicks: origin click back-projects to (0,0,0).
    origin_world = cam.back_project_to_plane(
        np.array([fiducials["origin"]]), z_world=0.0
    )[0]
    assert np.linalg.norm(origin_world) < SPACING_MM / 8.0, f"origin world {origin_world}"

    # +X click lands along +X at one spacing, +Y click along +Y.
    x_world = cam.back_project_to_plane(np.array([fiducials["x_axis"]]), z_world=0.0)[0]
    y_world = cam.back_project_to_plane(np.array([fiducials["y_axis"]]), z_world=0.0)[0]
    assert x_world[0] > SPACING_MM / 2.0 and abs(x_world[1]) < SPACING_MM / 4.0, x_world
    assert y_world[1] > SPACING_MM / 2.0 and abs(y_world[0]) < SPACING_MM / 4.0, y_world

    if request.config.getoption("--make-figures", default=False):
        _make_figure(K_true, images, detections, record, fiducials)


def test_stepped_mono_round_trips_through_record(stepped_mono_scene, tmp_path):
    _K, _imgs, detections, fiducials, pose_levels = stepped_mono_scene
    board = SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    record = calibrate_stepped_mono(
        detections=detections, fiducials=fiducials, clicked_level="peak",
        pose_levels=pose_levels, board=board,
        image_size=(W, H), camera=1, datum_index=0,
    )
    path = save_mono(record, tmp_path / "model")
    reloaded = load_mono(path)

    assert reloaded.board_type == "stepped"
    assert reloaded.world_frame.mode == "clicks"
    np.testing.assert_allclose(reloaded.camera_model.K, record.camera_model.K, rtol=1e-9)
    np.testing.assert_allclose(reloaded.camera_model.R, record.camera_model.R, atol=1e-9)
    assert reloaded.board_meta["clicked_level"] == "peak"
    assert float(reloaded.board_meta["step_height_mm"]) == STEP_MM


def test_stepped_mono_rejects_label_conflict(stepped_mono_scene):
    """Datum pose_level conflicting with the fiducial-derived label is a hard error."""
    _K, _imgs, detections, fiducials, pose_levels = stepped_mono_scene
    board = SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    # Flip the datum's level-A label so it contradicts the fiducial-derived label.
    bad = list(pose_levels)
    bad[0] = "trough" if pose_levels[0] == "peak" else "peak"
    with pytest.raises(ValueError, match="conflicts with the fiducial-derived"):
        calibrate_stepped_mono(
            detections=detections, fiducials=fiducials, clicked_level="peak",
            pose_levels=bad,
            board=board, image_size=(W, H), camera=1, datum_index=0,
        )


def _make_figure(K_true, images, detections, record, fiducials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam = record.camera_model
    # Reproject the datum's detected dots through the fitted model for residuals.
    datum = detections[0]
    ip = datum.image_points
    labels = np.asarray(datum.diagnostics["level_labels"])

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    a = labels == "A"
    ax[0].imshow(images[0], cmap="gray", origin="upper")
    ax[0].scatter(ip[a, 0], ip[a, 1], s=12, facecolors="none", edgecolors="tab:blue", label="peak (A)")
    ax[0].scatter(ip[~a, 0], ip[~a, 1], s=12, facecolors="none", edgecolors="tab:red", label="trough (B)")
    for key, mk in (("origin", "P0"), ("x_axis", "+X"), ("y_axis", "+Y")):
        px = fiducials[key]
        ax[0].plot(px[0], px[1], "y*", ms=14)
        ax[0].annotate(mk, (px[0], px[1]), color="yellow", fontsize=9)
    ax[0].set_title("datum: detection + fiducials")
    ax[0].set_xlabel("x [px]"); ax[0].set_ylabel("y [px]"); ax[0].legend(loc="upper right")

    ax[1].bar(range(len(record.per_view_rms)), record.per_view_rms, color="tab:green")
    ax[1].axhline(cam.rms, color="k", ls="--", label=f"overall {cam.rms:.3f}px")
    ax[1].set_title("per-view reprojection RMS")
    ax[1].set_xlabel("pose"); ax[1].set_ylabel("RMS [px]"); ax[1].legend()

    txt = (
        f"fx fit  = {cam.K[0,0]:.1f}\nfx true = {K_true[0,0]:.1f}\n"
        f"cx,cy   = {cam.K[0,2]:.1f}, {cam.K[1,2]:.1f}\n"
        f"image   = {W}x{H}\nRMS     = {cam.rms:.4f} px\n"
        f"n views = {record.board_meta['n_views']}\n"
        f"clicked = {record.board_meta['clicked_level']}\n"
        f"frame   = {record.world_frame.mode} "
        f"(swap={int(record.world_frame.swap_axes)}, "
        f"col={record.world_frame.col_sign}, row={record.world_frame.row_sign})"
    )
    ax[2].axis("off")
    ax[2].text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=12)
    ax[2].set_title("fitted model")

    fig.tight_layout()
    fig.savefig(_figpath("stepped-mono-fit"), dpi=110)
    plt.close(fig)
