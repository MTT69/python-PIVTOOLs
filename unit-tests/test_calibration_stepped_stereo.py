"""S3 — stepped (dual-level) pinhole STEREO + transmission-via-axes recovery tests.

Three things are validated, each non-circular:

1. The handedness classifier (``classify_stereo_config``). Pure projection geometry:
   the 2D cross product of the clicked (origin,+X,+Y) axis vectors keeps its sign
   under camera roll/tilt on one side of the board and FLIPS when the camera views
   the opposite face. Equal sign => same_side, opposite => transmission. This is the
   synthetic-mirror validation the plan asks for, at the unit level — it rests only
   on cv2.projectPoints and a sign, not on the calibrator.

2. Same-side end-to-end recovery. Cam2 is cam1 rotated about the board centroid by a
   KNOWN ``R_extra`` (so the board stays framed and R_stereo/T_stereo have closed-form
   ground truth). The orchestrator must auto-classify same_side, recover that
   transform, fit sub-pixel, and reconstruct a prescribed 3C velocity. This is the
   DaVis pinhole5 same-side bar's synthetic stand-in.

3. Record save/load round-trip.

The full TRANSMISSION Z-recovery (cam2's clicked level shifted by ~board thickness in
the shared frame) rests on ``compute_z_and_offsets``'s physical two-face model, ported
verbatim from v1; its numeric validation is the user's on-data / DaVis-synthetic check
(cannot run python near ~/Downloads, and a hand-built two-face render would be an
unverified verifier). Here the transmission PATH is exercised only where it is
non-circular: the classifier's mirror flip (test 1) and the explicit-override plumbing.

Run with --make-figures to drop a diagnostic PNG in PyPIVTools/figures/debug/.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from pivtools_cli.synthetic_calibration_common import make_camera_matrix
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams
from pivtools_gui.calibration.detection.stepped_levels import SteppedBoardSpec
from pivtools_gui.calibration.record import load_stereo, save_stereo
from pivtools_gui.calibration.stepped_calibrate import (
    _click_chirality,
    calibrate_stepped_stereo,
    classify_stereo_config,
)
from pivtools_gui.calibration.stereo_model import reconstruct_3c_at_points

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "debug"

SPACING_MM = 15.0
STEP_MM = 3.0
OFFSET_MM = SPACING_MM / 2.0
PEAK_COLS, PEAK_ROWS = 9, 9
TROUGH_COLS, TROUGH_ROWS = 8, 8
W, H = 1280, 1024
FILL = 0.40

# Distinct board orientations for intrinsic conditioning; index 0 = datum. Kept
# gentle (<=8 deg): the synthetic dot render starts dropping/merging dots past
# ~25 deg combined obliquity (pose tilt + the cam2 stereo angle below), and a
# partial detection mis-indexes the grid — a render limit, not a calibrator one
# (S1 detection robustness is tested separately; real oblique data is the user's
# transmission set). Here we want a clean pair to exercise the stereo orchestrator.
POSE_RVECS = [
    (0.04, 0.02, 0.00),
    (0.12, -0.06, 0.03),
    (-0.10, 0.11, -0.02),
    (0.06, 0.13, 0.02),
    (-0.12, -0.07, 0.02),
]
# Cam2 = cam1 rotated about the board centroid by this (≈14.3 deg about Y). Keeps
# the board framed in cam2 and makes R_stereo = R_extra, T_stereo = (I-R_extra)·d_datum.
RVEC_EXTRA = (0.0, 0.25, 0.0)


# ---------------------------------------------------------------------------
# Synthetic stepped board render (peak z=0, trough z=-step, +half-spacing xy)
# ---------------------------------------------------------------------------


def _board_points_mm():
    peak, trough = [], []
    for r in range(PEAK_ROWS):
        for c in range(PEAK_COLS):
            peak.append([c * SPACING_MM, r * SPACING_MM, 0.0])
    for r in range(TROUGH_ROWS):
        for c in range(TROUGH_COLS):
            trough.append(
                [c * SPACING_MM + OFFSET_MM, r * SPACING_MM + OFFSET_MM, -STEP_MM]
            )
    return np.array(peak, np.float64), np.array(trough, np.float64)


def _allpts():
    peak, trough = _board_points_mm()
    return np.vstack([peak, trough])


def _cam1_pose(rvec, K):
    """The (R, t) cam1 uses for a pose: board centroid placed on the optical axis."""
    fx = float(K[0, 0])
    allpts = _allpts()
    centre = allpts.mean(axis=0)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    board_w = (PEAK_COLS - 1) * SPACING_MM
    Z = fx * board_w / (FILL * W)
    t = np.array([0.0, 0.0, Z]) - R @ centre
    return R, t.reshape(3), Z, centre


def _render(R, t, K):
    """Project both levels through an explicit (R, t) pose and draw filled dots."""
    allpts = _allpts()
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    proj = cv2.projectPoints(
        allpts, rvec, np.asarray(t, np.float64).reshape(3), K, np.zeros(5)
    )[0].reshape(-1, 2)
    nn = cKDTree(proj).query(proj, k=2)[0][:, 1]
    radii = np.maximum(np.round(0.22 * nn).astype(int), 2)
    img = np.zeros((H, W), np.uint8)
    for (cx, cy), rad in zip(proj, radii):
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < W and 0 <= iy < H:
            cv2.circle(img, (ix, iy), int(rad), 255, -1)
    return img, proj


def _cam2_pose(R1, t1, Z, R_extra):
    """Cam1 pose rotated about the board centroid (centroid at [0,0,Z] in cam coords)."""
    d1 = np.array([0.0, 0.0, Z])
    R2 = R_extra @ R1
    t2 = R_extra @ (t1 - d1) + d1
    return R2, t2


def _fiducials(proj):
    """Datum fiducials: peak (0,0)=origin, (1,0)=+X, (0,1)=+Y."""
    return {
        "origin": proj[0].tolist(),
        "x_axis": proj[1].tolist(),
        "y_axis": proj[PEAK_COLS].tolist(),
    }


def _level_a_label(detection, proj, gt_z):
    """Physical level ('peak'|'trough') of detected level A, from ground truth.

    Stands in for the GUI per-pose click-to-label step (the parity-based A/B split is
    per-pose arbitrary — see the S2 mono test).
    """
    a_full = detection.level_data["a"]
    centers = np.asarray(a_full["centers"], dtype=np.float64)
    idx = cKDTree(proj).query(centers)[1]
    return "peak" if np.median(gt_z[idx]) > -STEP_MM / 2.0 else "trough"


def _gt_z():
    n_peak = PEAK_COLS * PEAK_ROWS
    return np.concatenate(
        [np.zeros(n_peak), np.full(TROUGH_COLS * TROUGH_ROWS, -STEP_MM)]
    )


def _figpath(slug: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.date(2026, 6, 6).isoformat()
    return FIG_DIR / f"{today}-calib2-{slug}.png"


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World->cam (R, t), OpenCV convention (+z_cam toward target, +y_cam image-down)."""
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.vstack([r, d, f])
    t = -R @ eye
    return R, t.reshape(3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def same_side_scene():
    """A same-side stereo pair with closed-form ground-truth R_stereo / T_stereo."""
    K = make_camera_matrix(W, H)
    R_extra, _ = cv2.Rodrigues(np.asarray(RVEC_EXTRA, np.float64))

    imgs1, projs1, imgs2, projs2 = [], [], [], []
    R1d = t1d = Zd = None
    for i, rv in enumerate(POSE_RVECS):
        R1, t1, Z, _centre = _cam1_pose(rv, K)
        im1, pr1 = _render(R1, t1, K)
        R2, t2 = _cam2_pose(R1, t1, Z, R_extra)
        im2, pr2 = _render(R2, t2, K)
        imgs1.append(im1)
        projs1.append(pr1)
        imgs2.append(im2)
        projs2.append(pr2)
        if i == 0:
            R1d, t1d, Zd = R1, t1, Z

    det = SteppedDetector(
        SteppedParams(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    )
    detections1 = [det.detect(im) for im in imgs1]
    detections2 = [det.detect(im) for im in imgs2]

    gt_z = _gt_z()
    pose_levels1 = [_level_a_label(d, p, gt_z) for d, p in zip(detections1, projs1)]
    pose_levels2 = [_level_a_label(d, p, gt_z) for d, p in zip(detections2, projs2)]

    # Closed-form ground truth (see _cam2_pose derivation).
    R_stereo_gt = R_extra
    T_stereo_gt = (np.eye(3) - R_extra) @ np.array([0.0, 0.0, Zd])

    return {
        "K": K,
        "imgs1": imgs1,
        "imgs2": imgs2,
        "detections1": detections1,
        "detections2": detections2,
        "fiducials1": _fiducials(projs1[0]),
        "fiducials2": _fiducials(projs2[0]),
        "pose_levels1": pose_levels1,
        "pose_levels2": pose_levels2,
        "R_stereo_gt": R_stereo_gt,
        "T_stereo_gt": T_stereo_gt,
    }


# ---------------------------------------------------------------------------
# 1. Handedness classifier — pure geometry (the synthetic-mirror validation)
# ---------------------------------------------------------------------------


def test_classifier_chirality_flips_across_the_board_plane():
    """Click chirality is preserved on one side, flips across to the other face."""
    K = make_camera_matrix(W, H)
    pts = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
    )  # origin,+X,+Y

    def clicks_from(R, t):
        rvec, _ = cv2.Rodrigues(R)
        px = cv2.projectPoints(pts, rvec, t.reshape(3), K, np.zeros(5))[0].reshape(
            -1, 2
        )
        return {"origin": px[0], "x_axis": px[1], "y_axis": px[2]}

    front_a = clicks_from(*_look_at([20, 10, 150], [0, 0, 0]))  # +Z side
    front_b = clicks_from(*_look_at([-30, 40, 180], [0, 0, 0]))  # +Z side, rolled
    back = clicks_from(*_look_at([10, -20, -160], [0, 0, 0]))  # -Z side (mirror)

    # Same side => identical sign => same_side; opposite faces => flip => transmission.
    assert _click_chirality(front_a) == _click_chirality(front_b)
    assert _click_chirality(front_a) != _click_chirality(back)
    assert classify_stereo_config(front_a, front_b) == "same_side"
    assert classify_stereo_config(front_a, back) == "transmission"
    assert classify_stereo_config(back, front_a) == "transmission"  # symmetric


def test_classifier_rejects_collinear_clicks():
    """Near-collinear +X/+Y clicks give an untrustworthy sign -> hard error."""
    bad = {"origin": [100.0, 100.0], "x_axis": [200.0, 100.0], "y_axis": [260.0, 100.3]}
    with pytest.raises(ValueError, match="ambiguous|collinear"):
        _click_chirality(bad)


# ---------------------------------------------------------------------------
# 2. Same-side end-to-end recovery
# ---------------------------------------------------------------------------


def _angle_between(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def test_stepped_stereo_same_side_recovery(same_side_scene, request):
    s = same_side_scene
    assert all(d.success for d in s["detections1"]), "a cam1 pose failed detection"
    assert all(d.success for d in s["detections2"]), "a cam2 pose failed detection"

    board = SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    rec = calibrate_stepped_stereo(
        detections1=s["detections1"],
        detections2=s["detections2"],
        fiducials1=s["fiducials1"],
        fiducials2=s["fiducials2"],
        clicked_level1="peak",
        clicked_level2="peak",
        pose_levels1=s["pose_levels1"],
        pose_levels2=s["pose_levels2"],
        board=board,
        image_size1=(W, H),
        image_size2=(W, H),
        cam1=1,
        cam2=2,
        datum_index=0,
        stereo_config="auto",
    )

    # Auto-classified same_side from the click handedness.
    assert rec.board_meta["stereo_config"] == "same_side"
    assert rec.board_type == "stepped"

    # Parity board_meta (Phase 4): compose method + view count + per-camera diagnostics.
    assert rec.board_meta["stereo_method"] == "compose"
    assert rec.board_meta["n_stereo_views"] >= 1
    assert set(rec.board_meta["view_diagnostics"]) == {"cam1", "cam2"}

    # Both cameras fit sub-pixel on noise-free renders.
    assert rec.model1.rms < 1.0 and rec.model2.rms < 1.0, (
        rec.model1.rms,
        rec.model2.rms,
    )
    assert max(rec.per_view_rms1) < 1.0 and max(rec.per_view_rms2) < 1.0

    # Stereo transform recovered against closed-form ground truth.
    ang = _angle_between(rec.R_stereo, s["R_stereo_gt"])
    assert ang < 0.5, f"R_stereo off by {ang:.3f} deg"
    t_err = float(np.linalg.norm(rec.T_stereo.reshape(3) - s["T_stereo_gt"]))
    assert t_err < 2.0, f"T_stereo off by {t_err:.3f} mm (gt {s['T_stereo_gt']})"

    # The point of stereo: reconstruct a prescribed 3C velocity at the board points.
    wp = _allpts()  # already in the shared frame (cam1 origin = peak (0,0))
    vel = np.array([0.30, -0.20, 0.45])  # mm/frame, uniform
    m1, m2 = rec.model1, rec.model2
    d1 = m1.project(wp + vel) - m1.project(wp)
    d2 = m2.project(wp + vel) - m2.project(wp)
    vr = reconstruct_3c_at_points(m1, m2, wp, d1, d2)
    assert (
        np.nanmax(np.abs(vr - vel)) < 0.02
    ), f"3C recon err {np.nanmax(np.abs(vr - vel)):.4f}"

    if request.config.getoption("--make-figures", default=False):
        _make_figure(s, rec, vel, vr)


def test_stepped_stereo_explicit_config_overrides_classifier(same_side_scene):
    """An explicit stereo_config is honoured verbatim (no auto-classify)."""
    s = same_side_scene
    board = SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    rec = calibrate_stepped_stereo(
        detections1=s["detections1"],
        detections2=s["detections2"],
        fiducials1=s["fiducials1"],
        fiducials2=s["fiducials2"],
        clicked_level1="peak",
        clicked_level2="peak",
        pose_levels1=s["pose_levels1"],
        pose_levels2=s["pose_levels2"],
        board=board,
        image_size1=(W, H),
        image_size2=(W, H),
        stereo_config="same_side",
    )
    assert rec.board_meta["stereo_config"] == "same_side"

    with pytest.raises(ValueError, match="stereo_config must be"):
        calibrate_stepped_stereo(
            detections1=s["detections1"],
            detections2=s["detections2"],
            fiducials1=s["fiducials1"],
            fiducials2=s["fiducials2"],
            clicked_level1="peak",
            clicked_level2="peak",
            pose_levels1=s["pose_levels1"],
            pose_levels2=s["pose_levels2"],
            board=board,
            image_size1=(W, H),
            image_size2=(W, H),
            stereo_config="nonsense",
        )


# ---------------------------------------------------------------------------
# 3. Record round-trip
# ---------------------------------------------------------------------------


def test_stepped_stereo_record_round_trips(same_side_scene, tmp_path):
    s = same_side_scene
    board = SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM)
    rec = calibrate_stepped_stereo(
        detections1=s["detections1"],
        detections2=s["detections2"],
        fiducials1=s["fiducials1"],
        fiducials2=s["fiducials2"],
        clicked_level1="peak",
        clicked_level2="peak",
        pose_levels1=s["pose_levels1"],
        pose_levels2=s["pose_levels2"],
        board=board,
        image_size1=(W, H),
        image_size2=(W, H),
    )
    path = save_stereo(rec, tmp_path / "stereo")
    reloaded = load_stereo(path)

    assert reloaded.board_type == "stepped"
    np.testing.assert_allclose(reloaded.R_stereo, rec.R_stereo, atol=1e-9)
    np.testing.assert_allclose(reloaded.T_stereo, rec.T_stereo, atol=1e-9)
    np.testing.assert_allclose(reloaded.model1.K, rec.model1.K, rtol=1e-9)
    np.testing.assert_allclose(reloaded.model2.K, rec.model2.K, rtol=1e-9)
    assert reloaded.board_meta["stereo_config"] == "same_side"
    assert float(reloaded.board_meta["step_height_mm"]) == STEP_MM


# ---------------------------------------------------------------------------
# Debug figure
# ---------------------------------------------------------------------------


def _make_figure(scene, rec, vel, vr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    m1, m2 = rec.model1, rec.model2
    c1 = (-m1.R.T @ m1.t).ravel()  # camera centres in the world frame
    c2 = (-m2.R.T @ m2.t).ravel()
    _allpts()

    fig = plt.figure(figsize=(18, 6))

    # (1) rig geometry: board + the two camera centres / optical axes.
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    peak, trough = _board_points_mm()
    ax0.scatter(peak[:, 0], peak[:, 1], peak[:, 2], s=6, c="tab:blue", label="peak")
    ax0.scatter(
        trough[:, 0], trough[:, 1], trough[:, 2], s=6, c="tab:red", label="trough"
    )
    for c, m, name, col in ((c1, m1, "cam1", "k"), (c2, m2, "cam2", "tab:green")):
        ax0.scatter(*c, s=80, marker="^", c=col)
        ax0.text(c[0], c[1], c[2], name, color=col)
        axis = m.R.T @ np.array([0.0, 0.0, 1.0])
        tip = c - axis * 0.4 * np.linalg.norm(
            c
        )  # axis points world->cam; draw toward board
        ax0.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]], col, lw=1)
    ax0.set_title(f"rig geometry ({rec.board_meta['stereo_config']})")
    ax0.set_xlabel("X [mm]")
    ax0.set_ylabel("Y [mm]")
    ax0.set_zlabel("Z [mm]")
    ax0.legend(loc="upper left")

    # (2) per-camera reprojection RMS.
    ax1 = fig.add_subplot(1, 3, 2)
    x1 = np.arange(len(rec.per_view_rms1))
    x2 = np.arange(len(rec.per_view_rms2))
    ax1.bar(
        x1 - 0.2, rec.per_view_rms1, width=0.4, label=f"cam1 ({m1.rms:.3f})", color="k"
    )
    ax1.bar(
        x2 + 0.2,
        rec.per_view_rms2,
        width=0.4,
        label=f"cam2 ({m2.rms:.3f})",
        color="tab:green",
    )
    ax1.set_title("per-view reprojection RMS")
    ax1.set_xlabel("pose")
    ax1.set_ylabel("RMS [px]")
    ax1.legend()

    # (3) numeric summary incl. recovered 3C velocity vs prescribed.
    ang = _angle_between(rec.R_stereo, scene["R_stereo_gt"])
    t_err = float(np.linalg.norm(rec.T_stereo.reshape(3) - scene["T_stereo_gt"]))
    recon_err = float(np.nanmax(np.abs(vr - vel)))
    txt = (
        f"config        = {rec.board_meta['stereo_config']}\n"
        f"R_stereo err  = {ang:.3f} deg\n"
        f"baseline fit  = {rec.board_meta['baseline_mm']:.2f} mm\n"
        f"baseline gt   = {np.linalg.norm(scene['T_stereo_gt']):.2f} mm\n"
        f"|T err|       = {t_err:.3f} mm\n"
        f"opt-axis ang  = {rec.board_meta['relative_angle_deg']:.2f} deg\n"
        f"cam1/2 RMS    = {m1.rms:.3f} / {m2.rms:.3f} px\n"
        f"3C vel gt     = [{vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}] mm/fr\n"
        f"3C recon max e= {recon_err:.4f} mm/fr"
    )
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.axis("off")
    ax2.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=12)
    ax2.set_title("stereo fit + 3C reconstruction")

    fig.tight_layout()
    fig.savefig(_figpath("stepped-stereo-fit"), dpi=110)
    plt.close(fig)
