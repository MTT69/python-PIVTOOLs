"""Stepped calibrator: clicked-frame handedness, physical level labels, reprojection gate.

Three 2026-10-22 changes to ``stepped_calibrate`` (fix/calibration-p7):

1. ``compute_z_and_offsets`` keys camera 2's transmission heights by its PHYSICAL label
   ('peak' = raised toward camera 2 = the back outer plane at -thickness, 'trough' = the
   pockets at -(thickness - step)), not by which one was clicked. Before, a camera 2
   origin on a back-face peak was placed one step wrong (Andre, cam2 RMS 40 px).
2. The sign of the clicked frame's +Z (= +X x +Y) is READ from camera 1's click
   handedness (``z_toward_camera_from_clicks``) and honoured: the level table is negated
   when +Z points into the board. Before, the trough was hard-wired to -step along +Z,
   which is only "farther from the camera" when +Z points at the camera, so +Y clicked
   below the origin gave a sub-pixel fit with the wrong baseline (Andre, -23.6 mm).
3. A reprojection gate after the pinhole fit removes dots above
   max(2 px, 4 x median residual) and refits (Zefanya: three two-tone edge dots at
   10.7 / 14.7 / 3.1 px moved fx by 0.9%; the gate removes those and nothing else).

The same-side scene is the sibling stereo test's; the rig recovery tests use the
synthetic generator so transmission (two faces, real thickness) is covered too.
"""

# ruff: noqa: F811  (same_side_scene is the sibling module's fixture, imported by name)
from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_calibration_stepped_stereo import same_side_scene  # noqa: F401  (fixture)
from test_calibration_stepped_stereo import (  # noqa: E402  (sibling-test reuse)
    PEAK_COLS,
    PEAK_ROWS,
    SPACING_MM,
    STEP_MM,
    TROUGH_COLS,
    H,
    W,
    _angle_between,
)

from pivtools_cli import generate_synthetic_stepped as gstep
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams
from pivtools_gui.calibration.detection.stepped_levels import SteppedBoardSpec
from pivtools_gui.calibration.stepped_calibrate import (
    calibrate_stepped_stereo,
    compute_z_and_offsets,
    z_toward_camera_from_clicks,
)

BOARD = SteppedBoardSpec(
    dot_spacing_mm=15.0, step_height_mm=3.0, board_thickness_mm=14.8
)
N_PEAK = PEAK_COLS * PEAK_ROWS


# ---------------------------------------------------------------------------
# 1 + 2. The height table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cam2_clicked", ["peak", "trough"])
def test_transmission_heights_follow_cam2_label_not_its_click(cam2_clicked):
    """Camera 2's back-face levels sit where the plate puts them whatever it clicked."""
    geo = compute_z_and_offsets("transmission", "peak", cam2_clicked, BOARD)
    assert geo["Cam1"]["z"] == {"peak": 0.0, "trough": -3.0}
    assert geo["Cam2"]["z"] == {"trough": -(14.8 - 3.0), "peak": -14.8}
    # The click still decides which level carries the in-plane half-pitch offset.
    other = "trough" if cam2_clicked == "peak" else "peak"
    assert geo["Cam2"]["xy_offset"] == {cam2_clicked: 0.0, other: BOARD.level_offset_mm}


def test_trough_origin_shifts_the_whole_table_by_one_step():
    geo = compute_z_and_offsets("transmission", "trough", "peak", BOARD)
    assert geo["Cam1"]["z"] == {"trough": 0.0, "peak": 3.0}
    assert geo["Cam2"]["z"] == {"trough": -(14.8 - 3.0) + 3.0, "peak": -14.8 + 3.0}


@pytest.mark.parametrize("config", ["same_side", "transmission"])
def test_z_into_the_board_negates_every_height(config):
    toward = compute_z_and_offsets(config, "peak", "peak", BOARD, z_toward_cam1=True)
    away = compute_z_and_offsets(config, "peak", "peak", BOARD, z_toward_cam1=False)
    for cam in ("Cam1", "Cam2"):
        for lvl in ("peak", "trough"):
            assert away[cam]["z"][lvl] == -toward[cam]["z"][lvl]
        assert away[cam]["xy_offset"] == toward[cam]["xy_offset"]


def test_z_direction_is_read_from_the_click_handedness():
    """Image pixels are y-down: +Y clicked ABOVE the origin -> +Z toward the camera."""
    o = (100.0, 100.0)
    assert z_toward_camera_from_clicks(
        {"origin": o, "x_axis": (200, 100), "y_axis": (100, 20)}
    )
    assert not z_toward_camera_from_clicks(
        {"origin": o, "x_axis": (200, 100), "y_axis": (100, 180)}
    )
    # Rolling the camera does not change the answer; mirroring (the other face) does.
    assert z_toward_camera_from_clicks(
        {"origin": o, "x_axis": (100, 20), "y_axis": (0, 100)}
    )


# ---------------------------------------------------------------------------
# 2. Same-side scene: the rig does not depend on how the frame was clicked
# ---------------------------------------------------------------------------


def _solve(s, fiducials1, fiducials2, level1, level2, detections2=None, n_poses=None):
    d1 = list(s["detections1"])[:n_poses]
    d2 = list(detections2 if detections2 is not None else s["detections2"])[:n_poses]
    return calibrate_stepped_stereo(
        detections1=d1,
        detections2=d2,
        fiducials1=fiducials1,
        fiducials2=fiducials2,
        clicked_level1=level1,
        clicked_level2=level2,
        pose_levels1=list(s["pose_levels1"])[:n_poses],
        pose_levels2=list(s["pose_levels2"])[:n_poses],
        board=SteppedBoardSpec(dot_spacing_mm=SPACING_MM, step_height_mm=STEP_MM),
        image_size1=(W, H),
        image_size2=(W, H),
        datum_index=0,
        stereo_config="auto",
    )


def _peak_fiducials(proj, c=1, r=1, y_up=False):
    """Clicks on the peak lattice at (c, r); +Y one row down the image, or one row up."""
    i = r * PEAK_COLS + c
    j = (r - 1 if y_up else r + 1) * PEAK_COLS + c
    return {
        "origin": proj[i].tolist(),
        "x_axis": proj[i + 1].tolist(),
        "y_axis": proj[j].tolist(),
    }


def _trough_fiducials(proj):
    """Clicks on the trough lattice: trough (0,0) origin, (1,0) +X, (0,1) +Y."""
    t = lambda c, r: N_PEAK + r * TROUGH_COLS + c  # noqa: E731
    return {
        "origin": proj[t(0, 0)].tolist(),
        "x_axis": proj[t(1, 0)].tolist(),
        "y_axis": proj[t(0, 1)].tolist(),
    }


def _camera_centre_z(model) -> float:
    return float((-np.asarray(model.R).T @ np.asarray(model.t).reshape(3))[2])


def test_y_click_direction_changes_the_frame_not_the_rig(
    same_side_scene,
):
    """+Y clicked above or below the origin: same R_stereo / T_stereo, mirrored +Z."""
    s = same_side_scene
    p1, p2 = s["projs1"][0], s["projs2"][0]
    down = _solve(s, _peak_fiducials(p1), _peak_fiducials(p2), "peak", "peak")
    up = _solve(
        s,
        _peak_fiducials(p1, y_up=True),
        _peak_fiducials(p2, y_up=True),
        "peak",
        "peak",
    )
    # The relative pose lives in camera coordinates, so it cannot see the world frame.
    assert _angle_between(down.R_stereo, up.R_stereo) < 1e-3
    np.testing.assert_allclose(down.T_stereo, up.T_stereo, atol=1e-3)
    assert down.model1.rms == pytest.approx(up.model1.rms, abs=1e-6)
    assert down.model2.rms == pytest.approx(up.model2.rms, abs=1e-6)
    # The frames are mirror images in Z, and the record says which way +Z points.
    assert down.board_meta["world_z_toward_camera"] == -1.0
    assert up.board_meta["world_z_toward_camera"] == 1.0
    assert _camera_centre_z(down.model1) < 0 < _camera_centre_z(up.model1)
    # Same physical rig either way: the camera is the same distance from the board.
    assert abs(_camera_centre_z(down.model1)) == pytest.approx(
        abs(_camera_centre_z(up.model1)), rel=1e-3
    )


def test_trough_origin_gives_the_same_rig(
    same_side_scene,
):
    """Clicking the datum on the recessed lattice (label 'trough') is a frame change only.

    All five poses. Until 2026-10-22 the non-datum builder applied the half-pitch
    offset by label while the stitch had applied it by role, so a trough-lattice
    datum put the two levels a full pitch apart in every non-datum pose (75 px RMS,
    fx 301 for 1280); the offset now follows the stitch roles.
    """
    s = same_side_scene
    peak = _solve(
        s,
        _peak_fiducials(s["projs1"][0], 0, 0),
        _peak_fiducials(s["projs2"][0], 0, 0),
        "peak",
        "peak",
    )
    trough = _solve(
        s,
        _trough_fiducials(s["projs1"][0]),
        _trough_fiducials(s["projs2"][0]),
        "trough",
        "trough",
    )
    assert _angle_between(peak.R_stereo, trough.R_stereo) < 0.05
    np.testing.assert_allclose(peak.T_stereo, trough.T_stereo, atol=0.5)
    assert trough.model1.rms < 1.0 and trough.model2.rms < 1.0


def _drop_peak_dots(detection, n_drop: int = 20):
    """A copy of ``detection`` with ``n_drop`` PEAK-level dots removed from the larger
    level, so the trough lattice (64) outnumbers the peak lattice (61) and the stitch
    takes the trough as its reference."""
    d = copy.deepcopy(detection)
    sizes = {k: len(d.level_data[k]["centers"]) for k in ("a", "b")}
    big = max(sizes, key=sizes.get)
    ld = d.level_data[big]
    keep = np.ones(sizes[big], bool)
    keep[:n_drop] = False
    gone = np.asarray(ld["centers"], dtype=np.float64)[~keep]
    ld["centers"] = np.asarray(ld["centers"])[keep]
    ld["grid_indices"] = np.asarray(ld["grid_indices"])[keep]
    ld["n_points"] = int(keep.sum())
    pts = np.asarray(d.image_points, dtype=np.float64)
    far = cKDTree(gone).query(pts)[0] > 0.5
    d.image_points = pts[far].astype(np.asarray(detection.image_points).dtype)
    return d


def test_stitch_reference_can_be_the_trough_lattice(same_side_scene):
    """Peak clicks on the datum, but a non-datum pose where the TROUGH lattice has the
    most dots (partial view): the stitch references the trough, and the builder must
    follow that role rather than the clicked label."""
    s = same_side_scene
    f1 = _peak_fiducials(s["projs1"][0], 0, 0)
    f2 = _peak_fiducials(s["projs2"][0], 0, 0)
    clean = _solve(s, f1, f2, "peak", "peak")
    partial2 = list(s["detections2"])
    partial2[2] = _drop_peak_dots(partial2[2])
    sizes = {k: len(partial2[2].level_data[k]["centers"]) for k in ("a", "b")}
    trough_slot = "a" if s["pose_levels2"][2] == "trough" else "b"
    assert sizes[trough_slot] > sizes["b" if trough_slot == "a" else "a"]
    got = _solve(s, f1, f2, "peak", "peak", detections2=partial2)
    assert got.model2.rms < 0.2, got.model2.rms
    assert _angle_between(got.R_stereo, clean.R_stereo) < 0.05
    np.testing.assert_allclose(got.T_stereo, clean.T_stereo, atol=0.5)


# ---------------------------------------------------------------------------
# 3. Reprojection gate
# ---------------------------------------------------------------------------


def _shift_dots(detection, fiducials, n=3, shift_px=8.0):
    """Move ``n`` datum-view dots (not the fiducials) by ``shift_px`` in x, in place."""
    d = copy.deepcopy(detection)
    fid = np.array([fiducials[k] for k in ("origin", "x_axis", "y_axis")], float)
    moved = []
    for lvl in ("a", "b"):
        ld = d.level_data[lvl]
        c = np.asarray(ld["centers"], dtype=np.float64)
        far = cKDTree(fid).query(c)[0] > 3 * SPACING_MM
        for i in np.where(far)[0][: n - len(moved)]:
            moved.append(c[i].copy())
            c[i, 0] += shift_px
        ld["centers"] = c.astype(np.asarray(ld["centers"]).dtype)
        if len(moved) >= n:
            break
    pts = np.asarray(d.image_points, dtype=np.float64)
    for m in moved:
        j = cKDTree(pts).query(m)[1]
        pts[j, 0] += shift_px
    d.image_points = pts.astype(np.asarray(detection.image_points).dtype)
    assert len(moved) == n
    return d


def test_gate_removes_corrupted_dots_and_reports_them(
    same_side_scene,
):
    s = same_side_scene
    f1, f2 = _peak_fiducials(s["projs1"][0], 0, 0), _peak_fiducials(
        s["projs2"][0], 0, 0
    )
    clean = _solve(s, f1, f2, "peak", "peak")
    assert clean.board_meta["n_gated_points1"] == 0
    assert clean.board_meta["n_gated_points2"] == 0
    assert clean.board_meta["gate_limit_px2"] >= 2.0

    bad2 = list(s["detections2"])
    bad2[0] = _shift_dots(bad2[0], f2)
    gated = _solve(s, f1, f2, "peak", "peak", detections2=bad2)
    assert gated.board_meta["n_gated_points1"] == 0
    assert gated.board_meta["n_gated_points2"] == 3
    assert gated.model2.rms == pytest.approx(clean.model2.rms, abs=0.02)
    assert _angle_between(gated.R_stereo, clean.R_stereo) < 0.05
    np.testing.assert_allclose(gated.T_stereo, clean.T_stereo, atol=0.5)


# ---------------------------------------------------------------------------
# Generator rigs: both configs, both origin lattices, through the production solve
# ---------------------------------------------------------------------------


def _lattice_label(lattice: int, nearer: int) -> str:
    """Physical label of a lattice (0 peak-xy, 1 trough-xy) for a camera whose nearer
    lattice (``gt['levels']``) is ``nearer``: the nearer one is raised toward it."""
    return "peak" if lattice == nearer else "trough"


def _trough_triple(visible: list, cols: int, rows: int):
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx = [
                N_PEAK_GEN + r * cols + c,
                N_PEAK_GEN + r * cols + c + 1,
                N_PEAK_GEN + (r + 1) * cols + c,
            ]
            if all(v[idx].all() for v in visible):
                return idx
    raise AssertionError("no visible trough triple")


N_PEAK_GEN = int(np.prod(gstep.STEPPED_PEAK_SHAPE))


@pytest.mark.parametrize("config", gstep.STEREO_CONFIGS)
@pytest.mark.parametrize("origin", ["peak", "trough"])
def test_generator_rig_is_recovered_for_both_configs_and_origins(
    tmp_path, config, origin
):
    """GT clicks (row-major, +Y down the image -> +Z into the board) on either lattice
    recover the generator's rig: this is the transmission Bug-A case end to end."""
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
    gt = dict(np.load(tmp_path / "stepped_ground_truth.npz"))
    det = SteppedDetector(SteppedParams(dot_spacing_mm=15.0, step_height_mm=3.0))
    dets, fids, levels, pose_levels = {}, {}, {}, {}
    if origin == "peak":
        idx = None
    else:
        idx = _trough_triple(
            [gt["visible_cam1"][0], gt["visible_cam2"][0]], *gstep.STEPPED_TROUGH_SHAPE
        )
    for cam in (1, 2):
        img = cv2.imread(
            str(tmp_path / f"cam{cam}" / "calib00001.png"), cv2.IMREAD_GRAYSCALE
        )
        d = det.detect(img)
        assert d.success, d.diagnostics.get("error")
        px = gt[f"points_px_cam{cam}"][0]
        nearer = int(gt["levels"][0, cam - 1])
        if idx is None:
            f = gt[f"fiducials_px_cam{cam}"]
            lattice = 0
        else:
            f = px[idx]
            lattice = 1
        fids[cam] = {
            "origin": f[0].tolist(),
            "x_axis": f[1].tolist(),
            "y_axis": f[2].tolist(),
        }
        levels[cam] = _lattice_label(lattice, nearer)
        # Detector level A's lattice from the truth (the GUI's click-to-label step).
        a = np.asarray(d.level_data["a"]["centers"], dtype=np.float64)
        a_lattice = int(np.median(cKDTree(px).query(a)[1]) >= N_PEAK_GEN)
        pose_levels[cam] = [_lattice_label(a_lattice, nearer)]
        dets[cam] = [d]
    board = SteppedBoardSpec(
        dot_spacing_mm=15.0,
        step_height_mm=3.0,
        board_thickness_mm=float(gt["thickness_mm"]),
    )
    rec = calibrate_stepped_stereo(
        detections1=dets[1],
        detections2=dets[2],
        fiducials1=fids[1],
        fiducials2=fids[2],
        clicked_level1=levels[1],
        clicked_level2=levels[2],
        pose_levels1=pose_levels[1],
        pose_levels2=pose_levels[2],
        board=board,
        image_size1=(int(gt["image_width"]), int(gt["image_height"])),
        image_size2=(int(gt["image_width"]), int(gt["image_height"])),
        stereo_config="auto",
        fix_k2=True,
    )
    assert rec.board_meta["stereo_config"] == config
    assert rec.board_meta["world_z_toward_camera"] == -1.0  # generator clicks +Y down
    assert rec.model1.rms < 0.5 and rec.model2.rms < 0.5, (
        rec.model1.rms,
        rec.model2.rms,
    )
    ang = _angle_between(rec.R_stereo, gt["R_stereo"])
    assert ang < 0.05, f"R_stereo off by {ang:.3f} deg ({config}, {origin} origin)"
    t_err = float(np.linalg.norm(rec.T_stereo.reshape(3) - gt["T_stereo"].reshape(3)))
    # Single-view fits carry the fx / depth trade-off into |T| (0.37 mm on the 104 mm
    # same-side baseline, 1.2 mm on the 826 mm transmission one, 2026-10-22).
    t_tol = max(0.5, 0.0025 * float(np.linalg.norm(gt["T_stereo"])))
    assert t_err < t_tol, f"T_stereo off by {t_err:.2f} mm ({config}, {origin} origin)"
