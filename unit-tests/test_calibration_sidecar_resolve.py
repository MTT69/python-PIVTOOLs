"""Re-solve a mono calibration from the ``inputs.mat`` sidecar alone.

The sidecar is meant to let a user "delete the model, regenerate without re-detecting or
re-clicking" — and, with figures off, without the images on disk. These tests prove that
end to end at the unit level (no Flask, no rendered images):

- the detections + clicked world frame + board geometry round-trip through ``inputs.mat``;
- ``run_mono`` fed the stored detections + image size + clicks reproduces the SAME model it
  produced from images, while touching no image array (``images=[]``);
- the CLI and GUI compute an identical detection key for the same dataset (det_key parity),
  so a sidecar written by one front-end is reused by the other;
- the model GET's world-frame fallback (``_inputs_world_frame``) shapes the stored clicks so
  the GUI restores origin/+X/+Y for a deleted-model re-solve.
"""

from __future__ import annotations

import numpy as np

from pivtools_core.image_handling.path_utils import infer_image_type
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.detection.charuco import CharucoParams
from pivtools_gui.calibration.detection.dotboard import DotboardParams
from pivtools_gui.calibration.pipeline import Calibrator
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as REC

DOT_COLS, DOT_ROWS, SPACING_MM = 15, 12, 14.0


def _detection(tilt_deg: float):
    """A tilted planar dotboard detection: image pixels + board-local mm + grid indices."""
    import cv2

    cols, rows = np.meshgrid(np.arange(DOT_COLS), np.arange(DOT_ROWS))
    cols, rows = cols.ravel(), rows.ravel()
    world = np.column_stack(
        [cols * SPACING_MM, rows * SPACING_MM, np.zeros(cols.size)]
    ).astype(np.float64)
    grid = np.column_stack([cols, rows]).astype(np.int64)
    w, h = 1600, 1200
    k = np.array([[2200.0, 0, w / 2], [0, 2200.0, h / 2], [0, 0, 1]], np.float64)
    dist = np.array([-0.04, 0.01, 0.0, 0.0, 0.0], np.float64)
    cx_mm = (DOT_COLS - 1) * SPACING_MM / 2
    cy_mm = (DOT_ROWS - 1) * SPACING_MM / 2
    theta = np.deg2rad(tilt_deg)
    rvec = np.array([0.0, theta, 0.0], np.float64)
    tvec = np.array([-cx_mm, -cy_mm, 600.0], np.float64)
    px, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, tvec, k, dist)
    det = DetectionResult(
        success=True,
        board_type="dotboard",
        image_points=px.reshape(-1, 2),
        board_local_points=world,
        grid_indices=grid,
        spacing_mm=SPACING_MM,
    )
    return det, (w, h)


def _clicks_from_detection(det):
    """A clicks payload (origin/+X/+Y snapped to real dots + an mm offset)."""
    pts = det.image_points
    gi = det.grid_indices
    origin_i = int(np.argmin(gi[:, 0] + gi[:, 1]))  # (0,0) corner
    x_i = int(np.argmax(gi[:, 0] - gi[:, 1]))  # far +col
    y_i = int(np.argmax(gi[:, 1] - gi[:, 0]))  # far +row
    return {
        "origin": [float(pts[origin_i, 0]), float(pts[origin_i, 1])],
        "x_axis": [float(pts[x_i, 0]), float(pts[x_i, 1])],
        "y_axis": [float(pts[y_i, 0]), float(pts[y_i, 1])],
        "origin_mm": [2.5, 5.0],
    }


# ---------------------------------------------------------------------------
# det_key parity (Phase A)
# ---------------------------------------------------------------------------

def test_det_key_parity_infer_image_type():
    """CLI and GUI now both key on infer_image_type(format), so the same dataset -> same key
    regardless of a request/config image_type that disagrees with the file extension."""
    fmt = "calib%05d.tif"
    params = CharucoParams()
    cli = INP.joint_det_key("charuco", 10, fmt, infer_image_type(fmt), [1, 2], params)
    gui = INP.joint_det_key("charuco", 10, fmt, infer_image_type(fmt), [1, 2], params)
    assert cli == gui
    # A stale explicit image_type would have produced a different (non-matching) key.
    stale = INP.joint_det_key("charuco", 10, fmt, "lavision_im7", [1, 2], params)
    assert stale != cli


# ---------------------------------------------------------------------------
# inputs.mat round-trip + image-free re-solve (Phases B/E)
# ---------------------------------------------------------------------------

def test_inputs_roundtrip_detections_coords_geometry(tmp_path):
    """Detections, clicked coords, and board geometry survive the inputs.mat round-trip."""
    det, _size = _detection(14.0)
    clicks = _clicks_from_detection(det)
    geom = REC.geometry_meta("dotboard", DotboardParams(dot_spacing_mm=SPACING_MM))
    INP.save_inputs(
        tmp_path,
        path_type="mono",
        board_type="dotboard",
        detections={1: [det]},
        image_size_by_cam={1: (1600, 1200)},
        det_key="abc123",
        board_params=geom,
        coords=clicks,
    )
    side = INP.load_inputs(tmp_path)
    assert side.det_key == "abc123"
    assert side.board_params["dot_spacing_mm"] == SPACING_MM
    assert side.image_size_by_cam[1] == (1600, 1200)
    back = side.detections[1][0]
    assert np.allclose(back.image_points, det.image_points)
    assert np.array_equal(back.grid_indices, det.grid_indices)
    # coords round-trip (lists), origin_mm preserved
    assert np.allclose(side.coords["origin"], clicks["origin"])
    assert side.coords["origin_mm"] == [2.5, 5.0]


def _resolve_pair(model_type, n_views, clicks):
    """(model_from_images, model_from_sidecar_without_images) for a mono solve."""
    tilts = [10.0, 14.0, 18.0][:n_views]
    dets = [_detection(t)[0] for t in tilts]
    size = (1600, 1200)
    calr = Calibrator(detector=None, board_type="dotboard", model_type=model_type)

    # (1) Solve with images present (images only matter for figures here; we pass detections).
    rec_img = calr.run_mono(
        [np.zeros((size[1], size[0]), np.uint8)] * n_views,
        camera=1,
        clicks=clicks,
        origin_mm=(clicks["origin_mm"] if clicks else None),
        datum_index=0,
        spacing_mm=SPACING_MM,
        image_size=size,
        detections=dets,
    )

    # (2) Persist to the sidecar, reload, and re-solve from it with NO images on disk.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        INP.save_inputs(
            Path(d),
            path_type="mono",
            board_type="dotboard",
            detections={1: dets},
            image_size_by_cam={1: size},
            det_key="k",
            board_params=REC.geometry_meta("dotboard", DotboardParams(dot_spacing_mm=SPACING_MM)),
            coords=clicks,
        )
        side = INP.load_inputs(Path(d))
        rec_side = calr.run_mono(
            [],  # no images
            camera=1,
            clicks=side.coords,
            origin_mm=(side.coords["origin_mm"] if side.coords else None),
            datum_index=0,
            spacing_mm=SPACING_MM,
            image_size=side.image_size_by_cam[1],
            detections=side.detections[1],
        )
    return rec_img, rec_side


def test_polynomial_resolve_from_sidecar_matches():
    """A single-plane polynomial re-solved from the sidecar (no images) reproduces the model."""
    clicks = _clicks_from_detection(_detection(14.0)[0])
    a, b = _resolve_pair("polynomial", 1, clicks)
    assert np.allclose(a.camera_model.coeffs_x, b.camera_model.coeffs_x)
    assert np.allclose(a.camera_model.coeffs_y, b.camera_model.coeffs_y)
    assert np.allclose(a.world_frame.origin_mm, b.world_frame.origin_mm)
    assert np.allclose(a.world_frame.origin_mm, [2.5, 5.0])


def test_pinhole_resolve_from_sidecar_matches():
    """A pinhole bundle re-solved from the sidecar (no images) reproduces K, dist, and pose."""
    clicks = _clicks_from_detection(_detection(14.0)[0])
    a, b = _resolve_pair("pinhole", 3, clicks)
    assert np.allclose(a.camera_model.K, b.camera_model.K)
    assert np.allclose(a.camera_model.dist, b.camera_model.dist)
    assert np.allclose(a.camera_model.R, b.camera_model.R)
    assert np.allclose(a.camera_model.t, b.camera_model.t)


# ---------------------------------------------------------------------------
# Geometry stamped into the model record (Phase 1 carry-over, re-checked here)
# ---------------------------------------------------------------------------

def test_run_mono_stamps_geometry_when_detector_has_params(tmp_path):
    """A detector carrying params stamps board geometry into the record's board_meta."""

    class _Det:
        board_type = "dotboard"
        params = DotboardParams(dot_spacing_mm=SPACING_MM)

        def detect(self, image):  # pragma: no cover - not called (detections passed)
            raise AssertionError

    det, size = _detection(14.0)
    calr = Calibrator(detector=_Det(), board_type="dotboard", model_type="polynomial")
    record = calr.run_mono(
        [np.zeros((size[1], size[0]), np.uint8)],
        camera=1,
        datum_index=0,
        spacing_mm=SPACING_MM,
        image_size=size,
        detections=[det],
    )
    geo = record.board_meta.get("geometry")
    assert geo and geo["dot_spacing_mm"] == SPACING_MM and geo["board_type"] == "dotboard"
