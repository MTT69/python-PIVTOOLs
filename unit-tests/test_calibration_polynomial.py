"""Polynomial (single-plane) calibration tests for the calibration package.

Hermetic and image-free: the polynomial path fits the datum view's detected dots to
world mm, so we exercise it with synthetic dot grids projected through a known camera
(no rendered images needed). Covers the least-squares fit, the duck-typed
``back_project_to_plane`` apply parity, the record save/load model_type dispatch, the
pipeline branch (including the origin_mm offset), and that the apply coordinate path is
model-agnostic.
"""

from __future__ import annotations

import numpy as np

from pivtools_gui.calibration import apply as c2apply
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import world_frame as WF
from pivtools_gui.calibration.camera_model import (
    CameraModel,
    DistortionModel,
    PolynomialModel,
    _poly_basis,
    fit_polynomial,
)
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.pipeline import Calibrator

# ---------------------------------------------------------------------------
# Synthetic planar dot grid + projection through a known camera
# ---------------------------------------------------------------------------

DOT_COLS, DOT_ROWS, SPACING_MM = 15, 12, 14.0


def _planar_board():
    """Board-local (X,Y,0) mm + (col,row) grid indices for a flat dot grid."""
    cols, rows = np.meshgrid(np.arange(DOT_COLS), np.arange(DOT_ROWS))
    cols = cols.ravel()
    rows = rows.ravel()
    world = np.column_stack(
        [cols * SPACING_MM, rows * SPACING_MM, np.zeros(cols.size)]
    ).astype(np.float64)
    grid = np.column_stack([cols, rows]).astype(np.int64)
    return world, grid


def _project(world, *, rvec, tvec, k, dist, size):
    import cv2

    px, _ = cv2.projectPoints(world.reshape(-1, 1, 3), rvec, tvec, k, dist)
    return px.reshape(-1, 2)


def _synthetic_detection(tilt_deg=14.0):
    """A tilted planar dotboard detection: image pixels + board-local mm + grid."""
    world, grid = _planar_board()
    w, h = 1600, 1200
    k = np.array([[2200.0, 0, w / 2], [0, 2200.0, h / 2], [0, 0, 1]], np.float64)
    dist = np.array([-0.04, 0.01, 0.0, 0.0, 0.0], np.float64)
    # Centre the board, tilt it about Y so the perspective is non-trivial.
    cx_mm = (DOT_COLS - 1) * SPACING_MM / 2
    cy_mm = (DOT_ROWS - 1) * SPACING_MM / 2
    theta = np.deg2rad(tilt_deg)
    rvec = np.array([0.0, theta, 0.0], np.float64)
    tvec = np.array([-cx_mm, -cy_mm, 600.0], np.float64)
    px = _project(world, rvec=rvec, tvec=tvec, k=k, dist=dist, size=(w, h))
    det = DetectionResult(
        success=True,
        board_type="dotboard",
        image_points=px,
        board_local_points=world,
        grid_indices=grid,
        spacing_mm=SPACING_MM,
    )
    return det, (w, h)


class _FixedDetector:
    """A BoardDetector that returns the same synthetic detection for every image."""

    board_type = "dotboard"

    def __init__(self, detection):
        self._det = detection

    def detect(self, image):
        return self._det


# ---------------------------------------------------------------------------
# Fit + evaluation
# ---------------------------------------------------------------------------


def test_fit_polynomial_recovers_exact_cubic():
    """A degree-3 target is recovered to machine precision (the model is exact)."""
    rng = np.random.RandomState(0)
    w, h = 1024, 768
    px = rng.uniform([0, 0], [w, h], size=(200, 2))
    s = (px[:, 0] - w / 2) / (w / 2)
    t = (px[:, 1] - h / 2) / (h / 2)
    basis = _poly_basis(s, t)
    cx = rng.uniform(-30, 30, 10)
    cy = rng.uniform(-30, 30, 10)
    with np.errstate(
        over="ignore", invalid="ignore", divide="ignore"
    ):  # macOS Accelerate matmul false-positive
        world = np.column_stack([basis @ cx, basis @ cy, np.zeros(len(px))])
    m = fit_polynomial(px, world, (w, h))
    assert np.allclose(m.coeffs_x, cx, atol=1e-9)
    assert np.allclose(m.coeffs_y, cy, atol=1e-9)
    assert m.rms_x_mm < 1e-9 and m.rms_y_mm < 1e-9


def test_fit_polynomial_approximates_perspective():
    """A cubic fits a tilted perspective + distortion map to sub-10-micron RMS."""
    det, size = _synthetic_detection(tilt_deg=14.0)
    world = det.board_local_points  # board frame == world for the default frame
    m = fit_polynomial(det.image_points, world, size)
    # Over a real perspective view a cubic is not exact, but should be very close.
    assert m.rms_x_mm < 0.01, m.rms_x_mm
    assert m.rms_y_mm < 0.01, m.rms_y_mm
    back = m.back_project_to_plane(det.image_points)
    assert np.allclose(back[:, :2], world[:, :2], atol=0.05)
    assert np.allclose(back[:, 2], 0.0)


def test_back_project_signature_parity():
    """back_project_to_plane accepts z/tilt (ignored) and matches CameraModel shape."""
    det, size = _synthetic_detection()
    m = fit_polynomial(det.image_points, det.board_local_points, size)
    a = m.back_project_to_plane(det.image_points)
    b = m.back_project_to_plane(det.image_points, z_world=5.0, tilt_x=0.1, tilt_y=0.2)
    assert a.shape == (det.n, 3)
    assert np.allclose(a, b)  # z/tilt are ignored (single plane)
    assert m.back_project_to_plane(np.empty((0, 2))).shape == (0, 3)


# ---------------------------------------------------------------------------
# Record dispatch
# ---------------------------------------------------------------------------


def test_save_load_model_type_dispatch(tmp_path):
    """A polynomial MonoRecord round-trips and loads back as a PolynomialModel."""
    det, size = _synthetic_detection()
    m = fit_polynomial(det.image_points, det.board_local_points, size)
    wf = REC.WorldFrame(
        mode="clicks",
        origin_grid=np.array([0.0, 0.0]),
        col_sign=1,
        row_sign=-1,
        origin_mm=np.array([10.0, 5.0]),
    )
    rec_in = REC.MonoRecord(
        camera=1,
        board_type="dotboard",
        camera_model=m,
        world_frame=wf,
        per_view_rms=[float(np.hypot(m.rms_x_mm, m.rms_y_mm))],
        board_meta={"spacing_mm": SPACING_MM, "n_views": 1},
    )
    path = REC.save_mono(rec_in, tmp_path)
    rec_out = REC.load_mono(path)
    assert isinstance(rec_out.camera_model, PolynomialModel)
    assert np.allclose(rec_out.camera_model.coeffs_x, m.coeffs_x)
    assert np.allclose(rec_out.camera_model.coeffs_y, m.coeffs_y)
    assert (rec_out.camera_model.x0, rec_out.camera_model.sx) == (m.x0, m.sx)
    assert np.isclose(rec_out.camera_model.rms_x_mm, m.rms_x_mm)
    assert np.allclose(rec_out.world_frame.origin_mm, [10.0, 5.0])
    assert np.allclose(
        rec_out.camera_model.back_project_to_plane(det.image_points),
        m.back_project_to_plane(det.image_points),
    )


def test_pinhole_record_still_loads(tmp_path):
    """Back-compat: a pinhole record (no model_type) still loads as a CameraModel."""
    cm = CameraModel(
        K=np.array([[2000.0, 0, 800], [0, 2000.0, 600], [0, 0, 1]]),
        dist=np.zeros(5),
        R=np.eye(3),
        t=np.array([[0.0], [0.0], [500.0]]),
        image_size=(1600, 1200),
        distortion_model=DistortionModel.STANDARD,
        rms=0.2,
    )
    rec_in = REC.MonoRecord(
        camera=1,
        board_type="dotboard",
        camera_model=cm,
        per_view_rms=[0.2],
        board_meta={"spacing_mm": SPACING_MM},
    )
    rec_out = REC.load_mono(REC.save_mono(rec_in, tmp_path))
    assert isinstance(rec_out.camera_model, CameraModel)
    assert np.allclose(rec_out.camera_model.K, cm.K)


# ---------------------------------------------------------------------------
# Pipeline branch + origin_mm
# ---------------------------------------------------------------------------


def test_run_mono_polynomial_pipeline():
    """Calibrator(model_type='polynomial') fits a single-plane model honouring origin_mm."""
    det, size = _synthetic_detection()
    images = [np.zeros((size[1], size[0]), np.uint8)]  # only datum is read
    calr = Calibrator(
        detector=_FixedDetector(det), board_type="dotboard", model_type="polynomial"
    )
    # No clicks -> default frame (origin at the min grid corner), origin_mm offset (10, 5).
    record = calr.run_mono(
        images, camera=1, datum_index=0, spacing_mm=SPACING_MM, origin_mm=(10.0, 5.0)
    )
    assert isinstance(record.camera_model, PolynomialModel)
    # The fit target is apply_world_frame(grid, spacing, wf) — back_project must match it.
    world = WF.apply_world_frame(det.grid_indices, SPACING_MM, record.world_frame)
    back = record.camera_model.back_project_to_plane(det.image_points)
    assert np.allclose(back[:, :2], world[:, :2], atol=0.05)
    # The origin dot (min grid corner) must read the user-specified (10, 5) mm.
    og = record.world_frame.origin_grid
    oi = int(
        np.argmin(
            np.abs(det.grid_indices[:, 0] - og[0])
            + np.abs(det.grid_indices[:, 1] - og[1])
        )
    )
    assert np.allclose(back[oi, :2], [10.0, 5.0], atol=0.05)


def test_apply_coordinates_is_model_agnostic():
    """apply.calibrate_coordinates works with a PolynomialModel (duck typing)."""
    det, size = _synthetic_detection()
    m = fit_polynomial(det.image_points, det.board_local_points, size)
    coords = c2apply.calibrate_coordinates(m, det.image_points)
    assert coords.shape == (det.n, 2)
    assert np.allclose(coords, det.board_local_points[:, :2], atol=0.05)
