"""Joint multi-camera ChArUco solve — the auto-linked (corner-id) path, in memory.

ChArUco's defining feature: corner ids are globally consistent across cameras and views, so the
global grid — and all cross-camera linking — is built automatically from the detections, with NO
manual anchors/bridges (``spec=None``). This locks in what
``manual_tools/validate_joint_charuco.py`` proved on the rendered 3-cam rig, but fully in memory
(synthetic detections projected through known intrinsics — no image fixtures, no /Volumes).

The solver itself is board-agnostic (the same r0 bundle the dotboard uses); what these tests
exercise is that ``resolve_global_grid(..., spec=None)`` short-circuits to
``global_grid_from_charuco`` and the downstream solve recovers the rig.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.global_grid import resolve_global_grid
from pivtools_gui.calibration.joint import run_joint, run_joint_polynomial

SPACING = 20.0
COLS, ROWS = 14, 9  # interior corner grid (a 15x10-square board)
IMG_WH = (1280, 1024)
_IMG_SIZES = {1: IMG_WH, 2: IMG_WH, 3: IMG_WH}
CX, CY = 640.0, 512.0
_FX = {1: 1300.0, 2: 1250.0, 3: 1350.0}
_CAM_RVEC = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([0.0, 0.25, 0.0]),
    3: np.array([0.0, -0.25, 0.0]),
}
# Overlapping global-index windows: adjacent cameras share corner ids (the charuco hallmark).
_WINDOWS = {1: range(0, 6), 2: range(4, 10), 3: range(8, 14)}
_N_VIEWS = 6


def _K(cam):
    return np.array([[_FX[cam], 0, CX], [0, _FX[cam], CY], [0, 0, 1.0]])


def _cid(gx: int, gy: int) -> int:
    """A globally-consistent corner id for grid (gx, gy) — the same in every camera."""
    return gy * COLS + gx


def _dataset(seed: int = 0, flat: bool = True):
    """Synthetic charuco detections for a fixed 3-camera rig over several board poses.

    Each detection carries global ``grid_indices`` AND ``point_ids`` (corner ids) — so
    ``resolve_global_grid`` classifies it charuco and builds the grid with no spec.
    """
    rng = np.random.default_rng(seed)
    half = np.array([(COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING, 0.0])

    def board_xyz(gx, gy):
        x, y = gx * SPACING, gy * SPACING
        z = 0.0 if flat else 0.8 * (1.0 - ((x - half[0]) / half[0]) ** 2)
        return np.array([x, y, z])

    poses = [(np.eye(3), np.zeros(3))]
    for _ in range(1, _N_VIEWS):
        rb = cv2.Rodrigues(rng.uniform(-0.30, 0.30, 3) * [1, 1, 0.3])[0]
        poses.append((rb, rng.uniform(-12, 12, 3)))

    detections = {c: [] for c in _CAM_RVEC}
    for cam in _CAM_RVEC:
        Rc = cv2.Rodrigues(_CAM_RVEC[cam])[0]
        tc = np.array([0.0, 0.0, 700.0]) - Rc @ half
        gidx = np.array(
            [[gx, gy] for gx in _WINDOWS[cam] for gy in range(ROWS)], dtype=np.int64
        )
        ids = np.array([_cid(int(gx), int(gy)) for gx, gy in gidx], dtype=np.int64)
        board_win = np.array([board_xyz(int(gx), int(gy)) for gx, gy in gidx])
        for v in range(_N_VIEWS):
            Rb, tb = poses[v]
            R = Rc @ Rb
            t = (Rc @ tb.reshape(3, 1) + tc.reshape(3, 1)).reshape(3)
            px = cv2.projectPoints(board_win, cv2.Rodrigues(R)[0], t, _K(cam), None)[
                0
            ].reshape(-1, 2)
            detections[cam].append(
                DetectionResult(
                    success=True,
                    board_type="charuco",
                    image_points=px,
                    board_local_points=np.column_stack(
                        [gidx * SPACING, np.zeros(len(gidx))]
                    ),
                    grid_indices=gidx,
                    point_ids=ids,
                    spacing_mm=SPACING,
                )
            )
    return detections


def test_charuco_resolve_grid_is_automatic():
    """resolve_global_grid with spec=None builds the global grid straight from corner ids."""
    detections = _dataset(seed=0)
    grid = resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
    # one entry per (cam, view), each equal to that detection's own (global) grid indices.
    assert set(grid) == {(c, v) for c in (1, 2, 3) for v in range(_N_VIEWS)}
    for (cam, v), gi in grid.items():
        np.testing.assert_array_equal(gi, detections[cam][v].grid_indices)


def test_charuco_joint_pinhole_recovers_intrinsics():
    """The r0 (full3d) solve over the auto grid recovers each camera's intrinsics."""
    detections = _dataset(seed=1, flat=False)
    grid = resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
    res = run_joint(
        detections,
        grid,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )
    assert res.converged
    for cam in (1, 2, 3):
        fx = res.models[cam].K[0, 0]
        assert (
            abs(fx - _FX[cam]) / _FX[cam] < 0.02
        ), f"cam{cam} fx {fx:.1f} vs {_FX[cam]}"
        assert res.models[cam].K[0, 0] == pytest.approx(
            res.models[cam].K[1, 1]
        )  # fx==fy
    assert res.rms_px < 0.5
    assert (
        res.cross_camera_board_agreement_mm == 0.0
    )  # one shared board by construction


def test_charuco_joint_polynomial_shares_one_frame():
    """Per-camera polynomials fitted in the shared corner-id frame agree on a shared corner."""
    detections = _dataset(seed=2, flat=True)
    grid = resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
    polys = run_joint_polynomial(
        detections,
        grid,
        SPACING,
        cameras=[1, 2, 3],
        datum_view=0,
        origin_mm=(0.0, 0.0),
        image_size_by_cam=_IMG_SIZES,
    )
    assert set(polys) == {1, 2, 3}

    shared = (4, 4)  # in cam1's window (0..5) and cam2's (4..9)
    nominal = np.array([shared[0] * SPACING, shared[1] * SPACING])

    def _world(cam):
        idx = [(int(g[0]), int(g[1])) for g in grid[(cam, 0)]].index(shared)
        px = detections[cam][0].image_points[idx]
        return polys[cam].back_project_to_plane(px.reshape(1, 2))[0, :2]

    w1, w2 = _world(1), _world(2)
    assert np.linalg.norm(w1 - nominal) < 0.5
    assert np.linalg.norm(w2 - nominal) < 0.5
    assert np.linalg.norm(w1 - w2) < 0.5  # one shared frame


def test_charuco_joint_drops_failed_view():
    """A single failed view is skipped (not anchored, not fatal); the rig still solves."""
    detections = _dataset(seed=3, flat=False)
    detections[2][3] = DetectionResult(  # cam2 view3 detects nothing
        success=False,
        board_type="charuco",
        image_points=np.empty((0, 2)),
        board_local_points=np.empty((0, 3)),
    )
    grid = resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
    assert (2, 3) not in grid  # the bad view is gone
    assert (2, 0) in grid and (1, 0) in grid  # the rest remain
    res = run_joint(
        detections,
        grid,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )
    assert res.converged
    for cam in (1, 2, 3):
        assert abs(res.models[cam].K[0, 0] - _FX[cam]) / _FX[cam] < 0.02


def test_charuco_mixed_with_dotboard_raises():
    """A detection set mixing id-bearing (charuco) and id-less (dotboard) views is rejected."""
    detections = _dataset(seed=4)
    detections[3][0].point_ids = None  # one view now looks like a dotboard
    with pytest.raises(ValueError, match="mixed detections"):
        resolve_global_grid(detections, spec=None, spacing_mm=SPACING)
