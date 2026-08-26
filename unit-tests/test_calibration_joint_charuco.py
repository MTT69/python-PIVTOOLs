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
from pivtools_gui.calibration.joint import _kabsch, run_joint, run_joint_polynomial

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


def _dataset(
    seed: int = 0, flat: bool = True, noise_px: float = 0.0, n_views: int = _N_VIEWS
):
    """Synthetic charuco detections for a fixed 3-camera rig over several board poses.

    Each detection carries global ``grid_indices`` AND ``point_ids`` (corner ids) — so
    ``resolve_global_grid`` classifies it charuco and builds the grid with no spec.

    The rig is RIGID by construction: ``Rc``/``tc`` are fixed per camera and the board poses
    are shared across cameras, so ``pose(cam, view) = rig_cam ∘ board_view`` exactly. That
    makes the camera centres a known ground truth — see ``_truth_camera_centres``.

    ``noise_px`` adds isotropic Gaussian corner noise. It is drawn from a SEPARATE stream so
    the board poses are identical at every noise level, which keeps runs comparable.
    ``n_views`` sets how many board poses are shot; the first is always the identity (datum).
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 10_000)
    half = np.array([(COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING, 0.0])

    def board_xyz(gx, gy):
        x, y = gx * SPACING, gy * SPACING
        z = 0.0 if flat else 0.8 * (1.0 - ((x - half[0]) / half[0]) ** 2)
        return np.array([x, y, z])

    poses = [(np.eye(3), np.zeros(3))]
    for _ in range(1, n_views):
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
        for v in range(n_views):
            Rb, tb = poses[v]
            R = Rc @ Rb
            t = (Rc @ tb.reshape(3, 1) + tc.reshape(3, 1)).reshape(3)
            px = cv2.projectPoints(board_win, cv2.Rodrigues(R)[0], t, _K(cam), None)[
                0
            ].reshape(-1, 2)
            if noise_px:
                px = px + noise_rng.normal(0.0, noise_px, px.shape)
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


def _truth_camera_centres() -> dict[int, np.ndarray]:
    """The rig ``_dataset`` builds, as camera centres in the board frame.

    ``_dataset`` places camera ``c`` at ``Rc = Rodrigues(_CAM_RVEC[c])``,
    ``tc = [0,0,700] - Rc @ half`` (world -> camera), so its centre is ``-Rc.T @ tc``.
    """
    half = np.array([(COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING, 0.0])
    out = {}
    for cam, rvec in _CAM_RVEC.items():
        Rc = cv2.Rodrigues(rvec)[0]
        tc = np.array([0.0, 0.0, 700.0]) - Rc @ half
        out[cam] = (-Rc.T @ tc).reshape(3)
    return out


def _solved_camera_centres(res) -> dict[int, np.ndarray]:
    """Camera centres from a solve: ``C = -R.T @ t`` (the model stores world -> camera)."""
    return {
        cam: (-m.R.T @ m.t).reshape(3) for cam, m in res.models.items()
    }


def _baselines(centres: dict[int, np.ndarray]) -> dict[tuple[int, int], float]:
    """Pairwise camera separations. Frame-independent, so they compare across gauges."""
    cams = sorted(centres)
    return {
        (a, b): float(np.linalg.norm(centres[a] - centres[b]))
        for i, a in enumerate(cams)
        for b in cams[i + 1 :]
    }


@pytest.mark.parametrize(
    "noise_px, n_views, baseline_tol_mm, shape_tol_mm",
    [(0.0, _N_VIEWS, 0.1, 0.1), (0.3, 20, 2.0, 2.0)],
)
def test_charuco_joint_recovers_camera_positions(
    noise_px, n_views, baseline_tol_mm, shape_tol_mm
):
    """The solve must recover WHERE the cameras are, not just what they see.

    ``_dataset`` builds a rigid rig, so the camera centres are known exactly. The solved
    world frame is gauge-fixed by the global grid and need not coincide with the board
    frame, so this compares two frame-independent quantities: pairwise baselines, and the
    residual after a rigid (Kabsch) alignment of the solved centres onto the truth.

    This is the assertion the joint suite never had -- it asserts convergence, ``fx``,
    ``fx == fy``, ``rms`` and board agreement, and nothing about camera geometry. A free
    6-DOF pose per (camera, view) lets the reported position absorb corner noise, which a
    reprojection residual cannot see.

    Tolerances are measured, not aspirational. On this synthetic rig (54 corners per
    camera-view, +/-0.3 rad tilts, 700 mm standoff) ``fx`` is only determined to ~1% at
    0.3 px noise with 6 views and ~0.3% with 20, and a focal-length error moves every camera
    along its optical axis. The rigid rig cannot beat that floor: measured 20-view, 0.3 px
    baseline error is 0.6-1.2 mm (free per-view poses: 8.9-9.1 mm). On the real 3-camera
    pattern (204 corners per view, 20 views) the same change took the 8-seed cam2-cam3 scatter
    from sd 6.7 mm to sd 0.018 mm -- see ``Downloads/gtcalib-joint/validation/pre_change``.
    """
    detections = _dataset(seed=1, flat=False, noise_px=noise_px, n_views=n_views)
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

    truth = _truth_camera_centres()
    solved = _solved_camera_centres(res)
    assert set(solved) == set(truth)

    truth_bl = _baselines(truth)
    solved_bl = _baselines(solved)
    worst_bl = max(abs(solved_bl[k] - truth_bl[k]) for k in truth_bl)

    # Rigid-align solved onto truth; the leftover is rig SHAPE error, gauge removed.
    cams = sorted(truth)
    A = np.array([solved[c] for c in cams])
    B = np.array([truth[c] for c in cams])
    R_align, t_align = _kabsch(A, B)
    resid = (A @ R_align.T + t_align) - B
    worst_shape = float(np.abs(resid).max())

    detail = "  ".join(
        f"cam{a}-cam{b}: {solved_bl[(a, b)]:.3f} vs {truth_bl[(a, b)]:.3f} mm"
        for (a, b) in sorted(truth_bl)
    )
    assert worst_bl < baseline_tol_mm, (
        f"baseline error {worst_bl:.3f} mm exceeds {baseline_tol_mm} mm "
        f"at {noise_px} px noise -- {detail}"
    )
    assert worst_shape < shape_tol_mm, (
        f"rig shape error {worst_shape:.3f} mm exceeds {shape_tol_mm} mm "
        f"at {noise_px} px noise"
    )


def test_charuco_joint_model_world_frame_is_the_board_plane():
    """CHARACTERISATION: ``models[c]`` must map BOARD-frame world mm -> pixels.

    ``apply.py`` back-projects pixels onto the sheet plane with ``z_world`` defaulting to
    0.0, documented and implemented as the calibration-board plane. That only holds if the
    model's world frame IS the board frame. The solve's internal gauge is free -- many
    parameterisations fit the corners identically -- but the OUTPUT frame is not, and
    nothing else in the suite pins it.

    Guards the reparameterisation: a gauge that puts the world origin anywhere else (a
    reference camera, say) leaves the reprojection RMS untouched and silently moves every
    calibrated vector field. This test is how that failure becomes visible.
    """
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
    board = np.array([res.board[k] for k in sorted(res.board)])
    assert abs(board[:, 2]).max() < 2.0, "released board should sit near the world Z=0 plane"

    for cam, model in sorted(res.models.items()):
        px = model.project(board)
        W, H = model.image_size
        on = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        assert on.sum() >= 20, f"cam{cam}: only {on.sum()} board points land on the sensor"
        back = model.back_project_to_plane(px[on], z_world=0.0)[:, :2]
        err = np.linalg.norm(back - board[on][:, :2], axis=1)
        assert err.max() < 0.5, (
            f"cam{cam}: world -> pixel -> world round trip is off by {err.max():.3f} mm; "
            f"the model's world frame is not the board plane"
        )


@pytest.mark.parametrize(
    "noise_px, tol_mm",
    [(0.0, 0.05), (0.3, 1.5)],
)
def test_charuco_joint_back_projects_off_plane(noise_px, tol_mm):
    """Back-projection to ``z_world != 0`` must land on the truth, not only at ``z = 0``.

    The on-plane round trip (previous test) passes on a per-view pose by construction: the
    pose was fitted to that view's corners, so ``z = 0`` is exact whatever the camera's
    position error. The pose error lives along the weakly observed depth+tilt direction,
    and only shows once a ray is intersected OFF the datum plane -- which is what
    ``apply.py`` does for any non-zero ``z_world`` / sheet tilt.

    Truth: ``_dataset`` places the rig at ``Rc, tc`` in the board frame with no distortion,
    and view 0 is the identity board pose, so the board frame IS the datum world frame.
    Points on ``z = +/-20 mm`` are projected through the truth model, then back-projected
    through the solved model at the same ``z_world``; the in-plane miss is the error.

    What this metric can and cannot see: it is the consumer-side quantity, so it folds in
    the solved intrinsics too, and on this rig the ``fx`` uncertainty (~1% at 0.3 px, 6
    views) sets a floor of ~1 mm at 20 mm off-plane. Measured: free per-view poses 1.65 mm,
    rigid rig 1.06 mm (seed 1). The tolerance bounds the rigid-rig behaviour; it is not a
    noise-floor claim. Substituting truth intrinsics to isolate the pose is NOT valid -- the
    solved pose co-adapts with the solved ``fx`` and the mixed model is off by >10 mm.
    """
    detections = _dataset(seed=1, flat=False, noise_px=noise_px)
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
    half = np.array([(COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING, 0.0])
    xs = np.linspace(0.0, (COLS - 1) * SPACING, 8)
    ys = np.linspace(0.0, (ROWS - 1) * SPACING, 6)
    gx, gy = np.meshgrid(xs, ys)
    worst = {}
    for cam, model in sorted(res.models.items()):
        Rc = cv2.Rodrigues(_CAM_RVEC[cam])[0]
        tc = np.array([0.0, 0.0, 700.0]) - Rc @ half
        W, H = model.image_size
        for z in (-20.0, 20.0):
            pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
            px = cv2.projectPoints(pts, cv2.Rodrigues(Rc)[0], tc, _K(cam), None)[
                0
            ].reshape(-1, 2)
            on = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            assert on.sum() >= 10, f"cam{cam} z={z}: only {on.sum()} points on sensor"
            back = model.back_project_to_plane(px[on], z_world=z)[:, :2]
            err = np.linalg.norm(back - pts[on][:, :2], axis=1)
            worst[(cam, z)] = float(err.max())
    detail = "  ".join(f"cam{c} z={z:+.0f}: {e:.3f}" for (c, z), e in sorted(worst.items()))
    assert max(worst.values()) < tol_mm, (
        f"off-plane back-projection error {max(worst.values()):.3f} mm exceeds {tol_mm} mm "
        f"at {noise_px} px noise -- {detail}"
    )


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
