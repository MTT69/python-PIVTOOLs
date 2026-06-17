"""S1·Phase 2b — joint multi-camera shared-board solve (run_joint).

Synthetic 3-camera fixed rig: the board is moved to several positions (views); every camera
images each position but sees only a partial, overlapping column window. The board has a known
dome bow. The solve must recover each camera's intrinsics, the shared bowed board, and a low
reprojection RMS — with one board all cameras agree on (cross-camera agreement 0 by design).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.joint import run_joint, run_joint_polynomial

SPACING = 15.0
COLS, ROWS = 16, 10
IMG_WH = (1280, 1024)
_IMG_SIZES = {1: IMG_WH, 2: IMG_WH, 3: IMG_WH}
BOW_MM = 1.0
CX, CY = 640.0, 512.0
_FX = {1: 1300.0, 2: 1250.0, 3: 1350.0}
_CAM_RVEC = {
    1: np.array([0.0, 0.0, 0.0]),
    2: np.array([0.0, 0.25, 0.0]),
    3: np.array([0.0, -0.25, 0.0]),
}
_WINDOWS = {1: range(0, 7), 2: range(5, 12), 3: range(10, 16)}
_N_VIEWS = 6


def _K(cam):
    return np.array([[_FX[cam], 0, CX], [0, _FX[cam], CY], [0, 0, 1.0]])


def _true_board():
    """(dict g->xyz) full bowed board over the global grid."""
    half_x, half_y = (COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING
    board = {}
    for gx in range(COLS):
        for gy in range(ROWS):
            x, y = gx * SPACING, gy * SPACING
            z = BOW_MM * (
                1.0
                - 0.5 * (((x - half_x) / half_x) ** 2 + ((y - half_y) / half_y) ** 2)
            )
            board[(gx, gy)] = np.array([x, y, z])
    return board


def _dataset(seed=0, flat=False):
    rng = np.random.default_rng(seed)
    board = _true_board()
    if flat:
        board = {g: np.array([p[0], p[1], 0.0]) for g, p in board.items()}
    centroid3 = np.array([(COLS - 1) / 2 * SPACING, (ROWS - 1) / 2 * SPACING, 0.0])

    # board poses per view (it moves); moderate tilts give triangulation strength
    poses = []
    for v in range(_N_VIEWS):
        if v == 0:
            poses.append((np.eye(3), np.zeros(3)))
        else:
            rb = cv2.Rodrigues(rng.uniform(-0.30, 0.30, 3) * [1, 1, 0.3])[0]
            tb = rng.uniform(-12, 12, 3)
            poses.append((rb, tb))

    detections, global_index = {c: [] for c in _CAM_RVEC}, {}
    for cam in _CAM_RVEC:
        Rc = cv2.Rodrigues(_CAM_RVEC[cam])[0]
        tc = np.array([0.0, 0.0, 700.0]) - Rc @ centroid3
        gidx = np.array(
            [[gx, gy] for gx in _WINDOWS[cam] for gy in range(ROWS)], dtype=np.int64
        )
        board_win = np.array([board[(int(gx), int(gy))] for gx, gy in gidx])
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
                    board_type="dotboard",
                    image_points=px,
                    board_local_points=np.column_stack(
                        [gidx * SPACING, np.zeros(len(gidx))]
                    ),
                    grid_indices=gidx,
                    spacing_mm=SPACING,
                )
            )
            global_index[(cam, v)] = gidx
    return detections, global_index, board


def _detrend(P):
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    return P[:, 2] - A @ coef


def test_joint_full3d_recovers_intrinsics_and_bow():
    detections, gi, true_board = _dataset(seed=0)
    res = run_joint(
        detections,
        gi,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )

    # per-camera intrinsics
    for cam in (1, 2, 3):
        fx = res.models[cam].K[0, 0]
        assert (
            abs(fx - _FX[cam]) / _FX[cam] < 0.02
        ), f"cam{cam} fx {fx:.1f} vs {_FX[cam]}"
        assert res.models[cam].K[0, 0] == pytest.approx(
            res.models[cam].K[1, 1]
        )  # fx==fy

    # shared bowed board recovered (detrended-z correlation with truth)
    keys = sorted(res.board)
    got = np.array([res.board[k] for k in keys])
    tru = np.array([true_board[k] for k in keys])
    corr = float(np.corrcoef(_detrend(got), _detrend(tru))[0, 1])
    assert corr > 0.9, f"bow corr {corr:.3f}"
    # in-plane stays near the certified grid (no spurious lateral release, dist=0)
    nominal = np.array([[k[0] * SPACING, k[1] * SPACING] for k in keys])
    assert np.max(np.abs(got[:, :2] - nominal)) < 1.0
    assert res.rms_px < 0.5
    assert res.cross_camera_board_agreement_mm == 0.0


def test_joint_z_only_locks_in_plane():
    detections, gi, true_board = _dataset(seed=1)
    res = run_joint(
        detections,
        gi,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="z_only",
        image_size_by_cam=_IMG_SIZES,
    )
    keys = sorted(res.board)
    got = np.array([res.board[k] for k in keys])
    nominal = np.array([[k[0] * SPACING, k[1] * SPACING] for k in keys])
    # in-plane EXACTLY nominal under z_only
    np.testing.assert_allclose(got[:, :2], nominal, atol=1e-9)
    tru = np.array([true_board[k] for k in keys])
    corr = float(np.corrcoef(_detrend(got), _detrend(tru))[0, 1])
    assert corr > 0.9
    assert res.rms_px < 0.5


def test_joint_none_keeps_board_flat():
    detections, gi, _ = _dataset(seed=2, flat=True)
    res = run_joint(
        detections,
        gi,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="none",
        image_size_by_cam=_IMG_SIZES,
    )
    got = np.array([res.board[k] for k in sorted(res.board)])
    assert np.max(np.abs(got[:, 2])) < 1e-6  # board untouched (flat)
    assert res.rms_px < 0.5


def test_joint_release_beats_none_on_bowed_board():
    """On a genuinely bowed board, releasing must reproject better than forcing it flat."""
    detections, gi, _ = _dataset(seed=3)
    flat = run_joint(
        detections,
        gi,
        SPACING,
        1,
        0,
        board_release="none",
        image_size_by_cam=_IMG_SIZES,
    )
    rel = run_joint(
        detections,
        gi,
        SPACING,
        1,
        0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )
    assert rel.rms_px < flat.rms_px


def test_joint_polynomial_shares_one_global_frame():
    """Per-camera polynomials, fitted in the shared global frame, agree on a shared dot.

    The defining invariant of the joint polynomial: a dot two cameras both see (same global
    index) must map to the SAME world coordinate through each camera's own polynomial — that is
    what "one shared frame" means. A flat board makes the planar map well-posed.
    """
    detections, gi, _ = _dataset(seed=7, flat=True)
    polys = run_joint_polynomial(
        detections,
        gi,
        SPACING,
        cameras=[1, 2, 3],
        datum_view=0,
        origin_mm=(0.0, 0.0),
        image_size_by_cam=_IMG_SIZES,
    )
    assert set(polys) == {1, 2, 3}

    # (5, 4) is in both cam1's window (0..6) and cam2's window (5..11): a shared physical dot.
    shared = (5, 4)
    nominal = np.array([shared[0] * SPACING, shared[1] * SPACING])

    def _world(cam):
        idx_list = [(int(g[0]), int(g[1])) for g in gi[(cam, 0)]]
        row = idx_list.index(shared)
        px = detections[cam][0].image_points[row]
        return polys[cam].back_project_to_plane(px.reshape(1, 2))[0, :2]

    w1, w2 = _world(1), _world(2)
    # Each polynomial maps the shared dot near its nominal world position ...
    assert np.linalg.norm(w1 - nominal) < 0.5
    assert np.linalg.norm(w2 - nominal) < 0.5
    # ... and the two cameras therefore agree on it (one shared frame).
    assert np.linalg.norm(w1 - w2) < 0.5


def test_joint_polynomial_requires_datum_view_in_grid():
    """A camera whose datum view is not resolved raises, never silently drops it."""
    detections, gi, _ = _dataset(seed=8, flat=True)
    gi.pop((2, 0))  # cam2's datum view is no longer in the resolved global grid
    with pytest.raises(ValueError, match="no resolved global grid for camera 2"):
        run_joint_polynomial(
            detections,
            gi,
            SPACING,
            cameras=[1, 2],
            datum_view=0,
            origin_mm=(0.0, 0.0),
            image_size_by_cam=_IMG_SIZES,
        )


def test_joint_polynomial_requires_image_size():
    """A camera missing from image_size_by_cam raises (no normalisation fallback)."""
    detections, gi, _ = _dataset(seed=9, flat=True)
    with pytest.raises(ValueError, match="image_size_by_cam missing camera 2"):
        run_joint_polynomial(
            detections,
            gi,
            SPACING,
            cameras=[1, 2],
            datum_view=0,
            origin_mm=(0.0, 0.0),
            image_size_by_cam={1: IMG_WH},
        )


def test_joint_polynomial_rejects_count_mismatch():
    """Global indices and image points must correspond one-for-one."""
    detections, gi, _ = _dataset(seed=10, flat=True)
    gi[(1, 0)] = gi[(1, 0)][:-1]  # drop one global index so the counts disagree
    with pytest.raises(ValueError, match="global indices but"):
        run_joint_polynomial(
            detections,
            gi,
            SPACING,
            cameras=[1],
            datum_view=0,
            origin_mm=(0.0, 0.0),
            image_size_by_cam=_IMG_SIZES,
        )


def test_joint_requires_image_size():
    """run_joint hard-requires real image sizes (no biased extent-inference fallback)."""
    detections, gi, _ = _dataset(seed=11)
    with pytest.raises(ValueError, match="image_size_by_cam is required"):
        run_joint(
            detections, gi, SPACING, 1, 0, board_release="none", image_size_by_cam=None
        )


def test_joint_validates_expected_cameras():
    """A camera expected but absent from the resolved grid fails loudly, not as a quiet subset."""
    detections, gi, _ = _dataset(seed=12)
    with pytest.raises(ValueError, match="expected"):
        run_joint(
            detections,
            gi,
            SPACING,
            1,
            0,
            board_release="none",
            image_size_by_cam=_IMG_SIZES,
            expected_cameras=[1, 2, 3, 4],
        )


def _gradient_image(cam, view):
    """A plain horizontal-gradient grayscale image so the image-based figures have content."""
    row = np.linspace(0, 255, IMG_WH[0], dtype=np.uint8)
    return np.tile(row, (IMG_WH[1], 1))


def test_joint_figures_full_suite(tmp_path):
    """write_joint_figures emits the per-camera + dewarp proof set for a >2-camera rig."""
    from pivtools_gui.calibration import figures

    detections, gi, _ = _dataset(seed=0)
    res = run_joint(
        detections,
        gi,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )
    figd = tmp_path / "figures"
    figures.write_joint_figures(
        figd,
        result=res,
        detections_by_cam=detections,
        global_index=gi,
        spacing=SPACING,
        board_type="dotboard",
        datum_view=0,
        image_loader=_gradient_image,
    )

    names = {p.name for p in figd.glob("*.png")}
    assert "boards_3d.png" in names  # orientation of the rig in space
    assert "dewarp_dots.png" in names  # >2 cameras: agreement scatter
    for cam in (1, 2, 3):
        assert f"reprojection_cam{cam}.png" in names
        assert f"detection_cam{cam}_00.png" in names  # one overlay per (cam, view)
        assert f"dewarp_cam{cam}.png" in names  # >2 cameras: per-camera dewarp panels


def test_joint_figures_geometry_only_without_images(tmp_path):
    """With no image_loader, image-free figures still write; image-based ones are skipped."""
    from pivtools_gui.calibration import figures

    detections, gi, _ = _dataset(seed=0)
    res = run_joint(
        detections,
        gi,
        SPACING,
        datum_camera=1,
        datum_view=0,
        board_release="full3d",
        image_size_by_cam=_IMG_SIZES,
    )
    figd = tmp_path / "figures_geom"
    figures.write_joint_figures(
        figd,
        result=res,
        detections_by_cam=detections,
        global_index=gi,
        spacing=SPACING,
        board_type="dotboard",
        datum_view=0,
        image_loader=None,
    )

    names = {p.name for p in figd.glob("*.png")}
    assert "boards_3d.png" in names
    assert "reprojection_cam1.png" in names
    assert "dewarp_dots.png" in names  # back-projected dots need no raw image
    assert not any(n.startswith("detection_") for n in names)
    assert not any(n.startswith("dewarp_cam") for n in names)
