"""End-to-end recovery tests for the calibration package.

Hermetic tracks (always run): synthetic ChArUco (OpenCV) calibration recovery —
planar + stereo — plus the world-frame resolver, distortion recovery, 3C
reconstruction, the file-IO apply path, and the ChArUco-Y regression.

No synthetic dotboard accuracy track: the renderer rounded dot centres to
integer pixels and limited the detector it tested (removed 2026-08-28).
Dotboard accuracy is checked on a real board with a known setup.

The SIG dotboard + 3C track lives in test_calibration_sig.py (skipped if the SIG
binary/ncgen are absent), to keep this suite fast and dependency-free.

Run with ``--make-figures`` to drop diagnostic PNGs in PyPIVTools/figures/debug/.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import pytest

from pivtools_cli.generate_synthetic_charuco import generate_charuco_dataset
from pivtools_cli.generate_synthetic_stereo import (
    compose_stereo_poses,
    generate_charuco_images,
    make_camera_matrix,
    make_poses,
    make_stereo_transform,
)
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import runio
from pivtools_gui.calibration import world_frame as WF
from pivtools_gui.calibration.camera_model import (
    CameraModel,
    DistortionModel,
    fit_intrinsics,
    fit_pose,
)
from pivtools_gui.calibration.detection.charuco import (
    CharucoBoardDetector,
    CharucoParams,
)
from pivtools_gui.calibration.pipeline import Calibrator
from pivtools_gui.calibration.stereo_model import (
    StereoCalibrator,
    reconstruct_3c_at_points,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "debug"
CHARUCO = CharucoParams(10, 7, 0.030, 0.5, "DICT_4X4_1000", 6)


def _figpath(slug: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.date(2026, 6, 4).isoformat()  # deterministic (no Date.now in tests)
    return FIG_DIR / f"{today}-calib2-{slug}.png"


# ---------------------------------------------------------------------------
# Fixtures (session-scoped renders)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def charuco_planar(tmp_path_factory):
    out = tmp_path_factory.mktemp("charuco") / "imgs"
    generate_charuco_dataset(
        output_dir=out,
        n_views=12,
        megapixels=1.0,
        sq_h=10,
        sq_v=7,
        sq_size=0.030,
        marker_ratio=0.5,
        dict_id=cv2.aruco.DICT_4X4_1000,
    )
    gt = np.load(out.parent / "ground_truth.npz")
    imgs = [
        cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted(out.glob("calib*.png"))
    ]
    return imgs, gt


@pytest.fixture(scope="session")
def stereo_render(tmp_path_factory):
    d = tmp_path_factory.mktemp("stereo")
    W = H = 1000
    cam = make_camera_matrix(W, H)
    fx = cam[0, 0]
    Rs, Ts = make_stereo_transform()
    # charuco
    cbw = 10 * 0.030
    cc = np.array([cbw / 2, 7 * 0.030 / 2, 0])
    cp1 = make_poses(12, cc, fx, W, cbw, (0.85, 0.65))
    cp2 = compose_stereo_poses(cp1, Rs, Ts)
    generate_charuco_images(d / "ch1", cam, cp1, W, H)
    generate_charuco_images(d / "ch2", cam, cp2, W, H)

    def load(p):
        return [cv2.imread(str(f), 0) for f in sorted(p.glob("calib*.png"))]

    return {
        "cam": cam,
        "R_stereo": Rs,
        "T_stereo": Ts,
        "charuco": (load(d / "ch1"), load(d / "ch2")),
    }


# ---------------------------------------------------------------------------
# ChArUco
# ---------------------------------------------------------------------------


def test_charuco_id_ordering():
    """grid_index = (id % (sh-1), id // (sh-1)) is row-major col-fast (runtime check)."""
    det = CharucoBoardDetector(CHARUCO)
    cm = det._corners_mm
    cols = CHARUCO.interior_cols
    sq = CHARUCO.square_size_mm
    for cid in range(cm.shape[0]):
        col, row = cid % cols, cid // cols
        # OpenCV places corner 0 at the first interior corner (1 square in).
        assert np.allclose(cm[cid], [(col + 1) * sq, (row + 1) * sq, 0.0], atol=1e-4)


def test_charuco_mono_recovery(charuco_planar):
    imgs, gt = charuco_planar
    det = CharucoBoardDetector(CHARUCO)
    rec = Calibrator(det, "charuco").run_mono(
        imgs, camera=1, spacing_mm=CHARUCO.square_size_mm
    )
    fx_gt = float(gt["camera_matrix"][0, 0])
    assert rec.camera_model.rms < 0.2
    assert abs(rec.camera_model.K[0, 0] - fx_gt) / fx_gt < 0.01
    assert rec.camera_model.K[0, 0] == pytest.approx(rec.camera_model.K[1, 1])  # fx==fy


def test_distortion_recovery_exact():
    """fit_intrinsics recovers injected k1,k2,p1,p2 + off-centre principal point."""
    det = CharucoBoardDetector(CHARUCO)
    board = det._corners_mm
    W = H = 1000
    K_true = np.array([[1000.0, 0, 500], [0, 1000, 520], [0, 0, 1]])
    dist_true = np.array([-0.12, 0.04, 0.001, -0.0008, 0.0])
    rng = np.random.default_rng(3)
    objs, imgs = [], []
    for v in range(14):
        rvec = np.zeros(3) if v == 0 else rng.uniform(-0.25, 0.25, 3) * [1, 1, 0.4]
        tvec = np.array(
            [rng.uniform(-20, 20), rng.uniform(-20, 20), 700 + rng.uniform(-40, 40)]
        )
        p, _ = cv2.projectPoints(board, rvec, tvec, K_true, dist_true)
        objs.append(board)
        imgs.append(p.reshape(-1, 2))
    K, dist, *_ = fit_intrinsics(objs, imgs, (W, H), DistortionModel.STANDARD)
    assert abs(K[0, 0] - 1000) < 1.0
    assert abs(K[1, 2] - 520) < 1.0
    assert np.allclose(dist[:4], dist_true[:4], atol=1e-3)


def test_world_frame_orthogonal(charuco_planar):
    imgs, _ = charuco_planar
    det = CharucoBoardDetector(CHARUCO)
    d0 = det.detect(imgs[0])
    ids = d0.point_ids.tolist()
    cols = CHARUCO.interior_cols

    def px(cid):
        return d0.image_points[ids.index(cid)]

    clicks = {"origin": px(0), "x_axis": px(1), "y_axis": px(cols)}
    wf = WF.resolve_world_frame(d0.grid_indices, d0.image_points, clicks)
    w = WF.apply_world_frame(d0.grid_indices, CHARUCO.square_size_mm, wf)
    w_o, w_x, w_y = w[ids.index(0)], w[ids.index(1)], w[ids.index(cols)]
    assert np.allclose(w_o[:2], [0, 0], atol=1e-6)
    assert np.dot(w_x[:2] - w_o[:2], w_y[:2] - w_o[:2]) == pytest.approx(0.0, abs=1e-6)
    assert w_x[0] > 0 and abs(w_x[1]) < 1e-6
    assert w_y[1] > 0 and abs(w_y[0]) < 1e-6


@pytest.mark.parametrize(
    "x_gi, y_gi, swap, sx, sy",
    [
        ((1, 0), (0, 1), False, 1, 1),
        ((-1, 0), (0, 1), False, -1, 1),
        ((1, 0), (0, -1), False, 1, -1),
        ((-1, 0), (0, -1), False, -1, -1),
        ((0, 1), (1, 0), True, 1, 1),
        ((0, -1), (1, 0), True, -1, 1),
        ((0, 1), (-1, 0), True, 1, -1),
        ((0, -2), (-3, 0), True, -1, -1),
    ],
)
def test_world_frame_every_sign_and_swap(x_gi, y_gi, swap, sx, sy):
    """Every click direction resolves to the documented sign/swap and maps the +X dot
    to +x, the +Y dot to +y (until 2026-08-26 only the (+,+, no swap) case was tested).
    """
    wf = WF.resolve_world_frame_from_grid(
        (3, 3), (3 + x_gi[0], 3 + x_gi[1]), (3 + y_gi[0], 3 + y_gi[1])
    )
    assert (wf.swap_axes, wf.col_sign, wf.row_sign) == (swap, sx, sy)
    gi = np.array([[3, 3], [3 + x_gi[0], 3 + x_gi[1]], [3 + y_gi[0], 3 + y_gi[1]]])
    w = WF.apply_world_frame(gi, 10.0, wf)
    assert np.allclose(w[0, :2], 0.0)
    assert w[1, 0] > 0 and abs(w[1, 1]) < 1e-9
    assert w[2, 1] > 0 and abs(w[2, 0]) < 1e-9


@pytest.mark.parametrize(
    "x_gi, y_gi, match",
    [
        ((3, 3), (3, 4), "snapped to the origin"),
        ((4, 3), (3, 3), "snapped to the origin"),
        ((5, 5), (3, 4), "diagonal"),
        ((5, 3), (6, 3), "along the \\+X axis"),
    ],
)
def test_world_frame_rejects_undecidable_clicks(x_gi, y_gi, match):
    """A tied, zero, or collinear click raises instead of resolving to +1 / column."""
    with pytest.raises(ValueError, match=match):
        WF.resolve_world_frame_from_grid((3, 3), x_gi, y_gi)


def test_charuco_y_not_inverted(charuco_planar):
    """Regression: calibration output must be physics-correct in Y (no inversion)."""
    imgs, _ = charuco_planar
    det = CharucoBoardDetector(CHARUCO)
    rec = Calibrator(det, "charuco").run_mono(
        imgs, camera=1, spacing_mm=CHARUCO.square_size_mm
    )
    d0 = det.detect(imgs[0])
    ids = d0.point_ids.tolist()
    cols = CHARUCO.interior_cols
    w = WF.apply_world_frame(d0.grid_indices, CHARUCO.square_size_mm, rec.world_frame)
    R, t = fit_pose(
        w, d0.image_points, rec.camera_model.K, rec.camera_model.dist, planar=True
    )
    cam = CameraModel(
        rec.camera_model.K, rec.camera_model.dist, R, t, rec.camera_model.image_size
    )
    # A feature one row in +Y must back-project to world Y > 0.
    wb = cam.back_project_to_plane(d0.image_points[ids.index(cols)][None, :])[0]
    assert wb[1] > 0


def test_charuco_stereo_recovery(stereo_render):
    imgs1, imgs2 = stereo_render["charuco"]
    det = CharucoBoardDetector(CHARUCO)
    rec = StereoCalibrator(det, "charuco").run_stereo(
        imgs1, imgs2, 1, 2, clicks=None, spacing_mm=CHARUCO.square_size_mm
    )
    ang = np.degrees(
        np.arccos(
            np.clip(
                (np.trace(rec.R_stereo @ stereo_render["R_stereo"].T) - 1) / 2, -1, 1
            )
        )
    )
    assert ang < 0.3
    assert float(np.linalg.norm(rec.T_stereo)) == pytest.approx(50.0, abs=1.0)
    # The vector, not just its length: a permuted or sign-flipped baseline axis has
    # the same norm and the same relative angle (review 2026-08-26).
    np.testing.assert_allclose(
        np.asarray(rec.T_stereo).reshape(3),
        1000.0 * np.asarray(stereo_render["T_stereo"]).reshape(3),  # render is in m
        atol=1.0,
    )


def test_stereo_origin_mm_reaches_cam2(stereo_render):
    """A non-zero origin_mm must shift BOTH cameras' world frames together (A2).

    Before the fix, cam2's independently-clicked frame dropped cam1's origin_mm,
    so the composed stereo pose shifted by exactly that offset. The relative pose
    is invariant under a world re-origin, so the two runs must agree.
    """
    imgs1, imgs2 = stereo_render["charuco"]

    def run(origin_mm):
        det = CharucoBoardDetector(CHARUCO)
        return StereoCalibrator(det, "charuco").run_stereo(
            imgs1,
            imgs2,
            1,
            2,
            clicks=None,
            spacing_mm=CHARUCO.square_size_mm,
            origin_mm=origin_mm,
        )

    base = run(None)
    shifted = run((50.0, -20.0))
    np.testing.assert_allclose(shifted.R_stereo, base.R_stereo, atol=1e-6)
    np.testing.assert_allclose(shifted.T_stereo, base.T_stereo, atol=1e-3)
    # The shift itself must be real: world points move by +d, so with pixels
    # fixed the pose compensates with t' = t - R @ d.
    d_world = np.array([[50.0], [-20.0], [0.0]])
    expected_t1 = base.model1.t - base.model1.R @ d_world
    np.testing.assert_allclose(shifted.model1.t, expected_t1, atol=1e-2)


# ---------------------------------------------------------------------------
# Stereo dotboard: camera 2 sees a different extent of the board (Design A)
# ---------------------------------------------------------------------------

_DOT_COLS, _DOT_ROWS = 14, 10
_DOT_SPACING_MM = 6.0
# Camera 2 misses the board's first columns: its grid walk starts there, so its indices
# are re-zeroed (grid_detection.py re-bases the BFS walk on the dots it found).
_CAM2_CLIP_COLS = 3
# The physical dots both cameras pick as origin / +X / +Y (full-board col, row).
_PICK_DOTS = {"origin": (5, 2), "x_axis": (9, 2), "y_axis": (5, 6)}


class _NoDetect:
    """Detector stand-in: the test hands run_stereo its detections, so nothing detects."""

    def detect(self, img):  # pragma: no cover - must never be called
        raise AssertionError("run_stereo was handed detections; it must not re-detect")


def _clipped_dotboard_pair(n_views: int = 6):
    """Synthetic dotboard detections for a stereo pair whose camera 2 is clipped.

    Exact projections (no image rendering, so no aliasing): cam1 sees every dot, cam2
    sees columns >= ``_CAM2_CLIP_COLS`` with its grid indices re-zeroed, as a real
    clipped detection would report them. Returns per-camera DetectionResult lists, the
    image size, the ground-truth (R_stereo, T_stereo mm), and the full-board pixel
    position of every dot in each camera's datum view (view 0).
    """
    from pivtools_gui.calibration.detection.base import DetectionResult

    W, H = 1600, 1200
    K = make_camera_matrix(W, H)
    sp_m = _DOT_SPACING_MM / 1000.0
    cols, rows = np.meshgrid(np.arange(_DOT_COLS), np.arange(_DOT_ROWS))
    gi_full = np.column_stack([cols.ravel(), rows.ravel()]).astype(np.int64)
    obj_m = np.column_stack(
        [gi_full[:, 0] * sp_m, gi_full[:, 1] * sp_m, np.zeros(len(gi_full))]
    )
    width_m = (_DOT_COLS - 1) * sp_m
    centre = np.array([width_m / 2, (_DOT_ROWS - 1) * sp_m / 2, 0.0])
    # Fill kept low so the 50 mm baseline + 5 deg yaw keeps camera 2 inside the frame.
    poses1 = make_poses(n_views, centre, K[0, 0], W, width_m, (0.3, 0.25))
    Rs, Ts = make_stereo_transform()
    poses2 = compose_stereo_poses(poses1, Rs, Ts)

    keep2 = gi_full[:, 0] >= _CAM2_CLIP_COLS
    gi_clip = gi_full[keep2] - np.array([_CAM2_CLIP_COLS, 0])
    det1, det2, datum_px = [], [], {}
    for v, ((rv1, tv1), (rv2, tv2)) in enumerate(zip(poses1, poses2)):
        p1, _ = cv2.projectPoints(obj_m, rv1, tv1, K, np.zeros(5))
        p2, _ = cv2.projectPoints(obj_m, rv2, tv2, K, np.zeros(5))
        p1, p2 = p1.reshape(-1, 2), p2.reshape(-1, 2)
        assert (p1 > 0).all() and (p1[:, 0] < W).all() and (p1[:, 1] < H).all()
        assert (p2 > 0).all() and (p2[:, 0] < W).all() and (p2[:, 1] < H).all()
        det1.append(
            DetectionResult(
                success=True,
                board_type="dotboard",
                image_points=p1,
                board_local_points=np.column_stack(
                    [gi_full * _DOT_SPACING_MM, np.zeros(len(gi_full))]
                ),
                grid_indices=gi_full,
                spacing_mm=_DOT_SPACING_MM,
            )
        )
        det2.append(
            DetectionResult(
                success=True,
                board_type="dotboard",
                image_points=p2[keep2],
                board_local_points=np.column_stack(
                    [gi_clip * _DOT_SPACING_MM, np.zeros(len(gi_clip))]
                ),
                grid_indices=gi_clip,
                spacing_mm=_DOT_SPACING_MM,
            )
        )
        if v == 0:
            datum_px = {1: p1, 2: p2}
    return det1, det2, (W, H), (Rs, Ts.reshape(3) * 1000.0), gi_full, datum_px


def _picks(gi_full, px):
    """Pixel clicks on the ``_PICK_DOTS`` physical dots, from one camera's datum view."""
    out = {}
    for role, (c, r) in _PICK_DOTS.items():
        i = int(np.flatnonzero((gi_full[:, 0] == c) & (gi_full[:, 1] == r))[0])
        out[role] = np.asarray(px[i], dtype=float)
    return out


def test_stereo_dotboard_clipped_cam2_requires_clicks2():
    """A clipped camera 2 re-zeroes its dot indices, so camera 1's frame cannot label it.

    Without ``clicks2`` the solve must refuse naming camera 2 (until 2026-10-19 it silently
    reused camera 1's frame and corresponded dots ``_CAM2_CLIP_COLS`` columns apart). With
    picks on the same physical dots in both cameras the pair recovers the rig exactly.
    """
    det1, det2, (W, H), (R_gt, T_gt_mm), gi_full, datum_px = _clipped_dotboard_pair()
    imgs = [np.zeros((H, W), np.uint8)] * len(det1)
    sc = StereoCalibrator(_NoDetect(), "dotboard")
    clicks1 = _picks(gi_full, datum_px[1])

    with pytest.raises(ValueError, match="camera 2"):
        sc.run_stereo(
            imgs,
            imgs,
            1,
            2,
            clicks=clicks1,
            spacing_mm=_DOT_SPACING_MM,
            det1=det1,
            det2=det2,
        )

    rec = sc.run_stereo(
        imgs,
        imgs,
        1,
        2,
        clicks=clicks1,
        clicks2=_picks(gi_full, datum_px[2]),
        spacing_mm=_DOT_SPACING_MM,
        det1=det1,
        det2=det2,
    )
    # Exact projections: correctly labelled dots correspond to ~0.005 px. Mislabelled
    # dots (3 columns apart) would sit tens of pixels off.
    assert rec.board_meta["stereo_rms_px"] < 0.05
    assert rec.board_meta["n_stereo_views"] == len(det1)
    ang = np.degrees(
        np.arccos(np.clip((np.trace(rec.R_stereo @ R_gt.T) - 1) / 2, -1, 1))
    )
    assert ang < 0.05
    # 1 mm as in test_charuco_stereo_recovery: the residual (~0.3 mm in z) is camera 2's
    # focal length fitted 0.1 % off from its clipped board, not the correspondence.
    np.testing.assert_allclose(np.asarray(rec.T_stereo).reshape(3), T_gt_mm, atol=1.0)


def test_stereo_correspondence_refuses_unequal_view_counts():
    """Views pair by index; a shorter camera-2 list must raise, not truncate (zip)."""
    from pivtools_gui.calibration.stereo_model import _stereo_correspondence

    det1, det2, *_ = _clipped_dotboard_pair(n_views=4)
    wf = WF.resolve_world_frame(det1[0].grid_indices, det1[0].image_points, None)
    with pytest.raises(ValueError, match="4 views and camera 2 has 3"):
        _stereo_correspondence(det1, det2[:3], wf, wf, _DOT_SPACING_MM)


# ---------------------------------------------------------------------------
# 3C reconstruction + apply file IO
# ---------------------------------------------------------------------------


def test_3c_reconstruction(stereo_render):
    imgs1, imgs2 = stereo_render["charuco"]
    det = CharucoBoardDetector(CHARUCO)
    rec = StereoCalibrator(det, "charuco").run_stereo(
        imgs1, imgs2, 1, 2, clicks=None, spacing_mm=CHARUCO.square_size_mm
    )
    d1 = det.detect(imgs1[0])
    wp = WF.apply_world_frame(d1.grid_indices, CHARUCO.square_size_mm, rec.world_frame)
    s = (wp[:, 1] - wp[:, 1].min()) / (np.ptp(wp[:, 1]) + 1e-9)
    vel = np.column_stack([0.2 + 0.6 * s, 0.1 + 0.8 * s, -0.1 + 0.5 * s])
    m1, m2 = rec.model1, rec.model2
    d1px = m1.project(wp + vel) - m1.project(wp)
    d2px = m2.project(wp + vel) - m2.project(wp)
    # The right-hand-rule solve recovers the prescribed field exactly: W = u_z in the
    # same right-handed world frame as the coordinates (the 4x3 Jacobian math), no flip.
    # There is no toward-cameras opt-in — W and world_z always share one frame.
    vr = reconstruct_3c_at_points(m1, m2, wp, d1px, d2px)
    assert np.nanmax(np.abs(vr - vel)) < 0.02


def test_apply_mono_fileio(tmp_path):
    """Round-trip the production coordinates.mat + B*.mat apply path; no Y flip."""
    import scipy.io

    K = np.array([[8000.0, 0, 800], [0, 8000, 600], [0, 0, 1]])
    cam = CameraModel(
        K, np.zeros(5), np.eye(3), np.array([[-50.0], [-40.0], [1000.0]]), (1600, 1200)
    )
    rec = REC.MonoRecord(
        camera=1, board_type="dotboard", camera_model=cam, world_frame=REC.WorldFrame()
    )
    X, Y = np.meshgrid(np.linspace(200, 1400, 20) + 1, np.linspace(150, 1050, 16) + 1)
    ud, od = tmp_path / "u", tmp_path / "c"
    ud.mkdir()
    cs = np.empty((1,), dtype=[("x", object), ("y", object)])
    cs["x"][0], cs["y"][0] = X.astype(np.float32), Y.astype(np.float32)
    scipy.io.savemat(str(ud / "coordinates.mat"), {"coordinates": cs}, oned_as="row")
    ps = np.empty((1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
    ps["ux"][0] = np.full(X.shape, 3.0, np.float16)
    ps["uy"][0] = np.full(X.shape, -2.0, np.float16)
    ps["b_mask"][0] = np.ones(X.shape, bool)
    scipy.io.savemat(str(ud / "B00001.mat"), {"piv_result": ps}, oned_as="row")
    runio.calibrate_mono_run(rec, ud, od, dt=0.001)
    bb = scipy.io.loadmat(str(od / "B00001.mat"), squeeze_me=True)["piv_result"]
    u = np.asarray(bb["ux"].tolist() if bb["ux"].dtype == object else bb["ux"])
    v = np.asarray(bb["uy"].tolist() if bb["uy"].dtype == object else bb["uy"])
    assert float(np.nanmean(u)) == pytest.approx(0.375, abs=0.01)
    assert float(np.nanmean(v)) == pytest.approx(-0.25, abs=0.01)  # no Y flip


def test_apply_mono_skips_diagnostic_mats(tmp_path):
    """store_planes/save_diagnostics sidecars must not be mistaken for vector files.

    Ensemble PIV drops ``planes_pass_N.mat`` (correlation planes) and ``warped_pass_N.mat``
    (first-pair warped images) into the same output dir. The apply vector glob is a bare
    ``*.mat``, so without an exclusion these reach ``read_vectors`` and KeyError on
    'piv_result'. Apply must skip them and calibrate only the real vector file.
    """
    import scipy.io

    K = np.array([[8000.0, 0, 800], [0, 8000, 600], [0, 0, 1]])
    cam = CameraModel(
        K, np.zeros(5), np.eye(3), np.array([[-50.0], [-40.0], [1000.0]]), (1600, 1200)
    )
    rec = REC.MonoRecord(
        camera=1, board_type="dotboard", camera_model=cam, world_frame=REC.WorldFrame()
    )
    X, Y = np.meshgrid(np.linspace(200, 1400, 20) + 1, np.linspace(150, 1050, 16) + 1)
    ud, od = tmp_path / "u", tmp_path / "c"
    ud.mkdir()
    cs = np.empty((1,), dtype=[("x", object), ("y", object)])
    cs["x"][0], cs["y"][0] = X.astype(np.float32), Y.astype(np.float32)
    scipy.io.savemat(str(ud / "coordinates.mat"), {"coordinates": cs}, oned_as="row")
    ps = np.empty((1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
    ps["ux"][0] = np.full(X.shape, 3.0, np.float16)
    ps["uy"][0] = np.full(X.shape, -2.0, np.float16)
    ps["b_mask"][0] = np.ones(X.shape, bool)
    scipy.io.savemat(str(ud / "00001.mat"), {"piv_result": ps}, oned_as="row")
    # Diagnostic sidecars — no piv_result/ensemble_result struct; must be ignored.
    scipy.io.savemat(
        str(ud / "planes_pass_1.mat"),
        {"AA": np.zeros((2, 2)), "pass_idx": 0},
        oned_as="row",
    )
    scipy.io.savemat(
        str(ud / "warped_pass_1.mat"),
        {"A_warped": np.zeros((2, 2)), "pass_idx": 0},
        oned_as="row",
    )
    scipy.io.savemat(
        str(ud / "fit_diagnostics_pass_1.mat"),
        {"N0_abs": np.zeros((2, 2)), "pass_idx": 0},
        oned_as="row",
    )

    written = runio.calibrate_mono_run(rec, ud, od, dt=0.001, vector_glob="*.mat")

    assert {Path(w).name for w in written} == {"00001.mat"}
    assert not (od / "planes_pass_1.mat").exists()
    assert not (od / "warped_pass_1.mat").exists()
    assert not (od / "fit_diagnostics_pass_1.mat").exists()


def test_apply_mono_parallel_pool(tmp_path, monkeypatch):
    """Multi-file apply exercises the ProcessPool fan-out (1-file tests run inline).

    Each frame carries a distinct displacement so a mixed-up frame/output pairing
    would fail the per-file value check; progress must tick once per frame.
    Workers are forced >1 so the pool path runs regardless of the ambient
    post_processing_workers knob (knob=1 would silently test the inline path).
    """
    import scipy.io

    monkeypatch.setattr(runio, "get_max_workers", lambda n_items=None, config=None: 4)

    K = np.array([[8000.0, 0, 800], [0, 8000, 600], [0, 0, 1]])
    cam = CameraModel(
        K, np.zeros(5), np.eye(3), np.array([[-50.0], [-40.0], [1000.0]]), (1600, 1200)
    )
    rec = REC.MonoRecord(
        camera=1, board_type="dotboard", camera_model=cam, world_frame=REC.WorldFrame()
    )
    X, Y = np.meshgrid(np.linspace(200, 1400, 20) + 1, np.linspace(150, 1050, 16) + 1)
    ud, od = tmp_path / "u", tmp_path / "c"
    ud.mkdir()
    cs = np.empty((1,), dtype=[("x", object), ("y", object)])
    cs["x"][0], cs["y"][0] = X.astype(np.float32), Y.astype(np.float32)
    scipy.io.savemat(str(ud / "coordinates.mat"), {"coordinates": cs}, oned_as="row")
    n_frames = 6
    for k in range(n_frames):
        ps = np.empty((1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
        ps["ux"][0] = np.full(X.shape, float(k + 1), np.float64)
        ps["uy"][0] = np.full(X.shape, -2.0, np.float64)
        ps["b_mask"][0] = np.ones(X.shape, bool)
        scipy.io.savemat(str(ud / f"B{k+1:05d}.mat"), {"piv_result": ps}, oned_as="row")

    progress = []
    written = runio.calibrate_mono_run(
        rec, ud, od, dt=0.001, progress_cb=lambda d, t: progress.append((d, t))
    )

    assert [Path(w).name for w in written] == [
        f"B{k+1:05d}.mat" for k in range(n_frames)
    ]
    assert progress == [(k + 1, n_frames) for k in range(n_frames)]
    # fx=8000, Z~1000mm -> 1 px = 0.125 mm; dt=0.001 s -> k+1 px = 0.125*(k+1) m/s
    for k in range(n_frames):
        bb = scipy.io.loadmat(str(od / f"B{k+1:05d}.mat"), squeeze_me=True)[
            "piv_result"
        ]
        u = np.asarray(bb["ux"].tolist() if bb["ux"].dtype == object else bb["ux"])
        assert float(np.nanmean(u)) == pytest.approx(0.125 * (k + 1), abs=0.01)


def test_apply_mono_worker_failure_raises(tmp_path, monkeypatch):
    """A corrupt vector file must fail the whole apply loudly (no silent skip),
    from the pool path too — the worker exception propagates via future.result()."""
    import scipy.io

    monkeypatch.setattr(runio, "get_max_workers", lambda n_items=None, config=None: 4)
    K = np.array([[8000.0, 0, 800], [0, 8000, 600], [0, 0, 1]])
    cam = CameraModel(
        K, np.zeros(5), np.eye(3), np.array([[-50.0], [-40.0], [1000.0]]), (1600, 1200)
    )
    rec = REC.MonoRecord(
        camera=1, board_type="dotboard", camera_model=cam, world_frame=REC.WorldFrame()
    )
    X, Y = np.meshgrid(np.linspace(200, 1400, 20) + 1, np.linspace(150, 1050, 16) + 1)
    ud, od = tmp_path / "u", tmp_path / "c"
    ud.mkdir()
    cs = np.empty((1,), dtype=[("x", object), ("y", object)])
    cs["x"][0], cs["y"][0] = X.astype(np.float32), Y.astype(np.float32)
    scipy.io.savemat(str(ud / "coordinates.mat"), {"coordinates": cs}, oned_as="row")
    for k in range(3):
        ps = np.empty((1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)])
        ps["ux"][0] = np.full(X.shape, 3.0, np.float64)
        ps["uy"][0] = np.full(X.shape, -2.0, np.float64)
        ps["b_mask"][0] = np.ones(X.shape, bool)
        scipy.io.savemat(str(ud / f"B{k+1:05d}.mat"), {"piv_result": ps}, oned_as="row")
    (ud / "B00004.mat").write_bytes(b"this is not a mat file")

    with pytest.raises(Exception):
        runio.calibrate_mono_run(rec, ud, od, dt=0.001)


def test_reconstruct_stereo_run_parallel(stereo_render, tmp_path, monkeypatch):
    """reconstruct_stereo_run plumbing over several frames (pool path): zero pixel
    displacement in both cameras must reconstruct to ~zero 3C velocity on the
    regular world grid, one output per common frame + one coordinates.mat.
    (The 3C math itself is covered by test_3c_reconstruction; workers forced >1
    so the pool path runs regardless of the ambient knob.)"""
    import scipy.io

    monkeypatch.setattr(runio, "get_max_workers", lambda n_items=None, config=None: 3)

    imgs1, imgs2 = stereo_render["charuco"]
    det = CharucoBoardDetector(CHARUCO)
    rec = StereoCalibrator(det, "charuco").run_stereo(
        imgs1, imgs2, 1, 2, clicks=None, spacing_mm=CHARUCO.square_size_mm
    )
    # Pixel coordinate grids: project one world-mm grid on the sheet through each model.
    d1 = det.detect(imgs1[0])
    wp = WF.apply_world_frame(d1.grid_indices, CHARUCO.square_size_mm, rec.world_frame)
    gx = np.linspace(wp[:, 0].min(), wp[:, 0].max(), 12)
    gy = np.linspace(wp[:, 1].min(), wp[:, 1].max(), 10)
    Xw, Yw = np.meshgrid(gx, gy)
    pts = np.column_stack([Xw.ravel(), Yw.ravel(), np.zeros(Xw.size)])
    px1 = rec.model1.project(pts).reshape(Xw.shape + (2,))
    px2 = rec.model2.project(pts).reshape(Xw.shape + (2,))

    n_frames = 3
    for cam_i, px in ((1, px1), (2, px2)):
        ud = tmp_path / f"u{cam_i}"
        ud.mkdir()
        cs = np.empty((1,), dtype=[("x", object), ("y", object)])
        cs["x"][0], cs["y"][0] = px[..., 0], px[..., 1]
        scipy.io.savemat(
            str(ud / "coordinates.mat"), {"coordinates": cs}, oned_as="row"
        )
        for k in range(n_frames):
            ps = np.empty(
                (1,), dtype=[("ux", object), ("uy", object), ("b_mask", object)]
            )
            ps["ux"][0] = np.zeros(Xw.shape)
            ps["uy"][0] = np.zeros(Xw.shape)
            ps["b_mask"][0] = np.ones(Xw.shape, bool)
            scipy.io.savemat(
                str(ud / f"B{k+1:05d}.mat"), {"piv_result": ps}, oned_as="row"
            )

    od = tmp_path / "stereo_out"
    progress = []
    written, n_skipped = runio.reconstruct_stereo_run(
        rec,
        tmp_path / "u1",
        tmp_path / "u2",
        od,
        dt=0.001,
        progress_cb=lambda d, t: progress.append((d, t)),
    )

    assert len(written) == n_frames
    assert n_skipped == 0
    assert progress == [(k + 1, n_frames) for k in range(n_frames)]
    assert (od / "coordinates.mat").exists()
    bb = scipy.io.loadmat(str(od / "B00001.mat"), squeeze_me=True)["piv_result"]
    for comp in ("ux", "uy", "uz"):
        arr = np.asarray(
            bb[comp].tolist() if bb[comp].dtype == object else bb[comp], dtype=float
        )
        assert np.nanmax(np.abs(arr)) < 1e-6


def test_reconstruct_stereo_run_reports_skipped_frames(tmp_path):
    """A frame pair whose reconstructed pass is empty in one camera is skipped and COUNTED.

    The pass reconstructed is the final populated one of coordinates.mat (pass 2 of the
    fixture layout). Camera 2's frame 3 is rewritten with that pass empty, so it cannot be
    reconstructed: it must be absent from ``written`` and reported in ``n_skipped`` while
    progress still advances over every frame.
    """
    import scipy.io
    from fixtures.vector_sets import (
        DEFAULT_PASSES,
        VECTOR_NAME_FORMAT,
        GridSpec,
        _frame_piv_result,
        write_stereo_vector_sets,
    )
    from test_calibration_self_cal import _stereo_record

    n_frames, skipped_frame = 4, 3
    # Inside the overlap of _stereo_record's 1600x1200 pair (as test_calibration_apply_chain).
    grid = GridSpec(origin_px=(600.0, 450.0), extent_px=(400.0, 300.0))
    write_stereo_vector_sets(tmp_path / "u1", tmp_path / "u2", n_frames, grid=grid)
    no_final_pass = DEFAULT_PASSES[:-1] + (None,)
    scipy.io.savemat(
        str(tmp_path / "u2" / VECTOR_NAME_FORMAT.format(skipped_frame)),
        {"piv_result": _frame_piv_result(skipped_frame, no_final_pass, grid)},
    )

    progress = []
    written, n_skipped = runio.reconstruct_stereo_run(
        _stereo_record(),
        tmp_path / "u1",
        tmp_path / "u2",
        tmp_path / "out",
        dt=1e-3,
        progress_cb=lambda d, t: progress.append((d, t)),
        vector_glob="[0-9]*.mat",
        interpolator="linear",
    )

    assert n_skipped == 1
    assert len(written) == n_frames - 1
    assert VECTOR_NAME_FORMAT.format(skipped_frame) not in {
        Path(w).name for w in written
    }
    assert progress[-1] == (n_frames, n_frames)


def test_record_save_load_roundtrip(tmp_path):
    K = np.array([[5000.0, 0, 400], [0, 5000, 300], [0, 0, 1]])
    cam = CameraModel(
        K,
        np.array([-0.01, 0.02, 0.0, 0.0, 0.0]),
        np.eye(3),
        np.array([[1.0], [2.0], [900.0]]),
        (800, 600),
    )
    wf = REC.WorldFrame(
        mode="clicks",
        origin_px=np.array([1.0, 2.0]),
        x_axis_px=np.array([3.0, 4.0]),
        y_axis_px=np.array([5.0, 6.0]),
        swap_axes=True,
        col_sign=-1,
        row_sign=1,
        origin_grid=np.array([2.0, 3.0]),
    )
    rec = REC.MonoRecord(
        camera=2,
        board_type="charuco",
        camera_model=cam,
        world_frame=wf,
        per_view_rms=[0.1, 0.2],
        board_meta={"spacing_mm": 30.0, "n_views": 2},
    )
    p = REC.save_mono(rec, tmp_path)
    r2 = REC.load_mono(p)
    assert np.allclose(r2.camera_model.K, cam.K)
    assert np.allclose(r2.camera_model.dist, cam.dist)
    assert r2.world_frame.swap_axes and r2.world_frame.col_sign == -1
    assert np.allclose(r2.world_frame.origin_grid, [2.0, 3.0])
    assert r2.board_meta["spacing_mm"] == pytest.approx(30.0)


def test_stereo_correspondence_matches_and_empties():
    """The cross-camera matcher pairs shared world points and yields nothing when the
    two cameras share none (the fail-loud signal run_stereo raises on for a non-same-side
    pair)."""
    from pivtools_gui.calibration.detection.base import DetectionResult
    from pivtools_gui.calibration.stereo_model import _stereo_correspondence
    from pivtools_gui.calibration.world_frame import resolve_world_frame_from_grid

    wf = resolve_world_frame_from_grid([0, 0], [1, 0], [0, 1])

    def det(grid):
        gi = np.asarray(grid, dtype=np.int64)
        return DetectionResult(
            success=True,
            board_type="dotboard",
            image_points=gi.astype(np.float64) * 10.0,
            board_local_points=np.column_stack(
                [gi.astype(np.float64), np.zeros(len(gi))]
            ),
            grid_indices=gi,
            point_ids=None,
            board_to_pixel=None,
            spacing_mm=1.0,
            synthetic_mask=None,
            diagnostics={},
        )

    shared = [
        [c, r] for r in range(3) for c in range(3)
    ]  # 9 shared dots (> MIN_SHARED_POINTS)
    obj, i1, i2 = _stereo_correspondence([det(shared)], [det(shared)], wf, wf, 1.0)
    assert len(obj) == 1 and obj[0].shape == (9, 3)
    assert i1[0].shape == (9, 2) and i2[0].shape == (9, 2)

    # cam2 sees a board shifted far away -> no shared world coordinates -> empty.
    disjoint = [[c + 100, r + 100] for r in range(3) for c in range(3)]
    obj2, _, _ = _stereo_correspondence([det(shared)], [det(disjoint)], wf, wf, 1.0)
    assert obj2 == []


def test_stereo_figures_render(stereo_render, tmp_path):
    """run_stereo with figure_dir completes and writes the cam2 + stereo proof figures
    using the stereoCalibrate-derived cam2 pose (the figure block is unguarded), and
    stamps the new stereo_rms_px / stereo_method meta."""
    imgs1, imgs2 = stereo_render["charuco"]
    det = CharucoBoardDetector(CHARUCO)
    figd = tmp_path / "figures"
    rec = StereoCalibrator(det, "charuco").run_stereo(
        imgs1,
        imgs2,
        1,
        2,
        clicks=None,
        spacing_mm=CHARUCO.square_size_mm,
        figure_dir=figd,
    )
    assert rec.board_meta["stereo_method"] == "stereoCalibrate"
    assert np.isfinite(rec.board_meta["stereo_rms_px"])
    pngs = [p.name for p in figd.glob("*.png")]
    assert any(n.startswith("cam2_") for n in pngs)
    assert any(("cameras_3d" in n) or ("dewarp_overlay" in n) for n in pngs)
