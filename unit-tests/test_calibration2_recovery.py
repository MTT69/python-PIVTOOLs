"""End-to-end recovery tests for the calibration2 package.

Hermetic tracks (always run): synthetic ChArUco (OpenCV) and dotboard (OpenCV)
calibration recovery — planar + stereo — plus the world-frame resolver, distortion
recovery, 3C reconstruction, the file-IO apply path, and the ChArUco-Y regression.

The SIG dotboard + 3C track lives in test_calibration2_sig.py (skipped if the SIG
binary/ncgen are absent), to keep this suite fast and dependency-free.

Run with ``--make-figures`` to drop diagnostic PNGs in PyPIVTools/figures/debug/.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration2.detection.charuco import CharucoBoardDetector, CharucoParams
from pivtools_gui.calibration2.detection.dotboard import DotboardDetector, DotboardParams
from pivtools_gui.calibration2.pipeline import Calibrator
from pivtools_gui.calibration2.stereo_model import (
    StereoCalibrator, reconstruct_3c_at_points, camera_z_sign,
)
from pivtools_gui.calibration2.camera_model import (
    CameraModel, DistortionModel, fit_intrinsics, fit_pose,
)
from pivtools_gui.calibration2 import world_frame as WF
from pivtools_gui.calibration2 import record as REC
from pivtools_gui.calibration2 import runio

from pivtools_cli.generate_synthetic_charuco import generate_charuco_dataset
from pivtools_cli.generate_synthetic_stereo import (
    make_camera_matrix, make_poses, compose_stereo_poses, make_stereo_transform,
    generate_charuco_images, generate_dotboard_images,
    DOT_COLS, DOT_ROWS, DOT_SPACING_MM,
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
        output_dir=out, n_views=12, megapixels=1.0,
        sq_h=10, sq_v=7, sq_size=0.030, marker_ratio=0.5,
        dict_id=cv2.aruco.DICT_4X4_1000,
    )
    gt = np.load(out.parent / "ground_truth.npz")
    imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted(out.glob("calib*.png"))]
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
    # dotboard
    dbw = (DOT_COLS - 1) * DOT_SPACING_MM / 1000.0
    dc = np.array([dbw / 2, (DOT_ROWS - 1) * DOT_SPACING_MM / 2000.0, 0])
    dp1 = make_poses(12, dc, fx, W, dbw, (0.80, 0.55))
    dp2 = compose_stereo_poses(dp1, Rs, Ts)
    generate_dotboard_images(d / "dot1", cam, dp1, W, H)
    generate_dotboard_images(d / "dot2", cam, dp2, W, H)
    load = lambda p: [cv2.imread(str(f), 0) for f in sorted(p.glob("calib*.png"))]
    return {
        "cam": cam, "R_stereo": Rs, "T_stereo": Ts,
        "charuco": (load(d / "ch1"), load(d / "ch2")),
        "dotboard": (load(d / "dot1"), load(d / "dot2"), dp1, dp2),
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
    rec = Calibrator(det, "charuco").run_mono(imgs, camera=1, spacing_mm=CHARUCO.square_size_mm)
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
        tvec = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), 700 + rng.uniform(-40, 40)])
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
    px = lambda cid: d0.image_points[ids.index(cid)]
    clicks = {"origin": px(0), "x_axis": px(1), "y_axis": px(cols)}
    wf = WF.resolve_world_frame(d0.grid_indices, d0.image_points, clicks)
    w = WF.apply_world_frame(d0.grid_indices, CHARUCO.square_size_mm, wf)
    w_o, w_x, w_y = w[ids.index(0)], w[ids.index(1)], w[ids.index(cols)]
    assert np.allclose(w_o[:2], [0, 0], atol=1e-6)
    assert np.dot(w_x[:2] - w_o[:2], w_y[:2] - w_o[:2]) == pytest.approx(0.0, abs=1e-6)
    assert w_x[0] > 0 and abs(w_x[1]) < 1e-6
    assert w_y[1] > 0 and abs(w_y[0]) < 1e-6


def test_charuco_y_not_inverted(charuco_planar):
    """Regression: calibration2 output must be physics-correct in Y (no inversion)."""
    imgs, _ = charuco_planar
    det = CharucoBoardDetector(CHARUCO)
    rec = Calibrator(det, "charuco").run_mono(imgs, camera=1, spacing_mm=CHARUCO.square_size_mm)
    d0 = det.detect(imgs[0])
    ids = d0.point_ids.tolist()
    cols = CHARUCO.interior_cols
    w = WF.apply_world_frame(d0.grid_indices, CHARUCO.square_size_mm, rec.world_frame)
    R, t = fit_pose(w, d0.image_points, rec.camera_model.K, rec.camera_model.dist, planar=True)
    cam = CameraModel(rec.camera_model.K, rec.camera_model.dist, R, t, rec.camera_model.image_size)
    # A feature one row in +Y must back-project to world Y > 0.
    wb = cam.back_project_to_plane(d0.image_points[ids.index(cols)][None, :])[0]
    assert wb[1] > 0


def test_charuco_stereo_recovery(stereo_render):
    imgs1, imgs2 = stereo_render["charuco"]
    det = CharucoBoardDetector(CHARUCO)
    rec = StereoCalibrator(det, "charuco").run_stereo(
        imgs1, imgs2, 1, 2, clicks=None, spacing_mm=CHARUCO.square_size_mm)
    ang = np.degrees(np.arccos(np.clip(
        (np.trace(rec.R_stereo @ stereo_render["R_stereo"].T) - 1) / 2, -1, 1)))
    assert ang < 0.3
    assert float(np.linalg.norm(rec.T_stereo)) == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# Dotboard
# ---------------------------------------------------------------------------

def test_dotboard_mono_recovery(stereo_render, make_figures):
    imgs1, _imgs2, _p1, _p2 = stereo_render["dotboard"]
    det = DotboardDetector(DotboardParams(DOT_SPACING_MM))
    d0 = det.detect(imgs1[0])
    assert d0.success and d0.n >= 170
    rec = Calibrator(det, "dotboard").run_mono(imgs1, camera=1, spacing_mm=DOT_SPACING_MM)
    assert rec.camera_model.rms < 1.0
    assert abs(rec.camera_model.K[0, 0] - 1000) / 1000 < 0.01
    if make_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(imgs1[0], cmap="gray")
        ax.scatter(d0.image_points[:, 0], d0.image_points[:, 1], s=8, c="lime")
        ax.set_title(f"dotboard detect {d0.n} dots (fx={rec.camera_model.K[0,0]:.0f})")
        fig.savefig(_figpath("dotboard-detect"), dpi=110)
        plt.close(fig)


def test_dotboard_stereo_recovery(stereo_render):
    imgs1, imgs2, p1, p2 = stereo_render["dotboard"]
    cam = stereo_render["cam"]
    det = DotboardDetector(DotboardParams(DOT_SPACING_MM))

    def clicks_for(pose):
        rvec, tvec = pose
        objs = np.array([[0, 0, 0], [0.015, 0, 0], [0, 0.015, 0]], float)
        pr, _ = cv2.projectPoints(objs, rvec, tvec, cam, np.zeros(5))
        pr = pr.reshape(-1, 2)
        return {"origin": pr[0], "x_axis": pr[1], "y_axis": pr[2]}

    rec = StereoCalibrator(det, "dotboard").run_stereo(
        imgs1, imgs2, 1, 2,
        clicks=clicks_for(p1[0]), clicks2=clicks_for(p2[0]),
        spacing_mm=DOT_SPACING_MM)
    ang = np.degrees(np.arccos(np.clip(
        (np.trace(rec.R_stereo @ stereo_render["R_stereo"].T) - 1) / 2, -1, 1)))
    assert ang < 0.5
    assert float(np.linalg.norm(rec.T_stereo)) == pytest.approx(50.0, abs=1.5)


# ---------------------------------------------------------------------------
# 3C reconstruction + apply file IO
# ---------------------------------------------------------------------------

def test_3c_reconstruction(stereo_render):
    imgs1, imgs2 = stereo_render["charuco"]
    det = CharucoBoardDetector(CHARUCO)
    rec = StereoCalibrator(det, "charuco").run_stereo(
        imgs1, imgs2, 1, 2, clicks=None, spacing_mm=CHARUCO.square_size_mm)
    d1 = det.detect(imgs1[0])
    wp = WF.apply_world_frame(d1.grid_indices, CHARUCO.square_size_mm, rec.world_frame)
    s = (wp[:, 1] - wp[:, 1].min()) / (np.ptp(wp[:, 1]) + 1e-9)
    vel = np.column_stack([0.2 + 0.6 * s, 0.1 + 0.8 * s, -0.1 + 0.5 * s])
    m1, m2 = rec.model1, rec.model2
    d1px = m1.project(wp + vel) - m1.project(wp)
    d2px = m2.project(wp + vel) - m2.project(wp)
    # Raw solve recovers the prescribed field exactly (the 4x3 Jacobian math).
    vr_raw = reconstruct_3c_at_points(m1, m2, wp, d1px, d2px, z_toward_cameras=False)
    assert np.nanmax(np.abs(vr_raw - vel)) < 0.02
    # The toward-cameras convention (default) only flips w by the camera-side sign;
    # u, v are untouched.
    vr = reconstruct_3c_at_points(m1, m2, wp, d1px, d2px)
    zs = camera_z_sign(m1, m2)
    assert np.allclose(vr[:, :2], vr_raw[:, :2])
    assert np.allclose(vr[:, 2], zs * vr_raw[:, 2])


def test_apply_mono_fileio(tmp_path):
    """Round-trip the production coordinates.mat + B*.mat apply path; no Y flip."""
    import scipy.io
    K = np.array([[8000.0, 0, 800], [0, 8000, 600], [0, 0, 1]])
    cam = CameraModel(K, np.zeros(5), np.eye(3), np.array([[-50.0], [-40.0], [1000.0]]), (1600, 1200))
    rec = REC.MonoRecord(camera=1, board_type="dotboard", camera_model=cam, world_frame=REC.WorldFrame())
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


def test_record_save_load_roundtrip(tmp_path):
    K = np.array([[5000.0, 0, 400], [0, 5000, 300], [0, 0, 1]])
    cam = CameraModel(K, np.array([-0.01, 0.02, 0.0, 0.0, 0.0]), np.eye(3),
                      np.array([[1.0], [2.0], [900.0]]), (800, 600))
    wf = REC.WorldFrame(mode="clicks", origin_px=np.array([1.0, 2.0]),
                        x_axis_px=np.array([3.0, 4.0]), y_axis_px=np.array([5.0, 6.0]),
                        swap_axes=True, col_sign=-1, row_sign=1, origin_grid=np.array([2.0, 3.0]))
    rec = REC.MonoRecord(camera=2, board_type="charuco", camera_model=cam, world_frame=wf,
                         per_view_rms=[0.1, 0.2], board_meta={"spacing_mm": 30.0, "n_views": 2})
    p = REC.save_mono(rec, tmp_path)
    r2 = REC.load_mono(p)
    assert np.allclose(r2.camera_model.K, cam.K)
    assert np.allclose(r2.camera_model.dist, cam.dist)
    assert r2.world_frame.swap_axes and r2.world_frame.col_sign == -1
    assert np.allclose(r2.world_frame.origin_grid, [2.0, 3.0])
    assert r2.board_meta["spacing_mm"] == pytest.approx(30.0)
