"""Hermetic tests for the stepped-board 3D polynomial model (calibration, S7).

Covers the new ``Polynomial3DModel`` (world->image cubic, degree-3 in X,Y x degree-1
in Z): fit recovery, the analytic Jacobian, the Newton back-projection (flat + tilted
sheet), .mat round-trips (mono + the R/T-less stereo pair), and model-agnostic 3C
reconstruction with two polynomial models. No DaVis data, no GUI, no network.
"""

import numpy as np
import pytest

from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.camera_model import (
    Polynomial3DModel,
    _poly_basis_3d,
    fit_polynomial3d,
)
from pivtools_gui.calibration.record import WorldFrame
from pivtools_gui.calibration.stereo_model import camera_z_sign, reconstruct_3c_at_points

np.seterr(all="ignore")  # macOS Accelerate raises spurious matmul warnings; results are finite.

DOT_SPACING_MM = 28.89
STEP_MM = 3.0
INTERLEAVE_MM = 7.5
IMAGE_SIZE = (2048, 2048)


def _two_plane_board(n: int = 15) -> np.ndarray:
    """A stepped datum view's world points: peak at Z=0, trough at Z=-step (+interleave)."""
    xs = np.arange(n) * DOT_SPACING_MM
    xs = xs - xs.mean()
    X, Y = np.meshgrid(xs, xs)
    peak = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    trough = np.column_stack(
        [X.ravel() + INTERLEAVE_MM, Y.ravel() + INTERLEAVE_MM, np.full(X.size, -STEP_MM)]
    )
    return np.vstack([peak, trough])


def _synthesise(world: np.ndarray, spec_u: dict, spec_v: dict) -> np.ndarray:
    """Image points from a KNOWN forward cubic, so a correct fit is exact."""
    p = (world[:, 0] - world[:, 0].mean()) / (np.ptp(world[:, 0]) / 2)
    q = (world[:, 1] - world[:, 1].mean()) / (np.ptp(world[:, 1]) / 2)
    r = (world[:, 2] + STEP_MM / 2) / (STEP_MM / 2)
    a = _poly_basis_3d(p, q, r)
    cu, cv = np.zeros(20), np.zeros(20)
    for i, v in spec_u.items():
        cu[i] = v
    for i, v in spec_v.items():
        cv[i] = v
    return np.column_stack([a @ cu, a @ cv])


@pytest.fixture
def world():
    return _two_plane_board()


@pytest.fixture
def model(world):
    img = _synthesise(world, {0: 1024, 1: 900, 4: 20, 10: 8}, {0: 1024, 4: 900, 1: -15, 14: 6})
    return fit_polynomial3d(world, img, IMAGE_SIZE)


def test_fit_recovers_known_map_to_machine_precision(world):
    img = _synthesise(world, {0: 1024, 1: 900, 4: 20, 10: 8}, {0: 1024, 4: 900, 1: -15, 14: 6})
    m = fit_polynomial3d(world, img, IMAGE_SIZE)
    assert m.model_type == "polynomial3d"
    assert m.rms_px < 1e-6
    assert len(m.plane_rms_px) == 2  # two Z levels
    np.testing.assert_allclose(m.project(world), img, atol=1e-6)


def test_requires_two_distinct_planes():
    flat = np.column_stack([np.random.default_rng(0).normal(size=(40, 2)) * 50, np.zeros(40)])
    with pytest.raises(RuntimeError, match=r">=2 distinct Z planes"):
        fit_polynomial3d(flat, np.random.default_rng(1).normal(size=(40, 2)), IMAGE_SIZE)


def test_analytic_jacobian_matches_finite_difference(model, world):
    wp = world[::23]
    j_analytic = model.jacobian(wp)
    j_fd = np.empty_like(j_analytic)
    h = 1e-3
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        j_fd[:, :, k] = (model.project(wp + step) - model.project(wp - step)) / (2 * h)
    np.testing.assert_allclose(j_analytic, j_fd, atol=1e-6)


@pytest.mark.parametrize("z_sheet", [0.0, -STEP_MM, -1.5])
def test_back_projection_round_trip_flat_sheet(model, world, z_sheet):
    # Points known to lie on the sheet z = z_sheet.
    xy = world[::17, :2]
    wp = np.column_stack([xy, np.full(len(xy), z_sheet)])
    px = model.project(wp)
    bp = model.back_project_to_plane(px, z_world=z_sheet)
    np.testing.assert_allclose(bp[:, :2], xy, atol=1e-6)
    np.testing.assert_allclose(bp[:, 2], z_sheet, atol=1e-9)


def test_back_projection_round_trip_tilted_sheet(model, world):
    # Sheet contract (shared with the pinhole path):
    #   Z = z_world + X*tan(tilt_y) + Y*tan(tilt_x)
    tilt_x, tilt_y = np.deg2rad(2.5), np.deg2rad(-1.2)
    xy = world[::17, :2]
    z_true = np.tan(tilt_y) * xy[:, 0] + np.tan(tilt_x) * xy[:, 1]
    wp = np.column_stack([xy, z_true])
    px = model.project(wp)
    bp = model.back_project_to_plane(px, z_world=0.0, tilt_x=tilt_x, tilt_y=tilt_y)
    np.testing.assert_allclose(bp[:, :2], xy, atol=1e-5)


def test_tilt_convention_matches_pinhole_contract(model):
    """Identical (z_world, tilt_x, tilt_y) must mean the SAME sheet through the
    polynomial3d model as through the pinhole model (the A1 tilt-axis contract).

    Both back-projections must land on Z = z_world + X*tan(tilt_y) + Y*tan(tilt_x);
    a swapped coupling in either model puts its points on the transposed plane.
    """
    import cv2

    from pivtools_gui.calibration.camera_model import CameraModel, DistortionModel

    z0, tilt_x, tilt_y = 1.5, np.deg2rad(3.0), np.deg2rad(-2.0)

    def plane_residual(world_pts):
        return world_pts[:, 2] - (
            z0
            + world_pts[:, 0] * np.tan(tilt_y)
            + world_pts[:, 1] * np.tan(tilt_x)
        )

    # Polynomial3d: back-project a pixel grid spanning the fitted FOV.
    px_grid = np.column_stack([
        np.linspace(900, 1150, 9),
        np.linspace(900, 1150, 9)[::-1],
    ])
    bp_poly = model.back_project_to_plane(px_grid, z_world=z0, tilt_x=tilt_x, tilt_y=tilt_y)
    np.testing.assert_allclose(plane_residual(bp_poly), 0.0, atol=1e-6)

    # Pinhole: a synthetic camera looking at the board from +Z.
    K = np.array([[2000.0, 0, 1024], [0, 2000.0, 1024], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([np.pi, 0.02, 0.0]))  # board-facing flip
    t = -R @ np.array([[5.0], [-10.0], [800.0]])        # camera centre at (5,-10,800)
    cam = CameraModel(K=K, dist=np.zeros(4), R=R, t=t, image_size=(2048, 2048),
                      distortion_model=DistortionModel.STANDARD, rms=0.0)
    bp_pin = cam.back_project_to_plane(px_grid, z_world=z0, tilt_x=tilt_x, tilt_y=tilt_y)
    np.testing.assert_allclose(plane_residual(bp_pin), 0.0, atol=1e-6)


def test_mono_record_round_trips(model, world, tmp_path):
    wf = WorldFrame(
        mode="clicks", origin_px=np.array([10.0, 10]), x_axis_px=np.array([20.0, 10]),
        y_axis_px=np.array([10.0, 20]), swap_axes=False, col_sign=1, row_sign=1,
        origin_grid=np.array([0.0, 0]), origin_mm=np.array([0.0, 0]),
    )
    mr = rec.MonoRecord(
        camera=1, board_type="stepped", camera_model=model, world_frame=wf,
        per_view_rms=[model.rms_px],
    )
    loaded = rec.load_mono(rec.save_mono(mr, tmp_path))
    assert isinstance(loaded.camera_model, Polynomial3DModel)
    assert loaded.camera_model.model_type == "polynomial3d"
    np.testing.assert_allclose(loaded.camera_model.project(world), model.project(world), atol=1e-9)
    np.testing.assert_allclose(loaded.camera_model.plane_rms_px, model.plane_rms_px, atol=1e-12)


def _stereo_pair(world):
    """Two genuinely different cameras (independent Z-sensitivity -> W observable)."""
    img1 = _synthesise(world, {0: 1024, 1: 900, 4: 20, 10: 30}, {0: 1024, 4: 900, 1: -15})
    img2 = _synthesise(world, {0: 1100, 1: 880, 4: 40}, {0: 980, 4: 870, 1: -30, 10: 30})
    m1 = fit_polynomial3d(world, img1, IMAGE_SIZE)
    m2 = fit_polynomial3d(world, img2, IMAGE_SIZE)
    return m1, m2


def test_stereo_record_round_trips_without_pose(world, tmp_path):
    m1, m2 = _stereo_pair(world)
    wf = WorldFrame(mode="clicks", origin_px=np.array([10.0, 10]), x_axis_px=np.array([20.0, 10]),
                    y_axis_px=np.array([10.0, 20]), swap_axes=False, col_sign=1, row_sign=1,
                    origin_grid=np.array([0.0, 0]), origin_mm=np.array([0.0, 0]))
    sr = rec.StereoRecord(
        cam1=1, cam2=2, board_type="stepped", model1=m1, model2=m2,
        R_stereo=None, T_stereo=None, world_frame=wf,
        per_view_rms1=[m1.rms_px], per_view_rms2=[m2.rms_px],
    )
    loaded = rec.load_stereo(rec.save_stereo(sr, tmp_path))
    assert isinstance(loaded.model1, Polynomial3DModel)
    assert loaded.R_stereo is None and loaded.T_stereo is None
    np.testing.assert_allclose(loaded.model2.project(world), m2.project(world), atol=1e-9)


def test_poly_camera_z_sign_uses_stored_convention(world):
    m1, m2 = _stereo_pair(world)
    assert camera_z_sign(m1, m2) == 1.0  # stepped clicked-level convention


def test_3c_reconstruction_recovers_known_displacement(world):
    m1, m2 = _stereo_pair(world)
    pts = world[::29]
    u_true = np.tile([0.7, -0.3, 0.15], (len(pts), 1))  # mm
    d1 = np.einsum("nij,nj->ni", m1.jacobian(pts), u_true)
    d2 = np.einsum("nij,nj->ni", m2.jacobian(pts), u_true)
    recovered = reconstruct_3c_at_points(m1, m2, pts, d1, d2, z_toward_cameras=True)
    np.testing.assert_allclose(recovered, u_true, atol=1e-9)
