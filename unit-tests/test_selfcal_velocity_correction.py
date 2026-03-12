"""
Test self-calibration velocity correction via tilted-plane back-projection.

Uses the same synthetic stereo camera pair as test_self_calibration_recovery.py.
Proves that _pixels_to_world_mm with (z_world, tilt_x, tilt_y) from self-cal
recovers true world velocities for EACH camera independently, while Z=0
projection introduces systematic magnification error.

This is the complete self-cal velocity correction: each camera's pixel
displacements are calibrated to world velocities on the true laser sheet
plane. Stereo reconstruction then consumes these corrected per-camera fields.

No real data, no C libraries, no config.yaml.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.calibration.vector_calibration_production import (
    VectorCalibrator,
    _pixels_to_world_mm,
)
from pivtools_gui.stereo_reconstruction.self_calibration import PinholeCamera

# ---------------------------------------------------------------------------
# Constants — match test_self_calibration_recovery.py
# ---------------------------------------------------------------------------

TRUE_Z_OFFSET = 0.3       # mm
TRUE_TILT_X = 0.002       # rad
TRUE_TILT_Y = -0.001      # rad
STEREO_ANGLE_DEG = 30.0
FOCAL_LENGTH_PX = 1000.0
IMAGE_SIZE = (1024, 1024)  # (width, height)
BASELINE_MM = 200.0

# Known velocity field (mm/frame)
UX_TRUE = 2.0
UY_TRUE = 1.0

# PIV grid: 20x20 points spanning ±30mm on the tilted sheet
GRID_RANGE = 30.0
N_GRID = 20


# ---------------------------------------------------------------------------
# Synthetic stereo camera pair (copied from test_self_calibration_recovery.py)
# ---------------------------------------------------------------------------

def create_stereo_cameras():
    """Create the same symmetric stereo camera pair used by self-cal tests."""
    w, h = IMAGE_SIZE
    theta = math.radians(STEREO_ANGLE_DEG / 2.0)

    K = np.array([
        [FOCAL_LENGTH_PX, 0.0, w / 2.0],
        [0.0, FOCAL_LENGTH_PX, h / 2.0],
        [0.0, 0.0, 1.0],
    ])
    dist = np.zeros(5)

    z_cam = BASELINE_MM / (2.0 * math.tan(theta))
    cam1_pos = np.array([BASELINE_MM / 2.0, 0.0, z_cam])
    cam2_pos = np.array([-BASELINE_MM / 2.0, 0.0, z_cam])

    def look_at_rotation(cam_pos, target=np.array([0, 0, 0])):
        z_axis = target - cam_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        up = np.array([0.0, -1.0, 0.0])
        x_axis = np.cross(z_axis, up)
        norm = np.linalg.norm(x_axis)
        if norm < 1e-10:
            up = np.array([0.0, 0.0, -1.0])
            x_axis = np.cross(z_axis, up)
            norm = np.linalg.norm(x_axis)
        x_axis = x_axis / norm
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        return R

    R1 = look_at_rotation(cam1_pos)
    R2 = look_at_rotation(cam2_pos)
    t1 = (-R1 @ cam1_pos).reshape(3, 1)
    t2 = (-R2 @ cam2_pos).reshape(3, 1)

    cam1 = PinholeCamera(K=K.copy(), dist=dist.copy(), R=R1, t=t1, image_size=IMAGE_SIZE)
    cam2 = PinholeCamera(K=K.copy(), dist=dist.copy(), R=R2, t=t2, image_size=IMAGE_SIZE)
    return cam1, cam2


def _camera_to_opencv(cam):
    """Extract (K, dist, rvec, tvec) from PinholeCamera for _pixels_to_world_mm."""
    rvec, _ = cv2.Rodrigues(cam.R)
    return cam.K, cam.dist, rvec.flatten(), cam.t.flatten()


# ---------------------------------------------------------------------------
# Grid and projection helpers
# ---------------------------------------------------------------------------

def _make_grid_on_tilted_sheet():
    """Generate world points on Z = z0 + x*tan(ty) + y*tan(tx)."""
    xs = np.linspace(-GRID_RANGE, GRID_RANGE, N_GRID)
    ys = np.linspace(-GRID_RANGE, GRID_RANGE, N_GRID)
    xx, yy = np.meshgrid(xs, ys)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    zz_flat = TRUE_Z_OFFSET + xx_flat * np.tan(TRUE_TILT_Y) + yy_flat * np.tan(TRUE_TILT_X)
    return xx_flat, yy_flat, zz_flat


def _project_to_pixels(world_xyz, cam):
    """Project (N,3) world points to (N,2) pixel coordinates via PinholeCamera."""
    return cam.project(world_xyz)


def _compute_velocities_for_camera(cam):
    """
    For one camera: generate grid, project, apply known displacement,
    back-project uncorrected (Z=0) and corrected (tilted plane).

    Returns dict with vel_uncorrected, vel_corrected, vel_true, coords, px.
    """
    xx, yy, zz = _make_grid_on_tilted_sheet()
    n = len(xx)
    K, dist, rvec, tvec = _camera_to_opencv(cam)

    # Original world positions on tilted sheet
    world_orig = np.column_stack([xx, yy, zz])

    # Displaced world positions (apply known in-plane velocity)
    world_disp = np.column_stack([
        xx + UX_TRUE,
        yy + UY_TRUE,
        TRUE_Z_OFFSET + (xx + UX_TRUE) * np.tan(TRUE_TILT_Y) + (yy + UY_TRUE) * np.tan(TRUE_TILT_X),
    ])

    # Project to pixel space
    px_orig = _project_to_pixels(world_orig, cam)
    px_disp = _project_to_pixels(world_disp, cam)

    # --- Uncorrected: back-project to Z=0 ---
    coords_z0 = _pixels_to_world_mm(px_orig, K, dist, rvec, tvec)
    coords_z0_disp = _pixels_to_world_mm(px_disp, K, dist, rvec, tvec)
    vel_uncorrected = coords_z0_disp - coords_z0

    # --- Corrected: back-project to tilted plane ---
    coords_corr = _pixels_to_world_mm(
        px_orig, K, dist, rvec, tvec,
        z_world=TRUE_Z_OFFSET, tilt_x=TRUE_TILT_X, tilt_y=TRUE_TILT_Y,
    )
    coords_corr_disp = _pixels_to_world_mm(
        px_disp, K, dist, rvec, tvec,
        z_world=TRUE_Z_OFFSET, tilt_x=TRUE_TILT_X, tilt_y=TRUE_TILT_Y,
    )
    vel_corrected = coords_corr_disp - coords_corr

    vel_true = np.column_stack([np.full(n, UX_TRUE), np.full(n, UY_TRUE)])
    true_world_xy = np.column_stack([xx, yy])

    return {
        "vel_uncorrected": vel_uncorrected,
        "vel_corrected": vel_corrected,
        "vel_true": vel_true,
        "coords_corr": coords_corr,
        "true_world_xy": true_world_xy,
        "px_orig": px_orig,
        "K": K, "dist": dist, "rvec": rvec, "tvec": tvec,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stereo_cameras():
    return create_stereo_cameras()


@pytest.fixture(scope="module")
def cam1_data(stereo_cameras):
    return _compute_velocities_for_camera(stereo_cameras[0])


@pytest.fixture(scope="module")
def cam2_data(stereo_cameras):
    return _compute_velocities_for_camera(stereo_cameras[1])


# ---------------------------------------------------------------------------
# Tests — Camera 1
# ---------------------------------------------------------------------------

class TestCamera1:
    def test_uncorrected_has_systematic_error(self, cam1_data):
        """Z=0 projection introduces measurable velocity error."""
        rel_err = np.abs(cam1_data["vel_uncorrected"] - cam1_data["vel_true"]) / np.abs(cam1_data["vel_true"])
        assert np.mean(rel_err) > 0.0005, f"Expected >0.05% error, got {np.mean(rel_err)*100:.4f}%"

    def test_corrected_eliminates_error(self, cam1_data):
        """Tilted-plane projection recovers true velocity."""
        rel_err = np.abs(cam1_data["vel_corrected"] - cam1_data["vel_true"]) / np.abs(cam1_data["vel_true"])
        mean_err = np.mean(rel_err)
        # Residual ~1e-5 from OpenCV float32 internals + displaced points
        # sampling slightly different Z on the tilted sheet
        assert mean_err < 1e-4, f"Expected <0.01% error, got {mean_err*100:.6f}%"

    def test_coordinates_on_correct_plane(self, cam1_data):
        """Corrected coordinates recover true world XY positions."""
        err = np.abs(cam1_data["coords_corr"] - cam1_data["true_world_xy"])
        assert np.max(err) < 0.01, f"Expected <10 µm position error, got {np.max(err)*1000:.1f} µm"


# ---------------------------------------------------------------------------
# Tests — Camera 2
# ---------------------------------------------------------------------------

class TestCamera2:
    def test_uncorrected_has_systematic_error(self, cam2_data):
        """Z=0 projection introduces measurable velocity error."""
        rel_err = np.abs(cam2_data["vel_uncorrected"] - cam2_data["vel_true"]) / np.abs(cam2_data["vel_true"])
        assert np.mean(rel_err) > 0.0005, f"Expected >0.05% error, got {np.mean(rel_err)*100:.4f}%"

    def test_corrected_eliminates_error(self, cam2_data):
        """Tilted-plane projection recovers true velocity."""
        rel_err = np.abs(cam2_data["vel_corrected"] - cam2_data["vel_true"]) / np.abs(cam2_data["vel_true"])
        mean_err = np.mean(rel_err)
        assert mean_err < 1e-4, f"Expected <0.01% error, got {mean_err*100:.6f}%"

    def test_coordinates_on_correct_plane(self, cam2_data):
        """Corrected coordinates recover true world XY positions."""
        err = np.abs(cam2_data["coords_corr"] - cam2_data["true_world_xy"])
        assert np.max(err) < 0.01, f"Expected <10 µm position error, got {np.max(err)*1000:.1f} µm"


# ---------------------------------------------------------------------------
# Tests — Cross-camera consistency (the stereo alignment check)
# ---------------------------------------------------------------------------

class TestStereoPairConsistency:
    """
    Self-cal's main stereo benefit: both cameras agree on world positions.
    With Z=0 projection, each camera maps the same physical point to a
    different world coordinate. With the corrected tilted plane, they agree.
    """

    def test_uncorrected_cameras_disagree(self, cam1_data, cam2_data):
        """Without correction, cameras report different world positions for the same grid."""
        diff = np.abs(cam1_data["coords_corr"] - cam2_data["coords_corr"])
        # With correction both should agree, but on Z=0 they shouldn't
        coords1_z0 = _pixels_to_world_mm(
            cam1_data["px_orig"], cam1_data["K"], cam1_data["dist"],
            cam1_data["rvec"], cam1_data["tvec"],
        )
        coords2_z0 = _pixels_to_world_mm(
            cam2_data["px_orig"], cam2_data["K"], cam2_data["dist"],
            cam2_data["rvec"], cam2_data["tvec"],
        )
        misreg = np.abs(coords1_z0 - coords2_z0)
        # Different viewing angles → different Z=0 projections
        assert np.max(misreg) > 0.01, (
            f"Expected >10 µm misregistration, got {np.max(misreg)*1000:.1f} µm"
        )

    def test_corrected_cameras_agree(self, cam1_data, cam2_data):
        """With correction, both cameras recover the same world XY positions."""
        diff = np.abs(cam1_data["coords_corr"] - cam2_data["coords_corr"])
        assert np.max(diff) < 0.01, (
            f"Expected <10 µm agreement, got {np.max(diff)*1000:.1f} µm"
        )

    def test_corrected_velocities_agree(self, cam1_data, cam2_data):
        """With correction, both cameras recover the same world velocities."""
        diff = np.abs(cam1_data["vel_corrected"] - cam2_data["vel_corrected"])
        assert np.max(diff) < 0.001, (
            f"Expected <1 µm/frame velocity agreement, got {np.max(diff)*1000:.1f} µm/frame"
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_default_args_match_old_behavior(self, cam1_data):
        """With z=0, tilt=0, corrected path == uncorrected path."""
        pts_px = cam1_data["px_orig"]
        K, dist, rvec, tvec = cam1_data["K"], cam1_data["dist"], cam1_data["rvec"], cam1_data["tvec"]
        coords_default = _pixels_to_world_mm(pts_px, K, dist, rvec, tvec)
        coords_explicit = _pixels_to_world_mm(pts_px, K, dist, rvec, tvec,
                                               z_world=0.0, tilt_x=0.0, tilt_y=0.0)
        np.testing.assert_array_equal(coords_default, coords_explicit)

    def test_tilt_creates_spatially_varying_error(self, cam1_data):
        """Tilted sheet causes different magnification errors across the field."""
        rel_err = np.abs(cam1_data["vel_uncorrected"] - cam1_data["vel_true"]) / np.abs(cam1_data["vel_true"])
        assert np.std(rel_err[:, 0]) > 0.0001, (
            f"Expected spatially varying error, got std={np.std(rel_err[:, 0]):.6f}"
        )


# ---------------------------------------------------------------------------
# VectorCalibrator integration — proves self-cal flows through production code
# ---------------------------------------------------------------------------

class TestVectorCalibratorIntegration:
    """
    Exercise VectorCalibrator.calibrate_vectors() and calibrate_coordinates()
    with self-cal params, proving the production code path uses them.
    """

    def test_calibrator_velocity_correction(self, stereo_cameras):
        """VectorCalibrator with self-cal params produces correct velocities."""
        cam = stereo_cameras[0]
        K, dist, rvec, tvec = _camera_to_opencv(cam)

        # Build a minimal VectorCalibrator without loading a model file
        calibrator = VectorCalibrator.__new__(VectorCalibrator)
        calibrator.camera_matrix = K
        calibrator.dist_coeffs = dist
        calibrator.rvec = rvec
        calibrator.tvec = tvec
        calibrator.dt = 1.0  # 1 frame
        calibrator.image_height = IMAGE_SIZE[1]
        calibrator.z_world = TRUE_Z_OFFSET
        calibrator.tilt_x = TRUE_TILT_X
        calibrator.tilt_y = TRUE_TILT_Y

        # Generate grid on tilted sheet
        xx, yy, zz = _make_grid_on_tilted_sheet()
        world_orig = np.column_stack([xx, yy, zz])
        world_disp = np.column_stack([
            xx + UX_TRUE, yy + UY_TRUE,
            TRUE_Z_OFFSET + (xx + UX_TRUE) * np.tan(TRUE_TILT_Y) + (yy + UY_TRUE) * np.tan(TRUE_TILT_X),
        ])

        # Project to pixels
        px_orig = cam.project(world_orig)
        px_disp = cam.project(world_disp)

        # PIV displacement in pixels
        piv_disp = px_disp - px_orig

        # Convert pixel coords to "uncalibrated" convention (1-based, y-up)
        coords_x_uncal = px_orig[:, 0].reshape(N_GRID, N_GRID) + 1
        coords_y_uncal = IMAGE_SIZE[1] - px_orig[:, 1].reshape(N_GRID, N_GRID)
        ux_px = piv_disp[:, 0].reshape(N_GRID, N_GRID)
        # uy in uncal convention: positive = upward in image = negative in raw pixels
        uy_px = -piv_disp[:, 1].reshape(N_GRID, N_GRID)

        # Calibrate through VectorCalibrator (production code path)
        ux_ms, uy_ms = calibrator.calibrate_vectors(
            ux_px, uy_px, coords_x_uncal, coords_y_uncal,
        )

        # dt=1, so m/s = mm/1000. True velocity: ux=2mm, uy=1mm → 0.002, 0.001 m/s
        # calibrate_vectors() negates uy (OpenCV y-down → physical y-up), so
        # the output uy_ms has the opposite sign from the world Y displacement.
        ux_true_ms = UX_TRUE / 1000.0
        uy_true_ms = -UY_TRUE / 1000.0  # negated by calibrate_vectors

        rel_err_ux = np.abs(ux_ms - ux_true_ms) / np.abs(ux_true_ms)
        rel_err_uy = np.abs(uy_ms - uy_true_ms) / np.abs(uy_true_ms)

        assert np.mean(rel_err_ux) < 1e-4, (
            f"ux error {np.mean(rel_err_ux)*100:.4f}% > 0.01%"
        )
        assert np.mean(rel_err_uy) < 1e-4, (
            f"uy error {np.mean(rel_err_uy)*100:.4f}% > 0.01%"
        )

    def test_calibrator_without_selfcal_has_error(self, stereo_cameras):
        """VectorCalibrator without self-cal (z=0) has systematic velocity error."""
        cam = stereo_cameras[0]
        K, dist, rvec, tvec = _camera_to_opencv(cam)

        calibrator = VectorCalibrator.__new__(VectorCalibrator)
        calibrator.camera_matrix = K
        calibrator.dist_coeffs = dist
        calibrator.rvec = rvec
        calibrator.tvec = tvec
        calibrator.dt = 1.0
        calibrator.image_height = IMAGE_SIZE[1]
        calibrator.z_world = 0.0  # No self-cal correction
        calibrator.tilt_x = 0.0
        calibrator.tilt_y = 0.0

        xx, yy, zz = _make_grid_on_tilted_sheet()
        world_orig = np.column_stack([xx, yy, zz])
        world_disp = np.column_stack([
            xx + UX_TRUE, yy + UY_TRUE,
            TRUE_Z_OFFSET + (xx + UX_TRUE) * np.tan(TRUE_TILT_Y) + (yy + UY_TRUE) * np.tan(TRUE_TILT_X),
        ])

        px_orig = cam.project(world_orig)
        px_disp = cam.project(world_disp)
        piv_disp = px_disp - px_orig

        coords_x_uncal = px_orig[:, 0].reshape(N_GRID, N_GRID) + 1
        coords_y_uncal = IMAGE_SIZE[1] - px_orig[:, 1].reshape(N_GRID, N_GRID)
        ux_px = piv_disp[:, 0].reshape(N_GRID, N_GRID)
        uy_px = -piv_disp[:, 1].reshape(N_GRID, N_GRID)

        ux_ms, uy_ms = calibrator.calibrate_vectors(
            ux_px, uy_px, coords_x_uncal, coords_y_uncal,
        )

        ux_true_ms = UX_TRUE / 1000.0
        rel_err = np.abs(ux_ms - ux_true_ms) / ux_true_ms
        assert np.mean(rel_err) > 0.0005, (
            f"Expected >0.05% error without self-cal, got {np.mean(rel_err)*100:.4f}%"
        )


# ---------------------------------------------------------------------------
# End-to-end: self-cal recovery → velocity correction
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    The definitive test: run self-calibration on synthetic stereo images to
    RECOVER z_offset/tilt_x/tilt_y (imperfect estimates), then feed those
    recovered values into velocity correction and verify the velocities
    are still accurate. This proves the full pipeline works.
    """

    @pytest.fixture(scope="class")
    def self_cal_result(self, stereo_cameras):
        """Run self-calibration to recover sheet params from synthetic images."""
        from pivtools_gui.stereo_reconstruction.self_calibration import run_self_calibration

        np.random.seed(42)
        cam1, cam2 = stereo_cameras

        # Generate synthetic particle images on the tilted sheet
        from test_self_calibration_recovery import (
            generate_particles,
            render_particles,
            N_PARTICLES,
            WORLD_BOUNDS,
            WINDOW_SIZE,
            OVERLAP,
            CONVERGENCE_THRESHOLD,
        )

        images_cam1, images_cam2 = [], []
        for _ in range(20):
            particles = generate_particles(
                N_PARTICLES,
                x_range=(WORLD_BOUNDS[0], WORLD_BOUNDS[1]),
                y_range=(WORLD_BOUNDS[2], WORLD_BOUNDS[3]),
                z_offset=TRUE_Z_OFFSET,
                tilt_x=TRUE_TILT_X,
                tilt_y=TRUE_TILT_Y,
            )
            images_cam1.append(render_particles(particles, cam1))
            images_cam2.append(render_particles(particles, cam2))

        return run_self_calibration(
            cam1, cam2, images_cam1, images_cam2,
            world_bounds=WORLD_BOUNDS,
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            max_iterations=10,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            quality_threshold=0.1,
        )

    def test_selfcal_converged(self, self_cal_result):
        """Prerequisite: self-cal must converge for the rest to be meaningful."""
        assert self_cal_result.converged

    def test_recovered_params_used_for_velocity(self, stereo_cameras, self_cal_result):
        """
        Feed RECOVERED (not ground-truth) self-cal params into velocity
        correction and verify velocities are accurate for both cameras.
        """
        recovered_z = self_cal_result.z_offset
        recovered_tx = self_cal_result.tilt_x
        recovered_ty = self_cal_result.tilt_y

        for cam_idx, cam in enumerate(stereo_cameras):
            K, dist, rvec, tvec = _camera_to_opencv(cam)

            xx, yy, zz = _make_grid_on_tilted_sheet()
            world_orig = np.column_stack([xx, yy, zz])
            world_disp = np.column_stack([
                xx + UX_TRUE, yy + UY_TRUE,
                TRUE_Z_OFFSET + (xx + UX_TRUE) * np.tan(TRUE_TILT_Y) + (yy + UY_TRUE) * np.tan(TRUE_TILT_X),
            ])

            px_orig = cam.project(world_orig)
            px_disp = cam.project(world_disp)

            # Use RECOVERED self-cal params (not ground truth)
            coords_corr = _pixels_to_world_mm(
                px_orig, K, dist, rvec, tvec,
                z_world=recovered_z, tilt_x=recovered_tx, tilt_y=recovered_ty,
            )
            coords_disp = _pixels_to_world_mm(
                px_disp, K, dist, rvec, tvec,
                z_world=recovered_z, tilt_x=recovered_tx, tilt_y=recovered_ty,
            )
            vel = coords_disp - coords_corr
            vel_true = np.column_stack([np.full(len(xx), UX_TRUE), np.full(len(xx), UY_TRUE)])

            rel_err = np.abs(vel - vel_true) / np.abs(vel_true)
            mean_err = np.mean(rel_err)

            # Even with imperfect recovery, error should be <0.05%
            # (self-cal recovers z to ~0.05mm, tilts to ~0.001rad)
            assert mean_err < 5e-4, (
                f"Camera {cam_idx+1}: velocity error {mean_err*100:.4f}% > 0.05% "
                f"using recovered params (z={recovered_z:.4f}, "
                f"tx={recovered_tx:.6f}, ty={recovered_ty:.6f})"
            )

    def test_recovered_better_than_uncorrected(self, stereo_cameras, self_cal_result):
        """Recovered params give better velocities than no correction at all."""
        cam = stereo_cameras[0]
        K, dist, rvec, tvec = _camera_to_opencv(cam)

        xx, yy, zz = _make_grid_on_tilted_sheet()
        world_orig = np.column_stack([xx, yy, zz])
        world_disp = np.column_stack([
            xx + UX_TRUE, yy + UY_TRUE,
            TRUE_Z_OFFSET + (xx + UX_TRUE) * np.tan(TRUE_TILT_Y) + (yy + UY_TRUE) * np.tan(TRUE_TILT_X),
        ])

        px_orig = cam.project(world_orig)
        px_disp = cam.project(world_disp)
        vel_true = np.column_stack([np.full(len(xx), UX_TRUE), np.full(len(xx), UY_TRUE)])

        # Uncorrected (Z=0)
        c0 = _pixels_to_world_mm(px_orig, K, dist, rvec, tvec)
        d0 = _pixels_to_world_mm(px_disp, K, dist, rvec, tvec)
        err_uncorrected = np.mean(np.abs((d0 - c0) - vel_true) / np.abs(vel_true))

        # Corrected with recovered params
        cr = _pixels_to_world_mm(px_orig, K, dist, rvec, tvec,
                                  z_world=self_cal_result.z_offset,
                                  tilt_x=self_cal_result.tilt_x,
                                  tilt_y=self_cal_result.tilt_y)
        dr = _pixels_to_world_mm(px_disp, K, dist, rvec, tvec,
                                  z_world=self_cal_result.z_offset,
                                  tilt_x=self_cal_result.tilt_x,
                                  tilt_y=self_cal_result.tilt_y)
        err_corrected = np.mean(np.abs((dr - cr) - vel_true) / np.abs(vel_true))

        improvement = err_uncorrected / err_corrected
        assert improvement > 10, (
            f"Expected >10x improvement, got {improvement:.1f}x "
            f"(uncorrected: {err_uncorrected*100:.4f}%, corrected: {err_corrected*100:.6f}%)"
        )


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def test_diagnostic_figure(cam1_data, cam2_data, make_figures, output_dir):
    """Velocity error maps for both cameras: uncorrected vs corrected."""
    if not make_figures:
        pytest.skip("--make-figures not set")

    import matplotlib.pyplot as plt

    fig_dir = output_dir / "selfcal_velocity_correction"
    fig_dir.mkdir(exist_ok=True)

    true_xy = cam1_data["true_world_xy"]
    xx = true_xy[:, 0].reshape(N_GRID, N_GRID)
    yy = true_xy[:, 1].reshape(N_GRID, N_GRID)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (cam_data, cam_label) in enumerate([
        (cam1_data, "Camera 1 (right)"),
        (cam2_data, "Camera 2 (left)"),
    ]):
        for col, (vel_key, title) in enumerate([
            ("vel_uncorrected", "Uncorrected (Z=0)"),
            ("vel_corrected", "Corrected (tilted plane)"),
        ]):
            ax = axes[row, col]
            vel = cam_data[vel_key]
            err_pct = np.abs(vel - cam_data["vel_true"]) / np.abs(cam_data["vel_true"]) * 100
            ux_err = err_pct[:, 0].reshape(N_GRID, N_GRID)
            im = ax.pcolormesh(xx, yy, ux_err, shading="auto", cmap="RdYlGn_r")
            ax.set_title(f"{cam_label}\n{title}")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label="ux error (%)")

    fig.suptitle(
        f"Self-cal velocity correction (stereo pair)\n"
        f"z_offset={TRUE_Z_OFFSET}mm, tilt_x={TRUE_TILT_X:.4f}rad, "
        f"tilt_y={TRUE_TILT_Y:.4f}rad, stereo_angle={STEREO_ANGLE_DEG}°",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "stereo_velocity_error_comparison.png", dpi=150)
    plt.close(fig)
