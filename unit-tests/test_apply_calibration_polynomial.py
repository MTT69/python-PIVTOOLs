#!/usr/bin/env python3
"""
test_apply_calibration_polynomial.py

End-to-end integration test: synthetic Poiseuille PIV images →
instantaneous PIV → polynomial vector calibration via process_vectors() →
verify recovered physical velocity matches ground truth.

Mirrors test_apply_calibration_pinhole.py but exercises the polynomial
calibration path (PolynomialVectorCalibrator.process_vectors).

Usage:
    pytest unit-tests/test_apply_calibration_polynomial.py -v
    pytest unit-tests/test_apply_calibration_polynomial.py -v --make-figures
"""

import os
import shutil
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config, reload_config
from pivtools_core.vector_loading import read_mat_contents
from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
    PolynomialVectorCalibrator,
    fit_polynomial_from_points,
    save_polynomial_to_config,
)

# ---------------------------------------------------------------------------
# Paths to pre-generated synthetic data
# ---------------------------------------------------------------------------
UNIT_DIR = Path(__file__).resolve().parent
DATA_DIR = UNIT_DIR / "poiseuille_cam_model"
GT_PATH = DATA_DIR / "ground_truth.npz"
WORKSPACE_DIR = DATA_DIR / "test_workspace_polynomial"


# ---------------------------------------------------------------------------
# Ground-truth velocity function
# ---------------------------------------------------------------------------
def poiseuille_ux(y_mm, u_max, H_phys_m):
    """Analytical Poiseuille velocity at physical y-coordinate (mm).

    The channel spans [-H_phys/2, +H_phys/2] in metres, centred on the
    camera's optical axis.  y_mm is the calibrated coordinate in mm.

    Returns ux in m/s.
    """
    y_m = y_mm / 1000.0
    y_norm = y_m / (H_phys_m / 2.0)
    return u_max * (1.0 - y_norm ** 2)


# ---------------------------------------------------------------------------
# Module-scoped fixture: run PIV once, then polynomial calibration
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def polynomial_calibration_result():
    """
    Build workspace, run instantaneous PIV, fit polynomial from the
    camera model (using the known pinhole parameters to generate point
    correspondences), then run process_vectors() and return results.
    """
    if not GT_PATH.exists():
        pytest.skip(
            f"Synthetic data not found at {DATA_DIR}. "
            "Run: python unit-tests/generate_poiseuille_cam_model.py"
        )

    gt = dict(np.load(str(GT_PATH)))

    # Clean previous run
    if WORKSPACE_DIR.exists():
        _robust_rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(exist_ok=True)

    # ---- directory structure ----
    source_dir = WORKSPACE_DIR / "source"
    source_dir.mkdir(exist_ok=True)
    base_path = WORKSPACE_DIR / "output"
    base_path.mkdir(exist_ok=True)

    # Copy synthetic images
    for name in ("B00001_A.tif", "B00001_B.tif"):
        shutil.copy2(DATA_DIR / "Cam1" / name, source_dir / name)

    # ---- config.yaml ----
    config_dict = {
        "paths": {
            "source_paths": [str(source_dir)],
            "base_paths": [str(base_path)],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {
            "num_images": 1,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
            "vector_format": ["B%05d.mat"],
        },
        "processing": {
            "backend": "cpu",
            "omp_threads": 1,
            "dask_workers_per_node": 1,
            "dask_memory_limit": "2GB",
            "dask_max_in_flight_per_worker": 3,
        },
        "batches": {"batch_size": 1},
        "instantaneous_piv": {
            "window_size": [[64, 64], [32, 32]],
            "overlap": [50],
            "runs": [1],
            "peak_finder": "gauss6",
        },
        "outlier_detection": {
            "enabled": True,
            "methods": [
                {"type": "median_2d", "threshold": 3.0, "epsilon": 0.1}
            ],
        },
        "infilling": {
            "mid_pass": {"method": "local_median", "parameters": {"ksize": 3}},
            "final_pass": {"method": "local_median", "parameters": {"ksize": 3}},
        },
        "filters": [],
        "masking": {"enabled": False},
        "calibration": {
            "active": "polynomial",
            "polynomial": {"dt": 0.001},
        },
    }

    config_path = WORKSPACE_DIR / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    reload_config()
    config = Config(path=str(config_path))

    # ---- run instantaneous PIV ----
    from dask.distributed import Client
    from pivtools_core.instantaneous import run_instantaneous_piv
    from pivtools_cli.piv.save_results import get_output_path

    output_path = get_output_path(
        config, camera=1, use_uncalibrated=True, piv_type="instantaneous"
    )

    client = Client(processes=False, n_workers=1, threads_per_worker=1)
    try:
        run_instantaneous_piv(
            config=config,
            client=client,
            camera_num=1,
            source_path=source_dir,
            output_path=output_path,
            base_path=base_path,
        )
    finally:
        client.close()

    # ---- fit polynomial from known camera model ----
    # Use the known pinhole camera to generate point correspondences.
    # Create a grid of world points and project them through the camera.
    camera_matrix = gt["camera_matrix"]
    dist_coeffs = gt["dist_coeffs"]
    rvec = gt["rvec"]
    tvec = gt["tvec"]
    image_height = int(gt["image_height"])
    image_width = int(camera_matrix[0, 2] * 2)

    # Generate a grid of world points (mm) covering the calibration volume
    world_x = np.linspace(-50, 50, 25)
    world_y = np.linspace(-30, 30, 20)
    WX, WY = np.meshgrid(world_x, world_y)
    world_pts_mm = np.column_stack([WX.ravel(), WY.ravel()])
    world_pts_3d = np.column_stack([
        world_pts_mm[:, 0] / 1000.0,   # mm → m
        world_pts_mm[:, 1] / 1000.0,
        np.zeros(len(world_pts_mm)),
    ])

    # Project to image coordinates using the known camera model
    import cv2
    img_pts_2d, _ = cv2.projectPoints(
        world_pts_3d, rvec, tvec, camera_matrix, dist_coeffs,
    )
    img_pts = img_pts_2d.reshape(-1, 2)

    # Filter to points within image bounds
    in_bounds = (
        (img_pts[:, 0] >= 10) & (img_pts[:, 0] < image_width - 10) &
        (img_pts[:, 1] >= 10) & (img_pts[:, 1] < image_height - 10)
    )
    img_pts = img_pts[in_bounds]
    world_pts_mm = world_pts_mm[in_bounds]

    assert len(img_pts) >= 50, f"Only {len(img_pts)} in-bounds points"

    # Fit polynomial
    fit_result = fit_polynomial_from_points(
        img_pts, world_pts_mm, (image_width, image_height),
    )

    # Save to config
    save_polynomial_to_config(
        camera_num=1, fit_result=fit_result, dt=0.001, config=config,
    )

    # Reload config after save
    config = Config(path=str(config_path))

    # ---- run polynomial calibration via process_vectors() ----
    calibrator = PolynomialVectorCalibrator(
        base_dir=str(base_path),
        camera_num=1,
        type_name="instantaneous",
        config=config,
    )

    result = calibrator.process_vectors()

    # Get calibrated output path
    from pivtools_core.paths import get_data_paths
    calib_paths = get_data_paths(
        base_path,
        num_frame_pairs=config.num_frame_pairs,
        cam=1,
        type_name="instantaneous",
    )

    return {
        "result": result,
        "calib_output": calib_paths["data_dir"],
        "gt": gt,
        "workspace_dir": WORKSPACE_DIR,
        "config": config,
        "fit_result": fit_result,
    }


# ===================================================================
# Tests on process_vectors() result
# ===================================================================

class TestProcessVectorsResult:
    """Verify process_vectors() succeeded and produced output files."""

    def test_process_vectors_success(self, polynomial_calibration_result):
        result = polynomial_calibration_result["result"]
        assert result["success"], f"process_vectors failed: {result}"

    def test_calibrated_output_exists(self, polynomial_calibration_result):
        out = polynomial_calibration_result["calib_output"]
        assert (out / "B00001.mat").exists(), "Calibrated vector file not found"
        assert (out / "coordinates.mat").exists(), "Calibrated coordinates not found"

    def test_processed_frame_count(self, polynomial_calibration_result):
        result = polynomial_calibration_result["result"]
        assert result["processed_frames"] == 1
        assert result["successful_frames"] == 1


# ===================================================================
# Tests on calibrated velocity
# ===================================================================

class TestCalibratedVelocity:
    """Verify calibrated velocities match analytical Poiseuille profile."""

    def test_calibrated_ux_profile(self, polynomial_calibration_result):
        """Calibrated ux should match Poiseuille profile at each window."""
        ux, uy, x_mm, y_mm, valid = _load_calibrated_with_coords(
            polynomial_calibration_result
        )
        gt = polynomial_calibration_result["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])

        gt_ux = poiseuille_ux(y_mm, u_max, H_phys)

        # Exclude near-wall windows
        interior = valid & (np.abs(y_mm) < (H_phys * 1000 / 2.0) * 0.85)
        assert interior.sum() > 10, "Too few interior windows"

        err = np.abs(ux[interior] - gt_ux[interior])
        median_err = np.nanmedian(err)
        assert median_err < 0.05, (
            f"[polynomial] median |ux - gt| = {median_err:.4f} m/s, expected < 0.05"
        )

    def test_calibrated_uy_near_zero(self, polynomial_calibration_result):
        """Calibrated uy should be ~0 (Poiseuille has no vertical velocity)."""
        _, uy, _, y_mm, valid = _load_calibrated_with_coords(
            polynomial_calibration_result
        )
        gt = polynomial_calibration_result["gt"]
        H_phys = float(gt["H_phys"])

        interior = valid & (np.abs(y_mm) < (H_phys * 1000 / 2.0) * 0.85)
        median_uy = np.nanmedian(np.abs(uy[interior]))
        assert median_uy < 0.03, (
            f"[polynomial] median |uy| = {median_uy:.4f} m/s, expected < 0.03"
        )


# ===================================================================
# Tests on calibrated coordinates
# ===================================================================

class TestCalibratedCoordinates:
    """Verify coordinate convention and physical reasonableness."""

    def test_coordinates_convention(self, polynomial_calibration_result):
        """x increases left-to-right, y increases bottom-to-top (mm)."""
        out = polynomial_calibration_result["calib_output"]
        from pivtools_core.coordinate_utils import extract_coordinates

        coords_mat = loadmat(
            str(out / "coordinates.mat"),
            struct_as_record=False, squeeze_me=True,
        )
        x, y = extract_coordinates(coords_mat["coordinates"], run=1)

        if x.ndim == 2:
            dx = np.diff(x, axis=1)
            assert np.all(dx >= 0), "x-coordinates do not increase left-to-right"
            dy = np.diff(y, axis=0)
            assert np.all(dy <= 0), "y-coordinates do not increase bottom-to-top"
        else:
            assert x.max() > x.min(), "x range is zero"
            assert y.max() > y.min(), "y range is zero"

    def test_coordinate_range_reasonable(self, polynomial_calibration_result):
        """Physical coordinates should be within a reasonable range (< 500mm)."""
        out = polynomial_calibration_result["calib_output"]
        from pivtools_core.coordinate_utils import extract_coordinates

        coords_mat = loadmat(
            str(out / "coordinates.mat"),
            struct_as_record=False, squeeze_me=True,
        )
        x, y = extract_coordinates(coords_mat["coordinates"], run=1)

        assert abs(x.max()) < 500.0, f"x_max={x.max():.1f}mm unreasonably large"
        assert abs(y.max()) < 500.0, f"y_max={y.max():.1f}mm unreasonably large"
        assert abs(x.min()) < 500.0, f"x_min={x.min():.1f}mm unreasonably large"
        assert abs(y.min()) < 500.0, f"y_min={y.min():.1f}mm unreasonably large"


# ===================================================================
# Diagnostic figures
# ===================================================================

class TestDiagnostics:
    def test_make_figures(self, polynomial_calibration_result, make_figures):
        """Generate diagnostic plots when --make-figures is passed."""
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        gt = polynomial_calibration_result["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])

        ux, uy, x_mm, y_mm, valid = _load_calibrated_with_coords(
            polynomial_calibration_result
        )

        gt_ux = np.full_like(ux, np.nan)
        gt_ux[valid] = poiseuille_ux(y_mm[valid], u_max, H_phys)

        pct_ux = np.full_like(ux, np.nan)
        pct_ux[valid] = (ux[valid] - gt_ux[valid]) / u_max * 100.0

        fig_dir = polynomial_calibration_result["workspace_dir"] / "figures"
        fig_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Polynomial calibration verification (Poiseuille)", fontsize=14)

        _plot_field(axes[0, 0], gt_ux, valid, "Ground truth ux (m/s)", "viridis")
        _plot_field(axes[0, 1], ux, valid, "Calibrated ux (m/s)", "viridis")

        vlim = max(np.nanmax(np.abs(pct_ux[valid])), 0.5) if valid.any() else 1.0
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        _plot_field(axes[1, 0], pct_ux, valid, "ux error (% of u_max)", "RdBu_r", norm=norm)

        abs_uy = np.full_like(uy, np.nan)
        abs_uy[valid] = uy[valid]
        vlim_uy = max(np.nanmax(np.abs(abs_uy[valid])), 0.001) if valid.any() else 0.01
        norm_uy = TwoSlopeNorm(vmin=-vlim_uy, vcenter=0, vmax=vlim_uy)
        _plot_field(axes[1, 1], abs_uy, valid, "uy (m/s) — should be ~0", "RdBu_r", norm=norm_uy)

        fig.tight_layout()
        out_path = fig_dir / "polynomial_calibration_verification.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _robust_rmtree(path):
    """Remove directory tree, handling file locking."""
    def _onerror(func, fpath, _exc_info):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    for attempt in range(3):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError:
            time.sleep(0.5)
    shutil.rmtree(path, ignore_errors=True)


def _load_calibrated_with_coords(result):
    """Load calibrated vectors + physical coordinates (mm)."""
    out = result["calib_output"]
    from pivtools_core.coordinate_utils import extract_coordinates

    arr = read_mat_contents(str(out / "B00001.mat"))
    ux = arr[0, 0].astype(float)
    uy = arr[0, 1].astype(float)
    b_mask = arr[0, 2]

    valid = (b_mask == 0) & np.isfinite(ux) & np.isfinite(uy)

    coords_mat = loadmat(
        str(out / "coordinates.mat"),
        struct_as_record=False, squeeze_me=True,
    )
    x_mm, y_mm = extract_coordinates(coords_mat["coordinates"], run=1)

    if x_mm.shape != ux.shape:
        if x_mm.ndim == 1 and ux.ndim == 2:
            x_mm = np.broadcast_to(x_mm[np.newaxis, :], ux.shape).copy()
            y_mm = np.broadcast_to(y_mm[:, np.newaxis], ux.shape).copy()

    return ux, uy, x_mm, y_mm, valid


def _plot_field(ax, field, valid, title, cmap, norm=None):
    """Plot a 2-D field with invalid windows masked out."""
    import matplotlib.pyplot as _plt

    display = np.where(valid, field, np.nan)
    im = ax.imshow(display, cmap=cmap, norm=norm, origin="upper", aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("window col")
    ax.set_ylabel("window row")
    _plt.colorbar(im, ax=ax, shrink=0.85)
