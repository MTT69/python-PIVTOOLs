#!/usr/bin/env python3
"""
test_apply_calibration_pinhole.py

End-to-end integration test: synthetic PIV images → instantaneous PIV →
pinhole vector calibration → verify recovered physical velocity matches
ground truth.

Uses images from poiseuille_cam_model/ with known Poiseuille velocity:
  ux(y) = u_max * (1 - (2*y/H_phys - 1)^2),  uy = 0
projected through a pinhole camera with barrel distortion (k1=-0.15).

Parametrized by calibration model type (dotboard, charuco) to verify both
code paths produce identical results from the same camera model.

Usage:
    pytest unit-tests/test_apply_calibration_pinhole.py -v
    pytest unit-tests/test_apply_calibration_pinhole.py -v --make-figures
"""

import os
import shutil
import stat
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from scipy.io import loadmat, savemat

# Ensure production code is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config, reload_config
from pivtools_core.instantaneous import run_instantaneous_piv
from pivtools_core.vector_loading import read_mat_contents
from pivtools_cli.piv.save_results import get_output_path
from pivtools_gui.calibration.vector_calibration_production import VectorCalibrator

# ---------------------------------------------------------------------------
# Paths to pre-generated synthetic data
# ---------------------------------------------------------------------------
UNIT_DIR = Path(__file__).resolve().parent
DATA_DIR = UNIT_DIR / "poiseuille_cam_model"
GT_PATH = DATA_DIR / "ground_truth.npz"
WORKSPACE_DIR = DATA_DIR / "test_workspace"


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
# Module-scoped fixture: run PIV once, shared across all tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def piv_workspace():
    """
    Build a workspace inside the source data directory, write config,
    run instantaneous PIV, and return everything downstream fixtures need.

    Output lives at poiseuille_cam_model/test_workspace/ so it can be
    inspected after the run.  Cleaned and rebuilt every session.
    """
    if not GT_PATH.exists():
        pytest.skip(
            f"Synthetic data not found at {DATA_DIR}. "
            "Run: python unit-tests/generate_poiseuille_cam_model.py"
        )

    gt = dict(np.load(str(GT_PATH)))

    # Clean previous run and rebuild
    if WORKSPACE_DIR.exists():
        _robust_rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(exist_ok=True)

    # ---- directory structure ----
    source_dir = WORKSPACE_DIR / "source"
    source_dir.mkdir(exist_ok=True)

    base_path = WORKSPACE_DIR / "output"
    base_path.mkdir(exist_ok=True)

    # Copy synthetic images into the source tree
    for name in ("B00001_A.tif", "B00001_B.tif"):
        shutil.copy2(DATA_DIR / "Cam1" / name, source_dir / name)

    # ---- calibration model files ----
    # tvec stored in metres by the generator; calibration expects mm.
    model_fields = {
        "camera_matrix": gt["camera_matrix"],
        "dist_coeffs": gt["dist_coeffs"],
        "rvecs": gt["rvec"].reshape(1, 3),
        "tvecs": gt["tvec"].reshape(1, 3) * 1000.0,  # m → mm
        "image_height": int(gt["image_height"]),
        "dot_spacing_mm": 15.0,
    }

    for subdir, fname in [
        ("dotboard_planar/model", "dotboard_model.mat"),
        ("charuco_planar/model", "camera_model.mat"),
    ]:
        model_dir = base_path / "calibration" / "Cam1" / subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        savemat(str(model_dir / fname), model_fields)

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
            "active": "dotboard",
            "dotboard": {"dt": 0.001},
            "charuco": {"dt": 0.001},
        },
    }

    config_path = WORKSPACE_DIR / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # ---- load config ----
    reload_config()
    config = Config(path=str(config_path))

    # ---- run instantaneous PIV ----
    from dask.distributed import Client

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

    return {
        "workspace_dir": WORKSPACE_DIR,
        "source_dir": source_dir,
        "base_path": base_path,
        "config_path": config_path,
        "config": config,
        "gt": gt,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Parametrized fixture: calibrate for each model type
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", params=["dotboard", "charuco"])
def calibration_result(request, piv_workspace):
    """Run vector calibration for a given model type."""
    model_type = request.param
    ws = piv_workspace
    config = ws["config"]
    base_path = ws["base_path"]

    # Clear any previous calibrated output to avoid stale data
    calib_out = get_output_path(
        config, camera=1, use_uncalibrated=False, piv_type="instantaneous"
    )
    if calib_out.exists():
        _robust_rmtree(calib_out)

    calibrator = VectorCalibrator(
        base_dir=str(base_path),
        camera_num=1,
        model_type=model_type,
        dt=0.001,
        type_name="instantaneous",
        config=config,
    )
    calibrator.process_run(num_frame_pairs=1)

    return {
        "model_type": model_type,
        "calib_output": calib_out,
        "gt": ws["gt"],
        "workspace_dir": ws["workspace_dir"],
    }


# ===================================================================
# Tests on uncalibrated (PIV) output — NOT parametrized by model type
# ===================================================================

class TestUncalibratedOutput:
    """Verify PIV ran correctly before calibration."""

    def test_uncalibrated_output_exists(self, piv_workspace):
        out = piv_workspace["output_path"]
        assert (out / "B00001.mat").exists(), "PIV vector file not found"
        assert (out / "coordinates.mat").exists(), "Coordinates file not found"

    def test_uncalibrated_displacement_range(self, piv_workspace):
        """Pixel displacements should span ~0 (walls) to ~12.5 (centre)."""
        out = piv_workspace["output_path"]
        arr = read_mat_contents(str(out / "B00001.mat"))
        ux = arr[0, 0]
        b_mask = arr[0, 2]

        valid = (b_mask == 0) & np.isfinite(ux)
        assert valid.sum() > 10, "Too few valid vectors"

        max_ux = np.nanmax(np.abs(ux[valid]))
        assert max_ux > 8.0, f"max |ux_px|={max_ux:.2f}, expected > 8 (Poiseuille peak)"
        assert max_ux < 16.0, f"max |ux_px|={max_ux:.2f}, expected < 16"


# ===================================================================
# Tests on calibrated output — parametrized by model type
# ===================================================================

class TestCalibratedOutput:
    """Verify calibrated velocities match analytical Poiseuille profile."""

    def test_calibrated_output_exists(self, calibration_result):
        out = calibration_result["calib_output"]
        assert (out / "B00001.mat").exists(), "Calibrated vector file not found"
        assert (out / "coordinates.mat").exists(), "Calibrated coordinates not found"

    def test_calibrated_ux_profile(self, calibration_result):
        """Calibrated ux should match Poiseuille profile at each window."""
        ux, uy, x_mm, y_mm, valid = _load_calibrated_with_coords(calibration_result)
        gt = calibration_result["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])

        gt_ux = poiseuille_ux(y_mm, u_max, H_phys)

        # Exclude near-wall windows where particle loss degrades accuracy
        interior = valid & (np.abs(y_mm) < (H_phys * 1000 / 2.0) * 0.85)
        assert interior.sum() > 10, "Too few interior windows"

        err = np.abs(ux[interior] - gt_ux[interior])
        median_err = np.nanmedian(err)
        assert median_err < 0.03, (
            f"[{calibration_result['model_type']}] "
            f"median |ux - gt| = {median_err:.4f} m/s, expected < 0.03"
        )

    def test_calibrated_uy_near_zero(self, calibration_result):
        """Calibrated uy should be ~0 (Poiseuille has no vertical velocity)."""
        _, uy, _, y_mm, valid = _load_calibrated_with_coords(calibration_result)
        gt = calibration_result["gt"]
        H_phys = float(gt["H_phys"])

        interior = valid & (np.abs(y_mm) < (H_phys * 1000 / 2.0) * 0.85)
        median_uy = np.nanmedian(np.abs(uy[interior]))
        assert median_uy < 0.02, (
            f"[{calibration_result['model_type']}] "
            f"median |uy| = {median_uy:.4f} m/s, expected < 0.02"
        )

    def test_calibrated_coordinates_convention(self, calibration_result):
        """x increases left-to-right, y increases bottom-to-top (mm)."""
        out = calibration_result["calib_output"]
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

    def test_distortion_correction(self, calibration_result):
        """With distortion, calibrated coords should differ from naive scaling.

        A naive (no distortion) calibration would give different physical
        positions for off-centre windows.  We verify that the calibrated
        coordinate grid is NOT perfectly rectangular — the distortion
        correction should produce slight curvature.
        """
        _, _, x_mm, y_mm, valid = _load_calibrated_with_coords(calibration_result)
        if x_mm.ndim < 2:
            pytest.skip("Need 2D coordinate grid for distortion check")

        # In a perfect rectangle, each column has identical x values.
        # With barrel distortion correction, outer columns should show
        # measurable row-to-row variation (pincushion effect in world coords).
        col_mid = x_mm.shape[1] // 2
        col_edge = x_mm.shape[1] - 2  # near right edge

        x_var_centre = np.nanstd(x_mm[:, col_mid])
        x_var_edge = np.nanstd(x_mm[:, col_edge])

        # Edge column should have MORE x-variation than centre column
        assert x_var_edge > x_var_centre * 1.5, (
            f"Expected distortion effect: edge x-std ({x_var_edge:.4f}) "
            f"should exceed centre ({x_var_centre:.4f}) by >50%"
        )

    def test_make_figures(self, calibration_result, make_figures):
        """Generate diagnostic plots when --make-figures is passed."""
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        model_type = calibration_result["model_type"]
        gt = calibration_result["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])

        ux, uy, x_mm, y_mm, valid = _load_calibrated_with_coords(calibration_result)

        # Analytical ground truth at each window location
        gt_ux = np.full_like(ux, np.nan)
        gt_ux[valid] = poiseuille_ux(y_mm[valid], u_max, H_phys)
        gt_uy = np.zeros_like(uy)

        # Percentage error (relative to u_max for ux, absolute for uy)
        pct_ux = np.full_like(ux, np.nan)
        pct_ux[valid] = (ux[valid] - gt_ux[valid]) / u_max * 100.0

        abs_uy = np.full_like(uy, np.nan)
        abs_uy[valid] = uy[valid]

        fig_dir = calibration_result["workspace_dir"] / "figures"
        fig_dir.mkdir(exist_ok=True)

        # --- 3×2 figure ---
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle(
            f"Calibration verification — {model_type} (Poiseuille + distortion)",
            fontsize=14, fontweight="bold",
        )

        # Row 0: ground truth
        _plot_field(axes[0, 0], gt_ux, valid, f"Ground truth ux (m/s)", "viridis")
        _plot_field(axes[0, 1], gt_uy, valid, "Ground truth uy (m/s)", "viridis")

        # Row 1: calibrated (recovered)
        _plot_field(axes[1, 0], ux, valid, "Calibrated ux (m/s)", "viridis")
        _plot_field(axes[1, 1], uy, valid, "Calibrated uy (m/s)", "RdBu_r")

        # Row 2: error
        vlim_ux = max(np.nanmax(np.abs(pct_ux[valid])), 0.5) if valid.any() else 1.0
        norm_ux = TwoSlopeNorm(vmin=-vlim_ux, vcenter=0, vmax=vlim_ux)
        _plot_field(axes[2, 0], pct_ux, valid, "ux error (% of u_max)", "RdBu_r", norm=norm_ux)

        vlim_uy = max(np.nanmax(np.abs(abs_uy[valid])), 0.001) if valid.any() else 0.01
        norm_uy = TwoSlopeNorm(vmin=-vlim_uy, vcenter=0, vmax=vlim_uy)
        _plot_field(axes[2, 1], abs_uy, valid, "uy (m/s) — should be ~0", "RdBu_r", norm=norm_uy)

        # Summary text
        interior = valid & (np.abs(y_mm) < (H_phys * 1000 / 2.0) * 0.85)
        if interior.any():
            err_ux = np.abs(ux[interior] - gt_ux[interior])
            stats_text = (
                f"Interior windows ({interior.sum()}/{valid.sum()} valid):\n"
                f"  ux: median err = {np.nanmedian(err_ux):.4f} m/s, "
                f"max err = {np.nanmax(err_ux):.4f} m/s, "
                f"median % err = {np.nanmedian(err_ux/u_max*100):.2f}%\n"
                f"  uy: median |uy| = {np.nanmedian(np.abs(uy[interior])):.4f} m/s"
            )
        else:
            stats_text = "No interior windows"

        fig.text(
            0.5, 0.01, stats_text, ha="center", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        out_path = fig_dir / f"calibration_verification_{model_type}.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

        print(f"  Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _robust_rmtree(path):
    """Remove directory tree, handling Windows/OneDrive file locking."""
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


def _load_calibrated_vectors(calibration_result):
    """Load calibrated ux, uy, valid mask from the output .mat file."""
    out = calibration_result["calib_output"]
    arr = read_mat_contents(str(out / "B00001.mat"))
    ux = arr[0, 0].astype(float)
    uy = arr[0, 1].astype(float)
    b_mask = arr[0, 2]

    valid = (b_mask == 0) & np.isfinite(ux) & np.isfinite(uy)
    return ux, uy, valid


def _load_calibrated_with_coords(calibration_result):
    """Load calibrated vectors + physical coordinates (mm)."""
    out = calibration_result["calib_output"]
    from pivtools_core.coordinate_utils import extract_coordinates

    ux, uy, valid = _load_calibrated_vectors(calibration_result)

    coords_mat = loadmat(
        str(out / "coordinates.mat"),
        struct_as_record=False, squeeze_me=True,
    )
    x_mm, y_mm = extract_coordinates(coords_mat["coordinates"], run=1)

    # Ensure shapes match velocity grid
    if x_mm.shape != ux.shape:
        # Coordinates may be 1-D; broadcast to 2-D
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
