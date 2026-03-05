#!/usr/bin/env python3
"""
test_multicam_merge_pinhole.py

End-to-end multi-camera integration test: synthetic PIV images (2 cameras) →
instantaneous PIV → pinhole vector calibration → global coordinate alignment →
vector merging → verify recovered physical velocity matches ground truth.

Uses images from poiseuille_multicam/ with known Poiseuille velocity:
  ux(y) = u_max * (1 - y_norm^2),  uy = 0
projected through a pinhole camera with barrel distortion (k1=-0.15).

Two cameras see overlapping portions of the physical domain:
  Cam1: top portion (positive y_phys)
  Cam2: bottom portion, offset 20px up + 10px left

Global coordinate alignment applies a non-trivial shift: datum_physical is
set to [5.0, 3.0] mm, so the physical origin (world [0,0,0]) maps to
(5, 3) mm in the aligned coordinate system. This exercises the alignment
shift computation and application — not just a no-op identity.

Usage:
    pytest unit-tests/test_multicam_merge_pinhole.py -v
    pytest unit-tests/test_multicam_merge_pinhole.py -v --make-figures
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

# Ensure production code is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_core.config import Config, reload_config
from pivtools_core.instantaneous import run_instantaneous_piv
from pivtools_core.vector_loading import read_mat_contents
from pivtools_cli.piv.save_results import get_output_path

# ---------------------------------------------------------------------------
# Paths to pre-generated synthetic data
# ---------------------------------------------------------------------------
UNIT_DIR = Path(__file__).resolve().parent
DATA_DIR = UNIT_DIR / "poiseuille_multicam"
GT_PATH = DATA_DIR / "ground_truth.npz"
WORKSPACE_DIR = DATA_DIR / "test_workspace"

# The datum_physical offset applied during alignment.  The physical origin
# (world [0,0,0]) is mapped to these coordinates in the aligned system.
# Deliberately non-zero so alignment does real work.
DATUM_PHYSICAL = [5.0, 3.0]


# ---------------------------------------------------------------------------
# Ground-truth velocity function
# ---------------------------------------------------------------------------
def poiseuille_ux(y_mm, u_max, H_phys_m, y_offset_mm=0.0):
    """Analytical Poiseuille velocity at physical y-coordinate (mm).

    Parameters
    ----------
    y_mm : array-like
        Calibrated y-coordinate in mm (in aligned coordinate system).
    u_max : float
        Peak centreline velocity (m/s).
    H_phys_m : float
        Channel height in metres.
    y_offset_mm : float
        y-coordinate of the channel centre in aligned coordinates.
        The Poiseuille profile is centred on y = y_offset_mm.
    """
    y_m = (y_mm - y_offset_mm) / 1000.0
    y_norm = y_m / (H_phys_m / 2.0)
    return u_max * (1.0 - y_norm ** 2)


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


def _make_model(gt, cam_rvec, cam_tvec):
    """Build a calibration model dict for savemat."""
    return {
        "camera_matrix": gt["camera_matrix"],
        "dist_coeffs": gt["dist_coeffs"],
        "rvecs": cam_rvec.reshape(1, 3),
        "tvecs": cam_tvec.reshape(1, 3) * 1000.0,  # m → mm
        "image_height": int(gt["image_height"]),
        "dot_spacing_mm": 15.0,
    }


def _load_calibrated_with_coords(calib_output):
    """Load calibrated vectors + physical coordinates (mm) from output dir."""
    from pivtools_core.coordinate_utils import extract_coordinates

    arr = read_mat_contents(str(calib_output / "B00001.mat"))
    ux = arr[0, 0].astype(float)
    uy = arr[0, 1].astype(float)
    b_mask = arr[0, 2]
    valid = (b_mask == 0) & np.isfinite(ux) & np.isfinite(uy)

    coords_mat = loadmat(
        str(calib_output / "coordinates.mat"),
        struct_as_record=False, squeeze_me=True,
    )
    x_mm, y_mm = extract_coordinates(coords_mat["coordinates"], run=1)

    if x_mm.shape != ux.shape:
        if x_mm.ndim == 1 and ux.ndim == 2:
            x_mm = np.broadcast_to(x_mm[np.newaxis, :], ux.shape).copy()
            y_mm = np.broadcast_to(y_mm[:, np.newaxis], ux.shape).copy()

    return ux, uy, x_mm, y_mm, valid


def _load_merged_with_coords(merged_dir):
    """Load merged vectors + coordinates from Merged output dir."""
    from pivtools_core.coordinate_utils import extract_coordinates

    arr = read_mat_contents(str(merged_dir / "B00001.mat"))
    ux = arr[0, 0].astype(float)
    uy = arr[0, 1].astype(float)
    b_mask = arr[0, 2]
    valid = (b_mask == 0) & np.isfinite(ux) & np.isfinite(uy)

    coords_mat = loadmat(
        str(merged_dir / "coordinates.mat"),
        struct_as_record=False, squeeze_me=True,
    )
    x_mm, y_mm = extract_coordinates(coords_mat["coordinates"], run=1)

    if x_mm.shape != ux.shape:
        if x_mm.ndim == 1 and ux.ndim == 2:
            x_mm = np.broadcast_to(x_mm[np.newaxis, :], ux.shape).copy()
            y_mm = np.broadcast_to(y_mm[:, np.newaxis], ux.shape).copy()

    return ux, uy, x_mm, y_mm, valid


# ---------------------------------------------------------------------------
# Module-scoped fixture: run full pipeline once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def piv_workspace():
    """
    Build workspace, run PIV for both cameras, calibrate with alignment,
    and merge. Returns everything downstream tests need.
    """
    if not GT_PATH.exists():
        pytest.skip(
            f"Synthetic data not found at {DATA_DIR}. "
            "Run: python unit-tests/generate_poiseuille_multicam.py"
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

    # Copy synthetic images (source_dir is parent of Cam1/, Cam2/)
    for cam_name in ("Cam1", "Cam2"):
        cam_source = source_dir / cam_name
        cam_source.mkdir(exist_ok=True)
        for name in ("B00001_A.tif", "B00001_B.tif"):
            shutil.copy2(DATA_DIR / cam_name / name, cam_source / name)

    # ---- calibration model files (per camera) ----
    for cam_num, cam_rvec, cam_tvec in [
        (1, gt["rvec_cam1"], gt["tvec_cam1"]),
        (2, gt["rvec_cam2"], gt["tvec_cam2"]),
    ]:
        model_dir = (base_path / "calibration" / f"Cam{cam_num}"
                     / "dotboard_planar" / "model")
        model_dir.mkdir(parents=True, exist_ok=True)
        savemat(str(model_dir / "dotboard_model.mat"),
                _make_model(gt, cam_rvec, cam_tvec))

    # ---- overlap feature point (raw pixels, 0-based, y-down) ----
    overlap_on_cam1 = gt["overlap_on_cam1"].tolist()
    overlap_on_cam2 = gt["overlap_on_cam2"].tolist()

    # ---- config.yaml ----
    config_dict = {
        "paths": {
            "source_paths": [str(source_dir)],
            "base_paths": [str(base_path)],
            "camera_count": 2,
            "camera_numbers": [1, 2],
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
            "global_coordinates": {
                "enabled": True,
                "datum_camera": 1,
                "datum_pixel": overlap_on_cam1,
                "datum_physical": DATUM_PHYSICAL,
                "overlap_pairs": [{
                    "camera_a": 1,
                    "camera_b": 2,
                    "pixel_on_a": overlap_on_cam1,
                    "pixel_on_b": overlap_on_cam2,
                }],
            },
        },
    }

    config_path = WORKSPACE_DIR / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # ---- load config ----
    reload_config()
    config = Config(path=str(config_path))

    # ---- run instantaneous PIV for both cameras ----
    # Use a fresh Dask client per camera to avoid stale scattered data
    # (in-process mode with processes=False can lose scattered keys between runs)
    from dask.distributed import Client
    for cam_num in [1, 2]:
        client = Client(processes=False, n_workers=1, threads_per_worker=1)
        try:
            output_path = get_output_path(
                config, camera=cam_num, use_uncalibrated=True,
                piv_type="instantaneous",
            )
            run_instantaneous_piv(
                config=config,
                client=client,
                camera_num=cam_num,
                source_path=source_dir,
                output_path=output_path,
                base_path=base_path,
            )
        finally:
            client.close()

    # ---- pre-compute alignment shifts ----
    from pivtools_gui.calibration.global_coordinate_alignment import (
        GlobalCoordinateAligner,
    )
    aligner = GlobalCoordinateAligner(base_dir=base_path, config=config)
    alignment = aligner.precompute_camera_shifts(type_name="instantaneous")

    # ---- calibrate both cameras (with alignment fused in) ----
    from pivtools_gui.calibration.vector_calibration_production import (
        VectorCalibrator,
    )
    for cam_num in [1, 2]:
        calibrator = VectorCalibrator(
            base_dir=str(base_path),
            camera_num=cam_num,
            model_type="dotboard",
            dt=0.001,
            type_name="instantaneous",
            config=config,
        )
        calibrator.process_run(num_frame_pairs=1, alignment=alignment)

    # ---- merge ----
    # Pass num_frame_pairs and vector_format explicitly because VectorMerger
    # reads them from get_config() singleton which may not be our test config.
    from pivtools_gui.vector_merging.vector_merger import VectorMerger
    merger = VectorMerger(
        base_dir=base_path,
        cameras=[1, 2],
        type_name="instantaneous",
        num_frame_pairs=1,
        vector_format="B%05d.mat",
    )
    merge_result = merger.merge_all_frames()

    # ---- collect output paths ----
    uncal_cam1 = get_output_path(
        config, camera=1, use_uncalibrated=True, piv_type="instantaneous",
    )
    uncal_cam2 = get_output_path(
        config, camera=2, use_uncalibrated=True, piv_type="instantaneous",
    )
    cal_cam1 = get_output_path(
        config, camera=1, use_uncalibrated=False, piv_type="instantaneous",
    )
    cal_cam2 = get_output_path(
        config, camera=2, use_uncalibrated=False, piv_type="instantaneous",
    )
    merged_dir = merger.output_dir

    return {
        "workspace_dir": WORKSPACE_DIR,
        "source_dir": source_dir,
        "base_path": base_path,
        "config": config,
        "gt": gt,
        "uncal_cam1": uncal_cam1,
        "uncal_cam2": uncal_cam2,
        "cal_cam1": cal_cam1,
        "cal_cam2": cal_cam2,
        "merged_dir": merged_dir,
        "merge_result": merge_result,
        "alignment": alignment,
    }


# ===================================================================
# Tests on uncalibrated (PIV) output
# ===================================================================

class TestUncalibratedOutput:
    """Verify PIV ran correctly for both cameras before calibration."""

    def test_uncalibrated_output_exists(self, piv_workspace):
        for cam_key in ("uncal_cam1", "uncal_cam2"):
            out = piv_workspace[cam_key]
            assert (out / "B00001.mat").exists(), f"PIV vector file not found for {cam_key}"
            assert (out / "coordinates.mat").exists(), f"Coordinates not found for {cam_key}"

    def test_uncalibrated_displacement_range(self, piv_workspace):
        """Both cameras should show Poiseuille-like displacements."""
        for cam_key in ("uncal_cam1", "uncal_cam2"):
            out = piv_workspace[cam_key]
            arr = read_mat_contents(str(out / "B00001.mat"))
            ux = arr[0, 0]
            b_mask = arr[0, 2]

            valid = (b_mask == 0) & np.isfinite(ux)
            assert valid.sum() > 10, f"Too few valid vectors for {cam_key}"

            max_ux = np.nanmax(np.abs(ux[valid]))
            assert max_ux > 8.0, (
                f"[{cam_key}] max |ux_px|={max_ux:.2f}, expected > 8 (Poiseuille peak)"
            )
            assert max_ux < 16.0, (
                f"[{cam_key}] max |ux_px|={max_ux:.2f}, expected < 16"
            )


# ===================================================================
# Tests on alignment
# ===================================================================

class TestAlignment:
    """Verify global coordinate alignment applied non-trivial shifts."""

    def test_alignment_shifts_are_nontrivial(self, piv_workspace):
        """With datum_physical=[5,3], alignment shifts should be ~(5,3) mm.

        The overlap feature point projects to ~(0,0) mm natively on both
        cameras.  datum_physical=[5,3] forces a ~5mm x-shift and ~3mm
        y-shift on all cameras.  This verifies the alignment code actually
        computed and will apply real shifts.
        """
        alignment = piv_workspace["alignment"]
        assert alignment is not None, "Alignment should not be None"

        for cam_num in [1, 2]:
            shift_x, shift_y = alignment["camera_shifts"][cam_num]
            assert abs(shift_x - DATUM_PHYSICAL[0]) < 0.5, (
                f"Cam{cam_num} x-shift = {shift_x:.4f}, expected ~{DATUM_PHYSICAL[0]}"
            )
            assert abs(shift_y - DATUM_PHYSICAL[1]) < 0.5, (
                f"Cam{cam_num} y-shift = {shift_y:.4f}, expected ~{DATUM_PHYSICAL[1]}"
            )

    def test_cameras_see_different_y_ranges(self, piv_workspace):
        """Cameras should see different y-ranges (after alignment).

        Cam1 has tvec_y = -0.01333 (maps to +y in world), but calibration
        negates world-y (OpenCV y-down → physical y-up), so Cam1 sees
        negative physical y and Cam2 sees positive physical y.
        This catches swapped cameras or reversed y-axis.
        """
        _, _, _, y1, v1 = _load_calibrated_with_coords(piv_workspace["cal_cam1"])
        _, _, _, y2, v2 = _load_calibrated_with_coords(piv_workspace["cal_cam2"])

        cam1_y_mean = np.nanmean(y1[v1])
        cam2_y_mean = np.nanmean(y2[v2])

        # After y-negation: Cam1 sees lower y, Cam2 sees higher y
        assert cam2_y_mean > cam1_y_mean, (
            f"Cam2 mean y = {cam2_y_mean:.1f} mm should be > "
            f"Cam1 mean y = {cam1_y_mean:.1f} mm (after y-negation in calibration)"
        )

        # They should be well separated (>10mm apart in mean y)
        separation = abs(cam2_y_mean - cam1_y_mean)
        assert separation > 10.0, (
            f"Camera y-separation = {separation:.1f} mm, expected > 10 mm"
        )

    def test_datum_pixel_maps_to_datum_physical(self, piv_workspace):
        """The overlap feature point should map to datum_physical in aligned coords.

        This is the strongest alignment validation: the datum pixel on Cam1
        should map to exactly (5.0, 3.0) mm in the calibrated coordinate
        system.  We check by finding the nearest window centre and verifying
        the coordinate is close to datum_physical.
        """
        _, _, x1, y1, v1 = _load_calibrated_with_coords(piv_workspace["cal_cam1"])

        # The datum pixel (overlap feature point on Cam1) is the physical
        # origin projected to Cam1.  In the aligned system, it should be at
        # datum_physical = [5.0, 3.0].
        # Since we can't check the exact pixel, verify that the coordinate
        # grid includes values near datum_physical.
        dx = DATUM_PHYSICAL[0]
        dy = DATUM_PHYSICAL[1]

        # Find nearest valid window to (5.0, 3.0)
        dist = np.full_like(x1, np.inf)
        dist[v1] = np.sqrt((x1[v1] - dx) ** 2 + (y1[v1] - dy) ** 2)
        nearest_dist = np.nanmin(dist)

        # Window spacing is ~2.1 mm, so nearest window should be within
        # one grid cell of the datum point.
        assert nearest_dist < 3.0, (
            f"No window within 3mm of datum_physical {DATUM_PHYSICAL}: "
            f"nearest distance = {nearest_dist:.2f} mm"
        )


# ===================================================================
# Tests on calibrated output (per-camera)
# ===================================================================

class TestCalibratedOutput:
    """Verify per-camera calibrated velocities match Poiseuille."""

    def test_calibrated_output_exists(self, piv_workspace):
        for cam_key in ("cal_cam1", "cal_cam2"):
            out = piv_workspace[cam_key]
            assert (out / "B00001.mat").exists(), f"Calibrated vectors not found for {cam_key}"
            assert (out / "coordinates.mat").exists(), f"Calibrated coords not found for {cam_key}"

    @pytest.mark.parametrize("cam_key", ["cal_cam1", "cal_cam2"])
    def test_per_camera_ux_profile(self, piv_workspace, cam_key):
        """Each camera's calibrated ux should match Poiseuille at its window locations."""
        gt = piv_workspace["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])

        ux, uy, x_mm, y_mm, valid = _load_calibrated_with_coords(
            piv_workspace[cam_key]
        )
        # Poiseuille centre is at y = datum_physical[1] in aligned coords
        gt_ux = poiseuille_ux(y_mm, u_max, H_phys, y_offset_mm=DATUM_PHYSICAL[1])

        # Exclude near-wall windows (wall at ±H_phys/2 from channel centre)
        y_from_centre = np.abs(y_mm - DATUM_PHYSICAL[1])
        interior = valid & (y_from_centre < (H_phys * 1000 / 2.0) * 0.85)
        assert interior.sum() > 10, f"[{cam_key}] Too few interior windows"

        err = np.abs(ux[interior] - gt_ux[interior])
        median_err = np.nanmedian(err)
        assert median_err < 0.03, (
            f"[{cam_key}] median |ux - gt| = {median_err:.4f} m/s, expected < 0.03"
        )



    @pytest.mark.parametrize("cam_key", ["cal_cam1", "cal_cam2"])
    def test_coordinate_scale_factor(self, piv_workspace, cam_key):
        """Physical grid spacing should match expected px/mm conversion.

        With 32px windows at 50% overlap -> 16px spacing, and ~15 px/mm,
        expected physical spacing is ~1.067 mm. Allows 0.15 mm tolerance
        for tilt-induced perspective distortion.
        """
        _, _, x_mm, y_mm, valid = _load_calibrated_with_coords(
            piv_workspace[cam_key]
        )
        if x_mm.ndim == 2 and x_mm.shape[1] > 1:
            dx_mm = np.abs(np.diff(x_mm[0, :]))
            median_dx = np.nanmedian(dx_mm[dx_mm > 0])
        else:
            pytest.skip("Need 2D grid for spacing check")

        expected_spacing = 16.0 / 15.0  # 16px / 15 px/mm ~ 1.067 mm
        assert abs(median_dx - expected_spacing) < 0.15, (
            f"[{cam_key}] median x-spacing = {median_dx:.4f} mm, "
            f"expected ~{expected_spacing:.3f} mm (16px / 15 px/mm)"
        )

# ===================================================================
# Tests on merged output (the main event)
# ===================================================================

class TestMergedOutput:
    """Verify merged velocity field recovers Poiseuille across both cameras."""

    def test_merge_succeeded(self, piv_workspace):
        result = piv_workspace["merge_result"]
        assert result.get("success", False), f"Merge failed: {result.get('error', 'unknown')}"

    def test_merged_output_exists(self, piv_workspace):
        merged_dir = piv_workspace["merged_dir"]
        assert (merged_dir / "B00001.mat").exists(), "Merged vector file not found"
        assert (merged_dir / "coordinates.mat").exists(), "Merged coordinates not found"

    def test_merged_ux_profile(self, piv_workspace):
        """Merged ux should match Poiseuille profile across the full domain."""
        gt = piv_workspace["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])

        ux, uy, x_mm, y_mm, valid = _load_merged_with_coords(
            piv_workspace["merged_dir"]
        )
        gt_ux = poiseuille_ux(y_mm, u_max, H_phys, y_offset_mm=DATUM_PHYSICAL[1])

        y_from_centre = np.abs(y_mm - DATUM_PHYSICAL[1])
        interior = valid & (y_from_centre < (H_phys * 1000 / 2.0) * 0.85)
        assert interior.sum() > 10, "Too few interior windows in merged data"

        err = np.abs(ux[interior] - gt_ux[interior])
        median_err = np.nanmedian(err)
        assert median_err < 0.01, (
            f"Merged median |ux - gt| = {median_err:.4f} m/s, expected < 0.01"
        )

    def test_merged_uy_near_zero(self, piv_workspace):
        """Merged |uy| should be ~0 (Poiseuille has no vertical velocity)."""
        ux, uy, x_mm, y_mm, valid = _load_merged_with_coords(
            piv_workspace["merged_dir"]
        )
        gt = piv_workspace["gt"]
        H_phys = float(gt["H_phys"])

        y_from_centre = np.abs(y_mm - DATUM_PHYSICAL[1])
        interior = valid & (y_from_centre < (H_phys * 1000 / 2.0) * 0.85)
        median_uy = np.nanmedian(np.abs(uy[interior]))
        assert median_uy < 0.005, (
            f"Merged median |uy| = {median_uy:.4f} m/s, expected < 0.005"
        )

    def test_merged_domain_wider_than_single_camera(self, piv_workspace):
        """Merged y-range should exceed either camera's y-range individually.

        This is a stronger check than just coverage percentage — it directly
        verifies that two cameras were combined into a wider field.
        """
        _, _, _, y_m, v_m = _load_merged_with_coords(piv_workspace["merged_dir"])
        _, _, _, y1, v1 = _load_calibrated_with_coords(piv_workspace["cal_cam1"])
        _, _, _, y2, v2 = _load_calibrated_with_coords(piv_workspace["cal_cam2"])

        merged_range = np.nanmax(y_m[v_m]) - np.nanmin(y_m[v_m])
        cam1_range = np.nanmax(y1[v1]) - np.nanmin(y1[v1])
        cam2_range = np.nanmax(y2[v2]) - np.nanmin(y2[v2])

        assert merged_range > cam1_range, (
            f"Merged y-range ({merged_range:.1f} mm) should exceed "
            f"Cam1 ({cam1_range:.1f} mm)"
        )
        assert merged_range > cam2_range, (
            f"Merged y-range ({merged_range:.1f} mm) should exceed "
            f"Cam2 ({cam2_range:.1f} mm)"
        )

    def test_merged_domain_coverage(self, piv_workspace):
        """Merged y-coordinates should span at least 80% of the physical domain."""
        gt = piv_workspace["gt"]
        H_phys = float(gt["H_phys"])

        _, _, _, y_mm, valid = _load_merged_with_coords(
            piv_workspace["merged_dir"]
        )

        y_range = np.nanmax(y_mm[valid]) - np.nanmin(y_mm[valid])
        theoretical_range = H_phys * 1000  # mm
        coverage = y_range / theoretical_range
        assert coverage > 0.80, (
            f"Merged y-range = {y_range:.1f} mm, theoretical = {theoretical_range:.1f} mm, "
            f"coverage = {coverage:.1%}, expected > 80%"
        )

    def test_overlap_region_continuity(self, piv_workspace):
        """Gradient check: dux/dy should be smooth across the merge boundary.

        In Poiseuille flow, dux/dy is smooth and monotonic. If alignment is off,
        there will be a sharp spike in the derivative at the merge boundary.
        Assert that the max gradient in the overlap region does not exceed
        the max gradient in single-camera regions by more than 1.5x.
        """
        ux, _, x_mm, y_mm, valid = _load_merged_with_coords(
            piv_workspace["merged_dir"]
        )

        if ux.ndim < 2 or ux.shape[0] < 5:
            pytest.skip("Need 2D grid with enough rows for gradient check")

        ux_clean = np.where(valid, ux, np.nan)
        dux_dy = np.gradient(ux_clean, axis=0)

        # Find overlap region from per-camera y-extents
        cal_cam1_data = _load_calibrated_with_coords(piv_workspace["cal_cam1"])
        cal_cam2_data = _load_calibrated_with_coords(piv_workspace["cal_cam2"])
        y1_min, y1_max = np.nanmin(cal_cam1_data[3]), np.nanmax(cal_cam1_data[3])
        y2_min, y2_max = np.nanmin(cal_cam2_data[3]), np.nanmax(cal_cam2_data[3])

        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)

        if overlap_y_min >= overlap_y_max:
            pytest.skip("No overlap region detected between cameras")

        row_y = np.nanmean(y_mm, axis=1) if y_mm.ndim == 2 else y_mm
        overlap_rows = (row_y >= overlap_y_min) & (row_y <= overlap_y_max)
        single_rows = ~overlap_rows

        if not np.any(overlap_rows) or not np.any(single_rows):
            pytest.skip("Cannot separate overlap and single-camera regions")

        grad_overlap = np.nanmax(np.abs(dux_dy[overlap_rows, :]))
        grad_single = np.nanmax(np.abs(dux_dy[single_rows, :]))

        if grad_single < 1e-6:
            pytest.skip("Single-camera gradient too small for meaningful comparison")

        ratio = grad_overlap / grad_single
        assert ratio < 1.5, (
            f"Gradient spike at merge boundary: overlap max grad = {grad_overlap:.4f}, "
            f"single-camera max grad = {grad_single:.4f}, ratio = {ratio:.2f}, "
            "expected < 1.5"
        )

    def test_merged_coordinates_reflect_datum_offset(self, piv_workspace):
        """Merged coordinate centre should be near datum_physical, not at (0,0).

        With datum_physical=[5,3], the median x-coordinate should be near 5mm
        and the median y-coordinate should be near 3mm.  If alignment was
        skipped or applied incorrectly, coordinates would centre near (0,0).
        """
        _, _, x_mm, y_mm, valid = _load_merged_with_coords(
            piv_workspace["merged_dir"]
        )

        median_x = np.nanmedian(x_mm[valid])
        median_y = np.nanmedian(y_mm[valid])

        # Median should be near datum_physical (within ~5mm, given asymmetric
        # camera coverage)
        assert abs(median_x - DATUM_PHYSICAL[0]) < 8.0, (
            f"Merged median x = {median_x:.1f} mm, expected near {DATUM_PHYSICAL[0]} mm. "
            "Alignment may not have been applied."
        )
        assert abs(median_y - DATUM_PHYSICAL[1]) < 8.0, (
            f"Merged median y = {median_y:.1f} mm, expected near {DATUM_PHYSICAL[1]} mm. "
            "Alignment may not have been applied."
        )

        # Crucially: coordinates should NOT be centred on (0, 0)
        dist_from_origin = np.sqrt(median_x ** 2 + median_y ** 2)
        dist_from_datum = np.sqrt(
            (median_x - DATUM_PHYSICAL[0]) ** 2
            + (median_y - DATUM_PHYSICAL[1]) ** 2
        )
        assert dist_from_datum < dist_from_origin, (
            f"Merged coords closer to origin ({dist_from_origin:.1f} mm) than to "
            f"datum_physical ({dist_from_datum:.1f} mm). Alignment may be missing."
        )


# ===================================================================
# Diagnostic figures (gated by --make-figures)
# ===================================================================

class TestDiagnosticFigures:
    """Generate diagnostic plots when --make-figures is passed."""

    def test_make_figures(self, piv_workspace, make_figures):
        if not make_figures:
            pytest.skip("Pass --make-figures to generate diagnostic plots")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pivtools_core.coordinate_utils import extract_coordinates

        gt = piv_workspace["gt"]
        u_max = float(gt["u_max"])
        H_phys = float(gt["H_phys"])
        y_offset = DATUM_PHYSICAL[1]

        # --- Load uncalibrated (pixel) data for both cameras ---
        def _load_uncal(out_dir):
            arr = read_mat_contents(str(out_dir / "B00001.mat"))
            ux_px = arr[0, 0].astype(float)
            uy_px = arr[0, 1].astype(float)
            b_mask = arr[0, 2]
            valid = (b_mask == 0) & np.isfinite(ux_px) & np.isfinite(uy_px)
            coords_mat = loadmat(
                str(out_dir / "coordinates.mat"),
                struct_as_record=False, squeeze_me=True,
            )
            x_px, y_px = extract_coordinates(coords_mat["coordinates"], run=1)
            if x_px.shape != ux_px.shape and x_px.ndim == 1 and ux_px.ndim == 2:
                x_px = np.broadcast_to(x_px[np.newaxis, :], ux_px.shape).copy()
                y_px = np.broadcast_to(y_px[:, np.newaxis], ux_px.shape).copy()
            return ux_px, uy_px, x_px, y_px, valid

        u1_ux, u1_uy, u1_x, u1_y, u1_v = _load_uncal(piv_workspace["uncal_cam1"])
        u2_ux, u2_uy, u2_x, u2_y, u2_v = _load_uncal(piv_workspace["uncal_cam2"])

        # --- Load calibrated (mm, m/s) data for both cameras ---
        c1_ux, c1_uy, c1_x, c1_y, c1_v = _load_calibrated_with_coords(
            piv_workspace["cal_cam1"]
        )
        c2_ux, c2_uy, c2_x, c2_y, c2_v = _load_calibrated_with_coords(
            piv_workspace["cal_cam2"]
        )

        # --- Load merged data ---
        m_ux, m_uy, m_x, m_y, m_v = _load_merged_with_coords(
            piv_workspace["merged_dir"]
        )
        m_gt_ux = np.full_like(m_ux, np.nan)
        m_gt_ux[m_v] = poiseuille_ux(m_y[m_v], u_max, H_phys,
                                      y_offset_mm=y_offset)

        # ---- 3-row x 3-col figure ----
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(
            "Multi-Camera Pipeline: Uncalibrated \u2192 Calibrated \u2192 Merged"
            f"  (datum_physical={DATUM_PHYSICAL})",
            fontsize=15, fontweight="bold",
        )

        # Helper: imshow a 2-D field with NaN masking
        def _imshow(ax, field, valid, title, cmap="viridis", **kwargs):
            display = np.where(valid, field, np.nan)
            im = ax.imshow(display, cmap=cmap, origin="upper", aspect="auto",
                           **kwargs)
            ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.8)

        # ---- Row 0: Uncalibrated (pixel displacements) ----
        _imshow(axes[0, 0], u1_ux, u1_v, "Cam1 uncal ux (px)")
        axes[0, 0].set_ylabel("Uncalibrated\n(pixels)", fontsize=11,
                              fontweight="bold")

        _imshow(axes[0, 1], u2_ux, u2_v, "Cam2 uncal ux (px)")

        ax = axes[0, 2]
        if u1_y.ndim == 2:
            mid1 = u1_y.shape[1] // 2
            ax.plot(u1_ux[:, mid1], u1_y[:, mid1], 'b.-', markersize=3,
                    alpha=0.7, label="Cam1")
        if u2_y.ndim == 2:
            mid2 = u2_y.shape[1] // 2
            ax.plot(u2_ux[:, mid2], u2_y[:, mid2], 'r.-', markersize=3,
                    alpha=0.7, label="Cam2")
        ax.set_xlabel("ux displacement (px)")
        ax.set_ylabel("y (px, uncalibrated)")
        ax.set_title("Both cameras (mid-col profile)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Row 1: Calibrated (m/s, mm coordinates) ----
        _imshow(axes[1, 0], c1_ux, c1_v, "Cam1 calibrated ux (m/s)")
        axes[1, 0].set_ylabel("Calibrated\n(m/s, mm)", fontsize=11,
                              fontweight="bold")

        _imshow(axes[1, 1], c2_ux, c2_v, "Cam2 calibrated ux (m/s)")

        ax = axes[1, 2]
        if c1_y.ndim == 2:
            mid1 = c1_y.shape[1] // 2
            ax.plot(c1_ux[:, mid1], c1_y[:, mid1], 'b.-', markersize=3,
                    alpha=0.7, label="Cam1")
        if c2_y.ndim == 2:
            mid2 = c2_y.shape[1] // 2
            ax.plot(c2_ux[:, mid2], c2_y[:, mid2], 'r.-', markersize=3,
                    alpha=0.7, label="Cam2")
        y_gt = np.linspace(-H_phys * 500 + y_offset, H_phys * 500 + y_offset, 200)
        ax.plot(poiseuille_ux(y_gt, u_max, H_phys, y_offset_mm=y_offset),
                y_gt, 'k--', linewidth=1, alpha=0.4, label="GT")
        ax.set_xlabel("ux (m/s)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Both cameras (mid-col profile)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---- Row 2: Merged ----
        _imshow(axes[2, 0], m_ux, m_v, "Merged ux (m/s)")
        axes[2, 0].set_ylabel("Merged", fontsize=11, fontweight="bold")

        ax = axes[2, 1]
        if m_y.ndim == 2:
            mid_m = m_y.shape[1] // 2
            ax.plot(m_ux[:, mid_m], m_y[:, mid_m], 'b-', linewidth=2,
                    label="Merged")
            ax.plot(m_gt_ux[:, mid_m], m_y[:, mid_m], 'k--', linewidth=1,
                    alpha=0.5, label="GT")
        else:
            ax.plot(m_ux, m_y, 'b-', linewidth=2, label="Merged")
            ax.plot(m_gt_ux, m_y, 'k--', linewidth=1, alpha=0.5, label="GT")
        ax.set_xlabel("ux (m/s)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Merged vs Ground Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2, 2]
        pct_err = np.full_like(m_ux, np.nan)
        pct_err[m_v] = (m_ux[m_v] - m_gt_ux[m_v]) / u_max * 100.0
        if m_y.ndim == 2:
            ax.plot(pct_err[:, mid_m], m_y[:, mid_m], 'r-', linewidth=2)
        else:
            ax.plot(pct_err, m_y, 'r-', linewidth=2)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Error (% of u_max)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Merged Error")
        ax.grid(True, alpha=0.3)

        # Summary stats
        y_from_centre = np.abs(m_y - y_offset)
        interior = m_v & (y_from_centre < (H_phys * 1000 / 2.0) * 0.85)
        if interior.any():
            err_ux = np.abs(m_ux[interior] - m_gt_ux[interior])
            alignment = piv_workspace["alignment"]
            s1 = alignment["camera_shifts"][1]
            s2 = alignment["camera_shifts"][2]
            stats_text = (
                f"Interior ({interior.sum()}/{m_v.sum()} valid):  "
                f"ux median err = {np.nanmedian(err_ux):.4f} m/s,  "
                f"max = {np.nanmax(err_ux):.4f},  "
                f"median % = {np.nanmedian(err_ux/u_max*100):.2f}%  |  "
                f"|uy| = {np.nanmedian(np.abs(m_uy[interior])):.4f}  |  "
                f"y: [{np.nanmin(m_y[m_v]):.1f}, {np.nanmax(m_y[m_v]):.1f}] mm  |  "
                f"shifts: Cam1=({s1[0]:.2f},{s1[1]:.2f}), "
                f"Cam2=({s2[0]:.2f},{s2[1]:.2f}) mm"
            )
        else:
            stats_text = "No interior windows"

        fig.text(
            0.5, 0.005, stats_text, ha="center", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        fig_dir = piv_workspace["workspace_dir"] / "figures"
        fig_dir.mkdir(exist_ok=True)
        out_path = fig_dir / "multicam_merge_verification.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

        print(f"  Figure saved: {out_path}")
