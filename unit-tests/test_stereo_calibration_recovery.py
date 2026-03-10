#!/usr/bin/env python3
"""
test_stereo_calibration_recovery.py

Verifies that stereo calibration recovers known ground-truth inter-camera
parameters from synthetic stereo image pairs.

Tests individual methods directly (detect_pattern, _match_object_points,
_perform_stereo_calibration) rather than process_camera_pair which needs
full config infrastructure.

Usage:
    pytest unit-tests/test_stereo_calibration_recovery.py -v
"""

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pivtools_gui.stereo_reconstruction.stereo_dotboard_calibration_production import (
    StereoDotboardCalibrator,
)
from pivtools_gui.stereo_reconstruction.stereo_charuco_calibration_production import (
    StereoCharucoCalibrator,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SYNTH_DIR = Path(__file__).resolve().parent / "synthetic_calibration"
GT_PATH = SYNTH_DIR / "stereo_ground_truth.npz"
DOTBOARD_CAM1 = SYNTH_DIR / "stereo_dotboard" / "cam1"
DOTBOARD_CAM2 = SYNTH_DIR / "stereo_dotboard" / "cam2"
CHARUCO_CAM1 = SYNTH_DIR / "stereo_charuco" / "cam1"
CHARUCO_CAM2 = SYNTH_DIR / "stereo_charuco" / "cam2"

# Board params (must match generator)
DOT_SPACING_MM = 15.0
SQUARES_H = 10
SQUARES_V = 7
SQUARE_SIZE = 0.030
MARKER_RATIO = 0.5
ARUCO_DICT = "DICT_4X4_1000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MinimalConfig:
    """Stand-in for Config that avoids loading config.yaml."""
    stereo_calibration = {}
    stereo_dotboard_calibration = {}
    stereo_charuco_calibration = {}
    charuco_calibration = {}
    dt = 1.0
    calibration_image_format = "calib%05d.png"


def rotation_angle_error(R1, R2):
    """Angle in degrees between two rotation matrices."""
    R_diff = R1.T @ R2
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(math.acos(cos_angle))


def baseline_magnitude_error(T_recovered, T_gt):
    """Relative error in baseline magnitude."""
    mag_rec = np.linalg.norm(T_recovered)
    mag_gt = np.linalg.norm(T_gt)
    return abs(mag_rec - mag_gt) / mag_gt


def _read_images(cam_dir):
    """Read all calib*.png images from a directory."""
    paths = sorted(cam_dir.glob("calib*.png"))
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stereo_ground_truth():
    """Load stereo ground-truth parameters."""
    if not GT_PATH.exists():
        pytest.fail(
            f"Stereo synthetic data not found at {GT_PATH}. "
            "Run: python pivtools_cli/generate_synthetic_stereo.py"
        )
    return dict(np.load(str(GT_PATH)))


@pytest.fixture(scope="module")
def tmpdir_module():
    """Module-scoped temp directory for calibrator output."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def dotboard_calibrator(tmpdir_module):
    """StereoDotboardCalibrator configured for synthetic data."""
    return StereoDotboardCalibrator(
        config=_MinimalConfig(),
        source_dir=str(SYNTH_DIR / "stereo_dotboard"),
        base_dir=str(tmpdir_module),
        camera_pair=[1, 2],
        dot_spacing_mm=DOT_SPACING_MM,
        file_pattern="calib%05d.png",
    )


@pytest.fixture(scope="module")
def charuco_calibrator(tmpdir_module):
    """StereoCharucoCalibrator configured for synthetic data."""
    return StereoCharucoCalibrator(
        config=_MinimalConfig(),
        source_dir=str(SYNTH_DIR / "stereo_charuco"),
        base_dir=str(tmpdir_module),
        camera_pair=[1, 2],
        squares_h=SQUARES_H,
        squares_v=SQUARES_V,
        square_size=SQUARE_SIZE,
        marker_ratio=MARKER_RATIO,
        aruco_dict=ARUCO_DICT,
        file_pattern="calib%05d.png",
    )


@pytest.fixture(scope="module")
def dotboard_detections(dotboard_calibrator):
    """Run detect_pattern on all 10 stereo dotboard image pairs."""
    imgs_cam1 = _read_images(DOTBOARD_CAM1)
    imgs_cam2 = _read_images(DOTBOARD_CAM2)
    assert len(imgs_cam1) == len(imgs_cam2), "Mismatched image counts"

    detections = []
    for img1, img2 in zip(imgs_cam1, imgs_cam2):
        r1 = dotboard_calibrator.detect_pattern(img1)
        r2 = dotboard_calibrator.detect_pattern(img2)
        detections.append((r1, r2))
    return detections


@pytest.fixture(scope="module")
def charuco_detections(charuco_calibrator):
    """Run detect_pattern on all 10 stereo charuco image pairs."""
    imgs_cam1 = _read_images(CHARUCO_CAM1)
    imgs_cam2 = _read_images(CHARUCO_CAM2)
    assert len(imgs_cam1) == len(imgs_cam2), "Mismatched image counts"

    detections = []
    for img1, img2 in zip(imgs_cam1, imgs_cam2):
        r1 = charuco_calibrator.detect_pattern(img1)
        r2 = charuco_calibrator.detect_pattern(img2)
        detections.append((r1, r2))
    return detections


@pytest.fixture(scope="module")
def dotboard_stereo_result(dotboard_calibrator, dotboard_detections, stereo_ground_truth):
    """Run _match_object_points + _perform_stereo_calibration for dotboard."""
    objp = dotboard_calibrator.make_object_points()
    objpoints, imgpoints1, imgpoints2 = [], [], []

    for r1, r2 in dotboard_detections:
        if not r1[0] or not r2[0]:
            continue
        match = dotboard_calibrator._match_object_points(objp, r1, r2)
        if match is None:
            continue
        obj_pts, img1, img2 = match
        objpoints.append(obj_pts)
        imgpoints1.append(img1)
        imgpoints2.append(img2)

    assert len(objpoints) >= 3, f"Only {len(objpoints)} matched pairs (need >= 3)"

    W = int(stereo_ground_truth["image_width"])
    H = int(stereo_ground_truth["image_height"])
    result = dotboard_calibrator._perform_stereo_calibration(
        objpoints, imgpoints1, imgpoints2, (W, H),
    )
    result["num_matched_pairs"] = len(objpoints)
    return result


@pytest.fixture(scope="module")
def charuco_stereo_result(charuco_calibrator, charuco_detections, stereo_ground_truth):
    """Run _match_object_points + _perform_stereo_calibration for charuco."""
    objp = charuco_calibrator.make_object_points()
    objpoints, imgpoints1, imgpoints2 = [], [], []

    for r1, r2 in charuco_detections:
        if not r1[0] or not r2[0]:
            continue
        match = charuco_calibrator._match_object_points(objp, r1, r2)
        if match is None:
            continue
        obj_pts, img1, img2 = match
        objpoints.append(obj_pts)
        imgpoints1.append(img1)
        imgpoints2.append(img2)

    assert len(objpoints) >= 3, f"Only {len(objpoints)} matched pairs (need >= 3)"

    W = int(stereo_ground_truth["image_width"])
    H = int(stereo_ground_truth["image_height"])
    result = charuco_calibrator._perform_stereo_calibration(
        objpoints, imgpoints1, imgpoints2, (W, H),
    )
    result["num_matched_pairs"] = len(objpoints)
    return result


# ---------------------------------------------------------------------------
# Dotboard tests
# ---------------------------------------------------------------------------

def test_dotboard_detection_succeeds(dotboard_detections):
    """At least 8/10 views detect in both cameras."""
    both_found = sum(1 for r1, r2 in dotboard_detections if r1[0] and r2[0])
    assert both_found >= 8, f"Only {both_found}/10 pairs detected in both cameras"


def test_dotboard_detection_grid_size(dotboard_detections):
    """Each successful detection has a reasonable point count (>50)."""
    for r1, r2 in dotboard_detections:
        if r1[0]:
            assert len(r1[1]) > 50, f"Cam1 detection too small: {len(r1[1])} points"
        if r2[0]:
            assert len(r2[1]) > 50, f"Cam2 detection too small: {len(r2[1])} points"


def test_dotboard_match_common_points(dotboard_stereo_result):
    """_match_object_points found enough common points."""
    assert dotboard_stereo_result["num_matched_pairs"] >= 8


def test_dotboard_stereo_rms(dotboard_stereo_result):
    """Stereo RMS reprojection error < 1.0 px."""
    rms = dotboard_stereo_result["stereo_rms_error"]
    assert rms < 1.0, f"Stereo RMS {rms:.4f} px exceeds 1.0 px threshold"


def test_dotboard_baseline_recovery(dotboard_stereo_result, stereo_ground_truth):
    """||T|| magnitude within 10% of ground truth.

    Note: dotboard object points are in mm (dot_spacing_mm), so T_recovered
    is in mm. Ground truth T_stereo is in meters — scale by 1000 to compare.
    """
    T_rec = dotboard_stereo_result["translation_vector"]
    T_gt = stereo_ground_truth["T_stereo"] * 1000.0  # m -> mm
    err = baseline_magnitude_error(T_rec, T_gt)
    assert err < 0.10, f"Baseline magnitude error {err:.4f} exceeds 10%"


def test_dotboard_rotation_recovery(dotboard_stereo_result, stereo_ground_truth):
    """Rotation angle error < 2 degrees."""
    R_rec = dotboard_stereo_result["rotation_matrix"]
    R_gt = stereo_ground_truth["R_stereo"]
    err_deg = rotation_angle_error(R_rec, R_gt)
    assert err_deg < 2.0, f"Rotation error {err_deg:.2f} deg exceeds 2 deg threshold"


# ---------------------------------------------------------------------------
# ChArUco tests
# ---------------------------------------------------------------------------

def test_charuco_detection_succeeds(charuco_detections):
    """At least 8/10 views detect with valid corners."""
    both_found = sum(1 for r1, r2 in charuco_detections if r1[0] and r2[0])
    assert both_found >= 8, f"Only {both_found}/10 pairs detected in both cameras"


def test_charuco_detection_has_ids(charuco_detections):
    """Detected corners have integer IDs."""
    for r1, r2 in charuco_detections:
        if r1[0]:
            ids = r1[2]
            assert ids is not None and len(ids) > 0
            assert np.issubdtype(ids.dtype, np.integer), f"IDs dtype {ids.dtype} not integer"
        if r2[0]:
            ids = r2[2]
            assert ids is not None and len(ids) > 0
            assert np.issubdtype(ids.dtype, np.integer), f"IDs dtype {ids.dtype} not integer"


def test_charuco_match_common_ids(charuco_stereo_result):
    """_match_object_points found enough common ID-matched points."""
    assert charuco_stereo_result["num_matched_pairs"] >= 8


def test_charuco_stereo_rms(charuco_stereo_result):
    """Stereo RMS reprojection error < 1.0 px."""
    rms = charuco_stereo_result["stereo_rms_error"]
    assert rms < 1.0, f"Stereo RMS {rms:.4f} px exceeds 1.0 px threshold"


def test_charuco_baseline_recovery(charuco_stereo_result, stereo_ground_truth):
    """||T|| magnitude within 10% of ground truth."""
    T_rec = charuco_stereo_result["translation_vector"]
    T_gt = stereo_ground_truth["T_stereo"]
    err = baseline_magnitude_error(T_rec, T_gt)
    assert err < 0.10, f"Baseline magnitude error {err:.4f} exceeds 10%"


def test_charuco_rotation_recovery(charuco_stereo_result, stereo_ground_truth):
    """Rotation angle error < 2 degrees."""
    R_rec = charuco_stereo_result["rotation_matrix"]
    R_gt = stereo_ground_truth["R_stereo"]
    err_deg = rotation_angle_error(R_rec, R_gt)
    assert err_deg < 2.0, f"Rotation error {err_deg:.2f} deg exceeds 2 deg threshold"
