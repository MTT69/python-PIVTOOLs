"""Smoke tests on real calibration data (run with ``-m slow``; skipped when absent).

The data is the external ``calibration_check`` volume (``$PIVTOOLS_CAL_DATA``, default
``/Volumes/DATA/calibration_check``), read-only. Every test reads through the production
calibration reader and detector, then compares against a reference that was not produced
by this code, so a pass means agreement with an independent result.

* Andre (stepped board, LaVision .im7): ``SteppedDetector`` against the DaVis
  ``MarkPositionTable.xml`` pixel positions, on both cameras of all three projects.
* Bailey (dot board, six views per camera): ``DotboardDetector`` on every view, against
  the DaVis mark table of the six-view calibration.
* Merle (three-camera joint dot board): the stored joint model and sidecar load.

Thresholds sit below the values measured on 2026-10-19 (quoted at each constant) so a
regression trips them and normal run-to-run variation does not.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pivtools_core.image_handling.calibration_loader import read_calibration_frame_at
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration.detection.dotboard import DotboardDetector, DotboardParams
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams

pytestmark = pytest.mark.slow

IM7_FORMAT = "B%05d.im7"
IM7_TYPE = "lavision_im7"
# Each DaVis camera folder holds single-camera files, so the reader gets the subfolder flag
# and the rig camera count.
NUM_CAMERAS = 2

# A DaVis mark counts as "shared" when a detected point lies within this distance.
SHARED_MARK_RADIUS_PX = 5.0
# Agreement tolerance on shared marks (the brief's 1 px).
AGREE_TOL_PX = 1.0

ANDRE_PROJECTS = (
    "Project_FlowMaster_260129_103855_x0_2",
    "Project_FlowMaster_260129_141523_x-50",
    "Project_FlowMaster_260129_152105_x25",
)
# Measured over the 3 projects x 2 cameras: 94.4-100 % of DaVis marks shared, median
# distance 0.19-0.36 px, 91.5-100 % of shared marks within 1 px.
ANDRE_MIN_SHARED_FRACTION = 0.90
ANDRE_MAX_MEDIAN_PX = 0.5
ANDRE_MIN_WITHIN_TOL_FRACTION = 0.85

BAILEY_SPACING_MM = 21.2
# Measured on the eight views DaVis marked with >= 100 dots: 91.5-100 % of DaVis marks
# shared, median distance 0.64-1.27 px (a ~0.5 px mean offset in y; dot diameter is
# 10.6 mm, ~140 px, so centroid conventions differ by sub-pixel amounts).
BAILEY_MIN_SHARED_FRACTION = 0.90
BAILEY_MAX_MEDIAN_PX = 1.5
BAILEY_VIEWS = range(1, 7)
# The detector finds no grid on these two views; DaVis itself marked only 39 and 48 dots
# on them (partial boards), against 93-231 on the others.
BAILEY_LOW_MARK_VIEWS = {(1, 3), (2, 2)}

MERLE_CAMERAS = [1, 2, 3]
MERLE_VIEWS_PER_CAMERA = 5
# Measured joint RMS 0.217 px.
MERLE_MAX_RMS_PX = 0.5


def _davis_marks(table: Path, camera: int, view: int = 1) -> np.ndarray:
    """DaVis ``RawPos`` (x, y) of every mark in one camera's ``view`` (1-based), shape (N, 2).

    A DaVis table lists one ``<Camera>`` element per camera (the file in each camera folder
    carries both), each holding that camera's ``<View>`` elements.
    """
    cams = ET.parse(table).getroot().findall("Camera")
    views = cams[camera - 1].findall("View")
    marks = views[view - 1].findall("Mark")
    return np.array(
        [
            [float(m.find("RawPos").get("x")), float(m.find("RawPos").get("y"))]
            for m in marks
        ]
    )


def _agreement(detected: np.ndarray, davis: np.ndarray) -> dict:
    """Nearest-detected-point statistics for every DaVis mark.

    Marks lie 15-21 mm apart (hundreds of pixels), so a nearest point within
    ``SHARED_MARK_RADIUS_PX`` is the same dot, never a neighbour.
    """
    dist, idx = cKDTree(detected).query(davis)
    shared = dist < SHARED_MARK_RADIUS_PX
    off = (detected[idx] - davis)[shared]
    return {
        "n_detected": len(detected),
        "n_davis": len(davis),
        "n_shared": int(shared.sum()),
        "n_within_tol": int((dist < AGREE_TOL_PX).sum()),
        "median_px": float(np.median(dist[shared])) if shared.any() else float("nan"),
        "max_shared_px": float(dist[shared].max()) if shared.any() else float("nan"),
        "mean_offset_px": tuple(off.mean(axis=0)) if shared.any() else (np.nan,) * 2,
    }


def _read_im7(folder: Path, camera: int, frame: int) -> np.ndarray:
    return read_calibration_frame_at(
        folder,
        camera,
        frame,
        IM7_FORMAT,
        IM7_TYPE,
        num_cameras=NUM_CAMERAS,
        use_camera_subfolders=True,
        use_cache=False,
    )


@pytest.mark.parametrize("camera", [1, 2])
@pytest.mark.parametrize("project", ANDRE_PROJECTS)
def test_andre_stepped_detection_matches_davis_marks(
    real_calibration_data, project, camera
):
    folder = real_calibration_data / "andre" / project / "Properties/Calibration"
    folder = folder / f"camera{camera}"
    image = _read_im7(folder, camera, 1)

    result = SteppedDetector(SteppedParams()).detect(image)

    assert result.success, result.diagnostics
    stats = _agreement(
        result.image_points, _davis_marks(folder / "MarkPositionTable.xml", camera)
    )
    print(f"\n[andre {project[-6:]} cam{camera}] {stats}")
    assert stats["n_shared"] / stats["n_davis"] >= ANDRE_MIN_SHARED_FRACTION, stats
    assert stats["median_px"] <= ANDRE_MAX_MEDIAN_PX, stats
    assert (
        stats["n_within_tol"] / stats["n_shared"] >= ANDRE_MIN_WITHIN_TOL_FRACTION
    ), stats


def _bailey_cases():
    for cam in (1, 2):
        for view in BAILEY_VIEWS:
            marks = (
                pytest.mark.xfail(
                    reason="partial board: DaVis marked <= 48 dots and the detector "
                    "finds no grid",
                    strict=False,
                )
                if (cam, view) in BAILEY_LOW_MARK_VIEWS
                else ()
            )
            yield pytest.param(cam, view, marks=marks, id=f"cam{cam}-view{view}")


@pytest.mark.parametrize("camera,view", _bailey_cases())
def test_bailey_dotboard_detection_on_every_view(real_calibration_data, camera, view):
    folder = (
        real_calibration_data / "bailey_calibration" / "Calibration" / f"camera{camera}"
    )
    image = _read_im7(folder, camera, view)

    result = DotboardDetector(DotboardParams(dot_spacing_mm=BAILEY_SPACING_MM)).detect(
        image
    )

    assert result.success, result.diagnostics.get("error")
    davis = _davis_marks(folder / "MarkPositionTable.xml", camera, view)
    stats = _agreement(result.image_points, davis)
    print(f"\n[bailey cam{camera} view{view}] {stats}")
    assert stats["n_shared"] / stats["n_davis"] >= BAILEY_MIN_SHARED_FRACTION, stats
    assert stats["median_px"] <= BAILEY_MAX_MEDIAN_PX, stats


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="the volume's Merle joint record predates the 2026-08-26 format "
    "(no pose_diversity); the strict reader refuses it. Regenerate it (into a copy, the "
    "volume is read-only), point PIVTOOLS_CAL_DATA at the copy, and drop this marker.",
)
def test_merle_joint_model_and_sidecar_load(real_calibration_data):
    model_dir = (
        real_calibration_data / "merle/Calibration/calibration/joint_dotboard/model"
    )

    joint = REC.load_joint(model_dir)
    inputs = INP.load_inputs(model_dir)

    assert joint.cameras == MERLE_CAMERAS
    assert joint.board_type == "dotboard"
    assert joint.rms_px <= MERLE_MAX_RMS_PX
    assert set(joint.models) == set(MERLE_CAMERAS)
    assert inputs.path_type == "joint"
    assert {c: len(v) for c, v in inputs.detections.items()} == {
        c: MERLE_VIEWS_PER_CAMERA for c in MERLE_CAMERAS
    }
    # The sidecar's image sizes are the models' image sizes (same cameras, same frames).
    assert inputs.image_size_by_cam == {
        c: tuple(joint.models[c].image_size) for c in MERLE_CAMERAS
    }
