"""Stepped pinhole fits enforce fx == fy (DaVis convention, square pixels).

DaVis ``Calibration.xml`` holds ``FocalLengthPixel`` x == y on every dataset, and the
planar, stereo and joint paths already fix the aspect ratio. The stepped path used to fit
free and fixed aspect and keep the free fit when it won by 5 %; it now runs one fit with
``CALIB_FIX_ASPECT_RATIO``. These tests pin that on the synthetic stepped scenes of
``test_calibration_stepped_mono`` (multi-view and single-view branches) and
``test_calibration_stepped_stereo`` (both cameras of a same-side pair), with the RMS and
focal tolerances those tests already use.

Those scenes have square truth pixels, so the old code also kept the fixed fit there.
They pin exact equality (OpenCV holds fx / fy at the seed ratio 1.0, so fx and fy are
bit-identical) and that RMS stays in tolerance. The discriminating case renders the mono
scene through fy = 1.10 * fx. There the free fit wins by far more than 5 %, so the old code
returned fx != fy (verified: fx 1279.7 vs fy 1407.7 on the pre-change code).
"""

from __future__ import annotations

import numpy as np
import pytest
import test_calibration_stepped_mono as MONO
import test_calibration_stepped_stereo as STEREO

from pivtools_cli.synthetic_calibration_common import make_camera_matrix
from pivtools_gui.calibration.detection.stepped import SteppedDetector, SteppedParams
from pivtools_gui.calibration.detection.stepped_levels import SteppedBoardSpec
from pivtools_gui.calibration.stepped_calibrate import (
    calibrate_stepped_mono,
    calibrate_stepped_stereo,
)

# Re-export the sibling tests' module-scoped synthetic scenes as fixtures of this module.
stepped_mono_scene = MONO.stepped_mono_scene
same_side_scene = STEREO.same_side_scene

# Tolerances copied from the existing stepped tests (noise-free synthetic renders).
RMS_MAX_PX = 1.0
FX_REL_TOL_MULTI = 0.03  # test_stepped_mono_recovers_intrinsics_and_frame
FX_REL_TOL_SINGLE = 0.05  # test_stepped_mono_single_view_pinhole

# Non-square truth pixels for the discriminating case. 10 % is far past the old 5 %
# free-vs-fixed gate, so the old code would have kept the free fit (fx != fy).
ANISO_FY_OVER_FX = 1.10


def _board() -> SteppedBoardSpec:
    return SteppedBoardSpec(
        dot_spacing_mm=STEREO.SPACING_MM, step_height_mm=STEREO.STEP_MM
    )


def _mono(scene, n_views: int):
    _K_true, _images, detections, fiducials, pose_levels = scene
    return calibrate_stepped_mono(
        detections=detections[:n_views],
        fiducials=fiducials,
        clicked_level="peak",
        pose_levels=pose_levels[:n_views],
        board=_board(),
        image_size=(MONO.W, MONO.H),
        camera=1,
        datum_index=0,
    )


def test_mono_multi_view_fit_has_square_pixels(stepped_mono_scene):
    """>= 3 poses (``fit_pinhole``): fx == fy exactly, focal and RMS within tolerance."""
    K_true = stepped_mono_scene[0]
    n_views = len(stepped_mono_scene[2])
    assert n_views >= 3, "scene must exercise the multi-view Zhang branch"

    cam = _mono(stepped_mono_scene, n_views).camera_model

    assert cam.K[0, 0] == cam.K[1, 1], (cam.K[0, 0], cam.K[1, 1])
    fx_true = float(K_true[0, 0])
    assert abs(float(cam.K[0, 0]) - fx_true) / fx_true < FX_REL_TOL_MULTI
    assert cam.rms < RMS_MAX_PX, cam.rms


def test_mono_single_view_fit_has_square_pixels(stepped_mono_scene):
    """< 3 poses (``fit_pinhole_single_view``): fx == fy exactly from one plate."""
    K_true = stepped_mono_scene[0]

    cam = _mono(stepped_mono_scene, 1).camera_model

    assert cam.K[0, 0] == cam.K[1, 1], (cam.K[0, 0], cam.K[1, 1])
    fx_true = float(K_true[0, 0])
    assert abs(float(cam.K[0, 0]) - fx_true) / fx_true < FX_REL_TOL_SINGLE
    assert cam.rms < RMS_MAX_PX, cam.rms


def test_stereo_fit_has_square_pixels_on_both_cameras(same_side_scene):
    """Both cameras of a stepped stereo pair: fx == fy exactly, RMS within tolerance."""
    s = same_side_scene

    record = calibrate_stepped_stereo(
        detections1=s["detections1"],
        detections2=s["detections2"],
        fiducials1=s["fiducials1"],
        fiducials2=s["fiducials2"],
        clicked_level1="peak",
        clicked_level2="peak",
        pose_levels1=s["pose_levels1"],
        pose_levels2=s["pose_levels2"],
        board=_board(),
        image_size1=(STEREO.W, STEREO.H),
        image_size2=(STEREO.W, STEREO.H),
        cam1=1,
        cam2=2,
        datum_index=0,
        stereo_config="auto",
    )

    fx_true = float(s["K"][0, 0])
    for name, cam in (("cam1", record.model1), ("cam2", record.model2)):
        assert cam.K[0, 0] == cam.K[1, 1], (name, cam.K[0, 0], cam.K[1, 1])
        assert abs(float(cam.K[0, 0]) - fx_true) / fx_true < FX_REL_TOL_MULTI, name
        assert cam.rms < RMS_MAX_PX, (name, cam.rms)


@pytest.fixture(scope="module")
def anisotropic_mono_scene():
    """The mono stepped scene rendered through a truth K with fy = 1.10 * fx."""
    K_true = make_camera_matrix(MONO.W, MONO.H)
    K_true[1, 1] *= ANISO_FY_OVER_FX
    images, projs = [], []
    for rv in MONO.POSE_RVECS:
        img, proj = MONO._render_pose(rv, K_true)
        images.append(img)
        projs.append(proj)
    fiducials = {
        "origin": projs[0][0].tolist(),
        "x_axis": projs[0][1].tolist(),
        "y_axis": projs[0][MONO.PEAK_COLS].tolist(),
    }
    det = SteppedDetector(
        SteppedParams(dot_spacing_mm=MONO.SPACING_MM, step_height_mm=MONO.STEP_MM)
    )
    detections = [det.detect(im) for im in images]
    gt_z = np.concatenate(
        [
            np.zeros(MONO.PEAK_COLS * MONO.PEAK_ROWS),
            np.full(MONO.TROUGH_COLS * MONO.TROUGH_ROWS, -MONO.STEP_MM),
        ]
    )
    pose_levels = [MONO._level_a_label(d, p, gt_z) for d, p in zip(detections, projs)]
    return K_true, images, detections, fiducials, pose_levels


def test_square_pixels_enforced_even_when_truth_is_not_square(anisotropic_mono_scene):
    """Discriminating case: a free fit would win here by far more than 5 %, so the old
    free-vs-fixed choice returned fx != fy. The constraint must hold regardless."""
    detections = anisotropic_mono_scene[2]
    assert all(d.success for d in detections), "an anisotropic pose failed detection"

    cam = _mono(anisotropic_mono_scene, len(detections)).camera_model

    assert cam.K[0, 0] == cam.K[1, 1], (cam.K[0, 0], cam.K[1, 1])
