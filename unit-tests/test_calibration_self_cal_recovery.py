"""End-to-end stereo self-calibration recovery for calibration.

Builds a synthetic stereo pair, renders particle images on a laser sheet with a
KNOWN ``(z_offset, tilt_x, tilt_y)``, then drives the calibration ``self_cal.run``
entry and asserts the recovered sheet matches. This also verifies the frame-
consistency claim in ``self_cal`` — the ``PinholeCamera`` is built from the stored
``CameraModel`` and the recovered params land in the calibration world frame.

Needs the ``libbulkxcorr2d`` C extension (the ensemble correlation routine); the
whole module skips cleanly when it is not built.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import self_cal as SC
from pivtools_gui.calibration.camera_model import CameraModel

# Skip the whole module if the C correlation extension is unavailable.
try:
    from pivtools_gui.stereo_reconstruction.self_calibration import _load_xcorr_library
    _load_xcorr_library()
except Exception as e:  # pragma: no cover - environment dependent
    pytest.skip(f"libbulkxcorr2d C extension unavailable: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Known sheet + synthetic geometry (mirrors test_self_calibration_recovery.py)
# ---------------------------------------------------------------------------

TRUE_Z_OFFSET = 0.3       # mm
TRUE_TILT_X = 0.002       # rad
TRUE_TILT_Y = -0.001      # rad
STEREO_ANGLE_DEG = 30.0
N_IMAGE_PAIRS = 20
N_PARTICLES = 2000
WORLD_BOUNDS = (-40.0, 40.0, -40.0, 40.0)  # mm — pin the dewarp to the seeded region
WINDOW_SIZE = 64
OVERLAP = 50.0
FOCAL_LENGTH_PX = 1000.0
IMAGE_SIZE = (1024, 1024)
BASELINE_MM = 200.0
PARTICLE_SIGMA = 2.0
PARTICLE_INTENSITY = 200


def _stereo_models():
    """A symmetric calibration CameraModel pair looking at the origin."""
    w, h = IMAGE_SIZE
    theta = math.radians(STEREO_ANGLE_DEG / 2.0)
    K = np.array([[FOCAL_LENGTH_PX, 0.0, w / 2.0],
                  [0.0, FOCAL_LENGTH_PX, h / 2.0],
                  [0.0, 0.0, 1.0]])
    z_cam = BASELINE_MM / (2.0 * math.tan(theta))
    pos1 = np.array([BASELINE_MM / 2.0, 0.0, z_cam])
    pos2 = np.array([-BASELINE_MM / 2.0, 0.0, z_cam])

    def look_at(cam_pos, target=np.zeros(3)):
        z_axis = target - cam_pos
        z_axis /= np.linalg.norm(z_axis)
        up = np.array([0.0, -1.0, 0.0])
        x_axis = np.cross(z_axis, up)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=0)

    R1, R2 = look_at(pos1), look_at(pos2)
    t1 = (-R1 @ pos1).reshape(3, 1)
    t2 = (-R2 @ pos2).reshape(3, 1)
    m1 = CameraModel(K.copy(), np.zeros(5), R1, t1, IMAGE_SIZE)
    m2 = CameraModel(K.copy(), np.zeros(5), R2, t2, IMAGE_SIZE)
    return m1, m2


def _render(particles, model):
    w, h = model.image_size
    image = np.zeros((h, w), dtype=np.float32)
    pts2d = model.project(particles)
    half = int(math.ceil(4 * PARTICLE_SIGMA))
    kx = np.arange(2 * half + 1) - half
    kxx, kyy = np.meshgrid(kx, kx)
    for px, py in pts2d:
        ix, iy = int(round(px)), int(round(py))
        if ix < half or ix >= w - half or iy < half or iy >= h - half:
            continue
        dx, dy = px - ix, py - iy
        kernel = PARTICLE_INTENSITY * np.exp(
            -((kxx - dx) ** 2 + (kyy - dy) ** 2) / (2 * PARTICLE_SIGMA ** 2)
        )
        image[iy - half:iy + half + 1, ix - half:ix + half + 1] += kernel
    noise = np.random.normal(0, 5, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def _stereo_record(m1, m2) -> REC.StereoRecord:
    return REC.StereoRecord(
        cam1=1, cam2=2, board_type="dotboard", model1=m1, model2=m2,
        R_stereo=m2.R @ m1.R.T, T_stereo=m2.t - (m2.R @ m1.R.T) @ m1.t,
        world_frame=REC.WorldFrame(), per_view_rms1=[0.1], per_view_rms2=[0.1],
        board_meta={"spacing_mm": 14.0},
    )


@pytest.fixture(scope="module")
def recovery():
    np.random.seed(42)
    m1, m2 = _stereo_models()
    imgs1, imgs2 = [], []
    for _ in range(N_IMAGE_PAIRS):
        x = np.random.uniform(WORLD_BOUNDS[0], WORLD_BOUNDS[1], N_PARTICLES)
        y = np.random.uniform(WORLD_BOUNDS[2], WORLD_BOUNDS[3], N_PARTICLES)
        z = TRUE_Z_OFFSET + x * np.tan(TRUE_TILT_Y) + y * np.tan(TRUE_TILT_X)
        particles = np.column_stack([x, y, z])
        imgs1.append(_render(particles, m1))
        imgs2.append(_render(particles, m2))
    record = _stereo_record(m1, m2)
    result = SC.run(
        record, imgs1, imgs2,
        window_size=WINDOW_SIZE, overlap=OVERLAP,
        world_bounds=WORLD_BOUNDS,
    )
    return record, result, imgs1, imgs2


def test_z_offset_recovered(recovery):
    _, result, *_ = recovery
    assert abs(result.z_offset - TRUE_Z_OFFSET) < 0.05, result.z_offset


def test_tilt_recovered(recovery):
    _, result, *_ = recovery
    assert abs(result.tilt_x - TRUE_TILT_X) < 0.001, result.tilt_x
    assert abs(result.tilt_y - TRUE_TILT_Y) < 0.001, result.tilt_y


def test_recovered_sheet_bakes_into_record(tmp_path, recovery):
    """The recovered sheet bakes into the extrinsics and the block reloads as baked."""
    record, result, *_ = recovery
    SC.rebake_record(record, result.z_offset, result.tilt_x, result.tilt_y)
    record.self_cal = SC.baked_block(
        result, n_images=N_IMAGE_PAIRS, window_size=WINDOW_SIZE, overlap=OVERLAP,
    )
    out = REC.load_stereo(REC.save_stereo(record, tmp_path))
    # applied sheet zeroed (it lives in the poses); recovered sheet in fitted_*
    assert (out.sc_z_offset, out.sc_tilt_x, out.sc_tilt_y) == (0.0, 0.0, 0.0)
    assert abs(float(out.self_cal["fitted_z_offset"]) - TRUE_Z_OFFSET) < 0.05
    assert abs(float(out.self_cal["fitted_tilt_x"]) - TRUE_TILT_X) < 0.001
    assert int(out.self_cal["baked"]) == 1
