"""Stereo self-calibration tests for the calibration package.

The C-extension-free tests below exercise the calibration bridge, the dewarp world
bounds, the record ``self_cal`` round-trip, and that the stereo reconstruction
honours a stored sheet — they run everywhere. The end-to-end recovery test (which
needs the ``libbulkxcorr2d`` C extension) lives in
``test_calibration_self_cal_recovery.py`` and skips cleanly when the lib is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration import self_cal as SC
from pivtools_gui.calibration.camera_model import CameraModel
from pivtools_gui.calibration.stereo_model import reconstruct_3c_field


# ---------------------------------------------------------------------------
# Synthetic stereo pair (no rendering, no C extension)
# ---------------------------------------------------------------------------

K = np.array([[2000.0, 0.0, 800.0], [0.0, 2000.0, 600.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1600, 1200)


def _Ry(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _stereo_pair(angle_deg: float = 15.0):
    """Two pinhole CameraModels viewing the Z=0 plane from +/- angle_deg about Y."""
    t = np.array([[0.0], [0.0], [600.0]])
    m1 = CameraModel(K, np.zeros(5), _Ry(+angle_deg), t.copy(), IMAGE_SIZE)
    m2 = CameraModel(K, np.zeros(5), _Ry(-angle_deg), t.copy(), IMAGE_SIZE)
    return m1, m2


# ---------------------------------------------------------------------------
# Bridge + world bounds
# ---------------------------------------------------------------------------

def test_pinhole_from_model_bridges_fields():
    """pinhole_from_model copies K, dist, R, t, image_size verbatim."""
    m1, _ = _stereo_pair()
    ph = SC.pinhole_from_model(m1)
    assert np.allclose(ph.K, m1.K)
    assert np.allclose(ph.dist, m1.dist)
    assert np.allclose(ph.R, m1.R)
    assert np.allclose(ph.t.reshape(3, 1), m1.t.reshape(3, 1))
    assert ph.image_size == m1.image_size
    # The bridged camera projects identically to the calibration model.
    pts = np.array([[0.0, 0.0, 0.0], [50.0, -30.0, 0.0]])
    assert np.allclose(ph.project(pts), m1.project(pts), atol=1e-6)


def test_stereo_world_bounds_overlap():
    """Two cameras straddling the origin yield a finite, origin-spanning overlap box."""
    m1, m2 = _stereo_pair()
    xmin, xmax, ymin, ymax = SC.stereo_world_bounds(m1, m2)
    assert xmin < 0 < xmax and ymin < 0 < ymax
    assert np.isfinite([xmin, xmax, ymin, ymax]).all()
    # Symmetric rig about Y -> symmetric X overlap about 0.
    assert xmax == pytest.approx(-xmin, rel=0.05)


def test_stereo_world_bounds_no_overlap_raises():
    """Non-overlapping FOVs raise rather than returning an inverted box."""
    t1 = np.array([[5000.0], [0.0], [600.0]])
    t2 = np.array([[-5000.0], [0.0], [600.0]])
    m1 = CameraModel(K, np.zeros(5), np.eye(3), t1, IMAGE_SIZE)
    m2 = CameraModel(K, np.zeros(5), np.eye(3), t2, IMAGE_SIZE)
    with pytest.raises(ValueError, match="no overlapping FOV"):
        SC.stereo_world_bounds(m1, m2)


# ---------------------------------------------------------------------------
# Record self_cal block round-trip
# ---------------------------------------------------------------------------

def _stereo_record(self_cal=None) -> REC.StereoRecord:
    m1, m2 = _stereo_pair()
    return REC.StereoRecord(
        cam1=1, cam2=2, board_type="dotboard",
        model1=m1, model2=m2,
        R_stereo=m2.R @ m1.R.T, T_stereo=m2.t - (m2.R @ m1.R.T) @ m1.t,
        world_frame=REC.WorldFrame(),
        per_view_rms1=[0.1], per_view_rms2=[0.12],
        board_meta={"spacing_mm": 14.0},
        self_cal=self_cal or {},
    )


def test_self_cal_block_roundtrips(tmp_path):
    """A self_cal block survives save_stereo -> load_stereo with values intact."""
    block = SC.result_to_block(
        _FakeResult(z=3.5, tx=0.012, ty=-0.008, conv=True, rms=0.07, n=4),
        n_images=20, window_size=64, overlap=50.0,
    )
    rec = _stereo_record(self_cal=block)
    path = REC.save_stereo(rec, tmp_path)
    out = REC.load_stereo(path)
    assert out.sc_z_offset == pytest.approx(3.5)
    assert out.sc_tilt_x == pytest.approx(0.012)
    assert out.sc_tilt_y == pytest.approx(-0.008)
    assert int(out.self_cal["converged"]) == 1
    assert float(out.self_cal["final_rms_disparity"]) == pytest.approx(0.07)
    assert str(out.self_cal["source"]) == "auto"


def test_absent_self_cal_defaults_to_zero(tmp_path):
    """A stereo record with no self-cal loads with an empty block and zero sheet."""
    rec = _stereo_record(self_cal=None)
    out = REC.load_stereo(REC.save_stereo(rec, tmp_path))
    assert out.self_cal == {}
    assert (out.sc_z_offset, out.sc_tilt_x, out.sc_tilt_y) == (0.0, 0.0, 0.0)


def test_manual_block_marks_source():
    block = SC.manual_block(z_offset=2.0, tilt_x=0.0, tilt_y=0.01)
    assert block["source"] == "manual"
    assert block["z_offset"] == 2.0 and block["tilt_y"] == 0.01


# ---------------------------------------------------------------------------
# The consumer honours a stored sheet (apply contract)
# ---------------------------------------------------------------------------

def test_reconstruct_honours_stored_sheet():
    """reconstruct_3c_field places cam1's grid on the (z_offset, tilt) plane.

    This is the apply contract: the stored self_cal params reach the reconstruction
    unchanged and move the world Z onto the recovered sheet.
    """
    m1, m2 = _stereo_pair()
    H = W = 8
    gx, gy = np.meshgrid(np.linspace(450, 1150, W), np.linspace(350, 850, H))
    coords1 = np.stack([gx, gy], axis=-1)
    zero = np.zeros((H, W))
    coords2 = coords1.copy()  # displacement is zero -> only the geometry matters

    z_off, tx, ty = 5.0, 0.02, 0.03
    wx, wy, wz, *_ = reconstruct_3c_field(
        m1, m2, coords1, zero, zero, coords2, zero, zero,
        dt=1.0, z_world=z_off, tilt_x=tx, tilt_y=ty,
    )
    expected = z_off + wx * np.tan(ty) + wy * np.tan(tx)
    assert np.allclose(wz, expected, atol=1e-6)

    # With no sheet correction the grid sits on Z=0, and the two differ.
    _, _, wz0, *_ = reconstruct_3c_field(
        m1, m2, coords1, zero, zero, coords2, zero, zero, dt=1.0,
    )
    assert np.allclose(wz0, 0.0, atol=1e-6)
    assert not np.allclose(wz, wz0)


# ---------------------------------------------------------------------------
# Minimal SelfCalibrationResult stand-in for result_to_block
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, *, z, tx, ty, conv, rms, n):
        self.z_offset = z
        self.tilt_x = tx
        self.tilt_y = ty
        self.converged = conv
        self.final_rms_disparity = rms
        self.n_iterations = n
