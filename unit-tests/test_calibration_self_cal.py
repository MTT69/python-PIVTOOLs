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
from pivtools_gui.calibration.stereo_model import (
    reconstruct_3c_field,
    regular_world_grid,
)


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


def test_baked_block_roundtrips(tmp_path):
    """A baked self_cal block survives save_stereo -> load_stereo.

    The applied sheet is zero (it lives in the extrinsics); the recovered sheet is in
    fitted_* and baked=1 marks the record.
    """
    block = SC.baked_block(
        _FakeResult(z=3.5, tx=0.012, ty=-0.008, conv=True, rms=0.07, n=4),
        n_images=20, window_size=64, overlap=50.0,
    )
    rec = _stereo_record(self_cal=block)
    path = REC.save_stereo(rec, tmp_path)
    out = REC.load_stereo(path)
    # applied sheet zeroed -> reconstruction applies nothing further
    assert (out.sc_z_offset, out.sc_tilt_x, out.sc_tilt_y) == (0.0, 0.0, 0.0)
    # fitted provenance preserved
    assert float(out.self_cal["fitted_z_offset"]) == pytest.approx(3.5)
    assert float(out.self_cal["fitted_tilt_x"]) == pytest.approx(0.012)
    assert float(out.self_cal["fitted_tilt_y"]) == pytest.approx(-0.008)
    assert int(out.self_cal["baked"]) == 1
    assert int(out.self_cal["converged"]) == 1
    assert float(out.self_cal["final_rms_disparity"]) == pytest.approx(0.07)


def test_absent_self_cal_defaults_to_zero(tmp_path):
    """A stereo record with no self-cal loads with an empty block and zero sheet."""
    rec = _stereo_record(self_cal=None)
    out = REC.load_stereo(REC.save_stereo(rec, tmp_path))
    assert out.self_cal == {}
    assert (out.sc_z_offset, out.sc_tilt_x, out.sc_tilt_y) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Rebake: world-frame redefinition baked into the extrinsics (plan item 17)
# ---------------------------------------------------------------------------

def test_rebake_record_invariant_and_updates_stereo():
    """rebake_record changes both poses but keeps (R_stereo, T_stereo) invariant.

    The stored R_stereo/T_stereo are recomputed so they stay consistent with the
    rebaked models.
    """
    from pivtools_gui.calibration.stereo_model import compose_stereo

    rec = _stereo_record()
    Rs0, Ts0 = compose_stereo(rec.model1, rec.model2)
    R1_before = rec.model1.R.copy()

    SC.rebake_record(rec, 3.0, 0.011, -0.006)

    assert not np.allclose(rec.model1.R, R1_before)        # poses moved
    Rs1, Ts1 = compose_stereo(rec.model1, rec.model2)
    assert np.allclose(Rs0, Rs1, atol=1e-12)               # cross-camera pose invariant
    assert np.allclose(Ts0, Ts1, atol=1e-9)
    assert np.allclose(rec.R_stereo, Rs1, atol=1e-12)      # stored values updated
    assert np.allclose(rec.T_stereo, Ts1, atol=1e-9)


def test_rebake_record_rejects_polynomial():
    """A model without extrinsics raises rather than silently skipping the rebake."""
    import types

    rec = _stereo_record()
    rec.model1 = types.SimpleNamespace()  # no R/t
    with pytest.raises(ValueError, match="requires pinhole"):
        SC.rebake_record(rec, 1.0, 0.0, 0.0)


def test_baked_record_reconstruction_equivalence():
    """The rebaked record's grid (Z=0) is the original's sheet grid, frame-redefined.

    The regular output grid is built per record, so the two grids are different
    lattices — pointwise equality no longer applies. What rebaking must preserve is
    the physics: mapping the rebaked grid through g_corr lands it (a) on the original
    sheet plane, (b) with the same spacing (a rigid frame redefinition cannot change
    magnification), and (c) covering the same imaged region.
    """
    from pivtools_gui.calibration.self_cal_frame import plane_to_world_correction

    z, tx, ty = 4.0, 0.015, -0.01
    H = W = 6
    gx, gy = np.meshgrid(np.linspace(450, 1150, W), np.linspace(350, 850, H))
    coords1 = np.stack([gx, gy], axis=-1)
    coords2 = coords1.copy()

    old = _stereo_record()
    gX_o, gY_o, gZ_o, sp_o = regular_world_grid(
        old.model1, old.model2, coords1, coords2, z, tx, ty)

    new = _stereo_record()
    SC.rebake_record(new, z, tx, ty)
    gX_n, gY_n, gZ_n, sp_n = regular_world_grid(
        new.model1, new.model2, coords1, coords2)

    R_corr, t_corr = plane_to_world_correction(z, tx, ty)
    world_new = np.stack([gX_n.ravel(), gY_n.ravel(), gZ_n.ravel()], axis=1)
    mapped = (R_corr @ world_new.T).T + t_corr

    # (a) mapped rebaked grid lies on the original sheet plane
    mx, my, mz = mapped[:, 0], mapped[:, 1], mapped[:, 2]
    assert np.allclose(mz, z + mx * np.tan(ty) + my * np.tan(tx), atol=1e-6)
    # (b) spacing preserved by the rigid frame redefinition
    assert sp_n == pytest.approx(sp_o, rel=1e-3)
    # (c) same imaged region: bounding boxes agree within one grid spacing
    for m_axis, o_grid in ((mx, gX_o), (my, gY_o)):
        assert m_axis.min() == pytest.approx(o_grid.min(), abs=sp_o)
        assert m_axis.max() == pytest.approx(o_grid.max(), abs=sp_o)


# ---------------------------------------------------------------------------
# The consumer honours a stored sheet (apply contract)
# ---------------------------------------------------------------------------

def test_reconstruct_honours_stored_sheet():
    """regular_world_grid places the output grid on the (z_offset, tilt) plane.

    This is the apply contract: the stored self_cal params reach the grid builder
    unchanged and move the world Z onto the recovered sheet; the reconstruction then
    solves at those points (zero displacement -> zero velocity).
    """
    m1, m2 = _stereo_pair()
    H = W = 8
    gx, gy = np.meshgrid(np.linspace(450, 1150, W), np.linspace(350, 850, H))
    coords1 = np.stack([gx, gy], axis=-1)
    zero = np.zeros((H, W))
    coords2 = coords1.copy()  # displacement is zero -> only the geometry matters

    z_off, tx, ty = 5.0, 0.02, 0.03
    gX, gY, gZ, _sp = regular_world_grid(m1, m2, coords1, coords2, z_off, tx, ty)
    expected = z_off + gX * np.tan(ty) + gY * np.tan(tx)
    assert np.allclose(gZ, expected, atol=1e-6)

    # The grid points really are on the plane the models see: projecting into a
    # camera and back-projecting onto the same sheet is the identity.
    world = np.stack([gX.ravel(), gY.ravel(), gZ.ravel()], axis=1)
    roundtrip = m1.back_project_to_plane(m1.project(world), z_off, tx, ty)
    assert np.allclose(roundtrip, world, atol=1e-6)

    # Zero displacements reconstruct to zero velocity on the unmasked points.
    U, V, W3, mask = reconstruct_3c_field(
        m1, m2, (gX, gY, gZ), coords1, zero, zero, coords2, zero, zero, dt=1.0,
    )
    assert not mask.all()
    assert np.allclose(U[~mask], 0.0, atol=1e-9)
    assert np.allclose(V[~mask], 0.0, atol=1e-9)
    assert np.allclose(W3[~mask], 0.0, atol=1e-9)

    # With no sheet correction the grid sits on Z=0; the sheet grid does not.
    _, _, gZ0, _ = regular_world_grid(m1, m2, coords1, coords2)
    assert np.allclose(gZ0, 0.0, atol=1e-6)
    assert not np.allclose(gZ, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Minimal SelfCalibrationResult stand-in for baked_block
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, *, z, tx, ty, conv, rms, n):
        self.z_offset = z
        self.tilt_x = tx
        self.tilt_y = ty
        self.converged = conv
        self.final_rms_disparity = rms
        self.n_iterations = n
