"""Regular-world-grid stereo reconstruction tests.

Covers ``regular_world_grid`` (grid regularity, spacing = median vector pitch,
overlap failures) and the symmetric ``reconstruct_3c_field`` (uniform-flow
ground truth, bmask propagation from BOTH cameras). Pure synthetic pinhole pairs —
no rendering, no C extension.
"""

from __future__ import annotations

import numpy as np
import pytest

from pivtools_gui.calibration.camera_model import CameraModel
from pivtools_gui.calibration.stereo_model import (
    reconstruct_3c_field,
    regular_world_grid,
)


# ---------------------------------------------------------------------------
# Synthetic stereo pair (matches test_calibration_self_cal.py's factory)
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


def _piv_coords(nx=24, ny=18, x0=200.0, x1=1400.0, y0=150.0, y1=1050.0):
    """(H,W,2) image-down pixel meshgrid standing in for a camera's PIV grid."""
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    return np.stack([gx, gy], axis=-1)


def _uniform_disp(model: CameraModel, coords: np.ndarray, U_mm) -> tuple:
    """Per-camera pixel displacement fields for a uniform world displacement U_mm."""
    flat = coords.reshape(-1, 2)
    world = model.back_project_to_plane(flat, 0.0, 0.0, 0.0)
    disp = model.project(world + np.asarray(U_mm, dtype=np.float64)) - flat
    H, W = coords.shape[:2]
    return disp[:, 0].reshape(H, W), disp[:, 1].reshape(H, W)


# ---------------------------------------------------------------------------
# regular_world_grid
# ---------------------------------------------------------------------------

def test_regular_grid_is_regular():
    """Constant spacing along both axes, snapped axes, Z on the sheet plane."""
    m1, m2 = _stereo_pair()
    coords = _piv_coords()
    z_off, tx, ty = 3.0, 0.01, -0.02
    gX, gY, gZ, sp = regular_world_grid(m1, m2, coords, coords.copy(), z_off, tx, ty)

    assert np.allclose(np.diff(gX, axis=1), sp, atol=1e-9)
    assert np.allclose(np.diff(gY, axis=0), sp, atol=1e-9)
    assert np.allclose(gX, gX[:1, :])          # rows identical
    assert np.allclose(gY, gY[:, :1])          # cols identical
    # Axes snapped to spacing multiples.
    assert np.allclose(np.round(gX[0, :] / sp) * sp, gX[0, :], atol=1e-9)
    assert np.allclose(np.round(gY[:, 0] / sp) * sp, gY[:, 0], atol=1e-9)
    assert np.allclose(gZ, z_off + gX * np.tan(ty) + gY * np.tan(tx), atol=1e-12)


def test_auto_spacing_is_median_pitch():
    """Auto spacing tracks the analytic magnification of the symmetric rig.

    At Z=600 with f=2000 the in-plane magnification is ~600/2000 = 0.3 mm/px
    (cos-angle corrections are small at 15 deg), so the auto spacing should be
    close to pixel_step * 0.3.
    """
    m1, m2 = _stereo_pair()
    coords = _piv_coords()
    step_px = float(np.median(np.diff(coords[0, :, 0])))
    _, _, _, sp = regular_world_grid(m1, m2, coords, coords.copy())
    assert sp == pytest.approx(step_px * 600.0 / 2000.0, rel=0.05)


def test_no_overlap_raises():
    """Cameras with disjoint world footprints raise, not an inverted/empty grid."""
    t1 = np.array([[5000.0], [0.0], [600.0]])
    t2 = np.array([[-5000.0], [0.0], [600.0]])
    m1 = CameraModel(K, np.zeros(5), np.eye(3), t1, IMAGE_SIZE)
    m2 = CameraModel(K, np.zeros(5), np.eye(3), t2, IMAGE_SIZE)
    coords = _piv_coords()
    with pytest.raises(ValueError, match="overlap"):
        regular_world_grid(m1, m2, coords, coords.copy())


def test_degenerate_grid_raises():
    """An overlap narrower than ~2 vector pitches raises rather than emitting < 2x2.

    Straight-down cameras offset so their world footprints overlap by only ~10 mm in
    x, while the vector pitch is ~16 mm — the snapped grid would be a single column.
    """
    t1 = np.array([[175.0], [0.0], [600.0]])
    t2 = np.array([[-175.0], [0.0], [600.0]])
    m1 = CameraModel(K, np.zeros(5), np.eye(3), t1, IMAGE_SIZE)
    m2 = CameraModel(K, np.zeros(5), np.eye(3), t2, IMAGE_SIZE)
    coords = _piv_coords()
    with pytest.raises(ValueError, match="too small"):
        regular_world_grid(m1, m2, coords, coords.copy())


# ---------------------------------------------------------------------------
# reconstruct_3c_field on the regular grid
# ---------------------------------------------------------------------------

def test_uniform_flow_ground_truth():
    """A uniform 3C world displacement reconstructs exactly on unmasked points."""
    m1, m2 = _stereo_pair()
    coords1 = _piv_coords()
    coords2 = _piv_coords()
    U_mm = np.array([0.4, -0.3, 0.25])  # mm/frame

    ux1, uy1 = _uniform_disp(m1, coords1, U_mm)
    ux2, uy2 = _uniform_disp(m2, coords2, U_mm)

    grid = regular_world_grid(m1, m2, coords1, coords2)[:3]
    U, V, W, mask = reconstruct_3c_field(
        m1, m2, grid, coords1, ux1, uy1, coords2, ux2, uy2,
        dt=1.0, interpolator="linear",
    )
    assert not mask.all()
    assert np.isfinite(U[~mask]).all()
    # dt=1 -> m/s = mm/frame / 1000
    assert np.allclose(U[~mask], U_mm[0] / 1000.0, rtol=1e-3)
    assert np.allclose(V[~mask], U_mm[1] / 1000.0, rtol=1e-3)
    assert np.allclose(W[~mask], U_mm[2] / 1000.0, rtol=1e-3)
    # Every non-finite point is masked (the overlap-trim contract).
    assert mask[~np.isfinite(U) | ~np.isfinite(V) | ~np.isfinite(W)].all()


def test_bmasks_propagate_symmetrically():
    """A masked block in either camera masks the grid points that project into it.

    Regression for the previously asymmetric path: bmask1 used to be applied on
    cam1's grid directly; now both masks travel through their camera's projection.
    """
    m1, m2 = _stereo_pair()
    coords1 = _piv_coords()
    coords2 = _piv_coords()
    H, W = coords1.shape[:2]
    zero = np.zeros((H, W))

    gX, gY, gZ, _ = regular_world_grid(m1, m2, coords1, coords2)
    world = np.stack([gX.ravel(), gY.ravel(), gZ.ravel()], axis=1)

    for cam_idx, (model, coords) in enumerate(((m1, coords1), (m2, coords2))):
        bmask = np.zeros((H, W), dtype=bool)
        bmask[4:9, 6:12] = True  # interior block of masked vectors

        kwargs = {"bmask1": bmask} if cam_idx == 0 else {"bmask2": bmask}
        _, _, _, mask = reconstruct_3c_field(
            m1, m2, (gX, gY, gZ), coords1, zero, zero, coords2, zero, zero,
            dt=1.0, interpolator="linear", **kwargs,
        )
        _, _, _, mask_ref = reconstruct_3c_field(
            m1, m2, (gX, gY, gZ), coords1, zero, zero, coords2, zero, zero,
            dt=1.0, interpolator="linear",
        )

        # Grid points whose projection lands well inside the masked pixel block
        # must be masked; points far from it must be unaffected.
        proj = model.project(world)
        xs = coords[0, :, 0]
        ys = coords[:, 0, 1]
        in_block = (
            (proj[:, 0] > xs[6]) & (proj[:, 0] < xs[11])
            & (proj[:, 1] > ys[4]) & (proj[:, 1] < ys[8])
        ).reshape(mask.shape)
        assert in_block.any(), "test block must be visible on the grid"
        assert mask[in_block].all()
        # The mask only ever grows relative to the no-bmask reference, and points
        # already unmasked far from the block stay unmasked.
        assert (mask | mask_ref).sum() == mask.sum()
        far = ~in_block & ~mask_ref
        # Erode 'far' by excluding anything near the block through either camera:
        # bilinear resampling bleeds one PIV cell beyond the block edge.
        proj_own = model.project(world)
        near_block = (
            (proj_own[:, 0] > xs[5]) & (proj_own[:, 0] < xs[12])
            & (proj_own[:, 1] > ys[3]) & (proj_own[:, 1] < ys[9])
        ).reshape(mask.shape)
        assert not mask[far & ~near_block].any()
