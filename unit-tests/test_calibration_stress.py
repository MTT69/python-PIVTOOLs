"""Ensemble Reynolds-stress calibration (calibration).

A Reynolds stress is a tensor; it transforms by the local pixel->world Jacobian as
``R_world = J R_px J^T / (dt^2 1e6)``. These C-free tests cover: reduction to the
legacy isotropic scalar; the mirror flipping the cross-term sign automatically; the
exact tensor transform for a known anisotropic Jacobian; symmetry; and the runio
``ensemble_result.mat`` round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.io

from pivtools_gui.calibration import apply as APPLY
from pivtools_gui.calibration import runio
from pivtools_gui.calibration.pipeline import build_scale_factor_record

IMAGE_SIZE = (1600, 1200)


@dataclass
class _AffineModel:
    """A model whose back-projection is a known affine map world = M @ px + c.

    Its local Jacobian is exactly M everywhere, so it pins down the tensor transform.
    """

    M: np.ndarray
    c: np.ndarray
    image_size: Tuple[int, int] = IMAGE_SIZE

    def back_project_to_plane(self, pts_px, z_world=0.0, tilt_x=0.0, tilt_y=0.0):
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        xy = pts @ self.M.T + self.c
        return np.column_stack([xy, np.zeros(len(pts))])


def test_isotropic_reduces_to_scalar():
    """Scale-factor (J = s*I) -> UU*s^2, VV*s^2, UV*s^2 (the legacy scalar)."""
    p, dt = 10.0, 5e-4
    # +X right, +Y down -> col_sign=+1, row_sign=+1 -> J = diag(mmpp, mmpp), isotropic.
    m = build_scale_factor_record(
        camera=1,
        origin_px=(0, 0),
        px_per_mm=p,
        image_size=IMAGE_SIZE,
        dt=dt,
        x_dir="right",
        y_dir="down",
    ).camera_model
    coords = np.stack(np.meshgrid([0.0, 10.0], [0.0, 20.0]), axis=-1)
    UU = np.full((2, 2), 4.0)
    VV = np.full((2, 2), 9.0)
    UV = np.full((2, 2), 2.0)
    cu, cv, cuv = APPLY.calibrate_stress_tensor(m, coords, UU, VV, UV, dt)
    s2 = (1.0 / p) ** 2 / dt**2 / 1e6
    assert np.allclose(cu, 4.0 * s2)
    assert np.allclose(cv, 9.0 * s2)
    assert np.allclose(cuv, 2.0 * s2)


def test_mirror_flips_cross_term_only():
    """+X left flips UV sign vs +X right; UU/VV unchanged (this replaces invert_ux)."""
    p, dt = 10.0, 5e-4
    coords = np.stack(np.meshgrid([0.0, 10.0], [0.0, 20.0]), axis=-1)
    UU = np.full((2, 2), 4.0)
    VV = np.full((2, 2), 9.0)
    UV = np.full((2, 2), 2.0)
    mr = build_scale_factor_record(
        camera=1,
        origin_px=(0, 0),
        px_per_mm=p,
        image_size=IMAGE_SIZE,
        dt=dt,
        x_dir="right",
        y_dir="down",
    ).camera_model
    ml = build_scale_factor_record(
        camera=1,
        origin_px=(0, 0),
        px_per_mm=p,
        image_size=IMAGE_SIZE,
        dt=dt,
        x_dir="left",
        y_dir="down",
    ).camera_model
    uuR, vvR, uvR = APPLY.calibrate_stress_tensor(mr, coords, UU, VV, UV, dt)
    uuL, vvL, uvL = APPLY.calibrate_stress_tensor(ml, coords, UU, VV, UV, dt)
    assert np.allclose(uuL, uuR) and np.allclose(vvL, vvR)
    assert np.allclose(uvL, -uvR) and not np.allclose(uvR, 0.0)


def test_anisotropic_tensor_transform_exact():
    """Known affine M -> R_world == M R_px M^T / (dt^2 1e6) exactly (rotation + shear)."""
    dt = 1e-3
    theta = np.deg2rad(30.0)
    M = np.array(
        [
            [0.02 * np.cos(theta), -0.03 * np.sin(theta)],
            [0.02 * np.sin(theta), 0.03 * np.cos(theta)],
        ]
    )  # anisotropic + rotated
    model = _AffineModel(M=M, c=np.array([5.0, -2.0]))
    coords = np.array([[100.0, 200.0], [300.0, 50.0]])
    UU = np.array([4.0, 1.0])
    VV = np.array([9.0, 16.0])
    UV = np.array([2.0, -3.0])
    cu, cv, cuv = APPLY.calibrate_stress_tensor(model, coords, UU, VV, UV, dt)
    scale = 1.0 / dt**2 / 1e6
    for k in range(2):
        R = np.array([[UU[k], UV[k]], [UV[k], VV[k]]])
        Rw = M @ R @ M.T * scale
        assert np.isclose(cu[k], Rw[0, 0])
        assert np.isclose(cv[k], Rw[1, 1])
        assert np.isclose(cuv[k], Rw[0, 1])
        assert np.isclose(Rw[0, 1], Rw[1, 0])  # symmetric


def test_local_jacobian_recovers_affine():
    M = np.array([[0.5, 0.1], [-0.2, 0.4]])
    model = _AffineModel(M=M, c=np.array([0.0, 0.0]))
    J = APPLY.local_jacobians(model, np.array([[10.0, 20.0], [0.0, 0.0]]))
    assert np.allclose(J[0], M) and np.allclose(J[1], M)


def test_runio_ensemble_roundtrip(tmp_path):
    """calibrate_ensemble_file reads ensemble_result, calibrates velocity + stresses."""
    p, dt = 10.0, 5e-4
    m = build_scale_factor_record(
        camera=1,
        origin_px=(0, 0),
        px_per_mm=p,
        image_size=IMAGE_SIZE,
        dt=dt,
        x_dir="right",
        y_dir="down",
    ).camera_model
    H = W = 2
    xg = np.array([[0.0, 10.0], [0.0, 10.0]])
    yg = np.array([[0.0, 0.0], [20.0, 20.0]])
    er = np.empty(
        (1,),
        dtype=[
            ("ux", "O"),
            ("uy", "O"),
            ("UU_stress", "O"),
            ("VV_stress", "O"),
            ("UV_stress", "O"),
        ],
    )
    er["ux"][0] = np.full((H, W), 8.0)
    er["uy"][0] = np.full((H, W), 4.0)
    er["UU_stress"][0] = np.full((H, W), 4.0)
    er["VV_stress"][0] = np.full((H, W), 9.0)
    er["UV_stress"][0] = np.full((H, W), 2.0)
    scipy.io.savemat(str(tmp_path / "ensemble_result.mat"), {"ensemble_result": er})

    runio.calibrate_ensemble_file(
        tmp_path / "ensemble_result.mat", tmp_path / "out.mat", m, {0: (xg, yg)}, dt
    )
    out = scipy.io.loadmat(str(tmp_path / "out.mat"))["ensemble_result"].reshape(-1)
    mmpp = 1.0 / p
    s2 = mmpp**2 / dt**2 / 1e6
    assert np.allclose(out["UU_stress"][0], 4.0 * s2)
    assert np.allclose(out["VV_stress"][0], 9.0 * s2)
    assert np.allclose(out["UV_stress"][0], 2.0 * s2)  # +Y down -> no sign flip
    assert np.allclose(out["ux"][0], 8.0 * mmpp / dt / 1000.0)


def test_ensemble_shape_mismatch_raises(tmp_path):
    """A grid that does not match coordinates.mat must FAIL, not write raw px through.

    The old behaviour logged a warning and left the field uncalibrated, so the output
    silently mixed m/s and px/frame. Calibration2 now refuses rather than emit a
    plausible-but-wrong field.
    """
    import pytest

    dt = 5e-4
    m = build_scale_factor_record(
        camera=1,
        origin_px=(0, 0),
        px_per_mm=10.0,
        image_size=IMAGE_SIZE,
        dt=dt,
        x_dir="right",
        y_dir="down",
    ).camera_model
    # coords are 2x2; the ensemble velocity field is 3x3 -> mismatch.
    xg = np.array([[0.0, 10.0], [0.0, 10.0]])
    yg = np.array([[0.0, 0.0], [20.0, 20.0]])
    er = np.empty((1,), dtype=[("ux", "O"), ("uy", "O")])
    er["ux"][0] = np.full((3, 3), 8.0)
    er["uy"][0] = np.full((3, 3), 4.0)
    scipy.io.savemat(str(tmp_path / "ensemble_result.mat"), {"ensemble_result": er})
    with pytest.raises(ValueError, match="does not match coordinates"):
        runio.calibrate_ensemble_file(
            tmp_path / "ensemble_result.mat", tmp_path / "out.mat", m, {0: (xg, yg)}, dt
        )


def test_resolve_dt_precedence_and_no_silent_default():
    """dt resolution: explicit > model-stamped, and RAISE if neither is given.

    Pins B1's per-camera intent (each model's own stamped dt wins) and the no-silent-1.0
    rule — velocity scales with dt, so an unresolved dt must fail, not default. There is
    no config source: every generated record stamps dt, so an unstamped record is stale.
    """
    import pytest

    assert runio.resolve_dt(2.0, 0.5) == 2.0  # explicit wins
    assert runio.resolve_dt(None, 0.5) == 0.5  # model-stamped next
    # Two cameras with different stamped dt each resolve to THEIR OWN value (not cam1's).
    rec_a = build_scale_factor_record(
        camera=1, origin_px=(0, 0), px_per_mm=10.0, image_size=IMAGE_SIZE, dt=0.5
    ).board_meta["dt"]
    rec_b = build_scale_factor_record(
        camera=2, origin_px=(0, 0), px_per_mm=10.0, image_size=IMAGE_SIZE, dt=2.0
    ).board_meta["dt"]
    assert runio.resolve_dt(None, rec_a) == 0.5
    assert runio.resolve_dt(None, rec_b) == 2.0
    with pytest.raises(ValueError, match="re-generate the model"):
        runio.resolve_dt(None, None)
