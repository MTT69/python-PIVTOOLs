"""S1·Phase 3 — unified JointRecord storage + load_camera_model resolver."""

from __future__ import annotations

import numpy as np
import pytest

from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration.camera_model import CameraModel, PolynomialModel


def _cam(fx, cx, cy, tz):
    return CameraModel(
        K=np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]]),
        dist=np.array([-0.01, 0.02, 0.001, -0.0005, 0.0]),
        R=np.eye(3),
        t=np.array([[1.0], [2.0], [tz]]),
        image_size=(1280, 1024),
    )


def _joint_record():
    board = {
        (0, 0): np.array([0.0, 0.0, 0.0]),
        (1, 0): np.array([15.0, 0.0, 0.1]),
        (0, 1): np.array([0.0, 15.0, -0.2]),
        (2, 3): np.array([30.0, 45.0, 0.4]),
    }
    return REC.JointRecord(
        cameras=[1, 2],
        board_type="dotboard",
        models={1: _cam(1200, 640, 512, 700), 2: _cam(1250, 600, 500, 720)},
        board=board,
        world_frame=REC.WorldFrame(mode="global_grid", origin_mm=np.array([5.0, -3.0])),
        spacing_mm=15.0,
        board_release="full3d",
        per_camera_rms={1: 0.18, 2: 0.21},
        rms_px=0.20,
        board_meta={"n_union": 4, "converged": 1},
    )


def test_joint_record_roundtrip(tmp_path):
    rec = _joint_record()
    p = REC.save_joint(rec, tmp_path)
    r2 = REC.load_joint(p)
    assert r2.cameras == [1, 2]
    assert r2.board_release == "full3d"
    assert r2.spacing_mm == pytest.approx(15.0)
    for c in (1, 2):
        np.testing.assert_allclose(r2.models[c].K, rec.models[c].K)
        np.testing.assert_allclose(r2.models[c].dist, rec.models[c].dist)
        np.testing.assert_allclose(r2.models[c].t, rec.models[c].t)
        assert r2.per_camera_rms[c] == pytest.approx(rec.per_camera_rms[c])
    for k, v in rec.board.items():
        np.testing.assert_allclose(r2.board[k], v)
    np.testing.assert_allclose(r2.world_frame.origin_mm, [5.0, -3.0])


def test_load_camera_model_prefers_joint(tmp_path):
    """With a joint record present, the resolver returns its per-camera view."""
    root = tmp_path
    mono_dir = root / "Cam1" / "dotboard_planar" / "model"
    mono_dir.mkdir(parents=True)
    joint_dir = root / "joint_dotboard" / "model"
    REC.save_joint(_joint_record(), joint_dir)

    cam, wf = REC.load_camera_model(mono_dir, 1)
    np.testing.assert_allclose(cam.K, _cam(1200, 640, 512, 700).K)
    assert wf.mode == "global_grid"


def test_load_camera_model_falls_back_to_mono(tmp_path):
    """With no joint record, the resolver loads the legacy per-camera mono record."""
    root = tmp_path
    mono_dir = root / "Cam1" / "dotboard_planar" / "model"
    mono_dir.mkdir(parents=True)
    mono = REC.MonoRecord(
        camera=1,
        board_type="dotboard",
        camera_model=_cam(999, 640, 512, 700),
        world_frame=REC.WorldFrame(mode="clicks"),
    )
    REC.save_mono(mono, mono_dir)

    cam, wf = REC.load_camera_model(mono_dir, 1)
    assert cam.K[0, 0] == pytest.approx(999)
    assert wf.mode == "clicks"


def _poly():
    return PolynomialModel(
        coeffs_x=np.arange(10, dtype=float),
        coeffs_y=np.arange(10, 20, dtype=float),
        x0=640.0,
        sx=640.0,
        y0=512.0,
        sy=512.0,
        image_size=(1280, 1024),
        rms_x_mm=0.05,
        rms_y_mm=0.04,
    )


def test_mono_record_for_camera_pinhole_joint_never_shadows_polynomial(tmp_path):
    """A pinhole joint record must NOT shadow a per-camera polynomial (different model type)."""
    root = tmp_path
    mono_dir = root / "Cam1" / "dotboard_planar" / "model"
    mono_dir.mkdir(parents=True)
    REC.save_mono(
        REC.MonoRecord(
            camera=1,
            board_type="dotboard",
            camera_model=_poly(),
            world_frame=REC.WorldFrame(mode="global_grid"),
        ),
        mono_dir,
    )
    REC.save_joint(_joint_record(), root / "joint_dotboard" / "model")

    # Explicit polynomial request: the joint pinhole must be ignored.
    poly = REC.mono_record_for_camera(mono_dir, 1, model_type="polynomial")
    assert isinstance(poly.camera_model, PolynomialModel)
    # Explicit pinhole request: the joint record is preferred.
    pin = REC.mono_record_for_camera(mono_dir, 1, model_type="pinhole")
    assert isinstance(pin.camera_model, CameraModel)
    np.testing.assert_allclose(pin.camera_model.K, _cam(1200, 640, 512, 700).K)
    # Unspecified request with both present is ambiguous -> raise, never silently pick pinhole.
    with pytest.raises(ValueError, match="both exist"):
        REC.mono_record_for_camera(mono_dir, 1, model_type=None)


def test_load_camera_model_joint_and_mono_agree(tmp_path):
    """The resolver returns the same model whether it comes from joint or legacy storage."""
    root = tmp_path
    mono_dir = root / "Cam2" / "dotboard_planar" / "model"
    mono_dir.mkdir(parents=True)
    rec = _joint_record()
    # legacy mono for cam2
    REC.save_mono(
        REC.MonoRecord(
            camera=2,
            board_type="dotboard",
            camera_model=rec.models[2],
            world_frame=rec.world_frame,
        ),
        mono_dir,
    )
    cam_mono, _ = REC.load_camera_model(mono_dir, 2)
    # now add the joint record (should take precedence and match)
    REC.save_joint(rec, root / "joint_dotboard" / "model")
    cam_joint, _ = REC.load_camera_model(mono_dir, 2)
    np.testing.assert_allclose(cam_joint.K, cam_mono.K)
    np.testing.assert_allclose(cam_joint.t, cam_mono.t)
