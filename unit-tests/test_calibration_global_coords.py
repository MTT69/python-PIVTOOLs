"""Unit tests for calibration.global_coords.compute_camera_shifts (datum chain)."""

import numpy as np
import pytest

from pivtools_gui.calibration.camera_model import CameraModel, DistortionModel
from pivtools_gui.calibration.global_coords import compute_camera_shifts
from pivtools_gui.calibration.record import MonoRecord


def _model(D: float = 100.0) -> CameraModel:
    """A trivial pinhole looking straight at the z=0 plane from distance D."""
    f, cx, cy = 1000.0, 512.0, 512.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    return CameraModel(
        K=K,
        dist=np.zeros(5),
        R=np.eye(3),
        t=np.array([[0.0], [0.0], [D]]),
        image_size=(1024, 1024),
        distortion_model=DistortionModel.STANDARD,
        rms=0.1,
    )


def _rec(cam: int, D: float = 100.0) -> MonoRecord:
    return MonoRecord(camera=cam, board_type="dotboard", camera_model=_model(D))


def test_datum_shift_places_origin():
    """The datum shift maps the datum pixel's physical to the requested origin."""
    recs = {1: _rec(1)}
    shifts = compute_camera_shifts(recs, 1, [600.0, 400.0], [0.0, 0.0], [])
    # back-projected datum physical = ((u-cx)*D/f, (v-cy)*D/f) = (8.8, -11.2) mm
    np.testing.assert_allclose(shifts[1], (-8.8, 11.2), atol=1e-6)


def test_chain_consistency_identical_views():
    """Same model + same physical feature on both sides -> identical shift (link adds nothing)."""
    recs = {1: _rec(1), 2: _rec(2)}
    P = [600.0, 400.0]
    pairs = [{"camera_a": 1, "camera_b": 2, "pixel_on_a": P, "pixel_on_b": P}]
    shifts = compute_camera_shifts(recs, 1, [512.0, 512.0], [0.0, 0.0], pairs)
    assert set(shifts) == {1, 2}
    np.testing.assert_allclose(shifts[1], shifts[2], atol=1e-9)


def test_broken_chain_link_is_skipped():
    """A pair whose cam_a is not yet placed is skipped (no crash, cam_b absent)."""
    recs = {1: _rec(1), 2: _rec(2), 3: _rec(3)}
    # pair references cam 2 (not placed from datum 1) as the source -> skipped
    pairs = [
        {
            "camera_a": 2,
            "camera_b": 3,
            "pixel_on_a": [512, 512],
            "pixel_on_b": [512, 512],
        }
    ]
    shifts = compute_camera_shifts(recs, 1, [512.0, 512.0], [0.0, 0.0], pairs)
    assert set(shifts) == {1}


def test_out_of_order_multi_hop_chain():
    """Pairs given out of chain order still resolve (fixpoint): 1 -> 3 -> 4."""
    recs = {1: _rec(1), 3: _rec(3), 4: _rec(4)}
    P = [512.0, 512.0]
    pairs = [
        {
            "camera_a": 3,
            "camera_b": 4,
            "pixel_on_a": P,
            "pixel_on_b": P,
        },  # before its parent
        {"camera_a": 1, "camera_b": 3, "pixel_on_a": P, "pixel_on_b": P},
    ]
    shifts = compute_camera_shifts(recs, 1, P, [0.0, 0.0], pairs)
    assert set(shifts) == {1, 3, 4}
    np.testing.assert_allclose(shifts[1], shifts[3], atol=1e-9)
    np.testing.assert_allclose(shifts[3], shifts[4], atol=1e-9)


def test_missing_model_raises():
    with pytest.raises(KeyError):
        compute_camera_shifts({2: _rec(2)}, 1, [512.0, 512.0], [0.0, 0.0], [])
