"""Global-frame offset baked into the model (calibration2).

The per-camera placement into a shared multi-camera rig frame is stored as
``WorldFrame.world_offset_mm`` and added to calibrated coordinates at apply time
(velocities are offset-invariant). These C-free tests cover the round-trip, the
apply-path semantics, and a two-camera end-to-end stitch.
"""

from __future__ import annotations

import numpy as np

from pivtools_gui.calibration2 import apply as APPLY
from pivtools_gui.calibration2 import global_coords as GC
from pivtools_gui.calibration2 import record as REC
from pivtools_gui.calibration2.pipeline import build_scale_factor_record

IMAGE_SIZE = (1600, 1200)


def _sf(camera, origin, px_per_mm=1.0, x="right", y="up"):
    return build_scale_factor_record(
        camera=camera, origin_px=origin, px_per_mm=px_per_mm,
        image_size=IMAGE_SIZE, dt=1.0, x_dir=x, y_dir=y)


def test_world_offset_roundtrips(tmp_path):
    rec = _sf(1, (0.0, 0.0))
    rec.world_frame.world_offset_mm = np.array([12.5, -7.0])
    REC.save_mono(rec, tmp_path)
    loaded = REC.load_mono(tmp_path)
    assert np.allclose(loaded.world_frame.world_offset_mm, [12.5, -7.0])


def test_absent_offset_is_none(tmp_path):
    rec = _sf(1, (0.0, 0.0))
    REC.save_mono(rec, tmp_path)
    loaded = REC.load_mono(tmp_path)
    assert loaded.world_frame.world_offset_mm is None


def test_offset_adds_to_coordinates_not_velocity():
    """calibrate_coordinates adds the offset; velocity is offset-invariant."""
    m = _sf(1, (0.0, 0.0), px_per_mm=1.0).camera_model
    pts = np.array([[10.0, 0.0], [0.0, 10.0]])
    base = APPLY.calibrate_coordinates(m, pts)
    shifted = APPLY.calibrate_coordinates(m, pts, offset_mm=[100.0, 50.0])
    assert np.allclose(shifted - base, [100.0, 50.0])
    # velocity does NOT take an offset (it cancels in the displacement difference).
    # mmpp=1, dt=1 -> velocity = disp/1000 (mm->m); +Y up flips v.
    u, v = APPLY.calibrate_displacements(m, pts, np.array([[3.0, 0.0], [0.0, 4.0]]), dt=1.0)
    assert np.allclose(u, [3.0e-3, 0.0]) and np.allclose(v, [0.0, -4.0e-3])  # +Y up -> -4


def test_two_camera_stitch_via_baked_offset():
    """compute_camera_shifts -> bake into records -> the shared feature coincides."""
    r1 = _sf(1, (0.0, 0.0))
    r2 = _sf(2, (100.0, 50.0))
    pairs = [{"camera_a": 1, "camera_b": 2,
              "pixel_on_a": [200.0, 0.0], "pixel_on_b": [300.0, 0.0]}]
    shifts = GC.compute_camera_shifts(
        {1: r1, 2: r2}, datum_camera=1, datum_pixel=[0.0, 0.0],
        datum_physical=[10.0, 5.0], overlap_pairs=pairs)

    # Bake the computed shifts in, exactly as /global/save does.
    r1.world_frame.world_offset_mm = np.array(shifts[1])
    r2.world_frame.world_offset_mm = np.array(shifts[2])

    g1 = APPLY.calibrate_coordinates(
        r1.camera_model, np.array([[200.0, 0.0]]), offset_mm=r1.world_frame.world_offset_mm)[0]
    g2 = APPLY.calibrate_coordinates(
        r2.camera_model, np.array([[300.0, 0.0]]), offset_mm=r2.world_frame.world_offset_mm)[0]
    assert np.allclose(g1, g2)
    # datum reads its prescribed physical point.
    d = APPLY.calibrate_coordinates(
        r1.camera_model, np.array([[0.0, 0.0]]), offset_mm=r1.world_frame.world_offset_mm)[0]
    assert np.allclose(d, [10.0, 5.0])
