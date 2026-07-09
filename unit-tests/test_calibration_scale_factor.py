"""Scale-factor method tests for the calibration package.

C-extension-free, runs everywhere. Covers the ``ScaleFactorModel`` back-projection
algebra, the axis-toggle sign mapping, the ``.mat`` record round-trip, the
multi-camera global-shift stitch, and — the load-bearing guard — that velocity
through the shared ``apply.calibrate_displacements`` reproduces the v1 scale-factor
formula ``disp_px / px_per_mm / dt / 1000`` exactly (no silent algorithm change).
"""

from __future__ import annotations

import numpy as np

from pivtools_gui.calibration import apply as APPLY
from pivtools_gui.calibration import global_coords as GC
from pivtools_gui.calibration import record as REC
from pivtools_gui.calibration.camera_model import ScaleFactorModel
from pivtools_gui.calibration.pipeline import build_scale_factor_record

IMAGE_SIZE = (1600, 1200)


# ---------------------------------------------------------------------------
# back_project_to_plane algebra
# ---------------------------------------------------------------------------

def test_back_project_known_points():
    """Origin + scale + default signs map pixels to hand-computed world mm."""
    # px_per_mm = 2 -> mm_per_pixel = 0.5; origin at (100, 200); +X right, +Y up.
    rec = build_scale_factor_record(
        camera=1, origin_px=(100.0, 200.0), px_per_mm=2.0,
        image_size=IMAGE_SIZE, dt=0.001, x_dir="right", y_dir="up",
    )
    model = rec.camera_model
    # 50 px right of origin -> +25 mm in X; 40 px BELOW origin (row +40) with +Y up
    # -> world Y = -20 mm (down is negative).
    pts = np.array([[100.0, 200.0], [150.0, 200.0], [100.0, 240.0]])
    world = model.back_project_to_plane(pts)
    assert np.allclose(world[0, :2], [0.0, 0.0])
    assert np.allclose(world[1, :2], [25.0, 0.0])
    assert np.allclose(world[2, :2], [0.0, -20.0])
    assert np.allclose(world[:, 2], 0.0)  # planar


def test_axis_toggles():
    """+Y up vs down flips row_sign; +X left flips col_sign; swap swaps axes."""
    yup = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=1.0,
        image_size=IMAGE_SIZE, dt=1.0, y_dir="up",
    ).camera_model
    ydown = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=1.0,
        image_size=IMAGE_SIZE, dt=1.0, y_dir="down",
    ).camera_model
    p = np.array([[0.0, 10.0]])  # 10 px down the image
    assert yup.back_project_to_plane(p)[0, 1] == -10.0   # +Y up -> world Y negative
    assert ydown.back_project_to_plane(p)[0, 1] == +10.0

    xleft = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=1.0,
        image_size=IMAGE_SIZE, dt=1.0, x_dir="left",
    ).camera_model
    assert xleft.back_project_to_plane(np.array([[10.0, 0.0]]))[0, 0] == -10.0

    swapped = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=1.0,
        image_size=IMAGE_SIZE, dt=1.0, x_dir="right", y_dir="down", swap_axes=True,
    ).camera_model
    # swap: X follows the row delta, Y follows the column delta.
    w = swapped.back_project_to_plane(np.array([[3.0, 7.0]]))[0]
    assert w[0] == 7.0 and w[1] == 3.0


# ---------------------------------------------------------------------------
# origin_mm — world position assigned to the picked origin pixel
# ---------------------------------------------------------------------------

def test_origin_mm_maps_picked_pixel():
    """The picked pixel back-projects to origin_mm, for every sign/swap combination.

    The offset is baked in by shifting the model's world-zero pixel; the world frame
    keeps the PICKED pixel + origin_mm so the GUI restores what the user entered.
    """
    picked = (321.0, 654.0)
    origin_mm = (12.5, -7.25)
    for x_dir in ("right", "left"):
        for y_dir in ("up", "down"):
            for swap in (False, True):
                rec = build_scale_factor_record(
                    camera=1, origin_px=picked, px_per_mm=3.2,
                    image_size=IMAGE_SIZE, dt=1.0,
                    x_dir=x_dir, y_dir=y_dir, swap_axes=swap,
                    origin_mm=origin_mm,
                )
                w = rec.camera_model.back_project_to_plane(np.array([picked]))[0]
                assert np.allclose(w[:2], origin_mm), (x_dir, y_dir, swap)
                assert np.allclose(rec.world_frame.origin_px, picked)
                assert np.allclose(rec.world_frame.origin_mm, origin_mm)


def test_origin_mm_displacement_invariant():
    """A constant world offset cancels in displacements — velocities are unchanged."""
    px_per_mm, dt = 16.72, 0.0005
    rng = np.random.default_rng(1)
    coords = rng.uniform(10, 1500, size=(32, 2))
    disp = rng.uniform(-8, 8, size=(32, 2))

    base = build_scale_factor_record(
        camera=1, origin_px=(800.0, 600.0), px_per_mm=px_per_mm,
        image_size=IMAGE_SIZE, dt=dt,
    ).camera_model
    offset = build_scale_factor_record(
        camera=1, origin_px=(800.0, 600.0), px_per_mm=px_per_mm,
        image_size=IMAGE_SIZE, dt=dt, origin_mm=(1000.0, -42.0),
    ).camera_model
    u0, v0 = APPLY.calibrate_displacements(base, coords, disp, dt)
    u1, v1 = APPLY.calibrate_displacements(offset, coords, disp, dt)
    assert np.allclose(u0, u1)
    assert np.allclose(v0, v1)


def test_origin_mm_roundtrip(tmp_path):
    """save_mono/load_mono preserves the picked origin + origin_mm on the world frame."""
    rec = build_scale_factor_record(
        camera=2, origin_px=(640.0, 512.0), px_per_mm=12.5,
        image_size=(1280, 1024), dt=0.0005, y_dir="down", origin_mm=(3.0, 4.0),
    )
    loaded = REC.load_mono(REC.save_mono(rec, tmp_path))
    assert np.allclose(loaded.world_frame.origin_px, [640.0, 512.0])
    assert np.allclose(loaded.world_frame.origin_mm, [3.0, 4.0])
    # The model's world-zero pixel (the baked shift) survives the round-trip too.
    assert np.allclose(loaded.camera_model.origin_px, rec.camera_model.origin_px)


# ---------------------------------------------------------------------------
# Velocity parity with v1 (the no-silent-algorithm-change guard)
# ---------------------------------------------------------------------------

def test_velocity_matches_v1():
    """apply.calibrate_displacements on a ScaleFactorModel == v1 scale-factor math.

    v1: ux_ms = ux_px / px_per_mm / dt / 1000. Default frame (+X right, +Y up) gives
    col_sign +1, row_sign -1; v1 operated on y-up coordinates, so its uy carries the
    same sign as our row_sign=-1 frame on y-up displacement. The two are algebraically
    identical; they differ only in float operation order (we carry mm_per_pixel =
    1/px_per_mm and subtract two back-projected positions), so equality holds to
    ~1e-14 absolute, far below any physical significance. We check X (no sign
    ambiguity) and the Y magnitude.
    """
    px_per_mm = 28.89
    dt = 0.000125
    rng = np.random.default_rng(0)
    coords = rng.uniform(10, 1500, size=(64, 2))
    disp = rng.uniform(-8, 8, size=(64, 2))

    model = build_scale_factor_record(
        camera=1, origin_px=(800.0, 600.0), px_per_mm=px_per_mm,
        image_size=IMAGE_SIZE, dt=dt, x_dir="right", y_dir="up",
    ).camera_model
    u, v = APPLY.calibrate_displacements(model, coords, disp, dt)

    v1_ux = disp[:, 0] / px_per_mm / dt / 1000.0
    v1_uy_mag = np.abs(disp[:, 1]) / px_per_mm / dt / 1000.0
    assert np.allclose(u, v1_ux, rtol=1e-9, atol=1e-12)
    assert np.allclose(np.abs(v), v1_uy_mag, rtol=1e-9, atol=1e-12)
    assert np.max(np.abs(u - v1_ux)) < 1e-10  # pure float reordering, not algorithm


# ---------------------------------------------------------------------------
# Record round-trip
# ---------------------------------------------------------------------------

def test_record_roundtrip(tmp_path):
    """save_mono/load_mono preserves every scale-factor field + model_type."""
    rec = build_scale_factor_record(
        camera=3, origin_px=(640.0, 512.0), px_per_mm=12.5,
        image_size=(1280, 1024), dt=0.0005, x_dir="left", y_dir="down", swap_axes=True,
    )
    path = REC.save_mono(rec, tmp_path)
    loaded = REC.load_mono(path)

    assert loaded.board_type == "scale_factor"
    m0, m1 = rec.camera_model, loaded.camera_model
    assert isinstance(m1, ScaleFactorModel)
    assert np.allclose(m1.origin_px, m0.origin_px)
    assert m1.mm_per_pixel == m0.mm_per_pixel
    assert (m1.col_sign, m1.row_sign, m1.swap_axes) == (m0.col_sign, m0.row_sign, m0.swap_axes)
    assert m1.image_size == m0.image_size
    assert loaded.camera == 3
    assert float(loaded.board_meta["px_per_mm"]) == 12.5
    assert float(loaded.board_meta["dt"]) == 0.0005


def test_frame_idx_stamped_and_roundtrips(tmp_path):
    """The pick frame is stamped into board_meta so the GUI restores the overlay on it.

    Omitted when not given (no spurious key), preserved through save/load when given.
    """
    rec_no = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=10.0, image_size=(100, 100), dt=1.0)
    assert "frame_idx" not in rec_no.board_meta

    rec = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=10.0, image_size=(100, 100),
        dt=1.0, frame_idx=37)
    assert rec.board_meta["frame_idx"] == 37
    loaded = REC.load_mono(REC.save_mono(rec, tmp_path))
    assert int(loaded.board_meta["frame_idx"]) == 37


# ---------------------------------------------------------------------------
# Multi-camera global stitch (reuses the existing chain, model-agnostic)
# ---------------------------------------------------------------------------

def test_global_shift_scale_factor():
    """compute_camera_shifts places the datum and stitches an overlap pair.

    Two cameras, identical 1 mm/px scale, default frame. Camera 1 is the datum:
    its origin pixel (0,0) is declared physical (10, 5). Camera 2 sees the same
    physical feature; matching pixels in the overlap fix camera 2's shift so the
    feature reads identically in both.
    """
    r1 = build_scale_factor_record(
        camera=1, origin_px=(0.0, 0.0), px_per_mm=1.0,
        image_size=IMAGE_SIZE, dt=1.0,
    )
    r2 = build_scale_factor_record(
        camera=2, origin_px=(100.0, 50.0), px_per_mm=1.0,
        image_size=IMAGE_SIZE, dt=1.0,
    )
    records = {1: r1, 2: r2}
    # Datum: cam1 pixel (0,0) -> physical (10, 5).
    # Overlap: a feature at cam1 pixel (200, 0) and cam2 pixel (300, 0) is the same point.
    pairs = [{"camera_a": 1, "camera_b": 2,
              "pixel_on_a": [200.0, 0.0], "pixel_on_b": [300.0, 0.0]}]
    shifts = GC.compute_camera_shifts(
        records, datum_camera=1, datum_pixel=[0.0, 0.0],
        datum_physical=[10.0, 5.0], overlap_pairs=pairs,
    )
    # Datum shift puts cam1 origin at (10, 5): world(0,0)=(0,0)+shift -> shift=(10,5).
    assert np.allclose(shifts[1], [10.0, 5.0])

    # The shared feature must read identically in both global frames.
    w1 = APPLY.calibrate_coordinates(r1.camera_model, np.array([[200.0, 0.0]]))[0]
    w2 = APPLY.calibrate_coordinates(r2.camera_model, np.array([[300.0, 0.0]]))[0]
    g1 = w1 + np.asarray(shifts[1])
    g2 = w2 + np.asarray(shifts[2])
    assert np.allclose(g1, g2)
