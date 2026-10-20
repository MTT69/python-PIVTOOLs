"""Stereo self-calibration tests for the calibration package.

The C-extension-free tests below exercise the calibration bridge, the dewarp world
bounds, the record ``self_cal`` round-trip, and that the stereo reconstruction
honours a stored sheet — they run everywhere. The end-to-end recovery test (which
needs the ``libbulkxcorr2d`` C extension) lives in
``test_calibration_self_cal_recovery.py`` and skips cleanly when the lib is absent.
"""

from __future__ import annotations

from pathlib import Path

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
        cam1=1,
        cam2=2,
        board_type="dotboard",
        model1=m1,
        model2=m2,
        R_stereo=m2.R @ m1.R.T,
        T_stereo=m2.t - (m2.R @ m1.R.T) @ m1.t,
        world_frame=REC.WorldFrame(),
        per_view_rms1=[0.1],
        per_view_rms2=[0.12],
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
        {},
        n_images=20,
        window_size=64,
        overlap=50.0,
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

    assert not np.allclose(rec.model1.R, R1_before)  # poses moved
    Rs1, Ts1 = compose_stereo(rec.model1, rec.model2)
    assert np.allclose(Rs0, Rs1, atol=1e-12)  # cross-camera pose invariant
    assert np.allclose(Ts0, Ts1, atol=1e-9)
    assert np.allclose(rec.R_stereo, Rs1, atol=1e-12)  # stored values updated
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
        old.model1, old.model2, coords1, coords2, z, tx, ty
    )

    new = _stereo_record()
    SC.rebake_record(new, z, tx, ty)
    gX_n, gY_n, gZ_n, sp_n = regular_world_grid(
        new.model1, new.model2, coords1, coords2
    )

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
        m1,
        m2,
        (gX, gY, gZ),
        coords1,
        zero,
        zero,
        coords2,
        zero,
        zero,
        dt=1.0,
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
# Repeated self-cal: the baked block reports the TOTAL sheet in the clicked frame
# ---------------------------------------------------------------------------

# Two successive self-cal increments (z_offset mm, tilt_x rad, tilt_y rad). Each is
# measured in the frame left by the previous bake.
SHEET_A = (0.8, 0.02, -0.015)
SHEET_B = (-0.3, -0.007, 0.011)
# Exact composition is pure linear algebra: anything above round-off is a bug.
EXACT_TOL = 1e-9
# Naive increment summation drops second-order rotation/translation coupling; for
# SHEET_A/SHEET_B it is off by far more than round-off.
NAIVE_MIN_ERR = 1e-6
FIGURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "debug"
    / "2026-10-17-selfcal-cumulative.png"
)


def _sheet_result(sheet) -> "_FakeResult":
    z, tx, ty = sheet
    return _FakeResult(z=z, tx=tx, ty=ty, conv=True, rms=0.05, n=3)


def _bake(rec: REC.StereoRecord, sheet) -> None:
    """One self-cal bake exactly as the CLI/GUI callers do it (prev read first)."""
    prev = rec.self_cal
    SC.rebake_record(rec, *sheet)
    rec.self_cal = SC.baked_block(
        _sheet_result(sheet), prev, n_images=20, window_size=64, overlap=50.0
    )


def _ground_truth_total(sheets) -> tuple:
    """Total sheet in the ORIGINAL frame, independent of ``cumulative_sheet``.

    Composes the per-bake NEW->OLD maps directly, carries three non-collinear points
    of the final Z=0 plane into the original frame, and fits
    ``Z = z + X*tan(ty) + Y*tan(tx)`` through them by a 3x3 linear solve.
    """
    from pivtools_gui.calibration.self_cal_frame import plane_to_world_correction

    pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    # Map final-frame points back through each bake, newest first.
    for sheet in reversed(sheets):
        R_c, t_c = plane_to_world_correction(*sheet)
        pts = (R_c @ pts.T).T + t_c
    A = np.column_stack([np.ones(3), pts[:, 0], pts[:, 1]])
    z, a, b = np.linalg.solve(A, pts[:, 2])
    return float(z), float(np.arctan(b)), float(np.arctan(a))


def _fitted(block: dict) -> tuple:
    return (
        float(block["fitted_z_offset"]),
        float(block["fitted_tilt_x"]),
        float(block["fitted_tilt_y"]),
    )


def test_single_bake_cumulative_fields():
    """First bake: fitted_* = the result, cum_* encode plane_to_world_correction."""
    import cv2

    from pivtools_gui.calibration.self_cal_frame import plane_to_world_correction

    rec = _stereo_record()
    _bake(rec, SHEET_A)
    sc = rec.self_cal

    assert _fitted(sc) == pytest.approx(SHEET_A, abs=1e-15)
    last = (
        sc["last_fitted_z_offset"],
        sc["last_fitted_tilt_x"],
        sc["last_fitted_tilt_y"],
    )
    assert last == SHEET_A
    assert sc["n_bakes"] == 1
    assert (sc["z_offset"], sc["tilt_x"], sc["tilt_y"], sc["baked"]) == (
        0.0,
        0.0,
        0.0,
        1,
    )

    R_c, t_c = plane_to_world_correction(*SHEET_A)
    rvec = np.array([sc["cum_rvec_x"], sc["cum_rvec_y"], sc["cum_rvec_z"]])
    t_cum = np.array([sc["cum_t_x"], sc["cum_t_y"], sc["cum_t_z"]])
    assert np.allclose(cv2.Rodrigues(rvec)[0], R_c, atol=1e-15)
    assert np.array_equal(t_cum, t_c)


def test_two_bakes_report_total_sheet_in_original_frame():
    """Second bake composes: fitted_* is the total sheet, not the increment."""
    import cv2

    rec = _stereo_record()
    R1_orig, t1_orig = rec.model1.R.copy(), rec.model1.t.copy()
    _bake(rec, SHEET_A)
    _bake(rec, SHEET_B)
    sc = rec.self_cal

    truth = _ground_truth_total([SHEET_A, SHEET_B])
    assert _fitted(sc) == pytest.approx(truth, abs=EXACT_TOL)
    assert (
        sc["last_fitted_z_offset"],
        sc["last_fitted_tilt_x"],
        sc["last_fitted_tilt_y"],
    ) == SHEET_B
    assert sc["n_bakes"] == 2

    # cum_* is the actual geometric map between the original and rebaked poses:
    # X_cam = R1 X_orig + t1 = R1'' X_final + t1''  ->  R_tot = R1^T R1'',
    # t_tot = R1^T (t1'' - t1).
    R_tot = cv2.Rodrigues(
        np.array([sc["cum_rvec_x"], sc["cum_rvec_y"], sc["cum_rvec_z"]])
    )[0]
    t_tot = np.array([sc["cum_t_x"], sc["cum_t_y"], sc["cum_t_z"]])
    assert np.allclose(R_tot, R1_orig.T @ rec.model1.R, atol=1e-12)
    assert np.allclose(t_tot, (R1_orig.T @ (rec.model1.t - t1_orig)).ravel(), atol=1e-9)

    # Why exact composition matters: summing increments is measurably wrong.
    naive = tuple(a + b for a, b in zip(SHEET_A, SHEET_B))
    assert np.max(np.abs(np.subtract(naive, truth))) > NAIVE_MIN_ERR


def test_stale_baked_block_raises():
    """A baked prev block without cum_* is stale: fail loudly, never assume identity."""
    stale = {
        "z_offset": 0.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
        "baked": 1,
        "fitted_z_offset": 0.5,
        "fitted_tilt_x": 0.0,
        "fitted_tilt_y": 0.0,
    }
    with pytest.raises(ValueError, match="regenerate"):
        SC.baked_block(
            _sheet_result(SHEET_B), stale, n_images=20, window_size=64, overlap=50.0
        )


def test_cumulative_block_roundtrips(tmp_path):
    """All cumulative fields survive save_stereo -> load_stereo with type intact."""
    rec = _stereo_record()
    _bake(rec, SHEET_A)
    _bake(rec, SHEET_B)
    out = REC.load_stereo(REC.save_stereo(rec, tmp_path))
    keys = (
        "cum_rvec_x",
        "cum_rvec_y",
        "cum_rvec_z",
        "cum_t_x",
        "cum_t_y",
        "cum_t_z",
        "last_fitted_z_offset",
        "last_fitted_tilt_x",
        "last_fitted_tilt_y",
        "fitted_z_offset",
        "fitted_tilt_x",
        "fitted_tilt_y",
    )
    for k in keys:
        assert float(out.self_cal[k]) == rec.self_cal[k], k
    assert int(out.self_cal["n_bakes"]) == 2

    # A third bake on the reloaded record still composes against the saved map.
    _bake(out, SHEET_A)
    truth = _ground_truth_total([SHEET_A, SHEET_B, SHEET_A])
    assert _fitted(out.self_cal) == pytest.approx(truth, abs=EXACT_TOL)
    assert int(out.self_cal["n_bakes"]) == 3


def test_cumulative_sweep_figure(make_figures):
    """Debug figure: reported total vs ground truth across second-bake magnitudes."""
    if not make_figures:
        pytest.skip("figure generation needs --make-figures")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = np.linspace(0.0, 3.0, 31)
    reported, truth, naive = [], [], []
    for s in scales:
        sheet_b = tuple(s * v for v in SHEET_B)
        rec = _stereo_record()
        _bake(rec, SHEET_A)
        _bake(rec, sheet_b)
        reported.append(_fitted(rec.self_cal))
        truth.append(_ground_truth_total([SHEET_A, sheet_b]))
        naive.append(tuple(a + b for a, b in zip(SHEET_A, sheet_b)))
    reported, truth, naive = map(np.asarray, (reported, truth, naive))
    exact_err = reported - truth
    naive_err = naive - truth
    max_exact = float(np.max(np.abs(exact_err)))

    # Display units: z in mm, tilts in mrad.
    unit_scale = np.array([1.0, 1e3, 1e3])
    labels = ("z_offset [mm]", "tilt_x [mrad]", "tilt_y [mrad]")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for j in range(3):
        ax = axes[0, j]
        ax.plot(scales, truth[:, j] * unit_scale[j], "k-", lw=2, label="ground truth")
        ax.plot(
            scales,
            reported[:, j] * unit_scale[j],
            "o",
            ms=4,
            mfc="none",
            label="reported fitted_*",
        )
        ax.plot(scales, naive[:, j] * unit_scale[j], "--", label="naive sum")
        ax.set_ylabel(f"total {labels[j]}")
        ax.legend(fontsize=8)
        axes[1, j].plot(scales, exact_err[:, j] * unit_scale[j], "o-", ms=3)
        axes[1, j].set_ylabel(f"exact residual {labels[j]}")
        axes[2, j].plot(scales, naive_err[:, j] * unit_scale[j], "r.-")
        axes[2, j].set_ylabel(f"naive-sum error {labels[j]}")
        axes[2, j].set_xlabel("second-bake scale s (B -> s*B)")
    fig.suptitle(
        "Repeated self-cal: total sheet in clicked frame, A="
        f"{SHEET_A}, B={SHEET_B} (mm, rad); max |exact residual| = {max_exact:.2e}"
    )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)

    assert FIGURE_PATH.exists()
    assert max_exact < EXACT_TOL


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
