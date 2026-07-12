"""Synthetic-point honesty (triage B1) + per-view diagnostics persistence (B4).

B1: points the detector synthesises (template rescue of a missed dot, model
infill of a droplet-biased dot) are flagged in ``DetectionResult.synthetic_mask``
instead of passing silently as measurements. The fit is UNCHANGED — the mask is
for figures and diagnostics only (the bailey A/B run decides exclusion).

B4: per-view scalar diagnostics are summarised by ``pipeline.view_diagnostics_summary``
and persisted in ``board_meta["view_diagnostics"]``, surviving the .mat round-trip
(including the nested cam1/cam2 layout the stereo record uses).
"""

from pathlib import Path

import cv2
import numpy as np

from pivtools_gui.calibration import record as rec
from pivtools_gui.calibration.camera_model import CameraModel, DistortionModel
from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.detection.dotboard import DotboardDetector, DotboardParams
from pivtools_gui.calibration.detection.grid_detection import detect_grid_automatic
from pivtools_gui.calibration.pipeline import view_diagnostics_summary

OUT_DIR = Path(__file__).resolve().parent / "test_output" / "detection_flags"

N_COLS, N_ROWS = 10, 8
SPACING_PX = 40  # dot pitch
DOT_RADIUS = 8
MARGIN = 60
# Displacement for the droplet-biased dot: above the refine threshold (2 px)
# but inside the step-2 RANSAC gate (0.15 * spacing = 6 px), so the dot survives
# grid assembly and is caught + infilled by _refine_grid_outliers.
INFILL_SHIFT_PX = 4
FAINT_GRAY = 235  # faint dot: missed by the blob detector, NCC still ~1


def _dot_grid_image(displace=None, faint=None) -> np.ndarray:
    """White background, black dots on a regular grid.

    displace : (col, row) dot drawn INFILL_SHIFT_PX off-lattice (forces infill).
    faint : (col, row) dot drawn at FAINT_GRAY (blob detector misses it; the
        template-matching rescue can still find it — NCC is contrast-invariant).
    """
    h = 2 * MARGIN + (N_ROWS - 1) * SPACING_PX
    w = 2 * MARGIN + (N_COLS - 1) * SPACING_PX
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x = MARGIN + c * SPACING_PX
            y = MARGIN + r * SPACING_PX
            color = 0
            if faint is not None and (c, r) == tuple(faint):
                color = FAINT_GRAY
            if displace is not None and (c, r) == tuple(displace):
                x += INFILL_SHIFT_PX
            cv2.circle(img, (x, y), DOT_RADIUS, color, -1, lineType=cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# B1 — synthetic mask
# ---------------------------------------------------------------------------


def test_clean_grid_has_no_synthetic_points():
    ok, grid, info = detect_grid_automatic(_dot_grid_image())
    assert ok
    mask = grid["synthetic_mask"]
    assert mask.dtype == bool and len(mask) == len(grid["centers"])
    assert mask.sum() == 0
    assert info["n_rescued"] == 0 and info["n_infilled"] == 0


def test_forced_infill_mask_matches_diagnostics(make_figures):
    img = _dot_grid_image(displace=(4, 3), faint=(6, 5))
    ok, grid, info = detect_grid_automatic(img)
    assert ok
    mask = grid["synthetic_mask"]

    # The displaced dot MUST be infilled; the faint dot may or may not be
    # rescued, but the mask must agree with the diagnostics either way.
    assert info["n_infilled"] >= 1
    assert int(mask.sum()) == info["n_rescued"] + info["n_infilled"]
    assert info["n_synthetic"] == int(mask.sum())

    # The infilled point sits on the lattice (model prediction), not at the
    # displaced blob: every synthetic point is within 1 px of a lattice node.
    centers = np.asarray(grid["centers"], dtype=np.float64)
    lattice_err = np.abs((centers[mask] - MARGIN) % SPACING_PX)
    lattice_err = np.minimum(lattice_err, SPACING_PX - lattice_err)
    assert np.all(lattice_err < 1.0)

    if make_figures:
        from pivtools_gui.calibration import figures

        det = DotboardDetector(DotboardParams(dot_spacing_mm=15.0)).detect(img)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        figures.write_detection_figure(
            img,
            det,
            OUT_DIR / "forced_infill_detection.png",
            title="Forced infill (displaced dot) + faint-dot rescue",
        )


def test_extra_cluster_does_not_corrupt_mask():
    """A disconnected dot cluster (reflection-like) must not enter the grid or
    desync the synthetic mask: counts stay survivors-consistent and the mask
    stays length-aligned with the points (guards the Step-9 filter line)."""
    h = 2 * MARGIN + (N_ROWS - 1) * SPACING_PX
    w = 2 * MARGIN + (N_COLS - 1) * SPACING_PX + 4 * SPACING_PX
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x = MARGIN + c * SPACING_PX
            y = MARGIN + r * SPACING_PX
            if (c, r) == (4, 3):
                x += INFILL_SHIFT_PX
            cv2.circle(img, (x, y), DOT_RADIUS, 0, -1, lineType=cv2.LINE_AA)
    ox = MARGIN + (N_COLS - 1) * SPACING_PX + 3 * SPACING_PX
    for r in range(3):
        for c in range(3):
            cv2.circle(
                img,
                (ox + c * SPACING_PX, MARGIN + r * SPACING_PX),
                DOT_RADIUS,
                0,
                -1,
                lineType=cv2.LINE_AA,
            )

    ok, grid, info = detect_grid_automatic(img)
    assert ok
    mask = grid["synthetic_mask"]
    centers = grid["centers"]
    assert len(mask) == len(centers)  # alignment invariant
    assert info["n_grid_points"] == N_COLS * N_ROWS  # island excluded
    assert int(mask.sum()) == info["n_rescued"] + info["n_infilled"]
    assert info["n_synthetic"] == int(mask.sum())


def test_detection_result_carries_mask_and_diagnostics():
    img = _dot_grid_image(displace=(4, 3))
    det = DotboardDetector(DotboardParams(dot_spacing_mm=15.0)).detect(img)
    assert det.success
    assert det.synthetic_mask is not None
    assert det.synthetic_mask.dtype == bool
    assert len(det.synthetic_mask) == det.n
    assert det.synthetic_mask.sum() >= 1
    # info scalars now reach DetectionResult.diagnostics (B4 feedstock)
    for key in ("n_rescued", "n_infilled", "ransac_n_rejected", "edge_fraction"):
        assert key in det.diagnostics
    assert (
        det.diagnostics["n_infilled"]
        == int(det.synthetic_mask.sum()) - det.diagnostics["n_rescued"]
    )


# ---------------------------------------------------------------------------
# B4 — per-view diagnostics summary + .mat persistence
# ---------------------------------------------------------------------------


def _fake_detection(
    success=True, n=12, n_rescued=0, n_infilled=0, warning=None
) -> DetectionResult:
    diag = {
        "n_rescued": n_rescued,
        "n_infilled": n_infilled,
        "ransac_n_rejected": 1,
        "edge_fraction": 0.05,
    }
    if warning:
        diag["warning"] = warning
    n_synth = n_rescued + n_infilled
    mask = np.zeros(n, dtype=bool)
    mask[:n_synth] = True
    return DetectionResult(
        success=success,
        board_type="dotboard",
        image_points=np.random.default_rng(0).uniform(0, 100, (n, 2)),
        board_local_points=np.zeros((n, 3)),
        synthetic_mask=mask if success else None,
        diagnostics=diag,
    )


def test_view_diagnostics_summary_arrays():
    dets = [
        _fake_detection(n_rescued=2),
        _fake_detection(n_infilled=3, warning="partial board"),
    ]
    s = view_diagnostics_summary(dets)
    np.testing.assert_array_equal(s["view_index"], [0, 1])
    np.testing.assert_array_equal(s["success"], [1, 1])
    np.testing.assert_array_equal(s["n_rescued"], [2, 0])
    np.testing.assert_array_equal(s["n_infilled"], [0, 3])
    np.testing.assert_array_equal(s["n_synthetic"], [2, 3])
    np.testing.assert_array_equal(s["ransac_n_rejected"], [1, 1])
    assert s["warnings"] == "view 1: partial board"


def _pinhole() -> CameraModel:
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 512], [0, 0, 1]])
    return CameraModel(
        K=K,
        dist=np.zeros(5),
        R=np.eye(3),
        t=np.zeros((3, 1)),
        image_size=(1024, 1024),
        distortion_model=DistortionModel.STANDARD,
        rms=0.1,
    )


def test_mono_record_view_diagnostics_roundtrip(tmp_path):
    vd = view_diagnostics_summary(
        [_fake_detection(n_rescued=1), _fake_detection(warning="partial board")]
    )
    record = rec.MonoRecord(
        camera=1,
        board_type="dotboard",
        camera_model=_pinhole(),
        per_view_rms=[0.1, 0.2],
        board_meta={"spacing_mm": 15.0, "view_diagnostics": vd},
    )
    rec.save_mono(record, tmp_path)
    loaded = rec.load_mono(tmp_path)
    got = loaded.board_meta["view_diagnostics"]
    np.testing.assert_array_equal(got["n_rescued"], vd["n_rescued"])
    np.testing.assert_array_equal(got["n_synthetic"], vd["n_synthetic"])
    np.testing.assert_array_equal(got["edge_fraction"], vd["edge_fraction"])
    assert got["warnings"] == "view 1: partial board"


def test_stereo_record_nested_view_diagnostics_roundtrip(tmp_path):
    vd1 = view_diagnostics_summary([_fake_detection(n_rescued=1)])
    vd2 = view_diagnostics_summary([_fake_detection(n_infilled=2)])
    record = rec.StereoRecord(
        cam1=1,
        cam2=2,
        board_type="dotboard",
        model1=_pinhole(),
        model2=_pinhole(),
        R_stereo=np.eye(3),
        T_stereo=np.array([[100.0], [0], [0]]),
        per_view_rms1=[0.1],
        per_view_rms2=[0.2],
        board_meta={"spacing_mm": 15.0, "view_diagnostics": {"cam1": vd1, "cam2": vd2}},
    )
    rec.save_stereo(record, tmp_path)
    loaded = rec.load_stereo(tmp_path)
    got = loaded.board_meta["view_diagnostics"]
    # single-view summaries: size-1 arrays come back as scalars (_scalar), so
    # compare via np.asarray on both sides
    np.testing.assert_array_equal(
        np.asarray(got["cam1"]["n_rescued"]).reshape(-1), vd1["n_rescued"]
    )
    np.testing.assert_array_equal(
        np.asarray(got["cam2"]["n_infilled"]).reshape(-1), vd2["n_infilled"]
    )
    np.testing.assert_array_equal(
        np.asarray(got["cam2"]["n_synthetic"]).reshape(-1), vd2["n_synthetic"]
    )
