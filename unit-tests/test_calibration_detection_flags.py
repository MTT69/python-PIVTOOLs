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
# Displacement for the droplet-biased dot: above the outlier threshold (2 px)
# but inside the step-2 RANSAC gate (0.15 * spacing = 6 px), so the dot survives
# grid assembly and is caught + DROPPED by _drop_grid_outliers (nothing is infilled
# from the homography since 2026-08-26: it cannot tell a droplet from distortion).
INFILL_SHIFT_PX = 4
FAINT_GRAY = 235  # faint dot: missed by the blob detector, NCC still ~1


def _dot_grid_image(displace=None, faint=None, margin_right=MARGIN) -> np.ndarray:
    """White background, black dots on a regular grid.

    displace : (col, row) dot drawn INFILL_SHIFT_PX off-lattice (forces a drop).
    faint : (col, row) dot drawn at FAINT_GRAY (blob detector misses it; the
        template-matching rescue can still find it — NCC is contrast-invariant).
    margin_right : distance from the last column to the right image edge. Below
        DOT_RADIUS the last column is cut by the border.
    """
    h = 2 * MARGIN + (N_ROWS - 1) * SPACING_PX
    w = MARGIN + margin_right + (N_COLS - 1) * SPACING_PX
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
    assert info["n_rescued"] == 0 and info["n_outliers_dropped"] == 0


def test_border_clipped_dots_are_rejected():
    """A dot cut by the image edge is dropped, never measured.

    fitEllipse on a truncated contour shifts the centre inward by up to the
    clipped width (measured 2026-08-28 on the synthetic stereo fixture: +/-2 px in x
    at the frame edge, under the 2 px outlier gate). The rest of the grid is
    unaffected, and the clean grid keeps every dot.
    """
    img = _dot_grid_image(margin_right=DOT_RADIUS // 2)
    ok, grid, info = detect_grid_automatic(img)
    assert ok
    centers = np.asarray(grid["centers"], float)
    assert len(centers) == N_ROWS * (N_COLS - 1)
    assert info["n_border_dropped"] == N_ROWS
    last_full_col_x = MARGIN + (N_COLS - 2) * SPACING_PX
    assert centers[:, 0].max() < last_full_col_x + 0.5
    # Every surviving centre sits on the drawn lattice.
    cols = np.round((centers[:, 0] - MARGIN) / SPACING_PX)
    rows = np.round((centers[:, 1] - MARGIN) / SPACING_PX)
    lattice = np.column_stack([MARGIN + cols * SPACING_PX, MARGIN + rows * SPACING_PX])
    assert np.abs(centers - lattice).max() < 0.1

    ok_clean, grid_clean, info_clean = detect_grid_automatic(_dot_grid_image())
    assert ok_clean and len(grid_clean["centers"]) == N_ROWS * N_COLS
    assert info_clean["n_border_dropped"] == 0


def test_forced_infill_mask_matches_diagnostics(make_figures):
    img = _dot_grid_image(displace=(4, 3), faint=(6, 5))
    ok, grid, info = detect_grid_automatic(img)
    assert ok
    mask = grid["synthetic_mask"]

    # The displaced dot MUST be dropped, never replaced by a model prediction; the
    # faint dot may or may not be rescued, but the mask must agree with the
    # diagnostics either way.
    assert info["n_outliers_dropped"] >= 1
    assert int(mask.sum()) == info["n_rescued"]
    assert info["n_synthetic"] == int(mask.sum())

    # No surviving point sits at the displaced blob's position, and no point was
    # fabricated at its lattice node: the node is absent from the grid.
    centers = np.asarray(grid["centers"], dtype=np.float64)
    gi = np.asarray(grid["grid_indices"])
    node = np.array([MARGIN + 4 * SPACING_PX, MARGIN + 3 * SPACING_PX], dtype=np.float64)
    assert np.min(np.linalg.norm(centers - node, axis=1)) > 0.5 * SPACING_PX
    assert len(gi) == len(centers)

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
    # island excluded; a lattice dot whose blob the island disturbed is DROPPED now
    # (it used to be infilled), so the count is the lattice minus the drops
    assert info["n_outliers_dropped"] <= 1
    assert info["n_grid_points"] + info["n_outliers_dropped"] == N_COLS * N_ROWS
    assert int(mask.sum()) == info["n_rescued"]
    assert info["n_synthetic"] == int(mask.sum())


def test_detection_result_carries_mask_and_diagnostics():
    img = _dot_grid_image(displace=(4, 3))
    det = DotboardDetector(DotboardParams(dot_spacing_mm=15.0)).detect(img)
    assert det.success
    assert det.synthetic_mask is not None
    assert det.synthetic_mask.dtype == bool
    assert len(det.synthetic_mask) == det.n
    # the displaced dot is dropped, not fabricated: nothing synthetic remains
    assert det.synthetic_mask.sum() == det.diagnostics["n_rescued"]
    # info scalars now reach DetectionResult.diagnostics (B4 feedstock)
    for key in (
        "n_rescued",
        "n_outliers_dropped",
        "n_border_dropped",
        "ransac_n_rejected",
        "edge_fraction",
    ):
        assert key in det.diagnostics
    assert det.diagnostics["n_outliers_dropped"] >= 1


# ---------------------------------------------------------------------------
# Outlier gate — distortion must survive, reflections must not (Package A, 2026-09-01)
# ---------------------------------------------------------------------------

# A larger board rendered through a perspective tilt, optionally through radial lens
# distortion, optionally with mirrored rows below the board (a reflection in the
# surface the board stands on). Sized so the gate, not the step-2 RANSAC, decides.
PERSP_N_COLS, PERSP_N_ROWS = 24, 16
PERSP_SPACING_PX = 60
PERSP_DOT_RADIUS = 6
PERSP_MARGIN = 80
# Row pitch at the far edge of the board over the pitch at the near edge (the tilt).
PERSP_FAR_PITCH_RATIO = 0.9
# Barrel term: the corner of the board moves DISTORTION_K1 * corner radius (~12 px at
# this size). A plain homography leaves 3-6 px residuals at the periphery, the regime
# measured on the gF_Ramp joint calibration (2-11 px on a 4872x3248 frame).
DISTORTION_K1 = 0.015
# Mirror plane below the last row, in units of the last row pitch. 0.5 would make the
# first mirrored row an exact continuation of the lattice (no gate can see it); 0.6 puts
# a crease at the seam like a real reflection.
REFLECTION_GAP_FRAC = 0.6
# Kept dots must sit on the rendered lattice to this tolerance (anti-aliased circles).
LATTICE_TOL_PX = 0.75


def _perspective_board_points(n_reflected_rows: int, k1: float):
    """Board dots + mirrored rows, in pixels, after tilt and radial distortion.

    Returns ``(board_xy, reflected_xy, image_shape)``. The board is a homography of the
    lattice (rows compress toward the bottom by PERSP_FAR_PITCH_RATIO); the mirrored
    rows are the last ``n_reflected_rows`` board rows reflected about a horizontal
    plane REFLECTION_GAP_FRAC pitches below the last row. Barrel distortion with
    coefficient ``k1`` (radius normalised by the board's corner radius) is then applied
    about the image centre to every dot.
    """
    n_cols, n_rows, s, m = PERSP_N_COLS, PERSP_N_ROWS, PERSP_SPACING_PX, PERSP_MARGIN
    persp = (1.0 / PERSP_FAR_PITCH_RATIO - 1.0) / ((n_rows - 1) * s)
    c, r = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    u = c.ravel() * s
    v = r.ravel() * s
    w_div = 1.0 + persp * v
    half_width = (n_cols - 1) * s / 2
    x = m + (u - half_width) / w_div + half_width
    y = m + v / w_div
    board = np.column_stack([x, y])

    row_y = np.array([m + (rr * s) / (1.0 + persp * rr * s) for rr in range(n_rows)])
    y_seam = row_y[-1] + REFLECTION_GAP_FRAC * (row_y[-1] - row_y[-2])
    reflected = [
        np.column_stack([x[r.ravel() == rr], 2.0 * y_seam - y[r.ravel() == rr]])
        for rr in range(n_rows - 1, n_rows - 1 - n_reflected_rows, -1)
    ]
    all_pts = np.vstack([board] + reflected)

    width = int(2 * m + (n_cols - 1) * s)
    height = int(all_pts[:, 1].max() + m)
    centre = np.array([width / 2, height / 2])
    d = all_pts - centre
    r_max = np.linalg.norm(board - centre, axis=1).max()
    r2 = (np.linalg.norm(d, axis=1) / r_max) ** 2
    all_pts = centre + d * (1.0 + k1 * r2)[:, None]
    n_board = len(board)
    return all_pts[:n_board], all_pts[n_board:], (height, width)


def _render_dots(points: np.ndarray, shape) -> np.ndarray:
    img = np.full(shape, 255, dtype=np.uint8)
    for x, y in points:
        cv2.circle(
            img,
            (int(round(x)), int(round(y))),
            PERSP_DOT_RADIUS,
            0,
            -1,
            lineType=cv2.LINE_AA,
        )
    return img


def _split_kept(centers: np.ndarray, board: np.ndarray, reflected: np.ndarray):
    """Count kept dots nearer a board dot than a reflected dot, and the converse."""
    from scipy.spatial import cKDTree

    d_board = cKDTree(board).query(centers)[0]
    if len(reflected):
        d_refl = cKDTree(reflected).query(centers)[0]
    else:
        d_refl = np.full(len(centers), np.inf)
    on_board = d_board < d_refl
    return int(on_board.sum()), int((~on_board).sum()), d_board[on_board]


def test_distorted_board_periphery_survives():
    """Radial lens distortion is not an outlier: the whole board is kept.

    Regression for the gF_Ramp amputation (2026-09-01): a homography-only gate
    dropped 33-41% of the board because it read distortion as disagreement.
    """
    board, reflected, shape = _perspective_board_points(0, DISTORTION_K1)
    # Fixture self-check: the distortion is large enough that a plain homography
    # cannot absorb it (otherwise this test would not exercise the gate).
    c, r = np.meshgrid(np.arange(PERSP_N_COLS), np.arange(PERSP_N_ROWS))
    src = np.column_stack([c.ravel(), r.ravel()]).astype(np.float32)
    H, _ = cv2.findHomography(src, board.astype(np.float32), method=0)
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.linalg.norm(pred - board, axis=1).max() > 3.0

    ok, grid, info = detect_grid_automatic(_render_dots(board, shape))
    assert ok
    assert info["ransac_n_rejected"] == 0
    assert info["n_outliers_dropped"] == 0
    centers = np.asarray(grid["centers"], dtype=np.float64)
    n_board, n_refl, d_board = _split_kept(centers, board, reflected)
    assert n_board == PERSP_N_COLS * PERSP_N_ROWS and n_refl == 0
    assert d_board.max() < LATTICE_TOL_PX


def test_distorted_board_with_reflection_keeps_whole_board():
    """Distortion plus a two-row reflection: the board is kept whole and no
    reflected dot survives (whichever stage rejects the reflection)."""
    board, reflected, shape = _perspective_board_points(2, DISTORTION_K1)
    ok, grid, info = detect_grid_automatic(_render_dots(np.vstack([board, reflected]), shape))
    assert ok
    centers = np.asarray(grid["centers"], dtype=np.float64)
    n_board, n_refl, d_board = _split_kept(centers, board, reflected)
    assert n_refl == 0
    assert n_board == PERSP_N_COLS * PERSP_N_ROWS
    assert d_board.max() < LATTICE_TOL_PX
    assert info["ransac_n_rejected"] + info["n_outliers_dropped"] >= 1


def test_reflection_seam_rows_dropped():
    """No distortion, two mirrored rows that pass the step-2 RANSAC: the gate must
    drop the reflection without eating the board.

    Guards the failure mode found while porting the radial gate (2026-09-01): with
    the board an exact homography, a single-round robust radial fit leans into a
    coherent reflection block and drops 10-13% of the board next to the seam. The
    drop-and-refit iteration removes that; this test pins it.
    """
    board, reflected, shape = _perspective_board_points(2, 0.0)
    ok, grid, info = detect_grid_automatic(_render_dots(np.vstack([board, reflected]), shape))
    assert ok
    centers = np.asarray(grid["centers"], dtype=np.float64)
    n_board, n_refl, d_board = _split_kept(centers, board, reflected)
    assert n_refl == 0
    assert info["n_outliers_dropped"] >= 1
    # the two board dots at the seam corners sit on the crease and may go either way
    assert n_board >= PERSP_N_COLS * PERSP_N_ROWS - 2
    assert d_board.max() < LATTICE_TOL_PX


def test_outlier_gate_refit_floor_is_visible():
    """A refit round may not run on fewer than the minimum dot count.

    14 of a 6x5 grid's 30 dots are scattered 8-15 px off-lattice, so the first robust
    round sides with the 16 clean dots, below ``_OUTLIER_FIT_MIN_DOTS``. The gate must
    stop there with a WARNING and gate on the residuals it already has, never refit
    12 parameters on 16 dots.
    """
    from loguru import logger

    from pivtools_gui.calibration.detection.grid_detection import (
        _OUTLIER_FIT_MIN_DOTS,
        _drop_grid_outliers,
    )

    n_cols, n_rows, pitch = 6, 5, 50.0
    rng = np.random.default_rng(3)
    grid = {}
    centers = []
    for r in range(n_rows):
        for c in range(n_cols):
            grid[(c, r)] = len(centers)
            centers.append([100.0 + c * pitch, 100.0 + r * pitch])
    centers = np.asarray(centers)
    scattered = rng.choice(len(centers), size=14, replace=False)
    shift = rng.uniform(8.0, 15.0, size=(14, 2)) * rng.choice([-1.0, 1.0], size=(14, 2))
    centers[scattered] += shift
    assert len(grid) >= _OUTLIER_FIT_MIN_DOTS > len(grid) - len(scattered)

    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        pruned, dropped = _drop_grid_outliers(grid, np.asarray(centers))
    finally:
        logger.remove(sink)
    assert len(pruned) + len(dropped) == len(grid)
    assert set(pruned) | set(dropped) == set(grid)
    assert any("fewer than" in m for m in messages), messages


# ---------------------------------------------------------------------------
# B4 — per-view diagnostics summary + .mat persistence
# ---------------------------------------------------------------------------


def _fake_detection(
    success=True, n=12, n_rescued=0, n_outliers_dropped=0, n_border_dropped=0, warning=None
) -> DetectionResult:
    diag = {
        "n_rescued": n_rescued,
        "n_outliers_dropped": n_outliers_dropped,
        "n_border_dropped": n_border_dropped,
        "ransac_n_rejected": 1,
        "edge_fraction": 0.05,
    }
    if warning:
        diag["warning"] = warning
    n_synth = n_rescued  # dropped dots are absent, not synthetic
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
        _fake_detection(n_outliers_dropped=3, n_border_dropped=4, warning="partial board"),
    ]
    s = view_diagnostics_summary(dets)
    np.testing.assert_array_equal(s["view_index"], [0, 1])
    np.testing.assert_array_equal(s["success"], [1, 1])
    np.testing.assert_array_equal(s["n_rescued"], [2, 0])
    np.testing.assert_array_equal(s["n_outliers_dropped"], [0, 3])
    np.testing.assert_array_equal(s["n_border_dropped"], [0, 4])
    np.testing.assert_array_equal(s["n_synthetic"], [2, 0])
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
    vd2 = view_diagnostics_summary([_fake_detection(n_outliers_dropped=2)])
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
        np.asarray(got["cam2"]["n_outliers_dropped"]).reshape(-1), vd2["n_outliers_dropped"]
    )
    np.testing.assert_array_equal(
        np.asarray(got["cam2"]["n_synthetic"]).reshape(-1), vd2["n_synthetic"]
    )
