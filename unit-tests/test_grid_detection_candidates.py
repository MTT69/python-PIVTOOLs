"""Two flat-field windows x two polarities, winner by final gated grid (DETECTOR_VERSION 4).

``detect_dotboard_blobs`` runs the flat field with every dot-diameter estimate in
``{min(70, est), max(70, est)}`` (``est`` = 1.3 % of the long side) and both polarities;
``detect_grid_automatic`` finishes a grid for each candidate and keeps the largest FINAL
(RANSAC-validated, outlier-gated, connected) grid.

What each test pins:

* the candidate set and its order (estimate first, so ties keep the old single-window
  result), and the stepped caller's most-blobs default;
* a 75 px-dot board at the Merle frame size (2560x1664) detects every visible dot and
  reports all four candidates. A clean synthetic render does NOT reproduce the Merle
  camera 3 failure -- that one is a clutter effect (the small window passes many small
  clutter fragments, the median blob area drops and the 3x-median band rejects the dots),
  so the window that wins there is pinned by the real-data test in
  ``test_real_calibration_data.py``;
* the ramp-lit 2600 px board (the case that made the window a cap, 2026-10-18) stays
  180/180;
* selection by FINAL grid: a jittered half-pitch fragment set that walks a larger BFS grid
  than the real board loses once the outlier gate has run;
* small specks (contourArea well under the dots') are rejected by the 0.15x-median band
  alone, now that the fixed 50 px^2 floor is gone;
* ``DETECTOR_VERSION`` is 4 and detections cached by detector 3 are stale.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

# ruff: noqa: F811  (the imported ``env`` pytest fixture is re-bound by the test parameter)
from test_calibration_detect_views_routes import env  # noqa: F401  (pytest fixture)
from test_calibration_detect_views_routes import FMT, N_VIEWS, _post_views

from pivtools_cli.generate_synthetic_dotboard import generate
from pivtools_cli.synthetic_calibration_common import render_dots
from pivtools_gui.calibration import inputs_store as INP
from pivtools_gui.calibration.detection import grid_detection as GD

# Merle camera 3 frame size and the board geometry that gives ~75 px dots on it: the
# generator draws dots 0.44 x the local pitch, and an 11-column board fills 70 % of the
# 2560 px width at a ~179 px pitch.
MERLE_W, MERLE_H = 2560, 1664
DOTS75_COLS, DOTS75_ROWS = 11, 7
DOTS75_SPACING_MM = 21.2
# The 2026-10-18 ramp-lit set: 2598x1924, 15x12 dots, every dot visible in every view.
RAMP_MEGAPIXELS, RAMP_ASPECT = 5.0, 1.35
RAMP_N_DOTS = 15 * 12
N_SYNTH_VIEWS = 2
# Sub-pixel agreement with the projected ground truth on a clean render.
GT_TOL_PX = 0.5


def _dots_matched(found: np.ndarray, truth: np.ndarray) -> int:
    """Number of ground-truth dots with a detected point within ``GT_TOL_PX``."""
    return int((cKDTree(found).query(truth)[0] < GT_TOL_PX).sum())


# ---------------------------------------------------------------------------
# Candidate set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1664, 2560), [33, 70]),  # Merle: estimate below the anchor
        ((1680, 1248), [21, 70]),  # John (portrait): long side is the height
        ((3600, 5400), [70]),  # estimate == anchor: one window, not two
        ((4000, 6000), [78, 70]),  # estimate above the anchor
    ],
)
def test_candidate_dot_diameters(shape, expected):
    assert GD._candidate_dot_diameters(shape) == expected


def test_blob_candidates_cover_every_window_and_polarity():
    """Four candidates on a 2560 px frame; the default (stepped) pick is the most blobs."""
    img = np.full((MERLE_H, MERLE_W), 200, np.uint8)
    for x in range(400, 2200, 180):
        for y in range(300, 1400, 180):
            cv2.circle(img, (x, y), 37, 40, -1)

    centers, info = GD.detect_dotboard_blobs(img)

    cands = info["_candidates"]
    assert [(c["kernel_px"], c["invert"]) for c in cands] == [
        (33, False),
        (33, True),
        (70, False),
        (70, True),
    ]
    best = max(cands, key=lambda c: c["n_blobs"])
    assert len(centers) == best["n_blobs"] == info["n_blobs_detected"]
    assert info["kernel_px"] == best["kernel_px"]
    assert "_polarity_results" not in info


# ---------------------------------------------------------------------------
# Synthetic boards through the production detector
# ---------------------------------------------------------------------------


def test_75px_dots_at_merle_frame_size_detect_fully(tmp_path):
    gt = generate(
        tmp_path,
        n_views=N_SYNTH_VIEWS,
        megapixels=MERLE_W * MERLE_H / 1e6,
        aspect=MERLE_W / MERLE_H,
        cols=DOTS75_COLS,
        rows=DOTS75_ROWS,
        spacing_mm=DOTS75_SPACING_MM,
    )
    for i, path in enumerate(sorted(Path(tmp_path).glob("calib*.png"))):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert img.shape == (MERLE_H, MERLE_W)
        px, visible = gt["points_px"][i], gt["visible"][i]
        nn = np.sort(np.linalg.norm(px[:, None] - px[None], axis=2), axis=1)[:, 1]
        assert 70 <= 0.44 * np.median(nn) <= 80  # the board really carries ~75 px dots

        ok, grid, info = GD.detect_grid_automatic(img)

        assert ok, info.get("error")
        assert len(grid["centers"]) == int(visible.sum())
        assert _dots_matched(grid["centers"], px[visible]) == int(visible.sum())
        assert [c["kernel_px"] for c in info["candidates"]] == [33, 33, 70, 70]
        assert info["kernel_px"] in (33, 70)
        winner_n = max(c["n_final"] for c in info["candidates"])
        assert winner_n == info["n_grid_points"]


def test_ramp_lit_2600px_board_stays_180_of_180(tmp_path):
    gt = generate(
        tmp_path,
        n_views=N_SYNTH_VIEWS,
        megapixels=RAMP_MEGAPIXELS,
        aspect=RAMP_ASPECT,
        illumination="ramp",
    )
    for i, path in enumerate(sorted(Path(tmp_path).glob("calib*.png"))):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert max(img.shape) == 2598
        px = gt["points_px"][i]
        assert int(gt["visible"][i].sum()) == RAMP_N_DOTS

        ok, grid, info = GD.detect_grid_automatic(img)

        assert ok, info.get("error")
        assert len(grid["centers"]) == RAMP_N_DOTS
        assert _dots_matched(grid["centers"], px) == RAMP_N_DOTS
        # The cap case: the image-relative window must be the one that delivers it.
        assert info["kernel_px"] == 33


# Specks: radius 1.5-3 px discs (contourArea ~7-30 px^2) in the dot polarity, well clear of
# every dot and of the frame edge. The 1 MP default board carries dots of several hundred
# px^2, so 0.15x median sits far above every speck.
N_SPECKS = 20
SPECK_RADIUS_PX = (1.5, 3.0)
SPECK_CLEARANCE_PITCH = 0.5  # minimum speck-to-dot distance, in local dot pitches
SPECK_EDGE_MARGIN_PX = 10
SPECK_MIN_SEPARATION_PX = 12  # keeps two specks from merging into one larger contour
SPECK_MATCH_PX = 2.0
SPECK_SEED = 20261021


def _place_specks(
    truth: np.ndarray, shape: tuple, rng: np.random.Generator
) -> np.ndarray:
    """``N_SPECKS`` centres away from every dot, the frame edge and each other."""
    h, w = shape
    nn = cKDTree(truth).query(truth, k=2)[0][:, 1]
    clearance = SPECK_CLEARANCE_PITCH * float(np.median(nn))
    dots = cKDTree(truth)
    specks: list = []
    for _ in range(100 * N_SPECKS):
        c = rng.uniform(
            [SPECK_EDGE_MARGIN_PX, SPECK_EDGE_MARGIN_PX],
            [w - SPECK_EDGE_MARGIN_PX, h - SPECK_EDGE_MARGIN_PX],
        )
        if dots.query(c)[0] < clearance:
            continue
        if specks and np.min(np.linalg.norm(np.array(specks) - c, axis=1)) < (
            SPECK_MIN_SEPARATION_PX
        ):
            continue
        specks.append(c)
        if len(specks) == N_SPECKS:
            return np.array(specks)
    raise RuntimeError(f"placed only {len(specks)} of {N_SPECKS} specks")


def test_small_specks_rejected_by_median_band(tmp_path):
    """No fixed area floor: sub-50 px^2 specks reach the median band and die there."""
    gt = generate(tmp_path, n_views=1)
    img = cv2.imread(str(Path(tmp_path) / "calib00001.png"), cv2.IMREAD_UNCHANGED)
    truth = gt["points_px"][0][gt["visible"][0]]

    ok0, grid0, info0 = GD.detect_grid_automatic(img)
    assert ok0, info0.get("error")

    rng = np.random.default_rng(SPECK_SEED)
    specks = _place_specks(truth, img.shape, rng)
    radii = rng.uniform(*SPECK_RADIUS_PX, size=N_SPECKS)
    speck_layer = render_dots(specks, radii, (img.shape[1], img.shape[0]))
    specked = np.maximum(
        img, speck_layer
    )  # dots are 255 on 0: specks in the dot polarity

    # The specks must actually exercise the band: they survive circularity, under 50 px^2.
    contours, _ = cv2.findContours(
        (speck_layer > 127).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    small_round = [
        c
        for c in contours
        if len(c) >= 5
        and cv2.contourArea(c) < 50
        and 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2 > 0.4
    ]
    assert len(small_round) >= N_SPECKS // 2

    ok, grid, info = GD.detect_grid_automatic(specked)

    assert ok, info.get("error")
    assert len(grid["centers"]) == len(grid0["centers"])
    assert cKDTree(grid["centers"]).query(specks)[0].min() > SPECK_MATCH_PX


# ---------------------------------------------------------------------------
# Selection by final grid, not BFS grid
# ---------------------------------------------------------------------------

BOARD_COLS, BOARD_ROWS, BOARD_PITCH_PX = 10, 8, 120.0
BOARD_ORIGIN_PX = (200.0, 200.0)
# Half-pitch fragment lattice: walks a larger BFS grid than the board, but its centres
# scatter by more than the 2 px outlier gate, so most of it is dropped.
JUNK_COLS, JUNK_ROWS, JUNK_PITCH_PX = 22, 18, 60.0
JUNK_ORIGIN_PX = (100.0, 100.0)
JUNK_JITTER_PX = 3.0
FRAME_SHAPE = (1400, 1600)
DOT_RADIUS_PX = 25


def _candidate(centers: np.ndarray, flat_field: np.ndarray, kernel_px: int) -> dict:
    """A blob candidate shaped like an entry of ``detect_dotboard_blobs``' ``_candidates``."""
    return {
        "centers": centers.astype(np.float32),
        "flat_field": flat_field,
        "thresh": None,
        "invert": False,
        "kernel_px": kernel_px,
        "n_blobs": len(centers),
        "n_border_dropped": 0,
    }


def _lattice(cols: int, rows: int, pitch: float, origin) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([gx.ravel(), gy.ravel()]) * pitch + np.asarray(origin)


def test_real_board_beats_a_larger_bfs_fragment_set_after_the_gate():
    rng = np.random.default_rng(2026_10_20)
    board = _lattice(BOARD_COLS, BOARD_ROWS, BOARD_PITCH_PX, BOARD_ORIGIN_PX)
    junk = _lattice(JUNK_COLS, JUNK_ROWS, JUNK_PITCH_PX, JUNK_ORIGIN_PX)
    junk = junk + rng.normal(0.0, JUNK_JITTER_PX, junk.shape)
    flat_field = np.zeros(FRAME_SHAPE, np.uint8)
    for x, y in board:
        cv2.circle(flat_field, (int(round(x)), int(round(y))), DOT_RADIUS_PX, 255, -1)

    # Junk listed FIRST so that neither order nor a tie can hand the board the win.
    winner, finished, summaries = GD._grid_from_candidates(
        [_candidate(junk, flat_field, 33), _candidate(board, flat_field, 70)],
        k_neighbors=9,
    )

    junk_s, board_s = summaries
    assert junk_s["n_bfs"] > board_s["n_bfs"]  # the old largest-BFS rule picks the junk
    assert board_s["n_final"] == BOARD_COLS * BOARD_ROWS
    assert junk_s["n_final"] < board_s["n_final"]
    assert winner is not None and winner["kernel_px"] == 70
    assert finished["n_final"] == BOARD_COLS * BOARD_ROWS
    np.testing.assert_allclose(
        np.sort(finished["centers"], axis=0), np.sort(board, axis=0), atol=1e-3
    )


def test_tie_on_final_grid_goes_to_fewer_rescued_points():
    board = _lattice(BOARD_COLS, BOARD_ROWS, BOARD_PITCH_PX, BOARD_ORIGIN_PX)
    flat_field = np.zeros(FRAME_SHAPE, np.uint8)
    for x, y in board:
        cv2.circle(flat_field, (int(round(x)), int(round(y))), DOT_RADIUS_PX, 255, -1)
    # The same board with one interior dot missing: the rescue puts it back, so both
    # candidates finish at the full board, one of them with a rescued point.
    holed = np.delete(board, BOARD_COLS * 3 + 4, axis=0)

    winner, finished, summaries = GD._grid_from_candidates(
        [_candidate(holed, flat_field, 33), _candidate(board, flat_field, 70)],
        k_neighbors=9,
    )

    assert [s["n_final"] for s in summaries] == [BOARD_COLS * BOARD_ROWS] * 2
    assert winner["kernel_px"] == 70
    assert finished["n_rescued"] == 0


def test_no_candidate_with_a_grid_fails_loudly():
    flat_field = np.zeros(FRAME_SHAPE, np.uint8)
    few = _lattice(3, 3, BOARD_PITCH_PX, BOARD_ORIGIN_PX)

    ok, grid, info = GD.detect_grid_automatic(flat_field)

    assert not ok and grid is None
    assert "all 4 blob candidates" in info["error"]
    winner, finished, summaries = GD._grid_from_candidates(
        [_candidate(few, flat_field, 33)], k_neighbors=9
    )
    assert winner is None and finished is None
    assert summaries[0]["n_bfs"] == 0


# ---------------------------------------------------------------------------
# DETECTOR_VERSION 4
# ---------------------------------------------------------------------------


def test_detector_version_is_4():
    assert GD.DETECTOR_VERSION == 4
    assert INP.DETECTOR_VERSION == GD.DETECTOR_VERSION


def test_detections_from_detector_version_3_are_stale(env, monkeypatch):
    """Removing the area floor can change the points: a v3 sidecar must not be reused."""
    client, _base, _params = env
    monkeypatch.setattr(INP, "DETECTOR_VERSION", 3)
    _post_views(client, camera=1)
    monkeypatch.setattr(INP, "DETECTOR_VERSION", GD.DETECTOR_VERSION)

    restored = client.post(
        "/calibration/restore_detections",
        json={
            "source_path_idx": 0,
            "board": "dotboard",
            "image_format": FMT,
            "frame_total": N_VIEWS,
            "camera": 1,
        },
    ).get_json()

    assert restored["exists"] is True
    assert restored["fresh"] is False
    assert restored["by_camera"] is None
