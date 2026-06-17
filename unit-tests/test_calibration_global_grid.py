"""Unit tests for the global-grid resolver (S1·Phase 1).

These exercise ``resolve_global_grid`` directly at the ``DetectionResult`` level: a known
global board, three fixed cameras each seeing a partial (overlapping) column window, the
board imaged at several poses, and each view's grid indices RELABELLED by an arbitrary
signed permutation + offset (what a per-view BFS produces). The resolver must invert all
that and recover the one true global index for every dot in every (camera, view).

Covers: the datum 3-click frame, within-camera 1-click anchors resolved by the same-camera
orientation prior, cross-camera links via >=2 overlap correspondences, the consistency
guard (inconsistent clicks raise, not mislabel), and the ChArUco short-circuit.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pivtools_gui.calibration.detection.base import DetectionResult
from pivtools_gui.calibration.global_grid import (
    _ORIENTATIONS,
    Anchor,
    Correspondence,
    GlobalGridSpec,
    resolve_global_grid,
    resolve_global_grid_partial,
)

SPACING = 15.0
K = np.array([[1200.0, 0, 640.0], [0, 1200.0, 512.0], [0, 0, 1.0]])

# Fixed camera extrinsics (world->camera) — three distinct viewpoints of one rig.
_CAM_POSES = {
    1: (np.array([0.00, 0.00, 0.0]), np.array([-50.0, -60.0, 520.0])),
    2: (np.array([0.00, 0.32, 0.0]), np.array([-95.0, -60.0, 540.0])),
    3: (np.array([0.00, -0.32, 0.0]), np.array([-10.0, -60.0, 540.0])),
}
# Each camera sees a partial column window; adjacent windows overlap (a connected chain).
_CAM_WINDOWS = {1: range(0, 7), 2: range(4, 10), 3: range(7, 12)}
_ROWS = range(0, 9)
_N_VIEWS = 4


def _orient_map(gidx: np.ndarray, orient, offset) -> np.ndarray:
    """Relabel global indices by a signed permutation + integer offset (a BFS local frame)."""
    swap, sx, sy = orient
    g = gidx.astype(np.int64)
    if not swap:
        lx, ly = sx * g[:, 0], sy * g[:, 1]
    else:
        lx, ly = sx * g[:, 1], sy * g[:, 0]
    return np.column_stack([lx + offset[0], ly + offset[1]]).astype(np.int64)


def _board_pose(view: int, rng) -> tuple:
    """Per-view board pose: out-of-plane tilt + translation, in-plane rotation kept small.

    Small in-plane rotation keeps each camera's global +X/+Y screen directions stable across
    its views, which is what the single-correspondence orientation prior leans on (and what
    real operators do — they tilt the board, they do not spin or flip it)."""
    if view == 0:
        return cv2.Rodrigues(np.zeros(3))[0], np.zeros(3)
    rvec = np.array(
        [
            rng.uniform(-0.30, 0.30),  # tilt about X (out of plane)
            rng.uniform(-0.30, 0.30),  # tilt about Y (out of plane)
            rng.uniform(-0.08, 0.08),  # in-plane spin — deliberately small
        ]
    )
    tb = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-15, 15)])
    return cv2.Rodrigues(rvec)[0], tb


def _make_dataset(seed: int = 0):
    """Build detections_by_cam, the truth global indices, and the pixel lookup."""
    rng = np.random.default_rng(seed)
    board_poses = [_board_pose(v, rng) for v in range(_N_VIEWS)]

    detections = {c: [] for c in _CAM_POSES}
    truth = {}  # (cam,view) -> (M,2) global indices in point order
    pixel_of = {}  # (cam,view) -> {(gx,gy): (x,y)}
    for cam, cols in _CAM_WINDOWS.items():
        rvec_c, tvec_c = _CAM_POSES[cam]
        for v in range(_N_VIEWS):
            Rb, tb = board_poses[v]
            gidx = np.array([[gx, gy] for gx in cols for gy in _ROWS], dtype=np.int64)
            board_mm = np.column_stack(
                [gidx[:, 0] * SPACING, gidx[:, 1] * SPACING, np.zeros(len(gidx))]
            )
            world = (Rb @ board_mm.T).T + tb
            px = cv2.projectPoints(world, rvec_c, tvec_c, K, None)[0].reshape(-1, 2)

            orient = _ORIENTATIONS[int(rng.integers(len(_ORIENTATIONS)))]
            offset = (int(rng.integers(-5, 5)), int(rng.integers(-5, 5)))
            local = _orient_map(gidx, orient, offset)
            local_mm = np.column_stack(
                [local[:, 0] * SPACING, local[:, 1] * SPACING, np.zeros(len(local))]
            )
            detections[cam].append(
                DetectionResult(
                    success=True,
                    board_type="dotboard",
                    image_points=px,
                    board_local_points=local_mm,
                    grid_indices=local,
                    spacing_mm=SPACING,
                )
            )
            truth[(cam, v)] = gidx
            pixel_of[(cam, v)] = {
                (int(gx), int(gy)): px[i] for i, (gx, gy) in enumerate(gidx)
            }
    return detections, truth, pixel_of


def _spec(pixel_of) -> GlobalGridSpec:
    """The realistic click record: 3-click datum, 1-click within-camera, 2-click cross-camera."""
    p = pixel_of
    datum_clicks = {
        "origin": p[(1, 0)][(0, 0)],
        "x_axis": p[(1, 0)][(1, 0)],
        "y_axis": p[(1, 0)][(0, 1)],
        "origin_mm": [0.0, 0.0],
    }
    anchors = []
    # Camera 1 other views: the global origin dot is in-window -> one click each.
    for v in range(1, _N_VIEWS):
        anchors.append(Anchor(1, v, [Correspondence(p[(1, v)][(0, 0)], "origin")]))
    # Camera 2 first view: two overlap dots shared with cam1 view0 (cols 4..6). Their delta
    # (1,3) has distinct non-zero components, so it fixes orientation outright (not diagonal).
    anchors.append(
        Anchor(
            2,
            0,
            [
                Correspondence(p[(2, 0)][(4, 0)], (1, 0), p[(1, 0)][(4, 0)]),
                Correspondence(p[(2, 0)][(5, 3)], (1, 0), p[(1, 0)][(5, 3)]),
            ],
        )
    )
    # Camera 2 other views: one shared dot with cam2 view0 (origin not in cam2's window).
    for v in range(1, _N_VIEWS):
        anchors.append(
            Anchor(2, v, [Correspondence(p[(2, v)][(4, 0)], (2, 0), p[(2, 0)][(4, 0)])])
        )
    # Camera 3 first view: two overlap dots shared with cam2 view0 (cols 7..9).
    anchors.append(
        Anchor(
            3,
            0,
            [
                Correspondence(p[(3, 0)][(7, 0)], (2, 0), p[(2, 0)][(7, 0)]),
                Correspondence(p[(3, 0)][(9, 3)], (2, 0), p[(2, 0)][(9, 3)]),
            ],
        )
    )
    for v in range(1, _N_VIEWS):
        anchors.append(
            Anchor(3, v, [Correspondence(p[(3, v)][(8, 0)], (3, 0), p[(3, 0)][(8, 0)])])
        )
    return GlobalGridSpec(
        datum_camera=1, datum_view=0, datum_clicks=datum_clicks, anchors=anchors
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_resolve_global_grid_multicam(seed):
    """Every dot in every (camera, view) recovers its true global index, across seeds."""
    detections, truth, pixel_of = _make_dataset(seed)
    resolved = resolve_global_grid(detections, _spec(pixel_of), spacing_mm=SPACING)
    assert set(resolved) == set(truth)
    for key, g_true in truth.items():
        np.testing.assert_array_equal(
            resolved[key], g_true, err_msg=f"global index mismatch at {key}"
        )


def test_cross_camera_dots_agree():
    """Two cameras seeing the same physical dot must assign it the same global index."""
    detections, truth, pixel_of = _make_dataset(7)
    resolved = resolve_global_grid(detections, _spec(pixel_of), spacing_mm=SPACING)
    # Dot (5,3) is in both cam1 (cols 0..6) and cam2 (cols 4..9) windows, view 0.
    g1 = resolved[(1, 0)][list(map(tuple, truth[(1, 0)])).index((5, 3))]
    g2 = resolved[(2, 0)][list(map(tuple, truth[(2, 0)])).index((5, 3))]
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(g1, [5, 3])


def test_inconsistent_correspondences_raise():
    """A cross-camera link whose two clicks name different physical dots must raise."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    # Corrupt cam2 view0's second correspondence to point at the WRONG dot in cam1.
    bad = [a for a in spec.anchors if a.camera == 2 and a.view == 0][0]
    bad.correspondences[1] = Correspondence(
        pixel_of[(2, 0)][(6, 2)], (1, 0), pixel_of[(1, 0)][(4, 5)]  # mismatched ref dot
    )
    with pytest.raises(ValueError):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)


def test_broken_chain_raises():
    """An anchor whose reference is never resolved (disconnected) must raise, not hang."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    # Reference a camera (9) that does not exist -> never resolves -> broken chain.
    spec.anchors.append(
        Anchor(
            3,
            1,
            [
                Correspondence(
                    pixel_of[(3, 1)][(8, 0)], (9, 9), pixel_of[(3, 0)][(8, 0)]
                )
            ],
        )
    )
    with pytest.raises(ValueError):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)


def test_snap_distance_gate_rejects_offgrid_click():
    """A datum/anchor click far from any dot must raise, not snap to a distant dot (H4)."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    # Move the +X datum click way off the grid (1000 px away from any dot).
    spec.datum_clicks["x_axis"] = pixel_of[(1, 0)][(1, 0)] + np.array([1000.0, 1000.0])
    with pytest.raises(ValueError, match="click is"):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)


def test_failed_detection_in_anchor_raises():
    """A non-datum anchor view that failed detection must raise a clear error (M5/H3)."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    detections[2][1]
    detections[2][1] = DetectionResult(
        success=False,
        board_type="dotboard",
        image_points=np.empty((0, 2)),
        board_local_points=np.empty((0, 3)),
    )
    with pytest.raises(ValueError, match="failed detection"):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)


def test_out_of_range_datum_raises():
    """An out-of-range datum view index raises ValueError, not IndexError (M5)."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    spec.datum_view = 99
    with pytest.raises(ValueError, match="out of range"):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)


def test_self_referential_anchor_raises():
    """An anchor that references its own (camera, view) raises (M4)."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    spec.anchors.append(
        Anchor(
            2,
            1,
            [
                Correspondence(
                    pixel_of[(2, 1)][(4, 0)], (2, 1), pixel_of[(2, 1)][(4, 0)]
                )
            ],
        )
    )
    with pytest.raises(ValueError, match="references itself"):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)


def test_correspondence_missing_ref_pixel_raises():
    """A non-origin correspondence without ref_pixel raises at construction (L4)."""
    with pytest.raises(ValueError, match="ref_pixel"):
        Correspondence([1.0, 2.0], (1, 0))


def test_empty_anchor_raises():
    """An anchor with no correspondences raises at construction."""
    with pytest.raises(ValueError, match="no correspondences"):
        Anchor(2, 1, [])


def test_mixed_board_types_raise():
    """A dataset mixing ChArUco (ids) and dotboard (no ids) detections raises (H3)."""
    detections, truth, pixel_of = _make_dataset(0)
    # Give one camera-view global ids -> mixed set.
    d = detections[1][0]
    detections[1][0] = DetectionResult(
        success=True,
        board_type="charuco",
        image_points=d.image_points,
        board_local_points=d.board_local_points,
        grid_indices=d.grid_indices,
        point_ids=np.arange(d.n),
        spacing_mm=SPACING,
    )
    with pytest.raises(ValueError, match="mixed"):
        resolve_global_grid(detections, _spec(pixel_of), spacing_mm=SPACING)


def test_charuco_failed_view_skipped():
    """A failed ChArUco view is skipped, not raised on: one bad frame must not abort the grid —
    the good view resolves and the failed one is simply omitted (the solve uses what detected).
    """
    rng = np.random.default_rng(0)
    detections = {1: []}
    ids = np.arange(20)
    grid = np.column_stack([ids % 5, ids // 5]).astype(np.int64)
    detections[1].append(
        DetectionResult(
            success=True,
            board_type="charuco",
            image_points=rng.uniform(0, 1000, (20, 2)),
            board_local_points=np.column_stack([grid * SPACING, np.zeros(20)]),
            grid_indices=grid,
            point_ids=ids,
            spacing_mm=SPACING,
        )
    )
    detections[1].append(
        DetectionResult(
            success=False,
            board_type="charuco",
            image_points=np.empty((0, 2)),
            board_local_points=np.empty((0, 3)),
            point_ids=np.empty((0,)),
        )
    )
    out = resolve_global_grid(detections)
    assert (1, 0) in out  # the good view resolved
    assert (1, 1) not in out  # the failed view was skipped, not raised on


def test_charuco_shortcircuit():
    """ChArUco detections (global corner ids) resolve with no spec and no clicks."""
    rng = np.random.default_rng(0)
    detections = {1: [], 2: []}
    for cam in (1, 2):
        for v in range(2):
            ids = np.arange(20)
            grid = np.column_stack([ids % 5, ids // 5]).astype(np.int64)
            px = rng.uniform(0, 1000, size=(20, 2))
            detections[cam].append(
                DetectionResult(
                    success=True,
                    board_type="charuco",
                    image_points=px,
                    board_local_points=np.column_stack([grid * SPACING, np.zeros(20)]),
                    grid_indices=grid,
                    point_ids=ids,
                    spacing_mm=SPACING,
                )
            )
    resolved = resolve_global_grid(detections)  # no spec needed
    for cam in (1, 2):
        for v in range(2):
            np.testing.assert_array_equal(
                resolved[(cam, v)], detections[cam][v].grid_indices
            )


# ---------------------------------------------------------------------------
# Partial (non-raising) resolver — drives the GUI live overlay while clicking.
# It must agree with the strict resolver on a complete spec, and degrade gracefully
# (resolve what it can, report a per-view reason for the rest) on a partial/bad one.
# ---------------------------------------------------------------------------


def test_partial_matches_strict_on_complete_spec():
    """On a fully-specified, valid spec the partial resolver returns exactly the strict result
    and an empty unresolved list — the overlay and the solve see the same grid."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    strict = resolve_global_grid(detections, spec, spacing_mm=SPACING)
    resolved, unresolved = resolve_global_grid_partial(
        detections, spec, spacing_mm=SPACING
    )
    assert unresolved == []
    assert set(resolved) == set(strict)
    for k in strict:
        np.testing.assert_array_equal(resolved[k], strict[k])


def test_partial_resolves_datum_only_with_no_anchors():
    """A datum-only spec (no links yet) resolves just the datum view, no spurious failures —
    the first thing the user sees after the origin/+X/+Y clicks."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    spec.anchors = []
    resolved, unresolved = resolve_global_grid_partial(
        detections, spec, spacing_mm=SPACING
    )
    assert set(resolved) == {(1, 0)}
    assert unresolved == []


def test_partial_isolates_a_misclicked_anchor():
    """One mis-clicked anchor is dropped with a reason; the datum and the other cameras (which
    don't depend on it) stay resolved — a bad click never blanks the whole overlay."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    # Corrupt the cam1 view1 anchor to an off-grid click (image corner, far from any dot).
    spec.anchors = [
        (
            Anchor(1, 1, [Correspondence([3.0, 3.0], "origin")])
            if (a.camera, a.view) == (1, 1)
            else a
        )
        for a in spec.anchors
    ]
    resolved, unresolved = resolve_global_grid_partial(
        detections, spec, spacing_mm=SPACING
    )
    assert (1, 0) in resolved  # datum unaffected
    assert (1, 1) not in resolved  # the bad anchor's view is dropped
    assert (2, 0) in resolved and (
        3,
        0,
    ) in resolved  # independent cameras still resolve
    bad = [(c, v, r) for c, v, r in unresolved if (c, v) == (1, 1)]
    assert len(bad) == 1 and "click is" in bad[0][2]


def test_partial_flags_broken_chain_without_raising():
    """Anchors whose reference camera was never linked are reported as unresolved (not connected),
    not raised — and never appear in the resolved grid."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    # Drop every cam2 anchor: cam2 never resolves, so cam3's links (which reference cam2) dangle.
    spec.anchors = [a for a in spec.anchors if a.camera != 2]
    resolved, unresolved = resolve_global_grid_partial(
        detections, spec, spacing_mm=SPACING
    )
    assert (1, 0) in resolved and (1, 3) in resolved  # cam1 fully resolved
    assert all(
        c != 2 for c, _ in [(k[0], k[1]) for k in resolved]
    )  # cam2 absent (no anchors)
    cam3 = [(c, v, r) for c, v, r in unresolved if c == 3]
    assert cam3 and all("not connected" in r for _, _, r in cam3)
    assert all((c, v) not in resolved for c, v, _ in unresolved)


def test_partial_bad_datum_returns_empty_with_reason():
    """A datum that doesn't resolve is terminal: empty grid + one reason against the datum view
    (the GUI shows the dots and 'fix origin/+X/+Y')."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)
    spec.datum_clicks = {
        **spec.datum_clicks,
        "origin": [3.0, 3.0],
    }  # off-grid origin click
    resolved, unresolved = resolve_global_grid_partial(
        detections, spec, spacing_mm=SPACING
    )
    assert resolved == {}
    assert len(unresolved) == 1 and unresolved[0][0] == 1 and unresolved[0][1] == 0


# ---------------------------------------------------------------------------
# Collinear cross-camera bridge + rig arrangement hint
#
# When a new camera overlaps the resolved grid in a single collinear strip (one column or row —
# common for side-by-side cameras that tile a wide board), the two clicks pin translation but
# leave a mirror-image orientation ambiguity, and the mirror FOLD fits the dots exactly as well
# as the truth (a relabelled lattice is still a lattice). No dot-based test can separate them; a
# coarse rig direction (which way this camera's coverage extends) is the one external bit that can.
# ---------------------------------------------------------------------------


def _collinear_bridge_spec(pixel_of) -> GlobalGridSpec:
    """Datum + cam1 views + a cam2 view0 bridge of two dots in the SAME column (collinear)."""
    p = pixel_of
    datum_clicks = {
        "origin": p[(1, 0)][(0, 0)],
        "x_axis": p[(1, 0)][(1, 0)],
        "y_axis": p[(1, 0)][(0, 1)],
        "origin_mm": [0.0, 0.0],
    }
    anchors = [
        Anchor(1, v, [Correspondence(p[(1, v)][(0, 0)], "origin")])
        for v in range(1, _N_VIEWS)
    ]
    # Two shared dots in column gx=4 (delta (0,3) — collinear), so orientation is ambiguous from
    # the clicks alone: the gx-axis sign (which way cam2 extends) is unconstrained.
    anchors.append(
        Anchor(
            2,
            0,
            [
                Correspondence(p[(2, 0)][(4, 0)], (1, 0), p[(1, 0)][(4, 0)]),
                Correspondence(p[(2, 0)][(4, 3)], (1, 0), p[(1, 0)][(4, 3)]),
            ],
        )
    )
    return GlobalGridSpec(
        datum_camera=1, datum_view=0, datum_clicks=datum_clicks, anchors=anchors
    )


def test_collinear_bridge_is_ambiguous_without_hint():
    """A single-column overlap leaves a mirror ambiguity the clicks can't resolve — raise, not guess."""
    detections, _truth, pixel_of = _make_dataset(0)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_global_grid(
            detections, _collinear_bridge_spec(pixel_of), spacing_mm=SPACING
        )


def test_collinear_bridge_resolved_by_extend_hint():
    """The correct rig direction (cam2 extends toward +X, into its exclusive columns) recovers
    the true global indices from the same single-column overlap."""
    detections, truth, pixel_of = _make_dataset(0)
    spec = _collinear_bridge_spec(pixel_of)
    spec.camera_extends = {
        2: (1.0, 0.0)
    }  # cam2 sees cols 4..9 — its coverage extends +X
    resolved = resolve_global_grid(detections, spec, spacing_mm=SPACING)
    np.testing.assert_array_equal(resolved[(2, 0)], truth[(2, 0)])


def test_extend_hint_pointing_the_wrong_way_picks_the_fold():
    """The hint genuinely decides it: point it the wrong way and the resolver takes the mirror
    fold (not the truth) — proof the disambiguation rests on the hint, nothing is guessed.
    """
    detections, truth, pixel_of = _make_dataset(0)
    spec = _collinear_bridge_spec(pixel_of)
    spec.camera_extends = {2: (-1.0, 0.0)}
    resolved = resolve_global_grid(detections, spec, spacing_mm=SPACING)
    assert not np.array_equal(resolved[(2, 0)], truth[(2, 0)])


def test_orientation_candidates_expose_both_layouts_for_the_picker():
    """The confirm-on-overlay picker gets exactly the two mirror layouts a collinear bridge
    allows — opposite extend directions, equal footprints, equal fit — to render for the user.
    """
    from pivtools_gui.calibration.global_grid import first_view_orientation_candidates

    detections, _truth, pixel_of = _make_dataset(0)
    spec = _collinear_bridge_spec(pixel_of)
    cands = first_view_orientation_candidates(
        detections, spec, 2, 0, spacing_mm=SPACING
    )
    assert len(cands) == 2
    ex = sorted(c["extend"][0] for c in cands)
    assert ex[0] < 0 < ex[1]  # one extends -X, the other +X (the fold)
    assert (
        abs(cands[0]["rms"] - cands[1]["rms"]) < 1e-6
    )  # fits are equal — dots can't decide


def test_picking_a_candidate_extend_resolves_that_layout():
    """Feeding a candidate's `extend` back as camera_extends reproduces that layout — the pick
    persists and the headless solve is deterministic."""
    from pivtools_gui.calibration.global_grid import first_view_orientation_candidates

    detections, truth, pixel_of = _make_dataset(0)
    spec = _collinear_bridge_spec(pixel_of)
    cands = first_view_orientation_candidates(
        detections, spec, 2, 0, spacing_mm=SPACING
    )
    truth_max = int(truth[(2, 0)][:, 0].max())
    correct = next(
        c for c in cands if c["gx_range"][1] == truth_max
    )  # the layout matching truth
    spec.camera_extends = {2: tuple(correct["extend"])}
    resolved = resolve_global_grid(detections, spec, spacing_mm=SPACING)
    np.testing.assert_array_equal(resolved[(2, 0)], truth[(2, 0)])


def test_no_candidates_when_bridge_is_unambiguous():
    """A non-collinear bridge resolves outright, so the picker is never shown (empty list)."""
    from pivtools_gui.calibration.global_grid import first_view_orientation_candidates

    detections, _truth, pixel_of = _make_dataset(0)
    spec = _spec(pixel_of)  # the realistic 2-non-collinear-dot bridge
    assert (
        first_view_orientation_candidates(detections, spec, 2, 0, spacing_mm=SPACING)
        == []
    )


def test_extend_hint_along_ambiguous_axis_still_raises():
    """A hint pointing ALONG the collinear strip (here +Y, the click axis) can't separate the
    mirror candidates — the resolver must still refuse rather than pick arbitrarily."""
    detections, _truth, pixel_of = _make_dataset(0)
    spec = _collinear_bridge_spec(pixel_of)
    spec.camera_extends = {
        2: (0.0, 1.0)
    }  # along the shared column — does not break the tie
    with pytest.raises(ValueError, match="does not clearly pick|ambiguous"):
        resolve_global_grid(detections, spec, spacing_mm=SPACING)
