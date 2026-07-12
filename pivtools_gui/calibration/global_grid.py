"""calibration.global_grid — one global dot index across all cameras and views.

Dotboard grid indices are LOCAL per view: each view's BFS walk normalises to its own
min-corner (``grid_detection.py``), so the same physical dot carries different indices
in different views and cameras. The joint shared-board solve needs the opposite — one
GLOBAL index ``(gx, gy)`` per physical dot, identical everywhere it is seen. This module
assigns that, tied together by a few user clicks (each snapped to the detected dot
centroid, so a link names the *same physical dot* to sub-pixel precision):

  - **datum view**: origin / +X / +Y clicks fix the global axes and origin (3 clicks).
    Reuses ``world_frame.resolve_world_frame`` — the global index of a datum dot is just
    its grid index re-expressed through that frame.
  - **every other view**: one or more reference-dot correspondences to an already-anchored
    view. One correspondence pins the translation; orientation (the signed permutation of
    the two grid axes) is then resolved by the planar homography that best maps candidate
    global positions to the detected pixels, broken by a same-camera orientation prior.

Orientation ambiguity. A regular dot lattice is symmetric: a 180-degree relabel of a view
(and, for a square-visible patch, a 90-degree one) still fits a homography, so a *single*
correspondence cannot fully fix orientation. We break it two ways, never silently:

  - within one camera, the operator does not flip the board between shots, so we prefer the
    candidate whose image-space axis directions agree with that camera's already-resolved
    view (the orientation prior). When the prior does not clearly favour one candidate
    (low agreement, or a near tie), we RAISE rather than guess — the caller then asks for an
    explicit anchor click on that view (the auto-resolve-with-click-on-uncertainty design).
  - linking a *new* camera needs >= 2 non-collinear correspondences, which determine the
    signed permutation outright (no prior).

Consistency guarantees and their limit (be honest). The per-view homography RMS catches a
mis-click that lands off the grid or scrambles the lattice; the snap-distance gate catches a
click not near any dot; the multi-link check catches a view whose several correspondences
disagree. What this module structurally CANNOT catch is a *coherent in-grid translation*
mis-click — both reference clicks snapped to the wrong-but-self-consistent dots — because a
homography re-fits its own translation freely. That residual case is caught downstream: the
joint solve's single-rigid-board residual blows up, and the GUI overlay shows the wrong
numbers. So this module raises on everything it can see and never *silently* mislabels, but
"raises on every possible mis-click" is not claimed.

ChArUco needs none of this: corner ids are already globally consistent, so the grid indices a
ChArUco detection carries ARE the global indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .detection.base import DetectionResult
from .record import WorldFrame
from .world_frame import resolve_world_frame

# A view is addressed by (camera, view_index); view_index is 0-based within a camera.
ViewKey = Tuple[int, int]
# same_as target: another view, or the literal global origin.
SameAs = Union[ViewKey, str]

# The eight signed permutations of two grid axes (swap, x-sign, y-sign).
_ORIENTATIONS: Tuple[Tuple[bool, int, int], ...] = tuple(
    (swap, sx, sy) for swap in (False, True) for sx in (1, -1) for sy in (1, -1)
)

# Default reprojection tolerance (px) for a planar homography over correctly labelled dots.
# Correct labelling fits to sub-pixel (plus a little lens distortion); a wrong labelling
# misses by tens to hundreds of px, so the exact value is not delicate.
DEFAULT_CONSISTENCY_TOL_PX = 3.0
# A click must snap to a dot within this fraction of the local dot spacing, else it is
# treated as a mis-click and raises (rather than snapping to the wrong neighbour).
SNAP_FRACTION = 0.5
# Orientation-prior trust gates: the winning candidate must agree with the prior at least
# this well (dot product of unit axis directions, summed over the two axes; max +2), and beat
# the runner-up by at least this margin — else the prior is deemed unreliable and we raise.
AGREE_FLOOR = 0.4
AGREE_MARGIN = 0.25
# Rig-arrangement (extend-hint) trust gates: the winning candidate's grid must extend along the
# hint direction at least this well (cosine of the centroid-from-anchor vector against the hint;
# max +1), and beat the runner-up by this margin — else the hint lies along the ambiguous axis
# and cannot separate the candidates, so we raise rather than guess.
EXTEND_FLOOR = 0.15
EXTEND_MARGIN = 0.30
# Fallback footprint-match gate (grid-index units): when two folds extend the SAME direction the
# cosine test above ties, but a confirmed candidate's stored vector still reproduces its OWN
# centroid exactly while the fold lands ≥1 column off. The nearest candidate must beat the runner-up
# by this many grid units, else the choice is a genuine tie and we raise.
EXTEND_VEC_MARGIN = 0.5


@dataclass
class Correspondence:
    """One assertion: ``pixel`` in this view is the same physical dot as a known one.

    ``same_as`` is either another ``(camera, view)`` whose global indices are already
    resolved (then ``ref_pixel`` locates the same physical dot there, snapped to its
    centroid), or the string ``"origin"`` (the dot is the global origin, index (0, 0)).
    """

    pixel: Sequence[float]
    same_as: SameAs
    ref_pixel: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.same_as != "origin":
            if (not isinstance(self.same_as, tuple)) or len(self.same_as) != 2:
                raise ValueError(
                    f"Correspondence.same_as must be 'origin' or a (camera, view) tuple, "
                    f"got {self.same_as!r}"
                )
            if self.ref_pixel is None:
                raise ValueError(
                    f"Correspondence to {self.same_as} needs ref_pixel (the same physical "
                    f"dot in that view)"
                )


@dataclass
class Anchor:
    """How one non-datum view ties into the global grid.

    ``correspondences`` holds one entry for a within-camera view (orientation resolved by the
    same-camera prior) or two-or-more for a new camera's first view (orientation determined
    outright). An empty list means "auto-resolve" — left to the GUI layer (S2·P7), which
    fills it or flags the view for a click; this module requires at least one.
    """

    camera: int
    view: int
    correspondences: list

    def __post_init__(self) -> None:
        if not self.correspondences:
            raise ValueError(
                f"Anchor ({self.camera},{self.view}) has no correspondences (auto-resolve is "
                f"handled by the GUI layer, not resolve_global_grid)"
            )


@dataclass
class GlobalGridSpec:
    """The full click record needed to rebuild the global grid headlessly."""

    datum_camera: int
    datum_view: int
    datum_clicks: dict  # {origin, x_axis, y_axis, origin_mm}
    anchors: list = field(default_factory=list)
    # Optional coarse rig arrangement: {camera: (dx, dy)} unit-ish direction, in GLOBAL index
    # space, along which that camera's coverage extends relative to the cameras it bridges to.
    # Used only to break the lattice-symmetry tie for a camera's first view when the overlap is a
    # single collinear strip (clicks alone leave a mirror ambiguity, and the mirror fold fits the
    # dots just as well — only this external direction distinguishes them). Empty = not provided.
    camera_extends: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Snapping (with a distance gate)
# ---------------------------------------------------------------------------


def _local_spacing_px(pts: np.ndarray) -> float:
    """Median nearest-neighbour distance among detected points (the dot pitch in px)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.inf
    from scipy.spatial import cKDTree

    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def _snap(pts: np.ndarray, click_xy, max_dist: float, what: str) -> int:
    """Index of the detected dot nearest the click, raising if none is within ``max_dist``."""
    from scipy.spatial import cKDTree

    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        raise ValueError(
            f"global_grid: cannot snap {what} — no detected dots in this view"
        )
    dist, idx = cKDTree(p).query(np.asarray(click_xy, dtype=np.float64).reshape(2))
    if dist > max_dist:
        raise ValueError(
            f"global_grid: {what} click is {dist:.1f}px from the nearest dot "
            f"(> {max_dist:.1f}px gate) — likely a mis-click, not snapping to a far dot"
        )
    return int(idx)


# ---------------------------------------------------------------------------
# Index algebra
# ---------------------------------------------------------------------------


def _global_from_frame(grid_indices: np.ndarray, wf: WorldFrame) -> np.ndarray:
    """Datum view: re-express local grid indices as global (gx, gy) through ``wf``.

    Same swap/sign logic as ``world_frame.apply_world_frame`` but in integer index units
    (no spacing, no origin_mm) — the global index is what every other view aligns to.
    """
    gi = np.asarray(grid_indices, dtype=np.float64).reshape(-1, 2)
    og = np.asarray(wf.origin_grid, dtype=np.float64).reshape(2)
    du = gi[:, 0] - og[0]
    dv = gi[:, 1] - og[1]
    if not wf.swap_axes:
        gx = wf.col_sign * du
        gy = wf.row_sign * dv
    else:
        gx = wf.col_sign * dv
        gy = wf.row_sign * du
    return np.rint(np.column_stack([gx, gy])).astype(np.int64)


def _map_indices(
    grid_indices: np.ndarray,
    local_anchor: np.ndarray,
    global_anchor: Tuple[int, int],
    orientation: Tuple[bool, int, int],
) -> np.ndarray:
    """Map a view's local grid indices to global indices under one orientation.

    Translation is pinned by ``local_anchor`` <-> ``global_anchor``; ``orientation`` is the
    (swap, x-sign, y-sign) signed permutation of the two grid axes.
    """
    swap, sx, sy = orientation
    gi = np.asarray(grid_indices, dtype=np.float64).reshape(-1, 2)
    du = gi[:, 0] - float(local_anchor[0])
    dv = gi[:, 1] - float(local_anchor[1])
    if not swap:
        gx = global_anchor[0] + sx * du
        gy = global_anchor[1] + sy * dv
    else:
        gx = global_anchor[0] + sx * dv
        gy = global_anchor[1] + sy * du
    return np.rint(np.column_stack([gx, gy])).astype(np.int64)


def _homography_fit(
    global_idx: np.ndarray, image_points: np.ndarray, spacing_mm: float
) -> Tuple[float, Optional[np.ndarray]]:
    """RMS (px) of the best planar homography mapping nominal global positions to pixels.

    Correct labelling -> a real board-plane-to-image homography -> sub-pixel RMS. A wrong
    labelling has no consistent homography -> large RMS. Least-squares over ALL points (not
    RANSAC): we want every dot to fit, and an inconsistent labelling to be exposed. Returns
    ``inf`` for a degenerate set (fewer than 4 dots, or all dots on one global row/column, or
    a non-finite homography) so such a view never passes the consistency gate by accident.
    """
    gi = np.asarray(global_idx, dtype=np.int64).reshape(-1, 2)
    if len(gi) < 4:
        return np.inf, None
    if (gi[:, 0].max() - gi[:, 0].min()) < 1 or (gi[:, 1].max() - gi[:, 1].min()) < 1:
        return np.inf, None  # collinear in index space -> no 2D homography
    src = gi.astype(np.float64) * float(spacing_mm)
    dst = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None or not np.isfinite(H).all():
        return np.inf, None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(((proj - dst) ** 2).sum(axis=1).mean()))
    return rms, H


def _axis_screen_dirs(
    H: np.ndarray, global_idx: np.ndarray, spacing_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Image-space unit directions of global +X and +Y, evaluated at the patch centroid.

    Under perspective the axis screen-direction varies across the board, so we sample it at
    the centroid of the actually-visible dots (not from the raw homography columns, which give
    the direction at the board origin and can sit far outside a partial window).
    """
    gc = np.asarray(global_idx, dtype=np.float64).reshape(-1, 2).mean(axis=0)
    base = np.array([gc, gc + [1.0, 0.0], gc + [0.0, 1.0]]) * float(spacing_mm)
    proj = cv2.perspectiveTransform(base.reshape(-1, 1, 2), H).reshape(-1, 2)

    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    return _unit(proj[1] - proj[0]), _unit(proj[2] - proj[0])


# ---------------------------------------------------------------------------
# Per-view orientation resolution
# ---------------------------------------------------------------------------


def _orientations_satisfying(local_anchors: list, global_anchors: list) -> list:
    """Orientations whose index map reproduces every correspondence delta.

    One correspondence constrains nothing (all eight survive — it only pins translation).
    Two-plus correspondences each demand that the local index delta map to the global one; a
    *non-degenerate* pair (delta with distinct, non-zero components) leaves exactly one, while
    a degenerate pair (e.g. a diagonal delta, which is swap-symmetric) leaves several. Returns
    the full surviving set so the caller can disambiguate or report ambiguity.
    """
    la = np.asarray(local_anchors, dtype=np.float64)
    ga = np.asarray(global_anchors, dtype=np.float64)
    if len(la) < 2:
        return list(_ORIENTATIONS)
    out = []
    base_global = (int(round(ga[0, 0])), int(round(ga[0, 1])))
    for orient in _ORIENTATIONS:
        ok = True
        for k in range(1, len(la)):
            mapped = _map_indices(la[k].reshape(1, 2), la[0], base_global, orient)[0]
            if not np.array_equal(mapped, np.rint(ga[k]).astype(np.int64)):
                ok = False
                break
        if ok:
            out.append(orient)
    return out


def _pick_by_extend(
    scored: list, anchor_global: Sequence[float], extend_hint: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the candidate the user confirmed, given ``extend_hint`` — the global-index vector from
    the anchored dot to a candidate's centroid (see ``first_view_orientation_candidates``).

    Two readings of the hint, tried in order:
      1. DIRECTION. Mirror folds across a clean axis put their centroids on opposite sides of the
         anchor, so a coarse ``±X``/``±Y`` hint separates them by the sign of the cosine. This is
         the original, forgiving reading — a hand-given rig direction need not match the footprint.
      2. NEAREST FOOTPRINT. When both candidates extend the SAME way (e.g. a seam that folds along
         one column while both halves still run −Y), their directions coincide and step 1 ties. The
         confirmed candidate's stored vector still reproduces its own centroid exactly, while the
         fold lands a column off, so the full-vector distance separates them where direction cannot.
    A hint that separates the candidates under neither reading points along the genuinely ambiguous
    axis, so we raise rather than guess. ``scored`` is the (orientation, gidx, H, rms) survivors.
    """
    h = np.asarray(extend_hint, dtype=np.float64).reshape(2)
    nh = float(np.linalg.norm(h))
    if nh < 1e-9:
        raise ValueError(
            "global_grid: rig arrangement hint is a zero vector — give a direction"
        )
    ag = np.asarray(anchor_global, dtype=np.float64).reshape(2)
    # Each candidate's centroid-from-anchor vector, in global-index space (the same quantity
    # first_view_orientation_candidates stored as the candidate's `extend`).
    cands = [
        (gidx.astype(np.float64).mean(axis=0) - ag, gidx, H, rms)
        for _orient, gidx, H, rms in scored
    ]

    # 1. Direction: align each candidate's extent with the hint direction (cosine).
    hd = h / nh
    by_dir = sorted(
        (
            (
                float(np.dot(v / max(float(np.linalg.norm(v)), 1e-9), hd)),
                rms,
                gidx,
                H,
            )
            for v, gidx, H, rms in cands
        ),
        key=lambda s: (-s[0], s[1]),
    )
    best = by_dir[0][0]
    runner = by_dir[1][0] if len(by_dir) > 1 else -np.inf
    if best >= EXTEND_FLOOR and (best - runner) >= EXTEND_MARGIN:
        return by_dir[0][2], by_dir[0][3]

    # 2. Nearest footprint: the confirmed vector reproduces one candidate's centroid offset exactly.
    by_vec = sorted(
        ((float(np.linalg.norm(v - h)), rms, gidx, H) for v, gidx, H, rms in cands),
        key=lambda s: (s[0], s[1]),
    )
    nearest = by_vec[0][0]
    next_nearest = by_vec[1][0] if len(by_vec) > 1 else np.inf
    if (next_nearest - nearest) >= EXTEND_VEC_MARGIN:
        return by_vec[0][2], by_vec[0][3]

    raise ValueError(
        f"global_grid: the rig arrangement hint does not clearly pick an orientation "
        f"(direction margin {best - runner:.2f}, footprint margin "
        f"{next_nearest - nearest:.2f} grid units) — the hint points along the ambiguous "
        f"axis; click >= 2 overlap dots spanning distinct rows AND columns instead"
    )


def _resolve_view(
    det: DetectionResult,
    correspondences: list,
    spacing_mm: float,
    prior: Optional[Tuple[np.ndarray, np.ndarray]],
    tol_px: float,
    extend_hint: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve one view's global indices.

    ``correspondences`` is a list of (local_index, global_index). Candidate orientations are
    those consistent with the correspondence deltas; each is scored by its planar-homography
    RMS and only those within ``tol_px`` kept. One survivor -> use it. Several -> break the tie
    by trust order: the same-camera ``prior`` (within a camera the board is not flipped between
    shots), else a coarse ``extend_hint`` (the rig direction this camera's coverage extends,
    for a new camera's first view where there is no prior and the overlap may be one collinear
    strip). With neither — or when the chosen signal does not clearly separate the candidates —
    raise, so a mis-click or genuine ambiguity is reported, never guessed. Returns
    (global_indices (N,2), H).
    """
    local_anchors = [c[0] for c in correspondences]
    global_anchors = [c[1] for c in correspondences]

    cand_orients = _orientations_satisfying(local_anchors, global_anchors)
    if not cand_orients:
        raise ValueError(
            "global_grid: overlap correspondences are mutually inconsistent — no single "
            "orientation reproduces all of them (do the clicks name the same physical dots?)"
        )

    la, ga = local_anchors[0], global_anchors[0]
    scored = []
    best_rms = np.inf
    for orient in cand_orients:
        gidx = _map_indices(det.grid_indices, la, ga, orient)
        rms, H = _homography_fit(gidx, det.image_points, spacing_mm)
        best_rms = min(best_rms, rms)
        if np.isfinite(rms) and rms <= tol_px:
            scored.append((orient, gidx, H, rms))

    if not scored:
        raise ValueError(
            f"global_grid: no orientation fits this view within {tol_px}px "
            f"(best homography RMS {best_rms:.2f}px) — likely a mis-clicked reference dot"
        )

    if len(scored) == 1:
        _, gidx, H, _ = scored[0]
        return gidx, H

    # Several orientations fit the dots equally (lattice symmetry; the mirror fold fits as well
    # as the truth). Break the tie by trust order: same-camera prior, then the coarse rig hint.
    if prior is not None:
        px, py = prior
        ranked = []
        for orient, gidx, H, rms in scored:
            gx_dir, gy_dir = _axis_screen_dirs(H, gidx, spacing_mm)
            agree = float(np.dot(gx_dir, px) + np.dot(gy_dir, py))
            ranked.append((agree, rms, gidx, H))
        ranked.sort(key=lambda s: (-s[0], s[1]))

        best_agree = ranked[0][0]
        runner_up = ranked[1][0] if len(ranked) > 1 else -np.inf
        if best_agree < AGREE_FLOOR or (best_agree - runner_up) < AGREE_MARGIN:
            raise ValueError(
                f"global_grid: orientation prior is unreliable here (best agreement "
                f"{best_agree:.2f}, runner-up {runner_up:.2f}) — this view needs an explicit "
                f"anchor click (a steeply different view, or the prior does not separate the "
                f"lattice symmetries)"
            )
        _, _, gidx, H = ranked[0]
        return gidx, H

    if extend_hint is not None:
        return _pick_by_extend(scored, global_anchors[0], extend_hint)

    raise ValueError(
        "global_grid: orientation is ambiguous (regular-lattice symmetry) with no same-camera "
        "prior — for a camera's first view either give a rig arrangement hint (which way this "
        "camera's view extends: +X/-X/+Y/-Y), or click >= 2 overlap dots that are NOT "
        "diagonal/collinear (distinct row and column spans)"
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _detection(detections_by_cam: dict, cam: int, view: int) -> DetectionResult:
    """Fetch a detection with clear ValueErrors for missing/out-of-range/failed views."""
    if cam not in detections_by_cam:
        raise ValueError(f"global_grid: camera {cam} not in detections")
    views = detections_by_cam[cam]
    if not (0 <= view < len(views)):
        raise ValueError(
            f"global_grid: view {view} out of range for camera {cam} (has {len(views)} views)"
        )
    det = views[view]
    if not det.success:
        raise ValueError(f"global_grid: view ({cam},{view}) failed detection")
    if det.grid_indices is None or det.n == 0:
        raise ValueError(f"global_grid: view ({cam},{view}) has no grid indices")
    return det


def _classify_boards(detections_by_cam: dict) -> str:
    """Return 'charuco' if every successful detection has global ids, 'dotboard' if none do.

    Raises on a mixed set (some with ids, some without) — that has no single resolution path.
    """
    with_ids = without_ids = 0
    for views in detections_by_cam.values():
        for d in views:
            if not d.success:
                continue
            if d.point_ids is not None:
                with_ids += 1
            else:
                without_ids += 1
    if with_ids and without_ids:
        raise ValueError(
            "global_grid: mixed detections — some carry global ids (ChArUco), some do not "
            "(dotboard); resolve one board type at a time"
        )
    if with_ids:
        return "charuco"
    return "dotboard"


def global_grid_from_charuco(detections_by_cam: dict) -> dict:
    """ChArUco: grid indices are already global (derived from corner ids).

    Emits one entry per successfully-detected (cam, view). A view that failed detection (or carries
    no grid indices) is skipped, not raised on: one bad frame must not abort the whole grid — the
    solve simply uses the views that detected.
    """
    out = {}
    for cam, views in detections_by_cam.items():
        for v in range(len(views)):
            det = views[v]
            if not det.success or det.grid_indices is None or det.n == 0:
                continue
            out[(cam, v)] = np.asarray(det.grid_indices, dtype=np.int64).reshape(-1, 2)
    return out


def resolve_global_grid(
    detections_by_cam: dict,
    spec: Optional[GlobalGridSpec] = None,
    spacing_mm: Optional[float] = None,
    tol_px: float = DEFAULT_CONSISTENCY_TOL_PX,
) -> dict:
    """Assign a global (gx, gy) index to every detected dot in every (camera, view).

    Parameters
    ----------
    detections_by_cam : {camera: [DetectionResult per view]}
    spec : the click record (datum clicks + per-view anchors). Not needed for ChArUco.
    spacing_mm : board spacing; defaults to the datum detection's ``spacing_mm``.
    tol_px : per-view homography consistency tolerance.

    Returns
    -------
    {(camera, view): (N,2) int global indices}, index-aligned with that detection's points.
    """
    if _classify_boards(detections_by_cam) == "charuco":
        return global_grid_from_charuco(detections_by_cam)

    if spec is None:
        raise ValueError(
            "resolve_global_grid: a GlobalGridSpec is required for dotboard"
        )

    datum_key = (spec.datum_camera, spec.datum_view)
    datum = _detection(detections_by_cam, spec.datum_camera, spec.datum_view)
    sp = float(spacing_mm if spacing_mm is not None else datum.spacing_mm)

    # 1. Datum view -> global axes and origin via the clicked world frame. Gate the three
    #    datum clicks so an off-grid click is caught here rather than silently snapping.
    gate = SNAP_FRACTION * _local_spacing_px(datum.image_points)
    for name in ("origin", "x_axis", "y_axis"):
        _snap(datum.image_points, spec.datum_clicks[name], gate, f"datum {name}")
    wf = resolve_world_frame(datum.grid_indices, datum.image_points, spec.datum_clicks)
    datum_global = _global_from_frame(datum.grid_indices, wf)
    datum_rms, datum_H = _homography_fit(datum_global, datum.image_points, sp)
    if datum_H is None or datum_rms > tol_px:
        raise ValueError(
            f"global_grid: datum view {datum_key} world frame does not fit a planar "
            f"homography (RMS {datum_rms:.2f}px > {tol_px}px) — check the origin/+X/+Y clicks"
        )
    global_index = {datum_key: datum_global}
    # Orientation prior is per-camera: the most recently resolved view of that camera.
    prior_by_cam = {spec.datum_camera: _axis_screen_dirs(datum_H, datum_global, sp)}

    # 2. Resolve anchors, deferring any whose reference views are not yet resolved.
    pending = list(spec.anchors)
    while pending:
        progressed = False
        for a in list(pending):
            if (a.camera, a.view) == datum_key:
                raise ValueError(
                    f"global_grid: anchor ({a.camera},{a.view}) duplicates the datum view"
                )
            det = _detection(detections_by_cam, a.camera, a.view)
            det_gate = SNAP_FRACTION * _local_spacing_px(det.image_points)

            ready = True
            corr = []
            for c in a.correspondences:
                if c.same_as != "origin" and tuple(c.same_as) == (a.camera, a.view):
                    raise ValueError(
                        f"global_grid: anchor ({a.camera},{a.view}) references itself"
                    )
                i_click = _snap(
                    det.image_points, c.pixel, det_gate, f"anchor ({a.camera},{a.view})"
                )
                local_idx = np.asarray(det.grid_indices[i_click], dtype=np.int64)
                if c.same_as == "origin":
                    g = (0, 0)
                else:
                    ref_key = (int(c.same_as[0]), int(c.same_as[1]))
                    if ref_key not in global_index:
                        ready = False
                        break
                    rdet = _detection(detections_by_cam, ref_key[0], ref_key[1])
                    ref_gate = SNAP_FRACTION * _local_spacing_px(rdet.image_points)
                    j = _snap(
                        rdet.image_points,
                        c.ref_pixel,
                        ref_gate,
                        f"ref dot in {ref_key}",
                    )
                    g = tuple(int(x) for x in global_index[ref_key][j])
                corr.append((local_idx, g))
            if not ready:
                continue

            gidx, H = _resolve_view(
                det,
                corr,
                sp,
                prior_by_cam.get(a.camera),
                tol_px,
                extend_hint=spec.camera_extends.get(a.camera),
            )
            global_index[(a.camera, a.view)] = gidx
            prior_by_cam[a.camera] = _axis_screen_dirs(H, gidx, sp)
            pending.remove(a)
            progressed = True

        if not progressed:
            stuck = [(p.camera, p.view) for p in pending]
            raise ValueError(
                f"global_grid: anchors {stuck} are not connected to the datum through any "
                f"resolved view (broken overlap chain, or a self/forward reference)"
            )

    # 3. Multi-link consistency: every correspondence's clicked dot must end up at the global
    #    index of its reference dot. Tautological for a single-link chain, but catches a view
    #    that is over-constrained by two links that disagree (a coherent single-link shift is
    #    not visible here — see the module docstring; the joint solve + overlay catch it).
    for a in spec.anchors:
        det = detections_by_cam[a.camera][a.view]
        det_gate = SNAP_FRACTION * _local_spacing_px(det.image_points)
        for c in a.correspondences:
            i_click = _snap(
                det.image_points, c.pixel, det_gate, f"recheck ({a.camera},{a.view})"
            )
            got = tuple(int(x) for x in global_index[(a.camera, a.view)][i_click])
            if c.same_as == "origin":
                want = (0, 0)
            else:
                rk = (int(c.same_as[0]), int(c.same_as[1]))
                rdet = detections_by_cam[rk[0]][rk[1]]
                ref_gate = SNAP_FRACTION * _local_spacing_px(rdet.image_points)
                j = _snap(rdet.image_points, c.ref_pixel, ref_gate, f"recheck ref {rk}")
                want = tuple(int(x) for x in global_index[rk][j])
            if got != want:
                raise ValueError(
                    f"global_grid: inconsistent links at ({a.camera},{a.view}) — a dot resolves "
                    f"to {got} but a correspondence requires {want}"
                )

    return global_index


def _is_spec_failure(exc: BaseException) -> bool:
    """True for an *intentional* resolve failure (a spec/click problem the operator can fix).

    Every deliberate raise in this module is a ``ValueError`` whose message starts with
    ``"global_grid:"``. The partial resolver reports those as per-view reasons. An unexpected
    ``ValueError`` (e.g. a shape/dtype error out of numpy or a bug surfacing through cv2) does
    NOT carry the prefix, so it is re-raised rather than silently shown to the operator as a
    "mis-click" — a real bug must stay visible.
    """
    return isinstance(exc, ValueError) and str(exc).startswith("global_grid:")


def resolve_global_grid_partial(
    detections_by_cam: dict,
    spec: Optional[GlobalGridSpec] = None,
    spacing_mm: Optional[float] = None,
    tol_px: float = DEFAULT_CONSISTENCY_TOL_PX,
) -> Tuple[Dict[ViewKey, np.ndarray], List[Tuple[int, int, str]]]:
    """Non-raising sibling of :func:`resolve_global_grid`: resolve every view that *can* be
    resolved from a (possibly half-built or partly-invalid) spec, and report why the rest could
    not — instead of failing the whole call on the first bad anchor.

    Returns ``(global_index, unresolved)`` where ``unresolved`` is a list of
    ``(camera, view, reason)``. ``resolve_global_grid`` stays the authority used by the solve
    (``run_joint_from_spec``); this one drives the GUI live overlay while the user is still
    clicking, where a partial spec is the normal state and one mis-clicked camera must not blank
    the whole grid. Both share every geometric primitive (``_snap``/``_resolve_view``/
    ``_homography_fit``/...); only the orchestration differs (collect vs raise), so the geometry
    cannot drift between them. Only *intentional* resolve failures (see :func:`_is_spec_failure`)
    are turned into reasons; an unexpected error is re-raised so bugs stay visible.
    """
    if _classify_boards(detections_by_cam) == "charuco":
        return global_grid_from_charuco(detections_by_cam), []
    if spec is None:
        raise ValueError(
            "resolve_global_grid_partial: a GlobalGridSpec is required for dotboard"
        )

    unresolved: List[Tuple[int, int, str]] = []
    datum_key = (spec.datum_camera, spec.datum_view)

    # 1. Datum view. A bad datum is terminal — nothing else can be placed without the frame, so
    #    report it against the datum view and return an empty grid (the overlay shows the dots,
    #    the status shows "fix origin/+X/+Y").
    try:
        datum = _detection(detections_by_cam, spec.datum_camera, spec.datum_view)
        sp = float(spacing_mm if spacing_mm is not None else datum.spacing_mm)
        gate = SNAP_FRACTION * _local_spacing_px(datum.image_points)
        for name in ("origin", "x_axis", "y_axis"):
            _snap(datum.image_points, spec.datum_clicks[name], gate, f"datum {name}")
        wf = resolve_world_frame(
            datum.grid_indices, datum.image_points, spec.datum_clicks
        )
        datum_global = _global_from_frame(datum.grid_indices, wf)
        datum_rms, datum_H = _homography_fit(datum_global, datum.image_points, sp)
        if datum_H is None or datum_rms > tol_px:
            raise ValueError(
                f"global_grid: datum world frame does not fit a planar homography "
                f"(RMS {datum_rms:.2f}px > {tol_px}px) — check the origin/+X/+Y clicks"
            )
    except KeyError as exc:
        # A missing datum click key (origin/x_axis/y_axis) — malformed spec, report terminally.
        return {}, [
            (
                spec.datum_camera,
                spec.datum_view,
                f"global_grid: datum is missing a click ({exc})",
            )
        ]
    except ValueError as exc:
        if not _is_spec_failure(exc):
            raise
        return {}, [(spec.datum_camera, spec.datum_view, str(exc))]

    global_index: Dict[ViewKey, np.ndarray] = {datum_key: datum_global}
    prior_by_cam = {spec.datum_camera: _axis_screen_dirs(datum_H, datum_global, sp)}

    # 2. Anchors, in dependency order. A failed anchor is dropped with its reason rather than
    #    aborting; a "not ready" anchor (its reference view is not yet resolved) stays pending.
    pending = list(spec.anchors)
    while pending:
        progressed = False
        for a in list(pending):
            try:
                if (a.camera, a.view) == datum_key:
                    raise ValueError(
                        f"global_grid: anchor ({a.camera},{a.view}) duplicates the datum view"
                    )
                det = _detection(detections_by_cam, a.camera, a.view)
                det_gate = SNAP_FRACTION * _local_spacing_px(det.image_points)
                ready = True
                corr = []
                for c in a.correspondences:
                    if c.same_as != "origin" and tuple(c.same_as) == (a.camera, a.view):
                        raise ValueError(
                            f"global_grid: anchor ({a.camera},{a.view}) references itself"
                        )
                    i_click = _snap(
                        det.image_points,
                        c.pixel,
                        det_gate,
                        f"anchor ({a.camera},{a.view})",
                    )
                    local_idx = np.asarray(det.grid_indices[i_click], dtype=np.int64)
                    if c.same_as == "origin":
                        g = (0, 0)
                    else:
                        ref_key = (int(c.same_as[0]), int(c.same_as[1]))
                        if ref_key not in global_index:
                            ready = False
                            break
                        rdet = _detection(detections_by_cam, ref_key[0], ref_key[1])
                        ref_gate = SNAP_FRACTION * _local_spacing_px(rdet.image_points)
                        j = _snap(
                            rdet.image_points,
                            c.ref_pixel,
                            ref_gate,
                            f"ref dot in {ref_key}",
                        )
                        g = tuple(int(x) for x in global_index[ref_key][j])
                    corr.append((local_idx, g))
                if not ready:
                    continue  # leave pending; a later round may resolve the reference
                gidx, H = _resolve_view(
                    det,
                    corr,
                    sp,
                    prior_by_cam.get(a.camera),
                    tol_px,
                    extend_hint=spec.camera_extends.get(a.camera),
                )
            except ValueError as exc:
                if not _is_spec_failure(exc):
                    raise  # a genuine numpy/cv2 error, not a click problem — surface it
                unresolved.append((a.camera, a.view, str(exc)))
                pending.remove(a)
                progressed = True  # disposed of this anchor — still making progress
                continue
            global_index[(a.camera, a.view)] = gidx
            prior_by_cam[a.camera] = _axis_screen_dirs(H, gidx, sp)
            pending.remove(a)
            progressed = True

        if not progressed:
            # Every remaining anchor references a view that never resolved (broken overlap chain,
            # or it points at a camera that has not been linked yet).
            for p in pending:
                unresolved.append(
                    (
                        p.camera,
                        p.view,
                        "global_grid: not connected to the datum through any resolved view",
                    )
                )
            break

    # 3. Multi-link consistency: drop (don't trust) a resolved view whose links disagree. This is
    #    defensive — phase 2's orientation gate (_orientations_satisfying) already rejects
    #    disagreeing links, so for a view resolved from its own correspondences this recheck is
    #    tautological and rarely fires. It mirrors the strict path's recheck.
    for a in spec.anchors:
        if (a.camera, a.view) not in global_index:
            continue
        try:
            det = detections_by_cam[a.camera][a.view]
            det_gate = SNAP_FRACTION * _local_spacing_px(det.image_points)
            for c in a.correspondences:
                i_click = _snap(
                    det.image_points,
                    c.pixel,
                    det_gate,
                    f"recheck ({a.camera},{a.view})",
                )
                got = tuple(int(x) for x in global_index[(a.camera, a.view)][i_click])
                if c.same_as == "origin":
                    want = (0, 0)
                else:
                    rk = (int(c.same_as[0]), int(c.same_as[1]))
                    if rk not in global_index:
                        continue
                    rdet = detections_by_cam[rk[0]][rk[1]]
                    ref_gate = SNAP_FRACTION * _local_spacing_px(rdet.image_points)
                    j = _snap(
                        rdet.image_points, c.ref_pixel, ref_gate, f"recheck ref {rk}"
                    )
                    want = tuple(int(x) for x in global_index[rk][j])
                if got != want:
                    raise ValueError(
                        f"global_grid: inconsistent links at ({a.camera},{a.view}) — a dot "
                        f"resolves to {got} but a correspondence requires {want}"
                    )
        except ValueError as exc:
            if not _is_spec_failure(exc):
                raise
            unresolved.append((a.camera, a.view, str(exc)))
            global_index.pop((a.camera, a.view), None)

    # 4. Cascade. If the recheck dropped a view, any view resolved *off* it (its links reference a
    #    key no longer present) is no longer trustworthy — drop it too, transitively. This
    #    guarantees the partial path never reports a dependent of a dropped view as resolved,
    #    matching the strict path which never commits such a dependent. It rarely runs (step 3
    #    rarely pops), but it keeps the invariant under any future change to the resolver.
    changed = True
    while changed:
        changed = False
        for a in spec.anchors:
            key = (a.camera, a.view)
            if key not in global_index:
                continue
            for c in a.correspondences:
                if c.same_as == "origin":
                    continue
                rk = (int(c.same_as[0]), int(c.same_as[1]))
                if rk not in global_index:
                    global_index.pop(key, None)
                    unresolved.append(
                        (
                            a.camera,
                            a.view,
                            f"global_grid: depends on view {rk}, which was dropped",
                        )
                    )
                    changed = True
                    break

    return global_index, unresolved


def first_view_orientation_candidates(
    detections_by_cam: dict,
    spec: GlobalGridSpec,
    camera: int,
    view: int,
    spacing_mm: Optional[float] = None,
    tol_px: float = DEFAULT_CONSISTENCY_TOL_PX,
) -> List[dict]:
    """The orientations a new camera's first-view anchor allows, for the confirm-on-overlay picker.

    When the overlap is a thin collinear strip the clicks leave a mirror ambiguity that fits the
    dots equally well; the GUI shows each candidate's footprint and the user picks the real one.
    Each candidate carries the ``extend`` direction (global-index vector from the anchored dot to
    the candidate's centroid) that, fed back as ``camera_extends[camera]``, reproduces that choice
    headlessly — so the pick persists and the CLI is deterministic.

    Returns ``[]`` when the view is not ambiguous (one orientation, or it already resolves), when
    its reference views are not resolved, or when the anchor/detection is missing — the GUI then
    has nothing to ask. Candidates are ordered by descending homography fit (best first), though
    for a true lattice fold the fits are equal and the order is immaterial.
    """
    anchor = next(
        (a for a in spec.anchors if a.camera == camera and a.view == view), None
    )
    if anchor is None:
        return []
    resolved, _ = resolve_global_grid_partial(
        detections_by_cam, spec, spacing_mm, tol_px
    )
    if (camera, view) in resolved:
        return (
            []
        )  # resolved unambiguously (single orientation, or a prior/hint settled it)

    try:
        det = _detection(detections_by_cam, camera, view)
    except ValueError:
        return []
    sp = float(spacing_mm if spacing_mm is not None else det.spacing_mm)
    det_gate = SNAP_FRACTION * _local_spacing_px(det.image_points)

    corr = []
    for c in anchor.correspondences:
        try:
            i_click = _snap(
                det.image_points, c.pixel, det_gate, f"anchor ({camera},{view})"
            )
            local_idx = np.asarray(det.grid_indices[i_click], dtype=np.int64)
            if c.same_as == "origin":
                g = (0, 0)
            else:
                ref_key = (int(c.same_as[0]), int(c.same_as[1]))
                if ref_key not in resolved:
                    return (
                        []
                    )  # reference not resolved — cannot offer a concrete footprint
                rdet = _detection(detections_by_cam, ref_key[0], ref_key[1])
                ref_gate = SNAP_FRACTION * _local_spacing_px(rdet.image_points)
                j = _snap(
                    rdet.image_points, c.ref_pixel, ref_gate, f"ref dot in {ref_key}"
                )
                g = tuple(int(x) for x in resolved[ref_key][j])
            corr.append((local_idx, g))
        except ValueError:
            return []  # a mis-snapped click — no clean candidates to show

    orients = _orientations_satisfying([c[0] for c in corr], [c[1] for c in corr])
    la, ga = corr[0]
    ga = np.asarray(ga, dtype=np.float64)
    out = []
    for o in orients:
        gidx = _map_indices(det.grid_indices, la, ga, o)
        rms, _H = _homography_fit(gidx, det.image_points, sp)
        if not (np.isfinite(rms) and rms <= tol_px):
            continue
        extend = gidx.astype(np.float64).mean(axis=0) - ga
        out.append(
            {
                "extend": [float(extend[0]), float(extend[1])],
                "gx_range": [int(gidx[:, 0].min()), int(gidx[:, 0].max())],
                "gy_range": [int(gidx[:, 1].min()), int(gidx[:, 1].max())],
                "n": int(len(gidx)),
                "rms": float(rms),
            }
        )
    out.sort(key=lambda c: c["rms"])
    return out if len(out) > 1 else []
