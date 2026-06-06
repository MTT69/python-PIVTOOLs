"""calibration2.stepped_calibrate — stepped (dual-level) pinhole calibration.

The stepped board's object points lie on two parallel Z-planes, so they are
non-coplanar. Zhang's homography-based intrinsic init assumes coplanarity; feeding
the raw two-Z points straight to ``cv2.calibrateCamera`` is the fragile path (v1
saw fx 3956 vs 68321 from a single-pose non-planar init). This module preserves v1's
canonical **two-stage fit** verbatim (no silent algorithm change): a multi-image
Zhang init on one coplanar level extracted from every pose, then a 3D refine on both
levels with ``CALIB_USE_INTRINSIC_GUESS``. The datum pose is then re-solved with
``fit_pose(planar=False)`` (SQPNP) in the user's clicked world frame, so the model
carries one ``(R, t, K, dist)`` covering both surfaces (invariant 4).

The canonical math (``compute_z_and_offsets``, ``build_object_points``,
``fit_pinhole``, ``assign_absolute_grid_indices``) is ported unchanged from the
orphaned v1 ``calibration_stepped.stepped_calibration_production``; the level
separation / stitch geometry lives in ``detection.stepped_levels``. Output is the
unified ``CameraModel`` + ``WorldFrame`` + ``MonoRecord``.

Stereo (S3) reuses the same mono machinery per camera and composes the two
world-frame poses into ``R_stereo`` / ``T_stereo`` (CLAUDE.md gotcha #7). The
same-side vs transmission classification is derived deterministically from the two
cameras' fiducial clicks (``classify_stereo_config``) instead of v1's fit-cam2-twice
auto-test; the resulting Z shift lands in cam2's fitted pose (invariant 6).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from .camera_model import (
    CameraModel,
    DistortionModel,
    fit_polynomial3d,
    fit_pose,
    per_view_rms,
)
from .detection.base import DetectionResult
from .detection.stepped_levels import SteppedBoardSpec, stitch_levels_pose_local
from .record import MonoRecord, StereoRecord, WorldFrame
from .stereo_model import camera_z_sign, compose_stereo


# ---------------------------------------------------------------------------
# Fiducial -> absolute grid index assignment (datum pose only) — ported verbatim
# ---------------------------------------------------------------------------

def _pixel_to_grid_index(pixel_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map a pixel position to fractional grid indices using homography inverse."""
    H_inv = np.linalg.inv(H)
    p = np.array([pixel_px[0], pixel_px[1], 1.0], dtype=np.float64)
    q = H_inv @ p
    return (q[:2] / q[2]).astype(np.float32)


def _inject_click_into_level(level_data: dict, click_px: np.ndarray) -> Tuple[dict, int]:
    """Inject a fiducial click into a level's detected grid.

    Uses the homography to determine the grid index from the clicked pixel
    position. If a detected dot already exists at that grid index, returns it.
    Otherwise injects the click as a new point.
    """
    centers = level_data['centers']
    gi = level_data['grid_indices']
    H = level_data['H']

    fractional_gi = _pixel_to_grid_index(click_px, H)
    click_gi = np.round(fractional_gi).astype(np.int32)

    existing = np.where((gi[:, 0] == click_gi[0]) & (gi[:, 1] == click_gi[1]))[0]
    if len(existing) > 0:
        return level_data, int(existing[0])

    new_centers = np.vstack([centers, click_px.reshape(1, 2)])
    new_gi = np.vstack([gi, click_gi.reshape(1, 2)])
    updated = dict(level_data)
    updated['centers'] = new_centers
    updated['grid_indices'] = new_gi
    return updated, len(new_centers) - 1


def assign_absolute_grid_indices(
    fiducials: dict,
    level_A_data: dict,
    level_B_data: Optional[dict],
    clicked_level: str,
    board: SteppedBoardSpec,
) -> dict:
    """Assign absolute grid indices using the user-clicked origin.

    The clicked dot becomes grid index (0, 0); all other dots are relative to it.
    Ported verbatim from v1 (no silent algorithm change).

    Args:
        fiducials: dict with 'origin', 'x_axis', 'y_axis' image-down pixel coords.
        level_A_data, level_B_data: ``run_single_level_detection`` dicts (B may be None).
        clicked_level: 'peak' or 'trough' — the face the origin click landed on.
        board: board specification.

    Returns:
        dict with the clicked level and (if present) the other level, each
        {'centers', 'grid_indices'}; plus 'orientation'
        {swap_axes, col_sign, row_sign} and '_origin_on_level' ('A' or 'B').
    """
    origin_px = np.array(fiducials['origin'], dtype=np.float32)
    x_axis_px = np.array(fiducials['x_axis'], dtype=np.float32)
    y_axis_px = np.array(fiducials['y_axis'], dtype=np.float32)

    # Determine which level the clicked dot belongs to using homography residual.
    frac_A = _pixel_to_grid_index(origin_px, level_A_data['H'])
    res_A = float(np.max(np.abs(frac_A - np.round(frac_A))))
    if level_B_data is not None:
        frac_B = _pixel_to_grid_index(origin_px, level_B_data['H'])
        res_B = float(np.max(np.abs(frac_B - np.round(frac_B))))
    else:
        res_B = np.inf

    if res_A <= res_B:
        origin_data = level_A_data
        other_data = level_B_data
    else:
        origin_data = level_B_data
        other_data = level_A_data

    logger.debug(
        f"Origin on {'A' if res_A <= res_B else 'B'} "
        f"(residual A={res_A:.3f}, B={res_B:.3f})"
    )

    # Inject origin click into the origin level's grid.
    origin_data, origin_idx_in_level = _inject_click_into_level(origin_data, origin_px)

    # Inject +X and +Y clicks into whichever level they belong to.
    for click_px in [x_axis_px, y_axis_px]:
        def grid_residual_for_level(ld, point):
            if ld is None or 'H' not in ld:
                return np.inf
            frac_gi = _pixel_to_grid_index(point, ld['H'])
            return float(np.max(np.abs(frac_gi - np.round(frac_gi))))

        res_origin = grid_residual_for_level(origin_data, click_px)
        res_other = grid_residual_for_level(other_data, click_px)
        if res_origin <= res_other:
            origin_data, _ = _inject_click_into_level(origin_data, click_px)
        elif other_data is not None:
            other_data, _ = _inject_click_into_level(other_data, click_px)

    origin_bfs = origin_data['grid_indices'][origin_idx_in_level].copy()

    # Determine axis orientation from fiducial vectors vs BFS vectors.
    fid_x_vec = x_axis_px - origin_px
    fid_y_vec = y_axis_px - origin_px
    bfs_vec1 = origin_data['vec1']
    bfs_vec2 = origin_data['vec2']

    def alignment(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    align_x_v1 = alignment(fid_x_vec, bfs_vec1)
    align_x_v2 = alignment(fid_x_vec, bfs_vec2)

    if abs(align_x_v1) >= abs(align_x_v2):
        col_sign = 1 if align_x_v1 > 0 else -1
        row_sign = 1 if alignment(fid_y_vec, bfs_vec2) > 0 else -1
        swap_axes = False
    else:
        col_sign = 1 if align_x_v2 > 0 else -1
        row_sign = 1 if alignment(fid_y_vec, bfs_vec1) > 0 else -1
        swap_axes = True

    def transform_indices(gi, origin_bfs_indices):
        abs_gi = gi.copy()
        if swap_axes:
            abs_gi = abs_gi[:, ::-1]
            ob = origin_bfs_indices[::-1]
        else:
            ob = origin_bfs_indices
        abs_gi[:, 0] = col_sign * (abs_gi[:, 0] - ob[0])
        abs_gi[:, 1] = row_sign * (abs_gi[:, 1] - ob[1])
        return abs_gi

    # Origin level: transform BFS -> absolute (origin = grid (0,0)).
    abs_gi_origin = transform_indices(origin_data['grid_indices'].copy(), origin_bfs)

    result = {}
    result[clicked_level] = {
        'centers': origin_data['centers'].copy(),
        'grid_indices': abs_gi_origin,
    }

    # Other level: BFS + anchor offset.
    other_level = "trough" if clicked_level == "peak" else "peak"
    if other_data is not None:
        origin_centers = result[clicked_level]['centers']
        origin_abs_gi = result[clicked_level]['grid_indices']
        origin_tree = cKDTree(origin_centers)

        spacing = board.dot_spacing_mm
        level_offset = board.level_offset_mm

        origin_phys_x = origin_abs_gi[:, 0].astype(float) * spacing
        origin_phys_y = origin_abs_gi[:, 1].astype(float) * spacing

        other_bfs_gi = other_data['grid_indices'].copy()
        dummy_origin = np.array([0, 0], dtype=np.int32)
        oriented_other_gi = transform_indices(other_bfs_gi, dummy_origin)

        other_centers = other_data['centers'].copy()
        n_other = len(other_centers)

        anchor_offsets = []
        for i in range(n_other):
            dists, idxs = origin_tree.query(other_centers[i], k=4)
            neighbor_gi = origin_abs_gi[idxs]
            gi_range = neighbor_gi.max(axis=0) - neighbor_gi.min(axis=0)

            if gi_range[0] != 1 or gi_range[1] != 1:
                continue

            mean_phys_x = float(np.mean(origin_phys_x[idxs]))
            mean_phys_y = float(np.mean(origin_phys_y[idxs]))
            expected_gi_col = round((mean_phys_x - level_offset) / spacing)
            expected_gi_row = round((mean_phys_y - level_offset) / spacing)
            expected_abs = np.array([expected_gi_col, expected_gi_row], dtype=np.int32)

            off = expected_abs - oriented_other_gi[i]
            anchor_offsets.append(off)

        if len(anchor_offsets) > 0:
            anchor_offsets_arr = np.array(anchor_offsets)
            offset_tuples = [tuple(o) for o in anchor_offsets_arr]
            offset_counts = Counter(offset_tuples)
            best_offset, best_count = offset_counts.most_common(1)[0]
            abs_offset = np.array(best_offset, dtype=np.int32)

            consensus_pct = 100 * best_count / len(anchor_offsets)
            logger.debug(
                f"{other_level} anchoring: offset=({abs_offset[0]}, {abs_offset[1]}), "
                f"consensus={consensus_pct:.0f}% ({best_count}/{len(anchor_offsets)})"
            )

            abs_gi_other = oriented_other_gi + abs_offset

            result[other_level] = {
                'centers': other_centers,
                'grid_indices': abs_gi_other,
            }
        else:
            logger.warning(f"No anchor points found for {other_level} level")

    # Expose the fiducial-derived orientation so non-datum poses reuse the SAME
    # convention instead of re-deriving it from raw BFS vectors (back-view cameras
    # silently disagree with their datum otherwise, blowing up the intrinsic fit).
    result['orientation'] = {
        'swap_axes': bool(swap_axes),
        'col_sign': int(col_sign),
        'row_sign': int(row_sign),
    }
    # Which of level_A / level_B the fiducial origin click landed on.
    result['_origin_on_level'] = 'A' if res_A <= res_B else 'B'

    return result


# ---------------------------------------------------------------------------
# Object points + per-level Z / xy-offset — ported verbatim
# ---------------------------------------------------------------------------

def build_object_points(
    grid_indices: np.ndarray, z_mm: float, spacing_mm: float, xy_offset_mm: float = 0.0
) -> np.ndarray:
    """Create 3D world object points from grid indices."""
    n = len(grid_indices)
    obj = np.zeros((n, 3), dtype=np.float32)
    obj[:, 0] = grid_indices[:, 0] * spacing_mm + xy_offset_mm
    obj[:, 1] = grid_indices[:, 1] * spacing_mm + xy_offset_mm
    obj[:, 2] = z_mm
    return obj


def compute_z_and_offsets(
    stereo_config: str,
    cam1_clicked_level: str,
    cam2_clicked_level: str,
    board: SteppedBoardSpec,
) -> dict:
    """Compute Z values and XY offsets for each camera's levels.

    Convention: cam1's clicked level is at Z=0 (reference plane). The clicked level
    on both cameras has xy_offset=0; the other level has xy_offset=level_offset_mm.
    Ported verbatim from v1. Mono uses only the 'Cam1' slice (cam1's Z values are
    independent of stereo_config).
    """
    step = board.step_height_mm
    thickness = board.board_thickness_mm
    offset = board.level_offset_mm

    cam1_other = "trough" if cam1_clicked_level == "peak" else "peak"

    if cam1_clicked_level == "peak":
        cam1_z = {"peak": 0.0, "trough": -step}
    else:
        cam1_z = {"trough": 0.0, "peak": step}

    cam1_xy_offset = {cam1_clicked_level: 0.0, cam1_other: offset}

    cam2_other = "trough" if cam2_clicked_level == "peak" else "peak"

    if stereo_config == "same_side":
        cam2_z = cam1_z.copy()
    else:
        # Transmission: cam2 sees opposite face.
        shift = 0.0 if cam1_clicked_level == "peak" else step
        cam2_z = {
            cam2_clicked_level: -(thickness - step) + shift,
            cam2_other: -thickness + shift,
        }

    cam2_xy_offset = {cam2_clicked_level: 0.0, cam2_other: offset}

    return {
        'Cam1': {'z': cam1_z, 'xy_offset': cam1_xy_offset},
        'Cam2': {'z': cam2_z, 'xy_offset': cam2_xy_offset},
    }


# ---------------------------------------------------------------------------
# Two-stage pinhole fit — ported verbatim
# ---------------------------------------------------------------------------

def fit_pinhole(
    obj_views: list,
    img_views: list,
    image_size: Tuple[int, int],
    fix_aspect: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, list, list]:
    """Fit OpenCV pinhole model using multiple views of 3D object points.

    Two-step approach (ported verbatim from v1):
      1. Multi-image Zhang init: extract one coplanar z-level from every pose, feed
         all of them as separate "images" to cv2.calibrateCamera (K=None). Each pose
         contributes one homography; OpenCV solves the overdetermined IAC system from
         all homographies jointly — the same mechanism that makes dotboard/charuco
         calibration robust at PIV magnification. The z-level closest to z=0 is
         chosen (best-conditioned Zhang problem).
      2. Refine with all views using true 3D coordinates (both z-levels). The stepped
         board's two-Z non-coplanarity breaks the fx-tz ridge the coplanar init
         cannot resolve alone.

    Returns (rms, K, dist, rvecs, tvecs).
    """
    W, H = image_size

    # Step 1: Multi-image Zhang init on one coplanar level from ALL poses.
    unique_z = np.unique(np.round(obj_views[0][:, 2], 2))
    z_for_init = unique_z[np.argmin(np.abs(unique_z))]

    init_objs = []
    init_imgs = []
    for obj, img in zip(obj_views, img_views):
        mask = np.abs(obj[:, 2] - z_for_init) < 0.5
        if mask.sum() < 4:
            continue
        obj_z0 = np.column_stack([
            obj[mask, 0], obj[mask, 1], np.zeros(mask.sum())
        ]).astype(np.float32)
        init_objs.append(obj_z0)
        init_imgs.append(img[mask].reshape(-1, 1, 2).astype(np.float32))

    init_flags = cv2.CALIB_FIX_K3
    if fix_aspect:
        init_flags |= cv2.CALIB_FIX_ASPECT_RATIO

    _, K_init, dist_init, _, _ = cv2.calibrateCamera(
        init_objs, init_imgs, (W, H), None, None, flags=init_flags)

    # Step 2: Refine with all views using true 3D coordinates.
    obj_list = [obj.astype(np.float32) for obj in obj_views]
    img_list = [img.reshape(-1, 1, 2).astype(np.float32) for img in img_views]

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    if fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, (W, H), K_init.copy(), dist_init.copy(), flags=flags)

    return rms, K, dist, rvecs, tvecs


# ---------------------------------------------------------------------------
# Non-datum pose object-point construction (re-stitch with datum orientation)
# ---------------------------------------------------------------------------

def _build_non_datum_pose_view(
    level_A: Optional[dict],
    level_B: Optional[dict],
    geo_cam: dict,
    A_label: str,
    B_label: str,
    board: SteppedBoardSpec,
    orientation_override: Optional[dict] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    """Build a NON-DATUM pose's (obj_points, img_points) using BOTH levels, stitched
    into one pose-local frame via ``stitch_levels_pose_local``.

    ``orientation_override`` (the datum pose's fiducial-derived swap/col/row) is
    forwarded so every non-datum stitch reuses the same convention — without it,
    strongly-foreshortened poses can invert chirality and break the multi-view fit.
    Per-dot z / xy_offset come from the operator-supplied A/B labels via geo_cam.

    Returns (obj_np, img_np, stitch_meta) or (None, None, None).
    """
    spacing = board.dot_spacing_mm
    stitched = stitch_levels_pose_local(
        level_A, level_B, board, orientation_override=orientation_override,
    )
    if stitched is None:
        return None, None, None

    ref = stitched['reference']
    other = stitched.get('other')
    stitch_meta = dict(stitched.get('metadata', {}))

    def _level_name(source_letter):
        return A_label if source_letter == 'A' else B_label

    all_obj = []
    all_img = []
    for entry in (ref, other):
        if entry is None:
            continue
        level_name = _level_name(entry['source_level'])
        gi = np.asarray(entry['grid_indices'], dtype=np.int32)
        centers = np.asarray(entry['centers'], dtype=np.float32)
        z_mm = geo_cam['z'][level_name]
        xy_off = geo_cam['xy_offset'][level_name]
        obj = build_object_points(gi, z_mm, spacing, xy_off)
        all_obj.append(obj)
        all_img.append(centers)

    if not all_obj:
        return None, None, None

    return np.vstack(all_obj), np.vstack(all_img), stitch_meta


# ---------------------------------------------------------------------------
# Mono orchestrator
# ---------------------------------------------------------------------------

def _level_dicts(detection: DetectionResult) -> Tuple[Optional[dict], Optional[dict]]:
    """Pull the raw per-level grid dicts the detector stashed in diagnostics."""
    diag = detection.diagnostics or {}
    return diag.get("_level_a_full"), diag.get("_level_b_full")


def _write_poly3d_figure(figure_dir, model, datum_obj, datum_img, images,
                         datum_index, prefix) -> None:
    """Datum-view reprojection + residual figure for the 3D polynomial fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    obj = np.asarray(datum_obj, dtype=np.float64)
    img = np.asarray(datum_img, dtype=np.float64)
    proj = model.project(obj)
    res = proj - img
    z = obj[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    if images is not None and 0 <= datum_index < len(images):
        ax.imshow(images[datum_index], cmap="gray")
    sc = ax.scatter(img[:, 0], img[:, 1], c=z, cmap="coolwarm", s=12, label="detected")
    ax.scatter(proj[:, 0], proj[:, 1], facecolors="none", edgecolors="lime",
               s=24, linewidths=0.6, label="projected")
    ax.set_title(f"poly3d datum reprojection (RMS {model.rms_px:.3f} px)")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px, image-down)")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax, label="world Z (mm)")

    ax2 = axes[1]
    mag = np.linalg.norm(res, axis=1)
    q = ax2.quiver(img[:, 0], img[:, 1], res[:, 0], res[:, 1], mag,
                   cmap="viridis", angles="xy")
    ax2.set_title("reprojection residuals")
    ax2.set_xlabel("u (px)")
    ax2.set_ylabel("v (px, image-down)")
    ax2.invert_yaxis()
    fig.colorbar(q, ax=ax2, label="|residual| (px)")

    fig.tight_layout()
    fig.savefig(figure_dir / f"{prefix}poly3d_reprojection.png", dpi=110)
    plt.close(fig)


def _calibrate_stepped_mono_poly3d(
    datum_obj: np.ndarray,
    datum_img: np.ndarray,
    fiducials: Dict[str, Sequence[float]],
    orientation: dict,
    clicked_level: str,
    board: SteppedBoardSpec,
    spacing: float,
    camera: int,
    image_size: Tuple[int, int],
    datum: DetectionResult,
    images: Optional[Sequence[np.ndarray]],
    datum_index: int,
    figure_dir: Optional[Path],
    figure_prefix: str,
) -> MonoRecord:
    """Fit a single-view 3D polynomial (world->image) on the datum view's two planes.

    The two physical levels of ONE view give the Z leverage, so no extra poses are
    needed (matches DaVis ``poly1`` = 1 image/camera). +Z is toward the camera by the
    stepped clicked-level convention (the dotted peak face is nearer the camera than
    the trough), so ``world_z_toward_camera = +1``.
    """
    model = fit_polynomial3d(datum_obj, datum_img, image_size, world_z_toward_camera=1.0)
    logger.info(
        f"stepped poly3d cam{camera}: rms={model.rms_px:.3f}px, "
        f"per-plane={[round(v, 3) for v in model.plane_rms_px]}"
    )
    wf = WorldFrame(
        mode="clicks",
        origin_px=np.asarray(fiducials['origin'], dtype=np.float64).reshape(2),
        x_axis_px=np.asarray(fiducials['x_axis'], dtype=np.float64).reshape(2),
        y_axis_px=np.asarray(fiducials['y_axis'], dtype=np.float64).reshape(2),
        swap_axes=bool(orientation['swap_axes']),
        col_sign=int(orientation['col_sign']),
        row_sign=int(orientation['row_sign']),
        origin_grid=np.array([0.0, 0.0]),
        origin_mm=np.array([0.0, 0.0]),
    )
    meta = {
        "spacing_mm": float(spacing),
        "step_height_mm": float(board.step_height_mm),
        "board_thickness_mm": float(board.board_thickness_mm),
        "level_offset_mm": float(board.level_offset_mm),
        "clicked_level": str(clicked_level),
        "model_type": "polynomial3d",
        "n_views": 1,
    }
    if figure_dir is not None:
        try:
            _write_poly3d_figure(figure_dir, model, datum_obj, datum_img,
                                 images, datum_index, figure_prefix)
        except Exception:  # figures never abort the fit
            logger.warning("stepped poly3d figure writing failed (non-fatal)")
    return MonoRecord(
        camera=int(camera), board_type="stepped", camera_model=model,
        world_frame=wf, per_view_rms=[float(model.rms_px)], board_meta=meta,
    )


def calibrate_stepped_mono(
    detections: Sequence[DetectionResult],
    fiducials: Dict[str, Sequence[float]],
    clicked_level: str,
    pose_levels: Sequence[str],
    board: SteppedBoardSpec,
    image_size: Tuple[int, int],
    camera: int = 1,
    datum_index: int = 0,
    distortion_model: DistortionModel = DistortionModel.STANDARD,
    geo_override: Optional[dict] = None,
    model_type: str = "pinhole",
    images: Optional[Sequence[np.ndarray]] = None,
    figure_dir: Optional[Path] = None,
    figure_prefix: str = "",
) -> MonoRecord:
    """Calibrate one camera from stepped-board poses (pinhole) or one datum view (poly).

    ``model_type`` selects the model family. ``'pinhole'`` (default) runs the multi-view
    two-stage fit below. ``'polynomial3d'`` fits a single-view 3D cubic forward map
    (world->image) on the datum view's two planes only (the DaVis ``poly`` model); it
    needs no extra poses and ignores everything after the datum assembly. See
    ``camera_model.Polynomial3DModel``.

    Parameters
    ----------
    detections : one ``DetectionResult`` per pose (from ``SteppedDetector.detect``),
        each carrying the raw per-level dicts in ``diagnostics``.
    fiducials : {'origin', 'x_axis', 'y_axis'} image-down pixel coords on the datum.
    clicked_level : 'peak' | 'trough' — the face the origin fiducial click landed on.
        This level is the world Z=0 datum plane.
    pose_levels : per-pose label ('peak'|'trough') for level A of each pose,
        position-aligned to ``detections``. The datum entry is cross-checked against
        the fiducial-derived label (a mismatch is a hard error).
    board : board geometry.
    image_size : (width, height) in pixels.
    camera : camera number for the record.
    datum_index : index of the datum pose in ``detections``.
    distortion_model : distortion model for the resulting ``CameraModel``.
    geo_override : optional ``{'z': {...}, 'xy_offset': {...}}`` slice from
        ``compute_z_and_offsets`` to use instead of the cam1 default. The default
        places the clicked level at world Z=0 in this camera's own frame; stereo
        passes the cam2 slice so cam2's levels sit at their SHARED-frame absolute Z
        (invariant 6 — the same/transmission Z shift lands in the fitted pose, not a
        post-hoc negation). Intrinsics are unaffected by a global Z shift (it is
        absorbed per-pose by ``tvec``); only the datum pose's placement moves.
    images : optional per-pose image list (position-aligned to ``detections``), used
        ONLY to draw the detection-overlay + world-frame proof figures. Ignored unless
        ``figure_dir`` is set.
    figure_dir : optional directory; when set, the stepped proof figures (detection
        overlays, world frame, reprojection on the fitted views, distortion map, 3D
        boards) are written there. Figure failures are non-fatal.
    figure_prefix : filename prefix for the figures (``"cam1_"`` / ``"cam2_"`` in a
        stereo run; empty for mono — the empty-prefix run also gets the 3D boards plot).

    Returns
    -------
    MonoRecord with a ``CameraModel`` (one (R,t,K,dist) covering both surfaces) and
    the resolved ``WorldFrame``.
    """
    if clicked_level not in ('peak', 'trough'):
        raise ValueError(f"clicked_level must be 'peak' or 'trough', got {clicked_level!r}")
    n_poses = len(detections)
    if n_poses == 0:
        raise ValueError("calibrate_stepped_mono: no detections provided")
    if not (0 <= datum_index < n_poses):
        raise ValueError(f"datum_index {datum_index} out of range [0, {n_poses})")
    if len(pose_levels) != n_poses:
        raise ValueError(
            f"pose_levels has length {len(pose_levels)} but there are {n_poses} poses"
        )
    for i, lbl in enumerate(pose_levels):
        if lbl not in ('peak', 'trough'):
            raise ValueError(f"pose_levels[{i}]={lbl!r}, expected 'peak' or 'trough'")

    other_level = 'trough' if clicked_level == 'peak' else 'peak'
    if geo_override is not None:
        geo_cam = geo_override
    else:
        geo_cam = compute_z_and_offsets('same_side', clicked_level, clicked_level, board)['Cam1']
    spacing = board.dot_spacing_mm

    # ---- Datum pose: fiducial-anchored world frame ----
    datum = detections[datum_index]
    lv_A, lv_B = _level_dicts(datum)
    if lv_A is None and lv_B is None:
        raise RuntimeError("datum pose has no detected grid level")

    absolute = assign_absolute_grid_indices(fiducials, lv_A, lv_B, clicked_level, board)

    missing = [name for name in (clicked_level, other_level) if absolute.get(name) is None]
    if missing:
        raise RuntimeError(
            f"datum pose missing {' and '.join(missing)} level(s); a stepped datum "
            f"needs both levels to anchor the world frame. Pick a different datum "
            f"frame or reduce board tilt."
        )

    orientation = absolute['orientation']
    origin_on_level = absolute.get('_origin_on_level')
    if origin_on_level == 'A':
        datum_A_label = clicked_level
    elif origin_on_level == 'B':
        datum_A_label = other_level
    else:
        raise RuntimeError("datum fiducial origin-on-level unresolved; re-run fiducial snap")

    if pose_levels[datum_index] != datum_A_label:
        raise ValueError(
            f"pose_levels[{datum_index}]={pose_levels[datum_index]!r} conflicts with the "
            f"fiducial-derived datum A_label={datum_A_label!r} (origin landed on level "
            f"{origin_on_level}, clicked_level={clicked_level!r}). Correct the datum "
            f"pose label or re-pick clicked_level."
        )

    # Datum object/image points are already in the world frame (clicked dot = origin).
    datum_obj = []
    datum_img = []
    for level_name in (clicked_level, other_level):
        data = absolute[level_name]
        z_mm = geo_cam['z'][level_name]
        xy_off = geo_cam['xy_offset'][level_name]
        datum_obj.append(build_object_points(data['grid_indices'], z_mm, spacing, xy_off))
        datum_img.append(np.asarray(data['centers'], dtype=np.float32))
    datum_obj = np.vstack(datum_obj).astype(np.float32)
    datum_img = np.vstack(datum_img).astype(np.float32)

    W_H = (int(image_size[0]), int(image_size[1]))
    if model_type == "polynomial3d":
        # Single datum view, two planes -> 3D cubic forward map. No multi-pose fit,
        # no pinhole intrinsics/pose: the polynomial absorbs the clicked frame directly.
        return _calibrate_stepped_mono_poly3d(
            datum_obj=datum_obj, datum_img=datum_img, fiducials=fiducials,
            orientation=orientation, clicked_level=clicked_level, board=board,
            spacing=spacing, camera=camera, image_size=W_H, datum=datum,
            images=images, datum_index=datum_index, figure_dir=figure_dir,
            figure_prefix=figure_prefix,
        )
    if model_type != "pinhole":
        raise ValueError(
            f"model_type must be 'pinhole' or 'polynomial3d', got {model_type!r}"
        )

    # ---- Accumulate pose views (datum first) ----
    pose_obj_views = [datum_obj]
    pose_img_views = [datum_img]
    # Original-detection indices + detections for the views actually fed to the fit,
    # position-aligned to pose_obj_views (datum first). Used only for figure labelling.
    used_pose_indices = [datum_index]
    used_detections = [datum]

    for pose_idx in range(n_poses):
        if pose_idx == datum_index:
            continue
        det = detections[pose_idx]
        a_pose, b_pose = _level_dicts(det)
        if a_pose is None or b_pose is None:
            logger.warning(
                f"pose {pose_idx}: only one level detected — both required for "
                f"per-pose labelling, skipping"
            )
            continue
        A_label = pose_levels[pose_idx]
        B_label = 'trough' if A_label == 'peak' else 'peak'
        obj_np, img_np, meta = _build_non_datum_pose_view(
            a_pose, b_pose, geo_cam, A_label, B_label, board,
            orientation_override=orientation,
        )
        if obj_np is None:
            continue
        pose_obj_views.append(obj_np.astype(np.float32))
        pose_img_views.append(img_np.astype(np.float32))
        used_pose_indices.append(pose_idx)
        used_detections.append(det)
        if meta and not meta.get('degraded_single_level', False):
            cp = meta.get('consensus_pct', float('nan'))
            if cp == cp and cp < 80.0:
                logger.warning(
                    f"pose {pose_idx}: stitch consensus only {cp:.0f}% — included, "
                    f"treat with suspicion"
                )

    if len(pose_obj_views) < 3:
        raise RuntimeError(
            f"need >=3 usable poses for the intrinsic fit, got {len(pose_obj_views)}"
        )

    # ---- Two-stage pinhole fit (free + fixed aspect, keep lower RMS) ----
    W_H = (int(image_size[0]), int(image_size[1]))
    rms_free, K_free, dist_free, rvecs_free, tvecs_free = fit_pinhole(
        pose_obj_views, pose_img_views, W_H, fix_aspect=False)
    rms_fixed, K_fixed, dist_fixed, rvecs_fixed, tvecs_fixed = fit_pinhole(
        pose_obj_views, pose_img_views, W_H, fix_aspect=True)

    if rms_fixed <= rms_free * 1.05:
        rms, K, dist, rvecs, tvecs = rms_fixed, K_fixed, dist_fixed, rvecs_fixed, tvecs_fixed
    else:
        rms, K, dist, rvecs, tvecs = rms_free, K_free, dist_free, rvecs_free, tvecs_free

    # ---- Datum pose in the world frame (SQPNP, both levels non-coplanar) ----
    R, t = fit_pose(datum_obj, datum_img, K, dist, planar=False)

    cam = CameraModel(
        K=K, dist=dist, R=R, t=t, image_size=W_H,
        distortion_model=distortion_model, rms=float(rms),
    )

    pv = per_view_rms(pose_obj_views, pose_img_views, K, dist, rvecs, tvecs)

    wf = WorldFrame(
        mode="clicks",
        origin_px=np.asarray(fiducials['origin'], dtype=np.float64).reshape(2),
        x_axis_px=np.asarray(fiducials['x_axis'], dtype=np.float64).reshape(2),
        y_axis_px=np.asarray(fiducials['y_axis'], dtype=np.float64).reshape(2),
        swap_axes=bool(orientation['swap_axes']),
        col_sign=int(orientation['col_sign']),
        row_sign=int(orientation['row_sign']),
        origin_grid=np.array([0.0, 0.0]),  # clicked dot is re-anchored to grid (0,0)
        origin_mm=np.array([0.0, 0.0]),
    )

    meta = {
        "spacing_mm": float(spacing),
        "step_height_mm": float(board.step_height_mm),
        "board_thickness_mm": float(board.board_thickness_mm),
        "level_offset_mm": float(board.level_offset_mm),
        "clicked_level": str(clicked_level),
        "n_views": int(len(pose_obj_views)),
    }

    # ---- Proof figures (optional; drawn while the fit arrays are live) ----
    if figure_dir is not None:
        from . import figures as c2figs
        try:
            c2figs.write_stepped_figures(
                figure_dir, images=images, used_detections=used_detections,
                used_pose_indices=used_pose_indices, pose_obj_views=pose_obj_views,
                pose_img_views=pose_img_views, rvecs=rvecs, tvecs=tvecs,
                per_view=list(pv), rms=float(rms), cam=cam, wf=wf, spacing=float(spacing),
                datum_index=datum_index, datum_detection=datum, prefix=figure_prefix,
            )
        except Exception:  # figures never abort the fit
            logger.warning("stepped mono figure writing failed (non-fatal)")

    return MonoRecord(
        camera=int(camera),
        board_type="stepped",
        camera_model=cam,
        world_frame=wf,
        per_view_rms=list(pv),
        board_meta=meta,
    )


# ---------------------------------------------------------------------------
# Stereo: same-side / transmission classification + orchestrator
# ---------------------------------------------------------------------------

# Minimum sine of the angle between the two clicked axis vectors for the
# handedness sign to be trustworthy. Below this the clicks are near-collinear
# and the chirality is ambiguous, so we refuse to auto-classify.
_CHIRALITY_MIN_SIN = 0.05


def _click_chirality(fiducials: Dict[str, Sequence[float]]) -> float:
    """Signed handedness of one camera's (origin, +X, +Y) fiducial clicks.

    The 2D cross product of the two clicked axis vectors in image-down pixels:

        chi = (x_axis - origin) x (y_axis - origin)

    Its SIGN is invariant to camera roll and tilt (rotation preserves the cross
    product; foreshortening only shrinks it) as long as the camera stays on one
    side of the board plane. Crossing to the opposite face — a mirror — flips the
    sign. That is exactly the same-side vs transmission distinction, read straight
    off the clicks without touching the BFS grid (whose index ordering is not
    physically anchored). Returns +1.0 or -1.0; raises if the clicks are
    near-collinear (sign untrustworthy).
    """
    o = np.asarray(fiducials["origin"], dtype=np.float64).reshape(2)
    vx = np.asarray(fiducials["x_axis"], dtype=np.float64).reshape(2) - o
    vy = np.asarray(fiducials["y_axis"], dtype=np.float64).reshape(2) - o
    nx, ny = np.linalg.norm(vx), np.linalg.norm(vy)
    if nx < 1e-9 or ny < 1e-9:
        raise ValueError("click chirality undefined: +X or +Y click coincides with origin")
    cross = float(vx[0] * vy[1] - vx[1] * vy[0])
    if abs(cross) < _CHIRALITY_MIN_SIN * nx * ny:
        raise ValueError(
            "click chirality ambiguous: +X and +Y clicks are near-collinear "
            f"(sin angle {abs(cross) / (nx * ny):.3f} < {_CHIRALITY_MIN_SIN}); "
            "pass an explicit stereo_config"
        )
    return 1.0 if cross > 0.0 else -1.0


def classify_stereo_config(
    fiducials1: Dict[str, Sequence[float]],
    fiducials2: Dict[str, Sequence[float]],
) -> str:
    """Derive 'same_side' | 'transmission' from the two cameras' click handedness.

    Equal click chirality => both cameras view the same face => 'same_side'.
    Opposite chirality => the cameras view opposite faces => 'transmission'.
    Replaces v1's fit-cam2-twice-keep-lower-RMS auto-test with one deterministic
    geometric read (invariant 6). Raises (via ``_click_chirality``) if either
    camera's clicks are too collinear to read a sign.
    """
    chi1 = _click_chirality(fiducials1)
    chi2 = _click_chirality(fiducials2)
    return "same_side" if chi1 == chi2 else "transmission"


def calibrate_stepped_stereo(
    detections1: Sequence[DetectionResult],
    detections2: Sequence[DetectionResult],
    fiducials1: Dict[str, Sequence[float]],
    fiducials2: Dict[str, Sequence[float]],
    clicked_level1: str,
    clicked_level2: str,
    pose_levels1: Sequence[str],
    pose_levels2: Sequence[str],
    board: SteppedBoardSpec,
    image_size1: Tuple[int, int],
    image_size2: Tuple[int, int],
    cam1: int = 1,
    cam2: int = 2,
    datum_index: int = 0,
    stereo_config: str = "auto",
    distortion_model: DistortionModel = DistortionModel.STANDARD,
    model_type: str = "pinhole",
    images1: Optional[Sequence[np.ndarray]] = None,
    images2: Optional[Sequence[np.ndarray]] = None,
    figure_dir: Optional[Path] = None,
) -> StereoRecord:
    """Calibrate a stepped-board stereo pair into one shared world frame.

    Each camera is calibrated with the mono machinery (``calibrate_stepped_mono``):
    cam1 defines the shared frame with its clicked level at world Z=0; cam2 is fit
    with its levels placed at their absolute Z in that SAME frame, via the cam2
    slice of ``compute_z_and_offsets`` for the resolved same/transmission config.
    The stereo relation is then composed from the two world-frame poses
    (``compose_stereo`` — gotcha #7); no ``cv2.stereoCalibrate``, no common 3D points
    (in transmission the cameras see different faces).

    Parameters mirror ``calibrate_stepped_mono`` per camera. ``stereo_config`` is
    'auto' (classify from click handedness — the default), 'same_side', or
    'transmission'. ``clicked_level{1,2}`` is each camera's datum-origin face; for a
    checkerboard board (level_offset = spacing/2) transmission requires the two
    cameras to have clicked OPPOSITE levels at the same physical (x, y).

    ``images1``/``images2`` + ``figure_dir`` are figures-only (each camera gets its
    ``cam1_``/``cam2_`` mono proof figures, then the stereo rig + dewarp pair).

    Returns a ``StereoRecord`` whose ``world_frame`` is cam1's clicked frame.
    """
    if stereo_config == "auto":
        resolved_config = classify_stereo_config(fiducials1, fiducials2)
        logger.info(f"stepped stereo: auto-classified config = {resolved_config!r}")
    elif stereo_config in ("same_side", "transmission"):
        resolved_config = stereo_config
    else:
        raise ValueError(
            f"stereo_config must be 'auto', 'same_side', or 'transmission', "
            f"got {stereo_config!r}"
        )

    geo = compute_z_and_offsets(resolved_config, clicked_level1, clicked_level2, board)

    # A stereo run prefixes each camera's mono figures so they don't collide; the
    # cam-specific prefix also suppresses the per-camera 3D-boards plot (the stereo
    # run gets one cameras-relative-to-board figure instead).
    cam1_prefix = "cam1_" if figure_dir is not None else ""
    cam2_prefix = "cam2_" if figure_dir is not None else ""

    # Cam1 defines the shared world frame (clicked level at Z=0). Its geometry is
    # config-independent, so the mono default (== geo['Cam1']) is correct as-is.
    rec1 = calibrate_stepped_mono(
        detections=detections1, fiducials=fiducials1, clicked_level=clicked_level1,
        pose_levels=pose_levels1, board=board, image_size=image_size1,
        camera=cam1, datum_index=datum_index, distortion_model=distortion_model,
        model_type=model_type,
        images=images1, figure_dir=figure_dir, figure_prefix=cam1_prefix,
    )
    # Cam2 into the SAME frame: its levels sit at their absolute Z (geo['Cam2']).
    rec2 = calibrate_stepped_mono(
        detections=detections2, fiducials=fiducials2, clicked_level=clicked_level2,
        pose_levels=pose_levels2, board=board, image_size=image_size2,
        camera=cam2, datum_index=datum_index, distortion_model=distortion_model,
        geo_override=geo["Cam2"], model_type=model_type,
        images=images2, figure_dir=figure_dir, figure_prefix=cam2_prefix,
    )

    model1, model2 = rec1.camera_model, rec2.camera_model

    if model_type == "polynomial3d":
        # A polynomial pair has no extrinsic pose: no compose_stereo, no baseline/angle.
        # 3C reconstruction works model-agnostically from each model's project/jacobian
        # (camera_z_sign reads the stored world_z_toward_camera convention). This is the
        # DaVis poly representation -- per-camera fit quality, no rig geometry.
        logger.info(
            f"stepped stereo poly3d ({resolved_config}): cam{cam1} RMS={model1.rms_px:.3f}px, "
            f"cam{cam2} RMS={model2.rms_px:.3f}px (baseline/angle n/a for polynomial)"
        )
        meta = {
            "spacing_mm": float(board.dot_spacing_mm),
            "step_height_mm": float(board.step_height_mm),
            "board_thickness_mm": float(board.board_thickness_mm),
            "level_offset_mm": float(board.level_offset_mm),
            "stereo_config": str(resolved_config),
            "clicked_level1": str(clicked_level1),
            "clicked_level2": str(clicked_level2),
            "model_type": "polynomial3d",
            "z_sign_toward_cameras": camera_z_sign(model1, model2),
        }
        return StereoRecord(
            cam1=int(cam1), cam2=int(cam2), board_type="stepped",
            model1=model1, model2=model2,
            R_stereo=None, T_stereo=None,
            world_frame=rec1.world_frame,
            per_view_rms1=list(rec1.per_view_rms), per_view_rms2=list(rec2.per_view_rms),
            board_meta=meta,
        )

    R_stereo, T_stereo = compose_stereo(model1, model2)

    # Physical stereo angle = angle between the two optical axes in the world frame
    # (R.T @ [0,0,1] is the camera's optical axis). This is the PIV-meaningful angle,
    # NOT the rotation angle of R_stereo, which conflates the axis change with any
    # roll difference between the camera frames (ported from v1).
    axis1 = model1.R.T @ np.array([0.0, 0.0, 1.0])
    axis2 = model2.R.T @ np.array([0.0, 0.0, 1.0])
    relative_angle_deg = float(np.degrees(np.arccos(
        np.clip(float(np.dot(axis1, axis2)), -1.0, 1.0))))
    baseline_mm = float(np.linalg.norm(T_stereo))

    logger.info(
        f"stepped stereo ({resolved_config}): optical-axis angle="
        f"{relative_angle_deg:.1f}deg, baseline={baseline_mm:.1f}mm, "
        f"cam{cam1} RMS={model1.rms:.3f}, cam{cam2} RMS={model2.rms:.3f}"
    )

    meta = {
        "spacing_mm": float(board.dot_spacing_mm),
        "step_height_mm": float(board.step_height_mm),
        "board_thickness_mm": float(board.board_thickness_mm),
        "level_offset_mm": float(board.level_offset_mm),
        "stereo_config": str(resolved_config),
        "clicked_level1": str(clicked_level1),
        "clicked_level2": str(clicked_level2),
        "baseline_mm": baseline_mm,
        "relative_angle_deg": relative_angle_deg,
        "z_sign_toward_cameras": camera_z_sign(model1, model2),
    }

    # ---- Stereo-only rig figures (cameras-vs-board + dewarp anaglyph) ----
    # cam1's mono figures were already written above; this adds the geometry pair.
    # The datum board world points are recomputed via the SAME helpers the mono path
    # uses (deterministic — not a divergent re-derivation), purely to draw the board.
    if figure_dir is not None and images1 is not None and images2 is not None:
        from . import figures as c2figs
        try:
            datum1 = detections1[datum_index]
            lv_A, lv_B = _level_dicts(datum1)
            absolute = assign_absolute_grid_indices(
                fiducials1, lv_A, lv_B, clicked_level1, board)
            other1 = 'trough' if clicked_level1 == 'peak' else 'peak'
            geo_cam1 = geo["Cam1"]
            board_world = np.vstack([
                build_object_points(
                    absolute[lvl]['grid_indices'], geo_cam1['z'][lvl],
                    board.dot_spacing_mm, geo_cam1['xy_offset'][lvl])
                for lvl in (clicked_level1, other1) if absolute.get(lvl) is not None
            ]).astype(np.float64)
            c2figs.write_stereo_figures(
                figure_dir, model1=model1, model2=model2,
                R_stereo=R_stereo, T_stereo=T_stereo,
                img1=images1[datum_index], img2=images2[datum_index],
                datum_board_world=board_world, spacing=float(board.dot_spacing_mm),
            )
        except Exception:  # figures never abort the fit
            logger.warning("stepped stereo rig figure writing failed (non-fatal)")

    return StereoRecord(
        cam1=int(cam1), cam2=int(cam2), board_type="stepped",
        model1=model1, model2=model2,
        R_stereo=R_stereo, T_stereo=T_stereo,
        world_frame=rec1.world_frame,
        per_view_rms1=list(rec1.per_view_rms), per_view_rms2=list(rec2.per_view_rms),
        board_meta=meta,
    )
