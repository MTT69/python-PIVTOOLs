"""
Stepped Board Stereo Calibration — Production Module.

Pinhole camera calibration using stepped dotboard targets with dots on two
alternating Z-planes (peak/trough). Extracted from prototype in
manual_tools/stepped_calibration_board/.

Pipeline:
  1. Blob detection (flat-field + contour/ellipse)
  2. Row-based level separation (alternating rows → two Z-planes)
  3. Per-level grid detection (BFS + RANSAC)
  4. Fiducial-based absolute grid index assignment
  5. Two-step pinhole fitting per camera
  6. Stereo calibration (fixed intrinsics)
  7. Save per-camera + stereo models
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io
from loguru import logger
from scipy.spatial import cKDTree

from pivtools_gui.calibration.grid_detection import (
    _bfs_grid_walk_dict,
    _filter_connected_dict,
    _find_grid_directions,
    _rescue_missing_dots,
    detect_dotboard_blobs,
    find_largest_grid_component,
    to_grayscale_2d,
)
from pivtools_gui.calibration.calibration_io import (
    read_calibration_image_with_fallback,
    find_calibration_images,
    get_camera_input_dir,
)


# ============================================================================
# Board Specification
# ============================================================================

@dataclass
class SteppedBoardSpec:
    dot_spacing_mm: float = 15.0
    step_height_mm: float = 3.0
    level_offset_mm: Optional[float] = None
    board_thickness_mm: float = 14.8

    def __post_init__(self):
        if self.level_offset_mm is None:
            self.level_offset_mm = self.dot_spacing_mm / 2.0


# ============================================================================
# Helpers
# ============================================================================


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


# ============================================================================
# Detection Functions
# ============================================================================

def find_grid_vectors(centers: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Find the two dominant grid direction vectors.

    Delegates to the shared ``_find_grid_directions`` with k=5 (critical
    for stepped boards after level separation — k>5 pulls in diagonals
    to the removed level's row positions).

    Returns (vec1, vec2, spacing_px) or None.
    """
    n = len(centers)
    if n < 9:
        return None

    tree = cKDTree(centers)
    nn_dists, _ = tree.query(centers, k=2)
    spacing_px = float(np.median(nn_dists[:, 1]))

    result = _find_grid_directions(centers, spacing_px, k=5)
    if result is None:
        return None

    v1, v2 = result
    return v1, v2, spacing_px


def run_single_level_detection(
    centers: np.ndarray,
    original_gray: np.ndarray,
    flat_field: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Full single-level pipeline: direction finding (k=5) → reciprocal BFS →
    RANSAC → template rescue → connected component filter.

    Parameters
    ----------
    centers : ndarray, shape (N, 2)
        Blob centers for this level only.
    original_gray : ndarray
        Original grayscale image (for diagnostics).
    flat_field : ndarray, optional
        Flat-field image from blob detection (for template rescue).
    """
    result = find_grid_vectors(centers)
    if result is None:
        return None
    vec1, vec2, spacing_px = result

    # Reciprocal BFS grid walk
    tree = cKDTree(centers)
    grid_dict = _bfs_grid_walk_dict(centers, vec1, vec2, tree)

    if len(grid_dict) < 9:
        return None

    # RANSAC homography validation
    grid_keys = list(grid_dict.keys())
    src_pts = np.array(grid_keys, dtype=np.float32)
    dst_pts = np.array([centers[grid_dict[k]] for k in grid_keys], dtype=np.float32)

    ransac_thresh = 0.15 * spacing_px
    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000, confidence=0.995,
    )
    if H is None:
        return None

    inlier_mask = inlier_mask.flatten().astype(bool)
    validated = {k: grid_dict[k] for k, inl in zip(grid_keys, inlier_mask) if inl}

    if len(validated) < 9:
        return None

    # Re-fit clean homography
    v_keys = list(validated.keys())
    v_src = np.array(v_keys, dtype=np.float32)
    v_dst = np.array([centers[validated[k]] for k in v_keys], dtype=np.float32)
    H_clean, _ = cv2.findHomography(v_src, v_dst, method=0)
    if H_clean is not None:
        H = H_clean

    # Template rescue for missing interior dots
    if flat_field is not None:
        validated, rescued_centers, rescued_nodes = _rescue_missing_dots(
            validated, centers, flat_field, spacing_px,
        )
        all_centers = rescued_centers
    else:
        all_centers = list(centers)

    # Prune orphaned points
    validated = _filter_connected_dict(validated)

    # Convert to ndarray format (keep raw BFS indices — no zero-basing.
    # assign_absolute_grid_indices re-anchors to the fiducial click, so
    # zero-basing is redundant and would invalidate H.)
    final_keys = list(validated.keys())
    final_idx = [validated[k] for k in final_keys]
    ctrs = np.array([all_centers[i] for i in final_idx], dtype=np.float32)
    gi = np.array(final_keys, dtype=np.int32)

    # Largest connected component
    if len(ctrs) >= 9:
        comp_mask, n_comp, _ = find_largest_grid_component(gi)
        if n_comp > 1:
            ctrs = ctrs[comp_mask]
            gi = gi[comp_mask]

    if len(ctrs) < 9:
        return None

    n_cols = int(gi[:, 0].max() - gi[:, 0].min()) + 1
    n_rows = int(gi[:, 1].max() - gi[:, 1].min()) + 1

    return {
        'centers': ctrs,
        'grid_indices': gi,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'n_points': len(ctrs),
        'vec1': vec1,
        'vec2': vec2,
        'spacing_px': spacing_px,
        'H': H,
    }


def cluster_into_rows(centers: np.ndarray, spacing_px: float) -> Tuple[np.ndarray, List[float]]:
    """Cluster blob centers into horizontal rows by y-coordinate.

    For a two-level board with diagonal interleaving, consecutive rows alternate
    between levels and are spaced by ~half the same-level grid spacing in y.
    """
    gap_thresh = spacing_px * 0.3

    # Sort by y
    sorted_idx = np.argsort(centers[:, 1])
    sorted_y = centers[sorted_idx, 1]
    dy = np.diff(sorted_y)

    row_labels_sorted = np.zeros(len(centers), dtype=int)
    current_row = 0
    for i in range(len(dy)):
        if dy[i] > gap_thresh:
            current_row += 1
        row_labels_sorted[i + 1] = current_row

    row_labels = np.zeros(len(centers), dtype=int)
    row_labels[sorted_idx] = row_labels_sorted

    n_rows = current_row + 1
    row_y_values = []
    row_counts = []
    for r in range(n_rows):
        mask_r = row_labels == r
        row_y_values.append(float(np.median(centers[mask_r, 1])))
        row_counts.append(int(np.sum(mask_r)))

    # Split over-populated rows
    median_count = float(np.median(row_counts))
    split_thresh = median_count * 1.3
    splits_done = True
    while splits_done:
        splits_done = False
        for r in range(n_rows):
            mask_r = row_labels == r
            count = int(np.sum(mask_r))
            if count <= split_thresh:
                continue

            row_indices = np.where(mask_r)[0]
            row_ys = centers[row_indices, 1]
            order = np.argsort(row_ys)
            sorted_ys = row_ys[order]
            sorted_indices = row_indices[order]

            dy_internal = np.diff(sorted_ys)
            if len(dy_internal) == 0:
                continue

            max_gap_pos = int(np.argmax(dy_internal))
            max_gap = dy_internal[max_gap_pos]

            if max_gap < gap_thresh * 0.3:
                continue

            logger.debug(f"Splitting row {r} ({count} dots, max internal gap={max_gap:.1f}px)")

            new_label = n_rows
            for idx in sorted_indices[max_gap_pos + 1:]:
                row_labels[idx] = new_label
            n_rows += 1
            splits_done = True
            break

    # Renumber in y-order
    unique_rows = sorted(set(row_labels))
    row_medians = []
    for r in unique_rows:
        mask_r = row_labels == r
        row_medians.append((r, float(np.median(centers[mask_r, 1]))))
    row_medians.sort(key=lambda x: x[1])

    old_to_new = {old: new for new, (old, _) in enumerate(row_medians)}
    row_labels = np.array([old_to_new[r] for r in row_labels], dtype=int)

    n_rows = len(row_medians)
    row_y_values = [y for _, y in row_medians]

    return row_labels, row_y_values


def separate_levels(centers: np.ndarray, row_labels: np.ndarray,
                    row_y_values: List[float]) -> dict:
    """Separate dots into two levels using alternating row parity.

    Even rows → Level A, odd rows → Level B (whichever parity has more
    dots gets Level A). Peak/trough assignment is determined later by
    user fiducial clicks, not by auto-detection.

    Returns dict with 'mask_level_A', 'mask_level_B', 'n_rows'.
    """
    n_rows = len(row_y_values)

    # Drop rows with too few dots
    row_counts = np.bincount(row_labels, minlength=n_rows)
    median_count = float(np.median(row_counts[row_counts > 0]))
    min_row_dots = max(5, median_count * 0.3)

    keep_mask = np.ones(len(centers), dtype=bool)
    for r in range(n_rows):
        if row_counts[r] < min_row_dots:
            keep_mask[row_labels == r] = False

    if not np.all(keep_mask):
        centers_clean = centers[keep_mask]
        tree = cKDTree(centers_clean)
        nn_dists = tree.query(centers_clean, k=2)[0]
        spacing_px = float(np.median(nn_dists[:, 1]))
        row_labels_clean, row_y_values = cluster_into_rows(centers_clean, spacing_px)
    else:
        centers_clean = centers
        row_labels_clean = row_labels

    n_rows = len(row_y_values)

    even_rows = row_labels_clean % 2 == 0
    odd_rows = row_labels_clean % 2 == 1

    # Level A = whichever parity has more dots
    n_even = int(np.sum(even_rows))
    n_odd = int(np.sum(odd_rows))
    level_A_mask = even_rows if n_even >= n_odd else odd_rows
    level_B_mask = odd_rows if n_even >= n_odd else even_rows

    return {
        'centers': centers_clean,
        'row_labels': row_labels_clean,
        'mask_level_A': level_A_mask,
        'mask_level_B': level_B_mask,
        'n_rows': n_rows,
    }


# ============================================================================
# Absolute Grid Index Assignment
# ============================================================================

def assign_absolute_grid_indices(fiducials: dict, level_A_data: dict,
                                  level_B_data: Optional[dict],
                                  clicked_level: str,
                                  board: SteppedBoardSpec) -> dict:
    """Assign absolute grid indices using user-clicked origin.

    The clicked dot becomes grid index (0, 0). All other dots are relative to it.

    Args:
        fiducials: dict with 'origin', 'x_axis', 'y_axis' keys, each having 'snapped_px'
        level_A_data: detection result for level A (from run_single_level_detection)
        level_B_data: detection result for level B (may be None)
        clicked_level: 'peak' or 'trough'
        board: board specification

    Returns:
        dict with 'peak' and/or 'trough' keys, each containing 'centers' and 'grid_indices'
    """
    origin_px = np.array(fiducials['origin'], dtype=np.float32)
    x_axis_px = np.array(fiducials['x_axis'], dtype=np.float32)
    y_axis_px = np.array(fiducials['y_axis'], dtype=np.float32)

    # Determine which level the clicked dot belongs to using homography residual
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

    logger.debug(f"Origin on {'A' if res_A <= res_B else 'B'} (residual A={res_A:.3f}, B={res_B:.3f})")

    # Inject origin click into the origin level's grid
    origin_data, origin_idx_in_level = _inject_click_into_level(origin_data, origin_px)

    # Inject +X and +Y clicks into whichever level they belong to
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

    # Determine axis orientation from fiducial vectors vs BFS vectors
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

    # Origin level: transform BFS → absolute (origin = grid (0,0))
    abs_gi_origin = transform_indices(origin_data['grid_indices'].copy(), origin_bfs)

    result = {}
    result[clicked_level] = {
        'centers': origin_data['centers'].copy(),
        'grid_indices': abs_gi_origin,
    }

    # Other level: BFS + anchor offset
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
            logger.debug(f"{other_level} anchoring: offset=({abs_offset[0]}, {abs_offset[1]}), "
                         f"consensus={consensus_pct:.0f}% ({best_count}/{len(anchor_offsets)})")

            abs_gi_other = oriented_other_gi + abs_offset

            result[other_level] = {
                'centers': other_centers,
                'grid_indices': abs_gi_other,
            }
        else:
            logger.warning(f"No anchor points found for {other_level} level")

    # Expose the fiducial-derived orientation so non-datum poses (which
    # run through stitch_levels_pose_local) can apply the SAME convention
    # instead of re-deriving it from raw BFS vectors. Without this,
    # back-view cameras (image +x = physical -x) silently disagree with
    # their own datum and cv2.calibrateCamera blows up the intrinsic fit.
    result['orientation'] = {
        'swap_axes': bool(swap_axes),
        'col_sign': int(col_sign),
        'row_sign': int(row_sign),
    }
    # Which of level_A / level_B the fiducial origin click landed on.
    # Lets generate_model resolve physical peak/trough without redoing
    # the homography-residual comparison.
    result['_origin_on_level'] = 'A' if res_A <= res_B else 'B'

    return result


# ============================================================================
# Pose-local level stitching (no fiducials required)
# ============================================================================

def stitch_levels_pose_local(
    level_A_data: Optional[dict],
    level_B_data: Optional[dict],
    board: SteppedBoardSpec,
    orientation_override: Optional[dict] = None,
) -> Optional[dict]:
    """Stitch level A and level B into a single pose-local shared frame,
    using only nearest-neighbour geometry (no fiducial clicks needed).

    This is the fiducial-free equivalent of `assign_absolute_grid_indices`
    for non-datum poses. It uses the same cross-level anchoring logic
    (each "other level" dot is located relative to its 4 nearest
    "reference level" neighbours via the known ±half-spacing physical
    offset), but uses the BFS walk's own indices as the reference frame
    instead of a user-clicked origin. The output frame is pose-local —
    its absolute position is arbitrary — which is fine because
    `cv2.calibrateCamera` treats each view's object points as an
    independent world frame for intrinsic fitting. Only the datum pose
    needs a globally meaningful frame (via `assign_absolute_grid_indices`
    with the user's click) because that frame feeds the stereo pose
    composition.

    Parameters
    ----------
    level_A_data, level_B_data : dict or None
        Output of `run_single_level_detection`.
    board : SteppedBoardSpec
        Board geometry (needs dot_spacing_mm and level_offset_mm).
    orientation_override : dict or None
        Optional `{'swap_axes', 'col_sign', 'row_sign'}` from the datum
        pose's `assign_absolute_grid_indices` result. When provided, the
        stitch uses these sign constants instead of deriving them from
        raw BFS `vec1`/`vec2` dot products. This is REQUIRED for any
        camera viewing the board from behind (image `+x` → physical
        `−x`) because the raw BFS direction disagrees with the
        fiducial-anchored datum, leading to a chirality mismatch that
        blows up the multi-view intrinsic fit. When None, the legacy
        BFS-derived logic runs — safe for front-view cameras.

    Returns
    -------
    dict with keys 'reference' and optionally 'other', each containing
    'centers', 'grid_indices', and a 'source_level' flag ('A' or 'B'),
    plus a 'metadata' sub-dict carrying the stitch audit trail:
    n_reference, n_other, n_anchors, consensus_pct, swap_axes, col_sign,
    row_sign, degraded_single_level. `degraded_single_level` is True
    when only one level was detected (no cross-level anchoring possible)
    — callers should surface this as a warning. Returns None if no
    usable level was detected.
    """
    if level_A_data is None and level_B_data is None:
        return None
    if level_A_data is None:
        return {
            'reference': {
                'centers': np.asarray(level_B_data['centers']).copy(),
                'grid_indices': np.asarray(level_B_data['grid_indices']).copy(),
                'source_level': 'B',
            },
            'metadata': {
                'n_reference': int(len(level_B_data['centers'])),
                'n_other': 0,
                'n_anchors': 0,
                'consensus_pct': float('nan'),
                'swap_axes': False,
                'col_sign': 1,
                'row_sign': 1,
                'degraded_single_level': True,
            },
        }
    if level_B_data is None:
        return {
            'reference': {
                'centers': np.asarray(level_A_data['centers']).copy(),
                'grid_indices': np.asarray(level_A_data['grid_indices']).copy(),
                'source_level': 'A',
            },
            'metadata': {
                'n_reference': int(len(level_A_data['centers'])),
                'n_other': 0,
                'n_anchors': 0,
                'consensus_pct': float('nan'),
                'swap_axes': False,
                'col_sign': 1,
                'row_sign': 1,
                'degraded_single_level': True,
            },
        }

    # Pick the larger level as the reference frame
    n_A = len(level_A_data['centers'])
    n_B = len(level_B_data['centers'])
    if n_A >= n_B:
        ref_data = level_A_data
        ref_letter = 'A'
        other_data = level_B_data
        other_letter = 'B'
    else:
        ref_data = level_B_data
        ref_letter = 'B'
        other_data = level_A_data
        other_letter = 'A'

    ref_centers = np.asarray(ref_data['centers']).copy()
    ref_gi = np.asarray(ref_data['grid_indices']).copy()

    # Decide the axis orientation for the other level's BFS grid. If the
    # caller supplied an orientation_override (from the datum pose's
    # fiducial-anchored result), trust it verbatim. Otherwise derive the
    # orientation from raw BFS vec1/vec2 dot products — the legacy path,
    # which is wrong for back-view cameras whose image axes don't agree
    # with the fiducial convention.
    if orientation_override is not None:
        swap_axes = bool(orientation_override['swap_axes'])
        col_sign = int(orientation_override['col_sign'])
        row_sign = int(orientation_override['row_sign'])
    else:
        ref_vec1 = np.asarray(ref_data.get('vec1', [1.0, 0.0]), dtype=np.float64)
        ref_vec2 = np.asarray(ref_data.get('vec2', [0.0, 1.0]), dtype=np.float64)
        other_vec1 = np.asarray(other_data.get('vec1', [1.0, 0.0]), dtype=np.float64)
        other_vec2 = np.asarray(other_data.get('vec2', [0.0, 1.0]), dtype=np.float64)

        def _unit(v):
            n = float(np.linalg.norm(v))
            return v / n if n > 1e-9 else v

        ref_v1u = _unit(ref_vec1)
        ref_v2u = _unit(ref_vec2)
        other_v1u = _unit(other_vec1)
        other_v2u = _unit(other_vec2)

        # Dot products tell us how `other`'s axes map onto `ref`'s
        d11 = float(np.dot(other_v1u, ref_v1u))
        d12 = float(np.dot(other_v1u, ref_v2u))
        d21 = float(np.dot(other_v2u, ref_v1u))
        d22 = float(np.dot(other_v2u, ref_v2u))

        if abs(d11) >= abs(d12):
            # other.vec1 → ref.vec1 (maybe flipped)
            swap_axes = False
            col_sign = 1 if d11 > 0 else -1
            row_sign = 1 if d22 > 0 else -1
        else:
            # other.vec1 → ref.vec2 (swapped)
            swap_axes = True
            col_sign = 1 if d12 > 0 else -1
            row_sign = 1 if d21 > 0 else -1

    # Apply the same orientation transform to BOTH levels so anchor
    # offsets are computed in a single consistent frame. The datum path
    # (assign_absolute_grid_indices) does this via transform_indices —
    # we must do the same here, otherwise col_sign/row_sign mismatches
    # scatter the offsets and tank consensus.
    def _orient(gi):
        out = gi.copy()
        if swap_axes:
            out = out[:, ::-1]
        out[:, 0] = col_sign * out[:, 0]
        out[:, 1] = row_sign * out[:, 1]
        return out

    oriented_ref_gi = _orient(ref_gi)
    other_bfs_gi = np.asarray(other_data['grid_indices'], dtype=np.int32).copy()
    oriented_other_gi = _orient(other_bfs_gi)

    # Cross-level anchor — identical to assign_absolute_grid_indices'
    # logic, but operating on the pose-local reference frame instead
    # of the fiducial-anchored absolute frame.
    spacing = board.dot_spacing_mm
    level_offset = board.level_offset_mm
    ref_phys_x = oriented_ref_gi[:, 0].astype(float) * spacing
    ref_phys_y = oriented_ref_gi[:, 1].astype(float) * spacing

    ref_tree = cKDTree(ref_centers)
    other_centers = np.asarray(other_data['centers']).copy()
    n_other = len(other_centers)
    if n_other == 0:
        return {
            'reference': {
                'centers': ref_centers, 'grid_indices': oriented_ref_gi,
                'source_level': ref_letter,
            },
            'metadata': {
                'n_reference': int(len(ref_centers)),
                'n_other': 0,
                'n_anchors': 0,
                'consensus_pct': float('nan'),
                'swap_axes': bool(swap_axes),
                'col_sign': int(col_sign),
                'row_sign': int(row_sign),
                'degraded_single_level': True,
            },
        }

    anchor_offsets = []
    for i in range(n_other):
        dists, idxs = ref_tree.query(other_centers[i], k=min(4, len(ref_centers)))
        idxs = np.atleast_1d(idxs)
        neighbor_gi = oriented_ref_gi[idxs]
        gi_range = neighbor_gi.max(axis=0) - neighbor_gi.min(axis=0)
        # Skip dots whose 4 nearest neighbours aren't a unit-square cell
        # (i.e. edge dots where the grid corner collapses the cell).
        if gi_range[0] != 1 or gi_range[1] != 1:
            continue
        mean_phys_x = float(np.mean(ref_phys_x[idxs]))
        mean_phys_y = float(np.mean(ref_phys_y[idxs]))
        expected_col = round((mean_phys_x - level_offset) / spacing)
        expected_row = round((mean_phys_y - level_offset) / spacing)
        expected_abs = np.array([expected_col, expected_row], dtype=np.int32)
        off = expected_abs - oriented_other_gi[i]
        anchor_offsets.append(off)

    if not anchor_offsets:
        logger.warning(
            f"stitch_levels_pose_local: no anchor points for {other_letter} "
            f"level ({n_other} dots, reference has {len(ref_centers)})"
        )
        return {
            'reference': {
                'centers': ref_centers, 'grid_indices': ref_gi,
                'source_level': ref_letter,
            },
            'metadata': {
                'n_reference': int(len(ref_centers)),
                'n_other': int(n_other),
                'n_anchors': 0,
                'consensus_pct': float('nan'),
                'swap_axes': bool(swap_axes),
                'col_sign': int(col_sign),
                'row_sign': int(row_sign),
                'degraded_single_level': True,
            },
        }

    anchor_arr = np.array(anchor_offsets)
    offset_tuples = [tuple(o) for o in anchor_arr]
    best_offset, best_count = Counter(offset_tuples).most_common(1)[0]
    abs_offset = np.array(best_offset, dtype=np.int32)
    consensus_pct = 100 * best_count / len(anchor_offsets)
    logger.debug(
        f"stitch_levels_pose_local: {other_letter} consensus offset="
        f"({abs_offset[0]}, {abs_offset[1]}), "
        f"consensus={consensus_pct:.0f}% ({best_count}/{len(anchor_offsets)}), "
        f"swap_axes={swap_axes}, col_sign={col_sign}, row_sign={row_sign}"
    )

    other_gi_final = oriented_other_gi + abs_offset
    return {
        'reference': {
            'centers': ref_centers, 'grid_indices': oriented_ref_gi,
            'source_level': ref_letter,
        },
        'other': {
            'centers': other_centers, 'grid_indices': other_gi_final,
            'source_level': other_letter,
        },
        'metadata': {
            'n_reference': int(len(ref_centers)),
            'n_other': int(n_other),
            'n_anchors': int(len(anchor_offsets)),
            'consensus_pct': float(consensus_pct),
            'swap_axes': bool(swap_axes),
            'col_sign': int(col_sign),
            'row_sign': int(row_sign),
            'degraded_single_level': False,
        },
    }


# ============================================================================
# Object Point + Z Assignment
# ============================================================================

def build_object_points(grid_indices: np.ndarray, z_mm: float,
                        spacing_mm: float, xy_offset_mm: float = 0.0) -> np.ndarray:
    """Create 3D world object points from grid indices."""
    n = len(grid_indices)
    obj = np.zeros((n, 3), dtype=np.float32)
    obj[:, 0] = grid_indices[:, 0] * spacing_mm + xy_offset_mm
    obj[:, 1] = grid_indices[:, 1] * spacing_mm + xy_offset_mm
    obj[:, 2] = z_mm
    return obj


def compute_z_and_offsets(stereo_config: str, cam1_clicked_level: str,
                          cam2_clicked_level: str, board: SteppedBoardSpec) -> dict:
    """Compute Z values and XY offsets for each camera's levels.

    Convention: cam1's clicked level is at Z=0 (reference plane).
    The clicked level on both cameras has xy_offset=0.
    The other level has xy_offset=level_offset_mm.
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
        # Transmission: cam2 sees opposite face
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


# ============================================================================
# Pinhole Fitting
# ============================================================================

def fit_pinhole(obj_views: list, img_views: list,
                image_size: Tuple[int, int],
                fix_aspect: bool = False
                ) -> Tuple[float, np.ndarray, np.ndarray, list, list]:
    """Fit OpenCV pinhole model using multiple views of 3D object points.

    Two-step approach:
      1. Multi-image Zhang init: extract one coplanar z-level from
         every pose, feed all of them as separate "images" to
         cv2.calibrateCamera (K=None).  Each pose contributes one
         homography; OpenCV solves the overdetermined IAC system
         from all homographies jointly — the same mechanism that
         makes dotboard/charuco calibration robust at PIV
         magnification.  The z-level closest to z=0 is chosen
         (best-conditioned Zhang problem).
      2. Refine with all views using true 3D coordinates (both
         z-levels).  The stepped board's two-Z non-coplanarity
         breaks the fx-tz ridge that the coplanar init cannot
         resolve alone.

    Returns (rms, K, dist, rvecs, tvecs).
    """
    W, H = image_size

    # Step 1: Multi-image Zhang init on one coplanar level from ALL
    # poses.  Each pose's single-level subset is a separate "image"
    # for Zhang, giving N homographies instead of 1.
    unique_z = np.unique(np.round(obj_views[0][:, 2], 2))
    z_for_init = unique_z[np.argmin(np.abs(unique_z))]

    init_objs = []
    init_imgs = []
    for obj, img in zip(obj_views, img_views):
        mask = np.abs(obj[:, 2] - z_for_init) < 0.5
        if mask.sum() < 4:
            continue
        obj_z0 = np.column_stack([
            obj[mask, 0], obj[mask, 1],
            np.zeros(mask.sum())
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


# ============================================================================
# Main Calibrator Class
# ============================================================================

class SteppedCalibrator:
    """Stepped board calibration: dot detection on two Z-planes + pinhole fitting."""

    # Subclasses override to redirect saved model output (e.g.
    # `SteppedPlanarCalibrator` uses "stepped_planar").
    _subdir_name: str = "stepped_board"

    def __init__(self, config=None, source_path_idx=0, base_path_idx=0,
                 camera_pair=None, dot_spacing_mm=15.0, step_height_mm=3.0,
                 board_thickness_mm=14.8, dt=1.0, datum_frame=1):
        if config is not None:
            cfg = config.stepped_board_calibration
            self.dot_spacing_mm = cfg.get('dot_spacing_mm', dot_spacing_mm)
            self.step_height_mm = cfg.get('step_height_mm', step_height_mm)
            self.board_thickness_mm = cfg.get('board_thickness_mm', board_thickness_mm)
            self.dt = cfg.get('dt', dt)
            self.camera_pair = cfg.get('camera_pair', camera_pair or [1, 2])
            self.datum_frame = cfg.get('datum_frame', datum_frame)
            self.source_path_idx = source_path_idx
            self.base_dir = Path(config.base_paths[base_path_idx])
            self._config = config
        else:
            self.dot_spacing_mm = dot_spacing_mm
            self.step_height_mm = step_height_mm
            self.board_thickness_mm = board_thickness_mm
            self.dt = dt
            self.camera_pair = camera_pair or [1, 2]
            self.datum_frame = datum_frame
            self.source_path_idx = source_path_idx
            self.base_dir = None
            self._config = None

        self.board = SteppedBoardSpec(
            dot_spacing_mm=self.dot_spacing_mm,
            step_height_mm=self.step_height_mm,
            board_thickness_mm=self.board_thickness_mm,
        )

    # --- Phase 1: Detection ---

    def detect_single_camera(self, camera: int, frame_idx: int) -> dict:
        """Detect blobs + separate levels + per-level grid.

        Returns dict with keys: blobs, level_A, level_B,
        image_size, plus internal _-prefixed keys
        """
        # Read image
        file_pattern = self._config.calibration_image_format
        if isinstance(file_pattern, tuple):
            file_pattern = file_pattern[0]
        img = read_calibration_image_with_fallback(
            None, camera, frame_idx, self._config, self.source_path_idx, file_pattern
        )
        if img is None:
            raise FileNotFoundError(f"Could not read calibration image for camera {camera}, frame {frame_idx}")

        gray = to_grayscale_2d(img)
        gray_f = gray.astype(np.float64)
        vmax = gray_f.max()
        if vmax > 0:
            gray_uint8 = (gray_f / vmax * 255).astype(np.uint8)
        else:
            gray_uint8 = np.zeros_like(gray, dtype=np.uint8)

        h, w = gray_uint8.shape[:2]

        # Stage 1: Blob detection (flat-field + contour/ellipse pipeline)
        blobs, blob_info = detect_dotboard_blobs(gray_uint8)
        if len(blobs) < 9:
            raise ValueError(f"Blob detection failed for camera {camera}")

        # Stage 2: Row-based level separation
        tree = cKDTree(blobs)
        nn_dists = tree.query(blobs, k=2)[0]
        spacing_px = float(np.median(nn_dists[:, 1]))

        logger.info(f"Camera {camera}: {len(blobs)} blobs detected")

        row_labels, row_y_values = cluster_into_rows(blobs, spacing_px)
        level_info = separate_levels(blobs, row_labels, row_y_values)

        centers_clean = level_info['centers']
        level_A_mask = level_info['mask_level_A']
        level_B_mask = level_info['mask_level_B']

        # Stage 3: Per-level grid detection with template rescue
        flat_field = blob_info.get('flat_field')
        level_A = run_single_level_detection(centers_clean[level_A_mask], gray_uint8, flat_field=flat_field)
        level_B = run_single_level_detection(centers_clean[level_B_mask], gray_uint8, flat_field=flat_field)

        if level_A is not None:
            logger.info(f"Camera {camera} Level A: {level_A['n_cols']}x{level_A['n_rows']} grid, "
                        f"{level_A['n_points']} points")
        if level_B is not None:
            logger.info(f"Camera {camera} Level B: {level_B['n_cols']}x{level_B['n_rows']} grid, "
                        f"{level_B['n_points']} points")

        return {
            'blobs': blobs.tolist(),
            'level_A': {
                'centers': level_A['centers'].tolist() if level_A else [],
                'n_points': level_A['n_points'] if level_A else 0,
                'grid_indices': level_A['grid_indices'].tolist() if level_A and 'grid_indices' in level_A else [],
            },
            'level_B': {
                'centers': level_B['centers'].tolist() if level_B else [],
                'n_points': level_B['n_points'] if level_B else 0,
                'grid_indices': level_B['grid_indices'].tolist() if level_B and 'grid_indices' in level_B else [],
            },
            'image_size': [h, w],
            # Internal data for model generation (not serialized to JSON)
            '_level_A_full': level_A,
            '_level_B_full': level_B,
            '_gray_uint8': gray_uint8,
            '_centers_clean': centers_clean,
            '_level_A_mask': level_A_mask,
            '_level_B_mask': level_B_mask,
            '_blob_info': blob_info,
        }

    @staticmethod
    def snap_to_nearest(click_px: tuple, detection_data: dict) -> dict:
        """Snap a fiducial click to the nearest detected blob.

        Returns: {snapped_x, snapped_y, snap_dist}
        """
        blobs = np.array(detection_data['blobs'], dtype=np.float32)
        tree = cKDTree(blobs)
        dist, idx = tree.query(np.array(click_px, dtype=np.float32))
        return {
            'snapped_x': float(blobs[idx][0]),
            'snapped_y': float(blobs[idx][1]),
            'snap_dist': float(dist),
        }

    # --- Phase 2: Model Generation ---

    def _build_non_datum_pose_view(self, level_A, level_B, geo_cam,
                                    A_label, B_label,
                                    orientation_override=None):
        """Build a NON-DATUM pose's (obj_points, img_points) using
        BOTH detected levels, stitched into one pose-local frame via
        `stitch_levels_pose_local`. That helper uses the same
        nearest-neighbour cross-level anchoring logic as
        `assign_absolute_grid_indices` — but without requiring a
        fiducial click — so the two BFS sub-lattices end up in a
        single shared pose-local frame where `cv2.calibrateCamera`
        can fit them as one rigid view.

        `orientation_override` (required in practice) is forwarded to
        `stitch_levels_pose_local` so every non-datum stitch reuses
        the datum's fiducial-derived (swap_axes, col_sign, row_sign)
        convention instead of re-deriving it from raw BFS vectors.
        Without the override, strongly foreshortened poses (cam with
        +ry≈18°) produce BFS dot-product signs that can invert and
        leave one view in a mirrored chirality relative to the rest,
        blowing up `cv2.calibrateCamera`.

        The per-dot `xy_offset` + `z` assignment comes from the
        operator-supplied `A_label`/`B_label` (via `compute_z_and_offsets`).
        `stitch_levels_pose_local` owns the inter-level integer offset
        (which BFS `(col, row)` of the 'other' sub-lattice corresponds
        to the 'reference' sub-lattice's origin) via its 4-NN anchor
        consensus — no downstream refinement. An earlier correlation-
        based refinement used to run here but was removed after
        `orientation_override` was wired: on clean data it always
        settled on `(0, 0)`, and on strongly-foreshortened poses its
        pdist-correlation landscape got flat enough that the argmax
        picked a spurious `(0, ±1)` shift on a <0.002 margin,
        silently corrupting the world frame. Trust the stitch.

        Returns (obj_np, img_np, per_level_obj, stitch_meta) or
        (None, None, None, None) if no usable level was detected.
        """
        spacing = self.board.dot_spacing_mm
        stitched = stitch_levels_pose_local(
            level_A, level_B, self.board,
            orientation_override=orientation_override,
        )
        if stitched is None:
            return None, None, None, None

        ref = stitched['reference']
        other = stitched.get('other')
        stitch_meta = stitched.get('metadata', {})

        def _level_name(source_letter):
            return A_label if source_letter == 'A' else B_label

        all_obj = []
        all_img = []
        all_gi = []  # stitched (col, row) per dot for the figure helper
        per_level_obj = {}
        level_source_per_dot = []

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
            all_gi.append(gi)
            per_level_obj[level_name] = obj
            level_source_per_dot.extend([level_name] * len(centers))

        if not all_obj:
            return None, None, None, None

        stitch_meta = dict(stitch_meta)
        stitch_meta['A_label'] = A_label
        stitch_meta['B_label'] = B_label
        stitch_meta['level_source_per_dot'] = level_source_per_dot
        stitch_meta['centers_stacked'] = np.vstack(all_img)
        stitch_meta['gi_stacked'] = np.vstack(all_gi)
        return (
            np.vstack(all_obj),
            np.vstack(all_img),
            per_level_obj,
            stitch_meta,
        )

    def _fit_per_camera_pinhole(
        self,
        cam_num: int,
        detections_per_pose: list,
        absolute_datum: dict,
        geo_cam: dict,
        image_size: tuple,
        clicked_level: str,
        datum_pose_index: int,
        pose_level_labels_by_position: list,
    ) -> dict:
        """Build per-pose (obj, img) views for ONE camera from the
        datum's fiducial-anchored indices + operator-provided per-pose
        peak/trough labels, then run cv2.calibrateCamera jointly across
        all poses.

        Used by both SteppedCalibrator (twice per stereo pair) and
        SteppedPlanarCalibrator (once per camera). Stereo composition
        is performed by the caller.

        Parameters
        ----------
        cam_num : int
        detections_per_pose : list of dicts keyed by str(cam_num)
        absolute_datum : dict from assign_absolute_grid_indices for
            the datum pose (already fiducial-anchored, may be truncated
            to the clicked level only).
        geo_cam : one camera's slice of compute_z_and_offsets result
            ({'z': {'peak': mm, 'trough': mm},
              'xy_offset': {'peak': [dx,dy], 'trough': [dx,dy]}}).
        image_size : [H, W] from detect_single_camera
        clicked_level : 'peak' or 'trough' — operator declaration for
            the DATUM pose only. Combined with the fiducial's
            `_origin_on_level` field it determines whether the datum's
            level A / level B sub-lattices are labelled peak or trough.
        datum_pose_index : 0-based index of the datum in the pose list
        pose_level_labels_by_position : list[str] — one entry per pose,
            position-aligned to `detections_per_pose`. Each entry is
            'peak' or 'trough' and means "the label assigned to level A
            of this pose". The opposite label goes to level B. The
            datum pose's entry MUST match the fiducial-derived datum
            A_label — a mismatch is a hard error (an operator would
            otherwise silently produce a half-dot-offset world frame).
            No auto-detection fallback exists: unreliable heuristics
            were the exact failure mode this parameter replaces.

        Returns
        -------
        dict with 'success' and, on success, all fields required by
        both the old cam_results[cam_num] and pinhole_results[cam_num]
        entries. On failure, {'success': False, 'error': str}.
        """
        spacing = self.board.dot_spacing_mm
        num_poses = len(detections_per_pose)

        if len(pose_level_labels_by_position) != num_poses:
            return {
                'success': False,
                'error': (
                    f"Camera {cam_num}: pose_level_labels_by_position has "
                    f"length {len(pose_level_labels_by_position)} but "
                    f"detections_per_pose has {num_poses} entries. The two "
                    f"must be position-aligned."
                ),
            }
        for i, lbl in enumerate(pose_level_labels_by_position):
            if lbl not in ('peak', 'trough'):
                return {
                    'success': False,
                    'error': (
                        f"Camera {cam_num} pose index {i}: "
                        f"pose_level_labels_by_position[{i}]={lbl!r}, "
                        f"expected 'peak' or 'trough'."
                    ),
                }

        pose_obj_views: list = []
        pose_img_views: list = []
        pose_frame_indices: list = []
        # Per-pose (A_label, B_label) tuples — surfaced so the detection
        # figure can colour each pose's dots by resolved peak/trough
        # instead of raw separate_levels A/B naming. Every entry comes
        # from `pose_level_labels_by_position` (operator-supplied); the
        # datum entry is cross-checked against the fiducial-derived
        # label below.
        pose_level_labels: list = []
        per_level_obj_datum: dict = {}

        # Datum's fiducial-derived orientation convention. Propagating
        # this to every non-datum stitch ensures back-view cameras
        # (image +x → physical -x) use a chirally-consistent world frame
        # across the datum and all non-datum views — without this,
        # cv2.calibrateCamera cannot reconcile the mirror-flipped frames
        # and the intrinsic fit blows up.
        datum_orientation = absolute_datum.get('orientation')
        datum_origin_on_level = absolute_datum.get('_origin_on_level')

        if datum_orientation is None:
            return {
                'success': False,
                'error': (
                    f"Camera {cam_num} datum pose {datum_pose_index + 1}: "
                    f"datum 'orientation' field missing. "
                    f"assign_absolute_grid_indices must populate it so "
                    f"every non-datum stitch reuses the same "
                    f"(swap_axes, col_sign, row_sign) convention — "
                    f"without this, chirality drift across poses blows "
                    f"up the multi-view intrinsic fit."
                ),
            }

        # ---- Datum pose ----
        # BOTH levels must be present — a one-level datum would prevent
        # the non-datum poses from anchoring their shared 3D world frame
        # against a known per-level xy offset.
        other_level = 'trough' if clicked_level == 'peak' else 'peak'
        missing_levels = [
            name for name in [clicked_level, other_level]
            if absolute_datum.get(name) is None
        ]
        if missing_levels:
            return {
                'success': False,
                'error': (
                    f"Camera {cam_num} datum pose {datum_pose_index + 1}: "
                    f"missing {' and '.join(missing_levels)} level(s). "
                    f"A stepped-board datum must have both peak and "
                    f"trough detected — they anchor the world frame for "
                    f"every non-datum pose. Pick a different datum "
                    f"frame, improve detection on this one, or reduce "
                    f"the board tilt."
                ),
            }

        datum_all_obj = []
        datum_all_img = []
        for level_name in [clicked_level, other_level]:
            data = absolute_datum[level_name]
            gi = data['grid_indices']
            centers = data['centers']
            z_mm = geo_cam['z'][level_name]
            xy_off = geo_cam['xy_offset'][level_name]
            obj = build_object_points(gi, z_mm, spacing, xy_off)
            datum_all_obj.append(obj)
            datum_all_img.append(np.array(centers, dtype=np.float32))
            per_level_obj_datum[level_name] = obj
        pose_obj_views.append(np.vstack(datum_all_obj).astype(np.float32))
        pose_img_views.append(np.vstack(datum_all_img).astype(np.float32))
        pose_frame_indices.append(datum_pose_index)

        # Datum's (A_label, B_label) from fiducial `_origin_on_level`.
        # If the origin click landed on level A → A has the clicked_level
        # label and B has the other. Missing `_origin_on_level` means
        # the fiducial snap did not finish: refuse to proceed rather
        # than guess with a blind ('peak', 'trough') default.
        if datum_origin_on_level == 'A':
            datum_A_label, datum_B_label = clicked_level, other_level
        elif datum_origin_on_level == 'B':
            datum_A_label, datum_B_label = other_level, clicked_level
        else:
            return {
                'success': False,
                'error': (
                    f"Camera {cam_num} datum pose {datum_pose_index + 1}: "
                    f"fiducial origin-on-level is unresolved "
                    f"(_origin_on_level={datum_origin_on_level!r}). Re-run "
                    f"fiducial snap before generating the model."
                ),
            }

        # Cross-check: the operator's per-pose label for the datum must
        # match the fiducial-derived label. Mismatch would silently
        # swap peak↔trough on level A and offset the whole world frame
        # by half a dot-spacing in xy.
        datum_A_label_from_list = pose_level_labels_by_position[datum_pose_index]
        if datum_A_label_from_list != datum_A_label:
            return {
                'success': False,
                'error': (
                    f"Camera {cam_num} datum pose {datum_pose_index + 1}: "
                    f"pose_level_labels_by_position[{datum_pose_index}]="
                    f"{datum_A_label_from_list!r} conflicts with the "
                    f"fiducial-derived datum A_label={datum_A_label!r} "
                    f"(origin click landed on level "
                    f"{datum_origin_on_level}, clicked_level="
                    f"{clicked_level!r}). Either correct the datum "
                    f"entry in cam{cam_num}_pose_levels or re-pick "
                    f"cam{cam_num}_clicked_level to match the face "
                    f"your operator actually clicked."
                ),
            }
        pose_level_labels.append((datum_A_label, datum_B_label))

        # ---- Non-datum poses ----
        # Labels come straight from `pose_level_labels_by_position`.
        # No auto-detection: operator-supplied labels are authoritative.
        for pose_idx in range(num_poses):
            if pose_idx == datum_pose_index:
                continue
            det_cam = detections_per_pose[pose_idx].get(str(cam_num))
            if det_cam is None:
                logger.warning(
                    f"Camera {cam_num} pose {pose_idx}: no detection, skipping"
                )
                continue
            lv_A_pose = det_cam.get('_level_A_full')
            lv_B_pose = det_cam.get('_level_B_full')
            if lv_A_pose is None or lv_B_pose is None:
                logger.warning(
                    f"Camera {cam_num} pose {pose_idx + 1}: only one "
                    f"level detected (A={lv_A_pose is not None}, "
                    f"B={lv_B_pose is not None}) — both levels are "
                    f"required for per-pose peak/trough labelling, "
                    f"skipping pose"
                )
                continue
            A_label = pose_level_labels_by_position[pose_idx]
            B_label = 'trough' if A_label == 'peak' else 'peak'
            logger.debug(
                f"Camera {cam_num} pose {pose_idx}: operator label "
                f"A={A_label}, B={B_label}"
            )
            # The stitch's 4-NN anchor consensus can land on a
            # diagonally-wrong integer offset for back-view cameras,
            # producing object points that don't form a valid rigid
            # body. `_build_non_datum_pose_view` applies a
            # correlation-based refinement using the real geo_cam
            # xy_offsets to pick the correct integer quadrant,
            # producing a fittable view.
            #
            # `orientation_override=datum_orientation` forces every
            # non-datum stitch to reuse the datum pose's fiducial-
            # derived (swap_axes, col_sign, row_sign) convention. Per-
            # pose re-derivation from raw BFS vectors was the root
            # cause of the +Y-axis stitch bug on strongly foreshortened
            # poses (cam1 +ry=+18° produces noisy median BFS vectors
            # whose dot-product signs can invert, flipping chirality
            # relative to the datum — cv2.calibrateCamera then sees
            # some poses in a left-handed frame and others in a right-
            # handed one, and no single K can reconcile them).
            obj_np, img_np, pose_per_level_obj, stitch_meta = self._build_non_datum_pose_view(
                lv_A_pose, lv_B_pose, geo_cam, A_label, B_label,
                orientation_override=datum_orientation,
            )
            if obj_np is None:
                continue

            pose_obj_views.append(obj_np.astype(np.float32))
            pose_img_views.append(img_np.astype(np.float32))
            pose_frame_indices.append(pose_idx)
            pose_level_labels.append((A_label, B_label))
            if stitch_meta and not stitch_meta.get('degraded_single_level', False):
                cp = stitch_meta.get('consensus_pct', float('nan'))
                if cp == cp and cp < 80.0:  # cp == cp filters NaN
                    logger.warning(
                        f"Camera {cam_num} pose {pose_idx + 1}: stitching "
                        f"consensus only {cp:.0f}% "
                        f"({stitch_meta.get('n_anchors', 0)} anchors, "
                        f"{stitch_meta.get('n_other', 0)} other-level dots) "
                        f"— pose included but treat with suspicion"
                    )

        # ---- Pinhole fit ----
        W_H = (image_size[1], image_size[0])  # (W, H)
        rms_free, K_free, dist_free, rvecs_free, tvecs_free = fit_pinhole(
            pose_obj_views, pose_img_views, W_H, fix_aspect=False)
        rms_fixed, K_fixed, dist_fixed, rvecs_fixed, tvecs_fixed = fit_pinhole(
            pose_obj_views, pose_img_views, W_H, fix_aspect=True)

        if rms_fixed <= rms_free * 1.05:
            rms, K, dist = rms_fixed, K_fixed, dist_fixed
            rvecs, tvecs = rvecs_fixed, tvecs_fixed
        else:
            rms, K, dist = rms_free, K_free, dist_free
            rvecs, tvecs = rvecs_free, tvecs_free

        # Datum pose extrinsics come from index 0 of the joint fit
        # output (datum was placed there during pose accumulation).
        rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3, 1)

        logger.info(f"Camera {cam_num}: RMS={rms:.4f}px, "
                    f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
                    f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

        return {
            'success': True,
            # cam_results fields
            'obj_points': pose_obj_views[0],  # datum for figure compat
            'img_points': pose_img_views[0],  # datum for figure compat
            'image_size': W_H,
            'per_level_obj': per_level_obj_datum,
            'obj_views_per_pose': pose_obj_views,
            'img_views_per_pose': pose_img_views,
            'pose_indices': pose_frame_indices,
            'pose_level_labels': pose_level_labels,
            # pinhole_results fields
            'rms': rms,
            'K': K,
            'dist': dist,
            'rvec': rvec,
            'tvec': tvec,
            'rvecs_all': rvecs,
            'tvecs_all': tvecs,
        }

    def generate_model(self, detections: list, fiducials: dict, params: dict,
                       datum_pose_index: int = 0,
                       progress_callback=None) -> dict:
        """Generate pinhole + stereo models from detections and fiducials.

        Args:
            detections: List of dicts, one per pose, each with the
                single-pose dict shape {str(cam): detect_single_camera result}.
                The datum pose's fiducial clicks define the world origin for
                the stereo composition; every pose contributes an independent
                view to the per-camera joint fit, breaking the fx<->tz
                depth-focal ridge at PIV magnification.
            fiducials: {str(cam1): {origin, x_axis, y_axis},
                        str(cam2): same}. Anchored to the datum pose only.
            params: {stereo_config, cam1_clicked_level, cam2_clicked_level,
                     cam1_pose_levels, cam2_pose_levels, frame_indices}

                stereo_config: 'auto' | 'same_side' | 'transmission'.
                    'auto' (default) runs cam2 twice — once under each
                    configuration — and picks the one with lower
                    reprojection RMS. The wrong config assigns cam2's Z
                    values that are off by ~board_thickness_mm, which
                    pushes the fit error to orders of magnitude larger
                    than the correct config, so the decision is robust.

                cam{1,2}_clicked_level: 'peak' | 'trough'. Required.
                    Label of the dot the operator clicked as the datum
                    fiducial ORIGIN on this camera. Combined with the
                    snapped origin's sub-lattice ('_origin_on_level')
                    this determines whether the datum pose's level A /
                    level B are peak or trough. No defaulting — missing
                    or 'auto' is a hard error.

                cam{1,2}_pose_levels: dict[int, str] — per-pose peak/
                    trough label for THIS camera, keyed by frame_idx.
                    Each value is 'peak' or 'trough' and represents the
                    label to assign to level A of that pose (level B
                    gets the opposite). Required for every pose in
                    `frame_indices`. Replaces the old dot-product auto-
                    labeller which misclassified poses on real data.

                frame_indices: list[int] — the 1-based frame number for
                    each entry in `detections`, in the same order.
                    Required. Used to map cam*_pose_levels dicts onto
                    the position-indexed `detections` list.

            datum_pose_index: 0-based index into the detections list that
                identifies the datum pose. Must lie in [0, len(detections)).
            progress_callback: optional callable({progress, stage})

        Click convention (CRITICAL for stereo accuracy):
            Stereo pose composition `R_stereo = R2 @ R1.T`,
            `T_stereo = t2 - R_stereo @ t1` is only correct when both
            cameras' fiducial-anchored world frames represent the SAME
            physical reference point — i.e. both cameras' clicked
            origin dots must lie at the same physical (x, y) on the
            board. The auto-detector ONLY catches Z misassignment
            (same_side vs transmission), NOT xy misalignment from
            wrong level clicks.

            For `same_side` (both cameras viewing the same face):
              Both cameras must click the SAME level on the SAME
              physical dot. cam1 PEAK → cam2 PEAK; cam1 TROUGH →
              cam2 TROUGH.

            For `transmission` (cameras viewing opposite faces) on a
            checkerboard board where `level_offset_mm = dot_spacing/2`
            (face B troughs sit directly behind face A peaks, sharing
            the same xy):
              Both cameras must click OPPOSITE levels.
              cam1 PEAK → cam2 TROUGH; cam1 TROUGH → cam2 PEAK.

            A misclick produces a stereo pose silently offset by half
            a dot-spacing in xy. Per-camera RMS stays clean, so the
            symptom is only visible at the stereo composition stage.
            The post-fit baseline-magnitude check below catches gross
            misclicks (baseline far from `board_thickness_mm` for
            transmission, or far from 0 for same_side) and emits a
            warning, but small misclicks may still slip through —
            documenting and following the convention is the only
            reliable defence.

        Returns: {success, cam1_rms, cam2_rms, stereo_rms,
                  stereo_config_resolved, warnings, ...}
            warnings: list of strings, possibly empty. Includes any
                baseline-magnitude sanity warnings; consumers should
                surface these to the operator.
        """
        def _progress(pct, stage):
            if progress_callback:
                progress_callback({'progress': pct, 'stage': stage})

        detections_per_pose = list(detections)
        if not detections_per_pose:
            return {'success': False, 'error': 'detections_per_pose is empty'}
        if not (0 <= datum_pose_index < len(detections_per_pose)):
            return {
                'success': False,
                'error': f'datum_pose_index {datum_pose_index} out of range '
                         f'[0, {len(detections_per_pose)})',
            }

        num_poses = len(detections_per_pose)

        cam1 = self.camera_pair[0]
        cam2 = self.camera_pair[1]
        stereo_config = params.get('stereo_config', 'auto') or 'auto'
        cam1_clicked_level = params.get('cam1_clicked_level')
        cam2_clicked_level = params.get('cam2_clicked_level')
        cam1_pose_levels = params.get('cam1_pose_levels')
        cam2_pose_levels = params.get('cam2_pose_levels')
        frame_indices = params.get('frame_indices')

        # --- Per-pose label + frame-index validation ---
        # No fallbacks: a silent auto-detect is what we removed.
        if frame_indices is None:
            return {
                'success': False,
                'error': (
                    "params['frame_indices'] is required — pass the 1-based "
                    "frame number for each entry in `detections` so pose_"
                    "levels dicts can be mapped onto position indices."
                ),
            }
        if len(frame_indices) != num_poses:
            return {
                'success': False,
                'error': (
                    f"frame_indices has length {len(frame_indices)} but "
                    f"detections has {num_poses} entries — must match."
                ),
            }
        if cam1_pose_levels is None or cam2_pose_levels is None:
            return {
                'success': False,
                'error': (
                    "params['cam1_pose_levels'] and ['cam2_pose_levels'] "
                    "are required. Each is a dict[frame_idx → 'peak'|'trough'] "
                    "with one entry per frame in `frame_indices`."
                ),
            }

        # Coerce dict keys to int (YAML / JSON round-trips sometimes leave
        # them as str) and convert to position-aligned lists.
        def _dict_keys_as_int(d, cam_label):
            try:
                return {int(k): v for k, v in d.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cam{cam_label}_pose_levels has a non-integer key: {exc}"
                )

        try:
            cam1_pose_levels_int = _dict_keys_as_int(cam1_pose_levels, 1)
            cam2_pose_levels_int = _dict_keys_as_int(cam2_pose_levels, 2)
        except ValueError as exc:
            return {'success': False, 'error': str(exc)}

        def _expand_labels(per_frame, cam_label):
            labels_by_position = []
            for f in frame_indices:
                if f not in per_frame:
                    return None, (
                        f"cam{cam_label}_pose_levels is missing frame_idx={f}. "
                        f"Every frame in frame_indices must have an explicit "
                        f"'peak' or 'trough' label — there is no auto-detect "
                        f"fallback."
                    )
                val = per_frame[f]
                if val not in ('peak', 'trough'):
                    return None, (
                        f"cam{cam_label}_pose_levels[{f}]={val!r}, "
                        f"expected 'peak' or 'trough'."
                    )
                labels_by_position.append(val)
            return labels_by_position, None

        cam1_labels_pos, err = _expand_labels(cam1_pose_levels_int, 1)
        if err:
            return {'success': False, 'error': err}
        cam2_labels_pos, err = _expand_labels(cam2_pose_levels_int, 2)
        if err:
            return {'success': False, 'error': err}

        _progress(5, 'assigning_grid_indices')

        # Datum-pose detections drive fiducial anchoring, stereo composition,
        # and the level labelling convention for every other pose. `det1` /
        # `det2` always refer to the datum pose throughout the rest of this
        # method for backward compatibility with the figure-generation code
        # that expects a single "current detection" per camera.
        datum_pose = detections_per_pose[datum_pose_index]
        det1 = datum_pose[str(cam1)]
        det2 = datum_pose[str(cam2)]

        level_A_1 = det1.get('_level_A_full')
        level_B_1 = det1.get('_level_B_full')
        level_A_2 = det2.get('_level_A_full')
        level_B_2 = det2.get('_level_B_full')

        if level_A_1 is None and level_B_1 is None:
            return {'success': False, 'error': f'No grid detected for camera {cam1}'}
        if level_A_2 is None and level_B_2 is None:
            return {'success': False, 'error': f'No grid detected for camera {cam2}'}

        # Assign absolute grid indices using fiducials
        fid1 = fiducials[str(cam1)]
        fid2 = fiducials[str(cam2)]

        # No defaulting: clicked_level MUST be an explicit 'peak' or
        # 'trough' per camera. The operator declares which face their
        # origin fiducial click landed on, and that is combined with
        # the snapped origin's level-A/B detection to decide the datum
        # pose's A_label / B_label. Silent defaults here would hide
        # exactly the bug we removed the auto-detector to fix.
        for cam_num, cl in ((cam1, cam1_clicked_level), (cam2, cam2_clicked_level)):
            if cl not in ('peak', 'trough'):
                return {
                    'success': False,
                    'error': (
                        f"Camera {cam_num}: cam{cam_num}_clicked_level="
                        f"{cl!r}, expected 'peak' or 'trough' (no default)."
                    ),
                }

        absolute1 = assign_absolute_grid_indices(
            fid1, level_A_1, level_B_1, cam1_clicked_level, self.board
        )
        absolute2 = assign_absolute_grid_indices(
            fid2, level_A_2, level_B_2, cam2_clicked_level, self.board
        )

        _progress(15, 'computing_geometry')

        # Compute geo dicts for BOTH stereo configurations up front —
        # they're cheap (just arithmetic on board constants) and we may
        # need both for auto-detection.
        geo_ss = compute_z_and_offsets('same_side',   cam1_clicked_level, cam2_clicked_level, self.board)
        geo_tx = compute_z_and_offsets('transmission', cam1_clicked_level, cam2_clicked_level, self.board)
        spacing = self.board.dot_spacing_mm

        # Datum pose image size drives the shared (W, H) the joint
        # calibrateCamera call wants. All poses from the same camera must
        # have the same image size (enforced by detect_single_camera).
        image_size_1 = det1['image_size']  # [H, W]
        image_size_2 = det2['image_size']

        _progress(30, 'fitting_cam1')

        # Cam1: stereo_config has no effect on cam1's Z values
        # (compute_z_and_offsets sets cam1_z purely from cam1_clicked_level
        # and step_height), so a single fit suffices — use geo_ss['Cam1']
        # arbitrarily. Any reference to cam1 geometry downstream uses this
        # same slice.
        result_cam1 = self._fit_per_camera_pinhole(
            cam_num=cam1,
            detections_per_pose=detections_per_pose,
            absolute_datum=absolute1,
            geo_cam=geo_ss['Cam1'],
            image_size=image_size_1,
            clicked_level=cam1_clicked_level,
            datum_pose_index=datum_pose_index,
            pose_level_labels_by_position=cam1_labels_pos,
        )
        if not result_cam1['success']:
            return result_cam1
        logger.info(
            f"Camera {cam1}: {len(result_cam1['obj_views_per_pose'])} pose views "
            f"(datum at index 0, {len(result_cam1['obj_views_per_pose'])-1} non-datum)"
        )
        _progress(45, 'fitting_cam2')

        # Cam2: stereo_config DOES affect cam2's Z values. In 'auto' mode
        # we fit both configurations and pick the one with lower RMS.
        # The wrong config offsets cam2's Z by ~board_thickness_mm, which
        # pushes reprojection error into an impossible range — the gap
        # between the two fits is large and the decision is unambiguous.
        resolved_config: str
        result_cam2: dict
        rms_ss: Optional[float] = None
        rms_tx: Optional[float] = None

        if stereo_config == 'auto':
            r_ss = self._fit_per_camera_pinhole(
                cam_num=cam2,
                detections_per_pose=detections_per_pose,
                absolute_datum=absolute2,
                geo_cam=geo_ss['Cam2'],
                image_size=image_size_2,
                clicked_level=cam2_clicked_level,
                datum_pose_index=datum_pose_index,
                pose_level_labels_by_position=cam2_labels_pos,
            )
            r_tx = self._fit_per_camera_pinhole(
                cam_num=cam2,
                detections_per_pose=detections_per_pose,
                absolute_datum=absolute2,
                geo_cam=geo_tx['Cam2'],
                image_size=image_size_2,
                clicked_level=cam2_clicked_level,
                datum_pose_index=datum_pose_index,
                pose_level_labels_by_position=cam2_labels_pos,
            )
            rms_ss = r_ss.get('rms') if r_ss.get('success') else None
            rms_tx = r_tx.get('rms') if r_tx.get('success') else None

            if rms_ss is not None and (rms_tx is None or rms_ss <= rms_tx):
                resolved_config = 'same_side'
                result_cam2 = r_ss
            elif rms_tx is not None:
                resolved_config = 'transmission'
                result_cam2 = r_tx
            else:
                # Both fits failed — return the same_side error so the
                # operator gets a diagnostic rather than a silent skip.
                return r_ss if not r_ss.get('success') else r_tx

            logger.info(
                f"Camera {cam2}: auto stereo_config = '{resolved_config}' "
                f"(same_side RMS={rms_ss}, transmission RMS={rms_tx})"
            )
        else:
            if stereo_config not in ('same_side', 'transmission'):
                return {
                    'success': False,
                    'error': (
                        f"stereo_config must be 'auto', 'same_side', or "
                        f"'transmission', got {stereo_config!r}"
                    ),
                }
            resolved_config = stereo_config
            geo_for_cam2 = geo_ss if stereo_config == 'same_side' else geo_tx
            result_cam2 = self._fit_per_camera_pinhole(
                cam_num=cam2,
                detections_per_pose=detections_per_pose,
                absolute_datum=absolute2,
                geo_cam=geo_for_cam2['Cam2'],
                image_size=image_size_2,
                clicked_level=cam2_clicked_level,
                datum_pose_index=datum_pose_index,
                pose_level_labels_by_position=cam2_labels_pos,
            )
            if not result_cam2['success']:
                return result_cam2

        logger.info(
            f"Camera {cam2}: {len(result_cam2['obj_views_per_pose'])} pose views "
            f"(datum at index 0, {len(result_cam2['obj_views_per_pose'])-1} non-datum)"
        )

        # The resolved geo is the one that actually drove the cam2 fit —
        # used downstream for figure generation + save.
        geo = geo_ss if resolved_config == 'same_side' else geo_tx
        stereo_config = resolved_config  # overwrite for downstream references

        # Populate cam_results / pinhole_results dicts in the same shape
        # the rest of generate_model expects (figures, stereo composition,
        # save).
        cam_results: dict = {}
        pinhole_results: dict = {}
        for cam_num, result in ((cam1, result_cam1), (cam2, result_cam2)):
            cam_results[cam_num] = {
                'obj_points': result['obj_points'],
                'img_points': result['img_points'],
                'image_size': result['image_size'],
                'per_level_obj': result['per_level_obj'],
                'obj_views_per_pose': result['obj_views_per_pose'],
                'img_views_per_pose': result['img_views_per_pose'],
                'pose_indices': result['pose_indices'],
                'pose_level_labels': result['pose_level_labels'],
            }
            pinhole_results[cam_num] = {
                'rms': result['rms'],
                'K': result['K'],
                'dist': result['dist'],
                'rvec': result['rvec'],
                'tvec': result['tvec'],
                'rvecs_all': result['rvecs_all'],
                'tvecs_all': result['tvecs_all'],
                'pose_indices': result['pose_indices'],
                'image_size': result['image_size'],
                'obj_points': result['obj_points'],
            }
        _progress(65, 'fitting_cam2')

        _progress(75, 'stereo_calibration')

        # Derive stereo pose from individual camera extrinsics.
        # In a transmission setup, cam1 and cam2 see different faces of the board
        # at different Z planes — there are NO common 3D points. stereoCalibrate
        # requires common points and would fail. Instead, derive the relative pose
        # from the individual solvePnP results (matches prototype approach).
        pr1 = pinhole_results[cam1]
        pr2 = pinhole_results[cam2]

        R1, _ = cv2.Rodrigues(pr1['rvec'])
        R2, _ = cv2.Rodrigues(pr2['rvec'])
        t1 = pr1['tvec'].reshape(3, 1)
        t2 = pr2['tvec'].reshape(3, 1)

        R_stereo = R2 @ R1.T
        T_stereo = t2 - R_stereo @ t1

        baseline_mm = float(np.linalg.norm(T_stereo))
        # Physical stereo angle: the angle between the two cameras' optical
        # axes in the world frame. R1.T @ [0,0,1] is cam1's optical axis.
        # This is the quantity physically meaningful for stereo PIV — it is
        # NOT the same as the rotation angle of R_stereo, which conflates
        # the optical-axis change with any roll difference between the
        # cameras' coordinate frames (e.g. cameras placed above and below
        # a laser sheet are perpendicular to each other but their frames
        # are rolled ~180° relative to each other).
        axis1 = R1.T @ np.array([0.0, 0.0, 1.0])
        axis2 = R2.T @ np.array([0.0, 0.0, 1.0])
        optical_angle_rad = np.arccos(np.clip(float(np.dot(axis1, axis2)), -1.0, 1.0))
        relative_angle_deg = float(np.degrees(optical_angle_rad))

        logger.info(f"Stereo (derived): optical-axis angle={relative_angle_deg:.1f}°, "
                    f"baseline={baseline_mm:.1f}mm")

        # No baseline-magnitude sanity check. An earlier version warned
        # when |T_stereo| drifted from `board_thickness_mm` in
        # transmission mode, but that expectation is wrong: |T_stereo|
        # is the physical camera-to-camera distance in the world frame,
        # which depends on the real rig geometry (e.g. ~7000 mm for a
        # ±45° angular stereo setup at d0=5000 mm), NOT on board
        # thickness. The wrong-fiducial failure mode the warning was
        # meant to catch surfaces much more reliably as elevated
        # reprojection RMS — that's the right health metric.
        warnings: list = []

        _progress(90, 'saving_models')

        # Save models
        cam1_path = self._save_per_camera_model(cam1, pr1)
        cam2_path = self._save_per_camera_model(cam2, pr2)
        stereo_path = self._save_stereo_model(
            cam1, cam2, pr1, pr2, R_stereo, T_stereo, relative_angle_deg,
        )

        # Clear any stale self-calibration from a previous stereo model
        sc_path = (self.base_dir / "calibration"
                   / f"stereo_cam{cam1}_cam{cam2}" / "self_calibration.yaml")
        if sc_path.exists():
            sc_path.unlink()
            logger.info("Cleared stale self-calibration file")

        # Generate diagnostic figures
        try:
            from pivtools_gui.calibration.calibration_figures import (
                make_camera_placement_html,
                make_stepped_detection_figure,
                make_stepped_reprojection_figure,
                make_dewarp_overlay_figure,
            )
            stereo_fig_dir = self.base_dir / "calibration" / f"stereo_cam{cam1}_cam{cam2}" / "figures"
            stereo_fig_dir.mkdir(parents=True, exist_ok=True)

            # Per-camera detection figures — one PNG per pose per camera.
            # Single-pose runs produce detection.png (unchanged path).
            # Multi-pose runs produce detection_pose_NN.png for every pose
            # plus detection.png for the datum (for backward compat with
            # any consumer that reads the generic file name).
            for cam_num, det_datum, abs_datum_cam, cam_clicked, cam_fids in [
                (cam1, det1, absolute1, cam1_clicked_level, fid1),
                (cam2, det2, absolute2, cam2_clicked_level, fid2),
            ]:
                cam_fig_dir = self.base_dir / "calibration" / f"Cam{cam_num}" / "stepped_board" / "figures"
                cam_fig_dir.mkdir(parents=True, exist_ok=True)

                # Resolve peak/trough → (A or B) for the datum pose
                # from the fiducial origin-level decision stashed by
                # assign_absolute_grid_indices. This lets the figure
                # colour dots by physical meaning instead of the
                # arbitrary Level A / Level B row-parity naming.
                datum_origin_on = abs_datum_cam.get('_origin_on_level')
                if datum_origin_on == 'A':
                    datum_peak_level = 'A' if cam_clicked == 'peak' else 'B'
                elif datum_origin_on == 'B':
                    datum_peak_level = 'B' if cam_clicked == 'peak' else 'A'
                else:
                    datum_peak_level = None
                datum_trough_level = (
                    'B' if datum_peak_level == 'A'
                    else 'A' if datum_peak_level == 'B'
                    else None
                )

                # Per-pose (A_label, B_label) tuples from the joint fit
                # (one per pose that made it in, including the datum).
                pose_labels_for_cam = cam_results[cam_num].get(
                    'pose_level_labels', [],
                )
                pose_indices_for_cam = cam_results[cam_num].get(
                    'pose_indices', [],
                )
                labels_by_pose_idx = {
                    int(idx): tup
                    for idx, tup in zip(pose_indices_for_cam, pose_labels_for_cam)
                }

                # Datum pose — always written to detection.png for
                # single-pose backward compatibility. Fiducials are
                # overlaid on this figure so the operator can see the
                # board origin/axis convention on the datum directly.
                if det_datum.get('_gray_uint8') is not None:
                    make_stepped_detection_figure(
                        det_datum.get('_gray_uint8'),
                        det_datum.get('_level_A_full'),
                        det_datum.get('_level_B_full'),
                        cam_fig_dir / "detection.png",
                        title=f"Cam {cam_num} (datum, pose {datum_pose_index})",
                        blob_info=det_datum.get('_blob_info'),
                        peak_level=datum_peak_level,
                        trough_level=datum_trough_level,
                        fiducials=cam_fids,
                    )

                # Every pose — one PNG per pose. Peak/trough labels come
                # from the auto-labeller's per-pose decision surfaced in
                # `pose_level_labels`. Fiducials are not drawn on
                # non-datum poses.
                if num_poses > 1:
                    for pose_idx, pose_dets in enumerate(detections_per_pose):
                        det_p = pose_dets.get(str(cam_num))
                        if det_p is None:
                            continue
                        gray_p = det_p.get('_gray_uint8')
                        if gray_p is None:
                            continue
                        tag = 'datum' if pose_idx == datum_pose_index else 'view'
                        pose_peak_level = None
                        pose_trough_level = None
                        if pose_idx in labels_by_pose_idx:
                            A_lbl, B_lbl = labels_by_pose_idx[pose_idx]
                            pose_peak_level = 'A' if A_lbl == 'peak' else 'B'
                            pose_trough_level = 'B' if pose_peak_level == 'A' else 'A'
                        pose_fiducials = cam_fids if pose_idx == datum_pose_index else None
                        make_stepped_detection_figure(
                            gray_p,
                            det_p.get('_level_A_full'),
                            det_p.get('_level_B_full'),
                            cam_fig_dir / f"detection_pose_{pose_idx:02d}.png",
                            title=f"Cam {cam_num} pose {pose_idx} ({tag})",
                            blob_info=det_p.get('_blob_info'),
                            peak_level=pose_peak_level,
                            trough_level=pose_trough_level,
                            fiducials=pose_fiducials,
                        )

            # Reprojection error scatter — per-camera, colored by pose
            # in multi-view mode (outliers visible at a glance) or by
            # Z-plane in single-pose mode (original behaviour).
            reproj_data = {}
            for cam_num, pr, cr in [
                (cam1, pr1, cam_results[cam1]),
                (cam2, pr2, cam_results[cam2]),
            ]:
                reproj_data[cam_num] = {
                    'K': pr['K'], 'dist': pr['dist'],
                    'rvec': pr['rvec'], 'tvec': pr['tvec'],
                    'rms': pr['rms'],
                    'obj_points': cr['obj_points'],
                    'img_points': cr['img_points'],
                    # Multi-pose fields (ignored by the figure helper
                    # if the pose list has length 1)
                    'obj_views_per_pose': cr.get('obj_views_per_pose'),
                    'img_views_per_pose': cr.get('img_views_per_pose'),
                    'rvecs_all': pr.get('rvecs_all'),
                    'tvecs_all': pr.get('tvecs_all'),
                    'pose_indices': pr.get('pose_indices'),
                }
            make_stepped_reprojection_figure(
                reproj_data,
                stereo_fig_dir / "reprojection_errors.png",
            )

            # Camera placement figure — 3D scene + orthogonal projections.
            # `obj_points` is the datum-pose 3D point cloud in THIS
            # camera's fiducial-anchored world frame. Cam1 and cam2
            # share the same xy layout by construction (via
            # compute_z_and_offsets), so either camera's point cloud is
            # a valid "world board" to visualise; we use cam1's.
            cam_placement_data = [
                {
                    'label': f'Cam {cam1}',
                    'rvec': pr1['rvec'],
                    'tvec': pr1['tvec'],
                    'color': '#d62728',  # red
                    'rvecs_all': pr1.get('rvecs_all'),
                    'tvecs_all': pr1.get('tvecs_all'),
                    'obj_points': cam_results[cam1].get('obj_points'),
                },
                {
                    'label': f'Cam {cam2}',
                    'rvec': pr2['rvec'],
                    'tvec': pr2['tvec'],
                    'color': '#1f77b4',  # blue
                    'rvecs_all': pr2.get('rvecs_all'),
                    'tvecs_all': pr2.get('tvecs_all'),
                    'obj_points': cam_results[cam2].get('obj_points'),
                },
            ]
            make_camera_placement_html(
                cam_placement_data,
                stereo_fig_dir / "camera_placement.html",
                title=f"Camera Placement — Cam{cam1} ↔ Cam{cam2} ({stereo_config})",
            )

            # Multi-pair dewarp overlays (red-cyan stereo verification)
            # Pair A: Cam1 peak Z + Cam2 trough Z
            # Pair B: Cam1 trough Z + Cam2 peak Z
            gray1 = det1.get('_gray_uint8')
            gray2 = det2.get('_gray_uint8')
            if gray1 is not None and gray2 is not None:
                cr1 = cam_results[cam1]
                cr2 = cam_results[cam2]
                geo1 = geo['Cam1']
                geo2 = geo['Cam2']
                level_offset = self.board.level_offset_mm

                overlay_pairs = []
                # Pair A: cam1 peak + cam2 trough
                if 'peak' in geo1['z'] and 'trough' in geo2['z']:
                    overlay_pairs.append((
                        "Pair A: Cam1 peak + Cam2 trough",
                        geo1['z']['peak'], geo2['z']['trough'],
                        "overlay_pair_A.png",
                        cr1['per_level_obj'].get('peak'),
                        cr2['per_level_obj'].get('trough'),
                        0.0,
                    ))
                # Pair B: cam1 trough + cam2 peak
                if 'trough' in geo1['z'] and 'peak' in geo2['z']:
                    overlay_pairs.append((
                        "Pair B: Cam1 trough + Cam2 peak",
                        geo1['z']['trough'], geo2['z']['peak'],
                        "overlay_pair_B.png",
                        cr1['per_level_obj'].get('trough'),
                        cr2['per_level_obj'].get('peak'),
                        level_offset,
                    ))

                for label, z1_mm, z2_mm, filename, c1_lvl_obj, c2_lvl_obj, grid_off in overlay_pairs:
                    make_dewarp_overlay_figure(
                        gray1, gray2, pr1, pr2,
                        cr1['obj_points'], cr2['obj_points'],
                        stereo_fig_dir / filename,
                        title=f"{label}: Cam{cam1} (red) Z={z1_mm:.1f}mm, Cam{cam2} (cyan) Z={z2_mm:.1f}mm",
                        z1=z1_mm, z2=z2_mm,
                        cam1_level_obj=c1_lvl_obj,
                        cam2_level_obj=c2_lvl_obj,
                        dot_spacing_mm=self.board.dot_spacing_mm,
                        grid_offset=grid_off,
                    )

            logger.info("Diagnostic figures saved")
        except Exception as e:
            logger.debug(f"Diagnostic figures skipped: {e}")

        # Save config snapshot
        if self._config is not None:
            try:
                snapshot_path = self._config.save_calibration_snapshot(self.base_dir)
                logger.debug(f"Calibration snapshot saved: {snapshot_path}")
            except Exception as e:
                logger.warning(f"Failed to save calibration snapshot: {e}")

        _progress(100, 'complete')

        return {
            'success': True,
            'cam1_rms': float(pr1['rms']),
            'cam2_rms': float(pr2['rms']),
            'stereo_rms': None,
            'warnings': warnings,
            'relative_angle_deg': relative_angle_deg,
            'baseline_mm': baseline_mm,
            'cam1_model_path': str(cam1_path),
            'cam2_model_path': str(cam2_path),
            'stereo_model_path': str(stereo_path),
            'cam1_clicked_level_resolved': cam1_clicked_level,
            'cam2_clicked_level_resolved': cam2_clicked_level,
            'stereo_config_resolved': resolved_config,
            'stereo_config_rms_same_side': (float(rms_ss) if rms_ss is not None else None),
            'stereo_config_rms_transmission': (float(rms_tx) if rms_tx is not None else None),
            'num_poses_total': num_poses,
            'cam1_poses_used': pr1.get('pose_indices', [datum_pose_index]),
            'cam2_poses_used': pr2.get('pose_indices', [datum_pose_index]),
            'cam1_details': {
                'focal_length': [float(pr1['K'][0, 0]), float(pr1['K'][1, 1])],
                'principal_point': [float(pr1['K'][0, 2]), float(pr1['K'][1, 2])],
                'distortion': pr1['dist'].flatten().tolist(),
            },
            'cam2_details': {
                'focal_length': [float(pr2['K'][0, 0]), float(pr2['K'][1, 1])],
                'principal_point': [float(pr2['K'][0, 2]), float(pr2['K'][1, 2])],
                'distortion': pr2['dist'].flatten().tolist(),
            },
        }

    def _save_per_camera_model(self, cam_num: int, pr: dict) -> Path:
        """Save in VectorCalibrator-compatible .mat schema."""
        model_dir = self.base_dir / "calibration" / f"Cam{cam_num}" / self._subdir_name / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "camera_model.mat"

        W_img, H_img = pr['image_size']

        # Multi-view: save rvecs/tvecs as (N, 3) with datum at index 0.
        # Single-pose (N=1) collapses to (3,) on load via squeeze_me=True,
        # which all downstream VectorCalibrator/global_coordinate_alignment/
        # camera_model_utils consumers handle correctly through their
        # existing `rvecs.ndim == 1` / `rvecs[0].flatten()` branching.
        rvecs_all = pr.get('rvecs_all') or [pr['rvec']]
        tvecs_all = pr.get('tvecs_all') or [pr['tvec']]
        rvecs_stacked = np.stack(
            [np.asarray(rv, dtype=np.float64).reshape(3) for rv in rvecs_all],
            axis=0,
        )
        tvecs_stacked = np.stack(
            [np.asarray(tv, dtype=np.float64).reshape(3) for tv in tvecs_all],
            axis=0,
        )
        pose_indices = np.asarray(
            pr.get('pose_indices', [0]), dtype=np.int32,
        )

        save_dict = {
            'camera_matrix': pr['K'],
            'dist_coeffs': pr['dist'],
            'rvecs': rvecs_stacked,
            'tvecs': tvecs_stacked,
            'rms_error': float(pr['rms']),
            'image_width': W_img,
            'image_height': H_img,
            'image_size': np.array([W_img, H_img]),
            'dot_spacing_mm': self.dot_spacing_mm,
            'datum_frame': self.datum_frame,
            'num_poses': len(rvecs_all),
            'pose_frame_indices': pose_indices,  # 0-based pose indices in
                                                  # the sequence order used by
                                                  # generate_model
            'dt': self.dt,
            'object_points': np.asarray(pr['obj_points'], dtype=np.float64),
        }

        scipy.io.savemat(str(model_path), save_dict)
        logger.info(f"Saved camera {cam_num} model: {model_path}")
        return model_path

    def _save_stereo_model(self, cam1_num: int, cam2_num: int,
                            pr1: dict, pr2: dict,
                            R: np.ndarray, T: np.ndarray,
                            relative_angle_deg: float) -> Path:
        """Save stereo model in standard schema.

        R, T are the relative stereo pose derived from individual extrinsics:
            R_stereo = R2 @ R1.T
            T_stereo = t2 - R_stereo @ t1
        """
        model_dir = (self.base_dir / "calibration"
                     / f"stereo_cam{cam1_num}_cam{cam2_num}" / "model")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "stereo_model.mat"

        H1, W1 = pr1['image_size'][1], pr1['image_size'][0]

        # Cam1 extrinsics from solvePnP
        R1, _ = cv2.Rodrigues(pr1['rvec'])
        t1 = pr1['tvec'].reshape(3, 1)

        # Cam2 absolute extrinsics: R2_abs = R_stereo @ R1, t2_abs = R_stereo @ t1 + T_stereo
        R2 = R @ R1
        t2 = R @ t1 + T.reshape(3, 1)
        rvec2, _ = cv2.Rodrigues(R2)

        # Stereo rectification — needed by stereo reconstruction for
        # triangulation. stereoRectify only requires R, T, K, and dist
        # (not common 3D points), so it works with the derived pose.
        rect_R1, rect_R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            pr1['K'], pr1['dist'], pr2['K'], pr2['dist'],
            (W1, H1), R, T.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1,
        )

        save_dict = {
            'camera_matrix_1': pr1['K'],
            'dist_coeffs_1': pr1['dist'],
            'camera_matrix_2': pr2['K'],
            'dist_coeffs_2': pr2['dist'],
            'rotation_matrix': R,
            'translation_vector': T,
            'rectification_R1': rect_R1,
            'rectification_R2': rect_R2,
            'projection_P1': P1,
            'projection_P2': P2,
            'image_size': np.array([W1, H1]),
            'cam1_rms_error': float(pr1['rms']),
            'cam2_rms_error': float(pr2['rms']),
            'relative_angle_deg': relative_angle_deg,
            'num_image_pairs': 1,
            'dot_spacing_mm': self.dot_spacing_mm,
            'step_height_mm': self.step_height_mm,
            'board_thickness_mm': self.board_thickness_mm,
            'rvecs_1': pr1['rvec'].reshape(3, 1),
            'tvecs_1': t1,
            'rvecs_2': rvec2.reshape(3, 1),
            'tvecs_2': t2,
            'datum_frame': self.datum_frame,
            'object_points_1': np.asarray(pr1['obj_points'], dtype=np.float64),
            'object_points_2': np.asarray(pr2['obj_points'], dtype=np.float64),
        }

        scipy.io.savemat(str(model_path), save_dict)
        logger.info(f"Saved stereo model: {model_path}")
        return model_path
