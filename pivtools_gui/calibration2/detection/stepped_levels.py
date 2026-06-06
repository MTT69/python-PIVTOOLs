"""calibration2.detection.stepped_levels — two-Z level separation + stitch geometry.

The stepped (dual-level) dotboard carries dots on two parallel Z-planes (peak /
trough) separated by a known machined ``step_height_mm``. The trough grid is
interleaved by half a dot spacing in both x and y. A single image therefore shows
two interleaved grids that must be separated before a per-level grid walk can run,
then stitched back into one consistent grid frame.

This module is the canonical home for that geometry in calibration2. It is a
faithful port of the (now-orphaned) v1
``calibration_stepped.stepped_calibration_production`` detection helpers — the
math is preserved exactly (no silent algorithm changes). Only the blob/BFS/rescue
primitives are imported from the still-shared ``grid_detection`` module, exactly as
``detection.dotboard`` already does.

Fiducial-coupled absolute-index assignment is deliberately NOT ported here: in
calibration2 the world frame (origin / +X / +Y) is resolved later by
``world_frame``/the stepped calibrator from the user's clicks, not baked into
detection. The detector emits levels in a pose-local stitched frame; the calibrator
applies the clicked frame + peak/trough labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from pivtools_gui.calibration.grid_detection import (
    _bfs_grid_walk_dict,
    _filter_connected_dict,
    _find_grid_directions,
    _rescue_missing_dots,
    find_largest_grid_component,
)


@dataclass
class SteppedBoardSpec:
    """Machined geometry of a dual-level stepped dotboard."""

    dot_spacing_mm: float = 15.0
    step_height_mm: float = 3.0
    level_offset_mm: Optional[float] = None
    board_thickness_mm: float = 14.8

    def __post_init__(self) -> None:
        if self.level_offset_mm is None:
            self.level_offset_mm = self.dot_spacing_mm / 2.0


# ---------------------------------------------------------------------------
# Per-level grid detection (k=5 after level separation)
# ---------------------------------------------------------------------------

def find_grid_vectors(
    centers: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Find the two dominant grid direction vectors for ONE separated level.

    Delegates to the shared ``_find_grid_directions`` with k=5 (critical for
    stepped boards after level separation — k>5 pulls in diagonals to the removed
    level's row positions). Returns (vec1, vec2, spacing_px) or None.
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
    """Full single-level pipeline: direction finding (k=5) -> reciprocal BFS ->
    RANSAC -> template rescue -> connected component filter.

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
        validated, rescued_centers, _rescued_nodes = _rescue_missing_dots(
            validated, centers, flat_field, spacing_px,
        )
        all_centers = rescued_centers
    else:
        all_centers = list(centers)

    # Prune orphaned points
    validated = _filter_connected_dict(validated)

    # Convert to ndarray format (keep raw BFS indices — no zero-basing; the stitch
    # re-anchors levels relative to each other, so zero-basing is redundant and
    # would invalidate H).
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


# ---------------------------------------------------------------------------
# Row clustering + level separation (alternating-row parity)
# ---------------------------------------------------------------------------

def cluster_into_rows(
    centers: np.ndarray, spacing_px: float
) -> Tuple[np.ndarray, List[float]]:
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

            logger.debug(
                f"Splitting row {r} ({count} dots, max internal gap={max_gap:.1f}px)"
            )

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


def separate_levels(
    centers: np.ndarray, row_labels: np.ndarray, row_y_values: List[float]
) -> dict:
    """Separate dots into two levels using alternating row parity.

    Even rows -> Level A, odd rows -> Level B (whichever parity has more dots gets
    Level A). Peak/trough assignment is determined later by the user's clicked
    level, not by auto-detection.

    Returns dict with 'centers', 'row_labels', 'mask_level_A', 'mask_level_B',
    'n_rows'.
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


# ---------------------------------------------------------------------------
# Cross-level stitch into one pose-local frame
# ---------------------------------------------------------------------------

def stitch_levels_pose_local(
    level_A_data: Optional[dict],
    level_B_data: Optional[dict],
    board: SteppedBoardSpec,
    orientation_override: Optional[dict] = None,
) -> Optional[dict]:
    """Stitch level A and level B into a single pose-local shared frame, using only
    nearest-neighbour geometry (no fiducial clicks needed).

    Each "other level" dot is located relative to its 4 nearest "reference level"
    neighbours via the known +/- half-spacing physical offset, using the BFS walk's
    own indices as the reference frame. The output frame is pose-local — its
    absolute position is arbitrary — which is fine because ``cv2.calibrateCamera``
    treats each view's object points as an independent world frame for intrinsic
    fitting; only the datum pose needs a globally meaningful frame (resolved later
    from the user's clicks).

    Parameters
    ----------
    level_A_data, level_B_data : dict or None
        Output of ``run_single_level_detection``.
    board : SteppedBoardSpec
        Board geometry (needs dot_spacing_mm and level_offset_mm).
    orientation_override : dict or None
        Optional ``{'swap_axes', 'col_sign', 'row_sign'}`` from the datum pose's
        resolved frame. When provided, the stitch uses these sign constants instead
        of deriving them from raw BFS ``vec1``/``vec2`` dot products. REQUIRED for a
        camera viewing the board from behind (image +x -> physical -x), where the
        raw BFS direction disagrees with the fiducial-anchored datum and a chirality
        mismatch would blow up the multi-view intrinsic fit. When None, the
        BFS-derived logic runs — safe for front-view cameras.

    Returns
    -------
    dict with keys 'reference' and optionally 'other', each containing 'centers',
    'grid_indices', and a 'source_level' flag ('A' or 'B'), plus a 'metadata'
    sub-dict: n_reference, n_other, n_anchors, consensus_pct, swap_axes, col_sign,
    row_sign, degraded_single_level. Returns None if no usable level was detected.
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

    # Decide the axis orientation for the other level's BFS grid. Trust the
    # orientation_override (from the datum pose's resolved frame) verbatim when
    # given; otherwise derive from raw BFS vec1/vec2 dot products.
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
            # other.vec1 -> ref.vec1 (maybe flipped)
            swap_axes = False
            col_sign = 1 if d11 > 0 else -1
            row_sign = 1 if d22 > 0 else -1
        else:
            # other.vec1 -> ref.vec2 (swapped)
            swap_axes = True
            col_sign = 1 if d12 > 0 else -1
            row_sign = 1 if d21 > 0 else -1

    # Apply the same orientation transform to BOTH levels so anchor offsets are
    # computed in a single consistent frame.
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

    # Cross-level anchor: each other-level dot is at +half-spacing from the mean of
    # its 4 nearest reference-level neighbours.
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
        # Skip dots whose 4 nearest neighbours aren't a unit-square cell.
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
