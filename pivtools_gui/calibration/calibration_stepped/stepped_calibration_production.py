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

    return result


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
      1. Use a single Z-plane (z=0) from the first view to get initial K via Zhang's method
      2. Refine with all views using true 3D coordinates

    Returns (rms, K, dist, rvecs, tvecs).
    """
    W, H = image_size

    # Step 1: Get initial K from first view's Z=0 plane
    first_obj = obj_views[0]
    first_img = img_views[0]
    unique_z = np.unique(np.round(first_obj[:, 2], 2))
    z_for_init = unique_z[np.argmin(np.abs(unique_z))]
    init_mask = np.abs(first_obj[:, 2] - z_for_init) < 0.5
    init_obj_z0 = np.column_stack([first_obj[init_mask, 0], first_obj[init_mask, 1],
                                    np.zeros(init_mask.sum())]).astype(np.float32)
    init_img = first_img[init_mask].reshape(-1, 1, 2).astype(np.float32)

    init_flags = cv2.CALIB_FIX_K3
    if fix_aspect:
        init_flags |= cv2.CALIB_FIX_ASPECT_RATIO

    _, K_init, dist_init, _, _ = cv2.calibrateCamera(
        [init_obj_z0], [init_img], (W, H), None, None, flags=init_flags)

    # Step 2: Refine with all views (split by Z-plane for OpenCV)
    obj_list = []
    img_list = []
    for obj, img in zip(obj_views, img_views):
        z_vals = obj[:, 2]
        for z in np.sort(np.unique(np.round(z_vals, 2))):
            mask = np.abs(z_vals - z) < 0.5
            obj_list.append(obj[mask].astype(np.float32))
            img_list.append(img[mask].reshape(-1, 1, 2).astype(np.float32))

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

    def generate_model(self, detections: dict, fiducials: dict, params: dict,
                       progress_callback=None) -> dict:
        """Generate pinhole + stereo models from detections and fiducials.

        Args:
            detections: {cam1: detect_single_camera result, cam2: same}
            fiducials: {cam1: {origin:[x,y], x_axis:[x,y], y_axis:[x,y]},
                        cam2: same}
            params: {stereo_config, cam1_clicked_level, cam2_clicked_level}
            progress_callback: optional callable({progress, stage})

        Returns: {success, cam1_rms, cam2_rms, stereo_rms, ...}
        """
        def _progress(pct, stage):
            if progress_callback:
                progress_callback({'progress': pct, 'stage': stage})

        cam1 = self.camera_pair[0]
        cam2 = self.camera_pair[1]
        stereo_config = params['stereo_config']
        cam1_clicked_level = params['cam1_clicked_level']
        cam2_clicked_level = params['cam2_clicked_level']

        _progress(5, 'assigning_grid_indices')

        # Get full detection data (with internal numpy arrays)
        det1 = detections[str(cam1)]
        det2 = detections[str(cam2)]

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

        absolute1 = assign_absolute_grid_indices(
            fid1, level_A_1, level_B_1, cam1_clicked_level, self.board
        )
        absolute2 = assign_absolute_grid_indices(
            fid2, level_A_2, level_B_2, cam2_clicked_level, self.board
        )

        _progress(15, 'computing_geometry')

        # Compute Z and offsets
        geo = compute_z_and_offsets(stereo_config, cam1_clicked_level, cam2_clicked_level, self.board)
        spacing = self.board.dot_spacing_mm

        # Build 3D-2D correspondences per camera
        image_size_1 = det1['image_size']  # [H, W]
        image_size_2 = det2['image_size']

        cam_results = {}
        for cam_num, absolute, geo_cam, img_size, clicked_level in [
            (cam1, absolute1, geo['Cam1'], image_size_1, cam1_clicked_level),
            (cam2, absolute2, geo['Cam2'], image_size_2, cam2_clicked_level),
        ]:
            all_obj = []
            all_img = []
            per_level_obj = {}  # {level_name: obj_points} for multi-pair overlay
            other_level = "trough" if clicked_level == "peak" else "peak"

            for level_name in [clicked_level, other_level]:
                data = absolute.get(level_name)
                if data is None:
                    continue
                gi = data['grid_indices']
                centers = data['centers']
                z_mm = geo_cam['z'][level_name]
                xy_off = geo_cam['xy_offset'][level_name]
                obj = build_object_points(gi, z_mm, spacing, xy_off)
                all_obj.append(obj)
                all_img.append(np.array(centers, dtype=np.float32))
                per_level_obj[level_name] = obj

            if not all_obj:
                return {'success': False, 'error': f'No correspondences for camera {cam_num}'}

            cam_results[cam_num] = {
                'obj_points': np.vstack(all_obj),
                'img_points': np.vstack(all_img),
                'image_size': (img_size[1], img_size[0]),  # (W, H)
                'per_level_obj': per_level_obj,
            }

        _progress(30, 'fitting_cam1')

        # Fit pinhole per camera (try both free-aspect and fixed-aspect, pick best)
        pinhole_results = {}
        for cam_idx, cam_num in enumerate([cam1, cam2]):
            cr = cam_results[cam_num]
            obj_views = [cr['obj_points']]
            img_views = [cr['img_points']]
            W_H = cr['image_size']

            rms_free, K_free, dist_free, rvecs_free, tvecs_free = fit_pinhole(
                obj_views, img_views, W_H, fix_aspect=False)
            rms_fixed, K_fixed, dist_fixed, rvecs_fixed, tvecs_fixed = fit_pinhole(
                obj_views, img_views, W_H, fix_aspect=True)

            if rms_fixed <= rms_free * 1.05:
                # Prefer fixed-aspect (fx=fy) if within 5% of free
                rms, K, dist, rvecs, tvecs = rms_fixed, K_fixed, dist_fixed, rvecs_fixed, tvecs_fixed
            else:
                rms, K, dist, rvecs, tvecs = rms_free, K_free, dist_free, rvecs_free, tvecs_free

            # Single consistent pose via solvePnP using all points + fitted K
            # Use calibrateCamera's first rvec/tvec as initial guess for convergence
            all_obj = cr['obj_points'].astype(np.float64)
            all_img = cr['img_points'].reshape(-1, 1, 2).astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(
                all_obj, all_img, K, dist,
                rvec=rvecs[0], tvec=tvecs[0], useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return {'success': False, 'error': f'solvePnP failed for camera {cam_num}'}

            pinhole_results[cam_num] = {
                'rms': rms,
                'K': K,
                'dist': dist,
                'rvec': rvec,
                'tvec': tvec,
                'image_size': W_H,
            }

            logger.info(f"Camera {cam_num}: RMS={rms:.4f}px, "
                        f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
                        f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

            _progress(30 + 20 * (cam_idx + 1), f'fitting_cam{cam_idx + 1}')

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
        angle_rad = np.arccos(np.clip((np.trace(R_stereo) - 1) / 2, -1, 1))
        relative_angle_deg = float(np.degrees(angle_rad))

        logger.info(f"Stereo (derived): angle={relative_angle_deg:.1f}°, "
                    f"baseline={baseline_mm:.1f}mm")

        _progress(90, 'saving_models')

        # Save models
        cam1_path = self._save_per_camera_model(cam1, pr1)
        cam2_path = self._save_per_camera_model(cam2, pr2)
        stereo_path = self._save_stereo_model(
            cam1, cam2, pr1, pr2, R_stereo, T_stereo, relative_angle_deg,
        )

        # Generate diagnostic figures
        try:
            from pivtools_gui.calibration.calibration_figures import (
                make_camera_placement_figure,
                make_stepped_detection_figure,
                make_stepped_reprojection_figure,
                make_dewarp_overlay_figure,
            )
            stereo_fig_dir = self.base_dir / "calibration" / f"stereo_cam{cam1}_cam{cam2}" / "figures"
            stereo_fig_dir.mkdir(parents=True, exist_ok=True)

            # Per-camera detection figures (3-panel with blob info)
            for cam_num, det in [(cam1, det1), (cam2, det2)]:
                cam_fig_dir = self.base_dir / "calibration" / f"Cam{cam_num}" / "stepped_board" / "figures"
                cam_fig_dir.mkdir(parents=True, exist_ok=True)
                level_a_fig = det.get('_level_A_full')
                level_b_fig = det.get('_level_B_full')
                gray_img = det.get('_gray_uint8')
                if gray_img is not None:
                    make_stepped_detection_figure(
                        gray_img, level_a_fig, level_b_fig,
                        cam_fig_dir / "detection.png",
                        title=f"Cam {cam_num}",
                        blob_info=det.get('_blob_info'),
                    )

            # Reprojection error scatter (per-camera, colored by Z-plane)
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
                }
            make_stepped_reprojection_figure(
                reproj_data,
                stereo_fig_dir / "reprojection_errors.png",
            )

            # Camera placement figure (top-down + side view)
            make_camera_placement_figure(
                [
                    {'label': f'Cam {cam1}', 'rvec': pr1['rvec'], 'tvec': pr1['tvec'], 'color': 'red'},
                    {'label': f'Cam {cam2}', 'rvec': pr2['rvec'], 'tvec': pr2['tvec'], 'color': 'blue'},
                ],
                stereo_fig_dir / "camera_placement.png",
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
            'relative_angle_deg': relative_angle_deg,
            'baseline_mm': baseline_mm,
            'cam1_model_path': str(cam1_path),
            'cam2_model_path': str(cam2_path),
            'stereo_model_path': str(stereo_path),
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
        model_dir = self.base_dir / "calibration" / f"Cam{cam_num}" / "stepped_board" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "camera_model.mat"

        W_img, H_img = pr['image_size']

        save_dict = {
            'camera_matrix': pr['K'],
            'dist_coeffs': pr['dist'],
            'rvecs': pr['rvec'].reshape(3, 1),
            'tvecs': pr['tvec'].reshape(3, 1),
            'rms_error': float(pr['rms']),
            'image_width': W_img,
            'image_height': H_img,
            'image_size': np.array([W_img, H_img]),
            'dot_spacing_mm': self.dot_spacing_mm,
            'datum_frame': self.datum_frame,
            'dt': self.dt,
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

        save_dict = {
            'camera_matrix_1': pr1['K'],
            'dist_coeffs_1': pr1['dist'],
            'camera_matrix_2': pr2['K'],
            'dist_coeffs_2': pr2['dist'],
            'rotation_matrix': R,
            'translation_vector': T,
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
        }

        scipy.io.savemat(str(model_path), save_dict)
        logger.info(f"Saved stereo model: {model_path}")
        return model_path
