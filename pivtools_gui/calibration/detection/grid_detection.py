"""
grid_detection.py

Shared dotboard grid detection functions used by both planar and stereo
calibration pipelines. Single canonical source — no copies elsewhere.

Algorithm: photometric flat-fielding → contour/ellipse blob detection →
direction histogram → reciprocal BFS grid assembly → RANSAC homography →
template-matching rescue → connected component filtering.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.ndimage import uniform_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Utility functions (unchanged)
# ---------------------------------------------------------------------------


def to_grayscale_2d(img: np.ndarray) -> np.ndarray:
    """Convert image to 2D grayscale."""
    if img.ndim == 3:
        if img.shape[0] == 1:
            return img[0, :, :]
        elif img.shape[-1] == 1:
            return img[:, :, 0]
        elif img.shape[-1] in (3, 4):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.squeeze(img)
            if gray.ndim == 3:
                return cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            return gray
    return img.copy()


def apply_mask_to_image(
    img: np.ndarray,
    mask: np.ndarray,
    fill_value: int = 255,
) -> np.ndarray:
    """Apply mask: fill excluded regions (mask=0) with fill_value."""
    masked_img = img.copy()
    masked_img[mask == 0] = fill_value
    return masked_img


# ---------------------------------------------------------------------------
# Connected component analysis (unchanged)
# ---------------------------------------------------------------------------


def find_largest_grid_component(
    grid_indices: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Find the largest connected component of grid points.

    Two points are connected if they are grid-neighbors:
    - Same row, adjacent columns (|col_diff| == 1, row_diff == 0)
    - Same column, adjacent rows (col_diff == 0, |row_diff| == 1)

    Parameters
    ----------
    grid_indices : np.ndarray
        Array of (col, row) grid indices for each point, shape (N, 2)

    Returns
    -------
    mask : np.ndarray
        Boolean mask of points belonging to largest component
    n_components : int
        Total number of connected components found
    component_sizes : np.ndarray
        Size of each component
    """
    n_points = len(grid_indices)

    if n_points == 0:
        return np.array([], dtype=bool), 0, np.array([])

    index_to_point: Dict[Tuple[int, int], int] = {}
    for i, gi in enumerate(grid_indices):
        key = (int(gi[0]), int(gi[1]))
        index_to_point[key] = i

    rows = []
    cols = []
    for i, gi in enumerate(grid_indices):
        col, row = int(gi[0]), int(gi[1])
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_key = (col + dc, row + dr)
            if neighbor_key in index_to_point:
                j = index_to_point[neighbor_key]
                rows.append(i)
                cols.append(j)

    if len(rows) == 0:
        return np.ones(n_points, dtype=bool), n_points, np.ones(n_points, dtype=int)

    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    n_components, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    largest_component = np.argmax(component_sizes)
    mask = labels == largest_component
    return mask, n_components, component_sizes


# ---------------------------------------------------------------------------
# Photometric flat-field blob detection
# ---------------------------------------------------------------------------


def _photometric_flat_field(
    img_f32: np.ndarray,
    invert: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Photometric flat-fielding via downsampled morphological division.

    Works in float32 throughout to preserve the full sensor dynamic range.
    Kernel sized to ~2x dot diameter (not image percentage) so it correctly
    bridges dots without smearing dark-board regions where a bright surround
    steals all contrast in 8-bit.

    Parameters
    ----------
    img_f32 : ndarray, float32
        Grayscale float32 input image (native sensor values).
    invert : bool
        If True, invert polarity (for light-on-dark dots).

    Returns
    -------
    flat_field : ndarray, uint8
        Contrast-normalized flat-field image.
    thresh : ndarray, uint8
        Binary threshold image after Otsu + morph cleanup.
    """
    work_img = img_f32.astype(np.float32)
    if invert:
        work_img = work_img.max() - work_img

    # Morphological close at ~2x dot diameter to estimate local background.
    # Kernel sized to dot scale (~70px at 5MP), NOT image scale.
    dot_diam_est = max(70, int(max(work_img.shape) * 0.013))
    k_size = int(dot_diam_est * 2.5) | 1
    scale = 0.25
    small_img = cv2.resize(
        work_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    small_k_size = max(int(k_size * scale) | 1, 3)
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (small_k_size, small_k_size)
    )

    small_bg = cv2.morphologyEx(small_img, cv2.MORPH_CLOSE, small_kernel)
    bg = cv2.resize(
        small_bg, (work_img.shape[1], work_img.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    # Photometric division in float32 — preserves contrast in dark board regions
    bg_safe = np.maximum(bg, 1.0)
    flat_float = np.clip((bg_safe - work_img) / bg_safe, 0, 1)
    flat_field = (flat_float * 255).astype(np.uint8)

    # Otsu binarization + morphological cleanup
    _, thresh = cv2.threshold(flat_field, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    return flat_field, thresh


def _extract_blob_centers(thresh: np.ndarray) -> np.ndarray:
    """Extract sub-pixel blob centers from a binary threshold image.

    Finds external contours, filters by circularity (> 0.4) and area
    (0.15x–3.0x median), then fits a least-squares ellipse to each
    surviving contour for sub-pixel center accuracy.

    Parameters
    ----------
    thresh : ndarray, uint8
        Binary threshold image (255 = blob, 0 = background).

    Returns
    -------
    centers : ndarray, shape (N, 2), dtype float32
        Sub-pixel blob centers. Empty (0, 2) array if none found.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Geometric filtering: circularity + minimum area
    circular_contours = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or area < 50:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.40:
            circular_contours.append(cnt)

    if not circular_contours:
        return np.empty((0, 2), dtype=np.float32)

    # Median-area band-pass filter: reject dots that are too small or too large
    areas = np.array([cv2.contourArea(c) for c in circular_contours])
    median_area = np.median(areas)

    valid_centers = []
    for cnt, area in zip(circular_contours, areas):
        if 0.15 * median_area < area < 3.0 * median_area:
            ellipse = cv2.fitEllipse(cnt)
            valid_centers.append([ellipse[0][0], ellipse[0][1]])

    if not valid_centers:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(valid_centers, dtype=np.float32)


def detect_dotboard_blobs(
    gray_uint8: np.ndarray,
    mask: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect dotboard blobs using photometric flat-fielding and ellipse fitting.

    Works in float32 throughout to preserve the full sensor dynamic range.
    Auto-polarity: tries both dark-on-light and light-on-dark, returns
    both results so the caller can select by grid quality (not just blob count).

    Parameters
    ----------
    gray_uint8 : ndarray
        Grayscale image (any dtype — promoted to float32 internally).
    mask : ndarray, optional
        Binary mask (255=keep, 0=exclude). Excluded regions filled with
        the image mean before flat-fielding.
    **kwargs
        Ignored (backward compatibility for callers passing old params).

    Returns
    -------
    centers : ndarray, shape (N, 2)
        Detected blob centers (from best polarity by count).
        Empty (0, 2) array if detection failed.
    info : dict
        Diagnostic metadata: ``flat_field``, ``thresh``, ``image_mode``,
        ``n_blobs_detected``, and ``_polarity_results`` (list of per-polarity
        results for grid-quality selection by ``detect_grid_automatic``).
    """
    info: Dict[str, Any] = {}

    # Promote to float32 for full dynamic range preservation
    work = gray_uint8.astype(np.float32)
    if mask is not None:
        valid_pixels = gray_uint8[mask != 0]
        if len(valid_pixels) > 0:
            work[mask == 0] = float(np.mean(valid_pixels))

    # Try both polarities
    polarity_results = []
    for invert in [False, True]:
        ff, thresh = _photometric_flat_field(work, invert=invert)
        centers = _extract_blob_centers(thresh)
        polarity_results.append(
            {
                "centers": centers,
                "flat_field": ff,
                "thresh": thresh,
                "invert": invert,
                "n_blobs": len(centers),
            }
        )

    # Default selection: most blobs (overridden by detect_grid_automatic)
    best = max(polarity_results, key=lambda r: r["n_blobs"])
    centers = best["centers"]
    info["flat_field"] = best["flat_field"]
    info["thresh"] = best["thresh"]
    info["image_mode"] = "inverted" if best["invert"] else "original"
    info["n_blobs_detected"] = len(centers)
    info["_polarity_results"] = polarity_results

    logger.debug(
        f"Flat-field blob detection: {polarity_results[0]['n_blobs']} original, "
        f"{polarity_results[1]['n_blobs']} inverted → default {info['image_mode']} "
        f"({len(centers)} blobs)"
    )

    if len(centers) < 9:
        info["error"] = f"Too few blobs detected: {len(centers)}"
        return np.empty((0, 2), dtype=np.float32), info

    return centers, info


# ---------------------------------------------------------------------------
# Direction histogram for grid axis detection
# ---------------------------------------------------------------------------


def _find_grid_directions(
    centers: np.ndarray,
    median_spacing: float,
    k: int = 9,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find the two dominant grid direction vectors from blob positions.

    Uses a direction histogram of all k-nearest-neighbor vectors to find
    the two strongest angular peaks. Robust to steep perspective where the
    two grid axes have very different spacings (e.g. 94 px vs 163 px).

    Parameters
    ----------
    centers : ndarray, shape (N, 2)
        Blob centers.
    median_spacing : float
        Median nearest-neighbor spacing in pixels.
    k : int
        Number of nearest neighbors to query per blob. Default 9 for
        planar/stereo boards. Use k=5 for stepped boards after level
        separation (prevents diagonal contamination).

    Returns
    -------
    (v1, v2) : tuple of ndarray, each shape (2,)
        The two grid direction vectors. Returns None if directions
        cannot be established.
    """
    n = len(centers)
    if n < 9:
        return None

    tree = cKDTree(centers)
    k_actual = min(k, n)
    _, nn_idxs = tree.query(centers, k=k_actual)

    # Build direction vectors from all NN pairs within distance range
    all_vecs = []
    min_dist = 0.3 * median_spacing
    max_dist = 3.0 * median_spacing
    for i in range(n):
        for ki in range(1, k_actual):
            j = nn_idxs[i, ki]
            vec = centers[j] - centers[i]
            dist = np.linalg.norm(vec)
            if min_dist < dist < max_dist:
                all_vecs.append(vec)

    if len(all_vecs) < 20:
        return None
    all_vecs = np.array(all_vecs, dtype=np.float32)

    # Fold angles to [0, π) — opposite directions are the same axis
    angles = np.arctan2(all_vecs[:, 1], all_vecs[:, 0]) % np.pi

    # Histogram with 1° bins, smoothed to suppress jitter
    hist, bin_edges = np.histogram(angles, bins=180, range=(0, np.pi))
    hist_smooth = uniform_filter1d(hist.astype(float), size=5, mode="wrap")

    # Peak 1: strongest direction
    sorted_bins = np.argsort(hist_smooth)[::-1]
    peak1_angle = (bin_edges[sorted_bins[0]] + bin_edges[sorted_bins[0] + 1]) / 2

    # Peak 2: next strongest with 60°–120° separation (roughly orthogonal).
    # Requiring near-orthogonality prevents diagonal capture: under moderate
    # perspective the diagonal direction can be a strong histogram peak, but
    # it is always ~45° from the true grid axes, not ~90°.
    peak2_angle = None
    for b in sorted_bins[1:]:
        candidate = (bin_edges[b] + bin_edges[b + 1]) / 2
        sep = min(abs(candidate - peak1_angle), np.pi - abs(candidate - peak1_angle))
        if np.radians(60) < sep < np.radians(120):
            peak2_angle = candidate
            break

    if peak2_angle is None:
        return None

    # Extract representative vector for each peak: flip to same half-plane, take median
    def _representative_vector(target_angle, tol=np.radians(15)):
        sep = np.minimum(
            np.abs(angles - target_angle), np.pi - np.abs(angles - target_angle)
        )
        cluster = all_vecs[sep < tol].copy()
        if len(cluster) == 0:
            return None
        ref_dir = np.array([np.cos(target_angle), np.sin(target_angle)])
        cluster[cluster @ ref_dir < 0] *= -1
        return np.median(cluster, axis=0)

    v1 = _representative_vector(peak1_angle)
    v2 = _representative_vector(peak2_angle)
    if v1 is None or v2 is None:
        return None

    # Convention: v1 = column direction (larger |x| component, pointing +x),
    # v2 = row direction (larger |y| component, pointing +y).
    # This ensures grid_indices produce a consistent left-to-right, top-to-bottom
    # ordering that downstream calibration expects.
    if abs(v1[0]) < abs(v2[0]):
        v1, v2 = v2, v1
    if v1[0] < 0:
        v1 = -v1
    if v2[1] < 0:
        v2 = -v2

    logger.debug(
        f"Grid directions: v1=[{v1[0]:.1f}, {v1[1]:.1f}] (|{np.linalg.norm(v1):.1f}|), "
        f"v2=[{v2[0]:.1f}, {v2[1]:.1f}] (|{np.linalg.norm(v2):.1f}|), "
        f"angle={np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), 0, 1))):.1f}°"
    )
    return v1, v2


# ---------------------------------------------------------------------------
# Reciprocal BFS grid walk
# ---------------------------------------------------------------------------


def _bfs_grid_walk_dict(
    centers: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    tree: cKDTree,
) -> Dict[Tuple[int, int], int]:
    """Reciprocal BFS grid walk with directional distance bands.

    Starting from the most-central blob, walks the grid in 4 directions
    (±v1, ±v2). Each step:
    1. Predicts the next position from the locally-adaptive step vector.
    2. Searches a 1.8x radius for candidates.
    3. Filters by directional distance band (0.4x–1.6x step magnitude).
    4. Filters by angle cone (< 30° drift from step direction).
    5. Checks reciprocity: the candidate must see the current point as
       its reverse nearest neighbor.

    The locally-adaptive step vectors propagate through the BFS, allowing
    the walk to track perspective-induced spacing gradients.

    Parameters
    ----------
    centers : ndarray, shape (N, 2)
        Blob centers.
    v1, v2 : ndarray, shape (2,)
        Grid direction vectors.
    tree : cKDTree
        Pre-built spatial index on centers.

    Returns
    -------
    grid : dict
        Mapping ``{(col, row): center_index}``.
    """
    # Seed from the blob nearest the centroid
    centroid = np.mean(centers, axis=0)
    _, seed_idx = tree.query(centroid)

    grid: Dict[Tuple[int, int], int] = {(0, 0): seed_idx}
    visited = {seed_idx}
    queue = deque([(seed_idx, 0, 0, v1, v2)])

    # (dc, dr, uses_v1, sign)
    directions = [
        (1, 0, True, 1),
        (-1, 0, True, -1),
        (0, 1, False, 1),
        (0, -1, False, -1),
    ]

    cos_30 = np.cos(np.radians(30))

    while queue:
        curr_idx, c, r, local_v1, local_v2 = queue.popleft()
        curr_pos = centers[curr_idx]

        for dc, dr, is_v1, sign in directions:
            target_coord = (c + dc, r + dr)
            if target_coord in grid:
                continue

            step_vector = local_v1 if is_v1 else local_v2
            predicted_pos = curr_pos + (step_vector * sign)
            step_mag = np.linalg.norm(step_vector)

            # Wide candidate search
            candidate_idxs = tree.query_ball_point(predicted_pos, step_mag * 1.8)

            best_candidate = None
            best_dist = float("inf")
            best_actual_vector = None

            for cand_idx in candidate_idxs:
                if cand_idx in visited:
                    continue

                actual_vector = (centers[cand_idx] - curr_pos) * sign
                v_mag = np.linalg.norm(actual_vector)

                # 1. Direction-specific distance band
                if not (0.4 * step_mag <= v_mag <= 1.6 * step_mag):
                    continue

                # 2. Angle cone (< 30°)
                cos_angle = np.clip(
                    np.dot(actual_vector, step_vector) / (v_mag * step_mag),
                    -1.0,
                    1.0,
                )
                if cos_angle <= cos_30:
                    continue

                # 3. Reciprocity: stepping BACK from the candidate by the ideal
                # lattice vector must land nearest to the current blob. (Stepping
                # back by actual_vector is a tautology -- it reconstructs curr_pos
                # exactly -- so the ideal step is what makes this a real check:
                # an off-lattice decoy fails it, a lattice neighbour passes.)
                expected_reverse = centers[cand_idx] - (step_vector * sign)
                _, reverse_idx = tree.query(expected_reverse, k=1)
                if reverse_idx != curr_idx:
                    continue

                # Keep closest to predicted position
                dist_to_pred = np.linalg.norm(centers[cand_idx] - predicted_pos)
                if dist_to_pred < best_dist:
                    best_dist = dist_to_pred
                    best_candidate = cand_idx
                    best_actual_vector = actual_vector

            if best_candidate is not None:
                grid[target_coord] = best_candidate
                visited.add(best_candidate)
                queue.append(
                    (
                        best_candidate,
                        c + dc,
                        r + dr,
                        best_actual_vector if is_v1 else local_v1,
                        best_actual_vector if not is_v1 else local_v2,
                    )
                )

    return grid


# ---------------------------------------------------------------------------
# Template-matching rescue for missing interior dots
# ---------------------------------------------------------------------------


def _rescue_missing_dots(
    grid: Dict[Tuple[int, int], int],
    centers: np.ndarray,
    flat_field: np.ndarray,
    median_spacing: float,
) -> Tuple[Dict[Tuple[int, int], int], list, list]:
    """Rescue missing interior dots via local homography + template matching.

    For each missing interior grid position (a hole surrounded by existing
    points), predicts the pixel location via a local homography fitted from
    the 16 nearest grid neighbors, then confirms via normalized cross-
    correlation against a template extracted from the closest healthy dot.

    Parameters
    ----------
    grid : dict
        Mapping ``{(col, row): center_index}``.
    centers : ndarray, shape (N, 2)
        Blob centers (will be extended with rescued points).
    flat_field : ndarray, uint8
        Flat-field image for template matching.
    median_spacing : float
        Median dot spacing in pixels.

    Returns
    -------
    rescued_grid : dict
        Updated grid with rescued points appended.
    rescued_centers : list
        Extended center list (original + rescued).
    rescued_nodes : list
        Grid coordinates of rescued points.
    """
    rescued_grid = grid.copy()
    rescued_centers = list(centers)
    rescued_nodes: List[Tuple[int, int]] = []

    if len(grid) < 4:
        return rescued_grid, rescued_centers, rescued_nodes

    # Compute row/col bounds to find interior holes
    row_bounds: Dict[int, list] = {}
    col_bounds: Dict[int, list] = {}
    for c, r in grid.keys():
        if r not in row_bounds:
            row_bounds[r] = [c, c]
        else:
            row_bounds[r] = [min(row_bounds[r][0], c), max(row_bounds[r][1], c)]
        if c not in col_bounds:
            col_bounds[c] = [r, r]
        else:
            col_bounds[c] = [min(col_bounds[c][0], r), max(col_bounds[c][1], r)]

    # Interior holes: positions within both row and column extents
    missing_coords = []
    for c in range(min(col_bounds.keys()), max(col_bounds.keys()) + 1):
        for r in range(min(row_bounds.keys()), max(row_bounds.keys()) + 1):
            if (c, r) not in grid:
                if r in row_bounds and row_bounds[r][0] < c < row_bounds[r][1]:
                    if c in col_bounds and col_bounds[c][0] < r < col_bounds[c][1]:
                        missing_coords.append((c, r))

    if not missing_coords:
        return rescued_grid, rescued_centers, rescued_nodes

    W_temp = int(median_spacing * 0.35)
    h, w = flat_field.shape[:2]

    for mc, mr in missing_coords:
        # Local homography from 16 nearest grid points
        grid_points = sorted(
            rescued_grid.keys(), key=lambda p: (p[0] - mc) ** 2 + (p[1] - mr) ** 2
        )
        local_pts = grid_points[: min(16, len(grid_points))]

        src_pts = np.array(local_pts, dtype=np.float32)
        dst_pts = np.array(
            [rescued_centers[rescued_grid[p]] for p in local_pts],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            continue

        pred_pt = cv2.perspectiveTransform(
            np.array([mc, mr], dtype=np.float32).reshape(1, 1, 2),
            H,
        )[0][0]
        pred_x, pred_y = float(pred_pt[0]), float(pred_pt[1])

        # Template from the nearest healthy neighbor (matches local perspective)
        closest_idx = local_pts[0]
        ref_pos = rescued_centers[rescued_grid[closest_idx]]
        ref_px, ref_py = int(ref_pos[0]), int(ref_pos[1])

        if (
            ref_py - W_temp < 0
            or ref_py + W_temp >= h
            or ref_px - W_temp < 0
            or ref_px + W_temp >= w
        ):
            continue
        template = flat_field[
            ref_py - W_temp : ref_py + W_temp, ref_px - W_temp : ref_px + W_temp
        ]
        if template.size == 0:
            continue

        # Search ROI around predicted position
        W_search = int(median_spacing * 0.6)
        s_x1, s_y1 = int(pred_x - W_search), int(pred_y - W_search)
        s_x2, s_y2 = int(pred_x + W_search), int(pred_y + W_search)

        if s_y1 < 0 or s_y2 >= h or s_x1 < 0 or s_x2 >= w:
            continue

        roi = flat_field[s_y1:s_y2, s_x1:s_x2]
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            continue

        try:
            res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.65:
                cx_global = s_x1 + max_loc[0] + W_temp
                cy_global = s_y1 + max_loc[1] + W_temp
                rescued_centers.append((float(cx_global), float(cy_global)))
                rescued_grid[(mc, mr)] = len(rescued_centers) - 1
                rescued_nodes.append((mc, mr))
        except cv2.error as e:
            logger.debug(f"Template match rescue failed at grid ({mc},{mr}): {e}")

    n_rescued = len(rescued_nodes)
    if n_rescued > 0:
        logger.info(f"Template rescue: {n_rescued} missing interior dot(s) recovered")

    return rescued_grid, rescued_centers, rescued_nodes


def _filter_connected_dict(
    grid: Dict[Tuple[int, int], int],
) -> Dict[Tuple[int, int], int]:
    """Prune orphaned grid points that lack any orthogonal neighbor."""
    filtered = {}
    for (c, r), idx in grid.items():
        if any(
            (c + dc, r + dr) in grid for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        ):
            filtered[(c, r)] = idx
    return filtered


def _refine_grid_outliers(
    grid: Dict[Tuple[int, int], int],
    centers,
    residual_threshold: float = 2.0,
) -> Tuple[Dict[Tuple[int, int], int], list, list]:
    """Detect dots displaced by water droplets and infill from the grid model.

    Fits a homography from grid indices to pixel positions, identifies dots
    whose residual exceeds the threshold (contaminated by droplets or
    occlusions), removes them, re-fits from clean dots only, then infills
    the outlier positions with model predictions.

    Parameters
    ----------
    grid : dict
        Mapping ``{(col, row): center_index}``.
    centers : list or ndarray
        Center positions (will be extended with infilled points).
    residual_threshold : float
        Max reprojection residual in pixels to accept a dot (default 2.0).

    Returns
    -------
    grid : dict
        Updated grid with outlier positions infilled.
    centers : list
        Extended center list.
    infilled_nodes : list
        Grid coordinates of infilled points.
    """
    if len(grid) < 9:
        return grid, centers, []

    centers = list(centers)
    grid_keys = list(grid.keys())
    src_pts = np.array(grid_keys, dtype=np.float32)
    dst_pts = np.array([centers[grid[k]] for k in grid_keys], dtype=np.float32)

    # First pass: fit with all dots, find outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, residual_threshold)
    if H is None:
        return grid, centers, []

    predicted = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(dst_pts - predicted, axis=1)
    outlier_mask = residuals > residual_threshold

    if not outlier_mask.any():
        return grid, centers, []

    # Second pass: re-fit from clean dots only for better prediction
    clean_src = src_pts[~outlier_mask]
    clean_dst = dst_pts[~outlier_mask]
    H_clean, _ = cv2.findHomography(
        clean_src, clean_dst, cv2.RANSAC, residual_threshold
    )
    if H_clean is None:
        H_clean = H

    # Infill: replace outlier positions with model predictions
    outlier_keys = [grid_keys[i] for i in range(len(grid_keys)) if outlier_mask[i]]
    outlier_src = np.array(outlier_keys, dtype=np.float32)
    infilled_positions = cv2.perspectiveTransform(
        outlier_src.reshape(-1, 1, 2),
        H_clean,
    ).reshape(-1, 2)

    infilled_nodes: List[Tuple[int, int]] = []
    for key, new_pos in zip(outlier_keys, infilled_positions):
        centers.append(new_pos)
        grid[key] = len(centers) - 1
        infilled_nodes.append(key)

    n_infilled = len(infilled_nodes)
    if n_infilled > 0:
        logger.info(
            f"Grid outlier refinement: {n_infilled} droplet-biased dot(s) infilled from model"
        )

    return grid, centers, infilled_nodes


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def detect_grid_automatic(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    grid_spacing_mm: Optional[float] = None,
    k_neighbors: int = 9,
    **kwargs,
) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Automatically detect a dotboard grid from an image.

    Full pipeline: photometric flat-fielding → contour/ellipse blob detection →
    direction histogram → reciprocal BFS grid assembly → RANSAC homography →
    template-matching rescue → connected component filtering.

    Parameters
    ----------
    img : ndarray
        Input image (any format — converted to grayscale internally).
    mask : ndarray, optional
        Binary mask (255=keep, 0=exclude).
    grid_spacing_mm : float, optional
        Known grid spacing in mm (passed through to output).
    k_neighbors : int
        k-NN count for direction histogram. Default 9 for planar/stereo.
        Use 5 for stepped boards after level separation.
    **kwargs
        Ignored (backward compatibility for callers passing old params
        like ``detector`` or ``clahe_clip_limit``).

    Returns
    -------
    success : bool
    grid_data : dict or None
        Contains: ``centers``, ``grid_indices``, ``synthetic_mask`` (bool per
        point: True = rescued/infilled, not a measured blob centroid),
        ``n_cols``, ``n_rows``, ``spacing_px``, ``angle_deg``, ``grid_spacing_mm``.
    info : dict
        Detection metadata and diagnostics.
    """
    gray = to_grayscale_2d(img)

    if mask is not None:
        gray = apply_mask_to_image(gray, mask, fill_value=int(np.mean(gray)))

    info: Dict[str, Any] = {"method": "automatic_grid_detection"}

    # Step 1: Blob detection — try both polarities, select by grid quality
    # (not just blob count). More raw blobs doesn't help if grid assembly fails.
    _, blob_info = detect_dotboard_blobs(gray, mask=None)  # mask already applied
    polarity_results = blob_info.get("_polarity_results", [])

    best_grid = None
    best_centers = None
    best_flat_field = None
    best_spacing = None
    best_polarity_info = {}

    for pr in polarity_results:
        ctrs = pr["centers"]
        if len(ctrs) < 16:
            continue

        # Estimate spacing for this polarity
        pr_tree = cKDTree(ctrs)
        pr_nn_dists, _ = pr_tree.query(ctrs, k=2)
        pr_spacing = float(np.median(pr_nn_dists[:, 1]))

        # Try to build a grid
        dir_result = _find_grid_directions(ctrs, pr_spacing, k=k_neighbors)
        if dir_result is None:
            continue
        v1, v2 = dir_result

        try:
            grid_dict = _bfs_grid_walk_dict(ctrs, v1, v2, pr_tree)
        except Exception as e:
            logger.debug(f"BFS grid walk failed for polarity candidate: {e}")
            continue

        if best_grid is None or len(grid_dict) > len(best_grid):
            best_grid = grid_dict
            best_centers = ctrs
            best_flat_field = pr["flat_field"]
            best_spacing = pr_spacing
            best_polarity_info = {
                "image_mode": "inverted" if pr["invert"] else "original",
                "n_blobs_detected": pr["n_blobs"],
            }

    if best_grid is None or len(best_grid) < 9:
        info["error"] = "Grid assembly failed on both polarities"
        return False, None, info

    grid_dict = best_grid
    centers = best_centers
    flat_field = best_flat_field
    spacing_px = best_spacing
    info.update(best_polarity_info)
    info["flat_field"] = flat_field
    info["spacing_px"] = spacing_px

    cKDTree(centers)
    n_bfs = len(grid_dict)
    logger.debug(
        f"BFS found {n_bfs} of {len(centers)} blobs ({info.get('image_mode', '?')} polarity)"
    )

    # Step 2: RANSAC homography validation
    grid_keys = list(grid_dict.keys())
    src_pts = np.array(grid_keys, dtype=np.float32)
    dst_pts = np.array([centers[grid_dict[k]] for k in grid_keys], dtype=np.float32)

    ransac_thresh = 0.15 * spacing_px
    info["ransac_threshold"] = float(ransac_thresh)
    H_matrix, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.995,
    )
    if H_matrix is None:
        info["error"] = "RANSAC homography failed"
        return False, None, info

    inlier_mask = inlier_mask.flatten().astype(bool)
    validated_grid = {
        k: grid_dict[k] for k, inlier in zip(grid_keys, inlier_mask) if inlier
    }
    n_rejected = int(np.sum(~inlier_mask))
    if n_rejected > 0:
        logger.debug(
            f"RANSAC rejected {n_rejected} outliers, kept {len(validated_grid)}"
        )
    info["ransac_n_rejected"] = n_rejected

    if len(validated_grid) < 9:
        info["error"] = f"RANSAC left only {len(validated_grid)} inliers (need >= 9)"
        return False, None, info

    # Step 3: Template-matching rescue for missing interior dots
    validated_grid, rescued_centers, rescued_nodes = _rescue_missing_dots(
        validated_grid,
        centers,
        flat_field,
        spacing_px,
    )

    # Step 4: Grid smoothness enforcement — detect droplet-biased dots, infill from model
    validated_grid, rescued_centers, infilled_nodes = _refine_grid_outliers(
        validated_grid,
        rescued_centers,
    )

    # Step 5: Prune orphaned points
    validated_grid = _filter_connected_dict(validated_grid)

    # Step 6: Convert dict grid to ndarray format
    all_centers_list = rescued_centers
    grid_keys_final = list(validated_grid.keys())
    grid_center_indices = [validated_grid[k] for k in grid_keys_final]

    final_centers = np.array(
        [all_centers_list[i] for i in grid_center_indices],
        dtype=np.float32,
    )
    grid_indices = np.array(grid_keys_final, dtype=np.int32)

    # Provenance masks: rescued (template-matched) and infilled (model-predicted)
    # points, named in the pre-normalization key frame — the same frame as
    # grid_keys_final (the index shift below rewrites the ndarray, not the dict
    # keys). A node rescued in Step 3 and then infilled in Step 4 ends up in both
    # lists; infill is its final state, so it counts as infilled only. The two
    # masks are therefore disjoint, and the counts stay self-consistent
    # (n_rescued + n_infilled == n_synthetic) after the prune/island filters
    # below trim them in lockstep with the points.
    infilled_set = set(infilled_nodes)
    rescued_set = set(rescued_nodes) - infilled_set
    rescued_mask = np.array([k in rescued_set for k in grid_keys_final], dtype=bool)
    infilled_mask = np.array([k in infilled_set for k in grid_keys_final], dtype=bool)
    synthetic_mask = rescued_mask | infilled_mask

    # Normalize so minimum index is (0, 0)
    if len(grid_indices) > 0:
        grid_indices[:, 0] -= grid_indices[:, 0].min()
        grid_indices[:, 1] -= grid_indices[:, 1].min()

    # Step 9: Largest connected component (filter reflections)
    if len(final_centers) >= 9:
        comp_mask, n_comp, comp_sizes = find_largest_grid_component(grid_indices)
        if n_comp > 1:
            n_island = int(np.sum(~comp_mask))
            final_centers = final_centers[comp_mask]
            rescued_mask = rescued_mask[comp_mask]
            infilled_mask = infilled_mask[comp_mask]
            synthetic_mask = synthetic_mask[comp_mask]
            grid_indices = grid_indices[comp_mask]
            grid_indices[:, 0] -= grid_indices[:, 0].min()
            grid_indices[:, 1] -= grid_indices[:, 1].min()
            logger.debug(f"Removed {n_island} island points ({n_comp} components → 1)")
        info["n_components_found"] = n_comp
        info["component_sizes"] = comp_sizes.tolist()

    if len(final_centers) < 9:
        info["error"] = f"Final grid has only {len(final_centers)} points (need >= 9)"
        return False, None, info

    # Re-fit clean homography from final points
    H_clean, _ = cv2.findHomography(
        grid_indices.astype(np.float32),
        final_centers,
        method=0,
    )
    if H_clean is not None:
        H_matrix = H_clean

    # Compute grid dimensions
    n_cols = int(grid_indices[:, 0].max()) + 1
    n_rows = int(grid_indices[:, 1].max()) + 1

    # Rotation angle from homography
    angle_deg = float(np.degrees(np.arctan2(H_matrix[1, 0], H_matrix[0, 0])))

    info["n_cols"] = n_cols
    info["n_rows"] = n_rows
    info["angle_deg"] = angle_deg
    info["homography_matrix"] = H_matrix.tolist()
    info["success"] = True
    info["n_grid_points"] = len(final_centers)

    # Edge warning
    h, w = gray.shape[:2]
    edge_margin = spacing_px * 0.5
    near_edge = (
        (final_centers[:, 0] < edge_margin)
        | (final_centers[:, 0] > w - edge_margin)
        | (final_centers[:, 1] < edge_margin)
        | (final_centers[:, 1] > h - edge_margin)
    )
    edge_fraction = float(np.mean(near_edge))
    info["edge_fraction"] = edge_fraction
    if edge_fraction > 0.15:
        info["warning"] = (
            f"Possible partial board: {edge_fraction:.0%} of points near image edge"
        )
        logger.warning(info["warning"])

    # The provenance masks are filtered in lockstep with the points (Step 9). A
    # future edit that trims one array but forgets a mask desyncs the figure and
    # the persisted diagnostics silently — fail loudly instead. A raise (not an
    # assert) so the guard survives `python -O`.
    n_final = len(final_centers)
    if not (
        len(grid_indices) == n_final
        and len(rescued_mask) == n_final
        and len(infilled_mask) == n_final
        and len(synthetic_mask) == n_final
    ):
        raise RuntimeError("synthetic/provenance mask desynced from final grid points")

    # Counts are survivors-in-the-final-grid, not operations attempted: a
    # rescued/infilled dot pruned as an orphan or island above is gone from both
    # the mask and these numbers, so the figure and the persisted diagnostics agree.
    info["n_rescued"] = int(np.count_nonzero(rescued_mask))
    info["n_infilled"] = int(np.count_nonzero(infilled_mask))
    info["n_synthetic"] = int(np.count_nonzero(synthetic_mask))

    grid_data = {
        "centers": final_centers,
        "grid_indices": grid_indices,
        "synthetic_mask": synthetic_mask,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "spacing_px": spacing_px,
        "angle_deg": angle_deg,
        "grid_spacing_mm": grid_spacing_mm,
    }

    logger.info(
        f"Grid detection SUCCESS: {n_cols}x{n_rows} grid with {len(final_centers)} points"
    )
    return True, grid_data, info
