"""
grid_detection.py

Shared dotboard grid detection functions used by both planar and stereo
calibration pipelines. Single canonical source — no copies elsewhere.

Algorithm: blob detection → cKDTree neighbor analysis → BFS grid index
assignment → connected component filtering → RANSAC homography → subpixel
refinement.
"""

from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


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


def apply_mask_to_image(img: np.ndarray, mask: np.ndarray, fill_value: int = 255) -> np.ndarray:
    """Apply mask: fill excluded regions (mask=0) with fill_value."""
    masked_img = img.copy()
    masked_img[mask == 0] = fill_value
    return masked_img


def find_largest_grid_component(
    grid_indices: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Find the largest connected component of grid points.

    Two points are connected if they are grid-neighbors:
    - Same row, adjacent columns (|col_diff| == 1, row_diff == 0)
    - Same column, adjacent rows (col_diff == 0, |row_diff| == 1)

    This is used to filter out reflections, which form separate grid components
    that are not connected to the main calibration grid.

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

    # Build adjacency based on grid-neighbor relationship
    # Create lookup: grid_index -> point_index
    index_to_point: Dict[Tuple[int, int], int] = {}
    for i, gi in enumerate(grid_indices):
        key = (int(gi[0]), int(gi[1]))
        index_to_point[key] = i

    # Find all neighbor pairs (4-connected in grid space)
    rows = []
    cols = []

    for i, gi in enumerate(grid_indices):
        col, row = int(gi[0]), int(gi[1])
        # Check 4-connected neighbors: right, left, down, up
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor_key = (col + dc, row + dr)
            if neighbor_key in index_to_point:
                j = index_to_point[neighbor_key]
                rows.append(i)
                cols.append(j)

    if len(rows) == 0:
        # No connections found - each point is its own component
        # Return all points (can't determine which is "main" grid)
        return np.ones(n_points, dtype=bool), n_points, np.ones(n_points, dtype=int)

    # Build sparse adjacency matrix
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    # Find connected components
    n_components, labels = connected_components(adjacency, directed=False)

    # Compute component sizes
    component_sizes = np.bincount(labels)

    # Find largest component
    largest_component = np.argmax(component_sizes)

    # Return mask for largest component
    mask = labels == largest_component

    return mask, n_components, component_sizes


def _assign_grid_indices_bfs(
    centers: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    spacing_px: float,
) -> np.ndarray:
    """
    Assign (col, row) grid indices by BFS walk over the neighbor graph.

    Unlike a global affine inversion, this uses only LOCAL neighbor
    relationships and is therefore robust to perspective distortion,
    lens distortion, and any other smooth non-linear warping.

    Algorithm:
    1. Build a neighbor graph: for each blob, find the nearest blob in
       each of the 4 grid directions (±vec1, ±vec2).
    2. Pick the most-central blob as seed with index (0, 0).
    3. BFS: propagate grid indices to neighbors.

    Parameters
    ----------
    centers : ndarray, shape (N, 2)
        Detected blob centers in pixel coordinates.
    vec1 : ndarray, shape (2,)
        Median column direction vector (roughly horizontal, pointing right).
    vec2 : ndarray, shape (2,)
        Median row direction vector (roughly vertical, pointing down).
    spacing_px : float
        Median inter-dot spacing in pixels.

    Returns
    -------
    grid_indices : ndarray, shape (N, 2), dtype int32
        (col, row) for each center. Unassigned points get (-9999, -9999).
    """
    n = len(centers)
    tree = cKDTree(centers)
    # Search radius must cover the LONGER grid axis. Under perspective
    # distortion one axis can be much larger than median NN spacing
    # (e.g. 163 px vs spacing_px=94 px).  Using just spacing_px * 1.5
    # would miss the far-axis neighbors entirely.
    max_vec_len = max(np.linalg.norm(vec1), np.linalg.norm(vec2))
    max_dist = max(max_vec_len, spacing_px) * 1.5

    # Normalized direction templates for the 4 grid neighbors
    directions = [
        (vec1 / np.linalg.norm(vec1), (1, 0)),    # +col
        (-vec1 / np.linalg.norm(vec1), (-1, 0)),   # -col
        (vec2 / np.linalg.norm(vec2), (0, 1)),     # +row
        (-vec2 / np.linalg.norm(vec2), (0, -1)),    # -row
    ]

    # For each point, find the best neighbor in each direction
    # A neighbor in direction d is the nearest point that lies within
    # a cone around d and at a plausible distance.
    cos_thresh = np.cos(np.radians(30))  # 30° half-angle cone

    neighbors = [[-1] * 4 for _ in range(n)]  # neighbors[i][dir_idx] = j or -1
    for i in range(n):
        # Query nearby points
        nearby_idx = tree.query_ball_point(centers[i], max_dist)
        for dir_idx, (d_hat, _) in enumerate(directions):
            best_j = -1
            best_dist = max_dist
            for j in nearby_idx:
                if j == i:
                    continue
                delta = centers[j] - centers[i]
                dist = np.linalg.norm(delta)
                if dist < spacing_px * 0.4 or dist > max_dist:
                    continue
                cos_angle = np.dot(delta / dist, d_hat)
                if cos_angle > cos_thresh and dist < best_dist:
                    best_dist = dist
                    best_j = j
            neighbors[i][dir_idx] = best_j

    # Seed: pick the point closest to the centroid of all points
    centroid = centers.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(centers - centroid, axis=1)))

    # BFS
    grid_indices = np.full((n, 2), -9999, dtype=np.int32)
    grid_indices[seed] = [0, 0]
    visited = np.zeros(n, dtype=bool)
    visited[seed] = True

    queue = deque([seed])
    while queue:
        i = queue.popleft()
        ci, ri = grid_indices[i]
        for dir_idx, (_, (dc, dr)) in enumerate(directions):
            j = neighbors[i][dir_idx]
            if j < 0 or visited[j]:
                continue
            grid_indices[j] = [ci + dc, ri + dr]
            visited[j] = True
            queue.append(j)

    return grid_indices


def detect_grid_automatic(
    img: np.ndarray,
    detector: cv2.SimpleBlobDetector,
    mask: Optional[np.ndarray] = None,
    grid_spacing_mm: Optional[float] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Automatically detect grid from blob positions - NO pattern size needed.

    Uses OpenCV primitives:
    - SimpleBlobDetector for blob detection
    - Neighbor analysis to find grid vectors
    - RANSAC for robust homography fitting and outlier rejection

    Parameters
    ----------
    img : ndarray
        Input image
    detector : cv2.SimpleBlobDetector
        Blob detector
    mask : ndarray, optional
        Binary mask (255=keep, 0=exclude)
    grid_spacing_mm : float, optional
        Known grid spacing in mm (for calibration output)

    Returns
    -------
    success : bool
    grid_data : dict or None
        Contains: centers, grid_indices, n_cols, n_rows, spacing_px, angle_deg, grid_spacing_mm
    info : dict
        Detection metadata and diagnostics
    """
    gray = to_grayscale_2d(img)
    original_gray = gray.copy()

    if mask is not None:
        gray = apply_mask_to_image(gray, mask, fill_value=255)

    info: Dict[str, Any] = {'method': 'automatic_grid_detection'}

    # Apply CLAHE to normalize uneven illumination before detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 1: Detect blobs (histogram-based single pass)
    mean_intensity = np.mean(gray)
    if mean_intensity > 127:
        # Light background, dark dots - use original
        keypoints = detector.detect(gray)
        info['image_mode'] = 'original'
    else:
        # Dark background, light dots - use inverted
        keypoints = detector.detect(255 - gray)
        info['image_mode'] = 'inverted'

    # Fallback: if too few found, try the other mode
    if len(keypoints) < 9:
        fallback = detector.detect(255 - gray) if mean_intensity > 127 else detector.detect(gray)
        if len(fallback) > len(keypoints):
            keypoints = fallback
            info['image_mode'] = 'inverted' if mean_intensity > 127 else 'original'

    if len(keypoints) < 9:
        info['error'] = f'Too few blobs detected: {len(keypoints)}'
        return False, None, info

    centers = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    info['n_blobs_detected'] = len(centers)
    logger.debug(f"Detected {len(centers)} blobs")

    # Step 2: Find grid spacing using cKDTree (O(N log N) instead of O(N^2))
    n_points = len(centers)

    tree = cKDTree(centers)
    # Use k=5 to detect perspective-distorted grids (stereo/Scheimpflug)
    # where the two grid directions have very different spacings.
    k_query = min(5, n_points)
    nn_dists, _ = tree.query(centers, k=k_query)
    spacing_px = np.median(nn_dists[:, 1])
    info['spacing_px'] = float(spacing_px)
    logger.debug(f"Estimated grid spacing: {spacing_px:.1f} pixels")

    # For perspective-distorted grids, the 3rd-nearest-neighbor captures the
    # secondary spacing (interior points have 2 short + 2 long neighbors).
    # Use this to set a search radius that covers both grid directions.
    if k_query >= 4:
        secondary_spacing_px = np.median(nn_dists[:, 3])
        search_radius = secondary_spacing_px * 1.4
    else:
        search_radius = spacing_px * 1.4

    # Step 3: Find the two grid direction vectors (rotation-invariant)
    #
    # Previous approach assumed the grid was roughly axis-aligned (horizontal
    # < 20° from x-axis, vertical > 70°).  Stereo calibration boards can be
    # rotated 25-30° or more, so we instead find the two dominant neighbor
    # directions from the data.
    #
    # Use k=4 nearest neighbors (not radius search) for direction finding.
    # Interior grid points always have 4 grid neighbors as their nearest
    # neighbors, so diagonals (5th+ nearest) are naturally excluded.  A
    # radius search can include diagonals when the grid axes have similar
    # spacing, corrupting the direction histogram.
    angle_tolerance_deg = 20
    angle_tolerance = np.radians(angle_tolerance_deg)

    # Use k-nearest neighbors: each point's 4 nearest neighbors give
    # the grid direction vectors.  This is robust to perspective distortion
    # and avoids diagonal contamination that radius searches suffer from.
    k_nn = min(5, n_points)  # 4 neighbors + self
    nn_dists_k, nn_idxs_k = tree.query(centers, k=k_nn)

    # Build direction vectors from k-NN (skip self at index 0)
    all_vecs_list = []
    min_dist = spacing_px * 0.4
    for i in range(n_points):
        for ki in range(1, k_nn):
            j = nn_idxs_k[i, ki]
            d = nn_dists_k[i, ki]
            if d > min_dist:
                all_vecs_list.append(centers[j] - centers[i])

    if len(all_vecs_list) < 20:
        info['error'] = 'Not enough neighbor pairs found'
        return False, None, info

    all_vecs = np.array(all_vecs_list, dtype=np.float32)

    # Map all vectors into the upper half-plane so opposing directions merge.
    # A vector and its negation represent the same grid direction.
    # Fold so that y > 0, or if y == 0 then x > 0.  This maps angles to [0, π).
    half_vecs = all_vecs.copy()
    flip_mask = (half_vecs[:, 1] < 0) | ((half_vecs[:, 1] == 0) & (half_vecs[:, 0] < 0))
    half_vecs[flip_mask] *= -1
    half_angles = np.arctan2(half_vecs[:, 1], half_vecs[:, 0])  # [0, π)

    # Find dominant direction: histogram with 1° bins, pick the peak
    n_bins = 180
    hist, bin_edges = np.histogram(np.degrees(half_angles), bins=n_bins, range=(0, 180))
    # Smooth histogram to handle noise (circular convolution with ±3° window)
    kernel = np.ones(7) / 7
    hist_smooth = np.convolve(np.tile(hist, 3), kernel, mode='same')[n_bins:2*n_bins]
    peak1_bin = int(np.argmax(hist_smooth))
    dir1_angle_deg = peak1_bin + 0.5  # center of bin

    # Second direction: mask out ±25° around first peak, find next peak
    # (grid directions are approximately orthogonal)
    mask_width = 25
    suppressed = hist_smooth.copy()
    for offset in range(-mask_width, mask_width + 1):
        suppressed[(peak1_bin + offset) % n_bins] = 0
    peak2_bin = int(np.argmax(suppressed))
    dir2_angle_deg = peak2_bin + 0.5

    dir1_angle = np.radians(dir1_angle_deg)
    dir2_angle = np.radians(dir2_angle_deg)

    # Classify vectors into direction 1 vs direction 2
    angle_diff_1 = np.abs(half_angles - dir1_angle)
    angle_diff_1 = np.minimum(angle_diff_1, np.pi - angle_diff_1)  # wrap-aware
    angle_diff_2 = np.abs(half_angles - dir2_angle)
    angle_diff_2 = np.minimum(angle_diff_2, np.pi - angle_diff_2)

    dir1_mask = angle_diff_1 < angle_tolerance
    dir2_mask = angle_diff_2 < angle_tolerance

    dir1_vecs = half_vecs[dir1_mask]
    dir2_vecs = half_vecs[dir2_mask]

    logger.debug(f"Found {len(dir1_vecs)} dir1 ({dir1_angle_deg:.1f}°), {len(dir2_vecs)} dir2 ({dir2_angle_deg:.1f}°) neighbor pairs")

    if len(dir1_vecs) < 10 or len(dir2_vecs) < 10:
        info['error'] = f'Not enough neighbors in grid directions ({len(dir1_vecs)}, {len(dir2_vecs)})'
        return False, None, info

    # Take median to get robust grid vectors
    raw_vec1 = np.median(dir1_vecs, axis=0)
    raw_vec2 = np.median(dir2_vecs, axis=0)

    # Convention: vec1 = column direction (more horizontal component),
    # vec2 = row direction (more vertical component).
    # Ensure vec1 has larger |x| component, vec2 has larger |y| component.
    if abs(raw_vec1[0]) < abs(raw_vec2[0]):
        raw_vec1, raw_vec2 = raw_vec2, raw_vec1

    # Normalize vec1 to point RIGHT (+x), vec2 to point DOWN (+y)
    if raw_vec1[0] < 0:
        raw_vec1 *= -1
    if raw_vec2[1] < 0:
        raw_vec2 *= -1

    vec1 = raw_vec1
    vec2 = raw_vec2

    info['grid_vec1'] = vec1.tolist()
    info['grid_vec2'] = vec2.tolist()
    logger.debug(f"Grid vector 1 (col): [{vec1[0]:.1f}, {vec1[1]:.1f}]")
    logger.debug(f"Grid vector 2 (row): [{vec2[0]:.1f}, {vec2[1]:.1f}]")

    # Step 4: Assign grid indices via BFS neighborhood walk
    #
    # A global affine inversion fails under perspective distortion because the
    # mapping from grid indices → pixel positions is projective, not affine.
    # Instead we build the grid by walking the neighbor graph: each dot's grid
    # index is determined by its local relationship to already-indexed neighbors.
    # This is robust to arbitrary smooth distortions.

    grid_indices = _assign_grid_indices_bfs(centers, vec1, vec2, spacing_px)

    # Shift so minimum is (0, 0)
    col_min, row_min = grid_indices[:, 0].min(), grid_indices[:, 1].min()
    grid_indices[:, 0] -= col_min
    grid_indices[:, 1] -= row_min

    logger.debug(f"Grid index range: x=[0, {grid_indices[:, 0].max()}], y=[0, {grid_indices[:, 1].max()}]")

    # Step 4.5: Filter out reflections using connected component analysis
    # Real grid and reflections form separate connected components in grid-index space
    # Keep only the largest component (the real calibration grid)
    component_mask, n_components, component_sizes = find_largest_grid_component(grid_indices)

    info['n_components_found'] = n_components
    info['component_sizes'] = component_sizes.tolist()

    if n_components > 1:
        n_rejected = np.sum(~component_mask)
        n_kept = np.sum(component_mask)
        logger.info(
            f"Reflection filtering: Found {n_components} grid components, "
            f"keeping largest ({n_kept} points), rejecting {n_rejected} points (likely reflections)"
        )
        info['n_reflection_points_rejected'] = int(n_rejected)

        # Apply filter
        centers = centers[component_mask]
        grid_indices = grid_indices[component_mask]

        # Re-normalize grid indices after filtering
        col_min, row_min = grid_indices[:, 0].min(), grid_indices[:, 1].min()
        grid_indices[:, 0] -= col_min
        grid_indices[:, 1] -= row_min

        logger.debug(f"After reflection filter - Grid index range: x=[0, {grid_indices[:, 0].max()}], y=[0, {grid_indices[:, 1].max()}]")
    else:
        info['n_reflection_points_rejected'] = 0
        logger.debug("No reflections detected (single connected component)")

    # Step 5: Use RANSAC homography to robustly reject outliers
    # Homography (8 DOF) correctly models perspective-distorted grids,
    # unlike affine (6 DOF) which fails at oblique viewing angles.
    src_pts = grid_indices.astype(np.float32)
    dst_pts = centers.astype(np.float32)

    ransac_thresh = 0.15 * spacing_px
    H_matrix, inliers = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.995,
    )

    if H_matrix is None:
        info['error'] = 'RANSAC homography failed'
        return False, None, info

    inliers = inliers.flatten().astype(bool)
    n_inliers = np.sum(inliers)
    n_outliers = len(inliers) - n_inliers
    logger.debug(f"RANSAC homography: {n_inliers} inliers, {n_outliers} outliers rejected")

    centers = centers[inliers]
    grid_indices = grid_indices[inliers]

    # Step 6: Remove duplicate grid positions (keep best fit)
    src_h = np.hstack([grid_indices.astype(np.float32), np.ones((len(grid_indices), 1), dtype=np.float32)])
    projected = (H_matrix @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    residuals = np.sqrt(np.sum((centers - projected) ** 2, axis=1))

    pos_to_points: Dict[Tuple[int, int], list] = defaultdict(list)
    for i, gi in enumerate(grid_indices):
        pos_key = (gi[0], gi[1])
        pos_to_points[pos_key].append((i, residuals[i]))

    keep_indices = []
    n_dups = 0
    for pos_key, point_list in pos_to_points.items():
        if len(point_list) == 1:
            keep_indices.append(point_list[0][0])
        else:
            best_idx = min(point_list, key=lambda x: x[1])[0]
            keep_indices.append(best_idx)
            n_dups += len(point_list) - 1

    if n_dups > 0:
        logger.debug(f"Removed {n_dups} duplicate grid positions")

    keep_indices_arr = np.array(keep_indices)
    centers = centers[keep_indices_arr]
    grid_indices = grid_indices[keep_indices_arr]

    # Recompute dimensions
    n_cols = grid_indices[:, 0].max() + 1
    n_rows = grid_indices[:, 1].max() + 1
    info['n_cols'] = int(n_cols)
    info['n_rows'] = int(n_rows)

    # Extract rotation angle from homography
    angle_deg = np.degrees(np.arctan2(H_matrix[1, 0], H_matrix[0, 0]))
    info['angle_deg'] = float(angle_deg)
    info['homography_matrix'] = H_matrix.tolist()

    logger.info(f"Automatic detection: {n_cols} cols x {n_rows} rows, {len(centers)} points")

    # Step 7: Subpixel refinement on original image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    try:
        centers_refined = cv2.cornerSubPix(
            original_gray,
            centers.reshape(-1, 1, 2),
            (11, 11),
            (-1, -1),
            criteria
        )
        centers = centers_refined.reshape(-1, 2)
    except cv2.error:
        pass  # Keep original if refinement fails

    # Build output
    grid_data = {
        'centers': centers,
        'grid_indices': grid_indices,
        'n_cols': int(n_cols),
        'n_rows': int(n_rows),
        'spacing_px': spacing_px,
        'angle_deg': angle_deg,
        'grid_spacing_mm': grid_spacing_mm,
    }

    info['success'] = True
    info['n_grid_points'] = len(centers)
    logger.info(f"Grid detection SUCCESS: {n_cols}x{n_rows} grid with {len(centers)} points")

    return True, grid_data, info
