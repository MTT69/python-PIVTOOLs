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
    max_dist = spacing_px * 1.5  # generous search radius for neighbors

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

    # Step 3: Find HORIZONTAL and VERTICAL grid vectors (vectorized)
    angle_tolerance_deg = 20
    angle_tolerance = np.radians(angle_tolerance_deg)

    # Find all neighbor pairs within distance tolerance using cKDTree
    pairs = tree.query_pairs(r=search_radius)
    # Filter pairs below minimum distance and build directed index arrays
    i_list, j_list = [], []
    min_dist = spacing_px * 0.6
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        if d > min_dist:
            # Add both directions for symmetric neighbor analysis
            i_list.append(i)
            j_list.append(j)
            i_list.append(j)
            j_list.append(i)
    i_idx = np.array(i_list, dtype=np.intp)
    j_idx = np.array(j_list, dtype=np.intp)

    if len(i_idx) < 20:
        info['error'] = 'Not enough neighbor pairs found'
        return False, None, info

    # Compute vectors for all valid pairs
    all_vecs = centers[j_idx] - centers[i_idx]  # (M, 2)
    angles_from_horiz = np.arctan2(np.abs(all_vecs[:, 1]), np.abs(all_vecs[:, 0]))

    # Separate horizontal and vertical
    horiz_mask = angles_from_horiz < angle_tolerance
    vert_mask = angles_from_horiz > (np.pi / 2 - angle_tolerance)

    horizontal_vecs_arr = all_vecs[horiz_mask]
    vertical_vecs_arr = all_vecs[vert_mask]

    logger.debug(f"Found {len(horizontal_vecs_arr)} horizontal, {len(vertical_vecs_arr)} vertical neighbor pairs")

    if len(horizontal_vecs_arr) < 10 or len(vertical_vecs_arr) < 10:
        info['error'] = 'Not enough axis-aligned neighbors found'
        return False, None, info

    # Normalize horizontal vectors to point RIGHT (+x) - vectorized
    horizontal_vecs_arr[horizontal_vecs_arr[:, 0] < 0] *= -1

    # Normalize vertical vectors to point DOWN (+y in image pixel coords)
    # This matches the old OpenCV findCirclesGrid convention where row 0 is at the top
    vertical_vecs_arr[vertical_vecs_arr[:, 1] < 0] *= -1

    # Take median to get robust grid vectors
    vec1 = np.median(horizontal_vecs_arr, axis=0)  # X direction (right)
    vec2 = np.median(vertical_vecs_arr, axis=0)    # Y direction (down in image = +y in pixels)

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
