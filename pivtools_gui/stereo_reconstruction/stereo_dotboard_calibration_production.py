#!/usr/bin/env python3
"""
stereo_dotboard_calibration_production.py

Production-ready stereo calibration using automatic RANSAC-based grid detection.
No longer requires specifying pattern_cols and pattern_rows - grid dimensions
are automatically detected.

Saves results to: {BASE_DIR}/calibration/stereo_cam{N}_cam{M}/
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pivtools_core.config import Config, get_config, reload_config
from pivtools_core.image_handling.calibration_loader import get_calibration_frame_count

from pivtools_gui.stereo_reconstruction.stereo_calibration_base import BaseStereoCalibrator


# ===================== CONFIGURATION VARIABLES =====================
# Set these variables for your calibration setup (CLI mode)

SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo"
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo/processed"
CAMERA_PAIR = [1, 2]
FILE_PATTERN = "planar_calibration_plate_%02d.tif"

# CAMERA_SUBFOLDERS: List of subfolder names for each camera (index matches camera number - 1).
#                    e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
#                    Set to [] (empty list) for container formats or when images are in SOURCE_DIR directly.
CAMERA_SUBFOLDERS = ["Cam1", "Cam2"]

# CALIBRATION_SUBFOLDER: Subfolder within camera folders for calibration images.
#                        Leave empty "" to look directly in camera folders.
CALIBRATION_SUBFOLDER = "calibration"

# Grid pattern parameters
# NOTE: PATTERN_COLS and PATTERN_ROWS are no longer required - auto-detected
DOT_SPACING_MM = 12.2222  # Physical dot spacing (required)
ENHANCE_DOTS = False

# Number of calibration images to use (set to None to use all available)
NUM_CALIBRATION_IMAGES = None

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# ===================================================================


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


def detect_grid_automatic(
    img: np.ndarray,
    detector: cv2.SimpleBlobDetector,
    mask: Optional[np.ndarray] = None,
    grid_spacing_mm: Optional[float] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Automatically detect grid from blob positions - NO pattern size needed.

    Uses RANSAC-based detection to find grid structure and reject outliers.

    Returns
    -------
    success : bool
    grid_data : dict or None
        Contains: centers, grid_indices, n_cols, n_rows, spacing_px, angle_deg
    info : dict
        Detection metadata
    """
    gray = to_grayscale_2d(img)
    original_gray = gray.copy()

    if mask is not None:
        gray = gray.copy()
        gray[mask == 0] = 255

    info: Dict[str, Any] = {'method': 'automatic_grid_detection'}

    # Step 1: Detect blobs
    keypoints_orig = detector.detect(gray)
    keypoints_inv = detector.detect(255 - gray)

    if len(keypoints_inv) > len(keypoints_orig):
        keypoints = keypoints_inv
        info['image_mode'] = 'inverted'
    else:
        keypoints = keypoints_orig
        info['image_mode'] = 'original'

    if len(keypoints) < 9:
        info['error'] = f'Too few blobs detected: {len(keypoints)}'
        return False, None, info

    centers = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    info['n_blobs_detected'] = len(centers)

    # Step 2: Find grid spacing
    n_points = len(centers)
    all_distances = np.zeros((n_points, n_points))
    for i in range(n_points):
        all_distances[i] = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
        all_distances[i, i] = np.inf

    nn_distances = np.min(all_distances, axis=1)
    spacing_px = np.median(nn_distances)
    info['spacing_px'] = float(spacing_px)

    # Step 3: Find grid vectors
    horizontal_vecs = []
    vertical_vecs = []
    angle_tolerance = np.radians(20)

    for i in range(n_points):
        for j in range(n_points):
            dist = all_distances[i, j]
            if i != j and dist < spacing_px * 1.4 and dist > spacing_px * 0.6:
                vec = centers[j] - centers[i]
                angle_from_horiz = np.arctan2(abs(vec[1]), abs(vec[0]))
                if angle_from_horiz < angle_tolerance:
                    horizontal_vecs.append(vec)
                elif angle_from_horiz > (np.pi / 2 - angle_tolerance):
                    vertical_vecs.append(vec)

    if len(horizontal_vecs) < 10 or len(vertical_vecs) < 10:
        info['error'] = 'Not enough axis-aligned neighbors found'
        return False, None, info

    horizontal_vecs_arr = np.array(horizontal_vecs)
    vertical_vecs_arr = np.array(vertical_vecs)

    for i in range(len(horizontal_vecs_arr)):
        if horizontal_vecs_arr[i, 0] < 0:
            horizontal_vecs_arr[i] = -horizontal_vecs_arr[i]

    for i in range(len(vertical_vecs_arr)):
        if vertical_vecs_arr[i, 1] > 0:
            vertical_vecs_arr[i] = -vertical_vecs_arr[i]

    vec1 = np.median(horizontal_vecs_arr, axis=0)
    vec2 = np.median(vertical_vecs_arr, axis=0)

    info['grid_vec1'] = vec1.tolist()
    info['grid_vec2'] = vec2.tolist()

    # Step 4: Compute grid coordinates
    origin_idx = np.argmin(centers[:, 0] - centers[:, 1])
    origin = centers[origin_idx]

    A = np.column_stack([vec1, vec2])
    A_inv = np.linalg.inv(A)

    grid_coords_float = []
    for pt in centers:
        delta = pt - origin
        coords = A_inv @ delta
        grid_coords_float.append(coords)

    grid_coords_float = np.array(grid_coords_float)
    grid_indices = np.round(grid_coords_float).astype(np.int32)

    col_min, row_min = grid_indices[:, 0].min(), grid_indices[:, 1].min()
    grid_indices[:, 0] -= col_min
    grid_indices[:, 1] -= row_min

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
    else:
        info['n_reflection_points_rejected'] = 0

    # Step 5: RANSAC fitting
    src_pts = grid_indices.astype(np.float32)
    dst_pts = centers.astype(np.float32)
    ransac_thresh = 0.3 * spacing_px

    affine_matrix, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99
    )

    if affine_matrix is None:
        info['error'] = 'RANSAC failed to fit affine transform'
        return False, None, info

    inliers = inliers.flatten().astype(bool)
    centers = centers[inliers]
    grid_indices = grid_indices[inliers]

    # Step 6: Remove duplicates
    src_clean = grid_indices.astype(np.float32)
    predicted = cv2.transform(src_clean.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
    residuals = np.sqrt(np.sum((centers - predicted) ** 2, axis=1))

    pos_to_points: Dict[Tuple[int, int], list] = defaultdict(list)
    for i, gi in enumerate(grid_indices):
        pos_key = (gi[0], gi[1])
        pos_to_points[pos_key].append((i, residuals[i]))

    keep_indices = []
    for pos_key, point_list in pos_to_points.items():
        if len(point_list) == 1:
            keep_indices.append(point_list[0][0])
        else:
            best_idx = min(point_list, key=lambda x: x[1])[0]
            keep_indices.append(best_idx)

    keep_indices_arr = np.array(keep_indices)
    centers = centers[keep_indices_arr]
    grid_indices = grid_indices[keep_indices_arr]

    n_cols = grid_indices[:, 0].max() + 1
    n_rows = grid_indices[:, 1].max() + 1
    info['n_cols'] = int(n_cols)
    info['n_rows'] = int(n_rows)

    angle_deg = np.degrees(np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]))
    info['angle_deg'] = float(angle_deg)

    # Subpixel refinement
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
        pass

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

    return True, grid_data, info


def apply_cli_settings_to_config() -> Config:
    """Update config.yaml with CLI-mode hardcoded settings."""
    config = get_config()

    config.data["paths"]["source_paths"] = [SOURCE_DIR]
    config.data["paths"]["base_paths"] = [BASE_DIR]
    config.data["paths"]["camera_subfolders"] = CAMERA_SUBFOLDERS
    config.data["paths"]["camera_count"] = len(CAMERA_PAIR)
    config.data["paths"]["camera_numbers"] = CAMERA_PAIR

    config.data["calibration"]["image_format"] = FILE_PATTERN
    config.data["calibration"]["subfolder"] = CALIBRATION_SUBFOLDER

    if NUM_CALIBRATION_IMAGES is not None:
        config.data["calibration"]["num_images"] = NUM_CALIBRATION_IMAGES
    else:
        config.save()
        config = reload_config()
        detected_count = get_calibration_frame_count(camera=CAMERA_PAIR[0], config=config)
        if detected_count > 0:
            config.data["calibration"]["num_images"] = detected_count
            logger.info(f"Auto-detected {detected_count} calibration images")

    # Stereo-specific params - no longer need pattern_cols/rows
    config.data["calibration"]["stereo_dotboard"]["camera_pair"] = CAMERA_PAIR
    config.data["calibration"]["stereo_dotboard"]["dot_spacing_mm"] = DOT_SPACING_MM
    config.data["calibration"]["stereo_dotboard"]["enhance_dots"] = ENHANCE_DOTS
    config.data["calibration"]["stereo_dotboard"]["datum_camera"] = 1  # Default

    config.save()
    logger.info("Updated config.yaml with CLI settings")

    return reload_config()


class StereoDotboardCalibrator(BaseStereoCalibrator):
    """Stereo calibration using automatic RANSAC-based grid detection.

    Grid dimensions (cols, rows) are automatically detected - no need to specify.
    Handles variable grid sizes between cameras by finding intersection of detected points.

    Parameters
    ----------
    config : Config, optional
        Configuration object
    dot_spacing_mm : float
        Physical spacing between dots in millimeters (required)
    enhance_dots : bool
        Whether to apply dot enhancement for better detection
    datum_camera : int
        Which camera defines the coordinate system origin (1 or 2, default: 1)
    **base_kwargs
        Additional arguments passed to BaseStereoCalibrator

    Example
    -------
    >>> calibrator = StereoDotboardCalibrator(
    ...     source_dir="/path/to/images",
    ...     base_dir="/path/to/output",
    ...     camera_pair=[1, 2],
    ...     dot_spacing_mm=28.89,
    ... )
    >>> result = calibrator.process_camera_pair()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        # Pattern-specific params
        dot_spacing_mm: float = 28.89,
        enhance_dots: bool = False,
        datum_camera: int = 1,
        # Legacy params (ignored)
        pattern_cols: Optional[int] = None,
        pattern_rows: Optional[int] = None,
        asymmetric: bool = False,
        # Base class params
        source_dir: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        camera_pair: Optional[List[int]] = None,
        file_pattern: Optional[str] = None,
        camera_subfolders: Optional[List[str]] = None,
        source_path_idx: int = 0,
        dt: float = 1.0,
    ):
        # Get params from config if available
        if config is not None:
            stereo_cfg = config.stereo_dotboard_calibration
            dot_spacing_mm = stereo_cfg.get('dot_spacing_mm', dot_spacing_mm)
            enhance_dots = stereo_cfg.get('enhance_dots', enhance_dots)
            datum_camera = stereo_cfg.get('datum_camera', datum_camera)
            dt = stereo_cfg.get('dt', dt)

        self.dot_spacing_mm = dot_spacing_mm
        self.enhance_dots = enhance_dots
        self.datum_camera = datum_camera

        # Track detected dimensions per frame
        self._detected_cols: Optional[int] = None
        self._detected_rows: Optional[int] = None

        super().__init__(
            config=config,
            source_dir=source_dir,
            base_dir=base_dir,
            camera_pair=camera_pair,
            file_pattern=file_pattern,
            camera_subfolders=camera_subfolders,
            source_path_idx=source_path_idx,
            dt=dt,
        )

    def _create_detector(self) -> cv2.SimpleBlobDetector:
        """Create optimized blob detector for circle grid detection."""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 5000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 5
        return cv2.SimpleBlobDetector_create(params)

    def _enhance_dots_image(self, img: np.ndarray, fixed_radius: int = 9) -> np.ndarray:
        """Enhance white dots for better detection."""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        for cnt in contours:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            center = (int(round(x)), int(round(y)))
            cv2.circle(output, center, fixed_radius, (255,), -1)
        return output

    def detect_pattern(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Detect grid using automatic RANSAC-based detection.

        Returns
        -------
        tuple
            (found: bool, centers: np.ndarray or None, grid_data: dict or None)
            grid_data includes grid_indices, n_cols, n_rows for point matching
        """
        gray = to_grayscale_2d(image)

        if self.enhance_dots:
            gray = self._enhance_dots_image(gray)

        success, grid_data, info = detect_grid_automatic(
            gray, self.detector, mask=None, grid_spacing_mm=self.dot_spacing_mm
        )

        if success and grid_data is not None:
            self._detected_cols = grid_data['n_cols']
            self._detected_rows = grid_data['n_rows']
            logger.debug(f"Auto-detected {grid_data['n_cols']}x{grid_data['n_rows']} grid with {len(grid_data['centers'])} points")
            return True, grid_data['centers'], grid_data

        return False, None, None

    def make_object_points(self) -> np.ndarray:
        """Create placeholder object points - actual points created dynamically."""
        # This is called by base class but we override matching
        # Return empty array - actual points created in _match_object_points
        return np.array([], dtype=np.float32)

    def make_object_points_dynamic(
        self,
        grid_indices: np.ndarray,
    ) -> np.ndarray:
        """Create 3D object points from grid indices."""
        obj_points = []
        for idx in grid_indices:
            col, row = idx[0], idx[1]
            x = col * self.dot_spacing_mm
            y = row * self.dot_spacing_mm
            obj_points.append([x, y, 0.0])
        return np.array(obj_points, dtype=np.float32)

    def _match_object_points(
        self,
        objp: np.ndarray,
        result1: Tuple,
        result2: Tuple,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Match object points between cameras using grid indices.

        Finds intersection of detected grid positions and returns matched points.

        Returns
        -------
        tuple or None
            (obj_pts, img_pts_1, img_pts_2) with matched points only
        """
        # Extract grid data from detection results
        # result format: (found, centers, grid_data)
        if len(result1) < 3 or len(result2) < 3:
            return None

        grid_data_1 = result1[2]
        grid_data_2 = result2[2]

        if grid_data_1 is None or grid_data_2 is None:
            return None

        centers_1 = grid_data_1['centers']
        centers_2 = grid_data_2['centers']
        indices_1 = grid_data_1['grid_indices']
        indices_2 = grid_data_2['grid_indices']

        # Build lookup from grid position to index
        pos_to_idx_1 = {(gi[0], gi[1]): i for i, gi in enumerate(indices_1)}
        pos_to_idx_2 = {(gi[0], gi[1]): i for i, gi in enumerate(indices_2)}

        # Find common grid positions
        common_positions = set(pos_to_idx_1.keys()) & set(pos_to_idx_2.keys())

        if len(common_positions) < 9:
            logger.warning(f"Too few common grid points: {len(common_positions)}")
            return None

        # Build matched arrays
        obj_pts = []
        img_pts_1 = []
        img_pts_2 = []

        for pos in sorted(common_positions):
            col, row = pos
            idx_1 = pos_to_idx_1[pos]
            idx_2 = pos_to_idx_2[pos]

            obj_pts.append([col * self.dot_spacing_mm, row * self.dot_spacing_mm, 0.0])
            img_pts_1.append(centers_1[idx_1])
            img_pts_2.append(centers_2[idx_2])

        logger.info(f"Matched {len(obj_pts)} common grid points between cameras")

        return (
            np.array(obj_pts, dtype=np.float32),
            np.array(img_pts_1, dtype=np.float32),
            np.array(img_pts_2, dtype=np.float32),
        )

    def get_pattern_params(self) -> Dict[str, Any]:
        """Get pattern-specific parameters for saving."""
        return {
            'pattern_type': 'circle_grid_automatic',
            'detected_cols': self._detected_cols,
            'detected_rows': self._detected_rows,
            'dot_spacing_mm': self.dot_spacing_mm,
            'enhance_dots': self.enhance_dots,
            'datum_camera': self.datum_camera,
        }


def main():
    """Main entry point using hardcoded configuration."""
    if USE_CONFIG_DIRECTLY:
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()
    else:
        config = apply_cli_settings_to_config()

    calibrator = StereoDotboardCalibrator(config=config)
    calibrator.run()


if __name__ == "__main__":
    main()
