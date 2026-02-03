#!/usr/bin/env python3
"""
planar_calibration_multiview.py

Pure Multi-View Dotboard Calibration script.
- Aggregates multiple views of a calibration board.
- Solves for Intrinsics (Camera Matrix + Distortion) using OpenCV.
- Saves grid detections and final model to .mat files.
- Visualizes detections for every image.

Now uses RANSAC-based automatic grid detection (no pattern size required).
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pivtools_core.config import get_config, reload_config
from pivtools_core.image_handling.load_images import read_image
from pivtools_core.image_handling.calibration_loader import (
    read_calibration_image as core_read_calibration_image,
    get_calibration_frame_count,
)

# ===================== CONFIGURATION =====================

# SOURCE_DIR: Root directory containing data
SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images/enhanced"

# BASE_DIR: Output directory.
# Results save to: {BASE_DIR}/calibration/Cam{N}/dotboard_planar/...
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/planar_images"

# CALIBRATION_SUBFOLDER: Subfolder for images (leave "" if in root)
CALIBRATION_SUBFOLDER = ""

# CAMERA_NUMS: List of camera numbers to process (1-based), e.g. [1, 2] for stereo
CAMERA_NUMS = [1]

# CAMERA_SUBFOLDERS: List of subfolder names for each camera (index matches camera number - 1).
#                    e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
#                    Set to [] (empty list) for container formats or when images are in SOURCE_DIR directly.
CAMERA_SUBFOLDERS = []

# FILE_PATTERN: Image naming format (e.g., "calib%05d.tif", "*.tif")
FILE_PATTERN = "planar_calibration_plate_%02d.tif"

# GRID PARAMETERS
# NOTE: PATTERN_COLS and PATTERN_ROWS are no longer needed - grid is auto-detected
DOT_SPACING_MM = 12.22  # Physical dot spacing in mm (required for calibration)
ENHANCE_DOTS = False

# Number of calibration images to use (set to None to use all available)
NUM_CALIBRATION_IMAGES = None

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# LOGGING SETUP
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==========================================================================
# AUTOMATIC GRID DETECTION FUNCTIONS
# ==========================================================================

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
    - RANSAC for robust affine fitting and outlier rejection

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
    logger.debug(f"Detected {len(centers)} blobs")

    # Step 2: Find grid spacing (vectorized for speed)
    n_points = len(centers)

    # Compute all pairwise distances using broadcasting (much faster than loop)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, N, 2)
    all_distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    np.fill_diagonal(all_distances, np.inf)

    # Find nearest neighbor distance for each point
    nn_distances = np.min(all_distances, axis=1)
    spacing_px = np.median(nn_distances)
    info['spacing_px'] = float(spacing_px)
    logger.debug(f"Estimated grid spacing: {spacing_px:.1f} pixels")

    # Step 3: Find HORIZONTAL and VERTICAL grid vectors (vectorized)
    angle_tolerance_deg = 20
    angle_tolerance = np.radians(angle_tolerance_deg)

    # Find all neighbor pairs within distance tolerance
    dist_mask = (all_distances > spacing_px * 0.6) & (all_distances < spacing_px * 1.4)
    i_idx, j_idx = np.where(dist_mask)

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

    # Step 4: Compute grid coordinates for each point
    # Solve: point = origin + col * vec1 + row * vec2
    # Use BOTTOM-LEFT point as origin (min x, max y in image pixel coords)
    origin_idx = np.argmin(centers[:, 0] - centers[:, 1])  # Bottom-left
    origin = centers[origin_idx]

    logger.debug(f"Origin (bottom-left): ({origin[0]:.1f}, {origin[1]:.1f})")

    # Build transformation matrix: [vec1, vec2] @ [col, row].T = point - origin
    A = np.column_stack([vec1, vec2])
    A_inv = np.linalg.inv(A)

    grid_coords_float = []
    for pt in centers:
        delta = pt - origin
        coords = A_inv @ delta  # [col, row]
        grid_coords_float.append(coords)

    grid_coords_float = np.array(grid_coords_float)

    # Round to nearest integer for grid indices
    grid_indices = np.round(grid_coords_float).astype(np.int32)

    # Shift so minimum is (0, 0) - ensures top-left of detected grid is (0,0)
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

    # Step 5: Use RANSAC to robustly fit affine transform and reject outliers
    src_pts = grid_indices.astype(np.float32)
    dst_pts = centers.astype(np.float32)

    ransac_thresh = 0.15 * spacing_px  # Tighter tolerance for better outlier rejection
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
    n_inliers = np.sum(inliers)
    n_outliers = len(inliers) - n_inliers
    logger.debug(f"RANSAC: {n_inliers} inliers, {n_outliers} outliers rejected")

    centers = centers[inliers]
    grid_indices = grid_indices[inliers]

    # Step 6: Remove duplicate grid positions (keep best fit)
    # Compute residuals for remaining points
    src_clean = grid_indices.astype(np.float32)
    predicted = cv2.transform(src_clean.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
    residuals = np.sqrt(np.sum((centers - predicted) ** 2, axis=1))

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

    # Extract rotation angle from affine matrix
    angle_deg = np.degrees(np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]))
    info['angle_deg'] = float(angle_deg)
    info['affine_matrix'] = affine_matrix.tolist()

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


def apply_cli_settings_to_config():
    """Update config.yaml with CLI-mode hardcoded settings.

    This function writes the hardcoded configuration variables to config.yaml,
    ensuring the centralized image loading system uses the correct paths and settings.

    Returns
    -------
    Config
        The reloaded config object with updated settings
    """
    config = get_config()

    # Paths
    config.data["paths"]["source_paths"] = [SOURCE_DIR]
    config.data["paths"]["base_paths"] = [BASE_DIR]
    config.data["paths"]["camera_subfolders"] = CAMERA_SUBFOLDERS
    config.data["paths"]["camera_count"] = len(CAMERA_NUMS)
    config.data["paths"]["camera_numbers"] = CAMERA_NUMS

    # Calibration settings
    config.data["calibration"]["image_format"] = FILE_PATTERN
    config.data["calibration"]["subfolder"] = CALIBRATION_SUBFOLDER
    if NUM_CALIBRATION_IMAGES is not None:
        config.data["calibration"]["num_images"] = NUM_CALIBRATION_IMAGES

    # Dotboard-specific params (for planar calibration)
    # NOTE: pattern_cols and pattern_rows are no longer required - grid is auto-detected
    config.data["calibration"]["dotboard"]["dot_spacing_mm"] = DOT_SPACING_MM
    config.data["calibration"]["dotboard"]["enhance_dots"] = ENHANCE_DOTS
    config.data["calibration"]["dotboard"]["datum_frame"] = 1  # Default datum frame

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


class MultiViewCalibrator:
    """
    Multi-view dotboard calibrator using automatic RANSAC-based grid detection.

    No longer requires pattern_cols/pattern_rows - grid dimensions are automatically
    detected from the calibration images.
    """

    def __init__(
        self,
        source_dir,
        base_dir,
        camera_count,
        file_pattern,
        dot_spacing_mm=28.89,
        enhance_dots=False,
        config=None,
        datum_frame=1,
        # Legacy params (ignored but kept for backward compatibility)
        pattern_cols=None,
        pattern_rows=None,
        asymmetric=False,
    ):
        """
        Initialize the multi-view calibrator.

        Parameters
        ----------
        source_dir : str or Path
            Directory containing calibration images
        base_dir : str or Path
            Output directory for calibration results
        camera_count : int
            Number of cameras to process
        file_pattern : str
            Image file naming pattern (e.g., 'calib%05d.tif')
        dot_spacing_mm : float
            Physical spacing between dots in millimeters (required for calibration)
        enhance_dots : bool
            Whether to apply dot enhancement for better detection
        config : Config, optional
            Configuration object
        datum_frame : int
            Which calibration image defines the world coordinate origin (1-based, default: 1)
        pattern_cols, pattern_rows : int, optional
            DEPRECATED: Grid dimensions are now automatically detected
        asymmetric : bool
            DEPRECATED: Only symmetric grids supported with automatic detection
        """
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.camera_count = camera_count
        self.file_pattern = file_pattern
        self.dot_spacing_mm = dot_spacing_mm
        self.enable_dot_enhancement = enhance_dots
        self._config = config
        self.datum_frame = datum_frame

        # These will be populated during detection
        self._detected_cols: Optional[int] = None
        self._detected_rows: Optional[int] = None

        # Create blob detector
        self.detector = self._create_blob_detector()

        # Setup output structure
        self._setup_directories()

    def _create_blob_detector(self):
        """Create optimized blob detector for circle grid detection"""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50  # Reduced to catch smaller dots
        params.maxArea = 5000
        # Shape filtering to reject distorted reflections (relaxed thresholds)
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        params.filterByConvexity = False  # Keep disabled - not useful for dots
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 10  # Faster detection (was 5)
        return cv2.SimpleBlobDetector_create(params)

    def _setup_directories(self):
        """Create output directories with /dotboard_planar structure"""
        for cam_num in range(1, self.camera_count + 1):
            # NEW PATH STRUCTURE: .../CamX/dotboard_planar/
            cam_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "dotboard_planar"
            (cam_base / "indices").mkdir(parents=True, exist_ok=True)
            (cam_base / "model").mkdir(parents=True, exist_ok=True)
            (cam_base / "figures").mkdir(parents=True, exist_ok=True)

    def _get_camera_input_dir(self, cam_num: int) -> Path:
        """Get the input directory for calibration images.

        Uses build_calibration_camera_path for path resolution from
        calibration_sources config.
        """
        if self._config is not None:
            from pivtools_core.image_handling.path_utils import build_calibration_camera_path
            return build_calibration_camera_path(self._config, source_path_idx=0, camera=cam_num)

        # Fallback: use source_dir directly
        return self.source_dir

    def _is_container_format(self):
        """Check if file pattern is a container format (.set, .cine).

        Uses config.calibration_is_container_format when available, otherwise
        falls back to pattern-based detection.

        Note: IM7 files with % patterns (e.g., B%05d.im7) are individual files,
        not containers. Only treat as container if it's a single file without
        a printf-style pattern.
        """
        if self._config is not None:
            return self._config.calibration_is_container_format

        # Fallback: pattern-based detection
        pattern_lower = self.file_pattern.lower()
        # If pattern has %, it's individual numbered files, not a container
        if "%" in self.file_pattern:
            return False
        # Only .set and .cine are true multi-frame containers
        return '.set' in pattern_lower or '.cine' in pattern_lower

    def _read_image(self, img_path=None, camera=1, img_index=1):
        """Read calibration image using core utilities.

        When config is available, uses the unified core reader which handles
        all formats consistently. Falls back to direct reading for CLI mode.

        Parameters
        ----------
        img_path : str or Path, optional
            Path to image file (used for fallback/CLI mode)
        camera : int
            Camera number (1-based)
        img_index : int
            Image index (1-based)

        Returns
        -------
        np.ndarray or None
            Image as uint8 array, or None if reading failed
        """
        # Use core reader when config is available
        if self._config is not None:
            try:
                img = core_read_calibration_image(
                    idx=img_index,
                    camera=camera,
                    config=self._config,
                    source_path_idx=0,
                    normalize_uint8=True,
                )
                if img is not None and img.ndim == 3:
                    img = img[0]  # Extract single frame if needed
                logger.info(f"Read image {img_index} shape: {img.shape}, dtype: {img.dtype}")
                return img
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Error reading image {img_index}: {e}")
                return None
            except Exception as e:
                logger.warning(f"Unexpected error reading image {img_index}: {e}")
                return None

        # Fallback: direct reading for CLI mode without config
        return self._read_image_direct(img_path, camera, img_index)

    def _read_image_direct(self, img_path, camera=1, img_index=1):
        """Direct image reading fallback for CLI mode without config."""
        try:
            img_path_lower = str(img_path).lower()
            if self._is_container_format():
                if '.set' in img_path_lower:
                    img = read_image(str(img_path), camera_no=camera, im_no=img_index)
                elif '.cine' in img_path_lower:
                    from pivtools_core.image_handling.readers.cine_reader import read_cine_single
                    img = read_cine_single(str(img_path), idx=img_index)
                else:
                    img = read_image(str(img_path))
            elif '.im7' in img_path_lower:
                img = read_image(str(img_path), camera_no=camera, frames=1, frames_per_camera=1)
            else:
                img = read_image(str(img_path))

            if img is None:
                return None

            logger.info(f"Read image shape: {img.shape}, dtype: {img.dtype}")

            # Normalize to uint8
            if img.dtype == np.bool_:
                img = img.astype(np.uint8) * 255
            elif img.dtype in [np.float32, np.float64]:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            elif img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)

            return img
        except Exception as e:
            logger.warning(f"Error reading image {img_path}: {e}")
            return None

    def enhance_dots_image(self, img, fixed_radius=9):
        """Enhance dots for easier detection"""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        for cnt in contours:
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            center = (int(round(x)), int(round(y)))
            cv2.circle(output, center, fixed_radius, (255,), -1)
        return output

    def make_object_points_dynamic(
        self,
        grid_indices: np.ndarray,
        n_cols: int,
        n_rows: int,
    ) -> np.ndarray:
        """
        Create real-world 3D coordinates (Z=0) for detected grid points.

        Unlike the old make_object_points(), this maps each detected point
        to its world coordinates based on its grid index.

        Parameters
        ----------
        grid_indices : np.ndarray
            Array of (col, row) indices for each detected point, shape (N, 2)
        n_cols : int
            Detected number of columns
        n_rows : int
            Detected number of rows

        Returns
        -------
        np.ndarray
            3D object points, shape (N, 3), with Z=0
        """
        obj_points = []
        for idx in grid_indices:
            col, row = idx[0], idx[1]
            x = col * self.dot_spacing_mm
            y = row * self.dot_spacing_mm
            obj_points.append([x, y, 0.0])
        return np.array(obj_points, dtype=np.float32)

    def detect_grid(self, img) -> Tuple[bool, Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Detect grid points using automatic RANSAC-based detection.

        No pattern size required - grid dimensions are automatically detected.

        Parameters
        ----------
        img : np.ndarray
            Input image

        Returns
        -------
        success : bool
            Whether detection succeeded
        centers : np.ndarray or None
            Detected dot centers, shape (N, 2)
        grid_data : dict or None
            Detection metadata including grid_indices, n_cols, n_rows
        """
        gray = to_grayscale_2d(img)

        if self.enable_dot_enhancement:
            gray = self.enhance_dots_image(gray)

        # Use automatic detection
        success, grid_data, info = detect_grid_automatic(
            gray,
            self.detector,
            mask=None,
            grid_spacing_mm=self.dot_spacing_mm
        )

        if success and grid_data is not None:
            # Store detected dimensions
            self._detected_cols = grid_data['n_cols']
            self._detected_rows = grid_data['n_rows']
            return True, grid_data['centers'], grid_data

        return False, None, None

    def save_visualization(
        self,
        img,
        grid_points,
        img_idx,
        cam_base,
        filename,
        grid_data: Optional[Dict[str, Any]] = None,
    ):
        """Save a figure showing the detected grid indices"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img[0, :, :]  # Shape (1, H, W)
                elif img.shape[-1] == 1:
                    img = img[:, :, 0]  # Shape (H, W, 1)
                elif img.shape[-1] in (3, 4):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img = np.squeeze(img)

            ax.imshow(img, cmap="gray")

            # Plot points
            ax.scatter(grid_points[:, 0], grid_points[:, 1], c='r', s=20)

            # Annotate with grid indices if available
            if grid_data is not None and 'grid_indices' in grid_data:
                grid_indices = grid_data['grid_indices']
                n_cols = grid_data.get('n_cols', 0)
                n_rows = grid_data.get('n_rows', 0)

                # Mark origin (0,0)
                origin_mask = (grid_indices[:, 0] == 0) & (grid_indices[:, 1] == 0)
                if np.any(origin_mask):
                    origin_pt = grid_points[origin_mask][0]
                    ax.scatter(origin_pt[0], origin_pt[1], c='cyan', s=100, marker='*', zorder=10)
                    ax.text(origin_pt[0] + 5, origin_pt[1], "(0,0)", color='cyan', fontsize=10)

                # Annotate every 10th point with its grid index
                for i, (pt, gi) in enumerate(zip(grid_points, grid_indices)):
                    if i % 10 == 0:
                        ax.text(pt[0], pt[1], f"{gi[0]},{gi[1]}", color='yellow', fontsize=8)

                ax.set_title(f"Detection: {filename} ({n_cols}x{n_rows} grid, {len(grid_points)} points)")
            else:
                # Legacy behavior for fixed pattern size
                ax.text(grid_points[0, 0], grid_points[0, 1], "Start (0,0)", color='cyan', fontsize=12)
                ax.text(grid_points[-1, 0], grid_points[-1, 1], "End", color='cyan', fontsize=12)
                ax.set_title(f"Detection: {filename}")

            ax.axis('off')

            out_path = cam_base / "figures" / f"detected_{img_idx:03d}.png"
            plt.savefig(out_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save visualization for {filename}: {e}")

    def run(self):
        """Run calibration using automatic grid detection (no pattern size required)."""
        logger.info("Starting Multi-View Dotboard Calibration (Automatic Detection)...")

        for cam_num in range(1, self.camera_count + 1):
            logger.info(f"--- Processing Camera {cam_num} ---")

            # Path setup: calibration_source / camera_folder (via build_calibration_camera_path)
            cam_input_dir = self._get_camera_input_dir(cam_num)
            cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "dotboard_planar"

            # Find images
            image_files = []
            is_container = self._is_container_format()

            if is_container:
                container = cam_input_dir / self.file_pattern
                if container.exists():
                    image_files = [str(container)]
            elif "%" in self.file_pattern:
                i = 1
                while True:
                    f = cam_input_dir / (self.file_pattern % i)
                    if not f.exists():
                        break
                    image_files.append(str(f))
                    i += 1
            else:
                image_files = sorted([str(f) for f in cam_input_dir.glob(self.file_pattern)])

            if not image_files:
                logger.error(f"No images found for Camera {cam_num}")
                continue

            all_objpoints = []  # 3D points in real world space
            all_imgpoints = []  # 2D points in image plane
            valid_indices_map = {}  # Store pixel values and grid data for saving
            grid_data_map = {}  # Store grid_data per image for visualization

            img_shape = None
            processed_count = 0
            detected_cols = None
            detected_rows = None

            # Loop logic adjustment for containers vs files
            loop_range = range(1, len(image_files) + 1) if not is_container else range(1, 101)

            logger.info(f"Scanning images with automatic grid detection...")

            for idx in loop_range:
                if not is_container:
                    img_path = image_files[idx - 1]
                    img_name = Path(img_path).name
                else:
                    img_path = image_files[0]
                    img_name = f"frame_{idx}"
                    try:
                        test = self._read_image(img_path, cam_num, idx)
                        if test is None:
                            break
                    except Exception:
                        break

                img = self._read_image(img_path, cam_num, idx)
                if img is None:
                    continue

                if img_shape is None:
                    img_shape = img.shape[:2][::-1]  # (width, height)

                # Use automatic detection (returns centers + grid_data)
                found, corners, grid_data = self.detect_grid(img)

                if found and grid_data is not None:
                    # Create object points dynamically based on detected grid indices
                    obj_pts = self.make_object_points_dynamic(
                        grid_data['grid_indices'],
                        grid_data['n_cols'],
                        grid_data['n_rows']
                    )

                    all_objpoints.append(obj_pts)
                    all_imgpoints.append(corners)

                    # Store for saving
                    valid_indices_map[idx] = {
                        'centers': corners,
                        'grid_indices': grid_data['grid_indices'],
                        'n_cols': grid_data['n_cols'],
                        'n_rows': grid_data['n_rows'],
                    }
                    grid_data_map[idx] = grid_data

                    # Track detected dimensions
                    if detected_cols is None:
                        detected_cols = grid_data['n_cols']
                        detected_rows = grid_data['n_rows']

                    # Visualization with grid data
                    self.save_visualization(img, corners, idx, cam_output_base, img_name, grid_data)
                    processed_count += 1
                    logger.info(f"  [+] Image {idx}: Grid detected ({grid_data['n_cols']}x{grid_data['n_rows']}, {len(corners)} points)")
                else:
                    logger.debug(f"  [-] Image {idx}: Grid not found.")

            if processed_count < 1:
                logger.error("Not enough valid images for calibration (Need >= 1).")
                continue

            # --- CALIBRATION ---
            logger.info(f"Calibrating with {processed_count} valid views...")

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_objpoints, all_imgpoints, img_shape, None, None
            )

            logger.info(f"Calibration Complete. RMS Error: {ret:.4f} pixels")

            # Apply datum frame transformation if specified
            if self.datum_frame > 0 and (self.datum_frame - 1) < len(rvecs):
                datum_idx = self.datum_frame - 1
                logger.info(f"Using frame {self.datum_frame} as datum (world origin)")
                # The datum frame's extrinsics define the world coordinate system
                # All rvecs/tvecs are already relative to each calibration plane

            # --- SAVE RESULTS ---
            detections_struct = {}
            for img_idx, data in valid_indices_map.items():
                detections_struct[f"image_{img_idx}"] = data['centers']

            model_data = {
                "camera_matrix": mtx,
                "dist_coeffs": dist,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "rms_error": ret,
                "image_width": img_shape[0],
                "image_height": img_shape[1],
                "detections_pixel_coords": detections_struct,
                "timestamp": datetime.now().isoformat(),
                "detected_cols": detected_cols,
                "detected_rows": detected_rows,
                "dot_spacing_mm": self.dot_spacing_mm,
                "datum_frame": self.datum_frame,
                # Legacy fields (set to detected values for backward compat)
                "pattern_cols": detected_cols,
                "pattern_rows": detected_rows,
            }

            out_file = cam_output_base / "model" / "dotboard_model.mat"
            savemat(str(out_file), model_data)
            logger.info(f"Saved model to: {out_file}")

            # --- SAVE INDIVIDUAL DOT CENTERS TO INDICES DIRECTORY ---
            for img_idx, data in valid_indices_map.items():
                grid_indices = data['grid_indices']

                indices_data = {
                    "centers_px": data['centers'],
                    "grid_points": data['centers'],  # Alias for compatibility
                    "grid_indices": grid_indices,
                    "grid_row": grid_indices[:, 1],  # Row index (y)
                    "grid_col": grid_indices[:, 0],  # Column index (x)
                    "pattern_cols": data['n_cols'],
                    "pattern_rows": data['n_rows'],
                    "detected_cols": data['n_cols'],
                    "detected_rows": data['n_rows'],
                    "dot_spacing_mm": self.dot_spacing_mm,
                    "frame_index": img_idx,
                }

                indices_file = cam_output_base / "indices" / f"indexing_{img_idx}.mat"
                savemat(str(indices_file), indices_data)

            logger.info(f"Saved {len(valid_indices_map)} detection files to indices directory")

    def process_single_camera(
        self,
        cam_num: int,
        progress_callback=None,
        save_visualizations: bool = False,
    ) -> dict:
        """
        Process a single camera for calibration with automatic grid detection.

        This method is designed for GUI integration where we need:
        - Progress updates during processing
        - Return value with results (instead of just saving files)
        - Optional visualization saving
        - Automatic grid detection (no pattern size required)

        Parameters
        ----------
        cam_num : int
            Camera number (1-based)
        progress_callback : callable, optional
            Function called with dict containing:
            - progress: int (0-100)
            - processed_images: int
            - valid_images: int
            - total_images: int
        save_visualizations : bool
            Whether to save detection visualization figures

        Returns
        -------
        dict
            success: bool
            camera_matrix: list (3x3)
            dist_coeffs: list
            rms_error: float
            num_images_used: int
            detected_cols: int
            detected_rows: int
            model_path: str
            error: str (if failed)
        """
        logger.info(f"--- Processing Camera {cam_num} (Automatic Detection) ---")

        # Path setup: calibration_source / camera_folder (via build_calibration_camera_path)
        cam_input_dir = self._get_camera_input_dir(cam_num)
        cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "dotboard_planar"

        # Ensure directories exist
        (cam_output_base / "indices").mkdir(parents=True, exist_ok=True)
        (cam_output_base / "model").mkdir(parents=True, exist_ok=True)
        if save_visualizations:
            (cam_output_base / "figures").mkdir(parents=True, exist_ok=True)

        # Find images
        image_files = []
        is_container = self._is_container_format()
        logger.info(f"Looking for images in {cam_input_dir} with pattern '{self.file_pattern}' (container={is_container})")

        if is_container:
            container = cam_input_dir / self.file_pattern
            if container.exists():
                image_files = [str(container)]
        elif "%" in self.file_pattern:
            i = 1
            while True:
                f = cam_input_dir / (self.file_pattern % i)
                if not f.exists():
                    break
                image_files.append(str(f))
                i += 1
        else:
            image_files = sorted([str(f) for f in cam_input_dir.glob(self.file_pattern)])

        logger.info(f"Found {len(image_files)} image files for Camera {cam_num}")
        if not image_files:
            return {"success": False, "error": f"No images found for Camera {cam_num} in {cam_input_dir}"}

        # Determine loop range
        if is_container:
            loop_range = range(1, 101)  # Arbitrary limit for containers
        else:
            loop_range = range(1, len(image_files) + 1)

        total_images = len(image_files) if not is_container else 100

        all_objpoints = []
        all_imgpoints = []
        valid_indices_map = {}

        img_shape = None
        processed_count = 0
        valid_count = 0
        detected_cols = None
        detected_rows = None

        logger.info("Scanning images with automatic grid detection...")

        for idx in loop_range:
            processed_count += 1

            # Report progress (reserve 10% for final calibration)
            if progress_callback:
                progress = int(processed_count / total_images * 90) if total_images > 0 else 0
                progress_callback({
                    "progress": min(progress, 90),
                    "processed_images": processed_count,
                    "valid_images": valid_count,
                    "total_images": total_images,
                })

            if not is_container:
                if idx > len(image_files):
                    break
                img_path = image_files[idx - 1]
                img_name = Path(img_path).name
            else:
                img_path = image_files[0]
                img_name = f"frame_{idx}"
                try:
                    test = self._read_image(img_path, cam_num, idx)
                    if test is None:
                        break
                except Exception:
                    break

            img = self._read_image(img_path, cam_num, idx)
            if img is None:
                continue

            if img_shape is None:
                img_shape = img.shape[:2][::-1]  # (width, height)
                if is_container:
                    total_images = processed_count

            # Use automatic detection
            found, corners, grid_data = self.detect_grid(img)

            if found and grid_data is not None:
                # Create object points dynamically
                obj_pts = self.make_object_points_dynamic(
                    grid_data['grid_indices'],
                    grid_data['n_cols'],
                    grid_data['n_rows']
                )

                all_objpoints.append(obj_pts)
                all_imgpoints.append(corners)
                valid_indices_map[idx] = {
                    'centers': corners,
                    'grid_indices': grid_data['grid_indices'],
                    'n_cols': grid_data['n_cols'],
                    'n_rows': grid_data['n_rows'],
                }
                valid_count += 1

                # Track detected dimensions
                if detected_cols is None:
                    detected_cols = grid_data['n_cols']
                    detected_rows = grid_data['n_rows']

                if save_visualizations:
                    self.save_visualization(img, corners, idx, cam_output_base, img_name, grid_data)

                logger.info(f"  [+] Image {idx}: Grid detected ({grid_data['n_cols']}x{grid_data['n_rows']}, {len(corners)} points)")
            else:
                logger.debug(f"  [-] Image {idx}: Grid not found.")

        if valid_count < 1:
            return {"success": False, "error": f"Only {valid_count} valid detections, need at least 1"}

        # --- CALIBRATION ---
        logger.info(f"Calibrating with {valid_count} valid views...")

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_objpoints, all_imgpoints, img_shape, None, None
            )
        except Exception as e:
            return {"success": False, "error": f"OpenCV calibration failed: {e}"}

        logger.info(f"Calibration Complete. RMS Error: {ret:.4f} pixels")

        # --- SAVE RESULTS ---
        detections_struct = {}
        for img_idx, data in valid_indices_map.items():
            detections_struct[f"image_{img_idx}"] = data['centers']

        model_data = {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "rms_error": ret,
            "reprojection_error": ret,  # Alias for compatibility
            "num_images_used": valid_count,
            "image_width": img_shape[0],
            "image_height": img_shape[1],
            "detections_pixel_coords": detections_struct,
            "timestamp": datetime.now().isoformat(),
            "detected_cols": detected_cols,
            "detected_rows": detected_rows,
            "dot_spacing_mm": self.dot_spacing_mm,
            "datum_frame": self.datum_frame,
            # Legacy fields for backward compat
            "pattern_cols": detected_cols,
            "pattern_rows": detected_rows,
        }

        out_file = cam_output_base / "model" / "dotboard_model.mat"
        savemat(str(out_file), model_data)
        logger.info(f"Saved model to: {out_file}")

        # Save per-image detection files
        for img_idx, data in valid_indices_map.items():
            grid_indices = data['grid_indices']
            indices_data = {
                "grid_points": data['centers'],
                "centers_px": data['centers'],
                "grid_indices": grid_indices,
                "grid_row": grid_indices[:, 1],
                "grid_col": grid_indices[:, 0],
                "pattern_cols": data['n_cols'],
                "pattern_rows": data['n_rows'],
                "detected_cols": data['n_cols'],
                "detected_rows": data['n_rows'],
                "dot_spacing_mm": self.dot_spacing_mm,
                "frame_index": img_idx,
            }
            indices_file = cam_output_base / "indices" / f"indexing_{img_idx}.mat"
            savemat(str(indices_file), indices_data)

        logger.info(f"Saved {len(valid_indices_map)} detection files to indices directory")

        # Final progress
        if progress_callback:
            progress_callback({
                "progress": 100,
                "processed_images": processed_count,
                "valid_images": valid_count,
                "total_images": processed_count,
            })

        return {
            "success": True,
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.flatten().tolist(),
            "rms_error": float(ret),
            "num_images_used": valid_count,
            "detected_cols": detected_cols,
            "detected_rows": detected_rows,
            "model_path": str(out_file),
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Planar Calibration - Starting (Automatic Grid Detection)")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        config = get_config()

        # Extract settings from config
        source_dir = config.data["paths"]["source_paths"][0]
        base_dir = config.data["paths"]["base_paths"][0]
        camera_nums = config.data["paths"].get("camera_numbers", [1])
        file_pattern = config.data["calibration"]["image_format"]
        # pattern_cols and pattern_rows are no longer required - automatic detection
        dot_spacing_mm = config.data["calibration"]["dotboard"]["dot_spacing_mm"]
        enhance_dots = config.data["calibration"]["dotboard"].get("enhance_dots", False)
        datum_frame = config.data["calibration"]["dotboard"].get("datum_frame", 1)
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

        # Use hardcoded settings
        source_dir = SOURCE_DIR
        base_dir = BASE_DIR
        camera_nums = CAMERA_NUMS
        file_pattern = FILE_PATTERN
        dot_spacing_mm = DOT_SPACING_MM
        enhance_dots = ENHANCE_DOTS
        datum_frame = 1  # Default

    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {base_dir}")
    logger.info(f"Cameras: {camera_nums}")
    logger.info(f"Dot spacing: {dot_spacing_mm}mm (grid size auto-detected)")
    logger.info(f"Datum frame: {datum_frame}")

    failed_cameras = []

    for camera_num in camera_nums:
        logger.info(f"Processing Camera {camera_num}...")
        try:
            # Create calibrator - no pattern_cols/pattern_rows needed
            calibrator = MultiViewCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_count=1,  # Process one at a time
                file_pattern=file_pattern,
                dot_spacing_mm=dot_spacing_mm,
                enhance_dots=enhance_dots,
                config=config,
                datum_frame=datum_frame,
            )
            result = calibrator.process_single_camera(camera_num, save_visualizations=True)
            if result.get("success"):
                detected_info = f"{result.get('detected_cols', '?')}x{result.get('detected_rows', '?')} grid"
                logger.info(f"Camera {camera_num} completed: RMS={result['rms_error']:.4f} px, {result['num_images_used']} images, {detected_info}")
            else:
                logger.error(f"Camera {camera_num} failed: {result.get('error', 'Unknown error')}")
                failed_cameras.append(camera_num)
        except Exception as e:
            logger.error(f"Camera {camera_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_cameras.append(camera_num)

    logger.info("=" * 60)
    if failed_cameras:
        logger.error(f"Calibration failed for cameras: {failed_cameras}")
    else:
        logger.info("Planar calibration completed successfully for all cameras")