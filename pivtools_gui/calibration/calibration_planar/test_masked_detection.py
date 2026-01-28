#!/usr/bin/env python3
"""
Test script for masked dotboard detection with interactive mask drawing.

This script allows:
1. Interactive mask drawing using mouse
2. Loading a pre-saved mask image
3. Testing detection with/without mask
4. Visualizing raw blob detections to diagnose issues

Usage:
    python test_masked_detection.py

Controls for interactive mask drawing:
    - Left click + drag: Draw mask region (exclude area)
    - Right click: Undo last region
    - 's': Save mask and run detection
    - 'c': Clear all mask regions
    - 'q': Quit without saving
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pivtools_core.config import get_config
from pivtools_core.image_handling.load_images import read_image
from pivtools_core.image_handling.path_utils import build_calibration_camera_path


# ==========================================================================
# CONFIGURATION - Modify these parameters for your test
# ==========================================================================

# Set to True for interactive mask drawing, False to use EXCLUDE_REGIONS below
INTERACTIVE_MASK = True

# If not using interactive mode, define rectangular exclusion regions:
EXCLUDE_REGIONS = [
    {'bottom': 0.30},  # Exclude bottom 30%
]

# Or load a pre-saved mask file (set to None to create new mask)
LOAD_MASK_FILE = None  # e.g., "detection_mask.png"

# Use CALIB_CB_CLUSTERING for more robust detection (recommended)
USE_CLUSTERING = True

# Use automatic grid detection (no pattern size needed!)
USE_AUTOMATIC_DETECTION = True

# ==========================================================================


def create_blob_detector(min_area=200, max_area=5000):
    """Create blob detector for circle grid detection."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 5
    return cv2.SimpleBlobDetector_create(params)


def read_calibration_image(img_path, camera=1, img_index=1):
    """Read and normalize calibration image to uint8."""
    img = read_image(str(img_path), camera_no=camera, frames=1, frames_per_camera=1)

    if img is None:
        return None

    print(f"Read image shape: {img.shape}, dtype: {img.dtype}")

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


def to_grayscale_2d(img):
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


def apply_mask_to_image(img, mask, fill_value=255):
    """Apply mask: fill excluded regions (mask=0) with fill_value."""
    masked_img = img.copy()
    masked_img[mask == 0] = fill_value
    return masked_img


def detect_raw_blobs(img, detector, mask=None):
    """
    Detect all blobs in image (without grid fitting).

    This shows what the blob detector finds, regardless of grid pattern.
    Useful for diagnosing why grid detection fails.
    """
    gray = to_grayscale_2d(img)

    if mask is not None:
        gray = apply_mask_to_image(gray, mask, fill_value=255)

    # Detect on both original and inverted
    keypoints_orig = detector.detect(gray)
    keypoints_inv = detector.detect(255 - gray)

    return {
        'original': keypoints_orig,
        'inverted': keypoints_inv,
        'gray': gray,
    }


def detect_grid_with_mask(img, pattern_size, detector, mask=None, asymmetric=False, use_clustering=True):
    """
    Detect circle grid with optional mask using OpenCV's findCirclesGrid.
    Requires exact pattern size - use detect_grid_automatic for flexible detection.
    """
    gray = to_grayscale_2d(img)
    original_gray = gray.copy()

    if mask is not None:
        gray = apply_mask_to_image(gray, mask, fill_value=255)

    flags = cv2.CALIB_CB_ASYMMETRIC_GRID if asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID
    if use_clustering:
        flags |= cv2.CALIB_CB_CLUSTERING

    flag_desc = f"{'ASYMMETRIC' if asymmetric else 'SYMMETRIC'}"
    if use_clustering:
        flag_desc += " + CLUSTERING"

    print(f"  Detection flags: {flag_desc}")

    for test_img, label in [(gray, "Original"), (255 - gray, "Inverted")]:
        print(f"  Trying {label} image...")

        found, centers = cv2.findCirclesGrid(
            test_img, pattern_size, flags=flags, blobDetector=detector
        )

        if found:
            print(f"    -> SUCCESS on {label}")
            centers = centers.reshape(-1, 2).astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            centers_refined = cv2.cornerSubPix(
                original_gray,
                centers.reshape(-1, 1, 2),
                (11, 11),
                (-1, -1),
                criteria
            )

            return True, centers_refined.reshape(-1, 2).astype(np.float32), f"{label} ({flag_desc})"
        else:
            print(f"    -> Failed on {label}")

    return False, None, "Failed"


def detect_grid_automatic(img, detector, mask=None, grid_spacing_mm=None):
    """
    Automatically detect grid from blob positions - NO pattern size needed.

    Uses OpenCV primitives:
    - SimpleBlobDetector for blob detection
    - Neighbor analysis to find grid vectors
    - Direct grid coordinate computation (more robust than kmeans)

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
    grid_data : dict
    info : dict
    """
    gray = to_grayscale_2d(img)
    original_gray = gray.copy()

    if mask is not None:
        gray = apply_mask_to_image(gray, mask, fill_value=255)

    info = {'method': 'automatic_grid_detection_v2'}

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
    print(f"  Detected {len(centers)} blobs")

    # Step 2: Find grid spacing
    n_points = len(centers)

    # Compute all pairwise distances
    all_distances = np.zeros((n_points, n_points))
    for i in range(n_points):
        all_distances[i] = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
        all_distances[i, i] = np.inf

    # Find nearest neighbor distance for each point
    nn_distances = np.min(all_distances, axis=1)
    spacing_px = np.median(nn_distances)
    info['spacing_px'] = float(spacing_px)
    print(f"  Estimated grid spacing: {spacing_px:.1f} pixels")

    # Step 3: Find HORIZONTAL and VERTICAL grid vectors separately
    # For axis-aligned grids, neighbors should be directly left/right or up/down
    horizontal_vecs = []  # Neighbors mostly horizontal (|dy| < |dx|)
    vertical_vecs = []    # Neighbors mostly vertical (|dx| < |dy|)

    angle_tolerance_deg = 20  # Max deviation from axis in degrees
    angle_tolerance = np.radians(angle_tolerance_deg)

    for i in range(n_points):
        for j in range(n_points):
            dist = all_distances[i, j]
            if i != j and dist < spacing_px * 1.4 and dist > spacing_px * 0.6:
                vec = centers[j] - centers[i]
                angle_from_horiz = np.arctan2(abs(vec[1]), abs(vec[0]))

                if angle_from_horiz < angle_tolerance:
                    # Nearly horizontal
                    horizontal_vecs.append(vec)
                elif angle_from_horiz > (np.pi/2 - angle_tolerance):
                    # Nearly vertical
                    vertical_vecs.append(vec)

    print(f"  Found {len(horizontal_vecs)} horizontal, {len(vertical_vecs)} vertical neighbor pairs")

    if len(horizontal_vecs) < 10 or len(vertical_vecs) < 10:
        info['error'] = f'Not enough axis-aligned neighbors found'
        return False, None, info

    horizontal_vecs = np.array(horizontal_vecs)
    vertical_vecs = np.array(vertical_vecs)

    # Normalize horizontal vectors to point RIGHT (+x)
    for i in range(len(horizontal_vecs)):
        if horizontal_vecs[i, 0] < 0:
            horizontal_vecs[i] = -horizontal_vecs[i]

    # Normalize vertical vectors to point UP (-y in image pixel coords)
    for i in range(len(vertical_vecs)):
        if vertical_vecs[i, 1] > 0:  # In image coords, +y is down
            vertical_vecs[i] = -vertical_vecs[i]

    # Take median to get robust grid vectors
    vec1 = np.median(horizontal_vecs, axis=0)  # X direction (right)
    vec2 = np.median(vertical_vecs, axis=0)    # Y direction (up in Cartesian = -y in image)

    info['grid_vec1'] = vec1.tolist()
    info['grid_vec2'] = vec2.tolist()
    print(f"  Grid vector 1 (col): [{vec1[0]:.1f}, {vec1[1]:.1f}]")
    print(f"  Grid vector 2 (row): [{vec2[0]:.1f}, {vec2[1]:.1f}]")

    # Step 4: Compute grid coordinates for each point
    # Solve: point = origin + col * vec1 + row * vec2
    # Use BOTTOM-LEFT point as origin (min x, max y in image pixel coords)
    # This gives Cartesian: (0,0) at bottom-left, x right, y up
    origin_idx = np.argmin(centers[:, 0] - centers[:, 1])  # Bottom-left
    origin = centers[origin_idx]

    print(f"  Origin (bottom-left): ({origin[0]:.1f}, {origin[1]:.1f})")

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

    print(f"  Grid index range: x=[0, {grid_indices[:, 0].max()}], y=[0, {grid_indices[:, 1].max()}]")

    # Step 5: Use RANSAC to robustly fit affine transform and reject outliers
    # Source points: grid indices (as float)
    # Destination points: actual pixel coordinates
    src_pts = grid_indices.astype(np.float32)
    dst_pts = centers.astype(np.float32)

    # cv2.estimateAffine2D with RANSAC
    # Returns: transform matrix (2x3), inlier mask
    ransac_thresh = 0.3 * spacing_px  # Max reprojection error
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
    print(f"  RANSAC: {n_inliers} inliers, {n_outliers} outliers rejected")

    centers = centers[inliers]
    grid_indices = grid_indices[inliers]

    # Step 6: Remove duplicate grid positions (keep best fit)
    from collections import defaultdict

    # Compute residuals for remaining points
    src_clean = grid_indices.astype(np.float32)
    predicted = cv2.transform(src_clean.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)
    residuals = np.sqrt(np.sum((centers - predicted) ** 2, axis=1))

    pos_to_points = defaultdict(list)
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
        print(f"  Removed {n_dups} duplicate grid positions")

    keep_indices = np.array(keep_indices)
    centers = centers[keep_indices]
    grid_indices = grid_indices[keep_indices]

    # Recompute dimensions
    n_cols = grid_indices[:, 0].max() + 1
    n_rows = grid_indices[:, 1].max() + 1
    info['n_cols'] = int(n_cols)
    info['n_rows'] = int(n_rows)

    # Extract rotation angle from affine matrix
    angle_deg = np.degrees(np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]))
    info['angle_deg'] = float(angle_deg)
    info['affine_matrix'] = affine_matrix.tolist()

    print(f"  Final grid: {n_cols} cols x {n_rows} rows, {len(centers)} points")

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
    print(f"  SUCCESS: {n_cols}x{n_rows} grid with {len(centers)} points")

    return True, grid_data, info


def create_rectangular_mask(shape, exclude_regions):
    """Create mask excluding rectangular regions."""
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    for region in exclude_regions:
        top = int(region.get('top', 0) * h)
        bottom = int(region.get('bottom', 0) * h)
        left = int(region.get('left', 0) * w)
        right = int(region.get('right', 0) * w)

        if top > 0:
            mask[:top, :] = 0
        if bottom > 0:
            mask[h - bottom:, :] = 0
        if left > 0:
            mask[:, :left] = 0
        if right > 0:
            mask[:, w - right:] = 0

    return mask


class InteractiveMaskDrawer:
    """
    Interactive mask drawing using OpenCV.

    Controls:
        - Left click + drag: Draw rectangle to exclude
        - Right click: Undo last rectangle
        - 's': Save and continue
        - 'c': Clear all
        - 'q': Quit
    """

    def __init__(self, img):
        self.original = img.copy()
        self.display = img.copy()
        self.mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        self.rectangles = []
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.done = False
        self.cancelled = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                # Store rectangle (ensure proper order)
                x1 = min(self.start_point[0], end_point[0])
                y1 = min(self.start_point[1], end_point[1])
                x2 = max(self.start_point[0], end_point[0])
                y2 = max(self.start_point[1], end_point[1])
                if x2 - x1 > 5 and y2 - y1 > 5:  # Minimum size
                    self.rectangles.append((x1, y1, x2, y2))
                self.update_mask()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo last rectangle
            if self.rectangles:
                self.rectangles.pop()
                self.update_mask()

    def update_mask(self):
        """Rebuild mask from rectangles."""
        self.mask = np.ones(self.original.shape[:2], dtype=np.uint8) * 255
        for (x1, y1, x2, y2) in self.rectangles:
            self.mask[y1:y2, x1:x2] = 0

    def draw(self):
        """Draw current state."""
        # Create display with mask overlay
        self.display = self.original.copy()

        # Convert to BGR if grayscale for colored overlay
        if len(self.display.shape) == 2:
            self.display = cv2.cvtColor(self.display, cv2.COLOR_GRAY2BGR)

        # Draw existing rectangles (semi-transparent red)
        overlay = self.display.copy()
        for (x1, y1, x2, y2) in self.rectangles:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, self.display, 0.7, 0, self.display)

        # Draw rectangle outlines
        for (x1, y1, x2, y2) in self.rectangles:
            cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw current rectangle being drawn
        if self.drawing and self.start_point and self.current_point:
            cv2.rectangle(self.display, self.start_point, self.current_point, (0, 255, 0), 2)

        # Add instructions
        cv2.putText(self.display, "Left-drag: exclude region | Right-click: undo | 's': save | 'c': clear | 'q': quit",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(self.display, f"Excluded regions: {len(self.rectangles)}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def run(self):
        """Run interactive mask drawing."""
        window_name = "Draw Mask - Exclude regions (red)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("INTERACTIVE MASK DRAWING")
        print("=" * 60)
        print("  Left-drag:   Draw rectangle to EXCLUDE from detection")
        print("  Right-click: Undo last rectangle")
        print("  's':         Save mask and run detection")
        print("  'c':         Clear all rectangles")
        print("  'q':         Quit without saving")
        print("=" * 60 + "\n")

        while not self.done:
            self.draw()
            cv2.imshow(window_name, self.display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                self.done = True
            elif key == ord('c'):
                self.rectangles = []
                self.update_mask()
            elif key == ord('q'):
                self.done = True
                self.cancelled = True

        cv2.destroyAllWindows()

        if self.cancelled:
            return None
        return self.mask


def visualize_results(img, mask, centers, pattern_size, blob_info, output_path=None,
                      use_clustering=True, grid_data=None):
    """Create comprehensive visualization of detection results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    gray = to_grayscale_2d(img)

    # Handle automatic detection vs fixed pattern
    if grid_data is not None:
        expected_dots = None  # Unknown for automatic
        detected_cols = grid_data['n_cols']
        detected_rows = grid_data['n_rows']
        auto_mode = True
    else:
        expected_dots = pattern_size[0] * pattern_size[1]
        detected_cols = pattern_size[0]
        detected_rows = pattern_size[1]
        auto_mode = False

    # 1. Original image
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 2. Mask overlay on image
    mask_overlay = np.stack([gray, gray, gray], axis=-1)
    mask_overlay[mask == 0] = [255, 100, 100]  # Red tint for excluded areas
    axes[0, 1].imshow(mask_overlay)
    axes[0, 1].set_title('Mask Overlay (red=excluded)')
    axes[0, 1].axis('off')

    # 3. Masked image (what detection sees)
    masked_img = apply_mask_to_image(gray, mask, fill_value=255)
    axes[0, 2].imshow(masked_img, cmap='gray')
    axes[0, 2].set_title('Masked Image (input to detector)')
    axes[0, 2].axis('off')

    # 4. Raw blob detections on MASKED image (IMPORTANT for debugging)
    axes[1, 0].imshow(masked_img, cmap='gray')
    kp_orig = blob_info['original']
    kp_inv = blob_info['inverted']

    # Plot blobs - use the one with more detections
    if len(kp_inv) > len(kp_orig):
        kp_best = kp_inv
        kp_label = "inverted"
    else:
        kp_best = kp_orig
        kp_label = "original"

    for kp in kp_best:
        circle = plt.Circle((kp.pt[0], kp.pt[1]), kp.size/2,
                            fill=False, color='lime', linewidth=1)
        axes[1, 0].add_patch(circle)
        axes[1, 0].plot(kp.pt[0], kp.pt[1], 'r.', markersize=2)

    if expected_dots:
        axes[1, 0].set_title(f'Blob Detections ({kp_label})\n'
                             f'Found: {len(kp_best)} | Expected: {expected_dots}')
    else:
        axes[1, 0].set_title(f'Blob Detections ({kp_label})\n'
                             f'Found: {len(kp_best)} blobs')
    axes[1, 0].axis('off')

    # 5. Grid detection result
    axes[1, 1].imshow(gray, cmap='gray')
    if centers is not None:
        # Color points by row for automatic detection
        if grid_data is not None and 'grid_indices' in grid_data:
            grid_indices = grid_data['grid_indices']
            row_colors = grid_indices[:, 1]  # Color by y-index

            # Draw grid lines connecting adjacent points
            # Build a lookup from grid index to center position
            idx_to_center = {(gi[0], gi[1]): centers[i] for i, gi in enumerate(grid_indices)}

            # Draw horizontal lines (same y, adjacent x)
            for (gx, gy), pt in idx_to_center.items():
                # Connect to right neighbor
                if (gx + 1, gy) in idx_to_center:
                    neighbor = idx_to_center[(gx + 1, gy)]
                    axes[1, 1].plot([pt[0], neighbor[0]], [pt[1], neighbor[1]],
                                   'c-', linewidth=0.5, alpha=0.5)
                # Connect to upper neighbor (in Cartesian: y+1 is up, but in image y+1 is visually up)
                if (gx, gy + 1) in idx_to_center:
                    neighbor = idx_to_center[(gx, gy + 1)]
                    axes[1, 1].plot([pt[0], neighbor[0]], [pt[1], neighbor[1]],
                                   'm-', linewidth=0.5, alpha=0.5)

            # Plot points on top of lines
            scatter = axes[1, 1].scatter(centers[:, 0], centers[:, 1], c=row_colors,
                                        cmap='rainbow', s=25, marker='o', edgecolors='white', linewidths=0.5)

            # Mark origin (0,0) clearly
            origin_mask = (grid_indices[:, 0] == 0) & (grid_indices[:, 1] == 0)
            if np.any(origin_mask):
                origin_pt = centers[origin_mask][0]
                axes[1, 1].scatter(origin_pt[0], origin_pt[1], c='white', s=150, marker='*',
                                  edgecolors='black', linewidths=1, zorder=10, label='Origin (0,0)')

            # Mark max corner
            max_x, max_y = grid_indices[:, 0].max(), grid_indices[:, 1].max()
            max_mask = (grid_indices[:, 0] == max_x) & (grid_indices[:, 1] == max_y)
            if np.any(max_mask):
                max_pt = centers[max_mask][0]
                axes[1, 1].scatter(max_pt[0], max_pt[1], c='yellow', s=150, marker='*',
                                  edgecolors='black', linewidths=1, zorder=10, label=f'({max_x},{max_y})')

            # Label corner points and some others
            corners = [(0, 0), (max_x, 0), (0, max_y), (max_x, max_y)]
            for gx, gy in corners:
                if (gx, gy) in idx_to_center:
                    pt = idx_to_center[(gx, gy)]
                    axes[1, 1].annotate(f'({gx},{gy})', (pt[0], pt[1]),
                                       xytext=(8, 8), textcoords='offset points',
                                       fontsize=8, fontweight='bold', color='white',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

            axes[1, 1].legend(loc='upper right', fontsize=8)
            title = f'AUTOMATIC DETECTION: SUCCESS\n{detected_cols}x{detected_rows} grid, {len(centers)} points\nCyan=x-lines, Magenta=y-lines'
        else:
            axes[1, 1].scatter(centers[:, 0], centers[:, 1], c='lime', s=15, marker='o',
                             edgecolors='darkgreen', linewidths=0.5)
            axes[1, 1].scatter(centers[0, 0], centers[0, 1], c='cyan', s=100, marker='x',
                             linewidths=2, label='Start (0,0)')
            axes[1, 1].scatter(centers[-1, 0], centers[-1, 1], c='yellow', s=100, marker='x',
                             linewidths=2, label='End')

            cols = pattern_size[0]
            for i in range(0, len(centers), max(1, len(centers) // 15)):
                r, c = divmod(i, cols)
                axes[1, 1].text(centers[i, 0] + 5, centers[i, 1], f'{r},{c}', color='cyan', fontsize=7)

            title = f'GRID DETECTION: SUCCESS\n{len(centers)} points (expected {expected_dots})'
            axes[1, 1].legend(loc='upper right')

        axes[1, 1].set_title(title)
    else:
        if auto_mode:
            axes[1, 1].set_title('AUTOMATIC DETECTION: FAILED\nCould not fit grid to blobs')
        else:
            axes[1, 1].set_title(f'GRID DETECTION: FAILED\nExpected {expected_dots} dots')
    axes[1, 1].axis('off')

    # 6. Diagnosis / Help
    axes[1, 2].axis('off')

    diagnosis_text = "DIAGNOSIS:\n\n"

    if auto_mode:
        mode_note = "AUTOMATIC mode"
    else:
        mode_note = "CLUSTERING" if use_clustering else "standard"

    if centers is not None:
        diagnosis_text += f"SUCCESS! ({mode_note})\n\n"

        if grid_data is not None:
            diagnosis_text += f"Detected grid:\n"
            diagnosis_text += f"  {grid_data['n_cols']} cols x {grid_data['n_rows']} rows\n"
            diagnosis_text += f"  {len(centers)} points total\n"
            diagnosis_text += f"  Spacing: {grid_data['spacing_px']:.1f} px\n"
            diagnosis_text += f"  Rotation: {grid_data['angle_deg']:.1f} deg\n\n"
            if grid_data.get('grid_spacing_mm'):
                diagnosis_text += f"Physical spacing: {grid_data['grid_spacing_mm']} mm\n"
        else:
            diagnosis_text += f"Found {len(centers)} dots in\n"
            diagnosis_text += f"{detected_cols}x{detected_rows} pattern.\n\n"

        diagnosis_text += "\nGrid indices shown as col,row\n"
        diagnosis_text += "Colors indicate row number."
    else:
        # Analyze why it failed
        best_blob_count = len(kp_best)
        diagnosis_text += f"FAILED ({mode_note})\n\n"
        diagnosis_text += f"Blobs found: {best_blob_count}\n\n"

        if auto_mode:
            if best_blob_count < 9:
                diagnosis_text += "ISSUE: Too few blobs (<9)\n"
                diagnosis_text += "• Need at least 3x3 grid\n"
                diagnosis_text += "• Adjust mask or blob params\n"
            else:
                diagnosis_text += "ISSUE: Could not fit grid\n"
                diagnosis_text += "• Blobs may not form regular grid\n"
                diagnosis_text += "• Too many outlier detections\n"
                diagnosis_text += "• Try adjusting mask\n"
        else:
            if expected_dots:
                diagnosis_text += f"Expected: {expected_dots} dots\n"
                diagnosis_text += f"Difference: {best_blob_count - expected_dots:+d}\n\n"

            diagnosis_text += "• Try AUTOMATIC mode instead\n"
            diagnosis_text += "• Or adjust pattern size\n"

    axes[1, 2].text(0.05, 0.95, diagnosis_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    plt.show()


def save_mask(mask, output_path):
    """Save mask as PNG image."""
    cv2.imwrite(str(output_path), mask)
    print(f"Saved mask to: {output_path}")


def load_mask(mask_path):
    """Load mask from PNG image."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask from {mask_path}")
    # Ensure binary
    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def main():
    """Main test function."""
    print("=" * 60)
    print("Masked Dotboard Detection Test")
    print("=" * 60)

    # Load config
    config = get_config()
    dotboard_cfg = config.data.get("calibration", {}).get("dotboard", {})

    CAMERA = dotboard_cfg.get("camera", 1)
    PATTERN_COLS = dotboard_cfg.get("pattern_cols", 30)
    PATTERN_ROWS = dotboard_cfg.get("pattern_rows", 20)
    ASYMMETRIC = dotboard_cfg.get("asymmetric", False)
    IMAGE_INDEX = dotboard_cfg.get("image_index", 0)
    SOURCE_PATH_IDX = dotboard_cfg.get("source_path_idx", 0)
    DOT_SPACING_MM = dotboard_cfg.get("dot_spacing_mm", 10.0)

    print(f"Camera: {CAMERA}")
    print(f"Pattern (for OpenCV mode): {PATTERN_COLS} x {PATTERN_ROWS} = {PATTERN_COLS * PATTERN_ROWS} dots")
    print(f"Dot spacing: {DOT_SPACING_MM} mm")
    print(f"Automatic detection: {USE_AUTOMATIC_DETECTION}")

    # Get image path
    cam_input_dir = build_calibration_camera_path(config, SOURCE_PATH_IDX, CAMERA)
    print(f"Image directory: {cam_input_dir}")

    image_format = config.calibration_image_format
    print(f"Image format: {image_format}")

    if "%" in image_format:
        img_num = IMAGE_INDEX + 1 if not config.data["calibration"].get("zero_based_indexing", False) else IMAGE_INDEX
        img_path = cam_input_dir / (image_format % img_num)
    else:
        files = sorted(cam_input_dir.glob(image_format))
        if IMAGE_INDEX < len(files):
            img_path = files[IMAGE_INDEX]
        else:
            print(f"ERROR: Image index {IMAGE_INDEX} out of range (found {len(files)} files)")
            return

    print(f"Loading: {img_path}")

    if not img_path.exists():
        print(f"ERROR: Image not found at {img_path}")
        return

    # Read image
    img = read_calibration_image(img_path, CAMERA, IMAGE_INDEX + 1)
    if img is None:
        print("ERROR: Failed to read image")
        return

    gray = to_grayscale_2d(img)
    print(f"Image shape: {gray.shape}")

    # Setup output directory
    output_dir = Path(config.base_paths[SOURCE_PATH_IDX]) / "calibration" / f"Cam{CAMERA}" / "dotboard_planar" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get or create mask
    if LOAD_MASK_FILE:
        mask_path = output_dir / LOAD_MASK_FILE
        print(f"\nLoading mask from: {mask_path}")
        mask = load_mask(mask_path)
    elif INTERACTIVE_MASK:
        print("\nStarting interactive mask drawing...")
        drawer = InteractiveMaskDrawer(gray)
        mask = drawer.run()

        if mask is None:
            print("Mask drawing cancelled.")
            return

        # Save the mask for future use
        save_mask(mask, output_dir / "detection_mask.png")
    else:
        print(f"\nUsing rectangular exclusion: {EXCLUDE_REGIONS}")
        mask = create_rectangular_mask(gray.shape, EXCLUDE_REGIONS)

    # Create detector
    detector = create_blob_detector()
    pattern_size = (PATTERN_COLS, PATTERN_ROWS)

    # Get raw blob detections (for diagnosis)
    print("\n" + "=" * 60)
    print("STEP 1: Analyzing raw blob detections (before grid fitting)")
    print("=" * 60)
    blob_info = detect_raw_blobs(img, detector, mask=mask)
    n_orig = len(blob_info['original'])
    n_inv = len(blob_info['inverted'])
    n_expected = PATTERN_COLS * PATTERN_ROWS

    print(f"  Blobs on original (dark dots on light bg): {n_orig}")
    print(f"  Blobs on inverted (light dots on dark bg): {n_inv}")
    print(f"  Expected dots for {PATTERN_COLS}x{PATTERN_ROWS} grid: {n_expected}")

    # Diagnosis
    best_count = max(n_orig, n_inv)
    if best_count < n_expected:
        print(f"\n  WARNING: Found fewer blobs ({best_count}) than expected ({n_expected})")
        print(f"           Missing ~{n_expected - best_count} dots")
        print(f"           With CLUSTERING, this may still work if pattern is recognizable")
    elif best_count > n_expected * 1.1:
        print(f"\n  WARNING: Found more blobs ({best_count}) than expected ({n_expected})")
        print(f"           Extra ~{best_count - n_expected} false detections (noise/reflection)")
        print(f"           Consider expanding mask to exclude more of the reflection")

    # Test grid detection
    print("\n" + "=" * 60)
    print("STEP 2: Grid detection")
    print("=" * 60)

    grid_data = None  # For automatic detection results

    if USE_AUTOMATIC_DETECTION:
        print("  Mode: AUTOMATIC (no pattern size required)")
        print(f"  Grid spacing: {DOT_SPACING_MM} mm")

        found, grid_data, auto_info = detect_grid_automatic(
            img, detector, mask=mask, grid_spacing_mm=DOT_SPACING_MM
        )

        if found:
            centers = grid_data['centers']
            method = f"Automatic ({grid_data['n_cols']}x{grid_data['n_rows']} detected)"
        else:
            centers = None
            method = "Automatic (failed)"
            print(f"  Error: {auto_info.get('error', 'Unknown')}")
    else:
        print(f"  Mode: OpenCV findCirclesGrid")
        print(f"  Pattern size: {PATTERN_COLS} cols x {PATTERN_ROWS} rows")
        print(f"  Using clustering: {USE_CLUSTERING}")

        found, centers, method = detect_grid_with_mask(
            img, pattern_size, detector, mask=mask, asymmetric=ASYMMETRIC,
            use_clustering=USE_CLUSTERING
        )

    print("\n" + "-" * 60)
    if found:
        print(f"RESULT: SUCCESS!")
        print(f"  Method: {method}")
        print(f"  Found {len(centers)} grid points")

        # Show grid extent
        x_min, y_min = centers.min(axis=0)
        x_max, y_max = centers.max(axis=0)
        print(f"  Grid extent: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")

        if grid_data:
            print(f"  Detected: {grid_data['n_cols']} cols x {grid_data['n_rows']} rows")
            print(f"  Spacing: {grid_data['spacing_px']:.1f} px")
            print(f"  Rotation: {grid_data['angle_deg']:.2f} deg")
    else:
        print("RESULT: FAILED - Grid not detected")
        print("\nPossible issues:")
        print("  1. Mask may be excluding too many dots")
        print("  2. Too many false blobs from reflection")
        print("  3. Dots may be too small/large for blob detector")

    # Visualize results
    print("\n" + "=" * 60)
    print("STEP 3: Generating visualization")
    print("=" * 60)
    visualize_results(
        img, mask, centers, pattern_size, blob_info,
        output_path=output_dir / "masked_detection_result.png",
        use_clustering=USE_CLUSTERING,
        grid_data=grid_data
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if found:
        print("Detection: SUCCESS")
        print(f"  Grid points found: {len(centers)}")
        if grid_data:
            print(f"  Grid size: {grid_data['n_cols']} cols x {grid_data['n_rows']} rows")
            print(f"  Pixel spacing: {grid_data['spacing_px']:.1f} px")
            if grid_data.get('grid_spacing_mm'):
                mm_per_px = grid_data['grid_spacing_mm'] / grid_data['spacing_px']
                print(f"  Scale: {mm_per_px:.4f} mm/px ({1/mm_per_px:.2f} px/mm)")

        # Generate pinhole camera model
        print("\n" + "-" * 60)
        print("PINHOLE CAMERA MODEL")
        print("-" * 60)

        if grid_data and grid_data.get('grid_spacing_mm'):
            # Build object points (3D world coordinates in mm)
            # z=0 for planar calibration
            obj_points = []
            img_points = []

            for i, (center, idx) in enumerate(zip(grid_data['centers'], grid_data['grid_indices'])):
                x_mm = idx[0] * grid_data['grid_spacing_mm']
                y_mm = idx[1] * grid_data['grid_spacing_mm']
                obj_points.append([x_mm, y_mm, 0.0])
                img_points.append(center)

            obj_points = np.array([obj_points], dtype=np.float32)  # Shape: (1, N, 3)
            img_points = np.array([img_points], dtype=np.float32)  # Shape: (1, N, 2)

            # Get image size
            img_size = (gray.shape[1], gray.shape[0])  # (width, height)

            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, img_size, None, None
            )

            print(f"  RMS Reprojection Error: {ret:.4f} pixels")
            print()
            print("  Camera Matrix (K):")
            print(f"    fx = {camera_matrix[0, 0]:.2f} px")
            print(f"    fy = {camera_matrix[1, 1]:.2f} px")
            print(f"    cx = {camera_matrix[0, 2]:.2f} px (principal point x)")
            print(f"    cy = {camera_matrix[1, 2]:.2f} px (principal point y)")
            print()
            print("  Distortion Coefficients:")
            print(f"    k1 = {dist_coeffs[0, 0]:.6f}")
            print(f"    k2 = {dist_coeffs[0, 1]:.6f}")
            print(f"    p1 = {dist_coeffs[0, 2]:.6f} (tangential)")
            print(f"    p2 = {dist_coeffs[0, 3]:.6f} (tangential)")
            print(f"    k3 = {dist_coeffs[0, 4]:.6f}")
            print()

            # Compute scale factor
            focal_length_px = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
            mm_per_px = grid_data['grid_spacing_mm'] / grid_data['spacing_px']
            print(f"  Scale Factor:")
            print(f"    {mm_per_px:.6f} mm/px")
            print(f"    {1/mm_per_px:.2f} px/mm")
            print()

            # Store in grid_data for later use
            grid_data['camera_matrix'] = camera_matrix
            grid_data['dist_coeffs'] = dist_coeffs
            grid_data['rms_error'] = ret

        else:
            print("  Cannot compute - grid_spacing_mm not set")

        print("-" * 60)
        print(f"\n  Mask saved to: {output_dir / 'detection_mask.png'}")
        print(f"  Result figure: {output_dir / 'masked_detection_result.png'}")
    else:
        print("Detection: FAILED")
        print("\nNext steps to try:")
        print("  1. Adjust mask to exclude more/less of the reflection")
        if not USE_AUTOMATIC_DETECTION:
            print("  2. Enable USE_AUTOMATIC_DETECTION = True")
            print(f"     (Currently using fixed {PATTERN_COLS}x{PATTERN_ROWS} pattern)")
        else:
            print("  2. Check that blobs form a regular grid pattern")
            print("  3. Adjust blob detector min/max area")
        print(f"\n  Result figure: {output_dir / 'masked_detection_result.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
