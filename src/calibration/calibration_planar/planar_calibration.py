"""
Planar calibration tool for PIV images.
This module provides functions for performing planar calibration on PIV images,
with robust dot detection and grid organization.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

# ---------- Image Loading and Basic Utilities ----------


def load_image(path):
    """Load an image from path and convert to grayscale float32."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img


def save_figure(fig, filename, dpi=300):
    """Save a matplotlib figure as a TIFF file with high resolution."""
    if not str(filename).endswith(".tif") and not str(filename).endswith(".tiff"):
        filename = f"{filename}.tif"

    try:
        fig.savefig(filename, format="tiff", dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving figure: {e}")
        return False


def get_output_path(save_path, suffix=""):
    """Generate an output path with optional suffix."""
    if save_path is None:
        return None

    # Remove extension if present
    path_str = str(save_path)
    if "." in Path(path_str).name:
        base_path = path_str.rsplit(".", 1)[0]
    else:
        base_path = path_str

    # Add suffix if provided
    if suffix:
        return f"{base_path}_{suffix}"
    else:
        return base_path


# ---------- Robust Dot Detection Methods ----------


def detect_dots_threshold(image, threshold_method="otsu", min_area=5, max_area=500):
    """
    Detect dots using thresholding and connected components analysis.

    Args:
        image: Input grayscale image
        threshold_method: Method for thresholding ('otsu', 'adaptive', or int value)
        min_area: Minimum dot area
        max_area: Maximum dot area

    Returns:
        Array of detected dot centers (col, row) = (x, y)
    """
    # Normalize image for processing
    if image.dtype != np.uint8:
        img_proc = (255 * (image - image.min()) / (image.max() - image.min())).astype(
            np.uint8
        )
    else:
        img_proc = image.copy()

    # Invert if needed (we want white dots on black background)
    mean_val = np.mean(img_proc)
    if mean_val > 127:
        img_proc = 255 - img_proc

    # Apply threshold to isolate dots
    if threshold_method == "otsu":
        _, binary = cv2.threshold(img_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            img_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Use a fixed threshold value
        try:
            thresh_val = int(threshold_method)
            _, binary = cv2.threshold(img_proc, thresh_val, 255, cv2.THRESH_BINARY)
        except ValueError:
            # Default to Otsu if invalid value provided
            _, binary = cv2.threshold(
                img_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    # Find connected components (the dots)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Filter components by size to remove noise and background
    # Skip the first component (id=0) which is the background
    centers = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            # Correct: OpenCV returns (x, y)
            x, y = centroids[i]
            centers.append([x, y])

    print(f"Threshold-based detection found {len(centers)} dots")
    return np.array(centers) if centers else np.empty((0, 2))


def detect_dots(image, debug=False):
    """
    Only use threshold-based dot detection.
    Ensures output is (col, row) = (x, y) order.

    Args:
        image: Input grayscale image
        debug: Whether to display debug visualizations

    Returns:
        Array of detected dot centers (col, row)
    """
    # Automatically determine size parameters based on image size
    h, w = image.shape
    min_area = max(5, int((min(h, w) / 150) ** 2))
    max_area = int((min(h, w) / 15) ** 2)
    print(f"Using dot size parameters: min_area={min_area}, max_area={max_area}")

    # Only threshold-based detection
    threshold_dots = detect_dots_threshold(image, "otsu", min_area, max_area)
    centers = threshold_dots

    if len(centers) == 0:
        print("Warning: No dots detected by threshold method")
        return np.empty((0, 2))

    # Debug visualization if requested
    if debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap="gray")
        if len(centers) > 0:
            ax.scatter(centers[:, 0], centers[:, 1], c="g", s=40, marker="+")
        ax.set_title(f"Threshold Method: {len(centers)} dots")
        plt.tight_layout()
        plt.show()
        plt.close()

    return centers


# ---------- User Interaction ----------


def find_nearest_dot(click_point, dots):
    """Find the nearest detected dot to a clicked point."""
    if len(dots) == 0:
        return click_point, float("inf")

    # Calculate distances to all dots
    distances = np.sqrt(np.sum((dots - click_point) ** 2, axis=1))

    # Find index of closest dot
    closest_idx = np.argmin(distances)

    return dots[closest_idx], distances[closest_idx]


def get_user_points_with_snapping(img, dots):
    """Get user-clicked points from an image and snap them to the nearest detected dots."""
    # Click order: datum, right of datum, above datum
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray")
    plt.title(
        "Click: (1) datum, (2) right of datum, (3) above datum. Points will snap to nearest dot."
    )

    # Show dots for reference
    plt.scatter(dots[:, 0], dots[:, 1], c="g", s=20, alpha=0.5)

    pts = []
    snapped_pts = []
    labels = ["D", "R", "A"]
    while len(pts) < 3:
        p = plt.ginput(1, timeout=-1)
        if not p:
            continue
        x, y = p[0]
        click_point = np.array([x, y])  # store (col,row)

        # Find nearest dot and snap to it
        nearest_dot, distance = find_nearest_dot(click_point, dots)

        # Store both clicked and snapped points
        pts.append(click_point)
        snapped_pts.append(nearest_dot)

        # Draw original click
        plt.plot(x, y, "rx")

        # Draw line from click to snapped position if they're different
        if np.any(click_point != nearest_dot):
            plt.plot([x, nearest_dot[0]], [y, nearest_dot[1]], "y-", alpha=0.5)

        # Draw snapped point
        plt.plot(nearest_dot[0], nearest_dot[1], "ro")

        # Draw a circle to highlight the selected point
        circle = plt.Circle(
            (nearest_dot[0], nearest_dot[1]),
            9,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
        plt.gca().add_patch(circle)

        plt.text(
            nearest_dot[0] + 15,
            nearest_dot[1],
            labels[len(pts) - 1],
            color="yellow",
            fontsize=12,
            fontweight="bold",
        )
        plt.draw()

    plt.close()

    datum = snapped_pts[0]
    right = snapped_pts[1]
    above = snapped_pts[2]
    return datum, right, above


# ---------- Grid Organization and Calibration ----------


def organize_grid_points(
    points, datum, right, above, dot_distance_mm=1.0, tolerance=0.5, angle_tol_deg=30
):
    """
    Organize detected points into a continuous grid using neighbor search from the datum.
    Each dot is assigned a unique (i, j) index by traversing the grid using geometric proximity and angle constraints.
    """
    # Compute axes
    ex = (right - datum) / np.linalg.norm(right - datum)
    ey = (above - datum) / np.linalg.norm(above - datum)
    dx = np.linalg.norm(right - datum)
    dy = np.linalg.norm(above - datum)
    angle_tol = np.deg2rad(angle_tol_deg)

    # Prepare structures
    assigned = {}
    used = set()
    datum_idx = np.argmin(np.linalg.norm(points - datum, axis=1))
    assigned[(0, 0)] = datum_idx
    used.add(datum_idx)
    queue = [((0, 0), datum_idx)]

    # Helper: find nearest unused neighbor in a direction with angle constraint
    def find_neighbor(idx, direction_vec, expected_dist, axis_tol, angle_tol):
        base = points[idx]
        candidates = []
        for i, pt in enumerate(points):
            if i in used:
                continue
            vec = pt - base
            proj = np.dot(vec, direction_vec)
            perp = np.linalg.norm(vec - proj * direction_vec)
            dist = np.linalg.norm(vec)
            # Angle between vec and direction_vec
            if dist == 0:
                continue
            angle = np.arccos(np.clip(np.dot(vec, direction_vec) / dist, -1.0, 1.0))
            # Must be in the correct direction, within distance and angle tolerance
            if (
                proj > expected_dist * 0.5
                and abs(proj - expected_dist) < expected_dist * axis_tol
                and perp < expected_dist * axis_tol
                and angle < angle_tol
            ):
                candidates.append((i, proj, perp, angle, dist))
        if not candidates:
            return None
        # Choose the one with smallest perp, then angle, then distance
        candidates.sort(key=lambda x: (x[2], x[3], abs(x[1] - expected_dist), x[4]))
        return candidates[0][0]

    # BFS: expand in all four directions from each assigned point
    while queue:
        (i, j), idx = queue.pop(0)
        # Right neighbor
        n_idx = find_neighbor(idx, ex, dx, tolerance, angle_tol)
        if n_idx is not None and (i + 1, j) not in assigned:
            assigned[(i + 1, j)] = n_idx
            used.add(n_idx)
            queue.append(((i + 1, j), n_idx))
        # Left neighbor
        n_idx = find_neighbor(idx, -ex, dx, tolerance, angle_tol)
        if n_idx is not None and (i - 1, j) not in assigned:
            assigned[(i - 1, j)] = n_idx
            used.add(n_idx)
            queue.append(((i - 1, j), n_idx))
        # Up neighbor
        n_idx = find_neighbor(idx, ey, dy, tolerance, angle_tol)
        if n_idx is not None and (i, j + 1) not in assigned:
            assigned[(i, j + 1)] = n_idx
            used.add(n_idx)
            queue.append(((i, j + 1), n_idx))
        # Down neighbor
        n_idx = find_neighbor(idx, -ey, dy, tolerance, angle_tol)
        if n_idx is not None and (i, j - 1) not in assigned:
            assigned[(i, j - 1)] = n_idx
            used.add(n_idx)
            queue.append(((i, j - 1), n_idx))

    # Build output arrays
    grid_indices = []
    grid_points = []
    for (i, j), idx in assigned.items():
        grid_indices.append((i, j))
        grid_points.append(points[idx])
    grid_indices = np.array(grid_indices)
    grid_points = np.array(grid_points)

    # Diagnostics
    valid_points = np.zeros(len(points), dtype=bool)
    for idx in assigned.values():
        valid_points[idx] = True

    x_grid = np.full(len(points), -1)
    y_grid = np.full(len(points), -1)
    for (i, j), idx in assigned.items():
        x_grid[idx] = i
        y_grid[idx] = j

    all_projections = {
        "x_norm": np.zeros(len(points)),
        "y_norm": np.zeros(len(points)),
        "x_grid": x_grid,
        "y_grid": y_grid,
        "x_residual": np.zeros(len(points)),
        "y_residual": np.zeros(len(points)),
        "valid_points": valid_points,
    }

    print(
        f"Grid organization: {len(grid_points)} points assigned out of {len(points)} detected"
    )
    print(
        f"Grid organization rejection rate: {100 * (1 - len(grid_points) / len(points)):.1f}%"
    )
    print(
        f"Grid size: {np.max(grid_indices[: , 0]) + 1}x{np.max(grid_indices[: , 1]) + 1}"
    )

    # Estimate scale from median neighbor distances
    if len(grid_points) > 1:
        dxs = []
        dys = []
        for (i, j), idx in assigned.items():
            # Right neighbor
            if (i + 1, j) in assigned:
                dxs.append(np.linalg.norm(points[assigned[(i + 1, j)]] - points[idx]))
            # Up neighbor
            if (i, j + 1) in assigned:
                dys.append(np.linalg.norm(points[assigned[(i, j + 1)]] - points[idx]))
        dx_pixels = np.median(dxs) if dxs else dx
        dy_pixels = np.median(dys) if dys else dy
    else:
        dx_pixels = dx
        dy_pixels = dy

    scale_x = dot_distance_mm / dx_pixels
    scale_y = dot_distance_mm / dy_pixels

    return grid_points, grid_indices, scale_x, scale_y, all_projections


def calculate_homography(
    grid_points, grid_indices, dot_distance_mm, ransac_threshold=3.0
):
    """Calculate homography matrix with RANSAC for robust outlier rejection."""
    # Convert grid points to the format expected by OpenCV (x, y)
    src_points = np.array([[p[0], p[1]] for p in grid_points], dtype=np.float32)

    # Convert grid indices to physical world coordinates
    world_points = np.array(
        [[idx[0] * dot_distance_mm, idx[1] * dot_distance_mm] for idx in grid_indices],
        dtype=np.float32,
    )

    # Find homography using RANSAC
    H, inlier_mask = cv2.findHomography(
        src_points, world_points, cv2.RANSAC, ransac_threshold
    )
    inlier_mask = inlier_mask.ravel().astype(bool)

    print(f"RANSAC kept {np.sum(inlier_mask)} inliers out of {len(grid_points)} points")
    return H, world_points, inlier_mask


def fit_camera_model(image_points, world_points):
    """Fit a camera model to the calibration points."""
    # Convert points to the format expected by OpenCV
    img_pts = np.array([[p[0], p[1]] for p in image_points], dtype=np.float32)

    # Add Z=0 to world points to make them 3D
    world_pts = np.hstack([world_points, np.zeros((world_points.shape[0], 1))]).astype(
        np.float32
    )

    # Reshape to (n, 1, 3) and (n, 1, 2) as required by OpenCV
    world_pts = world_pts.reshape(-1, 1, 3)
    img_pts = img_pts.reshape(-1, 1, 2)

    # Get image size from the range of image points
    h = int(np.max(image_points[:, 0])) + 100
    w = int(np.max(image_points[:, 1])) + 100

    # Prepare object point sets (single position with multiple points)
    objp = world_pts
    objpoints = [objp]
    imgpoints = [img_pts]

    # Find camera calibration parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    result = {
        "camera_matrix": mtx,
        "dist_coeffs": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reprojection_error": mean_error,
    }

    return result


def dewarp_image(image, homography, mm_per_pixel=0.1):
    """Dewarp an image using homography with physical scaling."""
    h, w = image.shape[:2]

    # Compute inverse of homography (physical to image)
    np.linalg.inv(homography)

    # Find the physical coordinates of image corners
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )

    # Transform corners to physical coordinates
    physical_corners = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)

    # Compute physical size of the calibration area
    min_x, min_y = np.min(physical_corners, axis=0)
    max_x, max_y = np.max(physical_corners, axis=0)

    physical_width = max_x - min_x
    physical_height = max_y - min_y

    # Compute dewarped image size at desired resolution
    dewarped_w = int(physical_width / mm_per_pixel)
    dewarped_h = int(physical_height / mm_per_pixel)

    # Ensure minimum size
    dewarped_w = max(dewarped_w, 100)
    dewarped_h = max(dewarped_h, 100)

    # Create transform from physical coordinates to dewarped image
    physical_to_dewarped = np.array(
        [
            [1 / mm_per_pixel, 0, -min_x / mm_per_pixel],
            [0, 1 / mm_per_pixel, -min_y / mm_per_pixel],
            [0, 0, 1],
        ]
    )

    # Combined transform: image -> physical -> dewarped
    transform = physical_to_dewarped @ homography

    # Perform the dewarping
    dewarped = cv2.warpPerspective(image, transform, (dewarped_w, dewarped_h))

    # Return effective resolution (mm/pixel)
    effective_resolution = mm_per_pixel

    return dewarped, transform, effective_resolution


# ---------- Visualization ----------


def visualize_detected_dots(img, dots, save_path=None):
    """Visualize detected dots on the original image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray")

    if len(dots) > 0:
        ax.scatter(
            dots[:, 0],
            dots[:, 1],
            c="r",
            s=40,
            marker="o",
            alpha=0.7,
            label="Detected Dots",
        )

    ax.set_title(f"Grid Detection: {len(dots)} dots detected")
    ax.legend(loc="upper right")
    ax.axis("equal")
    plt.tight_layout()

    if save_path:
        save_figure(fig, f"{save_path}_detected_dots")

    plt.show()
    plt.close()


def visualize_grid(
    img, grid_points, grid_indices, inlier_mask=None, save_path=None, all_dots=None
):
    """Visualize grid points with their indices and grid structure.
    Optionally overlays all detected threshold dots for context."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, cmap="gray")

    # Overlay all detected threshold dots for context (if provided)
    if all_dots is not None and len(all_dots) > 0:
        ax.scatter(
            all_dots[:, 0],
            all_dots[:, 1],
            c="gray",
            s=20,
            marker=".",
            alpha=0.4,
            label="All threshold dots",
        )

    # Ensure grid_indices is a numpy array for boolean indexing
    grid_indices = np.asarray(grid_indices)

    # Use all points if no mask provided
    if inlier_mask is None:
        inlier_mask = np.ones(len(grid_points), dtype=bool)

    # Plot points (green for inliers, red for outliers)
    ax.scatter(
        grid_points[inlier_mask, 0],
        grid_points[inlier_mask, 1],
        c="g",
        s=40,
        marker="o",
        label="Inliers",
    )

    if np.sum(~inlier_mask) > 0:
        ax.scatter(
            grid_points[~inlier_mask, 0],
            grid_points[~inlier_mask, 1],
            c="r",
            s=40,
            marker="x",
            label="Outliers",
        )

    # Filter indices to only use inliers
    inlier_points = grid_points[inlier_mask]
    inlier_grid_indices = grid_indices[inlier_mask]

    # Group by rows
    row_groups = {}
    for i, (_, row_idx) in enumerate(inlier_grid_indices):
        if row_idx not in row_groups:
            row_groups[row_idx] = []
        row_groups[row_idx].append(i)

    # Group by columns
    col_groups = {}
    for i, (col_idx, _) in enumerate(inlier_grid_indices):
        if col_idx not in col_groups:
            col_groups[col_idx] = []
        col_groups[col_idx].append(i)

    # Draw row lines
    for row, indices in row_groups.items():
        if len(indices) > 1:
            # Sort by x-coordinate
            sorted_idx = sorted(indices, key=lambda i: inlier_points[i, 0])
            ax.plot(
                inlier_points[sorted_idx, 0],
                inlier_points[sorted_idx, 1],
                "g-",
                linewidth=1.5,
            )

    # Draw column lines
    for col, indices in col_groups.items():
        if len(indices) > 1:
            # Sort by y-coordinate
            sorted_idx = sorted(indices, key=lambda i: inlier_points[i, 1])
            ax.plot(
                inlier_points[sorted_idx, 0],
                inlier_points[sorted_idx, 1],
                "b-",
                linewidth=1.5,
            )

    # Add grid indices as text
    for i, (col_idx, row_idx) in enumerate(inlier_grid_indices):
        ax.text(
            inlier_points[i, 0] + 10,
            inlier_points[i, 1],
            f"({col_idx},{row_idx})",
            color="cyan",
            fontsize=8,
        )

    ax.set_title("Calibration Grid with Point Indices")
    ax.legend(loc="upper right")
    ax.set_axis_off()  # Remove axes for cleaner look
    plt.tight_layout()

    if save_path:
        save_figure(fig, f"{save_path}_indexed_grid")

    plt.show()
    plt.close()


def visualize_dewarped_image(
    dewarped_image, dot_distance_mm, effective_resolution, save_path=None
):
    """Visualize the dewarped image with physical grid lines in mm."""
    h, w = dewarped_image.shape[:2]

    # Calculate how many mm the image covers
    width_mm = w * effective_resolution
    height_mm = h * effective_resolution

    fig, ax = plt.subplots(figsize=(12, 12 * height_mm / width_mm))
    ax.imshow(dewarped_image, cmap="gray", extent=[0, width_mm, 0, height_mm])

    # Draw grid lines every dot_distance_mm
    x_lines = np.arange(0, width_mm + 1, dot_distance_mm)
    y_lines = np.arange(0, height_mm + 1, dot_distance_mm)

    # Draw vertical grid lines
    for x in x_lines:
        ax.axvline(x=x, color="r", linestyle="-", alpha=0.3)

    # Draw horizontal grid lines
    for y in y_lines:
        ax.axhline(y=y, color="b", linestyle="-", alpha=0.3)

    # Add a scale bar
    scale_bar_length_mm = 5 * dot_distance_mm  # 5x the dot distance
    bar_start_x = width_mm - scale_bar_length_mm - dot_distance_mm / 2
    bar_start_y = dot_distance_mm / 2
    ax.plot(
        [bar_start_x, bar_start_x + scale_bar_length_mm],
        [bar_start_y, bar_start_y],
        "k-",
        linewidth=3,
    )
    ax.text(
        bar_start_x + scale_bar_length_mm / 2,
        bar_start_y + dot_distance_mm / 2,
        f"{scale_bar_length_mm} mm",
        ha="center",
        color="black",
        fontweight="bold",
    )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Dewarped Image with Physical Grid (mm)")
    ax.axis("equal")
    plt.tight_layout()

    if save_path:
        save_figure(fig, f"{save_path}_dewarped")

    plt.show()
    plt.close()


# Fix the visualization function to properly draw connections between dots
def visualize_grid_diagnostics(
    img, points, datum, right, above, all_projections, tolerance, save_path=None
):
    """Visualize the grid organization focusing only on the actual dot positions and their assigned indices."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, cmap="gray")

    # Get valid points and their grid indices
    valid_points = all_projections["valid_points"]

    # Get grid indices for valid points
    valid_dot_positions = points[valid_points]
    valid_i_idx = all_projections["x_grid"][valid_points]
    valid_j_idx = all_projections["y_grid"][valid_points]

    # Plot all detected dots in gray
    ax.scatter(
        points[:, 0],
        points[:, 1],
        color="gray",
        s=20,
        marker=".",
        alpha=0.3,
        label="All detected dots",
    )

    # Plot valid dots (accepted into the grid) in green
    ax.scatter(
        valid_dot_positions[:, 0],
        valid_dot_positions[:, 1],
        color="lime",
        s=40,
        marker="o",
        label="Dots used in calibration",
    )

    # Add grid indices as text labels
    for k, (i, j) in enumerate(zip(valid_i_idx, valid_j_idx)):
        ax.text(
            valid_dot_positions[k, 0] + 10,
            valid_dot_positions[k, 1],
            f"({i},{j})",
            color="cyan",
            fontsize=8,
        )

    # Create lookup table to quickly find dot by grid index
    grid_lookup = {}
    for k, (i, j) in enumerate(zip(valid_i_idx, valid_j_idx)):
        grid_lookup[(i, j)] = k

    # Draw grid lines (only connect adjacent points)
    i_values = np.unique(valid_i_idx)
    j_values = np.unique(valid_j_idx)

    # Draw horizontal lines
    for j in j_values:
        # Get all i indices for this row
        i_in_row = sorted(
            [i for i, j_idx in zip(valid_i_idx, valid_j_idx) if j_idx == j]
        )

        # Connect adjacent points
        for i_idx in range(len(i_in_row) - 1):
            i1, i2 = i_in_row[i_idx], i_in_row[i_idx + 1]

            # Skip if not adjacent
            if i2 - i1 > 1:
                continue

            # Get point indices
            p1_idx = grid_lookup[(i1, j)]
            p2_idx = grid_lookup[(i2, j)]

            # Draw line
            ax.plot(
                [valid_dot_positions[p1_idx, 0], valid_dot_positions[p2_idx, 0]],
                [valid_dot_positions[p1_idx, 1], valid_dot_positions[p2_idx, 1]],
                "g-",
                linewidth=1.5,
            )

    # Draw vertical lines
    for i in i_values:
        # Get all j indices for this column
        j_in_col = sorted(
            [j for i_idx, j in zip(valid_i_idx, valid_j_idx) if i_idx == i]
        )

        # Connect adjacent points
        for j_idx in range(len(j_in_col) - 1):
            j1, j2 = j_in_col[j_idx], j_in_col[j_idx + 1]

            # Skip if not adjacent
            if j2 - j1 > 1:
                continue

            # Get point indices
            p1_idx = grid_lookup[(i, j1)]
            p2_idx = grid_lookup[(i, j2)]

            # Draw line
            ax.plot(
                [valid_dot_positions[p1_idx, 0], valid_dot_positions[p2_idx, 0]],
                [valid_dot_positions[p1_idx, 1], valid_dot_positions[p2_idx, 1]],
                "b-",
                linewidth=1.5,
            )

    # Add key information
    ax.text(
        0.02,
        0.02,
        f"Tolerance: {tolerance}\n"
        f"Accepted: {np.sum(valid_points)}/{len(points)} ({100 * np.sum(valid_points) / len(points):.1f}%)\n"
        f"Grid size: {len(i_values)}x{len(j_values)}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    ax.set_title("Grid Organization: Actual Dot Positions")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        save_figure(fig, f"{save_path}_grid_diagnostics")

    plt.show()
    plt.close()


# ---------- Save Calibration Data ----------


def save_calibration_results(save_path, calibration_data, format="mat"):
    """Save calibration results to a file."""
    if format.lower() == "mat":
        # Ensure save_path has .mat extension
        if not str(save_path).endswith(".mat"):
            save_path = str(save_path) + ".mat"

        # Convert any None values to empty arrays for MATLAB compatibility
        mat_data = {}
        for key, value in calibration_data.items():
            if value is None:
                mat_data[key] = np.array([])
            elif isinstance(value, str):
                # MATLAB can't handle arbitrary strings well
                mat_data[key] = value
            elif isinstance(value, Path):
                mat_data[key] = str(value)
            else:
                mat_data[key] = value

        try:
            savemat(save_path, mat_data)
            print(f"Calibration data saved to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving .mat file: {e}")
            return False

    else:  # 'npz' format
        # Ensure save_path has .npz extension
        if not str(save_path).endswith(".npz"):
            save_path = str(save_path) + ".npz"

        try:
            np.savez_compressed(save_path, **calibration_data)
            print(f"Calibration data saved to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving .npz file: {e}")
            return False


# ---------- Main Calibration Function ----------


def calibrate(
    image_path,
    dot_distance_mm=28.9,
    output_resolution_mm=0.1,
    display=True,
    save_path=None,
    save_format="mat",
    ransac_threshold=3.0,
    grid_tolerance=0.5,
    debug=False,
):
    """
    Perform planar calibration on an image with physical scaling.

    Args:
        image_path: Path to the input image
        dot_distance_mm: Physical distance between dots in mm
        output_resolution_mm: Resolution of output dewarped image in mm/pixel
        display: Whether to display results
        save_path: Path to save calibration results
        save_format: Format to save results ('mat' or 'npz')
        ransac_threshold: Maximum allowed reprojection error in RANSAC (pixels)
        grid_tolerance: Tolerance for assigning dots to grid positions (0.3-0.7 recommended)
        debug: Whether to show detailed debugging visualizations

    Returns:
        Dictionary with calibration results
    """
    # Step 1: Load image
    print(f"Loading image: {image_path}")
    img = load_image(image_path)

    # Step 2: Detect dots
    print("Detecting calibration dots...")
    dots = detect_dots(img, debug=debug)

    if len(dots) < 10:
        print(f"Warning: Only {len(dots)} dots detected. Results may be unreliable.")

    # Skip displaying the detected dots visualization
    # Step 3: Get user input for datum points
    print("Please select the datum, right, and above points...")
    datum, right, above = get_user_points_with_snapping(img, dots)

    # Step 4: Organize dots into grid structure
    print(f"Organizing points into a grid (tolerance={grid_tolerance})...")
    grid_points, grid_indices, scale_x, scale_y, all_projections = organize_grid_points(
        dots, datum, right, above, dot_distance_mm, tolerance=grid_tolerance
    )

    # Always show the grid diagnostics visualization if display is enabled
    if display:
        visualize_grid_diagnostics(
            img,
            dots,
            datum,
            right,
            above,
            all_projections,
            grid_tolerance,
            save_path=get_output_path(save_path),
        )

    if len(grid_points) < 10:
        print(
            f"Warning: Only {len(grid_points)} points assigned to grid. Results may be unreliable."
        )
        print(
            f"Try increasing the grid_tolerance (current: {grid_tolerance}) to include more dots."
        )
    else:
        print(f"Assigned {len(grid_points)} points to grid structure.")

    # Step 6: Calculate homography with RANSAC
    print("Calculating homography...")
    H, world_points, inlier_mask = calculate_homography(
        grid_points, grid_indices, dot_distance_mm, ransac_threshold
    )

    # Step 7: Visualize grid (uses organized and inlier points, overlays all threshold dots)
    if display:
        visualize_grid(
            img,
            grid_points,
            grid_indices,
            inlier_mask,
            save_path=get_output_path(save_path),
            all_dots=dots,
        )

    # Step 8: Fit camera model
    print("Fitting camera model...")
    try:
        camera_model = fit_camera_model(
            grid_points[inlier_mask], world_points[inlier_mask]
        )
        print(f"Reprojection error: {camera_model['reprojection_error']:.6f}")
    except Exception as e:
        print(f"Error fitting camera model: {e}")
        camera_model = {
            "camera_matrix": None,
            "dist_coeffs": None,
            "rvecs": None,
            "tvecs": None,
            "reprojection_error": None,
        }

    # Step 9: Dewarp image using homography
    print("Dewarping image...")
    dewarped, transform, effective_resolution = dewarp_image(
        img, H, mm_per_pixel=output_resolution_mm
    )

    # Step 10: Visualize dewarped image (moved after dewarping)
    if display:
        visualize_dewarped_image(
            dewarped,
            dot_distance_mm,
            effective_resolution,
            save_path=get_output_path(save_path),
        )

    # Step 11: Prepare results dictionary
    results = {
        "image_path": str(image_path),
        "grid_points": grid_points,
        "grid_indices": grid_indices,
        "inlier_mask": inlier_mask,
        "homography": H,
        "transform": transform,
        "dewarped": dewarped,
        "dot_distance_mm": dot_distance_mm,
        "effective_resolution": effective_resolution,
        "datum": datum,
        "right": right,
        "above": above,
        "camera_matrix": camera_model["camera_matrix"],
        "dist_coeffs": camera_model["dist_coeffs"],
        "rvecs": camera_model["rvecs"],
        "tvecs": camera_model["tvecs"],
        "reprojection_error": camera_model["reprojection_error"],
        "world_points": world_points,
        "grid_tolerance": grid_tolerance,
    }

    # Step 12: Save calibration data
    if save_path:
        out_path = get_output_path(save_path)
        save_calibration_results(out_path, results, save_format)

    print("Calibration complete.")
    return results


# ---------- Command Line Interface ----------

if __name__ == "__main__":
    calibrate(
        "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/Calibration/Cam1/enhanced.tif",
        save_path=str(Path.cwd() / "calibration_results"),
        dot_distance_mm=28.9,
        output_resolution_mm=0.1,
        display=True,
        save_format="mat",
        ransac_threshold=0.5,
        grid_tolerance=0.7,  # Increased from default to be more lenient
        debug=True,
    )
