#!/usr/bin/env python3
"""
dewarp_images.py

Loads calibration results, loads all images for Cam1 and Cam2, undistorts (dewarps) them using the single-camera calibration parameters, and saves the dewarped images to new folders.
"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Paths
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Stereo_Images/calibration/"
CAM1_DIR = os.path.join(BASE_DIR, "Cam1")
CAM2_DIR = os.path.join(BASE_DIR, "Cam2")
PATTERN = "planar_calibration_plate_*.tif"
CALIB_FILE = os.path.join(BASE_DIR, "stereo_calibration_results.npz")

# Output folders
OUT_CAM1 = os.path.join(BASE_DIR, "Cam1_dewarped")
OUT_CAM2 = os.path.join(BASE_DIR, "Cam2_dewarped")
os.makedirs(OUT_CAM1, exist_ok=True)
os.makedirs(OUT_CAM2, exist_ok=True)

# Pattern parameters (match stereo_cv2.py)
PATTERN_COLS = 10
PATTERN_ROWS = 10
PATTERN_SIZE = (PATTERN_COLS, PATTERN_ROWS)
DOT_SPACING_MM = 28.89  # renamed from SQUARE_SIZE_MM
ASYMMETRIC = False

# Flag to enable/disable visualization
VISUALIZE = True


def create_blob_detector():
    """Create a blob detector tuned for circle grid detection"""
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = 200
    p.maxArea = 1000
    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False
    p.minThreshold = 0
    p.maxThreshold = 255
    p.thresholdStep = 5
    detector = cv2.SimpleBlobDetector_create(p)
    return detector


def make_object_points(pattern_size, dot_spacing_mm, asymmetric=False):
    """Create 3D object points for calibration grid in physical coordinates (mm)"""
    cols, rows = pattern_size
    objp = []
    for i in range(rows):
        for j in range(cols):
            if asymmetric:
                x = j * dot_spacing_mm + (0.5 * dot_spacing_mm if (i % 2 == 1) else 0.0)
                y = i * dot_spacing_mm
            else:
                x = j * dot_spacing_mm
                y = i * dot_spacing_mm
            objp.append([x, y, 0.0])
    return np.array(objp, dtype=np.float32)


def detect_grid_in_image(img, pattern_size, detector, asymmetric=False, debug=False):
    """
    Detect circle grid in an image and return centers.
    Tries both original and inverted images for better detection.

    Args:
        img: Input image (grayscale)
        pattern_size: Tuple (cols, rows) for the grid pattern
        detector: Blob detector instance
        asymmetric: Whether grid is asymmetric
        debug: Whether to show debugging visualizations

    Returns:
        (found, centers) - found is boolean, centers is Nx2 array of (x,y) points
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Determine grid detection flags
    grid_flags = 0
    if asymmetric:
        grid_flags |= cv2.CALIB_CB_ASYMMETRIC_GRID
    else:
        grid_flags |= cv2.CALIB_CB_SYMMETRIC_GRID

    # Try both original and inverted images
    for test_img, label in [(gray, "Original"), (255 - gray, "Inverted")]:
        # The key fix: Use the correct parameter name "blobDetector"
        found, centers = cv2.findCirclesGrid(
            test_img, pattern_size, flags=grid_flags, blobDetector=detector
        )

        if found and debug:
            # Visualize the detected grid
            vis = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, centers, found)
            cv2.imshow(f"Grid Detection ({label})", vis)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        if found:
            print(f"Grid detected ({label} image)")
            return True, centers.reshape(-1, 2).astype(np.float32)

    return False, None


def compute_homography_for_dewarping(image_points, object_points_2d):
    """
    Compute homography matrix mapping image points to physical coordinates

    Args:
        image_points: Detected grid points in image (Nx2)
        object_points_2d: Physical grid coordinates in mm (Nx2)

    Returns:
        H: Homography matrix
    """
    H, status = cv2.findHomography(image_points, object_points_2d, cv2.RANSAC, 3.0)
    inlier_count = np.sum(status)
    print(f"Homography RANSAC: {inlier_count}/{len(status)} inliers")
    return H


def dewarp_image_with_homography(img, homography, output_size=None):
    """
    Dewarp image using homography matrix

    Args:
        img: Input image
        homography: Homography matrix
        output_size: Optional tuple (width, height) for output image

    Returns:
        Dewarped image
    """
    h, w = img.shape[:2]
    if output_size is None:
        output_size = (w, h)

    dewarped = cv2.warpPerspective(
        img, homography, output_size, flags=cv2.INTER_LANCZOS4
    )
    return dewarped


def calculate_dewarped_size(H, img_shape, dot_spacing_mm):
    """
    Calculate output image size and transformation so that the dewarped image
    covers the full area of the original image, mapped into physical coordinates (mm).
    """
    h, w = img_shape[:2]
    # Transform image corners to physical space
    corners = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    ).reshape(-1, 1, 2)
    physical_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    min_x = np.min(physical_corners[:, 0])
    max_x = np.max(physical_corners[:, 0])
    min_y = np.min(physical_corners[:, 1])
    max_y = np.max(physical_corners[:, 1])

    # Use 1 pixel per mm
    width_px = int(np.ceil(max_x - min_x))
    height_px = int(np.ceil(max_y - min_y))

    # Map physical coordinates (mm) to pixel coordinates in output image
    physical_to_pixel = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
    )

    # Combined transformation: image -> physical (H), then physical -> pixel
    combined_H = physical_to_pixel @ H

    return (width_px, height_px), combined_H


def visualize_dewarping(original, dewarped, grid_points=None, dewarped_points=None):
    """
    Visualize original vs dewarped image, optionally with grid points

    Args:
        original: Original image
        dewarped: Dewarped image
        grid_points: Optional grid points detected in original image
        dewarped_points: Optional grid points in dewarped image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Display original image
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")

    if grid_points is not None:
        axes[0].scatter(
            grid_points[:, 0], grid_points[:, 1], c="r", s=40, marker="o", alpha=0.7
        )

    # Display dewarped image
    axes[1].imshow(dewarped, cmap="gray")
    axes[1].set_title("Dewarped Image")

    if dewarped_points is not None:
        axes[1].scatter(
            dewarped_points[:, 0],
            dewarped_points[:, 1],
            c="g",
            s=40,
            marker="o",
            alpha=0.7,
        )

    plt.tight_layout()
    plt.show()


def visualize_grid_indexing_and_error(img, grid_points, objp_2d, H):
    """
    Visualize grid indexing and reprojection error.
    Shows detected grid points, their indices, and error vectors.
    """
    # Project object points to image using inverse homography
    H_inv = np.linalg.inv(H)
    objp_h = np.hstack([objp_2d, np.ones((objp_2d.shape[0], 1))])
    projected = (H_inv @ objp_h.T).T
    projected = projected[:, :2] / projected[:, 2:]

    errors = np.linalg.norm(grid_points - projected, axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray")
    # Draw detected grid points
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c="r", s=40, label="Detected")
    # Draw projected points
    ax.scatter(projected[:, 0], projected[:, 1], c="g", s=30, label="Reprojected")
    # Draw error vectors
    for i in range(len(grid_points)):
        ax.plot(
            [grid_points[i, 0], projected[i, 0]],
            [grid_points[i, 1], projected[i, 1]],
            "y-",
            alpha=0.6,
        )
        ax.text(
            grid_points[i, 0] + 8, grid_points[i, 1], f"{i}", color="cyan", fontsize=8
        )
    mean_err = errors.mean()
    ax.set_title(f"Grid Indexing & Reprojection Error\nMean error: {mean_err:.2f} px")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def calculate_reprojection_error(grid_points, objp_2d, H):
    """
    Calculate mean reprojection error between detected grid points and
    object points projected into image space using inverse homography.
    Matches planar_calibration.py logic.
    """
    H_inv = np.linalg.inv(H)
    objp_h = np.hstack([objp_2d, np.ones((objp_2d.shape[0], 1))])
    projected = (H_inv @ objp_h.T).T
    projected = projected[:, :2] / projected[:, 2:]
    errors = np.linalg.norm(grid_points - projected, axis=1)
    mean_err = errors.mean()
    return mean_err, errors


def visualize_original_dewarped_with_error(original, dewarped, grid_points, objp_2d, H):
    """
    Show original and dewarped images side-by-side, overlay grid points and reprojection error.
    Figure auto-closes after 1 second.
    """
    mean_err, errors = calculate_reprojection_error(grid_points, objp_2d, H)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Original image
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].scatter(grid_points[:, 0], grid_points[:, 1], c="r", s=40, label="Grid")
    # Draw error vectors
    H_inv = np.linalg.inv(H)
    objp_h = np.hstack([objp_2d, np.ones((objp_2d.shape[0], 1))])
    projected = (H_inv @ objp_h.T).T
    projected = projected[:, :2] / projected[:, 2:]
    for i in range(len(grid_points)):
        axes[0].plot(
            [grid_points[i, 0], projected[i, 0]],
            [grid_points[i, 1], projected[i, 1]],
            "y-",
            alpha=0.5,
        )
    axes[0].legend()

    # Dewarped image
    axes[1].imshow(dewarped, cmap="gray")
    axes[1].set_title(f"Dewarped\nMean reprojection error: {mean_err:.2f} px")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1.0)
    plt.close()


def fit_camera_model(image_points, object_points_2d):
    """
    Fit a camera model using OpenCV's calibrateCamera and calculate reprojection error.
    image_points: Nx2 array of detected grid points in image (pixels)
    object_points_2d: Nx2 array of physical grid coordinates (mm)
    Returns: dict with camera_matrix, dist_coeffs, rvecs, tvecs, reprojection_error
    """
    img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    obj_pts = (
        np.hstack([object_points_2d, np.zeros((object_points_2d.shape[0], 1))])
        .astype(np.float32)
        .reshape(-1, 1, 3)
    )
    objpoints = [obj_pts]
    imgpoints = [img_pts]
    # Estimate image size from points
    w = int(np.max(image_points[:, 0])) + 100
    h = int(np.max(image_points[:, 1])) + 100
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


def auto_select_datum_right_above(grid_points):
    """
    Automatically select datum (bottom-left), right, and above points from detected grid points.
    Assumes grid_points is Nx2 array (x, y).
    Returns: datum, right, above (each as np.array([x, y]))
    """
    # Datum: point with minimum x + minimum y (bottom-left)
    idx_datum = np.argmin(grid_points[:, 0] + grid_points[:, 1])
    datum = grid_points[idx_datum]

    # Find right: closest point with similar y, larger x
    y_tol = (np.max(grid_points[:, 1]) - np.min(grid_points[:, 1])) / 20
    right_candidates = [
        i
        for i in range(len(grid_points))
        if abs(grid_points[i, 1] - datum[1]) < y_tol
        and grid_points[i, 0] > datum[0] + 1
    ]
    if right_candidates:
        idx_right = min(right_candidates, key=lambda i: grid_points[i, 0] - datum[0])
        right = grid_points[idx_right]
    else:
        # Fallback: next largest x
        idx_right = np.argmin(
            np.where(grid_points[:, 0] > datum[0], grid_points[:, 0] - datum[0], np.inf)
        )
        right = grid_points[idx_right]

    # Find above: closest point with similar x, larger y
    x_tol = (np.max(grid_points[:, 0]) - np.min(grid_points[:, 0])) / 20
    above_candidates = [
        i
        for i in range(len(grid_points))
        if abs(grid_points[i, 0] - datum[0]) < x_tol
        and grid_points[i, 1] > datum[1] + 1
    ]
    if above_candidates:
        idx_above = min(above_candidates, key=lambda i: grid_points[i, 1] - datum[1])
        above = grid_points[idx_above]
    else:
        # Fallback: next largest y
        idx_above = np.argmin(
            np.where(grid_points[:, 1] > datum[1], grid_points[:, 1] - datum[1], np.inf)
        )
        above = grid_points[idx_above]

    return datum, right, above


def organize_grid_points(
    points, datum, right, above, dot_distance_mm=1.0, tolerance=0.5, angle_tol_deg=30
):
    """
    Organize detected points into a continuous grid using neighbor search from the datum.
    Returns grid_points, grid_indices, scale_x, scale_y, all_projections.
    """
    # ...copy logic from planar_calibration.py (see previous code for details)...
    # For brevity, use a comment to indicate unchanged code.
    # ...existing code from planar_calibration.py organize_grid_points...


def process_camera_images(
    cam_dir,
    out_dir,
    pattern_size,
    dot_spacing_mm,
    asymmetric,
    detector,
    visualize=False,
):
    """
    Process all calibration images for one camera

    Args:
        cam_dir: Input directory with images
        out_dir: Output directory for dewarped images
        pattern_size: Grid pattern size (cols, rows)
        dot_spacing_mm: Physical spacing between dots in mm
        asymmetric: Whether grid is asymmetric
        detector: Blob detector
        visualize: Whether to visualize results
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # List all calibration images
    image_files = sorted(glob.glob(os.path.join(cam_dir, PATTERN)))
    if not image_files:
        print(f"No images found in {cam_dir}")
        return

    print(f"Processing {len(image_files)} images from {os.path.basename(cam_dir)}")

    # Create 3D object points (physical coordinates in mm)
    objp = make_object_points(pattern_size, dot_spacing_mm, asymmetric)
    objp_2d = objp[:, :2]
    grid_points_list = []
    image_names = []
    for img_path in image_files:
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not open {img_path}")
            continue

        # Detect grid
        found, grid_points = detect_grid_in_image(
            img, pattern_size, detector, asymmetric, debug=False
        )

        if not found or grid_points.shape[0] < 3:
            print(
                f"Grid not found or too few points in {os.path.basename(img_path)} - skipping"
            )
            continue

        print(f"Processing {os.path.basename(img_path)}")

        # Generate object points (physical coordinates in mm) in grid order
        cols, rows = pattern_size
        objp_2d = np.array(
            [
                [j * dot_spacing_mm, i * dot_spacing_mm]
                for i in range(rows)
                for j in range(cols)
            ],
            dtype=np.float32,
        )

        # Use only as many points as detected (should match grid_points)
        objp_2d = objp_2d[: grid_points.shape[0]]

        # Compute homography for dewarping using grid points
        H, status = cv2.findHomography(grid_points, objp_2d, cv2.RANSAC, 3.0)
        output_size, combined_H = calculate_dewarped_size(H, img.shape, dot_spacing_mm)
        dewarped = dewarp_image_with_homography(img, combined_H, output_size)

        # Calculate and print reprojection error (matches planar_calibration.py)
        mean_err, errors = calculate_reprojection_error(grid_points, objp_2d, H)
        print(f"Mean reprojection error (homography): {mean_err:.2f} px")

        # Camera calibration and reprojection error (OpenCV)
        camera_model = fit_camera_model(grid_points, objp_2d)
        print(
            f"OpenCV camera calibration reprojection error: {camera_model['reprojection_error']:.2f} px"
        )
        print(f"Camera matrix:\n{camera_model['camera_matrix']}")

        # Save dewarped image
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, dewarped)
        print(f"Saved dewarped image: {out_path}")

        # Visualize if requested
        if visualize:
            visualize_original_dewarped_with_error(
                img, dewarped, grid_points, objp_2d, H
            )
        grid_points_list.append(grid_points)
        image_names.append(os.path.basename(img_path))
    # Return all grid points and image names for indexing comparison
    return grid_points_list, image_names


def compare_camera_grid_indexing_all(
    cam1_points_list, cam2_points_list, cam1_names, cam2_names, pattern_size
):
    """
    Compare grid indexing between two cameras for all images by plotting their detected grid points.
    Assumes bottom-left is (0,0), x increases right, y increases up.
    """
    num_images = min(len(cam1_points_list), len(cam2_points_list))
    for i in range(num_images):
        cam1_points = cam1_points_list[i]
        cam2_points = cam2_points_list[i]
        cam1_name = cam1_names[i] if i < len(cam1_names) else f"Cam1_{i}"
        cam2_name = cam2_names[i] if i < len(cam2_names) else f"Cam2_{i}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Camera 1
        axes[0].scatter(cam1_points[:, 0], cam1_points[:, 1], c="r", s=40)
        for idx, (x, y) in enumerate(cam1_points):
            axes[0].text(x, y, f"{idx}", color="blue", fontsize=8)
        axes[0].set_title(f"Camera 1 Grid Indexing\n{cam1_name}")
        axes[0].set_xlabel("X (pixels)")
        axes[0].set_ylabel("Y (pixels)")
        axes[0].invert_yaxis()
        # Camera 2
        axes[1].scatter(cam2_points[:, 0], cam2_points[:, 1], c="g", s=40)
        for idx, (x, y) in enumerate(cam2_points):
            axes[1].text(x, y, f"{idx}", color="blue", fontsize=8)
        axes[1].set_title(f"Camera 2 Grid Indexing\n{cam2_name}")
        axes[1].set_xlabel("X (pixels)")
        axes[1].set_ylabel("Y (pixels)")
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.show()


def run_stereo_calibration(
    cam1_points_list, cam2_points_list, pattern_size, dot_spacing_mm
):
    """
    Run stereo calibration using corresponding grid points from both cameras.
    """
    # Prepare object points (same for all images)
    cols, rows = pattern_size
    objp = np.array(
        [
            [j * dot_spacing_mm, i * dot_spacing_mm, 0.0]
            for i in range(rows)
            for j in range(cols)
        ],
        dtype=np.float32,
    )
    # Only use as many points as detected in each image
    objpoints = []
    imgpoints1 = []
    imgpoints2 = []
    num_pairs = min(len(cam1_points_list), len(cam2_points_list))
    for i in range(num_pairs):
        n_pts = min(cam1_points_list[i].shape[0], cam2_points_list[i].shape[0])
        objpoints.append(objp[:n_pts].reshape(-1, 1, 3))
        imgpoints1.append(cam1_points_list[i][:n_pts].reshape(-1, 1, 2))
        imgpoints2.append(cam2_points_list[i][:n_pts].reshape(-1, 1, 2))

    # Estimate image size from points
    img_size = (
        int(max([np.max(pts[:, 0]) for pts in cam1_points_list]) + 100),
        int(max([np.max(pts[:, 1]) for pts in cam1_points_list]) + 100),
    )

    # Calibrate each camera individually to get initial camera matrices
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
        objpoints, imgpoints1, img_size, None, None
    )
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
        objpoints, imgpoints2, img_size, None, None
    )

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        mtx1,
        dist1,
        mtx2,
        dist2,
        img_size,
        criteria=criteria,
        flags=flags,
    )

    print("\nStereo Calibration Results:")
    print("R (rotation):\n", R)
    print("T (translation):\n", T)
    print("E (essential matrix):\n", E)
    print("F (fundamental matrix):\n", F)
    print("Stereo reprojection error:", ret)
    # Optionally, save results to file
    # np.savez("stereo_calibration_results.npz", mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T, E=E, F=F)


# Main execution
if __name__ == "__main__":
    # Create blob detector for grid detection
    detector = create_blob_detector()

    # Process Camera 1 images
    cam1_grid_points_list, cam1_image_names = process_camera_images(
        CAM1_DIR,
        OUT_CAM1,
        PATTERN_SIZE,
        DOT_SPACING_MM,
        ASYMMETRIC,
        detector,
        visualize=VISUALIZE,
    )
    # Process Camera 2 images
    cam2_grid_points_list, cam2_image_names = process_camera_images(
        CAM2_DIR,
        OUT_CAM2,
        PATTERN_SIZE,
        DOT_SPACING_MM,
        ASYMMETRIC,
        detector,
        visualize=VISUALIZE,
    )
    # Compare grid indexing for all images if both cameras have detected points
    if cam1_grid_points_list and cam2_grid_points_list:
        compare_camera_grid_indexing_all(
            cam1_grid_points_list,
            cam2_grid_points_list,
            cam1_image_names,
            cam2_image_names,
            PATTERN_SIZE,
        )
        # Run stereo calibration
        run_stereo_calibration(
            cam1_grid_points_list, cam2_grid_points_list, PATTERN_SIZE, DOT_SPACING_MM
        )
    print("Dewarping completed.")
