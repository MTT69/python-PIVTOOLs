"""
Camera Calibration from ChArUco Images

Reads images from IMAGES_FOLDER, detects ChArUco corners,
calibrates camera, and saves the model.

Usage:
    python calibrate_from_folder.py
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these parameters for your board
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
IMAGES_FOLDER = SCRIPT_DIR / "images_proc_cam1"

# ChArUco board parameters
SQUARES_H = 10          # Number of squares horizontally (columns)
SQUARES_V = 9           # Number of squares vertically (rows)
SQUARE_SIZE = 0.03      # Physical square size in meters (arbitrary, just consistent)
MARKER_RATIO = 0.5      # Marker size / square size (typically 0.5)
ARUCO_DICT = cv2.aruco.DICT_4X4_1000  # ArUco dictionary

# Detection settings
MIN_CORNERS = 6
IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

# =============================================================================


def create_detector():
    """Create ChArUco board and detector."""
    marker_size = SQUARE_SIZE * MARKER_RATIO
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_H, SQUARES_V),
        SQUARE_SIZE,
        marker_size,
        dictionary
    )

    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        cv2.aruco.DetectorParameters()
    )

    return board, detector


def process_images(folder, board, detector, detections_dir):
    """
    Load images, detect ChArUco corners, save detection visualizations.

    Returns:
        obj_points: List of 3D object points per valid image
        img_points: List of 2D image points per valid image
        valid_count: Number of valid images
        img_size: (width, height) tuple
    """
    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(f"Found {len(image_files)} images")
    print("-" * 50)

    obj_points, img_points = [], []
    img_size = None
    stats = {'empty': 0, 'no_detect': 0, 'valid': 0}

    for img_file in image_files:
        image = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        # Skip empty/black images
        if np.mean(image) < 10:
            print(f"  {img_file.name}: SKIP (empty)")
            stats['empty'] += 1
            continue

        # Convert 16-bit to 8-bit
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)

        if img_size is None:
            h, w = image.shape[:2]
            img_size = (w, h)

        # Detect ChArUco corners
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if ids is None or len(corners) < MIN_CORNERS:
            print(f"  {img_file.name}: SKIP (insufficient corners)")
            stats['no_detect'] += 1
            continue

        # Match to 3D object points
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < MIN_CORNERS:
            stats['no_detect'] += 1
            continue

        obj_points.append(obj_pts)
        img_points.append(img_pts)
        stats['valid'] += 1
        print(f"  {img_file.name}: OK ({len(corners)} corners)")

        # Save detection visualization
        save_detection_figure(
            image, corners, ids, marker_corners,
            img_file.stem, detections_dir
        )

    print("-" * 50)
    print(f"Valid: {stats['valid']}, Empty: {stats['empty']}, "
          f"No detection: {stats['no_detect']}")

    return obj_points, img_points, stats['valid'], img_size


def save_detection_figure(image, corners, ids, marker_corners,
                          name, output_dir):
    """Save figure showing detected corners on the image."""
    # Convert to color for visualization
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Draw marker outlines
    if marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners)

    # Draw ChArUco corners
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name} - {len(corners)} corners detected")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_detection.png", dpi=150)
    plt.close(fig)


def calibrate(obj_points, img_points, img_size):
    """Run OpenCV camera calibration."""
    if len(obj_points) < 3:
        raise ValueError(f"Need >= 3 images, got {len(obj_points)}")

    print(f"\nCalibrating with {len(obj_points)} images...")
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print(f"RMS error: {rms:.4f} pixels")

    return rms, mtx, dist


def save_model(path, mtx, dist, rms, img_size, num_images):
    """Save calibration to JSON."""
    data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "rms_error": float(rms),
        "image_size": list(img_size),
        "num_images_used": num_images,
        "board": {
            "squares_h": SQUARES_H,
            "squares_v": SQUARES_V,
            "square_size_m": SQUARE_SIZE,
            "marker_ratio": MARKER_RATIO
        }
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def main():
    print("=" * 60)
    print("Camera Calibration")
    print("=" * 60)

    # Create output directory for detections
    detections_dir = IMAGES_FOLDER / "detections"
    detections_dir.mkdir(exist_ok=True)

    # Create detector
    print(f"\nBoard: {SQUARES_H}x{SQUARES_V} squares")
    print(f"Square size: {SQUARE_SIZE*100:.1f}cm, Marker ratio: {MARKER_RATIO}")

    board, detector = create_detector()

    # Process images
    print(f"\nProcessing images from {IMAGES_FOLDER.name}/")
    obj_pts, img_pts, num_valid, img_size = process_images(
        IMAGES_FOLDER, board, detector, detections_dir
    )

    if num_valid < 3:
        print("ERROR: Need at least 3 valid images")
        return

    print(f"\nDetection figures saved to: {detections_dir}")

    # Calibrate
    print("\nCalibrating...")
    rms, mtx, dist = calibrate(obj_pts, img_pts, img_size)

    print(f"\nCamera Matrix:")
    print(f"  fx={mtx[0, 0]:.1f}, fy={mtx[1, 1]:.1f}")
    print(f"  cx={mtx[0, 2]:.1f}, cy={mtx[1, 2]:.1f}")
    print(f"\nDistortion: k1={dist[0, 0]:.6f}, k2={dist[0, 1]:.6f}, "
          f"k3={dist[0, 4]:.6f}")

    # Save model
    save_model(
        IMAGES_FOLDER / "camera_model.json",
        mtx, dist, rms, img_size, num_valid
    )

    print("\nDone!")

if __name__ == "__main__":
    main()
