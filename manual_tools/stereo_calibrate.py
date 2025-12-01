"""
Stereo Camera Calibration from ChArUco Images (LaVision IMS format)

This script:
1. Loads im7 files containing stereo image pairs (left + right cameras)
2. Detects ChArUco corners on both cameras
3. Saves detection visualizations
4. Performs stereo calibration
5. Tests rectification on sample images

Usage:
    python stereo_calibrate.py
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from pivtools_core.image_handling.readers.lavision_reader import read_lavision_im7

# =============================================================================
# CONFIGURATION - Edit these parameters
# =============================================================================

# Image source
IMAGES_FOLDER = (
    Path("manual_tools") / "MT3" / "Baseline_20mps_OFIDropConfig2_25-12-01_1311_02"
)

# Camera settings
LEFT_CAMERA = 1         # Camera number for left view (1-based)
RIGHT_CAMERA = 2        # Camera number for right view (1-based)
SINGLE_FRAME_MODE = True  # True: 1 frame/camera, False: 2 frames (A/B pair)

# ChArUco board parameters
SQUARES_H = 10          # Number of squares horizontally (columns)
SQUARES_V = 9           # Number of squares vertically (rows)
SQUARE_SIZE = 0.03      # Physical square size in meters
MARKER_RATIO = 0.5      # Marker size / square size
ARUCO_DICT = cv2.aruco.DICT_4X4_1000

# Detection settings
MIN_CORNERS = 6

# =============================================================================


def read_stereo_im7(file_path, left_cam, right_cam, single_frame=True):
    """
    Read left and right camera images from im7 file.

    Uses pivtools_core lavision reader which handles the im7 structure:
    - Each im7 contains all cameras for one time instance
    - Frame indexing depends on mode:
        - Single frame (1 per camera): cam1->frame0, cam2->frame1
        - Double frame (A/B pairs):    cam1->frames0,1, cam2->frames2,3

    Args:
        file_path: Path to .im7 file
        left_cam: Left camera number (1-based)
        right_cam: Right camera number (1-based)
        single_frame: If True, 1 frame per camera. If False, 2 (A/B pair).

    Returns:
        left_img, right_img: numpy arrays (H, W)
    """
    # frames_per_camera determines the stride in the im7 file
    # single_frame=True  -> 1 frame/camera -> cam1=frame0, cam2=frame1
    # single_frame=False -> 2 frames/camera -> cam1=frames0,1, cam2=frames2,3
    frames_per_camera = 1 if single_frame else 2

    # read_lavision_im7(file_path, camera_no, frames, frames_per_camera)
    # Returns shape (frames, H, W)
    left_data = read_lavision_im7(
        str(file_path),
        camera_no=left_cam,
        frames=1,  # We only need 1 frame for calibration
        frames_per_camera=frames_per_camera
    )
    right_data = read_lavision_im7(
        str(file_path),
        camera_no=right_cam,
        frames=1,
        frames_per_camera=frames_per_camera
    )

    # Return single frame from each camera
    left_img = left_data[0]
    right_img = right_data[0]

    return left_img, right_img


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


def to_8bit(image):
    """Convert image to 8-bit for OpenCV processing."""
    if image.dtype in (np.float32, np.float64):
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = ((image - img_min) / (img_max - img_min) * 255)
        else:
            image = np.zeros_like(image)
    elif image.dtype == np.uint16:
        image = image / 256

    return image.astype(np.uint8)


def detect_corners(image, detector):
    """Detect ChArUco corners in image."""
    img_8bit = to_8bit(image)

    if len(img_8bit.shape) == 3:
        gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_8bit

    corners, ids, marker_corners, _ = detector.detectBoard(gray)
    return corners, ids, marker_corners, img_8bit


def save_detection_figure(img_8bit, corners, ids, marker_corners,
                          name, output_dir):
    """Save detection visualization."""
    if len(img_8bit.shape) == 2:
        vis = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    else:
        vis = img_8bit.copy()

    if marker_corners is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners)

    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    n_corners = len(corners) if corners is not None else 0
    ax.set_title(f"{name} - {n_corners} corners")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def process_stereo_images(folder, board, detector, output_dir):
    """
    Process all im7 files and detect corners on both cameras.

    Returns:
        obj_points: List of 3D points (same for both cameras)
        img_points_left: List of 2D points from left camera
        img_points_right: List of 2D points from right camera
        img_size: (width, height)
        sample_pair: (left_img, right_img) for rectification test
    """
    im7_files = sorted(folder.glob("*.im7"))

    if not im7_files:
        raise ValueError(f"No .im7 files found in {folder}")

    print(f"Found {len(im7_files)} im7 files")
    print("-" * 60)

    obj_points = []
    img_points_left = []
    img_points_right = []
    img_size = None
    sample_pair = None

    stats = {'valid': 0, 'left_fail': 0, 'right_fail': 0, 'empty': 0}

    for im7_file in im7_files:
        try:
            left_img, right_img = read_stereo_im7(
                im7_file, LEFT_CAMERA, RIGHT_CAMERA, SINGLE_FRAME_MODE
            )
        except Exception as e:
            print(f"  {im7_file.name}: SKIP (read error: {e})")
            continue

        # Skip empty images
        if np.mean(left_img) < 10 or np.mean(right_img) < 10:
            print(f"  {im7_file.name}: SKIP (empty)")
            stats['empty'] += 1
            continue

        if img_size is None:
            h, w = left_img.shape[:2]
            img_size = (w, h)

        # Detect corners
        corners_L, ids_L, markers_L, img8_L = detect_corners(left_img, detector)
        corners_R, ids_R, markers_R, img8_R = detect_corners(right_img, detector)

        # Check left camera
        if ids_L is None or len(ids_L) < MIN_CORNERS:
            print(f"  {im7_file.name}: SKIP (left: insufficient corners)")
            stats['left_fail'] += 1
            continue

        # Check right camera
        if ids_R is None or len(ids_R) < MIN_CORNERS:
            print(f"  {im7_file.name}: SKIP (right: insufficient corners)")
            stats['right_fail'] += 1
            continue

        # Find common corner IDs (both cameras must see the same corners)
        common_ids = np.intersect1d(ids_L.flatten(), ids_R.flatten())
        if len(common_ids) < MIN_CORNERS:
            print(f"  {im7_file.name}: SKIP ({len(common_ids)} common corners)")
            continue

        # Filter to common corners only
        mask_L = np.isin(ids_L.flatten(), common_ids)
        mask_R = np.isin(ids_R.flatten(), common_ids)

        corners_L_common = corners_L[mask_L]
        corners_R_common = corners_R[mask_R]
        ids_common = ids_L[mask_L]

        # Match to object points
        obj_pts, img_pts_L = board.matchImagePoints(corners_L_common, ids_common)
        _, img_pts_R = board.matchImagePoints(corners_R_common, ids_common)

        if obj_pts is None or len(obj_pts) < MIN_CORNERS:
            print(f"  {im7_file.name}: SKIP (point matching failed)")
            continue

        obj_points.append(obj_pts)
        img_points_left.append(img_pts_L)
        img_points_right.append(img_pts_R)
        stats['valid'] += 1

        print(f"  {im7_file.name}: OK "
              f"(L:{len(corners_L)}, R:{len(corners_R)}, common:{len(common_ids)})")

        # Save detection figures
        save_detection_figure(
            img8_L, corners_L, ids_L, markers_L,
            f"{im7_file.stem}_left", output_dir
        )
        save_detection_figure(
            img8_R, corners_R, ids_R, markers_R,
            f"{im7_file.stem}_right", output_dir
        )

        # Keep first valid pair for rectification test
        if sample_pair is None:
            sample_pair = (img8_L, img8_R)

    print("-" * 60)
    print(f"Valid pairs: {stats['valid']}, Empty: {stats['empty']}, "
          f"Left fail: {stats['left_fail']}, Right fail: {stats['right_fail']}")

    return obj_points, img_points_left, img_points_right, img_size, sample_pair


def stereo_calibrate(obj_points, img_points_left, img_points_right, img_size):
    """
    Perform stereo calibration.

    Returns:
        Calibration results dict
    """
    if len(obj_points) < 3:
        raise ValueError(f"Need >= 3 valid stereo pairs, got {len(obj_points)}")

    print(f"\nCalibrating with {len(obj_points)} stereo pairs...")

    # First calibrate each camera individually
    print("  Calibrating left camera...")
    rms_L, mtx_L, dist_L, _, _ = cv2.calibrateCamera(
        obj_points, img_points_left, img_size, None, None
    )
    print(f"    Left RMS: {rms_L:.4f} px")

    print("  Calibrating right camera...")
    rms_R, mtx_R, dist_R, _, _ = cv2.calibrateCamera(
        obj_points, img_points_right, img_size, None, None
    )
    print(f"    Right RMS: {rms_R:.4f} px")

    # Stereo calibration
    print("  Stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC  # Use individual camera calibrations

    ret = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        mtx_L, dist_L,
        mtx_R, dist_R,
        img_size,
        flags=flags
    )
    rms_stereo, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F = ret

    print(f"  Stereo RMS: {rms_stereo:.4f} px")

    # Compute baseline (distance between cameras)
    baseline = np.linalg.norm(T)
    print(f"  Baseline: {baseline*1000:.1f} mm")

    return {
        'mtx_left': mtx_L,
        'dist_left': dist_L,
        'mtx_right': mtx_R,
        'dist_right': dist_R,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'rms_left': rms_L,
        'rms_right': rms_R,
        'rms_stereo': rms_stereo,
        'baseline': baseline,
        'img_size': img_size
    }


def compute_rectification(calib, img_size):
    """Compute stereo rectification maps."""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        calib['mtx_left'], calib['dist_left'],
        calib['mtx_right'], calib['dist_right'],
        img_size, calib['R'], calib['T'],
        alpha=0  # 0 = crop to valid pixels only
    )

    map1_L, map2_L = cv2.initUndistortRectifyMap(
        calib['mtx_left'], calib['dist_left'],
        R1, P1, img_size, cv2.CV_32FC1
    )

    map1_R, map2_R = cv2.initUndistortRectifyMap(
        calib['mtx_right'], calib['dist_right'],
        R2, P2, img_size, cv2.CV_32FC1
    )

    return {
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2,
        'Q': Q,
        'map_left': (map1_L, map2_L),
        'map_right': (map1_R, map2_R),
        'roi_left': roi1,
        'roi_right': roi2
    }


def rectify_images(left_img, right_img, rect_maps):
    """Apply rectification to image pair."""
    rect_L = cv2.remap(
        left_img,
        rect_maps['map_left'][0],
        rect_maps['map_left'][1],
        cv2.INTER_LINEAR
    )
    rect_R = cv2.remap(
        right_img,
        rect_maps['map_right'][0],
        rect_maps['map_right'][1],
        cv2.INTER_LINEAR
    )
    return rect_L, rect_R


def save_rectification_test(left_img, right_img, rect_maps, output_path):
    """
    Save rectification test figure.

    Shows original and rectified images with horizontal lines to verify
    that corresponding points are on the same row (epipolar constraint).
    """
    rect_L, rect_R = rectify_images(left_img, right_img, rect_maps)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original images
    axes[0, 0].imshow(left_img, cmap='gray')
    axes[0, 0].set_title("Original Left")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(right_img, cmap='gray')
    axes[0, 1].set_title("Original Right")
    axes[0, 1].axis('off')

    # Rectified images with epipolar lines
    axes[1, 0].imshow(rect_L, cmap='gray')
    axes[1, 0].set_title("Rectified Left")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(rect_R, cmap='gray')
    axes[1, 1].set_title("Rectified Right")
    axes[1, 1].axis('off')

    # Draw horizontal lines on rectified images
    h = rect_L.shape[0]
    for y in range(0, h, h // 10):
        axes[1, 0].axhline(y, color='lime', linewidth=0.5, alpha=0.7)
        axes[1, 1].axhline(y, color='lime', linewidth=0.5, alpha=0.7)

    fig.suptitle(
        "Stereo Rectification Test\n"
        "Green lines should align with same features in both images",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved rectification test: {output_path}")


def save_stereo_model(calib, output_path):
    """Save stereo calibration to JSON."""
    data = {
        "description": "Stereo camera calibration from ChArUco images",
        "left_camera": {
            "camera_matrix": calib['mtx_left'].tolist(),
            "distortion_coefficients": calib['dist_left'].tolist(),
            "rms_error": float(calib['rms_left'])
        },
        "right_camera": {
            "camera_matrix": calib['mtx_right'].tolist(),
            "distortion_coefficients": calib['dist_right'].tolist(),
            "rms_error": float(calib['rms_right'])
        },
        "stereo": {
            "rotation_matrix": calib['R'].tolist(),
            "translation_vector": calib['T'].tolist(),
            "essential_matrix": calib['E'].tolist(),
            "fundamental_matrix": calib['F'].tolist(),
            "rms_error": float(calib['rms_stereo']),
            "baseline_m": float(calib['baseline'])
        },
        "image_size": list(calib['img_size']),
        "board": {
            "squares_h": SQUARES_H,
            "squares_v": SQUARES_V,
            "square_size_m": SQUARE_SIZE,
            "marker_ratio": MARKER_RATIO
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved stereo model: {output_path}")


def main():
    print("=" * 60)
    print("Stereo Camera Calibration")
    print("=" * 60)

    # Create output directory
    output_dir = IMAGES_FOLDER / "stereo_calibration"
    output_dir.mkdir(exist_ok=True)

    detections_dir = output_dir / "detections"
    detections_dir.mkdir(exist_ok=True)

    # Print config
    print("\nConfiguration:")
    print(f"  Images folder: {IMAGES_FOLDER}")
    print(f"  Left camera: {LEFT_CAMERA}, Right camera: {RIGHT_CAMERA}")
    print(f"  Single frame mode: {SINGLE_FRAME_MODE}")
    print(f"  Board: {SQUARES_H}x{SQUARES_V} squares")

    # Create detector
    print("\nCreating ChArUco detector...")
    board, detector = create_detector()

    # Process images
    print("\nProcessing stereo images...")
    results = process_stereo_images(
        IMAGES_FOLDER, board, detector, detections_dir
    )
    obj_pts, img_pts_L, img_pts_R, img_size, sample_pair = results

    if len(obj_pts) < 3:
        print("\nERROR: Need at least 3 valid stereo pairs")
        return

    print(f"\nDetection figures saved to: {detections_dir}")

    # Stereo calibration
    print("\n" + "=" * 60)
    print("Stereo Calibration")
    print("=" * 60)

    calib = stereo_calibrate(obj_pts, img_pts_L, img_pts_R, img_size)

    # Print results
    print("\n--- Left Camera ---")
    print(f"  fx={calib['mtx_left'][0, 0]:.1f}, "
          f"fy={calib['mtx_left'][1, 1]:.1f}")
    print(f"  cx={calib['mtx_left'][0, 2]:.1f}, "
          f"cy={calib['mtx_left'][1, 2]:.1f}")

    print("\n--- Right Camera ---")
    print(f"  fx={calib['mtx_right'][0, 0]:.1f}, "
          f"fy={calib['mtx_right'][1, 1]:.1f}")
    print(f"  cx={calib['mtx_right'][0, 2]:.1f}, "
          f"cy={calib['mtx_right'][1, 2]:.1f}")

    print("\n--- Stereo Geometry ---")
    print(f"  Baseline: {calib['baseline']*1000:.2f} mm")
    print(f"  Translation: "
          f"[{calib['T'][0, 0]:.4f}, {calib['T'][1, 0]:.4f}, "
          f"{calib['T'][2, 0]:.4f}] m")

    # Save model
    save_stereo_model(calib, output_dir / "stereo_model.json")

    # Rectification test
    if sample_pair is not None:
        print("\n" + "=" * 60)
        print("Rectification Test")
        print("=" * 60)

        rect_maps = compute_rectification(calib, img_size)
        save_rectification_test(
            sample_pair[0], sample_pair[1],
            rect_maps,
            output_dir / "rectification_test.png"
        )

    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
