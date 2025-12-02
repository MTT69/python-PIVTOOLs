#!/usr/bin/env python3
"""
charuco_calibration_production.py

Production-ready ChArUco board calibration for camera intrinsic parameters.
Uses OpenCV's ChArUco detection with multi-image aggregation for robust calibration.

This module provides:
- ChArUco board detection in calibration images
- Multi-image camera calibration using cv2.calibrateCamera
- Compatible output format with PlanarCalibrator for VectorCalibrator use
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from pivtools_core.image_handling.load_images import read_image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Standard ArUco dictionaries mapping
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


class ChArUcoCalibrator:
    """
    ChArUco board camera calibration.

    Uses ChArUco boards (combined chessboard + ArUco markers) for robust
    camera intrinsic calibration. Aggregates corner detections across
    multiple images for improved accuracy.

    Attributes:
        squares_h: Number of squares horizontally (columns)
        squares_v: Number of squares vertically (rows)
        square_size: Physical square size in meters
        marker_ratio: Marker size / square size (typically 0.5)
        aruco_dict: ArUco dictionary type (e.g., "DICT_4X4_1000")
        min_corners: Minimum corners required per image
    """

    def __init__(
        self,
        source_dir: Path,
        base_dir: Path,
        camera_count: int = 1,
        file_pattern: str = "*.tif",
        squares_h: int = 10,
        squares_v: int = 9,
        square_size: float = 0.03,
        marker_ratio: float = 0.5,
        aruco_dict: str = "DICT_4X4_1000",
        min_corners: int = 6,
        dt: float = 1.0,
    ):
        """
        Initialize ChArUco calibrator.

        Args:
            source_dir: Source directory containing calibration subdirectory
            base_dir: Base output directory for results
            camera_count: Number of cameras to process
            file_pattern: Glob pattern for calibration images
            squares_h: Number of squares horizontally (columns)
            squares_v: Number of squares vertically (rows)
            square_size: Physical square size in meters
            marker_ratio: Marker size relative to square size (typically 0.5)
            aruco_dict: ArUco dictionary name (e.g., "DICT_4X4_1000")
            min_corners: Minimum corners required to use an image
            dt: Time step between frames in seconds (for downstream use)
        """
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.camera_count = camera_count
        self.file_pattern = file_pattern
        self.squares_h = squares_h
        self.squares_v = squares_v
        self.square_size = square_size
        self.marker_ratio = marker_ratio
        self.aruco_dict_name = aruco_dict
        self.min_corners = min_corners
        self.dt = dt

        # Create board and detector
        self.board, self.detector = self._create_detector()

        # Setup output directories
        self._setup_directories()

    def _create_detector(self) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
        """
        Create ChArUco board and detector.

        Returns:
            Tuple of (CharucoBoard, CharucoDetector)
        """
        marker_size = self.square_size * self.marker_ratio
        dict_id = ARUCO_DICT_MAP.get(self.aruco_dict_name, cv2.aruco.DICT_4X4_1000)
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        board = cv2.aruco.CharucoBoard(
            (self.squares_h, self.squares_v),
            self.square_size,
            marker_size,
            dictionary,
        )

        detector = cv2.aruco.CharucoDetector(
            board,
            cv2.aruco.CharucoParameters(),
            cv2.aruco.DetectorParameters(),
        )

        return board, detector

    def _setup_directories(self):
        """Create necessary output directories."""
        for cam_num in range(1, self.camera_count + 1):
            cam_base = self.base_dir / "calibration" / f"Cam{cam_num}"
            (cam_base / "detections").mkdir(parents=True, exist_ok=True)
            (cam_base / "model").mkdir(parents=True, exist_ok=True)

    def _is_container_format(self) -> bool:
        """Check if file pattern is a container format (.set, .im7)."""
        return ".set" in self.file_pattern.lower() or ".im7" in self.file_pattern.lower()

    def _read_calibration_image(
        self, img_path: Path, camera: int = 1, img_index: int = 1
    ) -> Optional[np.ndarray]:
        """
        Read calibration image with container format support.

        Args:
            img_path: Path to image file or container
            camera: Camera number (1-based, for container formats)
            img_index: Image index (1-based, for .set files)

        Returns:
            Image data as uint8, or None if read failed
        """
        try:
            if self._is_container_format():
                if ".set" in str(img_path).lower():
                    img = read_image(str(img_path), camera_no=camera, im_no=img_index)
                elif ".im7" in str(img_path).lower():
                    img = read_image(str(img_path), camera_no=camera)
                else:
                    img = read_image(str(img_path))
            else:
                img = read_image(str(img_path))

            if img is None:
                return None

            # Normalize to uint8
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype in [np.float32, np.float64]:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            elif img.dtype == np.bool_:
                img = img.astype(np.uint8) * 255

            return img

        except Exception as e:
            logger.warning(f"Failed to read image {img_path}: {e}")
            return None

    def _find_calibration_images(self, cam_input_dir: Path) -> List[Path]:
        """
        Find all calibration images matching the pattern.

        Args:
            cam_input_dir: Directory to search

        Returns:
            List of image paths
        """
        if self._is_container_format():
            container_file = cam_input_dir / self.file_pattern
            if container_file.exists():
                return [container_file]
            return []

        # Glob pattern matching
        if "*" in self.file_pattern or "?" in self.file_pattern:
            return sorted(cam_input_dir.glob(self.file_pattern))

        # Numbered pattern (e.g., "calib%05d.tif")
        if "%" in self.file_pattern:
            files = []
            i = 1
            while True:
                try:
                    filename = self.file_pattern % i
                except TypeError:
                    break
                filepath = cam_input_dir / filename
                if filepath.exists():
                    files.append(filepath)
                    i += 1
                else:
                    break
            return files

        # Single file
        single = cam_input_dir / self.file_pattern
        return [single] if single.exists() else []

    def detect_charuco_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco corners in an image.

        Args:
            image: Input image (grayscale or color)

        Returns:
            Tuple of (found, corners, corner_ids, marker_corners, marker_ids)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect board
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        if ids is None or len(corners) < self.min_corners:
            return False, None, None, marker_corners, marker_ids

        return True, corners, ids, marker_corners, marker_ids

    def _save_detection_visualization(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        ids: np.ndarray,
        marker_corners: Optional[np.ndarray],
        filename: str,
        output_dir: Path,
    ):
        """
        Save visualization of detected corners.

        Args:
            image: Original image
            corners: Detected ChArUco corners
            ids: Corner IDs
            marker_corners: Detected marker corners
            filename: Base filename for output
            output_dir: Directory to save visualization
        """
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

        # Create and save figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{filename} - {len(corners)} corners detected")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}_detection.png", dpi=150)
        plt.close(fig)

    def process_camera(
        self,
        cam_num: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process all calibration images for one camera.

        Aggregates corner detections across all valid images and runs
        a single cv2.calibrateCamera call for robust calibration.

        Args:
            cam_num: Camera number (1-based)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with calibration results
        """
        logger.info(f"Processing Camera {cam_num}")

        # Setup paths
        is_container = self._is_container_format()
        if is_container:
            cam_input_dir = self.source_dir / "calibration"
        else:
            cam_input_dir = self.source_dir / "calibration" / f"Cam{cam_num}"

        cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}"
        detections_dir = cam_output_base / "detections"

        if not cam_input_dir.exists():
            logger.error(f"Calibration directory not found: {cam_input_dir}")
            return {"success": False, "error": "Directory not found"}

        # Find images
        image_files = self._find_calibration_images(cam_input_dir)
        if not image_files:
            logger.error(f"No calibration images found in {cam_input_dir}")
            return {"success": False, "error": "No images found"}

        logger.info(f"Found {len(image_files)} images")

        # Collect detections across all images
        all_obj_points = []
        all_img_points = []
        img_size = None
        stats = {"empty": 0, "no_detect": 0, "valid": 0}
        valid_images = []

        total_images = len(image_files) if not is_container else 100  # Estimate for containers
        processed = 0

        for idx, img_path in enumerate(image_files):
            # For container formats, iterate through images
            if is_container:
                # Process up to 100 images from container
                for img_idx in range(1, 101):
                    image = self._read_calibration_image(img_path, camera=cam_num, img_index=img_idx)
                    if image is None:
                        break

                    success = self._process_single_image(
                        image,
                        f"{img_path.stem}_img{img_idx:03d}",
                        all_obj_points,
                        all_img_points,
                        valid_images,
                        stats,
                        detections_dir,
                    )

                    if img_size is None and image is not None:
                        h, w = image.shape[:2]
                        img_size = (w, h)

                    processed += 1
                    if progress_callback:
                        progress_callback({
                            "camera": cam_num,
                            "processed_images": processed,
                            "valid_images": stats["valid"],
                            "progress": min(95, int((processed / total_images) * 100)),
                        })
            else:
                image = self._read_calibration_image(img_path, camera=cam_num, img_index=idx + 1)
                if image is None:
                    continue

                if img_size is None:
                    h, w = image.shape[:2]
                    img_size = (w, h)

                self._process_single_image(
                    image,
                    img_path.stem,
                    all_obj_points,
                    all_img_points,
                    valid_images,
                    stats,
                    detections_dir,
                )

                processed += 1
                if progress_callback:
                    progress_callback({
                        "camera": cam_num,
                        "processed_images": processed,
                        "total_images": len(image_files),
                        "valid_images": stats["valid"],
                        "progress": int((processed / len(image_files)) * 95),
                    })

        logger.info(
            f"Valid: {stats['valid']}, Empty: {stats['empty']}, No detection: {stats['no_detect']}"
        )

        if stats["valid"] < 3:
            logger.error(f"Need at least 3 valid images, got {stats['valid']}")
            return {
                "success": False,
                "error": f"Insufficient valid images ({stats['valid']})",
                "stats": stats,
            }

        # Run calibration with aggregated points
        logger.info(f"Calibrating with {len(all_obj_points)} images...")

        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points, all_img_points, img_size, None, None
        )

        logger.info(f"RMS reprojection error: {rms:.4f} pixels")

        # Calculate per-axis errors
        all_errors = []
        all_errors_x = []
        all_errors_y = []

        for i in range(len(all_obj_points)):
            proj, _ = cv2.projectPoints(
                all_obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            proj = proj.reshape(-1, 2)
            img_pts = all_img_points[i].reshape(-1, 2)
            err_vec = img_pts - proj
            all_errors.extend(np.linalg.norm(err_vec, axis=1).tolist())
            all_errors_x.extend(err_vec[:, 0].tolist())
            all_errors_y.extend(err_vec[:, 1].tolist())

        # Save camera model (compatible with VectorCalibrator)
        model_data = {
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "rvecs": np.array([r.flatten() for r in rvecs]),
            "tvecs": np.array([t.flatten() for t in tvecs]),
            "reprojection_error": rms,
            "reprojection_error_x_mean": float(np.mean(np.abs(all_errors_x))),
            "reprojection_error_y_mean": float(np.mean(np.abs(all_errors_y))),
            "reprojection_errors": np.array(all_errors),
            "reprojection_errors_x": np.array(all_errors_x),
            "reprojection_errors_y": np.array(all_errors_y),
            "num_images": stats["valid"],
            "image_size": list(img_size),
            "timestamp": datetime.now().isoformat(),
            "dt": self.dt,
            "board_params": {
                "squares_h": self.squares_h,
                "squares_v": self.squares_v,
                "square_size": self.square_size,
                "marker_ratio": self.marker_ratio,
                "aruco_dict": self.aruco_dict_name,
            },
        }

        model_path = cam_output_base / "model" / "camera_model.mat"
        savemat(str(model_path), model_data)
        logger.info(f"Saved camera model: {model_path}")

        # Also save as JSON for easy inspection
        json_data = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "rms_error": float(rms),
            "image_size": list(img_size),
            "num_images_used": stats["valid"],
            "board": {
                "squares_h": self.squares_h,
                "squares_v": self.squares_v,
                "square_size_m": self.square_size,
                "marker_ratio": self.marker_ratio,
                "aruco_dict": self.aruco_dict_name,
            },
        }

        json_path = cam_output_base / "model" / "camera_model.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        if progress_callback:
            progress_callback({
                "camera": cam_num,
                "processed_images": processed,
                "valid_images": stats["valid"],
                "progress": 100,
            })

        return {
            "success": True,
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.tolist(),
            "rms_error": float(rms),
            "num_images_used": stats["valid"],
            "stats": stats,
            "model_path": str(model_path),
        }

    def _process_single_image(
        self,
        image: np.ndarray,
        name: str,
        all_obj_points: List,
        all_img_points: List,
        valid_images: List,
        stats: Dict,
        detections_dir: Path,
    ) -> bool:
        """
        Process a single calibration image.

        Args:
            image: Image data
            name: Image name for logging
            all_obj_points: List to append 3D object points
            all_img_points: List to append 2D image points
            valid_images: List to track valid image names
            stats: Dictionary to update statistics
            detections_dir: Directory to save detection visualizations

        Returns:
            True if image was valid and added to calibration
        """
        # Skip empty/black images
        if np.mean(image) < 10:
            logger.debug(f"  {name}: SKIP (empty)")
            stats["empty"] += 1
            return False

        # Detect corners
        found, corners, ids, marker_corners, marker_ids = self.detect_charuco_corners(image)

        if not found:
            logger.debug(f"  {name}: SKIP (insufficient corners)")
            stats["no_detect"] += 1
            return False

        # Match to 3D object points
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)

        if obj_pts is None or len(obj_pts) < self.min_corners:
            stats["no_detect"] += 1
            return False

        all_obj_points.append(obj_pts)
        all_img_points.append(img_pts)
        valid_images.append(name)
        stats["valid"] += 1

        logger.info(f"  {name}: OK ({len(corners)} corners)")

        # Save detection visualization
        self._save_detection_visualization(
            image, corners, ids, marker_corners, name, detections_dir
        )

        return True

    def process_all_cameras(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process all cameras.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with results for all cameras
        """
        results = {
            "total_cameras": self.camera_count,
            "processed_cameras": 0,
            "camera_results": {},
        }

        for cam_num in range(1, self.camera_count + 1):
            def camera_progress(data):
                if progress_callback:
                    progress_callback({
                        "current_camera": cam_num,
                        "total_cameras": self.camera_count,
                        "processed_cameras": results["processed_cameras"],
                        **data,
                    })

            result = self.process_camera(cam_num, progress_callback=camera_progress)
            results["camera_results"][cam_num] = result
            results["processed_cameras"] += 1

        return results

    def run(self):
        """Run calibration for all cameras (CLI entry point)."""
        logger.info(f"Starting ChArUco calibration for {self.camera_count} cameras")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Board: {self.squares_h}x{self.squares_v} squares")
        logger.info(f"Square size: {self.square_size * 100:.1f}cm")
        logger.info(f"Marker ratio: {self.marker_ratio}")
        logger.info(f"ArUco dict: {self.aruco_dict_name}")

        for cam_num in range(1, self.camera_count + 1):
            try:
                self.process_camera(cam_num)
            except Exception as e:
                logger.error(f"Failed to process Camera {cam_num}: {e}")
                continue

        logger.info("ChArUco calibration completed")


def main():
    """CLI entry point with example configuration."""
    import argparse

    parser = argparse.ArgumentParser(description="ChArUco board camera calibration")
    parser.add_argument("--source", "-s", required=True, help="Source directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--cameras", "-c", type=int, default=1, help="Number of cameras")
    parser.add_argument("--pattern", "-p", default="*.tif", help="File pattern")
    parser.add_argument("--squares-h", type=int, default=10, help="Squares horizontally")
    parser.add_argument("--squares-v", type=int, default=9, help="Squares vertically")
    parser.add_argument("--square-size", type=float, default=0.03, help="Square size (meters)")
    parser.add_argument("--marker-ratio", type=float, default=0.5, help="Marker ratio")
    parser.add_argument("--aruco-dict", default="DICT_4X4_1000", help="ArUco dictionary")
    parser.add_argument("--min-corners", type=int, default=6, help="Minimum corners per image")

    args = parser.parse_args()

    calibrator = ChArUcoCalibrator(
        source_dir=args.source,
        base_dir=args.output,
        camera_count=args.cameras,
        file_pattern=args.pattern,
        squares_h=args.squares_h,
        squares_v=args.squares_v,
        square_size=args.square_size,
        marker_ratio=args.marker_ratio,
        aruco_dict=args.aruco_dict,
        min_corners=args.min_corners,
    )

    calibrator.run()


if __name__ == "__main__":
    main()
