#!/usr/bin/env python3
"""
charuco_calibration_production.py

Production-ready ChArUco board calibration for camera intrinsic parameters.
Uses OpenCV's ChArUco detection with multi-image aggregation for robust calibration.
Saves results to: {BASE_DIR}/calibration/Cam{N}/charuco_planar/
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.io import savemat

from pivtools_core.config import get_config, reload_config
from pivtools_gui.calibration.calibration_io import (
    ARUCO_DICT_MAP,
    is_container_format,
    read_calibration_image_with_fallback,
    read_calibration_image_direct,
    find_calibration_images,
    get_camera_input_dir,
    create_charuco_detector,
)

# ===================== CONFIGURATION VARIABLES =====================

# -------------------- PATH CONFIGURATION --------------------
# SOURCE_DIR: Root directory containing your data.
SOURCE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/Cam1"

# BASE_DIR: The output directory where calibration results will be saved.
#           Results are saved to: {BASE_DIR}/calibration/Cam{N}/charuco_planar/...
BASE_DIR = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/test"

# CALIBRATION_SUBFOLDER: Subfolder within the source path for calibration images.
#                        Leave empty "" to look directly in SOURCE_DIR.
CALIBRATION_SUBFOLDER = ""

# -------------------- CAMERA CONFIGURATION --------------------
# CAMERA_NUMS: List of camera numbers to process (1-based), e.g. [1, 2] for stereo
CAMERA_NUMS = [1]

# CAMERA_SUBFOLDERS: List of subfolder names for each camera (index matches camera number - 1).
#                    e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
#                    Set to [] (empty list) for container formats or when images are in SOURCE_DIR directly.
CAMERA_SUBFOLDERS = []

# FILE_PATTERN: The naming pattern for calibration images (e.g., "calib%05d.tif" or "*.tif")
FILE_PATTERN = "calib%05d.tif"

# -------------------- CHARUCO BOARD SETTINGS --------------------
# These must match your physical calibration target exactly.
SQUARES_H = 10              # Number of squares horizontally
SQUARES_V = 9               # Number of squares vertically
SQUARE_SIZE_M = 0.03        # Physical square size in METERS
MARKER_RATIO = 0.5          # Ratio of marker size to square size (usually 0.5)
ARUCO_DICT = "DICT_4X4_1000" # ArUco dictionary used
MIN_CORNERS = 6             # Minimum number of corners required to accept an image

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# ===================================================================


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

    # ChArUco-specific params
    config.data["calibration"]["charuco"]["squares_h"] = SQUARES_H
    config.data["calibration"]["charuco"]["squares_v"] = SQUARES_V
    config.data["calibration"]["charuco"]["square_size"] = SQUARE_SIZE_M
    config.data["calibration"]["charuco"]["marker_ratio"] = MARKER_RATIO
    config.data["calibration"]["charuco"]["aruco_dict"] = ARUCO_DICT
    config.data["calibration"]["charuco"]["min_corners"] = MIN_CORNERS

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


class ChArUcoCalibrator:
    def __init__(
        self,
        source_dir,
        base_dir,
        camera_count=1,
        file_pattern="*.tif",
        squares_h=10,
        squares_v=9,
        square_size=0.03,
        marker_ratio=0.5,
        aruco_dict="DICT_4X4_1000",
        min_corners=6,
        dt=1.0,
        calibration_input_path=None,
        config=None,
        model_type="pinhole",
        source_path_idx: int = 0,
    ):
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
        self.calibration_input_path = Path(calibration_input_path) if calibration_input_path else None
        self._config = config
        self.model_type = model_type
        self.source_path_idx = source_path_idx

        # Create board and detector
        self.board, self.detector = self._create_detector()

        # Setup output directories
        self._setup_directories()

    def _create_detector(self) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
        return create_charuco_detector(
            self.squares_h, self.squares_v, self.square_size,
            self.marker_ratio, self.aruco_dict_name,
        )

    def _setup_directories(self):
        """Create necessary output directories including charuco_planar subfolder."""
        for cam_num in range(1, self.camera_count + 1):
            # Output path: .../calibration/CamX/charuco_planar/...
            cam_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "charuco_planar"
            (cam_base / "detections").mkdir(parents=True, exist_ok=True)
            (cam_base / "model").mkdir(parents=True, exist_ok=True)
            (cam_base / "indices").mkdir(parents=True, exist_ok=True)

    def _get_camera_input_dir(self, cam_num: int) -> Path:
        return get_camera_input_dir(
            cam_num, self._config, self.source_path_idx,
            self.source_dir, self.calibration_input_path,
        )

    def _is_container_format(self) -> bool:
        return is_container_format(self._config, self.file_pattern)

    def _read_calibration_image(
        self, img_path: Path = None, camera: int = 1, img_index: int = 1
    ) -> Optional[np.ndarray]:
        return read_calibration_image_with_fallback(
            img_path, camera, img_index, self._config,
            self.source_path_idx, self.file_pattern,
        )

    def _read_calibration_image_direct(
        self, img_path: Path, camera: int = 1, img_index: int = 1
    ) -> Optional[np.ndarray]:
        return read_calibration_image_direct(
            img_path, camera, img_index, self.file_pattern, self._config,
        )

    def _find_calibration_images(self, cam_input_dir: Path) -> List[Path]:
        return find_calibration_images(cam_input_dir, self.file_pattern, self._config)

    def detect_charuco_corners(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ChArUco corners in an image."""
        if len(image.shape) == 3:
            if image.shape[-1] == 1:
                gray = image[:, :, 0]  # Squeeze singleton channel
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # ArUco detector requires uint8 input
        if gray.dtype != np.uint8:
            gmin, gmax = float(gray.min()), float(gray.max())
            if gmax > gmin:
                gray = ((gray.astype(np.float64) - gmin) / (gmax - gmin) * 255).astype(np.uint8)
            else:
                gray = np.zeros(gray.shape, dtype=np.uint8)

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
        """Save visualization of detected corners."""
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners)

        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

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
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all calibration images for one camera.

        Args:
            cam_num: Camera number to process
            progress_callback: Optional callback function receiving progress dict with:
                - processed_images: int
                - valid_images: int
                - total_images: int
                - progress: int (0-100)
            save_visualizations: Whether to save detection visualization PNGs

        Returns:
            Dict with success status and calibration results:
                - success: bool
                - camera_matrix: list (if success)
                - dist_coeffs: list (if success)
                - rms_error: float (if success)
                - num_images_used: int (if success)
                - model_path: str (if success)
                - error: str (if not success)
        """
        logger.info(f"Processing Camera {cam_num}")

        is_container = self._is_container_format()
        cam_input_dir = self._get_camera_input_dir(cam_num)

        # Output path structure: .../CamX/charuco_planar/...
        cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "charuco_planar"
        detections_dir = cam_output_base / "detections"
        indices_dir = cam_output_base / "indices"

        # Ensure directories exist
        detections_dir.mkdir(parents=True, exist_ok=True)
        indices_dir.mkdir(parents=True, exist_ok=True)
        (cam_output_base / "model").mkdir(parents=True, exist_ok=True)

        if not cam_input_dir.exists():
            error_msg = f"Calibration directory not found: {cam_input_dir}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        image_files = self._find_calibration_images(cam_input_dir)
        if not image_files:
            error_msg = f"No calibration images found in {cam_input_dir}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        logger.info(f"Found {len(image_files)} images (or container files)")

        # Limit to num_images from config (if set)
        max_images = None
        if self._config:
            max_images = self._config.data.get("calibration", {}).get("num_images")
        if max_images and max_images > 0 and not is_container:
            if len(image_files) > max_images:
                logger.info(f"Limiting to {max_images} of {len(image_files)} images (from config num_images)")
                image_files = image_files[:max_images]

        all_obj_points = []
        all_img_points = []
        img_size = None
        stats = {"empty": 0, "no_detect": 0, "valid": 0}
        valid_images = []
        best_image = None         # For model summary figure background
        best_image_npts = 0

        # Store per-frame detection data for indices saving
        # Key: frame index (1-based), Value: dict with corners, ids, filename
        valid_indices_map: Dict[int, Dict[str, Any]] = {}

        # Count total images for progress tracking
        total_images = len(image_files)
        if is_container:
            # Limit container frames if num_images set
            total_images = max_images if max_images and max_images > 0 else 100

        processed_count = 0

        # --- Worker: read + detect a single frame (no shared state mutation) ---
        def _detect_one_charuco(frame_idx, img_path, img_name):
            """Returns (frame_idx, image, detection_result_or_None, status)."""
            image = self._read_calibration_image(img_path, camera=cam_num, img_index=frame_idx)
            if image is None:
                return frame_idx, None, None, "none"

            if np.mean(image) < 10:
                return frame_idx, image, None, "empty"

            found, corners, ids, marker_corners, marker_ids = self.detect_charuco_corners(image)
            if not found:
                return frame_idx, image, None, "no_detect"

            obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
            if obj_pts is None or len(obj_pts) < self.min_corners:
                return frame_idx, image, None, "no_detect"

            corners_2d = corners.reshape(-1, 2) if corners is not None else np.array([])
            ids_flat = ids.flatten() if ids is not None else np.array([])

            # Save detection visualization in worker (thread-safe file I/O)
            if save_visualizations and detections_dir is not None:
                self._save_detection_visualization(
                    image, corners, ids, marker_corners, img_name, detections_dir
                )

            return frame_idx, image, {
                "corners": corners_2d,
                "ids": ids_flat,
                "name": img_name,
                "obj_pts": obj_pts,
                "img_pts": img_pts,
            }, "valid"

        # --- For containers, discover frame count first ---
        if is_container:
            container_count = 0
            container_limit = max_images if max_images and max_images > 0 else 100
            for probe_idx in range(1, container_limit + 1):
                probe_img = self._read_calibration_image(image_files[0], camera=cam_num, img_index=probe_idx)
                if probe_img is None:
                    break
                container_count += 1
                if img_size is None:
                    h, w = probe_img.shape[:2]
                    img_size = (w, h)
            total_images = container_count
            logger.info(f"Container has {container_count} frames")

        # --- Build work items ---
        work_items = []
        if is_container:
            for frame_idx in range(1, total_images + 1):
                work_items.append((frame_idx, image_files[0], f"{image_files[0].stem}_img{frame_idx:03d}"))
        else:
            for idx, img_path in enumerate(image_files):
                work_items.append((idx + 1, img_path, img_path.stem))

        # --- Parallel detection ---
        max_workers = min(os.cpu_count() or 4, len(work_items), 8)
        raw_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_detect_one_charuco, fi, ip, nm): fi
                for fi, ip, nm in work_items
            }
            for future in as_completed(futures):
                frame_idx, image, det_result, status = future.result()
                raw_results[frame_idx] = (image, det_result, status)
                processed_count += 1
                valid_count_tmp = sum(1 for _, _, s in raw_results.values() if s == "valid")
                if progress_callback:
                    progress = int((processed_count / max(total_images, 1)) * 90)
                    progress_callback({
                        "processed_images": processed_count,
                        "valid_images": valid_count_tmp,
                        "total_images": total_images,
                        "progress": min(progress, 90),
                    })

        # --- Collect results in frame order ---
        for frame_idx in sorted(raw_results.keys()):
            image, det_result, status = raw_results[frame_idx]

            if status == "empty":
                stats["empty"] += 1
            elif status == "no_detect":
                stats["no_detect"] += 1
            elif status == "valid" and det_result is not None:
                all_obj_points.append(det_result["obj_pts"])
                all_img_points.append(det_result["img_pts"])
                valid_images.append(det_result["name"])
                stats["valid"] += 1

                entry = {"corners": det_result["corners"], "ids": det_result["ids"], "name": det_result["name"]}
                if not is_container:
                    # Find original filename from work_items
                    for fi, ip, nm in work_items:
                        if fi == frame_idx:
                            entry["original_filename"] = Path(ip).name
                            break
                valid_indices_map[frame_idx] = entry

                n_pts = len(det_result["corners"])
                if n_pts > best_image_npts:
                    best_image = image.copy() if image is not None else None
                    best_image_npts = n_pts

                if img_size is None and image is not None:
                    h, w = image.shape[:2]
                    img_size = (w, h)

                logger.info(f"  {det_result['name']}: OK ({n_pts} corners)")

                # Generate per-frame detection figure
                try:
                    from pivtools_gui.calibration.calibration_figures import make_charuco_detection_figure
                    figures_dir = cam_output_base / "figures"
                    figures_dir.mkdir(parents=True, exist_ok=True)
                    board_params = {
                        "squares_h": self.squares_h,
                        "squares_v": self.squares_v,
                        "square_size_mm": self.square_size * 1000.0,
                        "marker_ratio": self.marker_ratio,
                    }
                    make_charuco_detection_figure(
                        image, det_result["corners"], det_result["ids"],
                        board_params,
                        figures_dir / f"detection_{frame_idx:03d}.png",
                        title=det_result["name"],
                    )
                except Exception:
                    pass

        logger.info(
            f"Valid: {stats['valid']}, Empty: {stats['empty']}, No detection: {stats['no_detect']}"
        )

        if stats["valid"] < 1:
            error_msg = f"No valid images found (0 of {total_images})"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Save per-frame indices (shared by both model types)
        logger.info(f"Saving per-frame detection indices for {len(valid_indices_map)} frames...")
        for frame_idx, detection_data in valid_indices_map.items():
            indices_data = {
                "corners": detection_data["corners"],
                "corner_ids": detection_data["ids"],
                "corner_count": len(detection_data["corners"]),
                "original_filename": detection_data.get("original_filename", ""),
                "frame_index": frame_idx,
                "board_params": {
                    "squares_h": self.squares_h,
                    "squares_v": self.squares_v,
                    "square_size": self.square_size,
                    "square_size_mm": self.square_size * 1000.0,
                    "marker_ratio": self.marker_ratio,
                    "aruco_dict": self.aruco_dict_name,
                },
            }
            indices_file = indices_dir / f"indexing_{frame_idx}.mat"
            savemat(str(indices_file), indices_data)

        # Run calibration
        logger.info(f"Calibrating with {len(all_obj_points)} images (model_type={self.model_type})...")

        if self.model_type == "polynomial":
            # --- POLYNOMIAL MODEL FITTING ---
            from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
                fit_polynomial_from_points, save_polynomial_to_config
            )

            # Stack all detected points into flat arrays
            all_img_flat = np.vstack([pts.reshape(-1, 2) for pts in all_img_points])
            all_obj_flat = np.vstack([pts.reshape(-1, 3)[:, :2] for pts in all_obj_points])
            all_obj_flat *= 1000.0  # meters -> mm

            try:
                fit_result = fit_polynomial_from_points(all_img_flat, all_obj_flat, img_size)
            except Exception as e:
                return {"success": False, "error": f"Polynomial fitting failed: {e}"}

            logger.info(f"Polynomial fit complete. RMS error: {fit_result['rms_fit_error_px']:.4f} px")

            # Save polynomial coefficients to config
            # Read dt from config (consistent with dotboard detector),
            # falling back to constructor arg
            dt = self._config.data.get("calibration", {}).get("charuco", {}).get("dt", self.dt) if self._config else self.dt
            if self._config:
                save_polynomial_to_config(cam_num, fit_result, dt, config=self._config)

            # Save polynomial model .mat
            poly_model_data = {
                "model_type": "polynomial",
                "mm_per_pixel": fit_result["mm_per_pixel"],
                "origin_x": fit_result["origin"]["x"],
                "origin_y": fit_result["origin"]["y"],
                "normalisation_nx": fit_result["normalisation"]["nx"],
                "normalisation_ny": fit_result["normalisation"]["ny"],
                "coefficients_x": np.array(fit_result["coefficients_x"]),
                "coefficients_y": np.array(fit_result["coefficients_y"]),
                "rms_fit_error_px": fit_result["rms_fit_error_px"],
                "num_images": stats["valid"],
                "image_size": np.array([img_size[0], img_size[1]]),
                "image_width": img_size[0],
                "image_height": img_size[1],
                "timestamp": datetime.now().isoformat(),
                "dt": self.dt,
                "dot_spacing_mm": self.square_size * 1000.0,
                "board_params": {
                    "squares_h": self.squares_h,
                    "squares_v": self.squares_v,
                    "square_size": self.square_size,
                    "square_size_mm": self.square_size * 1000.0,
                    "marker_ratio": self.marker_ratio,
                    "aruco_dict": self.aruco_dict_name,
                },
            }

            model_path = cam_output_base / "model" / "polynomial_model.mat"
            savemat(str(model_path), poly_model_data)
            logger.info(f"Saved polynomial model: {model_path}")

            return {
                "success": True,
                "model_type": "polynomial",
                "rms_error": float(fit_result["rms_fit_error_px"]),
                "mm_per_pixel": float(fit_result["mm_per_pixel"]),
                "num_images_used": stats["valid"],
                "model_path": str(model_path),
            }

        # --- PINHOLE MODEL FITTING (default) ---
        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points, all_img_points, img_size, None, None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO,
        )

        logger.info(f"RMS reprojection error: {rms:.4f} pixels")

        # Calculate errors
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

        # Save model
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
            "image_height": img_size[1],
            "timestamp": datetime.now().isoformat(),
            "dt": self.dt,
            "dot_spacing_mm": self.square_size * 1000.0,
            "board_params": {
                "squares_h": self.squares_h,
                "squares_v": self.squares_v,
                "square_size": self.square_size,
                "square_size_mm": self.square_size * 1000.0,
                "marker_ratio": self.marker_ratio,
                "aruco_dict": self.aruco_dict_name,
            },
        }

        model_path = cam_output_base / "model" / "camera_model.mat"
        savemat(str(model_path), model_data)
        logger.info(f"Saved camera model: {model_path}")

        # JSON save
        json_data = {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "rms_error": float(rms),
            "image_size": list(img_size),
            "num_images_used": stats["valid"],
        }
        json_path = cam_output_base / "model" / "camera_model.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Generate calibration model summary figure
        try:
            from pivtools_gui.calibration.calibration_figures import make_calibration_model_figure
            figures_dir = cam_output_base / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            make_calibration_model_figure(
                mtx, dist, rvecs, tvecs, rms,
                all_img_points, all_obj_points,
                img_size,
                figures_dir / "model_summary.png",
                best_image=best_image,
            )
        except Exception:
            pass

        return {
            "success": True,
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.flatten().tolist(),
            "rms_error": float(rms),
            "num_images_used": stats["valid"],
            "model_path": str(model_path),
        }

    def _process_single_image_with_data(
        self,
        image: np.ndarray,
        name: str,
        all_obj_points: List,
        all_img_points: List,
        valid_images: List,
        stats: Dict,
        detections_dir: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single calibration image and return detection data.

        Returns:
            Dict with corners and ids if detection successful, None otherwise.
        """
        if np.mean(image) < 10:
            stats["empty"] += 1
            return None

        found, corners, ids, marker_corners, marker_ids = self.detect_charuco_corners(image)

        if not found:
            stats["no_detect"] += 1
            return None

        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)

        if obj_pts is None or len(obj_pts) < self.min_corners:
            stats["no_detect"] += 1
            return None

        all_obj_points.append(obj_pts)
        all_img_points.append(img_pts)
        valid_images.append(name)
        stats["valid"] += 1

        logger.info(f"  {name}: OK ({len(corners)} corners)")

        if detections_dir is not None:
            self._save_detection_visualization(
                image, corners, ids, marker_corners, name, detections_dir
            )

        # Return detection data for indices saving
        # Reshape corners from (N, 1, 2) to (N, 2) for cleaner storage
        corners_2d = corners.reshape(-1, 2) if corners is not None else np.array([])
        ids_flat = ids.flatten() if ids is not None else np.array([])

        return {
            "corners": corners_2d,
            "ids": ids_flat,
            "name": name,
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
        """Process a single calibration image (legacy wrapper)."""
        result = self._process_single_image_with_data(
            image, name, all_obj_points, all_img_points,
            valid_images, stats, detections_dir
        )
        return result is not None

    def process_all_cameras(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all cameras with progress tracking.

        Args:
            progress_callback: Optional callback function receiving progress dict with:
                - current_camera: int or None
                - processed_cameras: int
                - total_cameras: int
                - progress: int (0-100)
                - camera_results: dict of per-camera status
            save_visualizations: Whether to save detection visualization PNGs

        Returns:
            Dict with overall results:
                - success: bool
                - processed_cameras: int
                - camera_results: dict mapping camera number to result dict
        """
        camera_results = {}
        processed_cameras = 0
        total_cameras = self.camera_count

        for cam_num in range(1, self.camera_count + 1):
            # Report starting this camera
            if progress_callback:
                progress_callback({
                    "current_camera": cam_num,
                    "processed_cameras": processed_cameras,
                    "total_cameras": total_cameras,
                    "progress": int((processed_cameras / total_cameras) * 100),
                })

            try:
                # Create per-camera progress callback that reports to parent
                def camera_progress(data):
                    if progress_callback:
                        progress_callback({
                            "current_camera": cam_num,
                            "processed_cameras": processed_cameras,
                            "total_cameras": total_cameras,
                            "progress": int((processed_cameras / total_cameras) * 100),
                            "processed_images": data.get("processed_images", 0),
                            "valid_images": data.get("valid_images", 0),
                            "total_images": data.get("total_images", 0),
                        })

                result = self.process_camera(
                    cam_num,
                    progress_callback=camera_progress,
                    save_visualizations=save_visualizations,
                )
                camera_results[cam_num] = result
                processed_cameras += 1

            except Exception as e:
                logger.error(f"Failed to process Camera {cam_num}: {e}")
                camera_results[cam_num] = {"success": False, "error": str(e)}
                processed_cameras += 1

        # Final progress report
        if progress_callback:
            progress_callback({
                "current_camera": None,
                "processed_cameras": processed_cameras,
                "total_cameras": total_cameras,
                "progress": 100,
            })

        # Determine overall success
        success_count = sum(1 for r in camera_results.values() if r.get("success"))

        return {
            "success": success_count > 0,
            "processed_cameras": processed_cameras,
            "successful_cameras": success_count,
            "camera_results": camera_results,
        }

    def run(self):
        """Run calibration (CLI mode)."""
        logger.info("=" * 60)
        logger.info("ChArUco Calibration - Starting")
        logger.info("=" * 60)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Board: {self.squares_h}x{self.squares_v} squares, {self.square_size}m size")

        results = self.process_all_cameras(save_visualizations=True)

        logger.info("=" * 60)
        logger.info("ChArUco Calibration - Complete")
        logger.info(f"Processed {results['processed_cameras']} cameras")
        logger.info(f"Successful: {results['successful_cameras']}")

        for cam_num, result in results["camera_results"].items():
            if result.get("success"):
                logger.info(f"  Camera {cam_num}: RMS={result['rms_error']:.4f} px, {result['num_images_used']} images")
            else:
                logger.error(f"  Camera {cam_num}: FAILED - {result.get('error', 'Unknown error')}")

        logger.info("=" * 60)


def main():
    """Main entry point using hardcoded configuration.

    Updates config.yaml with the hardcoded settings, then runs ChArUco calibration.
    """
    logger.info("=" * 60)
    logger.info("ChArUco Calibration - Starting")
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
        squares_h = config.data["calibration"]["charuco"]["squares_h"]
        squares_v = config.data["calibration"]["charuco"]["squares_v"]
        square_size_m = config.data["calibration"]["charuco"]["square_size"]
        marker_ratio = config.data["calibration"]["charuco"].get("marker_ratio", 0.5)
        aruco_dict = config.data["calibration"]["charuco"].get("aruco_dict", "DICT_4X4_1000")
        min_corners = config.data["calibration"]["charuco"].get("min_corners", 6)
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

        # Use hardcoded settings
        source_dir = SOURCE_DIR
        base_dir = BASE_DIR
        camera_nums = CAMERA_NUMS
        file_pattern = FILE_PATTERN
        squares_h = SQUARES_H
        squares_v = SQUARES_V
        square_size_m = SQUARE_SIZE_M
        marker_ratio = MARKER_RATIO
        aruco_dict = ARUCO_DICT
        min_corners = MIN_CORNERS

    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {base_dir}")
    logger.info(f"Cameras: {camera_nums}")
    logger.info(f"Board: {squares_h}x{squares_v} squares, {square_size_m}m size")

    failed_cameras = []

    for camera_num in camera_nums:
        logger.info(f"Processing Camera {camera_num}...")
        try:
            # Create calibrator using config - calibration_sources paths are used
            calibrator = ChArUcoCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_count=1,  # Process one at a time
                file_pattern=file_pattern,
                squares_h=squares_h,
                squares_v=squares_v,
                square_size=square_size_m,
                marker_ratio=marker_ratio,
                aruco_dict=aruco_dict,
                min_corners=min_corners,
                config=config,
            )
            result = calibrator.process_camera(camera_num, save_visualizations=True)
            if result.get("success"):
                logger.info(f"Camera {camera_num} completed: RMS={result['rms_error']:.4f} px, {result['num_images_used']} images")
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
        logger.info("ChArUco calibration completed successfully for all cameras")


if __name__ == "__main__":
    main()