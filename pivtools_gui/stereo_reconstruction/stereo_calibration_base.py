#!/usr/bin/env python3
"""
stereo_calibration_base.py

Base class for stereo camera calibration.
Provides shared stereo calibration logic (stereoCalibrate, stereoRectify) that is
inherited by both dotboard (circle grid) and ChArUco stereo calibrators.
"""

import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.io import savemat

from pivtools_core.config import Config, get_config
from pivtools_core.image_handling.calibration_loader import read_calibration_image

from .calibration_io import (
    find_calibration_images,
    is_container_format,
)


class BaseStereoCalibrator(ABC):
    """Base class for stereo camera calibration.

    Subclasses must implement:
    - _create_detector(): Create the pattern detector
    - detect_pattern(): Detect calibration pattern in image
    - make_object_points(): Generate 3D object points for pattern
    - get_pattern_params(): Get pattern-specific parameters for saving

    The base class handles:
    - Image loading (standard and container formats)
    - Stereo calibration (cv2.stereoCalibrate, cv2.stereoRectify)
    - Result saving (.mat files)
    - Progress callbacks
    - Visualization
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        source_dir: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        camera_pair: Optional[List[int]] = None,
        file_pattern: Optional[str] = None,
        camera_subfolders: Optional[List[str]] = None,
        source_path_idx: int = 0,
        dt: Optional[float] = None,
    ):
        """Initialize stereo calibrator.

        Parameters
        ----------
        config : Config, optional
            Configuration object. If provided, settings are read from config.
        source_dir : str or Path, optional
            Override for source directory (where calibration images are)
        base_dir : str or Path, optional
            Override for output directory (where results are saved)
        camera_pair : list[int], optional
            Override for [cam1, cam2] pair (e.g., [1, 2])
        file_pattern : str, optional
            Override for calibration image pattern (e.g., "calib%05d.tif")
        camera_subfolders : list[str], optional
            List of camera folder names (e.g., ["Cam1", "Cam2"]). Index = camera_num - 1.
            Set to None or [] for containers or when images are in source_dir directly.
        source_path_idx : int, optional
            Index into config.calibration_sources (default: 0)
        dt : float, optional
            Time step between frames in seconds
        """
        # Store config reference
        self._config = config if config is not None else get_config()
        self._source_path_idx = source_path_idx

        # Resolve paths from config or explicit params
        stereo_cfg = self._config.stereo_calibration if self._config else {}

        # Source directory
        if source_dir is not None:
            self.source_dir = Path(source_dir)
        elif self._config is not None:
            self.source_dir = Path(self._config.source_paths[source_path_idx])
        else:
            raise ValueError("Either config or source_dir must be provided")

        # Base (output) directory
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif self._config is not None:
            self.base_dir = Path(self._config.base_paths[source_path_idx])
        else:
            raise ValueError("Either config or base_dir must be provided")

        # Camera pair
        self.camera_pair = camera_pair or stereo_cfg.get("camera_pair", [1, 2])

        # File pattern
        if file_pattern is not None:
            self.file_pattern = file_pattern
        elif self._config is not None:
            self.file_pattern = self._config.calibration_image_format
        else:
            self.file_pattern = "*.tif"

        # Camera subfolders (CLI override, otherwise use config)
        self.camera_subfolders = camera_subfolders

        # Time step - use explicit dt, or fall back to unified config.dt
        if dt is not None:
            self.dt = dt
        elif self._config is not None:
            self.dt = self._config.dt
        else:
            self.dt = 1.0

        # Initialize detector (implemented by subclass)
        self.detector = self._create_detector()

    @abstractmethod
    def _create_detector(self) -> Any:
        """Create the pattern detector.

        Returns
        -------
        Any
            Detector object (e.g., SimpleBlobDetector for dotboard,
            (CharucoBoard, CharucoDetector) tuple for ChArUco)
        """

    @abstractmethod
    def detect_pattern(
        self, image: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], ...]:
        """Detect calibration pattern in image.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)

        Returns
        -------
        tuple
            (found: bool, points: np.ndarray or None, ...)
            points shape: (N, 2) if found
            Additional return values are subclass-specific (e.g., corner IDs for ChArUco)
        """

    @abstractmethod
    def make_object_points(self) -> np.ndarray:
        """Generate 3D object points for the calibration pattern.

        Returns
        -------
        np.ndarray
            Shape (N, 3) with Z=0 for planar targets
        """

    @abstractmethod
    def get_pattern_params(self) -> Dict[str, Any]:
        """Get pattern-specific parameters for saving to output files.

        Returns
        -------
        dict
            Pattern parameters (e.g., pattern_type, pattern_cols, dot_spacing_mm)
        """

    def _is_container_format(self) -> bool:
        """Check if file pattern is a container format (.set, .cine)."""
        return is_container_format(self._config, self.file_pattern)

    def _read_calibration_image_centralized(
        self,
        camera: int,
        img_index: int,
    ) -> Optional[np.ndarray]:
        """Read calibration image using centralized calibration_loader.

        Parameters
        ----------
        camera : int
            Camera number (1-based)
        img_index : int
            Image index (1-based)

        Returns
        -------
        np.ndarray or None
            Image as uint8 array, or None if read failed
        """
        try:
            img = read_calibration_image(
                idx=img_index,
                camera=camera,
                config=self._config,
                source_path_idx=self._source_path_idx,
                normalize_uint8=False,  # Preserve native dtype for float32 flat-fielding
            )
            return img

        except Exception as e:
            logger.warning(
                f"Failed to read image camera={camera}, index={img_index}: {e}"
            )
            return None

    def _get_camera_input_dir(self, cam_num: int) -> Path:
        """Get the input directory for calibration images.

        Uses build_calibration_camera_path for path resolution from
        calibration_sources config.

        Parameters
        ----------
        cam_num : int
            Camera number (1-based)

        Returns
        -------
        Path
            Path to calibration images for this camera
        """
        if self._config is not None:
            from pivtools_core.image_handling.path_utils import (
                build_calibration_camera_path,
            )

            return build_calibration_camera_path(
                self._config, source_path_idx=self._source_path_idx, camera=cam_num
            )

        # Fallback: use source_dir directly
        return self.source_dir

    def _get_output_dir(self, cam1: int, cam2: int) -> Path:
        """Get the output directory for stereo calibration results.

        Parameters
        ----------
        cam1 : int
            First camera number
        cam2 : int
            Second camera number

        Returns
        -------
        Path
            Output directory: {base_dir}/calibration/stereo_cam{n}_cam{m}/
        """
        return self.base_dir / "calibration" / f"stereo_cam{cam1}_cam{cam2}"

    def _find_calibration_images(self, cam_input_dir: Path) -> List[Path]:
        """Find all calibration images matching the pattern."""
        return find_calibration_images(cam_input_dir, self.file_pattern, self._config)

    def _perform_stereo_calibration(
        self,
        objpoints: List[np.ndarray],
        imgpoints1: List[np.ndarray],
        imgpoints2: List[np.ndarray],
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Perform OpenCV stereo calibration.

        This is the shared stereo calibration logic used by all subclasses.

        Parameters
        ----------
        objpoints : list[np.ndarray]
            List of 3D object points, each (N, 1, 3) or (N, 3)
        imgpoints1 : list[np.ndarray]
            List of 2D image points for camera 1, each (N, 1, 2) or (N, 2)
        imgpoints2 : list[np.ndarray]
            List of 2D image points for camera 2, each (N, 1, 2) or (N, 2)
        image_size : tuple
            Image size as (width, height)

        Returns
        -------
        dict
            Calibration results including camera matrices, distortion coefficients,
            rotation, translation, essential/fundamental matrices, rectification data
        """
        # Reshape points for OpenCV if needed
        objpoints_cv = [p.reshape(-1, 1, 3).astype(np.float32) for p in objpoints]
        imgpoints1_cv = [p.reshape(-1, 1, 2).astype(np.float32) for p in imgpoints1]
        imgpoints2_cv = [p.reshape(-1, 1, 2).astype(np.float32) for p in imgpoints2]

        # Individual camera calibration
        logger.info("Calibrating individual cameras...")
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
            objpoints_cv,
            imgpoints1_cv,
            image_size,
            None,
            None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO,
        )
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            objpoints_cv,
            imgpoints2_cv,
            image_size,
            None,
            None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO,
        )
        logger.info(f"Camera 1 RMS error: {ret1:.5f}")
        logger.info(f"Camera 2 RMS error: {ret2:.5f}")

        # Stereo calibration
        logger.info("Performing stereo calibration...")
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        flags = cv2.CALIB_FIX_INTRINSIC

        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints_cv,
            imgpoints1_cv,
            imgpoints2_cv,
            mtx1,
            dist1,
            mtx2,
            dist2,
            image_size,
            criteria=criteria,
            flags=flags,
        )
        logger.info(f"Stereo RMS error: {ret:.5f}")

        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx1,
            dist1,
            mtx2,
            dist2,
            image_size,
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
        )

        # Calculate relative angle between cameras
        angle_rad = math.acos(max(-1, min(1, (np.trace(R) - 1) / 2)))
        angle_deg = np.degrees(angle_rad)
        logger.info(f"Relative angle between cameras: {angle_deg:.2f} degrees")

        return {
            "camera_matrix_1": mtx1,
            "dist_coeffs_1": dist1,
            "camera_matrix_2": mtx2,
            "dist_coeffs_2": dist2,
            "rvecs_1": np.array([r.flatten() for r in rvecs1]),
            "tvecs_1": np.array([t.flatten() for t in tvecs1]),
            "rvecs_2": np.array([r.flatten() for r in rvecs2]),
            "tvecs_2": np.array([t.flatten() for t in tvecs2]),
            "rotation_matrix": R,
            "translation_vector": T,
            "essential_matrix": E,
            "fundamental_matrix": F,
            "rectification_R1": R1,
            "rectification_R2": R2,
            "projection_P1": P1,
            "projection_P2": P2,
            "disparity_to_depth_Q": Q,
            "valid_pixel_ROI1": roi1,
            "valid_pixel_ROI2": roi2,
            "stereo_rms_error": ret,
            "cam1_rms_error": ret1,
            "cam2_rms_error": ret2,
            "relative_angle_deg": angle_deg,
        }

    def _save_stereo_results(
        self,
        cam1: int,
        cam2: int,
        calibration_result: Dict[str, Any],
        successful_pairs: List[str],
        image_size: Tuple[int, int],
        indices_data: Dict[int, Dict[str, Any]],
    ) -> Path:
        """Save stereo calibration results to .mat files.

        Parameters
        ----------
        cam1 : int
            First camera number
        cam2 : int
            Second camera number
        calibration_result : dict
            Result from _perform_stereo_calibration()
        successful_pairs : list[str]
            List of successfully processed image filenames
        image_size : tuple
            Image size as (width, height)
        indices_data : dict
            Per-frame detection data: {frame_idx: {grid_points_cam1, grid_points_cam2, ...}}

        Returns
        -------
        Path
            Path to the saved stereo_model.mat file
        """
        output_dir = self._get_output_dir(cam1, cam2)

        # Create directories
        (output_dir / "model").mkdir(parents=True, exist_ok=True)
        (output_dir / "indices").mkdir(parents=True, exist_ok=True)
        (output_dir / "detections").mkdir(parents=True, exist_ok=True)

        # Build stereo model data
        stereo_data = {
            **calibration_result,
            "num_image_pairs": len(successful_pairs),
            "image_size": list(image_size),
            "timestamp": datetime.now().isoformat(),
            "dt": self.dt,
            "camera_pair": self.camera_pair,
            "pattern_params": self.get_pattern_params(),
            "successful_filenames": successful_pairs,
        }

        # Save main stereo model
        model_path = output_dir / "model" / "stereo_model.mat"
        savemat(str(model_path), stereo_data)
        logger.info(f"Saved stereo model: {model_path}")

        # Save per-frame indices
        for frame_idx, data in indices_data.items():
            indices_file = output_dir / "indices" / f"indexing_{frame_idx}.mat"
            savemat(str(indices_file), data)

        logger.info(
            f"Saved {len(indices_data)} indexing files to {output_dir / 'indices'}"
        )

        return model_path

    def _save_detection_visualization(
        self,
        image: np.ndarray,
        grid_points: np.ndarray,
        cam_num: int,
        frame_idx: int,
        output_dir: Path,
        filename: str,
        reprojection_error: Optional[float] = None,
    ):
        """Save visualization of detected grid points.

        Parameters
        ----------
        image : np.ndarray
            Original image
        grid_points : np.ndarray
            Detected grid points (N, 2)
        cam_num : int
            Camera number
        frame_idx : int
            Frame index
        output_dir : Path
            Output directory for detections
        filename : str
            Original filename
        reprojection_error : float, optional
            Reprojection error to display
        """
        fig = None
        try:
            # Ensure output directory exists
            detections_dir = output_dir / "detections"
            detections_dir.mkdir(parents=True, exist_ok=True)

            if len(image.shape) == 2:
                vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[-1] == 1:
                # Handle (H, W, 1) grayscale images
                vis = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                vis = image.copy()

            # Draw detected points
            for pt in grid_points:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

            title = f"Cam{cam_num} - {filename} - {len(grid_points)} points"
            if reprojection_error is not None:
                title += f" - RMS: {reprojection_error:.3f}px"
            ax.set_title(title)
            ax.axis("off")

            plt.tight_layout()
            out_path = detections_dir / f"detection_cam{cam_num}_{frame_idx:03d}.png"
            plt.savefig(out_path, dpi=100, bbox_inches="tight")

        except Exception as e:
            logger.warning(f"Failed to save visualization: {e}")
        finally:
            # Always close the figure to prevent memory leaks
            if fig is not None:
                plt.close(fig)

    def process_camera_pair(
        self,
        cam1: Optional[int] = None,
        cam2: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Process a camera pair for stereo calibration.

        Parameters
        ----------
        cam1 : int, optional
            First camera number (default: from camera_pair)
        cam2 : int, optional
            Second camera number (default: from camera_pair)
        progress_callback : callable, optional
            Progress callback receiving dict with keys:
            - progress: int (0-100)
            - stage: str ('detecting', 'calibrating', 'saving', 'complete')
            - processed_pairs: int
            - valid_pairs: int
            - total_pairs: int
        save_visualizations : bool
            Whether to save detection visualizations

        Returns
        -------
        dict
            success: bool
            stereo_rms_error: float (if success)
            cam1_rms_error: float (if success)
            cam2_rms_error: float (if success)
            num_pairs_used: int (if success)
            model_path: str (if success)
            relative_angle_deg: float (if success)
            error: str (if not success)
        """
        # Use default camera pair if not specified
        cam1_val: int = cam1 if cam1 is not None else self.camera_pair[0]
        cam2_val: int = cam2 if cam2 is not None else self.camera_pair[1]

        logger.info(f"Processing stereo pair: Camera {cam1_val} and Camera {cam2_val}")

        is_container = self._is_container_format()
        output_dir = self._get_output_dir(cam1_val, cam2_val)

        # Get number of calibration images from config
        if self._config is None:
            return {"success": False, "error": "Config required for stereo calibration"}

        # Use calibration_image_count (not num_frame_pairs which is for PIV images)
        num_frame_pairs = self._config.calibration_image_count
        if num_frame_pairs == 0:
            return {
                "success": False,
                "error": "No calibration images found (calibration_image_count=0)",
            }

        logger.info(f"Processing {num_frame_pairs} calibration frame pairs")

        # For container formats, validate the container exists
        if is_container:
            # Container file path comes from calibration_sources (no camera subfolders)
            container_file = self._get_camera_input_dir(cam1_val)
            if not container_file.exists():
                return {
                    "success": False,
                    "error": f"Container file not found: {container_file}",
                }
            logger.info(f"Using container file: {container_file}")

        # Collect calibration data
        objpoints: List[np.ndarray] = []
        imgpoints1: List[np.ndarray] = []
        imgpoints2: List[np.ndarray] = []
        successful_pairs: List[str] = []
        indices_data: Dict[int, Dict[str, Any]] = {}
        objp = self.make_object_points()
        image_size: Optional[Tuple[int, int]] = None

        processed_count = 0

        # --- Worker: read + detect + match one stereo frame ---
        def _process_stereo_frame(img_index):
            """Returns (img_index, result_dict_or_None). Thread-safe: no shared state mutation."""
            filename = f"frame_{img_index:05d}"

            # Read both cameras in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(
                    self._read_calibration_image_centralized,
                    camera=cam1_val,
                    img_index=img_index,
                )
                f2 = executor.submit(
                    self._read_calibration_image_centralized,
                    camera=cam2_val,
                    img_index=img_index,
                )
                img1 = f1.result()
                img2 = f2.result()

            if img1 is None or img2 is None:
                return img_index, None

            # Detect pattern in both images
            result1 = self.detect_pattern(img1)
            result2 = self.detect_pattern(img2)

            found1 = result1[0]
            found2 = result2[0]
            grid1 = result1[1] if found1 else None
            grid2 = result2[1] if found2 else None

            if not found1 or not found2 or grid1 is None or grid2 is None:
                logger.debug(f"Pattern not found in pair {filename}")
                return img_index, None

            # Match points between cameras
            match_result = self._match_object_points(objp, result1, result2)
            if match_result is None:
                logger.warning(f"Could not match object points in {filename}")
                return img_index, None

            if isinstance(match_result, tuple) and len(match_result) == 3:
                obj_pts, matched_grid1, matched_grid2 = match_result
                grid1 = matched_grid1
                grid2 = matched_grid2
            else:
                obj_pts = match_result
                if len(grid1) != len(grid2):
                    logger.warning(
                        f"Point count mismatch in {filename}: {len(grid1)} vs {len(grid2)}"
                    )
                    return img_index, None

            info1 = result1[3] if len(result1) >= 4 else None
            info2 = result2[3] if len(result2) >= 4 else None
            grid_data_1 = result1[2] if len(result1) >= 3 else None
            grid_data_2 = result2[2] if len(result2) >= 3 else None

            return img_index, {
                "filename": filename,
                "obj_pts": obj_pts,
                "grid1": grid1,
                "grid2": grid2,
                "img1": img1,
                "img2": img2,
                "result1": result1,
                "result2": result2,
                "match_result": match_result,
                "info1": info1,
                "info2": info2,
                "grid_data_1": grid_data_1,
                "grid_data_2": grid_data_2,
                "found1": found1,
                "found2": found2,
                "image_size": img1.shape[:2][::-1],
            }

        # --- For containers, discover frame count first ---
        if is_container:
            container_count = 0
            for probe_idx in range(1, num_frame_pairs + 1):
                probe_img = self._read_calibration_image_centralized(
                    camera=cam1_val, img_index=probe_idx
                )
                if probe_img is None:
                    break
                container_count += 1
            num_frame_pairs = container_count
            logger.info(f"Container has {container_count} frames")

        # --- Parallel detection across frames ---
        max_workers = min(os.cpu_count() or 4, num_frame_pairs, 8)
        raw_results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_stereo_frame, idx): idx
                for idx in range(1, num_frame_pairs + 1)
            }
            for future in as_completed(futures):
                img_index, result = future.result()
                raw_results[img_index] = result
                processed_count += 1
                valid_so_far = sum(1 for r in raw_results.values() if r is not None)
                if progress_callback:
                    progress_callback(
                        {
                            "progress": min(
                                int((processed_count / num_frame_pairs) * 70), 70
                            ),
                            "stage": "detecting",
                            "processed_pairs": processed_count,
                            "valid_pairs": valid_so_far,
                            "total_pairs": num_frame_pairs,
                        }
                    )

        # --- Collect results in frame order ---
        for img_index in sorted(raw_results.keys()):
            r = raw_results[img_index]
            if r is None:
                continue

            if image_size is None:
                image_size = r["image_size"]

            obj_pts = r["obj_pts"]
            grid1, grid2 = r["grid1"], r["grid2"]
            filename = r["filename"]
            match_result = r["match_result"]

            objpoints.append(obj_pts)
            imgpoints1.append(grid1)
            imgpoints2.append(grid2)

            frame_idx = len(successful_pairs) + 1
            successful_pairs.append(filename)

            # Store indices data
            frame_indices = {
                "grid_points_cam1": grid1,
                "grid_points_cam2": grid2,
                "object_points": obj_pts,
                "frame_index": frame_idx,
                "original_filename": filename,
            }

            # Derive matched grid indices from object points
            grid_data_1, grid_data_2 = r["grid_data_1"], r["grid_data_2"]
            if isinstance(match_result, tuple) and len(match_result) == 3:
                dot_spacing = getattr(self, "dot_spacing_mm", None)
                if dot_spacing and dot_spacing > 0:
                    matched_indices = np.round(obj_pts[:, :2] / dot_spacing).astype(int)
                    frame_indices["grid_indices_cam1"] = matched_indices
                    frame_indices["grid_indices_cam2"] = matched_indices
            else:
                if (
                    grid_data_1 is not None
                    and isinstance(grid_data_1, dict)
                    and "grid_indices" in grid_data_1
                ):
                    frame_indices["grid_indices_cam1"] = grid_data_1["grid_indices"]
                if (
                    grid_data_2 is not None
                    and isinstance(grid_data_2, dict)
                    and "grid_indices" in grid_data_2
                ):
                    frame_indices["grid_indices_cam2"] = grid_data_2["grid_indices"]

            if (
                grid_data_1 is not None
                and isinstance(grid_data_1, np.ndarray)
                and grid_data_1.ndim == 1
            ):
                frame_indices["corner_ids_cam1"] = grid_data_1
            if (
                grid_data_2 is not None
                and isinstance(grid_data_2, np.ndarray)
                and grid_data_2.ndim == 1
            ):
                frame_indices["corner_ids_cam2"] = grid_data_2

            indices_data[frame_idx] = frame_indices

            # Save visualizations
            img1, img2 = r["img1"], r["img2"]
            info1, info2 = r["info1"], r["info2"]
            found1, found2 = r["found1"], r["found2"]

            if save_visualizations:
                self._save_detection_visualization(
                    img1, grid1, cam1_val, frame_idx, output_dir, filename
                )
                self._save_detection_visualization(
                    img2, grid2, cam2_val, frame_idx, output_dir, filename
                )

            try:
                figures_dir = output_dir / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)

                if info1 is not None and isinstance(grid_data_1, dict):
                    from .calibration_figures import make_detection_figure

                    make_detection_figure(
                        img1,
                        found1,
                        grid_data_1,
                        info1,
                        figures_dir / f"detection_cam{cam1_val}_{frame_idx:03d}.png",
                        title=f"Cam{cam1_val} {filename}",
                    )
                if info2 is not None and isinstance(grid_data_2, dict):
                    from .calibration_figures import make_detection_figure

                    make_detection_figure(
                        img2,
                        found2,
                        grid_data_2,
                        info2,
                        figures_dir / f"detection_cam{cam2_val}_{frame_idx:03d}.png",
                        title=f"Cam{cam2_val} {filename}",
                    )

                if info1 is not None and "board_params" in info1:
                    from .calibration_figures import make_charuco_detection_figure

                    make_charuco_detection_figure(
                        img1,
                        grid1,
                        grid_data_1 if isinstance(grid_data_1, np.ndarray) else None,
                        info1["board_params"],
                        figures_dir / f"detection_cam{cam1_val}_{frame_idx:03d}.png",
                        title=f"Cam{cam1_val} {filename}",
                    )
                if info2 is not None and "board_params" in info2:
                    from .calibration_figures import make_charuco_detection_figure

                    make_charuco_detection_figure(
                        img2,
                        grid2,
                        grid_data_2 if isinstance(grid_data_2, np.ndarray) else None,
                        info2["board_params"],
                        figures_dir / f"detection_cam{cam2_val}_{frame_idx:03d}.png",
                        title=f"Cam{cam2_val} {filename}",
                    )
            except Exception as e:
                logger.warning(
                    f"Stereo detection figure failed for frame {frame_idx}: {e}"
                )

            logger.info(f"Successfully processed pair {filename}")

        # Check if we have enough pairs
        if len(successful_pairs) < 3:
            return {
                "success": False,
                "error": f"Insufficient valid image pairs: {len(successful_pairs)} (need >= 3)",
            }

        logger.info(f"Using {len(successful_pairs)} image pairs for calibration")

        # Report calibration stage
        if progress_callback:
            progress_callback(
                {
                    "progress": 70,
                    "stage": "calibrating",
                    "processed_pairs": processed_count,
                    "valid_pairs": len(successful_pairs),
                    "total_pairs": num_frame_pairs,
                }
            )

        # Perform stereo calibration
        # At this point we have at least 3 valid pairs, so image_size is definitely set
        assert image_size is not None, "image_size should be set after processing pairs"
        try:
            calibration_result = self._perform_stereo_calibration(
                objpoints, imgpoints1, imgpoints2, image_size
            )
        except cv2.error as e:
            return {"success": False, "error": f"OpenCV calibration failed: {e}"}
        except Exception as e:
            logger.exception(f"Unexpected error during calibration: {e}")
            return {"success": False, "error": str(e)}

        # Report saving stage
        if progress_callback:
            progress_callback(
                {
                    "progress": 90,
                    "stage": "saving",
                    "processed_pairs": processed_count,
                    "valid_pairs": len(successful_pairs),
                    "total_pairs": num_frame_pairs,
                }
            )

        # Save results
        # At this point image_size is guaranteed to be set since we have successful_pairs > 0
        assert image_size is not None, "image_size should be set after processing pairs"
        model_path = self._save_stereo_results(
            cam1_val,
            cam2_val,
            calibration_result,
            successful_pairs,
            image_size,
            indices_data,
        )

        # Clear any stale self-calibration from a previous stereo model
        sc_path = output_dir / "self_calibration.yaml"
        if sc_path.exists():
            sc_path.unlink()
            logger.info("Cleared stale self-calibration file")

        # Report completion
        if progress_callback:
            progress_callback(
                {
                    "progress": 100,
                    "stage": "complete",
                    "processed_pairs": processed_count,
                    "valid_pairs": len(successful_pairs),
                    "total_pairs": num_frame_pairs,
                }
            )

        return {
            "success": True,
            "stereo_rms_error": calibration_result["stereo_rms_error"],
            "cam1_rms_error": calibration_result["cam1_rms_error"],
            "cam2_rms_error": calibration_result["cam2_rms_error"],
            "num_pairs_used": len(successful_pairs),
            "model_path": str(model_path),
            "relative_angle_deg": calibration_result["relative_angle_deg"],
        }

    def _match_object_points(
        self,
        objp: np.ndarray,
        result1: Tuple,
        result2: Tuple,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Match object points to detected image points.

        This base implementation assumes all points are detected in order.
        Subclasses can override to match by corner ID (ChArUco) or grid index
        (automatic dotboard detection with variable grid sizes).

        Parameters
        ----------
        objp : np.ndarray
            Full object points array (N, 3) - may be empty for automatic detection
        result1 : tuple
            Detection result from camera 1
        result2 : tuple
            Detection result from camera 2

        Returns
        -------
        np.ndarray or tuple or None
            For fixed-size patterns: matched object points array
            For automatic detection: tuple of (obj_pts, img_pts_1, img_pts_2)
            None if matching failed
        """
        # Base implementation: assume all points detected in order
        grid1 = result1[1]
        expected_count = len(objp)

        if len(grid1) == expected_count:
            return objp

        # If fewer points detected, cannot match without IDs
        return None

    def run(self):
        """Run stereo calibration (CLI mode)."""
        logger.info("=" * 60)
        logger.info("Stereo Calibration - Starting")
        logger.info("=" * 60)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.base_dir}")
        logger.info(f"Camera pair: {self.camera_pair}")

        result = self.process_camera_pair(save_visualizations=True)

        logger.info("=" * 60)
        if result["success"]:
            logger.info("Stereo Calibration - Complete")
            logger.info(f"Stereo RMS error: {result['stereo_rms_error']:.5f}")
            logger.info(f"Camera 1 RMS error: {result['cam1_rms_error']:.5f}")
            logger.info(f"Camera 2 RMS error: {result['cam2_rms_error']:.5f}")
            logger.info(f"Relative angle: {result['relative_angle_deg']:.2f} degrees")
            logger.info(f"Images used: {result['num_pairs_used']}")
            logger.info(f"Model saved: {result['model_path']}")
        else:
            logger.error(f"Stereo Calibration - FAILED: {result['error']}")
        logger.info("=" * 60)

        return result
