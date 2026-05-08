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

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.io import savemat

from pivtools_core.config import get_config, reload_config

from pivtools_gui.calibration.grid_detection import (
    to_grayscale_2d,
    detect_grid_automatic,
)
from pivtools_gui.calibration.calibration_io import (
    is_container_format,
    read_calibration_image_with_fallback,
    read_calibration_image_direct,
    find_calibration_images,
    get_camera_input_dir,
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

# Number of calibration images to use (set to None to use all available)
NUM_CALIBRATION_IMAGES = None

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

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

    # Output subdirectory name under `calibration/Cam{N}/`. Subclasses
    # override this to land their results in a different folder without
    # having to touch any of the discovery / save / figure plumbing.
    output_subdir_name: str = "dotboard_planar"

    def __init__(
        self,
        source_dir,
        base_dir,
        camera_count,
        file_pattern,
        dot_spacing_mm=28.89,
        config=None,
        datum_frame=1,
        model_type="pinhole",
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
        config : Config, optional
            Configuration object
        datum_frame : int
            Which calibration image defines the world coordinate origin (1-based, default: 1)
        model_type : str
            Calibration model type: "pinhole" (OpenCV) or "polynomial" (3rd-order bivariate)
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
        self._config = config
        self.datum_frame = datum_frame
        self.model_type = model_type

        # These will be populated during detection
        self._detected_cols: Optional[int] = None
        self._detected_rows: Optional[int] = None

        # Setup output structure
        self._setup_directories()

    def _setup_directories(self):
        """Create output directories with /dotboard_planar structure"""
        for cam_num in range(1, self.camera_count + 1):
            # NEW PATH STRUCTURE: .../CamX/dotboard_planar/
            cam_base = self.base_dir / "calibration" / f"Cam{cam_num}" / self.output_subdir_name
            (cam_base / "indices").mkdir(parents=True, exist_ok=True)
            (cam_base / "model").mkdir(parents=True, exist_ok=True)
            (cam_base / "figures").mkdir(parents=True, exist_ok=True)

    def _get_camera_input_dir(self, cam_num: int) -> Path:
        """Get the input directory for calibration images."""
        return get_camera_input_dir(cam_num, self._config, 0, self.source_dir)

    def _is_container_format(self):
        """Check if file pattern is a container format (.set, .cine)."""
        return is_container_format(self._config, self.file_pattern)

    def _read_image(self, img_path=None, camera=1, img_index=1):
        """Read calibration image using core reader with CLI fallback."""
        return read_calibration_image_with_fallback(
            img_path, camera, img_index, self._config, 0, self.file_pattern,
        )

    def _read_image_direct(self, img_path, camera=1, img_index=1):
        """Direct image reading fallback for CLI mode without config."""
        return read_calibration_image_direct(
            img_path, camera, img_index, self.file_pattern, self._config,
        )

    def make_object_points_dynamic(
        self,
        grid_indices: np.ndarray,
        n_cols: int,
        n_rows: int,
    ) -> np.ndarray:
        """Create real-world 3D coordinates (Z=0) for detected grid points.

        Uses the flip-and-offset convention so the stored world frame is
        physics-up with its origin at the bottom row of the detected
        grid, matching the stereo dotboard calibrator and the stepped
        board paths. See ``make_object_points_dynamic`` in
        ``stereo_dotboard_calibration_production.py`` for the rationale.

        The ``VectorCalibrator`` that consumes this model does NOT apply
        an output-time Y negation anymore — it used to, paired with a
        construction-time y-down convention. Both sides were flipped in
        the same patch so the net physics-up behaviour is preserved
        while the origin moves to the bottom.

        Parameters
        ----------
        grid_indices : np.ndarray
            Array of (col, row) indices for each detected point, shape
            (N, 2). Indices are already normalised by
            ``detect_grid_automatic`` so the minimum along each axis is 0.
        n_cols, n_rows : int
            Detected grid dimensions (kept for API compatibility; the
            flip uses the per-call max of ``grid_indices[:, 1]`` so
            partial detections still place the origin at the bottom of
            whatever was matched, not at a nominal n_rows-1).

        Returns
        -------
        np.ndarray
            3D object points, shape (N, 3), with Z=0.
        """
        obj_points = np.zeros((len(grid_indices), 3), dtype=np.float32)
        obj_points[:, 0] = grid_indices[:, 0] * self.dot_spacing_mm
        grid_y = grid_indices[:, 1]
        obj_points[:, 1] = (grid_y.max() - grid_y) * self.dot_spacing_mm
        return obj_points

    def detect_grid(self, img):
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
        info : dict or None
            Diagnostic info from detect_grid_automatic
        """
        gray = to_grayscale_2d(img)

        # Use automatic detection
        success, grid_data, info = detect_grid_automatic(
            gray,
            mask=None,
            grid_spacing_mm=self.dot_spacing_mm,
        )

        if success and grid_data is not None:
            # Store detected dimensions
            self._detected_cols = grid_data['n_cols']
            self._detected_rows = grid_data['n_rows']
            return True, grid_data['centers'], grid_data, info

        return False, None, None, info

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
            cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / self.output_subdir_name

            # Find images
            is_container = self._is_container_format()
            image_files = [str(f) for f in find_calibration_images(cam_input_dir, self.file_pattern, self._config)]

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

                img = self._read_image(img_path, cam_num, idx)
                if img is None:
                    if is_container:
                        break  # End of container frames
                    continue

                if img_shape is None:
                    img_shape = img.shape[:2][::-1]  # (width, height)

                # Use automatic detection (returns centers + grid_data + info)
                found, corners, grid_data, _info = self.detect_grid(img)

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
                all_objpoints, all_imgpoints, img_shape, None, None,
                flags=cv2.CALIB_FIX_ASPECT_RATIO,
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

            dt = self._config.data.get("calibration", {}).get("dotboard", {}).get("dt", 1.0) if self._config else 1.0
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
                "dt": dt,
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
        cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / self.output_subdir_name

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

        # Limit to num_images from config (if set)
        max_images = None
        if self._config:
            max_images = self._config.data.get("calibration", {}).get("num_images")
        if max_images and max_images > 0:
            if not is_container and len(image_files) > max_images:
                logger.info(f"Limiting to {max_images} of {len(image_files)} images (from config num_images)")
                image_files = image_files[:max_images]

        # Determine loop range
        if is_container:
            container_limit = max_images if max_images and max_images > 0 else 101
            loop_range = range(1, container_limit + 1)
        else:
            loop_range = range(1, len(image_files) + 1)

        total_images = len(image_files) if not is_container else (max_images or 100)

        all_objpoints = []
        all_imgpoints = []
        valid_indices_map = {}

        img_shape = None
        processed_count = 0
        valid_count = 0
        detected_cols = None
        detected_rows = None
        best_image = None         # For model summary figure background
        best_image_npts = 0

        logger.info("Scanning images with automatic grid detection...")

        # --- Worker function for parallel detection ---
        def _detect_one(idx):
            """Read and detect grid in a single image. Returns (idx, img, found, corners, grid_data, det_info, img_name)."""
            if not is_container:
                if idx > len(image_files):
                    return idx, None, False, None, None, None, None
                img_path = image_files[idx - 1]
                img_name = Path(img_path).name
            else:
                img_path = image_files[0]
                img_name = f"frame_{idx}"

            img = self._read_image(img_path, cam_num, idx)
            if img is None:
                return idx, None, False, None, None, None, img_name

            found, corners, grid_data, det_info = self.detect_grid(img)
            return idx, img, found, corners, grid_data, det_info, img_name

        # --- For containers, discover frame count first (sequential) ---
        if is_container:
            container_count = 0
            for probe_idx in loop_range:
                probe_img = self._read_image(image_files[0], cam_num, probe_idx)
                if probe_img is None:
                    break
                container_count += 1
                if img_shape is None:
                    img_shape = probe_img.shape[:2][::-1]
            total_images = container_count
            loop_range = range(1, container_count + 1)
            logger.info(f"Container has {container_count} frames")

        # --- Parallel detection across images ---
        max_workers = min(os.cpu_count() or 4, len(loop_range), 8)
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_detect_one, idx): idx
                for idx in loop_range
            }
            for future in as_completed(futures):
                idx, img, found, corners, grid_data, det_info, img_name = future.result()
                results[idx] = (img, found, corners, grid_data, det_info, img_name)
                processed_count += 1
                if found and grid_data is not None:
                    valid_count += 1
                if progress_callback:
                    progress = int(processed_count / max(total_images, 1) * 90)
                    progress_callback({
                        "progress": min(progress, 90),
                        "processed_images": processed_count,
                        "valid_images": valid_count,
                        "total_images": total_images,
                    })

        # --- Collect results in frame order ---
        for idx in sorted(results.keys()):
            img, found, corners, grid_data, det_info, img_name = results[idx]
            if img is None:
                continue

            if img_shape is None:
                img_shape = img.shape[:2][::-1]

            if found and grid_data is not None:
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

                if detected_cols is None:
                    detected_cols = grid_data['n_cols']
                    detected_rows = grid_data['n_rows']
                if len(corners) > best_image_npts:
                    best_image = img.copy()
                    best_image_npts = len(corners)

                try:
                    from pivtools_gui.calibration.calibration_figures import make_detection_figure
                    figures_dir = cam_output_base / "figures"
                    figures_dir.mkdir(parents=True, exist_ok=True)
                    make_detection_figure(
                        img, True, grid_data, det_info,
                        figures_dir / f"detection_{idx:03d}.png",
                        title=img_name,
                    )
                except Exception as e:
                    logger.debug(f"Detection figure skipped for image {idx}: {e}")

                logger.info(f"  [+] Image {idx}: Grid detected ({grid_data['n_cols']}x{grid_data['n_rows']}, {len(corners)} points)")
            else:
                try:
                    from pivtools_gui.calibration.calibration_figures import make_detection_figure
                    figures_dir = cam_output_base / "figures"
                    figures_dir.mkdir(parents=True, exist_ok=True)
                    make_detection_figure(
                        img, False, None, det_info or {},
                        figures_dir / f"detection_{idx:03d}.png",
                        title=img_name,
                    )
                except Exception:
                    pass
                logger.debug(f"  [-] Image {idx}: Grid not found.")

        if valid_count < 1:
            return {"success": False, "error": f"Only {valid_count} valid detections, need at least 1"}

        # --- SAVE PER-IMAGE DETECTION FILES (shared by both model types) ---
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

        # --- CALIBRATION ---
        logger.info(f"Calibrating with {valid_count} valid views (model_type={self.model_type})...")

        if self.model_type == "polynomial":
            # --- POLYNOMIAL MODEL FITTING ---
            # Polynomial calibration is a 2D bivariate fit and must use ONLY the
            # datum frame.  Multi-image fitting stacks points from poses at
            # differing world-Z which the model cannot represent — pinhole is
            # the right choice for multi-image calibration.
            from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import (
                fit_polynomial_from_points, save_polynomial_to_config
            )

            sorted_valid_frames = sorted(valid_indices_map.keys())
            if self.datum_frame not in valid_indices_map:
                return {
                    "success": False,
                    "error": (
                        f"Polynomial calibration requires the datum frame "
                        f"(frame {self.datum_frame}) to be detected, but detection "
                        f"failed on that frame. Valid frames: {sorted_valid_frames}."
                    ),
                }

            datum_list_idx = sorted_valid_frames.index(self.datum_frame)
            datum_img_pts = all_imgpoints[datum_list_idx]
            datum_obj_pts = all_objpoints[datum_list_idx]

            warnings_out = []
            if len(sorted_valid_frames) > 1:
                ignored = len(sorted_valid_frames) - 1
                msg = (
                    f"Polynomial calibration uses only the datum frame "
                    f"(frame {self.datum_frame}). {ignored} additional detected "
                    f"frame(s) were ignored — the polynomial model is a 2D fit "
                    f"and is not designed for multi-image calibration. Use "
                    f"pinhole for multi-image."
                )
                logger.warning(msg)
                warnings_out.append(msg)

            all_img_flat = datum_img_pts.reshape(-1, 2)
            all_obj_flat = datum_obj_pts.reshape(-1, 3)[:, :2]
            # Object points are already in mm (from dot_spacing_mm)

            try:
                fit_result = fit_polynomial_from_points(all_img_flat, all_obj_flat, img_shape)
            except Exception as e:
                return {"success": False, "error": f"Polynomial fitting failed: {e}"}

            logger.info(f"Polynomial fit complete. RMS error: {fit_result['rms_fit_error_px']:.4f} px")

            # Save polynomial coefficients to config
            dt = self._config.data.get("calibration", {}).get("dotboard", {}).get("dt", 1.0) if self._config else 1.0
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
                "num_images_used": 1,
                "image_width": img_shape[0],
                "image_height": img_shape[1],
                "image_size": np.array([img_shape[0], img_shape[1]]),
                "dot_spacing_mm": self.dot_spacing_mm,
                "datum_frame": self.datum_frame,
                "timestamp": datetime.now().isoformat(),
            }

            out_file = cam_output_base / "model" / "polynomial_model.mat"
            savemat(str(out_file), poly_model_data)
            logger.info(f"Saved polynomial model to: {out_file}")

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
                "model_type": "polynomial",
                "rms_error": float(fit_result["rms_fit_error_px"]),
                "mm_per_pixel": float(fit_result["mm_per_pixel"]),
                "num_images_used": 1,
                "detected_cols": detected_cols,
                "detected_rows": detected_rows,
                "model_path": str(out_file),
                "warnings": warnings_out,
            }

        # --- PINHOLE MODEL FITTING (default) ---
        # TODO: factory-intrinsics / seed-from-prior workflow.
        #   Scope: add `seed_model_path` and `freeze_intrinsics` kwargs to
        #   MultiViewCalibrator.__init__, load K + dist from the prior .mat,
        #   and pass CALIB_USE_INTRINSIC_GUESS | FIX_FOCAL_LENGTH
        #   | FIX_PRINCIPAL_POINT | FIX_K1..K3 | ZERO_TANGENT_DIST so only
        #   extrinsics get solved. ~30 lines here + a conditional branch.
        #   Keep it CLI-only — factory-intrinsics is a per-project power-user
        #   decision that belongs in scripts, not the GUI (GUI file pickers
        #   make it too easy to seed from a stale or wrong-camera .mat and
        #   silently produce garbage extrinsics). Existing loader:
        #   camera_model_utils.py::load_pinhole_camera. Defer until a real
        #   camera with real factory specs motivates the design choices.
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_objpoints, all_imgpoints, img_shape, None, None,
                flags=cv2.CALIB_FIX_ASPECT_RATIO,
            )
        except Exception as e:
            return {"success": False, "error": f"OpenCV calibration failed: {e}"}

        logger.info(f"Calibration Complete. RMS Error: {ret:.4f} pixels")

        # --- SAVE RESULTS ---
        detections_struct = {}
        for img_idx, data in valid_indices_map.items():
            detections_struct[f"image_{img_idx}"] = data['centers']

        dt = self._config.data.get("calibration", {}).get("dotboard", {}).get("dt", 1.0) if self._config else 1.0
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
            "dt": dt,
            # Legacy fields for backward compat
            "pattern_cols": detected_cols,
            "pattern_rows": detected_rows,
        }

        out_file = cam_output_base / "model" / "dotboard_model.mat"
        savemat(str(out_file), model_data)
        logger.info(f"Saved model to: {out_file}")

        # Generate calibration model summary figure
        try:
            from pivtools_gui.calibration.calibration_figures import make_calibration_model_figure
            figures_dir = cam_output_base / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            make_calibration_model_figure(
                mtx, dist, rvecs, tvecs, ret,
                all_imgpoints, all_objpoints,
                img_shape,
                figures_dir / "model_summary.png",
                best_image=best_image,
            )
            logger.info(f"Saved model summary figure to: {figures_dir / 'model_summary.png'}")
        except Exception as e:
            logger.debug(f"Model summary figure skipped: {e}")

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