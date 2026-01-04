#!/usr/bin/env python3
"""
planar_calibration_multiview.py

Pure Multi-View Dotboard Calibration script.
- Aggregates multiple views of a calibration board.
- Solves for Intrinsics (Camera Matrix + Distortion) using OpenCV.
- Saves grid detections and final model to .mat files.
- Visualizes detections for every image.
"""

import glob
import logging
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from pivtools_core.config import get_config, reload_config
from pivtools_core.image_handling.load_images import read_image

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
PATTERN_COLS = 10
PATTERN_ROWS = 10
DOT_SPACING_MM = 12.22
ASYMMETRIC = False
ENHANCE_DOTS = False

# Number of calibration images to use (set to None to use all available)
NUM_CALIBRATION_IMAGES = None

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# LOGGING SETUP
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    config.data["calibration"]["dotboard"]["pattern_cols"] = PATTERN_COLS
    config.data["calibration"]["dotboard"]["pattern_rows"] = PATTERN_ROWS
    config.data["calibration"]["dotboard"]["dot_spacing_mm"] = DOT_SPACING_MM
    config.data["calibration"]["dotboard"]["asymmetric"] = ASYMMETRIC
    config.data["calibration"]["dotboard"]["enhance_dots"] = ENHANCE_DOTS

    # Save to disk so centralized loader picks up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")

    # Reload to ensure fresh state
    return reload_config()


class MultiViewCalibrator:
    def __init__(
        self,
        source_dir,
        base_dir,
        camera_count,
        file_pattern,
        pattern_cols=10,
        pattern_rows=10,
        dot_spacing_mm=28.89,
        asymmetric=False,
        enhance_dots=False,
        calibration_subfolder="",
        config=None,
    ):
        self.source_dir = Path(source_dir)
        self.base_dir = Path(base_dir)
        self.camera_count = camera_count
        self.file_pattern = file_pattern
        self.pattern_size = (pattern_cols, pattern_rows)
        self.dot_spacing_mm = dot_spacing_mm
        self.asymmetric = asymmetric
        self.enable_dot_enhancement = enhance_dots
        self.calibration_subfolder = calibration_subfolder
        self._config = config

        # Create blob detector
        self.detector = self._create_blob_detector()

        # Setup output structure
        self._setup_directories()

    def _create_blob_detector(self):
        """Create optimized blob detector for circle grid detection"""
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 5000 # Increased slightly for varying depths
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minThreshold = 0
        params.maxThreshold = 255
        params.thresholdStep = 5
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

        Path structure: source / camera_folder / calibration_subfolder
        """
        base_path = self.source_dir

        # Get camera folder from config
        if self._config is not None:
            camera_folder = self._config.get_calibration_camera_folder(cam_num)
            if camera_folder:
                base_path = base_path / camera_folder

        # Add calibration subfolder if set
        if self.calibration_subfolder:
            base_path = base_path / self.calibration_subfolder

        return base_path

    def _is_container_format(self):
        """Check if file pattern is a container format (.set, .im7, .cine)."""
        pattern_lower = self.file_pattern.lower()
        return '.set' in pattern_lower or '.im7' in pattern_lower or '.cine' in pattern_lower

    def _read_image(self, img_path, camera=1, img_index=1):
        """Robust image reader handling standard and container formats (.tif, .set, .im7, .cine)"""
        try:
            if self._is_container_format():
                if '.set' in str(img_path).lower():
                    img = read_image(str(img_path), camera_no=camera, im_no=img_index)
                elif '.im7' in str(img_path).lower():
                    img = read_image(str(img_path), camera_no=camera)
                elif '.cine' in str(img_path).lower():
                    # .cine files: use dedicated single-frame reader
                    from pivtools_core.image_handling.readers.cine_reader import read_cine_single
                    img = read_cine_single(str(img_path), idx=img_index)
                else:
                    img = read_image(str(img_path))
            else:
                img = read_image(str(img_path))

            if img is None:
                return None

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
                # Simple downscale for detection, usually safe for grids
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

    def make_object_points(self):
        """Create real-world 3D coordinates (Z=0) for the board"""
        cols, rows = self.pattern_size
        objp = []
        for i in range(rows):
            for j in range(cols):
                if self.asymmetric:
                    x = j * self.dot_spacing_mm + (0.5 * self.dot_spacing_mm if (i % 2 == 1) else 0.0)
                    y = i * self.dot_spacing_mm
                else:
                    x = j * self.dot_spacing_mm
                    y = i * self.dot_spacing_mm
                objp.append([x, y, 0.0])
        return np.array(objp, dtype=np.float32)

    def detect_grid(self, img):
        """Detect grid points with subpixel refinement for accurate dot centers"""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if self.enable_dot_enhancement:
            gray = self.enhance_dots_image(gray)

        flags = cv2.CALIB_CB_ASYMMETRIC_GRID if self.asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID

        for test_img, label in [(gray, "Original"), (255 - gray, "Inverted")]:
            found, centers = cv2.findCirclesGrid(
                test_img, self.pattern_size, flags=flags, blobDetector=self.detector
            )
            if found:
                centers = centers.reshape(-1, 2).astype(np.float32)

                # Subpixel refinement using cornerSubPix
                # This significantly improves center accuracy for circular dots
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

                # Window size for subpixel search - adjust based on typical dot size
                # Using ~half the expected dot diameter gives good results
                win_size = (11, 11)  # Search window
                zero_zone = (-1, -1)  # No dead zone

                # cornerSubPix expects (N, 1, 2) shape
                centers_refined = cv2.cornerSubPix(
                    gray,  # Use original gray (not inverted) for refinement
                    centers.reshape(-1, 1, 2),
                    win_size,
                    zero_zone,
                    criteria
                )

                return True, centers_refined.reshape(-1, 2).astype(np.float32)

        return False, None

    def save_visualization(self, img, grid_points, img_idx, cam_base, filename):
        """Save a figure showing the detected grid indices"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ax.imshow(img, cmap="gray")
            cols = self.pattern_size[0]

            # Plot points
            ax.scatter(grid_points[:, 0], grid_points[:, 1], c='r', s=20)

            # Annotate corners or all points to show orientation
            # Annotating first and last point to verify ordering
            ax.text(grid_points[0,0], grid_points[0,1], "Start (0,0)", color='cyan', fontsize=12)
            ax.text(grid_points[-1,0], grid_points[-1,1], "End", color='cyan', fontsize=12)

            # Optional: Annotate every 10th point
            for i, (x,y) in enumerate(grid_points):
                if i % 10 == 0:
                    r, c = divmod(i, cols)
                    ax.text(x, y, f"{r},{c}", color='yellow', fontsize=8)

            ax.set_title(f"Detection: {filename}")
            ax.axis('off')
            
            out_path = cam_base / "figures" / f"detected_{img_idx:03d}.png"
            plt.savefig(out_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to save visualization for {filename}: {e}")

    def run(self):
        logger.info("Starting Multi-View Dotboard Calibration...")
        
        objp_base = self.make_object_points() # Shape: (N, 3)

        for cam_num in range(1, self.camera_count + 1):
            logger.info(f"--- Processing Camera {cam_num} ---")

            # Path setup: source / camera_folder / calibration_subfolder
            cam_input_dir = self._get_camera_input_dir(cam_num)
            cam_output_base = self.base_dir / "calibration" / f"Cam{cam_num}" / "dotboard_planar"

            # Find images
            image_files = []
            is_container = self._is_container_format()
            
            if is_container:
                container = cam_input_dir / self.file_pattern
                if container.exists(): image_files = [str(container)]
            elif "%" in self.file_pattern:
                i = 1
                while True:
                    f = cam_input_dir / (self.file_pattern % i)
                    if not f.exists(): break
                    image_files.append(str(f))
                    i += 1
            else:
                image_files = sorted([str(f) for f in cam_input_dir.glob(self.file_pattern)])

            if not image_files:
                logger.error(f"No images found for Camera {cam_num}")
                continue

            # Containers are treated as 1 file, but might have many frames. 
            # If standard files, we iterate list. If container, we might need a fixed range or metadata.
            # Assuming standard files or single container loop for now.
            
            all_objpoints = [] # 3d point in real world space
            all_imgpoints = [] # 2d points in image plane.
            valid_indices_map = {} # Store pixel values for saving later
            
            img_shape = None
            processed_count = 0

            # Loop logic adjustment for containers vs files
            loop_range = range(1, len(image_files) + 1) if not is_container else range(1, 101) # Arbitrary limit for container safety if length unknown

            logger.info(f"Scanning images...")

            for idx in loop_range:
                if not is_container:
                    img_path = image_files[idx-1]
                    img_name = Path(img_path).name
                else:
                    img_path = image_files[0]
                    img_name = f"frame_{idx}"
                    # Check if we've run out of container frames by trying to read
                    try:
                        test = self._read_image(img_path, cam_num, idx)
                        if test is None: break
                    except: break

                img = self._read_image(img_path, cam_num, idx)
                if img is None: continue
                
                if img_shape is None:
                    img_shape = img.shape[:2][::-1] # (width, height)

                found, corners = self.detect_grid(img)

                if found:
                    all_objpoints.append(objp_base)
                    all_imgpoints.append(corners)
                    
                    # Store for .mat saving
                    valid_indices_map[idx] = corners
                    
                    # Visualization
                    self.save_visualization(img, corners, idx, cam_output_base, img_name)
                    processed_count += 1
                    logger.info(f"  [+] Image {idx}: Grid detected.")
                else:
                    logger.debug(f"  [-] Image {idx}: Grid not found.")

            if processed_count < 3:
                logger.error("Not enough valid images for calibration (Need > 3).")
                continue

            # --- CALIBRATION ---
            logger.info(f"Calibrating with {processed_count} valid views...")
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_objpoints, all_imgpoints, img_shape, None, None
            )

            logger.info(f"Calibration Complete. RMS Error: {ret:.4f} pixels")

            # --- SAVE RESULTS ---

            # Prepare detections dictionary for .mat
            # We want to save the pixel locations of every point for every valid image
            detections_struct = {}
            for img_idx, points in valid_indices_map.items():
                detections_struct[f"image_{img_idx}"] = points

            model_data = {
                "camera_matrix": mtx,
                "dist_coeffs": dist,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "rms_error": ret,
                "image_width": img_shape[0],
                "image_height": img_shape[1],
                "detections_pixel_coords": detections_struct, # Store the raw grid points
                "timestamp": datetime.now().isoformat(),
                "pattern_cols": self.pattern_size[0],
                "pattern_rows": self.pattern_size[1],
                "dot_spacing_mm": self.dot_spacing_mm
            }

            out_file = cam_output_base / "model" / "dotboard_model.mat"
            savemat(str(out_file), model_data)
            logger.info(f"Saved model to: {out_file}")

            # --- SAVE INDIVIDUAL DOT CENTERS TO INDICES DIRECTORY ---
            # Save per-image .mat files with dot centers for overlay purposes
            # Each file contains: centers (Nx2), grid_indices (Nx2 for row,col)
            cols, rows = self.pattern_size
            for img_idx, points in valid_indices_map.items():
                # Create grid indices (row, col) for each detected point
                # Points are ordered row-major: (0,0), (0,1), ..., (0,cols-1), (1,0), ...
                grid_indices = np.zeros((len(points), 2), dtype=np.int32)
                for i in range(len(points)):
                    row, col = divmod(i, cols)
                    grid_indices[i] = [row, col]

                indices_data = {
                    "centers_px": points,  # Nx2 array of (x, y) pixel coordinates
                    "grid_row": grid_indices[:, 0],  # Row index for each dot
                    "grid_col": grid_indices[:, 1],  # Column index for each dot
                    "pattern_cols": cols,
                    "pattern_rows": rows,
                    "dot_spacing_mm": self.dot_spacing_mm
                }

                indices_file = cam_output_base / "indices" / f"dot_centers_{img_idx:03d}.mat"
                savemat(str(indices_file), indices_data)

            logger.info(f"Saved {len(valid_indices_map)} dot center files to indices directory")

    def process_single_camera(
        self,
        cam_num: int,
        progress_callback=None,
        save_visualizations: bool = False,
    ) -> dict:
        """
        Process a single camera for calibration with progress callback support.

        This method is designed for GUI integration where we need:
        - Progress updates during processing
        - Return value with results (instead of just saving files)
        - Optional visualization saving

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
            model_path: str
            error: str (if failed)
        """
        logger.info(f"--- Processing Camera {cam_num} ---")

        objp_base = self.make_object_points()

        # Path setup: source / camera_folder / calibration_subfolder
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
            return {"success": False, "error": f"No images found for Camera {cam_num} in {cam_input_dir}"}

        # Determine loop range
        if is_container:
            # For containers, we need to probe the number of frames
            loop_range = range(1, 101)  # Arbitrary limit
        else:
            loop_range = range(1, len(image_files) + 1)

        total_images = len(image_files) if not is_container else 100  # Estimate for containers

        all_objpoints = []
        all_imgpoints = []
        valid_indices_map = {}

        img_shape = None
        processed_count = 0
        valid_count = 0

        logger.info("Scanning images...")

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
                # Check if we've run out of container frames
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
                # Update total_images estimate for containers
                if is_container:
                    total_images = processed_count  # Will be updated as we go

            found, corners = self.detect_grid(img)

            if found:
                all_objpoints.append(objp_base)
                all_imgpoints.append(corners)
                valid_indices_map[idx] = corners
                valid_count += 1

                if save_visualizations:
                    self.save_visualization(img, corners, idx, cam_output_base, img_name)

                logger.info(f"  [+] Image {idx}: Grid detected.")
            else:
                logger.debug(f"  [-] Image {idx}: Grid not found.")

        if valid_count < 3:
            return {"success": False, "error": f"Only {valid_count} valid detections, need at least 3"}

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
        for img_idx, points in valid_indices_map.items():
            detections_struct[f"image_{img_idx}"] = points

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
            "pattern_cols": self.pattern_size[0],
            "pattern_rows": self.pattern_size[1],
            "dot_spacing_mm": self.dot_spacing_mm
        }

        out_file = cam_output_base / "model" / "dotboard_model.mat"
        savemat(str(out_file), model_data)
        logger.info(f"Saved model to: {out_file}")

        # Save per-image detection files
        cols, rows = self.pattern_size
        for img_idx, points in valid_indices_map.items():
            indices_data = {
                "grid_points": points,  # Use grid_points for consistency with loader
                "centers_px": points,   # Keep for backwards compat
                "pattern_cols": cols,
                "pattern_rows": rows,
                "dot_spacing_mm": self.dot_spacing_mm,
                "frame_index": img_idx,
            }
            # Use indexing_N.mat naming for consistency with loader
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
            "model_path": str(out_file),
        }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Planar Calibration - Starting")
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
        calibration_subfolder = config.data["calibration"].get("subfolder", "")
        pattern_cols = config.data["calibration"]["dotboard"]["pattern_cols"]
        pattern_rows = config.data["calibration"]["dotboard"]["pattern_rows"]
        dot_spacing_mm = config.data["calibration"]["dotboard"]["dot_spacing_mm"]
        asymmetric = config.data["calibration"]["dotboard"].get("asymmetric", False)
        enhance_dots = config.data["calibration"]["dotboard"].get("enhance_dots", False)
    else:
        # Apply CLI settings to config.yaml so centralized loaders work correctly
        config = apply_cli_settings_to_config()

        # Use hardcoded settings
        source_dir = SOURCE_DIR
        base_dir = BASE_DIR
        camera_nums = CAMERA_NUMS
        file_pattern = FILE_PATTERN
        calibration_subfolder = CALIBRATION_SUBFOLDER
        pattern_cols = PATTERN_COLS
        pattern_rows = PATTERN_ROWS
        dot_spacing_mm = DOT_SPACING_MM
        asymmetric = ASYMMETRIC
        enhance_dots = ENHANCE_DOTS

    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {base_dir}")
    logger.info(f"Cameras: {camera_nums}")
    logger.info(f"Pattern: {pattern_cols}x{pattern_rows}, {dot_spacing_mm}mm spacing")

    failed_cameras = []

    for camera_num in camera_nums:
        logger.info(f"Processing Camera {camera_num}...")
        try:
            # Create calibrator using config - all settings are now in config.yaml
            calibrator = MultiViewCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_count=1,  # Process one at a time
                file_pattern=file_pattern,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
                dot_spacing_mm=dot_spacing_mm,
                asymmetric=asymmetric,
                enhance_dots=enhance_dots,
                calibration_subfolder=calibration_subfolder,
                config=config,
            )
            result = calibrator.process_single_camera(camera_num, save_visualizations=True)
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
        logger.info("Planar calibration completed successfully for all cameras")