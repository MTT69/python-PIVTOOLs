#!/usr/bin/env python3
"""
scale_factor_calibration_production.py

Scale Factor Calibration - Production Module.

Converts pixel-based PIV results to physical units (mm, m/s).
- Coordinates: pixels -> mm (zero-based at bottom-left)
- Velocities: pixels/frame -> m/s

Contains ScaleFactorCalibrator class that can be:
- Imported and used by GUI views
- Run standalone via CLI with hardcoded settings
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import scipy.io
from loguru import logger as loguru_logger

from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import load_coords_from_directory

if TYPE_CHECKING:
    from pivtools_core.config import PIVConfig

# ===================== CLI CONFIGURATION =====================
# These settings are used when running the script directly

# SOURCE_DIR: Root directory containing source images (used for path resolution)
SOURCE_DIR = "/path/to/source"

# BASE_DIR: Output directory containing uncalibrated PIV results
# Results are in: {BASE_DIR}/piv_results/Cam{N}/...
BASE_DIR = "/path/to/output"

# CAMERA_SUBFOLDERS: List of subfolder names for each camera
# e.g., ["Cam1", "Cam2"] means camera 1 uses "Cam1/", camera 2 uses "Cam2/"
CAMERA_SUBFOLDERS = ["Cam1", "Cam2"]

# CAMERA_NUMS: List of camera numbers to process (1-based)
CAMERA_NUMS = [1, 2]

# SCALE FACTOR PARAMETERS
DT = 1.0              # Time between frames (seconds)
PX_PER_MM = 28.89     # Pixels per millimeter

# PROCESSING OPTIONS
IMAGE_COUNT = 1000              # Number of vector files to process per camera
TYPE_NAME = "instantaneous"     # Type of data: "instantaneous" or "ensemble"

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# LOGGING SETUP (for CLI mode)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== HELPER FUNCTIONS =====================


def _process_vector_file(args: Tuple) -> bool:
    """
    Process a single vector file for scale factor calibration.

    This is a module-level function to enable multiprocessing.
    Automatically detects ensemble data (with stress tensors) and calibrates
    both velocities and stresses.

    Args:
        args: Tuple of (run, vector_file_uncal, vector_file_cal, px_per_mm, dt)

    Returns:
        True if successful, False otherwise
    """
    run, vector_file_uncal, vector_file_cal, px_per_mm, dt = args
    try:
        mat = scipy.io.loadmat(
            str(vector_file_uncal), struct_as_record=False, squeeze_me=True
        )

        # Check for either piv_result (instantaneous) or ensemble_result (ensemble)
        if "piv_result" in mat:
            piv_result = mat["piv_result"]
            result_key = "piv_result"
        elif "ensemble_result" in mat:
            piv_result = mat["ensemble_result"]
            result_key = "ensemble_result"
        else:
            loguru_logger.warning(
                f"Vector file {vector_file_uncal} missing 'piv_result' or 'ensemble_result' field."
            )
            return False

        # Ensure piv_result is iterable (handle single-pass case)
        if not hasattr(piv_result, '__len__') or isinstance(piv_result, np.void):
            piv_result = [piv_result]

        # Check if this is ensemble data with stress tensors (check first valid cell)
        has_stresses = False
        for cell in piv_result:
            if hasattr(cell, 'UU_stress') or hasattr(cell, 'VV_stress') or hasattr(cell, 'UV_stress'):
                has_stresses = True
                break

        # Compute stress calibration factor (velocity_scale²)
        # velocity_scale = 1 / (px_per_mm * dt * 1000)
        # stress_scale = velocity_scale² = 1 / (px_per_mm² * dt² * 10⁶)
        stress_scale = 1.0 / (px_per_mm ** 2) / (dt ** 2) / 1e6

        # Build output struct array with appropriate dtype
        if has_stresses:
            # Ensemble data with stress tensors
            piv_dtype = np.dtype([
                ("ux", "O"), ("uy", "O"), ("b_mask", "O"),
                ("UU_stress", "O"), ("VV_stress", "O"), ("UV_stress", "O")
            ])
        else:
            # Standard instantaneous data
            piv_dtype = np.dtype([("ux", "O"), ("uy", "O"), ("b_mask", "O")])

        out_piv = np.empty(len(piv_result), dtype=piv_dtype)

        for idx, cell in enumerate(piv_result):
            ux = getattr(cell, "ux", None)
            uy = getattr(cell, "uy", None)
            b_mask = getattr(
                cell,
                "b_mask",
                np.zeros_like(ux) if ux is not None else np.array([]),
            )

            if ux is not None and uy is not None:
                # Convert pixels/frame to m/s
                # Formula: (px/frame) / (px/mm) / (s/frame) / 1000 = m/s
                ux_calib = ux / px_per_mm / dt / 1000
                uy_calib = uy / px_per_mm / dt / 1000

                if has_stresses:
                    # Get and calibrate stress tensors
                    UU_stress = getattr(cell, "UU_stress", None)
                    VV_stress = getattr(cell, "VV_stress", None)
                    UV_stress = getattr(cell, "UV_stress", None)

                    # Calibrate stresses: pixels²/frame² -> m²/s²
                    UU_calib = UU_stress * stress_scale if UU_stress is not None else np.array([])
                    VV_calib = VV_stress * stress_scale if VV_stress is not None else np.array([])
                    UV_calib = UV_stress * stress_scale if UV_stress is not None else np.array([])

                    out_piv[idx] = (ux_calib, uy_calib, b_mask, UU_calib, VV_calib, UV_calib)
                else:
                    out_piv[idx] = (ux_calib, uy_calib, b_mask)
            else:
                if has_stresses:
                    out_piv[idx] = (np.array([]), np.array([]), np.array([]),
                                   np.array([]), np.array([]), np.array([]))
                else:
                    out_piv[idx] = (np.array([]), np.array([]), np.array([]))

        scipy.io.savemat(
            str(vector_file_cal), {result_key: out_piv}, do_compression=True
        )
        return True

    except Exception as e:
        loguru_logger.error(
            f"Error processing vector file {vector_file_uncal}: {e}", exc_info=True
        )
        return False


# ===================== CALIBRATOR CLASS =====================


class ScaleFactorCalibrator:
    """
    Scale factor calibration service.

    Converts pixel-based coordinates and velocities to physical units.

    - Coordinates: pixels -> mm (zero-based at bottom-left)
    - Velocities: pixels/frame -> m/s

    This class can be used from both CLI and GUI contexts.
    """

    def __init__(
        self,
        base_path: Path,
        source_path_idx: int = 0,
        type_name: str = "instantaneous",
        dt: Optional[float] = None,
        px_per_mm: Optional[float] = None,
        config: Optional["PIVConfig"] = None,
    ):
        """
        Initialize scale factor calibrator.

        Args:
            base_path: Base output directory
            source_path_idx: Index into config source_paths (for getting settings)
            type_name: Type of data (instantaneous, ensemble)
            dt: Time between frames in seconds (optional, reads from config if not provided)
            px_per_mm: Pixels per millimeter (optional, reads from config if not provided)
            config: Optional config object to read dt/px_per_mm from
        """
        self.base_path = Path(base_path)
        self.source_path_idx = source_path_idx
        self.type_name = type_name

        # Read dt and px_per_mm from config if not provided directly
        if config is not None:
            sf_cfg = config.data.get("calibration", {}).get("scale_factor", {})
            self.dt = dt if dt is not None else sf_cfg.get("dt", 1.0)
            self.px_per_mm = px_per_mm if px_per_mm is not None else sf_cfg.get("px_per_mm", 1.0)
        else:
            self.dt = dt if dt is not None else 1.0
            self.px_per_mm = px_per_mm if px_per_mm is not None else 1.0

    def calibrate_coordinates(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates to mm, zero-based at bottom-left.

        Args:
            x: X coordinates in pixels
            y: Y coordinates in pixels

        Returns:
            Tuple of (x_mm, y_mm) calibrated coordinates
        """
        # Zero-base: subtract first value
        x0 = x.flat[0] if x.size > 0 else 0
        y0 = y.flat[0] if y.size > 0 else 0

        x_calib = (x - x0) / self.px_per_mm
        # Flip y-axis and negate for bottom-left origin
        y_calib = -np.flipud((y - y0) / self.px_per_mm)

        return x_calib, y_calib

    def calibrate_vectors(
        self, ux_px: np.ndarray, uy_px: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel velocities to m/s.

        Args:
            ux_px: X velocity in pixels/frame
            uy_px: Y velocity in pixels/frame

        Returns:
            Tuple of (ux_ms, uy_ms) velocities in m/s
        """
        ux_ms = ux_px / self.px_per_mm / self.dt / 1000
        uy_ms = uy_px / self.px_per_mm / self.dt / 1000
        return ux_ms, uy_ms

    def calibrate_stresses(self, stress_px: np.ndarray) -> np.ndarray:
        """
        Convert stress tensor from pixels²/frame² to m²/s².

        For ensemble PIV, Reynolds stresses (UU, VV, UV) are computed before
        calibration and have units of velocity². The calibration factor must
        therefore be squared compared to velocity calibration.

        Velocity:  (px/frame) / (px/mm) / (s/frame) / 1000 = m/s
        Stress:    (px²/frame²) / (px²/mm²) / (s²/frame²) / 10⁶ = m²/s²

        Args:
            stress_px: Stress tensor in pixels²/frame²

        Returns:
            Stress tensor in m²/s²
        """
        # Stress has velocity² units, so square the calibration factor
        # velocity_scale = 1 / (px_per_mm * dt * 1000)
        # stress_scale = velocity_scale² = 1 / (px_per_mm² * dt² * 10⁶)
        stress_scale = 1.0 / (self.px_per_mm ** 2) / (self.dt ** 2) / 1e6
        return stress_px * stress_scale

    def process_camera(
        self,
        camera_num: int,
        image_count: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single camera's data.

        Args:
            camera_num: Camera number to process
            image_count: Number of images to process
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with processing results
        """
        cfg = get_config()

        paths_uncal = get_data_paths(
            base_dir=self.base_path,
            num_frame_pairs=cfg.num_frame_pairs,
            cam=camera_num,
            type_name=self.type_name,
            use_uncalibrated=True,
        )
        paths_calib = get_data_paths(
            base_dir=self.base_path,
            num_frame_pairs=cfg.num_frame_pairs,
            cam=camera_num,
            type_name=self.type_name,
            use_uncalibrated=False,
        )

        data_dir_uncal = paths_uncal["data_dir"]
        data_dir_cal = paths_calib["data_dir"]
        data_dir_cal.mkdir(parents=True, exist_ok=True)

        coords_path_uncal = data_dir_uncal / "coordinates.mat"
        coords_path_cal = data_dir_cal / "coordinates.mat"

        # Count files to process
        coords_exists = coords_path_uncal.exists()
        vector_files = []

        # Check for ensemble data (single file) vs instantaneous (many files)
        ensemble_file_uncal = data_dir_uncal / "ensemble_result.mat"
        ensemble_file_cal = data_dir_cal / "ensemble_result.mat"
        is_ensemble = self.type_name == "ensemble" or ensemble_file_uncal.exists()

        if is_ensemble and ensemble_file_uncal.exists():
            # Ensemble data: single file
            vector_files.append((1, ensemble_file_uncal, ensemble_file_cal))
        else:
            # Instantaneous data: many files
            for run in range(1, image_count + 1):
                vector_file_uncal = data_dir_uncal / (cfg.vector_format % run)
                vector_file_cal = data_dir_cal / (cfg.vector_format % run)
                if vector_file_uncal.exists():
                    vector_files.append((run, vector_file_uncal, vector_file_cal))

        total_files = (1 if coords_exists else 0) + len(vector_files)

        result = {
            "camera": camera_num,
            "total_files": total_files,
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "coords_processed": False,
            "success": True,
            "error": None,
        }

        if total_files == 0:
            result["success"] = False
            result["error"] = f"No files found to process for camera {camera_num}"
            loguru_logger.warning(result["error"])
            return result

        processed = 0

        # Process coordinates
        if coords_exists:
            success = self._process_coordinates(coords_path_uncal, coords_path_cal)
            result["coords_processed"] = success
            processed += 1
            if success:
                result["successful_files"] += 1
            else:
                result["failed_files"] += 1
            result["processed_files"] = processed

            if progress_callback:
                progress_callback(
                    {
                        "camera": camera_num,
                        "processed_files": processed,
                        "total_files": total_files,
                        "progress": int((processed / total_files) * 100),
                    }
                )

        # Process vector files in parallel
        if vector_files:
            vector_args = [
                (run, uncal, cal, self.px_per_mm, self.dt)
                for run, uncal, cal in vector_files
            ]

            max_workers = min(4, len(vector_files))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_vector_file, args)
                    for args in vector_args
                ]

                for future in as_completed(futures):
                    try:
                        success = future.result()
                        if success:
                            result["successful_files"] += 1
                        else:
                            result["failed_files"] += 1
                    except Exception as e:
                        loguru_logger.error(f"Future failed with exception: {e}")
                        result["failed_files"] += 1

                    processed += 1
                    result["processed_files"] = processed

                    if progress_callback:
                        progress_callback(
                            {
                                "camera": camera_num,
                                "processed_files": processed,
                                "total_files": total_files,
                                "progress": int((processed / total_files) * 100),
                            }
                        )

        return result

    def _process_coordinates(
        self, coords_path_uncal: Path, coords_path_cal: Path
    ) -> bool:
        """
        Process coordinates file.

        Args:
            coords_path_uncal: Path to uncalibrated coordinates
            coords_path_cal: Path to save calibrated coordinates

        Returns:
            True if successful
        """
        try:
            # Use centralized coordinate loading
            x_list, y_list = load_coords_from_directory(coords_path_uncal.parent)

            if not x_list:
                loguru_logger.warning(f"No coordinates found in {coords_path_uncal.parent}")
                return False

            # Build output struct array
            coord_dtype = np.dtype([("x", "O"), ("y", "O")])
            out_coords = np.empty(len(x_list), dtype=coord_dtype)

            processed_runs = 0
            for run_idx, (x, y) in enumerate(zip(x_list, y_list)):
                if x is not None and y is not None and x.size > 0 and y.size > 0:
                    x_calib, y_calib = self.calibrate_coordinates(x, y)
                    out_coords[run_idx] = (x_calib, y_calib)
                    processed_runs += 1
                else:
                    out_coords[run_idx] = (np.array([]), np.array([]))

            scipy.io.savemat(
                str(coords_path_cal), {"coordinates": out_coords}, do_compression=True
            )
            loguru_logger.info(f"Updated coordinates for {processed_runs} runs")
            return True

        except FileNotFoundError:
            loguru_logger.warning(f"No coordinates.mat found in {coords_path_uncal.parent}")
            return False
        except KeyError as e:
            loguru_logger.warning(f"Invalid coordinates file: {e}")
            return False
        except Exception as e:
            loguru_logger.error(f"Error processing coordinates: {e}", exc_info=True)
            return False

    def process_all_cameras(
        self,
        cameras: List[int],
        image_count: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process all specified cameras.

        This is the main entry point for multi-camera calibration.

        Args:
            cameras: List of camera numbers to process
            image_count: Number of images to process per camera
            progress_callback: Optional callback for progress updates.
                Called with dict containing:
                - current_camera: Current camera being processed
                - processed_cameras: Number of cameras completed
                - total_cameras: Total camera count
                - camera_progress: Per-camera progress dict
                - overall_progress: Overall progress percentage

        Returns:
            Dictionary with overall results
        """
        total_cameras = len(cameras)
        cfg = get_config()

        # First pass: count total files across all cameras
        total_files = 0
        camera_file_counts = {}

        for cam_num in cameras:
            paths_uncal = get_data_paths(
                base_dir=self.base_path,
                num_frame_pairs=cfg.num_frame_pairs,
                cam=cam_num,
                type_name=self.type_name,
                use_uncalibrated=True,
            )
            data_dir_uncal = paths_uncal["data_dir"]
            coords_path_uncal = data_dir_uncal / "coordinates.mat"

            camera_files = 0
            if coords_path_uncal.exists():
                camera_files += 1

            for run in range(1, image_count + 1):
                vector_file = data_dir_uncal / (cfg.vector_format % run)
                if vector_file.exists():
                    camera_files += 1

            camera_file_counts[cam_num] = camera_files
            total_files += camera_files

        overall_result = {
            "total_cameras": total_cameras,
            "processed_cameras": 0,
            "total_files": total_files,
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "camera_results": {},
            "success": True,
            "error": None,
        }

        if total_files == 0:
            overall_result["success"] = False
            overall_result["error"] = "No files found to process across all cameras"
            loguru_logger.warning(overall_result["error"])
            return overall_result

        # Process each camera
        total_processed_files = 0

        for cam_idx, cam_num in enumerate(cameras):
            loguru_logger.info(f"Processing camera {cam_num} ({cam_idx + 1}/{total_cameras})")

            def camera_progress(data: Dict[str, Any]):
                nonlocal total_processed_files
                total_processed_files = (
                    overall_result["processed_files"] + data["processed_files"]
                )
                if progress_callback:
                    progress_callback(
                        {
                            "current_camera": cam_num,
                            "processed_cameras": cam_idx,
                            "total_cameras": total_cameras,
                            "camera_processed_files": data["processed_files"],
                            "camera_total_files": data["total_files"],
                            "overall_processed_files": total_processed_files,
                            "overall_total_files": total_files,
                            "overall_progress": int(
                                (total_processed_files / total_files) * 100
                            ),
                        }
                    )

            cam_result = self.process_camera(
                camera_num=cam_num,
                image_count=image_count,
                progress_callback=camera_progress,
            )

            overall_result["camera_results"][cam_num] = cam_result
            overall_result["processed_files"] += cam_result["processed_files"]
            overall_result["successful_files"] += cam_result["successful_files"]
            overall_result["failed_files"] += cam_result["failed_files"]
            overall_result["processed_cameras"] = cam_idx + 1

        # Aggregate success status
        if overall_result["successful_files"] == 0:
            overall_result["success"] = False
            # Collect errors from camera results
            errors = [
                cam_result.get("error")
                for cam_result in overall_result["camera_results"].values()
                if cam_result.get("error")
            ]
            overall_result["error"] = "; ".join(errors) if errors else "No files were successfully processed"

        return overall_result


# ===================== CLI FUNCTIONS =====================


def apply_cli_settings_to_config():
    """Update config.yaml with CLI-mode hardcoded settings.

    This function writes the hardcoded configuration variables to config.yaml,
    ensuring the calibration system uses the correct paths and settings.

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

    # Ensure calibration section exists
    if "calibration" not in config.data:
        config.data["calibration"] = {}

    # Scale factor settings
    config.data["calibration"]["scale_factor"] = {
        "dt": DT,
        "px_per_mm": PX_PER_MM,
    }

    # Save to disk so other components pick up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")
    logger.info(f"  dt = {DT} seconds")
    logger.info(f"  px_per_mm = {PX_PER_MM}")
    logger.info(f"  cameras = {CAMERA_NUMS}")

    # Reload to ensure fresh state
    return reload_config()


def main():
    """Run scale factor calibration for all configured cameras."""
    logger.info("=" * 60)
    logger.info("Scale Factor Calibration - Production Script")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        cfg = get_config()

        # Extract settings from config
        base_dir = cfg.data["paths"]["base_paths"][0]
        camera_nums = cfg.data["paths"].get("camera_numbers", [1, 2])
        dt = cfg.data["calibration"].get("scale_factor", {}).get("dt", 1.0)
        px_per_mm = cfg.data["calibration"].get("scale_factor", {}).get("px_per_mm", 1.0)
        image_count = cfg.data.get("processing", {}).get("num_frame_pairs", 1000)
        type_name = cfg.data.get("processing", {}).get("type_name", "instantaneous")
    else:
        # Apply hardcoded settings to config
        cfg = apply_cli_settings_to_config()

        # Use hardcoded settings
        base_dir = BASE_DIR
        camera_nums = CAMERA_NUMS
        dt = DT
        px_per_mm = PX_PER_MM
        image_count = IMAGE_COUNT
        type_name = TYPE_NAME

    # Create calibrator instance - reads dt/px_per_mm from config
    calibrator = ScaleFactorCalibrator(
        base_path=Path(base_dir),
        source_path_idx=0,
        type_name=type_name,
        dt=dt,
        px_per_mm=px_per_mm,
        config=cfg,
    )

    logger.info(f"Processing {len(camera_nums)} camera(s): {camera_nums}")
    logger.info(f"Image count per camera: {image_count}")
    logger.info(f"Type: {type_name}")

    # Progress callback for CLI output
    def progress_callback(progress_data):
        current_cam = progress_data.get("current_camera", "?")
        overall_progress = progress_data.get("overall_progress", 0)
        processed = progress_data.get("overall_processed_files", 0)
        total = progress_data.get("overall_total_files", 0)
        logger.info(f"  Camera {current_cam}: {overall_progress}% ({processed}/{total} files)")

    # Run calibration for all cameras
    result = calibrator.process_all_cameras(
        cameras=camera_nums,
        image_count=image_count,
        progress_callback=progress_callback,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Calibration Complete!")
    logger.info(f"  Total files: {result['total_files']}")
    logger.info(f"  Successful: {result['successful_files']}")
    logger.info(f"  Failed: {result['failed_files']}")
    logger.info(f"  Cameras processed: {result['processed_cameras']}")
    logger.info("=" * 60)

    # Per-camera breakdown
    for cam_num, cam_result in result.get("camera_results", {}).items():
        status = "OK" if cam_result.get("failed_files", 0) == 0 else "ERRORS"
        logger.info(
            f"  Camera {cam_num}: {cam_result.get('successful_files', 0)}/"
            f"{cam_result.get('total_files', 0)} files [{status}]"
        )

    return result


if __name__ == "__main__":
    main()
