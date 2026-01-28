#!/usr/bin/env python3
"""
transform_production.py

Production module for applying geometric transformations to PIV vector fields.
- Supports YAML config or hardcoded variables for CLI use
- Transforms PIV vector files and coordinates
- Does NOT transform statistics files (user must recalculate)
- Uses per-camera transform configuration
- Can be run from command line or called from GUI

Pattern matches: planar_calibration_production.py
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from pivtools_core.config import Config, get_config, reload_config

from .transform_operations import (
    VALID_TRANSFORMATIONS,
    apply_transformation_to_piv_result,
    apply_transformation_to_coordinates,
    coords_to_structured_array,
    load_mat_for_transform,
    save_mat_from_transform,
    simplify_transformations,
    validate_transformations,
)

# ===================== CONFIGURATION =====================

# BASE_DIR: Root directory containing PIV data
BASE_DIR = "/path/to/experiment"

# CAMERA TRANSFORMS: Per-camera transformation operations
# These will be simplified before application
CAMERA_TRANSFORMS = {
    1: ["flip_ud"],
    2: ["rotate_90_cw"],
}

# DATA TYPE
TYPE_NAME = "instantaneous"
USE_MERGED = False

# USE_CONFIG_DIRECTLY: If True, load settings from config.yaml
USE_CONFIG_DIRECTLY = True

# LOGGING SETUP
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ===================== HELPER FUNCTIONS =====================


def get_data_dir(base_dir: Path, num_frame_pairs: int, camera: int,
                 type_name: str, use_merged: bool, use_stereo: bool = False,
                 stereo_camera_pair: Optional[tuple] = None) -> Path:
    """Build path to calibrated PIV data directory."""
    num_str = str(num_frame_pairs)
    if use_stereo:
        if stereo_camera_pair is None:
            raise ValueError("stereo_camera_pair required when use_stereo=True")
        cam_pair_str = f"Cam{stereo_camera_pair[0]}_Cam{stereo_camera_pair[1]}"
        return base_dir / "stereo_calibrated" / num_str / cam_pair_str / type_name
    elif use_merged:
        return base_dir / "calibrated_piv" / num_str / "merged" / f"Cam{camera}" / type_name
    else:
        return base_dir / "calibrated_piv" / num_str / f"Cam{camera}" / type_name


# ===================== PROCESSOR CLASS =====================


class TransformProcessor:
    """
    Production class for applying transformations to PIV data.

    Designed for:
    - GUI integration with progress callbacks
    - Command-line execution via __main__
    - Per-camera transform configuration from YAML

    NOTE: This processor does NOT transform statistics files.
    Users must manually recalculate statistics after applying transforms.
    """

    def __init__(
        self,
        base_dir: Path,
        camera_transforms: Dict[int, List[str]],
        type_name: str = "instantaneous",
        use_merged: bool = False,
        use_stereo: bool = False,
        stereo_camera_pair: Optional[tuple] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the transform processor.

        Parameters
        ----------
        base_dir : Path
            Base directory containing PIV data
        camera_transforms : dict
            Dictionary mapping camera numbers to transform lists
        type_name : str
            Data type ('instantaneous', 'ensemble')
        use_merged : bool
            Whether to transform merged data
        use_stereo : bool
            Whether to transform stereo data
        stereo_camera_pair : tuple, optional
            Camera pair (cam1, cam2) for stereo paths
        config : Config, optional
            Config object for accessing settings
        """
        self.base_dir = Path(base_dir)
        self.type_name = type_name
        self.use_merged = use_merged
        self.use_stereo = use_stereo
        self.stereo_camera_pair = stereo_camera_pair
        self._config = config or get_config()

        # Simplify transforms before storing
        self.camera_transforms = {}
        for cam, ops in camera_transforms.items():
            simplified = simplify_transformations(ops)
            if simplified:  # Only store non-empty
                self.camera_transforms[cam] = simplified

        logger.info(f"Simplified transforms: {self.camera_transforms}")

    def process_camera(
        self,
        camera: int,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        max_workers: int = 8,
    ) -> Dict:
        """
        Apply transformations to a single camera's data.

        Transforms:
        - Vector files in calibrated_piv/{N}/Cam{camera}/{type}/
        - Coordinates in coordinates.mat

        Does NOT transform statistics files.

        Parameters
        ----------
        camera : int
            Camera number (1-based)
        progress_callback : callable, optional
            Progress callback function
        max_workers : int
            Maximum parallel workers

        Returns
        -------
        dict
            Result with success, transformed_files, elapsed_time, error
        """
        start_time = time.time()

        transforms = self.camera_transforms.get(camera, [])
        if not transforms:
            return {
                "success": True,
                "message": f"No transforms configured for camera {camera}",
                "transformed_files": 0,
            }

        logger.info(f"Processing camera {camera} with transforms: {transforms}")

        try:
            data_dir = get_data_dir(
                self.base_dir,
                self._config.num_frame_pairs,
                camera,
                self.type_name,
                self.use_merged,
                self.use_stereo,
                self.stereo_camera_pair,
            )

            if not data_dir.exists():
                return {
                    "success": False,
                    "error": f"Data directory not found: {data_dir}",
                }

            vector_files = list(data_dir.glob("*.mat"))
            # Exclude coordinates.mat from parallel processing
            vector_files = [f for f in vector_files if f.name != "coordinates.mat"]

            total_files = len(vector_files)
            processed_files = 0

            # Transform coordinates first (once)
            coords_file = data_dir / "coordinates.mat"
            if coords_file.exists():
                logger.info("Transforming coordinates...")
                coords_mat = load_mat_for_transform(coords_file)
                coords = coords_mat.get("coordinates")
                if coords is not None:
                    for trans in transforms:
                        if isinstance(coords, np.ndarray) and coords.dtype == object:
                            for run_idx in range(coords.size):
                                apply_transformation_to_coordinates(coords, run_idx + 1, trans)
                        else:
                            apply_transformation_to_coordinates(coords, 1, trans)
                    coords_struct = coords_to_structured_array(coords)
                    save_mat_from_transform(coords_file, {"coordinates": coords_struct})
                    logger.info("Coordinates transformed.")

            # Transform vector files in parallel
            num_workers = min(os.cpu_count() or 1, len(vector_files), max_workers)

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for mat_file in vector_files:
                    future = executor.submit(
                        _transform_vector_file_worker,
                        str(mat_file),
                        transforms,
                    )
                    futures[future] = mat_file

                for future in as_completed(futures):
                    try:
                        success = future.result()
                        if success:
                            processed_files += 1
                        if progress_callback:
                            progress = int(processed_files / total_files * 100) if total_files > 0 else 100
                            progress_callback({
                                "progress": progress,
                                "processed_files": processed_files,
                                "total_files": total_files,
                            })
                    except Exception as e:
                        logger.error(f"Error transforming {futures[future]}: {e}")

            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "camera": camera,
                "transforms": transforms,
                "transformed_files": processed_files,
                "total_files": total_files,
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            logger.exception(f"Error processing camera {camera}: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }

    def process_all_cameras(
        self,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        max_workers: int = 8,
    ) -> Dict:
        """Process all cameras with configured transforms."""
        results = {}
        total_cameras = len(self.camera_transforms)

        for idx, camera in enumerate(self.camera_transforms.keys()):
            logger.info(f"Processing camera {camera} ({idx + 1}/{total_cameras})")

            result = self.process_camera(camera, None, max_workers)
            results[camera] = result

            if progress_callback:
                progress_callback({
                    "progress": int((idx + 1) / total_cameras * 100),
                    "current_camera": camera,
                    "processed_cameras": idx + 1,
                    "total_cameras": total_cameras,
                })

        return {
            "success": all(r.get("success", False) for r in results.values()),
            "camera_results": results,
        }


def _transform_vector_file_worker(mat_file_path: str, transforms: List[str]) -> bool:
    """Worker function for parallel vector file transformation."""
    from scipy.io import loadmat, savemat
    import warnings
    # Import inside worker for multiprocessing compatibility
    from pivtools_core.vector_loading import is_run_valid

    try:
        mat = loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)

        if "piv_result" not in mat:
            return False

        piv_result = mat["piv_result"]

        # Ensure array
        if not isinstance(piv_result, np.ndarray):
            piv_result = np.array([piv_result])
        elif piv_result.ndim == 0:
            piv_result = np.array([piv_result.item()])

        # Apply transforms to each VALID run (use centralized validation)
        for run_idx in range(piv_result.size):
            pr = piv_result[run_idx]
            try:
                if is_run_valid(pr, fields=("ux",), require_2d=False, reject_all_nan=True):
                    for trans in transforms:
                        apply_transformation_to_piv_result(pr, trans)
            except Exception as e:
                logger.warning(f"Error validating run {run_idx + 1} in {mat_file_path}: {e}")
                continue

        # Remove any legacy pending_transformations field
        if "pending_transformations" in mat:
            del mat["pending_transformations"]
        if "piv_result_original" in mat:
            del mat["piv_result_original"]

        # Save back
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            savemat(mat_file_path, mat, do_compression=True)

        return True

    except Exception as e:
        logger.error(f"Error transforming {mat_file_path}: {e}")
        return False


# ===================== CLI ENTRY POINT =====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Transform Production - Starting")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        logger.info("Loading settings from config.yaml")
        config = get_config()

        camera_transforms = config.transforms_cameras
        type_name = config.transforms_type_name

        if not camera_transforms:
            logger.error("No transforms configured. Add transforms to config.yaml.")
            exit(1)

        # Loop through active_paths for batch processing
        active_paths = config.active_paths
        logger.info(f"Processing {len(active_paths)} active path(s)")
        logger.info(f"Camera transforms: {camera_transforms}")
        logger.info(f"Type: {type_name}")

        results = []
        for path_idx in active_paths:
            base_dir = config.base_paths[path_idx]
            logger.info("")
            logger.info("=" * 40)
            logger.info(f"Path {path_idx + 1}/{len(active_paths)}: {base_dir}")
            logger.info("=" * 40)

            processor = TransformProcessor(
                base_dir=base_dir,
                camera_transforms=camera_transforms,
                type_name=type_name,
                use_merged=USE_MERGED,
                config=config,
            )

            result = processor.process_all_cameras()
            result["path_idx"] = path_idx
            result["base_dir"] = str(base_dir)
            results.append(result)

            if result["success"]:
                for cam, res in result["camera_results"].items():
                    logger.info(f"  Camera {cam}: {res.get('transformed_files', 0)} files")
            else:
                logger.error(f"Path {path_idx + 1}: FAILED")
                for cam, res in result["camera_results"].items():
                    if not res.get("success"):
                        logger.error(f"    Camera {cam}: {res.get('error')}")

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        success_count = sum(1 for r in results if r["success"])
        for r in results:
            status = "OK" if r["success"] else "FAILED"
            logger.info(f"  Path {r['path_idx'] + 1}: {status}")
        logger.info(f"Total: {success_count}/{len(results)} paths succeeded")
        logger.info("")
        logger.info("NOTE: Statistics files were NOT transformed.")
        logger.info("Please recalculate statistics if needed.")

        all_success = all(r["success"] for r in results)
        exit(0 if all_success else 1)
    else:
        logger.info("Using hardcoded settings")
        config = get_config()

        base_dir = BASE_DIR
        camera_transforms = CAMERA_TRANSFORMS
        type_name = TYPE_NAME

        logger.info(f"Base dir: {base_dir}")
        logger.info(f"Camera transforms: {camera_transforms}")
        logger.info(f"Type: {type_name}")

        if not camera_transforms:
            logger.error("No transforms configured. Set CAMERA_TRANSFORMS.")
            exit(1)

        processor = TransformProcessor(
            base_dir=base_dir,
            camera_transforms=camera_transforms,
            type_name=type_name,
            use_merged=USE_MERGED,
            config=config,
        )

        result = processor.process_all_cameras()

        logger.info("=" * 60)
        if result["success"]:
            logger.info("Transformation completed successfully")
            for cam, res in result["camera_results"].items():
                logger.info(f"  Camera {cam}: {res.get('transformed_files', 0)} files in {res.get('elapsed_time', 0):.2f}s")
            logger.info("")
            logger.info("NOTE: Statistics files were NOT transformed.")
            logger.info("Please recalculate statistics if needed.")
        else:
            logger.error("Transformation failed for some cameras")
            for cam, res in result["camera_results"].items():
                if not res.get("success"):
                    logger.error(f"  Camera {cam}: {res.get('error')}")

        exit(0 if result["success"] else 1)
