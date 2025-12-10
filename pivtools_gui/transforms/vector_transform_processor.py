#!/usr/bin/env python3
"""
Vector Transform Processor

Standalone processor for applying geometric transformations to PIV vector fields.
Can be run via CLI or called from GUI with progress callbacks.

Similar pattern to MultiViewCalibrator from planar_calibration_production.py.

Usage (CLI):
    python -m pivtools_gui.transforms.vector_transform_processor \\
        --base-path /data/experiment \\
        --transformations flip_ud rotate_90_cw \\
        --camera 1
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from loguru import logger

from pivtools_core.config import Config, get_config
from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import is_run_valid

from .transform_operations import (
    COORDINATES_FILENAME,
    VALID_TRANSFORMATIONS,
    apply_transformation_to_coordinates,
    apply_transformation_to_piv_result,
    backup_original_data,
    coords_mat_to_saveable,
    coords_to_structured_array,
    has_original_backup,
    load_mat_for_transform,
    mat_dict_to_saveable,
    process_frame_worker,
    restore_original_data,
    save_mat_from_transform,
    simplify_transformations,
    validate_transformations,
)


class VectorTransformProcessor:
    """
    Standalone processor for applying geometric transformations to PIV vector fields.

    Supports:
    - Single frame transformation with preview/backup capability
    - Batch transformation across all frames and cameras
    - Progress callbacks for GUI integration
    - CLI mode for batch processing

    Example (programmatic):
        processor = VectorTransformProcessor(
            base_path=Path("/data/experiment"),
            transformations=["flip_ud", "rotate_90_cw"],
            camera=1,
        )
        result = processor.transform_all_frames(
            source_frame=1,
            source_camera=1,
            progress_callback=my_progress_handler,
        )

    Example (CLI):
        python -m pivtools_gui.transforms.vector_transform_processor \\
            --base-path /data/experiment \\
            --transformations flip_ud \\
            --camera 1
    """

    def __init__(
        self,
        base_path: Path,
        transformations: List[str],
        camera: Optional[int] = None,
        type_name: str = "instantaneous",
        use_merged: bool = False,
        config: Optional[Config] = None,
    ):
        """
        Initialize the processor with transformation parameters.

        Args:
            base_path: Base directory containing PIV data
            transformations: List of transformations to apply (in order)
            camera: Camera number (None for all cameras from config)
            type_name: Data type ('instantaneous', 'ensemble', 'statistics')
            use_merged: Whether to use merged data
            config: Optional config object (loads default if None)
        """
        self.base_path = Path(base_path)
        self.transformations = transformations
        self.camera = camera
        self.type_name = type_name
        self.use_merged = use_merged
        self.config = config or get_config()

        # Validate transformations on init (allow empty for status/clear operations)
        if transformations:
            is_valid, error = validate_transformations(transformations)
            if not is_valid:
                raise ValueError(error)

        # Validate config has required keys
        if not hasattr(self.config, 'vector_format') or not self.config.vector_format:
            raise ValueError("Config missing required 'vector_format'")
        if not hasattr(self.config, 'num_frame_pairs') or self.config.num_frame_pairs <= 0:
            raise ValueError("Config missing or invalid 'num_frame_pairs'")

    def _get_data_paths(self, camera: int) -> Dict:
        """Get data paths for a specific camera."""
        return get_data_paths(
            base_dir=self.base_path,
            num_frame_pairs=self.config.num_frame_pairs,
            cam=camera,
            type_name=self.type_name,
            use_merged=self.use_merged,
        )

    def _get_mat_file_path(self, data_dir: Path, frame: int) -> Path:
        """
        Get the correct mat file path based on type_name.

        For ensemble data, returns ensemble_result.mat.
        For instantaneous data, returns the frame-numbered file.

        Args:
            data_dir: Directory containing the mat files
            frame: Frame number (ignored for ensemble)

        Returns:
            Path to the mat file
        """
        if self.type_name == "ensemble":
            return data_dir / "ensemble_result.mat"
        else:
            return data_dir / (self.config.vector_format % frame)

    def _get_piv_result_key(self) -> str:
        """Get the key name for piv_result based on type_name."""
        if self.type_name == "ensemble":
            return "ensemble_result"
        else:
            return "piv_result"

    def transform_single_frame(
        self,
        frame: int,
        camera: int,
        transformation: Optional[str] = None,
    ) -> Dict:
        """
        Transform a single frame with backup capability.

        Creates a backup on first transformation, allowing later restoration.
        Subsequent transformations are cumulative.

        Args:
            frame: Frame number (1-based)
            camera: Camera number
            transformation: Single transformation to apply (uses first from list if None)

        Returns:
            Dict with keys:
            - success: bool
            - has_original: bool (True if backup exists)
            - pending_transformations: list of applied transformations
            - error: str (if failed)
        """
        trans = transformation or (self.transformations[0] if self.transformations else None)
        if not trans:
            return {"success": False, "error": "No transformation specified"}

        is_valid, error = validate_transformations([trans])
        if not is_valid:
            return {"success": False, "error": error}

        try:
            paths = self._get_data_paths(camera)
            data_dir = Path(paths["data_dir"])

            # Load the mat file (handles ensemble vs instantaneous)
            mat_file = self._get_mat_file_path(data_dir, frame)
            if not mat_file.exists():
                return {"success": False, "error": f"Data file not found: {mat_file}"}

            mat = load_mat_for_transform(mat_file)
            piv_result_key = self._get_piv_result_key()
            if piv_result_key not in mat:
                return {"success": False, "error": f"'{piv_result_key}' not found in {mat_file.name}"}
            piv_result = mat[piv_result_key]

            # Load coordinates if they exist
            coords_file = data_dir / COORDINATES_FILENAME
            coords_mat = None
            coords = None
            if coords_file.exists():
                coords_mat = load_mat_for_transform(coords_file)
                coords = coords_mat["coordinates"]

            # Create backups on first transformation
            mat, coords_mat = backup_original_data(mat, coords_mat)

            # Initialize or update pending transformations list with robust conversion
            if "pending_transformations" not in mat:
                mat["pending_transformations"] = []
            else:
                pt = mat["pending_transformations"]
                if isinstance(pt, np.ndarray):
                    # Handle numpy arrays (may be object array of strings)
                    if pt.ndim == 0:
                        # Scalar array (single element)
                        item = pt.item()
                        mat["pending_transformations"] = [str(item)] if item else []
                    else:
                        # Multi-element array
                        mat["pending_transformations"] = [str(x) for x in pt.flatten().tolist()]
                elif isinstance(pt, str):
                    # Single string
                    mat["pending_transformations"] = [pt]
                elif not isinstance(pt, list):
                    # Other iterable
                    mat["pending_transformations"] = list(pt) if pt else []

            mat["pending_transformations"].append(trans)
            # Simplify the transformation list (e.g., rotate_90_cw + rotate_90_ccw = [])
            mat["pending_transformations"] = simplify_transformations(mat["pending_transformations"])

            # Apply transformation to all non-empty runs in piv_result
            if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
                num_runs = piv_result.size
                for run_idx in range(num_runs):
                    pr = piv_result[run_idx]
                    try:
                        # Use centralized validation from vector_loading
                        if is_run_valid(pr, fields=("ux",), require_2d=False, reject_all_nan=True):
                            apply_transformation_to_piv_result(pr, trans)
                            if coords is not None:
                                apply_transformation_to_coordinates(
                                    coords, run_idx + 1, trans
                                )
                    except Exception as e:
                        logger.warning(f"Error transforming run {run_idx + 1}: {e}")
            else:
                # Single run
                apply_transformation_to_piv_result(piv_result, trans)
                if coords is not None:
                    apply_transformation_to_coordinates(coords, 1, trans)

            # Save back the mat file (convert structs to proper format)
            save_mat_from_transform(mat_file, mat_dict_to_saveable(mat))

            # Save coordinates if they were loaded
            if coords_mat is not None:
                save_mat_from_transform(coords_file, coords_mat_to_saveable(coords_mat))

            # Also save to config.yaml for CLI integration
            config = get_config()
            config.set_camera_transforms(camera, mat["pending_transformations"])
            config.save()

            return {
                "success": True,
                "has_original": True,
                "pending_transformations": mat["pending_transformations"],
            }

        except Exception as e:
            logger.exception(f"Error transforming frame {frame}: {e}")
            return {"success": False, "error": str(e)}

    def clear_transformations(self, frame: int, camera: int) -> Dict:
        """
        Restore original data for a frame (undo all transformations).

        Args:
            frame: Frame number (1-based)
            camera: Camera number

        Returns:
            Dict with keys:
            - success: bool
            - has_original: bool (False after clear)
            - error: str (if failed)
        """
        try:
            paths = self._get_data_paths(camera)
            data_dir = Path(paths["data_dir"])

            # Load the mat file (handles ensemble vs instantaneous)
            mat_file = self._get_mat_file_path(data_dir, frame)
            if not mat_file.exists():
                return {"success": False, "error": f"Data file not found: {mat_file}"}

            mat = load_mat_for_transform(mat_file)

            if not has_original_backup(mat):
                return {"success": False, "error": "No original backup found for this frame"}

            # Load coordinates if they exist
            coords_file = data_dir / COORDINATES_FILENAME
            coords_mat = None
            if coords_file.exists():
                coords_mat = load_mat_for_transform(coords_file)

            # Restore from backups
            mat, coords_mat = restore_original_data(mat, coords_mat)

            # Save back (convert structs to proper format)
            save_mat_from_transform(mat_file, mat_dict_to_saveable(mat))

            if coords_mat is not None:
                save_mat_from_transform(coords_file, coords_mat_to_saveable(coords_mat))

            # Clear transforms from config.yaml
            config = get_config()
            config.clear_camera_transforms(camera)
            config.save()

            return {"success": True, "has_original": False}

        except Exception as e:
            logger.exception(f"Error clearing transformations: {e}")
            return {"success": False, "error": str(e)}

    def get_transform_status(self, frame: int, camera: int) -> Dict:
        """
        Check transformation status for a frame.

        Args:
            frame: Frame number (1-based)
            camera: Camera number

        Returns:
            Dict with keys:
            - success: bool
            - has_original: bool
            - pending_transformations: list
            - error: str (if failed)
        """
        try:
            paths = self._get_data_paths(camera)
            data_dir = Path(paths["data_dir"])

            # Load the mat file (handles ensemble vs instantaneous)
            mat_file = self._get_mat_file_path(data_dir, frame)
            if not mat_file.exists():
                return {"success": False, "error": f"Data file not found: {mat_file.name}"}

            mat = load_mat_for_transform(mat_file)

            has_backup = has_original_backup(mat)
            pending_transforms = []
            if "pending_transformations" in mat:
                pt = mat["pending_transformations"]
                if isinstance(pt, np.ndarray):
                    pending_transforms = pt.tolist()
                elif isinstance(pt, list):
                    pending_transforms = pt
                else:
                    pending_transforms = [str(pt)]

            return {
                "success": True,
                "has_original": has_backup,
                "pending_transformations": pending_transforms,
            }

        except Exception as e:
            logger.exception(f"Error getting transform status: {e}")
            return {"success": False, "error": str(e)}

    def transform_all_frames(
        self,
        source_frame: int,
        source_camera: int,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        max_workers: int = 8,
    ) -> Dict:
        """
        Apply transformations to all frames across cameras.

        Gets pending transformations from the source frame, then applies them
        to all other frames. Removes backups from source frame after success.

        Args:
            source_frame: Frame that has pending transformations
            source_camera: Camera that has pending transformations
            progress_callback: Optional function called with progress dict:
                - progress: int (0-100)
                - processed_frames: int
                - total_frames: int
                - processed_cameras: int
                - total_cameras: int
                - current_camera: int
            max_workers: Maximum parallel workers for processing

        Returns:
            Dict with keys:
            - success: bool
            - total_frames: int
            - total_cameras: int
            - elapsed_time: float
            - error: str (if failed)
        """
        start_time = time.time()

        try:
            # Load source frame to get pending transformations
            source_paths = self._get_data_paths(source_camera)
            source_data_dir = Path(source_paths["data_dir"])
            source_mat_file = self._get_mat_file_path(source_data_dir, source_frame)

            if not source_mat_file.exists():
                return {"success": False, "error": f"Source data file not found: {source_mat_file}"}

            source_mat = load_mat_for_transform(source_mat_file)

            # Get pending transformations
            if "pending_transformations" not in source_mat:
                return {"success": False, "error": "No pending transformations found on source frame"}

            transformations = source_mat["pending_transformations"]
            if isinstance(transformations, np.ndarray):
                transformations = transformations.tolist()
            elif not isinstance(transformations, list):
                transformations = [str(transformations)]

            if not transformations:
                return {"success": False, "error": "No transformations to apply"}

            # Validate transformations
            is_valid, error = validate_transformations(transformations)
            if not is_valid:
                return {"success": False, "error": error}

            logger.info(f"Applying transformations: {transformations}")

            # Remove backups from source frame (it's already transformed)
            # Handle both piv_result and ensemble_result
            if "piv_result_original" in source_mat:
                del source_mat["piv_result_original"]
            if "ensemble_result_original" in source_mat:
                del source_mat["ensemble_result_original"]
            source_mat["pending_transformations"] = []

            save_mat_from_transform(source_mat_file, mat_dict_to_saveable(source_mat))

            # Remove original from source coordinates
            source_coords_file = source_data_dir / COORDINATES_FILENAME
            if source_coords_file.exists():
                coords_mat = load_mat_for_transform(source_coords_file)
                if "coordinates_original" in coords_mat:
                    del coords_mat["coordinates_original"]
                    save_mat_from_transform(source_coords_file, coords_mat_to_saveable(coords_mat))

            # Get cameras to process
            all_cameras = self.config.camera_numbers if self.camera is None else [self.camera]

            # Calculate total work
            total_frames_to_process = 0
            camera_frame_map = {}

            for cam in all_cameras:
                paths = self._get_data_paths(cam)
                data_dir = Path(paths["data_dir"])

                vector_files = []

                if self.type_name == "ensemble":
                    # Ensemble has a single file - only process other cameras
                    if cam != source_camera:
                        mat_file = self._get_mat_file_path(data_dir, 1)  # frame ignored for ensemble
                        if mat_file.exists():
                            vector_files.append((1, mat_file))
                else:
                    # Instantaneous - process all frames
                    for frame in range(1, self.config.num_frame_pairs + 1):
                        # Skip source frame for source camera (already transformed)
                        if cam == source_camera and frame == source_frame:
                            continue

                        mat_file = self._get_mat_file_path(data_dir, frame)
                        if mat_file.exists():
                            vector_files.append((frame, mat_file))

                camera_frame_map[cam] = {
                    "data_dir": data_dir,
                    "vector_files": vector_files,
                }
                total_frames_to_process += len(vector_files)

            if total_frames_to_process == 0:
                return {
                    "success": True,
                    "total_frames": 0,
                    "total_cameras": len(all_cameras),
                    "elapsed_time": time.time() - start_time,
                }

            processed_frames = 0
            processed_cameras = 0

            # Process each camera
            for cam in all_cameras:
                cam_data = camera_frame_map[cam]
                data_dir = cam_data["data_dir"]
                vector_files = cam_data["vector_files"]

                if not vector_files:
                    processed_cameras += 1
                    continue

                # Process coordinates for this camera
                coords_file = data_dir / COORDINATES_FILENAME

                if coords_file.exists():
                    logger.info(f"Transforming coordinates for camera {cam}")
                    coords_mat = load_mat_for_transform(coords_file)
                    coords = coords_mat["coordinates"]

                    # Apply transformations to all runs in coordinates
                    if isinstance(coords, np.ndarray) and coords.dtype == object:
                        num_coord_runs = coords.size
                        for run_idx in range(num_coord_runs):
                            for trans in transformations:
                                apply_transformation_to_coordinates(coords, run_idx + 1, trans)
                    else:
                        for trans in transformations:
                            apply_transformation_to_coordinates(coords, 1, trans)

                    coords_struct = coords_to_structured_array(coords)
                    save_mat_from_transform(coords_file, {"coordinates": coords_struct})

                # Process frames in parallel
                num_workers = min(os.cpu_count() or 1, len(vector_files), max_workers)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            process_frame_worker,
                            frame,
                            mat_file,
                            coords_file if coords_file.exists() else None,
                            transformations,
                        )
                        for frame, mat_file in vector_files
                    ]

                    for future in as_completed(futures):
                        future.result()  # Check for exceptions
                        processed_frames += 1

                        if progress_callback:
                            progress = int((processed_frames / total_frames_to_process) * 100)
                            progress_callback({
                                "progress": progress,
                                "processed_frames": processed_frames,
                                "total_frames": total_frames_to_process,
                                "processed_cameras": processed_cameras,
                                "total_cameras": len(all_cameras),
                                "current_camera": cam,
                            })

                processed_cameras += 1
                logger.info(f"Completed camera {cam} ({processed_cameras}/{len(all_cameras)})")

            elapsed_time = time.time() - start_time
            logger.info(
                f"Transformations complete: {total_frames_to_process} frames "
                f"across {len(all_cameras)} cameras in {elapsed_time:.1f}s"
            )

            return {
                "success": True,
                "total_frames": total_frames_to_process,
                "total_cameras": len(all_cameras),
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            logger.exception(f"Error in transform_all_frames: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Apply geometric transformations to PIV vector fields"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base directory containing PIV data",
    )
    parser.add_argument(
        "--transformations",
        nargs="+",
        required=True,
        choices=VALID_TRANSFORMATIONS,
        help="Transformations to apply (in order)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera number (omit to process all cameras)",
    )
    parser.add_argument(
        "--type",
        default="instantaneous",
        choices=["instantaneous", "ensemble", "statistics"],
        help="Data type to transform",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Transform merged data",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    def cli_progress_callback(info: Dict):
        """Print progress to terminal."""
        progress = info.get("progress", 0)
        processed = info.get("processed_frames", 0)
        total = info.get("total_frames", 0)
        camera = info.get("current_camera", "?")
        print(f"\rCamera {camera}: {progress}% ({processed}/{total} frames)", end="", flush=True)

    print("PIVTools Vector Transformation")
    print(f"Base path: {args.base_path}")
    print(f"Transformations: {' -> '.join(args.transformations)}")
    print(f"Camera: {args.camera or 'all'}")
    print(f"Type: {args.type}")
    print()

    processor = VectorTransformProcessor(
        base_path=args.base_path,
        transformations=args.transformations,
        camera=args.camera,
        type_name=args.type,
        use_merged=args.merged,
    )

    # For CLI, we need a "source frame" - use frame 1
    # First apply transformations to frame 1 to set up pending list
    print("Applying to initial frame...")
    for trans in args.transformations:
        result = processor.transform_single_frame(frame=1, camera=args.camera or 1, transformation=trans)
        if not result["success"]:
            print(f"Error: {result['error']}")
            sys.exit(1)

    print("Applying to all frames...")
    result = processor.transform_all_frames(
        source_frame=1,
        source_camera=args.camera or 1,
        progress_callback=cli_progress_callback,
        max_workers=args.max_workers,
    )

    print()  # New line after progress

    if result["success"]:
        print(f"Success: {result['total_frames']} frames transformed")
        print(f"Elapsed time: {result['elapsed_time']:.1f}s")
        sys.exit(0)
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)
