"""
Scale Factor Calibration Service.

Provides reusable calibration logic that can be called from CLI or GUI.
Converts pixel-based measurements to physical units (mm, m/s).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from loguru import logger

from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths


def _process_vector_file(args: Tuple) -> bool:
    """
    Process a single vector file for scale factor calibration.

    This is a module-level function to enable multiprocessing.

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

        if "piv_result" not in mat:
            logger.warning(
                f"Vector file {vector_file_uncal} missing 'piv_result' field."
            )
            return False

        piv_result = mat["piv_result"]

        # Build output struct array
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
                out_piv[idx] = (ux_calib, uy_calib, b_mask)
            else:
                out_piv[idx] = (np.array([]), np.array([]), np.array([]))

        scipy.io.savemat(
            str(vector_file_cal), {"piv_result": out_piv}, do_compression=True
        )
        return True

    except Exception as e:
        logger.error(
            f"Error processing vector file {vector_file_uncal}: {e}", exc_info=True
        )
        return False


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
        dt: float,
        px_per_mm: float,
        base_path: Path,
        source_path_idx: int = 0,
        type_name: str = "instantaneous",
    ):
        """
        Initialize scale factor calibrator.

        Args:
            dt: Time between frames in seconds
            px_per_mm: Pixels per millimeter
            base_path: Base output directory
            source_path_idx: Index into config source_paths (for getting settings)
            type_name: Type of data (instantaneous, ensemble)
        """
        self.dt = dt
        self.px_per_mm = px_per_mm
        self.base_path = Path(base_path)
        self.source_path_idx = source_path_idx
        self.type_name = type_name

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
        }

        if total_files == 0:
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
                        logger.error(f"Future failed with exception: {e}")
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
            mat = scipy.io.loadmat(
                str(coords_path_uncal), struct_as_record=False, squeeze_me=True
            )
            coordinates = mat.get("coordinates", None)

            if coordinates is None:
                logger.warning(f"No 'coordinates' field in {coords_path_uncal}")
                return False

            # Build output struct array
            coord_dtype = np.dtype([("x", "O"), ("y", "O")])
            out_coords = np.empty(len(coordinates), dtype=coord_dtype)

            processed_runs = 0
            for run_idx, run_coords in enumerate(coordinates):
                x = getattr(run_coords, "x", None)
                y = getattr(run_coords, "y", None)

                if x is not None and y is not None:
                    x_calib, y_calib = self.calibrate_coordinates(x, y)
                    out_coords[run_idx] = (x_calib, y_calib)
                    processed_runs += 1
                else:
                    out_coords[run_idx] = (np.array([]), np.array([]))

            scipy.io.savemat(
                str(coords_path_cal), {"coordinates": out_coords}, do_compression=True
            )
            logger.info(f"Updated coordinates for {processed_runs} runs")
            return True

        except Exception as e:
            logger.error(f"Error processing coordinates: {e}", exc_info=True)
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
        }

        if total_files == 0:
            return overall_result

        # Process each camera
        total_processed_files = 0

        for cam_idx, cam_num in enumerate(cameras):
            logger.info(f"Processing camera {cam_num} ({cam_idx + 1}/{total_cameras})")

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

        return overall_result
