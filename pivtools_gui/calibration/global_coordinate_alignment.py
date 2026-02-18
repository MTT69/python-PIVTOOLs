"""
Global Coordinate Alignment Module.

Provides coordinate alignment across multiple cameras by:
1. Converting a user-selected datum pixel to calibrated physical coordinates
2. Computing shifts to place the datum at a desired physical origin
3. Chaining shifts through adjacent camera pairs (chain topology)
4. Optionally negating ux (and UV_stress) when invert_ux is enabled

Supports scale_factor, dotboard, and charuco calibration methods.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.io import loadmat, savemat

from pivtools_core.config import Config
from pivtools_core.coordinate_utils import extract_coordinates
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import load_coords_from_directory


class GlobalCoordinateAligner:
    """Aligns coordinates across cameras using datum and overlap pairs."""

    ALIGNMENT_MARKER = "alignment_applied.json"

    def __init__(self, base_dir, config: Config):
        self.base_dir = Path(base_dir)
        self.config = config

    def _get_marker_path(self, camera_num: int, type_name: str) -> Path:
        """Path to alignment marker for a camera."""
        paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=self.config.num_frame_pairs,
            cam=camera_num,
            type_name=type_name,
        )
        return paths["data_dir"] / self.ALIGNMENT_MARKER

    def _check_already_aligned(self, type_name: str) -> list:
        """Check if any cameras have alignment markers. Returns list of already-aligned cameras."""
        aligned = []
        cameras_to_check = set()
        gc = self.config.global_coordinates_config
        datum_cam = gc.get("datum_camera", 1)
        cameras_to_check.add(datum_cam)
        for pair in self.config.global_coordinates_overlap_pairs:
            cameras_to_check.add(pair["camera_a"])
            cameras_to_check.add(pair["camera_b"])
        for cam in sorted(cameras_to_check):
            marker = self._get_marker_path(cam, type_name)
            if marker.exists():
                aligned.append(cam)
        return aligned

    def _write_alignment_marker(self, camera_num: int, type_name: str, shift_x: float, shift_y: float):
        """Write alignment marker after successful shift."""
        import json
        from datetime import datetime

        marker_path = self._get_marker_path(camera_num, type_name)
        marker_data = {
            "aligned_at": datetime.now().isoformat(),
            "shift_x": shift_x,
            "shift_y": shift_y,
            "method": self.config.active_calibration_method,
        }
        marker_path.write_text(json.dumps(marker_data, indent=2))

    @staticmethod
    def clear_alignment_marker(data_dir: Path):
        """Remove alignment marker from a data directory (called after fresh calibration)."""
        marker = data_dir / GlobalCoordinateAligner.ALIGNMENT_MARKER
        if marker.exists():
            marker.unlink()

    def pixel_to_calibrated_physical(
        self,
        pixel_xy: Tuple[float, float],
        camera_num: int,
        type_name: str,
    ) -> Tuple[float, float]:
        """Convert an image pixel to calibrated physical mm.

        Uses the active calibration method's model to perform the conversion.

        Parameters
        ----------
        pixel_xy : tuple of (float, float)
            Pixel coordinates (x, y) in image space
        camera_num : int
            Camera number (1-based)
        type_name : str
            'instantaneous' or 'ensemble'

        Returns
        -------
        tuple of (float, float)
            Physical coordinates (x_mm, y_mm)
        """
        method = self.config.active_calibration_method

        if method == "scale_factor":
            return self._pixel_to_physical_scale_factor(pixel_xy, camera_num, type_name)
        elif method in ("dotboard", "charuco"):
            return self._pixel_to_physical_pinhole(pixel_xy, camera_num, method)
        else:
            raise ValueError(
                f"Global coordinate alignment not supported for method '{method}'. "
                f"Supported: scale_factor, dotboard, charuco"
            )

    def precompute_camera_shifts(self, type_name: str) -> Optional[Dict]:
        """Pre-compute alignment shifts for all cameras. No data file I/O.

        Uses the calibration model to convert datum/overlap pixels to physical
        coordinates and compute the chain of shifts. Does NOT read or write any
        data .mat files — only reads model files via pixel_to_calibrated_physical().

        Parameters
        ----------
        type_name : str
            'instantaneous' or 'ensemble'

        Returns
        -------
        dict or None
            None if alignment is disabled. Otherwise a dict:
            {
                "camera_shifts": {cam_num: (shift_x, shift_y), ...},
                "invert_ux": bool,
                "datum_physical_x": float,
            }
        """
        gc = self.config.global_coordinates_config
        if not gc.get("enabled", False):
            return None

        datum_pixel = gc.get("datum_pixel")
        if datum_pixel is None:
            raise ValueError(
                "datum_pixel not set in global_coordinates config. "
                "Click a point on the datum camera image to set the datum pixel."
            )

        method = self.config.active_calibration_method
        if method == "polynomial":
            raise ValueError(
                "Global coordinate alignment not supported for method 'polynomial'. "
                "Supported methods: scale_factor, dotboard, charuco. "
                "Change the active calibration method first."
            )

        datum_physical = gc.get("datum_physical", [0.0, 0.0])
        overlap_pairs = self.config.global_coordinates_overlap_pairs
        invert_ux = gc.get("invert_ux", False)
        datum_camera = gc.get("datum_camera", 1)

        # Convert datum pixel to calibrated physical on datum camera
        datum_calibrated = self.pixel_to_calibrated_physical(
            tuple(datum_pixel), datum_camera, type_name
        )
        logger.debug(
            f"Datum pixel {datum_pixel} on cam {datum_camera} -> "
            f"calibrated physical ({datum_calibrated[0]:.4f}, {datum_calibrated[1]:.4f}) mm"
        )

        # Compute datum shift
        datum_shift_x = datum_physical[0] - datum_calibrated[0]
        datum_shift_y = datum_physical[1] - datum_calibrated[1]
        camera_shifts = {datum_camera: (datum_shift_x, datum_shift_y)}

        # Chain through pairs
        for pair in sorted(overlap_pairs, key=lambda p: (p["camera_a"], p["camera_b"])):
            cam_a = pair["camera_a"]
            cam_b = pair["camera_b"]
            pixel_a = pair.get("pixel_on_a")
            pixel_b = pair.get("pixel_on_b")

            if pixel_a is None or pixel_b is None:
                logger.warning(
                    f"Skipping pair ({cam_a}, {cam_b}): incomplete feature points"
                )
                continue

            if cam_a not in camera_shifts:
                logger.warning(
                    f"Skipping pair ({cam_a}, {cam_b}): cam {cam_a} has no shift yet "
                    f"(chain broken)"
                )
                continue

            phys_a = self.pixel_to_calibrated_physical(tuple(pixel_a), cam_a, type_name)
            shift_a = camera_shifts[cam_a]
            phys_a_shifted = (phys_a[0] + shift_a[0], phys_a[1] + shift_a[1])

            phys_b = self.pixel_to_calibrated_physical(tuple(pixel_b), cam_b, type_name)

            shift_b_x = phys_a_shifted[0] - phys_b[0]
            shift_b_y = phys_a_shifted[1] - phys_b[1]
            camera_shifts[cam_b] = (shift_b_x, shift_b_y)
            logger.info(
                f"Camera {cam_b} shift: ({shift_b_x:.4f}, {shift_b_y:.4f}) mm "
                f"(from pair ({cam_a}, {cam_b}))"
            )

        return {
            "camera_shifts": camera_shifts,
            "invert_ux": invert_ux,
            "datum_physical_x": datum_physical[0],
        }

    def apply_alignment(self, type_name: str, force: bool = False) -> Dict[str, any]:
        """Apply global coordinate alignment to all cameras using chain topology.

        Steps:
        1. Pre-compute shifts via precompute_camera_shifts()
        2. Apply shifts to each camera's coordinates.mat
        3. If invert_ux is enabled, negate ux and UV_stress for all cameras

        Parameters
        ----------
        type_name : str
            'instantaneous' or 'ensemble'
        force : bool
            If True, skip idempotency check and apply even if already aligned.

        Returns
        -------
        dict
            Results including shifts applied per camera
        """
        gc = self.config.global_coordinates_config
        logger.debug(f"apply_alignment: global_coordinates config = {gc}")

        # Pre-compute shifts (validates config, reads only model files)
        alignment = self.precompute_camera_shifts(type_name)
        if alignment is None:
            logger.info("Global coordinate alignment is disabled, skipping")
            return {"status": "skipped", "reason": "disabled"}

        overlap_pairs = self.config.global_coordinates_overlap_pairs
        num_cameras = self.config.camera_count
        if num_cameras > 1 and not overlap_pairs:
            logger.warning(
                f"Multi-camera setup ({num_cameras} cameras) but no overlap_pairs configured. "
                f"Only the datum camera will be aligned. Set overlap pairs to chain all cameras."
            )

        # Idempotency guard: check for already-aligned cameras
        if not force:
            already_aligned = self._check_already_aligned(type_name)
            if already_aligned:
                msg = (
                    f"Cameras {already_aligned} already have alignment applied. "
                    f"Re-running will double the coordinate shifts. "
                    f"Re-calibrate first to get fresh coordinates, or use force=True to override."
                )
                raise ValueError(msg)

        camera_shifts = alignment["camera_shifts"]
        invert_ux = alignment["invert_ux"]
        datum_physical_x = alignment["datum_physical_x"]
        datum_camera = gc.get("datum_camera", 1)

        logger.debug(
            f"apply_alignment: invert_ux={invert_ux}, camera_shifts={camera_shifts}"
        )

        # Apply shifts to each camera's coordinates.mat
        results = {"cameras": {}}
        for cam, (sx, sy) in camera_shifts.items():
            self._apply_shift_to_camera(cam, type_name, sx, sy)
            source = "datum" if cam == datum_camera else f"overlap chain"
            results["cameras"][cam] = {
                "shift_x": sx,
                "shift_y": sy,
                "source": source,
            }

        # Apply invert_ux if needed
        if invert_ux:
            all_cameras = set(camera_shifts.keys())
            for p in overlap_pairs:
                all_cameras.add(p["camera_a"])
                all_cameras.add(p["camera_b"])
            for cam in sorted(all_cameras):
                self._apply_invert_ux_to_camera(cam, type_name, datum_physical_x)
            logger.info(f"Applied invert_ux to {len(all_cameras)} cameras")

        results["status"] = "completed"
        results["invert_ux"] = invert_ux
        logger.info(f"Global coordinate alignment completed for {len(results['cameras'])} cameras")
        return results

    def _pixel_to_physical_scale_factor(
        self,
        pixel_xy: Tuple[float, float],
        camera_num: int,
        type_name: str,
    ) -> Tuple[float, float]:
        """Convert pixel to physical using scale factor method.

        Replicates ScaleFactorCalibrator.calibrate_coordinates() logic.

        Raw pixel (px_x, px_y) is first converted to uncalibrated coordinates
        (matching save_results.py: x_uncal = px_x + 1, y_uncal = H - px_y),
        then the calibration formula is applied:
          x_mm = (x_uncal - grid_x0) / px_per_mm
          y_mm = -(y_uncal - grid_y0) / px_per_mm
        """
        sf_cfg = self.config.data.get("calibration", {}).get("scale_factor", {})
        px_per_mm = sf_cfg.get("px_per_mm", 1.0)

        # Load uncalibrated coordinates to get grid origin
        paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=self.config.num_frame_pairs,
            cam=camera_num,
            type_name=type_name,
            use_uncalibrated=True,
        )
        coords_path = paths["data_dir"] / "coordinates.mat"
        if not coords_path.exists():
            raise FileNotFoundError(
                f"Uncalibrated coordinates not found: {coords_path}. "
                f"Run PIV processing first."
            )

        mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        coords = mat["coordinates"]
        # Get first run's coordinates
        if hasattr(coords, "__len__") and not isinstance(coords, np.void):
            first = coords[0]
        else:
            first = coords
        x = np.asarray(first.x)
        y = np.asarray(first.y)

        # Grid reference points (same as ScaleFactorCalibrator.calibrate_coordinates)
        grid_x0 = x.flat[0] if x.size > 0 else 0
        grid_y0 = float(np.min(y)) if y.size > 0 else 0

        # Convert raw pixel to uncalibrated coordinates (matching save_results.py)
        # save_results.py: x_grid = x_centers + 1, y_grid = (H-1) - y_centers + 1 = H - y_centers
        H = self.config.image_shape[0]
        px_x, px_y = pixel_xy
        px_x_uncal = px_x + 1       # 1-based
        px_y_uncal = H - px_y        # flipped: image-downward → physical-upward, 1-based

        # Apply calibration formula (pointwise equivalent of calibrate_coordinates)
        # y=0 at bottom of grid, y increasing upward (matching pinhole convention)
        x_mm = (px_x_uncal - grid_x0) / px_per_mm
        y_mm = (px_y_uncal - grid_y0) / px_per_mm

        return (x_mm, y_mm)

    def _pixel_to_physical_pinhole(
        self,
        pixel_xy: Tuple[float, float],
        camera_num: int,
        method: str,
    ) -> Tuple[float, float]:
        """Convert pixel to physical using dotboard or charuco pinhole model.

        Uses _pixels_to_world_mm from vector_calibration_production.py logic
        with the saved camera model.

        The production calibration (vector_calibration_production.py) passes
        *uncalibrated* coordinates to the pinhole model, not raw pixels.
        Uncalibrated coords use: x = px_x + 1, y = H - px_y (from save_results.py).
        We must do the same conversion here for consistency.
        """
        model = self._load_pinhole_model(camera_num, method)
        camera_matrix = model["camera_matrix"]
        dist_coeffs = model["dist_coeffs"]
        rvec = model["rvec"]
        tvec = model["tvec"]

        # Convert raw pixel to uncalibrated coordinates (matching save_results.py)
        # This is what the production calibration passes to _pixels_to_world_mm
        H = self.config.image_shape[0]
        px_x_uncal = pixel_xy[0] + 1    # 1-based
        px_y_uncal = H - pixel_xy[1]     # flipped: image-downward → physical-upward

        pts_uncal = np.array([[px_x_uncal, px_y_uncal]], dtype=np.float32)
        world_pts = _pixels_to_world_mm(pts_uncal, camera_matrix, dist_coeffs, rvec, tvec)

        return (float(world_pts[0, 0]), float(world_pts[0, 1]))

    def _load_pinhole_model(self, camera_num: int, method: str) -> dict:
        """Load calibration model for dotboard or charuco."""
        calib_paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=1,
            cam=camera_num,
            type_name="",
            calibration=True,
        )
        calib_dir = calib_paths["calib_dir"]

        if method == "charuco":
            model_path = calib_dir / "charuco_planar" / "model" / "camera_model.mat"
        else:  # dotboard
            model_path = calib_dir / "dotboard_planar" / "model" / "dotboard_model.mat"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Calibration model not found: {model_path}. "
                f"Run 'Generate Model' first."
            )

        model_data = loadmat(str(model_path), squeeze_me=True, struct_as_record=False)

        camera_matrix = np.array(model_data["camera_matrix"]).astype(np.float64)
        dist_coeffs = np.array(model_data["dist_coeffs"]).flatten().astype(np.float64)

        rvecs = model_data["rvecs"]
        tvecs = model_data["tvecs"]

        # Use first view's rvec/tvec (or datum_frame's if we extend later)
        if rvecs.ndim == 1:
            rvec = rvecs.astype(np.float64)
            tvec = tvecs.astype(np.float64)
        else:
            rvec = rvecs[0].flatten().astype(np.float64)
            tvec = tvecs[0].flatten().astype(np.float64)

        return {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvec": rvec,
            "tvec": tvec,
        }

    def _apply_shift_to_camera(
        self,
        camera_num: int,
        type_name: str,
        shift_x: float,
        shift_y: float,
    ):
        """Apply coordinate shift to a camera's coordinates.mat (all runs).

        Loads the calibrated coordinates, adds shift to x and y arrays, saves back.
        """
        paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=self.config.num_frame_pairs,
            cam=camera_num,
            type_name=type_name,
        )
        coords_path = paths["data_dir"] / "coordinates.mat"

        if not coords_path.exists():
            raise FileNotFoundError(
                f"Calibrated coordinates not found: {coords_path}. "
                f"Run vector calibration first."
            )

        mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        coordinates = mat["coordinates"]

        # Determine number of runs
        if hasattr(coordinates, "__len__") and not isinstance(coordinates, np.void):
            num_runs = len(coordinates)
        else:
            num_runs = 1

        # Check for stereo (z field)
        first_coord = coordinates[0] if num_runs > 1 else coordinates
        has_z = (
            hasattr(first_coord, "z")
            if hasattr(first_coord, "__getattr__")
            else (
                "z" in first_coord.dtype.names
                if hasattr(first_coord, "dtype") and first_coord.dtype.names
                else False
            )
        )

        if has_z:
            dtype = [("x", object), ("y", object), ("z", object)]
        else:
            dtype = [("x", object), ("y", object)]
        coords_struct = np.empty((num_runs,), dtype=dtype)

        for i in range(num_runs):
            cx, cy = extract_coordinates(mat["coordinates"], i + 1)

            if has_z:
                c_el = coordinates[i] if num_runs > 1 else coordinates
                cz = np.asarray(c_el.z) if hasattr(c_el, "z") else np.asarray(c_el["z"])

            cx = cx + shift_x
            cy = cy + shift_y

            coords_struct["x"][i] = cx
            coords_struct["y"][i] = cy
            if has_z:
                coords_struct["z"][i] = cz

        savemat(str(coords_path), {"coordinates": coords_struct}, do_compression=True)

        # Write alignment marker
        self._write_alignment_marker(camera_num, type_name, shift_x, shift_y)

        logger.info(
            f"Applied shift ({shift_x:.4f}, {shift_y:.4f}) mm to "
            f"Cam{camera_num} {type_name} ({num_runs} runs)"
        )

    def _apply_invert_ux_to_camera(
        self, camera_num: int, type_name: str, datum_physical_x: float
    ):
        """Negate ux and UV_stress in vectors, and reflect x in coordinates.

        This implements the physical inversion of the x-velocity component
        when the positive x-direction in the global coordinate system is
        opposite to the image-space x-direction. Also reflects x-coordinates
        around datum_physical_x so the axis direction matches.

        Vector files are processed in parallel via ThreadPoolExecutor.
        """
        paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=self.config.num_frame_pairs,
            cam=camera_num,
            type_name=type_name,
        )
        data_dir = paths["data_dir"]

        if type_name == "ensemble":
            mat_files = [data_dir / "ensemble_result.mat"]
            var_name = "ensemble_result"
        else:
            fmt = self.config.vector_format
            mat_files = [
                data_dir / (fmt % i)
                for i in range(1, self.config.num_frame_pairs + 1)
                if (data_dir / (fmt % i)).exists()
            ]
            var_name = "piv_result"

        logger.debug(
            f"invert_ux: Cam{camera_num} {type_name} - "
            f"{len(mat_files)} files, datum_physical_x={datum_physical_x}"
        )

        def _invert_single_file(mat_file: Path) -> bool:
            """Load a single .mat, negate ux/UV_stress, save. Returns True on success."""
            mat = loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
            if var_name not in mat:
                logger.warning(f"invert_ux: {mat_file.name} has no '{var_name}' key")
                return False

            result = mat[var_name]
            if hasattr(result, "__len__") and not isinstance(result, np.void):
                items = result
            else:
                items = [result]

            for item in items:
                if hasattr(item, "ux"):
                    item.ux = -np.asarray(item.ux)
                if hasattr(item, "UV_stress"):
                    item.UV_stress = -np.asarray(item.UV_stress)

            savemat(str(mat_file), mat, do_compression=True)
            return True

        # Process vector files in parallel
        files_processed = 0
        max_workers = min(os.cpu_count() or 4, len(mat_files), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_invert_single_file, f): f for f in mat_files}
            for future in as_completed(futures):
                if future.result():
                    files_processed += 1

        # Reflect x in coordinates.mat (single file, no need for threading)
        coords_path = data_dir / "coordinates.mat"
        if coords_path.exists():
            cmat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
            coordinates = cmat["coordinates"]
            if hasattr(coordinates, "__len__") and not isinstance(coordinates, np.void):
                items = coordinates
            else:
                items = [coordinates]
            for item in items:
                item.x = 2.0 * datum_physical_x - np.asarray(item.x)
            savemat(str(coords_path), cmat, do_compression=True)
        else:
            logger.warning(f"invert_ux: coordinates.mat not found at {coords_path}")

        logger.info(
            f"Applied invert_ux to Cam{camera_num} {type_name} "
            f"({files_processed} files + coordinates)"
        )


def _pixels_to_world_mm(
    pts_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Convert pixel coordinates to world coordinates (mm) on Z=0 plane.

    Duplicated from vector_calibration_production.py for standalone use.
    """
    if pts_px.size == 0:
        return pts_px

    pts_normalized = cv2.undistortPoints(
        pts_px.reshape(-1, 1, 2).astype(np.float32),
        camera_matrix,
        dist_coeffs,
        P=None,
    ).reshape(-1, 2)

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    t = tvec.flatten()

    world_pts = np.zeros((pts_normalized.shape[0], 2), dtype=np.float64)

    for i, (xn, yn) in enumerate(pts_normalized):
        ray = np.array([xn, yn, 1.0])
        ray_world = R_inv @ ray
        t_world = R_inv @ t

        if abs(ray_world[2]) < 1e-10:
            world_pts[i] = [np.nan, np.nan]
            continue

        s = t_world[2] / ray_world[2]
        P_world = s * ray_world - t_world
        world_pts[i] = P_world[:2]

    return world_pts
