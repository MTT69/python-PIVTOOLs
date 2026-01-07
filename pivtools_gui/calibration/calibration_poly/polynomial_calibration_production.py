#!/usr/bin/env python3
"""
polynomial_calibration_production.py

Polynomial (DAVIS) Calibration - Standalone Production Script.

Applies polynomial distortion correction and converts to physical units.
- Reads polynomial coefficients from Calibration.xml or config
- Converts coordinates: pixels -> mm (with distortion correction)
- Converts velocities: pixels/frame -> m/s

Run directly or import PolynomialVectorCalibrator for integration.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from loguru import logger
import numpy as np
from scipy.io import savemat

from pivtools_core.config import get_config, reload_config
from pivtools_core.paths import get_data_paths
from pivtools_core.vector_loading import read_mat_contents, load_coords_from_directory

if TYPE_CHECKING:
    from pivtools_core.config import PIVConfig


# ============================================================================
# HARDCODED SETTINGS - EDIT THESE FOR YOUR EXPERIMENT (CLI mode only)
# ============================================================================

# SOURCE_DIR: Root directory containing source images and Calibration.xml
SOURCE_DIR = "/path/to/source"

# BASE_DIR: Output directory containing uncalibrated PIV results
BASE_DIR = "/path/to/output"

# CAMERA_NUMS: List of camera numbers to process (1-based)
CAMERA_NUMS = [1, 2]

# POLYNOMIAL CALIBRATION PARAMETERS
DT = 70e-6  # Time between frames (seconds) - typical for high-speed PIV

# CALIBRATION SUBFOLDER: Where calibration images and XML are located
# Calibration.xml should be at: SOURCE_DIR/CALIBRATION_SUBFOLDER/Calibration.xml
CALIBRATION_SUBFOLDER = "calibration"

# XML_PATH: Direct path to Calibration.xml (overrides SOURCE_DIR/CALIBRATION_SUBFOLDER)
# Leave empty to use derived path from SOURCE_DIR + CALIBRATION_SUBFOLDER
XML_PATH = ""

# USE_XML: If True, read coefficients from XML; if False, use config.yaml values
USE_XML = True

# PROCESSING OPTIONS
TYPE_NAME = "instantaneous"     # Type of data: "instantaneous" or "ensemble"

# USE_CONFIG_DIRECTLY: If True, skip updating config.yaml with above parameters
# and load calibration settings directly from the existing config.yaml
USE_CONFIG_DIRECTLY = True

# ============================================================================


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
    config.data["paths"]["camera_count"] = len(CAMERA_NUMS)
    config.data["paths"]["camera_numbers"] = CAMERA_NUMS

    # Ensure calibration section exists
    if "calibration" not in config.data:
        config.data["calibration"] = {}

    # Calibration subfolder (where XML and images are)
    config.data["calibration"]["subfolder"] = CALIBRATION_SUBFOLDER

    # Polynomial settings
    if "polynomial" not in config.data["calibration"]:
        config.data["calibration"]["polynomial"] = {}
    config.data["calibration"]["polynomial"]["dt"] = DT

    # Save to disk so other components pick up changes
    config.save()
    logger.info("Updated config.yaml with CLI settings")
    logger.info(f"  dt = {DT} seconds")
    logger.info(f"  calibration_subfolder = {CALIBRATION_SUBFOLDER}")
    logger.info(f"  cameras = {CAMERA_NUMS}")

    # Reload to ensure fresh state
    return reload_config()


# ============================================================================
# XML PARSING
# ============================================================================


def read_calibration_xml(
    source_path_idx: int = 0,
    xml_path: Optional[str] = None,
    config: Optional["PIVConfig"] = None
):
    """
    Read Calibration.xml from direct path or calibration subfolder.

    Args:
        source_path_idx: Index into config source_paths (fallback if xml_path not provided)
        xml_path: Direct path to Calibration.xml (takes priority if provided)
        config: Optional config object (uses get_config() if not provided)

    Returns:
        Dict with status, file path, and cameras data
    """
    cfg = config if config is not None else get_config()

    # Determine XML file path
    if xml_path:
        # Use direct path if provided
        xml_file = Path(xml_path)
        logger.debug(f"Using direct xml_path: {xml_file}")
    else:
        # Fallback to derived path from source_path + subfolder
        logger.debug(f"Starting read_calibration_xml with index: {source_path_idx}")

        # Ensure source_paths exists and has the index
        if not hasattr(cfg, "source_paths") or source_path_idx >= len(cfg.source_paths):
            raise ValueError("Invalid source_path_idx")

        source_root = Path(cfg.source_paths[source_path_idx])

        # XML is in the calibration subfolder (with calibration images)
        calib_subfolder = cfg.data.get("calibration", {}).get("subfolder", "")
        if calib_subfolder:
            xml_file = source_root / calib_subfolder / "Calibration.xml"
        else:
            # Fallback to source root
            xml_file = source_root / "Calibration.xml"

    if not xml_file.exists():
        raise FileNotFoundError(f"Calibration.xml not found at {xml_file}")

    logger.info(f"Reading Calibration.xml from: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    cameras_data = {}

    # Find all CoordinateMapper elements
    mappers = root.findall(".//CoordinateMapper")

    for mapper in mappers:
        cam_id = mapper.get("CameraIdentifier")
        if not cam_id:
            continue

        logger.info(f"Found CoordinateMapper for Camera: {cam_id}")

        # Initialize camera entry
        cam_data = {}

        # Get PolynomialParameters
        poly_params = mapper.find("PolynomialParameters")
        if poly_params is None:
            logger.warning(f"No PolynomialParameters found for {cam_id}")
            continue

        mapping = poly_params.find("PolynomialMapping")
        if mapping is None:
            logger.warning(f"No PolynomialMapping found for {cam_id}")
            continue

        # Extract Origin
        origin = mapping.find("Origin")
        if origin is not None:
            cam_data["origin"] = {k: float(v) for k, v in origin.attrib.items()}
            logger.debug(f"Found Origin for {cam_id}: {cam_data['origin']}")
        else:
            logger.warning(f"No Origin found for {cam_id}")

        # Extract NormalisationFactor
        # Try inside PolynomialMapping first
        norm = mapping.find("NormalisationFactor")
        if norm is None:
            # Try inside PolynomialParameters
            norm = poly_params.find("NormalisationFactor")

        if norm is not None:
            cam_data["normalisation"] = {
                k: float(v) for k, v in norm.attrib.items()
            }
            logger.debug(f"Found NormalisationFactor for {cam_id}: {cam_data['normalisation']}")
        else:
            logger.warning(f"No NormalisationFactor found for {cam_id}")

        # Extract PixelPerMmFactor from CommonParameters
        common_params = poly_params.find("CommonParameters")
        if common_params is not None:
            ppm = common_params.find("PixelPerMmFactor")
            if ppm is not None:
                val = float(ppm.get("Value", 0))
                if val != 0:
                    cam_data["mm_per_pixel"] = 1.0 / val
                    logger.debug(f"Found mm_per_pixel for {cam_id}: {cam_data['mm_per_pixel']}")

        # Extract Polynomial3rdOrder Coefficients
        poly3 = mapping.find("Polynomial3rdOrder")
        if poly3 is not None:
            coeffs_a = poly3.find("CoefficientsA")
            if coeffs_a is not None:
                cam_data["coefficients_a"] = {
                    k: float(v) for k, v in coeffs_a.attrib.items()
                }

            coeffs_b = poly3.find("CoefficientsB")
            if coeffs_b is not None:
                cam_data["coefficients_b"] = {
                    k: float(v) for k, v in coeffs_b.attrib.items()
                }
            logger.debug(f"Found Polynomial coefficients for {cam_id}")
        else:
            logger.warning(f"No Polynomial3rdOrder found for {cam_id}")

        cameras_data[cam_id] = cam_data

    return {"status": "success", "file": str(xml_file), "cameras": cameras_data}


def convert_davis_coeffs_to_array(coeff_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert dictionary of DAVIS coefficients (e.g. {'a_o': 1, 'a_s': 2...})
    to an array of 10 floats matching the order expected by evaluate_polynomial_terms.

    Order: [1, s, s^2, s^3, t, t^2, t^3, s*t, s^2*t, s*t^2]
    Keys can be prefixed with 'a_' or 'b_' or just match the suffix.
    """
    # Map suffixes to indices
    mapping = {
        'o': 0,
        's': 1,
        's2': 2,
        's3': 3,
        't': 4,
        't2': 5,
        't3': 6,
        'st': 7,
        's2t': 8,
        'st2': 9
    }

    arr = np.zeros(10, dtype=float)

    for k, v in coeff_dict.items():
        # Remove prefix 'a_' or 'b_' if present
        if k.startswith('a_') or k.startswith('b_'):
            suffix = k.split('_', 1)[1]
        else:
            suffix = k

        if suffix in mapping:
            arr[mapping[suffix]] = float(v)

    return arr


# ============================================================================
# POLYNOMIAL EVALUATION
# ============================================================================


def evaluate_polynomial_terms(s: np.ndarray, t: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Evaluate DAVIS 3rd-order polynomial:
    coeff order =
        [1, s, s^2, s^3, t, t^2, t^3, s*t, s^2*t, s*t^2]

    Parameters
    ----------
    s : ndarray
        normalized coordinate s(x')
    t : ndarray
        normalized coordinate t(y')
    coeffs : list or array
        polynomial coefficients in DAVIS ordering

    Returns
    -------
    ndarray
        polynomial value (dx or dy)
    """
    s2 = s * s
    s3 = s2 * s
    t2 = t * t
    t3 = t2 * t

    terms = [
        np.ones_like(s),   # 1
        s,
        s2,
        s3,
        t,
        t2,
        t3,
        s * t,
        s2 * t,
        s * t2
    ]

    # sum(coeff_i * term_i)
    out = np.zeros_like(s, dtype=float)
    for c, T in zip(coeffs, terms):
        out += c * T
    return out


# ============================================================================
# CALIBRATOR CLASS
# ============================================================================


class PolynomialVectorCalibrator:
    """
    Polynomial vector calibrator for DAVIS-style calibration.

    Can be initialized either:
    1. With explicit parameters (dx_coeff, dy_coeff, etc.)
    2. From config using get_polynomial_camera_params()
    """

    def __init__(
        self,
        base_dir: Path,
        camera_num: int = 1,
        dt: Optional[float] = None,
        mm_per_pixel: Optional[float] = None,
        dx_coeff: Optional[np.ndarray] = None,
        dy_coeff: Optional[np.ndarray] = None,
        x_origin: Optional[float] = None,
        y_origin: Optional[float] = None,
        nx: Optional[float] = None,
        ny: Optional[float] = None,
        vector_pattern: str = "%05d.mat",
        type_name: str = "instantaneous",
        config: Optional["PIVConfig"] = None,
    ):
        """
        Initialize polynomial vector calibrator.

        Args:
            base_dir: Base output directory
            camera_num: Camera number to process
            dt: Time between frames (reads from config if not provided)
            mm_per_pixel: Scale factor (mm per pixel)
            dx_coeff: X polynomial coefficients (10 terms)
            dy_coeff: Y polynomial coefficients (10 terms)
            x_origin: Polynomial normalization origin X
            y_origin: Polynomial normalization origin Y
            nx: Normalization factor X
            ny: Normalization factor Y
            vector_pattern: Vector file naming pattern
            type_name: Data type (instantaneous, ensemble)
            config: Optional config object to read parameters from
        """
        self.base_dir = Path(base_dir)
        self.camera_num = camera_num
        self._config = config
        self.vector_pattern = vector_pattern
        self.type_name = type_name

        # Read parameters from config if available
        cfg = config if config is not None else get_config()
        poly_cfg = cfg.polynomial_calibration
        cam_params = cfg.get_polynomial_camera_params(camera_num)

        # dt: explicit > config.polynomial.dt > 1.0
        if dt is not None:
            self.dt = dt
        else:
            self.dt = poly_cfg.get("dt", 1.0)

        # Read camera-specific params from config if not explicitly provided
        if cam_params:
            # mm_per_pixel
            if mm_per_pixel is not None:
                self.mm_per_pixel = mm_per_pixel
            else:
                self.mm_per_pixel = cam_params.get("mm_per_pixel", 1.0)

            # Coefficients
            if dx_coeff is not None:
                self.dx_coeff = dx_coeff
            else:
                coeffs_x = cam_params.get("coefficients_x", [])
                self.dx_coeff = np.array(coeffs_x) if coeffs_x else np.zeros(10)

            if dy_coeff is not None:
                self.dy_coeff = dy_coeff
            else:
                coeffs_y = cam_params.get("coefficients_y", [])
                self.dy_coeff = np.array(coeffs_y) if coeffs_y else np.zeros(10)

            # Origin
            origin = cam_params.get("origin", {})
            if x_origin is not None:
                self.x_origin = x_origin
            else:
                self.x_origin = origin.get("x", origin.get("s_o", 0.0))

            if y_origin is not None:
                self.y_origin = y_origin
            else:
                self.y_origin = origin.get("y", origin.get("t_o", 0.0))

            # Normalisation
            norm = cam_params.get("normalisation", {})
            if nx is not None:
                self.nx = nx
            else:
                self.nx = norm.get("nx", 1.0)

            if ny is not None:
                self.ny = ny
            else:
                self.ny = norm.get("ny", 1.0)
        else:
            # No config params - use explicit or defaults
            self.mm_per_pixel = mm_per_pixel if mm_per_pixel is not None else 1.0
            self.dx_coeff = dx_coeff if dx_coeff is not None else np.zeros(10)
            self.dy_coeff = dy_coeff if dy_coeff is not None else np.zeros(10)
            self.x_origin = x_origin if x_origin is not None else 0.0
            self.y_origin = y_origin if y_origin is not None else 0.0
            self.nx = nx if nx is not None else 1.0
            self.ny = ny if ny is not None else 1.0

        logger.info(f"Initialized PolynomialCalibrator for Camera {camera_num}")
        logger.info(f"Time step: {self.dt} seconds")
        logger.info(f"MM per pixel: {self.mm_per_pixel}")

    def calibrate_coordinates(self, x_px: np.ndarray, y_px: np.ndarray) -> tuple:
        """
        Convert pixel coordinates to physical coordinates (mm) using DAVIS polynomial.

        Parameters
        ----------
        x_px : ndarray
            X coordinates in pixels
        y_px : ndarray
            Y coordinates in pixels

        Returns
        -------
        tuple
            (x_mm, y_mm) calibrated coordinates in mm
        """
        # Ensure inputs are float arrays
        x_px = np.asarray(x_px, dtype=np.float64)
        y_px = np.asarray(y_px, dtype=np.float64)

        if self.nx <= 1.0 or self.ny <= 1.0:
            logger.warning(f"Normalization factors nx={self.nx}, ny={self.ny} are suspiciously small (<=1). Coordinates might explode.")

        # normalized DAVIS coords
        s = 2 * (x_px - self.x_origin) / self.nx
        t = 2 * (y_px - self.y_origin) / self.ny

        # Debug ranges to catch explosion early
        logger.debug(f"Normalized coords range - s: [{s.min():.2f}, {s.max():.2f}], t: [{t.min():.2f}, {t.max():.2f}]")

        # evaluate dx, dy
        dx = evaluate_polynomial_terms(s, t, self.dx_coeff)
        dy = evaluate_polynomial_terms(s, t, self.dy_coeff)

        # back-mapped world coordinates (in pixels)
        x_world_px = x_px - dx
        y_world_px = y_px - dy

        # convert px -> mm
        x_mm = x_world_px * self.mm_per_pixel
        y_mm = y_world_px * self.mm_per_pixel

        return x_mm, y_mm

    def calibrate_vectors(
        self,
        ux_px: np.ndarray,
        uy_px: np.ndarray,
        coords_x_px: np.ndarray,
        coords_y_px: np.ndarray
    ) -> tuple:
        """
        Convert pixel-based velocity vectors to m/s using DAVIS polynomial.

        Parameters
        ----------
        ux_px : ndarray
            X velocity in pixels/frame
        uy_px : ndarray
            Y velocity in pixels/frame
        coords_x_px : ndarray
            X coordinates in pixels
        coords_y_px : ndarray
            Y coordinates in pixels

        Returns
        -------
        tuple
            (u_ms, v_ms) calibrated velocities in m/s
        """
        # Ensure inputs are float arrays
        ux_px = np.asarray(ux_px, dtype=np.float64)
        uy_px = np.asarray(uy_px, dtype=np.float64)
        coords_x_px = np.asarray(coords_x_px, dtype=np.float64)
        coords_y_px = np.asarray(coords_y_px, dtype=np.float64)

        # Check for shape mismatch and transpose coordinates if needed
        if ux_px.shape != coords_x_px.shape:
            if ux_px.shape == coords_x_px.T.shape:
                logger.warning(f"Shape mismatch detected. Transposing coordinates from {coords_x_px.shape} to {ux_px.shape}")
                coords_x_px = coords_x_px.T
                coords_y_px = coords_y_px.T
            else:
                raise ValueError(f"Shape mismatch: ux {ux_px.shape} vs coords {coords_x_px.shape}")

        x0_pix = coords_x_px
        y0_pix = coords_y_px

        x1_pix = x0_pix + ux_px
        y1_pix = y0_pix + uy_px

        # normalized DAVIS coords
        s0 = 2 * (x0_pix - self.x_origin) / self.nx
        t0 = 2 * (y0_pix - self.y_origin) / self.ny

        s1 = 2 * (x1_pix - self.x_origin) / self.nx
        t1 = 2 * (y1_pix - self.y_origin) / self.ny

        # evaluate dx, dy at center and displaced points
        dx0 = evaluate_polynomial_terms(s0, t0, self.dx_coeff)
        dy0 = evaluate_polynomial_terms(s0, t0, self.dy_coeff)

        dx1 = evaluate_polynomial_terms(s1, t1, self.dx_coeff)
        dy1 = evaluate_polynomial_terms(s1, t1, self.dy_coeff)

        # back-mapped world coordinates (in pixels)
        x0_world_px = x0_pix - dx0
        y0_world_px = y0_pix - dy0

        x1_world_px = x1_pix - dx1
        y1_world_px = y1_pix - dy1

        # world displacement (px)
        u_world_px = x1_world_px - x0_world_px
        v_world_px = y1_world_px - y0_world_px

        # convert px -> mm
        u_world_mm = u_world_px * self.mm_per_pixel
        v_world_mm = v_world_px * self.mm_per_pixel

        # convert mm -> m/s
        u_ms = (u_world_mm * 1e-3) / self.dt
        v_ms = (v_world_mm * 1e-3) / self.dt

        return u_ms, v_ms

    def process_vectors(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process all vector files in the directory using polynomial calibration.

        Reads num_frame_pairs from config to determine path structure and file count.
        Loads coordinates from uncalib_dir, calibrates them, and then processes vectors.

        Parameters
        ----------
        progress_callback : callable, optional
            Callback for progress updates

        Returns
        -------
        dict
            Result with processed_frames, total_frames, successful_frames
        """
        logger.info("Processing vector files with polynomial calibration...")

        # Get num_frame_pairs from config
        cfg = self._config if self._config is not None else get_config()
        num_frame_pairs = cfg.num_frame_pairs

        logger.info(f"Using num_frame_pairs={num_frame_pairs} from config")

        # Get data paths
        paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=num_frame_pairs,
            cam=self.camera_num,
            type_name=self.type_name,
            use_uncalibrated=True
        )

        uncalib_dir = paths["data_dir"]

        # Get output directory
        calib_paths = get_data_paths(
            self.base_dir,
            num_frame_pairs=num_frame_pairs,
            cam=self.camera_num,
            type_name=self.type_name,
            calibration=False  # We want the calibrated output dir
        )
        calib_dir = calib_paths["data_dir"]
        calib_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load and Calibrate Coordinates
        logger.info("Loading coordinates...")
        try:
            x_coords_list, y_coords_list = load_coords_from_directory(uncalib_dir, runs=None)
        except Exception as e:
            logger.error(f"Failed to load coordinates: {e}")
            raise

        if not x_coords_list:
            raise ValueError("No coordinate data found")

        # Prepare structure for calibrated coordinates
        num_runs = len(x_coords_list)
        coord_dtype = np.dtype([("x", "O"), ("y", "O")])
        coordinates = np.empty(num_runs, dtype=coord_dtype)

        # Keep track of calibrated coordinates for vector processing
        calibrated_coords_cache = []

        for i, (x_px, y_px) in enumerate(zip(x_coords_list, y_coords_list)):
            if x_px is not None and y_px is not None and x_px.size > 0:
                x_mm, y_mm = self.calibrate_coordinates(x_px, y_px)
                coordinates[i] = (x_mm, y_mm)
                calibrated_coords_cache.append((x_px, y_px))  # Store original PX for velocity calc
            else:
                coordinates[i] = (np.array([]), np.array([]))
                calibrated_coords_cache.append((None, None))

        # Save calibrated coordinates
        coords_path = calib_dir / "coordinates.mat"
        savemat(str(coords_path), {"coordinates": coordinates})
        logger.info(f"Saved calibrated coordinates to {coords_path}")

        # 2. Process Vector Files
        processed_vectors = []

        for i in range(1, num_frame_pairs + 1):
            vector_file = uncalib_dir / (self.vector_pattern % i)

            # Skip files that don't exist (consistent with stereo calibration pattern)
            if not vector_file.exists():
                if i <= 5:
                    logger.warning(f"Vector file not found: {vector_file}")
                continue

            # Load uncalibrated vectors with all runs
            vector_data_all = read_mat_contents(str(vector_file), return_all_runs=True)

            # vector_data_all shape is (R, 3, H, W)
            file_num_runs = vector_data_all.shape[0]

            logger.debug(f"File {vector_file.name}: Found {file_num_runs} runs. Cache size: {len(calibrated_coords_cache)}")

            # Create object array for MATLAB struct
            piv_dtype = np.dtype(
                [("ux", "O"), ("uy", "O"), ("b_mask", "O")]
            )
            piv_result = np.empty(file_num_runs, dtype=piv_dtype)

            has_valid_data = False

            for r in range(file_num_runs):
                # Ensure we have coordinates for this run
                if r >= len(calibrated_coords_cache):
                    logger.warning(f"Run {r} exceeds coordinate cache size {len(calibrated_coords_cache)}")
                    piv_result[r] = (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                    )
                    continue

                coords_x_px, coords_y_px = calibrated_coords_cache[r]
                if coords_x_px is None:
                    logger.warning(f"Run {r} has no coordinates")
                    piv_result[r] = (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                    )
                    continue

                # Extract data for this run
                # Handle both object array (list of arrays) and dense array (R, 3, H, W)
                if vector_data_all.dtype == object:
                    run_data = vector_data_all[r]
                    if run_data.size == 0 or run_data.shape[1] == 0:  # Handle (3, 0) empty placeholder
                        logger.warning(f"Run {r} has empty vector data (object array)")
                        piv_result[r] = (np.array([]), np.array([]), np.array([]))
                        continue
                    ux_px = run_data[0]
                    uy_px = run_data[1]
                    b_mask = run_data[2]
                else:
                    if vector_data_all[r].size == 0:
                        logger.warning(f"Run {r} has empty vector data")
                        piv_result[r] = (np.array([]), np.array([]), np.array([]))
                        continue
                    ux_px = vector_data_all[r, 0, :, :]
                    uy_px = vector_data_all[r, 1, :, :]
                    b_mask = vector_data_all[r, 2, :, :]

                if ux_px.size == 0:
                    logger.warning(f"Run {r} has empty ux_px")
                    piv_result[r] = (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                    )
                    continue

                has_valid_data = True

                # Calibrate vectors
                u_ms, v_ms = self.calibrate_vectors(
                    ux_px, uy_px, coords_x_px, coords_y_px
                )

                # Store in struct array
                piv_result[r] = (u_ms, v_ms, b_mask)

            # Save result
            if has_valid_data:
                output_file = calib_dir / (self.vector_pattern % i)
                savemat(str(output_file), {"piv_result": piv_result})
                processed_vectors.append(i)

            # Progress callback
            if progress_callback:
                progress = (i / num_frame_pairs) * 100
                progress_callback(
                    {
                        "processed_frames": i,
                        "total_frames": num_frame_pairs,
                        "progress": progress,
                        "successful_frames": len(processed_vectors),
                    }
                )

        logger.info(
            f"Successfully processed {len(processed_vectors)} vector files into {calib_dir}"
        )

        return {
            "success": True,
            "processed_frames": len(processed_vectors),
            "total_frames": num_frame_pairs,
            "successful_frames": len(processed_vectors),
        }

    @classmethod
    def process_all_cameras(
        cls,
        base_dir: Path,
        cameras: List[int],
        xml_data: Dict[str, Any],
        dt: float,
        vector_pattern: str = "%05d.mat",
        type_name: str = "instantaneous",
        config: Optional["PIVConfig"] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process all cameras sequentially with progress tracking.

        Args:
            base_dir: Base output directory
            cameras: List of camera numbers to process
            xml_data: Parsed XML data from read_calibration_xml
            dt: Time between frames
            vector_pattern: Vector file naming pattern
            type_name: Data type (instantaneous, ensemble)
            config: Optional config object
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with overall results
        """
        cfg = config if config is not None else get_config()
        num_frame_pairs = cfg.num_frame_pairs

        total_cameras = len(cameras)
        overall_result = {
            "total_cameras": total_cameras,
            "processed_cameras": 0,
            "total_files": 0,
            "processed_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "camera_results": {},
        }

        for idx, cam_num in enumerate(cameras):
            logger.info(f"Processing camera {cam_num} ({idx + 1}/{total_cameras})")

            # Find camera data in XML
            cam_key = None
            for key in xml_data.get("cameras", {}):
                if str(cam_num) in key:
                    cam_key = key
                    break

            if not cam_key:
                logger.warning(f"No XML data found for camera {cam_num}")
                overall_result["camera_results"][cam_num] = {
                    "status": "failed",
                    "error": "No XML data found",
                }
                continue

            cam_params = xml_data["cameras"][cam_key]

            # Extract parameters from XML
            dx_coeff = convert_davis_coeffs_to_array(cam_params.get("coefficients_a", {}))
            dy_coeff = convert_davis_coeffs_to_array(cam_params.get("coefficients_b", {}))

            origin_dict = cam_params.get("origin", {})
            x_origin = origin_dict.get("s_o", 0.0)
            y_origin = origin_dict.get("t_o", 0.0)

            norm_dict = cam_params.get("normalisation", {})
            nx = norm_dict.get("nx", 1.0)
            ny = norm_dict.get("ny", 1.0)

            mm_per_pixel = cam_params.get("mm_per_pixel", 1.0)

            # Create calibrator with explicit parameters (from XML)
            calibrator = cls(
                base_dir=base_dir,
                camera_num=cam_num,
                dt=dt,
                mm_per_pixel=mm_per_pixel,
                dx_coeff=dx_coeff,
                dy_coeff=dy_coeff,
                x_origin=x_origin,
                y_origin=y_origin,
                nx=nx,
                ny=ny,
                vector_pattern=vector_pattern,
                type_name=type_name,
                config=config,
            )

            # Progress wrapper
            def camera_progress(data: Dict[str, Any]):
                if progress_callback:
                    progress_callback({
                        "current_camera": cam_num,
                        "processed_cameras": idx,
                        "total_cameras": total_cameras,
                        "camera_processed_frames": data.get("processed_frames", 0),
                        "camera_total_frames": data.get("total_frames", 0),
                        "overall_progress": int(((idx + data.get("progress", 0) / 100) / total_cameras) * 100),
                    })

            try:
                result = calibrator.process_vectors(progress_callback=camera_progress)
                overall_result["camera_results"][cam_num] = {
                    "status": "completed",
                    "processed_frames": result.get("processed_frames", 0),
                    "successful_frames": result.get("successful_frames", 0),
                }
                overall_result["successful_files"] += result.get("successful_frames", 0)
            except Exception as e:
                logger.error(f"Camera {cam_num} failed: {e}")
                overall_result["camera_results"][cam_num] = {
                    "status": "failed",
                    "error": str(e),
                }
                overall_result["failed_files"] += 1

            overall_result["processed_cameras"] = idx + 1

        return overall_result


# ============================================================================
# CLI MAIN
# ============================================================================


def main():
    """Run polynomial calibration for all configured cameras."""
    logger.info("=" * 60)
    logger.info("Polynomial Calibration - Production Script")
    logger.info("=" * 60)

    if USE_CONFIG_DIRECTLY:
        # Load settings directly from existing config.yaml
        logger.info("Loading settings directly from config.yaml (USE_CONFIG_DIRECTLY=True)")
        cfg = get_config()

        # Extract settings from config
        base_dir = cfg.data["paths"]["base_paths"][0]
        camera_nums = cfg.data["paths"].get("camera_numbers", [1, 2])
        calibration_subfolder = cfg.data["calibration"].get("subfolder", "")
        dt = cfg.data["calibration"].get("polynomial", {}).get("dt", 1.0)
        type_name = cfg.data.get("processing", {}).get("type_name", "instantaneous")
        # Get xml_path and use_xml from config
        xml_path = cfg.data["calibration"].get("polynomial", {}).get("xml_path", "")
        use_xml = cfg.data["calibration"].get("polynomial", {}).get("use_xml", True)
    else:
        # Apply hardcoded settings to config
        cfg = apply_cli_settings_to_config()

        # Use hardcoded settings
        base_dir = BASE_DIR
        camera_nums = CAMERA_NUMS
        calibration_subfolder = CALIBRATION_SUBFOLDER
        dt = DT
        type_name = TYPE_NAME
        xml_path = XML_PATH
        use_xml = USE_XML

    logger.info(f"use_xml: {use_xml}")
    if xml_path:
        logger.info(f"xml_path: {xml_path}")

    # Get vector format from config
    vec_fmt = cfg.vector_format
    if isinstance(vec_fmt, list):
        vec_fmt = vec_fmt[0]

    # Progress callback for CLI output
    def progress_callback(progress_data):
        current_cam = progress_data.get("current_camera", "?")
        overall_progress = progress_data.get("overall_progress", 0)
        camera_frames = progress_data.get("camera_processed_frames", 0)
        camera_total = progress_data.get("camera_total_frames", 0)
        logger.info(
            f"  Camera {current_cam}: {overall_progress}% "
            f"({camera_frames}/{camera_total} frames)"
        )

    if use_xml:
        # Read XML and use XML data for calibration
        logger.info("Reading Calibration.xml...")
        try:
            xml_result = read_calibration_xml(
                source_path_idx=0,
                xml_path=xml_path if xml_path else None,
                config=cfg
            )
        except FileNotFoundError as e:
            logger.error(f"Failed to read XML: {e}")
            if not USE_CONFIG_DIRECTLY and not xml_path:
                logger.error(f"Expected location: {SOURCE_DIR}/{CALIBRATION_SUBFOLDER}/Calibration.xml")
            return

        if xml_result.get("status") != "success":
            logger.error("Failed to parse Calibration.xml")
            return

        logger.info(f"Found {len(xml_result['cameras'])} camera(s) in XML")
        for cam_key in xml_result["cameras"]:
            logger.info(f"  - {cam_key}")

        logger.info(f"Processing {len(camera_nums)} camera(s): {camera_nums}")
        logger.info(f"Type: {type_name}")

        # Run calibration for all cameras using XML data
        result = PolynomialVectorCalibrator.process_all_cameras(
            base_dir=Path(base_dir),
            cameras=camera_nums,
            xml_data=xml_result,
            dt=dt,
            vector_pattern=vec_fmt,
            type_name=type_name,
            config=cfg,
            progress_callback=progress_callback,
        )
    else:
        # Use config values directly (no XML)
        logger.info("Using coefficients from config.yaml (use_xml=False)")
        logger.info(f"Processing {len(camera_nums)} camera(s): {camera_nums}")
        logger.info(f"Type: {type_name}")

        # Process each camera using config values
        overall_result = {
            "total_cameras": len(camera_nums),
            "processed_cameras": 0,
            "successful_files": 0,
            "failed_files": 0,
            "camera_results": {},
        }

        for idx, cam_num in enumerate(camera_nums):
            logger.info(f"Processing camera {cam_num} ({idx + 1}/{len(camera_nums)})")

            # Create calibrator - will read params from config
            calibrator = PolynomialVectorCalibrator(
                base_dir=Path(base_dir),
                camera_num=cam_num,
                dt=dt,
                vector_pattern=vec_fmt,
                type_name=type_name,
                config=cfg,
            )

            def camera_progress(data: Dict[str, Any]):
                progress_callback({
                    "current_camera": cam_num,
                    "processed_cameras": idx,
                    "total_cameras": len(camera_nums),
                    "camera_processed_frames": data.get("processed_frames", 0),
                    "camera_total_frames": data.get("total_frames", 0),
                    "overall_progress": int(((idx + data.get("progress", 0) / 100) / len(camera_nums)) * 100),
                })

            try:
                cam_result = calibrator.process_vectors(progress_callback=camera_progress)
                overall_result["camera_results"][cam_num] = {
                    "status": "completed",
                    "processed_frames": cam_result.get("processed_frames", 0),
                    "successful_frames": cam_result.get("successful_frames", 0),
                }
                overall_result["successful_files"] += cam_result.get("successful_frames", 0)
            except Exception as e:
                logger.error(f"Camera {cam_num} failed: {e}")
                overall_result["camera_results"][cam_num] = {
                    "status": "failed",
                    "error": str(e),
                }
                overall_result["failed_files"] += 1

            overall_result["processed_cameras"] = idx + 1

        result = overall_result

    # Summary
    logger.info("=" * 60)
    logger.info("Calibration Complete!")
    logger.info(f"  Cameras processed: {result['processed_cameras']}")
    logger.info(f"  Successful files: {result['successful_files']}")
    logger.info(f"  Failed files: {result['failed_files']}")
    logger.info("=" * 60)

    # Per-camera breakdown
    for cam_num, cam_result in result.get("camera_results", {}).items():
        status = cam_result.get("status", "unknown")
        if status == "completed":
            frames = cam_result.get("successful_frames", 0)
            logger.info(f"  Camera {cam_num}: {frames} frames [OK]")
        else:
            error = cam_result.get("error", "Unknown error")
            logger.info(f"  Camera {cam_num}: FAILED - {error}")

    return result


if __name__ == "__main__":
    main()
