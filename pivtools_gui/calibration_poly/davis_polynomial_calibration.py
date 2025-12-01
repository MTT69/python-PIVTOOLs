import xml.etree.ElementTree as ET
from pathlib import Path
from loguru import logger
from pivtools_core.config import get_config
from pivtools_core.paths import get_data_paths
import numpy as np
from scipy.io import savemat
from pivtools_core.vector_loading import read_mat_contents, load_coords_from_directory


def read_calibration_xml(source_path_idx):
    logger.debug(f"Starting read_calibration_xml with index: {source_path_idx}")
    cfg = get_config()
    # Ensure source_paths exists and has the index
    if not hasattr(cfg, "source_paths") or source_path_idx >= len(cfg.source_paths):
        raise ValueError("Invalid source_path_idx")

    source_root = Path(cfg.source_paths[source_path_idx])
    xml_path = source_root / "Properties" / "Calibration" / "Calibration.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"Calibration.xml not found at {xml_path}")

    logger.info(f"Reading Calibration.xml from: {xml_path}")
    tree = ET.parse(xml_path)
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

    return {"status": "success", "file": str(xml_path), "cameras": cameras_data}


def evaluate_polynomial_terms(s, t, coeffs):
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


class PolynomialVectorCalibrator:
    def __init__(
        self,
        base_dir,
        camera_num,
        dt,
        mm_per_pixel,
        dx_coeff,
        dy_coeff,
        x_origin,
        y_origin,
        nx,
        ny,
        vector_pattern="%05d.mat",
        type_name="instantaneous",
    ):
        self.base_dir = Path(base_dir)
        self.camera_num = camera_num
        self.dt = 70E-6
        self.mm_per_pixel = mm_per_pixel
        self.dx_coeff = dx_coeff
        self.dy_coeff = dy_coeff
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.nx = nx
        self.ny = ny
        self.vector_pattern = vector_pattern
        self.type_name = type_name

        logger.info(f"Initialized PolynomialCalibrator for Camera {camera_num}")
        logger.info(f"Time step: {dt} seconds")
        logger.info(f"MM per pixel: {mm_per_pixel}")

    def calibrate_coordinates(self, x_px, y_px):
        """
        Convert pixel coordinates to physical coordinates (mm) using DAVIS polynomial.
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
        if logger:
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

    def calibrate_vectors(self, ux_px, uy_px, coords_x_px, coords_y_px):
        """
        Convert pixel-based velocity vectors to m/s using DAVIS polynomial.
        """
        # Ensure inputs are float arrays
        ux_px = np.asarray(ux_px, dtype=np.float64)
        uy_px = np.asarray(uy_px, dtype=np.float64)
        coords_x_px = np.asarray(coords_x_px, dtype=np.float64)
        coords_y_px = np.asarray(coords_y_px, dtype=np.float64)

        # Check for shape mismatch and transpose coordinates if needed
        if ux_px.shape != coords_x_px.shape:
            if ux_px.shape == coords_x_px.T.shape:
                if logger:
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

    def process_run(self, image_count, progress_cb=None):
        """
        Process all vector files in the directory using polynomial calibration.
        Loads coordinates from uncalib_dir, calibrates them, and then processes vectors.
        """
        logger.info("Processing vector files with polynomial calibration...")
        
        # Get data paths
        paths = get_data_paths(
            self.base_dir,
            num_images=image_count,
            cam=self.camera_num,
            type_name=self.type_name,
            use_uncalibrated=True
        )
        
        uncalib_dir = paths["data_dir"]
        
        # Get output directory
        calib_paths = get_data_paths(
            self.base_dir,
            num_images=image_count,
            cam=self.camera_num,
            type_name=self.type_name,
            calibration=False # We want the calibrated output dir
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

        for i in range(1, image_count + 1):
            vector_file = uncalib_dir / (self.vector_pattern % i)

            if not vector_file.exists():
                if i <= 5:
                    logger.warning(f"Vector file not found: {vector_file}")
                continue

            # Load uncalibrated vectors with all runs
            vector_data_all = read_mat_contents(str(vector_file), return_all_runs=True)

            # vector_data_all shape is (R, 3, H, W)
            file_num_runs = vector_data_all.shape[0]
            
            if logger:
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
                    if logger:
                        logger.warning(f"Run {r} exceeds coordinate cache size {len(calibrated_coords_cache)}")
                    piv_result[r] = (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                    )
                    continue

                coords_x_px, coords_y_px = calibrated_coords_cache[r]
                if coords_x_px is None:
                    if logger:
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
                    if run_data.size == 0 or run_data.shape[1] == 0: # Handle (3, 0) empty placeholder
                         if logger:
                            logger.warning(f"Run {r} has empty vector data (object array)")
                         piv_result[r] = (np.array([]), np.array([]), np.array([]))
                         continue
                    ux_px = run_data[0]
                    uy_px = run_data[1]
                    b_mask = run_data[2]
                else:
                    if vector_data_all[r].size == 0:
                        if logger:
                            logger.warning(f"Run {r} has empty vector data")
                        piv_result[r] = (np.array([]), np.array([]), np.array([]))
                        continue
                    ux_px = vector_data_all[r, 0, :, :]
                    uy_px = vector_data_all[r, 1, :, :]
                    b_mask = vector_data_all[r, 2, :, :]

                if ux_px.size == 0:
                    if logger:
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
            else:
                # If file exists but no valid runs, maybe warn?
                pass

            # Progress callback
            if progress_cb:
                progress = (i / image_count) * 100
                progress_cb(
                    {
                        "processed_frames": i,
                        "total_frames": image_count,
                        "progress": progress,
                        "successful_frames": len(processed_vectors),
                    }
                )

        logger.info(
            f"Successfully processed {len(processed_vectors)} vector files into {calib_dir}"
        )


def convert_davis_coeffs_to_array(coeff_dict):
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


# if __name__ == "__main__":
#     import sys
    
#     # Setup logging to console
#     logger.remove()
#     logger.add(sys.stderr, level="DEBUG")
    
#     print("Starting Polynomial Calibration Test...")
    
#     try:
#         cfg = get_config()
        
#         # Parameters from config or defaults
#         source_path_idx = 0
#         camera_num = 1
        
#         # Try to get from config if available
#         if hasattr(cfg, "calibration") and hasattr(cfg.calibration, "scale_factor"):
#              source_path_idx = cfg.calibration.scale_factor.source_path_idx
#              # dt might be there too
        
#         # Read calibration XML
#         print(f"Reading calibration XML for source index {source_path_idx}...")
#         calib_data = read_calibration_xml(source_path_idx)
        
#         if "cameras" not in calib_data:
#             print("Error: No camera data found in XML")
#             sys.exit(1)
            
#         # Find camera data
#         # keys might be "1", "2" or "Camera 1" etc.
#         # Let's try to find a key that contains our camera number
#         cam_key = None
#         for key in calib_data["cameras"]:
#             if str(camera_num) in key:
#                 cam_key = key
#                 break
        
#         if not cam_key:
#             print(f"Error: Camera {camera_num} not found in calibration data. Available: {list(calib_data['cameras'].keys())}")
#             # Fallback for testing if no XML match but we want to test the class
#             # sys.exit(1)
#             print("Proceeding with dummy coefficients for testing...")
#             cam_params = {}
#         else:
#             cam_params = calib_data["cameras"][cam_key]
#             print(f"Found parameters for {cam_key}")
        
#         # Extract parameters
#         dx_coeff = convert_davis_coeffs_to_array(cam_params.get("coefficients_a", {}))
#         dy_coeff = convert_davis_coeffs_to_array(cam_params.get("coefficients_b", {}))
        
#         # Origin
#         # XML attributes are s_o, t_o
#         origin_dict = cam_params.get("origin", {})
#         x_origin = origin_dict.get("s_o", origin_dict.get("x", origin_dict.get("X", 0.0)))
#         y_origin = origin_dict.get("t_o", origin_dict.get("y", origin_dict.get("Y", 0.0)))
        
#         # Normalisation
#         # XML attributes are nx, ny
#         norm_dict = cam_params.get("normalisation", {})
#         nx = norm_dict.get("nx", norm_dict.get("x", norm_dict.get("X", 1.0)))
#         ny = norm_dict.get("ny", norm_dict.get("y", norm_dict.get("Y", 1.0)))
        
#         # mm_per_pixel
#         mm_per_pixel = cam_params.get("mm_per_pixel", 1.0)
        
#         # DT from config or default
#         dt = 70E-6
        
#         print(f"Parameters:")
#         print(f"  DT: {dt}")
#         print(f"  MM/Px: {mm_per_pixel}")
#         print(f"  Origin: ({x_origin}, {y_origin})")
#         print(f"  Norm: ({nx}, {ny})")
        
#         base_dir = cfg.base_paths[source_path_idx]
#         print(f"Processing data in: {base_dir}")
        
#         # Handle vector format which might be a list in config
#         vec_fmt = cfg.vector_format
#         if isinstance(vec_fmt, list):
#             vec_fmt = vec_fmt[0]
            
#         print(base_dir)
#         calibrator = PolynomialVectorCalibrator(
#             base_dir=base_dir,
#             camera_num=camera_num,
#             dt=70E-6,
#             mm_per_pixel=mm_per_pixel,
#             dx_coeff=dx_coeff,
#             dy_coeff=dy_coeff,
#             x_origin=x_origin,
#             y_origin=y_origin,
#             nx=nx,
#             ny=ny,
#             vector_pattern=vec_fmt,
#             type_name="instantaneous" # Or from config
#         )
        
#         # Run for a few images
#         num_images = 30
#         print(f"Running for {num_images} images...")
        
#         calibrator.process_run(num_images, progress_cb=lambda x: print(f"Progress: {x['progress']:.1f}%"))
        
#         print("Done!")
        
#     except Exception as e:
#         print(f"Test failed: {e}")
#         import traceback
#         traceback.print_exc()
