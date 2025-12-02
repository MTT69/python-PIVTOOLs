from pathlib import Path
import logging
import os
import shutil

import yaml

_CONFIG = None  # singleton cache
_LOGGING_INITIALIZED = False  # Track if logging has been set up


class Config:
    def __init__(self, path=None):
        if path is None:
            path = self._get_config_path()
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)
            # Use the first base_path, first camera, and image_format for dtype detection
            # source_path = Path(self.source_paths[0])
            # camera_folder = f"Cam{self.camera_numbers[0]}"
            # # Use correct image format for dtype detection
            # if self.time_resolved:
            #     file_path = source_path / camera_folder / (self.image_format % 1)
            # else:
            #     file_path = source_path / camera_folder / (self.image_format[0] % 1)
            # img = tifffile.imread(file_path) # bye bye
            # self.image_dtype = img.dtype
        
        # Cache for auto-detected image shape
        self._detected_image_shape = None
        
        # Cache for auto-computed parameters
        self._auto_compute_cache = None
        
        # Setup logging only once globally
        self._setup_logging()

        # Store the config path for saving
        self._config_path = path if path is not None else self._get_config_path()

    def _get_config_path(self):
        """Get the path to the config file in the current working directory."""
        # Always use current working directory (like CLI)
        cwd_config_path = Path.cwd() / 'config.yaml'
        
        # If config doesn't exist, copy from package default or create programmatically
        if not cwd_config_path.exists():
            package_default = Path(__file__).parent / 'config.yaml'
            if package_default.exists():
                shutil.copy2(package_default, cwd_config_path)
            else:
                # Fallback: create default config programmatically (like CLI)
                default_config = """
paths:
  base_paths:
  - /set_me
  source_paths:
  - /set_me
  camera_numbers:
  - 1
  camera_count: 1
  camera_subfolders: []
images:
  num_images: 1000
  image_format:
  - B%05d_A.tif
  - B%05d_B.tif
  vector_format:
  - '%05d.mat'
  time_resolved: false
  dtype: float32
  zero_based_indexing: false
  pairing_mode: sequential
  pairing_skip: 0
  num_frame_pairs: 1000
batches:
  size: 25
logging:
  file: pypiv.log
  level: INFO
  console: true
processing:
  instantaneous: true
  ensemble: false
  backend: cpu
  debug: false
  auto_compute_params: false
  omp_threads: 2
  dask_workers_per_node: 4
  dask_threads_per_worker: 1
  dask_memory_limit: 3GB
  filter_worker_count: 1
  always_batch: true
outlier_detection:
  enabled: true
  methods:
  - threshold: 0.3
    type: peak_mag
  - epsilon: 0.2
    threshold: 2
    type: median_2d
infilling:
  mid_pass:
    method: biharmonic
    parameters:
      ksize: 3
  final_pass:
    enabled: true
    method: biharmonic
    parameters:
      ksize: 3
ensemble_outlier_detection:
  enabled: true
  methods:
  - epsilon: 0.2
    threshold: 2
    type: median_2d
ensemble_infilling:
  mid_pass:
    method: biharmonic
    parameters:
      ksize: 3
  final_pass:
    enabled: true
    method: biharmonic
    parameters:
      ksize: 3
plots:
  save_extension: .png
  save_pickle: true
  fontsize: 14
  title_fontsize: 16
videos:
- endpoint: ''
  type: instantaneous
  use_merged: false
  variable: ux
  video_length: 100
statistics_extraction: null
instantaneous_piv:
  window_size:
  - - 128
    - 128
  - - 64
    - 64
  - - 32
    - 32
  overlap:
  - 50
  - 50
  - 50
  runs:
  - 3
  time_resolved: false
  window_type: gaussian
  num_peaks: 1
  peak_finder: gauss3
  secondary_peak: false
ensemble_piv:
  window_size:
  - - 128
    - 128
  - - 64
    - 64
  - - 16
    - 16
  overlap:
  - 50
  - 50
  - 50
  type:
  - std
  - std
  - std
  runs:
  - 3
  store_planes: false
  save_diagnostics: false
  sum_window:
  - 16
  - 16
  resume_from_pass: 0
calibration_format:
  image_format: calib%05d.tif
calibration:
  active: polynomial
  scale_factor:
    dt: 0.56
    px_per_mm: 3.41
    source_path_idx: 0
  pinhole:
    source_path_idx: 0
    camera: 1
    image_index: 0
    file_pattern: calib%05d.tif
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    enhance_dots: true
    asymmetric: false
    dot_distance_mm: 28.9
    grid_tolerance: 0.5
    ransac_threshold: 3
    dt: 0.0275
  stereo:
    source_path_idx: 0
    camera_pair:
    - 1
    - 2
    file_pattern: planar_calibration_plate_*.tif
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    enhance_dots: true
    asymmetric: false
    dt: 2
filters: []
masking:
  enabled: true
  mask_file_pattern: mask_Cam%d.mat
  mask_threshold: 0.01
  mode: rectangular
  rectangular:
    top: 0
    bottom: 0
    left: 0
    right: 0

"""
                with open(cwd_config_path, 'w') as f:
                    f.write(default_config.strip())
        
        return cwd_config_path

    @property
    def config_path(self):
        """Get the path to the config file."""
        return self._config_path

    @property
    def config_dict(self):
        """Access to raw config dictionary for advanced usage."""
        return self.data

    @property
    def time_resolved(self):
        return self.data["images"].get("time_resolved", False)

    @property
    def image_format(self):
        """
        Return image format as a tuple.

        Always returns a tuple for consistency:
        - Single format: ("format",)
        - A/B pair: ("format_A", "format_B")
        """
        raw = self.data["images"].get("image_format")
        if raw is None:
            # Default
            if self.time_resolved:
                return ("B%05d.tiff",)
            else:
                return ("B%05d_A.tiff", "B%05d_B.tiff")

        if isinstance(raw, str):
            return (raw,)
        elif isinstance(raw, (list, tuple)):
            return tuple(raw)
        else:
            raise ValueError(f"Invalid image_format type: {type(raw)}")

    @property
    def image_type(self) -> str:
        """
        Return image type: 'standard', 'cine', 'lavision_set', 'lavision_im7'.

        If explicitly set in config, returns that value.
        Otherwise, auto-detects from image_format pattern.

        Returns
        -------
        str
            One of: 'standard', 'cine', 'lavision_set', 'lavision_im7'
        """
        explicit_type = self.data.get("images", {}).get("image_type")
        if explicit_type:
            return explicit_type
        return self._detect_image_type()

    def _detect_image_type(self) -> str:
        """Auto-detect image type from format string."""
        fmt = self.image_format[0].lower()
        if '.cine' in fmt:
            return "cine"
        elif '.set' in fmt:
            return "lavision_set"
        elif '.im7' in fmt:
            return "lavision_im7"
        elif '.ims' in fmt:
            return "lavision_im7"  # .ims treated same as .im7
        else:
            return "standard"

    @property
    def is_container_format(self) -> bool:
        """Return True if format stores multiple frames in single container.

        Container formats:
        - cine: Single-camera video container (one file per camera)
        - lavision_set: Multi-camera container (all cameras in one file)
        - lavision_im7: Multi-camera container (all cameras per file)
        """
        return self.image_type in ("cine", "lavision_set", "lavision_im7")

    @property
    def is_single_camera_container(self) -> bool:
        """Return True if container has one camera per file (like .cine).

        .cine files contain frames from a single camera. Multi-camera setups
        have separate .cine files per camera (e.g., Camera1.cine, Camera2.cine).
        """
        return self.image_type == "cine"

    @property
    def is_multi_camera_container(self) -> bool:
        """Return True if container has all cameras in one file (like .set, .im7).

        .set and .im7 files store data from all cameras in a single file,
        with camera_no parameter used to extract specific camera data.
        """
        return self.image_type in ("lavision_set", "lavision_im7")

    @property
    def base_paths(self):
        return [Path(p) for p in self.data["paths"]["base_paths"]]

    @property
    def source_paths(self):
        return [Path(s) for s in self.data["paths"]["source_paths"]]

    @property
    def camera_count(self):
        """Return the total number of cameras."""
        return self.data["paths"].get("camera_count", 1)

    @property
    def camera_numbers(self):
        """Return list of camera numbers to process."""
        numbers = self.data["paths"]["camera_numbers"]
        max_allowed = self.camera_count
        if any(n > max_allowed or n < 1 for n in numbers):
            raise ValueError(f"Camera numbers {numbers} must be between 1 and {max_allowed}")
        return numbers

    @property
    def camera_folders(self):
        return [self.get_camera_folder(n) for n in self.camera_numbers]

    @property
    def num_images(self):
        """Return the number of image files (not pairs)."""
        return self.data["images"]["num_images"]

    @property
    def num_frame_pairs(self):
        """
        Calculate the number of frame pairs based on image type and pairing mode.

        The calculation depends on the image type and time_resolved setting:

        Container formats:
        - lavision_set: Each .set entry is one pair → num_images pairs
        - lavision_im7: Each .im7 file is one pair → num_images pairs
        - cine + time_resolved: Sequential overlapping → num_images - 1 pairs
        - cine + skip: Non-overlapping → num_images // 2 pairs

        Standard formats:
        - A/B format (len=2): num_images pairs (1A+1B, 2A+2B, ...)
        - time_resolved: num_images - 1 pairs (1+2, 2+3, 3+4, ...)
        - skip: num_images // 2 pairs (1+2, 3+4, 5+6, ...)

        Returns
        -------
        int
            Number of frame pairs that can be formed from the image files
        """
        num_images = self.num_images
        image_type = self.image_type

        # LaVision .set: depends on time_resolved
        # - Non-time-resolved: each entry has A+B pair internally
        # - Time-resolved: each entry has ONE frame per camera, pair across entries
        if image_type == "lavision_set":
            if self.time_resolved:
                # Sequential overlapping: 100 entries → 99 pairs
                return max(0, num_images - 1)
            else:
                # Each entry is a complete pair
                return num_images

        # LaVision .im7: depends on time_resolved
        # - Non-time-resolved: each file has A+B pair internally
        # - Time-resolved: each file has ONE frame, pair across files
        if image_type == "lavision_im7":
            if self.time_resolved:
                # Sequential overlapping: 100 files → 99 pairs
                return max(0, num_images - 1)
            else:
                # Each file is a complete pair
                return num_images

        # CINE: depends on time_resolved setting
        if image_type == "cine":
            if self.time_resolved:
                # Sequential overlapping: 100 frames → 99 pairs
                return max(0, num_images - 1)
            else:
                # Skip mode: 100 frames → 50 non-overlapping pairs
                return num_images // 2

        # Standard formats: A/B format
        if len(self.image_format) == 2:
            return num_images

        # Standard formats: time-resolved or skip
        if self.time_resolved:
            return max(0, num_images - 1)

        # Skip frames (non-overlapping)
        return num_images // 2

    @property
    def pairing_mode(self):
        """
        Return frame pairing mode.

        Values:
        - 'sequential': Standard (1+2, 2+3, 3+4, ...) for time-resolved or (1A+1B, 2A+2B) for non-time-resolved
        - 'skip': Skip frames (1+2, 3+4, 5+6, ...) for time-resolved only
        """
        return self.data.get("images", {}).get("pairing_mode", "sequential")

    def get_frame_pair_indices(self, pair_number: int) -> tuple:
        """
        Get the file/frame indices for a given pair number.

        For container formats, indexing complexity is hidden from the user.
        The returned indices are ready to use with the appropriate reader.

        Args:
            pair_number: 1-based pair number (pair 1, pair 2, etc.)

        Returns:
            tuple: (frame_a_idx, frame_b_idx) for the reader to use

        Examples by image_type:
            lavision_set (non-time-resolved):
                pair 1 → (1, 1) - reader extracts A+B from entry 1
            lavision_set (time-resolved):
                pair 1 → (1, 2), pair 2 → (2, 3) - pair frames from consecutive entries
            lavision_im7 (non-time-resolved):
                pair 1 → (1, 1) - reader extracts A+B from file 1
            lavision_im7 (time-resolved):
                pair 1 → (1, 2), pair 2 → (2, 3) - pair frames from consecutive files
            cine + time_resolved:
                pair 1 → (1, 2), pair 2 → (2, 3) - overlapping
            cine + skip:
                pair 1 → (1, 2), pair 2 → (3, 4) - non-overlapping
            standard A/B:
                pair 1 → (1, 1), pair 2 → (2, 2) - same index for A and B files
            standard time_resolved:
                pair 1 → (1, 2), pair 2 → (2, 3) - overlapping
            standard skip:
                pair 1 → (1, 2), pair 2 → (3, 4) - non-overlapping
        """
        image_type = self.image_type

        # LaVision .set: depends on time_resolved
        # - Non-time-resolved: A+B pair in same entry
        # - Time-resolved: pair frames from consecutive entries
        if image_type == "lavision_set":
            if self.time_resolved:
                # Time-resolved: pair across entries (entry N + entry N+1)
                return (pair_number, pair_number + 1)
            else:
                # Non-time-resolved: A+B in same entry
                return (pair_number, pair_number)

        # LaVision .im7: depends on time_resolved
        # - Non-time-resolved: each file has A+B pair → same file index for both
        # - Time-resolved: each file has ONE frame → pair across consecutive files
        if image_type == "lavision_im7":
            if self.time_resolved:
                # Time-resolved: pair across files (file N + file N+1)
                # Apply zero-based indexing to both file indices
                if self.zero_based_indexing:
                    file_a = pair_number - 1  # Pair 1 → files 0,1
                    file_b = pair_number
                else:
                    file_a = pair_number      # Pair 1 → files 1,2
                    file_b = pair_number + 1
                return (file_a, file_b)
            else:
                # Non-time-resolved: each file is a complete A+B pair
                file_idx = (pair_number - 1) if self.zero_based_indexing else pair_number
                return (file_idx, file_idx)

        # CINE: frame pairing depends on time_resolved
        # Note: zero_based_indexing is NOT applied for cine - the reader
        # handles FirstImageNo translation internally
        if image_type == "cine":
            if self.time_resolved:
                # Sequential overlapping: pair n = frames (n, n+1)
                frame_a = pair_number
                frame_b = pair_number + 1
            else:
                # Skip mode: pair n = frames ((n-1)*2+1, (n-1)*2+2)
                frame_a = (pair_number - 1) * 2 + 1
                frame_b = frame_a + 1
            return (frame_a, frame_b)

        # Standard formats below

        # A/B format (separate A and B files) - always use same index for both
        if len(self.image_format) == 2:
            file_idx = (pair_number - 1) if self.zero_based_indexing else pair_number
            return (file_idx, file_idx)

        # Time-resolved = sequential overlapping pairs
        if self.time_resolved:
            # Sequential mode: pair 1=(0,1), pair 2=(1,2), pair 3=(2,3)
            frame_a_idx = pair_number - 1
            frame_b_idx = pair_number
        else:
            # Non-time-resolved skip mode = non-overlapping pairs
            # Pair 1=(0,1), pair 2=(2,3), pair 3=(4,5), etc.
            start_idx = (pair_number - 1) * 2
            frame_a_idx = start_idx
            frame_b_idx = start_idx + 1

        # Apply zero-based indexing adjustment (if files start at 0 instead of 1)
        if not self.zero_based_indexing:
            frame_a_idx += 1
            frame_b_idx += 1

        return (frame_a_idx, frame_b_idx)

    @property
    def image_shape(self):
        """
        Return image shape (H, W).
        
        If shape is specified in config, use that.
        Otherwise, auto-detect from first image and cache the result.
        """
        # First check if explicitly set in config
        if "shape" in self.data.get("images", {}):
            return tuple(self.data["images"]["shape"])
        
        # Otherwise, auto-detect and cache
        if self._detected_image_shape is None:
            self._detected_image_shape = self._detect_image_shape()
            logging.info("Auto-detected image shape: %s", self._detected_image_shape)
        
        return self._detected_image_shape
    
    def _detect_image_shape(self) -> tuple:
        """
        Detect image shape by reading the first image.

        Handles all image formats including .set, .im7, .cine, and standard formats.
        For container formats, passes the required camera and image parameters.

        Returns
        -------
        tuple
            (H, W) shape of images
        """
        from .image_handling.load_images import read_image

        source_path = self.source_paths[0]
        camera_num = self.camera_numbers[0]
        image_format = self.image_format
        img_type = self.image_type

        logging.info(f"_detect_image_shape: image_format = {image_format}, image_type = {img_type}")
        logging.info(f"Source path: {source_path}, Camera: {camera_num}")

        format_str = image_format[0]  # Always tuple now

        # Determine camera_path based on image type (same as load_images)
        if img_type in ("lavision_set", "lavision_im7", "cine"):
            camera_path = source_path  # Container formats: no camera subdir
        else:
            folder = self.get_camera_folder(camera_num)
            camera_path = source_path / folder if folder else source_path

        logging.info(f"Camera path: {camera_path}")

        # Determine start index
        start_idx = 0 if self.zero_based_indexing else 1

        # Construct file path based on image type
        if img_type == "lavision_set":
            file_path = camera_path / format_str
        elif img_type == "lavision_im7":
            file_path = camera_path / (format_str % start_idx)
        elif img_type == "cine":
            # CINE: format uses %d for camera number
            cine_filename = format_str % camera_num
            file_path = camera_path / cine_filename
        else:
            # Standard files - use first format for shape detection
            file_path = camera_path / (format_str % start_idx)

        logging.info(f"Trying to read file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {file_path}. "
                "Check your source_path and image_format in config.yaml"
            )

        try:
            # Read with appropriate parameters for each format
            if img_type == "lavision_set":
                # For .set files, must provide camera_no and im_no
                img = read_image(str(file_path), camera_no=camera_num, im_no=1)
            elif img_type == "lavision_im7":
                # For .im7 files, must provide camera_no
                img = read_image(str(file_path), camera_no=camera_num)
            elif img_type == "cine":
                # For .cine files, read first frame (idx=1)
                img = read_image(str(file_path), idx=1, frames=2)
            else:
                # Regular files don't need extra parameters
                img = read_image(str(file_path))

            # Handle both single images and image pairs
            if img.ndim == 3 and img.shape[0] == 2:
                # Image pair returned (e.g., from .im7 or .set)
                shape = tuple(img.shape[1:])
            else:
                # Single image
                shape = tuple(img.shape)

            logging.info(f"Detected image shape: {shape}")
            return shape

        except Exception as e:
            logging.error("Failed to read image: %s", e)
            raise ValueError(
                f"Could not read image file {file_path}. Error: {e}. "
                "Check that the file exists and is a valid image format."
            )

    @property
    def piv_chunk_size(self):
        # Updated to use batches.size from config.yaml
        return self.data["batches"]["size"]

    @property
    def batch_size(self):
        """
        Batch size for image processing.

        Automatically capped at num_frame_pairs to prevent batches larger than available data.
        """
        configured_size = self.data.get("batches", {}).get("size", 30)
        max_size = self.num_frame_pairs

        # Cap batch size at number of frame pairs
        actual_size = min(configured_size, max_size)

        if actual_size < configured_size:
            logging.debug(
                f"Batch size capped at {actual_size} (configured: {configured_size}, "
                f"max allowed: {max_size} frame pairs)"
            )

        return actual_size

    @property
    def filter_type(self):
        # This is now optional, as filters block is used
        return self.data.get("pre_procesing", {}).get("filter_type", None)

    @property
    def filters(self):
        # Returns the list of filter dicts from config.yaml
        return self.data.get("filters", [])

    @property
    def vector_format(self):
        # Returns a single format string like "B%05d.mat"
        vf = self.data["images"].get("vector_format", ["B%05d.mat"])
        if isinstance(vf, (list, tuple)):
            return vf[0]
        return vf

    @property
    def statistics_extraction(self):
        # Returns the statistics_extraction block as a list, or empty list if not present
        return self.data.get("statistics_extraction", [])

    @property
    def instantaneous_runs(self):
        return self.data.get("instantaneous_piv", {}).get("runs", [])

    @property
    def instantaneous_runs_0based(self):
        runs = self.instantaneous_runs
        if runs:
            return [r - 1 for r in runs]
        else:
            # Default to last pass if runs is empty
            return [self.num_passes - 1]

    @property
    def instantaneous_window_sizes(self):
        return self.data.get("instantaneous_piv", {}).get("window_size", [])

    @property
    def instantaneous_overlaps(self):
        return self.data.get("instantaneous_piv", {}).get("overlap", [])

    @property
    def plots(self):
        # Return the 'plots' dict from config.yaml
        return self.data.get("plots", {})

    @property
    def plot_save_extension(self):
        return self.plots.get("save_extension", ".png")

    @property
    def plot_save_pickle(self):
        return self.plots.get("save_pickle", True)

    @property
    def plot_fontsize(self):
        return self.plots.get("fontsize", 14)

    @property
    def plot_title_fontsize(self):
        return self.plots.get("title_fontsize", 16)

    @property
    def videos(self):
        """
        Returns the 'videos' list from config.yaml. Ensures a list is returned.
        Each entry may contain: type, endpoint, use_merged, video_length, variable.
        """
        vids = self.data.get("videos", [])
        if vids is None:
            return []
        if isinstance(vids, dict):
            return [vids]
        return list(vids)

    @property
    def post_processing(self):
        # Returns the post_processing block as a list, or empty list if not present
        return self.data.get("post_processing", [])

    # --- Calibration specific settings ---
    @property
    def calibration_image_format(self):
        """Return calibration image filename pattern.
        Default 'Calib%05d.tif'. If user supplies a plain filename (no %), it
        is used directly. If a dict block calibration: { image_format: ... }
        exists use that, else look for images.calibration_image_format for
        backward compatibility.
        """
        # Preferred location
        calib_block = self.data.get("calibration_format", {}) or {}
        fmt = calib_block.get("image_format", None)
        if not fmt:
            # fallback legacy key
            fmt = self.data.get("images", {}).get("calibration_image_format", None)
        if not fmt:
            fmt = "calib%05d.tif"
        return fmt

    def calibration_filename(self, index: int = 1):
        fmt = self.calibration_image_format
        try:
            if "%" in fmt:
                return fmt % index
            return fmt
        except Exception:
            # On formatting error, just return fmt
            return fmt

    @property
    def calibration(self):
        """Return the full calibration block (dict) from config."""
        return self.data.get("calibration", {})

    @property
    def active_calibration_method(self):
        """Return the active calibration method name (e.g., 'pinhole', 'scale_factor')."""
        cal = self.calibration
        return cal.get("active", "pinhole")

    @property
    def active_calibration_params(self):
        """Return the parameters dict for the active calibration method."""
        cal = self.calibration
        active = cal.get("active", "pinhole")
        return cal.get(active, {})

    @property
    def scale_factor_calibration(self):
        """Return scale factor calibration parameters."""
        return self.calibration.get("scale_factor", {})

    @property
    def pinhole_calibration(self):
        """Return pinhole calibration parameters."""
        return self.calibration.get("pinhole", {})

    @property
    def stereo_calibration(self):
        """Return stereo calibration parameters."""
        return self.calibration.get("stereo", {})

    @property
    def charuco_calibration(self):
        """Return ChArUco board calibration parameters."""
        return self.calibration.get("charuco", {})

    def get_calibration_method_params(self, method: str):
        """Get parameters for a specific calibration method."""
        return self.calibration.get(method, {})

    def set_active_calibration_method(self, method: str):
        """Set the active calibration method."""
        if method in ["scale_factor", "pinhole", "stereo", "charuco"]:
            self.data["calibration"]["active"] = method
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    # --- PIV-specific properties from pypivtools ---
    @property
    def window_sizes(self):
        """Return PIV window sizes from instantaneous_piv configuration."""
        return self.data.get("instantaneous_piv", {}).get("window_size", [])

    @property
    def overlap(self):
        """Return PIV overlap percentages."""
        overlaps = self.data.get("instantaneous_piv", {}).get("overlap", [])
        # Ensure we have as many overlaps as window sizes
        if overlaps and len(overlaps) == 1 and len(self.window_sizes) > 1:
            overlaps = overlaps * len(self.window_sizes)
        return overlaps

    @property
    def num_peaks(self):
        """Return number of peaks to detect in correlation."""
        return self.data.get("instantaneous_piv", {}).get("num_peaks", 1)

    @property
    def dt(self):
        """Return time difference between frames."""
        # Check active calibration method
        active_method = self.active_calibration_method
        if active_method == "stereo":
            return self.stereo_calibration.get("dt", 1)
        elif active_method == "pinhole":
            return self.pinhole_calibration.get("dt", 1)
        elif active_method == "scale_factor":
            return self.scale_factor_calibration.get("dt", 1)
        elif active_method == "charuco":
            return self.charuco_calibration.get("dt", 1)
        return 1

    @property
    def window_type(self):
        """Return PIV window type (e.g., 'gaussian', 'A')."""
        return self.data.get("instantaneous_piv", {}).get("window_type", "A")

    @property
    def backend(self):
        """Return processing backend ('cpu' or 'gpu')."""
        return self.data.get("processing", {}).get("backend", "cpu").lower()

    @property
    def num_passes(self):
        """Return number of PIV passes."""
        return len(self.window_sizes)

    @property
    def debug(self):
        """Return debug flag."""
        return self.data.get("processing", {}).get("debug", False)

    @property
    def auto_compute_params(self):
        """Return True if compute parameters should be auto-detected."""
        return self.data.get("processing", {}).get("auto_compute_params", False)

    def _get_auto_compute_params(self):
        """
        Auto-detect optimal compute parameters based on system resources.
        Results are cached to avoid repeated detection.
        
        Returns
        -------
        dict
            Dictionary with keys: omp_threads, dask_workers_per_node, 
            dask_threads_per_worker, dask_memory_limit
        """
        # Return cached result if available
        if self._auto_compute_cache is not None:
            return self._auto_compute_cache
        
        import psutil
        import os
        
        # Get number of CPU cores
        cpu_count = os.cpu_count() or 4
        
        # Get total system memory in GB
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Workers per node = number of CPUs
        workers_per_node = cpu_count
        
        # OMP threads = 2 (as requested)
        omp_threads = 2
        
        # Dask memory = (total memory - 10%) / cpu_count
        # Reserve 10% for system overhead
        available_memory_gb = total_memory_gb * 0.9
        memory_per_worker_gb = available_memory_gb / cpu_count
        dask_memory_limit = f"{memory_per_worker_gb:.2f}GB"
        
        # Threads per worker = 1 (standard for CPU-bound tasks)
        threads_per_worker = 1
        
        logging.info("Auto-detected compute parameters:")
        logging.info("  CPU cores: %d", cpu_count)
        logging.info("  Total memory: %.2f GB", total_memory_gb)
        logging.info("  Workers per node: %d", workers_per_node)
        logging.info("  OMP threads: %d", omp_threads)
        logging.info("  Memory per worker: %s", dask_memory_limit)
        logging.info("  Threads per worker: %d", threads_per_worker)
        
        # Cache the result
        self._auto_compute_cache = {
            "omp_threads": omp_threads,
            "dask_workers_per_node": workers_per_node,
            "dask_threads_per_worker": threads_per_worker,
            "dask_memory_limit": dask_memory_limit,
        }
        
        return self._auto_compute_cache

    @property
    def omp_threads(self):
        """Return number of OMP threads as string."""
        if self.auto_compute_params:
            return str(self._get_auto_compute_params()["omp_threads"])
        return str(self.data.get("processing", {}).get("omp_threads", 1))

    @property
    def dask_workers_per_node(self):
        """Return number of Dask workers per node."""
        if self.auto_compute_params:
            return self._get_auto_compute_params()["dask_workers_per_node"]
        return self.data.get("processing", {}).get("dask_workers_per_node", 1)

    @property
    def dask_threads_per_worker(self):
        """Return number of threads per Dask worker."""
        if self.auto_compute_params:
            return self._get_auto_compute_params()["dask_threads_per_worker"]
        return self.data.get("processing", {}).get("dask_threads_per_worker", 1)

    @property
    def dask_memory_limit(self):
        """Return memory limit per Dask worker."""
        if self.auto_compute_params:
            return self._get_auto_compute_params()["dask_memory_limit"]
        return self.data.get("processing", {}).get("dask_memory_limit", "4GB")

    @property
    def filter_omp_threads(self) -> int:
        """
        OMP threads for pipelined filter workers.

        During pipelined batch processing, filter workers run concurrently with
        correlation workers. Using all cores for filtering causes oversubscription.
        This setting controls thread count for subsequent batches (first batch
        still uses all cores since no correlation is running yet).

        Default: 2

        Returns
        -------
        int
            Number of OMP threads for pipelined filter batches
        """
        return self.data.get("processing", {}).get("filter_omp_threads", 2)

    @property
    def filter_worker_count(self):
        """
        Number of workers dedicated to filtering.

        Auto-determined based on filter types:
        - 1 worker for temporal filters (time, POD) to avoid memory issues
        - 2 workers for spatial-only filters
        - Can be overridden by setting manually in config

        Returns
        -------
        int
            Number of workers dedicated to filtering
        """
        # Check if manually set
        manual_count = self.data.get("processing", {}).get(
            "filter_worker_count"
        )
        if manual_count is not None:
            return manual_count

        # Auto-determine based on filter types
        from pivtools_cli.preprocessing.preprocess import has_batch_filters

        if has_batch_filters(self):
            # Temporal filters (time, POD): use 1 worker to avoid memory issues
            return 1
        else:
            # Spatial filters only: can use more workers
            return 2

    def get_filter_worker_allocation(self, total_workers: int):
        """
        Determine filter and correlation worker counts.

        Parameters
        ----------
        total_workers : int
            Total number of available workers

        Returns
        -------
        tuple
            (filter_workers, correlation_workers)

        Examples
        --------
        10 cores, count=1 → (1, 9)
        5 cores, count=2 → (2, 3)
        """
        filter_workers = max(
            1, min(self.filter_worker_count, total_workers - 1)
        )
        return filter_workers, total_workers - filter_workers

    @property
    def always_batch(self):
        """
        Force batch mode even for spatial filters (unified pipeline).

        When True, ALL processing uses batched pipeline for consistency.
        Default True for simplified architecture.

        Returns
        -------
        bool
            Whether to always use batched processing
        """
        return self.data.get("processing", {}).get("always_batch", True)

    @property
    def auto_batch_size(self):
        """
        Auto-determine batch size based on filters.

        Returns optimal batch size:
        - Temporal filters (POD/time): 30-50 for temporal coherence
        - Spatial filters only: 10-20 for lower latency
        - No filters: 5-10 for minimal overhead

        Returns
        -------
        int
            Optimal batch size for current configuration
        """
        from pivtools_cli.preprocessing.preprocess import has_batch_filters

        if has_batch_filters(self):
            # POD/time need larger batches
            return min(50, self.num_frame_pairs)
        else:
            # Spatial or no filters: smaller batches
            return min(20, self.num_frame_pairs)

    @property
    def peak_finder(self):
        """Return peak finder method (converted to numeric code)."""
        peak_finder = self.data.get("instantaneous_piv", {}).get("peak_finder", "gauss3").lower()
        if peak_finder == "gauss3":
            return 3
        elif peak_finder == "gauss4":
            return 4
        elif peak_finder == "gauss5":
            return 5
        elif peak_finder == "gauss6":
            return 6
        else:
            raise ValueError(
                f"Invalid peak_finder: {peak_finder}. Must be 'gauss3', 'gauss4', 'gauss5', or 'gauss6'."
            )

    @property
    def ensemble_piv(self):
        """Return True if ensemble PIV is enabled."""
        return self.data.get("processing", {}).get("ensemble", False)

    # --- Ensemble PIV properties ---
    @property
    def ensemble_window_sizes(self):
        """Return ensemble PIV window sizes."""
        return self.data.get("ensemble_piv", {}).get("window_size", self.window_sizes)

    @property
    def ensemble_overlaps(self):
        """Return ensemble PIV overlap percentages."""
        overlaps = self.data.get("ensemble_piv", {}).get("overlap", self.overlap)
        # Ensure we have as many overlaps as window sizes
        if overlaps and len(overlaps) == 1 and len(self.ensemble_window_sizes) > 1:
            overlaps = overlaps * len(self.ensemble_window_sizes)
        return overlaps

    @property
    def ensemble_runs(self):
        """Return list of 1-based passes to save for ensemble PIV."""
        return self.data.get("ensemble_piv", {}).get("runs", [])

    @property
    def ensemble_runs_0based(self):
        """Return list of 0-based passes to save for ensemble PIV."""
        runs = self.ensemble_runs
        if runs:
            return [r - 1 for r in runs]
        else:
            # Default to last pass if runs is empty
            return [self.ensemble_num_passes - 1]

    @property
    def ensemble_num_passes(self):
        """Return number of ensemble PIV passes."""
        return len(self.ensemble_window_sizes)

    @property
    def ensemble_window_type(self):
        """Return ensemble PIV window type (e.g., 'gaussian')."""
        return self.data.get("ensemble_piv", {}).get("window_type", self.window_type)

    @property
    def ensemble_num_peaks(self):
        """Return number of peaks for ensemble PIV."""
        return self.data.get("ensemble_piv", {}).get("num_peaks", self.num_peaks)

    @property
    def ensemble_peak_finder(self):
        """Return peak finder method for ensemble PIV (converted to numeric code)."""
        peak_finder = self.data.get("ensemble_piv", {}).get("peak_finder", "gauss6").lower()
        if peak_finder == "gauss3":
            return 3
        elif peak_finder == "gauss4":
            return 4
        elif peak_finder == "gauss5":
            return 5
        elif peak_finder == "gauss6":
            return 6
        else:
            raise ValueError(
                f"Invalid ensemble peak_finder: {peak_finder}. Must be 'gauss3', 'gauss4', 'gauss5', or 'gauss6'."
            )

    @property
    def ensemble_noisy(self):
        """
        Return True if Gaussian weighting should be applied for noisy ensemble data.

        When enabled, applies Gaussian windowing to help with noisy images.
        """
        return self.data.get("ensemble_piv", {}).get("noisy", False)

    @property
    def ensemble_sum_window(self):
        """
        Return sum window size for 'single' ensemble mode.

        Used when ensemble_type is 'single' for a pass, defines the correlation
        summation window size.

        Returns
        -------
        list
            [height, width] of sum window
        """
        sum_window = self.data.get("ensemble_piv", {}).get("sum_window", [16, 16])

        # Validate sum_window if single mode is used
        ensemble_types = self.ensemble_type
        if 'single' in ensemble_types:
            if sum_window is None:
                raise ValueError(
                    "ensemble_sum_window must be defined when using 'single' mode in ensemble_type"
                )
            if not isinstance(sum_window, (list, tuple)) or len(sum_window) != 2:
                raise ValueError(
                    f"ensemble_sum_window must be a list/tuple of [height, width], got {sum_window}"
                )
            # Validate sum_window is larger than all window sizes for single-mode passes
            for pass_idx, pass_type in enumerate(ensemble_types):
                if pass_type == 'single':
                    win_size = self.ensemble_window_sizes[pass_idx]
                    if sum_window[0] < win_size[0] or sum_window[1] < win_size[1]:
                        raise ValueError(
                            f"Pass {pass_idx}: ensemble_sum_window {sum_window} must be >= "
                            f"window_size {win_size} for single mode"
                        )

        return sum_window

    @property
    def ensemble_type(self):
        """
        Return ensemble type for each pass.

        Types:
        - 'std': Standard ensemble averaging of correlation planes
        - 'single': Single-pass mode with sum window

        Returns
        -------
        list
            List of type strings, one per pass
        """
        default_types = ["std"] * self.ensemble_num_passes
        types = self.data.get("ensemble_piv", {}).get("type", default_types)

        # Validate ensemble types
        valid_types = {'std', 'standard', 'single'}
        for pass_idx, pass_type in enumerate(types):
            if pass_type not in valid_types:
                raise ValueError(
                    f"Pass {pass_idx}: Invalid ensemble_type '{pass_type}'. "
                    f"Must be one of {valid_types}"
                )

        # Normalize 'standard' to 'std' for consistency
        types = ['std' if t == 'standard' else t for t in types]

        # Validate list length matches number of passes
        if len(types) != self.ensemble_num_passes:
            raise ValueError(
                f"ensemble_type list length ({len(types)}) must match "
                f"number of ensemble passes ({self.ensemble_num_passes})"
            )

        return types

    @property
    def ensemble_store_planes(self):
        """
        Return True if correlation planes should be stored for ensemble PIV.

        When enabled, saves AA, BB, AB correlation planes in 4D format
        to files named 'planes_pass_{pass_number}.mat'.
        """
        return self.data.get("ensemble_piv", {}).get("store_planes", False)

    @property
    def ensemble_save_diagnostics(self):
        """
        Return True if diagnostic images should be saved for ensemble PIV.

        When enabled, saves diagnostic images to a 'filters' subdirectory:
        - First batch, first pair: original images and after each filter applied
        - Each pass: warped images (A_warped, B_warped) for the first image pair

        All images are saved as 8-bit TIFFs for easy visualization.
        """
        return self.data.get("ensemble_piv", {}).get("save_diagnostics", False)

    @property
    def ensemble_resume_from_pass(self) -> int:
        """
        Return the pass index to resume from (1-based).

        When set, ensemble processing will skip passes 1 through (resume_from_pass - 1)
        and load the predictor field from the existing ensemble_result.mat in the
        output directory.

        Returns
        -------
        int
            Pass number to resume from (1-based), or 0 if not resuming
            E.g., 4 means skip passes 1-3 and start processing from pass 4

        Example
        -------
        If you completed passes 1-3 with window sizes [128, 64, 32] and want to add
        pass 4 at window size 16:

        ensemble_piv:
          window_size:
          - [128, 128]
          - [64, 64]
          - [32, 32]
          - [16, 16]   # New pass
          resume_from_pass: 4

        The existing ensemble_result.mat in the output directory must contain
        passes 1-3. Pass 4 will be appended to it.
        """
        return self.data.get("ensemble_piv", {}).get("resume_from_pass", 0)

    @property
    def outlier_detection_enabled(self):
        """Return True if outlier detection is enabled."""
        return self.data.get("outlier_detection", {}).get("enabled", True)
    
    @property
    def outlier_detection_methods(self):
        """Return list of outlier detection methods with their parameters."""
        return self.data.get("outlier_detection", {}).get("methods", [])
    
    @property
    def infilling_mid_pass(self):
        """Return mid-pass infilling configuration."""
        return self.data.get("infilling", {}).get("mid_pass", {
            "method": "local_median",
            "parameters": {"ksize": 3}
        })
    
    @property
    def infilling_final_pass(self):
        """Return final-pass infilling configuration."""
        return self.data.get("infilling", {}).get("final_pass", {
            "enabled": True,
            "method": "local_median",
            "parameters": {"ksize": 3}
        })

    # --- Ensemble-specific outlier detection and infilling ---
    @property
    def ensemble_outlier_detection_enabled(self) -> bool:
        """Return True if ensemble outlier detection is enabled."""
        return self.data.get("ensemble_outlier_detection", {}).get("enabled", True)

    @property
    def ensemble_outlier_detection_methods(self) -> list:
        """Return list of ensemble outlier detection methods with their parameters."""
        return self.data.get("ensemble_outlier_detection", {}).get("methods", [])

    @property
    def ensemble_infilling_mid_pass(self) -> dict:
        """Return ensemble mid-pass infilling configuration."""
        return self.data.get("ensemble_infilling", {}).get("mid_pass", {
            "method": "biharmonic",
            "parameters": {"ksize": 3}
        })

    @property
    def ensemble_infilling_final_pass(self) -> dict:
        """Return ensemble final-pass infilling configuration."""
        return self.data.get("ensemble_infilling", {}).get("final_pass", {
            "enabled": True,
            "method": "biharmonic",
            "parameters": {"ksize": 3}
        })

    @property
    def secondary_peak(self):
        """Return True if secondary peak detection is enabled."""
        return self.data.get("instantaneous_piv", {}).get("secondary_peak", False)

    # --- Logging properties ---
    @property
    def log_file(self) -> str:
        """Return log file path."""
        return self.data.get("logging", {}).get("file", "pypiv.log")

    @property
    def log_level(self) -> str:
        """Return log level as string."""
        return self.data.get("logging", {}).get("level", "INFO").upper()

    @property
    def log_console(self) -> bool:
        """Return True if console logging is enabled."""
        return self.data.get("logging", {}).get("console", True)

    def _setup_logging(self):
        """Setup logging based on configuration. Only runs once globally."""
        global _LOGGING_INITIALIZED
        
        if _LOGGING_INITIALIZED:
            return
        
        _LOGGING_INITIALIZED = True
        
        log_level = getattr(logging, self.log_level, logging.INFO)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Add console handler if requested
        if self.log_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        logging.info(
            "Logging initialized. Level: %s, File: %s", self.log_level, self.log_file
        )

    @property
    def image_dtype(self):
        """Return image data type as numpy dtype."""
        import numpy as np
        dtype_str = self.data.get("images", {}).get("dtype", "uint16")
        return np.dtype(dtype_str)

    # --- Masking properties ---
    @property
    def masking_enabled(self):
        """Return whether masking is enabled."""
        return self.data.get("masking", {}).get("enabled", False)

    @property
    def mask_file_pattern(self):
        """Return mask filename pattern. Default 'mask_Cam%d.mat'."""
        return self.data.get("masking", {}).get("mask_file_pattern", "mask_Cam%d.mat")

    @property
    def mask_mode(self):
        """
        Return masking mode: 'file' or 'rectangular'.
        
        Returns
        -------
        str
            'file' to load mask from .mat file, 'rectangular' for edge masking
        """
        return self.data.get("masking", {}).get("mode", "file")

    @property
    def mask_rectangular_settings(self):
        """
        Return rectangular masking settings (pixels to mask from each edge).
        
        Returns
        -------
        dict
            Dictionary with keys: top, bottom, left, right (all in pixels)
        """
        default = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        return self.data.get("masking", {}).get("rectangular", default)

    @property
    def mask_threshold(self):
        """
        Return mask threshold for vector masking.
        
        This threshold determines when a vector is masked based on the fraction
        of masked pixels within its interrogation window:
        - 0.0: mask vector if any pixel in window is masked
        - 0.5: mask vector if >50% of pixels in window are masked (default)
        - 1.0: only mask vector if all pixels in window are masked
        
        Returns
        -------
        float
            Threshold value between 0.0 and 1.0
        """
        return self.data.get("masking", {}).get("mask_threshold", 0.5)

    def get_mask_path(self, camera_num: int, source_path_idx: int = 0):
        """
        Get the full path to the mask file for a given camera.
        
        Parameters
        ----------
        camera_num : int
            Camera number (e.g., 1 for Cam1)
        source_path_idx : int, optional
            Index into source_paths list, defaults to 0
            
        Returns
        -------
        Path
            Full path to the mask .mat file
        """
        mask_filename = self.mask_file_pattern % camera_num
        return self.source_paths[source_path_idx] / mask_filename

    @property
    def zero_based_indexing(self):
        return self.data.get("images", {}).get("zero_based_indexing", False)

    @property
    def camera_subfolders(self):
        return self.data.get("paths", {}).get("camera_subfolders", [])

    def get_camera_folder(self, camera_num: int) -> str:
        """Get the subfolder name for a specific camera.

        Container formats (.cine, .set, .im7) don't use camera subfolders:
        - .set/.im7: All cameras in one file
        - .cine: Separate files per camera in source dir (uses %d in pattern)
        """
        # Container formats don't use camera subfolders
        if self.is_container_format:
            return ""

        subfolders = self.camera_subfolders
        # camera_num is 1-based
        idx = camera_num - 1

        if subfolders and idx < len(subfolders) and subfolders[idx]:
            return subfolders[idx]

        if self.camera_count == 1:
            return ""

        return f"Cam{camera_num}"


def get_config(refresh: bool = False) -> Config:
    """Return shared Config instance. Pass refresh=True to reload from disk."""
    global _CONFIG
    if refresh or _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def reload_config() -> Config:
    """Explicit convenience to force reload."""
    return get_config(refresh=True)
