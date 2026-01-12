from pathlib import Path
from typing import List, Optional, Tuple
import logging
import os
import shutil

import yaml

# ===================== ENDPOINT CONSTRAINTS =====================
# Tool-specific source_endpoint constraints - defines what data sources each tool can use
# source_endpoint values: "regular" (per-camera), "merged" (multi-camera merged), "stereo" (3D stereo PIV)
TOOL_ALLOWED_SOURCE_ENDPOINTS = {
    "video": ["regular", "merged", "stereo"],  # All source endpoints allowed
    "merging": ["regular"],  # Only regular (per-camera) data can be merged
    "statistics": ["regular", "merged", "stereo"],
    "transforms": ["regular", "merged", "stereo"],
}

# Tool-specific type_name constraints - defines what temporal types each tool can use
# type_name values: "instantaneous" (frame-by-frame), "ensemble" (averaged result)
TOOL_ALLOWED_TYPE_NAMES = {
    "video": ["instantaneous"],  # No ensemble (no temporal sequence for animation)
    "merging": ["instantaneous", "ensemble"],  # Both temporal types can be merged
    "statistics": ["instantaneous", "ensemble"],  # Statistics on either type
    "transforms": ["instantaneous", "ensemble"],  # Transforms on either type
}

# Legacy alias for backward compatibility
TOOL_ALLOWED_ENDPOINTS = TOOL_ALLOWED_SOURCE_ENDPOINTS

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

    def save(self):
        """Save current config state to YAML file."""
        self._normalize_calibration_block()
        with open(self._config_path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    def save_timestamped_copy(self, destination_dir: Path, timestamp: str = None) -> Path:
        """Save a timestamped copy of the config file for traceability.

        Args:
            destination_dir: Directory to save the config copy to
            timestamp: Optional timestamp string. If None, generates current timestamp.
                       Format: YYYY-MM-DD_HH-MM-SS

        Returns:
            Path to the saved config file
        """
        from datetime import datetime

        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create destination directory if it doesn't exist
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Build filename with timestamp
        dest_path = destination_dir / f"config_{timestamp}.yaml"

        # Copy the original config file (preserves exact formatting and comments)
        if Path(self._config_path).exists():
            shutil.copy2(self._config_path, dest_path)
        else:
            # Fallback: save current state if original file doesn't exist
            with open(dest_path, 'w') as f:
                yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

        return dest_path

    def _normalize_calibration_block(self):
        """Reorder calibration block keys for consistent organization.

        Groups image-related settings together at the top, followed by
        active method selection, then method-specific configs.
        """
        if "calibration" not in self.data:
            return

        cal = self.data["calibration"]

        # Define desired key order: image settings first, then method configs
        image_settings = [
            "image_format",
            "num_images",
            "image_type",
            "zero_based_indexing",
            "use_camera_subfolders",
            "subfolder",
            "camera_subfolders",
            "path_order",
        ]
        meta_settings = ["active", "piv_type"]
        method_configs = [
            "scale_factor",
            "dotboard",
            "charuco",
            "stereo_dotboard",
            "polynomial",
            "stereo_charuco",
        ]

        # Build ordered dict
        ordered = {}

        # Add image settings first
        for key in image_settings:
            if key in cal:
                ordered[key] = cal[key]

        # Add meta settings
        for key in meta_settings:
            if key in cal:
                ordered[key] = cal[key]

        # Add method configs
        for key in method_configs:
            if key in cal:
                ordered[key] = cal[key]

        # Add any remaining keys not in our lists
        for key, value in cal.items():
            if key not in ordered:
                ordered[key] = value

        self.data["calibration"] = ordered

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
calibration:
  image_format: calib%05d.tif
  num_images: 10
  image_type: standard
  zero_based_indexing: false
  use_camera_subfolders: false
  subfolder: ''
  camera_subfolders: []
  path_order: camera_first
  active: polynomial
  piv_type: instantaneous
  active_paths: []
  cameras: []
  scale_factor:
    dt: 0.56
    px_per_mm: 3.41
    source_path_idx: 0
  dotboard:
    camera: 1
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    enhance_dots: true
    asymmetric: false
    grid_tolerance: 0.5
    ransac_threshold: 3
    dt: 0.0275
    source_path_idx: 0
  charuco:
    camera: 1
    squares_h: 10
    squares_v: 9
    square_size: 0.03
    marker_ratio: 0.5
    aruco_dict: DICT_4X4_1000
    min_corners: 6
    dt: 1
    source_path_idx: 0
  stereo_dotboard:
    camera_pair:
    - 1
    - 2
    stereo_model_type: charuco
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
merging:
  type_name: instantaneous
  endpoint: ''
  max_workers: 8
  cameras: []
  active_paths: []
statistics:
  active_paths: []
  cameras: []
  include_merged: false
  gamma_radius: 5
  save_figures: true
  type_name: instantaneous
  enabled_methods:
    mean_velocity: true
    reynolds_stress: true
    normal_stress: true
    mean_tke: true
    mean_vorticity: true
    mean_divergence: true
    inst_velocity: true
    inst_fluctuations: true
    inst_vorticity: true
    inst_divergence: true
    inst_gamma: true
transforms:
  base_path_idx: 0
  type_name: instantaneous
  active_paths: []
  include_merged: false
  cameras: {}
video:
  base_path_idx: 0
  camera: 1
  data_source: calibrated
  variable: ux
  run: 1
  piv_type: instantaneous
  cmap: default
  lower: ''
  upper: ''
  fps: 30
  crf: 15
  resolution: 1080p
  active_paths: []
  cameras: []
  include_merged: false

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

        For IM7: Returns False if images_use_camera_subfolders is True,
        since each file contains only one camera's data in that case.
        """
        if self.image_type == "lavision_im7" and self.images_use_camera_subfolders:
            return False  # Single-camera files when using subfolders
        return self.image_type in ("lavision_set", "lavision_im7")

    @property
    def images_use_camera_subfolders(self) -> bool:
        """Return True if PIV images use camera subfolders for IM7 files.

        When True, IM7 files are expected in camera subdirectories:
        - source_path/Cam1/B00001.im7
        - source_path/Cam2/B00001.im7

        Each file contains only ONE camera's data (no camera_no parameter needed).

        When False (default), all cameras are in one IM7 file per time instance:
        - source_path/B00001.im7 contains all cameras
        - camera_no parameter is used to extract specific camera data.
        """
        return self.data.get("images", {}).get("use_camera_subfolders", False)

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

    # ===================== STEREO PAIR PROPERTIES =====================

    @property
    def stereo_pairs(self) -> List[Tuple[int, int]]:
        """Auto-derive stereo camera pairs from camera_numbers order.

        Cameras are paired sequentially: [1,2,3,4] -> [(1,2), (3,4)]
        This means cameras 1,2 form stereo pair 1, cameras 3,4 form stereo pair 2.

        Returns
        -------
        List[Tuple[int, int]]
            List of (cam1, cam2) tuples for stereo processing
        """
        cameras = self.camera_numbers
        pairs = []
        for i in range(0, len(cameras) - 1, 2):
            if i + 1 < len(cameras):
                pairs.append((cameras[i], cameras[i + 1]))
        return pairs

    @property
    def is_stereo_setup(self) -> bool:
        """Return True if this is a stereo PIV setup.

        Determined by calibration.active being 'stereo_dotboard' or 'stereo_charuco'.

        Returns
        -------
        bool
            True if stereo calibration is active
        """
        active = self.active_calibration_method
        return active in ("stereo_dotboard", "stereo_charuco")

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
            # For .set files, source_path IS the .set file - use directly
            # (don't append format_str as that would create invalid path)
            file_path = camera_path
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
                # For .im7 files, check if single-camera or multi-camera
                if self.images_use_camera_subfolders:
                    # Single-camera file: don't pass camera_no
                    img = read_image(str(file_path))
                else:
                    # Multi-camera file: pass camera_no
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

    # --- Statistics properties ---
    @property
    def statistics(self) -> dict:
        """Return full statistics configuration block."""
        return self.data.get("statistics", {})

    @property
    def statistics_enabled_methods(self) -> dict:
        """Return dictionary of enabled statistics methods.

        Returns
        -------
        dict
            Dictionary with method names as keys and booleans as values.
            Keys match frontend IDs for 1:1 mapping.
            E.g., {'mean_velocity': True, 'reynolds_stress': True, ...}
        """
        default_methods = {
            # Mean/time-averaged statistics
            "mean_velocity": True,
            "reynolds_stress": True,
            "normal_stress": True,
            "mean_tke": True,
            "mean_vorticity": True,
            "mean_divergence": True,
            # Instantaneous (per-frame) statistics
            "inst_velocity": True,
            "inst_fluctuations": True,
            "inst_vorticity": True,
            "inst_divergence": True,
            "inst_gamma": True,
        }
        return self.statistics.get("enabled_methods", default_methods)

    @property
    def statistics_enabled_list(self) -> list:
        """Return list of enabled statistics method names.

        Returns
        -------
        list
            List of method names that are enabled (True).
            E.g., ['mean_velocity', 'reynolds_stress', 'tke']
        """
        methods = self.statistics_enabled_methods
        return [name for name, enabled in methods.items() if enabled]

    @property
    def statistics_gamma_radius(self) -> int:
        """Return gamma function radius parameter.

        Used for gamma1 and gamma2 vortex identification.

        Returns
        -------
        int
            Radius in grid points (default 5)
        """
        return self.statistics.get("gamma_radius", 5)

    @property
    def statistics_save_figures(self) -> bool:
        """Return whether to save statistics figures.

        Returns
        -------
        bool
            True to save figures (default True)
        """
        return self.statistics.get("save_figures", True)

    @property
    def statistics_type_name(self) -> str:
        """Return statistics type name for folder organization.

        Returns
        -------
        str
            Type name (default 'instantaneous')
        """
        return self.statistics.get("type_name", "instantaneous")

    @property
    def statistics_source_endpoint(self) -> str:
        """Return source endpoint for statistics.

        Determines what data type statistics are computed on:
        - 'instantaneous': Single-frame PIV vectors
        - 'ensemble': Ensemble-averaged vectors
        - 'merged': Multi-camera merged vectors
        - 'stereo': 3D stereo PIV vectors

        Returns
        -------
        str
            Source endpoint (default 'regular')
        """
        return self.statistics.get("source_endpoint", "regular")

    @property
    def statistics_workflow(self) -> str:
        """Return statistics workflow preference.

        Options:
        - 'per_camera': Compute stats for each camera independently
        - 'after_merge': Only compute stats on merged data
        - 'both': Compute per-camera stats then merged stats

        Returns
        -------
        str
            Workflow preference (default 'per_camera')
        """
        return self.statistics.get("workflow", "per_camera")

    @property
    def statistics_process_cameras(self) -> bool:
        """Return whether to process individual camera data.

        Returns
        -------
        bool
            True to process individual cameras (default True)
        """
        return self.statistics.get("process_cameras", True)

    @property
    def statistics_process_merged(self) -> bool:
        """Return whether to process merged camera data.

        Returns
        -------
        bool
            True to process merged data (default False)
        """
        return self.statistics.get("process_merged", False)

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

    # --- Video properties (single dict format) ---

    @property
    def video(self) -> dict:
        """Return full video configuration block."""
        return self.data.get("video", {})

    @property
    def video_base_path_idx(self) -> int:
        """Return base path index for video operations."""
        return self.video.get("base_path_idx", 0)

    @property
    def video_camera(self) -> int:
        """Return camera number for video (1-based)."""
        return self.video.get("camera", 1)

    @property
    def video_data_source(self) -> str:
        """Return data source: 'calibrated', 'uncalibrated', 'merged', 'inst_stats'."""
        return self.video.get("data_source", "calibrated")

    @property
    def video_variable(self) -> str:
        """Return variable name for video."""
        return self.video.get("variable", "ux")

    @property
    def video_run(self) -> int:
        """Return run number (1-based)."""
        return self.video.get("run", 1)

    @property
    def video_piv_type(self) -> str:
        """Return PIV type: 'instantaneous' or 'ensemble'."""
        return self.video.get("piv_type", "instantaneous")

    @property
    def video_cmap(self) -> str:
        """Return colormap name. 'default' means auto-select."""
        return self.video.get("cmap", "default")

    @property
    def video_lower_limit(self) -> Optional[float]:
        """Return lower color limit. None or '' means auto."""
        val = self.video.get("lower", "")
        if val == "" or val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @property
    def video_upper_limit(self) -> Optional[float]:
        """Return upper color limit. None or '' means auto."""
        val = self.video.get("upper", "")
        if val == "" or val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @property
    def video_fps(self) -> int:
        """Return video frame rate."""
        return self.video.get("fps", 30)

    @property
    def video_crf(self) -> int:
        """Return video CRF quality (lower = higher quality)."""
        return self.video.get("crf", 15)

    @property
    def video_resolution(self) -> tuple:
        """Return video resolution as (height, width) tuple."""
        res = self.video.get("resolution", "1080p")
        if isinstance(res, str):
            if res == "4k":
                return (2160, 3840)
            return (1080, 1920)
        elif isinstance(res, (list, tuple)) and len(res) >= 2:
            return (res[0], res[1])
        return (1080, 1920)

    @property
    def video_resolution_str(self) -> str:
        """Return resolution as string: '1080p' or '4k'."""
        res = self.video.get("resolution", "1080p")
        if isinstance(res, str):
            return res
        elif isinstance(res, (list, tuple)):
            if res[0] >= 2160:
                return "4k"
        return "1080p"

    @property
    def video_source_endpoint(self) -> str:
        """Return source endpoint for video creation.

        Determines what data type to create video from:
        - 'instantaneous': Single-frame PIV vectors (has temporal sequence)
        - 'merged': Multi-camera merged vectors (has temporal sequence)

        Note: 'ensemble' is not allowed (no temporal sequence - just mean field).

        Returns
        -------
        str
            Source endpoint (default 'regular')
        """
        return self.video.get("source_endpoint", "regular")

    @property
    def videos(self):
        """DEPRECATED: Use video property instead. Returns list for backward compatibility."""
        # If old format exists, return it
        vids = self.data.get("videos", None)
        if vids is not None:
            if vids is None:
                return []
            if isinstance(vids, dict):
                return [vids]
            return list(vids)
        # Otherwise, return new format as single-item list
        vid = self.data.get("video", {})
        if vid:
            return [vid]
        return []

    @property
    def post_processing(self):
        # Returns the post_processing block as a list, or empty list if not present
        return self.data.get("post_processing", [])

    # --- Calibration specific settings ---
    # All calibration settings are now unified under the 'calibration' block

    @property
    def calibration_image_format(self) -> str:
        """Return calibration image filename pattern.

        Now reads from unified calibration block.
        Default 'calib%05d.tif'.
        """
        calib_block = self.data.get("calibration", {}) or {}
        fmt = calib_block.get("image_format", None)
        return fmt

    def calibration_filename(self, index: int = 1) -> str:
        """Generate calibration filename for a given index."""
        fmt = self.calibration_image_format
        try:
            if "%" in fmt:
                return fmt % index
            return fmt
        except Exception:
            return fmt

    @property
    def calibration_image_count(self) -> int:
        """Return number of calibration images expected."""
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("num_images", 10)

    @property
    def calibration_image_type(self) -> str:
        """Return calibration image type: 'standard', 'cine', 'lavision_set', 'lavision_im7'.

        If explicitly set in config, returns that value.
        Otherwise, auto-detects from calibration_image_format pattern.
        """
        calib_block = self.data.get("calibration", {}) or {}
        explicit_type = calib_block.get("image_type")
        if explicit_type:
            return explicit_type
        return self._detect_calibration_image_type()

    def _detect_calibration_image_type(self) -> str:
        """Auto-detect calibration image type from format string."""
        fmt = self.calibration_image_format.lower()
        if '.cine' in fmt:
            return "cine"
        elif '.set' in fmt:
            return "lavision_set"
        elif '.im7' in fmt:
            return "lavision_im7"
        elif '.ims' in fmt:
            return "lavision_im7"
        else:
            return "standard"

    @property
    def calibration_is_container_format(self) -> bool:
        """Return True if calibration format stores multiple frames in single container."""
        return self.calibration_image_type in ("cine", "lavision_set", "lavision_im7")

    @property
    def calibration_zero_based_indexing(self) -> bool:
        """Return True if calibration image indices start at 0."""
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("zero_based_indexing", False)

    @property
    def calibration_use_camera_subfolders(self) -> bool:
        """Return True if calibration images use camera subfolders (Cam1/, Cam2/).

        When True, calibration images are expected in camera subdirectories:
        - source_path/Cam1/calibration_subfolder/image.tif
        - source_path/Cam2/calibration_subfolder/image.tif

        When False (default), all calibration images are in a single directory:
        - source_path/calibration_subfolder/image.tif

        This applies to both standard formats (TIFF, PNG, etc.) and IM7 files.
        Container formats (.set, .cine) never use camera subfolders.
        """
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("use_camera_subfolders", False)

    @property
    def calibration_subfolder(self) -> str:
        """Return subfolder for calibration images.

        Path structure depends on calibration_path_order:
        - camera_first: source_path / camera_subfolder / calibration_subfolder / image_file
        - calibration_first: source_path / calibration_subfolder / camera_subfolder / image_file
        """
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("subfolder", "")

    @property
    def calibration_camera_subfolders(self) -> list:
        """Return custom camera subfolder names for calibration images.

        Independent from paths.camera_subfolders - specifically for calibration.
        If not set, returns empty list (will use default Cam1, Cam2... pattern).

        Example: ["camera1", "camera2"] for cameras in folders named camera1/, camera2/
        """
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("camera_subfolders", [])

    @property
    def calibration_path_order(self) -> str:
        """Return path order for calibration images.

        Controls the order of camera folder and calibration subfolder in the path:
        - 'camera_first': source/camera_folder/calibration_subfolder/file (default)
        - 'calibration_first': source/calibration_subfolder/camera_folder/file

        Returns
        -------
        str
            Path order: 'camera_first' or 'calibration_first'
        """
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("path_order", "camera_first")

    def get_calibration_camera_folder(self, camera_num: int) -> str:
        """Get the subfolder name for calibration images of a specific camera.

        Container formats (.cine, .set) never use camera subfolders.
        Standard and IM7 formats respect the calibration_use_camera_subfolders setting.

        Uses calibration.camera_subfolders if set, otherwise falls back to
        default Cam{N} pattern for multi-camera setups.
        """
        # SET and CINE never use camera subfolders
        if self.calibration_image_type in ("lavision_set", "cine"):
            return ""

        # Standard and IM7 formats: check calibration_use_camera_subfolders
        if not self.calibration_use_camera_subfolders:
            return ""

        # Use calibration-specific camera subfolders if available
        subfolders = self.calibration_camera_subfolders
        if subfolders:
            idx = camera_num - 1  # camera_num is 1-based
            if idx < len(subfolders) and subfolders[idx]:
                return subfolders[idx]

        # Generate default folder name for multi-camera setups
        if self.camera_count > 1:
            return f"Cam{camera_num}"

        return ""

    def get_calibration_image_path(self, camera: int, index: int, source_path_idx: int = 0) -> Path:
        """Build full path to a calibration image.

        Path structure: source_path / camera_subfolder / calibration_subfolder / image_file

        Parameters
        ----------
        camera : int
            Camera number (1-based)
        index : int
            Image index (1-based or 0-based depending on calibration_zero_based_indexing)
        source_path_idx : int
            Index into source_paths list

        Returns
        -------
        Path
            Full path to the calibration image file
        """
        source_path = self.source_paths[source_path_idx]
        camera_folder = self.get_calibration_camera_folder(camera)

        # Build base path: source_path / camera_subfolder
        if camera_folder:
            camera_path = source_path / camera_folder
        else:
            camera_path = source_path

        # Add calibration subfolder if specified
        if self.calibration_subfolder:
            camera_path = camera_path / self.calibration_subfolder

        image_type = self.calibration_image_type
        fmt = self.calibration_image_format

        # For container formats, the path is just the container file
        if image_type == "lavision_set":
            return camera_path / fmt
        elif image_type == "cine":
            # CINE pattern uses %d for camera number
            if "%" in fmt:
                return camera_path / (fmt % camera)
            return camera_path / fmt
        elif image_type == "lavision_im7":
            # IM7 uses %d for frame index
            if "%" in fmt:
                return camera_path / (fmt % index)
            return camera_path / fmt
        else:
            # Standard formats use %d for frame index
            if "%" in fmt:
                return camera_path / (fmt % index)
            return camera_path / fmt

    @property
    def calibration(self):
        """Return the full calibration block (dict) from config."""
        return self.data.get("calibration", {})

    @property
    def active_calibration_method(self):
        """Return the active calibration method name (e.g., 'dotboard', 'scale_factor')."""
        cal = self.calibration
        return cal.get("active", "dotboard")

    @property
    def active_calibration_params(self):
        """Return the parameters dict for the active calibration method."""
        cal = self.calibration
        active = cal.get("active", "dotboard")
        return cal.get(active, {})

    @property
    def scale_factor_calibration(self):
        """Return scale factor calibration parameters."""
        return self.calibration.get("scale_factor", {})

    @property
    def dotboard_calibration(self):
        """Return dotboard calibration parameters."""
        return self.calibration.get("dotboard", {})

    @property
    def stereo_calibration(self):
        """Return stereo calibration parameters (shared stereo settings)."""
        return self.calibration.get("stereo", {})

    @property
    def stereo_dotboard_calibration(self):
        """Return stereo dotboard calibration parameters."""
        return self.calibration.get("stereo_dotboard", {})

    @property
    def charuco_calibration(self):
        """Return ChArUco board calibration parameters."""
        return self.calibration.get("charuco", {})

    @property
    def polynomial_calibration(self):
        """Return polynomial calibration parameters."""
        return self.calibration.get("polynomial", {})

    def get_polynomial_camera_params(self, camera_num: int) -> dict:
        """Get polynomial parameters for a specific camera.

        Parameters
        ----------
        camera_num : int
            Camera number (1-based)

        Returns
        -------
        dict
            Camera-specific polynomial parameters including:
            - origin: {x, y} - pixel origin for normalization
            - normalisation: {nx, ny} - normalization factors
            - mm_per_pixel: float - scale factor
            - coefficients_x: list[float] - 10 polynomial coefficients for X
            - coefficients_y: list[float] - 10 polynomial coefficients for Y
        """
        cameras = self.polynomial_calibration.get("cameras", {})
        # Try both string and int keys for compatibility
        return cameras.get(str(camera_num), cameras.get(camera_num, {}))

    @property
    def stereo_charuco_calibration(self):
        """Return stereo ChArUco calibration parameters."""
        return self.calibration.get("stereo_charuco", {})

    @property
    def calibration_piv_type(self) -> str:
        """Return PIV type for calibration: 'instantaneous' or 'ensemble'.

        This determines which vector data directory to use when calibrating vectors.
        """
        calib_block = self.data.get("calibration", {}) or {}
        return calib_block.get("piv_type", "instantaneous")

    def get_calibration_method_params(self, method: str):
        """Get parameters for a specific calibration method."""
        return self.calibration.get(method, {})

    def set_active_calibration_method(self, method: str):
        """Set the active calibration method."""
        if method in ["scale_factor", "dotboard", "stereo_dotboard", "charuco", "polynomial", "stereo_charuco"]:
            self.data["calibration"]["active"] = method
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    # --- Merging properties ---
    @property
    def merging(self) -> dict:
        """Return full merging configuration block."""
        return self.data.get("merging", {})

    @property
    def merging_type_name(self) -> str:
        """Return default vector type for merging.

        Returns
        -------
        str
            Vector type: 'instantaneous', 'ensemble', etc.
        """
        return self.merging.get("type_name", "instantaneous")

    @property
    def merging_cameras(self) -> list:
        """Return default cameras to merge.

        Falls back to camera_numbers if not explicitly set.

        Returns
        -------
        list
            List of camera numbers to merge (e.g., [1, 2])
        """
        cameras = self.merging.get("cameras")
        if cameras:
            return cameras
        return self.camera_numbers

    @property
    def merging_base_path_idx(self) -> int:
        """Return default base path index for merging operations.

        Returns
        -------
        int
            Index into base_paths list (default 0)
        """
        return self.merging.get("base_path_idx", 0)

    @property
    def merging_source_endpoint(self) -> str:
        """Return source endpoint for vector merging.

        Determines what data type to merge:
        - 'instantaneous': Single-frame PIV vectors
        - 'ensemble': Ensemble-averaged vectors

        Note: 'stereo' and 'merged' are not allowed (3D vectors can't merge).

        Returns
        -------
        str
            Source endpoint (default 'regular')
        """
        return self.merging.get("source_endpoint", "regular")

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
        if active_method == "stereo_dotboard":
            return self.stereo_dotboard_calibration.get("dt", 1)
        elif active_method == "dotboard":
            return self.dotboard_calibration.get("dt", 1)
        elif active_method == "scale_factor":
            return self.scale_factor_calibration.get("dt", 1)
        elif active_method == "charuco":
            return self.charuco_calibration.get("dt", 1)
        elif active_method == "stereo_charuco":
            return self.stereo_charuco_calibration.get("dt", 1)
        elif active_method == "polynomial":
            return self.polynomial_calibration.get("dt", 1)
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
        return 1

    @property
    def dask_memory_limit(self):
        """Return memory limit per Dask worker."""
        if self.auto_compute_params:
            return self._get_auto_compute_params()["dask_memory_limit"]
        return self.data.get("processing", {}).get("dask_memory_limit", "4GB")

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

    # --- Ensemble PIV properties ---
    @property
    def ensemble_window_sizes(self):
        """Return ensemble PIV window sizes."""
        return self.data.get("ensemble_piv", {}).get("window_size", self.window_sizes)

    @property
    def ensemble_overlaps(self):
        """Return ensemble PIV overlap percentages."""
        overlaps = self.data.get("ensemble_piv", {}).get("overlap", self.overlap)
        n_passes = len(self.ensemble_window_sizes)

        # Broadcast single overlap to all passes
        if overlaps and len(overlaps) == 1 and n_passes > 1:
            overlaps = overlaps * n_passes

        # Validate array length matches window_sizes
        if overlaps and len(overlaps) != n_passes:
            raise ValueError(
                f"ensemble_piv.overlap has {len(overlaps)} entries but "
                f"ensemble_piv.window_size has {n_passes} entries. "
                f"These must match (or use a single overlap value to broadcast)."
            )

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
    def ensemble_fit_offset(self) -> bool:
        """Enable/disable offset (+C) term in stacked Gaussian fitting.

        When True (default): y = amp * exp(...) + c
        When False: y = amp * exp(...) (no offset term)
        """
        return self.data.get("ensemble_piv", {}).get("fit_offset", True)

    @property
    def ensemble_mask_center_pixel(self) -> bool:
        """Enable/disable center pixel masking for autocorrelation.

        When True (default): Exclude center pixel of AA/BB planes from fitting
        to remove camera self-noise spike at zero lag.
        When False: Include all pixels (for synthetic data or testing).

        The cross-correlation (AB) center pixel is never masked since it
        contains valid displacement signal.
        """
        return self.data.get("ensemble_piv", {}).get("mask_center_pixel", True)

    @property
    def ensemble_fit_method(self) -> str:
        """Return fitting method for ensemble PIV.

        Options:
        - 'gaussian': Levenberg-Marquardt 16-parameter stacked Gaussian (default)
        - 'kspace': K-space transfer function with 6 parameters

        The k-space method offers better noise robustness by:
        - Algebraic cancellation of particle shape (6 params vs 16)
        - Adaptive SNR-based wavenumber bounds
        - Forward-model fitting that emphasizes high-SNR components
        """
        method = self.data.get("ensemble_piv", {}).get("fit_method", "gaussian")
        valid_methods = {'gaussian', 'kspace'}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_fit_method '{method}'. "
                f"Must be one of {valid_methods}"
            )
        return method

    @property
    def ensemble_kspace_snr_threshold(self) -> float:
        """Return SNR threshold for k-space adaptive bounds.

        Wavenumbers with SNR below this threshold are excluded from fitting.
        Higher values are more conservative (exclude more noise).

        Default: 3.0
        """
        return self.data.get("ensemble_piv", {}).get("kspace_snr_threshold", 3.0)

    @property
    def ensemble_kspace_soft_weighting(self) -> bool:
        """Return whether to use anisotropic soft decay weighting in k-space fitting.

        When True (default): Uses combined SNR × anisotropic soft decay weighting:
            w(k) = w_snr(k) * exp(-k_x²/k0_x² - k_y²/k0_y²)

        where k0_x and k0_y are computed from Sigma_xx and Sigma_yy estimates.

        This naturally down-weights high-k regions where:
        - Signal-to-noise is poor
        - The Gaussian model becomes less accurate

        Benefits:
        - Avoids hard k_max cutoffs
        - Automatically adapts to signal quality
        - Handles anisotropic stresses (different turbulence in x vs y)
        - Reduces bias from model mismatch at high k

        When False: Uses uniform weighting within k_max bounds.

        Default: True
        """
        return self.data.get("ensemble_piv", {}).get("kspace_soft_weighting", True)

    @property
    def ensemble_image_warp_interpolation(self) -> str:
        """Return interpolation method for image warping in ensemble PIV.

        This controls the cv2.remap interpolation when warping images based on
        the predictor field from the previous pass. The choice of interpolation
        may affect:
        - Particle image sharpness (PSF)
        - Measured Reynolds stress (peak width)
        - Processing speed

        Options:
        - 'nearest': cv2.INTER_NEAREST (fastest, no smoothing, may cause aliasing)
        - 'linear': cv2.INTER_LINEAR (bilinear, moderate smoothing)
        - 'cubic': cv2.INTER_CUBIC (bicubic, smoothest, default)

        Default: 'cubic'
        """
        method = self.data.get("ensemble_piv", {}).get(
            "image_warp_interpolation", "cubic"
        )
        valid_methods = {'nearest', 'linear', 'cubic'}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_image_warp_interpolation '{method}'. "
                f"Must be one of {valid_methods}"
            )
        return method

    @property
    def ensemble_predictor_interpolation(self) -> str:
        """Return interpolation method for predictor field in ensemble PIV.

        This controls the cv2.remap interpolation when upsampling the predictor
        field from coarse to fine grids and from window centers to dense pixel
        coordinates.

        Options:
        - 'nearest': cv2.INTER_NEAREST (fastest, may cause blocky artifacts)
        - 'linear': cv2.INTER_LINEAR (bilinear, good balance)
        - 'cubic': cv2.INTER_CUBIC (bicubic, smoothest, default)

        Default: 'cubic'
        """
        method = self.data.get("ensemble_piv", {}).get(
            "predictor_interpolation", "cubic"
        )
        valid_methods = {'nearest', 'linear', 'cubic'}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_predictor_interpolation '{method}'. "
                f"Must be one of {valid_methods}"
            )
        return method

    @property
    def ensemble_skip_background_subtraction(self) -> bool:
        """Skip background subtraction in ensemble PIV (debug/testing only).

        When True, skips the single-pass optimization formula:
            R_ensemble = <A⊗B> - <A>⊗<B>

        And instead uses raw correlation planes directly:
            R_ensemble = <A⊗B>

        WARNING: This is for testing/debugging only. Without background
        subtraction, correlation planes will have elevated noise floors
        which may affect fitting quality.

        Default: False
        """
        return self.data.get("ensemble_piv", {}).get("skip_background_subtraction", False)

    @property
    def ensemble_background_subtraction_method(self) -> str:
        """Background subtraction method for ensemble PIV.

        Options:
        - 'correlation': R = <A⊗B> - <A>⊗<B> (current default, single-pass)
          Correlates raw images, then subtracts correlation of mean images.
          More memory efficient (single pass through data).

        - 'image': R = <(A-Ā)⊗(B-B̄)> (two-pass, subtract mean images first)
          First pass computes mean images, second pass correlates mean-subtracted
          images. Requires two iterations through the data but may be more
          numerically stable for certain fitting methods (e.g., k-space).

        Both methods are mathematically equivalent but may differ numerically
        due to order of operations and floating-point precision.

        Default: 'correlation'
        """
        method = self.data.get("ensemble_piv", {}).get(
            "background_subtraction_method", "correlation"
        )
        valid_methods = {'correlation', 'image'}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_background_subtraction_method '{method}'. "
                f"Must be one of {valid_methods}"
            )
        return method

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
    def ensemble_gradient_correction(self) -> bool:
        """Apply gradient correction to Reynolds stresses.

        When True, applies velocity gradient correction to Reynolds stress estimates:
            UU_corrected = UU_stress - 0.5 * sig_A_x * (dU/dy)²
            VV_corrected = VV_stress - 0.5 * sig_A_y * (dV/dx)²
            UV_corrected = UV_stress - 0.5 * sig_A_x * (dU/dy + dV/dx)

        This correction accounts for velocity gradient bias in the stress estimates,
        which is particularly important in regions with strong velocity gradients
        (e.g., near walls in boundary layer flows).

        The correction requires sig_A_x and sig_A_y fields from Gaussian fitting,
        which are only available in uncalibrated ensemble PIV results.

        Default: False
        """
        return self.data.get("ensemble_piv", {}).get("gradient_correction", False)

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

        For .set files, masks are stored in a dedicated storage directory
        (e.g., /path/to/file_data/) with the set filename in the mask name.
        E.g., source_path="/data/experiment.set" -> "/data/experiment_data/mask_experiment_Cam1.mat"

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
        base_pattern = self.mask_file_pattern % camera_num
        storage_dir = self.get_storage_directory(source_path_idx)

        # For .set files, include the set filename to disambiguate
        if self.image_type == "lavision_set":
            # source_path IS the .set file, so get stem from it directly
            set_stem = self.source_paths[source_path_idx].stem  # e.g., "experiment"
            # Insert set name: "mask_Cam1.mat" -> "mask_experiment_Cam1.mat"
            mask_filename = base_pattern.replace("mask_", f"mask_{set_stem}_")
        else:
            mask_filename = base_pattern

        return storage_dir / mask_filename

    @property
    def zero_based_indexing(self):
        return self.data.get("images", {}).get("zero_based_indexing", False)

    @property
    def camera_subfolders(self):
        return self.data.get("paths", {}).get("camera_subfolders", [])

    @property
    def active_paths(self) -> list:
        """Return list of active path indices to process.

        Supports GUI override via PIV_ACTIVE_PATHS environment variable.
        Defaults to all paths if not specified.

        Returns
        -------
        list[int]
            List of 0-indexed path indices to process
        """
        # Check environment variable override (from GUI)
        env_paths = os.environ.get('PIV_ACTIVE_PATHS')
        if env_paths:
            try:
                indices = [int(i) for i in env_paths.split(',') if i.strip()]
                # Validate indices
                max_idx = len(self.source_paths) - 1
                return [i for i in indices if 0 <= i <= max_idx]
            except ValueError:
                pass

        # Fall back to config file
        paths_data = self.data.get("paths", {})
        active = paths_data.get("active_paths")

        if active is None:
            # Default: all paths are active
            return list(range(len(self.source_paths)))

        # Validate indices
        max_idx = len(self.source_paths) - 1
        return [i for i in active if 0 <= i <= max_idx]

    # --- .set file path helpers ---

    def get_storage_directory(self, source_path_idx: int = 0) -> Path:
        """Get storage directory for masks (per-.set-file storage).

        For .set files: derives storage from filename (e.g., /path/to/file_data/)
        For other formats: returns source_path itself

        Parameters
        ----------
        source_path_idx : int, optional
            Index into source_paths list, defaults to 0

        Returns
        -------
        Path
            Directory for storing masks and other per-dataset files
        """
        source_path = self.source_paths[source_path_idx]
        if self.image_type == "lavision_set":
            # .set files: storage is sibling directory named {stem}_data
            return source_path.parent / f"{source_path.stem}_data"
        return source_path

    def get_source_directory(self, source_path_idx: int = 0) -> Path:
        """Get source directory (parent for .set, same for directories).

        Use this for calibration images and other assets relative to source.
        For .set files: returns parent directory (e.g., /path/to/)
        For other formats: returns source_path itself

        Parameters
        ----------
        source_path_idx : int, optional
            Index into source_paths list, defaults to 0

        Returns
        -------
        Path
            Base directory for calibration images and related assets
        """
        source_path = self.source_paths[source_path_idx]
        if self.image_type == "lavision_set":
            return source_path.parent
        return source_path

    def get_set_file_path(self, source_path_idx: int = 0) -> Path:
        """Get the .set file path. For lavision_set, source_path IS the file.

        Parameters
        ----------
        source_path_idx : int, optional
            Index into source_paths list, defaults to 0

        Returns
        -------
        Path
            Full path to the .set file

        Raises
        ------
        ValueError
            If image_type is not lavision_set
        """
        if self.image_type != "lavision_set":
            raise ValueError("get_set_file_path() only valid for lavision_set image_type")
        return self.source_paths[source_path_idx]

    # --- Transform properties ---

    @property
    def transforms(self) -> dict:
        """Return full transforms configuration block."""
        return self.data.get("transforms", {})

    @property
    def transforms_cameras(self) -> dict:
        """Return per-camera transform operations.

        Returns
        -------
        dict
            Dictionary with camera numbers (int) as keys and operation lists as values.
            E.g., {1: ['flip_ud', 'rotate_90_cw'], 2: ['flip_lr']}
        """
        cameras = self.transforms.get("cameras", {})
        # Normalize keys to integers and extract operations
        return {int(k): v.get("operations", []) if isinstance(v, dict) else v
                for k, v in cameras.items()}

    def get_camera_transforms(self, camera: int) -> list:
        """Get transform operations for a specific camera.

        Parameters
        ----------
        camera : int
            Camera number (1-based)

        Returns
        -------
        list
            List of transformation operation names (already simplified)
        """
        cameras = self.transforms_cameras
        return cameras.get(camera, [])

    def set_camera_transforms(self, camera: int, operations: list):
        """Set transform operations for a specific camera.

        Parameters
        ----------
        camera : int
            Camera number (1-based)
        operations : list
            List of transformation operation names
        """
        if "transforms" not in self.data:
            self.data["transforms"] = {"cameras": {}}
        if "cameras" not in self.data["transforms"]:
            self.data["transforms"]["cameras"] = {}

        # Use integer key - YAML handles this correctly
        self.data["transforms"]["cameras"][camera] = {"operations": operations}

    def clear_camera_transforms(self, camera: int):
        """Clear all transforms for a specific camera.

        Parameters
        ----------
        camera : int
            Camera number (1-based)
        """
        if "transforms" in self.data and "cameras" in self.data["transforms"]:
            # Check for both int and string keys (backwards compatibility)
            cameras = self.data["transforms"]["cameras"]
            if camera in cameras:
                cameras[camera]["operations"] = []
            elif str(camera) in cameras:
                cameras[str(camera)]["operations"] = []

    @property
    def transforms_base_path_idx(self) -> int:
        """Return base path index for transform operations.

        Returns
        -------
        int
            Index into base_paths list (default 0)
        """
        return self.transforms.get("base_path_idx", 0)

    @property
    def transforms_type_name(self) -> str:
        """Return data type name for transform operations.

        Returns
        -------
        str
            Either 'instantaneous' or 'ensemble' (default 'instantaneous')
        """
        return self.transforms.get("type_name", "instantaneous")

    @property
    def transforms_source_endpoint(self) -> str:
        """Return source endpoint for transform operations.

        Determines what data type to transform:
        - 'instantaneous': Single-frame PIV vectors
        - 'ensemble': Ensemble-averaged vectors
        - 'merged': Multi-camera merged vectors
        - 'stereo': 3D stereo PIV vectors

        All endpoints are allowed for transforms.

        Returns
        -------
        str
            Source endpoint (default 'regular')
        """
        return self.transforms.get("source_endpoint", "regular")

    def get_camera_folder(self, camera_num: int) -> str:
        """Get the subfolder name for a specific camera.

        Container formats (.cine, .set) don't use camera subfolders:
        - .set: All cameras in one file
        - .cine: Separate files per camera in source dir (uses %d in pattern)

        IM7 depends on images_use_camera_subfolders:
        - False (default): All cameras in one .im7 file, no subfolder
        - True: Single-camera .im7 files in camera subfolders
        """
        # SET and CINE never use camera subfolders
        if self.image_type in ("lavision_set", "cine"):
            return ""

        # IM7: check images_use_camera_subfolders
        if self.image_type == "lavision_im7":
            if not self.images_use_camera_subfolders:
                return ""  # Multi-camera file, no subfolder
            # Fall through to use camera subfolders

        subfolders = self.camera_subfolders
        # camera_num is 1-based
        idx = camera_num - 1

        if subfolders and idx < len(subfolders) and subfolders[idx]:
            return subfolders[idx]

        if self.camera_count == 1:
            return ""

        return f"Cam{camera_num}"

    # ===================== ENDPOINT VALIDATION =====================

    def get_allowed_endpoints(self, tool: str) -> List[str]:
        """Get allowed source endpoints for a specific tool.

        Parameters
        ----------
        tool : str
            Tool name: 'video', 'merging', 'statistics', 'transforms'

        Returns
        -------
        List[str]
            List of allowed source endpoint names: 'regular', 'merged', 'stereo'
        """
        return TOOL_ALLOWED_SOURCE_ENDPOINTS.get(tool, [])

    def get_allowed_type_names(self, tool: str) -> List[str]:
        """Get allowed type names for a specific tool.

        Parameters
        ----------
        tool : str
            Tool name: 'video', 'merging', 'statistics', 'transforms'

        Returns
        -------
        List[str]
            List of allowed type names: 'instantaneous', 'ensemble'
        """
        return TOOL_ALLOWED_TYPE_NAMES.get(tool, [])

    def validate_endpoint_for_tool(self, tool: str, endpoint: str) -> Tuple[bool, str]:
        """Validate that a source endpoint is allowed for a tool.

        Parameters
        ----------
        tool : str
            Tool name: 'video', 'merging', 'statistics', 'transforms'
        endpoint : str
            Source endpoint to validate: 'regular', 'merged', 'stereo'

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
            If valid, error_message is empty string.
        """
        allowed = self.get_allowed_endpoints(tool)
        if endpoint not in allowed:
            return False, f"Source endpoint '{endpoint}' not allowed for {tool}. Allowed: {allowed}"
        return True, ""

    def validate_type_name_for_tool(self, tool: str, type_name: str) -> Tuple[bool, str]:
        """Validate that a type name is allowed for a tool.

        Parameters
        ----------
        tool : str
            Tool name: 'video', 'merging', 'statistics', 'transforms'
        type_name : str
            Type name to validate: 'instantaneous', 'ensemble'

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
            If valid, error_message is empty string.
        """
        allowed = self.get_allowed_type_names(tool)
        if type_name not in allowed:
            return False, f"Type name '{type_name}' not allowed for {tool}. Allowed: {allowed}"
        return True, ""
    
    @property
    def cluster_type(self):
        cluster_type = (
            self.data.get("processing", {}).get("cluster_type", "local").lower()
        )
        if cluster_type not in ["local", "slurm"]:
            raise ValueError("cluster_type must be 'local' or 'slurm'")
        return cluster_type

    @property
    def n_nodes(self):
        if self.cluster_type == "slurm":
            n_nodes = self.data.get("processing", {}).get("slurm", {}).get("nnodes", None)
            if n_nodes is None:
                raise ValueError("n_nodes must be set for Slurm cluster")
            return int(n_nodes)
        else:
            return None

    @property
    def slurm_walltime(self):
        if self.cluster_type == "slurm":
            walltime = self.data.get("processing", {}).get("slurm", {}).get("walltime", "01:00:00")
            return walltime
        else:
            return None
    @property
    def slurm_memory_limit(self):
        if self.cluster_type == "slurm":
            mem_limit = self.data.get("processing", {}).get("slurm", {}).get("memory_limit", "100GB")
            return mem_limit
        else:
            return None

    @property
    def slurm_partition(self):
        if self.cluster_type == "slurm":
            partition = self.data.get("processing", {}).get("slurm", {}).get("partition", "normal")
            return partition
        else:
            return None
    @property
    def slurm_interface(self):
        if self.cluster_type == "slurm":
            interface = self.data.get("processing", {}).get("slurm", {}).get("interface", "ib0")
            return interface
        else:
            return None
    @property
    def slurm_job_extra(self):
        if self.cluster_type == "slurm":
            job_extra = self.data.get("processing", {}).get("slurm", {}).get("job_extra", [])
            return job_extra
        else:
            return None
        
    @property
    def slurm_job_prologue(self):
        if self.cluster_type == "slurm":
            prologue = self.data.get("processing", {}).get("slurm", {}).get("prologue", [])
            return prologue
        else:
            return None
        
def get_config(refresh: bool = False) -> Config:
    """Return shared Config instance. Pass refresh=True to reload from disk."""
    global _CONFIG
    if refresh or _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def reload_config() -> Config:
    """Explicit convenience to force reload."""
    return get_config(refresh=True)
