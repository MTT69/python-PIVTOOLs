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
        """Get the path to the config file, preferring user config over package default."""
        # Determine user config directory
        if os.name == 'nt':  # Windows
            user_config_dir = Path(os.environ.get('APPDATA', '')) / 'pivtools'
        else:  # Unix-like (macOS, Linux)
            user_config_dir = Path.home() / '.config' / 'pivtools'
        
        user_config_dir.mkdir(parents=True, exist_ok=True)
        user_config_path = user_config_dir / 'config.yaml'
        
        # If user config doesn't exist, copy from package default
        if not user_config_path.exists():
            package_default = Path(__file__).parent.parent / 'pivtools_core' / 'config.yaml'
            if package_default.exists():
                shutil.copy2(package_default, user_config_path)
        
        return user_config_path

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
        if self.time_resolved:
            return self.data["images"].get("image_format", "B%05d.tiff")
        else:
            # Expect a list of two formats in the config for A and B images
            fmts = self.data["images"].get(
                "image_format", ["B%05d_A.tiff", "B%05d_B.tiff"]
            )
            return tuple(fmts)

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
        return [f"Cam{n}" for n in self.camera_numbers]

    @property
    def num_images(self):
        return self.data["images"]["num_images"]

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
        
        Returns
        -------
        tuple
            (H, W) shape of images
        """
        from pivtools_core.image_handling.load_images import read_image
        
        source_path = self.source_paths[0]
        camera_num = self.camera_numbers[0]
        image_format = self.image_format
        
        try:
            # Handle different image format cases
            if '.set' in str(image_format):
                # For .set files, they're in the source directory
                if isinstance(image_format, tuple):
                    file_path = source_path / image_format[0]
                else:
                    file_path = source_path / image_format
                img = read_image(str(file_path), camera_no=camera_num, im_no=1)
            elif '.im7' in str(image_format):
                # For .im7 files, they're in the source directory (all cameras in one file)
                if isinstance(image_format, tuple):
                    file_path = source_path / (image_format[0] % 1)
                else:
                    file_path = source_path / (image_format % 1)
                img = read_image(str(file_path), camera_no=camera_num)
            else:
                # Regular files in camera subdirectories
                camera_path = source_path / f"Cam{camera_num}"
                
                if isinstance(image_format, tuple):
                    # Non-time-resolved: use first format (A frame)
                    file_path = camera_path / (image_format[0] % 1)
                else:
                    # Time-resolved: single format
                    file_path = camera_path / (image_format % 1)
                
                img = read_image(str(file_path))
            
            # Handle both single images and image pairs
            if img.ndim == 3 and img.shape[0] == 2:
                # Image pair returned (e.g., from .im7)
                return tuple(img.shape[1:])
            else:
                # Single image
                return tuple(img.shape)
                
        except Exception as e:
            logging.error("Failed to auto-detect image shape: %s", e)
            logging.error("Please specify 'shape' in config.yaml under 'images' section")
            raise ValueError(
                f"Could not auto-detect image shape. Please add 'shape: [H, W]' "
                f"to the 'images' section of your config.yaml. Error: {e}"
            )

    @property
    def piv_chunk_size(self):
        # Updated to use batches.size from config.yaml
        return self.data["batches"]["size"]

    @property
    def batch_size(self):
        """Batch size for image processing."""
        return self.data.get("batches", {}).get("size", 30)

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

    def get_calibration_method_params(self, method: str):
        """Get parameters for a specific calibration method."""
        return self.calibration.get(method, {})

    def set_active_calibration_method(self, method: str):
        """Set the active calibration method."""
        if method in ["scale_factor", "pinhole", "stereo"]:
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


def get_config(refresh: bool = False) -> Config:
    """Return shared Config instance. Pass refresh=True to reload from disk."""
    global _CONFIG
    if refresh or _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def reload_config() -> Config:
    """Explicit convenience to force reload."""
    return get_config(refresh=True)
