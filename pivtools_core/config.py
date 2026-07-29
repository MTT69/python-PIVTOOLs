import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
        with open(path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)
        if self.data is None:
            self.data = {}
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

        # Migrate old pairing keys to new stride-based model
        self._migrate_pairing_config()

    def _migrate_pairing_config(self):
        """Migrate old pairing keys (time_resolved, zero_based_indexing, etc.)
        to new stride-based model (start_index, frame_stride, pair_stride).

        Called on every load. Only runs if old keys exist and new keys don't.
        Auto-saves the config after migration so it doesn't re-run.
        """
        images = self.data.get("images", {})

        # Only migrate if old keys exist and new keys don't
        has_old = "time_resolved" in images or "pairing_mode" in images
        has_new = "frame_stride" in images
        if not has_old or has_new:
            return

        # Read old values BEFORE removing them (direct dict access, not properties)
        old_time_resolved = images.get("time_resolved", False)
        old_zero_based = images.get("zero_based_indexing", False)

        # Determine image type from format (direct dict access, not property)
        image_type_val = images.get("image_type")
        if not image_type_val:
            fmt_raw = images.get("image_format")
            if fmt_raw:
                fmt = fmt_raw[0] if isinstance(fmt_raw, (list, tuple)) else fmt_raw
                fmt_lower = fmt.lower()
                if ".cine" in fmt_lower:
                    image_type_val = "cine"
                elif ".set" in fmt_lower:
                    image_type_val = "lavision_set"
                elif ".im7" in fmt_lower or ".ims" in fmt_lower:
                    image_type_val = "lavision_im7"
                else:
                    image_type_val = "standard"
            else:
                image_type_val = "standard"

        # Determine format count for A/B detection
        fmt_raw = images.get("image_format")
        format_count = 1
        if isinstance(fmt_raw, (list, tuple)):
            format_count = len(fmt_raw)

        # start_index: cine always 1 (reader handles FirstImageNo), others from zero_based
        if image_type_val == "cine":
            images["start_index"] = 1
        else:
            images["start_index"] = 0 if old_zero_based else 1

        # Determine strides and preset
        if format_count == 2:
            # A/B format (separate A and B files)
            images["frame_stride"] = 0
            images["pair_stride"] = 1
            images["pairing_preset"] = "ab_format"
        elif image_type_val in ("lavision_set", "lavision_im7"):
            if old_time_resolved:
                images["frame_stride"] = 1
                images["pair_stride"] = 1
                images["pairing_preset"] = "time_resolved"
            else:
                images["frame_stride"] = 0
                images["pair_stride"] = 1
                images["pairing_preset"] = "pre_paired"
        elif image_type_val == "cine":
            if old_time_resolved:
                images["frame_stride"] = 1
                images["pair_stride"] = 1
                images["pairing_preset"] = "time_resolved"
            else:
                images["frame_stride"] = 1
                images["pair_stride"] = 2
                images["pairing_preset"] = "skip_frames"
        else:
            # Standard format with single pattern
            if old_time_resolved:
                images["frame_stride"] = 1
                images["pair_stride"] = 1
                images["pairing_preset"] = "time_resolved"
            else:
                images["frame_stride"] = 1
                images["pair_stride"] = 2
                images["pairing_preset"] = "skip_frames"

        # Remove old keys
        images.pop("time_resolved", None)
        images.pop("zero_based_indexing", None)
        images.pop("pairing_mode", None)
        images.pop("pairing_skip", None)
        images.pop("num_frame_pairs", None)  # Was a dead key, now fully computed

        logging.info(
            "Migrated pairing config: start_index=%s, frame_stride=%s, "
            "pair_stride=%s, preset=%s",
            images["start_index"],
            images["frame_stride"],
            images["pair_stride"],
            images["pairing_preset"],
        )

        # Auto-save so migration doesn't re-run
        self.save()

    def save(self):
        """Save current config state to YAML file.

        Uses atomic write (write to temp file, then os.replace) to prevent
        corruption from interrupted writes or cloud sync (e.g. OneDrive).
        """
        self._normalize_calibration_block()
        tmp_path = str(self._config_path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)
        # Retry os.replace to handle transient locks from OneDrive/cloud sync
        for attempt in range(5):
            try:
                os.replace(tmp_path, self._config_path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    def save_timestamped_copy(
        self, destination_dir: Path, timestamp: str = None
    ) -> Path:
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
            with open(dest_path, "w", encoding="utf-8") as f:
                yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

        return dest_path

    # The YAML calibration block is a pointer only. Everything else (image format,
    # board geometry, dt, global coordinates, ...) lives in the per-source
    # settings sidecar — see pivtools_core.calibration_settings.
    _CALIBRATION_POINTER_KEYS = ("calibration_sources", "source", "source_idx", "active")

    def _normalize_calibration_block(self):
        """Strip the calibration block to its pointer keys on save.

        Any non-pointer key found here predates the settings sidecar and is
        stale by definition — drop it and log what was dropped at INFO.
        """
        cal = self.data.get("calibration")
        if not isinstance(cal, dict):
            return

        dropped = [k for k in cal if k not in self._CALIBRATION_POINTER_KEYS]
        if dropped:
            logging.getLogger(__name__).info(
                "Dropping stale calibration keys from config.yaml (calibration "
                "state now lives in the per-source settings sidecar): %s",
                ", ".join(sorted(dropped)),
            )

        self.data["calibration"] = {
            k: cal[k] for k in self._CALIBRATION_POINTER_KEYS if k in cal
        }

    def _get_config_path(self):
        """Path to config.yaml in the current working directory. No fallbacks."""
        cwd_config_path = Path.cwd() / "config.yaml"
        if not cwd_config_path.exists():
            raise FileNotFoundError(
                f"No config.yaml found in {Path.cwd()}. "
                "Run 'pivtools-cli init' to create a workspace config."
            )
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
        """Backward-compatible: True when frames are individual (not pre-paired).

        Derives from frame_stride: frame_stride > 0 means individual frames
        that need to be paired across files/entries.
        """
        return self.frame_stride > 0

    # ===================== STRIDE-BASED PAIRING PROPERTIES =====================

    @property
    def start_index(self):
        """Return the starting file/frame index (0 or 1)."""
        return self.data.get("images", {}).get("start_index", 1)

    @property
    def frame_stride(self):
        """Gap between frame A and frame B within one pair.

        0 = pre-paired (A/B files or container with A+B in one entry)
        1 = consecutive frames (1+2, or 3+4, etc.)
        N = custom gap (frame 1 paired with frame 1+N)
        """
        return self.data.get("images", {}).get("frame_stride", 0)

    @property
    def pair_stride(self):
        """Gap between the start of consecutive pairs.

        1 = overlapping (time-resolved: 1+2, 2+3, 3+4)
        2 = non-overlapping (skip: 1+2, 3+4, 5+6)
        N = custom gap
        """
        return self.data.get("images", {}).get("pair_stride", 1)

    @property
    def pairing_preset(self):
        """Return the pairing preset name.

        Values: 'time_resolved', 'skip_frames', 'ab_format', 'pre_paired', 'custom'
        """
        return self.data.get("images", {}).get("pairing_preset", "ab_format")

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
            return (raw,) if raw else ("B%05d_A.tiff", "B%05d_B.tiff")
        elif isinstance(raw, (list, tuple)):
            filtered = tuple(r for r in raw if r)
            if not filtered:
                # Empty list/tuple — fall back to defaults
                if self.time_resolved:
                    return ("B%05d.tiff",)
                else:
                    return ("B%05d_A.tiff", "B%05d_B.tiff")
            return filtered
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
        if ".cine" in fmt:
            return "cine"
        elif ".set" in fmt:
            return "lavision_set"
        elif ".im7" in fmt:
            return "lavision_im7"
        elif ".ims" in fmt:
            return "lavision_im7"  # .ims treated same as .im7
        else:
            return "standard"

    @property
    def is_container_format(self) -> bool:
        """Return True if format stores multiple frames in single container.

        Container formats (one file holds many timestamps):
        - cine: Single-camera video container (one file per camera)
        - lavision_set: Multi-camera container (all cameras in one file)

        NOT containers (one file per timestamp):
        - lavision_im7 with % pattern: Each IM7 file is one timestamp,
          even though it may contain multiple cameras within that timestamp.
          We count IM7 files to get the number of frame pairs.
        """
        image_type = self.image_type
        image_format = self.image_format

        # IM7 files with % pattern are individual numbered files (one per timestamp)
        if image_type == "lavision_im7" and "%" in image_format:
            return False

        # Only .set and .cine are true multi-timestamp containers
        return image_type in ("cine", "lavision_set")

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
        return self.data.get("paths", {}).get("camera_count", 1)

    @property
    def camera_numbers(self):
        """Return list of camera numbers to process.

        Supports a single-camera override via the PIV_CAMERA environment
        variable. Invalid values fail loudly (no silent fallback): a
        non-integer raises ValueError, and out-of-range is caught by the
        existing range check below.

        WARNING: the override is PROCESS-GLOBAL. It must only ever be set in
        short-lived CLI/subprocess contexts (the instantaneous/ensemble CLI
        handlers set it, then the process exits; PIV jobs run as subprocesses).
        Do NOT set PIV_CAMERA in the long-lived GUI Flask process — every
        caller of this property (statistics, video, merging, transforms, ...)
        would silently flip to single-camera mode for the rest of the process.
        """
        max_allowed = self.camera_count
        env_cam = os.environ.get("PIV_CAMERA")
        if env_cam:
            numbers = [int(env_cam)]  # non-int raises ValueError — intentional
        else:
            numbers = self.data["paths"]["camera_numbers"]
        if any(n > max_allowed or n < 1 for n in numbers):
            raise ValueError(
                f"Camera numbers {numbers} must be between 1 and {max_allowed}"
            )
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
        return self.data.get("images", {}).get("num_images", 100)

    @property
    def num_loops(self) -> int:
        """Number of acquisition loops (separate source files/folders).

        When > 1, multiple sources are combined into one larger dataset.
        Works with any image type:
        - lavision_set: separate .set files (e.g., loop=0.set, loop=1.set)
        - cine: separate folders each containing .cine files
          (e.g., experiment_0/, experiment_1/)
        - standard/im7: separate folders each containing image files

        The loop pattern is detected from the last number in the
        source_path name (file or directory).
        """
        return self.data.get("images", {}).get("num_loops", 1)

    @property
    def per_loop_frame_pairs(self) -> int:
        """Frame pairs within a single loop (original stride calculation).

        This is the number of pairs from one source (file or folder) before
        considering multiple loops. The total across all loops is num_frame_pairs.
        """
        num_images = self.num_images
        fs = self.frame_stride
        ps = self.pair_stride

        if fs == 0:
            return num_images

        if ps <= 0:
            ps = 1

        return max(0, (num_images - 1 - fs) // ps + 1)

    @property
    def num_frame_pairs(self):
        """Total frame pairs across all loops.

        Uses the unified stride formula per loop, multiplied by num_loops:
        - frame_stride == 0 (pre-paired/A-B): num_images pairs per loop
        - frame_stride > 0: (num_images - 1 - frame_stride) // pair_stride + 1 per loop

        Examples (single loop):
            time_resolved (fs=1, ps=1): 100 images → 99 pairs
            skip_frames   (fs=1, ps=2): 100 images → 50 pairs
            ab_format     (fs=0, ps=1): 100 images → 100 pairs
            pre_paired    (fs=0, ps=1): 100 images → 100 pairs

        With num_loops=3 and 40 per-loop pairs: returns 120.

        Returns
        -------
        int
            Number of frame pairs (always >= 0)
        """
        return self.num_loops * self.per_loop_frame_pairs

    def get_loop_source_path(self, source_path: Path, loop_idx: int) -> Path:
        """Get the source file/directory path for a specific loop index.

        Infers the loop pattern from the LAST number in the source_path name.
        Works with both files (.set) and directories (.cine, standard):
        - source_path=.../loop=0.set, loop_idx=3 -> .../loop=3.set
        - source_path=.../experiment_0, loop_idx=2 -> .../experiment_2
        - source_path=.../30deg_250hz_1000dt_0, loop_idx=1 -> .../30deg_250hz_1000dt_1

        Uses the last number to avoid matching numbers embedded in the
        experiment description (e.g., "30degree" or "250hz").

        Args:
            source_path: Path to the base source (loop 0) — file or directory
            loop_idx: Zero-based loop index

        Returns:
            Path to the source for the given loop

        Raises:
            ValueError: If no number can be detected in the name
        """
        if self.num_loops <= 1:
            return source_path

        source_path = Path(source_path)
        name = source_path.name

        # Find the LAST number in the name (avoids matching "30degree", "250hz", etc.)
        matches = list(re.finditer(r"(\d+)", name))
        if not matches:
            raise ValueError(f"Cannot detect loop number in: {name}")

        match = matches[-1]
        base_number = int(match.group(1))
        new_number = base_number + loop_idx

        # Preserve zero-padding width (e.g., "001" stays 3 digits)
        num_width = match.end(1) - match.start(1)
        if num_width > 1:
            new_number_str = str(new_number).zfill(num_width)
        else:
            new_number_str = str(new_number)

        new_name = name[: match.start(1)] + new_number_str + name[match.end(1) :]
        return source_path.parent / new_name

    def resolve_loop_for_pair(self, global_pair_1based: int) -> tuple:
        """Resolve a global pair number to (loop_idx, local_pair_1based).

        E.g., with 40 per-loop pairs:
          pair 1  -> (0, 1)
          pair 40 -> (0, 40)
          pair 41 -> (1, 1)
          pair 80 -> (1, 40)

        Args:
            global_pair_1based: 1-based global pair number

        Returns:
            Tuple of (loop_idx, local_pair_1based)
        """
        per_loop = self.per_loop_frame_pairs
        loop_idx = (global_pair_1based - 1) // per_loop
        local_pair = (global_pair_1based - 1) % per_loop + 1
        return (loop_idx, local_pair)

    @property
    def pairing_mode(self):
        """Backward-compatible: derives from pairing_preset."""
        preset = self.pairing_preset
        if preset in ("time_resolved", "ab_format", "pre_paired"):
            return "sequential"
        return "skip"

    def get_frame_pair_indices(self, pair_number: int) -> tuple:
        """Get the file/frame indices for a given pair number.

        Uses the unified stride formula:
            pair_start = start_index + (pair_number - 1) * pair_stride
            frame_a = pair_start
            frame_b = pair_start + frame_stride  (or pair_start if frame_stride == 0)

        Args:
            pair_number: 1-based pair number (pair 1, pair 2, etc.)

        Returns:
            tuple: (frame_a_idx, frame_b_idx) for the reader to use

        Examples:
            ab_format (si=1, fs=0, ps=1):
                pair 1 → (1, 1), pair 2 → (2, 2)
            time_resolved (si=1, fs=1, ps=1):
                pair 1 → (1, 2), pair 2 → (2, 3)
            skip_frames (si=1, fs=1, ps=2):
                pair 1 → (1, 2), pair 2 → (3, 4)
            pre_paired (si=1, fs=0, ps=1):
                pair 1 → (1, 1), pair 2 → (2, 2)
            zero-based time_resolved (si=0, fs=1, ps=1):
                pair 1 → (0, 1), pair 2 → (1, 2)
        """
        si = self.start_index
        fs = self.frame_stride
        ps = self.pair_stride

        pair_start = si + (pair_number - 1) * ps

        if fs == 0:
            return (pair_start, pair_start)
        else:
            return (pair_start, pair_start + fs)

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

        logging.debug(
            f"_detect_image_shape: image_format = {image_format}, image_type = {img_type}"
        )
        logging.debug(f"Source path: {source_path}, Camera: {camera_num}")

        format_str = image_format[0]  # Always tuple now

        # Determine camera_path based on image type (same as load_images)
        if img_type in ("lavision_set", "lavision_im7", "cine"):
            camera_path = source_path  # Container formats: no camera subdir
        else:
            folder = self.get_camera_folder(camera_num)
            camera_path = source_path / folder if folder else source_path

        logging.debug(f"Camera path: {camera_path}")

        # Determine start index
        start_idx = self.start_index

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

        logging.debug(f"Trying to read file: {file_path}")

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

            logging.debug(f"Detected image shape: {shape}")
            return shape

        except Exception as e:
            logging.error("Failed to read image: %s", e)
            raise ValueError(
                f"Could not read image file {file_path}. Error: {e}. "
                "Check that the file exists and is a valid image format."
            )

    @property
    def batch_size(self):
        """
        Batch size for image processing.

        Automatically capped at per_loop_frame_pairs to prevent batches from
        crossing loop boundaries (each batch reads from a single .set file).
        """
        configured_size = self.data.get("batches", {}).get("size", 10)
        max_size = self.per_loop_frame_pairs

        # Cap batch size at per-loop frame pairs (batches must not cross loop boundaries)
        actual_size = min(configured_size, max_size)

        if actual_size < configured_size:
            logging.debug(
                f"Batch size capped at {actual_size} (configured: {configured_size}, "
                f"max allowed: {max_size} per-loop frame pairs)"
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
    def gain_normalisation(self) -> bool:
        """Two-pass per-frame laser-gain normalisation (preprocessing.gain_normalisation).

        When True, an eager pre-pass computes ensemble mean images <A>, <B>
        per camera and a per-frame gain by least-squares regression against
        them over unmasked pixels:

            g_i = sum(I_i * ref) / sum(ref**2)

        (gain-only, no offset — the model validated on Cam4 2026-07-27).
        Each frame is then divided by its scalar gain at the head of the
        filter pipeline, before pixel-mask zeroing and all configured
        filters, in both the ensemble and instantaneous pipelines. Removes
        per-frame laser energy drift and A/B pulse-energy imbalance at
        source. Costs two extra full reads of the dataset per camera per
        run. Default: False.
        """
        return self.data.get("preprocessing", {}).get("gain_normalisation", False)

    @property
    def vector_format(self):
        # Returns a single format string like "%05d.mat".
        # Guard the read side: a stored value lacking a printf integer specifier
        # (e.g. ['']) makes every `vector_format % i` raise "not all arguments
        # converted". A concurrent GUI save with stale state has corrupted this
        # key before, so a YAML already on disk may hold ['']. Fall back to the
        # default rather than crash, but log so the corruption stays visible.
        # Default matches the CLI template and frontend fallback (gotcha #1).
        default = "%05d.mat"
        vf = self.data["images"].get("vector_format", [default])
        first = vf[0] if isinstance(vf, (list, tuple)) and vf else vf
        if isinstance(first, str) and re.search(r"%[0-9]*d", first):
            return first
        logging.getLogger(__name__).warning(
            "Invalid vector_format %r in config (no printf integer specifier); "
            "falling back to %r",
            vf,
            default,
        )
        return default

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
            "mean_stresses": True,
            "mean_tke": True,
            "mean_vorticity": True,
            "mean_divergence": True,
            "mean_peak_height": False,
            "correlation_quality": False,
            # Instantaneous (per-frame) statistics
            "inst_velocity": True,
            "inst_stresses": True,
            "inst_vorticity": True,
            "inst_divergence": True,
            "inst_gamma": True,
        }
        methods = self.statistics.get("enabled_methods", default_methods)

        # Migrate legacy key names to canonical names
        legacy_map = {
            "reynolds_stress": "mean_stresses",
            "normal_stress": "mean_stresses",
            "inst_fluctuations": "inst_stresses",
        }
        for old_key, new_key in legacy_map.items():
            if old_key in methods:
                # Carry over enabled state (True wins if either legacy key was True)
                if methods[old_key] and new_key not in methods:
                    methods[new_key] = True
                elif methods[old_key] and new_key in methods:
                    methods[new_key] = methods[new_key] or methods[old_key]
                del methods[old_key]

        # Ensure all canonical keys are present with defaults
        for key, default_val in default_methods.items():
            if key not in methods:
                methods[key] = default_val

        return methods

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
        return self.video.get("cmap", "viridis")

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
    def video_resolution(self):
        """Return video resolution as (height, width) tuple, or None for native."""
        res = self.video.get("resolution", "1080p")
        if isinstance(res, str):
            if res == "native":
                return None
            if res == "4k":
                return (2160, 3840)
            return (1080, 1920)
        elif isinstance(res, (list, tuple)) and len(res) >= 2:
            return (res[0], res[1])
        return (1080, 1920)

    @property
    def video_resolution_str(self) -> str:
        """Return resolution as string: 'native', '1080p', or '4k'."""
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

    # --- Calibration pointer ---
    # The YAML holds only the pointer to calibration data (sources, active
    # method). Image format, board geometry, dt, etc. live in the per-source
    # settings sidecar (pivtools_core.calibration_settings.load_settings).

    @property
    def calibration_sources(self) -> list:
        """Return list of calibration source paths (REQUIRED for calibration).

        These are direct paths to calibration image locations:
        - For container formats (.set, .cine): path to the container file
        - For standard formats (.tiff, etc.): path to directory containing images

        Camera subfolders (Cam1/, Cam2/) are applied relative to these paths
        when use_camera_subfolders is True.
        """
        calib_block = self.data.get("calibration", {}) or {}
        sources = calib_block.get("calibration_sources", [])
        return [Path(s) for s in sources if s is not None] if sources else []

    def get_calibration_source(self, source_path_idx: int = 0) -> Path:
        """Get calibration source path for the given index.

        Parameters
        ----------
        source_path_idx : int
            Index into calibration_sources list (default: 0)

        Returns
        -------
        Path
            Calibration source path

        Raises
        ------
        ValueError
            If calibration_sources is not configured
        IndexError
            If source_path_idx is out of range
        """
        sources = self.calibration_sources
        if not sources:
            raise ValueError(
                "calibration.calibration_sources not configured in config.yaml. "
                "Please specify direct paths to calibration images."
            )
        if source_path_idx >= len(sources):
            raise IndexError(
                f"calibration_sources index {source_path_idx} out of range "
                f"(have {len(sources)} sources)"
            )
        return sources[source_path_idx]

    @property
    def calibration(self):
        """Return the full calibration block (dict) from config."""
        return self.data.get("calibration", {})

    @property
    def active_calibration_method(self):
        """Return the active calibration method name (e.g., 'dotboard', 'scale_factor')."""
        cal = self.calibration
        return cal.get("active", "scale_factor")

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
    def window_type(self):
        """Return PIV window type (e.g., 'gaussian', 'A')."""
        return self.data.get("instantaneous_piv", {}).get("window_type", "gaussian")

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
        Auto-detect the Dask per-worker memory limit from system resources.
        Results are cached to avoid repeated detection.

        Scope note (2026-07-27): this used to derive ``dask_workers_per_node``
        from ``os.cpu_count()`` and pin ``omp_threads`` to 2. It no longer
        touches either — both now always come from ``processing:`` in the
        config. The thread budget a run spends is ``dask_workers_per_node *
        omp_threads``, and deriving one factor from the core count while the
        other stayed configured made that product unpredictable: on a 20-core
        machine it silently became 20 * 2 = 40 threads. Those two knobs are a
        tuning decision, not something to infer. Memory per worker genuinely
        does depend on the machine, so it stays here.

        Returns
        -------
        dict
            Dictionary with key: dask_memory_limit
        """
        # Return cached result if available
        if self._auto_compute_cache is not None:
            return self._auto_compute_cache

        import psutil

        # Get total system memory in GB
        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Divide by the CONFIGURED worker count — these processes are what
        # actually hold memory, and it is no longer cpu_count.
        workers = max(1, int(self.dask_workers_per_node))

        # Dask memory = (total memory - 10%) / workers
        # Reserve 10% for system overhead
        available_memory_gb = total_memory_gb * 0.9
        memory_per_worker_gb = available_memory_gb / workers
        dask_memory_limit = f"{memory_per_worker_gb:.2f}GB"

        logging.warning(
            "auto_compute_params now sizes dask_memory_limit ONLY (changed "
            "2026-07-27). dask_workers_per_node and omp_threads come from "
            "processing: — if this config relied on them being auto-detected, "
            "set them explicitly, or the run uses the defaults (%d worker(s), "
            "%s threads).",
            workers,
            self.omp_threads,
        )
        logging.info("Auto-detected compute parameters:")
        logging.info("  Total memory: %.2f GB", total_memory_gb)
        logging.info("  Workers per node (from config): %d", workers)
        logging.info("  Memory per worker: %s", dask_memory_limit)

        # Cache the result
        self._auto_compute_cache = {
            "dask_memory_limit": dask_memory_limit,
        }

        return self._auto_compute_cache

    @property
    def omp_threads(self):
        """Return number of OMP threads as string.

        Always from ``processing.omp_threads`` — see ``_get_auto_compute_params``
        for why ``auto_compute_params`` no longer overrides this.
        """
        return str(self.data.get("processing", {}).get("omp_threads", 4))

    @property
    def dask_workers_per_node(self):
        """Return number of Dask workers per node.

        Always from ``processing.dask_workers_per_node`` — see
        ``_get_auto_compute_params``.
        """
        return self.data.get("processing", {}).get("dask_workers_per_node", 1)

    @property
    def dask_memory_limit(self):
        """Return memory limit per Dask worker."""
        if self.auto_compute_params:
            return self._get_auto_compute_params()["dask_memory_limit"]
        return self.data.get("processing", {}).get("dask_memory_limit", "12GB")

    @property
    def dask_max_in_flight_per_worker(self):
        """Max concurrent tasks queued per Dask worker in sliding window.

        Higher values improve pipelining (I/O overlaps with CPU work).
        Default 3: while one batch correlates (CPU), the next loads (I/O),
        with slack for uneven completion.

        On HPC with fast storage, users can increase to 4-6.
        """
        return self.data.get("processing", {}).get("dask_max_in_flight_per_worker", 3)

    @property
    def post_processing_workers(self):
        """Max parallel workers for post-processing (merge, calibrate, statistics, transforms).

        Default: min(cpu_count, 16). Set lower on shared HPC filesystems where
        disk bandwidth saturates before all cores are busy with I/O.
        """
        val = self.data.get("processing", {}).get("post_processing_workers")
        if val is not None:
            return int(val)
        return min(os.cpu_count() or 4, 16)

    @property
    def open_dashboard(self):
        """Whether to automatically open the Dask dashboard in a browser tab."""
        return self.data.get("processing", {}).get("open_dashboard", False)

    @property
    def dask_nanny(self):
        """Whether to use nanny processes for Dask workers.

        When True, each worker is managed by a nanny process that monitors
        memory usage and captures exit codes/reasons when workers crash.
        Useful for diagnosing worker failures on HPC clusters.

        Default: True.
        """
        return self.data.get("processing", {}).get("dask_nanny", False)

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
    def peak_finder(self):
        """Return peak finder method (converted to numeric code)."""
        peak_finder = (
            self.data.get("instantaneous_piv", {}).get("peak_finder", "gauss6").lower()
        )
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
    def peak_fit_impl(self):
        """Peak-fit implementation: 'batch' (default) or 'scalar'.

        'batch' is the lockstep one-window-per-SIMD-lane LM fitter
        (peak_locate_lm_batch.c) for instantaneous gauss4/5/6 fits — 2-3x
        faster per fit, gate-verified against the scalar oracle (bit-identical
        under the libm-exp reference build). Requires a clang/clang-cl build;
        on a build without the batch fitter (plain MSVC cl) correlator init
        raises with instructions to set 'scalar' (no silent fallback). gauss3
        and multi-peak fits always use the scalar path regardless.
        """
        v = self.data.get("instantaneous_piv", {}).get("peak_fit_impl", "batch").lower()
        if v not in ("scalar", "batch"):
            raise ValueError(
                f"Invalid peak_fit_impl: {v}. Must be 'scalar' or 'batch'."
            )
        return v

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
        """Return ensemble PIV window type (e.g., 'square', 'gaussian')."""
        return self.data.get("ensemble_piv", {}).get("window_type", "square")

    @property
    def ensemble_num_peaks(self):
        """Return number of peaks for ensemble PIV."""
        return self.data.get("ensemble_piv", {}).get("num_peaks", self.num_peaks)

    @property
    def ensemble_peak_finder(self):
        """Return peak finder method for ensemble PIV (converted to numeric code)."""
        peak_finder = (
            self.data.get("ensemble_piv", {}).get("peak_finder", "gauss6").lower()
        )
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
        sum_window = self.data.get("ensemble_piv", {}).get("sum_window", [32, 32])

        # Validate sum_window if single mode is used
        ensemble_types = self.ensemble_type
        if "single" in ensemble_types:
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
                if pass_type == "single":
                    win_size = self.ensemble_window_sizes[pass_idx]
                    if sum_window[0] < win_size[0] or sum_window[1] < win_size[1]:
                        raise ValueError(
                            f"Pass {pass_idx}: ensemble_sum_window {sum_window} must be >= "
                            f"window_size {win_size} for single mode"
                        )

        return sum_window

    @property
    def ensemble_sum_fitting_window_enabled(self):
        """
        Return whether sum_fitting_window extraction is enabled.

        Returns
        -------
        bool
            True if extraction is enabled, False otherwise (default)
        """
        return self.data.get("ensemble_piv", {}).get("sum_fitting_window_enabled", True)

    @property
    def ensemble_sum_fitting_window(self):
        """
        Return fitting window size for ensemble correlation planes.

        Only used when sum_fitting_window_enabled is True.
        Correlations are computed on full sum_window but only the central
        sum_fitting_window region is extracted for storage and fitting.

        Returns
        -------
        list or None
            [height, width] of fitting window, or None if disabled.
            Defaults to ensemble_sum_window when unset (no extraction shrink).
        """
        # Check if feature is enabled
        if not self.ensemble_sum_fitting_window_enabled:
            return None

        fit_window = self.data.get("ensemble_piv", {}).get("sum_fitting_window", None)

        if fit_window is None:
            fit_window = self.ensemble_sum_window

        # Validate: must be smaller than or equal to sum_window
        sum_window = self.ensemble_sum_window
        if fit_window[0] > sum_window[0] or fit_window[1] > sum_window[1]:
            raise ValueError(
                f"sum_fitting_window {fit_window} must be <= sum_window {sum_window}"
            )

        # Validate: must be positive and even (for symmetric extraction)
        if fit_window[0] <= 0 or fit_window[1] <= 0:
            raise ValueError(f"sum_fitting_window must be positive, got {fit_window}")

        return fit_window

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
        valid_types = {"std", "standard", "single"}
        for pass_idx, pass_type in enumerate(types):
            if pass_type not in valid_types:
                raise ValueError(
                    f"Pass {pass_idx}: Invalid ensemble_type '{pass_type}'. "
                    f"Must be one of {valid_types}"
                )

        # Normalize 'standard' to 'std' for consistency
        types = ["std" if t == "standard" else t for t in types]

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
    def ensemble_fit_method(self) -> str:
        """Return fitting method for ensemble PIV.

        - 'kspace' (default, and the only production method): the batched-LM
          one-stage joint fit of the raw k-space transfer ratio
          (``fit_windows_kspace_lm``: mu, Sigma, gain g, in-model noise floor
          N0; 7 parameters for the default Gaussian shape). ``kspace_shape``
          (see ``ensemble_kspace_shape``) optionally adds per-axis quartic
          kurtosis terms. No other tuning knobs — the former
          ``lm_soft_weighting`` and ``lm_k_max_cap`` ablation knobs guarded
          pathologies of the removed two-stage design; stale keys in old
          workspace configs are ignored.

        A stale ``fit_method: gaussian`` or ``kspace_linear`` fails loudly
        here rather than silently falling back — 'kspace_linear' (the
        closed-form fitter in ``kspace_linear_fitting.py``) was retired from
        production selection on 2026-07-21; the module and its unit tests
        remain as a reference implementation. The former ``kspace_kurtosis``
        toggle was removed with the two-stage/GSL designs (kurtosis rejected
        in THOSE implementations; the 2026-07-17 bake-off validated quartic
        terms in the joint LM, now available via ``kspace_shape``); a stale
        key in old workspace configs is ignored.
        """
        method = self.data.get("ensemble_piv", {}).get("fit_method", "kspace")
        valid_methods = {"kspace"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_fit_method '{method}'. Must be one of "
                f"{valid_methods}. (The 'gaussian' method, the GSL-based "
                f"k-space fitter, and the 'kspace_linear' closed-form fitter "
                f"were removed from production selection.)"
            )
        return method

    @property
    def ensemble_kspace_shape(self) -> str:
        """Displacement-PDF shape model for the 'kspace' LM ensemble fitter.

        The fitter reads the Reynolds stresses from the Gaussian decay of the
        k-space transfer ratio. When the flow's displacement PDF is
        non-Gaussian (excess kurtosis gamma_2 != 0), ln|T| is curved, not
        parabolic, and forcing a Gaussian through it biases Sigma by
        approximately gamma_2 * band_depth / 6 (measured +15-20 % streamwise
        below y+ 400 on the channel benchmarks). Each quartic term adds one
        free-signed coefficient b4 per axis to the exponent
        (-2 pi^2 k^T Sigma k - b4x*kx^4 - b4y*ky^4), the next cumulant of
        the displacement PDF on that axis (kappa_4 = -24*b4/(2 pi)^4,
        gamma_2 = kappa_4/Sigma^2). The term self-extinguishes (b4 -> 0)
        where the data has no curvature, so no gating is applied; fitted b4
        values are reported in the fit_diagnostics sidecar.

        Options:
        - 'gaussian' (default): the historical 7-parameter model, unchanged.
        - 'kx4': + b4x*kx^4 (streamwise kurtosis), 8 parameters.
        - 'ky4': + b4y*ky^4 (transverse/wall-normal kurtosis), 8 parameters.
        - 'kx4+ky4': both terms, 9 parameters — orientation-agnostic; no
          cross kx^2*ky^2 term (tested and rejected).

        Only meaningful for ``fit_method: kspace`` (validation rejects a
        non-gaussian shape with any other fitter rather than silently
        ignoring it). Selection evidence:
        wiki/sessions/2026-07-17-fitter-bakeoff-e-debug.md.
        """
        shape = self.data.get("ensemble_piv", {}).get("kspace_shape", "gaussian")
        valid_shapes = {"gaussian", "kx4", "ky4", "kx4+ky4"}
        if shape not in valid_shapes:
            raise ValueError(
                f"Invalid ensemble_kspace_shape '{shape}'. Must be one of "
                f"{valid_shapes}."
            )
        return shape

    @property
    def ensemble_kspace_floor(self) -> str:
        """Noise-floor model for the 'kspace' LM ensemble fitter.

        White sensor noise does not reach the transfer ratio flat: the
        image-deformation pipeline colours it (warp-kernel fractional-shift
        filtering x per-pair window-mean removal x envelope divide). The
        coloured floor fits att = 1 - N0 * P(k;fx,fy)/F_ref with P the
        ANALYTIC pipeline colour (kspace_floor_psd, no simulation), evaluated
        per window at (fx, fy) = frac(predictor/2); N0 stays the free
        per-window level. The flat assumption inflated noisy vv+ 1.37x and
        centre uu+ 1.77x of DNS (ky ring-1 floor +30-65 % at all f); the
        coloured floor lands both on DNS with the clean control unchanged.

        Options:
        - 'coloured' (default): att = 1 - N0*P(k;fx,fy)/F_ref, N0 in [0, 10],
          tail-seeded. Correct for every background method (the window-mean
          DC hole is included exactly when the bg method contains
          'window_mean') and both correlation chains (std and single).
        - 'flat': the pre-2026-07-24 model att = 1 - N0/F_ref on a
          byte-identical code path — for reproducing older runs only.

        Derivation + validation:
        wiki/sessions/2026-07-22-noise-floor-colouring-psim.md.
        """
        floor = self.data.get("ensemble_piv", {}).get("kspace_floor", "coloured")
        valid_floors = {"coloured", "flat"}
        if floor not in valid_floors:
            raise ValueError(
                f"Invalid ensemble_kspace_floor '{floor}'. Must be one of "
                f"{valid_floors}."
            )
        return floor

    @property
    def ensemble_image_warp_interpolation(self) -> str:
        """Return interpolation method for image warping in ensemble PIV.

        This controls the fused warp kernel interpolation when warping images
        based on the predictor field from the previous pass. The choice of
        interpolation may affect:
        - Particle image sharpness (PSF)
        - Measured Reynolds stress (peak width)
        - Processing speed

        Options:
        - 'cubic': Bicubic (Keys a=-0.75, 4x4 stencil, matches cv2.INTER_CUBIC). Default.
        - 'lanczos': Lanczos-3 (windowed sinc, 6x6 stencil). Sharper frequency
          preservation for particle images, ~1.5x slower than cubic.

        Default: 'cubic'
        """
        method = self.data.get("ensemble_piv", {}).get(
            "image_warp_interpolation", "cubic"
        )
        valid_methods = {"cubic", "lanczos"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_image_warp_interpolation '{method}'. "
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
        return self.data.get("ensemble_piv", {}).get(
            "skip_background_subtraction", False
        )

    @property
    def ensemble_background_subtraction_method(self) -> str:
        """Background subtraction method for ensemble PIV.

        Options:
        - 'correlation': R = <A⊗B> - <A>⊗<B> (current default, single-pass)
          Correlates raw images, then subtracts correlation of mean images.
          More memory efficient (single pass through data).

        - 'image': R = <(A-Ā)⊗(B-B̄)> (two-phase, subtract mean images first)
          Phase A computes the ensemble-mean images over all pairs, phase B
          correlates the mean-subtracted images. Requires two sweeps through
          the data.

        - 'window_mean': each interrogation window has its own weighted mean
          subtracted per image pair inside the C correlator, before weighting
          and FFT (Westerweel's per-window mean2 removal). Removes the pedestal
          at source, per pair — robust to pair-to-pair brightness fluctuation
          that the ensemble-level methods cannot remove. No <A>⊗<B> term is
          subtracted on top (it would over-subtract).

        - 'correlation+window_mean': both — per-pair window-mean removal in
          the C correlator AND subtraction of the mean-image correlation at
          finalize. The mean images get the same window-mean treatment
          (bMeanSubtract=1), so the subtracted term is exactly the stationary
          residual left in the per-pair sums (the window-mean operator is
          linear: mean(A_i − m(A_i)) = Ā − m(Ā)) — not over-subtraction.
          Removes stationary structure (reflections/glare) that survives the
          per-pair window mean, while keeping the per-pair pedestal removal.

        - 'image+window_mean': both, two-sweep — per pair, subtract the mean
          images (kills stationary structure) then window-mean removal in C
          (kills per-pair DC scatter). No finalize-time subtraction.

        - 'correlation+dc_zero' / 'image+dc_zero': the ensemble-level method
          plus a finalize-time DC-zero — after background subtraction, each
          window's correlation-plane mean is subtracted (exactly zeroing the
          plane's DC bin) BEFORE the envelope divide. This neutralises the
          per-pair DC scatter and any flicker residual parked in the DC bin
          without touching per-pair data: for std passes it is equivalent to
          'window_mean' at the fitter (the per-pair window mean only ever
          zeroes the DC bin there), but with no per-pair machinery and, in
          single mode, no A-mask aperture smear. Intended for clean or
          gain-normalised data (2026-07-27 bg-method investigation).

        'correlation' and 'image' are mathematically equivalent for stationary
        brightness but differ under drift; 'window_mean' is the per-pair
        method; the '+window_mean' variants combine an ensemble-level method
        with the per-pair one (both exact); the '+dc_zero' variants combine an
        ensemble-level method with the finalize-time DC-bin zero.

        Default: 'correlation'
        """
        method = self.data.get("ensemble_piv", {}).get(
            "background_subtraction_method", "correlation"
        )
        valid_methods = {
            "correlation",
            "image",
            "window_mean",
            "correlation+window_mean",
            "image+window_mean",
            "correlation+dc_zero",
            "image+dc_zero",
        }
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_background_subtraction_method '{method}'. "
                f"Must be one of {valid_methods}"
            )
        return method

    @property
    def ensemble_window_mean_in_correlator(self) -> bool:
        """True when the per-pair window-mean removal runs in the C correlator.

        Drives the bMeanSubtract flag of bulkxcorr2d_accumulate_triple for the
        per-pair sweeps: 'window_mean' and both '+window_mean' combined modes.
        """
        return self.ensemble_background_subtraction_method in {
            "window_mean",
            "correlation+window_mean",
            "image+window_mean",
        }

    @property
    def ensemble_bg_base_method(self) -> str:
        """Ensemble-level part of the background method: the part before '+'.

        'correlation' / 'image' / 'window_mean'. Drives the sweep-count
        dispatch (two sweeps for base 'image') and the finalize-time
        subtraction branch (mean-image correlation for base 'correlation').
        """
        return self.ensemble_background_subtraction_method.split("+", 1)[0]

    @property
    def ensemble_dc_zero(self) -> bool:
        """True when the finalize-time DC-zero runs ('+dc_zero' variants).

        Drives the per-window plane-mean subtraction applied to the ensemble
        correlation planes after background subtraction and before the
        envelope divide in SinglePassAccumulator.finalize_pass. The ordering
        is load-bearing: the envelope divide spreads any DC-bin anomaly into
        the low-k fit bins, so the DC bin must be zeroed first.
        """
        return "dc_zero" in self.ensemble_background_subtraction_method.split("+")

    @property
    def ensemble_per_pair_normalization(self) -> bool:
        """Normalize each image pair's correlation planes before accumulation.

        When True, the C correlator scales every pair's AA plane by 1/e_AA,
        BB by 1/e_BB and AB by 1/sqrt(e_AA*e_BB), where e = the zero-lag
        weighted window energy of that pair. Every pair then enters the
        ensemble with unit auto peaks — equal weight to the stress regardless
        of brightness, seeding or contrast (correlation-coefficient planes).
        The geometric-mean AB scaling keeps T = F_AB/sqrt(F_AA*F_BB) invariant.

        Requires background_subtraction_method exactly 'window_mean'
        (validated in validate_ensemble_config): the energies must be
        fluctuation energies, and any ensemble-level background term
        ('correlation'/'image', plain or '+window_mean') is built from raw
        full-frame mean images — inconsistent with per-pair-normalized sums.

        Default: False
        """
        return self.data.get("ensemble_piv", {}).get("per_pair_normalization", False)

    @property
    def ensemble_envelope_divide(self) -> bool:
        """Divide the ensemble correlation planes by the pair-count envelope.

        Applies to all three planes (AA, BB, AB) in both 'std' and 'single'
        pass types, at finalize time in SinglePassAccumulator.finalize_pass.
        When False (default) the planes keep the linear pair-count tent
        R(lag)*E(lag) — Westerweel-faithful, no plane-level 1/E division.
        When True the pre-2026-07-27 behaviour is restored: AA/BB divided by
        their auto envelope, AB by its true (possibly asymmetric) envelope.

        The coloured noise floor follows this toggle automatically: when the
        divide is off, build_P_grid receives a unit envelope so the analytic
        floor describes undivided planes (its internal 1/env_auto becomes a
        no-op). The plane sidecar records the state in 'env_divided'.

        Rationale (2026-07-27, clean synthetic): in single mode the divide
        spreads the autos' off-trend DC bin into the low-k fit bins
        (ring-1 T hole), which the LM fit levers into a large stress
        under-read; leaving the planes undivided removes the hole with only
        a small common-mode tent attenuation in T.

        Default: False
        """
        return self.data.get("ensemble_piv", {}).get("envelope_divide", False)

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
        return self.data.get("infilling", {}).get(
            "mid_pass", {"method": "nearest", "parameters": {}}
        )

    @property
    def infilling_final_pass(self):
        """Return final-pass infilling configuration."""
        return self.data.get("infilling", {}).get(
            "final_pass", {"enabled": True, "method": "biharmonic", "parameters": {}}
        )

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
        return self.data.get("ensemble_infilling", {}).get(
            "mid_pass", {"method": "nearest", "parameters": {}}
        )

    @property
    def ensemble_infilling_final_pass(self) -> dict:
        """Return ensemble final-pass infilling configuration."""
        return self.data.get("ensemble_infilling", {}).get(
            "final_pass", {"enabled": True, "method": "biharmonic", "parameters": {}}
        )

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
    def ensemble_persist_images(self) -> bool:
        """Persist all filtered images in worker RAM for ensemble multi-pass.

        When True (HPC with lots of RAM):
        - After filtering, all images are persisted in worker memory
        - Each pass reads from RAM instead of re-loading from disk
        - Avoids redundant I/O + filter computation per pass

        When False (default, desktop / memory-constrained):
        - Sliding window re-computes filters on demand per pass
        - Memory bounded to max_in_flight batches at a time
        """
        return self.data.get("ensemble_piv", {}).get("persist_images", False)

    @property
    def ensemble_predictor_smoothing(self) -> bool:
        """Enable Gaussian smoothing of the predictor field between passes.

        When True (default), the predictor field is Gaussian-smoothed before
        upscaling to the next pass grid. This reduces noise in instantaneous
        pair processing but destroys real velocity gradients near walls.

        For ensemble PIV (converged over many pairs), noise is not a concern
        and smoothing only destroys signal. Set to False for ensemble.

        Default: False
        """
        return self.data.get("ensemble_piv", {}).get("predictor_smoothing", False)

    @property
    def ensemble_predictor_rounding(self) -> bool:
        """Round the predictor field to the nearest EVEN integer (nearest 2 px).

        The symmetric warp splits the predictor half per frame (A sampled at
        -d/2, B at +d/2), so an even-integer predictor makes both half-shifts
        integer: no sub-pixel interpolation, no interpolation attenuation, and
        the fitter's noise PSD is exactly flat (P_noise = 1, frac(pred/2) = 0).
        The dense per-pixel half-shifts inside the C warp kernel are rounded
        too, so transition bands between predictor grid nodes stay integer.

        Cost: the warp no longer removes sub-pixel shear inside windows —
        worst case adds (2 px)^2/12 to the measured AB width near strong
        gradients (gradient correction becomes first-order there).

        Default: False
        """
        return self.data.get("ensemble_piv", {}).get("predictor_rounding", False)

    @property
    def ensemble_predictor_boundary_conditions(self) -> list:
        """Predictor boundary conditions for ensemble PIV.

        Each BC is a dict:
          - y_position: int, pixels from bottom (or top) of image
          - ux: float, x-displacement (pixels/frame), default 0.0
          - uy: float, y-displacement (pixels/frame), default 0.0
          - edge: str, "bottom" (default) or "top"

        For all predictor grid points between the image edge and
        y_position, the predictor interpolation uses (ux, uy)
        instead of edge-replicated padding.

        Default: [] (no BCs, backward-compatible)
        """
        raw = self.data.get("ensemble_piv", {}).get("predictor_boundary_conditions", [])
        if not raw:
            return []
        validated = []
        for i, bc in enumerate(raw):
            if not isinstance(bc, dict):
                continue
            y_pos = bc.get("y_position")
            if y_pos is None:
                continue
            edge = bc.get("edge", "bottom")
            if edge not in ("bottom", "top"):
                raise ValueError(
                    f"predictor_boundary_conditions[{i}].edge must be "
                    f"'bottom' or 'top', got {edge!r}"
                )
            validated.append(
                {
                    "y_position": int(y_pos),
                    "ux": float(bc.get("ux", 0.0)),
                    "uy": float(bc.get("uy", 0.0)),
                    "edge": edge,
                }
            )
        return validated

    @property
    def instantaneous_predictor_smoothing(self) -> bool:
        """Enable Gaussian smoothing of the predictor field between passes.

        When True (default), the predictor field is Gaussian-smoothed before
        upscaling to the next pass grid. Recommended for instantaneous PIV
        where per-pair noise is significant.

        Default: True (backward compatible)
        """
        return self.data.get("instantaneous_piv", {}).get("predictor_smoothing", True)

    @property
    def instantaneous_image_warp_interpolation(self) -> str:
        """Return interpolation method for image warping in instantaneous PIV.

        This controls the fused warp kernel interpolation when warping images
        based on the predictor field from the previous pass.

        Options:
        - 'cubic': Bicubic (Keys a=-0.75, 4x4 stencil). Default.
        - 'lanczos': Lanczos-3 (windowed sinc, 6x6 stencil). Sharper frequency
          preservation, ~1.5x slower than cubic.

        Default: 'cubic'
        """
        method = self.data.get("instantaneous_piv", {}).get(
            "image_warp_interpolation", "cubic"
        )
        valid_methods = {"cubic", "lanczos"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid instantaneous_image_warp_interpolation '{method}'. "
                f"Must be one of {valid_methods}"
            )
        return method

    @property
    def instantaneous_save_mode(self) -> str:
        """Return save mode for instantaneous PIV results.

        Options:
        - 'full': Save all 11 fields per pass.
        - 'minimal': Save only ux, uy, b_mask — the 3 fields read downstream.

        Default: 'minimal'
        """
        mode = self.data.get("instantaneous_piv", {}).get("save_mode", "minimal")
        valid = {"full", "minimal"}
        if mode not in valid:
            raise ValueError(
                f"Invalid instantaneous save_mode '{mode}'. " f"Must be one of {valid}"
            )
        return mode

    @property
    def instantaneous_save_compression(self) -> bool:
        """Return whether to use zlib compression when saving instantaneous .mat files.

        When True, scipy.io.savemat uses zlib compression — smaller files, slower writes.
        When False (default), saves uncompressed — faster writes, larger files.

        Default: False
        """
        return self.data.get("instantaneous_piv", {}).get("save_compression", False)

    @property
    def secondary_peak(self):
        """Return True if secondary peak detection is enabled."""
        return self.data.get("instantaneous_piv", {}).get("secondary_peak", False)

    @property
    def dump_correlation_planes(self) -> bool:
        """Return True if instantaneous correlation planes are dumped for debugging.

        Debug-only (CPU backend). Dumps the weighted correlation planes of EVERY
        image pair to <output>/debug_corr_planes/*_corrplanes.npz — roughly
        N_pairs x n_windows x win_h x win_w x 4 bytes per pass, so run it on a
        dataset restricted to the pair(s) of interest.

        Default: False
        """
        return self.data.get("instantaneous_piv", {}).get(
            "dump_correlation_planes", False
        )

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

        # Suppress harmless CancelledError noise from Bokeh/Tornado WebSocket cleanup.
        # These fire when a browser tab closes while Bokeh is mid-write.
        # asyncio passes the traceback via exc_info=, not in the message string,
        # so we must check record.exc_info for the exception type.
        class _CancelledErrorFilter(logging.Filter):
            def filter(self, record):
                if record.exc_info and record.exc_info[0] is not None:
                    import asyncio

                    if issubclass(record.exc_info[0], asyncio.CancelledError):
                        return False
                return True

        root_logger.addFilter(_CancelledErrorFilter())

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

        logging.debug(
            "Logging initialized. Level: %s, File: %s", self.log_level, self.log_file
        )

    @property
    def image_dtype(self):
        """Return image data type as numpy dtype."""
        import numpy as np

        dtype_str = self.data.get("images", {}).get("dtype", "float32")
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
        - 0.01: mask vector if >1% of pixels in window are masked (default)
        - 1.0: only mask vector if all pixels in window are masked

        Returns
        -------
        float
            Threshold value between 0.0 and 1.0
        """
        return self.data.get("masking", {}).get("mask_threshold", 0.01)

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
        """Backward-compatible: True when start_index == 0."""
        return self.start_index == 0

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
        env_paths = os.environ.get("PIV_ACTIVE_PATHS")
        if env_paths:
            try:
                indices = [int(i) for i in env_paths.split(",") if i.strip()]
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
            raise ValueError(
                "get_set_file_path() only valid for lavision_set image_type"
            )
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
        return {
            int(k): v.get("operations", []) if isinstance(v, dict) else v
            for k, v in cameras.items()
        }

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

        A single-camera setup (camera_count == 1) always resolves images at
        the source root; camera_subfolders entries never apply to it.

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

        # Single camera resolves at the source root — an explicit subfolder
        # entry is stale state from a previous multi-camera session, never a
        # valid override (the GUI hides subfolder inputs at count 1).
        if self.camera_count == 1:
            return ""

        subfolders = self.camera_subfolders
        # camera_num is 1-based
        idx = camera_num - 1

        if subfolders and idx < len(subfolders) and subfolders[idx]:
            return subfolders[idx]

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
            return (
                False,
                f"Source endpoint '{endpoint}' not allowed for {tool}. Allowed: {allowed}",
            )
        return True, ""

    def validate_type_name_for_tool(
        self, tool: str, type_name: str
    ) -> Tuple[bool, str]:
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
            return (
                False,
                f"Type name '{type_name}' not allowed for {tool}. Allowed: {allowed}",
            )
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
            n_nodes = (
                self.data.get("processing", {}).get("slurm", {}).get("nnodes", None)
            )
            if n_nodes is None:
                raise ValueError("n_nodes must be set for Slurm cluster")
            return int(n_nodes)
        else:
            return None

    @property
    def slurm_walltime(self):
        if self.cluster_type == "slurm":
            walltime = (
                self.data.get("processing", {})
                .get("slurm", {})
                .get("walltime", "01:00:00")
            )
            return walltime
        else:
            return None

    @property
    def slurm_memory_limit(self):
        if self.cluster_type == "slurm":
            mem_limit = (
                self.data.get("processing", {})
                .get("slurm", {})
                .get("memory_limit", "100GB")
            )
            return mem_limit
        else:
            return None

    @property
    def slurm_partition(self):
        if self.cluster_type == "slurm":
            partition = (
                self.data.get("processing", {})
                .get("slurm", {})
                .get("partition", "normal")
            )
            return partition
        else:
            return None

    @property
    def slurm_interface(self):
        if self.cluster_type == "slurm":
            interface = (
                self.data.get("processing", {}).get("slurm", {}).get("interface", "ib0")
            )
            return interface
        else:
            return None

    @property
    def slurm_job_extra(self):
        if self.cluster_type == "slurm":
            job_extra = (
                self.data.get("processing", {}).get("slurm", {}).get("job_extra", [])
            )
            return job_extra
        else:
            return None

    @property
    def slurm_job_prologue(self):
        if self.cluster_type == "slurm":
            prologue = (
                self.data.get("processing", {}).get("slurm", {}).get("prologue", [])
            )
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
