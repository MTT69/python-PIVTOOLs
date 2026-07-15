"""
Configuration Validation Module

Validates PIV configuration before processing starts.
Used by both instantaneous.py and ensemble.py entry points.
"""

import logging
import re
from typing import List, Tuple

from pivtools_core.config import Config
from pivtools_core.fft_sizes import BUILT_FFT_SIZES
from pivtools_core.image_handling.load_images import create_piv_frame_reader
from pivtools_core.image_handling.path_utils import (
    format_to_glob,
    validate_images_generic,
)

logger = logging.getLogger(__name__)


def validate_config(config: Config) -> Tuple[bool, str, List[str]]:
    """
    Validate configuration before starting PIV processing.

    Checks:
    - Source and base paths alignment
    - Active paths configuration
    - Source paths exist
    - Base paths exist (creates if missing)
    - Image files for each camera

    Args:
        config: Configuration object

    Returns:
        Tuple of (is_valid, error_message, warnings)
        - is_valid: True if configuration is valid
        - error_message: Combined error messages if invalid
        - warnings: List of warning messages (processing continues)
    """
    errors = []
    warnings = []

    # Check source_paths and base_paths have the same length
    if len(config.source_paths) != len(config.base_paths):
        errors.append(
            f"source_paths ({len(config.source_paths)}) and base_paths ({len(config.base_paths)}) "
            "must have the same number of entries for paired processing"
        )

    # Check at least one active path
    if not config.active_paths:
        errors.append(
            "No active paths configured. Set active_paths in config or check indices are valid."
        )

    # Check source paths exist
    for i, source_path in enumerate(config.source_paths):
        if not source_path.exists():
            errors.append(f"Source path {i+1} does not exist: {source_path}")

    # Check base paths exist (create if missing)
    for i, base_path in enumerate(config.base_paths):
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created base path {i+1}: {base_path}")
            except Exception as e:
                errors.append(f"Failed to create base path {i+1}: {base_path} - {e}")

    if errors:
        return False, "\n".join(errors), warnings

    # Check image files for each camera
    camera_numbers = config.camera_numbers
    source_path = config.source_paths[0]
    image_type = config.image_type  # Already case-normalized via _detect_image_type()

    for camera_num in camera_numbers:
        # Determine camera path
        format_str = config.image_format[0]
        if image_type in ("lavision_set", "lavision_im7", "cine"):
            camera_path = source_path
        else:
            folder = config.get_camera_folder(camera_num)
            camera_path = source_path / folder if folder else source_path

        # Use the canonical validator (handles .set, .cine, .im7, standard)
        read_frame = create_piv_frame_reader(camera_path, camera_num, config)

        result = validate_images_generic(
            camera_path,
            camera_num,
            format_str,
            image_type,
            config.num_images,
            config.start_index == 0,
            read_frame_fn=read_frame,
        )

        if result["error"]:
            errors.append(f"Camera {camera_num}: {result['error']}")
        elif result["valid"]:
            # Check for processing subset (found > expected)
            found = result["found_count"]
            if isinstance(found, int) and found > config.num_images:
                warnings.append(
                    f"Camera {camera_num}: Processing subset - using {config.num_images} of {found} available files"
                )

        # Supplementary: multi-loop source validation
        if config.num_loops > 1:
            for loop_idx in range(config.num_loops):
                loop_path = config.get_loop_source_path(source_path, loop_idx)
                if not loop_path.exists():
                    errors.append(
                        f"Camera {camera_num}: Loop {loop_idx} source not found: {loop_path}"
                    )

        # Supplementary: indexing mismatch check (standard types only)
        if image_type == "standard" and result["valid"] and camera_path.exists():
            pattern = format_to_glob(format_str)
            matching_files = list(camera_path.glob(pattern))
            if matching_files:
                indices = []
                for f in matching_files:
                    try:
                        m = re.search(r"(\d+)", f.name)
                        if m:
                            indices.append(int(m.group(1)))
                    except Exception:
                        pass
                if indices:
                    min_idx = min(indices)
                    expected_min = config.start_index
                    if min_idx != expected_min:
                        warnings.append(
                            f"Camera {camera_num}: File indexing mismatch - found files starting at {min_idx}, "
                            f"but start_index is {expected_min}"
                        )

    # Window sizes must be sizes the codelet FFT engine was built for.
    # Fail loud here, at config load, rather than deep in the C hot path.
    errors.extend(validate_window_sizes(config))

    if errors:
        return False, "\n".join(errors), warnings

    # Memory estimation check
    memory_warning = validate_memory_for_images(config)
    if memory_warning:
        warnings.append(memory_warning)

    # Ensemble-specific validation (when ensemble processing is enabled)
    if config.data.get("processing", {}).get("ensemble", False):
        ens_valid, ens_errors, ens_warnings = validate_ensemble_config(config)
        errors.extend(ens_errors)
        warnings.extend(ens_warnings)
        if not ens_valid:
            return False, "\n".join(errors), warnings

    return True, "", warnings


def _check_built_sizes(window_sizes, label: str) -> List[str]:
    """Return an error per [h, w] pass whose axes are not built codelet sizes."""
    errors = []
    for pass_idx, wh in enumerate(window_sizes or []):
        try:
            h, w = int(wh[0]), int(wh[1])
        except (TypeError, ValueError, IndexError):
            errors.append(f"{label} pass {pass_idx + 1}: malformed window size {wh!r}")
            continue
        bad = [v for v in (h, w) if v not in BUILT_FFT_SIZES]
        if bad:
            errors.append(
                f"{label} pass {pass_idx + 1}: window size [{h}, {w}] uses unsupported "
                f"size(s) {bad}. The FFT engine only supports {list(BUILT_FFT_SIZES)}."
            )
    return errors


def validate_window_sizes(config: Config) -> List[str]:
    """Window axis lengths must be sizes the codelet FFT engine was built for.

    Checks the instantaneous schedule always (it also drives the predictor
    passes) and the ensemble schedule when ensemble processing is enabled.

    In ``single`` ensemble mode the correlation is computed on the full
    ``sum_window`` (see ``cpu_ensemble.py`` -> ``window_sizes_for_computation``),
    so ``sum_window`` is a genuine FFT size and is checked too. Only
    ``sum_fitting_window`` is a correlation-plane crop (not an FFT), so it is
    intentionally not checked here.
    """
    errors = _check_built_sizes(config.window_sizes, "instantaneous_piv.window_size")
    if config.data.get("processing", {}).get("ensemble", False):
        errors.extend(
            _check_built_sizes(config.ensemble_window_sizes, "ensemble_piv.window_size")
        )
        # Read raw (not via properties) so a malformed type/sum_window still
        # produces a clear, collected error here rather than raising mid-validation.
        ensemble_types = config.data.get("ensemble_piv", {}).get("type", []) or []
        if "single" in ensemble_types:
            sum_window = config.data.get("ensemble_piv", {}).get("sum_window", [16, 16])
            errors.extend(_check_built_sizes([sum_window], "ensemble_piv.sum_window"))
    return errors


def validate_ensemble_config(config: Config) -> Tuple[bool, List[str], List[str]]:
    """
    Validate ensemble PIV configuration before processing starts.

    Catches ValueError from config properties (which validate internally)
    and adds additional cross-field checks not covered by individual properties.

    Args:
        config: Configuration object

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # 1. Validate ensemble_type list (length, valid values)
    try:
        ensemble_types = config.ensemble_type
    except ValueError as e:
        errors.append(f"Ensemble type: {e}")
        return False, errors, warnings

    # 2. Validate window sizes and overlaps
    try:
        window_sizes = config.ensemble_window_sizes
    except ValueError as e:
        errors.append(f"Ensemble window sizes: {e}")
        return False, errors, warnings

    try:
        overlaps = config.ensemble_overlaps
    except ValueError as e:
        errors.append(f"Ensemble overlaps: {e}")
        return False, errors, warnings

    # 3. Validate overlap values are in range
    for i, ovlp in enumerate(overlaps):
        if ovlp < 0 or ovlp > 95:
            errors.append(f"Pass {i+1}: overlap {ovlp}% out of range (0-95%)")

    # 4. Window sizes should decrease or stay the same across passes
    for i in range(1, len(window_sizes)):
        prev = window_sizes[i - 1]
        curr = window_sizes[i]
        if curr[0] > prev[0] or curr[1] > prev[1]:
            warnings.append(
                f"Pass {i+1}: window size {curr} is larger than pass {i} ({prev}). "
                "Typically window sizes decrease across passes."
            )

    # 5. Validate sum_window when single mode is used
    if "single" in ensemble_types:
        try:
            config.ensemble_sum_window
        except ValueError as e:
            errors.append(f"Ensemble sum window: {e}")

    # 6. Validate sum_fitting_window when enabled
    if config.ensemble_sum_fitting_window_enabled:
        try:
            config.ensemble_sum_fitting_window
        except ValueError as e:
            errors.append(f"Ensemble sum fitting window: {e}")

    # 7. Validate fit_method
    try:
        config.ensemble_fit_method
    except ValueError as e:
        errors.append(f"Ensemble fit method: {e}")

    # 8. Validate background subtraction / per-pair normalization consistency
    try:
        bg_method = config.ensemble_background_subtraction_method
    except ValueError as e:
        errors.append(f"Ensemble background subtraction: {e}")
        bg_method = None
    if config.ensemble_per_pair_normalization and bg_method != "window_mean":
        errors.append(
            "ensemble_piv.per_pair_normalization requires "
            "background_subtraction_method exactly 'window_mean' (the per-pair "
            "energies must be fluctuation energies, and any ensemble-level "
            "background term — 'correlation'/'image', plain or '+window_mean' — "
            "is built from raw mean images, inconsistent with normalized sums), "
            f"got '{bg_method}'."
        )

    # 9. Validate resume_from_pass
    resume = config.ensemble_resume_from_pass
    num_passes = config.ensemble_num_passes
    if resume != 0:
        if resume < 1 or resume > num_passes:
            errors.append(
                f"resume_from_pass={resume} is out of range. "
                f"Must be 0 (disabled) or 1-{num_passes}."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def estimate_memory_requirement(
    image_height: int, image_width: int, batch_size: int
) -> float:
    """
    Estimate minimum memory per worker in GB for PIV processing.

    Uses a 3x multiplier on raw image batch size to account for
    raw images + filtered copies + warped copies in memory simultaneously.

    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        batch_size: Number of image pairs per batch

    Returns:
        Estimated minimum memory in GB
    """
    # 3 × batch_size × 2 images_per_pair × H × W × 4 bytes (float32)
    bytes_needed = 3 * batch_size * 2 * image_height * image_width * 4
    return bytes_needed / (1024**3)


def parse_memory_limit_gb(memory_limit: str) -> float:
    """
    Parse a Dask memory limit string (e.g. '12GB', '512MB') to GB.

    Args:
        memory_limit: Memory string with unit suffix

    Returns:
        Memory in GB
    """
    match = re.match(r"^([\d.]+)\s*(GB|MB)$", memory_limit.strip(), re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "MB":
        return value / 1024
    return value


def validate_memory_for_images(config: Config) -> str:
    """
    Check if configured memory per worker is sufficient for the image
    resolution and batch size.

    Returns:
        Warning message string, or empty string if memory is sufficient.
    """
    try:
        image_shape = config.image_shape  # (H, W) — reads an actual image
    except Exception:
        return ""  # Can't check if we can't read images

    h, w = image_shape
    batch_size = config.batch_size
    memory_limit = config.dask_memory_limit
    megapixels = (h * w) / 1e6

    required_gb = estimate_memory_requirement(h, w, batch_size)
    configured_gb = parse_memory_limit_gb(memory_limit)

    if configured_gb <= 0:
        return ""

    if required_gb > configured_gb:
        return (
            f"Memory per worker ({memory_limit}) may be insufficient for "
            f"{megapixels:.1f} MP images with batch size {batch_size}. "
            f"Estimated minimum: {required_gb:.1f} GB. "
            f"Reduce batch size or increase memory per worker in Performance Settings "
            f"to avoid out-of-memory crashes."
        )

    return ""


def validate_batch_size_for_pod(config: Config, batch_size: int) -> Tuple[bool, str]:
    """
    Validate batch size is sufficient for POD filtering.

    POD requires a minimum number of images for meaningful decomposition.
    A batch size < 10 will likely not produce useful results.

    Args:
        config: Configuration object
        batch_size: Configured batch size

    Returns:
        Tuple of (is_valid, warning_message)
    """
    # Check if POD is configured
    filters = config.filters or []
    has_pod = any(f.get("type") == "pod" for f in filters)

    if not has_pod:
        return True, ""

    MIN_POD_BATCH_SIZE = 10

    if batch_size < MIN_POD_BATCH_SIZE:
        return False, (
            f"POD filter requires batch_size >= {MIN_POD_BATCH_SIZE} for meaningful decomposition. "
            f"Current batch_size: {batch_size}. Either increase batch_size or remove POD filter."
        )

    if batch_size < 20:
        return True, (
            f"POD filter works best with batch_size >= 20. "
            f"Current batch_size: {batch_size} may produce suboptimal results."
        )

    return True, ""


def log_validation_result(
    is_valid: bool,
    error_msg: str,
    warnings: List[str],
    config: Config,
) -> None:
    """
    Log validation results in a formatted way.

    Args:
        is_valid: Whether validation passed
        error_msg: Error message if failed
        warnings: List of warnings
        config: Configuration object
    """
    logger.info("=" * 80)
    logger.info("VALIDATING CONFIGURATION")
    logger.info("=" * 80)

    if not is_valid:
        logger.error("Configuration validation failed!")
        logger.error("=" * 80)
        logger.error("ERRORS:")
        logger.error(error_msg)
        logger.error("=" * 80)
        logger.error(
            "\nPlease fix the configuration errors in config.yaml and try again."
        )
        return

    logger.info("Configuration validated successfully")
    logger.info(f"  Source paths: {config.source_paths}")
    logger.info(f"  Cameras: {config.camera_numbers}")
    logger.info(f"  Image files: {config.num_images}")
    logger.info(f"  Frame pairs: {config.num_frame_pairs}")
    logger.info(f"  Image format: {config.image_format}")

    if warnings:
        logger.info("")
        logger.info("NOTES:")
        for warning in warnings:
            logger.info(f"  - {warning}")

    logger.info("=" * 80)
    logger.info("")
