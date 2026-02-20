"""
Configuration Validation Module

Validates PIV configuration before processing starts.
Used by both instantaneous.py and ensemble.py entry points.
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

from pivtools_core.config import Config
from pivtools_core.image_handling.path_utils import format_to_glob


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
        if image_type in ("lavision_set", "lavision_im7"):
            camera_path = source_path
        else:
            folder = config.get_camera_folder(camera_num)
            camera_path = source_path / folder if folder else source_path

        if not camera_path.exists():
            errors.append(f"Camera {camera_num} path does not exist: {camera_path}")
            continue

        # Count files
        if image_type == "lavision_set":
            # Set files: single file
            if camera_path.is_file():
                set_file = camera_path
            else:
                set_file = camera_path / format_str

            if not set_file.exists():
                errors.append(f"Camera {camera_num}: Set file not found: {set_file}")

            # Multi-loop validation: check all loop .set files exist
            if config.num_loops > 1:
                for loop_idx in range(config.num_loops):
                    loop_path = config.get_loop_source_path(source_path, loop_idx)
                    if not loop_path.exists():
                        errors.append(
                            f"Camera {camera_num}: Loop {loop_idx} .set file not found: {loop_path}"
                        )

        elif image_type == "lavision_im7":
            # IM7 files
            pattern = format_to_glob(format_str)
            matching_files = list(camera_path.glob(pattern))
            expected = config.num_images
            if len(matching_files) != expected:
                errors.append(
                    f"Camera {camera_num}: Found {len(matching_files)} IM7 files, expected {expected}. "
                    f"Path: {camera_path}, Pattern: {pattern}"
                )
        else:
            # Standard files
            expected = config.num_images
            if len(config.image_format) == 2:
                # A/B format: count A files
                pattern_a = format_to_glob(config.image_format[0])
                matching_files = list(camera_path.glob(pattern_a))
            else:
                # Time-resolved: count all files
                pattern = format_to_glob(format_str)
                matching_files = list(camera_path.glob(pattern))

            # Check for indexing mismatch
            if image_type == "standard" and matching_files:
                indices = []
                for f in matching_files:
                    try:
                        match = re.search(r'(\d+)', f.name)
                        if match:
                            idx = int(match.group(1))
                            indices.append(idx)
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

            if len(matching_files) < expected:
                # ERROR: Not enough files
                all_files = sorted([f.name for f in camera_path.iterdir() if f.is_file()])[:5]
                file_list = ', '.join(all_files) if all_files else "(empty folder)"
                errors.append(
                    f"Camera {camera_num}: Missing files - found {len(matching_files)}, expected {expected}.\n"
                    f"  Path: {camera_path}\n"
                    f"  Pattern: {format_str}\n"
                    f"  Found files: {file_list}"
                )
            elif len(matching_files) > expected:
                # WARNING: Processing subset (this is fine!)
                warnings.append(
                    f"Camera {camera_num}: Processing subset - using {expected} of {len(matching_files)} available files"
                )

    if errors:
        return False, "\n".join(errors), warnings

    # Ensemble-specific validation (when ensemble processing is enabled)
    if config.data.get("processing", {}).get("ensemble", False):
        ens_valid, ens_errors, ens_warnings = validate_ensemble_config(config)
        errors.extend(ens_errors)
        warnings.extend(ens_warnings)
        if not ens_valid:
            return False, "\n".join(errors), warnings

    return True, "", warnings


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
            errors.append(
                f"Pass {i+1}: overlap {ovlp}% out of range (0-95%)"
            )

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
    if 'single' in ensemble_types:
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
        fit_method = config.ensemble_fit_method
    except ValueError as e:
        errors.append(f"Ensemble fit method: {e}")
        fit_method = None

    # 8. K-space SNR threshold must be positive
    if fit_method == 'kspace':
        snr = config.ensemble_kspace_snr_threshold
        if snr <= 0:
            errors.append(
                f"kspace_snr_threshold must be positive, got {snr}"
            )
        warnings.append(
            "K-space fitting is BETA. Results should be validated against Gaussian fitting."
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
    has_pod = any(f.get('type') == 'pod' for f in filters)

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
        logger.error("\nPlease fix the configuration errors in config.yaml and try again.")
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
