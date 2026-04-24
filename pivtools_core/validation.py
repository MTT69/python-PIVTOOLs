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
from pivtools_core.image_handling.load_images import create_piv_frame_reader
from pivtools_core.image_handling.path_utils import format_to_glob, validate_images_generic


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
            camera_path, camera_num, format_str, image_type,
            config.num_images, config.start_index == 0,
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

        # Supplementary: multi-loop .set validation
        if image_type == "lavision_set" and config.num_loops > 1:
            for loop_idx in range(config.num_loops):
                loop_path = config.get_loop_source_path(source_path, loop_idx)
                if not loop_path.exists():
                    errors.append(
                        f"Camera {camera_num}: Loop {loop_idx} .set file not found: {loop_path}"
                    )

        # Supplementary: indexing mismatch check (standard types only)
        if image_type == "standard" and result["valid"] and camera_path.exists():
            pattern = format_to_glob(format_str)
            matching_files = list(camera_path.glob(pattern))
            if matching_files:
                indices = []
                for f in matching_files:
                    try:
                        m = re.search(r'(\d+)', f.name)
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


def validate_stereo_ensemble_config(config: Config) -> Tuple[bool, List[str], List[str]]:
    """Validate stereo ensemble CoC configuration before processing.

    Checks stereo-specific prerequisites (calibration, self-calibration)
    plus shared ensemble window/overlap/type validity using stereo_ensemble
    config properties.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    # 1. Stereo calibration must exist
    if not config.is_stereo_setup:
        errors.append(
            "Stereo ensemble requires stereo calibration "
            f"(active method is '{config.active_calibration_method}', "
            "expected 'stereo_dotboard' or 'stereo_charuco')"
        )

    # 2. Self-calibration must be completed
    if not config.has_self_calibration:
        errors.append(
            "Stereo ensemble CoC requires completed self-calibration. "
            "Run 'pivtools-cli self-calibrate' first."
        )

    # 3. Camera pair validation
    cam_a, cam_b = config.stereo_ensemble_camera_pair
    cam_nums = config.camera_numbers
    if cam_a not in cam_nums:
        errors.append(f"Camera {cam_a} from stereo_ensemble camera_pair not in camera_numbers {cam_nums}")
    if cam_b not in cam_nums:
        errors.append(f"Camera {cam_b} from stereo_ensemble camera_pair not in camera_numbers {cam_nums}")

    # 4. Window sizes and overlaps (using stereo_ensemble properties)
    try:
        window_sizes = config.stereo_ensemble_window_sizes
    except ValueError as e:
        errors.append(f"Stereo ensemble window sizes: {e}")
        return False, errors, warnings

    try:
        overlaps = config.stereo_ensemble_overlaps
    except ValueError as e:
        errors.append(f"Stereo ensemble overlaps: {e}")
        return False, errors, warnings

    # 5. Overlap range check
    for i, ovlp in enumerate(overlaps):
        if ovlp < 0 or ovlp > 95:
            errors.append(f"Pass {i+1}: overlap {ovlp}% out of range (0-95%)")

    # 6. Window sizes should decrease across passes
    for i in range(1, len(window_sizes)):
        prev = window_sizes[i - 1]
        curr = window_sizes[i]
        if curr[0] > prev[0] or curr[1] > prev[1]:
            warnings.append(
                f"Pass {i+1}: window size {curr} is larger than pass {i} ({prev}). "
                "Typically window sizes decrease across passes."
            )

    # 7. Single mode sum_window validation
    try:
        ensemble_types = config.stereo_ensemble_type
    except ValueError as e:
        errors.append(f"Stereo ensemble type: {e}")
        return False, errors, warnings

    if 'single' in ensemble_types:
        try:
            config.stereo_ensemble_sum_window
        except ValueError as e:
            errors.append(f"Stereo ensemble sum window: {e}")

    # 8. Sum fitting window validation
    if config.stereo_ensemble_sum_fitting_window_enabled:
        try:
            config.stereo_ensemble_sum_fitting_window
        except ValueError as e:
            errors.append(f"Stereo ensemble sum fitting window: {e}")

    # 9. Fit method validation
    try:
        config.stereo_ensemble_fit_method
    except ValueError as e:
        errors.append(f"Stereo ensemble fit method: {e}")

    # 10. Resume validation
    resume = config.stereo_ensemble_resume_from_pass
    num_passes = config.stereo_ensemble_num_passes
    if resume != 0:
        if resume < 1 or resume > num_passes:
            errors.append(
                f"resume_from_pass={resume} is out of range. "
                f"Must be 0 (disabled) or 1-{num_passes}."
            )

    # 11. Config consistency: stereo_ensemble must agree with ensemble for inner correlator
    se_section = config.data.get("stereo_ensemble_piv", {})
    for key in ("window_size", "overlap", "type", "sum_window"):
        if se_section.get(key) is not None:
            se_val = getattr(config, f"stereo_ensemble_{key}s" if key == "window_size" else f"stereo_ensemble_{key}" if key != "overlap" else "stereo_ensemble_overlaps")
            ens_val = getattr(config, f"ensemble_{key}s" if key == "window_size" else f"ensemble_{key}" if key != "overlap" else "ensemble_overlaps")
            if se_val != ens_val:
                warnings.append(
                    f"stereo_ensemble_piv.{key} differs from ensemble_piv.{key}. "
                    f"The inner correlator uses ensemble_piv values for taper weights "
                    f"and window grids. Override ensemble_piv to match, or leave "
                    f"stereo_ensemble_piv.{key} unset to use fallback."
                )

    # 12. World bounds validation (if specified)
    wb = config.stereo_ensemble_world_bounds
    if wb is not None:
        if len(wb) != 4:
            errors.append(f"world_bounds must have 4 elements [x_min, x_max, y_min, y_max], got {len(wb)}")
        elif wb[0] >= wb[1] or wb[2] >= wb[3]:
            errors.append(
                f"world_bounds must have x_min < x_max and y_min < y_max, "
                f"got [{wb[0]}, {wb[1]}, {wb[2]}, {wb[3]}]"
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings
