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

    for camera_num in camera_numbers:
        # Determine camera path
        format_str = config.image_format[0]
        if '.set' in str(format_str) or '.im7' in str(format_str):
            camera_path = source_path
        else:
            folder = config.get_camera_folder(camera_num)
            camera_path = source_path / folder if folder else source_path

        if not camera_path.exists():
            errors.append(f"Camera {camera_num} path does not exist: {camera_path}")
            continue

        # Count files
        if '.set' in str(format_str):
            # Set files: single file
            if camera_path.is_file():
                set_file = camera_path
            else:
                set_file = camera_path / format_str

            if not set_file.exists():
                errors.append(f"Camera {camera_num}: Set file not found: {set_file}")

        elif '.im7' in str(format_str):
            # IM7 files
            pattern = format_str.replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
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
                pattern_a = config.image_format[0].replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
                matching_files = list(camera_path.glob(pattern_a))
            else:
                # Time-resolved: count all files
                pattern = format_str.replace("%05d", "*").replace("%04d", "*").replace("%d", "*")
                matching_files = list(camera_path.glob(pattern))

            # Check for indexing mismatch
            if not ('.set' in str(format_str) or '.im7' in str(format_str)) and matching_files:
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
                    expected_min = 0 if config.zero_based_indexing else 1
                    if min_idx != expected_min:
                        warnings.append(
                            f"Camera {camera_num}: File indexing mismatch - found files starting at {min_idx}, "
                            f"but zero_based_indexing is {'enabled' if config.zero_based_indexing else 'disabled'} "
                            f"(expects {expected_min})"
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

    return True, "", warnings


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
