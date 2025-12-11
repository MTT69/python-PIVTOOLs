"""
Unified batch processing utilities for PIVTools modules.

Provides consistent multi-path + multi-camera iteration patterns
following the PIV processing convention: paths outer, cameras inner.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchTarget:
    """Represents a single processing target in batch processing.

    Attributes:
        path_idx: Index into source_paths/base_paths arrays
        base_path: Output directory for this target
        source_path: Input directory for this target (optional, defaults to base_path)
        camera: Camera number (1-based), or None for merged data
        is_merged: True if this target represents merged data
    """
    path_idx: int
    base_path: Path
    source_path: Optional[Path] = None
    camera: Optional[int] = None
    is_merged: bool = False

    def __post_init__(self):
        if self.source_path is None:
            self.source_path = self.base_path

    @property
    def label(self) -> str:
        """Human-readable label for this target."""
        if self.is_merged:
            return f"Path {self.path_idx + 1}: Merged"
        elif self.camera is not None:
            return f"Path {self.path_idx + 1}: Camera {self.camera}"
        else:
            return f"Path {self.path_idx + 1}"


def iter_batch_targets(
    base_paths: List[Path],
    active_paths: List[int],
    cameras: List[int],
    include_merged: bool = False,
    single_camera_pair: bool = False,
    source_paths: Optional[List[Path]] = None,
) -> List[BatchTarget]:
    """
    Generate batch targets following PIV pattern: paths outer, cameras inner.

    This is the core utility for consistent batch processing across all
    PIVTools modules (calibration, statistics, video, transforms, merging).

    Args:
        base_paths: List of output directories
        active_paths: Indices into base_paths to process
        cameras: Camera numbers to process (1-based)
        include_merged: Whether to add merged data targets
        single_camera_pair: True for stereo calibration (process one pair per path,
                           skip camera loop entirely)
        source_paths: Optional list of input directories (defaults to base_paths)

    Returns:
        List of BatchTarget objects in processing order

    Examples:
        # Normal multi-camera processing
        >>> targets = iter_batch_targets(
        ...     base_paths=[Path("/data/exp1"), Path("/data/exp2")],
        ...     active_paths=[0, 1],
        ...     cameras=[1, 2],
        ...     include_merged=True
        ... )
        # Returns: [Path0-Cam1, Path0-Cam2, Path0-Merged, Path1-Cam1, Path1-Cam2, Path1-Merged]

        # Stereo calibration (one pair per path)
        >>> targets = iter_batch_targets(
        ...     base_paths=[Path("/data/exp1")],
        ...     active_paths=[0],
        ...     cameras=[1, 2],
        ...     single_camera_pair=True
        ... )
        # Returns: [Path0 (stereo pair)]
    """
    if source_paths is None:
        source_paths = base_paths

    targets = []

    for path_idx in active_paths:
        # Validate path index
        if path_idx >= len(base_paths) or path_idx >= len(source_paths):
            logger.warning(f"Skipping invalid path index {path_idx}")
            continue

        base_path = base_paths[path_idx]
        source_path = source_paths[path_idx]

        # For stereo calibration, process camera pair once per path
        if single_camera_pair:
            targets.append(BatchTarget(
                path_idx=path_idx,
                base_path=base_path,
                source_path=source_path,
                camera=None,  # Pair processed together
                is_merged=False
            ))
            continue

        # Process individual cameras
        for camera in cameras:
            targets.append(BatchTarget(
                path_idx=path_idx,
                base_path=base_path,
                source_path=source_path,
                camera=camera,
                is_merged=False
            ))

        # Process merged data if requested
        if include_merged and len(cameras) > 1:
            targets.append(BatchTarget(
                path_idx=path_idx,
                base_path=base_path,
                source_path=source_path,
                camera=cameras[0] if cameras else 1,  # Use first camera for path resolution
                is_merged=True
            ))

    return targets


def run_batch_with_progress(
    targets: List[BatchTarget],
    process_fn: Callable[[BatchTarget], Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    Execute batch processing with progress tracking.

    Args:
        targets: List of BatchTarget objects to process
        process_fn: Function to process each target, returns result dict
        progress_callback: Optional callback(current, total, message)

    Returns:
        List of result dictionaries from process_fn
    """
    results = []
    total = len(targets)

    for idx, target in enumerate(targets, start=1):
        logger.info(f"Processing {idx}/{total}: {target.label}")

        if progress_callback:
            progress_callback(idx, total, target.label)

        try:
            result = process_fn(target)
            result["target_label"] = target.label
            result["success"] = True
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {target.label}: {e}", exc_info=True)
            results.append({
                "target_label": target.label,
                "success": False,
                "error": str(e),
            })

    return results


def count_batch_targets(
    num_paths: int,
    num_cameras: int,
    include_merged: bool = False,
    single_camera_pair: bool = False,
) -> int:
    """
    Calculate total number of targets for progress estimation.

    Args:
        num_paths: Number of active paths
        num_cameras: Number of cameras
        include_merged: Whether merged data is included
        single_camera_pair: Whether processing stereo pairs

    Returns:
        Total number of targets that will be generated
    """
    if single_camera_pair:
        return num_paths

    targets_per_path = num_cameras
    if include_merged and num_cameras > 1:
        targets_per_path += 1

    return num_paths * targets_per_path
