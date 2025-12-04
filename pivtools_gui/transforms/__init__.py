"""
Vector field transformation module.

Provides geometric transformations for PIV vector fields with support for:
- Single frame transformations
- Batch processing across all frames and cameras
- CLI for batch operations
- GUI integration with progress callbacks
"""

from .transform_operations import (
    apply_transformation_to_piv_result,
    apply_transformation_to_coordinates,
    backup_original_data,
    restore_original_data,
    has_original_backup,
    process_frame_worker,
    VALID_TRANSFORMATIONS,
)
from .vector_transform_processor import VectorTransformProcessor

__all__ = [
    "VectorTransformProcessor",
    "apply_transformation_to_piv_result",
    "apply_transformation_to_coordinates",
    "backup_original_data",
    "restore_original_data",
    "has_original_backup",
    "process_frame_worker",
    "VALID_TRANSFORMATIONS",
]
