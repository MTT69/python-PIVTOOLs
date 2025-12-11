"""
Vector field transformation module.

Provides geometric transformations for PIV vector fields with support for:
- Single frame transformations
- Batch processing across all frames and cameras
- CLI for batch operations
- GUI integration with progress callbacks
- Transformation simplification/cancellation
"""

from .transform_operations import (
    apply_transformation_to_piv_result,
    apply_transformation_to_coordinates,
    backup_original_data,
    coords_mat_to_saveable,
    coords_to_structured_array,
    mat_dict_to_saveable,
    piv_result_to_structured_array,
    restore_original_data,
    has_original_backup,
    process_frame_worker,
    simplify_transformations,
    validate_transformations,
    PIV_RESULT_FIELDS,
    VALID_TRANSFORMATIONS,
)
from .vector_transform_processor import VectorTransformProcessor
from .transform_production import TransformProcessor

__all__ = [
    "TransformProcessor",
    "VectorTransformProcessor",
    "apply_transformation_to_piv_result",
    "apply_transformation_to_coordinates",
    "backup_original_data",
    "coords_mat_to_saveable",
    "coords_to_structured_array",
    "mat_dict_to_saveable",
    "piv_result_to_structured_array",
    "restore_original_data",
    "has_original_backup",
    "process_frame_worker",
    "simplify_transformations",
    "validate_transformations",
    "PIV_RESULT_FIELDS",
    "VALID_TRANSFORMATIONS",
]
