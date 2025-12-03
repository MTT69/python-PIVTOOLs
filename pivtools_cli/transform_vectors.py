#!/usr/bin/env python3
"""
PIVTools Vector Transformation CLI

Wrapper for VectorTransformProcessor CLI interface.
Apply geometric transformations to PIV vector fields from the command line.

Usage:
    python -m pivtools_cli.transform_vectors \\
        --base-path /data/experiment \\
        --transformations flip_ud rotate_90_cw \\
        --camera 1

    # Transform all cameras
    python -m pivtools_cli.transform_vectors \\
        --base-path /data/experiment \\
        --transformations flip_ud

Available transformations:
    - flip_ud: Flip vertically (up-down)
    - flip_lr: Flip horizontally (left-right)
    - rotate_90_cw: Rotate 90 degrees clockwise
    - rotate_90_ccw: Rotate 90 degrees counter-clockwise
    - swap_ux_uy: Swap ux and uy components
    - invert_ux_uy: Invert (negate) ux and uy components
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

from pivtools_gui.transforms import (
    VectorTransformProcessor,
    VALID_TRANSFORMATIONS,
)


def cli_progress_callback(info: Dict):
    """Print progress to terminal."""
    progress = info.get("progress", 0)
    processed = info.get("processed_frames", 0)
    total = info.get("total_frames", 0)
    camera = info.get("current_camera", "?")
    print(f"\rCamera {camera}: {progress}% ({processed}/{total} frames)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Apply geometric transformations to PIV vector fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Flip all cameras vertically
    python -m pivtools_cli.transform_vectors --base-path ./data --transformations flip_ud

    # Rotate camera 1 by 90 degrees clockwise
    python -m pivtools_cli.transform_vectors --base-path ./data --transformations rotate_90_cw --camera 1

    # Chain multiple transformations (applied in order)
    python -m pivtools_cli.transform_vectors --base-path ./data --transformations flip_ud rotate_90_cw
""",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base directory containing PIV data",
    )
    parser.add_argument(
        "--transformations",
        nargs="+",
        required=True,
        choices=VALID_TRANSFORMATIONS,
        help="Transformations to apply (in order)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera number (omit to process all cameras)",
    )
    parser.add_argument(
        "--type",
        default="instantaneous",
        choices=["instantaneous", "ensemble", "statistics"],
        help="Data type to transform",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Transform merged data",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel workers (default: 8)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually transforming",
    )

    args = parser.parse_args()

    print("PIVTools Vector Transformation")
    print("=" * 40)
    print(f"Base path: {args.base_path}")
    print(f"Transformations: {' -> '.join(args.transformations)}")
    print(f"Camera: {args.camera or 'all'}")
    print(f"Type: {args.type}")
    print(f"Merged: {args.merged}")
    print()

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        return 0

    try:
        processor = VectorTransformProcessor(
            base_path=args.base_path,
            transformations=args.transformations,
            camera=args.camera,
            type_name=args.type,
            use_merged=args.merged,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # For CLI, we need a "source frame" - use frame 1
    # First apply transformations to frame 1 to set up pending list
    print("Applying to initial frame...")
    source_camera = args.camera or 1

    for trans in args.transformations:
        result = processor.transform_single_frame(
            frame=1,
            camera=source_camera,
            transformation=trans,
        )
        if not result["success"]:
            print(f"Error: {result['error']}")
            return 1

    print("Applying to all frames...")
    result = processor.transform_all_frames(
        source_frame=1,
        source_camera=source_camera,
        progress_callback=cli_progress_callback,
        max_workers=args.max_workers,
    )

    print()  # New line after progress

    if result["success"]:
        print(f"\nSuccess: {result['total_frames']} frames transformed")
        print(f"Cameras processed: {result['total_cameras']}")
        print(f"Elapsed time: {result['elapsed_time']:.1f}s")
        return 0
    else:
        print(f"\nError: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
