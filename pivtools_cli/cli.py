#!/usr/bin/env python3
"""
PIVTOOLs CLI - Command line interface for PIVTOOLs

Commands:
  init                 - Initialize a new PIVTOOLs workspace
  instantaneous        - Run instantaneous PIV processing
  ensemble             - Run ensemble PIV processing
  detect-planar        - Detect dot/circle grid, generate camera model
  detect-charuco       - Detect ChArUco board, generate camera model
  detect-stereo-planar - Detect dot/circle grid, generate stereo model
  detect-stereo-charuco- Detect ChArUco board, generate stereo model
  apply-calibration    - Apply calibration to vectors (pixels to m/s)
  align-coordinates    - Apply global coordinate alignment to calibrated vectors
  transform            - Apply geometric transforms to vectors
  merge                - Merge multi-camera vector fields
  statistics           - Compute PIV statistics
  video                - Create visualization videos
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_active_paths_from_args(args, config):
    """
    Get active paths, with CLI override support.

    If args.active_paths is provided (e.g., "0,1,2"), parse and return those indices.
    Otherwise, return config.active_paths.

    Returns list of path indices.
    """
    if hasattr(args, 'active_paths') and args.active_paths:
        # Parse comma-separated indices
        return [int(i.strip()) for i in args.active_paths.split(',')]
    return config.active_paths


# =============================================================================
# APPLY-CALIBRATION COMMAND
# =============================================================================

def apply_calibration_command(args):
    """Apply calibration to PIV vectors (pixels to physical units m/s)."""
    from pivtools_core.config import get_config

    config = get_config()

    # Apply CLI overrides
    cameras = [args.camera] if args.camera else config.camera_numbers
    type_name = args.type_name or "instantaneous"
    runs_to_process = None
    if args.runs:
        runs_to_process = [int(r) for r in args.runs.split(",")]

    # Determine calibration method (CLI override or config)
    method = args.method or config.active_calibration_method

    # Get active paths (with CLI override support)
    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Apply Calibration - Starting")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Active paths: {len(active_paths)}")
    print(f"Cameras: {cameras}")
    print(f"Type: {type_name}")
    print(f"Runs: {runs_to_process or 'all'}")

    results = []
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx + 1}/{len(active_paths)}: {base_dir}")
        print("-" * 40)

        # Pre-compute alignment shifts for this path (no data file I/O)
        alignment = None
        if getattr(args, 'align_coordinates', False) and config.global_coordinates_enabled:
            try:
                from pivtools_gui.calibration.global_coordinate_alignment import GlobalCoordinateAligner
                aligner = GlobalCoordinateAligner(base_dir, config)
                alignment = aligner.precompute_camera_shifts(type_name)
                if alignment:
                    print(f"  Alignment pre-computed: {len(alignment['camera_shifts'])} cameras, invert_ux={alignment['invert_ux']}")
            except Exception as e:
                print(f"  WARNING: Failed to pre-compute alignment: {e}")

        for camera in cameras:
            try:
                if method == "scale_factor":
                    from pivtools_gui.calibration.scale_factor_calibration_production import ScaleFactorCalibrator
                    calibrator = ScaleFactorCalibrator(
                        base_path=base_dir,
                        type_name=type_name,
                        config=config,
                        alignment=alignment,
                    )
                    result = calibrator.process_camera(
                        camera_num=camera,
                        image_count=config.num_frame_pairs,
                    )
                elif method == "polynomial":
                    from pivtools_gui.calibration.calibration_poly.polynomial_calibration_production import PolynomialVectorCalibrator
                    calibrator = PolynomialVectorCalibrator(
                        base_dir=base_dir,
                        camera_num=camera,
                        type_name=type_name,
                        config=config,
                    )
                    result = calibrator.process_vectors()
                else:
                    # dotboard or charuco -> pinhole VectorCalibrator
                    from pivtools_gui.calibration.vector_calibration_production import VectorCalibrator
                    calibrator = VectorCalibrator(
                        base_dir=base_dir,
                        camera_num=camera,
                        model_type=method,
                        type_name=type_name,
                        runs=runs_to_process,
                        config=config,
                        z_world=config.self_calibration_z_offset,
                        tilt_x=config.self_calibration_tilt_x,
                        tilt_y=config.self_calibration_tilt_y,
                    )
                    calibrator.process_run(alignment=alignment)
                    result = {"success": True, "calibrated_count": "N/A"}

                result["path_idx"] = path_idx
                result["camera"] = camera
                results.append(result)

                if result.get("success"):
                    print(f"  Camera {camera}: OK - {result.get('calibrated_count', 0)} files")
                else:
                    print(f"  Camera {camera}: FAILED - {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"  Camera {camera}: FAILED - {e}")
                results.append({"success": False, "error": str(e), "path_idx": path_idx, "camera": camera})

        # Report alignment result for this path
        if alignment:
            print(f"  Alignment applied (fused):")
            for cam, (sx, sy) in alignment["camera_shifts"].items():
                print(f"    Cam {cam}: dx={sx:.4f} mm, dy={sy:.4f} mm")
            if alignment["invert_ux"]:
                print(f"  invert_ux applied")

    if getattr(args, 'align_coordinates', False) and not config.global_coordinates_enabled:
        print("\nWarning: --align-coordinates specified but global_coordinates.enabled is False in config")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} operations succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# ALIGN-COORDINATES COMMAND
# =============================================================================

def align_coordinates_command(args):
    """Apply global coordinate alignment to already-calibrated vectors."""
    from pivtools_core.config import get_config
    from pivtools_gui.calibration.global_coordinate_alignment import GlobalCoordinateAligner

    config = get_config()

    if not config.global_coordinates_enabled:
        print("Error: global_coordinates.enabled is False in config.yaml")
        print("Set calibration.global_coordinates.enabled: true and configure datum/overlap points first.")
        sys.exit(1)

    type_name = args.type_name or "instantaneous"
    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Global Coordinate Alignment")
    print("=" * 60)
    print(f"Type: {type_name}")
    print(f"Active paths: {active_paths}")
    print(f"Datum pixel: {config.global_coordinates_datum_pixel}")
    print(f"Datum physical: {config.global_coordinates_datum_physical}")

    success = True
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx}: {base_dir}")
        print("-" * 40)

        try:
            aligner = GlobalCoordinateAligner(base_dir, config)
            result = aligner.apply_alignment(type_name, force=getattr(args, 'force', False))
            if result.get("status") == "completed":
                print("  Alignment applied successfully")
                for cam_key, cam_info in result.get("cameras", {}).items():
                    print(f"    Cam {cam_key}: dx={cam_info['shift_x']:.4f} mm, dy={cam_info['shift_y']:.4f} mm")
                if result.get("invert_ux"):
                    print("  invert_ux applied")
            else:
                print(f"  Failed: {result.get('error', 'Unknown')}")
                success = False
        except Exception as e:
            print(f"  Failed: {e}")
            success = False

    sys.exit(0 if success else 1)


# =============================================================================
# SELF-CALIBRATE COMMAND
# =============================================================================

def self_calibrate_command(args):
    """Run stereo self-calibration (Wieneke 2005) to correct laser sheet misalignment."""
    import math
    from pivtools_core.config import get_config
    from pivtools_gui.calibration.services.self_calibration_service import (
        run_self_cal_job, save_self_cal_to_config, result_to_dict,
    )

    config = get_config()

    # Parse camera pair
    if args.camera_pair:
        camera_pair = [int(c.strip()) for c in args.camera_pair.split(",")]
    elif config.stereo_pairs:
        camera_pair = list(config.stereo_pairs[0])
    else:
        camera_pair = [1, 2]

    cam1_num, cam2_num = camera_pair[0], camera_pair[1]

    # Determine method
    method = args.method or config.active_calibration_method
    if method.startswith("stereo_"):
        method = method.replace("stereo_", "")

    # Parameters
    source_path_idx = int(args.active_paths.split(",")[0]) if args.active_paths else 0
    sc_cfg = config.self_calibration_config
    n_images = args.n_images or sc_cfg.get("n_images", 20)
    window_size = args.window_size or sc_cfg.get("window_size", 64)
    overlap = sc_cfg.get("overlap", 50.0)

    base_dir = str(config.base_paths[0])

    print("=" * 60)
    print("Self-Calibration (Wieneke 2005)")
    print("=" * 60)
    print(f"Camera pair: {cam1_num}, {cam2_num}")
    print(f"Method: {method}")
    print(f"Source images: {n_images} (source path idx {source_path_idx})")
    print(f"Window: {window_size} px, overlap: {overlap}%")
    print()

    def progress_cb(data):
        status = data.get("status", "")
        pct = data.get("progress", 0)
        print(f"  [{pct:3d}%] {status}")

    try:
        result = run_self_cal_job(
            config=config,
            base_dir=base_dir,
            source_path_idx=source_path_idx,
            cam1_num=cam1_num,
            cam2_num=cam2_num,
            method=method,
            n_images=n_images,
            window_size=window_size,
            overlap=overlap,
            progress_callback=progress_cb,
        )

        # Save to config
        save_self_cal_to_config(
            config, result,
            n_images=n_images,
            window_size=window_size,
            overlap=overlap,
        )

        # Print summary
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Converged:    {result.converged}")
        print(f"  Iterations:   {result.n_iterations}")
        print(f"  Z-offset:     {result.z_offset:.4f} mm")
        print(f"  Tilt X:       {math.degrees(result.tilt_x):.4f}°")
        print(f"  Tilt Y:       {math.degrees(result.tilt_y):.4f}°")
        print(f"  Final RMS:    {result.final_rms_disparity:.4f} px")
        print()

        if result.converged:
            print("Self-calibration params saved to config.yaml")
            print("These will be applied automatically during 'apply-calibration'.")
        else:
            print("WARNING: Self-calibration did not converge.")
            print("Results saved but may not improve accuracy.")

        sys.exit(0 if result.converged else 1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# APPLY-STEREO COMMAND
# =============================================================================

def apply_stereo_command(args):
    """Apply stereo calibration for 3D velocity reconstruction."""
    from pivtools_core.config import get_config
    from pivtools_gui.stereo_reconstruction.stereo_reconstruction_production import StereoReconstructor

    config = get_config()

    # Parse camera pair from CLI or config
    if args.camera_pair:
        camera_pair = [int(c.strip()) for c in args.camera_pair.split(",")]
    else:
        camera_pair = config.data.get("calibration", {}).get("stereo_dotboard", {}).get("camera_pair", [1, 2])

    method = args.method  # "dotboard" or "charuco", None uses config default
    type_name = args.type_name or "instantaneous"
    runs_to_process = [int(r) for r in args.runs.split(",")] if args.runs else None

    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Stereo 3D Reconstruction - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")
    print(f"Camera pair: {camera_pair}")
    print(f"Method: {method or 'from config'}")
    print(f"Type: {type_name}")
    print(f"Runs: {runs_to_process or 'all'}")

    results = []
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx + 1}/{len(active_paths)}: {base_dir}")
        print("-" * 40)

        try:
            reconstructor = StereoReconstructor(
                base_dir=base_dir,
                camera_pair=camera_pair,
                model_type=method,
                type_name=type_name,
                runs=runs_to_process,
                config=config,
            )
            reconstructor.process_run()
            result = {"success": True}
            result["path_idx"] = path_idx
            results.append(result)

            print(f"  Stereo reconstruction: OK")
        except Exception as e:
            print(f"  FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} paths succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# DETECT-PLANAR COMMAND (Generate camera model from dot/circle grid)
# =============================================================================

def detect_planar_command(args):
    """Detect dot/circle grid and generate camera model."""
    from pivtools_core.config import get_config
    from pivtools_gui.calibration.calibration_planar.planar_calibration_production import MultiViewCalibrator

    config = get_config()

    # Apply CLI calibration source override
    if args.calibration_source:
        config.data.setdefault("calibration", {})["calibration_sources"] = [args.calibration_source]
        print(f"Using calibration source override: {args.calibration_source}")

    # Determine cameras to process
    cameras = [args.camera] if args.camera else config.camera_numbers

    # Get paths
    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Planar Camera Calibration - Starting")
    print("=" * 60)
    print(f"Cameras: {cameras}")
    print(f"Active paths: {len(active_paths)}")

    # Get calibration settings from config
    dotboard_cfg = config.data.get("calibration", {}).get("dotboard", {})
    pattern_cols = dotboard_cfg.get("pattern_cols", 10)
    pattern_rows = dotboard_cfg.get("pattern_rows", 10)
    dot_spacing_mm = dotboard_cfg.get("dot_spacing_mm", 28.89)
    asymmetric = dotboard_cfg.get("asymmetric", False)

    calib_cfg = config.data.get("calibration", {})
    file_pattern = calib_cfg.get("image_format", "calib%05d.tif")

    model_type = getattr(args, "model_type", "pinhole") or "pinhole"
    print(f"Grid: {pattern_cols}x{pattern_rows}, spacing: {dot_spacing_mm}mm, model: {model_type}")

    results = []
    for path_idx in active_paths:
        source_dir = config.source_paths[path_idx]
        base_dir = config.base_paths[path_idx]
        print(f"\nPath {path_idx + 1}/{len(active_paths)}:")
        print(f"  Source: {source_dir}")
        print(f"  Base: {base_dir}")
        print("-" * 40)

        try:
            calibrator = MultiViewCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_count=len(cameras),
                file_pattern=file_pattern,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
                dot_spacing_mm=dot_spacing_mm,
                asymmetric=asymmetric,
                model_type=model_type,
                config=config,
            )

            for camera in cameras:
                result = calibrator.process_single_camera(
                    cam_num=camera,
                    save_visualizations=True,
                )
                result["path_idx"] = path_idx
                result["camera"] = camera
                results.append(result)

                if result.get("success"):
                    print(f"  Camera {camera}: OK - RMS error: {result.get('rms_error', 0):.4f}")
                else:
                    print(f"  Camera {camera}: FAILED - {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"  FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} camera calibrations succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# DETECT-CHARUCO COMMAND (Generate camera model from ChArUco board)
# =============================================================================

def detect_charuco_command(args):
    """Detect ChArUco board and generate camera model."""
    from pivtools_core.config import get_config
    from pivtools_gui.calibration.calibration_charuco.charuco_calibration_production import ChArUcoCalibrator

    config = get_config()

    # Apply CLI calibration source override
    if args.calibration_source:
        config.data.setdefault("calibration", {})["calibration_sources"] = [args.calibration_source]
        print(f"Using calibration source override: {args.calibration_source}")

    # Determine cameras to process
    cameras = [args.camera] if args.camera else config.camera_numbers

    # Get paths
    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("ChArUco Camera Calibration - Starting")
    print("=" * 60)
    print(f"Cameras: {cameras}")
    print(f"Active paths: {len(active_paths)}")

    # Get calibration settings from config
    charuco_cfg = config.data.get("calibration", {}).get("charuco", {})
    squares_h = charuco_cfg.get("squares_h", 10)
    squares_v = charuco_cfg.get("squares_v", 9)
    square_size = charuco_cfg.get("square_size", 0.03)
    marker_ratio = charuco_cfg.get("marker_ratio", 0.5)
    aruco_dict = charuco_cfg.get("aruco_dict", "DICT_4X4_1000")
    min_corners = charuco_cfg.get("min_corners", 6)
    dt = charuco_cfg.get("dt", 1.0)

    calib_cfg = config.data.get("calibration", {})
    file_pattern = calib_cfg.get("image_format", "calib%05d.tif")

    model_type = getattr(args, "model_type", "pinhole") or "pinhole"
    print(f"Board: {squares_h}x{squares_v} squares, size: {square_size}m, model: {model_type}")

    results = []
    for path_idx in active_paths:
        source_dir = config.source_paths[path_idx]
        base_dir = config.base_paths[path_idx]
        print(f"\nPath {path_idx + 1}/{len(active_paths)}:")
        print(f"  Source: {source_dir}")
        print(f"  Base: {base_dir}")
        print("-" * 40)

        try:
            calibrator = ChArUcoCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_count=len(cameras),
                file_pattern=file_pattern,
                squares_h=squares_h,
                squares_v=squares_v,
                square_size=square_size,
                marker_ratio=marker_ratio,
                aruco_dict=aruco_dict,
                min_corners=min_corners,
                dt=dt,
                config=config,
                model_type=model_type,
            )

            for camera in cameras:
                result = calibrator.process_camera(
                    cam_num=camera,
                    save_visualizations=True,
                )
                result["path_idx"] = path_idx
                result["camera"] = camera
                results.append(result)

                if result.get("success"):
                    print(f"  Camera {camera}: OK - RMS error: {result.get('rms_error', 0):.4f}")
                else:
                    print(f"  Camera {camera}: FAILED - {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"  FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} camera calibrations succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# DETECT-STEREO-PLANAR COMMAND (Generate stereo camera model from dot/circle grid)
# =============================================================================

def detect_stereo_planar_command(args):
    """Detect dot/circle grid and generate stereo camera model."""
    from pivtools_core.config import get_config
    from pivtools_gui.stereo_reconstruction.stereo_dotboard_calibration_production import StereoDotboardCalibrator

    config = get_config()

    # Apply CLI calibration source override
    if args.calibration_source:
        config.data.setdefault("calibration", {})["calibration_sources"] = [args.calibration_source]
        print(f"Using calibration source override: {args.calibration_source}")

    # Get paths
    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Stereo Planar Camera Calibration - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")

    # Get calibration settings from config
    stereo_dotboard_cfg = config.data.get("calibration", {}).get("stereo_dotboard", {})
    camera_pair = stereo_dotboard_cfg.get("camera_pair", [1, 2])
    pattern_cols = stereo_dotboard_cfg.get("pattern_cols", 10)
    pattern_rows = stereo_dotboard_cfg.get("pattern_rows", 10)
    dot_spacing_mm = stereo_dotboard_cfg.get("dot_spacing_mm", 28.89)
    asymmetric = stereo_dotboard_cfg.get("asymmetric", False)

    calib_cfg = config.data.get("calibration", {})
    file_pattern = calib_cfg.get("image_format", "calib%05d.tif")

    print(f"Camera pair: {camera_pair}")
    print(f"Grid: {pattern_cols}x{pattern_rows}, spacing: {dot_spacing_mm}mm")

    results = []
    for path_idx in active_paths:
        source_dir = config.source_paths[path_idx]
        base_dir = config.base_paths[path_idx]
        print(f"\nPath {path_idx + 1}/{len(active_paths)}:")
        print(f"  Source: {source_dir}")
        print(f"  Base: {base_dir}")
        print("-" * 40)

        try:
            calibrator = StereoDotboardCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_pair=camera_pair,
                file_pattern=file_pattern,
                pattern_cols=pattern_cols,
                pattern_rows=pattern_rows,
                dot_spacing_mm=dot_spacing_mm,
                asymmetric=asymmetric,

                config=config,
            )

            result = calibrator.process_camera_pair(save_visualizations=True)
            result["path_idx"] = path_idx
            results.append(result)

            if result.get("success"):
                print(f"  Stereo calibration: OK - RMS error: {result.get('rms_error', 0):.4f}")
            else:
                print(f"  Stereo calibration: FAILED - {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"  FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} stereo calibrations succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# DETECT-STEREO-CHARUCO COMMAND (Generate stereo camera model from ChArUco board)
# =============================================================================

def detect_stereo_charuco_command(args):
    """Detect ChArUco board and generate stereo camera model."""
    from pivtools_core.config import get_config
    from pivtools_gui.stereo_reconstruction.stereo_charuco_calibration_production import StereoCharucoCalibrator

    config = get_config()

    # Apply CLI calibration source override
    if args.calibration_source:
        config.data.setdefault("calibration", {})["calibration_sources"] = [args.calibration_source]
        print(f"Using calibration source override: {args.calibration_source}")

    # Get paths
    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Stereo ChArUco Camera Calibration - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")

    # Get calibration settings from config
    stereo_charuco_cfg = config.data.get("calibration", {}).get("stereo_charuco", {})
    camera_pair = stereo_charuco_cfg.get("camera_pair", [1, 2])

    charuco_cfg = config.data.get("calibration", {}).get("charuco", {})
    squares_h = charuco_cfg.get("squares_h", 10)
    squares_v = charuco_cfg.get("squares_v", 9)
    square_size = charuco_cfg.get("square_size", 0.03)
    marker_ratio = charuco_cfg.get("marker_ratio", 0.5)
    aruco_dict = charuco_cfg.get("aruco_dict", "DICT_4X4_1000")
    min_corners = charuco_cfg.get("min_corners", 6)

    calib_cfg = config.data.get("calibration", {})
    file_pattern = calib_cfg.get("image_format", "calib%05d.tif")

    print(f"Camera pair: {camera_pair}")
    print(f"Board: {squares_h}x{squares_v} squares, size: {square_size}m")

    results = []
    for path_idx in active_paths:
        source_dir = config.source_paths[path_idx]
        base_dir = config.base_paths[path_idx]
        print(f"\nPath {path_idx + 1}/{len(active_paths)}:")
        print(f"  Source: {source_dir}")
        print(f"  Base: {base_dir}")
        print("-" * 40)

        try:
            calibrator = StereoCharucoCalibrator(
                source_dir=source_dir,
                base_dir=base_dir,
                camera_pair=camera_pair,
                file_pattern=file_pattern,
                squares_h=squares_h,
                squares_v=squares_v,
                square_size=square_size,
                marker_ratio=marker_ratio,
                aruco_dict=aruco_dict,
                min_corners=min_corners,
                config=config,
            )

            result = calibrator.process_camera_pair(save_visualizations=True)
            result["path_idx"] = path_idx
            results.append(result)

            if result.get("success"):
                print(f"  Stereo calibration: OK - RMS error: {result.get('rms_error', 0):.4f}")
            else:
                print(f"  Stereo calibration: FAILED - {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"  FAILED - {e}")
            import traceback
            traceback.print_exc()
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} stereo calibrations succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# TRANSFORM COMMAND
# =============================================================================

def transform_command(args):
    """Apply geometric transforms to PIV vector fields."""
    from pivtools_core.config import get_config
    from pivtools_gui.transforms.transform_production import TransformProcessor

    config = get_config()

    # Get camera transforms from config or CLI
    if args.operations:
        # Parse CLI operations: "flip_ud,rotate_90_cw"
        ops = [op.strip() for op in args.operations.split(",")]
        cameras = [args.camera] if args.camera else config.camera_numbers
        camera_transforms = {cam: ops for cam in cameras}
    else:
        # Use transforms from config
        camera_transforms = config.transforms_cameras or {}
        if not camera_transforms:
            print("Error: No transforms configured. Use --operations or set transforms.cameras in config.yaml")
            sys.exit(1)

    type_name = args.type_name or config.transforms_type_name or "instantaneous"

    # Determine source_endpoint
    # Priority: new --source-endpoint flag > legacy --merged flag > config
    source_endpoint = args.source_endpoint

    # Handle legacy --merged flag for backward compatibility
    if source_endpoint is None and args.merged:
        source_endpoint = "merged"

    # Fall back to config
    if source_endpoint is None:
        source_endpoint = config.transforms_source_endpoint

    # Determine use_merged and use_stereo from source_endpoint
    use_merged = source_endpoint == "merged"
    use_stereo = source_endpoint == "stereo"

    # Update config if CLI specified source_endpoint
    if args.source_endpoint or args.merged:
        config.data.setdefault("transforms", {})["source_endpoint"] = source_endpoint or "regular"
        config.save()
        print(f"Updated config: transforms.source_endpoint = {source_endpoint or 'regular'}")

    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Vector Transform - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")
    print(f"Camera transforms: {camera_transforms}")
    print(f"Type: {type_name}")
    print(f"Source endpoint: {source_endpoint or 'regular'}")

    results = []
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx + 1}/{len(active_paths)}: {base_dir}")
        print("-" * 40)

        try:
            processor = TransformProcessor(
                base_dir=base_dir,
                camera_transforms=camera_transforms,
                type_name=type_name,
                use_merged=use_merged,
                config=config,
            )
            result = processor.process_all_cameras()
            result["path_idx"] = path_idx
            results.append(result)

            if result.get("success"):
                for cam, res in result.get("camera_results", {}).items():
                    print(f"  Camera {cam}: {res.get('transformed_files', 0)} files")
            else:
                print(f"  FAILED")
        except Exception as e:
            print(f"  FAILED - {e}")
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} paths succeeded")
    print("\nNOTE: Statistics files were NOT transformed. Recalculate if needed.")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# MERGE COMMAND
# =============================================================================

def merge_command(args):
    """Merge multi-camera vector fields using Hanning blend."""
    from pivtools_core.config import get_config
    from pivtools_gui.vector_merging.vector_merger import VectorMerger

    config = get_config()

    # Apply CLI overrides
    if args.cameras:
        cameras = [int(c) for c in args.cameras.split(",")]
    else:
        cameras = config.merging_cameras or config.camera_numbers
    type_name = args.type_name or config.merging_type_name or "instantaneous"

    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    if len(cameras) < 2:
        print("Error: Merging requires at least 2 cameras")
        sys.exit(1)

    print("=" * 60)
    print("Vector Merging - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")
    print(f"Cameras: {cameras}")
    print(f"Type: {type_name}")

    results = []
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx + 1}/{len(active_paths)}: {base_dir}")
        print("-" * 40)

        try:
            merger = VectorMerger(
                base_dir=base_dir,
                cameras=cameras,
                type_name=type_name,
            )
            result = merger.merge_all_frames()
            result["path_idx"] = path_idx
            results.append(result)

            if result.get("success"):
                print(f"  Merged {result.get('processed_count', 0)} frames")
                print(f"  Output: {result.get('output_dir', '')}")
            else:
                print(f"  FAILED - {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"  FAILED - {e}")
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} paths succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# STATISTICS COMMAND
# =============================================================================

def statistics_command(args):
    """Compute PIV statistics (mean, Reynolds stresses, TKE, vorticity, etc.)."""
    from pivtools_core.config import get_config
    from pivtools_core.paths import get_data_paths
    from pivtools_gui.vector_statistics.instantaneous_statistics import VectorStatisticsProcessor

    config = get_config()

    # Apply CLI overrides
    cameras = [args.camera] if args.camera else config.camera_numbers
    type_name = args.type_name or "instantaneous"

    # Determine workflow and source_endpoint
    # Priority: new flags > legacy flags > config
    source_endpoint = args.source_endpoint
    workflow = args.workflow

    # Handle legacy flags for backward compatibility
    if source_endpoint is None:
        if args.stereo:
            source_endpoint = "stereo"
        elif args.merged:
            source_endpoint = "merged"

    if workflow is None:
        if source_endpoint == "stereo":
            workflow = "stereo"
        elif source_endpoint == "merged":
            workflow = "after_merge"

    # Determine final flags based on source_endpoint/workflow
    use_stereo = source_endpoint == "stereo" or workflow == "stereo"
    use_merged = source_endpoint == "merged" or workflow == "after_merge"

    if use_stereo:
        # Update config to persist stereo workflow
        config.data.setdefault("statistics", {})["workflow"] = "stereo"
        config.data["statistics"]["source_endpoint"] = "stereo"
        config.save()
        print("Updated config: statistics.workflow = stereo, statistics.source_endpoint = stereo")
    elif use_merged:
        # Update config to persist the workflow
        config.data.setdefault("statistics", {})["workflow"] = "after_merge"
        config.data["statistics"]["source_endpoint"] = "merged"
        config.save()
        print("Updated config: statistics.workflow = after_merge, statistics.source_endpoint = merged")
    else:
        # Fall back to config workflow
        use_merged = config.statistics_workflow == "after_merge"

    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Vector Statistics - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")
    print(f"Cameras: {cameras}")
    print(f"Type: {type_name}")
    print(f"Source endpoint: {source_endpoint or 'regular'}")
    print(f"Workflow: {workflow or 'per_camera'}")

    # Get required config values
    num_frame_pairs = config.num_frame_pairs
    vector_format = config.vector_format[0] if isinstance(config.vector_format, list) else config.vector_format
    gamma_radius = config.statistics_gamma_radius

    results = []
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx + 1}/{len(active_paths)}: {base_dir}")
        print("-" * 40)

        # Determine targets based on workflow
        if use_stereo:
            # Stereo: single combined 3D result
            stereo_pairs = config.stereo_pairs
            cam_pair = stereo_pairs[0] if stereo_pairs else (config.camera_numbers[0], config.camera_numbers[1] if len(config.camera_numbers) > 1 else 2)
            targets = [("stereo", cam_pair)]
        elif use_merged:
            targets = [("merged", None)]
        else:
            targets = [("camera", cam) for cam in cameras]

        for target_type, target_value in targets:
            try:
                # Construct data_dir based on target type
                if target_type == "stereo":
                    cam_pair = target_value
                    paths = get_data_paths(
                        base_dir=base_dir,
                        num_frame_pairs=num_frame_pairs,
                        cam=cam_pair[0],
                        type_name=type_name,
                        use_stereo=True,
                        stereo_camera_pair=cam_pair
                    )
                    data_dir = paths["data_dir"]
                    label = f"Stereo Cam{cam_pair[0]}_Cam{cam_pair[1]}"
                    processor = VectorStatisticsProcessor(
                        data_dir=data_dir,
                        base_dir=base_dir,
                        num_frame_pairs=num_frame_pairs,
                        vector_format=vector_format,
                        type_name=type_name,
                        use_merged=False,
                        use_stereo=True,
                        stereo_camera_pair=cam_pair,
                        camera=cam_pair[0],
                        gamma_radius=gamma_radius,
                        config=config,
                    )
                elif target_type == "merged":
                    data_dir = base_dir / "calibrated_piv" / str(num_frame_pairs) / "Merged" / type_name
                    label = "Merged"
                    processor = VectorStatisticsProcessor(
                        data_dir=data_dir,
                        base_dir=base_dir,
                        num_frame_pairs=num_frame_pairs,
                        vector_format=vector_format,
                        type_name=type_name,
                        use_merged=True,
                        camera=1,
                        gamma_radius=gamma_radius,
                        config=config,
                    )
                else:  # camera
                    cam = target_value
                    data_dir = base_dir / "calibrated_piv" / str(num_frame_pairs) / f"Cam{cam}" / type_name
                    label = f"Camera {cam}"
                    processor = VectorStatisticsProcessor(
                        data_dir=data_dir,
                        base_dir=base_dir,
                        num_frame_pairs=num_frame_pairs,
                        vector_format=vector_format,
                        type_name=type_name,
                        use_merged=False,
                        camera=cam,
                        gamma_radius=gamma_radius,
                        config=config,
                    )
                result = processor.process()
                result["path_idx"] = path_idx
                result["target"] = label
                results.append(result)

                if result.get("success"):
                    print(f"  {label}: OK")
                else:
                    print(f"  {label}: FAILED - {result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"  {label}: FAILED - {e}")
                results.append({"success": False, "error": str(e), "path_idx": path_idx, "target": label})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} operations succeeded")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# VIDEO COMMAND
# =============================================================================

def video_command(args):
    """Create visualization video from PIV data."""
    from pivtools_core.config import get_config
    from pivtools_gui.video_maker.video_maker import VideoMaker

    config = get_config()

    # Apply CLI overrides (use config values as defaults)
    camera = args.camera if args.camera else config.video_camera
    variable = args.variable if args.variable else config.video_variable
    run = args.run if args.run else config.video_run
    data_source = args.data_source if args.data_source else config.video_data_source
    fps = args.fps if args.fps else config.video_fps
    crf = args.crf if args.crf else config.video_crf
    test_mode = args.test if args.test else False

    # Parse resolution
    if args.resolution:
        parts = args.resolution.lower().split("x")
        if len(parts) == 2:
            resolution = (int(parts[1]), int(parts[0]))  # (height, width)
        elif args.resolution == "4k":
            resolution = (2160, 3840)
        else:
            resolution = (1080, 1920)
    else:
        resolution = config.video_resolution

    # Parse color limits
    lower_limit = args.lower if args.lower is not None else config.video_lower_limit
    upper_limit = args.upper if args.upper is not None else config.video_upper_limit
    cmap = args.cmap if args.cmap else (config.video_cmap if config.video_cmap != "default" else None)

    active_paths = get_active_paths_from_args(args, config)
    if not active_paths:
        print("Error: No active paths configured in config.yaml")
        sys.exit(1)

    print("=" * 60)
    print("Video Creation - Starting")
    print("=" * 60)
    print(f"Active paths: {len(active_paths)}")
    print(f"Camera: {camera}")
    print(f"Variable: {variable}")
    print(f"Run: {run}")
    print(f"Data source: {data_source}")
    print(f"FPS: {fps}, CRF: {crf}")
    print(f"Resolution: {resolution[1]}x{resolution[0]}")
    print(f"Test mode: {test_mode}")

    results = []
    for path_idx in active_paths:
        base_dir = Path(config.base_paths[path_idx])
        print(f"\nPath {path_idx + 1}/{len(active_paths)}: {base_dir}")
        print("-" * 40)

        try:
            maker = VideoMaker(
                base_dir=base_dir,
                camera=camera,
                type_name="instantaneous",  # Video always from instantaneous
                config=config,  # Pass config for stereo pair access
            )

            result = maker.create_video(
                variable=variable,
                run=run,
                fps=fps,
                crf=crf,
                resolution=resolution,
                cmap=cmap,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                test_mode=test_mode,
                test_frames=50 if test_mode else None,
                data_source=data_source,
            )
            result["path_idx"] = path_idx
            results.append(result)

            if result.get("success"):
                print(f"  Created: {result.get('out_path', '')}")
                print(f"  Frames: {result.get('frames', 0)}, Time: {result.get('elapsed_sec', 0):.1f}s")
            else:
                print(f"  FAILED - {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"  FAILED - {e}")
            results.append({"success": False, "error": str(e), "path_idx": path_idx})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r.get("success"))
    print(f"Total: {success_count}/{len(results)} videos created")

    sys.exit(0 if success_count == len(results) else 1)


# =============================================================================
# INIT COMMAND
# =============================================================================

def init_command(args):
    """Initialize a new PIVTOOLs workspace with default config.yaml"""
    cwd = Path.cwd()

    # Check if config.yaml already exists
    config_path = cwd / "config.yaml"
    if config_path.exists():
        if not args.force:
            print(f"config.yaml already exists at {config_path}")
            print("Use --force to overwrite")
            return
        else:
            print(f"Overwriting existing config.yaml")

    # Get the default config from package
    try:
        import pivtools_core
        default_config = Path(pivtools_core.__file__).parent / "config.yaml"

        if not default_config.exists():
            # Fallback: create a basic config
            create_default_config(config_path)
        else:
            shutil.copy2(default_config, config_path)
            print(f"Created config.yaml at {config_path}")

    except ImportError:
        # Fallback if package not properly installed
        create_default_config(config_path)

    print("PIVTOOLs workspace initialized!")
    print(f"Edit {config_path} to configure your PIV analysis")

def create_default_config(config_path):
    """Create a default config.yaml file"""
    default_config = """
paths:
  base_paths:
  - /setme
  source_paths:
  - /setme
  active_paths:
  - 0
  camera_numbers:
  - 1
  camera_count: 1
  camera_subfolders: []
images:
  num_images: 100
  image_format:
  - B%05d_A.tif
  - B%05d_B.tif
  vector_format:
  - '%05d.mat'
  dtype: float32
  start_index: 1
  frame_stride: 0
  pair_stride: 1
  pairing_preset: ab_format
  image_type: standard
  use_camera_subfolders: false
batches:
  size: 25
logging:
  file: pypiv.log
  level: INFO
  console: true
processing:
  backend: cpu
  debug: false
  auto_compute_params: false
  omp_threads: 4
  dask_workers_per_node: 2
  dask_memory_limit: 8GB
  dask_max_in_flight_per_worker: 3
  cluster_type: local
  open_dashboard: false
  dask_nanny: false
  post_processing_workers: null
  always_batch: true
  instantaneous: true
  ensemble: false
outlier_detection:
  enabled: true
  methods:
  - threshold: 0.2
    type: peak_mag
  - epsilon: 0.2
    threshold: 2
    type: median_2d
infilling:
  mid_pass:
    method: local_median
    parameters: {}
  final_pass:
    enabled: true
    method: local_median
    parameters: {}
ensemble_outlier_detection:
  enabled: true
  methods:
  - epsilon: 0.2
    threshold: 2
    type: median_2d
ensemble_infilling:
  mid_pass:
    method: local_median
    parameters: {}
  final_pass:
    enabled: true
    method: local_median
    parameters: {}
plots:
  save_extension: .png
  save_pickle: true
  fontsize: 14
  title_fontsize: 16
video:
  base_path_idx: 0
  camera: 1
  data_source: calibrated
  variable: ux
  run: 1
  piv_type: instantaneous
  cmap: viridis
  source_endpoint: regular
  lower: ''
  upper: ''
  fps: 30
  crf: 15
  resolution: 1080p
statistics:
  enabled_methods:
    mean_velocity: true
    mean_stresses: true
    mean_tke: true
    mean_vorticity: true
    mean_divergence: true
    mean_peak_height: false
    inst_velocity: true
    inst_stresses: true
    inst_vorticity: true
    inst_divergence: true
    inst_gamma: true
  gamma_radius: 5
  save_figures: true
  type_name: instantaneous
  source_endpoint: regular
instantaneous_piv:
  window_size:
  - - 128
    - 128
  - - 64
    - 64
  - - 32
    - 32
  - - 16
    - 16
  overlap:
  - 50
  - 50
  - 50
  - 50
  runs:
  - 3
  - 4
  time_resolved: false
  window_type: gaussian
  num_peaks: 1
  peak_finder: gauss6
  secondary_peak: false
  predictor_smoothing: true
  image_warp_interpolation: cubic
  save_mode: minimal
  save_compression: false
ensemble_piv:
  fit_method: kspace
  skip_background_subtraction: false
  image_warp_interpolation: cubic
  predictor_interpolation: cubic
  kspace_k_max_cap: 0.35
  kspace_soft_weighting: true
  fit_offset: true
  background_subtraction_method: correlation
  gradient_correction: false
  mask_center_pixel: true
  window_size:
  - - 128
    - 128
  - - 64
    - 64
  - - 32
    - 32
  overlap:
  - 50
  - 50
  - 50
  type:
  - std
  - std
  - std
  runs:
  - 1
  - 2
  - 3
  store_planes: false
  save_diagnostics: false
  sum_window:
  - 32
  - 32
  sum_fitting_window_enabled: true
  sum_fitting_window:
  - 32
  - 32
  resume_from_pass: 0
  predictor_smoothing: false
  predictor_boundary_conditions: []
  persist_images: false
  window_type: square
calibration:
  image_format: calib_%02d.tif
  num_images: 1
  image_type: standard
  zero_based_indexing: false
  use_camera_subfolders: false
  calibration_sources: []
  camera_subfolders: []
  global_coordinates:
    enabled: false
    datum_camera: 1
    datum_pixel: null
    datum_physical: [0.0, 0.0]
    datum_frame: 1
    overlap_pairs: []
    invert_ux: false
  active: scale_factor
  piv_type: instantaneous
  scale_factor:
    dt: 1
    px_per_mm: 1
    source_path_idx: 0
  dotboard:
    camera: 1
    dot_spacing_mm: 1
    dt: 1
    model_type: pinhole
    source_path_idx: 0
  charuco:
    camera: 1
    squares_h: 10
    squares_v: 9
    square_size: 0.03
    marker_ratio: 0.5
    aruco_dict: DICT_4X4_1000
    min_corners: 6
    dt: 1
    model_type: pinhole
    source_path_idx: 0
  stereo_dotboard:
    camera_pair:
    - 1
    - 2
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 1
    dt: 1
    stereo_model_type: dotboard
  polynomial:
    xml_path: ''
    use_xml: true
    dt: 1
    source_path_idx: 0
  stereo_charuco:
    camera_pair:
    - 1
    - 2
    squares_h: 10
    squares_v: 9
    square_size: 0.03
    marker_ratio: 0.5
    aruco_dict: DICT_4X4_1000
    min_corners: 6
    dt: 1
  self_calibration:
    n_images: 20
    window_size: 64
    overlap: 50.0
filters: []
masking:
  enabled: false
  mask_file_pattern: mask_Cam%d.mat
  mask_threshold: 0.01
  mode: file
  rectangular:
    top: 0
    bottom: 0
    left: 0
    right: 0
merging:
  type_name: instantaneous
  base_path_idx: 0
transforms:
  base_path_idx: 0
  type_name: instantaneous
  cameras: {}
  source_endpoint: regular

"""

    with open(config_path, 'w') as f:
        f.write(default_config.strip())
    print(f"Created default config.yaml at {config_path}")

def instantaneous_command(args):
    """Run instantaneous PIV processing."""
    import os
    from pivtools_core import instantaneous

    if args.active_paths:
        os.environ['PIV_ACTIVE_PATHS'] = args.active_paths

    print("=" * 60)
    print("Instantaneous PIV Processing")
    print("=" * 60)
    if args.active_paths:
        print(f"Active paths override: {args.active_paths}")

    try:
        instantaneous.main()
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def ensemble_command(args):
    """Run ensemble PIV processing."""
    import os
    from pivtools_core import ensemble

    if args.active_paths:
        os.environ['PIV_ACTIVE_PATHS'] = args.active_paths

    print("=" * 60)
    print("Ensemble PIV Processing")
    print("=" * 60)
    if args.active_paths:
        print(f"Active paths override: {args.active_paths}")

    try:
        ensemble.main()
    except SystemExit as e:
        sys.exit(e.code if e.code is not None else 0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="PIVTOOLs - Particle Image Velocimetry Tools",
        prog="pivtools-cli"
    )
    import logging
    logging.info("Starting PIVTOOLs CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new PIVTOOLs workspace with default config.yaml"
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing config.yaml"
    )
    init_parser.set_defaults(func=init_command)

    # instantaneous command
    instantaneous_parser = subparsers.add_parser(
        "instantaneous",
        help="Run instantaneous PIV processing"
    )
    instantaneous_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    instantaneous_parser.set_defaults(func=instantaneous_command)

    # ensemble command
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Run ensemble PIV processing"
    )
    ensemble_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    ensemble_parser.set_defaults(func=ensemble_command)

    # detect-planar command (single camera)
    detect_planar_parser = subparsers.add_parser(
        "detect-planar",
        help="Detect dot/circle grid and generate camera model"
    )
    detect_planar_parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera number to process (default: all from config)"
    )
    detect_planar_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    detect_planar_parser.add_argument(
        "--calibration-source", "-cs", default=None,
        help="Direct path to calibration images (overrides config.calibration_sources)"
    )
    detect_planar_parser.add_argument(
        "--model-type", default="pinhole",
        choices=["pinhole", "polynomial"],
        help="Calibration model type: pinhole (OpenCV) or polynomial (DaVis-compatible)"
    )
    detect_planar_parser.set_defaults(func=detect_planar_command)

    # detect-charuco command (single camera)
    detect_charuco_parser = subparsers.add_parser(
        "detect-charuco",
        help="Detect ChArUco board and generate camera model"
    )
    detect_charuco_parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera number to process (default: all from config)"
    )
    detect_charuco_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    detect_charuco_parser.add_argument(
        "--calibration-source", "-cs", default=None,
        help="Direct path to calibration images (overrides config.calibration_sources)"
    )
    detect_charuco_parser.add_argument(
        "--model-type", default="pinhole",
        choices=["pinhole", "polynomial"],
        help="Calibration model type: pinhole (OpenCV) or polynomial (DaVis-compatible)"
    )
    detect_charuco_parser.set_defaults(func=detect_charuco_command)

    # detect-stereo-planar command
    detect_stereo_planar_parser = subparsers.add_parser(
        "detect-stereo-planar",
        help="Detect dot/circle grid and generate stereo camera model"
    )
    detect_stereo_planar_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    detect_stereo_planar_parser.add_argument(
        "--calibration-source", "-cs", default=None,
        help="Direct path to calibration images (overrides config.calibration_sources)"
    )
    detect_stereo_planar_parser.set_defaults(func=detect_stereo_planar_command)

    # detect-stereo-charuco command
    detect_stereo_charuco_parser = subparsers.add_parser(
        "detect-stereo-charuco",
        help="Detect ChArUco board and generate stereo camera model"
    )
    detect_stereo_charuco_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    detect_stereo_charuco_parser.add_argument(
        "--calibration-source", "-cs", default=None,
        help="Direct path to calibration images (overrides config.calibration_sources)"
    )
    detect_stereo_charuco_parser.set_defaults(func=detect_stereo_charuco_command)

    # apply-calibration command
    apply_calibration_parser = subparsers.add_parser(
        "apply-calibration",
        help="Apply calibration to PIV vectors (pixels to m/s)"
    )
    apply_calibration_parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera number to process (default: all from config)"
    )
    apply_calibration_parser.add_argument(
        "--type-name", "-t", default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: instantaneous)"
    )
    apply_calibration_parser.add_argument(
        "--runs", "-r", default=None,
        help="Comma-separated run numbers to process (default: all)"
    )
    apply_calibration_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    apply_calibration_parser.add_argument(
        "--method", "-m", default=None,
        choices=["dotboard", "charuco", "scale_factor", "polynomial"],
        help="Calibration method (default: from config.yaml calibration.active)"
    )
    apply_calibration_parser.add_argument(
        "--align-coordinates", action="store_true",
        help="Apply global coordinate alignment after calibration (reads datum/overlap from config.yaml)"
    )
    apply_calibration_parser.set_defaults(func=apply_calibration_command)

    # apply-stereo command
    apply_stereo_parser = subparsers.add_parser(
        "apply-stereo",
        help="Apply stereo calibration for 3D velocity reconstruction (ux, uy, uz)"
    )
    apply_stereo_parser.add_argument(
        "--method", "-m", default=None,
        choices=["dotboard", "charuco"],
        help="Stereo calibration method (default: from config stereo_dotboard.stereo_model_type)"
    )
    apply_stereo_parser.add_argument(
        "--camera-pair", "-c", default=None,
        help="Camera pair as 'CAM1,CAM2' (e.g., '1,2'). Default: from config"
    )
    apply_stereo_parser.add_argument(
        "--type-name", "-t", default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: instantaneous)"
    )
    apply_stereo_parser.add_argument(
        "--runs", "-r", default=None,
        help="Comma-separated run numbers to process (default: all)"
    )
    apply_stereo_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    apply_stereo_parser.set_defaults(func=apply_stereo_command)

    # transform command
    transform_parser = subparsers.add_parser(
        "transform",
        help="Apply geometric transforms to PIV vector fields"
    )
    transform_parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera number (default: all from config)"
    )
    transform_parser.add_argument(
        "--type-name", "-t", default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: from config or instantaneous)"
    )
    transform_parser.add_argument(
        "--operations", "-o", default=None,
        help="Comma-separated transforms: flip_ud,flip_lr,rotate_90_cw,rotate_90_ccw,rotate_180"
    )
    transform_parser.add_argument(
        "--merged", "-m", action="store_true",
        help="Transform merged data instead of per-camera (deprecated: use --source-endpoint merged)"
    )
    transform_parser.add_argument(
        "--source-endpoint", "-s", default=None,
        choices=["regular", "merged", "stereo"],
        help="Data source: regular (per-camera), merged, or stereo"
    )
    transform_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    transform_parser.set_defaults(func=transform_command)

    # merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multi-camera vector fields using Hanning blend"
    )
    merge_parser.add_argument(
        "--cameras", "-c", default=None,
        help="Comma-separated camera numbers to merge (default: from config)"
    )
    merge_parser.add_argument(
        "--type-name", "-t", default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: from config or instantaneous)"
    )
    merge_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    merge_parser.set_defaults(func=merge_command)

    # statistics command
    statistics_parser = subparsers.add_parser(
        "statistics",
        help="Compute PIV statistics (mean, Reynolds stresses, TKE, vorticity)"
    )
    statistics_parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera number to process (default: all from config)"
    )
    statistics_parser.add_argument(
        "--type-name", "-t", default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: instantaneous)"
    )
    statistics_parser.add_argument(
        "--merged", "-m", action="store_true",
        help="Process merged data instead of per-camera"
    )
    statistics_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    statistics_parser.add_argument(
        "--stereo", action="store_true",
        help="Process stereo PIV data (deprecated: use --source-endpoint stereo)"
    )
    statistics_parser.add_argument(
        "--source-endpoint", "-s", default=None,
        choices=["regular", "merged", "stereo"],
        help="Data source: regular (per-camera), merged, or stereo"
    )
    statistics_parser.add_argument(
        "--workflow", "-w", default=None,
        choices=["per_camera", "after_merge", "both", "stereo"],
        help="Workflow: per_camera, after_merge, both, or stereo"
    )
    statistics_parser.set_defaults(func=statistics_command)

    # video command
    video_parser = subparsers.add_parser(
        "video",
        help="Create visualization video from PIV data"
    )
    video_parser.add_argument(
        "--camera", "-c", type=int, default=None,
        help="Camera number (default: from config)"
    )
    video_parser.add_argument(
        "--variable", "-v", default=None,
        help="Variable to visualize: ux, uy, uz, mag, vorticity, divergence, u_prime, etc."
    )
    video_parser.add_argument(
        "--run", "-r", type=int, default=None,
        help="Run number (default: 1)"
    )
    video_parser.add_argument(
        "--data-source", "-d", default=None,
        choices=["calibrated", "uncalibrated", "merged", "stereo", "inst_stats"],
        help="Data source (default: calibrated)"
    )
    video_parser.add_argument(
        "--fps", type=int, default=None,
        help="Frame rate (default: 30)"
    )
    video_parser.add_argument(
        "--crf", type=int, default=None,
        help="Video quality 0-51, lower=better (default: 15)"
    )
    video_parser.add_argument(
        "--resolution", default=None,
        help="Output resolution: WIDTHxHEIGHT or '4k' (default: 1920x1080)"
    )
    video_parser.add_argument(
        "--cmap", default=None,
        help="Colormap name (default: auto)"
    )
    video_parser.add_argument(
        "--lower", type=float, default=None,
        help="Lower color limit (default: auto)"
    )
    video_parser.add_argument(
        "--upper", type=float, default=None,
        help="Upper color limit (default: auto)"
    )
    video_parser.add_argument(
        "--test", action="store_true",
        help="Test mode: only process 50 frames"
    )
    video_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    video_parser.set_defaults(func=video_command)

    # align-coordinates command
    align_parser = subparsers.add_parser(
        "align-coordinates",
        help="Apply global coordinate alignment to calibrated vectors"
    )
    align_parser.add_argument(
        "--type-name", "-t", default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: instantaneous)"
    )
    align_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')"
    )
    align_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force alignment even if already applied (WARNING: will double shifts)"
    )
    align_parser.set_defaults(func=align_coordinates_command)

    # self-calibrate command
    selfcal_parser = subparsers.add_parser(
        "self-calibrate",
        help="Run stereo self-calibration (Wieneke 2005) to correct laser sheet misalignment"
    )
    selfcal_parser.add_argument(
        "--camera-pair", "-c", default=None,
        help="Camera pair as 'CAM1,CAM2' (e.g., '1,2'). Default: from config"
    )
    selfcal_parser.add_argument(
        "--method", "-m", default=None,
        choices=["dotboard", "charuco"],
        help="Calibration method (default: from config)"
    )
    selfcal_parser.add_argument(
        "--n-images", "-n", type=int, default=None,
        help="Number of source images for self-cal (default: 20)"
    )
    selfcal_parser.add_argument(
        "--window-size", "-w", type=int, default=None,
        help="Correlation window size in pixels (default: 64)"
    )
    selfcal_parser.add_argument(
        "--active-paths", "-p", default=None,
        help="Source path index (e.g., '0')"
    )
    selfcal_parser.set_defaults(func=self_calibrate_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)

if __name__ == "__main__":
    main()