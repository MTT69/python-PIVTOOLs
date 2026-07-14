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
    if hasattr(args, "active_paths") and args.active_paths:
        # Parse comma-separated indices
        return [int(i.strip()) for i in args.active_paths.split(",")]
    return config.active_paths


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
            print(
                "Error: No transforms configured. Use --operations or set transforms.cameras in config.yaml"
            )
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
        config.data.setdefault("transforms", {})["source_endpoint"] = (
            source_endpoint or "regular"
        )
        config.save()
        print(
            f"Updated config: transforms.source_endpoint = {source_endpoint or 'regular'}"
        )

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
                print("  FAILED")
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
    from pivtools_gui.vector_statistics.instantaneous_statistics import (
        VectorStatisticsProcessor,
    )

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
        print(
            "Updated config: statistics.workflow = stereo, statistics.source_endpoint = stereo"
        )
    elif use_merged:
        # Update config to persist the workflow
        config.data.setdefault("statistics", {})["workflow"] = "after_merge"
        config.data["statistics"]["source_endpoint"] = "merged"
        config.save()
        print(
            "Updated config: statistics.workflow = after_merge, statistics.source_endpoint = merged"
        )
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
    vector_format = (
        config.vector_format[0]
        if isinstance(config.vector_format, list)
        else config.vector_format
    )
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
            cam_pair = (
                stereo_pairs[0]
                if stereo_pairs
                else (
                    config.camera_numbers[0],
                    config.camera_numbers[1] if len(config.camera_numbers) > 1 else 2,
                )
            )
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
                        stereo_camera_pair=cam_pair,
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
                    data_dir = (
                        base_dir
                        / "calibrated_piv"
                        / str(num_frame_pairs)
                        / "Merged"
                        / type_name
                    )
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
                    data_dir = (
                        base_dir
                        / "calibrated_piv"
                        / str(num_frame_pairs)
                        / f"Cam{cam}"
                        / type_name
                    )
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
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "path_idx": path_idx,
                        "target": label,
                    }
                )

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
    cmap = (
        args.cmap
        if args.cmap
        else (config.video_cmap if config.video_cmap != "default" else None)
    )

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
                print(
                    f"  Frames: {result.get('frames', 0)}, Time: {result.get('elapsed_sec', 0):.1f}s"
                )
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
            print("Overwriting existing config.yaml")

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
  size: 10
logging:
  file: pypiv.log
  level: INFO
  console: true
processing:
  backend: cpu
  debug: false
  auto_compute_params: false
  omp_threads: 4
  dask_workers_per_node: 1
  dask_memory_limit: 12GB
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
    size: 5
    type: median_2d
infilling:
  mid_pass:
    method: nearest
    parameters: {}
  final_pass:
    enabled: true
    method: biharmonic
    parameters: {}
ensemble_outlier_detection:
  enabled: true
  methods:
  - epsilon: 0.2
    threshold: 2
    size: 5
    type: median_2d
ensemble_infilling:
  mid_pass:
    method: nearest
    parameters: {}
  final_pass:
    enabled: true
    method: biharmonic
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
  dump_correlation_planes: false
ensemble_piv:
  fit_method: kspace
  skip_background_subtraction: false
  image_warp_interpolation: cubic
  background_subtraction_method: correlation
  per_pair_normalization: false
  predictor_rounding: false
  gradient_correction: false
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
  # Single unified calibration block (pinhole-based). RIG-SPECIFIC placeholders below.
  # --- Image sourcing (consumed by the calibration image loaders) ---
  calibration_sources: []
  image_format: calib%05d.png
  image_type: standard
  zero_based_indexing: false
  use_camera_subfolders: false
  camera_subfolders: []
  piv_type: instantaneous
  # --- Calibration math ---
  active: charuco             # charuco | dotboard | scale_factor (stereo chosen per command)
  source: ''                 # calibration image dir; '' -> calibration_sources[source_idx]
  source_idx: 0
  n_views: 10
  start_index: 1
  datum_index: 0
  distortion_model: standard # standard | rational | tilted
  fix_aspect_ratio: true
  use_release_object: false
  world_frame: default       # 'default' or path to a clicks JSON {origin,x_axis,y_axis}
  camera: 1
  camera_pair:
  - 1
  - 2
  dt: 1                      # seconds between frames — RIG-SPECIFIC, no safe default
  z_world: 0.0              # light-sheet plane Z (mm) + tilt (rad), applied at apply time
  tilt_x: 0.0
  tilt_y: 0.0
  interpolator: lanczos     # stereo cam2 resample kernel: linear | cubic | lanczos
  charuco:
    squares_h: 10
    squares_v: 7
    square_size: 0.03
    marker_ratio: 0.5
    aruco_dict: DICT_4X4_1000
    min_corners: 6
    model_type: pinhole      # pinhole | polynomial (polynomial is planar-only)
  dotboard:
    dot_spacing_mm: 15.0
    k_neighbors: 9
    model_type: pinhole
    fix_k2: false            # pin r⁴ radial term (CALIB_FIX_K2); set true only for few-view (<3) fits
  scale_factor:
    px_per_mm: 1
    dt: 1
  global_coordinates:
    enabled: false
    datum_camera: 1
    datum_pixel: null
    datum_physical: [0.0, 0.0]
    datum_frame: 1
    overlap_pairs: []
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

    # Explicit UTF-8: the template holds superscripts (e.g. r⁴) that Windows'
    # default cp1252 codec cannot encode.
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(default_config.strip())
    print(f"Created default config.yaml at {config_path}")


def instantaneous_command(args):
    """Run instantaneous PIV processing."""
    from pivtools_core import instantaneous

    if args.active_paths:
        os.environ["PIV_ACTIVE_PATHS"] = args.active_paths
    # instantaneous/ensemble delegate to main(), which reads config.camera_numbers,
    # so the camera override is passed via env var (statistics/transform own their
    # loop and build a local list instead). Use `is not None` so an explicit bad
    # value like 0 reaches the property and fails loud; pop a stale value when the
    # flag is absent so a missing flag always means "all cameras".
    if args.camera is not None:
        os.environ["PIV_CAMERA"] = str(args.camera)
    else:
        os.environ.pop("PIV_CAMERA", None)

    print("=" * 60)
    print("Instantaneous PIV Processing")
    print("=" * 60)
    if args.active_paths:
        print(f"Active paths override: {args.active_paths}")
    if args.camera is not None:
        print(f"Camera override: {args.camera}")

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
    from pivtools_core import ensemble

    if args.active_paths:
        os.environ["PIV_ACTIVE_PATHS"] = args.active_paths
    # instantaneous/ensemble delegate to main(), which reads config.camera_numbers,
    # so the camera override is passed via env var (statistics/transform own their
    # loop and build a local list instead). Use `is not None` so an explicit bad
    # value like 0 reaches the property and fails loud; pop a stale value when the
    # flag is absent so a missing flag always means "all cameras".
    if args.camera is not None:
        os.environ["PIV_CAMERA"] = str(args.camera)
    else:
        os.environ.pop("PIV_CAMERA", None)

    print("=" * 60)
    print("Ensemble PIV Processing")
    print("=" * 60)
    if args.active_paths:
        print(f"Active paths override: {args.active_paths}")
    if args.camera is not None:
        print(f"Camera override: {args.camera}")

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
        description="PIVTOOLs - Particle Image Velocimetry Tools", prog="pivtools-cli"
    )
    import logging

    logging.info("Starting PIVTOOLs CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new PIVTOOLs workspace with default config.yaml"
    )
    init_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing config.yaml"
    )
    init_parser.set_defaults(func=init_command)

    # instantaneous command
    instantaneous_parser = subparsers.add_parser(
        "instantaneous", help="Run instantaneous PIV processing"
    )
    instantaneous_parser.add_argument(
        "--active-paths",
        "-p",
        default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')",
    )
    instantaneous_parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera number to process (default: all from config)",
    )
    instantaneous_parser.set_defaults(func=instantaneous_command)

    # ensemble command
    ensemble_parser = subparsers.add_parser(
        "ensemble", help="Run ensemble PIV processing"
    )
    ensemble_parser.add_argument(
        "--active-paths",
        "-p",
        default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')",
    )
    ensemble_parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera number to process (default: all from config)",
    )
    ensemble_parser.set_defaults(func=ensemble_command)

    # transform command
    transform_parser = subparsers.add_parser(
        "transform", help="Apply geometric transforms to PIV vector fields"
    )
    transform_parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera number (default: all from config)",
    )
    transform_parser.add_argument(
        "--type-name",
        "-t",
        default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: from config or instantaneous)",
    )
    transform_parser.add_argument(
        "--operations",
        "-o",
        default=None,
        help="Comma-separated transforms: flip_ud,flip_lr,rotate_90_cw,rotate_90_ccw,rotate_180",
    )
    transform_parser.add_argument(
        "--merged",
        "-m",
        action="store_true",
        help="Transform merged data instead of per-camera (deprecated: use --source-endpoint merged)",
    )
    transform_parser.add_argument(
        "--source-endpoint",
        "-s",
        default=None,
        choices=["regular", "merged", "stereo"],
        help="Data source: regular (per-camera), merged, or stereo",
    )
    transform_parser.add_argument(
        "--active-paths",
        "-p",
        default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')",
    )
    transform_parser.set_defaults(func=transform_command)

    # merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge multi-camera vector fields using Hanning blend"
    )
    merge_parser.add_argument(
        "--cameras",
        "-c",
        default=None,
        help="Comma-separated camera numbers to merge (default: from config)",
    )
    merge_parser.add_argument(
        "--type-name",
        "-t",
        default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: from config or instantaneous)",
    )
    merge_parser.add_argument(
        "--active-paths",
        "-p",
        default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')",
    )
    merge_parser.set_defaults(func=merge_command)

    # statistics command
    statistics_parser = subparsers.add_parser(
        "statistics",
        help="Compute PIV statistics (mean, Reynolds stresses, TKE, vorticity)",
    )
    statistics_parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera number to process (default: all from config)",
    )
    statistics_parser.add_argument(
        "--type-name",
        "-t",
        default=None,
        choices=["instantaneous", "ensemble"],
        help="Data type (default: instantaneous)",
    )
    statistics_parser.add_argument(
        "--merged",
        "-m",
        action="store_true",
        help="Process merged data instead of per-camera",
    )
    statistics_parser.add_argument(
        "--active-paths",
        "-p",
        default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')",
    )
    statistics_parser.add_argument(
        "--stereo",
        action="store_true",
        help="Process stereo PIV data (deprecated: use --source-endpoint stereo)",
    )
    statistics_parser.add_argument(
        "--source-endpoint",
        "-s",
        default=None,
        choices=["regular", "merged", "stereo"],
        help="Data source: regular (per-camera), merged, or stereo",
    )
    statistics_parser.add_argument(
        "--workflow",
        "-w",
        default=None,
        choices=["per_camera", "after_merge", "both", "stereo"],
        help="Workflow: per_camera, after_merge, both, or stereo",
    )
    statistics_parser.set_defaults(func=statistics_command)

    # video command
    video_parser = subparsers.add_parser(
        "video", help="Create visualization video from PIV data"
    )
    video_parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=None,
        help="Camera number (default: from config)",
    )
    video_parser.add_argument(
        "--variable",
        "-v",
        default=None,
        help="Variable to visualize: ux, uy, uz, mag, vorticity, divergence, u_prime, etc.",
    )
    video_parser.add_argument(
        "--run", "-r", type=int, default=None, help="Run number (default: 1)"
    )
    video_parser.add_argument(
        "--data-source",
        "-d",
        default=None,
        choices=["calibrated", "uncalibrated", "merged", "stereo", "inst_stats"],
        help="Data source (default: calibrated)",
    )
    video_parser.add_argument(
        "--fps", type=int, default=None, help="Frame rate (default: 30)"
    )
    video_parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Video quality 0-51, lower=better (default: 15)",
    )
    video_parser.add_argument(
        "--resolution",
        default=None,
        help="Output resolution: WIDTHxHEIGHT or '4k' (default: 1920x1080)",
    )
    video_parser.add_argument(
        "--cmap", default=None, help="Colormap name (default: auto)"
    )
    video_parser.add_argument(
        "--lower", type=float, default=None, help="Lower color limit (default: auto)"
    )
    video_parser.add_argument(
        "--upper", type=float, default=None, help="Upper color limit (default: auto)"
    )
    video_parser.add_argument(
        "--test", action="store_true", help="Test mode: only process 50 frames"
    )
    video_parser.add_argument(
        "--active-paths",
        "-p",
        default=None,
        help="Comma-separated path indices to process (e.g., '0,1,2')",
    )
    video_parser.set_defaults(func=video_command)

    # calibration (unified pinhole calibration) subcommands
    from pivtools_cli.calibration_cli import register_calibration_subparsers

    register_calibration_subparsers(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
