#!/usr/bin/env python3
"""
PIVTOOLs CLI - Command line interface for PIVTOOLs
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

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
  - ./data
  source_paths:
  - ./images
  camera_numbers:
  - 1
  camera_count: 1
images:
  num_images: 100
  image_format: B%05d.tif
  vector_format: '%05d.mat'
  time_resolved: false
  dtype: float32
batches:
  size: 25
logging:
  file: pypiv.log
  level: INFO
  console: true
processing:
  instantaneous: true
  ensemble: false
  stereo: false
  backend: cpu
  debug: false
  auto_compute_params: false
  omp_threads: 4
  dask_workers_per_node: 5
  dask_threads_per_worker: 1
  dask_memory_limit: 12GB
outlier_detection:
  enabled: true
  methods:
  - threshold: 0.25
    type: peak_mag
  - epsilon: 0.2
    threshold: 2
    type: median_2d
infilling:
  mid_pass:
    method: biharmonic
    parameters:
      ksize: 3
  final_pass:
    enabled: true
    method: biharmonic
    parameters:
      ksize: 3
plots:
  save_extension: .png
  save_pickle: true
  fontsize: 14
  title_fontsize: 16
videos:
- endpoint: ''
  type: instantaneous
  use_merged: false
  variable: ux
  video_length: 100
statistics_extraction: null
instantaneous_piv:
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
  runs:
  - 3
  time_resolved: false
  window_type: gaussian
  num_peaks: 1
  peak_finder: gauss3
  secondary_peak: false
calibration:
  image_format: calib%05d.tif
  num_images: 10
  image_type: standard
  zero_based_indexing: false
  use_camera_subfolders: false
  subfolder: ''
  active: pinhole
  scale_factor:
    dt: 0.56
    px_per_mm: 3.41
  pinhole:
    camera: 1
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    enhance_dots: true
    asymmetric: false
    grid_tolerance: 0.5
    ransac_threshold: 3
    dt: 0.0275
  charuco:
    camera: 1
    squares_h: 10
    squares_v: 9
    square_size: 0.03
    marker_ratio: 0.5
    aruco_dict: DICT_4X4_1000
    min_corners: 6
    dt: 1
  stereo:
    camera_pair:
    - 1
    - 2
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    enhance_dots: true
    asymmetric: false
    dt: 2
filters:
- type: pod
masking:
  enabled: true
  mask_file_pattern: mask_Cam%d.mat
  mask_threshold: 0.01
  mode: file
  rectangular:
    top: 64
    bottom: 64
    left: 0
    right: 0
"""

    with open(config_path, 'w') as f:
        f.write(default_config.strip())
    print(f"Created default config.yaml at {config_path}")

def run_command(args):
    """Run PIV analysis using the current config"""
    try:
        from pivtools_core.example import main as run_piv
        run_piv()
    except ImportError as e:
        print(f"Error importing PIVTOOLs: {e}")
        print("Make sure PIVTOOLs is properly installed")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="PIVTOOLs - Particle Image Velocimetry Tools",
        prog="pivtools-cli"
    )

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

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run PIV analysis using config.yaml"
    )
    run_parser.set_defaults(func=run_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)

if __name__ == "__main__":
    main()