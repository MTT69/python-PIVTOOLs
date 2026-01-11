# PIVTools CLI Reference

Complete command-line reference for PIVTools.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Commands](#commands)
   - [init](#init)
   - [instantaneous](#instantaneous)
   - [ensemble](#ensemble)
   - [detect-planar](#detect-planar)
   - [detect-charuco](#detect-charuco)
   - [detect-stereo-planar](#detect-stereo-planar)
   - [detect-stereo-charuco](#detect-stereo-charuco)
   - [apply-calibration](#apply-calibration)
   - [transform](#transform)
   - [merge](#merge)
   - [statistics](#statistics)
   - [video](#video)
4. [Workflows](#workflows)
5. [Environment Variables](#environment-variables)

---

## Overview

The PIVTools CLI provides command-line access to all PIV processing operations. It uses a subcommand pattern where each operation is a separate command.

```
pivtools-cli <command> [options]
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `init` | Initialize a new PIVTools workspace |
| `instantaneous` | Run instantaneous PIV processing |
| `ensemble` | Run ensemble PIV processing |
| `detect-planar` | Detect dot/circle grid, generate camera model |
| `detect-charuco` | Detect ChArUco board, generate camera model |
| `detect-stereo-planar` | Detect dot/circle grid, generate stereo model |
| `detect-stereo-charuco` | Detect ChArUco board, generate stereo model |
| `apply-calibration` | Apply calibration to vectors (pixels to m/s) |
| `transform` | Apply geometric transforms to vectors |
| `merge` | Merge multi-camera vector fields |
| `statistics` | Compute PIV statistics |
| `video` | Create visualization videos |

---

## Installation

```bash
pip install pivtools
```

After installation, the `pivtools-cli` command is available globally.

```bash
pivtools-cli --help
```

---

## Commands

### init

Initialize a new PIVTools workspace with a default `config.yaml` file.

**Usage:**
```bash
pivtools-cli init [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Overwrite existing config.yaml |

**Examples:**
```bash
# Initialize new workspace
pivtools-cli init

# Overwrite existing config
pivtools-cli init --force
```

**Notes:**
- Creates `config.yaml` in the current directory
- If `config.yaml` already exists, use `--force` to overwrite
- The default config includes sensible defaults for most PIV experiments

---

### instantaneous

Run instantaneous (per-frame) PIV processing using Dask-native parallelization.

**Usage:**
```bash
pivtools-cli instantaneous [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Process all active paths from config.yaml
pivtools-cli instantaneous

# Process only specific paths
pivtools-cli instantaneous -p 0,2

# Process single path
pivtools-cli instantaneous --active-paths 0
```

**Configuration (config.yaml):**
```yaml
instantaneous_piv:
  window_size:
    - [128, 128]  # Pass 1
    - [64, 64]    # Pass 2
    - [32, 32]    # Pass 3
  overlap:
    - 50  # 50% overlap
    - 50
    - 50
  runs: [3]  # Save pass 3 results
  window_type: gaussian
  peak_finder: gauss3

processing:
  backend: cpu  # or 'gpu'
  dask_workers_per_node: 5
  dask_threads_per_worker: 1
  dask_memory_limit: 12GB

paths:
  active_paths: [0, 1]  # Which path sets to process
```

**Output:**
- Vector fields saved to `{base_path}/piv/instantaneous/cam{N}/uncalibrated/run{R}/`
- Format: `.mat` files (one per frame pair)
- Includes: ux, uy, peak_mag, flags

#### Rectangular Windows

PIVTools supports **rectangular interrogation windows** where height ≠ width. This is useful when:

- Flow is predominantly in one direction (e.g., channel flow, boundary layers)
- Particle images have anisotropic seeding density
- You need higher spatial resolution in one direction while maintaining signal quality in another

**Configuration Syntax:**

Window size is specified as `[height, width]` in pixels (row-major convention):

```yaml
instantaneous_piv:
  window_size:
    - [64, 128]   # Pass 1: 64 px tall × 128 px wide (wide rectangle)
    - [32, 64]    # Pass 2: 32 px tall × 64 px wide
    - [16, 32]    # Pass 3: 16 px tall × 32 px wide
  overlap:
    - 50
    - 50
    - 50
```

**Key Considerations:**

| Aspect | Details |
|--------|---------|
| **Dimension Order** | Always `[height, width]` — height is vertical (Y), width is horizontal (X) |
| **Aspect Ratios** | Common ratios: 1:2 (e.g., `[32, 64]`), 1:4 (e.g., `[16, 64]`), 2:1 (e.g., `[64, 32]`) |
| **Grid Density** | More windows in the direction with smaller dimension |
| **Displacement Limits** | First pass allows displacements up to `window_size/2` in each direction |
| **Overlap** | Applied as percentage of each dimension independently |

**Example: Horizontal Channel Flow**

For flow predominantly in the X (horizontal) direction, use wide windows:

```yaml
instantaneous_piv:
  window_size:
    - [32, 128]   # Wide: captures large X displacement, high Y resolution
    - [16, 64]    # Narrower for refinement
    - [8, 32]     # Final pass
  overlap:
    - 50
    - 50
    - 75          # Higher overlap on final pass for dense vectors
```

**Example: Vertical Jet Flow**

For flow predominantly in the Y (vertical) direction, use tall windows:

```yaml
instantaneous_piv:
  window_size:
    - [128, 32]   # Tall: captures large Y displacement, high X resolution
    - [64, 16]
    - [32, 8]
```

**Grid Calculation:**

For a 2048×2048 image with window `[64, 128]` and 50% overlap:

- Y direction: spacing = 32 px → ~63 windows
- X direction: spacing = 64 px → ~31 windows
- Total grid: 63 × 31 = 1,953 vectors

**Notes:**
- Rectangular windows work with both instantaneous and ensemble modes
- Masking correctly adapts to rectangular window grids
- The predictor-corrector handles rectangular windows for multi-pass refinement
- Very extreme aspect ratios (e.g., 1:8) may reduce correlation quality

---

### ensemble

Run ensemble PIV processing for time-averaged correlation analysis.

**Usage:**
```bash
pivtools-cli ensemble [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Process all active paths from config.yaml
pivtools-cli ensemble

# Process specific paths
pivtools-cli ensemble -p 0,1
```

**Configuration (config.yaml):**
```yaml
ensemble_piv:
  window_size:
    - [128, 128]
    - [64, 64]
    - [32, 32]
  overlap:
    - 50
    - 50
    - 50
  runs: [3]
  type:
    - std   # Standard ensemble
    - std
    - std
```

**Note:** Rectangular windows are fully supported for ensemble PIV. See [Rectangular Windows](#rectangular-windows) under instantaneous PIV for configuration details. Example:

```yaml
ensemble_piv:
  window_size:
    - [64, 128]   # Wide rectangle for horizontal flow
    - [32, 64]
  sum_window: [64, 128]  # Sum window must also be rectangular
```

**Output:**
- Ensemble result saved to `{base_path}/piv/ensemble/cam{N}/uncalibrated/run{R}/`
- Contains: mean velocities, Reynolds stresses (UU, VV, UV), correlation statistics

---

### detect-planar

Detect dot/circle grid calibration target and generate single-camera model.

**Usage:**
```bash
pivtools-cli detect-planar [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: all from config) |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Generate calibration for all cameras
pivtools-cli detect-planar

# Generate calibration for specific camera
pivtools-cli detect-planar -c 1

# Process only specific paths
pivtools-cli detect-planar -p 0,1
```

**Configuration (config.yaml):**
```yaml
calibration:
  image_format: calib%05d.tif
  subfolder: ""  # subfolder within source_path for calibration images

  pinhole:
    pattern_cols: 10        # Number of dots horizontally
    pattern_rows: 10        # Number of dots vertically
    dot_spacing_mm: 28.89   # Physical spacing between dots in mm
    asymmetric: false       # Use asymmetric circle grid
    enhance_dots: true      # Apply image enhancement for better detection
```

**Output:**
- Camera model saved to `{base_path}/calibration/Cam{N}/pinhole_planar/model/pinhole_model.mat`
- Detection visualizations saved to `{base_path}/calibration/Cam{N}/pinhole_planar/detections/`
- Dot center indices saved to `{base_path}/calibration/Cam{N}/pinhole_planar/indices/`

**Notes:**
- Requires calibration images with visible dot/circle grid pattern
- At least 3 valid images with detected grids are required
- RMS reprojection error is reported upon completion

---

### detect-charuco

Detect ChArUco board calibration target and generate single-camera model.

**Usage:**
```bash
pivtools-cli detect-charuco [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: all from config) |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Generate calibration for all cameras
pivtools-cli detect-charuco

# Generate calibration for specific camera
pivtools-cli detect-charuco -c 1

# Process only specific paths
pivtools-cli detect-charuco -p 0,1
```

**Configuration (config.yaml):**
```yaml
calibration:
  image_format: calib%05d.tif
  subfolder: ""

  charuco:
    squares_h: 10           # Number of squares horizontally
    squares_v: 9            # Number of squares vertically
    square_size: 0.03       # Physical square size in METERS
    marker_ratio: 0.5       # Ratio of marker size to square size
    aruco_dict: DICT_4X4_1000  # ArUco dictionary
    min_corners: 6          # Minimum corners required per image
    dt: 1.0                 # Time between frames
```

**ArUco Dictionary Options:**
- `DICT_4X4_50`, `DICT_4X4_100`, `DICT_4X4_250`, `DICT_4X4_1000`
- `DICT_5X5_50`, `DICT_5X5_100`, `DICT_5X5_250`, `DICT_5X5_1000`
- `DICT_6X6_50`, `DICT_6X6_100`, `DICT_6X6_250`, `DICT_6X6_1000`
- `DICT_7X7_50`, `DICT_7X7_100`, `DICT_7X7_250`, `DICT_7X7_1000`

**Output:**
- Camera model saved to `{base_path}/calibration/Cam{N}/charuco_planar/model/charuco_model.mat`
- Detection visualizations saved to `{base_path}/calibration/Cam{N}/charuco_planar/detections/`

**Notes:**
- ChArUco boards combine checkerboard and ArUco markers for robust detection
- Partial board visibility is supported (requires at least `min_corners` detected)
- RMS reprojection error is reported upon completion

---

### detect-stereo-planar

Detect dot/circle grid calibration target and generate stereo camera model.

**Usage:**
```bash
pivtools-cli detect-stereo-planar [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Generate stereo calibration
pivtools-cli detect-stereo-planar

# Process only specific paths
pivtools-cli detect-stereo-planar -p 0
```

**Configuration (config.yaml):**
```yaml
calibration:
  image_format: calib%05d.tif
  subfolder: ""

  stereo:
    camera_pair: [1, 2]     # Camera numbers for stereo pair
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    asymmetric: false
    enhance_dots: true
```

**Output:**
- Stereo model saved to `{base_path}/calibration/stereo_cam{N}_cam{M}/`
- Includes: intrinsic matrices, extrinsic parameters, fundamental/essential matrices

---

### detect-stereo-charuco

Detect ChArUco board calibration target and generate stereo camera model.

**Usage:**
```bash
pivtools-cli detect-stereo-charuco [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Generate stereo calibration
pivtools-cli detect-stereo-charuco

# Process only specific paths
pivtools-cli detect-stereo-charuco -p 0
```

**Configuration (config.yaml):**
```yaml
calibration:
  image_format: calib%05d.tif
  subfolder: ""

  stereo:
    camera_pair: [1, 2]

  charuco:
    squares_h: 10
    squares_v: 9
    square_size: 0.03
    marker_ratio: 0.5
    aruco_dict: DICT_4X4_1000
    min_corners: 6
```

**Output:**
- Stereo model saved to `{base_path}/calibration/stereo_cam{N}_cam{M}/`
- Includes: intrinsic matrices, extrinsic parameters, fundamental/essential matrices

---

### apply-calibration

Apply calibration to PIV vectors, converting from pixels to physical units (m/s).

**Usage:**
```bash
pivtools-cli apply-calibration [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: all from config) |
| `--type-name` | `-t` | Data type: `instantaneous` or `ensemble` |
| `--runs` | `-r` | Comma-separated run numbers (default: all) |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Apply calibration to all cameras, all runs
pivtools-cli apply-calibration

# Apply to specific camera
pivtools-cli apply-calibration -c 1

# Apply to ensemble data
pivtools-cli apply-calibration -t ensemble

# Apply to specific runs
pivtools-cli apply-calibration -r 1,2,3

# Apply to specific paths
pivtools-cli apply-calibration -p 0,1
```

**Supported Calibration Methods:**
- `scale_factor` - Simple px/mm scaling with dt
- `pinhole` - Pinhole camera model (planar/circle grid)
- `charuco` - ChArUco board detection
- `polynomial` - DaVis XML polynomial calibration
- `stereo` - Stereo camera pair

**Configuration (config.yaml):**
```yaml
calibration:
  active: pinhole  # or scale_factor, charuco, polynomial, stereo

  scale_factor:
    dt: 0.56      # Time between frames (s or ms)
    px_per_mm: 3.41

  pinhole:
    pattern_cols: 10
    pattern_rows: 10
    dot_spacing_mm: 28.89
    dt: 0.0275
```

**Output:**
- Calibrated vectors saved to `{base_path}/piv/{type}/cam{N}/calibrated/run{R}/`

---

### transform

Apply geometric transforms to PIV vector fields.

**Usage:**
```bash
pivtools-cli transform [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: all from config) |
| `--type-name` | `-t` | Data type: `instantaneous` or `ensemble` |
| `--operations` | `-o` | Comma-separated transforms |
| `--merged` | `-m` | Transform merged data instead of per-camera |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Available Transforms:**
- `flip_ud` - Flip vertically
- `flip_lr` - Flip horizontally
- `rotate_90_cw` - Rotate 90 degrees clockwise
- `rotate_90_ccw` - Rotate 90 degrees counter-clockwise
- `rotate_180` - Rotate 180 degrees
- `swap_ux_uy` - Swap velocity components
- `invert_ux_uy` - Negate ux and uy
- `scale_velocity:N` - Scale velocities by factor N
- `scale_coords:N` - Scale coordinates by factor N

**Examples:**
```bash
# Apply transforms from config.yaml
pivtools-cli transform

# Apply specific transforms
pivtools-cli transform -o flip_ud,rotate_90_cw

# Transform merged data
pivtools-cli transform --merged -o flip_lr

# Transform specific camera
pivtools-cli transform -c 1 -o rotate_180
```

**Configuration (config.yaml):**
```yaml
transforms:
  type_name: instantaneous
  cameras:
    1: [flip_ud, rotate_90_cw]
    2: [flip_lr]
```

**Notes:**
- Transforms are applied in-place (original files are modified)
- Statistics files are NOT transformed - recalculate if needed

---

### merge

Merge multi-camera vector fields using Hanning window blending.

**Usage:**
```bash
pivtools-cli merge [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--cameras` | `-c` | Comma-separated camera numbers (default: from config) |
| `--type-name` | `-t` | Data type: `instantaneous` or `ensemble` |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Examples:**
```bash
# Merge cameras from config.yaml
pivtools-cli merge

# Merge specific cameras
pivtools-cli merge -c 1,2,3

# Merge ensemble data
pivtools-cli merge -t ensemble

# Merge specific paths
pivtools-cli merge -p 0,1
```

**Configuration (config.yaml):**
```yaml
merging:
  type_name: instantaneous
  cameras: [1, 2]
```

**Requirements:**
- At least 2 cameras required
- Cameras must have overlapping regions
- Calibrated data recommended (but not required)

**Output:**
- Merged vectors saved to `{base_path}/piv/{type}/merged/run{R}/`

---

### statistics

Compute PIV statistics (mean, Reynolds stresses, TKE, vorticity, etc.).

**Usage:**
```bash
pivtools-cli statistics [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: all from config) |
| `--type-name` | `-t` | Data type: `instantaneous` or `ensemble` |
| `--merged` | `-m` | Process merged data instead of per-camera |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Computed Statistics:**
- Mean velocity (U, V, W if stereo)
- Velocity fluctuations (u', v', w')
- Reynolds stresses (u'u', v'v', u'v')
- Turbulent kinetic energy (TKE)
- Mean vorticity
- Mean divergence
- Gamma vortex criterion

**Examples:**
```bash
# Compute statistics for all cameras
pivtools-cli statistics

# Compute for merged data
pivtools-cli statistics --merged

# Compute for specific camera
pivtools-cli statistics -c 1

# Compute for ensemble data
pivtools-cli statistics -t ensemble

# Compute for specific paths
pivtools-cli statistics -p 0,1
```

**Configuration (config.yaml):**
```yaml
statistics:
  enabled_methods:
    mean_velocity: true
    reynolds_stress: true
    normal_stress: true
    mean_tke: true
    mean_vorticity: true
    mean_divergence: true
    inst_velocity: true
    inst_fluctuations: true
    inst_vorticity: true
    inst_divergence: true
    inst_gamma: true
  gamma_radius: 5
  save_figures: false
```

**Output:**
- Statistics saved to `{base_path}/piv/{type}/cam{N}/statistics/`
- Includes mean fields and per-frame instantaneous fields

---

### video

Create visualization videos from PIV data.

**Usage:**
```bash
pivtools-cli video [options]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: from config) |
| `--variable` | `-v` | Variable to visualize |
| `--run` | `-r` | Run number (default: 1) |
| `--data-source` | `-d` | Data source |
| `--fps` | | Frame rate (default: 30) |
| `--crf` | | Video quality 0-51 (default: 15, lower=better) |
| `--resolution` | | Output resolution (e.g., '1920x1080' or '4k') |
| `--cmap` | | Colormap name |
| `--lower` | | Lower color limit |
| `--upper` | | Upper color limit |
| `--test` | | Test mode: only process 50 frames |
| `--active-paths` | `-p` | Comma-separated path indices to process (e.g., '0,1,2') |

**Variables:**
- `ux`, `uy`, `uz` - Velocity components
- `mag` - Velocity magnitude
- `vorticity` - Vorticity
- `divergence` - Divergence
- `u_prime`, `v_prime` - Velocity fluctuations

**Data Sources:**
- `calibrated` - Calibrated vectors
- `uncalibrated` - Raw vectors
- `merged` - Merged multi-camera data
- `inst_stats` - Instantaneous statistics

**Examples:**
```bash
# Create video with defaults from config
pivtools-cli video

# Create velocity magnitude video
pivtools-cli video -v mag

# Create high-quality 4K video
pivtools-cli video --resolution 4k --crf 10

# Create test video (50 frames only)
pivtools-cli video --test

# Custom color limits
pivtools-cli video -v vorticity --lower -100 --upper 100

# Use specific colormap
pivtools-cli video --cmap viridis

# Process specific paths
pivtools-cli video -p 0,1
```

**Configuration (config.yaml):**
```yaml
video:
  camera: 1
  variable: ux
  run: 1
  data_source: calibrated
  cmap: default
  lower: ''
  upper: ''
  fps: 30
  crf: 15
  resolution: 1080p
```

**Output:**
- Video saved to `{base_path}/videos/{variable}_cam{N}_run{R}.mp4`

---

## Workflows

### Complete PIV Analysis Workflow

```bash
# 1. Initialize workspace
cd /path/to/experiment
pivtools-cli init

# 2. Edit config.yaml with your settings
# (paths, cameras, window sizes, calibration, etc.)

# 3. Generate camera calibration model
pivtools-cli detect-planar  # or detect-charuco

# 4. Run instantaneous PIV
pivtools-cli instantaneous

# 5. Apply calibration to vectors
pivtools-cli apply-calibration

# 6. Apply transforms if needed (flip, rotate)
pivtools-cli transform -o flip_ud

# 7. Merge multi-camera data (if applicable)
pivtools-cli merge

# 8. Compute statistics
pivtools-cli statistics

# 9. Create visualization video
pivtools-cli video -v mag
```

### Stereo PIV Workflow

```bash
# 1. Generate stereo calibration
pivtools-cli detect-stereo-planar  # or detect-stereo-charuco

# 2. Run PIV processing
pivtools-cli instantaneous

# 3. Apply stereo calibration (reconstructs 3D vectors)
pivtools-cli apply-calibration

# 4. Compute statistics
pivtools-cli statistics
```

### Ensemble PIV Workflow

```bash
# Run ensemble averaging
pivtools-cli ensemble

# Apply calibration to ensemble results
pivtools-cli apply-calibration -t ensemble

# Compute statistics on ensemble
pivtools-cli statistics -t ensemble
```

### Batch Processing Multiple Experiments

```bash
# Process multiple experiments sequentially
for exp in exp1 exp2 exp3; do
    echo "Processing $exp..."
    (cd /data/$exp && pivtools-cli instantaneous)
done

# Or process in parallel (background jobs)
for exp in exp1 exp2 exp3; do
    (cd /data/$exp && pivtools-cli instantaneous) &
done
wait
echo "All experiments complete"
```

### Processing Specific Path Sets

```bash
# Process only first two path sets
pivtools-cli instantaneous -p 0,1

# Process paths separately
pivtools-cli instantaneous -p 0
pivtools-cli instantaneous -p 1
pivtools-cli instantaneous -p 2
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PIV_ACTIVE_PATHS` | Override active paths (comma-separated indices) |
| `MALLOC_TRIM_THRESHOLD_` | Set to "0" for better memory management |
| `OMP_NUM_THREADS` | Control OpenMP thread count |

**Example:**
```bash
# Override active paths via environment
PIV_ACTIVE_PATHS=0,1 pivtools-cli instantaneous

# Control threading
OMP_NUM_THREADS=4 pivtools-cli instantaneous
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error or partial failure |

All commands exit with code 0 on success, 1 on any error.

---

## See Also

- [Transforms and Merging](transforms_and_merging.md)
- [Dask Memory Architecture](DASK_MEMORY_ARCHITECTURE.md)
- [Ensemble PIV Dataflow](ENSEMBLE_PIV_DATAFLOW.md)
