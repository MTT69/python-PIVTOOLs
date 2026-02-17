# CLAUDE.md - PIVTOOLs Reference

> Comprehensive reference for PIVTOOLs: a fullstack application for Particle Image Velocimetry analysis.
> Covers the GUI (Flask + React), processing pipeline (Dask-distributed), C extensions, build system, and data formats.
> Use this document to understand file locations, module responsibilities, and inter-component dependencies when making any changes.

---

## Rules

- **Always update docs after changes:** After completing any feature/refactor, update this `CLAUDE.md` and `MEMORY.md` before finishing. Don't wait to be asked.
  - **CLAUDE.md** (this file) is public, checked into git. Put project knowledge here: architecture, conventions, patterns, gotchas, key file locations, build commands, code conventions, collection schemas. Any contributor using Claude Code gets this context automatically.
  - **MEMORY.md** is private, lives on the local machine (`~/.claude/projects/.../memory/MEMORY.md`), never committed. Put personal workflow preferences, local env quirks, session-to-session learnings specific to how the user works, and things still being validated.

---

## Architecture Overview

```
PIVTOOLs is a Flask (Python) + Next.js (React/TypeScript) fullstack app.
The Python backend serves REST endpoints; the React frontend consumes them.
Config is stored as config.yaml (single source of truth), loaded via Config class.

                    ┌──────────────────────┐
                    │   config.yaml        │  <-- single source of truth
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌────────────┐   ┌──────────────┐   ┌──────────────┐
     │ pivtools_  │   │ pivtools_gui │   │ PIVTOOLs-GUI │
     │ core       │   │ (Flask BE)   │   │ (React FE)   │
     │            │   │              │   │              │
     │ Config     │◄──│ app.py       │◄──│ hooks/*.ts   │
     │ VectorLoad │   │ blueprints   │   │ components   │
     │ ImageLoad  │   │ /backend/*   │   │ fetch()      │
     └────────────┘   └──────────────┘   └──────────────┘
```

**Communication pattern:** Frontend hooks call `fetch('/backend/...')` → Flask routes → pivtools_core utilities → return JSON.

**Config flow:** Frontend `useConfigUpdate` → `POST /backend/update_config` → `recursive_update(cfg.data, payload)` → saves to `config.yaml` → `reload_config()`.

---

## Quick Reference: Feature → File Map

| Feature | Backend Module | Frontend Hook | Frontend Component |
|---------|---------------|---------------|-------------------|
| Path/image setup | `app.py` routes | `useConfigUpdate` | `PathsConfig`, `ImageConfig` |
| Image viewing | `app.py` `/get_frame_pair` | `useImagePair` | `ImagePairViewer` |
| Image filters | `app.py` `/filter` | `useImageFilters` | `FilterComponents` |
| Masking | `masking/app/views.py` | `useMaskingConfig` | `Masking`, `PolygonMaskEditor` |
| PIV processing | `piv_runner.py` | `usePivRunner` | `RunPIV` |
| Calibration (scale) | `calibration/app/scale_factor_views.py` | `useScaleFactorCalibration` | `ScaleFactorCalibration` |
| Calibration (dotboard) | `calibration/app/dotboard_views.py` | `useDotboardCalibration` | `DotboardCalibration` |
| Calibration (ChArUco) | `calibration/app/charuco_views.py` | `useChArUcoCalibration` | `ChArUcoCalibration` |
| Calibration (polynomial) | `calibration/app/polynomial_views.py` | `useCalibration` | `PolynomialCalibration` |
| Calibration (stereo) | `calibration/app/stereo_*_views.py` | `useStereoCalibration` | `StereoCalibration` |
| Self-calibration | `stereo_reconstruction/self_calibration.py` | N/A (script) | N/A |
| Calibration images | `calibration/app/shared_views.py` | `useCalibrationImageViewer` | `CalibrationImageViewer` |
| Global coordinates | `calibration/global_coordinate_alignment.py` | `useGlobalCoordinates` | `GCInlineControls` (inline in CalibrationImageViewer settings bar) |
| Vector viewing | `plotting/app/plotting_views.py` | `useVectorViewer` | `VectorViewer` |
| Transforms (GUI) | `plotting/app/transform_views.py` | (in `useVectorViewer`) | (in `VectorViewer`) |
| Transforms (CLI) | `transforms/transform_production.py` | N/A | N/A |
| Vector merging | `vector_merging/app/views.py` | (inline in `VectorViewer.tsx`) | (in `VectorViewer`) |
| Statistics | `vector_statistics/app/views.py` | `useStatisticsCalculation` | (in `VectorViewer`) |
| Video maker | `video_maker/app/views.py` | `useVideoMaker` | `VideoMaker` |
| Instantaneous PIV config | - | `useInstantaneousPivConfig` | `InstantaneousPIV` |
| Ensemble PIV config | - | `useEnsemblePivConfig` | `EnsemblePIV` |
| Job tracking | `calibration/services/job_manager.py` | (in various hooks) | - |

---

## Backend: `pivtools_core` (GUI-Relevant Only)

> These modules provide the data layer. The GUI imports but does NOT own them.

### `config.py` - Central Configuration

**Singleton pattern:** `get_config() -> Config` (cached), `reload_config()` resets cache.

```python
class Config:
    __init__(path: Optional[str] = None)  # Loads config.yaml
    save()                                 # Write to YAML
    save_timestamped_copy(destination_dir: Path, timestamp: str = None) -> Path
    save_calibration_snapshot(base_path: Path) -> Path  # Saves calibration block to base_path/calibration/calibration.yaml
    load_calibration_snapshot(base_path: Path) -> dict   # Static method, loads snapshot. Raises FileNotFoundError if missing

    # --- Path properties ---
    config_path: Path
    base_paths: List[Path]           # Output directories
    source_paths: List[Path]         # Image source directories
    camera_count: int
    camera_numbers: List[int]
    camera_folders: List[str]
    get_camera_folder(camera_num: int) -> str  # Respects camera_subfolders config
    stereo_pairs: List[Tuple[int,int]]
    is_stereo_setup: bool

    # --- Image properties ---
    image_format: Tuple[str, ...]    # Always tuple: ("fmt",) or ("fmtA", "fmtB")
    image_type: str                  # "standard" | "cine" | "lavision_set" | "lavision_im7"
    image_shape: Tuple[int, int]     # (H, W) - auto-detected and cached
    num_images: int                  # Total image FILES
    start_index: int                 # First frame number (0 or 1)
    frame_stride: int                # Gap between A and B within a pair (0=pre-paired)
    pair_stride: int                 # Gap between starts of consecutive pairs
    pairing_preset: str              # "ab_format" | "skip_frames" | "time_resolved" | "pre_paired" | "custom"
    num_frame_pairs: int             # Computed: fs==0 → num_images, else (num_images-1-fs)//ps+1
    time_resolved: bool              # Backward-compat: True when frame_stride > 0
    pairing_mode: str                # Backward-compat: derived from preset
    zero_based_indexing: bool        # Backward-compat: True when start_index == 0
    is_container_format: bool
    is_multi_camera_container: bool
    get_frame_pair_indices(pair_number: int) -> Tuple[int, int]  # Stride formula

    # --- Processing properties ---
    batch_size: int                  # Capped at num_frame_pairs
    piv_chunk_size: int
    vector_format: str               # e.g. "B%05d.mat"
    filters: List[dict]

    # --- Calibration properties ---
    active_calibration_method: str   # "scale_factor" | "dotboard" | "charuco" | "polynomial" | "stereo_*"
    calibration_piv_type: str
    global_coordinates_config: dict          # calibration.global_coordinates section
    global_coordinates_enabled: bool
    global_coordinates_datum_pixel: Optional[List[float]]
    global_coordinates_datum_physical: List[float]   # [x_mm, y_mm], default [0,0]
    global_coordinates_datum_frame: int
    global_coordinates_overlap_points: list  # [{target_camera, pixel_on_datum_cam, pixel_on_target, target_frame}] (legacy)
    global_coordinates_overlap_pairs: list   # [{camera_a, camera_b, pixel_on_a, pixel_on_b, frame_a, frame_b}] (chain topology, with backward compat from overlap_points)
    global_coordinates_invert_ux: bool       # True to negate ux + UV_stress + reflect x-coords during alignment

    # --- Statistics properties ---
    statistics: dict
    statistics_enabled_methods: dict  # {method_name: bool}
    statistics_enabled_list: list
    statistics_gamma_radius: int
    statistics_type_name: str
    statistics_source_endpoint: str

    # --- Transform properties ---
    transforms_cameras: dict         # {cam_num: [operation_list]}
    get_camera_transforms(camera: int) -> list
    set_camera_transforms(camera: int, operations: list)
    clear_camera_transforms(camera: int)
    transforms_type_name: str
    transforms_base_path_idx: int

    # --- Video properties ---
    video: dict

    # --- Masking ---
    masking_enabled: bool
    get_mask_path(camera: int, source_path_idx: int) -> Path

    # --- PIV parameters ---
    instantaneous_runs: list
    instantaneous_window_sizes: list
    instantaneous_overlaps: list

    # --- Tool constraints ---
    get_allowed_endpoints(tool_name: str) -> list
```

**Config YAML top-level keys:**
`paths`, `images`, `batches`, `logging`, `processing`, `outlier_detection`, `infilling`, `ensemble_outlier_detection`, `ensemble_infilling`, `plots`, `videos`, `statistics_extraction`, `instantaneous_piv`, `ensemble_piv`, `calibration`, `filters`, `masking`, `merging`, `statistics`, `transforms`, `video`, `post_processing`

**Tool endpoint constraints** (defined at top of config.py):
```python
TOOL_ALLOWED_SOURCE_ENDPOINTS = {
    "video": ["regular", "merged", "stereo"],
    "merging": ["regular"],                    # Only per-camera data
    "statistics": ["regular", "merged", "stereo"],
    "transforms": ["regular", "merged", "stereo"],
}
TOOL_ALLOWED_TYPE_NAMES = {
    "video": ["instantaneous"],                # No ensemble (no temporal sequence)
    "merging": ["instantaneous", "ensemble"],
    "statistics": ["instantaneous", "ensemble"],
    "transforms": ["instantaneous", "ensemble"],
}
```

### `vector_loading.py` - Vector Data I/O

Loads `.mat` files containing PIV results (MATLAB struct format).

```python
@dataclass
class RunValidationResult:
    valid_runs: List[int]
    total_runs: int
    single_run: bool

# --- Validation ---
is_run_valid(struct, fields=("ux","uy"), require_2d=True, reject_all_nan=True) -> bool
find_valid_runs(file_path, var_name="piv_result", fields=("ux","uy"), ...) -> RunValidationResult
find_valid_piv_runs(file_path, one_based=False, result_key="piv_result") -> RunValidationResult
find_valid_coord_runs(file_path, one_based=False) -> RunValidationResult
find_valid_ensemble_runs(file_path, one_based=False) -> RunValidationResult
get_first_valid_run(...) -> Optional[int]
get_highest_valid_run(...) -> Optional[int]
find_non_empty_run(piv_result, var: str, run=1, ...) -> Tuple[Optional[Any], int]

# --- Loading ---
read_mat_contents(file_path, run_index=None, return_all_runs=False, var_name="piv_result") -> np.ndarray
    # Returns shape (1, C, H, W) where C=3 (ux,uy,b_mask) or C=4 (ux,uy,uz,b_mask for stereo)
load_vectors_from_directory(data_dir: Path, config: Config, runs=None) -> da.Array
    # Returns Dask array (N_files, R, C, H, W)
load_coords_from_directory(data_dir: Path, runs=None) -> Tuple[List[ndarray], List[ndarray]]

# --- Variable inspection ---
get_plottable_vars(file_path, var_name="piv_result", ...) -> List[str]
get_plottable_vars_from_struct(struct) -> List[str]
EXCLUDED_VARS = {"win_ctrs_x", "win_ctrs_y", "window_size", "n_windows", "predictor_field"}

# --- Mask I/O ---
save_mask_to_mat(file_path: str, mask: np.ndarray, polygons)
read_mask_from_mat(file_path: str) -> Tuple[np.ndarray, list]
    # polygons: list of {"index": int, "name": str, "points": list}
```

**Vector .mat file format:**
```
piv_result (or ensemble_result):
  .ux    -> (H, W) float64   # x-velocity
  .uy    -> (H, W) float64   # y-velocity
  .uz    -> (H, W) float64   # z-velocity (stereo only)
  .b_mask -> (H, W) float64  # binary mask (1=valid, 0=invalid)
  Optional: .UU_stress, .VV_stress, .UV_stress (ensemble)

coordinates:
  .x -> (H, W) or (N,) float64  # x coordinates
  .y -> (H, W) or (N,) float64  # y coordinates
```

### `paths.py` - Output Directory Structure

```python
get_data_paths(base_path, num_frame_pairs, camera_num, type_name, use_uncalibrated=False) -> dict
    # Returns: {"data_dir": Path, "stats_dir": Path, ...}
```

**Output directory convention:**
```
base_path/
  {type_name}/                  # "instantaneous" or "ensemble"
    {num_frame_pairs}/
      Cam{N}/                   # Per-camera vectors
        B00001.mat ... B0NNNN.mat
        coordinates.mat
      Merged/                   # After merging
      Stereo Cam1_Cam2/         # After stereo reconstruction
```

### `image_handling/` - Image Reading

```
load_images.py:     read_pair(idx, source_path, camera, cfg) -> np.ndarray
                    read_image(path, **kwargs) -> np.ndarray
                    load_mask_for_camera(camera, cfg, source_path_idx) -> Optional[np.ndarray]

path_utils.py:      build_piv_camera_path(cfg, source_path_idx, camera_num) -> Path
                    validate_images_generic(...) -> dict
                    validate_single_pattern(...) -> dict

readers/__init__.py: get_reader(extension) -> callable  # Reader registry
readers/lavision_reader.py: read_lavision_pair(...), read_lavision_set_pair(...)
readers/cine_reader.py:     read_cine_pair(...)
readers/generic_readers.py: read_tiff(...), read_png_jpeg(...)
```

---

## Backend: `pivtools_gui` Modules

### `app.py` - Main Flask Application

**Blueprint registration:**
| Blueprint | Prefix | Source |
|-----------|--------|--------|
| `api_bp` | `/backend` | app.py (main routes) |
| `vector_plot_bp` | `/backend/plot` | plotting/app/plotting_views.py |
| `transform_bp` | `/backend/plot` | plotting/app/transform_views.py (active, used by frontend) |
| `masking_bp` | `/backend` | masking/app/views.py |
| `calibration_bp` | `/backend` | calibration/app/views.py (aggregator) |
| `video_maker_bp` | `/backend/video` | video_maker/app/views.py |
| `statistics_bp` | `/backend` | vector_statistics/app/views.py |
| `merging_bp` | `/backend` | vector_merging/app/views.py |

**Main app.py routes (under `/backend/`):**

```
GET  /get_frame_pair       ?camera=&idx=&source_path_idx=&format=&auto_limits=
POST /preload_images        {camera, start_idx, count, source_path_idx, format}
POST /filter                {camera, start_idx, filters, masking, source_path_idx}
GET  /processing_status
POST /processing_status     {cancel: true}
GET  /get_processed_pair    ?frame=&type=&camera=&source_path_idx=&auto_limits=
POST /filter_single_frame   {camera, frame_idx, filters, source_path_idx, auto_limits}
POST /download_image        {type, frame, data, frame_idx, camera}
GET  /config                -> full config JSON (includes computed pairing properties)
GET  /preview_frame_pairs   ?count=5 -> first N pairs with resolved filenames
POST /update_config         {any config subset} -> recursive merge + save
POST /validate_files        {source_path_idx} -> per-camera validation
POST /run_piv               {cameras, source_path_idx, base_path_idx, active_paths, mode}
GET  /piv_status            ?job_id=
POST /cancel_run            {job_id}
GET  /piv_logs              ?job_id=&lines=&offset=
GET  /get_uncalibrated_count ?basepath_idx=&camera=&type=&job_id=
GET  /check_output_exists   ?active_paths=
POST /clear_output          {active_paths, camera_numbers}
GET  /system_info            -> {version, python_version, platform, dask, c_libraries, fftw_wisdom}
```

**Key internal functions:**
```python
manage_cache_size()           # LRU eviction with frame-1 pinning
make_raw_cache_key(source_path_idx, camera, idx, img_format, cfg) -> tuple
get_percentile_stats(img_array) -> {"vmin_pct": float, "vmax_pct": float}
get_cached_pair(frame, typ, camera, source_path_idx, cfg, auto_limits) -> (b64_a, b64_b, stats)
compute_batch_window(target_idx, batch_size, total) -> (start, end)
recursive_update(d, u)        # Deep-merge dict u into d
get_active_calibration_params(cfg) -> (method_name, params_dict)
_preload_surrounding_frames(source_path_idx, camera, current_idx, cfg, window, img_format, auto_limits)
```

### `utils.py` - Shared Utilities

```python
camera_number(camera: Union[str, int]) -> int     # "Cam1" | "1" | 1 → 1
camera_folder(camera: Union[str, int]) -> str      # DEPRECATED, use config.get_camera_folder()
numpy_to_png_base64(arr: np.ndarray, compress_level=1) -> str
numpy_to_base64(arr: np.ndarray, format="png", compress_level=1, jpeg_quality=85) -> str
    # Auto-normalizes non-uint8 to 0-255 range
```

### `piv_runner.py` - PIV Job Subprocess Manager

Spawns PIV computation as separate subprocess (keeps Flask responsive).

```python
get_runner() -> PivRunner                          # Singleton
class PivRunner:
    start_piv_job(cameras, source_path_idx, base_path_idx, active_paths, mode) -> dict
    get_job_status(job_id: str) -> dict
    list_jobs() -> list
    cancel_job(job_id: str) -> bool
```

### `calibration/` - Calibration Module

**Router:** `calibration/app/views.py` aggregates sub-blueprints:

| Sub-blueprint | Routes prefix | Purpose |
|---------------|--------------|---------|
| `scale_factor_bp` | `/calibrate/scale_factor` | px/mm + dt calibration |
| `dotboard_bp` | `/calibrate/dotboard` | Grid detection calibration |
| `charuco_bp` | `/calibrate/charuco` | ChArUco board calibration |
| `polynomial_bp` | `/calibrate/polynomial` | DaVis XML polynomial calibration |
| `stereo_dotboard_bp` | `/calibrate/stereo_dotboard` | Stereo dotboard calibration |
| `stereo_charuco_bp` | `/calibrate/stereo_charuco` | Stereo ChArUco calibration |
| `calibration_shared_bp` | `/calibrate` | Shared: datum, status, image loading |

**Key calibration routes:**
```
POST /calibrate/scale_factor/run       {cameras, active_paths, dt, px_per_mm}
POST /calibrate/dotboard/run           {cameras, active_paths, ...params}
POST /calibrate/dotboard/detect_one    {camera, frame_idx} -> preview detection
POST /calibrate/charuco/run            {cameras, active_paths, ...params}
POST /calibrate/polynomial/run         {cameras, active_paths}
POST /calibrate/stereo_dotboard/run    {camera_pair, ...params}
POST /calibrate/stereo_charuco/run     {camera_pair, ...params}
GET  /calibrate/job/<job_id>           -> job status
GET  /calibrate/calibration_image      ?camera=&idx=&source_path_idx=
POST /calibrate/set_datum              {base_path_idx, type_name, camera, datum_x, datum_y}
POST /calibrate/global_coordinates/pixel_to_physical  {pixel_x, pixel_y, camera, source_path_idx, type_name}
GET  /calibrate/calibration/snapshot       ?base_path_idx= -> {exists, date?, calibration_method?}
POST /calibrate/calibration/snapshot/load  {base_path_idx} -> restores saved calibration config
```

**Production files** (do the actual calibration work):
- `scale_factor_calibration_production.py` - Simple px→mm scaling
- `global_coordinate_alignment.py` - Global coordinate alignment (shifts coordinates.mat after calibration). Uses chain topology (cam N↔cam N+1 pairs) instead of star topology. Supports `invert_ux` to negate ux + UV_stress and reflect x-coordinates around datum_physical_x when x-direction is reversed. Auto-applied when `global_coordinates.enabled` in vectors_calibrate route. **Idempotency guard:** writes `alignment_applied.json` sidecar marker after alignment; blocks re-application unless `force=True`. Fresh calibration (vector or scale_factor) clears the marker automatically. Pre-flight validates: datum_pixel set, method not polynomial, overlap_pairs for multi-camera.
- `calibration_planar/planar_calibration_production.py` - Dotboard detection + model. Optimized: histogram-based single blob detection, cKDTree neighbor finding, reduced RANSAC iterations, vectorized object points, no double image reads for containers
- `calibration_charuco/charuco_calibration_production.py` - ChArUco detection
- `calibration_poly/polynomial_calibration_production.py` - DaVis XML polynomial
- `vector_calibration_production.py` - Applies calibration to vectors
- `stereo_reconstruction/stereo_dotboard_calibration_production.py` - Stereo dotboard. Same optimizations as planar: histogram blob detection, cKDTree, reduced RANSAC, vectorized object points
- `stereo_reconstruction/stereo_calibration_base.py` - Stereo base class. Parallel camera image reads via ThreadPoolExecutor
- `stereo_reconstruction/stereo_charuco_calibration_production.py` - Stereo ChArUco detection
- `stereo_reconstruction/self_calibration.py` - Stereo self-calibration (Wieneke 2005). Detects and corrects laser-sheet Z-offset and tilts via iterative disparity minimization. Key exports: `PinholeCamera`, `SelfCalibrationResult`, `run_self_calibration()`, `compute_dewarp_maps()`. Reuses `bulkxcorr2d_accumulate` C library for ensemble cross-camera correlation, `median_outlier_detection` + `infill_local_median` for disparity cleaning. Includes pure-Python FFT fallback when C library unavailable. Test script: `scripts/test_self_calibration.py` (synthetic data + 5 diagnostic figures).

**Dotboard calibration performance optimizations** (shared by planar + stereo):
- **Blob detection:** Histogram-based single pass (checks `mean_intensity > 127` to decide original vs inverted), with fallback if <9 keypoints found
- **Neighbor finding:** `scipy.spatial.cKDTree` replaces O(N^2) pairwise distance matrix — O(N log N) for spacing estimation and pair finding
- **RANSAC:** Reduced to `maxIters=1000, confidence=0.97` (was 2000/0.99). Threshold `0.15 * spacing_px` unchanged
- **Object points:** Vectorized with NumPy (no Python loop)
- **Container reads:** Eliminated duplicate image reads (was read-to-test then read-again)
- **Multi-camera (planar):** `dotboard_views.py` processes all cameras in parallel via `ThreadPoolExecutor`
- **Stereo reads:** Both camera images read in parallel via `ThreadPoolExecutor(max_workers=2)`
- **Preserved:** Reflection filtering (connected components), RANSAC outlier threshold, grid deduplication, subpixel refinement

**`services/job_manager.py`** - Shared across all long-running operations:
```python
job_manager = JobManager()  # Module-level singleton
class JobManager:
    create_job(job_type: str, **metadata) -> str    # Returns job_id
    update_job(job_id, **kwargs)
    complete_job(job_id, **result_data)
    fail_job(job_id, error: str)
    get_job(job_id) -> dict
    get_job_with_timing(job_id) -> dict              # Adds elapsed_seconds
    add_timing_info(job_data) -> dict
```

### `transforms/` - Geometric Transformations

**Two execution paths (shared core logic in `transform_operations.py`):**

| Path | Processor | Used by | Backup/Restore |
|------|-----------|---------|----------------|
| GUI (plotting) | `VectorTransformProcessor` | Frontend via `/backend/plot/transform_*` | Yes (per-frame `_original` backup in .mat) |
| CLI | `TransformProcessor` | `pivtools-cli transform` command | No (direct batch apply) |

**GUI routes (under `/backend/plot/`, registered via `plotting/app/transform_views.py`):**
```
POST /plot/transform_frame              {base_path, camera, frame, transformation, merged}
POST /plot/clear_transform              {base_path, camera, frame, merged}
GET  /plot/check_transform_status       ?base_path=&camera=&frame=&merged=
POST /plot/transform_all_frames         {base_path, camera, transformations, merged, image_count}
GET  /plot/transform_all_frames/status/<job_id>
```

**Valid transformations:** `flip_ud`, `flip_lr`, `rotate_90_cw`, `rotate_90_ccw`, `rotate_180`, `swap_ux_uy`, `invert_ux_uy`, `invert_ux`, `invert_uy`, `scale_velocity:<factor>`, `scale_coords:<factor>`

**Ensemble-aware:** Transforms correctly handle Reynolds stresses (UU_stress, VV_stress, UV_stress) in calibrated ensemble data:
- Geometric transforms (flip/rotate): stress fields are spatially transformed alongside velocities
- `scale_velocity:k`: stresses scaled by k² (velocity² units)
- `swap_ux_uy`: UU_stress swapped with VV_stress (UV unchanged)
- `invert_ux_uy`: stresses unchanged (both negated: (-u')(-v') = u'v')
- `invert_ux`: UV_stress negated ((-u')v' = -(u'v')), UU/VV unchanged
- `invert_uy`: UV_stress negated (u'(-v') = -(u'v')), UU/VV unchanged

**Key files:**
- `transform_operations.py` - `apply_transformation_to_piv_result()`, `simplify_transformations()` (shared by both paths)
- `vector_transform_processor.py` - `VectorTransformProcessor` (GUI: per-frame with backup/restore)
- `transform_production.py` - `TransformProcessor` (CLI: direct batch apply)
- `plotting/app/transform_views.py` - Flask endpoints (registered at `/backend/plot`)

### `plotting/` - Vector Field Visualization

**Routes (under `/backend/plot/`):**
```
GET  /plot_vector          ?frame=&camera=&var=&run=&base_path_idx=&type_name=&use_uncalibrated=&var_source=&cmap=&lower=&upper=&source_endpoint=
GET  /plot_stats            (similar params, for statistics data)
GET  /plot_ensemble         (similar params, for ensemble data)
GET  /check_vars           ?camera=&base_path_idx=&type_name=&use_uncalibrated=&source_endpoint=
GET  /check_limits         ?file_path=&var=&run=
GET  /check_runs           ?camera=&base_path_idx=&type_name=
GET  /check_available_data ?base_path_idx=&type_name=&camera=
GET  /get_coordinate_at_point  ?camera=&base_path_idx=&type_name=&x=&y=&use_uncalibrated=
GET  /get_vector_at_position   ?camera=&frame=&run=&base_path_idx=&type_name=&i=&j=&source_endpoint=&var_source=
GET  /get_stats_value_at_position  (similar)
GET  /get_uncalibrated_image   ?camera=&frame=&base_path_idx=&type_name=
```

**Supporting files:**
- `shared_utils.py` - `parse_plot_params()`, `validate_and_get_paths()`, `load_piv_result()`, `create_and_return_plot()`, `VARIABLE_UNITS` (NOTE: plot_vector reads .mat from disk each time - no server-side caching of vector plots)
- `plot_maker.py` - `make_scalar_settings()`, matplotlib rendering

### `vector_merging/` - Multi-Camera Merging

**Routes (under `/backend/`):**
```
GET  /merge_vectors/constraints     -> allowed endpoints, stereo_blocked
POST /merge_vectors/merge_one       {frame_idx, cameras?, type_name?}
POST /merge_vectors/merge_all       {base_path_idx?, type_name?, cameras?}
POST /merge_vectors/validate        {cameras?, type_name?}
GET  /merge_vectors/status/<job_id>
```

All three POST routes accept `cameras` (list of ints) and `type_name` from request body, falling back to config. Cameras must be a continuous/adjacent range (e.g., [1,2,3] not [1,3]). Supports both instantaneous and ensemble type_name.

**`vector_merger.py`:**
```python
class VectorMerger:
    __init__(base_dir, cameras, type_name, config, max_workers=8)
    merge_single_frame(frame_idx, runs=None) -> dict
    merge_all_frames(progress_callback=None) -> dict
    # Uses Hanning window blending for overlapping regions
```

### `vector_statistics/` - Statistical Analysis

**Routes (under `/backend/`):**
```
GET  /statistics/constraints              -> workflow_options, allowed_source_endpoints
POST /statistics/compute                  {base_path_idx, process_merged, process_stereo, type_name?}
GET  /statistics/compute_status/<job_id>
```

**`instantaneous_statistics.py`:**
```python
class VectorStatisticsProcessor:
    __init__(data_dir, stats_dir, config, type_name="instantaneous")
    compute_all(progress_callback=None) -> dict
    # Computes: mean velocity, Reynolds stress, TKE, vorticity, divergence, gamma, mean_peak_height
    # Per-frame: inst_vorticity, inst_divergence, inst_fluctuations, inst_gamma
```

### `video_maker/` - Animation Generation

**Routes (under `/backend/video/`):**
```
GET  /list_videos               ?base_path_idx=&camera=
POST /create                    {variable, run, cmap, lower, upper, fps, crf, resolution, ...}
GET  /status/<job_id>
GET  /download/<job_id>
GET  /inspect                   ?base_path_idx=&camera=&type_name=&source_endpoint=
DELETE /delete/<video_name>     ?base_path_idx=&camera=
```

**`video_maker.py`:**
```python
class VideoMaker:
    __init__(data_dir, output_dir, config)
    create_video(variable, run, cmap, lower, upper, fps, crf, resolution, progress_callback) -> dict
find_all_valid_runs_from_file(file_path, var_name="piv_result") -> List[int]
```

### `masking/` - Image Masking

**Routes (under `/backend/`):**
```
GET  /masking/config             -> masking config section
POST /masking/config             {enabled, mode, rectangular, mask_file_pattern, ...}
POST /masking/save_polygon_mask  {camera, source_path_idx, polygons}
GET  /masking/load_polygon_mask  ?camera=&source_path_idx=
```

---

## Frontend: `PIVTOOLs-GUI/src/`

### Page Structure (`app/page.tsx`)

Main layout is a tabbed interface:
```
Tabs: Setup | Viewer
  Setup sub-tabs: Paths | Image | Calibration | Masking | Filters | PIV | Run
  Viewer: ImagePairViewer | VectorViewer | VideoMaker
```

Config flows top-down: `page.tsx` fetches config on mount, passes as props. Updates go through `useConfigUpdate` hook.

### Hooks (state management + backend communication)

#### `useConfigUpdate.ts`
```typescript
useConfigUpdate() -> { updateConfig(payload) -> Promise<{success, data?, error?}>, isUpdating, updateError }
useAutoValidation(config) -> ValidationState
    // Auto-validates when config changes, calls POST /backend/validate_files
```

#### `useImagePair.ts`
```typescript
useImagePair(config, camera, sourcePathIdx) -> {
    currentFrame, setCurrentFrame,
    imageA, imageB,           // base64 strings
    isLoading, error,
    totalFrames,
}
// Calls: GET /backend/get_frame_pair, POST /backend/preload_images
```

#### `useImageFilters.ts`
```typescript
useImageFilters(config) -> {
    filters, addFilter, removeFilter, updateFilter,
    applyFilters, isProcessing,
    processedA, processedB,
}
// Calls: POST /backend/filter, GET /backend/processing_status, GET /backend/get_processed_pair
```

#### `useInstantaneousPivConfig.ts` / `useEnsemblePivConfig.ts`
```typescript
// Manage PIV processing parameters (window sizes, overlaps, passes, etc.)
// Calls: POST /backend/update_config
```

#### `useMaskingConfig.ts`
```typescript
useMaskingConfig(config) -> {
    maskingConfig, updateMasking, saveMask, loadMask,
}
// Calls: POST /backend/masking/config, POST /backend/masking/save_polygon_mask
```

#### `usePivRunner.ts`
```typescript
usePivRunner() -> {
    startPiv, cancelPiv, jobStatus, isRunning, logs,
}
// Calls: POST /backend/run_piv, GET /backend/piv_status, POST /backend/cancel_run, GET /backend/piv_logs
```

#### `useVectorViewer.ts`
```typescript
// Most complex hook - manages all vector viewing state
useVectorViewer({backendUrl, config}) -> {
    frame, setFrame, totalFrames,
    plotImageUrl,              // base64 plot from backend
    variable, setVariable,
    run, setRun,
    camera, setCamera,
    dataSource, setDataSource,  // DataSourceType
    availableDataSources,       // AvailableDataSources
    groupedVariables,           // GroupedVariables
    hoverData,                  // HoverData (x, y, ux, uy at cursor)
    colorLimits, setColorLimits,
    cmap, setCmap,
    // ... transforms, merging, statistics controls
}
// Calls: GET /backend/plot/plot_vector, /check_vars, /check_available_data, /check_limits, /check_runs
//        GET /backend/plot/get_vector_at_position, /get_coordinate_at_point
```

**DataSourceType enum:**
`"calibrated_instantaneous"` | `"uncalibrated_instantaneous"` | `"calibrated_ensemble"` | `"uncalibrated_ensemble"` | `"merged_instantaneous"` | `"merged_ensemble"` | `"stereo_instantaneous"` | `"stereo_ensemble"` | `"statistics"` | `"merged_statistics"` | `"stereo_statistics"`

#### Vector Merging (inline in `VectorViewer.tsx`)
```typescript
// Inline hook: useVectorMerging(backendUrl, basePathIdx, cameraOptions, maxFrameCount, config, dataSource)
// Derives effectiveTypeName from dataSource (avoids async config race)
// Sends cameras + type_name in request body to backend
// Calls: POST /backend/merge_vectors/merge_one, /merge_all, GET /merge_vectors/status/*, /constraints
// Note: useVectorMerging.ts standalone file has been deleted (was dead code)
```

#### `useStatisticsCalculation.tsx`
```typescript
useStatisticsCalculation(backendUrl, basePathIdx, cameraOptions, imageCount, config, dataSource?) -> {
    compute, jobStatus, constraints, enabledMethods,
}
// dataSource param: derives type_name/source_endpoint directly (avoids async config race)
// Calls: POST /backend/statistics/calculate, GET /statistics/status/*, /constraints
```

#### `useVideoMaker.tsx`
```typescript
useVideoMaker(config) -> {
    createVideo, listVideos, videoStatus, deleteVideo, inspectData,
}
// Calls: POST /backend/video/create, GET /video/list_videos, /status/*, /inspect, DELETE /delete/*
```

#### Calibration hooks
```typescript
useCalibration(config)                  // General calibration state
useScaleFactorCalibration(config)       // POST /backend/calibrate/scale_factor/run
useDotboardCalibration(config)          // POST /backend/calibrate/dotboard/run, /detect_one
useChArUcoCalibration(config)           // POST /backend/calibrate/charuco/run
useDotboardCalibration(config)          // POST /backend/calibrate/dotboard/run
useStereoCalibration(config)            // POST /backend/calibrate/stereo_dotboard/run
useStereoCharucoCalibration(config)     // POST /backend/calibrate/stereo_charuco/run
useCalibrationValidation(config)        // Validation state for calibration
useCalibrationImageViewer(config)       // GET /backend/calibrate/calibration_image
useCalibrationSnapshot(basePathIdx)     // GET/POST /backend/calibration/snapshot - check/load saved calibration
```

#### `filterDefinitions.ts`
Defines available image filter types and their parameter schemas for the UI.

### Key Components

#### Viewers
- **`VectorViewer.tsx`** - Main vector field display. Uses `useVectorViewer` hook. Canvas-based rendering with hover tooltips, colorbar, data source switching.
- **`ImagePairViewer.tsx`** - Side-by-side raw/processed image display with frame navigation. Uses `useImagePair`.
- **`VideoMaker.tsx`** - Video creation UI. Uses `useVideoMaker`.
- **`CalibrationImageViewer.tsx`** - Calibration image display with detection overlay.
- **`zoomableCanvas.tsx`** - Reusable pan/zoom canvas component (used by VectorViewer and ImagePairViewer). Uses canvas-based overlay for detection dots (not SVG) and Uint32Array LUT for fast colormap pixel writes.
- **`colorbar.tsx`** - Matplotlib-style colorbar component.

#### Setup Panels
- **`PathsConfig.tsx`** - Source/base path configuration, camera count.
- **`ImageConfig.tsx`** - Image format, num_images, pairing preset (unified dropdown), start_index, frame pair preview.
- **`Calibration.tsx`** - Calibration method selector (tabs for each method).
- **`ScaleFactorCalibration.tsx`** / `DotboardCalibration.tsx` / `ChArUcoCalibration.tsx` / `PolynomialCalibration.tsx` / `StereoCalibration.tsx` / `StereoCharucoCalibration.tsx` - Method-specific calibration UIs.
- **`Masking.tsx`** - Mask configuration + polygon editor.
- **`InstantaneousPIV.tsx`** / `EnsemblePIV.tsx` - PIV processing parameter panels (window sizes, passes, overlap).
- **`RunPIV.tsx`** - PIV execution controls with progress monitoring.
- **`POD.tsx`** - Proper Orthogonal Decomposition settings.
- **`ValidationAlert.tsx`** - Shows file validation results.

#### Shared Components
- **`CameraSelector.tsx`** - Camera number dropdown.
- **`ColormapSelect.tsx`** - Colormap picker.
- **`DataSourceToggle.tsx`** - Calibrated/uncalibrated/merged/stereo data source selector.
- **`OutlierDetectionSettings.tsx`** - Outlier detection parameter UI.
- **`InfillingSettings.tsx`** - Vector infilling parameter UI.
- **`PerformanceSettings.tsx`** - Dask/threading performance settings.
- **`FilterComponents.tsx`** - Dynamic filter chain UI (uses filterDefinitions).
- **`PolygonMaskEditor.tsx`** - Interactive polygon drawing on image canvas.

### Contexts
- **`PivJobContext.tsx`** - React context for sharing PIV job state across components.

### Lib
- **`utils.ts`** - `cn()` (tailwind class merger)
- **`defaultConfig.ts`** - Default config shape for initialization
- **`imageUtils.ts`** - Base64 image handling utilities
- **`colormaps.ts`** - Colormap definitions for the frontend colorbar

---

## Processing Pipeline

> The processing engine runs separately from the GUI (as subprocess or CLI). It uses Dask for distributed computation and C extensions (via ctypes) for performance-critical FFT cross-correlation and Gaussian fitting.

### Entry Points

| Mode | Entry | Invoked By |
|------|-------|-----------|
| Instantaneous | `python -m pivtools_core.instantaneous` → `main()` | CLI or `piv_runner.py` subprocess |
| Ensemble | `python -m pivtools_core.ensemble` → `main()` | CLI or `piv_runner.py` subprocess |

### Instantaneous vs Ensemble

| Aspect | Instantaneous | Ensemble |
|--------|---------------|----------|
| **Goal** | Independent velocity field per image pair | Time-averaged velocity + Reynolds stresses from all pairs |
| **Task granularity** | 1 task = 1 batch (correlate + save, atomic) | 1 task = 1 batch (correlate for accumulation) |
| **Result** | `B00001.mat` ... `B0NNNN.mat` per pair | Single `ensemble_result.mat` after all passes |
| **Peak fitting** | Per-pair, inside C code | After ALL pairs accumulated, distributed across workers |
| **Multi-pass** | Each pair through all passes independently | All pairs through one pass, then next pass |
| **Background subtraction** | N/A | `R_AB = <A*B> - <A>*<B>` (single-pass formula) |
| **Reynolds stresses** | Not computed (use statistics module) | `UU = sig_AB - sig_A` (displacement variance) |

### Data Pipeline (Both Modes)

```
1. VALIDATE    validate_config(config)
2. CLUSTER     start_cluster() → LocalCluster or SLURMCluster
3. PER (path, camera):
   a. LOAD      load_images() → da.Array(N, 2, H, W) [lazy, ~N KB]
   b. SCATTER   scatter_immutable_data() → broadcast cache + masks to workers
   c. RECHUNK   rechunk_for_batched_processing(batch_size) → (batch_size, 2, H, W) chunks
   d. FILTER    create_filter_pipeline() → map_blocks(apply_all_filters_slim) [lazy]
   e. PROCESS   sliding window: submit filter→correlate tasks, bounded memory
   f. SAVE      save results + coordinates.mat
```

**Instantaneous step (e):** `correlate_and_save_batch()` → multi-pass correlation + peak finding + save per batch. Uses `as_completed()` for any-order processing.

**Ensemble step (e):** `process_pass_sliding_window()` → accumulate correlation sums per worker, reduce, then `accumulator.finalize_pass()` → distributed Gaussian (or k-space) fitting → extract velocities + stresses.

### `pivtools_core/instantaneous.py`

**Goal:** Dask-native instantaneous PIV pipeline: images → per-pair multi-pass correlation → save .mat files.

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `main()` | — | None | CLI entry. Validates config, starts cluster, iterates paths×cameras, calls `run_instantaneous_piv()`. |
| `run_instantaneous_piv` | `config, client, camera_num, source_path, output_path, base_path, vector_masks, pixel_mask` | `List[str]` | Full pipeline for one camera: load → scatter → rechunk → filter → sliding window → save coords. |
| `process_instantaneous_sliding_window` | `client, images, num_chunks, workers, scattered_config, pass_idx, scattered, config, output_path` | `List[str]` | Bounded sliding window: `max_in_flight = min(2*workers, chunks)`. Uses `as_completed()`. |
| `signal_handler` | `signum, frame` | None | SIGTERM/SIGINT → cancel futures → `os._exit(1)`. |

### `pivtools_core/ensemble.py`

**Goal:** Dask-distributed ensemble PIV: lazy image loading → sliding window accumulation → multi-pass with predictor refinement → Gaussian/k-space fitting → velocity + stress fields.

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `main()` | — | None | CLI entry. Sets `MALLOC_TRIM_THRESHOLD_=0`, starts cluster, iterates paths×cameras. |
| `run_ensemble_piv` | `config, client, camera_num, source_path, output_path, base_path, vector_masks, pixel_mask` | `str` | Full pipeline. Multi-pass: scatter predictor → sliding window → finalize pass → extract predictor → next pass. |
| `process_pass_sliding_window` | `client, images, num_chunks, workers, scattered_config, pass_idx, scattered_predictor, scattered, config, output_path` | `dict` | Bounded memory accumulation. Round-robin worker pinning. Chained per-worker submission. |
| `signal_handler` | `signum, frame` | None | Same as instantaneous. |

### Ensemble Key Data Structures

**Correlation batch result** (flows between workers):
```python
{"corr_AA_sum": ndarray, "corr_BB_sum": ndarray, "corr_AB_sum": ndarray,  # flat: n_win * corr_h * corr_w
 "warp_A_sum": ndarray, "warp_B_sum": ndarray,  # (H, W)
 "n_images": int, "n_win_x": int, "n_win_y": int,
 "smoothed_predictor": Optional[ndarray], "vector_mask": Optional[ndarray]}
```

**`PIVEnsemblePassResult`** (20+ fields): `ux_mat`, `uy_mat`, `UU_stress`, `VV_stress`, `UV_stress`, `peakheight`, `nan_reason`, `sig_AB_x/y/xy`, `sig_A_x/y/xy`, `c_A/B/AB`, `b_mask`, `pred_x/y`, `window_size`, `win_ctrs_x/y`

### Stereo Ensemble PIV (Correlation-of-Correlations)

Extends ensemble PIV to stereo setups, computing 3D velocity + 6 Reynolds stresses using a cross-camera correlation-of-correlations (CoC) approach.

**Key files:**
| File | Purpose |
|------|---------|
| `pivtools_core/stereo_ensemble.py` | Dask-native orchestration (mirrors `ensemble.py`) |
| `pivtools_cli/piv/piv_backend/cpu_stereo_ensemble.py` | CPU backend (`StereoEnsembleCorrelatorCPU`, composition over `EnsembleCorrelatorCPU`) |
| `pivtools_cli/piv/piv_backend/stereo_ensemble_accumulator.py` | Dual-camera + CoC buffers, `finalize_pass()` → 3D velocity + 6 stresses |
| `pivtools_cli/piv/stereo_ensemble_result.py` | `PIVStereoEnsemblePassResult` dataclass |

**Config:** `stereo_ensemble_piv` section — all keys fall back to `ensemble_piv` if null. `resume_from_pass` (1-based, 0=no resume) does NOT fall back to ensemble.

**CLI:** `stereo-ensemble` command. **Output:** `base_path/uncalibrated_piv/{N}/Stereo Cam{A}_Cam{B}/stereo_ensemble/` → `stereo_ensemble_result.mat`, `stereo_coordinates.mat`

**CoC math:** Per-frame N=1 C lib calls per camera → per-window FFT cross-correlate → accumulate. Gives `Sigma_12_xx = R_xx - sin²θ·R_zz`. Combined with `A = (Sigma_11_xx + Sigma_22_xx)/2 = R_xx + sin²θ·R_zz` → `R_xx = (A+B)/2`, `R_zz = (A-B)/(2sin²θ)`.

**3D velocity:** `ux = (d1_x + d2_x)/2`, `uy = (d1_y + d2_y)/2`, `uz = (d1_x - d2_x)/(2*sin_th)`

**Sign conventions on save:** Same as standard ensemble (uy negated, UV_stress negated, VW_stress negated, pred_y negated). `load_stereo_ensemble_result()` in `save_results.py` reverses these on load.

**Worker pattern:** `correlate_stereo_batch_and_accumulate()` recreates `StereoEnsembleCorrelatorCPU` per batch (dewarp maps ~20ms, negligible vs correlation cost). Uses `+` not `+=` for Dask retry safety. Dual-camera sliding window: each chunk triggers 2 filter futures (cam1 + cam2), `wait([cam1_future, cam2_future])` before submitting correlation.

**Diagnostics:** `stereo_ensemble_store_planes` saves correlation planes per pass. `stereo_ensemble_save_diagnostics` saves dewarped images. Both fall back to ensemble equivalents.

### K-Space Transfer Function Fitting [BETA]

> **WARNING: K-space fitting is BETA. Experimental — API may change.**

Alternative to Levenberg-Marquardt Gaussian fitting. Works in Fourier domain: `T(k) = F(R_AB) / sqrt(F(R_AA) * F(R_BB))` — particle shape cancels, reducing 16 → 6 parameters. Pure Python/NumPy/SciPy (~50-100x slower per window than C Gaussian, but only ~25% total runtime impact). Improves Reynolds stress accuracy by 40-90%.

- **Config:** `ensemble_piv.fit_method: kspace`, `kspace_snr_threshold: 3.0`
- **File:** `pivtools_cli/piv/piv_backend/kspace_fitting.py`
- **Status codes:** 0=success, 1=no converge, 2=low SNR, 3=displacement > 3/4 window, 5=negative variance (consistent with Gaussian codes)
- **Noise estimation:** High-k annular ring (0.4 < |k| < 0.5) for rotationally symmetric noise floor estimate
- **Soft weighting:** Anisotropic decay `exp(-k_x²/k0_x² - k_y²/k0_y²)` matching elliptical transfer function shape; k_max cap at 0.35 (soft) or 0.25 (hard)
- **1D regressions:** Forced through origin (DC-normalised T(0)=1); per-axis k_max bounds; window-size-aware k_min = 1.5/N
- **Gradient correction:** K-space does not estimate σ_A (particle image variance) — it's algebraically cancelled in Fourier space. When gradient correction is enabled, only the window averaging term (L²/12) is applied; the particle extent term (σ_A) is omitted. This is the dominant correction (~95-97% of total). `sig_A_x/y/xy` fields are saved as zero in the output .mat file.

### Sum Fitting Window Feature

Computes correlations on larger `sum_window` but extracts central `sum_fitting_window` for storage and fitting. Central extraction happens in C code (`bulkxcorr2d_accumulate`). Benefits: 4x memory reduction, 4x faster fitting. Config: `ensemble_sum_fitting_window_enabled`, `ensemble_sum_fitting_window`.

### Supporting Modules

#### `pivtools_core/validation.py`
Validates config before processing: checks paths, file counts, indexing consistency.

| Function | Returns | Description |
|----------|---------|-------------|
| `validate_config(config)` | `(bool, str, List[str])` | `(is_valid, error_msg, warnings)`. Calls `validate_ensemble_config` when ensemble enabled. |
| `validate_ensemble_config(config)` | `(bool, List[str], List[str])` | `(is_valid, errors, warnings)`. Validates ensemble types, windows, overlaps, sum windows, fit method, resume_from_pass. |
| `validate_batch_size_for_pod(config, batch_size)` | `(bool, str)` | Warns if batch_size < 20 for POD filter |
| `log_validation_result(is_valid, error_msg, warnings, config)` | None | Logs formatted results |

#### `pivtools_core/window_utils.py`
Window positioning and sizing for both standard and single-mode ensemble PIV.

| Function | Returns | Description |
|----------|---------|-------------|
| `compute_window_centers(image_shape, window_size, overlap, validate)` | `WindowCenterResult` | Standard PIV window grid. Y-axis anchored to bottom. |
| `compute_window_centers_single_mode(image_shape, window_size, sum_window, overlap, validate)` | `WindowCenterResult` | Single mode: spacing from small window, positions in padded coords. |
| `compute_padding_for_single_mode(window_size, sum_window)` | `(top, bottom, left, right)` | Asymmetric padding (ceil top/left, floor bottom/right). |
| `apply_single_mode_padding(image, window_size, sum_window, pad_value)` | `(padded, padding)` | Supports 2D/3D/4D. |
| `validate_window_configuration(image_shape, window_size, overlap, ...)` | `(bool, str)` | Checks feasibility. |
| `get_window_grid_shape(image_shape, window_size, overlap, ...)` | `(n_win_y, n_win_x)` | For pre-allocation. |

`WindowCenterResult`: `win_ctrs_x`, `win_ctrs_y`, `n_win_x`, `n_win_y`, `win_spacing_x`, `win_spacing_y`, `padding`

#### `pivtools_core/batch_utils.py`
Unified batch iteration: "paths outer, cameras inner" pattern.

| Function | Returns | Description |
|----------|---------|-------------|
| `iter_batch_targets(base_paths, active_paths, cameras, ...)` | `List[BatchTarget]` | Generates processing targets. Supports merged/stereo. |
| `run_batch_with_progress(targets, process_fn, progress_callback)` | `List[Dict]` | Iterates with exception handling per target. |
| `count_batch_targets(num_paths, num_cameras, ...)` | `int` | For progress estimation. |

`BatchTarget`: `path_idx`, `base_path`, `source_path`, `camera`, `is_merged`, `.label`

#### `pivtools_core/coordinate_utils.py`
Extracts coordinates from MATLAB .mat files.

| Function | Returns | Description |
|----------|---------|-------------|
| `extract_coordinates(coords, run)` | `(x, y)` | 1-based run number. Handles multi-run and single-run structs. |
| `extract_coordinate_bounds(coords, run)` | `(xmin, xmax, ymin, ymax)` | Coordinate extent. |
| `get_num_coordinate_runs(coords)` | `int` | Number of runs in coordinate struct. |

#### `pivtools_cli/processing/dask_pipeline.py`
Dask-centric utilities shared by both pipelines.

| Function | Description |
|----------|-------------|
| `rechunk_for_batched_processing(images, batch_size)` | `(N,2,H,W)` per-pair → `(batch_size,2,H,W)` chunks |
| `create_filter_pipeline(images, config, pixel_mask, ...)` | `map_blocks(apply_all_filters_slim)` — lazy filtering |
| `apply_all_filters_slim(block, spatial_specs, temporal_specs, ...)` | Applies pixel mask → spatial → temporal filters |
| `scatter_immutable_data(client, config, vector_masks, pixel_mask, ...)` | Creates correlator, broadcasts cache + masks |
| `correlate_and_save_batch(batch, start_img_idx, config, ...)` | [Instantaneous] Correlate + save per batch |
| `correlate_single_batch_and_accumulate(batch, accumulated, ...)` | [Ensemble] Accumulate correlation sums |
| `reduce_ensemble_results(results)` | Merges per-worker accumulated dicts |
| `extract_predictor_field(pass_result)` | `np.stack([uy, ux], axis=-1)` for next pass |

**Spatial filters:** gaussian (sigma), median (size), norm, maxnorm, lmax (all via `scipy.ndimage`)
**Temporal filters:** pod (SVD-based), time

### Dask Patterns

**Lazy loading:** `dask.delayed` per image pair (~1 KB each), stacked into `da.Array(N, 2, H, W)`. No data loaded until `.compute()`.

**Sliding window I/O:** Bounds memory to ~2 batches per worker. `max_in_flight = min(2*num_workers, num_chunks)`. Completed filter futures are replaced, keeping pipeline full.

**Data scattering:** Immutable data (correlator cache, masks, config) broadcast once. Predictor field re-scattered per pass.

**Worker pinning (ensemble):** Round-robin `workers[chunk_idx % num_workers]`. Per-worker accumulation via chained futures (uses `+` not `+=` for Dask retry safety).

**Memory management:**
- `gc.collect()` on client between cameras — **NOT on workers** (FFTW causes SIGSEGV)
- `MALLOC_TRIM_THRESHOLD_=0` uploaded to workers (prevents glibc memory hoarding)
- Explicit `del` of accumulated data and scattered predictor per pass

---

## Build System & C Extensions

### Package: `pivtools` v0.4.3

- **Build backend:** `setuptools>=61.0` + `wheel` + `cibuildwheel>=2.16`
- **Python:** `>=3.12` (targets: 3.12, 3.13, 3.14)
- **License:** BSD-3-Clause
- **Entry points:** `pivtools-cli` → `pivtools_cli.cli:main`, `pivtools-gui` → `pivtools_gui.app:main`

### Key Dependencies

| Category | Packages |
|----------|----------|
| **Core** | dask==2025.7.0, numpy==2.2.6, scipy==1.16.1, opencv-python==4.12.0.88, pandas==2.3.1 |
| **CLI** | numba==0.61.2, scikit-image==0.25.2, scikit-learn==1.7.2, lvpyio (non-macOS) |
| **GUI** | Flask==3.1.1, flask-cors, matplotlib==3.10.5, imageio-ffmpeg |

### C Libraries (compiled at build time via `setup.py`)

| Library | Sources | External Deps | Purpose |
|---------|---------|---------------|---------|
| `libbulkxcorr2d` | `peak_locate_lm.c`, `PIV_2d_cross_correlate.c`, `xcorr.c`, `xcorr_cache.c` | FFTW3f, OpenMP | Cross-correlation engine (instantaneous + ensemble accumulation) |
| `libinterp2custom` | `interp2custom.c` | OpenMP | LUT-based 2D image warping interpolation |
| `libmarquadt` | `marquadt_gaussian.c` | GSL, OpenMP | Ensemble 16-parameter stacked Gaussian fitting |

**Python-C interface:** All via `ctypes.CDLL` at runtime. No Cython/cffi.

**Array conventions:**
- `libbulkxcorr2d` / `libmarquadt`: C-contiguous (row-major)
- `libinterp2custom`: Fortran-contiguous (column-major) — note different `SUB2IND_2D` macro

### C Extension Functions

#### `PIV_2d_cross_correlate.c` — Bulk PIV Engine (EXPORTED)

| Function | Description |
|----------|-------------|
| `bulkxcorr2d(fImageA_stack, fImageB_stack, fMask, nImageSize[2], N_images, fWinCtrsX/Y, nWindows[2], fWindowWeightA/B, bEnsemble, nWindowSize[2], nPeaks, iPeakFinder, → fPkLocX/Y, fPkHeight, fSx/y/xy, fCorrelPlane_Out)` | **Instantaneous:** Parallel over (image × window). Extract sub-images → apply taper → subtract mean → xcorr → LM peak fit. |
| `bulkxcorr2d_accumulate(... nFitWindowSize[2] ...)` | **Ensemble:** Accumulates correlation planes per window. Parallel over windows, sequential over images. Supports central region extraction. |

#### `xcorr.c` — FFT Cross-Correlation

| Function | Description |
|----------|-------------|
| `xcorr(a, b, c, N)` | Full xcorr: create plan → execute → destroy. Thread-safe (OMP critical). |
| `xcorr_create_plan(N, sPlan*)` | Pre-create FFTW plans (NOT thread-safe). Tries MEASURE then ESTIMATE. |
| `xcorr_preplanned(a, b, c, sPlan*)` | Thread-safe xcorr with pre-created plan. `fftshift(IFFT(FFT(a) .* conj(FFT(b))))`. |

#### `peak_locate_lm.c` — Levenberg-Marquardt Peak Finder (internal, not exported)

Self-contained LM Gaussian fitting (no GSL). Fit types: 3=parabolic, 4=circular Gaussian, 5=elliptical, 6=rotated elliptical (6 DOF with inverse covariance). Max 20 iterations, tolerance 1e-6, 5×5 peak region.

#### `marquadt_gaussian.c` — Ensemble Gaussian Fitting (EXPORTED)

`fit_stacked_gaussian_batch_export(...)`: Fits 16 parameters per window (3 amplitudes, 3 offsets, 3 sigma_A, 3 delta, 2 center_A, 2 center_AB). Uses **delta parameterization**: params[9,10,11] internally represent `delta = sigma_AB - sigma_A` (Reynolds stress directly), with `sigma_AB` reconstructed as `sigma_A + max(delta, 0)` during fitting. Output converts back to `sigma_AB` so Python sees total widths. Split convergence tolerances: XTOL=1e-4 (relaxed for small deltas), GTOL=FTOL=1e-6 (scale-independent). Constraint-aware Jacobian: cols [9,10] zeroed when delta < 0. Uses GSL `gsl_multifit_nlinear` with Cholesky solver + geodesic acceleration. OpenMP `schedule(dynamic, 16)`. Python initial guess builders (`_build_initial_guess`, `_build_initial_guesses_vectorized`) convert `sigma_AB → delta` before passing to C.

#### `interp2custom.c` — Image Warping (EXPORTED)

`interp2custom(y, N, f_i, f_j, yi, n_interp)`: LUT-based 2D interpolation (32768-entry table). Kernel types: Lanczos (windowed sinc) or Gaussian. OpenMP-parallel.

#### `xcorr_cache.c` — FFTW Wisdom Persistence

Saves/loads FFTW wisdom to `~/.pypivtools_fftw_wisdom`. Uses C11 atomics for thread-safe initialization.

### Static Library Bundling

FFTW and GSL are bundled as static archives for reproducible builds:
```
static_fftw/{windows,linux,macos_arm64}/  → libfftw3f.{lib,a}, libfftw3f_omp.a
static_gsl/{windows,linux,macos_arm64}/   → libgsl.{lib,a}, libgslcblas.{lib,a}
```
Override with: `FFTW_INC_DIR`, `FFTW_LIB_DIR`, `GSL_DIR`

### Compilation

| Platform | Compiler | Flags | Extension |
|----------|----------|-------|-----------|
| Windows | MSVC (`cl`) | `/O2 /std:c11 /experimental:c11atomics /openmp:experimental /MT` | `.dll` |
| macOS | `gcc-15` (Homebrew) | `-O3 -fPIC -fopenmp -DFFTW_THREADS` | `.so` |
| Linux | `gcc` | `-O3 -fPIC -fopenmp -DFFTW_THREADS` | `.so` |

### CI/CD (`.github/workflows/publish-to-pypi.yml`)

Triggered on GitHub release or manual dispatch. Builds wheels for {ubuntu, macos, windows} × {cp312, cp313, cp314}. macOS: arm64 only. Linux: x86_64 (manylinux2014). Windows: AMD64. Publishes to PyPI via `twine`.

### CLI (`pivtools_cli/cli.py`)

| Command | Description |
|---------|-------------|
| `init` | Create default `config.yaml` |
| `instantaneous` | Run instantaneous PIV |
| `ensemble` | Run ensemble PIV |
| `detect-planar` / `detect-charuco` | Detect calibration targets |
| `detect-stereo-planar` / `detect-stereo-charuco` | Stereo calibration detection |
| `apply-calibration` / `apply-stereo` | Apply calibration (px → m/s). `--align-coordinates` flag auto-applies global alignment. |
| `align-coordinates` | Apply global coordinate alignment to calibrated vectors (reads datum/overlap from config). `--force` flag overrides idempotency guard. |
| `transform` | Geometric transforms (`flip_ud`, `flip_lr`, `rotate_90_cw/ccw`, `rotate_180`, `swap_ux_uy`, `invert_ux_uy`, `invert_ux`, `invert_uy`, `scale_velocity:N`, `scale_coords:N`) |
| `merge` | Multi-camera Hanning window blending |
| `statistics` | Mean, TKE, vorticity, divergence, gamma |
| `video` | Visualization video (instantaneous only) |

Common flags: `--active-paths`, `--type-name`, `--source-endpoint` (`regular`/`merged`/`stereo`)

### Profiling (`profile/profile_piv.py`)

Standalone PIV profiling script that bypasses Dask entirely. Loads real images from disk, creates a minimal Config, and calls `InstantaneousCorrelatorCPU.correlate_batch()` directly with per-section timing instrumentation.

**Instrumentation** (in `cpu_instantaneous.py`):
- `self.profiling_enabled = False` — disabled by default, zero overhead
- `self.threading_enabled = True` — set False to bypass thread pool (for A/B comparison)
- `self._profile_section(pass_idx, section)` — context manager that times named code sections
- `self.get_profile_summary()` — returns `{pass_idx: {section_name: elapsed_seconds}}`
- `self._run_parallel(fn, args_list)` — dispatches to pool or direct calls based on `threading_enabled`

**9 timed sections in `correlate_batch`:** `predictor_corrector`, `set_lib_args`, `bulkxcorr2d`, `post_processing`, `outlier_detection`, `secondary_peaks`, `infilling`, `padding_stacking`, `result_construction`

**3 sub-timings in `_predictor_corrector_batch` (pass > 0):** `pc_gaussian_smooth`, `pc_dense_and_predictor_remap`, `pc_mesh_and_image_warp`

**Usage:**
```
python profile/profile_piv.py 4mp                     # 4MP images, 3 iterations
python profile/profile_piv.py 25mp --iterations 5     # 25MP images, 5 iterations
python profile/profile_piv.py both --threads 8        # Both presets, 8 OMP threads
python profile/profile_piv.py 4mp --no-outlier        # Disable outlier detection
python profile/profile_piv.py 4mp --windows 64,32     # Custom pass sizes
python profile/profile_piv.py 4mp --pairs 12          # Production batch size
python profile/profile_piv.py 4mp --no-threading      # Disable thread pool (baseline)
```

**Thread parallelism in correlators:**

Both `cpu_instantaneous.py` and `cpu_ensemble.py` use `cv2.setNumThreads(1)` + class-level `ThreadPoolExecutor(omp_threads)` for thread-parallel GIL-releasing operations. The pool is idle during C library calls (OpenMP) and vice versa — no contention within a worker. Threading gives **2.2x overall speedup** at batch size N=12.

**GIL behaviour matters:** cv2.remap and scipy.ndimage release the GIL (2.7-3.2x speedup with 4 threads). Biharmonic infilling is ~67% GIL-bound (no benefit from threading — 0.95x). Use `local_median` for mid-pass infilling (8.8x faster than biharmonic, threads at 2.1x).

- **Instantaneous:** Gaussian smoothing, fused dense+predictor remap, fused mesh+warp, outlier detection, and infilling all use `self._run_parallel()`. Per-image mesh computed inside each `_warp_pair` thread (avoids large `(N,H,W,2)` allocations).
- **Ensemble:** `_get_image_prime_batch()` parallelizes the `for n in range(N)` cv2.remap loop via `self._pool`. Benefits all 3 callers: `correlate_batch_for_accumulation`, `compute_warp_sums_only`, `correlate_mean_subtracted_batch`. Mesh is NOT per-image (single predictor field → single mesh for all images), so no fusing opportunity.

---

### Processing Config Keys

**`processing` block:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | str | `"cpu"` | `"cpu"` or `"gpu"` |
| `omp_threads` | int | 1 | OpenMP threads for C extensions |
| `dask_workers_per_node` | int | 1 | Dask worker count |
| `dask_memory_limit` | str | `"4GB"` | Per-worker memory |
| `cluster_type` | str | `"local"` | `"local"` or `"slurm"` |
| `open_dashboard` | bool | `false` | Auto-open Dask dashboard in browser on cluster start |

**`instantaneous_piv` block:**

| Key | Property | Description |
|-----|----------|-------------|
| `window_size` | `instantaneous_window_sizes` | List of `[h,w]` per pass |
| `overlap` | `instantaneous_overlaps` | List of `%` per pass (broadcasts single to all) |
| `runs` | `instantaneous_runs_0based` | Which passes to save (0-based) |
| `peak_finder` | `peak_finder` | `"gauss3"`→3, `"gauss4"`→4, `"gauss5"`→5, `"gauss6"`→6 DOF |

**`ensemble_piv` block:**

| Key | Property | Description |
|-----|----------|-------------|
| `window_size` | `ensemble_window_sizes` | Falls back to instantaneous if unset |
| `overlap` | `ensemble_overlaps` | With validation and broadcast |
| `type` | `ensemble_type` | List of `"std"` or `"single"` per pass |
| `sum_window` | `ensemble_sum_window` | `[h,w]` for single mode |
| `fit_method` | `ensemble_fit_method` | `"gaussian"` or `"kspace"` [BETA] |
| `background_subtraction_method` | `ensemble_background_subtraction_method` | `"correlation"` or `"image"` |
| `gradient_correction` | `ensemble_gradient_correction` | Reynolds stress gradient correction |
| `resume_from_pass` | `ensemble_resume_from_pass` | 1-based pass to resume (0=fresh start) |
| `correlation_normalization` | `ensemble_correlation_normalization` | `"none"` (default) or `"per_frame"` — per-frame mean-sub + energy normalization |

**Outlier detection / infilling** (parallel structure for instantaneous and ensemble):
`outlier_detection.enabled`, `.methods` (list of `{type, threshold, epsilon}`), `infilling.mid_pass`, `.final_pass`

**Ensemble stress post-processing** (final pass, STEP 7c in `single_pass_accumulator.py`):
After velocity outlier detection + infilling, runs stress-specific quality checks:
- **Stress outlier detection:** Reuses config-driven outlier methods (`median_2d`, `sigma`) on `(UU_stress, VV_stress)` — catches windows with plausible velocity but bad stress estimates. Velocity-specific methods (`peak_mag`, `div_vort`) are filtered out automatically.
- **Realizability constraint:** Cauchy-Schwarz check `UV² ≤ UU·VV` — flags physically impossible stress states.
- Stress outliers are infilled from neighbors (velocity fields untouched).
- `nan_reason` codes: 0=success, -1=masked, 10=velocity outlier, **11=stress outlier**.

---

## ASCII Dependency Map

```
                                ┌──────────────────────┐
                                │    config.yaml       │ ◄── single source of truth
                                └──────────┬───────────┘
                                           │
                                ┌──────────▼───────────┐
                                │  pivtools_core/      │
                                │  config.py (Config)  │ ◄── singleton: get_config()
                                └──┬──┬──┬──┬──┬──┬───┘
                                   │  │  │  │  │  │
     ┌─────────────────────────────┘  │  │  │  │  └──────────────────────────┐
     ▼                                │  │  │  │                             ▼
  image_handling/              paths  │  │  │  validation.py          window_utils.py
  ├─load_images.py             .py   │  │  │                          batch_utils.py
  ├─path_utils.py                    │  │  │                          coordinate_utils.py
  └─readers/ (lavision,cine,generic) │  │  │
     │                               │  │  │
     │  ┌────────────────────────────┘  │  └──────────────────────────────┐
     │  │              ┌────────────────┘                                  │
     │  │              │                                                   │
     │  ▼              ▼                                                   ▼
     │  vector_loading.py                                    ┌─────────────────────────┐
     │  (.mat I/O)                                           │ pivtools_cli/           │
     │                                                       │  cli.py (CLI entry)     │
     ├───────────────────────────────────────────────────┐   │  piv_cluster/cluster.py │
     │                                                   │   │  piv/piv_backend/       │
     │         ┌─── GUI PATH ───┐    ┌─ PROCESSING PATH ─┤   │   ├─factory.py          │
     │         │                │    │                    │   │   ├─cpu_instantaneous.py │
     ▼         ▼                │    ▼                    │   │   ├─cpu_ensemble.py      │
  ┌─────────────────────┐  ┌───────────────────────────┐ │   │   ├─kspace_fitting.py[B] │
  │ pivtools_gui/app.py │  │ pivtools_core/            │ │   │   └─single_pass_accum.  │
  │ (Flask + /backend/) │  │  instantaneous.py         │ │   │  piv/save_results.py    │
  │                     │  │  ensemble.py              │ │   │  processing/            │
  │ piv_runner.py ──────┼──►                           │─┼──►│   └─dask_pipeline.py    │
  │  (spawns subprocess)│  │  main() → Dask cluster    │ │   │  lib/ (C extensions)    │
  └──┬──┬──┬──┬──┬──┬──┘  │  → sliding window I/O     │ │   │   ├─libbulkxcorr2d      │
     │  │  │  │  │  │     │  → correlate → save        │ │   │   ├─libinterp2custom    │
     ▼  ▼  ▼  ▼  ▼  ▼     └───────────────────────────┘ │   │   └─libmarquadt         │
  calibration/  transforms/                              │   └─────────────────────────┘
  plotting/     merging/                                 │              │
  statistics/   video/                                   │     via ctypes.CDLL
  masking/                                               │              │
     │                                                   │              ▼
     │  All GUI modules import:                          │   ┌──────────────────────┐
     │  - pivtools_core.config                           │   │ C Libraries          │
     │  - pivtools_gui.utils                             │   │  xcorr.c (FFTW3f)   │
     │  - job_manager (shared singleton)                 │   │  PIV_2d_xcorr.c     │
     └───────────────────────────────────────────────────┘   │  peak_locate_lm.c   │
                                                             │  marquadt_gaussian.c │
  ┌─────────────────────────────────────────────────────┐    │   (GSL + OpenMP)     │
  │             PIVTOOLs-GUI/src/ (React)               │    │  interp2custom.c     │
  │                                                     │    │  xcorr_cache.c       │
  │  page.tsx ──┬─► PathsConfig ──► useConfigUpdate     │    │   (FFTW wisdom)      │
  │             ├─► ImageConfig                         │    └──────────────────────┘
  │             ├─► Calibration ──► use*Calibration      │
  │             ├─► Masking ─────► useMaskingConfig      │   [B] = BETA feature
  │             ├─► InstPIV ─────► useInstPivConfig      │
  │             ├─► EnsemblePIV ─► useEnsemblePivConfig  │
  │             ├─► RunPIV ──────► usePivRunner           │
  │             ├─► ImagePairViewer ► useImagePair        │
  │             ├─► VectorViewer ─► useVectorViewer       │
  │             │   ├─► useVectorMerging                  │
  │             │   └─► useStatisticsCalculation          │
  │             └─► VideoMaker ──► useVideoMaker          │
  │                                                       │
  │  All hooks: fetch('/backend/...') → Flask endpoints   │
  └───────────────────────────────────────────────────────┘
```

---

## Data Formats

### Frontend ↔ Backend Communication

| Direction | Format | Example |
|-----------|--------|---------|
| Config read | `GET /backend/config` → full YAML as JSON | `{"paths": {...}, "images": {...}}` |
| Config write | `POST /backend/update_config` ← partial JSON | `{"images": {"num_images": 500}}` |
| Images | base64-encoded PNG or JPEG strings | `"iVBORw0KGgo..."` |
| Plots | base64-encoded PNG from matplotlib | same |
| Job status | JSON with progress | `{"status": "running", "progress": 45, "job_id": "..."}` |

### Image Formats Supported
`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.im7`, `.set`, `.cine`, `.raw`, `.cr2`, `.nef`, `.arw`

Container formats: `.set` and `.cine` (single file, multiple frames). `.im7` with `%` pattern is NOT a container.

### Vector .mat File Structure

**Instantaneous** (`B00001.mat` ... `B0NNNN.mat`, var: `piv_result`):
```
piv_result (multi-run object array or single struct):
  .ux        (H, W) float    x-velocity (negated for physical coords on save)
  .uy        (H, W) float    y-velocity
  .uz        (H, W) float    z-velocity (stereo only)
  .b_mask    (H, W) float    binary mask (1=valid)
  .win_ctrs_x/y               window centers
  .peak_mag  (H, W) float    peak height
  .predictor_field             predictor displacement
  .window_size                 [h, w]
```

**Ensemble** (`ensemble_result.mat`, var: `ensemble_result`):
```
ensemble_result (same struct convention, plus):
  .UU_stress  (H, W)   Reynolds normal stress (x)
  .VV_stress  (H, W)   Reynolds normal stress (y)
  .UV_stress  (H, W)   Reynolds shear stress
  .sig_AB_x/y/xy        displacement uncertainty (cross-corr)
  .sig_A_x/y/xy         particle size (auto-corr)
  .c_A/B/AB             Gaussian offsets
  .peakheight            normalized peak height
  .nan_reason            failure reason per window
  .pred_x/y              predictor displacement
```

**Coordinates** (`coordinates.mat`, var: `coordinates`):
```
coordinates.x  (H, W) or (N,) float16   x-coords (1-based, y=0 at bottom)
coordinates.y  (H, W) or (N,) float16   y-coords
```

**Masks** (`.mat`):
```
mask       (H, W) uint8     binary mask
polygons   cell array       [{index, name, points(Nx2)}]
```

### Output Directory Structure

```
base_path/
  uncalibrated_piv/{N}/Cam{C}/{type}/    B00001.mat ... coordinates.mat
  calibrated_piv/{N}/Cam{C}/{type}/      (after apply-calibration)
  calibrated_piv/{N}/Merged/{type}/      (after merge)
  stereo_calibrated/{N}/CamX_CamY/{type}/ (after stereo)
  statistics/{N}/...                      mean, TKE, vorticity, etc.
  videos/{N}/...                          .mp4 files
```

---

## Code Style & Conventions

### Python (GUI)
- Flask with Blueprints. Each module: `app/views.py` (routes) + separate business logic files.
- Config via `get_config()` singleton. Never pass config paths.
- Logging: `loguru.logger` (not stdlib).
- Camera normalization: `camera_number()` before use. Always 1-based.
- Long ops: daemon threads + `job_manager` singleton. Return `job_id` immediately.
- Images to frontend: base64 via `numpy_to_base64()`.

### Python (Processing)
- Entry points: `if __name__ == "__main__": main()` pattern.
- Dask-native: lazy arrays, `client.submit()`, `client.scatter()`, `as_completed()`.
- No global mutable state on workers. Uses `+` not `+=` for Dask retry safety.
- Signal handling: register SIGTERM/SIGINT, cancel futures, `os._exit(1)`.
- Memory: explicit `del`, `gc.collect()` on client only, `MALLOC_TRIM_THRESHOLD_=0`.
- Config properties return validated/coerced values (e.g., `omp_threads` → str, `image_format` → tuple).

### C Extensions
- Row-major indexing everywhere EXCEPT `interp2custom.c` (column-major, Fortran-contiguous).
- OpenMP for parallelism. FFTW plan creation in OMP critical sections.
- Error codes via `common.h`: 0=none, 1=nomem, 2=no_fwd_plan, 4=no_bwd_plan, 9=out_of_bounds.
- No dynamic allocation in hot loops. Pre-allocated buffers.
- FFTW wisdom cached to `~/.pypivtools_fftw_wisdom`.

### TypeScript Frontend
- Next.js (App Router), React 18+, TypeScript, shadcn/ui + Tailwind.
- Custom hooks own state. Components are presentational.
- Raw `fetch()` in hooks. All endpoints: `/backend/...`.
- Hook naming: `use{Feature}`. Each hook → one backend module.

### Shared
- Job lifecycle: `create_job()` → `update_job("running")` → `complete_job()` / `fail_job()`.
- Camera numbering: 1-based everywhere.
- Frame numbering: 1-based for pair numbers. Zero-based only in file naming.
- Errors: backend `{"error": "msg"}` + HTTP status; frontend hooks expose `error` state.

---

## Validation & Benchmarking

Benchmark scripts compare PIVTOOLs output against ground truth (DNS data).

**Key files:**
- `validation/README.md` — **Read this first.** Comprehensive reference for all benchmark scripts, data formats, and recorded results.
- `validation/benchmark_comparison.py` — Planar benchmark
- `validation/stereo_benchmark_comparison.py` — Stereo benchmark (hardcoded `data_root` in `main()`)

**Unit conventions:** PIVTOOLs stores velocity in m/s (×1000→mm/s for display), stresses in (m/s)² (×1e6→(mm/s)²).

**Ground truth format:** Auto-detects MATLAB v5 (`profiles.mat`) vs v7.3/HDF5 (`ensemble_statistics_full.mat`). HDF5 `ref_profile` has DNS velocity (2049 pts) but NO stresses; stresses come from `ensemble_stats` (255 pts) — never mix without interpolation.

**Common flags:** `--num-frames`/`-n` controls the subdirectory (e.g., `calibrated_piv/1000/`). y+ offset of `+1.0` works for current channel dataset.

---

## Gotchas & Common Pitfalls

### Config & Data
- `config.py` is very large (~2600 lines) — read in chunks
- `image_format` is always a tuple, even for single format
- `camera_folder()` in utils.py is DEPRECATED — use `config.get_camera_folder()`
- `zero_based_indexing` exists in TWO places: images (now `start_index`) and calibration (unchanged)
- `time_resolved` exists in TWO places: images (now stride-based) and `instantaneous_piv` (unchanged)
- Config sync to backend is async — hooks that read config for type_name/source_endpoint may see stale values; pass directly instead
- **Mixed dict key types in jsonify:** Flask's `jsonify()` calls `json.dumps(sort_keys=True)` which raises `TypeError` when dict has both int and str keys. Always use `str(camera_num)` as keys.
- **Statistics config legacy keys:** Backend defaults were `reynolds_stress`/`normal_stress`/`inst_fluctuations` but frontend `ALL_STAT_KEYS` uses `mean_stresses`/`inst_stresses`/`mean_peak_height`. Fixed with migration in `statistics_enabled_methods` property.
- **Calibration source change resets GC fields:** When `calibration_sources` changes in `update_config`, pixel-dependent global coordinate fields (`datum_pixel`, `overlap_points`, `overlap_pairs`, `invert_ux`, `datum_camera`) are auto-reset. User-intent fields (`enabled`, `datum_physical`, `datum_frame`) are preserved. Guard: no reset on first-time setup (empty → populated).
- **Calibration snapshots:** Auto-saved to `base_path/calibration/calibration.yaml` after every calibration completion (all 5 methods). Snapshot load preserves source-related config keys (`calibration_sources`, `image_format`, etc.).

### Frontend
- **Clearable number inputs:** Use `type="text" inputMode="numeric"` with a separate string state. `onChange` sets string, `onBlur` parses+clamps+saves. Never use `Number(e.target.value) || default` on onChange — it prevents clearing.
- **Prefetch cache key** MUST include all rendering-relevant settings (cmap, lower, upper, offsets, axis limits) — otherwise stale images served. `settingsVersionRef` counter in cache key invalidates on changes.
- **Transform cache invalidation:** After applying/clearing transforms, must manually increment `settingsVersionRef` and clear prefetch buffer BEFORE calling `handleRender()`.
- **TypeScript `??` vs `||`:** Cannot mix without parens. Use `?? (parseFloat(x) || 0)` not `?? parseFloat(x) || 0`.
- **Calibration hook mount race:** `useDotboardCalibration` and `useStereoCalibration` had auto-save effects that fired on mount with empty default state. Fix: `configLoadedRef` guard.
- **`vmin_pct`/`vmax_pct` convention:** Must be percentages (0-100) of data range, NOT raw pixel values.

### Backend Processing
- `gc.collect()` on Dask workers causes SIGSEGV (FFTW conflict) — only GC on client
- `interp2custom.c` uses column-major indexing (unlike all other C code)
- K-space fitting is BETA — experimental, may change
- **`coordinates.mat` race condition:** `transform_all_frames()` transforms coordinates once per camera then must pass `None` (not `coords_file`) to parallel workers. Workers re-reading and re-saving causes double-transform and Windows "Access is denied".
- **`invert_ux`/`invert_uy` UV_stress signs:** When negating only one velocity component, UV_stress must also be negated: `(-u')v' = -(u'v')`. But `invert_ux_uy` leaves UV_stress unchanged. Uses XOR logic.
- **Batch transform frame count:** Source frame was already processed but excluded from `total_frames_to_process`. Fix: add 1 and start `processed_frames` at 1.

### PIV Polling & Status Image
- Polling interval: 1000ms. Image refresh interval: 3000ms.
- Status image picks a random available frame each interval.
- Off-tab optimization: `PivJobProvider` receives `activeTab` prop; skips heavyweight image fetch when user is NOT on PIV/Ensemble tab.
- Young file filter: `get_uncalibrated_count()` skips files with `st_mtime < now - 5` seconds (avoids reading partially-written .mat files).
- Truncated .mat handling: `load_piv_result()` catches all `loadmat()` exceptions and re-raises as `FileNotFoundError`.

---

## Conclusions & Architectural Notes

1. **Config is king.** Nearly every function reads `get_config()`. Frontend mirrors this: all setup panels → `POST /backend/update_config`. Processing reads config properties that validate and coerce values.

2. **Two execution paths, shared core.** GUI (Flask) and CLI (`pivtools-cli`) both use `pivtools_core` for config/validation and `pivtools_cli` for computation. GUI spawns processing as subprocess; CLI runs directly.

3. **Dask as the computation layer.** Both instantaneous and ensemble use sliding window I/O with bounded memory. Lazy loading (~1 KB per pair), broadcast scattering for immutable data, worker pinning for locality.

4. **C extensions for performance.** Three shared libraries (cross-correlation, interpolation, Gaussian fitting) loaded via ctypes. FFTW for FFT, GSL for nonlinear least-squares, OpenMP for parallelism. Static-linked to bundled FFTW/GSL.

5. **Blueprint-per-feature** (GUI) is consistent. Each feature: `module/app/views.py` (routes) + business logic. Predictable for new features.

6. **Frontend hooks are the API contract.** Each `use*` hook encapsulates all fetch calls. Modify the hook to change frontend-backend communication.

7. **Data flows unidirectionally:** config.yaml → Config → Flask/CLI → processing → .mat files → plotting/statistics.

8. **Key gotchas:** See the "Gotchas & Common Pitfalls" section above for a comprehensive list.

9. **This file is the single source of documentation truth.** The `docs/` directory has been removed. All architectural, processing, and API documentation lives here.

10. **Stereo support** is threaded throughout: config detection, path conventions (`Stereo CamX_CamY`), and tool constraints (`is_stereo_setup`).
