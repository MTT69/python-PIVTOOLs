# Stereo Ensemble PIV — Correlation-of-Correlations Pipeline

## Overview

Standard stereo PIV extracts only **5 of 6 Reynolds stress components** — R_xx and R_zz are coupled (the system is rank-deficient). The Correlation-of-Correlations (CoC) method resolves this by cross-correlating per-frame correlation maps from the two cameras, providing the missing constraint:

```
Σ_12 = R_xx − sin²θ · R_zz
```

Combined with the standard coupled observable `A = R_xx + sin²θ · R_zz`, this allows full decoupling:

```
R_xx = (A + Σ_12) / 2
R_zz = (A − Σ_12) / (2 · sin²θ)
```

The pipeline runs as a CLI command and produces all 6 Reynolds stress components (UU, VV, WW, UV, UW, VW) plus 3D velocity fields (ux, uy, uz).

---

## Quick Start

### Prerequisites

1. A completed stereo calibration (dotboard or ChArUco) with `stereo_model.mat` present
2. Images from both cameras accessible via `config.yaml`
3. Standard ensemble PIV config parameters set (window sizes, overlaps, passes)

### Configuration

Add a `stereo_ensemble_piv` section to your `config.yaml`:

```yaml
stereo_ensemble_piv:
  camera_pair: [1, 2]                    # Which cameras to use
  dewarp_output_size: [512, 512]         # (H, W) of dewarped images
  world_bounds: null                     # [x_min, x_max, y_min, y_max] mm, null = auto-detect
  self_calibration:
    z_offset: 0.0                        # mm (from self-calibration)
    tilt_x: 0.0                          # radians
    tilt_y: 0.0                          # radians
  # All other keys fall back to ensemble_piv if null:
  window_size: null                      # e.g. [[128,128],[64,64]]
  overlap: null                          # e.g. [50, 50]
  type: null                             # e.g. ["std", "std"]
  sum_window: null                       # For single mode
```

Any key set to `null` inherits from the `ensemble_piv` section — you only need to override what differs.

### Running

```bash
# Basic usage
pivtools-cli stereo-ensemble

# With active paths override
pivtools-cli stereo-ensemble --active-paths "0,1"

# Or as a Python module
python -m pivtools_core.stereo_ensemble
```

### Output

Results are saved to:
```
base_path/uncalibrated_piv/{N}/Stereo Cam{A}_Cam{B}/stereo_ensemble/
  ├── stereo_ensemble_result.mat    # Velocities + stresses + diagnostics
  └── stereo_coordinates.mat        # World coordinates (mm)
```

---

## Pipeline Architecture

```
Raw images (cam1, cam2)
  │
  ▼ Dewarp to common world-XY plane
  │ (using stereo model + self-cal corrections)
  │
  ├──► Per-camera ensemble correlation (AA, BB, AB)    ← C library
  │    via bulkxcorr2d_accumulate (standard)
  │
  └──► Per-frame CoC cross-correlation                 ← C lib (N=1) + NumPy FFT
       of single-frame correlation maps
  │
  ▼ Accumulate across all frames
  │
  ├──► Per-camera Gaussian fitting                     ← C library (libmarquadt)
  │    → displacements + spread parameters
  │
  └──► CoC Gaussian fitting                            ← SciPy (6-param on 5×5)
       → cross-camera spread
  │
  ▼ 3D velocity reconstruction
  │  ux = (d1_x + d2_x) / 2
  │  uy = (d1_y + d2_y) / 2
  │  uz = (d1_x − d2_x) / (2·sinθ)
  │
  ▼ All 6 Reynolds stresses
  │  R_yy, R_xy, R_yz, R_xz from standard stereo equations
  │  R_xx, R_zz decoupled via CoC (the key innovation)
  │
  ▼ Outlier detection + infilling
  │
  ▼ Predictor → next pass (image deformation)
  │
  ▼ Save stereo_ensemble_result.mat + stereo_coordinates.mat
```

---

## File Map

| File | Purpose |
|------|---------|
| `pivtools_core/stereo_ensemble.py` | Dask orchestration — main pipeline, sliding window, multi-pass |
| `pivtools_cli/piv/piv_backend/cpu_stereo_ensemble.py` | CPU backend — dewarping, dual-camera correlation, per-frame CoC |
| `pivtools_cli/piv/piv_backend/stereo_ensemble_accumulator.py` | Accumulator — buffer management, finalize_pass with 3D velocity + 6 stresses |
| `pivtools_cli/piv/stereo_ensemble_result.py` | Result dataclasses — `PIVStereoEnsemblePassResult` |
| `pivtools_cli/piv/save_results.py` | Save functions — `save_stereo_ensemble_result()`, `save_stereo_ensemble_coordinates()` |
| `pivtools_cli/piv/piv_backend/factory.py` | Factory — `make_stereo_ensemble_correlator()` |
| `pivtools_cli/processing/dask_pipeline.py` | Dask worker functions — `correlate_stereo_batch_and_accumulate()`, `reduce_stereo_ensemble_results()` |
| `pivtools_cli/cli.py` | CLI command — `stereo-ensemble` |
| `pivtools_core/config.py` | Config properties — `stereo_ensemble_*` with fallback to `ensemble_piv` |

---

## How C Acceleration Is Used

The pipeline does **not** introduce any new C libraries. It reuses the existing three via composition:

### Per-Camera Correlation — `libbulkxcorr2d` (C + FFTW)

`StereoEnsembleCorrelatorCPU` wraps an internal `EnsembleCorrelatorCPU` instance. Both cameras' dewarped images are correlated through the standard C library path:

```python
# In cpu_stereo_ensemble.py
result_cam1 = self._correlator.correlate_batch_for_accumulation(dw_cam1, ...)
result_cam2 = self._correlator.correlate_batch_for_accumulation(dw_cam2, ...)
```

Each call internally runs `bulkxcorr2d_accumulate()` three times (AA, BB, AB) with full N frames — identical to standard ensemble.

### CoC Per-Frame Correlation — `libbulkxcorr2d` with N=1

For CoC, we need individual per-frame correlation planes (not the accumulated sum). The C library is called with N=1 for each frame:

```python
# In cpu_stereo_ensemble.py::_compute_coc_batch()
for n in range(N):
    temp_cam1_AB.fill(0)
    temp_cam2_AB.fill(0)
    # N=1 C library calls → single-frame correlation planes
    self._correlator.lib.bulkxcorr2d_accumulate(..., 1, ...)  # cam1
    self._correlator.lib.bulkxcorr2d_accumulate(..., 1, ...)  # cam2
    # Per-window FFT cross-correlate (NumPy)
    for w in range(total_windows):
        F1 = np.fft.fft2(R11_frame[w], s=(coc_h, coc_w))
        F2 = np.fft.fft2(R22_frame[w], s=(coc_h, coc_w))
        coc_sum[w] += np.fft.fftshift(np.real(np.fft.ifft2(F1 * np.conj(F2))))
```

The C library handles the expensive correlation; the per-window FFT cross-correlation is pure NumPy but operates on small (e.g., 127×127) planes.

### Per-Camera Peak Fitting — `libmarquadt` (C + GSL)

After all frames are accumulated, per-camera Gaussian fitting is dispatched to Dask workers via the standard `fit_windows_openmp()` path — the same 16-parameter stacked Gaussian fitter (C + GSL + OpenMP) used by standard ensemble PIV.

### CoC Spread Fitting — SciPy (Pure Python)

The CoC planes are fitted with a 6-parameter elliptical Gaussian via `scipy.optimize.least_squares` on a 5×5 region around the peak. This extracts the cross-camera covariance spread — fundamentally different from displacement peak fitting.

**Why no C fitter for CoC:** The CoC fit extracts *width* (variance) from a tiny region, not displacement. At ~1-5ms per window via SciPy, it's not a bottleneck (< 5% of total pipeline time). The standard C fitter is designed for the 16-parameter AA+BB+AB stacked problem and would need a completely different implementation.

### Image Warping — OpenCV (C-accelerated)

Dewarping (`cv2.remap`) and predictor warping both use OpenCV's C-accelerated implementation, parallelized via `ThreadPoolExecutor` with `cv2.setNumThreads(1)` to prevent contention — same pattern as standard ensemble.

### Summary Table

| Component | Acceleration | Library |
|-----------|-------------|---------|
| Per-camera correlation (AA, BB, AB) | C + FFTW + OpenMP | `libbulkxcorr2d` |
| CoC per-frame correlation | C (N=1) + NumPy FFT | `libbulkxcorr2d` + NumPy |
| Per-camera peak fitting | C + GSL + OpenMP | `libmarquadt` |
| CoC spread fitting | Python | SciPy `least_squares` |
| Dewarping | C (OpenCV) + threading | `cv2.remap` |
| Predictor warping | C (OpenCV) + threading | `cv2.remap` |

---

## Key Design Decisions

### Composition Over Inheritance

`StereoEnsembleCorrelatorCPU` does NOT inherit from `EnsembleCorrelatorCPU`. Instead it creates an internal instance:

```python
class StereoEnsembleCorrelatorCPU:
    def __init__(self, config, cam1, cam2, ...):
        self._correlator = EnsembleCorrelatorCPU(config, ...)
```

This avoids fragile coupling to the standard correlator's internals while reusing all its functionality (C library loading, window weights, predictor warping, etc.).

### Dual-Camera Sliding Window

The Dask orchestration layer submits paired camera chunks. Each chunk index triggers 2 filter futures (cam1 + cam2). Both must complete before the correlation task is submitted:

```python
# In stereo_ensemble.py::process_stereo_pass_sliding_window()
wait([cam1_future, cam2_future])
client.submit(correlate_stereo_batch_and_accumulate, ...)
```

Memory usage: ~4 batches per worker (2 cameras × 2 in flight), bounded by the sliding window.

### Worker-Side Correlator Reconstruction

On Dask workers, the `StereoEnsembleCorrelatorCPU` is recreated per batch from scattered setup data. Dewarp map recomputation (~20ms) is negligible versus correlation cost. The expensive parts (FFTW planning, window weights) come from the scattered `precomputed_cache`.

### Dask Retry Safety

All accumulation uses `+` not `+=` for Dask retry safety — if a worker fails and the task is retried, the result is correct because we never mutate shared state.

---

## Reynolds Stress Mathematics

### Standard 5 Observables

From per-camera displacement variance Σ_ii (sig_AB - sig_A):

```
R_yy = (Σ_11_yy + Σ_22_yy) / 2
R_xy = (Σ_11_xy + Σ_22_xy) / 2
R_yz = (Σ_11_xy − Σ_22_xy) / (2·sinθ)
R_xz = (Σ_11_xx − Σ_22_xx) / (4·sinθ)
A    = (Σ_11_xx + Σ_22_xx) / 2          ← COUPLED: A = R_xx + sin²θ·R_zz
```

### CoC: The Missing Equation

```
Σ_12_xx = (Σ_11_xx + Σ_22_xx − spread_C_xx) / 2   ← from CoC spread fitting
B       = Σ_12_xx                                    ← B = R_xx − sin²θ·R_zz
```

### Decoupling

```
R_xx = (A + B) / 2
R_zz = (A − B) / (2·sin²θ)
```

Validation: if R_zz < 0 (physically impossible for a variance), fall back to coupled `A` for that window.

---

## Output Format

### `stereo_ensemble_result.mat`

Multi-run structure (one run per pass), variable name `stereo_ensemble_result`:

| Field | Shape | Description |
|-------|-------|-------------|
| `ux_mat` | (H, W) | In-plane x velocity (dewarped pixels) |
| `uy_mat` | (H, W) | In-plane y velocity (dewarped pixels) |
| `uz_mat` | (H, W) | Out-of-plane z velocity (dewarped pixels) |
| `UU_stress` | (H, W) | R_xx — in-plane normal stress (x) |
| `VV_stress` | (H, W) | R_yy — in-plane normal stress (y) |
| `WW_stress` | (H, W) | R_zz — out-of-plane normal stress (from CoC) |
| `UV_stress` | (H, W) | R_xy — in-plane shear stress |
| `UW_stress` | (H, W) | R_xz — out-of-plane shear stress (x) |
| `VW_stress` | (H, W) | R_yz — out-of-plane shear stress (y) |
| `d1_x`, `d1_y` | (H, W) | Camera 1 dewarped displacements |
| `d2_x`, `d2_y` | (H, W) | Camera 2 dewarped displacements |
| `Sigma_12_xx` | (H, W) | Cross-camera covariance diagnostic |
| `peakheight` | (H, W) | Average normalized peak height (cam1 + cam2) |
| `nan_reason` | (H, W) | Failure codes (0=OK, 2-6=fitting, 10=outlier, -1=masked) |
| `b_mask` | (H, W) | Binary vector mask |
| `pred_x`, `pred_y` | (H, W) | Predictor displacement field |
| `window_size` | (2,) | Window height, width |
| `win_ctrs_x`, `win_ctrs_y` | (N,) | Window center coordinates |
| `stereo_angle` | scalar | Stereo half-angle (radians) |
| `mm_per_pixel` | scalar | Dewarped pixel size (mm/pixel) |

**Sign conventions on save:** `uy` negated, `UV_stress` negated, `VW_stress` negated, `pred_y` negated (same as standard ensemble).

### `stereo_coordinates.mat`

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | (n_win_y, n_win_x) | World x coordinates (mm) |
| `y` | (n_win_y, n_win_x) | World y coordinates (mm) |

---

## Config Properties

All properties follow a fallback pattern — check `stereo_ensemble_piv` first, fall back to `ensemble_piv`:

| Property | Config Key | Fallback | Description |
|----------|-----------|----------|-------------|
| `stereo_ensemble_camera_pair` | `stereo_ensemble_piv.camera_pair` | `[1, 2]` | Camera numbers |
| `stereo_ensemble_dewarp_output_size` | `stereo_ensemble_piv.dewarp_output_size` | `[512, 512]` | Dewarped image (H, W) |
| `stereo_ensemble_world_bounds` | `stereo_ensemble_piv.world_bounds` | `null` (auto) | World bounds in mm |
| `stereo_ensemble_self_cal_z` | `stereo_ensemble_piv.self_calibration.z_offset` | `0.0` | Self-cal Z offset (mm) |
| `stereo_ensemble_self_cal_tilt_x` | `stereo_ensemble_piv.self_calibration.tilt_x` | `0.0` | Self-cal tilt X (rad) |
| `stereo_ensemble_self_cal_tilt_y` | `stereo_ensemble_piv.self_calibration.tilt_y` | `0.0` | Self-cal tilt Y (rad) |
| `stereo_ensemble_num_passes` | `stereo_ensemble_piv.window_size` length | falls back to `ensemble_piv` | Number of passes |
| `stereo_ensemble_window_sizes` | `stereo_ensemble_piv.window_size` | `ensemble_window_sizes` | Per-pass window sizes |
| `stereo_ensemble_overlaps` | `stereo_ensemble_piv.overlap` | `ensemble_overlaps` | Per-pass overlap % |
| `stereo_ensemble_type` | `stereo_ensemble_piv.type` | `ensemble_type` | Per-pass type (`"std"` or `"single"`) |
| `stereo_ensemble_sum_window` | `stereo_ensemble_piv.sum_window` | `ensemble_sum_window` | Sum window for single mode |
| `stereo_ensemble_resume_from_pass` | `stereo_ensemble_piv.resume_from_pass` | `0` (no resume) | 1-based pass to resume from |
| `stereo_ensemble_store_planes` | `stereo_ensemble_piv.store_planes` | `ensemble_store_planes` | Save correlation planes |
| `stereo_ensemble_save_diagnostics` | `stereo_ensemble_piv.save_diagnostics` | `ensemble_save_diagnostics` | Save diagnostic images |

---

## Comparison with Standard Ensemble

| Feature | Standard Ensemble | Stereo Ensemble | Notes |
|---------|------------------|-----------------|-------|
| Per-camera correlation | C-accelerated | C-accelerated | Identical — delegates to same C lib |
| Peak fitting | C-accelerated (16-param) | C-accelerated (16-param) | Identical — per camera |
| CoC correlation | N/A | C (N=1) + NumPy FFT | New: per-frame cross-correlation |
| CoC spread fitting | N/A | SciPy (6-param, 5×5) | New: extracts cross-camera variance |
| Stress components | 3 (UU, VV, UV) | 6 (UU, VV, WW, UV, UW, VW) | WW from CoC, UW/VW from stereo |
| Velocity components | 2 (ux, uy) | 3 (ux, uy, uz) | uz from displacement difference |
| Dewarping | N/A | cv2.remap (threaded) | New: maps raw images to world plane |
| Outlier detection | Yes | Yes | Same algorithms |
| Infilling | 3 fields (ux, uy, stresses) | 9 fields (ux, uy, uz, 6 stresses) | Extended for 3D |
| Multi-pass predictor | Yes | Yes (in-plane only) | uz not in predictor |
| Vector masking | Yes | Yes | Per-pass masks |
| Single mode | Yes | Yes (via config fallback) | Same code path |
| Background subtraction | Yes | Yes (per camera) | Correlation or image method |
| Resume from pass | Yes | Yes | `stereo_ensemble_piv.resume_from_pass` |
| Diagnostics saving | Yes | Yes | `store_planes` + `save_diagnostics` |
| GUI integration | Via piv_runner | **CLI only** | Advanced feature |

---

## Resume from Pass

Set `stereo_ensemble_piv.resume_from_pass` to skip already-completed passes and resume from a later pass. This avoids recomputing expensive multi-pass runs when only later passes need changes.

```yaml
stereo_ensemble_piv:
  resume_from_pass: 3    # Skip passes 1-2, resume from pass 3
```

**How it works:**
1. Loads `stereo_ensemble_result.mat` from the output directory
2. Restores passes 1 through (N-1) into the accumulator
3. Extracts the predictor field from the last loaded pass
4. Continues processing from pass N onwards
5. Backs up the existing result file before overwriting

**Requirements:**
- `resume_from_pass` must be 2 or higher (pass 1 has no prior state to load)
- The existing `stereo_ensemble_result.mat` must contain at least `resume_from_pass - 1` passes
- Set to `0` (default) for a fresh start

---

## Diagnostics Saving

Two diagnostic modes are available for debugging correlation quality:

### Correlation Planes (`store_planes`)

Saves per-camera correlation planes and CoC planes for each pass:

```yaml
stereo_ensemble_piv:
  store_planes: true
```

**Output:** `planes_pass_{N}.mat` in the output directory containing:
- `cam1_AA`, `cam1_BB`, `cam1_AB` — Camera 1 averaged correlation planes (4D: n_win_y, n_win_x, corr_h, corr_w)
- `cam2_AA`, `cam2_BB`, `cam2_AB` — Camera 2 averaged correlation planes
- `coc_sum` — Averaged CoC cross-correlation planes (4D: n_win_y, n_win_x, coc_h, coc_w)

### Warped Images (`save_diagnostics`)

Saves first-pair dewarped images for both cameras:

```yaml
stereo_ensemble_piv:
  save_diagnostics: true
```

**Output:** `warped_pass_{N}.mat` containing `cam1_A_warped`, `cam1_B_warped`, `cam2_A_warped`, `cam2_B_warped` — dewarped (and predictor-warped for pass > 0) images from the first frame pair.

Both config keys fall back to their `ensemble_piv` equivalents if not explicitly set.

---

## Limitations and Known Gaps

### Not Yet Implemented

1. **Stereo-specific config validation** — `validate_config()` validates ensemble parameters but does not check stereo-specific constraints (camera pair exists, stereo model file present, dewarp size reasonable). These are validated at runtime — errors appear later rather than at startup.

### By Design

2. **CLI only** — Stereo ensemble PIV is not exposed through the GUI. This is appropriate for an advanced feature. Results can be viewed in VectorViewer after processing.

3. **No calibration step** — Output is in dewarped pixel units. A separate calibration step (dividing by dt) is needed to convert to m/s. This matches the standard pipeline pattern.

4. **In-plane predictor only** — The multi-pass predictor uses only ux/uy (not uz). Out-of-plane displacements are typically small enough that this is acceptable.

---

## Troubleshooting

### Common Errors

**`FileNotFoundError: No stereo calibration model found`**
- Ensure stereo calibration has been run and `stereo_model.mat` exists at:
  `base_path/calibration/stereo_cam{A}_cam{B}/model/stereo_model.mat`

**`Could not auto-compute world bounds`**
- Camera models may have degenerate geometry. Set explicit `world_bounds` in config.

**`R_zz < 0` warnings**
- Some windows have noisy CoC extraction. The pipeline falls back to the coupled `A` value for those windows. A small number of warnings is normal; large numbers suggest insufficient frames or poor image quality.

**Memory errors**
- Stereo ensemble uses roughly 2× the memory of standard ensemble (two cameras). Reduce `batch_size` or increase `dask_memory_limit`.

### Performance Tips

- **Batch size**: Larger batches amortize CoC overhead. Use 30+ frames per batch.
- **Dewarp size**: Smaller `dewarp_output_size` reduces per-window count but may lose resolution. Start with the standard image size and adjust.
- **Workers**: Multiple Dask workers parallelize the dual-camera correlation. Use `dask_workers_per_node >= 2`.
- **OMP threads**: Set `omp_threads` to match physical cores per worker (not hyperthreads).
