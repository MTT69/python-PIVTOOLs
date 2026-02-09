# PIV Benchmark Validation

Compares PIV results against DNS/JHTDB ground truth for turbulent channel flow validation.

## Scripts

| Script | Purpose | Components |
|--------|---------|------------|
| `benchmark_comparison.py` | Planar PIV (2C): ensemble and instantaneous modes | U+, V+, uu+, vv+, -uv+ |
| `stereo_benchmark_comparison.py` | Stereo PIV (3C): instantaneous statistics only | U+, V+, W+, uu+, vv+, ww+, uv+, uw+, vw+ |
| `compare_ground_truths.py` | Compares different ground truth datasets against each other |

## Metrics

- Mean velocity profiles: U+ (and V+, W+ for stereo) vs y+ (log scale)
- Reynolds normal stresses: uu+, vv+ (and ww+ for stereo) vs y+
- Reynolds shear stress: -uv+ (and uw+, vw+ for stereo) vs y+
- Combined stresses plot: uu+, vv+, -uv+ on a single axis
- R² and RMS% error computed over y+ range [10, 500]

All y+ axes use **log scale** with range [1, Re_tau].

---

## Ground Truth Data Format

Two formats are supported. The scripts auto-detect which is present.

### Format A: MATLAB v5 (scipy-readable)

```
ground_truth_dir/
    wall_units.mat      # Contains wall_units struct:
                        #   u_tau, nu, delta_nu, h_mm, Re_tau
    profiles.mat        # Contains profiles.win_1px struct:
                        #   y_mm, y_plus, U, V, uu, vv, uv,
                        #   U_plus, uu_plus, vv_plus, uv_plus
                        #   (stereo also: W, ww, uw, vw, ww_plus)
```

### Format B: MATLAB v7.3 / HDF5 (h5py-readable)

```
ground_truth_dir/
    diagnostics.mat             # HDF5 - contains diagnostics group:
                                #   u_tau, nu, delta_nu, h_mm, Re_tau
    ensemble_statistics_full.mat  # HDF5 - contains:
                                #   ref_profile: DNS velocity (U, V, W, y_mm) - 2049 points
                                #   ensemble_stats: pre-averaged profiles per window size
                                #     y_plus, uu_plus, vv_plus, uv_plus (object refs)
                                #     (stereo also: ww_profile, uw_profile, vw_profile, etc.)
```

**Auto-detection logic:** Scripts check for `wall_units.mat` first, fall back to `diagnostics.mat`. Same for `profiles.mat` → `ensemble_statistics_full.mat`.

---

## PIV Data Directory Structure

### Ensemble mode (`-m ensemble`)

```
base_dir/
    calibrated_piv/
        {num_frames}/           # e.g. 1000 or 4000
            Cam1/
                ensemble/
                    ensemble_result.mat   # Array of structs per pass:
                                          #   ux, uy (m/s), UU_stress, VV_stress, UV_stress ((m/s)²)
                    coordinates.mat       # Array of structs per pass:
                                          #   x, y (mm, already calibrated)
```

Runs are 0-indexed passes. Typical: run 0 = 64x64, run 1 = 16x16, run 2 = 8x8.

### Instantaneous mode (`-m instantaneous`)

Uses pre-computed statistics (from the statistics module), NOT raw per-frame vectors:

```
base_dir/
    statistics/
        {num_frames}/           # e.g. 1000 or 4000
            Cam1/
                instantaneous/
                    mean_stats/
                        mean_stats.mat    # Contains piv_result and coordinates arrays
                                          #   ux, uy (m/s), uu, vv, uv ((m/s)²)
```

### Stereo mode (separate script)

```
stereo_base/
    statistics/{num_frames}/stereo/Cam1_Cam2/instantaneous/mean_stats/mean_stats.mat
    stereo_calibrated/{num_frames}/Cam1_Cam2/instantaneous/coordinates.mat
```

**Important:** Stereo stats are at `statistics/.../stereo/Cam1_Cam2/...` but coordinates come from `stereo_calibrated/.../Cam1_Cam2/...` (separate file).

---

## Usage: `benchmark_comparison.py`

```bash
python benchmark_comparison.py --gt-dir <path> --base-dir <path> [options]
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--gt-dir` | `-g` | *required* | Ground truth directory (containing `profiles.mat`/`diagnostics.mat`) |
| `--base-dir` | `-b` | None | Base directory containing PIV results |
| `--mode` | `-m` | `instantaneous` | `instantaneous` or `ensemble` |
| `--runs` | `-r` | None | Comma-separated run indices (0-based), e.g. `0,1,2` |
| `--windows` | `-w` | None | Comma-separated window sizes for labels, e.g. `64,16,8` |
| `--labels` | `-l` | None | Custom output folder labels |
| `--ensemble-dir` | `-e` | None | Direct path to ensemble folder (overrides `--base-dir`) |
| `--y-plus-offset` | `-y` | `0.0` | Offset to add to y+ coordinates |
| `--num-frames` | `-n` | `1000` | Frame count subdirectory in paths (e.g. `1000` or `4000`) |
| `--output-dir` | `-o` | None | Custom output directory (default: auto-named in `validation/`) |

**Note:** When both `--runs` and `--windows` are provided, the multi-run path is used (generates per-window + combined comparison plots). Otherwise single-run mode is used.

### Examples

#### Ensemble, 1000 images, 3 passes
```bash
python benchmark_comparison.py \
    -m ensemble \
    -g /path/to/ensemble_statistics_first_1000 \
    -b /path/to/validation_first_1000 \
    -r 0,1,2 -w 64,16,8 -y 1.0
```

#### Instantaneous, 1000 images, pass 3 (16x16)
```bash
python benchmark_comparison.py \
    -m instantaneous \
    -g /path/to/ensemble_statistics_first_1000 \
    -b /path/to/validation_first_1000 \
    -r 2 -w 16 -y 1.0
```

#### 4000-image case with custom output directory
```bash
python benchmark_comparison.py \
    -m ensemble \
    -g /path/to/ensemble_statistics_4000 \
    -b /path/to/validation_4000 \
    -r 0,1,2 -w 64,16,8 -y 1.0 \
    -n 4000 -o validation/benchmark_results_ensemble_4000
```

#### Direct ensemble directory (bypasses path construction)
```bash
python benchmark_comparison.py \
    -m ensemble \
    -g /path/to/ground_truth \
    -e /path/to/ensemble_folder \
    -r 0,2 -w 64,16
```

## Usage: `stereo_benchmark_comparison.py`

```bash
python stereo_benchmark_comparison.py [--run RUN_IDX] [--x-min X_MIN] [--x-max X_MAX]
```

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--run` | `-r` | `2` | Run index (0-based) |
| `--x-min` | | `5.0` | Minimum x to include (mm) |
| `--x-max` | | `145.0` | Maximum x to include (mm) |

**Note:** The stereo script has hardcoded data paths pointing to the `4000_images_channel` processing directory. Edit `main()` to change data locations.

---

## Output

### Default output directories

| Mode | Default output |
|------|---------------|
| Planar ensemble | `validation/benchmark_results_ensemble/` |
| Planar instantaneous | `validation/benchmark_results/` |
| Stereo | `validation/benchmark_results_stereo/` |
| Custom (`-o`) | Whatever path you specify |

### Per-window subfolder contents

| File | Description |
|------|-------------|
| `U_plus_profile.png` | Mean streamwise velocity U+ vs y+ (log scale) |
| `V_plus_profile.png` | Wall-normal velocity V+ vs y+ (log scale) |
| `U_plus_linear.png` | Mean velocity U+ vs y+ (linear scale, for near-wall detail) |
| `reynolds_stresses.png` | Individual uu+, vv+, -uv+ panels |
| `combined_stresses.png` | All stresses on one axis |
| `trace_invariant.png` | Diagnostic: checks rotation consistency |

### Multi-run combined plots (in parent directory)

| File | Description |
|------|-------------|
| `U_plus_profile_combined.png` | All window sizes overlaid |
| `V_plus_profile_combined.png` | All window sizes overlaid |
| `U_plus_linear_combined.png` | All window sizes overlaid |
| `reynolds_stresses_combined.png` | All window sizes overlaid |
| `trace_invariant_combined.png` | All window sizes overlaid |

### Stereo-specific plots

| File | Description |
|------|-------------|
| `W_plus_profile.png` | Spanwise velocity W+ |
| `velocities_combined.png` | V+ and W+ panels |
| `normal_stresses.png` | uu+, vv+, ww+ panels |
| `shear_stresses.png` | uv+, uw+, vw+ panels |
| `combined_stresses.png` | All stresses on one axis |

---

## Unit Conventions

This is the most common source of errors. Get this right or everything breaks.

| Quantity | PIVTOOLs stores as | Ground truth (DNS) | Conversion |
|----------|-------------------|-------------------|------------|
| Velocity (ux, uy, uz) | **m/s** | mm/s | `× 1000` |
| Stresses (uu, vv, uv, ...) | **(m/s)²** | (mm/s)² | `× 1e6` |
| Coordinates (x, y) | **mm** (calibrated) | mm | No conversion |
| Wall units (u_tau, nu, delta_nu) | — | **mm/s**, mm²/s, mm | — |

**Plus-unit normalisation:**
- `U+ = U_mm / u_tau`
- `uu+ = uu_mm² / u_tau²`
- `y+ = y_mm / delta_nu`

Typical wall units for Re_tau ≈ 1000 channel:
- `u_tau ≈ 8.42 mm/s`
- `delta_nu ≈ 0.15 mm`
- `Re_tau ≈ 998`

---

## Processing Pipeline (What Happens Inside)

1. **Load wall units** from ground truth directory (`wall_units.mat` or `diagnostics.mat`)
2. **Load ground truth profiles** from `profiles.mat` or `ensemble_statistics_full.mat`
3. **Load PIV data**: ensemble_result.mat (ensemble) or mean_stats.mat (instantaneous/stereo)
4. **X-averaging**: Average profiles across x-direction, excluding edge vectors (`x_exclude_vectors=4` for planar, `x_min`/`x_max` range for stereo)
5. **Unit conversion**: m/s → mm/s, (m/s)² → (mm/s)²
6. **Wall-unit normalisation**: Divide by u_tau (velocities) or u_tau² (stresses)
7. **y+ offset**: Optional shift applied (typically `+1.0` for this dataset)
8. **Interpolate ground truth** onto PIV y+ grid for error computation
9. **Compute R² and RMS%** over y+ ∈ [10, 500]
10. **Plot** all comparisons with log-scale y+ axis

---

## HDF5 Ground Truth: How Object References Work

The `ensemble_statistics_full.mat` (v7.3) stores arrays as HDF5 object references. Each field in `ensemble_stats` (e.g. `y_plus`, `uu_plus`) is an array of references, one per window size.

```python
import h5py
with h5py.File('ensemble_statistics_full.mat', 'r') as f:
    es = f['ensemble_stats']
    # Get y_plus for first window size (index 0)
    refs = np.array(es['y_plus']).flatten()
    y_plus = np.array(f[refs[0]]).flatten()  # Dereference
```

**Key detail:** `ref_profile` contains DNS velocities (U, V, W) at 2049 points but typically has NO stress data. The stresses come from `ensemble_stats` pre-averaged profiles at 255 points (for 16x16 window). The planar script interpolates DNS velocities onto the ensemble y+ grid so all arrays have consistent length.

Window size indices in `ensemble_stats`: typically `[16, 8, 6, 4]` — index 0 is 16x16.

---

## Gotchas and Lessons Learned

### Unit mismatch is the #1 failure mode
If U+ R² is wildly negative (e.g. -600000), the unit conversion is wrong. Check that PIV data is truly in m/s. Print raw ux range — should be ~0.05–0.19 m/s for this channel flow. If it's 5–19, data may be in cm/s or miscalibrated.

### HDF5 array length mismatch
`ref_profile` has 2049 DNS points. `ensemble_stats` profiles have 255 points. Never mix them in the same array without interpolation. The planar script uses ensemble_stats y_plus as the common grid and interpolates DNS velocities onto it.

### The `num_frames` subdirectory
PIV output paths contain a subdirectory matching the number of frames processed: `calibrated_piv/1000/Cam1/...` vs `calibrated_piv/4000/Cam1/...`. Use `-n` to set this. If you get "file not found", check this matches your actual directory structure.

### y+ offset
The PIV coordinate origin may not align with the wall. A `y_plus_offset` of `+1.0` (`-y 1.0`) works well for the current channel dataset. This shifts y+ values after coordinate conversion.

### Instantaneous mode uses pre-computed statistics
The instantaneous benchmark does NOT process raw per-frame .mat files. It reads from `statistics/{N}/Cam1/instantaneous/mean_stats/mean_stats.mat`, which must be computed first via the statistics module in the GUI or CLI.

### Run indices are 0-based
Pass 1 (largest window) = run 0, Pass 2 = run 1, etc. The `--runs` argument takes 0-based indices. Not all runs may be valid — check with `find_valid_runs()` from `vector_loading.py` or just try and look at the error.

### Stereo coordinates come from a different file
Stereo `mean_stats.mat` does NOT contain coordinates. They must be loaded from `stereo_calibrated/{N}/Cam1_Cam2/instantaneous/coordinates.mat` separately.

### Stereo script has hardcoded paths
Unlike the planar script which takes all paths as arguments, `stereo_benchmark_comparison.py` has a `data_root` hardcoded in `main()`. Edit this if your data is elsewhere.

### scipy vs h5py
`scipy.io.loadmat()` only reads MATLAB v5 format. MATLAB v7.3 files are HDF5 and raise `NotImplementedError`. Both scripts catch this and fall back to `h5py`. If you see this error, it's handled — not a bug.

### Ensemble stresses vs instantaneous stresses
- **Ensemble** `ensemble_result.mat`: stresses are `UU_stress`, `VV_stress`, `UV_stress`
- **Instantaneous** `mean_stats.mat`: stresses are `uu`, `vv`, `uv`
- Both are in (m/s)² and need `× 1e6` to convert to (mm/s)²

---

## Benchmark Results Summary

### Planar — 1000 images

**Ensemble:**

| Window | U+ R² | U+ RMS% | uu+ R² | vv+ R² | -uv+ R² |
|--------|-------|---------|--------|--------|---------|
| 64x64 | 0.9670 | 4.3 | 0.4160 | 0.7774 | -0.9445 |
| 16x16 | 0.9874 | 2.1 | 0.8952 | 0.8317 | 0.2896 |
| 8x8 | 0.9863 | 2.2 | 0.8656 | 0.8586 | 0.2975 |

**Instantaneous (16x16):**

| Window | U+ R² | U+ RMS% | uu+ R² | vv+ R² | -uv+ R² |
|--------|-------|---------|--------|--------|---------|
| 16x16 | 0.9954 | 1.3 | 0.9832 | 0.8762 | 0.8481 |

### Planar — 4000 images

**Ensemble:**

| Window | U+ R² | U+ RMS% | uu+ R² | vv+ R² | -uv+ R² |
|--------|-------|---------|--------|--------|---------|
| 64x64 | 0.9559 | 5.0 | 0.8068 | 0.7697 | 0.6610 |
| 16x16 | 0.9880 | 2.3 | 0.9366 | 0.8906 | 0.6991 |
| 8x8 | 0.9874 | 2.2 | 0.9045 | 0.9079 | 0.6976 |

**Instantaneous (16x16):**

| Window | U+ R² | U+ RMS% | uu+ R² | vv+ R² | -uv+ R² |
|--------|-------|---------|--------|--------|---------|
| 16x16 | 0.9980 | 0.9 | 0.9938 | 0.9408 | 0.9312 |
