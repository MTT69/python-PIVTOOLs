# PIV Benchmark Comparison

Compares PIV results against ground truth (e.g., DNS/JHTDB data) for validation.

## Metrics

- Mean velocity profile U+ vs y+
- Reynolds normal stresses uu+, vv+ vs y+
- Reynolds shear stress -uv+ vs y+

## Required Directory Structure

### Ground Truth Directory (`--gt-dir`)

```
ground_truth_dir/
    profiles.mat        # Contains profiles.win_1px struct with:
                        #   y_mm, y_plus, U, V, uu, vv, uv,
                        #   U_plus, uu_plus, vv_plus, uv_plus
    wall_units.mat      # Contains wall_units struct with:
                        #   u_tau, nu, delta_nu, h_mm, Re_tau
```

### Base Directory (`--base-dir`)

For **ensemble** mode:
```
base_dir/
    calibrated_piv/
        1000/
            Cam1/
                ensemble/
                    ensemble_result.mat   # Array of structs per pass with:
                                          #   ux, uy, UU_stress, VV_stress, UV_stress
                    coordinates.mat       # Array of structs per pass with:
                                          #   x, y (in mm)
```

For **instantaneous** mode:
```
base_dir/
    statistics/
        1000/
            Cam1/
                instantaneous/
                    mean_stats/
                        mean_stats.mat    # Contains piv_result and coordinates arrays
```

## Usage

```bash
python benchmark_comparison.py --gt-dir <ground_truth_path> --base-dir <base_path> [options]
```

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--gt-dir` | `-g` | Path to ground truth directory containing `profiles.mat` and `wall_units.mat` |
| `--base-dir` | `-b` | Base path to PIV results (or use `--ensemble-dir` for direct path) |

### Optional Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--mode` | `-m` | `instantaneous` | PIV mode: `instantaneous` or `ensemble` |
| `--runs` | `-r` | None | Comma-separated run indices (0-based), e.g., `0,2` |
| `--windows` | `-w` | None | Comma-separated window sizes for labels, e.g., `64,16` |
| `--labels` | `-l` | None | Custom output folder labels, e.g., `pass1,pass3` |
| `--ensemble-dir` | `-e` | None | Direct path to ensemble folder (overrides `--base-dir`) |
| `--y-plus-offset` | `-y` | `0.0` | Offset to add to y+ coordinates for alignment |

## Examples

### Single run (default settings)
```bash
python benchmark_comparison.py \
    -g /path/to/ground_truth \
    -b /path/to/piv_base \
    -m ensemble
```

### Multiple passes with window size labels
```bash
python benchmark_comparison.py \
    -m ensemble \
    -g /path/to/ground_truth \
    -b /path/to/piv_base \
    -r 0,2 \
    -w 64,16
```

### With y+ offset correction
```bash
python benchmark_comparison.py \
    -m ensemble \
    -g /path/to/ground_truth \
    -b /path/to/piv_base \
    -r 0,2 \
    -w 64,16 \
    -y 2.5
```

### Using direct ensemble directory path
```bash
python benchmark_comparison.py \
    -m ensemble \
    -g /path/to/ground_truth \
    -e /path/to/ensemble_folder \
    -r 0,2 \
    -w 64,16
```

## Output

Results are saved to:
- `validation/benchmark_results_ensemble/` (ensemble mode)
- `validation/benchmark_results/` (instantaneous mode)

Each window size gets its own subfolder containing:
- `U_plus_profile.png` - Mean velocity profile (semi-log)
- `V_plus_profile.png` - Wall-normal velocity profile
- `reynolds_stresses.png` - uu+, vv+, -uv+ profiles
- `U_plus_linear.png` - Mean velocity (linear scale)

Combined comparison plots are saved in the parent output directory.

## Benchmark Summary

The script prints a summary table with R² and RMS error metrics:

```
Window       U+ R²      U+ RMS%    uu+ R²     vv+ R²     -uv+ R²
--------------------------------------------------------------
64x64        0.9670     4.3        0.4160     0.7774     -0.9445
16x16        0.9874     2.1        0.8952     0.8317     0.2896
```
