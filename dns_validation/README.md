# DNS Validation Benchmark Suite

Validation tools for benchmarking PIV algorithms against DNS ground truth from the Johns Hopkins Turbulence Database (JHTDB) channel flow at Re_tau = 1000.

This package accompanies the PIVTOOLs paper and provides everything needed to reproduce the validation figures: ground truth profiles, synthetic image generator configurations, calibration targets, benchmark scripts, and pre-computed results with CSV data.

## Contents

```
dns_validation/
├── benchmark_comparison.py        # Per-method benchmark (planar + ensemble)
├── stereo_benchmark_comparison.py # Stereo 3-component benchmark
├── cross_method_comparison.py     # Multi-method comparison figures
├── export_figure_csvs.py          # CSV export for figure reproducibility
├── tcf_direct_stats.py            # Ground truth computation from particle data
├── Euro_sig_configs/              # EUROSIG synthetic image generator configs
│   ├── sigconf_planar.cdl         #   Planar, clean (85k particles)
│   ├── sigconf_planar_noisy_A.cdl #   Planar, noisy frame A (22k + noise)
│   ├── sigconf_planar_noisy_B.cdl #   Planar, noisy frame B (22k + noise)
│   ├── SIGconf_Stereo_cam1.cdl    #   Stereo cam1, clean
│   ├── SIGconf_Stereo_cam2.cdl    #   Stereo cam2, clean
│   ├── SIGconf_Stereo_cam1_noisy_A.cdl
│   ├── SIGconf_Stereo_cam1_noisy_B.cdl
│   ├── SIGconf_Stereo_cam2_noisy_A.cdl
│   └── SIGconf_Stereo_cam2_noisy_B.cdl
├── planar_calibration_images/     # Dotboard calibration targets (20 TIFF)
├── Stereo_calibration_images/     # Stereo calibration targets (20 per camera)
│   ├── cam1/
│   └── cam2/
├── cross_method_fine/             # Pre-computed results: clean comparison
│   ├── direct_stats.mat           #   Ground truth profiles + CI
│   ├── mean_velocity_comparison.png
│   ├── mean_velocity_comparison.csv
│   ├── combined_stresses_comparison.png
│   ├── combined_stresses_comparison.csv
│   └── stresses_comparison.png
└── noisy_cross_method_coarse/     # Pre-computed results: noisy comparison
    ├── direct_stats.mat
    ├── mean_velocity_comparison.png
    ├── mean_velocity_comparison.csv
    ├── combined_stresses_comparison.png
    ├── combined_stresses_comparison.csv
    └── stresses_comparison.png
```

**Synthetic particle images** are provided separately (see [Synthetic Images](#synthetic-images)).



No PIVTOOLs installation is required for plotting from CSV data. The benchmark scripts import from `dns_validation.benchmark_comparison` and require PIVTOOLs output `.mat` files.

## Reproduce Paper Figures

The CSV files in `cross_method_fine/` and `noisy_cross_method_coarse/` contain the exact data plotted in the paper figures. Each CSV has columns for the DNS reference (with 95% CI bounds for stresses) and each PIV method, with separate y+ columns per method to preserve native grid resolution.

To regenerate the figures from CSV:

```python
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('cross_method_fine/combined_stresses_comparison.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract DNS reference
dns_yp = np.array([float(r['DNS_y_plus']) for r in rows if r['DNS_y_plus']])
dns_uu = np.array([float(r['DNS_uu_plus']) for r in rows if r['DNS_uu_plus']])

# Extract a PIV method
ens_yp = np.array([float(r['Ensemble_8x16_y_plus']) for r in rows if r['Ensemble_8x16_y_plus']])
ens_uu = np.array([float(r['Ensemble_8x16_uu_plus']) for r in rows if r['Ensemble_8x16_uu_plus']])

plt.semilogx(dns_yp, dns_uu, 'k-', label='DNS')
plt.semilogx(ens_yp, ens_uu, 'o', markersize=3, label='Ensemble (8x16)')
plt.xlabel(r'$y^+$')
plt.ylabel(r"$\overline{u'u'}^+$")
plt.legend()
plt.show()
```

## End-to-End Reproduction

To reproduce the full pipeline from raw images through to benchmark figures:

### 1. Obtain the synthetic images

Synthetic particle images are provided in the following structure:

```
images/
├── planar/
│   ├── clean/       # Case A: 85k particles, no noise
│   │   └── Cam1/    # B00001_A.tif, B00001_B.tif, ... (4000 pairs)
│   └── noisy/       # Case B: 22k particles, SNR ~8
│       └── Cam1/
├── stereo/
│   ├── clean/
│   │   ├── Cam1/
│   │   └── Cam2/
│   └── noisy/
│       ├── Cam1/
│       └── Cam2/
└── calibration/     # (also provided in dns_validation/)
    ├── planar/
    └── stereo/
```

###  Calibrate

Use the provided calibration images to generate camera models

###  Run PIV

Process the synthetic images with PIVTOOLs

Configure multi-pass window sizes in `config.yaml`. The paper uses:
- Instantaneous/stereo: `[128, 64, 32, 16]` with 50% overlap
- Ensemble: `[64, 32, 16, [8,16], [8,16]]` with 50% overlap

###  Compute statistics

For instantaneous and stereo results, compute time-averaged statistics:

This produces `mean_stats.mat` containing mean velocities and Reynolds stresses.

###  Run benchmarks

Compare against DNS ground truth using the provided `direct_stats.mat`:

```bash
# Instantaneous benchmark
python -m dns_validation.benchmark_comparison \
    --mode instantaneous \
    --runs 2,3 --windows 32,16 \
    --gt-dir dns_validation/cross_method_fine \
    --base-dir path/to/piv_output \
    --num-frames 4000 \
    --output-dir path/to/results

# Cross-method comparison
python -m dns_validation.cross_method_comparison \
    --gt-dir dns_validation/cross_method_fine \
    --output-dir path/to/results \
    --inst-stats path/to/instantaneous/mean_stats.mat \
    --ens-dir path/to/ensemble_result_dir \
    --stereo-stats path/to/stereo/mean_stats.mat
```

## Scripts

### 1. `benchmark_comparison.py` — Single-Method Benchmark

Compares instantaneous or ensemble PIV output against DNS ground truth. Generates mean velocity, Reynolds stress, residual, trace invariant, and noise decomposition plots.

**CLI:**

```bash
# Instantaneous, two window sizes
python -m dns_validation.benchmark_comparison \
    --mode instantaneous \
    --runs 2,3 --windows 32,16 \
    --gt-dir path/to/ground_truth \
    --base-dir path/to/piv_results \
    --num-frames 4000 \
    --output-dir path/to/output

# Ensemble, direct path
python -m dns_validation.benchmark_comparison \
    --mode ensemble \
    --runs 3,4 --windows 8,8 \
    --labels 8x16_pass4,8x16_pass5 \
    --gt-dir path/to/ground_truth \
    --ensemble-dir path/to/ensemble_result_dir \
    --num-frames 4000 \
    --output-dir path/to/output
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--mode` / `-m` | `instantaneous` or `ensemble` |
| `--runs` / `-r` | Comma-separated 0-based run indices |
| `--windows` / `-w` | Comma-separated window sizes for labels |
| `--labels` / `-l` | Custom output folder names |
| `--gt-dir` / `-g` | Ground truth directory (required) |
| `--base-dir` / `-b` | PIV results base directory (instantaneous) |
| `--ensemble-dir` / `-e` | Direct path to `ensemble_result.mat` directory |
| `--num-frames` / `-n` | Frame count subdirectory (default: 1000) |
| `--output-dir` / `-o` | Output directory |
| `--y-plus-offset` / `-y` | Additional y+ offset (default: 0.0, added on top of hardcoded +1.0) |
| `--show-fit-lines` | Show log-law and viscous sublayer lines (default: off) |

**Run index mapping:** 0-based. For a 4-pass config `[128, 64, 32, 16]`: index 2 = 32x32, index 3 = 16x16. Early passes may have empty data in the statistics file.

**Path conventions:**
- Instantaneous: `base_dir/statistics/{num_frames}/Cam1/instantaneous/mean_stats/mean_stats.mat`
- Ensemble: `ensemble_dir/ensemble_result.mat` + `ensemble_dir/coordinates.mat`
- Stereo: call loading functions directly with NaN-aware profile extraction (stereo grids have NaN borders from camera overlap)

**Plots generated per run:** `U_plus_profile.png`, `reynolds_stresses.png`, `V_plus_profile.png`, `U_plus_linear.png`, `U_plus_profile_smooth.png`, `reynolds_stresses_smooth.png`, `trace_invariant.png`, `residuals.png`, `noise_gradient_decomposition.png`, `combined_stresses.png`

### 2. `stereo_benchmark_comparison.py` — Stereo Benchmark

Compares stereo PIV (3-component: U, V, W) and all 6 Reynolds stress tensor components against DNS. Requires a LaTeX installation (uses `text.usetex: True`).

### 3. `cross_method_comparison.py` — Multi-Method Comparison

Publication-quality plots comparing one pass from each of instantaneous, ensemble, and stereo against DNS on the same axes. Uses the Okabe-Ito colourblind-safe palette.

**Python API:**

```python
from dns_validation.cross_method_comparison import compare_methods

compare_methods(
    gt_dir='path/to/ground_truth',
    output_dir='path/to/output',
    # Instantaneous
    inst_stats_path='path/to/mean_stats.mat',
    inst_run_idx=3,            # 0-based
    inst_y_offset=3.0,         # additional y+ offset
    inst_window_label='16x16', # appears in legend
    # Ensemble
    ens_ensemble_path='path/to/ensemble_result.mat',
    ens_coords_path='path/to/coordinates.mat',
    ens_run_idx=3,
    ens_y_offset=0.8,
    ens_window_label='8x16',
    # Stereo
    stereo_stats_path='path/to/stereo/mean_stats.mat',
    stereo_run_idx=3,
    stereo_y_offset=0.8,
    stereo_window_label='16x16',
    stereo_trim_top=10,        # trim N high-y points (NaN border)
    # Options
    trim_near_wall=1,          # trim N near-wall points from inst/stereo
    title_suffix='SNR 8',      # appended to plot titles
)
```

**Outputs:** `mean_velocity_comparison.png`, `stresses_comparison.png` (1x3 subplots), `combined_stresses_comparison.png` (all stresses on one axis with two-legend layout).



**CSV format (stresses):**

| Column | Description |
|--------|-------------|
| `DNS_y_plus` | DNS wall-normal coordinate |
| `DNS_uu_plus` | DNS streamwise normal stress |
| `DNS_vv_plus` | DNS wall-normal normal stress |
| `DNS_neg_uv_plus` | DNS Reynolds shear stress (negated) |
| `DNS_uu_ci_lo/hi` | 95% confidence interval bounds |
| `{Method}_y_plus` | PIV y+ (on its own grid) |
| `{Method}_uu_plus` | PIV stress values |

### 5. `tcf_direct_stats.py` — Ground Truth Computation

Computes turbulence statistics directly from JHTDB particle position pairs. Reads `B#####_A.data` / `B#####_B.data` files and produces `direct_stats.mat` containing:

- 1D profiles on cosine-stretched (Chebyshev) y-bins: mean velocity, Reynolds stress tensor, all in wall units
- 2D fields on a uniform grid: mean velocities and all 6 stress components
- 95% confidence intervals via 2000-iteration frame-level bootstrap
- Wall unit parameters: u_tau, delta_nu, Re_tau

**Configuration** is at the top of the script (data paths, bin count, grid resolution). Requires particle position files from JHTDB. Pre-computed ground truth is provided in `cross_method_fine/direct_stats.mat` and `noisy_cross_method_coarse/direct_stats.mat`.

## Synthetic Images

Synthetic particle images are hosted separately due to their size. The `Euro_sig_configs/` directory contains the EUROSIG configuration files (Lecordier & Westerweel, EUROPIV II) used to generate them.

**Case A (ideal):** 85,000 particles/image (~5.2 ppw at 16x16), no noise.
- `sigconf_planar.cdl` — planar
- `SIGconf_Stereo_cam{1,2}.cdl` — stereo at +/-45 deg

**Case B (degraded):** 22,000 particles/image (~1.3 ppw at 16x16), Gaussian noise (mean=80, std=16, SNR~8).
- `sigconf_planar_noisy_{A,B}.cdl` — planar with independent noise per frame
- `SIGconf_Stereo_cam{1,2}_noisy_{A,B}.cdl` — stereo with noise

Both cases use 2048x2048 px images, 3 px particle diameter, 16 px (1.2 mm) laser sheet thickness, and 4000 image pairs from JHTDB Re_tau=1000 channel flow.

## Calibration Images

Synthetic dotboard calibration targets are provided for both planar and stereo configurations:

- `planar_calibration_images/` — 20 TIFF images for planar pinhole calibration
- `Stereo_calibration_images/cam1/` and `cam2/` — 20 TIFF images per camera for stereo calibration

These are used with PIVTOOLs' `detect-planar` / `detect-stereo-planar` CLI commands to produce calibration models before processing.

## Ground Truth Data Format

The `direct_stats.mat` files contain:

```
y_plus          (N,)      wall-normal coordinate in wall units
U_plus          (N, 3)    mean velocity [U, V, W] in wall units
stress_plus     (N, 3, 3) Reynolds stress tensor in wall units
stress_ci_lo    (N, 3, 3) lower 95% CI bound
stress_ci_hi    (N, 3, 3) upper 95% CI bound
umean_ci_lo     (N, 3)    lower 95% CI for mean velocity
umean_ci_hi     (N, 3)    upper 95% CI for mean velocity
u_tau           scalar    friction velocity (mm/s)
delta_nu        scalar    viscous length scale (mm)
Re_tau          scalar    friction Reynolds number
```

Two ground truth files are provided, one per test case:
- `cross_method_fine/direct_stats.mat` — computed from 85,000-particle images (Case A)
- `noisy_cross_method_coarse/direct_stats.mat` — computed from 22,000-particle images (Case B)


## Unit Conventions

| Quantity | PIVTOOLs storage | Display / benchmark |
|----------|-----------------|-------------------|
| Velocity | m/s | mm/s (multiply by 1000) |
| Stress | (m/s)^2 | (mm/s)^2 (multiply by 1e6) |
| Coordinates | mm | wall units y+ = y / delta_nu |

