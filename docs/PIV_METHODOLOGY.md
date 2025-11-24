# PIV Methodology: Instantaneous, Ensemble Standard, and Ensemble Single Mode

**Comprehensive Technical Documentation**
*Last Updated: 2025*

---

## Table of Contents

1. [Coordinate Systems & Indexing Conventions](#1-coordinate-systems--indexing-conventions)
2. [Instantaneous PIV](#2-instantaneous-piv)
3. [Ensemble PIV: Standard Mode](#3-ensemble-piv-standard-mode)
4. [Ensemble PIV: Single Mode](#4-ensemble-piv-single-mode)
5. [Masking Strategy](#5-masking-strategy)
6. [Reynolds Stress Decomposition](#6-reynolds-stress-decomposition)
7. [Comparison Matrix](#7-comparison-matrix)

---

## 1. Coordinate Systems & Indexing Conventions

### 1.1 Indexing Philosophy

**PyPIVTools uses 0-based Cartesian indexing throughout:**

- **Origin:** `(0, 0)` represents the bottom-left corner conceptually
- **X-axis:** Horizontal, increasing to the right (column index)
- **Y-axis:** Vertical, increasing upward (row index)
- **Array Storage:** NumPy row-major (C-style): `array[row, col]` = `array[y, x]`

### 1.2 Physical vs Storage Indexing

```
Physical Interpretation (Cartesian):
Y (vertical, height, upward)
▲
│  (0,H-1)              (W-1,H-1)
│    ┌──────────────────────┐
│    │                      │
│    │      Image           │
│    │                      │
│    └──────────────────────┘
│  (0,0)                  (W-1,0)
└──────────────────────────────► X (horizontal, width, rightward)

Coordinate Convention:
  - Origin (0,0) is at BOTTOM-LEFT corner
  - X increases rightward (horizontal)
  - Y increases upward (vertical)

Array Storage (Row-Major, NumPy):
  - Arrays stored as array[row, col] = array[y, x]
  - array[0, 0]      = pixel at y=0, x=0 (bottom-left physically)
  - array[H-1, W-1]  = pixel at y=H-1, x=W-1 (top-right physically)
  - Row index 0 = y-coordinate 0 (bottom of image)
  - Row index H-1 = y-coordinate H-1 (top of image)

Note: When displaying with matplotlib, use origin='lower' to match Cartesian convention

Displacement Fields:
 ux_mat[iy, ix] = displacement in X direction at grid point (x=ix, y=iy)
 uy_mat[iy, ix] = displacement in Y direction at grid point (x=ix, y=iy)
```

### 1.3 Window Center Coordinates

Window centers use **floating-point pixel coordinates**:

```
For 128-pixel window in 4872-pixel image (0-based):
  First center: (128 - 1) / 2.0 = 63.5
  Last center:  4872 - (128 + 1) / 2.0 = 4807.5

Window spans pixels [0, 127] centered at 63.5
```

**Mathematical Formula:**
```
first_ctr = (win_size - 1) / 2.0
last_ctr = image_size - (win_size + 1) / 2.0
n_windows = floor((last_ctr - first_ctr) / spacing) + 1
```

---

## 2. Instantaneous PIV

### 2.1 Overview

**Instantaneous PIV** processes each image pair independently to extract instantaneous velocity fields. It uses a multi-pass iterative window deformation approach.

### 2.2 Window Positioning

#### Standard Window Grid

```python
# Example: 64x64 window, 50% overlap, 512x512 image
window_size = (64, 64)  # (height, width)
overlap = 50            # percent

# Spacing between windows
win_spacing_x = round((1 - 0.5) * 64) = 32 pixels
win_spacing_y = round((1 - 0.5) * 64) = 32 pixels

# Window centers (0-based)
first_ctr_x = (64 - 1) / 2 = 31.5
last_ctr_x = 512 - (64 + 1) / 2 = 479.5

# Number of windows
n_win_x = floor((479.5 - 31.5) / 32) + 1 = 15
n_win_y = 15

# Center positions
win_ctrs_x = [31.5, 63.5, 95.5, ..., 479.5]  # 15 centers
win_ctrs_y = [31.5, 63.5, 95.5, ..., 479.5]  # 15 centers
```

**Grid Layout:**
```
                         X →
       31.5    63.5    95.5   ...   479.5
      ┌───┬───┬───┬───────────┬───┐
31.5  │ ● │ ● │ ● │    ...    │ ● │
      ├───┼───┼───┼───────────┼───┤
63.5  │ ● │ ● │ ● │    ...    │ ● │
      ├───┼───┼───┼───────────┼───┤
95.5  │ ● │ ● │ ● │    ...    │ ● │
 Y    │ . │ . │ . │     .     │ . │
 ↓    │ . │ . │ . │     .     │ . │
      ├───┼───┼───┼───────────┼───┤
479.5 │ ● │ ● │ ● │    ...    │ ● │
      └───┴───┴───┴───────────┴───┘

● = window center
Grid = 15 × 15 = 225 vectors
```

### 2.3 Window Weighting

**Gaussian Window (Default):**
```python
# Creates smooth edges to reduce correlation errors
window_weight = exp(-((x - cx)² + (y - cy)²) / (2σ²))

# Where σ is chosen based on window size
σ = win_size / 6  # Approximately 3σ = half-window
```

**Standard Window ('A' type):**
```python
# Uniform weighting (all 1s)
window_weight = ones((win_height, win_width))
```

### 2.4 Multi-Pass Strategy

**Typical 3-pass configuration:**

```yaml
instantaneous_piv:
  window_size:
    - [128, 128]  # Pass 1: Large window (robust)
    - [64, 64]    # Pass 2: Medium window
    - [32, 32]    # Pass 3: Small window (high resolution)
  overlap: [50, 50, 50]
  runs: [3]  # Save only final pass
```

**Process Flow:**

```
Pass 1 (128×128):
  Image A ──┐
            ├──► Cross-correlate ──► Displacement field (coarse)
  Image B ──┘                              │
                                           │
                                           ▼
Pass 2 (64×64):                    Predictor field
  Image A ──► Warp ──┐                     │
                     ├──► Cross-correlate ─┤
  Image B ──► Warp ──┘                     │
                                           ▼
Pass 3 (32×32):                    Predictor field
  Image A ──► Warp ──┐                     │
                     ├──► Cross-correlate ─┴──► Final velocity
  Image B ──► Warp ──┘
```

**Image Warping (Predictor-Corrector):**
```python
# For pass i > 0, warp images using previous pass displacement
predictor_field = smooth(pass[i-1].displacement)

# Warp images to remove bulk motion
Image_A' = warp(Image_A, -predictor_field / 2)
Image_B' = warp(Image_B, +predictor_field / 2)

# Correlation measures residual displacement
residual = correlate(Image_A', Image_B')

# Total displacement
total_displacement = predictor_field + residual
```

### 2.5 Instantaneous Output Fields

**Primary Outputs (per image pair, per pass):**

| Field | Shape | Type | Units | Description |
|-------|-------|------|-------|-------------|
| `ux_mat` | `(n_win_y, n_win_x)` | float32 | pixels | X-displacement |
| `uy_mat` | `(n_win_y, n_win_x)` | float32 | pixels | Y-displacement |
| `primary_peak_mag` | `(n_win_y, n_win_x)` | float32 | - | Correlation peak height |
| `secondary_peak_mag` | `(n_win_y, n_win_x)` | float32 | - | 2nd peak height (if enabled) |
| `peak_ratio` | `(n_win_y, n_win_x)` | float32 | - | primary / secondary |

**Grid Coordinates:**

| Field | Shape | Type | Units | Description |
|-------|-------|------|-------|-------------|
| `win_ctrs_x` | `(n_win_x,)` | float32 | pixels | X-coordinates of window centers |
| `win_ctrs_y` | `(n_win_y,)` | float32 | pixels | Y-coordinates of window centers |

**Velocity Conversion to Physical Units:**
```python
# From config calibration
px_per_mm = 3.41  # pixels per millimeter
dt = 0.56         # microseconds between frames

# Convert displacement to velocity
vx = (ux_mat / px_per_mm) / (dt * 1e-6)  # mm/s → m/s
vy = (uy_mat / px_per_mm) / (dt * 1e-6)  # mm/s → m/s
```

### 2.6 Instantaneous Masking

**Vector masks applied AFTER correlation:**
```python
# Mask out invalid regions (boundaries, objects)
if vector_mask is not None:
    ux_mat[vector_mask] = 0.0  # or np.nan
    uy_mat[vector_mask] = 0.0
```

### 2.7 Outlier Detection & Infilling

**Mid-pass outliers (passes 1 to N-1):**
```python
# Detect outliers
outlier_mask = detect_outliers(ux, uy, method='median_2d', threshold=2)

# Infill for predictor field
ux_infilled, uy_infilled = infill(ux, uy, outlier_mask, method='biharmonic')

# Use infilled field as predictor for next pass
```

**Final pass outliers:**
```python
# Optionally infill (or leave as NaN for user analysis)
if config.infilling_final_pass['enabled']:
    ux_final, uy_final = infill(ux, uy, outlier_mask, method='biharmonic')
```

---

## 3. Ensemble PIV: Standard Mode

### 3.1 Overview

**Ensemble PIV** averages correlation planes across multiple image pairs before peak detection. This provides:
- **Mean velocity field** (time-averaged)
- **Reynolds stress tensor** (turbulence statistics)
- **Uncertainty quantification** via Gaussian fitting

### 3.2 Key Difference from Instantaneous

```
Instantaneous:
  For each image pair:
    Correlate → Find peak → Extract velocity
  Output: N velocity fields (one per pair)

Ensemble:
  Correlate all N pairs → Average correlation planes → Fit peak
  Output: 1 mean velocity field + stress tensor
```

### 3.3 Ensemble Averaging Process

**Mathematical Formulation:**

```
For N image pairs {(A₁, B₁), (A₂, B₂), ..., (Aₙ, Bₙ)}:

1. Warp each pair using predictor field:
   A'ᵢ = warp(Aᵢ, -δ_pred/2)
   B'ᵢ = warp(Bᵢ, +δ_pred/2)

2. Remove mean intensity:
   Ā = (1/N) Σ A'ᵢ
   B̄ = (1/N) Σ B'ᵢ

   Ãᵢ = A'ᵢ - Ā  (fluctuation)
   B̃ᵢ = B'ᵢ - B̄  (fluctuation)

3. Compute ensemble correlation planes:
   C_AB = (1/N) Σ FFT⁻¹[FFT(Ãᵢ) × FFT*(B̃ᵢ)]  (cross-correlation)
   C_AA = (1/N) Σ FFT⁻¹[FFT(Ãᵢ) × FFT*(Ãᵢ)]  (auto-correlation A)
   C_BB = (1/N) Σ FFT⁻¹[FFT(B̃ᵢ) × FFT*(B̃ᵢ)]  (auto-correlation B)

4. Fit stacked Gaussian to extract parameters
```

### 3.4 Window Positioning (Same as Instantaneous)

**Standard ensemble mode uses identical window positioning to instantaneous:**

```python
# Example: 64×64 window, 50% overlap
result = compute_window_centers(
    image_shape=(512, 512),
    window_size=(64, 64),
    overlap=50
)

# Same formulas as instantaneous mode
win_ctrs_x = [31.5, 63.5, ..., 479.5]  # 15 centers
win_ctrs_y = [31.5, 63.5, ..., 479.5]  # 15 centers
```

### 3.5 Gaussian Fitting (Levenberg-Marquardt)

**Stacked Gaussian Model:**

The ensemble correlation planes are modeled as a sum of three 2D Gaussians:

```
C_total(x, y) = G_A(x, y) + G_B(x, y) + G_AB(x, y)

Where:
  G_A(x, y)  = h_A  × exp(-½[(x-x₀)²/σ²_Ax + (y-y₀)²/σ²_Ay + 2ρ_A(x-x₀)(y-y₀)/(σ_Ax·σ_Ay)])
  G_B(x, y)  = h_B  × exp(-½[(x-x₀)²/σ²_Bx + (y-y₀)²/σ²_By + 2ρ_B(x-x₀)(y-y₀)/(σ_Bx·σ_By)])
  G_AB(x, y) = h_AB × exp(-½[(x-μ_x)²/σ²_ABx + (y-μ_y)²/σ²_ABy + 2ρ_AB(x-μ_x)(y-μ_y)/(σ_ABx·σ_ABy)])
```

**Parameters (13 total per window):**

| Parameter | Symbol | Physical Meaning |
|-----------|--------|------------------|
| `h_A` | Peak height | Auto-correlation A amplitude |
| `h_B` | Peak height | Auto-correlation B amplitude |
| `h_AB` | Peak height | Cross-correlation amplitude |
| `σ_Ax, σ_Ay, ρ_A` | Spread | Particle distribution spread (Frame A) |
| `σ_Bx, σ_By, ρ_B` | Spread | Particle distribution spread (Frame B) |
| `σ_PDx, σ_PDy, ρ_PD` | Spread | Particle displacement variance (Reynolds stress) |
| `x₀, y₀` | Position | Center of auto-correlations (should be ~window_center) |
| `μ_x, μ_y` | Position | Cross-correlation peak (mean displacement) |

**Fitting Process:**
```c
// C library call (GSL Levenberg-Marquardt)
fit_stacked_gaussian_export(
    win_size,          // Correlation plane size
    X1, X2,            // Grid coordinates
    real_corr,         // Concatenated [C_AA, C_BB, C_AB]
    initial_guess,     // 13 parameters
    out_params,        // Fitted parameters
    out_status         // Fit success/failure
);
```

### 3.6 Ensemble Output Fields

**Mean Velocity Fields:**

| Field | Shape | Type | Units | Description |
|-------|-------|------|-------|-------------|
| `ux_mat` | `(n_win_y, n_win_x)` | float32 | pixels | Mean X-displacement |
| `uy_mat` | `(n_win_y, n_win_x)` | float32 | pixels | Mean Y-displacement |

**Reynolds Stress Tensor Components:**

| Field | Shape | Type | Units | Description |
|-------|-------|------|-------|-------------|
| `UU_stress` | `(n_win_y, n_win_x)` | float32 | pixels² | ⟨u'u'⟩ Normal stress in X |
| `VV_stress` | `(n_win_y, n_win_x)` | float32 | pixels² | ⟨v'v'⟩ Normal stress in Y |
| `UV_stress` | `(n_win_y, n_win_x)` | float32 | pixels² | ⟨u'v'⟩ Shear stress |

**Peak Heights (Correlation Amplitudes):**

| Field | Shape | Type | Units | Description |
|-------|-------|------|-------|-------------|
| `peakheights_A` | `(n_win_y, n_win_x)` | float32 | - | Auto-correlation A amplitude |
| `peakheights_B` | `(n_win_y, n_win_x)` | float32 | - | Auto-correlation B amplitude |
| `peakheights_AB` | `(n_win_y, n_win_x)` | float32 | - | Cross-correlation amplitude |

**Gaussian Width Parameters (Uncertainty Measures):**

| Field | Shape | Type | Units | Description |
|-------|-------|------|-------|-------------|
| `sig_A_x, sig_A_y, sig_A_xy` | `(n_win_y, n_win_x)` | float32 | pixels | Particle spread (Frame A) |
| `sig_PD_x, sig_PD_y, sig_PD_xy` | `(n_win_y, n_win_x)` | float32 | pixels | Displacement variance (turbulence) |
| `sig_AB_x, sig_AB_y, sig_AB_xy` | `(n_win_y, n_win_x)` | float32 | pixels | Total spread = sig_A + sig_PD |

**Fit Quality:**

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `nan_reason` | `(n_win_y, n_win_x)` | int32 | 0=valid, 1=fit_failed, 2=outlier |

### 3.7 Extraction from Fitted Parameters

```python
# After Gaussian fitting, extract fields from parameters
for iy in range(n_win_y):
    for ix in range(n_win_x):
        params = gauss_results[iy, ix, :]  # 13 parameters

        # Peak heights
        peakheights_A[iy, ix] = params[0]   # h_A
        peakheights_B[iy, ix] = params[1]   # h_B
        peakheights_AB[iy, ix] = params[2]  # h_AB

        # Particle spread (Frame A)
        sig_A_x[iy, ix] = params[3]   # σ_Ax
        sig_A_y[iy, ix] = params[4]   # σ_Ay
        sig_A_xy[iy, ix] = params[5]  # ρ_A

        # Reynolds stress (Particle Displacement variance)
        sig_PD_x[iy, ix] = params[6]   # σ_PDx → ⟨u'u'⟩
        sig_PD_y[iy, ix] = params[7]   # σ_PDy → ⟨v'v'⟩
        sig_PD_xy[iy, ix] = params[8]  # ρ_PD  → ⟨u'v'⟩

        # Mean displacement (from cross-correlation peak)
        x_offset = win_size[1] / 2 + 1
        y_offset = win_size[0] / 2 + 1

        ux_mat[iy, ix] = params[11] - x_offset  # μ_x - center
        uy_mat[iy, ix] = params[12] - y_offset  # μ_y - center

        # Reynolds stresses = particle displacement variance
        UU_stress[iy, ix] = sig_PD_x[iy, ix]  # ⟨u'u'⟩
        VV_stress[iy, ix] = sig_PD_y[iy, ix]  # ⟨v'v'⟩
        UV_stress[iy, ix] = sig_PD_xy[iy, ix] # ⟨u'v'⟩
```

---

## 4. Ensemble PIV: Single Mode

### 4.1 Motivation

**Problem with Standard Ensemble PIV:**

When particle images are sparse or have high displacement, standard ensemble averaging can suffer from **particle dropout bias**:

```
Standard Mode (both frames use same 64×64 window):
  Frame A Window:     Frame B Window:
  ┌──────────┐       ┌──────────┐
  │ ●  ●     │       │      ●  ●│  ← Particles moved out!
  │   ●   ●  │       │   ●   ●  │
  │ ●     ●  │       │ ●     ●  │
  └──────────┘       └──────────┘

  Problem: Different particles in A vs B → biased correlation
```

**Single Mode Solution:**

Use a **small window for Frame A** (captures same particles) and **large window for Frame B** (captures displaced particles):

```
Single Mode (4×4 window for A, 16×16 for B):
  Frame A Window:     Frame B Window:
     ┌──┐         ┌──────────────┐
     │●●│         │              │
     │●●│         │  ●  ●        │  ← Same particles!
     └──┘         │   ●   ●      │
                  │ ●     ●      │
                  └──────────────┘

  Benefit: Consistent particle set → unbiased statistics
```

### 4.2 Asymmetric Window Weighting

**Frame A: Small Weighted Window**
```python
# "singlepix" mode - only center 4×4 region weighted as 1
window_weight_A = create_singlepix_window(
    small_size=(4, 4),      # Actual interrogation region
    sum_window=(16, 16)     # Padded to match Frame B size
)

# Result: 16×16 array
# ┌──────────────────┐
# │ 0  0  0  0  0  0 │
# │ 0  0  0  0  0  0 │
# │ 0  0  1  1  0  0 │  ← Center 4×4 = 1
# │ 0  0  1  1  0  0 │
# │ 0  0  0  0  0  0 │
# │ 0  0  0  0  0  0 │
# └──────────────────┘
```

**Frame B: Full SumWindow**
```python
# "bsingle" mode - full 16×16 weighted as 1
window_weight_B = create_bsingle_window(
    sum_window=(16, 16)
)

# Result: 16×16 array of all 1s
# ┌──────────────────┐
# │ 1  1  1  1  1  1 │
# │ 1  1  1  1  1  1 │
# │ 1  1  1  1  1  1 │
# │ 1  1  1  1  1  1 │
# │ 1  1  1  1  1  1 │
# │ 1  1  1  1  1  1 │
# └──────────────────┘
```

### 4.3 Image Padding for Single Mode

**Why Padding is Needed:**

The small window (4×4) is positioned on the grid, but the correlation requires a 16×16 region around each position. Images must be padded to accommodate this.

**Padding Calculation:**
```python
# MATLAB reference: PIV_2D_wdef_ensemble.m lines 161-164
window_size = (4, 4)      # Small window (Frame A)
sum_window = (16, 16)     # Large window (correlation size)

pad_top = ceil((16 - 4) / 2) = 6 pixels
pad_bottom = floor((16 - 4) / 2) = 6 pixels
pad_left = ceil((16 - 4) / 2) = 6 pixels
pad_right = floor((16 - 4) / 2) = 6 pixels

# Padded image dimensions
H_padded = H + 12  # Original height + pad_top + pad_bottom
W_padded = W + 12  # Original width + pad_left + pad_right
```

**Padding Application:**
```python
# Zero-padding (constant value = 0)
image_A_padded = np.pad(
    image_A,
    pad_width=((6, 6), (6, 6)),  # (top, bottom), (left, right)
    mode='constant',
    constant_values=0
)

image_B_padded = np.pad(
    image_B,
    pad_width=((6, 6), (6, 6)),
    mode='constant',
    constant_values=0
)
```

### 4.4 Window Center Positioning for Single Mode

**Key Concept:** Grid spacing is based on the **small window** (determines resolution), but centers are positioned relative to the **SumWindow** (determines correlation feasibility).

```python
# Configuration
window_size = (4, 4)      # Small window
sum_window = (16, 16)     # Large window
overlap = 50              # Applied to small window
image_shape = (512, 512)  # Original image

# Spacing based on SMALL window
win_spacing_x = round((1 - 0.5) * 4) = 2 pixels  # High resolution!
win_spacing_y = round((1 - 0.5) * 4) = 2 pixels

# Padded image dimensions
H_padded = 512 + 12 = 524
W_padded = 512 + 12 = 524

# First center positioned relative to SumWindow
# (Using formula for window_size != 1)
first_ctr_x = 0.5 + 16/2 = 8.5  # On padded image
first_ctr_y = 0.5 + 16/2 = 8.5

# Last center ensures full SumWindow fits
last_ctr_x = 524 - 16/2 + 0.5 = 515.5
last_ctr_y = 524 - 16/2 + 0.5 = 515.5

# Number of windows (many due to 2-pixel spacing!)
n_win_x = floor((515.5 - 8.5) / 2) + 1 = 254
n_win_y = floor((515.5 - 8.5) / 2) + 1 = 254

# Window centers
win_ctrs_x = [8.5, 10.5, 12.5, ..., 515.5]  # 254 centers (on padded image)
win_ctrs_y = [8.5, 10.5, 12.5, ..., 515.5]  # 254 centers
```

**Grid Visualization:**
```
Padded Image (524 × 524):
┌────────────────────────────────────┐
│         Padding (6 pixels)         │
│  ┌──────────────────────────────┐  │
│  │                              │  │
│  │    Original Image (512×512)  │  │
│  │                              │  │
│  │   ●─●─●─●─●─●─● ...          │  │ ← Window centers
│  │   │ │ │ │ │ │ │              │  │   (2-pixel spacing)
│  │   ●─●─●─●─●─●─●              │  │
│  │   │ │ │ │ │ │ │              │  │
│  │   . . . . . . .              │  │
│  │                              │  │
│  └──────────────────────────────┘  │
│                                    │
└────────────────────────────────────┘

Each ● represents center of 16×16 SumWindow on padded image
Grid spacing = 2 pixels (from 4×4 small window)
```

### 4.5 Correlation Size

**Critical Detail:** The C library must receive the **SumWindow size**, not the small window size:

```python
# Standard mode
win_size_arr = np.array([64, 64], dtype=np.int32)  # 64×64 correlation

# Single mode
win_size_arr = np.array([16, 16], dtype=np.int32)  # 16×16 correlation!
                                                     # (NOT [4, 4])
```

**Correlation Plane Sizes:**
```
Standard Mode:
  64×64 window → 64×64 correlation plane

Single Mode:
  4×4 small window + 16×16 SumWindow → 16×16 correlation plane
```

### 4.6 Displacement Extraction

**Offset Calculation Uses SumWindow Size:**

```python
# Standard mode (64×64)
x_offset = 64 / 2 + 1 = 33
y_offset = 64 / 2 + 1 = 33

# Single mode (16×16 SumWindow)
x_offset = 16 / 2 + 1 = 9   # Uses SumWindow, NOT small window!
y_offset = 16 / 2 + 1 = 9

# Extract displacement from Gaussian fit
ux = fitted_params[11] - x_offset
uy = fitted_params[12] - y_offset
```

### 4.7 Single Mode Output Fields

**Same structure as standard ensemble mode:**

| Field | Shape | Description |
|-------|-------|-------------|
| `ux_mat` | `(254, 254)` | Mean X-displacement (high resolution!) |
| `uy_mat` | `(254, 254)` | Mean Y-displacement |
| `UU_stress` | `(254, 254)` | Reynolds normal stress ⟨u'u'⟩ |
| `VV_stress` | `(254, 254)` | Reynolds normal stress ⟨v'v'⟩ |
| `UV_stress` | `(254, 254)` | Reynolds shear stress ⟨u'v'⟩ |
| ... | ... | (All other ensemble fields) |

**Key Advantage:** Much higher spatial resolution (254×254 vs 15×15 for same image) with unbiased statistics!

### 4.8 Configuration Example

```yaml
ensemble_piv:
  window_size:
    - [128, 128]    # Pass 1: Standard mode, coarse
    - [64, 64]      # Pass 2: Standard mode, medium
    - [4, 4]        # Pass 3: Single mode, fine resolution!
  overlap: [50, 50, 50]
  sum_window: [16, 16]  # Required for single mode
  type:
    - 'std'         # Pass 1: standard
    - 'std'         # Pass 2: standard
    - 'single'      # Pass 3: single mode
```

---

## 5. Masking Strategy

### 5.1 Pixel Mask → Vector Mask Conversion

**Objective:** Convert a boolean pixel mask (e.g., masking out boundaries or solid objects) into a vector mask that identifies which PIV vectors should be invalidated.

### 5.2 Convolution-Based Approach

**Process (matches MATLAB `compute_b_mask.m`):**

```python
# Input: pixel_mask[H, W] - True = masked pixel
# Output: vector_mask[n_win_y, n_win_x] - True = masked vector

# 1. Convert to float
im_mask = pixel_mask.astype(float32)  # True → 1.0, False → 0.0

# 2. Convolve with box filter (window size)
box_filter_y = ones((win_height, 1)) / win_height
box_filter_x = ones((1, win_width)) / win_width

f_mask = convolve(im_mask, box_filter_y, mode='constant')
f_mask = convolve(f_mask, box_filter_x, mode='constant')

# Result: f_mask[y, x] = fraction of masked pixels in window centered at (x, y)

# 3. Interpolate at window centers
win_y_idx = round(win_ctrs_y).astype(int)
win_x_idx = round(win_ctrs_x).astype(int)

# Clip to valid indices
win_y_idx = np.clip(win_y_idx, 0, H - 1)
win_x_idx = np.clip(win_x_idx, 0, W - 1)

# Sample filtered mask
mask_values = f_mask[win_y_idx, win_x_idx]  # Grid of mask fractions

# 4. Apply threshold
threshold = 0.5  # Default: mask if >50% of window is masked
vector_mask = mask_values > threshold
```

### 5.3 Threshold Interpretation

```
threshold = 0.0:  Mask vector if ANY pixel in window is masked
threshold = 0.5:  Mask vector if >50% of pixels in window are masked (default)
threshold = 1.0:  Mask vector ONLY if ALL pixels in window are masked
```

**Visual Example:**
```
Pixel Mask (20×20):
┌────────────────────┐
│ ░░░░░░░░░░░░░░░░░░ │  ░ = masked (True)
│ ░░░░░░░░░░░░░░░░░░ │  ░ = masked boundary
│ ░░                ░░ │
│ ░░                ░░ │
│ ░░      ●         ░░ │  ● = window center
│ ░░                ░░ │
│ ░░                ░░ │
│ ░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░ │
└────────────────────┘

Convolved Mask (fractions):
  0.8  0.6  0.2  0.1  0.1  ...
  0.6  0.4  0.1  0.0  0.0  ...
  0.3  0.2  0.0  0.0  0.0  ...  ← At window center: 0.0 (not masked)
  ...

Vector Mask (threshold=0.5):
  True  True  False False False ...  ← Top-left vectors masked
  True  False False False False ...
  False False False False False ...
  ...
```

### 5.4 Single Mode Masking

**For single mode, use the SumWindow size for mask computation:**

```python
# Standard mode
vector_mask = compute_vector_mask(
    pixel_mask,
    window_size=(64, 64),
    overlap=50
)

# Single mode
vector_mask = compute_vector_mask(
    pixel_mask,
    window_size=(4, 4),       # Small window determines grid
    sum_window=(16, 16),      # SumWindow for mask convolution
    overlap=50
)
```

### 5.5 Mask Application

**Instantaneous PIV:**
```python
# Applied after correlation
ux_mat[vector_mask] = 0.0  # or np.nan
uy_mat[vector_mask] = 0.0
```

**Ensemble PIV:**
```python
# Passed to C library as b_mask
b_mask = vector_mask.astype(float32).ravel()

# Library sets correlation = 0 for masked windows
# After Gaussian fitting:
ux_mat[vector_mask] = 0.0
uy_mat[vector_mask] = 0.0
```

---

## 6. Reynolds Stress Decomposition

### 6.1 Theoretical Background

**Reynolds Decomposition:**

```
Instantaneous velocity = Mean + Fluctuation
  u(x, t) = ⟨u⟩(x) + u'(x, t)
  v(x, t) = ⟨v⟩(x) + v'(x, t)

Where:
  ⟨u⟩ = (1/N) Σ u(x, tᵢ)  (ensemble average)
  u' = u - ⟨u⟩             (fluctuation)
```

**Reynolds Stress Tensor:**

```
τᵢⱼ = ⟨uᵢ' uⱼ'⟩ = (1/N) Σ uᵢ'(tₙ) uⱼ'(tₙ)

In 2D:
  τₓₓ = ⟨u'u'⟩  (normal stress in x)
  τᵧᵧ = ⟨v'v'⟩  (normal stress in y)
  τₓᵧ = ⟨u'v'⟩  (shear stress)
```

### 6.2 Extraction from Ensemble PIV

**From Gaussian Fitting:**

The particle displacement variance parameters directly provide Reynolds stresses:

```python
# Fitted parameters from stacked Gaussian
σ_PDx = params[6]   # Variance of u' displacement
σ_PDy = params[7]   # Variance of v' displacement
ρ_PD = params[8]    # Covariance of u', v'

# Reynolds stresses (in pixel² units)
UU_stress[iy, ix] = σ_PDx  # ⟨u'u'⟩
VV_stress[iy, ix] = σ_PDy  # ⟨v'v'⟩
UV_stress[iy, ix] = ρ_PD   # ⟨u'v'⟩
```

**Physical Interpretation:**

```
UU_stress = Variance of X-displacement
          = How much particles fluctuate horizontally

VV_stress = Variance of Y-displacement
          = How much particles fluctuate vertically

UV_stress = Covariance of X and Y displacements
          = Correlation between horizontal and vertical fluctuations
```

### 6.3 Conversion to Physical Units

```python
# From config calibration
px_per_mm = 3.41     # pixels/mm
dt = 0.56e-6         # seconds

# Convert pixel² → (m/s)²
scale = (1 / px_per_mm / 1000) / dt  # pixels → meters, dt → velocity

UU_stress_physical = UU_stress * scale²  # m²/s²
VV_stress_physical = VV_stress * scale²  # m²/s²
UV_stress_physical = UV_stress * scale²  # m²/s²
```

### 6.4 Derived Quantities

**Turbulent Kinetic Energy (TKE):**
```python
# TKE = ½⟨u'² + v'²⟩
TKE = 0.5 * (UU_stress + VV_stress)  # m²/s²
```

**Turbulence Intensity:**
```python
# RMS fluctuations
u_rms = sqrt(UU_stress)  # m/s
v_rms = sqrt(VV_stress)  # m/s

# Turbulence intensity (%)
U_mean = sqrt(ux_mat² + uy_mat²)  # Mean velocity magnitude
TI = 100 * sqrt(TKE) / U_mean
```

**Reynolds Shear Stress (τ_xy):**
```python
# In fluid dynamics: τ_xy = -ρ⟨u'v'⟩
# Where ρ = fluid density

# For water at 20°C: ρ = 998 kg/m³
rho = 998  # kg/m³

tau_xy = -rho * UV_stress_physical  # Pa (N/m²)
```

### 6.5 Anisotropy Analysis

```python
# Principal stresses (eigenvalues of stress tensor)
stress_tensor = np.array([
    [UU_stress, UV_stress],
    [UV_stress, VV_stress]
])

eigenvalues, eigenvectors = np.linalg.eig(stress_tensor)

λ_1 = eigenvalues[0]  # Major principal stress
λ_2 = eigenvalues[1]  # Minor principal stress

# Anisotropy measure
anisotropy = (λ_1 - λ_2) / (λ_1 + λ_2)

# 0 = isotropic turbulence
# 1 = highly anisotropic (1D turbulence)
```

### 6.6 Uncertainty Quantification

**Total Correlation Width:**
```python
# Total variance = particle spread + displacement variance
sig_AB_x = sig_A_x + sig_PD_x  # Total X spread
sig_AB_y = sig_A_y + sig_PD_y  # Total Y spread

# Narrower peaks → more certain measurement
# Wider peaks → higher uncertainty
```

**Peak Signal-to-Noise Ratio:**
```python
# Compare cross-correlation peak to auto-correlations
SNR = peakheights_AB / sqrt(peakheights_A * peakheights_B)

# High SNR (>0.5) → reliable measurement
# Low SNR (<0.2) → questionable quality
```

---

## 7. Comparison Matrix

### 7.1 Feature Comparison

| Feature | Instantaneous | Ensemble Standard | Ensemble Single |
|---------|---------------|-------------------|-----------------|
| **Processing** | Per image pair | Averaged over N pairs | Averaged over N pairs |
| **Output** | N velocity fields | 1 mean field + stresses | 1 mean field + stresses |
| **Window Size** | Symmetric (e.g., 64×64) | Symmetric (e.g., 64×64) | Asymmetric (4×4 / 16×16) |
| **Grid Resolution** | Moderate | Moderate | High |
| **Padding** | None | None | Required (SumWindow) |
| **Peak Detection** | 3-point Gaussian fit | Levenberg-Marquardt (13 params) | Levenberg-Marquardt (13 params) |
| **Reynolds Stresses** | Not computed | Computed (σ_PD) | Computed (σ_PD) |
| **Particle Dropout** | Not an issue | Can bias results | Mitigated |
| **Best For** | Time-resolved flow | Turbulence statistics | High-resolution statistics |

### 7.2 Computational Cost

**For 100 image pairs, 512×512 images, 64×64 windows:**

| Mode | Grid Size | Correlations | Fitting | Relative Cost |
|------|-----------|--------------|---------|---------------|
| Instantaneous | 15×15 | 22,500 (100×15×15) | 22,500 simple | 1.0× |
| Ensemble Std | 15×15 | 675 (3×15×15) | 225 complex | 0.5× |
| Ensemble Single | 254×254 | 194,088 (3×254×254) | 64,516 complex | 5.0× |

**Notes:**
- Ensemble modes are cheaper per vector (average correlation planes)
- Single mode produces 286× more vectors (254×254 vs 15×15)
- Single mode benefits from GPU acceleration due to high parallelism

### 7.3 Output Comparison

**Instantaneous PIV (100 pairs):**
```
File Structure:
  results/
    00001.mat: {ux, uy, peak_mag} - 15×15 each
    00002.mat: {ux, uy, peak_mag} - 15×15 each
    ...
    00100.mat: {ux, uy, peak_mag} - 15×15 each

Total: 100 files × 3 fields × 15×15 = 67.5 KB
```

**Ensemble Standard (100 pairs):**
```
File Structure:
  results/
    ensemble_pass3.mat:
      ux, uy           : 15×15 (mean velocity)
      UU, VV, UV       : 15×15 (Reynolds stresses)
      peakheights_AB   : 15×15 (correlation amplitude)
      sig_PD_x, sig_PD_y: 15×15 (turbulence measures)
      ... (13 fields total)

Total: 1 file × 13 fields × 15×15 = 2.9 KB
```

**Ensemble Single (100 pairs):**
```
File Structure:
  results/
    ensemble_pass3.mat:
      ux, uy           : 254×254 (high-res mean velocity!)
      UU, VV, UV       : 254×254 (high-res stresses)
      peakheights_AB   : 254×254
      sig_PD_x, sig_PD_y: 254×254
      ... (13 fields total)

Total: 1 file × 13 fields × 254×254 = 838 KB
```

### 7.4 Use Case Recommendations

**Choose Instantaneous PIV when:**
- Need time-resolved velocity evolution
- Analyzing transient phenomena
- Tracking vortex dynamics
- Computing POD/DMD modes
- Temporal frequency analysis

**Choose Ensemble Standard Mode when:**
- Need turbulence statistics
- Studying steady/statistically-stationary flow
- Measuring Reynolds stresses
- Moderate spatial resolution acceptable
- Limited computational resources

**Choose Ensemble Single Mode when:**
- Need high spatial resolution AND turbulence statistics
- Particles are sparse or highly displaced
- Concerned about particle dropout bias
- Have GPU acceleration available
- Analyzing boundary layers or shear flows

---

## 8. Implementation Code Paths

### 8.1 Instantaneous PIV

```python
# Entry point: pivtools_core/example.py
from pivtools_cli.piv.piv import perform_piv_and_save

# Key files:
# - pivtools_cli/piv/piv_backend/cpu_instantaneous.py
# - pivtools_cli/piv/piv_backend/factory.py

# Window centers computed in:
# - cpu_instantaneous.py::_compute_window_centres()
#   → calls window_utils.compute_window_centers()

# Correlation performed in:
# - cpu_instantaneous.py::correlate_batch()
#   → calls C library: libbulkxcorr2d.so

# Outputs saved in:
# - pivtools_cli/piv/save_results.py::save_result_distributed()
```

### 8.2 Ensemble PIV

```python
# Entry point: pivtools_core/example.py
from pivtools_cli.piv.piv_backend.cpu_ensemble import perform_ensemble_piv

# Key files:
# - pivtools_cli/piv/piv_backend/cpu_ensemble.py
# - pivtools_cli/piv/piv_backend/factory.py

# Window centers computed in:
# - cpu_ensemble.py::_compute_window_centres_ensemble()
#   → calls window_utils.compute_window_centers() OR
#   → calls window_utils.compute_window_centers_single_mode()

# Correlation performed in:
# - cpu_ensemble.py::correlate_batch()
#   → applies padding for single mode
#   → calls C library: libbulkxcorr2d.so

# Gaussian fitting performed in:
# - cpu_ensemble.py::_fit_windows_batch()
#   → calls C library: libmarquadt.so (GSL Levenberg-Marquardt)

# Outputs saved in:
# - pivtools_cli/piv/save_results.py::save_ensemble_result_distributed()
```

### 8.3 Masking

```python
# Entry point: pivtools_core/example.py
from pivtools_core.image_handling.load_images import (
    load_mask_for_camera,
    compute_vector_mask
)

# Key files:
# - pivtools_core/image_handling/load_images.py

# Pixel mask loaded:
# - load_mask_for_camera()
#   → reads .mat file or creates rectangular mask

# Vector mask computed:
# - compute_vector_mask()
#   → calls window_utils.compute_window_centers() OR
#   → calls window_utils.compute_window_centers_single_mode()
#   → convolves pixel mask with box filter
#   → interpolates at window centers
```

---

## 9. Mathematical Reference

### 9.1 Cross-Correlation Definition

```
C_AB(Δx, Δy) = Σ Σ [A(x, y) × B(x + Δx, y + Δy)]
               x y

In Fourier domain (faster):
  C_AB = FFT⁻¹[FFT(A) × FFT*(B)]
```

### 9.2 Window Center Formula Derivation

```
For window of size W in image of size N:

- Window spans pixels [0, W-1] when centered at position c
- Left edge of window: c - (W-1)/2
- Right edge of window: c + (W-1)/2

For window to fit fully in image:
  Left edge ≥ 0:  c - (W-1)/2 ≥ 0  →  c ≥ (W-1)/2
  Right edge < N: c + (W-1)/2 < N  →  c < N - (W-1)/2

But we want half-pixel precision:
  First center: c_first = (W - 1) / 2.0
  Last center:  c_last = N - (W + 1) / 2.0

This ensures:
  - First window left edge = 0
  - Last window right edge = N - 1
```

### 9.3 Ensemble Decomposition

```
Mean:
  ⟨C_AB⟩ = (1/N) Σ C_AB_i
           i=1

Variance:
  Var(C_AB) = ⟨C_AB²⟩ - ⟨C_AB⟩²
            = (1/N) Σ C_AB_i² - [(1/N) Σ C_AB_i]²

Standard Error of Mean:
  SE = sqrt(Var / N)
```

---

## 10. Troubleshooting Guide

### 10.1 Common Issues

**Issue: Different number of vectors between instantaneous and ensemble**
```
Cause: Using different window sizes or image padding

Solution: Check that window_sizes and overlap match exactly
```

**Issue: Single mode gives NaN results**
```
Cause: SumWindow too small or missing configuration

Solution: Ensure sum_window ≥ 4× window_size in config
```

**Issue: Masked vectors still showing non-zero values**
```
Cause: Mask applied before predictor field added back

Solution: Masking happens AFTER adding predictor in ensemble mode
```

**Issue: Reynolds stresses are negative**
```
Cause: Variances should always be ≥ 0; negative means fit failed

Solution: Check nan_reason field, increase num_peaks, or adjust window size
```

---

## 11. References

### 11.1 Theoretical Background

1. **Adrian & Westerweel (2011).** *Particle Image Velocimetry.* Cambridge University Press.
2. **Raffel et al. (2018).** *Particle Image Velocimetry: A Practical Guide.* Springer.
3. **Sciacchitano & Wieneke (2016).** "PIV uncertainty propagation." *Meas. Sci. Technol.* 27: 084006.

### 11.2 Ensemble PIV

4. **Meinhart et al. (2000).** "On the existence of uniform momentum zones in a turbulent boundary layer." *Phys. Fluids* 12: 2185.
5. **Westerweel et al. (2004).** "Single-pixel resolution ensemble correlation for micro-PIV." *Exp. Fluids* 37: 375-384.

### 11.3 Implementation

6. **MATLAB PIV Suite (LaVision DaVis)** - Original reference implementation
7. **PyPIVTools Documentation** - This codebase

---

**End of Document**
