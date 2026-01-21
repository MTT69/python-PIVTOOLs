# Sum Fitting Window Feature

## Overview

The `sum_fitting_window` feature allows computing correlations on a larger window (`sum_window`) while extracting, storing, and fitting only the central region. This provides memory savings and faster fitting without sacrificing correlation quality.

## Motivation

In ensemble PIV with single mode, correlation planes are computed on a `sum_window` (e.g., 32×32) to capture the full correlation peak. However:

1. **Memory**: Storing full 32×32 planes for thousands of windows consumes significant memory
2. **Fitting speed**: Fitting to 1024 pixels per window is slower than necessary
3. **Edge noise**: FFT correlation edges often contain noise that can degrade fits

The solution: compute on full `sum_window`, but extract only the central `sum_fitting_window` region immediately after correlation.

## Configuration

Add to `config.yaml` under `ensemble_piv`:

```yaml
ensemble_piv:
  sum_window:
  - 32
  - 32
  sum_fitting_window_enabled: true   # Enable central extraction
  sum_fitting_window:                # Size to extract (must be <= sum_window)
  - 16
  - 16
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sum_fitting_window_enabled` | bool | `false` | Enable/disable the feature |
| `sum_fitting_window` | [h, w] | - | Size of central region to extract |

### Validation Rules

- `sum_fitting_window` must be ≤ `sum_window` in both dimensions
- Both values must be positive
- Feature only applies to `single` mode passes (not `std` mode)

## How It Works

### Without `sum_fitting_window` (default)
```
sum_window = [32, 32]
→ Correlate 32×32
→ Store 32×32
→ Fit 32×32
```

### With `sum_fitting_window`
```
sum_window = [32, 32], sum_fitting_window = [16, 16]
→ Correlate 32×32 (full FFT, captures complete peak)
→ Extract central 16×16 immediately in C code
→ Store 16×16 (4× less memory)
→ Fit 16×16 (4× fewer pixels, faster)
```

## Implementation Details

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Config                                                          │
│  ensemble_sum_fitting_window_enabled → bool                     │
│  ensemble_sum_fitting_window → [h,w] or None                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ cpu_ensemble.py                                                 │
│  window_sizes_for_computation = sum_window (FFT size)           │
│  window_sizes_for_corr = fit_window or sum_window (output)      │
│  Buffer allocation uses output size                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PIV_2d_cross_correlate.c :: bulkxcorr2d_accumulate()           │
│  nWindowSize = [32,32]     (FFT computation)                    │
│  nFitWindowSize = [16,16]  (output extraction)                  │
│                                                                 │
│  1. FFT workspace allocated at nWindowSize (32×32)              │
│  2. Output buffer sized at nFitWindowSize (16×16)               │
│  3. Extraction offsets: start_y = (32-16)/2 = 8                 │
│  4. After each FFT correlation, extract central 16×16           │
│  5. Accumulate extracted region to output                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ gaussian_fitting.py                                             │
│  _get_pass_grid() uses fit_window for coordinate grid           │
│  Fitting receives 16×16 planes                                  │
│  Peak positions relative to 16×16 grid                          │
└─────────────────────────────────────────────────────────────────┘
```

### C Code Central Extraction

```c
/* Extraction offsets (centered) */
int start_y = (nWindowSize[0] - out_h) / 2;  // (32-16)/2 = 8
int start_x = (nWindowSize[1] - out_w) / 2;  // (32-16)/2 = 8

/* After FFT correlation, accumulate only central region */
for (i = 0; i < out_h; ++i) {
    for (j = 0; j < out_w; ++j) {
        int src_idx = (start_y + i) * nWindowSize[1] + (start_x + j);
        int dst_idx = i * out_w + j;
        out_ptr[dst_idx] += fCorrelPlane[src_idx];
    }
}
```

### Gaussian Fitting Extraction

The Marquardt-Levenberg fitter in `marquadt_gaussian.c` also extracts a region around the peak for fitting. The `pass_idx` parameter controls extraction behavior:

- **Pass 0**: Find peak location from initial guess, extract around peak
- **Pass > 0**: Extract from center (after image warping, peak should be centered)

## Benefits

| Metric | 32×32 → 16×16 | Improvement |
|--------|---------------|-------------|
| Memory per plane | 1024 → 256 floats | 4× reduction |
| Total correlation storage | 4× smaller | Significant for large grids |
| Fitting pixels | 1024 → 256 | 4× faster fitting |
| Edge noise | Included | Excluded |

## Files Modified

| File | Changes |
|------|---------|
| `pivtools_core/config.py` | Added `ensemble_sum_fitting_window_enabled` and `ensemble_sum_fitting_window` properties |
| `pivtools_cli/lib/PIV_2d_cross_correlate.h` | Added `nFitWindowSize` parameter |
| `pivtools_cli/lib/PIV_2d_cross_correlate.c` | Central extraction in accumulation loop |
| `pivtools_cli/piv/piv_backend/cpu_ensemble.py` | Track computation vs output sizes, pass both to C |
| `pivtools_cli/piv/piv_backend/single_pass_accumulator.py` | Use fit_window for corr_size; background correlations use `bulkxcorr2d_accumulate` for consistent extraction |
| `pivtools_cli/piv/piv_backend/gaussian_fitting.py` | Grid coordinates use fit_window, validation updated |
| `pivtools_cli/lib/marquadt_gaussian.c` | Pass-dependent extraction (peak vs center) |

## Background Subtraction

The ensemble PIV formula requires background subtraction:
```
R_AB = <A⋆B> - <A>⋆<B>
```

With `fit_window` enabled, both the raw correlations and background correlations must use:
1. **Same FFT computation size** (sum_window, e.g., 32×32)
2. **Same central extraction** (fit_window, e.g., 16×16)

The `_correlate_mean_images()` function uses `bulkxcorr2d_accumulate` with N=1 to ensure the background correlation physics matches the raw correlation physics. This guarantees proper subtraction of the DC component.

## Backward Compatibility

- Feature is **disabled by default** (`sum_fitting_window_enabled: false`)
- When disabled, behavior is identical to previous versions
- No changes to stored result formats
- Existing configs work without modification

## Usage Example

```yaml
ensemble_piv:
  window_size:
  - [8, 8]      # Particle window (single mode)
  overlap:
  - 50
  type:
  - single
  sum_window:
  - 32
  - 32
  sum_fitting_window_enabled: true
  sum_fitting_window:
  - 16
  - 16
```

This configuration:
1. Uses 8×8 particle windows embedded in 32×32 sum_window
2. Computes full 32×32 FFT correlations
3. Extracts central 16×16 for storage and fitting
4. Reduces memory by 4× and speeds up fitting
