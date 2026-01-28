# K-Space Transfer Function Fitting for Ensemble PIV

## Overview

This document describes the implementation of k-space (Fourier domain) transfer function fitting as an alternative to physical-space Levenberg-Marquardt Gaussian fitting for Reynolds stress extraction in ensemble PIV.

## Mathematical Foundation

### The Problem with Physical-Space Fitting

In traditional ensemble PIV, we fit a 2D Gaussian to the cross-correlation peak:

```
R_AB(x,y) = A_AB × exp(-[(x-μx)²/2σ²_AB_x + (y-μy)²/2σ²_AB_y])
```

The cross-correlation width σ_AB combines two contributions:
- **Particle image width** (σ_particle): From the optical transfer function
- **Velocity PDF width** (σ_velocity): The Reynolds stress we want to measure

```
σ²_AB = σ²_particle + σ²_velocity
```

To isolate σ²_velocity (the Reynolds stress), we must also fit the autocorrelation R_AA to get σ²_particle, then subtract:

```
UU = σ²_AB - σ²_A
```

This requires fitting **16 parameters** (amplitudes, offsets, widths for R_AA, R_BB, R_AB) and the subtraction amplifies noise.

### The K-Space Insight

In Fourier space, convolution becomes multiplication. The key insight is that **particle shape cancels algebraically**:

```
F(R_AB) = F(particle ⊗ particle) × F(velocity_PDF)
        = |F(particle)|² × T(k)
```

where T(k) is the **transfer function** encoding only velocity PDF parameters.

For autocorrelations:
```
F(R_AA) = |F(particle)|²
F(R_BB) = |F(particle)|²
```

Therefore:
```
T(k) = F(R_AB) / sqrt(F(R_AA) × F(R_BB))
```

The particle contribution cancels exactly, leaving only the velocity PDF information.

### Transfer Function Model

The transfer function for a Gaussian velocity PDF is:

```
T(k) = A × exp(-2π²k^T Σ k) × exp(-2πi k·μ)
       \_____magnitude_____/   \___phase___/
```

Where:
- **μ = (μx, μy)**: Mean displacement
- **Σ**: Covariance matrix (Reynolds stress tensor)
  - Σ_xx = UU stress
  - Σ_yy = VV stress
  - Σ_xy = UV stress
- **A**: Amplitude

This reduces the problem from **16 parameters to 6**.

### Extracting Parameters

**Magnitude** encodes stresses:
```
ln|T(k)| = ln(A) - 2π² k^T Σ k
```

Along principal axes (k_y=0 or k_x=0):
```
ln|T(k_x, 0)| = ln(A) - 2π² Σ_xx k_x²
```

This is a parabola in k² - linear regression on ln|T| vs k² gives Σ directly.

**Phase** encodes displacement:
```
phase(T(k)) = -2π k·μ
```

Linear slope of phase vs k gives mean displacement.

---

## Implementation Details

### File Structure

```
pivtools_cli/piv/piv_backend/
├── kspace_fitting.py      # NEW: K-space fitting module
├── gaussian_fitting.py    # Existing LM Gaussian fitter
└── single_pass_accumulator.py  # Modified: dispatch logic

pivtools_core/
└── config.py              # Modified: new config properties
```

### Core Algorithm (`kspace_fitting.py`)

```python
def fit_windows_kspace(R_AA, R_BB, R_AB, mask_flat, corr_size, config, pass_idx, snr_threshold):
    """
    Main entry point - drop-in replacement for fit_windows_openmp()
    Returns (gauss_flat, status_flat, initial_guess_flat) in 16-element format
    """
```

#### Step 1: FFT with Proper Centering

```python
# Correlation planes have peak at center (index N/2)
# Use ifftshift before FFT to move peak to index 0 for correct phase
F_AA = fftshift(fft2(ifftshift(R_AA_2d)))
F_BB = fftshift(fft2(ifftshift(R_BB_2d)))
F_AB = fftshift(fft2(ifftshift(R_AB_2d)))
```

**Engineering Decision**: Without `ifftshift`, the correlation peak at index N/2 introduces a phase shift of (-1)^n across FFT bins, corrupting phase estimation.

#### Step 2: Compute Reference Spectrum

```python
# Use magnitude for real positive reference
F_ref = np.sqrt(np.abs(F_AA) * np.abs(F_BB))
```

**Engineering Decision**: Using `|F_AA|` instead of `F_AA` avoids issues when F_AA has small negative values at high k (due to Gibbs ringing or noise).

#### Step 3: Compute Transfer Function

```python
epsilon = np.max(np.abs(F_ref)) * 1e-8
T_measured = F_AB / (F_ref + epsilon)
```

#### Step 4: Adaptive K-Bounds

This was the critical engineering challenge. Initial attempts used SNR-based bounds:

```python
k_max = sqrt(ln(SNR) / (2π² σ²))  # Theoretical bound
```

**Problem**: This assumes we know σ (which we're trying to fit) and doesn't account for when F_ref becomes unreliable.

**Solution**: Compute k_max from where |F_ref| drops below 1% of its DC value:

```python
def _compute_kmax_from_profile(k_axis, F_profile, F_dc, threshold_frac=0.01):
    threshold = F_dc * threshold_frac
    below_threshold = F_pos < threshold
    if np.any(below_threshold):
        idx = np.argmax(below_threshold)
        k_max = k_pos[max(0, idx - 1)]
    return np.clip(k_max, 0.05, 0.25)  # Hard cap at 0.25
```

**Why this matters**: At high k, F_ref approaches zero, causing T = F_AB/F_ref to explode. We observed |T| > 1 at k=0.3 (impossible for a proper transfer function). Setting k_max ≈ 0.1-0.25 based on F_ref decay avoids this.

#### Step 5: 1D Initial Estimates (Warm Start)

Extract initial guesses via linear regression along principal axes:

```python
def _fit_1d_axis(T_complex, k_axis, center_idx, k_max, axis):
    # Magnitude fit: ln|T| = -2π² Σ k² (parabola)
    valid_mask_mag = (|k| > 0.01) & (|k| < k_max)
    slope = polyfit(k², ln|T|)[0]
    Sigma = -slope / (2π²)

    # Phase fit: phase(T) = -2π k μ (linear)
    # Use smaller k range to avoid phase wrapping
    valid_mask_phase = (|k| > 0.02) & (|k| < min(k_max, 0.25))
    slope_phase = polyfit(k, phase)[0]
    mu = -slope_phase / (2π)

    return mu, Sigma
```

**Engineering Decision**: Phase estimation uses a smaller k range (< 0.25) than magnitude estimation because phase wrapping causes issues at higher k values.

#### Step 6: Full 6-Parameter Optimization

```python
def _fit_transfer_function_full(T_measured, F_ref, K_X, K_Y, k_max_x, k_max_y, initial_guess):
    # Elliptical mask for valid k-points
    k_mask = (K_X²/k_max_x² + K_Y²/k_max_y²) <= 1.0

    def residual_func(params):
        mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy, A = params

        # Phase term
        phase = -2π(K_X*mu_x + K_Y*mu_y)
        phase_term = exp(1j * phase)

        # Decay term
        quad_form = Sigma_xx*K_X² + 2*Sigma_xy*K_X*K_Y + Sigma_yy*K_Y²
        decay_term = exp(-2π² * quad_form)

        # Model
        T_model = A * decay_term * phase_term

        # Complex residual (real + imag parts)
        diff = T_measured - T_model
        return concatenate([diff.real, diff.imag])

    result = least_squares(residual_func, initial_guess, bounds=bounds)
    return result.x
```

### Output Format Mapping

K-space results are mapped to the existing 16-element format for compatibility:

| Index | Field | K-Space Value |
|-------|-------|---------------|
| 0-2 | amp_A, amp_B, amp_AB | Peak correlation values |
| 3-5 | c_A, c_B, c_AB | 0 (not used) |
| 6-8 | sig_A_x, sig_A_y, sig_A_xy | NaN (particle shape cancels) |
| **9-11** | **sig_AB_x, sig_AB_y, sig_AB_xy** | **Σ_xx, Σ_yy, Σ_xy (stresses)** |
| 12-13 | x0_A, y0_A | Window center |
| **14-15** | **x0_AB, y0_AB** | **center + μ_x, center + μ_y** |

---

## Engineering Decisions Summary

### 1. FFT Centering (`ifftshift` before `fft2`)

**Problem**: Correlation planes have peak at pixel N/2, but FFT expects signal at index 0.

**Impact**: Without this fix, displacement estimates were wrong by ~1.4 pixels and had wrong sign.

**Solution**: Apply `ifftshift` to move peak to index 0 before FFT, then `fftshift` after to center k=0.

### 2. Magnitude-Based Reference (`|F_AA|` not `F_AA`)

**Problem**: F_AA can have small negative values at high k, causing `sqrt(F_AA * F_BB)` to produce complex numbers.

**Solution**: Use `sqrt(|F_AA| * |F_BB|)` for a real positive reference.

### 3. Adaptive K-Bounds from F_ref Decay

**Problem**: At high k, F_ref → 0, causing T = F_AB/F_ref to explode (observed |T| > 1).

**Impact**: Including high-k points with |T| > 1 completely broke the optimizer.

**Solution**: Set k_max where |F_ref| drops to 1% of DC, capped at 0.25.

**Result**: k_max typically ~0.1 for our test data, well within reliable signal region.

### 4. Separate K-Ranges for Phase vs Magnitude

**Problem**: Phase wrapping occurs at higher k values when displacement is large.

**Solution**: Use k < 0.25 for phase estimation vs k < k_max for magnitude.

### 5. Weighted Least Squares for 1D Fits

**Problem**: High-k points have lower SNR and should contribute less.

**Solution**: Weight by |T| to emphasize high-signal regions.

---

## Performance

### Implementation Language

K-space fitting is implemented in **pure Python/NumPy/SciPy**, unlike the Gaussian fitter which uses C with OpenMP parallelization.

**Rationale for Python implementation:**
1. FFTs already use optimized FFTW/MKL libraries via NumPy
2. Linear algebra uses BLAS/LAPACK
3. Easier to debug, modify, and maintain
4. Performance is acceptable for typical PIV workloads
5. Can port to C later if profiling identifies bottleneck

### Timing Benchmarks

Measured on Apple M1 Pro, single-threaded:

| Windows | Correlation Size | K-space (Python) | Per Window |
|---------|------------------|------------------|------------|
| 49 | 32×32 | 53 ms | 1.08 ms |
| 49 | 64×64 | 62 ms | 1.26 ms |
| 100 | 32×32 | 98 ms | 0.98 ms |
| 100 | 64×64 | 135 ms | 1.35 ms |
| 225 | 32×32 | 255 ms | 1.13 ms |
| 225 | 64×64 | 290 ms | 1.29 ms |
| 500 | 32×32 | 473 ms | 0.95 ms |
| 500 | 64×64 | 628 ms | 1.26 ms |

### Comparison with C/OpenMP Gaussian Fitting

| Metric | K-space (Python) | Gaussian (C/OpenMP) |
|--------|------------------|---------------------|
| Per-window time | ~1.0-1.3 ms | ~0.01-0.02 ms |
| Speed ratio | 1× | **50-100× faster** |
| 225 windows | ~255 ms | ~3-5 ms |
| Parallelization | Single-threaded* | OpenMP multi-threaded |

*NumPy FFT and BLAS operations may use multiple cores internally.

### Practical Impact

Despite being slower per-window, k-space fitting has **minimal impact on total PIV processing time**:

```
Typical ensemble PIV run (100 images, 2 passes):
  - Image loading & filtering: ~1-2 s
  - Correlation computation: ~0.5-1 s per pass
  - Gaussian fitting: ~0.01 s per pass
  - K-space fitting: ~0.3 s per pass
  - Total overhead: +0.5-0.6 s for k-space vs Gaussian

Full run: ~3-4 s (Gaussian) vs ~4-5 s (K-space) = ~25% slower overall
```

The 25% increase in total runtime is acceptable given the **40-90% improvement in Reynolds stress accuracy**.

### Optimization Opportunities

If faster k-space fitting is needed:

1. **Vectorize across windows**: Process all windows simultaneously (batch FFT)
2. **GPU acceleration**: CuPy for FFTs, CUDA for optimization
3. **C extension**: Port scipy.optimize.least_squares to custom C code
4. **Reduce iterations**: Use tighter initial bounds from 1D fits
5. **Cython**: JIT compile critical loops

---

## Validation Results

### Synthetic Test (Known Ground Truth)

Test: 64×64 correlation window, Σ_xx = 0.25, μ_x = 0.3

| Parameter | True | Fitted | Error |
|-----------|------|--------|-------|
| μ_x | 0.3000 | 0.3000 | 0.0000 |
| μ_y | 0.0000 | 0.0000 | 0.0000 |
| Σ_xx | 0.2500 | 0.2500 | 0.0000 |
| Σ_yy | 0.2500 | 0.2500 | 0.0000 |
| Σ_xy | 0.0000 | -0.0000 | 0.0000 |

### Full PIV Pipeline Test (100 images, UU=2, VV=3)

| Method | Pass | UU | VV | UU Error | VV Error |
|--------|------|-------|-------|----------|----------|
| Gaussian | 1 | 2.392 | 3.392 | +19.6% | +13.1% |
| Gaussian | 2 | 2.315 | 3.213 | +15.7% | +7.1% |
| **K-space** | **1** | **2.312** | **3.243** | **+15.6%** | **+8.1%** |
| **K-space** | **2** | **2.190** | **3.021** | **+9.5%** | **+0.7%** |

### Improvement Summary

- **UU stress**: K-space reduces error by ~40% (9.5% vs 15.7%)
- **VV stress**: K-space reduces error by ~90% (0.7% vs 7.1%)
- **Displacement**: Both methods achieve sub-0.1 pixel accuracy

---

## Usage

### Configuration

In `config.yaml`:

```yaml
ensemble_piv:
  fit_method: kspace          # 'gaussian' (default) or 'kspace'
  kspace_snr_threshold: 3.0   # SNR threshold for k-bounds
  # ... other settings
```

### Programmatic Access

```python
from pivtools_cli.piv.piv_backend.kspace_fitting import (
    fit_windows_kspace,
    plot_kspace_diagnostic,
)

# Generate diagnostic plot
result = plot_kspace_diagnostic(
    R_AA_2d, R_BB_2d, R_AB_2d,
    true_params={'mu_x': 0.3, 'Sigma_xx': 0.25, ...},
    save_path='kspace_diagnostic.png',
)
```

### Running Comparison Test

```bash
cd PyPIVTools
python tests/run_rs_tests.py kspace
```

---

## Diagnostic Plot

The `plot_kspace_diagnostic()` function generates a 12-panel figure showing:

1. **Row 1**: Physical-space correlation planes (R_AA, R_BB, R_AB) + log|F_ref| with k-bounds
2. **Row 2**: Transfer function |T| and phase(T) + 1D magnitude profiles with fits
3. **Row 3**: Phase profiles + summary panel + linearized ln|T| vs k² plot

Key features:
- Red/white dashed ellipse shows adaptive k-bounds
- Green dashed lines show ground truth (if provided)
- Red solid lines show fitted values

---

## Status Codes

| Code | Description |
|------|-------------|
| -1 | Masked/skipped |
| 0 | Success |
| 1 | Optimization did not converge |
| 2 | SNR too low |
| 3 | Negative variance fitted |
| 4 | Displacement exceeds ½ window |

---

## Limitations and Future Work

### Current Limitations

1. **No particle shape output**: sig_A fields are NaN (by design - particle shape cancels)
2. **Assumes Gaussian velocity PDF**: Non-Gaussian PDFs may require different models
3. **Single-threaded optimization**: scipy.optimize.least_squares is single-threaded

### Potential Improvements

1. **GPU acceleration**: FFTs and optimization could run on GPU
2. **Robust fitting**: Use iteratively reweighted least squares for outlier rejection
3. **Non-Gaussian PDFs**: Extend model to handle skewness/kurtosis
4. **Automatic SNR-based method selection**: Fall back to Gaussian when k-space SNR is low

---

## References

1. Scharnowski, S., & Kähler, C. J. (2016). "Estimation of Reynolds stresses from PIV measurements with single-pixel resolution." *Experiments in Fluids*, 57(5), 1-12.

2. Westerweel, J. (2008). "On velocity gradients in PIV interrogation." *Experiments in Fluids*, 44(5), 831-842.
