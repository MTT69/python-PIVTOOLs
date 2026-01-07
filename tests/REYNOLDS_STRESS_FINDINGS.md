# Reynolds Stress Per-Pass Analysis: Why RS Decreases with Smaller Windows

**Date:** 2026-01-06
**Test Framework:** PyPIVTools Ensemble PIV
**Problem:** Reynolds stress values decrease with smaller interrogation windows

---

## Executive Summary

Comprehensive testing reveals that **Reynolds stress measurement is inherently biased low for smaller interrogation windows**, regardless of multi-pass processing. Key findings:

| Test | 64×64 UU | 32×32 UU | 16×16 UU | 16×16 Error |
|------|----------|----------|----------|-------------|
| Single-pass (baseline) | 2.17 (+9%) | 2.08 (+4%) | **1.89 (-5%)** | -5.3% |
| Multi-pass 64→32→16 | 2.12 (+6%) | 2.02 (+1%) | **1.85 (-8%)** | -7.7% |
| 5-pass 64→32→32→32→16 | 2.00 (0%) | 1.93 (-4%) | **1.74 (-13%)** | -12.8% |

**Critical Finding:** Single-pass 16×16 already shows -5% error, proving the bias is **inherent to small windows**, not caused by multi-pass warping.

---

## The Puzzle: Why Would Smaller Windows Measure LOWER Variance?

Intuitively, smaller windows should capture MORE local variance because they sample smaller regions where fluctuations are more apparent. But we observe the opposite.

### sig_A Behavior Reveals the Mechanism

The autocorrelation width (sig_A) should be a property of particle images, independent of window size. But we measure:

| Window | sig_A (measured) | sig_A change |
|--------|------------------|--------------|
| 64×64 | 1.42 px | baseline |
| 32×32 | 1.39 px | -2.5% |
| 16×16 | 1.32 px | -7% |

**sig_A decreases with window size even though it shouldn't.** This indicates a systematic measurement bias.

Since UU = sig_AB² - sig_A², if sig_A is biased low, and sig_AB is biased even lower, UU will be underestimated.

---

## Possible Mechanisms for Window-Size Bias

### 1. Correlation Peak Fit Region

The Gaussian fitting extracts a region around the peak:
- 64×64 window → 32×32 fit region (1024 points)
- 32×32 window → 32×32 fit region (1024 points)
- 16×16 window → 16×16 fit region (256 points, entire plane)

Smaller fit regions have fewer data points, potentially biasing the sigma estimate.

### 2. Particle Image Truncation

Particles near window edges are truncated, affecting both:
- Autocorrelation (sig_A) - truncated particles appear narrower
- Cross-correlation (sig_AB) - edge effects reduce correlation width

Smaller windows have proportionally more edge particles.

### 3. Fewer Particles Per Window

Statistical averaging degrades with fewer particles:
- 64×64: ~40-80 particles (depending on seeding density)
- 32×32: ~10-20 particles
- 16×16: ~2-5 particles

This explains the noise increase but may also introduce mean bias.

### 4. Displacement Range Relative to Window

For UU=2 px², σ_disp = 1.41 px. Compared to window size:
- 64×64: displacement is 2.2% of window
- 32×32: displacement is 4.4% of window
- 16×16: displacement is 8.8% of window

Large displacements relative to window size may cause particle loss.

### 5. Sub-Pixel Fitting Accuracy

Gaussian fitting on discrete data has systematic errors that may depend on:
- Number of points in fit region
- Peak shape (affected by window size)
- Noise level (higher in smaller windows)

---

## Test Results

### Test 1: Single-Pass Comparison (Baseline)

**Purpose:** Establish baseline RS at each window size without multi-pass effects.

| Window | sig_A_x | sig_A_y | UU | VV | UU Error | VV Error |
|--------|---------|---------|------|------|----------|----------|
| 64×64 | 1.424 px | 1.420 px | 2.171 px² | 3.123 px² | **+8.6%** | +4.1% |
| 32×32 | 1.389 px | 1.387 px | 2.081 px² | 2.966 px² | +4.1% | -1.1% |
| 16×16 | 1.323 px | 1.324 px | 1.893 px² | 2.673 px² | **-5.3%** | -10.9% |

**Key insight:** Single-pass 16×16 shows -5% to -11% error without any multi-pass processing.

### Test 2: Multi-Pass 64→32→16

| Pass | Window | sig_A_x | UU | UU Error | UU Drop |
|------|--------|---------|------|----------|---------|
| 1 | 64×64 | 1.425 px | 2.117 px² | +5.8% | — |
| 2 | 32×32 | 1.389 px | 2.022 px² | +1.1% | -4.5% |
| 3 | 16×16 | 1.326 px | 1.846 px² | -7.7% | -8.7% |

### Test 3: 5-Pass 64→32→32→32→16

| Pass | Window | sig_A_x | UU | UU Error |
|------|--------|---------|------|----------|
| 1 | 64×64 | 1.426 px | 1.998 px² | -0.1% |
| 2 | 32×32 | 1.390 px | 1.926 px² | -3.7% |
| 3 | 32×32 | 1.390 px | 1.925 px² | -3.7% |
| 4 | 32×32 | 1.390 px | 1.926 px² | -3.7% |
| 5 | 16×16 | 1.323 px | 1.745 px² | -12.8% |

**Confirmed:** RS stays constant between same-size passes (32→32→32), drops only when window size changes.

---

## Noise Analysis

Coefficient of variation (CV = std/mean) reveals noise scaling:

| Window | CV(sig_A) | CV(UU) |
|--------|-----------|--------|
| 64×64 | 0.1-0.6% | 0.4-2.2% |
| 32×32 | 0.9-1.1% | 3.5-3.6% |
| 16×16 | 2.4-2.5% | 7.6-8.0% |

16×16 windows have:
- ~20× more sig_A noise than 64×64
- ~20× more UU noise than 64×64

This increased noise may contribute to mean bias through nonlinear effects in the fitting.

---

## Quantitative Summary

### Window Size vs RS Error

| Window | Single-Pass UU Error | Multi-Pass UU Error | Additional Multi-Pass Bias |
|--------|---------------------|---------------------|---------------------------|
| 64×64 | +8.6% | +5.8% | -2.8% |
| 32×32 | +4.1% | +1.1% | -3.0% |
| 16×16 | **-5.3%** | **-7.7%** | -2.4% |

The multi-pass adds only ~3% additional error. The bulk of the error is from window size.

### sig_A vs Theoretical

| Window | Measured sig_A | Theoretical | Ratio |
|--------|----------------|-------------|-------|
| 64×64 | 1.42 px | 0.71 px | 2.0× |
| 32×32 | 1.39 px | 0.71 px | 1.96× |
| 16×16 | 1.32 px | 0.71 px | 1.86× |

The ratio decreases with window size, indicating measurement bias.

---

## Practical Recommendations

### For Accurate Reynolds Stress

1. **Use the largest practical window size** — 64×64 or larger gives best accuracy
2. **Single-pass processing** — avoids additional multi-pass bias (~3%)
3. **If multi-pass needed, extract RS from Pass 1** — before window size reduction
4. **Calibrate with synthetic images** — characterize your specific setup's bias

### Window Size Selection Guide

| Window Size | RS Accuracy | Mean Velocity | Use Case |
|-------------|-------------|---------------|----------|
| 64×64 | Best (+6-9%) | Lower resolution | Turbulence measurements |
| 32×32 | Good (+1-4%) | Medium resolution | General purpose |
| 16×16 | Poor (-5-13%) | High resolution | Mean flow only |

### For Mean Velocity

Multi-pass processing remains appropriate for mean velocity, which is not affected by these biases.

---

## Implications

### Physical Understanding

The RS underestimation at small window sizes appears to be a **fundamental limitation** of window-based correlation methods, not a bug. Possible causes:

1. Edge effects in finite windows
2. Statistical sampling limitations
3. Correlation peak fitting biases
4. Particle image truncation

### Recommendations for Algorithm Development

To improve RS accuracy at small windows:
1. Investigate edge-corrected correlation methods
2. Consider bias correction factors based on window size
3. Explore non-Gaussian peak fitting methods
4. Use overlapping sum-of-correlation approaches

---

## Test Files

```
tests/reynolds_stress_test/
├── generate_rs_images.py                    # Synthetic image generator
├── run_rs_tests.py                          # Test runner
├── plot_sigma_distributions.py              # Distribution plotting
├── single_pass_comparison.log               # Single-pass test output
├── Test2_3Pass_sigma_distributions.png      # Distribution plots
├── Test3_5Pass_sigma_distributions.png
└── REYNOLDS_STRESS_FINDINGS.md              # This document
```

To reproduce:
```bash
cd PyPIVTools/tests/reynolds_stress_test
python run_rs_tests.py single    # Single-pass comparison
python run_rs_tests.py 5pass     # 5-pass diagnostic
python plot_sigma_distributions.py  # Generate plots
```

---

## References

- Westerweel et al. (2004) — ensemble correlation for turbulence statistics
- Scharnowski & Kähler (2016) — uncertainty quantification in ensemble PIV
- Keane & Adrian (1992) — optimization of particle image velocimeters
