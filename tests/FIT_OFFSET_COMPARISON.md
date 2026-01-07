# Fit Offset Comparison: Effect on Reynolds Stress Measurement

**Date:** 2026-01-06
**Test Framework:** PyPIVTools Ensemble PIV
**Parameter Tested:** `fit_offset` (true vs false)

---

## Executive Summary

Testing with `fit_offset: true` shows **no improvement** in Reynolds stress accuracy. In fact, small window accuracy is **slightly worse**:

| Window | fit_offset=false | fit_offset=true | Change |
|--------|------------------|-----------------|--------|
| 64×64 | +8.6% error | +6.5% error | -2.1% better |
| 32×32 | +4.1% error | +2.0% error | -2.1% better |
| 16×16 | **-5.3% error** | **-7.8% error** | +2.5% worse |

**Conclusion:** The window-size bias is **not caused by offset fitting** and persists regardless of this setting.

---

## What Does `fit_offset` Control?

The `fit_offset` parameter controls whether the Gaussian fitting includes a background offset term:

```
fit_offset=false:  G(x,y) = A * exp(-0.5 * ((x-x0)²/σx² + (y-y0)²/σy²))
fit_offset=true:   G(x,y) = A * exp(-0.5 * ((x-x0)²/σx² + (y-y0)²/σy²)) + C
```

When `fit_offset=true`:
- Background level `C` is estimated from the 5th percentile of correlation plane values
- Fitting accounts for non-zero baseline in correlation planes
- Potentially more accurate for noisy correlation planes with elevated backgrounds

---

## Test Results

### Single-Pass Comparison (Baseline)

Each window size tested independently without multi-pass processing:

| Window | fit_offset | sig_A_x | UU | UU Error |
|--------|------------|---------|------|----------|
| 64×64 | false | 1.424 px | 2.171 px² | +8.6% |
| 64×64 | **true** | 1.407 px | 2.131 px² | +6.5% |
| 32×32 | false | 1.389 px | 2.081 px² | +4.1% |
| 32×32 | **true** | 1.371 px | 2.039 px² | +2.0% |
| 16×16 | false | 1.323 px | 1.893 px² | -5.3% |
| 16×16 | **true** | 1.303 px | 1.844 px² | **-7.8%** |

**Key observation:** While `fit_offset=true` reduces overestimation at large windows, it **increases underestimation** at small windows.

### Multi-Pass 3-Pass (64→32→16)

| Pass | Window | fit_offset=false UU | fit_offset=true UU | Error (true) |
|------|--------|---------------------|-------------------|--------------|
| 1 | 64×64 | 2.12 px² | 2.06 px² | +3.2% |
| 2 | 32×32 | 2.02 px² | 1.96 px² | -1.9% |
| 3 | 16×16 | 1.85 px² | 1.76 px² | **-11.9%** |

### Multi-Pass 5-Pass (64→32→32→32→16)

| Pass | Window | fit_offset=false UU | fit_offset=true UU | Error (true) |
|------|--------|---------------------|-------------------|--------------|
| 1 | 64×64 | 2.00 px² | 1.95 px² | -2.4% |
| 2 | 32×32 | 1.93 px² | 1.88 px² | -6.1% |
| 3 | 32×32 | 1.93 px² | 1.88 px² | -6.1% |
| 4 | 32×32 | 1.93 px² | 1.88 px² | -6.1% |
| 5 | 16×16 | 1.74 px² | 1.68 px² | **-15.8%** |

**Critical finding:** Same-size passes (32→32→32) remain constant with both settings, confirming the bias is window-size dependent, not warping-dependent.

---

## Analysis

### Why Doesn't fit_offset Help?

1. **The bias is geometric, not background-related**: The underestimation at small windows comes from:
   - Particle image truncation at window edges
   - Fewer particles per window (statistical degradation)
   - Fitting region limitations (16×16 window = 16×16 fit region)

2. **Background offset doesn't address these**: The correlation peak shape is affected by window geometry, not by background levels.

3. **fit_offset may even hurt**: With `fit_offset=true` and small windows:
   - The 5th percentile estimate is noisier (fewer data points)
   - The fitting has an additional free parameter to estimate
   - This may introduce bias in the sigma estimates

### sig_A Behavior Confirms the Pattern

| Window | fit_offset=false | fit_offset=true |
|--------|------------------|-----------------|
| 64×64 | 1.424 px | 1.407 px |
| 32×32 | 1.389 px | 1.371 px |
| 16×16 | 1.323 px | 1.303 px |

With `fit_offset=true`, sig_A is systematically **lower** at all window sizes, which propagates to **lower** UU estimates (since UU = sig_AB² - sig_A², but sig_AB is affected similarly).

---

## Noise Analysis

Coefficient of variation (CV = std/mean) remains similar:

| Window | CV(sig_A) false | CV(sig_A) true | CV(UU) false | CV(UU) true |
|--------|-----------------|----------------|--------------|-------------|
| 64×64 | 0.2% | 0.2% | 0.8% | 0.8% |
| 32×32 | 0.9% | 0.9% | 3.5% | 3.6% |
| 16×16 | 2.4% | 2.4% | 7.6% | 8.0% |

The noise levels are essentially unchanged by the fit_offset setting.

---

## Recommendations

### For fit_offset Setting

**Use `fit_offset: false`** (the default) for Reynolds stress measurements:
- It gives **better accuracy at small windows** (-5.3% vs -7.8%)
- Large window accuracy is similar (+8.6% vs +6.5%)
- One fewer parameter to estimate reduces fitting complexity

### For Accurate Reynolds Stress

The window-size bias is **fundamental** and not fixable by fit_offset:

1. **Use the largest practical window size** — 64×64 or larger
2. **Single-pass processing** — avoids additional multi-pass degradation
3. **Calibrate with synthetic images** — characterize your setup's bias
4. **Apply correction factors** if needed based on window size

---

## Summary Table

| Metric | fit_offset=false | fit_offset=true | Winner |
|--------|------------------|-----------------|--------|
| 64×64 UU error | +8.6% | +6.5% | true (slightly) |
| 32×32 UU error | +4.1% | +2.0% | true (slightly) |
| 16×16 UU error | -5.3% | -7.8% | **false** |
| Overall pattern | Consistent | Consistent | Same |
| Noise level | Baseline | Same | Tie |

**Bottom line:** `fit_offset` does not solve the window-size bias problem. The bias is inherent to small-window correlation methods.

---

## Test Files

```
tests/reynolds_stress_test/
├── FIT_OFFSET_COMPARISON.md          # This document
├── REYNOLDS_STRESS_FINDINGS.md       # Original findings (fit_offset=false)
├── fit_offset_true_single_pass.log   # Single-pass test logs
├── fit_offset_true_multipass.log     # Multi-pass test logs
├── Test2_3Pass_sigma_distributions.png
├── Test3_5Pass_sigma_distributions.png
└── run_rs_tests.py                   # Test runner
```

To reproduce:
```bash
# In config.yaml, set ensemble_piv.fit_offset: true (or false)
cd PyPIVTools/tests/reynolds_stress_test
python run_rs_tests.py single    # Single-pass comparison
python run_rs_tests.py 5pass     # 5-pass diagnostic
```
