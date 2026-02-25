# Noise-Aware K-Space Fitting (v3) — Design Document

> Refined after experimental validation on channel flow DNS data with synthetic camera noise.
> The original v2 plan proposed a 3-parameter Stage 1 fit — testing showed this suffers from
> N-sigma_A degeneracy. The actual fix is much simpler: corner-based noise floor estimation
> and subtraction from F_ref, applied inside the existing k-space fitter on all passes.

---

## 1. Problem Statement

Camera noise adds a flat power spectral density to auto-correlations but NOT cross-correlations:

```
F_ref_noisy = sqrt(|F_AA + sigma_n^2| * |F_BB + sigma_n^2|)
            ≈ F_ref_clean + N    (at high k where particle signal decays)
F_AB_noisy  = F_AB_clean          (cross-correlation unaffected by uncorrelated noise)
```

This inflated F_ref biases the transfer function `T(k) = F_AB / F_ref`:
- T(k) is suppressed → fitted Sigma (Reynolds stresses) are overestimated
- SNR estimation, k_max bounds, and soft-weighting are all computed from the noisy F_ref
- The predictor field extracted from each pass carries this bias forward

**Observed impact** (4000-image channel flow, sigma_n = 2% of particle peak):
- Sigma_xx (u'u') overpredicted by ~20%
- Sigma_yy (v'v') overpredicted by ~60-100% (noise adds constant delta; larger fraction of small v'v')
- Sigma_xy (u'v') unaffected (isotropic noise doesn't rotate stress ellipse)

**Key finding from pass 1 vs pass 4 comparison:**
- Pass 1 (no predictor): noise correction recovers 60% of v'v' overprediction
- Pass 4 (3 rounds of noisy predictor): only 16% recovery — predictor contamination dominates

This means noise correction must happen on ALL passes inside the existing fitter, not as a post-hoc correction on final planes. Each pass gets a better F_ref, extracts a better predictor, and the improvement cascades.

---

## 2. Experimental Findings

### What was tested (using `tools/test_noise_aware_kspace.py`)

Three noise floor estimation strategies were compared on clean and noisy channel flow data:

| Method | Region | Clean N | Noisy N | Clean Sigma_yy bias | Noisy Sigma_yy reduction |
|--------|--------|---------|---------|---------------------|--------------------------|
| **Annular ring** (0.4 < \|k\| < 0.5) | Ring | 0.128 | 0.815 +/- 0.025 | -14.3% (bad) | +38.8% |
| **Corners** (\|k_x\|>0.35 AND \|k_y\|>0.35) | Corners | 0.007 | 0.472 +/- 0.260 | +3.0% (neutral) | +16.8% (pass 4) |
| **Oracle** (clean-noisy F_ref difference) | High k | perfect | 0.544 +/- 0.116 | perfect | +15.9% (pass 4) |

### Key conclusions

1. **The annular ring (current SNR estimator) contaminates clean data.** Inner edge at |k|=0.4 still has significant particle Gaussian tail: `exp(-2*pi^2 * sigma_A * 0.4^2) ≈ 0.04`. This biases N high even on clean data.

2. **Corners give clean estimates.** At |k_x|>0.35 AND |k_y|>0.35, the particle Gaussian has decayed in BOTH directions. Clean data gives N ≈ 0.007 (correct). The autocorrelation is circular — the ring contamination is NOT from anisotropy.

3. **N estimation is NOT the bottleneck.** Oracle N (computed from clean-noisy F_ref difference) gives essentially the same correction as corners on pass 4 (~16% vs ~17%). The remaining gap is from predictor contamination through earlier passes.

4. **The 3-parameter fit (N, sigma_A_x, sigma_A_y) has fatal N-sigma_A degeneracy.** At low SNR the optimizer absorbs noise into sigma_A instead of N, returning N=0 when noise is clearly present. This approach is abandoned.

5. **Pass 1 vs Pass 4 confirms predictor contamination.** Pass 1 noise correction is 3-4x more effective than pass 4. The fix must happen per-pass inside the loop.

6. **sigma_A recovery is unnecessary.** The algebraic cancellation T = F_AB / F_ref already eliminates sigma_A from the transfer function. The 3-5% gradient correction improvement from knowing sigma_A is not worth the degeneracy risk.

### Pass 1 vs Pass 4 results (64x64 and 16x16 windows)

**Pass 1** (64x64 windows, no predictor):
| Method | Sigma_xx reduction | Sigma_yy reduction | N |
|--------|-------|-------|---|
| Clean corners | -2.2% | +4.2% | 0.009 |
| Noisy corners | +21.6% | +60.0% | 0.822 |
| Noisy oracle | +14.6% | +37.5% | 0.707 |

**Pass 4** (16x16 windows, 3 rounds of noisy predictor):
| Method | Sigma_xx reduction | Sigma_yy reduction | N |
|--------|-------|-------|---|
| Clean corners | -4.6% | +3.0% | 0.007 |
| Noisy corners | +8.4% | +16.8% | 0.472 |
| Noisy oracle | +8.6% | +15.9% | 0.544 |

---

## 3. The Fix

### Overview

Corner-based noise floor estimation and subtraction inside `_fit_single_window_kspace()`, **active only on pass 0 (unwarped images)**. On subsequent passes where images have been warped by the predictor, noise subtraction is disabled because bicubic interpolation colors the noise spectrum.

### The change

After computing `F_ref = sqrt(|F_AA| * |F_BB|)`:

```python
if noise_subtraction:  # True only on pass 0
    noise_corners = (np.abs(K_X) > 0.35) & (np.abs(K_Y) > 0.35)
    N_floor = max(float(np.median(F_ref[noise_corners])), 0.0)
    epsilon_floor = F_ref[center_idx_y, center_idx_x] * 1e-8
    F_ref = np.maximum(F_ref - N_floor, epsilon_floor)
else:
    N_floor = 0.0
```

The `noise_subtraction` flag is set by `fit_windows_kspace()`: `noise_subtraction=(pass_idx == 0)`.

### Why only pass 0

When images are warped by `cv2.remap` with bicubic interpolation (symmetric warping: ±Δ/2), the interpolation kernel acts as a **low-pass filter on the noise**. The filter's transfer function H(k) depends on the fractional pixel displacement:

- **Integer displacement** (frac = 0.0): H(k) = 1 everywhere → noise stays white
- **Half-pixel displacement** (frac = 0.5): maximum smoothing → H(k) drops at high k

This creates two problems for corner-based estimation on warped passes:

1. **Systematic underestimation:** Corner pixels sample the high-k region where H²(k) ≈ 0, giving N_corner ≈ 0. But at signal frequencies (low k), H²(k) ≈ 1, so the true noise is ≈ N. We subtract 0 when we should subtract N.

2. **Spatially-varying bias (wavy artifacts):** The fractional pixel part of Δ/2 varies with position (e.g., across y+ in channel flow). Where the predictor displacement crosses integer pixel boundaries, the fractional part oscillates between 0 and 0.5, causing the undercorrection to wave on and off. This produces a one-sided error pattern: stresses touch truth at integer displacements but are never beneath (interpolation can only attenuate noise, never amplify it, so corners always underestimate).

**Experimental evidence (4000-image channel flow, sigma_n = 2%):**
- 64x64 (pass 0, unwarped): vv+ R² = 0.83, uu+ R² = 0.90 — good
- 32x32 (pass 1, warped): vv+ R² = -13.23 — catastrophic wavy artifacts
- Mean velocity unaffected: U+ R² = 0.98 (noise only affects stress extraction)
- Shear stress unaffected: -uv+ R² = 0.93 (isotropic noise doesn't create shear)

### Why corners, not ring (pass 0)

The existing annular ring estimator (lines 311-313) at |k| = 0.4-0.5 has its inner edge contaminated by particle shape tail. The Gaussian particle image has significant power at |k|=0.4:

```
exp(-2*pi^2 * sigma_A * k^2) at k=0.4 ≈ 0.04  (4% of DC, for typical sigma_A ≈ 3)
```

But corners at |k_x|>0.35 AND |k_y|>0.35 require the Gaussian to have decayed in BOTH dimensions:

```
exp(-2*pi^2 * sigma_A * (0.35^2 + 0.35^2)) ≈ 0.002  (0.2% of DC)
```

This is 20x cleaner. The tradeoff: fewer pixels in corners (~12% of grid) vs ring (~15%), giving slightly higher variance. But the bias elimination is far more important than variance.

### What stays unchanged

Everything else in the k-space fitter is preserved:
- DC normalization T_norm = T/T(0)
- Physical-space peak detection for displacement initialization
- Soft anisotropic weighting (computed from noise-corrected F_ref on pass 0)
- k_max cascade (from noise-corrected F_ref profiles on pass 0)
- 5-parameter TRF fit for (mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy)
- 16-element output format
- SNR gate (more accurate with clean F_ref on pass 0)
- Status codes 0-5
- sigma_A fields remain NaN (no sigma_A fitting)

### How pass 0 correction cascades

The noise-corrected pass 0 produces a better predictor. Even though passes 1+ don't subtract noise, they benefit from the improved predictor:

```
Pass 0:  correlate unwarped → N subtraction → better stresses → better predictor
Pass 1:  warp with better predictor → correlate → fit (no N subtraction) → improved predictor
  ...
Pass N:  final result benefits from cascade of improved predictors from pass 0
```

No changes to the multi-pass loop, accumulator, correlator, or any other file.

---

## 4. Ideas Evaluated (Updated)

### From the current k-space implementation
| ID | Idea | Status | Reason |
|----|------|--------|--------|
| A1 | Transfer function T = F_AB / F_ref (algebraic particle cancellation) | KEEP | Fundamental advantage |
| A2 | DC normalization T_norm = T/T(0) (eliminates amplitude) | KEEP | Reduces params 6 → 5 |
| A3 | Anisotropic soft weighting w_snr * w_soft | KEEP (improved) | Now uses F_ref_clean → wider bandwidth, more accurate |
| A4 | Adaptive k_max from F_ref 1% decay | KEEP (improved) | Now from F_ref_clean → wider bounds when noise removed |
| A5 | SNR estimation from high-k region | KEEP (improved) | Now from F_ref_clean → more accurate SNR |
| A6 | 1D log-magnitude regression for initial Sigma | KEEP (improved) | Noise-corrected F_ref gives better regression |
| A7 | Physical-space peak for initial displacement | KEEP | Immune to phase wrapping |
| A9 | 5-parameter TRF fit | KEEP | Core fit unchanged |
| A10 | Window-level SNR rejection gate | KEEP | Now more accurate with clean F_ref |

### From the proposed noise floor approach
| ID | Idea | Status | Reason |
|----|------|--------|--------|
| B1 | Model noise floor N as free parameter (3-param fit) | ABANDONED | Fatal N-sigma_A degeneracy at low SNR |
| B2 | Fit F_AB = (S_AA - N) * T directly | NOT NEEDED | Simple subtraction sufficient |
| B3 | Data-derived weights from F_ref_clean | PARTIALLY KEPT | Existing weights improved by using F_ref_clean |
| B4 | Use all wavenumbers | NOT NEEDED | Existing k_max cascade works better with clean F_ref |
| B5 | N as diagnostic | OPTIONAL | Can log N_floor per window if desired |
| B6 | sigma_A recovery from Stage 1 | ABANDONED | Degeneracy risk, algebraic cancellation makes it unnecessary |

### New approach (validated)
| ID | Idea | Status | Reason |
|----|------|--------|--------|
| E1 | Corner-based N estimation (|k_x|>0.35 AND |k_y|>0.35) | ADOPTED | Clean on noiseless data (N=0.007), no ring contamination |
| E2 | Simple subtraction F_ref_clean = F_ref - N | ADOPTED | Sufficient, no fitting needed |
| E3 | Per-pass correction inside existing fitter | REVISED → pass 0 only | Warped passes have colored noise from interpolation; corner estimation unreliable |
| E4 | Pass 0 only: noise subtraction gated by pass_idx==0 | ADOPTED | Interpolation colors noise on warped passes; pass 0 cascade provides most benefit |

---

## 5. Pros and Cons

### PROS

**P1. Fixes Reynolds stress overprediction from camera noise (HIGH impact)**
Primary goal. Pass 0 noise correction produces a better predictor that cascades through all subsequent passes. 64x64 single-pass: vv+ R² = 0.83 (vs negative without correction).

**P2. Zero architectural changes (HIGH impact)**
Modification is entirely inside `_fit_single_window_kspace()` + `fit_windows_kspace()`. No changes to the multi-pass loop, accumulator, correlator, save format, or any other file.

**P3. Improves all downstream quantities on pass 0 (MEDIUM impact)**
SNR estimation, k_max bounds, soft weighting, and 1D regressions all benefit from noise-corrected F_ref on the first pass. Better predictor cascades improvements.

**P4. Negligible computational cost (LOW impact)**
Corner median: ~5 microseconds per window on pass 0 only. No iterative fitting, no optimization.

**P5. No new tuning parameters**
Corner threshold 0.35 is derived from particle physics (Gaussian decay), not arbitrary. `snr_threshold` remains the only user-facing parameter.

**P6. Safe on warped passes (HIGH impact)**
By disabling noise subtraction on passes > 0, avoids the wavy stress artifacts caused by interpolation-colored noise. Previous version applied subtraction on all passes and produced vv+ R² = -13.23 at 32x32.

### CONS

**C1. Corner estimation has higher variance than ring (LOW risk)**
~12% of k-grid pixels vs ~15% for ring. Variance ~0.260 vs ~0.025 on our test data. But the bias elimination (N=0.007 vs 0.128 on clean data) is far more important.

**C2. Assumes white noise floor (LOW risk)**
N is a scalar (flat PSD). Valid for camera read noise + shot noise. Could be violated by aggressive image preprocessing that creates colored noise, but ensemble-averaged correlation planes smooth this out.

**C3. Does not recover sigma_A (ACCEPTED tradeoff)**
The 3-5% gradient correction improvement from sigma_A is sacrificed to avoid the N-sigma_A degeneracy. The window averaging term (L^2/12) provides 95-97% of the gradient correction.

**C4. Corners may be sparse for small correlation windows (LOW risk)**
For 16x16 correlation windows, corners at |k_x|>0.35 AND |k_y|>0.35 give ~25 pixels. Median of 25 values is still robust. For 8x8, only ~6 pixels — may need fallback to ring or wider corners. Not a concern for typical ensemble PIV (32x32 or larger sum windows).

**C5. No direct noise correction on warped passes (ACCEPTED tradeoff)**
Passes > 0 (warped images) do not get noise subtraction. The pass 0 correction cascades through the predictor, providing indirect improvement. Applying corner-based subtraction on warped passes was experimentally shown to cause wavy stress artifacts (vv+ R² = -13.23 at 32x32) due to interpolation-colored noise.

**C6. Interpolation colors noise (FUNDAMENTAL LIMITATION)**
Bicubic image warping acts as a low-pass filter on camera noise. The filter's transfer function depends on the fractional pixel displacement, which varies spatially. This makes any frequency-domain noise estimation unreliable on warped passes. A future improvement could estimate H(k) from the local displacement and correct accordingly, but the complexity is not currently justified.

---

## 6. Implementation Plan

### Single file modified: `pivtools_cli/piv/piv_backend/kspace_fitting.py`

**A. Add `noise_subtraction` parameter to `_fit_single_window_kspace()`**

New kwarg `noise_subtraction: bool = True`. Noise estimation and subtraction wrapped in conditional:

```python
if noise_subtraction:
    noise_corners = (np.abs(K_X) > 0.35) & (np.abs(K_Y) > 0.35)
    N_floor = max(float(np.median(F_ref[noise_corners])), 0.0)
    epsilon_floor = F_ref[center_idx_y, center_idx_x] * 1e-8
    F_ref = np.maximum(F_ref - N_floor, epsilon_floor)
else:
    N_floor = 0.0
```

**B. Gate by pass_idx in `fit_windows_kspace()`**

Call site passes `noise_subtraction=(pass_idx == 0)` so only the first (unwarped) pass gets correction.

**C. N_floor in diagnostics**

When `return_diagnostics=True`, N_floor is included in the diagnostics dict (0.0 on passes > 0).

**D. No changes to:**
- `_fit_transfer_function_full()` (unchanged)
- `_fit_1d_axis()` (unchanged)
- `_build_params_from_fit()` (unchanged)
- `_compute_kmax_from_profile()` (unchanged)
- `single_pass_accumulator.py` (unchanged)
- Any other file in the codebase

### Config

Optional new config key (backward compatible, defaults to current behavior):
```yaml
ensemble_piv:
  kspace_noise_subtraction: true   # default true when fit_method is kspace
```

---

## 7. Verification Strategy

1. **Clean data neutrality:** Run on noiseless data, verify N ≈ 0 and stresses unchanged within 5%
2. **Noisy data improvement:** Run on noisy data, verify Sigma_yy overprediction reduced
3. **All-pass cascade:** Compare pass-by-pass stress evolution with and without noise correction
4. **Regression:** Ensure no degradation on existing validation datasets
5. **Edge cases:** Low SNR windows, small correlation windows (16x16), highly anisotropic turbulence

---

## 8. Summary

The noise-aware k-space fitting is a minimal, targeted fix: estimate noise floor N from corners of k-space, subtract from F_ref. Applied per-window inside the existing fitter, it automatically corrects all passes in the multi-pass loop. The original v2 plan's 3-parameter Stage 1 fit was experimentally shown to suffer from N-sigma_A degeneracy and is abandoned. The simpler corner-based approach is bias-free on clean data, requires no new tuning parameters, and confines all changes to ~5 lines inside `_fit_single_window_kspace()`.
