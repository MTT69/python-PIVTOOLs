# Noise-Aware K-Space Fitting (v2) — Design Document

> Full analysis of the proposed improvement to k-space fitting for ensemble PIV Reynolds stress extraction.
> Combines the current transfer function approach with noise floor modeling.

---

## 1. Problem Statement

The current k-space fitting (`kspace_fitting.py`) works well but has one acknowledged weakness: **k_max selection**. It chains three independent heuristics into a triple-minimum:

```python
k_max_x = min(_compute_kmax(Sigma_xx_init, snr), k_max_x_from_profile, k_max_limit)
```

- `_compute_kmax_from_profile`: where F_ref drops to 1% of DC — why 1%? Not 0.5%? Not 2%?
- `_compute_kmax(Sigma, snr)`: depends on initial Sigma estimate (circular: Sigma from 1D fit within k_max, then k_max from Sigma)
- `k_max_limit`: hardcoded 0.25 or 0.35 depending on soft weighting mode

Because of the `min()`, the most conservative heuristic always wins. If any one is overly restrictive, we lose bandwidth. The weighting `w_snr * w_soft` further depends on initial Sigma estimates, adding another circularity layer. Only ~10% of spectral data is used.

**The suggestion:** instead of avoiding noise-dominated regions, model the noise floor N explicitly as a fit parameter.

---

## 2. Ideas Evaluated

### From the current k-space implementation
| ID | Idea | Keep? | Reason |
|----|------|-------|--------|
| A1 | Transfer function T = F_AB / F_ref (algebraic particle cancellation) | YES | Fundamental advantage over Gaussian |
| A2 | DC normalization T_norm = T/T(0) (eliminates amplitude) | YES | Reduces params 6 → 5, handles illumination mismatch |
| A3 | Anisotropic soft weighting w_snr * w_soft | REPLACE | Circular (depends on Sigma_init), replaced by data-derived weights |
| A4 | Adaptive k_max from F_ref 1% decay | ELIMINATE | Replaced by noise floor model |
| A5 | SNR-based k_max | ELIMINATE | Replaced by noise floor model |
| A6 | 1D log-magnitude regression for initial Sigma | IMPROVE | Use noise-corrected F_ref for wider bandwidth |
| A7 | Physical-space peak for initial displacement | YES | Immune to phase wrapping, works for large displacements |
| A8 | Elliptical k-mask | ELIMINATE | Not needed with data-derived weights |
| A9 | 5-parameter TRF fit | YES | Core fit unchanged |
| A10 | Window-level SNR rejection gate | YES | Fast pre-screening, keeps bad windows out |

### From the proposed noise floor approach
| ID | Idea | Keep? | Reason |
|----|------|-------|--------|
| B1 | Model noise floor N as free parameter | YES (in Stage 1) | Eliminates k_max, physically meaningful |
| B2 | Fit F_AB = (S_AA - N) * T directly | NO | Dynamic range problem (5-7 orders); use normalized formulation instead |
| B3 | Self-weighting from (S_AA - N) | YES (as w = F_ref_clean / DC) | Non-circular, data-derived, naturally anisotropic |
| B4 | Use all wavenumbers | YES | With weights, high-k data contributes minimally but constrains N |
| B5 | N as diagnostic | YES | Per-window noise mapping, ensemble convergence metric |
| B6 | sigma_A recovery from noise-corrected reference | YES | Enables full gradient correction (currently missing ~3-6%) |

### From the C Gaussian fitter (ideas to port)
| ID | Idea | Keep? | Reason |
|----|------|-------|--------|
| C1 | Stacked model (simultaneous AA, BB, AB fit) | NO | K-space cancellation makes this unnecessary |
| C2 | Delta parameterization | NO | Not applicable (k-space fits Sigma directly) |
| C3 | Constraint-aware Jacobian | NO | TRF bounds handle this |
| C4 | sigma_A estimation | YES (via Stage 1) | Enables gradient correction |

### New ideas from combining approaches
| ID | Idea | Keep? | Reason |
|----|------|-------|--------|
| D1 | Two-stage fit: Stage 1 noise+particle, Stage 2 transfer function | YES | Decouples N from Sigma, avoids degeneracy |
| D2 | Data-derived weights: w = F_ref_clean / F_ref_clean(0) | YES | Non-circular, anisotropic, replaces all heuristics |
| D3 | Output sigma_A in params[6:8] for gradient correction | YES | Backward compatible, zero downstream changes |

---

## 3. The Optimal Method

### Overview

A **two-stage hybrid** that fits the noise floor from F_ref alone (Stage 1), then uses the noise-corrected F_ref as data-derived weights for the transfer function fit (Stage 2).

### Stage 0: FFT and Setup (unchanged)

```
F_AA = fftshift(fft2(ifftshift(R_AA)))
F_BB = fftshift(fft2(ifftshift(R_BB)))
F_AB = fftshift(fft2(ifftshift(R_AB)))
F_ref = sqrt(|F_AA| * |F_BB|)          # real positive reference
```

SNR gate: `snr = DC_power / noise_power`. Reject if `snr < snr_threshold` (status 2).

### Stage 1: Noise Floor + Particle Shape (3 parameters)

**Model** (real-valued, positive):
```
F_ref_model(k) = (F_ref(0) - N) * exp(-2*pi^2 * (sigma_A_x * k_x^2 + sigma_A_y * k_y^2)) + N
```

**Physical interpretation:**
- `(F_ref(0) - N)` = signal-only DC magnitude (particle power at zero frequency)
- `exp(-2*pi^2 * sigma_A * k^2)` = particle autocorrelation spectral decay
- `+ N` = flat white noise floor from camera read noise + shot noise

**Parameters:** N, sigma_A_x, sigma_A_y (3 total)

**Initialization:**
- N_init = median(F_ref in annular ring 0.4 < |k| < 0.5) — already computed
- sigma_A from 1D regression: `ln(F_ref(k_x, 0) - N_init) = const - 2*pi^2 * sigma_A_x * k_x^2`

**Bounds:** N in [0, percentile_75(F_ref at high k)], sigma_A in [0.1, 50]

**Solver:** `least_squares` with normalized residual `(F_ref - F_ref_model) / F_ref(0)`, TRF, max 100 evals. Very fast: 3 params, real data, smooth Gaussian.

**Output:** N_fitted, sigma_A_x, sigma_A_y, F_ref_clean = max(F_ref - N_fitted, epsilon)

**Fallback on failure:** N from annular ring median, sigma_A = 3.0

### Stage 2a: Initial Guesses (improved)

**Displacement:** Physical-space cross-correlation peak with 3-point Gaussian sub-pixel refinement (unchanged — immune to phase wrapping).

**Variance:** 1D log-magnitude regression, but now using noise-corrected reference:
```
ln|T(k_x, 0)| = ln|F_AB(k_x, 0)| - ln(F_ref_clean(k_x, 0))
               = -2*pi^2 * Sigma_xx * k_x^2
```

Key improvement: since F_ref_clean has noise subtracted, the regression extends to higher k without noise floor bias. Use all k where `F_ref_clean > 1% of F_ref_clean(0)`.

### Stage 2b: Full 5-Parameter Nonlinear Fit

**Noise-corrected transfer function:**
```
T_norm(k) = [F_AB(k) / F_ref_clean(k)] / [F_AB(0) / F_ref_clean(0)]
```

**Model (unchanged):**
```
T_model(k) = exp(-2*pi^2 * k^T * Sigma * k) * exp(-2*pi*i * k . mu)
```

**Data-derived weights:**
```
w(k) = F_ref_clean(k) / F_ref_clean(0)
```

Properties of these weights:
- **Non-circular:** depend on data + Stage 1 N, never on Sigma being fitted
- **Naturally anisotropic:** if particle is elongated, weights reflect real F_ref shape (not assumed from Sigma_init)
- **Self-attenuating:** at high k where F_ref ~ N, F_ref_clean ~ epsilon, so w ~ 0. No hard cutoff.
- **Bounded:** always in [0, 1] since F_ref_clean(k) <= F_ref_clean(0)
- **No k_mask needed:** fit over all wavenumbers

**Residual:**
```python
def residual(params):
    mu_x, mu_y, Sigma_xx, Sigma_yy, Sigma_xy = params
    phase = -2*pi * (K_X * mu_x + K_Y * mu_y)
    quad = Sigma_xx * K_X^2 + 2*Sigma_xy * K_X * K_Y + Sigma_yy * K_Y^2
    T_model = exp(-2*pi^2 * quad) * exp(1j * phase)
    diff = w * (T_norm - T_model)
    return [diff.real, diff.imag]
```

**Bounds (unchanged):** mu in [-0.75*window, 0.75*window], Sigma_xx/yy >= 0, Sigma_xy in [-50, 50]

**Solver:** `scipy.optimize.least_squares`, TRF, ftol=xtol=1e-8, max_nfev=250.

### Output Format (16-element, backward compatible)

```
params[0:3]  = amp_A, amp_B, amp_AB                    # amplitudes (unchanged)
params[3:6]  = 0.0, 0.0, 0.0                           # offsets (unused)
params[6]    = sigma_A_x                                # particle variance x (NEW — was NaN)
params[7]    = sigma_A_y                                # particle variance y (NEW — was NaN)
params[8]    = 0.0                                      # particle cross-term (zero, isotropic rotation)
params[9]    = sigma_A_x + Sigma_xx                     # sig_AB_x (NEW format)
params[10]   = sigma_A_y + Sigma_yy                     # sig_AB_y (NEW format)
params[11]   = 0.0 + Sigma_xy                           # sig_AB_xy (unchanged in effect)
params[12:14] = center_x, center_y                      # position A (unchanged)
params[14:16] = center_x + mu_x, center_y + mu_y       # position AB (unchanged)
```

**Why this is backward compatible:**

The accumulator (`single_pass_accumulator.py:888`) computes:
```python
UU_stress = max(sig_AB_x - sig_A_x, 0) = max((sigma_A_x + Sigma_xx) - sigma_A_x, 0) = Sigma_xx
```

Same Reynolds stress, but now `sig_A_x = sigma_A_x` flows into gradient correction:
```python
# gradient_correction.py — now gets actual sigma_A instead of 0
Corr_uu = (du/dx)^2 * (L_x^2/12 + sig_A_x) + (du/dy)^2 * (L_y^2/12 + sig_A_y)
```

---

## 4. What Gets Eliminated vs Preserved

### Eliminated
1. `_compute_kmax_from_profile()` — F_ref 1% decay threshold
2. `_compute_kmax(sigma, snr)` — sigma-based k_max from SNR
3. Triple-minimum cascade: `k_max = min(decay, sigma-based, hard_cap)`
4. Hardcoded caps: 0.25 (hard), 0.35 (soft)
5. Elliptical k_mask: `K_X^2/k_max_x^2 + K_Y^2/k_max_y^2 <= 1`
6. Circular w_soft: `exp(-K_X^2/k0_x^2 - K_Y^2/k0_y^2)` (depends on Sigma_init)

**All replaced by:** `w(k) = F_ref_clean(k) / F_ref_clean(0)` — one line, data-derived.

### Preserved
1. DC normalization (T_norm = T/T(0), eliminates amplitude ambiguity)
2. Phase-magnitude decoupling (mu from peak, Sigma from log-magnitude)
3. Algebraic particle cancellation via F_ref
4. Geometric mean reference for two-camera case
5. SNR gate (fast pre-screening)
6. Status codes 0-5 (unchanged)
7. 16-element output format (full backward compatibility)
8. Physical-space peak detection for displacement init (immune to wrapping)

---

## 5. Detailed Pros and Cons

### PROS

**P1. Eliminates k_max heuristic cascade (HIGH impact)**
The most sensitive tuning parameter in the current method is gone. Three heuristics with arbitrary thresholds and a circular dependency are replaced by a physically motivated noise floor estimate.

**P2. Non-circular weighting (HIGH impact)**
Current w_soft depends on Sigma_init (which depends on k_max, which depends on Sigma_init). New weights depend only on the data and Stage 1 N. No circularity.

**P3. Uses all spectral data (MEDIUM impact)**
Current: ~10% of k-space within elliptical mask (~370 points for 64x64).
New: 100% (~4096 points, but high-k points contribute minimally through weights).
More data = better conditioning, more information extracted.

**P4. sigma_A recovery enables full gradient correction (MEDIUM impact)**
Currently missing ~3-6% of gradient correction (particle extent term). New method provides sigma_A from Stage 1 at no extra cost.

**P5. N as diagnostic (LOW-MEDIUM impact)**
Per-window noise floor tracks camera health, illumination uniformity, ensemble convergence.

**P6. Naturally anisotropic weights (LOW-MEDIUM impact)**
Current w_soft assumes Gaussian decay shape parameterized by Sigma_init. New weights use the actual F_ref shape — captures real particle anisotropy (astigmatism, sheet thickness effects) without assumptions.

**P7. Two-stage decouples N from Sigma (LOW impact but important)**
Fitting N simultaneously with Sigma could create partial degeneracy. Stage 1 fits N from F_ref alone (independent of F_AB and velocity parameters). Stage 2 uses N as fixed. No coupling.

### CONS

**C1. Computational cost ~10-20% increase per window (LOW impact)**
Stage 1 adds ~50us per window. Stage 2 uses more residual points (full grid vs 10% mask) but simpler weight computation. Net: ~10-20% slower per window. Since k-space is ~25% of total pipeline time, this is ~3-5% total increase. Negligible.

**C2. Assumes white noise (LOW risk)**
N is a scalar (flat PSD). Valid for camera read noise + shot noise (the dominant sources). Could break with aggressive image preprocessing that creates colored noise. In practice, ensemble-averaged correlation planes are clean.

**C3. Isotropic sigma_A_xy = 0 assumption (LOW risk)**
We fit sigma_A_x and sigma_A_y (per-axis) but not sigma_A_xy (cross-term / rotation). Particles are usually nearly round or axis-aligned. If rotation matters, extend to 4 params in Stage 1.

**C4. Stage 1 model assumes Gaussian particle shape (LOW risk)**
The particle autocorrelation is well-approximated by a Gaussian for most PIV conditions. Non-Gaussian particle shapes (e.g., defocused rings, elongated streaks) would make Stage 1 sigma_A less accurate, but the weights w = F_ref_clean / DC still work because they use the actual F_ref, not the Gaussian model.

**C5. F_ref_clean(k) can go negative at some k (HANDLED)**
When F_ref(k) < N due to noise fluctuations: `F_ref_clean = max(F_ref - N, epsilon)`, weight ~ epsilon / DC ~ 0. These points contribute negligibly. Bounds on N prevent systematic issues.

**C6. Division by small F_ref_clean at high k amplified by outliers (HANDLED)**
At high k, F_ref_clean ~ epsilon, so T_norm = F_AB / epsilon ~ large. Normally w(k) ~ epsilon / DC ~ 0, so the weighted residual is small. However, isolated outliers (hot pixels, cosmic rays, periodic artifacts) can produce |F_AB| ~ 10^4 at a single high-k point, giving |T_norm| ~ 10^4 / epsilon ~ 10^10 with w ~ 10^-7, yielding a weighted residual of ~10^3 — orders of magnitude above the good data (O(0.01)). A few such points can dominate the least-squares cost and corrupt the fit.

**Mitigation:** Retain a generous hard safety boundary at k_max_safety = 0.45 (Nyquist/2). Points beyond this are excluded regardless of weights. Within this boundary, the soft weights operate as designed. This is belt-and-suspenders: the weights handle the smooth signal-to-noise transition, while the hard cap protects against pathological outliers that the weight cancellation cannot absorb.

```python
# Safety mask — exclude extreme high-k where outliers live
k_safety = 0.45
safety_mask = (K_X**2 + K_Y**2) <= k_safety**2
# Apply BEFORE weighting — no outlier can enter the fit
K_X_safe = K_X[safety_mask]
K_Y_safe = K_Y[safety_mask]
T_norm_safe = T_norm[safety_mask]
w_safe = w[safety_mask]
```

This adds one line of masking and preserves ~80% of the k-plane (vs ~10% with current elliptical mask), while eliminating the corner regions where outliers concentrate.

**C7. Two-camera noise subtraction from geometric mean (HANDLED)**
For stereo PIV: `F_ref = sqrt((S_A + N_A)(S_B + N_B))`. At high k where S → 0: `F_ref → sqrt(N_A * N_B)`. The annular ring estimate gives N_eff = sqrt(N_A * N_B) automatically. Single N parameter works for the geometric mean.

---

## 6. Comparison to Current Method

| Aspect | Current k-space | Noise-aware k-space v2 |
|--------|----------------|------------------------|
| Noise handling | Avoid (k-mask + soft weight) | Model (N as parameter in Stage 1) |
| k_max needed? | Yes (most sensitive parameter) | No (eliminated) |
| Weighting | w_snr * w_soft (depends on Sigma_init, circular) | F_ref_clean / DC (data-only, non-circular) |
| Data used | ~10% (within elliptical mask) | ~80% (within k_safety = 0.45 circle, weights handle attenuation) |
| Parameters | Stage: 5 | Stage 1: 3 (N, sigma_A_x, sigma_A_y), Stage 2: 5 (same) |
| sigma_A output | NaN (not estimated) | Estimated from F_ref Gaussian decay |
| Gradient correction | Window term only (L^2/12) | Full: L^2/12 + sigma_A |
| Anisotropic handling | Assumes Gaussian w_soft from Sigma_init | Uses actual F_ref shape |
| Per-window cost | Reference baseline | ~10-20% slower |
| Code complexity | k_max cascade + soft weight = ~60 lines | Stage 1 fit + 1-line weights = ~40 lines |
| Tuning parameters | snr_threshold, 1% threshold, 0.25/0.35 caps | snr_threshold only |

---

## 7. Implementation Plan

### Single file modified: `pivtools_cli/piv/piv_backend/kspace_fitting.py`

**A. New function: `_fit_noise_and_particle()`**
- Input: F_ref, K_X, K_Y, center indices
- Output: N_eff, sigma_A_x, sigma_A_y, F_ref_clean
- 3-parameter least_squares on F_ref
- Fallback: N from annular ring, sigma_A = 3.0

**B. Modify `_fit_single_window_kspace()`**
- Replace k_max cascade (lines 334-388) with call to `_fit_noise_and_particle()`
- Compute w = F_ref_clean / F_ref_clean(0)
- Pass w and F_ref_clean downstream

**C. Modify `_fit_transfer_function_full()`**
- Remove: k_max_x, k_max_y, noise_floor, sigma_xx_estimate, sigma_yy_estimate parameters
- Remove: elliptical k_mask construction
- Remove: w_snr * w_soft computation
- Add: pre-computed weights parameter
- Apply hard safety mask at k_safety = 0.45 before fitting (outlier protection, see C6)
- Use all k-points within safety mask, with pre-computed weights

**D. Modify `_fit_1d_axis()`**
- Use F_ref_clean instead of raw F_ref
- Widen k-range (noise-corrected reference is usable to higher k)

**E. Modify `_build_params_from_fit()`**
- Accept sigma_A_x, sigma_A_y
- Set params[6:8] = sigma_A_x, sigma_A_y, 0.0
- Set params[9:11] = sigma_A_x + Sigma_xx, sigma_A_y + Sigma_yy, Sigma_xy

**F. Remove/deprecate:** `_compute_kmax_from_profile()`, `_compute_kmax()` (keep for diagnostics only)

**G. Update `plot_kspace_diagnostic()`**
- Replace k-bounds ellipses with weight heatmap
- Add F_ref vs F_ref_model overlay
- Show noise floor and sigma_A in summary

### Zero changes needed in other files
- `single_pass_accumulator.py` — `safe_extract` handles finite values, stress formula gives same result
- `gradient_correction.py` — reads sig_A from pass results, now gets actual values
- `save_results.py` — format-agnostic
- `validation.py` — checks unchanged

---

## 8. Verification Strategy

1. **Synthetic planes:** Known N, sigma_A, Sigma, mu → verify all parameters recovered
2. **Regression on real data:** Compare Reynolds stress fields to current kspace output
3. **Gradient correction:** Compare sigma_A to Gaussian fitter's estimate (~10-20% agreement expected)
4. **Edge cases:** Low SNR, high turbulence, anisotropic particles, stereo
5. **Performance:** Time per window within 2x of current kspace

---

## 9. Summary

The noise-aware k-space fitting replaces the weakest part of the current implementation (k_max heuristic cascade) with a physically motivated noise floor model. The two-stage design decouples noise/particle estimation from velocity estimation, avoiding N-Sigma degeneracy. The data-derived weights `w = F_ref_clean / F_ref_clean(0)` are non-circular, naturally anisotropic, and require zero tuning parameters. As a bonus, sigma_A estimation enables full gradient correction. All changes are confined to `kspace_fitting.py` with zero downstream modifications needed.
