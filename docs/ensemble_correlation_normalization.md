# Ensemble Correlation Plane Aggregation Methods

PIVTOOLs supports two methods for aggregating cross-correlation planes across image pairs in ensemble PIV. The method is controlled by:

```yaml
ensemble_piv:
  correlation_normalization: none   # or "per_frame"
```

---

## Method 1: Raw Accumulation with Background Subtraction (`none`, default)

This is the classical ensemble PIV approach. Correlation planes are accumulated without modification, then the DC bias (background) is removed afterwards by correlating the mean images.

### Per frame, per window (C library)

Each frame contributes a raw, unnormalized correlation plane:

```
W_A = weight_A ⊙ A_i          (element-wise: apply taper to sub-image)
W_B = weight_B ⊙ B_i

R_i = IFFT( FFT(W_B) · conj(FFT(W_A)) )     (cross-correlation via FFT)

S += R_i                                       (accumulate raw)
```

The per-window mean is computed (`mean_A`, `mean_B`) but **not subtracted** — the DC content of each window remains in the correlation. The amplitude of each frame's correlation plane scales with the product of the image intensities, so bright frames produce larger correlation amplitudes and contribute more to the sum.

### After all frames (Python, `finalize_pass`)

The accumulated sum still contains the DC pedestal from each frame's mean intensity. This is removed by computing the background correlation from the mean images:

```
R_raw  = S / N                          (average raw correlation)

Ā = (1/N) Σ A'_i                        (mean warped image A)
B̄ = (1/N) Σ B'_i                        (mean warped image B)

R_bg   = Ā ⊗ B̄                          (background: correlate the means)

R_ensemble = R_raw − R_bg               (DC-free ensemble correlation)
```

The same subtraction is applied to the auto-correlations (AA, BB).

Finally, the ensemble correlation is normalized by the geometric mean of the auto-correlation peaks:

```
R_normalized = R_AB_ensemble / sqrt( R_AA_peak · R_BB_peak )
```

A 6-DOF Gaussian is fitted to `R_normalized` to extract displacement (peak position) and Reynolds stresses (peak widths).

### Single-mode correction

In single mode (small particle window × large sum window), the AA and BB auto-correlations use the full sum-window weight on both sides, but AB uses different weights (particle × sum). This creates an area asymmetry. A scale correction factor is applied:

```
ab_correction = sqrt( sum_window_area / particle_window_area )
R_AB_corrected = R_AB_ensemble × ab_correction
```

### Characteristics

- **Default, well-tested** — the established method for this codebase
- **Frame weighting**: Each frame's contribution scales as `intensity_A · intensity_B · correlation_quality`. Both image brightness and signal quality affect the weight — they cannot be separated.
- **Background subtraction**: Accurate when the mean image is a good estimate of the DC component (requires many frames)
- **Two-step process**: Accumulate in C → subtract background in Python

---

## Method 2: Per-Frame Normalization (`per_frame`)

Each frame's correlation is mean-subtracted and energy-normalized **before** accumulation. The normalization by `sqrt(var_A · var_B)` produces the Pearson cross-correlation coefficient, which removes the intensity dependence but preserves the signal-quality dependence.

### Per frame, per window (C library)

```
W_A = weight_A ⊙ A_i                    (apply taper)
W_B = weight_B ⊙ B_i

mean_A = sum(W_A) / N_px                (weighted mean)
mean_B = sum(W_B) / N_px

W_A -= mean_A                           (subtract mean → zero-mean)
W_B -= mean_B

var_A = Σ W_A[k]²                       (variance of zero-mean signal)
var_B = Σ W_B[k]²

R_i = IFFT( FFT(W_B) · conj(FFT(W_A)) )    (cross-correlation of zero-mean windows)

S += R_i / sqrt(var_A · var_B)               (Pearson coefficient — normalize then accumulate)
```

The result `R_i / sqrt(var_A · var_B)` is the **Pearson cross-correlation coefficient** for that window pair. Its peak value ranges from 0 (no correlation) to 1 (perfect match), independent of image intensity. However, the peak value still depends on how well the particle patterns in A and B actually correlate — i.e., signal quality.

If `sqrt(var_A · var_B) < 1e-12` (near-zero variance, e.g. uniform region), that frame is skipped for that window.

### What this means for frame weighting

The Pearson normalization **removes** the intensity dependence: a frame with 2× brightness produces the same normalized correlation as one with 1× brightness (the `sqrt(var_A · var_B)` denominator scales identically).

But it **preserves** signal-quality differences. Frames with:
- Sharp, well-resolved particles → stronger Pearson peak → larger contribution
- Out-of-focus or noisy images → weaker Pearson peak → smaller contribution
- Good particle density → more matching pairs → stronger peak
- Loss-of-pairs (particles leaving the window) → weaker peak

This is generally desirable: high-quality frames naturally contribute more to the ensemble average, while low-quality frames contribute less. What is removed is the spurious weighting by raw brightness, which carries no information about displacement.

### After all frames (Python, `finalize_pass`)

Because the DC component was removed per-frame before correlation, the background subtraction step is **skipped entirely**:

```
R_ensemble = S / N                       (already DC-free and normalized)
```

No `Ā ⊗ B̄` computation is needed. The result goes directly to geometric-mean normalization → Gaussian fitting.

### Single-mode correction

The AB scale correction is also **skipped** because the per-frame normalization by `sqrt(var_A · var_B)` already accounts for the actual weighted energies in each window, absorbing the area asymmetry.

### Characteristics

- **Intensity-independent weighting**: Removes the brightness component of frame weighting; a frame with 2× laser intensity does not get 2× influence
- **Signal-quality-dependent weighting**: Frames with better particle images (higher Pearson coefficient) still contribute more — this is a feature, not a bug
- **Reduces spatial bias**: In flows with non-uniform illumination (laser sheet edges, reflections), raw accumulation biases displacement toward the bright-frame displacement; per-frame normalization eliminates this
- **One-step process**: Everything happens in the C inner loop; no Python background subtraction needed
- **Preserves displacement**: Peak position (displacement) is invariant under per-frame scalar normalization
- **Preserves stress information**: Gaussian peak widths (which encode Reynolds stresses) are invariant under scalar normalization of each frame's correlation

---

## Side-by-Side Comparison

| Aspect | `none` (default) | `per_frame` |
|--------|-------------------|-------------|
| DC removal | Post-hoc: `R_raw − Ā⊗B̄` | Per-frame: mean subtracted before FFT |
| Frame weighting | `∝ intensity² × quality` | `∝ quality` (Pearson coefficient) |
| Intensity dependence | Yes — bright frames dominate | No — normalized out |
| Signal quality dependence | Yes | Yes (preserved via Pearson coefficient) |
| Background subtraction | Required (Python) | Skipped |
| Single-mode AB correction | Required | Skipped (absorbed) |
| Compute cost per frame | Lower (no extra division) | Slightly higher (mean-sub + variance + normalize) |
| Total compute cost | Higher (extra mean-image correlation) | Lower (no background correlation step) |
| Sensitivity to N | Background estimate improves with more frames | No background estimate needed |
| Uniform illumination | Both methods equivalent | Both methods equivalent |
| Non-uniform illumination | Bright regions dominate | Intensity bias removed |

---

## Mathematical Detail

### What each frame contributes

Consider a single window in frame `i`. Let `I_i` be the mean intensity of that window (a proxy for brightness) and `ρ_i` be the underlying Pearson correlation strength (a measure of how well the particle patterns match).

The unnormalized cross-correlation amplitude scales as:

```
‖R_i‖ ∝ I_i² · ρ_i
```

The intensity-squared term arises because both the signal and the DC pedestal scale with intensity, and the cross-correlation is bilinear.

After per-frame normalization:

```
‖R_i / sqrt(var_A · var_B)‖ ∝ ρ_i
```

The `I_i²` factor cancels because `var_A ∝ I_i²` and `var_B ∝ I_i²`, so `sqrt(var_A · var_B) ∝ I_i²`.

### Equivalence under uniform illumination

When all frames have identical intensity `I_i = I` (constant):

- **Raw**: `S = Σ R_i = I² · Σ ρ_i · R₀`
- **Per-frame**: `S = Σ (R_i / I²σ²) = (1/σ²) · Σ ρ_i · R₀`

Both yield the same peak position and the same relative peak widths (stresses). The absolute scale differs by `I²σ²` but is removed by the geometric-mean normalization step.

When `I_i` varies across frames, raw accumulation weights frame `i` by `I_i² · ρ_i`, while per-frame normalization weights by `ρ_i` alone.

---

## When to Use Each

**Use `none` (default)** when:
- You have uniform, stable illumination across all frames
- You want to match established PIV processing workflows
- You are comparing results against a known reference processed without per-frame normalization

**Use `per_frame`** when:
- Laser intensity varies between pulses (common in high-repetition-rate systems)
- There are spatially non-uniform intensity patterns (reflections, sheet edges)
- Some frames have significantly different particle density (seeding variation)
- You want frame contributions weighted purely by signal quality (Pearson coefficient), not by brightness
