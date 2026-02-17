# K-Space vs Gaussian Fitting for Ensemble PIV

Technical reference for the two Reynolds stress extraction methods implemented in PIVTOOLs.

---

## 1. The Problem

Ensemble PIV accumulates correlation planes across many image pairs. The ensemble-averaged cross-correlation R_AB encodes both the **particle image shape** and the **displacement probability distribution**:

```
R_AB(s) = f_particle(s) ⊛ f_displacement(s)
```

where ⊛ denotes convolution. Under the Gaussian assumption, the cross-correlation peak width is:

```
σ²_AB = σ²_particle + σ²_displacement
```

The particle image shape is a nuisance — we want the displacement distribution parameters (mean velocity and Reynolds stresses). The two fitting methods differ in how they separate these contributions.

---

## 2. Gaussian Fitting (Physical Space)

**File:** `pivtools_cli/piv/piv_backend/gaussian_fitting.py` (Python wrapper) + `pivtools_cli/lib/marquadt_gaussian.c` (C/GSL solver)

### Model

Fits three 2D Gaussians simultaneously (a "stacked" fit) to the autocorrelation planes R_AA, R_BB and cross-correlation plane R_AB:

```
R(x,y) = A · exp(-½ · [x,y] · Σ⁻¹ · [x,y]ᵀ) + c
```

### Parameters (16 per window)

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0-2 | A_A, A_B, A_AB | Peak amplitudes |
| 3-5 | c_A, c_B, c_AB | Background offsets |
| 6-8 | σ_A_x, σ_A_y, σ_A_xy | Autocorrelation widths (particle image) |
| 9-11 | δ_x, δ_y, δ_xy | Delta = σ_AB - σ_A (internally), output as σ_AB |
| 12-13 | x0_A, y0_A | Autocorrelation peak centre |
| 14-15 | x0_AB, y0_AB | Cross-correlation peak centre (= mean displacement) |

### Reynolds Stress Extraction

Requires **subtraction** of two independently estimated quantities:

```
UU = σ²_AB_x - σ²_A_x
VV = σ²_AB_y - σ²_A_y
UV = σ²_AB_xy - σ²_A_xy
```

The C fitter internally uses a **delta parameterisation** where params[9-11] represent δ = σ_AB - σ_A directly, with σ_AB reconstructed as σ_A + max(δ, 0). This constrains σ_AB ≥ σ_A (non-negative Reynolds stress). The output is converted back to σ_AB values for Python consumption.

### Implementation

- **Solver:** GSL `gsl_multifit_nlinear` with Cholesky decomposition and geodesic acceleration (`lmaccel` trust region)
- **Tolerances:** XTOL=1e-4 (relaxed for small deltas), GTOL=FTOL=1e-6
- **Extraction region:** 32×32 pixels around the correlation peak
- **Parallelism:** OpenMP `schedule(dynamic, 16)` across windows
- **Centre pixel masking:** Optional masking of the AA/BB centre pixel to avoid camera self-noise spikes
- **Initial guesses:** Pass 0 uses HWHM-based σ estimation from the correlation peak shape; subsequent passes interpolate σ fields from the previous pass result

### Strengths

- Fast: ~20 μs/window (C + OpenMP)
- Mature: well-established in the literature (Scharnowski et al. 2012)
- Offset fitting: handles non-zero correlation backgrounds
- Extensive validation and diagnostic tooling

### Weaknesses

- **Subtraction noise:** Reynolds stress = σ²_AB - σ²_A. Both estimates have errors, and the errors add in the subtraction. When displacement variance is small relative to particle size, the SNR of the stress estimate degrades severely.
- **16 parameters:** Large parameter space with coupling between amplitude, offset, and width parameters. Condition number O(10³-10⁶).
- **σ_AB ≥ σ_A constraint:** Creates asymmetric positive bias in low-turbulence regions where noise can push σ_A above σ_AB.
- **Sensitive to initial guess quality:** 16-parameter LM has many local minima. HWHM-based initial guesses can be poor for asymmetric or low-SNR peaks.

---

## 3. K-Space Fitting (Fourier Space)

**File:** `pivtools_cli/piv/piv_backend/kspace_fitting.py` (pure Python/NumPy/SciPy)

### Core Insight

In Fourier space, convolution becomes multiplication. The convolution theorem gives:

```
F(R_AB) = F(f_particle) · F(f_displacement)
```

Since the autocorrelation R_AA is the self-convolution of the particle image:

```
F(R_AA) = |F(f_particle)|²
```

We can define a **transfer function** that algebraically cancels the particle contribution:

```
T(k) = F(R_AB) / √(F(R_AA) · F(R_BB))
```

This transfer function encodes **only** the displacement distribution parameters. No subtraction is needed — the particle image shape is divided out.

### Model

For a Gaussian displacement distribution, the transfer function is:

```
T(k) = A · exp(-2π² · kᵀ · Σ · k) · exp(-2πi · k · μ)
```

where:
- **μ = (μ_x, μ_y)** — mean displacement, encoded in the **phase** of T(k)
- **Σ = [[Σ_xx, Σ_xy], [Σ_xy, Σ_yy]]** — displacement variance-covariance (Reynolds stresses), encoded in the **magnitude decay** of T(k)
- **A** — amplitude scaling

### Parameters (5 per window)

| Parameter | Description |
|-----------|-------------|
| μ_x, μ_y | Mean displacement |
| Σ_xx | UU Reynolds stress (directly) |
| Σ_yy | VV Reynolds stress (directly) |
| Σ_xy | UV Reynolds shear stress (directly) |

The amplitude A is removed by DC-normalising the transfer function: T_norm = T / T(0), leaving 5 free parameters. This also handles the case where F_AA(0) ≠ F_BB(0) (different illumination between frames).

### Reynolds Stress Extraction

**Direct** — no subtraction:

```
UU = Σ_xx   (fitted parameter)
VV = Σ_yy   (fitted parameter)
UV = Σ_xy   (fitted parameter)
```

The particle image contribution was cancelled by the division in Fourier space. This is the fundamental advantage.

### Implementation Pipeline

The fitting proceeds in stages for each window:

#### Stage 1: FFT and Reference Computation
```python
F_AA = FFT(R_AA)
F_BB = FFT(R_BB)
F_AB = FFT(R_AB)
F_ref = sqrt(|F_AA| · |F_BB|)          # particle image power spectrum
T_measured = F_AB / (F_ref + ε)         # raw transfer function
T_norm = T_measured / T_measured(0)     # DC-normalised
```

The `ifftshift` before FFT is critical because correlation planes have their peak at the centre (index N/2), and the FFT expects the signal centred at index 0 for correct phase computation.

#### Stage 2: Adaptive Frequency Bounds

Two mechanisms determine which wavenumbers to include:

1. **F_ref decay profile:** Find where |F_ref(k)| drops to 1% of its DC value along each axis. This gives per-axis bounds k_max_x and k_max_y.

2. **SNR-based bound:** `k_max = √(ln(SNR) / (2π²σ²))` — the wavenumber where the transfer function magnitude equals the noise floor.

The more conservative (smaller) of these bounds is used, with a hard cap at 0.25 cycles/pixel (half Nyquist) when using hard cutoffs, or 0.45 when using soft weighting (since the weights handle high-k attenuation).

An **elliptical mask** combines the per-axis bounds:
```
K_X²/k_max_x² + K_Y²/k_max_y² ≤ 1.0
```

#### Stage 3: Initial Guesses via 1D Linear Regression

Along each axis independently:

- **Mean displacement** from phase slope: `phase(T) = -2πk·μ`, so `μ = -slope/(2π)` via weighted linear regression of phase vs k.

- **Variance** from log-magnitude slope: `log|T| = -2π²Σk²`, so `Σ = -slope/(2π²)` via weighted linear regression of log|T| vs k².

These 1D regressions use **log differences** (`log|F_AB| - log|F_ref|`) rather than explicit division, avoiding noise amplification. They provide fast, robust initial guesses for the nonlinear fit.

Phase estimation uses a more conservative k range than magnitude estimation to avoid phase wrapping at high wavenumbers.

#### Stage 4: Full 5-Parameter Nonlinear Fit

**Solver:** `scipy.optimize.least_squares` with Trust Region Reflective (TRF) method.

**Residual:**
```python
residual = weights · (T_norm - T_model)
# Split into real and imaginary components for the real-valued solver
```

**Bounds:** Σ_xx, Σ_yy ≥ 0 (physical constraint); displacements bounded by ±10 pixels.

**Convergence:** ftol=xtol=1e-8, max 200 function evaluations.

### Weighting Strategy

The fit uses a product of two weight functions:

```python
# SNR-based: emphasises high-signal spectral regions
w_snr = |F_ref| / √noise_floor

# Soft decay: isotropic Gaussian rolloff matching transfer function bandwidth
σ_avg = (Σ_xx_est + Σ_yy_est) / 2
k0² = 1 / (2π² · σ_avg)
w_soft = exp(-K_R² / k0²)

# Combined (normalised)
weights = w_snr · w_soft
```

Design choice: **isotropic** soft weighting was chosen over anisotropic after empirical testing showed it provides more balanced weight between x and y directions. The anisotropic variant was tried (using separate k0_x, k0_y from Σ_xx, Σ_yy estimates) but abandoned.

### Output Format

K-space results are packed into the same 16-element parameter array as the Gaussian fitter for drop-in compatibility:

| Index | Value |
|-------|-------|
| 0-2 | Amplitudes (from correlation peak values) |
| 3-5 | 0.0 (offsets not used) |
| 6-8 | NaN → 0.0 after safe_extract (σ_A not estimated) |
| 9-11 | Σ_xx, Σ_yy, Σ_xy (Reynolds stresses directly) |
| 12-13 | Window centre (autocorrelation peak) |
| 14-15 | Window centre + μ (displacement) |

Since σ_A = 0 in the output, the downstream stress extraction `UU = σ_AB - σ_A = Σ_xx - 0 = Σ_xx` gives the correct result through the same code path as Gaussian fitting.

### Status Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Did not converge (max iterations exceeded) |
| 2 | Low SNR (window-level rejection) |
| 3 | Negative variance in result |
| 4 | Displacement exceeds ½ window |

### Strengths

- **No subtraction:** Reynolds stresses estimated directly — no error amplification from differencing two noisy estimates
- **5 parameters:** Well-conditioned problem, Jacobian condition number O(10-100)
- **Phase-magnitude decoupling:** Displacement errors (phase) do not propagate into stress estimates (magnitude), and vice versa. The Fisher information matrix is approximately block-diagonal.
- **Robust initial guesses:** 1D linear regressions are closed-form, fast, and use all valid spectral data
- **Peak locking immunity:** Displacement from phase slope regression over many wavenumber bins, not sub-pixel peak location
- **Non-Gaussian detection:** Transfer function oscillations (instead of smooth decay) indicate bimodal or non-Gaussian displacement distributions

### Weaknesses

- **Speed:** ~50-100× slower per window than the C Gaussian fitter (pure Python/SciPy). Total pipeline impact ~25% since fitting is not the bottleneck. A future C/FFTW implementation could close this gap.
- **No σ_A estimation:** Particle image variance is cancelled, not estimated. This means gradient correction only applies the window averaging term (L²/12), omitting the particle extent term (σ_A). The window term dominates (~95-97% of total correction).
- **k_max sensitivity:** The choice of frequency bounds affects fit quality. The adaptive F_ref decay method with soft weighting mitigates this but it remains the most sensitive tuning parameter.

---

## 4. Head-to-Head Comparison

### Error Propagation

The fundamental difference is **subtraction vs division**.

**Gaussian:** `RS = σ²_AB - σ²_A`, with `Var(RS) = Var(σ²_AB) + Var(σ²_A) - 2·Cov(...)`. The variances of both estimates add.

**K-space:** `RS = Σ` (direct fit parameter). Error is determined solely by the Fisher information of the 5-parameter transfer function model.

**Pathological example (low turbulence):**
- Particle σ_A = 3.0 px, displacement σ_disp = 0.3 px
- σ_AB = √(9 + 0.09) = 3.015 px
- Fitting precision ~0.05 px on each σ
- Gaussian: signal = 0.09 px², noise ≈ 0.43 px² → **SNR ≈ 0.2**
- K-space: estimates 0.09 px² directly from transfer function curvature over ~50-100 spectral bins

### Regime Performance

| Regime | Gaussian | K-Space |
|--------|----------|---------|
| **Low turbulence** (σ_disp << σ_particle) | Catastrophic: differencing nearly identical numbers, constraint bias | Graceful: precision degrades but no bias cliff |
| **Moderate turbulence** (σ_disp ~ σ_particle) | Good: well-separated widths, good SNR | Good: rich curvature information |
| **High turbulence** (σ_disp >> σ_particle) | Truncation effects (broad peak exceeds extraction window) | Narrow useful bandwidth but steep, easily measured curvature |

### Bias Sources

| Source | Gaussian | K-Space |
|--------|----------|---------|
| σ_AB ≥ σ_A constraint | Systematic positive bias at low turbulence | N/A (no constraint needed) |
| Peak locking | Moderate (~0.01 px with LM fitting) | Immune (phase slope over many k-bins) |
| Truncation | σ biased low when peak exceeds window | N/A (works in frequency domain) |
| Offset-σ coupling | Offset absorbs Gaussian tails → σ bias | N/A (no offset parameter) |
| k_max selection | N/A | Data-dependent boundary; mitigated by soft weighting |
| Velocity gradients | σ²_gradient = (du/dy · L/12)² added to stress | Same bias (indistinguishable from turbulence) |

### Numerical Conditioning

| Property | Gaussian | K-Space |
|----------|----------|---------|
| Parameters | 16 | 5 |
| Jacobian condition number | O(10³-10⁶) | O(10-100) |
| Local minima | Many (amplitude-σ-offset coupling) | Few (smooth landscape) |
| Initial guess sensitivity | High (HWHM can fail) | Low (1D regression robust) |

### Accuracy

Synthetic benchmarks show **40-90% improvement** in Reynolds stress accuracy with k-space fitting, with the largest gains in the low-turbulence regime where accurate stress measurement is both critical and hardest.

Mean displacement accuracy is comparable between the two methods for well-resolved particles (d_τ > 2 px), with k-space having a slight advantage from phase slope regression.

---

## 5. Design Choices and Rationale

### Why DC Normalisation?

The raw transfer function `T(k) = F_AB / F_ref` has an amplitude ambiguity: if illumination differs between frames A and B, then `F_AA(0) ≠ F_BB(0)`, and the geometric mean `F_ref = √(|F_AA|·|F_BB|)` does not perfectly cancel the particle contribution at DC. Normalising by `T(0)` removes this ambiguity and reduces the problem from 6 to 5 parameters.

### Why Isotropic Soft Weighting?

The transfer function for anisotropic turbulence (Σ_xx ≠ Σ_yy) decays at different rates in k_x and k_y. Anisotropic soft weighting `exp(-k_x²/k0_x² - k_y²/k0_y²)` was the natural choice to match this, but empirical testing showed it can over-suppress one direction when the initial Σ estimates are inaccurate. Isotropic weighting with `σ_avg = (Σ_xx + Σ_yy)/2` provides more balanced coverage and lets the optimizer find the anisotropy itself.

### Why Log Differences for Initial Guesses?

The transfer function is computed as T = F_AB / F_ref, but this division amplifies noise where F_ref is small. For the 1D initial guesses, we instead use:

```
log|T| = log|F_AB| - log|F_ref|
```

This avoids explicit division entirely. The log-space subtraction is numerically stable and gives the same regression result as fitting log|T| directly. The full nonlinear fit does use the explicit division T = F_AB / (F_ref + ε), but only within the masked elliptical region where F_ref is sufficiently large.

### Why an Elliptical k-mask?

The particle image can be non-isotropic (e.g. astigmatism, sheet thickness effects), causing F_ref to decay faster in one direction. Using a single isotropic k_max would either include too much noise in one direction or exclude useful signal in the other. The per-axis k_max with elliptical masking adapts to the actual particle image shape.

### Why Not Deconvolution?

Non-parametric deconvolution (dividing F_AB by F_ref at all wavenumbers, then inverse-FFT) is the obvious Fourier-space approach, but it amplifies noise at high k where F_ref → 0. Our approach is parametric: we fit a Gaussian model to the transfer function within the reliable bandwidth. This is equivalent to regularised deconvolution with a Gaussian prior on the displacement PDF.

### Why 16-Element Output?

The k-space fitter packs its 5 fitted parameters into the same 16-element array layout as the Gaussian fitter. This makes the two methods fully interchangeable via a single config switch (`fit_method: kspace`), with no changes needed downstream in stress extraction, outlier detection, infilling, saving, or visualisation. The unused fields (offsets, σ_A) are set to zero/NaN.

### Gradient Correction with K-Space

The gradient correction formula has two terms:

```
Correction = gradient² × (L²/12 + σ_A)
              window term   particle term
```

K-space fitting does not estimate σ_A (it was cancelled in Fourier space). When gradient correction is enabled with k-space, only the window averaging term (L²/12) is applied. This is the dominant term: for a 32×32 window, L²/12 ≈ 85 px² vs typical σ_A ≈ 2-5 px² (~3-6% of total correction). The omission is logged explicitly.

---

## 6. Configuration

To switch between methods, set in `config.yaml`:

```yaml
ensemble_piv:
  fit_method: gaussian   # default, 16-parameter LM in C
  # or
  fit_method: kspace     # 5-parameter transfer function in Python
```

K-space fitting has no user-facing tuning parameters. The `kspace_snr_threshold` (default 3.0) is an internal per-window rejection gate equivalent to the Gaussian fitter's peak amplitude checks — it should not need adjustment.

Both methods are fully compatible with all other pipeline features: multi-pass processing, sum_fitting_window, background subtraction, outlier detection, infilling, gradient correction, and diagnostic plane saving.

---

## 7. Literature Context

The physical-space subtraction approach was established by **Scharnowski, Hain & Kähler (2012)**, "Reynolds stress estimation up to single-pixel resolution using PIV-measurements" (Exp. Fluids 52, 1519-1527).

The theoretical foundation for the transfer function approach draws on:
- **Westerweel (2008)**, "On velocity gradients in PIV interrogation" — introduced the transfer function concept for PIV spatial filtering analysis
- **Theunissen & Edwards (2018)**, "A general approach to evaluate the ensemble cross-correlation response for PIV using Kernel density estimation" — generalised the transfer function to arbitrary (non-Gaussian) displacement distributions
- The **optical transfer function (OTF)** analogy from imaging science, where division by the system response isolates the object signal

The complete k-space fitting pipeline as implemented here — with adaptive weighting, 1D regression warm-start, DC normalisation, and elliptical frequency bounds — does not appear to have been published in the ensemble PIV literature. The closest work uses the transfer function for spatial resolution analysis rather than as a fitting target for Reynolds stress extraction.

---

## 8. Future Work

- **C/FFTW implementation:** The per-window FFT + 5-parameter TRF fit could be implemented in C with FFTW, closing the speed gap to ~3-5× (vs current ~50-100×). The FFT is O(N log N) and the 5-parameter fit is much cheaper than the 16-parameter LM.
- **σ_A estimation from F_ref:** A Gaussian fit to the reference spectrum `log|F_ref(k)| = -2π²σ²_A k² + const` would recover σ_A for full gradient correction. This is a simple 1D linear regression on data already computed.
- **Non-Gaussian displacement models:** The k-space framework naturally extends to mixture models (bimodal distributions show interference fringes in |T(k)|) and higher-order cumulant extraction (skewness, kurtosis from phase nonlinearity).
