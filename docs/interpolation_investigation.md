# Interpolation Methods for PIV Image Deformation

> Investigation into warp-induced spectral bias in the k-space fitter, comparing
> interpolation methods by their transfer function |H(k)|^2.
>
> Date: February 2026

---

## Background

Multi-pass iterative PIV deforms images using a predictor displacement field.
The interpolation kernel used during this deformation acts as a spatial filter —
it attenuates (or amplifies) different spatial frequencies by a transfer function
H(k). In the k-space fitter, which works in Fourier domain, any departure from
|H(k)|^2 = 1.0 directly biases the fitted displacement variances (Reynolds
stresses).

This investigation measured |H(k)|^2 empirically: shift 500 real PIV windows
by a half-pixel (worst case for interpolation), compute PSD(shifted)/PSD(original),
and average. The test data came from `warped_pass_1.mat` (2048x2048 raw PIV image).

---

## Methods Tested

### 1. Bicubic (cv2.INTER_CUBIC) — Current Production
Keys cubic convolution with a = -0.75, 4x4 stencil. This is what `cv2.remap`
uses and what the current production pipeline deploys.

### 2. Lanczos-4 (cv2.INTER_LANCZOS4)
Windowed sinc with radius 4 (8x8 stencil). Available as a one-line change in
cv2.remap via `cv2.INTER_LANCZOS4`.

### 3. Whittaker / Windowed Sinc (custom Python)
Separable windowed-sinc interpolator with configurable kernel radius and window
function. Tested radii 3, 4, 6, 8, 12, 16 with Hamming, Hann, Blackman, Kaiser,
and unwindowed (truncated sinc). Implementation in `tools/test_whittaker_interpolation.py`.

### 4. FFT Phase Shift (baseline)
Exact Fourier-domain shift: multiply spectrum by `exp(-2j*pi*(kx*dx + ky*dy))`,
then IFFT. Gives |H(k)|^2 = 1.0000 by construction. Assumes periodic boundaries.

### 5. NUFFT via FINUFFT (type 2 transform)
Non-Uniform FFT: evaluates a 2D Fourier series at arbitrary (non-uniform) target
points. Mathematically equivalent to ideal sinc interpolation with tunable accuracy
(eps parameter). Implementation in `tools/test_nufft_interpolation.py`.

---

## Results

### Transfer Function |H(k)|^2 (16x16 windows, dx=0.5, 500 windows)

| Method         | |H(k=0.125)|^2 | |H(k=0.25)|^2 | |H(k=0.375)|^2 | Max error in fitting range |
|----------------|----------------|---------------|----------------|---------------------------|
| Bicubic        | 0.9800         | 0.9299        | 0.8761         | ~7% at |k|=0.25           |
| Lanczos-4      | 1.0056         | 1.0056        | 1.0292         | ~2.4% at Nyquist          |
| Whittaker r=3  | ~1.00          | ~0.99         | ~0.97          | ~3%                       |
| Whittaker r=6  | ~1.00          | ~1.00         | ~1.00          | <0.5%                     |
| Whittaker r=8  | ~1.00          | ~1.00         | ~1.00          | <0.1%                     |
| NUFFT e-6      | 1.0000         | 1.0000        | 1.0000         | 0.0%                      |
| NUFFT e-3      | 0.9999         | 1.0005        | 1.0002         | 0.05%                     |
| FFT            | 1.0000         | 1.0000        | 1.0000         | 0.0% (by construction)    |

### Timing (per window, 16x16)

| Method      | Time/window |
|-------------|-------------|
| Bicubic     | 0.1 ms      |
| Lanczos-4   | 0.1 ms      |
| FFT         | 0.1 ms      |
| NUFFT e-6   | 2.2 ms      |

### Full-Image Timing (2048x2048)

| Method      | Time      | Ratio vs Bicubic |
|-------------|-----------|------------------|
| Bicubic     | 119 ms    | 1.0x             |
| NUFFT e-6   | 517 ms    | 4.4x             |

---

## Key Findings

### 1. Bicubic has significant high-k attenuation (~7%)
At |k| = 0.25 (the middle of the k-space fitting range), bicubic attenuates
power by ~7%. This systematically biases displacement variances downward — the
k-space fitter sees less power than is really there, attributing it to smaller
particle images / tighter peaks. For Reynolds stress estimation, this means the
fitted `sigma_AB - sigma_A` (which gives the stress) is biased.

### 2. Lanczos-4 is much better (~2.4% error) and nearly free
Lanczos-4 overshoots slightly near Nyquist but is flat (within ~0.5%) across
the entire fitting range |k| < 0.25. Since it's available via a single flag
change in cv2.remap, it's the obvious first improvement.

### 3. Whittaker r=8 is nearly exact but slow in Python
Windowed sinc with radius 8 achieves <0.1% error everywhere. However, the pure
Python separable implementation (using np.roll for periodic boundaries) is much
too slow for production. It would need a C implementation — either in the fused
warp kernel or as a standalone interpolator.

### 4. NUFFT achieves exact |H(k)|^2 = 1.0 but is 4.4x slower
FINUFFT's type 2 transform correctly evaluates the Fourier series at shifted
points, achieving the same spectral fidelity as FFT phase shift. However, it
wraps at boundaries (periodic assumption), and at 4.4x the cost of bicubic,
it may not be worth the overhead for production ensemble PIV where hundreds of
image pairs are processed.

### 5. FFT phase shift is exact but assumes periodic boundaries
FFT shift gives |H(k)|^2 = 1.0 by construction, but it wraps image content
around the edges. For real PIV images (not periodic), this contaminates boundary
windows. The wrap-around is hidden in the |H(k)|^2 measurement (which averages
out boundary effects over 500 interior windows) but affects individual
cross-correlation peaks.

### 6. Why FFT/NUFFT aren't standard in PIV
Despite being spectrally exact, Fourier-based methods assume periodicity that
real images violate. When shifting a small window, FFT borrows intensity from
the opposite edge — severe for 16x16 or 32x32 windows where the boundary
fraction is large. Additionally, FFT applies a single uniform shift, whereas
production PIV needs spatially varying deformation. NUFFT handles non-uniform
points but still inherits the periodicity assumption and is significantly
slower. The PIV community generally accepts bicubic's ~7% attenuation as
"good enough" — or uses Lanczos/sinc if accuracy is critical.

---

## FINUFFT Implementation Notes

The working NUFFT implementation (in `tools/test_nufft_interpolation.py`) uses
FINUFFT's type 2 transform, which evaluates:

    c[j] = sum_{k1,k2} f[k1, k2] * exp(i * (k1 * x[j] + k2 * y[j]))

### The Bug That Took Three Sessions to Find

The original implementation had a single bug: **the x and y arguments were swapped**.

FINUFFT's `nufft2d2(x, y, f)` convention is that `k1` (the first dimension of `f`)
multiplies `x` (the first positional argument). Since `F = fftshift(fft2(img))` has
shape `(ny, nx)`, the first dimension contains y-modes. Therefore the first argument
must carry y-phase coordinates:

```python
# CORRECT: y-phase first (for k1 = y-modes), x-phase second (for k2 = x-modes)
result = finufft.nufft2d2(PY.ravel(), PX.ravel(), F, eps=eps)
```

### Why Earlier Fix Attempts Failed

1. **Coefficient correction (multiply negative modes by (-1)^k):** This addressed a
   real aliasing property of the DFT but was unnecessary because `fftshift` already
   places frequencies at the correct NUFFT modes. The aliasing changes shift direction
   but not |H(k)|^2. This fix was a no-op.

2. **2N zero-padded spectrum:** Placed N DFT coefficients at positive modes [0, N-1]
   within a 2N mode array to avoid aliasing. This broke conjugate symmetry — for real
   signals, DFT frequency k=N-1 is conjugate(F[1]). Evaluated as a positive mode at
   non-integer positions, `exp(2*pi*i*(N-1)*p/N) != exp(-2*pi*i*p/N)`, so the
   conjugate pairs no longer cancel their imaginary parts. Taking `np.real()` kills
   all non-DC content, producing a constant output.

3. **4-quadrant NUFFT split:** Attempted to handle positive/negative modes separately.
   Unnecessarily complex and still didn't fix the underlying argument swap.

The fix was literally swapping two variable names in one function call.

---

## Fused C Kernel: Implemented Interpolation Modes

The fused warp C kernel (`pivtools_cli/lib/fused_warp.c`) supports two interpolation
modes for Phase C (image sampling). Predictor upsampling (Phase A) always uses bicubic
regardless of mode — smooth displacement fields don't benefit from Lanczos.

### Mode 0: Bicubic (Keys a=-0.75, 4×4 stencil)

Matches `cv2.INTER_CUBIC`. 16 multiply-adds per pixel. ~7% attenuation at |k|=0.25.

### Mode 1: Lanczos-3 (windowed sinc, 6×6 stencil) — LUT-accelerated

Windowed sinc with a=3. 36 multiply-adds per pixel. ~2-3% max error at Nyquist.

The naive implementation (12 `sinf` calls per pixel) made Lanczos 5-6x slower than
bicubic, wiping out all fusion gains. This was solved with a precomputed weight LUT:
- **4096 entries × 6 weights × 4 bytes ≈ 96 KB** — fits in L2 cache
- Linear interpolation between entries (max error ~10⁻⁶, negligible vs float32)
- Built once before the OpenMP parallel region, read-only shared across threads
- Reduced Lanczos/bicubic cost ratio from 5-6x to **~1.5x** (the irreducible minimum
  from the stencil size ratio 36/16)

### Measured Performance (fused C kernel vs cv2.remap production pipeline)

| Size   | cv2 cubic | cv2 lanc4 | C bicubic | C lanczos3 | Speedup cub | Speedup lan |
|--------|-----------|-----------|-----------|------------|-------------|-------------|
| 1 MP   | 17.3 ms   | 19.1 ms   | 5.8 ms    | 9.3 ms     | 3.0x        | 2.0x        |
| 4 MP   | 64.1 ms   | 67.4 ms   | 9.6 ms    | 14.4 ms    | 6.7x        | 4.7x        |
| 25 MP  | 355.3 ms  | 351.0 ms  | 61.5 ms   | 83.7 ms    | 5.8x        | 4.2x        |

### Future Options (not currently needed)

Higher-accuracy methods documented for reference if the ~2-3% Lanczos error proves
insufficient for stress estimation:

- **Whittaker r=8 (16×16 stencil):** <0.1% error. ~16x more work per pixel than bicubic.
  Straightforward to add as a new `interp_mode` — same LUT approach, wider stencil.
- **NUFFT gather:** Exact (0.0% error) but requires FFTW infrastructure, 4x memory
  overhead, ~80 lines of new C code. Over-engineered for the actual accuracy needs.

---

## Decision: Production Pipeline (February 2026)

**The production pipeline supports bicubic and Lanczos-3 via the fused C kernel.**

- **Bicubic** (`interp_mode=0`): Default. Keys a=-0.75, 4×4 stencil. ~7% attenuation
  at |k|=0.25.
- **Lanczos-3** (`interp_mode=1`): LUT-accelerated windowed sinc, 6×6 stencil.
  ~2-3% max error at Nyquist. Only ~1.5x slower than bicubic thanks to LUT.
  Still 4-5x faster than the cv2.remap production cubic pipeline.

The fused kernel replaces the multi-step Python/cv2 pipeline entirely (predictor
upsample → symmetric maps → image warp → single OpenMP pass). It is a hard
requirement like `libbulkxcorr2d`. Correctness verified against a vectorized Python
Lanczos-3 reference (not cv2.INTER_LANCZOS4, which is Lanczos-4/8-tap).

Implementation: `pivtools_cli/lib/fused_warp.c`, header: `fused_warp.h`.
Test suite: `manual_tools/test_fused_warp.py`.
Integration plan: `docs/fused_warp_kernel_plan.md`.

---

## Test Scripts

| Script | Purpose |
|--------|---------|
| `tools/test_whittaker_interpolation.py` | Whittaker windowed sinc vs bicubic/Lanczos/FFT |
| `tools/test_nufft_interpolation.py` | NUFFT (FINUFFT) vs bicubic/Lanczos/FFT |
| `tools/compare_pass1_vs_all.py` | Pass-1 vs warped-pass Sigma_xx comparison |

All scripts take a `<planes_dir>` argument pointing to a directory with
`warped_pass_1.mat` or `planes_pass_{N}.mat` files from ensemble PIV.
