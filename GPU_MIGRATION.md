# GPU Migration Strategy: Apple Silicon Hybrid Approach

> Reference document for migrating PIVTOOLs compute kernels to GPU on Apple Silicon (Mac Studio with 500GB unified RAM).
> The strategy is a **hybrid CPU+GPU approach** that leverages unified memory for zero-copy handoffs between GPU bulk operations and existing CPU code.

---

## Why Apple Silicon Unified Memory Changes Everything

Traditional GPU computing has two major bottlenecks:
1. **PCIe transfer latency** (CPU RAM <-> GPU VRAM): 10-30 GB/s, adds milliseconds per transfer
2. **Limited GPU VRAM**: Typically 8-48GB, requires careful memory management

Apple Silicon unified memory eliminates both:
- **Zero-copy transfers**: CPU and GPU access the same physical memory. A GPU-computed array can be read by C code (or NumPy) without any copy or serialization.
- **500GB capacity**: The entire dataset (all images, all correlation planes, all results) can live in GPU-accessible memory simultaneously.

This means even small GPU kernels are worth porting, since the overhead of "switching" between CPU and GPU is effectively zero.

---

## The Hybrid Split

The core principle: **GPU handles massively parallel bulk work; CPU keeps iterative/branching kernels that already work well.**

```
                        GPU (MLX)                          CPU (keep as-is)
                  ──────────────────                 ──────────────────────
Images in unified memory ─┐
                          |
              ┌─ Window extraction (gather)
              ├─ Taper/weight application
              ├─ Mean subtraction
              ├─ Batched FFT xcorr          ──>  Peak finding (existing C LM solver)
              ├─ fftshift + normalization        Outlier detection (existing Python)
              └─ Energy computation               Infilling (existing Python)
                                                  .mat I/O (existing Python)

Image warping (predictor-corrector) <────────  Predictor smoothing (gaussian_filter)
Image filters (gaussian, median, norm)
Statistics (mean, TKE, vorticity, etc.)

Ensemble accumulation (running sums) ──>  Gaussian fitting (existing C GSL solver)
                                          K-space fitting (existing Python/SciPy)
```

### Zero-Copy Handoff in Practice

```python
import mlx.core as mx
import numpy as np

# GPU: batched FFT cross-correlation
corr_planes_mlx = mx.fft.fftshift(mx.fft.ifft2(mx.fft.fft2(windows_A) * mx.conj(mx.fft.fft2(windows_B))))

# "Transfer" to CPU = pointer cast, ~0 ns on unified memory
corr_planes_np = np.array(corr_planes_mlx, copy=False)

# CPU: existing C peak finder reads same memory, completely unchanged
self.lib.bulkxcorr2d(...)
```

The C extensions don't need to know a GPU was involved.

---

## Framework Choice

| | MLX (Apple) | Metal Compute Shaders | PyTorch MPS |
|---|---|---|---|
| API feel | NumPy-like | C++/Metal Shading Language | PyTorch tensors |
| FFT support | Yes (`mlx.core.fft`) | Via vkFFT or custom | Yes (`torch.fft`) |
| Porting effort | Low (Python) | High (MSL rewrite) | Medium |
| Fine control | Medium | Full | Medium |
| Best for | Array ops, FFTs, reductions | Custom kernels (peak fitting) | ML-adjacent workloads |

**Recommendation**: MLX for the bulk array operations (FFTs, filters, image warping, statistics). Metal Compute Shaders only if a custom kernel is needed (e.g., GPU peak fitting in a later phase).

---

## What Goes to GPU

### 1. FFT Cross-Correlation (`xcorr.c` -> MLX batched FFT)

**Current**: FFTW3f single-precision FFTs, one per window, OpenMP parallel across windows in `PIV_2d_cross_correlate.c`.

**Proposed**: Single batched FFT call for ALL windows across ALL images simultaneously.

```
Current (CPU, OpenMP):
  for each window (OpenMP parallel):
    extract sub-image -> taper -> mean subtract -> FFT -> conj multiply -> IFFT -> fftshift

Proposed (GPU):
  Batch extract ALL windows at once           (gather op, all N_images x N_windows)
  Batch apply ALL tapers at once              (element-wise multiply)
  Batch subtract ALL means at once            (reduction + broadcast)
  Batch FFT ALL windows at once               (single mx.fft.fft2 call)
  Batch conj multiply ALL at once             (element-wise)
  Batch IFFT ALL at once                      (single mx.fft.ifft2 call)
  Batch fftshift + normalize ALL at once      (element-wise)
  -> hand off to CPU peak finder (zero-copy)
```

For a typical 50x50 window grid (2500 windows) x 10 images = 25,000 FFTs in a single GPU dispatch vs. 25,000 sequential (OpenMP-parallelized) FFTW calls.

**Speedup estimate**: 10-50x on xcorr. Since xcorr is 60-80% of total PIV runtime, this translates to **3-8x overall**.

### 2. Image Warping / Predictor-Corrector (`cv2.remap` -> GPU texture sampling)

**Current**: `cv2.remap()` with INTER_CUBIC, called per-image per-component in a Python loop (`cpu_instantaneous.py:667-722`).

```python
# Current: sequential loop
for i in range(N):
    for d in range(2):
        delta_ab_dense[i, ..., d] = cv2.remap(...)
```

**Proposed**: Single batched GPU call for all images and components.

```python
# Proposed: batched GPU operation
delta_ab_dense = mx_remap(delta_ab_old, map_x, map_y)  # (N, H, W, 2) at once
```

Texture sampling / interpolation is literally what GPU hardware was designed for (dedicated texture mapping units).

**Speedup estimate**: 5-20x on the predictor-corrector step.

### 3. Image Filters (scipy.ndimage -> GPU convolution)

**Current**: `gaussian_filter`, `median_filter` from scipy.ndimage, applied per-image via Dask `map_blocks` in `dask_pipeline.py:apply_all_filters_slim()`.

**Proposed**: MLX convolution or Metal Performance Shaders (MPS has built-in `MPSImageGaussianBlur`, `MPSImageMedian`).

```python
# Current
from scipy.ndimage import gaussian_filter
filtered = gaussian_filter(image, sigma=1.0)

# Proposed: all N images at once
filtered_batch = mx.conv2d(all_images, gaussian_kernel)
```

**Speedup estimate**: 5-15x on filtering.

### 4. Statistics (NumPy -> MLX)

**Current**: NumPy operations in `instantaneous_statistics.py` - mean, variance, finite differences for vorticity/divergence, ProcessPoolExecutor for frame-level parallelism.

**Proposed**: Near-mechanical translation to MLX.

```python
# Current
mean_ux = np.nanmean(all_ux, axis=0)
UU_stress = np.nanmean((all_ux - mean_ux)**2, axis=0)

# Proposed
mean_ux = mx.nanmean(all_ux, axis=0)
UU_stress = mx.nanmean((all_ux - mean_ux)**2, axis=0)
```

All operations (mean velocity, Reynolds stress, TKE, vorticity, divergence) are element-wise or simple reductions.

**Speedup estimate**: 5-20x on statistics computation.

### 5. Ensemble Accumulation (`bulkxcorr2d_accumulate` -> GPU running sums)

**Current**: C code in `PIV_2d_cross_correlate.c`, parallel over windows, sequential over images, accumulating correlation sums.

**Proposed**: Keep running GPU tensor, accumulate directly.

```python
# Accumulation stays on GPU the whole time
for batch in image_batches:
    corr_planes = gpu_xcorr(batch)        # GPU FFT
    corr_sum_gpu += corr_planes           # GPU addition, stays in GPU memory

# Final fitting: zero-copy handoff to existing C GSL solver
corr_sum_np = np.array(corr_sum_gpu, copy=False)
fit_stacked_gaussian_batch_export(corr_sum_np, ...)  # existing C code, unchanged
```

With 500GB unified RAM, the accumulated correlation planes for ALL windows live in memory permanently.

**Speedup estimate**: 10-30x on ensemble accumulation.

---

## What Stays on CPU (Unchanged)

| Component | Why keep on CPU |
|---|---|
| `peak_locate_lm.c` | Small data (5x5 region), iterative LM with per-window convergence, already fast |
| `marquadt_gaussian.c` | 16-param GSL solver, variable convergence causes GPU thread divergence |
| `kspace_fitting.py` | scipy.optimize per-window, Python overhead dominates |
| Outlier detection | Neighborhood comparisons with branching logic |
| Infilling | Sparse interpolation, small fraction of runtime |
| Dask orchestration | Still useful for multi-camera, batch management, progress tracking |
| All I/O | Disk-bound, not compute-bound |
| Config, GUI, Flask | Not compute |

These components benefit from unified memory because they read GPU-produced data with zero copy, but their actual computation stays on CPU.

---

## Implementation Architecture

### Entry Point: Factory Pattern (Already Exists)

The existing `factory.py` and `processing.backend` config key support this directly:

```python
# factory.py
def make_correlator_backend(config):
    if config.backend == "cpu":
        return InstantaneousCorrelatorCPU(config)    # existing, unchanged
    elif config.backend == "gpu":
        return InstantaneousCorrelatorApple(config)  # new hybrid class
```

Config change to enable:
```yaml
processing:
  backend: "gpu"    # was "cpu"
```

### New File: `piv/piv_backend/apple_instantaneous.py`

```python
class InstantaneousCorrelatorApple(CrossCorrelator):
    """
    Hybrid GPU+CPU correlator for Apple Silicon.

    GPU (MLX): window extraction, FFT xcorr, image warping, filtering
    CPU (existing C): peak finding, Gaussian fitting

    Uses unified memory for zero-copy handoffs between GPU and CPU stages.
    """

    def correlate_batch(self, images, config, vector_masks=None):
        # Same interface as InstantaneousCorrelatorCPU
        # Same return type (PIVResult)
        # Internal implementation uses MLX for bulk ops
        ...
```

### What Changes vs. What Doesn't

**New files:**
- `piv/piv_backend/apple_instantaneous.py` - hybrid GPU correlator
- `piv/piv_backend/apple_ensemble.py` - hybrid GPU ensemble correlator (Phase 5)
- `piv/piv_backend/mlx_ops.py` - shared MLX utility functions (batched FFT, window extraction, etc.)

**Modified files (minimal):**
- `piv/piv_backend/factory.py` - add `"gpu"` case to route to new classes
- `pivtools_core/config.py` - `backend` property already exists, may need validation for "gpu"

**Unchanged files (everything else):**
- `PIV_2d_cross_correlate.c`, `xcorr.c`, `peak_locate_lm.c`, `marquadt_gaussian.c`, `interp2custom.c`
- `instantaneous.py`, `ensemble.py`, `dask_pipeline.py`
- All GUI code, all calibration code, all CLI code
- `vector_loading.py`, `config.py` (core logic), `paths.py`

---

## Phased Rollout

| Phase | What | Effort | Cumulative Overall Speedup |
|-------|------|--------|---------------------------|
| **1** | GPU FFT cross-correlation | 2 weeks | **3-8x** |
| **2** | GPU image warping (predictor-corrector) | 1 week | **4-12x** |
| **3** | GPU image filters | 1 week | **5-13x** |
| **4** | GPU statistics | 1 week | **5-15x** |
| **5** | GPU ensemble accumulation | 2 weeks | **8-20x** (ensemble mode) |

**Phase 1 alone delivers most of the benefit.** Each phase is independently useful, independently testable, and the CPU fallback always works.

### Phase 1 Detail: GPU FFT Cross-Correlation

This is the highest-impact change. The new correlator class would:

1. **Extract all windows** from all images as a batched gather operation:
   ```python
   # Shape: (N_images * N_windows, win_h, win_w)
   windows_A = extract_all_windows(images_A, win_ctrs_x, win_ctrs_y, win_size)
   windows_B = extract_all_windows(images_B, win_ctrs_x, win_ctrs_y, win_size)
   ```

2. **Apply tapers and mean-subtract** (element-wise, batched):
   ```python
   windows_A *= taper_weight  # broadcast over batch dim
   windows_B *= taper_weight
   windows_A -= mx.mean(windows_A, axis=(-2,-1), keepdims=True)
   windows_B -= mx.mean(windows_B, axis=(-2,-1), keepdims=True)
   ```

3. **Batched FFT cross-correlation** (single call):
   ```python
   FA = mx.fft.fft2(windows_A)
   FB = mx.fft.fft2(windows_B)
   corr = mx.fft.fftshift(mx.fft.ifft2(FA * mx.conj(FB)).real, axes=(-2,-1))
   ```

4. **Normalize** (element-wise, batched):
   ```python
   energy_A = mx.sum(windows_A**2, axis=(-2,-1), keepdims=True)
   energy_B = mx.sum(windows_B**2, axis=(-2,-1), keepdims=True)
   corr *= correl_weight / mx.sqrt(energy_A * energy_B)
   ```

5. **Hand off to existing C peak finder** (zero-copy):
   ```python
   corr_np = np.array(corr, copy=False)  # zero-copy on unified memory
   # Reshape to match existing C code's expected layout
   # Call existing peak_locate_lm via ctypes - completely unchanged
   ```

---

## Dependencies

### Required
- **MLX**: `pip install mlx` (Apple's ML framework, MIT license)
  - NumPy-like API
  - Metal GPU backend optimized for Apple Silicon
  - Lazy evaluation
  - FFT support via `mlx.core.fft`
  - macOS 13.5+ required

### Optional (for advanced phases)
- **Metal Compute Shaders**: Only if custom GPU kernels needed (e.g., GPU peak fitting)
- **PyTorch MPS**: Alternative to MLX if torch ecosystem integration is preferred

### Not Required
- CUDA (not available on macOS)
- OpenCL (deprecated on macOS)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MLX FFT precision differs from FFTW | Validate against CPU results with tolerance; single-precision throughout already |
| MLX missing a needed operation | Fall back to NumPy (zero-copy) for that operation |
| Performance regression on small datasets | Keep CPU backend as default; GPU only beneficial when N_windows > ~100 |
| macOS-only lock-in | GPU backend is opt-in; CPU backend unchanged and works everywhere |
| MLX API instability | Pin MLX version in requirements; API is stabilizing (1.0+) |

---

## Validation Strategy

Each phase should include:
1. **Numerical validation**: Compare GPU output against CPU output for the same input, assert max absolute difference < tolerance (1e-5 for float32)
2. **Performance benchmarking**: Time GPU vs CPU path on representative datasets (small: 100 pairs, medium: 1000 pairs, large: 10000 pairs)
3. **Regression tests**: Existing test suite must pass with `backend: "gpu"` producing equivalent results to `backend: "cpu"`

---

## Notes

- The `processing.backend` config key already exists and is set to `"cpu"` by default
- The `factory.py` pattern already supports pluggable backends
- FFTW wisdom caching (`xcorr_cache.c`) is not needed for GPU FFTs (Metal compiles shaders at first use, then caches automatically)
- `gc.collect()` on Dask workers causes SIGSEGV with FFTW - this issue disappears for GPU FFT operations since FFTW is not involved
- The `interp2custom.c` column-major indexing quirk is irrelevant if replaced with GPU texture sampling
