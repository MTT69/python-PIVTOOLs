# Ensemble PIV Pipeline: Code Review & Recommendations

This document provides a comprehensive code review of the ensemble PIV pipeline, identifying legacy code, unused functions, performance bottlenecks, and optimization opportunities. The primary focus is on peak fitting speed improvements.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Legacy Code Investigation](#2-legacy-code-investigation)
3. [Function Definition Analysis](#3-function-definition-analysis)
4. [Unused Code Identification](#4-unused-code-identification)
5. [Performance Bottlenecks](#5-performance-bottlenecks)
6. [Deep Dive: Peak Fitting Speed](#6-deep-dive-peak-fitting-speed)
7. [Parallelization Recommendations](#7-parallelization-recommendations)
8. [Code Quality Issues](#8-code-quality-issues)
9. [Recommended Refactoring](#9-recommended-refactoring)
10. [Action Items (Prioritized)](#10-action-items-prioritized)

---

## 1. Executive Summary

### Overall Code Health Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Architecture | Good | Clean separation of concerns, factory pattern |
| Performance | Needs Work | Peak fitting is sequential, not batched |
| Legacy Code | Present | MATLAB files in Python directories |
| Documentation | Mixed | Some excellent, some missing |
| Test Coverage | Unknown | No tests observed in explored files |

### Top Priority Recommendations

1. **HIGH**: Batch peak fitting to C library (current: 1 window per call)
2. **HIGH**: Remove MATLAB `.m` files from Python directories
3. **MEDIUM**: Consolidate duplicate window center computations
4. **MEDIUM**: Improve outlier detection vectorization
5. **LOW**: Add type hints to public APIs

---

## 2. Legacy Code Investigation

### MATLAB Files in Python Directories

The following MATLAB files exist in Python source directories and should be removed or moved to a separate `matlab_reference/` directory:

| File | Location | Status |
|------|----------|--------|
| `EnsemblePIV.m` | `pivtools_cli/piv/piv_backend/` | Legacy - remove or archive |
| `PIV_2D_wdef_ensemble.m` | `pivtools_cli/piv/piv_backend/` | Legacy - remove or archive |
| `evaluate_correlation_planes.m` | `pivtools_cli/piv/piv_backend/` | Legacy - remove or archive |
| `process_correlation_planes.m` | `pivtools_cli/piv/piv_backend/` | Legacy - remove or archive |
| `POD_filter.m` | `pivtools_cli/preprocessing/` | Legacy - remove or archive |
| `Time_filter.m` | `pivtools_cli/preprocessing/` | Legacy - remove or archive |

**Recommendation**: Create `docs/matlab_reference/` and move these files there with a README explaining they are historical reference implementations.

### Duplicate C Library Implementations

Three different peak fitting implementations exist:

| File | GSL Required | Status | Notes |
|------|--------------|--------|-------|
| `marquadt_gaussian.c` | Yes | **ACTIVE** | Used by ensemble PIV |
| `peak_locate_gsl.c` | Yes | **UNUSED?** | Alternative GSL implementation |
| `peak_locate_lm.c` | No | **UNUSED?** | GSL-free fallback |

**Investigation Findings**:

```bash
# Search for usage of peak_locate_gsl
$ grep -r "peak_locate_gsl" pivtools_cli/
# No results found in Python code

# Search for usage of peak_locate_lm
$ grep -r "peak_locate_lm" pivtools_cli/
# No results found in Python code
```

**Recommendation**:
- Confirm `peak_locate_gsl.c` and `peak_locate_lm.c` are truly unused
- If unused, remove from build and archive
- If used by instantaneous mode, document clearly

### Instantaneous vs Ensemble Correlator Duplication

Two separate correlator implementations exist with significant overlap:

| Feature | `cpu_instantaneous.py` | `cpu_ensemble.py` |
|---------|----------------------|------------------|
| Window centers | Own implementation | Uses `window_utils.py` |
| C library loading | `libbulkxcorr2d.so` | `libbulkxcorr2d.so` (same!) |
| Window weights | Computed locally | Uses base class `_window_weight_fun` |
| Cache mechanism | Separate | Separate |

**Code Duplication Examples**:

1. **Library loading** is duplicated in both files
2. **Window center computation** exists in 3 places:
   - `cpu_instantaneous.py` (lines 515-549)
   - `cpu_ensemble.py:_compute_window_centres_ensemble()` (lines 359-397)
   - `window_utils.py:compute_window_centers()` (lines 143-272)

**Recommendation**: Refactor to use `window_utils.py` consistently in both correlators.

---

## 3. Function Definition Analysis

### Similar Functions Across Files

#### Window Center Computation (3 implementations)

**Location 1**: `cpu_instantaneous.py` (embedded in correlate method)
```python
# Inline computation, not factored out
first_ctr_x = (win_width - 1) / 2.0
last_ctr_x = W - (win_width + 1) / 2.0
n_win_x = int(np.floor((last_ctr_x - first_ctr_x) / win_spacing_x)) + 1
```

**Location 2**: `cpu_ensemble.py:_compute_window_centres_ensemble()`
```python
def _compute_window_centres_ensemble(self, pass_idx, config):
    # Calls window_utils but adds extra processing
    result = compute_window_centers(...)
    return (result.win_spacing_x, result.win_spacing_y, ...)
```

**Location 3**: `window_utils.py:compute_window_centers()`
```python
def compute_window_centers(image_shape, window_size, overlap, validate=True):
    # Centralized implementation - this should be THE source of truth
    # ...
    return WindowCenterResult(...)
```

**Issue**: `cpu_instantaneous.py` doesn't use `window_utils.py` at all.

#### Library Loading (2 implementations)

**cpu_ensemble.py** (lines 151-217):
```python
@classmethod
def _load_libraries(cls):
    """Load C libraries once per process to avoid DLL thrashing."""
    # ... loads libmarquadt.so and libbulkxcorr2d.so
```

**cpu_instantaneous.py** (similar pattern, different location):
```python
def _load_library(self):
    # ... loads libbulkxcorr2d.so only
```

**Recommendation**: Create shared `lib_loader.py` module.

### Redundant Caching Mechanisms

Both correlators maintain separate caches with overlapping content:

```python
# cpu_ensemble.py cache
self.win_ctrs_x          # Per-pass window centers
self.win_ctrs_y
self.cached_dense_maps   # Interpolation maps
self.cached_predictor_maps
self.win_weights_A       # Window weights
self.win_weights_B

# cpu_instantaneous.py cache
self.win_ctrs_x          # Same concept
self.win_ctrs_y
self.cached_interp_maps  # Similar concept
self.window_weights      # Same concept
```

**Recommendation**: Create `CorrelatorCache` dataclass shared between both.

### Inconsistent Naming Conventions

| Concept | cpu_ensemble.py | cpu_instantaneous.py | window_utils.py |
|---------|-----------------|---------------------|-----------------|
| Window centers X | `win_ctrs_x` | `win_ctrs_x` | `win_ctrs_x` |
| Window spacing | `win_spacing_x` | `spacing_x` | `win_spacing_x` |
| Number of windows | `n_win_x` | `num_win_x` | `n_win_x` |
| Pass index | `pass_idx` | N/A | N/A |

**Minor inconsistency**, but `spacing_x` vs `win_spacing_x` should be unified.

---

## 4. Unused Code Identification

### Functions Never Called

**In `cpu_ensemble.py`**:

```python
def correlate_batch(self, images, config, vector_masks=None):
    """
    Not used for ensemble PIV - use correlate_batch_for_accumulation instead.
    """
    raise NotImplementedError(...)  # Line 500-503
```

**Status**: This is a placeholder to satisfy abstract base class. Acceptable but should have `@abstractmethod` in base class or be removed.

### Dead Configuration Options

In configuration, some options appear unused or partially implemented:

```yaml
# Suspected unused/incomplete
ensemble_piv:
  debug_correlation_planes: false  # Check if actually used
  gpu_backend: false               # GPU ensemble not implemented
```

**Investigation needed**: Grep for these config keys in codebase.

### Commented-Out Code Blocks

**In `xcorr.c`** (line 122):
```c
//fftwf_plan_with_nthreads(1);  // Commented out threading config
```

**In multiple Python files**: Search for `# TODO` and `# FIXME` comments.

### Legacy Fallback Paths

**In `gaussian_fitting.py`** (lines 720-732):
```python
elif predictor_field is not None:
    # Fallback to raw predictor if smoothed not available (shouldn't happen)
    logging.warning(
        f"Pass {pass_idx + 1}: Smoothed predictor not available, "
        f"falling back to raw predictor field"
    )
```

**Status**: This fallback exists but "shouldn't happen" - indicates potential dead code or error handling that could be removed with assertions.

---

## 5. Performance Bottlenecks

### Peak Fitting: The #1 Bottleneck

**Current Architecture**:
```
                    CURRENT: SEQUENTIAL PER-WINDOW
    ================================================================

    Python Loop:
    for i in range(n_windows):           # 100,000+ windows
        window = extract_window(i)       # ~100 μs
        guess = build_initial_guess()    # ~50 μs
        result = C_library_call()        # ~500 μs per window!
        validate(result)                 # ~20 μs
                                         ─────────
                                         ~670 μs × 100,000 = 67 seconds!

    Breakdown for 100,000 windows:
    - C library calls: 50+ seconds
    - Python overhead: 15+ seconds
    - Data movement: 2+ seconds
```

**Profiling Estimate**:
- Per-window C call overhead: ~50 μs (ctypes marshalling)
- LM iterations per window: 10-100 (depending on convergence)
- LM iteration cost: ~5-50 μs
- **Total per-window**: 500-5000 μs

**For a 4K image with 32x32 windows at 50% overlap**:
- Grid size: ~250 × 125 = 31,250 windows
- Time estimate: 31,250 × 1 ms = **31 seconds** (fitting only!)

### Memory Allocation Patterns

**Identified Issues**:

1. **Per-window allocations in C library** (`marquadt_gaussian.c`):
```c
gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(T, &fdf_params, m, p);
gsl_vector *wts = gsl_vector_alloc(m);
// ... used once, then freed
gsl_vector_free(wts);
gsl_multifit_nlinear_free(work);
```

2. **Correlation plane copies** (`single_pass_accumulator.py:765-768`):
```python
return {
    "corr_AA_sum": correl_AA_sum.copy(),  # Necessary copy
    "corr_BB_sum": correl_BB_sum.copy(),  # Because buffers are reused
    "corr_AB_sum": correl_AB_sum.copy(),
```

3. **Repeated np.ascontiguousarray calls** (`cpu_ensemble.py`):
```python
np.ascontiguousarray(image_a, dtype=np.float32)  # Often already contiguous
```

### Unnecessary Data Copies

**In finalize_pass() sigma interpolation**:
```python
sigma_fields = {
    'sig_AB_x': prev_pass.sig_AB_x.copy().astype(np.float32),  # Copy + cast
    'sig_AB_y': prev_pass.sig_AB_y.copy().astype(np.float32),
    # ...
}
```

Could be optimized to:
```python
sigma_fields = {
    'sig_AB_x': np.ascontiguousarray(prev_pass.sig_AB_x, dtype=np.float32),
    # Only copies if needed
}
```

### Suboptimal Dask Task Graph Structure

**Current**: Scatter → Submit → Gather pattern is good, but:

```python
# single_pass_accumulator.py:406-438
for i, worker in enumerate(workers):
    # Each scatter creates a separate task
    scattered = client.scatter(chunk_dict, workers=[worker])
    scattered_chunks.append((scattered, worker))

# Then submit in second loop
for scattered, worker in scattered_chunks:
    fut = client.submit(_fit_windows_batch_from_scattered, scattered, ...)
```

**Potential issue**: Two loops where one could suffice. Consider:
```python
# Combined scatter and submit
futures = []
for i, worker in enumerate(workers):
    chunk_dict = build_chunk(i)
    fut = client.submit(_fit_from_chunk, chunk_dict, workers=[worker])
    futures.append(fut)
```

---

## 6. Deep Dive: Peak Fitting Speed

### Current Implementation Analysis

**File**: `gaussian_fitting.py:286-418`

```python
def _fit_windows_batch_optimized(...):
    marquadt_lib = _load_marquadt_lib()

    # Loop over non-masked windows
    for i, idx in enumerate(valid_indices):  # SEQUENTIAL!
        # Extract window from correlation planes
        AA_win = _get_window(AA_chunk, idx, win_size)
        BB_win = _get_window(BB_chunk, idx, win_size)
        AB_win = _get_window(AB_chunk, idx, win_size)

        # Build initial guess
        initial_guess, real_corr = _build_initial_guess(...)

        # *** THE BOTTLENECK: One C call per window ***
        marquadt_lib.fit_stacked_gaussian_export(
            ctypes.c_size_t(win_size[0] * win_size[1]),
            X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            real_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            initial_guess.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_status.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
```

### C Library Current Interface

**File**: `marquadt_gaussian.c`

```c
// Current signature: ONE window at a time
int fit_stacked_gaussian_export(
    size_t n,                      // Points per plane (e.g., 64*64 = 4096)
    const double *X1,              // Y-coordinates (length n)
    const double *X2,              // X-coordinates (length n)
    const double *y,               // Stacked data [AA; BB; AB] (length 3*n)
    const double *initial_guess,   // Initial params (length 13)
    double *out_params,            // Output params (length 13)
    int *out_status                // Output status
);
```

### Optimization Strategy 1: Batch C Library Extension

**Proposed Interface**:

```c
// NEW: Batch multiple windows in single call
int fit_stacked_gaussian_batch(
    size_t n_windows,              // Number of windows to fit (e.g., 1000)
    size_t n_points,               // Points per window (e.g., 4096)
    const double *X1,              // Shared Y-coords (length n_points)
    const double *X2,              // Shared X-coords (length n_points)
    const double *y,               // Stacked data (length n_windows * 3 * n_points)
    const double *initial,         // Initial params (length n_windows * 13)
    double *out_params,            // Output params (length n_windows * 13)
    int *out_status,               // Output status (length n_windows)
    int n_threads                  // OpenMP threads to use
);
```

**Implementation Sketch**:

```c
#include <omp.h>

int fit_stacked_gaussian_batch(
    size_t n_windows,
    size_t n_points,
    const double *X1,
    const double *X2,
    const double *y,
    const double *initial,
    double *out_params,
    int *out_status,
    int n_threads
) {
    omp_set_num_threads(n_threads);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t w = 0; w < n_windows; w++) {
        // Pointers into batch arrays
        const double *y_w = y + w * 3 * n_points;
        const double *init_w = initial + w * 13;
        double *out_w = out_params + w * 13;
        int *status_w = out_status + w;

        // Call existing single-window implementation
        fit_stacked_gaussian(n_points, X1, X2, y_w, init_w, out_w, status_w);
    }

    return 0;
}
```

**Expected Speedup**:
- Removes Python loop overhead: ~15 seconds saved
- OpenMP parallelization: 4-8x speedup depending on cores
- **Total**: 5-15x improvement

### Optimization Strategy 2: Pre-allocated GSL Workspaces

**Current Issue**: Each call allocates/frees GSL workspace:

```c
gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(...);
// ... use work ...
gsl_multifit_nlinear_free(work);
```

**Improvement**: Thread-local pre-allocated workspaces:

```c
#include <omp.h>

static __thread gsl_multifit_nlinear_workspace *tls_work = NULL;
static __thread size_t tls_work_size = 0;

int fit_stacked_gaussian_batch_prealloc(...) {
    #pragma omp parallel
    {
        // Initialize thread-local workspace once
        if (tls_work == NULL || tls_work_size != n_points * 3) {
            if (tls_work) gsl_multifit_nlinear_free(tls_work);
            tls_work = gsl_multifit_nlinear_alloc(T, &params, n_points * 3, 13);
            tls_work_size = n_points * 3;
        }

        #pragma omp for schedule(dynamic, 64)
        for (size_t w = 0; w < n_windows; w++) {
            // Reuse tls_work instead of allocating
            fit_stacked_gaussian_reuse_workspace(
                n_points, X1, X2, y + w*3*n_points, ...
                tls_work  // Pass pre-allocated workspace
            );
        }
    }
}
```

### Optimization Strategy 3: GPU Acceleration (CUDA)

For maximum speedup, GPU-accelerated batch fitting:

**Architecture**:
```
                    GPU BATCH FITTING
    ================================================================

    Host (CPU):                    Device (GPU):
    +------------------+           +---------------------------+
    | Prepare batches  |           | CUDA Kernels:             |
    | - Pack data      |  ------>  | - residual_kernel()       |
    | - Transfer H2D   |           | - jacobian_kernel()       |
    +------------------+           | - solve_kernel()          |
                                   +---------------------------+
    +------------------+           |
    | Unpack results   |  <------  | Results D2H
    +------------------+

    Speedup: 20-100x for batch sizes > 1000
```

**cuSOLVER Batch LM Sketch**:

```python
# Python wrapper using CuPy
import cupy as cp
from cupy.cuda import cusolver

def fit_windows_gpu_batch(AA_gpu, BB_gpu, AB_gpu, initial_gpu, n_windows):
    """
    GPU-accelerated batch Gaussian fitting using cuSOLVER batched least squares.
    """
    # Stack correlation data on GPU
    batch_data = cp.concatenate([AA_gpu, BB_gpu, AB_gpu], axis=-1)  # (n_windows, 3*n_points)

    # Initialize parameters on GPU
    params = initial_gpu.copy()  # (n_windows, 13)

    # Batch LM iterations
    for iteration in range(max_iter):
        # Compute residuals for all windows in parallel
        residuals = compute_residuals_kernel(batch_data, params)

        # Compute Jacobians for all windows
        jacobians = compute_jacobian_kernel(batch_data, params)

        # Solve (J^T J + lambda I) delta = -J^T r for all windows
        # Using cuSOLVER batched Cholesky
        delta = cusolver_batch_solve(jacobians, residuals)

        # Update parameters
        params += delta

        # Check convergence (vectorized)
        if cp.all(cp.linalg.norm(delta, axis=1) < tol):
            break

    return params.get()  # Transfer back to CPU
```

**Expected Speedup**: 20-100x depending on batch size and GPU.

### Optimization Strategy 4: Vectorized Python Fallback (JAX)

For systems without GPU or custom C modifications:

```python
import jax
import jax.numpy as jnp
from jax import vmap, jit

@jit
def gaussian_residual(params, X1, X2, y_stack):
    """Vectorized Gaussian residual computation."""
    amp_A, amp_B, amp_AB = params[0], params[1], params[2]
    sx_A, sy_A, sxy_A = params[3], params[4], params[5]
    sx_AB, sy_AB, sxy_AB = params[6], params[7], params[8]
    x0_A, y0_A = params[9], params[10]
    x0_AB, y0_AB = params[11], params[12]

    # Compute shape A (shared by AA, BB)
    dx_A = X1 - x0_A
    dy_A = X2 - y0_A
    # ... Mahalanobis distance computation ...
    exp_A = jnp.exp(-0.5 * quad_A)

    # Residuals
    r_AA = amp_A * exp_A - y_stack[:n]
    r_BB = amp_B * exp_A - y_stack[n:2*n]
    r_AB = amp_AB * exp_AB - y_stack[2*n:]

    return jnp.concatenate([r_AA, r_BB, r_AB])

# Vectorize over batch dimension
batched_residual = vmap(gaussian_residual, in_axes=(0, None, None, 0))

@jit
def batch_lm_step(params_batch, X1, X2, y_batch, lambda_):
    """Single LM step for all windows."""
    residuals = batched_residual(params_batch, X1, X2, y_batch)
    jacobians = jax.jacfwd(batched_residual)(params_batch, X1, X2, y_batch)
    # ... solve LM system ...
    return params_batch + delta

def fit_batch_jax(AA, BB, AB, initial_guess, max_iter=100, tol=1e-8):
    """Full batch fitting using JAX."""
    params = initial_guess
    for i in range(max_iter):
        params_new = batch_lm_step(params, X1, X2, y_stack, lambda_)
        if jnp.all(jnp.abs(params_new - params) < tol):
            break
        params = params_new
    return params
```

**Expected Speedup**: 5-20x on CPU, 50-200x with JAX GPU backend.

### Comparison of Optimization Strategies

| Strategy | Effort | Speedup | Compatibility | Notes |
|----------|--------|---------|---------------|-------|
| Batch C + OpenMP | Medium | 5-15x | All platforms | Recommended first |
| Pre-alloc workspaces | Low | 1.2-1.5x | All platforms | Easy add-on |
| GPU (cuSOLVER) | High | 20-100x | CUDA GPUs | Best for HPC |
| JAX vectorized | Medium | 5-20x | All platforms | Fallback option |

---

## 7. Parallelization Recommendations

### Correlation Workers: Better Load Balancing

**Current Issue**: Round-robin assignment doesn't account for masked windows.

```python
# Current: Equal chunk sizes
windows_per_worker = (total_windows + num_workers - 1) // num_workers
for i, worker in enumerate(workers):
    start_idx = i * windows_per_worker
    end_idx = min((i + 1) * windows_per_worker, total_windows)
```

**Improvement**: Mask-aware load balancing:

```python
# Count valid (non-masked) windows per region
valid_counts = []
for i in range(num_workers):
    start = i * windows_per_worker
    end = min((i + 1) * windows_per_worker, total_windows)
    valid_counts.append(np.sum(~mask_flat[start:end]))

# Rebalance based on valid counts
# ... adjust boundaries to equalize work ...
```

### Filter Workers: Additional Parallelism

**Current**: One batch per filter worker at a time.

**Potential**: Pipeline POD SVD computation with filter application:

```
Current:          [=====POD SVD=====][===Apply Filters===]

Potential:        [=====POD SVD=====]
                          [===Apply Filters===]
                                  (start next batch SVD while applying)
```

**Note**: This requires careful memory management but could hide SVD latency.

### Memory Reduction: Streaming Sigma Interpolation

**Current**: All 6 sigma fields interpolated before fitting starts.

**Improvement**: Interpolate on-demand during fitting:

```python
# Instead of pre-computing all
for key in sigma_keys:
    sigma_dict[key] = interpolate(prev_pass.get(key))

# Interpolate lazily
class LazySigmaInterpolator:
    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = interpolate(self._prev_pass.get(key))
        return self._cache[key]
```

---

## 8. Code Quality Issues

### Magic Numbers (Hardcoded Thresholds)

| Location | Value | Context | Should Be |
|----------|-------|---------|-----------|
| `gaussian_fitting.py:205` | `1e-12` | AA_central threshold | `config.ensemble_amplitude_floor` |
| `gaussian_fitting.py:502` | `1e-6` | Peak value floor | `MIN_PEAK_VALUE` constant |
| `marquadt_gaussian.c:35-45` | `1e-5`, `1e-9`, etc. | Parameter clamping | Named constants |
| `outlier_detection.py:102` | `0.2` | MAD epsilon | `config.outlier_epsilon` |

**Recommendation**: Create `constants.py`:

```python
# pivtools_cli/piv/constants.py
MIN_AMPLITUDE = 1e-12
MIN_SIGMA = 1e-9
MAD_EPSILON = 0.2
PEAK_VALUE_FLOOR = 1e-6
```

### Missing Type Hints

**Example** (`single_pass_accumulator.py:306`):

```python
# Current
def finalize_pass(self, pass_idx, client, scattered_cache, predictor_field=None, output_path=None):

# Should be
def finalize_pass(
    self,
    pass_idx: int,
    client: Client,
    scattered_cache: dict,
    predictor_field: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> PIVEnsemblePassResult:
```

### Inconsistent Error Handling

**Pattern 1**: Silent failure with logging:
```python
except Exception as e:
    logging.error("Error in correlation: %s", e)
    traceback.print_exc()
    continue  # Silently skip
```

**Pattern 2**: Raise with message:
```python
if not filepath.exists():
    raise FileNotFoundError(f"Ensemble result file not found: {filepath}")
```

**Recommendation**: Standardize - critical errors should raise, recoverable errors should log and continue with documented behavior.

### Documentation Gaps

| File | Issue |
|------|-------|
| `cpu_ensemble.py` | Missing class-level docstring explaining ensemble vs instantaneous |
| `gaussian_fitting.py` | Missing module-level docstring |
| `xcorr.c` | Missing function documentation for `multiply_conjugate` |

---

## 9. Recommended Refactoring

### Priority 1: Consolidate Window Center Computation

**Create** `pivtools_core/grid.py`:

```python
"""
Unified grid computation for all PIV modes.
All correlators should use these functions exclusively.
"""

from pivtools_core.window_utils import (
    compute_window_centers,
    compute_window_centers_single_mode,
    WindowCenterResult,
)

# Re-export for easy import
__all__ = [
    'compute_window_centers',
    'compute_window_centers_single_mode',
    'WindowCenterResult',
]
```

**Update**: `cpu_instantaneous.py` to use `grid.py`.

### Priority 2: Unify Correlator Base Class

**Enhance** `base.py`:

```python
class CrossCorrelator(ABC):
    """Base class for all PIV correlators."""

    # Class-level library cache (shared)
    _lib_cache: ClassVar[Dict[str, ctypes.CDLL]] = {}

    @classmethod
    def get_library(cls, name: str) -> ctypes.CDLL:
        """Load library with caching."""
        if name not in cls._lib_cache:
            cls._lib_cache[name] = cls._load_library_impl(name)
        return cls._lib_cache[name]

    @abstractmethod
    def correlate_batch(self, images: np.ndarray, config: Config) -> Any:
        """Correlate batch of image pairs."""
        pass

    def _window_weight_fun(self, win_size, win_type, sum_window=None):
        """Shared window weight computation."""
        # ... existing implementation ...
```

### Priority 3: Remove MATLAB Files

```bash
# Create archive directory
mkdir -p docs/matlab_reference/piv_backend
mkdir -p docs/matlab_reference/preprocessing

# Move files
mv pivtools_cli/piv/piv_backend/*.m docs/matlab_reference/piv_backend/
mv pivtools_cli/preprocessing/*.m docs/matlab_reference/preprocessing/

# Create README
cat > docs/matlab_reference/README.md << 'EOF'
# MATLAB Reference Implementations

These files are historical MATLAB implementations that served as references
during Python port development. They are NOT used by the production code.

## Files

### piv_backend/
- `EnsemblePIV.m` - Original ensemble PIV driver
- `PIV_2D_wdef_ensemble.m` - Window deformation ensemble
- `evaluate_correlation_planes.m` - Correlation evaluation
- `process_correlation_planes.m` - Correlation processing

### preprocessing/
- `POD_filter.m` - POD filter implementation
- `Time_filter.m` - Time filter implementation

## Note

For production code, see the Python implementations in:
- `pivtools_cli/piv/piv_backend/cpu_ensemble.py`
- `pivtools_cli/preprocessing/filters.py`
EOF
```

### Priority 4: Create Shared Constants Module

```python
# pivtools_cli/piv/constants.py
"""
Shared constants for PIV processing.

These should be the ONLY place magic numbers are defined.
"""

# Numerical stability
MIN_AMPLITUDE = 1e-12
MIN_SIGMA = 1e-9
MIN_PEAK_VALUE = 1e-6

# Outlier detection
MAD_EPSILON = 0.2
DEFAULT_OUTLIER_THRESHOLD = 2.0

# Fitting
MAX_LM_ITERATIONS = 100
LM_XTOL = 1e-8
LM_GTOL = 1e-8

# Validation
MAX_NORMALIZED_PEAK_HEIGHT = 1.0
MIN_NORMALIZED_PEAK_HEIGHT = 0.0
```

---

## 10. Action Items (Prioritized)

### HIGH Priority (Performance/Correctness Impact)

| # | Task | Effort | Impact | Files |
|---|------|--------|--------|-------|
| 1 | Implement batch C peak fitting with OpenMP | 2-3 days | 5-15x speedup | `marquadt_gaussian.c`, `gaussian_fitting.py` |
| 2 | Pre-allocate GSL workspaces per thread | 0.5 day | 1.2-1.5x speedup | `marquadt_gaussian.c` |
| 3 | Remove/archive MATLAB files | 0.5 day | Cleanup | `*.m` files |
| 4 | Fix `cpu_instantaneous.py` to use `window_utils.py` | 1 day | Code consistency | `cpu_instantaneous.py` |

### MEDIUM Priority (Code Quality)

| # | Task | Effort | Impact | Files |
|---|------|--------|--------|-------|
| 5 | Create shared `lib_loader.py` | 0.5 day | Reduce duplication | New file |
| 6 | Create `constants.py` for magic numbers | 0.5 day | Maintainability | New file |
| 7 | Add type hints to public APIs | 1-2 days | Developer experience | Multiple |
| 8 | Verify and remove unused C files | 0.5 day | Cleanup | `peak_locate_*.c` |
| 9 | Mask-aware load balancing | 1 day | Better parallelism | `single_pass_accumulator.py` |

### LOW Priority (Nice to Have)

| # | Task | Effort | Impact | Files |
|---|------|--------|--------|-------|
| 10 | Add comprehensive docstrings | 2 days | Documentation | Multiple |
| 11 | GPU batch fitting (CUDA) | 1-2 weeks | 20-100x speedup | New files |
| 12 | JAX vectorized fallback | 1 week | Alternative backend | New files |
| 13 | Unit tests for peak fitting | 2-3 days | Reliability | New test files |

---

## Appendix: Batch Peak Fitting Implementation Checklist

### C Library Changes

- [ ] Add `fit_stacked_gaussian_batch()` function
- [ ] Add OpenMP parallelization with `#pragma omp parallel for`
- [ ] Use thread-local GSL workspaces
- [ ] Update library exports in header file
- [ ] Add `n_threads` parameter handling
- [ ] Test with different batch sizes

### Python Wrapper Changes

- [ ] Add ctypes definition for new function
- [ ] Modify `_fit_windows_batch_optimized()` to call batch function
- [ ] Pack data into contiguous arrays for batch call
- [ ] Handle non-uniform mask distribution
- [ ] Add fallback to per-window for small batches

### Build System Changes

- [ ] Add OpenMP flags to compilation (`-fopenmp`)
- [ ] Update `setup.py` or `CMakeLists.txt`
- [ ] Test compilation on Linux, macOS, Windows

### Testing

- [ ] Verify numerical equivalence with per-window implementation
- [ ] Benchmark speedup on various grid sizes
- [ ] Test thread scaling (1, 2, 4, 8, 16 threads)
- [ ] Memory profiling to ensure no leaks

---

*Document generated: Analysis of PyPIVTools ensemble PIV pipeline*
*Focus: Peak fitting speed optimization and code quality improvements*
