# Ensemble PIV Pipeline: Technical Data Flow Documentation

This document provides a comprehensive technical overview of the ensemble PIV pipeline implementation in PyPIVTools. It covers the complete data flow from image loading through final result saving, with detailed explanations of the parallelization architecture, mathematical foundations, and key algorithms.

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Dask Parallelization Architecture](#2-dask-parallelization-architecture)
3. [Image Loading](#3-image-loading)
4. [Preprocessing & Filtering](#4-preprocessing--filtering)
5. [Single-Pass Ensemble Correlation](#5-single-pass-ensemble-correlation)
6. [Peak Fitting (Gaussian Sub-Pixel Localization)](#6-peak-fitting-gaussian-sub-pixel-localization)
7. [Multi-Pass Iteration](#7-multi-pass-iteration)
8. [Outlier Detection & Infilling](#8-outlier-detection--infilling)
9. [Result Saving](#9-result-saving)
10. [Key Data Structures](#10-key-data-structures)
11. [File Reference](#11-file-reference)

---

## 1. Overview & Architecture

### High-Level Pipeline Diagram

```
                              ENSEMBLE PIV PIPELINE
    ============================================================================

    +-----------------+     +-------------------+     +----------------------+
    |  Config YAML    |---->|  example.py       |---->| UnifiedBatchPipeline |
    |  (config.yaml)  |     |  (entry point)    |     | (batch_pipeline.py)  |
    +-----------------+     +-------------------+     +----------------------+
                                                               |
                                    +--------------------------+
                                    |
                                    v
    +===========================================================================+
    |                         PIPELINED BATCH PROCESSING                         |
    |                                                                           |
    |   +---------------+        +-------------------+        +---------------+ |
    |   | Filter Worker |  --->  | Correlation       |  --->  | Accumulator   | |
    |   | (POD, Time,   |        | Workers           |        | (Single-Pass) | |
    |   |  Spatial)     |        | (CPU Ensemble)    |        |               | |
    |   +---------------+        +-------------------+        +---------------+ |
    |         |                         |                           |          |
    |         |    OVERLAP: Filter      |                           |          |
    |         |    batch N while        |                           |          |
    |         |    correlating N-1      |                           |          |
    +===========================================================================+
                                    |
                                    v
    +===========================================================================+
    |                           PASS FINALIZATION                               |
    |                                                                           |
    |   +----------------+    +------------------+    +---------------------+   |
    |   | Mean Images    |--->| Background       |--->| Distributed         |   |
    |   | Computation    |    | Subtraction      |    | Gaussian Fitting    |   |
    |   +----------------+    +------------------+    +---------------------+   |
    |                                                          |               |
    |   +----------------+    +------------------+    +--------v-----------+   |
    |   | Predictor      |<---| Infilling        |<---| Outlier            |   |
    |   | Extraction     |    |                  |    | Detection          |   |
    |   +----------------+    +------------------+    +--------------------+   |
    +===========================================================================+
                                    |
                                    v
                         +---------------------+
                         | Progressive Save    |
                         | (ensemble_result.mat)|
                         +---------------------+
                                    |
                                    v
                         +---------------------+
                         | Next Pass           |
                         | (or Final Output)   |
                         +---------------------+
```

### Component Relationships

| Component | File | Responsibility |
|-----------|------|----------------|
| Entry Point | `pivtools_core/example.py` | Load config, start cluster, orchestrate |
| Pipeline | `pivtools_cli/processing/batch_pipeline.py` | Batch orchestration, pipelining |
| Correlator | `pivtools_cli/piv/piv_backend/cpu_ensemble.py` | C library calls, window warping |
| Accumulator | `pivtools_cli/piv/piv_backend/single_pass_accumulator.py` | Sum accumulation, finalization |
| Peak Fitting | `pivtools_cli/piv/piv_backend/gaussian_fitting.py` | Levenberg-Marquardt fitting |
| Factory | `pivtools_cli/piv/piv_backend/factory.py` | Correlator instantiation |
| Window Utils | `pivtools_core/window_utils.py` | Grid computation |

---

## 2. Dask Parallelization Architecture

### Worker Allocation Strategy

The pipeline divides the Dask cluster into two worker pools:

```
                    WORKER ALLOCATION
    ================================================

    Total Workers: N (from Dask cluster)
              |
              +---> Filter Workers: M (configurable)
              |           |
              |           +---> Worker 0: POD SVD, multi-threaded
              |           +---> Worker 1: Time filter, multi-threaded
              |           +---> ...
              |
              +---> Correlation Workers: N - M
                          |
                          +---> Worker M:   Pair 0 correlation
                          +---> Worker M+1: Pair 1 correlation
                          +---> ...
```

**Configuration** (`batch_pipeline.py:66-84`):
```python
def _allocate_workers(self):
    workers = list(self.client.scheduler_info()["workers"].keys())
    self.total_workers = len(workers)

    # Get allocation from config
    filter_count, corr_count = self.config.get_filter_worker_allocation(
        self.total_workers
    )

    # Assign workers to roles
    self.filter_workers = workers[:filter_count]
    self.corr_workers = workers[filter_count:]
```

### Scattering and Broadcasting Immutable Data

Before batch processing begins, immutable data is **broadcast** to all workers to avoid repeated serialization:

```
                    DATA SCATTERING
    ================================================

    Main Process                      Workers
    ============                      =======

    correlator_cache ----broadcast---> All Workers
    (window weights,                   (cached locally)
     interpolation grids,
     smoothing kernels)

    vector_masks --------broadcast---> All Workers
    (~1-10 KB per pass)                (cached locally)

    pixel_mask ----------broadcast---> All Workers
    (~1-4 MB)                          (cached locally)

    predictor_field -----broadcast---> All Workers
    (per-pass, ~1 MB)                  (cached locally)
```

**Implementation** (`batch_pipeline.py:688-718`):
```python
def _scatter_immutable_data(self, vector_masks):
    # Create and scatter correlator cache ONCE
    temp_correlator = make_correlator_backend(self.config, ensemble=True)
    correlator_cache = temp_correlator.get_cache_data()
    scattered_cache = self.client.scatter(correlator_cache, broadcast=True)

    # Scatter vector masks if present
    if vector_masks:
        scattered_masks = self.client.scatter(vector_masks, broadcast=True)

    # Scatter pixel mask if present
    if self.pixel_mask is not None:
        scattered_pixel_mask = self.client.scatter(self.pixel_mask, broadcast=True)
```

### Pipelined Batch Processing

The pipeline overlaps filtering and correlation to maximize throughput:

```
                    PIPELINED EXECUTION
    ================================================

    Time --->

    Batch 0: [===FILTER===][==========CORRELATE==========]
    Batch 1:               [===FILTER===][==========CORRELATE==========]
    Batch 2:                             [===FILTER===][==========CORRELATE==========]

    Filter Worker:  |-------|          |-------|          |-------|
    Corr Workers:            |-----------------|------------------|

    Key: Filter and correlation OVERLAP - no idle time between batches
```

**Implementation** (`batch_pipeline.py:219-274`):
```python
# Initialize first batch filter
filter_future = self.client.submit(_filter_batch_worker, batch_slice, ...)

while batch_idx < num_batches:
    # Wait for current filter
    filtered_batch = filter_future.result()

    # Start correlation (NON-BLOCKING)
    corr_futures = self._correlate_ensemble_batch_async(filtered_batch, ...)

    # OVERLAP: Start NEXT filter while THIS batch correlates
    if next_batch_idx < num_batches:
        filter_future = self.client.submit(_filter_batch_worker, next_slice, ...)

    # Wait for correlation and accumulate
    results = self.client.gather(corr_futures)
    for result in results:
        accumulator.accumulate_batch(result, pass_idx=pass_idx)
```

### Memory Management

```
                    MEMORY ARCHITECTURE
    ================================================

    Main Process Memory:
    +----------------------------------+
    | Task Graph: ~1 KB per image pair |  <-- Minimal footprint
    | Scattered Futures: ~100 bytes ea |
    +----------------------------------+

    Filter Worker Memory (Peak):
    +----------------------------------+
    | 1 Batch: batch_size * 2 * H * W  |  ~280 MB for 64 pairs
    | POD Covariance: N * N            |  ~16 MB for 64 pairs
    +----------------------------------+

    Correlation Worker Memory (Peak):
    +----------------------------------+
    | 1 Image Pair: 2 * H * W          |  ~80 MB
    | Correlation Planes: 3 * planes   |  ~10-50 MB
    | Warped Images: 2 * H * W         |  ~80 MB
    +----------------------------------+
```

---

## 3. Image Loading

### Lazy Dask Array Creation

Images are loaded **purely lazily** - no pixel data is loaded until a worker computes it:

```
                    LAZY LOADING ARCHITECTURE
    ================================================

    load_images() returns:
    +-------------------------------------------+
    | da.Array                                  |
    | Shape: (N_pairs, 2, H, W)                 |
    | Chunks: (1, 2, H, W)  <-- 1 pair per task |
    | Memory: ~1 KB per pair (just metadata)   |
    +-------------------------------------------+

    Each chunk is a delayed task:
    +-------------------------------------------+
    | delayed(load_pair)(idx, path, config)     |
    | - NOT executed until .compute() called   |
    | - Worker loads from disk at compute time |
    +-------------------------------------------+
```

**Implementation** (`pivtools_core/image_handling/load_images.py`):
```python
def load_images(camera: int, config: Config, source: Path = None) -> da.Array:
    # Creates one delayed task per image pair (pure lazy loading)
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, camera, config)
        for idx in range(1, num_pairs + 1)
    ]

    # Convert to Dask arrays (still lazy!)
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]

    # Stack into single array
    pairs_stack = da.stack(dask_pairs, axis=0)  # Shape: (N, 2, H, W)

    return pairs_stack
```

### Supported Image Formats

| Format | Extension | Reader | Notes |
|--------|-----------|--------|-------|
| TIFF | `.tif`, `.tiff` | tifffile | Standard scientific imaging |
| PNG | `.png` | PIL/OpenCV | Lossless compression |
| JPEG | `.jpg`, `.jpeg` | PIL/OpenCV | Lossy, not recommended |
| RAW | `.raw`, `.cr2`, `.nef`, `.arw` | rawpy | Camera RAW formats |
| LaVision IM7 | `.im7` | lvpyio | Single-frame containers |
| LaVision SET | `.set` | lvpyio | Multi-frame containers |
| CINE | `.cine` | pycine | High-speed video |

**Color Handling**: All readers automatically convert color to grayscale:
```python
# RGB/RGBA images converted internally
if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

### Frame Pairing Logic

```
                    FRAME PAIRING
    ================================================

    Time-Resolved (2 images per timestamp):
    +--------+--------+--------+--------+
    | A_0001 | B_0001 | A_0002 | B_0002 | ...
    +--------+--------+--------+--------+
          Pair 1           Pair 2

    Double-Frame (A/B format):
    +--------+        +--------+
    | A_0001 |  -->   | B_0001 |   = Pair 1
    +--------+        +--------+
    +--------+        +--------+
    | A_0002 |  -->   | B_0002 |   = Pair 2
    +--------+        +--------+

    LaVision SET (single container):
    +----------------------------------+
    | camera1_frameA, camera1_frameB,  |
    | camera2_frameA, camera2_frameB,  | All in one .set file
    | ...                              |
    +----------------------------------+
```

---

## 4. Preprocessing & Filtering

### Filter Application Order

```
                    FILTER PIPELINE
    ================================================

    Input: batch (N, 2, H, W)
              |
              v
    +-------------------+
    | 1. Pixel Mask     |  <-- Applied FIRST (zero out masked regions)
    +-------------------+
              |
              v
    +-------------------+
    | 2. Batch Filters  |  <-- Require full batch context
    |    - POD          |
    |    - Time         |
    +-------------------+
              |
              v
    +-------------------+
    | 3. Spatial Filters|  <-- Applied per-frame
    |    - Gaussian     |
    |    - Median       |
    |    - Normalize    |
    +-------------------+
              |
              v
    Output: batch_filtered (N, 2, H, W)
```

### Batch Filters

#### POD Filter (Proper Orthogonal Decomposition)

Removes coherent background structures using SVD decomposition:

```
                    POD FILTER ALGORITHM
    ================================================

    Input: M matrix of N images, each (H, W) reshaped to vectors
           M shape: (N, H*W)

    Step 1: Compute covariance
            C = M @ M.T   -->  shape: (N, N)

    Step 2: SVD decomposition
            C = PSI @ S @ PSI.T
            PSI: eigenvectors (N, N)
            S: eigenvalues (diagonal)

    Step 3: Find "noise mode" threshold
            For each mode i:
              - mean_psi = |mean(PSI[:, i])|
              - sig_diff = |S[i] - S[i+1]| / S[N//2]

            If mean_psi < 0.01 AND sig_diff < 0.01:
              --> Found noise mode boundary

    Step 4: Remove signal modes
            For j in [0, noise_mode):
              PHI_j = M.T @ PSI[:, j]  (spatial mode)
              TC_j = M @ PHI_j          (temporal coefficient)
              M = M - outer(TC_j, PHI_j)

    Output: M (with background removed)
```

#### Time Filter (Local Minimum Subtraction)

```
                    TIME FILTER
    ================================================

    For each frame type (A and B separately):

    Input: images shape (N, H, W)

    Step 1: Compute minimum across time dimension
            min_frame = min(images, axis=0)  -->  shape: (H, W)

    Step 2: Subtract from each image
            images[i] = images[i] - min_frame

    Step 3: Clip negative values
            images = max(images, 0)

    Effect: Removes static background, enhances particle contrast
```

### Spatial Filters

| Filter | Formula | Parameters |
|--------|---------|------------|
| Gaussian | `G(x,y) = exp(-(x^2+y^2)/(2*sigma^2))` | `sigma` |
| Median | `out[i,j] = median(window around i,j)` | `size` (tuple) |
| Normalize | `(img - local_min) / max(local_max - local_min, 1/max_gain)` | `size`, `max_gain` |
| Clip | `clip(img, low, high)` | `threshold` (tuple) |

### Filter Worker Execution

```
                    FILTER WORKER
    ================================================

    def _filter_batch_worker(batch_images, config, ...):
        # 1. Use ALL cores on this worker for multi-threading
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

        # 2. Load batch with threading scheduler
        with dask.config.set(scheduler='threads'):
            batch = batch_images.compute()  # Load from disk

        # 3. Apply filters
        batch_filtered = apply_filters_to_batch(
            batch, config,
            pixel_mask=pixel_mask,  # Applied first
        )

        return batch_filtered  # numpy array
```

---

## 5. Single-Pass Ensemble Correlation

### Mathematical Foundation

The single-pass ensemble formula computes correlation statistics without storing all warped images:

```
                    ENSEMBLE CORRELATION FORMULA
    ================================================

    Traditional (memory-intensive):
        R_AB = (1/N) * sum_i(A_i * B_i) - ((1/N) * sum_i(A_i)) * ((1/N) * sum_i(B_i))

    Single-Pass Equivalent:
        R_AA = <A*A> - <A>*<A>   (Auto-correlation A, background subtracted)
        R_BB = <B*B> - <B>*<B>   (Auto-correlation B, background subtracted)
        R_AB = <A*B> - <A>*<B>   (Cross-correlation, background subtracted)

    Where:
        <·> = ensemble average over all image pairs
        * = cross-correlation operation (FFT-based)

    Accumulation (streaming):
        sum_corr_AA += correlate(A_i, A_i)
        sum_corr_BB += correlate(B_i, B_i)
        sum_corr_AB += correlate(A_i, B_i)
        sum_warp_A += A_i
        sum_warp_B += B_i

    Finalization:
        A_mean = sum_warp_A / N
        B_mean = sum_warp_B / N
        R_AA_bg = correlate(A_mean, A_mean)
        R_BB_bg = correlate(B_mean, B_mean)
        R_AB_bg = correlate(A_mean, B_mean)

        R_AA_ensemble = (sum_corr_AA / N) - R_AA_bg
        R_BB_ensemble = (sum_corr_BB / N) - R_BB_bg
        R_AB_ensemble = (sum_corr_AB / N) - R_AB_bg
```

### SinglePassAccumulator Architecture

```
                    ACCUMULATOR STATE
    ================================================

    class SinglePassAccumulator:
        passes_data[pass_idx] = {
            # Running sums (NOT averaged yet)
            "sum_warp_A": np.zeros((H, W), dtype=np.float32),
            "sum_warp_B": np.zeros((H, W), dtype=np.float32),

            # Correlation plane sums (THREE planes for stacked Gaussian)
            "sum_corr_AA": np.zeros(plane_size, dtype=np.float32),
            "sum_corr_BB": np.zeros(plane_size, dtype=np.float32),
            "sum_corr_AB": np.zeros(plane_size, dtype=np.float32),

            # Grid info
            "n_win_x": n_win_x,
            "n_win_y": n_win_y,
            "corr_size": corr_size,
            "win_size": win_size,
        }

        n_images: int  # Total image pairs accumulated
        passes_results: list[PIVEnsemblePassResult]  # Completed passes
```

### Accumulation Loop

```python
# single_pass_accumulator.py:104-135
def accumulate_batch(self, batch_result: dict, pass_idx: int):
    pass_data = self.passes_data[pass_idx]

    # Accumulate warped images (for mean computation later)
    pass_data["sum_warp_A"] += batch_result["warp_A_sum"]
    pass_data["sum_warp_B"] += batch_result["warp_B_sum"]

    # Accumulate correlation planes (NO averaging yet)
    pass_data["sum_corr_AA"] += batch_result["corr_AA_sum"]
    pass_data["sum_corr_BB"] += batch_result["corr_BB_sum"]
    pass_data["sum_corr_AB"] += batch_result["corr_AB_sum"]

    # Store smoothed predictor for later use
    if batch_result.get("smoothed_predictor") is not None:
        pass_data["smoothed_predictor"] = batch_result["smoothed_predictor"]

    self.n_images += batch_result["n_images"]
```

### Window Weighting Functions

Different weighting functions affect correlation plane characteristics:

```
                    WINDOW WEIGHTS
    ================================================

    Square (uniform):
        W[i,j] = 1.0 for all pixels
        Effect: Maximum signal, but spectral leakage at edges

    Blackman (radial):
        W(r) = 0.42659 - 0.49656*cos(theta) + 0.076849*cos(2*theta)
        where theta = pi * (1 - r/R)
        Effect: Reduced edge artifacts, lower signal

    Gaussian:
        W(x,y) = exp(-0.5 * ((alpha*x/(m/2))^2 + (alpha*y/(n/2))^2))
        Effect: Smooth tapering, good noise rejection

    Single Mode (for ensemble single mode):
        Frame A: Small window centered in SumWindow (e.g., 4x4 in 16x16)
        Frame B: Full SumWindow (16x16)
        Effect: Reduces particle dropout bias
```

**Implementation** (`base.py:30-114`):
```python
def _window_weight_fun(self, win_size, win_type, sum_window=None):
    if win_type == 'square':
        weight = np.ones(win_size, dtype=np.float32)
    elif win_type == 'blackman':
        # Radial Blackman window
        a0, a1, a2 = 0.42659, 0.49656, 0.076849
        # ... compute radial distance and apply formula
    elif win_type == 'gaussian':
        alpha = 1.0
        weight = np.exp(-0.5 * ((alpha*xx/(m/2))**2 + (alpha*yy/(n/2))**2))
    elif win_type == 'singlepix':
        # Small window centered in sum_window
        weight = np.zeros(sum_window, dtype=np.float32)
        weight[start_row:end_row, start_col:end_col] = 1.0
```

### C Library Cross-Correlation

The actual correlation is performed by a C library using FFTW:

```c
// xcorr.c - Cross-correlation formula
// C = fftshift(IFFT(FFT(A) .* conj(FFT(B))))

unsigned xcorr_preplanned(const float *a, const float *b, float *c, sPlan *plan) {
    // 1. Copy input arrays
    memcpy(&ab_copy[0], a, numel * sizeof(float));
    memcpy(&ab_copy[numel], b, numel * sizeof(float));

    // 2. Forward FFT (both arrays simultaneously)
    fftwf_execute(plan->plan_AB_fft);

    // 3. Complex conjugate multiplication: C = A .* conj(B)
    multiply_conjugate(AB_copy, &AB_copy[numel_fft], C, numel_fft);

    // 4. Inverse FFT
    fftwf_execute(plan->plan_C_ifft);

    // 5. FFTshift and copy to output
    // ... (swap quadrants to center zero-frequency)
}
```

---

## 6. Peak Fitting (Gaussian Sub-Pixel Localization)

### 13-DOF Stacked Gaussian Model

The ensemble PIV uses a **stacked 3-plane model** that simultaneously fits AA, BB, and AB correlation planes:

```
                    STACKED GAUSSIAN MODEL
    ================================================

    Three correlation planes are fitted together:

    Plane AA (Auto-correlation A):
        F_AA(x,y) = amp_A * exp(-0.5 * Q_A)

    Plane BB (Auto-correlation B):
        F_BB(x,y) = amp_B * exp(-0.5 * Q_A)  <-- Same shape as AA!

    Plane AB (Cross-correlation):
        F_AB(x,y) = amp_AB * exp(-0.5 * Q_AB)

    Where:
        Q_A = [dx_A, dy_A]^T * Sigma_A^-1 * [dx_A, dy_A]
        Q_AB = [dx_AB, dy_AB]^T * (Sigma_A + Sigma_AB)^-1 * [dx_AB, dy_AB]

    Covariance matrices:
        Sigma_A = [[sx_A, sxy_A], [sxy_A, sy_A]]     (particle image shape)
        Sigma_AB = [[sx_AB, sxy_AB], [sxy_AB, sy_AB]] (displacement uncertainty)

    13 Parameters:
        [amp_A, amp_B, amp_AB,           # 3 amplitudes
         sx_A, sy_A, sxy_A,              # 3 covariance components (shape A)
         sx_AB, sy_AB, sxy_AB,           # 3 covariance components (displacement)
         x0_A, y0_A,                     # 2 center coordinates (A)
         x0_AB, y0_AB]                   # 2 center coordinates (AB = displacement)
```

### Levenberg-Marquardt Algorithm

The fitting uses GSL's trust-region Levenberg-Marquardt solver:

```
                    LM ALGORITHM
    ================================================

    Objective: Minimize sum of squared residuals
        f(p) = sum_i [ model(x_i, p) - data_i ]^2

    Iteration:
        1. Compute Jacobian J = df/dp at current parameters
        2. Solve: (J^T * J + lambda * diag(J^T * J)) * delta = -J^T * r
           where r = residuals
        3. Update: p_new = p + delta
        4. If fit improved: reduce lambda, accept step
           If worse: increase lambda, reject step
        5. Repeat until convergence

    Convergence criteria:
        - |delta_p| < xtol = 1e-8  (parameter tolerance)
        - |gradient| < gtol = 1e-8  (gradient tolerance)
        - Max iterations: 100
```

**C Implementation Optimization** (`marquadt_gaussian.c:52-118`):

```c
// Pre-compute Cholesky decomposition OUTSIDE the pixel loop
double sqrt_sx_A = sqrt(sx_A);
double term_A = sxy_A / sqrt_sx_A;
double LA_00 = sqrt_sx_A;
double LA_10 = term_A;
double LA_11 = sqrt(sy_A - term_A * term_A);

// Inverse of lower triangular matrix
double inv_LA_00 = 1.0 / LA_00;
double inv_LA_11 = 1.0 / LA_11;
double inv_LA_10 = -LA_10 * (inv_LA_00 * inv_LA_11);

// FAST inner loop: only 2 multiplications + 1 addition per pixel
for (i = 0; i < n; i++) {
    double dx_A = X1[i] - x0_A;
    double dy_A = X2[i] - y0_A;

    // Matrix multiplication via Cholesky factor
    double tA_0 = inv_LA_00 * dx_A;
    double tA_1 = inv_LA_10 * dx_A + inv_LA_11 * dy_A;

    // Quadratic form (Mahalanobis distance)
    double quad_A = tA_0 * tA_0 + tA_1 * tA_1;
    double exp_A = exp(-0.5 * quad_A);

    // Residuals for AA, BB, AB planes
    gsl_vector_set(f, i, (amp_A * exp_A) - y[i]);
    gsl_vector_set(f, i + n, (amp_B * exp_A) - y[i + n]);  // Same shape!
    gsl_vector_set(f, i + 2*n, (amp_AB * exp_AB) - y[i + 2*n]);
}
```

### Initial Guess Strategy

```
                    INITIAL GUESS
    ================================================

    Pass 0 (First iteration):
    -------------------------
    Displacement: Found by locating peak in AB plane
        max_idx = argmax(AB_win)
        guess_x_AB, guess_y_AB = unravel_index(max_idx, win_size)

    Amplitude: Peak values at peak locations
        amp_A = AA_win[central_index]
        amp_B = BB_win[central_index]
        amp_AB = AB_win[max_idx]

    Sigma A: Estimated from HWHM of AA plane
        Find pixels where AA >= peak_val / 2
        HWHM = (max_idx - min_idx) / 2
        sigma = HWHM / sqrt(2 * ln(2))

    Sigma AB: Estimated from (HWHM_AB - HWHM_A)
        Removes contribution of particle image size
        sigma_AB = max(HWHM_AB - HWHM_A, 0.1) / sqrt(2 * ln(2))

    Cross-terms: sxy_A = sxy_AB = 0 (assume axis-aligned for pass 0)

    Pass > 0 (Subsequent iterations):
    ----------------------------------
    Displacement: Still found from peak in AB plane (after warping)
    Amplitude: Still from peak values
    Sigma A: INTERPOLATED from previous pass (cubic interpolation if grid changed)
    Sigma AB: INTERPOLATED from previous pass
    Cross-terms: INTERPOLATED from previous pass (may be non-zero)
```

### Validation Checks

After fitting, results are validated (`gaussian_fitting.py:153-232`):

```python
def _validate_fitted_params(gauss_params, win_size, pass_idx, runtype, ...):
    # Check 1: AB peak height validity (must be in [0, 1] when normalized)
    if AA_central > 1e-12 and BB_central > 1e-12:
        AB_normalized = amp_AB / np.sqrt(AA_central * BB_central)
        if not (0 <= AB_normalized <= 1):
            return False, reason=2

    # Check 2: 1/2 displacement rule (peak must be in central half)
    # Only for pass > 0 or single mode
    if pass_idx > 0 or runtype == 'single':
        if abs(x0_AB - center_x) > half_x or abs(y0_AB - center_y) > half_y:
            return False, reason=3

    # Check 3: Negative sigmas
    if sx_AB < 0 or sy_AB < 0:
        return False, reason=5

    return True, 0
```

### Distributed Fitting Architecture

```
                    DISTRIBUTED FITTING
    ================================================

    Main Process:
    +------------------------------------------+
    | 1. Pre-chunk correlation planes          |
    |    - Split into N chunks (one per worker)|
    |    - Each chunk: ~1/N of total windows   |
    +------------------------------------------+
              |
              v
    +------------------------------------------+
    | 2. Scatter chunks to specific workers    |
    |    - client.scatter(chunk, workers=[w])  |
    |    - Returns futures (~100 bytes each)   |
    +------------------------------------------+
              |
              v
    +------------------------------------------+
    | 3. Submit fitting tasks with futures     |
    |    - Tiny task graph (only futures)      |
    |    - No large data in graph              |
    +------------------------------------------+

    Workers (in parallel):
    +------------------------------------------+
    | 4. Unpack scattered data                 |
    | 5. For each non-masked window:           |
    |    - Extract window from chunk           |
    |    - Build initial guess                 |
    |    - Call C library (per-window)         |
    |    - Validate result                     |
    | 6. Return fitted parameters              |
    +------------------------------------------+
```

---

## 7. Multi-Pass Iteration

### Predictor Field Extraction and Smoothing

After each pass, the displacement field becomes the predictor for the next pass:

```
                    PREDICTOR EXTRACTION
    ================================================

    After Pass N finalization:

    1. Extract displacement field:
       uy = pass_result.uy_mat  # Shape: (n_win_y, n_win_x)
       ux = pass_result.ux_mat

    2. Stack as predictor:
       predictor_field = stack([uy, ux], axis=-1)
       # Shape: (n_win_y, n_win_x, 2)
       # [:,:,0] = Y component
       # [:,:,1] = X component

    3. Pad for boundary extrapolation:
       predictor_field = pad(predictor_field, ((1,1), (1,1), (0,0)), mode='edge')
       # Shape: (n_win_y+2, n_win_x+2, 2)

    4. Broadcast to all workers for next pass
```

### Image Warping with Symmetric Deformation

```
                    SYMMETRIC WARPING
    ================================================

    Given predictor field delta_ab (from previous pass):

    1. Smooth predictor with Gaussian filter:
       delta_ab_smooth[:,:,0] = gaussian_filter(delta_ab[:,:,0], sigma)
       delta_ab_smooth[:,:,1] = gaussian_filter(delta_ab[:,:,1], sigma)

    2. Interpolate to dense image grid:
       delta_ab_dense = remap(delta_ab_smooth, interp_maps)
       # Shape: (H, W, 2)

    3. Split into symmetric warps:
       delta_0a = -delta_ab_dense / 2  # Frame A: backward warp
       delta_0b = +delta_ab_dense / 2  # Frame B: forward warp

    4. Create warped coordinate meshes:
       im_mesh_A = im_mesh + delta_0a
       im_mesh_B = im_mesh + delta_0b

    5. Warp images using cv2.remap:
       image_a_warped = cv2.remap(image_a, im_mesh_A[...,1], im_mesh_A[...,0], INTER_CUBIC)
       image_b_warped = cv2.remap(image_b, im_mesh_B[...,1], im_mesh_B[...,0], INTER_CUBIC)

    Effect:
    +-------------------+        +-------------------+
    |  Frame A          |        |  Frame B          |
    |  (moves backward) |        |  (moves forward)  |
    |       <----       |        |       ---->       |
    +-------------------+        +-------------------+
              |                          |
              +----------+---------------+
                         |
                         v
                  CORRELATION
               (residual displacement)
```

### Grid Interpolation Between Passes

When window sizes differ between passes, sigma fields must be interpolated:

```
                    SIGMA INTERPOLATION
    ================================================

    Previous Pass:                 Current Pass:
    Window: 64x64                  Window: 32x32
    Grid: 20x20                    Grid: 40x40

    Interpolation (cubic):

    +---+---+---+---+             +--+--+--+--+--+--+--+--+
    | * | * | * | * |             |  |  |  |  |  |  |  |  |
    +---+---+---+---+      -->    +--*--*--*--*--*--*--*--+
    | * | * | * | * |             |  |  |  |  |  |  |  |  |
    +---+---+---+---+             +--*--*--*--*--*--*--*--+
    | * | * | * | * |             |  |  |  |  |  |  |  |  |
    +---+---+---+---+             +--*--*--*--*--*--*--*--+
    20x20 sigma values            40x40 interpolated values

    Implementation:
    map_y, map_x = meshgrid(
        linspace(0, old_h-1, new_h),
        linspace(0, old_w-1, new_w),
        indexing='ij'
    )
    new_sigma = cv2.remap(old_sigma, map_x, map_y, INTER_CUBIC)
```

### Multi-Pass Flow Diagram

```
                    MULTI-PASS ITERATION
    ================================================

    Pass 1 (Coarse):
    +-----------+     +-----------+     +-----------+
    | 128x128   | --> | Correlate | --> | ux_1,uy_1 |
    | windows   |     | (no warp) |     | (coarse)  |
    +-----------+     +-----------+     +-----------+
                                              |
                                              v
                                    +-----------------+
                                    | Predictor Field |
                                    +-----------------+
                                              |
    Pass 2 (Medium):                          |
    +-----------+     +-----------+     +-----v-----+
    | 64x64     | --> | Correlate | --> | ux_2,uy_2 |
    | windows   |     | (warped)  |     | (medium)  |
    +-----------+     +-----------+     +-----------+
                                              |
                            ux_2 = residual + predictor
                                              |
                                              v
                                    +-----------------+
                                    | Predictor Field |
                                    +-----------------+
                                              |
    Pass 3 (Fine):                            |
    +-----------+     +-----------+     +-----v-----+
    | 32x32     | --> | Correlate | --> | ux_3,uy_3 |
    | windows   |     | (warped)  |     | (fine)    |
    +-----------+     +-----------+     +-----------+
                                              |
                            ux_3 = residual + predictor
                                              |
                                              v
                                    +-----------------+
                                    | Final Output    |
                                    +-----------------+
```

---

## 8. Outlier Detection & Infilling

### Detection Methods

```
                    OUTLIER DETECTION METHODS
    ================================================

    1. Peak Magnitude Threshold:
       outlier = (peakheight < threshold) | isnan(peakheight)
       Default threshold: 0.5

    2. Median 2D (Normalized Median Test):
       For each vector:
         r_0 = |median(8-neighbors) - vector|
         r_i = |each-neighbor - median|
         r_m = median(r_i)  # Median Absolute Deviation

         outlier_score = r_0 / (r_m + epsilon)
         outlier = (score > threshold) or (< 6 valid neighbors)
       Default: epsilon=0.2, threshold=2.0

    3. Sigma-Based:
       For each vector:
         mean_8 = mean of 8-neighbors (excluding center)
         std_8 = std of 8-neighbors

         outlier = |vector - mean_8| > sigma_threshold * std_8
       Default: sigma_threshold=2.0

```

### Infilling Algorithms

| Method | Speed | Smoothness | Description |
|--------|-------|------------|-------------|
| `local_median` | Fastest | Low | Replace with median of ksize x ksize neighbors |
| `knn` | Fast | Medium | K-nearest neighbors interpolation (KD-tree) |
| `griddata_linear` | Medium | Medium | Delaunay triangulation + linear interpolation |
| `rbf` | Slow | High | Radial basis function with thin plate spline |
| `biharmonic` | Slow | Highest | Solves biharmonic PDE: ∇⁴u = 0 |

**Biharmonic Inpainting** (default for mid-pass):
```python
# Solves: ∇⁴u = 0 with Dirichlet boundary conditions
# Provides C2-continuous smooth interpolation
ux_filled = inpaint_biharmonic(ux_with_nans, combined_mask)
uy_filled = inpaint_biharmonic(uy_with_nans, combined_mask)
```

### Application Timing

```
                    POST-PROCESSING ORDER
    ================================================

    After Gaussian fitting:

    1. Extract velocities from fitted parameters
       ux_mat = x0_AB - center_x
       uy_mat = y0_AB - center_y
       (+ add predictor for pass > 0)

    2. Apply VECTOR MASK (before outlier detection)
       ux_mat[vector_mask] = 0
       uy_mat[vector_mask] = 0
       nan_reason[vector_mask] = -1

    3. OUTLIER DETECTION
       outlier_mask = (statuses != 0)  # Fitting failures
       outlier_mask |= apply_outlier_detection(...)  # Statistical tests

       ux_mat[outlier_mask] = NaN
       uy_mat[outlier_mask] = NaN

    4. INFILLING
       Mid-pass: ALWAYS infill (required for predictor)
       Final pass: Configurable (may keep NaNs)

       ux_mat, uy_mat = apply_infilling(ux_mat, uy_mat, outlier_mask, config)

    5. Extract predictor for next pass (uses infilled data)
```

---

## 9. Result Saving

### Progressive Ensemble Saving

Results are saved **progressively** after each pass, not at the end:

```
                    PROGRESSIVE SAVING
    ================================================

    After Pass 1:
    +---------------------------+
    | ensemble_result.mat       |
    | - ensemble_result[0]      |  <-- Pass 1 only
    +---------------------------+

    After Pass 2:
    +---------------------------+
    | ensemble_result.mat       |
    | - ensemble_result[0]      |  <-- Pass 1
    | - ensemble_result[1]      |  <-- Pass 2 (appended)
    +---------------------------+

    After Pass 3:
    +---------------------------+
    | ensemble_result.mat       |
    | - ensemble_result[0]      |  <-- Pass 1
    | - ensemble_result[1]      |  <-- Pass 2
    | - ensemble_result[2]      |  <-- Pass 3 (appended)
    +---------------------------+

    Benefit: Crash recovery possible from any pass
```

### MATLAB .mat Format Structure

```matlab
% ensemble_result.mat structure
ensemble_result(pass_idx) = struct(
    'ux', ...,           % X displacement (n_win_y, n_win_x)
    'uy', ...,           % Y displacement (n_win_y, n_win_x)
    'UU_stress', ...,    % X variance (from sig_AB_x)
    'VV_stress', ...,    % Y variance (from sig_AB_y)
    'UV_stress', ...,    % XY covariance (from sig_AB_xy)
    'peakheight', ...,   % Normalized peak: amp_AB / sqrt(amp_A * amp_B)
    'nan_reason', ...,   % Status codes: 0=valid, -1=masked, 10=outlier
    'sig_AB_x', ...,     % Cross-correlation X width
    'sig_AB_y', ...,     % Cross-correlation Y width
    'sig_AB_xy', ...,    % Cross-correlation XY covariance
    'sig_A_x', ...,      % Auto-correlation X width
    'sig_A_y', ...,      % Auto-correlation Y width
    'sig_A_xy', ...,     % Auto-correlation XY covariance
    'b_mask', ...,       % Vector mask applied
    'pred_x', ...,       % Predictor X component (pass > 0)
    'pred_y', ...,       % Predictor Y component (pass > 0)
    'window_size', ...,  % Window size [height, width]
    'win_ctrs_x', ...,   % Window center X coordinates
    'win_ctrs_y', ...    % Window center Y coordinates
);
```

### Resume Capability

The pipeline supports resuming from intermediate passes:

```python
# In config.yaml
ensemble_piv:
  resume_from_pass: 2  # Skip passes 1-2, start from pass 3

# Pipeline behavior:
# 1. Load ensemble_result.mat
# 2. Extract predictor from pass 2
# 3. Start processing from pass 3
# 4. Merge with existing passes on save
```

---

## 10. Key Data Structures

### PIVEnsemblePassResult

```python
@dataclass
class PIVEnsemblePassResult:
    # Primary outputs
    ux_mat: np.ndarray          # (n_win_y, n_win_x) float32
    uy_mat: np.ndarray          # (n_win_y, n_win_x) float32

    # Uncertainty measures (Reynolds stresses)
    UU_stress: np.ndarray       # (n_win_y, n_win_x) float32
    VV_stress: np.ndarray       # (n_win_y, n_win_x) float32
    UV_stress: np.ndarray       # (n_win_y, n_win_x) float32

    # Quality measures
    peakheight: np.ndarray      # (n_win_y, n_win_x) float32
    nan_reason: np.ndarray      # (n_win_y, n_win_x) int32

    # Gaussian fit parameters
    sig_AB_x: np.ndarray        # (n_win_y, n_win_x) float32
    sig_AB_y: np.ndarray        # (n_win_y, n_win_x) float32
    sig_AB_xy: np.ndarray       # (n_win_y, n_win_x) float32
    sig_A_x: np.ndarray         # (n_win_y, n_win_x) float32
    sig_A_y: np.ndarray         # (n_win_y, n_win_x) float32
    sig_A_xy: np.ndarray        # (n_win_y, n_win_x) float32

    # Masking
    b_mask: Optional[np.ndarray]  # (n_win_y, n_win_x) bool

    # Predictor field (pass > 0 only)
    pred_x: Optional[np.ndarray]  # (n_win_y, n_win_x) float32
    pred_y: Optional[np.ndarray]  # (n_win_y, n_win_x) float32

    # Grid info
    window_size: tuple          # (height, width)
    win_ctrs_x: np.ndarray      # (n_win_x,) float32
    win_ctrs_y: np.ndarray      # (n_win_y,) float32
```

### Correlation Plane Shapes

```
Correlation planes are stored FLAT for efficient C library access:

For window size (64, 64) with grid (40, 30):
  - Total windows: 40 * 30 = 1200
  - Points per window: 64 * 64 = 4096
  - Plane size: 1200 * 4096 = 4,915,200 float32
  - Memory: ~19 MB per plane, ~57 MB for AA+BB+AB

Layout: [win_0_pix_0, win_0_pix_1, ..., win_0_pix_4095,
         win_1_pix_0, win_1_pix_1, ..., win_1_pix_4095,
         ...]

To extract window idx from flat array:
  start = idx * (corr_h * corr_w)
  end = start + (corr_h * corr_w)
  window = flat_array[start:end].reshape(corr_h, corr_w)
```

### Cache Dictionary Contents

```python
correlator.get_cache_data() -> dict:
    'win_ctrs_x': List[np.ndarray]      # Per-pass X centers
    'win_ctrs_y': List[np.ndarray]      # Per-pass Y centers
    'win_spacing_x': List[int]          # Per-pass X spacing
    'win_spacing_y': List[int]          # Per-pass Y spacing
    'win_ctrs_x_all': List[np.ndarray]  # Padded X centers
    'win_ctrs_y_all': List[np.ndarray]  # Padded Y centers
    'n_pre_all': List[tuple]            # Padding before grid
    'n_post_all': List[tuple]           # Padding after grid
    'ksize_filt': List[tuple]           # Smoothing kernel sizes
    'sd': List[float]                   # Smoothing sigma values
    'G_smooth_predictor': List[np.ndarray]  # Smoothing kernels
    'H': int                            # Image height
    'W': int                            # Image width
    'im_mesh': np.ndarray               # Base coordinate mesh
    'cached_dense_maps': List[tuple]    # Dense interpolation maps
    'cached_predictor_maps': List[tuple]  # Predictor interpolation maps
    'win_weights_A': List[np.ndarray]   # Frame A weights
    'win_weights_B': List[np.ndarray]   # Frame B weights
    'window_sizes_for_corr': List[tuple]  # Actual correlation sizes
```

---

## 11. File Reference

### Primary Pipeline Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `pivtools_core/example.py` | Entry point | `main()`, `validate_config()` |
| `pivtools_cli/processing/batch_pipeline.py` | Pipeline orchestration | `UnifiedBatchPipeline`, `_filter_batch_worker()` |
| `pivtools_cli/piv/piv_backend/cpu_ensemble.py` | Correlator | `EnsembleCorrelatorCPU`, `correlate_batch_for_accumulation()` |
| `pivtools_cli/piv/piv_backend/single_pass_accumulator.py` | Accumulation | `SinglePassAccumulator`, `finalize_pass()` |
| `pivtools_cli/piv/piv_backend/gaussian_fitting.py` | Peak fitting | `_fit_windows_batch_optimized()`, `_build_initial_guess()` |
| `pivtools_cli/piv/piv_backend/factory.py` | Factory | `make_correlator_backend()` |

### Supporting Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `pivtools_core/window_utils.py` | Grid computation | `compute_window_centers()`, `compute_window_centers_single_mode()` |
| `pivtools_cli/piv/piv_backend/base.py` | Base correlator | `CrossCorrelator`, `_window_weight_fun()` |
| `pivtools_cli/piv/piv_backend/outlier_detection.py` | Outlier detection | `apply_outlier_detection()` |
| `pivtools_cli/piv/piv_backend/infilling.py` | Gap filling | `apply_infilling()` |
| `pivtools_cli/preprocessing/preprocess.py` | Filter application | `apply_filters_to_batch()` |
| `pivtools_cli/preprocessing/filters.py` | Filter implementations | `_pod_filter_block()`, `_subtract_local_min()` |
| `pivtools_core/image_handling/load_images.py` | Image loading | `load_images()` |

### C Libraries

| File | Purpose | Key Functions |
|------|---------|---------------|
| `pivtools_cli/lib/xcorr.c` | Cross-correlation | `xcorr_preplanned()`, `multiply_conjugate()` |
| `pivtools_cli/lib/marquadt_gaussian.c` | Gaussian fitting | `fit_stacked_gaussian_export()` |
| `pivtools_cli/lib/PIV_2d_cross_correlate.c` | Bulk correlation | `bulkxcorr2d()` |

---

## Appendix: Key Formulas

### Cross-Correlation (FFT-based)
```
C(tau) = IFFT(FFT(A) * conj(FFT(B)))
```

### Normalized Peak Height
```
peak_normalized = amp_AB / sqrt(max(amp_A * amp_B, 1e-12))
```

### HWHM to Sigma Conversion
```
sigma = HWHM / sqrt(2 * ln(2)) ≈ HWHM / 1.177
```

### Displacement from Gaussian Center
```
ux = x0_AB - (corr_size_x / 2 + 1)
uy = y0_AB - (corr_size_y / 2 + 1)
```

### Absolute Displacement (pass > 0)
```
ux_absolute = ux_residual + predictor_x_smoothed
uy_absolute = uy_residual + predictor_y_smoothed
```

### Median Absolute Deviation (MAD)
```
MAD = 1.4826 * median(|x - median(x)|)
```
