# PIV Pipeline Profiling Report

**Date:** 2026-02-07
**Config:** 4-pass multi-grid (128 &rarr; 64 &rarr; 32 &rarr; 16), 50% overlap, Gaussian window, biharmonic infilling, median_2d outlier detection, 4 OMP threads, 3 iterations averaged
**Hardware:** Windows, 4 OMP threads

---

## 1. Executive Summary

The predictor-corrector image warping step dominates passes 2-4, consuming **65-92%** of per-pass runtime regardless of peak finder or image size. The actual FFT cross-correlation (`bulkxcorr2d`) — the operation one might assume is the bottleneck — accounts for only **5-19%** of later passes. At the finest grid (16x16), outlier detection and biharmonic infilling together take **19-23%**.

**The #1 optimization target is predictor-corrector**, specifically `mesh_compute` (broadcast arithmetic) and `image_warp` (cv2.remap). These two sub-operations alone account for **48-68%** of passes 2-4.

---

## 2. Grand Totals

| Dataset | Peak Finder | Grand Total | Speedup vs gauss6 |
|---------|-------------|------------:|-------------------:|
| 4 MP (2048x2048) | gauss6 | 0.549s | baseline |
| 4 MP (2048x2048) | gauss3 | 0.507s | **1.08x** (8%) |
| 25 MP (4600x5312) | gauss6 | 3.250s | baseline |
| 25 MP (4600x5312) | gauss3 | 3.058s | **1.06x** (6%) |

gauss3 (3-point parabolic) is 6-8% faster than gauss6 (6-DOF rotated elliptical Gaussian), with the saving coming entirely from the `bulkxcorr2d` step. The predictor-corrector cost is identical between peak finders since it operates on the previous pass result, not the fitting method.

---

## 3. Per-Pass Breakdown: 25 MP (4600x5312)

This is the production-relevant dataset — 574x663 = 380,562 windows at the finest grid.

### Pass 1 (128x128, 70x82 = 5,740 windows)

| Section | gauss6 | gauss3 | Notes |
|---------|-------:|-------:|-------|
| predictor_corrector | 0.053s (43%) | 0.052s (34%) | Skipped (no prior result) |
| bulkxcorr2d | 0.064s (52%) | 0.094s (61%) | FFT + peak fitting |
| outlier_detection | 0.002s (2%) | 0.002s (1%) | |
| infilling | 0.004s (4%) | 0.005s (3%) | |
| **TOTAL** | **0.124s** | **0.153s** | |

Pass 1 is the only pass where `bulkxcorr2d` dominates (no predictor-corrector). Both are fast — <0.2s.

### Pass 2 (64x64, 142x165 = 23,430 windows)

| Section | gauss6 | gauss3 | Notes |
|---------|-------:|-------:|-------|
| **predictor_corrector** | **0.802s (92%)** | **0.776s (90%)** | **Dominates** |
| &nbsp;&nbsp; mesh_compute | 0.277s (32%) | 0.275s (32%) | Broadcast arithmetic |
| &nbsp;&nbsp; image_warp | 0.343s (39%) | 0.311s (36%) | cv2.remap INTER_CUBIC |
| &nbsp;&nbsp; dense_remap | 0.160s (18%) | 0.165s (19%) | Grid-to-image remap |
| bulkxcorr2d | 0.057s (7%) | 0.071s (8%) | |
| outlier_detection | 0.007s (1%) | 0.007s (1%) | |
| infilling | 0.009s (1%) | 0.008s (1%) | |
| **TOTAL** | **0.877s** | **0.863s** | |

### Pass 3 (32x32, 286x331 = 94,666 windows)

| Section | gauss6 | gauss3 | Notes |
|---------|-------:|-------:|-------|
| **predictor_corrector** | **0.812s (81%)** | **0.788s (87%)** | **Dominates** |
| &nbsp;&nbsp; mesh_compute | 0.284s (28%) | 0.285s (32%) | |
| &nbsp;&nbsp; image_warp | 0.333s (33%) | 0.307s (34%) | |
| &nbsp;&nbsp; dense_remap | 0.171s (17%) | 0.172s (19%) | |
| bulkxcorr2d | 0.124s (12%) | 0.051s (6%) | gauss3 is 2.4x faster here |
| outlier_detection | 0.033s (3%) | 0.031s (4%) | |
| infilling | 0.025s (3%) | 0.027s (3%) | |
| **TOTAL** | **0.997s** | **0.902s** | |

### Pass 4 (16x16, 574x663 = 380,562 windows) &mdash; The Bottleneck Pass

| Section | gauss6 | gauss3 | Notes |
|---------|-------:|-------:|-------|
| **predictor_corrector** | **0.823s (66%)** | **0.800s (70%)** | **Still dominates** |
| &nbsp;&nbsp; mesh_compute | 0.283s (23%) | 0.283s (25%) | Identical cost |
| &nbsp;&nbsp; image_warp | 0.337s (27%) | 0.315s (28%) | |
| &nbsp;&nbsp; dense_remap | 0.175s (14%) | 0.174s (15%) | |
| bulkxcorr2d | 0.166s (13%) | 0.061s (5%) | **gauss3 is 2.7x faster** |
| outlier_detection | 0.140s (11%) | 0.135s (12%) | |
| infilling | 0.101s (8%) | 0.122s (11%) | |
| **TOTAL** | **1.251s** | **1.140s** | |

---

## 4. Per-Pass Breakdown: 4 MP (2048x2048)

Smaller dataset (255x255 = 65,025 windows at finest grid), useful for rapid iteration.

### Pass 4 (16x16, 255x255 = 65,025 windows)

| Section | gauss6 | gauss3 | Notes |
|---------|-------:|-------:|-------|
| **predictor_corrector** | **0.133s (63%)** | **0.135s (74%)** | |
| &nbsp;&nbsp; mesh_compute | 0.044s (21%) | 0.044s (24%) | |
| &nbsp;&nbsp; image_warp | 0.053s (25%) | 0.056s (30%) | |
| &nbsp;&nbsp; dense_remap | 0.029s (14%) | 0.029s (16%) | |
| bulkxcorr2d | 0.041s (19%) | 0.011s (6%) | **gauss3 is 3.7x faster** |
| outlier_detection | 0.020s (9%) | 0.020s (11%) | |
| infilling | 0.016s (7%) | 0.015s (8%) | |
| **TOTAL** | **0.212s** | **0.184s** | |

---

## 5. Cost Scaling: Where Time Goes at 25 MP

```
Pass 1 ██░░░░░░░░░░░░░░░░░░ 0.12s   3.8%  (5,740 windows)
Pass 2 █████████████████████████████░░░░░░░░░░░░ 0.88s  27.0%  (23,430 windows)
Pass 3 ██████████████████████████████████░░░░░░░░ 1.00s  30.7%  (94,666 windows)
Pass 4 ████████████████████████████████████████░░ 1.25s  38.5%  (380,562 windows)
                                                  ─────
                                                  3.25s
```

Pass 4 takes 38% of total time despite being the final (and often most important) pass. But the predictor-corrector cost is nearly constant across passes 2-4 (~0.8s each) because it operates on the full image, not per-window. The per-window costs (xcorr, outlier, infilling) scale with window count.

---

## 6. Optimization Opportunities (Ranked by Impact)

### Priority 1: Predictor-Corrector — `mesh_compute` (22-32% of pass time)

**Current:** NumPy broadcast arithmetic to build pixel-level displacement meshes from window-center fields. Two dense `(H, W)` arrays per component (x, y) per image.

**What it does:**
```python
im_mesh_A_x = (prev_ux / 2)[..., np.newaxis, np.newaxis] * dx_norm  # (N, ny, nx, tile_h, tile_w)
# Then reshape to (N, H, W) via nested broadcast + addition
```

**Proposed fix — move to C/Cython:**
- This is pure element-wise arithmetic on large arrays — perfect for a C kernel with OpenMP
- The current Python code creates multiple temporary arrays during broadcast
- A single fused C kernel `build_displacement_mesh(prev_ux, prev_uy, win_ctrs, image_shape, out_mesh)` would:
  - Eliminate temporary allocations
  - Parallelize over image rows with OpenMP
  - Potentially halve the 0.28s cost

**Expected saving:** 0.10-0.15s per pass &times; 3 passes = **0.30-0.45s** (9-14% of total at 25 MP)

### Priority 2: Predictor-Corrector — `image_warp` (27-39% of pass time)

**Current:** `cv2.remap()` with `INTER_CUBIC` interpolation, called 2N times per pass (2 images per pair &times; N pairs). Already uses OpenCV's internal parallelism.

**Proposed fixes:**
1. **Use `INTER_LINEAR` instead of `INTER_CUBIC`** for early passes (128, 64) where sub-pixel accuracy matters less. Cubic is ~2x slower than linear interpolation.
   - Expected saving: ~0.15s per early pass (passes 2-3)
2. **Combine remap maps:** Currently calls `cv2.remap` separately for each of 2 images. Could batch into a single call if OpenCV supports it, or use the existing `libinterp2custom.c` which is already compiled.
3. **Investigate `libinterp2custom.c`:** This C extension with Lanczos interpolation already exists in the codebase. Profile it against `cv2.remap` — if faster, switch the warping backend.

**Expected saving:** 0.10-0.20s per pass &times; 3 passes = **0.30-0.60s** (9-18% of total at 25 MP)

### Priority 3: Predictor-Corrector — `dense_remap` (14-19% of pass time)

**Current:** `cv2.remap()` mapping from coarse window grid to dense pixel grid, called 2N times (x and y components per pair).

**Proposed fix — replace with `scipy.ndimage.zoom` or direct bilinear interpolation:**
- The remap from (ny, nx) to (H, W) is a simple upscaling operation
- A custom bilinear upscale kernel would be faster than the general-purpose `cv2.remap`
- Or: pre-compute the mapping once and reuse across passes (the grid-to-pixel mapping is deterministic)

**Expected saving:** 0.05-0.10s per pass &times; 3 passes = **0.15-0.30s** (5-9%)

### Priority 4: Outlier Detection (11-12% at finest grid)

**Current:** `median_2d` with threshold=2.0, epsilon=0.2. Iterates over N images in Python loop.

**Proposed fix — vectorize across images:**
- Currently processes each image separately in a Python `for` loop
- Could batch into a single 3D array operation: `scipy.ndimage.median_filter` on (N, H, W)
- Or: move the per-window median to C with OpenMP

**Expected saving at 25 MP finest pass:** ~0.05-0.07s (4-5% of pass 4)

### Priority 5: Biharmonic Infilling (8-11% at finest grid)

**Current:** `skimage.inplace_biharmonic` or similar, per-image in Python loop.

**Proposed alternatives:**
- **Switch to `local_median` for mid-pass:** Biharmonic is ~3x more expensive than local_median. Since mid-pass infilling only feeds the next pass predictor (not final output), local_median is sufficient.
- **Keep biharmonic only for final pass:** Where interpolation quality matters for output.

**Expected saving:** Use `local_median` for passes 2-3, biharmonic only for pass 4:
- Saves ~0.02-0.05s per intermediate pass at 25 MP

### Priority 6: bulkxcorr2d Peak Fitting (gauss6 vs gauss3)

gauss6 is 2.4-3.7x slower than gauss3 in the correlation step, but this only represents 5-19% of pass time. The total impact is modest:

| Dataset | gauss6 xcorr | gauss3 xcorr | Saved | % of total |
|---------|------------:|------------:|------:|-----------:|
| 4 MP pass 4 | 0.041s | 0.011s | 0.030s | 5.5% |
| 25 MP pass 4 | 0.166s | 0.061s | 0.105s | 3.2% |

**Recommendation:** Use gauss3 for iterative development/testing. Use gauss6 for production runs where sub-pixel accuracy matters. The 6-8% overall speedup from gauss3 is useful but not transformative.

---

## 7. Quick Wins vs Strategic Investments

### Quick Wins (< 1 day implementation)

| Change | Expected Saving | Risk |
|--------|----------------:|------|
| Use `INTER_LINEAR` for passes 2-3 warp | 0.15-0.30s (5-9%) | Slight accuracy loss in early passes (acceptable — later passes correct) |
| Use `local_median` for mid-pass infilling | 0.05-0.10s (2-3%) | None — mid-pass infilling only feeds predictor |
| Use gauss3 for non-final passes | 0.05-0.15s (2-5%) | Slight accuracy loss in intermediate displacement fields |

### Strategic Investments (1-2 weeks)

| Change | Expected Saving | Risk |
|--------|----------------:|------|
| C kernel for `mesh_compute` | 0.30-0.45s (9-14%) | Moderate — needs careful testing of broadcast semantics |
| Profile `libinterp2custom` as warp backend | 0.15-0.30s (5-9%) | Low — C extension already exists |
| Vectorize outlier detection (batch N images) | 0.05-0.10s (2-3%) | Low — straightforward array operation change |

### Combined Theoretical Maximum

If all optimizations were implemented:
- **Quick wins alone:** ~0.25-0.55s saved → **2.70-3.00s** (17% faster at 25 MP)
- **All optimizations:** ~0.80-1.25s saved → **2.00-2.45s** (31-38% faster at 25 MP)

---

## 8. Key Takeaway

The PIV pipeline is **not FFT-bound** — it is **warp-bound**. The predictor-corrector step (image warping, mesh computation, grid remapping) consumes 65-92% of runtime on passes 2-4. Optimizing the cross-correlation or peak fitting yields diminishing returns.

The most impactful single optimization is a **fused C kernel for mesh_compute + dense_remap** that eliminates NumPy temporary arrays and parallelizes with OpenMP. This addresses 35-50% of the predictor-corrector cost, which in turn is 65-92% of pass time.

---

## Appendix: Predictor-Corrector Sub-Timing Breakdown (25 MP, gauss6)

```
                     Pass 2    Pass 3    Pass 4
                     ──────    ──────    ──────
gaussian_smooth      0.001s    0.001s    0.003s    (< 1%)
dense_remap          0.160s    0.171s    0.175s    (14-18%)
mesh_compute         0.277s    0.284s    0.283s    (23-32%)
predictor_remap      0.000s    0.001s    0.004s    (< 1%)
image_warp           0.343s    0.333s    0.337s    (27-39%)
                     ──────    ──────    ──────
PC total             0.802s    0.812s    0.823s
```

Notable: `mesh_compute`, `dense_remap`, and `image_warp` are nearly constant across passes 2-4. They operate on the full image resolution, not the window count. This means the predictor-corrector cost is O(image_pixels), not O(n_windows). Doubling the image size will roughly double these timings.
