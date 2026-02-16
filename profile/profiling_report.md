# PIV Pipeline Profiling Report

**Date:** 2026-02-16
**Config:** 4-pass multi-grid (128 &rarr; 64 &rarr; 32 &rarr; 16), 50% overlap, Gaussian window, biharmonic infilling, median_2d outlier detection, 4 OMP threads, 3 iterations averaged
**Hardware:** Windows, 20 logical cores, 4 OMP threads

---

## 1. Executive Summary

The pipeline uses a `ThreadPoolExecutor(4)` to parallelise GIL-releasing operations (cv2.remap, scipy.ndimage) across image pairs within a batch. At the production batch size of N=12, threading delivers a **2.2x overall speedup** (3.72s vs 8.19s).

The predictor-corrector step remains the dominant cost (54-73% of per-pass runtime), but threading reduces it from 2.21s to 0.78s at the finest grid — a **2.8x speedup**. The actual FFT cross-correlation (`bulkxcorr2d`) accounts for only 14-26% of pass time.

**Threading is effective for all sections except biharmonic infilling**, which is ~67% GIL-bound. The infilling regression is 9ms (0.2% of total) — not worth special-casing.

---

## 2. Threading Architecture

```
ThreadPoolExecutor(max_workers=4)    ← class-level, reused across passes
cv2.setNumThreads(1)                 ← prevents OpenCV internal threading
OpenMP threads = 4                   ← used only by C library (bulkxcorr2d)

Per-pass execution order (all sequential, no overlap):
  1. pc_gaussian_smooth        → pool (N tasks)     scipy.ndimage, releases GIL
  2. pc_dense_and_pred_remap   → pool (2N tasks)    cv2.remap, releases GIL
  3. pc_mesh_and_image_warp    → pool (N tasks)     cv2.remap + numpy, releases GIL
  4. bulkxcorr2d               → OpenMP (C code)    no pool, 4 OMP threads
  5. outlier_detection         → pool (N tasks)     scipy.ndimage, releases GIL
  6. infilling                 → pool (N tasks)     method-dependent GIL behaviour
```

The pool and OpenMP never run concurrently — they occupy different profiling sections. No thread contention by design.

### GIL Behaviour by Section

| Section | Primary work | GIL released? | Threading benefit |
|---------|-------------|:-------------:|:-----------------:|
| dense_and_pred_remap | cv2.remap | Yes | **3.2x** |
| mesh_and_image_warp | cv2.remap + numpy broadcast | Yes | **2.7x** |
| outlier_detection | scipy.ndimage.median_filter | Yes | **2.8x** |
| gaussian_smooth | scipy.ndimage.gaussian_filter | Yes | Good |
| infilling (local_median) | bottleneck.nanmedian (C) | Yes | **2.1x** |
| infilling (biharmonic) | skimage sparse solver | ~33% | **0.95x** (overhead > gain) |

---

## 3. Threading ON vs OFF

### Grand Totals — 4 MP (2048&times;2048)

| Batch size | Threading ON | Threading OFF | Speedup |
|:----------:|------------:|-------------:|--------:|
| N=1 | 0.611s | 0.695s | **1.14x** |
| N=12 | **3.716s** | **8.187s** | **2.20x** |

### Grand Totals — 25 MP (4600&times;5312)

| Batch size | Threading ON | Threading OFF | Speedup |
|:----------:|------------:|-------------:|--------:|
| N=1 | 3.611s | 4.222s | **1.17x** |
| N=12 | **21.102s** | — | — |

Per-pair at N=12: **1.76 s/pair** (21.102s / 12).

Threading helps even at N=1 because `dense_and_pred_remap` submits 2 tasks (d=0, d=1) per image. At N=12, the full 4-thread pool is utilised across all sections.

---

## 4. Per-Section Breakdown: 4 MP, N=12, Pass 4 (16&times;16, 65,025 windows)

This is the production-relevant configuration.

| Section | Threading ON | Threading OFF | Speedup | Notes |
|---------|------------:|-------------:|--------:|-------|
| **predictor_corrector** | **0.782s** | **2.211s** | **2.83x** | |
| &nbsp;&nbsp; dense_and_pred_remap | 0.250s | 0.809s | 3.24x | 2N cv2.remap calls |
| &nbsp;&nbsp; mesh_and_image_warp | 0.517s | 1.387s | 2.68x | N&times;2 cv2.remap + mesh compute |
| bulkxcorr2d | 0.369s | 0.418s | 1.13x | OpenMP (not pooled) |
| outlier_detection | 0.080s | 0.220s | 2.75x | scipy.ndimage |
| infilling (biharmonic) | 0.167s | 0.158s | 0.95x | GIL-bound, see &sect;6 |
| **TOTAL** | **1.438s** | **3.044s** | **2.12x** | |

---

## 5. Per-Pass Breakdown: 4 MP, N=12, Threading ON

```
Pass 1 ██░░░░░░░░░░░░░░░░░░ 0.26s   7.0%  (31x31 = 961 windows)
Pass 2 ██████████████████████████░░░░░░░░░░ 1.01s  27.2%  (63x63 = 3,969 windows)
Pass 3 ██████████████████████████████░░░░░░ 1.01s  27.2%  (127x127 = 16,129 windows)
Pass 4 █████████████████████████████████████ 1.44s  38.7%  (255x255 = 65,025 windows)
                                              ─────
                                              3.72s
```

### Pass 1 (128&times;128, 961 windows)

| Section | Time | % |
|---------|-----:|--:|
| predictor_corrector | 0.027s | 10% |
| bulkxcorr2d | 0.186s | 72% |
| outlier_detection | 0.005s | 2% |
| infilling | 0.026s | 10% |
| **TOTAL** | **0.259s** | |

Pass 1 is the only pass where `bulkxcorr2d` dominates (no predictor-corrector needed).

### Pass 2 (64&times;64, 3,969 windows)

| Section | Time | % |
|---------|-----:|--:|
| **predictor_corrector** | **0.785s** | **78%** |
| &nbsp;&nbsp; dense_and_pred_remap | 0.236s | 23% |
| &nbsp;&nbsp; mesh_and_image_warp | 0.536s | 53% |
| bulkxcorr2d | 0.140s | 14% |
| outlier_detection | 0.012s | 1% |
| infilling | 0.060s | 6% |
| **TOTAL** | **1.011s** | |

### Pass 3 (32&times;32, 16,129 windows)

| Section | Time | % |
|---------|-----:|--:|
| **predictor_corrector** | **0.782s** | **77%** |
| &nbsp;&nbsp; dense_and_pred_remap | 0.236s | 23% |
| &nbsp;&nbsp; mesh_and_image_warp | 0.533s | 53% |
| bulkxcorr2d | 0.128s | 13% |
| outlier_detection | 0.030s | 3% |
| infilling | 0.062s | 6% |
| **TOTAL** | **1.008s** | |

### Pass 4 (16&times;16, 65,025 windows)

| Section | Time | % |
|---------|-----:|--:|
| **predictor_corrector** | **0.782s** | **54%** |
| &nbsp;&nbsp; dense_and_pred_remap | 0.250s | 17% |
| &nbsp;&nbsp; mesh_and_image_warp | 0.517s | 36% |
| bulkxcorr2d | 0.369s | 26% |
| outlier_detection | 0.080s | 6% |
| infilling | 0.167s | 12% |
| **TOTAL** | **1.438s** | |

The predictor-corrector cost is nearly constant across passes 2-4 (~0.78s each) because it operates on the full image resolution, not per-window. The per-window costs (xcorr, outlier, infilling) scale with window count and become significant only at the finest grid.

### Per-Pass Breakdown: 25 MP, N=12, Threading ON

```
Pass 1 ██░░░░░░░░░░░░░░░░░░ 1.38s   6.6%  (70x82 = 5,740 windows)
Pass 2 █████████████████████████░░░░░░░░░░░ 5.48s  26.0%  (142x165 = 23,430 windows)
Pass 3 ████████████████████████████░░░░░░░░ 6.12s  29.0%  (286x331 = 94,666 windows)
Pass 4 ██████████████████████████████████████ 8.12s  38.5%  (574x663 = 380,562 windows)
                                              ─────
                                              21.10s
```

### Pass 4 (16&times;16, 380,562 windows) — 25 MP

| Section | Time | % |
|---------|-----:|--:|
| **predictor_corrector** | **4.773s** | **59%** |
| &nbsp;&nbsp; dense_and_pred_remap | 1.526s | 19% |
| &nbsp;&nbsp; mesh_and_image_warp | 3.160s | 39% |
| bulkxcorr2d | 1.931s | 24% |
| outlier_detection | 0.539s | 7% |
| infilling | 0.602s | 7% |
| **TOTAL** | **8.118s** | |

The predictor-corrector cost scales with pixel count (~6x from 4 MP to 25 MP, matching the 6.1x pixel ratio). Per-window costs (xcorr, outlier, infilling) also increase proportionally with the larger grid.

---

## 6. Biharmonic Infilling: Why Threading Doesn't Help

Isolated GIL test on a 255&times;255 PIV grid (5% outliers):

```
Sequential 12 calls:           0.061s  (5.0ms each)
Threaded 12 calls (4 workers): 0.046s  (1.33x speedup)
```

`inpaint_biharmonic` is **~67% GIL-bound** (sparse matrix construction in Python, `scipy.sparse.linalg.spsolve`). The 33% that releases the GIL gives a theoretical 1.33x max speedup with 4 threads. In practice, the closure creation and future collection overhead (~1ms per batch) eats the gain.

**Impact:** 9ms regression at N=12 pass 4 (0.158s &rarr; 0.167s) = **0.2% of pipeline total**. Not worth special-casing — the same `_run_parallel` path that gives 2.8x speedup on cv2.remap operations.

**Recommendation:** Use `local_median` for mid-pass infilling (faster, threads well at 2.1x). Reserve biharmonic for the final pass only if interpolation quality matters for the output.

---

## 7. Cost Scaling: Where Time Goes (N=12, Threading ON)

### 4 MP

```
                     Pass 1    Pass 2    Pass 3    Pass 4    TOTAL
                     ──────    ──────    ──────    ──────    ─────
predictor_corrector  0.027s    0.785s    0.782s    0.782s    2.376s  (64%)
bulkxcorr2d          0.186s    0.140s    0.128s    0.369s    0.823s  (22%)
outlier_detection    0.005s    0.012s    0.030s    0.080s    0.127s   (3%)
infilling            0.026s    0.060s    0.062s    0.167s    0.315s   (8%)
other                0.015s    0.014s    0.006s    0.040s    0.075s   (2%)
                     ──────    ──────    ──────    ──────    ─────
                     0.259s    1.011s    1.008s    1.438s    3.716s
```

### 25 MP

```
                     Pass 1    Pass 2    Pass 3    Pass 4    TOTAL
                     ──────    ──────    ──────    ──────    ─────
predictor_corrector  0.639s    4.719s    4.719s    4.773s   14.850s  (70%)
bulkxcorr2d          0.689s    0.638s    0.996s    1.931s    4.254s  (20%)
outlier_detection    0.008s    0.038s    0.129s    0.539s    0.714s   (3%)
infilling            0.048s    0.084s    0.204s    0.602s    0.938s   (4%)
other                         ~0.001    ~0.075    ~0.273    ~0.346s  (~2%)
                     ──────    ──────    ──────    ──────    ─────
                     1.383s    5.479s    6.123s    8.118s   21.102s
```

Predictor-corrector dominates at 64% (4 MP) and 70% (25 MP) of total. The constant cost per pass (passes 2-4) is O(image_pixels), not O(n_windows). The 25 MP dataset shows the same pattern at ~6x the cost, matching the pixel ratio.

---

## 8. Optimization Opportunities (Updated)

### Priority 1: Predictor-Corrector (64% of total, 2.38s)

Already threaded with 2.8x speedup. Remaining optimization targets:

**mesh_and_image_warp** (0.52s per pass, 36% of pass 4):
- `cv2.remap` with `INTER_CUBIC` is the core cost
- `INTER_LINEAR` for early passes (128, 64) would be ~2x faster per remap with negligible accuracy loss
- Expected saving: ~0.15s per early pass

**dense_and_pred_remap** (0.25s per pass, 17% of pass 4):
- Currently remaps from coarse (ny, nx) grid to full (H, W) resolution
- Pre-computing and caching the dense maps is already done; the remap itself is the bottleneck

**C kernel for mesh compute** (inside mesh_and_image_warp):
- The numpy broadcast `map_A = im_mesh - 0.5 * delta_ab_dense[i]` allocates ~195 MB temporaries at 25 MP
- A fused C kernel would eliminate temporary allocations and parallelise with OpenMP
- Expected saving: ~0.05-0.10s per pass

### Priority 2: bulkxcorr2d (22% of total, 0.82s)

Uses OpenMP internally. Scaling is good but limited by FFTW plan reuse and memory bandwidth. Peak finder choice matters:

| Peak finder | Pass 4 xcorr time | Notes |
|-------------|------------------:|-------|
| gauss3 | ~0.15s | 3-point parabolic, 2.4x faster |
| gauss6 | ~0.37s | 6-DOF elliptical Gaussian, most accurate |

Recommendation: gauss3 for iterative development, gauss6 for production.

### Priority 3: Infilling (8% of total, 0.32s)

| Method | Pass 4 (N=12) | Threading speedup | Notes |
|--------|-------------:|-----------:|-------|
| local_median | 0.019s | 2.1x | Fast, threads well, sufficient for mid-pass |
| biharmonic | 0.167s | 0.95x (loses) | Slow, GIL-bound, use for final pass only |

**Recommendation:** `local_median` for mid-pass, biharmonic only for final pass. This alone would save ~0.13s across passes 1-3.

### Priority 4: Outlier Detection (3% of total, 0.13s)

Already threaded with 2.75x speedup. Well-optimised for current workload.

---

## 9. Historical Comparison

| Metric | Feb 7 (N=1) | Feb 16 (N=1) | Feb 16 (N=12) | Notes |
|--------|------------:|-------------:|--------------:|-------|
| 4 MP grand total | 0.549s | 0.611s | 3.716s | N=1 +11% from code restructuring |
| 4 MP per-pair at N=12 | — | — | 0.310s | 3.716s / 12 |
| 25 MP grand total | 3.250s | 3.611s | 21.102s | Same pattern |
| 25 MP per-pair at N=12 | — | — | 1.758s | 21.102s / 12 |

The ~11% N=1 regression comes from code restructuring (fused predictor-corrector operations, different memory allocation patterns), not from threading overhead. At N=12, per-image throughput (0.31s/pair at 4 MP, 1.76s/pair at 25 MP) is far better than N=1 due to thread utilisation.

---

## 10. Key Takeaways

1. **Threading gives 2.2x overall speedup at N=12.** The pool is correctly sized to OMP threads (4), with no contention between pool and OpenMP (they run in different sections).

2. **The pipeline is warp-bound, not FFT-bound.** Predictor-corrector (cv2.remap) is 64% of runtime even after 2.8x threading speedup. The FFT cross-correlation is only 22%.

3. **GIL behaviour determines threading effectiveness.** cv2.remap and scipy.ndimage release the GIL (2.7-3.2x speedup). Biharmonic infilling is 67% GIL-bound (no benefit). Know your library's GIL behaviour before threading.

4. **Batch size matters.** N=1 gets 1.14x from threading. N=12 gets 2.20x. Production batch sizes of 12+ fully utilise the 4-thread pool.

5. **Use local_median for mid-pass infilling.** It's 8.8x faster than biharmonic and threads at 2.1x. Reserve biharmonic for the final pass where interpolation quality matters.

---

## Appendix A: Predictor-Corrector Sub-Timing (4 MP, N=12, Threading ON)

```
                     Pass 2    Pass 3    Pass 4
                     ──────    ──────    ──────
gaussian_smooth      0.002s    0.002s    0.002s    (< 1%)
dense_and_pred_remap 0.236s    0.236s    0.250s    (17-23%)
mesh_and_image_warp  0.536s    0.533s    0.517s    (36-53%)
                     ──────    ──────    ──────
PC total             0.785s    0.782s    0.782s
```

Nearly constant across passes 2-4 — these operations scale with image resolution, not window count.

## Appendix A.2: Predictor-Corrector Sub-Timing (25 MP, N=12, Threading ON)

```
                     Pass 2    Pass 3    Pass 4
                     ──────    ──────    ──────
gaussian_smooth      0.009s    0.009s    0.008s    (< 1%)
dense_and_pred_remap 1.534s    1.526s    1.526s    (19%)
mesh_and_image_warp  3.113s    3.125s    3.160s    (39%)
                     ──────    ──────    ──────
PC total             4.719s    4.719s    4.773s
```

Same pattern as 4 MP: nearly constant across passes 2-4. Cost is ~6x higher, matching the 6.1x pixel ratio. Per-pair: dense_and_pred_remap ~0.127s, mesh_and_image_warp ~0.263s.

## Appendix B: Reproducing These Results

```bash
# N=1, threading ON (default)
python profile/profile_piv.py 4mp --infilling biharmonic

# N=12, threading ON
python profile/profile_piv.py 4mp --pairs 12 --infilling biharmonic

# N=12, threading OFF (baseline)
python profile/profile_piv.py 4mp --pairs 12 --infilling biharmonic --no-threading

# 25 MP dataset, N=1
python profile/profile_piv.py 25mp --pairs 1 --infilling biharmonic

# 25 MP dataset, N=12
python profile/profile_piv.py 25mp --pairs 12 --infilling biharmonic

# Both datasets
python profile/profile_piv.py both --infilling biharmonic
```

Flags: `--pairs N`, `--iterations N`, `--threads N`, `--windows 128,64,32,16`, `--infilling METHOD`, `--peak-finder gauss3|gauss6`, `--no-outlier`, `--no-warmup`, `--no-threading`
