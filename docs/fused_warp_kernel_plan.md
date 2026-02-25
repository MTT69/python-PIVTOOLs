# Plan: Fused C Kernel for Predictor-Corrector Image Warping

## Context

The predictor-corrector step dominates PIV pipeline runtime at **54-87% of per-pass cost** (390 ms/pair at 25 MP). Profiling reveals this is not because `cv2.remap` is slow, but because the Python code performs **4 separate full-resolution remaps** plus **~1 GB of temporary float32 allocations** per pair at 25 MP:

1. 2x `cv2.remap` to upsample the coarse predictor field (dx, dy) to pixel resolution
2. Arithmetic to build coordinate maps (`/2`, negate, `+ im_mesh`) — 4 temporary (H,W,2) arrays
3. 2x `cv2.remap` to warp images A and B

A fused C kernel eliminates steps 1-2 entirely by interpolating the coarse predictor inline during the image warp, reducing 4 remaps + ~1 GB temps to **2 image warps + 0 temps**. Expected speedup: ~2.5-3x on the predictor-corrector step.

---

## Files to Create

| File | Purpose |
|------|---------|
| `pivtools_cli/lib/fused_warp.c` | Fused symmetric warp C kernel |
| `pivtools_cli/lib/fused_warp.h` | Header with export macros and function declarations |

## Files to Modify

| File | Change |
|------|--------|
| `setup.py` | Add `libfusedwarp` build step (no FFTW/GSL deps, just OpenMP + math) |
| `pyproject.toml` | Add `lib/fused_warp.c`, `lib/fused_warp.h`, `lib/libfusedwarp.*` to package-data |
| `MANIFEST.in` | Already covered by `recursive-include pivtools_cli/lib *.c *.h` — no change needed |
| `.github/workflows/publish-to-pypi.yml` | No change needed — cibuildwheel runs `setup.py build` which will pick up the new library automatically |
| `pivtools_cli/piv/piv_backend/cpu_instantaneous.py` | Load `libfusedwarp`, replace `_predictor_corrector_batch` internals for pass > 0 |
| `pivtools_cli/piv/piv_backend/cpu_ensemble.py` | Load `libfusedwarp`, replace `_get_im_mesh` + `_get_image_prime_batch` internals |
| `CLAUDE.md` | Document the new library |

---

## Step 1: C Header — `pivtools_cli/lib/fused_warp.h`

```c
#ifndef FUSED_WARP_H
#define FUSED_WARP_H

#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/**
 * Fused symmetric image warp with inline predictor interpolation.
 *
 * For each output pixel (row, col):
 *   1. Bilinearly interpolate the coarse predictor at (row, col)
 *   2. Compute source coords: src_A = pixel + pred/2, src_B = pixel - pred/2
 *   3. Bicubic-interpolate both source images at those coords
 *
 * Eliminates the need for:
 *   - Dense predictor upsampling (cv2.remap on predictor field)
 *   - Temporary coordinate map arrays (im_mesh_A, im_mesh_B)
 *   - Intermediate delta_ab_dense array
 *
 * @param img_a         Source image A, row-major float32 (H x W)
 * @param img_b         Source image B, row-major float32 (H x W)
 * @param out_a         Output warped image A, row-major float32 (H x W), pre-allocated
 * @param out_b         Output warped image B, row-major float32 (H x W), pre-allocated
 * @param pred_dy       Coarse predictor dy, row-major float32 (nPY x nPX)
 * @param pred_dx       Coarse predictor dx, row-major float32 (nPY x nPX)
 * @param H, W          Image dimensions
 * @param nPY, nPX      Coarse predictor grid dimensions (padded)
 * @param ctrs_y        Padded window centre y-coordinates, float32 array of length nPY
 * @param ctrs_x        Padded window centre x-coordinates, float32 array of length nPX
 * @param interp_mode   0 = bicubic (Keys kernel), 1 = bilinear
 *
 * @return 0 on success, nonzero on error
 */
EXPORT int fused_symmetric_warp(
    const float *img_a,
    const float *img_b,
    float       *out_a,
    float       *out_b,
    const float *pred_dy,
    const float *pred_dx,
    int H, int W,
    int nPY, int nPX,
    const float *ctrs_y,
    const float *ctrs_x,
    int interp_mode
);

/**
 * Batch version: warp N image pairs using the SAME predictor field.
 * Images are stacked as (N, H, W) in row-major order.
 * Each pair is processed in parallel via OpenMP.
 */
EXPORT int fused_symmetric_warp_batch(
    const float *imgs_a,       /* (N, H, W) */
    const float *imgs_b,       /* (N, H, W) */
    float       *outs_a,       /* (N, H, W) */
    float       *outs_b,       /* (N, H, W) */
    const float *pred_dy,      /* (nPY, nPX) — shared across batch */
    const float *pred_dx,      /* (nPY, nPX) — shared across batch */
    int N,
    int H, int W,
    int nPY, int nPX,
    const float *ctrs_y,
    const float *ctrs_x,
    int interp_mode
);

#endif /* FUSED_WARP_H */
```

Key decisions:
- **Coarse predictor passed directly** (not the dense upsampled version) — the C kernel does bilinear interpolation inline
- **Window centre arrays** passed instead of pre-built maps — the kernel computes the fractional grid index from pixel position and centre coordinates using `np.interp`-equivalent binary search
- **Batch version** for ensemble mode where N pairs share one predictor field — outer OpenMP loop over N, inner loop over pixels
- **Single-pair version** for instantaneous mode (per-image predictors) — outer OpenMP loop over rows
- **interp_mode flag** supports both bicubic (default) and bilinear

---

## Step 2: C Implementation — `pivtools_cli/lib/fused_warp.c`

### 2a. Bicubic kernel (Keys, a=-0.75 to match OpenCV INTER_CUBIC)

```c
static inline float keys_weight(float t) {
    float at = fabsf(t);
    if (at <= 1.0f)
        return ((1.25f * at - 2.25f) * at) * at + 1.0f;
    if (at < 2.0f)
        return ((-0.75f * at + 3.75f) * at - 6.0f) * at + 3.0f;
    return 0.0f;
}
```

### 2b. Bicubic 2D sample

```c
static inline float bicubic_sample(
    const float *img, int H, int W,
    float y, float x
) {
    if (y < 0.0f || y >= (float)(H - 1) || x < 0.0f || x >= (float)(W - 1))
        return 0.0f;  // BORDER_CONSTANT, borderValue=0

    int iy = (int)floorf(y);
    int ix = (int)floorf(x);
    float fy = y - iy;
    float fx = x - ix;

    float wy[4], wx[4];
    for (int m = 0; m < 4; m++) {
        wy[m] = keys_weight(fy - (m - 1));
        wx[m] = keys_weight(fx - (m - 1));
    }

    float val = 0.0f;
    for (int m = 0; m < 4; m++) {
        int row = iy + m - 1;
        row = row < 0 ? 0 : (row >= H ? H - 1 : row);
        for (int n = 0; n < 4; n++) {
            int col = ix + n - 1;
            col = col < 0 ? 0 : (col >= W ? W - 1 : col);
            val += wy[m] * wx[n] * img[row * W + col];
        }
    }
    return val;
}
```

### 2c. Bilinear 2D sample (for interp_mode=1)

```c
static inline float bilinear_sample(
    const float *img, int H, int W, float y, float x
) {
    if (y < 0.0f || y >= (float)(H - 1) || x < 0.0f || x >= (float)(W - 1))
        return 0.0f;
    int iy = (int)floorf(y);
    int ix = (int)floorf(x);
    float fy = y - iy;
    float fx = x - ix;
    return (1-fy) * ((1-fx) * img[iy*W+ix] + fx * img[iy*W+ix+1])
         +    fy  * ((1-fx) * img[(iy+1)*W+ix] + fx * img[(iy+1)*W+ix+1]);
}
```

### 2d. Inline predictor interpolation

```c
static inline float find_grid_index(float pixel, const float *ctrs, int n) {
    if (pixel <= ctrs[0]) return 0.0f;
    if (pixel >= ctrs[n - 1]) return (float)(n - 1);
    float spacing_approx = (ctrs[n-1] - ctrs[0]) / (float)(n - 1);
    int guess = (int)((pixel - ctrs[0]) / spacing_approx);
    guess = guess < 0 ? 0 : (guess >= n - 1 ? n - 2 : guess);
    while (guess > 0 && ctrs[guess] > pixel) guess--;
    while (guess < n - 2 && ctrs[guess + 1] < pixel) guess++;
    float frac = (pixel - ctrs[guess]) / (ctrs[guess + 1] - ctrs[guess]);
    return (float)guess + frac;
}

static inline float interp_predictor(
    const float *pred, int nPY, int nPX, float gy, float gx
) {
    int iy = (int)floorf(gy);
    int ix = (int)floorf(gx);
    iy = iy < 0 ? 0 : (iy >= nPY - 1 ? nPY - 2 : iy);
    ix = ix < 0 ? 0 : (ix >= nPX - 1 ? nPX - 2 : ix);
    float fy = gy - iy;
    float fx = gx - ix;
    return (1-fy) * ((1-fx) * pred[iy*nPX+ix] + fx * pred[iy*nPX+ix+1])
         +    fy  * ((1-fx) * pred[(iy+1)*nPX+ix] + fx * pred[(iy+1)*nPX+ix+1]);
}
```

### 2e. Main kernels

Single-pair: OpenMP over rows. Batch: `collapse(2)` over (N, rows).

### 2f. Border handling

- Predictor: `BORDER_REPLICATE` (clamping in `find_grid_index` and `interp_predictor`)
- Images: `BORDER_CONSTANT, borderValue=0` (return 0.0 in sample functions)

---

## Step 3: Build System — `setup.py`

Add after `libinterp2custom` block, before `libmarquadt`. No FFTW/GSL dependencies:

```python
# --- Build libfusedwarp ---
if use_msvc:
    output_file = build_dir / f"libfusedwarp{lib_ext}"
    cmd_fw = [
        compiler, *self.extra_compile, shared_flag,
        f"/Fo{build_dir}/",
        str(src_dir / "fused_warp.c"),
        f"/I{src_dir}",
        f"/Fe{output_file}"
    ]
else:
    cmd_fw = [
        compiler, *self.extra_compile, shared_flag,
        str(src_dir / "fused_warp.c"),
        f"-I{src_dir}",
        "-o", str(build_dir / f"libfusedwarp{lib_ext}"),
        "-lm", "-fopenmp"
    ]
self._run(cmd_fw)
if not (build_dir / f"libfusedwarp{lib_ext}").exists():
    raise RuntimeError(f"Build failed: libfusedwarp{lib_ext} not created")
self._cleanup_intermediates(build_dir)
```

---

## Step 4: Package Metadata — `pyproject.toml`

Add to `[tool.setuptools.package-data]` `"pivtools_cli"` list:
```
"lib/fused_warp.c",
"lib/fused_warp.h",
"lib/libfusedwarp.*",
```

**MANIFEST.in** — no change needed (existing `recursive-include pivtools_cli/lib *.c *.h` covers it).

**.github/workflows/publish-to-pypi.yml** — no change needed (cibuildwheel runs setup.py).

---

## Step 5: Python Integration

### 5a. Library loading (both cpu_instantaneous.py and cpu_ensemble.py)

```python
fw_path = os.path.join(lib_dir, f"libfusedwarp{lib_ext}")
if os.path.exists(fw_path):
    self.fused_warp_lib = ctypes.CDLL(fw_path)
    # Set argtypes/restype...
    self._use_fused_warp = True
else:
    self._use_fused_warp = False
```

### 5b. Replace warp path with C call + Python fallback

Keep predictor-to-window-grid remap in Python (tiny grid, negligible cost). Replace dense remap + coord arithmetic + image warp with single C call.

---

## Step 6: Verification

1. **Numerical correctness**: Synthetic image + predictor, compare C vs cv2.remap, assert `np.allclose(atol=1.0)`
2. **Timing**: Use existing `_profile_section` infrastructure, compare old vs new on 4 MP and 25 MP
3. **PIV accuracy**: Run 4000-image noisy validation, compare velocity and stress fields
4. **Cross-platform build**: Local Windows build; CI tests Linux + macOS automatically

---

## Implementation Order

1. Write `fused_warp.h` and `fused_warp.c`
2. Update `setup.py` — add build step
3. Update `pyproject.toml` — add package-data entries
4. Build locally — verify MSVC compilation
5. Write standalone correctness test script
6. Integrate into `cpu_instantaneous.py` with fallback
7. Integrate into `cpu_ensemble.py` with fallback
8. Profile old vs new
9. Update `CLAUDE.md`
