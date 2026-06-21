/*
 * fused_warp.c — Fused symmetric image warp for predictor-corrector PIV.
 *
 * Combines predictor upsampling + symmetric coordinate map + bicubic/Lanczos image
 * warp into a single OpenMP-parallel pass over output pixels.
 *
 * Build (MSVC):
 *   cl /O2 /std:c11 /openmp:experimental /MT /LD fused_warp.c /Fe:libfusedwarp.dll
 */

#define _USE_MATH_DEFINES   /* M_PI on MSVC — must precede any <math.h> include */

#include "fused_warp.h"
#include "common.h"

#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* ─────────────────────────────────────────────────────────────────────────── */
/* Implementation selector (scalar reference vs SIMD), mirroring the            */
/* bulkxcorr2d_set_timing_enabled flag pattern. Lets a single built .so run     */
/* either path so the test suite can assert scalar==SIMD equivalence and the    */
/* microbenchmark can A/B them. Default = SIMD. Callers read it into a local     */
/* once, before the OpenMP region — never per-iteration (volatile + thread-safe).*/
/* ─────────────────────────────────────────────────────────────────────────── */
static volatile int g_warp_impl = 1;   /* 0 = scalar reference, 1 = SIMD */
EXPORT void fused_warp_set_impl(int impl) { g_warp_impl = impl ? 1 : 0; }
EXPORT int  fused_warp_get_impl(void)     { return g_warp_impl; }

/* ─────────────────────────────────────────────────────────────────────────── */
/* Helpers                                                                     */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Branchless Keys bicubic weights (a = -0.75, matches cv2.INTER_CUBIC)       */
/*                                                                             */
/* For fractional position d in [0,1), computes the 4 weights at stencil      */
/* offsets -1, 0, +1, +2 from floor.  Eliminates branches and fabsf from      */
/* the general keys_weight() by exploiting the known |t| ranges:              */
/*   w[0]: |t| = 1+d  in [1,2)  — outer lobe                                */
/*   w[1]: |t| = d    in [0,1)  — inner lobe                                */
/*   w[2]: |t| = 1-d  in (0,1]  — inner lobe                                */
/*   w[3]: |t| = 2-d  in (1,2]  — outer lobe                                */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void keys_weights_4(float d, float w[4]) {
    float s0 = 1.0f + d;     /* |t| for stencil offset -1 */
    float q2 = 1.0f - d;     /* |t| for stencil offset +1 */
    float s3 = 2.0f - d;     /* |t| for stencil offset +2 */

    /* Inner lobe: (1.25*|t| - 2.25)*|t|^2 + 1   [a+2=1.25, a+3=2.25] */
    w[1] = (1.25f * d  - 2.25f) * d  * d  + 1.0f;
    w[2] = (1.25f * q2 - 2.25f) * q2 * q2 + 1.0f;

    /* Outer lobe: -0.75*|t|^3 + 3.75*|t|^2 - 6*|t| + 3 */
    w[0] = ((-0.75f * s0 + 3.75f) * s0 - 6.0f) * s0 + 3.0f;
    w[3] = ((-0.75f * s3 + 3.75f) * s3 - 6.0f) * s3 + 3.0f;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Bicubic sample from image with BORDER_CONSTANT = 0                         */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline float bicubic_sample(const float *img, float fy, float fx, int H, int W,
                                   int use_simd) {
    /* CSE: store floorf result to avoid MSVC calling it twice */
    float fy_floor = floorf(fy);
    float fx_floor = floorf(fx);
    int iy = (int)fy_floor - 1;
    int ix = (int)fx_floor - 1;
    float dy = fy - fy_floor;
    float dx = fx - fx_floor;
    float wy[4], wx[4];
    float val = 0.0f;
    int m, n, row, col;

    keys_weights_4(dy, wy);
    keys_weights_4(dx, wx);

    /* Interior fast path (impl=1): the whole 4x4 stencil is in bounds, so the
       per-tap bounds checks are provably dead. Same tap order (m outer, n inner)
       and same products as the border path → bit-identical, just branchless. This
       is the loop the SIMD path vectorises. impl=0 forces the bounds-checked path
       everywhere — the scalar reference oracle. */
    if (use_simd && iy >= 0 && iy + 3 < H && ix >= 0 && ix + 3 < W) {
        for (m = 0; m < 4; m++) {
            const float *r = img + (size_t)(iy + m) * W + ix;
            for (n = 0; n < 4; n++) {
                val += wy[m] * wx[n] * r[n];
            }
        }
    } else {
        for (m = 0; m < 4; m++) {
            row = iy + m;
            if (row < 0 || row >= H) continue;  /* BORDER_CONSTANT = 0 */
            for (n = 0; n < 4; n++) {
                col = ix + n;
                if (col < 0 || col >= W) continue;  /* BORDER_CONSTANT = 0 */
                val += wy[m] * wx[n] * img[row * W + col];
            }
        }
    }
    return val;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Lanczos-3 windowed sinc kernel (a=3, 6×6 stencil)                          */
/*                                                                             */
/* L(x) = sinc(x) * sinc(x/a)  for |x| < a                                   */
/*       = 0                    for |x| >= a                                   */
/*       = 1                    for |x| ≈ 0                                    */
/*                                                                             */
/* Sharper than bicubic (better frequency preservation for particle images),   */
/* at the cost of 36 vs 16 samples per pixel plus sinf calls.                  */
/* Predictor upsampling stays bicubic regardless — smooth displacement fields  */
/* don't benefit from Lanczos's sharper frequency response.                    */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline float lanczos_weight(float t, int a) {
    float at = fabsf(t);
    if (at < 1e-6f) return 1.0f;
    if (at >= (float)a) return 0.0f;
    float pi_t = (float)M_PI * at;
    float pi_t_a = pi_t / (float)a;
    return (sinf(pi_t) / pi_t) * (sinf(pi_t_a) / pi_t_a);
}

static inline void lanczos3_weights_6(float d, float w[6]) {
    /* Stencil offsets: -2, -1, 0, +1, +2, +3 from floor */
    int k;
    for (k = 0; k < 6; k++) {
        w[k] = lanczos_weight(d - (float)(k - 2), 3);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Lanczos-3 weight LUT                                                        */
/*                                                                             */
/* Precompute the 6 Lanczos-3 weights at 4096 uniformly spaced fractional     */
/* positions d in [0, 1].  At runtime each pixel does a table lookup + lerp   */
/* instead of 12 sinf calls → ~10x faster weight computation.                 */
/*                                                                             */
/* Table size: (4096+1) × 6 × 4 bytes ≈ 96 KB — fits comfortably in L2.      */
/* Built once per fused_symmetric_warp call (sequential, ~0.1 ms).            */
/* ─────────────────────────────────────────────────────────────────────────── */

#define LANCZOS3_LUT_SIZE 4096

static void build_lanczos3_lut(float (*lut)[6], int size) {
    int i, k;
    for (i = 0; i <= size; i++) {
        float d = (float)i / (float)size;
        for (k = 0; k < 6; k++) {
            lut[i][k] = lanczos_weight(d - (float)(k - 2), 3);
        }
    }
}

static inline void lanczos3_weights_6_lut(float d, const float (*lut)[6], float w[6]) {
    float pos = d * (float)LANCZOS3_LUT_SIZE;
    int idx = (int)pos;
    float frac = pos - (float)idx;
    int k;
    if (idx >= LANCZOS3_LUT_SIZE) { idx = LANCZOS3_LUT_SIZE - 1; frac = 0.0f; }
    for (k = 0; k < 6; k++) {
        w[k] = lut[idx][k] + frac * (lut[idx + 1][k] - lut[idx][k]);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Lanczos-3 sample from image with BORDER_CONSTANT = 0  (LUT-accelerated)    */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline float lanczos3_sample(const float *img, float fy, float fx,
                                    int H, int W, const float (*lut)[6], int use_simd) {
    float fy_floor = floorf(fy);
    float fx_floor = floorf(fx);
    int iy = (int)fy_floor - 2;   /* stencil starts at floor-2 */
    int ix = (int)fx_floor - 2;
    float dy = fy - fy_floor;
    float dx = fx - fx_floor;
    float wy[6], wx[6];
    float val = 0.0f;
    int m, n, row, col;

    lanczos3_weights_6_lut(dy, lut, wy);
    lanczos3_weights_6_lut(dx, lut, wx);

    /* Interior fast path (impl=1): whole 6x6 stencil in bounds → branchless,
       bit-identical to the border path (same m-outer/n-inner order). The SIMD path
       targets this. impl=0 forces the bounds-checked path — the scalar reference. */
    if (use_simd && iy >= 0 && iy + 5 < H && ix >= 0 && ix + 5 < W) {
        for (m = 0; m < 6; m++) {
            const float *r = img + (size_t)(iy + m) * W + ix;
            for (n = 0; n < 6; n++) {
                val += wy[m] * wx[n] * r[n];
            }
        }
        return val;
    }

    for (m = 0; m < 6; m++) {
        row = iy + m;
        if (row < 0 || row >= H) continue;  /* BORDER_CONSTANT = 0 */
        for (n = 0; n < 6; n++) {
            col = ix + n;
            if (col < 0 || col >= W) continue;  /* BORDER_CONSTANT = 0 */
            val += wy[m] * wx[n] * img[row * W + col];
        }
    }
    return val;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Bicubic predictor interpolation with precomputed y-weights                  */
/*                                                                             */
/* The predictor y-index (fiy) is constant for all columns in a row, so the   */
/* caller precomputes the 4 y-weights and base row index once per row.         */
/* Only the x-weights vary per pixel — saves 8 keys_weight evals per pixel.   */
/* Uses BORDER_REPLICATE (clampi) for predictor grid boundaries.               */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline float bicubic_pred_wy(
    const float *pred, const float *wy, int iy_base,
    float fx, int nPY, int nPX
) {
    float fx_floor = floorf(fx);
    int ix = (int)fx_floor - 1;
    float dx = fx - fx_floor;
    float wx[4];
    float val = 0.0f;
    int m, n, row, col;

    keys_weights_4(dx, wx);

    for (m = 0; m < 4; m++) {
        row = clampi(iy_base + m, 0, nPY - 1);  /* BORDER_REPLICATE */
        for (n = 0; n < 4; n++) {
            col = clampi(ix + n, 0, nPX - 1);
            val += wy[m] * wx[n] * pred[row * nPX + col];
        }
    }
    return val;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 1D linear lookup: map pixel coordinate to fractional predictor index        */
/*                                                                             */
/* Equivalent to np.interp(pixel_coords, ctrs, arange(len(ctrs)))             */
/* with clamp (BORDER_REPLICATE) outside the centre range.                     */
/* ─────────────────────────────────────────────────────────────────────────── */

static void build_pred_index_lut(
    float *lut,          /* output: lut[N] fractional predictor indices */
    int N,               /* number of pixels (H or W) */
    const float *ctrs,   /* window centres, length nC, assumed sorted strictly ascending */
    int nC               /* number of centres */
) {
    int p, seg;
    float coord, denom, t;

    if (nC == 1) {
        /* Single centre: all pixels map to index 0 */
        for (p = 0; p < N; p++) {
            lut[p] = 0.0f;
        }
        return;
    }

    seg = 0;  /* current segment index */
    for (p = 0; p < N; p++) {
        coord = (float)p;

        /* Clamp below first centre */
        if (coord <= ctrs[0]) {
            lut[p] = 0.0f;
            continue;
        }
        /* Clamp above last centre */
        if (coord >= ctrs[nC - 1]) {
            lut[p] = (float)(nC - 1);
            continue;
        }

        /* Advance segment until we bracket coord */
        while (seg < nC - 2 && ctrs[seg + 1] < coord) {
            seg++;
        }

        /* Guard against equal consecutive centres (division by zero) */
        denom = ctrs[seg + 1] - ctrs[seg];
        if (denom < 1e-12f) {
            lut[p] = (float)seg;
            continue;
        }

        /* Linear interpolation within segment */
        t = (coord - ctrs[seg]) / denom;
        lut[p] = (float)seg + t;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Per-row warp kernels — single source of truth for the inner loop, shared by  */
/* both the single-pair and batch entry points. The Phase A (predictor interp) / */
/* B (symmetric coords) / C (image sample) arithmetic is byte-for-byte the prior  */
/* inline bodies; the only parameterisation is the image / predictor / output     */
/* base pointers, so SIMD added here lands in every call site at once.            */
/* ─────────────────────────────────────────────────────────────────────────── */

static inline void warp_row_bicubic(
    const float *img_a, const float *img_b,
    float *out_a, float *out_b,
    const float *pred_dy, const float *pred_dx,
    const float *pred_idx_x, const float *pred_wy, int pred_iy_base,
    int i, int H, int W, int nPY, int nPX, int use_simd
) {
    int j;
    for (j = 0; j < W; j++) {
        float fix = pred_idx_x[j];
        int idx = i * W + j;

        /* Phase A: interpolate predictor (y-weights precomputed by caller) */
        float dense_dy = bicubic_pred_wy(pred_dy, pred_wy, pred_iy_base, fix, nPY, nPX);
        float dense_dx = bicubic_pred_wy(pred_dx, pred_wy, pred_iy_base, fix, nPY, nPX);

        /* Phase B: symmetric warp coordinates */
        float half_dy = 0.5f * dense_dy;
        float half_dx = 0.5f * dense_dx;

        /* Phase C: bicubic image sample */
        out_a[idx] = bicubic_sample(img_a, (float)i - half_dy, (float)j - half_dx, H, W, use_simd);
        out_b[idx] = bicubic_sample(img_b, (float)i + half_dy, (float)j + half_dx, H, W, use_simd);
    }
}

static inline void warp_row_lanczos(
    const float *img_a, const float *img_b,
    float *out_a, float *out_b,
    const float *pred_dy, const float *pred_dx,
    const float *pred_idx_x, const float *pred_wy, int pred_iy_base,
    int i, int H, int W, int nPY, int nPX,
    const float (*lut)[6], int use_simd
) {
    int j;
    for (j = 0; j < W; j++) {
        float fix = pred_idx_x[j];
        int idx = i * W + j;

        /* Phase A: bicubic predictor interpolation (predictor field stays bicubic) */
        float dense_dy = bicubic_pred_wy(pred_dy, pred_wy, pred_iy_base, fix, nPY, nPX);
        float dense_dx = bicubic_pred_wy(pred_dx, pred_wy, pred_iy_base, fix, nPY, nPX);

        /* Phase B: symmetric warp coordinates */
        float half_dy = 0.5f * dense_dy;
        float half_dx = 0.5f * dense_dx;

        /* Phase C: Lanczos-3 image sample (LUT weights) */
        out_a[idx] = lanczos3_sample(img_a, (float)i - half_dy, (float)j - half_dx, H, W, lut, use_simd);
        out_b[idx] = lanczos3_sample(img_b, (float)i + half_dy, (float)j + half_dx, H, W, lut, use_simd);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Main exported function                                                      */
/* ─────────────────────────────────────────────────────────────────────────── */

EXPORT int fused_symmetric_warp(
    const float *img_a, const float *img_b,
    float *out_a, float *out_b,
    const float *pred_dy, const float *pred_dx,
    int H, int W,
    int nPY, int nPX,
    const float *ctrs_y, const float *ctrs_x,
    int interp_mode
) {
    float *pred_idx_y, *pred_idx_x;

    /* Input validation: reject degenerate dimensions and null pointers */
    if (H <= 0 || W <= 0 || nPY <= 0 || nPX <= 0) return ERROR_NOMEM;
    if (!img_a || !img_b || !out_a || !out_b)       return ERROR_NOMEM;
    if (!pred_dy || !pred_dx || !ctrs_y || !ctrs_x)  return ERROR_NOMEM;

    /* Allocate 1D lookup tables: pixel coord -> fractional predictor index */
    pred_idx_y = (float *)malloc((size_t)H * sizeof(float));
    pred_idx_x = (float *)malloc((size_t)W * sizeof(float));
    if (!pred_idx_y || !pred_idx_x) {
        free(pred_idx_y);
        free(pred_idx_x);
        return ERROR_NOMEM;
    }

    /* Build 1D lookup tables (sequential — tiny cost) */
    build_pred_index_lut(pred_idx_y, H, ctrs_y, nPY);
    build_pred_index_lut(pred_idx_x, W, ctrs_x, nPX);

    /* interp_mode branch is hoisted outside the parallel region so the
       compiler can optimize each path independently (no per-pixel branch). */
    int use_simd = g_warp_impl;  /* read once, before any OpenMP region */

    if (interp_mode == 0) {
        /* ── Bicubic image warp ───────────────────────────────────────── */
        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < H; i++) {
            float fiy = pred_idx_y[i];
            float fiy_floor = floorf(fiy);
            int pred_iy_base = (int)fiy_floor - 1;
            float pred_frac_dy = fiy - fiy_floor;
            float pred_wy[4];
            keys_weights_4(pred_frac_dy, pred_wy);

            warp_row_bicubic(img_a, img_b, out_a, out_b, pred_dy, pred_dx,
                             pred_idx_x, pred_wy, pred_iy_base, i, H, W, nPY, nPX, use_simd);
        }
    } else {
        /* ── Lanczos-3 image warp (6×6 stencil, LUT-accelerated) ──── */
        float (*lanc_lut)[6] = (float (*)[6])malloc(
            (size_t)(LANCZOS3_LUT_SIZE + 1) * 6 * sizeof(float));
        if (!lanc_lut) {
            free(pred_idx_y); free(pred_idx_x);
            return ERROR_NOMEM;
        }
        build_lanczos3_lut(lanc_lut, LANCZOS3_LUT_SIZE);

        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < H; i++) {
            float fiy = pred_idx_y[i];
            float fiy_floor = floorf(fiy);
            int pred_iy_base = (int)fiy_floor - 1;
            float pred_frac_dy = fiy - fiy_floor;
            float pred_wy[4];
            keys_weights_4(pred_frac_dy, pred_wy);

            warp_row_lanczos(img_a, img_b, out_a, out_b, pred_dy, pred_dx,
                             pred_idx_x, pred_wy, pred_iy_base, i, H, W, nPY, nPX,
                             lanc_lut, use_simd);
        }

        free(lanc_lut);
    }

    free(pred_idx_y);
    free(pred_idx_x);
    return ERROR_NONE;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Batch version: warp N image pairs in a single OpenMP-parallel pass         */
/*                                                                             */
/* Same Phase A/B/C logic as single-pair, but loops over (image, row) with    */
/* collapse(2) for fine-grained work distribution.  1D LUTs built once,       */
/* shared across all images.  Predictor can be shared (ensemble) or per-image */
/* (instantaneous) via shared_predictor flag.                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

EXPORT int fused_symmetric_warp_batch(
    const float *imgs_a, const float *imgs_b,
    float *outs_a, float *outs_b,
    const float *pred_dy, const float *pred_dx,
    int N, int H, int W,
    int nPY, int nPX,
    const float *ctrs_y, const float *ctrs_x,
    int interp_mode, int shared_predictor
) {
    float *pred_idx_y, *pred_idx_x;

    /* Input validation */
    if (N <= 0 || H <= 0 || W <= 0 || nPY <= 0 || nPX <= 0) return ERROR_NOMEM;
    if (!imgs_a || !imgs_b || !outs_a || !outs_b)             return ERROR_NOMEM;
    if (!pred_dy || !pred_dx || !ctrs_y || !ctrs_x)           return ERROR_NOMEM;

    /* Build shared 1D LUTs once (same ctrs for all images) */
    pred_idx_y = (float *)malloc((size_t)H * sizeof(float));
    pred_idx_x = (float *)malloc((size_t)W * sizeof(float));
    if (!pred_idx_y || !pred_idx_x) {
        free(pred_idx_y);
        free(pred_idx_x);
        return ERROR_NOMEM;
    }
    build_pred_index_lut(pred_idx_y, H, ctrs_y, nPY);
    build_pred_index_lut(pred_idx_x, W, ctrs_x, nPX);

    /* Predictor stride: 0 if shared (ensemble), nPY*nPX if per-image (instantaneous) */
    int pred_stride = shared_predictor ? 0 : nPY * nPX;
    int use_simd = g_warp_impl;  /* read once, before the OpenMP region */

    /* Manual loop flattening: collapse(2) crashes on MSVC /openmp:experimental
     * with large iteration counts. Flatten (ni, i) → single index `ti`. */
    int total_rows = N * H;

    if (interp_mode == 0) {
        /* ── Bicubic image warp ───────────────────────────────────────────── */
        int ti;
        #pragma omp parallel for schedule(static)
        for (ti = 0; ti < total_rows; ti++) {
            int ni = ti / H;
            int i  = ti % H;

            const float *cur_pred_dy = pred_dy + ni * pred_stride;
            const float *cur_pred_dx = pred_dx + ni * pred_stride;
            const float *cur_img_a = imgs_a + (size_t)ni * H * W;
            const float *cur_img_b = imgs_b + (size_t)ni * H * W;
            float *cur_out_a = outs_a + (size_t)ni * H * W;
            float *cur_out_b = outs_b + (size_t)ni * H * W;

            float fiy = pred_idx_y[i];
            float fiy_floor = floorf(fiy);
            int pred_iy_base = (int)fiy_floor - 1;
            float pred_frac_dy = fiy - fiy_floor;
            float pred_wy[4];
            keys_weights_4(pred_frac_dy, pred_wy);

            warp_row_bicubic(cur_img_a, cur_img_b, cur_out_a, cur_out_b,
                             cur_pred_dy, cur_pred_dx, pred_idx_x, pred_wy,
                             pred_iy_base, i, H, W, nPY, nPX, use_simd);
        }
    } else {
        /* ── Lanczos-3 image warp (LUT-accelerated) ──────────────────────── */
        float (*lanc_lut)[6] = (float (*)[6])malloc(
            (size_t)(LANCZOS3_LUT_SIZE + 1) * 6 * sizeof(float));
        if (!lanc_lut) {
            free(pred_idx_y); free(pred_idx_x);
            return ERROR_NOMEM;
        }
        build_lanczos3_lut(lanc_lut, LANCZOS3_LUT_SIZE);

        int ti;
        #pragma omp parallel for schedule(static)
        for (ti = 0; ti < total_rows; ti++) {
            int ni = ti / H;
            int i  = ti % H;

            const float *cur_pred_dy = pred_dy + ni * pred_stride;
            const float *cur_pred_dx = pred_dx + ni * pred_stride;
            const float *cur_img_a = imgs_a + (size_t)ni * H * W;
            const float *cur_img_b = imgs_b + (size_t)ni * H * W;
            float *cur_out_a = outs_a + (size_t)ni * H * W;
            float *cur_out_b = outs_b + (size_t)ni * H * W;

            float fiy = pred_idx_y[i];
            float fiy_floor = floorf(fiy);
            int pred_iy_base = (int)fiy_floor - 1;
            float pred_frac_dy = fiy - fiy_floor;
            float pred_wy[4];
            keys_weights_4(pred_frac_dy, pred_wy);

            warp_row_lanczos(cur_img_a, cur_img_b, cur_out_a, cur_out_b,
                             cur_pred_dy, cur_pred_dx, pred_idx_x, pred_wy,
                             pred_iy_base, i, H, W, nPY, nPX, lanc_lut, use_simd);
        }

        free(lanc_lut);
    }

    free(pred_idx_y);
    free(pred_idx_x);
    return ERROR_NONE;
}
