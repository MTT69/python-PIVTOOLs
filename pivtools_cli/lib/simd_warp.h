/*
 * simd_warp.h — Explicit SIMD interior-stencil samplers for the fused image warp.
 *
 * Phase C of fused_warp.c samples each output pixel from a 4x4 (bicubic) or 6x6
 * (Lanczos-3) stencil. When the whole stencil is provably in bounds (the interior
 * fast path, >99.7% of a 2048^2 image), this header replaces the scalar tap loop
 * with vector intrinsics.
 *
 * Strategy — vectorise WITHIN the stencil, not across taps. A per-row horizontal
 * reduction of 4/6 taps needs a slow shuffle; instead we build a column accumulator
 *
 *     c[n] = sum_m  wy[m] * img[(iy+m)*W + ix + n]
 *
 * with one contiguous vector load (vld1q_f32) + one scalar-broadcast FMA
 * (vfmaq_n_f32) per stencil row, then collapse it with a SINGLE horizontal
 * reduction per output pixel:
 *
 *     val = sum_n  wx[n] * c[n]            (vaddvq_f32)
 *
 * This is the right shape for NEON, which has no hardware gather: every load is
 * contiguous, and there is exactly one reduction per pixel rather than 16 (bicubic)
 * or 36 (Lanczos) serial scalar FMAs.
 *
 * NUMERICS: the summation order and multiply grouping differ from the scalar
 * reference (m-outer/n-inner left-associated products), so the result is NOT
 * bit-identical — it is FP-reassociated. The fused_warp impl flag keeps the scalar
 * path (impl=0) as the bit-exact oracle; TestScalarSimdEquivalence asserts the two
 * agree to < 1e-3 on 0-255 data (~1e-6 relative — well inside FP32 reassociation
 * error even on the worst-case high-frequency checkerboard).
 *
 * On non-NEON targets (x86, MSVC) the #else fallback is the scalar interior loop
 * verbatim, so this header compiles everywhere and is behaviourally identical to
 * the Stage-3 interior split there. The x86 across-output-pixels gather strategy is
 * a separate build target and is not implemented here.
 *
 * Preconditions (caller-guaranteed): iy >= 0, iy + (taps-1) < H, ix >= 0,
 * ix + (taps-1) < W. The loads read [ix, ix+taps-1] on rows [iy, iy+taps-1], all
 * in bounds.
 */

#ifndef SIMD_WARP_H
#define SIMD_WARP_H

#include <stddef.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SIMD_WARP_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define SIMD_WARP_AVX2 1
/* Horizontal sum of a 128-bit float vector — the x86 analogue of NEON's single
   vaddvq_f32. SSE3 movehdup + movehl: 3 adds, no full shuffle. */
static inline float simd_warp_hsum128(__m128 v) {
    __m128 sh = _mm_movehdup_ps(v);   /* [v1, v1, v3, v3] */
    __m128 s  = _mm_add_ps(v, sh);    /* [v0+v1, *, v2+v3, *] */
    sh = _mm_movehl_ps(sh, s);        /* [v2+v3, ...] into lane 0 */
    s  = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}
#endif

/* ── Bicubic 4x4 interior sample ───────────────────────────────────────────── */
static inline float bicubic_sample_interior(const float *img, const float wy[4],
                                            const float wx[4], int iy, int ix, int W) {
#if defined(SIMD_WARP_NEON)
    const float *r0 = img + (size_t)(iy + 0) * W + ix;
    const float *r1 = img + (size_t)(iy + 1) * W + ix;
    const float *r2 = img + (size_t)(iy + 2) * W + ix;
    const float *r3 = img + (size_t)(iy + 3) * W + ix;

    /* Column accumulator: c[n] = sum_m wy[m] * img[(iy+m)*W + ix + n], n=0..3. */
    float32x4_t c = vmulq_n_f32(vld1q_f32(r0), wy[0]);
    c = vfmaq_n_f32(c, vld1q_f32(r1), wy[1]);
    c = vfmaq_n_f32(c, vld1q_f32(r2), wy[2]);
    c = vfmaq_n_f32(c, vld1q_f32(r3), wy[3]);

    /* val = dot(wx, c) — one horizontal reduction. */
    return vaddvq_f32(vmulq_f32(c, vld1q_f32(wx)));
#elif defined(SIMD_WARP_AVX2)
    const float *r0 = img + (size_t)(iy + 0) * W + ix;
    const float *r1 = img + (size_t)(iy + 1) * W + ix;
    const float *r2 = img + (size_t)(iy + 2) * W + ix;
    const float *r3 = img + (size_t)(iy + 3) * W + ix;

    /* Column accumulator: c[n] = sum_m wy[m] * img[(iy+m)*W + ix + n], n=0..3.
       One contiguous load + one broadcast-FMA per stencil row (vfmaq_n analogue). */
    __m128 c = _mm_mul_ps(_mm_loadu_ps(r0), _mm_set1_ps(wy[0]));
    c = _mm_fmadd_ps(_mm_loadu_ps(r1), _mm_set1_ps(wy[1]), c);
    c = _mm_fmadd_ps(_mm_loadu_ps(r2), _mm_set1_ps(wy[2]), c);
    c = _mm_fmadd_ps(_mm_loadu_ps(r3), _mm_set1_ps(wy[3]), c);

    /* val = dot(wx, c) — one horizontal reduction. */
    return simd_warp_hsum128(_mm_mul_ps(c, _mm_loadu_ps(wx)));
#else
    float val = 0.0f;
    for (int m = 0; m < 4; m++) {
        const float *r = img + (size_t)(iy + m) * W + ix;
        for (int n = 0; n < 4; n++) {
            val += wy[m] * wx[n] * r[n];
        }
    }
    return val;
#endif
}

/* ── Lanczos-3 6x6 interior sample ─────────────────────────────────────────── */
static inline float lanczos3_sample_interior(const float *img, const float wy[6],
                                             const float wx[6], int iy, int ix, int W) {
#if defined(SIMD_WARP_NEON)
    /* 6 columns = 4 (lo, lanes 0..3) + 2 (hi, lanes 4..5). Accumulate both across
       the 6 stencil rows, mirroring the bicubic column accumulator. */
    float32x4_t c_lo = vdupq_n_f32(0.0f);
    float32x2_t c_hi = vdup_n_f32(0.0f);
    for (int m = 0; m < 6; m++) {
        const float *r = img + (size_t)(iy + m) * W + ix;
        c_lo = vfmaq_n_f32(c_lo, vld1q_f32(r),     wy[m]);   /* cols 0..3 */
        c_hi = vfma_n_f32 (c_hi, vld1_f32 (r + 4), wy[m]);   /* cols 4..5 */
    }
    float32x4_t p_lo = vmulq_f32(c_lo, vld1q_f32(wx));
    float32x2_t p_hi = vmul_f32 (c_hi, vld1_f32 (wx + 4));
    return vaddvq_f32(p_lo) + vaddv_f32(p_hi);
#elif defined(SIMD_WARP_AVX2)
    /* 6 columns = 4 (lanes 0..3, full load) + 2 (lanes 0..1, 8-byte load). The hi
       pair uses an 8-byte _mm_load_sd so cols 6,7 are never touched (only ix+5<W is
       guaranteed). Accumulate both across the 6 stencil rows, mirroring bicubic. */
    __m128 c_lo = _mm_setzero_ps();
    __m128 c_hi = _mm_setzero_ps();   /* only lanes 0,1 carry data; lanes 2,3 stay 0 */
    for (int m = 0; m < 6; m++) {
        const float *r = img + (size_t)(iy + m) * W + ix;
        __m128 wym = _mm_set1_ps(wy[m]);
        c_lo = _mm_fmadd_ps(_mm_loadu_ps(r), wym, c_lo);                  /* cols 0..3 */
        __m128 hi = _mm_castpd_ps(_mm_load_sd((const double *)(r + 4)));  /* cols 4,5  */
        c_hi = _mm_fmadd_ps(hi, wym, c_hi);
    }
    __m128 p_lo = _mm_mul_ps(c_lo, _mm_loadu_ps(wx));
    __m128 p_hi = _mm_mul_ps(c_hi, _mm_castpd_ps(_mm_load_sd((const double *)(wx + 4))));
    /* Fold the two partials lane-wise (cols 4,5 land on lanes 0,1; lanes 2,3 of p_hi
       are 0), then one reduction over all 6 taps. */
    return simd_warp_hsum128(_mm_add_ps(p_lo, p_hi));
#else
    float val = 0.0f;
    for (int m = 0; m < 6; m++) {
        const float *r = img + (size_t)(iy + m) * W + ix;
        for (int n = 0; n < 6; n++) {
            val += wy[m] * wx[n] * r[n];
        }
    }
    return val;
#endif
}

#endif /* SIMD_WARP_H */
