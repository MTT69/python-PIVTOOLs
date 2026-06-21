/****************************************************************************
 * codelet_simd.h -- cross-compiler SIMD-lane layer for the batched codelet
 * FFT (Stage B). One PIV window per SIMD lane, zero cross-lane shuffles, so
 * every lane runs the identical scalar computation independently.
 *
 * Exactly ONE width is selected at compile time by a -D macro from setup.py:
 *
 *   -DPIVTOOLS_FFT_ISA_NEON4   v4  / 4 lanes  -- macOS arm64 (one NEON register)
 *   -DPIVTOOLS_FFT_ISA_VEXT8   v8  / 8 lanes  -- generic x86_64 (AVX2 via vector_size)
 *   -DPIVTOOLS_FFT_ISA_AVX512  v16 / 16 lanes -- Linux/HPC AVX-512 (one zmm)
 *   -DPIVTOOLS_FFT_ISA_AVX2    __m256 / 8     -- Windows MSVC (intrinsics; no vector_size)
 *
 * The GCC/Clang `vector_size` typedefs (v4/v8/v16) and the lane-typed codelet
 * functions (rfftN_v4, cfftN_v8, ...) are emitted into codelets_gen.h by
 * gen_codelet.py for the matching --isa render. MSVC cannot use vector_size,
 * so on Windows we take the `avx2` intrinsic render (__m256, rfftN_v8) and
 * supply the helpers with _mm256_* intrinsics.
 *
 * The batched engine (codelet_fft.c) is written ONLY against the PIV_V*
 * helpers and the codelet-name macros below, never raw operators, so a single
 * source serves all four routes.
 ****************************************************************************/
#ifndef CODELET_SIMD_H
#define CODELET_SIMD_H

#include <stdlib.h>         /* aligned_alloc / free */
#include "codelets_gen.h"   /* v4/v8/v16 typedefs + lane-typed codelets */

/* ------------------------------------------------------------------ *
 * Portable alignment attribute (used on stack vec scratch arrays).
 * ------------------------------------------------------------------ */
#if defined(_MSC_VER)
#  define PIV_ALIGN(n) __declspec(align(n))
#else
#  define PIV_ALIGN(n) __attribute__((aligned(n)))
#endif

/* Token-paste helper: build `rfft<NN><SUF>` with NN and SUF macro-expanded. */
#define PIV_CAT_(a, b) a##b
#define PIV_CAT(a, b)  PIV_CAT_(a, b)

/* ================================================================== *
 *  Windows / MSVC -- AVX2 intrinsics, 8 lanes (the `avx2` render).
 * ================================================================== */
#if defined(PIVTOOLS_FFT_ISA_AVX2)

#include <immintrin.h>

typedef __m256 PIV_VEC;
#define PIV_VLANES 8
#define PIV_CL_SUF _v8
/* MSVC __declspec(align(n)) requires an integer *literal*, not an expression
 * like (PIV_VLANES*4). One AVX2 vector is 8 floats = 32 bytes. */
#define PIV_VEC_ALIGN 32

static inline PIV_VEC piv_vset1(float x)             { return _mm256_set1_ps(x); }
static inline PIV_VEC piv_vzero(void)                { return _mm256_setzero_ps(); }
static inline PIV_VEC piv_vadd(PIV_VEC a, PIV_VEC b) { return _mm256_add_ps(a, b); }
static inline PIV_VEC piv_vsub(PIV_VEC a, PIV_VEC b) { return _mm256_sub_ps(a, b); }
static inline PIV_VEC piv_vmul(PIV_VEC a, PIV_VEC b) { return _mm256_mul_ps(a, b); }

/* lane j <- base[j*n + p]  (window-major gather; lane0 = base[0*n+p]) */
static inline PIV_VEC piv_vgather(const float *base, int p, int n) {
    return _mm256_set_ps(base[7*n+p], base[6*n+p], base[5*n+p], base[4*n+p],
                         base[3*n+p], base[2*n+p], base[1*n+p], base[0*n+p]);
}
/* base[j*n + p] <- lane j  (inverse of gather) */
static inline void piv_vscatter(float *base, int p, int n, PIV_VEC v) {
    PIV_ALIGN(32) float t[8];
    _mm256_store_ps(t, v);
    for (int j = 0; j < 8; ++j) base[j*n + p] = t[j];
}

#define PIV_VEC_ALLOC(n) ((PIV_VEC *)_mm_malloc((size_t)(n) * sizeof(PIV_VEC), 32))
#define PIV_VEC_FREE(p)  _mm_free(p)

/* ================================================================== *
 *  GCC / Clang -- vector_size typedefs, one generic helper set for
 *  widths 4 / 8 / 16. Operators broadcast/lower correctly on NEON & AVX.
 * ================================================================== */
#else

#if defined(PIVTOOLS_FFT_ISA_NEON4)
typedef v4  PIV_VEC;
#define PIV_VLANES 4
#define PIV_CL_SUF _v4
#elif defined(PIVTOOLS_FFT_ISA_VEXT8)
typedef v8  PIV_VEC;
#define PIV_VLANES 8
#define PIV_CL_SUF _v8
#elif defined(PIVTOOLS_FFT_ISA_AVX512)
typedef v16 PIV_VEC;
#define PIV_VLANES 16
#define PIV_CL_SUF _v16
#else
#error "codelet_simd.h: define one of PIVTOOLS_FFT_ISA_{NEON4,VEXT8,AVX512,AVX2}"
#endif

static inline PIV_VEC piv_vset1(float x) {
    PIV_VEC r;
    for (int i = 0; i < PIV_VLANES; ++i) r[i] = x;
    return r;
}
static inline PIV_VEC piv_vzero(void)                { return piv_vset1(0.0f); }
static inline PIV_VEC piv_vadd(PIV_VEC a, PIV_VEC b) { return a + b; }
static inline PIV_VEC piv_vsub(PIV_VEC a, PIV_VEC b) { return a - b; }
static inline PIV_VEC piv_vmul(PIV_VEC a, PIV_VEC b) { return a * b; }

/* lane j <- base[j*n + p]  (window-major gather; lane0 = base[0*n+p]) */
static inline PIV_VEC piv_vgather(const float *base, int p, int n) {
    PIV_VEC r;
    for (int j = 0; j < PIV_VLANES; ++j) r[j] = base[j*n + p];
    return r;
}
/* base[j*n + p] <- lane j  (inverse of gather) */
static inline void piv_vscatter(float *base, int p, int n, PIV_VEC v) {
    for (int j = 0; j < PIV_VLANES; ++j) base[j*n + p] = v[j];
}

static inline PIV_VEC *piv_vec_alloc_(size_t n) {
    void *p = (void *)0;
    if (posix_memalign(&p, sizeof(PIV_VEC), n * sizeof(PIV_VEC))) p = (void *)0;
    return (PIV_VEC *)p;
}
#define PIV_VEC_ALLOC(n) piv_vec_alloc_((size_t)(n))
#define PIV_VEC_FREE(p)  free(p)

#endif /* ISA dispatch */

/* Byte alignment of one vector (16/32/64), for stack scratch tagging.
 * (The MSVC/AVX2 branch sets this to the literal 32 above.) */
#ifndef PIV_VEC_ALIGN
#define PIV_VEC_ALIGN (PIV_VLANES * 4)
#endif

/* Lane-typed codelet function names for the selected render, e.g.
 * PIV_RFFT(16) -> rfft16_v4 on NEON, rfft16_v8 on AVX2. */
#define PIV_RFFT(NN)  PIV_CAT(PIV_CAT(rfft,  NN), PIV_CL_SUF)
#define PIV_IRFFT(NN) PIV_CAT(PIV_CAT(irfft, NN), PIV_CL_SUF)
#define PIV_CFFT(NN)  PIV_CAT(PIV_CAT(cfft,  NN), PIV_CL_SUF)
#define PIV_ICFFT(NN) PIV_CAT(PIV_CAT(icfft, NN), PIV_CL_SUF)

#endif /* CODELET_SIMD_H */
