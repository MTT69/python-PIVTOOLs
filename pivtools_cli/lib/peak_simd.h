#ifndef PEAK_SIMD_H
#define PEAK_SIMD_H

/******************************************************************************
 * peak_simd.h — vector layer for the batched (one-window-per-lane) LM peak
 * fitter. Sibling of codelet_simd.h, NOT an extension of it: the FFT needs
 * only add/sub/mul/broadcast, while lockstep LM needs division, sqrt,
 * compares, masked selects, double accumulation vectors, int converts and a
 * vector exp — a disjoint op set that would only bloat the FFT header.
 *
 * ONE SOURCE, FOUR ROUTES — all GCC/Clang `vector_size` ("vecext"), keyed off
 * the same PIVTOOLS_FFT_ISA_* macro as codelet_simd.h so the lane count
 * always equals the FFT engine's PIV_VLANES (the integration hands the fitter
 * packC[LANES][numel] batches):
 *
 *   PIVTOOLS_FFT_ISA_NEON4            -> 4 lanes  (macOS arm64)
 *   PIVTOOLS_FFT_ISA_VEXT8            -> 8 lanes  (generic Linux x86)
 *   PIVTOOLS_FFT_ISA_AVX512           -> 16 lanes (HPC / Iridis)
 *   PIVTOOLS_FFT_ISA_AVX2 + __clang__ -> 8 lanes  (Windows clang-cl)
 *   PIVTOOLS_FFT_ISA_AVX2 + MSVC cl   -> STUB (PK_BATCH_AVAILABLE 0)
 *
 * The Windows route deserves a note: the codelet FFT uses an __m256 intrinsic
 * render there because the *generated* header must also compile under plain
 * MSVC cl (the PIVTOOLS_WIN_COMPILER=cl escape hatch). This header is
 * hand-written for clang-cl (the enforced default compiler), which fully
 * supports vector_size — so Windows shares the one vecext source. Under real
 * MSVC cl the batch fitter compiles as an unavailable stub and selecting it
 * errors loudly (project rule: no silent fallbacks).
 *
 * PORTABILITY RULES (intersection of GCC>=9 and clang>=12, C mode):
 *   - vector_size types, lane subscript v[i], operators + - * / on
 *     float/double vectors, + << & | ~ ^ on int vectors, scalar broadcasts.
 *   - Vector comparisons (a < b) yield same-width signed-int vectors of
 *     0 / -1 per lane: pk_vf compare -> pk_vi mask, pk_vd compare -> pk_vl.
 *   - Select is ALWAYS the bitwise idiom (m & a) | (~m & b) — vector ?: in C
 *     is too new to rely on. Exact for floats (bit-level copy).
 *   - NO __builtin_elementwise_* (clang>=14 only, absent in GCC): sqrt is a
 *     per-lane loop that auto-vectorizes to vsqrtps/vsqrtpd — the TU MUST be
 *     compiled with -fno-math-errno (clang: /clang:-fno-math-errno) or the
 *     errno guard keeps it scalar (probe-verified on clang-cl).
 *   - __builtin_convertvector for value converts and mask width bridges.
 *   - memcpy type-puns for bitcasts (well-defined; same as fast_exp.h).
 ******************************************************************************/

#include <math.h>
#include <string.h>

#if defined(_MSC_VER) && !defined(__clang__)
/* Plain MSVC cl (PIVTOOLS_WIN_COMPILER=cl escape hatch): no vector_size.
 * The batch fitter is unavailable; peak_locate_lm_batch.c compiles a stub. */
#define PK_BATCH_AVAILABLE 0

#else

#if defined(PIVTOOLS_FFT_ISA_NEON4)
#define PK_LANES 4
#elif defined(PIVTOOLS_FFT_ISA_VEXT8) || defined(PIVTOOLS_FFT_ISA_AVX2)
#define PK_LANES 8
#elif defined(PIVTOOLS_FFT_ISA_AVX512)
#define PK_LANES 16
#else
#error "peak_simd.h: define one of PIVTOOLS_FFT_ISA_{NEON4,VEXT8,AVX2,AVX512} (setup.py passes the same macro as the FFT TUs)"
#endif

#define PK_BATCH_AVAILABLE 1

/* Vector types: W float/int32 lanes and W double/int64 lanes. The 64-bit
 * vectors are 2x the byte width and legalize to register pairs on AVX2 —
 * acceptable: they only appear on the cold (accepted-step) path. */
typedef float    pk_vf __attribute__((vector_size(PK_LANES * 4)));
typedef int      pk_vi __attribute__((vector_size(PK_LANES * 4)));
typedef double   pk_vd __attribute__((vector_size(PK_LANES * 8)));
typedef long long pk_vl __attribute__((vector_size(PK_LANES * 8)));

/* ── broadcasts ──────────────────────────────────────────────────────────── */
static inline pk_vf pk_vf_set1(float x)  { pk_vf r; for (int l = 0; l < PK_LANES; ++l) r[l] = x; return r; }
static inline pk_vi pk_vi_set1(int x)    { pk_vi r; for (int l = 0; l < PK_LANES; ++l) r[l] = x; return r; }
static inline pk_vd pk_vd_set1(double x) { pk_vd r; for (int l = 0; l < PK_LANES; ++l) r[l] = x; return r; }
#define pk_vf_zero() pk_vf_set1(0.0f)
#define pk_vd_zero() pk_vd_set1(0.0)
#define pk_vi_zero() pk_vi_set1(0)

/* Arithmetic uses raw operators (+ - * /) on pk_vf/pk_vd directly, and
 * (+ << & | ~ ^) on pk_vi/pk_vl; comparisons use raw operators and yield
 * 0/-1 masks (pk_vi from float compares, pk_vl from double compares).
 * These are part of this header's documented contract. */

/* ── bitcasts (memcpy puns) ──────────────────────────────────────────────── */
static inline pk_vi pk_castf2i(pk_vf x) { pk_vi r; memcpy(&r, &x, sizeof r); return r; }
static inline pk_vf pk_casti2f(pk_vi x) { pk_vf r; memcpy(&r, &x, sizeof r); return r; }
static inline pk_vl pk_castd2l(pk_vd x) { pk_vl r; memcpy(&r, &x, sizeof r); return r; }
static inline pk_vd pk_castl2d(pk_vl x) { pk_vd r; memcpy(&r, &x, sizeof r); return r; }

/* ── selects: the ONLY select idiom (bit-exact, branch-free) ─────────────── */
static inline pk_vf pk_self(pk_vi m, pk_vf a, pk_vf b) {         /* m ? a : b */
	return pk_casti2f((m & pk_castf2i(a)) | (~m & pk_castf2i(b)));
}
static inline pk_vi pk_seli(pk_vi m, pk_vi a, pk_vi b) {
	return (m & a) | (~m & b);
}
static inline pk_vd pk_seld(pk_vl m, pk_vd a, pk_vd b) {
	return pk_castl2d((m & pk_castd2l(a)) | (~m & pk_castd2l(b)));
}

/* ── min/max (compare + select; NaN-safe like fminf/fmaxf is NOT promised —
 *    matches the scalar fitter's clamp usage where args are finite) ───────── */
static inline pk_vf pk_minf(pk_vf a, pk_vf b) { return pk_self(a < b, a, b); }
static inline pk_vf pk_maxf(pk_vf a, pk_vf b) { return pk_self(a > b, a, b); }

/* ── sqrt: per-lane loops; vectorize to vsqrtps/vsqrtpd under
 *    -fno-math-errno (probe-verified) ─────────────────────────────────────── */
static inline pk_vf pk_sqrtf(pk_vf x) { pk_vf r; for (int l = 0; l < PK_LANES; ++l) r[l] = sqrtf(x[l]); return r; }
static inline pk_vd pk_sqrtd(pk_vd x) { pk_vd r; for (int l = 0; l < PK_LANES; ++l) r[l] = sqrt(x[l]);  return r; }

/* ── converts ────────────────────────────────────────────────────────────── */
#define pk_cvt_f2i(x) __builtin_convertvector((x), pk_vi)   /* truncating   */
#define pk_cvt_i2f(x) __builtin_convertvector((x), pk_vf)
#define pk_cvt_f2d(x) __builtin_convertvector((x), pk_vd)
#define pk_cvt_d2f(x) __builtin_convertvector((x), pk_vf)
/* mask width bridges: int32 0/-1 <-> int64 0/-1 (sign-extending convert) */
#define pk_mask_i2l(m) __builtin_convertvector((m), pk_vl)
#define pk_mask_l2i(m) __builtin_convertvector((m), pk_vi)

/* ── horizontal mask tests (scalar loops; cheap, once per iteration) ─────── */
static inline int pk_any(pk_vi m) { int r = 0; for (int l = 0; l < PK_LANES; ++l) r |= m[l]; return r != 0; }
static inline int pk_all(pk_vi m) { int r = -1; for (int l = 0; l < PK_LANES; ++l) r &= m[l]; return r != 0; }

/******************************************************************************
 * pk_vexpf — W-lane single-precision exp, arguments <= 0 only.
 *
 * Vectorization of the Cephes degree-5 piv_expf that was A/B-tested for the
 * SCALAR fitter and rejected (slower than libm — no lane parallelism to
 * amortize the polynomial; see peak_locate_lm.c). In W-lane lockstep the
 * economics flip: one polynomial evaluates W windows' pixels at once, with no
 * call and no vector-libm dependency on any platform.
 *
 * Algorithm and constants are verbatim from the archived fast_exp.h (commit
 * 0f92b72): magic-number round-to-nearest range reduction (pure float ops),
 * degree-5 minimax for exp(r) (peak rel err ~4.2e-9), 2^n reconstructed by
 * writing the IEEE-754 exponent field. The branchless underflow clamp keeps
 * n in [-126, 0] — REQUIRES arg <= 0, which the fitter guarantees (4/5-DOF
 * args are -(di^2+dj^2)/s^2; the 6-DOF |sxy| clamp keeps its quadratic form
 * positive-definite).
 *
 * ESCAPE HATCH: -DPIV_BATCH_LIBM_EXP routes through per-lane libm expf — the
 * structure-validation reference that isolates exp error from lockstep-logic
 * error in the gates.
 ******************************************************************************/
#define PK_EXP_MINLOGF (-87.3365478515625f)

#ifdef PIV_BATCH_LIBM_EXP
static inline pk_vf pk_vexpf(pk_vf x)
{
	pk_vf r;
	for (int l = 0; l < PK_LANES; ++l) r[l] = expf(x[l]);
	return r;
}
#else
static inline pk_vf pk_vexpf(pk_vf x)
{
	const pk_vf LOG2EF = pk_vf_set1(1.44269504088896341f);  /* 1 / ln2        */
	const pk_vf C1     = pk_vf_set1(0.693359375f);          /* ln2 high bits  */
	const pk_vf C2     = pk_vf_set1(-2.12194440e-4f);       /* ln2 - C1       */
	const pk_vf MAGIC  = pk_vf_set1(12582912.0f);           /* 1.5 * 2^23     */

	pk_vf xc = pk_maxf(x, pk_vf_set1(PK_EXP_MINLOGF));      /* branchless underflow clamp */

	pk_vf rn = (LOG2EF * xc + MAGIC) - MAGIC;               /* round-to-nearest n as float */
	pk_vi n  = pk_cvt_f2i(rn);                              /* one-way convert for exponent */

	pk_vf r  = xc - rn * C1 - rn * C2;

	pk_vf r2 = r * r;
	pk_vf p  = pk_vf_set1(1.9875691500e-4f);
	p = p * r + pk_vf_set1(1.3981999507e-3f);
	p = p * r + pk_vf_set1(8.3334519073e-3f);
	p = p * r + pk_vf_set1(4.1665795894e-2f);
	p = p * r + pk_vf_set1(1.6666665459e-1f);
	p = p * r + pk_vf_set1(5.0000001201e-1f);
	pk_vf er = p * r2 + r + pk_vf_set1(1.0f);

	pk_vi bits = (n + pk_vi_set1(127)) << 23;               /* 2^n via exponent field */
	return er * pk_casti2f(bits);
}
#endif /* PIV_BATCH_LIBM_EXP */

#endif /* vector_size available */

#endif /* PEAK_SIMD_H */
