#ifndef PIV_FAST_EXP_H
#define PIV_FAST_EXP_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/******************************************************************************
 * piv_expf — portable, vectorizable single-precision exp approximation.
 *
 * WHY THIS EXISTS
 *   The LM Gaussian peak-fit (peak_locate_lm.c) evaluates exp once per patch
 *   pixel per model evaluation, ~1000x per correlation window. A libm expf() is
 *   an opaque function CALL: slow on its own, and — more importantly — it blocks
 *   the compiler from SIMD-vectorizing the surrounding fit loop on every target
 *   that lacks a vector libm (macOS-ARM and Windows; only Linux/glibc ships a
 *   vector expf via libmvec, and only when it is opted into). Replacing the call
 *   with this inline arithmetic turns the loop into the same call-free shape the
 *   stable -O3 (GCC) / O2 (MSVC) auto-vectorizers already pack into SIMD
 *   elsewhere in this codebase (cf. multiply_conjugate in xcorr.c).
 *
 * ALGORITHM (Cephes single-precision expf, public domain)
 *   Range-reduce x = n*ln2 + r with |r| <= ln2/2, approximate exp(r) by a
 *   degree-5 minimax polynomial (Cephes coefficients; peak relative error
 *   ~4.2e-9, far below the 1e-3 px peak-position gate), then reconstruct 2^n by
 *   writing the IEEE-754 exponent field directly: (n + 127) << 23.
 *
 * TWO PORTABILITY-CRITICAL DETAILS
 *   1. The range-reduction rounding uses the magic-number trick
 *      (x*log2e + 1.5*2^23) - 1.5*2^23, which is pure float arithmetic and
 *      vectorizes on NEON, SSE2, and AVX alike. Two forms were rejected for
 *      blocking auto-vectorization on the default build (no -march / no /arch):
 *      floorf() lowers to a libm CALL on baseline SSE2 (no roundps), and the
 *      float->int->float round trip (float)(int)y makes gcc insert a
 *      conversion guard it reports as "unsupported control flow in loop". Only
 *      a single ONE-WAY (int)rn is taken here, for the exponent — that form
 *      vectorizes. The magic-number round is round-to-nearest (|r| <= ln2/2,
 *      same as Cephes' floor(y+0.5)), so the minimax coefficients still apply.
 *   2. The underflow guard is a branchless clamp (maxss), NOT an early return.
 *      The fit's exp argument is always <= 0; a dead/empty window can drive it
 *      to ~ -120, where n < -126 leaves the valid float-exponent range and the
 *      (n+127)<<23 bit-trick yields GARBAGE (a wrong value, not a slow denormal).
 *      Clamping x up to MINLOGF keeps n in range; exp() there is ~1.2e-38, so the
 *      clamp is numerically exact to float. A clamp keeps the loop branch-free;
 *      an early `return` would reintroduce control flow that defeats the vectorizer.
 *
 * ESCAPE HATCH
 *   Build with -DPIV_USE_LIBM_EXP to route PIV_EXP() back to libm expf() from
 *   this same source. That is the A/B reference: it isolates this header's effect
 *   for the accuracy gate (max |delta peak| < 1e-3 px vs the libm build).
 ******************************************************************************/

/* log(2^-126) ~= -87.3365; below this exp underflows toward the smallest normal. */
#define PIV_EXP_MINLOGF (-87.3365478515625f)

static inline float piv_expf(float x)
{
	/* Branchless underflow clamp (see detail 2 above). fmaxf lowers to a single
	   maxss/fmaxs on all three targets; a ?: ternary can emit a branch under MSVC,
	   which would defeat the residual-loop vectoriser on exactly that platform. */
	float xc = fmaxf(x, PIV_EXP_MINLOGF);

	/* Range reduction: r = x - n*ln2, n = round(x/ln2), |r| <= ln2/2.
	   ln2 is split into C1 (high bits) + C2 (low bits) for extra precision. */
	const float LOG2EF = 1.44269504088896341f;  /* 1 / ln2 */
	const float C1 = 0.693359375f;              /* high bits of ln2 */
	const float C2 = -2.12194440e-4f;           /* ln2 - C1 (low bits) */
	const float MAGIC = 12582912.0f;            /* 1.5 * 2^23 : round-to-nearest bias */

	/* n = round(x*LOG2EF) via the magic-number bias (pure float; see detail 1).
	   Adding then subtracting 1.5*2^23 snaps to the nearest integer; the single
	   one-way (int)rn below feeds the exponent reconstruction. */
	float rn = (LOG2EF * xc + MAGIC) - MAGIC;
	int n = (int)rn;

	float r = xc - rn * C1 - rn * C2;

	/* exp(r) via Cephes degree-5 minimax (Horner), assembled as 1 + r + P(r)*r^2. */
	float r2 = r * r;
	float p = 1.9875691500e-4f;
	p = p * r + 1.3981999507e-3f;
	p = p * r + 8.3334519073e-3f;
	p = p * r + 4.1665795894e-2f;
	p = p * r + 1.6666665459e-1f;
	p = p * r + 5.0000001201e-1f;
	float er = p * r2 + r + 1.0f;

	/* Reconstruct 2^n by writing the IEEE-754 exponent field. Valid because the
	   clamp keeps n in [-126, 0] (the fit's arg is always <= 0), so n+127 is in
	   [1, 127] -> a non-negative shift to a normal float, never a denormal or
	   overflow. memcpy type-pun is well-defined in both C and C++ (unlike a union
	   read-other-member, which is UB in C++) and compiles to the same reinterpret. */
	int32_t bits = (n + 127) << 23;
	float scale;
	memcpy(&scale, &bits, sizeof(scale));

	return er * scale;
}

/* Compile-time switch: default = polynomial; -DPIV_USE_LIBM_EXP = libm reference. */
#ifdef PIV_USE_LIBM_EXP
#define PIV_EXP(x) expf(x)
#else
#define PIV_EXP(x) piv_expf(x)
#endif

/* Residual-loop vectorisation hint. The omp simd reduction reassociates the
 * float sum, which is what lets the call-free piv_expf loop vectorize. With libm
 * expf the loop can't vectorize anyway (expf is an opaque call), so the pragma
 * there buys nothing, and we gate it off for the libm build to avoid the extra
 * reduction reorder. NOTE: this does NOT make the libm build byte-identical to
 * the pre-Lever-2 baseline — the loop restructure alone perturbs -O3 FMA codegen
 * and shifts the LM convergence stop by <=1e-3 px (measured 9.7e-4). It is under
 * the 1e-3 gate but is a real change, not bit-identical.
 * The pragma is also gated on _OPENMP: MSVC errors on `omp simd` without
 * /openmp:experimental rather than ignoring it, so a no-OpenMP build must omit it. */
#if defined(PIV_USE_LIBM_EXP) || !defined(_OPENMP)
#define PIV_SIMD_RESIDUAL
#else
#define PIV_SIMD_RESIDUAL _Pragma("omp simd reduction(+:residual_sum)")
#endif

#endif /* PIV_FAST_EXP_H */
