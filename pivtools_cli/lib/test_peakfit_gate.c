/******************************************************************************
 * test_peakfit_gate.c — standalone correctness gate for the batched
 * (one-window-per-lane) LM peak fitter. Pattern: test_codelet_gate.c.
 *
 * NOT part of the DLL link (setup.py compiles only its explicit TU list).
 * Build ad hoc, e.g. on Windows (x64 Native Tools prompt / after vcvars64):
 *
 *   clang-cl /O2 /arch:AVX2 /clang:-fno-math-errno /DPIVTOOLS_FFT_ISA_AVX2 ^
 *       peak_locate_lm.c peak_locate_lm_batch.c test_peakfit_gate.c ^
 *       /Fe:pkgate.exe && pkgate
 *
 * mac (CP5):    clang -O3 -fno-math-errno -mcpu=native -DPIVTOOLS_FFT_ISA_NEON4 \
 *                   peak_locate_lm.c peak_locate_lm_batch.c test_peakfit_gate.c -lm -o pkgate && ./pkgate
 * Iridis (CP5): clang -O3 -fno-math-errno -march=native -DPIVTOOLS_FFT_ISA_AVX512 \
 *                   peak_locate_lm.c peak_locate_lm_batch.c test_peakfit_gate.c -lm -o pkgate && ./pkgate
 *
 * Two flavors: add -DPIV_BATCH_LIBM_EXP for the libm-exp reference build
 * (tight structure-exactness thresholds); default = production poly exp.
 * Expected output ends with "GATE PASS".
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "peak_simd.h"
#include "peak_locate_lm.h"        /* scalar oracle */
#include "peak_locate_lm_batch.h"  /* device under test */

#if !PK_BATCH_AVAILABLE
int main(void)
{
	printf("peak_simd.h: batch fitter unavailable on this compiler (plain MSVC cl)\n"
	       "GATE SKIP\n");
	return 0;
}
#else

static int g_fail = 0;

#define CHECK(cond, ...) do { \
	if (!(cond)) { g_fail = 1; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
} while (0)

/* ── gate 0: pk_vexpf accuracy vs libm expf over the fitter's domain ─────── */
static void gate_vexpf_accuracy(void)
{
	/* Dense sweep of [-87.34, 0]: >=1e7 points hits every exponent bucket.
	 * P0 go: max relative error <= 1e-6 (Cephes deg-5 is ~4.2e-9; the bound
	 * has three orders of margin for range-reduction rounding). */
	const long NPTS = 20000000L;
	const float lo = PK_EXP_MINLOGF, hi = 0.0f;
	double max_rel = 0.0;
	float  worst_x = 0.0f;

	for (long i = 0; i < NPTS; i += PK_LANES) {
		pk_vf x;
		for (int l = 0; l < PK_LANES; ++l) {
			long k = i + l;
			x[l] = lo + (hi - lo) * ((float)k / (float)(NPTS - 1));
		}
		pk_vf y = pk_vexpf(x);
		for (int l = 0; l < PK_LANES; ++l) {
			double ref = exp((double)x[l]);   /* double reference */
			double rel = fabs((double)y[l] - ref) / ref;
			if (rel > max_rel) { max_rel = rel; worst_x = x[l]; }
		}
	}

	printf("gate 0  pk_vexpf accuracy: max rel err %.3e at x=%.6f  (bound 1e-6)\n",
	       max_rel, worst_x);
	CHECK(max_rel <= 1e-6, "pk_vexpf max rel err %.3e exceeds 1e-6", max_rel);
}

/* ── gate 0b: pk_vexpf underflow-clamp region stays finite and tiny ──────── */
static void gate_vexpf_clamp(void)
{
	pk_vf x;
	for (int l = 0; l < PK_LANES; ++l) x[l] = -120.0f - (float)l;  /* below MINLOGF */
	pk_vf y = pk_vexpf(x);
	for (int l = 0; l < PK_LANES; ++l) {
		CHECK(isfinite(y[l]) && y[l] >= 0.0f && y[l] < 1e-37f,
		      "clamp region lane %d: got %g (want tiny finite >= 0)", l, (double)y[l]);
	}
	printf("gate 0b pk_vexpf underflow clamp: ok\n");
}

/* ── gate 0c: select/compare/mask plumbing sanity ────────────────────────── */
static void gate_mask_plumbing(void)
{
	pk_vf a = pk_vf_set1(1.0f), b = pk_vf_set1(2.0f);
	pk_vi m = a < b;                       /* all lanes true -> -1 */
	CHECK(pk_all(m), "float compare mask not all-true");
	pk_vf s = pk_self(m, a, b);
	for (int l = 0; l < PK_LANES; ++l) CHECK(s[l] == 1.0f, "pk_self picked wrong side");
	m = a > b;
	CHECK(!pk_any(m), "float compare mask not all-false");

	pk_vd da = pk_vd_set1(4.0), db = pk_vd_set1(9.0);
	pk_vl dm = da <= db;
	pk_vd ds = pk_seld(dm, pk_sqrtd(da), db);
	for (int l = 0; l < PK_LANES; ++l) CHECK(ds[l] == 2.0, "pk_seld/pk_sqrtd wrong");

	pk_vi mi = pk_mask_l2i(dm);
	CHECK(pk_all(mi), "mask l2i bridge lost lanes");
	printf("gate 0c mask/select/sqrt plumbing: ok (lanes=%d)\n", PK_LANES);
}

/*═════════════════════════════════════════════════════════════════════════
 * P1 gates: batch fitter vs scalar oracle
 *═════════════════════════════════════════════════════════════════════════*/

#define PLANE_N 33                        /* test plane size (odd, > 8*PKSIZE) */
#define PLANE_NUMEL (PLANE_N * PLANE_N)

/* thresholds: tight for the libm-exp reference flavor, production for poly */
#ifdef PIV_BATCH_LIBM_EXP
#define A1_MAX_DLOC 1e-4
#define A1_MED_DLOC 1e-6
#else
#define A1_MAX_DLOC 1e-3
#define A1_MED_DLOC 1e-5
#endif

/* deterministic LCG so batteries are reproducible cross-platform */
static unsigned int g_rng = 0x12345678u;
static float frand(void) {                 /* uniform [0, 1) */
	g_rng = g_rng * 1664525u + 1013904223u;
	return (float)(g_rng >> 8) / 16777216.0f;
}

/* synthetic plane generators (match unit-tests/test_instantaneous_peaks.py) */
static void gen_gauss4(float *p, float amp, float di, float dj, float s)
{
	int c = (PLANE_N - 1) / 2;
	for (int i = 0; i < PLANE_N; ++i)
		for (int j = 0; j < PLANE_N; ++j) {
			float y = (float)(i - c) - di, x = (float)(j - c) - dj;
			p[i * PLANE_N + j] = amp * expf(-(y*y + x*x) / (s*s));
		}
}
static void gen_gauss5(float *p, float amp, float di, float dj, float sr, float sc)
{
	int c = (PLANE_N - 1) / 2;
	for (int i = 0; i < PLANE_N; ++i)
		for (int j = 0; j < PLANE_N; ++j) {
			float y = (float)(i - c) - di, x = (float)(j - c) - dj;
			p[i * PLANE_N + j] = amp * expf(-(y*y/(sr*sr) + x*x/(sc*sc)));
		}
}
static void gen_gauss6(float *p, float amp, float di, float dj,
                       float vr, float vc, float cov)
{
	int c = (PLANE_N - 1) / 2;
	for (int i = 0; i < PLANE_N; ++i)
		for (int j = 0; j < PLANE_N; ++j) {
			float y = (float)(i - c) - di, x = (float)(j - c) - dj;
			p[i * PLANE_N + j] = amp * expf(-0.5f * (y*y/vr + x*x/vc + 2.0f*y*x*cov));
		}
}

/* run the scalar oracle on one plane */
static void run_scalar(const float *plane, int fit_type, float out_loc[3], float out_std[3])
{
	int N[2] = { PLANE_N, PLANE_N };
	lsqpeaklocate_lm((float *)plane, N, out_loc, 1, fit_type, out_std);
}

/* run the batch fitter on W planes */
static void run_batch(const float *planes, int L_real, int fit_type,
                      float *loc /*[3][W]*/, float *std /*[3][W]*/)
{
	int N[2] = { PLANE_N, PLANE_N };
	lsqpeaklocate_lm_batch(planes, L_real, N, fit_type, loc, std);
}

/* fill one battery plane for the given case index (mix of clean shapes) */
static void gen_case(float *plane, int fit_type, int idx)
{
	float di = -0.45f + 0.9f * frand();
	float dj = -0.45f + 0.9f * frand();
	float amp = 100.0f + 900.0f * frand();
	float s1 = 1.0f + 1.5f * frand();
	float s2 = 1.0f + 1.5f * frand();
	(void)idx;
	if (fit_type == 4)      gen_gauss4(plane, amp, di, dj, s1);
	else if (fit_type == 5) gen_gauss5(plane, amp, di, dj, s1, s2);
	else                    gen_gauss6(plane, amp, di, dj, s1*s1, s2*s2,
	                                   (frand() - 0.5f) * 0.4f / (s1 * s2));
}

/* ── gate A1: clean battery, batch vs scalar oracle ──────────────────────── */
static int cmp_double(const void *a, const void *b) {
	double d = *(const double *)a - *(const double *)b;
	return (d > 0) - (d < 0);
}

static void gate_a1_battery(int fit_type)
{
	enum { NCASE = 512 };
	static float planes[PK_LANES][PLANE_NUMEL];
	static double dlocs[NCASE];
	int ndloc = 0, n_nan_mismatch = 0;

	g_rng = 0xC0FFEE00u + (unsigned)fit_type;   /* per-type reproducibility */

	for (int base = 0; base < NCASE; base += PK_LANES) {
		int nb = (NCASE - base) < PK_LANES ? (NCASE - base) : PK_LANES;
		for (int l = 0; l < nb; ++l) gen_case(planes[l], fit_type, base + l);

		float bloc[3 * PK_LANES], bstd[3 * PK_LANES];
		run_batch(&planes[0][0], nb, fit_type, bloc, bstd);

		for (int l = 0; l < nb; ++l) {
			float sloc[3], sstd[3];
			run_scalar(planes[l], fit_type, sloc, sstd);

			int s_nan = isnan(sloc[0]), b_nan = isnan(bloc[0 * PK_LANES + l]);
			if (s_nan != b_nan) { n_nan_mismatch++; continue; }
			if (s_nan) continue;

			double dr = fabs((double)sloc[0] - (double)bloc[0 * PK_LANES + l]);
			double dc = fabs((double)sloc[1] - (double)bloc[1 * PK_LANES + l]);
			double d = sqrt(dr * dr + dc * dc);
			if (ndloc < NCASE) dlocs[ndloc++] = d;
		}
	}

	qsort(dlocs, (size_t)ndloc, sizeof(double), cmp_double);
	double dmax = ndloc ? dlocs[ndloc - 1] : 0.0;
	double dmed = ndloc ? dlocs[ndloc / 2] : 0.0;

	printf("gate A1 type %d: n=%d  max |dloc| %.3e px  median %.3e px  nan-mismatch %d\n",
	       fit_type, ndloc, dmax, dmed, n_nan_mismatch);
	CHECK(n_nan_mismatch == 0, "A1 type %d: %d NaN mismatches on clean data", fit_type, n_nan_mismatch);
	CHECK(dmax < A1_MAX_DLOC, "A1 type %d: max dloc %.3e >= %g", fit_type, dmax, (double)A1_MAX_DLOC);
	CHECK(dmed < A1_MED_DLOC, "A1 type %d: median dloc %.3e >= %g", fit_type, dmed, (double)A1_MED_DLOC);
}

/* ── gate A3: NaN failure masks identical on pathological planes ─────────── */
static void gate_a3_nan_masks(int fit_type)
{
	enum { NPATH = 4 * PK_LANES };
	static float planes[NPATH][PLANE_NUMEL];
	int c = (PLANE_N - 1) / 2;

	g_rng = 0xDEAD0000u + (unsigned)fit_type;
	for (int k = 0; k < NPATH; ++k) {
		switch (k % 4) {
		case 0:  /* flat zero -> search-gate NaN */
			memset(planes[k], 0, sizeof(planes[k]));
			break;
		case 1:  /* uniform noise + spike -> LM-failure NaN */
			for (int t = 0; t < PLANE_NUMEL; ++t) planes[k][t] = frand();
			planes[k][c * PLANE_N + c] = 2.0f;
			break;
		case 2:  /* NaN-poisoned subwindow -> LM-failure NaN */
			gen_gauss4(planes[k], 1000.0f, 0.0f, 0.0f, 1.5f);
			planes[k][(c + 1) * PLANE_N + (c + 1)] = NAN;
			break;
		default: /* clean Gaussian -> finite (mask must agree too) */
			gen_gauss4(planes[k], 1000.0f, 0.2f, -0.3f, 1.5f);
			break;
		}
	}

	int mismatches = 0;
	for (int base = 0; base < NPATH; base += PK_LANES) {
		float bloc[3 * PK_LANES], bstd[3 * PK_LANES];
		run_batch(&planes[base][0], PK_LANES, fit_type, bloc, bstd);
		for (int l = 0; l < PK_LANES; ++l) {
			float sloc[3], sstd[3];
			run_scalar(planes[base + l], fit_type, sloc, sstd);
			if (isnan(sloc[0]) != isnan(bloc[0 * PK_LANES + l])) {
				mismatches++;
				printf("  A3 mismatch: case %d scalar %s batch %s\n", base + l,
				       isnan(sloc[0]) ? "NaN" : "finite",
				       isnan(bloc[0 * PK_LANES + l]) ? "NaN" : "finite");
			}
		}
	}
	printf("gate A3 type %d: NaN masks %s (%d planes)\n",
	       fit_type, mismatches ? "MISMATCH" : "identical", NPATH);
	CHECK(mismatches == 0, "A3 type %d: %d NaN-mask mismatches", fit_type, mismatches);
}

/* ── gate: lane independence (permutation + dead-lane freeze in one) ─────── */
/* Every lane's result must be independent of what the other lanes hold:
 * fit W different planes together, then each one alone in lane 0 with the
 * rest of the batch full of hard/noisy planes — results must be BIT-equal.
 * This kills both mask-blend aliasing (a converged lane perturbed by later
 * iterations) and any cross-lane leakage. */
static void gate_lane_independence(int fit_type)
{
	static float planes[PK_LANES][PLANE_NUMEL];
	static float noise[PK_LANES][PLANE_NUMEL];
	int c = (PLANE_N - 1) / 2;

	g_rng = 0xFEED0000u + (unsigned)fit_type;
	/* mixed batch: easy (converges fast), hard (noisy, many iters), dead  */
	for (int l = 0; l < PK_LANES; ++l) {
		gen_case(planes[l], fit_type, l);
		if (l % 3 == 1)   /* add noise -> divergent iteration counts */
			for (int t = 0; t < PLANE_NUMEL; ++t)
				planes[l][t] += 20.0f * (frand() - 0.5f);
		if (l % 5 == 4)   /* dead lane: flat plane -> search-gate NaN */
			memset(planes[l], 0, sizeof(planes[l]));
	}
	for (int l = 0; l < PK_LANES; ++l) {   /* filler batch of noisy planes */
		for (int t = 0; t < PLANE_NUMEL; ++t) noise[l][t] = frand();
		noise[l][c * PLANE_N + c] = 2.0f;
	}

	float together_loc[3 * PK_LANES], together_std[3 * PK_LANES];
	run_batch(&planes[0][0], PK_LANES, fit_type, together_loc, together_std);

	int diffs = 0;
	for (int l = 0; l < PK_LANES; ++l) {
		/* same plane alone in lane 0, surrounded by unrelated noise lanes */
		static float solo[PK_LANES][PLANE_NUMEL];
		memcpy(solo[0], planes[l], sizeof(solo[0]));
		for (int m = 1; m < PK_LANES; ++m) memcpy(solo[m], noise[m], sizeof(solo[m]));

		float solo_loc[3 * PK_LANES], solo_std[3 * PK_LANES];
		run_batch(&solo[0][0], PK_LANES, fit_type, solo_loc, solo_std);

		for (int comp = 0; comp < 3; ++comp) {
			float a = together_loc[comp * PK_LANES + l];
			float b = solo_loc[comp * PK_LANES + 0];
			/* bit-equality (NaN == NaN accepted) */
			if (!(memcmp(&a, &b, sizeof a) == 0 || (isnan(a) && isnan(b)))) {
				diffs++;
				printf("  lane %d comp %d: together %.9g vs solo %.9g\n", l, comp,
				       (double)a, (double)b);
			}
		}
	}
	printf("gate lane-independence type %d: %s\n", fit_type,
	       diffs ? "BROKEN" : "bit-equal");
	CHECK(diffs == 0, "lane independence type %d: %d component diffs", fit_type, diffs);
}

/* ── gate: zero-pad / partial tail ───────────────────────────────────────── */
static void gate_partial_tail(int fit_type)
{
	static float planes[PK_LANES][PLANE_NUMEL];
	g_rng = 0xBEEF0000u + (unsigned)fit_type;
	for (int l = 0; l < PK_LANES; ++l) gen_case(planes[l], fit_type, l);

	const int L_real = PK_LANES > 2 ? PK_LANES - 2 : 1;

	float full_loc[3 * PK_LANES], full_std[3 * PK_LANES];
	run_batch(&planes[0][0], PK_LANES, fit_type, full_loc, full_std);

	float part_loc[3 * PK_LANES], part_std[3 * PK_LANES];
	run_batch(&planes[0][0], L_real, fit_type, part_loc, part_std);

	int bad = 0;
	for (int l = 0; l < L_real; ++l)
		for (int comp = 0; comp < 3; ++comp) {
			float a = full_loc[comp * PK_LANES + l];
			float b = part_loc[comp * PK_LANES + l];
			if (!(memcmp(&a, &b, sizeof a) == 0 || (isnan(a) && isnan(b)))) bad++;
		}
	for (int l = L_real; l < PK_LANES; ++l)
		if (!isnan(part_loc[0 * PK_LANES + l])) bad++;   /* tail lanes must be NaN */

	printf("gate partial-tail type %d (L_real=%d): %s\n", fit_type, L_real,
	       bad ? "BROKEN" : "ok");
	CHECK(bad == 0, "partial tail type %d: %d violations", fit_type, bad);
}

int main(void)
{
	printf("peakfit gate: PK_LANES=%d, exp=%s\n", PK_LANES,
#ifdef PIV_BATCH_LIBM_EXP
	       "libm (reference hatch)"
#else
	       "poly (production)"
#endif
	);
	printf("batch available: %d, lanes: %d\n",
	       peakfit_batch_available(), peakfit_batch_lanes());

	gate_mask_plumbing();
	gate_vexpf_clamp();
	gate_vexpf_accuracy();

	for (int t = 4; t <= 6; ++t) {
		gate_a1_battery(t);
		gate_a3_nan_masks(t);
		gate_lane_independence(t);
		gate_partial_tail(t);
	}

	if (g_fail) { printf("GATE FAIL\n"); return 1; }
	printf("GATE PASS\n");
	return 0;
}

#endif /* PK_BATCH_AVAILABLE */
