/******************************************************************************
 * bench_peak_fit.c — standalone timing harness: scalar vs batched LM fitter.
 *
 * NOT part of the DLL link. Build (Windows, after vcvars64):
 *   clang-cl /O2 /arch:AVX2 /clang:-fno-math-errno /DPIVTOOLS_FFT_ISA_AVX2 ^
 *       peak_locate_lm.c peak_locate_lm_batch.c bench_peak_fit.c /Fe:pkbench.exe
 *
 * Run:  pkbench [noise_frac]      (default 0.02 = 2% amplitude noise)
 * Pin affinity externally for the P/E-core split, e.g.:
 *   start /affinity 0xFFFF pkbench       (8 P-cores = logical 0-15 on 8P+12E)
 *   start /affinity 0xFFFF0000 pkbench   (E-cores 16-27)
 *
 * Reports per-fit ns for scalar and batch per fit type, the speedup, and the
 * scalar iteration histogram (lockstep pays the max lane iteration count, so
 * divergence explains any efficiency loss).
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "peak_simd.h"
#include "peak_locate_lm.h"
#include "peak_locate_lm_batch.h"

#if !PK_BATCH_AVAILABLE
int main(void) { printf("batch fitter unavailable on this compiler\n"); return 0; }
#else

#define PLANE_N 33
#define PLANE_NUMEL (PLANE_N * PLANE_N)
#define NPLANES 4096            /* battery size (multiple of any lane count) */
#define NREP 5                  /* timing repetitions, best-of */

static unsigned int g_rng = 0xA5A5A5A5u;
static float frand(void) {
	g_rng = g_rng * 1664525u + 1013904223u;
	return (float)(g_rng >> 8) / 16777216.0f;
}

static float *g_planes;         /* [NPLANES][PLANE_NUMEL] */

static void gen_battery(int fit_type, float noise_frac)
{
	int c = (PLANE_N - 1) / 2;
	g_rng = 0xA5A50000u + (unsigned)fit_type;
	for (int k = 0; k < NPLANES; ++k) {
		float *p = g_planes + (size_t)k * PLANE_NUMEL;
		float di = -0.45f + 0.9f * frand();
		float dj = -0.45f + 0.9f * frand();
		float amp = 100.0f + 900.0f * frand();
		float s1 = 1.0f + 1.5f * frand();
		float s2 = 1.0f + 1.5f * frand();
		float cov = (frand() - 0.5f) * 0.4f / (s1 * s2);
		for (int i = 0; i < PLANE_N; ++i)
			for (int j = 0; j < PLANE_N; ++j) {
				float y = (float)(i - c) - di, x = (float)(j - c) - dj;
				float v;
				if (fit_type == 4)      v = amp * expf(-(y*y + x*x) / (s1*s1));
				else if (fit_type == 5) v = amp * expf(-(y*y/(s1*s1) + x*x/(s2*s2)));
				else                    v = amp * expf(-0.5f * (y*y/(s1*s1) + x*x/(s2*s2) + 2.0f*y*x*cov));
				p[i * PLANE_N + j] = v + noise_frac * amp * (frand() - 0.5f);
			}
	}
}

static double now_s(void)
{
	return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
	float noise_frac = (argc > 1) ? (float)atof(argv[1]) : 0.02f;
	int N[2] = { PLANE_N, PLANE_N };
	const int W = PK_LANES;

	g_planes = (float *)malloc((size_t)NPLANES * PLANE_NUMEL * sizeof(float));
	if (!g_planes) { fprintf(stderr, "alloc failed\n"); return 1; }

	printf("bench_peak_fit: lanes=%d planes=%d noise=%.3f exp=%s\n",
	       W, NPLANES, (double)noise_frac,
#ifdef PIV_BATCH_LIBM_EXP
	       "libm");
#else
	       "poly");
#endif
	printf("%-6s %14s %14s %9s\n", "type", "scalar ns/fit", "batch ns/fit", "speedup");

	for (int fit_type = 4; fit_type <= 6; ++fit_type) {
		gen_battery(fit_type, noise_frac);

		/* scalar timing (best of NREP) */
		double t_scalar = 1e30;
		volatile float sink = 0.0f;
		for (int rep = 0; rep < NREP; ++rep) {
			double t0 = now_s();
			for (int k = 0; k < NPLANES; ++k) {
				float loc[3], std[3];
				lsqpeaklocate_lm(g_planes + (size_t)k * PLANE_NUMEL, N, loc, 1, fit_type, std);
				sink += loc[0];
			}
			double dt = now_s() - t0;
			if (dt < t_scalar) t_scalar = dt;
		}

		/* batch timing (best of NREP) */
		double t_batch = 1e30;
		for (int rep = 0; rep < NREP; ++rep) {
			double t0 = now_s();
			for (int k = 0; k < NPLANES; k += W) {
				float loc[3 * PK_LANES], std[3 * PK_LANES];
				int nb = (NPLANES - k) < W ? (NPLANES - k) : W;
				lsqpeaklocate_lm_batch(g_planes + (size_t)k * PLANE_NUMEL, nb, N, fit_type, loc, std);
				sink += loc[0];
			}
			double dt = now_s() - t0;
			if (dt < t_batch) t_batch = dt;
		}

		printf("%-6d %14.0f %14.0f %8.2fx\n", fit_type,
		       t_scalar / NPLANES * 1e9, t_batch / NPLANES * 1e9,
		       t_scalar / t_batch);
		(void)sink;
	}

	free(g_planes);
	return 0;
}

#endif /* PK_BATCH_AVAILABLE */
