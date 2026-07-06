/******************************************************************************
 * peak_fit_bench.c — authoritative C battery microbench for the LM peak-fit.
 *
 * Measures pure per-fit kernel cost (us/fit) for gauss4/5/6 by calling the
 * static lm_gaussN_fit() directly on synthetic 5x5 correlation subwindows.
 * This bypasses BOTH the N^2 peak search in lsqpeaklocate_lm AND the
 * Python/ctypes overhead that dominates the manual_tools/peak_fit_ab.py --bench
 * path, so it can resolve the poly-vs-libm expf delta the harness cannot.
 *
 * Build (under vcvars64), e.g. shipping flags + polynomial:
 *   cl /O2 /openmp:experimental /I..\pivtools_cli\lib peak_fit_bench.c
 * libm reference: add /DPIV_USE_LIBM_EXP ; AVX2: add /arch:AVX2.
 *
 * The fit functions are file-static, so we pull the translation unit in via
 * #include; MSVC resolves peak_locate_lm.c's own quoted includes (common.h,
 * fast_exp.h, peak_locate_lm.h) relative to that file's directory.
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <intrin.h>

/* Which kernel translation unit to pull in (statics need #include, not link).
 * Default = current worktree (Lever 1+2, PIV_EXP switch). Override with
 * -DPIV_BENCH_PEAK_SRC=... to bench the original pre-Lever-2 kernel. */
#ifndef PIV_BENCH_PEAK_SRC
#define PIV_BENCH_PEAK_SRC "../pivtools_cli/lib/peak_locate_lm.c"
#endif
#include PIV_BENCH_PEAK_SRC

#ifndef BENCH_NP
#define BENCH_NP 4800          /* number of distinct subwindows (mirrors A/B harness) */
#endif

#define PK (PKSIZE_X * PKSIZE_Y)   /* 25 */

static float g_planes[BENCH_NP][PK];

/* Deterministic LCG -> reproducible planes, zero run-to-run RNG variance. */
static unsigned long long g_lcg = 0x9E3779B97F4A7C15ull;
static float frand(void) {            /* uniform [0,1) */
    g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((g_lcg >> 40) & 0xFFFFFF) / (float)0x1000000;
}
static float nrand(void) {            /* approx N(0,1), sum-of-12 */
    float s = 0.0f; for (int k = 0; k < 12; ++k) s += frand();
    return s - 6.0f;
}

/* Synthesize a realistic extracted 5x5 correlation peak: a Gaussian bump with
 * sub-pixel offset and PIV-typical width, half the set clean, half noisy. The
 * exact amplitude/width matter only insofar as they drive a realistic LM
 * iteration count (hence exp-call count); these track real instantaneous data. */
static void gen_planes(void) {
    const int c = (PKSIZE_X - 1) / 2;            /* center index = 2 */
    for (int p = 0; p < BENCH_NP; ++p) {
        float di = (frand() - 0.5f) * 0.8f;      /* sub-pixel offset [-0.4,0.4] */
        float dj = (frand() - 0.5f) * 0.8f;
        float sig = 1.0f + frand() * 0.6f;       /* width ~ 1.0..1.6 px */
        float A = 0.8f + frand() * 0.4f;
        int noisy = (p & 1);
        float noise_amp = noisy ? 0.05f : 0.0f;  /* SNR ~ 20 on noisy half */
        for (int i = 0; i < PKSIZE_X; ++i) {
            for (int j = 0; j < PKSIZE_Y; ++j) {
                float x = (float)(i - c) - di;
                float y = (float)(j - c) - dj;
                float v = A * expf(-(x*x + y*y) / (2.0f * sig * sig));
                v += 1e-3f + noise_amp * nrand();
                g_planes[p][i * PKSIZE_Y + j] = v;
            }
        }
    }
}

static void report_isa(void) {
    int r[4];
    __cpuidex(r, 7, 0);
    int avx2    = (r[1] >> 5)  & 1;   /* EBX bit 5  */
    int avx512f = (r[1] >> 16) & 1;   /* EBX bit 16 */
    __cpuid(r, 1);
    int avx = (r[2] >> 28) & 1;       /* ECX bit 28 */
    printf("# CPU ISA: AVX=%d AVX2=%d AVX512F=%d\n", avx, avx2, avx512f);
#ifdef PIV_USE_LIBM_EXP
    printf("# exp impl: libm expf (PIV_USE_LIBM_EXP)\n");
#else
    printf("# exp impl: polynomial piv_expf\n");
#endif
#ifdef _OPENMP
    printf("# _OPENMP defined: omp simd reduction pragma ACTIVE\n");
#else
    printf("# _OPENMP not defined: residual pragma inert\n");
#endif
}

typedef void (*fitfn)(const float*, const int*, float*, float*, float*);

static double bench_fit(int fittype, int reps, double *checksum_out) {
    const int N[2] = { PKSIZE_X, PKSIZE_Y };
    float peak[2], sig[3], fitval[PK];
    double checksum = 0.0;
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    /* Warmup (also primes caches and branch predictors). */
    for (int p = 0; p < BENCH_NP; ++p) {
        switch (fittype) {
            case 4: lm_gauss4_fit(g_planes[p], N, peak, fitval, sig); break;
            case 5: lm_gauss5_fit(g_planes[p], N, peak, fitval, sig); break;
            case 6: lm_gauss6_fit(g_planes[p], N, peak, fitval, sig); break;
        }
        checksum += peak[0] + peak[1] + sig[0];
    }

    QueryPerformanceCounter(&t0);
    for (int r = 0; r < reps; ++r) {
        for (int p = 0; p < BENCH_NP; ++p) {
            switch (fittype) {
                case 4: lm_gauss4_fit(g_planes[p], N, peak, fitval, sig); break;
                case 5: lm_gauss5_fit(g_planes[p], N, peak, fitval, sig); break;
                case 6: lm_gauss6_fit(g_planes[p], N, peak, fitval, sig); break;
            }
            checksum += peak[0] + peak[1] + sig[0];
        }
    }
    QueryPerformanceCounter(&t1);

    *checksum_out = checksum;
    double secs = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double nfits = (double)reps * (double)BENCH_NP;
    return secs / nfits * 1e6;   /* us per fit */
}

int main(int argc, char **argv) {
    int reps = (argc > 1) ? atoi(argv[1]) : 400;
    report_isa();
    printf("# NP=%d reps=%d  (%.2fM fits per type)\n",
           BENCH_NP, reps, (double)reps * BENCH_NP / 1e6);
    gen_planes();

    /* Run each type a few times; report the MIN (least contended) us/fit. */
    const int trials = 5;
    for (int ti = 0; ti < 3; ++ti) {
        int ft = (ti == 0) ? 4 : (ti == 1) ? 5 : 6;
        double best = 1e30, cs = 0.0;
        for (int t = 0; t < trials; ++t) {
            double cs_t;
            double us = bench_fit(ft, reps, &cs_t);
            if (us < best) best = us;
            cs = cs_t;
        }
        printf("gauss%d  %.4f us/fit   (checksum %.3f)\n", ft, best, cs);
    }
    return 0;
}
