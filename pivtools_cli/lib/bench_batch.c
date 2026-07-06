/* Isolated microbench of the in-repo batched codelet transform.
 * Times the "single" level (2 forward r2c + xcorr emit) on pre-packed,
 * cache-resident data — exactly what the standalone spike measured for
 * codelet8. Reports ns/window so it's directly comparable to the spike CSV
 * (fftw@32=3725, codelet8@32=2048; fftw@64=17305, codelet8@64=8599). */
#include "codelet_fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(int N) {
    int L = codelet_lanes();
    int numel = N * N;
    codelet_plan_b *p = codelet_plan_create_batched(N, N);
    float *pA = malloc((size_t)L * numel * sizeof(float));
    float *pB = malloc((size_t)L * numel * sizeof(float));
    float *pC = malloc((size_t)L * numel * sizeof(float));
    for (int i = 0; i < L * numel; ++i) { pA[i] = (float)rand()/RAND_MAX - 0.5f; pB[i] = (float)rand()/RAND_MAX - 0.5f; }

    /* warmup */
    for (int it = 0; it < 20; ++it) {
        codelet_forward_batch(p, pB, 0);
        codelet_forward_batch(p, pA, 1);
        codelet_emit_xcorr_batch(p, pC);
    }
    /* pick iters so each size runs ~0.5s */
    long iters = (N <= 32) ? 20000 : (N <= 64) ? 6000 : 1500;
    double t0 = now_s();
    for (long it = 0; it < iters; ++it) {
        codelet_forward_batch(p, pB, 0);
        codelet_forward_batch(p, pA, 1);
        codelet_emit_xcorr_batch(p, pC);
    }
    double dt = now_s() - t0;
    double ns_per_window = dt * 1e9 / ((double)iters * L);
    printf("  N=%3d  L=%d  %.1f ns/window  (%ld iters, %.3fs)  sink=%g\n",
           N, L, ns_per_window, iters, dt, pC[0]);
    free(pA); free(pB); free(pC);
    codelet_plan_destroy_batched(p);
}

int main(void) {
    printf("=== in-repo batched codelet transform (single level: 2x r2c + xcorr) ===\n");
    bench(32);
    bench(64);
    bench(128);
    return 0;
}
