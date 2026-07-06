/* Isolate the cost of convolve() (the per-pass correlation-plane weight).
 * This is the O(N^4) direct circular correlation suspected of dominating the
 * pipeline at large windows. Times one convolve() call per size. */
#include "xcorr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench(int Nsz) {
    int N[2] = {Nsz, Nsz};
    int numel = Nsz * Nsz;
    float *wA = malloc(numel * sizeof(float));
    float *wB = malloc(numel * sizeof(float));
    float *out = malloc(numel * sizeof(float));
    /* dense gaussian window weight (separable, but treated as 2D) */
    double s = Nsz / 4.0;
    for (int i = 0; i < Nsz; ++i)
        for (int j = 0; j < Nsz; ++j) {
            double di = i - Nsz/2.0, dj = j - Nsz/2.0;
            wA[i*Nsz+j] = wB[i*Nsz+j] = (float)exp(-(di*di+dj*dj)/(2*s*s));
        }
    /* warm + time a few calls */
    convolve(wA, wB, out, N);
    int reps = (Nsz <= 32) ? 50 : (Nsz <= 64) ? 10 : 3;
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) convolve(wA, wB, out, N);
    double dt = (now_s() - t0) / reps;
    printf("  convolve N=%3d : %8.2f ms/call   (sink=%g)\n", Nsz, dt*1e3, out[0]);
    free(wA); free(wB); free(out);
}

int main(void) {
    printf("=== convolve() cost per call (the per-pass correlation-plane weight) ===\n");
    bench(32); bench(64); bench(128);
    return 0;
}
