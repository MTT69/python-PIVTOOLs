/****************************************************************************
 * test_codelet_gate.c -- standalone correctness gate for codelet_fft.c.
 *
 * Proves the codelet engine reproduces the FFT correlation math WITHOUT any
 * FFTW: it compares the engine's cross-/auto-correlation surfaces against a
 * brute-force circular correlation computed in double precision. Also checks
 * peak recovery for a known sub-window shift.
 *
 * Stage B additions: the SIMD-lane-batched engine is gated against the scalar
 * engine -- each lane j must reproduce codelet_plan's single-window output for
 * window j (proves the math is identical and the gather/scatter transpose does
 * not leak across lanes), and against the same brute-force double reference.
 *
 * Build (no FFTW); the -D selects the SIMD width and must match the codelets_gen.h
 * render present (e.g. NEON-4 needs `--isa scalar vecext4`):
 *   gcc-15 -O3 -fopenmp -ffp-contract=fast -DPIVTOOLS_FFT_ISA_NEON4 \
 *          codelet_fft.c test_codelet_gate.c -lm -o test_codelet_gate
 *   ./test_codelet_gate
 ****************************************************************************/
#include "codelet_fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Deterministic pseudo-random in [-1,1). */
static unsigned g_seed = 12345u;
static float frand(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
}

/* Brute-force circular correlation matching IFFT(FFT(X).*conj(FFT(Y)))/numel:
 *   ref[m,n] = sum_{i,j} X[i,j] * Y[(i-m) mod H, (j-n) mod W]
 * then fftshift (rows by H/2, cols by W/2), into out. */
static void ref_corr(const float *X, const float *Y, double *out, int H, int W) {
    for (int m = 0; m < H; ++m) {
        for (int n = 0; n < W; ++n) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i) {
                int ii = ((i - m) % H + H) % H;
                for (int j = 0; j < W; ++j) {
                    int jj = ((j - n) % W + W) % W;
                    acc += (double)X[i * W + j] * (double)Y[ii * W + jj];
                }
            }
            int row_swap = (m + H / 2) % H;
            int col_swap = (n + W / 2) % W;
            out[row_swap * W + col_swap] = acc;
        }
    }
}

static double max_rel_diff(const float *a, const double *b, int n, double *peak_out) {
    double m = 0.0, peak = 0.0;
    for (int i = 0; i < n; ++i) if (fabs(b[i]) > peak) peak = fabs(b[i]);
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - b[i]);
        if (d > m) m = d;
    }
    *peak_out = peak;
    return m / (peak > 0 ? peak : 1.0);
}

static int argmax(const float *s, int numel) {
    int bi = 0; float bv = s[0];
    for (int i = 1; i < numel; ++i) if (s[i] > bv) { bv = s[i]; bi = i; }
    return bi;
}

static int check_shape(int H, int W) {
    const int numel = H * W;
    float *A = malloc(numel * sizeof(float));
    float *B = malloc(numel * sizeof(float));
    float *outAB = malloc(numel * sizeof(float));
    float *outAA = malloc(numel * sizeof(float));
    float *outBB = malloc(numel * sizeof(float));
    double *ref = malloc(numel * sizeof(double));

    for (int i = 0; i < numel; ++i) { A[i] = frand(); B[i] = frand(); }

    codelet_plan *p = codelet_plan_create(H, W);
    if (!p) { fprintf(stderr, "plan_create(%d,%d) FAILED\n", H, W); return 1; }

    /* cross-correlation: forward(A,0), forward(B,1), emit -> AB */
    codelet_forward(p, A, 0);
    codelet_forward(p, B, 1);
    codelet_emit_xcorr(p, outAB);
    codelet_emit_power(p, 0, outAA);   /* |FFT(A)|^2 -> autocorr A */
    codelet_emit_power(p, 1, outBB);   /* |FFT(B)|^2 -> autocorr B */

    double peak, eAB, eAA, eBB;
    ref_corr(A, B, ref, H, W); eAB = max_rel_diff(outAB, ref, numel, &peak);
    ref_corr(A, A, ref, H, W); eAA = max_rel_diff(outAA, ref, numel, &peak);
    ref_corr(B, B, ref, H, W); eBB = max_rel_diff(outBB, ref, numel, &peak);

    const double tol = 1e-4;
    int ok = (eAB < tol) && (eAA < tol) && (eBB < tol);
    printf("  %3dx%-3d  AB=%.2e AA=%.2e BB=%.2e   %s\n",
           H, W, eAB, eAA, eBB, ok ? "OK" : "** FAIL **");

    codelet_plan_destroy(p);
    free(A); free(B); free(outAB); free(outAA); free(outBB); free(ref);
    return ok ? 0 : 1;
}

/* Peak recovery: B is A shifted by (dy,dx); the cross-correlation peak must
 * land at the fftshift-centered location encoding that shift. */
static int check_peak(int N, int dy, int dx) {
    const int numel = N * N;
    float *A = calloc(numel, sizeof(float));
    float *B = calloc(numel, sizeof(float));
    float *out = malloc(numel * sizeof(float));
    int cy = N / 2, cx = N / 2;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double ra = (i - cy) * (i - cy) + (j - cx) * (j - cx);
            A[i * N + j] = (float)exp(-ra / (2.0 * 3.0 * 3.0));
            double rb = (i - cy - dy) * (i - cy - dy) + (j - cx - dx) * (j - cx - dx);
            B[i * N + j] = (float)exp(-rb / (2.0 * 3.0 * 3.0));
        }
    codelet_plan *p = codelet_plan_create(N, N);
    /* xcorr_preplanned(B, A): slot0=B, slot1=A */
    codelet_forward(p, B, 0);
    codelet_forward(p, A, 1);
    codelet_emit_xcorr(p, out);
    int bi = argmax(out, numel);
    int py = bi / N - N / 2, px = bi % N - N / 2;
    int ok = (py == dy && px == dx);
    printf("  peak N=%d shift(%+d,%+d) -> recovered(%+d,%+d)   %s\n",
           N, dy, dx, py, px, ok ? "OK" : "** FAIL **");
    codelet_plan_destroy(p);
    free(A); free(B); free(out);
    return ok ? 0 : 1;
}

/* Float-vs-float max relative diff (normalized by the reference's peak). */
static double max_rel_diff_ff(const float *a, const float *b, int n) {
    double m = 0.0, peak = 0.0;
    for (int i = 0; i < n; ++i) if (fabs((double)b[i]) > peak) peak = fabs((double)b[i]);
    for (int i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m / (peak > 0 ? peak : 1.0);
}

/* ----------------------------------------------------------------------- *
 * Batched engine gate: build LANES distinct windows, run the batched path,
 * and require each lane j to reproduce (a) the scalar engine's single-window
 * output for window j, within FMA-contraction tolerance, AND (b) the
 * brute-force double reference. Distinct per-lane windows mean any cross-lane
 * leak from the gather/scatter transpose would break (a) immediately.
 * ----------------------------------------------------------------------- */
static int check_shape_batch(int H, int W) {
    const int L = codelet_lanes();
    const int numel = H * W;

    float  *packA  = malloc((size_t)L * numel * sizeof(float));
    float  *packB  = malloc((size_t)L * numel * sizeof(float));
    float  *packAB = malloc((size_t)L * numel * sizeof(float));
    float  *packAA = malloc((size_t)L * numel * sizeof(float));
    float  *packBB = malloc((size_t)L * numel * sizeof(float));
    float  *sAB = malloc(numel * sizeof(float));
    float  *sAA = malloc(numel * sizeof(float));
    float  *sBB = malloc(numel * sizeof(float));
    double *ref = malloc(numel * sizeof(double));

    for (int l = 0; l < L; ++l)
        for (int i = 0; i < numel; ++i) {
            packA[(size_t)l * numel + i] = frand();
            packB[(size_t)l * numel + i] = frand();
        }

    codelet_plan_b *pb = codelet_plan_create_batched(H, W);
    codelet_plan   *ps = codelet_plan_create(H, W);
    if (!pb || !ps) { fprintf(stderr, "batched/scalar plan_create(%d,%d) FAILED\n", H, W); return 1; }

    codelet_forward_batch(pb, packA, 0);
    codelet_forward_batch(pb, packB, 1);
    codelet_emit_xcorr_batch(pb, packAB);
    codelet_emit_power_batch(pb, 0, packAA);
    codelet_emit_power_batch(pb, 1, packBB);

    double worst_scalar = 0.0, worst_ref = 0.0;
    int peak_ok = 1;
    for (int l = 0; l < L; ++l) {
        const float *Aj = &packA[(size_t)l * numel];
        const float *Bj = &packB[(size_t)l * numel];
        const float *lAB = &packAB[(size_t)l * numel];
        const float *lAA = &packAA[(size_t)l * numel];
        const float *lBB = &packBB[(size_t)l * numel];

        /* scalar single-window oracle for this lane */
        codelet_forward(ps, Aj, 0);
        codelet_forward(ps, Bj, 1);
        codelet_emit_xcorr(ps, sAB);
        codelet_emit_power(ps, 0, sAA);
        codelet_emit_power(ps, 1, sBB);

        double s1 = max_rel_diff_ff(lAB, sAB, numel);
        double s2 = max_rel_diff_ff(lAA, sAA, numel);
        double s3 = max_rel_diff_ff(lBB, sBB, numel);
        if (s1 > worst_scalar) worst_scalar = s1;
        if (s2 > worst_scalar) worst_scalar = s2;
        if (s3 > worst_scalar) worst_scalar = s3;

        double peak;
        ref_corr(Aj, Bj, ref, H, W); { double e = max_rel_diff(lAB, ref, numel, &peak); if (e > worst_ref) worst_ref = e; }
        ref_corr(Aj, Aj, ref, H, W); { double e = max_rel_diff(lAA, ref, numel, &peak); if (e > worst_ref) worst_ref = e; }
        ref_corr(Bj, Bj, ref, H, W); { double e = max_rel_diff(lBB, ref, numel, &peak); if (e > worst_ref) worst_ref = e; }

        if (argmax(lAB, numel) != argmax(sAB, numel)) peak_ok = 0;
    }

    const double tol_scalar = 1e-4;   /* lane vs scalar: codegen FMA diff only (typ ~1e-6) */
    const double tol_ref    = 2e-4;   /* lane vs double brute-force: float32 FFT accumulation */
    int ok = (worst_scalar < tol_scalar) && (worst_ref < tol_ref) && peak_ok;
    printf("  %3dx%-3d L=%d  vs_scalar=%.2e vs_ref=%.2e peak=%s   %s\n",
           H, W, L, worst_scalar, worst_ref, peak_ok ? "ok" : "BAD",
           ok ? "OK" : "** FAIL **");

    codelet_plan_destroy_batched(pb);
    codelet_plan_destroy(ps);
    free(packA); free(packB); free(packAB); free(packAA); free(packBB);
    free(sAB); free(sAA); free(sBB); free(ref);
    return ok ? 0 : 1;
}

/* Lane-independence / tail-like check: only lane `live` carries a shifted-blob
 * pair (peak at (dy,dx)); all other lanes carry unrelated random data. The live
 * lane must still recover its peak exactly -- i.e. a partially-populated batch
 * (as the production tail could present) is computed per-lane without bleed. */
static int check_peak_batch(int N, int dy, int dx, int live) {
    const int L = codelet_lanes();
    const int numel = N * N;
    float *packA = malloc((size_t)L * numel * sizeof(float));
    float *packB = malloc((size_t)L * numel * sizeof(float));
    float *packAB = malloc((size_t)L * numel * sizeof(float));

    for (int l = 0; l < L; ++l)
        for (int i = 0; i < numel; ++i) { packA[(size_t)l*numel+i] = frand(); packB[(size_t)l*numel+i] = frand(); }

    /* xcorr_preplanned(B, A): slot0=B, slot1=A. Put the blob pair in lane `live`. */
    float *Blive = &packB[(size_t)live * numel];   /* slot0 */
    float *Alive = &packA[(size_t)live * numel];   /* slot1 */
    int cy = N / 2, cx = N / 2;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double ra = (i - cy) * (i - cy) + (j - cx) * (j - cx);
            Alive[i * N + j] = (float)exp(-ra / (2.0 * 3.0 * 3.0));
            double rb = (i - cy - dy) * (i - cy - dy) + (j - cx - dx) * (j - cx - dx);
            Blive[i * N + j] = (float)exp(-rb / (2.0 * 3.0 * 3.0));
        }

    codelet_plan_b *pb = codelet_plan_create_batched(N, N);
    codelet_forward_batch(pb, packB, 0);
    codelet_forward_batch(pb, packA, 1);
    codelet_emit_xcorr_batch(pb, packAB);

    int bi = argmax(&packAB[(size_t)live * numel], numel);
    int py = bi / N - N / 2, px = bi % N - N / 2;
    int ok = (py == dy && px == dx);
    printf("  peak N=%d lane %d/%d shift(%+d,%+d) -> (%+d,%+d)   %s\n",
           N, live, L, dy, dx, py, px, ok ? "OK" : "** FAIL **");

    codelet_plan_destroy_batched(pb);
    free(packA); free(packB); free(packAB);
    return ok ? 0 : 1;
}

int main(void) {
    int fails = 0;
    printf("== codelet engine vs brute-force circular correlation ==\n");
    int sizes[] = {8, 12, 16, 24, 32, 48, 64, 96, 128};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
        fails += check_shape(sizes[i], sizes[i]);
    printf("== rectangular ==\n");
    fails += check_shape(16, 32);
    fails += check_shape(32, 16);
    fails += check_shape(24, 48);
    fails += check_shape(64, 32);
    fails += check_shape(96, 128);
    printf("== peak recovery (scalar) ==\n");
    fails += check_peak(32, 3, -2);
    fails += check_peak(64, -5, 7);
    fails += check_peak(16, 1, 1);

    printf("== BATCHED engine vs scalar oracle + brute-force (L=%d lanes) ==\n", codelet_lanes());
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
        fails += check_shape_batch(sizes[i], sizes[i]);
    printf("== batched rectangular ==\n");
    fails += check_shape_batch(16, 32);
    fails += check_shape_batch(32, 16);
    fails += check_shape_batch(24, 48);
    fails += check_shape_batch(96, 128);
    printf("== batched peak recovery / lane independence ==\n");
    fails += check_peak_batch(32, 3, -2, 0);
    fails += check_peak_batch(64, -5, 7, codelet_lanes() - 1);
    fails += check_peak_batch(16, 1, 1, codelet_lanes() / 2);

    printf(fails ? "\nGATE FAIL (%d)\n" : "\nGATE PASS\n", fails);
    return fails ? 1 : 0;
}
