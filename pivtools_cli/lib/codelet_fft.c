/****************************************************************************
 * codelet_fft.c -- separable 2D FFT assembled from generated 1D codelets.
 * See codelet_fft.h. Ported from the gate-verified reference
 * (fft_bench CodeletScalarBackend), preserving the exact correlation math.
 ****************************************************************************/
#include "codelet_fft.h"
#include "codelets_gen.h"   /* generated: rfftN_scalar / cfftN_scalar / icfftN_scalar / irfftN_scalar */
#include "codelet_simd.h"   /* Stage B: PIV_VEC + lane-typed codelets (rfftN_v4 / _v8 / _v16) */

#include <stdlib.h>
#include <string.h>

/* 1D codelet signatures (scalar render). */
typedef void (*rfft_fn)(const float *, float *, float *);          /* N real -> N/2+1 cplx */
typedef void (*cfft_fn)(const float *, const float *, float *, float *); /* N cplx -> N cplx (fwd/inv) */
typedef void (*irfft_fn)(const float *, const float *, float *);   /* N/2+1 cplx -> N real */

struct codelet_plan {
    int H, W, halfW, numel, numel_fft;

    rfft_fn  rfft;   /* length-W, rows  */
    irfft_fn irfft;  /* length-W, rows  */
    cfft_fn  cfft;   /* length-H, cols (forward) */
    cfft_fn  icfft;  /* length-H, cols (inverse) */

    /* 2D stage scratch (split re/im), H*halfW each. */
    float *rowR, *rowI, *colR, *colI;
    /* per-column scratch, length H (<=128). */
    float *cXr, *cXi, *cYr, *cYi;

    /* spectra + inverse scratch. */
    codelet_cplx *spec0, *spec1, *C;
    float        *raw;
};

int codelet_size_ok(int n) {
    switch (n) {
        case 8: case 12: case 16: case 24: case 32:
        case 48: case 64: case 96: case 128:
        /* 192/256 exist only for the scalar path: convolve() pads a 96/128
         * window to 2N for the correlation-plane weight. They are NOT
         * selectable interrogation-window sizes (Python BUILT_FFT_SIZES stays
         * 8..128) and have no batched/SIMD render. */
        case 192: case 256:
            return 1;
        default:
            return 0;
    }
}

/* Bind the row (length W) and column (length H) codelets. */
static int bind_codelets(codelet_plan *p) {
#define CL_ROW(NN) case NN: p->rfft = rfft##NN##_scalar; p->irfft = irfft##NN##_scalar; break;
    switch (p->W) {
        CL_ROW(8) CL_ROW(12) CL_ROW(16) CL_ROW(24) CL_ROW(32)
        CL_ROW(48) CL_ROW(64) CL_ROW(96) CL_ROW(128)
        CL_ROW(192) CL_ROW(256)   /* scalar-only: convolve weight padding (2N) */
        default: return 0;
    }
#undef CL_ROW
#define CL_COL(NN) case NN: p->cfft = cfft##NN##_scalar; p->icfft = icfft##NN##_scalar; break;
    switch (p->H) {
        CL_COL(8) CL_COL(12) CL_COL(16) CL_COL(24) CL_COL(32)
        CL_COL(48) CL_COL(64) CL_COL(96) CL_COL(128)
        CL_COL(192) CL_COL(256)   /* scalar-only: convolve weight padding (2N) */
        default: return 0;
    }
#undef CL_COL
    return 1;
}

codelet_plan *codelet_plan_create(int H, int W) {
    if (!codelet_size_ok(H) || !codelet_size_ok(W)) return NULL;

    codelet_plan *p = (codelet_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;

    p->H = H; p->W = W; p->halfW = W / 2 + 1;
    p->numel = H * W; p->numel_fft = H * p->halfW;

    if (!bind_codelets(p)) { free(p); return NULL; }

    const size_t nrc = (size_t)H * p->halfW;
    p->rowR = (float *)malloc(nrc * sizeof(float));
    p->rowI = (float *)malloc(nrc * sizeof(float));
    p->colR = (float *)malloc(nrc * sizeof(float));
    p->colI = (float *)malloc(nrc * sizeof(float));
    p->cXr  = (float *)malloc((size_t)H * sizeof(float));
    p->cXi  = (float *)malloc((size_t)H * sizeof(float));
    p->cYr  = (float *)malloc((size_t)H * sizeof(float));
    p->cYi  = (float *)malloc((size_t)H * sizeof(float));
    p->spec0 = (codelet_cplx *)malloc((size_t)p->numel_fft * sizeof(codelet_cplx));
    p->spec1 = (codelet_cplx *)malloc((size_t)p->numel_fft * sizeof(codelet_cplx));
    p->C     = (codelet_cplx *)malloc((size_t)p->numel_fft * sizeof(codelet_cplx));
    p->raw   = (float *)malloc((size_t)p->numel * sizeof(float));

    if (!p->rowR || !p->rowI || !p->colR || !p->colI ||
        !p->cXr || !p->cXi || !p->cYr || !p->cYi ||
        !p->spec0 || !p->spec1 || !p->C || !p->raw) {
        codelet_plan_destroy(p);
        return NULL;
    }
    return p;
}

void codelet_plan_destroy(codelet_plan *p) {
    if (!p) return;
    free(p->rowR); free(p->rowI); free(p->colR); free(p->colI);
    free(p->cXr); free(p->cXi); free(p->cYr); free(p->cYi);
    free(p->spec0); free(p->spec1); free(p->C); free(p->raw);
    free(p);
}

int codelet_plan_numel(const codelet_plan *p)     { return p->numel; }
int codelet_plan_numel_fft(const codelet_plan *p) { return p->numel_fft; }

/* Separable 2D r2c: rfft along rows (length W) -> cfft along columns (length H). */
static void r2c_2d(codelet_plan *p, const float *in, codelet_cplx *out) {
    const int H = p->H, W = p->W, halfW = p->halfW;
    for (int r = 0; r < H; ++r)
        p->rfft(&in[r * W], &p->rowR[r * halfW], &p->rowI[r * halfW]);   /* rows */
    for (int k = 0; k < halfW; ++k) {                                    /* columns */
        for (int r = 0; r < H; ++r) {
            p->cXr[r] = p->rowR[r * halfW + k];
            p->cXi[r] = p->rowI[r * halfW + k];
        }
        p->cfft(p->cXr, p->cXi, p->cYr, p->cYi);
        for (int r = 0; r < H; ++r) {
            out[r * halfW + k][0] = p->cYr[r];
            out[r * halfW + k][1] = p->cYi[r];
        }
    }
}

/* Separable 2D c2r: icfft along columns (length H) -> irfft along rows (length W). */
static void c2r_2d(codelet_plan *p, const codelet_cplx *in, float *out) {
    const int H = p->H, W = p->W, halfW = p->halfW;
    for (int k = 0; k < halfW; ++k) {                                    /* columns (inverse) */
        for (int r = 0; r < H; ++r) {
            p->cXr[r] = in[r * halfW + k][0];
            p->cXi[r] = in[r * halfW + k][1];
        }
        p->icfft(p->cXr, p->cXi, p->cYr, p->cYi);
        for (int r = 0; r < H; ++r) {
            p->colR[r * halfW + k] = p->cYr[r];
            p->colI[r * halfW + k] = p->cYi[r];
        }
    }
    for (int r = 0; r < H; ++r)                                          /* rows (Hermitian) */
        p->irfft(&p->colR[r * halfW], &p->colI[r * halfW], &out[r * W]);
}

/* Normalize by 1/numel and fftshift `raw` into `out` (verbatim from xcorr.c:299-318). */
static void finish(codelet_plan *p, float *out) {
    const int H = p->H, W = p->W, numel = p->numel;
    float *cc = p->raw;
    const float mul = 1.0f / (float)numel;
    for (int j = 0; j < numel; ++j) cc[j] *= mul;
    for (int row = 0; row < H; ++row) {
        int row_swap = (row + H / 2) % H;
        memcpy(&out[row * W + W / 2], &cc[row_swap * W],         (size_t)(W / 2) * sizeof(float));
        memcpy(&out[row * W],         &cc[row_swap * W + W / 2], (size_t)(W / 2) * sizeof(float));
    }
}

void codelet_forward(codelet_plan *p, const float *in, int slot) {
    r2c_2d(p, in, slot == 0 ? p->spec0 : p->spec1);
}

void codelet_emit_xcorr(codelet_plan *p, float *out) {
    const int n = p->numel_fft;
    /* C = spec0 .* conj(spec1)  (identical to multiply_conjugate in xcorr.c) */
    for (int i = 0; i < n; ++i) {
        const float Ar = p->spec0[i][0], Ai = p->spec0[i][1];
        const float Br = p->spec1[i][0], Bi = p->spec1[i][1];
        p->C[i][0] = Ar * Br + Ai * Bi;
        p->C[i][1] = Ai * Br - Ar * Bi;
    }
    c2r_2d(p, p->C, p->raw);
    finish(p, out);
}

void codelet_emit_power(codelet_plan *p, int slot, float *out) {
    const int n = p->numel_fft;
    const codelet_cplx *F = (slot == 0) ? p->spec0 : p->spec1;
    for (int i = 0; i < n; ++i) {
        p->C[i][0] = F[i][0] * F[i][0] + F[i][1] * F[i][1];
        p->C[i][1] = 0.0f;
    }
    c2r_2d(p, p->C, p->raw);
    finish(p, out);
}

/* ======================================================================== *
 *  Stage B -- SIMD-lane-batched engine (PIV_VLANES windows per call).
 *
 *  Identical math to the scalar engine above, just PIV_VEC-typed: the codelet
 *  butterflies, conjugate-multiply, power spectrum, normalize and fftshift all
 *  run lane-wise, so lane j reproduces the scalar single-window result for
 *  window j (within FMA-contraction tolerance). Spectra are stored SPLIT
 *  real/imag (the vec codelets emit Yr,Yi separately), unlike the scalar
 *  engine's interleaved codelet_cplx.
 * ======================================================================== */

/* Batched (lane-typed) 1D codelet signatures. */
typedef void (*rfft_b_fn)(const PIV_VEC *, PIV_VEC *, PIV_VEC *);
typedef void (*cfft_b_fn)(const PIV_VEC *, const PIV_VEC *, PIV_VEC *, PIV_VEC *);
typedef void (*irfft_b_fn)(const PIV_VEC *, const PIV_VEC *, PIV_VEC *);

struct codelet_plan_b {
    int H, W, halfW, numel, numel_fft;

    rfft_b_fn  rfft;          /* length-W, rows  */
    irfft_b_fn irfft;         /* length-W, rows  */
    cfft_b_fn  cfft, icfft;   /* length-H, cols  */

    PIV_VEC *in;                          /* numel     -- gather destination       */
    PIV_VEC *FAr, *FAi, *FBr, *FBi;       /* numel_fft -- the two split spectra     */
    PIV_VEC *Cr, *Ci;                     /* numel_fft -- product / power spectrum  */
    PIV_VEC *rowR, *rowI, *colR, *colI;   /* H*halfW   -- separable stage scratch   */
    PIV_VEC *rawv;                        /* numel     -- inverse output            */
    PIV_VEC *outv;                        /* numel     -- normalized + fftshifted   */
};

int codelet_lanes(void) { return PIV_VLANES; }

/* Bind row (length W) and column (length H) lane-typed codelets. */
static int bind_codelets_b(codelet_plan_b *p) {
#define CLB_ROW(NN) case NN: p->rfft = PIV_RFFT(NN); p->irfft = PIV_IRFFT(NN); break;
    switch (p->W) {
        CLB_ROW(8) CLB_ROW(12) CLB_ROW(16) CLB_ROW(24) CLB_ROW(32)
        CLB_ROW(48) CLB_ROW(64) CLB_ROW(96) CLB_ROW(128)
        default: return 0;
    }
#undef CLB_ROW
#define CLB_COL(NN) case NN: p->cfft = PIV_CFFT(NN); p->icfft = PIV_ICFFT(NN); break;
    switch (p->H) {
        CLB_COL(8) CLB_COL(12) CLB_COL(16) CLB_COL(24) CLB_COL(32)
        CLB_COL(48) CLB_COL(64) CLB_COL(96) CLB_COL(128)
        default: return 0;
    }
#undef CLB_COL
    return 1;
}

codelet_plan_b *codelet_plan_create_batched(int H, int W) {
    if (!codelet_size_ok(H) || !codelet_size_ok(W)) return NULL;

    codelet_plan_b *p = (codelet_plan_b *)calloc(1, sizeof(*p));
    if (!p) return NULL;

    p->H = H; p->W = W; p->halfW = W / 2 + 1;
    p->numel = H * W; p->numel_fft = H * p->halfW;

    if (!bind_codelets_b(p)) { free(p); return NULL; }

    const size_t nf = (size_t)p->numel_fft;
    const size_t nrc = (size_t)H * p->halfW;
    const size_t ne = (size_t)p->numel;

    p->in   = PIV_VEC_ALLOC(ne);
    p->FAr  = PIV_VEC_ALLOC(nf); p->FAi = PIV_VEC_ALLOC(nf);
    p->FBr  = PIV_VEC_ALLOC(nf); p->FBi = PIV_VEC_ALLOC(nf);
    p->Cr   = PIV_VEC_ALLOC(nf); p->Ci  = PIV_VEC_ALLOC(nf);
    p->rowR = PIV_VEC_ALLOC(nrc); p->rowI = PIV_VEC_ALLOC(nrc);
    p->colR = PIV_VEC_ALLOC(nrc); p->colI = PIV_VEC_ALLOC(nrc);
    p->rawv = PIV_VEC_ALLOC(ne); p->outv = PIV_VEC_ALLOC(ne);

    if (!p->in || !p->FAr || !p->FAi || !p->FBr || !p->FBi || !p->Cr || !p->Ci ||
        !p->rowR || !p->rowI || !p->colR || !p->colI || !p->rawv || !p->outv) {
        codelet_plan_destroy_batched(p);
        return NULL;
    }
    return p;
}

void codelet_plan_destroy_batched(codelet_plan_b *p) {
    if (!p) return;
    PIV_VEC_FREE(p->in);
    PIV_VEC_FREE(p->FAr); PIV_VEC_FREE(p->FAi);
    PIV_VEC_FREE(p->FBr); PIV_VEC_FREE(p->FBi);
    PIV_VEC_FREE(p->Cr);  PIV_VEC_FREE(p->Ci);
    PIV_VEC_FREE(p->rowR); PIV_VEC_FREE(p->rowI);
    PIV_VEC_FREE(p->colR); PIV_VEC_FREE(p->colI);
    PIV_VEC_FREE(p->rawv); PIV_VEC_FREE(p->outv);
    free(p);
}

/* Separable 2D r2c (lane-wise): rfft rows (length W) -> cfft cols (length H). */
static void r2c_2d_b(codelet_plan_b *p, const PIV_VEC *in, PIV_VEC *OR, PIV_VEC *OI) {
    const int H = p->H, W = p->W, halfW = p->halfW;
    PIV_ALIGN(PIV_VEC_ALIGN) PIV_VEC cXr[128], cXi[128], cYr[128], cYi[128];
    for (int r = 0; r < H; ++r)
        p->rfft(&in[r * W], &p->rowR[r * halfW], &p->rowI[r * halfW]);   /* rows */
    for (int k = 0; k < halfW; ++k) {                                    /* columns */
        for (int r = 0; r < H; ++r) { cXr[r] = p->rowR[r * halfW + k]; cXi[r] = p->rowI[r * halfW + k]; }
        p->cfft(cXr, cXi, cYr, cYi);
        for (int r = 0; r < H; ++r) { OR[r * halfW + k] = cYr[r]; OI[r * halfW + k] = cYi[r]; }
    }
}

/* Separable 2D c2r (lane-wise): icfft cols (length H) -> irfft rows (length W). */
static void c2r_2d_b(codelet_plan_b *p, const PIV_VEC *IR, const PIV_VEC *II, PIV_VEC *out) {
    const int H = p->H, W = p->W, halfW = p->halfW;
    PIV_ALIGN(PIV_VEC_ALIGN) PIV_VEC cXr[128], cXi[128], cYr[128], cYi[128];
    for (int k = 0; k < halfW; ++k) {                                    /* columns (inverse) */
        for (int r = 0; r < H; ++r) { cXr[r] = IR[r * halfW + k]; cXi[r] = II[r * halfW + k]; }
        p->icfft(cXr, cXi, cYr, cYi);
        for (int r = 0; r < H; ++r) { p->colR[r * halfW + k] = cYr[r]; p->colI[r * halfW + k] = cYi[r]; }
    }
    for (int r = 0; r < H; ++r)                                          /* rows (Hermitian) */
        p->irfft(&p->colR[r * halfW], &p->colI[r * halfW], &out[r * W]);
}

/* Normalize rawv by 1/numel and fftshift into outv (lane-wise; mirrors finish()). */
static void finish_b(codelet_plan_b *p) {
    const int H = p->H, W = p->W, numel = p->numel;
    const PIV_VEC mul = piv_vset1(1.0f / (float)numel);
    for (int j = 0; j < numel; ++j) p->rawv[j] = piv_vmul(p->rawv[j], mul);
    for (int row = 0; row < H; ++row) {
        int row_swap = (row + H / 2) % H;
        for (int j = 0; j < W / 2; ++j) {
            p->outv[row * W + W / 2 + j] = p->rawv[row_swap * W + j];
            p->outv[row * W + j]         = p->rawv[row_swap * W + W / 2 + j];
        }
    }
}

/* Scatter outv (PIV_VLANES windows) to packed [LANES][numel] float output. */
static void scatter_out(codelet_plan_b *p, float *out_packed) {
    const int numel = p->numel;
    for (int j = 0; j < numel; ++j) piv_vscatter(out_packed, j, numel, p->outv[j]);
}

void codelet_forward_batch(codelet_plan_b *p, const float *in_packed, int slot) {
    const int numel = p->numel;
    for (int j = 0; j < numel; ++j) p->in[j] = piv_vgather(in_packed, j, numel);
    if (slot == 0) r2c_2d_b(p, p->in, p->FAr, p->FAi);
    else           r2c_2d_b(p, p->in, p->FBr, p->FBi);
}

void codelet_emit_xcorr_batch(codelet_plan_b *p, float *out_packed) {
    const int n = p->numel_fft;
    /* C = spec0 .* conj(spec1) = FA .* conj(FB)  (matches scalar codelet_emit_xcorr) */
    for (int i = 0; i < n; ++i) {
        p->Cr[i] = piv_vadd(piv_vmul(p->FAr[i], p->FBr[i]), piv_vmul(p->FAi[i], p->FBi[i]));
        p->Ci[i] = piv_vsub(piv_vmul(p->FAi[i], p->FBr[i]), piv_vmul(p->FAr[i], p->FBi[i]));
    }
    c2r_2d_b(p, p->Cr, p->Ci, p->rawv);
    finish_b(p);
    scatter_out(p, out_packed);
}

void codelet_emit_power_batch(codelet_plan_b *p, int slot, float *out_packed) {
    const int n = p->numel_fft;
    const PIV_VEC *Fr = (slot == 0) ? p->FAr : p->FBr;
    const PIV_VEC *Fi = (slot == 0) ? p->FAi : p->FBi;
    const PIV_VEC z = piv_vzero();
    for (int i = 0; i < n; ++i) {
        p->Cr[i] = piv_vadd(piv_vmul(Fr[i], Fr[i]), piv_vmul(Fi[i], Fi[i]));
        p->Ci[i] = z;
    }
    c2r_2d_b(p, p->Cr, p->Ci, p->rawv);
    finish_b(p);
    scatter_out(p, out_packed);
}
