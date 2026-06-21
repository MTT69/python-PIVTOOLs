/****************************************************************************
 * codelet_fft.h -- permissive (BSD-clean) fixed-size 2D FFT for the PIV
 * cross-correlation hot path. Replaces GPL FFTW in libbulkxcorr2d.
 *
 * A 2D real->complex FFT is assembled separably from generated, fully-unrolled
 * 1D codelets (codelets_gen.h): rfft along the W-length rows, cfft along the
 * H-length columns. The transform is FFTW-unnormalized in the canonical
 * [H][W/2+1] complex layout, so the correlation math (conjugate-multiply,
 * power spectrum, 1/numel normalize + fftshift) is identical to the previous
 * FFTW path -- only the transform engine changed.
 *
 * Supported axis lengths (BUILT_FFT_SIZES): 8 12 16 24 32 48 64 96 128.
 * Each is 2^k or 2^k*3, which the mixed-radix codelet generator handles.
 *
 * Stage A: one window per call (scalar codelets). Correct and portable;
 * slower than FFTW. Kept as the remainder ("tail") fallback below.
 *
 * Stage B (codelet_plan_b): processes LANES windows per call, one PIV window
 * per SIMD lane (NEON-4 / AVX2-8 / AVX-512-16, chosen at compile time -- see
 * codelet_simd.h). Each lane runs the identical scalar computation, so the
 * batched result equals the scalar result per lane within float tolerance.
 ****************************************************************************/
#ifndef CODELET_FFT_H
#define CODELET_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Interleaved complex [re, im]; bit-compatible with the old fftwf_complex
 * (which was also `typedef float fftwf_complex[2]`), so buffer layouts and
 * the conjugate-multiply / power kernels are unchanged. */
typedef float codelet_cplx[2];

/* Opaque per-(H,W) plan: owns the codelet bindings and all scratch. Not
 * thread-safe to share; create one per OMP thread (as the old sPlan was). */
typedef struct codelet_plan codelet_plan;

/* True iff n is a built codelet size. */
int codelet_size_ok(int n);

/* Create a plan for an H-rows x W-cols real window. Returns NULL if either
 * axis is not a built size, or on allocation failure. */
codelet_plan *codelet_plan_create(int H, int W);
void          codelet_plan_destroy(codelet_plan *p);

/* numel = H*W ; numel_fft = H*(W/2+1). */
int codelet_plan_numel(const codelet_plan *p);
int codelet_plan_numel_fft(const codelet_plan *p);

/* Forward 2D r2c of a real window `in` (length numel) into internal spectrum
 * slot 0 or 1. Mirrors loading one of FFTW's batched (howmany=2) transforms. */
void codelet_forward(codelet_plan *p, const float *in, int slot);

/* Cross-correlation surface from the two loaded spectra:
 *   out = fftshift( IFFT( spec0 .* conj(spec1) ) ) / numel
 * (slot0 plays the role of `a`, slot1 of `b`, matching xcorr_preplanned). */
void codelet_emit_xcorr(codelet_plan *p, float *out);

/* Auto-correlation surface from one loaded spectrum:
 *   out = fftshift( IFFT( |spec[slot]|^2 ) ) / numel */
void codelet_emit_power(codelet_plan *p, int slot, float *out);

/* ====================================================================== *
 *  Stage B -- SIMD-lane-batched engine (LANES windows per call).
 *
 *  Packed buffers are window-major [LANES][numel] float: window j occupies
 *  in/out[j*numel .. j*numel+numel). The gather/scatter inside the engine
 *  are the ONLY thing that touches that layout (scalar element-wise loads/
 *  stores), so the caller's packed buffers need no special alignment.
 * ====================================================================== */

/* Number of windows processed per batched call (compile-time SIMD width). */
int codelet_lanes(void);

/* Opaque batched plan; one per OMP thread, like codelet_plan. */
typedef struct codelet_plan_b codelet_plan_b;

codelet_plan_b *codelet_plan_create_batched(int H, int W);
void            codelet_plan_destroy_batched(codelet_plan_b *p);

/* Forward 2D r2c of LANES real windows `in_packed` ([LANES][numel]) into
 * internal spectrum slot 0 or 1. */
void codelet_forward_batch(codelet_plan_b *p, const float *in_packed, int slot);

/* Cross-correlation surfaces for all LANES windows from the two loaded
 * spectra: out_packed[j] = fftshift( IFFT( spec0[j] .* conj(spec1[j]) ) )/numel.
 * out_packed is [LANES][numel]. (slot0 = `a`, slot1 = `b`.) */
void codelet_emit_xcorr_batch(codelet_plan_b *p, float *out_packed);

/* Auto-correlation surfaces for all LANES windows from one loaded spectrum:
 * out_packed[j] = fftshift( IFFT( |spec[slot][j]|^2 ) ) / numel. */
void codelet_emit_power_batch(codelet_plan_b *p, int slot, float *out_packed);

#ifdef __cplusplus
}
#endif

#endif /* CODELET_FFT_H */
