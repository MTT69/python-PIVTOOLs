#ifndef FUSED_WARP_H
#define FUSED_WARP_H

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Implementation selector — pick the warp inner-loop path at runtime.
 *   0 = scalar reference (the always-correct oracle)
 *   1 = SIMD (default)
 * One built library can run either, so tests can assert scalar==SIMD equivalence
 * and benchmarks can A/B them without a rebuild.
 */
EXPORT void fused_warp_set_impl(int impl);
EXPORT int  fused_warp_get_impl(void);

/*
 * Fused symmetric image warp for predictor-corrector PIV.
 *
 * Combines three operations into a single pass:
 *   1. Upsample coarse predictor field (nPY x nPX) to dense (H x W) via bicubic interpolation
 *   2. Build symmetric coordinate maps: map_A = pixel - delta/2, map_B = pixel + delta/2
 *   3. Warp both input images using bicubic or Lanczos-3 sampling
 *
 * Parameters:
 *   img_a, img_b   - Input images, row-major float32, size H*W each
 *   out_a, out_b   - Output warped images, row-major float32, size H*W each (caller-allocated)
 *   pred_dy         - Coarse predictor y-displacement, row-major, size nPY*nPX
 *   pred_dx         - Coarse predictor x-displacement, row-major, size nPY*nPX
 *   H, W            - Image dimensions (height, width)
 *   nPY, nPX        - Predictor grid dimensions
 *   ctrs_y          - Window centre y-coordinates in pixel space, size nPY
 *   ctrs_x          - Window centre x-coordinates in pixel space, size nPX
 *   interp_mode     - 0 = bicubic (Keys a=-0.75, 4x4 stencil, matches cv2.INTER_CUBIC)
 *                      1 = Lanczos-3 (windowed sinc, 6x6 stencil)
 *
 * Returns:
 *   0 = success
 *   1 = memory allocation failure
 *
 * Border handling:
 *   - Predictor upsampling: clamp (replicate) at grid boundaries
 *   - Image warping: constant zero outside [0, H-1] x [0, W-1]
 *
 * Thread safety:
 *   Uses OpenMP for row-wise parallelism. Each row is fully independent.
 */
EXPORT int fused_symmetric_warp(
    const float *img_a, const float *img_b,
    float *out_a, float *out_b,
    const float *pred_dy, const float *pred_dx,
    int H, int W,
    int nPY, int nPX,
    const float *ctrs_y, const float *ctrs_x,
    int interp_mode,
    int round_shifts           /* 1: round half-shifts to integer (pure pixel shift) */
);

/*
 * Batch version: warp N image pairs.
 *
 * Images are stacked as (N, H, W) in row-major order.
 * Predictor can be shared (ensemble) or per-image (instantaneous):
 *   shared_predictor=1: pred_dy/dx are (nPY, nPX) — same for all images
 *   shared_predictor=0: pred_dy/dx are (N, nPY, nPX) — separate per image
 *
 * OpenMP parallelizes over a manually flattened (image, row) index (total_rows =
 * N*H). collapse(2) is avoided because it crashes on MSVC /openmp:experimental at
 * large iteration counts.
 */
EXPORT int fused_symmetric_warp_batch(
    const float *imgs_a,       /* (N, H, W) stacked */
    const float *imgs_b,       /* (N, H, W) stacked */
    float       *outs_a,       /* (N, H, W) stacked */
    float       *outs_b,       /* (N, H, W) stacked */
    const float *pred_dy,      /* (nPY, nPX) if shared, (N, nPY, nPX) if per-image */
    const float *pred_dx,      /* same */
    int N,
    int H, int W,
    int nPY, int nPX,
    const float *ctrs_y,
    const float *ctrs_x,
    int interp_mode,
    int shared_predictor,      /* 1=shared (ensemble), 0=per-image (instantaneous) */
    int round_shifts           /* 1: round half-shifts to integer (pure pixel shift) */
);

#ifdef __cplusplus
}
#endif

#endif /* FUSED_WARP_H */
