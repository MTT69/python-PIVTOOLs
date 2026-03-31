/*
 * bench_warp.cu -- GPU fused symmetric warp benchmark.
 *
 * Replicates the FULL fused_warp.c pipeline:
 *   1. Build 1D predictor index LUTs (pixel -> fractional predictor index)
 *   2. Per pixel: bicubic predictor interpolation (BORDER_REPLICATE)
 *   3. Symmetric warp coordinates (+/-displacement/2)
 *   4. Bicubic image sample (Keys a=-0.75, BORDER_CONSTANT=0)
 *
 * Build (Windows):
 *   nvcc -O3 --shared -arch=sm_89 -o bench_warp.dll cuda/bench_warp.cu
 */

#include "common.cuh"
#include <math.h>
#include <stdlib.h>

/* ======================================================================= */
/* Device helpers                                                           */
/* ======================================================================= */

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ void keys_weights_4(float d, float w[4])
{
    float s0 = 1.0f + d;
    float q2 = 1.0f - d;
    float s3 = 2.0f - d;
    w[1] = (1.25f * d  - 2.25f) * d  * d  + 1.0f;
    w[2] = (1.25f * q2 - 2.25f) * q2 * q2 + 1.0f;
    w[0] = ((-0.75f * s0 + 3.75f) * s0 - 6.0f) * s0 + 3.0f;
    w[3] = ((-0.75f * s3 + 3.75f) * s3 - 6.0f) * s3 + 3.0f;
}

/* ======================================================================= */
/* Kernel                                                                   */
/* ======================================================================= */

/* Bicubic image sample, BORDER_CONSTANT=0 -- matches fused_warp.c */
__device__ __forceinline__ float bicubic_sample(
    const float * __restrict__ img, float fy, float fx, int H, int W)
{
    float fy_floor = floorf(fy);
    float fx_floor = floorf(fx);
    int iy = (int)fy_floor - 1;
    int ix = (int)fx_floor - 1;
    float dy = fy - fy_floor;
    float dx = fx - fx_floor;
    float wy[4], wx[4];
    keys_weights_4(dy, wy);
    keys_weights_4(dx, wx);

    float val = 0.0f;
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        int row = iy + m;
        if (row < 0 || row >= H) continue;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int col = ix + n;
            if (col < 0 || col >= W) continue;
            val += wy[m] * wx[n] * img[row * W + col];
        }
    }
    return val;
}

/* Predictor interpolation with precomputed y-weights, BORDER_REPLICATE */
__device__ __forceinline__ float bicubic_pred_wy(
    const float * __restrict__ pred, const float *wy, int iy_base,
    float fx, int nPY, int nPX)
{
    float fx_floor = floorf(fx);
    int ix = (int)fx_floor - 1;
    float ddx = fx - fx_floor;
    float wx[4];
    keys_weights_4(ddx, wx);

    float val = 0.0f;
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        int row = clampi(iy_base + m, 0, nPY - 1);
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int col = clampi(ix + n, 0, nPX - 1);
            val += wy[m] * wx[n] * pred[row * nPX + col];
        }
    }
    return val;
}

/* Full fused warp kernel: predictor upsample + symmetric warp + bicubic */
__global__ void fused_warp_kernel(
    const float * __restrict__ imgs_a,
    const float * __restrict__ imgs_b,
    float       * __restrict__ outs_a,
    float       * __restrict__ outs_b,
    const float * __restrict__ pred_dy,
    const float * __restrict__ pred_dx,
    const float * __restrict__ pred_idx_y,
    const float * __restrict__ pred_idx_x,
    int N, int H, int W, int nPY, int nPX)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n   = idx / (H * W);
    int rem = idx % (H * W);
    int i   = rem / W;
    int j   = rem % W;

    /* Phase A: predictor interpolation */
    float fiy = pred_idx_y[i];
    float fiy_floor = floorf(fiy);
    int pred_iy_base = (int)fiy_floor - 1;
    float pred_wy_arr[4];
    keys_weights_4(fiy - fiy_floor, pred_wy_arr);

    float fix = pred_idx_x[j];
    float dense_dy = bicubic_pred_wy(pred_dy, pred_wy_arr, pred_iy_base, fix, nPY, nPX);
    float dense_dx = bicubic_pred_wy(pred_dx, pred_wy_arr, pred_iy_base, fix, nPY, nPX);

    /* Phase B: symmetric warp */
    float half_dy = 0.5f * dense_dy;
    float half_dx = 0.5f * dense_dx;

    /* Phase C: bicubic image sample */
    const float *cur_a = imgs_a + (size_t)n * H * W;
    const float *cur_b = imgs_b + (size_t)n * H * W;
    int out_idx = (size_t)n * H * W + i * W + j;

    outs_a[out_idx] = bicubic_sample(cur_a, (float)i - half_dy, (float)j - half_dx, H, W);
    outs_b[out_idx] = bicubic_sample(cur_b, (float)i + half_dy, (float)j + half_dx, H, W);
}


/* ======================================================================= */
/* Host helpers                                                             */
/* ======================================================================= */

static void build_pred_index_lut(float *lut, int N, const float *ctrs, int nC)
{
    if (nC == 1) { for (int p = 0; p < N; p++) lut[p] = 0.0f; return; }
    int seg = 0;
    for (int p = 0; p < N; p++) {
        float coord = (float)p;
        if (coord <= ctrs[0])       { lut[p] = 0.0f; continue; }
        if (coord >= ctrs[nC - 1])  { lut[p] = (float)(nC - 1); continue; }
        while (seg < nC - 2 && ctrs[seg + 1] < coord) seg++;
        float denom = ctrs[seg + 1] - ctrs[seg];
        if (denom < 1e-12f)         { lut[p] = (float)seg; continue; }
        lut[p] = (float)seg + (coord - ctrs[seg]) / denom;
    }
}


/* ======================================================================= */
/* Exported: single-call benchmark -- used for correctness checking         */
/* ======================================================================= */

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
int gpu_fused_warp_bench(
    const float *h_imgs_a, const float *h_imgs_b,
    float *h_outs_a, float *h_outs_b,
    const float *h_pred_dy, const float *h_pred_dx,
    const float *h_ctrs_y, const float *h_ctrs_x,
    int N, int H, int W, int nPY, int nPX,
    float *out_times)
{
    size_t img_bytes  = (size_t)N * H * W * sizeof(float);
    size_t pred_bytes = (size_t)nPY * nPX * sizeof(float);

    float *h_piy = (float*)malloc(H * sizeof(float));
    float *h_pix = (float*)malloc(W * sizeof(float));
    build_pred_index_lut(h_piy, H, h_ctrs_y, nPY);
    build_pred_index_lut(h_pix, W, h_ctrs_x, nPX);

    float *d_ia=NULL,*d_ib=NULL,*d_oa=NULL,*d_ob=NULL;
    float *d_pdy=NULL,*d_pdx=NULL,*d_piy=NULL,*d_pix=NULL;
    CUDA_CHECK(cudaMalloc(&d_ia, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_ib, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_oa, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_ob, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_pdy, pred_bytes));
    CUDA_CHECK(cudaMalloc(&d_pdx, pred_bytes));
    CUDA_CHECK(cudaMalloc(&d_piy, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pix, W*sizeof(float)));

    GpuTimer tt,tc,th,td;
    tt.create(); tc.create(); th.create(); td.create();
    tt.record_start();
    th.record_start();
    CUDA_CHECK(cudaMemcpy(d_ia,h_imgs_a,img_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ib,h_imgs_b,img_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdy,h_pred_dy,pred_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdx,h_pred_dx,pred_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_piy,h_piy,H*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pix,h_pix,W*sizeof(float),cudaMemcpyHostToDevice));
    th.record_stop();

    int threads=256, total=N*H*W, blocks=(total+threads-1)/threads;
    tc.record_start();
    fused_warp_kernel<<<blocks,threads>>>(d_ia,d_ib,d_oa,d_ob,d_pdy,d_pdx,d_piy,d_pix,N,H,W,nPY,nPX);
    tc.record_stop();

    td.record_start();
    CUDA_CHECK(cudaMemcpy(h_outs_a,d_oa,img_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outs_b,d_ob,img_bytes,cudaMemcpyDeviceToHost));
    td.record_stop();
    tt.record_stop();

    out_times[0]=tt.elapsed_ms(); out_times[1]=tc.elapsed_ms();
    out_times[2]=th.elapsed_ms(); out_times[3]=td.elapsed_ms();

    free(h_piy); free(h_pix);
    cudaFree(d_ia);cudaFree(d_ib);cudaFree(d_oa);cudaFree(d_ob);
    cudaFree(d_pdy);cudaFree(d_pdx);cudaFree(d_piy);cudaFree(d_pix);
    tt.destroy();tc.destroy();th.destroy();td.destroy();
    return 0;
}


/* ======================================================================= */
/* Exported: sustained benchmark -- tight kernel loop, no Python overhead   */
/*                                                                          */
/* out_results[0] = total compute ms                                        */
/* out_results[1] = avg per iteration ms                                    */
/* out_results[2] = n_iterations                                            */
/* ======================================================================= */

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
int gpu_fused_warp_sustained(
    const float *h_imgs_a, const float *h_imgs_b,
    const float *h_pred_dy, const float *h_pred_dx,
    const float *h_ctrs_y, const float *h_ctrs_x,
    int N, int H, int W, int nPY, int nPX,
    int n_iterations,
    float *out_results)
{
    size_t img_bytes  = (size_t)N * H * W * sizeof(float);
    size_t pred_bytes = (size_t)nPY * nPX * sizeof(float);

    float *h_piy = (float*)malloc(H * sizeof(float));
    float *h_pix = (float*)malloc(W * sizeof(float));
    build_pred_index_lut(h_piy, H, h_ctrs_y, nPY);
    build_pred_index_lut(h_pix, W, h_ctrs_x, nPX);

    float *d_ia=NULL,*d_ib=NULL,*d_oa=NULL,*d_ob=NULL;
    float *d_pdy=NULL,*d_pdx=NULL,*d_piy=NULL,*d_pix=NULL;
    CUDA_CHECK(cudaMalloc(&d_ia, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_ib, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_oa, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_ob, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_pdy, pred_bytes));
    CUDA_CHECK(cudaMalloc(&d_pdx, pred_bytes));
    CUDA_CHECK(cudaMalloc(&d_piy, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pix, W*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ia,h_imgs_a,img_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ib,h_imgs_b,img_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdy,h_pred_dy,pred_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdx,h_pred_dx,pred_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_piy,h_piy,H*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pix,h_pix,W*sizeof(float),cudaMemcpyHostToDevice));

    int threads = 256;
    int total = N * H * W;
    int blocks = (total + threads - 1) / threads;

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        fused_warp_kernel<<<blocks, threads>>>(
            d_ia,d_ib,d_oa,d_ob,d_pdy,d_pdx,d_piy,d_pix,N,H,W,nPY,nPX);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed loop */
    GpuTimer timer;
    timer.create();
    timer.record_start();
    for (int i = 0; i < n_iterations; i++) {
        fused_warp_kernel<<<blocks, threads>>>(
            d_ia,d_ib,d_oa,d_ob,d_pdy,d_pdx,d_piy,d_pix,N,H,W,nPY,nPX);
    }
    timer.record_stop();
    float total_ms = timer.elapsed_ms();

    out_results[0] = total_ms;
    out_results[1] = total_ms / (float)n_iterations;
    out_results[2] = (float)n_iterations;

    timer.destroy();
    free(h_piy); free(h_pix);
    cudaFree(d_ia);cudaFree(d_ib);cudaFree(d_oa);cudaFree(d_ob);
    cudaFree(d_pdy);cudaFree(d_pdx);cudaFree(d_piy);cudaFree(d_pix);

    return 0;
}
