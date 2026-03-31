/*
 * bench_xcorr.cu — GPU cross-correlation benchmark via cuFFT.
 *
 * Replicates the xcorr.c pipeline:
 *   zero-pad (centred) → FFT(A), FFT(B) → A·conj(B) → IFFT → normalise → fftshift → extract
 *
 * Exposed as extern "C" so Python can load via ctypes.
 *
 * Build (Windows):
 *   nvcc -O3 -shared -o bench_xcorr.dll cuda/bench_xcorr.cu -lcufft
 *
 * Build (Linux):
 *   nvcc -O3 -shared -Xcompiler -fPIC -o bench_xcorr.so cuda/bench_xcorr.cu -lcufft
 */

#include "common.cuh"
#include <cufft.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── Kernel: zero-pad with centred placement ──────────────────────────── *
 *
 * Matches xcorr.c lines 37-48:
 *   fApad[ (i + N/2)*Npad_w + (j + N/2) ] = fA[ i*N_w + j ]
 *
 * Each thread handles one element of the PADDED array.
 * Threads outside the centred region write zero.
 */
__global__ void zero_pad_centred_kernel(
    const float * __restrict__ src,   /* (N_win, win_h, win_w) */
    float       * __restrict__ dst,   /* (N_win, pad_h, pad_w) */
    int N_win, int win_h, int win_w,
    int pad_h, int pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_win * pad_h * pad_w;
    if (idx >= total) return;

    int n  = idx / (pad_h * pad_w);
    int rem = idx % (pad_h * pad_w);
    int pi = rem / pad_w;
    int pj = rem % pad_w;

    int off_h = win_h / 2;
    int off_w = win_w / 2;

    int si = pi - off_h;
    int sj = pj - off_w;

    if (si >= 0 && si < win_h && sj >= 0 && sj < win_w)
        dst[idx] = src[n * win_h * win_w + si * win_w + sj];
    else
        dst[idx] = 0.0f;
}

/* ── Kernel: element-wise conjugate multiply ──────────────────────────── *
 *
 * C[i] = A[i] * conj(B[i])
 * cuFFT uses interleaved complex: cufftComplex = {float x, float y}
 */
__global__ void multiply_conjugate_kernel(
    const cufftComplex * __restrict__ A,
    const cufftComplex * __restrict__ B,
    cufftComplex       * __restrict__ C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float ar = A[i].x, ai = A[i].y;
    float br = B[i].x, bi = B[i].y;

    C[i].x = ar * br + ai * bi;   /* real part */
    C[i].y = ai * br - ar * bi;   /* imag part */
}

/* ── Kernel: normalise + fftshift + extract central region ────────────── *
 *
 * Combines three steps into one kernel:
 * 1. Divide by numel (pad_h * pad_w) to match FFTW normalisation
 * 2. fftshift: source index = ((pi + pad_h/2) % pad_h, (pj + pad_w/2) % pad_w)
 * 3. Extract: only write if output pixel is within (win_h, win_w)
 *
 * Each thread handles one output element of (N_win, win_h, win_w).
 */
__global__ void normalise_shift_extract_kernel(
    const float * __restrict__ ifft_out,   /* (N_win, pad_h, pad_w) */
    float       * __restrict__ corr_out,   /* (N_win, win_h, win_w) */
    int N_win, int win_h, int win_w,
    int pad_h, int pad_w, float inv_numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_win * win_h * win_w;
    if (idx >= total) return;

    int n  = idx / (win_h * win_w);
    int rem = idx % (win_h * win_w);
    int oi = rem / win_w;
    int oj = rem % win_w;

    /* Map output (oi, oj) → padded index with fftshift + centred extraction.
     * Output pixel (oi, oj) corresponds to padded pixel (oi + win_h/2, oj + win_w/2)
     * after fftshift: source = ((pi + pad_h/2) % pad_h, (pj + pad_w/2) % pad_w) */
    int pi = oi + win_h / 2;
    int pj = oj + win_w / 2;
    int si = (pi + pad_h / 2) % pad_h;
    int sj = (pj + pad_w / 2) % pad_w;

    float val = ifft_out[n * pad_h * pad_w + si * pad_w + sj];
    corr_out[idx] = val * inv_numel;
}

/* ── Main benchmark function ──────────────────────────────────────────── *
 *
 * Performs batched cross-correlation on GPU:
 *   For each of N_windows pairs (A[i], B[i]):
 *     1. Zero-pad to (2*win_h, 2*win_w) with centred placement
 *     2. Forward R2C FFT
 *     3. Conjugate multiply in frequency domain
 *     4. Inverse C2R FFT
 *     5. Normalise, fftshift, extract central (win_h, win_w)
 *
 * Timing results written to out_times[4]:
 *   [0] = total time (ms), including H2D + D2H
 *   [1] = compute-only time (ms), excluding transfers
 *   [2] = H2D transfer time (ms)
 *   [3] = D2H transfer time (ms)
 *
 * Returns 0 on success, -1 on error.
 */
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
int gpu_xcorr_bench(
    const float *h_windowsA,   /* Host: (N_windows, win_h, win_w) */
    const float *h_windowsB,   /* Host: (N_windows, win_h, win_w) */
    float       *h_corrOut,    /* Host: (N_windows, win_h, win_w) output */
    int N_windows,
    int win_h, int win_w,
    float *out_times)          /* Host: [total_ms, compute_ms, h2d_ms, d2h_ms] */
{
    int pad_h = win_h * 2;
    int pad_w = win_w * 2;
    int win_numel   = win_h * win_w;
    int pad_numel   = pad_h * pad_w;
    int fft_numel   = pad_h * (pad_w / 2 + 1);  /* R2C output */
    float inv_numel = 1.0f / (float)pad_numel;

    size_t win_bytes = (size_t)N_windows * win_numel * sizeof(float);
    size_t pad_bytes = (size_t)N_windows * pad_numel * sizeof(float);
    size_t fft_bytes = (size_t)N_windows * fft_numel * sizeof(cufftComplex);

    /* ── Device memory ─────────────────────────────────────────────────── */
    float *d_winA = NULL, *d_winB = NULL;
    float *d_padA = NULL, *d_padB = NULL;
    float *d_ifft = NULL;
    float *d_corrOut = NULL;
    cufftComplex *d_fftA = NULL, *d_fftB = NULL, *d_fftC = NULL;

    CUDA_CHECK(cudaMalloc(&d_winA, win_bytes));
    CUDA_CHECK(cudaMalloc(&d_winB, win_bytes));
    CUDA_CHECK(cudaMalloc(&d_padA, pad_bytes));
    CUDA_CHECK(cudaMalloc(&d_padB, pad_bytes));
    CUDA_CHECK(cudaMalloc(&d_fftA, fft_bytes));
    CUDA_CHECK(cudaMalloc(&d_fftB, fft_bytes));
    CUDA_CHECK(cudaMalloc(&d_fftC, fft_bytes));
    CUDA_CHECK(cudaMalloc(&d_ifft, pad_bytes));
    CUDA_CHECK(cudaMalloc(&d_corrOut, win_bytes));

    /* ── cuFFT plans ───────────────────────────────────────────────────── */
    cufftHandle plan_fwd, plan_inv;
    int n[2] = {pad_h, pad_w};
    int inembed[2] = {pad_h, pad_w};
    int onembed[2] = {pad_h, pad_w / 2 + 1};

    /* Batched R2C forward */
    CUFFT_CHECK(cufftPlanMany(&plan_fwd, 2, n,
                              inembed, 1, pad_numel,       /* input:  (pad_h, pad_w) real */
                              onembed, 1, fft_numel,       /* output: (pad_h, pad_w/2+1) complex */
                              CUFFT_R2C, N_windows));

    /* Batched C2R inverse */
    CUFFT_CHECK(cufftPlanMany(&plan_inv, 2, n,
                              onembed, 1, fft_numel,       /* input:  complex */
                              inembed, 1, pad_numel,       /* output: real */
                              CUFFT_C2R, N_windows));

    /* ── Timers ────────────────────────────────────────────────────────── */
    GpuTimer t_total, t_compute, t_h2d, t_d2h;
    t_total.create(); t_compute.create(); t_h2d.create(); t_d2h.create();

    int threads = 256;

    /* ── Total timer start ─────────────────────────────────────────────── */
    t_total.record_start();

    /* ── H2D transfer ──────────────────────────────────────────────────── */
    t_h2d.record_start();
    CUDA_CHECK(cudaMemcpy(d_winA, h_windowsA, win_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_winB, h_windowsB, win_bytes, cudaMemcpyHostToDevice));
    t_h2d.record_stop();

    /* ── Compute ───────────────────────────────────────────────────────── */
    t_compute.record_start();

    /* 1. Zero-pad with centred placement */
    {
        int total_pad = N_windows * pad_numel;
        int blocks = (total_pad + threads - 1) / threads;
        zero_pad_centred_kernel<<<blocks, threads>>>(d_winA, d_padA, N_windows, win_h, win_w, pad_h, pad_w);
        zero_pad_centred_kernel<<<blocks, threads>>>(d_winB, d_padB, N_windows, win_h, win_w, pad_h, pad_w);
    }
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA sync after zero-pad: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    /* 2. Forward FFT: R2C */
    CUFFT_CHECK(cufftExecR2C(plan_fwd, (cufftReal*)d_padA, d_fftA));
    CUFFT_CHECK(cufftExecR2C(plan_fwd, (cufftReal*)d_padB, d_fftB));
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA sync after forward FFT: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    /* 3. Conjugate multiply: C = A .* conj(B) */
    {
        int total_fft = N_windows * fft_numel;
        int blocks = (total_fft + threads - 1) / threads;
        multiply_conjugate_kernel<<<blocks, threads>>>(d_fftA, d_fftB, d_fftC, total_fft);
    }

    /* 4. Inverse FFT: C2R */
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_fftC, (cufftReal*)d_ifft));
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA sync after inverse FFT: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    /* 5. Normalise + fftshift + extract central region */
    {
        int total_out = N_windows * win_numel;
        int blocks = (total_out + threads - 1) / threads;
        normalise_shift_extract_kernel<<<blocks, threads>>>(
            d_ifft, d_corrOut, N_windows, win_h, win_w, pad_h, pad_w, inv_numel);
    }
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA sync after extract: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    t_compute.record_stop();

    /* ── D2H transfer ──────────────────────────────────────────────────── */
    t_d2h.record_start();
    CUDA_CHECK(cudaMemcpy(h_corrOut, d_corrOut, win_bytes, cudaMemcpyDeviceToHost));
    t_d2h.record_stop();

    t_total.record_stop();

    /* ── Report times ──────────────────────────────────────────────────── */
    out_times[0] = t_total.elapsed_ms();
    out_times[1] = t_compute.elapsed_ms();
    out_times[2] = t_h2d.elapsed_ms();
    out_times[3] = t_d2h.elapsed_ms();

    /* ── Cleanup ───────────────────────────────────────────────────────── */
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_winA); cudaFree(d_winB);
    cudaFree(d_padA); cudaFree(d_padB);
    cudaFree(d_fftA); cudaFree(d_fftB); cudaFree(d_fftC);
    cudaFree(d_ifft); cudaFree(d_corrOut);
    t_total.destroy(); t_compute.destroy(); t_h2d.destroy(); t_d2h.destroy();

    return 0;
}

/* ── Convenience: single-window xcorr for correctness testing ─────────── */
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
int gpu_xcorr_single(
    const float *h_A,      /* Host: (win_h, win_w) */
    const float *h_B,      /* Host: (win_h, win_w) */
    float       *h_C,      /* Host: (win_h, win_w) output */
    int win_h, int win_w)
{
    float times[4];
    return gpu_xcorr_bench(h_A, h_B, h_C, 1, win_h, win_w, times);
}
