#ifndef BENCH_COMMON_CUH
#define BENCH_COMMON_CUH

#include <stdio.h>
#include <cuda_runtime.h>

/* ── Error checking macros ─────────────────────────────────────────────── */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CUFFT_CHECK(call)                                                   \
    do {                                                                    \
        cufftResult err = (call);                                           \
        if (err != CUFFT_SUCCESS) {                                         \
            fprintf(stderr, "cuFFT error at %s:%d: code %d\n",            \
                    __FILE__, __LINE__, (int)err);                          \
            return -1;                                                      \
        }                                                                   \
    } while (0)

/* ── Timing helpers ────────────────────────────────────────────────────── */

struct GpuTimer {
    cudaEvent_t start, stop;

    void create() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    void destroy() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void record_start(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    void record_stop(cudaStream_t s = 0)  { cudaEventRecord(stop, s); }
    float elapsed_ms() {
        float ms;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

#endif /* BENCH_COMMON_CUH */
