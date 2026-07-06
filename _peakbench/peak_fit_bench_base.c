/* Baseline bench: original pre-Lever-2 kernel (8947092), plain libm expf,
 * no pred cache, no omp simd. Same harness as peak_fit_bench.c. */
#define PIV_BENCH_PEAK_SRC "peak_locate_lm_orig.c"
#include "peak_fit_bench.c"
