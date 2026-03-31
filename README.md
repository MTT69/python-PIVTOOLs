# GPU Benchmarks for PIVTOOLs

Standalone GPU benchmarks comparing cuFFT + custom CUDA kernels against the existing CPU C libraries (FFTW3f + fused_warp).

## Prerequisites

- NVIDIA GPU (tested on RTX 4060, A100, H100)
- CUDA Toolkit 12.6+ (`nvcc --version` to verify)
- Python 3.12+ with numpy, scipy

## Build

**Windows:**
```
build.bat
```

**Linux / HPC:**
```
chmod +x build.sh
./build.sh
```

This produces `bench_xcorr.dll/.so` and `bench_warp.dll/.so` in the project root.

## Run

```
python python/bench_fft.py
python python/bench_interpolation.py
```

### Options

```
python python/bench_fft.py --windows 32,64 --counts 1024,4096 --csv results.csv
python python/bench_interpolation.py --sizes 1024,2048 --batches 1,5,10 --csv results.csv
```

Both scripts:
- Verify numerical correctness before timing
- Report median of 10 timed runs after 3 warmup runs
- Show GPU total time (including PCIe transfer) and compute-only time separately
- Optionally compare against the C libraries if they're found in the main codebase

## What's Measured

### FFT Cross-Correlation (`bench_fft.py`)
Batched 2D cross-correlation matching `xcorr.c`:
zero-pad (centred) → R2C FFT → conjugate multiply → C2R IFFT → normalise → fftshift → extract

### Bicubic Interpolation (`bench_interpolation.py`)
Symmetric image warping matching `fused_warp.c`:
Keys a=-0.75 bicubic → 4x4 stencil → BORDER_CONSTANT=0
