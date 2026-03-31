#!/bin/bash
# Build GPU benchmark libraries (Linux / HPC)
# Requires: CUDA Toolkit with nvcc on PATH (e.g. module load cuda)

set -e

echo "Building bench_xcorr.so ..."
nvcc -O3 -shared -Xcompiler -fPIC -o bench_xcorr.so cuda/bench_xcorr.cu -lcufft
echo "  OK"

echo "Building bench_warp.so ..."
nvcc -O3 -shared -Xcompiler -fPIC -o bench_warp.so cuda/bench_warp.cu
echo "  OK"

echo ""
echo "Build complete. Run benchmarks with:"
echo "  python python/bench_fft.py"
echo "  python python/bench_interpolation.py"
