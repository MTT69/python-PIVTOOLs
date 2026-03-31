@echo off
REM Build GPU benchmark libraries (Windows)
REM Requires: CUDA Toolkit with nvcc on PATH

setlocal

set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"
set CUDA_INC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include"
set CUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\lib\x64"

echo Building bench_xcorr.dll ...
%NVCC% -O3 --shared -arch=sm_89 -o bench_xcorr.dll cuda/bench_xcorr.cu -I%CUDA_INC% -L%CUDA_LIB% -lcufft
if %errorlevel% neq 0 (
    echo FAILED: bench_xcorr.dll
    exit /b 1
)
echo   OK

echo Building bench_warp.dll ...
%NVCC% -O3 --shared -arch=sm_89 -o bench_warp.dll cuda/bench_warp.cu -I%CUDA_INC%
if %errorlevel% neq 0 (
    echo FAILED: bench_warp.dll
    exit /b 1
)
echo   OK

echo.
echo Build complete. Run benchmarks with:
echo   python python/bench_fft.py
echo   python python/bench_interpolation.py
