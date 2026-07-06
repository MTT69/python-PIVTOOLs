@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
set CC=clang-cl
set INC=/I..\pivtools_cli\lib
set B=/nologo /O2 /std:c11
rem poly needs the omp-simd pragma honored (-fopenmp-simd) + _OPENMP so the macro emits it.
set OMP=-fopenmp-simd -D_OPENMP=201511
set REPS=%1
if "%REPS%"=="" set REPS=300

echo ############## BUILDING (clang-cl 19) ##############
%CC% %B%            %INC% peak_fit_bench_base.c /Fe:cl_base_sse.exe  /Fo:cl_base_sse.obj   1>bc1.log 2>&1 || (echo FAIL base_sse & type bc1.log & exit /b 1)
%CC% %B% /arch:AVX2 %INC% peak_fit_bench_base.c /Fe:cl_base_avx2.exe /Fo:cl_base_avx2.obj  1>bc2.log 2>&1 || (echo FAIL base_avx2 & type bc2.log & exit /b 1)
%CC% %B%            /DPIV_USE_LIBM_EXP %INC% peak_fit_bench.c /Fe:cl_libm_sse.exe  /Fo:cl_libm_sse.obj  1>bc3.log 2>&1 || (echo FAIL libm_sse & type bc3.log & exit /b 1)
%CC% %B% /arch:AVX2 /DPIV_USE_LIBM_EXP %INC% peak_fit_bench.c /Fe:cl_libm_avx2.exe /Fo:cl_libm_avx2.obj 1>bc4.log 2>&1 || (echo FAIL libm_avx2 & type bc4.log & exit /b 1)
%CC% %B%            %OMP% %INC% peak_fit_bench.c /Fe:cl_poly_sse.exe  /Fo:cl_poly_sse.obj  1>bc5.log 2>&1 || (echo FAIL poly_sse & type bc5.log & exit /b 1)
%CC% %B% /arch:AVX2 %OMP% %INC% peak_fit_bench.c /Fe:cl_poly_avx2.exe /Fo:cl_poly_avx2.obj 1>bc6.log 2>&1 || (echo FAIL poly_avx2 & type bc6.log & exit /b 1)

echo.
echo ############## RUNNING (reps=%REPS%) ##############
echo ==================== clang ORIGINAL  SSE2 ====================
"%~dp0cl_base_sse.exe" %REPS%
echo ==================== clang ORIGINAL  AVX2 ====================
"%~dp0cl_base_avx2.exe" %REPS%
echo ==================== clang libm+L2   SSE2 ====================
"%~dp0cl_libm_sse.exe" %REPS%
echo ==================== clang libm+L2   AVX2 ====================
"%~dp0cl_libm_avx2.exe" %REPS%
echo ==================== clang poly+L2   SSE2 ====================
"%~dp0cl_poly_sse.exe" %REPS%
echo ==================== clang poly+L2   AVX2 ====================
"%~dp0cl_poly_avx2.exe" %REPS%
endlocal
