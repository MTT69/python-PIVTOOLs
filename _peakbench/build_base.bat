@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
set INC=/I..\pivtools_cli\lib
set BASE=/nologo /O2 /std:c11 /experimental:c11atomics /openmp:experimental /MT
set REPS=%1
if "%REPS%"=="" set REPS=300

echo --- ORIGINAL (pre-Lever-2, libm) SSE2 ---
cl %BASE% %INC% peak_fit_bench_base.c /Fe:base_sse.exe /Fo:base_sse.obj 1>build_base_sse.log 2>&1
if errorlevel 1 (echo BUILD FAILED base_sse & type build_base_sse.log & exit /b 1)
echo --- ORIGINAL (pre-Lever-2, libm) AVX2 ---
cl %BASE% /arch:AVX2 %INC% peak_fit_bench_base.c /Fe:base_avx2.exe /Fo:base_avx2.obj 1>build_base_avx2.log 2>&1
if errorlevel 1 (echo BUILD FAILED base_avx2 & type build_base_avx2.log & exit /b 1)

echo ==================== ORIGINAL  SSE2 (true baseline) ====================
"%~dp0base_sse.exe" %REPS%
echo ==================== ORIGINAL  AVX2 ====================
"%~dp0base_avx2.exe" %REPS%
endlocal
