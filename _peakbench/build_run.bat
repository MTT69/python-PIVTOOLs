@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"

set INC=/I..\pivtools_cli\lib
rem Production MSVC flags (setup.py:121), minus the lib-only /LD.
set BASE=/nologo /O2 /std:c11 /experimental:c11atomics /openmp:experimental /MT
set REPS=%1
if "%REPS%"=="" set REPS=300

echo ############## BUILDING ##############
echo --- poly  SSE2 (shipping default) ---
cl %BASE% %INC% peak_fit_bench.c /Fe:poly_sse.exe /Fo:poly_sse.obj 1>build_poly_sse.log 2>&1
if errorlevel 1 (echo BUILD FAILED poly_sse & type build_poly_sse.log & exit /b 1)
echo --- libm  SSE2 ---
cl %BASE% /DPIV_USE_LIBM_EXP %INC% peak_fit_bench.c /Fe:libm_sse.exe /Fo:libm_sse.obj 1>build_libm_sse.log 2>&1
if errorlevel 1 (echo BUILD FAILED libm_sse & type build_libm_sse.log & exit /b 1)
echo --- poly  AVX2 ---
cl %BASE% /arch:AVX2 %INC% peak_fit_bench.c /Fe:poly_avx2.exe /Fo:poly_avx2.obj 1>build_poly_avx2.log 2>&1
if errorlevel 1 (echo BUILD FAILED poly_avx2 & type build_poly_avx2.log & exit /b 1)
echo --- libm  AVX2 ---
cl %BASE% /arch:AVX2 /DPIV_USE_LIBM_EXP %INC% peak_fit_bench.c /Fe:libm_avx2.exe /Fo:libm_avx2.obj 1>build_libm_avx2.log 2>&1
if errorlevel 1 (echo BUILD FAILED libm_avx2 & type build_libm_avx2.log & exit /b 1)

echo.
echo ############## RUNNING (reps=%REPS%) ##############
echo ==================== poly  SSE2 (SHIPPING) ====================
poly_sse.exe %REPS%
echo ==================== libm  SSE2 ====================
libm_sse.exe %REPS%
echo ==================== poly  AVX2 ====================
poly_avx2.exe %REPS%
echo ==================== libm  AVX2 ====================
libm_avx2.exe %REPS%
endlocal
