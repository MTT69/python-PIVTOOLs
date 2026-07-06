@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
set INC=/I..\pivtools_cli\lib
set BASE=/nologo /c /O2 /std:c11 /experimental:c11atomics /Qvec-report:2

rem 1) poly, omp ON, AVX2  (== SHIPPING codegen path for the residual loop)
cl %BASE% /openmp:experimental /arch:AVX2 %INC% peak_fit_bench.c /Fo:vec_poly_omp_avx2.obj 1>vec_poly_omp_avx2.log 2>&1
rem 2) poly, omp OFF, AVX2  (native auto-vectorizer only; pragma compiles out)
cl %BASE% /arch:AVX2 %INC% peak_fit_bench.c /Fo:vec_poly_noomp_avx2.obj 1>vec_poly_noomp_avx2.log 2>&1
rem 3) libm, omp ON, AVX2  (expf call in the loop)
cl %BASE% /openmp:experimental /arch:AVX2 /DPIV_USE_LIBM_EXP %INC% peak_fit_bench.c /Fo:vec_libm_avx2.obj 1>vec_libm_avx2.log 2>&1
echo done
endlocal
