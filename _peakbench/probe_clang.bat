@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
where clang-cl
echo === version ===
clang-cl --version
echo === probe: O2 AVX2 + fopenmp-simd, forced _OPENMP, vec remarks on the poly build ===
clang-cl /nologo /c /O2 /arch:AVX2 /std:c11 -fopenmp-simd -D_OPENMP=201511 ^
  /clang:-Rpass=loop-vectorize /clang:-Rpass-missed=loop-vectorize ^
  /I..\pivtools_cli\lib peak_fit_bench.c /Fo:probe_clang.obj 2>probe_clang.log
echo exit=%errorlevel%
echo --- residual-loop remarks (peak_locate_lm.c lines 114/118/187/191/259/263) ---
findstr /C:"peak_locate_lm.c" probe_clang.log | findstr /C:"vectorize"
endlocal
