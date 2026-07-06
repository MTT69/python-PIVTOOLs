@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0\.."
set PY=..\..\python-PIVTOOLs\env\Scripts\python.exe
set PIVTOOLS_WIN_COMPILER=clang-cl
echo === clang-cl full setup.py build ===
"%PY%" setup.py build 2>&1
echo === stage libomp.dll next to the libs (runtime dep of /openmp) ===
for /f "delims=" %%i in ('where clang-cl') do set CLANGDIR=%%~dpi
echo clang dir: %CLANGDIR%
if exist "%CLANGDIR%libomp.dll" copy /y "%CLANGDIR%libomp.dll" pivtools_cli\lib\ >nul && echo copied libomp.dll
echo === built artifacts ===
dir /b pivtools_cli\lib\*.dll
