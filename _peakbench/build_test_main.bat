@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\pivtools_fullstack\python-PIVTOOLs"
set PY=env\Scripts\python.exe
echo ================== git state of main worktree ==================
git status --short --branch
echo ================== setup.py build (clang-cl) ==================
"%PY%" setup.py build 2>&1
echo ================== built DLLs ==================
dir /b pivtools_cli\lib\*.dll
echo ================== peak-fit GT gate ==================
"%PY%" -m pytest unit-tests\test_instantaneous_peaks.py -q
