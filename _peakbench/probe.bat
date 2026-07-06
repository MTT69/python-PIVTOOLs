@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl 2>&1 | findstr /C:"Version"
echo === Qvec-report probe ===
cl /nologo /O2 /Qvec-report:2 probe.c 2>&1
