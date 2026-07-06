@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0\.."
set PY=..\..\python-PIVTOOLs\env\Scripts\python.exe
echo === MSVC (cl) full setup.py build ===
"%PY%" setup.py build
echo === built DLLs ===
dir /b pivtools_cli\lib\*.dll
