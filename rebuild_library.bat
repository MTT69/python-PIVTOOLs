@echo off
REM Build script for PIV C library with updated coordinate system
REM Run this from the python-PIVTOOLs root directory

echo ========================================
echo Building PIV C Library
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "pypivtools\lib" (
    echo ERROR: pypivtools\lib directory not found!
    echo Please run this script from the python-PIVTOOLs root directory.
    pause
    exit /b 1
)

echo Step 1: Running Python setup script...
echo.
python setup_windows.py build_ext --inplace

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo Please check the error messages above.
    echo Common issues:
    echo   - Missing compiler (need MSVC or MinGW)
    echo   - Missing FFTW library
    echo   - Missing Python development headers
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo The C library has been updated with:
echo   - Row-major (C-contiguous) indexing
echo   - Fixed coordinate system (X=width, Y=height)
echo   - Corrected window center calculations
echo   - Improved mask handling
echo.
echo Next steps:
echo   1. Test with your PIV data
echo   2. Verify results match expected values
echo   3. Check COORDINATE_SYSTEM_FIX.md for details
echo.
pause
