# PyPIVTools Installation Guide

PyPIVTools requires compilation of C libraries with OpenMP support and links against FFTW libraries. This guide provides detailed, production-ready installation instructions for each supported operating system.

## Prerequisites

Before installation, ensure you have:
- Python 3.8 or later
- A C compiler with OpenMP support
- FFTW library (version 3.x)
- FFmpeg (for video processing)
- Git (for cloning the repository if needed)

## Quick Start

1. Clone or download the repository
2. Choose the appropriate setup file for your operating system:
   - Windows: `setup_windows.py`
   - macOS: `setup_macos.py`
   - Linux: `setup_linux.py`
3. Follow the detailed instructions below for your OS
4. Run `pip install -e .` (using the appropriate setup file)

## Windows Installation

### Visual Studio Build Tools

#### 1. Install Visual Studio
- Download and install Visual Studio 2019 or later from https://visualstudio.microsoft.com/
- During installation, select the "Desktop development with C++" workload
- Ensure "MSVC v142+ build tools" are included
- Install the latest Windows SDK

#### 2. Verify Visual Studio Installation


Alternatively, to run the Developer Command Prompt from the VS Code terminal:
- Open VS Code and press `Ctrl+Shift+P` to open the command palette
- Type and select "Developer Command Prompt for VS" (requires the C++ extension installed)
- In the terminal that opens, run:
```cmd
cl
```
Should show: Microsoft (R) C/C++ Optimizing Compiler Version X.X.X

#### 3. Install FFTW Library
Using vcpkg (recommended package manager for Windows):

**Important**: 
- Run Developer Command Prompt as  administrator for vcpkg installation
- Navigate to a directory where you have write permissions (e.g., your Documents or home directory)
- Do NOT clone into Program Files or other system directories

** Install vcpkg **
```cmd
# First navigate to a writable directory
cd %USERPROFILE% (you decide USERPROFILE)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg install fftw3[threads]
```

#### 3.5 Install FFmpeg

Using vcpkg:

```cmd
vcpkg install ffmpeg
```

#### 3.6 Add FFmpeg to PATH

To ensure FFmpeg is available from the command line, add the vcpkg bin directory to your PATH:

```cmd
setx PATH "%PATH%;C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\vcpkg\installed\x64-windows\bin"
```

Note: Adjust the path if your vcpkg installation is in a different location.

#### 3.7 Verify FFmpeg Installation

```cmd
ffmpeg -version
```

Should show FFmpeg version information.

#### 4. Set Environment Variables

For global vcpkg installation: (set user profile to where you installed the packages)
```cmd
setx FFTW_INC_PATH "C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\vcpkg\installed\x64-windows\include"
setx FFTW_LIB_PATH "C:\Users\mtt1e23\OneDrive - University of Southampton\Documents\vcpkg\installed\x64-windows\lib"
``

#### 6. Verify FFTW Installation
In Developer Command Prompt for VS:
```cmd
dir %FFTW_INC_PATH%\fftw3.h
dir %FFTW_LIB_PATH%\fftw3f.lib
```

It's recommended to install Python packages like PyPIVTools in a virtual environment to avoid conflicts with system packages. If you prefer not to, ensure your global Python environment is properly configured.

#### 7. Install PyPIVTools
In Developer Command Prompt for VS:
```cmd
pip install -e .
```



## macOS Installation

### 1. Install Xcode Command Line Tools
First, install Apple's command line tools:
```bash
xcode-select --install
```
Follow the prompts to install. This provides basic development tools but we'll need GCC for OpenMP support.

### 2. Install Homebrew
Install the Homebrew package manager:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. Add Homebrew to PATH
Add Homebrew to your PATH by adding these lines to your shell profile (~/.zshrc or ~/.bash_profile):
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Install GCC and FFTW
```bash
brew install gcc fftw
```

### 5. Verify GCC Installation
Check that GCC (not clang) is available:
```bash
/opt/homebrew/bin/gcc-15 --version
```
Should show: gcc-15 (Homebrew GCC X.X.X) X.X.X

If it shows clang, you have the wrong compiler. Ensure you're using the full path to Homebrew's GCC.

### 6. Verify FFTW Installation
Check that FFTW headers and libraries are installed:
```bash
ls /opt/homebrew/include/fftw3.h
ls /opt/homebrew/lib/libfftw3f.dylib
ls /opt/homebrew/lib/libfftw3f_threads.dylib
```

### 7. Set Environment Variables
Add these to your shell profile (~/.zshrc):
```bash
export CC=/opt/homebrew/bin/gcc-15
export CXX=/opt/homebrew/bin/g++-15
export FFTW_INC_PATH="$(brew --prefix fftw)/include"
export FFTW_LIB_PATH="$(brew --prefix fftw)/lib"
```

Make them permanent:
```bash
echo 'export CC=/opt/homebrew/bin/gcc-15' >> ~/.zshrc
echo 'export CXX=/opt/homebrew/bin/g++-15' >> ~/.zshrc
echo 'export FFTW_INC_PATH="$(brew --prefix fftw)/include"' >> ~/.zshrc
echo 'export FFTW_LIB_PATH="$(brew --prefix fftw)/lib"' >> ~/.zshrc
source ~/.zshrc
```

### 8. Verify Environment Variables
```bash
echo $CC
echo $CXX
echo $FFTW_INC_PATH
echo $FFTW_LIB_PATH
```

### 9. Install PyPIVTools
```bash
pip install -e .
```

## Linux Installation

### Ubuntu/Debian

#### 1. Update Package Lists
```bash
sudo apt-get update
sudo apt-get upgrade
```

#### 2. Install Build Tools
```bash
sudo apt-get install -y build-essential
```

#### 3. Install GCC and G++
```bash
sudo apt-get install -y gcc g++
```

#### 4. Install FFTW Development Libraries
```bash
sudo apt-get install -y libfftw3-dev libfftw3-single3
```

#### 5. Install Python Development Headers
```bash
sudo apt-get install -y python3-dev python3-pip
```

#### 6. Verify GCC Installation
```bash
gcc --version
```
Should show: gcc (Ubuntu X.X.X-XubuntuX) X.X.X

#### 7. Verify FFTW Installation
Check headers:
```bash
ls /usr/include/fftw3.h
ls /usr/include/fftw3.f03
```

Check libraries:
```bash
ls /usr/lib/x86_64-linux-gnu/libfftw3f.so
ls /usr/lib/x86_64-linux-gnu/libfftw3f_threads.so
```

#### 8. Set Environment Variables (Optional)
Usually not needed on Ubuntu/Debian, but if you have a custom FFTW installation:
```bash
export FFTW_INC_PATH=/usr/include
export FFTW_LIB_PATH=/usr/lib/x86_64-linux-gnu
```

#### 9. Verify Python Installation
```bash
python3 --version
pip3 --version
```

#### 10. Install PyPIVTools
```bash
pip3 install -e .
```

### Fedora/RHEL/CentOS

#### 1. Install Development Tools Group
For Fedora:
```bash
sudo dnf groupinstall -y "Development Tools"
```

For RHEL/CentOS 8+:
```bash
sudo dnf groupinstall -y "Development Tools"
```

For older versions:
```bash
sudo yum groupinstall -y "Development Tools"
```

#### 2. Install GCC and G++
For Fedora/RHEL 8+:
```bash
sudo dnf install -y gcc gcc-c++
```

For older versions:
```bash
sudo yum install -y gcc gcc-c++
```

#### 3. Install FFTW Development Libraries
For Fedora/RHEL 8+:
```bash
sudo dnf install -y fftw-devel fftw-libs-single
```

For older versions:
```bash
sudo yum install -y fftw-devel fftw-libs-single
```

#### 4. Install Python Development Headers
For Fedora/RHEL 8+:
```bash
sudo dnf install -y python3-devel python3-pip
```

For older versions:
```bash
sudo yum install -y python3-devel python3-pip
```

#### 5. Verify GCC Installation
```bash
gcc --version
```

#### 6. Verify FFTW Installation
Check headers:
```bash
ls /usr/include/fftw3.h
```

Check libraries:
```bash
ls /usr/lib64/libfftw3f.so
ls /usr/lib64/libfftw3f_threads.so
```

#### 7. Set Environment Variables (Optional)
```bash
export FFTW_INC_PATH=/usr/include
export FFTW_LIB_PATH=/usr/lib64
```

#### 8. Verify Python Installation
```bash
python3 --version
pip3 --version
```

#### 9. Install PyPIVTools
```bash
pip3 install -e .
```

## Post-Installation Verification

After installation, verify PyPIVTools works:
```python
import pypivtools
print("PyPIVTools installed successfully!")
print(f"Version: {pypivtools.__version__}")
```

## Troubleshooting

### Common Issues

#### "gcc: command not found" or "cl: command not found"
- **Windows**: Ensure Visual Studio is installed and Developer Command Prompt is used
- **macOS**: Install GCC via Homebrew, not just Xcode command line tools
- **Linux**: Install build-essential package

#### "fftw3.h: No such file or directory"
- **Windows**: Check FFTW_INC_PATH points to correct include directory
- **macOS**: Ensure Homebrew FFTW is installed
- **Linux**: Install libfftw3-dev package

#### "ld: library not found for -lfftw3f"
- Check FFTW_LIB_PATH points to correct library directory
- Ensure FFTW was compiled with single precision support

#### "unsupported option '-fopenmp'"
- **macOS**: Ensure you're using GCC from Homebrew, not Apple's clang
- **Windows**: OpenMP is enabled differently in MSVC - ensure correct compiler flags

#### Build fails with permission errors
- Ensure you have write permissions to the installation directory
- Try running with `sudo` (not recommended for pip installs)
- **Windows**: Administrator privileges are only needed for `setx` commands. Run other commands as regular user.

#### vcpkg installation issues
- **Windows**: If running Developer Command Prompt as administrator starts in System32, navigate to your desired directory first
- Ensure you're using Developer Command Prompt, not regular Command Prompt
- For global vcpkg, `VCPKG_ROOT` should be set after `vcpkg integrate install`
- **Permission denied errors**: Make sure you're cloning to a directory you can write to (not Program Files). Use `cd %USERPROFILE%` to go to your home directory

#### "ffmpeg: command not found"
- Ensure FFmpeg is installed via vcpkg: `vcpkg install ffmpeg`
- Verify that the vcpkg bin directory is added to your PATH
- Restart the command prompt after setting PATH variables

If you encounter issues:
1. Check that all environment variables are set correctly
2. Verify compiler and library installations
3. Try a clean installation in a virtual environment
4. Check the GitHub issues for similar problems

## Development Installation

For development with optional dependencies:
```bash
pip install -e ".[dev]"
```

This includes additional tools like black, isort, flake8, mypy, etc.

## Environment Variables Summary

| Variable | Description | Windows Default | macOS Default | Linux Default |
|----------|-------------|-----------------|---------------|---------------|
| CC | C compiler | cl (MSVC) | /opt/homebrew/bin/gcc-15 | gcc |
| CXX | C++ compiler | cl (MSVC) | /opt/homebrew/bin/g++-15 | g++ |
| FFTW_INC_PATH | FFTW include directory | C:\vcpkg\installed\x64-windows\include | /opt/homebrew/include | /usr/include |
| FFTW_LIB_PATH | FFTW library directory | C:\vcpkg\installed\x64-windows\lib | /opt/homebrew/lib | /usr/lib/x86_64-linux-gnu |

## Supported Platforms

- **Windows**: 10/11 with Visual Studio 2019+ and vcpkg
- **macOS**: 10.15+ with Homebrew GCC
- **Linux**: Ubuntu 18.04+, Fedora 30+, RHEL/CentOS 8+

## Performance Notes

- OpenMP is used for parallel processing - ensure your compiler supports it
- FFTW threading is enabled for multi-threaded FFT operations
- Release builds use optimization flags (-O3/-O2) for best performance