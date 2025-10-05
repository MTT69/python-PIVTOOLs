# PyPIVTools Installation Guide

PyPIVTools requires compilation of C libraries with OpenMP support and links against FFTW and GSL libraries.

## Prerequisites
Note that pip install -r requirements.txt must be ran afterwards

### macOS

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install dependencies:
   ```bash
   brew install gcc fftw gsl
   ```

3. Set up environment variables:
   ```bash
   export CC=/opt/homebrew/bin/gcc-15
   export CXX=/opt/homebrew/bin/g++-15
   export CPPFLAGS="-I/opt/homebrew/include"
   export LDFLAGS="-L/opt/homebrew/lib"
   export FFTW_LIB_PATH="$(brew --prefix fftw)/lib"
   export FFTW_INC_PATH="$(brew --prefix fftw)/include"
   export GSL_LIB_PATH="$(brew --prefix gsl)/lib"
   export GSL_INC_PATH="$(brew --prefix gsl)/include"
   ```

   **Note:** To make these permanent, add them to your `~/.zshrc` or `~/.bash_profile`:
   ```bash
   echo 'export CC=/opt/homebrew/bin/gcc-15' >> ~/.zshrc
   echo 'export CXX=/opt/homebrew/bin/g++-15' >> ~/.zshrc
   echo 'export CPPFLAGS="-I/opt/homebrew/include"' >> ~/.zshrc
   echo 'export LDFLAGS="-L/opt/homebrew/lib"' >> ~/.zshrc
   echo 'export FFTW_LIB_PATH="$(brew --prefix fftw)/lib"' >> ~/.zshrc
   echo 'export FFTW_INC_PATH="$(brew --prefix fftw)/include"' >> ~/.zshrc
   echo 'export GSL_LIB_PATH="$(brew --prefix gsl)/lib"' >> ~/.zshrc
   echo 'export GSL_INC_PATH="$(brew --prefix gsl)/include"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. Install PyPIVTools:
   ```bash
   pip install -e .
   ```

### Linux (Ubuntu/Debian)

1. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install gcc g++ libfftw3-dev libgsl-dev
   ```

2. Set up environment variables (optional, usually not needed on Linux):
   ```bash
   export FFTW_INC_PATH=/usr/include
   export FFTW_LIB_PATH=/usr/lib/x86_64-linux-gnu
   export GSL_INC_PATH=/usr/include
   export GSL_LIB_PATH=/usr/lib/x86_64-linux-gnu
   ```

3. Install PyPIVTools:
   ```bash
   pip install -e .
   ```

### Linux (Fedora/RHEL/CentOS)

1. Install dependencies:
   ```bash
   sudo dnf install gcc gcc-c++ fftw-devel gsl-devel
   # or for older versions:
   # sudo yum install gcc gcc-c++ fftw-devel gsl-devel
   ```

2. Set up environment variables (optional):
   ```bash
   export FFTW_INC_PATH=/usr/include
   export FFTW_LIB_PATH=/usr/lib64
   export GSL_INC_PATH=/usr/include
   export GSL_LIB_PATH=/usr/lib64
   ```

3. Install PyPIVTools:
   ```bash
   pip install -e .
   ```

### Windows

**Option 1: Using MinGW-w64 (Recommended)**

1. Install MSYS2 from https://www.msys2.org/

2. Open MSYS2 MinGW64 terminal and install dependencies:
   ```bash
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-fftw mingw-w64-x86_64-gsl
   ```

3. Set up environment variables in PowerShell or Command Prompt:
   ```powershell
   set CC=gcc
   set CXX=g++
   set FFTW_INC_PATH=C:\msys64\mingw64\include
   set FFTW_LIB_PATH=C:\msys64\mingw64\lib
   set GSL_INC_PATH=C:\msys64\mingw64\include
   set GSL_LIB_PATH=C:\msys64\mingw64\lib
   ```

4. Add MinGW to your PATH:
   ```powershell
   set PATH=C:\msys64\mingw64\bin;%PATH%
   ```

5. Install PyPIVTools:
   ```bash
   pip install -e .
   ```

**Option 2: Using Visual Studio**

1. Install Visual Studio 2019 or later with C++ build tools

2. Manually install FFTW and GSL libraries (or use vcpkg):
   ```bash
   vcpkg install fftw3 gsl
   ```

3. Set environment variables pointing to your vcpkg installation

4. Modify `setup.py` to use MSVC flags (see setup.py for platform detection)

5. Install PyPIVTools:
   ```bash
   pip install -e .
   ```

## Troubleshooting

### macOS: "clang: error: unsupported option '-fopenmp'"
Make sure you're using GCC, not clang. Verify with:
```bash
$CC --version  # Should show GCC, not Apple clang
```

### Linux: "fatal error: fftw3.h: No such file or directory"
Install the development packages:
```bash
sudo apt-get install libfftw3-dev libgsl-dev
```

### Windows: "error: gsl/gsl_rng.h: No such file or directory"
Ensure MSYS2 packages are installed and paths are set correctly. The paths should point to your MSYS2 installation directory.

### General: Build fails with linking errors
Make sure both `_INC_PATH` and `_LIB_PATH` environment variables are set for both FFTW and GSL.

## Verifying Installation

After installation, verify it works:
```python
import pypivtools
print("PyPIVTools installed successfully!")
```

## Development Installation

For development with optional dependencies:
```bash
pip install -e ".[dev]"
```

This includes additional tools like black, isort, flake8, etc.