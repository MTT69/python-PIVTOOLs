# PyPIVTools Distribution Guide

This guide explains how to transform PyPIVTools from a source-only package requiring compilation into a professionally distributed pip package with pre-compiled wheels for all major platforms.

## Current Problem

Currently, users must:
1. Install compilers (GCC, MSVC)
2. Install system libraries (FFTW)
3. Set environment variables
4. Compile C extensions during installation

This creates a poor user experience and high barrier to entry.

## Solution: Pre-compiled Wheels

The solution is to build platform-specific wheels containing pre-compiled C extensions, allowing users to simply run `pip install pypivtools`.

## Prerequisites

- GitHub repository with CI/CD
- PyPI account
- Understanding of cross-compilation

## Step 1: Modernize Package Structure

### Create pyproject.toml

Replace setup.py with modern packaging:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "pypivtools"
dynamic = ["version"]
description = "Particle Image Velocimetry Tools"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
]

[project.optional-dependencies]
dev = [
    "black",
    "isort",
    "flake8",
    "mypy",
    "pytest",
    "pytest-cov",
]
docs = [
    "sphinx",
    "sphinx-rtd-theme",
]

[tool.setuptools]
zip-safe = false
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
include = ["pypivtools*"]

[tool.setuptools_scm]
write_to = "src/pypivtools/_version.py"

[tool.setuptools.dynamic]
version = {attr = "pypivtools._version.get_version"}

[project.urls]
Homepage = "https://github.com/yourusername/pypivtools"
Documentation = "https://pypivtools.readthedocs.io/"
Repository = "https://github.com/yourusername/pypivtools"
Issues = "https://github.com/yourusername/pypivtools/issues"
Changelog = "https://github.com/yourusername/pypivtools/blob/main/CHANGELOG.md"
```

### Restructure Directory

```
pypivtools/
├── pyproject.toml
├── MANIFEST.in
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── src/
│   └── pypivtools/
│       ├── __init__.py
│       ├── _version.py
│       └── ...
├── pypivtools/
│   ├── lib/
│   │   ├── *.c
│   │   └── *.h
│   └── ...
└── tests/
```

### Create MANIFEST.in

```txt
include README.md
include LICENSE
include CHANGELOG.md
recursive-include pypivtools/lib *.c *.h
global-exclude *.pyc
global-exclude __pycache__
```

## Step 2: Update setup.py for Wheel Building

Modify setup.py to handle both wheel and source builds:

```python
import os
import pathlib
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Only build extensions if not building a wheel
BUILD_EXTENSIONS = not (os.environ.get('CIBUILDWHEEL', '0') == '1')

class BuildCLib(build_ext):
    def run(self):
        if not BUILD_EXTENSIONS:
            print("Skipping C extension build for wheel distribution")
            return

        # ... existing build code ...

if BUILD_EXTENSIONS:
    setup(
        name="pypivtools",
        # ... existing setup ...
        ext_modules=[Extension("dummy", sources=[])],
        cmdclass={"build_ext": BuildCLib},
    )
else:
    setup(
        name="pypivtools",
        # ... minimal setup for wheel ...
    )
```

## Step 3: Set Up cibuildwheel

### Install cibuildwheel

```bash
pip install cibuildwheel
```

### Create pyproject.toml Build Configuration

Add to pyproject.toml:

```toml
[tool.cibuildwheel]
build = "cp38-* cp39-* cp310-* cp311-* cp312-*"
skip = ["*-win32", "*-manylinux_i686", "*-musllinux*"]
archs = ["auto"]
before-build = "pip install -r requirements-build.txt"

[tool.cibuildwheel.linux]
before-build = """
yum install -y fftw-devel ||
apt-get update && apt-get install -y libfftw3-dev
"""

[tool.cibuildwheel.macos]
before-build = """
brew install fftw
export FFTW_INC_PATH="$(brew --prefix fftw)/include"
export FFTW_LIB_PATH="$(brew --prefix fftw)/lib"
"""

[tool.cibuildwheel.windows]
before-build = """
vcpkg install fftw3 --triplet x64-windows
set FFTW_INC_PATH=C:\\vcpkg\\installed\\x64-windows\\include
set FFTW_LIB_PATH=C:\\vcpkg\\installed\\x64-windows\\lib
"""
```

### Create requirements-build.txt

```
setuptools>=61.0
wheel
setuptools_scm
numpy>=1.21.0
scipy>=1.7.0
```

## Step 4: Set Up GitHub Actions CI/CD

### Create .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: |
        pytest tests/ -v --cov=pypivtools

    - name: Run linting
      run: |
        black --check src/
        isort --check-only src/
        flake8 src/
```

### Create .github/workflows/release.yml

```yaml
name: Release

on:
  release:
    types: [published]

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"

    - name: Install cibuildwheel
      run: python -m pip install cibuildwheel

    - name: Build wheels
      run: python -m cibuildwheel --output-dir wheelhouse

    - name: Store wheels
      uses: actions/upload-artifact@v3
      with:
        name: wheels-${{ matrix.os }}
        path: wheelhouse/*.whl

  build_sdist:
    name: Build source distribution
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"

    - name: Build sdist
      run: |
        pip install build
        python -m build --sdist

    - name: Store sdist
      uses: actions/upload-artifact@v3
      with:
        name: sdist
        path: dist/*.tar.gz

  publish:
    needs: [build_wheels, build_sdist]
    runs-on: ubuntu-latest

    steps:
    - name: Download artifacts
      uses: actions/download-artifact@v3

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        user: __token__
        password: ${{ secrets.PYPI_API_TOKEN }}
        packages-dir: artifacts/
```

## Step 5: Handle FFTW Dependency

### Option 1: Bundle FFTW in Wheels

For maximum portability, bundle FFTW libraries in the wheels:

```python
# In setup.py
from setuptools import setup
import platform
import os

# Platform-specific FFTW library handling
if platform.system() == 'Windows':
    libraries = ['fftw3f', 'fftw3f_threads']
    library_dirs = [os.environ.get('FFTW_LIB_PATH', 'C:\\vcpkg\\installed\\x64-windows\\lib')]
    include_dirs = [os.environ.get('FFTW_INC_PATH', 'C:\\vcpkg\\installed\\x64-windows\\include')]
elif platform.system() == 'Darwin':
    libraries = ['fftw3f', 'fftw3f_threads']
    library_dirs = [os.environ.get('FFTW_LIB_PATH', '/opt/homebrew/lib')]
    include_dirs = [os.environ.get('FFTW_INC_PATH', '/opt/homebrew/include')]
else:  # Linux
    libraries = ['fftw3f', 'fftw3f_threads']
    library_dirs = [os.environ.get('FFTW_LIB_PATH', '/usr/lib/x86_64-linux-gnu')]
    include_dirs = [os.environ.get('FFTW_INC_PATH', '/usr/include')]

ext_modules = [
    Extension(
        "pypivtools._libbulkxcorr2d",
        sources=["pypivtools/lib/peak_locate_lm.c", "pypivtools/lib/PIV_2d_cross_correlate.c",
                "pypivtools/lib/xcorr.c", "pypivtools/lib/xcorr_cache.c"],
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs + ["pypivtools/lib"],
        extra_compile_args=["-fopenmp" if platform.system() != 'Windows' else '/openmp'],
        extra_link_args=["-fopenmp" if platform.system() != 'Windows' else '/openmp'],
    ),
    Extension(
        "pypivtools._libinterp2custom",
        sources=["pypivtools/lib/interp2custom.c"],
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs + ["pypivtools/lib"],
    ),
]
```

### Option 2: Use System FFTW

For smaller wheels, rely on system FFTW (requires users to install it):

- Linux: Package managers handle this
- macOS: Homebrew
- Windows: vcpkg or manual installation

## Step 6: Test Wheels Locally

### Test Wheel Building

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for current platform
cibuildwheel --platform auto

# Test installation
pip install wheelhouse/pypivtools-*.whl --force-reinstall
python -c "import pypivtools; print('Success!')"
```

### Test Cross-Platform Compatibility

Use Docker for testing different Linux distributions:

```bash
# Test on manylinux
docker run -it -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64 /bin/bash
cd /io
pip install cibuildwheel
cibuildwheel --platform linux
```

## Step 7: Publish to PyPI

### Set Up PyPI Account

1. Create account at https://pypi.org/
2. Generate API token
3. Add token to GitHub repository secrets as `PYPI_API_TOKEN`

### Release Process

1. Update version in pyproject.toml or use git tags
2. Create GitHub release
3. CI/CD automatically builds and publishes wheels

## Step 8: Update Documentation

### Update README.md

```markdown
# PyPIVTools

[![PyPI version](https://badge.fury.io/py/pypivtools.svg)](https://pypi.org/project/pypivtools/)
[![Python versions](https://img.shields.io/pypi/pyversions/pypivtools.svg)](https://pypi.org/project/pypivtools/)
[![License](https://img.shields.io/pypi/l/pypivtools.svg)](https://github.com/yourusername/pypivtools/blob/main/LICENSE)

Particle Image Velocimetry Tools for Python.

## Installation

```bash
pip install pypivtools
```

No compilation required! Pre-built wheels are available for all major platforms.

## Development Installation

For development with optional dependencies:

```bash
git clone https://github.com/yourusername/pypivtools.git
cd pypivtools
pip install -e .[dev]
```
```

### Update Installation Instructions

Simplify to:

```markdown
## Installation

### From PyPI (Recommended)

```bash
pip install pypivtools
```

This installs pre-compiled wheels with no compilation required.

### From Source (Development)

See [install_instructions.md](install_instructions.md) for detailed setup instructions.
```

## Benefits of This Approach

1. **Better User Experience**: `pip install pypivtools` works immediately
2. **Cross-Platform**: Wheels for Windows, macOS, Linux
3. **No Compiler Required**: Users don't need build tools
4. **Faster Installation**: No compilation time
5. **Reliable**: Tested builds on CI/CD
6. **Professional**: Follows Python packaging best practices

## Maintenance Considerations

- Monitor CI/CD for build failures
- Update FFTW versions as needed
- Test on new Python versions
- Handle platform-specific issues
- Keep dependencies up to date

## Troubleshooting Wheel Builds

### Common Issues

1. **Missing libraries**: Ensure FFTW is installed in CI environment
2. **Architecture mismatches**: Use correct cibuildwheel archs setting
3. **Python version compatibility**: Test all supported versions
4. **macOS deployment target**: Set MACOSX_DEPLOYMENT_TARGET

### Debugging

```bash
# Test specific wheel
pip install wheelhouse/pypivtools-*.whl --force-reinstall
python -c "import pypivtools; pypivtools.test()"

# Check wheel contents
wheel unpack wheelhouse/pypivtools-*.whl
```

This transformation makes PyPIVTools accessible to a much wider audience while maintaining the high performance of compiled C extensions.