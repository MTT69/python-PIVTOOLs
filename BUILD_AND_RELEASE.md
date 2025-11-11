# Build and Release Guide for PIVTOOLs

## Overview
This guide explains how to build and release the unified `pivtools` package to PyPI.

## Key Files

### 1. `pyproject.toml`
- Defines the unified package `pivtools` with version `0.1.2`
- Includes all dependencies for CLI, GUI, and Core components
- Specifies entry points: `pivtools-cli` and `pivtools-gui`
- **Critical**: Includes package data to ensure shared libraries are packaged

### 2. `setup.py`
- Custom build command `BuildCLib` to compile C libraries
- **Includes dummy extension** to force platform-specific wheels
- Handles platform-specific compilation (Linux, macOS, Windows)

### 3. `.github/workflows/publish-to-pypi.yml`
- GitHub Actions workflow for automated building and publishing
- Builds wheels for Linux, Windows, and macOS
- Builds for Python 3.11 and 3.12
- **Fixed issues:**
  - Corrected Windows FFTW extraction path (`C:\fftw-3.3.10-dll64`)
  - Uses PowerShell `Expand-Archive` instead of unreliable `tar`
  - Added MSVC activation with `ilammy/msvc-dev-cmd@v1`
  - Ensures shared libraries are included via package data

## Critical Setup Changes

### setup.py - Platform-Specific Wheels
The `setup.py` now includes a dummy extension module to force setuptools to create platform-specific wheels:

```python
from setuptools import setup, Extension

dummy_ext = Extension(
    name="pivtools_cli.dummy_marker",
    sources=["pivtools_cli/dummy_marker.c"],
    py_limited_api=False
)

setup(
    cmdclass={"build_ext": BuildCLib},
    ext_modules=[dummy_ext],  # This forces platform-specific wheels
)
```

Without this, setuptools would create a pure Python wheel, which wouldn't work since we have C extensions.

### pyproject.toml - Package Data
Ensures shared libraries are included in wheels:

```toml
[tool.setuptools.package-data]
pivtools_cli = ["lib/*.so", "lib/*.dll", "lib/*.dylib"]
```

### GitHub Actions - Wheel Repair and Platform Fixes

The workflow includes wheel repair commands to bundle shared libraries:

- **Linux**: `auditwheel repair` - Bundles FFTW shared libraries
- **macOS**: `delocate-wheel` - Bundles FFTW shared libraries  
- **Windows**: `delvewheel repair` - Bundles FFTW DLLs

**Platform-specific fixes:**
- **Windows**: Corrected FFTW path to `C:\fftw-3.3.10-dll64`, uses PowerShell extraction, activates MSVC
- **macOS**: Uses `/opt/homebrew` (correct for ARM64 runners)
- **Linux**: Has CentOS mirror fallback for FFTW installation

This ensures the wheels are self-contained and work on systems without FFTW installed.

## Prerequisites

### 1. PyPI API Token
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create an API token (scope: entire account or just this project)
3. Add to GitHub repository:
   - Go to: Settings → Secrets and variables → Actions
   - Create secret named `PYPI_API_TOKEN`
   - Paste your token

### 2. FFTW Dependencies (for local builds)

#### Windows
```cmd
# Download FFTW from https://fftw.org/install/windows.html
# Extract to C:\fftw
set FFTW_INC_PATH=C:\fftw
set FFTW_LIB_PATH=C:\fftw
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install libfftw3-dev
export FFTW_INC_PATH=/usr/include
export FFTW_LIB_PATH=/usr/lib/x86_64-linux-gnu
```

#### macOS
```bash
brew install fftw gcc
export FFTW_INC_PATH=/opt/homebrew/include
export FFTW_LIB_PATH=/opt/homebrew/lib
export CC=gcc-14
```

## Local Build and Test

### Build the package locally
```bash
# Clean previous builds
python -m pip install --upgrade build
python -m build --wheel --outdir dist

# Or build both wheel and sdist
python -m build --outdir dist
```

### Test the wheel locally
```bash
# Create a test environment
python -m venv test_env
test_env\Scripts\activate  # Windows
# source test_env/bin/activate  # Linux/macOS

# Install the wheel
pip install dist/pivtools-0.1.2-*.whl

# Test imports
python -c "import pivtools_cli; import pivtools_gui; import pivtools_core"

# Test CLI
pivtools-cli --help

# Test GUI
pivtools-gui
```

## Release Process

### 1. Update Version
Edit `pyproject.toml` and update the version number:
```toml
version = "0.1.3"  # Increment as needed
```

### 2. Commit Changes
```bash
git add pyproject.toml
git commit -m "Bump version to 0.1.3"
git push origin main
```

### 3. Create and Push Tag
```bash
git tag v0.1.3
git push origin v0.1.3
```

### 4. Create GitHub Release
1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Choose the tag you just pushed (`v0.1.3`)
4. Add release title and description
5. Click "Publish release"

### 5. Automatic Build and Publish
The GitHub Actions workflow will automatically:
1. Build source distribution (sdist) on Linux
2. Build wheels for all platforms (Linux, Windows, macOS)
3. Test the wheels by importing all modules
4. Verify packages with `twine check`
5. Publish to PyPI using your API token

### 6. Monitor the Workflow
- Go to: Actions tab in your GitHub repository
- Watch the "Build and Publish to PyPI" workflow
- Check for any errors in the build logs

## Troubleshooting

### Build fails on a specific platform
- Check the Actions logs for that platform
- Verify FFTW installation commands in the workflow
- Test locally on that platform if possible

### Import test fails
- Check that all C libraries are being compiled
- Verify that wheel repair commands are bundling dependencies
- Test the wheel locally before releasing

### PyPI upload fails
- Verify `PYPI_API_TOKEN` is correctly set in GitHub secrets
- Check if the version already exists on PyPI
- Review PyPI upload logs in GitHub Actions

### Wheel is not platform-specific
- Verify `dummy_ext` is in `setup.py`
- Check that `ext_modules=[dummy_ext]` is in the `setup()` call
- The wheel filename should include platform tags (e.g., `cp311-cp311-win_amd64.whl`)

### Shared libraries not included in wheel
- Verify `[tool.setuptools.package-data]` section in `pyproject.toml`
- Check that the libraries are built to `pivtools_cli/lib/` during compilation
- Confirm the glob patterns match your library filenames

### Windows build fails
- MSVC should be automatically activated by the workflow
- FFTW should extract to `C:\fftw-3.3.10-dll64`
- Check that environment variables point to the correct path

## Package Structure

After installation, users get a unified package with three components:

```python
import pivtools_cli      # CLI tools and PIV processing
import pivtools_gui      # Web-based GUI
import pivtools_core     # Shared utilities
```

Command-line tools:
- `pivtools-cli` - Run PIV processing from command line
- `pivtools-gui` - Launch the web GUI

## Version Compatibility

- **Python**: 3.11, 3.12
- **Platforms**: Windows (64-bit), Linux (64-bit), macOS (64-bit)
- **Architecture**: x86_64 / AMD64

## Notes

- The workflow uses `skip-existing: true` to prevent errors if re-running
- Wheel repair ensures FFTW libraries are bundled (no separate installation needed)
- The dummy extension forces platform-specific wheels without affecting functionality
- All three components (CLI, GUI, Core) are installed as one package
