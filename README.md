# PIVTOOLs

Particle Image Velocimetry Tools - A comprehensive toolkit for PIV analysis with both command-line and GUI interfaces.

## Installation

Install PIVTOOLs with a single command:

```bash
pip install pivtools
```

This installs the complete toolkit including:
- **Core utilities** for image handling and vector processing
- **Command-line interface** (`pivtools-cli`) for automated workflows
- **Web-based GUI** (`pivtools-gui`) for interactive analysis

The package includes pre-compiled C extensions for optimal performance on Windows, macOS, and Linux.

## Quick Start

### Initialize a new PIVTOOLs workspace

```bash
pivtools-cli init
```

This creates a default `config.yaml` file in your current directory that you can edit to configure your PIV analysis.

### Run PIV analysis (command-line)

```bash
pivtools-cli run
```

This runs the PIV analysis using the `config.yaml` in your current directory.

### Launch the GUI

```bash
pivtools-gui
```

This starts the web-based GUI at http://localhost:5000 where you can interactively configure and run PIV analysis.

## Configuration

Edit the `config.yaml` file to configure:

- Input/output paths
- Image formats and processing parameters
- PIV algorithm settings
- Filtering and post-processing options
- Calibration parameters

See the example `config.yaml` for all available options.

## Requirements

- Python 3.8+
- FFTW library (automatically handled during installation)
- Camera calibration images (for accurate scaling)

## Development

For development installation with local C compilation:

```bash
# Install FFTW first (platform-specific)
# Windows: vcpkg install fftw3[threads]:x64-windows-static
# macOS: brew install fftw gcc
# Linux: apt-get install libfftw3-dev

# Set environment variables
export FFTW_INC_PATH=/path/to/fftw/include
export FFTW_LIB_PATH=/path/to/fftw/lib

# Install in development mode
pip install -e .
```

## License

MIT License

Particle Image Velocimetry Tools - A comprehensive toolkit for PIV analysis with both command-line and GUI interfaces.

## Installation

Install PIVTOOLs from PyPI:

```bash
pip install PIVTOOLs
```

This will install the package with pre-compiled C extensions for your platform.

## Quick Start

### Initialize a new PIV workspace

```bash
pivtools-cli init
```

This creates a `config.yaml` file in your current directory with default settings.

### Run PIV analysis (command line)

```bash
pivtools-cli run
```

### Launch the GUI

```bash
pivtools-gui
```

Then open http://localhost:5000 in your browser.

## Configuration

Edit the `config.yaml` file to configure:

- Input/output paths
- Image formats and processing parameters
- PIV algorithm settings
- Filtering and post-processing options

## Requirements

- Python 3.8+
- FFTW library (automatically handled during installation)
- See `requirements.txt` for full dependencies

## Development

For development installation:

```bash
git clone https://github.com/MTT69/python-PIVTOOLs.git
cd python-PIVTOOLs
pip install -e .
```

## Building from Source

If building from source, ensure FFTW is installed:

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install libfftw3-dev
```

### macOS
```bash
brew install fftw
```

### Windows
Install FFTW via vcpkg:
```bash
vcpkg install fftw3[threads]:x64-windows-static
```

Set environment variables:
```bash
set FFTW_INC_PATH=C:\path\to\fftw\include
set FFTW_LIB_PATH=C:\path\to\fftw\lib
```

## License

MIT License

## Contributing

Contributions welcome! Please see the GitHub repository for issues and pull requests.
