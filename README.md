# PIVTOOLs

Particle Image Velocimetry Tools - A comprehensive toolkit for PIV analysis with both command-line and GUI interfaces.

We present **PIVTOOLS**, an open-source Python framework built for community expansion with a React-based GUI, which integrates planar, stereoscopic, and ensemble PIV pipelines into a single end-to-end environment. Computationally intensive routines are implemented in optimised C and parallelised with Dask, enabling datasets of terabyte scale to be processed efficiently on both workstations and distributed HPC clusters.

The framework provides a complete pipeline from raw image import to image preprocessing, parallel vector computation, calibration, and interactive visualisation. Ensemble extensions allow direct estimation of Reynolds stresses from correlation maps, offering statistical fidelity beyond what is achievable with instantaneous methods. Validation against synthetic channel flow demonstrates mean velocity profiles accurate to within 1% of DNS reference down to \( y^+ \approx 40-50 \) with instantaneous PIV, and to \( y^+ \approx 15 \) with ensemble methods. Instantaneous Reynolds stresses agree with windowed DNS data, while ensemble processing recovers a higher fraction of turbulent energy due to reduced windowing effects.

## Features

- Planar, stereoscopic, and ensemble PIV pipelines
- React-based GUI for interactive analysis
- Optimized C extensions for performance
- Parallel processing with Dask
- Support for terabyte-scale datasets
- Complete pipeline from image import to visualization

## Installation

Install PIVTOOLs with a single command:

```bash
pip install pivtools
```

This installs the complete toolkit including:
- **Core utilities** for image handling and vector processing
- **Command-line interface** (`pivtools-cli`) for automated workflows
- **React-based GUI** (`pivtools-gui`) for interactive analysis

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

This starts the React-based GUI where you can interactively configure and run PIV analysis.

## Configuration

The `pivtools-cli` application places `config.yaml` in the current working directory.

The `pivtools-gui` application uses a different config location. Instead of placing `config.yaml` in the current working directory, the GUI stores it in the user's application data directory:

- Windows: `%APPDATA%\pivtools\config.yaml` (typically `C:\Users\<username>\AppData\Roaming\pivtools\config.yaml`)
- macOS/Linux: `~/.config/pivtools/config.yaml`

When the GUI starts, if no config exists in this location, it automatically copies the default config from the package to this user config directory.

For detailed configuration options, see [piv.tools/manual](https://piv.tools/manual).

## Requirements

- Python 3.10+

## License

MIT License

## Contributing

Contributions welcome! Please see the GitHub repository for issues and pull requests.
