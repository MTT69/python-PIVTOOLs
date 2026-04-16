# PIVtools

An open-source Python framework for particle image velocimetry (PIV) analysis, integrating planar, stereoscopic, and ensemble PIV pipelines with a React-based GUI and Dask-backed distributed processing.

PIVtools provides an end-to-end workflow: image loading, preprocessing, calibration, correlation, and visualisation driven from a single configuration file. Computationally intensive routines are implemented in C and parallelised with OpenMP and Dask, scaling from workstations to HPC clusters. Validation against synthetic channel flow at Re_τ = 1000 reproduces DNS mean velocity profiles to within 1% down to y⁺ ≈ 10 with instantaneous PIV, and y⁺ ≈ 1 with ensemble PIV.

## Features

- Planar, stereoscopic, and ensemble PIV pipelines
- Direct Reynolds stress extraction from ensemble correlation maps
- React-based GUI for interactive analysis
- Optimised C extensions parallelised with OpenMP and Dask
- Scales from local workstations to HPC clusters
- Complete pipeline from raw image import to vector field visualisation

## Installation

```bash
pip install pivtools
```

This installs the complete toolkit including:
- **Core utilities** for image handling and vector processing
- **Command-line interface** (`pivtools-cli`) for automated workflows
- **React-based GUI** (`pivtools-gui`) for interactive analysis

Pre-compiled C extensions are included for Windows, macOS, and Linux.

## Quick Start

### Initialise a workspace

```bash
pivtools-cli init
```

Creates a default `config.yaml` in the current directory.

### Run PIV analysis (command-line)

```bash
pivtools-cli run
```

Runs the PIV pipeline using `config.yaml` in the current directory.

### Launch the GUI

```bash
pivtools-gui
```

Starts the React-based GUI for interactive configuration and execution.

## Configuration

Both `pivtools-cli` and `pivtools-gui` use `config.yaml` in the current working directory. On first launch, a default `config.yaml` is copied from the package if none exists.

For detailed configuration options, see [piv.tools/manual](https://piv.tools/manual).

## Reproducing the paper

Validation data, benchmark scripts, synthetic image generator configs, calibration images, and pre-computed figures are deposited in the University of Southampton Pure repository:

> **DOI:** *[pending]*

This includes everything needed to reproduce the figures in the accompanying SoftwareX paper, including ground-truth statistics, CSV data for independent replotting, and scripts for end-to-end regeneration from PIVtools output.

## Requirements

- Python 3.12+

## Citation

If you use PIVtools in your research, please cite:

> Taylor, M. T., Lawson, J. M., & Ganapathisubramani, B. (2026). PIVtools: A comprehensive open source software for particle image velocimetry. *SoftwareX*. *[DOI pending]*

## License

BSD 3-Clause License — see [LICENSE](LICENSE) for details.

**Note:** Pre-built binaries link against GPL-licensed libraries (FFTW3, GSL). Binary distributions must comply with GPL terms unless these dependencies are replaced with alternatively-licensed implementations. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for complete details.

## Contributing

Contributions are welcome. Please see the GitHub repository for issues and pull requests.
