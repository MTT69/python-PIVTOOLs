import os
import pathlib
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildCLib(build_ext):
    def run(self):
        build_dir = pathlib.Path(__file__).parent / "pypivtools" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)
        lib_src_dir = pathlib.Path(__file__).parent / "pypivtools" / "lib"

        # Compiler detection
        compiler = os.environ.get('CC', 'gcc')

        # Get library paths - usually standard on Linux
        fftw_inc = os.environ.get('FFTW_INC_PATH', '/usr/include')
        fftw_lib = os.environ.get('FFTW_LIB_PATH', '/usr/lib/x86_64-linux-gnu')

        # Linux-specific settings
        lib_extension = '.so'
        openmp_flag = '-fopenmp'
        pic_flag = '-fPIC'
        shared_flag = '-shared'
        opt_flag = '-O3'
        sanitize_flag = '-fsanitize=address'  # Available on Linux

        # Check for required environment variables (optional on Linux, but good to verify)
        print("Note: On Linux, FFTW_INC_PATH and FFTW_LIB_PATH are usually not needed if installed via package manager.")
        if not os.environ.get('FFTW_INC_PATH'):
            print("FFTW_INC_PATH not set, using default: /usr/include")
        if not os.environ.get('FFTW_LIB_PATH'):
            print("FFTW_LIB_PATH not set, using default: /usr/lib/x86_64-linux-gnu")

        # Verify compiler
        try:
            result = subprocess.run([compiler, '--version'], capture_output=True, text=True, check=True)
            print(f"Using compiler: {compiler}")
            print(result.stdout.split('\n')[0])
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise EnvironmentError(f"Compiler {compiler} not found. Please install build-essential package.")

        # Verify FFTW headers
        fftw_header = pathlib.Path(fftw_inc) / "fftw3.h"
        if not fftw_header.exists():
            raise EnvironmentError(f"FFTW header not found at {fftw_header}. Please install libfftw3-dev.")

        # First library - using fast LM solver
        cmd = [compiler]
        if pic_flag:
            cmd.append(pic_flag)
        cmd.extend([
            openmp_flag,
            opt_flag,
            shared_flag,
            str(lib_src_dir / "peak_locate_lm.c"),
            str(lib_src_dir / "PIV_2d_cross_correlate.c"),
            str(lib_src_dir / "xcorr.c"),
            str(lib_src_dir / "xcorr_cache.c"),
            f"-I{lib_src_dir}",
            f"-I{fftw_inc}",
            f"-L{fftw_lib}",
            "-lfftw3f",
            "-lfftw3f_threads",
            "-lm",
            "-o",
            str(build_dir / f"libbulkxcorr2d{lib_extension}"),
        ])
        print(f"Building first library with command: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        # Second library
        cmd = [compiler]
        if pic_flag:
            cmd.append(pic_flag)
        cmd.extend([openmp_flag, opt_flag, shared_flag])
        if sanitize_flag:
            cmd.append(sanitize_flag)
        cmd.extend([
            str(lib_src_dir / "interp2custom.c"),
            f"-I{lib_src_dir}",
            f"-I{fftw_inc}",
            f"-L{fftw_lib}",
            "-o",
            str(build_dir / f"libinterp2custom{lib_extension}"),
        ])
        print(f"Building second library with command: {' '.join(cmd)}")
        subprocess.check_call(cmd)


# Read requirements from requirements.txt
def read_requirements():
    """Read and parse requirements.txt file."""
    requirements = []
    req_file = pathlib.Path(__file__).parent / "requirements.txt"

    if req_file.exists():
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and package extras like dask[complete]
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Skip lines that are just package names without versions (duplicates at bottom)
                    # Only include lines with version specifiers (==, >=, etc.)
                    if '==' in line or '>=' in line or '<=' in line or '~=' in line:
                        requirements.append(line)

    return requirements


setup(
    name="pypivtools",
    version="0.1.0",
    packages=["pypivtools"],
    ext_modules=[Extension("dummy", sources=[])],
    cmdclass={"build_ext": BuildCLib},
    install_requires=read_requirements(),
)