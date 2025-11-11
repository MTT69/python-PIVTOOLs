import os
import pathlib
import platform
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Dummy extension to force platform-specific wheel generation
# The actual C libraries are built by the custom BuildCLib command
dummy_ext = Extension(
    name="pivtools_cli._build_marker",
    sources=["pivtools_cli/_build_marker.c"],
    py_limited_api=False
)

class BuildCLib(build_ext):
    """Custom build_ext command to compile C libraries for ctypes"""

    def run(self):
        # Only build if we're actually installing/running setup
        if not self.dry_run:
            try:
                self.build_c_libraries()
            except Exception as e:
                print(f"Warning: Failed to build C libraries: {e}")
                print("This is expected if FFTW dependencies are not available.")
                print("Pre-built wheels should be used for distribution.")
        super().run()

    def build_c_libraries(self):
        """Build the C shared libraries"""
        system = platform.system().lower()

        build_dir = pathlib.Path(self.build_lib) / "pivtools_cli" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)

        lib_src_dir = pathlib.Path(__file__).parent / "pivtools_cli" / "lib"

        if system == 'windows':
            self._build_windows(build_dir, lib_src_dir)
        elif system == 'linux':
            self._build_linux(build_dir, lib_src_dir)
        elif system == 'darwin':
            self._build_macos(build_dir, lib_src_dir)
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

    def _build_windows(self, build_dir, lib_src_dir):
        """Build libraries for Windows"""
        # Check for MSVC
        if not self._find_executable('cl'):
            raise RuntimeError("MSVC compiler 'cl' not found. Run from x64 Developer Command Prompt.")

        # Check FFTW
        fftw_inc = os.environ.get('FFTW_INC_PATH')
        fftw_lib = os.environ.get('FFTW_LIB_PATH')
        if not fftw_inc or not fftw_lib:
            raise RuntimeError("FFTW environment variables not set. Install FFTW and set FFTW_INC_PATH/FFTW_LIB_PATH")

        compile_args = ['/O2', '/openmp:experimental', '/MT', '/LD']
        include_dirs = [f'/I{lib_src_dir}', f'/I{fftw_inc}']
        link_args = ['/link', f'/LIBPATH:{fftw_lib}']

        # Find FFTW lib
        fftw_lib_file = self._find_fftw_lib(fftw_lib)
        if not fftw_lib_file:
            raise RuntimeError(f"No FFTW library found in {fftw_lib}")

        builds = [
            {
                "name": "libbulkxcorr2d",
                "sources": ["peak_locate_lm.c", "PIV_2d_cross_correlate.c", "xcorr.c", "xcorr_cache.c"],
            },
            {
                "name": "libinterp2custom",
                "sources": ["interp2custom.c"],
            }
        ]

        for build in builds:
            sources = [str(lib_src_dir / src) for src in build["sources"]]
            output = str(build_dir / f"{build['name']}.dll")

            cmd = ['cl'] + compile_args + sources + include_dirs
            cmd += link_args + [fftw_lib_file, f'/OUT:{output}']

            print(f"Building {build['name']}...")
            self._run_command(cmd, cwd=str(build_dir))

    def _build_linux(self, build_dir, lib_src_dir):
        """Build libraries for Linux"""
        compiler = os.environ.get('CC', 'gcc')

        # Check compiler
        if not self._find_executable(compiler):
            raise RuntimeError(f"Compiler {compiler} not found")

        # Check FFTW
        fftw_inc = os.environ.get('FFTW_INC_PATH', '/usr/include')
        fftw_lib = os.environ.get('FFTW_LIB_PATH', '/usr/lib/x86_64-linux-gnu')

        if not (pathlib.Path(fftw_inc) / "fftw3.h").exists():
            raise RuntimeError(f"FFTW header not found at {fftw_inc}/fftw3.h")

        compile_args = ['-O3', '-fPIC', '-fopenmp', '-shared']
        include_dirs = [f'-I{lib_src_dir}', f'-I{fftw_inc}']
        link_args = [f'-L{fftw_lib}', '-lfftw3f', '-lm', '-fopenmp']

        builds = [
            {
                "name": "libbulkxcorr2d",
                "sources": ["peak_locate_lm.c", "PIV_2d_cross_correlate.c", "xcorr.c", "xcorr_cache.c"],
            },
            {
                "name": "libinterp2custom",
                "sources": ["interp2custom.c"],
            }
        ]

        for build in builds:
            sources = [str(lib_src_dir / src) for src in build["sources"]]
            output = str(build_dir / f"{build['name']}.so")

            cmd = [compiler] + compile_args + include_dirs + sources + link_args + ['-o', output]

            print(f"Building {build['name']}...")
            self._run_command(cmd)

    def _build_macos(self, build_dir, lib_src_dir):
        """Build libraries for macOS"""
        compiler = os.environ.get('CC', '/opt/homebrew/bin/gcc-14')

        # Check compiler
        if not self._find_executable(compiler):
            raise RuntimeError(f"Compiler {compiler} not found")

        # Check FFTW
        fftw_inc = os.environ.get('FFTW_INC_PATH', '/opt/homebrew/include')
        fftw_lib = os.environ.get('FFTW_LIB_PATH', '/opt/homebrew/lib')

        compile_args = ['-O3', '-fPIC', '-fopenmp', '-shared']
        include_dirs = [f'-I{lib_src_dir}', f'-I{fftw_inc}']
        link_args = [f'-L{fftw_lib}', '-lfftw3f', '-lm', '-fopenmp']

        builds = [
            {
                "name": "libbulkxcorr2d",
                "sources": ["peak_locate_lm.c", "PIV_2d_cross_correlate.c", "xcorr.c", "xcorr_cache.c"],
            },
            {
                "name": "libinterp2custom",
                "sources": ["interp2custom.c"],
            }
        ]

        for build in builds:
            sources = [str(lib_src_dir / src) for src in build["sources"]]
            output = str(build_dir / f"{build['name']}.so")

            cmd = [compiler] + compile_args + include_dirs + sources + link_args + ['-o', output]

            print(f"Building {build['name']}...")
            self._run_command(cmd)

    def _find_executable(self, name):
        """Check if executable exists"""
        try:
            subprocess.run([name, '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _find_fftw_lib(self, fftw_lib_path):
        """Find FFTW library file on Windows"""
        for lib_name in ['libfftw3f-3.lib', 'fftw3f.lib']:
            lib_path = os.path.join(fftw_lib_path, lib_name)
            if os.path.exists(lib_path):
                return lib_path
        return None

    def _run_command(self, cmd, cwd=None):
        """Run a command and check for errors"""
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Command failed with return code {result.returncode}")

# This setup.py is minimal since we use pyproject.toml
setup(
    cmdclass={"build_ext": BuildCLib},
    ext_modules=[dummy_ext],
)
