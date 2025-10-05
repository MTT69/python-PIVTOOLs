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

        # Cross-platform compiler detection
        compiler = os.environ.get('CC', 'gcc')

        # Get library paths (GSL no longer needed - using fast LM solver)
        fftw_inc = os.environ.get('FFTW_INC_PATH', '/usr/include')
        fftw_lib = os.environ.get('FFTW_LIB_PATH', '/usr/lib')

        # Platform-specific settings
        if sys.platform == 'win32':
            lib_extension = '.dll'
            openmp_flag = '/openmp'
            pic_flag = ''
            shared_flag = '/LD'
            opt_flag = '/O2'
            sanitize_flag = ''
        else:
            lib_extension = '.so'
            openmp_flag = '-fopenmp'
            pic_flag = '-fPIC'
            shared_flag = '-shared'
            opt_flag = '-O3'
            # Sanitizer not available on macOS
            sanitize_flag = (
                '-fsanitize=address' if sys.platform != 'darwin' else ''
            )

        # First library - using fast LM solver (no GSL needed)
        cmd = [compiler]
        if pic_flag:
            cmd.append(pic_flag)
        cmd.extend([
            openmp_flag,
            opt_flag,
            shared_flag,
            # Fast LM solver instead of GSL
            str(lib_src_dir / "peak_locate_lm.c"),
            str(lib_src_dir / "PIV_2d_cross_correlate.c"),
            str(lib_src_dir / "xcorr.c"),
            str(lib_src_dir / "xcorr_cache.c"),  # Wisdom caching
            f"-I{lib_src_dir}",
            f"-I{fftw_inc}",
            f"-L{fftw_lib}",
            "-lfftw3f",
            "-lfftw3f_threads",
            "-lm",
            "-o",
            str(build_dir / f"libbulkxcorr2d{lib_extension}"),
        ])
        subprocess.check_call(cmd)

        # Second library (same changes)
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
        subprocess.check_call(cmd)


setup(
    name="pypivtools",
    version="0.1.0",
    packages=["pypivtools"],
    ext_modules=[Extension("dummy", sources=[])],
    cmdclass={"build_ext": BuildCLib},
    install_requires=[
        "dask[complete]",
        "tifffile",
        "numpy",
        "scipy",
        "dask-image",
        "scipy",
        "typeguard",
        "opencv-python",
        "scikit-image",
    ],
    extras_require={
        "dev": ["black", "isort", "flake8", "cornflakes", "pandas"],
    },
)
