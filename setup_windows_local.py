import os
import pathlib
import subprocess
import shutil
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildCLib(build_ext):
    def run(self):
        build_dir = pathlib.Path(__file__).parent / "pypivtools" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)
        lib_src_dir = pathlib.Path(__file__).parent / "pypivtools" / "lib"

        # Environment checks
        fftw_inc = os.environ.get('FFTW_INC_PATH')
        fftw_lib = os.environ.get('FFTW_LIB_PATH')
        
        if not fftw_inc or not fftw_lib:
            raise EnvironmentError(
                "Missing FFTW_INC_PATH or FFTW_LIB_PATH environment variables.\n"
                "Install FFTW: vcpkg install fftw3[threads]:x64-windows-static"
            )

        if not shutil.which('cl'):
            raise EnvironmentError(
                "MSVC compiler 'cl' not found. Run from x64 Developer Command Prompt."
            )

        # Compilation settings
        compile_args = ['/O2', '/openmp:experimental', '/MT', '/LD']
        include_dirs = [f'/I{lib_src_dir}', f'/I{fftw_inc}']
        link_args = ['/link', f'/LIBPATH:{fftw_lib}']
        
        # Try static lib first, fall back to dynamic
        fftw_lib_file = None
        for lib_name in ['libfftw3f-3.lib', 'fftw3f.lib']:
            lib_path = os.path.join(fftw_lib, lib_name)
            if os.path.exists(lib_path):
                fftw_lib_file = lib_path
                break
        
        if not fftw_lib_file:
            raise EnvironmentError(f"No FFTW library found in {fftw_lib}")

        # Build configurations
        builds = [
            {
                "name": "libbulkxcorr2d",
                "sources": ["peak_locate_lm.c", "PIV_2d_cross_correlate.c", 
                           "xcorr.c", "xcorr_cache.c"],
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
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(build_dir))
            
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"Failed to build {build['name']}")

        # Cleanup
        for ext in ["*.obj", "*.exp", "*.lib"]:
            for f in build_dir.glob(ext):
                f.unlink(missing_ok=True)


setup(
    name="pypivtools",
    version="0.1.0",
    packages=["pypivtools"],
    package_data={"pypivtools": ["lib/*.dll"]},
    ext_modules=[Extension("dummy", sources=[])],
    cmdclass={"build_ext": BuildCLib},
    python_requires=">=3.8",
)