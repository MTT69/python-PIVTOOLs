import os
import pathlib
import subprocess
import sys
import shutil
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildCLib(build_ext):

    def run(self):
        build_dir = pathlib.Path(__file__).parent / "pypivtools" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)
        lib_src_dir = pathlib.Path(__file__).parent / "pypivtools" / "lib"

        compiler = os.environ.get('CC', 'cl')
        fftw_inc = os.environ.get('FFTW_INC_PATH')
        fftw_lib = os.environ.get('FFTW_LIB_PATH')

        lib_extension = '.dll'
        openmp_flag = '/openmp:experimental'
        shared_flag = '/LD'
        opt_flag = '/O2'

        required_vars = ['FFTW_INC_PATH', 'FFTW_LIB_PATH']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}. "
                                   f"Please set them to point to your FFTW installation.")

        required_libs = ['fftw3f.lib']
        missing_libs = []
        for lib in required_libs:
            lib_path = os.path.join(fftw_lib, lib)
            if not os.path.exists(lib_path):
                missing_libs.append(lib)
        if missing_libs:
            raise EnvironmentError(
                f"Missing required FFTW library files: {', '.join(missing_libs)} in {fftw_lib}.\n"
                f"Please install FFTW with threading support using: vcpkg install fftw3[threads]"
            )

        compiler_path = shutil.which(compiler)
        if not compiler_path or not os.path.basename(compiler).lower().startswith('cl'):
            raise EnvironmentError(
                f"This setup requires the Visual Studio MSVC compiler 'cl' on PATH.\n"
                f"Please open an x64 Developer Command Prompt for Visual Studio or run the appropriate vcvarsall.bat before installing.\n"
                f"Current CC value: {os.environ.get('CC', None)}\n"
                f"PATH contains cl: {bool(shutil.which('cl'))}"
            )

        # Print compiler version
        try:
            proc = subprocess.run([compiler], capture_output=True, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            print(f"Using compiler: {compiler}")
            for line in out.splitlines():
                if line.strip():
                    print(line.strip())
                    break
        except Exception as e:
            raise EnvironmentError(f"Compiler 'cl' invocation failed: {e}. Ensure Developer Command Prompt is used.")

        # Build DLLs
        builds = [
            {
                "name": "libbulkxcorr2d",
                "sources": [
                    str(lib_src_dir / "peak_locate_lm.c"),
                    str(lib_src_dir / "PIV_2d_cross_correlate.c"),
                    str(lib_src_dir / "xcorr.c"),
                    str(lib_src_dir / "xcorr_cache.c"),
                ],
                "output": str(build_dir / f"libbulkxcorr2d{lib_extension}")
            },
            {
                "name": "libinterp2custom",
                "sources": [str(lib_src_dir / "interp2custom.c")],
                "output": str(build_dir / f"libinterp2custom{lib_extension}")
            }
        ]

        for build in builds:
            cmd = [compiler, openmp_flag, opt_flag, shared_flag, f"/Fo{build_dir}/"]
            cmd.extend(build["sources"])
            cmd.append(f"/I{lib_src_dir}")
            cmd.append(f"/I{fftw_inc}")
            cmd.extend([
                "/link",
                f"/LIBPATH:{fftw_lib}",
                os.path.join(fftw_lib, 'fftw3f.lib'),
                f"/OUT:{build['output']}"
            ])
            print(f"Building {build['name']} with command: {' '.join(cmd)}")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                print(proc.stdout)
                print(proc.stderr)
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed to build {build['name']} DLL. See output above.")
            except Exception as e:
                raise RuntimeError(f"Error building {build['name']}: {e}")

        # Clean up .obj files
        for obj_file in build_dir.glob("*.obj"):
            try:
                obj_file.unlink()
            except Exception:
                pass


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