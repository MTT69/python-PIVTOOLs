# setup.py
import os
import platform
import pathlib
import subprocess
import sysconfig
from setuptools import find_packages, setup, Distribution
from setuptools.command.build import build
from setuptools.command.develop import develop
from setuptools.command.install import install
import shutil


class BinaryDistribution(Distribution):
    """Distribution that forces a platform-specific wheel."""

    def has_ext_modules(self):
        return True


class BuildCLibraries(build):
    """Custom build command that compiles C libraries before the standard build."""

    def run(self):
        print(">>> Building C libraries <<<")
        self.python_include = sysconfig.get_path("include")
        self.pkg_dir = pathlib.Path(__file__).parent
        if not self.dry_run:
            self.build_c_libraries()
        super().run()

    def build_c_libraries(self):
        build_dir = self.pkg_dir / "pivtools_cli" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)
        src_dir = self.pkg_dir / "pivtools_cli" / "lib"
        sys_name = platform.system().lower()

        # === macOS ===
        if sys_name == "darwin":
            sys_name = "macos"
            arch = platform.machine().lower()

            # Always use static FFTW from bundled libraries
            if arch == "arm64":
                fftw_static_dir = self.pkg_dir / "static_fftw" / "macos_arm64"
            else:
                raise RuntimeError(f"Unsupported macOS architecture: {arch}. Only arm64 is supported.")

            self.fftw_inc = fftw_static_dir / "include"
            self.fftw_lib = fftw_static_dir / "lib"
            fftw_lib_file = self.fftw_lib / "libfftw3f.a"
            fftw_omp_file = self.fftw_lib / "libfftw3f_omp.a"
            if not fftw_lib_file.exists():
                raise RuntimeError(f"FFTW static lib not found: {fftw_lib_file}")
            print(f"Using static FFTW from: {fftw_static_dir}")

            # Prefer Homebrew GCC (has OpenMP support) over Apple clang (no OpenMP)
            compiler = (
                shutil.which("gcc-15") or
                shutil.which("gcc-14") or
                shutil.which("gcc-13") or
                shutil.which("gcc")
            )
            if compiler is None or "/usr/bin/gcc" in str(compiler):
                # /usr/bin/gcc is actually Apple clang, not real GCC
                raise RuntimeError("No suitable GCC compiler found. Install via: brew install gcc")
            print(f"Using compiler: {compiler}")

            sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            print(f"Using SDK path: {sdk_path}")
            self.extra_compile = ["-O3", "-fPIC", "-fopenmp", "-DFFTW_THREADS", f"-I{self.fftw_inc}", "-isysroot", sdk_path]
            # Link statically with FFTW .a files
            self.extra_link = ["-lm", "-fopenmp", str(fftw_lib_file), str(fftw_omp_file), "-isysroot", sdk_path]
            shared_flag = "-shared"

            lib_ext = ".so"  # Use .so on macOS for compatibility with Python ctypes
            use_msvc = False

        # === Windows ===
        elif sys_name == "windows":
            # Always use static FFTW from bundled libraries
            fftw_dir = self.pkg_dir / "static_fftw" / "windows"
            if not fftw_dir.exists():
                raise RuntimeError(f"Static FFTW not found: {fftw_dir}")

            self.fftw_inc = fftw_dir / "include"
            self.fftw_lib = fftw_dir / "lib"
            fftw_lib_file = self.fftw_lib / "libfftw3f-3.lib"
            if not fftw_lib_file.exists():
                raise RuntimeError(f"FFTW static lib not found: {fftw_lib_file}")
            print(f"Using static FFTW from: {fftw_dir}")

            compiler = "cl"
            shared_flag = "/LD"  # Create DLL
            self.extra_compile = ["/O2", "/std:c11", "/experimental:c11atomics", "/openmp:experimental", "/MT"]
            self.extra_link = [str(fftw_lib_file)]
            lib_ext = ".dll"
            use_msvc = True

        # === Linux ===
        else:
            # Always use static FFTW from bundled libraries
            fftw_static_dir = self.pkg_dir / "static_fftw" / "linux"
            if not fftw_static_dir.exists():
                raise RuntimeError(f"Static FFTW not found: {fftw_static_dir}")

            self.fftw_inc = fftw_static_dir / "include"
            self.fftw_lib = fftw_static_dir / "lib"
            fftw_lib_file = self.fftw_lib / "libfftw3f.a"
            fftw_omp_file = self.fftw_lib / "libfftw3f_omp.a"
            if not fftw_lib_file.exists():
                raise RuntimeError(f"FFTW static lib not found: {fftw_lib_file}")
            print(f"Using static FFTW from: {fftw_static_dir}")

            compiler = os.environ.get("CC", "gcc")
            shared_flag = "-shared"
            self.extra_compile = ["-O3", "-fPIC", "-fopenmp", "-DFFTW_THREADS", f"-I{self.fftw_inc}", f"-I{self.python_include}"]
            # Link statically with FFTW .a files
            self.extra_link = ["-lm", "-fopenmp", str(fftw_lib_file), str(fftw_omp_file)]
            lib_ext = ".so"
            use_msvc = False

        # --- Build libbulkxcorr2d ---
        sources1 = [
            "peak_locate_lm.c",
            "PIV_2d_cross_correlate.c",
            "xcorr.c",
            "xcorr_cache.c",
        ]

        if use_msvc:
            # MSVC command structure: cl [flags] [sources] /I[include] /Fe[output] /link [libs]
            output_file = build_dir / f"libbulkxcorr2d{lib_ext}"
            cmd1 = [
                compiler, *self.extra_compile, shared_flag,
                f"/Fo{build_dir}/",
                *[str(src_dir / s) for s in sources1],
                f"/I{src_dir}", f"/I{self.fftw_inc}",
                f"/Fe{output_file}"
            ] + self.extra_link
        else:
            # GCC command structure: gcc [flags] [sources] -I[include] -o [output] [libs]
            cmd1 = [
                compiler, *self.extra_compile, shared_flag,
                *[str(src_dir / s) for s in sources1],
                f"-I{src_dir}", f"-I{self.fftw_inc}",
                "-o", str(build_dir / f"libbulkxcorr2d{lib_ext}")
            ] + self.extra_link
        self._run(cmd1)
        if not (build_dir / f"libbulkxcorr2d{lib_ext}").exists():
            raise RuntimeError(f"Build failed: {build_dir / f'libbulkxcorr2d{lib_ext}'} not created")

        # Clean up intermediate build files
        for pattern in ['*.obj', '*.exp', '*.lib']:
            for file in build_dir.glob(pattern):
                file.unlink()

        # --- Build libinterp2custom ---
        if use_msvc:
            # MSVC command structure
            output_file = build_dir / f"libinterp2custom{lib_ext}"
            cmd2 = [
                compiler, *self.extra_compile, shared_flag,
                f"/Fo{build_dir}/",
                str(src_dir / "interp2custom.c"),
                f"/I{src_dir}",
                f"/Fe{output_file}"
            ]
        else:
            # GCC command structure
            cmd2 = [
                compiler, *self.extra_compile, shared_flag,
                str(src_dir / "interp2custom.c"),
                f"-I{src_dir}",
                "-o", str(build_dir / f"libinterp2custom{lib_ext}")
            ]
        self._run(cmd2)
        if not (build_dir / f"libinterp2custom{lib_ext}").exists():
            raise RuntimeError(f"Build failed: {build_dir / f'libinterp2custom{lib_ext}'} not created")

        # Clean up intermediate build files
        for pattern in ['*.obj', '*.exp', '*.lib']:
            for file in build_dir.glob(pattern):
                file.unlink()

        # --- Build libmarquadt (for ensemble PIV) ---
        # Requires GSL (GNU Scientific Library)
        marquadt_src = src_dir / "marquadt_gaussian.c"
        if marquadt_src.exists():
            # Use static GSL from static_gsl folder
            if sys_name == "macos":
                arch = platform.machine().lower()
                if arch == "arm64":
                    gsl_dir = self.pkg_dir / "static_gsl" / "macos_arm64"
                else:
                    raise RuntimeError(f"Unsupported macOS architecture: {arch}. Only Apple Silicon (arm64) is supported.")
            elif sys_name == "windows":
                gsl_dir = self.pkg_dir / "static_gsl" / "windows"
            else:  # Linux
                gsl_dir = self.pkg_dir / "static_gsl" / "linux"

            if not gsl_dir.exists():
                raise RuntimeError(f"Static GSL not found: {gsl_dir}")

            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"

            if use_msvc:
                # MSVC style
                gsl_compile_flags = [f"/I{gsl_inc}"]
                gsl_link_flags = [str(gsl_lib / "gsl.lib"), str(gsl_lib / "gslcblas.lib")]
                output_file = build_dir / f"libmarquadt{lib_ext}"
                cmd_marquadt = [
                    compiler, *self.extra_compile, shared_flag,
                    f"/Fo{build_dir}/",
                    *gsl_compile_flags,
                    str(marquadt_src),
                    f"/I{src_dir}",
                    f"/Fe{output_file}",
                    *gsl_link_flags
                ]
            else:
                # GCC style
                gsl_compile_flags = [f"-I{gsl_inc}"]
                gsl_link_flags = [str(gsl_lib / "libgsl.a"), str(gsl_lib / "libgslcblas.a"), "-lm"]
                cmd_marquadt = [
                    compiler, *self.extra_compile, shared_flag,
                    *gsl_compile_flags,
                    str(marquadt_src),
                    f"-I{src_dir}",
                    "-o", str(build_dir / f"libmarquadt{lib_ext}"),
                    *gsl_link_flags
                ]

            try:
                self._run(cmd_marquadt)
                if (build_dir / f"libmarquadt{lib_ext}").exists():
                    print(f"Successfully built libmarquadt{lib_ext}")
                else:
                    print(f"WARNING: libmarquadt{lib_ext} build may have failed")
            except RuntimeError as e:
                print(f"WARNING: Failed to build libmarquadt: {e}")
                print("Ensemble PIV will not be available.")

        # Clean up intermediate build files
        for pattern in ['*.obj', '*.exp', '*.lib']:
            for file in build_dir.glob(pattern):
                file.unlink()

    def _run(self, cmd):
        print("RUN:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Build failed: {result.returncode}")


setup(
    packages=find_packages(),
    include_package_data=True,
    cmdclass={"build": BuildCLibraries},
    distclass=BinaryDistribution,
)
