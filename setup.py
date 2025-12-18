# setup.py
import os
import platform
import pathlib
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil
import sysconfig


class BuildCLib(build_ext):
    def run(self):
        self.python_include = sysconfig.get_path("include")
        self.pkg_dir = pathlib.Path(__file__).parent
        if not self.dry_run:
            self.build_c_libraries()

        dummy_ext = Extension(
            "pivtools_cli._build_marker",
            sources=[str(pathlib.Path(__file__).parent / "pivtools_cli" / "_build_marker.c")],
            include_dirs=[self.python_include, str(self.pkg_dir / "pivtools_cli" / "lib"), str(self.fftw_inc)],
            extra_compile_args=self.extra_compile,
            extra_link_args=self.extra_link,
            language="c",
        )
        self.extensions = [dummy_ext]
        super().run()

    def build_c_libraries(self):

        build_dir = self.pkg_dir / "pivtools_cli" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)
        src_dir = self.pkg_dir / "pivtools_cli" / "lib"
        self.fftw_inc = os.environ.get("FFTW_INC_DIR")
        self.fftw_lib = os.environ.get("FFTW_LIB_DIR")
        sys_name = platform.system().lower()
        static_root = self.pkg_dir / "static_fftw"

        
        # === macOS ===
        if sys_name == "darwin":
            sys_name = "macos"
            arch = platform.machine().lower()

            if self.fftw_inc and self.fftw_lib:
                self.fftw_inc = pathlib.Path(self.fftw_inc)
                self.fftw_lib = pathlib.Path(self.fftw_lib)
            else:
                try:
                    brew_prefix = subprocess.run(["brew", "--prefix", "fftw"], capture_output=True, text=True).stdout.strip()
                except Exception:
                    brew_prefix = ""

                if brew_prefix:
                    self.fftw_inc = pathlib.Path(brew_prefix) / "include"
                    self.fftw_lib = pathlib.Path(brew_prefix) / "lib"
                else:
                    if arch == "arm64":
                        static_root = self.pkg_dir / "static_fftw" / "macos_arm64"
                        self.fftw_inc = static_root / "include"
                        self.fftw_lib = static_root / "lib"
                        fftw_lib_file = self.fftw_lib / "libfftw3f.a"
                        if not fftw_lib_file.exists():
                            raise RuntimeError(f"FFTW static lib not found: {fftw_lib_file}")
                    else:
                        raise RuntimeError(f"No FFTW found for macOS {arch}")

            compiler = shutil.which("clang") or "/opt/homebrew/opt/llvm/bin/clang"
            if compiler is None:
                raise RuntimeError("No suitable compiler found (clang on macOS).")

            sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            self.extra_compile = ["-g","-O0", "-fPIC", "-fopenmp", f"-I{self.fftw_inc}", "-isysroot", sdk_path]
            self.extra_link = [ "-lm", "-fopenmp", "-L" + str(self.fftw_lib), "-lfftw3f", "-lfftw3f_threads", "-isysroot", sdk_path]
            shared_flag = "-shared"

            lib_ext = ".so"
            use_msvc = False

        # === Windows ===
        elif sys_name == "windows":
            fftw_dir = static_root / "windows"
            if not fftw_dir.exists():
                raise RuntimeError(f"Static FFTW not found: {fftw_dir}")

            self.fftw_inc = fftw_dir / "include"
            self.fftw_lib = fftw_dir / "lib"
            fftw_lib_file = self.fftw_lib / "libfftw3f-3.lib"
            if not fftw_lib_file.exists():
                raise RuntimeError(f"FFTW static lib not found: {fftw_lib_file}")

            compiler = "cl"
            shared_flag = "/LD"  # Create DLL
            extra_compile = ["/O2", "/openmp:experimental", "/MT"]
            extra_link = [str(fftw_lib_file)]
            lib_ext = ".dll"
            use_msvc = True

        # === Linux ===
        else:
            if self.fftw_inc and self.fftw_lib:
                self.fftw_inc = pathlib.Path(self.fftw_inc)
                self.fftw_lib = pathlib.Path(self.fftw_lib)

            else:
                fftw_dir = static_root / "linux"
                if not fftw_dir.exists():
                    raise RuntimeError(f"Static FFTW not found: {fftw_dir}")

                self.fftw_inc = fftw_dir / "include"
                self.fftw_lib = fftw_dir / "lib"
            #fftw_lib_file = fftw_lib / "libfftw3f.so"
            #if not fftw_lib_file.exists():
            #    raise RuntimeError(f"FFTW static lib not found: {fftw_lib_file}")

            compiler = os.environ.get("CC", "gcc")
            shared_flag = "-shared"
            extra_compile = ["-g","-O0", "-fPIC", "-fopenmp"]
            extra_link = [ "-lm", "-fopenmp"]
            lib_ext = ".so"
            extra_link += [f"-L{self.fftw_lib}", "-lfftw3f", "-lfftw3f_threads"]
            extra_compile += [f"-I{self.fftw_inc}"]
            extra_compile += [ f"-I{self.python_include}"]
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
            #self.pkg_dir = pathlib.Path(__file__).parent
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

dummy_ext = Extension("pivtools_cli._build_marker", sources=["pivtools_cli/_build_marker.c"])
setup(ext_modules=[dummy_ext],
    cmdclass={"build_ext": BuildCLib},
    include_package_data=True,
    packages=["pivtools_cli"],  # list your packages
)
