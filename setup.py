# setup.py
import os
import platform
import pathlib
import subprocess
import sysconfig
from setuptools import find_packages, setup, Distribution
from setuptools.command.build import build
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

        # Check for user-provided FFTW paths (allows system library usage)
        fftw_inc_env = os.environ.get("FFTW_INC_DIR")
        fftw_lib_env = os.environ.get("FFTW_LIB_DIR")
        use_system_fftw = bool(fftw_inc_env and fftw_lib_env)

        # === macOS ===
        if sys_name == "darwin":
            sys_name = "macos"
            arch = platform.machine().lower()

            if use_system_fftw:
                # User-provided system FFTW
                self.fftw_inc = pathlib.Path(fftw_inc_env)
                self.fftw_lib = pathlib.Path(fftw_lib_env)
                print(f"Using system FFTW from: INC={self.fftw_inc}, LIB={self.fftw_lib}")
                use_static_fftw = False
            else:
                # Bundled static FFTW
                if arch == "arm64":
                    fftw_static_dir = self.pkg_dir / "static_fftw" / "macos_arm64"
                else:
                    raise RuntimeError(f"Unsupported macOS architecture: {arch}. Only arm64 is supported.")

                if not fftw_static_dir.exists():
                    raise RuntimeError(
                        f"Static FFTW not found: {fftw_static_dir}\n"
                        "Set FFTW_INC_DIR and FFTW_LIB_DIR to use system FFTW."
                    )
                self.fftw_inc = fftw_static_dir / "include"
                self.fftw_lib = fftw_static_dir / "lib"
                use_static_fftw = True
                print(f"Using static FFTW from: {fftw_static_dir}")

            # Compiler setup
            compiler = (
                shutil.which("gcc-15") or
                shutil.which("gcc-14") or
                shutil.which("gcc-13") or
                shutil.which("gcc")
            )
            if compiler is None or "/usr/bin/gcc" in str(compiler):
                raise RuntimeError("No suitable GCC compiler found. Install via: brew install gcc")
            print(f"Using compiler: {compiler}")

            sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
            print(f"Using SDK path: {sdk_path}")

            self.extra_compile = ["-O3", "-fPIC", "-fopenmp", "-DFFTW_THREADS",
                                  f"-I{self.fftw_inc}", "-isysroot", sdk_path]

            if use_static_fftw:
                fftw_lib_file = self.fftw_lib / "libfftw3f.a"
                fftw_omp_file = self.fftw_lib / "libfftw3f_omp.a"
                self.extra_link = ["-lm", "-fopenmp", str(fftw_lib_file), str(fftw_omp_file), "-isysroot", sdk_path]
            else:
                self.extra_link = ["-lm", "-fopenmp", f"-L{self.fftw_lib}",
                                   "-lfftw3f", "-lfftw3f_omp", "-isysroot", sdk_path]

            shared_flag = "-shared"
            lib_ext = ".so"
            use_msvc = False

        # === Windows ===
        elif sys_name == "windows":
            if use_system_fftw:
                # User-provided system FFTW
                self.fftw_inc = pathlib.Path(fftw_inc_env)
                self.fftw_lib = pathlib.Path(fftw_lib_env)
                print(f"Using system FFTW from: INC={self.fftw_inc}, LIB={self.fftw_lib}")
                use_static_fftw = False
            else:
                # Bundled static FFTW
                fftw_dir = self.pkg_dir / "static_fftw" / "windows"
                if not fftw_dir.exists():
                    raise RuntimeError(
                        f"Static FFTW not found: {fftw_dir}\n"
                        "Set FFTW_INC_DIR and FFTW_LIB_DIR to use system FFTW."
                    )
                self.fftw_inc = fftw_dir / "include"
                self.fftw_lib = fftw_dir / "lib"
                use_static_fftw = True
                print(f"Using static FFTW from: {fftw_dir}")

            compiler = "cl"
            shared_flag = "/LD"
            self.extra_compile = ["/O2", "/std:c11", "/experimental:c11atomics", "/openmp:experimental", "/MT"]

            if use_static_fftw:
                fftw_lib_file = self.fftw_lib / "libfftw3f-3.lib"
                self.extra_link = [str(fftw_lib_file)]
            else:
                # Dynamic linking - user must have fftw3f.lib available
                self.extra_link = [f"/LIBPATH:{self.fftw_lib}", "fftw3f.lib"]

            lib_ext = ".dll"
            use_msvc = True

        # === Linux ===
        else:
            if use_system_fftw:
                # User-provided system FFTW (dynamic linking)
                self.fftw_inc = pathlib.Path(fftw_inc_env)
                self.fftw_lib = pathlib.Path(fftw_lib_env)
                print(f"Using system FFTW from: INC={self.fftw_inc}, LIB={self.fftw_lib}")
                use_static_fftw = False
            else:
                # Bundled static FFTW
                fftw_static_dir = self.pkg_dir / "static_fftw" / "linux"
                if not fftw_static_dir.exists():
                    raise RuntimeError(
                        f"Static FFTW not found: {fftw_static_dir}\n"
                        "Either set FFTW_INC_DIR and FFTW_LIB_DIR environment variables,\n"
                        "or ensure static_fftw/linux directory exists."
                    )
                self.fftw_inc = fftw_static_dir / "include"
                self.fftw_lib = fftw_static_dir / "lib"
                use_static_fftw = True
                print(f"Using static FFTW from: {fftw_static_dir}")

            compiler = os.environ.get("CC", "gcc")
            shared_flag = "-shared"
            self.extra_compile = ["-O3", "-fPIC", "-fopenmp", "-DFFTW_THREADS",
                                  f"-I{self.fftw_inc}", f"-I{self.python_include}"]

            if use_static_fftw:
                fftw_lib_file = self.fftw_lib / "libfftw3f.a"
                fftw_omp_file = self.fftw_lib / "libfftw3f_omp.a"
                self.extra_link = ["-lm", "-fopenmp", str(fftw_lib_file), str(fftw_omp_file)]
            else:
                # Dynamic linking with system FFTW
                self.extra_link = ["-lm", "-fopenmp", f"-L{self.fftw_lib}",
                                   "-lfftw3f", "-lfftw3f_omp"]

            lib_ext = ".so"
            use_msvc = False

        # Store for marquadt build
        self.use_static_fftw = use_static_fftw
        self.use_msvc = use_msvc
        self.compiler = compiler
        self.shared_flag = shared_flag
        self.lib_ext = lib_ext
        self.sys_name = sys_name
        self.build_dir = build_dir
        self.src_dir = src_dir

        # --- Build libbulkxcorr2d ---
        sources1 = [
            "peak_locate_lm.c",
            "PIV_2d_cross_correlate.c",
            "xcorr.c",
            "xcorr_cache.c",
        ]

        if use_msvc:
            output_file = build_dir / f"libbulkxcorr2d{lib_ext}"
            cmd1 = [
                compiler, *self.extra_compile, shared_flag,
                f"/Fo{build_dir}/",
                *[str(src_dir / s) for s in sources1],
                f"/I{src_dir}", f"/I{self.fftw_inc}",
                f"/Fe{output_file}"
            ] + self.extra_link
        else:
            cmd1 = [
                compiler, *self.extra_compile, shared_flag,
                *[str(src_dir / s) for s in sources1],
                f"-I{src_dir}", f"-I{self.fftw_inc}",
                "-o", str(build_dir / f"libbulkxcorr2d{lib_ext}")
            ] + self.extra_link

        self._run(cmd1)
        if not (build_dir / f"libbulkxcorr2d{lib_ext}").exists():
            raise RuntimeError(f"Build failed: {build_dir / f'libbulkxcorr2d{lib_ext}'} not created")

        self._cleanup_intermediates(build_dir)

        # --- Build libfusedwarp (no FFTW/GSL deps — only OpenMP + math) ---
        if use_msvc:
            output_file = build_dir / f"libfusedwarp{lib_ext}"
            cmd_fw = [
                compiler, *self.extra_compile, shared_flag,
                f"/Fo{build_dir}/",
                str(src_dir / "fused_warp.c"),
                f"/I{src_dir}",
                f"/Fe{output_file}"
            ]
        else:
            cmd_fw = [
                compiler, *self.extra_compile, shared_flag,
                str(src_dir / "fused_warp.c"),
                f"-I{src_dir}",
                "-o", str(build_dir / f"libfusedwarp{lib_ext}"),
                "-lm", "-fopenmp"
            ]
        self._run(cmd_fw)
        if not (build_dir / f"libfusedwarp{lib_ext}").exists():
            raise RuntimeError(f"Build failed: libfusedwarp{lib_ext} not created")
        print(f"Successfully built libfusedwarp{lib_ext}")

        self._cleanup_intermediates(build_dir)

        # --- Build libmarquadt (for ensemble PIV) ---
        self._build_marquadt()

        # --- Build libkspace (k-space fitting, requires GSL + FFTW) ---
        self._build_kspace()

        # --- Build libstereo_coc (stereo CoC accumulation, requires FFTW) ---
        self._build_stereo_coc()

        # --- Build libkspace_coc (CoC spread fitting, requires GSL + FFTW) ---
        self._build_kspace_coc()

    def _build_marquadt(self):
        """Build libmarquadt with GSL support."""
        marquadt_src = self.src_dir / "marquadt_gaussian.c"
        if not marquadt_src.exists():
            raise RuntimeError(f"marquadt source not found: {marquadt_src}")

        print("Building libmarquadt (requires GSL)")

        # Check for user-provided GSL
        gsl_dir_env = os.environ.get("GSL_DIR")

        if gsl_dir_env:
            gsl_dir = pathlib.Path(gsl_dir_env)
            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"
            use_static_gsl = False
            print(f"Using system GSL from: {gsl_dir}")
        else:
            # Bundled static GSL
            if self.sys_name == "macos":
                arch = platform.machine().lower()
                if arch == "arm64":
                    gsl_dir = self.pkg_dir / "static_gsl" / "macos_arm64"
                else:
                    raise RuntimeError(f"Unsupported macOS architecture: {arch}")
            elif self.sys_name == "windows":
                gsl_dir = self.pkg_dir / "static_gsl" / "windows"
            else:  # Linux
                gsl_dir = self.pkg_dir / "static_gsl" / "linux"

            if not gsl_dir.exists():
                raise RuntimeError(
                    f"Static GSL not found: {gsl_dir}\n"
                    "Set GSL_DIR environment variable to use system GSL."
                )

            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"
            use_static_gsl = True
            print(f"Using static GSL from: {gsl_dir}")

        if self.use_msvc:
            # MSVC style
            gsl_compile_flags = [f"/I{gsl_inc}"]
            if use_static_gsl:
                gsl_link_flags = [str(gsl_lib / "gsl.lib"), str(gsl_lib / "gslcblas.lib")]
            else:
                gsl_link_flags = [f"/LIBPATH:{gsl_lib}", "gsl.lib", "gslcblas.lib"]

            output_file = self.build_dir / f"libmarquadt{self.lib_ext}"
            cmd_marquadt = [
                self.compiler, *self.extra_compile, self.shared_flag,
                f"/Fo{self.build_dir}/",
                *gsl_compile_flags,
                str(marquadt_src),
                f"/I{self.src_dir}",
                f"/Fe{output_file}",
                *gsl_link_flags
            ]
        else:
            # GCC style
            gsl_compile_flags = [f"-I{gsl_inc}"]
            if use_static_gsl:
                # Static linking - include -fopenmp for OpenMP support
                gsl_link_flags = [str(gsl_lib / "libgsl.a"), str(gsl_lib / "libgslcblas.a"),
                                  "-lm", "-fopenmp"]
            else:
                # Dynamic linking
                gsl_link_flags = [f"-L{gsl_lib}", "-lgsl", "-lgslcblas", "-lm", "-fopenmp"]

            cmd_marquadt = [
                self.compiler, *self.extra_compile, self.shared_flag,
                *gsl_compile_flags,
                str(marquadt_src),
                f"-I{self.src_dir}",
                "-o", str(self.build_dir / f"libmarquadt{self.lib_ext}"),
                *gsl_link_flags
            ]

        self._run(cmd_marquadt)
        if not (self.build_dir / f"libmarquadt{self.lib_ext}").exists():
            raise RuntimeError(f"Build failed: {self.build_dir / f'libmarquadt{self.lib_ext}'} not created")
        print(f"Successfully built libmarquadt{self.lib_ext}")

        self._cleanup_intermediates(self.build_dir)

    def _build_kspace(self):
        """Build libkspace with GSL + FFTW support for k-space fitting."""
        kspace_src = self.src_dir / "kspace_fitting.c"
        if not kspace_src.exists():
            raise RuntimeError(f"kspace source not found: {kspace_src}")

        print("Building libkspace (requires GSL + FFTW)")

        # Check for user-provided GSL
        gsl_dir_env = os.environ.get("GSL_DIR")

        if gsl_dir_env:
            gsl_dir = pathlib.Path(gsl_dir_env)
            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"
            use_static_gsl = False
        else:
            if self.sys_name == "macos":
                arch = platform.machine().lower()
                if arch == "arm64":
                    gsl_dir = self.pkg_dir / "static_gsl" / "macos_arm64"
                else:
                    raise RuntimeError(f"Unsupported macOS architecture: {arch}")
            elif self.sys_name == "windows":
                gsl_dir = self.pkg_dir / "static_gsl" / "windows"
            else:
                gsl_dir = self.pkg_dir / "static_gsl" / "linux"

            if not gsl_dir.exists():
                raise RuntimeError(f"Static GSL not found: {gsl_dir}")

            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"
            use_static_gsl = True

        if self.use_msvc:
            gsl_compile_flags = [f"/I{gsl_inc}"]
            if use_static_gsl:
                gsl_link_flags = [str(gsl_lib / "gsl.lib"), str(gsl_lib / "gslcblas.lib")]
            else:
                gsl_link_flags = [f"/LIBPATH:{gsl_lib}", "gsl.lib", "gslcblas.lib"]

            output_file = self.build_dir / f"libkspace{self.lib_ext}"
            cmd_kspace = [
                self.compiler, *self.extra_compile, self.shared_flag,
                f"/Fo{self.build_dir}/",
                *gsl_compile_flags,
                f"/I{self.fftw_inc}",
                str(kspace_src),
                f"/I{self.src_dir}",
                f"/Fe{output_file}",
                *gsl_link_flags,
                *self.extra_link
            ]
        else:
            gsl_compile_flags = [f"-I{gsl_inc}"]
            if use_static_gsl:
                gsl_link_flags = [str(gsl_lib / "libgsl.a"), str(gsl_lib / "libgslcblas.a")]
            else:
                gsl_link_flags = [f"-L{gsl_lib}", "-lgsl", "-lgslcblas"]

            cmd_kspace = [
                self.compiler, *self.extra_compile, self.shared_flag,
                *gsl_compile_flags,
                str(kspace_src),
                f"-I{self.src_dir}",
                "-o", str(self.build_dir / f"libkspace{self.lib_ext}"),
                *gsl_link_flags,
                *self.extra_link
            ]

        self._run(cmd_kspace)
        if not (self.build_dir / f"libkspace{self.lib_ext}").exists():
            raise RuntimeError(f"Build failed: libkspace{self.lib_ext} not created")
        print(f"Successfully built libkspace{self.lib_ext}")

        self._cleanup_intermediates(self.build_dir)

    def _build_stereo_coc(self):
        """Build libstereo_coc for stereo ensemble CoC accumulation (FFTW only)."""
        stereo_src = self.src_dir / "stereo_coc_accumulate.c"
        xcorr_src = self.src_dir / "xcorr.c"
        cache_src = self.src_dir / "xcorr_cache.c"
        if not stereo_src.exists():
            raise RuntimeError(f"stereo_coc source not found: {stereo_src}")

        print("Building libstereo_coc (requires FFTW)")

        sources = [str(stereo_src), str(xcorr_src), str(cache_src)]

        if self.use_msvc:
            output_file = self.build_dir / f"libstereo_coc{self.lib_ext}"
            cmd = [
                self.compiler, *self.extra_compile, self.shared_flag,
                f"/Fo{self.build_dir}/",
                *sources,
                f"/I{self.src_dir}",
                f"/I{self.fftw_inc}",
                f"/Fe{output_file}",
                *self.extra_link
            ]
        else:
            cmd = [
                self.compiler, *self.extra_compile, self.shared_flag,
                *sources,
                f"-I{self.src_dir}",
                "-o", str(self.build_dir / f"libstereo_coc{self.lib_ext}"),
                *self.extra_link
            ]

        self._run(cmd)
        if not (self.build_dir / f"libstereo_coc{self.lib_ext}").exists():
            raise RuntimeError(f"Build failed: libstereo_coc{self.lib_ext} not created")
        print(f"Successfully built libstereo_coc{self.lib_ext}")
        self._cleanup_intermediates(self.build_dir)

    def _build_kspace_coc(self):
        """Build libkspace_coc for CoC spread fitting (GSL + FFTW)."""
        kspace_coc_src = self.src_dir / "kspace_coc_fitting.c"
        if not kspace_coc_src.exists():
            raise RuntimeError(f"kspace_coc source not found: {kspace_coc_src}")

        print("Building libkspace_coc (requires GSL + FFTW)")

        # GSL setup (same as _build_kspace)
        gsl_dir_env = os.environ.get("GSL_DIR")

        if gsl_dir_env:
            gsl_dir = pathlib.Path(gsl_dir_env)
            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"
            use_static_gsl = False
        else:
            if self.sys_name == "macos":
                arch = platform.machine().lower()
                if arch == "arm64":
                    gsl_dir = self.pkg_dir / "static_gsl" / "macos_arm64"
                else:
                    raise RuntimeError(f"Unsupported macOS architecture: {arch}")
            elif self.sys_name == "windows":
                gsl_dir = self.pkg_dir / "static_gsl" / "windows"
            else:
                gsl_dir = self.pkg_dir / "static_gsl" / "linux"

            if not gsl_dir.exists():
                raise RuntimeError(f"Static GSL not found: {gsl_dir}")

            gsl_inc = gsl_dir / "include"
            gsl_lib = gsl_dir / "lib"
            use_static_gsl = True

        if self.use_msvc:
            gsl_compile_flags = [f"/I{gsl_inc}"]
            if use_static_gsl:
                gsl_link_flags = [str(gsl_lib / "gsl.lib"), str(gsl_lib / "gslcblas.lib")]
            else:
                gsl_link_flags = [f"/LIBPATH:{gsl_lib}", "gsl.lib", "gslcblas.lib"]

            output_file = self.build_dir / f"libkspace_coc{self.lib_ext}"
            cmd = [
                self.compiler, *self.extra_compile, self.shared_flag,
                f"/Fo{self.build_dir}/",
                *gsl_compile_flags,
                f"/I{self.fftw_inc}",
                str(kspace_coc_src),
                f"/I{self.src_dir}",
                f"/Fe{output_file}",
                *gsl_link_flags,
                *self.extra_link
            ]
        else:
            gsl_compile_flags = [f"-I{gsl_inc}"]
            if use_static_gsl:
                gsl_link_flags = [str(gsl_lib / "libgsl.a"), str(gsl_lib / "libgslcblas.a")]
            else:
                gsl_link_flags = [f"-L{gsl_lib}", "-lgsl", "-lgslcblas"]

            cmd = [
                self.compiler, *self.extra_compile, self.shared_flag,
                *gsl_compile_flags,
                str(kspace_coc_src),
                f"-I{self.src_dir}",
                "-o", str(self.build_dir / f"libkspace_coc{self.lib_ext}"),
                *gsl_link_flags,
                *self.extra_link
            ]

        self._run(cmd)
        if not (self.build_dir / f"libkspace_coc{self.lib_ext}").exists():
            raise RuntimeError(f"Build failed: libkspace_coc{self.lib_ext} not created")
        print(f"Successfully built libkspace_coc{self.lib_ext}")
        self._cleanup_intermediates(self.build_dir)

    def _cleanup_intermediates(self, build_dir):
        """Clean up intermediate build files."""
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
