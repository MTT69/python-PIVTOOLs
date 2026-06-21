# setup.py
import os
import pathlib
import platform
import shutil
import subprocess
import sysconfig

from setuptools import Distribution, find_packages, setup
from setuptools.command.build import build


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
                print(
                    f"Using system FFTW from: INC={self.fftw_inc}, LIB={self.fftw_lib}"
                )
                use_static_fftw = False
            else:
                # Bundled static FFTW
                if arch == "arm64":
                    fftw_static_dir = self.pkg_dir / "static_fftw" / "macos_arm64"
                else:
                    raise RuntimeError(
                        f"Unsupported macOS architecture: {arch}. Only arm64 is supported."
                    )

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
                shutil.which("gcc-15")
                or shutil.which("gcc-14")
                or shutil.which("gcc-13")
                or shutil.which("gcc")
            )
            if compiler is None or "/usr/bin/gcc" in str(compiler):
                raise RuntimeError(
                    "No suitable GCC compiler found. Install via: brew install gcc"
                )
            print(f"Using compiler: {compiler}")

            sdk_path = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True
            ).strip()
            print(f"Using SDK path: {sdk_path}")

            self.extra_compile = [
                "-O3",
                "-fPIC",
                "-fopenmp",
                "-DFFTW_THREADS",
                f"-I{self.fftw_inc}",
                "-isysroot",
                sdk_path,
            ]

            if use_static_fftw:
                fftw_lib_file = self.fftw_lib / "libfftw3f.a"
                fftw_omp_file = self.fftw_lib / "libfftw3f_omp.a"
                self.extra_link = [
                    "-lm",
                    "-fopenmp",
                    str(fftw_lib_file),
                    str(fftw_omp_file),
                    "-isysroot",
                    sdk_path,
                ]
            else:
                self.extra_link = [
                    "-lm",
                    "-fopenmp",
                    f"-L{self.fftw_lib}",
                    "-lfftw3f",
                    "-lfftw3f_omp",
                    "-isysroot",
                    sdk_path,
                ]

            shared_flag = "-shared"
            lib_ext = ".so"
            use_msvc = False

        # === Windows ===
        elif sys_name == "windows":
            if use_system_fftw:
                # User-provided system FFTW
                self.fftw_inc = pathlib.Path(fftw_inc_env)
                self.fftw_lib = pathlib.Path(fftw_lib_env)
                print(
                    f"Using system FFTW from: INC={self.fftw_inc}, LIB={self.fftw_lib}"
                )
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
            self.extra_compile = [
                "/O2",
                "/std:c11",
                "/experimental:c11atomics",
                "/openmp:experimental",
                "/MT",
            ]

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
                print(
                    f"Using system FFTW from: INC={self.fftw_inc}, LIB={self.fftw_lib}"
                )
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
            self.extra_compile = [
                "-O3",
                "-fPIC",
                "-fopenmp",
                "-DFFTW_THREADS",
                f"-I{self.fftw_inc}",
                f"-I{self.python_include}",
            ]

            if use_static_fftw:
                fftw_lib_file = self.fftw_lib / "libfftw3f.a"
                fftw_omp_file = self.fftw_lib / "libfftw3f_omp.a"
                self.extra_link = [
                    "-lm",
                    "-fopenmp",
                    str(fftw_lib_file),
                    str(fftw_omp_file),
                ]
            else:
                # Dynamic linking with system FFTW
                self.extra_link = [
                    "-lm",
                    "-fopenmp",
                    f"-L{self.fftw_lib}",
                    "-lfftw3f",
                    "-lfftw3f_omp",
                ]

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

        # Resolve the Stage B SIMD width + flags for libbulkxcorr2d.
        self._resolve_fft_isa()

        # --- Build libbulkxcorr2d (FFTW-free: permissive codelet FFT) ---
        # Generate the fixed-size FFT codelet header for the supported window
        # sizes, then compile. This library no longer links FFTW (the codelet
        # engine replaces it); libkspace below still uses FFTW.
        self._generate_codelets()

        sources1 = [
            "peak_locate_lm.c",
            "PIV_2d_cross_correlate.c",
            "xcorr.c",
            "codelet_fft.c",
        ]

        # Strip FFTW from this library's flags (kept on extra_* for libkspace).
        fftw_tokens = {
            f"-I{self.fftw_inc}",
            "-DFFTW_THREADS",
            f"-L{self.fftw_lib}",
            "-lfftw3f",
            "-lfftw3f_omp",
            str(self.fftw_lib / "libfftw3f.a"),
            str(self.fftw_lib / "libfftw3f_omp.a"),
            str(self.fftw_lib / "libfftw3f-3.lib"),
        }
        bulk_compile = [f for f in self.extra_compile if f not in fftw_tokens]
        bulk_link = [f for f in self.extra_link if f not in fftw_tokens]

        # Stage B: append the SIMD width macro + arch + accuracy-preserving
        # throughput flags. These apply to libbulkxcorr2d ONLY (not the FFTW/GSL
        # libs). NO -ffast-math / /fp:fast -- they reassociate/flush and would
        # break the FFTW-parity the codelet engine is validated against.
        lto = os.environ.get("PIVTOOLS_FFT_LTO", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if use_msvc:
            simd_flags = [
                f"/D{self.fft_macro}",
                *self.fft_arch_flags,
                "/Ot",
                "/fp:precise",
            ]
            if lto:
                simd_flags += ["/GL"]
        else:
            simd_flags = [
                f"-D{self.fft_macro}",
                *self.fft_arch_flags,
                "-funroll-loops",
                "-fno-math-errno",
                "-ffp-contract=fast",
            ]
            if lto:
                simd_flags += ["-flto"]
        bulk_compile = bulk_compile + simd_flags
        print(
            f">>> [PIVTOOLS_FFT] libbulkxcorr2d  isa={self.fft_isa}  lanes={self.fft_lanes}  "
            f"macro={self.fft_macro}  render={self.fft_render}  "
            f"arch='{' '.join(self.fft_arch_flags)}'  lto={lto}"
        )
        print(f">>> [PIVTOOLS_FFT] bulk compile flags: {' '.join(bulk_compile)}")

        if use_msvc:
            output_file = build_dir / f"libbulkxcorr2d{lib_ext}"
            cmd1 = [
                compiler,
                *bulk_compile,
                shared_flag,
                f"/Fo{build_dir}/",
                *[str(src_dir / s) for s in sources1],
                f"/I{src_dir}",
                f"/Fe{output_file}",
            ] + bulk_link
        else:
            cmd1 = [
                compiler,
                *bulk_compile,
                shared_flag,
                *[str(src_dir / s) for s in sources1],
                f"-I{src_dir}",
                "-o",
                str(build_dir / f"libbulkxcorr2d{lib_ext}"),
            ] + bulk_link

        self._run(cmd1)
        if not (build_dir / f"libbulkxcorr2d{lib_ext}").exists():
            raise RuntimeError(
                f"Build failed: {build_dir / f'libbulkxcorr2d{lib_ext}'} not created"
            )

        self._cleanup_intermediates(build_dir)

        # --- Build libfusedwarp (no FFTW/GSL deps — only OpenMP + math) ---
        if use_msvc:
            output_file = build_dir / f"libfusedwarp{lib_ext}"
            cmd_fw = [
                compiler,
                *self.extra_compile,
                shared_flag,
                f"/Fo{build_dir}/",
                str(src_dir / "fused_warp.c"),
                f"/I{src_dir}",
                f"/Fe{output_file}",
            ]
        else:
            cmd_fw = [
                compiler,
                *self.extra_compile,
                shared_flag,
                str(src_dir / "fused_warp.c"),
                f"-I{src_dir}",
                "-o",
                str(build_dir / f"libfusedwarp{lib_ext}"),
                "-lm",
                "-fopenmp",
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
                gsl_link_flags = [
                    str(gsl_lib / "gsl.lib"),
                    str(gsl_lib / "gslcblas.lib"),
                ]
            else:
                gsl_link_flags = [f"/LIBPATH:{gsl_lib}", "gsl.lib", "gslcblas.lib"]

            output_file = self.build_dir / f"libmarquadt{self.lib_ext}"
            cmd_marquadt = [
                self.compiler,
                *self.extra_compile,
                self.shared_flag,
                f"/Fo{self.build_dir}/",
                *gsl_compile_flags,
                str(marquadt_src),
                f"/I{self.src_dir}",
                f"/Fe{output_file}",
                *gsl_link_flags,
            ]
        else:
            # GCC style
            gsl_compile_flags = [f"-I{gsl_inc}"]
            if use_static_gsl:
                # Static linking - include -fopenmp for OpenMP support
                gsl_link_flags = [
                    str(gsl_lib / "libgsl.a"),
                    str(gsl_lib / "libgslcblas.a"),
                    "-lm",
                    "-fopenmp",
                ]
            else:
                # Dynamic linking
                gsl_link_flags = [
                    f"-L{gsl_lib}",
                    "-lgsl",
                    "-lgslcblas",
                    "-lm",
                    "-fopenmp",
                ]

            cmd_marquadt = [
                self.compiler,
                *self.extra_compile,
                self.shared_flag,
                *gsl_compile_flags,
                str(marquadt_src),
                f"-I{self.src_dir}",
                "-o",
                str(self.build_dir / f"libmarquadt{self.lib_ext}"),
                *gsl_link_flags,
            ]

        self._run(cmd_marquadt)
        if not (self.build_dir / f"libmarquadt{self.lib_ext}").exists():
            raise RuntimeError(
                f"Build failed: {self.build_dir / f'libmarquadt{self.lib_ext}'} not created"
            )
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
                gsl_link_flags = [
                    str(gsl_lib / "gsl.lib"),
                    str(gsl_lib / "gslcblas.lib"),
                ]
            else:
                gsl_link_flags = [f"/LIBPATH:{gsl_lib}", "gsl.lib", "gslcblas.lib"]

            output_file = self.build_dir / f"libkspace{self.lib_ext}"
            cmd_kspace = [
                self.compiler,
                *self.extra_compile,
                self.shared_flag,
                f"/Fo{self.build_dir}/",
                *gsl_compile_flags,
                f"/I{self.fftw_inc}",
                str(kspace_src),
                f"/I{self.src_dir}",
                f"/Fe{output_file}",
                *gsl_link_flags,
                *self.extra_link,
            ]
        else:
            gsl_compile_flags = [f"-I{gsl_inc}"]
            if use_static_gsl:
                gsl_link_flags = [
                    str(gsl_lib / "libgsl.a"),
                    str(gsl_lib / "libgslcblas.a"),
                ]
            else:
                gsl_link_flags = [f"-L{gsl_lib}", "-lgsl", "-lgslcblas"]

            cmd_kspace = [
                self.compiler,
                *self.extra_compile,
                self.shared_flag,
                *gsl_compile_flags,
                str(kspace_src),
                f"-I{self.src_dir}",
                "-o",
                str(self.build_dir / f"libkspace{self.lib_ext}"),
                *gsl_link_flags,
                *self.extra_link,
            ]

        self._run(cmd_kspace)
        if not (self.build_dir / f"libkspace{self.lib_ext}").exists():
            raise RuntimeError(f"Build failed: libkspace{self.lib_ext} not created")
        print(f"Successfully built libkspace{self.lib_ext}")

        self._cleanup_intermediates(self.build_dir)

    def _built_fft_sizes(self):
        """Load the supported window sizes from the single source of truth,
        without importing the (heavy) pivtools_core package."""
        import importlib.util

        mod_path = self.pkg_dir / "pivtools_core" / "fft_sizes.py"
        spec = importlib.util.spec_from_file_location("pivtools_fft_sizes", mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.BUILT_FFT_SIZES)

    def _resolve_fft_isa(self):
        """Resolve the Stage B SIMD lane width + compile flags for the codelet
        engine in libbulkxcorr2d. Platform default, overridable by env:

          PIVTOOLS_FFT_ISA   one of {neon4, vext8, avx512, avx2} -- force width
          PIVTOOLS_FFT_MARCH arch flag string, replaces the default verbatim
                             (e.g. '-march=icelake-server' or '-march=skylake-avx512')
          PIVTOOLS_FFT_LTO   '1' to enable link-time optimisation (off by default)

        scalar = Stage A (no batching); build the main branch for that.
        """
        arch = platform.machine().lower()
        env_isa = os.environ.get("PIVTOOLS_FFT_ISA", "").strip().lower()

        if self.use_msvc:
            default = "avx2"
        elif self.sys_name == "macos" or arch in ("arm64", "aarch64"):
            default = "neon4"
        else:  # generic x86_64 Linux -- AVX2 via vector_size (HPC opts into avx512)
            default = "vext8"
        isa = env_isa or default

        # isa -> (-D macro, gen_codelet render, lane count). Arch flags are
        # decided separately below: the vector_size renders are ISA-agnostic in
        # source, so the arch flag (not the width) chooses NEON vs AVX lowering.
        table = {
            "neon4": ("PIVTOOLS_FFT_ISA_NEON4", "vecext4", 4),
            "vext8": ("PIVTOOLS_FFT_ISA_VEXT8", "vecext", 8),
            "avx512": ("PIVTOOLS_FFT_ISA_AVX512", "vecext16", 16),
            "avx2": ("PIVTOOLS_FFT_ISA_AVX2", "avx2", 8),
        }
        if isa not in table:
            raise RuntimeError(
                f"PIVTOOLS_FFT_ISA='{isa}' invalid. Choose one of: neon4, vext8, avx512, avx2.\n"
                "(scalar = Stage A; build the main branch for that.)"
            )

        # MSVC has no GCC vector_size; it must take the avx2 intrinsic render.
        # Conversely the avx2 render is intrinsic-only and won't build under GCC.
        if self.use_msvc and isa != "avx2":
            raise RuntimeError(
                f"On Windows/MSVC only PIVTOOLS_FFT_ISA=avx2 is supported (got '{isa}')."
            )
        if (not self.use_msvc) and isa == "avx2":
            raise RuntimeError(
                "PIVTOOLS_FFT_ISA=avx2 is the MSVC intrinsic render; on GCC/Clang use "
                "vext8 (8-wide AVX2 via vector_size) or avx512."
            )

        macro, render, lanes = table[isa]

        # Platform-driven arch/tuning flags. On arm64 every vector_size width
        # lowers to NEON (vecext4/vecext/vecext16 -> 1/2/4 NEON registers), so
        # -mcpu=native is correct for ALL widths there -- x86 AVX flags would be
        # rejected by gcc. On x86_64 the flag must enable the matching ISA.
        is_arm = arch in ("arm64", "aarch64")
        if self.use_msvc:
            arch_flags = ["/arch:AVX2"]
        elif is_arm:
            arch_flags = ["-mcpu=native"]
        else:  # x86_64 / other GCC
            if isa == "avx512":
                arch_flags = ["-march=native"]
            elif isa == "neon4":
                arch_flags = ["-msse4.2"]  # 4-wide via SSE
            else:  # vext8
                arch_flags = ["-mavx2", "-mfma"]

        march_env = os.environ.get("PIVTOOLS_FFT_MARCH", "").strip()
        if march_env:
            arch_flags = march_env.split()

        if any("native" in f for f in arch_flags):
            print(
                "NOTICE [PIVTOOLS_FFT]: arch flag uses 'native' -> tuned for THIS build host.\n"
                "        On an HPC cluster the login node often differs from the compute nodes.\n"
                "        If the binary SIGILLs or underperforms there, build on a compute node\n"
                "        (interactive job) or set PIVTOOLS_FFT_MARCH explicitly, e.g.\n"
                "        PIVTOOLS_FFT_MARCH='-march=icelake-server'."
            )

        self.fft_isa = isa
        self.fft_macro = macro
        self.fft_render = render
        self.fft_arch_flags = arch_flags
        self.fft_lanes = lanes

    def _generate_codelets(self):
        """Generate codelets_gen.h (the unrolled fixed-size FFTs) for the
        supported window sizes. Emits the scalar render (the remainder/"tail"
        fallback) plus the SIMD renders for the platform's compiler family:
        GCC/Clang get the full vector_size family so the width can be switched
        via PIVTOOLS_FFT_ISA without regenerating; MSVC gets the avx2 intrinsic
        render (vector_size is unavailable there)."""
        import sys

        sizes = [str(n) for n in self._built_fft_sizes()]
        # convolve() pads each window to 2N for the correlation-plane weight and
        # correlates via the codelet FFT; the 96/128 windows therefore need 192/256
        # codelets. These are codelet-internal (scalar path only) -- NOT selectable
        # interrogation-window sizes, so they are appended here for codegen but kept
        # out of BUILT_FFT_SIZES (which drives config validation). The SIMD render of
        # 192/256 is unused (static inline) and elided by the compiler.
        for extra in ("192", "256"):
            if extra not in sizes:
                sizes.append(extra)
        if self.use_msvc:
            renders = ["scalar", "avx2"]
        else:
            renders = ["scalar", "vecext4", "vecext", "vecext16"]
        gen = self.src_dir / "codelet_gen" / "gen_codelet.py"
        out_hdr = self.src_dir / "codelets_gen.h"
        if not gen.exists():
            raise RuntimeError(f"codelet generator not found: {gen}")
        cmd = [
            sys.executable,
            str(gen),
            "--emit",
            str(out_hdr),
            "--sizes",
            *sizes,
            "--isa",
            *renders,
        ]
        print(f">>> [PIVTOOLS_FFT] codegen renders: {renders}")
        self._run(cmd)
        if not out_hdr.exists():
            raise RuntimeError(f"codelet generation failed: {out_hdr} not created")

    def _cleanup_intermediates(self, build_dir):
        """Clean up intermediate build files."""
        for pattern in ["*.obj", "*.exp", "*.lib"]:
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
