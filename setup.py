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
        use_clang_cl = False  # set True in the Windows branch when clang-cl is selected

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

            # Compiler setup — Homebrew LLVM clang. It bundles OpenMP, so plain
            # -fopenmp works (identical flag path to Linux clang). Override with $CC.
            # The runtime needs an rpath to libomp.dylib in the LLVM lib dir.
            compiler = os.environ.get("CC") or self._brew_llvm_clang()
            print(f"Using compiler: {compiler}")
            llvm_lib = pathlib.Path(compiler).resolve().parent.parent / "lib"
            omp_rpath = [f"-Wl,-rpath,{llvm_lib}", f"-L{llvm_lib}", "-lomp"]

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
                    *omp_rpath,
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
                    *omp_rpath,
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

            # Windows toolchain: prefer clang-cl (faster codegen on the PIV hot
            # kernels — see the peak-fit A/B), fall back to MSVC cl when absent.
            # Override with PIVTOOLS_WIN_COMPILER=cl|clang-cl|<full path to either>.
            compiler = self._resolve_windows_compiler()
            use_clang_cl = pathlib.Path(compiler).stem.lower() == "clang-cl"
            print(f"Windows compiler: {compiler}  (clang-cl={use_clang_cl})")
            shared_flag = "/LD"
            if use_clang_cl:
                # clang-cl is cl-flag-compatible: it has native C11 atomics (no
                # /experimental:c11atomics) and /openmp for the real OpenMP runtime
                # (links libomp — libomp.dll is staged beside the DLLs after build).
                self.extra_compile = ["/O2", "/std:c11", "/openmp", "/MT"]
            else:
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

            # Linux: clang is the required toolchain. $CC overrides EXPLICITLY
            # (e.g. CC=gcc on an HPC node); there is no silent gcc fallback —
            # a missing clang is a hard error so the build never quietly differs.
            compiler = os.environ.get("CC") or shutil.which("clang")
            if not compiler:
                raise RuntimeError(
                    "clang not found. PIVTOOLs builds with clang.\n"
                    "Install clang, or set CC explicitly (e.g. CC=gcc) to override."
                )
            print(f"Linux compiler: {compiler}")
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
        self.use_clang_cl = use_clang_cl
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

        # libbulkxcorr2d bundles the LM peak fitter with the codelet FFT. They
        # share a call graph -- PIV_2d_cross_correlate.c calls lsqpeaklocate_lm
        # directly -- so they MUST link into one binary, but they want DIFFERENT
        # flags. The FFT TUs want AVX2/native + fast fp-contraction for
        # throughput; the LM kernel wants NONE of that: AVX2 FMA contraction
        # perturbs the Levenberg-Marquardt convergence path (and is a measured
        # wash), so peak_locate_lm.c compiles scalar/libm with no /arch. Hence a
        # split compile -- each TU to its own object with its own flags -- then
        # one link of the four objects.
        fft_sources = [
            "PIV_2d_cross_correlate.c",
            "xcorr.c",
            "codelet_fft.c",
        ]
        peak_source = "peak_locate_lm.c"

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
        base_compile = [f for f in self.extra_compile if f not in fftw_tokens]
        bulk_link = [f for f in self.extra_link if f not in fftw_tokens]

        # Stage B: the SIMD width macro + arch + accuracy-preserving throughput
        # flags. These go on the FFT TUs ONLY (not the peak TU, not the FFTW/GSL
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
        fft_compile = base_compile + simd_flags
        peak_compile = base_compile  # NO simd_flags: stable scalar/libm LM kernel
        print(
            f">>> [PIVTOOLS_FFT] libbulkxcorr2d  isa={self.fft_isa}  lanes={self.fft_lanes}  "
            f"macro={self.fft_macro}  render={self.fft_render}  "
            f"arch='{' '.join(self.fft_arch_flags)}'  lto={lto}"
        )
        print(f">>> [PIVTOOLS_FFT] FFT TU flags:  {' '.join(fft_compile)}")
        print(
            f">>> [PIVTOOLS_FFT] peak TU flags: {' '.join(peak_compile)}  "
            "(no /arch -- LM stability)"
        )

        # Compile each TU to its own object (one invocation per file keeps the
        # clang-cl /Fo vs gcc -o asymmetry trivial), then link all four.
        obj_ext = ".obj" if use_msvc else ".o"
        compile_units = [(peak_source, peak_compile)] + [
            (s, fft_compile) for s in fft_sources
        ]
        obj_paths = []
        for src_name, flags in compile_units:
            obj_path = build_dir / f"{pathlib.Path(src_name).stem}{obj_ext}"
            if use_msvc:
                cmd_obj = [
                    compiler,
                    *flags,
                    "/c",
                    f"/Fo{obj_path}",
                    str(src_dir / src_name),
                    f"/I{src_dir}",
                ]
            else:
                cmd_obj = [
                    compiler,
                    *flags,
                    "-c",
                    str(src_dir / src_name),
                    f"-I{src_dir}",
                    "-o",
                    str(obj_path),
                ]
            self._run(cmd_obj)
            if not obj_path.exists():
                raise RuntimeError(f"Build failed: {obj_path} not created")
            obj_paths.append(obj_path)

        # Link the four objects into the shared library. OpenMP must be on the
        # link line (/openmp pulls libomp on clang-cl; bulk_link carries
        # -fopenmp on clang/gcc). LTO, when enabled, needs its link-time flag.
        output_file = build_dir / f"libbulkxcorr2d{lib_ext}"
        if use_msvc:
            cmd_link = [
                compiler,
                shared_flag,
                *[str(o) for o in obj_paths],
                f"/Fe{output_file}",
                "/openmp",
                "/MT",
            ]
            if lto:
                cmd_link += ["/LTCG"]
            cmd_link += bulk_link
        else:
            cmd_link = [
                compiler,
                shared_flag,
                *[str(o) for o in obj_paths],
                "-o",
                str(output_file),
            ]
            if lto:
                cmd_link += ["-flto"]
            cmd_link += bulk_link
        self._run(cmd_link)
        if not output_file.exists():
            raise RuntimeError(f"Build failed: {output_file} not created")

        # Remove the intermediate objects (targeted: _cleanup_intermediates
        # globs *.obj but not the *.o produced on clang/gcc).
        for obj_path in obj_paths:
            obj_path.unlink(missing_ok=True)

        self._cleanup_intermediates(build_dir)

        # --- Build libfusedwarp (no FFTW/GSL deps — only OpenMP + math) ---
        # SIMD flags for the warp kernel ONLY (the FFTW/GSL libs above are untouched).
        # Phase C of fused_warp.c has an explicit-SIMD interior sampler (simd_warp.h):
        # NEON on arm64, AVX2 on x86. The AVX2 path is gated on the compiler
        # predefining __AVX2__, so the arch flag below is what actually lights it up —
        # without it the kernel silently compiles its scalar fallback. arm64 NEON is the
        # mandatory baseline (no flag needed; -mcpu=native only adds host tuning).
        # Native tuning NOW — redistributable (portable) wheels are a later phase. NOT
        # adding -ffast-math: it reassociates and would desync the scalar reference
        # (impl=0 oracle) from the SIMD path. Override the arch flag explicitly (e.g. an
        # HPC login node that differs from the compute nodes) with PIVTOOLS_WARP_MARCH.
        warp_arch = platform.machine().lower()
        warp_is_arm = warp_arch in ("arm64", "aarch64")
        if self.use_clang_cl:
            warp_simd_flags = ["/clang:-O3", "/clang:-march=native"]
        elif use_msvc:  # plain cl (explicit PIVTOOLS_WIN_COMPILER=cl) — cl has no /O3
            warp_simd_flags = ["/arch:AVX2"]  # AVX2 → 256-bit + FMA + __AVX2__
        elif warp_is_arm:  # macOS / Linux arm64 — NEON baseline + this-CPU tuning
            warp_simd_flags = ["-mcpu=native"]
        else:  # x86-64 clang/gcc — native lowering defines __AVX2__
            warp_simd_flags = ["-march=native"]

        warp_march_env = os.environ.get("PIVTOOLS_WARP_MARCH", "").strip()
        if warp_march_env:
            warp_simd_flags = warp_march_env.split()

        if use_msvc:
            output_file = build_dir / f"libfusedwarp{lib_ext}"
            cmd_fw = [
                compiler,
                *self.extra_compile,
                *warp_simd_flags,
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
                *warp_simd_flags,
                shared_flag,
                str(src_dir / "fused_warp.c"),
                f"-I{src_dir}",
                "-o",
                str(build_dir / f"libfusedwarp{lib_ext}"),
                "-lm",
                "-fopenmp",
            ]
        if warp_simd_flags:
            print(f"libfusedwarp SIMD flags: {' '.join(warp_simd_flags)}")
        self._run(cmd_fw)
        if not (build_dir / f"libfusedwarp{lib_ext}").exists():
            raise RuntimeError(f"Build failed: libfusedwarp{lib_ext} not created")
        print(f"Successfully built libfusedwarp{lib_ext}")

        self._cleanup_intermediates(build_dir)

        # --- Build libmarquadt (for ensemble PIV) ---
        self._build_marquadt()

        # --- Build libkspace (k-space fitting, requires GSL + FFTW) ---
        self._build_kspace()

        # --- Stage the OpenMP runtime for clang-cl ---
        # clang-cl's /openmp links libomp dynamically, so libomp.dll must sit
        # beside the built DLLs (it ships via the *.dll package_data glob). The
        # Python lib loaders call os.add_dll_directory(lib_dir) so the dependent
        # DLL is found at load time. MSVC's /openmp:experimental needs no such
        # redistributable.
        if self.use_clang_cl:
            libomp = pathlib.Path(self.compiler).resolve().parent / "libomp.dll"
            if libomp.is_file():
                shutil.copy2(libomp, self.build_dir / "libomp.dll")
                print(f"Staged OpenMP runtime: {libomp} -> {self.build_dir / 'libomp.dll'}")
            else:
                raise RuntimeError(
                    f"clang-cl /openmp build but libomp.dll not found at {libomp}.\n"
                    "The built DLLs will fail to load without it. Set "
                    "PIVTOOLS_WIN_COMPILER to a clang-cl whose directory contains libomp.dll."
                )

    def _brew_llvm_clang(self):
        """Resolve Homebrew LLVM clang on macOS ($(brew --prefix llvm)/bin/clang).
        Chosen over Apple clang because it bundles OpenMP (plain -fopenmp works,
        matching the Linux clang path)."""
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "llvm"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Homebrew LLVM clang not found. Install with: brew install llvm\n"
                "(or set CC to a clang that bundles OpenMP)."
            ) from exc
        clang = pathlib.Path(prefix) / "bin" / "clang"
        if not clang.is_file():
            raise RuntimeError(
                f"brew --prefix llvm = {prefix} but {clang} is missing. "
                "Run: brew install llvm"
            )
        return str(clang)

    def _resolve_windows_compiler(self):
        """Resolve clang-cl on Windows — the required toolchain. clang-cl ships
        inside Visual Studio (not on PATH), and the build already runs in a
        vcvars64 shell, so VSINSTALLDIR points at the matching toolset. Order:
          1. PIVTOOLS_WIN_COMPILER  (explicit override: clang-cl path, or 'cl' to opt out)
          2. VSINSTALLDIR/VC/Tools/Llvm/x64/bin/clang-cl.exe  (the active vcvars toolset)
          3. clang-cl on PATH
        Missing clang-cl is a HARD ERROR — no silent fall back to MSVC cl."""
        override = os.environ.get("PIVTOOLS_WIN_COMPILER")
        if override:
            return override

        vsinstall = os.environ.get("VSINSTALLDIR")
        if vsinstall:
            cand = (
                pathlib.Path(vsinstall)
                / "VC" / "Tools" / "Llvm" / "x64" / "bin" / "clang-cl.exe"
            )
            if cand.is_file():
                return str(cand)

        found = shutil.which("clang-cl")
        if found:
            return found

        raise RuntimeError(
            "clang-cl not found. PIVTOOLs builds with clang-cl on Windows.\n"
            "Install the 'C++ Clang tools for Windows' VS component and build from an\n"
            "'x64 Native Tools Command Prompt for VS 2022', or set PIVTOOLS_WIN_COMPILER\n"
            "to a clang-cl.exe (or to 'cl' to force the MSVC toolchain)."
        )

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
