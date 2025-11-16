# setup.py
import os
import platform
import pathlib
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildCLib(build_ext):
    def run(self):
        if not self.dry_run:
            self.build_c_libraries()
        super().run()

    def build_c_libraries(self):
        pkg_dir = pathlib.Path(__file__).parent
        build_dir = pkg_dir / "pivtools_cli" / "lib"
        build_dir.mkdir(parents=True, exist_ok=True)
        src_dir = pkg_dir / "pivtools_cli" / "lib"

        # --- Detect static FFTW ---
        static_root = pkg_dir / "static_fftw"
        sys_name = platform.system().lower()
        if sys_name == "darwin":
            sys_name = "macos"
            arch = platform.machine().lower()
            fftw_dir = static_root / ("macos_arm64" if arch == "arm64" else "macos_x86_64")
        elif sys_name == "windows":
            fftw_dir = static_root / "windows"
        elif sys_name == "linux":
            fftw_dir = static_root / "linux"
        else:
            raise RuntimeError(f"Unsupported OS: {sys_name}")

        if not fftw_dir.exists():
            raise RuntimeError(f"Static FFTW not found: {fftw_dir}")

        fftw_inc = fftw_dir / "include"
        fftw_lib = fftw_dir / "lib"
        fftw_lib_file = fftw_lib / ("libfftw3f.a" if sys_name != "windows" else "fftw3f.lib")

        # --- Compiler ---
        if sys_name == "windows":
            compiler = "cl"
            shared_flag = "/DLL"
            extra_compile = ["/O2", "/openmp:experimental", "/MT"]
            extra_link = ["/link", f"/LIBPATH:{fftw_lib}", str(fftw_lib_file)]
        else:
            compiler = os.environ.get("CC", "gcc")
            shared_flag = "-shared"
            extra_compile = ["-O3", "-fPIC", "-fopenmp"]
            extra_link = [str(fftw_lib_file), "-lm", "-fopenmp"]

        lib_ext = ".pyd" if sys_name == "windows" else ".so"

        # --- Build libbulkxcorr2d ---
        sources1 = [
            "peak_locate_lm.c",
            "PIV_2d_cross_correlate.c",
            "xcorr.c",
            "xcorr_cache.c",
        ]
        cmd1 = [
            compiler, *extra_compile, shared_flag,
            *[str(src_dir / s) for s in sources1],
            f"-I{src_dir}", f"-I{fftw_inc}",
            "-o", str(build_dir / f"libbulkxcorr2d{lib_ext}")
        ] + extra_link
        self._run(cmd1)

        # --- Build libinterp2custom ---
        cmd2 = [
            compiler, *extra_compile, shared_flag,
            str(src_dir / "interp2custom.c"),
            f"-I{src_dir}",
            "-o", str(build_dir / f"libinterp2custom{lib_ext}")
        ]
        self._run(cmd2)

    def _run(self, cmd):
        print("RUN:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"Build failed: {result.returncode}")


# Dummy extension to trigger build_ext
dummy_ext = Extension("pivtools_cli._build_marker", sources=["pivtools_cli/_build_marker.c"])

setup(
    ext_modules=[dummy_ext],
    cmdclass={"build_ext": BuildCLib},
)