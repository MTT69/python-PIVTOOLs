"""
Tests for pivtools_cpu_supported — the load-time ISA-floor check exported by
the scalar TU (peak_locate_lm.c). Binary wheels are built with AVX2+FMA on
x86; the loaders call this symbol after CDLL and refuse to run on CPUs below
that floor instead of dying later with SIGILL. On the dev machine that built
the library the check must, by construction, pass.
"""

import ctypes
import os


def _load_lib():
    lib_extension = ".dll" if os.name == "nt" else ".so"
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pivtools_cli",
            "lib",
            f"libbulkxcorr2d{lib_extension}",
        )
    )
    assert os.path.isfile(path), f"library not built: {path}"
    return ctypes.CDLL(path)


def test_cpu_supported_symbol_exists():
    lib = _load_lib()
    assert hasattr(lib, "pivtools_cpu_supported")


def test_cpu_supported_on_build_machine():
    lib = _load_lib()
    lib.pivtools_cpu_supported.restype = ctypes.c_int
    assert lib.pivtools_cpu_supported() == 1
