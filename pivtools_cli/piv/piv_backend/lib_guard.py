"""Load-time guard for the compiled libraries' CPU ISA floor."""

import ctypes


def require_cpu_supported(lib: ctypes.CDLL, lib_path: str) -> None:
    """Refuse to run on a CPU below the ISA floor the library was built with.

    ``pivtools_cpu_supported`` lives in the scalar TU (compiled with no arch
    flags), so calling it is always safe — even on a CPU the SIMD kernels
    would crash. A library without the symbol predates the guard and is
    stale; per project convention that fails loudly so it gets rebuilt,
    rather than silently running unguarded kernels.
    """
    if not hasattr(lib, "pivtools_cpu_supported"):
        raise RuntimeError(
            f"{lib_path} is a stale build without the pivtools_cpu_supported "
            "symbol. Rebuild the C libraries (cd PyPIVTools && python setup.py "
            "build) or reinstall the package."
        )
    if not lib.pivtools_cpu_supported():
        raise RuntimeError(
            "pivtools binary wheels require AVX2+FMA (Intel Haswell 2013+ / "
            "AMD Excavator+) and this CPU lacks them. Install from source "
            "instead: pip install --no-binary pivtools pivtools"
        )
