"""Wheel/sdist smoke test — run against an INSTALLED pivtools in a clean env.

Proves the four things a broken wheel gets wrong:
  1. the dependency closure imports (cv2, matplotlib, scipy, dask ...),
  2. both C libraries ship in the wheel and load via ctypes
     (on Windows this also proves libomp.dll was bundled),
  3. the CPU ISA-floor check passes on this machine,
  4. the config layer round-trips: Config() fails loudly without config.yaml,
     and `pivtools-cli init` creates one that Config() then loads.

Must not assume anything about the cwd — cibuildwheel runs tests outside the
project directory. Exits non-zero on any failure.
"""

import ctypes
import os
import subprocess
import sys
import tempfile

# --- 1. dependency closure ------------------------------------------------
import pivtools_cli  # noqa: E402
import pivtools_cli.cli  # noqa: F401, E402
import pivtools_cli.piv.piv_backend.cpu_ensemble  # noqa: F401, E402
import pivtools_cli.piv.piv_backend.cpu_instantaneous  # noqa: F401, E402
import pivtools_core  # noqa: E402

print(f"versions: core={pivtools_core.__version__} cli={pivtools_cli.__version__}")

# --- 2. + 3. ctypes loads + ISA floor --------------------------------------
lib_dir = os.path.join(os.path.dirname(pivtools_cli.__file__), "lib")
ext = ".dll" if os.name == "nt" else ".so"
for name in ("libbulkxcorr2d", "libfusedwarp", "libkspacefit"):
    lib_path = os.path.join(lib_dir, name + ext)
    if not os.path.isfile(lib_path):
        raise SystemExit(
            f"FAIL: {name}{ext} missing from installed package: {lib_path}"
        )
    lib = ctypes.CDLL(lib_path)
    print(f"loaded {name}{ext}")
    if name == "libbulkxcorr2d":
        # The ISA-floor symbol MUST exist — a wheel without it ships kernels
        # with no load-time guard, so its absence is a build regression.
        if not hasattr(lib, "pivtools_cpu_supported"):
            raise SystemExit(f"FAIL: pivtools_cpu_supported symbol missing from {name}")
        lib.pivtools_cpu_supported.restype = ctypes.c_int
        if lib.pivtools_cpu_supported() != 1:
            raise SystemExit(f"FAIL: pivtools_cpu_supported() == 0 in {name}")
        print(f"CPU ISA floor OK ({name})")
    if name == "libkspacefit":
        # Export-table probe. On Windows a symbol declared without
        # __declspec(dllexport) loads fine as a library but is absent from the
        # export table, so the DLL is useless to ctypes — catch that here
        # rather than at the first ensemble run.
        if not hasattr(lib, "kspace_lm_fit_batch"):
            raise SystemExit(f"FAIL: kspace_lm_fit_batch symbol missing from {name}")
        lib.kspace_lm_fit_max_threads.restype = ctypes.c_int
        print(f"OpenMP max threads = {lib.kspace_lm_fit_max_threads()} ({name})")

# --- 4. config round-trip ---------------------------------------------------
from pivtools_core.config import Config  # noqa: E402

orig_cwd = os.getcwd()
# ignore_cleanup_errors: Config() opens a pypiv.log handler inside the
# tempdir, and Windows cannot delete a directory holding an open file.
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
    try:
        os.chdir(td)
        try:
            Config()
            raise SystemExit("FAIL: Config() should raise without config.yaml")
        except FileNotFoundError as e:
            if "pivtools-cli init" not in str(e):
                raise SystemExit(f"FAIL: unexpected error message: {e}")
        print("fail-loud Config OK")

        subprocess.run([sys.executable, "-m", "pivtools_cli.cli", "init"], check=True)
        cfg = Config()
        print(
            f"config round-trip OK: fit_method={cfg.config_dict['ensemble_piv']['fit_method']}"
        )
    finally:
        os.chdir(orig_cwd)  # Windows: cannot delete the tempdir while cwd is inside it

print("SMOKE OK")
