"""Shared core for the PIV benchmark/profile harness (``bench.py``).

This module holds everything the ``scaling`` / ``profile`` / ``compare`` subcommands
need in common and nothing CLI-specific (``bench.py`` owns argparse):

* :func:`resolve_config` — load a user base ``config.yaml`` and override only the
  dataset / sweep keys, so PIV settings stay pinned and reproducible.
* :func:`build_provenance` / :func:`detect_fft_backend` — stamp every result with
  enough build/host context that two runs can be compared honestly (and a mismatch
  refused). Detection probes ``fftwf_execute`` — the one FFTW symbol that is present
  in every FFTW build and absent from the codelet build (``fftwf_plan_dft_2d`` is a
  false negative because the kernel uses real-to-complex transforms).
* result IO (timestamped CSV + JSON under ``results/``) and platform helpers.

No hard-coded dataset paths or machine constants live here — that was the rot in the
deleted legacy scripts. The caller passes everything in.
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import platform
import socket
import statistics
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import pivtools_cli
from pivtools_core.config import Config

# --- locations -------------------------------------------------------------

PROFILE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROFILE_DIR / "results"

# The one FFTW symbol guaranteed present in any FFTW build and absent from the
# codelet build. Verified against both worktrees' libbulkxcorr2d.so.
_FFTW_PROBE_SYMBOL = "fftwf_execute"

_LIB_STEM = "libbulkxcorr2d"


def lib_path() -> Path:
    """Absolute path of the ``libbulkxcorr2d`` shared library that *this* interpreter
    would load, resolved the same way the production correlator resolves it
    (``pivtools_cli/lib/``). Platform-correct extension."""
    ext = ".dll" if os.name == "nt" else ".so"
    return Path(pivtools_cli.__file__).resolve().parent / "lib" / f"{_LIB_STEM}{ext}"


def detect_fft_backend(lib: ctypes.CDLL) -> str:
    """Return ``"fftw"`` if the loaded binary links FFTW, else ``"codelet"``.

    Probes :data:`_FFTW_PROBE_SYMBOL`. Resolution ⇒ FFTW; ``AttributeError`` ⇒ the
    symbol is not exported, i.e. the FFTW-free codelet build.
    """
    try:
        getattr(lib, _FFTW_PROBE_SYMBOL)
        return "fftw"
    except AttributeError:
        return "codelet"


# --- C sub-kernel timing (FFT vs peak-fit) ---------------------------------


def kernel_timing_available(lib: ctypes.CDLL) -> bool:
    """True if the loaded ``.so`` exports the flag-gated sub-kernel timers (a build
    with the instrumentation; older binaries simply lack the symbols)."""
    return hasattr(lib, "bulkxcorr2d_get_timing")


def _bind_kernel_timing(lib: ctypes.CDLL) -> None:
    lib.bulkxcorr2d_set_timing_enabled.argtypes = [ctypes.c_int]
    lib.bulkxcorr2d_set_timing_enabled.restype = None
    lib.bulkxcorr2d_reset_timing.argtypes = []
    lib.bulkxcorr2d_reset_timing.restype = None
    lib.bulkxcorr2d_get_timing.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.bulkxcorr2d_get_timing.restype = None


@contextmanager
def kernel_timing(lib: ctypes.CDLL) -> Iterator[Optional[Any]]:
    """Enable the FFT-vs-peak-fit sub-kernel timers on ``lib`` for the duration.

    Pass the correlator's own ``lib`` handle so the timed globals are the ones the
    correlation actually updates. Yields a ``read()`` callable returning
    ``(t_fft_s, t_fit_s)`` — total thread-seconds accumulated since entry — or
    yields ``None`` if the binary lacks the timers (graceful on old builds).
    """
    if not kernel_timing_available(lib):
        yield None
        return
    _bind_kernel_timing(lib)
    lib.bulkxcorr2d_set_timing_enabled(1)
    lib.bulkxcorr2d_reset_timing()
    _fft = ctypes.c_double(0.0)
    _fit = ctypes.c_double(0.0)

    def read() -> tuple[float, float]:
        lib.bulkxcorr2d_get_timing(ctypes.byref(_fft), ctypes.byref(_fit))
        return _fft.value, _fit.value

    try:
        yield read
    finally:
        lib.bulkxcorr2d_set_timing_enabled(0)


# --- host / build provenance ----------------------------------------------


def _git(pkg_dir: Path, *args: str) -> Optional[str]:
    """Run ``git *args`` in ``pkg_dir``; return stripped stdout or ``None`` on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _cpu_model() -> str:
    """Best-effort human CPU model (e.g. 'AMD EPYC 7742'). Distinguishes nodes so a
    cross-hardware comparison is caught — see the EPYC-vs-Broadwell trap."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return platform.processor() or "unknown"


def _filesystem(path: Optional[str]) -> str:
    """Best-effort filesystem/mount type for ``path`` (warm-cache vs networked-FS
    matters for the I/O numbers). Returns ``"unknown"`` if it can't be determined."""
    if not path:
        return "unknown"
    try:
        if platform.system() == "Linux":
            # Walk up to the mount point, then look it up in /proc/mounts.
            target = os.path.realpath(path)
            mounts = {}
            for line in Path("/proc/mounts").read_text().splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    mounts[parts[1]] = parts[2]
            while target and target != "/":
                if target in mounts:
                    return mounts[target]
                target = os.path.dirname(target)
            return mounts.get("/", "unknown")
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["df", "-P", path], capture_output=True, text=True, check=True
            )
            # mount device on the data line; fstype isn't in df -P, report device tail.
            dev = out.stdout.splitlines()[-1].split()[0]
            return dev
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, IndexError):
        pass
    return "unknown"


def build_provenance(
    *, dataset: Optional[str] = None, cache_policy: Optional[str] = None
) -> dict[str, Any]:
    """Stamp the current build + host into a dict embedded in every result.

    ``compare`` refuses (or warns) when two stamps differ in anything but
    ``fft_backend`` / ``git_sha``, so this is the apples-to-apples guard. Includes a
    ``git_dirty`` flag because the codelet branch carries uncommitted work — a stamp
    that hides that would be dishonest.
    """
    pkg_dir = Path(pivtools_cli.__file__).resolve().parent.parent
    lp = lib_path()
    lib = ctypes.CDLL(str(lp))
    return {
        "git_sha": _git(pkg_dir, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(pkg_dir, "status", "--porcelain")),
        "fft_backend": detect_fft_backend(lib),
        "pivtools_version": getattr(pivtools_cli, "__version__", "unknown"),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "filesystem": _filesystem(dataset),
        "cache_policy": cache_policy,
        "lib_path": str(lp),
        "lib_mtime": (
            datetime.fromtimestamp(lp.stat().st_mtime).isoformat()
            if lp.exists()
            else None
        ),
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
        "omp_places": os.environ.get("OMP_PLACES", ""),
        "platform": platform.platform(),
        "stamped_at": datetime.now().isoformat(timespec="seconds"),
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Summary statistics over repeated measurements, for the reviewer-grade
    report: ``n``, ``mean``, sample ``std``, ``median``, ``iqr`` (q3-q1), and
    ``cov`` (coefficient of variation = std/mean), plus ``min`` / ``max``.

    A single value has ``std = iqr = cov = 0`` (no spread to report). Empty
    input is a bug upstream — raise rather than fabricate a statistic.
    """
    data = [float(v) for v in values]
    if not data:
        raise ValueError("summarize() got no values")
    mean = statistics.mean(data)
    if len(data) > 1:
        std = statistics.stdev(data)
        q1, _, q3 = statistics.quantiles(data, n=4)
        iqr = q3 - q1
    else:
        std = iqr = 0.0
    return {
        "n": len(data),
        "mean": mean,
        "std": std,
        "median": statistics.median(data),
        "iqr": iqr,
        "cov": (std / mean) if mean else 0.0,
        "min": min(data),
        "max": max(data),
    }


# Fields that SHOULD differ between the two arms of an A/B (that's the point).
_AB_EXPECTED_DIFF = (
    "fft_backend",
    "git_sha",
    "git_dirty",
    "lib_mtime",
    "lib_path",
    "stamped_at",
)
# Fields that MUST match or the comparison is invalid (cross-machine, cross-cache,
# cross-dataset A/B is not a valid comparison).
_AB_MUST_MATCH = (
    "hostname",
    "cpu_model",
    "cpu_count",
    "filesystem",
    "cache_policy",
    "pivtools_version",
    "platform",
)


def compare_provenance(
    prov_a: dict[str, Any], prov_b: dict[str, Any]
) -> dict[str, Any]:
    """Guard two provenance stamps for an apples-to-apples A/B.

    Returns ``{"ok": bool, "backends": (a, b), "warnings": [...], "notes": [...]}``.
    ``ok`` is False (caller should warn loudly, not silently proceed) when any
    :data:`_AB_MUST_MATCH` field differs, or when ``fft_backend`` does *not* differ
    (then the two runs are the same binary — there is nothing to compare).
    """
    warnings: list[str] = []
    notes: list[str] = []

    for field in _AB_MUST_MATCH:
        va, vb = prov_a.get(field), prov_b.get(field)
        if va != vb:
            warnings.append(
                f"{field} differs: A={va!r} B={vb!r} — comparison may be invalid"
            )

    ba, bb = prov_a.get("fft_backend"), prov_b.get("fft_backend")
    if ba == bb:
        warnings.append(
            f"both runs use the same FFT backend ({ba!r}) — nothing to A/B; "
            "did you run one arm in each worktree?"
        )
    else:
        notes.append(f"FFT backend A={ba!r} vs B={bb!r} (the intended comparison axis)")

    if prov_a.get("git_dirty") or prov_b.get("git_dirty"):
        notes.append("at least one arm has uncommitted changes (git_dirty=True)")

    return {
        "ok": not warnings,
        "backends": (ba, bb),
        "warnings": warnings,
        "notes": notes,
    }


# --- config resolution -----------------------------------------------------


def resolve_config(
    base_config_path: str,
    *,
    dataset: str,
    n_images: Optional[int] = None,
    image_format: Optional[Sequence[str]] = None,
    start_index: Optional[int] = None,
    workers: Optional[int] = None,
    threads: Optional[int] = None,
    worker_memory: Optional[str] = None,
) -> Config:
    """Load the user's base ``config.yaml`` and override *only* the dataset and sweep
    keys, leaving PIV settings (windows, passes, fit method, save mode) untouched.

    ``auto_compute_params`` is forced off: otherwise ``omp_threads`` /
    ``dask_workers_per_node`` are derived from core count and our sweep overrides are
    silently ignored.
    """
    config = Config(base_config_path)
    data = config.data

    data.setdefault("paths", {})["source_paths"] = [dataset]
    if image_format is not None:
        data.setdefault("images", {})["image_format"] = list(image_format)
    if start_index is not None:
        data.setdefault("images", {})["start_index"] = start_index
    if n_images is not None:
        data.setdefault("images", {})["num_images"] = n_images

    proc = data.setdefault("processing", {})
    proc["auto_compute_params"] = False  # else the overrides below are inert
    if threads is not None:
        proc["omp_threads"] = threads
    if workers is not None:
        proc["dask_workers_per_node"] = workers
    if worker_memory is not None:
        proc["dask_memory_limit"] = worker_memory

    return config


# --- result IO -------------------------------------------------------------


def timestamped_path(stem: str, ext: str) -> Path:
    """``results/<stem>_YYYYMMDD_HHMMSS.<ext>`` (results dir created if absent)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{stem}_{ts}.{ext}"


def write_csv_header(csv_path: Path, fields: Sequence[str]) -> None:
    """Create ``csv_path`` with a header row (overwrites)."""
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(fields)).writeheader()


def append_csv_row(csv_path: Path, fields: Sequence[str], row: dict[str, Any]) -> None:
    """Append one row, immediately flushed — crash-safe so a killed sweep keeps every
    completed config. Extra keys in ``row`` are ignored."""
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore").writerow(row)


def write_json(json_path: Path, payload: dict[str, Any]) -> None:
    """Write a profile result (per-pair budget + provenance) as pretty JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
