"""Profile mode for ``bench.py`` — the complete serial per-pair time budget.

Unlike the bare correlator profile (compute sections only), this exercises the real
production code paths and times the whole per-pair budget:

    read (load_images) → [per-pass compute sections] → save (save_piv_result_distributed)

so the breakdown figure can include I/O and saving, not just compute. It runs serially
(one batch, no Dask) on real files, which gives an accurate per-pair *compute* budget
and a single-stream I/O *floor* — aggregate I/O under worker contention is scaling
mode's job, not this (see README).

Page-cache control is mandatory here: a serial read loop warms the OS page cache, and
an A/B where one backend runs first warms the cache for the other. ``cache_policy`` with
symmetric per-run priming removes that order bias — every run reaches a known cache
state *before* the timed read.

The FFT segment is ``bulkxcorr2d``. When the kernel-timing flag is on (C build with the
sub-kernel timers), the payload also carries ``xcorr_fft`` + ``peak_fit`` (added by the
caller via the ctypes getter) — that's the isolated FFT-speedup number for the A/B.

Ensemble mode is a named follow-up; only instantaneous is implemented here.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import bench_common as bc
from pivtools_cli.piv.piv_backend.cpu_instantaneous import InstantaneousCorrelatorCPU
from pivtools_cli.piv.save_results import save_piv_result_distributed
from pivtools_core.image_handling.load_images import load_images

# Sub-sections nested *inside* predictor_corrector — excluding them from pass totals
# avoids double-counting (they are a finer breakdown of the same wall time).
_PC_SUB_SECTIONS = ("pc_gaussian_smooth", "pc_predictor_remap", "pc_fused_warp")

CACHE_WARM = "warm"
CACHE_COLD = "cold"


# --- cache control ---------------------------------------------------------


def _dataset_files(config, dataset: str, n_images: int) -> list[Path]:
    """Resolve the on-disk image files for the first ``n_images`` pairs, for cache
    eviction. Best-effort — returns whatever exists for the configured format."""
    files: list[Path] = []
    fmt = config.image_format  # tuple of one or two printf patterns
    start = config.data.get("images", {}).get("start_index", 1)
    src = Path(dataset)
    for idx in range(start, start + n_images):
        for pattern in fmt:
            try:
                p = src / (pattern % idx)
            except TypeError:
                continue
            if p.is_file():
                files.append(p)
    return files


def _vector_format(config) -> str:
    """Output filename pattern. Stored under ``images.vector_format`` (a one-element
    list in current configs); fall back to the production default."""
    vf = config.data.get("images", {}).get("vector_format")
    if isinstance(vf, (list, tuple)) and vf:
        return vf[0]
    if isinstance(vf, str):
        return vf
    return "B%05d.mat"


def evict_cache(files: list[Path]) -> bool:
    """Evict ``files`` from the OS page cache via ``vmtouch -e`` (no root, per-file).
    Returns True on success. Best-effort: missing tool / non-Linux ⇒ False so the
    caller can warn and fall back to warm."""
    if not files or shutil.which("vmtouch") is None:
        return False
    try:
        subprocess.run(
            ["vmtouch", "-e", *[str(f) for f in files]],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


# --- the budget -------------------------------------------------------------


def _read_batch(camera: int, config, dataset: str) -> tuple[np.ndarray, float]:
    """Materialise the image batch (the real disk read) and time it.

    ``load_images`` returns a lazy dask array; ``.compute()`` forces the read. Returns
    ``(images (N,2,H,W) float32, read_seconds)``.
    """
    lazy = load_images(camera, config, source=Path(dataset), batch_size=1)
    t0 = time.perf_counter()
    images = np.ascontiguousarray(lazy.compute())
    return images, time.perf_counter() - t0


def _pass_total(pass_sections: dict[str, float]) -> float:
    """Sum a pass's section times, excluding nested PC sub-sections."""
    return sum(v for k, v in pass_sections.items() if k not in _PC_SUB_SECTIONS)


def profile_instantaneous(
    config,
    *,
    dataset: str,
    n_images: int,
    iterations: int = 3,
    cache_policy: str = CACHE_WARM,
    do_warmup: bool = True,
    camera: Optional[int] = None,
) -> dict[str, Any]:
    """Run the per-pair budget for instantaneous PIV and return a JSON-able payload.

    :param config: a resolved :class:`Config` (dataset/sweep already applied).
    :param dataset: image directory (also used for cache eviction in cold mode).
    :param n_images: number of pairs in the batch.
    :param iterations: timed repeats (mean±std); excludes iteration 0 by construction
        of the warmup + priming below.
    :param cache_policy: ``"warm"`` (prime then read RAM-served) or ``"cold"``
        (evict then read from disk). Stamped into provenance.
    :param do_warmup: run one untimed correlation first so FFTW plan creation /
        first-touch allocation isn't charged to iteration 1.
    :param camera: camera number; defaults to the config's first active camera.
    """
    if camera is None:
        cams = config.data.get("paths", {}).get("camera_numbers") or [1]
        camera = cams[0]

    # Precondition #1: make the thread count explicit. The kernel reads
    # OMP_NUM_THREADS (all cores if unset), so the harness owns it — otherwise the
    # per-pair compute timing is at an unknown, machine-dependent thread count.
    omp_threads = str(config.omp_threads)
    os.environ["OMP_NUM_THREADS"] = omp_threads

    n_passes = len(config.window_sizes)

    # --- cache state: reach a known point BEFORE any timed read ---
    cache_note = cache_policy
    if cache_policy == CACHE_COLD:
        files = _dataset_files(config, dataset, n_images)
        if not evict_cache(files):
            cache_note = "cold-requested-but-warm (vmtouch unavailable)"
    else:
        # Warm: prime the page cache with a full read that we throw away, so the
        # timed reads below are steady-state and order-independent in an A/B.
        _read_batch(camera, config, dataset)

    correlator = InstantaneousCorrelatorCPU(config)
    correlator.profiling_enabled = True

    runs_to_save = config.instantaneous_runs_0based
    save_mode = config.instantaneous_save_mode
    save_compression = config.instantaneous_save_compression
    vector_fmt = _vector_format(config)

    # Read once (timed): on warm this is RAM-served; on cold it's first-touch disk.
    images, read_s = _read_batch(camera, config, dataset)
    n = images.shape[0]
    if n == 0:
        raise RuntimeError(f"No image pairs read from {dataset} (check image_format/start_index)")

    if do_warmup:
        correlator.correlate_batch(images, config)  # FFTW plans; untimed

    save_tmp = Path(tempfile.mkdtemp(prefix="bench_profile_save_"))
    kernel_split: Optional[dict[str, Any]] = None
    try:
        # section -> list of per-pair seconds (one entry per iteration), per pass
        per_pass: list[dict[str, list[float]]] = [{} for _ in range(n_passes)]
        save_per_iter: list[float] = []

        # The C sub-kernel timers (FFT vs peak-fit) accumulate across all the timed
        # correlate_batch calls below; read once at the end. Uses the correlator's own
        # lib handle so the globals are the ones the correlation updates.
        with bc.kernel_timing(correlator.lib) as read_kernel:
            for _ in range(iterations):
                piv_results = correlator.correlate_batch(images, config)
                profile = correlator.get_profile_summary()  # {pass: {section: sec}} batch totals

                for pass_idx in range(n_passes):
                    for section, elapsed in profile.get(pass_idx, {}).items():
                        per_pass[pass_idx].setdefault(section, []).append(elapsed / n)

                t0 = time.perf_counter()
                for i, piv_result in enumerate(piv_results):
                    save_piv_result_distributed(
                        piv_result, save_tmp, i + 1, runs_to_save,
                        vector_fmt=vector_fmt, save_mode=save_mode,
                        do_compression=save_compression,
                    )
                save_per_iter.append((time.perf_counter() - t0) / n)

            if read_kernel is not None:
                t_fft, t_fit = read_kernel()
                denom = iterations * n
                fft_fit = t_fft + t_fit
                kernel_split = {
                    "xcorr_fft_ms": 1e3 * t_fft / denom,
                    "peak_fit_ms": 1e3 * t_fit / denom,
                    "fft_fraction": (t_fft / fft_fit) if fft_fit > 0 else None,
                    "omp_threads": int(omp_threads),
                    "thread_summed": True,
                    "note": (
                        "per-pair = total thread-seconds / (iters*N). Thread-summed: at "
                        "omp_threads>1 these exceed the bulkxcorr2d wall by ~omp_threads; "
                        "at omp_threads=1 they reconcile with summed bulkxcorr2d. "
                        "fft_fraction is thread-count-independent — the A/B quantity."
                    ),
                }
    finally:
        shutil.rmtree(save_tmp, ignore_errors=True)

    # --- aggregate to mean±std per-pair (ms) ---
    def _stat(vals: list[float]) -> dict[str, float]:
        return {
            "mean_ms": 1e3 * statistics.fmean(vals),
            "std_ms": 1e3 * (statistics.stdev(vals) if len(vals) > 1 else 0.0),
        }

    passes = []
    compute_total_ms = 0.0
    for pass_idx in range(n_passes):
        sections = {sec: _stat(vals) for sec, vals in per_pass[pass_idx].items()}
        pass_compute = sum(
            s["mean_ms"] for sec, s in sections.items() if sec not in _PC_SUB_SECTIONS
        )
        compute_total_ms += pass_compute
        passes.append({
            "pass_idx": pass_idx,
            "window": config.window_sizes[pass_idx],
            "overlap": config.overlap[pass_idx],
            "sections_ms": sections,
            "pass_compute_ms": pass_compute,
        })

    read_pp_ms = 1e3 * read_s / n
    save_pp = _stat(save_per_iter)
    budget_total_ms = read_pp_ms + compute_total_ms + save_pp["mean_ms"]

    payload = {
        "kind": "profile",
        "mode": "instantaneous",
        "provenance": bc.build_provenance(dataset=dataset, cache_policy=cache_note),
        "dataset": dataset,
        "n_images": n,
        "iterations": iterations,
        "cache_policy": cache_note,
        "budget_per_pair_ms": {
            "read": read_pp_ms,
            "compute": compute_total_ms,
            "save": save_pp["mean_ms"],
            "total": budget_total_ms,
        },
        "save_ms": save_pp,
        "kernel_split": kernel_split,
        "passes": passes,
    }
    return payload


def profile(mode: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch by mode. Only ``instantaneous`` is implemented; ``ensemble`` is a
    named follow-up (different call path: correlate_batch_for_accumulation +
    finalize_pass) — raised explicitly rather than silently stubbed."""
    if mode == "instantaneous":
        return profile_instantaneous(**kwargs)
    if mode == "ensemble":
        raise NotImplementedError(
            "bench_profile ensemble mode is a follow-up; instantaneous is implemented. "
            "Ensemble needs the EnsembleCorrelatorCPU + SinglePassAccumulator path."
        )
    raise ValueError(f"unknown profile mode: {mode!r}")
