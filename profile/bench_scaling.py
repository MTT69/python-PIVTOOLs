"""Scaling mode for ``bench.py`` — distributed worker x thread sweep.

This runs the **full production Dask pipeline** (cluster -> load -> scatter ->
correlate -> save) at a range of worker/thread combinations and records
end-to-end throughput. It is deliberately end-to-end *only*: the correlator
runs inside Dask worker processes, so the per-section compute breakdown can't
cross the process boundary back to the client — that finer split is profile
mode's job (``bench_profile``). What scaling sees that profile can't is the
real thing under load: I/O contention (``load_s``), scatter cost, and the
throughput ceiling as cores fill and oversubscribe.

Two workloads:

* ``--n-images N`` fixed  -> a **strong-scaling** sweep: every config processes
  the *same* N pairs, so S(p)=T(1)/T(p) and E(p)=S(p)/p are honest.
* ``--n-images`` omitted  -> the legacy per-worker derivation (each worker gets
  a fixed number of batches), which keeps per-worker load constant but is *not*
  a clean strong-scaling curve. Recorded as ``workload_mode=per_worker`` so the
  distinction is never silently lost.

The sweep axes (thread / worker / matrix / oversub) are generated from the
machine's core budget (``--total-cores``, ``--max-workers``) rather than the
hard-coded ``TOTAL_CORES=20`` of the deleted script, so a 192-core Iridis node
and a laptop both produce sensible matrices.

FFTW-wisdom policy (precondition 3) is exposed as ``--fftw-wisdom`` and stamped
into every row. ``shared`` (default) is the real current behaviour — all worker
processes share ``$HOME/.pypivtools_fftw_wisdom`` and pay any write-race cost,
which is the architectural finding we want to film. ``per-worker`` would give
each process its own wisdom file, but the kernel hard-codes the path (only
``HOME`` is a lever), so a clean per-worker knob needs a C change
(``PIV_FFTW_WISDOM_PATH``) that is deliberately out of scope here — it raises
rather than silently faking isolation.
"""

from __future__ import annotations

import gc
import glob
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import bench_common as bc

# CSV schema: the proven scaling columns plus provenance so two result sets can
# be compared honestly (see bench_common.build_provenance / compare_provenance).
CSV_FIELDS = [
    # sweep axes
    "workers", "threads", "total_cores", "oversub_ratio", "label",
    "iteration", "workload_mode",
    # workload
    "n_pairs", "batch_size", "n_saved",
    # timings (seconds)
    "cluster_startup_s", "load_s", "scatter_s", "correlate_s", "pipeline_total_s",
    # derived throughput
    "per_pair_ms", "pairs_per_s",
    # validity
    "valid", "error",
    # provenance (stamped per row so a partial CSV is still self-describing)
    "fft_backend", "fftw_wisdom", "git_sha", "git_dirty",
    "hostname", "cpu_count", "cpu_model",
]

WORKLOAD_FIXED = "fixed"
WORKLOAD_PER_WORKER = "per_worker"

# Per-worker fallback workload (only used when --n-images is omitted).
_BATCHES_PER_WORKER = 10


# --- sweep generation ------------------------------------------------------


def _progression(cap: int) -> list[int]:
    """Geometric progression 1,2,4,8,... up to and including ``cap``.

    Replaces the deleted script's hard-coded ``[1,2,4,8,10,16,20]`` so the sweep
    scales from a laptop to a 192-core node. ``cap`` is always included even when
    it isn't a power of two (e.g. 20 -> [1,2,4,8,16,20], 192 -> ...,128,192).
    """
    if cap < 1:
        return [1]
    vals = []
    v = 1
    while v < cap:
        vals.append(v)
        v *= 2
    vals.append(cap)
    return sorted(set(vals))


def thread_sweep(total_cores: int) -> list[dict[str, Any]]:
    """1 worker, vary OMP threads up to the core budget — pure thread scaling."""
    return [
        {"workers": 1, "threads": t, "label": "thread_sweep"}
        for t in _progression(total_cores)
    ]


def worker_sweep(total_cores: int, max_workers: int, threads: int) -> list[dict[str, Any]]:
    """Fixed ``threads``/worker, vary workers — the cleanest strong-scaling axis.

    Capped so ``workers*threads <= total_cores`` (stay within physical cores) and
    ``workers <= max_workers`` (RAM ceiling)."""
    cap = min(max_workers, max(1, total_cores // max(1, threads)))
    return [
        {"workers": w, "threads": threads, "label": "worker_sweep"}
        for w in _progression(cap)
    ]


def matrix_sweep(total_cores: int, max_workers: int) -> list[dict[str, Any]]:
    """All w x t combos with ``w*t <= total_cores`` and ``w <= max_workers``."""
    configs = []
    for w in _progression(max_workers):
        for t in _progression(total_cores):
            if w * t <= total_cores and w <= max_workers:
                configs.append({"workers": w, "threads": t, "label": "matrix"})
    configs.sort(key=lambda c: (c["workers"] * c["threads"], c["workers"]))
    return configs


def oversub_sweep(total_cores: int, max_workers: int) -> list[dict[str, Any]]:
    """Oversubscribed combos: ``total_cores < w*t <= 2*total_cores`` (capped)."""
    configs = []
    for w in _progression(max_workers):
        for t in _progression(total_cores):
            cores = w * t
            if total_cores < cores <= 2 * total_cores and w <= max_workers:
                configs.append({"workers": w, "threads": t, "label": "oversub"})
    configs.sort(key=lambda c: (c["workers"] * c["threads"], c["workers"]))
    return configs


def build_config_list(
    sweep: str, total_cores: int, max_workers: int, worker_sweep_threads: int = 2
) -> list[dict[str, Any]]:
    """Build the deduplicated config list for the requested sweep(s)."""
    configs: list[dict[str, Any]] = []
    if sweep in ("threads", "all"):
        configs.extend(thread_sweep(total_cores))
    if sweep in ("workers", "all"):
        configs.extend(worker_sweep(total_cores, max_workers, worker_sweep_threads))
    if sweep in ("matrix", "all"):
        configs.extend(matrix_sweep(total_cores, max_workers))
    if sweep in ("oversub", "all"):
        configs.extend(oversub_sweep(total_cores, max_workers))

    seen = set()
    deduped = []
    for cfg in configs:
        key = (cfg["workers"], cfg["threads"])
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)
    return deduped


def workload_for(
    workers: int, batch_size: int, n_images: Optional[int]
) -> tuple[int, str]:
    """Resolve the number of pairs for one config.

    Fixed ``n_images`` -> strong scaling (same workload for every config).
    Omitted -> per-worker derivation (constant batches/worker, not strong)."""
    if n_images is not None:
        return n_images, WORKLOAD_FIXED
    return workers * _BATCHES_PER_WORKER * batch_size, WORKLOAD_PER_WORKER


# --- FFTW wisdom policy (precondition 3) ------------------------------------

WISDOM_SHARED = "shared"
WISDOM_PER_WORKER = "per-worker"


def _apply_wisdom_policy(policy: str) -> None:
    """Enforce the requested FFTW-wisdom policy before any worker starts.

    ``shared`` is a no-op (current behaviour: one ``$HOME`` wisdom file for all
    processes). ``per-worker`` is refused: the kernel hard-codes the wisdom path
    off ``HOME`` (xcorr_cache.c), so genuine per-process isolation needs a C
    ``PIV_FFTW_WISDOM_PATH`` env hook that is out of scope for this step. Failing
    loudly is correct — silently hijacking ``HOME`` per worker would corrupt
    unrelated state and fake a result we didn't actually measure."""
    if policy == WISDOM_SHARED:
        return
    if policy == WISDOM_PER_WORKER:
        raise NotImplementedError(
            "--fftw-wisdom per-worker needs a C change (read PIV_FFTW_WISDOM_PATH "
            "in xcorr_cache.c so each Dask worker gets its own wisdom file). The "
            "kernel currently keys wisdom off $HOME only; hijacking HOME per worker "
            "would corrupt unrelated state. This is precondition-3 / propagation "
            "work, deliberately deferred. Use --fftw-wisdom shared for now."
        )
    raise ValueError(f"unknown fftw-wisdom policy: {policy!r}")


# --- the Dask run loop -----------------------------------------------------


def _quiet_shutdown(client, cluster) -> None:
    """Tear the cluster down without the verbose distributed shutdown chatter."""
    import logging as _logging

    for name in (
        "distributed", "distributed.worker", "distributed.scheduler",
        "distributed.nanny", "distributed.core", "distributed.comm",
        "tornado.application", "tornado.general",
    ):
        _logging.getLogger(name).setLevel(_logging.CRITICAL)
    try:
        client.close(timeout=10)
    except Exception:
        pass
    try:
        cluster.close(timeout=10)
    except Exception:
        pass


def _run_sliding_window(
    client, images, num_chunks, max_in_flight,
    scattered_config, scattered, output_path, runs_0based, vector_format,
) -> list[str]:
    """Sliding-window submit/drain over the lazy image chunks.

    Inlined (rather than calling ``pivtools_core.instantaneous``) because that
    module runs ``Config()``, registers signal handlers, and sets
    ``OMP_NUM_THREADS`` at import time — side effects we must not trigger inside
    the harness. Mirrors the production sliding window in dask_pipeline terms.
    """
    from dask.distributed import as_completed
    from pivtools_cli.processing.dask_pipeline import correlate_and_save_batch

    pending = {}
    next_to_submit = 0
    saved: list[str] = []

    while next_to_submit < min(max_in_flight, num_chunks):
        chunk_start = sum(images.chunks[0][:next_to_submit])
        filter_future = client.compute(images.blocks[next_to_submit])
        corr_future = client.submit(
            correlate_and_save_batch,
            filter_future, chunk_start + 1,
            scattered_config, scattered["cache"], scattered["masks"],
            output_path, runs_0based, vector_format,
            pure=False,
        )
        pending[corr_future] = next_to_submit
        next_to_submit += 1

    ac = as_completed(list(pending.keys()))
    for completed in ac:
        saved.extend(completed.result())
        del pending[completed]
        if next_to_submit < num_chunks:
            chunk_start = sum(images.chunks[0][:next_to_submit])
            filter_future = client.compute(images.blocks[next_to_submit])
            corr_future = client.submit(
                correlate_and_save_batch,
                filter_future, chunk_start + 1,
                scattered_config, scattered["cache"], scattered["masks"],
                output_path, runs_0based, vector_format,
                pure=False,
            )
            pending[corr_future] = next_to_submit
            ac.add(corr_future)
            next_to_submit += 1

    return saved


def run_single(
    base_config_path: str,
    dataset: str,
    *,
    workers: int,
    threads: int,
    label: str,
    n_images: Optional[int],
    image_format: Optional[list[str]] = None,
    start_index: Optional[int] = None,
    worker_memory: Optional[str] = None,
    fftw_wisdom: str = WISDOM_SHARED,
    warmup_batches: int = 1,
    total_cores: int = 1,
    iteration: int = 0,
    provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the full Dask pipeline once and return a CSV-able result dict.

    Always returns a dict (errors land in the ``error`` column, ``valid=false``),
    so a single bad config never aborts the sweep.
    """
    oversub_ratio = round(workers * threads / total_cores, 2) if total_cores else 0.0
    prov = provenance or {}
    base_result: dict[str, Any] = {
        "workers": workers,
        "threads": threads,
        "total_cores": workers * threads,
        "oversub_ratio": oversub_ratio,
        "label": label,
        "iteration": iteration,
        "workload_mode": "",
        "n_pairs": 0,
        "batch_size": 0,
        "n_saved": 0,
        "cluster_startup_s": 0.0,
        "load_s": 0.0,
        "scatter_s": 0.0,
        "correlate_s": 0.0,
        "pipeline_total_s": 0.0,
        "per_pair_ms": 0.0,
        "pairs_per_s": 0.0,
        "valid": "false",
        "error": "",
        "fft_backend": prov.get("fft_backend", ""),
        "fftw_wisdom": fftw_wisdom,
        "git_sha": prov.get("git_sha", ""),
        "git_dirty": prov.get("git_dirty", ""),
        "hostname": prov.get("hostname", ""),
        "cpu_count": prov.get("cpu_count", ""),
        "cpu_model": prov.get("cpu_model", ""),
    }

    tmpdir = tempfile.mkdtemp(prefix=f"bench_scale_w{workers}_t{threads}_")
    try:
        _apply_wisdom_policy(fftw_wisdom)

        # Resolve the base config and pin workers/threads + a temp output dir.
        config = bc.resolve_config(
            base_config_path,
            dataset=dataset,
            image_format=image_format,
            start_index=start_index,
            workers=workers,
            threads=threads,
            worker_memory=worker_memory,
        )
        batch_size = config.batch_size
        n_pairs, workload_mode = workload_for(workers, batch_size, n_images)
        config.data.setdefault("images", {})["num_images"] = n_pairs
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        config.data.setdefault("paths", {})["base_paths"] = [output_dir]

        base_result.update({"n_pairs": n_pairs, "batch_size": batch_size,
                            "workload_mode": workload_mode})

        # Precondition #1: the kernel reads OMP_NUM_THREADS (all cores if unset);
        # the harness owns it. Also passed to workers via start_cluster below.
        os.environ["OMP_NUM_THREADS"] = str(threads)

        from pivtools_cli.piv_cluster.cluster import start_cluster
        from pivtools_core.image_handling.load_images import load_images
        from pivtools_cli.piv.save_results import get_output_path
        from pivtools_cli.processing.dask_pipeline import (
            create_filter_pipeline, scatter_immutable_data,
        )

        t0 = time.perf_counter()
        cluster, client = start_cluster(config=config, worker_omp_threads=str(threads))
        cluster_startup = time.perf_counter() - t0

        try:
            camera = (config.data.get("paths", {}).get("camera_numbers") or [1])[0]
            source_path = Path(dataset)
            output_path = get_output_path(
                config, camera, use_uncalibrated=True, base_path_idx=0,
                piv_type="instantaneous",
            )

            t_pipeline = time.perf_counter()

            t0 = time.perf_counter()
            images = load_images(camera, config, source=source_path, batch_size=batch_size)
            t_load = time.perf_counter() - t0

            t0 = time.perf_counter()
            scattered = scatter_immutable_data(client, config, None, None, ensemble=False)
            t_scatter = time.perf_counter() - t0

            images = create_filter_pipeline(images, config, None)
            scattered_config = client.scatter(config, broadcast=True)

            num_chunks = len(images.chunks[0])
            max_in_flight = min(
                config.dask_workers_per_node * config.dask_max_in_flight_per_worker,
                num_chunks,
            )
            runs_0based = config.instantaneous_runs_0based
            vector_format = config.vector_format

            # In-pipeline warmup: push a few batches through the real workers
            # (untimed) so DLL load, FFTW plan creation, OMP pool and page cache
            # are warm before the timed run. Discard the warmup output.
            warmup_chunks = min(warmup_batches * config.dask_workers_per_node, num_chunks)
            if warmup_chunks > 0:
                _run_sliding_window(
                    client, images, warmup_chunks, max_in_flight,
                    scattered_config, scattered, output_path, runs_0based, vector_format,
                )
                for f in glob.glob(os.path.join(str(output_path), "*.mat")):
                    os.remove(f)

            t0 = time.perf_counter()
            saved = _run_sliding_window(
                client, images, num_chunks, max_in_flight,
                scattered_config, scattered, output_path, runs_0based, vector_format,
            )
            t_correlate = time.perf_counter() - t0
            t_total = time.perf_counter() - t_pipeline

            n_saved = len(saved)
            per_pair_ms = (t_correlate / n_pairs * 1000.0) if n_pairs else 0.0
            pairs_per_s = (n_pairs / t_correlate) if t_correlate > 0 else 0.0
            valid = n_saved > 0 and n_saved % n_pairs == 0
            error = "" if valid else f"Expected multiple of {n_pairs} saved, got {n_saved}"

            base_result.update({
                "n_saved": n_saved,
                "cluster_startup_s": round(cluster_startup, 2),
                "load_s": round(t_load, 3),
                "scatter_s": round(t_scatter, 3),
                "correlate_s": round(t_correlate, 3),
                "pipeline_total_s": round(t_total, 3),
                "per_pair_ms": round(per_pair_ms, 2),
                "pairs_per_s": round(pairs_per_s, 2),
                "valid": str(valid).lower(),
                "error": error,
            })
        finally:
            _quiet_shutdown(client, cluster)

    except Exception as e:  # noqa: BLE001 - record, never abort the sweep
        import traceback
        traceback.print_exc()
        base_result["error"] = str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

    return base_result


# --- sweep orchestration ---------------------------------------------------


def _completed_keys(csv_path: Path) -> set:
    """(workers, threads, iteration) tuples already in the CSV (resume support)."""
    import csv as _csv

    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            try:
                done.add((int(r["workers"]), int(r["threads"]), int(r["iteration"])))
            except (KeyError, ValueError):
                continue
    return done


def run_sweep(
    base_config_path: str,
    dataset: str,
    configs: list[dict[str, Any]],
    csv_path: Path,
    *,
    n_images: Optional[int],
    n_iterations: int = 1,
    total_cores: int,
    max_workers: int,
    image_format: Optional[list[str]] = None,
    start_index: Optional[int] = None,
    worker_memory: Optional[str] = None,
    fftw_wisdom: str = WISDOM_SHARED,
    warmup_batches: int = 1,
) -> Path:
    """Run every config x iteration, appending each result to ``csv_path`` as it
    finishes (crash-safe). Skips rows already present so ``--resume`` works."""
    completed = _completed_keys(csv_path)
    if not csv_path.exists():
        bc.write_csv_header(csv_path, CSV_FIELDS)

    provenance = bc.build_provenance(dataset=dataset, cache_policy=f"scaling/{fftw_wisdom}")

    tasks = [
        (cfg, it)
        for cfg in configs
        for it in range(n_iterations)
        if (cfg["workers"], cfg["threads"], it) not in completed
    ]
    if not tasks:
        print("All configurations already complete. Nothing to run.")
        return csv_path

    total = len(tasks)
    started = datetime.now()
    walls: list[float] = []
    print(f"\nSCALING: {total} runs  | backend={provenance['fft_backend']} "
          f"wisdom={fftw_wisdom} | {dataset}")
    print(f"  results: {csv_path}\n")

    for i, (cfg, it) in enumerate(tasks):
        eta = ""
        if walls:
            avg = sum(walls) / len(walls)
            rem = (total - i) * avg
            eta = f"  ETA {(datetime.now() + timedelta(seconds=rem)).strftime('%H:%M:%S')} (~{rem/60:.0f}m)"
        print(f"[{i+1}/{total}] {cfg['label']}: {cfg['workers']}w x {cfg['threads']}t"
              f" = {cfg['workers']*cfg['threads']} cores{eta}", flush=True)

        t0 = time.perf_counter()
        result = run_single(
            base_config_path, dataset,
            workers=cfg["workers"], threads=cfg["threads"], label=cfg["label"],
            n_images=n_images, image_format=image_format, start_index=start_index,
            worker_memory=worker_memory, fftw_wisdom=fftw_wisdom,
            warmup_batches=warmup_batches, total_cores=total_cores,
            iteration=it, provenance=provenance,
        )
        walls.append(time.perf_counter() - t0)
        bc.append_csv_row(csv_path, CSV_FIELDS, result)

        if result["valid"] == "true":
            print(f"  -> {result['pairs_per_s']:.1f} pairs/s, "
                  f"{result['per_pair_ms']:.1f} ms/pair, "
                  f"load={result['load_s']:.1f}s correlate={result['correlate_s']:.1f}s\n",
                  flush=True)
        else:
            print(f"  -> INVALID: {result['error']}\n", flush=True)

    print(f"Done: {total} runs in {(datetime.now()-started).total_seconds()/60:.1f} min")
    return csv_path


def print_dry_run(
    configs: list[dict[str, Any]], n_iterations: int, n_images: Optional[int], batch_hint: int
) -> None:
    """Print the test matrix without running anything."""
    total = len(configs) * n_iterations
    mode = "fixed" if n_images is not None else "per_worker"
    print(f"\nDRY RUN: {len(configs)} configs x {n_iterations} iter = {total} runs"
          f"  (workload={mode})")
    print(f"\n{'#':>3} {'workers':>8} {'threads':>8} {'cores':>6} {'pairs':>8} {'label':<14}")
    print("-" * 52)
    for i, cfg in enumerate(configs, 1):
        w, t = cfg["workers"], cfg["threads"]
        n_pairs, _ = workload_for(w, batch_hint, n_images)
        print(f"{i:>3} {w:>8} {t:>8} {w*t:>6} {n_pairs:>8} {cfg['label']:<14}")
    print()
