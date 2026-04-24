# PIVTOOLs Profiling & Benchmarking

Scripts for measuring PIVTOOLs performance. Two flavours:

- **Benchmarks** measure end-to-end throughput while varying one axis (I/O, batch size, resolution, threads, workers). They answer *"how fast?"*
- **Profilers** use the correlators' built-in `_profile_section(...)` instrumentation to break a single run down into named sub-sections. They answer *"where is the time going?"*

All outputs (CSVs + PNGs) go to `results/` — the top of this folder stays source-only.

---

## Prerequisites

- Project venv at `python-PIVTOOLs/env/Scripts/python.exe`. Never use the system Python — it doesn't have the built C extensions.
- Real PIV images on disk. Default paths assume:
  - `#current_processing/4000_images_channel/planar_images/` (4MP)
  - `.../planar_images/1mp/` (1MP, created by `create_1mp_crops.py`)
- FFTW wisdom at `~/.pypivtools_fftw_wisdom` is primed automatically on first run; expect the very first benchmark to be slightly slower.

---

## Quick start

Run the full 8-step suite (takes ~1-2 hours on a 20-core machine):

```bash
env/Scripts/python.exe profile/run_all_benchmarks.py
```

Run a single step:

```bash
env/Scripts/python.exe profile/run_all_benchmarks.py --step 3   # thread scaling only
```

Run any script directly with its own CLI (see each script's docstring for full flags).

---

## Script catalogue

### Orchestrator

| Script | Purpose |
|---|---|
| `run_all_benchmarks.py` | Runs the 8-step suite in order. `--step N` runs only step N. |

### Benchmarks (end-to-end throughput)

| Script | Axis measured | Typical runtime |
|---|---|---|
| `benchmark_io_decomposition.py` | Image read, result write, Dask startup, save-mode combos — isolates each I/O component so the "residual I/O" bucket in the pipeline has a story. | ~5 min |
| `benchmark_batch_and_resolution.py` | Batch size sweep (1..20), resolution (4MP vs 1MP), 2-pass vs 3-pass, save I/O matrix. All at 10 OMP threads, no Dask. | ~10 min |
| `benchmark_scaling.py` | Full Dask pipeline: thread sweep, worker sweep, combined (workers × threads), oversubscription. Crash-safe CSV writes, auto-generates plots. `--sweep threads\|workers\|combined\|oversub\|all` | ~20–60 min per sweep |
| `create_1mp_crops.py` | Helper — centre-crops 4MP images to 1MP for the 1MP scaling benchmark. Not a benchmark itself. | ~1 min |

### Profilers (per-section timings for a single run)

| Script | Target | What it breaks down |
|---|---|---|
| `profile_piv.py` | Instantaneous (`InstantaneousCorrelatorCPU.correlate_batch`) | 9 top-level sections (predictor, xcorr, outlier, infill, ...) + 3 predictor sub-sections |
| `profile_ensemble.py` | Ensemble (overview) | Correlation-vs-finalization split per pass |
| `profile_ensemble_correlation.py` | Ensemble correlation only | 10-ish sections incl. `pc_gaussian_smooth`, `pc_fused_warp`, `xcorr_AB/AA/BB` |
| `profile_ensemble_fitting.py` | Ensemble finalization only | 13 accumulator sections incl. `fitting`, `velocity_extraction`, `outlier_detection`, `infilling` |

All profilers take a preset (`4mp`, `25mp`, `both`) and common flags: `--pairs`, `--iterations`, `--threads`, `--windows`, `--fit-method kspace\|gaussian`.

---

## When to run which

| Question | Script |
|---|---|
| Is the pipeline scaling across cores? | `benchmark_scaling.py --sweep threads` then `--sweep workers` |
| Why is a single correlation call slow? | `profile_piv.py 4mp` or `profile_ensemble_correlation.py 4mp` |
| Is fitting or correlation the bottleneck in ensemble mode? | `profile_ensemble.py` (overview), then drill into whichever dominates |
| How much does compression cost on save? | `benchmark_io_decomposition.py --test 2` |
| What batch size is optimal? | `benchmark_batch_and_resolution.py` |
| Does Dask scheduling eat measurable time? | `benchmark_io_decomposition.py --test 3` |

---

## Outputs

Every benchmark writes timestamped files into `results/`:

```
results/
  batch_sweep_YYYYMMDD_HHMMSS.csv
  image_read_YYYYMMDD_HHMMSS.csv
  result_write_YYYYMMDD_HHMMSS.csv
  dask_overhead_YYYYMMDD_HHMMSS.csv
  save_io_YYYYMMDD_HHMMSS.csv
  resolution_breakdown_YYYYMMDD_HHMMSS.csv
  resolution_summary_YYYYMMDD_HHMMSS.csv
  scaling_YYYYMMDD_HHMMSS.csv
  scaling_YYYYMMDD_HHMMSS_thread_scaling.png      # thread sweeps
  scaling_YYYYMMDD_HHMMSS_worker_scaling_t{T}.png # worker sweeps (per fixed thread count)
  scaling_YYYYMMDD_HHMMSS_strong_scaling.png
  scaling_YYYYMMDD_HHMMSS_scaling_heatmap.png
  scaling_YYYYMMDD_HHMMSS_cores_vs_throughput.png
```

Profilers print to stdout — no files. Redirect if you want to keep a transcript.

The current `results/` folder holds the most recent run of each family as a reference. Older runs were deleted during the 2026-04-13 cleanup.

### Resuming & re-plotting a crashed scaling run

```bash
env/Scripts/python.exe profile/benchmark_scaling.py --resume results/scaling_20260318_123221.csv
env/Scripts/python.exe profile/benchmark_scaling.py --plots-only results/scaling_20260318_123221.csv
```

---

## Gotchas

- **OneDrive is slow.** The `#current_processing/` source is inside OneDrive; cold reads can double benchmark times. Do a warm read first or use the image-read benchmark's "warm cache" number.
- **Don't trust one iteration.** All profilers accept `--iterations N` (default 3–5). Single runs on Windows get hit by background services.
- **Oversubscription plots are diagnostic, not aspirational.** Workers × threads > physical cores will show throughput collapse. That's the point — don't "fix" it by changing the sweep range.
- **Thread scaling plateau is expected around 8 threads** for 4MP (FFT memory bandwidth bound). Worker scaling continues further because each worker gets its own FFTW plan and memory.
- **`benchmark_scaling.py` creates temp directories** for its output and cleans them up. If it crashes, check `%TEMP%` for orphaned `pivtools_bench_*` folders.
