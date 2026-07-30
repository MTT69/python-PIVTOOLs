# PIVTOOLs benchmark & profile harness

One tool, `bench.py`, with three subcommands. It characterises PIVTOOLs speed for
the SoftwareX paper and proves whether the codelet FFT build is faster than the
FFTW build, by measuring **whatever `libbulkxcorr2d` is built in the worktree you
run it from** and stamping provenance so two runs can be compared honestly.

| Subcommand | Question | How it runs |
|---|---|---|
| `scaling` | How does throughput scale across workers and threads? | Full production Dask pipeline, end-to-end |
| `profile` | Where does the per-pair time go (read, each pass, save)? | Serial, one pair at a time, real production code paths |
| `compare` | Is the codelet FFT actually faster than FFTW? | Joins two result sets into A/B figures |

All results (CSV + JSON) go to `results/`. A/B figures go to `../figures/debug/`.
The top of this folder stays source-only.

---

## Modules

| File | Role |
|---|---|
| `bench.py` | Thin CLI — argparse dispatch only |
| `bench_common.py` | Config resolution, provenance + FFT-backend detection, the C sub-kernel timer binding, result IO, the A/B provenance guard |
| `bench_scaling.py` | Distributed worker×thread sweep (end-to-end) |
| `bench_profile.py` | Complete per-pair time budget (read + per-pass compute + save) |
| `bench_plots.py` | Scaling figures + A/B compare figures |
| `test_bench.py` | Unit tests for the pure logic (sweep gen, workload split, provenance guard) |

There are no hard-coded dataset paths or machine constants. A base `config.yaml`
supplies every PIV setting (windows, passes, fit method, save mode); the CLI
overrides only the dataset location, image count, sweep axes, and thread/worker
counts, so the science stays pinned and reproducible.

Run everything with the project venv (the interpreter that has the built C
extension) — the system Python won't load `libbulkxcorr2d`.

---

## Quick start

```bash
# Where does the per-pair time go? (serial, one backend, this machine)
python profile/bench.py profile --config base.yaml --dataset /path/to/images \
    --n-images 20 --threads 1

# How does it scale? (fixed 200-pair workload => strong scaling)
python profile/bench.py scaling --config base.yaml --dataset /path/to/images \
    --n-images 200 --sweep workers --max-workers 8

# Preview the sweep matrix without running anything
python profile/bench.py scaling --config base.yaml --dataset x --sweep all \
    --total-cores 8 --n-images 200 --dry-run
```

---

## Profile mode — the per-pair budget

Runs serially on the real files, exercising the production read, correlate, and
save paths, and reports a complete per-pair budget:

    read (load_images) → [per-pass compute sections] → save (save_piv_result_distributed)

Output is a JSON (`results/profile_<ts>.json`) plus the budget figure
(`../figures/debug/<date>-bench-profile-budget-<backend>.png`): a Read bar, one
stacked compute bar per pass (Prep/Warp, Cross-corr, Outlier, Infilling, Other),
and a Save bar. Sanity holds by construction — `read + Σ(sections) + save ≈ wall`.

Key flags: `--threads` (default 1), `--iterations` (default 3, mean±std),
`--cache {warm,cold}`, `--mode {instantaneous,ensemble}` (ensemble is a named
follow-up), `--camera`, `--no-warmup`, `--out`, `--no-plot`.

### C sub-kernel split — FFT vs peak-fit

When the kernel is built with the timing instrumentation (flag-gated
`bulkxcorr2d_{set_timing_enabled,reset_timing,get_timing}`), the JSON also carries
`kernel_split`: the per-pair FFT time vs peak-fit time inside `bulkxcorr2d`. This is
the sharpest A/B number — the codelet changes only the FFT half, so this isolates
the true FFT speedup from the constant fit cost.

- The numbers are **thread-summed**. At `--threads > 1` they exceed the
  `bulkxcorr2d` wall by ~thread count; at `--threads 1` they reconcile with it.
  `fft_fraction` is thread-count-independent — use it as the comparison quantity.
- Older binaries without the timers simply omit `kernel_split` (no error).

### Page-cache control

A serial read loop warms the OS page cache, and an A/B where one backend runs
first warms it for the other. Both policies reach a known cache state *before* the
timed read, so run order doesn't bias the read bar:

- `--cache warm` (default) — prime with a full discarded read, then time
  steady-state RAM-served reads. Reproducible; isolates compute.
- `--cache cold` — evict the dataset with `vmtouch -e` first, then time a true
  physical read. Linux/Iridis only; on Mac/Windows it warns and falls back to warm.

Neither is the production regime (concurrent workers over a parallel FS) — that is
scaling mode's `load_s`. The read bar is labelled by cache policy so this is never
hidden.

---

## Scaling mode — the worker×thread sweep

Runs the full Dask pipeline (cluster → load → scatter → correlate → save) at a
range of worker/thread combinations and records end-to-end throughput. It is
end-to-end **only**: the correlator runs inside worker processes, so the per-section
breakdown can't cross back to the client — that's profile mode's job. What scaling
sees that profile can't is the real behaviour under load — I/O contention
(`load_s`), scatter cost, and the throughput ceiling as cores fill.

Sweep axes are generated from the core budget, so a laptop and a 192-core node both
get sensible matrices:

| `--sweep` | What it varies |
|---|---|
| `threads` | 1 worker, OMP threads 1,2,4,…,total-cores |
| `workers` | Fixed `--worker-threads` (default 2), workers 1,2,4,… up to the core/RAM cap |
| `matrix` | All w×t with w·t ≤ total-cores |
| `oversub` | w·t in (total-cores, 2·total-cores] |
| `all` | The union, deduplicated by (workers, threads) |

| Flag | Meaning |
|---|---|
| `--n-images N` | **Fixed** workload → strong scaling S(p)=T(1)/T(p). Omit for the per-worker fallback (constant batches/worker, recorded as `workload_mode=per_worker`, *not* a clean strong-scaling curve). |
| `--total-cores` / `--max-workers` | Core budget and worker (RAM) cap. Default: `os.cpu_count()`. |
| `--worker-memory` | Per-worker Dask memory limit, e.g. `3.2GB`. |
| `--iterations` | Repeats per config (error bars). |
| `--warmup-batches` | Untimed warmup batches per worker before the timed run (default 1). |
| `--fftw-wisdom {shared,per-worker}` | FFTW wisdom policy — see below. |
| `--resume CSV` / `--plots-only CSV` | Append to / re-plot an existing CSV (crash-safe: every completed config is already written). |
| `--dry-run` | Print the matrix and exit. |

Output: `results/scaling_<ts>.csv` (one row per config, with provenance columns)
plus `_throughput.png` and `_strong_scaling.png` figures.

### FFTW wisdom policy (precondition 3)

Multiple worker *processes* share one wisdom file (`$HOME/.pypivtools_fftw_wisdom`);
the in-kernel lock guards threads, not processes. So the FFTW arm can pay a
write-race cost the codelet arm doesn't.

- `--fftw-wisdom shared` (default) is the real current behaviour — it films that
  contention.
- `--fftw-wisdom per-worker` **raises**. Giving each process its own wisdom file
  needs a C hook (`PIV_FFTW_WISDOM_PATH` in `xcorr_cache.c`); the kernel currently
  keys wisdom off `$HOME` only, and hijacking `HOME` per worker would corrupt
  unrelated state. That hook is deferred (precondition-3 / propagation work), so the
  flag refuses rather than faking isolation. The chosen policy is stamped into every
  row regardless.

---

## A/B — FFTW vs codelet

The harness is binary-agnostic. Build the same harness in both worktrees, run each
arm once, then compare:

```bash
# arm A — FFTW build
cd PyPIVTools && python profile/bench.py profile --config base.yaml --dataset DIR --n-images 20
                 python profile/bench.py scaling --config base.yaml --dataset DIR --n-images 200 --sweep workers

# arm B — codelet build
cd PyPIVTools-fftw && python profile/bench.py profile ...   # (same flags)
                      python profile/bench.py scaling ...

# join them
python profile/bench.py compare --a PyPIVTools/profile/results --b PyPIVTools-fftw/profile/results
```

`--a` / `--b` accept a results directory (newest `scaling_*.csv` and `profile_*.json`
are picked) or an explicit file. `compare`:

1. **Guards** the two provenance stamps — warns loudly if hostname, CPU model/count,
   filesystem, cache policy, dataset, or version differ (a cross-machine or
   cross-cache A/B is not valid), and flags the case where both arms use the *same*
   backend (nothing to compare).
2. Renders **throughput** A/B (`<date>-ab-fftw-vs-codelet-throughput.png`) when both
   sides have a scaling CSV — on I/O-bound data the curves overlap, the honest parity.
3. Renders the **per-pair budget** A/B (`<date>-ab-fftw-vs-codelet-budget.png`) when
   both sides have a profile JSON — solid (A) vs hatched (B), so the Cross-corr
   segment visibly shrinks while read/save/Other stay fixed, with the kernel
   FFT-vs-fit split annotated when both arms carry it.

> The codelet kernel does not yet carry the C timing instrumentation. Until it is
> propagated, the codelet profile JSON has no `kernel_split` and the budget figure
> simply omits the split annotation — everything else compares normally.

---

## Honest limits

- **Profile is single-stream.** Serial reads give an accurate per-pair *compute*
  budget and an I/O *floor*, not aggregate I/O under worker contention — that is
  scaling's `load_s`.
- **Strong scaling needs `--n-images`.** Without it the workload grows with workers
  and the speedup curve is not a strong-scaling curve.
- **Thread scaling plateaus** around the FFT memory-bandwidth limit; worker scaling
  goes further (each worker has its own FFTW plan and memory). Oversubscription rows
  show throughput collapse on purpose — diagnostic, not a bug to "fix".
- **Don't trust one iteration.** Use `--iterations`.

---

## Tests & lint

```bash
pytest profile/test_bench.py        # pure-function tests (no Dask, no figures)
pre-commit run --all-files          # from an env that has the toolchain
```

---

## Not done yet

- SLURM templates for Iridis (`slurm/scaling.sbatch`, `slurm/profile.sbatch`).
- Propagating the harness + the same C sub-kernel timing into `PyPIVTools-fftw`,
  rebuilding the codelet `.so`, and running the real A/B.
