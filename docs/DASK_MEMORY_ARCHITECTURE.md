# Dask and Memory Architecture for PIV Pipelines

This document explains the Dask distributed computing architecture and memory management patterns used in the instantaneous and ensemble PIV processing pipelines.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Worker Allocation Strategy](#worker-allocation-strategy)
3. [Data Scattering Patterns](#data-scattering-patterns)
4. [Filter-Correlation Concurrency](#filter-correlation-concurrency)
5. [Threading and Core Configuration](#threading-and-core-configuration)
6. [Task Graph Differences](#task-graph-differences)
7. [Memory Usage Patterns](#memory-usage-patterns)

---

## Architecture Overview

The PIV processing system uses a **two-stage worker architecture** to handle the fundamentally different parallelization characteristics of preprocessing (filtering) vs. correlation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          example.py (Main Process)                   │
│                                                                      │
│  1. Start Dask cluster                                               │
│  2. Load images (lazy - creates delayed tasks)                       │
│  3. Create UnifiedBatchPipeline                                      │
│  4. Call pipeline.process()                                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    UnifiedBatchPipeline                              │
│                                                                      │
│  Worker Allocation:                                                  │
│  ├── Filter Workers (1-2)     ─ Temporal filters (POD, time)        │
│  │                               need batch in memory                │
│  │                                                                   │
│  └── Correlation Workers (N-1) ─ PIV is embarrassingly parallel     │
│                                   each pair independent              │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Two Worker Pools?

**Filtering (Temporal/Batch)**:
- POD filtering requires the entire batch (e.g., 25-50 images) loaded into memory to compute SVD
- Time filtering (`subtract_local_min`) needs temporal context across frames
- Cannot distribute these across many workers without duplicating memory
- Uses **multi-threading within a single worker** to parallelize SVD computation

**Correlation (Embarrassingly Parallel)**:
- Each image pair is completely independent
- Can distribute across as many workers as available
- Each worker only needs ~280 MB (1 image pair + PIV overhead)

---

## Worker Allocation Strategy

### Configuration Settings

From `config.yaml`:

```yaml
processing:
  dask_workers_per_node: 4      # Total Dask workers
  dask_threads_per_worker: 1    # Threads per worker
  dask_memory_limit: 3GB        # Memory per worker
  filter_worker_count: 1        # Workers for filtering
  filter_omp_threads: 2         # OMP threads for pipelined filtering
  omp_threads: 2                # OMP threads for correlation workers
```

### Worker Assignment Logic

From `config.py:1775-1797`:

```python
def get_filter_worker_allocation(self, total_workers: int):
    """
    Determine filter and correlation worker counts.

    Examples:
        10 cores, count=1 → (1, 9)  # 1 filter, 9 correlation
        5 cores, count=2 → (2, 3)   # 2 filter, 3 correlation
    """
    filter_workers = max(1, min(self.filter_worker_count, total_workers - 1))
    return filter_workers, total_workers - filter_workers
```

### Automatic Filter Worker Count

The system auto-determines filter worker count based on filter types (`config.py:1744-1773`):

| Filter Type | Worker Count | Reason |
|-------------|--------------|--------|
| Temporal (POD, time) | 1 | Full batch must be in memory |
| Spatial only | 2 | Can parallelize somewhat |
| No filters | 2 | Default |

---

## Data Scattering Patterns

### What Gets Scattered (Broadcast)

The `_scatter_immutable_data()` method (`batch_pipeline.py:692-722`) broadcasts immutable data once to **all workers**:

```python
def _scatter_immutable_data(self, vector_masks):
    # 1. Correlator cache - FFT plans, window weights, grid info
    temp_correlator = make_correlator_backend(self.config, ensemble=...)
    correlator_cache = temp_correlator.get_cache_data()
    scattered_cache = self.client.scatter(correlator_cache, broadcast=True)

    # 2. Vector masks - boolean masks for each PIV pass
    if vector_masks:
        scattered_masks = self.client.scatter(vector_masks, broadcast=True)

    # 3. Pixel mask - intensity mask applied during preprocessing
    if self.pixel_mask is not None:
        scattered_pixel_mask = self.client.scatter(self.pixel_mask, broadcast=True)
```

**Broadcast = True**: Data is replicated to all workers once, avoiding repeated transfer.

### Per-Pass Scattering (Ensemble Only)

For multi-pass ensemble PIV, the predictor field is scattered at the start of each pass:

```python
# batch_pipeline.py:195-198
if predictor_field is not None:
    scattered_predictor = self.client.scatter(predictor_field, broadcast=True)
    logging.info(f"[Pass {pass_idx + 1}] Broadcast predictor field from previous pass")
```

### Per-Batch Scattering

After filtering, the filtered image pairs are scattered to correlation workers:

```python
# batch_pipeline.py:793-798
pairs = [filtered_batch[i] for i in range(filtered_batch.shape[0])]
pair_indices = list(range(len(pairs)))

# Scatter pairs to correlation workers ONLY (not filter workers)
scattered_pairs = self.client.scatter(pairs, workers=self.corr_workers)
```

**Key Point**: `workers=self.corr_workers` restricts data to correlation workers only, preventing unnecessary memory usage on filter workers.

### Cross-Batch Tree Reduction for Ensemble Results

Ensemble correlation results are accumulated using **cross-batch tree reduction** on workers to minimize data transfer:

```python
# batch_pipeline.py - cross-batch reduction pattern

# 1. Collect ALL correlation futures across ALL batches
all_corr_futures = []
for batch_idx in range(num_batches):
    # Filter batch (sequential - RAM constraint)
    filtered_batch = filter_future.result()

    # Submit correlation tasks (parallel)
    corr_futures = correlate_batch(filtered_batch, ...)

    # Collect futures (don't reduce yet!)
    all_corr_futures.extend(corr_futures)

# 2. Single tree reduction across everything
while len(all_corr_futures) > 1:
    new_futures = []
    for i in range(0, len(all_corr_futures), 2):
        if i + 1 < len(all_corr_futures):
            combined = self.client.submit(
                _reduce_ensemble_results,
                all_corr_futures[i],
                all_corr_futures[i + 1],
                workers=self.corr_workers,
            )
            new_futures.append(combined)
        else:
            new_futures.append(all_corr_futures[i])
    all_corr_futures = new_futures

# 3. Only final accumulated result transferred to main
pass_accumulated = all_corr_futures[0].result()
```

**Key Point**: Instead of transferring ~900MB per batch to main process, correlation planes are accumulated across ALL batches on workers. Only the final sum (900MB total, not 900MB × num_batches) is transferred.

| Batches | Before (per-batch) | After (cross-batch) |
|---------|-------------------|---------------------|
| 4 | 3.6 GB | 900 MB |
| 10 | 9 GB | 900 MB |
| 20 | 18 GB | 900 MB |

---

## Filter-Correlation Concurrency

### Pipelined Execution Model

The pipeline overlaps filtering of batch N+1 with correlation of batch N:

```
Time ─────────────────────────────────────────────────────────────────►

Batch 0:  [FILTER]──────────[CORRELATE]
                   │
Batch 1:           └──[FILTER]──────────[CORRELATE]
                              │
Batch 2:                      └──[FILTER]──────────[CORRELATE]
```

### Implementation Details

From `batch_pipeline.py:221-276`:

```python
while batch_idx < num_batches:
    # Wait for current filter to complete
    filtered_batch = filter_future.result()

    # Start correlation for THIS batch (NON-BLOCKING - returns futures)
    corr_futures = self._correlate_ensemble_batch_async(
        filtered_batch, scattered_cache, scattered_masks, ...
    )

    # OVERLAP: Submit NEXT filter WHILE this batch correlates
    next_batch_idx = batch_idx + 1
    if next_batch_idx < num_batches:
        filter_future = self.client.submit(
            _filter_batch_worker,
            next_slice,
            ...,
            is_first_batch=False,  # Use reduced OMP threads
            workers=[next_worker],
        )

    # NOW wait for current correlation
    results = self.client.gather(corr_futures)
```

### Consistent Threading Across All Batches

All filter batches use `config.omp_threads` for consistent threading:

From `_filter_batch_worker()` (`batch_pipeline.py:879-883`):

```python
# Use config.omp_threads for all batches (consistent threading)
worker_cores = int(config.omp_threads)

os.environ["OMP_NUM_THREADS"] = str(worker_cores)
os.environ["MKL_NUM_THREADS"] = str(worker_cores)
```

This ensures predictable resource usage regardless of batch position in the pipeline.

---

## Threading and Core Configuration

### Environment Variables

| Variable | Where Set | Purpose |
|----------|-----------|---------|
| `OMP_NUM_THREADS` | Main process (`example.py:231`) | Batch filtering in main process |
| `OMP_NUM_THREADS` | Workers (`cluster.py:115-121`) | Correlation workers |
| `OMP_NUM_THREADS` | Filter worker (`batch_pipeline.py:887`) | Filter workers during pipelining |
| `MKL_NUM_THREADS` | Filter worker | NumPy/SciPy linear algebra |
| `MALLOC_TRIM_THRESHOLD_` | Main process | Memory management |

### Threading Flow

```
example.py:
├── OMP_NUM_THREADS = cpu_count()  [for main process batch filtering]
├── worker_omp_threads = config.omp_threads  [stored for workers]
│
└── start_cluster():
    └── client.run(set_worker_omp_threads, omp_threads=worker_omp_threads)
        └── Each worker: OMP_NUM_THREADS = omp_threads (e.g., "2")

UnifiedBatchPipeline:
├── First filter batch:  OMP_NUM_THREADS = cpu_count()
└── Subsequent batches:  OMP_NUM_THREADS = config.filter_omp_threads
```

---

## Task Graph Differences

### Instantaneous PIV - Atomic Task Graph

Each image pair is processed and saved in a **single atomic task**:

```
┌───────────────────────────────────────────────────────────────┐
│                 _process_and_save_single_pair                  │
│                                                                │
│   image_pair ──► correlate_batch() ──► save_piv_result() ──► saved_path
│                      │                       │                 │
│                 [correlation]          [.mat write]            │
│                 [peak fitting]                                 │
│                 [outlier detection]                            │
│                 [infilling]                                    │
└───────────────────────────────────────────────────────────────┘
```

From `piv.py:24-73`:

```python
def _process_and_save_single_pair(
    image_pair, frame_number, config, scattered_masks,
    scattered_cache, output_path, runs_to_save, vector_format
) -> str:
    """
    Combined PIV processing and saving for a single image pair.
    Combines PIV computation and saving into a single atomic operation
    to reduce task graph complexity.
    """
    # Process PIV (correlation + peak fitting + outlier + infill)
    piv_result = _piv_single_pass(image_pair, config, scattered_masks, scattered_cache)

    # Save immediately to avoid accumulating results in memory
    saved_path = save_piv_result_distributed(
        piv_result, output_path, frame_number, runs_to_save, vector_format
    )

    return saved_path
```

### Ensemble PIV - Separated Task Graph

Ensemble processing requires **accumulation across all pairs** before peak fitting:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          BATCH PROCESSING                            │
│                                                                      │
│  Pair 1 ──► correlate_batch_for_accumulation() ──► {AA, BB, AB sums}│
│  Pair 2 ──► correlate_batch_for_accumulation() ──► {AA, BB, AB sums}│
│  ...                                                                 │
│  Pair N ──► correlate_batch_for_accumulation() ──► {AA, BB, AB sums}│
│                                                                      │
│                              ▼                                       │
│                     accumulator.accumulate_batch()                   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ (after all batches)
┌─────────────────────────────────────────────────────────────────────┐
│                          FINALIZATION                                │
│                                                                      │
│  Running sums ──► Single-pass formula ──► Peak fitting ──► Results  │
│                                                                      │
│  R_AB = <A⋆B> - <A>⋆<B>   (background subtraction)                  │
│  R_AA = <A⋆A> - <A>⋆<A>                                             │
│  R_BB = <B⋆B> - <B>⋆<B>                                             │
│                                                                      │
│  Gaussian fitting → displacements + uncertainty                      │
│  Outlier detection → infilling → save                                │
└─────────────────────────────────────────────────────────────────────┘
```

From `batch_pipeline.py:915-959`:

```python
def _correlate_ensemble_pair_worker(
    image_pair, pair_idx, config, scattered_cache, scattered_masks,
    scattered_predictor, pass_idx, output_path, is_first_batch
) -> dict:
    """
    Correlate single pair for ensemble ACCUMULATION.
    Returns correlation plane sums (AA, BB, AB) and warp sums.
    """
    correlator = EnsembleCorrelatorCPU(config, precomputed_cache=scattered_cache, ...)

    # Returns SUMS, not final results
    result = correlator.correlate_batch_for_accumulation(
        image_pair[np.newaxis, ...],  # Add batch dimension
        config,
        pass_idx=pass_idx,
        predictor_field=scattered_predictor,
        ...
    )

    return result  # Dict with: warp_A_sum, warp_B_sum, corr_AA_sum, corr_BB_sum, corr_AB_sum
```

### Side-by-Side Comparison

| Aspect | Instantaneous | Ensemble |
|--------|---------------|----------|
| Task granularity | 1 task = 1 pair (complete) | 1 task = 1 pair (partial sums) |
| Result type | `PIVResult` (final) | `dict` of running sums |
| Peak fitting | Per-pair, in task | After all pairs accumulated |
| Saving | Per-pair, in task | After finalization |
| Memory pattern | Stream and discard | Accumulate then finalize |

---

## Memory Usage Patterns

### Lazy Image Loading

From `load_images.py:268-346`:

```python
def load_images(camera, config, source=None) -> da.Array:
    """
    Load images using pure lazy loading.

    Memory Efficiency:
    - Creates N delayed objects (~1 KB each) for N images
    - Main process memory: ~N KB (minimal, just task graph)
    - Worker memory: Only 1 image pair at a time (~80 MB)
    - Each worker: load → process → save → free → next
    - Peak worker memory: ~280 MB (1 image + PIV overhead)
    """
    # Create one delayed task per image pair
    delayed_image_pairs = [
        delayed_image_pair(idx, camera_path, camera, config)
        for idx in range(1, num_pairs + 1)
    ]

    # Convert to Dask array - STILL LAZY, no computation yet!
    dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]
    pairs_stack = da.stack(dask_pairs, axis=0)

    return pairs_stack  # Shape: (num_frame_pairs, 2, H, W), all lazy
```

### Batch Loading for Filtering

When a batch is needed for filtering, it's loaded with multi-threading:

```python
# batch_pipeline.py:890-892
with dask.config.set(scheduler='threads', num_workers=worker_cores):
    batch = batch_images.compute()  # Now actually loads into memory
```

### Memory Budget Example

For 100 image pairs (1024x1024, float32):

| Component | Memory |
|-----------|--------|
| Main process (task graph) | ~100 KB |
| One image pair | ~8 MB |
| Filter batch (25 pairs) | ~200 MB |
| PIV processing overhead | ~200 MB |
| Correlator cache (scattered) | ~50 MB/worker |
| **Peak per filter worker** | **~450 MB** |
| **Peak per correlation worker** | **~280 MB** |

### Memory Cleanup

The pipeline explicitly cleans up memory after each pass:

```python
# batch_pipeline.py:289-294
if scattered_predictor is not None:
    del scattered_predictor
    scattered_predictor = None
    gc.collect()
    logging.debug(f"[Pass {pass_idx + 1}] Cleaned up scattered predictor")
```

---

## Summary Diagram

```
                           CONFIG SETTINGS
                    ┌───────────────────────────┐
                    │ dask_workers_per_node: 4  │
                    │ filter_worker_count: 1    │
                    │ omp_threads: 2            │
                    │ filter_omp_threads: 2     │
                    │ batch_size: 25            │
                    └───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        DASK CLUSTER (4 workers)                          │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────┐ │
│  │ FILTER WORKER 0 │    │          CORRELATION WORKERS 1-3            │ │
│  │                 │    │                                             │ │
│  │ OMP_THREADS=2-N │    │  OMP_THREADS=2 (all)                       │ │
│  │ (dynamic)       │    │                                             │ │
│  │                 │    │  ┌─────────┐ ┌─────────┐ ┌─────────┐      │ │
│  │ [Load batch]    │    │  │ Pair 1  │ │ Pair 2  │ │ Pair 3  │ ...  │ │
│  │ [POD filter]    │    │  │ correlate│ │correlate│ │correlate│      │ │
│  │ [Time filter]   │    │  └─────────┘ └─────────┘ └─────────┘      │ │
│  └────────┬────────┘    └──────────────────▲──────────────────────────┘ │
│           │                                │                             │
│           └── scattered_pairs ─────────────┘                             │
│                                                                          │
│  SCATTERED DATA (broadcast=True to all workers):                         │
│  ├── correlator_cache (FFT plans, window weights)                        │
│  ├── vector_masks (boolean masks per pass)                               │
│  ├── pixel_mask (intensity mask)                                         │
│  └── predictor_field (ensemble multi-pass only, per-pass)                │
└──────────────────────────────────────────────────────────────────────────┘

PIPELINE FLOW:

INSTANTANEOUS:
  images (lazy) → batch filter → scatter pairs → map(_process_and_save_single_pair) → .mat files
                                                      │
                                                      └── correlate + peak fit + save (atomic)

ENSEMBLE:
  images (lazy) → batch filter → scatter pairs → map(_correlate_ensemble_pair_worker) → gather sums
                                                      │
                                                      └── correlate only (returns sums)
                                                                    │
                                                                    ▼
                                              accumulator.accumulate_batch() [repeat per batch]
                                                                    │
                                                                    ▼
                                              accumulator.finalize_pass() → peak fit → save
```

---

## References

- `example.py`: Entry point, cluster setup, main loop
- `batch_pipeline.py`: `UnifiedBatchPipeline` class, worker allocation, pipelining
- `piv.py`: `_process_and_save_single_pair`, instantaneous task function
- `cpu_ensemble.py`: `EnsembleCorrelatorCPU`, correlation for accumulation
- `single_pass_accumulator.py`: `SinglePassAccumulator`, ensemble math
- `preprocess.py`: `apply_filters_to_batch`, filter application
- `cluster.py`: Dask cluster setup, worker logging
- `config.py`: Configuration properties, worker allocation logic
- `load_images.py`: Lazy image loading with Dask delayed
