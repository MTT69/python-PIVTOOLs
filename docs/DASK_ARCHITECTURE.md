# Dask Pipeline Architecture & The Rechunk Scheduling Problem

## Overview

This document explains the Dask distributed processing architecture used in PyPIVTools, a critical scheduling bug caused by the rechunk step, why `.persist()` masks the problem, and the correct fix.

---

## Current Pipeline Architecture

Both `instantaneous.py` and `ensemble.py` follow the same pipeline:

```
1. Load images       →  lazy dask array, (N, 2, H, W), chunks of (1, 2, H, W)
2. Rechunk           →  regroup to (batch_size, 2, H, W) chunks
3. Filter            →  map_blocks(apply_all_filters_slim) — still lazy
4. Persist           →  materialize filtered chunks on workers
5. Correlate         →  client.submit() per chunk using futures_of()
```

### Step 1: Image Loading (`load_images`)

Images are loaded as **one delayed task per image pair**:

```python
delayed_image_pairs = [
    delayed_image_pair(idx, camera_path, camera, config)
    for idx in range(1, num_pairs + 1)
]
dask_pairs = [to_dask_array(pair, config) for pair in delayed_image_pairs]
pairs_stack = da.stack(dask_pairs, axis=0)
```

This creates a dask array of shape `(N, 2, H, W)` with **N chunks of (1, 2, H, W)** — one chunk per image pair. Each chunk is an independent delayed task that reads from disk. Nothing is loaded into memory yet.

### Step 2: Rechunk (`rechunk_for_batched_processing`)

```python
images = images.rechunk((batch_size, 2, -1, -1))
```

Regroups the array from N chunks of `(1, 2, H, W)` into `ceil(N / batch_size)` chunks of `(batch_size, 2, H, W)`.

Example: 1000 image pairs with batch_size=25 → 40 chunks.

**This is where the scheduling problem originates.** See below.

### Step 3: Filter (`create_filter_pipeline`)

```python
filtered = images.map_blocks(
    apply_all_filters_slim,
    spatial_specs=spatial_specs,
    temporal_specs=temporal_specs,
    pixel_mask=pixel_mask,
    dtype=images.dtype,
    block_id=True,
)
```

Adds a lazy filter task on top of each rechunked chunk. Filters include spatial (gaussian, median, norm, etc.) and temporal (POD, time) operations. Temporal filters require multiple frames in the same chunk — this is **why batching exists**.

### Step 4: Persist

```python
images = images.persist()
```

Submits the entire upstream graph (load + rechunk + filter) to the distributed scheduler and caches results in worker memory. Returns immediately without waiting (no `.wait()`).

### Step 5: Correlate

```python
block_futures = futures_of(images)
for chunk_idx, block_future in enumerate(block_futures):
    future = client.submit(correlate_and_save_batch, block_future, ...)
```

Extracts futures from the persisted array and submits one correlation task per chunk. Each correlation task depends on exactly one persisted chunk.

---

## The Rechunk Scheduling Problem

### What rechunk does to the task graph

Before rechunk, we have N independent tasks:

```
load_pair_1    (independent)
load_pair_2    (independent)
...
load_pair_1000 (independent)
```

After rechunk to batch_size=25, we have 40 tasks with **cross-dependencies**:

```
load_pair_1  ─┐
load_pair_2  ─┤
load_pair_3  ─┤
...           ├──→  rechunk_chunk_0  ──→  filter_chunk_0
load_pair_24 ─┤
load_pair_25 ─┘

load_pair_26 ─┐
load_pair_27 ─┤
...           ├──→  rechunk_chunk_1  ──→  filter_chunk_1
load_pair_50 ─┘
```

Each rechunked output depends on 25 input tasks. The total graph has **1080 tasks** (1000 loads + 40 rechunks + 40 filters) with complex dependency structure.

### Why this kills worker utilisation

The Dask distributed scheduler uses **co-location heuristics**: it tries to schedule tasks near their dependencies to minimise data transfer. When it sees that `rechunk_chunk_0` depends on `load_pair_1` through `load_pair_25`, it tends to schedule all 25 loads on the **same worker**.

The result:

```
Worker A:  load_1, load_2, ... load_25 → rechunk_0 → filter_0 → correlate_0
Worker B:  (idle, waiting for work)
Worker C:  (idle, waiting for work)
Worker D:  (idle, waiting for work)
```

Then once Worker A finishes chunk 0, the scheduler discovers chunk 1 and starts loading those 25 images — possibly on Worker B this time, but the pattern remains **serial per chunk**. Workers take turns instead of working in parallel.

With 4 workers, you'd expect ~4x throughput. Instead you get ~1-1.5x because only 1-2 workers are active at any time. This explains the **3x performance gap** you observed on the Dask dashboard.

### The deeper issue: lazy graph discovery

Without `.persist()`, `futures_of(images)` only works on collections backed by futures. If you instead tried to build the graph manually (e.g., submitting correlation tasks that depend on the lazy filter graph), the scheduler discovers upstream work **lazily per submission**. It doesn't see the full picture, so it can't distribute the 1000 image loads evenly across workers upfront.

---

## Why `.persist()` Fixes It

When you call `images.persist()`:

1. **Full graph submission**: The entire upstream graph (1000 loads + 40 rechunks + 40 filters) is submitted to the scheduler **at once** as a single batch.

2. **Global optimisation**: The scheduler can now see all 1000 image-load tasks simultaneously. It distributes them across all workers roughly evenly (~250 loads per worker with 4 workers).

3. **All workers busy immediately**: Instead of one worker loading 25 images for chunk 0 while others idle, all 4 workers start loading images for different chunks in parallel.

4. **Data locality for correlation**: After persist, each filtered chunk is pinned to the worker that computed it. Correlation tasks submitted with `futures_of()` run on the same worker — zero data transfer.

### Summary table

| Metric | Without persist | With persist |
|---|---|---|
| Graph visibility | Scheduler discovers lazily | Full graph submitted at once |
| Image load distribution | Clustered per chunk (co-location) | Spread across all workers |
| Workers active | 1-2 at a time | All workers from the start |
| Data locality for correlation | Unpredictable | Perfect (pinned to worker) |
| Effective speedup (4 workers) | ~1-1.5x | ~3-4x |

### The cost of persist

Persist keeps **all filtered chunks in distributed RAM simultaneously**:

```
N pairs × 2 frames × H × W × dtype_bytes = total memory

Example: 1000 × 2 × 1024 × 1024 × 4 bytes (float32) = ~8 GB across workers
```

For large datasets this exceeds available memory. We need the scheduling benefit of persist without the memory cost.

---

## The Correct Fix: Eliminate Rechunk

The rechunk step is the root cause. It exists because temporal filters (POD, time) need multiple frames in the same chunk. But the rechunk creates cross-dependencies that break the scheduler's parallelism.

### Solution: load in batch-sized groups from the start

Instead of loading 1 pair per delayed task and then rechunking:

```python
# CURRENT (broken scheduling)
delayed_pairs = [delayed_image_pair(idx, ...) for idx in range(1, N+1)]
dask_pairs = [to_dask_array(pair, config) for pair in delayed_pairs]
images = da.stack(dask_pairs, axis=0)           # (N, 2, H, W), chunks (1,2,H,W)
images = images.rechunk((batch_size, 2, -1, -1)) # rechunk → scheduling problem
```

Load `batch_size` pairs per delayed task:

```python
# FIXED (no rechunk needed)
def delayed_batch_load(start_idx, batch_size, camera_path, camera, config):
    """Load batch_size image pairs in one task. Returns (batch_size, 2, H, W)."""
    pairs = []
    for idx in range(start_idx, start_idx + batch_size):
        pairs.append(read_pair(idx, camera_path, camera, config))
    return np.stack(pairs, axis=0)

num_batches = ceil(N / batch_size)
delayed_batches = [
    dask.delayed(delayed_batch_load)(
        start_idx=i * batch_size + 1,
        batch_size=min(batch_size, N - i * batch_size),
        camera_path=camera_path,
        camera=camera,
        config=config,
    )
    for i in range(num_batches)
]
dask_batches = [
    da.from_delayed(batch, shape=(batch_sz, 2, H, W), dtype=dtype)
    for batch, batch_sz in zip(delayed_batches, batch_sizes)
]
images = da.concatenate(dask_batches, axis=0)   # (N, 2, H, W), chunks (batch_size,2,H,W)

# NO RECHUNK NEEDED — chunks already in the right shape
```

### Resulting task graph

```
batch_load_0  (reads pairs 1-25)     ──→  filter_chunk_0  ──→  correlate_0
batch_load_1  (reads pairs 26-50)    ──→  filter_chunk_1  ──→  correlate_1
...
batch_load_39 (reads pairs 976-1000) ──→  filter_chunk_39 ──→  correlate_39
```

**80 total tasks. Zero cross-dependencies. Every task is independent.**

The scheduler distributes 40 batch-load tasks evenly across workers. All workers busy from the start. No persist needed for scheduling — though you can still persist if you want to decouple filter and correlation phases.

### Memory profile

Identical to the current persist approach, but **controlled**:
- Each worker processes its assigned chunks sequentially
- Peak memory per worker = 1 filtered chunk + correlation overhead
- No risk of OOM from persisting the entire dataset

### What stays the same

Everything downstream is unchanged:
- `create_filter_pipeline` (map_blocks) works identically — same chunk shape
- `futures_of` / `client.submit(correlate)` — same pattern
- `correlate_and_save_batch` / `correlate_and_reduce_on_worker` — same interface
- Ensemble multi-pass accumulation — same worker-side reduction

The only changes are in `load_images()` (batch loading) and removing the `rechunk_for_batched_processing()` call from `instantaneous.py` and `ensemble.py`.

---

## Sliding Window Persist (Alternative Approach)

If for any reason you want to keep the rechunk (e.g., flexibility in chunk boundaries), a **rolling persist window** can also solve the problem:

```python
# Persist only enough chunks to saturate all workers
window_size = num_workers * 2  # e.g., 8 chunks for 4 workers

for start in range(0, num_chunks, window_size):
    end = min(start + window_size, num_chunks)
    # Persist just this window of chunks
    window = images.blocks[start:end]
    window = window.persist()
    window_futures = futures_of(window)

    # Submit correlation tasks for this window
    for chunk_idx, future in enumerate(window_futures):
        client.submit(correlate, future, ...)

    # Previous window's memory is freed as futures complete
```

This gives you the scheduling benefit of persist (scheduler sees a batch of upstream work) while only holding `window_size` chunks in memory at once.

**Trade-off**: More complex code and slightly less pipelining compared to the batch-load approach.

---

## Recommendation

**Eliminate the rechunk.** Load images in batch-sized groups from the start. This is:
- Simpler (fewer moving parts)
- More memory-efficient (no intermediate rechunk layer)
- Naturally parallel (zero cross-dependencies)
- No persist required for scheduling (though still useful for pipelining)

The sliding window approach is a valid fallback if you need dynamic chunk boundaries that don't align with batch loading, but for PIV processing where batch_size is known upfront, batch loading is the cleaner solution.
