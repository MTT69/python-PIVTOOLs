# Unified Worker Accumulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify ensemble PIV into a single worker-accumulation code path that handles both persisted and non-persisted images, eliminating the redundant sliding-window path.

**Architecture:** `correlate_worker_batches` gains one optional parameter (`batch_images`) for pre-filtered image data. When provided (persisted mode), uses those arrays directly. When `None` (non-persisted), loads from disk as before. `process_pass_worker_accumulate` handles both modes: for persisted, it discovers chunk placement via `client.who_has()`, groups by home worker, and passes chunk futures. `process_pass_sliding_window` and `correlate_batch_ensemble` are deleted as dead code.

**Tech Stack:** Dask distributed, numpy, ctypes (C library)

---

## Context

Currently ensemble PIV has two code paths:
- **Non-persisted** (`process_pass_worker_accumulate`): One correlator per worker, accumulates across batches, copies once. Efficient — zero unnecessary transfer.
- **Persisted** (`process_pass_sliding_window`): Creates a NEW correlator per batch, copies ~650MB correlation planes each time, reduces via worker-affinity. Still suffers from the per-batch transfer overhead the worker-accumulation approach was designed to eliminate.

The persisted path gets no benefit from worker accumulation. Both paths should use the same efficient accumulation pattern.

---

### Task 1: Add `batch_images` parameter to `correlate_worker_batches`

**Files:**
- Modify: `pivtools_cli/processing/dask_pipeline.py:604-690`

**Step 1: Add the parameter and update docstring**

Change the function signature and docstring. Add `batch_images: Optional[List[np.ndarray]] = None` after `output_path`. When `batch_images` is provided, skip pipeline reconstruction and use the provided arrays. The `batch_indices` still control `is_first` logic (metadata/diagnostics). Change from:

```python
def correlate_worker_batches(
    batch_indices: list,
    config: Config,
    pass_idx: int,
    predictor_field: Optional[np.ndarray],
    cache: dict,
    masks: Optional[List[np.ndarray]],
    camera_num: int,
    source_path: str,
    pixel_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Accumulate correlation across multiple batches on one worker.

    Reconstructs the lazy image+filter pipeline locally, then processes
    each assigned batch sequentially. The EnsembleCorrelatorCPU's internal
    buffers accumulate across batches (C library's native += behavior).
    Correlation planes are copied out ONCE at the end.
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU
    from pivtools_core.image_handling.load_images import load_images

    # Reconstruct lazy image pipeline locally
    images = load_images(
        camera_num, config, source=Path(source_path),
        batch_size=config.batch_size,
    )
    images = create_filter_pipeline(images, config, pixel_mask)
```

to:

```python
def correlate_worker_batches(
    batch_indices: list,
    config: Config,
    pass_idx: int,
    predictor_field: Optional[np.ndarray],
    cache: dict,
    masks: Optional[List[np.ndarray]],
    camera_num: int = 0,
    source_path: str = "",
    pixel_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    batch_images: Optional[List[np.ndarray]] = None,
) -> dict:
    """Accumulate correlation across multiple batches on one worker.

    Two modes:
    - batch_images=None: Reconstructs lazy image+filter pipeline locally,
      loads each batch from disk sequentially.
    - batch_images provided: Uses pre-filtered image arrays directly
      (Dask auto-resolves futures before function entry).

    In both modes, the EnsembleCorrelatorCPU's internal buffers accumulate
    across batches (C library's native += behavior). Correlation planes
    are copied out ONCE at the end.
    """
    from pivtools_cli.piv.piv_backend.cpu_ensemble import EnsembleCorrelatorCPU

    # Reconstruct lazy image pipeline only when loading from disk
    images = None
    if batch_images is None:
        from pivtools_core.image_handling.load_images import load_images
        images = load_images(
            camera_num, config, source=Path(source_path),
            batch_size=config.batch_size,
        )
        images = create_filter_pipeline(images, config, pixel_mask)
```

**Step 2: Update the batch loop to use `batch_images` when available**

Change the batch loading line inside the loop from:

```python
    for i, batch_idx in enumerate(batch_indices):
        # Load + filter this batch locally (synchronous = current thread)
        batch_data = images.blocks[batch_idx].compute(scheduler='synchronous')
```

to:

```python
    for i, batch_idx in enumerate(batch_indices):
        # Get batch data: from persisted images or lazy pipeline
        if batch_images is not None:
            batch_data = batch_images[i]
        else:
            batch_data = images.blocks[batch_idx].compute(scheduler='synchronous')
```

**Step 3: Run existing tests to verify no regression**

Run: `cd /Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools && python -m pytest unit-tests/test_multipass_convergence.py -v -x --timeout=120`
Expected: All tests PASS (default `batch_images=None` preserves existing behavior)

**Step 4: Commit**

```bash
git add pivtools_cli/processing/dask_pipeline.py
git commit -m "feat(ensemble): add batch_images param to correlate_worker_batches for persisted mode"
```

---

### Task 2: Update `process_pass_worker_accumulate` to handle persisted images

**Files:**
- Modify: `pivtools_core/ensemble.py:204-361`

**Step 1: Add `images` parameter to function signature**

Change from:

```python
def process_pass_worker_accumulate(
    client,
    num_chunks,
    workers,
    scattered_config,
    pass_idx,
    scattered_predictor,
    scattered,
    config,
    output_path,
    camera_num,
    source_path,
    pixel_mask,
):
```

to:

```python
def process_pass_worker_accumulate(
    client,
    num_chunks,
    workers,
    scattered_config,
    pass_idx,
    scattered_predictor,
    scattered,
    config,
    output_path,
    camera_num,
    source_path,
    pixel_mask,
    images=None,
):
```

**Step 2: Update docstring**

Replace the existing docstring (lines ~218-232) with:

```python
    """
    Process one pass using per-worker accumulation.

    Each worker gets ONE long-running task that:
    1. Creates ONE EnsembleCorrelatorCPU
    2. Loops over its assigned batches, accumulating into internal buffers
    3. Returns the accumulated result ONCE

    Two modes:
    - images=None (non-persisted): Assigns chunks round-robin. Workers
      reconstruct the lazy image+filter pipeline locally and load from disk.
    - images provided (persisted dask array): Discovers where Dask placed
      each chunk via who_has(), groups by home worker, passes chunk futures
      as batch_images (Dask resolves them locally — zero transfer).

    Then tree-reduces K per-worker results into one final result.
    """
```

**Step 3: Add persisted chunk discovery after the imports**

After `from dask.distributed import as_completed` (line 233), add the persisted chunk grouping logic. Replace the existing chunk assignment block (lines 235-250) with:

```python
    num_workers = len(workers)
    diag_path = str(output_path) if config.ensemble_save_diagnostics else None

    if images is not None:
        # Persisted mode: discover where Dask placed each chunk, group by home worker
        chunk_futures = client.compute(
            [images.blocks[i] for i in range(num_chunks)]
        )
        # Discover placement — each future is on exactly one worker
        worker_chunks = {}  # worker_address → [(chunk_idx, future), ...]
        for chunk_idx, fut in enumerate(chunk_futures):
            who = client.who_has(fut)
            home_worker = list(who[fut.key])[0]
            if home_worker not in worker_chunks:
                worker_chunks[home_worker] = []
            worker_chunks[home_worker].append((chunk_idx, fut))
    else:
        # Non-persisted mode: assign chunks round-robin across workers
        worker_chunks_rr = {w: [] for w in workers}
        for chunk_idx in range(num_chunks):
            w = workers[chunk_idx % num_workers]
            worker_chunks_rr[w].append(chunk_idx)
        # Remove workers with no chunks, convert to same format
        worker_chunks = {
            w: [(idx, None) for idx in indices]
            for w, indices in worker_chunks_rr.items()
            if indices
        }

    logger.info(
        f"  Worker accumulation: {len(worker_chunks)} workers, "
        f"{num_chunks} chunks ({[len(c) for c in worker_chunks.values()]} per worker)"
        f"{' (persisted)' if images is not None else ' (from disk)'}"
    )
```

**Step 4: Update the task submission loop**

Replace the existing submission loop (lines 252-270) with:

```python
    # Submit one task per worker
    worker_futures = []
    for worker, chunk_info in worker_chunks.items():
        chunk_indices = [idx for idx, _ in chunk_info]
        chunk_futs = [fut for _, fut in chunk_info]
        # batch_images is list of futures (persisted) or None (from disk)
        batch_images = chunk_futs if all(f is not None for f in chunk_futs) else None

        fut = client.submit(
            correlate_worker_batches,
            batch_indices=chunk_indices,
            config=scattered_config,
            pass_idx=pass_idx,
            predictor_field=scattered_predictor,
            cache=scattered['cache'],
            masks=scattered['masks'],
            camera_num=camera_num,
            source_path=str(source_path),
            pixel_mask=pixel_mask,
            output_path=diag_path if 0 in chunk_indices else None,
            batch_images=batch_images,
            workers=[worker],
            pure=False,
        )
        worker_futures.append(fut)
```

The rest of the function (waiting, locality verification, tree reduction, transfer logging) stays exactly the same.

**Step 5: Run existing tests**

Run: `cd /Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools && python -m pytest unit-tests/test_multipass_convergence.py -v -x --timeout=120`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add pivtools_core/ensemble.py
git commit -m "feat(ensemble): process_pass_worker_accumulate handles persisted images via who_has grouping"
```

---

### Task 3: Unify `run_ensemble_piv` — remove branching, always use worker accumulation

**Files:**
- Modify: `pivtools_core/ensemble.py:423-622`

**Step 1: Remove the strategy branching**

Replace the strategy selection and pass processing block (lines 564-622):

```python
    images_are_persisted = (num_chunks == 1) or config.ensemble_persist_images

    logger.info(f"Processing passes {start_pass_idx + 1} to {num_passes} with {num_chunks} chunks each...")
    logger.info(f"  Strategy: {'sliding window (persisted)' if images_are_persisted else 'worker accumulation'}")

    for pass_idx in range(start_pass_idx, num_passes):
        ...
        # Process pass
        if images_are_persisted:
            accumulated = process_pass_sliding_window(
                client=client,
                images=images,
                num_chunks=num_chunks,
                workers=workers,
                scattered_config=scattered_config,
                pass_idx=pass_idx,
                scattered_predictor=scattered_predictor,
                scattered=scattered,
                config=config,
                output_path=output_path,
            )
        else:
            accumulated = process_pass_worker_accumulate(
                client=client,
                num_chunks=num_chunks,
                workers=workers,
                scattered_config=scattered_config,
                pass_idx=pass_idx,
                scattered_predictor=scattered_predictor,
                scattered=scattered,
                config=config,
                output_path=output_path,
                camera_num=camera_num,
                source_path=source_path,
                pixel_mask=scattered_pixel_mask,
            )
```

with:

```python
    images_are_persisted = (num_chunks == 1) or config.ensemble_persist_images

    logger.info(f"Processing passes {start_pass_idx + 1} to {num_passes} with {num_chunks} chunks each...")
    logger.info(f"  Strategy: worker accumulation{' (persisted)' if images_are_persisted else ' (from disk)'}")

    for pass_idx in range(start_pass_idx, num_passes):
        ...
        # Process pass — single path for both persisted and non-persisted
        accumulated = process_pass_worker_accumulate(
            client=client,
            num_chunks=num_chunks,
            workers=workers,
            scattered_config=scattered_config,
            pass_idx=pass_idx,
            scattered_predictor=scattered_predictor,
            scattered=scattered,
            config=config,
            output_path=output_path,
            camera_num=camera_num,
            source_path=source_path,
            pixel_mask=scattered_pixel_mask,
            images=images if images_are_persisted else None,
        )
```

Note: keep everything else in the for loop body unchanged (the `if _shutdown_requested` check, predictor scatter, workers list, timing, finalize, cleanup).

**Step 2: Run existing tests**

Run: `cd /Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools && python -m pytest unit-tests/test_multipass_convergence.py -v -x --timeout=120`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add pivtools_core/ensemble.py
git commit -m "feat(ensemble): unify run_ensemble_piv — always use worker accumulation"
```

---

### Task 4: Delete dead code — `process_pass_sliding_window` and `correlate_batch_ensemble`

**Files:**
- Modify: `pivtools_core/ensemble.py:50-58` (imports), `64-201` (function)
- Modify: `pivtools_cli/processing/dask_pipeline.py:569-601` (function)
- Modify: `pivtools_cli/processing/__init__.py` (exports)

**Step 1: Remove `process_pass_sliding_window` from `ensemble.py`**

Delete the entire function (lines 64-201) and remove `correlate_batch_ensemble` from the imports at line 56. Also remove `reduce_ensemble_results_inplace` from the import if it's no longer used (check: it was only used in `process_pass_sliding_window`'s Phase 2).

The imports block (lines 50-58) changes from:

```python
from pivtools_cli.processing.dask_pipeline import (
    create_filter_pipeline,
    scatter_immutable_data,
    reduce_ensemble_results,
    reduce_ensemble_results_inplace,
    extract_predictor_field,
    correlate_batch_ensemble,
    correlate_worker_batches,
)
```

to:

```python
from pivtools_cli.processing.dask_pipeline import (
    create_filter_pipeline,
    scatter_immutable_data,
    reduce_ensemble_results,
    extract_predictor_field,
    correlate_worker_batches,
)
```

**Step 2: Remove `correlate_batch_ensemble` from `dask_pipeline.py`**

Delete the function at lines 569-601.

**Step 3: Check if `reduce_ensemble_results_inplace` has any remaining callers**

Run: `grep -rn "reduce_ensemble_results_inplace" pivtools_cli/ pivtools_core/ --include="*.py"`

If it is ONLY referenced in `dask_pipeline.py` (definition) and `__init__.py` (export), and no longer imported by `ensemble.py`, it is dead code. Delete it from `dask_pipeline.py` (lines 693-719+).

**Step 4: Update `pivtools_cli/processing/__init__.py`**

Remove `correlate_batch_ensemble` (and `reduce_ensemble_results_inplace` if dead) from imports and `__all__`:

```python
from .dask_pipeline import (
    create_filter_pipeline,
    scatter_immutable_data,
    correlate_and_save_batch,
    reduce_ensemble_results,
    extract_predictor_field,
    correlate_worker_batches,
)

__all__ = [
    "create_filter_pipeline",
    "scatter_immutable_data",
    "correlate_and_save_batch",
    "reduce_ensemble_results",
    "extract_predictor_field",
    "correlate_worker_batches",
]
```

**Step 5: Run full test suite**

Run: `cd /Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools && python -m pytest unit-tests/ -v --timeout=120`
Expected: All tests PASS, no import errors

**Step 6: Commit**

```bash
git add pivtools_cli/processing/dask_pipeline.py pivtools_cli/processing/__init__.py pivtools_core/ensemble.py
git commit -m "refactor(ensemble): remove dead code — process_pass_sliding_window, correlate_batch_ensemble"
```

---

### Task 5: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the `ensemble.py` function table (around line 835)**

Remove the `process_pass_sliding_window` row. Update the description of `process_pass_worker_accumulate` to mention it handles both modes.

Replace:

```
| `process_pass_sliding_window` | `client, images, num_chunks, workers, scattered_config, pass_idx, scattered_predictor, scattered, config, output_path` | `dict` | Dynamic worker-affinity sliding window: `as_completed()` loop discovers where each batch landed via `client.who_has()`, pins reduction there (zero cross-worker transfer). Tree reduction merges K per-worker accumulators in O(log₂ K) rounds. |
```

with nothing (delete the row). Then add/update `process_pass_worker_accumulate` if not already present.

**Step 2: Update the `dask_pipeline.py` function table (around line 948)**

Remove the `correlate_batch_ensemble` row. Update `correlate_worker_batches` description:

```
| `correlate_worker_batches(batch_indices, config, ..., batch_images=None)` | [Ensemble] Per-worker accumulation. When `batch_images=None`, reconstructs image pipeline locally and loads from disk. When provided (persisted mode), uses pre-filtered arrays directly. Creates one correlator, processes all assigned batches, returns once. |
```

**Step 3: Update the "Worker accumulation (ensemble)" section (around line 967)**

Replace the two-mode description with:

```
**Worker accumulation (ensemble):** Single unified code path via `process_pass_worker_accumulate`.
One `correlate_worker_batches` task per worker. Each worker creates ONE `EnsembleCorrelatorCPU`,
processes all assigned batches sequentially — clearing buffers only on the first batch, letting
the C library's native `+=` accumulate across batches, copying correlation planes out ONCE.
After all workers complete, K per-worker results merge via tree reduction (`O(log₂ K)` rounds
of `reduce_ensemble_results` with safe `+` operator).

*Non-persisted (default):* Chunks assigned round-robin. Workers reconstruct the lazy image+filter
pipeline locally and load from disk. OS page cache makes re-reads on subsequent passes nearly free.

*Persisted (`persist_images: true`):* `who_has()` discovers where Dask placed each chunk during
`images.persist()`. Chunks grouped by home worker and passed as futures. Dask resolves futures
locally — zero cross-worker transfer. Same accumulation loop, same result format.
```

**Step 4: Update the ensemble step description (around line 798)**

Change from:
```
**Ensemble step (e):** `process_pass_sliding_window()` → accumulate correlation sums per worker, reduce, ...
```
to:
```
**Ensemble step (e):** `process_pass_worker_accumulate()` → per-worker accumulation of correlation sums, tree reduction, then `accumulator.finalize_pass()` → distributed Gaussian (or k-space) fitting → extract velocities + stresses.
```

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for unified worker accumulation"
```

---

## Verification

1. **Unit tests**: `python -m pytest unit-tests/test_multipass_convergence.py -v -x -s` — validates velocity accuracy against analytical Poiseuille flow
2. **Full suite**: `python -m pytest unit-tests/ -v --timeout=120` — no regressions
3. **Visual check**: Run with `--make-figures` and inspect diagnostic outputs
4. **Dask dashboard** (manual, real dataset): Verify one `correlate_worker_batches` task per worker per pass, regardless of persist setting

## Files Summary

| File | Change |
|------|--------|
| `pivtools_cli/processing/dask_pipeline.py` | Add `batch_images` param to `correlate_worker_batches`, delete `correlate_batch_ensemble` + `reduce_ensemble_results_inplace` |
| `pivtools_cli/processing/__init__.py` | Remove deleted function exports |
| `pivtools_core/ensemble.py` | Update `process_pass_worker_accumulate` for persisted mode, unify `run_ensemble_piv`, delete `process_pass_sliding_window` |
| `CLAUDE.md` | Update Dask Patterns, function tables, ensemble step description |
